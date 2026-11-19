"""Final-phase k-NN evaluator (mirror/final/eval/knn_fast.py) and flip averaging (flipavg.py): CPU tests with the tiny
random ViT and synthetic images. Nothing here downloads data or weights."""
import json

import numpy as np
import pytest
import torch

from mirror.data import make_loader, make_splits, mirror, synthetic_dataset
from mirror.final.eval import flipavg as FA
from mirror.final.eval import knn_fast as KF
from mirror.knn import evaluate_knn


@pytest.fixture(scope="module")
def setup():
    images, labels = synthetic_dataset(4, 16, 56, seed=0)
    sp = make_splits(labels, {"bank": 10, "val": 6})
    data = KF.EvalData.synthetic(num_classes=4, bank_per_class=10, val_per_class=6, img_size=56, seed=0)
    ad, model, _ = KF.tiny_loader("run", 3)
    return {"images": images, "labels": labels, "sp": sp, "data": data, "ad": ad, "model": model}


def _loaders(s, bs=8):
    return (make_loader(s["images"], s["labels"], s["sp"]["bank"], bs),
            make_loader(s["images"], s["labels"], s["sp"]["val"], bs))


def test_device_batches_equal_the_dataloader_batches(setup):
    s = setup
    bl, ql = _loaders(s, 5)
    for mine, (ref, y) in zip(s["data"].bank.batches(None, 5), bl):
        assert torch.equal(mine, ref) and mine.is_contiguous()
    assert torch.equal(s["data"].val.labels, torch.as_tensor(s["labels"][s["sp"]["val"]]))
    idx = np.array([5, 0, 7])
    got = next(s["data"].val.batches(idx, 8))
    assert torch.equal(got, torch.stack([s["data"].val.images[i] for i in idx]))


@pytest.mark.parametrize("pass_name,amp", [("fp32", None), ("bf16", "bf16")])
def test_predictions_equal_mirror_knn_evaluate_knn(setup, pass_name, amp):
    """Same batch size: the same features, so the same predictions as mirror.knn.evaluate_knn (on the CPU the bf16
    pass has no autocast, as mirror.knn)."""
    s = setup
    bl, ql = _loaders(s)
    ref = evaluate_knn(s["ad"], bl, ql, "cpu", 4, model=s["model"], amp=amp)
    r = KF.run_pass(s["ad"], s["model"], s["data"], pass_name, batch_size=8)
    assert torch.equal(ref["pred"], r["pred"]) and torch.equal(ref["pred_flip"], r["pred_flip"])
    assert r["summary"]["top1"] == ref["top1"] and r["summary"]["flip_consistency"] == ref["flip_consistency"]
    assert r["summary"]["n_bank"] == 40 and r["summary"]["n_query"] == 24 and r["summary"]["amp"] == (amp or "fp32")
    nd = int((r["pred"] != r["pred_flip"]).sum())
    assert r["summary"]["n_disagree"] == nd and len(r["disagreements"]) == nd
    if nd:
        d0 = r["disagreements"][0]
        assert {"vote_margin_rel", "vote_margin_rel_mirrored", "kth_sim_gap", "kth_sim_gap_mirrored"} <= set(d0)
        assert r["summary"]["max_vote_margin_rel"] >= r["summary"]["max_vote_margin_rel_one_copy"]


def test_flip_averaged_pass_equals_the_flip_averaged_adapter(setup):
    s = setup
    bl, ql = _loaders(s)
    ref = evaluate_knn(FA.FlipAveragedAdapter(s["ad"]), bl, ql, "cpu", 4, model=s["model"])
    r = KF.run_pass(s["ad"], s["model"], s["data"], "fp32_flipavg", batch_size=8)
    assert torch.equal(ref["pred"], r["pred"]) and torch.equal(ref["pred_flip"], r["pred_flip"])
    assert r["summary"]["flip_consistency"] == 100.0 and r["summary"]["flip_consistency_by_construction"]
    assert r["summary"]["forward_passes_per_image"] == 2
    x = torch.randn(3, 3, 56, 56, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        assert torch.equal(FA.cls_feature_flipavg(s["ad"], x, s["model"]),
                           FA.cls_feature_flipavg(s["ad"], mirror(x), s["model"]))


def test_query_and_bank_subsets_follow_the_must09_rule(setup):
    data = setup["data"]
    vl = data.val.labels.numpy()
    assert data.query_idx(None) is None and data.query_idx(10 ** 6) is None
    q = data.query_idx(8)                   # Protocol.query_loader(n): make_splits(labels, {"q": n // classes})
    assert np.array_equal(q, make_splits(vl, {"q": 2})["q"]) and len(q) == 8
    assert all((vl[q] == c).sum() == 2 for c in range(4))
    with pytest.raises(ValueError):
        data.query_idx(3)
    b = data.bank_idx(3)
    bl = data.bank.labels.numpy()
    assert len(b) == 12 and np.all(np.diff(b) > 0)
    for c in range(4):
        assert np.array_equal(b[bl[b] == c], np.flatnonzero(bl == c)[:3])
    r = KF.run_pass(setup["ad"], setup["model"], data, "fp32", n_query=8, bank_per_class=3, batch_size=8)
    assert r["summary"]["n_query"] == 8 and r["summary"]["n_bank"] == 12 and np.array_equal(r["query_rows"], q)
    assert KF.result_key("fp32", 8, 24, 3) == "fp32_q8_b3" and KF.result_key("bf16", 24, 24) == "bf16"


def test_vote_margin_and_near_tie_record():
    votes = torch.tensor([[3.0, 1.0, 0.0], [2.0, 2.0, 1.0]])
    assert torch.allclose(KF.vote_margin_rel(votes), torch.tensor([0.5, 0.0]))
    pred, pflip = torch.tensor([0, 1, 2, 0]), torch.tensor([0, 2, 2, 1])
    labels = torch.tensor([0, 1, 2, 1])
    m_x, m_m = torch.tensor([0.1, 1e-6, 0.3, 2e-4]), torch.tensor([0.2, 5e-5, 0.4, 1e-6])
    s = KF.flip_summary(pred, pflip, labels, m_x, m_m, torch.tensor([0.0, 1e-7, 0.0, 0.1]))
    assert s["flip_consistency"] == 50.0 and s["n_disagree"] == 2
    assert s["max_vote_margin_rel"] == pytest.approx(2e-4)            # both copies must be near-ties
    assert s["max_vote_margin_rel_one_copy"] == pytest.approx(1e-6)   # one copy (CORE-06 convention)
    assert s["n_disagree_not_near_tie"] == 1 and s["feature_identical_pct"] == 50.0


def test_exactly_invariant_mirror_model_in_fp32():
    """A Mirror-exact model with the even head (mirror.m1.build_vexact): fp32 flip consistency 100%, or only
    near-ties."""
    from mirror.adapters import TimmTinyAdapter
    from mirror.attention import attn_dims
    from mirror.m1 import build_vexact, random_head_structure, random_involution_fixing_ones
    from mirror.matching import random_pairing

    ad = TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float32)
    d, h, L = ad.embed_dim, ad.mlp_hidden, ad.depth
    H, dh = attn_dims(ad.model.blocks[0])
    A = random_involution_fixing_ones(d, d // 2 + 4, seed=3)
    perms = [random_pairing(h, h // 3, seed=10 + b) for b in range(L)]
    maps = [random_head_structure(H, dh, 1 + b % 2, seed=20 + b) for b in range(L)]
    m, _ = build_vexact(ad, A, list(range(L)), perms, maps, even_head=True)
    data = KF.EvalData.synthetic(num_classes=4, bank_per_class=10, val_per_class=6, img_size=56, seed=1)
    r = KF.run_pass(ad, m, data, "fp32", batch_size=8)
    s = r["summary"]
    assert s["flip_consistency"] == 100.0 or s["n_disagree_not_near_tie"] == 0
    assert s["feature_flip_max_abs"] < 1e-5


def test_save_predictions_is_the_must09_format(tmp_path):
    p = KF.save_predictions(tmp_path / "knn" / "step7_bf16_q8.npz", torch.tensor([1, 2]), torch.tensor([1, 3]),
                            torch.tensor([1, 2]), np.array([4, 9]))
    with np.load(p) as z:
        assert {k: z[k].dtype for k in z.files} == {"pred": np.int16, "pred_flip": np.int16, "labels": np.int16,
                                                     "query_rows": np.int64}
        assert z["pred_flip"].tolist() == [1, 3]


def test_evaluate_request_outputs_resume_and_original_cache(setup, tmp_path):
    data = setup["data"]
    ev = KF.KnnEvaluator(data, "cpu", 8, cache_dir=tmp_path / "cache")
    run = tmp_path / "runs" / "tiny" / "mirror_exact" / "s0"
    calls = []

    def loader(run_dir, step, device):
        calls.append(step)
        return KF.tiny_loader(run_dir, step, device)

    def orig(name, device):
        return KF.tiny_original(name, device)

    req = {"run_dir": str(run), "step": 3, "passes": ["bf16", "fp32"], "n_query": None, "priority": 2,
           "fp32_n_query": 8}
    rec = KF.evaluate_request(ev, req, loader, orig)
    assert calls == [3] and set(rec["passes"]) == {"bf16", "fp32_q8"}
    j = json.loads((run / "eval" / "step3.json").read_text(encoding="utf-8"))
    assert j["passes"]["bf16"]["n_query"] == 24 and j["passes"]["fp32_q8"]["n_query"] == 8
    assert j["passes"]["bf16"]["predictions"] == "knn/step3_bf16.npz" and (run / "knn" / "step3_fp32_q8.npz").is_file()
    assert set(j["original_reference"]) == {"bf16"} and 0 <= j["original_reference"]["bf16"]["top1"] <= 100
    assert (run / "eval" / "step3_meta.json").is_file()
    meta = json.loads((run / "eval" / "step3_meta.json").read_text(encoding="utf-8"))
    assert meta["experiment"] == "fp_eval" and "git" in meta and meta["config"]["k"] == 20
    assert list((tmp_path / "cache" / "knn_original").glob("*/bf16.json"))       # the original's result on disk
    KF.evaluate_request(ev, req, loader, orig)                                    # finished: nothing loaded again
    assert calls == [3]
    (run / "ckpt").mkdir()
    (run / "ckpt" / "step3.pt").write_bytes(b"new weights")                       # another checkpoint: redone
    rec2 = KF.evaluate_request(ev, req, loader, orig)
    assert calls == [3, 3] and rec2["passes"]["bf16"]["settings"]["checkpoint_sha256"] == KF.sha256_file(
        run / "ckpt" / "step3.pt")


def test_trainer_request_fields_and_adapter_reuse(setup, tmp_path):
    """The trainer's EvalQueue keys (ckpt, tags, created_utc) are accepted; the original adapter that load_student
    returns is passed back to it (adapter=) for the next request of the same run spec."""
    ev = KF.KnnEvaluator(setup["data"], "cpu", 8)
    run = tmp_path / "runs" / "final" / "A05" / "tiny" / "svdllm" / "s0"
    (run / "ckpt").mkdir(parents=True)
    (run / "spec.json").write_text(json.dumps({"adapter": {"name": "tiny", "img_size": 56}, "model": "tiny"}),
                                   encoding="utf-8")
    (run / "ckpt" / "step5.pt").write_bytes(b"five")
    got, adapters = [], {}

    def load_student(run_dir, step, device="cpu", adapter=None):
        got.append(adapter)
        ad, m, info = KF.tiny_loader(run_dir, step, device)
        return (adapter or ad), m, {"method": "svdllm"}

    base = {"run_dir": str(run), "passes": ["bf16"], "n_query": 8, "with_original": False, "priority": 0,
            "tags": {"experiment": "A05", "student": "svdllm"}, "created_utc": "2026-11-19T00:00:00Z"}
    rec = KF.evaluate_request(ev, {**base, "step": 5, "ckpt": str(run / "ckpt" / "step5.pt")}, load_student,
                              adapters=adapters)
    KF.evaluate_request(ev, {**base, "step": 6}, load_student, adapters=adapters)
    assert got[0] is None and got[1] is not None and "unknown_fields" not in rec["request"]
    assert rec["checkpoint_sha256"] == KF.sha256_file(run / "ckpt" / "step5.pt") and rec["model"] == "tiny"
    meta = json.loads((run / "eval" / "step5_meta.json").read_text(encoding="utf-8"))
    assert meta["experiment"] == "A05"


def test_original_request_with_flip_average_and_cost(setup, tmp_path):
    ev = KF.KnnEvaluator(setup["data"], "cpu", 8)
    out = tmp_path / "A07" / "original"
    rec = KF.evaluate_request(ev, {"original": True, "model": "tiny", "out_dir": str(out), "passes": "bf16",
                                   "flip_average": True},
                              build_original=lambda n, d: KF.tiny_original(n, d))
    p = rec["passes"]
    assert set(p) == {"bf16", "bf16_flipavg"} and rec["tag"] == "original" and "original_reference" not in rec
    assert p["bf16_flipavg"]["flip_consistency"] == 100.0
    assert p["bf16_flipavg"]["gflops_per_image"] == pytest.approx(2 * p["bf16"]["gflops_per_image"])
    assert rec["cost"]["source"].startswith("dense") and (out / "knn" / "original_bf16_flipavg.npz").is_file()


def test_bad_requests_are_refused():
    for bad in ({"step": 1}, {"run_dir": "x", "passes": ["fp16"]}, {"original": True, "model": "m"},
                {"run_dir": "x", "passes": []}, [1, 2]):
        with pytest.raises(ValueError):
            KF.normalise_request(bad)
    r = KF.normalise_request({"run_dir": "x", "passes": "bf16, fp32", "step": "780", "with_original": True})
    assert r["passes"] == ["bf16", "fp32"] and r["step"] == 780 and r["with_original"] == ["bf16", "fp32"]
