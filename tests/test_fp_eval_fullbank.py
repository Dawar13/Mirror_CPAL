"""Full-train features and the full-bank k-NN (mirror/final/eval/fullbank.py, scripts/fp_fullbank.py): CPU tests with
tiny random ViTs and a synthetic image stream. Nothing here downloads data or weights."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mirror.data import mirror, normalize_for, synthetic_dataset
from mirror.final.eval import fullbank as FB
from mirror.final.eval import knn_fast as KF
from mirror.final.eval.linear_protocol import feature_types, linear_feature

ROOT = Path(__file__).resolve().parents[1]
N, C, PX = 40, 4, 56


def load_script(name):
    key = f"{name}_fp_eval_fullbank_test"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


def _stream(n=N):
    images, labels = synthetic_dataset(C, n // C, PX, seed=7)

    def stream_train(start=0, batch_size=8):
        for s in range(start, n, batch_size):
            yield images[s:s + batch_size], labels[s:s + batch_size], np.arange(s, min(s + batch_size, n)) + 500

    return stream_train, images, labels


def test_chunked_topk_merge_equals_the_monolithic_knn():
    g = torch.Generator().manual_seed(0)
    bank = torch.randn(1000, 16, generator=g)
    labels = torch.randint(0, 5, (1000,), generator=g)
    q = F.normalize(torch.randn(70, 16, generator=g), dim=-1)
    top = FB.topk_chunked(bank.numpy(), {"q": q}, k=20, device="cpu", bank_chunk=97, query_chunk=13)
    ts, ti = top["q"]
    ref = (q @ F.normalize(bank, dim=-1).T).topk(20, dim=1)
    assert torch.equal(ti, ref.indices) and torch.allclose(ts, ref.values, atol=1e-6, rtol=0)
    pred, margin = FB.vote(ts, ti, labels, 5)
    ref_pred, ref_margin = KF.knn_vote(F.normalize(bank, dim=-1), labels, q, 5)
    assert torch.equal(pred, ref_pred) and torch.allclose(margin, ref_margin, atol=1e-6)
    small = FB.topk_chunked(bank[:7].numpy(), {"q": q}, k=20, device="cpu", bank_chunk=3)   # bank smaller than k
    assert small["q"][0].shape == (70, 7) and torch.isfinite(small["q"][0]).all()


def test_linear_feature_is_the_dinov2_layout():
    ad, model, _ = KF.tiny_loader("run", 1)
    x = torch.randn(3, 3, PX, PX, generator=torch.Generator().manual_seed(1))
    d, p = ad.embed_dim, ad.n_prefix_tokens
    with torch.no_grad():
        f = linear_feature(ad, x, model)
        t = ad.layer0_tokens(x, model)
        outs = []
        for blk in model.blocks:
            t = blk(t)
            outs.append(model.norm(t))
    L = len(model.blocks)
    assert f.shape == (3, 5 * d) and L == 4
    for j, b in enumerate(range(L - 4, L)):                              # class tokens of the last 4 blocks, in order
        assert torch.allclose(f[:, j * d:(j + 1) * d], outs[b][:, 0], atol=1e-6)
    assert torch.allclose(f[:, 4 * d:], outs[-1][:, p:].mean(1), atol=1e-6)    # mean patch token of the last block
    with torch.no_grad():
        assert torch.equal(f[:, 3 * d:4 * d], ad.cls_feature(x, model))      # the k-NN feature
    assert feature_types(d) == [(1, False, 3 * d, 4 * d), (1, True, 3 * d, 5 * d), (4, False, 0, 4 * d),
                                (4, True, 0, 5 * d)]


def test_stream_features_for_several_models_resumes_and_matches_direct_features(tmp_path):
    stream_train, images, labels = _stream()
    ad, student, _ = KF.tiny_loader("run", 2)
    models = [FB.FeatureModel("orig", ad), FB.FeatureModel("stud", ad, student, kinds=("knn",)),
              FB.FeatureModel("fa", ad, kinds=("knn", "linear"), flipavg=True)]
    factory = FB.make_stream_factory(stream_train, 8)

    def crashing(start):
        for i, batch in enumerate(factory(start)):
            if i == 2:
                raise RuntimeError("preempted")
            yield batch

    out = tmp_path / "fb"
    with pytest.raises(RuntimeError, match="preempted"):
        FB.stream_features(models, crashing, N, out, "cpu", micro_batch=8, flush_every=1)
    prog = json.loads((out / "progress.json").read_text(encoding="utf-8"))
    assert prog["train"] == {"orig": 16, "stud": 16, "fa": 16, "_labels_ids": 16}
    seen = []

    def counting(start):
        seen.append(start)
        return factory(start)

    FB.stream_features(models, counting, N, out, "cpu", micro_batch=8, flush_every=1)
    assert seen == [16]
    lab = np.load(out / "train_labels.npy")
    assert np.array_equal(lab, labels) and np.array_equal(np.load(out / "train_ids.npy"), np.arange(N) + 500)
    x8 = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    with torch.no_grad():
        direct_lin = torch.cat([linear_feature(ad, normalize_for(ad, x8[s:s + 8])) for s in range(0, N, 8)])
        direct_cls = torch.cat([ad.cls_feature(normalize_for(ad, x8[s:s + 8]), student) for s in range(0, N, 8)])
    assert np.array_equal(np.load(out / "orig" / "train_linear.npy"), direct_lin.half().numpy())
    assert np.array_equal(np.load(out / "orig" / "train_knn.npy"), direct_lin[:, 192:256].half().numpy())
    assert np.array_equal(np.load(out / "stud" / "train_knn.npy"), direct_cls.half().numpy())
    assert not (out / "stud" / "train_linear.npy").exists()
    xf = normalize_for(ad, x8[:8])
    with torch.no_grad():
        fa = 0.5 * (linear_feature(ad, xf) + linear_feature(ad, mirror(xf)))
    assert np.array_equal(np.load(out / "fa" / "train_linear.npy")[:8], fa.half().numpy())
    FB.stream_features(models, lambda s: (_ for _ in ()).throw(AssertionError("no stream needed")), N, out, "cpu")
    # validation features, then the full-bank k-NN: equal to the k-NN of knn_fast on the same stored features
    val = KF.EvalData.synthetic(num_classes=C, bank_per_class=1, val_per_class=5, img_size=PX, seed=9).val
    FB.val_features(models, val, out, "cpu", batch_size=8)
    for fm in models:
        rec = FB.fullbank_knn(out, fm.name, C, "cpu", bank_chunk=11, query_chunk=7)
        bank = F.normalize(torch.from_numpy(np.load(out / fm.name / "train_knn.npy")).float(), dim=-1)
        qx = F.normalize(torch.from_numpy(np.load(out / fm.name / "val_knn.npy")).float(), dim=-1)
        ref_pred, _ = KF.knn_vote(bank, torch.from_numpy(lab), qx, C)
        with np.load(out / fm.name / "fullbank_knn_pred.npz") as z:
            assert np.array_equal(z["pred"], ref_pred.numpy().astype(np.int16))
        assert rec["n_bank"] == N and rec["n_query"] == 20 and 0 <= rec["top1"] <= 100
    assert json.loads((out / "fa" / "fullbank_knn.json").read_text(encoding="utf-8"))["flip_consistency"] == 100.0


def test_role_stream_resumes_at_the_right_shard(tmp_path, monkeypatch):
    """The default stream (mirror.final.data.cache get_role("stream_train").batches, which ignores start) resumes at
    the shard holding image `start` and skips inside it: every start gives exactly the remaining images."""
    FBS = load_script("fp_fullbank")
    import mirror.final.data.cache as C

    shard_rows = np.array([5, 7, 4])
    np.savez(tmp_path / C.STREAM_INDEX, shard_rows=shard_rows)
    all_ids = np.concatenate([np.stack([np.full(k, i), np.arange(k)], 1) for i, k in enumerate(shard_rows)])

    class FakeRole:
        def __len__(self):
            return int(shard_rows.sum())

        def batches(self, batch_size=256, start_shard=0, **kw):
            for s in range(int(shard_rows[:start_shard].sum()), len(self), batch_size):
                ids = all_ids[s:s + batch_size]
                yield np.zeros((len(ids), 2, 2, 3), np.uint8), ids[:, 0], ids

    monkeypatch.setattr(C, "get_role", lambda name, cache_dir=None: FakeRole())
    factory, n = FBS.role_stream_factory(tmp_path, 3)
    assert n == 16
    for start in (0, 4, 5, 6, 12, 15):
        assert np.array_equal(np.concatenate([b[2] for b in factory(start)]), all_ids[start:])


def test_fp_fullbank_dry_run_end_to_end(tmp_path):
    FBS = load_script("fp_fullbank")
    out = tmp_path / "A10"
    res = FBS.main(["--dry-run", "--out", str(out)])
    rows = {r["name"]: r for r in res["rows"]}
    assert set(rows) == {"original", "student", "original_flipavg"}
    assert rows["original_flipavg"]["fullbank_flip_consistency"] == 100.0 and rows["original_flipavg"]["kinds"] == "knn"
    for name in ("original", "student"):
        assert 0 <= rows[name]["linear_val_top1"] <= 100 and rows[name]["n_bank"] == 96
        lp = json.loads((out / name / "linear_protocol.json").read_text(encoding="utf-8"))
        assert lp["dinov2"]["commit"].startswith("7764ea0f") and len(lp["trials"]) == 4 * 3
        assert lp["n_holdout"] > 0 and lp["chosen"]["name"] in {t["name"] for t in lp["trials"]}
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["experiment"].startswith("A10") and (out / "summary.csv").is_file()
    assert np.load(out / "train_ids.npy").shape == (96, 2)                    # (shard, row) ids, as the infra stream
    res2 = FBS.main(["--dry-run", "--out", str(out), "--stage", "knn,linear"])        # everything reused
    assert [r["linear_val_top1"] for r in res2["rows"]] == [r["linear_val_top1"] for r in res["rows"]]
    with pytest.raises(SystemExit):
        FBS.parse_models(["a=x", "a=y"], "", "")
    assert FBS.parse_models(["m=runs/x/s0@780", "o=original:dinov2_vitb14_reg"], "m", "o") == [
        ("m", {"run_dir": "runs/x/s0", "step": 780, "kinds": ("knn",), "flipavg": False}),
        ("o", {"original": "dinov2_vitb14_reg", "kinds": ("knn", "linear"), "flipavg": True})]
