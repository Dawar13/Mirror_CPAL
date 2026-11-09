"""Flip-averaging baseline (scripts/flip_avg_baseline.py): CPU tests with the tiny random ViT and synthetic data."""
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from mirror.adapters import TimmTinyAdapter
from mirror.data import make_loader, mirror, normalize_for, synthetic_dataset
from mirror.knn import evaluate_knn
from mirror.knn import extract_features as knn_features

ROOT = Path(__file__).resolve().parents[1]
FAST = ["--max-iter", "60", "--wd-grid", "1e-3,1e-2", "--no-wd-refine"]


def load_script(name):
    key = f"{name}_flipavgtest"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod          # dataclasses need the module registered while it executes
        spec.loader.exec_module(mod)
    return sys.modules[key]


FA = load_script("flip_avg_baseline")
T2 = FA.T2


def tiny_images(n_per_class=3, seed=5, img=56):
    raw, labels = synthetic_dataset(2, n_per_class, img, seed=seed)
    return torch.from_numpy(raw).permute(0, 3, 1, 2), labels


def tiny_loaders(seed=0):
    data = T2.TinyData(4, 20, 8, seed=seed, batch_size=16)
    return data, data.loader("bank"), data.loader("val")


# ---------------------------------------------------------------- exact invariance of the flip-averaged features
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_flip_averaged_features_are_exactly_invariant(dtype):
    ad = TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=dtype)
    raw, _ = tiny_images()
    x = normalize_for(ad, raw)
    wrapped = FA.FlipAveragedAdapter(ad)
    with torch.no_grad():
        f, fm = wrapped.cls_feature(x), wrapped.cls_feature(mirror(x))
        p, pm = FA.flip_averaged_probe_feature(ad, x), FA.flip_averaged_probe_feature(ad, mirror(x))
        plain, plain_m = ad.cls_feature(x).float(), ad.cls_feature(mirror(x)).float()
    # Bit for bit, in every precision (bf16 included): the sum of the two passes is commutative.
    assert torch.equal(f, fm) and torch.equal(p, pm)
    # The definition: the CLS token after the final norm averaged over the image and its mirror (before any
    # L2 normalisation); the probe feature [CLS, mean patch] averaged as a whole.
    assert torch.equal(f, 0.5 * (plain + plain_m))
    assert torch.equal(p, 0.5 * (T2.probe_feature(ad, x) + T2.probe_feature(ad, mirror(x))))
    # Teeth: the plain features of an image and its mirror differ.
    assert (plain - plain_m).abs().max() > 1e-3
    # The wrapper delegates everything else to the adapter.
    assert wrapped.depth == ad.depth and wrapped.blocks is ad.blocks and wrapped.dtype == ad.dtype


def test_knn_feature_is_averaged_before_l2_normalisation():
    ad = TimmTinyAdapter(img_size=56, seed=0)
    raw, labels = tiny_images()
    loader = make_loader(raw.permute(0, 2, 3, 1).numpy(), labels, np.arange(len(labels)), batch_size=4)
    got, _ = knn_features(FA.FlipAveragedAdapter(ad), loader, "cpu")
    x = normalize_for(ad, raw)
    with torch.no_grad():
        want = F.normalize(0.5 * (ad.cls_feature(x) + ad.cls_feature(mirror(x))), dim=-1)
        after = F.normalize(F.normalize(ad.cls_feature(x), dim=-1) + F.normalize(ad.cls_feature(mirror(x)), dim=-1), dim=-1)
    assert torch.allclose(got, want, atol=1e-6)
    assert not torch.allclose(got, after, atol=1e-4)        # the other order would be a different feature


def test_flip_averaged_knn_has_full_flip_consistency_on_the_tiny_model():
    ad = T2.build_adapter("tiny", True, "cpu")
    data, bank, val = tiny_loaders()
    r = FA.knn_eval(ad, bank, val, "cpu", data.num_classes, flip_average=True, amp="bf16")
    assert r["flip_consistency"] == 100.0 and r["feature_identical_pct"] == 100.0 and r["feature_flip_max_abs"] == 0.0
    assert torch.equal(r["pred"], r["pred_flip"])
    plain = FA.knn_eval(ad, bank, val, "cpu", data.num_classes, flip_average=False, amp="bf16")
    assert plain["flip_consistency"] < 100.0 and plain["feature_identical_pct"] == 0.0     # teeth


# ---------------------------------------------------------------- plain mode is the paper's k-NN
def test_plain_mode_reproduces_evaluate_knn():
    ad = T2.build_adapter("tiny", True, "cpu")
    data, bank, val = tiny_loaders()
    other = TimmTinyAdapter(img_size=56, seed=3).model        # a different network through `model=`
    for model in (None, other):
        ref = evaluate_knn(ad, bank, val, "cpu", data.num_classes, k=20, temperature=0.07, model=model, amp="bf16")
        got = FA.knn_eval(ad, bank, val, "cpu", data.num_classes, k=20, temperature=0.07, model=model, amp="bf16")
        assert got["top1"] == ref["top1"] and got["flip_consistency"] == ref["flip_consistency"]
        assert torch.equal(got["pred"], ref["pred"]) and torch.equal(got["pred_flip"], ref["pred_flip"])
        assert (got["n_bank"], got["n_query"]) == (ref["n_bank"], ref["n_query"])


# ---------------------------------------------------------------- cost: the collector's model
def test_cost_reproduces_the_paper_tables():
    tables = ROOT / "results" / "paper" / "tables"
    sources = {"vlocal_k9": ROOT / "runs" / "t2" / "probe" / "dinov2_vitb14_reg" / "vlocal_k9.json",
               "healed_mirror_k9": ROOT / "runs" / "t2" / "probe" / "dinov2_vitb14_reg" / "healed_mirror_k9.json",
               "vexact_k12": ROOT / "runs" / "m3_vexact" / "k12" / "summary.json"}
    need = [tables / "cost_model.json", tables / "T_main_zeroshot.csv", *sources.values()]
    if not all(p.exists() for p in need):
        pytest.skip("paper tables or result files not present")
    arch = next(a for a in json.loads((tables / "cost_model.json").read_text())["architectures"]
                if a["model"] == "dinov2_vitb14_reg")
    g = math.isqrt(arch["n_patches"])
    fake = SimpleNamespace(grid_hw=(g, g), embed_dim=arch["d"], mlp_hidden=arch["h"], depth=arch["depth"],
                           n_tokens=arch["T"], patch_size=arch["patch"],
                           model=SimpleNamespace(patch_embed=SimpleNamespace(proj=SimpleNamespace(in_channels=3))))
    mine = FA.cost_arch(fake, arch["model_params_dense"])
    assert mine == {k: arch[k] for k in mine}
    main = pd.read_csv(tables / "T_main_zeroshot.csv")
    main = main[main["model"] == "dinov2_vitb14_reg"].set_index(["k", "variant"])
    orig = FA.variant_cost({"family": "original", "method": "original", "k": 0}, mine)
    assert orig["gflops_per_forward"] == pytest.approx(main.loc[(0, "original"), "gflops_model"], rel=1e-9)
    for name in ("vlocal_k9", "healed_mirror_k9"):
        info = json.loads(sources[name].read_text())["variant_info"]
        c = FA.variant_cost(info, mine)
        assert c["gflops_per_forward"] == pytest.approx(main.loc[(9, "vlocal"), "gflops_model"], rel=1e-9)
        assert c["cost_model_params"] == info["model_params"] == main.loc[(9, "vlocal"), "model_params"]
    mi = json.loads(sources["vexact_k12"].read_text())["method_info"]
    c = FA.variant_cost({"family": "healed_vexact", "method": "vexact", "k": 12, **mi}, mine)
    assert c["gflops_per_forward"] == pytest.approx(main.loc[(12, "vexact"), "gflops_model"], rel=1e-9)
    assert c["cost_model_params"] == mi["model_params"]
    assert [FA.rotations_per_block(m) for m in ("mirror", "vlocal", "random", "vexact", "svd")] == [1, 1, 1, 0, 0]


# ---------------------------------------------------------------- CLI: the --tiny path end to end
@pytest.fixture(scope="module")
def tiny_inputs(tmp_path_factory):
    """Tiny operators and M1 structure (as tests/test_tier2.py makes them), a tiny healed Mirror run (k = 3) and a
    tiny healed V-exact run at full conversion (k = 4, the even head)."""
    from test_tier2 import _tiny_m1_structure, _tiny_operators

    root = tmp_path_factory.mktemp("flipavg_inputs")
    ops = _tiny_operators(root / "ops.npz")
    s = _tiny_m1_structure(root / "m1_structure.npz", ops)
    load_script("heal").main(["--config", str(ROOT / "configs" / "m3_pilot.yaml"), "--tiny", "--method", "mirror",
                              "--out-dir", str(root / "heal")])
    load_script("heal_vexact").main(["--config", str(ROOT / "configs" / "m3_vexact.yaml"), "--tiny", "--k", "4",
                                     "--lr", "1e-3", "--max-steps", "3", "--out-dir", str(root / "vexact")])
    return {"ops": ops, "m1": s, "heal": root / "heal", "vexact": root / "vexact"}


def test_cli_tiny_end_to_end_and_resume(tiny_inputs, tmp_path, monkeypatch):
    out = tmp_path / "flipavg"
    args = ["--tiny", "--variants", "original,vlocal_k2,healed_mirror_k3", "--out", str(out),
            "--operators", str(tiny_inputs["ops"]), "--m1-structure", str(tiny_inputs["m1"]),
            "--heal-dir", str(tiny_inputs["heal"]), "--vexact-dir", str(tiny_inputs["vexact"]), "--vexact-fp32", *FAST]
    res = FA.main(args)
    rows = {(r["variant"], r["mode"]): r for r in res["rows"]}
    assert list(rows) == [("original", "plain"), ("original", "flipavg"), ("vlocal_k2", "plain"),
                          ("vlocal_k2", "flipavg"), ("healed_mirror_k3", "plain"), ("healed_mirror_k3", "flipavg"),
                          ("healed_vexact_k4", "plain"), ("healed_vexact_k4", "plain_fp32")]
    # Flip-averaged rows: exactly invariant, measured.
    for (v, mode), r in rows.items():
        if mode == "flipavg":
            assert r["flip_consistency"] == 100.0 and r["feature_identical_pct"] == 100.0 and r["feature_flip_max_abs"] == 0
    assert rows[("original", "plain")]["flip_consistency"] < 100.0
    # Plain mode is the paper's k-NN (mirror.knn.evaluate_knn on the same data).
    ad = T2.build_adapter("tiny", True, "cpu")
    data, bank, val = tiny_loaders()
    ref = evaluate_knn(ad, bank, val, "cpu", data.num_classes, amp="bf16")
    assert rows[("original", "plain")]["knn_top1"] == ref["top1"]
    assert rows[("original", "plain")]["flip_consistency"] == ref["flip_consistency"]
    # Cost: two forward passes double the analytic GFLOPs; parameters do not change with the mode.
    for v in ("original", "vlocal_k2", "healed_mirror_k3"):
        p, fa = rows[(v, "plain")], rows[(v, "flipavg")]
        assert (p["forwards_per_image"], fa["forwards_per_image"]) == (1, 2)
        assert fa["gflops"] == pytest.approx(2 * p["gflops"], rel=1e-12) and fa["params"] == p["params"]
        assert p["probe_top1"] is not None and fa["probe_top1"] is not None
    o, vx = rows[("original", "plain")], rows[("healed_vexact_k4", "plain")]
    assert vx["params"] < o["params"] and vx["gflops"] < o["gflops"] and o["params_frac"] == 1.0
    assert vx["vexact_checks_pass"] and vx["vexact_even_head"] and vx["forwards_per_image"] == 1
    assert rows[("healed_vexact_k4", "plain_fp32")]["probe_top1"] is None
    assert rows[("healed_vexact_k4", "plain_fp32")]["knn_amp"] == "none"
    # Outputs: CSV with the requested columns, report without em-dashes, meta JSON, predictions.
    df = pd.read_csv(out / "flipavg.csv")
    for c in ("variant", "mode", "knn_top1", "flip_consistency", "probe_top1", "forwards_per_image", "gflops", "params",
              "seed", "git_hash"):
        assert c in df.columns
    assert len(df) == 8 and (df["seed"] == 0).all() and df["git_hash"].nunique() == 1
    md = (out / "FLIPAVG.md").read_text(encoding="utf-8")
    assert chr(0x2014) not in md and "Exactly mirror-invariant options" in md and "healed_vexact_k4" in md
    meta = json.loads((out / "flipavg_meta_tiny.json").read_text())
    assert meta["seed"] == 0 and meta["git"]["git_hash"] and "before the L2 normalisation" in meta["features"]["knn"]
    with np.load(out / "knn" / "tiny" / "original__flipavg_pred.npz") as z:
        assert np.array_equal(z["pred"], z["pred_flip"]) and len(z["labels"]) == 32
    # Rerun: every (variant, mode) is finished, so nothing is recomputed and the rows come back unchanged.

    def boom(*a, **k):
        raise AssertionError("recomputed")

    monkeypatch.setattr(FA, "knn_eval", boom)
    monkeypatch.setattr(T2, "extract_features", boom)
    monkeypatch.setattr(T2, "run_probe", boom)
    res2 = FA.main(args)
    assert [(r["variant"], r["mode"], r["knn_top1"], r["probe_top1"]) for r in res2["rows"]] == \
           [(r["variant"], r["mode"], r["knn_top1"], r["probe_top1"]) for r in res["rows"]]
    # A changed k-NN setting redoes the k-NN part only (cached probe features and probes are kept).
    monkeypatch.setattr(FA, "knn_eval", lambda *a, **k: {"top1": 1.0, "flip_consistency": 100.0, "n_bank": 80,
                                                         "n_query": 32, "feature_flip_max_abs": 0.0,
                                                         "feature_identical_pct": 100.0, "pred": torch.zeros(32, dtype=torch.long),
                                                         "pred_flip": torch.zeros(32, dtype=torch.long),
                                                         "labels": torch.zeros(32, dtype=torch.long)})
    res3 = FA.main(["--tiny", "--variants", "original", "--out", str(out), "--knn-k", "10", *FAST])
    assert [r["knn_top1"] for r in res3["rows"]] == [1.0, 1.0]
    assert [r["probe_top1"] for r in res3["rows"]] == [rows[("original", m)]["probe_top1"] for m in ("plain", "flipavg")]


def test_modes_per_variant():
    full, part = T2.parse_variant("healed_vexact_k12"), T2.parse_variant("healed_vexact_k6")
    assert FA.modes_for(full, 12) == ["plain"] and FA.modes_for(full, 12, vexact_fp32=True) == ["plain", "plain_fp32"]
    assert FA.modes_for(part, 12) == ["plain", "flipavg"]          # not invariant below full conversion
    for name in ("original", "healed_mirror_k9", "vlocal_k9"):
        assert FA.modes_for(T2.parse_variant(name), 12) == ["plain", "flipavg"]
