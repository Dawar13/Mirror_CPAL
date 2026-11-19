"""DINOv2 linear protocol on cached features (mirror/final/eval/linear_protocol.py) and the flip-averaging cost
(flipavg.py): CPU tests on synthetic features. Nothing here downloads data or weights."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from mirror.data import mirror
from mirror.final.eval import flipavg as FA
from mirror.final.eval import knn_fast as KF
from mirror.final.eval import linear_protocol as LP

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def T2():
    spec = importlib.util.spec_from_file_location("tier2_linear_fp_eval_test", ROOT / "scripts" / "tier2_linear.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tier2_linear_fp_eval_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_parallel_grid_equals_torch_sgd_with_cosine_annealing():
    """The hand-written update equals DINOv2's: one torch.optim.SGD (momentum 0.9, no weight decay, one parameter
    group per classifier) over separate nn.Linear classifiers, the summed cross-entropy, CosineAnnealingLR stepped
    every iteration (float64)."""
    d, C, T, B = 3, 4, 12, 6
    cfg = LP.LinearConfig(base_lrs=(0.5, 2.0, 8.0), batch_size=B, iterations=T, seed=3)
    g = torch.Generator().manual_seed(0)
    X = torch.randn(30, 5 * d, generator=g, dtype=torch.float64)
    y = torch.randint(0, C, (30,), generator=g)
    grid = LP.LinearGrid(d, C, cfg, "cpu", seed=3, dtype=torch.float64)
    G = grid.G
    lins, groups = [], []
    for t, (n, a, s, e) in enumerate(grid.types):
        for j in range(G):
            lin = nn.Linear(e - s, C).double()
            with torch.no_grad():
                lin.weight.copy_(grid.W[t][:, j * C:(j + 1) * C].T)
                lin.bias.copy_(grid.b[t][j * C:(j + 1) * C])
            lins.append((lin, s, e))
            groups.append({"params": lin.parameters(), "lr": float(grid.lrs[j])})
    opt = torch.optim.SGD(groups, momentum=0.9, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T, eta_min=0)
    stream = LP.BatchStream(np.arange(30), B, 7, "cpu")
    for it in range(T):
        idx = stream.next()
        xb, yb = X[idx], y[idx]
        loss = sum(nn.CrossEntropyLoss()(lin(xb[:, s:e]), yb) for lin, s, e in lins)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        grid.step(xb, yb, it, T)
    for t, (n, a, s, e) in enumerate(grid.types):
        for j in range(G):
            lin = lins[t * G + j][0]
            assert torch.allclose(grid.W[t][:, j * C:(j + 1) * C].T, lin.weight, atol=1e-10, rtol=0)
            assert torch.allclose(grid.b[t][j * C:(j + 1) * C], lin.bias, atol=1e-10, rtol=0)
    assert cfg.lrs() == pytest.approx([lr * B / 256 for lr in (0.5, 2.0, 8.0)])
    assert LP.LinearConfig().lrs()[0] == pytest.approx(1e-5 * 1024 / 256) and len(LP.LinearConfig().base_lrs) == 13


def _separable(n_per, d, C, seed, signal_cols):
    """Gaussian features with class means (shared by every split: fixed seed) in `signal_cols` only."""
    means = np.random.default_rng(42).normal(size=(C, len(signal_cols))).astype(np.float32) * 4.0
    g = np.random.default_rng(seed)
    y = np.repeat(np.arange(C), n_per)
    X = g.normal(size=(len(y), 5 * d)).astype(np.float32)
    X[:, signal_cols] += means[y]
    return X.astype(np.float16), y


def test_protocol_on_separable_features_chooses_on_the_training_hold_out():
    d, C = 8, 4
    signal = np.arange(4 * d, 5 * d)                       # class signal only in the mean patch token
    Xtr, ytr = _separable(100, d, C, 0, signal)
    Xv, yv = _separable(50, d, C, 1, signal)
    cfg = LP.LinearConfig(base_lrs=(1e-3, 1e-1, 1.0), batch_size=32, iterations=150, seed=0, log_every=0)
    res = LP.run_linear_protocol(Xtr, ytr, Xv, yv, d, C, cfg)
    assert res["val_top1"] >= 95.0 and res["chosen"]["avgpool"] is True
    assert res["n_holdout"] == 40 and res["n_selection_train"] == 360 and res["n_val"] == 200
    assert len(res["trials"]) == 12 and all("val_top1_refit" in t for t in res["trials"])
    assert res["val_top1_dinov2_rule"] >= res["val_top1"] - 1e-9 and "comparison" in res["dinov2_rule_note"]
    assert res["val_pred"].shape == (200,) and len(res["deviations"]) == 3
    shuffled = np.random.default_rng(5).permutation(yv)    # the choice never looks at validation labels
    res2 = LP.run_linear_protocol(Xtr, ytr, Xv, shuffled, d, C, cfg)
    pick = ("name", "holdout_top1", "lr")
    assert [res2["chosen"][k] for k in pick] == [res["chosen"][k] for k in pick]
    res3 = LP.run_linear_protocol(Xtr, ytr, Xv, yv, d, C, LP.LinearConfig(**{**cfg.__dict__, "refit": "chosen"}))
    assert res3["chosen"] == res["chosen"] and res3["val_top1"] >= 95.0 and "val_top1_dinov2_rule" not in res3


def test_stratified_holdout_equals_tier2_linear(T2):
    labels = np.random.default_rng(0).integers(0, 7, size=300)
    for frac, seed in ((0.1, 0), (0.25, 3)):
        a, b = LP.stratified_holdout(labels, frac, seed), T2.stratified_holdout(labels, frac, seed)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_flip_averaged_features_and_analytic_cost(T2, tmp_path):
    ad, model, _ = KF.tiny_loader("run", 1)
    x = torch.randn(2, 3, 56, 56, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        assert torch.equal(FA.probe_feature(ad, x, model), T2.probe_feature(ad, x, model))
        for fn in (FA.cls_feature_flipavg, FA.probe_feature_flipavg, FA.linear_feature_flipavg):
            assert torch.equal(fn(ad, x, model), fn(ad, mirror(x), model))
        assert torch.equal(FA.FlipAveragedAdapter(ad).cls_feature(x, model), FA.cls_feature_flipavg(ad, x, model))
    single = FA.analytic_cost(ad)                           # the dense original
    cr = FA._collect_results()
    assert single["gflops_per_forward"] == pytest.approx(cr.analytic_cost(FA.cost_arch(ad), [])["gflops_model"])
    two = FA.flipavg_cost(single)
    assert two["gflops_per_image"] == pytest.approx(2 * single["gflops_per_forward"])
    assert two["forward_passes_per_image"] == 2
    assert FA.analytic_cost(ad, model, info={"cost": {"gflops": 1.25}})["gflops_per_forward"] == 1.25
    (tmp_path / "metrics.json").write_text(json.dumps({"cost": {"gflops": 2.5, "params_model": 7}}), encoding="utf-8")
    c = FA.analytic_cost(ad, model, run_dir=tmp_path)
    assert c["gflops_per_forward"] == 2.5 and c["source"] == "metrics.json cost"
    assert FA.analytic_cost(ad, model)["gflops_per_forward"] is None       # a student without spec or recorded cost
