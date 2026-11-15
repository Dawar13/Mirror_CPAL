"""CORE-03 (mirror/stacking.py, scripts/core03_stacking.py): CPU tests with the tiny random ViT and synthetic data.
Float64 where exactness is tested."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mirror import baselines as B
from mirror import mirror_family as MF
from mirror import stacking as ST
from mirror.adapters import TimmTinyAdapter
from mirror.attention import fold_attn
from mirror.data import make_loader, synthetic_dataset
from mirror.heal_vexact import flip_check, span_exactness
from mirror.surgery import fold_mlp

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_core03_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


C = load_script("core03_stacking")


# ---------------------------------------------------------------- fixtures (module scope: one structure fit)
@pytest.fixture(scope="module")
def ad64():
    return TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float64)


@pytest.fixture(scope="module")
def data64():
    images, labels = synthetic_dataset(4, 12, 56, seed=3)
    idx = np.arange(len(labels))
    return {"fit": make_loader(images, labels, idx[:16], 8), "bank": make_loader(images, labels, idx[16:40], 16),
            "query": make_loader(images, labels, idx[40:], 16), "num_classes": 4}


@pytest.fixture(scope="module")
def struct64(ad64, data64):
    return MF.fit_span_structure(ad64, data64["fit"], "cpu", model_name="tiny64", top_k=3, workers=1)


@pytest.fixture(scope="module")
def mirror64(ad64, struct64):
    return MF.build_mirror_variant(ad64, "mirror_exact", range(ad64.depth), MF.COMPONENTS, struct64)


@pytest.fixture(scope="module")
def stats64(ad64, data64):
    return B.collect_stats(ad64, data64["fit"], "cpu", list(range(ad64.depth)), ["svdllm"], MF.COMPONENTS)


def _mats(model, b):
    f, g = fold_mlp(model.blocks[b]), fold_attn(model.blocks[b])
    return {"fc1": f["W1"], "fc2": f["W2"], "wq": g["Wq"], "wk": g["Wk"], "wv": g["Wv"], "wo": g["Wo"]}


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def probe(n=3, seed=5):
    return torch.randn(n, 3, 56, 56, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------- block structure
def test_bases_and_block_structure(ad64, struct64, mirror64):
    model, info = mirror64
    s = info["structure"]
    A = s["A"]
    cache = ST._BasisCache(A, s["perms"], s["heads_after"], info["blocks"])
    # Orthonormal bases with the right +1 / -1 split.
    Q = cache.A.dense()
    assert np.abs(Q.T @ Q - np.eye(len(Q))).max() < 1e-12
    sign = np.r_[np.ones(cache.A.n_plus), -np.ones(cache.A.n_minus)]
    assert np.abs(Q.T @ A @ Q - np.diag(sign)).max() < 1e-10
    for i in range(len(info["blocks"])):
        pb = cache.perm(i)
        E = pb.dense()
        P = np.eye(pb.n)[np.asarray(s["perms"][i])]                         # (P x)[k] = x[perm[k]]
        assert np.abs(E.T @ E - np.eye(pb.n)).max() < 1e-12
        assert np.abs(E.T @ P @ E - np.diag(np.r_[np.ones(pb.n_plus), -np.ones(pb.n_minus)])).max() < 1e-12
        for which in ("B", "C"):
            Ph = ST.head_involution(s["heads_after"][i], which)
            assert np.abs(Ph - Ph.T).max() < 1e-12 and np.abs(Ph @ Ph - np.eye(len(Ph))).max() < 1e-10
    # Every projected matrix is block diagonal; the block sizes add up to the Mirror-exact count.
    d, h, D = ad64.embed_dim, ad64.mlp_hidden, ad64.model.blocks[0].attn.qkv.out_features // 3
    layout = ST.stack_layout(d, h, D, A, s["perms"], s["heads_after"], info["blocks"], MF.COMPONENTS)
    kept = {(r["block"], r["name"].lower()): r["kept"] for r in info["per_matrix"]}
    for b in info["blocks"]:
        mats = _mats(model, b)
        for name, W in mats.items():
            out_b, in_b = ST.matrix_bases(name, cache, b)
            Fp, Fm, res = ST.split_blocks(W, out_b, in_b)
            assert res < 1e-12, (b, name, res)
            lay = {e["sign"]: e for e in layout if e["block"] == b and e["name"] == name}
            assert list(Fp.shape) == lay["even"]["shape"] and list(Fm.shape) == lay["odd"]["shape"]
            assert Fp.numel() + Fm.numel() == kept[(b, name)]
            assert rel(ST.merge_blocks(Fp, Fm, out_b, in_b), W) < 1e-12
        # Teeth: the original (unprojected) weights are not block diagonal.
        orig = _mats(ad64.model, b)
        out_b, in_b = ST.matrix_bases("fc1", cache, b)
        assert ST.split_blocks(orig["fc1"], out_b, in_b)[2] > 1e-2


def test_truncate_block_full_rank_reproduces_the_block():
    g = torch.Generator().manual_seed(0)
    F = torch.randn(9, 6, dtype=torch.float64, generator=g)
    X = torch.randn(40, 6, dtype=torch.float64, generator=g)
    for inner, M in (("svd", None), ("svdllm", X.T @ X)):
        Fn, _ = ST.truncate_block(F, 6, inner, M, force=True)
        assert rel(Fn, F) < 1e-12
        Fr, inf = ST.truncate_block(F, 2, inner, M)
        assert int(torch.linalg.matrix_rank(Fr)) == 2 and 0 < inf["discarded_energy"] < 1
    # The plain truncation equals baselines.lowrank_factors (the arithmetic the low-rank baseline uses).
    Wb, _, _, _ = B.lowrank_factors("svd", F, 3)
    assert rel(ST.truncate_block(F, 3, "svd")[0], Wb) < 1e-14


@pytest.mark.parametrize("inner", ["svd", "svdllm"])
def test_full_rank_reproduces_mirror_exact(ad64, struct64, mirror64, stats64, inner):
    model0, info0 = mirror64
    S = int(info0["params_total"])
    for total, force in ((S, True), (10 * S, False)):          # SVD forced at full rank; budget far above the count
        model, info = ST.build_stacked(ad64, struct64, total, inner=inner, stats=stats64, base=mirror64, force=force)
        assert info["full_rank"] and info["params_total"] == S and info["params_matrices"] == info0["params_matrices"]
        assert info["below_budget"] == (total > S)
        for b in info["blocks"]:
            new, old = _mats(model, b), _mats(model0, b)
            for name in new:
                assert rel(new[name], old[name]) < 1e-10, (inner, b, name)
        per = {(e["block"], e["name"]): e for e in info["per_matrix"]}
        for r in info0["per_matrix"]:
            assert per[(r["block"], r["name"].lower())]["params"] == r["kept"]


# ---------------------------------------------------------------- budgets
@pytest.mark.parametrize("fraction", [0.85, 0.70, 0.55, 0.40, 0.10])
def test_budgets_are_met(ad64, struct64, mirror64, stats64, fraction):
    info0 = mirror64[1]
    span, S = int(info0["span_dense"]), int(info0["params_total"])
    T = ST.budget_for_fraction(span, fraction)
    assert T == math.ceil(fraction * span - 1e-9)
    for inner in ("svd", "svdllm"):
        model, info = ST.build_stacked(ad64, struct64, T, inner=inner, stats=stats64, base=mirror64)
        assert info["budget_total"] == T
        if T >= S:
            assert info["full_rank"] and info["params_total"] == S
        else:
            assert T <= info["params_total"] <= T + info["gap_step"]
            assert not info["full_rank"]
        for e in info["blocks_alloc"]:
            m, n = e["shape"]
            assert 0 <= e["rank"] <= min(m, n) and e["params"] <= m * n
        chk = ST.check_stacked(model, info)
        assert chk["ok"], chk["failures"]                                    # float64: ranks exactly as allocated
        assert chk["params_realised"] == info["params_matrices"]
    # Low rank alone at the same point (mirror/baselines.py compress at the CORE-03 budget).
    budget = MF.span_budget(ad64.model, struct64, range(ad64.depth), MF.COMPONENTS)
    for weights in ("mirror", "dense"):
        fb = ST.fraction_budget(budget, T, weights)
        for method in ("svd", "svdllm"):
            m, inf = B.compress(ad64.model, method, list(range(ad64.depth)), MF.COMPONENTS, fb, stats64,
                                verbose=False)
            assert inf["budget_total"] == T and inf["params_total"] >= min(T, span)
            assert inf["params_total"] - T <= inf["gap_step"] or inf["params_total"] == span
            assert B.structure_check(m, inf)["ok"]


def test_allocation_is_proportional():
    layout = [{"block": 0, "name": "fc1", "sign": "even", "shape": [40, 30], "dense": 1200, "mirror": 1500},
              {"block": 0, "name": "fc1", "sign": "odd", "shape": [10, 30], "dense": 300, "mirror": 1500}]
    al = ST.allocate_stack(layout, 750)
    assert [e["target"] for e in al] == [600, 150]                           # ceil(m n T / S)
    assert sum(e["params"] for e in al) >= 750
    assert sum(e["params"] for e in al) - 750 <= max(m + n for m, n in (e["shape"] for e in al))
    full = ST.allocate_stack(layout, 1500)
    assert all(e["dense_storage"] and e["rank"] == min(e["shape"]) for e in full)


# ---------------------------------------------------------------- exactness (float64)
@pytest.mark.parametrize("inner", ["svd", "svdllm"])
def test_stacked_model_is_exactly_flip_invariant(ad64, struct64, mirror64, stats64, data64, inner):
    L = ad64.depth
    span = int(mirror64[1]["span_dense"])
    model, info = ST.build_stacked(ad64, struct64, ST.budget_for_fraction(span, 0.40), inner=inner, stats=stats64,
                                   base=mirror64)
    assert not info["full_rank"]
    s = info["structure"]
    xs = [probe(3, 5), probe(2, 6)]
    ex = span_exactness(ad64, model, xs, s["A"], L, s["reg_perm"])
    assert ex["pass"] and ex["max_rel_inside"] < 1e-10
    fc = flip_check(ad64, model, data64["bank"], data64["query"], xs, data64["num_classes"], "cpu", k=5)
    assert fc["flip_consistency"] == 100.0 and fc["max_abs_unit_feature_diff"] < 1e-10
    assert fc["cls_feature_rel_max"] < 1e-10
    # Teeth: the stacked model is not Mirror-exact (ranks were cut) and the original is not invariant.
    assert max(rel(_mats(model, 0)[n], _mats(mirror64[0], 0)[n]) for n in ("fc1", "wq")) > 1e-3
    assert flip_check(ad64, ad64.model, data64["bank"], data64["query"], xs, 4, "cpu", k=5)[
        "max_abs_unit_feature_diff"] > 1e-6


def test_trainable_form_stays_invariant_after_a_step(ad64, struct64, mirror64, stats64, data64):
    import copy

    span = int(mirror64[1]["span_dense"])
    model, info = ST.build_stacked(ad64, struct64, ST.budget_for_fraction(span, 0.40), inner="svd", base=mirror64)
    assert not info["full_rank"]
    ref = copy.deepcopy(model)
    model, cons, spec = ST.make_trainable_stacked(model, info, ad64)
    x = probe(2, 7)
    with torch.no_grad():
        assert rel(ad64.cls_feature(x, model), ad64.cls_feature(x, ref)) < 1e-10
    frozen = {id(p) for c in cons if hasattr(c, "frozen_parameters") for p in c.frozen_parameters(model)}
    params = [p for p in model.parameters() if id(p) not in frozen]
    factor_ids = {id(p) for n, p in model.named_parameters() if ".U_" in n or ".V_" in n or n.endswith(".c")}
    assert factor_ids <= {id(p) for p in params}                           # the factors and even biases train
    assert id(model.norm.affine.weight) in frozen                           # the final affine does not
    for p in model.parameters():
        p.requires_grad_(id(p) not in frozen)
    opt = torch.optim.SGD(params, lr=1e-2)
    loss = ad64.cls_feature(probe(2, 8), model).pow(2).sum()
    loss.backward()
    opt.step()
    for c in cons:
        c(model)                                                           # embeddings re-projected
    with torch.no_grad():
        assert rel(ad64.cls_feature(x, model), ad64.cls_feature(x, ref)) > 1e-8      # the step changed the model
        f, ft = ad64.cls_feature(x, model), ad64.cls_feature(torch.flip(x, dims=[-1]), model)
        assert float((f - ft).abs().max() / f.abs().max()) < 1e-10                  # still exactly invariant
    assert cons[0].check(model)["ok"]                                      # every factor keeps its rank
    assert any(not e["dense_storage_even"] for e in spec["ranks"])


# ---------------------------------------------------------------- scoring
def _pts(above: int, n: int = 4):
    return [{"point": f"p{i}", "mirror_lowrank": 50.0 + (1 if i < above else -1), "lowrank": 50.0} for i in range(n)]


def test_score_p31_rules():
    b, l_, d = C.PREREG_MODELS
    s = C.score_p31({b: _pts(4), l_: _pts(3), d: _pts(0)})
    assert s["verdict"] == "PASS" and s["passed"] == [b, l_]
    s = C.score_p31({b: _pts(4), l_: _pts(2), d: _pts(2)})                # 3 of 4 needed per model
    assert s["verdict"] == "FAIL"
    s = C.score_p31({b: _pts(3), l_: _pts(1), d: _pts(4)})                # DeiT-III counts as the second model
    assert s["verdict"] == "PASS"
    s = C.score_p31({b: _pts(3), l_: _pts(3, 3)})                         # L incomplete but already passing
    assert s["verdict"] == "PASS" and s["models"][l_]["status"] == "pass"
    s = C.score_p31({b: _pts(3), l_: _pts(1, 2)})                         # L and DeiT-III still open
    assert s["verdict"] == "incomplete"
    s = C.score_p31({b: _pts(4), l_: _pts(0), d: _pts(1)})                # only B passes: no second model
    assert s["verdict"] == "FAIL"
    ties = [{"point": "p", "mirror_lowrank": 50.0, "lowrank": 50.0}] * 4  # equal is not above
    assert C.score_p31({b: ties, l_: ties, d: ties})["verdict"] == "FAIL"


def test_curve_choice_reads_the_method_selection_set_only():
    rows = []
    for m, msel, val in (("svd", 40.0, 70.0), ("svdllm", 41.0, 60.0)):
        rows.append({"model": "m", "fit_seed": 0, "point": "f0.55", "fraction": 0.55, "curve": "lowrank", "method": m,
                     "msel_top1": msel, "val_top1": val, "status": "ok"})
    for m, msel, val in (("mirror_svd", 30.0, 50.0), ("mirror_svdllm", 35.0, 40.0)):
        rows.append({"model": "m", "fit_seed": 0, "point": "f0.55", "fraction": 0.55, "curve": "mirror_lowrank",
                     "method": m, "msel_top1": msel, "val_top1": val, "status": "ok"})
    df = pd.DataFrame(rows)
    for score_inner, want in (("svd", "mirror_svd"), ("select", "mirror_svdllm"), ("svdllm", "mirror_svdllm")):
        cur = pd.DataFrame(C.build_curves(df, 0, score_inner))
        sc = cur[cur["scored"]]
        assert set(sc["method"]) == {"svdllm", want}                       # svdllm wins on msel despite lower val


# ---------------------------------------------------------------- CLI, tiny, end to end
@pytest.fixture(scope="module")
def cli_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("core03")
    res = C.main(["--tiny", "--out", str(out)])
    return out, res


def test_cli_rows_budgets_and_flip(cli_run):
    out, res = cli_run
    df = pd.read_csv(out / "rows_tiny.csv")
    assert len(df) == 2 + 4 * 4 and (df["status"] == "ok").all() and res["rows_added"] == len(df)
    mirror = df[df["method"] == "mirror_exact"].iloc[0]
    S = int(mirror["params_total"])
    span = int(mirror["span_dense"])
    for _, r in df[df["curve"] == "mirror_lowrank"].iterrows():
        T = int(r["budget_total"])
        assert T == math.ceil(float(r["fraction"]) * span - 1e-9)
        if T >= S:
            assert r["same_as"] == "mirror_exact" and int(r["params_total"]) == S and r["below_budget"]
            assert r["val_top1"] == mirror["val_top1"]
        else:
            assert T <= int(r["params_total"]) <= T + int(r["gap_step"]) and bool(r["structure_check_ok"])
        assert r["flip_consistency_fp32"] == 100.0                         # exactly invariant (fp32 features)
    lr = df[df["curve"] == "lowrank"]
    assert (lr["params_total"] >= lr["budget_total"]).all() and lr["structure_check_ok"].astype(bool).all()
    assert (lr["model_params"] < df[df["method"] == "original"].iloc[0]["model_params"]).all()
    assert set(df[df["curve"] == "lowrank"]["method"]) == {"svd", "svdllm"}
    cur = pd.read_csv(out / "curves.csv")
    assert set(cur[cur["scored"]]["role"]) == {"lowrank_scored", "mirror_lowrank_scored"}
    assert (cur[cur["role"] == "mirror_lowrank_scored"]["method"] == "mirror_svd").all()
    p31 = json.loads((out / "p31.json").read_text())
    assert p31["verdict"] in ("PASS", "FAIL") and p31["score_inner"] == "svd" and p31["models"]["tiny"]["n_points"] == 4
    boot = pd.read_csv(out / "bootstrap.csv")
    assert len(boot) == 4 and (boot["n_resamples"] == 10_000).all()
    text = (out / "SUMMARY.md").read_text(encoding="utf-8")
    assert chr(0x2014) not in text and "P3.1" in text and "TINY" in text
    meta = json.loads((out / "meta.json").read_text())
    assert meta["experiment"] == "CORE-03" and meta["seeds"]["fit_seed"] == 0 and meta["git"]["git_hash"]
    assert meta["extra"]["score_inner"] == "svd" and meta["config"]["score_inner"] == "svd"
    assert (out / "done_tiny_s0.flag").exists()
    for _, r in df.iterrows():
        with np.load(C.preds_path(out, "tiny", r["point"], r["method"], 0)) as z:
            assert len(z["val_pred"]) == r["n_val"] and len(z["msel_pred"]) == r["n_msel"]


def test_cli_resume_skips_finished_rows(cli_run):
    out, _ = cli_run
    before = pd.read_csv(out / "rows_tiny.csv")
    res = C.main(["--tiny", "--out", str(out)])
    assert res["rows_added"] == 0
    pd.testing.assert_frame_equal(pd.read_csv(out / "rows_tiny.csv"), before)
    # Drop one stacked row and one low-rank row at the lowest budget: only those are redone, with the same result.
    drop = (before["point"] == "f0.40") & before["method"].isin(["mirror_svdllm", "svd"])
    before[~drop].to_csv(out / "rows_tiny.csv", index=False)
    res = C.main(["--tiny", "--out", str(out)])
    assert res["rows_added"] == 2
    after = pd.read_csv(out / "rows_tiny.csv")
    for m in ("mirror_svdllm", "svd"):
        old = before[(before["point"] == "f0.40") & (before["method"] == m)].iloc[0]
        new = after[(after["point"] == "f0.40") & (after["method"] == m)].iloc[0]
        assert new["val_top1"] == old["val_top1"] and new["params_total"] == old["params_total"]
        assert new["preds_sha"] == old["preds_sha"]
    res = C.main(["--tiny", "--out", str(out), "--summary-only"])
    assert res["p31"]["verdict"] in ("PASS", "FAIL")
