"""Findings track F1 to F3: helpers, the P1 evaluator and end-to-end runs on synthetic inputs (CPU)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mirror import findings as F  # noqa: E402
from mirror.fit import random_involution  # noqa: E402
from mirror.matching import random_pairing  # noqa: E402
from mirror.surgery import equivariant_param_count  # noqa: E402


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def f1():
    return _script("f1_tax")


@pytest.fixture(scope="module")
def f2():
    return _script("f2_drift")


@pytest.fixture(scope="module")
def f3():
    return _script("f3_artefacts")


# ----------------------------------------------------------------------------------------------
# Synthetic run folders in the layout of scripts/run_e1.py
# ----------------------------------------------------------------------------------------------
def make_run(root: Path, arm: str, fit_seed: int, model: str, L=4, d=8, h=16, same_ops=False, with_G=True):
    rng = np.random.default_rng(100 * fit_seed + len(model))
    res = root / f"{arm}_s{fit_seed}" / "results"
    res.mkdir(parents=True)
    d_plus = [5] * (L + 1) if same_ops else [int(x) for x in rng.integers(3, 7, size=L + 1)]
    A0 = random_involution(d, d_plus[0], seed=7)
    A = [A0 if same_ops else random_involution(d, d_plus[l], seed=1000 * fit_seed + l) for l in range(L + 1)]
    layers = pd.DataFrame([{
        "layer": l, "d": d, "d_plus": d_plus[l], "d_minus": d - d_plus[l], "d_zero": 0,
        "dminus_frac": (d - d_plus[l]) / d, "agree_shared": 0.1, "agree_null": 0.0,
        "rel_e_A_layer": 0.1 + 0.01 * l, "rel_e_A_shared": 0.5, "rel_e_A_shared_norm": 0.4, "rel_e_I": 1.0,
        "rel_e_null_shared": 1.2, "rel_e_null_layer": 1.5, "sq_norm_xt": 1.0, "n_tokens": 100,
        "ratio_null_over_shared": 2.4, "ratio_shared_over_I": 0.5, "rel_e_A_used": 0.1 + 0.01 * l,
        "rel_e_null_used": 1.5, "git_hash": "inputhash", "seed": 0} for l in range(L + 1)])
    layers.to_csv(res / "e1_layers.csv", index=False)
    perms, rows = [], []
    for b in range(1, L + 1):
        n_pairs = int(rng.integers(2, h // 2))
        n_self = h - 2 * n_pairs
        perms.append(random_pairing(h, n_pairs, seed=b))
        tr_in, tr_out = 2 * d_plus[b - 1] - d, 2 * d_plus[b] - d
        eq1, eq2 = equivariant_param_count(h, d, n_self, tr_in), equivariant_param_count(h, d, n_self, tr_out)
        rows.append({"block": b, "h": h, "d": d, "n_pairs": n_pairs, "n_self": n_self, "twin_frac": 2 * n_pairs / h,
                     "dminus_frac": (d - d_plus[b - 1]) / d, "evidence": True, "mean_score": 0.5,
                     "mean_score_pairs": 0.6, "mean_score_self": 0.4, "s1": 0.1 * b, "s2": 0.05 * b, "sb1": 0.0,
                     "sb2": 0.0, "s1_null": 0.4, "s2_null": 0.4, "s1_over_null": 0.25 * b, "s2_over_null": 0.125 * b,
                     "tr_P": n_self, "tr_A_in": tr_in, "tr_A_out": tr_out, "eq_params_fc1": eq1, "eq_params_fc2": eq2,
                     "dense_params": 2 * h * d, "param_saving_frac": 1 - (eq1 + eq2) / (2 * h * d), "svd_rank": 3,
                     "svd_rank_fc2": 3, "svd_params": 6 * (h + d), "git_hash": "inputhash", "seed": 0})
    pd.DataFrame(rows).to_csv(res / "e1_mlp.csv", index=False)
    A_f32 = np.stack(A).astype(np.float32) + np.float32(1e-6) * rng.standard_normal((L + 1, d, d)).astype(np.float32)
    G = []
    for l in range(L + 1):
        M = rng.standard_normal((d, d))
        G.append(M @ np.diag(np.linspace(10, 0.1, d)) @ M.T)
    payload = dict(A_layers=A_f32, eigvals=rng.standard_normal((L + 1, d)), C=np.zeros((L + 1, d, d), np.float32),
                   perms=np.stack(perms).astype(np.int32),
                   match_scores=rng.uniform(-1, 1, (L, h)).astype(np.float32),
                   self_corr=rng.uniform(-1, 1, (L, h)).astype(np.float32))
    if with_G:
        payload["G"] = np.stack(G).astype(np.float32)
    np.savez_compressed(res / "e1_operators.npz", **payload)
    meta = {"seed": 0, "fit_seed": fit_seed, "git": {"git_hash": "inputhash", "git_dirty": False},
            "config": {"model": {"name": model}, "stats": {"center": True}}, "operator_mode": "per_layer",
            "protocol": "v2", "tiny": False}
    (res / "e1_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return res


# ----------------------------------------------------------------------------------------------
# Helpers in mirror/findings.py
# ----------------------------------------------------------------------------------------------
def test_tax_formula_limits():
    assert F.tax_formula(16, 8, 0, 4) == 0.5                 # no self-twins: half is redundant
    assert F.tax_formula(16, 8, 16, 8) == 0.0                # P = I and A = I: every map is equivariant
    h, d, f, tr = 16, 8, 6, 2
    assert F.tax_formula(h, d, f, tr) == pytest.approx(1 - equivariant_param_count(h, d, f, tr) / (h * d))


def test_clean_involution_is_exact():
    A = random_involution(12, 7, seed=3)
    noisy = (A + 1e-5 * np.random.default_rng(0).standard_normal(A.shape)).astype(np.float32)
    Ac, V, s, info = F.clean_involution(noisy, return_info=True)
    assert np.abs(Ac @ Ac - np.eye(12)).max() < 1e-12 and np.abs(Ac - Ac.T).max() == 0
    assert info["d_plus"] == 7 and info["d_minus"] == 5 and info["max_eig_dev"] < 1e-3
    assert np.abs(Ac - A).max() < 1e-3
    assert np.allclose(Ac @ F.even_basis(V, s), F.even_basis(V, s))


def test_identical_operators_agree_and_angles_are_zero():
    A = random_involution(10, 6, seed=1)
    assert F.agreement(A, A) == pytest.approx(1.0)
    U = np.linalg.qr(np.random.default_rng(1).standard_normal((10, 4)))[0]
    assert F.subspace_agreement(A, A, U) == pytest.approx(1.0)
    G = np.diag(np.arange(1.0, 11.0))
    assert F.energy_weighted_agreement(A, A, G) == pytest.approx(1.0)
    _, V, s, _ = F.clean_involution(A, return_info=True)
    E = F.even_basis(V, s)
    R = np.linalg.qr(np.random.default_rng(2).standard_normal((6, 6)))[0]
    np.testing.assert_allclose(F.principal_cosines(E, E @ R), np.ones(6), atol=1e-12)
    assert F.agreement(A, -A) == pytest.approx(-1.0)


def test_principal_cosines_orthogonal_and_energy_subspace():
    I = np.eye(6)
    np.testing.assert_allclose(F.principal_cosines(I[:, :2], I[:, 2:4]), [0.0, 0.0], atol=1e-15)
    G = np.diag([10.0, 5.0, 3.0, 1.0, 1.0])
    for frac, k in ((0.5, 1), (0.74, 2), (0.9, 3), (0.91, 4), (1.0, 5)):
        U, kk, cap, _ = F.energy_subspace(G, frac)
        assert kk == k and U.shape == (5, k) and cap >= frac - 1e-12


def test_energy_restricted_agreement_ignores_low_energy_directions():
    d = 6
    G = np.diag([1.0, 1.0, 1.0, 1e-6, 1e-6, 1e-6])
    A = np.eye(d)
    B = np.diag([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    U, k, _, _ = F.energy_subspace(G, 0.95)
    assert k == 3
    assert F.agreement(A, B) == pytest.approx(0.0)
    assert F.subspace_agreement(A, B, U) == pytest.approx(1.0)


# ----------------------------------------------------------------------------------------------
# P1 evaluator and gate G3 on planted numbers
# ----------------------------------------------------------------------------------------------
def _layers(values):
    return {l: v for l, v in enumerate(values)}


def test_p1_plain_bullet_threshold_inclusive():
    # powers of two keep the boundary case exact in floating point
    e_all = _layers([0.125] * 9 + [0.5, 0.5, 0.5] + [0.75])
    r = F.evaluate_p1(e_all, _layers([0.125] * 9 + [0.25, 0.25, 0.25] + [0.75]), "plain")
    assert r["pass"] is True and r["value"] == 2.0 and r["result"] == "PASS"
    r = F.evaluate_p1(e_all, _layers([0.125] * 9 + [0.26, 0.26, 0.26] + [0.75]), "plain")
    assert r["pass"] is False and r["result"] == "FAIL" and r["value"] == pytest.approx(0.5 / 0.26)
    # only layers 9 to 11 count: changing layers 0 to 8 and 12 changes nothing
    e2 = _layers([9.0] * 9 + [0.5, 0.5, 0.5] + [9.0])
    assert F.evaluate_p1(e2, _layers([3.0] * 9 + [0.25, 0.25, 0.25] + [0.0]), "plain")["value"] == 2.0
    with pytest.raises(KeyError):
        F.evaluate_p1(e_all, {}, "plain")


def test_p1_registers_bullet_threshold_inclusive():
    early, late = [0.125] * 6, [0.25, 0.25, 0.25]
    ok = F.evaluate_p1(_layers([5.0] + early + [7.0, 7.0] + late + [9.0]), {}, "registers")
    assert ok["pass"] is True and ok["value"] == 2.0
    assert np.isnan(ok["masking_cut_late"])        # no masked errors given: the plain quantity is NaN
    bad = F.evaluate_p1(_layers([0.0] + early + [0.0, 0.0] + [0.26, 0.26, 0.26] + [0.0]), {}, "registers")
    assert bad["pass"] is False
    assert F.evaluate_p1(_layers([0.1] * 9 + [np.nan, 0.2, 0.2] + [0.1]), {}, "registers")["pass"] is None
    with pytest.raises(KeyError):
        F.evaluate_p1(_layers([0.1] * 8), {}, "registers")


def test_p1_constants_match_prereg():
    text = " ".join((ROOT / "PREREG_E2.md").read_text(encoding="utf-8").split())
    assert F.P1_TEXT["plain"] in text and F.P1_TEXT["registers"] in text
    assert (F.P1_LATE_LAYERS, F.P1_EARLY_LAYERS, F.P1_TOP_FRAC) == ((9, 10, 11), (1, 2, 3, 4, 5, 6), 0.01)
    assert F.P1_PLAIN_MIN_CUT == 2.0 and F.P1_REG_MAX_RATIO == 2.0


def test_gate_g3():
    assert F.gate_g3(True, False)["status"] == "PASS"
    assert F.gate_g3(False, True)["keep_claim_C3"] is True
    assert F.gate_g3(False, False)["status"] == "FAIL"
    assert F.gate_g3(None, False)["status"] == "INCOMPLETE"


# ----------------------------------------------------------------------------------------------
# F1
# ----------------------------------------------------------------------------------------------
def test_f1_end_to_end(tmp_path, f1):
    root, out = tmp_path / "x2", tmp_path / "f1"
    for s in (0, 1, 2):
        make_run(root, "perlayer_reg", s, "dinov2_vitb14_reg")
    make_run(root, "perlayer_plain", 0, "dinov2_vitb14")
    meta = f1.main(["--runs", str(root), "--out", str(out)])
    for rel in ("f1_layers_by_seed.csv", "f1_blocks_by_seed.csv", "f1_layers_summary.csv", "f1_blocks_summary.csv",
                "f1_ranges.csv", "f1_meta.json", "F1_SUMMARY.md", "plots/fig2_tax.png"):
        assert (out / rel).is_file(), rel
    blocks = pd.read_csv(out / "f1_blocks_by_seed.csv")
    assert blocks["tax_minus_saving"].abs().max() < 1e-12 and blocks["trace_matches_csv"].all()
    assert meta["max_abs_tax_minus_saving"] < 1e-12 and "git_hash" in meta["git"]
    # mean and sample std over the three seeds, recomputed by hand for block 2
    sel = blocks[(blocks["model"] == "dinov2_vitb14_reg") & (blocks["block"] == 2)]
    summ = pd.read_csv(out / "f1_blocks_summary.csv")
    row = summ[(summ["model"] == "dinov2_vitb14_reg") & (summ["block"] == 2)].iloc[0]
    assert row["tax_formula_mean"] == pytest.approx(sel["tax_formula"].mean())
    assert row["tax_formula_std"] == pytest.approx(sel["tax_formula"].std(ddof=1))
    assert row["n_seeds"] == 3
    # hand check of the formula for one row: fc1 uses layer b-1, fc2 layer b
    lay = pd.read_csv(root / "perlayer_reg_s1" / "results" / "e1_layers.csv").set_index("layer")
    r = blocks[(blocks["model"] == "dinov2_vitb14_reg") & (blocks["fit_seed"] == 1) & (blocks["block"] == 3)].iloc[0]
    tr1 = lay.loc[2, "d_plus"] - lay.loc[2, "d_minus"]
    tr2 = lay.loc[3, "d_plus"] - lay.loc[3, "d_minus"]
    expect = 0.5 * ((0.5 - r["n_self"] * tr1 / (2 * 16 * 8)) + (0.5 - r["n_self"] * tr2 / (2 * 16 * 8)))
    assert r["tax_formula"] == pytest.approx(expect)
    plain = summ[summ["model"] == "dinov2_vitb14"]
    assert (plain["n_seeds"] == 1).all() and plain["tax_formula_std"].isna().all()
    text = (out / "F1_SUMMARY.md").read_text(encoding="utf-8")
    assert "\u2014" not in text and "DINOv2 ViT-B/14 with registers" in text and "0, 1, 2" in text


# ----------------------------------------------------------------------------------------------
# F2
# ----------------------------------------------------------------------------------------------
def test_f2_identical_operators(tmp_path, f2):
    make_run(tmp_path / "x2", "perlayer_reg", 0, "dinov2_vitb14_reg", same_ops=True)
    out = tmp_path / "f2"
    f2.main(["--run", str(tmp_path / "x2" / "perlayer_reg_s0"), "--out", str(out)])
    agree = pd.read_csv(out / "f2_agreement.csv")
    # stored float32 operators carry independent 1e-6 noise per layer, so "identical" holds to ~1e-11
    np.testing.assert_allclose(agree["agree_full"], 1.0, atol=1e-9)
    np.testing.assert_allclose(agree["agree_energy"], 1.0, atol=1e-9)
    np.testing.assert_allclose(agree["agree_energy_weighted"], 1.0, atol=1e-9)
    nb = pd.read_csv(out / "f2_neighbours.csv")
    np.testing.assert_allclose(nb["pa_mean_cos"], 1.0, atol=1e-9)
    np.testing.assert_allclose(nb["pa_min_cos"], 1.0, atol=1e-9)
    assert (nb["pa_null_mean_cos"] < 1.0).all()
    assert (out / "plots" / "fig3a_drift.png").is_file() and (out / "plots" / "fig3a_drift_energy.png").is_file()
    meta = json.loads((out / "f2_meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 0 and "git_hash" in meta["git"] and meta["cleaning"]["d_plus_mismatch_rows"] == 0


def test_f2_several_runs_and_fast_path(tmp_path, f2):
    root = tmp_path / "x2"
    for s in (0, 1):
        make_run(root, "perlayer_reg", s, "dinov2_vitb14_reg")
    make_run(root, "perlayer_plain", 0, "dinov2_vitb14", with_G=False)
    out = tmp_path / "f2"
    f2.main(["--run", str(root / "perlayer_reg_s0"), str(root / "perlayer_reg_s1"),
             "--run", str(root / "perlayer_plain_s0"), "--out", str(out)])
    agree = pd.read_csv(out / "f2_agreement.csv")
    diag = agree[agree["l"] == agree["m"]]
    np.testing.assert_allclose(diag["agree_full"], 1.0, atol=1e-12)
    assert agree[agree["model"] == "dinov2_vitb14"]["agree_energy"].isna().all()   # no G: NaN, not a crash
    seeds = pd.read_csv(out / "f2_seed_agreement.csv")
    assert len(seeds) == 5 and set(seeds["model"]) == {"dinov2_vitb14_reg"}
    assert (seeds["pa_null_mean_cos"] < 1 - 1e-6).all()          # nulls differ between runs
    E0 = F.random_subspace(8, 3, seed=0)
    np.testing.assert_allclose(E0.T @ E0, np.eye(3), atol=1e-12)
    # the precomputed energy agreement equals the direct definition
    run = f2.resolve_run(str(root / "perlayer_reg_s0"))
    res = f2.analyse_run(run, 0.95, 0, 1)
    with np.load(run["npz"]) as z:
        G = z["G"].astype(np.float64)
    U, _, _, _ = F.energy_subspace(G[1], 0.95)
    assert res["energy"][1, 3] == pytest.approx(F.subspace_agreement(res["A"][1], res["A"][3], U), abs=1e-10)
    ew = agree[(agree["run"] == "perlayer_reg_s0") & (agree["l"] == 2) & (agree["m"] == 0)]["agree_energy_weighted"].iloc[0]
    assert ew == pytest.approx(F.energy_weighted_agreement(res["A"][2], res["A"][0], G[2]), abs=1e-10)


# ----------------------------------------------------------------------------------------------
# F3
# ----------------------------------------------------------------------------------------------
def test_f3_analyse_layer_planted(f3):
    # 2 images, CLS + 4 x 4 patch grid; all norms 1 except planted high tokens.
    H = W = 4
    N, T = 2, 1 + H * W
    nx, nxt = np.ones((N, T)), np.ones((N, T))
    nx[0, 1 + 1 * W + 0] = 100.0          # original image 0: patch (1, 0)
    nxt[0, 1 + 1 * W + 0] = 90.0          # mirror (un-flipped) image 0: same content position
    nx[1, 1 + 2 * W + 1] = 80.0           # original image 1: patch (2, 1)
    nxt[1, 1 + 2 * W + (W - 1 - 1)] = 70.0  # mirror image 1: at the same raw grid position, not mirrored
    err, sq = np.full((N, T), 0.1), np.ones((N, T))
    err[0, 1 + 1 * W + 0] = 5.0
    out = f3.analyse_layer(nx, nxt, err, sq, n_prefix=1, grid_hw=(H, W), top_frac=4 / (2 * N * T))
    assert out["frac_high_top1"] == pytest.approx(4 / (2 * N * T))
    assert out["n_high_mirror_top1"] == 2 and out["n_high_orig_top1"] == 2
    assert out["coincide_mirrored_top1"] == pytest.approx(0.5) and out["coincide_raw_top1"] == pytest.approx(0.5)
    assert out["coincide_chance_top1"] == pytest.approx(1 / 16)
    excluded = {(0, 1 + W), (1, 1 + 2 * W + 1), (1, 1 + 2 * W + 2)}
    keep = np.ones((N, T), bool)
    for i, j in excluded:
        keep[i, j] = False
    assert out["e_all"] == pytest.approx(err.sum() / sq.sum())
    assert out["e_excl_top1"] == pytest.approx(err[keep].sum() / sq[keep].sum())
    assert out["e_excl_3xmed"] == pytest.approx(out["e_excl_top1"])          # same tokens are > 3x median
    assert out["thr_3x_median"] == pytest.approx(3.0)
    assert np.isnan(out["e_registers"]) and out["e_excl_registers"] == pytest.approx(out["e_all"])
    assert out["share_of_high_top1_patches"] == pytest.approx(1.0)


def test_f3_errors_match_e1_definition(tiny64, f3):
    """Per-token errors summed over tokens equal stats.collect_heldout_errors (the E1 definition)."""
    from mirror.data import make_loader, synthetic_dataset
    from mirror.stats import collect_heldout_errors

    images, labels = synthetic_dataset(2, 3, 56, seed=1)
    loader = make_loader(images, labels, np.arange(len(labels)), 4)
    n, d = tiny64.depth + 1, tiny64.embed_dim
    ops = [random_involution(d, 40, seed=l) for l in range(n)]
    arrays = f3.collect_tokens(tiny64, loader, torch.device("cpu"), ops)
    ref = collect_heldout_errors(tiny64, loader, torch.device("cpu"), [{"A": A} for A in ops])
    df = f3.layer_table(arrays, tiny64.n_prefix_tokens, tiny64.grid_hw)
    for l in range(n):
        assert df.loc[l, "e_all"] == pytest.approx(ref["err"][l]["A"] / ref["sq_xt"][l], rel=1e-10)


def test_f3_tiny_end_to_end_and_gate(tmp_path, f3):
    out = tmp_path / "f3"
    r_plain = f3.main(["--tiny", "--out", str(out)])
    r_reg = f3.main(["--tiny", "--tiny-registers", "2", "--out", str(out)])
    for name, r in (("tiny", r_plain), ("tiny_reg2", r_reg)):
        j = json.loads((out / f"f3_{name}.json").read_text(encoding="utf-8"))
        assert j["seed"] == 0 and "git_hash" in j["git"] and j["tiny"] is True
        assert j["p1"]["pass"] in (True, False, None) and j["p1"]["result"] in ("PASS", "FAIL", "MISSING")
        assert j["prereg"]["sha256"] and j["operators"]["cleaning_max_eig_dev"] < 1e-3
        df = pd.read_csv(out / f"f3_{name}_layers.csv")
        assert len(df) == 5 and df["git_hash"].notna().all()
        assert df["e_all"].notna().all() and df["e_excl_top1"].notna().all()
    assert r_plain["p1"]["bullet"] == "plain" and r_reg["p1"]["bullet"] == "registers"
    assert r_reg["data"]["n_registers"] == 2
    reg_df = pd.read_csv(out / "f3_tiny_reg2_layers.csv")
    assert reg_df["e_registers"].notna().all() and reg_df["norm_mean_registers"].notna().all()
    gate = json.loads((out / "f3_gate_G3.json").read_text(encoding="utf-8"))
    assert gate["status"] in ("PASS", "FAIL", "INCOMPLETE") and gate["tiny"] is True
    assert (out / "plots" / "fig3b_artefacts.png").is_file()
    with np.load(out / "f3_tiny_tokens.npz") as z:
        assert z["err"].shape == (5, 20, 17)
