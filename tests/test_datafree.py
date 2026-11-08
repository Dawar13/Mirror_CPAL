"""Track D (data-free readout): principal angles, D1 on a planted tiny model, D2 on planted twins,
D3 on a planted block, and the script's --tiny mode. CPU only, no downloads."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mirror.datafree import (act_moments, clean_involution, compare_operators, eigenspaces, fit_layer0,
                             layer0_moments, layer0_parts, match_twins, pairing_agreement,
                             plant_mirror_symmetry, principal_cosines, propagate_chain,
                             random_involution_fixing_ones, twin_scores)
from mirror.fit import fit_involution, random_involution, random_orthogonal
from mirror.matching import random_pairing
from mirror.surgery import fold_mlp

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------------------------
# Principal angles and operator comparison
# ----------------------------------------------------------------------------------------------
def test_principal_cosines_identical_and_orthogonal():
    Q = random_orthogonal(20, np.random.default_rng(0))
    U = Q[:, :7]
    c = principal_cosines(U, U)
    assert c.shape == (7,)
    assert np.allclose(c, 1.0)
    # The same subspace in another basis.
    R = random_orthogonal(7, np.random.default_rng(1))
    assert np.allclose(principal_cosines(U, U @ R), 1.0)
    assert np.allclose(principal_cosines(U, Q[:, 7:14]), 0.0, atol=1e-12)


def test_compare_operators_identical_and_null():
    A = random_involution(40, 22, seed=0)
    c = compare_operators(A, A)
    assert c["mean_cos_even"] == pytest.approx(1.0)
    assert c["mean_cos_odd"] == pytest.approx(1.0)
    assert c["trace_agreement"] == pytest.approx(1.0)
    assert c["overlap_even"] == pytest.approx(1.0)
    assert (c["d_plus_ref"], c["d_minus_ref"]) == (22, 18)
    n = compare_operators(A, random_involution(40, 22, seed=1))
    assert n["trace_agreement"] < 0.5 and n["overlap_even"] < 0.9
    # -A swaps the eigenspaces: no agreement at all.
    assert compare_operators(A, -A)["trace_agreement"] == pytest.approx(-1.0)


def test_clean_involution_from_float32():
    A = random_involution(30, 17, seed=2)
    noisy = (A + 1e-5 * np.random.default_rng(0).standard_normal(A.shape)).astype(np.float32)
    B, info = clean_involution(noisy)
    assert np.allclose(B @ B, np.eye(30), atol=1e-10)
    assert np.allclose(B, B.T)
    assert info["d_plus"] == 17 and info["max_eig_dev"] < 1e-3
    assert compare_operators(A, B)["trace_agreement"] == pytest.approx(1.0, abs=1e-6)
    P, M = eigenspaces(B)
    assert P.shape[1] == 17 and M.shape[1] == 13


def test_random_involution_fixing_ones():
    A = random_involution_fixing_ones(16, 9, seed=0)
    assert np.allclose(A @ A, np.eye(16), atol=1e-12)
    assert np.allclose(A @ np.ones(16), np.ones(16), atol=1e-12)
    assert int(round(np.trace(A))) == 9 - 7


# ----------------------------------------------------------------------------------------------
# D2 on synthetic planted twins
# ----------------------------------------------------------------------------------------------
def planted_rows(d=32, h=120, n_pairs=40, noise=0.05, seed=0):
    """fc1 rows with w_j = A w_i for twin pairs and A w = w for self-twins, plus noise."""
    rng = np.random.default_rng(seed)
    A = random_involution(d, 18, seed=seed + 1)
    perm = random_pairing(h, n_pairs, seed=seed + 2)
    W = np.zeros((h, d))
    for i in range(h):
        j = perm[i]
        if j == i:
            v = rng.standard_normal(d)
            W[i] = 0.5 * (v + A @ v)
        elif i < j:
            W[i] = rng.standard_normal(d)
            W[j] = A @ W[i]
    W += noise * rng.standard_normal((h, d))
    return W, A, perm


def test_d2_recovers_planted_pairs():
    W, A, perm = planted_rows()
    S = twin_scores(torch.as_tensor(W), A)
    assert np.allclose(S, S.T)
    res = match_twins([S], top_k=10, workers=1)[0]
    ag = pairing_agreement(perm, res.perm)
    assert ag["recovery"] == 1.0 and ag["precision"] == 1.0 and ag["self_kept"] == 1.0
    assert ag["n_pairs_ref"] == 40


def test_d2_with_wrong_operator_recovers_little():
    W, A, perm = planted_rows()
    S = twin_scores(W, random_involution(A.shape[0], 18, seed=99))
    res = match_twins([S], top_k=10, workers=1)[0]
    assert pairing_agreement(perm, res.perm)["recovery"] < 0.2


def test_pairing_agreement_counts():
    ref = np.array([1, 0, 3, 2, 4, 5])        # pairs (0,1), (2,3); self 4, 5
    test = np.array([1, 0, 2, 4, 3, 5])       # pairs (0,1), (3,4); self 2, 5
    ag = pairing_agreement(ref, test)
    assert ag["n_common"] == 1 and ag["recovery"] == 0.5 and ag["precision"] == 0.5
    assert ag["self_kept"] == 0.5
    with pytest.raises(ValueError):
        pairing_agreement(ref, np.array([1, 2, 0, 3, 4, 5]))  # not an involution


# ----------------------------------------------------------------------------------------------
# D1 and D3 on a tiny model with a planted mirror symmetry
# ----------------------------------------------------------------------------------------------
@pytest.fixture
def planted_tiny():
    from mirror.adapters import TimmTinyAdapter

    ad = TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float64)
    A = random_involution_fixing_ones(ad.embed_dim, 36, seed=3)
    perm = plant_mirror_symmetry(ad, A, seed=4, n_pairs=80)
    return ad, A, perm


def test_planted_model_is_equivariant_in_the_data_path(planted_tiny):
    """The plant follows the data path's mirror convention (mirror.data.mirror, unflip_tokens)."""
    from mirror.data import mirror
    from mirror.stats import unflip_tokens

    ad, A, _ = planted_tiny
    x = torch.randn(2, 3, 56, 56, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    J = lambda t: t - t.mean(-1, keepdim=True)  # noqa: E731
    At = torch.as_tensor(A)
    for f in (ad.layer0_tokens, lambda z: ad.blocks[0](ad.layer0_tokens(z))):
        t = f(x)
        tm = unflip_tokens(f(mirror(x)), ad.grid_hw, ad.n_prefix_tokens)
        assert float((J(tm) - J(t) @ At).abs().max()) < 1e-10 * float(t.abs().max())


def test_d1_recovers_planted_A0(planted_tiny):
    ad, A, _ = planted_tiny
    parts = layer0_parts(ad)
    assert parts["affine_rel_err"] < 1e-12
    r = fit_layer0(parts)  # primary: sigma^2 = 1
    c = compare_operators(A, r["A"])
    assert c["mean_cos_even"] == pytest.approx(1.0, abs=1e-9)
    assert c["mean_cos_odd"] == pytest.approx(1.0, abs=1e-9)
    assert c["trace_agreement"] == pytest.approx(1.0, abs=1e-9)
    # Exact symmetry: the weighting does not matter, nor does dropping bias and prefix tokens.
    for kw in ({"rho": 0.25}, {"rho": 0.5}, {"rho": 0.9}, {"sigma2": 1.0, "terms": "position"}):
        assert compare_operators(A, fit_layer0(parts, **kw)["A"])["trace_agreement"] == pytest.approx(1.0, abs=1e-9)
    # The weighting: sigma^2 = 1 and rho are two views of the same mixture.
    m = layer0_moments(parts, sigma2=1.0)
    m2 = layer0_moments(parts, rho=m["rho"])
    assert m2["sigma2"] == pytest.approx(1.0) and np.allclose(m["C"], m2["C"])


def test_d1_wrong_flip_axis_fails(planted_tiny):
    """A vertical kernel flip must not recover the planted operator (the test has teeth)."""
    ad, A, _ = planted_tiny
    parts = layer0_parts(ad)
    wt = ad.model.patch_embed.proj.weight.detach().to(torch.float64)
    Wv = wt.flip(-2).reshape(wt.shape[0], -1).numpy()
    bad = dict(parts, Wf=Wv - Wv.mean(axis=0, keepdims=True))
    assert compare_operators(A, fit_layer0(bad)["A"])["trace_agreement"] < 0.9


def test_d1_matches_data_driven_A0(planted_tiny):
    from mirror.data import make_loader, synthetic_dataset
    from mirror.stats import collect_fit_stats

    ad, A, perm = planted_tiny
    images, labels = synthetic_dataset(2, 8, 56, seed=1)
    fs = collect_fit_stats(ad, make_loader(images, labels, np.arange(len(labels)), 8), torch.device("cpu"))
    A0_data = fit_involution(fs.residual.numpy()["C"][0])[0]
    A0_w = fit_layer0(layer0_parts(ad))["A"]
    assert compare_operators(A0_data, A0_w)["trace_agreement"] == pytest.approx(1.0, abs=1e-8)
    # D2 on the real folded fc1 of the planted block, with the data-free operator.
    S = twin_scores(fold_mlp(ad.blocks[0])["W1"], A0_w)
    res = match_twins([S], top_k=10, workers=1)[0]
    assert pairing_agreement(perm, res.perm)["recovery"] == 1.0


def test_act_moments_gelu_against_monte_carlo():
    mean = np.array([-1.0, 0.0, 0.5, 2.0])
    var = np.array([0.25, 1.0, 2.0, 0.5])
    nu, q = act_moments(mean, var)
    g = torch.Generator().manual_seed(0)
    z = torch.as_tensor(mean)[:, None] + torch.as_tensor(var).sqrt()[:, None] * torch.randn(4, 2_000_000, generator=g, dtype=torch.float64)
    u = torch.nn.functional.gelu(z)
    assert np.allclose(nu, u.mean(1).numpy(), atol=5e-3)
    assert np.allclose(q, u.var(1).numpy(), atol=5e-3)


def test_d3_recovers_planted_A1(planted_tiny):
    ad, A, perm = planted_tiny
    state0 = layer0_moments(layer0_parts(ad))
    for perms in ([perm], None):  # given pairing, and pairing from the weights (D2)
        ch = propagate_chain(ad.blocks, state0, 1, perms=perms)
        assert compare_operators(A, ch["A"][1])["trace_agreement"] == pytest.approx(1.0, abs=1e-9)
        assert pairing_agreement(perm, ch["perms"][0])["recovery"] == 1.0


# ----------------------------------------------------------------------------------------------
# Script
# ----------------------------------------------------------------------------------------------
def test_script_tiny(tmp_path):
    spec = importlib.util.spec_from_file_location("d_datafree_cli", ROOT / "scripts" / "d_datafree.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    R = mod.main(["--tiny", "--workers", "1", "--d2-blocks", "1,2", "--d3-blocks", "2", "--out", str(tmp_path)])
    for f in ("d_datafree.json", "D_DATAFREE_REPORT.md", "d_operators.npz", "tiny_operators.npz"):
        assert (tmp_path / f).exists(), f
    js = json.loads((tmp_path / "d_datafree.json").read_text(encoding="utf-8"))
    p5 = js["p5"]
    assert p5["part1"]["value"] == pytest.approx(1.0, abs=1e-9)
    assert p5["part2"]["value"] == 1.0 and p5["part2_strict"]["value"] == 1.0
    assert p5["pass"] and p5["pass_strict"] and p5["gate_G4"] == "pass"
    assert p5["thresholds"] == {"mean_cos_even": 0.8, "pair_recovery": 0.7}
    assert js["provenance"]["seed"] == 0 and js["provenance"]["git"]["git_hash"]
    assert js["planted"]["A0w_vs_planted"]["trace_agreement"] == pytest.approx(1.0, abs=1e-9)
    assert [r["layer"] for r in js["d3"]["rows"]].count(2) >= 5
    assert R["d1"]["rows"][0]["used_for_p5"] and R["d1"]["rows"][0]["sigma2"] == 1.0
    text = (tmp_path / "D_DATAFREE_REPORT.md").read_text(encoding="utf-8")
    assert "TINY TEST" in text and chr(0x2014) not in text  # no em-dashes in reports
