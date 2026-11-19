"""A02 metrics (mirror/stability.py) and scripts (scripts/fp_structure_fit.py, scripts/fp_a02_stability.py): known
operators (identical, orthogonal odd subspaces, a rotation of the odd subspace with a closed form), constructed
pairings and head maps, the gates, the sidecar and the reproduction check, and both dry runs (CPU, tiny models)."""
import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mirror import fit_weightspace as FW
from mirror import stability as STB
from mirror.attention import HeadMaps
from mirror.data import make_loader, synthetic_dataset
from mirror.fit import random_involution, random_orthogonal

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_fpst_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def involution_from(Q, odd):
    s = np.ones(Q.shape[0])
    s[list(odd)] = -1.0
    return (Q * s) @ Q.T


def sym_from(Q, lam):
    return (Q * np.asarray(lam, dtype=np.float64)) @ Q.T


# ---------------------------------------------------------------- operators
def test_identical_operators_have_distance_0_and_agreement_1():
    Q = random_orthogonal(12, np.random.default_rng(0))
    A = involution_from(Q, range(4))
    S = sym_from(Q, [-4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8])
    assert STB.signature(A) == (8, 4)
    assert STB.operator_distance(A, A) < 1e-15
    oa = STB.odd_agreement(A, A, S, S)
    for key in ("rho_ab", "rho_ba", "weighted", "unweighted_cos2"):
        assert abs(oa[key] - 1.0) < 1e-12, key
    assert np.allclose(STB.principal_cosines(A, A), 1.0)
    assert abs(STB.operator_distance(A, -A) - 2.0) < 1e-12                 # A_b = -A_a
    assert STB.d_minus_within(68, 58) is False and STB.d_minus_within(66, 58) is True   # |d - d0| <= 0.15 d0


def test_orthogonal_odd_subspaces_give_0():
    d = 12
    Q = random_orthogonal(d, np.random.default_rng(1))
    A_a, A_b = involution_from(Q, [0, 1, 2]), involution_from(Q, [3, 4])
    S_a = sym_from(Q, [-3, -2, -1] + [1.0] * 9)
    S_b = sym_from(Q, [1, 1, 1, -5, -4] + [2.0] * 7)
    oa = STB.odd_agreement(A_a, A_b, S_a, S_b)
    assert abs(oa["rho_ab"]) < 1e-12 and abs(oa["rho_ba"]) < 1e-12 and abs(oa["weighted"]) < 1e-12
    assert abs(oa["unweighted_cos2"]) < 1e-12
    assert abs(STB.operator_distance(A_a, A_b) - 2 * (3 + 2) / d) < 1e-12   # 2 (d-_a + d-_b) / d
    assert STB.rho(np.eye(d), S_a, A_b) != STB.rho(np.eye(d), S_a, A_b)    # no odd subspace: nan


@pytest.mark.parametrize("theta", [0.0, 0.3, math.pi / 4, 1.2, math.pi / 2])
def test_rotation_of_the_odd_subspace_has_the_closed_form(theta):
    """a's odd subspace span(q0, q1, q2) with energies |lam| = 3, 2, 1; b rotates q0 by theta towards an even
    direction q5. Then rho(a->b) = (3 cos^2 + 2 + 1) / 6, unweighted = (cos^2 + 2) / 3, distance = 4 sin^2 / d."""
    d = 10
    Q = random_orthogonal(d, np.random.default_rng(2))
    c, s = math.cos(theta), math.sin(theta)
    Qb = Q.copy()
    Qb[:, 0] = c * Q[:, 0] + s * Q[:, 5]
    Qb[:, 5] = -s * Q[:, 0] + c * Q[:, 5]
    A_a, A_b = involution_from(Q, [0, 1, 2]), involution_from(Qb, [0, 1, 2])
    S_a = sym_from(Q, [-3, -2, -1] + [1.0] * 7)
    S_b = sym_from(Qb, [-5, -1, -1] + [4.0] * 7)
    oa = STB.odd_agreement(A_a, A_b, S_a, S_b)
    assert abs(oa["rho_ab"] - (3 * c * c + 3) / 6) < 1e-12
    assert abs(oa["rho_ba"] - (5 * c * c + 2) / 7) < 1e-12
    assert abs(oa["weighted"] - 0.5 * ((3 * c * c + 3) / 6 + (5 * c * c + 2) / 7)) < 1e-12
    assert abs(oa["unweighted_cos2"] - (c * c + 2) / 3) < 1e-12
    assert abs(STB.operator_distance(A_a, A_b) - 4 * s * s / d) < 1e-12
    # One odd direction: both measures are cos^2 theta.
    one = STB.odd_agreement(involution_from(Q, [0]), involution_from(Qb, [0]), sym_from(Q, [-2] + [1.0] * 9))
    assert abs(one["rho_ab"] - c * c) < 1e-12 and abs(one["unweighted_cos2"] - c * c) < 1e-12
    assert one["rho_ba"] is None and one["weighted"] is None


def test_energy_weighting_matters():
    """Rotating the low-energy odd direction costs less than rotating the high-energy one."""
    d = 8
    Q = random_orthogonal(d, np.random.default_rng(3))
    S_a = sym_from(Q, [-10, -1] + [1.0] * 6)
    A_a = involution_from(Q, [0, 1])
    out = {}
    for which in (0, 1):
        Qb = Q.copy()
        Qb[:, which], Qb[:, 4] = Q[:, 4], Q[:, which]                      # swap an odd direction for an even one
        out[which] = STB.rho(A_a, S_a, involution_from(Qb, [0, 1]))
    assert abs(out[0] - 1 / 11) < 1e-12 and abs(out[1] - 10 / 11) < 1e-12


# ---------------------------------------------------------------- twins and head maps
def _perm(h, pairs):
    return FW.perm_from_pairs(h, pairs)


def _scores(h, pair_scores, self_score=1.0):
    s = np.full(h, self_score)
    for (i, j), v in pair_scores.items():
        s[i] = s[j] = v
    return s


def test_twin_agreement_on_constructed_pairings():
    h = 10
    pa = _perm(h, [(0, 1), (2, 3), (4, 5), (6, 7)])
    sa = _scores(h, {(0, 1): 0.9, (2, 3): 0.6, (4, 5): 0.4, (6, 7): 0.55})
    pb = _perm(h, [(0, 1), (2, 5), (3, 4), (6, 7)])
    sb = _scores(h, {(0, 1): 0.8, (2, 5): 0.7, (3, 4): 0.2, (6, 7): 0.3})
    assert STB.strong_pairs(pa, sa) == {(0, 1), (2, 3), (6, 7)} and STB.strong_pairs(pb, sb) == {(0, 1), (2, 5)}
    # Block 0 as above; block 1 identical in both fits (all strong); block 2 differs but is outside the blocks used.
    same = _perm(h, [(0, 9), (1, 8)])
    perms_a, perms_b = [pa, same, pa], [pb, same, _perm(h, [])]
    sc_a, sc_b = [sa, _scores(h, {(0, 9): 0.7, (1, 8): 0.9}), sa], [sb, _scores(h, {(0, 9): 0.7, (1, 8): 0.9}), sb]
    t = STB.twin_agreement(perms_a, sc_a, perms_b, sc_b, blocks=[0, 1])
    assert t["n_strong_a"] == 5 and t["n_strong_b"] == 4
    assert abs(t["a_to_b"] - 4 / 5) < 1e-12 and abs(t["b_to_a"] - 3 / 4) < 1e-12      # pooled over blocks 0 and 1
    assert abs(t["mean"] - 0.5 * (4 / 5 + 3 / 4)) < 1e-12
    assert abs(t["jaccard"] - 3 / 6) < 1e-12                                            # {(0,1), 2 of block 1} / 6
    b0 = t["per_block"][0]
    assert abs(b0["a_to_b"] - 2 / 3) < 1e-12 and abs(b0["b_to_a"] - 1 / 2) < 1e-12
    one_way = STB.twin_agreement(perms_a, sc_a, perms_b, None, blocks=[0])
    assert abs(one_way["a_to_b"] - 2 / 3) < 1e-12 and one_way["mean"] is None
    assert np.isnan(STB.twin_agreement([_perm(h, [])], [sa], [pb], [sb], blocks=[0])["a_to_b"])


def _self_maps(rng, H, dh):
    B = np.stack([random_involution(dh, 1 + h % dh, int(rng.integers(1 << 30))) for h in range(H)])
    C = np.stack([random_involution(dh, 1 + (h + 1) % dh, int(rng.integers(1 << 30))) for h in range(H)])
    return B, C


def test_head_agreement_on_constructed_maps():
    rng = np.random.default_rng(4)
    H, dh = 4, 3
    B, C = _self_maps(rng, H, dh)
    R = random_orthogonal(dh, rng)
    Ba, Ca = B.copy(), C.copy()
    Ba[0], Ba[1], Ca[0], Ca[1] = R, R.T, R, R.T                    # twin pair (0, 1), self-heads 2 and 3
    a0 = HeadMaps(sigma=np.array([1, 0, 2, 3]), B=Ba, C=Ca)
    Bb = Ba.copy()
    Bb[2] = -Ba[2]                                                  # same assignment, B of head 2 flipped
    b0 = HeadMaps(sigma=np.array([1, 0, 2, 3]), B=Bb, C=Ca.copy())
    a1 = HeadMaps(sigma=np.arange(H), B=B, C=C)                     # all self
    b1 = HeadMaps(sigma=np.array([1, 0, 2, 3]), B=Ba, C=Ca)         # heads 0 and 1 swapped instead
    for hm in (a0, b0, a1, b1):
        hm.check()
    out = STB.head_agreement([a0, a1], [b0, b1])
    assert out["n_heads"] == 8 and out["n_same"] == 6 and abs(out["share_same"] - 0.75) < 1e-12
    assert abs(out["B_rel_max"] - 2.0) < 1e-12 and abs(out["B_rel_mean"] - 2.0 / 6) < 1e-12
    assert out["C_rel_max"] < 1e-12
    assert STB.head_agreement([a0], [a0])["share_same"] == 1.0


def test_a02_gates():
    fits = [{"name": "fit0", "group": "imagenet", "d_minus": 68}, {"name": "fit1", "group": "imagenet", "d_minus": 75},
            {"name": "fit2", "group": "imagenet", "d_minus": 79}, {"name": "coco", "group": "coco", "d_minus": 10}]
    pairs = [{"a": "fit0", "b": "fit1", "weighted_agreement": 0.9, "twin_mean": 0.8, "head_share_same": 0.95},
             {"a": "fit0", "b": "fit2", "weighted_agreement": 0.81, "twin_mean": 0.69, "head_share_same": 0.9},
             {"a": "fit1", "b": "fit2", "weighted_agreement": 0.85, "twin_mean": 0.75, "head_share_same": 0.97},
             {"a": "fit0", "b": "coco", "weighted_agreement": 0.1, "twin_mean": 0.1, "head_share_same": 0.1}]
    g = STB.a02_gates(fits, pairs)
    assert g["A02-1"]["pass"] is False and g["A02-1"]["failing"] == ["fit2"]       # 79 > 68 * 1.15 = 78.2
    assert g["A02-2"]["pass"] is True and abs(g["A02-2"]["min"] - 0.81) < 1e-12     # COCO pairs do not count
    assert g["A02-3"]["pass"] is False and g["A02-3"]["failing"] == ["fit0-fit2"]
    assert g["A02-4"]["pass"] is True                                               # 0.90 is enough
    pairs[0]["twin_mean"] = None
    assert STB.a02_gates(fits, pairs)["A02-3"]["pass"] is None
    assert STB.a02_gates(fits[:1], [])["A02-1"]["pass"] is None


# ---------------------------------------------------------------- sidecar, reproduction check
@pytest.fixture(scope="module")
def tiny():
    return FW.dry_run_adapter(seed=0)


@pytest.fixture(scope="module")
def loader():
    images, labels = synthetic_dataset(4, 4, 56, seed=5)
    return make_loader(images, labels, np.arange(len(labels)), 8)


@pytest.fixture(scope="module")
def fitted(tiny, loader):
    return STB.fit_with_sidecar(tiny, loader, "cpu", model_name="tiny", role="fitX", fit_seed=7, top_k=3, workers=1)


def test_sidecar_contents_and_roundtrip(tiny, loader, fitted, tmp_path):
    from mirror.variants import collect_mirror_scores

    st, sc = fitted
    L, d, h = st.depth, st.d, st.h
    assert sc["perms"].shape == (L, h) and sc["twin_score"].shape == (L, h) and sc["S_span"].shape == (d, d)
    assert sc["S_eig"].shape == (d,) and sc["layer_eig"].shape == (L + 1, d) and sc["energies"].shape == (L + 1,)
    assert sc["d_minus_layers"].shape == (L + 1,) and list(sc["d_minus_span"]) == [STB.signature(st.A_star[k])[1]
                                                                                   for k in range(1, L + 1)]
    assert sc["meta"]["A_from_S_matches"] and st.meta["pass1"].startswith("variants.fit_structure")
    # twin_score is the symmetrised mirror correlation S[i, perm[i]] (recomputed with an independent pass).
    S = collect_mirror_scores(tiny, loader, "cpu", range(L))
    for b in range(L):
        p = np.asarray(st.perms[b])
        assert np.abs(sc["twin_score"][b] - S[b][np.arange(h), p]).max() < 1e-10
    # The sign rule of S_span is A*_L, and the energy weights are |eigenvalues| of S.
    lam = np.linalg.eigvalsh(sc["S_span"])
    assert np.allclose(lam, sc["S_eig"]) and STB.signature(st.A_star[L])[1] == int((lam < -1e-10 * abs(lam).max()).sum())
    back = STB.load_sidecar(STB.save_sidecar(tmp_path / "sc.npz", sc))
    assert back["role"] == "fitX" and back["fit_seed"] == 7 and np.array_equal(back["S_span"], sc["S_span"])
    assert back["meta"]["A_from_S_matches"] is True
    # Every metric of a fit against itself.
    c = STB.compare_fits(st, sc, st, sc, threshold=-1.0)                # every matched pair counts as strong
    assert c["operator_distance"] == 0.0 and abs(c["weighted_agreement"] - 1) < 1e-12
    assert abs(c["unweighted_cos2"] - 1) < 1e-12 and c["head_share_same"] == 1.0 and c["B_rel_max"] == 0.0
    assert c["twin_mean"] == 1.0 and c["twin_jaccard"] == 1.0 and max(c["layer_operator_distance"]) == 0.0
    with pytest.raises(ValueError):                                     # a sidecar of another fit is refused
        other = dict(sc)
        other["perms"] = np.roll(sc["perms"], 1, axis=1)
        STB.compare_fits(st, other, st, sc)


def test_compare_structures_detects_reproduction(tiny, loader, fitted):
    st, _ = fitted
    again, _ = STB.fit_with_sidecar(tiny, loader, "cpu", model_name="tiny", role="fitX", fit_seed=7, top_k=3, workers=1)
    rep = STB.compare_structures(again, st)
    assert rep["reproduces_exactly"] and rep["perm_entries_differing"] == 0 and rep["A_star_max_abs_diff"] == 0.0
    p = np.asarray(again.perms[0]).copy()
    i = int(np.flatnonzero(p != np.arange(len(p)))[0])
    j = int(p[i])
    p[i], p[j] = i, j                                                   # dissolve one pair
    changed = dataclasses.replace(again, perms=[p] + list(again.perms[1:]))
    rep2 = STB.compare_structures(changed, st)
    assert not rep2["reproduces_exactly"] and rep2["perm_entries_differing"] == 2 and rep2["A_star_equal"]


# ---------------------------------------------------------------- scripts: dry runs
def test_structure_fit_cli_dry_run_reference_and_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("MIRROR_FINAL_ALLOW_NO_PREREG", "1")
    mod = load_script("fp_structure_fit")
    a = mod.main(["--dry-run", "--role", "fit0", "--out", str(tmp_path / "a")])
    for f in ("tiny_fit0.npz", "tiny_fit0_sidecar.npz", "tiny_fit0_fit.json", "tiny_fit0_meta.json"):
        assert (tmp_path / "a" / f).is_file(), f
    assert a["reproduce"] is None and a["A_from_S_matches"] and a["fit_seed"] == 0
    ref = tmp_path / "a" / "tiny_fit0.npz"
    b = mod.main(["--dry-run", "--role", "fit0", "--out", str(tmp_path / "b"), "--reference", str(ref)])
    assert b["reproduce"]["reproduces_exactly"] is True                 # deterministic CPU refit on the same images
    t0 = ref.stat().st_mtime_ns
    again = mod.main(["--dry-run", "--role", "fit0", "--out", str(tmp_path / "a")])
    assert ref.stat().st_mtime_ns == t0 and again["d_minus"] == a["d_minus"]
    with pytest.raises(SystemExit):
        mod.main(["--dry-run", "--role", "coco", "--out", str(tmp_path / "c")])        # needs --fit-seed


def test_a02_cli_dry_run(tmp_path, monkeypatch):
    monkeypatch.setenv("MIRROR_FINAL_ALLOW_NO_PREREG", "1")
    mod = load_script("fp_a02_stability")
    out = tmp_path / "a02"
    res = mod.main(["--dry-run", "--out", str(out)])
    for f in ("stability.csv", "pairs.csv", "spectra.npz", "summary.md", "metrics.json", "meta.json",
              "knn/fit0.json", "knn/coco.npz"):
        assert (out / f).is_file(), f
    fits = pd.read_csv(out / "stability.csv")
    pairs = pd.read_csv(out / "pairs.csv")
    assert list(fits["name"]) == ["fit0", "fit1", "fit2", "fit3", "coco"] and len(pairs) == 10
    assert int(pairs["both_imagenet"].sum()) == 6
    assert ((pairs["operator_distance"] >= 0) & (pairs["unweighted_cos2"] <= 1 + 1e-9)).all()
    assert set(res["gates"]) >= {"A02-1", "A02-2", "A02-3", "A02-4"}
    with np.load(out / "spectra.npz") as z:
        assert z["S_eig"].shape[0] == 5 and z["layer_eig"].shape[1] == 4
    m = json.loads((out / "metrics.json").read_text())
    assert m["experiment"] == "A02" and len(m["twin_agreement_per_block"]) == 10
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert chr(0x2014) not in text and "A02-2" in text
    knn0 = json.loads((out / "knn" / "fit0.json").read_text())
    res2 = mod.main(["--dry-run", "--out", str(out)])                   # resume: k-NN reused, same metrics
    assert json.loads((out / "knn" / "fit0.json").read_text()) == knn0
    assert [p["weighted_agreement"] for p in res2["pairs"]] == [p["weighted_agreement"] for p in res["pairs"]]
