"""A01 weight-space operator (mirror/fit_weightspace.py, scripts/fp_a01_weightspace.py): CPU tests on tiny random
ViTs, float64 where exactness is tested.

1. the closed form beats every symmetric orthogonal involution (random draws and every eigen-sign pattern), in both
   modes (matched: among involutions with the given d-), with and without the A 1 = 1 constraint;
2. the objective equals the summed real projection damage of build_vexact on tiny folded models (plain timm, timm
   with registers and a swapped register pairing, the DINOv2 mock with materialised positions and 4 registers);
3. the alternation is monotone and matched mode keeps d- and every block's pair count;
4. the saved A*_w structure loads with load_span_structure and build_mirror_variant builds an exactly equivariant
   model from it (stream error at most 1e-10 in float64), also for a span k < L;
plus the script's dry run and resume."""
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mirror import fit_weightspace as FW
from mirror import mirror_family as MF
from mirror.adapters import Dinov2Adapter, ViTAdapter
from mirror.attention import attn_dims, fold_attn, gauge_twin_heads
from mirror.data import make_loader, synthetic_dataset
from mirror.fit import fit_involution, random_involution
from mirror.m1 import (build_vexact, feature_invariance, random_head_structure, random_involution_fixing_ones,
                       stream_equivariance)
from mirror.matching import random_pairing

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_fpws_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def probe(n=3, seed=5, size=56):
    return torch.randn(n, 3, size, size, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


def is_involution(A, tol=1e-10):
    A = np.asarray(A)
    return np.abs(A - A.T).max() < tol and np.abs(A @ A - np.eye(len(A))).max() < tol


# ---------------------------------------------------------------- fixtures (module scope)
@pytest.fixture(scope="module")
def reg64():
    """Tiny timm ViT with 2 registers (d 32, 3 blocks, 4 heads, h 128), float64."""
    return FW.dry_run_adapter(seed=0, dtype=torch.float64)


@pytest.fixture(scope="module")
def fit_loader():
    images, labels = synthetic_dataset(4, 4, 56, seed=3)
    return make_loader(images, labels, np.arange(len(labels)), 8)


@pytest.fixture(scope="module")
def fitted64(reg64, fit_loader):
    """A structure fitted by the canonical pipeline on synthetic images."""
    return MF.fit_span_structure(reg64, fit_loader, "cpu", model_name="reg64", top_k=3, workers=1, pairings=False)


def synthetic_structure(ad, seed=0, reg_perm=(1, 0)):
    """A SpanStructure with random operators, pairings and head maps (no activations needed)."""
    d, h, L = ad.embed_dim, ad.mlp_hidden, ad.depth
    H, dh = attn_dims(ad.model.blocks[0])
    perms = [random_pairing(h, 20 + 4 * b, seed=seed + b) for b in range(L)]
    heads = [random_head_structure(H, dh, b % 2, seed=seed + 20 + b) for b in range(L)]
    A_star = {k: random_involution_fixing_ones(d, d - 6 - k, seed=seed + 100 + k) for k in range(1, L + 1)}
    return MF.SpanStructure(model="reg64", fit_seed=0,
                            A_layers=[random_involution_fixing_ones(d, d - 5, seed=seed + l) for l in range(L + 1)],
                            perms=perms, heads=heads, A_star=A_star,
                            reg_perm={k: np.asarray(reg_perm, dtype=np.int64) for k in range(1, L + 1)},
                            meta={"n_pairs": [FW.n_pairs_of(p) for p in perms]})


# ---------------------------------------------------------------- 1. closed form against brute force
def _sign_patterns(M, centre, n_minus=None):
    """Every involution diagonal in the eigenbasis of M (of Pc M Pc on the complement of 1 when centred)."""
    d = M.shape[0]
    Q = FW.perp_basis(d) if centre else np.eye(d)
    _, U = np.linalg.eigh(Q.T @ M @ Q)
    V = Q @ U
    uu = np.full((d, d), 1.0 / d) if centre else 0.0
    for s in itertools.product([-1.0, 1.0], repeat=V.shape[1]):
        if n_minus is not None and int((np.asarray(s) < 0).sum()) != n_minus:
            continue
        yield (V * np.asarray(s)) @ V.T + uu


def _random_candidates(d, centre, rng, n=150, n_minus=None):
    for _ in range(n):
        dm = int(rng.integers(0, d if centre else d + 1)) if n_minus is None else int(n_minus)
        seed = int(rng.integers(1 << 30))
        yield random_involution_fixing_ones(d, d - dm, seed) if centre else random_involution(d, d - dm, seed)


@pytest.mark.parametrize("centre", [True, False])
@pytest.mark.parametrize("d", [2, 3, 4, 5, 6])
def test_closed_form_free_beats_every_involution(d, centre):
    rng = np.random.default_rng(100 * d + int(centre))
    for _ in range(2):
        X = rng.standard_normal((d, d))
        M = X + X.T + (0.3 * np.ones((d, d)) if centre else 0.0)
        A, info = FW.closed_form_operator(M, "free", centre=centre)
        assert is_involution(A)
        if centre:
            assert np.abs(A @ np.ones(d) - np.ones(d)).max() < 1e-12
        J = float(np.sum(A * M))
        assert abs(J - info["J"]) < 1e-12 and info["d_plus"] + info["d_minus"] == d
        best = max(float(np.sum(B * M)) for B in itertools.chain(_sign_patterns(M, centre),
                                                                   _random_candidates(d, centre, rng)))
        assert J >= best - 1e-10
        assert J <= best + 1e-10                         # it is one of the sign patterns


@pytest.mark.parametrize("centre", [True, False])
@pytest.mark.parametrize("d", [3, 4, 6])
def test_closed_form_matched_beats_every_involution_with_that_d_minus(d, centre):
    rng = np.random.default_rng(7 * d + int(centre))
    X = rng.standard_normal((d, d))
    M = X + X.T
    for m in range(0, d if centre else d + 1):
        A, info = FW.closed_form_operator(M, "matched", d_minus=m, centre=centre)
        assert is_involution(A) and info["d_minus"] == m and FW.d_minus_of(A) == m
        J = float(np.sum(A * M))
        cands = itertools.chain(_sign_patterns(M, centre, m), _random_candidates(d, centre, rng, 80, m))
        assert J >= max(float(np.sum(B * M)) for B in cands) - 1e-10
    with pytest.raises(ValueError):
        FW.closed_form_operator(M, "matched", d_minus=None)


def test_closed_form_agrees_with_the_activation_fit_rule():
    """Free mode on a centred matrix is fit.fit_involution's sign rule (the rule of the current A*)."""
    rng = np.random.default_rng(0)
    d = 12
    X = rng.standard_normal((d, d))
    M = X + X.T
    A, _ = FW.closed_form_operator(M, "free")
    B = fit_involution(FW.centre_matrix(M))[0]
    assert np.abs(A - B).max() < 1e-10


def test_repair_block_is_the_exact_optimum_on_small_blocks():
    rng = np.random.default_rng(1)
    for h in (6, 7):
        X = rng.standard_normal((h, h))
        G = X + X.T
        perm = FW.repair_block(G, top_k=h - 1)
        assert np.array_equal(perm[perm], np.arange(h))
        best = -np.inf
        for p in itertools.permutations(range(h)):            # every involution
            p = np.asarray(p)
            if np.array_equal(p[p], np.arange(h)):
                best = max(best, FW.pair_value(G, p))
        assert FW.pair_value(G, perm) >= best - 1e-6 * max(1.0, abs(best))
        for n in range(0, h // 2 + 1):                         # cut or fill to an exact pair count
            assert FW.n_pairs_of(FW.repair_block(G, n_pairs=n, top_k=h - 1)) == n


# ---------------------------------------------------------------- 2. objective = real projection damage
class _MockDinoAdapter(Dinov2Adapter):
    def __init__(self, model, img_size):
        ViTAdapter.__init__(self, model, img_size=img_size, patch_size=model.patch_size,
                            n_prefix_tokens=1 + model.num_register_tokens)


def _model(kind):
    if kind == "plain":
        from mirror.adapters import TimmTinyAdapter

        return TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float64), None
    if kind == "registers":
        return FW.dry_run_adapter(seed=1, dtype=torch.float64), [1, 0]
    from test_fold import make_mock                       # DINOv2 mock: 5 x 5 stored grid, 4 x 4 used (materialised)

    return _MockDinoAdapter(make_mock(num_register_tokens=4, img_size=70), 56), [1, 0, 2, 3]


def _sq(x):
    return float((x.double() ** 2).sum())


def _real_damage(ad, ref, conv, heads, k):
    out = {}
    for b in range(k):
        r, c = ref.blocks[b].mlp, conv.blocks[b].mlp
        out[f"mlp.{b}"] = _sq(c.fc1.weight - r.fc1.weight) + _sq(c.fc2.weight - r.fc2.weight)
        f0, _ = gauge_twin_heads(fold_attn(ref.blocks[b]), heads[b])      # build_vexact gauges twin heads
        f1 = fold_attn(conv.blocks[b])
        out[f"attn.{b}"] = sum(_sq(f1[key] - f0[key]) for key in ("Wq", "Wk", "Wv", "Wo"))
    p0, p1 = ref.patch_embed.proj, conv.patch_embed.proj
    out["patch"] = _sq(p1.weight - p0.weight) + _sq(p1.bias - p0.bias)
    H, W = ad.grid_hw
    P0, P1 = ref.pos_embed[0], conv.pos_embed[0]
    n_pre = P0.shape[0] - H * W
    out["pos"] = _sq(P1[n_pre:] - P0[n_pre:])
    out["cls"] = _sq(conv.cls_token - ref.cls_token) + (_sq(P1[0] - P0[0]) if n_pre >= 1 else 0.0)
    for name in ("register_tokens", "reg_token"):
        if isinstance(getattr(ref, name, None), torch.Tensor):
            out["reg"] = _sq(getattr(conv, name) - getattr(ref, name)) + (_sq(P1[1:n_pre] - P0[1:n_pre])
                                                                          if n_pre > 1 else 0.0)
    return out


@pytest.mark.parametrize("kind", ["plain", "registers", "dinov2_mock"])
def test_objective_equals_the_real_projection_damage(kind):
    ad, reg_perm = _model(kind)
    d, h, L = ad.embed_dim, ad.mlp_hidden, ad.depth
    H, dh = attn_dims(ad.model.blocks[0])
    A = random_involution_fixing_ones(d, d // 2 + 3, seed=11)
    perms = [random_pairing(h, h // 4 + b, seed=30 + b) for b in range(L)]
    heads = [random_head_structure(H, dh, 1 + b % 2, seed=40 + b) for b in range(L)]
    for k in (L, 2):
        terms = FW.collect_terms(ad, k, reg_perm=reg_perm, heads=heads, attn_terms=True,
                                 weights={"cls": 0.5, "attn": 2.0})
        assert len(terms.mlp) == k and {t.kind for t in terms.fixed} >= {"patch", "pos", "cls", "attn"}
        assert ("reg" in {t.kind for t in terms.fixed}) == (reg_perm is not None)
        conv, _ = build_vexact(ad, A, list(range(k)), perms[:k], heads[:k], reg_perm=reg_perm, even_head=(k == L))
        real = _real_damage(ad, FW.folded_copy(ad), conv, heads, k)
        rows = FW.term_values(terms, A, perms[:k])
        assert {r["name"] for r in rows} == set(real)
        for r in rows:
            assert abs(r["damage"] - real[r["name"]]) <= 1e-10 * max(real[r["name"]], 1e-12), (r["name"], r, real)
            assert r["damage"] > 0
        # J = tr(A M) and damage = weighted sum of real breaking shares = (sum of weights - J) / 2
        J = FW.objective(terms, A, perms[:k])
        M = FW.objective_matrix(terms, perms[:k])
        assert abs(J - float(np.sum(A * M))) <= 1e-12 * max(1.0, abs(J))
        want = sum(r["weight"] * real[r["name"]] / r["energy"] for r in rows)
        assert abs(FW.damage_of(terms, A, perms[:k]) - want) <= 1e-12
        assert abs(0.5 * (terms.weight_total() - J) - want) <= 1e-12
        if reg_perm is not None:                     # the register term rebuilt for another pairing is the same
            other = FW.with_register_perm(FW.collect_terms(ad, k, reg_perm=None, heads=heads), reg_perm)
            Mo = next(t.M for t in other.fixed if t.kind == "reg")
            assert np.abs(Mo - next(t.M for t in terms.fixed if t.kind == "reg")).max() < 1e-12


def test_block_gram_is_the_pairing_form_of_the_block_term(reg64):
    terms = FW.collect_terms(reg64)
    d, h = terms.d, terms.h
    A = random_involution_fixing_ones(d, 20, seed=2)
    perm = random_pairing(h, 30, seed=3)
    G = FW.block_gram(terms, 1, A)
    assert abs(FW.pair_value(G, perm) - float(np.sum(A * FW.mlp_matrix(terms, 1, perm).numpy()))) < 1e-10


# ---------------------------------------------------------------- 3. alternation
@pytest.mark.parametrize("mode", ["matched", "free"])
def test_alternation_is_monotone_and_matched_keeps_counts(reg64, mode):
    st = synthetic_structure(reg64)
    L, d = reg64.depth, reg64.embed_dim
    fit = FW.fit_weightspace(reg64, st, mode, max_rounds=3, tol=0.0, top_k=4)
    seq = [fit.J_source]
    for r in fit.rounds:
        seq += [r["J_after_pairing"], r["J"]]
    seq = seq[1:]                                    # round 0's J_after_pairing is the source's J
    assert all(b >= a - 1e-12 * max(1.0, abs(a)) for a, b in zip(seq, seq[1:])), seq
    assert fit.J > fit.J_source and fit.damage < fit.damage_source
    assert abs(fit.damage - 0.5 * (FW.collect_terms(reg64).weight_total() - fit.J)) < 1e-9
    assert is_involution(fit.A) and np.abs(fit.A @ np.ones(d) - np.ones(d)).max() < 1e-10
    assert all(np.array_equal(p[p], np.arange(len(p))) for p in fit.perms)
    src_counts = [st.n_pairs(b) for b in range(L)]
    if mode == "matched":
        assert fit.d_minus == FW.d_minus_of(st.A_star[L])
        assert all(r["d_minus"] == fit.d_minus and r["n_pairs"] == src_counts for r in fit.rounds)
        assert [FW.n_pairs_of(p) for p in fit.perms] == src_counts
    assert len(fit.rounds) <= 4 and fit.rounds[1]["neurons_changed"] > 0     # the alternation did something
    # Re-pairing for the same operator from an already re-paired state changes nothing (nothing is accepted).
    terms = FW.collect_terms(reg64, reg_perm=st.reg_perm[L])
    n = [st.n_pairs(b) for b in range(L)] if mode == "matched" else None
    p1, _ = FW._repair_round(terms, fit.A, st.perms, n, 4, 1)
    p2, recs = FW._repair_round(terms, fit.A, p1, n, 4, 1)
    assert all(np.array_equal(a, b) for a, b in zip(p1, p2)) and not any(r["accepted"] for r in recs)


# ---------------------------------------------------------------- 4. output structure, exact equivariance
def test_saved_structure_builds_an_exactly_equivariant_model(reg64, fitted64, fit_loader, tmp_path):
    L = reg64.depth
    fit = FW.fit_weightspace(reg64, fitted64, "matched", top_k=3)
    reg, cos = FW.rematch_registers(reg64, fit_loader, "cpu", fit.A)
    assert reg.shape == (2,) and np.array_equal(reg[reg], np.arange(2)) and cos.shape == (2, 2)
    st = FW.weightspace_structure(fitted64, fit, reg, source_path="src.npz", source_sha256="abc123",
                                  register_source="test")
    back = MF.load_span_structure(MF.save_span_structure(tmp_path / "w.npz", st))
    assert np.array_equal(back.A_star[L], FW.exact_involution(fit.A))
    assert all(np.array_equal(back.A_star[k], fitted64.A_star[k]) for k in range(1, L))
    assert all(np.array_equal(a, b) for a, b in zip(back.perms, fit.perms))
    assert np.array_equal(back.reg_perm[L], reg)
    assert all(np.array_equal(x.sigma, y.sigma) and np.array_equal(x.B, y.B) for x, y in zip(back.heads, fitted64.heads))
    of = back.meta["operator_fit"]
    assert of["mode"] == "matched" and of["source_sha256"] == "abc123" and of["k"] == L
    assert of["d_minus"] == FW.d_minus_of(fitted64.A_star[L]) and len(of["J_rounds"]) == len(fit.rounds)
    assert back.meta["n_pairs"] == [fitted64.n_pairs(b) for b in range(L)]
    model, info = MF.build_mirror_variant(reg64, "mirror_exact", range(L), MF.COMPONENTS, back)
    x = probe(3, 7)
    rows = stream_equivariance(reg64, model, x, back.A_star[L], reg_perm=back.reg_perm[L])
    assert max(r["rel_max"] for r in rows) <= 1e-10, rows
    assert feature_invariance(reg64, model, x)["rel_max"] <= 1e-10
    _, info0 = MF.build_mirror_variant(reg64, "mirror_exact", range(L), MF.COMPONENTS, fitted64)
    assert info["params_total"] == info0["params_total"] and info["gap"] == 0       # the same size as the source


def test_partial_span_k_less_than_L(reg64, fitted64, fit_loader):
    L, k = reg64.depth, 2
    fit = FW.fit_weightspace(reg64, fitted64, "matched", k=k, top_k=3)
    assert fit.k == k and len(fit.perms) == k
    assert fit.d_minus == FW.d_minus_of(fitted64.A_star[k])
    assert [FW.n_pairs_of(p) for p in fit.perms] == [fitted64.n_pairs(b) for b in range(k)]
    reg, _ = FW.rematch_registers(reg64, fit_loader, "cpu", fit.A, k)
    st = FW.weightspace_structure(fitted64, fit, reg)
    for kk in range(1, L + 1):
        if kk != k:
            assert np.array_equal(st.A_star[kk], fitted64.A_star[kk])
            assert np.array_equal(st.reg_perm[kk], fitted64.reg_perm[kk])
    assert all(np.array_equal(st.perms[b], fitted64.perms[b]) for b in range(k, L))
    assert st.meta["operator_fit"]["k"] == k and st.meta["A_star"][str(k)]["d_minus"] == fit.d_minus
    model, info = MF.build_mirror_variant(reg64, "mirror_exact", range(k), MF.COMPONENTS, st)
    rows = stream_equivariance(reg64, model, probe(3, 8), st.A_star[k], reg_perm=st.reg_perm[k])
    assert max(r["rel_max"] for r in rows if r["layer"] <= k) <= 1e-10, rows
    assert min(r["rel_max"] for r in rows if r["layer"] > k) > 1e-6            # teeth: blocks after the span
    _, info0 = MF.build_mirror_variant(reg64, "mirror_exact", range(k), MF.COMPONENTS, fitted64)
    assert info["params_total"] == info0["params_total"]


# ---------------------------------------------------------------- script: dry run and resume
def test_cli_dry_run_and_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("MIRROR_FINAL_ALLOW_NO_PREREG", "1")
    mod = load_script("fp_a01_weightspace")
    out = tmp_path / "a01"
    res = mod.main(["--dry-run", "--out", str(out)])
    for f in ("operators.csv", "rounds.csv", "blocks.csv", "terms.csv", "summary.md", "metrics.json", "meta.json",
              "structure/tiny_wspace_matched.npz", "structure/tiny_wspace_free.npz",
              "structure/tiny_wspace_matched_objective.npz", "knn/current.json", "knn/wspace_matched.npz"):
        assert (out / f).is_file(), f
    ops = pd.read_csv(out / "operators.csv").set_index("operator")
    assert set(ops.index) == {"current", "wspace_matched", "wspace_free"}
    cur, mat = ops.loc["current"], ops.loc["wspace_matched"]
    assert mat["d_minus"] == cur["d_minus"] and mat["n_pairs_span"] == cur["n_pairs_span"]
    assert mat["params_model"] == cur["params_model"] and mat["damage"] < cur["damage"]
    assert 0 <= mat["unweighted_cos2"] <= 1 and 0 <= mat["rho_to_current"] <= 1
    st = MF.load_span_structure(out / "structure" / "tiny_wspace_matched.npz")
    assert st.meta["operator_fit"]["register_pairing"].startswith("re-matched")
    meta = json.loads((out / "meta.json").read_text())
    pc = meta["extra"]["prereg_check"]
    assert meta["experiment"] == "A01" and (pc["ok"] or pc["allowed_without_check"] is True)
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert chr(0x2014) not in text and "Zero-shot k-NN" in text
    # Resume: nothing refitted, identical numbers.
    t0 = (out / "structure" / "tiny_wspace_matched.npz").stat().st_mtime_ns
    res2 = mod.main(["--dry-run", "--out", str(out)])
    assert (out / "structure" / "tiny_wspace_matched.npz").stat().st_mtime_ns == t0
    assert np.allclose([r["J"] for r in res2["operators"]], [r["J"] for r in res["operators"]], rtol=1e-12, atol=0)
    assert res2["knn"]["wspace_matched"]["top1"] == res["knn"]["wspace_matched"]["top1"]
    # A shorter span writes its own structure.
    mod.main(["--dry-run", "--out", str(out), "--k", "2", "--mode", "matched", "--skip-knn"])
    st2 = MF.load_span_structure(out / "structure" / "tiny_wspace_matched_k2.npz")
    assert st2.meta["operator_fit"]["k"] == 2 and math.isclose(
        FW.d_minus_of(st2.A_star[2]), FW.d_minus_of(MF.load_span_structure(out / "structure" / "tiny_s0.npz").A_star[2]))
