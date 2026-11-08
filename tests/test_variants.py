"""Track A variants (mirror/variants.py) and the --tiny paths of scripts/stage1_ablate.py and
scripts/f4_rule.py. CPU only, no downloads."""
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from mirror import variants as V
from mirror.data import normalize_for
from mirror.fit import agreement, random_involution
from mirror.matching import match_neurons, random_pairing
from mirror.surgery import (apply_surgery_two, breaking_share, equivariant_param_count, fold_mlp,
                            perm_trace, project_mlp)

ROOT = Path(__file__).resolve().parents[1]


def make_struct(adapter, seed=0, common=None):
    """Structure for the tiny model: per-layer random involutions with A 1 = 1 and varied d+, matched
    pairings with varied counts, C_l = s_l A_l and G_l = s_l I (so each A_l is the exact fit of C_l)."""
    L, d, h = adapter.depth, adapter.embed_dim, adapter.mlp_hidden
    dps = [d // 2 + 3 * (l % 3) + 1 for l in range(L + 1)]
    A = [common if common is not None else V.random_operator(d, dps[l], seed=seed * 100 + l) for l in range(L + 1)]
    dps = [V.d_plus_of(a) for a in A]
    scale = [1.0 + 10.0 * l for l in range(L + 1)]
    C = np.stack([s * a for s, a in zip(scale, A)])
    G = np.stack([s * np.eye(d) for s in scale])
    perms = [random_pairing(h, h // 3 + 5 * b, seed=seed * 100 + 50 + b) for b in range(L)]
    return V.Structure(A_layers=A, d_plus=dps, d_minus=[d - x for x in dps], perms=perms, C=C, G=G, center=True)


def random_scores(h, seed=0):
    S = np.random.default_rng(seed).uniform(-1, 1, (h, h))
    return 0.5 * (S + S.T)


# ---------------------------------------------------------------- involutions
def test_clean_involution_keeps_counts():
    rng = np.random.default_rng(0)
    A = random_involution(20, 7, seed=3)
    noisy = (A + 1e-6 * rng.standard_normal(A.shape)).astype(np.float32)
    C, dp, dm = V.clean_involution(noisy)
    assert (dp, dm) == (7, 13)
    assert np.abs(C @ C - np.eye(20)).max() < 1e-12
    assert np.array_equal(C, C.T)
    assert np.abs(C - A).max() < 1e-4
    with pytest.raises(ValueError):
        V.clean_involution(np.diag(np.r_[np.ones(10), 0.5 * np.ones(10)]))


@pytest.mark.parametrize("fix_ones", [True, False])
def test_random_involutions_have_requested_d_plus(fix_ones):
    d = 32
    for d_plus in (1, 9, 20, 31):
        A = V.random_operator(d, d_plus, seed=d_plus, fix_ones=fix_ones)
        ev = np.linalg.eigvalsh(A)
        assert int((ev > 0).sum()) == d_plus and int((ev < 0).sum()) == d - d_plus
        assert np.abs(A @ A - np.eye(d)).max() < 1e-12
        assert V.d_plus_of(A) == d_plus
        if fix_ones:
            assert np.abs(A @ np.ones(d) - np.ones(d)).max() < 1e-12


def test_random_operator_variant_matches_d_plus_per_layer(tiny64):
    struct = make_struct(tiny64)
    for k in (1, 3):
        A_in, A_out, _ = V.operator_variant("random_operator", struct, range(k), seed=4)
        assert [V.d_plus_of(a) for a in A_in] == struct.d_plus[:k]
        assert [V.d_plus_of(a) for a in A_out] == struct.d_plus[1:k + 1]
        assert all(agreement(a, struct.A_layers[b]) < 0.9 for b, a in enumerate(A_in))


# ---------------------------------------------------------------- every builder runs
def test_every_variant_builds_and_runs(tiny64, images64):
    struct = make_struct(tiny64)
    h = struct.h
    scores = {b: random_scores(h, seed=b) for b in range(struct.depth)}
    shares = V.mlp_block_shares(tiny64.model, struct)
    f0 = tiny64.cls_feature(images64)
    generic = {}
    for ablation, names in (("A1", V.A1_VARIANTS), ("A2", V.A2_VARIANTS), ("A3", ("n10",)),
                            ("A4", V.A4_VARIANTS), ("A5", V.A5_VARIANTS)):
        for variant in names:
            for k in (1, 2):
                pre = None
                if variant == "generic":
                    if not generic:
                        W1 = [fold_mlp(tiny64.model.blocks[b])["W1"] for b in range(2)]
                        generic.update(zip(range(2), V.generic_pairings(W1, [struct.n_pairs(b) for b in range(2)], 3)))
                    pre = generic
                spec = V.build_variant(ablation, variant, struct, k, seed=1, model=tiny64.model, precomputed=pre,
                                       scores=scores, shares=shares, top_k=3)
                assert len(spec.blocks) == k
                assert 0 < spec.params_kept <= spec.params_dense == 2 * h * struct.d * k
                m = V.variant_copy(tiny64.model, spec)
                f = tiny64.cls_feature(images64, m)
                assert torch.isfinite(f).all(), (ablation, variant, k)
                if spec.method == "mirror" and spec.alpha > 0:
                    assert (f - f0).abs().max().item() > 1e-8, (ablation, variant, k)  # it changed something
    assert torch.equal(tiny64.cls_feature(images64), f0)  # the original model is untouched
    hung = V.build_variant("A2", "hungarian", struct, 2, scores=scores)
    assert hung.projection == "approx" and hung.params_exact is False
    assert V.build_variant("A2", "hungarian_cycle", struct, 2, scores=scores).params_exact is True
    assert V.build_variant("A4", "last", struct, 2, shares=shares).blocks == [struct.depth - 2, struct.depth - 1]
    low = V.build_variant("A4", "lowest_share", struct, 2, shares=shares).blocks
    assert low == sorted(np.argsort(shares)[:2].tolist())


def test_spec_signature_identifies_identical_models(tiny64):
    struct = make_struct(tiny64)
    sig = lambda ab, var, k=2, **kw: V.spec_signature(V.build_variant(ab, var, struct, k, seed=1, **kw))  # noqa: E731
    same = {sig("A1", "per_layer"), sig("A2", "mirror"), sig("A4", "first"), sig("A5", "alpha1.00")}
    assert len(same) == 1
    others = [sig("A1", "per_layer", k=1), sig("A5", "alpha0.50"), sig("A1", "random_structure"),
              sig("A1", "svd"), sig("A2", "self_twins"), sig("A1", "identity")]
    assert len(set(others) | same) == len(others) + 1
    assert sig("A1", "random_operator") == sig("A1", "random_operator")   # deterministic seeds


# ---------------------------------------------------------------- parameter counts
def test_matched_parameter_counts(tiny64):
    struct = make_struct(tiny64)
    h, d, k = struct.h, struct.d, 3
    expected = sum(equivariant_param_count(h, d, perm_trace(struct.perms[b]), 2 * struct.d_plus[b] - d)
                   + equivariant_param_count(h, d, perm_trace(struct.perms[b]), 2 * struct.d_plus[b + 1] - d)
                   for b in range(k))
    assert sum(a + b for a, b in V.mirror_param_counts(struct, range(k))) == expected
    W1 = [fold_mlp(tiny64.model.blocks[b])["W1"] for b in range(k)]
    gen = dict(zip(range(k), V.generic_pairings(W1, [struct.n_pairs(b) for b in range(k)], 3)))
    for ab, var, kw in [("A1", "per_layer", {}), ("A1", "random_operator", {}), ("A1", "random_structure", {}),
                        ("A2", "mirror", {}), ("A2", "random_pairing", {}), ("A2", "generic", {"precomputed": gen}),
                        ("A3", "n10", {}), ("A4", "first", {}), ("A5", "alpha1.00", {})]:
        spec = V.build_variant(ab, var, struct, k, seed=2, **kw)
        assert spec.params_kept == expected, (ab, var)
    for b in range(k):
        assert int((gen[b] != np.arange(h)).sum() // 2) == struct.n_pairs(b)
    # identity A: (h d + f d) / 2 per matrix; all self-twins: h d+ per matrix
    ident = V.build_variant("A1", "identity", struct, k)
    assert ident.params_kept == sum(2 * (h * d + struct.n_self(b) * d) // 2 for b in range(k))
    selfs = V.build_variant("A2", "self_twins", struct, k)
    assert selfs.params_kept == sum(h * (struct.d_plus[b] + struct.d_plus[b + 1]) for b in range(k))
    # 0 <= alpha < 1 keeps every parameter
    assert V.build_variant("A5", "alpha0.50", struct, k).params_kept == 2 * h * d * k
    # SVD: largest rank per matrix whose factorisation fits the matrix's equivariant count
    svd = V.build_variant("A1", "svd", struct, k)
    for (r1, r2), (e1, e2) in zip(svd.ranks, V.mirror_param_counts(struct, range(k))):
        for r, e in ((r1, e1), (r2, e2)):
            assert r * (h + d) <= e < (r + 1) * (h + d)
    assert svd.params_kept == sum((r1 + r2) * (h + d) for r1, r2 in svd.ranks)


def _projection_matrix(h, d, A, perm, mode):
    """Matrix of W1 -> projected W1 on vec(W1) (row-major)."""
    T = np.zeros((h * d, h * d))
    for c in range(h * d):
        W = np.zeros(h * d)
        W[c] = 1.0
        W = torch.tensor(W.reshape(h, d))
        out = V.project_mlp_general(W, torch.zeros(h), W.T.clone(), torch.zeros(d), A, A, perm, mode=mode)
        T[:, c] = out["W1"].numpy().ravel()
    return T


def test_fixed_space_count_and_cycle_projection():
    h, d = 7, 4
    A = V.clean_involution(random_involution(d, 3, seed=1))[0]
    perm = np.array([1, 2, 0, 4, 3, 5, 6])          # a 3-cycle, a 2-cycle and two fixed points
    assert V.fixed_space_count(perm, d, 3) == 3 + 4 + 3 + 3
    T = _projection_matrix(h, d, A, perm, "cycle")
    assert np.allclose(T, T.T, atol=1e-12) and np.allclose(T @ T, T, atol=1e-12)   # orthogonal projection
    assert np.linalg.matrix_rank(T, tol=1e-9) == V.fixed_space_count(perm, d, 3)
    Ta = _projection_matrix(h, d, A, perm, "approx")
    assert not np.allclose(Ta @ Ta, Ta, atol=1e-6)                                  # not a projection
    inv = np.array([1, 0, 2, 4, 3, 5, 6])          # an involution: all three modes agree
    Te = _projection_matrix(h, d, A, inv, "exact")
    assert np.allclose(_projection_matrix(h, d, A, inv, "approx"), Te, atol=1e-12)
    assert np.allclose(_projection_matrix(h, d, A, inv, "cycle"), Te, atol=1e-12)
    assert V.fixed_space_count(inv, d, 3) == equivariant_param_count(h, d, perm_trace(inv), 2 * 3 - d)
    with pytest.raises(ValueError):
        V.project_mlp_general(torch.zeros(h, d), torch.zeros(h), torch.zeros(d, h), torch.zeros(d), A, A, perm)


def test_cycle_projection_is_equivariant_for_any_permutation(tiny64):
    d, h = tiny64.embed_dim, tiny64.mlp_hidden
    A_in = torch.tensor(V.random_operator(d, 30, seed=11))
    A_out = torch.tensor(V.random_operator(d, 40, seed=12))
    perm = np.random.default_rng(0).permutation(h)
    assert V.cycle_stats(perm)["neurons_in_longer_cycles"] > 0
    f = fold_mlp(tiny64.model.blocks[0])
    e = V.project_mlp_general(**f, A_in=A_in, A_out=A_out, perm=perm, mode="cycle")
    p = torch.as_tensor(perm)
    assert (e["W1"][p] - e["W1"] @ A_in).abs().max().item() < 1e-10       # P W1 = W1 A_in
    assert (e["W2"][:, p] - A_out @ e["W2"]).abs().max().item() < 1e-10   # A_out W2 = W2 P
    assert (e["b1"][p] - e["b1"]).abs().max().item() < 1e-12
    assert (A_out @ e["b2"] - e["b2"]).abs().max().item() < 1e-10
    a = V.project_mlp_general(**f, A_in=A_in, A_out=A_out, perm=perm, mode="approx")
    assert (a["W1"][p] - a["W1"] @ A_in).abs().max().item() > 1e-3        # approx is not exact


# ---------------------------------------------------------------- alpha (A5)
def test_alpha_zero_unchanged_and_alpha_one_full_projection(tiny64, images64):
    struct = make_struct(tiny64)
    blocks = [0, 1, 2]
    m0 = V.variant_copy(tiny64.model, V.build_variant("A5", "alpha0.00", struct, 3))
    assert (tiny64.cls_feature(images64, m0) - tiny64.cls_feature(images64)).abs().max().item() < 1e-10
    m1 = V.variant_copy(tiny64.model, V.build_variant("A5", "alpha1.00", struct, 3))
    ref = copy.deepcopy(tiny64.model)
    apply_surgery_two(ref, blocks, [struct.A_layers[b] for b in blocks], [struct.A_layers[b + 1] for b in blocks],
                      [struct.perms[b] for b in blocks])
    for (n1, p1), (n2, p2) in zip(m1.state_dict().items(), ref.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2), n1
    m5 = V.variant_copy(tiny64.model, V.build_variant("A5", "alpha0.50", struct, 3))
    for b in blocks:
        W0 = fold_mlp(tiny64.model.blocks[b])["W1"]
        assert (m5.blocks[b].mlp.fc1.weight - 0.5 * (W0 + m1.blocks[b].mlp.fc1.weight)).abs().max().item() < 1e-12


def test_identity_with_self_twins_is_even_only_reads(tiny64, images64):
    struct = make_struct(tiny64)
    h, d = struct.h, struct.d
    # A = I: every direction is even, so even-only reads keep everything and nothing is saved.
    I = np.eye(d)
    spec = V.mirror_spec("identity_self", [0, 1], [I, I], [I, I], [np.arange(h)] * 2, h, d)
    assert spec.params_kept == spec.params_dense
    m = V.variant_copy(tiny64.model, spec)
    assert (tiny64.cls_feature(images64, m) - tiny64.cls_feature(images64)).abs().max().item() < 1e-10
    # Per-layer A with all self-twins: fc1 reads only the even subspace of A_in, fc2 writes only the
    # even subspace of A_out.
    m = V.variant_copy(tiny64.model, V.build_variant("A2", "self_twins", struct, 2))
    for b in (0, 1):
        f = fold_mlp(tiny64.model.blocks[b])
        Pi = 0.5 * (torch.eye(d, dtype=torch.float64) + torch.tensor(struct.A_layers[b]))
        Po = 0.5 * (torch.eye(d, dtype=torch.float64) + torch.tensor(struct.A_layers[b + 1]))
        mlp = m.blocks[b].mlp
        assert (mlp.fc1.weight - f["W1"] @ Pi).abs().max().item() < 1e-12
        assert (mlp.fc1.bias - f["b1"]).abs().max().item() < 1e-12
        assert (mlp.fc2.weight - Po @ f["W2"]).abs().max().item() < 1e-12
        assert (mlp.fc2.bias - Po @ f["b2"]).abs().max().item() < 1e-12


# ---------------------------------------------------------------- pairings
def test_generic_pairing_is_involution_with_requested_pairs():
    rng = np.random.default_rng(0)
    h, d = 40, 64
    W = rng.standard_normal((h, d))
    planted = [(0, 1), (2, 7), (5, 30), (10, 11), (12, 39), (20, 21), (25, 33), (14, 18)]
    for a, b in planted:
        W[b] = W[a] + 0.01 * rng.standard_normal(d)
    for n in (8, 3, 15):
        perm = V.generic_pairings([W], [n], top_k=5)[0]
        assert np.array_equal(perm[perm], np.arange(h))
        pairs = {(i, int(perm[i])) for i in range(h) if perm[i] > i}
        assert len(pairs) == n
        if n <= 8:
            assert pairs <= set(planted)
        else:
            assert set(planted) <= pairs


def test_hungarian_relaxes_the_involution_matching():
    h = 30
    S = random_scores(h, seed=3)
    perm = V.hungarian_permutation(S)
    assert np.array_equal(np.sort(perm), np.arange(h))
    m = match_neurons(S, top_k=h - 1)
    assert S[np.arange(h), perm].sum() >= S[np.arange(h), m.perm].sum() - 1e-9
    st = V.cycle_stats(perm)
    assert st["n_fixed"] + 2 * st["n_2cycles"] + st["neurons_in_longer_cycles"] == h


# ---------------------------------------------------------------- shared operators, blocks, subsets
def test_shared_operators_and_span(tiny64):
    d = tiny64.embed_dim
    A_star = V.random_operator(d, 37, seed=9)
    struct = make_struct(tiny64, common=A_star)
    for name in ("shared_span", "shared_all"):
        A_in, A_out, info = V.operator_variant(name, struct, [0, 1, 2])
        assert agreement(A_in[0], A_star) > 1 - 1e-12 and A_in[0] is A_out[2]
    assert V.operator_variant("shared_span", struct, [0, 1, 2])[2]["layers"] == [0, 1, 2, 3]
    assert V.span_layers([3, 7]) == [3, 4, 7, 8]
    # Energy normalisation stops one high-energy layer from dominating the fit (the E1 failure).
    B = V.random_operator(d, 20, seed=10)
    C = np.stack([A_star, A_star, A_star, 1e6 * B])
    G = np.stack([np.eye(d)] * 3 + [1e6 * np.eye(d)])
    plain = V.fit_shared_operator(C, range(4), None)[0]
    normed = V.fit_shared_operator(C, range(4), V.layer_energies(G))[0]
    assert agreement(plain, B) > 1 - 1e-9
    assert agreement(normed, A_star) > agreement(plain, A_star) + 0.2


def test_fit_subset_rows_nested_and_pooled():
    p = V.TinyProtocol()
    fit0 = p.index["fit0_rows"]
    r4, s4, _ = V.fit_subset_rows(p, 4, 0)
    r8, _, _ = V.fit_subset_rows(p, 8, 0)
    assert s4 == 0 and len(r4) == 4 and set(r4) <= set(r8) <= set(fit0)
    assert np.array_equal(V.fit_subset_rows(p, 4, 0, "first")[0], np.sort(fit0[:4]))
    pooled, s, desc = V.fit_subset_rows(p, 30, 1)
    assert s == -1 and "fit0+fit1+fit2" in desc
    assert set(pooled) == set(np.concatenate([p.index[f"fit{i}_rows"] for i in range(3)]))
    with pytest.raises(ValueError):
        V.fit_subset_rows(p, 31, 0)


def test_load_fit_run_cleans_and_checks(tmp_path, tiny64):
    struct = make_struct(tiny64)
    res = tmp_path / "results"
    V.save_structure(res / "e1_operators.npz", struct)
    pd.DataFrame({"layer": range(struct.depth + 1), "d_plus": struct.d_plus}).to_csv(res / "e1_layers.csv", index=False)
    pd.DataFrame({"block": range(1, struct.depth + 1),
                  "n_pairs": [struct.n_pairs(b) for b in range(struct.depth)]}).to_csv(res / "e1_mlp.csv", index=False)
    (res / "e1_meta.json").write_text(json.dumps({"fit_seed": 2, "config": {"stats": {"center": True}}}))
    got = V.load_fit_run(tmp_path)
    assert got.d_plus == struct.d_plus and got.meta["fit_seed"] == 2
    assert got.meta["checks"] == {"d_plus_matches_e1_layers_csv": True, "n_pairs_matches_e1_mlp_csv": True}
    for a, b in zip(got.A_layers, struct.A_layers):
        assert np.abs(a @ a - np.eye(struct.d)).max() < 1e-12 and np.abs(a - b).max() < 1e-5
    assert all(np.array_equal(p, q) for p, q in zip(got.perms, struct.perms))
    with pytest.raises(SystemExit):
        V.resolve_fit_seed(1, got)
    pd.DataFrame({"layer": range(struct.depth + 1),
                  "d_plus": [x + 1 for x in struct.d_plus]}).to_csv(res / "e1_layers.csv", index=False)
    with pytest.raises(ValueError):
        V.load_fit_run(tmp_path)


# ---------------------------------------------------------------- F4 measures
def test_rule_measures_reduce_to_plain_and_match_direct_svd():
    rng = np.random.default_rng(1)
    for shape in ((48, 12), (12, 48)):          # fc1 and fc2 orientations
        W = torch.tensor(rng.standard_normal(shape))
        X = rng.standard_normal((200, shape[1])) * rng.uniform(0.1, 3, shape[1])
        M = torch.tensor(X.T @ X)
        I = torch.eye(shape[1], dtype=torch.float64)
        W_eq = W * torch.tensor(rng.uniform(0.3, 1, shape))
        r = 5
        assert abs(V.weighted_share(W, W_eq, I) - breaking_share(W, W_eq)) < 1e-12
        assert abs(V.svd_discarded_energy(W, r, I) - V.svd_discarded_energy(W, r)) < 1e-10
        lam, U = np.linalg.eigh(M.numpy())
        K = (U * np.sqrt(np.clip(lam, 0, None))) @ U.T
        direct = np.linalg.svd(W.numpy() @ K, compute_uv=False) ** 2
        assert np.allclose(V.weighted_spectrum(W, M).numpy()[:len(direct)], direct, rtol=1e-8, atol=1e-8)
        # Eckart-Young: the plain truncation error equals the plain discarded energy, and the
        # activation-aware optimum is never worse than the plain truncation in the weighted norm.
        assert abs(V.applied_svd_error(W, r) - V.svd_discarded_energy(W, r)) < 1e-10
        assert V.applied_svd_error(W, r, M) >= V.svd_discarded_energy(W, r, M) - 1e-12


def test_mlp_input_moments_match_direct_computation(tiny64):
    imgs = torch.from_numpy(np.random.default_rng(2).integers(0, 256, (3, 3, 56, 56), dtype=np.uint8))
    loader = [(imgs, torch.zeros(3, dtype=torch.long))]
    out = V.collect_mlp_input_moments(tiny64, loader, "cpu", blocks=[0, 2])
    d = tiny64.embed_dim
    with torch.no_grad():
        t = tiny64.layer0_tokens(normalize_for(tiny64, imgs))
        for b, blk in enumerate(tiny64.model.blocks):
            mid = t + blk.ls1(blk.attn(blk.norm1(t)))
            if b in (0, 2):
                U = F.layer_norm(mid, (d,), eps=blk.norm2.eps).reshape(-1, d)
                H = blk.mlp.act(blk.mlp.fc1(blk.norm2(mid))).reshape(-1, tiny64.mlp_hidden)
                assert (out["M1"][b] - U.T @ U).abs().max().item() < 1e-8
                assert (out["M2"][b] - H.T @ H).abs().max().item() < 1e-8
            t = blk(t)
    assert out["n_images"] == 3 and set(out["M1"]) == {0, 2}


def test_block_rule_measures_keys(tiny64):
    struct = make_struct(tiny64)
    d, h = struct.d, struct.h
    m = V.block_rule_measures(tiny64.model.blocks[1], struct.A_layers[1], struct.A_layers[2], struct.perms[1],
                              M1=torch.eye(d, dtype=torch.float64), M2=torch.eye(h, dtype=torch.float64),
                              M1_G=struct.G[1])
    assert abs(m["s_fc1_act"] - m["s_fc1_weight"]) < 1e-10 and abs(m["E_fc2_act"] - m["E_fc2_weight"]) < 1e-10
    assert abs(m["s_fc1_actG"] - m["s_fc1_weight"]) < 1e-10      # G_1 is a multiple of I here
    eq1, eq2 = V.mirror_param_counts(struct, [1])[0]
    assert (m["eq_params_fc1"], m["eq_params_fc2"]) == (eq1, eq2)
    assert m["svd_rank_fc1"] == eq1 // (h + d)


def test_f4_rule_hits_counts_ties_as_misses():
    spec = importlib.util.spec_from_file_location("f4_rule", ROOT / "scripts" / "f4_rule.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    hits = mod.rule_hits(np.array([0.1, -0.2, 0.3, 0.0, 0.2]), np.array([2.0, -1.0, -1.0, 1.0, 0.0]))
    assert hits.tolist() == [True, True, False, False, False]


# ---------------------------------------------------------------- CLI --tiny paths
def run_script(name, *args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / name), *args], capture_output=True, text=True,
                          timeout=1200)


def test_stage1_ablate_tiny_end_to_end_and_resumable(tmp_path):
    out = tmp_path / "a"
    common = ["--tiny", "--ks", "1,2", "--out", str(out), "--top-k", "3", "--a3-sizes", "4,30"]
    proc = run_script("stage1_ablate.py", "--ablation", "A1,A2,A3,A4,A5", *common)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    expected = {"A1": 14, "A2": 12, "A3": 4, "A4": 6, "A5": 10}
    for ab, n in expected.items():
        df = pd.read_csv(out / f"{ab}_s0.csv")
        assert len(df) == n, ab
        assert not df.duplicated(["ablation", "variant", "k", "fit_seed"]).any()
        assert df["git_hash"].notna().all() and (df["seed"] == 0).all()
        assert df["top1"].between(0, 100).all() and df["flip_consistency"].between(0, 100).all()
        meta = json.loads((out / f"{ab}_s0_meta.json").read_text(encoding="utf-8"))
        assert "git_hash" in meta["git"] and meta["fit_seed"] == 0
    a3 = pd.read_csv(out / "A3_s0.csv")
    assert set(a3.loc[a3["variant"] == "n30", "fit_seed"]) == {-1} and set(a3.loc[a3["variant"] == "n4", "fit_seed"]) == {0}
    a2 = pd.read_csv(out / "A2_s0.csv")
    assert not a2.loc[a2["variant"] == "hungarian", "params_exact"].any()
    a5 = pd.read_csv(out / "A5_s0.csv")
    assert (a5.loc[a5["variant"] == "alpha0.00", "mlp_params_kept_frac"] == 1).all()
    assert (a5.loc[a5["variant"] == "alpha1.00", "mlp_params_kept_frac"] < 0.6).all()
    a1 = pd.read_csv(out / "A1_s0.csv")
    assert (a1.loc[a1["variant"] == "svd", "params_vs_mirror_first_k"] <= 1).all()
    # The same model in another ablation is not evaluated again.
    a4 = pd.read_csv(out / "A4_s0.csv")
    for df, var in ((a2, "mirror"), (a4, "first"), (a5, "alpha1.00")):
        rows = df[df["variant"] == var].sort_values("k")
        assert rows["reused_from"].tolist() == ["A1/per_layer/k1/s0", "A1/per_layer/k2/s0"], var
        assert rows["top1"].tolist() == a1[a1["variant"] == "per_layer"].sort_values("k")["top1"].tolist()
    assert a1["reused_from"].isna().all()
    # A second run finds every row and adds nothing.
    proc = run_script("stage1_ablate.py", "--ablation", "A5", *common)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "skipped" in proc.stdout and len(pd.read_csv(out / "A5_s0.csv")) == 10


def test_f4_rule_tiny_end_to_end(tmp_path):
    out = tmp_path / "f4"
    proc = run_script("f4_rule.py", "--tiny", "--out", str(out), "--top-k", "3")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    df = pd.read_csv(out / "f4_tiny.csv")
    assert len(df) == 4 and df["block"].tolist() == [1, 2, 3, 4]
    for col in ("s_act", "E_act", "s_weight", "E_weight", "drop_mirror", "drop_svd", "rule_x", "rule_y",
                "git_hash", "seed", "prereg_commit", "n_query"):
        assert col in df.columns and df[col].notna().all(), col
    assert np.allclose(df["rule_x"], df["E_act"] - df["s_act"])
    assert np.allclose(df["rule_y"], df["drop_svd"] - df["drop_mirror"])
    p2 = json.loads((out / "f4_p2.json").read_text(encoding="utf-8"))
    assert p2["verdict"].startswith("INCOMPLETE") and p2["primary_all_models_run"]["n_cases"] == 4
    assert (out / "plots" / "fig4_rule.png").is_file()
    proc = run_script("f4_rule.py", "--tiny", "--out", str(out), "--top-k", "3")
    assert proc.returncode == 0 and "already in the CSV" in proc.stdout
