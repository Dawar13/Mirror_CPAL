"""Group measurement library (mirror/groups.py, mirror/group_stats.py; MUST-03). CPU only, no downloads.

  * conventions: a marker patch lands on the mapped token for LR, UD, R90, R180, R270 on the grids of
    DINOv2 with registers (16 x 16, 5 prefix tokens), DeiT-III/MAE (14 x 14, 1) and ViT-S/8 at 64 px
    (8 x 8, 1), with timm patch embeddings; LR alignment equals stats.unflip_tokens;
  * the group projection for |G| = 2 equals surgery.project_mlp_two to the last bit;
  * character formula = rank of the projector = brute-force null space of the constraints, for LR, R90 and
    Klein (fc1 and fc2, different input and output operators); for LR it is equivariant_param_count;
  * exactly equivariant synthetic layers: breaking share 0, orbit counts right, and the MLP function is
    equivariant, f(rho_in x) = rho_out f(x), for LR, R90 and Klein; random layers sit at the null level;
  * operator fits recover planted R90 and Klein operators; rho^4 = I and commutation are enforced;
  * hidden matchings: 1/2/4-cycle rule, Klein commutation repair, random actions keep the cycle type;
  * attention: the dense projection equals attention.project_attn for involutions, its free parameters
    equal attention.attention_free_params, and the R90 head structure gives an exact projection;
  * group surgery equals surgery.apply_surgery_two for an involution and makes the MLP branch equivariant
    for R90; with g = LR the paired statistics equal stats.collect_fit_stats; the twin-null pass with the
    true alignment reproduces the main pass.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mirror import groups as GR
from mirror.matching import random_pairing

ELS = ("LR", "UD", "R90", "R180", "R270")


# ---------------------------------------------------------------------------------------------- conventions
@pytest.mark.parametrize("img,patch,reg", [(224, 14, 4), (224, 16, 0), (64, 8, 0)],
                         ids=["dinov2_reg_16x16_p5", "deit3_mae_14x14", "vits8_64px_8x8"])
def test_marker_lands_on_mapped_token(img, patch, reg):
    from timm.models.vision_transformer import VisionTransformer

    torch.manual_seed(0)
    m = VisionTransformer(img_size=img, patch_size=patch, embed_dim=8, depth=1, num_heads=2, reg_tokens=reg)
    G, npre = img // patch, int(m.num_prefix_tokens)
    assert npre == 1 + reg
    rng = np.random.default_rng(1)
    for el in ELS:
        idx = GR.token_index(el, G)
        assert sorted(idx.tolist()) == list(range(G * G))
        for _ in range(3):
            r, c = (int(v) for v in rng.integers(0, G, 2))
            x = torch.zeros(1, 3, img, img)
            x[..., r * patch:(r + 1) * patch, c * patch:(c + 1) * patch] = torch.rand(3, patch, patch)
            with torch.no_grad():
                base = m.patch_embed(torch.zeros_like(x))
                t = m.patch_embed(GR.image_action(x, el)) - base           # [1, G*G, C], row-major
            hot = int(t.norm(dim=-1)[0].argmax())
            rr, cc = GR.grid_map(el, r, c, G)
            assert hot == rr * G + cc == idx[r * G + c]
            pre = torch.randn(1, npre, t.shape[-1])
            al = GR.align_tokens(torch.cat([pre, t], 1), idx, npre)
            assert torch.equal(al[:, :npre], pre)                           # prefix tokens keep their index
            assert int(al[0, npre:].norm(dim=-1).argmax()) == r * G + c       # marker back at (r, c)


def test_r90_is_counter_clockwise_and_composes():
    x = torch.arange(16.0).reshape(1, 1, 4, 4)
    y = GR.image_action(x, "R90")
    assert torch.equal(y[0, 0], torch.tensor(np.rot90(x[0, 0].numpy()).copy()))      # numpy: counter-clockwise
    assert y[0, 0, 0, 0] == x[0, 0, 0, 3]                                              # top-right goes top-left
    G = 6
    r90 = GR.token_index("R90", G)
    assert np.array_equal(r90[r90], GR.token_index("R180", G))
    assert np.array_equal(r90[r90][r90], GR.token_index("R270", G))
    assert np.array_equal(GR.token_index("LR", G)[GR.token_index("UD", G)], GR.token_index("R180", G))


def test_lr_alignment_equals_unflip_tokens():
    from mirror.stats import unflip_tokens

    t = torch.randn(2, 5 + 16 * 16, 7)
    assert torch.equal(GR.align_tokens(t, GR.token_index("LR", 16), 5), unflip_tokens(t, (16, 16), 5))


# ---------------------------------------------------------------------------------------------- helpers
def rand_inv(d, dp, seed):
    return GR.random_c2(d, dp, seed, fix_ones=False)


def klein_action(spec, seed=0):
    """Commuting involutions on hidden units from orbit types, e.g. ['4', 'a', 'ab', '1']."""
    h = sum({"1": 1, "a": 2, "b": 2, "ab": 2, "4": 4}[t] for t in spec)
    pa, pb = np.arange(h), np.arange(h)
    pos = 0
    for t in spec:
        if t == "1":
            pos += 1
        elif t == "4":
            u = list(range(pos, pos + 4))
            pa[u] = [u[1], u[0], u[3], u[2]]
            pb[u] = [u[2], u[3], u[0], u[1]]
            pos += 4
        else:
            if "a" in t:
                pa[pos], pa[pos + 1] = pos + 1, pos
            if "b" in t:
                pb[pos], pb[pos + 1] = pos + 1, pos
            pos += 2
    perm = np.random.default_rng(seed).permutation(h)
    inv = np.argsort(perm)
    return inv[pa[perm]], inv[pb[perm]]                 # relabel units (conjugate by a random permutation)


def cycles_perm(lengths, seed=0):
    n = sum(lengths)
    order = np.random.default_rng(seed).permutation(n)
    p, pos = np.arange(n), 0
    for L in lengths:
        cyc = order[pos:pos + L]
        p[cyc] = np.roll(cyc, -1)
        pos += L
    return p


def structures():
    """(group, BlockAction, h, d) for LR, R90 and Klein with different input and output operators."""
    out = []
    d, h = 6, 8
    p = random_pairing(h, 3, seed=1)
    out.append(("LR", GR.block_action("LR", [rand_inv(d, 4, 1)], [rand_inv(d, 2, 2)], [p]), h, d))
    d, h = 7, 9
    rin, rout = GR.random_c4(d, 2, 1, 2, seed=3, fix_ones=False), GR.random_c4(d, 1, 2, 2, seed=4, fix_ones=False)
    out.append(("R90", GR.block_action("R90", [rin], [rout], [cycles_perm([4, 2, 1, 1, 1], seed=5)]), h, d))
    d = 8
    pa, pb = klein_action(["4", "a", "ab", "1"], seed=6)
    h = len(pa)
    ia, ib = GR.random_klein(d, 2, 2, 3, 1, seed=7, fix_ones=False)
    oa, ob = GR.random_klein(d, 3, 1, 1, 3, seed=8, fix_ones=False)
    out.append(("Klein", GR.block_action("Klein", [ia, ib], [oa, ob], [pa, pb]), h, d))
    return out


def project_random(act, h, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    W1, b1 = torch.randn(h, d, generator=g, dtype=torch.float64), torch.randn(h, generator=g, dtype=torch.float64)
    W2, b2 = torch.randn(d, h, generator=g, dtype=torch.float64), torch.randn(d, generator=g, dtype=torch.float64)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}, GR.project_mlp_group(W1, b1, W2, b2, act)


# ---------------------------------------------------------------------------------------------- projection
def test_order2_projection_equals_surgery_bitwise():
    from mirror.surgery import project_mlp_two
    from mirror.variants import random_operator

    d, h = 32, 96
    A_in, A_out = random_operator(d, 20, 1), random_operator(d, 11, 2)
    p = random_pairing(h, 30, seed=3)
    act = GR.block_action("UD", [A_in], [A_out], [p])
    f, e = project_random(act, h, d)
    ref = project_mlp_two(**f, A_in=A_in, A_out=A_out, perm=p)
    for k in ("W1", "b1", "W2", "b2"):
        assert torch.equal(e[k], ref[k]), k


@pytest.mark.parametrize("idx", [0, 1, 2], ids=["LR", "R90", "Klein"])
def test_character_formula_equals_brute_force(idx):
    group, act, h, d = structures()[idx]
    fp = GR.mlp_free_params(act, h, d)
    for which, key in (("fc1", "free_fc1"), ("fc2", "free_fc2")):
        Pm = GR.projector_matrix(act, h, d, which)
        assert np.allclose(Pm @ Pm, Pm, atol=1e-12) and np.allclose(Pm, Pm.T, atol=1e-12)   # orthogonal projector
        rank = int(np.linalg.matrix_rank(Pm, tol=1e-9))
        assert fp[key] == rank == GR.constraint_nullity(act, h, d, which), (group, which, fp[key], rank)
    oid, sizes = GR.orbits([p for p in act.perms if p is not None])
    assert fp["free_b1"] == len(sizes)
    if group == "LR":
        from mirror.surgery import equivariant_param_count

        f = int((act.perms[1] == np.arange(h)).sum())
        assert fp["free_fc1"] == equivariant_param_count(h, d, f, int(round(np.trace(act.rho_in[1]))))
        assert fp["free_fc2"] == equivariant_param_count(h, d, f, int(round(np.trace(act.rho_out[1]))))


@pytest.mark.parametrize("idx", [0, 1, 2], ids=["LR", "R90", "Klein"])
def test_exactly_equivariant_layer(idx):
    group, act, h, d = structures()[idx]
    _, e = project_random(act, h, d, seed=4)
    sh = GR.mlp_group_shares(e, act)
    assert max(sh["s_fc1"], sh["s_fc2"], sh["s_b1"], sh["s_b2"]) < 1e-28
    # function level: f(rho_in(g) x) = rho_out(g) f(x) for every element, with the hidden permutation
    x = torch.randn(5, d, dtype=torch.float64)

    def mlp(x):
        return F.gelu(x @ e["W1"].T + e["b1"]) @ e["W2"].T + e["b2"]

    for Ri, Ro in zip(act.rho_in, act.rho_out):
        Ri = torch.eye(d, dtype=torch.float64) if Ri is None else torch.as_tensor(Ri)
        Ro = torch.eye(d, dtype=torch.float64) if Ro is None else torch.as_tensor(Ro)
        assert torch.allclose(mlp(x @ Ri.T), mlp(x) @ Ro.T, atol=1e-10)
    # orbit counts: planted cycle / orbit types come back
    gens = [p for p in act.perms[1:2]] if group != "Klein" else [act.perms[1], act.perms[2]]
    fr = GR.orbit_fractions(gens, [np.ones(h)] * len(gens))
    want = {"LR": (2 / 8, 6 / 8, 0), "R90": (3 / 9, 2 / 9, 4 / 9), "Klein": (1 / 9, 4 / 9, 4 / 9)}[group]
    assert (fr["frac_fixed"], fr["frac_2"], fr["frac_4"]) == pytest.approx(want)
    assert fr["frac_2_t50"] == pytest.approx(want[1])


@pytest.mark.parametrize("group", ["LR", "R90", "Klein"])
def test_random_layer_sits_at_null(group):
    d, h = 48, 192
    rng = np.random.default_rng(0)
    if group == "LR":
        op = {"d_plus": 30}
        gens = [random_pairing(h, 70, seed=1)]
    elif group == "R90":
        op = {"m_plus": 14, "m_minus": 10, "m_i_pairs": 12}
        gens = [cycles_perm([4] * 30 + [2] * 20 + [1] * 32, seed=1)]
    else:
        op = {"m_pp": 15, "m_pm": 12, "m_mp": 11, "m_mm": 10}
        gens = list(klein_action(["4"] * 25 + ["a"] * 15 + ["ab"] * 10 + ["1"] * 42, seed=1))
    fit = GR.block_action(group, GR.random_operator_like(group, op, d, 1), GR.random_operator_like(group, op, d, 2), gens)
    g = torch.Generator().manual_seed(9)
    f = {"W1": torch.randn(h, d, generator=g, dtype=torch.float64), "b1": torch.zeros(h, dtype=torch.float64),
         "W2": torch.randn(d, h, generator=g, dtype=torch.float64), "b2": torch.zeros(d, dtype=torch.float64)}
    s = GR.mlp_group_shares(f, fit)["s_mlp"]
    nulls = []
    for k in range(20):
        act = GR.block_action(group, GR.random_operator_like(group, op, d, 100 + k),
                              GR.random_operator_like(group, op, d, 200 + k), GR.random_perm_like(group, gens, rng))
        nulls.append(GR.mlp_group_shares(f, act)["s_mlp"])
    ns = GR.null_summary(nulls)
    assert abs(s / ns["mean"] - 1) < 0.05
    tax = GR.mlp_free_params(fit, h, d)["tax_mlp"]
    assert abs(s - tax) < 0.03                       # for an isotropic random layer E[s] = tax


# ---------------------------------------------------------------------------------------------- operators
def centred_data(n, d, seed):
    X = np.random.default_rng(seed).standard_normal((n, d))
    return X - X.mean(1, keepdims=True)


def test_fit_c4_recovers_planted_operator_and_enforces_order():
    d = 24
    rho = GR.random_c4(d, 8, 6, 5, seed=1)
    X = centred_data(4000, d, 2)
    C = X.T @ (X @ rho.T)
    fc = GR.fit_c4(C)
    assert np.abs(fc["rho"] - rho).max() < 1e-8
    assert (fc["m_plus"], fc["m_minus"], fc["m_i_pairs"]) == (8, 6, 5)
    assert GR.moment_error(C, (X * X).sum(), (X * X).sum(), fc["rho"]) == pytest.approx(0, abs=1e-6)
    noisy = X @ rho.T + 0.6 * centred_data(4000, d, 3)
    fc = GR.fit_c4(X.T @ noisy)
    assert np.abs(np.linalg.matrix_power(fc["rho"], 4) - np.eye(d)).max() < 1e-10
    assert np.abs(np.linalg.matrix_power(fc["rho_raw"], 4) - np.eye(d)).max() > 1e-3
    assert np.abs(fc["rho"] @ np.ones(d) - np.ones(d)).max() < 1e-10
    assert np.abs(fc["rho"] - rho).max() < 0.2


def test_fit_klein_recovers_planted_pair_and_commutes():
    d = 20
    A, B = GR.random_klein(d, 6, 5, 5, 4, seed=1)
    X = centred_data(3000, d, 2)
    kl = GR.fit_klein(X.T @ (X @ A), X.T @ (X @ B))
    assert np.abs(kl["A_a"] - A).max() < 1e-8 and np.abs(kl["A_b"] - B).max() < 1e-8
    noisy = [X.T @ (X @ M + 0.8 * centred_data(3000, d, s)) for M, s in ((A, 3), (B, 4))]
    kl = GR.fit_klein(*noisy)
    assert np.abs(kl["A_a"] @ kl["A_b"] - kl["A_b"] @ kl["A_a"]).max() < 1e-10
    m = GR.klein_multiplicities(kl["A_a"], kl["A_b"])
    assert sum(m.values()) == d and kl["m_pp"] >= 1
    a0, b0 = GR.fit_order2(noisy[0])[0], GR.fit_order2(noisy[1])[0]
    obj0 = np.trace(a0 @ noisy[0]) + np.trace(b0 @ noisy[1])
    obj = np.trace(kl["A_a"] @ noisy[0]) + np.trace(kl["A_b"] @ noisy[1])
    assert obj <= obj0 + 1e-9                       # the commuting optimum cannot beat the free one


def test_random_operators_keep_character():
    d = 30
    r = GR.random_c4(d, 9, 7, 7, seed=3)
    assert GR.c4_multiplicities(r) == {"m_plus": 9, "m_minus": 7, "m_i_pairs": 7}
    A = GR.random_c2(d, 13, seed=4)
    assert int(round(np.trace(A))) == 13 - 17 and np.abs(A @ A - np.eye(d)).max() < 1e-12
    a, b = GR.random_klein(d, 10, 8, 7, 5, seed=5)
    assert GR.klein_multiplicities(a, b) == {"m_pp": 10, "m_pm": 8, "m_mp": 7, "m_mm": 5}


# ---------------------------------------------------------------------------------------------- permutations
def test_hungarian_c4_keeps_1_2_4_cycles():
    p = cycles_perm([4, 4, 3, 2, 5, 1], seed=1)
    h = len(p)
    R = np.full((h, h), -0.1)
    R[np.arange(h), p] = 0.9
    res = GR.hungarian_c4(R)
    assert np.array_equal(res["perm_raw"], p)
    assert res["n_turned"] == 8 and res["n_cycles_turned"] == 2
    lens = GR.cycle_lengths(res["perm"])
    assert sorted(lens.tolist()) == [1] * 9 + [2, 4, 4]
    fr = GR.orbit_fractions([res["perm"]], [res["score"]])
    assert fr["frac_4"] == pytest.approx(8 / h) and fr["frac_4_t80"] == pytest.approx(8 / h)


def test_klein_commuting_repair_and_random_action():
    h = 12
    pa = np.arange(h)
    pb = np.arange(h)
    pa[[0, 1, 2, 3]] = [1, 0, 3, 2]
    pb[[1, 2]] = [2, 1]                                  # (0 1)(2 3) and (1 2) do not commute
    pa[[4, 5, 6, 7]] = [5, 4, 7, 6]
    pb[[4, 5, 6, 7]] = [6, 7, 4, 5]                      # a commuting 4-orbit
    assert GR.commute_share(pa, pb) < 1
    gain_a, gain_b = np.ones(h), np.zeros(h)
    kc = GR.klein_commuting(pa, pb, gain_a, gain_b)
    assert GR.commute_share(kc["perm_a"], kc["perm_b"]) == 1.0
    assert np.array_equal(kc["perm_a"], pa) and kc["n_units_changed"] == 2 and kc["n_orbits_fixed"] == 1
    na, nb = GR.random_klein_action_like(kc["perm_a"], kc["perm_b"], np.random.default_rng(0))
    assert GR.commute_share(na, nb) == 1.0
    _, s0 = GR.orbits([kc["perm_a"], kc["perm_b"]])
    _, s1 = GR.orbits([na, nb])
    assert sorted(s0.tolist()) == sorted(s1.tolist())
    p = cycles_perm([4, 4, 2, 1, 1, 3])
    q = GR.random_permutation_like(p, np.random.default_rng(1))
    assert sorted(GR.cycle_lengths(q).tolist()) == sorted(GR.cycle_lengths(p).tolist())


# ---------------------------------------------------------------------------------------------- attention
def tiny_attn(seed=0):
    from mirror.adapters import TimmTinyAdapter
    from mirror.attention import attn_dims, fold_attn

    ad = TimmTinyAdapter(img_size=56, seed=seed, randomise=True, dtype=torch.float64)
    blk = ad.model.blocks[1]
    return fold_attn(blk), attn_dims(blk), ad.embed_dim


def test_attention_c2_matches_existing():
    from mirror.attention import attention_free_params, project_attn
    from mirror.m1 import random_head_structure

    f, (H, dh), d = tiny_attn()
    hm = random_head_structure(H, dh, 1, seed=3)
    A_in, A_out = GR.random_c2(d, 40, 1), GR.random_c2(d, 30, 2)
    e = GR.project_attn_group(f, GR.attn_action_c2(hm, A_in, A_out))
    ref = project_attn(**f, A_in=A_in, A_out=A_out, head_maps=hm, gauge=False)
    for k in ref:
        assert torch.allclose(e[k], ref[k], atol=1e-12), k
    fa = GR.attn_free_params(GR.attn_action_c2(hm, A_in, A_out), d, H * dh)
    fr = attention_free_params(d, hm, A_in, A_out)
    for k in ("Wq", "Wk", "Wv", "Wo", "bq", "bk", "bv", "bo", "matrices"):
        assert fa[k] == fr[k], k


def test_attention_c4_projection_is_exact():
    f, (H, dh), d = tiny_attn()
    rng = np.random.default_rng(0)
    rin, rout = GR.random_c4(d, 20, 16, 14, seed=1), GR.random_c4(d, 18, 14, 16, seed=2)
    # synthetic head moments: one 4-cycle of heads with arbitrary maps; fit_heads_c4 closes the cycle
    M = np.zeros((H, H, dh, dh))
    sig = np.array([1, 2, 3, 0])
    for h in range(H):
        M[sig[h], h] = np.linalg.qr(rng.standard_normal((dh, dh)))[0] * 5 + 0.01 * rng.standard_normal((dh, dh))
    mom = {"M": {"q": M, "k": M, "v": M}, "E": {t: np.ones(H) for t in "qkv"}, "Et": {t: np.ones(H) for t in "qkv"}}
    hs = GR.fit_heads_c4(mom)
    assert np.array_equal(hs["sigma"], sig)
    act = GR.attn_action_c4(hs, rin, rout)
    P = act.P_qk[1]
    assert np.abs(np.linalg.matrix_power(P, 4) - np.eye(H * dh)).max() < 1e-10
    e = GR.project_attn_group(f, act)
    e2 = GR.project_attn_group(e, act)
    assert max(float((e[k] - e2[k]).abs().max()) for k in e) < 1e-12              # idempotent
    Pq = torch.as_tensor(P)
    assert torch.allclose(Pq @ e["Wq"], e["Wq"] @ torch.as_tensor(rin), atol=1e-12)  # P Wq = Wq rho_in
    assert torch.allclose(torch.as_tensor(rout) @ e["Wo"], e["Wo"] @ torch.as_tensor(act.P_v[1]), atol=1e-12)
    sh = GR.attn_shares(e, GR.project_attn_group(e, act))
    assert sh["s_attn"] < 1e-25
    rnd = GR.random_heads_c4_like(hs, seed=5)
    Pr = GR.attn_action_c4(rnd, rin, rout).P_qk[1]
    assert np.abs(np.linalg.matrix_power(Pr, 4) - np.eye(H * dh)).max() < 1e-10
    assert np.trace(Pr) == pytest.approx(np.trace(P), abs=1e-8)


def test_attention_klein_intersection():
    from mirror.m1 import random_head_structure

    f, (H, dh), d = tiny_attn()
    A, B = GR.random_klein(d, 20, 16, 14, 14, seed=1)
    hm = random_head_structure(H, dh, 0, seed=2, self_traces=[(0, 0)] * H)   # self-heads only
    hm.B[:] = np.eye(dh)                                     # identity head maps commute with everything
    hm.C[:] = np.eye(dh)
    a1, a2 = GR.attn_action_c2(hm, A, A), GR.attn_action_c2(hm, B, B)
    e, info = GR.project_attn_klein(f, a1, a2)
    for act in (a1, a2):
        e2 = GR.project_attn_group(e, act)
        assert max(float((e[k] - e2[k]).abs().max()) for k in e) < 1e-10
    assert info["sweeps"] <= 3                               # commuting structures converge at once


# ---------------------------------------------------------------------------------------------- surgery and stats
def test_group_surgery_matches_apply_surgery_two(tiny64):
    import copy

    from mirror.surgery import apply_surgery_two

    d, h = tiny64.embed_dim, tiny64.mlp_hidden
    blocks = [0, 2]
    Ain = [GR.random_c2(d, 30, 1), GR.random_c2(d, 35, 2)]
    Aout = [GR.random_c2(d, 33, 3), GR.random_c2(d, 28, 4)]
    perms = [random_pairing(h, 60, seed=5), random_pairing(h, 90, seed=6)]
    m1, m2 = copy.deepcopy(tiny64.model), copy.deepcopy(tiny64.model)
    GR.apply_group_surgery(m1, blocks, [[a] for a in Ain], [[a] for a in Aout], [[p] for p in perms], group="UD")
    apply_surgery_two(m2, blocks, Ain, Aout, perms)
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert torch.equal(p1, p2), n1


def test_group_surgery_r90_branch_equivariant(tiny64):
    import copy

    d = tiny64.embed_dim
    rin, rout = GR.random_c4(d, 20, 16, 14, seed=1), GR.random_c4(d, 22, 14, 14, seed=2)
    perm = cycles_perm([4] * 40 + [2] * 30 + [1] * 36, seed=3)
    m = copy.deepcopy(tiny64.model)
    GR.apply_group_surgery(m, [1], [[rin]], [[rout]], [[perm]], group="R90")
    blk = m.blocks[1]
    x = torch.randn(3, 7, d, dtype=torch.float64)

    def branch(x):
        return blk.ls2(blk.mlp(blk.norm2(x)))

    Ri, Ro = torch.as_tensor(rin), torch.as_tensor(rout)
    assert torch.allclose(branch(x @ Ri.T), branch(x) @ Ro.T, atol=1e-10)


def test_group_pass_lr_equals_collect_fit_stats():
    from mirror.adapters import TimmTinyAdapter
    from mirror.group_stats import collect_group_pass
    from mirror.stats import collect_fit_stats
    from mirror.variants import TinyProtocol

    ad = TimmTinyAdapter(img_size=56, seed=0)
    proto = TinyProtocol()
    ref = collect_fit_stats(ad, proto.fit_loader(0), "cpu", center=True, mlp=True)
    gp = collect_group_pass(ad, proto.fit_loader(0), "cpu", "LR", range(ad.depth), attn=True, inputs=True)
    r = ref.residual.numpy()
    assert gp.layers == list(range(ad.depth + 1))
    for k in ("C", "G", "sq_x", "sq_xt", "n"):
        assert np.array_equal(gp.residual[k], r[k]), k
    for b in range(ad.depth):
        assert np.array_equal(gp.R[b], ref.mlp.correlation(b))
    from mirror.variants import collect_mlp_input_moments

    mom = collect_mlp_input_moments(ad, proto.fit_loader(0), "cpu")
    for b in range(ad.depth):   # float32 forward of [x; g(x)] against x alone: equal to float32 rounding
        assert np.allclose(gp.inputs["M1"][b], mom["M1"][b].numpy(), rtol=1e-5, atol=1e-5)
        assert np.allclose(gp.inputs["M2"][b], mom["M2"][b].numpy(), rtol=1e-5, atol=1e-5)


def test_twin_null_pass_with_true_alignment_reproduces_main_pass():
    from mirror.adapters import TimmTinyAdapter
    from mirror.group_stats import collect_group_pass, collect_twin_null
    from mirror.variants import TinyProtocol

    ad = TimmTinyAdapter(img_size=56, seed=0)
    proto = TinyProtocol()
    gp = collect_group_pass(ad, proto.fit_loader(0), "cpu", "R90", [0, 1], residual=False, attn=False)
    true = GR.token_index("R90", ad.grid_hw)
    rnd = GR.random_permutation_like(true, np.random.default_rng(0))
    assert sorted(GR.cycle_lengths(rnd).tolist()) == sorted(GR.cycle_lengths(true).tolist())
    Rs = collect_twin_null(ad, proto.fit_loader(0), "cpu", "R90", [true, rnd], [0, 1])
    for b in (0, 1):
        assert np.array_equal(Rs[0][b], gp.R[b])
        assert not np.allclose(Rs[1][b], gp.R[b])
