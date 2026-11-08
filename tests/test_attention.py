"""M1 attention conversion: folding, projection, exact equivariance, head pairing and head maps."""
import copy
import itertools

import numpy as np
import pytest
import torch

from mirror.attention import (AttnPairedHooks, AttnStats, HeadMaps, absorb_attn, apply_attention_surgery,
                              attention_breaking_shares, attention_free_params, attn_branch, attn_dims,
                              breaking_shares_from, fit_head_maps, fold_attn, head_map_errors,
                              head_pair_scores, match_heads, pair_and_fit_heads, project_attn)
from mirror.data import mirror, normalize
from mirror.fit import random_involution, random_orthogonal
from mirror.stats import run_paired, unflip_tokens


def involution_fixing_ones(d, d_plus, seed):
    """Random involution with A 1 = 1 (needed for LayerNorm equivariance)."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(np.column_stack([np.ones(d), rng.standard_normal((d, d - 1))]))
    signs = np.concatenate([[1.0], np.ones(d_plus - 1), -np.ones(d - d_plus)])
    A = (Q * signs) @ Q.T
    return 0.5 * (A + A.T)


def random_head_maps(sigma, dh, seed, self_plus=None):
    """Consistent random maps: orthogonal B, C on pairs (B[g] = B[h]^T), involutions on self-heads."""
    rng = np.random.default_rng(seed)
    sigma = np.asarray(sigma)
    H = len(sigma)
    B, C = np.zeros((H, dh, dh)), np.zeros((H, dh, dh))
    for h in range(H):
        g = int(sigma[h])
        if g == h:
            pb, pc = (self_plus[h] if self_plus else (int(rng.integers(1, dh)), int(rng.integers(1, dh))))
            B[h] = random_involution(dh, pb, seed=int(rng.integers(1 << 30)))
            C[h] = random_involution(dh, pc, seed=int(rng.integers(1 << 30)))
        elif g > h:
            B[h], C[h] = random_orthogonal(dh, rng), random_orthogonal(dh, rng)
            B[g], C[g] = B[h].T, C[h].T
    return HeadMaps(sigma=sigma.copy(), B=B, C=C)


def tokens(shape, seed):
    return torch.randn(*shape, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------- folding
def test_fold_attn_is_exact(tiny64, images64):
    t = tiny64.layer0_tokens(images64)
    for blk in tiny64.model.blocks:
        folded = copy.deepcopy(blk)
        absorb_attn(folded)
        assert torch.allclose(folded.norm1.weight, torch.ones_like(folded.norm1.weight))
        assert torch.all(folded.norm1.bias == 0)
        assert torch.allclose(folded.ls1.gamma, torch.ones_like(folded.ls1.gamma))
        out = blk(t)
        assert (out - folded(t)).abs().max().item() < 1e-10
        assert (attn_branch(blk, t) - attn_branch(folded, t)).abs().max().item() < 1e-10
        # Not trivial: the folded quantities differ from the raw ones and the branch contributes.
        assert (fold_attn(blk)["Wq"] - blk.attn.qkv.weight[:64]).abs().max().item() > 1e-2
        assert attn_branch(blk, t).abs().max().item() > 1e-2
        t = out


def test_identity_projection_leaves_model_unchanged(tiny64, images64):
    m = copy.deepcopy(tiny64.model)
    L, d = tiny64.depth, tiny64.embed_dim
    H, dh = attn_dims(m.blocks[0])
    eye = np.eye(d)
    maps = apply_attention_surgery(m, list(range(L)), [eye] * L, [eye] * L,
                                   [HeadMaps.identity(H, dh)] * L)
    assert all(hm.n_self == H for hm in maps)
    a = tiny64.cls_feature(images64)
    b = tiny64.cls_feature(images64, m)
    assert (a - b).abs().max().item() < 1e-10


# ---------------------------------------------------------------- projection
@pytest.mark.parametrize("gauge", [True, False])
def test_projection_idempotent_and_share_zero(tiny64, gauge):
    blk = tiny64.model.blocks[1]
    H, dh = attn_dims(blk)
    d = tiny64.embed_dim
    A_in = involution_fixing_ones(d, 36, seed=3)
    A_out = involution_fixing_ones(d, 28, seed=4)
    hm = random_head_maps([2, 1, 0, 3], dh, seed=5)
    hm.check()
    f = fold_attn(blk)
    e = project_attn(**f, A_in=A_in, A_out=A_out, head_maps=hm, gauge=gauge)
    hm2 = hm.gauged() if gauge else hm
    e2 = project_attn(**e, A_in=A_in, A_out=A_out, head_maps=hm2, gauge=False)
    for k in e:
        assert (e[k] - e2[k]).abs().max().item() < 1e-12, k
    shares = breaking_shares_from(e, e2)
    assert all(v < 1e-24 for v in shares.values())
    # Random (unstructured) weights lose a sizeable share; twin-pair matrices lose about half.
    s = attention_breaking_shares(blk, A_in, A_out, hm)
    assert 0.2 < s["s_all"] < 0.8


def test_free_param_count_matches_projection_rank():
    """The projection is linear in all weights; its rank is the equivariant dimension."""
    H, dh, d = 4, 2, 6
    A_in = random_involution(d, 4, seed=1)      # tr 2
    A_out = random_involution(d, 1, seed=2)     # tr -4
    hm = random_head_maps([2, 1, 0, 3], dh, seed=3, self_plus={1: (1, 0), 3: (2, 1)})
    shapes = {"Wq": (H * dh, d), "Wk": (H * dh, d), "Wv": (H * dh, d), "bq": (H * dh,),
              "bk": (H * dh,), "bv": (H * dh,), "Wo": (d, H * dh), "bo": (d,)}
    sizes = {k: int(np.prod(s)) for k, s in shapes.items()}
    n = sum(sizes.values())

    def unpack(vec):
        out, i = {}, 0
        for k, s in shapes.items():
            out[k] = torch.tensor(vec[i:i + sizes[k]].reshape(s))
            i += sizes[k]
        return out

    for gauge in (False, True):
        T = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1.0
            p = project_attn(**unpack(e), A_in=A_in, A_out=A_out, head_maps=hm, gauge=gauge)
            T[:, j] = np.concatenate([p[k].numpy().ravel() for k in shapes])
        cnt = attention_free_params(d, hm, A_in, A_out)
        assert np.linalg.matrix_rank(T, tol=1e-9) == cnt["total"]
        assert cnt["original"] == n
        if not gauge:
            assert np.allclose(T @ T, T)        # a projection
            assert np.allclose(T, T.T)          # orthogonal (nearest point in Frobenius norm)


# ---------------------------------------------------------------- exact equivariance
@pytest.mark.parametrize("gauge", [True, False])
@pytest.mark.parametrize("two_ops", [False, True])
def test_projected_attention_branch_is_exactly_equivariant(tiny64, gauge, two_ops):
    d = tiny64.embed_dim
    H, dh = attn_dims(tiny64.model.blocks[0])
    A_in = involution_fixing_ones(d, 40, seed=7)
    A_out = involution_fixing_ones(d, 22, seed=8) if two_ops else A_in
    hm = random_head_maps([3, 1, 2, 0], dh, seed=9)
    m = copy.deepcopy(tiny64.model)
    new_maps = apply_attention_surgery(m, [0], [A_in], [A_out], [hm], gauge=gauge)
    blk = m.blocks[0]
    x = tokens((5, 17, d), 1)
    Ai, Ao = torch.tensor(A_in), torch.tensor(A_out)
    assert (attn_branch(blk, x @ Ai) - attn_branch(blk, x) @ Ao).abs().max().item() < 1e-10
    # Head relations hold for the written weights: y~_{sigma h} = B[h] y_h (rows: y_h B[h]^T).
    hmn = new_maps[0]
    D = H * dh
    y = blk.attn.qkv(blk.norm1(x)).reshape(5, 17, 3, H, dh)
    yt = blk.attn.qkv(blk.norm1(x @ Ai)).reshape(5, 17, 3, H, dh)
    for h in range(H):
        g = int(hmn.sigma[h])
        Bt, Ct = torch.tensor(hmn.B[h]), torch.tensor(hmn.C[h])
        for t, M in ((0, Bt), (1, Bt), (2, Ct)):
            assert (yt[:, :, t, g] - y[:, :, t, h] @ M.T).abs().max().item() < 1e-10
    assert D == tiny64.embed_dim
    # Breaking shares of the new block are zero; the original block is not equivariant.
    s = attention_breaking_shares(blk, A_in, A_out, hmn)
    assert s["s_all"] < 1e-24 and s["s_bo"] < 1e-24
    ob = tiny64.model.blocks[0]
    assert (attn_branch(ob, x @ Ai) - attn_branch(ob, x) @ Ao).abs().max().item() > 1e-3


def test_gauge_does_not_change_function(tiny64):
    from mirror.attention import gauge_twin_heads, write_absorbed_attn

    blk = copy.deepcopy(tiny64.model.blocks[2])
    hm = random_head_maps([1, 0, 3, 2], attn_dims(blk)[1], seed=11)
    x = tokens((3, 10, tiny64.embed_dim), 2)
    before = attn_branch(blk, x)
    g, hmg = gauge_twin_heads(fold_attn(blk), hm)
    write_absorbed_attn(blk, **g)
    assert (attn_branch(blk, x) - before).abs().max().item() < 1e-10
    assert all(np.allclose(hmg.B[h], np.eye(hm.dh)) for h in range(hm.H))


# ---------------------------------------------------------------- head pairing and maps
def _planted_stats(tiny64, block_idx, sigma, seed, self_plus=None, n_tokens=(6, 40)):
    """Model block projected onto a planted head structure; stats from x and x A."""
    d = tiny64.embed_dim
    H, dh = attn_dims(tiny64.model.blocks[block_idx])
    A = involution_fixing_ones(d, 34, seed=seed)
    hm = random_head_maps(sigma, dh, seed=seed + 1, self_plus=self_plus)
    m = copy.deepcopy(tiny64.model)
    apply_attention_surgery(m, [block_idx], [A], [A], [hm], gauge=False)
    blk = m.blocks[block_idx]
    st = AttnStats.for_model(m)
    x = tokens((*n_tokens, d), seed + 2)
    with torch.no_grad():
        st.update(block_idx, blk.attn.qkv(blk.norm1(x)), blk.attn.qkv(blk.norm1(x @ torch.tensor(A))))
    return st, hm


def test_head_pairing_recovers_planted_swap(tiny64):
    for sigma, seed in (([2, 1, 0, 3], 20), ([1, 0, 3, 2], 30), ([0, 1, 2, 3], 40)):
        st, hm = _planted_stats(tiny64, 1, sigma, seed)
        mom = st.moments(1)
        fit = pair_and_fit_heads(mom)
        assert np.array_equal(fit.sigma, np.array(sigma)), (fit.sigma, sigma)
        # Planted orbits fit exactly: score 1 on the chosen entries.
        S = fit.scores
        assert np.allclose(S[np.arange(4), fit.sigma], 1.0, atol=1e-10)
        err = head_map_errors(mom, fit)
        assert err["qk"].max() < 1e-10 and err["v"].max() < 1e-10
        # Same answer from the networkx matcher.
        assert np.array_equal(match_heads(S, method="networkx"), fit.sigma)


def test_fitted_head_maps_recover_planted_B_and_C(tiny64):
    st, hm = _planted_stats(tiny64, 2, [3, 1, 2, 0], seed=50, self_plus={1: (5, 11), 2: (16, 1)})
    fit = fit_head_maps(st.moments(2), hm.sigma)
    fit.check()
    assert np.abs(fit.B - hm.B).max() < 1e-8
    assert np.abs(fit.C - hm.C).max() < 1e-8


def test_fit_on_synthetic_moments_with_noise():
    """Planted data straight into AttnStats (no model): exact without noise, close with noise."""
    rng = np.random.default_rng(0)
    H, dh, N = 6, 4, 2000
    sigma = np.array([3, 1, 5, 0, 4, 2])
    hm = random_head_maps(sigma, dh, seed=1)
    y = rng.standard_normal((N, 3, H, dh)) * rng.uniform(0.5, 2.0, size=(1, 3, H, 1)) + 0.3
    yt = np.zeros_like(y)
    for h in range(H):
        g = sigma[h]
        yt[:, 0, g] = y[:, 0, h] @ hm.B[h].T
        yt[:, 1, g] = y[:, 1, h] @ hm.B[h].T
        yt[:, 2, g] = y[:, 2, h] @ hm.C[h].T
    for noise, tol in ((0.0, 1e-9), (0.05, 0.05)):
        st = AttnStats(1, H, dh)
        z = yt + noise * rng.standard_normal(yt.shape)
        st.update(0, torch.tensor(y.reshape(N, -1)), torch.tensor(z.reshape(N, -1)))
        fit = pair_and_fit_heads(st.moments(0))
        assert np.array_equal(fit.sigma, sigma)
        assert np.abs(fit.B - hm.B).max() < tol and np.abs(fit.C - hm.C).max() < tol


def _all_involutions(H):
    def rec(rest):
        if not rest:
            yield []
            return
        i, others = rest[0], rest[1:]
        for p in rec(others):
            yield [(i, i)] + p
        for k, j in enumerate(others):
            for p in rec(others[:k] + others[k + 1:]):
                yield [(i, j)] + p

    for pairs in rec(list(range(H))):
        s = np.arange(H)
        for a, b in pairs:
            s[a], s[b] = b, a
        yield s


def test_match_heads_is_exact():
    rng = np.random.default_rng(3)
    H = 7
    invs = list(_all_involutions(H))
    assert len(invs) == 232                      # number of involutions on 7 elements
    for _ in range(20):
        S = rng.uniform(0, 1, (H, H))
        S = 0.5 * (S + S.T)
        best = max(S[np.arange(H), s].sum() for s in invs)
        for method in ("dp", "networkx"):
            sig = match_heads(S, method=method)
            assert np.array_equal(sig[sig], np.arange(H))
            assert abs(S[np.arange(H), sig].sum() - best) < 1e-12


def test_head_scores_are_symmetric_and_bounded(tiny64):
    st, _ = _planted_stats(tiny64, 0, [2, 1, 0, 3], seed=60)
    S = head_pair_scores(st.moments(0))
    assert np.allclose(S, S.T)
    assert S.min() >= 0 and S.max() <= 1 + 1e-12


# ---------------------------------------------------------------- hooks
def test_paired_hooks_match_manual(tiny64):
    g = torch.Generator().manual_seed(4)
    imgs = torch.randint(0, 256, (2, 3, 56, 56), generator=g, dtype=torch.uint8)
    st = AttnStats.for_model(tiny64.model)
    with AttnPairedHooks(tiny64, tiny64.model, st.update):
        run_paired(tiny64, [(imgs, torch.zeros(2))], "cpu")
    x = normalize(imgs, torch.float64)
    t = tiny64.layer0_tokens(torch.cat([x, mirror(x)], 0))
    ref = AttnStats.for_model(tiny64.model)
    with torch.no_grad():
        for b, blk in enumerate(tiny64.model.blocks[:2]):
            y = blk.attn.qkv(blk.norm1(t))
            ref.update(b, y[:2], unflip_tokens(y[2:], tiny64.grid_hw, tiny64.n_prefix_tokens))
            t = blk(t)
    for b in range(2):
        assert torch.allclose(st.cross[b], ref.cross[b], rtol=1e-12, atol=1e-9)
        assert torch.allclose(st.sq_t[b], ref.sq_t[b], rtol=1e-12, atol=1e-9)
        assert st.n[b] == 2 * tiny64.n_tokens
    assert st.n[2] > 0 and st.n[3] > 0
