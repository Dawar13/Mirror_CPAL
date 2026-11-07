"""Tests 3 to 6: projection, parameter formula, folding, identity surgery; plus exact equivariance."""
import copy

import numpy as np
import torch

from mirror.fit import random_involution
from mirror.matching import random_pairing
from mirror.surgery import (absorb_block, apply_surgery, apply_svd_baseline, breaking_share,
                            equivariant_param_count, fold_mlp, perm_trace, project_mlp, svd_rank_for)


def perm_matrix(perm):
    P = np.zeros((len(perm), len(perm)))
    P[np.arange(len(perm)), perm] = 1.0
    return P


def involution_fixing_ones(d, d_plus, seed):
    """Random involution with A 1 = 1 (needed for LayerNorm equivariance)."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(np.column_stack([np.ones(d), rng.standard_normal((d, d - 1))]))
    signs = np.concatenate([[1.0], np.ones(d_plus - 1), -np.ones(d - d_plus)])
    A = (Q * signs) @ Q.T
    return 0.5 * (A + A.T)


# ---------------------------------------------------------------- test 3
def test_projection_breaking_share():
    rng = np.random.default_rng(0)
    h, d = 96, 48
    A = random_involution(d, d // 2, seed=1)          # tr A = 0
    perm = random_pairing(h, h // 2, seed=2)          # tr P = 0
    W0 = torch.tensor(rng.standard_normal((h, d)))
    W_eq = project_mlp(W0, torch.zeros(h), W0.T.clone(), torch.zeros(d), A, perm)["W1"]
    again = project_mlp(W_eq, torch.zeros(h), W0.T.clone(), torch.zeros(d), A, perm)["W1"]
    assert breaking_share(W_eq, again) < 1e-24       # equivariant -> share 0 (projection is idempotent)
    s_rand = breaking_share(W0, W_eq)
    assert 0.45 < s_rand < 0.55                       # random weights -> about 0.5


# ---------------------------------------------------------------- test 4
def test_param_formula_matches_projection_rank():
    d, h = 6, 8
    for d_plus, n_pairs, seed in [(3, 4, 0), (4, 2, 1), (6, 0, 2), (1, 3, 3), (5, 1, 4)]:
        A = random_involution(d, d_plus, seed=seed)
        perm = random_pairing(h, n_pairs, seed=seed + 10)
        P = perm_matrix(perm)
        # Operator on vec(W) for W -> (W + P W A) / 2; its rank is the equivariant dimension.
        T = np.zeros((h * d, h * d))
        for k in range(h * d):
            E = np.zeros(h * d)
            E[k] = 1.0
            W = E.reshape(h, d)
            T[:, k] = (0.5 * (W + P @ W @ A)).ravel()
        rank = np.linalg.matrix_rank(T, tol=1e-9)
        assert rank == equivariant_param_count(h, d, perm_trace(perm), int(round(np.trace(A))))
        # And the torch implementation is that operator.
        W = np.random.default_rng(seed).standard_normal((h, d))
        out = project_mlp(torch.tensor(W), torch.zeros(h), torch.tensor(W.T), torch.zeros(d), A, perm)
        assert np.allclose(out["W1"].numpy(), 0.5 * (W + P @ W @ A))
        assert np.allclose(out["W2"].numpy(), 0.5 * (W.T + A @ W.T @ P))


# ---------------------------------------------------------------- test 5
def test_folding_is_exact(tiny64, images64):
    t = tiny64.layer0_tokens(images64)
    for blk in tiny64.model.blocks:
        folded = copy.deepcopy(blk)
        absorb_block(folded)
        assert torch.allclose(folded.norm2.weight, torch.ones_like(folded.norm2.weight))
        assert torch.allclose(folded.ls2.gamma, torch.ones_like(folded.ls2.gamma))
        out = blk(t)
        assert (out - folded(t)).abs().max().item() < 1e-5
        # The test is not trivial: the MLP branch really contributes.
        assert (blk.ls2(blk.mlp(blk.norm2(t)))).abs().max().item() > 1e-2
        t = out


# ---------------------------------------------------------------- test 6
def test_identity_surgery_leaves_model_unchanged(tiny64, images64):
    m = copy.deepcopy(tiny64.model)
    L, d, h = tiny64.depth, tiny64.embed_dim, tiny64.mlp_hidden
    apply_surgery(m, list(range(L)), np.eye(d), [np.arange(h)] * L)
    a = tiny64.cls_feature(images64)
    b = tiny64.cls_feature(images64, m)
    assert (a - b).abs().max().item() < 1e-10


# ---------------------------------------------------------------- conventions check
def test_projected_mlp_branch_is_exactly_equivariant(tiny64):
    """After surgery with (A, P), the MLP branch satisfies f(x A) = f(x) A exactly (row vectors)."""
    d, h = tiny64.embed_dim, tiny64.mlp_hidden
    A = involution_fixing_ones(d, 40, seed=7)
    perm = random_pairing(h, 90, seed=8)
    m = copy.deepcopy(tiny64.model)
    apply_surgery(m, [0], A, [perm])
    blk = m.blocks[0]

    def branch(x):
        return blk.ls2(blk.mlp(blk.norm2(x)))

    x = torch.randn(5, 17, d, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    At = torch.tensor(A)
    assert (branch(x @ At) - branch(x) @ At).abs().max().item() < 1e-10
    # fc1 pre-activations of the mirror are the permuted originals: z(x A) = z(x) P.
    z = blk.mlp.fc1(blk.norm2(x))
    zt = blk.mlp.fc1(blk.norm2(x @ At))
    assert (zt - z[..., perm]).abs().max().item() < 1e-10
    # The original (unprojected) block is not equivariant, so the check has teeth.
    ob = tiny64.model.blocks[0]
    orig = ob.ls2(ob.mlp(ob.norm2(x @ At))) - ob.ls2(ob.mlp(ob.norm2(x))) @ At
    assert orig.abs().max().item() > 1e-3


def test_two_operator_surgery_maps_A_in_to_A_out(tiny64):
    """Per-layer variant: f(x A_in) = f(x) A_out after projection with different involutions."""
    from mirror.surgery import apply_surgery_two

    d, h = tiny64.embed_dim, tiny64.mlp_hidden
    A_in = involution_fixing_ones(d, 30, seed=11)
    A_out = involution_fixing_ones(d, 50, seed=12)
    perm = random_pairing(h, 70, seed=13)
    m = copy.deepcopy(tiny64.model)
    apply_surgery_two(m, [2], [A_in], [A_out], [perm])
    blk = m.blocks[2]
    x = torch.randn(4, 17, d, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    f = lambda t: blk.ls2(blk.mlp(blk.norm2(t)))  # noqa: E731
    assert (f(x @ torch.tensor(A_in)) - f(x) @ torch.tensor(A_out)).abs().max().item() < 1e-10


def test_svd_baseline_rank(tiny64):
    d, h = tiny64.embed_dim, tiny64.mlp_hidden
    r = svd_rank_for(h, d, equivariant_param_count(h, d, 100, 8))
    m = copy.deepcopy(tiny64.model)
    apply_svd_baseline(m, [1], [r])
    f = fold_mlp(m.blocks[1])
    assert torch.linalg.matrix_rank(f["W1"]).item() == r
    assert torch.linalg.matrix_rank(f["W2"]).item() == r
    assert r * (h + d) <= equivariant_param_count(h, d, 100, 8)
