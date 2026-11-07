"""Folding, projection onto mirror-equivariant MLP weights, breaking share and surgery.

Conventions (torch): fc1.weight W1 is [h, d], fc2.weight W2 is [d, h], column vectors.
A is the d x d residual involution, P the h x h neuron permutation (stored as an index array
`perm`, P[i, perm[i]] = 1). Equivariance of fc1 -> GELU -> fc2 needs
    P W1 = W1 A,  P b1 = b1,  A W2 = W2 P,  A b2 = b2.
With index arrays: P M = M[perm], M P = M[:, perm] (perm is an involution).

Folding (exact): W1' = W1 diag(gamma2), b1' = b1 + W1 beta2 (norm2 affine into fc1);
W2' = diag(ls2) W2, b2' = ls2 * b2 (LayerScale into fc2). The absorbed block has norm2 affine
set to identity and ls2 set to ones, and computes exactly the same function.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------------------------
# Folding
# ----------------------------------------------------------------------------------------------
def _ls_gamma(block: nn.Module):
    ls = getattr(block, "ls2", None)
    g = getattr(ls, "gamma", None)
    return g if isinstance(g, torch.Tensor) else None


@torch.no_grad()
def fold_mlp(block: nn.Module, dtype: torch.dtype = torch.float64) -> dict:
    """Folded MLP weights W1', b1', W2', b2' of one block (in `dtype`, on the block's device)."""
    fc1, fc2, norm2 = block.mlp.fc1, block.mlp.fc2, block.norm2
    W1 = fc1.weight.to(dtype)
    h, d = W1.shape
    b1 = fc1.bias.to(dtype) if fc1.bias is not None else W1.new_zeros(h)
    gamma = norm2.weight.to(dtype) if getattr(norm2, "weight", None) is not None else W1.new_ones(d)
    beta = norm2.bias.to(dtype) if getattr(norm2, "bias", None) is not None else W1.new_zeros(d)
    W2 = fc2.weight.to(dtype)
    b2 = fc2.bias.to(dtype) if fc2.bias is not None else W2.new_zeros(d)
    ls = _ls_gamma(block)
    ls = ls.to(dtype) if ls is not None else W2.new_ones(d)
    return {
        "W1": W1 * gamma[None, :],
        "b1": b1 + W1 @ beta,
        "W2": ls[:, None] * W2,
        "b2": ls * b2,
    }


@torch.no_grad()
def write_absorbed_mlp(block: nn.Module, W1, b1, W2, b2) -> None:
    """Write folded weights and reset norm2 affine and ls2 to the identity."""
    fc1, fc2, norm2 = block.mlp.fc1, block.mlp.fc2, block.norm2
    if fc1.bias is None or fc2.bias is None:
        raise ValueError("absorbed form needs fc1 and fc2 biases")
    fc1.weight.copy_(W1.to(fc1.weight))
    fc1.bias.copy_(b1.to(fc1.bias))
    fc2.weight.copy_(W2.to(fc2.weight))
    fc2.bias.copy_(b2.to(fc2.bias))
    if getattr(norm2, "weight", None) is not None:
        norm2.weight.fill_(1.0)
    if getattr(norm2, "bias", None) is not None:
        norm2.bias.zero_()
    g = _ls_gamma(block)
    if g is not None:
        g.fill_(1.0)


@torch.no_grad()
def absorb_block(block: nn.Module) -> None:
    write_absorbed_mlp(block, **fold_mlp(block))


# ----------------------------------------------------------------------------------------------
# Projection and breaking share
# ----------------------------------------------------------------------------------------------
def _as(t, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(t, dtype=like.dtype, device=like.device)


def project_mlp(W1, b1, W2, b2, A, perm) -> dict:
    """Nearest mirror-equivariant MLP weights for involutions A (residual) and perm (neurons)."""
    A = _as(A, W1)
    b1, W2, b2 = _as(b1, W1), _as(W2, W1), _as(b2, W1)
    p = torch.as_tensor(np.asarray(perm), dtype=torch.long, device=W1.device)
    return {
        "W1": 0.5 * (W1 + W1[p] @ A),
        "b1": 0.5 * (b1 + b1[p]),
        "W2": 0.5 * (W2 + (A @ W2)[:, p]),
        "b2": 0.5 * (b2 + A @ b2),
    }


def breaking_share(W: torch.Tensor, W_eq: torch.Tensor) -> float:
    """||W - W_eq||^2 / ||W||^2 (0 for an equivariant matrix, about 0.5 for random structure)."""
    num = float(((W - W_eq) ** 2).sum())
    den = float((W ** 2).sum())
    return num / den if den > 0 else float("nan")


def perm_trace(perm) -> int:
    perm = np.asarray(perm)
    return int((perm == np.arange(len(perm))).sum())


def equivariant_param_count(h: int, d: int, tr_P: int, tr_A: int) -> int:
    """Dimension of the equivariant subspace of h x d matrices: (h d + tr(P) tr(A)) / 2.

    Equals h+ d+ + h- d-, the multiply-adds of the block-diagonal form.
    """
    num = h * d + tr_P * tr_A
    assert num % 2 == 0, (h, d, tr_P, tr_A)
    return num // 2


def svd_rank_for(h: int, d: int, param_count: int) -> int:
    """Rank whose factorisation r (h + d) does not exceed param_count."""
    return int(math.floor(param_count / (h + d)))


def mlp_breaking_shares(block: nn.Module, A, perm) -> dict:
    f = fold_mlp(block)
    e = project_mlp(**f, A=A, perm=perm)
    return {
        "s1": breaking_share(f["W1"], e["W1"]),
        "s2": breaking_share(f["W2"], e["W2"]),
        "sb1": breaking_share(f["b1"], e["b1"]),
        "sb2": breaking_share(f["b2"], e["b2"]),
    }


# ----------------------------------------------------------------------------------------------
# Surgery and baselines
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def apply_surgery(model: nn.Module, blocks: list[int], A, P_list: list) -> None:
    """Replace the MLPs of `blocks` by their projected, absorbed equivariant versions (in place)."""
    if len(blocks) != len(P_list):
        raise ValueError("one permutation per block")
    for b, perm in zip(blocks, P_list):
        blk = model.blocks[b]
        write_absorbed_mlp(blk, **project_mlp(**fold_mlp(blk), A=A, perm=perm))


def project_mlp_two(W1, b1, W2, b2, A_in, A_out, perm) -> dict:
    """Equivariant projection when input and output streams have different involutions:
    P W1 = W1 A_in, P b1 = b1 (fc1 side) and A_out W2 = W2 P, A_out b2 = b2 (fc2 side), so that
    MLP(x A_in) = MLP(x) A_out."""
    e_in = project_mlp(W1, b1, W2, b2, A=A_in, perm=perm)
    e_out = project_mlp(W1, b1, W2, b2, A=A_out, perm=perm)
    return {"W1": e_in["W1"], "b1": e_in["b1"], "W2": e_out["W2"], "b2": e_out["b2"]}


@torch.no_grad()
def apply_surgery_two(model: nn.Module, blocks: list[int], A_in_list: list, A_out_list: list, P_list: list) -> None:
    """Surgery with a per-block input operator (fc1) and output operator (fc2), in place."""
    for b, A_in, A_out, perm in zip(blocks, A_in_list, A_out_list, P_list):
        blk = model.blocks[b]
        write_absorbed_mlp(blk, **project_mlp_two(**fold_mlp(blk), A_in=A_in, A_out=A_out, perm=perm))


def svd_truncate(W: torch.Tensor, r: int) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vh[:r]


@torch.no_grad()
def apply_svd_baseline(model: nn.Module, blocks: list[int], ranks) -> None:
    """Rank-r truncation of folded fc1 and fc2 in `blocks` (biases kept), in place."""
    if isinstance(ranks, int):
        ranks = [ranks] * len(blocks)
    for b, r in zip(blocks, ranks):
        blk = model.blocks[b]
        f = fold_mlp(blk)
        write_absorbed_mlp(blk, svd_truncate(f["W1"], int(r)), f["b1"], svd_truncate(f["W2"], int(r)), f["b2"])
