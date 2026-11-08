"""M2: efficient implementation of mirror-converted ViT blocks in even/odd coordinates, and the
timing baselines (dense, SVD factors, MLP width pruning), all in one framework (FastViT).

Conventions (as mirror.surgery and mirror.attention): torch Linear weights W are [out, in], tokens are
rows, y = x W^T + b. A stream basis is an orthogonal d x d matrix Q; a token x (standard coordinates)
is stored as x Q. For an involution A, `Basis.of(A)` holds Q = [Q+, Q-] with A Q+ = Q+, A Q- = -Q-
(the d+ even columns first), so a stored token is [x+, x-] (even part, odd part).

Input models are fully folded (mirror.fold.fold_layernorm_full(final_norm_affine=False), as
mirror.m1.build_vexact produces): every norm is an RMSNorm without affine, which commutes with any
orthogonal Q and is therefore applied unchanged in any stream basis; LayerScale gammas are 1; the
norm affines are folded into qkv and fc1; the final affine stays after the final RMSNorm.

MLP (PROJECT_CONTEXT 5.2). With P W1 = W1 A_in, P b1 = b1, A_out W2 = W2 P, A_out b2 = b2 and
V1 = W1 Q_in, a twin pair (i, j) has rows V1[i] = [a+, a-], V1[j] = [a+, -a-] and a self-twin s has
V1[s] = [c+, 0]. MirrorMLP computes
    u = a+ . x+ + b1_i,   v = a- . x-,   pre_i = u + v,   pre_j = u - v,   pre_s = c+ . x+ + b1_s,
    g = GELU(pre),
    y+ = sum_pairs p+ (g_i + g_j) + sum_self q+ g_s + b2+,    y- = sum_pairs p- (g_i - g_j),
where Q_out^T W2[:, i] = [p+; p-], Q_out^T W2[:, j] = [p+; -p-], Q_out^T W2[:, s] = [q+; 0]. So fc1 is
two dense maps (even part to [twin-pair u rows; self-twin rows], odd part to twin-pair v rows) and
fc2 is two dense maps back to the even and odd outputs: h+ d+ + h- d- multiply-adds per matrix
(h+ = pairs + self-twins, h- = pairs), which equals the free-parameter count (h d + tr(P) tr(A)) / 2.

Attention (PROJECT_CONTEXT 5.4), with the head maps valid after the twin gauge (B = C = I on twin
pairs, symmetric involutions on self-heads; attention.apply_attention_surgery returns them):
  * twin heads (h, g): Wq^g = Wq^h A_in (k and v alike), so q_h = u + v and q_g = u - v with u from
    the even and v from the odd part of the input: a pair costs one head;
  * self-head s: block-diagonal after rotating its head-internal basis by the eigenbasis U of B[s]
    (q and k together, so the logits are unchanged) and V of C[s] (v; Wo absorbs V);
  * the output projection reads (o_h + o_g, o_h - o_g) for pairs and the even and odd parts of the
    self-heads, and writes the even and odd outputs;
  * token mixing softmax(q k^T / sqrt(dh)) v is unchanged (scaled_dot_product_attention).
Multiply-adds per token: the free-parameter counts of attention.attention_free_params.

Stream bases (`bases[l]` is the basis of the stream entering block l; bases[L] the final stream):
  * the rotation into bases[0] is folded into the patch embedding, CLS token, registers, position and
    mask embeddings (free);
  * block b reads in bases[b] and its MLP writes in bases[b + 1]. If they differ, the stream is
    rotated once, x <- x Q_b^T Q_(b+1) (d^2 multiply-adds per token, counted, not folded);
  * dense blocks are rewritten in their basis (free);
  * the rotation out of bases[L] is applied to the CLS feature only, before the final affine
    (with the V-exact even head, y (I + A*)/2, only the even rows are kept).
V-exact uses one basis for every layer, so it needs no per-token rotation. V-local uses
bases[l] = eig(A_l): one rotation per converted block. If the attention output Wo was projected with
an operator that does not match bases[b] (M1 chose A_(b+1)), Wo cannot be structured in the block's
basis: wo_mode "fold" stores Wo dense in bases[b] (the rotation folded into it, no structure kept),
wo_mode "rotate" keeps Wo structured in eig(A_out) and rotates the attention output (a second d^2
rotation per token).

Alignment: with align > 1 every GEMM dimension of the structured, low-rank and pruned maps is padded
with zeros to a multiple of `align` (tensor-core friendly). Padding changes no output. Analytic
multiply-adds and parameter counts never include padding; executed multiply-adds are reported
separately (`macs_token_exec`).
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import HeadMaps, attention_free_params, check_attn_supported
from .fold import RMSNorm as FoldRMSNorm
from .heal import exact_involution
from .surgery import equivariant_param_count

__all__ = [
    "Basis", "BlockSpec", "eig_basis", "FastRMSNorm", "DenseAttention", "DenseMLP", "MirrorAttention",
    "MirrorMLP", "LowRankAttention", "LowRankMLP", "FastBlock", "FinalHead", "Embed", "FastViT",
    "build_mirror_mlp", "build_mirror_attention", "build_dense_attention", "build_dense_mlp",
    "build_lowrank_attention", "build_lowrank_mlp", "build_pruned_mlp", "build_fast_vit", "fast_vit_cost",
    "is_folded", "ceil_to", "floor_to", "prune_keep", "build_block",
]

_TOKEN_PARAMS = ("cls_token", "register_tokens", "reg_token", "pos_embed", "mask_token")


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
def ceil_to(n: int, a: int) -> int:
    return int(-(-int(n) // int(a)) * int(a)) if a > 1 else int(n)


def floor_to(n: int, a: int) -> int:
    return int((int(n) // int(a)) * int(a)) if a > 1 else int(n)


def _t(x, device, dtype=torch.float64) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def _pad2(M: torch.Tensor, rows: int | None = None, cols: int | None = None, row_front: int = 0,
          col_front: int = 0) -> torch.Tensor:
    """Zero-pad a 2-D tensor to rows x cols, with `row_front` zero rows and `col_front` zero columns first."""
    r, c = M.shape
    rows = r + row_front if rows is None else int(rows)
    cols = c + col_front if cols is None else int(cols)
    out = M.new_zeros(rows, cols)
    out[row_front:row_front + r, col_front:col_front + c] = M
    return out


def _pad1(v: torch.Tensor, n: int) -> torch.Tensor:
    out = v.new_zeros(int(n))
    out[:v.shape[0]] = v
    return out


def _sq(x: torch.Tensor) -> float:
    return float((x.double() ** 2).sum())


_EIG_CACHE: dict = {}


def eig_basis(A, tol: float = 1e-6) -> tuple[np.ndarray, int]:
    """(Q, d+) for a symmetric orthogonal involution A: Q = [Q+, Q-], even eigenvectors first (float64).
    Cached by content (eigh of 768 x 768 is slow on a contended CPU)."""
    A = np.asarray(A, dtype=np.float64)
    key = (A.shape, hashlib.sha1(np.ascontiguousarray(A).tobytes()).hexdigest())
    if key not in _EIG_CACHE:
        lam, V = np.linalg.eigh(0.5 * (A + A.T))
        if np.max(np.abs(np.abs(lam) - 1.0)) > tol:
            raise ValueError("operator is not an orthogonal involution")
        pos = lam > 0
        _EIG_CACHE[key] = (np.ascontiguousarray(np.concatenate([V[:, pos], V[:, ~pos]], axis=1)), int(pos.sum()))
    Q, dp = _EIG_CACHE[key]
    return Q.copy(), dp


@dataclass
class Basis:
    """Stream basis: Q (None = identity, no even/odd split) and the number of even coordinates."""

    Q: np.ndarray | None
    dp: int

    @classmethod
    def identity(cls, d: int) -> "Basis":
        return cls(None, int(d))

    @classmethod
    def of(cls, A) -> "Basis":
        Q, dp = eig_basis(exact_involution(A))
        return cls(Q, dp)

    def mat(self, d: int) -> np.ndarray:
        return np.eye(d) if self.Q is None else self.Q

    def same(self, other: "Basis") -> bool:
        if self.Q is None or other.Q is None:
            return self.Q is None and other.Q is None
        return self.dp == other.dp and (self.Q is other.Q or np.array_equal(self.Q, other.Q))

    def aligned_with(self, A, tol: float = 1e-8) -> bool:
        """True if Q^T A Q = diag(+1 (d+ times), -1): the basis is an eigenbasis of A with this split."""
        if self.Q is None:
            return False
        A = np.asarray(A, dtype=np.float64)
        d = A.shape[0]
        target = np.concatenate([np.ones(self.dp), -np.ones(d - self.dp)])
        return float(np.abs(self.Q.T @ A @ self.Q - np.diag(target)).max()) < tol


@dataclass
class BlockSpec:
    """How one block is built.

    attn: ("dense",) | ("mirror", A_in, A_out, head_maps_after_gauge) | ("svd", (r_q, r_k, r_v, r_o))
    mlp:  ("dense",) | ("mirror", A_in, A_out, perm) | ("svd", (r_fc1, r_fc2)) | ("prune", h_keep)
    """

    attn: tuple = ("dense",)
    mlp: tuple = ("dense",)

    @property
    def converted(self) -> bool:
        return self.attn[0] != "dense" or self.mlp[0] != "dense"


class _Split:
    """Even/odd slicing of a d-wide stream with alignment: the even map reads x[..., :ke] (zero weight
    columns beyond d+), the odd map reads x[..., ko:] (zero weight columns before d+); outputs are padded
    the same way and recombined as cat(y_even[..., :d+], y_odd[..., cut:])."""

    def __init__(self, d: int, dp: int, align: int):
        self.d, self.dp, self.align = int(d), int(dp), int(align)
        self.ke = min(self.d, ceil_to(self.dp, self.align))
        self.ko = min(floor_to(self.dp, self.align), self.dp)
        self.cut = self.dp - self.ko

    def in_even(self, E: torch.Tensor) -> torch.Tensor:          # [rows, d+] -> [rows, ke]
        return _pad2(E, cols=self.ke)

    def in_odd(self, O: torch.Tensor) -> torch.Tensor:           # [rows, d-] -> [rows, d - ko]
        return _pad2(O, col_front=self.cut)

    def out_even(self, E: torch.Tensor) -> torch.Tensor:         # [d+, cols] -> [ke, cols]
        return _pad2(E, rows=self.ke)

    def out_odd(self, O: torch.Tensor) -> torch.Tensor:          # [d-, cols] -> [d - ko, cols]
        return _pad2(O, row_front=self.cut)


def _cost(kind: str, matrix: int, bias: int, macs: int, macs_exec: int, dense_matrix: int, **extra) -> dict:
    return {"kind": kind, "matrix_params": int(matrix), "bias_params": int(bias), "macs_token": int(macs),
            "macs_token_exec": int(macs_exec), "dense_matrix_params": int(dense_matrix), **extra}


# ----------------------------------------------------------------------------------------------
# Modules
# ----------------------------------------------------------------------------------------------
class FastRMSNorm(nn.Module):
    """x / sqrt(mean(x^2) + eps), computed in float32 for half-precision inputs (as mirror.fold.RMSNorm)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.dim, self.eps = int(dim), float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        if hasattr(F, "rms_norm"):
            y = F.rms_norm(xf, (self.dim,), eps=self.eps)
        else:  # pragma: no cover (torch < 2.4)
            y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y.to(x.dtype)

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.eps:g}"


def _param(t: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(t.contiguous(), requires_grad=False)


def _sdpa(q, k, v, scale: float):
    return F.scaled_dot_product_attention(q, k, v, scale=scale)


class DenseAttention(nn.Module):
    """Plain fused-qkv attention in a given stream basis."""

    def __init__(self, Wqkv, bqkv, Wo, bo, H: int, dh: int, scale: float, cost: dict):
        super().__init__()
        self.qkv_w, self.qkv_b, self.proj_w, self.proj_b = _param(Wqkv), _param(bqkv), _param(Wo), _param(bo)
        self.H, self.dh, self.scale, self.cost = int(H), int(dh), float(scale), cost

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = F.linear(x, self.qkv_w, self.qkv_b).unflatten(-1, (3, self.H, self.dh)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        o = _sdpa(q, k, v, self.scale).transpose(1, 2).flatten(2)
        return F.linear(o, self.proj_w, self.proj_b)


class LowRankAttention(nn.Module):
    """SVD baseline: q, k, v as U (V x) with the three V factors stacked in one GEMM; Wo as Uo (Vo o)."""

    def __init__(self, V_qkv, Uq, Uk, Uv, bq, bk, bv, Vo, Uo, bo, ranks_pad, H: int, dh: int, scale: float,
                 cost: dict):
        super().__init__()
        self.V_qkv, self.Uq, self.Uk, self.Uv = _param(V_qkv), _param(Uq), _param(Uk), _param(Uv)
        self.bq, self.bk, self.bv, self.Vo, self.Uo, self.bo = (_param(t) for t in (bq, bk, bv, Vo, Uo, bo))
        rq, rk, _ = ranks_pad
        self.s1, self.s2 = int(rq), int(rq + rk)
        self.H, self.dh, self.scale, self.cost = int(H), int(dh), float(scale), cost

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.linear(x, self.V_qkv)
        q = F.linear(z[..., :self.s1], self.Uq, self.bq)
        k = F.linear(z[..., self.s1:self.s2], self.Uk, self.bk)
        v = F.linear(z[..., self.s2:], self.Uv, self.bv)
        qkv = torch.cat([q, k, v], dim=-1).unflatten(-1, (3, self.H, self.dh)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        o = _sdpa(q, k, v, self.scale).transpose(1, 2).flatten(2)
        return F.linear(F.linear(o, self.Vo), self.Uo, self.bo)


class MirrorAttention(nn.Module):
    """Even/odd attention (see the module docstring). Heads are ordered [pair heads h, pair heads g,
    self-heads]; q, k, v of all heads come from two GEMMs (even rows and odd rows)."""

    def __init__(self, We, be, Wodd, n_pairs: int, n_self: int, H: int, dh: int, scale: float, sin: _Split,
                 rows_e: int, rows_o: int, self_idx, out: dict, cost: dict):
        super().__init__()
        self.We, self.be, self.Wodd = _param(We), _param(be), _param(Wodd)
        self.np_, self.ns, self.H, self.dh, self.scale = int(n_pairs), int(n_self), int(H), int(dh), float(scale)
        self.ke, self.ko = sin.ke, sin.ko
        self.P = 3 * self.np_ * self.dh
        self.re, self.ro = int(rows_e), int(rows_o)
        self.register_buffer("self_idx", self_idx)
        self.out_dense = bool(out["dense"])
        if self.out_dense:
            self.Wd, self.bd = _param(out["W"]), _param(out["b"])
        else:
            self.Oe, self.obe, self.Oo = _param(out["Oe"]), _param(out["be"]), _param(out["Oo"])
            self.register_buffer("oe_idx", out["e_idx"])
            self.register_buffer("oo_idx", out["o_idx"])
            self.pad_e, self.pad_o = int(out["pad_e"]), int(out["pad_o"])
            self.dpo, self.cut = int(out["dpo"]), int(out["cut"])
        self.cost = cost

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ze = F.linear(x[..., :self.ke], self.We, self.be)
        zo = F.linear(x[..., self.ko:], self.Wodd)
        parts = []
        if self.np_:
            u = ze[..., :self.P].unflatten(-1, (3, self.np_, self.dh))
            v = zo[..., :self.P].unflatten(-1, (3, self.np_, self.dh))
            parts += [u + v, u - v]
        if self.ns:
            c = torch.cat([ze[..., self.P:self.re], zo[..., self.P:self.ro]], dim=-1)
            parts.append(c.index_select(-1, self.self_idx).unflatten(-1, (3, self.ns, self.dh)))
        heads = torch.cat(parts, dim=-2) if len(parts) > 1 else parts[0]          # [B, T, 3, H, dh]
        q, k, v = heads.permute(2, 0, 3, 1, 4).unbind(0)
        o = _sdpa(q, k, v, self.scale).transpose(1, 2)                            # [B, T, H, dh]
        if self.out_dense:
            return F.linear(o.flatten(2), self.Wd, self.bd)
        ein, oin = [], []
        if self.np_:
            oh, og = o[:, :, :self.np_], o[:, :, self.np_:2 * self.np_]
            ein.append((oh + og).flatten(2))
            oin.append((oh - og).flatten(2))
        if self.ns:
            os_ = o[:, :, 2 * self.np_:].flatten(2)
            ein.append(os_.index_select(-1, self.oe_idx))
            oin.append(os_.index_select(-1, self.oo_idx))
        ein = torch.cat(ein, dim=-1) if len(ein) > 1 else ein[0]
        oin = torch.cat(oin, dim=-1) if len(oin) > 1 else oin[0]
        if self.pad_e:
            ein = F.pad(ein, (0, self.pad_e))
        if self.pad_o:
            oin = F.pad(oin, (0, self.pad_o))
        ye = F.linear(ein, self.Oe, self.obe)
        yo = F.linear(oin, self.Oo)
        return torch.cat([ye[..., :self.dpo], yo[..., self.cut:]], dim=-1)


class DenseMLP(nn.Module):
    """fc1 -> act -> fc2 (also the pruned baseline, with a smaller hidden width)."""

    def __init__(self, W1, b1, W2, b2, act: nn.Module, cost: dict):
        super().__init__()
        self.W1, self.b1, self.W2, self.b2 = _param(W1), _param(b1), _param(W2), _param(b2)
        self.act, self.cost = act, cost

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(self.act(F.linear(x, self.W1, self.b1)), self.W2, self.b2)


class LowRankMLP(nn.Module):
    def __init__(self, V1, U1, b1, V2, U2, b2, act: nn.Module, cost: dict):
        super().__init__()
        self.V1, self.U1, self.b1, self.V2, self.U2, self.b2 = (_param(t) for t in (V1, U1, b1, V2, U2, b2))
        self.act, self.cost = act, cost

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.act(F.linear(F.linear(x, self.V1), self.U1, self.b1))
        return F.linear(F.linear(g, self.V2), self.U2, self.b2)


class MirrorMLP(nn.Module):
    """Twin butterfly MLP (see the module docstring). Hidden order: [pairs (u + v), pairs (u - v), self-twins];
    dummy (all-zero) pairs and self-twins pad the widths to the alignment."""

    def __init__(self, E1, b1, O1, E2, b2, O2, act: nn.Module, sin: _Split, sout: _Split, n_pairs_pad: int,
                 cost: dict):
        super().__init__()
        self.E1, self.b1, self.O1, self.E2, self.b2, self.O2 = (_param(t) for t in (E1, b1, O1, E2, b2, O2))
        self.act = act
        self.ke, self.ko, self.npad = sin.ke, sin.ko, int(n_pairs_pad)
        self.dpo, self.cut = sout.dp, sout.cut
        self.cost = cost

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ze = F.linear(x[..., :self.ke], self.E1, self.b1)
        zo = F.linear(x[..., self.ko:], self.O1)
        u = ze[..., :self.npad]
        ga, gb = self.act(u + zo), self.act(u - zo)
        ein = torch.cat([ga + gb, self.act(ze[..., self.npad:])], dim=-1)
        ye = F.linear(ein, self.E2, self.b2)
        yo = F.linear(ga - gb, self.O2)
        return torch.cat([ye[..., :self.dpo], yo[..., self.cut:]], dim=-1)


class FastBlock(nn.Module):
    """x <- x + attn(norm1(x)) [optionally rotated]; x <- x R + mlp(norm2(x)) (R: basis change, optional)."""

    def __init__(self, d: int, eps1: float, eps2: float, attn: nn.Module, mlp: nn.Module, rot=None, attn_rot=None):
        super().__init__()
        self.norm1, self.attn, self.norm2, self.mlp = FastRMSNorm(d, eps1), attn, FastRMSNorm(d, eps2), mlp
        self.register_buffer("rot_t", None if rot is None else rot.T.contiguous())
        self.register_buffer("attn_rot_t", None if attn_rot is None else attn_rot.T.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.attn(self.norm1(x))
        if self.attn_rot_t is not None:
            a = F.linear(a, self.attn_rot_t)
        x = x + a
        m = self.mlp(self.norm2(x))
        if self.rot_t is not None:
            x = F.linear(x, self.rot_t)
        return x + m

    @property
    def rotation_macs_token(self) -> int:
        return sum(int(t.numel()) for t in (self.rot_t, self.attn_rot_t) if t is not None)


class FinalHead(nn.Module):
    """CLS feature: RMSNorm of the CLS token, rotation out of the stream basis (rows kept: m), final affine."""

    def __init__(self, d: int, eps: float, weight, bias, out_map=None):
        super().__init__()
        self.norm = FastRMSNorm(d, eps)
        self.weight, self.bias = _param(weight), _param(bias)
        self.m = int(d if out_map is None else out_map.shape[0])
        self.register_buffer("out_w", None if out_map is None else out_map.T.contiguous())

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        c = self.norm(tokens[:, 0])
        if self.out_w is not None:
            c = F.linear(c[..., :self.m], self.out_w)
        return c * self.weight + self.bias


class Embed(nn.Module):
    """The folded model's own embedding (patch embedding, CLS, registers, positions), parameters rotated
    into the first stream basis. `base` is the model with its blocks and final norm removed."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.family = "dinov2" if hasattr(base, "prepare_tokens_with_masks") else "timm"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.base
        if self.family == "dinov2":
            return m.prepare_tokens_with_masks(x)
        t = m.patch_embed(x)
        t = m._pos_embed(t)
        t = m.patch_drop(t)
        return m.norm_pre(t)


class FastViT(nn.Module):
    """images -> final-norm CLS feature (the k-NN feature), the same function as the source model."""

    def __init__(self, embed: Embed, blocks: list, head: FinalHead, info: dict | None = None):
        super().__init__()
        self.embed, self.blocks, self.head = embed, nn.ModuleList(blocks), head
        self.info = dict(info or {})

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        t = self.embed(x)
        for blk in self.blocks:
            t = blk(t)
        return t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_tokens(x))


# ----------------------------------------------------------------------------------------------
# Builders: MLP
# ----------------------------------------------------------------------------------------------
def _check_perm(perm, h: int) -> np.ndarray:
    perm = np.asarray(perm, dtype=np.int64)
    if perm.shape != (h,) or not np.array_equal(perm[perm], np.arange(h)):
        raise ValueError("perm must be an involution on the hidden neurons")
    return perm


@torch.no_grad()
def build_mirror_mlp(W1, b1, W2, b2, act: nn.Module, bin_: Basis, bout: Basis, A_in, A_out, perm,
                     align: int = 1, tol: float = 1e-5) -> MirrorMLP:
    """MirrorMLP from exactly equivariant weights (P W1 = W1 A_in, P b1 = b1, A_out W2 = W2 P, A_out b2 = b2).

    Input tokens are in `bin_` (an eigenbasis of A_in), outputs in `bout` (of A_out). The even and odd
    blocks are read from the weights; the parts that must vanish are measured (relative Frobenius
    residual, raised above `tol`, which catches structure or bookkeeping errors)."""
    h, d = W1.shape
    dev = W1.device
    if not bin_.aligned_with(A_in):
        raise ValueError("fc1 operator does not match the input stream basis")
    if not bout.aligned_with(A_out):
        raise ValueError("fc2 operator does not match the output stream basis")
    perm = _check_perm(perm, h)
    idx = np.arange(h)
    pi = idx[perm > idx]
    pj, ps = perm[pi], idx[perm == idx]
    n_p, n_s = len(pi), len(ps)
    ti, tj, ts = (torch.as_tensor(a, dtype=torch.long, device=dev) for a in (pi, pj, ps))
    W1, b1, W2, b2 = (_t(x, dev) for x in (W1, b1, W2, b2))
    dpi, dpo = bin_.dp, bout.dp
    V1 = W1 @ _t(bin_.mat(d), dev)                                    # [h, d] in input coordinates
    a_e = 0.5 * (V1[ti, :dpi] + V1[tj, :dpi])
    a_o = 0.5 * (V1[ti, dpi:] - V1[tj, dpi:])
    c_e = V1[ts, :dpi]
    disc = (0.5 * _sq(V1[ti, :dpi] - V1[tj, :dpi]) + 0.5 * _sq(V1[ti, dpi:] + V1[tj, dpi:]) + _sq(V1[ts, dpi:]))
    bb, bs = 0.5 * (b1[ti] + b1[tj]), b1[ts]
    disc_b = 0.5 * _sq(b1[ti] - b1[tj])
    V2 = _t(bout.mat(d), dev).T @ W2                                  # [d, h] in output coordinates
    p_e = 0.5 * (V2[:dpo, ti] + V2[:dpo, tj])
    p_o = 0.5 * (V2[dpo:, ti] - V2[dpo:, tj])
    q_e = V2[:dpo, ts]
    disc += 0.5 * _sq(V2[:dpo, ti] - V2[:dpo, tj]) + 0.5 * _sq(V2[dpo:, ti] + V2[dpo:, tj]) + _sq(V2[dpo:, ts])
    c2 = _t(bout.mat(d), dev).T @ b2
    disc_b += _sq(c2[dpo:])
    res = float(np.sqrt(disc / max(_sq(W1) + _sq(W2), 1e-300)))
    res_b = float(np.sqrt(disc_b / max(_sq(b1) + _sq(b2), 1e-300)))
    if res > tol or res_b > tol:
        raise ValueError(f"MLP weights are not equivariant for this structure (relative residual {res:.2e}, "
                         f"bias {res_b:.2e}, tol {tol:g})")
    npad, nspad = ceil_to(n_p, align), ceil_to(n_s, align)
    sin, sout = _Split(d, dpi, align), _Split(d, dpo, align)
    E1 = sin.in_even(torch.cat([_pad2(a_e, rows=npad), _pad2(c_e, rows=nspad)], 0))
    b1p = torch.cat([_pad1(bb, npad), _pad1(bs, nspad)])
    O1 = sin.in_odd(_pad2(a_o, rows=npad))
    E2 = sout.out_even(torch.cat([_pad2(p_e, cols=npad), _pad2(q_e, cols=nspad)], 1))
    b2p = _pad1(c2[:dpo], sout.ke)
    O2 = sout.out_odd(_pad2(p_o, cols=npad))
    free1 = (n_p + n_s) * dpi + n_p * (d - dpi)
    free2 = (n_p + n_s) * dpo + n_p * (d - dpo)
    assert free1 == equivariant_param_count(h, d, n_s, 2 * dpi - d)
    assert free2 == equivariant_param_count(h, d, n_s, 2 * dpo - d)
    exec_ = sum(int(t.numel()) for t in (E1, O1, E2, O2))
    cost = _cost("mirror", free1 + free2, n_p + n_s + dpo, free1 + free2, exec_, 2 * h * d,
                 stored_params=exec_ + int(b1p.numel() + b2p.numel()), n_pairs=n_p, n_self=n_s,
                 fc1_params=free1, fc2_params=free2, residual=res, residual_bias=res_b)
    return MirrorMLP(E1, b1p, O1, E2, b2p, O2, copy.deepcopy(act), sin, sout, npad, cost)


@torch.no_grad()
def build_dense_mlp(W1, b1, W2, b2, act: nn.Module, Qi=None, Qo=None) -> DenseMLP:
    """Dense MLP reading in basis Qi and writing in basis Qo (None = identity); rotations folded (free)."""
    h, d = W1.shape
    dev = W1.device
    W1, b1, W2, b2 = (_t(x, dev) for x in (W1, b1, W2, b2))
    if Qi is not None:
        W1 = W1 @ _t(Qi, dev)
    if Qo is not None:
        Qo = _t(Qo, dev)
        W2, b2 = Qo.T @ W2, Qo.T @ b2
    n = 2 * h * d
    return DenseMLP(W1, b1, W2, b2, copy.deepcopy(act), _cost("dense", n, h + d, n, n, n, stored_params=n + h + d))


def _factors(W: torch.Tensor, r: int, r_pad: int):
    """Balanced rank-r SVD factors U [out, r_pad], V [r_pad, in] (zero padding beyond r)."""
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    root = S[:r].sqrt()
    return _pad2(U[:, :r] * root, cols=r_pad), _pad2(root[:, None] * Vh[:r], rows=r_pad)


@torch.no_grad()
def build_lowrank_mlp(W1, b1, W2, b2, act: nn.Module, ranks, align: int = 1) -> LowRankMLP:
    h, d = W1.shape
    dev = W1.device
    W1, b1, W2, b2 = (_t(x, dev) for x in (W1, b1, W2, b2))
    r1, r2 = (int(r) for r in ranks)
    U1, V1 = _factors(W1, r1, ceil_to(r1, align))
    U2, V2 = _factors(W2, r2, ceil_to(r2, align))
    n = (r1 + r2) * (h + d)
    exec_ = sum(int(t.numel()) for t in (U1, V1, U2, V2))
    return LowRankMLP(V1, U1, b1, V2, U2, b2, copy.deepcopy(act),
                      _cost("svd", n, h + d, n, exec_, 2 * h * d, stored_params=exec_ + h + d, ranks=[r1, r2]))


@torch.no_grad()
def prune_keep(W1: torch.Tensor, W2: torch.Tensor, h_keep: int) -> torch.Tensor:
    """Indices (sorted) of the h_keep neurons with the largest ||W1[i]|| ||W2[:, i]|| (weights only)."""
    h = W1.shape[0]
    h_keep = int(max(1, min(h, h_keep)))
    score = W1.double().norm(dim=1) * W2.double().norm(dim=0)
    return torch.sort(torch.topk(score, h_keep).indices).values


@torch.no_grad()
def build_pruned_mlp(W1, b1, W2, b2, act: nn.Module, h_keep: int, align: int = 1) -> DenseMLP:
    """Structured width pruning: keep the h_keep neurons with the largest ||W1[i]|| ||W2[:, i]|| (weights
    only, no data: a timing baseline), zero neurons pad the width to the alignment."""
    h, d = W1.shape
    dev = W1.device
    W1, b1, W2, b2 = (_t(x, dev) for x in (W1, b1, W2, b2))
    h_keep = int(max(1, min(h, h_keep)))
    keep = prune_keep(W1, W2, h_keep)
    hp = ceil_to(h_keep, align)
    W1p, b1p = _pad2(W1[keep], rows=hp), _pad1(b1[keep], hp)
    W2p = _pad2(W2[:, keep], cols=hp)
    n = 2 * h_keep * d
    exec_ = int(W1p.numel() + W2p.numel())
    return DenseMLP(W1p, b1p, W2p, b2, copy.deepcopy(act),
                    _cost("prune", n, h_keep + d, n, exec_, 2 * h * d, stored_params=exec_ + hp + d, h_keep=h_keep))


# ----------------------------------------------------------------------------------------------
# Builders: attention
# ----------------------------------------------------------------------------------------------
def _heads(W: torch.Tensor, H: int, dh: int) -> torch.Tensor:
    return W.reshape(H, dh, W.shape[-1])


@torch.no_grad()
def build_dense_attention(f: dict, H: int, dh: int, scale: float, Qi=None) -> DenseAttention:
    """Dense attention reading and writing in basis Qi (None = identity); rotations folded (free)."""
    dev = f["Wq"].device
    Wqkv = torch.cat([_t(f["Wq"], dev), _t(f["Wk"], dev), _t(f["Wv"], dev)], 0)
    bqkv = torch.cat([_t(f["bq"], dev), _t(f["bk"], dev), _t(f["bv"], dev)], 0)
    Wo, bo = _t(f["Wo"], dev), _t(f["bo"], dev)
    if Qi is not None:
        Q = _t(Qi, dev)
        Wqkv, Wo, bo = Wqkv @ Q, Q.T @ Wo, Q.T @ bo
    d, D = Wo.shape
    n = 4 * D * d
    return DenseAttention(Wqkv, bqkv, Wo, bo, H, dh, scale,
                          _cost("dense", n, 3 * D + d, n, n, n, stored_params=n + 3 * D + d, D=D))


@torch.no_grad()
def build_lowrank_attention(f: dict, H: int, dh: int, scale: float, ranks, align: int = 1) -> LowRankAttention:
    dev = f["Wq"].device
    rq, rk, rv, ro = (int(r) for r in ranks)
    pads = [ceil_to(r, align) for r in (rq, rk, rv, ro)]
    fac = [_factors(_t(f[key], dev), r, rp) for key, r, rp in zip(("Wq", "Wk", "Wv", "Wo"), (rq, rk, rv, ro), pads)]
    (Uq, Vq), (Uk, Vk), (Uv, Vv), (Uo, Vo) = fac
    d = f["Wq"].shape[1]
    D = f["Wq"].shape[0]
    n = (rq + rk + rv + ro) * (D + d)
    exec_ = sum(int(U.numel() + V.numel()) for U, V in fac)
    cost = _cost("svd", n, 3 * D + d, n, exec_, 4 * D * d, stored_params=exec_ + 3 * D + d, D=D, ranks=[rq, rk, rv, ro])
    return LowRankAttention(torch.cat([Vq, Vk, Vv], 0), Uq, Uk, Uv, _t(f["bq"], dev), _t(f["bk"], dev),
                            _t(f["bv"], dev), Vo, Uo, _t(f["bo"], dev), pads[:3], H, dh, scale, cost)


@torch.no_grad()
def build_mirror_attention(f: dict, H: int, dh: int, scale: float, bin_: Basis, A_in, A_out,
                           head_maps: HeadMaps, align: int = 1, wo_mode: str = "fold", tol: float = 1e-5):
    """MirrorAttention from exactly equivariant (projected, gauged) attention weights.

    `head_maps` must be the maps valid for these weights (after the twin gauge: B = C = I on pairs).
    Inputs are in `bin_` (an eigenbasis of A_in) and the output is returned in `bin_` too (the
    stream the residual adds into). If `bin_` is also an eigenbasis of A_out the output projection
    is structured; otherwise wo_mode "fold" stores it dense in `bin_`, and "rotate" keeps it
    structured in eig(A_out) and returns the rotation into `bin_` (d x d, applied to the output).
    Returns (module, attn_rot or None)."""
    if wo_mode not in ("fold", "rotate"):
        raise ValueError("wo_mode must be fold or rotate")
    hm = head_maps
    if hm.H != H or hm.dh != dh:
        raise ValueError("head maps do not match the attention shape")
    dev = f["Wq"].device
    d = f["Wq"].shape[1]
    if not bin_.aligned_with(A_in):
        raise ValueError("q, k, v operator does not match the stream basis")
    Qi = _t(bin_.mat(d), dev)
    dpi = bin_.dp
    pairs = hm.pairs
    selfs = [int(s) for s in hm.self_heads]
    n_p, n_s = len(pairs), len(selfs)
    hs = torch.as_tensor([h for h, _ in pairs], dtype=torch.long, device=dev)
    gs = torch.as_tensor([g for _, g in pairs], dtype=torch.long, device=dev)
    UB = [eig_basis(hm.B[s]) for s in selfs]
    UC = [eig_basis(hm.C[s]) for s in selfs]
    disc, disc_b = 0.0, 0.0
    pe_, po_, pb_, se_, so_, sb_ = {}, {}, {}, {}, {}, {}
    for t, M in (("q", UB), ("k", UB), ("v", UC)):
        W = _heads(_t(f["W" + t], dev), H, dh) @ Qi                      # [H, dh, d] in stream coordinates
        b = _t(f["b" + t], dev).reshape(H, dh)
        if n_p:
            Wh, Wg = W[hs], W[gs]
            pe_[t] = (0.5 * (Wh[..., :dpi] + Wg[..., :dpi])).reshape(n_p * dh, dpi)
            po_[t] = (0.5 * (Wh[..., dpi:] - Wg[..., dpi:])).reshape(n_p * dh, d - dpi)
            pb_[t] = (0.5 * (b[hs] + b[gs])).reshape(n_p * dh)
            disc += 0.5 * _sq(Wh[..., :dpi] - Wg[..., :dpi]) + 0.5 * _sq(Wh[..., dpi:] + Wg[..., dpi:])
            disc_b += 0.5 * _sq(b[hs] - b[gs])
        else:
            pe_[t], po_[t], pb_[t] = W.new_zeros(0, dpi), W.new_zeros(0, d - dpi), W.new_zeros(0)
        se, so, sb = [], [], []
        for s, (U, du) in zip(selfs, M):
            U = _t(U, dev)
            Z = U.T @ W[s]
            c = U.T @ b[s]
            se.append(Z[:du, :dpi])
            so.append(Z[du:, dpi:])
            sb.append(c[:du])
            disc += _sq(Z[:du, dpi:]) + _sq(Z[du:, :dpi])
            disc_b += _sq(c[du:])
        se_[t] = torch.cat(se, 0) if se else W.new_zeros(0, dpi)
        so_[t] = torch.cat(so, 0) if so else W.new_zeros(0, d - dpi)
        sb_[t] = torch.cat(sb, 0) if sb else W.new_zeros(0)
    We_u = torch.cat([pe_["q"], pe_["k"], pe_["v"], se_["q"], se_["k"], se_["v"]], 0)
    be_u = torch.cat([pb_["q"], pb_["k"], pb_["v"], sb_["q"], sb_["k"], sb_["v"]], 0)
    Wo_u = torch.cat([po_["q"], po_["k"], po_["v"], so_["q"], so_["k"], so_["v"]], 0)
    Re, Ro = We_u.shape[0], Wo_u.shape[0]
    P = 3 * n_p * dh
    # Gather index of the self-heads from cat(ze[P:Re], zo[P:Ro]): per type (q, k, v) and self-head,
    # [even coordinates, odd coordinates] in that head's eigenbasis.
    Le = Re - P
    idx = []
    off_e = {"q": 0, "k": se_["q"].shape[0], "v": se_["q"].shape[0] + se_["k"].shape[0]}
    off_o = {"q": 0, "k": so_["q"].shape[0], "v": so_["q"].shape[0] + so_["k"].shape[0]}
    for t, M in (("q", UB), ("k", UB), ("v", UC)):
        ce, co = off_e[t], off_o[t]
        for _, du in M:
            idx.append(np.arange(ce, ce + du))
            idx.append(Le + np.arange(co, co + dh - du))
            ce += du
            co += dh - du
    self_idx = torch.as_tensor(np.concatenate(idx) if idx else np.zeros(0), dtype=torch.long, device=dev)

    # Output projection.
    Wo = _t(f["Wo"], dev).reshape(d, H, dh)
    bo = _t(f["bo"], dev)
    D = H * dh
    attn_rot = None
    if bin_.aligned_with(A_out):
        bout = bin_
    elif wo_mode == "rotate":
        bout = Basis.of(A_out)
        attn_rot = _t(bout.mat(d), dev).T @ Qi                           # y_in = y_out (Q_out^T Q_in)
    else:
        bout = None
    if bout is None:                                                      # dense in the block basis
        cols = [Wo[:, h] for h, _ in pairs] + [Wo[:, g] for _, g in pairs]
        cols += [Wo[:, s] @ _t(V, dev) for s, (V, _) in zip(selfs, UC)]
        Wd = Qi.T @ torch.cat(cols, 1)
        out = {"dense": True, "W": Wd, "b": Qi.T @ bo}
        out_free, out_exec, out_bias = d * D, d * D, d
    else:
        Qo = _t(bout.mat(d), dev)
        dpo = bout.dp
        Y = (Qo.T @ Wo.reshape(d, D)).reshape(d, H, dh)
        oe, oo, e_idx, o_idx = [], [], [], []
        if n_p:
            Yh, Yg = Y[:, hs], Y[:, gs]                                   # [d, n_p, dh]
            oe.append((0.5 * (Yh[:dpo] + Yg[:dpo])).reshape(dpo, n_p * dh))
            oo.append((0.5 * (Yh[dpo:] - Yg[dpo:])).reshape(d - dpo, n_p * dh))
            disc += 0.5 * _sq(Yh[:dpo] - Yg[:dpo]) + 0.5 * _sq(Yh[dpo:] + Yg[dpo:])
        for i, (s, (V, dv)) in enumerate(zip(selfs, UC)):
            Z = Y[:, s] @ _t(V, dev)                                      # [d, dh]
            oe.append(Z[:dpo, :dv])
            oo.append(Z[dpo:, dv:])
            disc += _sq(Z[:dpo, dv:]) + _sq(Z[dpo:, :dv])
            e_idx.append(i * dh + np.arange(dv))
            o_idx.append(i * dh + dv + np.arange(dh - dv))
        Oe_u = torch.cat(oe, 1) if oe else Y.new_zeros(dpo, 0)
        Oo_u = torch.cat(oo, 1) if oo else Y.new_zeros(d - dpo, 0)
        cb = Qo.T @ bo
        disc_b += _sq(cb[dpo:])
        sout = _Split(d, dpo, align)
        Ce, Co = Oe_u.shape[1], Oo_u.shape[1]
        Oe = sout.out_even(_pad2(Oe_u, cols=ceil_to(Ce, align)))
        Oo = sout.out_odd(_pad2(Oo_u, cols=ceil_to(Co, align)))
        out = {"dense": False, "Oe": Oe, "be": _pad1(cb[:dpo], sout.ke), "Oo": Oo,
               "e_idx": torch.as_tensor(np.concatenate(e_idx) if e_idx else np.zeros(0), dtype=torch.long, device=dev),
               "o_idx": torch.as_tensor(np.concatenate(o_idx) if o_idx else np.zeros(0), dtype=torch.long, device=dev),
               "pad_e": ceil_to(Ce, align) - Ce, "pad_o": ceil_to(Co, align) - Co, "dpo": dpo, "cut": sout.cut}
        out_free, out_exec, out_bias = int(Oe_u.numel() + Oo_u.numel()), int(Oe.numel() + Oo.numel()), dpo
    tot = sum(_sq(_t(f[k], dev)) for k in ("Wq", "Wk", "Wv", "Wo"))
    tot_b = sum(_sq(_t(f[k], dev)) for k in ("bq", "bk", "bv", "bo"))
    res = float(np.sqrt(disc / max(tot, 1e-300)))
    res_b = float(np.sqrt(disc_b / max(tot_b, 1e-300)))
    if res > tol or res_b > tol:
        raise ValueError(f"attention weights are not equivariant for this structure (relative residual "
                         f"{res:.2e}, bias {res_b:.2e}, tol {tol:g})")
    sin = _Split(d, dpi, align)
    rows_e, rows_o = ceil_to(Re, align), ceil_to(Ro, align)
    We = sin.in_even(_pad2(We_u, rows=rows_e))
    Wodd = sin.in_odd(_pad2(Wo_u, rows=rows_o))
    be = _pad1(be_u, rows_e)
    qkv_free = int(We_u.numel() + Wo_u.numel())
    free = attention_free_params(d, hm, A_in, A_out)
    assert qkv_free == free["Wq"] + free["Wk"] + free["Wv"], (qkv_free, free)
    if not out["dense"]:
        assert out_free == free["Wo"], (out_free, free["Wo"])
    exec_ = int(We.numel() + Wodd.numel()) + out_exec
    cost = _cost("mirror", qkv_free + out_free, Re + out_bias, qkv_free + out_free, exec_, 4 * D * d,
                 stored_params=exec_ + rows_e + out_bias, D=D, n_pairs=n_p, n_self=n_s, qkv_params=qkv_free,
                 out_params=out_free, out_mode="structured" if not out["dense"] else "dense (rotation folded)",
                 out_rotated=attn_rot is not None, free_formula=int(free["matrices"]), residual=res,
                 residual_bias=res_b)
    mod = MirrorAttention(We, be, Wodd, n_p, n_s, H, dh, scale, sin, Re, Ro, self_idx, out, cost)
    return mod, attn_rot


# ----------------------------------------------------------------------------------------------
# Whole model
# ----------------------------------------------------------------------------------------------
def is_folded(model: nn.Module) -> bool:
    """True if every block norm is mirror.fold.RMSNorm and the final norm is Sequential(rms, ..., affine)."""
    norm = getattr(model, "norm", None)
    return (all(isinstance(b.norm1, FoldRMSNorm) and isinstance(b.norm2, FoldRMSNorm) for b in model.blocks)
            and isinstance(norm, nn.Sequential) and hasattr(norm, "rms") and hasattr(norm, "affine"))


def _block_tensors(blk: nn.Module, dev) -> dict:
    check_attn_supported(blk)
    for name in ("ls1", "ls2"):
        g = getattr(getattr(blk, name, None), "gamma", None)
        if isinstance(g, torch.Tensor) and not bool((g == 1).all()):
            raise ValueError(f"{name} is not folded (gamma != 1)")
    if not (isinstance(blk.norm1, FoldRMSNorm) and isinstance(blk.norm2, FoldRMSNorm)):
        raise ValueError("block norms must be RMSNorm (fold the model first)")
    attn, mlp = blk.attn, blk.mlp
    H = int(attn.num_heads)
    D = attn.qkv.out_features // 3
    dh = D // H
    qkv, proj = attn.qkv, attn.proj
    W = _t(qkv.weight, dev)
    b = _t(qkv.bias, dev) if qkv.bias is not None else W.new_zeros(3 * D)
    f = {"Wq": W[:D], "Wk": W[D:2 * D], "Wv": W[2 * D:], "bq": b[:D], "bk": b[D:2 * D], "bv": b[2 * D:],
         "Wo": _t(proj.weight, dev),
         "bo": _t(proj.bias, dev) if proj.bias is not None else W.new_zeros(proj.out_features)}
    fc1, fc2 = mlp.fc1, mlp.fc2
    if not (isinstance(fc1, nn.Linear) and isinstance(fc2, nn.Linear)):
        raise ValueError("MLP must be fc1 (Linear) -> act -> fc2 (Linear)")
    if not isinstance(getattr(mlp, "norm", nn.Identity()), nn.Identity):
        raise ValueError("MLP norm is not supported")
    W1 = _t(fc1.weight, dev)
    return {"attn": f, "H": H, "dh": dh, "scale": float(getattr(attn, "scale", dh ** -0.5)),
            "W1": W1, "b1": _t(fc1.bias, dev) if fc1.bias is not None else W1.new_zeros(W1.shape[0]),
            "W2": _t(fc2.weight, dev),
            "b2": _t(fc2.bias, dev) if fc2.bias is not None else W1.new_zeros(fc2.out_features),
            "act": mlp.act, "eps1": blk.norm1.eps, "eps2": blk.norm2.eps}


def build_block(blk: nn.Module, spec: BlockSpec, bin_: Basis, bout: Basis, align: int = 1, wo_mode: str = "fold",
                tol: float = 1e-5) -> FastBlock:
    """One FastBlock: input stream in `bin_`, output stream in `bout` (rotation between them if they differ)."""
    dev = next(blk.parameters()).device
    t = _block_tensors(blk, dev)
    d = t["W1"].shape[1]
    Qi = None if bin_.Q is None else bin_.Q
    Qo = None if bout.Q is None else bout.Q
    attn_rot = None
    kind = spec.attn[0]
    if kind == "dense":
        attn = build_dense_attention(t["attn"], t["H"], t["dh"], t["scale"], Qi)
    elif kind == "mirror":
        _, A_in, A_out, hm = spec.attn
        attn, attn_rot = build_mirror_attention(t["attn"], t["H"], t["dh"], t["scale"], bin_, A_in, A_out, hm,
                                                align, wo_mode, tol)
    elif kind == "svd":
        if Qi is not None:
            raise ValueError("the SVD baseline is built in the standard basis")
        attn = build_lowrank_attention(t["attn"], t["H"], t["dh"], t["scale"], spec.attn[1], align)
    else:
        raise ValueError(f"unknown attention kind {kind}")
    kind = spec.mlp[0]
    if kind == "dense":
        mlp = build_dense_mlp(t["W1"], t["b1"], t["W2"], t["b2"], t["act"], Qi, Qo)
    elif kind == "mirror":
        _, A_in, A_out, perm = spec.mlp
        mlp = build_mirror_mlp(t["W1"], t["b1"], t["W2"], t["b2"], t["act"], bin_, bout, A_in, A_out, perm, align, tol)
    elif kind in ("svd", "prune"):
        if Qi is not None or Qo is not None:
            raise ValueError("SVD and pruning baselines are built in the standard basis")
        if kind == "svd":
            mlp = build_lowrank_mlp(t["W1"], t["b1"], t["W2"], t["b2"], t["act"], spec.mlp[1], align)
        else:
            mlp = build_pruned_mlp(t["W1"], t["b1"], t["W2"], t["b2"], t["act"], spec.mlp[1], align)
    else:
        raise ValueError(f"unknown MLP kind {kind}")
    rot = None
    if not bin_.same(bout):
        rot = _t(bin_.mat(d), dev).T @ _t(bout.mat(d), dev)              # x_out = x_in (Q_in^T Q_out)
    return FastBlock(d, t["eps1"], t["eps2"], attn, mlp, rot, attn_rot)


@torch.no_grad()
def _build_embed(model: nn.Module, basis: Basis) -> Embed:
    saved = {k: getattr(model, k) for k in ("blocks", "norm", "head", "fc_norm") if hasattr(model, k)}
    try:
        model.blocks = nn.ModuleList()
        for k in ("norm", "head", "fc_norm"):
            if k in saved:
                setattr(model, k, nn.Identity())
        base = copy.deepcopy(model)
    finally:
        for k, v in saved.items():
            setattr(model, k, v)
    base.eval()
    if basis.Q is not None:
        ref = base.patch_embed.proj.weight
        Q = _t(basis.Q, ref.device)
        pe = base.patch_embed.proj
        W = pe.weight.double()
        if isinstance(pe, nn.Conv2d):
            Wn = torch.einsum("io,ichw->ochw", Q, W)
        elif isinstance(pe, nn.Linear):
            Wn = Q.T @ W
        else:
            raise NotImplementedError("patch_embed.proj must be Conv2d or Linear")
        pe.weight.copy_(Wn.to(pe.weight.dtype))
        if pe.bias is not None:
            pe.bias.copy_((pe.bias.double() @ Q).to(pe.bias.dtype))
        for name in _TOKEN_PARAMS:
            p = getattr(base, name, None)
            if isinstance(p, torch.Tensor):
                p.copy_((p.double() @ Q).to(p.dtype))
    return Embed(base)


@torch.no_grad()
def _build_head(norm: nn.Module, basis: Basis, d: int) -> FinalHead:
    if not (isinstance(norm, nn.Sequential) and hasattr(norm, "rms") and hasattr(norm, "affine")):
        raise ValueError("model.norm must be Sequential(rms, [even], affine) (fold with final_norm_affine=False)")
    dev = norm.affine.weight.device
    even = getattr(norm, "even", None)
    P = _t(even.P, dev) if even is not None else None
    M = None
    if basis.Q is not None or P is not None:
        M = _t(basis.mat(d), dev).T
        if P is not None:
            M = M @ P
        rn = M.norm(dim=1)
        keep = rn > 1e-6 * rn.max()
        m = int(keep.sum())
        if m < d and bool(keep[:m].all()):
            M = M[:m]                       # rows of a projected-out subspace (even head: the odd rows)
    return FinalHead(d, norm.rms.eps, norm.affine.weight.detach().double(), norm.affine.bias.detach().double(), M)


@torch.no_grad()
def build_fast_vit(model: nn.Module, specs: list, bases: list | None = None, align: int = 1, wo_mode: str = "fold",
                   tol: float = 1e-5, info: dict | None = None) -> FastViT:
    """FastViT computing the same function as the fully folded `model` (left untouched).

    specs: one BlockSpec per block. bases: L + 1 stream bases (default: identity everywhere). The new
    model has the source model's parameter dtype and device."""
    if not is_folded(model):
        raise ValueError("model must be fully folded (mirror.fold.fold_layernorm_full(final_norm_affine=False))")
    L = len(model.blocks)
    if len(specs) != L:
        raise ValueError("one BlockSpec per block")
    d = int(model.blocks[0].mlp.fc1.in_features)
    bases = [Basis.identity(d)] * (L + 1) if bases is None else list(bases)
    if len(bases) != L + 1:
        raise ValueError("bases needs L + 1 entries (stream entering each block, then the final stream)")
    ref = next(model.parameters())
    embed = _build_embed(model, bases[0])
    blocks = [build_block(model.blocks[b], specs[b], bases[b], bases[b + 1], align, wo_mode, tol) for b in range(L)]
    head = _build_head(model.norm, bases[L], d)
    fv = FastViT(embed, blocks, head, info)
    fv.to(device=ref.device, dtype=ref.dtype)
    fv.requires_grad_(False)
    fv.info.update({"align": int(align), "wo_mode": wo_mode,
                    "converted_blocks": [b for b, s in enumerate(specs) if s.converted],
                    "n_rotations": sum(1 for blk in fv.blocks if blk.rot_t is not None),
                    "n_attn_rotations": sum(1 for blk in fv.blocks if blk.attn_rot_t is not None)})
    return fv.eval()


def fast_vit_cost(fv: FastViT, n_tokens: int, n_patches: int) -> dict:
    """Analytic parameter and multiply-add counts of a FastViT (per image; see the module docstring).

    Counted multiply-adds: patch embedding (n_patches x kernel size x d), every matrix per token (its
    free parameters; low-rank r (in + out)), stream and attention-output rotations (d^2 per token),
    token mixing 2 T^2 D per block, and the rotation of the CLS feature. Not counted: biases, norms,
    softmax, GELU, residual and butterfly additions, gathers. `*_exec` include alignment padding."""
    T = int(n_tokens)
    patch = int(fv.embed.base.patch_embed.proj.weight.numel())
    out = {"macs_patch": int(n_patches) * patch, "macs_head": 0, "macs_rot": 0, "macs_mix": 0,
           "macs_matrices": 0, "macs_matrices_exec": 0, "matrix_params_converted": 0,
           "matrix_params_converted_dense": 0, "matrix_params_all": 0, "rotation_params": 0,
           "padding_params": 0, "n_converted": 0}
    for blk in fv.blocks:
        ca, cm = blk.attn.cost, blk.mlp.cost
        out["macs_matrices"] += T * (ca["macs_token"] + cm["macs_token"])
        out["macs_matrices_exec"] += T * (ca["macs_token_exec"] + cm["macs_token_exec"])
        out["macs_rot"] += T * blk.rotation_macs_token
        out["rotation_params"] += blk.rotation_macs_token
        out["macs_mix"] += 2 * T * T * ca["D"]
        out["matrix_params_all"] += ca["matrix_params"] + cm["matrix_params"]
        out["padding_params"] += (ca["macs_token_exec"] - ca["macs_token"]) + (cm["macs_token_exec"] - cm["macs_token"])
        if ca["kind"] != "dense" or cm["kind"] != "dense":
            out["n_converted"] += 1
            out["matrix_params_converted"] += ca["matrix_params"] + cm["matrix_params"]
            out["matrix_params_converted_dense"] += ca["dense_matrix_params"] + cm["dense_matrix_params"]
    if fv.head.out_w is not None:
        out["macs_head"] = int(fv.head.out_w.numel())
        out["rotation_params"] += int(fv.head.out_w.numel())
    base = out["macs_patch"] + out["macs_mix"] + out["macs_head"] + out["macs_rot"]
    out["macs"] = base + out["macs_matrices"]
    out["macs_exec"] = base + out["macs_matrices_exec"]
    out["gflops"] = 2.0 * out["macs"] / 1e9
    out["gflops_exec"] = 2.0 * out["macs_exec"] / 1e9
    return out
