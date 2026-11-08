"""Attention constraints for the healing trainer (mirror/heal.py): M1 attention in training.

MirrorAttnConstraint keeps the attention of the converted blocks exactly mirror-equivariant. After
every optimiser step the absorbed q, k, v and output weights and biases are re-projected with
mirror.attention.project_attn (gauge=False), using the head maps that hold after surgery (the maps
returned by attention.apply_attention_surgery). Then attn(x A_in) = attn(x) A_out, with the head
relations of PROJECT_CONTEXT 5.4. The norm1 affine and ls1 are frozen at the identity (the absorbed
form), as MirrorMLPConstraint freezes norm2 and ls2. Projection runs in float64 and is rounded once
to the parameter dtype.

LowRankAttention is the SVD counterpart: q, k, v (LowRankQKV) and the output projection
(heal.LowRankLinear) are factorised at fixed ranks, so the ranks stay exact while training. Call
m1.apply_attention_svd first (it absorbs norm1 and ls1 and truncates); the factorisation then
reproduces it.

Both follow the constraint protocol of mirror/heal.py (prepare, frozen_parameters, __call__, check,
describe).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .attention import HeadMaps, attn_dims, project_attn
from .heal import LowRankLinear, exact_involution

__all__ = ["attn_absorbed_parameters", "is_attn_absorbed", "MirrorAttnConstraint", "LowRankQKV",
           "LowRankAttention", "read_attn", "write_attn"]


def attn_absorbed_parameters(block: nn.Module) -> list:
    """norm1 affine and ls1 gamma (the parameters the absorbed attention form fixes)."""
    out = []
    for name in ("weight", "bias"):
        p = getattr(block.norm1, name, None)
        if isinstance(p, nn.Parameter):
            out.append(p)
    g = getattr(getattr(block, "ls1", None), "gamma", None)
    if isinstance(g, nn.Parameter):
        out.append(g)
    return out


def is_attn_absorbed(block: nn.Module) -> bool:
    w, b = getattr(block.norm1, "weight", None), getattr(block.norm1, "bias", None)
    g = getattr(getattr(block, "ls1", None), "gamma", None)
    return ((w is None or bool((w == 1).all())) and (b is None or bool((b == 0).all()))
            and (g is None or bool((g == 1).all())))


def read_attn(block: nn.Module, dtype=torch.float64) -> dict:
    """Raw (already absorbed) attention weights as Wq, Wk, Wv, bq, bk, bv, Wo, bo."""
    qkv, proj = block.attn.qkv, block.attn.proj
    W, b = qkv.weight.to(dtype), qkv.bias.to(dtype)
    D = W.shape[0] // 3
    return {"Wq": W[:D], "Wk": W[D:2 * D], "Wv": W[2 * D:], "bq": b[:D], "bk": b[D:2 * D], "bv": b[2 * D:],
            "Wo": proj.weight.to(dtype), "bo": proj.bias.to(dtype)}


@torch.no_grad()
def write_attn(block: nn.Module, e: dict) -> None:
    qkv, proj = block.attn.qkv, block.attn.proj
    qkv.weight.copy_(torch.cat([e["Wq"], e["Wk"], e["Wv"]], 0))
    qkv.bias.copy_(torch.cat([e["bq"], e["bk"], e["bv"]], 0))
    proj.weight.copy_(e["Wo"])
    proj.bias.copy_(e["bo"])


class MirrorAttnConstraint:
    """Keeps the attention of `blocks` exactly mirror-equivariant during training (see module doc).

    `head_maps_list` must hold the maps valid for the student's current weights, i.e. the list
    returned by attention.apply_attention_surgery (gauged twin pairs have B = C = I).
    """

    def __init__(self, blocks: Sequence[int], A_in_list: Sequence, A_out_list: Sequence,
                 head_maps_list: Sequence[HeadMaps], dtype: torch.dtype = torch.float64):
        if not (len(blocks) == len(A_in_list) == len(A_out_list) == len(head_maps_list)):
            raise ValueError("one A_in, A_out and HeadMaps per block")
        self.blocks = [int(b) for b in blocks]
        self.A_in = [exact_involution(A) for A in A_in_list]
        self.A_out = [exact_involution(A) for A in A_out_list]
        self.maps = []
        for hm in head_maps_list:
            hm.check(tol=1e-8)
            self.maps.append(HeadMaps(sigma=np.asarray(hm.sigma, dtype=np.int64).copy(),
                                      B=np.asarray(hm.B, dtype=np.float64).copy(),
                                      C=np.asarray(hm.C, dtype=np.float64).copy()))
        self.dtype = dtype
        self._cache: dict = {}

    def _ops(self, device: torch.device) -> list:
        """(A_in, A_out, maps with B and C as device tensors) per block, cached per device."""
        key = str(device)
        if key not in self._cache:
            kw = dict(dtype=self.dtype, device=device)
            self._cache[key] = [(torch.as_tensor(a, **kw), torch.as_tensor(b, **kw),
                                 HeadMaps(sigma=hm.sigma, B=torch.as_tensor(hm.B, **kw), C=torch.as_tensor(hm.C, **kw)))
                                for a, b, hm in zip(self.A_in, self.A_out, self.maps)]
        return self._cache[key]

    def prepare(self, model: nn.Module) -> None:
        for i, b in enumerate(self.blocks):
            blk = model.blocks[b]
            qkv, proj = blk.attn.qkv, blk.attn.proj
            if not (isinstance(qkv, nn.Linear) and isinstance(proj, nn.Linear)):
                raise ValueError(f"block {b}: attn.qkv and attn.proj must be dense nn.Linear layers")
            if qkv.bias is None or proj.bias is None:
                raise ValueError(f"block {b}: attn.qkv and attn.proj need biases")
            H, dh = attn_dims(blk)
            d = qkv.in_features
            if self.A_in[i].shape != (d, d) or self.A_out[i].shape != (d, d) or self.maps[i].H != H or self.maps[i].dh != dh:
                raise ValueError(f"block {b}: operator or head-map shape does not match the attention")
            if not is_attn_absorbed(blk):
                raise ValueError(f"block {b}: attention is not in absorbed form; run attention.apply_attention_surgery first")

    def frozen_parameters(self, model: nn.Module) -> list:
        return [p for b in self.blocks for p in attn_absorbed_parameters(model.blocks[b])]

    @torch.no_grad()
    def __call__(self, model: nn.Module) -> None:
        dev = model.blocks[self.blocks[0]].attn.qkv.weight.device
        for (Ai, Ao, hm), b in zip(self._ops(dev), self.blocks):
            blk = model.blocks[b]
            e = project_attn(**read_attn(blk, self.dtype), A_in=Ai, A_out=Ao, head_maps=hm, gauge=False)
            write_attn(blk, e)

    @torch.no_grad()
    def check(self, model: nn.Module) -> dict:
        """Relative distance of the current attention weights from their projection, per block:
        sqrt(sum ||W - P(W)||^2 / sum ||W||^2) over all eight tensors (0 when exactly equivariant)."""
        dev = model.blocks[self.blocks[0]].attn.qkv.weight.device
        per_block = []
        for (Ai, Ao, hm), b in zip(self._ops(dev), self.blocks):
            f = read_attn(model.blocks[b], self.dtype)
            e = project_attn(**f, A_in=Ai, A_out=Ao, head_maps=hm, gauge=False)
            num = sum(float(((f[k] - e[k]) ** 2).sum()) for k in f)
            den = sum(float((f[k] ** 2).sum()) for k in f)
            per_block.append((num / den) ** 0.5 if den > 0 else 0.0)
        return {"max_rel_violation": max(per_block), "per_block": per_block}

    def describe(self) -> dict:
        d_plus = lambda A: int(round((np.trace(A) + A.shape[0]) / 2))  # noqa: E731
        return {"type": "MirrorAttnConstraint", "blocks": self.blocks,
                "n_head_pairs": [hm.n_pairs for hm in self.maps], "n_self_heads": [hm.n_self for hm in self.maps],
                "d_plus_in": [d_plus(A) for A in self.A_in], "d_plus_out": [d_plus(A) for A in self.A_out]}


class LowRankQKV(nn.Module):
    """attn.qkv replacement: q, k and v each a LowRankLinear; output [q; k; v] like the fused layer.
    `weight` and `bias` are the dense concatenations (read-only), so folding and analysis code work."""

    def __init__(self, q: LowRankLinear, k: LowRankLinear, v: LowRankLinear):
        super().__init__()
        self.q, self.k, self.v = q, k, v
        self.in_features = int(q.in_features)
        self.out_features = int(q.out_features + k.out_features + v.out_features)

    @classmethod
    @torch.no_grad()
    def from_linear(cls, lin: nn.Linear, ranks) -> "LowRankQKV":
        D = lin.out_features // 3
        parts = []
        for i, r in enumerate(ranks):
            sub = nn.Linear(lin.in_features, D, bias=lin.bias is not None, device=lin.weight.device,
                            dtype=lin.weight.dtype)
            sub.weight.copy_(lin.weight[i * D:(i + 1) * D])
            if lin.bias is not None:
                sub.bias.copy_(lin.bias[i * D:(i + 1) * D])
            parts.append(LowRankLinear.from_linear(sub, int(r)))
        return cls(*parts)

    @property
    def ranks(self) -> tuple:
        return (self.q.rank, self.k.rank, self.v.rank)

    @property
    def weight(self) -> torch.Tensor:
        return torch.cat([self.q.weight, self.k.weight, self.v.weight], 0)

    @property
    def bias(self):
        if self.q.bias is None:
            return None
        return torch.cat([self.q.bias, self.k.bias, self.v.bias], 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.q(x), self.k(x), self.v(x)], dim=-1)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, ranks={self.ranks}"


class LowRankAttention:
    """SVD baseline for attention in training: `prepare` replaces attn.qkv by LowRankQKV and attn.proj
    by LowRankLinear (from their truncated SVD), so the ranks stay exact while training.

    Call m1.apply_attention_svd(student, blocks, ranks) first; the factorisation then reproduces it.
    `ranks` holds one (r_q, r_k, r_v, r_o) per block. With `freeze_absorbed` (default) norm1's affine
    and ls1 stay frozen at the identity, as in the mirror arm. Installing is idempotent.
    """

    def __init__(self, blocks: Sequence[int], ranks, freeze_absorbed: bool = True):
        self.blocks = [int(b) for b in blocks]
        if len(ranks) != len(self.blocks):
            raise ValueError("one (r_q, r_k, r_v, r_o) per block")
        self.ranks = [tuple(int(v) for v in r) for r in ranks]
        if any(len(r) != 4 for r in self.ranks):
            raise ValueError("ranks must be (r_q, r_k, r_v, r_o)")
        self.freeze_absorbed = bool(freeze_absorbed)

    def install(self, model: nn.Module) -> None:
        for b, (rq, rk, rv, ro) in zip(self.blocks, self.ranks):
            attn = model.blocks[b].attn
            if isinstance(attn.qkv, LowRankQKV):
                if attn.qkv.ranks != (rq, rk, rv):
                    raise ValueError(f"block {b} qkv already has ranks {attn.qkv.ranks}")
            else:
                attn.qkv = LowRankQKV.from_linear(attn.qkv, (rq, rk, rv))
            if isinstance(attn.proj, LowRankLinear):
                if attn.proj.rank != ro:
                    raise ValueError(f"block {b} proj already has rank {attn.proj.rank}")
            else:
                attn.proj = LowRankLinear.from_linear(attn.proj, ro)

    def prepare(self, model: nn.Module) -> None:
        if self.freeze_absorbed:
            for b in self.blocks:
                if not is_attn_absorbed(model.blocks[b]):
                    raise ValueError(f"block {b}: attention is not in absorbed form; run m1.apply_attention_svd first")
        self.install(model)

    def frozen_parameters(self, model: nn.Module) -> list:
        if not self.freeze_absorbed:
            return []
        return [p for b in self.blocks for p in attn_absorbed_parameters(model.blocks[b])]

    def __call__(self, model: nn.Module) -> None:
        return None  # the factorisation keeps the ranks exact; nothing to project

    @torch.no_grad()
    def check(self, model: nn.Module) -> dict:
        """Numerical ranks of U V formed in float64 from the factors (a float32 product would carry
        rounding noise in every direction)."""
        def rank(lin):
            if isinstance(lin, LowRankLinear):
                W = lin.U.to(torch.float64) @ lin.V.to(torch.float64)
            else:
                W = lin.weight.to(torch.float64)
            return int(torch.linalg.matrix_rank(W))

        ranks = []
        for b in self.blocks:
            attn = model.blocks[b].attn
            q = attn.qkv
            r = [rank(q.q), rank(q.k), rank(q.v)] if isinstance(q, LowRankQKV) else [-1, -1, -1]
            ranks.append(r + [rank(attn.proj)])
        return {"numerical_ranks": ranks, "target_ranks": [list(r) for r in self.ranks]}

    def describe(self) -> dict:
        return {"type": "LowRankAttention", "blocks": self.blocks, "ranks": [list(r) for r in self.ranks],
                "freeze_absorbed": self.freeze_absorbed}
