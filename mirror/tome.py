"""Token merging (ToMe; Bolya et al., "Token Merging: Your ViT but Faster", ICLR 2023, arXiv 2210.09461) as a forward
wrapper for the ViTs of this repo (DINOv2 hub models, timm ViTs, and healed MUST-09 students, which keep the block
layout norm1, attn.qkv, attn.proj, ls1, norm2, mlp, ls2).

The algorithm, as the paper describes it (Section 3 and 4.1; checked against the arXiv HTML version on 2026-11-15):
  * "we apply our token merging step between the attention and MLP branches of each transformer block";
  * similarity metric: "a dot product similarity metric (e.g., cosine similarity) between the keys of each token",
    with "K averaged over the attention heads";
  * bipartite soft matching: partition the tokens into two sets A and B of roughly equal size, "by alternating";
    draw one edge from each token in A to its most similar token in B; keep the r most similar edges; merge tokens
    that are still connected; concatenate the two sets back together;
  * merged features are averaged weighted by token size s (the number of patches a token represents), and sizes add;
  * proportional attention: softmax(Q K^T / sqrt(d) + log s), s the row vector of token sizes;
  * r is the number of tokens removed in every block ("a quantity of tokens, not a ratio"), at most half of the
    tokens that may be merged.
The official code (github.com/facebookresearch/ToMe, tome/merge.py) never merges the class token: its score row is set
to minus infinity, and it is never a destination because merges go from A to B and it sits in A. Here every prefix
token (CLS and registers, `n_protected`) is protected in the same way: prefix tokens are neither merged nor merged
into, keep size 1 and stay at the front of the sequence in their order. The A/B alternation is by position in the full
sequence (A = even positions, B = odd), as in the official code; for a model with the CLS token only this reproduces
the official layout exactly ([CLS, unmerged A tokens in order, B tokens]). Ties in the edge ranking are broken by a
stable sort (deterministic). The similarity scores are computed in float32 (float64 for float64 inputs) even under
bf16 autocast, so that bf16 rounding does not create ties; the merge itself runs in the stream's dtype.

Exactness. With r = 0 in every block the wrapper calls each block unchanged, so the model is the original bit for
bit. Once a block merges, later blocks run attention through attention_with_metric (scaled_dot_product_attention with
the log-size bias), which equals the block's own attention up to floating-point rounding when no token is merged
(tested in float64). ToMe is not mirror-equivariant: the alternating partition swaps A and B under a left-right flip
of an even-width grid, so a model that is exactly mirror-invariant need not stay so after merging.
"""
from __future__ import annotations

import copy
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["check_block", "attention_with_metric", "mlp_branch", "bipartite_soft_matching", "merge_wavg",
           "ToMeBlock", "apply_tome", "expected_tokens", "r_schedule"]


def _is_identity(mod) -> bool:
    return mod is None or isinstance(mod, nn.Identity)


def check_block(block: nn.Module) -> None:
    """Raise if the block does not have the pre-norm layout this wrapper re-implements."""
    for name in ("norm1", "attn", "norm2", "mlp"):
        if not hasattr(block, name):
            raise ValueError(f"block has no {name}: ToMe wrapper needs norm1, attn, norm2, mlp (ls1, ls2 optional)")
    attn = block.attn
    for name in ("qkv", "proj", "num_heads", "scale"):
        if not hasattr(attn, name):
            raise ValueError(f"attn has no {name}")
    for name in ("q_norm", "k_norm", "norm"):
        if not _is_identity(getattr(attn, name, None)):
            raise ValueError(f"attn.{name} is not supported (must be Identity)")
    if getattr(attn, "gate", None) is not None:
        raise ValueError("gated attention is not supported")
    # Evaluation only: drop path and dropout are not applied by the re-implemented forward.


def _ls(block: nn.Module, name: str, y: torch.Tensor) -> torch.Tensor:
    ls = getattr(block, name, None)
    return y if ls is None else ls(y)


def attention_with_metric(block: nn.Module, x: torch.Tensor, size: torch.Tensor | None = None) -> tuple:
    """(ls1(attn(norm1(x))), metric) with proportional attention.

    size [B, N, 1] adds log(size) to the attention logits of every query (None: plain attention). metric [B, N, dh]
    is the keys averaged over the heads (the ToMe similarity metric)."""
    attn = block.attn
    B, N, _ = x.shape
    H = int(attn.num_heads)
    qkv = attn.qkv(block.norm1(x))
    D = qkv.shape[-1] // 3
    q, k, v = qkv.reshape(B, N, 3, H, D // H).permute(2, 0, 3, 1, 4).unbind(0)       # [B, H, N, dh] each
    bias = None
    if size is not None:
        bias = size.log()[:, None, None, :, 0].to(q.dtype)                             # [B, 1, 1, N]
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=float(attn.scale))
    o = o.transpose(1, 2).reshape(B, N, D)
    o = attn.proj(o)
    return _ls(block, "ls1", o), k.mean(dim=1)


def mlp_branch(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return _ls(block, "ls2", block.mlp(block.norm2(x)))


def bipartite_soft_matching(metric: torch.Tensor, r: int, n_protected: int = 1) -> Callable | None:
    """ToMe's bipartite soft matching on metric [B, N, c]; returns merge(x, mode) -> [B, N - r, C] (None if r <= 0).

    Tokens 0 .. n_protected - 1 are never merged. The other tokens go to A (even positions) or B (odd positions);
    each A token proposes its most similar B token (cosine); the r best proposals are merged (A into B). Output layout:
    [protected tokens, unmerged A tokens in their order, B tokens in their order]."""
    B, N, _ = metric.shape
    n_rest = N - int(n_protected)
    r = min(int(r), n_rest // 2)
    if r <= 0:
        return None
    pos = torch.arange(int(n_protected), N, device=metric.device)
    a_idx, b_idx = pos[pos % 2 == 0], pos[pos % 2 == 1]
    na = int(a_idx.numel())
    with torch.no_grad():
        m = metric if metric.dtype == torch.float64 else metric.float()      # bf16 scores would tie far more often
        m = m / m.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(m.dtype).tiny)
        scores = m[:, a_idx] @ m[:, b_idx].transpose(-1, -2)                           # [B, |A|, |B|]
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True, stable=True)[..., None]    # [B, |A|, 1]
        unm_idx = edge_idx[:, r:].sort(dim=1)[0]
        src_idx = edge_idx[:, :r]
        dst_idx = node_idx[..., None].gather(dim=1, index=src_idx)

    def merge(x: torch.Tensor, mode: str = "sum") -> torch.Tensor:
        c = x.shape[-1]
        xa, xb = x[:, a_idx], x[:, b_idx]
        unm = xa.gather(1, unm_idx.expand(B, na - r, c))
        src = xa.gather(1, src_idx.expand(B, r, c))
        dst = xb.scatter_reduce(1, dst_idx.expand(B, r, c), src, reduce=mode)
        return torch.cat([x[:, :int(n_protected)], unm, dst], dim=1)

    return merge


def merge_wavg(merge: Callable, x: torch.Tensor, size: torch.Tensor | None = None) -> tuple:
    """Size-weighted average merge (ToMe): returns (merged x, merged sizes [B, N', 1])."""
    if size is None:
        size = torch.ones_like(x[..., :1])
    x = merge(x * size, "sum")
    size = merge(size, "sum")
    return x / size, size


class ToMeBlock(nn.Module):
    """One block with token merging between its attention and MLP branches. All blocks of a model share one state
    dict (r per block, protected tokens, proportional attention, current sizes, token counts); block 0 resets it."""

    def __init__(self, block: nn.Module, index: int, tome_state: dict):
        super().__init__()
        check_block(block)
        self.block = block
        self.index = int(index)
        self.tome_state = tome_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        st = self.tome_state
        if self.index == 0:
            st["size"], st["tokens"] = None, []
        r = int(st["r"][self.index])
        size = st["size"]
        if r <= 0 and size is None:
            x = self.block(x)                                   # nothing merged so far: the block itself
        else:
            y, metric = attention_with_metric(self.block, x, size if st["prop_attn"] else None)
            x = x + y
            if r > 0:
                merge = bipartite_soft_matching(metric, r, st["n_protected"])
                if merge is not None:
                    x, st["size"] = merge_wavg(merge, x, size)
            x = x + mlp_branch(self.block, x)
        st["tokens"].append(int(x.shape[1]))
        return x


def r_schedule(r, depth: int) -> list:
    """Constant r per block (ToMe's default schedule), or an explicit list."""
    if isinstance(r, (int, float)):
        return [int(r)] * int(depth)
    rs = [int(v) for v in r]
    if len(rs) != depth:
        raise ValueError(f"r list has {len(rs)} entries for {depth} blocks")
    return rs


def apply_tome(model: nn.Module, r, n_protected: int, prop_attn: bool = True, copy_model: bool = True) -> nn.Module:
    """A copy of `model` (or the model itself) whose blocks merge r tokens each (ToMe). The state is in
    model.tome_state (tokens after each block of the last forward pass in tome_state["tokens"])."""
    m = copy.deepcopy(model) if copy_model else model
    blocks = list(m.blocks)
    if any(isinstance(b, ToMeBlock) for b in blocks):
        raise ValueError("model already has ToMe blocks")
    state = {"r": r_schedule(r, len(blocks)), "n_protected": int(n_protected), "prop_attn": bool(prop_attn),
             "size": None, "tokens": []}
    m.blocks = nn.ModuleList([ToMeBlock(b, i, state) for i, b in enumerate(blocks)])
    m.tome_state = state
    return m


def expected_tokens(n_tokens: int, n_protected: int, r: Sequence[int]) -> list:
    """Token count after each block for a schedule r (each block removes min(r, unprotected // 2))."""
    out, n = [], int(n_tokens)
    for rb in r:
        n -= max(0, min(int(rb), (n - int(n_protected)) // 2))
        out.append(n)
    return out
