"""Paired statistics of a ViT and any group element g (generalises mirror/stats.py; MUST-03).

Each batch x runs together with g(x) as one batch [x; g(x)] (g from mirror.groups.image_action). Hooks
split every hooked tensor into the original half and the transformed half, re-index the transformed
half to the original token positions (mirror.groups.align_tokens; prefix tokens keep their index) and
fold both into float64 accumulators straight away. Raw activations are never stored.

The accumulators are the existing ones, unchanged:
  * stats.ResidualStats  per residual layer l: C_l = sum X^T Xt (channel-centred), G_l, sums of squares;
  * stats.MLPStats       per block: fc1 outputs (pre-activation), giving R[i, j] = corr(zt_i, z_j);
  * attention.AttnStats  per block: qkv cross-moments (mirror head against original head);
plus `InputMoments` (original half only): the second moments of the folded fc1 input (LayerNorm without
affine, as variants.collect_mlp_input_moments) and of the fc2 input (GELU output), for the
activation-weighted breaking shares. With g = LR the residual and MLP statistics are identical to
stats.collect_fit_stats (tested).

Layer and block indexing as in mirror/stats.py: layer 0 = tokens entering block 0, layer l = output of
block l - 1; blocks are 0-based. Only the requested blocks are hooked, and the forward stops after the
last block any accumulator needs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .groups import align_tokens, image_action, token_index
from .stats import MLPStats, ResidualStats, _flat64


class _Stop(Exception):
    pass


def prepare_images(adapter, imgs: torch.Tensor, device) -> torch.Tensor:
    """uint8 images are normalised with the adapter's statistics (data.normalize_for); float images are
    taken as already normalised (moved to the adapter's dtype)."""
    from .data import normalize_for

    imgs = imgs.to(device, non_blocking=True)
    if imgs.dtype == torch.uint8:
        return normalize_for(adapter, imgs)
    return imgs.to(adapter.dtype)


class InputMoments:
    """Per selected block: M1 = sum u^T u (u = LayerNorm(x) without affine, the folded fc1 input) and
    M2 = sum a^T a (a = fc2 input), over every token of the original images, float64."""

    def __init__(self, n_blocks: int, d: int, h: int, device):
        kw = dict(dtype=torch.float64, device=device)
        self.M1 = torch.zeros(n_blocks, d, d, **kw)
        self.M2 = torch.zeros(n_blocks, h, h, **kw)
        self.n = torch.zeros(n_blocks, **kw)

    def update_fc1(self, slot: int, x: torch.Tensor, eps: float) -> None:
        d = x.shape[-1]
        U = F.layer_norm(x.reshape(-1, d).to(torch.float64), (d,), None, None, eps)
        self.M1[slot] += U.T @ U
        self.n[slot] += U.shape[0]

    def update_fc2(self, slot: int, a: torch.Tensor) -> None:
        A = _flat64(a)
        self.M2[slot] += A.T @ A


class GroupPairedHooks:
    """Forward hooks for a batch [x; g(x)].

    residual_cb(layer, X, Xt), fc1_cb(block, Z, Zt), qkv_cb(block, Y, Yt) receive the original half and
    the aligned transformed half; `inputs` (InputMoments) sees the original half only. `blocks` selects
    the hooked blocks (MLP, attention, inputs); `layers` the residual layers. `stop_after` (a block
    index) ends the forward pass after that block.
    """

    def __init__(self, adapter, model, idx, blocks, layers=None, residual_cb=None, fc1_cb=None, qkv_cb=None,
                 inputs: InputMoments | None = None, stop_after: int | None = None):
        self.handles = []
        npre = adapter.n_prefix_tokens
        idx_t = torch.as_tensor(np.asarray(idx), dtype=torch.long)
        blocks = sorted(int(b) for b in blocks)
        bslot = {b: i for i, b in enumerate(blocks)}
        layers = sorted(int(l) for l in (layers or []))

        def split(t):
            b = t.shape[0] // 2
            return t[:b], align_tokens(t[b:], idx_t, npre)

        if residual_cb is not None and layers:
            if 0 in layers:
                def pre0(_mod, args):
                    residual_cb(0, *split(args[0]))

                self.handles.append(model.blocks[0].register_forward_pre_hook(pre0))
            for i, blk in enumerate(model.blocks):
                if i + 1 in layers:
                    def post(_mod, _args, out, i=i):
                        residual_cb(i + 1, *split(out))

                    self.handles.append(blk.register_forward_hook(post))
        for b in blocks:
            blk = model.blocks[b]
            if fc1_cb is not None:
                def fc1_hook(_mod, _args, out, b=b):
                    fc1_cb(b, *split(out))

                self.handles.append(blk.mlp.fc1.register_forward_hook(fc1_hook))
            if qkv_cb is not None:
                def qkv_hook(_mod, _args, out, b=b):
                    qkv_cb(b, *split(out))

                self.handles.append(blk.attn.qkv.register_forward_hook(qkv_hook))
            if inputs is not None:
                if not isinstance(blk.norm2, nn.LayerNorm):
                    raise ValueError(f"block {b}: norm2 is {type(blk.norm2).__name__}, expected nn.LayerNorm")

                def pre_norm(mod, args, s=bslot[b]):
                    x = args[0]
                    inputs.update_fc1(s, x[:x.shape[0] // 2], mod.eps)

                def pre_fc2(_mod, args, s=bslot[b]):
                    a = args[0]
                    inputs.update_fc2(s, a[:a.shape[0] // 2])

                self.handles.append(blk.norm2.register_forward_pre_hook(pre_norm))
                self.handles.append(blk.mlp.fc2.register_forward_pre_hook(pre_fc2))
        if stop_after is not None and int(stop_after) < len(model.blocks) - 1:
            def stop(_mod, _args, _out):
                raise _Stop()

            self.handles.append(model.blocks[int(stop_after)].register_forward_hook(stop))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


@torch.no_grad()
def run_group_paired(adapter, loader, device, element: str, hooks_kwargs: dict, model=None, desc: str = "") -> int:
    """Push every batch of `loader` and its image under `element` through the model; returns #images."""
    m = adapter.model if model is None else model
    idx = token_index(element, adapter.grid_hw)
    n = 0
    with GroupPairedHooks(adapter, m, idx, **hooks_kwargs):
        for imgs, _ in loader:
            x = prepare_images(adapter, imgs, device)
            try:
                adapter.forward_tokens(torch.cat([x, image_action(x, element)], dim=0), m)
            except _Stop:
                pass
            n += int(x.shape[0])
    return n


@dataclass
class GroupPass:
    """Numpy results of one generator pass over the fit images (selected blocks and layers)."""

    element: str
    blocks: list
    layers: list
    residual: dict | None         # C, G [n_layers, d, d], sq_x, sq_xt, n (indexed by position in `layers`)
    R: dict | None                # block -> R [h, h] float64 (corr(zt_i, z_j))
    attn: dict | None             # AttnStats.numpy() (indexed by position in `blocks`)
    inputs: dict | None           # M1 [n_blocks, d, d], M2 [n_blocks, h, h] (positions in `blocks`)
    n_images: int = 0


def collect_group_pass(adapter, loader, device, element: str, blocks, residual: bool = True, mlp: bool = True,
                       attn: bool = True, inputs: bool = False, center: bool = True, model=None) -> GroupPass:
    """One pass over `loader` with [x; g(x)] collecting every requested statistic for `blocks` (and the
    residual layers they read and write)."""
    from .attention import AttnStats, attn_dims
    from .variants import span_layers

    m = adapter.model if model is None else model
    blocks = sorted(int(b) for b in blocks)
    layers = span_layers(blocks) if residual else []
    lslot = {l: i for i, l in enumerate(layers)}
    bslot = {b: i for i, b in enumerate(blocks)}
    d, h = adapter.embed_dim, adapter.mlp_hidden
    res = ResidualStats(len(layers), d, device, center=center) if residual else None
    ms = MLPStats(len(blocks), h, device) if mlp else None
    ast = None
    if attn:
        H, dh = attn_dims(m.blocks[0])
        ast = AttnStats(len(blocks), H, dh, device)
    inp = InputMoments(len(blocks), d, h, device) if inputs else None
    kw = {"blocks": blocks, "layers": layers,
          "residual_cb": (lambda l, x, xt: res.update(lslot[l], x, xt)) if res is not None else None,
          "fc1_cb": (lambda b, z, zt: ms.update(bslot[b], z, zt)) if ms is not None else None,
          "qkv_cb": (lambda b, y, yt: ast.update(bslot[b], y, yt)) if ast is not None else None,
          "inputs": inp, "stop_after": blocks[-1] if blocks else None}
    n_img = run_group_paired(adapter, loader, device, element, kw, model=m, desc=f"pass {element}")
    out = GroupPass(element=element, blocks=blocks, layers=layers, residual=None, R=None, attn=None, inputs=None,
                    n_images=n_img)
    if res is not None:
        out.residual = res.numpy()
        del res
    if ms is not None:
        out.R = {b: ms.correlation(bslot[b]) for b in blocks}
        del ms
    if ast is not None:
        out.attn = ast.numpy()
        del ast
    if inp is not None:
        out.inputs = {"M1": inp.M1.cpu().numpy(), "M2": inp.M2.cpu().numpy(), "n": inp.n.cpu().numpy()}
        del inp
    return out


def attn_stats_from_numpy(arrs: dict):
    """Rebuild an attention.AttnStats (CPU, float64) from AttnStats.numpy() arrays, for .moments(b)."""
    from .attention import AttnStats

    H, dh = int(arrs["H"]), int(arrs["dh"])
    n_blocks = int(np.asarray(arrs["n"]).shape[0])
    st = AttnStats(n_blocks, H, dh, "cpu")
    for k in ("cross", "s", "s_t", "sq", "sq_t", "n"):
        getattr(st, k).copy_(torch.as_tensor(np.asarray(arrs[k], dtype=np.float64)))
    return st


def collect_residual_moments(adapter, loader, device, element: str, layers, center: bool = True, model=None) -> dict:
    """Residual cross-moments only (held-out operator errors): C, G, sq_x, sq_xt, n for `layers`."""
    m = adapter.model if model is None else model
    layers = sorted(int(l) for l in layers)
    lslot = {l: i for i, l in enumerate(layers)}
    res = ResidualStats(len(layers), adapter.embed_dim, device, center=center)
    kw = {"blocks": [], "layers": layers, "residual_cb": lambda l, x, xt: res.update(lslot[l], x, xt),
          "stop_after": max(layers) - 1 if max(layers) >= 1 else 0}
    run_group_paired(adapter, loader, device, element, kw, model=m, desc=f"held-out {element}")
    return res.numpy()


def collect_twin_null(adapter, loader, device, element: str, null_indices: list, blocks, model=None) -> list:
    """Twin-fraction null (PREREG_NEXT_PHASE.md Section 1): the batch [x; g(x)] is run once, and the
    transformed half's fc1 outputs are aligned with each random patch permutation in `null_indices`
    instead of g's own alignment. Returns one {block: R} per permutation. The forward stops after the
    last block of `blocks`, so the cost is one paired pass however many permutations are given (the
    accumulators, len(null_indices) x len(blocks) x h x h float64, set the memory)."""
    m = adapter.model if model is None else model
    blocks = sorted(int(b) for b in blocks)
    bslot = {b: i for i, b in enumerate(blocks)}
    h, npre = adapter.mlp_hidden, adapter.n_prefix_tokens
    stats = [MLPStats(len(blocks), h, device) for _ in null_indices]
    idx_t = [torch.as_tensor(np.asarray(ix), dtype=torch.long, device=device) for ix in null_indices]
    handles = []
    for b in blocks:
        def hook(_mod, _args, out, b=b):
            half = out.shape[0] // 2
            z, zg = out[:half], out[half:]
            for st, ix in zip(stats, idx_t):
                st.update(bslot[b], z, align_tokens(zg, ix, npre))

        handles.append(m.blocks[b].mlp.fc1.register_forward_hook(hook))
    if blocks[-1] < len(m.blocks) - 1:
        def stop(_mod, _args, _out):
            raise _Stop()

        handles.append(m.blocks[blocks[-1]].register_forward_hook(stop))
    try:
        with torch.no_grad():
            for imgs, _ in loader:
                x = prepare_images(adapter, imgs, device)
                try:
                    adapter.forward_tokens(torch.cat([x, image_action(x, element)], dim=0), m)
                except _Stop:
                    pass
    finally:
        for hd in handles:
            hd.remove()
    return [{b: st.correlation(bslot[b]) for b in blocks} for st in stats]
