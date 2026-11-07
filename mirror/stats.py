"""Streaming statistics of a ViT and its mirror. Raw activations are never stored.

Each batch x is run together with mirror(x) as one batch [x; mirror(x)]. Hooks see every layer's
tokens for both halves, un-flip the mirrored half so rows refer to the same image location, and
fold them into float64 accumulators straight away.

Layer indexing: layer 0 = tokens entering block 0; layer l (1..L) = output of block l-1.
Block indexing in this module is 0-based (block b = model.blocks[b]).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import mirror, normalize


def unflip_tokens(t: torch.Tensor, grid_hw: tuple[int, int], n_prefix: int) -> torch.Tensor:
    """Reverse the W axis of the patch grid; prefix tokens (CLS, registers) are unchanged.

    t: [B, n_prefix + H*W, C] with patches in row-major (H, W) order.
    """
    H, W = grid_hw
    B, T, C = t.shape
    if T != n_prefix + H * W:
        raise ValueError(f"expected {n_prefix + H * W} tokens, got {T}")
    patches = t[:, n_prefix:].reshape(B, H, W, C).flip(2).reshape(B, H * W, C)
    return torch.cat([t[:, :n_prefix], patches], dim=1)


def center_channels(x: torch.Tensor) -> torch.Tensor:
    """Subtract each token's own channel mean (the part LayerNorm removes anyway)."""
    return x - x.mean(dim=-1, keepdim=True)


class PairedHooks:
    """Forward hooks that hand (original, un-flipped mirror) pairs to callbacks.

    residual_cb(layer, X, Xt) with X, Xt: [B, T, d]
    fc1_cb(block, Z, Zt)      with Z, Zt: [B, T, h] (fc1 pre-activations)
    """

    def __init__(self, adapter, model, residual_cb=None, fc1_cb=None):
        self.handles = []
        grid, npre = adapter.grid_hw, adapter.n_prefix_tokens

        def split(t):
            b = t.shape[0] // 2
            return t[:b], unflip_tokens(t[b:], grid, npre)

        if residual_cb is not None:
            def pre0(_mod, args):
                residual_cb(0, *split(args[0]))

            self.handles.append(model.blocks[0].register_forward_pre_hook(pre0))
            for i, blk in enumerate(model.blocks):
                def post(_mod, _args, out, i=i):
                    residual_cb(i + 1, *split(out))

                self.handles.append(blk.register_forward_hook(post))
        if fc1_cb is not None:
            for i, blk in enumerate(model.blocks):
                def fc1_hook(_mod, _args, out, i=i):
                    fc1_cb(i, *split(out))

                self.handles.append(blk.mlp.fc1.register_forward_hook(fc1_hook))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


@torch.no_grad()
def run_paired(adapter, loader, device, residual_cb=None, fc1_cb=None, model=None, desc=""):
    """Push every batch of `loader` and its mirror through the model with the hooks attached."""
    from tqdm import tqdm

    m = adapter.model if model is None else model
    with PairedHooks(adapter, m, residual_cb, fc1_cb):
        for imgs, _ in tqdm(loader, desc=desc, leave=False):
            x = normalize(imgs.to(device, non_blocking=True), adapter.dtype)
            adapter.forward_tokens(torch.cat([x, mirror(x)], dim=0), m)


def _flat64(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1, t.shape[-1]).to(torch.float64)


# ----------------------------------------------------------------------------------------------
# Accumulators
# ----------------------------------------------------------------------------------------------
class ResidualStats:
    """Per layer: C_l = sum X^T Xt (d x d), sum ||X||^2, sum ||Xt||^2, token count."""

    def __init__(self, n_layers: int, d: int, device, center: bool = True):
        self.center = center
        self.C = torch.zeros(n_layers, d, d, dtype=torch.float64, device=device)
        self.sq_x = torch.zeros(n_layers, dtype=torch.float64, device=device)
        self.sq_xt = torch.zeros(n_layers, dtype=torch.float64, device=device)
        self.n = torch.zeros(n_layers, dtype=torch.float64, device=device)

    def update(self, layer: int, x: torch.Tensor, xt: torch.Tensor) -> None:
        X, Xt = _flat64(x), _flat64(xt)
        if self.center:
            X, Xt = center_channels(X), center_channels(Xt)
        self.C[layer] += X.T @ Xt
        self.sq_x[layer] += (X * X).sum()
        self.sq_xt[layer] += (Xt * Xt).sum()
        self.n[layer] += X.shape[0]

    def numpy(self) -> dict:
        return {"C": self.C.cpu().numpy(), "sq_x": self.sq_x.cpu().numpy(),
                "sq_xt": self.sq_xt.cpu().numpy(), "n": self.n.cpu().numpy()}


class MLPStats:
    """Per block: sums, squared sums and the cross-moment sum Zt^T Z (h x h) of fc1 outputs."""

    def __init__(self, n_blocks: int, h: int, device):
        kw = dict(dtype=torch.float64, device=device)
        self.n = torch.zeros(n_blocks, **kw)
        self.s_z = torch.zeros(n_blocks, h, **kw)
        self.s_zt = torch.zeros(n_blocks, h, **kw)
        self.s_z2 = torch.zeros(n_blocks, h, **kw)
        self.s_zt2 = torch.zeros(n_blocks, h, **kw)
        self.cross = torch.zeros(n_blocks, h, h, **kw)  # cross[b, i, j] = sum zt_i z_j

    def update(self, block: int, z: torch.Tensor, zt: torch.Tensor) -> None:
        Z, Zt = _flat64(z), _flat64(zt)
        self.n[block] += Z.shape[0]
        self.s_z[block] += Z.sum(0)
        self.s_zt[block] += Zt.sum(0)
        self.s_z2[block] += (Z * Z).sum(0)
        self.s_zt2[block] += (Zt * Zt).sum(0)
        self.cross[block] += Zt.T @ Z

    def correlation(self, block: int, eps: float = 1e-12) -> np.ndarray:
        """R[i, j] = corr(zt_i, z_j): mirrored-un-flipped neuron i against original neuron j."""
        n = self.n[block]
        mz, mzt = self.s_z[block] / n, self.s_zt[block] / n
        vz = (self.s_z2[block] / n - mz * mz).clamp_min(0)
        vzt = (self.s_zt2[block] / n - mzt * mzt).clamp_min(0)
        cov = self.cross[block] / n - torch.outer(mzt, mz)
        denom = torch.sqrt(torch.outer(vzt, vz))
        R = torch.where(denom > eps, cov / denom.clamp_min(eps), torch.zeros_like(cov))
        return R.clamp(-1.0, 1.0).cpu().numpy()


class ErrorStats:
    """Held-out errors sum ||Xt - X A||^2 for named operators per layer (computed directly)."""

    def __init__(self, operators: list[dict[str, torch.Tensor]], center: bool = True):
        self.center = center
        self.ops = operators
        dev = next(iter(operators[0].values())).device if operators[0] else "cpu"
        self.err = [{name: torch.zeros((), dtype=torch.float64, device=dev) for name in ops}
                    for ops in operators]
        self.err_identity = torch.zeros(len(operators), dtype=torch.float64, device=dev)
        self.sq_x = torch.zeros(len(operators), dtype=torch.float64, device=dev)
        self.sq_xt = torch.zeros(len(operators), dtype=torch.float64, device=dev)
        self.n = torch.zeros(len(operators), dtype=torch.float64, device=dev)

    def update(self, layer: int, x: torch.Tensor, xt: torch.Tensor) -> None:
        X, Xt = _flat64(x), _flat64(xt)
        if self.center:
            X, Xt = center_channels(X), center_channels(Xt)
        for name, A in self.ops[layer].items():
            diff = Xt - X @ A
            self.err[layer][name] += (diff * diff).sum()
        diff = Xt - X
        self.err_identity[layer] += (diff * diff).sum()
        self.sq_x[layer] += (X * X).sum()
        self.sq_xt[layer] += (Xt * Xt).sum()
        self.n[layer] += X.shape[0]

    def numpy(self) -> dict:
        return {
            "err": [{k: float(v) for k, v in e.items()} for e in self.err],
            "err_identity": self.err_identity.cpu().numpy(),
            "sq_x": self.sq_x.cpu().numpy(),
            "sq_xt": self.sq_xt.cpu().numpy(),
            "n": self.n.cpu().numpy(),
        }


# ----------------------------------------------------------------------------------------------
# Passes
# ----------------------------------------------------------------------------------------------
@dataclass
class FitStats:
    residual: ResidualStats
    mlp: MLPStats | None


def collect_fit_stats(adapter, loader, device, center: bool = True, mlp: bool = True) -> FitStats:
    res = ResidualStats(adapter.depth + 1, adapter.embed_dim, device, center=center)
    mstats = MLPStats(adapter.depth, adapter.mlp_hidden, device) if mlp else None
    run_paired(adapter, loader, device, residual_cb=res.update,
               fc1_cb=mstats.update if mstats is not None else None, desc="fit stats")
    return FitStats(res, mstats)


def collect_heldout_errors(adapter, loader, device, operators: list[dict[str, np.ndarray]],
                           center: bool = True) -> dict:
    """`operators[l]` maps a name to a d x d matrix A for layer l (identity is always included)."""
    cache: dict[int, torch.Tensor] = {}  # the same matrix used at several layers is copied once

    def conv(v):
        if id(v) not in cache:
            cache[id(v)] = torch.as_tensor(v, dtype=torch.float64, device=device)
        return cache[id(v)]

    ops_t = [{k: conv(v) for k, v in ops.items()} for ops in operators]
    acc = ErrorStats(ops_t, center=center)
    run_paired(adapter, loader, device, residual_cb=acc.update, desc="held-out errors")
    return acc.numpy()
