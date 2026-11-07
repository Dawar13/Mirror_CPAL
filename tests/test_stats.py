"""Test 7 (un-flip) plus checks that the hook-based statistics match direct computation."""
import numpy as np
import torch

from mirror.data import mirror
from mirror.stats import (ErrorStats, MLPStats, PairedHooks, ResidualStats, center_channels,
                          unflip_tokens)


def test_unflip_matches_flipped_symmetric_patch_embedding(tiny64):
    """With a mirror-symmetric patch kernel, embedding a flipped image and un-flipping the tokens
    must reproduce the original tokens exactly: checks the row-major (H, W) token order."""
    pe = tiny64.model.patch_embed
    with torch.no_grad():
        pe.proj.weight.copy_(0.5 * (pe.proj.weight + pe.proj.weight.flip(-1)))
    x = torch.randn(2, 3, 56, 56, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    prefix = torch.randn(2, 1, 64, dtype=torch.float64)
    tok = torch.cat([prefix, pe(x)], dim=1)
    tok_flip = torch.cat([prefix, pe(mirror(x))], dim=1)
    assert not torch.allclose(tok, tok_flip)
    back = unflip_tokens(tok_flip, tiny64.grid_hw, 1)
    assert (back - tok).abs().max().item() < 1e-12
    assert torch.equal(back[:, :1], prefix)  # prefix untouched


def test_unflip_is_involution():
    t = torch.arange(2 * 13 * 3, dtype=torch.float64).reshape(2, 13, 3)
    assert torch.equal(unflip_tokens(unflip_tokens(t, (3, 4), 1), (3, 4), 1), t)
    u = unflip_tokens(t, (3, 4), 1)
    assert torch.equal(u[:, 1 + 0 * 4 + 0], t[:, 1 + 0 * 4 + 3])  # (r=0, c=0) <- (0, 3)


@torch.no_grad()
def test_hook_stats_match_direct_computation(tiny64, images64):
    adapter, x = tiny64, images64
    B = x.shape[0]
    L, d, h = adapter.depth, adapter.embed_dim, adapter.mlp_hidden
    res = ResidualStats(L + 1, d, "cpu")
    mlp = MLPStats(L, h, "cpu")
    A = torch.tensor(np.linalg.qr(np.random.default_rng(0).standard_normal((d, d)))[0])
    err = ErrorStats([{"A": A} for _ in range(L + 1)])

    def both(layer, X, Xt):
        res.update(layer, X, Xt)
        err.update(layer, X, Xt)

    with PairedHooks(adapter, adapter.model, residual_cb=both, fc1_cb=mlp.update):
        adapter.forward_tokens(torch.cat([x, mirror(x)]))

    # Direct: run original and mirror separately.
    def layers(inp):
        out, fc1 = [], []
        t = adapter.layer0_tokens(inp)
        out.append(t)
        for blk in adapter.model.blocks:
            fc1.append(blk.mlp.fc1(blk.norm2(t + blk.ls1(blk.attn(blk.norm1(t))))))
            t = blk(t)
            out.append(t)
        return out, fc1

    o, z = layers(x)
    om, zm = layers(mirror(x))
    for l in (0, 2, L):
        X = center_channels(o[l].reshape(-1, d))
        Xt = center_channels(unflip_tokens(om[l], adapter.grid_hw, 1).reshape(-1, d))
        assert torch.allclose(res.C[l], X.T @ Xt, atol=1e-8)
        assert np.isclose(float(err.err[l]["A"]), float(((Xt - X @ A) ** 2).sum()))
        assert np.isclose(float(err.err_identity[l]), float(((Xt - X) ** 2).sum()))
        assert res.n[l] == B * adapter.n_tokens
    b = 1
    Z = z[b].reshape(-1, h).numpy()
    Zt = unflip_tokens(zm[b], adapter.grid_hw, 1).reshape(-1, h).numpy()
    R_direct = np.corrcoef(Zt.T, Z.T)[:h, h:]  # rows: mirrored neurons, cols: original neurons
    assert np.allclose(mlp.correlation(b), R_direct, atol=1e-8)
