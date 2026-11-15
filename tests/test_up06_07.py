"""UP-06 (scripts/up06_stack_quant_tome.py, mirror/tome.py) and UP-07 (scripts/up07_weights_pairing.py). CPU only, no
downloads: tiny random models and synthetic data."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from mirror.datafree import pairing_agreement, twin_scores
from mirror.matching import random_pairing
from mirror.tome import (apply_tome, attention_with_metric, bipartite_soft_matching, expected_tokens, merge_wavg)
from mirror.variants import random_involution_fixing_ones

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_up0607_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def U6():
    return load("up06_stack_quant_tome")


@pytest.fixture(scope="module")
def U7():
    return load("up07_weights_pairing")


def tiny_vit(img_size=56, reg_tokens=0, dtype=torch.float64, seed=0):
    """Tiny timm ViT (patch 14) with randomised affines, as a (model, n_prefix) pair."""
    from timm.models.vision_transformer import VisionTransformer

    from mirror.adapters import _randomise_affine

    torch.manual_seed(seed)
    m = VisionTransformer(img_size=img_size, patch_size=14, embed_dim=64, depth=4, num_heads=4, init_values=1e-5,
                          reg_tokens=reg_tokens, num_classes=0)
    _randomise_affine(m, seed)
    return m.to(dtype).eval(), int(m.num_prefix_tokens)


def tokens_of(model, x):
    t = model.patch_embed(x)
    t = model._pos_embed(t)
    return model.norm_pre(t)


def run_blocks(model, t):
    for blk in model.blocks:
        t = blk(t)
    return t


# ----------------------------------------------------------------------------------------------
# UP-06: INT8 round trip
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_int8_round_trip_error_bound(U6, dtype):
    g = torch.Generator().manual_seed(0)
    W = torch.randn(48, 32, generator=g, dtype=dtype) * torch.logspace(-3, 1, 48, dtype=dtype)[:, None]
    W[5] = 0.0                                                    # an all-zero output channel
    q, s, W_hat = U6.int8_per_channel(W)
    assert q.dtype == torch.int8 and s.dtype == torch.float32 and W_hat.dtype == dtype
    assert int(q.abs().max()) <= 127
    err = (W.double() - W_hat.double()).abs()
    assert bool((err <= s.double()[:, None] * (0.5 + 1e-5)).all())       # |W - q s| <= s / 2
    assert torch.equal(W_hat[5], torch.zeros_like(W_hat[5]))
    rows = [i for i in range(48) if i != 5]
    assert bool((q[rows].abs().amax(dim=1) == 127).all())                 # the largest entry of each row maps to +-127
    qn, sn, _ = U6.int8_per_channel(-W)
    assert torch.equal(qn, -q) and torch.equal(sn, s)                     # symmetric scheme: odd under W -> -W
    # a second round trip is (nearly) idempotent: the dequantised weights are on the grid
    q2, _, W2 = U6.int8_per_channel(W_hat)
    assert torch.equal(q2, q) and (W2.double() - W_hat.double()).abs().max() <= 1e-6 * W.abs().max()


def test_int8_quantises_every_linear_in_the_blocks(U6):
    model, _ = tiny_vit(dtype=torch.float32)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    info = U6.quantise_blocks_int8(model)
    assert info["n_matrices"] == 4 * len(model.blocks)                    # qkv, proj, fc1, fc2 per block
    assert info["max_abs_err_over_scale"] <= 0.5 + 1e-4
    assert info["bytes_int8"] < info["bytes_fp32"] / 3
    for n, p in model.named_parameters():
        changed = not torch.equal(p, before[n])
        quantised = n.startswith("blocks.") and n.endswith("weight") and (".qkv." in n or ".proj." in n or ".fc" in n)
        assert changed == quantised, n                                   # only the block Linear weights change


# ----------------------------------------------------------------------------------------------
# UP-06: ToMe
# ----------------------------------------------------------------------------------------------
class DinoAttention(nn.Module):
    """A replica of DINOv2's Attention (dinov2/layers/attention.py): no q_norm, k_norm or norm attributes."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, C))


class DinoBlock(nn.Module):
    def __init__(self, dim=32, heads=4):
        super().__init__()
        from timm.layers import LayerScale, Mlp

        self.norm1, self.attn, self.ls1 = nn.LayerNorm(dim), DinoAttention(dim, heads), LayerScale(dim, 0.7)
        self.norm2, self.mlp, self.ls2 = nn.LayerNorm(dim), Mlp(dim, 4 * dim), LayerScale(dim, 0.3)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


def test_tome_attention_matches_the_blocks_own_attention():
    """attention_with_metric without sizes is the block's attention branch (timm and a DINOv2-style block)."""
    torch.manual_seed(0)
    model, _ = tiny_vit()
    x = torch.randn(2, 3, 56, 56, dtype=torch.float64)
    t = tokens_of(model, x)
    blk = model.blocks[1]
    y, metric = attention_with_metric(blk, t)
    assert (y - blk.ls1(blk.attn(blk.norm1(t)))).abs().max() < 1e-12
    assert metric.shape == (2, t.shape[1], 16)                         # keys averaged over 4 heads of width 16
    dblk = DinoBlock().double().eval()
    z = torch.randn(3, 11, 32, dtype=torch.float64)
    yd, md = attention_with_metric(dblk, z)
    assert (yd - dblk.ls1(dblk.attn(dblk.norm1(z)))).abs().max() < 1e-12 and md.shape == (3, 11, 8)


@pytest.mark.parametrize("reg_tokens", [0, 4])
def test_tome_r0_is_the_identity(reg_tokens):
    model, npre = tiny_vit(reg_tokens=reg_tokens)
    x = torch.randn(2, 3, 56, 56, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = run_blocks(model, tokens_of(model, x))
        m0 = apply_tome(model, 0, npre)
        out = run_blocks(m0, tokens_of(m0, x))
    assert torch.equal(out, ref)                                        # bit for bit
    assert m0.tome_state["tokens"] == [ref.shape[1]] * len(model.blocks)
    assert model.blocks[0].__class__.__name__ == "Block"                # the original is untouched (copy)


@pytest.mark.parametrize("reg_tokens,r", [(0, 8), (4, 8), (4, 3), (0, 100)])
def test_tome_reduces_tokens_as_specified(reg_tokens, r):
    model, npre = tiny_vit(img_size=112, reg_tokens=reg_tokens)                 # 8 x 8 = 64 patches
    x = torch.randn(2, 3, 112, 112, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    n0 = npre + 64
    with torch.no_grad():
        m = apply_tome(model, r, npre)
        t0 = tokens_of(m, x)
        out = run_blocks(m, t0)
    exp = expected_tokens(n0, npre, [r] * 4)
    assert m.tome_state["tokens"] == exp and out.shape[1] == exp[-1]
    if r == 8:
        assert exp == [n0 - 8, n0 - 16, n0 - 24, n0 - 32]               # r tokens removed in every block
    if r == 100:
        assert exp[0] == npre + 32                                     # at most half of the mergeable tokens
    size = m.tome_state["size"]
    assert torch.equal(size[:, :npre, 0], torch.ones(2, npre, dtype=size.dtype))   # prefix tokens never merged
    assert bool((size.sum(dim=1)[:, 0] == n0).all())                   # every patch is accounted for once


def test_bipartite_matching_merges_duplicates_and_protects_prefix():
    g = torch.Generator().manual_seed(3)
    B, N, c, npre = 1, 13, 6, 3
    metric = torch.randn(B, N, c, generator=g, dtype=torch.float64)
    x = torch.randn(B, N, 5, generator=g, dtype=torch.float64)
    metric[:, 4] = metric[:, 7]                                        # positions 4 (set A) and 7 (set B) identical
    x[:, 4] = 2.0
    x[:, 7] = 5.0
    metric[:, 1] = metric[:, 9]                                        # a prefix token identical to a patch token
    merge = bipartite_soft_matching(metric, 1, n_protected=npre)
    y, size = merge_wavg(merge, x)
    assert y.shape == (B, N - 1, 5) and torch.equal(y[:, :npre], x[:, :npre])
    i = int((size[0, :, 0] == 2).nonzero()[0, 0])
    assert torch.allclose(y[0, i], torch.full((5,), 3.5, dtype=torch.float64))   # (2 + 5) / 2: the duplicates merged
    # weighted by size: merging a size-3 token with a size-1 token
    s0 = torch.ones(B, N, 1, dtype=torch.float64)
    s0[:, 4] = 3.0
    y2, size2 = merge_wavg(merge, x, s0)
    j = int((size2[0, :, 0] == 4).nonzero()[0, 0])
    assert torch.allclose(y2[0, j], torch.full((5,), (3 * 2.0 + 5.0) / 4, dtype=torch.float64))


def test_proportional_attention_equals_attention_over_the_duplicated_tokens():
    """A token of size s attends like s identical copies (softmax(QK^T + log s))."""
    model, _ = tiny_vit()
    blk = model.blocks[2]
    g = torch.Generator().manual_seed(4)
    t = torch.randn(1, 9, 64, dtype=torch.float64, generator=g)
    full = torch.cat([t, t[:, 5:6], t[:, 5:6]], dim=1)                  # token 5 three times
    size = torch.ones(1, 9, 1, dtype=torch.float64)
    size[:, 5] = 3.0
    y_full, _ = attention_with_metric(blk, full)
    y_prop, _ = attention_with_metric(blk, t, size)
    assert (y_full[:, :9] - y_prop).abs().max() < 1e-12


# ----------------------------------------------------------------------------------------------
# UP-07: joint scores recover a planted pairing
# ----------------------------------------------------------------------------------------------
def planted_mlp(d=24, h=96, n_pairs=30, noise1=0.0, noise2=0.0, seed=0):
    rng = np.random.default_rng(seed)
    A_in = random_involution_fixing_ones(d, 14, seed + 1)
    A_out = random_involution_fixing_ones(d, 11, seed + 2)
    perm = random_pairing(h, n_pairs, seed + 3)
    W1 = rng.standard_normal((h, d))
    W2 = rng.standard_normal((d, h))
    W1 = 0.5 * (W1 + W1[perm] @ A_in)                                  # P W1 = W1 A_in
    W2 = 0.5 * (W2 + (A_out @ W2)[:, perm])                            # A_out W2 = W2 P
    W1 = W1 + noise1 * rng.standard_normal(W1.shape)
    W2 = W2 + noise2 * rng.standard_normal(W2.shape)
    return W1, W2, A_in, A_out, perm


def test_up07_joint_matching_recovers_a_planted_pairing(U7):
    W1, W2, A_in, A_out, perm = planted_mlp()
    S2 = U7.column_scores(W2, A_out)
    assert np.allclose(S2, S2.T) and np.allclose(S2[np.arange(len(perm)), perm], 1.0)   # exact twins score 1
    for alpha in (0.0, 0.5, 1.0):
        found = U7.match_scores(U7.joint_scores(W1, W2, A_in, A_out, alpha))
        assert pairing_agreement(perm, found)["recovery"] == 1.0 and np.array_equal(found, perm)
    # noisy fc1, clean fc2: the joint score recovers what fc1 alone misses, and needs fc2 under A_out
    W1, W2, A_in, A_out, perm = planted_mlp(noise1=0.9, noise2=0.15, seed=5)
    rec = {name: pairing_agreement(perm, U7.match_scores(S))["recovery"] for name, S in {
        "fc1": twin_scores(W1, A_in),
        "joint": U7.joint_scores(W1, W2, A_in, A_out),
        "joint_swapped": U7.joint_scores(W1, W2, A_out, A_in)}.items()}
    assert rec["joint"] >= 0.95
    assert rec["joint"] > rec["fc1"] + 0.2
    assert rec["joint_swapped"] < rec["joint"] - 0.2                   # the operators act on the right sides


def test_up07_first_attempt_constants_match_the_stored_report(U7):
    """The 46.6% and 44.1% quoted next to the verdict are the first attempt's stored block-1 rows."""
    st = U7.first_attempt_check()
    if st is None:
        pytest.skip("runs/d/d_datafree.json not present")
    assert st["recovery"]["data_fc1"][1] == pytest.approx(U7.FIRST_ATTEMPT["data_fc1"]["recovery"], abs=1e-12)
    assert st["recovery"]["weights_fc1"][1] == pytest.approx(U7.FIRST_ATTEMPT["weights_fc1"]["recovery"], abs=1e-12)
    assert sorted(st["recovery"]["data_fc1"]) == list(range(1, 13)) and sorted(st["recovery"]["weights_fc1"]) == [1, 2, 3]


def test_up07_parse_blocks(U7):
    assert U7.parse_blocks("1-12", 12) == list(range(12))
    assert U7.parse_blocks("1,3, 5", 12) == [0, 2, 4]
    assert U7.parse_blocks("all", 4) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        U7.parse_blocks("1-13", 12)


# ----------------------------------------------------------------------------------------------
# CLI tiny modes
# ----------------------------------------------------------------------------------------------
def test_up07_cli_tiny(U7, tmp_path):
    out = tmp_path / "up07"
    R = U7.main(["--tiny", "--out", str(out), "--blocks", "1,2", "--workers", "1"])
    v = R["verdict"]
    assert v["primary"]["variant"] == "weights_joint" and v["primary"]["recovery"] == 1.0 and v["primary"]["pass"]
    assert R["planted"]["primary_vs_planted"]["recovery"] == 1.0
    assert R["planted"]["A0w_vs_planted"]["trace_agreement"] > 0.999
    assert {r["block"] for r in R["rows"]} == {1, 2}
    assert {r["variant"] for r in R["rows"]} == set(U7.VARIANTS)
    for f in ("pairing_by_block.csv", "operators_by_layer.csv", "pairings.npz", "up07.json", "summary.md", "meta.json"):
        assert (out / f).is_file(), f
    with np.load(out / "pairings.npz") as z:
        assert z["perm_weights_joint"].shape == (2, 256) and z["A_w"].shape == (3, 64, 64)
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert chr(0x2014) not in md and "TINY TEST" in md and "0.466" in md
    meta = json.loads((out / "meta.json").read_text())
    assert meta["experiment"] == "UP-07" and meta["config"]["tiny"] is True


def test_up06_cli_tiny(U6, tmp_path):
    out = tmp_path / "up06"
    args = ["--tiny", "--out", str(out), "--n-boot", "200", "--tome-r", "2"]
    r = U6.main(args)
    rows = r["rows"]
    assert set(rows) == set(U6.ALL_VARIANTS)
    assert rows["mirror_exact"]["flip_consistency_fp32"] == 100.0       # exact by structure
    exp = expected_tokens(17, 1, [2] * 4)
    assert rows["original+tome"]["tokens_final"] == exp[-1] == rows["mirror_exact+tome"]["tokens_final"]
    assert rows["original+int8"]["int8_n_matrices"] == 16 and rows["mirror_exact+int8"]["int8_mirror_max_rel_violation"] > 0
    for row in rows.values():
        assert 0 <= row["top1"] <= 100 and 0 <= row["flip_consistency"] <= 100
    comp = r["comparison"]
    assert set(comp["forms"]) == {"int8", "tome"} and comp["prediction_holds"] in (True, False)
    for form, c in comp["forms"].items():
        o, m = rows["original"]["top1"], rows["mirror_exact"]["top1"]
        assert c["drop_original"] == pytest.approx(o - rows[f"original+{form}"]["top1"])
        assert c["difference"] == pytest.approx(c["drop_mirror_exact"] - c["drop_original"])
        assert c["bootstrap_difference"]["diff"] == pytest.approx(c["difference"])
        assert c["pass"] == (abs(c["difference"]) <= 0.5)
    for f in ("rows.csv", "comparison.json", "summary.md", "meta.json"):
        assert (out / f).is_file(), f
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert chr(0x2014) not in md and "TINY TEST" in md
    assert json.loads((out / "meta.json").read_text())["experiment"] == "UP-06"
    # resumable: a rerun with the same settings evaluates nothing
    calls = []
    orig = U6.evaluate
    U6.evaluate = lambda *a, **k: calls.append(1) or orig(*a, **k)
    try:
        r2 = U6.main(args)
    finally:
        U6.evaluate = orig
    assert calls == [] and r2["rows"]["mirror_exact"]["top1"] == rows["mirror_exact"]["top1"]
