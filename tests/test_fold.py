"""X4: full LayerNorm folding (mirror/fold.py).

Checks on the tiny timm ViT (float64 and float32) and on a small mock of the DINOv2 hub model
(same attribute names and token preparation: CLS, position embedding, registers inserted after
CLS, mask token, bicubic position interpolation):
  * folded outputs equal the original's (1e-10 in float64, 1e-5 in float32);
  * the residual stream is mean-zero over channels at every layer;
  * every norm is the RMSNorm class and equals LayerNorm on the stream it sees;
  * the folded network commutes with an arbitrary orthogonal change of the stream basis.
"""
import copy
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mirror.adapters import TimmTinyAdapter
from mirror.fold import RMSNorm, apply_final, fold_copy, fold_layernorm_full


# ----------------------------------------------------------------------------------------------
# Mock of the DINOv2 hub model (dinov2/models/vision_transformer.py, dinov2/layers/*)
# ----------------------------------------------------------------------------------------------
class MockPatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.Identity()

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class MockAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = self.attn_drop((q @ k.transpose(-2, -1)).softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class MockMlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(0.0)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class MockLayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class NestedTensorBlock(nn.Module):
    """Same class name and layout as the block the DINOv2 hub builds."""

    def __init__(self, dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MockAttention(dim, num_heads)
        self.ls1 = MockLayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MockMlp(dim, dim * mlp_ratio)
        self.ls2 = MockLayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class MockDinoVisionTransformer(nn.Module):
    def __init__(self, img_size=56, patch_size=14, embed_dim=48, depth=3, num_heads=4,
                 num_register_tokens=0, interpolate_offset=0.1, interpolate_antialias=False):
        super().__init__()
        self.patch_size = patch_size
        self.num_tokens = 1
        self.num_register_tokens = num_register_tokens
        self.interpolate_offset = interpolate_offset
        self.interpolate_antialias = interpolate_antialias
        self.patch_embed = MockPatchEmbed(patch_size, 3, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.register_tokens = (nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
                                if num_register_tokens else None)
        self.blocks = nn.ModuleList([NestedTensorBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0, h0 = w // self.patch_size, h // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            kwargs["scale_factor"] = (float(w0 + self.interpolate_offset) / M,
                                      float(h0 + self.interpolate_offset) / M)
        else:
            kwargs["size"] = (w0, h0)
        patch_pos_embed = F.interpolate(patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
                                        mode="bicubic", antialias=self.interpolate_antialias, **kwargs)
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat((x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1)
        return x

    def forward_features(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        r = self.num_register_tokens
        return {"x_norm_clstoken": x_norm[:, 0], "x_norm_regtokens": x_norm[:, 1:r + 1],
                "x_norm_patchtokens": x_norm[:, r + 1:], "x_prenorm": x, "masks": masks}

    def forward(self, x, masks=None):
        return self.head(self.forward_features(x, masks)["x_norm_clstoken"])


@torch.no_grad()
def make_mock(num_register_tokens=4, img_size=56, dtype=torch.float64, seed=0):
    """Mock DINOv2 with random weights; all affines, LayerScales, biases and tokens non-trivial
    (token parameters get a channel-mean offset so centring matters)."""
    torch.manual_seed(seed)
    m = MockDinoVisionTransformer(img_size=img_size, num_register_tokens=num_register_tokens)
    g = torch.Generator().manual_seed(seed + 1)
    for mod in m.modules():
        if isinstance(mod, nn.LayerNorm):
            mod.weight.copy_(1.0 + 0.3 * torch.randn(mod.weight.shape, generator=g))
            mod.bias.copy_(0.1 * torch.randn(mod.bias.shape, generator=g))
        elif isinstance(mod, (nn.Linear, nn.Conv2d)) and mod.bias is not None:
            mod.bias.copy_(0.02 * torch.randn(mod.bias.shape, generator=g) + 0.05)
        elif isinstance(mod, MockLayerScale):
            mod.gamma.copy_(0.5 + torch.rand(mod.gamma.shape, generator=g))
    for name in ("cls_token", "pos_embed", "register_tokens", "mask_token"):
        p = getattr(m, name)
        if p is not None:
            p.copy_(0.5 * torch.randn(p.shape, generator=g) + 0.3)
    return m.to(dtype).eval()


def images(n=3, size=56, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, size, size, generator=g, dtype=dtype)


def max_abs(a, b):
    return (a - b).abs().max().item()


# ----------------------------------------------------------------------------------------------
# Stream statistics through hooks
# ----------------------------------------------------------------------------------------------
class StreamProbe:
    """Records max over tokens of |mean over channels| / rms at every block input, the last block
    output, and, for RMSNorm modules, the gap between RMSNorm and a plain LayerNorm."""

    def __init__(self, model):
        self.rel_mean = []
        self.rms_vs_ln = []
        self.handles = []
        blocks = list(model.blocks)
        for blk in blocks:
            self.handles.append(blk.register_forward_pre_hook(self._pre))
        self.handles.append(blocks[-1].register_forward_hook(self._post))
        for mod in model.modules():
            if isinstance(mod, RMSNorm):
                self.handles.append(mod.register_forward_hook(self._norm))

    @staticmethod
    def _rel(x):
        return (x.mean(-1).abs() / x.pow(2).mean(-1).sqrt()).max().item()

    def _pre(self, mod, args):
        self.rel_mean.append(self._rel(args[0]))

    def _post(self, mod, args, out):
        self.rel_mean.append(self._rel(out))

    def _norm(self, mod, args, out):
        ln = F.layer_norm(args[0], (args[0].shape[-1],), eps=mod.eps)
        self.rms_vs_ln.append(max_abs(out, ln))

    def close(self):
        for h in self.handles:
            h.remove()


# ----------------------------------------------------------------------------------------------
# timm tiny model
# ----------------------------------------------------------------------------------------------
def final_tokens(adapter, x, model, info=None):
    t = model.norm(adapter.forward_tokens(x, model))
    return t if info is None else apply_final(info, t)


def test_tiny_folded_matches_float64(tiny64, images64):
    folded, info = fold_copy(tiny64.model)
    assert info["family"] == "timm" and info["n_blocks"] == tiny64.depth
    ref = final_tokens(tiny64, images64, tiny64.model)
    new = final_tokens(tiny64, images64, folded, info)
    assert max_abs(new, ref) < 1e-10
    # The CLS feature path used by k-NN.
    cls_new = apply_final(info, tiny64.cls_feature(images64, folded))
    assert max_abs(cls_new, tiny64.cls_feature(images64)) < 1e-10
    # Not trivial: the folded stream really differs (it is the centred original stream), and
    # the returned affine is needed.
    pre_ref = tiny64.forward_tokens(images64)
    pre_new = tiny64.forward_tokens(images64, folded)
    assert max_abs(pre_new, pre_ref) > 1e-2
    assert max_abs(pre_new, pre_ref - pre_ref.mean(-1, keepdim=True)) < 1e-10
    assert max_abs(folded.norm(pre_new), ref) > 1e-2


def test_tiny_original_untouched(tiny64):
    before = {k: v.clone() for k, v in tiny64.model.state_dict().items()}
    fold_copy(tiny64.model)
    after = tiny64.model.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_tiny_final_affine_in_model(tiny64, images64):
    folded = copy.deepcopy(tiny64.model)
    info = fold_layernorm_full(folded, final_norm_affine=False)
    assert info["final_affine"] is None and info["final_affine_in_model"]
    assert isinstance(folded.norm.rms, RMSNorm)
    # Adapters work unchanged, and so does the full timm forward (head logits).
    assert max_abs(tiny64.cls_feature(images64, folded), tiny64.cls_feature(images64)) < 1e-10
    assert max_abs(folded(images64), tiny64.model(images64)) < 1e-10


def test_tiny_every_norm_is_rmsnorm(tiny64):
    folded, info = fold_copy(tiny64.model)
    assert not any(isinstance(m, nn.LayerNorm) for m in folded.modules())
    for blk in folded.blocks:
        assert type(blk.norm1) is RMSNorm and type(blk.norm2) is RMSNorm
        assert blk.norm1.eps == 1e-6 and blk.norm2.eps == 1e-6
        assert torch.equal(blk.ls1.gamma, torch.ones_like(blk.ls1.gamma))
        assert torch.equal(blk.ls2.gamma, torch.ones_like(blk.ls2.gamma))
    assert type(folded.norm) is RMSNorm
    assert info["n_rmsnorm"] == 2 * tiny64.depth + 1
    assert torch.equal(info["final_weight"], tiny64.model.norm.weight)
    assert torch.equal(info["final_bias"], tiny64.model.norm.bias)


def test_tiny_stream_mean_zero_every_layer(tiny64, images64):
    folded, _ = fold_copy(tiny64.model)
    probe = StreamProbe(folded)
    folded.norm(tiny64.forward_tokens(images64, folded))
    probe.close()
    assert len(probe.rel_mean) == tiny64.depth + 1
    assert max(probe.rel_mean) < 1e-12
    # RMSNorm equals LayerNorm (no affine) on the stream it actually sees.
    assert len(probe.rms_vs_ln) == 2 * tiny64.depth + 1
    assert max(probe.rms_vs_ln) < 1e-12
    # The original stream is not mean-zero, so the check is not trivial.
    orig = StreamProbe(tiny64.model)
    tiny64.forward_tokens(images64)
    orig.close()
    assert max(orig.rel_mean) > 1e-2


def test_tiny_folded_float32():
    a32 = TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float32)
    x = images(3, 56, torch.float32)
    folded, info = fold_copy(a32.model)
    ref = final_tokens(a32, x, a32.model)
    new = final_tokens(a32, x, folded, info)
    assert max_abs(new, ref) < 1e-5


def _rotate_stream_timm(model, Q):
    """Change of residual basis x -> x Q (tokens are rows) in a folded timm model, in place."""
    with torch.no_grad():
        pe = model.patch_embed.proj
        pe.weight.copy_(torch.einsum("i...,ij->j...", pe.weight, Q))
        pe.bias.copy_(pe.bias @ Q)
        model.cls_token.copy_(model.cls_token @ Q)
        model.pos_embed.copy_(model.pos_embed @ Q)
        for blk in model.blocks:
            for lin in (blk.attn.qkv, blk.mlp.fc1):       # read from the stream
                lin.weight.copy_(lin.weight @ Q)
            for lin in (blk.attn.proj, blk.mlp.fc2):      # write into the stream
                lin.weight.copy_(Q.T @ lin.weight)
                lin.bias.copy_(lin.bias @ Q)


def test_tiny_folded_commutes_with_orthogonal_basis_change(tiny64, images64):
    folded, info = fold_copy(tiny64.model)
    d = tiny64.embed_dim
    g = torch.Generator().manual_seed(7)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    rotated = copy.deepcopy(folded)
    _rotate_stream_timm(rotated, Q)
    out = rotated.norm(tiny64.forward_tokens(images64, rotated)) @ Q.T
    ref = final_tokens(tiny64, images64, tiny64.model)
    assert max_abs(apply_final(info, out), ref) < 1e-10


# ----------------------------------------------------------------------------------------------
# Mock DINOv2
# ----------------------------------------------------------------------------------------------
def _dino_outputs(m, x, info=None, masks=None):
    out = m.forward_features(x, masks)
    keys = ("x_norm_clstoken", "x_norm_regtokens", "x_norm_patchtokens")
    return {k: (out[k] if info is None else apply_final(info, out[k])) for k in keys}


@pytest.mark.parametrize("n_reg", [0, 4])
def test_mock_dinov2_matches_float64(n_reg):
    m = make_mock(num_register_tokens=n_reg)
    x = images()
    folded, info = fold_copy(m)
    assert info["family"] == "dinov2"
    expected = {"patch_embed.proj.weight", "patch_embed.proj.bias", "cls_token", "pos_embed", "mask_token"}
    if n_reg:
        expected.add("register_tokens")
    assert expected <= set(info["centred"])
    ref, new = _dino_outputs(m, x), _dino_outputs(folded, x, info)
    assert ref["x_norm_regtokens"].shape[1] == n_reg
    for k in ref:
        if ref[k].numel():
            assert max_abs(new[k], ref[k]) < 1e-10, k
    # The hub forward returns the CLS feature; with the affine applied it matches.
    assert max_abs(apply_final(info, folded(x)), m(x)) < 1e-10
    # Masked tokens (mask_token path) match as well.
    masks = torch.zeros(x.shape[0], 16, dtype=torch.bool)
    masks[:, ::3] = True
    ref_m, new_m = _dino_outputs(m, x, masks=masks), _dino_outputs(folded, x, info, masks=masks)
    assert max_abs(new_m["x_norm_patchtokens"], ref_m["x_norm_patchtokens"]) < 1e-10
    assert max_abs(new_m["x_norm_patchtokens"], new["x_norm_patchtokens"]) > 1e-3


def test_mock_dinov2_structure_and_mean_zero_stream():
    m = make_mock(num_register_tokens=4)
    x = images()
    folded, info = fold_copy(m)
    assert not any(isinstance(mod, nn.LayerNorm) for mod in folded.modules())
    for blk in folded.blocks:
        assert type(blk.norm1) is RMSNorm and type(blk.norm2) is RMSNorm
        assert torch.equal(blk.ls1.gamma, torch.ones_like(blk.ls1.gamma))
        assert torch.equal(blk.ls2.gamma, torch.ones_like(blk.ls2.gamma))
    assert type(folded.norm) is RMSNorm
    probe = StreamProbe(folded)
    folded.forward_features(x)
    probe.close()
    assert max(probe.rel_mean) < 1e-12
    assert max(probe.rms_vs_ln) < 1e-12
    # Registers sit right after CLS and are mean-zero too.
    t = folded.prepare_tokens_with_masks(x)
    assert t.shape[1] == 1 + 4 + 16
    assert t[:, 1:5].mean(-1).abs().max().item() < 1e-14
    orig = StreamProbe(m)
    m.forward_features(x)
    orig.close()
    assert max(orig.rel_mean) > 1e-2


def test_mock_dinov2_interpolated_pos_float32():
    # Position embedding for a 5 x 5 grid, images give 4 x 4: exercises the bicubic interpolation
    # (with DINOv2's float cast and interpolate_offset) on centred position embeddings.
    m = make_mock(num_register_tokens=4, img_size=70, dtype=torch.float32)
    x = images(3, 56, torch.float32)
    folded, info = fold_copy(m)
    ref, new = _dino_outputs(m, x), _dino_outputs(folded, x, info)
    for k in ref:
        assert max_abs(new[k], ref[k]) < 1e-5, k
    probe = StreamProbe(folded)
    folded.forward_features(x)
    probe.close()
    assert max(probe.rel_mean) < 1e-5


def test_mock_dinov2_block_chunks():
    # DINOv2 with block_chunks > 0 wraps blocks in BlockChunk(nn.ModuleList) padded with Identity.
    class BlockChunk(nn.ModuleList):
        def forward(self, x):
            for b in self:
                x = b(x)
            return x

    m = make_mock(num_register_tokens=4)
    blocks = list(m.blocks)
    m.blocks = nn.ModuleList([BlockChunk([blocks[0], blocks[1]]),
                              BlockChunk([nn.Identity(), nn.Identity(), blocks[2]])])
    x = images()
    folded, info = fold_copy(m)
    assert info["n_blocks"] == 3
    ref, new = _dino_outputs(m, x), _dino_outputs(folded, x, info)
    assert max_abs(new["x_norm_patchtokens"], ref["x_norm_patchtokens"]) < 1e-10
