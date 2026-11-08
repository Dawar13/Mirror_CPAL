"""CPU tests for mirror/adapters_timm.py. Random weights only (pretrained=False): no downloads."""
import copy
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # belt and braces: nothing here may reach the hub

import pytest
import torch

from mirror.adapters import IMAGENET_MEAN, IMAGENET_STD, _randomise_affine
from mirror.adapters_timm import (MODEL_REGISTRY, TimmPretrainedAdapter, build_model_adapter,
                                  check_layout, normalize_for, resolve_name)
from mirror.stats import unflip_tokens
from mirror.surgery import absorb_block

# short name -> (n_prefix_tokens, has LayerScale, feature)
EXPECT = {
    "deit3_b16": (1, True, "cls"),
    "mae_b16": (1, False, "cls"),
    "siglip2_b16": (0, False, "pooled"),
}
B = 2


@pytest.fixture(scope="module", params=sorted(EXPECT))
def built(request):
    short = request.param
    entry = MODEL_REGISTRY[short]
    torch.manual_seed(0)
    ad = TimmPretrainedAdapter(entry["timm_name"], img_size=224, pretrained=False,
                               feature=entry["feature"])
    _randomise_affine(ad.model, seed=0)  # LayerScale starts at 1e-5 or 1e-6; make blocks matter
    x = torch.randn(B, 3, 224, 224, generator=torch.Generator().manual_seed(1))
    return short, ad, x


def test_layout_attributes(built):
    short, ad, _ = built
    n_prefix, has_ls, feature = EXPECT[short]
    assert ad.grid_hw == (14, 14)
    assert ad.patch_size == 16
    assert ad.n_prefix_tokens == n_prefix == ad.model.num_prefix_tokens
    assert ad.n_tokens == n_prefix + 196
    assert ad.has_layerscale is has_ls
    assert ad.has_cls_token is (n_prefix == 1)
    assert ad.feature == feature
    assert (ad.depth, ad.embed_dim, ad.mlp_hidden) == (12, 768, 3072)
    for blk in ad.blocks:
        for name in ("norm1", "ls1", "norm2", "ls2"):
            assert hasattr(blk, name)
        assert hasattr(blk.attn, "qkv") and hasattr(blk.attn, "proj")
        assert hasattr(blk.mlp, "fc1") and hasattr(blk.mlp, "fc2")


def test_preprocessing_matches_registry(built):
    short, ad, _ = built
    pp = MODEL_REGISTRY[short]["preprocess"]
    assert ad.mean == pytest.approx(pp["mean"]) and ad.std == pytest.approx(pp["std"])
    assert ad.interpolation == pp["interpolation"]
    assert ad.crop_pct == pytest.approx(pp["crop_pct"])
    u8 = torch.full((1, 3, 4, 4), 255, dtype=torch.uint8)
    expect = (1.0 - torch.tensor(pp["mean"])) / torch.tensor(pp["std"])
    assert torch.allclose(normalize_for(ad, u8)[0, :, 0, 0], expect)


@torch.no_grad()
def test_tokens_reproduce_forward_features(built):
    _, ad, x = built
    m = ad.model
    seen = {}
    h = m.blocks[0].register_forward_pre_hook(lambda _mod, args: seen.setdefault("t0", args[0]))
    try:
        ref = m.forward_features(x)
    finally:
        h.remove()
    t0 = ad.layer0_tokens(x)
    assert t0.shape == (B, ad.n_tokens, ad.embed_dim)
    assert (t0 - seen["t0"]).abs().max() < 1e-5
    out = m.norm(ad.forward_tokens(x))
    assert out.shape == ref.shape
    assert (out - ref).abs().max() < 1e-5


@torch.no_grad()
def test_cls_feature(built):
    short, ad, x = built
    m = ad.model
    feats = m.forward_features(x)
    f = ad.cls_feature(x)
    assert f.shape == (B, ad.embed_dim)
    pooled = m.forward_head(feats, pre_logits=True)
    if EXPECT[short][2] == "cls":
        assert (f - feats[:, 0]).abs().max() < 1e-5
        # global_pool='token' with an Identity fc_norm: the model's own pooled output is the CLS
        assert (f - pooled).abs().max() < 1e-5
    else:
        assert m.attn_pool is not None  # SigLIP2: attention-pooling head, no CLS token
        assert (f - pooled).abs().max() < 1e-5
        assert (f - m(x)).abs().max() < 1e-5  # head is Identity (num_classes=0)
    # a modified copy is honoured through `model=`
    m2 = copy.deepcopy(m)
    m2.norm.bias.add_(1.0)
    assert (ad.cls_feature(x, m2) - f).abs().max() > 1e-3


@torch.no_grad()
def test_pos_embed_grid(built):
    _, ad, _ = built
    H, W = ad.grid_hw
    pos = ad.pos_embed_grid()
    assert pos.shape == (H, W, ad.embed_dim)
    # A black (all zero) image gives patch_embed output = conv bias, so the patch tokens entering
    # block 0 are bias + position embedding (norm_pre is Identity for these models).
    assert isinstance(ad.model.norm_pre, torch.nn.Identity)
    t0 = ad.layer0_tokens(torch.zeros(1, 3, 224, 224))
    bias = ad.model.patch_embed.proj.bias
    expect = pos.reshape(H * W, -1) + (0 if bias is None else bias)
    assert (t0[0, ad.n_prefix_tokens:] - expect).abs().max() < 1e-5


@torch.no_grad()
def test_row_major_and_unflip(built):
    """A single bright patch at (r, c) changes only token r*W + c; mirrored, it lands at (r, W-1-c)
    and unflip_tokens brings it back to (r, c)."""
    _, ad, _ = built
    H, W = ad.grid_hw
    p, npre = ad.patch_size, ad.n_prefix_tokens
    r, c = 3, 2
    img = torch.zeros(1, 3, 224, 224)
    img[..., r * p:(r + 1) * p, c * p:(c + 1) * p] = 1.0
    base = ad.layer0_tokens(torch.zeros_like(img))
    diff = (ad.layer0_tokens(img) - base).abs().sum(-1)[0]
    assert diff.shape[0] == npre + H * W
    changed = torch.nonzero(diff > 1e-6).flatten().tolist()
    assert changed == [npre + r * W + c]
    dm = ad.layer0_tokens(torch.flip(img, dims=[-1])) - base
    assert torch.nonzero(dm.abs().sum(-1)[0] > 1e-6).flatten().tolist() == [npre + r * W + (W - 1 - c)]
    back = unflip_tokens(dm, ad.grid_hw, npre).abs().sum(-1)[0]
    assert torch.nonzero(back > 1e-6).flatten().tolist() == [npre + r * W + c]
    # full token tensors from a real forward pass go through unflip twice unchanged
    t = ad.forward_tokens(img)
    assert torch.equal(unflip_tokens(unflip_tokens(t, ad.grid_hw, npre), ad.grid_hw, npre), t)


@torch.no_grad()
def test_absorb_block_keeps_outputs(built):
    """Existing surgery folding works for LayerScale (DeiT-III) and Identity LayerScale (MAE, SigLIP2)."""
    _, ad, x = built
    t = ad.layer0_tokens(x).double()
    blk = copy.deepcopy(ad.blocks[1]).double()
    ref = blk(t)
    absorb_block(blk)
    assert (blk(t) - ref).abs().max() < 1e-8


# ----------------------------------------------------------------------------------------------
# Cheap checks on a tiny timm ViT (error paths, factory)
# ----------------------------------------------------------------------------------------------
def test_raises_when_image_size_does_not_divide():
    with pytest.raises(ValueError, match="does not divide"):
        TimmPretrainedAdapter("vit_tiny_patch16_224", img_size=40, pretrained=False)


def test_raises_when_not_row_major():
    import timm

    m = timm.create_model("vit_tiny_patch16_224", pretrained=False, img_size=32)
    check_layout(m, 32)
    m.patch_embed.flatten = False
    with pytest.raises(ValueError, match="row-major"):
        check_layout(m, 32)


def test_cls_feature_needs_a_cls_token():
    with pytest.raises(ValueError, match="no CLS token"):
        TimmPretrainedAdapter("vit_tiny_patch16_224", img_size=32, pretrained=False,
                              feature="cls", class_token=False, global_pool="avg")
    ad = TimmPretrainedAdapter("vit_tiny_patch16_224", img_size=32, pretrained=False,
                               class_token=False, global_pool="avg")
    assert ad.feature == "pooled" and ad.n_prefix_tokens == 0


def test_registry_names_and_factory(monkeypatch):
    for short in EXPECT:
        entry = MODEL_REGISTRY[short]
        assert "(verify on VM)" in entry["note"]
        assert resolve_name(entry["timm_name"]) is entry
    with pytest.raises(KeyError):
        resolve_name("not_a_model")

    calls = []

    class FakeTimm:
        def __init__(self, model_name, img_size, pretrained, feature):
            calls.append(("timm", model_name, img_size, pretrained, feature))

        def to(self, device=None):
            return self

    class FakeHub:
        def __init__(self, hub_repo, name, img_size):
            calls.append(("hub", hub_repo, name, img_size))

        def to(self, device=None):
            return self

    import mirror.adapters as base
    import mirror.adapters_timm as mod

    monkeypatch.setattr(mod, "TimmPretrainedAdapter", FakeTimm)
    monkeypatch.setattr(base, "Dinov2Adapter", FakeHub)

    build_model_adapter("siglip2_b16", "cpu", pretrained=False)
    assert calls[-1] == ("timm", "vit_base_patch16_siglip_224.v2_webli", 224, False, "pooled")
    for name in ("dinov2_vits14_reg", "dinov2_vitb14_reg", "dinov2_vitl14_reg", "dinov2_vitb14"):
        ad = build_model_adapter(name, "cpu")
        assert calls[-1] == ("hub", "facebookresearch/dinov2", name, 224)
        assert ad.mean == IMAGENET_MEAN and ad.std == IMAGENET_STD and ad.feature == "cls"
    with pytest.raises(ValueError, match="torch.hub"):
        build_model_adapter("dinov2_vits14_reg", "cpu", pretrained=False)


def test_factory_builds_real_adapter_offline():
    """End to end through the factory with random weights (tiny image size keeps it quick)."""
    ad = build_model_adapter("deit3_b16", "cpu", img_size=64, pretrained=False)
    assert isinstance(ad, TimmPretrainedAdapter)
    assert ad.grid_hw == (4, 4) and ad.n_prefix_tokens == 1 and ad.has_layerscale
    assert ad.pos_embed_grid().shape == (4, 4, 768)
