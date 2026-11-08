"""Adapters for the generality track (G): pretrained timm ViTs plus a name-based factory.

`TimmPretrainedAdapter` wraps a timm `VisionTransformer` behind the same interface as
`mirror.adapters.ViTAdapter`:
    layer0_tokens(x, model=None)  tokens entering block 0, [B, T, d]
    forward_tokens(x, model=None) residual stream after the last block (before the final norm)
    cls_feature(x, model=None)    the feature used for k-NN, [B, d] (see FEATURE below)
    pos_embed_grid(model=None)    patch position embeddings as [H, W, d]
    blocks, grid_hw, n_prefix_tokens, embed_dim, mlp_hidden, depth
    mean, std, interpolation, crop_pct   the model's own preprocessing (from its pretrained_cfg)

`layer0_tokens` repeats timm's `forward_features` up to the first block (patch_embed, _pos_embed,
patch_drop, norm_pre), so running the blocks and then `model.norm` reproduces
`model.forward_features` exactly. Prefix tokens (CLS, registers) come first; patch tokens follow
in row-major (H, W) order, which `mirror.stats.unflip_tokens` relies on. The constructor raises if
that layout cannot be guaranteed.

`build_model_adapter(name, device)` builds any model in MODEL_REGISTRY. DINOv2 entries reuse the
torch.hub `Dinov2Adapter` from `mirror.adapters`; the others use `TimmPretrainedAdapter`.

Never call the factory with pretrained=True on the laptop: it downloads weights. Tests build the
same architectures with pretrained=False (random weights, no download).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import adapters as _adapters
from .adapters import IMAGENET_MEAN, IMAGENET_STD, ViTAdapter

# ----------------------------------------------------------------------------------------------
# Registry (verify on VM: names, tags and preprocessing were read on the laptop with timm 1.0.31,
# using timm.list_models(..., pretrained=True) and resolve_data_config on the pretrained_cfg,
# neither of which downloads anything; the weights themselves have not been loaded yet)
# ----------------------------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict] = {
    "deit3_b16": {
        "source": "timm",
        "timm_name": "deit3_base_patch16_224.fb_in1k",
        "feature": "cls",
        "why": "DeiT-III ViT-B/16, supervised on ImageNet-1k only, 224 px. Chosen over "
               "fb_in22k_ft_in1k so that the training data match the other ImageNet-1k models "
               "and no 22k pre-training is mixed in. Has LayerScale; CLS has no position "
               "embedding (no_embed_class=True).",
        "listing": "timm.list_models('*deit3*base*', pretrained=True) -> "
                   "deit3_base_patch16_224.fb_in1k, deit3_base_patch16_224.fb_in22k_ft_in1k, "
                   "deit3_base_patch16_384.fb_in1k, deit3_base_patch16_384.fb_in22k_ft_in1k",
        "preprocess": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD,
                       "interpolation": "bicubic", "crop_pct": 0.9},
        "note": "(verify on VM)",
    },
    "mae_b16": {
        "source": "timm",
        "timm_name": "vit_base_patch16_224.mae",
        "feature": "cls",
        "why": "MAE ViT-B/16 encoder after self-supervised pre-training only, no fine-tune "
               "(num_classes=0 in its pretrained_cfg). No LayerScale. Its CLS gives weak k-NN; "
               "the plan uses the Tier-2 ImageNet linear probe as its accuracy measure.",
        "listing": "timm.list_models('*mae*', pretrained=True) -> ViT entries: "
                   "vit_base_patch16_224.mae, vit_large_patch16_224.mae, "
                   "vit_huge_patch14_224.mae (the rest are ConvNeXt-V2 fcmae and Hiera)",
        "preprocess": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD,
                       "interpolation": "bicubic", "crop_pct": 0.9},
        "note": "(verify on VM)",
    },
    "siglip2_b16": {
        "source": "timm",
        "timm_name": "vit_base_patch16_siglip_224.v2_webli",
        "feature": "pooled",
        "why": "SigLIP2 ViT-B/16 image tower at 224 px with its attention-pooling ('map') head, "
               "which produces the embedding trained against text. No CLS token "
               "(num_prefix_tokens=0) and no LayerScale. The '_gap_' variant drops the map head "
               "and was not chosen. 'siglip2' does not match in timm: the SigLIP2 weights carry "
               "the tag .v2_webli.",
        "listing": "timm.list_models('*siglip2*', pretrained=True) -> []; "
                   "timm.list_models('*siglip*', pretrained=True), ViT-B/16 at 224 px with tag "
                   "v2_webli -> vit_base_patch16_siglip_224.v2_webli, "
                   "vit_base_patch16_siglip_gap_224.v2_webli",
        "preprocess": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5),
                       "interpolation": "bicubic", "crop_pct": 0.9},
        "note": "(verify on VM)",
    },
}
# DINOv2 via torch.hub (facebookresearch/dinov2). The preprocessing below is the DINOv2 evaluation
# transform as we understand it (resize 256 bicubic, centre crop 224, ImageNet mean and std) (verify).
_DINOV2_PREPROCESS = {"mean": IMAGENET_MEAN, "std": IMAGENET_STD,
                      "interpolation": "bicubic", "crop_pct": 224 / 256}
for _name in ("dinov2_vits14_reg", "dinov2_vitb14_reg", "dinov2_vitl14_reg",
              "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"):
    MODEL_REGISTRY[_name] = {
        "source": "torch_hub",
        "hub_repo": "facebookresearch/dinov2",
        "hub_name": _name,
        "feature": "cls",
        "why": "DINOv2 through the existing Dinov2Adapter (CLS after the final norm).",
        "preprocess": dict(_DINOV2_PREPROCESS),
        "note": "(verify on VM)",
    }
del _name

FEATURES = ("cls", "pooled")


# ----------------------------------------------------------------------------------------------
# Layout checks
# ----------------------------------------------------------------------------------------------
def check_layout(model: nn.Module, img_size: int) -> int:
    """Check that `model` fits the adapter interface; return its (square) patch size.

    Raises ValueError if the image size does not divide by the patch size, if patch tokens are not
    guaranteed to be in row-major (H, W) order, or if the blocks lack the expected module names.
    """
    from timm.layers import Format, PatchEmbed
    from timm.models.vision_transformer import VisionTransformer

    if not isinstance(model, VisionTransformer):
        raise ValueError(f"expected a timm VisionTransformer, got {type(model).__name__}")
    pe = model.patch_embed
    if not isinstance(pe, PatchEmbed):
        raise ValueError(f"unsupported patch embedding {type(pe).__name__}")
    ph, pw = pe.patch_size
    if ph != pw:
        raise ValueError(f"non-square patches {pe.patch_size} are not supported")
    if img_size % ph != 0:
        raise ValueError(f"image size {img_size} does not divide by patch size {ph}")
    if getattr(model, "dynamic_img_size", False) or getattr(pe, "dynamic_img_pad", False):
        raise ValueError("dynamic image size or padding is not supported (fixed grid needed)")
    # Row-major: conv output [B, C, H, W] -> flatten(2).transpose(1, 2) gives index r * W + c.
    if not (pe.flatten and pe.output_fmt == Format.NCHW):
        raise ValueError("patch tokens are not guaranteed to be row-major (H, W); "
                         f"flatten={pe.flatten}, output_fmt={pe.output_fmt}")
    g = img_size // ph
    if tuple(pe.grid_size) != (g, g):
        raise ValueError(f"patch grid {tuple(pe.grid_size)} does not match {img_size}/{ph}")
    if model.pos_embed is not None:
        n_pos = model.pos_embed.shape[1]
        expect = g * g + (0 if model.no_embed_class else int(model.num_prefix_tokens))
        if n_pos != expect:
            raise ValueError(f"pos_embed has {n_pos} entries, expected {expect}")
    for i, blk in enumerate(model.blocks):
        for path in ("norm1", "attn.qkv", "attn.proj", "ls1", "norm2", "mlp.fc1", "mlp.fc2", "ls2"):
            obj = blk
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    raise ValueError(f"block {i} has no {path}")
        for ls in ("ls1", "ls2"):
            mod = getattr(blk, ls)
            if not (isinstance(mod, nn.Identity) or isinstance(getattr(mod, "gamma", None), torch.Tensor)):
                raise ValueError(f"block {i}.{ls} is neither LayerScale nor Identity")
    return int(ph)


# ----------------------------------------------------------------------------------------------
# Adapter
# ----------------------------------------------------------------------------------------------
class TimmPretrainedAdapter(ViTAdapter):
    """A timm ViT (DeiT-III, MAE, SigLIP2 and similar) behind the ViTAdapter interface.

    feature: "cls" uses the CLS token after the final norm; "pooled" uses the model's own pooled
    embedding, `forward_head(norm(tokens), pre_logits=True)` (for SigLIP2 the attention-pooling
    head). If None, "cls" is used when the model has a CLS token and "pooled" otherwise.
    pretrained=False builds the same architecture with random weights (tests; no download).
    """

    def __init__(self, model_name: str, img_size: int = 224, pretrained: bool = True,
                 feature: str | None = None, **create_kwargs):
        import timm
        from timm.data import resolve_data_config

        model = timm.create_model(model_name, pretrained=pretrained, img_size=img_size,
                                  **create_kwargs)
        patch = check_layout(model, int(img_size))
        has_cls = getattr(model, "cls_token", None) is not None
        if feature is None:
            feature = "cls" if has_cls else "pooled"
        if feature not in FEATURES:
            raise ValueError(f"feature must be one of {FEATURES}, got {feature!r}")
        if feature == "cls" and not has_cls:
            raise ValueError(f"{model_name} has no CLS token; use feature='pooled'")
        super().__init__(model, img_size=img_size, patch_size=patch,
                         n_prefix_tokens=int(model.num_prefix_tokens))
        self.model_name = model_name
        self.feature = feature
        self.has_cls_token = has_cls
        self.has_layerscale = any(not isinstance(b.ls1, nn.Identity) or not isinstance(b.ls2, nn.Identity)
                                  for b in model.blocks)
        dcfg = resolve_data_config({}, model=model)
        self.mean = tuple(float(v) for v in dcfg["mean"])
        self.std = tuple(float(v) for v in dcfg["std"])
        self.interpolation = str(dcfg["interpolation"])
        self.crop_pct = float(dcfg["crop_pct"])

    def layer0_tokens(self, x: torch.Tensor, model: nn.Module | None = None) -> torch.Tensor:
        m = self._m(model)
        t = m.patch_embed(x)
        t = m._pos_embed(t)
        t = m.patch_drop(t)
        return m.norm_pre(t)

    def pooled_feature(self, x: torch.Tensor, model: nn.Module | None = None) -> torch.Tensor:
        """The model's own pooled embedding: forward_head(norm(tokens), pre_logits=True)."""
        m = self._m(model)
        return m.forward_head(m.norm(self.forward_tokens(x, m)), pre_logits=True)

    def cls_feature(self, x: torch.Tensor, model: nn.Module | None = None) -> torch.Tensor:
        if self.feature == "pooled":
            return self.pooled_feature(x, model)
        m = self._m(model)
        return m.norm(self.forward_tokens(x, m))[:, 0]

    def pos_embed_grid(self, model: nn.Module | None = None) -> torch.Tensor:
        m = self._m(model)
        if m.pos_embed is None:
            raise ValueError(f"{self.model_name} has no absolute position embedding")
        H, W = self.grid_hw
        pos = m.pos_embed[0]
        if not m.no_embed_class:
            pos = pos[self.n_prefix_tokens:]
        return pos.reshape(H, W, -1)


# ----------------------------------------------------------------------------------------------
# Factory and helpers
# ----------------------------------------------------------------------------------------------
def resolve_name(name: str) -> dict:
    """Registry entry for a short name, a full timm name or a DINOv2 hub name."""
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    for entry in MODEL_REGISTRY.values():
        if entry.get("timm_name") == name:
            return entry
    raise KeyError(f"unknown model {name!r}; known: {sorted(MODEL_REGISTRY)}")


def build_model_adapter(name: str, device=None, img_size: int = 224,
                        pretrained: bool = True) -> ViTAdapter:
    """Build the adapter for a registry name (or its full timm name) and move it to `device`.

    Every returned adapter has `mean`, `std`, `interpolation` and `crop_pct`. DINOv2 entries always
    download through torch.hub, so pretrained=False is refused for them.
    """
    entry = resolve_name(name)
    if entry["source"] == "timm":
        adapter = TimmPretrainedAdapter(entry["timm_name"], img_size=img_size,
                                        pretrained=pretrained, feature=entry["feature"])
    elif entry["source"] == "torch_hub":
        if not pretrained:
            raise ValueError("DINOv2 entries load through torch.hub and always download weights")
        adapter = _adapters.Dinov2Adapter(hub_repo=entry["hub_repo"], name=entry["hub_name"],
                                          img_size=img_size)
        pp = entry["preprocess"]
        adapter.mean, adapter.std = tuple(pp["mean"]), tuple(pp["std"])
        adapter.interpolation, adapter.crop_pct = pp["interpolation"], float(pp["crop_pct"])
        adapter.feature, adapter.model_name = "cls", entry["hub_name"]
    else:
        raise ValueError(f"unknown source {entry['source']!r}")
    return adapter.to(device=device) if device is not None else adapter


def normalize_for(adapter, x_uint8: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """[B, 3, H, W] uint8 -> float normalised with the adapter's own mean and std.

    `mirror.data.normalize` always uses the ImageNet constants, which is wrong for SigLIP2
    (mean 0.5, std 0.5). Adapters without `mean`/`std` fall back to ImageNet.
    """
    mean = getattr(adapter, "mean", IMAGENET_MEAN)
    std = getattr(adapter, "std", IMAGENET_STD)
    m = torch.tensor(mean, dtype=dtype, device=x_uint8.device).view(1, 3, 1, 1)
    s = torch.tensor(std, dtype=dtype, device=x_uint8.device).view(1, 3, 1, 1)
    return (x_uint8.to(dtype) / 255.0 - m) / s
