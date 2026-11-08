"""Model adapters: one interface over DINOv2 (torch.hub) and a tiny random timm ViT (tests).

Interface used by all other code:
    layer0_tokens(x, model=None)  tokens entering block 0, [B, T, d]
    blocks                        the nn.ModuleList of transformer blocks
    grid_hw                       (H, W) patch grid
    n_prefix_tokens               CLS plus register tokens (they precede the patch tokens)
    cls_feature(x, model=None)    final-norm CLS token, [B, d]
    pos_embed_grid(model=None)    patch position embedding as [H, W, d]

`model` lets the same adapter run a modified deep copy of the wrapped network (surgery variants).
Both implementations use the same block layout: blocks[i].norm1, attn.qkv, attn.proj, ls1.gamma,
norm2, mlp.fc1, mlp.fc2, ls2.gamma.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ViTAdapter:
    def __init__(self, model: nn.Module, img_size: int, patch_size: int, n_prefix_tokens: int):
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.grid_hw = (self.img_size // self.patch_size, self.img_size // self.patch_size)
        self.n_prefix_tokens = int(n_prefix_tokens)

    # ----- properties -------------------------------------------------------------------------
    @property
    def blocks(self) -> nn.ModuleList:
        return self.model.blocks

    @property
    def depth(self) -> int:
        return len(self.model.blocks)

    @property
    def embed_dim(self) -> int:
        return int(self.model.blocks[0].mlp.fc1.in_features)

    @property
    def mlp_hidden(self) -> int:
        return int(self.model.blocks[0].mlp.fc1.out_features)

    @property
    def n_tokens(self) -> int:
        return self.n_prefix_tokens + self.grid_hw[0] * self.grid_hw[1]

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def to(self, device=None, dtype=None) -> "ViTAdapter":
        self.model.to(device=device, dtype=dtype)
        return self

    def _m(self, model: nn.Module | None) -> nn.Module:
        return self.model if model is None else model

    # ----- forward pieces ---------------------------------------------------------------------
    def layer0_tokens(self, x: torch.Tensor, model: nn.Module | None = None) -> torch.Tensor:
        raise NotImplementedError

    def forward_tokens(self, x: torch.Tensor, model: nn.Module | None = None) -> torch.Tensor:
        """Residual stream after the last block (before the final norm)."""
        m = self._m(model)
        t = self.layer0_tokens(x, m)
        for blk in m.blocks:
            t = blk(t)
        return t

    def cls_feature(self, x: torch.Tensor, model: nn.Module | None = None) -> torch.Tensor:
        m = self._m(model)
        return m.norm(self.forward_tokens(x, m))[:, 0]

    def pos_embed_grid(self, model: nn.Module | None = None) -> torch.Tensor:
        raise NotImplementedError


class Dinov2Adapter(ViTAdapter):
    """facebookresearch/dinov2 via torch.hub. Downloads go under $HF_HOME/torch_hub if HF_HOME is set."""

    def __init__(self, hub_repo: str = "facebookresearch/dinov2", name: str = "dinov2_vitb14",
                 img_size: int = 224):
        hf_home = os.environ.get("HF_HOME")
        if hf_home and "TORCH_HOME" not in os.environ:
            torch.hub.set_dir(os.path.join(hf_home, "torch_hub"))
        # Prefer the local copy of the hub repo (no GitHub call; GitHub returned 504s on the VM).
        local = os.path.join(torch.hub.get_dir(), hub_repo.replace("/", "_") + "_main")
        if os.path.isfile(os.path.join(local, "hubconf.py")):
            model = torch.hub.load(local, name, source="local")
        else:
            model = torch.hub.load(hub_repo, name, trust_repo=True)
        if getattr(model, "chunked_blocks", False):
            raise ValueError("Chunked DINOv2 blocks are not supported; load with block_chunks=0.")
        n_prefix = 1 + int(getattr(model, "num_register_tokens", 0))
        super().__init__(model, img_size=img_size, patch_size=model.patch_size, n_prefix_tokens=n_prefix)

    def layer0_tokens(self, x, model=None):
        return self._m(model).prepare_tokens_with_masks(x)

    def pos_embed_grid(self, model=None):
        m = self._m(model)
        H, W = self.grid_hw
        d = m.pos_embed.shape[-1]
        dummy = torch.zeros(1, 1 + H * W, d, dtype=m.pos_embed.dtype, device=m.pos_embed.device)
        with torch.no_grad():
            pos = m.interpolate_pos_encoding(dummy, self.img_size, self.img_size)
        return pos[0, 1:].reshape(H, W, d)


class TimmTinyAdapter(ViTAdapter):
    """Tiny random timm ViT with the DINOv2 module names. For CPU tests only.

    With `randomise=True` the LayerNorm affines, LayerScale gammas and linear biases get
    non-trivial random values, so folding and surgery tests are not passed by accident
    (timm initialises LayerScale at 1e-5 and biases at 0).
    """

    def __init__(self, img_size: int = 56, seed: int = 0, randomise: bool = True,
                 dtype: torch.dtype = torch.float32):
        from timm.models.vision_transformer import VisionTransformer

        torch.manual_seed(seed)
        model = VisionTransformer(img_size=img_size, patch_size=14, embed_dim=64, depth=4,
                                  num_heads=4, init_values=1e-5)
        if randomise:
            _randomise_affine(model, seed)
        model = model.to(dtype)
        super().__init__(model, img_size=img_size, patch_size=14,
                         n_prefix_tokens=int(model.num_prefix_tokens))

    def layer0_tokens(self, x, model=None):
        m = self._m(model)
        t = m.patch_embed(x)
        t = m._pos_embed(t)
        t = m.patch_drop(t)
        return m.norm_pre(t)

    def pos_embed_grid(self, model=None):
        m = self._m(model)
        H, W = self.grid_hw
        pos = m.pos_embed[0]
        if not getattr(m, "no_embed_class", False):
            pos = pos[self.n_prefix_tokens:]
        return pos.reshape(H, W, -1)


@torch.no_grad()
def _randomise_affine(model: nn.Module, seed: int) -> None:
    g = torch.Generator().manual_seed(seed + 12345)
    for mod in model.modules():
        if isinstance(mod, nn.LayerNorm) and mod.elementwise_affine:
            mod.weight.copy_(1.0 + 0.3 * torch.randn(mod.weight.shape, generator=g))
            mod.bias.copy_(0.1 * torch.randn(mod.bias.shape, generator=g))
        elif isinstance(mod, nn.Linear) and mod.bias is not None:
            mod.bias.copy_(0.02 * torch.randn(mod.bias.shape, generator=g))
        gamma = getattr(mod, "gamma", None)
        if isinstance(gamma, nn.Parameter):
            gamma.copy_(0.5 + torch.rand(gamma.shape, generator=g))


def build_adapter(cfg: dict, device: torch.device) -> ViTAdapter:
    if cfg.get("tiny_mode"):
        adapter = TimmTinyAdapter(img_size=int(cfg["data"]["crop"]), seed=int(cfg["seed"]))
    elif cfg["model"]["name"].startswith("dinov2"):
        m = cfg["model"]
        adapter = Dinov2Adapter(hub_repo=m["hub_repo"], name=m["name"], img_size=int(cfg["data"]["crop"]))
    else:  # DeiT-III, MAE, SigLIP2 and other timm ViTs (track G)
        from .adapters_timm import build_model_adapter

        adapter = build_model_adapter(cfg["model"]["name"], img_size=int(cfg["data"]["crop"]))
    return adapter.to(device=device)
