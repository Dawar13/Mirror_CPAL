"""CORE-04 adapters: the new ViTs (CLIP, DINO v1, DINOv3, EVA-02, DINOv2-g) and the two CNNs.

Every model ID below is copied from configs/models/<short>.yaml (tests/test_core04.py checks that the two agree).
`mirror.adapters_timm.resolve_name` and `build_model_adapter` fall back to this module for names they do not
know, so scripts/measure_groups.py and scripts/must07_compressors.py accept the new ViT names unchanged.

Families
  timm_vit       timm VisionTransformer through adapters_timm.TimmPretrainedAdapter, unchanged:
                 clip_b16 (OpenAI CLIP ViT-B/16) and dino_b16 (DINO v1 ViT-B/16). CLIP's `.openai` weights were
                 trained with QuickGELU, and timm's plain `.openai` tag builds GELU (its own note: "natively
                 QuickGELU, use quickgelu model variant for original results"). The adapter passes
                 act_layer="quick_gelu" to timm.create_model, which overrides the model arguments of
                 vit_base_patch16_clip_224 and loads the same state dict (activations have no weights).
                 CLIP also has pre_norm=True: a LayerNorm (norm_pre) between the embeddings and block 1.
  eva            timm Eva models through `EvaAdapter`: dinov3_b16 (optional, DINOv3 licence) and eva02_b14
                 (optional). Their blocks take the rotary position embedding as an argument, so the adapter's
                 forward_tokens passes it to every block (a plain `blk(t)` would silently drop RoPE). Eva
                 blocks keep LayerScale as `gamma_1` / `gamma_2` parameters, not `ls1` / `ls2` modules.
  dinov2_swiglu  torch.hub DINOv2 through `Dinov2SwiGLUAdapter`: dinov2_vitg14_reg. Its MLP is SwiGLU (packed
                 `w12` gate and value, `w3` out; with xFormers installed the forward is one fused kernel, so
                 no hook could see the hidden product). The adapter replaces every MLP by `SwiGLUMLP`, which
                 computes the same function with separate modules (checked against the original on the VM by
                 `swiglu_conversion_error`; verify on VM).
  convnext, resnet  CNNs (structure only): built by `build_cnn_adapter`, measured by mirror/cnn_groups.py.

SwiGLU rule (DINOv2-g, EVA-02). The MLP is out(x) = W3 N(silu(Wg u + bg) * (Wx u + bx)) + b3 with u the folded
block input (LayerNorm without affine) and N the optional hidden LayerNorm of EVA-02 (`scale_mlp=True`). The
hidden units are the h entries of the product a = silu(g) * v; twins pair those units (the correlation that the
matching uses is computed on a, the input of N). Folding (exact): norm2's affine goes into Wg and Wx alike,
N's affine into W3 (W3 diag(gamma_h), b3 + W3 beta_h), LayerScale into W3 and b3. The group projection with a
hidden permutation P and stream operators rho_in, rho_out acts on both input matrices with the same P,
    Wg_eq = avg_g P^T Wg rho_in,   Wx_eq = avg_g P^T Wx rho_in   (biases bg, bx: avg_g P^T b),
and on the output matrix as for a GELU MLP, W3_eq = avg_g rho_out^T W3 P. The projected MLP is exactly
equivariant: g and v both move by P, the elementwise product and silu commute with P, an affine-free LayerNorm
over the hidden units commutes with every permutation, and W3 maps P a to rho_out times the output. The
breaking share of "fc1" is the share of the two input matrices together; the character formula counts the two
input matrices separately (each has the free parameters of one GELU fc1), so a SwiGLU block's tax is
1 - (free_g + free_x + free_out) / (3 h d). mirror/cnn_groups.py implements the projection
(`project_unit`); tests/test_core04.py checks exact equivariance on synthetic weights.

Rotary position embeddings (EVA-02, DINOv3). Attention logits depend on q_i^T R(theta(p_j) - theta(p_i)) k_j,
R a rotation by theta in each channel pair. The left-right mirror maps patch (r, c) to (r, G-1-c). A pair whose
angle is an even function of the mirrored coordinate (up to a constant) keeps its relative angles; a pair whose
angle is odd (up to a constant) negates them, and R(-a) = F R(a) F with F = diag(1, -1). So RoPE is exactly
equivariant under the flip, with q and k transformed by a fixed sign map S per head (S = -1 on the second channel
of every odd pair), if and only if every frequency pair is even or odd under the mirror: axial RoPE, where each
pair rotates with the row or with the column alone, passes; mixed 2D frequencies (theta = w_x c + w_y r with
both non-zero, timm rope_type "mixed") do not, unless their reflected frequency pairs are also present.
`rope_flip_check` decides this numerically for any Eva model (S found from the embedding, then the logits of
mirrored positions compared through timm's own apply_rot_embed_cat); both CORE-04 RoPE models are axial and
pass. Consequence for surgery: an exactly equivariant attention layer then needs q and k head maps that
commute with this RoPE structure (S on the rotary channels), but mirror/attention.py fits unconstrained head
maps and mirror/fold.py rejects Eva blocks. CORE-04 therefore measures MLP structure only on the RoPE models
(no attention statistics, no surgery) and says so in every output.

Surgery support (scripts/core04_generality.py reads `SURGERY_SUPPORT`): Mirror-exact runs through
scripts/must07_compressors.py, whose machinery (mirror/fold.py, mirror/m1.py, mirror/mirror_family.py,
mirror/baselines.py) handles blocks with fc1/fc2 MLPs, qkv attention, no norm_pre and LayerScale as ls1/ls2.
DINO v1 qualifies; CLIP (norm_pre LayerNorm), DINOv2-g (SwiGLU), EVA-02 and DINOv3 (Eva blocks, RoPE) do not.

CNNs. ConvNeXt-B at 224 px: every stride is a patchify (4 x 4 stem, 2 x 2 downsampling, no padding) on even
sizes (224, 56, 28, 14) and the 7 x 7 depthwise convolutions are centred, so the flip of the image is exactly the
flip of every feature grid. ResNet-50: a stride-2 layer with padding (7 x 7 stem, 3 x 3 max pool, 3 x 3
convolutions) maps the flipped image exactly onto the flipped grid only when its input size is odd; at 224 every
stride-2 layer shifts the mirror by one input pixel. ResNet-50 is therefore measured at 225 px (224 px crops
resized by bilinear interpolation, which commutes with the flip): 225, 113, 57, 29, 15 are all odd, the last
grid is 8 x 8. tests/test_core04.py checks both statements on random networks with flip-symmetric kernels.

Never call the builders with pretrained=True on the laptop: they download weights. Tests use pretrained=False.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import IMAGENET_MEAN, IMAGENET_STD, Dinov2Adapter, ViTAdapter

# DINOv2 evaluation transform, as in adapters_timm (resize 256 bicubic, centre crop 224) (verify).
_DINOV2_PREPROCESS = {"mean": IMAGENET_MEAN, "std": IMAGENET_STD, "interpolation": "bicubic",
                      "crop_pct": 224 / 256}

_NO_SURGERY_ROPE = ("Eva blocks with rotary position embeddings: mirror/fold.py rejects the block class and "
                    "mirror/attention.py fits head maps that ignore the RoPE constraint; structure only (MLP)")
_NO_SURGERY_SWIGLU = ("SwiGLU MLP: mirror/fold.py, mirror/m1.py, mirror/mirror_family.py and mirror/baselines.py "
                      "assume fc1/fc2; structure only (MLP)")
_NO_SURGERY_PRENORM = ("norm_pre LayerNorm before block 1: mirror/fold.py refuses it (its affine cannot be folded "
                       "into a linear write); Mirror-exact would need norm_pre replaced by an affine-free norm plus a "
                       "d x d map projected to commute with A*")

# ----------------------------------------------------------------------------------------------
# Registry (IDs from configs/models/*.yaml; verify on VM: weights not loaded on the laptop)
# ----------------------------------------------------------------------------------------------
EXTRA_REGISTRY: dict[str, dict] = {
    "clip_b16": {
        "source": "extra", "family": "timm_vit", "kind": "vit", "yaml": "configs/models/clip_b16.yaml",
        "timm_name": "vit_base_patch16_clip_224.openai", "create_kwargs": {"act_layer": "quick_gelu"},
        "feature": "cls", "optional": False,
        "why": "OpenAI CLIP ViT-B/16 image tower with QuickGELU (the .openai weights' own activation); the "
               "feature is the CLS token after the final norm, before the 512-d projection head.",
        "note": "(verify on VM: QuickGELU loading, CLS feature)",
    },
    "dino_b16": {
        "source": "extra", "family": "timm_vit", "kind": "vit", "yaml": "configs/models/dino_b16.yaml",
        "timm_name": "vit_base_patch16_224.dino", "create_kwargs": {}, "feature": "cls", "optional": False,
        "why": "DINO v1 ViT-B/16 backbone (self-supervised, ImageNet-1k), standard timm ViT.",
        "note": "(verify on VM)",
    },
    "dinov3_b16": {
        "source": "extra", "family": "eva", "kind": "vit", "yaml": "configs/models/dinov3_b16.yaml",
        "timm_name": "vit_base_patch16_dinov3.lvd1689m", "create_kwargs": {"global_pool": "token"},
        "feature": "cls", "optional": True, "gated": True,
        "why": "DINOv3 ViT-B/16 (timm mirror of the gated release; DINOv3 licence). global_pool='token' gives the "
               "CLS pooling of the original (timm's docstring). RoPE, 4 register tokens, LayerScale.",
        "note": "(verify on VM: access, weights load, RoPE grid at 224 px = 14 x 14)",
    },
    "eva02_b14": {
        "source": "extra", "family": "eva", "kind": "vit", "yaml": "configs/models/eva02_b14.yaml",
        "timm_name": "eva02_base_patch14_224.mim_in22k", "create_kwargs": {}, "feature": "pooled",
        "optional": True,
        "why": "EVA-02 ViT-B/14, MIM pre-trained on IN-21K at 224 px (the only 224 px B/14 weights). SwiGLU MLP "
               "with a hidden LayerNorm, RoPE plus an absolute position embedding. timm's default pooling is "
               "kept ('avg' with fc_norm), so the stored weights load unchanged; the feature is that pooled "
               "embedding (only used for k-NN sanity checks; no surgery on this model).",
        "note": "(verify on VM)",
    },
    "dinov2_vitg14_reg": {
        "source": "extra", "family": "dinov2_swiglu", "kind": "vit", "yaml": "configs/models/dinov2_vitg14_reg.yaml",
        "hub_repo": "facebookresearch/dinov2", "hub_name": "dinov2_vitg14_reg", "feature": "cls",
        "preprocess": dict(_DINOV2_PREPROCESS), "optional": False,
        "why": "DINOv2 ViT-g/14 with registers (40 blocks, d 1536, SwiGLU hidden 4096), the scale end point.",
        "note": "(verify on VM: SwiGLU module names, conversion error)",
    },
    "convnext_b": {
        "source": "extra", "family": "convnext", "kind": "cnn", "yaml": "configs/models/convnext_b.yaml",
        "timm_name": "convnext_base.fb_in1k", "img_size": 224, "optional": False,
        "why": "ConvNeXt-B supervised on ImageNet-1k; structure of the 36 pointwise MLPs.",
        "note": "(verify on VM)",
    },
    "resnet50": {
        "source": "extra", "family": "resnet", "kind": "cnn", "yaml": "configs/models/resnet50.yaml",
        "timm_name": "resnet50.tv_in1k", "img_size": 225, "optional": False,
        "why": "ResNet-50, torchvision IMAGENET1K_V1 weights; structure of the 1 x 1 convolutions of the 16 "
               "bottlenecks, measured at 225 px (see the module docstring).",
        "note": "(verify on VM)",
    },
}

# Mirror-exact surgery through scripts/must07_compressors.py: "must07" or the reason it cannot run.
SURGERY_SUPPORT = {
    "dino_b16": "must07",
    "clip_b16": _NO_SURGERY_PRENORM,
    "dinov3_b16": _NO_SURGERY_ROPE,
    "eva02_b14": _NO_SURGERY_ROPE + "; also SwiGLU",
    "dinov2_vitg14_reg": _NO_SURGERY_SWIGLU,
    "convnext_b": "CNN: structure only (PROJECT_NEXT_PHASE.md CORE-04)",
    "resnet50": "CNN: structure only (PROJECT_NEXT_PHASE.md CORE-04)",
}


def resolve_extra(name: str) -> dict | None:
    """Registry entry for a short name or a timm / hub name, or None (never raises)."""
    if name in EXTRA_REGISTRY:
        return EXTRA_REGISTRY[name]
    for entry in EXTRA_REGISTRY.values():
        if name in (entry.get("timm_name"), entry.get("hub_name")):
            return entry
    return None


def short_name(entry: dict) -> str:
    for k, v in EXTRA_REGISTRY.items():
        if v is entry:
            return k
    raise KeyError("entry not in EXTRA_REGISTRY")


# ----------------------------------------------------------------------------------------------
# SwiGLU
# ----------------------------------------------------------------------------------------------
class SwiGLUMLP(nn.Module):
    """out = fc2(norm(hidden(act(fc1_g(x)) * fc1_x(x)))). `hidden` is an Identity hook point on the product (the
    hidden units that twins pair); `norm` is EVA-02's hidden LayerNorm or Identity."""

    def __init__(self, fc1_g: nn.Linear, fc1_x: nn.Linear, fc2: nn.Linear, act: nn.Module | None = None,
                 norm: nn.Module | None = None):
        super().__init__()
        if fc1_g.weight.shape != fc1_x.weight.shape or fc2.in_features != fc1_g.out_features:
            raise ValueError("SwiGLU shapes do not match")
        self.fc1_g, self.fc1_x, self.fc2 = fc1_g, fc1_x, fc2
        self.act = act if act is not None else nn.SiLU()
        self.hidden = nn.Identity()
        self.norm = norm if norm is not None else nn.Identity()

    @property
    def in_features(self) -> int:
        return int(self.fc1_g.in_features)

    @property
    def hidden_features(self) -> int:
        return int(self.fc1_g.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.act(self.fc1_g(x)) * self.fc1_x(x)
        return self.fc2(self.norm(self.hidden(a)))


def _linear_like(weight: torch.Tensor, bias: torch.Tensor | None) -> nn.Linear:
    out_f, in_f = weight.shape
    lin = nn.Linear(in_f, out_f, bias=bias is not None, device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        lin.weight.copy_(weight)
        if bias is not None:
            lin.bias.copy_(bias)
    lin.weight.requires_grad_(weight.requires_grad)
    if bias is not None:
        lin.bias.requires_grad_(weight.requires_grad)
    return lin


def is_swiglu(mlp: nn.Module) -> bool:
    return isinstance(mlp, SwiGLUMLP)


def swiglu_from_module(mlp: nn.Module) -> SwiGLUMLP:
    """SwiGLUMLP computing the same function as `mlp`:
    * timm SwiGLU (fc1_g, fc1_x, act, norm, fc2): the modules are reused;
    * DINOv2 SwiGLUFFN / SwiGLUFFNFused / xformers SwiGLU with packed `w12` (gate = first half) and `w3`;
    * the same with unpacked `w1` (gate), `w2` (value), `w3`;
    * timm GluMlp with gate_last=False (fc1 packed, gate = first half, norm, fc2)."""
    if isinstance(mlp, SwiGLUMLP):
        return mlp
    if all(hasattr(mlp, n) for n in ("fc1_g", "fc1_x", "fc2")):
        act = getattr(mlp, "act", None)
        return SwiGLUMLP(mlp.fc1_g, mlp.fc1_x, mlp.fc2, act=act, norm=getattr(mlp, "norm", None))
    if hasattr(mlp, "w12") and hasattr(mlp, "w3"):
        W, b = mlp.w12.weight, mlp.w12.bias
        h = W.shape[0] // 2
        gate = _linear_like(W[:h], None if b is None else b[:h])
        value = _linear_like(W[h:], None if b is None else b[h:])
        return SwiGLUMLP(gate, value, mlp.w3, act=nn.SiLU())
    if all(hasattr(mlp, n) for n in ("w1", "w2", "w3")):
        return SwiGLUMLP(mlp.w1, mlp.w2, mlp.w3, act=nn.SiLU())
    if type(mlp).__name__ == "GluMlp" and hasattr(mlp, "fc1") and hasattr(mlp, "fc2"):
        if getattr(mlp, "gate_last", True):
            raise NotImplementedError("GluMlp with gate_last=True is not supported")
        W, b = mlp.fc1.weight, mlp.fc1.bias
        h = W.shape[0] // 2
        gate = _linear_like(W[:h], None if b is None else b[:h])
        value = _linear_like(W[h:], None if b is None else b[h:])
        return SwiGLUMLP(gate, value, mlp.fc2, act=mlp.act, norm=getattr(mlp, "norm", None))
    raise NotImplementedError(f"not a SwiGLU MLP: {type(mlp).__name__}")


def convert_swiglu_blocks(model: nn.Module) -> int:
    """Replace every SwiGLU-style block MLP by SwiGLUMLP (in place); returns how many were replaced."""
    n = 0
    for blk in model.blocks:
        mlp = blk.mlp
        if isinstance(mlp, SwiGLUMLP) or (hasattr(mlp, "fc1") and type(mlp).__name__ != "GluMlp"):
            continue
        blk.mlp = swiglu_from_module(mlp)
        n += 1
    return n


@torch.no_grad()
def swiglu_conversion_error(original: nn.Module, converted: nn.Module, x: torch.Tensor) -> float:
    """max |original(x) - converted(x)| / max |original(x)| (VM check of the DINOv2-g conversion)."""
    a, b = original(x), converted(x)
    return float((a - b).abs().max() / a.abs().max().clamp_min(1e-30))


# ----------------------------------------------------------------------------------------------
# Folding of a ViT MLP (GELU or SwiGLU), LayerScale as ls2.gamma or Eva's gamma_2
# ----------------------------------------------------------------------------------------------
def layerscale2(block: nn.Module) -> torch.Tensor | None:
    """The MLP branch's LayerScale vector: block.ls2.gamma (timm, DINOv2), block.gamma_2 (Eva) or None."""
    g = getattr(getattr(block, "ls2", None), "gamma", None)
    if isinstance(g, torch.Tensor):
        return g
    g = getattr(block, "gamma_2", None)
    return g if isinstance(g, torch.Tensor) else None


def _affine(norm: nn.Module, n: int, like: torch.Tensor, dtype) -> tuple:
    w = getattr(norm, "weight", None)
    b = getattr(norm, "bias", None)
    w = w.to(dtype) if isinstance(w, torch.Tensor) else like.new_ones(n, dtype=dtype)
    b = b.to(dtype) if isinstance(b, torch.Tensor) else like.new_zeros(n, dtype=dtype)
    return w, b


def _check_norm(norm: nn.Module, where: str) -> None:
    if not isinstance(norm, (nn.Identity, nn.LayerNorm)):
        raise NotImplementedError(f"{where}: {type(norm).__name__} is neither LayerNorm nor Identity")


@torch.no_grad()
def fold_vit_mlp(block: nn.Module, dtype: torch.dtype = torch.float64) -> dict:
    """Folded MLP of one pre-norm ViT block (exact, same function):
        {"W1": [h x d matrices], "b1": [h vectors], "W2": d x h, "b2": d, "kind": "gelu" | "swiglu"}.
    norm2's affine is folded into every input matrix, a hidden LayerNorm's affine and LayerScale into W2. For a
    GELU MLP the result equals mirror.surgery.fold_mlp (one input matrix)."""
    norm2, mlp = block.norm2, block.mlp
    _check_norm(norm2, "norm2")
    if is_swiglu(mlp):
        ins = [mlp.fc1_g, mlp.fc1_x]
        kind = "swiglu"
    elif hasattr(mlp, "fc1") and hasattr(mlp, "fc2"):
        ins = [mlp.fc1]
        kind = "gelu"
    else:
        raise NotImplementedError(f"unsupported MLP {type(mlp).__name__} (convert SwiGLU first)")
    hnorm = getattr(mlp, "norm", nn.Identity())
    _check_norm(hnorm, "mlp.norm")
    d = ins[0].in_features
    gamma, beta = _affine(norm2, d, ins[0].weight, dtype)
    W1s, b1s = [], []
    for lin in ins:
        W = lin.weight.to(dtype)
        b = lin.bias.to(dtype) if lin.bias is not None else W.new_zeros(W.shape[0])
        W1s.append(W * gamma[None, :])
        b1s.append(b + W @ beta)
    fc2 = mlp.fc2
    W2 = fc2.weight.to(dtype)
    b2 = fc2.bias.to(dtype) if fc2.bias is not None else W2.new_zeros(W2.shape[0])
    if isinstance(hnorm, nn.LayerNorm):
        gh, bh = _affine(hnorm, W2.shape[1], W2, dtype)
        b2 = b2 + W2 @ bh
        W2 = W2 * gh[None, :]
    ls = layerscale2(block)
    ls = ls.to(dtype) if ls is not None else W2.new_ones(W2.shape[0])
    return {"W1": W1s, "b1": b1s, "W2": ls[:, None] * W2, "b2": ls * b2, "kind": kind}


@torch.no_grad()
def write_absorbed_vit_mlp(block: nn.Module, W1, b1, W2, b2) -> None:
    """Write folded weights (W1, b1 lists for SwiGLU) and reset norm2, the hidden norm and LayerScale to identity."""
    mlp = block.mlp
    ins = [mlp.fc1_g, mlp.fc1_x] if is_swiglu(mlp) else [mlp.fc1]
    W1 = W1 if isinstance(W1, (list, tuple)) else [W1]
    b1 = b1 if isinstance(b1, (list, tuple)) else [b1]
    if len(W1) != len(ins):
        raise ValueError(f"{len(W1)} input matrices for {len(ins)} input layers")
    for lin, W, b in zip(ins, W1, b1):
        if lin.bias is None:
            raise ValueError("absorbed form needs input biases")
        lin.weight.copy_(W.to(lin.weight))
        lin.bias.copy_(b.to(lin.bias))
    if mlp.fc2.bias is None:
        raise ValueError("absorbed form needs an output bias")
    mlp.fc2.weight.copy_(W2.to(mlp.fc2.weight))
    mlp.fc2.bias.copy_(b2.to(mlp.fc2.bias))
    for norm in (block.norm2, getattr(mlp, "norm", None)):
        if isinstance(norm, nn.LayerNorm):
            if norm.weight is not None:
                norm.weight.fill_(1.0)
            if norm.bias is not None:
                norm.bias.zero_()
    ls = layerscale2(block)
    if ls is not None:
        ls.fill_(1.0)


def mlp_hidden_of(mlp: nn.Module) -> int:
    return int(mlp.fc1_g.out_features) if is_swiglu(mlp) else int(mlp.fc1.out_features)


def mlp_in_of(mlp: nn.Module) -> int:
    return int(mlp.fc1_g.in_features) if is_swiglu(mlp) else int(mlp.fc1.in_features)


class _MLPDims:
    """embed_dim and mlp_hidden for SwiGLU as well as GELU MLPs (mixin before ViTAdapter)."""

    @property
    def embed_dim(self) -> int:
        return mlp_in_of(self.model.blocks[0].mlp)

    @property
    def mlp_hidden(self) -> int:
        return mlp_hidden_of(self.model.blocks[0].mlp)

    @property
    def mlp_kind(self) -> str:
        return "swiglu" if is_swiglu(self.model.blocks[0].mlp) else "gelu"


# ----------------------------------------------------------------------------------------------
# DINOv2-g (torch.hub, SwiGLU)
# ----------------------------------------------------------------------------------------------
class Dinov2SwiGLUAdapter(_MLPDims, Dinov2Adapter):
    """DINOv2 hub model whose SwiGLU MLPs are replaced by SwiGLUMLP (same function, separate modules)."""

    def __init__(self, hub_repo: str = "facebookresearch/dinov2", name: str = "dinov2_vitg14_reg",
                 img_size: int = 224):
        import copy

        super().__init__(hub_repo=hub_repo, name=name, img_size=img_size)
        self._original_mlp0 = copy.deepcopy(self.model.blocks[0].mlp)   # kept for check_conversion
        self.original_mlp_class = type(self._original_mlp0).__name__
        self.n_swiglu_converted = convert_swiglu_blocks(self.model)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def check_conversion(self, n_tokens: int = 64, seed: int = 0) -> dict:
        """Block 1's original MLP against its SwiGLUMLP on random tokens (on the model's device and dtype)."""
        dev, dt = self.device, self.dtype
        orig = self._original_mlp0.to(device=dev, dtype=dt).eval()
        g = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn(2, n_tokens, self.embed_dim, generator=g).to(device=dev, dtype=dt)
        err = swiglu_conversion_error(orig, self.model.blocks[0].mlp, x)
        self._original_mlp0 = orig.cpu()
        return {"original_class": self.original_mlp_class, "n_converted": int(self.n_swiglu_converted),
                "max_rel_err_block1": err}


# ----------------------------------------------------------------------------------------------
# Eva (DINOv3, EVA-02)
# ----------------------------------------------------------------------------------------------
def check_layout_eva(model: nn.Module, img_size: int) -> int:
    """Check that a timm Eva model fits the adapter interface; return its (square) patch size."""
    from timm.layers import Format, PatchEmbed
    from timm.models.eva import Eva, EvaBlock

    if not isinstance(model, Eva):
        raise ValueError(f"expected a timm Eva model, got {type(model).__name__}")
    pe = model.patch_embed
    if not isinstance(pe, PatchEmbed):
        raise ValueError(f"unsupported patch embedding {type(pe).__name__}")
    ph, pw = pe.patch_size
    if ph != pw or img_size % ph != 0:
        raise ValueError(f"patch {pe.patch_size} and image size {img_size} do not give a square grid")
    if getattr(pe, "dynamic_img_pad", False):
        raise ValueError("dynamic padding is not supported")
    if model.dynamic_img_size:
        # NHWC from the patch embedding, flattened by view(B, -1, C) in _pos_embed: row-major (r * W + c).
        if pe.flatten or pe.output_fmt != Format.NHWC:
            raise ValueError("dynamic Eva model without NHWC patch output: token order unknown")
    elif not (pe.flatten and pe.output_fmt == Format.NCHW):
        raise ValueError("patch tokens are not guaranteed to be row-major (H, W)")
    if getattr(model, "patch_drop", None) is not None:
        raise ValueError("patch dropout is not supported")
    if getattr(model, "rope_mixed", False):
        raise ValueError("mixed (depth-dependent, 2D-frequency) RoPE is not flip-equivariant in general")
    g = img_size // ph
    if model.pos_embed is not None and not model.dynamic_img_size:
        n_pos = model.pos_embed.shape[1]
        expect = g * g + (0 if model.no_embed_class else int(model.num_prefix_tokens))
        if n_pos != expect:
            raise ValueError(f"pos_embed has {n_pos} entries, expected {expect}")
    for i, blk in enumerate(model.blocks):
        if not isinstance(blk, EvaBlock):
            raise ValueError(f"block {i} is {type(blk).__name__}, expected the pre-norm EvaBlock")
        for path in ("norm1", "attn", "norm2", "mlp"):
            if getattr(blk, path, None) is None:
                raise ValueError(f"block {i} has no {path}")
    return int(ph)


class EvaAdapter(_MLPDims, ViTAdapter):
    """A timm Eva model (DINOv3, EVA-02) behind the ViTAdapter interface. forward_tokens passes the rotary
    embedding to every block. SwiGLU MLPs are converted to SwiGLUMLP (same function)."""

    def __init__(self, model_name: str, img_size: int = 224, pretrained: bool = True, feature: str | None = None,
                 **create_kwargs):
        import timm
        from timm.data import resolve_data_config

        model = timm.create_model(model_name, pretrained=pretrained, img_size=img_size, **create_kwargs)
        patch = check_layout_eva(model, int(img_size))
        self.n_swiglu_converted = convert_swiglu_blocks(model)
        has_cls = getattr(model, "cls_token", None) is not None
        feature = feature or ("cls" if has_cls else "pooled")
        if feature not in ("cls", "pooled"):
            raise ValueError(f"feature must be cls or pooled, got {feature!r}")
        if feature == "cls" and not has_cls:
            raise ValueError(f"{model_name} has no CLS token; use feature='pooled'")
        super().__init__(model, img_size=img_size, patch_size=patch, n_prefix_tokens=int(model.num_prefix_tokens))
        self.model_name, self.feature, self.has_cls_token = model_name, feature, has_cls
        self.uses_rope = model.rope is not None
        self.has_layerscale = any(getattr(b, "gamma_2", None) is not None for b in model.blocks)
        dcfg = resolve_data_config({}, model=model)
        self.mean = tuple(float(v) for v in dcfg["mean"])
        self.std = tuple(float(v) for v in dcfg["std"])
        self.interpolation = str(dcfg["interpolation"])
        self.crop_pct = float(dcfg["crop_pct"])

    def _embed(self, x: torch.Tensor, m: nn.Module):
        t = m.patch_embed(x)
        t, rope = m._pos_embed(t)
        return m.norm_pre(t), rope

    def layer0_tokens(self, x: torch.Tensor, model: nn.Module | None = None) -> torch.Tensor:
        return self._embed(x, self._m(model))[0]

    def forward_tokens(self, x: torch.Tensor, model: nn.Module | None = None) -> torch.Tensor:
        m = self._m(model)
        t, rope = self._embed(x, m)
        for blk in m.blocks:
            t = blk(t, rope=rope)
        return t

    def cls_feature(self, x: torch.Tensor, model: nn.Module | None = None) -> torch.Tensor:
        m = self._m(model)
        t = m.norm(self.forward_tokens(x, m))
        if self.feature == "pooled":
            return m.forward_head(t, pre_logits=True)
        return t[:, 0]

    def pos_embed_grid(self, model: nn.Module | None = None) -> torch.Tensor:
        m = self._m(model)
        if m.pos_embed is None:
            raise ValueError(f"{self.model_name} has no absolute position embedding (RoPE only)")
        H, W = self.grid_hw
        pos = m.pos_embed[0]
        if not m.no_embed_class:
            pos = pos[self.n_prefix_tokens:]
        return pos.reshape(H, W, -1)

    @torch.no_grad()
    def rope_embed(self, model: nn.Module | None = None) -> torch.Tensor | None:
        """The rotary embedding the blocks receive for this grid, [H W, 2 dh] (sin, cos), or None."""
        m = self._m(model)
        if m.rope is None:
            return None
        x = torch.zeros(1, 3, self.img_size, self.img_size, dtype=self.dtype, device=self.device)
        return self._embed(x, m)[1]


# ----------------------------------------------------------------------------------------------
# RoPE flip check
# ----------------------------------------------------------------------------------------------
def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@torch.no_grad()
def rope_flip_check(emb: torch.Tensor, grid_hw, half: bool, element: str = "LR", seed: int = 0,
                    tol: float = 1e-4) -> dict:
    """Is RoPE with embedding `emb` ([H W, 2 dh], (sin, cos) concatenated, as timm's apply_rot_embed_cat takes it)
    equivariant under `element` (LR or UD) with a fixed sign map S on q and k?

    Per channel pair the angle theta(p) = atan2(sin, cos) is tested for theta(g p) = theta(p) + c (even, S = +1)
    or theta(g p) = -theta(p) + c (odd, S = -1 on the pair's second channel). If every pair is one or the other,
    S is built and the logits are compared directly through timm's apply_rot_embed_cat on random q and k:
    logits of the transformed positions with S q, S k against the original logits. Returns ok, the number of
    pairs, odd pairs, pairs that are neither, and the largest relative logit error."""
    from timm.layers.pos_embed_sincos import apply_rot_embed_cat

    from .groups import token_index

    emb = torch.as_tensor(emb).detach().to(torch.float64).cpu()
    if emb.dim() == 3 and emb.shape[0] == 1:
        emb = emb[0]
    if emb.dim() != 2:
        raise ValueError(f"expected a [N, 2 dh] embedding, got {tuple(emb.shape)}")
    N, two_dh = emb.shape
    dh = two_dh // 2
    sin, cos = emb[:, :dh].numpy(), emb[:, dh:].numpy()
    if half:
        first, second = np.arange(dh // 2), np.arange(dh // 2) + dh // 2
    else:
        first, second = np.arange(0, dh, 2), np.arange(1, dh, 2)
    idx = token_index(element, grid_hw)
    if len(idx) != N:
        raise ValueError(f"embedding has {N} positions, grid {grid_hw} has {len(idx)}")
    theta = np.arctan2(sin[:, first], cos[:, first])           # [N, pairs]
    th_g = theta[idx]                                            # angle at g(p)
    even_dev = _wrap(th_g - theta - _wrap(th_g[0] - theta[0])[None, :])
    odd_dev = _wrap(th_g + theta - _wrap(th_g[0] + theta[0])[None, :])
    is_even = np.abs(even_dev).max(axis=0) < tol
    is_odd = np.abs(odd_dev).max(axis=0) < tol
    neither = ~(is_even | is_odd)
    S = np.ones(dh)
    S[second[is_odd & ~is_even]] = -1.0
    out = {"element": element, "layout": "half" if half else "interleaved", "n_pairs": int(len(first)),
           "n_odd": int((is_odd & ~is_even).sum()), "n_even": int(is_even.sum()), "n_neither": int(neither.sum()),
           "sign_map": S}
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(N, dh, generator=g, dtype=torch.float64)
    k = torch.randn(N, dh, generator=g, dtype=torch.float64)
    St = torch.as_tensor(S)
    ref = apply_rot_embed_cat(q, emb, half=half) @ apply_rot_embed_cat(k, emb, half=half).T
    it = torch.as_tensor(idx)
    qm, km = torch.zeros_like(q), torch.zeros_like(k)
    qm[it], km[it] = q * St, k * St                             # the token at g(p) carries S q(p)
    got = apply_rot_embed_cat(qm, emb, half=half) @ apply_rot_embed_cat(km, emb, half=half).T
    err = float((got[it][:, it] - ref).abs().max() / ref.abs().max())
    out["max_rel_logit_err"] = err
    out["ok"] = bool(not neither.any() and err < 1e-6)
    return out


def rope_flip_check_adapter(adapter: EvaAdapter, element: str = "LR") -> dict:
    """rope_flip_check on an EvaAdapter's own embedding and rotation layout (sign map dropped for JSON)."""
    emb = adapter.rope_embed()
    if emb is None:
        return {"ok": None, "note": "no RoPE"}
    half = bool(getattr(adapter.model.blocks[0].attn, "rotate_half", False))
    r = rope_flip_check(emb, adapter.grid_hw, half, element)
    r.pop("sign_map", None)
    return r


# ----------------------------------------------------------------------------------------------
# CNNs
# ----------------------------------------------------------------------------------------------
def flip_exact_size(family: str, size: int) -> bool:
    """True when the flip of a size x size image is exactly the flip of every feature grid of the family."""
    if family == "convnext":
        return size % 32 == 0
    if family == "resnet":
        return size % 32 == 1
    raise ValueError(f"unknown CNN family {family!r}")


class CNNAdapter:
    """A timm CNN for the structure measurement: the model, its own normalisation and the measurement size.
    `prepare` normalises uint8 images and resizes them bilinearly (align_corners=False, which commutes with the
    flip) to `input_size` when the batch has another size."""

    def __init__(self, model: nn.Module, family: str, input_size: int, mean, std, name: str = "",
                 interpolation: str = "", crop_pct: float = float("nan")):
        if not flip_exact_size(family, int(input_size)):
            raise ValueError(f"{family} at {input_size} px: the flip is not exact on every feature grid "
                             "(ConvNeXt needs a multiple of 32, ResNet 32 k + 1)")
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.family, self.input_size, self.model_name = family, int(input_size), name
        self.mean, self.std = tuple(float(v) for v in mean), tuple(float(v) for v in std)
        self.interpolation, self.crop_pct = interpolation, crop_pct

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def to(self, device=None, dtype=None) -> "CNNAdapter":
        self.model.to(device=device, dtype=dtype)
        return self

    def prepare(self, imgs: torch.Tensor, device) -> torch.Tensor:
        x = imgs.to(device, non_blocking=True)
        if x.dtype == torch.uint8:
            m = torch.tensor(self.mean, dtype=self.dtype, device=x.device).view(1, 3, 1, 1)
            s = torch.tensor(self.std, dtype=self.dtype, device=x.device).view(1, 3, 1, 1)
            x = (x.to(self.dtype) / 255.0 - m) / s
        else:
            x = x.to(self.dtype)
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False,
                              antialias=False)
        return x

    def forward(self, x: torch.Tensor):
        return self.model.forward_features(x)


def build_cnn_adapter(entry: dict, pretrained: bool = True, input_size: int | None = None,
                      **create_kwargs) -> CNNAdapter:
    """ConvNeXt or ResNet from the registry entry; `create_kwargs` override timm's model arguments (tests build
    tiny random versions with pretrained=False)."""
    import timm
    from timm.data import resolve_data_config

    if entry.get("kind") != "cnn":
        raise ValueError("not a CNN entry")
    model = timm.create_model(entry["timm_name"], pretrained=pretrained, **create_kwargs)
    dcfg = resolve_data_config({}, model=model)
    return CNNAdapter(model, entry["family"], int(input_size or entry["img_size"]), dcfg["mean"], dcfg["std"],
                      name=entry["timm_name"], interpolation=str(dcfg["interpolation"]),
                      crop_pct=float(dcfg["crop_pct"]))


# ----------------------------------------------------------------------------------------------
# Factory (called by adapters_timm.build_model_adapter for the ViT entries)
# ----------------------------------------------------------------------------------------------
def build_vit_adapter(entry: dict, img_size: int = 224, pretrained: bool = True, **create_kwargs) -> ViTAdapter:
    """ViT adapter of an EXTRA_REGISTRY entry. Every returned adapter has mean, std, interpolation, crop_pct,
    feature and model_name. `create_kwargs` override timm's model arguments (tests: tiny random versions)."""
    fam = entry["family"]
    if entry.get("kind") != "vit":
        raise ValueError(f"{entry.get('timm_name')} is a CNN: use build_cnn_adapter and mirror/cnn_groups.py")
    if fam == "timm_vit":
        from .adapters_timm import TimmPretrainedAdapter

        kw = {**entry.get("create_kwargs", {}), **create_kwargs}
        return TimmPretrainedAdapter(entry["timm_name"], img_size=img_size, pretrained=pretrained,
                                     feature=entry["feature"], **kw)
    if fam == "eva":
        kw = {**entry.get("create_kwargs", {}), **create_kwargs}
        return EvaAdapter(entry["timm_name"], img_size=img_size, pretrained=pretrained, feature=entry["feature"], **kw)
    if fam == "dinov2_swiglu":
        if not pretrained or create_kwargs:
            raise ValueError("DINOv2 hub entries load through torch.hub and always download weights")
        ad = Dinov2SwiGLUAdapter(hub_repo=entry["hub_repo"], name=entry["hub_name"], img_size=img_size)
        pp = entry["preprocess"]
        ad.mean, ad.std = tuple(pp["mean"]), tuple(pp["std"])
        ad.interpolation, ad.crop_pct = pp["interpolation"], float(pp["crop_pct"])
        ad.feature, ad.model_name = entry["feature"], entry["hub_name"]
        return ad
    raise ValueError(f"unknown family {fam!r}")


def build_extra_adapter(name_or_entry, device=None, img_size: int | None = None, pretrained: bool = True,
                        **create_kwargs):
    """Any EXTRA_REGISTRY model (ViT or CNN) by short name, timm name or entry; moved to `device`."""
    entry = name_or_entry if isinstance(name_or_entry, dict) else resolve_extra(name_or_entry)
    if entry is None:
        raise KeyError(f"unknown model {name_or_entry!r}; known: {sorted(EXTRA_REGISTRY)}")
    if entry.get("kind") == "cnn":
        ad = build_cnn_adapter(entry, pretrained=pretrained, input_size=img_size, **create_kwargs)
    else:
        ad = build_vit_adapter(entry, img_size=int(img_size or 224), pretrained=pretrained, **create_kwargs)
    return ad.to(device=device) if device is not None else ad

