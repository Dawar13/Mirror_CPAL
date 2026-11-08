"""X4: full LayerNorm folding (SliceGPT-style) for pre-norm ViTs.

Works on DINOv2 hub models (facebookresearch/dinov2) and timm VisionTransformer models whose
blocks have the layout norm1, attn.qkv, attn.proj, ls1.gamma, norm2, mlp.fc1, mlp.fc2, ls2.gamma
and a final `model.norm`.

After `fold_layernorm_full(model)` the model computes exactly the same function, but:
  * every LayerNorm is an `RMSNorm` without affine parameters, x / sqrt(mean(x^2) + eps);
  * the residual stream is mean-zero over channels for every token, so RMSNorm equals LayerNorm
    (for a mean-zero x, mean(x^2) is the variance, so the same eps is exact) and RMSNorm commutes
    with any orthogonal matrix;
  * the norm1 and norm2 affines are folded into attn.qkv and mlp.fc1:
        W <- W diag(gamma),  b <- b + W beta;
  * the LayerScale gammas are folded into attn.proj and mlp.fc2 (the LayerScale modules stay in
    place with gamma = 1, the convention of mirror.surgery.write_absorbed_mlp);
  * every write into the stream is centred with C = I - 11^T/d: the patch embedding, CLS token,
    register tokens, position embeddings and DINOv2 mask token, attn.proj and mlp.fc2:
        W <- C W,  b <- b - mean(b);
  * the final norm's affine (gamma, beta) is returned in `info["final_affine"]` for callers to
    apply to the normalised tokens or the CLS feature (default), or kept inside the model.

Why it is exact: if x is the original stream, the folded stream is C x. Each write is centred, so
C x + C delta = C (x + delta). LayerNorm(x) = LayerNorm(C x) = RMSNorm(C x) because LayerNorm
subtracts the channel mean. Position-embedding interpolation (DINOv2 at 224 px, timm dynamic size)
acts on the spatial axes with the same weights for every channel, so it commutes with C.

The function edits `model` in place: pass a deep copy (or use `fold_copy`). Evaluation only:
dropout, drop-path and token dropout must be inactive (call `.eval()`).
"""
from __future__ import annotations

import copy
from collections import OrderedDict

import torch
import torch.nn as nn

# Pre-norm block classes known to compute x + ls1(attn(norm1(x))), then x + ls2(mlp(norm2(x))).
# "Block": timm and dinov2.layers.block.Block; "NestedTensorBlock": the class the DINOv2 hub uses.
SUPPORTED_BLOCKS = {"Block", "NestedTensorBlock"}

# Parameters that write directly into the residual stream (centred over their last dimension).
_TOKEN_PARAMS = ("cls_token", "register_tokens", "reg_token", "pos_embed", "mask_token")


# ----------------------------------------------------------------------------------------------
# Modules
# ----------------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """RMSNorm without affine parameters: x / sqrt(mean(x^2) + eps) over the last dimension."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y.to(x.dtype)

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.eps:g}"


class ChannelAffine(nn.Module):
    """y = x * weight + bias over the last dimension (the final norm's affine, taken out)."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
        self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}"


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
def _is_identity(mod) -> bool:
    return mod is None or isinstance(mod, nn.Identity)


def _family(model: nn.Module) -> str:
    if hasattr(model, "prepare_tokens_with_masks"):
        return "dinov2"
    if hasattr(model, "_pos_embed"):
        return "timm"
    raise ValueError("unknown model: expected a DINOv2 hub model (prepare_tokens_with_masks) "
                     "or a timm VisionTransformer (_pos_embed)")


def _iter_blocks(model: nn.Module):
    """Transformer blocks in order; flattens DINOv2 BlockChunk lists (their Identity padding is skipped)."""
    for b in model.blocks:
        if hasattr(b, "norm1"):
            yield b
        elif isinstance(b, (nn.ModuleList, nn.Sequential)):
            for c in b:
                if isinstance(c, nn.Identity):
                    continue
                if not hasattr(c, "norm1"):
                    raise ValueError(f"unexpected module inside a block chunk: {type(c).__name__}")
                yield c
        else:
            raise ValueError(f"unexpected block type: {type(b).__name__}")


def _ln_params(norm: nn.Module, d: int, where: str):
    """(gamma, beta, eps) of a LayerNorm over the last dimension, in float64."""
    if not isinstance(norm, nn.LayerNorm):
        raise NotImplementedError(f"{where}: expected nn.LayerNorm, found {type(norm).__name__}")
    if tuple(norm.normalized_shape) != (d,):
        raise NotImplementedError(f"{where}: LayerNorm over {tuple(norm.normalized_shape)}, expected ({d},)")
    ref = next(iter(norm.parameters()), None)
    dev = ref.device if ref is not None else torch.device("cpu")
    gamma = norm.weight.detach().double() if norm.weight is not None else torch.ones(d, dtype=torch.float64, device=dev)
    beta = norm.bias.detach().double() if norm.bias is not None else torch.zeros(d, dtype=torch.float64, device=dev)
    return gamma, beta, float(norm.eps)


def _write(p: torch.Tensor, value: torch.Tensor) -> None:
    p.copy_(value.to(dtype=p.dtype, device=p.device))


@torch.no_grad()
def _fold_norm_into(parent: nn.Module, attr: str, lin: nn.Linear, d: int, where: str) -> float:
    """Fold parent.<attr> (a LayerNorm) into the following linear layer; replace it by RMSNorm."""
    norm = getattr(parent, attr)
    gamma, beta, eps = _ln_params(norm, d, where)
    if not isinstance(lin, nn.Linear) or lin.in_features != d:
        raise NotImplementedError(f"{where}: the layer after the norm must be nn.Linear with in_features {d}")
    W = lin.weight.detach().double()
    b = lin.bias.detach().double() if lin.bias is not None else W.new_zeros(W.shape[0])
    W_new = W * gamma[None, :]
    b_new = b + W @ beta
    _write(lin.weight, W_new)
    if lin.bias is not None:
        _write(lin.bias, b_new)
    elif bool((b_new != 0).any()):
        lin.bias = nn.Parameter(b_new.to(dtype=lin.weight.dtype, device=lin.weight.device),
                                requires_grad=lin.weight.requires_grad)
    dev = lin.weight.device
    setattr(parent, attr, RMSNorm(d, eps).to(dev))
    return eps


@torch.no_grad()
def _fold_scale_and_centre(lin: nn.Linear, ls: nn.Module, d: int, where: str) -> None:
    """W <- C diag(gamma) W, b <- C (gamma * b) for a layer writing into the stream; gamma -> 1."""
    if not isinstance(lin, nn.Linear) or lin.out_features != d:
        raise NotImplementedError(f"{where}: the write into the stream must be nn.Linear with out_features {d}")
    W = lin.weight.detach().double()
    if _is_identity(ls):
        g = None
    else:
        g = getattr(ls, "gamma", None)
        if not isinstance(g, torch.Tensor):
            raise NotImplementedError(f"{where}: LayerScale without a `gamma` tensor ({type(ls).__name__})")
    gd = g.detach().double() if g is not None else W.new_ones(d)
    W = gd[:, None] * W
    _write(lin.weight, W - W.mean(dim=0, keepdim=True))
    if lin.bias is not None:
        b = gd * lin.bias.detach().double()
        _write(lin.bias, b - b.mean())
    if g is not None:
        g.fill_(1.0)


@torch.no_grad()
def _centre_last_dim(p: torch.Tensor) -> None:
    v = p.detach().double()
    _write(p, v - v.mean(dim=-1, keepdim=True))


@torch.no_grad()
def _fold_block(blk: nn.Module, d: int, idx: int) -> list[float]:
    where = f"block {idx}"
    if type(blk).__name__ not in SUPPORTED_BLOCKS:
        raise NotImplementedError(f"{where}: block class {type(blk).__name__} is not a known pre-norm block "
                                  f"(supported: {sorted(SUPPORTED_BLOCKS)}; extend SUPPORTED_BLOCKS if it is)")
    attn, mlp = blk.attn, blk.mlp
    for owner, attr, label in ((attn, "q_norm", "attn.q_norm"), (attn, "k_norm", "attn.k_norm"),
                               (attn, "norm", "attn.norm"), (mlp, "norm", "mlp.norm")):
        if not _is_identity(getattr(owner, attr, None)):
            raise NotImplementedError(f"{where}: {label} is not Identity; not supported")
    if getattr(attn, "gate", None) is not None:
        raise NotImplementedError(f"{where}: gated attention is not supported")
    if not (hasattr(mlp, "fc1") and hasattr(mlp, "fc2")):
        raise NotImplementedError(f"{where}: MLP without fc1/fc2 (for example SwiGLU) is not supported")
    eps1 = _fold_norm_into(blk, "norm1", attn.qkv, d, f"{where} norm1")
    eps2 = _fold_norm_into(blk, "norm2", mlp.fc1, d, f"{where} norm2")
    _fold_scale_and_centre(attn.proj, getattr(blk, "ls1", None), d, f"{where} attn.proj")
    _fold_scale_and_centre(mlp.fc2, getattr(blk, "ls2", None), d, f"{where} mlp.fc2")
    return [eps1, eps2]


@torch.no_grad()
def _centre_embedding(model: nn.Module, d: int, family: str) -> list[str]:
    centred = []
    pe = model.patch_embed
    if not _is_identity(getattr(pe, "norm", None)):
        raise NotImplementedError("patch_embed.norm is not Identity; not supported")
    proj = pe.proj
    if not isinstance(proj, (nn.Conv2d, nn.Linear)) or proj.weight.shape[0] != d:
        raise NotImplementedError(f"patch_embed.proj must be Conv2d or Linear with {d} outputs")
    W = proj.weight.detach().double()
    _write(proj.weight, W - W.mean(dim=0, keepdim=True))
    centred.append("patch_embed.proj.weight")
    if proj.bias is not None:
        b = proj.bias.detach().double()
        _write(proj.bias, b - b.mean())
        centred.append("patch_embed.proj.bias")
    for name in _TOKEN_PARAMS:
        p = getattr(model, name, None)
        if isinstance(p, torch.Tensor):
            if p.shape[-1] != d:
                raise NotImplementedError(f"{name} has last dimension {p.shape[-1]}, expected {d}")
            _centre_last_dim(p)
            centred.append(name)
    if family == "timm":
        if not _is_identity(getattr(model, "norm_pre", None)):
            raise NotImplementedError("timm norm_pre is not Identity; not supported")
        if not _is_identity(getattr(model, "fc_norm", None)):
            raise NotImplementedError("timm fc_norm is not Identity (global average pooling); not supported")
    return centred


# ----------------------------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def fold_layernorm_full(model: nn.Module, final_norm_affine: bool = True) -> dict:
    """Fold every LayerNorm and LayerScale of `model` (in place; pass a deep copy).

    final_norm_affine:
        True (default): the final norm becomes a plain `RMSNorm` and its affine is taken out of the
            model and returned as `info["final_affine"]` (a `ChannelAffine`). Callers apply it to
            the normalised tokens or the CLS feature:
                feat = info["final_affine"](model.norm(tokens))[:, 0]
            so the whole network up to that point commutes with orthogonal changes of the stream
            basis. Note that `model(x)` (and the timm head) then sees features without the affine.
        False: the affine stays inside the model, `model.norm = Sequential(rms=RMSNorm, affine=...)`,
            so `model.norm(tokens)` and `model(x)` match the original directly;
            `info["final_affine"]` is None.

    Returns info with keys: family, embed_dim, n_blocks, eps (sorted distinct LayerNorm eps),
    n_rmsnorm, centred (stream-writing parameters that were centred), final_affine,
    final_affine_in_model, final_weight and final_bias (the final norm's gamma and beta).
    """
    family = _family(model)
    final = getattr(model, "norm", None)
    if not isinstance(final, nn.LayerNorm):
        raise NotImplementedError(f"model.norm must be nn.LayerNorm, found {type(final).__name__}")
    d = int(final.normalized_shape[-1])

    centred = _centre_embedding(model, d, family)
    eps_all: list[float] = []
    blocks = list(_iter_blocks(model))
    for i, blk in enumerate(blocks):
        eps_all += _fold_block(blk, d, i)
        centred += [f"block {i} attn.proj", f"block {i} mlp.fc2"]

    gamma, beta, eps = _ln_params(final, d, "final norm")
    eps_all.append(eps)
    ref = final.weight if final.weight is not None else next(model.parameters())
    rms = RMSNorm(d, eps).to(ref.device)
    affine = ChannelAffine(gamma.to(dtype=ref.dtype, device=ref.device),
                           beta.to(dtype=ref.dtype, device=ref.device))
    if final_norm_affine:
        model.norm = rms
        final_affine = affine
    else:
        model.norm = nn.Sequential(OrderedDict([("rms", rms), ("affine", affine)]))
        final_affine = None

    return {
        "family": family,
        "embed_dim": d,
        "n_blocks": len(blocks),
        "eps": sorted(set(eps_all)),
        "n_rmsnorm": sum(isinstance(m, RMSNorm) for m in model.modules()),
        "centred": centred,
        "final_affine": final_affine,
        "final_affine_in_model": not final_norm_affine,
        "final_weight": affine.weight.detach().clone(),
        "final_bias": affine.bias.detach().clone(),
    }


def fold_copy(model: nn.Module, final_norm_affine: bool = True):
    """(folded deep copy of `model`, info); the original is left untouched."""
    m = copy.deepcopy(model)
    info = fold_layernorm_full(m, final_norm_affine=final_norm_affine)
    return m, info


def apply_final(info: dict, x: torch.Tensor) -> torch.Tensor:
    """Apply the returned final affine to normalised tokens or features (no-op if kept in the model)."""
    aff = info.get("final_affine")
    return x if aff is None else aff(x)
