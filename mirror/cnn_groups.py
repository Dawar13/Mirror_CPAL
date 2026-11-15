"""Unit-based symmetry measurement for CORE-04: the CNNs, and the ViTs that scripts/measure_groups.py cannot take.

scripts/measure_groups.py (with mirror/group_stats.py) measures ViTs whose blocks have fc1/fc2 MLPs, qkv attention
and take no extra forward arguments. This module measures everything else with the same maths (mirror/groups.py)
and the same definitions:
  * ConvNeXt-B: the pointwise MLP of every block (Linear, GELU, Linear on channels-last tokens);
  * ResNet-50: the two 1 x 1 convolutions of every bottleneck (conv1 with bn1, conv3 with bn3);
  * DINOv2-g (SwiGLU), EVA-02 (SwiGLU, RoPE) and DINOv3 (RoPE): the MLP of every block. Attention is not
    measured here (mirror/adapters_extra.py explains why RoPE needs constrained head maps).

Units. A unit is one MLP-like pair of layers: input matrices W1 (one, or two for SwiGLU) from an input space to
hidden units, and an output matrix W2 from hidden units to an output space, all folded (norm affines, LayerScale and
BatchNorm absorbed, so the folded unit computes the same function). Each unit names
  in_space   the token space W1 reads (operator rho_in);
  out_space  the token space W2 writes into (operator rho_out);
  hid_in     the hidden layer matched for W1 (permutation sigma_1);
  hid_out    the hidden layer matched for W2 (sigma_2; the same as hid_in for an MLP, the conv2 output for a
             bottleneck, whose 3 x 3 convolution sits between the two 1 x 1 convolutions and is not converted).
Spaces are shared between units when the same tensor is meant (a ViT's residual layer l is the output space of
block l and the input space of block l + 1), so they get one operator, exactly as in measure_groups.py.

Tokens. Every spatial position of a feature map is a token, as every patch is in a ViT. The action of a group
element on a feature map is the same flip of its spatial grid (LR: columns, UD: rows, R180: both); the transformed
image's feature map is aligned back to the original positions by applying the flip again (involutions only:
LR, UD, R180). ViT tokens keep prefix tokens in place (mirror.groups.align_tokens). For the CNNs the flip is exact on
every grid only at the sizes of mirror/adapters_extra.flip_exact_size (ConvNeXt 224 px, ResNet 225 px).

Per space and group: the involution fit of mirror.groups.fit_order2 on the cross-moment C = X^T Xt (channel-centred
where a LayerNorm follows: ViT residual layers and ConvNeXt's depthwise-convolution output; uncentred elsewhere:
ConvNeXt's block output, the ResNet stream, whose readers do not remove the channel mean), its error against the
identity, e(A)/e(I), and the mean error of random involutions with the same multiplicities, divided by the fitted
error (null_over_fit). Centred spaces fix the all-ones vector in the fit and in the null (fix_ones), as measure_groups;
uncentred spaces fix the mean activation direction in the null (`null_operator`), which the fit keeps by itself.

Per hidden layer and group: the mutual matching (mirror.matching.match_many) of the correlation
R[i, j] = corr(zt_i, z_j) between the aligned transformed and the original hidden units, z the pre-activation
(ViT GELU fc1 output, ConvNeXt fc1 output, ResNet bn1 and bn2 outputs) or, for SwiGLU, the gated product (the
hidden-units rule of mirror/adapters_extra.py). Orbit fractions (mirror.groups.orbit_fractions): twin fraction at
correlation 0.5, 0.6, 0.7, 0.8 (PREREG_NEXT_PHASE.md Section 1). A bottleneck's fractions are the mean of its two
hidden layers (each is also reported).

Per unit and group: breaking shares of W1 (all input matrices together), W2 and the biases under the group
projection (`project_unit`, which calls mirror.groups.project_mlp_group), activation-weighted shares with the
second moments of the folded inputs (M1: the affine-free LayerNorm output where a LayerNorm precedes W1, else the raw
input; M2: the input of W2), the same for --null-draws draws of random operators with the same multiplicities and
random hidden permutations with the same cycle type, and the free parameters of the character formula,
free = (1/|G|) sum_g tr(rho(g)) fix(sigma(g)) per matrix (SwiGLU: each input matrix counted), tax = 1 - free/dense.
Null seeds follow scripts/measure_groups.py (seed_for with slot = space slot, 3000 + hidden slot); for a ViT the
slots are the layer and block numbers, so the nulls equal measure_groups.py's draw for draw (tested).

Range summaries (metrics.json) use measure_groups.py's definitions: ratio_to_null = mean over the range's units of
(s_fc1 + s_fc2) / 2 divided by the mean of the matching null means; twin_fraction_t50 = mean over units of the
2-cycle share at 0.5; tax_mlp = 1 - sum free / sum dense over the range's matrices.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import groups as GR
from .stats import MLPStats, ResidualStats, _flat64

ENGINE_GROUPS = ("LR", "UD", "R180")
GROUP_INDEX = {g: i for i, g in enumerate(GR.GROUPS)}      # as scripts/measure_groups.py
SHARE_KEYS = ("s_fc1", "s_fc2", "s_mlp", "s_b1", "s_b2", "s_fc1_act", "s_fc2_act")
UNIT_KEY = ["group", "block"]
SPACE_KEY = ["group", "space"]
DRAW_KEY = ["group", "block", "draw"]


def seed_for(base: int, group: str, draw: int, slot: int) -> int:
    """Null seeds, the formula of scripts/measure_groups.py seed_for."""
    return int(base) * 10_000_000 + GROUP_INDEX[group] * 1_000_000 + int(draw) * 10_000 + int(slot)


class _Stop(Exception):
    pass


# ----------------------------------------------------------------------------------------------
# Specification
# ----------------------------------------------------------------------------------------------
@dataclass
class Capture:
    """A tensor seen by a forward hook. role: space (operator moments), hidden (matching correlation), m1 or m2
    (second moment of a folded input, original half only). layout: tokens [B, T, C] (prefix then row-major grid),
    nchw or nhwc."""

    key: str
    module: nn.Module
    where: str
    layout: str
    role: str
    centre: bool = False
    ln_eps: float | None = None


@dataclass
class Unit:
    name: str
    index: int
    stage: int
    in_space: str
    out_space: str
    hid_in: str
    hid_out: str
    m1: str
    m2: str
    fold: Callable[[], dict]
    kind: str = "mlp"


@dataclass
class MeasureSpec:
    name: str
    family: str
    captures: list
    units: list
    prepare: Callable
    forward: Callable
    slots: dict
    grid: tuple | None = None
    n_prefix: int = 0
    stop_module: nn.Module | None = None
    info: dict = field(default_factory=dict)

    def capture(self, key: str) -> Capture:
        for c in self.captures:
            if c.key == key:
                return c
        raise KeyError(key)

    def centre_of(self, space: str) -> bool:
        return bool(self.capture(space).centre)


# ----------------------------------------------------------------------------------------------
# Folding (CNNs; ViTs use mirror.adapters_extra.fold_vit_mlp)
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def fold_convnext_block(blk: nn.Module, dtype=torch.float64) -> dict:
    """ConvNeXt pointwise MLP: the channel LayerNorm's affine into fc1, the block's gamma into fc2."""
    n, fc1, fc2 = blk.norm, blk.mlp.fc1, blk.mlp.fc2
    W1 = fc1.weight.to(dtype)
    d = W1.shape[1]
    g = n.weight.to(dtype) if n.weight is not None else W1.new_ones(d)
    b = n.bias.to(dtype) if n.bias is not None else W1.new_zeros(d)
    b1 = fc1.bias.to(dtype) if fc1.bias is not None else W1.new_zeros(W1.shape[0])
    W2 = fc2.weight.to(dtype)
    b2 = fc2.bias.to(dtype) if fc2.bias is not None else W2.new_zeros(W2.shape[0])
    ls = blk.gamma.to(dtype) if getattr(blk, "gamma", None) is not None else W2.new_ones(W2.shape[0])
    return {"W1": [W1 * g[None, :]], "b1": [b1 + W1 @ b], "W2": ls[:, None] * W2, "b2": ls * b2, "kind": "mlp"}


def _bn_fold(conv: nn.Conv2d, bn: nn.BatchNorm2d, dtype):
    W = conv.weight.to(dtype)[:, :, 0, 0]
    s = bn.weight.to(dtype) / torch.sqrt(bn.running_var.to(dtype) + bn.eps)
    b = bn.bias.to(dtype) - s * bn.running_mean.to(dtype)
    if conv.bias is not None:
        b = b + s * conv.bias.to(dtype)
    return s[:, None] * W, b


@torch.no_grad()
def fold_bottleneck(blk: nn.Module, dtype=torch.float64) -> dict:
    """ResNet bottleneck 1 x 1 convolutions with their BatchNorms (eval statistics) folded in."""
    W1, b1 = _bn_fold(blk.conv1, blk.bn1, dtype)
    W2, b2 = _bn_fold(blk.conv3, blk.bn3, dtype)
    return {"W1": [W1], "b1": [b1], "W2": W2, "b2": b2, "kind": "bottleneck"}


# ----------------------------------------------------------------------------------------------
# Specs
# ----------------------------------------------------------------------------------------------
def vit_spec(adapter, blocks=None, name: str = "") -> MeasureSpec:
    """Units = the MLPs of `blocks` (0-based) of a ViT adapter (GELU or SwiGLU; RoPE handled by the adapter)."""
    from .adapters_extra import fold_vit_mlp, is_swiglu
    from .group_stats import prepare_images
    from .variants import span_layers

    m = adapter.model
    L = len(m.blocks)
    blocks = list(range(L)) if blocks is None else sorted(int(b) for b in blocks)
    caps, units, slots = [], [], {}
    for l in span_layers(blocks):
        if l == 0:
            caps.append(Capture("L0", m.blocks[0], "pre", "tokens", "space", centre=True))
        else:
            caps.append(Capture(f"L{l}", m.blocks[l - 1], "post", "tokens", "space", centre=True))
        slots[f"L{l}"] = l
    for b in blocks:
        blk = m.blocks[b]
        mlp = blk.mlp
        if not isinstance(blk.norm2, nn.LayerNorm):
            raise ValueError(f"block {b}: norm2 is {type(blk.norm2).__name__}, expected LayerNorm")
        sw = is_swiglu(mlp)
        caps.append(Capture(f"h{b}", mlp.hidden if sw else mlp.fc1, "post", "tokens", "hidden"))
        caps.append(Capture(f"m1_{b}", blk.norm2, "pre", "tokens", "m1", ln_eps=float(blk.norm2.eps)))
        hn = getattr(mlp, "norm", None)
        if isinstance(hn, nn.LayerNorm):
            caps.append(Capture(f"m2_{b}", hn, "pre", "tokens", "m2", ln_eps=float(hn.eps)))
        else:
            caps.append(Capture(f"m2_{b}", mlp.fc2, "pre", "tokens", "m2"))
        slots[f"h{b}"] = 3000 + b
        units.append(Unit(f"block{b + 1}", b, 0, f"L{b}", f"L{b + 1}", f"h{b}", f"h{b}", f"m1_{b}", f"m2_{b}",
                          (lambda blk=blk: fold_vit_mlp(blk)), "swiglu" if sw else "mlp"))
    return MeasureSpec(name=name or getattr(adapter, "model_name", "vit"), family="vit", captures=caps, units=units,
                       prepare=lambda imgs, dev: prepare_images(adapter, imgs, dev),
                       forward=lambda x: adapter.forward_tokens(x), slots=slots, grid=tuple(adapter.grid_hw),
                       n_prefix=int(adapter.n_prefix_tokens),
                       stop_module=m.blocks[blocks[-1]] if blocks[-1] < L - 1 else None,
                       info={"depth": L, "d": int(adapter.embed_dim), "h": int(adapter.mlp_hidden),
                             "grid": list(adapter.grid_hw), "n_prefix_tokens": int(adapter.n_prefix_tokens),
                             "mlp_kind": units[0].kind})


def convnext_spec(cnn, name: str = "") -> MeasureSpec:
    """Units = the pointwise MLPs of every ConvNeXt block (timm ConvNeXt, conv_mlp=False)."""
    m = cnn.model
    caps, units, slots = [], [], {}
    u = 0
    last = None
    for s, stage in enumerate(m.stages):
        for blk in stage.blocks:
            if getattr(blk, "use_conv_mlp", False) or not isinstance(blk.norm, nn.LayerNorm):
                raise NotImplementedError("ConvNeXt block with conv_mlp or a non-LayerNorm norm")
            if not isinstance(blk.shortcut, nn.Identity) or not isinstance(getattr(blk.mlp, "norm", nn.Identity()),
                                                                           nn.Identity):
                raise NotImplementedError("ConvNeXt block with a projection shortcut or an MLP norm")
            caps += [Capture(f"dw{u}", blk.norm, "pre", "nhwc", "space", centre=True),
                     Capture(f"out{u}", blk, "post", "nchw", "space", centre=False),
                     Capture(f"h{u}", blk.mlp.fc1, "post", "nhwc", "hidden"),
                     Capture(f"m1_{u}", blk.norm, "pre", "nhwc", "m1", ln_eps=float(blk.norm.eps)),
                     Capture(f"m2_{u}", blk.mlp.fc2, "pre", "nhwc", "m2")]
            slots.update({f"dw{u}": 2 * u, f"out{u}": 2 * u + 1, f"h{u}": 3000 + u})
            units.append(Unit(f"stage{s + 1}.block{len([x for x in units if x.stage == s]) + 1}", u, s, f"dw{u}",
                              f"out{u}", f"h{u}", f"h{u}", f"m1_{u}", f"m2_{u}",
                              (lambda blk=blk: fold_convnext_block(blk)), "mlp"))
            last = blk
            u += 1
    return MeasureSpec(name=name or cnn.model_name, family="convnext", captures=caps, units=units,
                       prepare=cnn.prepare, forward=cnn.forward, slots=slots, stop_module=last,
                       info={"input_size": cnn.input_size, "n_units": u, "n_stages": len(m.stages)})


def resnet_spec(cnn, name: str = "") -> MeasureSpec:
    """Units = the two 1 x 1 convolutions of every bottleneck (timm ResNet)."""
    m = cnn.model
    caps, units, slots = [], [], {}
    blocks = [(s, blk) for s, layer in enumerate((m.layer1, m.layer2, m.layer3, m.layer4)) for blk in layer]
    for u, (s, blk) in enumerate(blocks):
        if type(blk).__name__ != "Bottleneck":
            raise NotImplementedError(f"unit {u}: {type(blk).__name__} is not a Bottleneck")
        for conv in (blk.conv1, blk.conv3):
            if conv.kernel_size != (1, 1) or conv.groups != 1 or conv.stride != (1, 1):
                raise NotImplementedError("conv1 and conv3 must be plain 1 x 1 convolutions")
        if getattr(blk, "se", None) is not None or not isinstance(blk.bn1, nn.BatchNorm2d):
            raise NotImplementedError("bottleneck with SE or a non-BatchNorm norm")
        if u == 0:
            caps.append(Capture("res0", blk, "pre", "nchw", "space"))
        caps += [Capture(f"res{u + 1}", blk, "post", "nchw", "space"),
                 Capture(f"h1_{u}", blk.bn1, "post", "nchw", "hidden"),
                 Capture(f"h2_{u}", blk.bn2, "post", "nchw", "hidden"),
                 Capture(f"m1_{u}", blk, "pre", "nchw", "m1"),
                 Capture(f"m2_{u}", blk.conv3, "pre", "nchw", "m2")]
        slots.update({f"res{u}": u, f"res{u + 1}": u + 1, f"h1_{u}": 3000 + 2 * u, f"h2_{u}": 3001 + 2 * u})
        j = len([x for x in units if x.stage == s]) + 1
        units.append(Unit(f"layer{s + 1}.block{j}", u, s, f"res{u}", f"res{u + 1}", f"h1_{u}", f"h2_{u}",
                          f"m1_{u}", f"m2_{u}", (lambda blk=blk: fold_bottleneck(blk)), "bottleneck"))
    return MeasureSpec(name=name or cnn.model_name, family="resnet", captures=caps, units=units,
                       prepare=cnn.prepare, forward=cnn.forward, slots=slots, stop_module=blocks[-1][1],
                       info={"input_size": cnn.input_size, "n_units": len(blocks), "n_stages": 4})


# ----------------------------------------------------------------------------------------------
# Paired pass
# ----------------------------------------------------------------------------------------------
def _tokens(t: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "tokens":
        return t
    if layout == "nchw":
        return t.permute(0, 2, 3, 1).reshape(t.shape[0], -1, t.shape[1])
    if layout == "nhwc":
        return t.reshape(t.shape[0], -1, t.shape[-1])
    raise ValueError(f"unknown layout {layout!r}")


def align(t: torch.Tensor, layout: str, element: str, idx=None, n_prefix: int = 0) -> torch.Tensor:
    """The transformed image's tensor re-indexed to the original positions (involutions: the flip again)."""
    if element not in ENGINE_GROUPS:
        raise ValueError(f"the engine handles involutions only ({ENGINE_GROUPS}), not {element!r}")
    if layout == "tokens":
        return GR.align_tokens(t, idx, n_prefix)
    if layout == "nchw":
        return GR.image_action(t, element)
    if layout == "nhwc":
        return GR.image_action(t.permute(0, 3, 1, 2), element).permute(0, 2, 3, 1)
    raise ValueError(f"unknown layout {layout!r}")


@torch.no_grad()
def collect_pass(spec: MeasureSpec, loader, device, element: str, inputs: bool = True) -> dict:
    """One pass of [x; g(x)] over `loader`: space moments, hidden correlations and (with `inputs`) the second
    moments of the folded inputs. Raw activations are never kept."""
    idx = GR.token_index(element, spec.grid) if spec.grid is not None else None
    idx_t = None if idx is None else torch.as_tensor(idx, dtype=torch.long)
    space, hidden, mom, ssum = {}, {}, {}, {}

    def handle(cap: Capture, t):
        if isinstance(t, (tuple, list)):
            t = t[0]
        b = t.shape[0] // 2
        if cap.role in ("space", "hidden"):
            X = _tokens(t[:b], cap.layout)
            Xt = _tokens(align(t[b:], cap.layout, element, idx_t, spec.n_prefix), cap.layout)
            C = X.shape[-1]
            if cap.role == "space":
                if cap.key not in space:
                    space[cap.key] = ResidualStats(1, C, device, center=cap.centre)
                    ssum[cap.key] = torch.zeros(C, dtype=torch.float64, device=device)
                space[cap.key].update(0, X, Xt)
                ssum[cap.key] += _flat64(X).sum(0)
            else:
                if cap.key not in hidden:
                    hidden[cap.key] = MLPStats(1, C, device)
                hidden[cap.key].update(0, X, Xt)
        elif inputs:
            U = _flat64(_tokens(t[:b], cap.layout))
            C = U.shape[-1]
            if cap.ln_eps is not None:
                U = F.layer_norm(U, (C,), None, None, cap.ln_eps)
            if cap.key not in mom:
                mom[cap.key] = [torch.zeros(C, C, dtype=torch.float64, device=device), 0]
            mom[cap.key][0] += U.T @ U
            mom[cap.key][1] += U.shape[0]

    handles = []
    for cap in spec.captures:
        if cap.role in ("m1", "m2") and not inputs:
            continue
        if cap.where == "pre":
            handles.append(cap.module.register_forward_pre_hook(lambda _m, args, cap=cap: handle(cap, args[0])))
        else:
            handles.append(cap.module.register_forward_hook(lambda _m, _a, out, cap=cap: handle(cap, out)))
    if spec.stop_module is not None:
        def stop(_m, _a, _o):
            raise _Stop()

        handles.append(spec.stop_module.register_forward_hook(stop))
    n = 0
    try:
        for imgs, _ in loader:
            x = spec.prepare(imgs, device)
            try:
                spec.forward(torch.cat([x, GR.image_action(x, element)], dim=0))
            except _Stop:
                pass
            n += int(x.shape[0])
    finally:
        for h in handles:
            h.remove()
    out = {"n_images": n, "space": {}, "R": {}, "inputs": {}}
    for k, st in space.items():
        r = st.numpy()
        out["space"][k] = {"C": r["C"][0], "sq_x": float(r["sq_x"][0]), "sq_xt": float(r["sq_xt"][0]),
                           "n": float(r["n"][0]), "s_x": ssum[k].cpu().numpy()}
    for k, st in hidden.items():
        out["R"][k] = st.correlation(0)
    for k, (M, cnt) in mom.items():
        out["inputs"][k] = {"M": M.cpu().numpy(), "n": cnt}
    return out


def match_hidden(R: dict, top_k: int, workers: int) -> dict:
    """Mutual matching (mirror.matching.match_many) per hidden layer: perm, score, self_score."""
    from .matching import match_many

    keys = sorted(R)
    ms = match_many([R[k] for k in keys], top_k=top_k, workers=workers)
    return {k: {"perm": m.perm.astype(np.int64), "score": m.score.astype(np.float64),
                "self_score": np.diag(R[k]).astype(np.float64)} for k, m in zip(keys, ms)}


# ----------------------------------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------------------------------
def _pass_path(out: Path, element: str, fit_seed: int) -> Path:
    return Path(out) / "cache" / f"pass_{element}_fit{fit_seed}.npz"


def _inputs_path(out: Path, fit_seed: int) -> Path:
    return Path(out) / "cache" / f"inputs_fit{fit_seed}.npz"


def save_pass(path: Path, meta: dict, space: dict, match: dict) -> None:
    arrays, sk, hk = {}, sorted(space), sorted(match)
    for i, k in enumerate(sk):
        for f in ("C", "sq_x", "sq_xt", "n", "s_x"):
            arrays[f"s{i}_{f}"] = np.asarray(space[k][f])
    for i, k in enumerate(hk):
        for f in ("perm", "score", "self_score"):
            arrays[f"h{i}_{f}"] = np.asarray(match[k][f])
    meta = {**meta, "spaces": sk, "hidden": hk}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, meta_json=np.asarray(json.dumps(meta)), **arrays)


def load_pass(path: Path) -> dict:
    with np.load(path) as z:
        meta = json.loads(str(z["meta_json"]))
        space = {k: {f: (z[f"s{i}_{f}"] if f in ("C", "s_x") else float(z[f"s{i}_{f}"]))
                     for f in ("C", "sq_x", "sq_xt", "n", "s_x")} for i, k in enumerate(meta["spaces"])}
        match = {k: {f: z[f"h{i}_{f}"] for f in ("perm", "score", "self_score")} for i, k in enumerate(meta["hidden"])}
    return {"meta": meta, "space": space, "match": match}


def _cache_ok(path: Path, want: dict) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as z:
            meta = json.loads(str(z["meta_json"]))
    except Exception:
        return False
    return all(meta.get(k) == v for k, v in want.items())


# ----------------------------------------------------------------------------------------------
# Projection, shares, free parameters
# ----------------------------------------------------------------------------------------------
def _proj_in(W: torch.Tensor, b: torch.Tensor, rho, perm):
    act = GR.block_action("LR", [rho], [None], [perm])
    e = GR.project_mlp_group(W, b, W.new_zeros(1, W.shape[0]), W.new_zeros(1), act)
    return e["W1"], e["b1"]


def _proj_out(W2: torch.Tensor, b2: torch.Tensor, rho, perm):
    act = GR.block_action("LR", [None], [rho], [perm])
    e = GR.project_mlp_group(W2.new_zeros(W2.shape[1], 1), W2.new_zeros(W2.shape[1]), W2, b2, act)
    return e["W2"], e["b2"]


def project_unit(f: dict, rho_in, rho_out, perm_in, perm_out=None) -> dict:
    """Order-2 group projection of a folded unit (mirror.groups.project_mlp_group on each matrix): every input
    matrix with (rho_in, sigma_1), the output matrix with (rho_out, sigma_2); sigma_2 = sigma_1 when not given."""
    perm_out = perm_in if perm_out is None else perm_out
    W1, b1 = zip(*[_proj_in(W, b, rho_in, perm_in) for W, b in zip(f["W1"], f["b1"])])
    W2, b2 = _proj_out(f["W2"], f["b2"], rho_out, perm_out)
    return {"W1": list(W1), "b1": list(b1), "W2": W2, "b2": b2}


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else float("nan")


def unit_shares(f: dict, e: dict, M1=None, M2=None) -> dict:
    """Breaking shares ||W - W_eq||^2 / ||W||^2 (input matrices pooled) and, with the input second moments,
    the activation-weighted shares ||(W - W_eq) M^(1/2)||^2 / ||W M^(1/2)||^2."""
    def sq(a):
        return float((a * a).sum())

    out = {"s_fc1": _ratio(sum(sq(W - We) for W, We in zip(f["W1"], e["W1"])), sum(sq(W) for W in f["W1"])),
           "s_fc2": _ratio(sq(f["W2"] - e["W2"]), sq(f["W2"])),
           "s_b1": _ratio(sum(sq(b - be) for b, be in zip(f["b1"], e["b1"])), sum(sq(b) for b in f["b1"])),
           "s_b2": _ratio(sq(f["b2"] - e["b2"]), sq(f["b2"]))}
    out["s_mlp"] = 0.5 * (out["s_fc1"] + out["s_fc2"])
    if M1 is not None:
        num = den = 0.0
        for W, We in zip(f["W1"], e["W1"]):
            D = W - We
            num += float(((D @ M1) * D).sum())
            den += float(((W @ M1) * W).sum())
        out["s_fc1_act"] = _ratio(num, den)
    if M2 is not None:
        D = f["W2"] - e["W2"]
        out["s_fc2_act"] = _ratio(float(((D @ M2) * D).sum()), float(((f["W2"] @ M2) * f["W2"]).sum()))
    return out


def _fix(perm) -> int:
    perm = np.asarray(perm)
    return int((perm == np.arange(len(perm))).sum())


def unit_free(f: dict, rho_in, rho_out, perm_in, perm_out=None) -> dict:
    """Character-formula free parameters (order 2): per input matrix (h1 d_in + tr(rho_in) fix(sigma_1)) / 2, for
    the output matrix (d_out h2 + tr(rho_out) fix(sigma_2)) / 2; biases: hidden orbits and (d_out + tr) / 2."""
    perm_out = perm_in if perm_out is None else perm_out
    h1, d_in = f["W1"][0].shape
    d_out, h2 = f["W2"].shape
    n_in = len(f["W1"])
    t_in, t_out = float(np.trace(np.asarray(rho_in))), float(np.trace(np.asarray(rho_out)))
    f1 = n_in * GR.character_free([d_in, t_in], [h1, _fix(perm_in)])
    f2 = GR.character_free([d_out, t_out], [h2, _fix(perm_out)])
    dense1, dense2 = n_in * h1 * d_in, d_out * h2
    return {"free_fc1": f1, "free_fc2": f2, "dense_fc1": dense1, "dense_fc2": dense2,
            "free_b1": n_in * GR.character_free([1, 1], [h1, _fix(perm_in)]),
            "free_b2": GR.character_free([d_out, t_out], [1, 1]),
            "tax_fc1": 1.0 - f1 / dense1, "tax_fc2": 1.0 - f2 / dense2,
            "tax_mlp": 1.0 - (f1 + f2) / (dense1 + dense2)}


# ----------------------------------------------------------------------------------------------
# Analysis of one group
# ----------------------------------------------------------------------------------------------
def random_c2_fixing(d: int, d_plus: int, v, seed: int) -> np.ndarray:
    """Random involution with d_plus eigenvalues +1 whose +1 eigenspace contains the vector v (the construction of
    mirror.groups.random_c2 with v in place of the all-ones vector)."""
    if d_plus < 1:
        raise ValueError("fixing a vector needs d_plus >= 1")
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(np.column_stack([np.asarray(v, dtype=np.float64), rng.standard_normal((d, d - 1))]))
    s = np.concatenate([np.ones(int(d_plus)), -np.ones(d - int(d_plus))])
    A = (Q * s) @ Q.T
    return 0.5 * (A + A.T)


def null_operator(sp: dict, d: int, d_plus: int, centred: bool, seed: int) -> np.ndarray:
    """Null operator of a space: same multiplicities as the fit; the invariant direction kept fixed, as the fit keeps
    it: the all-ones vector for channel-centred spaces (as measure_groups.py), the mean activation for uncentred ones
    (a flip-symmetric data distribution leaves the per-channel mean unchanged, so a null that rotated it would
    count trivially symmetric structure as a fit)."""
    if centred:
        return GR.random_c2(d, d_plus, seed, fix_ones=d_plus >= 1)
    mu = np.asarray(sp.get("s_x"), dtype=np.float64) if sp.get("s_x") is not None else None
    if d_plus >= 1 and mu is not None and np.linalg.norm(mu) > 0:
        return random_c2_fixing(d, d_plus, mu / np.linalg.norm(mu), seed)
    return GR.random_c2(d, d_plus, seed, fix_ones=False)


def _err(sp: dict, A) -> tuple:
    e = GR.moment_error(sp["C"], sp["sq_x"], sp["sq_xt"], A)
    eI = sp["sq_x"] + sp["sq_xt"] - 2.0 * float(np.trace(np.asarray(sp["C"], dtype=np.float64)))
    return e, eI


def analyse_group(spec: MeasureSpec, P: dict, inputs: dict, group: str, folded: dict, n_draws: int, seed: int,
                  device, extra: dict) -> tuple:
    """Space rows, unit rows and per-draw rows for one involution group."""
    if group not in ENGINE_GROUPS:
        raise ValueError(f"group {group!r}: the engine handles {ENGINE_GROUPS}")
    spaces = sorted({u.in_space for u in spec.units} | {u.out_space for u in spec.units},
                    key=lambda s: spec.slots[s])
    hiddens = sorted({u.hid_in for u in spec.units} | {u.hid_out for u in spec.units}, key=lambda s: spec.slots[s])
    ops = {}
    for s in spaces:
        A, dp, dm = GR.fit_order2(P["space"][s]["C"], center=spec.centre_of(s))
        ops[s] = {"A": A, "d_plus": int(dp), "d_minus": int(dm), "d": int(A.shape[0])}
    match = P["match"]
    dev = torch.device(device)

    def t64(M):
        return None if M is None else torch.as_tensor(np.asarray(M), dtype=torch.float64, device=dev)

    Ms = {k: t64(v["M"]) for k, v in inputs.items()}
    fold_t = {u.index: {k: ([x.to(dev) for x in v] if isinstance(v, list) else (v.to(dev) if torch.is_tensor(v) else v))
                        for k, v in folded[u.index].items()} for u in spec.units}
    e_null = {s: [] for s in spaces}
    unit_fit, draw_rows = {}, []
    for draw in range(-1, int(n_draws)):
        if draw < 0:
            ops_k = {s: ops[s]["A"] for s in spaces}
            perms_k = {h: match[h]["perm"] for h in hiddens}
        else:
            ops_k = {s: null_operator(P["space"][s], ops[s]["d"], ops[s]["d_plus"], spec.centre_of(s),
                                      seed_for(seed, group, draw, spec.slots[s])) for s in spaces}
            perms_k = {h: GR.random_permutation_like(match[h]["perm"],
                                                     np.random.default_rng(seed_for(seed, group, draw, spec.slots[h])))
                       for h in hiddens}
            for s in spaces:
                e_null[s].append(_err(P["space"][s], ops_k[s])[0])
        for u in spec.units:
            f = fold_t[u.index]
            e = project_unit(f, ops_k[u.in_space], ops_k[u.out_space], perms_k[u.hid_in], perms_k[u.hid_out])
            sh = unit_shares(f, e, Ms.get(u.m1), Ms.get(u.m2))
            if draw < 0:
                unit_fit[u.index] = sh
            else:
                draw_rows.append({"group": group, "block": u.index + 1, "draw": draw, **sh})
    # ---- space rows
    space_rows = []
    for s in spaces:
        sp = P["space"][s]
        e, eI = _err(sp, ops[s]["A"])
        row = {"group": group, "space": s, "layer": spec.slots[s], "d": ops[s]["d"], "d_plus": ops[s]["d_plus"],
               "d_minus": ops[s]["d_minus"], "centred": spec.centre_of(s),
               "e_rel": _ratio(e, sp["sq_xt"]), "e_ratio_I": _ratio(e, eI), "n_tokens": sp["n"]}
        if e_null[s]:
            en = float(np.mean(e_null[s]))
            row.update({"e_null_ratio_I": _ratio(en, eI), "null_over_fit": en / e if e > 0 else float("inf")})
        space_rows.append({**row, **extra})
    # ---- unit rows
    unit_rows = []
    for u in spec.units:
        fit = unit_fit[u.index]
        f = folded[u.index]
        h1, d_in = f["W1"][0].shape
        d_out, h2 = f["W2"].shape
        fr = GR.orbit_fractions([match[u.hid_in]["perm"]], [match[u.hid_in]["score"]])
        row = {"group": group, "block": u.index + 1, "name": u.name, "stage": u.stage + 1, "kind": u.kind,
               "in_space": u.in_space, "out_space": u.out_space, "h": int(h1), "d": int(d_in), "h_out": int(h2),
               "d_out": int(d_out), "n_input_matrices": len(f["W1"])}
        if u.hid_out != u.hid_in:
            fr2 = GR.orbit_fractions([match[u.hid_out]["perm"]], [match[u.hid_out]["score"]])
            row.update({f"{k}_h1": v for k, v in fr.items()})
            row.update({f"{k}_h2": v for k, v in fr2.items()})
            fr = {k: (0.5 * (v + fr2[k]) if isinstance(v, float) else v) for k, v in fr.items()}
        row.update(fr)
        perm = match[u.hid_in]["perm"]
        row["n_pairs"] = int((perm != np.arange(len(perm))).sum() // 2)
        row["n_self"] = _fix(perm)
        for key in SHARE_KEYS:
            if key not in fit:
                continue
            row[key] = fit[key]
            vals = [r[key] for r in draw_rows if r["block"] == u.index + 1 and key in r]
            ns = GR.null_summary(vals)
            row[f"{key}_null_mean"], row[f"{key}_null_lo"], row[f"{key}_null_hi"] = ns["mean"], ns["lo"], ns["hi"]
            row[f"{key}_over_null"] = fit[key] / ns["mean"] if ns["n"] and ns["mean"] > 0 else float("nan")
        row["null_draws"] = int(n_draws)
        row.update(unit_free(f, ops[u.in_space]["A"], ops[u.out_space]["A"], match[u.hid_in]["perm"],
                             match[u.hid_out]["perm"]))
        unit_rows.append({**row, **extra})
    return space_rows, unit_rows, draw_rows


# ----------------------------------------------------------------------------------------------
# Summaries (scripts/measure_groups.py build_metrics definitions)
# ----------------------------------------------------------------------------------------------
def _nanmean(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(x)) if x.size and np.isfinite(x).any() else float("nan")


def default_ranges(spec: MeasureSpec, vit_ranges: str = "1-6,7-12") -> dict:
    """Range name -> unit indices (0-based). ViT: the measure_groups.py ranges plus both halves; CNN: both halves,
    each stage and the first half of the stages."""
    n = len(spec.units)
    out = {}
    if spec.family == "vit":
        for part in str(vit_ranges).split(","):
            part = part.strip()
            if not part:
                continue
            a, b = (part.split("-", 1) + [part])[:2]
            sel = [u.index for u in spec.units if int(a) <= u.index + 1 <= int(b)]
            if sel:
                out[part] = sel
    else:
        stages = sorted({u.stage for u in spec.units})
        for s in stages:
            out[f"stage{s + 1}"] = [u.index for u in spec.units if u.stage == s]
        half_st = stages[:len(stages) // 2]
        out[f"stages_1-{len(half_st)}"] = [u.index for u in spec.units if u.stage in half_st]
    out["first_half"] = [u.index for u in spec.units[:n // 2]]
    out["second_half"] = [u.index for u in spec.units[n // 2:]]
    return {k: v for k, v in out.items() if v}


def range_summary(blk, lay, drw, units: list, spec: MeasureSpec) -> dict:
    """One range of one group: means, null means and intervals, ratio to null, fractions, tax, operator fit."""
    rb1 = [i + 1 for i in units]
    sub = blk[blk["block"].isin(rb1)]
    r = {"blocks": rb1, "n_blocks": int(len(sub))}
    if not len(sub):
        return r
    for key in ("s_fc1", "s_fc2", "s_mlp", "s_fc1_act", "s_fc2_act"):
        if key not in sub:
            continue
        r[f"{key}_mean"] = _nanmean(sub[key])
        if f"{key}_null_mean" in sub:
            nm = _nanmean(sub[f"{key}_null_mean"])
            r[f"{key}_null_mean"] = nm
            r[f"{key}_ratio_to_null"] = r[f"{key}_mean"] / nm if nm and np.isfinite(nm) and nm > 0 else float("nan")
            if len(drw) and key in drw:
                per = drw[drw["block"].isin(rb1)].groupby("draw")[key].mean()
                ns = GR.null_summary(per.values)
                r[f"{key}_null_lo"], r[f"{key}_null_hi"], r[f"{key}_null_n"] = ns["lo"], ns["hi"], ns["n"]
    r["ratio_to_null"] = r.get("s_mlp_ratio_to_null", float("nan"))
    if "s_fc1_act_null_mean" in sub and "s_fc2_act_null_mean" in sub:
        act = 0.5 * (r.get("s_fc1_act_mean", np.nan) + r.get("s_fc2_act_mean", np.nan))
        act_null = 0.5 * (_nanmean(sub["s_fc1_act_null_mean"]) + _nanmean(sub["s_fc2_act_null_mean"]))
        r["ratio_to_null_act"] = act / act_null
    tags = [""] + [f"_t{int(round(100 * t))}" for t in GR.THRESHOLDS]
    for t in tags:
        for kind_ in ("2", "4", "fixed", "nontrivial"):
            col = f"frac_{kind_}{t}"
            if col in sub:
                r[col] = _nanmean(sub[col])
    r["twin_fraction_t50"] = r.get("frac_2_t50")
    fsum = float((sub["free_fc1"] + sub["free_fc2"]).sum())
    dsum = float((sub["dense_fc1"] + sub["dense_fc2"]).sum())
    r["free_mlp"], r["dense_mlp"] = fsum, dsum
    r["tax_mlp"] = 1.0 - fsum / dsum
    spaces = sorted({spec.units[i].in_space for i in units} | {spec.units[i].out_space for i in units})
    ls = lay[lay["space"].isin(spaces)] if len(lay) else lay
    for key in ("e_ratio_I", "null_over_fit"):
        if len(ls) and key in ls:
            r[f"{key}_mean"] = _nanmean(ls[key])
    return r


def build_metrics(out: Path, spec: MeasureSpec, groups: list, ranges: dict, extra: dict) -> dict:
    import pandas as pd

    def read(name):
        p = Path(out) / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    blk, lay, drw = read("blocks.csv"), read("layers.csv"), read("null_draws.csv")
    res = {"model": spec.name, "engine": "mirror/cnn_groups.py", "family": spec.family, "groups": {}, "scores": {},
           "ranges": {k: [i + 1 for i in v] for k, v in ranges.items()},
           "units": [{"block": u.index + 1, "name": u.name, "stage": u.stage + 1, "kind": u.kind,
                      "in_space": u.in_space, "out_space": u.out_space, "hid_in": u.hid_in, "hid_out": u.hid_out}
                     for u in spec.units],
           "definitions": {
               "ratio_to_null": "mean over the range's units of (s_fc1 + s_fc2) / 2, divided by the mean over the same "
                                "units of the null means of (s_fc1 + s_fc2) / 2 (PREREG_NEXT_PHASE.md Section 1)",
               "twin_fraction_t50": "mean over units of the share of hidden units in a 2-cycle of the matched "
                                    "permutation whose twin correlation is at least 0.5 (bottleneck: mean of its two "
                                    "hidden layers)",
               "tax_mlp": "1 - sum of character-formula free parameters / sum of dense parameters over the range's "
                          "converted matrices (SwiGLU: both input matrices counted)",
               "blocks": "1-based units in depth order (block 1 = code index 0)"}}
    for g in groups:
        gb = blk[blk["group"] == g] if len(blk) else blk
        if not len(gb):
            continue
        gl = lay[lay["group"] == g] if len(lay) else lay
        gd = drw[drw["group"] == g] if len(drw) else drw
        G = {"kind": GR.kind(g), "ranges": {name: range_summary(gb, gl, gd, idx, spec) for name, idx in ranges.items()}}
        res["groups"][g] = G
        sc = {}
        for name, r in G["ranges"].items():
            for key in ("ratio_to_null", "twin_fraction_t50", "tax_mlp"):
                sc[f"{key}_{name}"] = r.get(key)
        res["scores"][g] = sc
    res.update(extra)
    return res


# ----------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------
class _Csv:
    """Resumable CSV keyed by columns (variants.ResultsCsv with a bulk add)."""

    def __init__(self, path, key):
        from .variants import ResultsCsv

        self.inner = ResultsCsv(path, key)

    def done(self, key) -> bool:
        return self.inner.done(key)

    def add_many(self, rows) -> int:
        import pandas as pd

        inner = self.inner
        new = [r for r in rows if not inner.done(tuple(r[c] for c in inner.key))]
        for r in new:
            inner.rows.append(r)
            inner._done.add(inner._k(r))
        if new:
            inner.path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(inner.rows).to_csv(inner.path, index=False)
        return len(new)


def run_measurement(spec: MeasureSpec, fit_loader: Callable, device, out, groups=("LR", "UD"), fit_seed: int = 0,
                    null_draws: int = 20, seed: int = 0, top_k: int = 10, workers: int = 1, git: dict | None = None,
                    ranges: dict | None = None, log=print) -> dict:
    """Passes (cached under <out>/cache), analyses (resumable CSVs) and metrics.json for every group."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    git = git or {"git_hash": "unknown", "git_dirty": None}
    groups = [g for g in groups]
    bad = [g for g in groups if g not in ENGINE_GROUPS]
    if bad:
        raise ValueError(f"groups {bad} are not involutions; the engine handles {ENGINE_GROUPS}")
    timings = {}
    unit_names = [u.name for u in spec.units]
    want = {"model": spec.name, "fit_seed": int(fit_seed), "top_k": int(top_k), "units": unit_names}
    inputs_path = _inputs_path(out, fit_seed)
    inputs_ok = _cache_ok(inputs_path, {"model": spec.name, "fit_seed": int(fit_seed), "units": unit_names})
    for g in groups:
        path = _pass_path(out, g, fit_seed)
        if _cache_ok(path, {**want, "element": g}) and inputs_ok:
            log(f"pass {g}: cached ({path})")
            continue
        t0 = time.time()
        r = collect_pass(spec, fit_loader(), device, g, inputs=not inputs_ok)
        t_pass = time.time() - t0
        t1 = time.time()
        match = match_hidden(r["R"], top_k, workers)
        t_match = time.time() - t1
        save_pass(path, {**want, "element": g, "n_images": r["n_images"], "seconds_pass": round(t_pass, 1),
                         "seconds_match": round(t_match, 1)}, r["space"], match)
        if not inputs_ok:
            keys = sorted(r["inputs"])
            np.savez(inputs_path, meta_json=np.asarray(json.dumps({"model": spec.name, "fit_seed": int(fit_seed),
                                                                   "units": unit_names, "keys": keys})),
                     **{f"m{i}": r["inputs"][k]["M"] for i, k in enumerate(keys)},
                     **{f"n{i}": np.asarray(r["inputs"][k]["n"]) for i, k in enumerate(keys)})
            inputs_ok = True
        timings[f"pass_{g}"], timings[f"match_{g}"] = round(t_pass, 1), round(t_match, 1)
        log(f"pass {g}: {r['n_images']} images, pass {t_pass:.1f} s, matching {t_match:.1f} s")
        del r, match
    with np.load(inputs_path) as z:
        meta = json.loads(str(z["meta_json"]))
        inputs = {k: {"M": z[f"m{i}"], "n": int(z[f"n{i}"])} for i, k in enumerate(meta["keys"])}
    folded = {u.index: u.fold() for u in spec.units}
    blk_csv, lay_csv, drw_csv = (_Csv(out / "blocks.csv", UNIT_KEY), _Csv(out / "layers.csv", SPACE_KEY),
                                 _Csv(out / "null_draws.csv", DRAW_KEY))
    extra = {"fit_seed": int(fit_seed), "null_seed_base": int(seed), "git_hash": git.get("git_hash")}
    for g in groups:
        if all(blk_csv.done((g, u.index + 1)) for u in spec.units):
            log(f"{g}: rows present, skipped")
            continue
        t0 = time.time()
        P = load_pass(_pass_path(out, g, fit_seed))
        srows, urows, drows = analyse_group(spec, P, inputs, g, folded, null_draws, seed, device, extra)
        lay_csv.add_many(srows)
        drw_csv.add_many(drows)
        blk_csv.add_many(urows)
        timings[f"analyse_{g}"] = round(time.time() - t0, 1)
        log(f"{g}: {len(spec.units)} units, {null_draws} null draws, {time.time() - t0:.1f} s")
    ranges = ranges or default_ranges(spec)
    metrics = build_metrics(out, spec, groups, ranges,
                            {"info": spec.info, "depth": int(spec.info.get("depth", len(spec.units))),
                             "timings_s": timings, "fit_seed": int(fit_seed), "git": git,
                             "n_fit_images": load_pass(_pass_path(out, groups[0], fit_seed))["meta"]["n_images"]})
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return metrics


def tiny_check_flip(spec: MeasureSpec, x: torch.Tensor, element: str = "LR") -> dict:
    """Largest |aligned g(x) tensor - x tensor| per space capture, for a network that should be exactly invariant
    (tests): a diagnostic of the alignment conventions."""
    out = {}

    def hook(cap):
        def fn(t):
            if isinstance(t, (tuple, list)):
                t = t[0]
            b = t.shape[0] // 2
            a = align(t[b:], cap.layout, element, None if spec.grid is None else
                      torch.as_tensor(GR.token_index(element, spec.grid)), spec.n_prefix)
            out[cap.key] = float((a - t[:b]).abs().max() / t[:b].abs().max().clamp_min(1e-30))
        return fn

    hs = []
    for cap in spec.captures:
        if cap.role != "space":
            continue
        fn = hook(cap)
        if cap.where == "pre":
            hs.append(cap.module.register_forward_pre_hook(lambda _m, args, fn=fn: fn(args[0])))
        else:
            hs.append(cap.module.register_forward_hook(lambda _m, _a, o, fn=fn: fn(o)))
    try:
        with torch.no_grad():
            try:
                spec.forward(torch.cat([x, GR.image_action(x, element)], dim=0))
            except _Stop:
                pass
    finally:
        for h in hs:
            h.remove()
    return out


__all__ = ["Capture", "Unit", "MeasureSpec", "vit_spec", "convnext_spec", "resnet_spec", "collect_pass",
           "match_hidden", "project_unit", "unit_shares", "unit_free", "analyse_group", "default_ranges",
           "build_metrics", "run_measurement", "fold_convnext_block", "fold_bottleneck", "seed_for", "align",
           "ENGINE_GROUPS", "tiny_check_flip"]
