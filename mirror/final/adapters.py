"""Model adapters for the final phase: DeiT-III-B/16 (A08), DINOv2-L/14 with registers (A08) and SigLIP 2 B/16
(A15a), plus the activation-weighted ("whitened") projection of A15a.

Sources read for this module (no weights downloaded): timm 1.0.31 as installed in .venv
(timm/models/deit.py: deit3_base_patch16_224 = VisionTransformer(patch 16, 768, depth 12, heads 12,
no_embed_class=True, init_values=1e-6), pretrained cfg 'deit3_base_patch16_224.fb_in1k' with num_classes 1000;
timm/models/vision_transformer.py: VisionTransformer.__init__ (fc_norm is Identity unless global_pool is avg, avgmax
or max; self.norm is the final LayerNorm otherwise; attn_pool = AttentionPoolLatent for global_pool 'map'),
_pos_embed (no_embed_class: positions added to the patch tokens only, then CLS and registers concatenated),
pool / global_pool_nlc ('token' takes x[:, 0]), forward_head (pool, fc_norm, head_drop, head);
vit_base_patch16_siglip_224 = VisionTransformer(class_token=False, global_pool='map'), pretrained cfg
'vit_base_patch16_siglip_224.v2_webli' with num_classes 0; timm/layers/attention_pool.py AttentionPoolLatent),
configs/models/deit3_b16.yaml, siglip2_b16.yaml and dinov2_vitl14_reg.yaml (provenance records), the CORE-04 results
(results/CORE-04/surgery/rows_siglip2_b16.csv and info/siglip2_b16/*/mirror_exact_s0.json) and
notes/paper/method_spec.md Sections 1 to 7.

DeiT-III (A08). The classifier path in timm is norm (final LayerNorm) -> pool 'token' (class token, x[:, 0]) ->
fc_norm (Identity for global_pool 'token') -> head_drop (Dropout, identity in eval) -> head (Linear 768 -> 1000).
After full LayerNorm folding (mirror.fold, final affine kept) the final norm is Sequential(rms, affine);
install_even_classifier puts the even projection y (I + A*)/2 between the two (mirror.m1.install_even_head), so the
head reads affine(rms(t_cls) (I + A*)/2): the even part of the class token, exactly mirror-invariant, hence identical
logits for an image and its mirror. deit_head_path checks this path on the model itself. DeiT-III has no registers,
no position row for the class token (no_embed_class=True) and 1 + 196 = 197 tokens at 224 px; the existing structure,
conversion and budget code handles that (tests/test_fp_adapters.py runs the whole chain on a tiny DeiT-III).

DINOv2-L/14 reg (A08). 24 blocks, d = 1024, 16 heads, 4 registers (configs/models/dinov2_vitl14_reg.yaml). The code
paths are size-generic; the tests run a tiny analogue (timm VisionTransformer with reg_tokens=4, registers paired)
and the DINOv2 hub layout (mock with a position grid stored at another size, so materialise_pos_embed interpolates).
check_against_config compares a built adapter with the provenance YAML on the VM.

SigLIP 2 (A15a). No class token and an attention-pooling head (MAP: one learned latent query attends over all patch
tokens, then a projection and an MLP). The earlier runs fed k-NN the pooled embedding: TimmPretrainedAdapter with
feature 'pooled' returns model.forward_head(model.norm(tokens), pre_logits=True) (MODEL_REGISTRY['siglip2_b16'];
scripts/must07_compressors.Evaluator calls adapter.cls_feature). The collapse being explained (CORE-04, fit seed 0):
Mirror-exact zero-shot on blocks 1 to 6 with the span operator A*_6 (d+ 676, fitted over stream layers 0 to 6 of
fit set 0), embeddings projected, no even head, k-NN on the pooled embedding (100k bank, 50k validation, bf16
autocast): 7.84 (MLP only, "6mlp") and 13.45 (MLP and attention, "6both") against 75.54 for the original
(results/CORE-04/surgery/rows_siglip2_b16.csv).

A15a projection. The plain projection replaces a weight W (out x in) with the equivariant matrix closest in the
Frobenius norm, X = (W + R_out W R_in)/2, for the output and input involutions of that weight (fc1: R_out = P, R_in =
A*; fc2: A*, P; q and k: the head involution built from sigma and B, A*; v: from C, A*; o: A*, the C involution; patch
embedding: A*, the width flip of the kernel). The weighted projection takes the equivariant X closest in
||(W - X) M^(1/2)||_F, M the second moment of the weight's input on the fit images: the error on the actual inputs.
Derivation: in orthonormal eigenbases E_in = [E+, E-] of R_in and F = [F+, F-] of R_out, equivariance
X R_in = R_out X means F^T X E_in is block diagonal (even outputs read even inputs only, odd outputs odd inputs
only). The objective tr((W - X) M (W - X)^T) separates over output rows; for a row r = [r+, r-] (input coordinates)
of the even output group the equivariant row is [x+, 0] and
    f(x+) = (r+ - x+) N++ (r+ - x+)^T + 2 (r+ - x+) N+- r-^T + r- N-- r-^T,   N = E_in^T M E_in,
whose gradient vanishes at x+ = r+ + r- N-+ N++^(-1) (N symmetric); odd rows give x- = r- + r+ N+- N--^(-1).
In the original coordinates X = P_out+ W G+ + P_out- W G-, with P_out(+/-) = (I +/- R_out)/2,
G+ = [E+ + E- N-+ N++^(-1)] E+^T and G- = [E- + E+ N+- N--^(-1)] E-^T; since G+ R_in = G+ and G- R_in = -G-,
X R_in = R_out X exactly. If M commutes with R_in (inputs as likely mirrored as not, N+- = 0) X is the plain
projection. Biases are projected jointly: the input is augmented with a constant 1 (even under R_in), so the bias of
an even output absorbs the mean of the dropped odd part. N++ and N-- get the damping of SVD-LLM in this repository
(damp x mean of the diagonal; mirror/baselines.py). Position embeddings, class and register tokens are additive (a
constant input), so for them the plain projection already minimises the error on the tokens and is kept. Operator,
pairings and head maps are the fitted ones, unchanged: the whitened projection changes only how each weight is
projected, so the result is exactly equivariant like the plain one. (The plan's phrase "the basis whitened by its
activation covariance" could also mean refitting A* in a whitened stream basis; that is not what this tests (verify
the intended reading with Dawar).)

Public API: TimmModelAdapter, tiny_deit3, tiny_siglip2, tiny_dinov2_reg, model_layout, check_against_config,
deit_head_path, install_even_classifier, deit_logits, flip_averaged_logits, top1_eval, EvenOddClassifier,
siglip_feature_path, Involution, head_involution, patch_flip_perm, weighted_projection, weighted_error,
collect_projection_moments, build_whitened_vexact, CORE04_SIGLIP_ROWS.
"""
from __future__ import annotations

import contextlib
import copy
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..adapters import _randomise_affine
from ..adapters_timm import FEATURES, TimmPretrainedAdapter, check_layout
from ..attention import HeadMaps, apply_attention_surgery, attn_dims, fold_attn, gauge_twin_heads, project_attn, \
    write_absorbed_attn
from ..data import mirror, normalize_for
from ..efficient import eig_basis
from ..fold import fold_copy, fold_layernorm_full
from ..heal import exact_involution
from ..m1 import install_even_head, materialise_pos_embed, project_embeddings, register_info
from ..surgery import fold_mlp, project_mlp_two, write_absorbed_mlp
from ..utils import REPO_ROOT

__all__ = [
    "TimmModelAdapter", "tiny_deit3", "tiny_siglip2", "tiny_dinov2_reg", "model_layout", "check_against_config",
    "deit_head_path", "install_even_classifier", "deit_logits", "flip_averaged_logits", "top1_eval",
    "EvenOddClassifier", "siglip_feature_path", "Involution", "head_involution", "patch_flip_perm",
    "weighted_projection", "weighted_error", "collect_projection_moments", "build_whitened_vexact",
    "CORE04_SIGLIP_ROWS", "DEFAULT_DAMP",
]

DEFAULT_DAMP = 1e-6       # as mirror/baselines.py DEFAULT_DAMP for SVD-LLM (damp x mean diag)
CORE04_SIGLIP_ROWS = "results/CORE-04/surgery/rows_siglip2_b16.csv"


# ----------------------------------------------------------------------------------------------
# Adapters from model instances, tiny analogues
# ----------------------------------------------------------------------------------------------
class TimmModelAdapter(TimmPretrainedAdapter):
    """TimmPretrainedAdapter around an existing timm VisionTransformer instance (tiny test models, or a model
    built elsewhere): the same layout checks, feature choice and forward pieces."""

    def __init__(self, model: nn.Module, img_size: int, feature: str | None = None, model_name: str = "",
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        patch = check_layout(model, int(img_size))
        has_cls = getattr(model, "cls_token", None) is not None
        feature = feature or ("cls" if has_cls else "pooled")
        if feature not in FEATURES or (feature == "cls" and not has_cls):
            raise ValueError(f"feature {feature!r} does not fit this model (class token: {has_cls})")
        from ..adapters import ViTAdapter

        ViTAdapter.__init__(self, model, img_size=img_size, patch_size=patch, n_prefix_tokens=int(model.num_prefix_tokens))
        self.model_name, self.feature, self.has_cls_token = model_name, feature, has_cls
        self.has_layerscale = any(not isinstance(b.ls1, nn.Identity) or not isinstance(b.ls2, nn.Identity)
                                  for b in model.blocks)
        self.mean, self.std = tuple(float(v) for v in mean), tuple(float(v) for v in std)
        self.interpolation, self.crop_pct = "bicubic", 0.9


def _tiny(name: str, img_size: int, seed: int, dtype, feature, randomise: bool, mean, std, **kw):
    import timm

    torch.manual_seed(seed)
    m = timm.create_model(name, pretrained=False, img_size=img_size, **kw)
    if randomise:
        _randomise_affine(m, seed)
        with torch.no_grad():
            g = torch.Generator().manual_seed(seed + 7)
            for p in (getattr(m, "cls_token", None), getattr(m, "reg_token", None), getattr(m, "pos_embed", None)):
                if isinstance(p, torch.Tensor):
                    p.copy_(0.5 * torch.randn(p.shape, generator=g))
    return TimmModelAdapter(m.to(dtype).eval(), img_size, feature=feature, model_name=f"tiny:{name}", mean=mean, std=std)


def tiny_deit3(img_size: int = 56, patch: int = 14, dim: int = 64, depth: int = 4, heads: int = 4,
               num_classes: int = 10, seed: int = 0, dtype=torch.float64, randomise: bool = True) -> TimmModelAdapter:
    """DeiT-III built by timm's own deit3_base_patch16_224 entry (no_embed_class=True, LayerScale) with tiny sizes
    and random weights (pretrained=False: nothing is downloaded)."""
    return _tiny("deit3_base_patch16_224", img_size, seed, dtype, "cls", randomise, (0.485, 0.456, 0.406),
                 (0.229, 0.224, 0.225), patch_size=patch, embed_dim=dim, depth=depth, num_heads=heads,
                 num_classes=num_classes)


def tiny_siglip2(img_size: int = 56, patch: int = 14, dim: int = 64, depth: int = 4, heads: int = 4, seed: int = 0,
                 dtype=torch.float64, randomise: bool = True) -> TimmModelAdapter:
    """SigLIP 2 B/16's timm entry (vit_base_patch16_siglip_224: no class token, MAP head) with tiny sizes."""
    return _tiny("vit_base_patch16_siglip_224", img_size, seed, dtype, "pooled", randomise, (0.5, 0.5, 0.5),
                 (0.5, 0.5, 0.5), patch_size=patch, embed_dim=dim, depth=depth, num_heads=heads, num_classes=0)


def tiny_dinov2_reg(img_size: int = 56, patch: int = 14, dim: int = 64, depth: int = 4, heads: int = 4,
                    reg_tokens: int = 4, seed: int = 0, dtype=torch.float64, randomise: bool = True) -> TimmModelAdapter:
    """The tiny analogue of DINOv2-L/14 reg: a timm VisionTransformer with a class token, 4 register tokens and
    LayerScale (positions for every token; the DINOv2 hub layout is covered by the mock in tests/test_fold.py)."""
    from timm.models.vision_transformer import VisionTransformer

    torch.manual_seed(seed)
    m = VisionTransformer(img_size=img_size, patch_size=patch, embed_dim=dim, depth=depth, num_heads=heads,
                          init_values=1e-5, reg_tokens=reg_tokens, num_classes=0)
    if randomise:
        _randomise_affine(m, seed)
        with torch.no_grad():
            g = torch.Generator().manual_seed(seed + 7)
            for p in (m.cls_token, m.reg_token, m.pos_embed):
                p.copy_(0.5 * torch.randn(p.shape, generator=g))
    return TimmModelAdapter(m.to(dtype).eval(), img_size, feature="cls", model_name="tiny:dinov2_reg")


# ----------------------------------------------------------------------------------------------
# Layout facts and the provenance records
# ----------------------------------------------------------------------------------------------
def model_layout(adapter) -> dict:
    """Facts the conversion relies on, read from the model itself."""
    m = adapter.model
    blk = m.blocks[0]
    H, dh = attn_dims(blk)
    reg = register_info(m, adapter)
    pos = getattr(m, "pos_embed", None)
    n_pos = int(pos.shape[1]) if isinstance(pos, torch.Tensor) else 0
    Hg, Wg = adapter.grid_hw
    ls = any(isinstance(getattr(getattr(b, n, None), "gamma", None), torch.Tensor) for b in m.blocks
             for n in ("ls1", "ls2"))
    out = {"depth": len(m.blocks), "embed_dim": int(adapter.embed_dim), "heads": int(H), "head_dim": int(dh),
           "mlp_hidden": int(adapter.mlp_hidden), "patch_size": int(adapter.patch_size), "grid": [int(Hg), int(Wg)],
           "class_token": getattr(m, "cls_token", None) is not None, "registers": int(reg["R"]),
           "n_prefix_tokens": int(adapter.n_prefix_tokens), "n_tokens": int(adapter.n_tokens),
           "pos_embed_rows": n_pos, "pos_prefix_rows": n_pos - Hg * Wg if n_pos else None,
           "no_embed_class": bool(getattr(m, "no_embed_class", False)), "layerscale": bool(ls),
           "final_norm": type(m.norm).__name__}
    for name in ("global_pool", "pool_include_prefix"):
        if hasattr(m, name):
            out[name] = getattr(m, name)
    if hasattr(m, "attn_pool"):
        out["attn_pool"] = None if m.attn_pool is None else type(m.attn_pool).__name__
    if hasattr(m, "fc_norm"):
        out["fc_norm"] = type(m.fc_norm).__name__
    head = getattr(m, "head", None)
    if head is not None:
        out["head"] = (f"Linear({head.in_features}, {head.out_features})" if isinstance(head, nn.Linear)
                       else type(head).__name__)
    return out


def check_against_config(adapter, config, img_size: int | None = None) -> dict:
    """Compare a built adapter with a provenance record (configs/models/<name>.yaml, or its dict): blocks, width,
    heads, MLP width, registers, class token, patch size, LayerScale and the token count at the image size.
    Returns {"ok", "checks": [{field, expected, found, ok}], "layout"}."""
    if not isinstance(config, dict):
        import yaml

        path = Path(config)
        path = path if path.is_absolute() else REPO_ROOT / path
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    arch = config.get("architecture", config)
    lay = model_layout(adapter)
    img = int(img_size or adapter.img_size)
    checks = []

    def add(field, expected, found):
        if expected is None:
            return
        ok = (bool(expected) == bool(found)) if isinstance(expected, bool) else (int(expected) == int(found))
        checks.append({"field": field, "expected": expected, "found": found, "ok": bool(ok)})

    add("blocks", arch.get("blocks"), lay["depth"])
    add("width", arch.get("width"), lay["embed_dim"])
    add("heads", arch.get("heads"), lay["heads"])
    add("mlp_hidden", arch.get("mlp_hidden"), lay["mlp_hidden"])
    add("registers", arch.get("registers"), lay["registers"])
    add("cls_token", arch.get("cls_token"), int(lay["class_token"]))
    add("patch_size", arch.get("patch_size"), lay["patch_size"])
    if "layerscale" in arch:
        add("layerscale", bool(arch["layerscale"]), lay["layerscale"])
    if arch.get("patch_size") and arch.get("cls_token") is not None:
        g = img // int(arch["patch_size"])
        add(f"n_tokens at {img} px", int(arch["cls_token"]) + int(arch.get("registers") or 0) + g * g, lay["n_tokens"])
    return {"ok": all(c["ok"] for c in checks), "checks": checks, "layout": lay}


# ----------------------------------------------------------------------------------------------
# DeiT-III: the even classifier, logits, top-1 and its flip consistency
# ----------------------------------------------------------------------------------------------
def deit_head_path(model: nn.Module) -> dict:
    """Check that the classifier reads head(fc_norm(pool(norm(tokens)))) with token pooling (x[:, 0]), fc_norm
    Identity, no attention pooling and a Linear head, so nothing but the final norm sits between the class token
    and the head. Raises ValueError otherwise; returns the facts."""
    from timm.models.vision_transformer import VisionTransformer

    if not isinstance(model, VisionTransformer):
        raise ValueError(f"expected a timm VisionTransformer, got {type(model).__name__}")
    facts = {"global_pool": getattr(model, "global_pool", None),
             "attn_pool": None if getattr(model, "attn_pool", None) is None else type(model.attn_pool).__name__,
             "fc_norm": type(model.fc_norm).__name__, "head_drop": type(model.head_drop).__name__,
             "head": type(model.head).__name__, "class_token": getattr(model, "cls_token", None) is not None,
             "num_prefix_tokens": int(model.num_prefix_tokens), "no_embed_class": bool(model.no_embed_class),
             "final_norm": type(model.norm).__name__}
    problems = []
    if facts["global_pool"] != "token":
        problems.append(f"global_pool is {facts['global_pool']!r}, not 'token'")
    if facts["attn_pool"] is not None:
        problems.append("attention pooling present")
    if not isinstance(model.fc_norm, nn.Identity):
        problems.append("fc_norm is not Identity")
    if not isinstance(model.head_drop, (nn.Dropout, nn.Identity)):
        problems.append(f"head_drop is {facts['head_drop']}")
    if not isinstance(model.head, nn.Linear):
        problems.append("head is not a Linear classifier")
    if not facts["class_token"]:
        problems.append("no class token")
    if problems:
        raise ValueError("classifier path is not norm -> class token -> head: " + "; ".join(problems))
    facts["path"] = "norm -> class token (x[:, 0]) -> fc_norm (Identity) -> head_drop (eval: identity) -> head"
    facts["n_classes"] = int(model.head.out_features)
    return facts


@torch.no_grad()
def install_even_classifier(model: nn.Module, A, tol: float = 1e-6) -> dict:
    """Make the DeiT classifier read the even part of the class token: the even projection (I + A*)/2 between the
    final RMSNorm and the final affine (mirror.m1.install_even_head; the model must be folded with
    final_norm_affine=False, as mirror.m1.build_vexact leaves it). If an even head is already installed (full
    conversion with even_head=True) it is checked against A* and kept. Returns deit_head_path's facts."""
    facts = deit_head_path(model)
    norm = model.norm
    if not (isinstance(norm, nn.Sequential) and hasattr(norm, "rms") and hasattr(norm, "affine")):
        raise ValueError("model.norm must be Sequential(rms, [even], affine): fold with final_norm_affine=False")
    A = exact_involution(A)
    if hasattr(norm, "even"):
        P = 0.5 * (np.eye(A.shape[0]) + A)
        err = float(np.abs(norm.even.P.detach().double().cpu().numpy() - P).max())
        if err > tol:
            raise ValueError(f"an even head for another operator is installed (max |P - (I + A)/2| = {err:.2e})")
        facts["even_head"] = "already installed"
    else:
        install_even_head(model, A)
        facts["even_head"] = "installed"
    facts["classifier_input"] = "affine(rms(t_cls) (I + A*)/2): the even part of the class token"
    return facts


def deit_logits(adapter, x: torch.Tensor, model: nn.Module | None = None) -> torch.Tensor:
    """Logits of the model's own classifier, head(fc_norm(pool(norm(tokens)))), through the adapter's forward pieces
    (differentiable: used for the logit-distillation term and for top-1)."""
    m = adapter.model if model is None else model
    return m.forward_head(m.norm(adapter.forward_tokens(x, m)))


def flip_averaged_logits(adapter, x: torch.Tensor, model: nn.Module | None = None) -> torch.Tensor:
    """(logits(x) + logits(mirror x)) / 2: the teacher target of the DeiT logit term (PREREG Section 2.6)."""
    return 0.5 * (deit_logits(adapter, x, model) + deit_logits(adapter, mirror(x), model))


class EvenOddClassifier(nn.Module):
    """The DeiT classifier on top of an even/odd FastViT: head(fv(x)) (fv returns the final class-token feature)."""

    def __init__(self, fv: nn.Module, head: nn.Module):
        super().__init__()
        self.fv, self.head = fv, head

    def forward(self, x):
        return self.head(self.fv(x))


@torch.no_grad()
def top1_eval(adapter, model: nn.Module | None, loader, device=None, amp: str | None = None, flip: bool = True,
              logits_fn=None, return_preds: bool = False, max_batches: int | None = None) -> dict:
    """Top-1 of the model's own classifier and the flip consistency of the predicted class (the share of images
    whose predicted class equals that of the mirror image), with the relative top-two logit margin of every
    disagreeing image (the near-tie rule of PREREG_FINAL_PHASE.md Section 2.1, bar 1e-4). amp="bf16": bf16
    autocast on CUDA; otherwise the model's dtype (run float32 with TF32 off for the exactness claim).
    `logits_fn(x)` overrides the classifier (for example an EvenOddClassifier)."""
    m = adapter.model if model is None else model
    dev = torch.device(device) if device is not None else next(m.parameters()).device
    fn = logits_fn or (lambda z: deit_logits(adapter, z, m))
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if amp == "bf16" and dev.type == "cuda"
           else contextlib.nullcontext())
    preds, preds_f, labels, margins, margins_one = [], [], [], [], []
    for i, (imgs, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = normalize_for(adapter, imgs.to(dev))
        with ctx:
            lg = fn(x).float()
            lgf = fn(mirror(x)).float() if flip else None
        preds.append(lg.argmax(1).cpu())
        labels.append(torch.as_tensor(y))
        two = lg.double().topk(min(2, lg.shape[1]), dim=1).values
        mg = (two[:, 0] - two[:, 1]).abs() / two[:, 0].abs().clamp_min(1e-300)
        mg_one = mg
        if flip:
            preds_f.append(lgf.argmax(1).cpu())
            two_f = lgf.double().topk(min(2, lgf.shape[1]), dim=1).values
            mg_f = (two_f[:, 0] - two_f[:, 1]).abs() / two_f[:, 0].abs().clamp_min(1e-300)
            # PREREG_FINAL_PHASE.md Section 2.1: a near-tie needs the margin below 1e-4 in BOTH copies, so the larger
            # of the two decides; the smaller is kept as a diagnostic (the CORE-06 one-copy convention)
            mg, mg_one = torch.maximum(mg, mg_f), torch.minimum(mg, mg_f)
        margins.append(mg.cpu())
        margins_one.append(mg_one.cpu())
    pred, lab, marg = torch.cat(preds), torch.cat(labels), torch.cat(margins)
    marg_one = torch.cat(margins_one)
    out = {"top1": 100.0 * float((pred == lab).double().mean()), "n": int(len(pred)),
           "precision": "bf16 autocast" if (amp == "bf16" and dev.type == "cuda") else str(next(m.parameters()).dtype)}
    if flip:
        pf = torch.cat(preds_f)
        same = pred == pf
        dis = (~same).nonzero().flatten()
        out.update({"flip_consistency": 100.0 * float(same.double().mean()), "n_disagree": int(len(dis)),
                    "disagree_max_margin": float(marg[dis].max()) if len(dis) else None,
                    "max_margin_rel": float(marg[dis].max()) if len(dis) else None,
                    "disagree_max_margin_one_copy": float(marg_one[dis].max()) if len(dis) else None,
                    "n_disagree_not_near_tie": int((marg[dis] >= 1e-4).sum()) if len(dis) else 0,
                    "pass_with_near_ties": bool(len(dis) == 0 or bool((marg[dis] < 1e-4).all()))})
    if return_preds:
        out.update({"pred": pred.numpy(), "labels": lab.numpy()})
        if flip:
            out["pred_flip"] = torch.cat(preds_f).numpy()
    return out


# ----------------------------------------------------------------------------------------------
# SigLIP 2: the pooled feature
# ----------------------------------------------------------------------------------------------
def siglip_feature_path(model: nn.Module) -> dict:
    """How the SigLIP 2 k-NN feature is formed (as the CORE-04 runs used it): forward_head(norm(tokens),
    pre_logits=True), i.e. the attention-pooled embedding of all patch tokens after the final norm."""
    ap = getattr(model, "attn_pool", None)
    facts = {"class_token": getattr(model, "cls_token", None) is not None, "global_pool": getattr(model, "global_pool", None),
             "attn_pool": None if ap is None else type(ap).__name__, "fc_norm": type(model.fc_norm).__name__,
             "head": type(model.head).__name__, "pool_include_prefix": bool(getattr(model, "pool_include_prefix", False)),
             "feature": "forward_head(norm(tokens), pre_logits=True) (mirror.adapters_timm feature 'pooled')"}
    if ap is not None:
        facts.update({"latent_len": int(ap.latent_len), "pool_heads": int(ap.num_heads),
                      "pool_mlp": ap.mlp is not None, "pool_type": ap.pool})
        facts["note"] = ("the MAP head is a softmax over patch tokens, so it is invariant to their order, but its key, "
                         "value and output maps are not projected: a full conversion would need an even projection "
                         "before it; A15a converts blocks 1 to 6 only (zero-shot), as CORE-04 did")
    return facts


# ----------------------------------------------------------------------------------------------
# A15a: involutions and the weighted projection
# ----------------------------------------------------------------------------------------------
class Involution:
    """A symmetric orthogonal involution R on R^n, given as a dense matrix or as a permutation vector p
    (R[i, p[i]] = 1: R Y = Y[p], Y R = Y[:, p], the conventions of mirror/surgery.py). `augmented()` appends one
    fixed (even) coordinate, the constant input that carries a bias."""

    def __init__(self, R, augment: int = 0):
        a = R.detach().cpu().numpy() if isinstance(R, torch.Tensor) else np.asarray(R)
        if a.ndim == 1:
            p = a.astype(np.int64)
            if not np.array_equal(p[p], np.arange(len(p))):
                raise ValueError("the permutation must be an involution")
            self.perm, self.dense = p, None
        else:
            M = np.asarray(a, dtype=np.float64)
            if M.shape[0] != M.shape[1]:
                raise ValueError("an involution matrix must be square")
            self.perm, self.dense = None, 0.5 * (M + M.T)
        self.augment = int(augment)

    @property
    def n(self) -> int:
        return (len(self.perm) if self.perm is not None else self.dense.shape[0]) + self.augment

    def augmented(self) -> "Involution":
        out = copy.copy(self)
        out.augment = self.augment + 1
        return out

    def _base_n(self) -> int:
        return self.n - self.augment

    def left(self, Y: torch.Tensor) -> torch.Tensor:
        """R Y (rows)."""
        n0 = self._base_n()
        head = Y[:n0]
        head = head[torch.as_tensor(self.perm, device=Y.device)] if self.perm is not None else \
            torch.as_tensor(self.dense, dtype=Y.dtype, device=Y.device) @ head
        return torch.cat([head, Y[n0:]]) if self.augment else head

    def right(self, Y: torch.Tensor) -> torch.Tensor:
        """Y R (columns)."""
        n0 = self._base_n()
        head = Y[:, :n0]
        head = head[:, torch.as_tensor(self.perm, device=Y.device)] if self.perm is not None else \
            head @ torch.as_tensor(self.dense, dtype=Y.dtype, device=Y.device)
        return torch.cat([head, Y[:, n0:]], 1) if self.augment else head

    def basis(self, device=None) -> tuple:
        """(E+, E-): orthonormal bases of the +1 and -1 eigenspaces, float64 [n, n+] and [n, n-]."""
        n0 = self._base_n()
        if self.perm is not None:
            p = self.perm
            i = np.arange(n0)
            a, s = i[p > i], i[p == i]
            b = p[a]
            Ep = np.zeros((n0, len(a) + len(s)))
            Em = np.zeros((n0, len(a)))
            r = 1.0 / math.sqrt(2.0)
            Ep[a, np.arange(len(a))] = r
            Ep[b, np.arange(len(a))] = r
            Ep[s, len(a) + np.arange(len(s))] = 1.0
            Em[a, np.arange(len(a))] = r
            Em[b, np.arange(len(a))] = -r
        else:
            Q, dp = eig_basis(self.dense)
            Ep, Em = Q[:, :dp], Q[:, dp:]
        if self.augment:
            k = self.augment
            Ep = np.block([[Ep, np.zeros((n0, k))], [np.zeros((k, Ep.shape[1])), np.eye(k)]])
            Em = np.vstack([Em, np.zeros((k, Em.shape[1]))])
        kw = dict(dtype=torch.float64, device=device)
        return torch.as_tensor(Ep, **kw), torch.as_tensor(Em, **kw)

    def project_plain(self, W: torch.Tensor, R_in: "Involution") -> torch.Tensor:
        """(W + R_out W R_in) / 2 with R_out = self."""
        return 0.5 * (W + self.left(R_in.right(W)))


def head_involution(hm: HeadMaps, which: str = "B") -> np.ndarray:
    """The involution on the D = H dh query (B) or value (C) space: block (sigma(h), h) is M[h], so that
    W A* = R W is the condition Wq^g A* = B_h Wq^h, g = sigma(h) (mirror/attention.py project_attn)."""
    M = np.asarray(hm.B if which == "B" else hm.C, dtype=np.float64)
    H, dh = hm.H, hm.dh
    R = np.zeros((H * dh, H * dh))
    for h in range(H):
        g = int(hm.sigma[h])
        R[g * dh:(g + 1) * dh, h * dh:(h + 1) * dh] = M[h]
    return R


def patch_flip_perm(in_chans: int, patch: int) -> np.ndarray:
    """Permutation of the flattened patch (channel, row, column; the order of Conv2d weights and F.unfold) that
    reverses the column: W_flat[:, p] = flip_w(W)_flat, so the patch-embedding condition reads W F = A* W."""
    c, i, j = np.meshgrid(np.arange(in_chans), np.arange(patch), np.arange(patch), indexing="ij")
    return (c * patch * patch + i * patch + (patch - 1 - j)).reshape(-1).astype(np.int64)


def _aug_moment(M: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    n = M.shape[0]
    out = M.new_zeros(n + 1, n + 1)
    out[:n, :n] = M
    out[:n, n] = mean
    out[n, :n] = mean
    out[n, n] = 1.0
    return out


def weighted_projection(W, R_out, R_in, M, bias=None, mean=None, damp: float = DEFAULT_DAMP) -> tuple:
    """The equivariant (X, c) closest to (W, b) in E ||(W - X) u + (b - c)||^2 over inputs u with second moment M
    (normalised, sum u u^T / n) and mean `mean` (sum u / n), subject to X R_in = R_out X and R_out c = c (see the
    module docstring for the derivation). Without `bias` the objective is ||(W - X) M^(1/2)||_F^2. `R_out`, `R_in`:
    Involution or a dense matrix / permutation vector. float64 throughout. Returns (X, c or None, info) with the
    weighted errors of this and of the plain projection (weighted_error)."""
    R_out = R_out if isinstance(R_out, Involution) else Involution(R_out)
    R_in = R_in if isinstance(R_in, Involution) else Involution(R_in)
    W = torch.as_tensor(W).double()
    dev = W.device
    M = torch.as_tensor(M).to(device=dev, dtype=torch.float64)
    M = 0.5 * (M + M.T)
    if bias is not None:
        if mean is None:
            raise ValueError("a bias needs the input mean")
        b = torch.as_tensor(bias).to(device=dev, dtype=torch.float64)
        Wt = torch.cat([W, b[:, None]], 1)
        Mt = _aug_moment(M, torch.as_tensor(mean).to(device=dev, dtype=torch.float64))
        Ri = R_in.augmented()
    else:
        Wt, Mt, Ri = W, M, R_in
    Ep, Em = Ri.basis(dev)
    Npp, Nmm, Npm = Ep.T @ Mt @ Ep, Em.T @ Mt @ Em, Ep.T @ Mt @ Em

    def solve(N, rhs):
        if N.numel() == 0:
            return rhs.new_zeros(N.shape[0], rhs.shape[1])
        lam = float(damp) * float(torch.diagonal(N).mean()) if damp else 0.0
        return torch.linalg.solve(N + lam * torch.eye(N.shape[0], dtype=N.dtype, device=dev), rhs)

    Tp = Ep + Em @ solve(Npp, Npm).T          # E+ + E- N-+ N++^(-1)
    Tm = Em + Ep @ solve(Nmm, Npm.T).T        # E- + E+ N+- N--^(-1)
    Yp = (Wt @ Tp) @ Ep.T
    Ym = (Wt @ Tm) @ Em.T
    Xt = 0.5 * (Yp + R_out.left(Yp)) + 0.5 * (Ym - R_out.left(Ym))
    Xt = R_out.project_plain(Xt, Ri)          # removes rounding residue; the identity on equivariant matrices
    plain = R_out.project_plain(Wt, Ri)
    info = {"weighted_error": weighted_error(Wt, Xt, Mt), "weighted_error_plain": weighted_error(Wt, plain, Mt),
            "frobenius_change": float((Xt - Wt).norm() / Wt.norm().clamp_min(1e-300)),
            "frobenius_change_plain": float((plain - Wt).norm() / Wt.norm().clamp_min(1e-300)),
            "n_even_in": int(Ep.shape[1]), "n_odd_in": int(Em.shape[1]), "damp": float(damp)}
    if bias is not None:
        return Xt[:, :-1], Xt[:, -1], info
    return Xt, None, info


def weighted_error(W, X, M) -> float:
    """tr((W - X) M (W - X)^T) / tr(W M W^T): the activation-weighted breaking share (mirror/variants.py)."""
    D = (W - X).double()
    num = float(torch.einsum("ij,jk,ik->", D, M, D))
    den = float(torch.einsum("ij,jk,ik->", W.double(), M, W.double()))
    return num / den if den > 0 else float("nan")


# ----------------------------------------------------------------------------------------------
# A15a: input moments and the whitened conversion
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def collect_projection_moments(adapter, loader, device, blocks, components=("mlp", "attn"), patch: bool = True,
                               stats_device=None) -> dict:
    """Float64 sums of u u^T and u, and the count, for the input u of every weight the conversion projects, over
    the images of `loader` (fit images, no mirrors), through the folded copy of the original model (the same
    activations as SVD-LLM's statistics, mirror.baselines.collect_stats): the patch-embedding input (patch pixels,
    flattened as the kernel), and per block the qkv input (norm1 output), the attn.proj input (the attention
    context), the fc1 input (norm2 output) and the fc2 input (GELU output)."""
    comps = tuple(components)
    blocks = sorted(int(b) for b in blocks)
    folded, _ = fold_copy(adapter.model, final_norm_affine=False)
    folded.eval()
    sdev = torch.device(device if stats_device is None else stats_device)
    acc = {"blocks": {b: {} for b in blocks}}

    def add(slot: dict, name: str, X: torch.Tensor):
        X = X.reshape(-1, X.shape[-1]).to(device=sdev, dtype=torch.float64)
        e = slot.setdefault(name, {"M": torch.zeros(X.shape[1], X.shape[1], dtype=torch.float64, device=sdev),
                                   "s": torch.zeros(X.shape[1], dtype=torch.float64, device=sdev), "n": 0})
        e["M"] += X.T @ X
        e["s"] += X.sum(0)
        e["n"] += int(X.shape[0])

    handles = []
    pe = folded.patch_embed.proj
    if patch:
        k = pe.kernel_size[0] if isinstance(pe, nn.Conv2d) else None

        def pre_patch(_m, args):
            x = args[0]
            if k is not None:
                add(acc, "patch", F.unfold(x, kernel_size=k, stride=k).transpose(1, 2))

        handles.append(pe.register_forward_pre_hook(pre_patch))
    for b in blocks:
        blk = folded.blocks[b]
        slot = acc["blocks"][b]
        if "attn" in comps:
            handles.append(blk.norm1.register_forward_hook(lambda _m, _a, o, s=slot: add(s, "qkv", o)))
            handles.append(blk.attn.proj.register_forward_pre_hook(lambda _m, a, s=slot: add(s, "o", a[0])))
        if "mlp" in comps:
            handles.append(blk.mlp.fc1.register_forward_pre_hook(lambda _m, a, s=slot: add(s, "fc1", a[0])))
            handles.append(blk.mlp.fc2.register_forward_pre_hook(lambda _m, a, s=slot: add(s, "fc2", a[0])))
    n_img = 0
    try:
        for imgs, _ in loader:
            x = normalize_for(adapter, imgs.to(device))
            adapter.forward_tokens(x, folded)
            n_img += int(x.shape[0])
    finally:
        for h in handles:
            h.remove()
    acc["info"] = {"n_images": n_img, "blocks": [b + 1 for b in blocks], "components": list(comps),
                   "model_form": "folded copy of the original (mirror.fold.fold_copy, final affine kept)",
                   "inputs": "patch pixels; qkv: norm1 output; o: attention context; fc1: norm2 output; fc2: GELU "
                             "output; fit images only, no mirrors", "precision": "model dtype forward, float64 sums"}
    return acc


def _moment(entry: dict) -> tuple:
    n = max(int(entry["n"]), 1)
    return entry["M"] / n, entry["s"] / n


@torch.no_grad()
def build_whitened_vexact(adapter, A, blocks, perms, head_maps, moments: dict, reg_perm=None,
                          components=("mlp", "attn"), even_head: bool = False, damp: float = DEFAULT_DAMP,
                          weighted_patch: bool = True, model: nn.Module | None = None) -> tuple:
    """mirror.m1.build_vexact with the activation-weighted projection (A15a): the same deep copy, position-grid
    materialisation, full LayerNorm folding and plain projection of positions, class and register tokens; the patch
    embedding (if `weighted_patch` and its moments exist), the MLPs (fc1, fc2 with their biases) and the attention
    (Wq, Wk, Wv, Wo with their biases, in the fitted head frame, then gauged as build_vexact does) projected with
    weighted_projection and the input moments of `moments` (collect_projection_moments). Returns (model, info) with
    info as build_vexact's plus "weighted": per matrix the weighted error of the plain and of the weighted
    projection. Exactly equivariant over the span, like build_vexact."""
    from ..m1 import vexact_components

    comps = vexact_components(components)
    blocks = [int(b) for b in blocks]
    if not blocks or blocks != list(range(blocks[0], blocks[0] + len(blocks))):
        raise ValueError("a contiguous, non-empty span of blocks is needed")
    A = exact_involution(A)
    m = copy.deepcopy(adapter.model if model is None else model)
    materialised = materialise_pos_embed(m, adapter)
    fold_layernorm_full(m, final_norm_affine=False)
    rows = []
    emb = None
    if blocks[0] == 0:
        pe = m.patch_embed.proj
        W0 = pe.weight.detach().double().clone()
        b0 = pe.bias.detach().double().clone() if pe.bias is not None else None
        emb = project_embeddings(m, adapter, A, reg_perm)            # plain: positions, class and register tokens
        pm = moments.get("patch")
        if weighted_patch and pm is not None and isinstance(pe, nn.Conv2d):
            d, C, p, _ = W0.shape
            Mp, mu = _moment(pm)
            Mp, mu = Mp.to(W0.device), mu.to(W0.device)
            X, c, inf = weighted_projection(W0.reshape(d, -1), Involution(A), Involution(patch_flip_perm(C, p)), Mp,
                                            bias=b0, mean=mu if b0 is not None else None, damp=damp)
            pe.weight.copy_(X.reshape(W0.shape).to(pe.weight.dtype))
            if b0 is not None:
                pe.bias.copy_(c.to(pe.bias.dtype))
            rows.append({"block": None, "matrix": "patch_embed", **inf})
    k = len(blocks)
    maps_after = []
    for i, b in enumerate(blocks):
        blk = m.blocks[b]
        mb = moments["blocks"].get(b, {})
        if "mlp" in comps:
            f = fold_mlp(blk)
            P = Involution(np.asarray(perms[i], dtype=np.int64))
            if "fc1" in mb and "fc2" in mb:
                M1, mu1 = (t.to(f["W1"].device) for t in _moment(mb["fc1"]))
                M2, mu2 = (t.to(f["W1"].device) for t in _moment(mb["fc2"]))
                X1, c1, i1 = weighted_projection(f["W1"], P, Involution(A), M1, bias=f["b1"], mean=mu1, damp=damp)
                X2, c2, i2 = weighted_projection(f["W2"], Involution(A), P, M2, bias=f["b2"], mean=mu2, damp=damp)
                rows += [{"block": b + 1, "matrix": "fc1", **i1}, {"block": b + 1, "matrix": "fc2", **i2}]
                write_absorbed_mlp(blk, X1, c1, X2, c2)
            else:
                write_absorbed_mlp(blk, **project_mlp_two(**f, A_in=A, A_out=A, perm=perms[i]))
        if "attn" in comps:
            hm = head_maps[i]
            f = fold_attn(blk)
            if "qkv" in mb and "o" in mb:
                dev = f["Wq"].device
                Mi, mui = (t.to(dev) for t in _moment(mb["qkv"]))
                Mo, muo = (t.to(dev) for t in _moment(mb["o"]))
                RB, RC, RA = Involution(head_involution(hm, "B")), Involution(head_involution(hm, "C")), Involution(A)
                e = {}
                for key, R in (("q", RB), ("k", RB), ("v", RC)):
                    X, c, inf = weighted_projection(f["W" + key], R, RA, Mi, bias=f["b" + key], mean=mui, damp=damp)
                    e["W" + key], e["b" + key] = X, c
                    rows.append({"block": b + 1, "matrix": "W" + key, **inf})
                X, c, inf = weighted_projection(f["Wo"], RA, RC, Mo, bias=f["bo"], mean=muo, damp=damp)
                e["Wo"], e["bo"] = X, c
                rows.append({"block": b + 1, "matrix": "Wo", **inf})
                eg, hm_g = gauge_twin_heads(e, hm)                  # as build_vexact (gauge=True): B = C = I on pairs
                eg = project_attn(**eg, A_in=A, A_out=A, head_maps=hm_g, gauge=False)   # rounding clean-up only
                write_absorbed_attn(blk, **eg)
                maps_after.append(hm_g)
            else:
                maps_after += apply_attention_surgery(m, [b], [A], [A], [hm], gauge=True)
    if even_head:
        install_even_head(m, A)
    m.eval()
    from ..m1 import involution_trace64

    info = {"maps_after": maps_after if "attn" in comps else None, "embedding_shares": emb,
            "pos_materialised": materialised, "even_head": bool(even_head), "blocks": blocks,
            "d_plus": int((involution_trace64(A) + A.shape[0]) // 2), "components": list(comps),
            "weighted": rows, "damp": float(damp), "projection": "activation-weighted (A15a)"}
    return m, info
