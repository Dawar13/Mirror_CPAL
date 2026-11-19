"""Mirror on the first k blocks only: the A06 "price of exactness" point (FINAL_PHASE_PLAN.md v2, A06 and A06-3).

The student is Mirror-exact on blocks 0..k-1 with the span operator A*_k of the structure (stream layers 0..k, each
normalised to equal energy; mirror.m1.fit_span_operator), built exactly as MUST-07's size "<k>both" by
mirror.mirror_family.build_mirror_variant("mirror_exact", blocks 0..k-1, MLP and attention): the whole model is
LayerNorm-folded, the embeddings are projected with A*_k, MLP and attention of the span are projected, and no even head
is installed (k < depth). Blocks k..depth-1 stay dense (folded, the same function as the original's) and are trained.
For k = 9 on dinov2_vitb14_reg this is MUST-07's "9both": 42,852,816 matrix parameters plus one d x d basis change at
the span end (589,824), so 66,325,200 model parameters, 0.766 of 86,583,552 (results/MUST-07/rows_dinov2_vitb14_reg.csv;
partial_span_params reproduces it from the structure file). It is within +0.18% of the full conversion (66,203,360,
0.7646), the reference size reported as `target`.

Training (constraint protocol of mirror/heal.py; run the trainer with train_scope "all"):
  span blocks 0..k-1   re-projected with A*_k after every step (heal.MirrorMLPConstraint, heal_attn.MirrorAttnConstraint
                       with the head maps valid after surgery), their absorbed norm affines and LayerScale frozen;
  embeddings           trained and re-projected with A*_k after every step (PartialSpanEmbeddings, the re-projection of
                       heal_vexact.VExactEmbeddings);
  blocks k..depth-1    dense and trainable (every parameter, LayerScale included). VExactEmbeddings would freeze them
                       (it freezes everything outside the span); PartialSpanEmbeddings freezes only the final norm
                       affine, the mask token and any classifier head (heads never train in MUST-09), plus the absorbed
                       norms and LayerScale of the span blocks.
So the stream is exactly equivariant at layers 0..k (for the mirrored image the aligned tokens equal the original
tokens times A*_k), before and after every optimiser step; the later blocks and the final features are not.

Targets (a recommendation for the trainer, in info["targets"]; the heal agent decides): the flip-averaged final targets
used for every method in MUST-09, A*_k-symmetrised block targets for blocks 0..k-1 and centred teacher block outputs
for blocks k.. (the student's stream is the folded, centred one): TargetMode(final="symmetrised", blocks="symmetrised",
A=A*_k, reg_perm, k). Note that VExactTeacher(k < depth) on its own would give raw final targets.

Sources: FINAL_PHASE_PLAN.md A06; notes/paper/method_spec.md Sections 1, 4 and 5; mirror/mirror_family.py,
mirror/heal_vexact.py, mirror/heal.py, mirror/heal_attn.py; results/MUST-07 (the 9both count). No web sources.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from ... import mirror_family as MF
from ...budget import BUDGET_TOL, gap_ok
from ...heal import MirrorMLPConstraint, _absorbed_parameters, exact_involution
from ...heal_attn import MirrorAttnConstraint, attn_absorbed_parameters
from ...heal_vexact import VExactEmbeddings
from .sizes import full_model_params

METHOD = "mirror_partial"
SPEC_FORMAT = "fp-mirror-partial-v1"

__all__ = ["METHOD", "SPEC_FORMAT", "PartialSpanEmbeddings", "partial_span_params", "build_partial", "load_partial",
           "load", "partial_constraints"]


class PartialSpanEmbeddings(VExactEmbeddings):
    """VExactEmbeddings for a span that ends before the last block: re-projects the embeddings with A*_k after every
    optimiser step (and checks them, as VExactEmbeddings), but freezes only the final norm affine, the mask token, any
    classifier head (head, fc_norm) and the absorbed norm affines and LayerScale of the span blocks, so the unconverted
    blocks k..depth-1 train."""

    def frozen_parameters(self, model: nn.Module) -> list:
        out = []
        norm = getattr(model, "norm", None)
        if isinstance(norm, nn.Module):
            out += list(norm.parameters())
        mt = getattr(model, "mask_token", None)
        if isinstance(mt, nn.Parameter):
            out.append(mt)
        for name in ("head", "fc_norm"):
            mod = getattr(model, name, None)
            if isinstance(mod, nn.Module):
                out += list(mod.parameters())
        for b in self.blocks:
            out += list(_absorbed_parameters(model.blocks[b])) + list(attn_absorbed_parameters(model.blocks[b]))
        return out

    def describe(self) -> dict:
        d = super().describe()
        d.update({"type": "PartialSpanEmbeddings",
                  "frozen": "final norm affine, mask token, classifier head, absorbed norms and LayerScale of the span "
                            "blocks; the unconverted blocks train"})
        return d


def partial_span_params(model_or_adapter, structure, k: int, components=("mlp", "attn")) -> dict:
    """Parameter accounting of the k-block point and of the full conversion, from the structure alone (the MUST-07
    accounting: N - span dense + Mirror-exact's matrices + its basis changes). model_or_adapter may be a dict with
    d, h, H, dh, depth and full_params (no weights needed). Returns params_matrices, params_basis_change, span_dense,
    params_model, params_frac, full_params, target (the full conversion's model count, the size the point is compared
    with), target_frac, gap = (params_model - target) / target and eligible (0 <= gap <= 1%)."""
    comps = MF.normalise_components(components)
    m = model_or_adapter if isinstance(model_or_adapter, dict) else getattr(model_or_adapter, "model", model_or_adapter)
    L = int(structure.depth)
    k = int(k)
    if not 1 <= k <= L:
        raise ValueError(f"k must be in [1, {L}]")
    N = full_model_params(model_or_adapter)
    bk = MF.span_budget(m, structure, range(k), comps)
    bL = MF.span_budget(m, structure, range(L), comps)
    pm = N - int(bk.dense_total) + int(bk.total)
    full = N - int(bL.dense_total) + int(bL.total)
    gap = (pm - full) / full
    return {"k": k, "params_matrices": int(bk.matrix_total), "params_basis_change": int(bk.basis_change),
            "params_total": int(bk.total), "span_dense": int(bk.dense_total), "params_model": int(pm),
            "params_frac": pm / N, "full_params": int(N), "target": int(full), "target_frac": full / N,
            "target_matrices": int(bL.matrix_total), "gap": gap, "eligible": bool(gap_ok(gap, 0.0, BUDGET_TOL))}


def _structure_dict(binfo: dict, blocks: list) -> dict:
    s = binfo["structure"]
    return {"A_star": exact_involution(s["A"]), "perms": list(s["perms"]), "maps_after": list(s["heads_after"]),
            "reg_perm": np.asarray(s["reg_perm"], dtype=np.int64), "blocks": list(blocks), "heads": list(s["heads"])}


def partial_constraints(adapter, structure: dict, components=("mlp", "attn")) -> list:
    """The constraints of a k-block student from its structure (mirror.heal_structured.load_structure format:
    A_star, perms, maps_after, reg_perm, blocks), for training or for checks after loading."""
    comps = MF.normalise_components(components)
    blocks = [int(b) for b in structure["blocks"]]
    k = len(blocks)
    A = exact_involution(structure["A_star"])
    reg = structure.get("reg_perm")
    reg = None if reg is None or len(reg) == 0 else np.asarray(reg, dtype=np.int64)
    cons = []
    if "mlp" in comps:
        cons.append(MirrorMLPConstraint(blocks, [A] * k, [A] * k, structure["perms"]))
    if "attn" in comps:
        cons.append(MirrorAttnConstraint(blocks, [A] * k, [A] * k, structure["maps_after"]))
    cons.append(PartialSpanEmbeddings(adapter, A, blocks, reg, train=True))
    return cons


@torch.no_grad()
def build_partial(adapter, structure, k: int = 9, components=("mlp", "attn")) -> tuple:
    """Mirror-exact on blocks 0..k-1, the rest dense and trainable (see the module docstring).

    structure: a mirror_family.SpanStructure (or its path). Returns (model, constraints, spec, info); run
    mirror/heal.py with train_scope "all". info: params_model, params_frac, target (the full conversion's count),
    gap, eligible, the MUST-07 fields of build_mirror_variant and the structure in the mirror family's structure.npz
    format (mirror.heal_structured.save_structure)."""
    if isinstance(structure, (str, bytes)) or hasattr(structure, "__fspath__"):
        structure = MF.load_span_structure(structure)
    comps = MF.normalise_components(components)
    L = int(adapter.depth)
    k = int(k)
    if not 1 <= k < L:
        raise ValueError(f"k must be in [1, {L - 1}] (k = {L} is the full conversion, method mirror_exact)")
    blocks = list(range(k))
    model, binfo = MF.build_mirror_variant(adapter, "mirror_exact", blocks, comps, structure)
    st = _structure_dict(binfo, blocks)
    cons = partial_constraints(adapter, st, comps)
    sizes = partial_span_params(adapter, structure, k, comps)
    if sizes["params_total"] != int(binfo["params_total"]):
        raise AssertionError(f"accounting differs from the built model ({sizes['params_total']} against "
                             f"{binfo['params_total']})")
    targets = {"final": "symmetrised", "blocks": "symmetrised", "k": k,
               "note": "flip-averaged final targets (as every MUST-09 method); A*_k-symmetrised block targets for "
                       "blocks 0..k-1, centred teacher outputs for the later blocks"}
    spec = {"format": SPEC_FORMAT, "family": "mirror_partial", "method": METHOD, "k": k, "blocks": blocks,
            "components": list(comps),
            "base": {"materialise_pos_embed": bool(binfo["build"].get("pos_materialised")), "fold": True,
                     "even_head": False},
            "structure_file": "structure.npz", "params": sizes, "train_scope": "all", "targets": targets,
            "mirror": {"d_plus": int(binfo["build"]["d_plus"]),
                       "n_pairs": [int((np.asarray(p) != np.arange(len(p))).sum() // 2) for p in st["perms"]],
                       "n_head_pairs": [int(m.n_pairs) for m in st["maps_after"]]},
            "trainable": "span blocks re-projected with A*_k after every step, embeddings re-projected, blocks k.. "
                         "dense and free; final norm affine, mask token, heads and the span's absorbed norms and "
                         "LayerScale frozen"}
    info = {"method": METHOD, "k": k, "blocks": blocks, "components": list(comps), **sizes,
            "mirror_exact_info": MF.describe_info(binfo), "structure": st, "targets": targets, "train_scope": "all"}
    return model, cons, spec, info


@torch.no_grad()
def load_partial(adapter, spec: dict, state_dict: dict) -> nn.Module:
    """Rebuild a k-block student from its spec and model.state_dict(): the original model with the evaluation grid
    written in and LayerNorm folded (final affine kept in the model, no even head), then the state loaded."""
    from ...fold import fold_layernorm_full
    from ...m1 import materialise_pos_embed

    if spec.get("format") != SPEC_FORMAT:
        raise ValueError(f"not a {SPEC_FORMAT} spec")
    m = copy.deepcopy(adapter.model)
    materialise_pos_embed(m, adapter)
    fold_layernorm_full(m, final_norm_affine=False)
    m.load_state_dict(state_dict, strict=True)
    return m.eval()


load = load_partial
