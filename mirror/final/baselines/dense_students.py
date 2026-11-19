"""Dense students for the stage ablation (FINAL_PHASE_PLAN.md, A04): a width-pruned and a block-dropped DINOv2-B,
both healed with the shared recipe, to test whether Mirror Surgery is "just distillation into a smaller model".

Width-pruned student (width_pruned)
  The existing FLAP compressor (mirror/baselines.py, "flap"; FLAP: An et al., arXiv 2312.11983, code
  github.com/CASIA-IVA-Lab/FLAP, as recorded there) at the full-size Mirror-exact budget with the gate-B choice
  {"allocation": "uniform"} (results/MUST-07/gate_b.json: msel top-1 67.00, params_total 64,555,008 against
  64,554,464, gap +0.0008%). Checked against the plan's description, by reading the code and by
  tests/test_fp_baselines.py:
    * FLAP-style fluctuation scores: flap_scores, S_j = sample variance of input channel j of fc2 (GELU output) or of
      the attention projection (context) over the fit tokens, times ||W[:, j]||^2; a head scores the sum of its
      channels (uniform mode).
    * MLP hidden units are removed (fc1 rows, b1 entries, fc2 columns; smaller nn.Linear layers) and whole attention
      heads are removed in dense form (their rows of wq, wk, wv with biases and their columns of wo set to zero).
    * Budget: mirror.budget._flap_uniform keeps round(H x attention share) heads per block and fills the rest of
      each block's share with neurons, ending at most one neuron (2 d weights) above the budget, i.e. rounding
      towards more parameters, within 1% (gap_ok).
    * Bias compensation: b2 += W2[:, removed] mean(g_removed) and bo += Wo[:, removed channels] mean(c_removed).
    * The zeroed heads are not counted: per_matrix params = kept heads x dh x d, and structure_check confirms the
      realised count equals params_matrices. In training, heal_structured.make_trainable turns wq, wk, wv into
      IndexedLinear(axis "out") and wo into IndexedLinear(axis "in") holding only the kept heads, so the removed
      heads' q, k, v stay exactly zero and wo never reads their context while training.
  Caveat: a block that keeps zero heads would fail make_trainable (IndexedLinear needs at least one kept index).
  The uniform rule keeps round(H x attention share) heads per block; info["min_heads_kept"] reports the smallest
  count and width_pruned logs a warning at zero (check info["pruning"] on the VM (verify); MUST-09 never healed
  FLAP, so this conversion has not yet run at DINOv2-B size).

Block-dropped student (block_dropped, blockdrop_trainable, load_blockdrop)
  Greedy removal of n_drop whole blocks (the plan: 3 of 12, "chosen greedily by the smallest k-NN loss on the
  method-selection set"): each round removes, one at a time, every remaining candidate block, scores the shortened
  network with score_fn(model) (higher is better; on the VM: k-NN top-1 of the 10k method-selection queries against
  the 100k bank, the bank embedded by the shortened network) and removes the block whose removal keeps the highest
  score (ties to the lower block index). Every candidate's score and the removal order are recorded. With
  max_candidates = k, round 1 tries every block and later rounds try only the k remaining blocks with the best
  round-1 scores (fewer evaluations; None tries every remaining block in every round).
  Parameter count: the original model's parameters (hub position table and all, as for the 66,203,360 target)
  minus every parameter of the removed blocks (weights, biases, norms, LayerScale). One DINOv2-B block holds
  7,077,888 matrix weights plus 6,912 biases, 3,072 norm and 1,536 LayerScale entries (7,089,408), so 3 blocks give
  86,583,552 - 21,268,224 = 65,315,328 (0.7544), 1.34% below the target (arithmetic from the DINOv2-B shapes; the
  run prints the measured count). Whole blocks are the unit, so the 1% rule cannot be met and the student is
  reported with its gap (eligible False).
  Trainable form (blockdrop_trainable): the shortened network with the position grid of the evaluation size written
  in (mirror.m1.materialise_pos_embed, the same function at that size, as heal_structured.make_trainable does for
  every method) and every block in absorbed form (norm affines and LayerScale folded into the weights, the same
  function), so the trainable set is the other students': every block's weights and biases and the embeddings
  (patch embedding, position grid, CLS, registers) train; the absorbed norm affines and LayerScale, the final norm
  affine and the mask token stay frozen (BlockDropTrainable.frozen_parameters; run with train_scope "all").
  Block-output distillation: student block j is matched to teacher block kept[j] (spec["block_map"]); the stream is
  not folded, so the block targets are the teacher's raw block outputs (as for the unfolded low-rank baselines).

model_level(adapter_or_model, budget, params_total) converts a compressor's matrix count into the whole-model count
and its gap to the model-level target (N_orig - dense count of the converted matrices + budget.total; 66,203,360
for DINOv2-B at full conversion). svdllm_full and iosvd use it too.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from mirror import baselines as B
from mirror.budget import gap_ok

SPEC_FORMAT = "final-blockdrop-v1"
GATE_B_FLAP = {"allocation": "uniform"}       # results/MUST-07/gate_b.json, flap at the full size, fit seed 0

__all__ = ["SPEC_FORMAT", "GATE_B_FLAP", "model_level", "width_pruned", "block_params", "drop_blocks",
           "block_dropped", "BlockDropTrainable", "blockdrop_trainable", "blockdrop_skeleton", "load_blockdrop"]


def _module(adapter_or_model) -> nn.Module:
    inner = getattr(adapter_or_model, "model", None)
    return inner if isinstance(inner, nn.Module) and hasattr(inner, "blocks") else adapter_or_model


def _n_params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        from mirror.utils import log

        log(msg)


def model_level(adapter_or_model, budget, params_total: int, step: int = 0) -> dict:
    """Whole-model counts for a compressor whose converted matrices hold `params_total` parameters (basis changes
    included) at `budget` (mirror.budget.SpanBudget). Pass the ORIGINAL adapter or model: params_original counts it
    as stored (DINOv2: the hub's 37 x 37 position table and every norm, as in the paper's 0.7646).
    target = params_original - dense count of the converted matrices + budget.total (Mirror-exact's whole-model
    count); gap_model = (params_model - target) / target; eligible = 0 <= gap_model <= 1%, or within one allocation
    step (`step`) where that step is above 1% of the target (only on tiny test models)."""
    m = _module(adapter_or_model)
    n_orig = _n_params(m)
    dense = int(budget.dense_total)
    target = n_orig - dense + int(budget.total)
    params_model = n_orig - dense + int(params_total)
    gap = (params_model - target) / target
    step_rel = float(step) / target if step else 0.0
    return {"params_model": int(params_model), "params_original": int(n_orig), "params_frac": params_model / n_orig,
            "target": int(target), "target_frac": target / n_orig, "gap_model": float(gap),
            "eligible": bool(gap_ok(gap, step_rel)),
            "eligibility_rule": "0 <= (params_model - target) / target <= 1%, rounding towards more parameters "
                                "(within one allocation step where 1% is not reachable)"}


# ----------------------------------------------------------------------------------------------
# Width-pruned dense student
# ----------------------------------------------------------------------------------------------
def width_pruned(adapter, budget, stats, hparams: dict | None = None, *, verbose: bool = True) -> tuple:
    """(width-pruned copy, info): mirror.baselines.compress(adapter.model, "flap", ...) at `budget` with the gate-B
    hyperparameters {"allocation": "uniform"} unless `hparams` overrides them. `stats` must hold sum_g, sq_g (MLP)
    and sum_ctx, sq_ctx (attention): mirror.baselines.collect_stats(adapter, fit_loader, device, budget.blocks,
    methods=["flap"], components=budget.components).

    info: the compressor contract of mirror.baselines.compress (per_matrix entries of kind "width", params_total,
    budget_total, gap, gap_ok) plus "student" "width_pruned", the model-level counts (model_level), "structure_check"
    (removed heads exactly zero, realised count equal to params_matrices) and "pruning" (per block: heads kept,
    neurons kept, bias compensation)."""
    hp = {**GATE_B_FLAP, **(hparams or {})}
    model, info = B.compress(adapter.model, "flap", list(budget.blocks), budget.components, budget, stats, hp,
                             keep_tensors=True, verbose=False)
    sc = B.structure_check(model, info)
    per_block = {}
    for pm in info["per_matrix"]:
        row = per_block.setdefault(int(pm["block"]), {"block": int(pm["block"])})
        if pm["name"] == "wq":
            row.update(heads_kept=list(pm["heads_kept"]), heads=int(pm["heads"]))
        elif pm["name"] == "fc1":
            row["neurons"] = int(pm["n"])
        if pm.get("bias_compensation"):
            row.setdefault("bias_compensation", []).append(pm["name"])
    info.update({"student": "width_pruned", "hparams": dict(info["hparams"]),
                 "structure_check": {"ok": bool(sc["ok"]), "params_realised": int(sc["params_realised"]),
                                     "params_matrices": int(sc["params_matrices"]),
                                     "failed": [c for c in sc["checks"] if not c["ok"]][:5]},
                 "pruning": [per_block[b] for b in sorted(per_block)],
                 "heads_note": "removed heads are zero rows/columns, not counted; make_trainable keeps them at zero "
                               "(IndexedLinear holds the kept heads only)"})
    info.update(model_level(adapter, budget, info["params_total"], info["gap_step"]))
    heads = [r["heads"] for r in info["pruning"] if "heads" in r]
    info["min_heads_kept"] = min(heads) if heads else None
    if heads and min(heads) == 0:
        _log("width_pruned: WARNING a block keeps no attention head; heal_structured.make_trainable cannot convert it",
             True)
    _log(f"width_pruned (flap {hp}): matrices {info['params_total']:,} against {info['budget_total']:,} (gap "
         f"{100 * info['gap']:+.4f}%); model {info['params_model']:,} against target {info['target']:,} (gap "
         f"{100 * info['gap_model']:+.4f}%, eligible {info['eligible']}); structure ok {sc['ok']}", verbose)
    return model, info


# ----------------------------------------------------------------------------------------------
# Block-dropped dense student
# ----------------------------------------------------------------------------------------------
def block_params(model: nn.Module, b: int) -> int:
    """Every parameter of block b (weights, biases, norms, LayerScale)."""
    return _n_params(_module(model).blocks[int(b)])


def drop_blocks(model: nn.Module, kept, copy_weights: bool = True) -> nn.Module:
    """The network with only the blocks `kept` (original indices, in increasing order), renumbered 0..len(kept)-1.

    copy_weights=True: an independent deep copy. False: a light view for scoring that shares every submodule and
    tensor with `model` and has its own blocks container (the input model is not changed; do not train or modify
    the view). The blocks container keeps its type (timm: nn.Sequential, DINOv2: nn.ModuleList), so the adapters'
    layer0_tokens / blocks / norm path runs it unchanged; DINOv2's n_blocks attribute is updated."""
    m = _module(model)
    kept = [int(b) for b in kept]
    if not kept or kept != sorted(set(kept)) or kept[-1] >= len(m.blocks) or kept[0] < 0:
        raise ValueError(f"kept must be increasing block indices in [0, {len(m.blocks) - 1}], got {kept}")
    if copy_weights:
        new = copy.deepcopy(m)
    else:
        new = copy.copy(m)
        for k, v in list(vars(new).items()):        # own dictionaries, shared modules and tensors
            if k.startswith("_") and isinstance(v, dict):
                new.__dict__[k] = type(v)(v)
            elif k.startswith("_") and isinstance(v, set):
                new.__dict__[k] = set(v)
    mods = [new.blocks[b] for b in kept]
    new.blocks = nn.Sequential(*mods) if isinstance(m.blocks, nn.Sequential) else nn.ModuleList(mods)
    if isinstance(getattr(new, "n_blocks", None), int):
        new.n_blocks = len(kept)
    return new


def block_dropped(adapter, n_drop: int, score_fn, max_candidates: int | None = None, *, budget=None,
                  target: int | None = None, score_original: bool = True, verbose: bool = True) -> tuple:
    """(shortened deep copy, info): remove `n_drop` blocks of adapter.model greedily (see the module docstring).

    score_fn(model) -> float, higher is better; it receives a view of the shortened network (drop_blocks with
    copy_weights=False, eval mode) and must not modify it. max_candidates: None tries every remaining block in every
    round; k restricts rounds 2.. to the k remaining blocks with the best round-1 scores. The model-level target
    comes from `budget` (a SpanBudget: model_level's target) or `target` (an int); without either, gap and eligible
    are None.

    info: method "blockdrop", n_drop, order (teacher block indices in removal order), kept, dropped, rounds (per
    round: candidates with their scores, the removed block and its score), score_original (the unmodified network,
    if score_original), params_model, params_original, params_removed, params_frac, block_params, target,
    gap (= gap_model, relative to the target), eligible (False below the target), note."""
    base = adapter.model
    L = len(base.blocks)
    n_drop = int(n_drop)
    if not 1 <= n_drop < L:
        raise ValueError(f"n_drop must be in [1, {L - 1}]")
    if max_candidates is not None and int(max_candidates) < 1:
        raise ValueError("max_candidates must be at least 1")
    kept = list(range(L))
    s0 = float(score_fn(drop_blocks(base, kept, copy_weights=False))) if score_original else None
    if s0 is not None:
        _log(f"block_dropped: original network score {s0:.4f}", verbose)
    rounds, first = [], None
    for rnd in range(n_drop):
        if max_candidates is None or first is None:
            cands = list(kept)
        else:
            ranked = sorted(kept, key=lambda b: (-first[b], b))
            cands = sorted(ranked[:int(max_candidates)])
        scores = {}
        for b in cands:
            view = drop_blocks(base, [k for k in kept if k != b], copy_weights=False)
            scores[b] = float(score_fn(view))
            del view
            _log(f"block_dropped: round {rnd + 1}, without block {b} (0-based; kept {[k for k in kept if k != b]}) "
                 f"score {scores[b]:.4f}", verbose)
        best = max(cands, key=lambda b: (scores[b], -b))
        if first is None:
            first = dict(scores)
        rounds.append({"round": rnd + 1, "kept_before": list(kept), "candidates": [{"block": b, "score": scores[b]}
                                                                                   for b in cands],
                       "removed": int(best), "score_after": scores[best]})
        kept.remove(best)
    order = [r["removed"] for r in rounds]
    model = drop_blocks(base, kept, copy_weights=True).eval()
    n_orig = _n_params(base)
    removed = int(sum(block_params(base, b) for b in order))
    params_model = n_orig - removed
    if budget is not None:
        tgt = model_level(adapter, budget, int(budget.total))["target"]
    else:
        tgt = None if target is None else int(target)
    gap = None if tgt is None else (params_model - tgt) / tgt
    per_block = block_params(base, 0)
    info = {"method": "blockdrop", "student": "block_dropped", "n_drop": n_drop, "n_blocks_teacher": L,
            "order": order, "kept": list(kept), "dropped": sorted(order), "rounds": rounds, "score_original": s0,
            "max_candidates": None if max_candidates is None else int(max_candidates),
            "selection": "greedy; each round removes the block whose removal keeps the highest score_fn value "
                         "(ties to the lower block index)",
            "params_model": int(params_model), "params_original": int(n_orig), "params_removed": removed,
            "params_frac": params_model / n_orig, "block_params": int(per_block), "target": tgt,
            "gap": None if gap is None else float(gap), "gap_model": None if gap is None else float(gap),
            "eligible": None if gap is None else bool(gap_ok(gap)),
            "note": "whole blocks are the unit (one block holds %d parameters%s), so the 1%% rule cannot be met; the "
                    "student is reported with its gap" % (per_block, "" if tgt is None else
                                                          f", {100 * per_block / tgt:.2f}% of the target")}
    _log(f"block_dropped: removed {order} (0-based), kept {kept}; {params_model:,} parameters "
         f"({100 * params_model / n_orig:.2f}% of {n_orig:,})"
         + ("" if gap is None else f" against target {tgt:,}: gap {100 * gap:+.3f}% (eligible {info['eligible']}; "
                                   "below the target by construction)"), verbose)
    return model, info


class BlockDropTrainable:
    """Constraint (mirror/heal.py protocol) for the block-dropped student, run with train_scope "all": every block's
    weights and biases and the embeddings train; frozen are everything else (final norm affine, mask token, any head)
    and, in absorbed form, each block's norm affines and LayerScale (identity after absorption). Nothing is
    projected."""

    def __init__(self, n_blocks: int, absorbed: bool = True, embeddings: bool = True):
        self.n_blocks, self.absorbed, self.embeddings = int(n_blocks), bool(absorbed), bool(embeddings)

    def frozen_parameters(self, model: nn.Module) -> list:
        from mirror.heal import _absorbed_parameters
        from mirror.heal_attn import attn_absorbed_parameters
        from mirror.heal_vexact import span_frozen_parameters

        out = list(span_frozen_parameters(model, range(len(model.blocks)), self.embeddings))
        if self.absorbed:
            for blk in model.blocks:
                out += list(_absorbed_parameters(blk)) + list(attn_absorbed_parameters(blk))
        return out

    def __call__(self, model: nn.Module) -> None:
        return None

    @torch.no_grad()
    def check(self, model: nn.Module) -> dict:
        from mirror.heal import _is_absorbed
        from mirror.heal_attn import is_attn_absorbed

        ok_n = len(model.blocks) == self.n_blocks
        ok_abs = (not self.absorbed) or all(_is_absorbed(b) and is_attn_absorbed(b) for b in model.blocks)
        return {"ok": bool(ok_n and ok_abs), "n_blocks": len(model.blocks), "absorbed_identity": bool(ok_abs)}

    def describe(self) -> dict:
        return {"type": "BlockDropTrainable", "n_blocks": self.n_blocks, "absorbed": self.absorbed,
                "embeddings_train": self.embeddings,
                "frozen": "final norm affine, mask token, heads; absorbed norm affines and LayerScale"}


def _prepare(model: nn.Module, adapter, materialise: bool, absorb: bool) -> bool:
    from mirror.attention import absorb_attn
    from mirror.m1 import materialise_pos_embed
    from mirror.surgery import absorb_block

    done = bool(materialise_pos_embed(model, adapter)) if (materialise and adapter is not None) else False
    if absorb:
        for blk in model.blocks:
            absorb_block(blk)
            absorb_attn(blk)
    return done


def blockdrop_trainable(model: nn.Module, info: dict, adapter, *, absorb: bool = True) -> tuple:
    """(model, constraints, spec) for healing the block-dropped student; `model` (from block_dropped) is changed in
    place: position grid materialised (DINOv2) and blocks absorbed (absorb=True; the same function). Run the trainer
    with train_scope "all" and the returned constraints. spec (JSON) rebuilds the student with load_blockdrop and
    tells the trainer which teacher block each student block matches: block_map {student j: teacher kept[j]}."""
    kept = [int(b) for b in info["kept"]]
    if len(model.blocks) != len(kept):
        raise ValueError(f"model has {len(model.blocks)} blocks, info keeps {len(kept)}")
    materialised = _prepare(model, adapter, True, absorb)
    n = len(kept)
    cons = [BlockDropTrainable(n, absorbed=absorb)]
    spec = {"format": SPEC_FORMAT, "family": "blockdrop", "method": "blockdrop", "blocks": list(range(n)),
            "kept": kept, "order": [int(b) for b in info["order"]], "n_blocks_teacher": int(info["n_blocks_teacher"]),
            "block_map": {str(j): kept[j] for j in range(n)}, "teacher_blocks": kept,
            "base": {"materialise_pos_embed": bool(materialised), "absorbed": bool(absorb), "fold": False},
            "targets": {"blocks": "raw (student block j against teacher block kept[j])", "final": "as every method"},
            "params": {k: info.get(k) for k in ("params_model", "params_original", "params_removed", "params_frac",
                                                "target", "gap", "eligible", "block_params")},
            "trainable": "every block's weights and biases and the embeddings (patch embedding, position grid, CLS, "
                         "registers); frozen: absorbed norm affines and LayerScale, final norm affine, mask token"}
    return model, cons, spec


def blockdrop_skeleton(adapter, spec: dict) -> nn.Module:
    """The block-dropped student's structure with arbitrary weights (load a state dict into it)."""
    if spec.get("format") != SPEC_FORMAT:
        raise ValueError(f"not a {SPEC_FORMAT} spec")
    m = drop_blocks(adapter.model, spec["kept"], copy_weights=True)
    base = spec.get("base", {})
    _prepare(m, adapter, bool(base.get("materialise_pos_embed")), bool(base.get("absorbed")))
    return m.eval()


def load_blockdrop(adapter, spec: dict, state_dict: dict) -> nn.Module:
    """Rebuild a healed block-dropped student from its spec and state dict (strict), in eval mode, frozen."""
    m = blockdrop_skeleton(adapter, spec)
    m.load_state_dict(state_dict, strict=True)
    m.eval()
    m.requires_grad_(False)
    return m
