"""Student registry for the final phase (docs/final_phase_design.md, Section 4.1; FINAL_PHASE_PLAN.md Section 6).

build_student(method, ctx, **hp) -> StudentBuild turns a method name into a trainable student, its constraints,
its distillation targets (TargetMode) and a JSON spec from which load_student rebuilds a checkpoint.

Existing methods are built with the same library calls and arguments as scripts/must09_heal.py (build_compressed and
run_heal), so Gate P reproduces MUST-09:
  mirror_exact, random_structure, pair_random
        mirror.mirror_family.build_mirror_variant(adapter, method, all blocks, ("mlp", "attn"), SpanStructure,
        seed=random_seed (1)); the info gets A_star, perms, maps_after and reg_perm from its structure, as
        must09_heal.build_compressed does. The structure is the canonical file (ctx.structure), any SpanStructure
        file (hp structure=..., e.g. an A*_w fit) or "adopted" (resolved from results/A01/decision.json).
        pair_random is the A03 H1 control: the fitted operator with a random pairing (SpanStructure.exact_parts).
  svd, svdllm, slicegpt, flap (and the other mirror.baselines.METHODS_V2)
        mirror.budget.mirror_exact_budget(model, A*, blocks, perms, head maps) with the fitted structure,
        mirror.baselines.collect_stats(adapter, fit loader of fit seed 0, device, blocks, [method], components) and
        mirror.baselines.compress(model, method, blocks, components, budget, stats, hparams) with the gate-B
        hyperparameters of results/MUST-07/gate_b.json (hp hparams=... overrides).
  then  mirror.heal_structured.make_trainable(copy, info, adapter, lowrank_factors="balanced" as configs/must09.yaml),
        verify_rebuild, the conversion check (feature_gap against the compressed model, tolerance 1e-3 as MUST-09;
        on ctx.probe_images, training images in the jobs, random images in the tests), and the targets of
        mirror.heal_structured.distill_teacher with targets "sym": the mirror family gets flip-averaged final
        targets and block targets symmetrised with its own A* (k = depth); a compressor gets flip-averaged final
        targets and raw block targets (centred when its stream is LayerNorm-folded, SliceGPT), with the fitted
        register pairing in the flip alignment.
  original
        the original model distilled with the same recipe (A04 "healing without surgery"): the position grid
        materialised as make_trainable does, MUST-09's trainable set (blocks' matrices and biases, embeddings;
        norms, LayerScale, final norm and mask token frozen: heal_structured.StructuredTrainable with no records),
        flip-averaged final targets and raw block targets.
  rot90_probe
        the original model with rot90 targets (A15b): the student sees torch.rot90(x, 1, (2, 3)); targets are the
        teacher's raw outputs on x, the patch grid rotated back. Every block parameter and the embeddings train
        (mirror.heal_vexact.SpanTrainable); hp trainable="must09" uses the original's set instead.
Methods of the other agents (imported when called; a clear ImportError when the module is missing):
  mirror_partial   mirror.final.baselines.partial_span.build_partial(adapter, structure, k, components): flip-averaged
                   final targets, block targets symmetrised with A*_k for blocks < k (centred after).
  mirror_svdllm    mirror.final.baselines.mirror_plus_svd.build(adapter, structure, frac, stats, hparams): the mirror
                   family's targets with A*.
  svdllm_full      mirror.final.baselines.svdllm_full.compress(model, blocks, components, budget, stats, hparams, calib)
                   + training_hooks(info), then make_trainable(lowrank_factors="given").
  iosvd            mirror.final.baselines.iosvd.collect_extra_stats + compress, then make_trainable("given").
  blockdrop        mirror.final.baselines.dense_students.block_dropped + blockdrop_trainable; block_map from its info.
The statistics these receive (`stats`: collect_stats of the original model for "svdllm" on fit set 0; `calib`: the
fit-set loader) are this module's reading of design doc 4.2 and are marked (verify) until the modules exist.

load_student(run_dir, step=None, device="cpu") -> (adapter or None, model, info) rebuilds <run_dir>/ckpt/step<S>.pt
(the latest step by default) from <run_dir>/spec.json (+ structure.npz): mirror.heal_structured.build_skeleton for
specs in its format, else the loader the spec names (module and function).

Sources: scripts/must09_heal.py, mirror/heal_structured.py, mirror/mirror_family.py, configs/must09.yaml,
results/MUST-07/gate_b.json; the design doc for the other agents' signatures.
"""
from __future__ import annotations

import copy
import importlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from mirror.heal import exact_involution
from mirror.utils import REPO_ROOT, log

from .targets import TargetMode

MIRROR_FAMILY = ("mirror_exact", "random_structure", "pair_random")
LIBRARY_COMPRESSORS = ("svd", "svdllm", "asvd_scale", "slicegpt", "flap", "zipit")   # mirror.baselines.METHODS_V2
OWN_METHODS = ("original", "rot90_probe")
EXTERNAL_METHODS = {
    "mirror_partial": "mirror.final.baselines.partial_span",
    "mirror_svdllm": "mirror.final.baselines.mirror_plus_svd",
    "svdllm_full": "mirror.final.baselines.svdllm_full",
    "iosvd": "mirror.final.baselines.iosvd",
    "blockdrop": "mirror.final.baselines.dense_students",
}
METHODS = MIRROR_FAMILY + LIBRARY_COMPRESSORS + OWN_METHODS + tuple(EXTERNAL_METHODS)
FP_SPEC_KEY = "fp_heal"
DEFAULT_A01_DECISION = "results/A01/decision.json"

__all__ = ["TargetMode", "StudentBuild", "BuildContext", "build_student", "load_student", "resolve_structure",
           "student_folder", "METHODS", "MIRROR_FAMILY", "LIBRARY_COMPRESSORS", "OWN_METHODS", "EXTERNAL_METHODS"]


# ----------------------------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------------------------
@dataclass
class StudentBuild:
    """A built student (design doc 4.1). `fresh()` returns an identical new (model, constraints) pair (the
    learning-rate trials need one per rate); `blocks` are the loss blocks (None: every block of the student)."""

    method: str
    model: nn.Module
    constraints: list
    spec: dict
    structure: dict | None
    targets: TargetMode
    params: dict
    train_scope: str = "all"
    blocks: list | None = None
    fresh: Callable | None = None
    info: dict = field(default_factory=dict)


@dataclass
class BuildContext:
    """What every builder needs: the teacher adapter (the original model), how to rebuild it, the canonical
    structure, the fit-set statistics and the gate-B hyperparameters. Statistics, budgets and structures are
    computed once and cached."""

    adapter: Any
    model_name: str
    adapter_spec: dict
    device: Any = "cpu"
    structure: Any = None                     # canonical SpanStructure file (or a SpanStructure)
    components: tuple = ("mlp", "attn")
    random_seed: int = 1                      # random_structure / pair_random structure seed (configs/must09.yaml)
    lowrank_factors: str = "balanced"         # configs/must09.yaml
    gate_b: Any = None                        # results/MUST-07/gate_b.json (path or dict)
    fit_seed: int = 0
    fit_loader: Callable | None = None        # fit_loader(seed) -> DataLoader; default mirror.protocol.Protocol
    cache_dir: Any = "cache"
    probe_images: torch.Tensor | None = None  # images for the conversion check (normalised, [N, 3, H, W])
    conversion_tol: float = 1e-3              # configs/must09.yaml checks.conversion_tol
    a01_decision: Any = DEFAULT_A01_DECISION
    adopted_structures: dict | None = None    # {"wspace": A*_w file, "astar": current A* file} for "adopted"
    extra: dict = field(default_factory=dict)  # e.g. blockdrop_score_fn
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def depth(self) -> int:
        return int(self.adapter.depth)

    @property
    def blocks(self) -> list:
        return list(range(self.depth))

    @property
    def dev(self) -> torch.device:
        return torch.device(self.device)

    # ---- structure ----
    def span_struct(self, which=None):
        """The SpanStructure: ctx.structure (None), a path, "adopted" or a SpanStructure object."""
        from mirror.mirror_family import SpanStructure, load_span_structure

        src = self.structure if which is None else which
        if isinstance(src, SpanStructure):
            return src
        if src is None:
            raise ValueError("no structure: set BuildContext.structure (the canonical SpanStructure file)")
        path = resolve_structure(src, self.a01_decision, self.adopted_structures)
        key = ("struct", str(path))
        if key not in self._cache:
            st = load_span_structure(path)
            if st.depth != self.depth:
                raise ValueError(f"{path} has {st.depth} blocks, the model {self.depth}")
            self._cache[key] = st
        return self._cache[key]

    def fitted(self, which=None) -> dict:
        """must09_heal.load_span(random_span=False) for a SpanStructure file: A* of the full span (cleaned), the
        pairings, head maps and register pairing."""
        st = self.span_struct(which)
        L = self.depth
        return {"A_star": exact_involution(st.A_star[L]), "perms": list(st.perms), "heads": list(st.heads),
                "reg_perm": np.asarray(st.reg_perm[L], dtype=np.int64), "span_struct": st}

    def budget(self, which=None):
        """mirror.budget.mirror_exact_budget at full conversion, as must09_heal.build_compressed calls it."""
        key = ("budget", id(self.span_struct(which)))
        if key not in self._cache:
            from mirror.budget import mirror_exact_budget

            f = self.fitted(which)
            blocks = self.blocks
            self._cache[key] = mirror_exact_budget(self.adapter.model, f["A_star"], blocks,
                                                   [f["perms"][b] for b in blocks], [f["heads"][b] for b in blocks],
                                                   components=tuple(self.components), model_name=self.model_name)
        return self._cache[key]

    # ---- statistics ----
    def get_fit_loader(self, seed: int | None = None):
        seed = self.fit_seed if seed is None else int(seed)
        if self.fit_loader is not None:
            return self.fit_loader(seed)
        from mirror.protocol import Protocol

        key = ("protocol",)
        if key not in self._cache:     # as must09_heal.knn_runtime: batch 128, stats batch 16, 4 workers
            self._cache[key] = Protocol(_resolve(self.cache_dir), 128, 16, 4, self.dev.type == "cuda")
        return self._cache[key].fit_loader(seed)

    def stats(self, method: str):
        """mirror.baselines.collect_stats(adapter, fit loader, device, blocks, [method], components), cached."""
        key = ("stats", method)
        if key not in self._cache:
            from mirror.baselines import collect_stats

            self._cache[key] = collect_stats(self.adapter, self.get_fit_loader(), self.dev, self.blocks, [method],
                                             tuple(self.components))
        return self._cache[key]

    def gate_b_hparams(self, method: str):
        """The hyperparameters gate B chose for `method` (None when gate B has no entry or is not given)."""
        g = self.gate_b
        if g is None:
            return None
        if not isinstance(g, dict):
            g = json.loads(Path(_resolve(g)).read_text(encoding="utf-8"))
        for r in g.get("ranking") or []:
            if r.get("method") == method:
                return dict(r.get("hparams") or {})
        return None

    def probe(self, n: int = 8) -> torch.Tensor:
        if self.probe_images is not None:
            return self.probe_images
        g = torch.Generator().manual_seed(0)
        s = int(self.adapter.img_size)
        return torch.randn(n, 3, s, s, generator=g, dtype=self.adapter.dtype).to(self.dev)


def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def resolve_structure(src, a01_decision=DEFAULT_A01_DECISION, choices: dict | None = None) -> Path:
    """A SpanStructure path. "adopted" reads the A01 decision file (results/A01/decision.json, written by
    scripts/fp_a01_decide.py): its boolean "adopted" picks choices["wspace"] (A*_w adopted) or choices["astar"]
    (the job's adopted_structures, so every model and fit set gets its own file). Only when no choices are given
    at all does a structure path named in the decision file ("structure", "structure_file", "adopted_structure" or
    "path"; DINOv2-B's main structure) apply. A student may carry its own choices as
    structure: {adopted: {wspace: <file>, astar: <file>}} (the 9-block point and the A02 fit sets need their own)."""
    if isinstance(src, dict):
        if "adopted" in src:
            return resolve_structure("adopted", a01_decision, src["adopted"])
        if src.get("path"):
            return _resolve(src["path"])
        raise ValueError(f"structure {src!r}: give a path, 'adopted' or {{adopted: {{wspace: ..., astar: ...}}}}")
    if str(src) == "adopted":
        dec = _resolve(a01_decision)
        if not dec.exists():
            raise FileNotFoundError(f"structure 'adopted': {dec} does not exist (the A01 decision is not made yet)")
        d = json.loads(dec.read_text(encoding="utf-8"))
        if choices and any(choices.get(k) for k in ("wspace", "astar")):
            if "adopted" not in d:
                raise ValueError(f"{dec} has no boolean 'adopted'")
            pick = "wspace" if bool(d["adopted"]) else "astar"
            if not choices.get(pick):
                raise ValueError(f"structure 'adopted': the decision picks {pick!r}, but this job or student gives no "
                                 f"{pick} file (adopted_structures: {choices})")
            return _resolve(choices[pick])
        for key in ("structure", "structure_file", "adopted_structure", "path"):
            v = d.get(key)
            if isinstance(v, dict):
                v = v.get("path") or v.get("file")
            if v:
                return _resolve(v)
        raise ValueError(f"{dec} names no structure file and no adopted_structures choice applies "
                         "(give adopted_structures: {wspace: <A*_w file>, astar: <current A* file>})")
    return _resolve(src)


def student_folder(method: str, variant: str | None = None) -> str:
    """<method>[_<variant>] (design doc 4.4)."""
    return method if not variant else f"{method}_{re.sub(r'[^A-Za-z0-9.-]+', '-', str(variant))}"


# ----------------------------------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------------------------------
def build_student(method: str, ctx: BuildContext, **hp) -> StudentBuild:
    """Dispatch on the method name (module docstring). Keyword options: structure (path, "adopted" or a
    SpanStructure), hparams (compressor hyperparameters; default gate B), random_seed, lowrank_factors, logits
    (DeiT-III: add the logits term), k (mirror_partial, default 9), frac (mirror_svdllm), n_drop, score_fn and
    max_candidates (blockdrop), trainable (rot90_probe: "blocks" or "must09")."""
    if method in MIRROR_FAMILY:
        sb = _build_mirror_family(method, ctx, hp)
    elif method in LIBRARY_COMPRESSORS:
        sb = _build_library(method, ctx, hp)
    elif method in OWN_METHODS:
        sb = _build_original(method, ctx, hp)
    elif method in EXTERNAL_METHODS:
        sb = _EXTERNAL[method](ctx, hp)
    else:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    if hp.get("logits"):
        sb.targets.logits = True
        if sb.targets.input_transform is not None:
            raise ValueError("logits are not used with the rot90 probe")
    sb.spec.setdefault(FP_SPEC_KEY, {}).update(method=method, targets=sb.targets.describe(),
                                               builder_options={k: _json_safe(v) for k, v in hp.items()
                                                                if k not in ("score_fn",)})
    return sb


def _json_safe(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    return repr(v)


def _mirror_info(model_info: dict, method: str, blocks: list, comps: tuple, builder: str) -> dict:
    """must09_heal.build_compressed's post-processing of build_mirror_variant's info."""
    info = dict(model_info)
    st_ = info.get("structure") or {}
    info.update(kind="mirror", method=method, blocks=blocks, components=list(comps), builder=builder)
    for key, alt in (("A_star", "A"), ("perms", "perms"), ("maps_after", "heads_after"), ("reg_perm", "reg_perm")):
        if key not in info and alt in st_:
            info[key] = st_[alt]
    missing = [k for k in ("A_star", "perms", "maps_after", "reg_perm") if k not in info]
    if missing:
        raise ValueError(f"mirror-family info lacks {missing}")
    return info


def _build_mirror_family(method: str, ctx: BuildContext, hp: dict) -> StudentBuild:
    from mirror.mirror_family import build_mirror_variant

    st = ctx.span_struct(hp.get("structure"))
    blocks, comps = ctx.blocks, tuple(ctx.components)
    seed = int(hp.get("random_seed", ctx.random_seed))
    model, info = build_mirror_variant(ctx.adapter, method, blocks, comps, st, seed=seed)
    info = _mirror_info(info, method, blocks, comps, "mirror.mirror_family.build_mirror_variant")
    if method in ("random_structure", "pair_random"):
        info["random_structure"] = {"seed": seed, **(info.get("structure_info") or {})}
    info["structure_source"] = str(hp.get("structure") or ctx.structure)
    targets_struct = {"A_star": info["A_star"], "reg_perm": info["reg_perm"]}
    return _finish(method, ctx, hp, model, info, targets_struct, hparams=None)


def _build_library(method: str, ctx: BuildContext, hp: dict) -> StudentBuild:
    import mirror.baselines as lib

    if method not in lib.METHODS_V2:
        raise ValueError(f"unknown method {method!r}; mirror.baselines.METHODS_V2 = {lib.METHODS_V2}")
    fitted = ctx.fitted(hp.get("structure"))
    if hp.get("frac") is not None:
        # A06: a smaller model, frac of the original's parameters (MUST-09 accounting), one compression ratio for
        # every matrix (mirror.final.baselines.sizes.budget_for_fraction; SVD-LLM's own rule)
        from mirror.final.baselines.sizes import budget_for_fraction

        budget = budget_for_fraction(ctx.adapter, float(hp["frac"]), components=tuple(ctx.components))
    else:
        budget = ctx.budget(hp.get("structure"))
    hparams = hp["hparams"] if "hparams" in hp else ctx.gate_b_hparams(method)
    stats = ctx.stats(method)
    blocks, comps = ctx.blocks, tuple(ctx.components)
    model, info = lib.compress(ctx.adapter.model, method, blocks, comps, budget, stats, hparams)
    info = dict(info)
    info.update(kind="compressed", method=method, blocks=info.get("blocks", blocks),
                components=info.get("components", list(comps)), source="mirror.baselines.compress",
                budget={"total": getattr(budget, "total", None)}, fit_seed=int(ctx.fit_seed))
    targets_struct = {"A_star": fitted["A_star"], "reg_perm": fitted["reg_perm"]}
    return _finish(method, ctx, hp, model, info, targets_struct, hparams=hparams)


def _finish(method, ctx, hp, compressed: nn.Module, info: dict, tstruct: dict, hparams,
            lowrank_factors: str | None = None, extra_constraints: Callable | None = None) -> StudentBuild:
    """must09_heal.run_heal from the compressed model on: make_trainable, spec, verify_rebuild, conversion check,
    distill_teacher's targets."""
    from mirror.heal_structured import feature_gap, make_trainable, param_counts, verify_rebuild

    compressed.eval()
    factors = str(lowrank_factors or hp.get("lowrank_factors", ctx.lowrank_factors))

    def fresh():
        m, cons, _ = make_trainable(copy.deepcopy(compressed), info, adapter=ctx.adapter, lowrank_factors=factors)
        if extra_constraints is not None:
            cons = list(cons) + list(extra_constraints())
        return m, cons

    model, cons, spec = make_trainable(copy.deepcopy(compressed), info, adapter=ctx.adapter, lowrank_factors=factors)
    if extra_constraints is not None:
        cons = list(cons) + list(extra_constraints())
    spec.update(adapter=ctx.adapter_spec, targets="sym", model=ctx.model_name, hparams=hparams)
    verify_rebuild(ctx.adapter, spec, model)
    conv = feature_gap(ctx.adapter, compressed, model, ctx.probe())
    conv.update(tol=float(ctx.conversion_tol), passed=bool(conv["rel_max"] <= float(ctx.conversion_tol)))
    if not conv["passed"]:
        raise ValueError(f"{method}: the trainable form does not compute the compressed model's function "
                         f"(CLS feature max relative difference {conv['rel_max']:.2e} > {ctx.conversion_tol:g})")
    blocks = [int(b) for b in spec["blocks"]]
    if spec["family"] == "mirror":
        structure = {"A_star": info["A_star"], "perms": info["perms"], "maps_after": info["maps_after"],
                     "reg_perm": info["reg_perm"], "blocks": blocks}
        if blocks != list(range(len(blocks))):
            raise ValueError("the mirror span starts at block 0")
        # distill_teacher: VExactTeacher(adapter, A, len(blocks), reg_perm, symmetrise=True, "symmetrised")
        # (A as MUST-09 passed it: VExactTeacher cleans it once with exact_involution, and so does the helper)
        targets = TargetMode("symmetrised", "symmetrised", A=np.asarray(tstruct["A_star"]),
                             reg_perm=tstruct["reg_perm"], k=len(blocks))
        if len(blocks) != ctx.depth:
            targets = TargetMode("raw", "symmetrised", A=targets.A, reg_perm=targets.reg_perm, k=len(blocks))
    else:
        structure = None
        # distill_teacher: block targets centred for a folded stream, else raw; k = max(blocks) + 1 for a prefix span,
        # else the depth; flip-averaged final targets when k is the depth (VExactTeacher's even head), raw otherwise
        k_dt = max(blocks) + 1 if blocks == list(range(max(blocks) + 1)) else ctx.depth
        targets = TargetMode("symmetrised" if k_dt == ctx.depth else "raw",
                             "centred" if spec["base"].get("fold") else "raw", reg_perm=tstruct["reg_perm"])
    params = param_counts(ctx.adapter, model, spec, structure)
    params.update({k: info.get(k) for k in ("params_total", "params_matrices", "params_basis_change", "budget_total",
                                            "gap", "gap_ok") if k in info})
    spec.setdefault(FP_SPEC_KEY, {})["loader"] = {"kind": "heal_structured"}
    keep = {k: info.get(k) for k in ("method", "source", "builder", "params_total", "budget_total", "gap", "gap_ok",
                                     "budget", "fit_seed", "random_structure", "structure_info", "structure_source",
                                     "hparams", "model_form") if k in info}
    keep["conversion_check"] = conv
    return StudentBuild(method=method, model=model, constraints=list(cons), spec=spec, structure=structure,
                        targets=targets, params=params, train_scope="all", fresh=fresh, info=keep)


def _build_original(method: str, ctx: BuildContext, hp: dict) -> StudentBuild:
    """The original model as a student (module docstring: original, rot90_probe)."""
    from mirror.heal_structured import SPEC_FORMAT, StructuredTrainable, _base_spec, param_counts, verify_rebuild
    from mirror.heal_vexact import SpanTrainable
    from mirror.m1 import materialise_pos_embed

    blocks = ctx.blocks
    trainable = hp.get("trainable", "must09" if method == "original" else "blocks")
    if trainable not in ("must09", "blocks"):
        raise ValueError("trainable must be 'must09' or 'blocks'")

    def constraints():
        return [StructuredTrainable(blocks, [], embeddings=True)] if trainable == "must09" else \
            [SpanTrainable(blocks, embeddings=True)]

    def fresh():
        m = copy.deepcopy(ctx.adapter.model)
        materialise_pos_embed(m, ctx.adapter)
        return m.eval(), constraints()

    model, cons = fresh()
    spec = {"format": SPEC_FORMAT, "family": "compressed", "method": method, "blocks": blocks,
            "components": list(ctx.components), "base": _base_spec(model, ctx.adapter), "wrappers": [], "modules": [],
            "matrices": [], "compressor": {"source": "none: the original model"}, "lowrank_factors": None,
            "trainable": ("blocks' matrices and biases and the embeddings (MUST-09's set); norms, LayerScale, final "
                          "norm affine and mask token frozen") if trainable == "must09" else
                         "every block parameter and the embeddings; final norm affine, mask token and heads frozen",
            "adapter": ctx.adapter_spec, "model": ctx.model_name, "targets": "sym" if method == "original" else "rot90",
            "hparams": None}
    verify_rebuild(ctx.adapter, spec, model)
    src = hp.get("structure", ctx.structure)
    reg = ctx.fitted(src)["reg_perm"] if (method == "original" and src is not None) else None
    targets = (TargetMode("symmetrised", "raw", reg_perm=reg) if method == "original"
               else TargetMode("raw", "raw", input_transform="rot90"))
    params = param_counts(ctx.adapter, model, spec, None)
    spec[FP_SPEC_KEY] = {"loader": {"kind": "heal_structured"}}
    return StudentBuild(method=method, model=model, constraints=cons, spec=spec, structure=None, targets=targets,
                        params=params, train_scope="all", fresh=fresh,
                        info={"method": method, "trainable": trainable, "source": "the original model"})


# ---- methods of the other agents (imported when called) ----------------------------------------
def _module(method: str):
    name = EXTERNAL_METHODS[method]
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise ImportError(f"method {method!r} needs {name}, which is not importable yet ({e}); it is written by "
                          "another agent (docs/final_phase_design.md 4.2)") from e


def _external_spec(spec: dict, ctx: BuildContext, method: str, loader: dict) -> dict:
    spec = dict(spec or {})
    spec.setdefault("method", method)
    spec.setdefault("family", "external")
    spec.update(adapter=ctx.adapter_spec, model=ctx.model_name)
    spec.setdefault(FP_SPEC_KEY, {})["loader"] = loader
    return spec


def _params_of(info: dict, model: nn.Module) -> dict:
    out = {k: info.get(k) for k in ("params_model", "params_frac", "target", "gap", "eligible", "params_total",
                                    "budget_total") if k in info}
    out["model_params"] = int(sum(p.numel() for p in model.parameters()))
    return out


def _mirror_targets_from(info: dict, st, k: int, depth: int) -> TargetMode:
    """Symmetrised targets with the builder's own operator (info or info["structure"]; else A*_k of the structure)."""
    sd = info.get("structure") if isinstance(info.get("structure"), dict) else {}
    A = info.get("A_star", info.get("A", sd.get("A_star", sd.get("A"))))
    reg = info.get("reg_perm", sd.get("reg_perm"))
    if A is None or reg is None:
        parts = st.exact_parts("mirror_exact", k)
        A = parts["A"] if A is None else A
        reg = parts["reg_perm"] if reg is None else reg
    return TargetMode("symmetrised", "symmetrised", A=np.asarray(A), reg_perm=np.asarray(reg, dtype=np.int64),
                      k=int(k))


def _build_mirror_partial(ctx: BuildContext, hp: dict) -> StudentBuild:
    mod = _module("mirror_partial")
    st = ctx.span_struct(hp.get("structure"))
    k = int(hp.get("k", 9))

    def make():
        m, cons, spec, info = mod.build_partial(ctx.adapter, st, k, tuple(ctx.components))
        return m, list(cons), spec, info

    model, cons, spec, info = make()
    targets = _mirror_targets_from(info, st, k, ctx.depth)
    spec = _external_spec(spec, ctx, "mirror_partial", {"kind": "module", "module": EXTERNAL_METHODS["mirror_partial"],
                                                        "function": "load_partial"})
    structure = info.get("structure") if isinstance(info.get("structure"), dict) else None
    keep = {"k": k, **{kk: info.get(kk) for kk in ("params_model", "params_frac", "gap")}}
    return StudentBuild("mirror_partial", model, cons, spec, structure, targets, _params_of(info, model),
                        fresh=lambda: make()[:2], info=keep)


def _build_mirror_svdllm(ctx: BuildContext, hp: dict) -> StudentBuild:
    mod = _module("mirror_svdllm")
    st = ctx.span_struct(hp.get("structure"))
    frac = float(hp["frac"])
    hparams = hp.get("hparams", ctx.gate_b_hparams("svdllm"))
    stats = ctx.stats("svdllm")                 # (verify) which statistics the stacking builder expects

    def make():
        m, cons, spec, info = mod.build(ctx.adapter, st, frac, stats, hparams)
        return m, list(cons), spec, info

    model, cons, spec, info = make()
    targets = _mirror_targets_from(info, st, int(spec.get("k") or ctx.depth), ctx.depth)
    spec = _external_spec(spec, ctx, "mirror_svdllm", {"kind": "module", "module": EXTERNAL_METHODS["mirror_svdllm"],
                                                       "function": "load"})
    structure = info.get("structure") if isinstance(info.get("structure"), dict) else None
    return StudentBuild("mirror_svdllm", model, cons, spec, structure, targets, _params_of(info, model),
                        fresh=lambda: make()[:2], info={"frac": frac, "hparams": hparams})


def _build_lowrank_external(method: str, ctx: BuildContext, hp: dict) -> StudentBuild:
    mod = _module(method)
    fitted = ctx.fitted(hp.get("structure"))
    budget = ctx.budget(hp.get("structure"))
    blocks, comps = ctx.blocks, tuple(ctx.components)
    hparams = hp.get("hparams")
    stats = ctx.stats("svdllm")                 # (verify) the whitening statistics these builders expect
    if method == "svdllm_full":     # calib: only used with hparams closed_form_update=True (svdllm_full docstring)
        calib = {"adapter": ctx.adapter, "loader": ctx.get_fit_loader(), "device": ctx.dev}
        model, info = mod.compress(ctx.adapter.model, blocks, comps, budget, stats, hparams, calib)
        hooks = lambda: list(mod.training_hooks(info) or [])          # noqa: E731
    else:
        extra = mod.collect_extra_stats(ctx.adapter, ctx.get_fit_loader(), ctx.dev, blocks, comps)
        model, info = mod.compress(ctx.adapter.model, blocks, comps, budget, stats, extra, hparams)
        hooks = None
    info = dict(info)
    info.update(kind="compressed", method=method, blocks=info.get("blocks", blocks),
                components=info.get("components", list(comps)), source=EXTERNAL_METHODS[method],
                fit_seed=int(ctx.fit_seed))
    targets_struct = {"A_star": fitted["A_star"], "reg_perm": fitted["reg_perm"]}
    return _finish(method, ctx, hp, model, info, targets_struct, hparams=hparams, lowrank_factors="given",
                   extra_constraints=hooks)


def _build_blockdrop(ctx: BuildContext, hp: dict) -> StudentBuild:
    mod = _module("blockdrop")
    score_fn = hp.get("score_fn") or ctx.extra.get("blockdrop_score_fn")
    if score_fn is None:
        raise ValueError("blockdrop needs score_fn (the k-NN loss on the method-selection set); pass score_fn=... "
                         "or ctx.extra['blockdrop_score_fn']")
    n_drop = int(hp.get("n_drop", 3))
    model0, info = mod.block_dropped(ctx.adapter, n_drop, score_fn, hp.get("max_candidates"))

    def make():
        m, cons, spec = mod.blockdrop_trainable(copy.deepcopy(model0), info, ctx.adapter)
        return m, list(cons), spec

    model, cons, spec = make()
    kept = info.get("kept_blocks", info.get("kept"))
    bm = info.get("block_map") or ({j: int(b) for j, b in enumerate(kept)} if kept is not None else None)
    if bm is None:
        raise ValueError("blockdrop info names neither block_map nor kept_blocks")
    src = hp.get("structure", ctx.structure)
    reg = ctx.fitted(src)["reg_perm"] if src is not None else None
    targets = TargetMode("symmetrised", "raw", reg_perm=reg, block_map=bm)
    spec = _external_spec(spec, ctx, "blockdrop", {"kind": "module", "module": EXTERNAL_METHODS["blockdrop"],
                                                   "function": "load_blockdrop"})
    keep = {"n_drop": n_drop, "block_map": {str(k): v for k, v in bm.items()}}
    return StudentBuild("blockdrop", model, cons, spec, None, targets, _params_of(info, model),
                        fresh=lambda: make()[:2], info=keep)


_EXTERNAL = {"mirror_partial": _build_mirror_partial, "mirror_svdllm": _build_mirror_svdllm,
             "svdllm_full": lambda ctx, hp: _build_lowrank_external("svdllm_full", ctx, hp),
             "iosvd": lambda ctx, hp: _build_lowrank_external("iosvd", ctx, hp), "blockdrop": _build_blockdrop}


# ----------------------------------------------------------------------------------------------
# Loading checkpoints
# ----------------------------------------------------------------------------------------------
def _ckpt_steps(run: Path) -> list:
    out = []
    for p in (run / "ckpt").glob("step*.pt"):
        m = re.fullmatch(r"step(\d+)\.pt", p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def load_student(run_dir, step: int | None = None, device="cpu", adapter=None) -> tuple:
    """Rebuild a checkpoint of a lockstep student: (adapter or None, model, info).

    info: run_dir, step, ckpt, spec, method, variant, family, targets (the TargetMode description), the
    checkpoint's meta, the structure (mirror family: A_star and A, perms, maps_after, reg_perm) and the
    parameter counts where the spec allows them."""
    from mirror.heal_structured import (SPEC_FORMAT, build_adapter_from_spec, build_skeleton, load_structure,
                                        param_counts)

    run = Path(run_dir)
    spec = json.loads((run / "spec.json").read_text(encoding="utf-8"))
    fp = spec.get(FP_SPEC_KEY) or {}
    steps = _ckpt_steps(run)
    if not steps:
        raise FileNotFoundError(f"{run / 'ckpt'} holds no checkpoint")
    step = steps[-1] if step is None else int(step)
    ckpt = run / "ckpt" / f"step{step}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} does not exist (steps saved: {steps})")
    if adapter is None and spec.get("adapter"):
        try:
            adapter = build_adapter_from_spec(spec["adapter"], device)
        except Exception as e:                      # e.g. no model download allowed here
            log(f"load_student: the adapter could not be built ({type(e).__name__}: {e}); returning None")
            adapter = None
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    loader = fp.get("loader") or {"kind": "heal_structured"}
    if loader.get("kind") == "heal_structured" and spec.get("format") == SPEC_FORMAT:
        if adapter is None:
            raise ValueError("rebuilding this student needs the original model's adapter")
        model = build_skeleton(adapter, spec)
        model.load_state_dict(sd, strict=True)
    elif loader.get("kind") == "module":
        fn = getattr(importlib.import_module(loader["module"]), loader["function"])
        model = fn(adapter, spec, sd)
    else:
        raise ValueError(f"{run / 'spec.json'}: unknown loader {loader}")
    model.to(device=device).eval()
    model.requires_grad_(False)
    info = {"run_dir": str(run), "step": step, "ckpt": str(ckpt), "steps_saved": steps, "spec": spec,
            "method": fp.get("method", spec.get("method")), "variant": fp.get("variant"), "family": spec.get("family"),
            "targets": fp.get("targets"), "meta": payload.get("meta") if isinstance(payload, dict) else None,
            "structure": None}
    if (run / "structure.npz").exists():               # the mirror family, the partial span, Mirror + SVD
        st = load_structure(run / "structure.npz")
        st["A"] = st["A_star"]
        info["structure"] = st
    if adapter is not None and spec.get("format") == SPEC_FORMAT:
        try:
            info.update(param_counts(adapter, model, spec, info["structure"]))
        except Exception as e:
            info["param_counts_error"] = f"{type(e).__name__}: {e}"
    return adapter, model, info
