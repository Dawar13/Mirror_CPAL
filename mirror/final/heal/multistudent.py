"""Lockstep multi-student healing (FINAL_PHASE_PLAN.md Section 7 item 5; docs/final_phase_design.md Sections 4.3,
4.4 and 6): one process, one data stream, one teacher forward per micro-batch, N students updated in lockstep.

Equivalence with mirror.heal.heal() (Gate P). Every student follows heal()'s semantics exactly, as
scripts/must09_heal.py drives it:
  - the same seeding (mirror.utils.set_seed(seed)), constraint protocol (prepare, frozen_parameters, the
    projection once before training and after every optimiser step, check at the end; plus the optional hook
    begin_step(step, total, model) before every step, which no MUST-09 constraint has and which the full
    SVD-LLM baseline uses to switch its trained factor group), trainable set
    (mirror.heal.configure_trainable with the student's train_scope, "all" in MUST-09), plan
    (mirror.heal._make_plan), term weights and names, AdamW groups (weight decay on ndim >= 2 only; fused on CUDA
    float32 as in heal()), warm-up round(warmup_frac x total) then cosine (mirror.heal.lr_at), with the student's
    own learning rate;
  - the same data: the rows of mirror.heal._step_rows (shared by all students), the crops and flips of
    mirror.heal.sample_crop_params with one CPU generator seeded seed + 7919 (shared: every single run of heal()
    draws the same sequence), the same augmentation (crop_resize_flip and the ImageNet normalisation, bit for bit
    equal to mirror.heal.augment_batch, with the mean and std tensors cached on the device);
  - the same targets: mirror/final/heal/targets.py derives each student's targets from one teacher pass with
    VExactTeacher's own stream_target and final_targets, so they equal distill_teacher's targets;
  - the same loss (mirror.heal.distill_terms with the fixed normalisers, summed in the same term order, outside
    autocast as in heal()), gradient accumulation over micro-batches (loss x micro/batch), autocast (bf16 on
    CUDA, none on CPU or float64), no gradient clipping unless configured (clip_grad_norm_ is called with an
    infinite threshold, as heal() does, for the logged gradient norm);
  - the same fixed normalisers: the teacher target mean squares over the same calibration images (1,024 evenly
    spaced sorted training rows, centre view), computed once per distinct target mode;
  - the same held-out loss on the held-out rows (centre view, at the start, every eval_every steps and at the
    end) and the same loss history.
Students interact only through the shared teacher tensors, which carry no gradient. The tests check that one
student trained here equals heal() bit for bit and that N students in lockstep equal their single runs
(tests/test_fp_heal.py). One difference is documented: when the students of one job mix flip-symmetrised and
raw targets, the teacher runs on [x; mirror(x)] for all of them and a raw student takes the first half, which can
differ from a teacher run on x alone by floating-point rounding (MUST-09 used flip-averaged final targets for every
method, so Gate P is unaffected).

Efficiency (Section 6): uint8 data resident on the GPU (mirror/final/heal/data_gpu.py) or streamed from pinned
memory; crops on the device; one teacher forward per micro-batch; students run one after another inside the
micro-batch loop, so only one student's activation graph is alive; fused AdamW; optional torch.compile of each
student's blocks (off by default; compiled kernels round differently, so Gate P jobs keep it off). Losses and
gradient norms stay on the device and are copied to the host only every log_every steps, at hook steps, held-out
evaluations, saved training states and the end ("flush"). On CUDA with fused AdamW a student whose loss turns
non-finite is frozen without a host synchronisation (the optimiser's found_inf flag skips the update, as the AMP
grad scaler does); at the next flush it is marked diverged at the first non-finite step and leaves the lockstep,
with the history and state heal() would have stopped with. On the CPU, or with a non-fused optimiser, the check
runs every step, exactly as in heal().

Hook steps (default 0, 100, 300 and the last step): each live student's state dict is saved as
<run_dir>/ckpt/step<S>.pt and an evaluation request is written to the queue directory
(docs/final_phase_design.md 4.4); nothing is evaluated in this process. The training state of all students
(models, optimisers, the augmentation generator, RNG states, step, losses, normalisers) is saved atomically every
state_every steps; a resumed job continues bit for bit (tests/test_fp_heal.py).

Learning-rate selection runs the (method, rate) pairs as students of one lockstep job (select_lr_lockstep): each
has its own warm-up and cosine over the trial budget, the held-out loss at the end scores it, the lowest wins and
ties go to the smaller rate (diverged trials score +inf).
"""
from __future__ import annotations

import gc
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

from mirror.adapters import IMAGENET_MEAN, IMAGENET_STD
from mirror.heal import (HealConfig, HealData, _CsvLog, _autocast_factory, _describe, _jsonable, _make_plan,
                         _restore_rng_states, _rng_states, _save_train_state, _term_weights, as_config, centre_view,
                         configure_trainable, crop_resize_flip, distill_terms, lr_at, sample_crop_params, term_names)
from mirror.utils import env_info, git_info, log, set_seed

from .data_gpu import LockstepData
from .targets import SharedTeacher, TargetMode, student_features

LOCKSTEP_STATE_FORMAT = "mirror.final.heal lockstep state v1"
DEFAULT_HOOK_STEPS = (0, 100, 300)          # plus the last step
LOGITS_WEIGHT = 1.0

__all__ = ["LOCKSTEP_STATE_FORMAT", "DEFAULT_HOOK_STEPS", "LockstepStudent", "EvalQueue", "train_lockstep",
           "select_lr_lockstep", "choose_lr", "default_hook_steps", "save_checkpoint", "load_lockstep_state",
           "DeviceAugment"]


# ----------------------------------------------------------------------------------------------
# Students, hooks and the evaluation queue
# ----------------------------------------------------------------------------------------------
@dataclass
class LockstepStudent:
    """One student of a lockstep job. `blocks` are the student blocks whose outputs enter the loss (default:
    every block of the student); `tags` (method, variant, experiment, model, seed) go into the records and the
    evaluation requests."""

    name: str
    model: nn.Module
    constraints: list
    mode: TargetMode
    lr: float
    blocks: list | None = None
    run_dir: str | Path | None = None
    train_scope: str = "all"
    compile: bool = False
    tags: dict = field(default_factory=dict)


def default_hook_steps(total: int, steps: Iterable[int] = DEFAULT_HOOK_STEPS) -> list:
    """The configured steps below the budget plus the last step."""
    return sorted({int(s) for s in steps if 0 <= int(s) < int(total)} | {int(total)})


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-") or "x"


def _atomic_json(path: Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_jsonable(obj), indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


@dataclass
class EvalQueue:
    """Writes evaluation requests (docs/final_phase_design.md 4.4) after a checkpoint is saved:
    {"run_dir", "step", "passes", "n_query", "priority"} plus informational keys (ckpt, tags, created_utc).
    Full validation (n_query_full) at the first and last hook steps, n_query_partial in between; the fp32 pass
    at the last step (fp32_last, or the student's tags["fp32"] when set: the fp32 pass of all 50k queries and
    the 100k bank costs about 200k float32 forward passes, so jobs ask for it only for the exactly invariant
    students whose float32 flip consistency is scored); priority 2 for the last step, 1 for the first, 0
    otherwise (higher first)."""

    queue_dir: str | Path
    n_query_full: int = 50000
    n_query_partial: int = 10000
    fp32_last: bool = True
    priority_last: int = 2
    priority_first: int = 1
    priority_other: int = 0

    def request(self, run_dir, step: int, first: int, last: int, ckpt=None, tags: dict | None = None) -> dict:
        full = int(step) in (int(first), int(last))
        fp32 = (tags or {}).get("fp32")                 # a student's own choice overrides the job's fp32_last
        fp32 = self.fp32_last if fp32 is None else bool(fp32)
        passes = ["bf16"] + (["fp32"] if fp32 and int(step) == int(last) else [])
        prio = (self.priority_last if int(step) == int(last) else
                self.priority_first if int(step) == int(first) else self.priority_other)
        return {"run_dir": str(run_dir), "step": int(step), "passes": passes,
                "n_query": int(self.n_query_full if full else self.n_query_partial), "priority": int(prio),
                "ckpt": None if ckpt is None else str(ckpt), "tags": dict(tags or {}),
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def write(self, req: dict) -> Path:
        qd = Path(self.queue_dir)
        qd.mkdir(parents=True, exist_ok=True)
        t = time.time()
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(t)) + f"{int((t % 1) * 1e6):06d}"
        tags = req.get("tags") or {}
        rid = _slug("-".join(str(tags.get(k)) for k in ("experiment", "model", "student") if tags.get(k) is not None)
                    + f"-step{int(req['step'])}")
        return _atomic_json(qd / f"{stamp}_{rid}.json", req)


def save_checkpoint(model: nn.Module, path, step: int, meta: dict | None = None) -> Path:
    """Atomic torch.save of {"state_dict" (the full student state dict on the CPU, the model's dtype), "step",
    "meta"}; mirror.heal_structured.load_healed_model and students.load_student read it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    tmp = path.with_name(path.name + ".partial")
    torch.save({"state_dict": sd, "step": int(step), "meta": _jsonable(meta or {})}, tmp)
    os.replace(tmp, path)
    return path


# ----------------------------------------------------------------------------------------------
# Augmentation on the device
# ----------------------------------------------------------------------------------------------
class DeviceAugment:
    """mirror.heal.augment_batch with the normalisation constants cached on the device (so no host copy per
    call); the arithmetic is the same: crop_resize_flip, then (x / 255 - mean) / std in `dtype`."""

    def __init__(self, device, dtype: torch.dtype, out_size: int):
        self.dtype = dtype
        self.out_size = int(out_size)
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=dtype).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(IMAGENET_STD, dtype=dtype).view(1, 3, 1, 1).to(device)

    def __call__(self, x_uint8: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        x = x_uint8.permute(0, 3, 1, 2).to(self.dtype)
        y = crop_resize_flip(x, params, self.out_size)
        return (y.to(self.dtype) / 255.0 - self.mean) / self.std


# ----------------------------------------------------------------------------------------------
# Per-student runtime state
# ----------------------------------------------------------------------------------------------
@dataclass
class _Pending:
    step: int                 # optimiser step count after this step
    vec: torch.Tensor         # float64 [n_terms] accumulated terms (device)
    gn: torch.Tensor          # gradient norm (device)
    lr: float
    seconds: float


class _Run:
    """Everything heal() keeps for one student, set up in heal()'s order."""

    def __init__(self, s: LockstepStudent, cfg: HealConfig, teacher_model: nn.Module, n_prefix: int,
                 device: torch.device, model_dtype: torch.dtype, fused: bool, apply_constraints: bool,
                 shared: SharedTeacher):
        self.s = s
        self.name = s.name
        self.model = s.model
        self.mode = s.mode
        L = len(self.model.blocks)
        blocks = list(range(L)) if s.blocks is None else [int(b) for b in s.blocks]
        if not blocks or len(set(blocks)) != len(blocks) or min(blocks) < 0 or max(blocks) >= L:
            raise ValueError(f"{s.name}: blocks must be distinct indices in [0, {L})")
        self.capture = sorted(blocks)
        self.teacher_blocks = sorted({self.mode.teacher_block(j) for j in self.capture})
        if max(self.teacher_blocks) >= shared.depth:
            raise ValueError(f"{s.name}: block_map points beyond the teacher's {shared.depth} blocks")
        self.cfg = replace(cfg, lr=float(s.lr), blocks=list(blocks), train_scope=s.train_scope)
        self.constraints = list(s.constraints or [])
        for c in self.constraints:
            prep = getattr(c, "prepare", None)
            if callable(prep):
                prep(self.model)
        self.trainable = configure_trainable(self.model, blocks, s.train_scope, self.constraints)
        if not self.trainable:
            raise ValueError(f"{s.name}: no trainable parameters")
        if apply_constraints:
            with torch.no_grad():
                for c in self.constraints:
                    c(self.model)
        self.plan = _make_plan(teacher_model, self.model, self.cfg, n_prefix)
        self.weights = _term_weights(self.cfg)
        self.names = term_names(blocks)
        if self.mode.logits:
            self.weights["logits"] = LOGITS_WEIGHT
            self.names.append("logits")
        decay = [p for _, p in self.trainable if p.ndim >= 2 or self.cfg.decay_1d]
        no_decay = [p for _, p in self.trainable if not (p.ndim >= 2 or self.cfg.decay_1d)]
        groups = [g for g in ({"params": decay, "weight_decay": float(self.cfg.weight_decay)},
                              {"params": no_decay, "weight_decay": 0.0}) if g["params"]]
        self.fused = bool(fused)
        opt_kw = {"fused": True} if self.fused else {}
        self.opt = torch.optim.AdamW(groups, lr=float(self.cfg.lr), betas=tuple(self.cfg.betas),
                                     eps=float(self.cfg.eps), **opt_kw)
        self.params = [p for _, p in self.trainable]
        self.n_trainable = int(sum(p.numel() for p in self.params))
        if s.compile:
            for blk in self.model.blocks:
                blk.compile()
        self.key = shared.key(self.mode, self.capture)
        self.normalisers = None
        self.history: list = []
        self.held_initial = None
        self.held_final = None
        self.held_rows_log: list = []
        self.diverged = False
        self.steps_done = 0
        self.dead = torch.zeros((), dtype=torch.bool, device=device)
        self.pending: list = []
        self.acc: dict = {}
        self.lr_now = float(s.lr)
        self.train_log = None
        self.held_log = None
        self.run_dir = Path(s.run_dir) if s.run_dir is not None else None

    @property
    def live(self) -> bool:
        return not self.diverged

    def identity(self, total: int) -> dict:
        return {"name": self.name, "config": _jsonable(asdict(self.cfg)), "total_steps": int(total),
                "n_trainable": self.n_trainable, "target_key": repr(self.key), "train_scope": self.s.train_scope,
                "names": list(self.names)}

    def step_total(self) -> torch.Tensor:
        """weights . terms in heal()'s order (Python's sum starts at 0; 0 + x = x exactly)."""
        tot = None
        for k in self.names:
            w = self.weights[k] * self.acc[k]
            tot = w if tot is None else tot + w
        return tot


def _host_loss(run: _Run, vec) -> tuple:
    """(term values, loss) as heal() computes them on the host."""
    vals = [float(v) for v in vec]
    term_vals = {k: v for k, v in zip(run.names, vals)}
    loss_val = float(sum(run.weights[k] * v for k, v in term_vals.items()))
    return term_vals, loss_val


def _groups(runs: Sequence[_Run]) -> list:
    """[(key, [runs])] in order of first appearance."""
    out: dict = {}
    for r in runs:
        out.setdefault(r.key, []).append(r)
    return list(out.items())


def _tensor_bytes(run: _Run) -> int:
    seen, n = set(), 0
    tensors = list(run.model.parameters()) + list(run.model.buffers())
    tensors += [p.grad for p in run.model.parameters() if p.grad is not None]
    if run.opt is not None:
        for st in run.opt.state.values():
            tensors += [v for v in st.values() if isinstance(v, torch.Tensor)]
    for t in tensors:
        if id(t) not in seen:
            seen.add(id(t))
            n += t.numel() * t.element_size()
    return int(n)


# ----------------------------------------------------------------------------------------------
# Training state
# ----------------------------------------------------------------------------------------------
def load_lockstep_state(path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    st = torch.load(path, map_location="cpu", weights_only=False)   # our own file: RNG states need pickle
    if st.get("format") != LOCKSTEP_STATE_FORMAT:
        raise ValueError(f"{path} is not a {LOCKSTEP_STATE_FORMAT!r} file")
    return st


# ----------------------------------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------------------------------
def train_lockstep(teacher_adapter, students: Sequence[LockstepStudent], data, cfg, *, job_dir=None,
                   hook_steps: Iterable[int] = (), queue: EvalQueue | None = None, checkpoints: bool = True,
                   final_checkpoint: bool = True, state_every: int = 0, resume: bool = False,
                   logits_fn: Callable | None = None, data_mode: str = "auto", max_resident_gb: float | None = None,
                   fused="auto", sync_each_step="auto", heldout_initial: bool = True, step_hook: Callable | None = None,
                   ckpt_meta: dict | None = None, log_prefix: str = "lockstep") -> dict:
    """Heal every student in lockstep (module docstring). Returns {"students": {name: metrics + "history"},
    "job": job summary}.

    cfg            the shared HealConfig (lr and blocks are taken from each student; train_scope too).
    job_dir        where the training state (train_state.pt) and summary.json go.
    hook_steps     optimiser step counts at which every live student's checkpoint is saved (checkpoints) and an
                   evaluation request is queued (queue). 0 runs before the first step.
    state_every    save the training state every n steps (needs job_dir); resume continues from it.
    fused          "auto" (heal()'s rule: CUDA and float32), True or False.
    sync_each_step "auto" (True unless the optimiser is fused on CUDA) or a bool: check every step on the host
                   (heal()'s behaviour) instead of the device-side divergence flag.
    step_hook      optional callable(step, {name: model}) after the built-in hook (tests use it to interrupt).
    """
    cfg = as_config(cfg)
    t0 = time.time()
    students = list(students)
    if not students:
        raise ValueError("no students")
    names = [s.name for s in students]
    if len(set(names)) != len(names):
        raise ValueError(f"student names must be unique: {names}")
    teacher = teacher_adapter.model
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    device = next(teacher.parameters()).device
    model_dtype = next(students[0].model.parameters()).dtype
    for s in students:
        if s.model is teacher:
            raise ValueError("the student must be a separate copy of the teacher")
        if next(s.model.parameters()).device != device:
            raise ValueError("teacher and students must be on the same device")
        if next(s.model.parameters()).dtype != model_dtype:
            raise ValueError("every student of one job must have the same dtype (the batch is shared)")
        s.model.eval()
    dtype = torch.float64 if model_dtype == torch.float64 else torch.float32
    if isinstance(data, LockstepData):
        ldata = data
    elif isinstance(data, HealData):
        ldata = LockstepData.from_heal_data(data, device, data_mode, max_resident_gb=max_resident_gb,
                                            prefetch=int(cfg.prefetch))
    else:
        raise TypeError("data must be a LockstepData or a mirror.heal.HealData")
    if ldata.image_size < cfg.crop_size:
        raise ValueError(f"images ({ldata.image_size}) are smaller than crop_size ({cfg.crop_size})")
    if cfg.normaliser not in ("fixed", "batch"):
        raise ValueError("normaliser must be 'fixed' or 'batch'")
    jdir = Path(job_dir) if job_dir is not None else None
    if jdir is not None:
        jdir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg.seed))
    autocast, amp_mode = _autocast_factory(device, cfg.amp, model_dtype)
    B = int(cfg.batch_size)
    mb = max(1, min(int(cfg.micro_batch), B))
    n_train = len(ldata.train_rows)
    per_epoch = n_train // B
    if per_epoch < 1:
        raise ValueError(f"batch_size {B} is larger than the {n_train} training images")
    total = int(cfg.max_steps) if cfg.max_steps is not None else int(math.floor(cfg.epochs * per_epoch))
    if total < 1:
        raise ValueError("the budget gives no optimiser steps")
    warmup = int(cfg.warmup_steps) if cfg.warmup_steps is not None else int(round(cfg.warmup_frac * total))
    warmup = max(0, min(warmup, total))
    use_fused = (device.type == "cuda" and model_dtype == torch.float32) if fused == "auto" else bool(fused)
    sync_each = (not (use_fused and device.type == "cuda")) if sync_each_step == "auto" else bool(sync_each_step)
    if not sync_each and not use_fused:
        raise ValueError("the device-side divergence flag needs the fused optimiser (sync_each_step=True otherwise)")

    shared = SharedTeacher(teacher_adapter, logits_fn)
    state_path = jdir / "train_state.pt" if jdir is not None else None
    state = load_lockstep_state(state_path) if (resume and state_path is not None) else None
    runs = [_Run(s, cfg, teacher, int(teacher_adapter.n_prefix_tokens), device, model_dtype, use_fused,
                 apply_constraints=state is None, shared=shared) for s in students]
    identity = {"config": _jsonable(asdict(cfg)), "total_steps": int(total), "n_train": int(n_train),
                "n_heldout": 0 if ldata.heldout_rows is None else int(len(ldata.heldout_rows)),
                "students": [r.identity(total) for r in runs]}
    if state is not None:
        if state["identity"] != _jsonable(identity):
            raise ValueError(f"{state_path} was saved by another job (config, budget, students or data differ); "
                             "remove it or start without resume")
        if not 0 < int(state["step"]) < total:
            raise ValueError(f"{state_path}: saved step {state['step']} is outside 1..{total - 1}")

    by_key = _groups(runs)
    flip_all = any(r.mode.needs_flip for r in runs)

    def pass_needs(live_runs):
        cap = sorted({i for r in live_runs for i in r.teacher_blocks})
        return flip_all, cap, any(r.mode.logits for r in live_runs)

    # fixed normalisers: teacher target mean squares on the calibration images, once per target mode
    if state is not None:
        norms = state["normalisers"]
    elif cfg.normaliser == "fixed":
        sorted_train = np.sort(ldata.train_rows)
        n_cal = max(1, min(int(cfg.calib_images), n_train))
        calib = sorted_train[np.linspace(0, n_train - 1, n_cal).round().astype(np.int64)]
        norms = _calibrate(shared, ldata, np.unique(calib), cfg, by_key, pass_needs(runs), dtype, autocast)
    else:
        norms = None
    for r in runs:
        r.normalisers = None if norms is None else norms[repr(r.key)]
    held_rows = ldata.heldout_rows
    if held_rows is not None and cfg.max_heldout is not None:
        held_rows = held_rows[:int(cfg.max_heldout)]
    has_held = held_rows is not None and len(held_rows) > 0

    def held_eval(live_runs) -> dict:
        if not has_held or not live_runs:
            return {}
        return _heldout(shared, ldata, held_rows, cfg, live_runs, pass_needs(live_runs), dtype, autocast,
                        logits_fn)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_step, t_offset = 0, 0.0
    if state is not None:
        saved = {d["name"]: d for d in state["students"]}
        for r in runs:
            d = saved[r.name]
            r.model.load_state_dict(d["model"])
            if d.get("optimizer") is not None:
                r.opt.load_state_dict(d["optimizer"])
            r.history = [float(h) for h in d["history"]]
            r.held_initial = d["held_initial"]
            r.held_rows_log = list(d.get("held_rows_log") or [])
            r.diverged = bool(d["diverged"])
            r.steps_done = int(d["steps_done"])
            if r.diverged:
                r.opt = None
        start_step = int(state["step"])
        t_offset = float(state.get("seconds_train", 0.0))
    elif heldout_initial:
        for name, res in held_eval(runs).items():
            run = next(r for r in runs if r.name == name)
            run.held_initial = res
            run.held_rows_log.append({"step": 0, **res})

    keep = start_step if state is not None else None
    for r in runs:
        if r.run_dir is not None:
            r.run_dir.mkdir(parents=True, exist_ok=True)
            r.train_log = _CsvLog(r.run_dir / "train_log.csv",
                                  ["step", "epoch", "lr", "loss", *r.names, "grad_norm", "seconds"], keep_until=keep)
            r.held_log = _CsvLog(r.run_dir / "heldout_log.csv" if has_held else None, ["step", "total", *r.names],
                                 keep_until=keep)
            if state is None and r.held_initial is not None:
                r.held_log.write({"step": 0, **r.held_initial})
        else:
            r.train_log = _CsvLog(None, [])
            r.held_log = _CsvLog(None, [])
    log(f"{log_prefix}: {len(runs)} students x {total} steps x {B} images (micro {mb}), warm-up {warmup}, "
        f"amp {amp_mode}, data {ldata.mode}, fused AdamW {use_fused}, "
        f"{'host check every step' if sync_each else 'device-side divergence flag'}, "
        f"{len(by_key)} target modes, teacher on {'x and mirror(x)' if flip_all else 'x'}"
        + (f"; resumed from step {start_step}" if state is not None else ""))
    for r in runs:
        log(f"{log_prefix}:   {r.name}: lr {r.cfg.lr:g}, {r.n_trainable:,} trainable, {len(r.names)} terms, "
            f"targets {r.mode.final}/{r.mode.blocks}"
            + (f", held-out {r.held_initial['total']:.5f}" if r.held_initial else ""))

    gen = torch.Generator().manual_seed(int(cfg.seed) + 7919)   # augmentation draws (CPU), shared by all students
    if state is not None:
        gen.set_state(state["gen"])
        _restore_rng_states(state["rng"], device)
    augment = DeviceAugment(device, dtype, cfg.crop_size)
    S = ldata.image_size
    hook_set = {int(s) for s in hook_steps}
    pin = device.type == "cuda"
    hook_time = [0.0]
    t_loop = time.time()

    def elapsed() -> float:
        return t_offset + time.time() - t_loop - hook_time[0]

    def write_row(r: _Run, step: int, term_vals: dict, loss_val: float, gn: float, lr: float, seconds: float):
        r.train_log.write({"step": step, "epoch": round(step / per_epoch, 4), "lr": lr, "loss": loss_val,
                           **term_vals, "grad_norm": gn, "seconds": round(seconds, 3)})

    def flush() -> None:
        """Copy pending losses to the host: history, train_log rows, divergence (device-flag path)."""
        for r in runs:
            if not r.pending:
                continue
            vecs = torch.stack([p.vec for p in r.pending]).cpu()
            gns = torch.stack([p.gn.reshape(()).to(torch.float64) for p in r.pending]).cpu()
            for p, vec, gn in zip(r.pending, vecs, gns):
                term_vals, loss_val = _host_loss(r, vec)
                r.history.append(loss_val)
                if not math.isfinite(loss_val):
                    _mark_diverged(r, p.step - 1, log_prefix)
                    break
                r.steps_done = p.step
                write_row(r, p.step, term_vals, loss_val, float(gn), p.lr, p.seconds)
            r.pending = []

    def run_hooks(step: int) -> None:
        th = time.time()
        if step in hook_set:
            first, last = min(hook_set), max(hook_set | {total})
            for r in runs:
                if not r.live or r.run_dir is None:
                    continue
                ck = r.run_dir / "ckpt" / f"step{int(step)}.pt"
                marker = ck.with_suffix(".queued")
                if marker.exists() and ck.exists():
                    continue                                      # finished before an interruption
                if checkpoints:
                    save_checkpoint(r.model, ck, step, {**(ckpt_meta or {}), **r.s.tags, "student": r.name,
                                                        "lr": r.cfg.lr, "seed": int(cfg.seed)})
                if queue is not None and checkpoints:
                    q = queue.write(queue.request(r.run_dir, step, first, last, ck,
                                                  {**r.s.tags, "student": r.name}))
                    marker.write_text(q.name, encoding="utf-8")
        if step_hook is not None:
            step_hook(int(step), {r.name: r.model for r in runs if r.live})
        hook_time[0] += time.time() - th

    if start_step in hook_set or step_hook is not None:
        run_hooks(start_step)
    batches = ldata.step_batches(B, total, int(cfg.seed), start_step)
    try:
        for step, xb in zip(range(start_step, total), batches):
            live = [r for r in runs if r.live]
            if not live:
                break
            crop = sample_crop_params(B, S, S, cfg.crop_scale, cfg.crop_ratio, cfg.hflip_p, gen)
            crop_dev = crop.pin_memory().to(device, non_blocking=True) if pin else crop.to(device)
            for r in live:
                lr_hook = None
                for c in r.constraints:                       # optional hooks (full SVD-LLM's factor schedule)
                    bs = getattr(c, "begin_step", None)
                    if callable(bs):
                        bs(step, total, r.model)
                    if callable(getattr(c, "lr_at", None)):
                        lr_hook = c.lr_at                     # a student-specific schedule (per-stage warm-up)
                r.lr_now = (lr_hook(step, float(r.cfg.lr), total, warmup, cfg.min_lr_ratio) if lr_hook is not None
                            else lr_at(step, float(r.cfg.lr), total, warmup, cfg.min_lr_ratio))
                for g in r.opt.param_groups:
                    g["lr"] = r.lr_now
                r.opt.zero_grad(set_to_none=True)
                r.acc = {k: torch.zeros((), dtype=torch.float64, device=device) for k in r.names}
            flip, cap, need_logits = pass_needs(live)
            live_groups = _groups(live)
            clip = float(cfg.grad_clip) if cfg.grad_clip else float("inf")   # heal(): inf when not clipping
            for lo in range(0, B, mb):
                x = augment(xb[lo:lo + mb], crop_dev[lo:lo + mb])
                frac = x.shape[0] / B
                with autocast():
                    tp = shared.run(x, flip, cap, need_logits)
                for _, grp in live_groups:
                    with autocast():
                        tf = shared.targets(tp, grp[0].mode, grp[0].capture)
                    for r in grp:
                        with autocast():
                            sf = student_features(teacher_adapter, r.model, x, r.plan, r.mode, logits_fn)
                        terms = distill_terms(sf, tf, r.normalisers, dtype)
                        loss = sum(r.weights[k] * v for k, v in terms.items())
                        (loss * frac).backward()
                        for k, v in terms.items():
                            r.acc[k] += v.detach().double() * frac
                        del sf, terms, loss
                    del tf
                del tp, x
            now = elapsed()
            for r in live:
                vec = torch.stack([r.acc[k] for k in r.names])
                if sync_each:
                    term_vals, loss_val = _host_loss(r, vec.cpu())
                    r.history.append(loss_val)
                    if not math.isfinite(loss_val):
                        _mark_diverged(r, step, log_prefix)
                        continue
                    gn = torch.nn.utils.clip_grad_norm_(r.params, clip)
                    r.opt.step()
                    with torch.no_grad():
                        for c in r.constraints:
                            c(r.model)
                    r.steps_done = step + 1
                    write_row(r, step + 1, term_vals, loss_val, float(gn), r.lr_now, now)
                else:
                    bad = ~torch.isfinite(r.step_total())
                    r.dead = r.dead | bad
                    gn = torch.nn.utils.clip_grad_norm_(r.params, clip)
                    r.opt.found_inf = r.dead.to(torch.float32)
                    r.opt.step()
                    with torch.no_grad():
                        for c in r.constraints:
                            c(r.model)
                    r.pending.append(_Pending(step + 1, vec, gn.detach(), r.lr_now, now))
                r.acc = {}
            steps_done = step + 1
            is_log = bool(cfg.log_every) and steps_done % int(cfg.log_every) == 0
            is_eval = bool(cfg.eval_every) and has_held and steps_done % int(cfg.eval_every) == 0 and steps_done < total
            is_state = (state_path is not None and state_every and steps_done % int(state_every) == 0
                        and steps_done < total)
            if is_log or is_eval or is_state or steps_done in hook_set or steps_done == total:
                flush()
            if is_eval:
                for name, res in held_eval([r for r in runs if r.live]).items():
                    run = next(r for r in runs if r.name == name)
                    run.held_rows_log.append({"step": steps_done, **res})
                    run.held_log.write({"step": steps_done, **res})
            if is_log:
                el = elapsed() - t_offset
                n_run = steps_done - start_step
                log(f"{log_prefix}: step {steps_done}/{total} ({el / n_run:.2f} s/step, ETA "
                    f"{(total - steps_done) * el / n_run / 60:.1f} min): "
                    + "; ".join(f"{r.name} {r.history[-1]:.5f}" for r in runs if r.live and r.history))
            if is_state:
                _save_state(state_path, identity, steps_done, runs, gen, device, norms, elapsed())
            if steps_done in hook_set or step_hook is not None:
                run_hooks(steps_done)
    finally:
        for r in runs:
            r.train_log.close()
    flush()

    live = [r for r in runs if r.live]
    for name, res in held_eval(live).items():
        run = next(r for r in runs if r.name == name)
        run.held_final = res
        run.held_rows_log.append({"step": run.steps_done, **res})
        run.held_log.write({"step": run.steps_done, **res})
    for r in runs:
        r.held_log.close()
    peak = (torch.cuda.max_memory_allocated(device) / 2 ** 30) if device.type == "cuda" else None
    gi = git_info()
    results = {}
    for r in runs:
        checks = {}
        for i, c in enumerate(r.constraints):
            fn = getattr(c, "check", None)
            if callable(fn):
                key = f"{i}:{_describe(c)['type']}"
                try:
                    checks[key] = fn(r.model)
                except Exception as e:        # e.g. a rank check on a diverged student's NaN weights
                    checks[key] = {"error": f"{type(e).__name__}: {e}"}
                    log(f"{log_prefix}: {r.name}: constraint check {key} failed ({type(e).__name__}: {e})")
        ckpt = None
        if final_checkpoint and checkpoints and r.run_dir is not None and r.live:
            ckpt = r.run_dir / "ckpt" / f"step{r.steps_done}.pt"
            if not ckpt.exists():
                save_checkpoint(r.model, ckpt, r.steps_done, {**(ckpt_meta or {}), **r.s.tags, "student": r.name,
                                                              "lr": r.cfg.lr, "seed": int(cfg.seed)})
        finite = [h for h in r.history if math.isfinite(h)]
        m = {"config": asdict(r.cfg), "seed": int(cfg.seed), **gi, "env": env_info(device), "student": r.name,
             "tags": dict(r.s.tags), "target_mode": r.mode.describe(), "target_key": repr(r.key),
             "constraints": [_describe(c) for c in r.constraints], "constraint_checks": checks,
             "n_trainable": r.n_trainable, "n_trainable_tensors": len(r.trainable),
             "trainable_names": [n for n, _ in r.trainable], "amp": amp_mode, "share_prefix_used": r.plan.share,
             "first_grad_block": r.plan.first_grad, "normalisers": r.normalisers, "weights": r.weights,
             "total_steps": total, "steps_done": r.steps_done, "warmup_steps": warmup, "steps_per_epoch": per_epoch,
             "images_seen": r.steps_done * B, "diverged": r.diverged,
             "heldout_images": int(len(held_rows)) if has_held else 0,
             "heldout_initial": r.held_initial, "heldout_final": r.held_final,
             "train_loss_first": r.history[0] if r.history else None,
             "train_loss_last": r.history[-1] if r.history else None,
             "train_loss_last10_mean": float(np.mean(finite[-10:])) if finite else None,
             "seconds_total": round(time.time() - t0, 2), "seconds_train": round(elapsed(), 2),
             "peak_memory_gb": peak, "persistent_memory_gb": round(_tensor_bytes(r) / 2 ** 30, 4),
             "checkpoint": str(ckpt) if ckpt else None, "lockstep": {
                 "students": [x.name for x in runs], "data": ldata.describe(), "fused_adamw": use_fused,
                 "divergence_check": "host, every step" if sync_each else "device flag (found_inf), at flushes",
                 "teacher_flip": flip_all, "n_target_modes": len(by_key)},
             "resumed_from_step": start_step if state is not None else None,
             "hook_steps": sorted(hook_set), "hook_seconds": round(hook_time[0], 2),
             "state_path": str(state_path) if state_path is not None and state_every else None,
             "state_every": int(state_every or 0)}
        if r.run_dir is not None:
            _atomic_json(r.run_dir / "metrics.json", m)
            _atomic_json(r.run_dir / "heldout.json", {"initial": r.held_initial, "final": r.held_final,
                                                     "by_step": r.held_rows_log,
                                                     "rows": int(len(held_rows)) if has_held else 0,
                                                     "normalisers": r.normalisers, "weights": r.weights})
        results[r.name] = {**m, "history": list(r.history)}
    job = {"total_steps": total, "students": {r.name: {"lr": r.cfg.lr, "steps_done": r.steps_done,
                                                       "diverged": r.diverged,
                                                       "heldout_initial": (r.held_initial or {}).get("total"),
                                                       "heldout_final": (r.held_final or {}).get("total"),
                                                       "train_loss_last": r.history[-1] if r.history else None,
                                                       "n_trainable": r.n_trainable} for r in runs},
           "seconds_total": round(time.time() - t0, 2), "seconds_train": round(elapsed(), 2), "peak_memory_gb": peak,
           "data": ldata.describe(), "amp": amp_mode, "fused_adamw": use_fused, **gi}
    if jdir is not None:
        _atomic_json(jdir / "summary.json", job)
    log(f"{log_prefix}: done {total} steps in {job['seconds_train']:.1f} s; "
        + "; ".join(f"{r.name}: " + ("DIVERGED" if r.diverged else
                                       f"held-out {(r.held_initial or {}).get('total', float('nan')):.5f} -> "
                                       f"{(r.held_final or {}).get('total', float('nan')):.5f}") for r in runs))
    return {"students": results, "job": job}


def _mark_diverged(r: _Run, step: int, prefix: str) -> None:
    """heal()'s divergence: the loss of optimiser step `step` (0-based) is non-finite; no update was applied for it
    and none later; steps_done counts the completed steps before it."""
    r.diverged = True
    r.steps_done = int(step)
    r.pending = []
    r.opt = None
    for p in r.model.parameters():
        p.grad = None
    log(f"{prefix}: {r.name}: non-finite loss at step {step + 1}; it leaves the lockstep")


def _save_state(path, identity, step, runs, gen, device, norms, seconds) -> None:
    _save_train_state(path, {
        "format": LOCKSTEP_STATE_FORMAT, "identity": _jsonable(identity), "step": int(step),
        "students": [{"name": r.name, "model": r.model.state_dict(),
                      "optimizer": None if r.opt is None else r.opt.state_dict(), "history": list(r.history),
                      "held_initial": r.held_initial, "held_rows_log": list(r.held_rows_log),
                      "diverged": r.diverged, "steps_done": r.steps_done} for r in runs],
        "gen": gen.get_state(), "rng": _rng_states(device), "normalisers": norms, "seconds_train": seconds})


@torch.no_grad()
def _calibrate(shared: SharedTeacher, ldata: LockstepData, rows, cfg: HealConfig, by_key, needs, dtype,
               autocast) -> dict:
    """Fixed normalisers per target mode: mirror.heal._teacher_mean_squares over the calibration rows, with one
    teacher pass per chunk for every mode (sums outside autocast, as there)."""
    flip, cap, need_logits = needs
    sums = {repr(key): {} for key, _ in by_key}
    counts = {repr(key): {} for key, _ in by_key}
    for xb in ldata.chunks(rows, int(cfg.eval_batch or cfg.micro_batch), "train"):
        x = centre_view(xb, cfg.crop_size, dtype)
        with autocast():
            tp = shared.run(x, flip, cap, need_logits)
            feats_by_key = {repr(key): shared.targets(tp, grp[0].mode, grp[0].capture) for key, grp in by_key}
        for key, feats in feats_by_key.items():
            for k, v in feats.items():
                sums[key][k] = sums[key].get(k, 0.0) + float((v.double() ** 2).sum())
                counts[key][k] = counts[key].get(k, 0) + v.numel()
        del tp, feats_by_key
    return {key: {k: sums[key][k] / counts[key][k] for k in sums[key]} for key in sums}


@torch.no_grad()
def _heldout(shared: SharedTeacher, ldata: LockstepData, rows, cfg: HealConfig, runs, needs, dtype, autocast,
             logits_fn) -> dict:
    """mirror.heal._heldout for every run, with one teacher pass per chunk: {name: {"total", terms...}}."""
    flip, cap, need_logits = needs
    sse = {r.name: {} for r in runs}
    tsq = {r.name: {} for r in runs}
    cnt = {r.name: {} for r in runs}
    for xb in ldata.chunks(rows, int(cfg.eval_batch or cfg.micro_batch), "heldout"):
        x = centre_view(xb, cfg.crop_size, dtype)
        with autocast():
            tp = shared.run(x, flip, cap, need_logits)
        for _, grp in _groups(runs):
            with autocast():
                tf = shared.targets(tp, grp[0].mode, grp[0].capture)
            for r in grp:
                with autocast():
                    sf = student_features(shared.adapter, r.model, x, r.plan, r.mode, logits_fn)
                for k, t in tf.items():
                    t = t.double()
                    sse[r.name][k] = sse[r.name].get(k, 0.0) + float(((sf[k].double() - t) ** 2).sum())
                    tsq[r.name][k] = tsq[r.name].get(k, 0.0) + float((t ** 2).sum())
                    cnt[r.name][k] = cnt[r.name].get(k, 0) + t.numel()
                del sf
            del tf
        del tp, x
    out = {}
    for r in runs:
        n = r.normalisers
        terms = {k: (sse[r.name][k] / cnt[r.name][k]) / (n[k] if n is not None else tsq[r.name][k] / cnt[r.name][k])
                 for k in sse[r.name]}
        res = {"total": float(sum(r.weights[k] * v for k, v in terms.items()))}
        res.update(terms)
        out[r.name] = res
    return out


# ----------------------------------------------------------------------------------------------
# Learning-rate selection
# ----------------------------------------------------------------------------------------------
def choose_lr(trials: Sequence[dict]) -> float:
    """Lowest held-out loss wins; ties go to the smaller rate; diverged or non-finite trials score +inf."""
    if not trials:
        raise ValueError("no trials")

    def score(t):
        v = t.get("heldout_loss")
        v = float("inf") if (v is None or t.get("diverged") or not math.isfinite(float(v))) else float(v)
        return (v, float(t["lr"]))

    best = min(trials, key=score)
    if not math.isfinite(score(best)[0]):
        raise RuntimeError("every learning rate diverged")
    return float(best["lr"])


def select_lr_lockstep(teacher_adapter, candidates: dict, data, cfg, *, budget_steps: int = 200, job_dir=None,
                       resume: bool = False, state_every: int = 0, **kw) -> dict:
    """Short trials of every (method, rate) pair as students of one lockstep job (module docstring).

    candidates: {method_key: {"lrs": [...], "make": callable() -> dict(model, constraints, mode, blocks=None,
    train_scope="all", compile=False), "out_dir": Path or None, "trial_dir": callable(lr) -> Path (optional),
    "tags": dict}}. Every trial uses the same seed, data order and augmentation (shared), its own warm-up and cosine
    over `budget_steps`, no checkpoints, no evaluation hooks; its logs go to trial_dir(lr) (default
    <out_dir>/lr_<rate>/). Writes <out_dir>/lr_selection.json per method and returns {method_key: selection}. Extra
    keyword arguments go to train_lockstep (logits_fn, data_mode, fused, ...)."""
    cfg = as_config(cfg)
    tcfg = replace(cfg, max_steps=int(budget_steps), eval_every=0, save_checkpoint=False, checkpoint_path=None)
    students, index, trial_dirs = [], {}, {}
    for key, c in candidates.items():
        lrs = sorted({float(v) for v in c["lrs"]})
        if not lrs:
            raise ValueError(f"{key}: no learning rates")
        out_dir = Path(c["out_dir"]) if c.get("out_dir") is not None else None
        tdir = c.get("trial_dir") or ((lambda r, o=out_dir: o / f"lr_{r:.0e}") if out_dir is not None else None)
        for lr in lrs:
            built = c["make"]()
            name = f"{key}@lr{lr:.0e}"
            trial_dirs[name] = None if tdir is None else str(Path(tdir(lr)))
            students.append(LockstepStudent(
                name=name, model=built["model"], constraints=built["constraints"], mode=built["mode"], lr=lr,
                blocks=built.get("blocks"), run_dir=Path(tdir(lr)) if tdir is not None else None,
                train_scope=built.get("train_scope", "all"), compile=bool(built.get("compile", False)),
                tags={**(c.get("tags") or {}), "lr_trial": lr}))
            index[name] = (key, lr)
    res = train_lockstep(teacher_adapter, students, data, tcfg, job_dir=job_dir, hook_steps=(), queue=None,
                         checkpoints=False, final_checkpoint=False, state_every=state_every, resume=resume,
                         log_prefix="lr-selection", **kw)
    del students
    gc.collect()
    out = {}
    gi = git_info()
    for key, c in candidates.items():
        trials = []
        for name, (k2, lr) in index.items():
            if k2 != key:
                continue
            m = res["students"][name]
            hf = m["heldout_final"]
            score = float(hf["total"]) if (hf is not None and not m["diverged"]) else float("inf")
            if not math.isfinite(score):
                score = float("inf")
            trials.append({"lr": lr, "heldout_loss": score, "heldout_terms": hf,
                           "heldout_initial": (m["heldout_initial"] or {}).get("total"),
                           "train_loss_last": m["train_loss_last"], "diverged": m["diverged"],
                           "trial_dir": trial_dirs.get(name)})
        trials.sort(key=lambda t: t["lr"])
        sel = {"best_lr": choose_lr(trials), "lrs": [t["lr"] for t in trials], "budget_steps": int(budget_steps),
               "seed": int(cfg.seed), "trials": trials, "lockstep": True,
               "rule": "lowest held-out distillation loss on the LR-selection rows (centre view) after the trial "
                       "budget, each trial with its own warm-up and cosine; ties go to the smaller rate; diverged "
                       "trials score +inf",
               "heldout_images": int(res["students"][next(iter(index))]["heldout_images"]), **gi}
        if c.get("out_dir") is not None:
            _atomic_json(Path(c["out_dir"]) / "lr_selection.json", sel)
        log("lr selection " + key + ": " + ", ".join(f"{t['lr']:.0e} -> {t['heldout_loss']:.5f}" for t in trials)
            + f"; best {sel['best_lr']:.0e}")
        out[key] = sel
    return out
