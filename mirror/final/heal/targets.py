"""Shared teacher pass and per-student distillation targets for lockstep healing (docs/final_phase_design.md,
Sections 4.1 and 4.3).

One teacher forward per micro-batch serves every student of a lockstep job (mirror/final/heal/multistudent.py):
the original model runs on [x; mirror(x)] when any student needs flip-symmetrised targets (otherwise on x),
keeping the residual stream after every block that some student matches, the final-norm output and, when a
student asks for them, the logits. Each student's targets are then derived from these tensors exactly as
mirror.heal_vexact.VExactTeacher.distill_targets derives them: the same `stream_target` and `final_targets`
methods are called on the same tensors, so a student trained here sees bit for bit the targets that
mirror.heal.heal() gives it with the VExactTeacher that mirror.heal_structured.distill_teacher builds for it
(MUST-09). The only change is in `_TargetHelper.align`: the register pairing is gathered with an index tensor
cached on the device instead of being copied from the host at every call (the same gather, so the same values,
without a host synchronisation per call).

TargetMode (one per student):
  final      "symmetrised": the flip average of the teacher's final-norm tokens, (f(x) + align(f(mirror x))) / 2
             (VExactTeacher.final_targets at full span); "raw": the teacher's own final-norm tokens on x.
             Unlike VExactTeacher, this does not depend on the span size k: the partial-span student (A06) uses
             flip-averaged final targets with symmetrised block targets only for blocks below k.
  blocks     "symmetrised": the centred teacher stream symmetrised with the student's operator A,
             (c(x) + align(c(mirror x)) A) / 2, for teacher blocks i < k and only centred for i >= k
             (VExactTeacher.stream_target); "centred": the centred stream (a LayerNorm-folded student, SliceGPT);
             "raw": the teacher's own block outputs (an unfolded student: SVD, SVD-LLM, the original).
  A, reg_perm, k
             the student's stream operator (symmetrised blocks only), register pairing (used by every flip
             alignment; None = no pairing) and span size (None = every block).
  block_map  student block j is matched to teacher block block_map[j] (block-dropped students); None = identity.
  logits     add a "logits" term: normalised MSE between the student's logits and the flip-averaged teacher
             logits (weight 1, fixed normaliser like every other term). The logits come from a callable
             logits_fn(model, final_tokens) applied to the final-norm tokens of the same forward pass (default
             `default_logits`: timm's forward_head, else model.head on the CLS token), so no extra forward runs.
             A function with the signature (adapter, x, model), such as mirror.final.adapters.deit_logits, runs a
             forward of its own and is not used here; wrap the head part instead (see `logits_from_head`).
  input_transform
             "rot90" (A15b cause probe): the student sees torch.rot90(x, 1, (2, 3)); its targets are the
             teacher's raw outputs on x, and the student's patch grid is rotated back (torch.rot90(grid, -1))
             before the comparison. No flip averaging for this student (final "raw", blocks "raw" or "centred").

Term order. Target dictionaries list the block terms first (ascending student block), then "cls", "patch" and
"logits", the order VExactTeacher.distill_targets uses. mirror.heal sums the terms in this order, so keeping it
keeps the loss bit for bit.

Sources: mirror/heal_vexact.py (VExactTeacher), mirror/heal_structured.py (distill_teacher), mirror/heal.py
(_student_forward), mirror/m1.py (align_mirror_tokens), mirror/stats.py (unflip_tokens); timm 1.0.31
VisionTransformer.forward_head (installed package, read locally); torch.rot90 documentation
(https://pytorch.org/docs/stable/generated/torch.rot90.html).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

from mirror.data import mirror
from mirror.heal import _student_forward
from mirror.heal_vexact import VExactTeacher, _float_dtype, _no_autocast
from mirror.m1 import register_info
from mirror.stats import unflip_tokens

FINAL_MODES = ("symmetrised", "raw")
BLOCK_MODES = ("symmetrised", "centred", "raw")
INPUT_TRANSFORMS = (None, "rot90")

__all__ = ["FINAL_MODES", "BLOCK_MODES", "INPUT_TRANSFORMS", "TargetMode", "TeacherPass", "SharedTeacher",
           "default_logits", "logits_from_head", "student_features", "rotate_back_tokens", "array_digest"]


def array_digest(a) -> str:
    """Short content hash of an array (float64 or int64 view), used to group students with equal targets."""
    a = np.ascontiguousarray(np.asarray(a))
    a = a.astype(np.float64) if a.dtype.kind == "f" else a.astype(np.int64)
    return hashlib.sha1(a.tobytes() + str(a.shape).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------
# Target mode
# ----------------------------------------------------------------------------------------------
@dataclass
class TargetMode:
    """How one student's distillation targets are derived from the shared teacher pass (module docstring)."""

    final: str = "symmetrised"
    blocks: str = "raw"
    A: np.ndarray | None = None
    reg_perm: np.ndarray | None = None
    k: int | None = None
    block_map: dict | None = None
    logits: bool = False
    input_transform: str | None = None

    def __post_init__(self):
        if self.final not in FINAL_MODES:
            raise ValueError(f"final must be one of {FINAL_MODES}, not {self.final!r}")
        if self.blocks not in BLOCK_MODES:
            raise ValueError(f"blocks must be one of {BLOCK_MODES}, not {self.blocks!r}")
        if self.input_transform not in INPUT_TRANSFORMS:
            raise ValueError(f"input_transform must be one of {INPUT_TRANSFORMS}, not {self.input_transform!r}")
        if self.blocks == "symmetrised" and self.A is None:
            raise ValueError('blocks "symmetrised" needs the student\'s operator A')
        if self.A is not None:
            self.A = np.asarray(self.A, dtype=np.float64)
        if self.reg_perm is not None:
            self.reg_perm = np.asarray(self.reg_perm, dtype=np.int64)
        if self.k is not None:
            self.k = int(self.k)
            if self.k < 1:
                raise ValueError("k must be at least 1")
        if self.block_map is not None:
            self.block_map = {int(j): int(i) for j, i in dict(self.block_map).items()}
            if any(j < 0 or i < 0 for j, i in self.block_map.items()):
                raise ValueError("block_map indices must be non-negative")
        self.logits = bool(self.logits)
        if self.input_transform == "rot90" and (self.final != "raw" or self.blocks == "symmetrised" or self.logits):
            raise ValueError('input_transform "rot90" takes raw targets (final "raw", blocks "raw" or "centred", '
                             "no logits): the probe has no flip averaging")

    @property
    def needs_flip(self) -> bool:
        """The teacher must also see the mirrored images (flip-averaged final targets or logits, symmetrised
        block targets)."""
        return self.final == "symmetrised" or self.blocks == "symmetrised" or self.logits

    def resolved_k(self, depth: int) -> int:
        return int(depth) if self.k is None else int(self.k)

    def teacher_block(self, j: int) -> int:
        """Teacher block matched by student block j."""
        if self.block_map is None:
            return int(j)
        if int(j) not in self.block_map:
            raise KeyError(f"block_map has no entry for student block {j}")
        return self.block_map[int(j)]

    def reg_perm_for(self, n_registers: int) -> np.ndarray:
        """The register pairing as VExactTeacher uses it (None: the identity on the model's registers)."""
        if self.reg_perm is None or len(self.reg_perm) == 0:
            return np.arange(int(n_registers), dtype=np.int64)
        return self.reg_perm

    def key(self, capture: Sequence[int], depth: int, n_registers: int) -> tuple:
        """Students with equal keys get identical targets (and normalisers) from the same teacher pass."""
        uses_A = self.blocks == "symmetrised"
        uses_reg = self.final == "symmetrised" or uses_A
        return (self.final, self.blocks,
                array_digest(self.A) if uses_A else None,
                array_digest(self.reg_perm_for(n_registers)) if uses_reg else None,
                self.resolved_k(depth) if uses_A else None,
                tuple(sorted(self.block_map.items())) if self.block_map is not None else None,
                self.logits, tuple(int(j) for j in capture))

    def describe(self) -> dict:
        """JSON description (the operator by digest and d+, not in full)."""
        out = {"final": self.final, "blocks": self.blocks, "k": self.k, "logits": self.logits,
               "input_transform": self.input_transform,
               "block_map": None if self.block_map is None else {str(j): i for j, i in sorted(self.block_map.items())},
               "reg_perm": None if self.reg_perm is None else [int(v) for v in self.reg_perm]}
        if self.A is not None:
            tr = float(np.trace(self.A))
            out.update(A_digest=array_digest(self.A), A_dim=int(self.A.shape[0]),
                       A_d_plus=int(round((tr + self.A.shape[0]) / 2)))
        return out


# ----------------------------------------------------------------------------------------------
# Logits
# ----------------------------------------------------------------------------------------------
def default_logits(model: nn.Module, final_tokens: torch.Tensor) -> torch.Tensor:
    """Logits from the final-norm tokens of the same forward: timm's forward_head (pooling, fc_norm, head), else
    model.head on the CLS token."""
    fh = getattr(model, "forward_head", None)
    if callable(fh):
        return fh(final_tokens)
    head = getattr(model, "head", None)
    if head is None:
        raise ValueError("the model has neither forward_head nor head: pass a logits_fn")
    return head(final_tokens[:, 0])


def logits_from_head(head_fn: Callable) -> Callable:
    """Adapt a function head_fn(model, cls_token) -> logits (for example the even-projected DeiT-III head of
    mirror.final.adapters, if it exposes one) to the logits_fn(model, final_tokens) signature used here."""
    def fn(model, final_tokens):
        return head_fn(model, final_tokens[:, 0])
    return fn


# ----------------------------------------------------------------------------------------------
# Teacher pass and target derivation
# ----------------------------------------------------------------------------------------------
class _TargetHelper(VExactTeacher):
    """VExactTeacher at full span (even head, flip-averaged final target) for one (A, register pairing); its
    stream_target and final_targets derive the targets. `align` is the same gather as
    mirror.m1.align_mirror_tokens with the register index cached on the device."""

    def __init__(self, base, A, reg_perm):
        super().__init__(base, A, int(base.depth), reg_perm, symmetrise=True, final_target="symmetrised")
        self._idx: dict = {}

    def align(self, t: torch.Tensor) -> torch.Tensor:
        u = unflip_tokens(t, self.base.grid_hw, self.base.n_prefix_tokens)
        R = len(self.reg_perm)
        if not R:
            return u
        key = (str(u.device), int(u.shape[1]))
        idx = self._idx.get(key)
        if idx is None:
            idx = torch.arange(u.shape[1])
            idx[self.reg_start:self.reg_start + R] = self.reg_start + torch.as_tensor(np.asarray(self.reg_perm))
            idx = idx.to(u.device)
            self._idx[key] = idx
        return u[:, idx]


@dataclass
class TeacherPass:
    """Tensors of one teacher forward: B images (and their mirrors after them when `flipped`)."""

    B: int
    flipped: bool
    streams: dict                 # teacher block index -> residual stream after that block
    last: torch.Tensor            # stream after the last block
    final: torch.Tensor           # final-norm output
    logits: torch.Tensor | None = None


class SharedTeacher:
    """The original model's adapter, run once per micro-batch for every student (module docstring)."""

    def __init__(self, adapter, logits_fn: Callable | None = None):
        self.adapter = adapter
        self.model = adapter.model
        self.depth = int(adapter.depth)
        self.n_prefix = int(adapter.n_prefix_tokens)
        self.logits_fn = logits_fn or default_logits
        reg = register_info(self.model, adapter)
        self.n_registers = int(reg["R"])
        self._helpers: dict = {}

    def helper(self, mode: TargetMode) -> _TargetHelper:
        A = mode.A if mode.blocks == "symmetrised" else None
        reg = mode.reg_perm_for(self.n_registers)
        key = (array_digest(A) if A is not None else "identity", array_digest(reg))
        if key not in self._helpers:
            d = int(self.adapter.embed_dim)
            self._helpers[key] = _TargetHelper(self.adapter, np.eye(d) if A is None else A, reg)
        return self._helpers[key]

    def key(self, mode: TargetMode, capture: Sequence[int]) -> tuple:
        return mode.key(capture, self.depth, self.n_registers)

    @torch.no_grad()
    def run(self, x: torch.Tensor, flip: bool, capture: Iterable[int], logits: bool = False) -> TeacherPass:
        """The teacher on [x; mirror(x)] (flip) or x, as VExactTeacher.distill_targets runs it, keeping the streams
        after the blocks in `capture`, the last stream, the final-norm output and (logits) the logits."""
        m = self.model
        cap = {int(i) for i in capture}
        B = int(x.shape[0])
        t = self.adapter.layer0_tokens(torch.cat([x, mirror(x)], dim=0) if flip else x, m)
        streams = {}
        for i, blk in enumerate(m.blocks):
            t = blk(t)
            if i in cap:
                streams[i] = t
        final = m.norm(t)
        lg = self.logits_fn(m, final) if logits else None
        return TeacherPass(B=B, flipped=bool(flip), streams=streams, last=t, final=final, logits=lg)

    @torch.no_grad()
    def targets(self, tp: TeacherPass, mode: TargetMode, capture: Sequence[int]) -> dict:
        """The student's targets from one teacher pass, in VExactTeacher.distill_targets' order and values."""
        if mode.needs_flip and not tp.flipped:
            raise ValueError("this student's targets need the mirrored images in the teacher pass")
        h = self.helper(mode)
        B = tp.B
        k = mode.resolved_k(self.depth)
        feats = {}
        for j in sorted(int(c) for c in capture):
            i = mode.teacher_block(j)
            if i not in tp.streams:
                raise KeyError(f"teacher block {i} was not captured")
            t = tp.streams[i]
            if mode.blocks == "raw":
                with _no_autocast(t):
                    feats[f"block{j}"] = t[:B].to(_float_dtype(t))
            else:
                feats[f"block{j}"] = h.stream_target(t, B, symmetrise=mode.blocks == "symmetrised" and i < k)
        if mode.final == "symmetrised":
            f = h.final_targets(tp.last, tp.final, B)
        else:
            with _no_autocast(tp.final):
                f = tp.final[:B].to(_float_dtype(tp.final))
        feats["cls"] = f[:, 0]
        feats["patch"] = f[:, self.n_prefix:]
        if mode.logits:
            if tp.logits is None:
                raise ValueError("the teacher pass has no logits")
            lg = tp.logits
            with _no_autocast(lg):
                lv = lg.to(_float_dtype(lg))
                feats["logits"] = 0.5 * (lv[:B] + lv[B:]) if tp.flipped else lv[:B]
        return feats


# ----------------------------------------------------------------------------------------------
# Student side
# ----------------------------------------------------------------------------------------------
def rotate_back_tokens(t: torch.Tensor, grid_hw, n_prefix: int) -> torch.Tensor:
    """Tokens [B, n_prefix + H W, C] of a rot90(x, 1, (2, 3)) image put back on the grid of x: the patch grid is
    rotated with torch.rot90(grid, -1, (1, 2)); prefix tokens (CLS, registers) are unchanged. With n_prefix = 0
    the input holds patch tokens only."""
    H, W = (int(v) for v in grid_hw)
    if H != W:
        raise ValueError("rot90 needs a square patch grid")
    B, T, C = t.shape
    if T != n_prefix + H * W:
        raise ValueError(f"expected {n_prefix + H * W} tokens, got {T}")
    p = t[:, n_prefix:].reshape(B, H, W, C)
    p = torch.rot90(p, -1, (1, 2)).reshape(B, H * W, C)
    return torch.cat([t[:, :n_prefix], p], dim=1) if n_prefix else p


def student_features(adapter, model: nn.Module, x: torch.Tensor, plan, mode: TargetMode,
                     logits_fn: Callable | None = None) -> dict:
    """mirror.heal._student_forward with the mode's options: rot90 input (patch grids rotated back) and logits
    (from the final-norm output of the same forward, captured with a hook on model.norm)."""
    rot = mode.input_transform == "rot90"
    xs = torch.rot90(x, 1, (2, 3)) if rot else x
    box = {}
    hook = None
    if mode.logits:
        hook = model.norm.register_forward_hook(lambda mod, inp, out: box.__setitem__("final", out))
    try:
        sf = _student_forward(adapter, model, xs, plan, None)
    finally:
        if hook is not None:
            hook.remove()
    if rot:
        n_prefix = int(plan.n_prefix)
        out = {}
        for key, v in sf.items():
            if key == "cls":
                out[key] = v
            elif key == "patch":
                out[key] = rotate_back_tokens(v, adapter.grid_hw, 0)
            else:
                out[key] = rotate_back_tokens(v, adapter.grid_hw, n_prefix)
        sf = out
    if mode.logits:
        sf["logits"] = (logits_fn or default_logits)(model, box["final"])
    return sf
