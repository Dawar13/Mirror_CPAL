"""Structured trainable students for MUST-09 (PROJECT_NEXT_PHASE.md, Sections 4.8 and 5): one healing recipe for
every compressor, each student trained only in its own structured parameters.

make_trainable(model, info, adapter) turns a compressor's output into a model whose gradient steps keep the
structure exactly, plus the constraints for mirror/heal.py and a JSON spec from which load_healed_model rebuilds
the healed model. `info` follows the compressor contract (mirror/baselines.py compress, written separately):

    info["per_matrix"]: one dict per compressed matrix, {"block", "name" in fc1|fc2|wq|wk|wv|wo,
                        "kind" in lowrank|width|slice|merge, "params", and per kind:
                        lowrank: "rank" (optional "U_key", "V_key": given factors in info["tensors"]);
                        width:   "kept" (surviving hidden indices; optional "axis" in|out);
                        slice:   "d_in", "d_out", "basis_key" (or "basis_in_key" / "basis_out_key");
                        merge:   "groups" (lists of original hidden indices, one list per merged unit)}
    info["tensors"]:    bases, factors and other arrays named by the entries
    info["params_total"], info["budget_total"], info["gap"]: the compressor's own accounting (recorded only)

For the mirror family (Mirror-exact and its random-structure control) `info` has kind "mirror" and the span
structure: A_star, perms and maps_after (aligned with blocks; the head maps valid after surgery) and reg_perm.

What trains (matching E3: mirror.heal_vexact.SpanTrainable and VExactEmbeddings, so the comparison is like for
like): the structured parameters of the converted blocks (and their biases) and the embeddings (patch embedding,
position grid, CLS token, registers). Absorbed norms and LayerScale of the converted blocks stay frozen, as do the
final norm affine, the mask token and anything outside the converted blocks. The trainer runs with train_scope
"all". Mirror-exact re-projects its weights and embeddings after every step (mirror.heal.MirrorMLPConstraint,
mirror.heal_attn.MirrorAttnConstraint, mirror.heal_vexact.VExactEmbeddings); the baselines train freely inside
their structure:

    lowrank  LowRankLinear factors U [out, r], V [r, in]: the rank can never exceed r. Balanced factors of the
             rank-r SVD of the compressed weight (as the E3 SVD arm), or the given factors.
    width    the surviving rows or columns as a smaller dense matrix. An MLP whose fc1 rows and fc2 columns keep
             the same units becomes two smaller nn.Linear layers; any other matrix becomes an IndexedLinear (the
             kept outputs are written into a zero vector at fixed indices, or the kept inputs are gathered).
    slice    SlicedLinear y = Q_out (W_s (Q_in^T x) + b_s): W_s and b_s train, the bases are fixed buffers.
    merge    the merged MLP units as two smaller nn.Linear layers (fc1 rows of a group are tied, so the group's
             fc2 columns add up). Merging is supported for the MLP only.

Every conversion is checked: the relative error of each matrix (and bias) against the compressor's dense weights
is recorded in the spec and must stay below `tol`; the scripts also compare the features of the compressed model
and of the trainable student.

load_healed_model(run_dir, device) -> (adapter, model, info) rebuilds a healed student from <run_dir>/spec.json,
its state file and, for the mirror family, structure.npz. Downstream scripts (MUST-10, MUST-11) use it.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import HeadMaps
from .heal import LowRankLinear, MirrorMLPConstraint, _absorbed_parameters, exact_involution
from .heal_attn import LowRankQKV, MirrorAttnConstraint, attn_absorbed_parameters
from .heal_vexact import VExactEmbeddings, VExactTeacher, span_frozen_parameters, vexact_block_params

SPEC_FORMAT = "must09-healed-v1"
KINDS = ("lowrank", "width", "slice", "merge")
MATRIX_NAMES = ("fc1", "fc2", "wq", "wk", "wv", "wo")
MLP_NAMES = ("fc1", "fc2")
QKV_NAMES = ("wq", "wk", "wv")
SLOT_PATHS = ("mlp.fc1", "mlp.fc2", "attn.qkv", "attn.proj")
DEFAULT_TOL = 1e-4

__all__ = ["SPEC_FORMAT", "KINDS", "IndexedLinear", "SlicedLinear", "SplitQKV", "StructuredTrainable",
           "make_trainable", "module_spec", "module_from_spec", "module_macs", "dense_weight", "build_skeleton",
           "verify_rebuild", "save_structure", "load_structure", "write_spec", "load_healed_model", "adapter_spec",
           "build_adapter_from_spec", "distill_teacher", "block_costs", "stored_params", "feature_gap",
           "is_folded_model"]


# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------
def _f64(t) -> torch.Tensor:
    if isinstance(t, np.ndarray):
        return torch.from_numpy(np.asarray(t, dtype=np.float64))
    return t.detach().to(torch.float64)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """||a - b|| / ||b|| in float64 (0 when both are zero)."""
    a, b = a.detach().double(), b.detach().double()
    den = float(b.norm())
    num = float((a - b.to(a.device)).norm())
    return num / den if den > 0 else num


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha1(t.detach().to(torch.float64).cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, torch.Tensor):
        return _jsonable(o.detach().cpu().tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return repr(o)


def _linear(W: torch.Tensor, b: torch.Tensor | None, device, dtype) -> nn.Linear:
    lin = nn.Linear(W.shape[1], W.shape[0], bias=b is not None, device=device, dtype=dtype)
    with torch.no_grad():
        lin.weight.copy_(W)
        if b is not None:
            lin.bias.copy_(b)
    return lin


def is_folded_model(model: nn.Module) -> bool:
    """True if the block norms are mirror.fold.RMSNorm (the LayerNorm-folded, centred stream)."""
    from .fold import RMSNorm as FoldRMSNorm

    return isinstance(model.blocks[0].norm1, FoldRMSNorm)


# ----------------------------------------------------------------------------------------------
# Structured modules
# ----------------------------------------------------------------------------------------------
class IndexedLinear(nn.Module):
    """A width-pruned matrix inside a layer of fixed size; only the surviving part is stored and trains.

    axis "out": W [n, in], b [n]; y = zeros(out_features) with y[..., index] = x W^T + b (the other outputs stay 0).
    axis "in":  W [out, n], b [out]; y = x[..., index] W^T + b (the other inputs are ignored).
    `index` is a fixed buffer (sorted surviving indices)."""

    def __init__(self, in_features: int, out_features: int, axis: str, n: int, bias: bool = True, device=None,
                 dtype=None):
        super().__init__()
        if axis not in ("in", "out"):
            raise ValueError("axis must be 'in' or 'out'")
        self.in_features, self.out_features, self.axis, self.n = int(in_features), int(out_features), axis, int(n)
        full = self.out_features if axis == "out" else self.in_features
        if not 1 <= self.n <= full:
            raise ValueError(f"{self.n} kept of {full}")
        shape = (self.n, self.in_features) if axis == "out" else (self.out_features, self.n)
        self.weight = nn.Parameter(torch.zeros(shape, device=device, dtype=dtype))
        nb = self.n if axis == "out" else self.out_features
        self.bias = nn.Parameter(torch.zeros(nb, device=device, dtype=dtype)) if bias else None
        self.register_buffer("index", torch.zeros(self.n, dtype=torch.long, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.axis == "in":
            return F.linear(x.index_select(-1, self.index), self.weight, self.bias)
        y = F.linear(x, self.weight, self.bias)
        out = y.new_zeros(*y.shape[:-1], self.out_features)
        return out.index_copy(out.dim() - 1, self.index, y)

    @torch.no_grad()
    def dense(self) -> tuple:
        """(W [out, in], b [out] or None) in float64 as the full-size layer computes them."""
        W = torch.zeros(self.out_features, self.in_features, dtype=torch.float64, device=self.weight.device)
        if self.axis == "out":
            W[self.index] = self.weight.double()
            if self.bias is None:
                return W, None
            b = torch.zeros(self.out_features, dtype=torch.float64, device=self.weight.device)
            b[self.index] = self.bias.double()
            return W, b
        W[:, self.index] = self.weight.double()
        return W, None if self.bias is None else self.bias.double()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, axis={self.axis}, kept={self.n}"


class SlicedLinear(nn.Module):
    """SliceGPT-style sliced matrix: y = Q_out (W_s (Q_in^T x) + b_s), with W_s [d_out, d_in] and b_s [d_out]
    trainable and the bases Q_in [in, d_in] and Q_out [out, d_out] fixed buffers (absent on a side that is not
    sliced, where the identity is used)."""

    def __init__(self, in_features: int, out_features: int, d_in: int, d_out: int, bias: bool = True, device=None,
                 dtype=None):
        super().__init__()
        self.in_features, self.out_features = int(in_features), int(out_features)
        self.d_in, self.d_out = int(d_in), int(d_out)
        if not (1 <= self.d_in <= self.in_features and 1 <= self.d_out <= self.out_features):
            raise ValueError(f"slice ({self.d_out}, {self.d_in}) out of range for {self.out_features} x {self.in_features}")
        self.weight = nn.Parameter(torch.zeros(self.d_out, self.d_in, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(self.d_out, device=device, dtype=dtype)) if bias else None
        if self.d_in < self.in_features:
            self.register_buffer("basis_in", torch.zeros(self.in_features, self.d_in, device=device, dtype=dtype))
        else:
            self.basis_in = None
        if self.d_out < self.out_features:
            self.register_buffer("basis_out", torch.zeros(self.out_features, self.d_out, device=device, dtype=dtype))
        else:
            self.basis_out = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.basis_in is not None:
            x = x @ self.basis_in
        y = F.linear(x, self.weight, self.bias)
        if self.basis_out is not None:
            y = y @ self.basis_out.T
        return y

    @torch.no_grad()
    def dense(self) -> tuple:
        W = self.weight.double()
        b = None if self.bias is None else self.bias.double()
        if self.basis_in is not None:
            W = W @ self.basis_in.double().T
        if self.basis_out is not None:
            Qo = self.basis_out.double()
            W = Qo @ W
            b = None if b is None else Qo @ b
        return W, b

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, d_in={self.d_in}, "
                f"d_out={self.d_out}")


class SplitQKV(nn.Module):
    """attn.qkv replacement holding q, k and v as separate modules of any kind; output [q; k; v] like the fused
    layer. `weight` and `bias` are the dense concatenations (read-only, for analysis code)."""

    def __init__(self, q: nn.Module, k: nn.Module, v: nn.Module):
        super().__init__()
        self.q, self.k, self.v = q, k, v
        self.in_features = int(q.in_features)
        self.out_features = int(q.out_features + k.out_features + v.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.q(x), self.k(x), self.v(x)], dim=-1)

    @property
    def weight(self) -> torch.Tensor:
        return torch.cat([dense_weight(m)[0] for m in (self.q, self.k, self.v)], 0)

    @property
    def bias(self):
        bs = [dense_weight(m)[1] for m in (self.q, self.k, self.v)]
        if any(b is None for b in bs):
            return None
        return torch.cat(bs, 0)


@torch.no_grad()
def dense_weight(mod: nn.Module) -> tuple:
    """(W [out, in], b or None) in float64 for any linear-like module (from the factors for low rank)."""
    if isinstance(mod, LowRankLinear):
        W = mod.U.double() @ mod.V.double()
        return W, None if mod.bias is None else mod.bias.double()
    if isinstance(mod, (IndexedLinear, SlicedLinear)):
        return mod.dense()
    if isinstance(mod, (SplitQKV, LowRankQKV)):
        parts = [dense_weight(m) for m in (mod.q, mod.k, mod.v)]
        W = torch.cat([p[0] for p in parts], 0)
        b = None if any(p[1] is None for p in parts) else torch.cat([p[1] for p in parts], 0)
        return W, b
    W = mod.weight.detach().double()
    b = getattr(mod, "bias", None)
    return W, None if b is None else b.detach().double()


def module_macs(mod: nn.Module) -> int:
    """Multiply-adds per token of a linear-like module as it executes (no biases)."""
    if isinstance(mod, LowRankLinear):
        return int(mod.rank * (mod.in_features + mod.out_features))
    if isinstance(mod, IndexedLinear):
        return int(mod.weight.numel())
    if isinstance(mod, SlicedLinear):
        n = mod.weight.numel()
        n += 0 if mod.basis_in is None else mod.basis_in.numel()
        n += 0 if mod.basis_out is None else mod.basis_out.numel()
        return int(n)
    if isinstance(mod, (SplitQKV, LowRankQKV)):
        return int(sum(module_macs(m) for m in (mod.q, mod.k, mod.v)))
    return int(mod.weight.numel())


def module_spec(mod: nn.Module) -> dict:
    """JSON description of a linear-like module, enough to rebuild an empty one of the same shapes."""
    if isinstance(mod, LowRankLinear):
        return {"type": "lowrank", "in": mod.in_features, "out": mod.out_features, "rank": mod.rank,
                "bias": mod.bias is not None}
    if isinstance(mod, IndexedLinear):
        return {"type": "indexed", "in": mod.in_features, "out": mod.out_features, "axis": mod.axis, "n": mod.n,
                "bias": mod.bias is not None}
    if isinstance(mod, SlicedLinear):
        return {"type": "sliced", "in": mod.in_features, "out": mod.out_features, "d_in": mod.d_in,
                "d_out": mod.d_out, "bias": mod.bias is not None}
    if isinstance(mod, (SplitQKV, LowRankQKV)):
        return {"type": "split_qkv", "parts": [module_spec(m) for m in (mod.q, mod.k, mod.v)]}
    if isinstance(mod, nn.Linear):
        return {"type": "linear", "in": mod.in_features, "out": mod.out_features, "bias": mod.bias is not None}
    raise TypeError(f"no spec for module type {type(mod).__name__}")


def module_from_spec(spec: dict, device=None, dtype=None) -> nn.Module:
    t = spec["type"]
    if t == "lowrank":
        return LowRankLinear(spec["in"], spec["out"], spec["rank"], bias=spec["bias"], device=device, dtype=dtype)
    if t == "indexed":
        return IndexedLinear(spec["in"], spec["out"], spec["axis"], spec["n"], bias=spec["bias"], device=device,
                             dtype=dtype)
    if t == "sliced":
        return SlicedLinear(spec["in"], spec["out"], spec["d_in"], spec["d_out"], bias=spec["bias"], device=device,
                            dtype=dtype)
    if t == "split_qkv":
        return SplitQKV(*[module_from_spec(p, device, dtype) for p in spec["parts"]])
    if t == "linear":
        return nn.Linear(spec["in"], spec["out"], bias=spec["bias"], device=device, dtype=dtype)
    raise ValueError(f"unknown module type {t!r}")


# ----------------------------------------------------------------------------------------------
# Conversions (dense compressed weights -> structured trainable modules)
# ----------------------------------------------------------------------------------------------
def _tensor(tensors: dict, key: str) -> torch.Tensor:
    if key not in (tensors or {}):
        raise KeyError(f"info['tensors'] has no entry {key!r}")
    return _f64(tensors[key] if isinstance(tensors[key], (np.ndarray, torch.Tensor)) else np.asarray(tensors[key]))


def _default_axis(name: str) -> str:
    return "out" if name in ("fc1", "wq", "wk", "wv") else "in"


def _convert(W: torch.Tensor, b: torch.Tensor | None, e: dict, name: str, tensors: dict, device, dtype,
             tol: float, factors: str = "balanced") -> tuple:
    """(module, record) for one compressed matrix given as dense float64 W [out, in] and b."""
    kind = e["kind"]
    rec = {"block": int(e["block"]), "name": name, "kind": kind}
    if kind == "lowrank":
        r = int(e["rank"])
        ukey = e.get("U_key") or f"{int(e['block'])}.{name}.U"
        vkey = e.get("V_key") or f"{int(e['block'])}.{name}.V"
        if factors == "given":
            U, V = _tensor(tensors, ukey), _tensor(tensors, vkey)
            mod = LowRankLinear.from_factors(U.to(dtype=dtype), V.to(dtype=dtype),
                                             None if b is None else b.to(dtype), dtype=dtype).to(device)
            rec["factors"] = "given"
        else:
            mod = LowRankLinear.from_weight(W.to(device), r, None if b is None else b.to(device), dtype=dtype)
            rec["factors"] = "balanced SVD of the compressed weight"
        if mod.rank != r:
            raise ValueError(f"block {e['block']} {name}: factors have rank {mod.rank}, the entry says {r}")
        Wn, _ = dense_weight(mod)
        rec.update(rank=r, conversion_rel_err=_rel(Wn.cpu(), W.cpu()))
    elif kind == "width":
        axis = e.get("axis") or _default_axis(name)
        kept = torch.as_tensor(sorted(int(i) for i in e["kept"]), dtype=torch.long)
        full = W.shape[0] if axis == "out" else W.shape[1]
        if len(set(kept.tolist())) != len(kept) or (len(kept) and (kept.min() < 0 or kept.max() >= full)):
            raise ValueError(f"block {e['block']} {name}: kept indices invalid for size {full}")
        mod = IndexedLinear(W.shape[1], W.shape[0], axis, len(kept), bias=b is not None, device=device, dtype=dtype)
        drop = torch.ones(full, dtype=torch.bool)
        drop[kept] = False
        with torch.no_grad():
            mod.index.copy_(kept.to(device))
            if axis == "out":
                mod.weight.copy_(W[kept])
                if b is not None:
                    mod.bias.copy_(b[kept])
                lost = torch.cat([W[drop].flatten(), b[drop] if b is not None else W.new_zeros(0)])
            else:
                mod.weight.copy_(W[:, kept])
                if b is not None:
                    mod.bias.copy_(b)
                lost = W[:, drop].flatten()
        rec.update(axis=axis, n_kept=int(len(kept)), size=int(full), index_sha=_sha(kept.double()),
                   pruned_rel_norm=float(lost.norm()) / max(float(W.norm()), 1e-300))
    elif kind == "slice":
        d_in, d_out = int(e["d_in"]), int(e["d_out"])
        out_f, in_f = W.shape
        if d_out > out_f or d_in > in_f:
            raise ValueError(f"block {e['block']} {name}: slice ({d_out}, {d_in}) larger than {out_f} x {in_f}")
        mod = SlicedLinear(in_f, out_f, d_in, d_out, bias=b is not None, device=device, dtype=dtype)
        Ws, bs = W, b
        if mod.basis_in is not None:
            Q = _tensor(tensors, e.get("basis_in_key") or e["basis_key"])
            if Q.shape[0] != in_f or Q.shape[1] < d_in:
                raise ValueError(f"block {e['block']} {name}: input basis {tuple(Q.shape)} does not fit {in_f} x {d_in}")
            Qi = Q[:, :d_in].to(W.device)
            Ws = Ws @ Qi
            with torch.no_grad():
                mod.basis_in.copy_(Qi)
        if mod.basis_out is not None:
            Q = _tensor(tensors, e.get("basis_out_key") or e["basis_key"])
            if Q.shape[0] != out_f or Q.shape[1] < d_out:
                raise ValueError(f"block {e['block']} {name}: output basis {tuple(Q.shape)} does not fit {out_f} x {d_out}")
            Qo = Q[:, :d_out].to(W.device)
            Ws = Qo.T @ Ws
            bs = None if bs is None else Qo.T @ bs
            with torch.no_grad():
                mod.basis_out.copy_(Qo)
        with torch.no_grad():
            mod.weight.copy_(Ws)
            if bs is not None:
                mod.bias.copy_(bs)
        Wn, bn = dense_weight(mod)
        err = _rel(Wn.cpu(), W.cpu())
        if b is not None:
            err = max(err, _rel(bn.cpu(), b.cpu()) if float(b.norm()) > 0 else float(bn.norm()))
        rec.update(d_in=d_in, d_out=d_out, conversion_rel_err=err,
                   basis_in_sha=None if mod.basis_in is None else _sha(mod.basis_in),
                   basis_out_sha=None if mod.basis_out is None else _sha(mod.basis_out))
    elif kind == "merge":
        # attention channels merged in dense form (mirror.baselines zipit: the group's first index holds the merged
        # row of wq, wk, wv or column of wo, the other rows or columns are zero): stored as the width form on those
        # indices, so the zero rows stay zero while training
        kept = sorted(int(g[0]) for g in e["groups"])
        mod, rec = _convert(W, b, {**e, "kind": "width", "kept": kept}, name, tensors, device, dtype, tol, factors)
        rec.update(kind="merge", form="width on the first index of each group", n_groups=len(kept))
        if rec["pruned_rel_norm"] > tol:
            raise ValueError(f"block {e['block']} {name}: rows outside the merged indices are not zero "
                             f"(relative {rec['pruned_rel_norm']:.2e})")
        return mod, rec
    else:
        raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")
    if rec.get("conversion_rel_err", 0.0) > tol:
        raise ValueError(f"block {e['block']} {name}: the {kind} form differs from the compressed weight by "
                         f"{rec['conversion_rel_err']:.2e} (relative; tolerance {tol:g}). Check the compressor's entry.")
    rec["params"] = int(module_macs(mod))
    rec["params_compressor"] = None if e.get("params") is None else int(e["params"])
    return mod, rec


def _mlp_pair(mlp: nn.Module, e1: dict, e2: dict, b: int, device, dtype, tol: float) -> list:
    """Width or merge on both fc1 and fc2 with the same units: two smaller dense layers."""
    W1, b1 = dense_weight(mlp.fc1)
    W2, b2 = dense_weight(mlp.fc2)
    kind = e1["kind"]
    if kind == "width":
        kept = sorted(int(i) for i in e1["kept"])
        if sorted(int(i) for i in e2["kept"]) != kept:
            raise ValueError(f"block {b}: fc1 and fc2 keep different units")
        n = len(kept)
        if W1.shape[0] == n and W2.shape[1] == n:          # already shrunk by the compressor
            W1s, b1s, W2s, lost = W1, b1, W2, 0.0
        else:
            idx = torch.as_tensor(kept, dtype=torch.long, device=W1.device)
            drop = torch.ones(W1.shape[0], dtype=torch.bool, device=W1.device)
            drop[idx] = False
            W1s, b1s, W2s = W1[idx], None if b1 is None else b1[idx], W2[:, idx]
            lost = float(W2[:, drop].norm()) / max(float(W2.norm()), 1e-300)
            if lost > tol:
                raise ValueError(f"block {b}: fc2 columns of the pruned units are not zero (relative {lost:.2e}); "
                                 "removing them would change the compressed model")
        extra = {"n_kept": n, "pruned_fc2_rel_norm": lost, "index_sha": _sha(torch.as_tensor(kept, dtype=torch.float64))}
    else:
        groups = [sorted(int(i) for i in g) for g in e1["groups"]]
        if e2.get("groups") is not None and [sorted(int(i) for i in g) for g in e2["groups"]] != groups:
            raise ValueError(f"block {b}: fc1 and fc2 merge different groups")
        n = len(groups)
        if W1.shape[0] == n and W2.shape[1] == n:          # already merged by the compressor
            W1s, b1s, W2s, tie = W1, b1, W2, 0.0
        else:
            rows = [W1[g].mean(0) for g in groups]
            W1s = torch.stack(rows)
            b1s = None if b1 is None else torch.stack([b1[g].mean() for g in groups])
            W2s = torch.stack([W2[:, g].sum(1) for g in groups], 1)
            tie = max(float((W1[g] - r).norm()) for g, r in zip(groups, rows)) / max(float(W1.norm()), 1e-300)
            if b1 is not None:
                tie = max(tie, max(float((b1[g] - b1[g].mean()).abs().max()) for g in groups)
                          / max(float(b1.abs().max()), 1e-300))
            if tie > tol:
                raise ValueError(f"block {b}: fc1 rows inside a merge group are not tied (relative {tie:.2e})")
        extra = {"n_units": n, "tie_rel_err": tie}
    mlp.fc1 = _linear(W1s, b1s, device, dtype)
    mlp.fc2 = _linear(W2s, b2, device, dtype)
    return [{"block": b, "name": "fc1", "kind": kind, "shape": list(mlp.fc1.weight.shape), **extra,
             "params": int(mlp.fc1.weight.numel()),
             "params_compressor": None if e1.get("params") is None else int(e1["params"])},
            {"block": b, "name": "fc2", "kind": kind, "shape": list(mlp.fc2.weight.shape), **extra,
             "params": int(mlp.fc2.weight.numel()),
             "params_compressor": None if e2.get("params") is None else int(e2["params"])}]


_RECORD_PATHS = {"fc1": "mlp.fc1", "fc2": "mlp.fc2", "wq": "attn.qkv.q", "wk": "attn.qkv.k", "wv": "attn.qkv.v",
                 "wo": "attn.proj"}


def _install_block(model: nn.Module, b: int, entries: dict, tensors: dict, tol: float,
                   factors: str = "balanced") -> list:
    blk = model.blocks[b]
    ref = next(blk.parameters())
    device, dtype = ref.device, ref.dtype
    records = []
    mlp = blk.mlp
    e1, e2 = entries.get("fc1"), entries.get("fc2")
    k1, k2 = (e1 or {}).get("kind"), (e2 or {}).get("kind")
    if k1 == k2 and k1 in ("width", "merge"):
        records += _mlp_pair(mlp, e1, e2, b, device, dtype, tol)
    else:
        if "merge" in (k1, k2):
            raise NotImplementedError(f"block {b}: merge needs both fc1 and fc2 with the same groups")
        for name, e in (("fc1", e1), ("fc2", e2)):
            if e is None:
                continue
            W, bias = dense_weight(getattr(mlp, name))
            mod, rec = _convert(W, bias, e, name, tensors, device, dtype, tol, factors)
            setattr(mlp, name, mod)
            records.append(rec)
    attn = blk.attn
    if any(n in entries for n in QKV_NAMES):
        W, bias = dense_weight(attn.qkv)
        D = W.shape[0] // 3
        parts = []
        for i, name in enumerate(QKV_NAMES):
            Wi = W[i * D:(i + 1) * D]
            bi = None if bias is None else bias[i * D:(i + 1) * D]
            if name in entries:
                mod, rec = _convert(Wi, bi, entries[name], name, tensors, device, dtype, tol, factors)
                records.append(rec)
            else:
                mod = _linear(Wi, bi, device, dtype)
            parts.append(mod)
        attn.qkv = SplitQKV(*parts)
    if "wo" in entries:
        W, bias = dense_weight(attn.proj)
        mod, rec = _convert(W, bias, entries["wo"], "wo", tensors, device, dtype, tol, factors)
        attn.proj = mod
        records.append(rec)
    for r in records:
        r["path"] = _RECORD_PATHS[r["name"]]
    return records


# ----------------------------------------------------------------------------------------------
# Trainable set and structure checks (constraint protocol of mirror/heal.py)
# ----------------------------------------------------------------------------------------------
def _block_absorbed(blk: nn.Module) -> list:
    return list(_absorbed_parameters(blk)) + list(attn_absorbed_parameters(blk))


class StructuredTrainable:
    """Constraint for a compressed baseline: nothing is projected (the module forms keep the structure); freezes
    everything except the converted blocks and the embeddings, and inside the converted blocks the absorbed norm
    affines and LayerScale (E3's trainable set, mirror.heal_vexact.SpanTrainable plus LowRankMLP's frozen absorbed
    parameters). `check` confirms the structure of every converted matrix (ranks in float64 from the factors,
    widths and kept indices, slice shapes and unchanged bases)."""

    def __init__(self, blocks: Sequence[int], records: Sequence[dict], embeddings: bool = True):
        self.blocks = [int(b) for b in blocks]
        self.records = [dict(r) for r in records]
        self.embeddings = bool(embeddings)

    def frozen_parameters(self, model: nn.Module) -> list:
        out = list(span_frozen_parameters(model, self.blocks, self.embeddings))
        for b in self.blocks:
            out += _block_absorbed(model.blocks[b])
        return out

    def __call__(self, model: nn.Module) -> None:
        return None

    @torch.no_grad()
    def check(self, model: nn.Module) -> dict:
        rows, ok_all = [], True
        for r in self.records:
            mod = model.blocks[r["block"]].get_submodule(r["path"])
            row = {"block": r["block"], "name": r["name"], "kind": r["kind"]}
            if r["kind"] == "lowrank":
                ok = isinstance(mod, LowRankLinear) and mod.rank == r["rank"]
                row.update(target_rank=r["rank"], numerical_rank=mod.rank64() if ok else None,
                           rank_precision="float64 product of the factors")
                ok = ok and row["numerical_rank"] <= r["rank"]
            elif r["kind"] == "slice":
                ok = isinstance(mod, SlicedLinear) and mod.d_in == r["d_in"] and mod.d_out == r["d_out"]
                if ok:
                    row["basis_unchanged"] = ((r.get("basis_in_sha") is None or _sha(mod.basis_in) == r["basis_in_sha"])
                                              and (r.get("basis_out_sha") is None
                                                   or _sha(mod.basis_out) == r["basis_out_sha"]))
                    ok = row["basis_unchanged"]
                row.update(d_in=r["d_in"], d_out=r["d_out"])
            elif "shape" in r:                                   # width or merge MLP pair: smaller dense layers
                ok = isinstance(mod, nn.Linear) and list(mod.weight.shape) == r["shape"]
                row["shape"] = r["shape"]
            else:                                                # width as IndexedLinear
                ok = (isinstance(mod, IndexedLinear) and mod.n == r["n_kept"] and mod.axis == r["axis"]
                      and _sha(mod.index.double()) == r["index_sha"])
                row.update(n_kept=r["n_kept"], axis=r["axis"])
            row["ok"] = bool(ok)
            ok_all = ok_all and row["ok"]
            rows.append(row)
        lr = [x for x in rows if x["kind"] == "lowrank"]
        return {"ok": bool(ok_all), "n_matrices": len(rows), "kinds": dict(Counter(x["kind"] for x in rows)),
                "numerical_ranks": [x.get("numerical_rank") for x in lr], "target_ranks": [x["target_rank"] for x in lr],
                "per_matrix": rows}

    def describe(self) -> dict:
        return {"type": "StructuredTrainable", "blocks": self.blocks,
                "kinds": dict(Counter(r["kind"] for r in self.records)), "embeddings_train": self.embeddings,
                "frozen": "everything outside the converted blocks and the embeddings; norm affines and LayerScale "
                          "inside the converted blocks"}


# ----------------------------------------------------------------------------------------------
# make_trainable
# ----------------------------------------------------------------------------------------------
_WRAPPER_BUFFERS = ("entry", "attn_shortcut", "mlp_shortcut")


def _wrappers(model: nn.Module, blocks) -> list:
    """Blocks the compressor replaced by a wrapper class (mirror.baselines.SlicedBlock: SliceGPT's sliced residual
    stream in dense form, fixed projection buffers), with the buffer shapes build_skeleton needs."""
    out = []
    for b in blocks:
        blk = model.blocks[b]
        if type(blk).__name__ == "SlicedBlock":
            out.append({"block": int(b), "type": "SlicedBlock",
                        "buffers": {n: (None if getattr(blk, n, None) is None else list(getattr(blk, n).shape))
                                    for n in _WRAPPER_BUFFERS}})
    return out


def _base_spec(model: nn.Module, adapter) -> dict:
    pe, pe0 = getattr(model, "pos_embed", None), getattr(adapter.model, "pos_embed", None)
    materialised = isinstance(pe, torch.Tensor) and isinstance(pe0, torch.Tensor) and tuple(pe.shape) != tuple(pe0.shape)
    return {"materialise_pos_embed": bool(materialised), "fold": bool(is_folded_model(model)),
            "even_head": bool(hasattr(getattr(model, "norm", None), "even"))}


def _mirror_trainable(model: nn.Module, info: dict, adapter) -> tuple:
    if adapter is None:
        raise ValueError("the mirror family needs the adapter (VExactEmbeddings re-projects the embeddings)")
    comps = tuple(info.get("components") or ("mlp", "attn"))
    if set(comps) != {"mlp", "attn"}:
        raise NotImplementedError("Mirror-exact healing converts MLP and attention together (mirror.m1.build_vexact)")
    blocks = [int(b) for b in info["blocks"]]
    A = exact_involution(info["A_star"])
    perms = [np.asarray(p, dtype=np.int64) for p in info["perms"]]
    maps = list(info["maps_after"])
    if not (len(perms) == len(maps) == len(blocks)):
        raise ValueError("perms and maps_after must be aligned with blocks")
    reg_perm = info.get("reg_perm")
    k = len(blocks)
    cons = [MirrorMLPConstraint(blocks, [A] * k, [A] * k, perms),
            MirrorAttnConstraint(blocks, [A] * k, [A] * k, maps),
            VExactEmbeddings(adapter, A, blocks, reg_perm, train=True)]
    spec = {"format": SPEC_FORMAT, "family": "mirror", "method": info.get("method"), "blocks": blocks,
            "components": list(comps), "base": _base_spec(model, adapter), "modules": [], "matrices": [],
            "mirror": {"d_plus": int(round((np.trace(A) + A.shape[0]) / 2)), "structure_file": "structure.npz",
                       "n_pairs": [int((p != np.arange(len(p))).sum() // 2) for p in perms],
                       "n_head_pairs": [int(m.n_pairs) for m in maps], "builder": info.get("builder")},
            "trainable": "converted blocks (re-projected with A* after every step) and the embeddings (re-projected); "
                         "absorbed norms, LayerScale, final norm affine and mask token frozen"}
    return model, cons, spec


def make_trainable(model: nn.Module, info: dict, adapter=None, tol: float = DEFAULT_TOL,
                   lowrank_factors: str = "balanced") -> tuple:
    """Turn a compressor's output (`model`, changed in place) into its structured trainable form.

    Returns (model, constraints, spec): run mirror/heal.py with train_scope "all" and these constraints; `spec`
    (JSON) is what build_skeleton needs to rebuild the model (save it with write_spec). With an adapter the
    position grid of the evaluation size is written into DINOv2 models first (mirror.m1.materialise_pos_embed, the
    same function at that size), as V-exact and the E3 SVD arms do, so every method trains the same grid.
    lowrank_factors: "balanced" (default; every low-rank method alike, as the E3 SVD arm: U = U_r S_r^(1/2),
    V = S_r^(1/2) Vh_r of the compressed weight) or "given" (the compressor's factors, info["tensors"]
    "<b>.<name>.U" and ".V" or the entry's U_key and V_key)."""
    if lowrank_factors not in ("balanced", "given"):
        raise ValueError("lowrank_factors must be 'balanced' or 'given'")
    if info.get("kind") == "mirror":
        return _mirror_trainable(model, info, adapter)
    if adapter is not None:
        from .m1 import materialise_pos_embed

        materialise_pos_embed(model, adapter)
    entries = list(info.get("per_matrix") or [])
    if not entries:
        raise ValueError("info['per_matrix'] is empty: nothing is compressed")
    by_block: dict = {}
    for e in entries:
        if e.get("name") not in MATRIX_NAMES:
            raise ValueError(f"unknown matrix name {e.get('name')!r}; expected one of {MATRIX_NAMES}")
        b = int(e["block"])
        if e["name"] in by_block.setdefault(b, {}):
            raise ValueError(f"block {b} {e['name']} appears twice in per_matrix")
        by_block[b][e["name"]] = e
    blocks = sorted(int(b) for b in (info.get("blocks") or by_block))
    tensors = info.get("tensors") or {}
    records = []
    for b in blocks:
        records += _install_block(model, b, by_block.get(b, {}), tensors, tol, lowrank_factors)
    cons = [StructuredTrainable(blocks, records)]
    wrappers = _wrappers(model, blocks)
    base = _base_spec(model, adapter) if adapter is not None else {
        "materialise_pos_embed": False, "fold": bool(is_folded_model(model)),
        "even_head": bool(hasattr(getattr(model, "norm", None), "even"))}
    modules = [{"block": b, "path": p, "module": module_spec(model.blocks[b].get_submodule(p))}
               for b in blocks for p in SLOT_PATHS]
    spec = {"format": SPEC_FORMAT, "family": "compressed", "method": info.get("method"), "blocks": blocks,
            "components": list(info.get("components") or sorted({"mlp" if e["name"] in MLP_NAMES else "attn"
                                                                 for e in entries})),
            "base": base, "wrappers": wrappers, "modules": modules, "matrices": _jsonable(records),
            "compressor": {**{k: _jsonable(info.get(k)) for k in ("params_total", "params_matrices",
                                                                   "params_basis_change", "budget_total", "gap", "source",
                                                                   "model_form", "hparams") if k in info},
                           "basis": [{k: _jsonable(x.get(k)) for k in ("block", "name", "kind", "shape", "params")}
                                     for x in (info.get("basis") or [])]},
            "lowrank_factors": lowrank_factors,
            "trainable": "structured parameters of the converted blocks (and their biases) and the embeddings, "
                         "free (nothing re-projected); absorbed norms, LayerScale, final norm affine and mask token "
                         "frozen"}
    return model, cons, spec


# ----------------------------------------------------------------------------------------------
# Spec, structure file and rebuilding
# ----------------------------------------------------------------------------------------------
def build_skeleton(adapter, spec: dict) -> nn.Module:
    """A model with the healed student's structure (and arbitrary weights): the original model, the base changes
    (position grid, LayerNorm folding, even head) and the structured modules of the spec. Load the healed state
    into it with load_state_dict(strict=True)."""
    from .fold import fold_layernorm_full
    from .m1 import install_even_head, materialise_pos_embed

    m = copy.deepcopy(adapter.model)
    base = spec["base"]
    if base.get("materialise_pos_embed"):
        materialise_pos_embed(m, adapter)
    if base.get("fold"):
        fold_layernorm_full(m, final_norm_affine=False)
    if base.get("even_head"):
        install_even_head(m, np.eye(int(adapter.embed_dim)))      # P itself comes from the state
    ref = next(m.parameters())
    for w in spec.get("wrappers", []):
        if w["type"] != "SlicedBlock":
            raise ValueError(f"unknown block wrapper {w['type']!r}")
        from .baselines import SlicedBlock

        bufs = {n: (None if shp is None else torch.zeros(shp, device=ref.device, dtype=ref.dtype))
                for n, shp in w["buffers"].items()}
        m.blocks[int(w["block"])] = SlicedBlock(m.blocks[int(w["block"])], **bufs)
    for e in spec.get("modules", []):
        parent_path, attr = e["path"].rsplit(".", 1)
        parent = m.blocks[int(e["block"])].get_submodule(parent_path)
        setattr(parent, attr, module_from_spec(e["module"], ref.device, ref.dtype))
    return m.eval()


def verify_rebuild(adapter, spec: dict, model: nn.Module) -> dict:
    """Check, before training, that build_skeleton(spec) has exactly the student's state_dict keys, shapes and
    dtypes (so the healed state can be loaded back). Raises with the differences otherwise."""
    sk = build_skeleton(adapter, spec).state_dict()
    own = model.state_dict()
    missing, extra = sorted(set(own) - set(sk)), sorted(set(sk) - set(own))
    shape = [k for k in own if k in sk and (tuple(own[k].shape) != tuple(sk[k].shape) or own[k].dtype != sk[k].dtype)]
    if missing or extra or shape:
        raise ValueError(f"the spec does not rebuild the student: missing {missing[:5]}, unexpected {extra[:5]}, "
                         f"shape or dtype {shape[:5]}. The compressor changed the model in a way the spec does not "
                         "describe; extend mirror.heal_structured")
    return {"n_tensors": len(own), "ok": True}


def save_structure(path, A_star, perms, maps_after: Sequence[HeadMaps], reg_perm, blocks) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, A_star=np.asarray(A_star, dtype=np.float64), perms=np.stack([np.asarray(p, dtype=np.int64) for p in perms]),
             sigma=np.stack([np.asarray(m.sigma, dtype=np.int64) for m in maps_after]),
             B=np.stack([np.asarray(m.B, dtype=np.float64) for m in maps_after]),
             C=np.stack([np.asarray(m.C, dtype=np.float64) for m in maps_after]),
             reg_perm=np.asarray(reg_perm if reg_perm is not None else [], dtype=np.int64),
             blocks=np.asarray(blocks, dtype=np.int64))
    return path


def load_structure(path) -> dict:
    with np.load(path) as z:
        maps = [HeadMaps(sigma=z["sigma"][i], B=z["B"][i], C=z["C"][i]) for i in range(len(z["sigma"]))]
        return {"A_star": z["A_star"], "perms": [p for p in z["perms"]], "maps_after": maps,
                "reg_perm": z["reg_perm"], "blocks": [int(b) for b in z["blocks"]]}


def write_spec(run_dir, spec: dict, structure: dict | None = None) -> Path:
    """<run_dir>/spec.json (and structure.npz for the mirror family)."""
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    if spec.get("family") == "mirror":
        if structure is None:
            raise ValueError("the mirror family needs its structure (A_star, perms, maps_after, reg_perm)")
        save_structure(run / spec["mirror"]["structure_file"], structure["A_star"], structure["perms"],
                       structure["maps_after"], structure.get("reg_perm"), spec["blocks"])
    path = run / "spec.json"
    tmp = path.with_name("spec.json.tmp")
    tmp.write_text(json.dumps(_jsonable(spec), indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def adapter_spec(kind: str, **kw) -> dict:
    """How to rebuild the teacher adapter: kind "tiny" (img_size, seed), "dinov2" (hub_repo, name, img_size) or
    "registry" (name, img_size; mirror.adapters_timm.build_model_adapter)."""
    if kind not in ("tiny", "dinov2", "registry"):
        raise ValueError(f"unknown adapter kind {kind!r}")
    return {"kind": kind, **kw}


def build_adapter_from_spec(spec: dict, device="cpu"):
    kind = spec["kind"]
    if kind == "tiny":
        from .adapters import TimmTinyAdapter

        ad = TimmTinyAdapter(img_size=int(spec["img_size"]), seed=int(spec["seed"]), randomise=True,
                             dtype=getattr(torch, spec.get("dtype", "float32")))
    elif kind == "dinov2":
        from .adapters import Dinov2Adapter

        ad = Dinov2Adapter(hub_repo=spec["hub_repo"], name=spec["name"], img_size=int(spec["img_size"]))
    else:
        from .adapters_timm import build_model_adapter

        ad = build_model_adapter(spec["name"], img_size=int(spec["img_size"]))
    return ad.to(device=device)


def mirror_constraints(adapter, structure: dict) -> list:
    blocks = structure["blocks"]
    A = exact_involution(structure["A_star"])
    k = len(blocks)
    reg = structure.get("reg_perm")
    reg = None if reg is None or len(reg) == 0 else reg
    return [MirrorMLPConstraint(blocks, [A] * k, [A] * k, structure["perms"]),
            MirrorAttnConstraint(blocks, [A] * k, [A] * k, structure["maps_after"]),
            VExactEmbeddings(adapter, A, blocks, reg, train=False)]


def load_healed_model(run_dir, device="cpu", adapter=None, check: bool = True) -> tuple:
    """Rebuild a healed MUST-09 student: (adapter, model, info).

    `adapter` is the original model's adapter (built from spec["adapter"] unless given); run the student with
    adapter.cls_feature(x, model) and so on. info: spec, method, family, the state file's meta, for the mirror
    family the structure (A_star, perms, maps_after, reg_perm), the constraints (for checks) and, with `check`,
    every constraint's check on the loaded model."""
    run = Path(run_dir)
    spec = json.loads((run / "spec.json").read_text(encoding="utf-8"))
    if spec.get("format") != SPEC_FORMAT:
        raise ValueError(f"{run / 'spec.json'} is not a {SPEC_FORMAT} spec")
    if adapter is None:
        adapter = build_adapter_from_spec(spec["adapter"], device)
    model = build_skeleton(adapter, spec)
    payload = torch.load(run / spec["state_file"], map_location="cpu", weights_only=True)
    sd = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model.load_state_dict(sd, strict=True)
    model.to(device=device).eval()
    model.requires_grad_(False)
    info = {"run_dir": str(run), "spec": spec, "method": spec.get("method"), "family": spec.get("family"),
            "seed": spec.get("seed"), "targets": spec.get("targets"),
            "meta": payload.get("meta") if isinstance(payload, dict) else None, "structure": None}
    if spec.get("family") == "mirror":
        st = load_structure(run / spec["mirror"]["structure_file"])
        st["A"] = st["A_star"]                                  # A* under both names
        info["structure"] = st
        info["constraints"] = mirror_constraints(adapter, st)
    else:
        info["constraints"] = [StructuredTrainable(spec["blocks"], spec["matrices"])]
    info.update(param_counts(adapter, model, spec, info["structure"]))
    if check:
        info["checks"] = {f"{i}:{type(c).__name__}": c.check(model) for i, c in enumerate(info["constraints"])
                          if callable(getattr(c, "check", None))}
    return adapter, model, info


# ----------------------------------------------------------------------------------------------
# Targets, costs and checks
# ----------------------------------------------------------------------------------------------
def distill_teacher(adapter, spec: dict, targets: str, A, reg_perm) -> VExactTeacher:
    """The teacher for MUST-09 (mirror/heal.py calls its distill_targets).

    targets "sym" (default for every method): the final CLS and patch targets are the flip average of the
    teacher's final-norm features, with the fixed loss normalisers measured on them. Block targets follow each
    student's own stream: Mirror-exact (folded, exactly equivariant) gets the centred teacher stream symmetrised
    with its A* (loss-neutral up to a constant for a student with that symmetry, PREREG_E3.md Section 5); a
    baseline gets the teacher's own block outputs, centred if its stream is the folded one. targets "raw": the
    same without any flip averaging (the teacher's own final features; mirror blocks only centred)."""
    if targets not in ("sym", "raw"):
        raise ValueError("targets must be 'sym' or 'raw'")
    blocks = [int(b) for b in spec["blocks"]]
    sym = targets == "sym"
    if spec["family"] == "mirror":
        if blocks != list(range(len(blocks))):
            raise ValueError("the mirror span starts at block 0")
        return VExactTeacher(adapter, A, len(blocks), reg_perm, symmetrise=sym, final_target="symmetrised")
    block_targets = "centred" if spec["base"].get("fold") else "raw"
    k = max(blocks) + 1 if blocks == list(range(max(blocks) + 1)) else int(adapter.depth)
    return VExactTeacher(adapter, A, k, reg_perm, symmetrise=sym, final_target="symmetrised", block_targets=block_targets)


def block_costs(model: nn.Module, spec: dict, structure: dict | None = None, d: int | None = None,
                h: int | None = None) -> list:
    """Per converted block, multiply-adds per token of its matrices in the method's own efficient form:
    {"block", "mlp" (fc1 + fc2), "attn" (Wq + Wk + Wv + Wo), "rot" (basis-change multiply-adds per token in units of
    d^2), "basis_params", "mlp_exec", "attn_exec", "source"}. Input of scripts/collect_results.analytic_cost.

    Mirror family: its free parameters (mirror.heal_vexact.vexact_block_params; the even/odd form of
    mirror/efficient.py executes exactly these), no basis change at full conversion. Compressors: the counts the
    compressor itself reports per matrix (info["per_matrix"] params) and its basis changes (info["basis"], e.g.
    SliceGPT's shortcut matrices), which is its own analytic accounting; without them, the structured modules as this
    module executes them (low rank r (in + out), slices with their basis products). `*_exec`: always the latter."""
    blocks = [int(b) for b in spec["blocks"]]
    if spec["family"] == "mirror":
        if structure is None:
            raise ValueError("the mirror family needs its structure")
        d = d or int(model.blocks[0].mlp.fc1.in_features)
        h = h or int(model.blocks[0].mlp.fc1.out_features)
        rows = vexact_block_params(d, h, structure["A_star"], structure["perms"], structure["maps_after"])
        return [{"block": b, "mlp": int(r["mlp"]), "attn": int(r["attn"]), "rot": 0, "basis_params": 0,
                 "mlp_exec": int(r["mlp"]), "attn_exec": int(r["attn"]), "source": "free parameters"}
                for b, r in zip(blocks, rows)]
    d = d or int(model.blocks[blocks[0]].mlp.fc1.in_features)
    recs = spec.get("matrices") or []
    comp = spec.get("compressor") or {}
    own = bool(recs) and all(r.get("params_compressor") is not None for r in recs)
    out = []
    for b in blocks:
        blk = model.blocks[b]
        row = {"block": b, "mlp_exec": module_macs(blk.mlp.fc1) + module_macs(blk.mlp.fc2),
               "attn_exec": module_macs(blk.attn.qkv) + module_macs(blk.attn.proj)}
        if own:
            done = {r["name"] for r in recs if r["block"] == b}
            mlp = sum(r["params_compressor"] for r in recs if r["block"] == b and r["name"] in MLP_NAMES)
            attn = sum(r["params_compressor"] for r in recs if r["block"] == b and r["name"] not in MLP_NAMES)
            mlp += sum(module_macs(getattr(blk.mlp, n)) for n in MLP_NAMES if n not in done)      # dense, uncompressed
            if not done & set(QKV_NAMES):
                attn += module_macs(blk.attn.qkv)
            if "wo" not in done:
                attn += module_macs(blk.attn.proj)
            basis = sum(int(x.get("params") or 0) for x in comp.get("basis", []) if x.get("block") == b)
            row.update(mlp=int(mlp), attn=int(attn), basis_params=int(basis), rot=basis / float(d * d),
                       source="compressor's own per-matrix and basis-change counts")
        else:
            row.update(mlp=row["mlp_exec"], attn=row["attn_exec"], basis_params=0, rot=0,
                       source="structured modules as executed")
        out.append(row)
    return out


def param_counts(adapter, model: nn.Module, spec: dict, structure: dict | None = None) -> dict:
    """Parameter counts of a healed student (for load_healed_model and the Tier-2 tables):
    model_params                 parameters of the model as stored (nn.Parameter entries; the mirror family stores
                                 its projected matrices dense, so this equals the original's count there);
    model_params_stored          the same plus floating-point buffers (slice bases, SliceGPT shortcuts, even head);
    converted_params_kept        the structured or free count over the converted matrices, basis changes included
                                 (block_costs: the mirror family's free parameters, a compressor's own counts);
    converted_params_dense       the dense count of the same matrices in the original model;
    model_params_structured      the whole-model count with the converted matrices at their kept count (biases,
                                 norms and embeddings dense), the count of the paper tables; and its share of the
                                 original (model_params_frac)."""
    comps = set(spec.get("components") or ("mlp", "attn"))
    orig = adapter.model
    dense = 0
    for b in spec["blocks"]:
        blk = orig.blocks[int(b)]
        if "mlp" in comps:
            dense += blk.mlp.fc1.weight.numel() + blk.mlp.fc2.weight.numel()
        if "attn" in comps:
            dense += blk.attn.qkv.weight.numel() + blk.attn.proj.weight.numel()
    rows = block_costs(model, spec, structure)
    kept = int(sum((r["mlp"] if "mlp" in comps else 0) + (r["attn"] if "attn" in comps else 0) + r["basis_params"]
                   for r in rows))
    n_orig = int(sum(p.numel() for p in orig.parameters()))
    structured = n_orig - int(dense) + kept
    return {"model_params": int(sum(p.numel() for p in model.parameters())), "model_params_stored": stored_params(model),
            "converted_params_kept": kept, "converted_params_dense": int(dense), "model_params_structured": structured,
            "model_params_original": n_orig, "model_params_frac": structured / n_orig,
            "params_source": rows[0]["source"] if rows else None}


def stored_params(model: nn.Module) -> int:
    """Entries the model stores: parameters plus floating-point buffers (slice bases, the even head), not index
    buffers."""
    n = sum(p.numel() for p in model.parameters())
    n += sum(b.numel() for b in model.buffers() if b.is_floating_point())
    return int(n)


@torch.no_grad()
def feature_gap(adapter, ref: nn.Module, model: nn.Module, x: torch.Tensor) -> dict:
    """Largest difference of the k-NN feature (final-norm CLS) of two models on x, absolute and relative to the
    largest entry of the reference feature."""
    a = adapter.cls_feature(x, ref).double()
    b = adapter.cls_feature(x, model).double()
    ma, mr = float((a - b).abs().max()), float(a.abs().max())
    return {"max_abs": ma, "max_ref": mr, "rel_max": ma / mr if mr > 0 else float("nan"), "n_images": int(x.shape[0])}
