"""UP-02 (symmetric start) and UP-04 (tied from scratch; refund and heal): exactly UD-equivariant controlled ViTs.

PROJECT_NEXT_PHASE.md 4.4, UP-02 and UP-04. The controlled ViT of mirror/controlled.py is exactly equivariant to
the upside-down flip (UD) when one involution acts on every weight so that, for every image x, the tokens of UD(x),
re-indexed to the positions of x, equal A times the tokens of x at every residual layer, and the logits of UD(x)
equal those of x. The action (column vectors; every operator a symmetric orthogonal involution):

    tokens        patch (r, c) -> (G-1-r, c); CLS keeps its index (mirror.groups.token_index("UD", G))
    channels      A (d x d) with A 1 = 1, so LayerNorm without affine commutes with it
    MLP hidden    a permutation sigma(b) of the h units of block b (P M = M[sigma], as mirror/groups.py)
    attention     P_qk(b) on q and k, P_v(b) on v (D x D, head-block structure: head h goes to head s(h) for
                  both), so attention logits are unchanged and the head outputs move with P_v
    patch pixels  the vertical flip of every p x p kernel

Projection (Section 4.4 with |G| = 2: W_eq = (W + g.W)/2, the nearest equivariant weight in Frobenius norm):
    patch_embed   W <- (W + A flipUD(W))/2 (A on output channels, flipUD on kernel rows), b <- (b + A b)/2
    pos_embed     pos[t] <- (pos[t] + A pos[tau(t)])/2 (tau the token action, CLS fixed); cls <- (cls + A cls)/2
    LayerNorm     beta <- (beta + A beta)/2; gamma <- its mean over each connected component of the support of A
                  (diag(gamma) must commute with A: a permutation A gives its orbits, a dense A one component)
    qkv           Wq <- (Wq + P_qk Wq A)/2, bq <- (bq + P_qk bq)/2; Wk alike; Wv, bv with P_v
    proj          Wo <- (Wo + A Wo P_v)/2, bo <- (bo + A bo)/2
    fc1, fc2      W1 <- (W1 + P W1 A)/2, b1 <- (b1 + P b1)/2; W2 <- (W2 + A W2 P)/2, b2 <- (b2 + A b2)/2
    head          W <- (W + W A)/2 (UD-invariant logits); bias unchanged
On weights the action is an orthogonal involution, so the map is an orthogonal projection (idempotent and
self-adjoint): applied to gradients it gives the gradient of the loss restricted to the equivariant subspace.

Canonical structure (UP-02 and the UP-04 tied run; `canonical_structure`). The token and pixel actions are fixed
by UD. The rest is a choice, made so that every operator is a permutation of weight entries:
    channels      A swaps channel i with channel i + k for i < k (k = d_minus, default d / 2), the other
                  channels are fixed points. A permutation fixes the all-ones vector, so A 1 = 1 as for every
                  operator the Mirror code fits. d+ = d - k, d- = k: half and half by default.
    MLP hidden    unit i swapped with unit i + n for i < n, n = round(hidden_pairs h / 2) (default: every unit
                  in a pair), the rest fixed.
    attention     every head maps to itself; inside each head, dimension j is swapped with j + dh / 2, for q, k
                  and v alike.
Why permutations: (1) the projection averages weight entries in pairs, so it is exact in floating point (the two
entries of a pair get bit-identical values); (2) AdamW acts entry by entry, so it commutes with the action: with a
projected gradient the update is exactly equivariant, and "tied" training (project the gradient, step, project
the weights) is Adam on the shared parameters, up to Adam's epsilon; (3) the timm initialisation is i.i.d. over
channels and units, so which channels are paired does not change the distribution of the initial weights. In
training the canonical projection runs as PermProjector (one gather per tensor, bit for bit equal to project_).
Any orthogonal involution with A 1 = 1 and d- <= d / 2 is this permutation in another orthonormal basis that
fixes the all-ones vector; the basis matters only through the LayerNorm affine and Adam's per-entry scaling, both
of which the permutation basis keeps exactly equivariant. The projected initialisation has a smaller scale (the average of two
i.i.d. entries has half their variance); `init_rescale` multiplies the entries in pairs by sqrt(2) after the
projection so that every entry keeps the variance of the original initialisation (off by default: the plan says
"projected").

Fitted structure (UP-04 refund and heal; `fit_ud_structure`). From a trained unconstrained UD model, one UD pass
over the fit images (mirror.group_stats.collect_group_pass, fit seed 0, ImageNet-100 train): A* =
mirror.m1.fit_span_operator over every residual layer (one operator for the whole stream, each layer normalised to
equal energy, A* 1 = 1, as Mirror-exact), sigma(b) by mutual matching of fc1 outputs (as scripts/measure_groups.py,
top_k 10), P_qk(b) and P_v(b) from mirror.attention.pair_and_fit_heads (head pairing and per-head maps). A dense
A* commutes with no non-constant diagonal, so every LayerNorm affine is first folded into the next linear map
(`fold_norms_`: the function is unchanged; gamma = 1 and beta = 0 afterwards) and gamma stays a scalar.

Breaking shares (PREREG_NEXT_PHASE.md Section 1: ||W - W_eq||^2 / ||W||^2), `breaking_shares`: per block on folded
weights (norm affine folded, as every other measurement in the project) for fc1 and fc2
(mirror.groups.mlp_group_shares) and attention (mirror.groups.project_attn_group); per tensor on the raw
parameters; pooled over every parameter.

Free parameters (`free_param_counts`). A tensor on which the action is the involution g has (N + tr g) / 2 free
entries (the dimension of the +1 eigenspace); tr g is the product of the traces on each axis (a permutation's
trace is its number of fixed points). LayerNorm gamma: the number of connected components above. For the canonical
structure this equals the number of orbits of weight entries (tested by counting distinct values).

Training (`train_sym`; CLI scripts/train_controlled_sym.py) is the CORE-01 loop of mirror.controlled.train, step for
step: the same model builder and seeded initialisation, data on the device, augmentation (condition UD), schedule,
AdamW with the same parameter groups, bf16 and torch.compile, evaluation views, checkpoints (same file format and
names, so mirror.controlled.load_checkpoint_adapter and scripts/measure_groups.py read them) and exact resume.
controlled.train has no per-step hook, so the loop is restated here with three additions:
    sym_init     projected initial weights; no constraint during training
    tied         projected initial weights; every step: project the gradient, AdamW step, project the weights
    refund_heal  a finished unconstrained UD run (CORE-01): norms folded, the fitted structure, the weights
                 projected (the zero-shot refund, logged as epoch 0), then training with the projection kept
                 (by default 1 epoch, peak lr 1e-4 at batch 512, 10% warm-up, cosine to 0)
and a breaking-share row (under the run's own structure) after the initial projection and after every epoch.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import controlled as C
from . import groups as GR
from .utils import log

MODES = ("sym_init", "tied", "refund_heal")
DEFAULT_EXP = {"sym_init": "UP-02", "tied": "UP-04", "refund_heal": "UP-04"}
CONSTRAINED = ("tied", "refund_heal")
SYM_CONDITION = "UD"
STRUCTURE_FORMAT = "mirror-ud-structure-v1"
CATEGORIES = ("embed", "attn", "mlp", "norm", "head")


# ----------------------------------------------------------------------------------------------
# Operators
# ----------------------------------------------------------------------------------------------
class Op:
    """A symmetric orthogonal involution on one axis of length n: a permutation (`perm`, index array with
    perm[perm] = id; as a matrix P[i, perm[i]] = 1) or a dense float64 matrix (`dense`).
    left(M) = Op @ M along the first axis, right(M) = M @ Op along the last axis (Op = Op^T)."""

    def __init__(self, perm=None, dense=None, tol: float = 1e-8):
        if (perm is None) == (dense is None):
            raise ValueError("give exactly one of perm and dense")
        self._cache: dict = {}
        if perm is not None:
            p = np.asarray(perm, dtype=np.int64)
            if p.ndim != 1 or not np.array_equal(p[p], np.arange(len(p))):
                raise ValueError("perm must be an involution")
            self.perm, self.dense = p, None
        else:
            M = np.asarray(dense, dtype=np.float64)
            if M.ndim != 2 or M.shape[0] != M.shape[1]:
                raise ValueError("dense operator must be square")
            if np.abs(M - M.T).max() > tol or np.abs(M @ M - np.eye(len(M))).max() > tol:
                raise ValueError("dense operator must be a symmetric orthogonal involution")
            self.perm, self.dense = None, 0.5 * (M + M.T)

    @classmethod
    def identity(cls, n: int) -> "Op":
        return cls(perm=np.arange(int(n)))

    @property
    def n(self) -> int:
        return int(len(self.perm) if self.perm is not None else self.dense.shape[0])

    @property
    def is_perm(self) -> bool:
        return self.perm is not None

    @property
    def trace(self) -> int:
        if self.perm is not None:
            return int((self.perm == np.arange(self.n)).sum())
        return int(round(float(np.trace(self.dense))))

    def matrix(self) -> np.ndarray:
        if self.dense is not None:
            return self.dense.copy()
        M = np.zeros((self.n, self.n))
        M[np.arange(self.n), self.perm] = 1.0
        return M

    def _t(self, device) -> torch.Tensor:
        key = str(device)
        if key not in self._cache:
            if self.perm is not None:
                self._cache[key] = torch.as_tensor(self.perm, dtype=torch.long, device=device)
            else:
                self._cache[key] = torch.as_tensor(self.dense, dtype=torch.float64, device=device)
        return self._cache[key]

    def left(self, M: torch.Tensor) -> torch.Tensor:
        t = self._t(M.device)
        if self.perm is not None:
            return M.index_select(0, t)
        shape = M.shape
        return (t.to(M.dtype) @ M.reshape(shape[0], -1)).reshape(shape)

    def right(self, M: torch.Tensor) -> torch.Tensor:
        t = self._t(M.device)
        if self.perm is not None:
            return M.index_select(M.dim() - 1, t)
        return M @ t.to(M.dtype)

    def components(self) -> np.ndarray:
        """Labels of the connected components of the support of the matrix (orbits for a permutation)."""
        if self.perm is not None:
            return np.minimum(np.arange(self.n), self.perm)
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        M = np.abs(self.dense)
        _, lab = connected_components(csr_matrix(M > 1e-9 * M.max()), directed=False)
        return lab

    def diag_project(self, g: torch.Tensor) -> torch.Tensor:
        """Nearest diag(g) that commutes with the operator: the mean of g over each component."""
        if self.perm is not None:
            return 0.5 * (g + g.index_select(0, self._t(g.device)))
        key = ("comp", str(g.device))
        if key not in self._cache:
            lab = self.components()
            self._cache[key] = (torch.as_tensor(lab, dtype=torch.long, device=g.device), int(lab.max()) + 1)
        lab, k = self._cache[key]
        s = torch.zeros(k, dtype=g.dtype, device=g.device).index_add_(0, lab, g)
        cnt = torch.bincount(lab, minlength=k).to(g.dtype)
        return (s / cnt)[lab]

    def n_components(self) -> int:
        return int(len(np.unique(self.components())))

    def describe(self) -> dict:
        n_plus = (self.n + self.trace) // 2
        out = {"n": self.n, "kind": "permutation" if self.is_perm else "dense", "trace": self.trace,
               "n_plus": n_plus, "n_minus": self.n - n_plus}
        if self.is_perm:
            out["n_pairs"] = int((self.perm != np.arange(self.n)).sum() // 2)
            out["n_fixed"] = self.trace
        return out


def pair_perm(n: int, n_pairs: int) -> np.ndarray:
    """Involution on n items swapping i and i + n_pairs for i < n_pairs (the rest fixed)."""
    n, k = int(n), int(n_pairs)
    if k < 0 or 2 * k > n:
        raise ValueError(f"cannot form {k} pairs from {n} items")
    p = np.arange(n)
    a = np.arange(k)
    p[a], p[a + k] = a + k, a
    return p


def within_head_perm(H: int, dh: int) -> np.ndarray:
    """Head-block permutation on H x dh: inside every head, dimension j <-> j + dh // 2."""
    base = pair_perm(dh, dh // 2)
    return np.concatenate([h * dh + base for h in range(H)])


# ----------------------------------------------------------------------------------------------
# Structure
# ----------------------------------------------------------------------------------------------
@dataclass
class UDStructure:
    """The UD action on one controlled ViT: channel operator A, per block hidden, qk and v operators."""

    kind: str                      # "canonical" or "fitted"
    A: Op
    hidden: list                   # L Ops (permutations) on the h MLP units
    qk: list                       # L Ops on the D = H dh query / key dimensions
    v: list                        # L Ops on the value dimensions
    grid: int
    patch: int
    info: dict = field(default_factory=dict)

    @property
    def depth(self) -> int:
        return len(self.hidden)

    @property
    def tokens(self) -> np.ndarray:
        """Token action including the CLS token (index 0, fixed)."""
        return np.concatenate([[0], 1 + GR.token_index("UD", int(self.grid))]).astype(np.int64)

    @property
    def token_op(self) -> Op:
        return Op(perm=self.tokens)

    def describe(self) -> dict:
        return {"kind": self.kind, "grid": self.grid, "patch": self.patch, "A": self.A.describe(),
                "hidden": [o.describe() for o in self.hidden], "qk": [o.describe() for o in self.qk],
                "v": [o.describe() for o in self.v], "info": self.info}


def _dims(model: nn.Module) -> dict:
    from .attention import attn_dims

    blk = model.blocks[0]
    H, dh = attn_dims(blk)
    return {"d": int(blk.mlp.fc1.in_features), "h": int(blk.mlp.fc1.out_features), "H": H, "dh": dh,
            "L": len(model.blocks), "patch": int(model.patch_embed.patch_size[0]),
            "grid": int(model.patch_embed.grid_size[0])}


def canonical_structure(model: nn.Module, d_minus: int = 0, hidden_pairs: float = 1.0) -> UDStructure:
    """The fixed canonical UD structure (module docstring). d_minus 0 means d // 2."""
    m = _dims(model)
    k = int(d_minus) if int(d_minus) > 0 else m["d"] // 2
    n_h = int(round(float(hidden_pairs) * m["h"] / 2.0))
    A = Op(perm=pair_perm(m["d"], k))
    hid = Op(perm=pair_perm(m["h"], n_h))
    head = Op(perm=within_head_perm(m["H"], m["dh"]))
    return UDStructure("canonical", A, [hid] * m["L"], [head] * m["L"], [head] * m["L"], m["grid"], m["patch"],
                       {"d_minus": k, "d_plus": m["d"] - k, "hidden_pairs": n_h, "hidden_fixed": m["h"] - 2 * n_h,
                        "heads": "every head maps to itself; dimension j <-> j + dh/2 inside each head",
                        "channels": f"channel i <-> i + {k} for i < {k}",
                        "hidden": f"unit i <-> i + {n_h} for i < {n_h}"})


@torch.no_grad()
def fit_ud_structure(model: nn.Module, arch: str, loader, device, top_k: int = 10, workers: int = 1) -> UDStructure:
    """Fitted UD structure of a trained controlled ViT (module docstring). The model is not changed."""
    from .attention import pair_and_fit_heads
    from .group_stats import attn_stats_from_numpy, collect_group_pass
    from .m1 import fit_span_operator
    from .matching import match_many

    t0 = time.time()
    m = _dims(model)
    adapter = C.ControlledAdapter(copy.deepcopy(model).to(device), arch)
    L = m["L"]
    gp = collect_group_pass(adapter, loader, device, "UD", list(range(L)), residual=True, mlp=True, attn=True,
                            inputs=False)
    res = gp.residual
    if list(gp.layers) != list(range(L + 1)):
        raise RuntimeError(f"unexpected residual layers {gp.layers}")
    A, a_info = fit_span_operator(res["C"], list(range(L + 1)), res["G"])
    ms = match_many([gp.R[b] for b in range(L)], top_k=int(top_k), workers=int(workers))
    ast = attn_stats_from_numpy(gp.attn)
    qk, v, heads = [], [], []
    for b in range(L):
        hm = pair_and_fit_heads(ast.moments(b))
        hm.check(1e-8)
        qk.append(Op(dense=GR.head_rep(hm.sigma, hm.B)))
        v.append(Op(dense=GR.head_rep(hm.sigma, hm.C)))
        heads.append({"sigma": [int(s) for s in hm.sigma], "n_pairs": hm.n_pairs, "n_self": hm.n_self})
    info = {"d_plus": int(a_info["d_plus"]), "d_minus": int(a_info["d_minus"]), "n_fit_images": int(gp.n_images),
            "top_k": int(top_k), "operator": "mirror.m1.fit_span_operator over layers 0..L (energy-normalised)",
            "hidden_pairs": [int(r.n_pairs) for r in ms], "hidden_fixed": [int(r.n_self) for r in ms],
            "heads": heads, "seconds": round(time.time() - t0, 1)}
    del adapter
    return UDStructure("fitted", Op(dense=A), [Op(perm=r.perm) for r in ms], qk, v, m["grid"], m["patch"], info)


def _op_arrays(prefix: str, op: Op) -> dict:
    return {f"{prefix}_perm": op.perm} if op.is_perm else {f"{prefix}_dense": op.dense}


def _op_from(z, prefix: str) -> Op:
    if f"{prefix}_perm" in z:
        return Op(perm=z[f"{prefix}_perm"])
    return Op(dense=z[f"{prefix}_dense"])


def save_structure(path, st: UDStructure, extra: dict | None = None) -> Path:
    path = Path(path)
    arrays = _op_arrays("A", st.A)
    for b in range(st.depth):
        arrays.update(_op_arrays(f"hid{b}", st.hidden[b]))
        arrays.update(_op_arrays(f"qk{b}", st.qk[b]))
        arrays.update(_op_arrays(f"v{b}", st.v[b]))
    meta = {"format": STRUCTURE_FORMAT, "kind": st.kind, "grid": st.grid, "patch": st.patch, "depth": st.depth,
            "info": st.info, "extra": extra or {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, meta_json=np.asarray(json.dumps(meta, default=str)), **arrays)
    tmp.replace(path)
    return path


def load_structure(path) -> tuple[UDStructure, dict]:
    with np.load(path) as z:
        meta = json.loads(str(z["meta_json"]))
        if meta.get("format") != STRUCTURE_FORMAT:
            raise ValueError(f"{path} is not a UD structure file")
        L = int(meta["depth"])
        st = UDStructure(meta["kind"], _op_from(z, "A"), [_op_from(z, f"hid{b}") for b in range(L)],
                         [_op_from(z, f"qk{b}") for b in range(L)], [_op_from(z, f"v{b}") for b in range(L)],
                         int(meta["grid"]), int(meta["patch"]), meta.get("info", {}))
    return st, meta.get("extra", {})


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------
# Folding
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def fold_norms_(model: nn.Module) -> None:
    """Fold every LayerNorm affine into the next linear map, in place (norm1 into qkv, norm2 into fc1, the final
    norm into the head); afterwards every gamma is 1 and every beta 0, and the function is unchanged."""
    from .attention import absorb_attn
    from .surgery import absorb_block

    for blk in model.blocks:
        absorb_attn(blk)
        absorb_block(blk)
    norm, head = model.norm, model.head
    if not isinstance(model.fc_norm, nn.Identity):
        raise ValueError("fc_norm must be Identity (CLS pooling)")
    g, b = norm.weight.double(), norm.bias.double()
    W = head.weight.double()
    head.bias.copy_((head.bias.double() + W @ b).to(head.bias))
    head.weight.copy_((W * g[None, :]).to(head.weight))
    norm.weight.fill_(1.0)
    norm.bias.zero_()


# ----------------------------------------------------------------------------------------------
# Projection
# ----------------------------------------------------------------------------------------------
@dataclass
class TensorAction:
    name: str
    param: nn.Parameter
    category: str
    fn: object                      # float64 tensor -> projected float64 tensor
    trace: int | None               # trace of the involution on the tensor (None: LayerNorm gamma)
    n_free_special: int | None = None


def _avg(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + y)


def tensor_actions(model: nn.Module, st: UDStructure) -> list:
    """Every parameter of the controlled ViT with its projection and the trace of the action on it."""
    m = _dims(model)
    if m["L"] != st.depth or m["grid"] != st.grid or m["patch"] != st.patch:
        raise ValueError("structure does not match the model")
    A, tr_A = st.A, st.A.trace
    tok = st.token_op
    out = []

    def add(name, p, cat, fn, trace, special=None):
        out.append(TensorAction(name, p, cat, fn, trace, special))

    def norm_pair(prefix, norm, cat="norm"):
        add(f"{prefix}.weight", norm.weight, cat, A.diag_project, None, A.n_components())
        add(f"{prefix}.bias", norm.bias, cat, lambda x: _avg(x, A.left(x)), tr_A)

    p = model.patch_embed.proj
    c_in, ps = int(p.weight.shape[1]), int(p.weight.shape[-1])
    add("cls_token", model.cls_token, "embed", lambda x: _avg(x, A.right(x)), tr_A)
    add("pos_embed", model.pos_embed, "embed",
        lambda x: _avg(x, A.right(x.index_select(1, tok._t(x.device)))), tok.trace * tr_A)
    add("patch_embed.proj.weight", p.weight, "embed", lambda x: _avg(x, A.left(x.flip(-2))),
        tr_A * c_in * (ps % 2) * ps)
    add("patch_embed.proj.bias", p.bias, "embed", lambda x: _avg(x, A.left(x)), tr_A)
    for b, blk in enumerate(model.blocks):
        hid, qk, v = st.hidden[b], st.qk[b], st.v[b]
        D = qk.n
        pre = f"blocks.{b}"
        norm_pair(f"{pre}.norm1", blk.norm1)

        def qkv_w(x, qk=qk, v=v, D=D):
            q, k, vv = x[:D], x[D:2 * D], x[2 * D:]
            return torch.cat([_avg(q, qk.left(A.right(q))), _avg(k, qk.left(A.right(k))),
                              _avg(vv, v.left(A.right(vv)))], 0)

        def qkv_b(x, qk=qk, v=v, D=D):
            q, k, vv = x[:D], x[D:2 * D], x[2 * D:]
            return torch.cat([_avg(q, qk.left(q)), _avg(k, qk.left(k)), _avg(vv, v.left(vv))], 0)

        add(f"{pre}.attn.qkv.weight", blk.attn.qkv.weight, "attn", qkv_w, (2 * qk.trace + v.trace) * tr_A)
        add(f"{pre}.attn.qkv.bias", blk.attn.qkv.bias, "attn", qkv_b, 2 * qk.trace + v.trace)
        add(f"{pre}.attn.proj.weight", blk.attn.proj.weight, "attn", lambda x, v=v: _avg(x, A.left(v.right(x))),
            tr_A * v.trace)
        add(f"{pre}.attn.proj.bias", blk.attn.proj.bias, "attn", lambda x: _avg(x, A.left(x)), tr_A)
        norm_pair(f"{pre}.norm2", blk.norm2)
        add(f"{pre}.mlp.fc1.weight", blk.mlp.fc1.weight, "mlp", lambda x, hid=hid: _avg(x, hid.left(A.right(x))),
            hid.trace * tr_A)
        add(f"{pre}.mlp.fc1.bias", blk.mlp.fc1.bias, "mlp", lambda x, hid=hid: _avg(x, hid.left(x)), hid.trace)
        add(f"{pre}.mlp.fc2.weight", blk.mlp.fc2.weight, "mlp", lambda x, hid=hid: _avg(x, A.left(hid.right(x))),
            tr_A * hid.trace)
        add(f"{pre}.mlp.fc2.bias", blk.mlp.fc2.bias, "mlp", lambda x: _avg(x, A.left(x)), tr_A)
    norm_pair("norm", model.norm)
    n_cls = int(model.head.weight.shape[0])
    add("head.weight", model.head.weight, "head", lambda x: _avg(x, A.right(x)), n_cls * tr_A)
    add("head.bias", model.head.bias, "head", lambda x: x, n_cls)
    names = {a.name for a in out}
    missing = [n for n, _ in model.named_parameters() if n not in names]
    if missing:
        raise ValueError(f"parameters without a UD action: {missing}")
    return out


@torch.no_grad()
def project_(model: nn.Module, st: UDStructure, target: str = "param", actions: list | None = None) -> None:
    """Project the weights (target "param") or their gradients ("grad") onto the UD-equivariant subspace, in
    place. Computed in float64, rounded once to the parameter dtype."""
    for a in actions if actions is not None else tensor_actions(model, st):
        t = a.param if target == "param" else a.param.grad
        if t is None:
            continue
        t.copy_(a.fn(t.detach().to(torch.float64)).to(t.dtype))


def is_permutation_structure(st: UDStructure) -> bool:
    return st.A.is_perm and all(o.is_perm for o in list(st.hidden) + list(st.qk) + list(st.v))


class PermProjector:
    """Fast projection for a structure whose action permutes weight entries (the canonical one): per tensor, the
    flat index map of g is precomputed, and the projection is x <- (x + x[g]) / 2 with one gather per tensor and
    two multi-tensor kernels. Bit for bit equal to project_ (rounding to float32 commutes with the halving)."""

    def __init__(self, actions: list):
        self.items = []
        for a in actions:
            p = a.param
            ids = torch.arange(p.numel(), dtype=torch.float64, device=p.device).reshape(p.shape)
            g = (2.0 * a.fn(ids) - ids).reshape(-1)
            gi = g.round().long()
            if not torch.equal(gi.double(), g) or not torch.equal(gi[gi], torch.arange(p.numel(), device=p.device)):
                raise ValueError(f"{a.name}: the action is not a permutation of entries")
            if torch.equal(gi, torch.arange(p.numel(), device=p.device)):
                continue                                   # identity on this tensor (the head bias)
            self.items.append((p, gi))

    @torch.no_grad()
    def __call__(self, target: str = "param") -> None:
        ts, gs = [], []
        for p, gi in self.items:
            t = p if target == "param" else p.grad
            if t is None:
                continue
            ts.append(t)
            gs.append(t.reshape(-1).index_select(0, gi).view_as(t))
        if ts:
            torch._foreach_add_(ts, gs)
            torch._foreach_mul_(ts, 0.5)


def make_projector(model: nn.Module, st: UDStructure, actions: list | None = None):
    """callable(target) projecting the weights ("param") or gradients ("grad") in place: PermProjector for
    permutation structures, project_ otherwise."""
    actions = actions if actions is not None else tensor_actions(model, st)
    if is_permutation_structure(st):
        return PermProjector(actions)
    return lambda target="param": project_(model, st, target, actions)


@torch.no_grad()
def restore_init_scale_(model: nn.Module, st: UDStructure) -> None:
    """After a projection: multiply every entry that sits in a pair (not a fixed point of the action) by sqrt(2),
    so it keeps the variance of the i.i.d. initialisation. Canonical (permutation) structures only."""
    if st.kind != "canonical":
        raise ValueError("restore_init_scale_ needs the canonical structure")
    for a in tensor_actions(model, st):
        if a.trace is None:          # LayerNorm gamma: ones at initialisation
            continue
        ids = torch.arange(a.param.numel(), dtype=torch.float64, device=a.param.device).reshape(a.param.shape)
        fixed = a.fn(ids) == ids
        a.param.mul_(torch.where(fixed, 1.0, math.sqrt(2.0)).to(a.param.dtype))


def free_param_counts(model: nn.Module, st: UDStructure) -> dict:
    """Free parameters of the UD-equivariant subspace, per tensor and category, against the dense count."""
    per, cat = {}, {c: {"dense": 0, "free": 0} for c in CATEGORIES}
    for a in tensor_actions(model, st):
        N = int(a.param.numel())
        if a.trace is None:
            free = int(a.n_free_special)
        else:
            if (N + a.trace) % 2:
                raise ValueError(f"{a.name}: N + trace is odd")
            free = (N + int(a.trace)) // 2
        per[a.name] = {"dense": N, "free": free}
        cat[a.category]["dense"] += N
        cat[a.category]["free"] += free
    dense = sum(v["dense"] for v in cat.values())
    free = sum(v["free"] for v in cat.values())
    return {"dense": dense, "free": free, "free_frac": free / dense, "by_category": cat, "per_tensor": per}


# ----------------------------------------------------------------------------------------------
# Breaking shares
# ----------------------------------------------------------------------------------------------
def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else float("nan")


@torch.no_grad()
def breaking_shares(model: nn.Module, st: UDStructure, first_blocks: int = 6) -> dict:
    """UD breaking shares under the structure `st` (module docstring). Keys: s_all (pooled over every raw
    parameter), s_<category>_raw, s_max_tensor and max_tensor, s_mlp_b1_<k> and s_mlp_all (mean over blocks of
    (s_fc1 + s_fc2) / 2 on folded weights), s_attn_b1_<k>, s_attn_all, and per block s_fc1_bN, s_fc2_bN,
    s_attn_bN (N from 1)."""
    from .attention import fold_attn
    from .surgery import fold_mlp

    out = {}
    num_c = {c: 0.0 for c in CATEGORIES}
    den_c = {c: 0.0 for c in CATEGORIES}
    s_max, n_max = -1.0, ""
    for a in tensor_actions(model, st):
        x = a.param.detach().to(torch.float64)
        num = float(((x - a.fn(x)) ** 2).sum())
        den = float((x ** 2).sum())
        num_c[a.category] += num
        den_c[a.category] += den
        if den > 0 and num / den > s_max:
            s_max, n_max = num / den, a.name
    for c in CATEGORIES:
        out[f"s_{c}_raw"] = _ratio(num_c[c], den_c[c])
    out["s_all"] = _ratio(sum(num_c.values()), sum(den_c.values()))
    out["s_max_tensor"], out["max_tensor"] = max(s_max, 0.0), (n_max if s_max > 0 else "")
    A_np = st.A.matrix()
    mlp, attn = [], []
    for b, blk in enumerate(model.blocks):
        act = GR.BlockAction("UD", [None, A_np], [None, A_np], [None, st.hidden[b].perm])
        sh = GR.mlp_group_shares(fold_mlp(blk), act)
        f = fold_attn(blk)
        aact = GR.AttnAction("UD", [None, A_np], [None, A_np], [None, st.qk[b].matrix()], [None, st.v[b].matrix()])
        ash = GR.attn_shares(f, GR.project_attn_group(f, aact))
        out[f"s_fc1_b{b + 1}"], out[f"s_fc2_b{b + 1}"], out[f"s_attn_b{b + 1}"] = sh["s_fc1"], sh["s_fc2"], ash["s_attn"]
        mlp.append(sh["s_mlp"])
        attn.append(ash["s_attn"])
    k = min(int(first_blocks), len(mlp))
    out[f"s_mlp_b1_{k}"] = float(np.mean(mlp[:k]))
    out["s_mlp_all"] = float(np.mean(mlp))
    out[f"s_attn_b1_{k}"] = float(np.mean(attn[:k]))
    out["s_attn_all"] = float(np.mean(attn))
    return out


def share_columns(depth: int, first_blocks: int = 6) -> list:
    k = min(int(first_blocks), int(depth))
    cols = ["epoch", "step", "s_all", "s_max_tensor", "max_tensor"] + [f"s_{c}_raw" for c in CATEGORIES]
    cols += [f"s_mlp_b1_{k}", "s_mlp_all", f"s_attn_b1_{k}", "s_attn_all"]
    for b in range(1, int(depth) + 1):
        cols += [f"s_fc1_b{b}", f"s_fc2_b{b}", f"s_attn_b{b}"]
    return cols


@torch.no_grad()
def ud_function_gap(model: nn.Module, x: torch.Tensor) -> dict:
    """Gap between the logits of UD(x) and those of x (zero, to rounding, for an exactly UD-equivariant model)."""
    was = model.training
    model.eval()
    try:
        a, b = model(x), model(torch.flip(x, dims=[-2]))
    finally:
        model.train(was)
    return {"logit_rel_gap": float((a - b).norm() / a.norm().clamp_min(1e-300)),
            "logit_max_abs_gap": float((a - b).abs().max())}


# ----------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------
@dataclass
class SymTrainConfig(C.TrainConfig):
    exp: str = "UP-02"
    condition: str = SYM_CONDITION
    mode: str = "sym_init"
    d_minus: int = 0                     # canonical channel action: swapped pairs (0: d // 2)
    hidden_pairs: float = 1.0            # canonical hidden action: share of MLP units in pairs
    init_rescale: bool = False           # canonical: restore the initialisation variance after projecting
    project_grads: bool = True           # constrained modes: project the gradient before the step
    source_ckpt: str | None = None       # refund_heal: final checkpoint of an unconstrained UD run
    fit_seed: int = 0
    fit_images: int = 2000
    fit_batch: int = 64
    top_k: int = 10
    match_workers: int = 1
    first_blocks: int = 6
    results_root: str = "results"


SYM_RESUME_KEYS = ("mode", "d_minus", "hidden_pairs", "init_rescale", "project_grads", "source_ckpt", "fit_seed",
                   "fit_images", "top_k")
SHARE_FILE = "breaking_share.csv"


def run_name(cfg: SymTrainConfig) -> str:
    return f"{cfg.arch}_{cfg.condition}_{cfg.mode}_s{cfg.seed}"


def sym_run_dir(cfg: SymTrainConfig) -> Path:
    return Path(cfg.run_dir) if cfg.run_dir else Path(cfg.out_root) / cfg.exp / run_name(cfg)


def results_dir(cfg: SymTrainConfig) -> Path:
    return Path(cfg.results_root) / cfg.exp / run_name(cfg)


def final_checkpoint(run_dir, require_complete: bool = True) -> Path:
    """Last ckpt_e<NNN>.pt of a controlled-training run (the run must be complete unless told otherwise)."""
    run_dir = Path(run_dir)
    cks = C.list_checkpoints(run_dir)
    if not cks:
        raise FileNotFoundError(f"no checkpoints in {run_dir}")
    if require_complete:
        meta = run_dir / "meta.json"
        done = (run_dir / "last_done.flag").exists() or (
            meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get("extra", {}).get("status") == "complete")
        if not done:
            raise RuntimeError(f"{run_dir} is not finished (no last_done.flag, meta status not complete)")
    return cks[-1][1]


def _check_cfg(cfg: SymTrainConfig) -> None:
    if cfg.mode not in MODES:
        raise ValueError(f"unknown mode {cfg.mode!r}; choose from {MODES}")
    if cfg.condition != SYM_CONDITION:
        raise ValueError("UP-02 and UP-04 use the UD condition only")
    if cfg.mode == "refund_heal" and not cfg.source_ckpt:
        raise ValueError("refund_heal needs source_ckpt (the final checkpoint of an unconstrained UD run)")
    if cfg.drop_path != 0.0:
        raise ValueError("drop_path must be 0 (CORE-01 recipe)")


def _sym_view(cfg: SymTrainConfig) -> dict:
    d = {k: getattr(cfg, k) for k in SYM_RESUME_KEYS}
    if d["source_ckpt"]:
        d["source_ckpt"] = str(Path(d["source_ckpt"]).resolve())
    return json.loads(json.dumps(d))


def _prepare_structure(cfg: SymTrainConfig, model: nn.Module, run_dir: Path, device, fresh: bool):
    """(structure, info). refund_heal on a fresh start also loads the source weights, folds the norms and fits
    (or reloads) the structure; the weights are projected by the caller."""
    st_path = run_dir / "structure.npz"
    if cfg.mode != "refund_heal":
        return canonical_structure(model, cfg.d_minus, cfg.hidden_pairs), {}
    src_path = Path(cfg.source_ckpt)
    want = {"source_ckpt": str(src_path.resolve()), "fit_seed": int(cfg.fit_seed), "fit_images": int(cfg.fit_images),
            "top_k": int(cfg.top_k)}
    src_info = {}
    if fresh:
        src = C.load_checkpoint(src_path)
        if src["arch"] != cfg.arch or int(src["num_classes"]) != int(model.num_classes):
            raise ValueError(f"source checkpoint {src_path} is {src['arch']} with {src['num_classes']} classes")
        model.load_state_dict(src["model"], strict=True)
        fold_norms_(model)
        src_info = {"source_ckpt": str(src_path), "source_sha256": file_sha256(src_path),
                    "source_condition": src.get("condition"), "source_seed": src.get("seed"),
                    "source_epoch": src.get("epoch"), "source_val": src.get("val")}
        del src
    if st_path.exists():
        st, extra = load_structure(st_path)
        if all(extra.get(k) == v for k, v in want.items()):
            return st, {**extra, **src_info}
        if not fresh:
            raise ValueError(f"{st_path} was fitted with other settings; use --force-new")
    if not fresh:
        raise FileNotFoundError(f"{st_path} missing for a resumed refund_heal run; use --force-new")
    loader = C.fit_loader(cfg.fit_seed, cfg.fit_images, batch_size=cfg.fit_batch, cache_dir=cfg.cache_dir)
    log(f"fitting the UD structure of {src_path} on {cfg.fit_images} fit images (fit seed {cfg.fit_seed})")
    model.eval()
    st = fit_ud_structure(model, cfg.arch, loader, device, top_k=cfg.top_k, workers=cfg.match_workers)
    model.train()
    extra = {**want, **src_info}
    save_structure(st_path, st, extra)
    log(f"structure: d+ {st.info['d_plus']}, d- {st.info['d_minus']}, hidden pairs {st.info['hidden_pairs']}, "
        f"{st.info['seconds']} s")
    return st, extra


def train_sym(cfg: SymTrainConfig) -> dict:
    """Train one UP-02 / UP-04 run (module docstring). Resumes from <run_dir>/last.pt when present.

    Outputs in run_dir (as mirror.controlled.train): ckpt_e<NNN>.pt, last.pt, train_log.csv, val_acc.csv,
    meta.json, plus breaking_share.csv (one row after the initial projection and one after every epoch) and,
    for refund_heal, structure.npz. On completion results/<exp>/<run>/ gets summary.json, breaking_share.csv
    and meta.json."""
    from .heal import lr_at
    from .meta import write_meta

    session_t0 = time.time()
    _check_cfg(cfg)
    device = torch.device(cfg.device)
    amp, use_compile = C._resolve(cfg, device)
    run_dir = sym_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # ---- data (as controlled.train) ----------------------------------------------------------
    data = C.load_in100(cfg.cache_dir)
    tr_lab_np, va_lab_np = data["train_labels"], data["val_labels"]
    num_classes = int(max(tr_lab_np.max(), va_lab_np.max()) + 1)
    n_train, B = int(len(tr_lab_np)), int(cfg.batch_size)
    spe = n_train // B
    if spe < 1:
        raise ValueError(f"batch {B} is larger than the training set ({n_train})")
    total_steps = spe * int(cfg.epochs)
    warmup_steps = int(round(float(cfg.warmup_epochs) * spe))
    base_lr = float(cfg.lr) * B / float(cfg.lr_ref_batch)
    ckpt_set = C.ckpt_epochs_for(cfg.epochs, cfg.ckpt_epochs)
    data_fp = {"n_train": n_train, "n_val": int(len(va_lab_np)),
               "train_labels_sha": C.labels_fingerprint(tr_lab_np), "val_labels_sha": C.labels_fingerprint(va_lab_np),
               "size": int(data["train_images"].shape[1])}
    resume_view = C._resume_view(cfg, amp, data_fp)
    resume_view["sym"] = _sym_view(cfg)

    # ---- files -------------------------------------------------------------------------------
    last_path = run_dir / "last.pt"
    train_csv = C._Csv(run_dir / "train_log.csv", C.TRAIN_LOG_COLUMNS)
    val_csv = C._Csv(run_dir / "val_acc.csv", C.VAL_COLUMNS)
    meta_path = run_dir / "meta.json"
    if cfg.force_new:
        for p in [last_path, meta_path, train_csv.path, val_csv.path, run_dir / SHARE_FILE, run_dir / "structure.npz",
                  run_dir / "last_done.flag"] + [p for _, p in C.list_checkpoints(run_dir)]:
            if p.exists():
                p.unlink()
    fresh = not last_path.exists()
    if fresh and meta_path.exists() and \
            json.loads(meta_path.read_text()).get("extra", {}).get("status") == "complete":
        raise RuntimeError(f"{run_dir} holds a completed run but no last.pt; use --force-new to redo it")

    # ---- model, structure, starting weights --------------------------------------------------
    torch.manual_seed(int(cfg.seed))
    model = C.build_model(cfg.arch, num_classes=num_classes, img_size=cfg.img_size,
                          drop_path_rate=cfg.drop_path).to(device)
    model.train()
    st, st_extra = _prepare_structure(cfg, model, run_dir, device, fresh)
    actions = tensor_actions(model, st)
    if fresh:
        project_(model, st, "param", actions)
        if cfg.init_rescale and cfg.mode != "refund_heal":
            restore_init_scale_(model, st)
    counts = free_param_counts(model, st)
    share_cols = share_columns(len(model.blocks), cfg.first_blocks)
    share_csv = C._Csv(run_dir / SHARE_FILE, share_cols)
    constrained = cfg.mode in CONSTRAINED
    projector = make_projector(model, st, actions) if constrained else None

    params = [p for p in model.parameters() if p.requires_grad]
    opt_kw = {"lr": base_lr, "betas": tuple(cfg.betas)}
    if device.type == "cuda":
        opt_kw["fused"] = True
    opt = torch.optim.AdamW(C.param_groups(model, cfg.weight_decay), **opt_kw)
    ckpt_common = {"format": C.CKPT_FORMAT, "exp": cfg.exp, "arch": cfg.arch, "condition": cfg.condition,
                   "condition_spec": C.CONDITIONS[cfg.condition], "seed": int(cfg.seed), "img_size": int(cfg.img_size),
                   "patch_size": C.PATCH_SIZE, "num_classes": num_classes, "drop_path_rate": float(cfg.drop_path),
                   "store_size": data_fp["size"], "config": resume_view, "sym_mode": cfg.mode,
                   "sym_structure": st.kind}

    # ---- resume or fresh start ---------------------------------------------------------------
    step, wall_before, accum = 0, 0.0, None
    if not fresh:
        state = torch.load(last_path, map_location=device, weights_only=True)
        if state.get("format") != C.CKPT_FORMAT or state.get("kind") != "last":
            raise ValueError(f"{last_path} is not a controlled-training last state")
        if state["config"] != resume_view:
            diff = sorted(k for k in set(state["config"]) | set(resume_view)
                          if state["config"].get(k) != resume_view.get(k))
            raise ValueError(f"{last_path} was written with different settings ({diff}); "
                             "use the same settings, or --force-new to start over")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"]["torch"].cpu())
        if device.type == "cuda" and state["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["rng"]["cuda"]])
        step = int(state["step"])
        wall_before = float(state.get("wall_seconds", 0.0))
        accum = state.get("epoch_accum")
        done = step // spe
        train_csv.reset(keep=lambda r: int(r["epoch"]) <= done)
        val_csv.reset(keep=lambda r: int(r["epoch"]) <= done)
        share_csv.reset(keep=lambda r: int(r["epoch"]) <= done)
        log(f"resumed {run_dir} at step {step} of {total_steps} (epoch {done} done)")
        if step >= total_steps:
            log("run already complete")
            return _finish_sym(cfg, run_dir, train_csv, val_csv, share_csv, wall_before, device, amp, use_compile,
                               spe, total_steps, ckpt_set, model, data_fp, base_lr, warmup_steps, st, st_extra,
                               counts, write=False)
    else:
        for _, p in C.list_checkpoints(run_dir):
            p.unlink()
        train_csv.reset()
        val_csv.reset()
        share_csv.reset()

    run_extra = {"status": "running", "steps_per_epoch": spe, "total_steps": total_steps, "ckpt_epochs": ckpt_set,
                 "data": data_fp, "amp": amp, "compile": use_compile, "base_lr": base_lr,
                 "warmup_steps": warmup_steps, "sym_mode": cfg.mode, "structure": st.describe(),
                 "structure_source": st_extra, "param_counts": {k: v for k, v in counts.items() if k != "per_tensor"}}
    write_meta(run_dir, experiment=cfg.exp, config=asdict(cfg), seeds=C._seeds(cfg), models=C._models(cfg, model),
               device=device, wall_seconds=wall_before, extra=run_extra)

    # ---- tensors on the device ---------------------------------------------------------------
    tr_u8 = torch.from_numpy(np.array(data["train_images"], dtype=np.uint8)).to(device)
    tr_y = torch.from_numpy(np.ascontiguousarray(tr_lab_np, dtype=np.int64)).to(device)
    va_u8 = torch.from_numpy(np.array(data["val_images"], dtype=np.uint8)).to(device)
    va_y = torch.from_numpy(np.ascontiguousarray(va_lab_np, dtype=np.int64)).to(device)
    log(f"{cfg.mode}: {spe} steps per epoch, {total_steps} steps, peak lr {base_lr:g}, warm-up {warmup_steps} steps; "
        f"structure {st.kind}, free parameters {counts['free']:,} of {counts['dense']:,}")
    gpu_info = {}
    if device.type == "cuda":
        free_b, total_b = torch.cuda.mem_get_info(device)
        gpu_info = {"gpu_free_gib_at_start": round(free_b / 2**30, 2), "gpu_total_gib": round(total_b / 2**30, 2)}

    compile_error = None
    fwd = model
    if use_compile:
        try:
            fwd = torch.compile(model, dynamic=False)
        except Exception as exc:            # noqa: BLE001
            compile_error = f"{type(exc).__name__}: {exc}"[:500]
            log(f"torch.compile failed ({compile_error}); training in eager mode")
    compiled_ok = fwd is model

    def forward_backward(x, y):
        opt.zero_grad(set_to_none=True)
        with C.autocast_ctx(device, amp):
            logits = fwd(x)
        loss = F.cross_entropy(logits.float(), y, label_smoothing=float(cfg.label_smoothing))
        loss.backward()
        return logits, loss

    def evaluate_and_save(epoch: int) -> None:
        t = time.time()
        acc = C.evaluate_views(model, va_u8, va_y, C.VIEWS, cfg.eval_batch, amp, cfg.img_size)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ev = time.time() - t
        val_csv.append({"epoch": epoch, "step": step, **{k: f"{v:.4f}" if isinstance(v, float) else v
                                                          for k, v in acc.items()}, "eval_seconds": f"{ev:.2f}"})
        C._atomic_save({**ckpt_common, "kind": "checkpoint", "epoch": int(epoch), "step": int(step),
                        "val": acc, "model": C._model_state(model)}, run_dir / C.checkpoint_name(epoch))
        log(f"epoch {epoch}: val top-1 " + ", ".join(f"{v} {acc['top1_' + v]:.2f}" for v in C.VIEWS) + f" ({ev:.1f} s)")

    def track(epoch: int) -> None:
        s = breaking_shares(model, st, cfg.first_blocks)
        share_csv.append({"epoch": epoch, "step": step, **{k: (repr(v) if isinstance(v, float) else v)
                                                           for k, v in s.items()}})
        k = min(int(cfg.first_blocks), len(model.blocks))
        log(f"epoch {epoch}: UD breaking share all {s['s_all']:.3e}, MLP blocks 1-{k} {s[f's_mlp_b1_{k}']:.3e}, "
            f"largest tensor {s['s_max_tensor']:.3e} ({s['max_tensor']})")

    def save_last(acc_state) -> None:
        rng = {"torch": torch.get_rng_state(),
               "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        C._atomic_save({**ckpt_common, "kind": "last", "epoch": step // spe, "step": int(step),
                        "model": C._model_state(model), "optimizer": opt.state_dict(),
                        "scheduler": {"type": "linear warm-up then cosine, per step", "base_lr": base_lr,
                                      "warmup_steps": warmup_steps, "total_steps": total_steps,
                                      "min_lr_ratio": float(cfg.min_lr_ratio), "step": int(step)},
                        "rng": rng, "epoch_accum": acc_state,
                        "wall_seconds": wall_before + time.time() - session_t0}, last_path)

    if step == 0:
        if 0 in ckpt_set:
            evaluate_and_save(0)
        track(0)

    # ---- training loop (controlled.train, plus the projection) --------------------------------
    last_save = time.time()
    while step < total_steps:
        epoch = step // spe
        k0 = step % spe
        perm = torch.from_numpy(C.epoch_permutation(n_train, cfg.seed, epoch)[:spe * B]).to(device)
        crops, sym = C.epoch_aug_params(spe * B, cfg.seed, epoch, cfg.condition, data_fp["size"],
                                        tuple(cfg.crop_scale), tuple(cfg.crop_ratio))
        crops, sym = crops.to(device), sym.to(device)
        if accum is not None and k0 > 0:
            acc_state = {k: v.to(device) for k, v in accum.items()}
        else:
            acc_state = {"loss_sum": torch.zeros((), dtype=torch.float32, device=device),
                         "correct": torch.zeros((), dtype=torch.int64, device=device),
                         "gn_sum": torch.zeros((), dtype=torch.float32, device=device),
                         "n": torch.zeros((), dtype=torch.int64, device=device)}
        accum = None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        t_ep = time.time()
        lr = base_lr
        for k in range(k0, spe):
            sl = slice(k * B, (k + 1) * B)
            idx = perm[sl]
            x = C.augment_train(tr_u8[idx], crops[sl], sym[sl], cfg.img_size, cfg.condition)
            y = tr_y[idx]
            lr = lr_at(step, base_lr, total_steps, warmup_steps, cfg.min_lr_ratio)
            for g in opt.param_groups:
                g["lr"] = lr
            if compiled_ok:
                logits, loss = forward_backward(x, y)
            else:
                try:
                    logits, loss = forward_backward(x, y)
                except torch.cuda.OutOfMemoryError:
                    raise
                except Exception as exc:        # noqa: BLE001
                    compile_error = f"{type(exc).__name__}: {exc}"[:500]
                    log(f"torch.compile failed ({compile_error}); continuing in eager mode")
                    fwd = model
                    logits, loss = forward_backward(x, y)
                compiled_ok = True
            if constrained and cfg.project_grads:
                projector("grad")
            gn = (torch.nn.utils.clip_grad_norm_(params, float(cfg.grad_clip)) if cfg.grad_clip > 0
                  else C._grad_norm(params))
            opt.step()
            if constrained:
                projector("param")
            step += 1
            acc_state["loss_sum"] += loss.detach() * B
            acc_state["correct"] += (logits.detach().argmax(dim=-1) == y).sum()
            acc_state["gn_sum"] += gn.detach().float()
            acc_state["n"] += B
            if cfg.log_every and step % int(cfg.log_every) == 0:
                n_seen = max(int(acc_state["n"]), 1)
                log(f"e {epoch + 1}/{cfg.epochs} step {step}/{total_steps} loss "
                    f"{float(acc_state['loss_sum']) / n_seen:.4f} top1 {100.0 * int(acc_state['correct']) / n_seen:.2f} "
                    f"lr {lr:.3e} gn {float(gn):.3f}")
            stop_now = cfg.stop_after_steps is not None and step >= int(cfg.stop_after_steps)
            due = ((cfg.save_every_steps and step % int(cfg.save_every_steps) == 0)
                   or (cfg.save_every_min and time.time() - last_save >= 60.0 * float(cfg.save_every_min)))
            if (stop_now or due) and step % spe != 0:
                save_last(acc_state)
                last_save = time.time()
            if stop_now:
                if step % spe == 0:
                    break
                log(f"stopped after step {step} (simulated preemption); last state saved")
                return {"stopped": True, "step": step, "run_dir": str(run_dir)}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ep_s = time.time() - t_ep
        e = step // spe
        n_seen = max(int(acc_state["n"]), 1)
        timed = spe - k0
        peak_a = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
        peak_r = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.0
        if e in ckpt_set:
            evaluate_and_save(e)
        track(e)
        train_csv.append({
            "epoch": e, "step": step, "lr_end": f"{lr:.6e}",
            "train_loss": repr(float(acc_state["loss_sum"]) / n_seen),
            "train_top1": repr(100.0 * int(acc_state["correct"]) / n_seen),
            "grad_norm_mean": repr(float(acc_state["gn_sum"]) / (n_seen // B)),
            "epoch_seconds": f"{ep_s:.3f}", "images_per_s": f"{timed * B / max(ep_s, 1e-9):.1f}",
            "peak_mem_alloc_gib": f"{peak_a:.3f}", "peak_mem_reserved_gib": f"{peak_r:.3f}",
            "steps_timed": timed, "resumed": int(k0 > 0),
            "wall_seconds": f"{wall_before + time.time() - session_t0:.1f}"})
        log(f"epoch {e}/{cfg.epochs} done: {ep_s:.1f} s, {timed * B / max(ep_s, 1e-9):.0f} img/s")
        stop_here = cfg.stop_after_steps is not None and step >= int(cfg.stop_after_steps) and step < total_steps
        if (stop_here or e in ckpt_set or e == cfg.epochs
                or (cfg.save_every_epochs and e % int(cfg.save_every_epochs) == 0)
                or (cfg.save_every_min and time.time() - last_save >= 60.0 * float(cfg.save_every_min))):
            save_last(None)
            last_save = time.time()
        if stop_here:
            log(f"stopped after step {step} (simulated preemption); last state saved")
            return {"stopped": True, "step": step, "run_dir": str(run_dir)}

    wall = wall_before + time.time() - session_t0
    return _finish_sym(cfg, run_dir, train_csv, val_csv, share_csv, wall, device, amp, use_compile, spe, total_steps,
                       ckpt_set, model, data_fp, base_lr, warmup_steps, st, st_extra, counts, write=True,
                       compile_error=compile_error, gpu_info=gpu_info)


def _float_rows(rows: list, key: str) -> list:
    out = []
    for r in rows:
        try:
            out.append(float(r[key]))
        except (KeyError, TypeError, ValueError):
            pass
    return out


def _finish_sym(cfg, run_dir, train_csv, val_csv, share_csv, wall, device, amp, use_compile, spe, total_steps,
                ckpt_set, model, data_fp, base_lr, warmup_steps, st, st_extra, counts, write: bool,
                compile_error: str | None = None, gpu_info: dict | None = None) -> dict:
    from .meta import write_meta

    train_rows, val_rows, share_rows = train_csv.rows(), val_csv.rows(), share_csv.rows()
    timing = C.timing_summary(train_rows, val_rows, wall, cfg.epochs, len(ckpt_set))
    k = min(int(cfg.first_blocks), len(model.blocks))
    keys = ["s_all", "s_max_tensor", f"s_mlp_b1_{k}", "s_mlp_all", f"s_attn_b1_{k}", "s_attn_all"]
    shares = {"start": {key: share_rows[0][key] for key in keys} if share_rows else {},
              "end": {key: share_rows[-1][key] for key in keys} if share_rows else {},
              "max_over_epochs": {key: max(_float_rows(share_rows, key), default=float("nan")) for key in keys},
              "n_rows": len(share_rows)}
    val0 = val_rows[0] if val_rows else {}
    summary = {"exp": cfg.exp, "mode": cfg.mode, "run": run_name(cfg), "run_dir": str(run_dir), "arch": cfg.arch,
               "condition": cfg.condition, "seed": int(cfg.seed), "epochs": int(cfg.epochs),
               "final_val": val_rows[-1] if val_rows else {}, "epoch0_val": val0,
               "params_dense": counts["dense"], "params_free": counts["free"], "params_free_frac": counts["free_frac"],
               "params_by_category": counts["by_category"], "breaking_share": shares,
               "structure": st.describe(), "structure_source": st_extra, "timing": timing,
               "trained_parameters": "free (constrained)" if cfg.mode in CONSTRAINED else "dense (unconstrained)"}
    out = {"stopped": False, "run_dir": str(run_dir), "timing": timing, "final_val": summary["final_val"],
           "summary": summary, "checkpoints": [str(p) for _, p in C.list_checkpoints(run_dir)]}
    if write:
        extra = {"status": "complete", "steps_per_epoch": spe, "total_steps": total_steps, "ckpt_epochs": ckpt_set,
                 "data": data_fp, "amp": amp, "compile": use_compile, "base_lr": base_lr, "warmup_steps": warmup_steps,
                 "timing": timing, "final_val": summary["final_val"], "condition_spec": C.CONDITIONS[cfg.condition],
                 "compile_error": compile_error, "compiled": bool(use_compile and compile_error is None),
                 "gpu": gpu_info or {}, "sym_mode": cfg.mode, "structure": st.describe(),
                 "structure_source": st_extra, "param_counts": {k_: v for k_, v in counts.items() if k_ != "per_tensor"},
                 "breaking_share": shares}
        write_meta(run_dir, experiment=cfg.exp, config=asdict(cfg), seeds=C._seeds(cfg), models=C._models(cfg, model),
                   device=device, wall_seconds=wall, extra=extra)
    if write or not (results_dir(cfg) / "summary.json").exists():
        res = results_dir(cfg)
        res.mkdir(parents=True, exist_ok=True)
        (res / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        shutil.copyfile(share_csv.path, res / SHARE_FILE)
        shutil.copyfile(val_csv.path, res / "val_acc.csv")
        write_meta(res, experiment=cfg.exp, config=asdict(cfg), seeds=C._seeds(cfg), models=C._models(cfg, model),
                   device=device, wall_seconds=wall, extra={"run_dir": str(run_dir), "summary": "summary.json"})
        log(f"done: {run_dir}; wall {timing['wall_minutes']} min; final top-1 "
            f"{summary['final_val'].get('top1_orig')}; summary in {res}")
    return out
