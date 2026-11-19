"""Mirror + SVD-LLM (A06, FINAL_PHASE_PLAN.md v2): SVD-LLM applied inside the even and odd blocks of the Mirror-exact
model, ranks allocated globally by whitened energy, parameters matched to a fraction of the full model. The student is
exactly mirror-equivariant (at full conversion exactly flip-invariant) at build time and after any gradient step.

Block structure (mirror/stacking.py, CORE-03, reused here). After Mirror-exact surgery every converted matrix
W [out, in] of the folded model satisfies R_out W = W R_in for two orthogonal symmetric involutions:

    matrix   R_out                              R_in
    fc1      P (neuron pairing)                 A* (span operator)
    fc2      A*                                 P
    wq, wk   P_B (head involution, maps B)      A*
    wv       P_C (head involution, maps C)      A*
    wo       A*                                 P_C

with the head maps valid after surgery (twin heads gauged to B = C = I). In orthonormal eigenbases E+ (eigenvalue +1)
and E- (-1) of R_out and R_in, W is block diagonal: W = E_out+ W_ee E_in+^T + E_out- W_oo E_in-^T, and the two blocks
hold the Mirror-exact count m+ n+ + m- n-. Bases: A* uses efficient.eig_basis (the stream basis of the even/odd form);
P uses (e_i +- e_j)/sqrt(2) per twin pair and e_s per self-twin (stacking.PermSignBasis); P_B and P_C use head-local
bases (HeadSignBasis): [a; X[h] a]/sqrt(2) and [a; -X[h] a]/sqrt(2) on a twin pair (h, g), the eigenbasis of X[s] on a
self-head s, as efficient.build_mirror_attention executes them.

SVD-LLM inside the blocks. Each block F (m x n) is compressed with the arithmetic of mirror.baselines.lowrank_factors
"svdllm" (same ridge): L = chol(M_s + lam I), lam = damp mean(diag M_s), F L = U S V^T, F' = U_r S_r V_r^T L^-1.
M_s is the block's input moment E_in,s^T M E_in,s, where M is the matrix's uncentred input moment on the fit images
through the folded original model (mirror.baselines.collect_stats, as the SVD-LLM baseline uses it: M1 for fc1, M2 for
fc2, Nin for wq, wk, wv, Mctx for wo; Mctx rotated into the twin gauge, stacking._moment).
Why the even and odd inputs decouple. E_in,s^T M E_in,s are the diagonal blocks of M in the eigenbasis of R_in, and
they are also the diagonal blocks of the mirror-symmetrised moment (M + R_in M R_in) / 2, whose off-diagonal blocks
are zero; the symmetrised moment is the second moment of the flip-augmented calibration set as an exactly equivariant
model sees it (each image and its mirror; the mirror's input is x R_in). So we whiten with the symmetrised moment.
Under it x+ and x- are uncorrelated, and because the even and odd outputs land in orthogonal subspaces the output
error of the whole matrix splits exactly into an even and an odd term, each SVD-LLM's whitened truncation problem with
its own block moment. Three consequences: (i) per-block SVD-LLM is the optimal equivariant truncation of the matrix for
that moment; (ii) the result is the same whether we approximate the projected Mirror-exact weight or the original
weight (projection keeps exactly the diagonal blocks); (iii) the discarded squared whitened singular values of all
blocks of a matrix add up to its output error, which makes one global ranking by whitened energy meaningful. Without
symmetrisation the cross moment M+- would let the even block regress odd inputs out of even ones, a correlation that
flips sign under the mirror.

Allocation (allocate_by_energy). One global greedy over every block of every converted matrix. A unit is one more rank
of a block, costing m + n parameters, gaining its squared whitened singular value s_k^2; with energy "relative"
(default) the gain is divided by the matrix's total whitened energy (sum of s^2 over both its blocks, its whitened
output energy), so matrices with different output scales compare by the share of their own output they keep;
"absolute" uses s_k^2 itself. Units are taken in order of gain per parameter (a heap; within a block the gains fall)
until the model reaches the target. A block is never factorised beyond its break-even rank floor(m n / (m + n)); the next unit then
stores it dense (cost m n minus what its factors held, gain the rest of its energy), as budget._lowrank_alloc does
for the SVD-LLM baseline. The model ends at most one unit above the target (gap reported; eligible when 0 <= gap <= 1%
of the matrix budget).

Counting (the MUST-09 accounting; sizes.py): params_model = N - converted dense + sum over blocks of r (m + n) (m n when
stored dense) + basis changes (none at full conversion). The bases cost nothing in the even/odd form (the stream is
held in the A* basis, the neuron basis is a butterfly, the self-head rotations are absorbed into q, k and Wo as for
Mirror-exact). Biases, norms and embeddings are dense, as everywhere.

Trainable form (build returns it). Every converted Linear becomes a MirrorFactorLinear holding, per block, the factors
U [m, r] and V [n, r] (block = U V^T) or a dense block W [m, n], and the bias as its coordinates b_e in the even output
subspace. The eigenbases are fixed buffers stored once per model in a BasisBank (registered as model.mirror_bases,
persistent, so a checkpoint carries them: the eigenvectors of a degenerate involution are not reproducible across
machines, and the factors are coordinates in exactly these bases). Whatever values the factors take, R_out W = W R_in
and R_out b = b hold, so the model stays exactly equivariant without any projection step; the ranks cannot grow. The
forward computes the factors in the original coordinates once per call (in the parameter dtype, outside autocast),
then runs x -> x V'^T U'^T + b, or the dense product when that is cheaper. Factors start "balanced" (default: U_r
S_r^(1/2), S_r^(1/2) V_r^T of the plain SVD of the truncated block, the convention heal_structured.make_trainable uses
for the SVD-LLM baseline) or "whitened" (SVD-LLM's own W'_u = U_r S_r^(1/2), W'_v = S_r^(1/2) V_r^T L^-1).
The embeddings train and are re-projected after every step (heal_vexact.VExactEmbeddings), the even head is a buffer;
frozen: absorbed norms and LayerScale of the converted blocks, the final norm affine, the mask token, heads (as
Mirror-exact in MUST-09). Run the trainer with train_scope "all".

API
  build(adapter, structure, frac, stats, hparams=None) -> (model, constraints, spec, info)
  load(adapter, spec, state_dict) -> model           (the state dict must be model.state_dict(): it holds the bases)
  export_dense(model) -> model with plain nn.Linear layers (the dense equivalent, computed in float64)
  refine_bases(model, structure) -> max change      (float64 exactness of a model trained in float32; A09)
  save_structure(path, info["structure"])           (the mirror-family structure.npz, for constraints and audits)

Sources: FINAL_PHASE_PLAN.md A06; mirror/stacking.py (block structure, bases, block moments); mirror/baselines.py
(lowrank_factors svdllm, collect_stats; SVD-LLM, Wang et al., arXiv 2403.07378, as checked there); mirror/efficient.py
(eig_basis, attention even/odd form); mirror/heal_vexact.py (VExactEmbeddings). No web sources were needed.
"""
from __future__ import annotations

import copy
import heapq
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import mirror_family as MF
from ... import stacking as ST
from ...attention import HeadMaps
from ...baselines import DEFAULT_DAMP
from ...budget import BUDGET_TOL, gap_ok
from ...efficient import eig_basis
from ...heal import _absorbed_parameters, exact_involution
from ...heal_attn import attn_absorbed_parameters
from ...heal_structured import SplitQKV, save_structure as _hs_save_structure
from ...heal_vexact import VExactEmbeddings, span_frozen_parameters
from .sizes import full_model_params, target_params

METHOD = "mirror_svdllm"
SPEC_FORMAT = "fp-mirror-svdllm-v1"
SIGNS = ("even", "odd")
TAGS = {"even": "e", "odd": "o"}
MATRICES = ("fc1", "fc2", "wq", "wk", "wv", "wo")
DEFAULT_HPARAMS = {"damp": DEFAULT_DAMP, "energy": "relative", "min_rank": 0, "dense_switch": True,
                   "factors": "balanced", "ranks": None, "k": None, "tol": 1e-5}

__all__ = ["METHOD", "SPEC_FORMAT", "DEFAULT_HPARAMS", "HeadSignBasis", "BasisBank", "MirrorFactorLinear",
           "MirrorFactorQKV", "MirrorFactorTrainable", "whitened_svd", "allocate_by_energy", "build", "load",
           "constraints_for", "export_dense", "refine_bases", "save_structure", "block_costs", "resolve_hparams"]


# ----------------------------------------------------------------------------------------------
# Bases
# ----------------------------------------------------------------------------------------------
class HeadSignBasis(ST.DenseSignBasis):
    """Head-local orthonormal even/odd basis of the D x D head involution of `hm` (which "B": queries and keys,
    "C": values; block (sigma(h), h) = X[h], stacking.head_involution). Twin pair (h, g), h < g: even columns
    [e_k at head h; X[h] e_k at head g] / sqrt(2), odd columns the same with -X[h] e_k (k = 1..dh); self-head s: the
    eigenbasis of X[s] (efficient.eig_basis). Even columns first: pairs in order, then the self-heads' even parts."""

    def __init__(self, hm: HeadMaps, which: str = "B"):           # no super().__init__: no eigh of the D x D matrix
        if which not in ("B", "C"):
            raise ValueError("which must be 'B' or 'C'")
        X = np.asarray(hm.B if which == "B" else hm.C, dtype=np.float64)
        H, dh = hm.H, hm.dh
        D = H * dh
        r = 1.0 / math.sqrt(2.0)
        plus, minus = [], []
        for h, g in hm.pairs:
            for k in range(dh):
                for sgn, out in ((1.0, plus), (-1.0, minus)):
                    v = np.zeros(D)
                    v[h * dh + k] = r
                    v[g * dh:(g + 1) * dh] = sgn * r * X[h][:, k]
                    out.append(v)
        for s in hm.self_heads:
            U, du = eig_basis(X[int(s)])
            for k in range(dh):
                v = np.zeros(D)
                v[int(s) * dh:(int(s) + 1) * dh] = U[:, k]
                (plus if k < du else minus).append(v)
        cols = plus + minus
        self.Q = np.ascontiguousarray(np.stack(cols, axis=1)) if cols else np.zeros((D, 0))
        self.n = int(D)
        self.n_plus = int(len(plus))
        self.n_minus = int(len(minus))
        self.trace = int(self.n_plus - self.n_minus)
        self._dev = {}


class BasisBank(nn.Module):
    """Every even/odd basis of a converted model, stored once as persistent buffers (registered as
    model.mirror_bases; the MirrorFactorLinear layers hold a plain reference to it, so state_dict lists each basis
    once). Keys: "A" (the span operator), "P<b>" (neuron pairing of block b), "B<b>", "C<b>" (head involutions).
    Kinds: "dense" (buffers <key>__Qp [n, n+], <key>__Qm [n, n-]) and "perm" (index and scale vectors of the
    butterfly, no n x n matrix). Coordinates are rows: up(key, Y, sign) maps [..., n_sign] to [..., n] (Y E_sign^T),
    down(key, X) maps [..., n] to ([..., n+], [..., n-])."""

    def __init__(self):
        super().__init__()
        self.meta: dict = {}

    def _reg(self, name: str, t: torch.Tensor) -> None:
        self.register_buffer(name, t, persistent=True)

    def add(self, key: str, basis, dtype=torch.float32, device=None) -> None:
        """Add a stacking.PermSignBasis, stacking.DenseSignBasis or HeadSignBasis under `key` (once)."""
        if key in self.meta:
            return
        n, npl, nmi = int(basis.n), int(basis.n_plus), int(basis.n_minus)
        if isinstance(basis, ST.PermSignBasis):
            i, j, s = (np.asarray(basis._idx[k], dtype=np.int64) for k in ("i", "j", "s"))
            npair = len(i)
            src_p, sc_p = np.zeros(n, dtype=np.int64), np.zeros(n)
            src_m, sc_m = np.zeros(n, dtype=np.int64), np.zeros(n)
            r = 1.0 / math.sqrt(2.0)
            src_p[i], src_p[j], src_p[s] = np.arange(npair), np.arange(npair), npair + np.arange(len(s))
            sc_p[i], sc_p[j], sc_p[s] = r, r, 1.0
            src_m[i], src_m[j] = np.arange(npair), np.arange(npair)
            sc_m[i], sc_m[j] = r, -r
            lng = dict(dtype=torch.long, device=device)
            flt = dict(dtype=dtype, device=device)
            for nm, v, kw in (("src_p", src_p, lng), ("scale_p", sc_p, flt), ("src_m", src_m, lng),
                              ("scale_m", sc_m, flt), ("pi", i, lng), ("pj", j, lng), ("ps", s, lng)):
                self._reg(f"{key}__{nm}", torch.as_tensor(v, **kw))
            self.meta[key] = {"key": key, "kind": "perm", "n": n, "n_plus": npl, "n_minus": nmi, "n_pairs": int(npair)}
        else:
            Q = torch.as_tensor(np.asarray(basis.Q, dtype=np.float64))
            self._reg(f"{key}__Qp", Q[:, :npl].contiguous().to(dtype=dtype, device=device))
            self._reg(f"{key}__Qm", Q[:, npl:].contiguous().to(dtype=dtype, device=device))
            self.meta[key] = {"key": key, "kind": "dense", "n": n, "n_plus": npl, "n_minus": nmi}

    @classmethod
    def skeleton(cls, layout, dtype=torch.float32, device=None) -> "BasisBank":
        """An empty bank with the buffers of `layout` (spec["bases"]), to load a state dict into."""
        bank = cls()
        for m in layout:
            key, n, npl, nmi = m["key"], int(m["n"]), int(m["n_plus"]), int(m["n_minus"])
            if m["kind"] == "perm":
                npair = int(m["n_pairs"])
                lng = dict(dtype=torch.long, device=device)
                for nm, size in (("src_p", n), ("src_m", n), ("pi", npair), ("pj", npair), ("ps", n - 2 * npair)):
                    bank._reg(f"{key}__{nm}", torch.zeros(size, **lng))
                for nm in ("scale_p", "scale_m"):
                    bank._reg(f"{key}__{nm}", torch.zeros(n, dtype=dtype, device=device))
            else:
                bank._reg(f"{key}__Qp", torch.zeros(n, npl, dtype=dtype, device=device))
                bank._reg(f"{key}__Qm", torch.zeros(n, nmi, dtype=dtype, device=device))
            bank.meta[key] = dict(m)
        return bank

    def layout(self) -> list:
        return [dict(v) for v in self.meta.values()]

    def _b(self, key: str, name: str) -> torch.Tensor:
        return getattr(self, f"{key}__{name}")

    def n_sign(self, key: str, sign: str) -> int:
        return int(self.meta[key]["n_plus" if sign == "even" else "n_minus"])

    def up(self, key: str, Y: torch.Tensor, sign: str) -> torch.Tensor:
        m = self.meta[key]
        if Y.shape[-1] != self.n_sign(key, sign):
            raise ValueError(f"basis {key} {sign}: {Y.shape[-1]} coordinates, expected {self.n_sign(key, sign)}")
        if self.n_sign(key, sign) == 0:
            return Y.new_zeros(tuple(Y.shape[:-1]) + (m["n"],))
        if m["kind"] == "dense":
            Q = self._b(key, "Qp" if sign == "even" else "Qm").to(Y.dtype)
            return Y @ Q.T
        p = "p" if sign == "even" else "m"
        return Y.index_select(-1, self._b(key, f"src_{p}")) * self._b(key, f"scale_{p}").to(Y.dtype)

    def down(self, key: str, X: torch.Tensor) -> tuple:
        m = self.meta[key]
        if m["kind"] == "dense":
            return X @ self._b(key, "Qp").to(X.dtype), X @ self._b(key, "Qm").to(X.dtype)
        r = 1.0 / math.sqrt(2.0)
        xi, xj = X.index_select(-1, self._b(key, "pi")), X.index_select(-1, self._b(key, "pj"))
        return torch.cat([(xi + xj) * r, X.index_select(-1, self._b(key, "ps"))], -1), (xi - xj) * r

    def rows(self, key: str, sign: str, like: torch.Tensor) -> torch.Tensor:
        """E_sign^T [n_sign, n] in the dtype and device of `like`."""
        m = self.meta[key]
        if m["kind"] == "dense":
            return self._b(key, "Qp" if sign == "even" else "Qm").to(like.dtype).T
        k = self.n_sign(key, sign)
        return self.up(key, torch.eye(k, dtype=like.dtype, device=like.device), sign)

    def extra_repr(self) -> str:
        return ", ".join(f"{k}:{v['kind']}({v['n_plus']}+{v['n_minus']})" for k, v in self.meta.items())


# ----------------------------------------------------------------------------------------------
# Trainable modules
# ----------------------------------------------------------------------------------------------
def _no_autocast(t: torch.Tensor):
    return torch.autocast(device_type=t.device.type, enabled=False)


class MirrorFactorLinear(nn.Module):
    """y = x W^T + b with W = E_out+ B_e E_in+^T + E_out- B_o E_in-^T and b = E_out+ b_e, where each block B_s is
    U_s V_s^T (kind "lowrank"; U [m, r], V [n, r]), a dense W_s [m, n] (kind "dense") or zero ("empty"). The bases
    come from the model's BasisBank (out_key, in_key). Exactly equivariant whatever the parameter values:
    R_out W R_in = W and R_out b = b. `weight` and `bias` are the dense equivalents (read-only)."""

    def __init__(self, bank: BasisBank, out_key: str, in_key: str, parts: dict, bias_even: torch.Tensor | None = None,
                 dtype=None, device=None):
        super().__init__()
        self._bank = [bank]                          # plain reference: the bank is registered once, on the model
        self.out_key, self.in_key = str(out_key), str(in_key)
        mo, mi = bank.meta[self.out_key], bank.meta[self.in_key]
        self.out_features, self.in_features = int(mo["n"]), int(mi["n"])
        kw = dict(dtype=dtype, device=device)
        self.kinds, self.ranks, self.shapes = {}, {}, {}
        for sign in SIGNS:
            tag = TAGS[sign]
            m_s, n_s = bank.n_sign(self.out_key, sign), bank.n_sign(self.in_key, sign)
            self.shapes[sign] = (m_s, n_s)
            p = parts.get(sign) or ("empty",)
            kind, rank = p[0], 0
            if kind == "lowrank":
                U, V = p[1], p[2]
                if U.ndim != 2 or V.ndim != 2 or U.shape[1] != V.shape[1] or tuple(U.shape[:1]) != (m_s,) \
                        or tuple(V.shape[:1]) != (n_s,):
                    raise ValueError(f"{sign} factors {tuple(U.shape)}, {tuple(V.shape)} do not fit a "
                                     f"{m_s} x {n_s} block")
                rank = int(U.shape[1])
                if rank == 0:
                    kind = "empty"
                else:
                    setattr(self, f"U_{tag}", nn.Parameter(U.detach().to(**kw).contiguous()))
                    setattr(self, f"V_{tag}", nn.Parameter(V.detach().to(**kw).contiguous()))
            elif kind == "dense":
                W = p[1]
                if tuple(W.shape) != (m_s, n_s):
                    raise ValueError(f"{sign} dense block {tuple(W.shape)}, expected {(m_s, n_s)}")
                if m_s * n_s == 0:
                    kind = "empty"
                else:
                    rank = min(m_s, n_s)
                    setattr(self, f"W_{tag}", nn.Parameter(W.detach().to(**kw).contiguous()))
            elif kind != "empty":
                raise ValueError(f"unknown block kind {kind!r}")
            self.kinds[sign], self.ranks[sign] = kind, int(rank)
        if bias_even is not None:
            if tuple(bias_even.shape) != (bank.n_sign(self.out_key, "even"),):
                raise ValueError("bias_even must hold the even output coordinates")
            self.b_e = nn.Parameter(bias_even.detach().to(**kw).contiguous())
        else:
            self.register_parameter("b_e", None)
        R = sum(self.ranks.values())
        self.use_dense = R * (self.in_features + self.out_features) >= self.in_features * self.out_features

    @property
    def bank(self) -> BasisBank:
        return self._bank[0]

    def _ref(self) -> torch.Tensor:
        for p in self.parameters():
            return p
        return next(iter(self.bank.buffers()))

    def _factors(self, dtype=None):
        """(V' [R, in], U' [R, out]) in the original coordinates (block = U V^T; a dense block enters as U = W, V = I
        or U = I, V = W^T, whichever has the smaller rank), or (None, None) when every block is empty."""
        bank = self.bank
        Vs, Us = [], []
        for sign in SIGNS:
            tag, kind = TAGS[sign], self.kinds[sign]
            if kind == "lowrank":
                U, V = getattr(self, f"U_{tag}"), getattr(self, f"V_{tag}")
                if dtype is not None:
                    U, V = U.to(dtype), V.to(dtype)
                Vs.append(bank.up(self.in_key, V.T, sign))
                Us.append(bank.up(self.out_key, U.T, sign))
            elif kind == "dense":
                W = getattr(self, f"W_{tag}")
                if dtype is not None:
                    W = W.to(dtype)
                m, n = W.shape
                if n <= m:
                    Vs.append(bank.rows(self.in_key, sign, W))
                    Us.append(bank.up(self.out_key, W.T, sign))
                else:
                    Vs.append(bank.up(self.in_key, W, sign))
                    Us.append(bank.rows(self.out_key, sign, W))
        if not Vs:
            return None, None
        return torch.cat(Vs, 0), torch.cat(Us, 0)

    def _bias(self, dtype=None):
        if self.b_e is None:
            return None
        b = self.b_e if dtype is None else self.b_e.to(dtype)
        return self.bank.up(self.out_key, b[None], "even")[0]

    @torch.no_grad()
    def dense(self, dtype=None) -> tuple:
        """(W [out, in], b [out] or None), computed in `dtype` (default: the parameter dtype), without autograd (the
        forward pass is what trains)."""
        ref = self._ref()
        dt = dtype or ref.dtype
        with _no_autocast(ref):
            Vt, Ut = self._factors(dt)
            W = torch.zeros(self.out_features, self.in_features, dtype=dt, device=ref.device) if Vt is None \
                else Ut.T @ Vt
            return W, self._bias(dt)

    @property
    def weight(self) -> torch.Tensor:
        return self.dense()[0]

    @property
    def bias(self):
        return self.dense()[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _no_autocast(x):
            Vt, Ut = self._factors()
            b = self._bias()
            W = Ut.T @ Vt if (Vt is not None and self.use_dense) else None
        if Vt is None:
            y = x.new_zeros(tuple(x.shape[:-1]) + (self.out_features,))
            return y if b is None else y + b.to(y.dtype)
        if W is not None:
            return F.linear(x, W, b)
        return F.linear(F.linear(x, Vt), Ut.T, b)

    def counted_params(self) -> int:
        """Matrix parameters as counted: r (m + n) per factorised block, m n per dense block."""
        n = 0
        for sign in SIGNS:
            m_s, n_s = self.shapes[sign]
            if self.kinds[sign] == "lowrank":
                n += self.ranks[sign] * (m_s + n_s)
            elif self.kinds[sign] == "dense":
                n += m_s * n_s
        return int(n)

    def macs(self) -> int:
        """Multiply-adds per token as this module executes (original coordinates)."""
        if self.use_dense:
            return int(self.in_features * self.out_features)
        return int(sum(self.ranks.values()) * (self.in_features + self.out_features))

    def extra_repr(self) -> str:
        parts = ", ".join(f"{s}={self.kinds[s]}{self.shapes[s]}r{self.ranks[s]}" for s in SIGNS)
        return f"in={self.in_features}, out={self.out_features}, {parts}, bases=({self.out_key}, {self.in_key})"


class MirrorFactorQKV(SplitQKV):
    """attn.qkv replacement: q, k, v each a MirrorFactorLinear; output [q; k; v] like the fused layer (weight and
    bias are the dense concatenations, so folding and analysis code keep working)."""


def _modules_of(blk: nn.Module) -> dict:
    qkv = blk.attn.qkv
    return {"fc1": blk.mlp.fc1, "fc2": blk.mlp.fc2, "wq": getattr(qkv, "q", None), "wk": getattr(qkv, "k", None),
            "wv": getattr(qkv, "v", None), "wo": blk.attn.proj}


# ----------------------------------------------------------------------------------------------
# Constraint (mirror/heal.py protocol)
# ----------------------------------------------------------------------------------------------
class MirrorFactorTrainable:
    """Trainable set of a Mirror + SVD-LLM student: freezes everything outside the converted blocks and the
    embeddings, and inside the converted blocks the absorbed norm affines and LayerScale. Projects nothing: the
    factorised modules keep the structure and the ranks by construction. `check` confirms the module layout against
    the build records and measures the structure in float64 (off-diagonal share of every matrix in its even/odd
    bases, odd part of every bias; zero up to the rounding of the stored bases)."""

    def __init__(self, blocks, records):
        self.blocks = [int(b) for b in blocks]
        self.records = [dict(r) for r in records]

    def frozen_parameters(self, model: nn.Module) -> list:
        out = list(span_frozen_parameters(model, self.blocks, True))
        for b in self.blocks:
            out += list(_absorbed_parameters(model.blocks[b])) + list(attn_absorbed_parameters(model.blocks[b]))
        return out

    def __call__(self, model: nn.Module) -> None:
        return None

    @torch.no_grad()
    def check(self, model: nn.Module) -> dict:
        ok, worst, worst_b, rows = True, 0.0, 0.0, []
        for r in self.records:
            mod = _modules_of(model.blocks[r["block"]])[r["name"]]
            good = isinstance(mod, MirrorFactorLinear)
            if good:
                for sign in SIGNS:
                    good = good and mod.kinds[sign] == r[f"kind_{sign}"] and mod.ranks[sign] == r[f"rank_{sign}"]
                W, b = mod.dense(torch.float64)
                bank = mod.bank
                Lp, Lm = bank.down(mod.out_key, W.T)                     # [in, m+], [in, m-]
                pp, pm = bank.down(mod.in_key, Lp.T)
                mp, mm = bank.down(mod.in_key, Lm.T)
                tot = float((W * W).sum())
                off = math.sqrt((float((pm * pm).sum()) + float((mp * mp).sum())) / tot) if tot > 0 else 0.0
                bo = 0.0
                if b is not None and float(b.norm()) > 0:
                    bo = float(bank.down(mod.out_key, b[None])[1].norm() / b.norm())
                worst, worst_b = max(worst, off), max(worst_b, bo)
                rows.append({"block": r["block"], "name": r["name"], "offdiag": off, "bias_odd": bo})
            ok = ok and good
        return {"ok": bool(ok), "max_offdiag": worst, "max_bias_odd": worst_b, "n_matrices": len(self.records),
                "per_matrix": rows}

    def describe(self) -> dict:
        return {"type": "MirrorFactorTrainable", "blocks": self.blocks, "n_matrices": len(self.records),
                "frozen": "everything outside the converted blocks and the embeddings; norm affines and LayerScale "
                          "inside the converted blocks; nothing projected (structure by construction)"}


# ----------------------------------------------------------------------------------------------
# Whitening and allocation
# ----------------------------------------------------------------------------------------------
def whitened_svd(F_: torch.Tensor, M: torch.Tensor | None, damp: float = DEFAULT_DAMP) -> tuple:
    """(U, S, Vh, L, lam) of SVD-LLM's whitening for block F [m, n] with input moment M [n, n], in float64, with the
    arithmetic of mirror.baselines.lowrank_factors "svdllm": L = chol(sym(M) + lam I), lam = damp mean(diag sym(M)),
    F L = U S Vh. M None: plain SVD (L = I, lam = 0)."""
    Fd = F_.to(torch.float64)
    n = Fd.shape[1]
    if M is None:
        L = torch.eye(n, dtype=torch.float64, device=Fd.device)
        lam = 0.0
    else:
        M = M.to(Fd)
        if tuple(M.shape) != (n, n):
            raise ValueError(f"moment {tuple(M.shape)} does not fit a block with {n} inputs")
        Ms = 0.5 * (M + M.T)
        lam = float(damp) * float(torch.diagonal(Ms).mean())
        L, err = torch.linalg.cholesky_ex(Ms + lam * torch.eye(n, dtype=Fd.dtype, device=Fd.device))
        if int(err) != 0:
            raise RuntimeError(f"Cholesky failed (damp {damp}); increase damp")
    U, S, Vh = torch.linalg.svd(Fd @ L, full_matrices=False)
    return U, S, Vh, L, lam


def _truncate(U, S, Vh, L, r: int) -> torch.Tensor:
    B = (U[:, :r] * S[:r]) @ Vh[:r]
    return torch.linalg.solve_triangular(L, B, upper=False, left=False)        # X L = B


def _factor_pair(Ft: torch.Tensor, r: int, U, S, Vh, L, mode: str) -> tuple:
    """(U [m, r], V [n, r]) with U V^T = the truncated block Ft (rank r)."""
    if mode == "whitened":
        sq = S[:r].sqrt()
        Vt = torch.linalg.solve_triangular(L, sq[:, None] * Vh[:r], upper=False, left=False)
        return U[:, :r] * sq, Vt.T
    P, Sg, Qh = torch.linalg.svd(Ft, full_matrices=False)
    sq = Sg[:r].sqrt()
    return P[:, :r] * sq, (sq[:, None] * Qh[:r]).T


def allocate_by_energy(entries, budget: int, energy: str = "relative", min_rank: int = 0,
                       dense_switch: bool = True) -> list:
    """Global greedy rank allocation (see the module docstring). entries: one dict per block with "shape" (m, n),
    "sv2" (squared whitened singular values, descending) and "norm" (the matrix's total whitened energy, used by
    energy "relative"). Units are taken in order of gain per parameter until the blocks hold at least `budget`
    parameters (min_rank ranks per block first). Returns one dict per entry: rank, dense_storage, params, kept and
    discarded energy shares, and the order in which its last unit was taken."""
    if energy not in ("relative", "absolute"):
        raise ValueError("energy must be 'relative' or 'absolute'")
    st = []
    for e in entries:
        m, n = (int(v) for v in e["shape"])
        full = min(m, n)
        cap = 0 if full == 0 else (min(full, (m * n) // (m + n)) if dense_switch else full)
        sv2 = np.asarray(e.get("sv2", np.zeros(full)), dtype=np.float64)
        if len(sv2) != full:
            raise ValueError(f"entry with shape {(m, n)} has {len(sv2)} singular values, expected {full}")
        st.append({"m": m, "n": n, "full": full, "cap": cap, "r": 0, "dense": False, "params": 0, "sv2": sv2,
                   "norm": float(e.get("norm", 1.0)), "last": -1})

    def unit(s):
        if s["full"] == 0 or s["dense"]:
            return None
        if s["r"] < s["cap"]:
            return float(s["sv2"][s["r"]]), s["m"] + s["n"], "rank"
        if dense_switch and s["r"] < s["full"]:
            return float(s["sv2"][s["r"]:].sum()), s["m"] * s["n"] - s["r"] * (s["m"] + s["n"]), "dense"
        return None

    def score(u, s):
        gain, cost, _ = u
        if energy == "relative":
            gain = gain / s["norm"] if s["norm"] > 0 else 0.0
        return math.inf if cost <= 0 else gain / cost

    def take(s, u, step):
        _, cost, kind = u
        if kind == "rank":
            s["r"] += 1
            s["params"] += cost
        else:
            s["dense"] = True
            s["params"] = s["m"] * s["n"]
        s["last"] = step
        return cost

    total, step = 0, 0
    for s in st:
        while s["r"] < min(int(min_rank), s["cap"]):
            total += take(s, unit(s), step)
            step += 1
    heap = []
    for i, s in enumerate(st):
        u = unit(s)
        if u is not None:
            heap.append((-score(u, s), i))
    heapq.heapify(heap)
    while total < int(budget) and heap:
        _, i = heapq.heappop(heap)
        s = st[i]
        total += take(s, unit(s), step)
        step += 1
        u = unit(s)
        if u is not None:
            heapq.heappush(heap, (-score(u, s), i))
    out = []
    for s in st:
        rank = s["full"] if s["dense"] else s["r"]
        e_tot = float(s["sv2"].sum())
        kept = float(s["sv2"][:rank].sum())
        out.append({"rank": int(rank), "full": int(s["full"]), "dense_storage": bool(s["dense"]),
                    "params": int(s["params"]), "kept_energy": kept / e_tot if e_tot > 0 else 1.0,
                    "discarded_energy": 1.0 - kept / e_tot if e_tot > 0 else 0.0, "last_unit": int(s["last"])})
    return out


# ----------------------------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------------------------
def resolve_hparams(hparams: dict | None) -> dict:
    hp = {**DEFAULT_HPARAMS, **(hparams or {})}
    bad = set(hp) - set(DEFAULT_HPARAMS)
    if bad:
        raise ValueError(f"unknown hyperparameters {sorted(bad)} (allowed {sorted(DEFAULT_HPARAMS)})")
    if hp["energy"] not in ("relative", "absolute"):
        raise ValueError("energy must be 'relative' or 'absolute'")
    if hp["factors"] not in ("balanced", "whitened"):
        raise ValueError("factors must be 'balanced' or 'whitened'")
    if hp["ranks"] is not None and not (hp["ranks"] in ("full", "dense") or isinstance(hp["ranks"], dict)):
        raise ValueError("ranks must be None, 'full', 'dense' or a dict '<block>.<name>.<sign>' -> rank")
    return hp


def _bank_keys(name: str, b: int) -> tuple:
    """(out_key, in_key) of matrix `name` in block b."""
    return {"fc1": (f"P{b}", "A"), "fc2": ("A", f"P{b}"), "wq": (f"B{b}", "A"), "wk": (f"B{b}", "A"),
            "wv": (f"C{b}", "A"), "wo": ("A", f"C{b}")}[name]


class _Bases:
    """Build-time float64 bases of one span (stacking objects; HeadSignBasis for the heads)."""

    def __init__(self, A, perms, heads_after, blocks):
        self.blocks = [int(b) for b in blocks]
        self.obj = {"A": ST.DenseSignBasis(exact_involution(A))}
        for i, b in enumerate(self.blocks):
            self.obj[f"P{b}"] = ST.PermSignBasis(perms[i])
            self.obj[f"B{b}"] = HeadSignBasis(heads_after[i], "B")
            self.obj[f"C{b}"] = HeadSignBasis(heads_after[i], "C")

    def pair(self, name: str, b: int) -> tuple:
        o, i = _bank_keys(name, b)
        return self.obj[o], self.obj[i]


def _biases(mats: dict) -> dict:
    a, m = mats["_attn"], mats["_mlp"]
    return {"fc1": m["b1"], "fc2": m["b2"], "wq": a["bq"], "wk": a["bk"], "wv": a["bv"], "wo": a["bo"]}


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    den = float(b.norm())
    return float((a - b).norm()) / den if den > 0 else float((a - b).norm())


def _override_rank(hp: dict, b: int, name: str, sign: str, shape) -> tuple | None:
    """(rank, dense) from hparams["ranks"], or None to allocate."""
    ranks = hp["ranks"]
    if ranks is None:
        return None
    m, n = shape
    if ranks == "full":
        return min(m, n), False
    if ranks == "dense":
        return min(m, n), True
    key = f"{b}.{name}.{sign}"
    if key not in ranks:
        raise KeyError(f"ranks has no entry {key!r}")
    v = ranks[key]
    if v == "dense":
        return min(m, n), True
    r = int(v)
    if not 0 <= r <= min(m, n):
        raise ValueError(f"rank {r} out of range for the {m} x {n} block {key}")
    return r, False


@torch.no_grad()
def build(adapter, structure, frac, stats, hparams=None) -> tuple:
    """Mirror + SVD-LLM student at `frac` of the full model (see the module docstring).

    structure: a mirror_family.SpanStructure (or the path of one); stats: mirror.baselines.collect_stats of the
    original model on the fit images for every block, methods ["svdllm"], both components (None only with hparams
    ranks "full" or "dense": plain SVD). hparams (DEFAULT_HPARAMS): damp, energy, min_rank, dense_switch, factors,
    ranks (None: allocate; "full": every block factorised at full rank; "dense": every block dense, i.e.
    Mirror-exact; or a dict "<block>.<name>.<sign>" -> rank or "dense"), k (span size, default the full model),
    tol (largest off-diagonal share accepted in the projected weights).
    Returns (model, constraints, spec, info); run mirror/heal.py with train_scope "all" and these constraints."""
    hp = resolve_hparams(hparams)
    if isinstance(structure, (str, Path)):
        structure = MF.load_span_structure(structure)
    L = int(adapter.depth)
    k = L if hp["k"] is None else int(hp["k"])
    if not 1 <= k <= L:
        raise ValueError(f"k must be in [1, {L}]")
    comps = MF.COMPONENTS
    blocks = list(range(k))
    if stats is None and hp["ranks"] not in ("full", "dense"):
        raise ValueError("Mirror + SVD-LLM needs baselines.collect_stats statistics (methods ['svdllm'])")
    base, binfo = MF.build_mirror_variant(adapter, "mirror_exact", blocks, comps, structure)
    s = binfo["structure"]
    A = exact_involution(s["A"])
    perms, heads_pre, heads_after = list(s["perms"]), list(s["heads"]), list(s["heads_after"])
    ref = next(base.parameters())
    dtype, dev = ref.dtype, ref.device
    bases = _Bases(A, perms, heads_after, blocks)
    damp = float(hp["damp"])

    N = full_model_params(adapter)
    span_dense = int(binfo["span_dense"])
    basis = int(binfo["params_basis_change"])
    target = target_params(adapter, frac)
    outside = N - span_dense
    T_m = int(target - outside - basis)
    if T_m <= 0 and hp["ranks"] is None:
        raise ValueError(f"fraction {frac}: the target {target:,} leaves no parameters for the converted matrices "
                         f"({outside + basis:,} are outside them or basis changes)")

    def blocks_of(b, i, name, mats):
        out_b, in_b = bases.pair(name, b)
        Fp, Fm, res = ST.split_blocks(mats[name], out_b, in_b)
        if res > float(hp["tol"]):
            raise ValueError(f"block {b} {name}: weights are not equivariant for this structure (off-diagonal share "
                             f"{res:.2e}, tol {hp['tol']:g})")
        Mp = Mm = None
        if stats is not None:
            M = ST._moment(stats, b, name, heads_pre[i]).to(device=Fp.device, dtype=torch.float64)
            Mp, Mm = ST.sign_moments(in_b, M)
        return {"even": (Fp, Mp), "odd": (Fm, Mm)}, res

    # pass 1: whitened spectra of every block
    entries, worst_res = [], 0.0
    for i, b in enumerate(blocks):
        mats = ST._read(base.blocks[b], comps)
        for name in MATRICES:
            parts, res = blocks_of(b, i, name, mats)
            worst_res = max(worst_res, res)
            rows = []
            for sign in SIGNS:
                Fs, Ms = parts[sign]
                m, n = (int(v) for v in Fs.shape)
                if min(m, n) == 0 or hp["ranks"] is not None:
                    sv2 = np.zeros(min(m, n))
                else:
                    sv = torch.linalg.svdvals(Fs.to(torch.float64) @ whitened_svd_factor(Ms, n, damp, Fs))
                    sv2 = (sv.double() ** 2).cpu().numpy()
                rows.append({"block": b, "name": name, "sign": sign, "shape": (m, n), "sv2": sv2})
            norm = float(sum(r["sv2"].sum() for r in rows))
            for r in rows:
                r["norm"] = norm
            entries += rows
    if hp["ranks"] is None:
        alloc = allocate_by_energy(entries, T_m, hp["energy"], int(hp["min_rank"]), bool(hp["dense_switch"]))
    else:
        alloc = []
        for e in entries:
            m, n = e["shape"]
            r, dense = _override_rank(hp, e["block"], e["name"], e["sign"], (m, n))
            alloc.append({"rank": int(r), "full": min(m, n), "dense_storage": bool(dense) and m * n > 0,
                          "params": int(m * n if dense else r * (m + n)), "kept_energy": float("nan"),
                          "discarded_energy": float("nan"), "last_unit": -1})
    for e, a in zip(entries, alloc):
        e.update(a)
    by = {(e["block"], e["name"], e["sign"]): e for e in entries}

    # pass 2: factors, modules
    bank = BasisBank()
    for key, obj in bases.obj.items():
        bank.add(key, obj, dtype, dev)
    model = base
    model.mirror_bases = bank
    per_matrix, records, worst_err = [], [], 0.0
    tol_mod = 1e-10 if dtype == torch.float64 else 1e-4
    for i, b in enumerate(blocks):
        blk = model.blocks[b]
        mats = ST._read(blk, comps)
        biases = _biases(mats)
        mods = {}
        for name in MATRICES:
            parts, _ = blocks_of(b, i, name, mats)
            out_b, in_b = bases.pair(name, b)
            mparts, trunc, row = {}, {}, {"block": b, "name": name, "kind": "mirror_lowrank",
                                          "shape": list(mats[name].shape), "dense": int(mats[name].numel())}
            params, disc_num, disc_den = 0, 0.0, 0.0
            for sign in SIGNS:
                Fs, Ms = parts[sign]
                e = by[(b, name, sign)]
                m, n = e["shape"]
                r = int(e["rank"])
                if m * n == 0 or (r == 0 and not e["dense_storage"]):
                    mparts[sign] = ("empty",)
                    trunc[sign] = torch.zeros(m, n, dtype=torch.float64, device=Fs.device)
                elif e["dense_storage"]:
                    mparts[sign] = ("dense", Fs)
                    trunc[sign] = Fs
                else:
                    U, S, Vh, Lw, _ = whitened_svd(Fs, Ms, damp)
                    Ft = _truncate(U, S, Vh, Lw, r)
                    Uf, Vf = _factor_pair(Ft, r, U, S, Vh, Lw, hp["factors"])
                    mparts[sign] = ("lowrank", Uf, Vf)
                    trunc[sign] = Ft
                params += int(e["params"])
                sv2 = np.asarray(e["sv2"])
                if np.isfinite(e["discarded_energy"]):
                    disc_num += float(e["discarded_energy"]) * float(sv2.sum())
                    disc_den += float(sv2.sum())
                row.update({f"shape_{sign}": [m, n], f"rank_{sign}": r, f"full_{sign}": min(m, n),
                            f"dense_storage_{sign}": bool(e["dense_storage"]), f"params_{sign}": int(e["params"]),
                            f"discarded_energy_{sign}": float(e["discarded_energy"])})
            bb = biases[name].to(torch.float64)
            bp, bm = out_b.down(bb[:, None])
            if float(bm.norm()) > float(hp["tol"]) * max(float(bb.norm()), 1e-300):
                raise ValueError(f"block {b} {name}: the projected bias has an odd part ({float(bm.norm()):.2e})")
            o_key, i_key = _bank_keys(name, b)
            mod = MirrorFactorLinear(bank, o_key, i_key, mparts, bp[:, 0], dtype=dtype, device=dev)
            want = ST.merge_blocks(trunc["even"], trunc["odd"], out_b, in_b)
            Wm, bmod = mod.dense(torch.float64)
            err = max(_rel(Wm, want), _rel(bmod, bb))
            worst_err = max(worst_err, err)
            if err > tol_mod:
                raise AssertionError(f"block {b} {name}: the trainable module differs from its blocks ({err:.2e})")
            if mod.counted_params() != params:
                raise AssertionError(f"block {b} {name}: module holds {mod.counted_params()} counted parameters, "
                                     f"allocation {params}")
            row.update({"params": int(params), "params_compressor": int(params),
                        "mirror": int(sum(x * y for x, y in (parts["even"][0].shape, parts["odd"][0].shape))),
                        "discarded_energy": disc_num / disc_den if disc_den > 0 else 0.0, "module_error": err})
            per_matrix.append(row)
            records.append({"block": b, "name": name, **{f"kind_{sg}": mod.kinds[sg] for sg in SIGNS},
                            **{f"rank_{sg}": mod.ranks[sg] for sg in SIGNS}})
            mods[name] = mod
        blk.mlp.fc1, blk.mlp.fc2 = mods["fc1"], mods["fc2"]
        blk.attn.qkv = MirrorFactorQKV(mods["wq"], mods["wk"], mods["wv"])
        blk.attn.proj = mods["wo"]
    model.eval()

    pm = int(sum(r["params"] for r in per_matrix))
    params_model = int(outside + pm + basis)
    gap_m = (pm + basis - T_m) / T_m if T_m > 0 else float("nan")
    step = max((e["shape"][0] + e["shape"][1] for e in entries), default=0)
    eligible = bool(T_m > 0 and gap_ok(gap_m, 0.0, BUDGET_TOL))
    reg_perm = np.asarray(s["reg_perm"], dtype=np.int64)
    cons = [MirrorFactorTrainable(blocks, records), VExactEmbeddings(adapter, A, blocks, reg_perm, train=True)]
    structure_out = {"A_star": A, "perms": perms, "maps_after": heads_after, "reg_perm": reg_perm, "blocks": blocks,
                     "heads": heads_pre}
    hp_json = {k_: (v if not isinstance(v, dict) else dict(v)) for k_, v in hp.items()}
    sizes = {"params_full": int(N), "params_outside": int(outside), "span_dense": span_dense,
             "params_matrices": pm, "params_basis_change": basis, "params_model": params_model,
             "params_frac": params_model / N, "target": int(target), "budget_matrices": T_m,
             "gap": (params_model - target) / target, "gap_matrices": gap_m, "gap_step": int(step),
             "eligible": eligible, "mirror_exact_matrices": int(binfo["params_matrices"]),
             "mirror_exact_params_model": int(outside + int(binfo["params_total"]))}
    spec = {"format": SPEC_FORMAT, "family": "mirror_svdllm", "method": METHOD, "frac": float(frac), "k": k,
            "blocks": blocks, "components": list(comps),
            "base": {"materialise_pos_embed": bool(binfo["build"].get("pos_materialised")), "fold": True,
                     "even_head": k == L},
            "bases": bank.layout(),
            "modules": [{"block": r["block"], "name": r["name"], "out_key": _bank_keys(r["name"], r["block"])[0],
                         "in_key": _bank_keys(r["name"], r["block"])[1],
                         **{sg: {"kind": r[f"kind_{sg}"], "rank": r[f"rank_{sg}"],
                                 "shape": list(by[(r["block"], r["name"], sg)]["shape"])} for sg in SIGNS},
                         "bias": True} for r in records],
            "matrices": [{"block": r["block"], "name": r["name"], "params_compressor": r["params"]}
                         for r in per_matrix],
            "compressor": {"basis": ([{"block": blocks[-1], "name": "span_exit", "params": basis}] if basis else [])},
            "structure_file": "structure.npz", "hparams": hp_json, "params": sizes, "train_scope": "all",
            "targets": {"final": "symmetrised", "blocks": "symmetrised", "k": k,
                        "note": "as Mirror-exact (VExactTeacher semantics): A*-symmetrised block targets for blocks "
                                "0..k-1, flip-averaged final targets"},
            "trainable": "even/odd factors and even-subspace biases of the converted blocks (equivariant by "
                         "construction) and the embeddings (re-projected with A* after every step); absorbed norms, "
                         "LayerScale, final norm affine, mask token and heads frozen"}
    info = {"method": METHOD, "hparams": hp_json, "frac": float(frac), "components": list(comps), "blocks": blocks,
            "k": k, "full": k == L, **sizes, "below_budget": bool(pm + basis < T_m),
            "allocation": [{kk: (list(v) if kk == "shape" else v) for kk, v in e.items() if kk not in ("sv2", "norm")}
                           for e in entries],
            "per_matrix": per_matrix, "block_costs": block_costs(model, spec), "structure": structure_out,
            "max_offdiag_residual": float(worst_res), "max_module_error": float(worst_err),
            "zero_rank_blocks": int(sum(1 for e in entries if e["rank"] == 0 and min(e["shape"]) > 0)),
            "dense_blocks": int(sum(1 for e in entries if e["dense_storage"])), "targets": spec["targets"],
            "train_scope": "all", "build": binfo.get("build", {})}
    return model, cons, spec, info


def whitened_svd_factor(M: torch.Tensor | None, n: int, damp: float, like: torch.Tensor) -> torch.Tensor:
    """The whitening factor L of whitened_svd (identity without a moment), for the spectra of pass 1."""
    if M is None:
        return torch.eye(n, dtype=torch.float64, device=like.device)
    M = M.to(device=like.device, dtype=torch.float64)
    Ms = 0.5 * (M + M.T)
    lam = float(damp) * float(torch.diagonal(Ms).mean())
    L, err = torch.linalg.cholesky_ex(Ms + lam * torch.eye(n, dtype=torch.float64, device=like.device))
    if int(err) != 0:
        raise RuntimeError(f"Cholesky failed (damp {damp}); increase damp")
    return L


def block_costs(model: nn.Module, spec: dict) -> list:
    """Per converted block: {"block", "mlp", "attn", "rot", "basis_params", "mlp_exec", "attn_exec", "source"}, the
    input format of scripts/collect_results.analytic_cost (counted multiply-adds per token of the even/odd form equal
    the counted parameters; *_exec: as the trainable modules execute in original coordinates)."""
    by = {}
    for r in spec["matrices"]:
        by.setdefault(int(r["block"]), {})[r["name"]] = int(r["params_compressor"])
    basis = {int(x["block"]): int(x["params"]) for x in spec.get("compressor", {}).get("basis", [])}
    d = int(model.blocks[0].mlp.fc1.in_features)
    out = []
    for b in spec["blocks"]:
        mods = _modules_of(model.blocks[int(b)])
        ex = {n: (m.macs() if isinstance(m, MirrorFactorLinear) else int(m.weight.numel())) for n, m in mods.items()}
        bp = basis.get(int(b), 0)
        out.append({"block": int(b), "mlp": by[int(b)]["fc1"] + by[int(b)]["fc2"],
                    "attn": sum(by[int(b)][n] for n in ("wq", "wk", "wv", "wo")), "rot": bp / float(d * d),
                    "basis_params": bp, "mlp_exec": ex["fc1"] + ex["fc2"],
                    "attn_exec": sum(ex[n] for n in ("wq", "wk", "wv", "wo")),
                    "source": "Mirror + SVD-LLM counted block parameters (even/odd form)"})
    return out


# ----------------------------------------------------------------------------------------------
# Load, export, refine, structure file
# ----------------------------------------------------------------------------------------------
def save_structure(path, structure: dict) -> Path:
    """Write the structure (info["structure"]) in the mirror family's structure.npz format
    (mirror.heal_structured.save_structure; read back with mirror.heal_structured.load_structure)."""
    return _hs_save_structure(path, structure["A_star"], structure["perms"], structure["maps_after"],
                              structure.get("reg_perm"), structure["blocks"])


@torch.no_grad()
def load(adapter, spec: dict, state_dict: dict) -> nn.Module:
    """Rebuild a Mirror + SVD-LLM student from its spec and a state dict saved with model.state_dict() (the state
    holds the factors, the embeddings, the even head and the bases, so no structure file is needed). The model is in
    eval mode, in the adapter's dtype and device; train mode and requires_grad are the caller's business."""
    from ...fold import fold_layernorm_full
    from ...m1 import install_even_head, materialise_pos_embed

    if spec.get("format") != SPEC_FORMAT:
        raise ValueError(f"not a {SPEC_FORMAT} spec")
    m = copy.deepcopy(adapter.model)
    materialise_pos_embed(m, adapter)
    fold_layernorm_full(m, final_norm_affine=False)
    if spec["base"].get("even_head"):
        install_even_head(m, np.eye(int(adapter.embed_dim)))          # P itself comes from the state
    ref = next(m.parameters())
    dtype, dev = ref.dtype, ref.device
    bank = BasisBank.skeleton(spec["bases"], dtype, dev)
    m.mirror_bases = bank
    by_block: dict = {}
    for e in spec["modules"]:
        parts = {}
        for sign in SIGNS:
            p = e[sign]
            mm, nn_ = (int(v) for v in p["shape"])
            if p["kind"] == "lowrank":
                r = int(p["rank"])
                parts[sign] = ("lowrank", torch.zeros(mm, r, dtype=dtype, device=dev),
                               torch.zeros(nn_, r, dtype=dtype, device=dev))
            elif p["kind"] == "dense":
                parts[sign] = ("dense", torch.zeros(mm, nn_, dtype=dtype, device=dev))
            else:
                parts[sign] = ("empty",)
        b_even = torch.zeros(bank.n_sign(e["out_key"], "even"), dtype=dtype, device=dev) if e.get("bias") else None
        by_block.setdefault(int(e["block"]), {})[e["name"]] = MirrorFactorLinear(bank, e["out_key"], e["in_key"], parts,
                                                                                 b_even, dtype=dtype, device=dev)
    for b, mods in by_block.items():
        blk = m.blocks[b]
        blk.mlp.fc1, blk.mlp.fc2 = mods["fc1"], mods["fc2"]
        blk.attn.qkv = MirrorFactorQKV(mods["wq"], mods["wk"], mods["wv"])
        blk.attn.proj = mods["wo"]
    m.load_state_dict(state_dict, strict=True)
    return m.eval()


def constraints_for(adapter, spec: dict, structure: dict) -> list:
    """The constraints of a loaded student (for checks or for further training): MirrorFactorTrainable from the
    spec's module records and VExactEmbeddings with the structure's A* and register pairing (structure: info
    ["structure"] or mirror.heal_structured.load_structure of the saved structure.npz)."""
    records = [{"block": int(e["block"]), "name": e["name"],
                **{f"kind_{sg}": e[sg]["kind"] for sg in SIGNS}, **{f"rank_{sg}": int(e[sg]["rank"]) for sg in SIGNS}}
               for e in spec["modules"]]
    reg = structure.get("reg_perm")
    reg = None if reg is None or len(reg) == 0 else np.asarray(reg, dtype=np.int64)
    blocks = [int(b) for b in spec["blocks"]]
    return [MirrorFactorTrainable(blocks, records),
            VExactEmbeddings(adapter, exact_involution(structure["A_star"]), blocks, reg, train=True)]


@torch.no_grad()
def export_dense(model: nn.Module) -> nn.Module:
    """A deep copy in which every MirrorFactorLinear (and MirrorFactorQKV) is a plain nn.Linear holding the dense
    equivalent, computed in float64 and rounded once to the model dtype, and without the basis bank. It computes the
    same function (to that rounding) and can be used by any code that expects the folded Mirror-exact layout (the
    even/odd form, the exactness audit, evaluation)."""
    m = copy.deepcopy(model)
    for blk in m.blocks:
        for parent, attr in ((blk.mlp, "fc1"), (blk.mlp, "fc2"), (blk.attn, "qkv"), (blk.attn, "proj")):
            mod = getattr(parent, attr)
            if isinstance(mod, MirrorFactorQKV):
                parts = [x.dense(torch.float64) for x in (mod.q, mod.k, mod.v)]
                W = torch.cat([p[0] for p in parts], 0)
                b = torch.cat([p[1] for p in parts], 0)
                ref = mod.q._ref()
            elif isinstance(mod, MirrorFactorLinear):
                W, b = mod.dense(torch.float64)
                ref = mod._ref()
            else:
                continue
            lin = nn.Linear(W.shape[1], W.shape[0], bias=b is not None, device=ref.device, dtype=ref.dtype)
            lin.weight.copy_(W)
            if b is not None:
                lin.bias.copy_(b)
            setattr(parent, attr, lin)
    if hasattr(m, "mirror_bases"):
        del m.mirror_bases
    return m


@torch.no_grad()
def refine_bases(model: nn.Module, structure: dict) -> float:
    """Make the stored dense bases exact for the structure's involutions in the bank's dtype (float64 audits of a
    model trained in float32): Q_s <- polar(Pi_s Q_s) with Pi_+- = (I +- R)/2 computed in float64 from the structure
    (A_star, maps_after; mirror.heal_structured.load_structure format), so each basis spans exactly its eigenspace and
    moves by no more than its own rounding; the butterfly scales are rewritten as 1/sqrt(2). Call after
    model.double(). Returns the largest change of a basis entry."""
    bank = model.mirror_bases
    blocks = [int(b) for b in structure["blocks"]]
    R = {"A": exact_involution(structure["A_star"])}
    for i, b in enumerate(blocks):
        for w in ("B", "C"):
            R[f"{w}{b}"] = ST.head_involution(structure["maps_after"][i], w)
    worst = 0.0
    for key, meta in bank.meta.items():
        if meta["kind"] == "perm":
            r = 1.0 / math.sqrt(2.0)
            for nm, sgn in (("scale_p", 1.0), ("scale_m", -1.0)):
                buf = bank._b(key, nm)
                pi, pj = bank._b(key, "pi"), bank._b(key, "pj")
                new = buf.clone()
                new[pi] = r
                new[pj] = sgn * r
                if nm == "scale_p":
                    new[bank._b(key, "ps")] = 1.0
                worst = max(worst, float((new - buf).abs().max()) if buf.numel() else 0.0)
                buf.copy_(new)
            continue
        if key not in R:
            raise KeyError(f"the structure has no involution for basis {key}")
        Rm = torch.as_tensor(R[key], dtype=torch.float64)
        eye = torch.eye(Rm.shape[0], dtype=torch.float64)
        for nm, sgn in (("Qp", 1.0), ("Qm", -1.0)):
            buf = bank._b(key, nm)
            if buf.numel() == 0:
                continue
            Q = buf.detach().to(device="cpu", dtype=torch.float64)
            X = 0.5 * (eye + sgn * Rm) @ Q
            U, _, Vh = torch.linalg.svd(X, full_matrices=False)
            new = U @ Vh
            worst = max(worst, float((new - Q).abs().max()))
            buf.copy_(new.to(dtype=buf.dtype, device=buf.device))
    return worst
