"""A09: exactness audit of Mirror Surgery at three levels (FINAL_PHASE_PLAN.md v2, item A09; PREREG_FINAL_PHASE.md,
Sections 1 and 2.6). The proposition and the per-component conditions are in docs/exactness.md; this module extends
the MUST-02 audit (scripts/must02_exactness.py) with precise levels, a rounding floor, the even/odd form and a bitwise
check.

Levels (never conflated):
  E1  exact in real arithmetic: Proposition 3, a theorem about the weights. Checked in float64 on weights that meet
      the conditions to float64 rounding (precision "float64_exact": the stored weights re-projected once in float64,
      a change of a few float32 units in the last place, see reproject_exact). A09-1: the largest relative
      equivariance error is at most 1e-10 at every tensor.
  E2  numerically exact at a precision: the equivariance error at that precision is no larger than the rounding error
      the model itself makes there. A09-2: at every tensor the float32 equivariance error is at most 2 times the
      model's own float32-versus-float64 deviation, and the float32 k-NN flip consistency is 100% for both forms
      (feature_flip_check with a k-NN function; the bank lives in the evaluation code). The deviation is recorded
      against two float64 references, because the wording allows both: "cast", the same stored weights run in
      float64 (arithmetic rounding only; dev_*), and "exact", the float64 re-projected model (arithmetic plus the
      rounding of storing the projected weights in float32, which the float32 equivariance error also contains;
      dev_exact_*). score_a09 scores "cast" by default and reports the other beside it.
  E3  bitwise identical: features or predictions of x and mirror(x) equal bit for bit. Only the even/odd form can
      reach it in places, because there the mirror is a sign flip of the odd coordinates, which is exact in floating
      point (bitwise_check reports where it holds and why it does not elsewhere).

Forms
  dense      the model as stored: dense projected weights, the original architecture (folded norms, even head).
  even_odd   mirror.efficient.build_fast_vit of the same weights (one basis, no rotations; `align` as timed).
  original   the unconverted model, for its own rounding floor (no equivariance is expected of it).

Precisions
  float64        the stored weights cast to float64 (an exact cast): the reference of the rounding floor;
  float64_exact  the stored weights re-projected in float64 (the E1 check; its deviation from "float64" is the
                 effect of storing the weights in float32);
  float32        the stored weights, TF32 off;
  bf16_autocast  float32 weights under torch.autocast(bfloat16): the precision of the k-NN protocol;
  bf16           weights and inputs in bfloat16: the precision of the latency benchmark.

What is measured at every tensor (component output) and image: the equivariance error
||T out(x) - out(mirror x)|| / ||out(mirror x)|| over the whole tensor of the image (`eq_rel_tensor_max`, maximum over
images, the primary number as in MUST-02) and per token (`eq_rel_token_max`), with T the component's expected action
(stream x -> x A*; hidden units permuted; heads mapped by the head maps; token grid aligned; invariance after the even
projection; in the even/odd basis A* is a sign flip of the odd coordinates and twin units and heads swap), and the
deviation of the same tensor from the form's float64 run, ||out_p(x) - out_64(x)|| / ||out_64(x)|| (`dev_rel_tensor_max`,
maximum over the images and their mirrors). The precisions of one form run in lockstep, block by block, so only one
block's reference tensors are held at a time.

Inputs are normalised images (float64 from uint8 is best: every precision then starts from the same values). Every
batch runs as [x; mirror(x)] in one forward pass. Attention logits and probabilities are recomputed from the hooked
queries and keys in the run's precision, because fused kernels never materialise them.

Public API: AuditStructure, COMPONENTS, PRECISIONS, token_index, reproject_exact, build_even_odd, weight_conditions,
audit_forms, score_a09, feature_flip_check, make_knn_fn, compute_features, bitwise_check, snap_embedding_exact,
mirror_attention_parts, mirror_mlp_parts.
"""
from __future__ import annotations

import contextlib
import copy
import math
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...attention import attn_dims
from ...data import mirror
from ...fold import RMSNorm as FoldRMSNorm
from ...heal import MirrorMLPConstraint, exact_involution
from ...heal_attn import MirrorAttnConstraint
from ...m1 import project_embeddings, register_info

__all__ = [
    "AuditStructure", "COMPONENTS", "PRECISIONS", "DEFAULT_PRECISIONS", "A09_1_TOL", "A09_2_FACTOR", "token_index",
    "sign_vector", "reproject_exact", "build_even_odd", "weight_conditions", "audit_forms", "score_a09",
    "feature_flip_check", "make_knn_fn", "compute_features", "bitwise_check", "snap_embedding_exact",
    "mirror_attention_parts", "mirror_mlp_parts", "precision_context", "prepare_input",
]

PRECISIONS = ("float64", "float64_exact", "float32", "bf16_autocast", "bf16")
DEFAULT_PRECISIONS = ("float64", "float64_exact", "float32", "bf16_autocast")
A09_1_TOL = 1e-10          # pre-registered (PREREG_FINAL_PHASE.md, A09-1)
A09_2_FACTOR = 2.0         # pre-registered (A09-2)
NEAR_TIE = 1e-4            # PREREG_FINAL_PHASE.md Section 2.1 (relative top-two margin)

# ----------------------------------------------------------------------------------------------
# Components: the plan's list, the condition on the weights and why the relation passes (docs/exactness.md)
# ----------------------------------------------------------------------------------------------
COMPONENTS = OrderedDict([
    ("patch embedding", {
        "rows": ("patch_embed",),
        "condition": "flip_w(W) = A* W on the output channels and A* b = b (W the conv kernel, flip_w reverses its width)",
        "why": "patch (r, G-1-c) of the mirrored image is the width-flipped patch (r, c) of the image"}),
    ("position embeddings", {
        "rows": ("tokens0_patches",),
        "condition": "pos(r, G-1-c) = pos(r, c) A* on the grid used at the evaluation size",
        "why": "the patch token at (r, G-1-c) of the mirrored image is (embedding + position) of (r, c) times A*"}),
    ("class and register tokens", {
        "rows": ("tokens0_prefix",),
        "condition": "c A* = c (class token and its position row); r_pi(j) = r_j A* for registers (pairs swap)",
        "why": "the class token keeps its index; paired registers swap under the alignment"}),
    ("folded LayerNorm", {
        "rows": ("norm1", "norm2"),
        "condition": "every LayerNorm is an RMSNorm without affine (gamma, beta folded into the next linear layer)",
        "why": "RMSNorm(x A*) = RMSNorm(x) A* for any orthogonal A*, token by token"}),
    ("attention Q and K", {
        "rows": ("q", "k", "attn_logits", "attn_probs"),
        "condition": "Wq^g A* = B_h Wq^h, Wk^g A* = B_h Wk^h, bq^g = B_h bq^h, bk^g = B_h bk^h for g = sigma(h)",
        "why": "logits (B q)^T (B k) = q^T k, so head g on the mirror equals head h on the image with tokens aligned; "
               "softmax over keys commutes with the token permutation"}),
    ("attention V and O", {
        "rows": ("v", "head_out", "attn_proj"),
        "condition": "Wv^g A* = C_h Wv^h, bv^g = C_h bv^h, A* Wo^h = Wo^g C_h, A* bo = bo",
        "why": "the attention-weighted values of head g on the mirror are C_h times those of head h"}),
    ("head maps", {
        "rows": (),
        "condition": "sigma is an involution; B_h, C_h orthogonal, B_g = B_h^T on twin pairs (B = C = I after the "
                     "gauge), symmetric involutions on self-heads",
        "why": "the output projection sums over heads, so relabelling heads by sigma changes nothing"}),
    ("MLP fc1, GELU, fc2 and biases", {
        "rows": ("fc1", "act", "fc2"),
        "condition": "P W1 = W1 A*, P b1 = b1, A* W2 = W2 P, A* b2 = b2",
        "why": "fc1(x A*) = P fc1(x); GELU acts unit by unit and commutes with P; fc2 maps P g to fc2(g) A*"}),
    ("residual additions", {
        "rows": ("resid_attn", "block_out", "ls1", "ls2"),
        "condition": "the same A* on both branches and on the stream (LayerScale folded, gamma = 1)",
        "why": "x A* + f(x) A* = (x + f(x)) A*"}),
    ("final norm", {
        "rows": ("final_rms", "final_rms_cls"),
        "condition": "RMSNorm without affine",
        "why": "keeps the A* relation"}),
    ("even projection", {
        "rows": ("even_proj", "final_affine", "cls_feature"),
        "condition": "E = (I + A*)/2 between the final RMSNorm and the final affine",
        "why": "A* E = E, so the projected tokens are invariant; the per-channel affine keeps invariance"}),
    ("DeiT classifier head", {
        "rows": ("logits",),
        "condition": "none on the head: it reads the invariant class-token feature (fc_norm is Identity)",
        "why": "identical inputs give identical logits"}),
])
_ROW_GROUP = {r: g for g, v in COMPONENTS.items() for r in v["rows"]}
_ROW_GROUP.update({"tokens0": "position embeddings"})

SPACES = {
    "stream": "x -> x A* (tokens aligned)",
    "patch": "x -> x A* (patch grid aligned)",
    "heads": "head g -> M[sigma(g)] head sigma(g) (B for q, k; C for v)",
    "attention": "head g -> head sigma(g), query and key tokens aligned",
    "hidden": "unit i -> unit perm(i)",
    "invariant": "identity (tokens aligned)",
    "eo_stream": "odd coordinates change sign (tokens aligned)",
    "eo_patch": "odd coordinates change sign (patch grid aligned)",
    "eo_heads": "twin heads swap; odd coordinates of self-heads change sign",
    "eo_attention": "twin heads swap, query and key tokens aligned",
    "eo_hidden": "twin units (u + v, u - v) swap",
    "none": "no equivariance expected (rounding floor only)",
}


# ----------------------------------------------------------------------------------------------
# Structure of an exactly equivariant model
# ----------------------------------------------------------------------------------------------
@dataclass
class AuditStructure:
    """The exact symmetry of a converted model: A* (exact involution, float64), the neuron pairings and the head
    maps valid for the stored weights (after the twin gauge) of blocks 0..k-1, the register involution."""

    A: np.ndarray
    perms: list
    maps: list
    reg_perm: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    k: int | None = None
    source: dict = field(default_factory=dict)

    def __post_init__(self):
        self.A = exact_involution(np.asarray(self.A, dtype=np.float64))
        self.reg_perm = np.asarray(self.reg_perm if self.reg_perm is not None else [], dtype=np.int64)
        self.perms = [np.asarray(p, dtype=np.int64) for p in self.perms]
        if self.k is None:
            self.k = len(self.perms)

    @property
    def d(self) -> int:
        return int(self.A.shape[0])

    @property
    def d_plus(self) -> int:
        return int(round((float(np.trace(self.A)) + self.d) / 2))

    @classmethod
    def from_info(cls, info: dict, source: str = "info") -> "AuditStructure":
        """From a structure dict: mirror.heal_structured.load_healed_model's info["structure"] (A_star, perms,
        maps_after, reg_perm, blocks), the heal agent's StudentBuild.structure, or build_vexact's info plus parts."""
        st = info.get("structure", info) if isinstance(info, dict) else info
        A = st.get("A_star", st.get("A"))
        maps = st.get("maps_after", st.get("heads_after", st.get("maps")))
        if A is None or maps is None or st.get("perms") is None:
            raise ValueError("the structure needs A_star (or A), perms and maps_after (head maps after the gauge)")
        blocks = st.get("blocks")
        k = len(blocks) if blocks is not None else len(st["perms"])
        if blocks is not None and [int(b) for b in blocks] != list(range(k)):
            raise ValueError(f"converted blocks {list(blocks)} do not start at block 0")
        return cls(A=A, perms=list(st["perms"])[:k], maps=list(maps)[:k], reg_perm=st.get("reg_perm"), k=k,
                   source={"from": source})


def token_index(grid_hw, n_prefix: int, reg_perm=None, reg_start: int = 1) -> torch.Tensor:
    """idx with t_mirror[:, idx] = the mirrored image's tokens at the original's positions (patch grid un-flipped,
    register j replaced by register pi(j), class token unchanged); idx is an involution (as in MUST-02)."""
    H, W = (int(v) for v in grid_hw)
    idx = np.arange(int(n_prefix) + H * W)
    idx[n_prefix:] = n_prefix + np.arange(H * W).reshape(H, W)[:, ::-1].reshape(-1)
    if reg_perm is not None and len(reg_perm):
        R = len(reg_perm)
        idx[reg_start:reg_start + R] = reg_start + np.asarray(reg_perm, dtype=np.int64)
    return torch.as_tensor(idx, dtype=torch.long)


def sign_vector(d: int, dp: int, dtype=torch.float64, device=None) -> torch.Tensor:
    """+1 on the d+ even coordinates, -1 on the odd ones (A* in its eigenbasis)."""
    s = torch.ones(int(d), dtype=dtype, device=device)
    s[int(dp):] = -1
    return s


# ----------------------------------------------------------------------------------------------
# Model forms
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def reproject_exact(model: nn.Module, adapter, st: AuditStructure, dtype=torch.float64) -> nn.Module:
    """A copy of a converted model in `dtype` whose weights meet the conditions to that dtype's rounding: MLP and
    attention of blocks 0..k-1 re-projected with A* (the healing constraints of mirror/heal.py and
    mirror/heal_attn.py, computed in float64), embeddings re-projected (mirror.m1.project_embeddings) and the even
    head rebuilt from A* in float64. The projection is idempotent, so this moves each weight by its float32
    rounding only (the change is reported by the audit as the deviation of float64_exact from float64)."""
    m = copy.deepcopy(model).to(dtype)
    k = int(st.k)
    blocks = list(range(k))
    A = st.A
    MirrorMLPConstraint(blocks, [A] * k, [A] * k, st.perms[:k], dtype=torch.float64)(m)
    MirrorAttnConstraint(blocks, [A] * k, [A] * k, st.maps[:k], dtype=torch.float64)(m)
    reg = st.reg_perm if len(st.reg_perm) else None
    project_embeddings(m, adapter, A, reg)
    even = getattr(getattr(m, "norm", None), "even", None)
    if even is not None:
        P = 0.5 * (np.eye(A.shape[0]) + A)
        even.P.copy_(torch.as_tensor(0.5 * (P + P.T), dtype=even.P.dtype, device=even.P.device))
    return m.eval()


@torch.no_grad()
def build_even_odd(model: nn.Module, st: AuditStructure, align: int = 8, tol: float = 1e-5):
    """The even/odd form (mirror.efficient.FastViT) of a fully converted model, as MUST-09 timed it
    (scripts/must09_heal.even_odd_form): one basis eig(A*) for every layer, twin gauge maps, Wo structured."""
    from ...efficient import Basis, BlockSpec, build_fast_vit

    L = len(model.blocks)
    if int(st.k) != L:
        raise ValueError(f"the even/odd form needs the full conversion (k = {st.k}, depth {L})")
    A = st.A
    specs = [BlockSpec(("mirror", A, A, st.maps[b]), ("mirror", A, A, st.perms[b])) for b in range(L)]
    return build_fast_vit(model, specs, [Basis.of(A)] * (L + 1), align=int(align), wo_mode="fold", tol=tol)


def precision_context(precision: str, device) -> contextlib.AbstractContextManager:
    if precision == "bf16_autocast":
        return torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def prepare_input(x: torch.Tensor, precision: str) -> torch.Tensor:
    return x.to({"float64": torch.float64, "float64_exact": torch.float64, "float32": torch.float32,
                 "bf16_autocast": torch.float32, "bf16": torch.bfloat16}[precision])


def _model_dtype(precision: str):
    return {"float64": torch.float64, "float64_exact": torch.float64, "float32": torch.float32,
            "bf16_autocast": torch.float32, "bf16": torch.bfloat16}[precision]


# ----------------------------------------------------------------------------------------------
# Even/odd internals (the same operations as mirror.efficient MirrorAttention and MirrorMLP forward)
# ----------------------------------------------------------------------------------------------
def mirror_attention_parts(mod, x: torch.Tensor, key_perm: torch.Tensor | None = None, key_perm_from: int = 0) -> dict:
    """MirrorAttention.forward step by step: ze, zo (the two GEMMs), q, k, v [B, H, T, dh] in the even/odd head
    layout, o [B, T, H, dh] and the output. `key_perm` reorders keys and values along the token axis before the
    attention, for the images from index `key_perm_from` on (the same function; used to test the summation order
    over keys without changing the batch the GEMMs see)."""
    ze = F.linear(x[..., :mod.ke], mod.We, mod.be)
    zo = F.linear(x[..., mod.ko:], mod.Wodd)
    parts = []
    if mod.np_:
        u = ze[..., :mod.P].unflatten(-1, (3, mod.np_, mod.dh))
        v = zo[..., :mod.P].unflatten(-1, (3, mod.np_, mod.dh))
        parts += [u + v, u - v]
    if mod.ns:
        c = torch.cat([ze[..., mod.P:mod.re], zo[..., mod.P:mod.ro]], dim=-1)
        parts.append(c.index_select(-1, mod.self_idx).unflatten(-1, (3, mod.ns, mod.dh)))
    heads = torch.cat(parts, dim=-2) if len(parts) > 1 else parts[0]
    q, k, v = heads.permute(2, 0, 3, 1, 4).unbind(0)
    if key_perm is not None:
        p0 = int(key_perm_from)
        k = torch.cat([k[:p0], k[p0:][:, :, key_perm]])
        v = torch.cat([v[:p0], v[p0:][:, :, key_perm]])
    o = F.scaled_dot_product_attention(q, k, v, scale=mod.scale).transpose(1, 2)
    if mod.out_dense:
        out = F.linear(o.flatten(2), mod.Wd, mod.bd)
    else:
        ein, oin = [], []
        if mod.np_:
            oh, og = o[:, :, :mod.np_], o[:, :, mod.np_:2 * mod.np_]
            ein.append((oh + og).flatten(2))
            oin.append((oh - og).flatten(2))
        if mod.ns:
            os_ = o[:, :, 2 * mod.np_:].flatten(2)
            ein.append(os_.index_select(-1, mod.oe_idx))
            oin.append(os_.index_select(-1, mod.oo_idx))
        ein = torch.cat(ein, dim=-1) if len(ein) > 1 else ein[0]
        oin = torch.cat(oin, dim=-1) if len(oin) > 1 else oin[0]
        if mod.pad_e:
            ein = F.pad(ein, (0, mod.pad_e))
        if mod.pad_o:
            oin = F.pad(oin, (0, mod.pad_o))
        ye = F.linear(ein, mod.Oe, mod.obe)
        yo = F.linear(oin, mod.Oo)
        out = torch.cat([ye[..., :mod.dpo], yo[..., mod.cut:]], dim=-1)
    return {"ze": ze, "zo": zo, "q": q, "k": k, "v": v, "o": o, "out": out}


def mirror_mlp_parts(mod, x: torch.Tensor) -> dict:
    """MirrorMLP.forward step by step: pre-activations [u + v, u - v, self] and GELU outputs in the same layout."""
    ze = F.linear(x[..., :mod.ke], mod.E1, mod.b1)
    zo = F.linear(x[..., mod.ko:], mod.O1)
    u = ze[..., :mod.npad]
    pa, pb, ps = u + zo, u - zo, ze[..., mod.npad:]
    ga, gb, gs = mod.act(pa), mod.act(pb), mod.act(ps)
    ein = torch.cat([ga + gb, gs], dim=-1)
    ye = F.linear(ein, mod.E2, mod.b2)
    yo = F.linear(ga - gb, mod.O2)
    out = torch.cat([ye[..., :mod.dpo], yo[..., mod.cut:]], dim=-1)
    return {"pre": torch.cat([pa, pb, ps], dim=-1), "act": torch.cat([ga, gb, gs], dim=-1), "out": out}


def _eo_head_action(mod) -> tuple:
    """(head permutation, signs {q, k, v} [H, dh]) of a MirrorAttention: twin heads (u + v, u - v) swap; the odd
    coordinates of a self-head (gathered from the odd GEMM) change sign."""
    H, dh, np_, ns = int(mod.H), int(mod.dh), int(mod.np_), int(mod.ns)
    perm = np.concatenate([np.arange(np_, 2 * np_), np.arange(0, np_), np.arange(2 * np_, H)]).astype(np.int64)
    dev = mod.We.device
    signs = {}
    Le = int(mod.re) - int(mod.P)
    if ns:
        sg = torch.where(mod.self_idx < Le, 1.0, -1.0).to(torch.float64).reshape(3, ns, dh)
    for i, t in enumerate(("q", "k", "v")):
        S = torch.ones(H, dh, dtype=torch.float64, device=dev)
        if ns:
            S[2 * np_:] = sg[i].to(dev)
        signs[t] = S
    return torch.as_tensor(perm, device=dev), signs


def _eo_hidden_perm(mod) -> torch.Tensor:
    n, w = int(mod.npad), int(mod.E1.shape[0])
    p = np.concatenate([np.arange(n, 2 * n), np.arange(0, n), np.arange(2 * n, n + w)])
    return torch.as_tensor(p, dtype=torch.long, device=mod.E1.device)


# ----------------------------------------------------------------------------------------------
# Recording
# ----------------------------------------------------------------------------------------------
def _new_acc(meta: dict) -> dict:
    return {**meta, "n_images": 0, "eq_rel_tensor_max": 0.0, "eq_rel_token_max": 0.0, "eq_worst_image": None,
            "eq_max_abs_err": 0.0, "eq_max_abs_ref": 0.0, "eq_err_sq": 0.0, "eq_ref_sq": 0.0,
            "dev_rel_tensor_max": float("nan"), "dev_rel_token_max": float("nan"), "dev_n_images": 0,
            "dev_exact_rel_tensor_max": float("nan"), "dev_exact_rel_token_max": float("nan")}


class _Recorder:
    """Running maxima for one (form, precision) run. tap(key, raw, action) gets the raw output of a component on
    the batch [x; mirror x] with the token axis at position 1 (features as [2B, 1, C])."""

    def __init__(self, form: str, precision: str, equivariant: bool, is_ref: bool, store: dict | None,
                 store_exact: dict | None = None, writes_exact: bool = False):
        self.form, self.precision = form, precision
        self.equivariant, self.is_ref, self.store = bool(equivariant), bool(is_ref), store
        self.store_exact, self.writes_exact = store_exact, bool(writes_exact)
        self.acc: OrderedDict = OrderedDict()
        self.B = 0
        self.offset = 0

    def add(self, key, meta):
        if key not in self.acc:
            self.acc[key] = _new_acc(meta)

    def tap(self, key, raw: torch.Tensor, action) -> None:
        a = self.acc[key]
        B = self.B
        if self.equivariant and action is not None:
            X_t, Xt = action(raw)
            E = X_t - Xt
            n, T = E.shape[0], E.shape[1]
            e_img = E.reshape(n, -1).norm(dim=1)
            r_img = Xt.reshape(n, -1).norm(dim=1)
            rel_img = e_img / r_img.clamp_min(1e-300)
            rel_tok = E.reshape(n, T, -1).norm(dim=2) / Xt.reshape(n, T, -1).norm(dim=2).clamp_min(1e-300)
            i = int(rel_img.argmax())
            if float(rel_img[i]) >= a["eq_rel_tensor_max"]:
                a["eq_rel_tensor_max"], a["eq_worst_image"] = float(rel_img[i]), self.offset + i
            a["eq_rel_token_max"] = max(a["eq_rel_token_max"], float(rel_tok.max()))
            a["eq_max_abs_err"] = max(a["eq_max_abs_err"], float(E.abs().max()))
            a["eq_max_abs_ref"] = max(a["eq_max_abs_ref"], float(Xt.abs().max()))
            a["eq_err_sq"] += float((E * E).sum())
            a["eq_ref_sq"] += float((Xt * Xt).sum())
        a["n_images"] += B
        if self.is_ref:
            if self.store is not None:
                self.store[key] = raw.detach()
            return
        if self.store is not None and key in self.store:
            self._dev(a, "dev", raw, self.store[key])
            a["dev_n_images"] += int(raw.shape[0])
        if self.writes_exact and self.store_exact is not None:
            self.store_exact[key] = raw.detach()
        elif self.store_exact is not None and key in self.store_exact:
            self._dev(a, "dev_exact", raw, self.store_exact[key])

    @staticmethod
    def _dev(a: dict, prefix: str, raw: torch.Tensor, ref: torch.Tensor) -> None:
        R = ref.double()
        D = raw.double() - R
        n, T = D.shape[0], D.shape[1]
        dev_img = D.reshape(n, -1).norm(dim=1) / R.reshape(n, -1).norm(dim=1).clamp_min(1e-300)
        dev_tok = D.reshape(n, T, -1).norm(dim=2) / R.reshape(n, T, -1).norm(dim=2).clamp_min(1e-300)
        for name, val in ((f"{prefix}_rel_tensor_max", float(dev_img.max())),
                          (f"{prefix}_rel_token_max", float(dev_tok.max()))):
            a[name] = val if not math.isfinite(a[name]) else max(a[name], val)


def _meta(stage, b, component, space, in_span, note="") -> dict:
    group = _ROW_GROUP.get(component, "")
    return {"stage": stage, "block": None if b is None else b + 1, "code_index": b, "component": component,
            "plan_component": group, "condition": COMPONENTS[group]["condition"] if group else "",
            "space": space, "action": SPACES[space], "in_span": bool(in_span), "note": note}


class _Run:
    """One (form, precision) model with its hooks. Forward pieces: embed(x), block(b, t), final(t)."""

    def __init__(self, form: str, precision: str, model: nn.Module, adapter, st: AuditStructure | None,
                 head: nn.Module | None, device, is_ref: bool, store: dict | None, dp: int | None = None,
                 store_exact: dict | None = None):
        self.form, self.precision, self.model, self.adapter, self.st = form, precision, model, adapter, st
        self.head, self.device = head, torch.device(device)
        self.rec = _Recorder(form, precision, equivariant=(form != "original"), is_ref=is_ref, store=store,
                             store_exact=store_exact, writes_exact=(precision == "float64_exact"))
        self.handles = []
        self.L = len(model.blocks)
        self.k = self.L if st is None else int(st.k)
        dev = self.device
        rp = st.reg_perm if (st is not None and len(st.reg_perm)) else None
        reg = register_info(adapter.model, adapter)
        self.idx = token_index(adapter.grid_hw, adapter.n_prefix_tokens, rp, int(reg["start"])).to(dev)
        self.idx_patch = token_index(adapter.grid_hw, 0).to(dev)
        self.n_prefix = int(adapter.n_prefix_tokens)
        if st is not None:
            self.A = torch.as_tensor(st.A, dtype=torch.float64, device=dev)
            self.perms = [torch.as_tensor(p, dtype=torch.long, device=dev) for p in st.perms]
            self.sigma = [torch.as_tensor(np.asarray(hm.sigma), dtype=torch.long, device=dev) for hm in st.maps]
            self.Bm = [torch.as_tensor(np.asarray(hm.B, dtype=np.float64), device=dev) for hm in st.maps]
            self.Cm = [torch.as_tensor(np.asarray(hm.C, dtype=np.float64), device=dev) for hm in st.maps]
            self.s = sign_vector(st.d, st.d_plus if dp is None else dp, device=dev)

    # ----- actions (return (T out(x), aligned out(mirror x)) in float64, token axis 1) -------------
    def _halves(self, t, idx):
        B = self.rec.B
        if t.shape[1] == 1:                       # a single token (class token, feature): nothing to align
            return t[:B].double(), t[B:].double()
        return t[:B].double(), t[B:].double()[:, idx]

    def act_stream(self, t):
        X, Xt = self._halves(t, self.idx)
        return (X * self.s if self.form == "even_odd" else X @ self.A), Xt

    def act_patch(self, t):
        if t.dim() == 4:
            t = t.reshape(t.shape[0], -1, t.shape[-1])
        X, Xt = self._halves(t, self.idx_patch)
        return (X * self.s if self.form == "even_odd" else X @ self.A), Xt

    def act_invariant(self, t):
        return self._halves(t, self.idx)

    def act_hidden(self, b):
        def f(t):
            X, Xt = self._halves(t, self.idx)
            return X[..., self.perms[b]], Xt
        return f

    def act_heads(self, M, sigma):
        def f(t):
            X, Xt = self._halves(t, self.idx)
            return torch.einsum("bthd,hed->bthe", X, M)[:, :, sigma], Xt
        return f

    def act_attention(self, sigma):
        def f(t):
            B = self.rec.B
            X = t[:B].double()[:, :, sigma]
            Xt = t[B:].double()[:, self.idx][:, :, :, self.idx]
            return X, Xt
        return f

    def act_eo_heads(self, perm, S):
        def f(t):
            X, Xt = self._halves(t, self.idx)
            return X[:, :, perm] * S, Xt
        return f

    def act_eo_hidden(self, hperm):
        def f(t):
            X, Xt = self._halves(t, self.idx)
            return X[..., hperm], Xt
        return f

    # ----- hooks --------------------------------------------------------------------------------
    def _reg(self, key, meta):
        self.rec.add(key, meta)
        return key

    def _out(self, module, key, action, transform=None):
        def hook(_m, _args, out):
            t = out if transform is None else transform(out)
            self.rec.tap(key, t, action)
        self.handles.append(module.register_forward_hook(hook))

    def _in(self, module, key, action):
        def hook(_m, args):
            self.rec.tap(key, args[0], action)
        self.handles.append(module.register_forward_pre_hook(hook))

    def attach(self) -> None:
        if self.form == "even_odd":
            self._attach_even_odd()
        else:
            self._attach_dense()

    def _sp(self, space):
        return "none" if self.form == "original" else space

    def _act(self, fn):
        return None if self.form == "original" else fn

    def _attach_dense(self) -> None:
        m = self.model
        orig = self.form == "original"
        key = self._reg(("embed", None, "patch_embed"), _meta("embed", None, "patch_embed", self._sp("patch"), True))
        self._out(m.patch_embed, key, self._act(self.act_patch))
        for name in ("tokens0", "tokens0_prefix", "tokens0_patches"):
            if name != "tokens0_prefix" or self.n_prefix:
                self._reg(("embed", None, name), _meta("embed", None, name, self._sp("stream"), True))
        for b, blk in enumerate(m.blocks):
            self._attach_dense_block(b, blk, b < self.k and not orig)
        norm = m.norm
        full = self.k == self.L and not orig
        if isinstance(norm, nn.Sequential) and hasattr(norm, "rms"):
            key = self._reg(("final", None, "final_rms"), _meta("final", None, "final_rms", self._sp("stream"), full))
            self._out(norm.rms, key, self._act(self.act_stream))
            if hasattr(norm, "even"):
                for name, mod in (("even_proj", norm.even), ("final_affine", norm.affine)):
                    key = self._reg(("final", None, name), _meta("final", None, name, "invariant", full))
                    self._out(mod, key, self.act_invariant)
        else:
            key = self._reg(("final", None, "final_norm"), _meta("final", None, "final_norm", self._sp("stream"), False,
                                                                 "not a folded norm"))
            self._out(norm, key, None)
        has_even = hasattr(norm, "even")
        self._reg(("final", None, "cls_feature"), _meta("final", None, "cls_feature",
                                                         self._sp("invariant"), full and has_even))
        if self.head is not None:
            self._reg(("final", None, "logits"), _meta("final", None, "logits", self._sp("invariant"),
                                                        full and has_even))

    def _attach_dense_block(self, b, blk, in_span) -> None:
        sp = (lambda s: s) if in_span else (lambda s: "none")
        act = (lambda f: f) if in_span else (lambda f: None)

        def add(name, space, note=""):
            return self._reg(("block", b, name), _meta("block", b, name, sp(space), in_span, note))

        self._out(blk.norm1, add("norm1", "stream", "RMSNorm (LayerNorm folded)"), act(self.act_stream))
        attn = blk.attn
        H, dh = attn_dims(blk)
        keys = {n: add(n, "heads") for n in ("q", "k", "v")}
        keys["logits"] = add("attn_logits", "attention", "recomputed from the hooked q and k")
        keys["probs"] = add("attn_probs", "attention", "softmax of the recomputed logits")
        scale = float(attn.scale)

        def on_qkv(_m, _args, y, b=b, H=H, dh=dh, keys=keys, scale=scale, in_span=in_span):
            y = y.reshape(y.shape[0], y.shape[1], 3, H, dh)
            q, k_, v = y[:, :, 0], y[:, :, 1], y[:, :, 2]
            if in_span:
                self.rec.tap(keys["q"], q, self.act_heads(self.Bm[b], self.sigma[b]))
                self.rec.tap(keys["k"], k_, self.act_heads(self.Bm[b], self.sigma[b]))
                self.rec.tap(keys["v"], v, self.act_heads(self.Cm[b], self.sigma[b]))
            else:
                for n, t in (("q", q), ("k", k_), ("v", v)):
                    self.rec.tap(keys[n], t, None)
            logits = (q.transpose(1, 2) * scale) @ k_.transpose(1, 2).transpose(-2, -1)
            probs = logits.softmax(dim=-1)
            a = self.act_attention(self.sigma[b]) if in_span else None
            self.rec.tap(keys["logits"], logits.permute(0, 2, 1, 3), a)
            self.rec.tap(keys["probs"], probs.permute(0, 2, 1, 3), a)

        self.handles.append(attn.qkv.register_forward_hook(on_qkv))
        ko = add("head_out", "heads", "attention-weighted values per head (input of attn.proj)")

        def on_proj_in(_m, args, b=b, H=H, dh=dh, ko=ko, in_span=in_span):
            t = args[0]
            t = t.reshape(t.shape[0], t.shape[1], H, dh)
            self.rec.tap(ko, t, self.act_heads(self.Cm[b], self.sigma[b]) if in_span else None)

        self.handles.append(attn.proj.register_forward_pre_hook(on_proj_in))
        self._out(attn.proj, add("attn_proj", "stream", "attention branch"), act(self.act_stream))
        if getattr(blk, "ls1", None) is not None and not isinstance(blk.ls1, nn.Identity):
            self._out(blk.ls1, add("ls1", "stream", "LayerScale"), act(self.act_stream))
        self._in(blk.norm2, add("resid_attn", "stream", "x + attention branch"), act(self.act_stream))
        self._out(blk.norm2, add("norm2", "stream", "RMSNorm (LayerNorm folded)"), act(self.act_stream))
        self._out(blk.mlp.fc1, add("fc1", "hidden", "pre-activation"), self.act_hidden(b) if in_span else None)
        self._out(blk.mlp.act, add("act", "hidden", "GELU output"), self.act_hidden(b) if in_span else None)
        self._out(blk.mlp.fc2, add("fc2", "stream", "MLP branch"), act(self.act_stream))
        if getattr(blk, "ls2", None) is not None and not isinstance(blk.ls2, nn.Identity):
            self._out(blk.ls2, add("ls2", "stream", "LayerScale"), act(self.act_stream))
        self._out(blk, add("block_out", "stream", "x + MLP branch: the block output"), act(self.act_stream))

    def _attach_even_odd(self) -> None:
        fv = self.model
        key = self._reg(("embed", None, "patch_embed"), _meta("embed", None, "patch_embed", "eo_patch", True,
                                                               "in the even/odd basis"))
        pe = fv.embed.base.patch_embed
        self._out(pe, key, self.act_patch)
        for name in ("tokens0", "tokens0_prefix", "tokens0_patches"):
            if name != "tokens0_prefix" or self.n_prefix:
                self._reg(("embed", None, name), _meta("embed", None, name, "eo_stream", True))
        for b, blk in enumerate(fv.blocks):
            self._attach_eo_block(b, blk)
        has_cls = getattr(self.adapter.model, "cls_token", None) is not None
        key = self._reg(("final", None, "final_rms_cls"), _meta("final", None, "final_rms_cls", "eo_stream", has_cls,
                                                                 "final RMSNorm of the class token"))
        self._out(fv.head.norm, key, self.act_stream, transform=lambda o: o[:, None])
        inv = has_cls and int(fv.head.m) < int(self.st.d)          # an even head keeps only the even rows
        self._reg(("final", None, "cls_feature"), _meta("final", None, "cls_feature", "invariant", inv,
                                                         "even rows, out of the basis, final affine"))
        if self.head is not None:
            self._reg(("final", None, "logits"), _meta("final", None, "logits", "invariant", inv))

    def _attach_eo_block(self, b, blk) -> None:
        def add(name, space, note=""):
            return self._reg(("block", b, name), _meta("block", b, name, space, True, note))

        self._out(blk.norm1, add("norm1", "eo_stream", "RMSNorm"), self.act_stream)
        mod = blk.attn
        perm_h, S = _eo_head_action(mod)
        keys = {n: add(n, "eo_heads") for n in ("q", "k", "v")}
        keys["logits"] = add("attn_logits", "eo_attention", "recomputed from q and k")
        keys["probs"] = add("attn_probs", "eo_attention", "softmax of the recomputed logits")
        keys["o"] = add("head_out", "eo_heads", "attention-weighted values per head")
        keys["out"] = add("attn_proj", "eo_stream", "attention branch")
        scale = float(mod.scale)

        def on_attn(m_, args, out, keys=keys, perm_h=perm_h, S=S, scale=scale):
            p = mirror_attention_parts(m_, args[0])
            for n in ("q", "k", "v"):
                self.rec.tap(keys[n], p[n].transpose(1, 2), self.act_eo_heads(perm_h, S[n]))
            logits = (p["q"] * scale) @ p["k"].transpose(-2, -1)
            a = self.act_attention(perm_h)
            self.rec.tap(keys["logits"], logits.permute(0, 2, 1, 3), a)
            self.rec.tap(keys["probs"], logits.softmax(dim=-1).permute(0, 2, 1, 3), a)
            self.rec.tap(keys["o"], p["o"], self.act_eo_heads(perm_h, S["v"]))
            self.rec.tap(keys["out"], out, self.act_stream)

        self.handles.append(mod.register_forward_hook(on_attn))
        self._in(blk.norm2, add("resid_attn", "eo_stream", "x + attention branch"), self.act_stream)
        self._out(blk.norm2, add("norm2", "eo_stream", "RMSNorm"), self.act_stream)
        mlp = blk.mlp
        hperm = _eo_hidden_perm(mlp)
        k1, ka, k2 = add("fc1", "eo_hidden", "pre-activations [u + v, u - v, self]"), \
            add("act", "eo_hidden", "GELU outputs"), add("fc2", "eo_stream", "MLP branch")

        def on_mlp(m_, args, out, k1=k1, ka=ka, k2=k2, hperm=hperm):
            p = mirror_mlp_parts(m_, args[0])
            self.rec.tap(k1, p["pre"], self.act_eo_hidden(hperm))
            self.rec.tap(ka, p["act"], self.act_eo_hidden(hperm))
            self.rec.tap(k2, out, self.act_stream)

        self.handles.append(mlp.register_forward_hook(on_mlp))
        self._out(blk, add("block_out", "eo_stream", "the block output"), self.act_stream)

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    # ----- forward pieces -----------------------------------------------------------------------
    def ctx(self):
        return precision_context(self.precision, self.device)

    def embed(self, x):
        if self.form == "even_odd":
            t = self.model.embed(x)
        else:
            t = self.adapter.layer0_tokens(x, self.model)
        act = self.act_stream if self.form != "original" else None
        self.rec.tap(("embed", None, "tokens0"), t, act)
        p = self.n_prefix
        if p:
            pre_act = (lambda u: self.act_stream_prefix(u)) if act is not None else None
            self.rec.tap(("embed", None, "tokens0_prefix"), t[:, :p], pre_act)
        self.rec.tap(("embed", None, "tokens0_patches"), t[:, p:], self.act_patch if act is not None else None)
        return t

    def act_stream_prefix(self, u):
        idx = self.idx[:self.n_prefix]
        X, Xt = self._halves(u, idx)
        return (X * self.s if self.form == "even_odd" else X @ self.A), Xt

    def block(self, b, t):
        return self.model.blocks[b](t)

    def final(self, t):
        if self.form == "even_odd":
            f = self.model.head(t)
        else:
            f = self.model.norm(t)[:, 0]
        full = self.form != "original"
        self.rec.tap(("final", None, "cls_feature"), f[:, None], self.act_invariant if full else None)
        if self.head is not None:
            logits = self.head(f)
            self.rec.tap(("final", None, "logits"), logits[:, None], self.act_invariant if full else None)
        return f


def _rows_of(run: _Run) -> list:
    out = []
    for a in run.rec.acc.values():
        r = {k: v for k, v in a.items() if k not in ("eq_err_sq", "eq_ref_sq")}
        r["eq_rel_fro_all"] = math.sqrt(a["eq_err_sq"] / a["eq_ref_sq"]) if a["eq_ref_sq"] > 0 else float("nan")
        r["form"], r["precision"] = run.form, run.precision
        if run.form == "original" or r["space"] == "none":
            for k_ in ("eq_rel_tensor_max", "eq_rel_token_max", "eq_max_abs_err", "eq_max_abs_ref", "eq_rel_fro_all"):
                r[k_] = float("nan")
        eq = r["eq_rel_tensor_max"]
        for col, dcol in (("ratio_eq_to_dev", "dev_rel_tensor_max"), ("ratio_eq_to_dev_exact", "dev_exact_rel_tensor_max")):
            dev = r[dcol]
            r[col] = (eq / dev if (math.isfinite(eq) and math.isfinite(dev) and dev > 0)
                      else (0.0 if eq == 0 else float("nan")))
        out.append(r)
    return out


# ----------------------------------------------------------------------------------------------
# The lockstep audit
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def _lockstep(runs: list, batches) -> None:
    """Runs of one form, the reference first: embeddings, then block by block, then the final pieces, for every
    batch [x; mirror x]. The reference's tensors of the current stage are kept for the deviation and then freed."""
    stores = [s for s in (runs[0].rec.store, runs[0].rec.store_exact) if s is not None]
    for r in runs:
        r.attach()
    try:
        offset = 0
        for x in batches:
            x = x.to(runs[0].device)
            xx = torch.cat([x, mirror(x)], dim=0)
            B = int(x.shape[0])
            for r in runs:
                r.rec.B, r.rec.offset = B, offset
            toks = {}
            for r in runs:
                with r.ctx():
                    toks[r] = r.embed(prepare_input(xx, r.precision))
            for s_ in stores:
                s_.clear()
            for b in range(runs[0].L):
                for r in runs:
                    with r.ctx():
                        toks[r] = r.block(b, toks[r])
                for s_ in stores:
                    s_.clear()
            for r in runs:
                with r.ctx():
                    r.final(toks[r])
            for s_ in stores:
                s_.clear()
            offset += B
    finally:
        for r in runs:
            r.detach()


def _head_copy(head: nn.Module | None, precision: str):
    if head is None:
        return None
    return copy.deepcopy(head).to(_model_dtype(precision)).eval()


@torch.no_grad()
def audit_forms(adapter, model: nn.Module, st: AuditStructure, batches, *, forms=("dense", "even_odd"),
                precisions=DEFAULT_PRECISIONS, original: nn.Module | None = None, head: nn.Module | None = None,
                align: int = 8, device=None, even_odd: nn.Module | None = None, log=None) -> dict:
    """The A09 per-component audit. `model` is the converted model as stored (any float dtype; it is not
    modified); `batches` are normalised images (float64 preferred); `original` the unconverted model for its own
    floor (optional); `head` a classifier head reading the final class-token feature (DeiT-III), applied in each
    precision. Returns {"rows": [...], "forms": {...}, "notes": [...]}: one row per (form, precision, component)."""
    device = torch.device(device) if device is not None else next(model.parameters()).device
    torch.backends.cuda.matmul.allow_tf32 = False       # float32 means full float32 matmuls
    torch.backends.cudnn.allow_tf32 = False             # (the patch embedding is a convolution)
    unknown = [p for p in precisions if p not in PRECISIONS]
    if unknown:
        raise ValueError(f"unknown precisions {unknown}; choose from {PRECISIONS}")
    precisions = [p for p in PRECISIONS if p in set(precisions) | {"float64"}]   # references first
    notes = []
    batches = list(batches)
    rows = []
    say = log or (lambda m: None)
    for form in forms:
        if form == "even_odd" and int(st.k) != len(model.blocks):
            notes.append("even/odd form skipped: it needs the full conversion")
            continue
        store: dict = {}
        store_exact: dict | None = {} if "float64_exact" in precisions else None
        runs = []
        stored64 = next(model.parameters()).dtype == torch.float64
        base32 = copy.deepcopy(model).float().eval()
        base64 = copy.deepcopy(model).double().eval()          # an exact cast of the stored weights
        exact64 = reproject_exact(model, adapter, st, torch.float64) if "float64_exact" in precisions else None
        fv32 = fv64 = None
        if form == "even_odd":
            fv32 = even_odd if even_odd is not None else build_even_odd(base32, st, align=align)
            fv32 = fv32.float().eval()
            # the float64 reference has the even/odd weights of the stored model: built from the float64 weights
            # when the model is stored in float64, the float32 even/odd weights cast up otherwise
            fv64 = build_even_odd(base64, st, align=align).double() if stored64 else copy.deepcopy(fv32).double()
        used32 = False
        for p in precisions:                       # every run gets its own module (hooks are per module)
            src = base32 if form == "dense" else fv32
            if p == "float64":
                m = base64 if form == "dense" else fv64
            elif p == "float64_exact":
                m = exact64 if form == "dense" else build_even_odd(exact64, st, align=align).double()
            elif p == "bf16":
                m = copy.deepcopy(src).to(torch.bfloat16)
            elif not used32:
                m, used32 = src, True
            else:
                m = copy.deepcopy(src)
            m.eval()
            runs.append(_Run(form, p, m, adapter, st, _head_copy(head, p), device, is_ref=(p == "float64"),
                             store=store, store_exact=store_exact))
        say(f"A09 {form}: {', '.join(precisions)} in lockstep on {sum(int(x.shape[0]) for x in batches)} images")
        _lockstep(runs, batches)
        for r in runs:
            rows += _rows_of(r)
        del runs, base32, base64, exact64, fv32, fv64
    if original is not None:
        store = {}
        o32 = copy.deepcopy(original).float().eval()
        runs = [_Run("original", "float64", copy.deepcopy(original).double().eval(), adapter, None,
                     _head_copy(head, "float64"), device, True, store),
                _Run("original", "float32", o32, adapter, None, _head_copy(head, "float32"), device, False, store)]
        say("A09 original: float64 and float32 in lockstep (rounding floor)")
        _lockstep(runs, batches)
        for r in runs:
            rows += _rows_of(r)
    for r in rows:
        scored = r["form"] != "original" and r["precision"] == "float32" and r["in_span"] and r["space"] != "none"
        for flag, dcol in (("a09_2_pass", "dev_rel_tensor_max"), ("a09_2_pass_exact_floor", "dev_exact_rel_tensor_max")):
            dev, eq = r[dcol], r["eq_rel_tensor_max"]
            r[flag] = bool(math.isfinite(dev) and eq <= A09_2_FACTOR * dev) if scored else None
    return {"rows": rows, "precisions": precisions, "forms": list(forms), "notes": notes,
            "n_images": int(sum(int(x.shape[0]) for x in batches))}


def score_a09(rows: list, flip: dict | None = None, floor: str = "exact") -> dict:
    """A09-1, A09-2 and A09-3 from the audit rows and the flip checks
    (flip: {(form, precision): feature_flip_check result} or {"form/precision": ...}).

    A09-2 needs "the model's own float32-versus-float64 deviation". floor="exact" (default; PREREG_FINAL_PHASE.md
    Section 2.6) reads it as the distance from the float64 re-projected model (arithmetic plus the rounding of storing
    the projected weights in float32, which the float32 equivariance error also contains); floor="cast" as the same
    stored weights run in float64 (arithmetic rounding only), reported as a diagnostic. Both verdicts are reported;
    `floor` picks the one in "verdict"."""
    if floor not in ("cast", "exact"):
        raise ValueError("floor must be 'cast' or 'exact'")
    flip = {("/".join(k) if isinstance(k, tuple) else k): v for k, v in (flip or {}).items()}
    span = [r for r in rows if r["in_span"] and r["form"] != "original" and r["space"] != "none"]

    def worst(rs, col):
        rs = [r for r in rs if math.isfinite(r[col])]
        return max(rs, key=lambda r: r[col]) if rs else None

    def where(r):
        if r is None:
            return None
        return f"{r['form']} {r['stage']}{'' if r['block'] is None else ' ' + str(r['block'])} {r['component']}"

    out = {}
    e1 = [r for r in span if r["precision"] == "float64_exact"]
    w1 = worst(e1, "eq_rel_tensor_max")
    if not e1 or w1 is None:
        out["A09-1"] = {"verdict": "PENDING", "reason": "no float64_exact rows"}
    else:
        val = w1["eq_rel_tensor_max"]
        by_form = {}
        for f in sorted({r["form"] for r in e1}):
            w = worst([r for r in e1 if r["form"] == f], "eq_rel_tensor_max")
            by_form[f] = None if w is None else w["eq_rel_tensor_max"]
        out["A09-1"] = {"verdict": "PASS" if val <= A09_1_TOL else "FAIL", "max_rel_error": val, "where": where(w1),
                        "bar": A09_1_TOL, "n_tensors": len(e1), "max_rel_error_by_form": by_form}
    e2 = [r for r in span if r["precision"] == "float32"]
    if not e2:
        out["A09-2"] = {"verdict": "PENDING", "reason": "no float32 rows"}
    else:
        flag, rcol = ("a09_2_pass", "ratio_eq_to_dev") if floor == "cast" else ("a09_2_pass_exact_floor",
                                                                                "ratio_eq_to_dev_exact")
        fails = [r for r in e2 if not r.get(flag)]
        fails_other = [r for r in e2 if not r.get("a09_2_pass_exact_floor" if floor == "cast" else "a09_2_pass")]
        worst_ratio = worst(e2, rcol)
        fc = {}
        for form in sorted({r["form"] for r in e2}):
            f = flip.get(f"{form}/float32")
            fc[form] = None if f is None else f.get("pass_with_near_ties", f.get("pass"))
        tensors_ok = not fails
        flips_ok = all(v is True for v in fc.values()) if fc and all(v is not None for v in fc.values()) else None
        verdict = "FAIL" if (not tensors_ok or flips_ok is False) else ("PASS" if flips_ok else "PENDING")
        other = "exact" if floor == "cast" else "cast"
        out["A09-2"] = {"verdict": verdict, "floor": floor, "tensors_pass": tensors_ok, "n_tensors": len(e2),
                        "n_tensors_fail": len(fails), f"tensors_pass_{other}_floor": not fails_other,
                        f"n_tensors_fail_{other}_floor": len(fails_other),
                        "worst_ratio": None if worst_ratio is None else worst_ratio[rcol],
                        "worst_ratio_where": where(worst_ratio), "factor": A09_2_FACTOR,
                        "failing": [where(r) for r in fails[:20]], "flip_consistency_fp32_pass": fc,
                        "max_eq_error_fp32": worst(e2, "eq_rel_tensor_max")["eq_rel_tensor_max"],
                        "max_eq_error_fp32_where": where(worst(e2, "eq_rel_tensor_max")),
                        "reason": "" if verdict != "PENDING" else "flip consistency (k-NN) not measured for every form"}
    e3 = [r for r in span if r["precision"] in ("bf16_autocast", "bf16")]
    rep = {}
    for form in sorted({r["form"] for r in e3}):
        for p in sorted({r["precision"] for r in e3 if r["form"] == form}):
            w = worst([r for r in e3 if r["form"] == form and r["precision"] == p], "eq_rel_tensor_max")
            f = flip.get(f"{form}/{p}")
            rep[f"{form}/{p}"] = {"max_rel_error": w["eq_rel_tensor_max"] if w else None, "where": where(w),
                                  "flip_consistency": None if f is None else f.get("flip_consistency"),
                                  "n_disagree": None if f is None else f.get("n_disagree")}
    out["A09-3"] = {"verdict": "REPORT", "by_form": rep, "target": "100% flip consistency for the even/odd form"}
    return out


# ----------------------------------------------------------------------------------------------
# Weight-level conditions (E1 conditions on the stored parameters)
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def weight_conditions(adapter, model: nn.Module, st: AuditStructure) -> list:
    """Relative residual of each component's condition on the stored weights (cast exactly to float64): what E1
    needs. Zero up to the storage dtype's rounding for a correctly converted model."""
    A = torch.as_tensor(st.A, dtype=torch.float64)
    rows = []

    def rel(a, b):
        return float((a - b).norm() / b.norm().clamp_min(1e-300)) if b.numel() else 0.0

    def row(component, group, value, note=""):
        rows.append({"component": component, "plan_component": group, "rel_residual": float(value),
                     "condition": COMPONENTS[group]["condition"] if group in COMPONENTS else "", "note": note,
                     "dtype": str(next(model.parameters()).dtype).replace("torch.", "")})

    pe = model.patch_embed.proj
    W = pe.weight.detach().double().cpu()
    row("patch_embed.weight", "patch embedding", rel(W.flip(-1), torch.einsum("oi,ichw->ochw", A, W)))
    if pe.bias is not None:
        b = pe.bias.detach().double().cpu()
        row("patch_embed.bias", "patch embedding", rel(A @ b, b))
    Hg, Wg = adapter.grid_hw
    reg = register_info(model, adapter)
    R = int(reg["R"])
    has_cls = getattr(model, "cls_token", None) is not None
    pos = getattr(model, "pos_embed", None)
    if isinstance(pos, torch.Tensor):
        P = pos.detach().double().cpu()[0]
        n_pre = P.shape[0] - Hg * Wg
        grid = P[n_pre:].reshape(Hg, Wg, -1)
        row("pos_embed.grid", "position embeddings", rel(grid.flip(1) @ A, grid))
        if n_pre >= 1 and has_cls:
            row("pos_embed.cls", "class and register tokens", rel(P[0] @ A, P[0]))
        if n_pre > 1 and R:
            rp = torch.as_tensor(st.reg_perm if len(st.reg_perm) else np.arange(R))
            Rp = P[1:1 + R]
            row("pos_embed.registers", "class and register tokens", rel(Rp @ A, Rp[rp]))
    if has_cls:
        c = model.cls_token.detach().double().cpu().reshape(-1)
        row("cls_token", "class and register tokens", rel(c @ A, c))
    if R:
        r = getattr(model, reg["name"]).detach().double().cpu()[0]
        rp = torch.as_tensor(st.reg_perm if len(st.reg_perm) else np.arange(R))
        row("registers", "class and register tokens", rel(r @ A, r[rp]), f"pairing {rp.tolist()}")
    k = int(st.k)
    for b in range(len(model.blocks)):
        blk = model.blocks[b]
        folded = isinstance(blk.norm1, FoldRMSNorm) and isinstance(blk.norm2, FoldRMSNorm)
        row(f"block {b + 1} norms", "folded LayerNorm", 0.0 if folded else float("nan"),
            "RMSNorm without affine" if folded else "not folded")
        if b >= k:
            continue
        p = torch.as_tensor(st.perms[b])
        fc1, fc2 = blk.mlp.fc1, blk.mlp.fc2
        W1, b1 = fc1.weight.detach().double().cpu(), fc1.bias.detach().double().cpu()
        W2, b2 = fc2.weight.detach().double().cpu(), fc2.bias.detach().double().cpu()
        row(f"block {b + 1} fc1.weight", "MLP fc1, GELU, fc2 and biases", rel(W1[p], W1 @ A))
        row(f"block {b + 1} fc1.bias", "MLP fc1, GELU, fc2 and biases", rel(b1[p], b1))
        row(f"block {b + 1} fc2.weight", "MLP fc1, GELU, fc2 and biases", rel(A @ W2, W2[:, p]))
        row(f"block {b + 1} fc2.bias", "MLP fc1, GELU, fc2 and biases", rel(A @ b2, b2))
        hm = st.maps[b]
        try:
            hm.check(tol=1e-8)
            ok = 0.0
        except Exception:      # noqa: BLE001
            ok = float("nan")
        row(f"block {b + 1} head maps", "head maps", ok, f"{hm.n_pairs} twin pairs, {hm.n_self} self-heads")
        H, dh = attn_dims(blk)
        qkv = blk.attn.qkv
        Wqkv, bqkv = qkv.weight.detach().double().cpu(), qkv.bias.detach().double().cpu()
        D = H * dh
        d = Wqkv.shape[1]
        sig = np.asarray(hm.sigma)
        Bm, Cm = torch.as_tensor(np.asarray(hm.B, np.float64)), torch.as_tensor(np.asarray(hm.C, np.float64))
        for t, i0, M, grp in (("q", 0, Bm, "attention Q and K"), ("k", D, Bm, "attention Q and K"),
                              ("v", 2 * D, Cm, "attention V and O")):
            Wt = Wqkv[i0:i0 + D].reshape(H, dh, d)
            bt = bqkv[i0:i0 + D].reshape(H, dh)
            lhs = torch.stack([Wt[sig[h]] @ A for h in range(H)])
            rhs = torch.stack([M[h] @ Wt[h] for h in range(H)])
            row(f"block {b + 1} W{t}", grp, rel(lhs, rhs))
            row(f"block {b + 1} b{t}", grp, rel(torch.stack([bt[sig[h]] for h in range(H)]),
                                                 torch.stack([M[h] @ bt[h] for h in range(H)])))
        Wo = blk.attn.proj.weight.detach().double().cpu().reshape(d, H, dh)
        bo = blk.attn.proj.bias.detach().double().cpu()
        lhs = torch.stack([A @ Wo[:, h] for h in range(H)])
        rhs = torch.stack([Wo[:, sig[h]] @ Cm[h] for h in range(H)])
        row(f"block {b + 1} Wo", "attention V and O", rel(lhs, rhs))
        row(f"block {b + 1} bo", "attention V and O", rel(A @ bo, bo))
    even = getattr(getattr(model, "norm", None), "even", None)
    if even is not None:
        Pm = even.P.detach().double().cpu()
        row("even_head.P", "even projection", rel(Pm, 0.5 * (torch.eye(A.shape[0], dtype=torch.float64) + A)))
    return rows


# ----------------------------------------------------------------------------------------------
# Flip consistency (feature level here; the k-NN function comes from the evaluation code)
# ----------------------------------------------------------------------------------------------
def make_knn_fn(bank_feats: torch.Tensor, bank_labels: torch.Tensor, num_classes: int, k: int = 20,
                temperature: float = 0.07, chunk: int = 1024):
    """A k-NN function with the arithmetic of mirror.knn.knn_predict (cosine, top-k, votes exp(sim / T)) that also
    returns the relative top-two vote margin. Takes raw features (it L2-normalises them, as the protocol does).
    The evaluation agent's mirror/final/eval/knn_fast.py can be passed to feature_flip_check instead: any callable
    query_features -> predictions (or (predictions, margins))."""
    bank = F.normalize(bank_feats.float(), dim=-1)
    labels = bank_labels.to(bank.device)
    kk = min(int(k), bank.shape[0])

    @torch.no_grad()
    def fn(q: torch.Tensor):
        q = F.normalize(q.float().to(bank.device), dim=-1)
        preds, margins = [], []
        for s in range(0, q.shape[0], chunk):
            sim = q[s:s + chunk] @ bank.T
            top_sim, top_idx = sim.topk(kk, dim=1)
            w = (top_sim / temperature).exp()
            votes = torch.zeros(sim.shape[0], num_classes, device=sim.device, dtype=w.dtype)
            votes.scatter_add_(1, labels[top_idx], w)
            preds.append(votes.argmax(dim=1))
            two = votes.topk(min(2, num_classes), dim=1).values
            m = (two[:, 0] - two[:, 1]) / two[:, 0].clamp_min(1e-300) if two.shape[1] > 1 else torch.ones_like(two[:, 0])
            margins.append(m)
        return torch.cat(preds), torch.cat(margins)

    return fn


@torch.no_grad()
def compute_features(feature_fn, loader, normalise, precision: str, device, flip: bool = False,
                     max_images: int | None = None) -> tuple:
    """(features in the run's output dtype, labels) of every image of `loader` (uint8 images, labels)."""
    feats, labels, n = [], [], 0
    for imgs, y in loader:
        x = prepare_input(normalise(imgs.to(device)), precision)
        if flip:
            x = mirror(x)
        with precision_context(precision, device):
            f = feature_fn(x)
        feats.append(f.detach().clone())
        labels.append(torch.as_tensor(y))
        n += int(x.shape[0])
        if max_images is not None and n >= max_images:
            break
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def feature_flip_check(feature_fn, loader, normalise, *, precision: str = "float32", device="cpu", knn_fn=None,
                       classifier: bool = False, max_images: int | None = None) -> dict:
    """Feature-level and (with knn_fn) prediction-level flip check of a feature function (adapter.cls_feature of a
    model, a FastViT, or logits for a classifier) on the images of `loader`. Feature level: largest absolute and
    relative difference between f(x) and f(mirror x), and the share of images whose features are identical bit for
    bit. With knn_fn (query features -> predictions or (predictions, relative top-two margins)): flip consistency
    and the near-tie rule of PREREG_FINAL_PHASE.md Section 2.1 (every disagreeing image has a margin below 1e-4).
    For a classifier pass feature_fn = logits and knn_fn = None: predictions are the argmax of the logits."""
    f, y = compute_features(feature_fn, loader, normalise, precision, device, flip=False, max_images=max_images)
    ft, _ = compute_features(feature_fn, loader, normalise, precision, device, flip=True, max_images=max_images)
    d = (ft.double() - f.double())
    same_bits = (ft == f).all(dim=1)
    out = {"precision": precision, "n_images": int(f.shape[0]), "max_abs_feature_diff": float(d.abs().max()),
           "max_abs_feature_ref": float(f.double().abs().max()),
           "max_rel_feature_diff_per_image": float((d.norm(dim=1) / f.double().norm(dim=1).clamp_min(1e-300)).max()),
           "bitwise_identical_share": float(same_bits.double().mean()),
           "n_bitwise_identical": int(same_bits.sum())}
    if knn_fn is None and classifier:
        knn_fn = _argmax_fn
    if knn_fn is not None:
        r, rt = knn_fn(f), knn_fn(ft)
        pred, marg = (r if isinstance(r, tuple) else (r, None))
        pred_t, marg_t = (rt if isinstance(rt, tuple) else (rt, None))
        pred, pred_t = pred.cpu(), pred_t.cpu()
        same = pred == pred_t
        dis = (~same).nonzero().flatten()
        near = None
        if marg is not None and marg_t is not None:
            # PREREG_FINAL_PHASE.md Section 2.1: a disagreement is a near-tie only if its relative top-two margin is
            # below 1e-4 in BOTH copies (the image and its mirror), so the larger of the two margins decides; the
            # smaller one is kept as a diagnostic (the CORE-06 one-copy convention)
            m_strict = torch.maximum(marg.cpu(), marg_t.cpu())[dis]
            m_one = torch.minimum(marg.cpu(), marg_t.cpu())[dis]
            near = bool((m_strict < NEAR_TIE).all()) if len(dis) else True
            out["disagree_margins"] = [float(v) for v in m_strict[:50]]
            out["disagree_margins_one_copy"] = [float(v) for v in m_one[:50]]
            out["max_margin_rel"] = float(m_strict.max()) if len(dis) else None
            out["n_disagree_not_near_tie"] = int((m_strict >= NEAR_TIE).sum())
        fc = 100.0 * float(same.double().mean())
        out.update({"flip_consistency": fc, "n_disagree": int(len(dis)), "pass": bool(len(dis) == 0),
                    "pass_with_near_ties": bool(len(dis) == 0 or bool(near)),
                    "top1": 100.0 * float((pred == y.cpu()).double().mean()),
                    "pred": pred.numpy(), "pred_flip": pred_t.numpy(), "labels": y.cpu().numpy()})
    return out


def _argmax_fn(logits: torch.Tensor):
    two = logits.double().topk(min(2, logits.shape[1]), dim=1).values
    m = (two[:, 0] - two[:, 1]).abs() / two[:, 0].abs().clamp_min(1e-300)
    return logits.argmax(dim=1), m


# ----------------------------------------------------------------------------------------------
# Bitwise check of the even/odd form (E3)
# ----------------------------------------------------------------------------------------------
def _ulp(t: torch.Tensor) -> torch.Tensor:
    t = t.abs()
    if t.dtype == torch.bfloat16:
        eps = 2.0 ** -7
    elif t.dtype == torch.float16:
        eps = 2.0 ** -10
    elif t.dtype == torch.float32:
        eps = 2.0 ** -23
    else:
        eps = 2.0 ** -52
    tiny = torch.finfo(t.dtype).tiny
    return (torch.exp2(torch.floor(torch.log2(t.double().clamp_min(tiny)))) * eps).clamp_min(tiny * eps)


def _cmp(a: torch.Tensor, b: torch.Tensor) -> dict:
    """a, b already aligned (same layout and dtype): exact equality and the distance in units of the last place."""
    eq = (a == b)
    diff = (a.double() - b.double()).abs()
    ulps = diff / _ulp(b)
    return {"n_entries": int(eq.numel()), "share_identical": float(eq.double().mean()) if eq.numel() else 1.0,
            "all_identical": bool(eq.all()), "max_abs_diff": float(diff.max()) if diff.numel() else 0.0,
            "max_ulp": float(ulps.max()) if ulps.numel() else 0.0}


@torch.no_grad()
def snap_embedding_exact(fv: nn.Module, dp: int, grid_hw, reg_perm=None) -> nn.Module:
    """A copy of an even/odd model whose embedding parameters are exactly symmetric in its own dtype: even rows of
    the patch kernels symmetric in the width, odd rows antisymmetric, odd coordinates of the patch bias and of the
    class token zero, the position grid paired column by column (pos(r, G-1-c) = s * pos(r, c) exactly), registers
    paired or even. Averages are fl((a + b)/2), which is the same for both members of a pair, so the symmetry is
    exact. A diagnostic only (bitwise_check): it isolates the summation-order sources of difference."""
    m = copy.deepcopy(fv)
    base = m.embed.base
    W = base.patch_embed.proj.weight
    d = W.shape[0]
    s = sign_vector(d, dp, dtype=W.dtype, device=W.device)
    W.copy_((W + W.flip(-1) * s[:, None, None, None]) / 2)
    if base.patch_embed.proj.bias is not None:
        base.patch_embed.proj.bias[dp:] = 0
    H, Wg = (int(v) for v in grid_hw)
    pos = getattr(base, "pos_embed", None)
    has_cls = getattr(base, "cls_token", None) is not None
    reg = None
    for name in ("register_tokens", "reg_token"):
        if isinstance(getattr(base, name, None), torch.Tensor):
            reg = name
    R = int(getattr(base, reg).shape[1]) if reg else 0
    rp = torch.as_tensor(np.arange(R) if reg_perm is None or not len(reg_perm) else np.asarray(reg_perm),
                         dtype=torch.long)
    if isinstance(pos, torch.Tensor):
        P = pos.data[0]
        n_pre = P.shape[0] - H * Wg
        g = P[n_pre:].reshape(H, Wg, d)
        g.copy_((g + g.flip(1) * s) / 2)
        if n_pre >= 1 and has_cls:
            P[0, dp:] = 0
        if n_pre > 1 and R:
            Rp = P[1:1 + R]
            Rp.copy_((Rp + Rp[rp] * s) / 2)
    if has_cls:
        base.cls_token.data[..., dp:] = 0
    if R:
        r = getattr(base, reg).data[0]
        r.copy_((r + r[rp] * s) / 2)
    return m


@torch.no_grad()
def bitwise_check(fv: nn.Module, x: torch.Tensor, *, dp: int, grid_hw, n_prefix: int, reg_perm=None,
                  reg_start: int = 1, head: nn.Module | None = None, controlled: bool = True,
                  snapped: bool = True) -> dict:
    """E3 for the even/odd form in its own dtype: are the features (and logits, predictions) of x and mirror(x)
    identical bit for bit, and where not, why. Stages compared after the exact alignment (a gather) and the exact
    sign flip of the odd coordinates: patch embedding, layer-0 tokens (class and register tokens, patch tokens),
    every block's attention branch, residual sum, MLP branch and output, the final feature and the logits.

    Diagnostics that say why: (a) parameter level: are the embedding parameters symmetric bit for bit in the stored
    basis; (b) controlled blocks: each block run on the original's input and its exactly mirrored copy, so every
    difference starts inside that block: norm, q/k/v, the attention with keys in their natural (mirrored) order and
    with keys reordered to the original's order (the same function, the summation order over keys removed), MLP;
    (c) snapped: the whole comparison again with the embedding parameters made exactly symmetric
    (snap_embedding_exact), which leaves only summation-order sources."""
    dev = next(fv.parameters()).device
    dt = next(fv.parameters()).dtype
    x = x.to(device=dev, dtype=dt)
    idx = token_index(grid_hw, n_prefix, reg_perm, reg_start).to(dev)
    idx_p = token_index(grid_hw, 0).to(dev)
    d = int(fv.blocks[0].norm1.dim)
    s = sign_vector(d, dp, dtype=dt, device=dev)
    B = int(x.shape[0])

    def stream_pair(t, ix):
        return t[:B] * s, t[B:][:, ix]

    def run(model) -> tuple:
        cap = OrderedDict()
        hs = [model.embed.base.patch_embed.register_forward_hook(
            lambda _m, _a, o: cap.__setitem__("patch_embed", o.reshape(o.shape[0], -1, o.shape[-1])))]
        for b, blk in enumerate(model.blocks):
            hs.append(blk.attn.register_forward_hook(lambda _m, _a, o, b=b: cap.__setitem__(f"block {b + 1} attn", o)))
            hs.append(blk.norm2.register_forward_pre_hook(
                lambda _m, a, b=b: cap.__setitem__(f"block {b + 1} resid_attn", a[0])))
            hs.append(blk.mlp.register_forward_hook(lambda _m, _a, o, b=b: cap.__setitem__(f"block {b + 1} mlp", o)))
            hs.append(blk.register_forward_hook(lambda _m, _a, o, b=b: cap.__setitem__(f"block {b + 1} out", o)))
        try:
            xx = torch.cat([x, mirror(x)])
            t0 = model.embed(xx)
            cap["tokens0_prefix"] = t0[:, :n_prefix]
            cap["tokens0_patches"] = t0[:, n_prefix:]
            t = t0
            for blk in model.blocks:
                t = blk(t)
            feat = model.head(t)
        finally:
            for h in hs:
                h.remove()
        stages = OrderedDict()
        for key, t in cap.items():
            if key == "patch_embed" or key == "tokens0_patches":
                a, b_ = stream_pair(t, idx_p)
            elif key == "tokens0_prefix":
                a, b_ = stream_pair(t, idx[:n_prefix])
            else:
                a, b_ = stream_pair(t, idx)
            stages[key] = _cmp(a, b_)
        stages["cls_feature"] = _cmp(feat[:B], feat[B:])
        feats_same = (feat[:B] == feat[B:]).all(dim=1)
        res = {"stages": stages, "features_identical_share": float(feats_same.double().mean()),
               "n_images": B}
        if head is not None:
            lg = head.to(dt)(feat)
            res["stages"]["logits"] = _cmp(lg[:B], lg[B:])
            pr = lg.argmax(dim=1)
            res["predictions_identical_share"] = float((pr[:B] == pr[B:]).double().mean())
        first = next((k for k, v in stages.items() if not v["all_identical"]), None)
        res["first_difference"] = first
        return res, t0

    main, t0 = run(fv)
    out = {"dtype": str(dt).replace("torch.", ""), "d_plus": int(dp), **main}
    # (a) parameter level
    base = fv.embed.base
    W = base.patch_embed.proj.weight
    params = {"patch_embed.weight": _cmp(W[:, :, :, :] * s[:, None, None, None], W.flip(-1))}
    if base.patch_embed.proj.bias is not None:
        params["patch_embed.bias(odd part zero)"] = _cmp(base.patch_embed.proj.bias[dp:],
                                                         torch.zeros_like(base.patch_embed.proj.bias[dp:]))
    pos = getattr(base, "pos_embed", None)
    H, Wg = (int(v) for v in grid_hw)
    if isinstance(pos, torch.Tensor):
        P = pos[0]
        n_pre = P.shape[0] - H * Wg
        g = P[n_pre:].reshape(H, Wg, d)
        params["pos_embed.grid"] = _cmp(g * s, g.flip(1))
    if getattr(base, "cls_token", None) is not None:
        c = base.cls_token.reshape(-1)
        params["cls_token(odd part zero)"] = _cmp(c[dp:], torch.zeros_like(c[dp:]))
    out["parameters"] = params
    # (b) controlled blocks
    if controlled:
        ctl = []
        t = t0
        for b, blk in enumerate(fv.blocks):
            xin = t[:B]
            xm = (xin * s)[:, idx]
            both = torch.cat([xin, xm])
            r = {"block": b + 1}
            n1 = blk.norm1(both)
            r["norm1"] = _cmp(*stream_pair(n1, idx))["share_identical"]
            pn = mirror_attention_parts(blk.attn, n1)
            perm_h, S = _eo_head_action(blk.attn)
            qa = pn["q"].transpose(1, 2)
            r["q"] = _cmp(qa[:B][:, :, perm_h] * S["q"].to(dt), qa[B:][:, idx])["share_identical"]
            a_nat = _cmp(*stream_pair(pn["out"], idx))
            pm = mirror_attention_parts(blk.attn, n1, key_perm=idx, key_perm_from=B)    # mirrored half only
            a_mat = _cmp(*stream_pair(pm["out"], idx))
            r.update({"attn_natural_key_order": a_nat["share_identical"], "attn_natural_max_ulp": a_nat["max_ulp"],
                      "attn_matched_key_order": a_mat["share_identical"], "attn_matched_max_ulp": a_mat["max_ulp"]})
            # sign flip alone (same token positions, separate calls of the same shape)
            a0, a1 = blk.attn(n1[:B]), blk.attn(n1[:B] * s)
            r["attn_sign_only"] = _cmp(a0 * s, a1)["share_identical"]
            ain = pn["out"][:B]
            rin = xin + ain
            rboth = torch.cat([rin, (rin * s)[:, idx]])
            mo = blk.mlp(blk.norm2(rboth))
            r["mlp_given_exact_input"] = _cmp(*stream_pair(mo, idx))["share_identical"]
            m0 = blk.mlp(blk.norm2(rin))
            r["mlp_sign_only"] = _cmp(m0 * s, blk.mlp(blk.norm2(rin * s)))["share_identical"]
            r["mlp_position_only"] = _cmp(m0[:, idx], blk.mlp(blk.norm2(rin[:, idx])))["share_identical"]
            ctl.append(r)
            t = blk(t)
        out["controlled_blocks"] = ctl
    # (c) snapped embedding
    if snapped:
        sn, _ = run(snap_embedding_exact(fv, dp, grid_hw, reg_perm))
        out["snapped"] = {"stages": sn["stages"], "first_difference": sn["first_difference"],
                          "features_identical_share": sn["features_identical_share"],
                          **({"predictions_identical_share": sn["predictions_identical_share"]}
                             if "predictions_identical_share" in sn else {})}
    out["why"] = _why(out)
    # Bitwise equality of grid-flipped token rows also depends on how the matrix products split their work: on the CPU
    # with 3 or more threads some rows take another kernel path (tests/test_fp_audit.py), so the result is recorded
    # with the conditions it holds under.
    out["conditions"] = {"device": str(x.device), "dtype": str(next(fv.parameters()).dtype),
                         "torch_threads": int(torch.get_num_threads()),
                         "cuda_device": torch.cuda.get_device_name(x.device) if x.device.type == "cuda" else None}
    return out


def _why(res: dict) -> list:
    """Plain-language reasons from the measured shares (no reason is asserted without its measurement)."""
    why = []
    st = res["stages"]
    if res.get("first_difference") is None:
        why.append("every compared stage is identical bit for bit")
        return why
    why.append(f"first stage that differs: {res['first_difference']}")
    asym = {k: v for k, v in res.get("parameters", {}).items() if not v["all_identical"]}
    if asym:
        why.append("embedding parameters are symmetric only to rounding in the stored even/odd basis ("
                   + ", ".join(f"{k}: {100 * v['share_identical']:.2f}% of entries identical" for k, v in asym.items())
                   + "): the projection is rounded in the standard basis before the rotation into the even/odd basis")
    ctl = res.get("controlled_blocks") or []
    if ctl:
        def lo(key):
            return min(r[key] for r in ctl)

        why.append(f"inside a block given an exactly mirrored input (lowest share over blocks): norm "
                   f"{100 * lo('norm1'):.2f}%, queries {100 * lo('q'):.2f}%, MLP {100 * lo('mlp_given_exact_input'):.2f}% "
                   f"of entries identical; attention output {100 * lo('attn_natural_key_order'):.2f}% with keys in the "
                   f"mirrored order and {100 * lo('attn_matched_key_order'):.2f}% with keys reordered to the original "
                   "order")
        if lo("mlp_sign_only") == 1.0 and lo("attn_sign_only") == 1.0:
            why.append("the sign flip alone (same token positions) is exact in the MLP and the attention: 100% identical")
        else:
            why.append(f"the sign flip alone (same token positions) leaves MLP {100 * lo('mlp_sign_only'):.2f}% and "
                       f"attention {100 * lo('attn_sign_only'):.2f}% identical")
        if lo("mlp_position_only") < 1.0:
            why.append(f"moving the same tokens to other positions changes the MLP result ({100 * lo('mlp_position_only'):.2f}% "
                       "identical): the kernels' arithmetic depends on the position (GEMM tiling or vectorised and "
                       "scalar code paths), so mirrored tokens, which sit at other positions, round differently")
        if lo("attn_matched_key_order") > lo("attn_natural_key_order"):
            why.append("reordering the keys to the original order raises the attention share: the order in which "
                       "attention sums over keys (softmax normaliser and value sum) is a source of difference")
    sn = res.get("snapped")
    if sn is not None:
        pe = sn["stages"].get("patch_embed")
        why.append(f"with exactly symmetric embedding parameters the first difference is {sn['first_difference']}"
                   + (f" (patch embedding {100 * pe['share_identical']:.2f}% identical: the convolution sums the "
                      "flipped kernel window in another order)" if pe is not None and not pe["all_identical"] else ""))
    if "predictions_identical_share" in res:
        why.append(f"predictions identical for {100 * res['predictions_identical_share']:.2f}% of images; features "
                   f"identical for {100 * res['features_identical_share']:.2f}%")
    else:
        why.append(f"features identical bit for bit for {100 * res['features_identical_share']:.2f}% of images")
    return why
