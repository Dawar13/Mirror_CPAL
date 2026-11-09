"""V-exact healing: an exactly mirror-equivariant student over the converted span, healed by
distillation from the original model (PAPER_EXPERIMENTS.md, Stage 3 "V-exact full conversion,
healed", and the V-exact arm of the M3 grid). Used by scripts/heal_vexact.py.

Student (build_vexact_student). mirror.m1.build_vexact with one operator A* for blocks 0..k-1
(stream layers 0..k): the model fully LayerNorm-folded (every norm an RMSNorm on a centred stream),
the embeddings projected, MLP and attention projected with A_in = A_out = A*, and for full
conversion (k = depth) the even head y -> y (I + A*)/2 between the final RMSNorm and the final
affine. While training:
  MLP, attention  re-projected after every optimiser step with A_in = A_out = A*
                  (heal.MirrorMLPConstraint; heal_attn.MirrorAttnConstraint with the head maps that
                  hold after surgery). The absorbed norm and LayerScale parameters stay frozen.
  embeddings      patch embedding, position grid (with its CLS row), CLS token and registers. With
                  train_embeddings=True (default) they train and are re-projected after every step
                  with m1.project_embeddings (VExactEmbeddings); the trainer then runs with
                  train_scope "all" and VExactEmbeddings freezes everything else. With False they
                  stay frozen at their projection (train_scope "converted").
  even head       a fixed buffer, never trained. The final affine after it stays frozen (it is
                  outside the converted span and acts per channel, so it cannot break invariance).
  everything else frozen: unconverted blocks (k < depth), the DINOv2 mask token, classifier heads.
So the student stays exactly equivariant over the span during training: for the mirrored image the
tokens at mirrored positions equal A* times the original tokens at every layer 0..k, and at
k = depth the final features of an image and of its mirror are equal (CLS) or grid-flipped
(patches). Exactness is limited only by the float32 rounding of the projected weights.

Teacher and targets (VExactTeacher). The teacher is the original, unfolded model. Its outputs are
mapped into the student's space before the loss (mirror/heal.py calls `distill_targets`):
  block terms  the student's stream is the folded stream, which equals the original stream centred
               over channels (mirror/fold.py), so the teacher's block output is centred first. It
               is then symmetrised with the student's exact symmetry,
                   T(x) = (c(x) + M[c(mirror x)]) / 2,    M[y] = align(y) A*,
               where align un-flips the patch grid and swaps paired registers (M is an orthogonal
               involution). T(mirror x) = M[T(x)], the symmetry the student has, and T is the
               orthogonal projection of the teacher onto functions with that symmetry (for a
               flip-closed image distribution). The loss of an image and of its mirror are then
               equal, and the student can match T exactly in principle.
  final terms  k < depth: the teacher's final-norm CLS and patch tokens, unchanged (blocks k and
               later are not converted, so the student's output has no exact symmetry; as in the
               pilot).
               k = depth, final_target "symmetrised" (default): the mirror average of the teacher's
               final features, (f(x) + align(f(mirror x))) / 2. No A* here: the even head makes the
               student's final features of x and mirror(x) equal up to align, so this is the
               projection of the teacher onto invariant features. Random-flip training with the raw
               teacher converges to the same target in expectation; making it explicit removes the
               teacher's mirror-odd part, which the student cannot represent, from the loss (it only
               adds gradient noise and an irreducible floor). One restriction remains: the student's
               features before the final affine lie in the even subspace of A*, and the mirror
               average need not (A* is not the teacher's last-layer operator). That part is left to
               the least-squares fit in the k-NN feature space, so the student learns the best even
               approximation of the teacher's invariant features, weighted by the final affine.
               final_target "even": the even projection, before the final affine, of the mirror-
               averaged normalised teacher tokens, (n_avg (I + A*)/2) gamma + beta (n: the teacher's
               final LayerNorm without its affine). It lies in the student's reachable set and
               differs from "symmetrised" only through the gamma weighting of the least-squares fit.
  symmetrise=False turns the mirror averaging off (block targets only centred, final targets raw),
  for checks and ablations. With it on, the teacher runs on the batch and its mirror (twice the
  teacher compute of the pilot).
  block_targets (default: "symmetrised" with symmetrise, else "centred") sets the block terms on
  their own: "raw" gives the teacher's own block outputs (not centred, not symmetrised), as the M3
  pilot uses them. E3's SVD arm with V-exact targets (configs/e3_svd_sym.yaml) uses "raw" block
  targets with the flip-averaged final target: centring belongs to the folded stream, and the A*
  symmetrisation is loss-neutral only for a student with that symmetry (PREREG_E3.md).

E3 (PREREG_E3.md) also uses: vexact_block_params (the per-block accounting that vexact_param_counts
sums), matched_svd_ranks (SVD ranks at those per-block counts) and SpanTrainable (V-exact's trainable
set for a student without the projections).
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import attention_free_params, attn_dims
from .data import mirror
from .heal import MirrorMLPConstraint, exact_involution
from .heal_attn import MirrorAttnConstraint
from .knn import extract_features, knn_predict
from .m1 import (_check_perm, align_mirror_tokens, attention_svd_ranks, build_vexact, feature_invariance,
                 involution_trace64, mlp_mirror_params, project_embeddings, register_info, stream_equivariance)
from .surgery import equivariant_param_count, perm_trace, svd_rank_for

FINAL_TARGETS = ("symmetrised", "even")
BLOCK_TARGETS = ("symmetrised", "centred", "raw")

__all__ = ["FINAL_TARGETS", "BLOCK_TARGETS", "VExactTeacher", "embedding_parameters", "span_frozen_parameters",
           "VExactEmbeddings", "SpanTrainable", "train_scope_for", "build_vexact_student", "vexact_block_params",
           "vexact_param_counts", "matched_svd_ranks", "span_exactness", "flip_check"]


def _float_dtype(t: torch.Tensor) -> torch.dtype:
    return torch.float64 if t.dtype == torch.float64 else torch.float32


def _no_autocast(t: torch.Tensor):
    return torch.autocast(device_type=t.device.type, enabled=False)


def _d_plus(A) -> int:
    A = np.asarray(A)
    return int((involution_trace64(A) + A.shape[0]) // 2)


# ----------------------------------------------------------------------------------------------
# Teacher
# ----------------------------------------------------------------------------------------------
class VExactTeacher:
    """The original model's adapter with V-exact distillation targets (see the module docstring).

    Everything except `distill_targets` is delegated to `base` (model, n_prefix_tokens,
    layer0_tokens, grid_hw, ...), so the trainer runs the student exactly as with the plain adapter.
    """

    def __init__(self, base, A, k: int, reg_perm=None, symmetrise: bool = True, final_target: str = "symmetrised",
                 block_targets: str | None = None):
        if final_target not in FINAL_TARGETS:
            raise ValueError(f"final_target must be one of {FINAL_TARGETS}, not {final_target!r}")
        if block_targets is None:
            block_targets = "symmetrised" if symmetrise else "centred"
        if block_targets not in BLOCK_TARGETS:
            raise ValueError(f"block_targets must be one of {BLOCK_TARGETS}, not {block_targets!r}")
        if block_targets == "symmetrised" and not symmetrise:
            raise ValueError('block_targets "symmetrised" needs symmetrise=True (the teacher must also see the mirror)')
        L = int(base.depth)
        if not 1 <= int(k) <= L:
            raise ValueError(f"k must be in [1, {L}]")
        self.base = base
        self.A = exact_involution(A)
        self.k = int(k)
        self.even_head = self.k == L
        reg = register_info(base.model, base)
        self.reg_start = int(reg["start"])
        self.reg_perm = _check_perm(reg_perm, int(reg["R"]))
        self.symmetrise = bool(symmetrise)
        self.final_target = final_target
        self.block_targets = block_targets
        if self.even_head and final_target == "even" and not isinstance(base.model.norm, nn.LayerNorm):
            raise ValueError('final_target "even" needs the teacher\'s final norm to be nn.LayerNorm')
        self._mats: dict = {}

    def __getattr__(self, name):
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def describe(self) -> dict:
        if self.even_head:
            final = self.final_target if self.symmetrise else ("raw" if self.final_target == "symmetrised" else "even, not averaged")
        else:
            final = "raw (no even head)"
        if self.block_targets == "raw":
            blocks = "raw (the teacher's own block outputs, not centred or symmetrised)"
        else:
            blocks = "centred" + (", symmetrised with A*" if self.block_targets == "symmetrised" else "")
        return {"type": "VExactTeacher", "k": self.k, "even_head": self.even_head, "symmetrise": self.symmetrise,
                "block_targets": blocks, "final_target": final, "d_plus": _d_plus(self.A),
                "reg_perm": [int(v) for v in self.reg_perm]}

    def _mat(self, name: str, device, dtype) -> torch.Tensor:
        key = (name, str(device), dtype)
        if key not in self._mats:
            M = self.A if name == "A" else 0.5 * (np.eye(self.A.shape[0]) + self.A)
            self._mats[key] = torch.as_tensor(M, dtype=dtype, device=device)
        return self._mats[key]

    def align(self, t: torch.Tensor) -> torch.Tensor:
        """Mirrored image's tokens at the original's positions (grid un-flipped, registers paired)."""
        return align_mirror_tokens(t, self.base.grid_hw, self.base.n_prefix_tokens,
                                   self.reg_perm if len(self.reg_perm) else None, self.reg_start)

    def stream_target(self, t: torch.Tensor, B: int, symmetrise: bool | None = None) -> torch.Tensor:
        """Centred teacher stream for the first B images; symmetrised with M if `symmetrise` (the
        batch then holds the B images followed by their mirrors)."""
        sym = self.symmetrise if symmetrise is None else bool(symmetrise)
        dt = _float_dtype(t)
        with _no_autocast(t):
            c = t.to(dt)
            c = c - c.mean(dim=-1, keepdim=True)
            if not sym:
                return c[:B]
            return 0.5 * (c[:B] + self.align(c[B:]) @ self._mat("A", c.device, dt))

    def final_targets(self, t: torch.Tensor, final: torch.Tensor, B: int) -> torch.Tensor:
        """Final-norm token targets [B, T, d] from the last stream `t` and the teacher's final norm
        output `final` (both for the batch and, when symmetrising, its mirror)."""
        dt = _float_dtype(final)
        with _no_autocast(final):
            if not self.even_head:
                return final[:B].to(dt)
            if self.final_target == "symmetrised":
                f = final.to(dt)
                return 0.5 * (f[:B] + self.align(f[B:])) if self.symmetrise else f[:B]
            norm = self.base.model.norm
            n = F.layer_norm(t.to(dt), norm.normalized_shape, None, None, norm.eps)
            n = 0.5 * (n[:B] + self.align(n[B:])) if self.symmetrise else n[:B]
            n = n @ self._mat("P", n.device, dt)
            if norm.weight is not None:
                n = n * norm.weight.to(dt)
            if norm.bias is not None:
                n = n + norm.bias.to(dt)
            return n

    @torch.no_grad()
    def distill_targets(self, x: torch.Tensor, plan):
        """Targets for mirror/heal.py: {"cls", "patch", "block<i>" for i in plan.capture} and start=None
        (the student always computes its own embeddings; they differ from the teacher's)."""
        m = self.base.model
        B = x.shape[0]
        t = self.base.layer0_tokens(torch.cat([x, mirror(x)], dim=0) if self.symmetrise else x, m)
        feats = {}
        for i, blk in enumerate(m.blocks):
            t = blk(t)
            if i in plan.capture:
                if self.block_targets == "raw":
                    with _no_autocast(t):
                        feats[f"block{i}"] = t[:B].to(_float_dtype(t))
                else:
                    feats[f"block{i}"] = self.stream_target(
                        t, B, symmetrise=self.block_targets == "symmetrised" and i < self.k)
        f = self.final_targets(t, m.norm(t), B)
        feats["cls"] = f[:, 0]
        feats["patch"] = f[:, plan.n_prefix:]
        return feats, None


# ----------------------------------------------------------------------------------------------
# Embedding side and scope
# ----------------------------------------------------------------------------------------------
def embedding_parameters(model: nn.Module) -> list:
    """(name, parameter) of the V-exact embedding side, the parameters m1.project_embeddings writes:
    patch embedding, position embedding (CLS row and patch grid), CLS token and registers."""
    proj = model.patch_embed.proj
    out = [("patch_embed.proj.weight", proj.weight)]
    if proj.bias is not None:
        out.append(("patch_embed.proj.bias", proj.bias))
    for name in ("pos_embed", "cls_token", "register_tokens", "reg_token"):
        p = getattr(model, name, None)
        if isinstance(p, nn.Parameter):
            out.append((name, p))
    return out


def span_frozen_parameters(model: nn.Module, blocks: Sequence[int], embeddings: bool) -> list:
    """Every parameter that is neither in `blocks` nor (with `embeddings`) an embedding parameter: what a
    V-exact student keeps frozen (unconverted blocks, final norm affine, mask token, heads)."""
    keep = {id(p) for b in blocks for p in model.blocks[int(b)].parameters()}
    if embeddings:
        keep |= {id(p) for _, p in embedding_parameters(model)}
    return [p for p in model.parameters() if id(p) not in keep]


def train_scope_for(train_embeddings: bool) -> str:
    """HealConfig.train_scope to use with VExactEmbeddings(train=train_embeddings)."""
    return "all" if train_embeddings else "converted"


class SpanTrainable:
    """Constraint (mirror/heal.py protocol) that only freezes: with train_scope "all", everything outside
    the converted blocks and the embedding side (embedding_parameters) stays frozen, so a student
    without the V-exact projections trains the same set of tensors as a V-exact student with
    train_embeddings (E3's SVD arm, PREREG_E3.md). Nothing is projected; the embeddings train freely."""

    def __init__(self, blocks: Sequence[int], embeddings: bool = True):
        self.blocks = [int(b) for b in blocks]
        self.embeddings = bool(embeddings)

    def frozen_parameters(self, model: nn.Module) -> list:
        return span_frozen_parameters(model, self.blocks, self.embeddings)

    def __call__(self, model: nn.Module) -> None:
        return None

    def describe(self) -> dict:
        return {"type": "SpanTrainable", "blocks": self.blocks, "embeddings_train": self.embeddings,
                "frozen": "everything outside the converted blocks" + (" and the embeddings" if self.embeddings else "")}


class VExactEmbeddings:
    """Constraint (mirror/heal.py protocol) for the embedding side of a V-exact student whose span
    starts at block 0, and for what trains outside the span.

    train=True: the embeddings train (run the trainer with train_scope "all") and are re-projected
    after every optimiser step with m1.project_embeddings, in float64, rounded once to the parameter
    dtype. `frozen_parameters` freezes everything that is neither in the converted blocks nor an
    embedding parameter (unconverted blocks, final norm affine, mask token, heads).
    train=False: the embeddings stay frozen at their projection (train_scope "converted"); nothing is
    re-projected. `check` reports the embedding breaking shares (0 when exact) and the even head.
    """

    def __init__(self, adapter, A, blocks: Sequence[int], reg_perm=None, train: bool = True):
        blocks = [int(b) for b in blocks]
        if not blocks or blocks != list(range(len(blocks))):
            raise ValueError("V-exact healing converts blocks 0..k-1 (the embeddings belong to the span)")
        self.adapter = adapter
        self.blocks = blocks
        self.A = exact_involution(A)
        self.reg_perm = None if reg_perm is None else np.asarray(reg_perm, dtype=np.int64)
        self.train = bool(train)

    def _shares(self, model: nn.Module) -> dict:
        return project_embeddings(model, self.adapter, self.A, self.reg_perm, write=False)

    def prepare(self, model: nn.Module) -> None:
        shares = self._shares(model)
        worst = max((v for v in shares.values() if math.isfinite(v)), default=0.0)
        if worst > 1e-8:
            raise ValueError(f"embeddings are not projected for this operator (largest share {worst:.2e}); "
                             "build the student with mirror.m1.build_vexact")

    def frozen_parameters(self, model: nn.Module) -> list:
        return span_frozen_parameters(model, self.blocks, self.train)

    @torch.no_grad()
    def __call__(self, model: nn.Module) -> None:
        if self.train:
            project_embeddings(model, self.adapter, self.A, self.reg_perm, write=True)

    @torch.no_grad()
    def check(self, model: nn.Module) -> dict:
        shares = self._shares(model)
        out = {"max_share": max((v for v in shares.values() if math.isfinite(v)), default=0.0),
               "shares": shares, "embeddings_train": self.train}
        even = getattr(getattr(model, "norm", None), "even", None)
        if even is not None:
            P = 0.5 * (np.eye(self.A.shape[0]) + self.A)
            out["even_head_max_abs_err"] = float(np.abs(even.P.double().cpu().numpy() - P).max())
        return out

    def describe(self) -> dict:
        return {"type": "VExactEmbeddings", "blocks": self.blocks, "d_plus": _d_plus(self.A),
                "embeddings": "trained, re-projected every step" if self.train else "frozen after projection",
                "reg_perm": None if self.reg_perm is None else [int(v) for v in self.reg_perm]}


# ----------------------------------------------------------------------------------------------
# Student
# ----------------------------------------------------------------------------------------------
def build_vexact_student(adapter, A, k: int, perms: Sequence, head_maps: Sequence, reg_perm=None,
                         train_embeddings: bool = True, even_head: bool | None = None):
    """V-exact student for blocks 0..k-1 and its constraints.

    `perms` and `head_maps` are indexed by block (the fitted neuron pairings and the M1 head maps);
    the even head is installed at full conversion unless `even_head` says otherwise. Returns
    (model, [MirrorMLPConstraint, MirrorAttnConstraint, VExactEmbeddings], build info). Run the
    trainer with train_scope = train_scope_for(train_embeddings).
    """
    L = int(adapter.depth)
    k = int(k)
    if not 1 <= k <= L:
        raise ValueError(f"k must be in [1, {L}]")
    blocks = list(range(k))
    even = (k == L) if even_head is None else bool(even_head)
    A = exact_involution(A)
    p_k = [perms[b] for b in blocks]
    model, info = build_vexact(adapter, A, blocks, p_k, [head_maps[b] for b in blocks], reg_perm=reg_perm,
                               even_head=even)
    As = [A] * k
    cons = [MirrorMLPConstraint(blocks, As, As, p_k),
            MirrorAttnConstraint(blocks, As, As, info["maps_after"]),
            VExactEmbeddings(adapter, A, blocks, reg_perm, train=train_embeddings)]
    return model, cons, info


def vexact_block_params(d: int, h: int, A, perms: Sequence, head_maps: Sequence) -> list:
    """Per converted block (aligned with `perms` and `head_maps`), the matrix parameters V-exact keeps:
    fc1 and fc2 (surgery.equivariant_param_count with A_in = A_out = A), their sum "mlp"
    (m1.mlp_mirror_params), Wq, Wk, Wv and Wo (attention.attention_free_params) and their sum "attn".
    vexact_param_counts sums these, so this is the accounting behind its totals."""
    if len(perms) != len(head_maps):
        raise ValueError("one perm and one HeadMaps per block")
    A = exact_involution(A)
    tr_a = involution_trace64(A)
    rows = []
    for b, (perm, hm) in enumerate(zip(perms, head_maps)):
        fc = equivariant_param_count(h, d, perm_trace(perm), tr_a)
        mlp = mlp_mirror_params(h, d, perm, A, A)
        if mlp != 2 * fc:
            raise AssertionError(f"block {b}: fc1 + fc2 = {2 * fc} but mlp_mirror_params gives {mlp}")
        att = attention_free_params(d, hm, A, A)
        rows.append({"block": b, "fc1": int(fc), "fc2": int(fc), "mlp": int(mlp),
                     **{key: int(att[key]) for key in ("Wq", "Wk", "Wv", "Wo")}, "attn": int(att["matrices"])})
    return rows


def matched_svd_ranks(block_params: Sequence[dict], h: int, d: int, D: int) -> list:
    """SVD ranks at V-exact's per-block budgets (E3): per block, fc1 and fc2 ranks from
    surgery.svd_rank_for and (r_q, r_k, r_v, r_o) from m1.attention_svd_ranks, the rules of the M3 pilot's
    SVD arm (the largest rank whose factorisation r (rows + cols) does not exceed the V-exact count of
    that matrix). Each row also holds the kept counts and the relative shortfall of SVD against V-exact,
    for the MLP and the attention of that block separately."""
    rows = []
    for p in block_params:
        r1, r2 = svd_rank_for(h, d, p["fc1"]), svd_rank_for(h, d, p["fc2"])
        ra = attention_svd_ranks(p, d, D)
        mlp_svd, att_svd = (r1 + r2) * (h + d), sum(ra) * (D + d)
        rows.append({"block": int(p["block"]), "mlp_ranks": [int(r1), int(r2)], "attn_ranks": [int(r) for r in ra],
                     "mlp_kept_svd": int(mlp_svd), "mlp_kept_vexact": int(p["mlp"]),
                     "attn_kept_svd": int(att_svd), "attn_kept_vexact": int(p["attn"]),
                     "mlp_rel_gap": (p["mlp"] - mlp_svd) / p["mlp"], "attn_rel_gap": (p["attn"] - att_svd) / p["attn"]})
    return rows


def vexact_param_counts(adapter, A, k: int, perms: Sequence, head_maps: Sequence) -> dict:
    """Matrix parameters of the converted blocks (fc1, fc2, Wq, Wk, Wv, Wo) kept by V-exact and dense,
    and the whole-model count after conversion (biases and embeddings counted as dense). Same
    convention as scripts/m1_attention.variant_params("vexact"). Sums vexact_block_params."""
    d, h = adapter.embed_dim, adapter.mlp_hidden
    H, dh = attn_dims(adapter.model.blocks[0])
    D = H * dh
    A = exact_involution(A)
    blocks = range(int(k))
    per = vexact_block_params(d, h, A, [perms[b] for b in blocks], [head_maps[b] for b in blocks])
    mlp_kept = sum(p["mlp"] for p in per)
    att_kept = sum(p["attn"] for p in per)
    mlp_dense, att_dense = 2 * h * d * int(k), 4 * D * d * int(k)
    n_model = int(sum(p.numel() for p in adapter.model.parameters()))
    dense, kept = mlp_dense + att_dense, mlp_kept + att_kept
    dp = _d_plus(A)
    return {"mlp_params_kept": int(mlp_kept), "attn_params_kept": int(att_kept), "converted_params_dense": int(dense),
            "converted_params_kept": int(kept), "converted_kept_frac": kept / dense if dense else 1.0,
            "model_params_original": n_model, "model_params": int(n_model - dense + kept),
            "model_params_frac": (n_model - dense + kept) / n_model, "d_plus": dp, "d_minus": int(A.shape[0] - dp)}


# ----------------------------------------------------------------------------------------------
# Checks after healing
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def span_exactness(adapter, model: nn.Module, xs: Sequence[torch.Tensor], A, k: int, reg_perm=None,
                   tol: float = 1e-4) -> dict:
    """M1 exactness probe (scripts/m1_attention.exactness_stage) for a given model: per stream layer
    0..depth, the largest |aligned mirrored tokens - original tokens A*| over the images, relative to
    the largest |original token entry|; pass if below `tol` at every layer inside the span (0..k)."""
    reg = register_info(model, adapter)
    perm = reg_perm if (reg_perm is not None and len(reg_perm)) else None
    acc = {}
    for x in xs:
        for r in stream_equivariance(adapter, model, x, A, perm, reg["start"]):
            c = acc.setdefault(r["layer"], {"max_abs_err": 0.0, "max_abs_ref": 0.0, "err_sq": 0.0, "ref_sq": 0.0})
            c["max_abs_err"] = max(c["max_abs_err"], r["max_abs_err"])
            c["max_abs_ref"] = max(c["max_abs_ref"], r["max_abs_ref"])
            c["err_sq"] += r["err_sq"]
            c["ref_sq"] += r["ref_sq"]
    rows = []
    for layer, c in sorted(acc.items()):
        rel = c["max_abs_err"] / c["max_abs_ref"] if c["max_abs_ref"] > 0 else float("nan")
        rows.append({"layer": int(layer), "in_span": layer <= k, "max_abs_err": c["max_abs_err"],
                     "max_abs_ref": c["max_abs_ref"], "rel_max": rel,
                     "rel_fro": math.sqrt(c["err_sq"] / c["ref_sq"]) if c["ref_sq"] > 0 else float("nan")})
    inside = [r["rel_max"] for r in rows if r["in_span"]]
    return {"k": int(k), "tol": float(tol), "max_rel_inside": max(inside), "pass": bool(all(v < tol for v in inside)),
            "n_images": int(sum(x.shape[0] for x in xs)), "dtype": str(next(model.parameters()).dtype), "rows": rows}


@torch.no_grad()
def flip_check(adapter, model: nn.Module, bank_loader, query_loader, probe_xs: Sequence[torch.Tensor],
               num_classes: int, device, k: int = 20, temperature: float = 0.07) -> dict:
    """The M1 full-conversion check (scripts/m1_attention.flip_stage) for a given model, in the model's
    dtype without autocast (fp32; run with TF32 off): k-NN flip consistency on the query images
    against the bank (must be 100% for an exactly invariant feature), top-1 on this small protocol,
    and the largest relative CLS feature difference between an image and its mirror."""
    bf, by = extract_features(adapter, bank_loader, device, model)
    qf, qy = extract_features(adapter, query_loader, device, model)
    qft, _ = extract_features(adapter, query_loader, device, model, flip=True)
    pred = knn_predict(bf, by, qf, num_classes, k, temperature)
    pred_t = knn_predict(bf, by, qft, num_classes, k, temperature)
    inv = [feature_invariance(adapter, model, x) for x in probe_xs]
    fc = 100.0 * float((pred == pred_t).float().mean())
    return {"flip_consistency": fc, "pass": bool(fc == 100.0), "top1_subset": 100.0 * float((pred == qy).float().mean()),
            "max_abs_unit_feature_diff": float((qft - qf).abs().max()),
            "cls_feature_rel_max": max(i["rel_max"] for i in inv) if inv else None,
            "n_bank": int(bf.shape[0]), "n_query": int(qf.shape[0]), "knn_k": int(k), "temperature": float(temperature),
            "dtype": str(next(model.parameters()).dtype),
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32), "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32)}
