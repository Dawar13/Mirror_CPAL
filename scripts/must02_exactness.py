#!/usr/bin/env python
"""MUST-02: exactness audit of Mirror-exact, component by component (PROJECT_NEXT_PHASE.md, MUST-02). The
proposition, the proof sketch and the per-component conditions are in docs/exactness.md.

    python scripts/must02_exactness.py --model dinov2_vitb14_reg                          # zero-shot, fp32
    python scripts/must02_exactness.py --model dinov2_vitb14_reg --dtype float64          # zero-shot, float64
    python scripts/must02_exactness.py --model dinov2_vitb14_reg --vexact-run runs/m3_vexact_5090/k12
    python scripts/must02_exactness.py --model dinov2_vitb14_reg --healed-run <healed run directory>
    python scripts/must02_exactness.py --tiny                                              # CPU smoke test

Model (one of three sources; the whole model is converted unless a healed run says otherwise):
  zero-shot     (default) mirror.m1.build_vexact with the span operator A* for all blocks, the neuron
                pairings, the M1 head maps and the register pairing, and the even head. With --span-structure
                (the canonical new-phase file, mirror.mirror_family.load_span_structure; results/MUST-07/
                structure/<model>_s<seed>.npz) all four come from that file, and its path and sha256 are
                recorded. Otherwise the structure is read from --m1-structure (default: the model's M1 file, as in scripts/tier2_linear.py) when
                it exists and holds A* for k = depth; otherwise it is fitted here exactly as
                scripts/m1_attention.py fits it (its structure_stage, on fit seed --fit-seed, saved under
                <out>/structure/). The neuron pairings and per-layer cross-moments come from --operators
                (scripts/run_e1.py output); --tiny fits them in-run (scripts/m1_attention.tiny_operators).
  --vexact-run  a scripts/heal_vexact.py run directory (<out_dir>/k<K>): the student rebuilt as that script
                builds it (its run config from summary.json, load_vexact_structure, build_vexact_student),
                then final/trained_blocks.pt loaded with mirror.heal.load_trained_blocks and checked tensor
                by tensor (scripts/tier2_linear.checkpoint_matches).
  --healed-run  any healed model, through mirror.heal_structured.load_healed_model(run_dir, device) ->
                (adapter, model, info), imported lazily. A* is info["A_star"] (top level or under
                info["structure"], where mirror.heal_structured puts it) if given, else read from the model's
                even head (A* = 2 P - I); the span from info["structure"]["blocks"] (default: all blocks).
                The register pairing, the neuron pairings and the head maps are then read from the weights
                (infer_structure): an exactly equivariant model satisfies them to rounding; the residuals and
                the agreement with the pairings in info are recorded.

Data: --n-images validation images (default 1,000, mirror.protocol.Protocol.query_loader: the first image
of every class), their indices saved, in --dtype (float32 or float64), TF32 off. Each batch runs as
[x; mirror(x)] in one forward pass.

Measurement: forward hooks at the output of every component of every block (plus the embeddings and the
final norm). For each component the expected action on its output space is applied to the original half,
in float64, and compared with the mirrored half at aligned positions (patch grid un-flipped, register j
replaced by register pi(j), CLS unchanged):
  stream    residual-stream tokens and the outputs of norms, LayerScale and branches: x -> x A*
  patch     patch embedding output (patch tokens only): x -> x A*
  head-q, head-k   per-head queries and keys: q_g -> B[sigma(g)] q_sigma(g)
  head-v, head-out per-head values and attention-weighted values (input of attn.proj): C instead of B
  attention        logits and probabilities [head, query, key], recomputed from the hooked q and k in the
                   model's dtype (the fused kernel never materialises them): head g -> head sigma(g), both
                   token axes aligned
  hidden    MLP pre-activation (fc1) and GELU output: unit i -> unit perm(i)
  invariant even projection, final affine and the CLS feature: identity (exact invariance)
Per component: max over images of ||T out(x) - out(mirror x)|| / ||out(mirror x)|| over the whole tensor of
an image (rel_tensor_max, the primary number) and per token (rel_token_max: the norm over every other axis
of one token, for attention the query token), plus the M1 convention max |error| / max |reference|.
Flip check (pre-registered setting): the k-NN rule of mirror.heal_vexact.flip_check (cosine k-NN of the
final-norm CLS feature, k = 20, T = 0.07) with fp32 features for the bank and the queries (no autocast, TF32
off), on all 50,000 validation images against the full 100k bank (--flip-queries, --flip-bank-per-class to
shrink it for smoke runs), plus the largest absolute CLS feature difference between an image and its mirror
in fp32. On CUDA the same check under bf16 autocast is reported beside it (flip_check_bf16). Prediction
(MUST-02): every component below --tol (1e-5) in fp32 and 100% fp32 flip consistency.

Outputs under --out (default results/MUST-02/<model>/<variant>[_fp64]/; tiny: tiny_out/MUST-02/<variant>):
  must02_components.csv  one row per (block, component)
  metrics.json           verdict, worst component, flip check, structure provenance, validation indices
  val_indices.npy        the validation images used (Protocol indices)
  meta.json              mirror.meta.write_meta (config, seeds, git hash, environment, wall time)
"""
from __future__ import annotations

import sys

import torch

DEFAULT_GPU_FRAC = 0.15


def _gpu_frac_from_argv(argv) -> float:
    """--gpu-frac from the command line, read before any other import so the cap is set first."""
    try:
        for i, s in enumerate(argv):
            if s == "--gpu-frac" and i + 1 < len(argv):
                return float(argv[i + 1])
            if s.startswith("--gpu-frac="):
                return float(s.split("=", 1)[1])
    except ValueError:
        pass
    return DEFAULT_GPU_FRAC


if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(_gpu_frac_from_argv(sys.argv[1:]))  # shared GPU: cap first

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from collections import OrderedDict  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.attention import HeadMaps, attn_dims, fold_attn  # noqa: E402
from mirror.data import make_loader, make_splits, mirror, normalize_for  # noqa: E402
from mirror.heal import exact_involution  # noqa: E402
from mirror.m1 import build_vexact, register_info  # noqa: E402
from mirror.meta import write_meta  # noqa: E402
from mirror.utils import REPO_ROOT, git_info, log, set_seed  # noqa: E402

EXPERIMENT = "MUST-02"
DEFAULT_TOL = 1e-5
CSV_NAME = "must02_components.csv"
SPACES = {
    "stream": "x -> x A* (tokens aligned)",
    "patch": "x -> x A* (patch grid aligned)",
    "head-q": "q_g -> B[sigma(g)] q_sigma(g) (tokens aligned)",
    "head-k": "k_g -> B[sigma(g)] k_sigma(g) (tokens aligned)",
    "head-v": "v_g -> C[sigma(g)] v_sigma(g) (tokens aligned)",
    "head-out": "o_g -> C[sigma(g)] o_sigma(g) (tokens aligned)",
    "attention": "head g -> head sigma(g), query and key tokens aligned",
    "hidden": "unit i -> unit perm(i) (tokens aligned)",
    "invariant": "identity (tokens aligned)",
}


# ----------------------------------------------------------------------------------------------
# Script loading and the GPU cap
# ----------------------------------------------------------------------------------------------
def _t2():
    """scripts/tier2_linear.py under the key every Tier-2 script uses (imported once per process)."""
    key = "_t2_tier2_linear"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, REPO_ROOT / "scripts" / "tier2_linear.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            del sys.modules[key]
            raise
    return sys.modules[key]


def load_script(name: str, gpu_frac: float | None = None):
    """scripts/<name>.py (tier2_linear.load_script: one import per process), then this process's cap again
    (the imported scripts set their own caps at import time)."""
    t2 = _t2()
    if gpu_frac is not None:
        t2.apply_memory_cap(gpu_frac)
    return t2 if name == "tier2_linear" else t2.load_script(name)


def _resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


# ----------------------------------------------------------------------------------------------
# Structure of an exactly equivariant model
# ----------------------------------------------------------------------------------------------
@dataclass
class ExactStructure:
    """What the exact symmetry of a Mirror-exact model is: the span operator A* (exact involution, float64),
    the converted blocks 0..k-1, the register involution, and per converted block the neuron involution
    and the head maps valid for the model's weights (after the twin gauge)."""

    A: np.ndarray
    k: int
    reg_perm: np.ndarray
    perms: list
    maps: list
    source: dict = field(default_factory=dict)

    @property
    def d_plus(self) -> int:
        return int(round((float(np.trace(self.A)) + self.A.shape[0]) / 2))


def token_index(grid_hw, n_prefix: int, reg_perm=None, reg_start: int = 1) -> torch.Tensor:
    """Index array idx with t_mirror[:, idx] = the mirrored image's tokens at the original's positions:
    patch grid un-flipped left-right, register j replaced by register pi(j), CLS unchanged (the same
    alignment as mirror.m1.align_mirror_tokens, as one gather that also serves the key axis of attention)."""
    H, W = (int(v) for v in grid_hw)
    idx = np.arange(int(n_prefix) + H * W)
    idx[n_prefix:] = n_prefix + np.arange(H * W).reshape(H, W)[:, ::-1].reshape(-1)
    if reg_perm is not None and len(reg_perm):
        R = len(reg_perm)
        idx[reg_start:reg_start + R] = reg_start + np.asarray(reg_perm, dtype=np.int64)
    return torch.as_tensor(idx, dtype=torch.long)


def operator_from_even_head(model: nn.Module):
    """A* from the even head y -> y (I + A*) / 2 inside model.norm (None without one)."""
    even = getattr(getattr(model, "norm", None), "even", None)
    if even is None or not hasattr(even, "P"):
        return None
    P = even.P.detach().double().cpu().numpy()
    return exact_involution(2.0 * P - np.eye(P.shape[0]))


def _polar(M: torch.Tensor) -> torch.Tensor:
    U, _, Vh = torch.linalg.svd(M)
    return U @ Vh


@torch.no_grad()
def infer_structure(adapter, model: nn.Module, A, k: int | None = None, tol: float = 1e-3) -> tuple:
    """Read the exact symmetry of a Mirror-exact model from its weights, given A* (float64, CPU):
      registers  r_j A* = r_pi(j)  (projected register tokens; pi(j) = the nearest r_i to r_j A*);
      neurons    P W1 = W1 A*, P b1 = b1  (perm(i) = the row of [W1, b1] nearest to row i of [W1 A*, b1]);
      heads      Wq^g A* = B Wq^h, Wk^g A* = B Wk^h, bq^g = B bq^h, bk^g = B bk^h for g = sigma(h): for each
                 h the head g with the smallest residual of the best orthogonal B (polar factor), then C
                 from Wv^g A* = C Wv^h, bv^g = C bv^h and Wo^g^T = C Wo^h^T A*.
    Raises ValueError if sigma or a neuron pairing is not an involution or a relative residual exceeds `tol`
    (the model is then not exactly equivariant under A*). Returns (ExactStructure, residuals)."""
    A64 = torch.as_tensor(np.asarray(A, dtype=np.float64))
    L = len(model.blocks)
    k = L if k is None else int(k)
    res = {"registers": 0.0, "mlp": [], "qk": [], "vo": []}
    reg = register_info(model, adapter)
    reg_perm = np.zeros(0, dtype=np.int64)
    if reg["R"]:
        r = getattr(model, reg["name"]).detach().double().cpu()[0]
        pi = torch.cdist(r @ A64, r).argmin(1)         # matching only; residuals computed directly (cdist cancels)
        reg_perm = pi.numpy().astype(np.int64)
        res["registers"] = float(((r @ A64 - r[pi]).norm(dim=1) / r.norm(dim=1).clamp_min(1e-300)).max())
    perms, maps = [], []
    for b in range(k):
        blk = model.blocks[b]
        W1 = blk.mlp.fc1.weight.detach().double().cpu()
        b1 = (blk.mlp.fc1.bias.detach().double().cpu() if blk.mlp.fc1.bias is not None else W1.new_zeros(W1.shape[0]))
        X = torch.cat([W1, b1[:, None]], 1)
        Y = torch.cat([W1 @ A64, b1[:, None]], 1)
        perm = torch.cdist(Y, X).argmin(1)
        res["mlp"].append(float(((Y - X[perm]).norm(dim=1) / X.norm(dim=1).clamp_min(1e-300)).max()))
        perm = perm.numpy().astype(np.int64)
        if not np.array_equal(perm[perm], np.arange(len(perm))):
            raise ValueError(f"block {b + 1}: the neuron pairing read from the weights is not an involution")
        perms.append(perm)
        f = {key: v.detach().double().cpu() for key, v in fold_attn(blk).items()}
        H, dh = attn_dims(blk)
        d = W1.shape[1]
        Wq, Wk, Wv = (f[key].reshape(H, dh, d) for key in ("Wq", "Wk", "Wv"))
        bq, bk, bv = (f[key].reshape(H, dh, 1) for key in ("bq", "bk", "bv"))
        Wo = f["Wo"].reshape(d, H, dh)
        Xqk = [torch.cat([Wq[h], Wk[h], bq[h], bk[h]], 1) for h in range(H)]
        Yqk = [torch.cat([Wq[g] @ A64, Wk[g] @ A64, bq[g], bk[g]], 1) for g in range(H)]
        sigma = np.zeros(H, dtype=np.int64)
        Bm, Cm = np.zeros((H, dh, dh)), np.zeros((H, dh, dh))
        r_qk, r_vo = [], []
        for h in range(H):
            best = None
            for g in range(H):
                Bg = _polar(Yqk[g] @ Xqk[h].T)
                rr = float((Yqk[g] - Bg @ Xqk[h]).norm() / Yqk[g].norm().clamp_min(1e-300))
                if best is None or rr < best[0]:
                    best = (rr, g, Bg)
            rr, g, Bg = best
            sigma[h], Bm[h] = g, Bg.numpy()
            r_qk.append(rr)
            Xv = torch.cat([Wv[h], bv[h], Wo[:, h].T @ A64], 1)
            Yv = torch.cat([Wv[g] @ A64, bv[g], Wo[:, g].T], 1)
            Cg = _polar(Yv @ Xv.T)
            Cm[h] = Cg.numpy()
            r_vo.append(float((Yv - Cg @ Xv).norm() / Yv.norm().clamp_min(1e-300)))
        if not np.array_equal(sigma[sigma], np.arange(H)):
            raise ValueError(f"block {b + 1}: the head map read from the weights is not an involution: {sigma.tolist()}")
        res["qk"].append(max(r_qk))
        res["vo"].append(max(r_vo))
        maps.append(HeadMaps(sigma=sigma, B=Bm, C=Cm, info={"source": "weights"}))
    worst = max([res["registers"]] + res["mlp"] + res["qk"] + res["vo"])
    res["max"] = worst
    if worst > tol:
        raise ValueError(f"the weights are not exactly equivariant under the given A*: largest relative residual "
                         f"{worst:.2e} > {tol:g} (registers {res['registers']:.1e}, MLP {max(res['mlp'], default=0):.1e}, "
                         f"q/k {max(res['qk'], default=0):.1e}, v/o {max(res['vo'], default=0):.1e})")
    st = ExactStructure(A=exact_involution(A), k=k, reg_perm=reg_perm, perms=perms, maps=maps,
                        source={"reg_perm": "weights", "perms": "weights", "maps": "weights"})
    return st, res


# ----------------------------------------------------------------------------------------------
# The audit
# ----------------------------------------------------------------------------------------------
def _new_acc(meta: dict) -> dict:
    return {**meta, "n_images": 0, "rel_tensor_max": 0.0, "rel_token_max": 0.0, "worst_image": None,
            "max_abs_err": 0.0, "max_abs_ref": 0.0, "err_sq": 0.0, "ref_sq": 0.0, "min_token_ref": math.inf}


class ExactnessAudit:
    """Forward hooks on every component of `model`. Run it on batches [x; mirror(x)]: each hook applies the
    component's expected action to the original half in float64, compares it with the aligned mirrored half
    and keeps running maxima (nothing is stored). See the module docstring for the actions."""

    def __init__(self, adapter, model: nn.Module, st: ExactStructure, device):
        self.adapter, self.model, self.st = adapter, model, st
        self.device = torch.device(device)
        self.L = len(model.blocks)
        dev = self.device
        self.A = torch.as_tensor(np.asarray(st.A, dtype=np.float64), device=dev)
        reg = register_info(model, adapter)
        self.reg_start = int(reg["start"])
        rp = st.reg_perm if (st.reg_perm is not None and len(st.reg_perm)) else None
        self.idx = token_index(adapter.grid_hw, adapter.n_prefix_tokens, rp, self.reg_start).to(dev)
        self.idx_patch = token_index(adapter.grid_hw, 0).to(dev)
        self.perms = [torch.as_tensor(np.asarray(p), dtype=torch.long, device=dev) for p in st.perms]
        self.sigma = [torch.as_tensor(np.asarray(hm.sigma), dtype=torch.long, device=dev) for hm in st.maps]
        self.Bm = [torch.as_tensor(np.asarray(hm.B, dtype=np.float64), device=dev) for hm in st.maps]
        self.Cm = [torch.as_tensor(np.asarray(hm.C, dtype=np.float64), device=dev) for hm in st.maps]
        self.acc: OrderedDict = OrderedDict()
        self.B = 0
        self.offset = 0
        self.handles = []
        self.notes = []

    # ----- actions (each returns (T out(x), aligned out(mirror x)), float64, token axis 1) -----------
    def _halves(self, t: torch.Tensor, idx: torch.Tensor):
        B = self.B
        return t[:B].double(), t[B:].double()[:, idx]

    def _stream(self, t):
        X, Xt = self._halves(t, self.idx)
        return X @ self.A, Xt

    def _patch(self, t):
        if t.dim() == 4:                       # [B, H, W, C] (NHWC patch embedding)
            t = t.reshape(t.shape[0], -1, t.shape[-1])
        X, Xt = self._halves(t, self.idx_patch)
        return X @ self.A, Xt

    def _invariant(self, t):
        return self._halves(t, self.idx)

    def _hidden(self, t, b):
        X, Xt = self._halves(t, self.idx)
        return X[..., self.perms[b]], Xt

    def _heads(self, y, M, sigma):
        """y [2B, T, H, dh]; head g of the mirror (aligned) against M[sigma(g)] y_sigma(g) of the original."""
        X, Xt = self._halves(y, self.idx)
        return torch.einsum("bthd,hed->bthe", X, M)[:, :, sigma], Xt

    def _attention(self, P, sigma):
        """P [2B, H, T, T] (query, key); returned with the query axis first: [B, T, H, T]."""
        B = self.B
        X = P[:B].double()[:, sigma]
        Xt = P[B:].double()[:, :, self.idx][:, :, :, self.idx]
        return X.permute(0, 2, 1, 3), Xt.permute(0, 2, 1, 3)

    # ----- accumulation ---------------------------------------------------------------------------
    def _add(self, key, meta: dict):
        if key not in self.acc:
            self.acc[key] = _new_acc(meta)

    def _rec(self, key, pair) -> None:
        X_t, Xt = pair
        E = X_t - Xt
        B, T = E.shape[0], E.shape[1]
        e_img = E.reshape(B, -1).norm(dim=1)
        r_img = Xt.reshape(B, -1).norm(dim=1)
        e_tok = E.reshape(B, T, -1).norm(dim=2)
        r_tok = Xt.reshape(B, T, -1).norm(dim=2)
        rel_img = e_img / r_img.clamp_min(1e-300)
        rel_tok = e_tok / r_tok.clamp_min(1e-300)
        a = self.acc[key]
        i = int(rel_img.argmax())
        if float(rel_img[i]) >= a["rel_tensor_max"]:
            a["rel_tensor_max"], a["worst_image"] = float(rel_img[i]), self.offset + i
        a["rel_token_max"] = max(a["rel_token_max"], float(rel_tok.max()))
        a["max_abs_err"] = max(a["max_abs_err"], float(E.abs().max()))
        a["max_abs_ref"] = max(a["max_abs_ref"], float(Xt.abs().max()))
        a["err_sq"] += float((E * E).sum())
        a["ref_sq"] += float((Xt * Xt).sum())
        a["min_token_ref"] = min(a["min_token_ref"], float(r_tok.min()))
        a["n_images"] += B

    # ----- hooks ------------------------------------------------------------------------------------
    def _hook_out(self, module, fn):
        self.handles.append(module.register_forward_hook(lambda _m, _args, out: fn(out)))

    def _hook_in(self, module, fn):
        self.handles.append(module.register_forward_pre_hook(lambda _m, args: fn(args[0])))

    def _meta(self, stage, b, component, space, in_span, note=""):
        return {"stage": stage, "block": None if b is None else b + 1, "code_index": b, "component": component,
                "space": space, "action": SPACES[space], "in_span": bool(in_span), "note": note}

    def attach(self) -> None:
        m, st = self.model, self.st
        L, k = self.L, int(st.k)
        key = ("embed", None, "patch_embed")
        self._add(key, self._meta("embed", None, "patch_embed", "patch", True, "flipped kernels"))
        self._hook_out(m.patch_embed, lambda o, key=key: self._rec(key, self._patch(o)))
        key = ("embed", None, "tokens0")
        self._add(key, self._meta("embed", None, "tokens0", "stream", True,
                                  "patch embedding + positions, CLS, registers: the stream entering block 1"))
        self._hook_in(m.blocks[0], lambda t, key=key: self._rec(key, self._stream(t)))
        for b, blk in enumerate(m.blocks):
            self._attach_block(b, blk, b < k)
        norm = m.norm
        full = k == L
        if isinstance(norm, nn.Sequential) and hasattr(norm, "rms"):
            key = ("final", None, "final_rms")
            self._add(key, self._meta("final", None, "final_rms", "stream", full, "final RMSNorm (no affine)"))
            self._hook_out(norm.rms, lambda o, key=key: self._rec(key, self._stream(o)))
            if hasattr(norm, "even"):
                for name, mod, note in (("even_proj", norm.even, "y (I + A*) / 2"),
                                        ("final_affine", norm.affine, "final affine (gamma, beta): the features")):
                    key = ("final", None, name)
                    self._add(key, self._meta("final", None, name, "invariant", full, note))
                    self._hook_out(mod, lambda o, key=key: self._rec(key, self._invariant(o)))
                key = ("final", None, "cls_feature")
                self._add(key, self._meta("final", None, "cls_feature", "invariant", full,
                                          "final-norm CLS token: the k-NN feature"))
                self._hook_out(norm.affine, lambda o, key=key: self._rec(key, self._cls(o)))
            else:
                self.notes.append("no even head: the final affine is not checked (a per-channel affine does not "
                                  "commute with A*)")
        else:
            key = ("final", None, "final_norm")
            self._add(key, self._meta("final", None, "final_norm", "stream", False,
                                      "not a folded norm: not expected to be exact"))
            self._hook_out(norm, lambda o, key=key: self._rec(key, self._stream(o)))

    def _cls(self, o):
        X, Xt = self._invariant(o)
        return X[:, :1], Xt[:, :1]

    def _attach_block(self, b: int, blk: nn.Module, in_span: bool) -> None:
        def add(name, space, note=""):
            key = ("block", b, name)
            self._add(key, self._meta("block", b, name, space, in_span, note))
            return key

        stream = lambda key: (lambda t: self._rec(key, self._stream(t)))  # noqa: E731
        self._hook_out(blk.norm1, stream(add("norm1", "stream", "RMSNorm (LayerNorm folded)")))
        attn = blk.attn
        if in_span:
            H, dh = attn_dims(blk)
            keys = {n: add(n, s) for n, s in (("q", "head-q"), ("k", "head-k"), ("v", "head-v"))}
            keys["logits"] = add("attn_logits", "attention", "q k^T scale, recomputed from the hooked q and k")
            keys["probs"] = add("attn_probs", "attention", "softmax of the logits, in the model's dtype")
            scale = float(attn.scale)

            def on_qkv(y, b=b, H=H, dh=dh, keys=keys, scale=scale):
                y = y.reshape(y.shape[0], y.shape[1], 3, H, dh)
                q, k_, v = y[:, :, 0], y[:, :, 1], y[:, :, 2]
                self._rec(keys["q"], self._heads(q, self.Bm[b], self.sigma[b]))
                self._rec(keys["k"], self._heads(k_, self.Bm[b], self.sigma[b]))
                self._rec(keys["v"], self._heads(v, self.Cm[b], self.sigma[b]))
                logits = (q.transpose(1, 2) * scale) @ k_.transpose(1, 2).transpose(-2, -1)
                self._rec(keys["logits"], self._attention(logits, self.sigma[b]))
                self._rec(keys["probs"], self._attention(logits.softmax(dim=-1), self.sigma[b]))

            self._hook_out(attn.qkv, on_qkv)
            ko = add("head_out", "head-out", "attention-weighted values per head (input of attn.proj)")
            self._hook_in(attn.proj, lambda t, ko=ko, b=b, H=H, dh=dh: self._rec(
                ko, self._heads(t.reshape(t.shape[0], t.shape[1], H, dh), self.Cm[b], self.sigma[b])))
        self._hook_out(attn.proj, stream(add("attn_proj", "stream", "attention branch")))
        if getattr(blk, "ls1", None) is not None:
            self._hook_out(blk.ls1, stream(add("ls1", "stream", "LayerScale (folded: gamma = 1)")))
        self._hook_in(blk.norm2, stream(add("resid_attn", "stream", "x + attention branch")))
        self._hook_out(blk.norm2, stream(add("norm2", "stream", "RMSNorm (LayerNorm folded)")))
        if in_span:
            k1 = add("fc1", "hidden", "pre-activation")
            self._hook_out(blk.mlp.fc1, lambda t, k1=k1, b=b: self._rec(k1, self._hidden(t, b)))
            ka = add("act", "hidden", "GELU output")
            self._hook_out(blk.mlp.act, lambda t, ka=ka, b=b: self._rec(ka, self._hidden(t, b)))
        self._hook_out(blk.mlp.fc2, stream(add("fc2", "stream", "MLP branch")))
        if getattr(blk, "ls2", None) is not None:
            self._hook_out(blk.ls2, stream(add("ls2", "stream", "LayerScale (folded: gamma = 1)")))
        self._hook_out(blk, stream(add("block_out", "stream", "x + MLP branch: the block output")))

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    @torch.no_grad()
    def run(self, batches) -> "ExactnessAudit":
        """Each batch: normalised images [B, 3, S, S] in the model's dtype."""
        if not self.handles:
            self.attach()
        try:
            for x in batches:
                x = x.to(self.device)
                self.B = int(x.shape[0])
                self.model.norm(self.adapter.forward_tokens(torch.cat([x, mirror(x)], dim=0), self.model))
                self.offset += self.B
        finally:
            self.detach()
        return self

    def rows(self, tol: float = DEFAULT_TOL) -> list:
        out = []
        for a in self.acc.values():
            r = {k_: v for k_, v in a.items() if k_ not in ("err_sq", "ref_sq")}
            r["rel_fro_all"] = math.sqrt(a["err_sq"] / a["ref_sq"]) if a["ref_sq"] > 0 else float("nan")
            r["rel_maxabs"] = a["max_abs_err"] / a["max_abs_ref"] if a["max_abs_ref"] > 0 else float("nan")
            if not math.isfinite(r["min_token_ref"]):
                r["min_token_ref"] = float("nan")
            r["pass"] = bool(r["rel_tensor_max"] < tol) if a["in_span"] and a["n_images"] else None
            r["pass_token"] = bool(r["rel_token_max"] < tol) if a["in_span"] and a["n_images"] else None
            out.append(r)
        return out


def audit_model(adapter, model, st: ExactStructure, batches, device, tol: float = DEFAULT_TOL) -> list:
    """Rows of the per-component audit of `model` on `batches` (normalised images in the model's dtype)."""
    return ExactnessAudit(adapter, model, st, device).run(batches).rows(tol)


def summarise(rows: list, tol: float) -> dict:
    span = [r for r in rows if r["in_span"] and r["n_images"]]
    if not span:
        return {"n_components": 0}
    worst = max(span, key=lambda r: r["rel_tensor_max"])
    worst_tok = max(span, key=lambda r: r["rel_token_max"])

    def where(r):
        return (f"block {r['block']}" if r["block"] is not None else r["stage"]) + f" {r['component']}"

    return {"n_components": len(span), "tol": tol,
            "max_rel_tensor": worst["rel_tensor_max"], "worst_component": where(worst),
            "max_rel_token": worst_tok["rel_token_max"], "worst_component_token": where(worst_tok),
            "max_rel_maxabs": max(r["rel_maxabs"] for r in span),
            "all_below_tol": bool(all(r["pass"] for r in span)),
            "all_below_tol_per_token": bool(all(r["pass_token"] for r in span)),
            "n_above_tol": int(sum(not r["pass"] for r in span)),
            "n_above_tol_per_token": int(sum(not r["pass_token"] for r in span)),
            "max_by_space": {s: max(r["rel_tensor_max"] for r in span if r["space"] == s)
                             for s in sorted({r["space"] for r in span})}}


# ----------------------------------------------------------------------------------------------
# Models: zero-shot, healed V-exact (scripts/heal_vexact.py), any healed run (mirror.heal_structured)
# ----------------------------------------------------------------------------------------------
def _m1_args(a, tiny: bool, model_name: str, fit_seed: int, random_seed: int = 1, margin: float = 0.0):
    return SimpleNamespace(model=model_name, tiny=tiny, fit_seed=int(fit_seed), random_seed=int(random_seed),
                           register_margin=float(margin), force=False, no_save_stats=True, seed=int(a.seed),
                           hub_repo=a.hub_repo, img_size=int(a.img_size))


def build_adapter(a, device):
    """The model as scripts/m1_attention.py builds it (DINOv2 from torch.hub, other timm models through
    mirror.adapters_timm; --tiny: the random 4-block timm ViT), float32, eval mode."""
    m1 = load_script("m1_attention", a.gpu_frac)
    return m1.build_adapter(_m1_args(a, a.tiny, a.model, a.fit_seed), device)


def zeroshot_structure(a, adapter, proto, device, out: Path, git: dict) -> tuple:
    """(A*, reg_perm, perms, head maps, provenance) for full conversion, as scripts/m1_attention.py makes them."""
    m1 = load_script("m1_attention", a.gpu_frac)
    t2 = _t2()
    L, d, h = adapter.depth, adapter.embed_dim, adapter.mlp_hidden
    if getattr(a, "span_structure", None):
        # The canonical structure file of the new phase (mirror.mirror_family SpanStructure, written by
        # scripts/must07_compressors.py to results/MUST-07/structure/<model>_s<fit seed>.npz).
        from mirror.mirror_family import load_span_structure

        path = _resolve(a.span_structure)
        if not path.exists():
            raise SystemExit(f"--span-structure {path} not found")
        ss = load_span_structure(path)
        if (ss.depth, ss.d, ss.h) != (L, d, h):
            raise SystemExit(f"{path}: structure for depth {ss.depth}, d {ss.d}, h {ss.h}; the model has {(L, d, h)}")
        name = "tiny" if a.tiny else a.model
        if ss.model and ss.model != name:
            log(f"WARNING: {path} was fitted on {ss.model!r}, this run is {name!r}")
        src = {"span_structure": str(path), "span_structure_sha256": t2.sha256(path), "fit_seed": int(ss.fit_seed),
               "fitted_here": False, "structure_model": ss.model,
               "parts": "A* (k = depth), neuron pairings, head maps (gauged by build_vexact) and register pairing"}
        return (exact_involution(ss.A_star[L]), np.asarray(ss.reg_perm[L], dtype=np.int64), list(ss.perms),
                list(ss.heads), src)
    defaults = {} if a.tiny else dict(t2.MODEL_DEFAULTS.get(a.model, {}))
    sdir = out / "structure"
    if a.operators:
        ops_path = _resolve(a.operators)
    elif a.tiny:
        sdir.mkdir(parents=True, exist_ok=True)
        ops_path = sdir / "tiny_operators.npz"
        if not ops_path.exists():
            m1.tiny_operators(adapter, proto.fit_loader(a.fit_seed), device, ops_path)
    elif defaults.get("operators"):
        ops_path = _resolve(defaults["operators"])
    else:
        raise SystemExit("--operators is required for this model (a scripts/run_e1.py e1_operators.npz)")
    if not ops_path.exists():
        raise SystemExit(f"{ops_path} not found: pass --operators (scripts/run_e1.py results/e1_operators.npz)")
    ops = m1.load_operators(ops_path, L, d, h)
    name = "tiny" if a.tiny else a.model
    S, src = None, None
    given = _resolve(a.m1_structure) if a.m1_structure else None
    if given is not None and not given.exists():
        raise SystemExit(f"--m1-structure {given} not found")
    path = given or (_resolve(defaults["m1_structure"]) if defaults.get("m1_structure") else None)
    if path is not None and path.exists():
        with np.load(path) as z:
            saved = {"fit_seed": int(z["fit_seed"]), "random_seed": int(z["random_seed"]),
                     "register_margin": float(z["register_margin"])}
        S = m1._load_structure(path, _m1_args(a, a.tiny, name, saved["fit_seed"], saved["random_seed"],
                                               saved["register_margin"]), ops, [L])
        if S is not None and L not in S["A_star"]:
            S = None
        if S is None:
            if given is not None:
                raise SystemExit(f"{path} has no span operator for k = {L}; rerun scripts/m1_attention.py with --ks {L}")
            log(f"{path} has no span operator for k = {L}: fitting the structure here")
        else:
            src = {"m1_structure": str(path), "fit_seed": saved["fit_seed"], "fitted_here": False}
    if S is None:
        log(f"fitting the M1 structure on fit seed {a.fit_seed} (scripts/m1_attention.structure_stage) into {sdir}")
        sdir.mkdir(parents=True, exist_ok=True)
        tags = {"git_hash": git["git_hash"], "seed": a.seed, "fit_seed": a.fit_seed, "random_seed": 1}
        S = m1.structure_stage(_m1_args(a, a.tiny, name, a.fit_seed), adapter, {"fit": proto.fit_loader(a.fit_seed)},
                               ops, [L], sdir, device, git, tags)
        src = {"m1_structure": str(sdir / "m1_structure.npz"), "fit_seed": int(a.fit_seed), "fitted_here": True}
    src.update({"operators": str(ops_path), "operators_sha256": ops["sha256"]})
    return S["A_star"][L], np.asarray(S["reg_perm"][L], dtype=np.int64), ops["perms"], S["maps"], src


def load_zeroshot(a, adapter, proto, device, out: Path, git: dict, dtype) -> tuple:
    A, reg_perm, perms, maps, src = zeroshot_structure(a, adapter, proto, device, out, git)
    L = adapter.depth
    adapter.model.to(dtype)                 # build in the audit dtype: the projection is exact to that dtype
    model, info = build_vexact(adapter, A, list(range(L)), perms, maps, reg_perm=reg_perm, even_head=True)
    st = ExactStructure(A=exact_involution(A), k=L, reg_perm=reg_perm, perms=list(perms), maps=info["maps_after"],
                        source={"kind": "zero-shot mirror.m1.build_vexact (full conversion, even head)", **src})
    return model, st


def load_vexact_run(a, device) -> tuple:
    """A scripts/heal_vexact.py run (<out_dir>/k<K>), rebuilt and loaded as that script documents."""
    hv = load_script("heal_vexact", a.gpu_frac)
    t2 = _t2()
    from mirror.heal import load_trained_blocks
    from mirror.heal_vexact import build_vexact_student
    from mirror.utils import load_config

    run = _resolve(a.vexact_run)
    ck = run / "final" / "trained_blocks.pt"
    if not ck.is_file():
        raise SystemExit(f"{ck} not found (a scripts/heal_vexact.py run directory <out_dir>/k<K> is expected)")
    summ_path = run / "summary.json"
    if summ_path.exists():
        summ = json.loads(summ_path.read_text(encoding="utf-8"))
        cfg, cfg_src = dict(summ["config"]), str(summ_path)
        k = int(summ["k"])
    else:
        cfg, cfg_src = load_config(_resolve(a.vexact_config), tiny=a.tiny), str(a.vexact_config)
        m = re.fullmatch(r"k(\d+)", run.name)
        if not m:
            raise SystemExit(f"{summ_path} missing and {run.name} is not k<K>: cannot tell the span")
        k = int(m.group(1))
        log(f"WARNING: {summ_path} missing; rebuilding from {cfg_src} with k = {k}")
    cfg["k"], cfg["method"] = k, "vexact"
    if bool(cfg.get("tiny_mode")) != bool(a.tiny):
        raise SystemExit(f"{cfg_src} is {'a tiny' if cfg.get('tiny_mode') else 'not a tiny'} run; pass --tiny accordingly")
    adapter = hv.heal_cli.build_teacher(cfg, device)
    adapter.model.float().eval()
    st_ = hv.load_vexact_structure(cfg, adapter, k)
    train_emb = bool((cfg.get("vexact") or {}).get("train_embeddings", True))
    model, cons, binfo = build_vexact_student(adapter, st_["A_star"], k, st_["perms"], st_["heads"], st_["reg_perm"],
                                              train_embeddings=train_emb)
    for c in cons:
        prep = getattr(c, "prepare", None)
        if callable(prep):
            prep(model)
    load_trained_blocks(model, ck)
    loaded = t2.checkpoint_matches(model, ck, list(range(k)))
    model.eval()
    st = ExactStructure(A=exact_involution(st_["A_star"]), k=k, reg_perm=np.asarray(st_["reg_perm"], dtype=np.int64),
                        perms=[st_["perms"][b] for b in range(k)], maps=binfo["maps_after"],
                        source={"kind": "healed V-exact (scripts/heal_vexact.py)", "run": str(run), "checkpoint": str(ck),
                                "checkpoint_sha256": t2.sha256(ck), "config": cfg_src,
                                "span_source": st_.get("span_source"), "operators_source": st_.get("operators_source"),
                                "heads_source": st_.get("heads_source"), "checkpoint_check": loaded,
                                "heal_seed": cfg.get("seed")})
    return adapter, model, st


def _info_get(info: dict, *keys):
    for key in keys:
        v = info.get(key) if isinstance(info, dict) else None
        if v is None and isinstance(info, dict) and isinstance(info.get("structure"), dict):
            v = info["structure"].get(key)
        if v is not None:
            return v
    return None


def load_healed_run(a, device) -> tuple:
    """Any healed model through mirror.heal_structured.load_healed_model (imported here, lazily)."""
    from mirror.heal_structured import load_healed_model

    run = _resolve(a.healed_run)
    adapter, model, info = load_healed_model(run, device)
    adapter.model.float().eval()
    model.float().eval()
    info = info or {}
    A = _info_get(info, "A_star", "A")
    a_src = "info"
    if A is None:
        A = operator_from_even_head(model)
        a_src = "even head of the model (A* = 2 P - I)"
    if A is None:
        raise SystemExit(f"{run}: no A* in the loader's info and no even head in the model; not a Mirror-exact model")
    A = exact_involution(np.asarray(A, dtype=np.float64))
    L = len(model.blocks)
    k = _info_get(info, "k")
    blocks = _info_get(info, "blocks")
    if blocks is None and isinstance(info.get("spec"), dict):
        blocks = info["spec"].get("blocks")
    if k is None and blocks is not None:
        blocks = [int(b) for b in blocks]
        if blocks != list(range(len(blocks))):
            raise SystemExit(f"{run}: converted blocks {blocks} do not start at block 1; not a Mirror-exact span")
        k = len(blocks)
    k = int(k) if k is not None else L
    st, res = infer_structure(adapter, model, A, k)
    agree = {}
    for key, mine in (("reg_perm", st.reg_perm), ("perms", st.perms)):
        given = _info_get(info, key)
        if given is not None:
            given = [np.asarray(p) for p in given] if key == "perms" else np.asarray(given)
            same = (all(np.array_equal(np.asarray(g), m) for g, m in zip(given, mine)) if key == "perms"
                    else np.array_equal(given, mine))
            agree[key] = bool(same)
    given_maps = _info_get(info, "maps_after")
    if given_maps is not None:
        agree["head_sigma"] = bool(all(np.array_equal(np.asarray(g.sigma), np.asarray(m.sigma))
                                       for g, m in zip(given_maps, st.maps)))
    st.source.update({"kind": "healed run (mirror.heal_structured.load_healed_model)", "run": str(run), "A_star": a_src,
                      "k": k, "inference_residuals": res, "agrees_with_info": agree,
                      "method": info.get("method"), "family": info.get("family"),
                      "info_keys": sorted(str(x) for x in info.keys()) if isinstance(info, dict) else None})
    return adapter, model, st


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def load_data(a, adapter, device) -> dict:
    """The audit's validation subset (indices) and its loader; the flip check's bank and queries (default:
    the full 100k bank and all 50,000 validation images, the paper protocol); the fit loader (proto)."""
    pin = torch.device(device).type == "cuda"
    nw = 0 if a.tiny else a.num_workers
    if a.tiny:
        from mirror.variants import TinyProtocol

        proto = TinyProtocol(img_size=int(adapter.img_size), batch_size=a.knn_batch_size)
        idx = np.asarray(proto.query_idx)[:a.n_images]
        flip_idx = np.asarray(proto.query_idx) if not a.flip_queries else np.asarray(proto.query_idx)[:a.flip_queries]
        bank = proto.bank_loader()
        num_classes, desc = proto.num_classes, "tiny synthetic"
    else:
        from mirror.protocol import Protocol

        proto = Protocol(_resolve(a.cache_dir), a.knn_batch_size, 16, a.num_workers, pin)
        n_req = int(math.ceil(a.n_images / 1000.0)) * 1000
        idx = np.asarray(proto.query_loader(n_req).dataset.indices)[:a.n_images]
        flip_idx = np.asarray(proto.query_loader(a.flip_queries or None).dataset.indices)
        bank_rows = np.asarray(proto.index["bank_rows"])
        if a.flip_bank_per_class:
            sub = make_splits(proto.train_labels[bank_rows], {"b": int(a.flip_bank_per_class)})["b"]
            bank = make_loader(proto.train, proto.train_labels, bank_rows[sub], a.knn_batch_size, nw, pin)
        else:
            bank = proto.bank_loader()
        num_classes, desc = 1000, "ImageNet validation (Protocol.query_loader: first images of each class)"
    query = make_loader(proto.val, proto.val_labels, idx, a.batch_size, nw, pin)
    flip_query = make_loader(proto.val, proto.val_labels, flip_idx, a.knn_batch_size, nw, pin)
    return {"proto": proto, "idx": idx, "query": query, "bank": bank, "flip_query": flip_query,
            "num_classes": num_classes, "description": desc, "n_bank": int(len(bank.dataset)),
            "n_flip_query": int(len(flip_idx))}


def image_batches(adapter, loader, device, dtype) -> list:
    out = []
    for imgs, _ in loader:
        out.append(normalize_for(adapter, imgs.to(device)).to(dtype))
    return out


# ----------------------------------------------------------------------------------------------
# Flip check on the full protocol
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def cls_features(adapter, model, loader, device, dtype, flip: bool = False, amp: str | None = None) -> tuple:
    """Final-norm CLS features (the k-NN feature, not normalised) of every image of `loader`, in loader order:
    in the model's dtype, or under bf16 autocast on CUDA with amp="bf16" (returned as float32)."""
    import contextlib

    device = torch.device(device)
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if amp == "bf16" and device.type == "cuda"
           else contextlib.nullcontext())
    feats, labels = [], []
    for imgs, y in loader:
        x = normalize_for(adapter, imgs.to(device, non_blocking=True)).to(dtype)
        if flip:
            x = mirror(x)
        with ctx:
            f = adapter.cls_feature(x, model)
        # clone: cls_feature returns a view of the whole token tensor, and .to(dtype) of an fp32 tensor is no copy, so
        # without it every stored feature kept its batch's [B, N, d] tokens alive (out of memory on the 100k bank)
        feats.append((f.float() if amp == "bf16" else f.to(dtype)).clone())
        labels.append(torch.as_tensor(y))
    return torch.cat(feats), torch.cat(labels).to(device)


@torch.no_grad()
def flip_consistency_check(adapter, model, bank_loader, query_loader, num_classes: int, device, dtype,
                           k: int = 20, temperature: float = 0.07, amp: str | None = None) -> dict:
    """The k-NN flip check of mirror.heal_vexact.flip_check (cosine k-NN of the final-norm CLS feature against
    the bank, k = 20, T = 0.07; flip consistency = share of queries whose prediction for the image equals the
    prediction for its mirror; must be 100% for an exactly invariant feature), run on the given (full) bank and
    queries, with the raw feature differences added: max |f(mirror x) - f(x)| over every query and channel,
    the same on unit-normalised features, and the largest per-image ||f(mirror x) - f(x)|| / ||f(x)||.
    Precision: the model's dtype without autocast (amp None; fp32 with TF32 off), or bf16 autocast on CUDA."""
    from mirror.knn import knn_predict

    import torch.nn.functional as F

    bf, by = cls_features(adapter, model, bank_loader, device, dtype, amp=amp)
    qf, qy = cls_features(adapter, model, query_loader, device, dtype, amp=amp)
    qft, _ = cls_features(adapter, model, query_loader, device, dtype, flip=True, amp=amp)
    d = qft - qf
    raw_max, ref_max = float(d.abs().max()), float(qf.abs().max())
    rel_img = float((d.double().norm(dim=1) / qf.double().norm(dim=1).clamp_min(1e-300)).max())
    bn, qn, qtn = F.normalize(bf, dim=-1), F.normalize(qf, dim=-1), F.normalize(qft, dim=-1)
    pred = knn_predict(bn, by, qn, num_classes, k, temperature)
    pred_t = knn_predict(bn, by, qtn, num_classes, k, temperature)
    same = pred == pred_t
    fc = 100.0 * float(same.float().mean())
    on_cuda = torch.device(device).type == "cuda"
    precision = "bf16 autocast" if (amp == "bf16" and on_cuda) else str(dtype).replace("torch.", "")
    return {"precision": precision, "flip_consistency": fc, "pass": bool(fc == 100.0),
            "n_disagree": int((~same).sum()), "n_query": int(qf.shape[0]), "n_bank": int(bf.shape[0]),
            "top1": 100.0 * float((pred == qy).float().mean()),
            "max_abs_feature_diff": raw_max, "max_abs_feature_ref": ref_max,
            "max_abs_feature_diff_rel": raw_max / ref_max if ref_max > 0 else float("nan"),
            "max_rel_feature_diff_per_image": rel_img, "max_abs_unit_feature_diff": float((qtn - qn).abs().max()),
            "knn_k": int(k), "temperature": float(temperature),
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32), "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32)}


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2_vitb14_reg")
    p.add_argument("--hub-repo", default="facebookresearch/dinov2")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--operators", default=None, help="scripts/run_e1.py e1_operators.npz (default per model, as in "
                                                     "scripts/tier2_linear.py MODEL_DEFAULTS)")
    p.add_argument("--m1-structure", default=None, help="m1_structure.npz with A* for k = depth (default per model; "
                                                        "fitted here when absent)")
    p.add_argument("--span-structure", default=None,
                   help="mirror.mirror_family SpanStructure file (results/MUST-07/structure/<model>_s<seed>.npz): the "
                        "source of A* (k = depth), neuron pairings, head maps and register pairing, instead of "
                        "--m1-structure or a fit here")
    p.add_argument("--fit-seed", type=int, default=0, help="fit set for the structure when it is fitted here")
    p.add_argument("--vexact-run", default=None, help="scripts/heal_vexact.py run directory <out_dir>/k<K>")
    p.add_argument("--vexact-config", default="configs/m3_vexact.yaml",
                   help="fallback config when <vexact-run>/summary.json is missing")
    p.add_argument("--healed-run", default=None, help="healed run directory for mirror.heal_structured.load_healed_model")
    p.add_argument("--name", default=None, help="variant name for the output folder (default from the source)")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--n-images", type=int, default=1000, help="validation images (fixed subset, recorded by index)")
    p.add_argument("--batch-size", type=int, default=16, help="images per forward pass (run with their mirrors)")
    p.add_argument("--tol", type=float, default=DEFAULT_TOL, help="the MUST-02 prediction: every component below this")
    p.add_argument("--flip-queries", type=int, default=0,
                   help="validation queries for the flip check (0: all 50,000, the pre-registered setting)")
    p.add_argument("--flip-bank-per-class", type=int, default=0,
                   help="bank images per class for the flip check (0: the full 100k bank, the paper protocol)")
    p.add_argument("--no-flip-check", action="store_true", help="skip the k-NN flip check")
    p.add_argument("--no-bf16-flip", action="store_true",
                   help="skip the bf16 flip check reported beside the fp32 one (CUDA only; on CPU it equals fp32)")
    p.add_argument("--knn-batch-size", type=int, default=256, help="batch size of the flip check's feature passes")
    p.add_argument("--knn-k", type=int, default=20)
    p.add_argument("--knn-temperature", type=float, default=0.07)
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="output directory (default results/MUST-02/<model>/<variant>)")
    p.add_argument("--gpu-frac", type=float, default=DEFAULT_GPU_FRAC, help="per-process GPU memory cap (shared GPU)")
    p.add_argument("--device", default=None)
    p.add_argument("--tiny", action="store_true", help="tiny random ViT, synthetic data, CPU")
    a = p.parse_args(argv)
    if a.vexact_run and a.healed_run:
        p.error("--vexact-run and --healed-run are alternatives")
    if a.span_structure and (a.m1_structure or a.operators):
        p.error("--span-structure replaces --m1-structure and --operators")
    if a.span_structure and (a.vexact_run or a.healed_run):
        p.error("--span-structure is for the zero-shot model; healed runs carry their own structure")
    return a


def variant_name(a) -> str:
    if a.name:
        base = a.name
    elif a.vexact_run:
        run = Path(a.vexact_run)
        base = f"vexact_{run.parent.name}_{run.name}" if run.parent.name else f"vexact_{run.name}"
    elif a.healed_run:
        base = f"healed_{Path(a.healed_run).name}"
    else:
        base = "zeroshot"
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base)
    return base + ("_fp64" if a.dtype == "float64" else "")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)


def main(argv=None) -> dict:
    a = parse_args(argv)
    t_start = time.time()
    t2 = load_script("tier2_linear", a.gpu_frac)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() and not a.tiny else "cpu"))
    torch.backends.cuda.matmul.allow_tf32 = False      # exactness needs full fp32 matmuls
    torch.backends.cudnn.allow_tf32 = False            # (the patch embedding is a convolution)
    torch.set_float32_matmul_precision("highest")
    set_seed(a.seed)
    git = git_info()
    dtype = torch.float64 if a.dtype == "float64" else torch.float32
    name = variant_name(a)
    model_tag = "tiny" if a.tiny else a.model
    out = Path(a.out) if a.out else (REPO_ROOT / "tiny_out" / EXPERIMENT / name if a.tiny
                                     else REPO_ROOT / "results" / EXPERIMENT / model_tag / name)
    out.mkdir(parents=True, exist_ok=True)
    log(f"{EXPERIMENT} exactness audit: {model_tag}, {name}, {a.dtype}, out {out}, device {device}, "
        f"commit {git['git_hash'][:10]}{' (dirty)' if git.get('git_dirty') else ''}")

    t0 = time.time()
    if a.vexact_run:
        adapter, model, st = load_vexact_run(a, device)
        data = load_data(a, adapter, device)
    elif a.healed_run:
        adapter, model, st = load_healed_run(a, device)
        data = load_data(a, adapter, device)
    else:
        adapter = build_adapter(a, device)
        data = load_data(a, adapter, device)
        model, st = load_zeroshot(a, adapter, data["proto"], device, out, git, dtype)
    load_healed = a.vexact_run or a.healed_run
    if load_healed and dtype == torch.float64:
        log("note: the healed weights were projected and stored in float32; in float64 the audit measures the "
            "float32 rounding of the stored weights, not exact arithmetic")
    t2.apply_memory_cap(a.gpu_frac)
    adapter.model.to(dtype)
    model.to(dtype).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    t_build = time.time() - t0
    L = len(model.blocks)
    log(f"model: depth {L}, d {adapter.embed_dim}, A* d+ {st.d_plus}, converted blocks 1..{st.k}, registers "
        f"{len(st.reg_perm)} (pairing {st.reg_perm.tolist()}); structure: {st.source.get('kind')}")

    t0 = time.time()
    xs = image_batches(adapter, data["query"], device, dtype)
    n_img = int(sum(x.shape[0] for x in xs))
    audit = ExactnessAudit(adapter, model, st, device)
    rows = audit.run(xs).rows(a.tol)
    t_audit = time.time() - t0
    summ = summarise(rows, a.tol)
    log(f"audit on {n_img} images ({a.dtype}): {summ.get('n_components')} components inside the span; largest "
        f"relative error {summ.get('max_rel_tensor', float('nan')):.2e} (whole tensor, {summ.get('worst_component')}), "
        f"{summ.get('max_rel_token', float('nan')):.2e} (per token, {summ.get('worst_component_token')}); tol {a.tol:g}")

    t0 = time.time()
    flip = flip_bf16 = None
    if st.k == L and not a.no_flip_check:
        log(f"flip check ({a.dtype}, no autocast): {data['n_flip_query']} validation queries against a bank of "
            f"{data['n_bank']} train images")
        flip = flip_consistency_check(adapter, model, data["bank"], data["flip_query"], data["num_classes"], device,
                                      dtype, k=a.knn_k, temperature=a.knn_temperature)
        log(f"flip check ({flip['precision']}): k-NN flip consistency {flip['flip_consistency']:.3f}% "
            f"({flip['n_disagree']} of {flip['n_query']} differ); largest absolute CLS feature difference "
            f"{flip['max_abs_feature_diff']:.2e} (largest entry {flip['max_abs_feature_ref']:.2e}); "
            f"on unit features {flip['max_abs_unit_feature_diff']:.2e}")
        if not a.no_bf16_flip and device.type == "cuda" and dtype == torch.float32:
            flip_bf16 = flip_consistency_check(adapter, model, data["bank"], data["flip_query"], data["num_classes"],
                                               device, dtype, k=a.knn_k, temperature=a.knn_temperature, amp="bf16")
            log(f"flip check (bf16 autocast, reported beside fp32): {flip_bf16['flip_consistency']:.3f}% "
                f"({flip_bf16['n_disagree']} differ); largest absolute feature difference "
                f"{flip_bf16['max_abs_feature_diff']:.2e}")
    t_flip = time.time() - t0

    import pandas as pd

    df = pd.DataFrame(rows)
    for col in ("block", "code_index", "worst_image"):
        df[col] = df[col].astype("Int64")
    df.insert(0, "dtype", a.dtype)
    df.insert(0, "variant", name)
    df.insert(0, "model", model_tag)
    df["n_images_total"] = n_img
    df["git_hash"] = git["git_hash"]
    df["seed"] = a.seed
    df.to_csv(out / CSV_NAME, index=False)
    np.save(out / "val_indices.npy", np.asarray(data["idx"], dtype=np.int64))
    comps_ok = bool(summ.get("all_below_tol"))
    # The prediction needs both parts; without a flip check (--no-flip-check, partial span) it is not decided.
    holds = (comps_ok and bool(flip["pass"])) if flip is not None else (False if not comps_ok else None)
    word = {True: "HOLDS", False: "DOES NOT HOLD", None: "NOT DECIDED (components only)"}[holds]
    verdict = (f"{word}: largest relative error inside the span "
               f"{summ.get('max_rel_tensor', float('nan')):.2e} over the whole tensor of an image "
               f"({summ.get('worst_component')}), {summ.get('max_rel_token', float('nan')):.2e} per token "
               f"({summ.get('worst_component_token')}), tolerance {a.tol:g}, {a.dtype}; "
               + (f"k-NN flip consistency {flip['flip_consistency']:.3f}% ({flip['precision']}, {flip['n_query']} "
                  f"queries), largest absolute feature difference {flip['max_abs_feature_diff']:.2e}"
                  if flip is not None else "no flip check")
               + (f"; bf16 alongside: {flip_bf16['flip_consistency']:.3f}%" if flip_bf16 is not None else ""))
    log(verdict)
    metrics = {
        "experiment": EXPERIMENT, "model": model_tag, "variant": name, "dtype": a.dtype, "tol": a.tol,
        "prediction": "every component below tol in fp32 (whole-tensor relative error per image) and 100% k-NN "
                      "flip consistency in fp32 on all validation queries",
        "prediction_holds": holds, "verdict": verdict, "summary": summ,
        "per_token_note": "rel_token_max is the stricter per-token number; the prediction is judged on the "
                          "whole-tensor number (rel_tensor_max), the per-token verdict is reported beside it",
        "components_below_tol": comps_ok, "components_below_tol_per_token": bool(summ.get("all_below_tol_per_token")),
        "flip_check": flip, "flip_check_bf16": flip_bf16, "n_images": n_img, "images": data["description"],
        "val_indices": [int(i) for i in data["idx"]], "n_bank": data["n_bank"],
        "structure": {"k": int(st.k), "depth": L, "d_plus": st.d_plus, "d_minus": int(st.A.shape[0] - st.d_plus),
                      "reg_perm": [int(v) for v in st.reg_perm],
                      "n_neuron_pairs": [int((np.asarray(p) != np.arange(len(p))).sum() // 2) for p in st.perms],
                      "n_head_pairs": [int(hm.n_pairs) for hm in st.maps], "source": st.source},
        "notes": audit.notes, "git_hash": git["git_hash"], "seed": a.seed,
        "tf32": {"matmul": bool(torch.backends.cuda.matmul.allow_tf32), "cudnn": bool(torch.backends.cudnn.allow_tf32)},
        "timing_s": {"build": round(t_build, 1), "audit": round(t_audit, 1), "flip_check": round(t_flip, 1),
                     "total": round(time.time() - t_start, 1)},
    }
    (out / "metrics.json").write_text(json.dumps(_clean(metrics), indent=2), encoding="utf-8")
    write_meta(out, experiment=EXPERIMENT, config=vars(a), seeds={"seed": a.seed, "fit_seed": a.fit_seed},
               models={model_tag: {"hub_repo": None if a.tiny else a.hub_repo, "img_size": a.img_size,
                                   "variant": name, "structure": st.source}},
               device=device, wall_seconds=time.time() - t_start,
               extra={"n_images": n_img, "dtype": a.dtype, "csv": CSV_NAME, "prediction_holds": holds})
    log(f"wrote {out / CSV_NAME}, metrics.json and meta.json in {time.time() - t_start:.0f} s")
    return {"out": str(out), "rows": rows, "metrics": metrics}


if __name__ == "__main__":
    main()
