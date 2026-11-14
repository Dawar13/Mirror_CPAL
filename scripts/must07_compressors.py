#!/usr/bin/env python
"""MUST-07 (zero-shot comparison of 12 methods at matched parameters) and MUST-08 (matched component ablation of
Mirror-exact), PROJECT_NEXT_PHASE.md Sections 4 and 5 and PREREG_NEXT_PHASE.md.

    # structures first (two short GPU passes, then CPU matchings; can run beside other GPU work)
    python scripts/must07_compressors.py --model dinov2_vitb14_reg --fit-seeds 0,1,2 --structure-only --workers 16
    # MUST-07 on DINOv2-B-reg, fit seed 0, full model first (gate B), then the other sizes
    python scripts/must07_compressors.py --model dinov2_vitb14_reg --fit-seeds 0 --sizes full
    python scripts/must07_compressors.py --model dinov2_vitb14_reg --fit-seeds 0 --sizes 6mlp,6both,9both
    # fit seeds 1 and 2 at the two 6-block sizes (hyperparameters as chosen on seed 0's method-selection k-NN)
    python scripts/must07_compressors.py --model dinov2_vitb14_reg --fit-seeds 1,2 --sizes 6mlp,6both         --skip-flip-seeds 1,2
    # MUST-08 (same code path; rows already scored by MUST-07 are copied)
    python scripts/must07_compressors.py --experiment MUST-08 --model dinov2_vitb14_reg --fit-seeds 0,1,2 \
        --reuse-from results/MUST-07
    python scripts/must07_compressors.py --tiny --out tiny_out/must07      # tiny random ViT, synthetic data, CPU

Methods (MUST-07 numbering): 1 mirror_exact, 2 mirror_local (ablation), 3 svd, 4 svdllm, 5 asvd_scale, 6 slicegpt,
7 flap, 8 zipit, 9 pair_actcorr, 10 pair_wcos, 11 pair_random, 12 random_structure. Methods 1, 2 and 9 to 12 come
from mirror/mirror_family.py; methods 3 to 8 from mirror/baselines.py (compress, HPARAM_GRIDS). Until
mirror/baselines.py has METHODS_V2 and mirror/budget.py exists, a clearly marked SHIM below stands in (plain SVD,
SVD-LLM-style whitening, SliceGPT-style MLP slicing and activation-weighted MLP pruning built on the existing
mirror/baselines.py; attention by plain SVD). Remove the SHIM at integration.

Sizes: "<k><components>" with k a number of blocks from block 1 (code index 0) or "full" (every block), and
components "mlp", "attn" or "both" (default both): 6mlp, 6both, 9both, full; MUST-08 adds 6attn, fullmlp, fullattn.
"full" also projects the embeddings and installs the even head for the A*-type Mirror methods (the span starts at
block 1, so the embeddings belong to it at every size; the even head only at the full model).

Budgets (Section 4.7): every method is matched to Mirror-exact's parameter count over the same converted matrices,
basis changes counted (mirror_family.span_budget). A compressor grid point is eligible only if its count is at or
above the budget and within --budget-tol (1%) of it (rounding toward more parameters); otherwise it is logged and
skipped. Mirror-local is an ablation: its count, with one d x d rotation per block, is reported with its gap.

Data roles (Section 4.1): fit sets Protocol.fit_loader(seed) for every statistic and structure; k-NN bank
Protocol.bank_loader (100k train); method-selection queries Protocol.msel_loader (10k train, disjoint); validation
Protocol.query_loader (50k), for reporting only. Every choice (hyperparameter grid point, best baseline for gate B)
uses k-NN of the method-selection queries against the bank. Validation numbers are never read by a choice.

Evaluation per row (paper protocol 4.6: CLS after the final norm, cosine, k = 20, T = 0.07): one bank pass, shared by
the method-selection and validation queries; bf16 autocast features (comparable with earlier results) for top-1 and
flip_consistency; for every full-model row and the original (--fp32-flip full, default) also fp32 features of the bank
and of the validation images and their mirrors: flip_consistency_fp32 and the largest unit-feature difference. bf16
rounding breaks exact ties (the MUST-00 audit: 99.76% in bf16 against 100% in fp32 for an exactly invariant model),
so exact-invariance claims are scored on the fp32 value. --fp32-flip exact restricts the fp32 pass to rows that are
exactly invariant by construction (A*-type methods, both components, full model); --fp32-queries and
--fp32-bank-per-class make it use subsets (as M1's flip check: 5,000 queries, 10 bank images per class). Grid points that are not chosen are scored on the bank and
the method-selection queries only; the chosen point's model and bank features are kept, so its validation pass is the
only extra work. Images are read once per role into (pinned) host memory (--preload cpu) or GPU memory (gpu).

Per row: validation top-1 and flip consistency, method-selection top-1, per-image predictions (npz, for the paired
bootstrap), parameters (whole model; converted span with basis changes; gap to the budget), analytic multiply-adds per
image (matrices at their kept counts, d^2 per token per basis change, token mixing 2 T^2 D per block, patch
embedding; biases, norms and activations not counted), wall time, images per second and peak GPU memory.

Outputs (resumable: a rerun skips finished rows and grid points; one process per output directory at a time):
  <out>/rows_<model>.csv           one row per (model, size, method, fit seed), plus the original (fit seed -1)
  <out>/grid_<model>.csv           every compressor grid point with its method-selection top-1
  <out>/preds/<model>/...npz       per-image predictions; <out>/preds/<model>/labels.npz
  <out>/info/<model>/...json       full info of every row (per-matrix counts, structure provenance)
  <out>/bootstrap.csv              Mirror-exact against every other method, validation, fit seed 0, 10,000 resamples
  <out>/gate_b.json                MUST-07 only: ranking of methods 3 to 12 by method-selection k-NN at full size
  <out>/SUMMARY.md, <out>/meta.json (mirror/meta.py), <out>/meta/<model>_<time>.json (each invocation)
Structures (operators, pairings, head maps, register pairings) are fitted once per model and fit seed and cached in
--structure-dir (shared by MUST-07 and MUST-08).
"""
from __future__ import annotations

import sys

import torch


def _early_gpu_frac(argv, default: float = 0.3) -> float:
    for i, s in enumerate(argv):
        if s == "--gpu-frac" and i + 1 < len(argv):
            return float(argv[i + 1])
        if s.startswith("--gpu-frac="):
            return float(s.split("=", 1)[1])
    return default


if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(_early_gpu_frac(sys.argv[1:]))  # shared GPU: cap before allocating

import argparse  # noqa: E402
import copy  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror import variants as V  # noqa: E402
from mirror import mirror_family as MF  # noqa: E402
from mirror.bootstrap import paired_bootstrap  # noqa: E402
from mirror.data import mirror, normalize_for  # noqa: E402
from mirror.knn import knn_predict  # noqa: E402
from mirror.meta import write_meta  # noqa: E402
from mirror.utils import REPO_ROOT, git_info, log, set_seed  # noqa: E402

METHOD_ORDER = ("mirror_exact", "mirror_local", "svd", "svdllm", "asvd_scale", "slicegpt", "flap", "zipit",
                "pair_actcorr", "pair_wcos", "pair_random", "random_structure")
METHOD_NUMBER = {m: i + 1 for i, m in enumerate(METHOD_ORDER)}
COMPRESSORS = ("svd", "svdllm", "asvd_scale", "slicegpt", "flap", "zipit")
BASELINES = METHOD_ORDER[2:]          # PREREG_NEXT_PHASE.md MUST-07 operational: "every baseline" = methods 3 to 12
LABEL = {
    "original": "original model", "mirror_exact": "Mirror-exact", "mirror_local": "Mirror-local (ablation)",
    "svd": "plain truncated SVD", "svdllm": "SVD-LLM-style whitened SVD", "asvd_scale": "ASVD-style activation scaling",
    "slicegpt": "SliceGPT-style slicing", "flap": "FLAP-style pruning", "zipit": "ZipIt-style merging",
    "pair_actcorr": "pairing by same-image correlation", "pair_wcos": "pairing by weight cosine",
    "pair_random": "random pairing", "random_structure": "random structure",
}
KEY = ["model", "size", "method", "fit_seed"]
GRID_KEY = ["model", "size", "method", "fit_seed", "hparams"]
EXPERIMENTS = {
    "MUST-07": {"sizes": "6mlp,6both,9both,full", "tiny_sizes": "2mlp,2both,3both,full", "methods": "all"},
    "MUST-08": {"sizes": "6mlp,6attn,6both,fullmlp,fullattn,full",
                "tiny_sizes": "2mlp,2attn,2both,fullmlp,fullattn,full", "methods": "mirror_exact"},
}
SIX = {"MUST-07": ("6mlp", "6both"), "tiny": ("2mlp", "2both")}


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", default="MUST-07", choices=sorted(EXPERIMENTS))
    p.add_argument("--model", default="dinov2_vitb14_reg", help="registry name (mirror/adapters_timm.py)")
    p.add_argument("--sizes", default=None, help="comma list such as 6mlp,6both,9both,full (default per experiment)")
    p.add_argument("--methods", default=None,
                   help="comma list of methods, or 'all', 'mirror' (methods 1, 2, 9 to 12), 'compressors' (3 to 8)")
    p.add_argument("--fit-seeds", default=None, help="fit sets to use (default 0,1,2; tiny 0)")
    p.add_argument("--grid-seeds", default="0",
                   help="fit seeds whose compressor grids are searched on the method-selection set; other seeds reuse "
                        "the point chosen on the smallest of these (when that row exists, otherwise they search too)")
    p.add_argument("--random-seed", type=int, default=1,
                   help="random pairing and random structure use seed random_seed + fit_seed")
    p.add_argument("--skip-flip-seeds", default="", help="fit seeds whose rows skip the bf16 mirrored validation pass")
    p.add_argument("--fp32-flip", default="full", choices=["full", "exact", "none"],
                   help="fp32 flip-consistency pass: every full-model row and the original (full), only rows that claim "
                        "exact invariance (exact), or none")
    p.add_argument("--fp32-queries", type=int, default=None,
                   help="validation queries of the fp32 pass (default: all; n takes n/1000 per class, as M1's flip check)")
    p.add_argument("--fp32-bank-per-class", type=int, default=None,
                   help="bank images per class of the fp32 pass (default: the whole bank)")
    p.add_argument("--out", default=None, help="output directory (default results/<experiment>; tiny: .../tiny)")
    p.add_argument("--structure-dir", default=None,
                   help="fitted structures (default results/MUST-07/structure, shared with MUST-08; tiny: <out>/structure)")
    p.add_argument("--operators-template", default=None,
                   help="optional run_e1.py operator file per fit seed, e.g. runs/x2/perlayer_reg_s{seed}/results/"
                        "e1_operators.npz ({model} and {seed} are filled in): replaces the first structure pass")
    p.add_argument("--reuse-from", default=None, help="another output directory whose identical rows are copied")
    p.add_argument("--main-model", default=None, help="model of gate B and the predictions (default dinov2_vitb14_reg)")
    p.add_argument("--bootstrap-fit-seed", type=int, default=0)
    p.add_argument("--bootstrap-resamples", type=int, default=10_000)
    p.add_argument("--budget-tol", type=float, default=0.01, help="largest relative excess over the budget (4.7)")
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"))
    p.add_argument("--eval-batch", type=int, default=None, help="feature-extraction batch (default 512; tiny 32)")
    p.add_argument("--stats-batch-size", type=int, default=16, help="paired passes (images with their mirrors)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--preload", default="cpu", choices=["cpu", "gpu", "loader"],
                   help="keep each image role in pinned host memory (cpu), on the GPU (gpu), or re-read it (loader)")
    p.add_argument("--queries", type=int, default=None, help="subsample the validation queries (debugging only)")
    p.add_argument("--top-k", type=int, default=None,
                   help="matching candidates per neuron (default 10, as every earlier fit; tiny 3, for speed)")
    p.add_argument("--workers", type=int, default=None, help="processes for the neuron matchings (default 8; tiny 1)")
    p.add_argument("--register-margin", type=float, default=0.0)
    p.add_argument("--knn-k", type=int, default=20)
    p.add_argument("--knn-temperature", type=float, default=0.07)
    p.add_argument("--gpu-frac", type=float, default=0.3, help="per-process GPU memory cap (set at import)")
    p.add_argument("--device", default=None)
    p.add_argument("--tiny", action="store_true", help="tiny random ViT, synthetic data, CPU")
    p.add_argument("--shim", action="store_true", help="use the compressor SHIM even if mirror/baselines.py has METHODS_V2")
    p.add_argument("--force-structure", action="store_true", help="refit the structures")
    p.add_argument("--structure-only", action="store_true", help="fit (and cache) the structures, then stop")
    p.add_argument("--postprocess-only", action="store_true", help="only rewrite bootstrap, gate B and the summary")
    return p.parse_args(argv)


@dataclass(frozen=True)
class Size:
    name: str
    k: int
    components: tuple

    @property
    def comp_label(self) -> str:
        return "both" if len(self.components) == 2 else self.components[0]


def parse_size(name: str, depth: int) -> Size:
    m = re.fullmatch(r"(\d+|full)(mlp|attn|both)?", name.strip())
    if not m:
        raise ValueError(f"size {name!r}: use <k><mlp|attn|both> or full<mlp|attn|both>")
    k = depth if m.group(1) == "full" else int(m.group(1))
    if not 1 <= k <= depth:
        raise ValueError(f"size {name!r}: k = {k} outside 1..{depth}")
    comp = m.group(2) or "both"
    comps = {"mlp": ("mlp",), "attn": ("attn",), "both": ("mlp", "attn")}[comp]
    full = k == depth
    canon = ("full" if comp == "both" else f"full{comp}") if full else f"{k}{comp}"
    return Size(canon, k, comps)


def _ints(s) -> list:
    return [int(x) for x in str(s).split(",") if str(x).strip()]


# ----------------------------------------------------------------------------------------------
# ---- SHIM (remove at integration): compressors until mirror/baselines.py has METHODS_V2 ----------
# ----------------------------------------------------------------------------------------------
SHIM_METHODS = ("svd", "svdllm", "slicegpt", "flap")
SHIM_GRIDS = {"svd": [{}], "svdllm": [{"damp": 1e-6}, {"damp": 1e-3}], "slicegpt": [{}], "flap": [{}]}
SHIM_NEEDS = {"svd": (), "svdllm": ("M1", "M2"), "slicegpt": ("S",), "flap": ("abs_g",)}
_CANON = {"fc1": "fc1", "fc2": "fc2", "wq": "Wq", "q": "Wq", "wk": "Wk", "k": "Wk", "wv": "Wv", "v": "Wv",
          "wo": "Wo", "o": "Wo", "proj": "Wo"}


def _canon_matrix(name: str) -> str:
    key = str(name).split(".")[-1].lower()
    if key not in _CANON:
        raise ValueError(f"SHIM: cannot map budget matrix name {name!r}")
    return _CANON[key]


def shim_collect_stats(adapter, loader, device, blocks, methods, components):
    """SHIM of baselines.collect_stats: the existing collect_baseline_stats for what the shim methods need."""
    from mirror import baselines as B

    need = set()
    for m in methods:
        need.update(SHIM_NEEDS.get(m, ()))
    return B.collect_baseline_stats(adapter, loader, device, list(blocks), sorted(need or {"abs_g"}))


@torch.no_grad()
def shim_compress(model, method, blocks, components, budget, stats, hparams=None):
    """SHIM of baselines.compress. Per converted matrix the target is Mirror-exact's count plus a pro-rata share of
    its basis change; every allocation rounds up (toward more parameters, Section 4.7). MLP: svd (plain rank
    truncation), svdllm (baselines.asvd_truncate, ridge `damp`), slicegpt (baselines slicegpt, MLP input and output
    slicing), flap (baselines prune_act). Attention (when converted): plain SVD for every method. No basis changes."""
    from mirror import baselines as B
    from mirror.m1 import apply_attention_svd
    from mirror.surgery import svd_truncate, write_absorbed_mlp

    if method not in SHIM_METHODS:
        raise ValueError(f"SHIM has no method {method!r}")
    hp = dict(hparams or {})
    comps = MF.normalise_components(components)
    blocks = [int(b) for b in blocks]
    mats = list(budget.matrices)
    me = sum(int(m.mirror_exact) for m in mats)
    extra = int(budget.total) - me
    target = {(int(m.block), _canon_matrix(m.name)): int(m.mirror_exact) * (1.0 + extra / me) for m in mats}
    out = copy.deepcopy(model)
    blk0 = out.blocks[0]
    d, h = int(blk0.mlp.fc1.in_features), int(blk0.mlp.fc1.out_features)
    D = int(blk0.attn.qkv.out_features // 3)
    rows, unit = [], 0
    for b in blocks:
        blk = out.blocks[b]
        if "mlp" in comps:
            t1, t2 = target[(b, "fc1")], target[(b, "fc2")]
            if method in ("svd", "svdllm"):
                r1, r2 = (min(math.ceil(t / (h + d)), h, d) for t in (t1, t2))
                f = B.fold_mlp(blk)
                if method == "svd":
                    W1n, W2n = svd_truncate(f["W1"], r1), svd_truncate(f["W2"], r2)
                else:
                    damp = float(hp.get("damp", B.DEFAULT_DAMP))
                    W1n, _ = B.asvd_truncate(f["W1"], stats.M1[b].to(f["W1"].device), r1, damp)
                    W2n, _ = B.asvd_truncate(f["W2"], stats.M2[b].to(f["W1"].device), r2, damp)
                write_absorbed_mlp(blk, W1n, f["b1"], W2n, f["b2"])
                k1, k2 = r1 * (h + d), r2 * (h + d)
                unit += 2 * (h + d)
            elif method == "slicegpt":
                d_in, d_out = min(math.ceil(t1 / h), d), min(math.ceil(t2 / h), d)
                B.apply_baseline(out, "slicegpt", [b], [(d_in, d_out)], stats)
                k1, k2 = h * d_in, h * d_out
                unit += 2 * h
            else:
                n = min(math.ceil((t1 + t2) / (2 * d)), h)
                B.apply_baseline(out, "prune_act", [b], [n], stats)
                k1 = k2 = n * d
                unit += 2 * d
            rows += [{"block": b, "name": "fc1", "shape": [h, d], "dense": h * d, "kept": int(k1)},
                     {"block": b, "name": "fc2", "shape": [d, h], "dense": h * d, "kept": int(k2)}]
        if "attn" in comps:
            ranks = tuple(min(math.ceil(target[(b, n)] / (D + d)), D, d) for n in MF.ATTN_MATRICES)
            apply_attention_svd(out, [b], [ranks])
            for n, r in zip(MF.ATTN_MATRICES, ranks):
                rows.append({"block": b, "name": n, "shape": [D, d] if n != "Wo" else [d, D], "dense": D * d,
                             "kept": int(r * (D + d))})
            unit += 4 * (D + d)
    pm = int(sum(r["kept"] for r in rows))
    total = int(budget.total)
    out.eval()
    return out, {"method": method, "hparams": hp, "params_matrices": pm, "params_basis_change": 0, "params_total": pm,
                 "budget_total": total, "gap": pm - total, "gap_rel": (pm - total) / total, "per_matrix": rows,
                 "tensors": {}, "granularity": int(unit), "shim": True,
                 "note": "SHIM (scripts/must07_compressors.py); attention by plain SVD; no basis changes"}
# ---- end SHIM ------------------------------------------------------------------------------------


@dataclass
class CompressorAPI:
    kind: str
    methods: tuple
    grids: dict
    collect_stats: object
    compress: object


def compressor_api(force_shim: bool = False) -> CompressorAPI:
    """The real methods of mirror/baselines.py (contract: METHODS_V2, HPARAM_GRIDS, collect_stats, compress) when
    they and mirror/budget.py exist; otherwise the SHIM."""
    if not force_shim:
        try:
            from mirror import baselines as B
            from mirror import budget as _budget  # noqa: F401
            if all(hasattr(B, n) for n in ("METHODS_V2", "HPARAM_GRIDS", "collect_stats", "compress")):
                return CompressorAPI("real", tuple(B.METHODS_V2), dict(B.HPARAM_GRIDS), B.collect_stats, B.compress)
        except ImportError:
            pass
    return CompressorAPI("shim", SHIM_METHODS, SHIM_GRIDS, shim_collect_stats, shim_compress)


# ----------------------------------------------------------------------------------------------
# Images and features
# ----------------------------------------------------------------------------------------------
class ImageSet:
    """One data role read once from its loader: uint8 images [N, 3, H, W] and labels, kept in (pinned) host memory,
    on the GPU, or re-read from the loader each time (store 'loader')."""

    def __init__(self, make_loader, store: str, device: torch.device, name: str):
        self.name, self.store, self.device, self.make_loader = name, store, device, make_loader
        self.x = None
        if store == "loader":
            ys = [torch.as_tensor(y) for _, y in make_loader()]
            self.y = torch.cat(ys).long()
            self.n = int(len(self.y))
            return
        loader = make_loader()
        n = len(loader.dataset)
        x, ys, s = None, [], 0
        for imgs, y in loader:
            if x is None:
                x = torch.empty((n,) + tuple(imgs.shape[1:]), dtype=imgs.dtype)
            x[s:s + len(imgs)] = imgs
            s += len(imgs)
            ys.append(torch.as_tensor(y))
        self.y = torch.cat(ys).long()
        self.n = int(len(self.y))
        if store == "gpu" and device.type == "cuda":
            x = x.to(device)
        elif device.type == "cuda":
            try:
                x = x.pin_memory()
            except RuntimeError:
                pass
        self.x = x

    def batches(self, batch: int):
        if self.x is None:
            for imgs, _ in self.make_loader():
                yield imgs.to(self.device, non_blocking=True)
            return
        for s in range(0, self.n, batch):
            yield self.x[s:s + batch].to(self.device, non_blocking=True)


class Evaluator:
    """k-NN evaluation with one bank pass per model (paper protocol, mirror/knn.py semantics)."""

    def __init__(self, adapter, proto, device, amp, batch: int, store: str, num_classes: int, k: int, temperature: float,
                 queries=None):
        self.adapter, self.proto, self.device, self.amp = adapter, proto, device, amp
        self.batch, self.store, self.num_classes, self.k, self.T = int(batch), store, int(num_classes), int(k), float(temperature)
        self.queries = queries
        self.fp32_queries, self.fp32_bank_per_class = None, None
        self._sets = {}
        self.images_done = 0

    def _bank_subset(self):
        from mirror.data import make_loader, make_splits

        p = self.proto
        rows = np.asarray(p.index["bank_rows"])
        sub = make_splits(np.asarray(p.train_labels)[rows], {"b": int(self.fp32_bank_per_class)})["b"]
        return make_loader(p.train, p.train_labels, rows[sub], p.bs, p.nw, p.pin)

    def set(self, role: str) -> ImageSet:
        if role not in self._sets:
            make = {"bank": self.proto.bank_loader, "msel": self.proto.msel_loader,
                    "val": lambda: self.proto.query_loader(self.queries),
                    "bank_fp32": self._bank_subset,
                    "val_fp32": lambda: self.proto.query_loader(self.fp32_queries)}[role]
            t = time.time()
            self._sets[role] = ImageSet(make, self.store, self.device, role)
            log(f"{role}: {self._sets[role].n} images read in {time.time() - t:.1f} s ({self.store})")
        return self._sets[role]

    def _ctx(self, amp):
        import contextlib

        if amp == "bf16" and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    @torch.no_grad()
    def feats(self, model, role: str, flip_too: bool = False, amp="default"):
        amp = self.amp if amp == "default" else amp
        iset = self.set(role)
        f, ft = [], []
        for xb in iset.batches(self.batch):
            x = normalize_for(self.adapter, xb)
            with self._ctx(amp):
                f.append(F.normalize(self.adapter.cls_feature(x, model).float(), dim=-1))
                if flip_too:
                    ft.append(F.normalize(self.adapter.cls_feature(mirror(x), model).float(), dim=-1))
            self.images_done += x.shape[0] * (2 if flip_too else 1)
        y = iset.y.to(self.device)
        return torch.cat(f), (torch.cat(ft) if flip_too else None), y

    def predict(self, bank_f, bank_y, q):
        return knn_predict(bank_f, bank_y, q, self.num_classes, self.k, self.T)

    def bank_msel(self, model) -> dict:
        bf, _, by = self.feats(model, "bank")
        mf, _, my = self.feats(model, "msel")
        pred = self.predict(bf, by, mf)
        return {"bank_f": bf, "bank_y": by, "msel_pred": pred.cpu(), "msel_y": my.cpu(),
                "msel_top1": 100.0 * float((pred == my).float().mean()), "n_bank": int(bf.shape[0]),
                "n_msel": int(mf.shape[0])}

    def val(self, model, bm: dict, flip: bool = True) -> dict:
        qf, qft, qy = self.feats(model, "val", flip_too=flip)
        pred = self.predict(bm["bank_f"], bm["bank_y"], qf)
        out = {"val_pred": pred.cpu(), "val_y": qy.cpu(), "top1": 100.0 * float((pred == qy).float().mean()),
               "n_val": int(qf.shape[0]), "flip_consistency": float("nan"), "max_abs_unit_feature_diff": float("nan"),
               "val_pred_flip": None}
        if flip:
            pt = self.predict(bm["bank_f"], bm["bank_y"], qft)
            out.update({"val_pred_flip": pt.cpu(), "flip_consistency": 100.0 * float((pred == pt).float().mean()),
                        "max_abs_unit_feature_diff": float((qft - qf).abs().max())})
        return out

    def flip_fp32(self, model) -> dict:
        """fp32 features (no autocast; TF32 is off): bank, validation and mirrored validation (whole sets unless
        --fp32-bank-per-class / --fp32-queries ask for subsets)."""
        bank = "bank" if self.fp32_bank_per_class is None else "bank_fp32"
        val = "val" if self.fp32_queries is None else "val_fp32"
        bf, _, by = self.feats(model, bank, amp=None)
        qf, qft, qy = self.feats(model, val, flip_too=True, amp=None)
        p, pt = self.predict(bf, by, qf), self.predict(bf, by, qft)
        return {"flip_consistency_fp32": 100.0 * float((p == pt).float().mean()),
                "top1_fp32": 100.0 * float((p == qy).float().mean()),
                "max_abs_unit_feature_diff_fp32": float((qft - qf).abs().max()),
                "n_bank_fp32": int(bf.shape[0]), "n_val_fp32": int(qf.shape[0]),
                "val_pred_fp32": p.cpu(), "val_pred_flip_fp32": pt.cpu()}


# ----------------------------------------------------------------------------------------------
# Costs
# ----------------------------------------------------------------------------------------------
def inventory(adapter) -> dict:
    m = adapter.model
    T = int(adapter.n_tokens)
    H, W = adapter.grid_hw
    D = [int(blk.attn.qkv.out_features // 3) for blk in m.blocks]
    dense_all = sum(int(blk.attn.qkv.weight.numel() + blk.attn.proj.weight.numel() + blk.mlp.fc1.weight.numel()
                        + blk.mlp.fc2.weight.numel()) for blk in m.blocks)
    patch = int(H * W * m.patch_embed.proj.weight.numel())
    macs = patch + T * dense_all + sum(2 * T * T * Db for Db in D)
    return {"n_model": int(sum(p.numel() for p in m.parameters())), "dense_all": int(dense_all), "patch_macs": patch,
            "T": T, "D": D, "macs": int(macs)}


def _attn_width(blk, default: int) -> int:
    qkv = getattr(getattr(blk, "attn", None), "qkv", None)
    return int(qkv.out_features // 3) if hasattr(qkv, "out_features") else int(default)


def _realised_width(entry: dict | None, default: int, dh: int) -> int:
    """Query or value width of one block from a compressor's per_matrix entry: merged groups (ZipIt), kept rows or
    kept heads (FLAP, heads zeroed in dense form); otherwise the stored width."""
    if not entry:
        return int(default)
    if entry.get("groups") is not None:
        return int(len(entry["groups"]))
    if entry.get("n") is not None:
        return int(entry["n"])
    if entry.get("heads") is not None:
        return int(entry["heads"]) * int(dh)
    return int(default)


def mixing_macs(inv: dict, model, per_matrix=None) -> int:
    """Token mixing per image, T^2 (n_q + n_v) per block (q k^T and attention times v), with the realised query and
    value widths (dense: n_q = n_v = D)."""
    T = inv["T"]
    by = {}
    for e in per_matrix or []:
        name = str(e.get("name", "")).lower()
        if name in ("wq", "wv"):
            by[(int(e["block"]), name)] = e
    total = 0
    for i, blk in enumerate(model.blocks):
        D = _attn_width(blk, inv["D"][i])
        H = int(getattr(getattr(blk, "attn", None), "num_heads", 1) or 1)
        dh = max(D // H, 1)
        total += T * T * (_realised_width(by.get((i, "wq")), D, dh) + _realised_width(by.get((i, "wv")), D, dh))
    return int(total)


def variant_costs(inv: dict, model, params_matrices: int, span_dense: int, basis: int, per_matrix=None) -> dict:
    """Whole-model parameters and analytic multiply-adds per image (see the module docstring)."""
    T = inv["T"]
    mats = inv["dense_all"] - int(span_dense) + int(params_matrices)
    macs = inv["patch_macs"] + T * mats + mixing_macs(inv, model, per_matrix) + T * int(basis)
    params = inv["n_model"] - int(span_dense) + int(params_matrices) + int(basis)
    return {"model_params": int(params), "model_params_frac": params / inv["n_model"], "macs": int(macs),
            "gflops": 2.0 * macs / 1e9, "gflops_frac": macs / inv["macs"]}


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
def _free(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _peak(device):
    return round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 3) if device.type == "cuda" else None


def _reset_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _i16(t):
    return None if t is None else np.asarray(t, dtype=np.int16)


def _jsonable(o):
    from mirror.meta import _jsonable as j

    return j(o)


def preds_path(out: Path, model: str, size: str, method: str, seed: int) -> Path:
    return out / "preds" / model / size / f"{method}_s{seed}.npz"


def info_path(out: Path, model: str, size: str, method: str, seed: int) -> Path:
    return out / "info" / model / size / f"{method}_s{seed}.json"


def save_preds(path: Path, ev: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: _i16(ev.get(k)) for k in ("val_pred", "val_pred_flip", "msel_pred", "val_pred_fp32", "val_pred_flip_fp32")
              if ev.get(k) is not None}
    np.savez_compressed(path, **arrays)
    return sha1(path)


def save_labels(out: Path, model: str, val_y, msel_y) -> None:
    path = out / "preds" / model / "labels.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, val=np.asarray(val_y, dtype=np.int16), msel=np.asarray(msel_y, dtype=np.int16))


def needs_fp32(mode: str, method: str, size: Size, depth: int) -> bool:
    full = size.k == depth
    if mode == "none" or not full:
        return False
    if mode == "full":
        return True
    return method in MF.EXACT_FAMILY and len(size.components) == 2      # exact invariance claimed


def claims_exact(method: str, size: Size, depth: int) -> bool:
    return size.k == depth and method in MF.EXACT_FAMILY and len(size.components) == 2


# ----------------------------------------------------------------------------------------------
# Structure
# ----------------------------------------------------------------------------------------------
def get_structure(a, adapter, proto, device, model_name: str, seed: int, sdir: Path):
    path = sdir / f"{model_name}_s{seed}.npz"
    if path.exists() and not a.force_structure:
        st = MF.load_span_structure(path)
        want = {"top_k": a.top_k, "register_margin": a.register_margin}
        got = {k: st.meta.get(k) for k in want}
        if st.model != model_name or st.fit_seed != seed or any(float(got[k]) != float(v) for k, v in want.items()):
            raise SystemExit(f"{path} was fitted with model {st.model}, seed {st.fit_seed}, {got}; this run has "
                             f"{model_name}, {seed}, {want}: use --force-structure or another --structure-dir")
        return st, path
    base = None
    if a.operators_template:
        op = Path(a.operators_template.format(model=model_name, seed=seed))
        op = op if op.is_absolute() else REPO_ROOT / op
        if op.exists():
            base = V.load_fit_run(op)
            if base.meta.get("fit_seed") is not None and int(base.meta["fit_seed"]) != seed:
                raise SystemExit(f"{op} is for fit seed {base.meta['fit_seed']}, not {seed}")
            log(f"structure pass 1 from {op}")
    t = time.time()
    log(f"fitting the structure of {model_name}, fit seed {seed} (fit set: train role fit{seed})")
    st = MF.fit_span_structure(adapter, proto.fit_loader(seed), device, model_name=model_name, fit_seed=seed,
                               top_k=a.top_k, workers=a.workers, register_margin=a.register_margin, base=base)
    MF.save_span_structure(path, st)
    log(f"structure saved to {path} ({time.time() - t:.1f} s); pairs per block {st.meta['n_pairs']}")
    _free(device)
    return MF.load_span_structure(path), path        # always use the saved copy (identical on resume)


# ----------------------------------------------------------------------------------------------
# Rows
# ----------------------------------------------------------------------------------------------
def base_row(model_name, size: Size, method, seed, git, a) -> dict:
    return {"model": model_name, "size": size.name, "k": size.k, "components": "+".join(size.components),
            "method": method, "number": METHOD_NUMBER.get(method, 0), "fit_seed": seed,
            "random_seed": (a.random_seed + seed) if method in ("pair_random", "random_structure") else "",
            "git_hash": git["git_hash"], "git_dirty": git["git_dirty"]}


def fill_eval(row: dict, bm: dict, vv: dict, fp: dict | None) -> None:
    row.update({"val_top1": vv["top1"], "flip_consistency": vv["flip_consistency"],
                "max_abs_unit_feature_diff": vv["max_abs_unit_feature_diff"], "msel_top1": bm["msel_top1"],
                "n_bank": bm["n_bank"], "n_msel": bm["n_msel"], "n_val": vv["n_val"]})
    if fp is not None:
        row.update({k: fp[k] for k in ("flip_consistency_fp32", "top1_fp32", "max_abs_unit_feature_diff_fp32",
                                       "n_bank_fp32", "n_val_fp32")})
    else:
        row.update({"flip_consistency_fp32": float("nan"), "top1_fp32": float("nan"),
                    "max_abs_unit_feature_diff_fp32": float("nan"), "n_bank_fp32": 0, "n_val_fp32": 0})


def fill_params(row: dict, info: dict, bf: dict, inv: dict, model, tol: float, tiny: bool, matched: bool) -> None:
    pt, bt = int(info["params_total"]), int(bf["budget_total"])
    gap = pt - bt
    gap_rel = gap / bt if bt else float("nan")
    row.update({"params_matrices": int(info["params_matrices"]), "params_basis_change": int(info["params_basis_change"]),
                "params_total": pt, "params_total_absorbed": int(info.get("params_total_absorbed", pt)),
                "budget_total": bt, "budget_total_reported": int(info.get("budget_total", bt)), "gap": gap,
                "gap_rel": gap_rel, "span_dense": int(bf["span_dense"]),
                "span_kept_frac": pt / bf["span_dense"] if bf["span_dense"] else float("nan"),
                "within_budget": (bool(-1e-12 <= gap_rel <= tol) if matched else "n/a (ablation)")})
    row.update(variant_costs(inv, model, info["params_matrices"], bf["span_dense"], info["params_basis_change"],
                             info.get("per_matrix")))


def budget_ok(info: dict, bf: dict, tol: float, tiny: bool) -> tuple:
    """At or above the budget, and within tol of it; or, when the method reports that its smallest step is larger
    than 1% of the budget (gap_reachable False), within one step (its gap_ok)."""
    bt = int(bf["budget_total"])
    gap_rel = (int(info["params_total"]) - bt) / bt
    if gap_rel < -1e-12:
        return False, f"below the budget ({100 * gap_rel:.3f}%)"
    one_step = bool(info.get("gap_ok")) and not bool(info.get("gap_reachable", True))
    if not tiny and gap_rel > tol and not one_step:
        return False, f"more than {100 * tol:.1f}% above the budget ({100 * gap_rel:.3f}%)"
    return True, ""


def _strip_arrays(o):
    """Replace tensors and arrays by their shape and dtype (info dicts carry factors, bases and structures)."""
    if isinstance(o, torch.Tensor) or (isinstance(o, np.ndarray) and o.size > 64):
        return {"array": list(o.shape), "dtype": str(o.dtype)}
    if isinstance(o, dict):
        return {k: _strip_arrays(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_strip_arrays(v) for v in o]
    return o


def write_info(out: Path, model_name, size: Size, method, seed, info: dict, extra: dict) -> None:
    path = info_path(out, model_name, size.name, method, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = {k: v for k, v in info.items() if k not in ("tensors", "structure")}
    path.write_text(json.dumps(_jsonable(_strip_arrays({**keep, **extra})), indent=1), encoding="utf-8")


def try_reuse(a, out: Path, rows, model_name, size: Size, method, seed, struct_sha, depth: int) -> bool:
    if not a.reuse_from:
        return False
    src = Path(a.reuse_from)
    src = src if src.is_absolute() else REPO_ROOT / src
    csv = src / f"rows_{model_name}.csv"
    if not csv.exists():
        return False
    import pandas as pd

    df = pd.read_csv(csv)
    hit = df[(df["model"] == model_name) & (df["size"] == size.name) & (df["method"] == method)
             & (df["fit_seed"].astype(int) == seed) & (df["status"] == "ok")]
    if hit.empty or str(hit.iloc[0].get("structure_sha", "")) != struct_sha:
        return False
    r = hit.iloc[0].to_dict()
    has_fp32 = float(r.get("n_val_fp32", 0) or 0) > 0
    if needs_fp32(a.fp32_flip, method, size, depth) and not has_fp32:
        return False                                  # this run needs the fp32 flip pass the source row lacks
    pp = preds_path(src, model_name, size.name, method, seed)
    if not pp.exists():
        return False
    dst = preds_path(out, model_name, size.name, method, seed)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pp, dst)
    ip = info_path(src, model_name, size.name, method, seed)
    if ip.exists():
        di = info_path(out, model_name, size.name, method, seed)
        di.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ip, di)
    r["reused_from"] = str(csv)
    rows.add(r)
    log(f"{model_name} {size.name} {method} seed {seed}: copied from {csv}")
    return True


def run_mirror(a, ctx, size: Size, method, seed, struct, budget, bf) -> dict:
    ev, device, inv = ctx["ev"], ctx["device"], ctx["inv"]
    t0 = time.time()
    _reset_peak(device)
    rseed = a.random_seed + seed
    model, info = MF.build_mirror_variant(ctx["adapter"], method, range(size.k), size.components, struct, seed=rseed,
                                          budget=budget)
    t_build = time.time() - t0
    n0 = ev.images_done
    te = time.time()
    bm = ev.bank_msel(model)
    vv = ev.val(model, bm, flip=seed not in ctx["skip_flip"])
    fp = ev.flip_fp32(model) if needs_fp32(a.fp32_flip, method, size, ctx["depth"]) else None
    t_eval = time.time() - te
    row = base_row(ctx["model_name"], size, method, seed, ctx["git"], a)
    fill_eval(row, bm, vv, fp)
    fill_params(row, info, bf, inv, model, a.budget_tol, a.tiny, matched=(method != "mirror_local"))
    sha = save_preds(preds_path(ctx["out"], ctx["model_name"], size.name, method, seed), {**vv, **bm, **(fp or {})})
    save_labels(ctx["out"], ctx["model_name"], vv["val_y"], bm["msel_y"])
    row.update({"hparams": "{}", "selected_by": "none (no hyperparameters)", "n_grid": 0,
                "claims_exact_invariance": claims_exact(method, size, ctx["depth"]),
                "status": "ok", "preds_sha": sha, "structure_sha": ctx["struct_sha"], "reused_from": "",
                "build_s": round(t_build, 2), "eval_s": round(t_eval, 2), "time_s": round(time.time() - t0, 2),
                "img_per_s": round((ev.images_done - n0) / max(t_eval, 1e-9), 1), "peak_mem_gb": _peak(device)})
    write_info(ctx["out"], ctx["model_name"], size, method, seed, info, {"row": row, "budget": bf})
    del model, bm
    _free(device)
    return row


def chosen_hparams(rows, model_name, size: Size, method, grid_seed: int):
    for r in rows.rows:
        if (r["model"] == model_name and r["size"] == size.name and r["method"] == method
                and int(r["fit_seed"]) == grid_seed and r.get("status") == "ok"):
            return json.loads(r["hparams"])
    return None


def run_compressor(a, ctx, size: Size, method, seed, budget, bf, stats) -> dict:
    ev, device, inv, api, gridcsv = ctx["ev"], ctx["device"], ctx["inv"], ctx["api"], ctx["grid"]
    model_name = ctx["model_name"]
    t0 = time.time()
    _reset_peak(device)
    grid = list(api.grids.get(method, [{}])) or [{}]
    selected_by = "method-selection k-NN over the grid (ties: first in grid order)"
    grid_seeds = _ints(a.grid_seeds)
    if seed not in grid_seeds and grid_seeds:
        g0 = min(grid_seeds)
        hp0 = chosen_hparams(ctx["rows"], model_name, size, method, g0)
        if hp0 is not None:
            grid = [hp0]
            selected_by = f"reused: chosen on fit seed {g0} by method-selection k-NN"
    blocks = list(range(size.k))
    scores, best = {}, None
    n0 = ev.images_done
    t_eval = 0.0
    for hp in grid:
        hj = json.dumps(hp, sort_keys=True)
        gkey = (model_name, size.name, method, seed, hj)
        if gridcsv.done(gkey):
            done_ok = [r for r in gridcsv.rows if V._norm_key(r["hparams"]) == hj and r["model"] == model_name
                       and r["size"] == size.name and r["method"] == method and int(r["fit_seed"]) == seed
                       and r["status"] == "ok"]
            if done_ok:
                scores[hj] = float(done_ok[-1]["msel_top1"])
                continue                              # a logged budget miss is tried again (the method may be fixed)
        tb = time.time()
        model, info = api.compress(ctx["adapter"].model, method, blocks, size.components, budget, stats, hp)
        tb = time.time() - tb
        ok, why = budget_ok(info, bf, a.budget_tol, a.tiny)
        grow = {"model": model_name, "size": size.name, "method": method, "fit_seed": seed, "hparams": hj,
                "params_total": int(info["params_total"]), "budget_total": bf["budget_total"],
                "gap_rel": (int(info["params_total"]) - bf["budget_total"]) / bf["budget_total"],
                "git_hash": ctx["git"]["git_hash"], "build_s": round(tb, 2)}
        if not ok:
            gridcsv.add({**grow, "status": f"budget_miss: {why}", "msel_top1": float("nan"), "eval_s": 0.0})
            log(f"{model_name} {size.name} {method} {hj}: skipped, {why}")
            del model
            continue
        te = time.time()
        bm = ev.bank_msel(model)
        t_eval += time.time() - te
        gridcsv.add({**grow, "status": "ok", "msel_top1": bm["msel_top1"], "eval_s": round(time.time() - te, 2)})
        scores[hj] = bm["msel_top1"]
        log(f"{model_name} {size.name} {method} seed {seed} {hj}: method-selection top-1 {bm['msel_top1']:.2f}")
        if best is None or bm["msel_top1"] > best[2]["msel_top1"]:
            best = (hj, (model, info), bm)
        else:
            del model, bm
        _free(device)
    row = base_row(model_name, size, method, seed, ctx["git"], a)
    order = [json.dumps(hp, sort_keys=True) for hp in grid]
    eligible = [hj for hj in order if hj in scores]
    if not eligible:
        row.update({"status": "budget_miss", "hparams": "", "selected_by": selected_by, "n_grid": len(grid),
                    "time_s": round(time.time() - t0, 2)})
        return row
    top = max(scores[hj] for hj in eligible)
    choice = next(hj for hj in eligible if scores[hj] == top)
    if best is None or best[0] != choice:          # resumed after the grid: rebuild the chosen point
        model, info = api.compress(ctx["adapter"].model, method, blocks, size.components, budget, stats, json.loads(choice))
        te = time.time()
        bm = ev.bank_msel(model)
        t_eval += time.time() - te
    else:
        model, info = best[1]
        bm = best[2]
    te = time.time()
    vv = ev.val(model, bm, flip=seed not in ctx["skip_flip"])
    fp = ev.flip_fp32(model) if needs_fp32(a.fp32_flip, method, size, ctx["depth"]) else None
    t_eval += time.time() - te
    fill_eval(row, bm, vv, fp)
    fill_params(row, info, bf, inv, model, a.budget_tol, a.tiny, matched=True)
    sha = save_preds(preds_path(ctx["out"], model_name, size.name, method, seed), {**vv, **bm, **(fp or {})})
    save_labels(ctx["out"], model_name, vv["val_y"], bm["msel_y"])
    row.update({"hparams": choice, "hparams_resolved": json.dumps(_jsonable(info.get("hparams", {})), sort_keys=True),
                "gap_note": info.get("gap_note", ""), "selected_by": selected_by, "n_grid": len(grid),
                "claims_exact_invariance": False,
                "status": "ok", "preds_sha": sha, "structure_sha": ctx["struct_sha"], "reused_from": "",
                "compressor_api": api.kind, "build_s": float("nan"), "eval_s": round(t_eval, 2),
                "time_s": round(time.time() - t0, 2),
                "img_per_s": round((ev.images_done - n0) / max(t_eval, 1e-9), 1), "peak_mem_gb": _peak(device)})
    write_info(ctx["out"], model_name, size, method, seed, info, {"row": row, "budget": bf, "grid_scores": scores})
    del model, bm, best
    _free(device)
    return row


def run_original(a, ctx) -> None:
    rows, ev, inv, device = ctx["rows"], ctx["ev"], ctx["inv"], ctx["device"]
    if rows.done((ctx["model_name"], "original", "original", -1)):
        return
    t0 = time.time()
    _reset_peak(device)
    m = ctx["adapter"].model
    bm = ev.bank_msel(m)
    vv = ev.val(m, bm, flip=True)
    fp = ev.flip_fp32(m) if a.fp32_flip == "full" else None
    row = {"model": ctx["model_name"], "size": "original", "k": 0, "components": "", "method": "original", "number": 0,
           "fit_seed": -1, "random_seed": "", "git_hash": ctx["git"]["git_hash"], "git_dirty": ctx["git"]["git_dirty"]}
    fill_eval(row, bm, vv, fp)
    row.update({"params_matrices": inv["dense_all"], "params_basis_change": 0, "params_total": inv["dense_all"],
                "model_params": inv["n_model"], "model_params_frac": 1.0, "macs": inv["macs"],
                "gflops": 2.0 * inv["macs"] / 1e9, "gflops_frac": 1.0, "hparams": "{}", "status": "ok",
                "selected_by": "", "n_grid": 0, "claims_exact_invariance": False,
                "preds_sha": save_preds(preds_path(ctx["out"], ctx["model_name"], "original", "original", -1),
                                        {**vv, **bm, **(fp or {})}),
                "structure_sha": "", "reused_from": "", "time_s": round(time.time() - t0, 2), "peak_mem_gb": _peak(device)})
    save_labels(ctx["out"], ctx["model_name"], vv["val_y"], bm["msel_y"])
    rows.add(row)
    log(f"original {ctx['model_name']}: top-1 {vv['top1']:.2f}, flip {vv['flip_consistency']:.2f} (bf16), "
        f"method-selection top-1 {bm['msel_top1']:.2f}")
    del bm
    _free(device)


# ----------------------------------------------------------------------------------------------
# Post-processing: bootstrap, gate B, predictions, summary
# ----------------------------------------------------------------------------------------------
def load_rows(out: Path):
    import pandas as pd

    frames = [pd.read_csv(f) for f in sorted(out.glob("rows_*.csv"))]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["fit_seed"] = df["fit_seed"].astype(int)
    return df


def bootstrap_all(out: Path, df, fit_seed: int, n_resamples: int):
    """Mirror-exact against every other method on validation, per model and size, fit seed `fit_seed` (resumable;
    keyed by the two prediction files' hashes)."""
    import pandas as pd

    csv = V.ResultsCsv(out / "bootstrap.csv", ["model", "size", "method", "fit_seed", "sha_a", "sha_b"])
    ok = df[(df["status"] == "ok") & (df["fit_seed"] == fit_seed)]
    for (model, size), g in ok.groupby(["model", "size"]):
        me = g[g["method"] == "mirror_exact"]
        if me.empty:
            continue
        lab_path = out / "preds" / model / "labels.npz"
        pa = preds_path(out, model, size, "mirror_exact", fit_seed)
        if not lab_path.exists() or not pa.exists():
            continue
        with np.load(lab_path) as z:
            y = z["val"].astype(np.int64)
        with np.load(pa) as z:
            ca = (z["val_pred"].astype(np.int64) == y)
        sha_a = str(me.iloc[0]["preds_sha"])
        for _, r in g.iterrows():
            if r["method"] == "mirror_exact":
                continue
            key = (model, size, r["method"], fit_seed, sha_a, str(r["preds_sha"]))
            if csv.done(key):
                continue
            pb = preds_path(out, model, size, r["method"], fit_seed)
            if not pb.exists():
                continue
            with np.load(pb) as z:
                cb = (z["val_pred"].astype(np.int64) == y)
            res = paired_bootstrap(ca, cb, n_resamples=n_resamples, seed=0)
            csv.add({"model": model, "size": size, "method": r["method"], "fit_seed": fit_seed, "sha_a": sha_a,
                     "sha_b": str(r["preds_sha"]), "a": "mirror_exact", **res})
    return pd.read_csv(out / "bootstrap.csv") if (out / "bootstrap.csv").exists() else None


def gate_b(df, main_model: str, size: str = "full", fit_seed: int = 0) -> dict:
    """Gate B: methods 3 to 12 ranked by method-selection k-NN top-1 at `size` on `main_model`, fit seed `fit_seed`.
    Reads only the method-selection column (never validation). Ties (equal top-1) go to the lower MUST-07 number."""
    sub = df[(df["model"] == main_model) & (df["size"] == size) & (df["fit_seed"] == fit_seed)
             & (df["method"].isin(BASELINES)) & (df["status"] == "ok")]
    recs = [{"method": r["method"], "number": METHOD_NUMBER[r["method"]], "msel_top1": float(r["msel_top1"]),
             "hparams": json.loads(r["hparams"]) if isinstance(r["hparams"], str) and r["hparams"] else {},
             "params_total": int(r["params_total"]), "budget_total": int(r["budget_total"]),
             "gap_rel": float(r["gap_rel"]), "n_msel": int(r["n_msel"])} for _, r in sub.iterrows()]
    recs.sort(key=lambda x: (-x["msel_top1"], x["number"]))
    ties = []
    for i in range(1, len(recs)):
        if recs[i]["msel_top1"] == recs[i - 1]["msel_top1"]:
            ties.append([recs[i - 1]["method"], recs[i]["method"]])
    for i, r in enumerate(recs):
        r["rank"] = i + 1
    present = {r["method"] for r in recs}
    missing = [m for m in BASELINES if m not in present]
    return {"gate": "B", "experiment": "MUST-07", "model": main_model, "size": size, "fit_seed": fit_seed,
            "selection_data": "method-selection queries (10k ImageNet-train images, role msel) against the k-NN bank; "
                              "validation is not read",
            "rule": "rank methods 3 to 12 (every non-Mirror method; Mirror-local is an ablation) by method-selection "
                    "k-NN top-1 at the full-model size, each at its own method-selection-chosen hyperparameters; ties "
                    "go to the lower MUST-07 method number",
            "ranking": recs, "ties": ties, "best": recs[0]["method"] if recs else None,
            "second_best": recs[1]["method"] if len(recs) > 1 else None, "missing_methods": missing,
            "complete": not missing}


def seed_stats(df, model, size, method, col="val_top1") -> dict:
    g = df[(df["model"] == model) & (df["size"] == size) & (df["method"] == method) & (df["status"] == "ok")]
    vals = {int(s): float(v) for s, v in zip(g["fit_seed"], g[col])}
    arr = np.array(list(vals.values()), dtype=float)
    return {"by_seed": vals, "n": len(arr), "mean": float(arr.mean()) if len(arr) else float("nan"),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")}


def _common_diff(a: dict, b: dict) -> tuple:
    seeds = sorted(set(a["by_seed"]) & set(b["by_seed"]))
    if not seeds:
        return float("nan"), seeds
    return float(np.mean([a["by_seed"][s] - b["by_seed"][s] for s in seeds])), seeds


def p7_checks(df, boot, main_model: str, sizes6, fit_seeds=(0, 1, 2)) -> dict:
    """P7.1 and P7.2 (PREREG_NEXT_PHASE.md, MUST-07 operational): per 6-block size, mean over fit seeds 0, 1, 2 of
    Mirror-exact minus each baseline, and the fit-seed-0 paired bootstrap interval."""
    out = {}
    for size in sizes6:
        me = seed_stats(df, main_model, size, "mirror_exact")
        rows = []
        for m in BASELINES:
            st = seed_stats(df, main_model, size, m)
            diff, seeds = _common_diff(me, st)
            b = None
            if boot is not None:
                bb = boot[(boot["model"] == main_model) & (boot["size"] == size) & (boot["method"] == m)]
                if not bb.empty:
                    b = bb.iloc[-1]
            complete = (st["n"] > 0 and set(fit_seeds) <= set(st["by_seed"]) and set(fit_seeds) <= set(me["by_seed"])
                        and b is not None)
            p71 = bool(diff >= 2.0 and b is not None and float(b["ci_low"]) > 0) if st["n"] else None
            p72 = bool(diff >= 10.0) if (st["n"] and m in ("zipit", "pair_actcorr")) else None
            rows.append({"method": m, "number": METHOD_NUMBER[m], "n_seeds": st["n"], "seeds_used": seeds,
                         "mirror_mean": me["mean"], "method_mean": st["mean"], "diff_mean": diff,
                         "ci_low": float(b["ci_low"]) if b is not None else float("nan"),
                         "ci_high": float(b["ci_high"]) if b is not None else float("nan"),
                         "complete": complete, "p71_pass": p71, "p72_pass": p72})
        all_present = all(r["n_seeds"] > 0 for r in rows)
        all_complete = all(r["complete"] for r in rows)
        p71 = all(r["p71_pass"] for r in rows) if all_present else None
        p72_rows = [r for r in rows if r["method"] in ("zipit", "pair_actcorr")]
        p72 = all(r["p72_pass"] for r in p72_rows) if all(r["n_seeds"] > 0 for r in p72_rows) else None
        out[size] = {"rows": rows, "mirror": me, "P7.1": p71, "P7.2": p72, "complete": all_complete}
    return out


def p8_checks(df, main_model: str, tiny: bool) -> dict:
    k6 = "2" if tiny else "6"
    spans = {"6 blocks" if not tiny else "2 blocks": (f"{k6}mlp", f"{k6}attn", f"{k6}both"),
             "full model": ("fullmlp", "fullattn", "full")}
    orig = df[(df["model"] == main_model) & (df["method"] == "original")]
    n_model = int(orig.iloc[0]["model_params"]) if not orig.empty else None
    out = {}
    for span, (s_mlp, s_attn, s_both) in spans.items():
        sav = {}
        for s in (s_mlp, s_attn, s_both):
            st = seed_stats(df, main_model, s, "mirror_exact", "model_params")
            sav[s] = (n_model - st["mean"]) if (n_model is not None and st["n"]) else float("nan")
        p81 = bool(sav[s_mlp] > 0.5 * sav[s_both]) if all(np.isfinite([sav[s_mlp], sav[s_both]])) else None
        acc = {s: seed_stats(df, main_model, s, "mirror_exact") for s in (s_mlp, s_attn, s_both)}
        diff, seeds = _common_diff(acc[s_mlp], acc[s_both])
        out[span] = {"saving": sav, "P8.1": p81, "acc": acc, "mlp_minus_both": diff, "seeds": seeds,
                     "P8.2": (bool(diff <= 0.5) if np.isfinite(diff) else None) if span != "full model" else "n/a"}
    return out


def _f(x, nd=2) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "n/a" if not math.isfinite(x) else f"{x:.{nd}f}"


def _g(x) -> str:
    """GFLOPs: one decimal, more for the tiny model."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    return _f(x, 1 if x >= 1 else 4)


def _ms(st: dict, nd=2) -> str:
    if not st["n"]:
        return "not run"
    if st["n"] == 1:
        return f"{_f(st['mean'], nd)} (n = 1)"
    return f"{_f(st['mean'], nd)} +- {_f(st['std'], nd)} (n = {st['n']})"


def _verdict(v) -> str:
    return "not scored" if v is None else ("PASS" if v is True else ("FAIL" if v is False else str(v)))


def write_summary(out: Path, a, df, boot, gate, git, api_kind: str, budget_kind: str) -> Path:
    tiny = a.tiny
    main = main_model_of(a)
    L = [f"# {a.experiment}: " + ("zero-shot comparison at matched parameters" if a.experiment == "MUST-07"
                                  else "matched component ablation of Mirror-exact")
         + (" (TINY TEST: random weights, synthetic data)" if tiny else ""), "",
         f"Written by `scripts/must07_compressors.py` on {time.strftime('%Y-%m-%d %H:%M:%S')}, commit "
         f"`{git['git_hash'][:10]}`{' (uncommitted changes)' if git['git_dirty'] else ''}. Compressors: **{api_kind}**"
         f"{' (the local SHIM, not the methods of the paper)' if api_kind == 'shim' else ''}; budget accounting: "
         f"**{budget_kind}**.", "",
         "Protocol: k-NN top-1 with the 100k-image train bank (100 per class) and the 50k validation images as queries "
         "(CLS after the final norm, cosine, k = 20, T = 0.07, bf16 features). Method selection: the 10k "
         "method-selection train images as queries against the same bank; every hyperparameter and gate B use only "
         "these. Flip consistency: share of validation images whose prediction equals that of the mirrored image, in "
         "bf16 (column flip) and, for full-model rows, in fp32 (column flip fp32; bf16 rounding breaks exact ties, so "
         "exact invariance is judged on the fp32 value). Parameters: the converted matrices with basis changes; budget "
         "= Mirror-exact's count for the same span and components; gap = (method - budget) / budget. GFLOPs: analytic, "
         "per image, 2 x multiply-adds. Values are mean +- standard deviation over fit seeds (n given).", ""]
    if df is None:
        (out / "SUMMARY.md").write_text("\n".join(L + ["No rows yet.", ""]), encoding="utf-8")
        return out / "SUMMARY.md"
    for model in sorted(df["model"].unique()):
        d = df[df["model"] == model]
        o = d[d["method"] == "original"]
        L += [f"## {model}", ""]
        if not o.empty:
            r = o.iloc[0]
            L += [f"Original: top-1 {_f(r['val_top1'])}, flip {_f(r['flip_consistency'])} (bf16), flip fp32 "
                  f"{_f(r.get('flip_consistency_fp32'))}, method-selection top-1 {_f(r['msel_top1'])}, "
                  f"{int(r['model_params']):,} parameters, {_g(r['gflops'])} GFLOPs.", ""]
        sizes = [s for s in d["size"].unique() if s != "original"]
        comp_order = {"mlp": 0, "attn": 1, "mlp+attn": 2}
        sizes.sort(key=lambda s: (int(d[d["size"] == s]["k"].iloc[0]),
                                  comp_order.get(str(d[d["size"] == s]["components"].iloc[0]), 3)))
        for s in sizes:
            ds = d[d["size"] == s]
            L += [f"### {s} (blocks 1 to {int(ds['k'].iloc[0])}, {ds['components'].iloc[0]})", "",
                  "| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | "
                  "GFLOPs | chosen hyperparameters |", "|---|---|---|---|---|---|---|---|---|---|---|"]
            for m in METHOD_ORDER:
                g = ds[(ds["method"] == m)]
                if g.empty:
                    continue
                okg = g[g["status"] == "ok"]
                if okg.empty:
                    L.append(f"| {METHOD_NUMBER[m]} | {LABEL[m]} | {g.iloc[0]['status']} | | | | | | | | |")
                    continue
                st = seed_stats(d, model, s, m)
                r0 = okg.sort_values("fit_seed").iloc[0]
                s0 = okg[okg["fit_seed"] == 0]
                msel = _f(s0.iloc[0]["msel_top1"]) if not s0.empty else "n/a"
                L.append(f"| {METHOD_NUMBER[m]} | {LABEL[m]} | {_ms(st)} | {msel} | {_f(r0['flip_consistency'])} | "
                         f"{_f(r0.get('flip_consistency_fp32'))} | {int(r0['params_total']):,} | "
                         f"{_f(100 * float(r0['gap_rel']), 2)}% | {int(r0['model_params']):,} "
                         f"({_f(r0['model_params_frac'], 3)}) | {_g(r0['gflops'])} | {r0['hparams'] if r0['hparams'] != '{}' else ''} |")
            f32 = ds[(ds["status"] == "ok") & (ds.get("n_val_fp32", 0).fillna(0).astype(int) > 0)] \
                if "n_val_fp32" in ds else ds.iloc[0:0]
            if not f32.empty:
                L.append(f"fp32 flip pass: {int(f32['n_val_fp32'].iloc[0]):,} validation queries and their mirrors "
                         f"against {int(f32['n_bank_fp32'].iloc[0]):,} bank images (TF32 off).")
            L.append("")
    if a.experiment == "MUST-07":
        sizes6 = SIX["tiny"] if tiny else SIX["MUST-07"]
        chk = p7_checks(df, boot, main, sizes6)
        L += ["## Predictions P7.1 and P7.2 (PREREG_NEXT_PHASE.md)", "",
              f"Model `{main}`. P7.1: Mirror-exact beats every baseline (methods 3 to 12) by at least 2 points (mean over "
              "fit seeds 0, 1 and 2) with the fit-seed-0 paired bootstrap interval above 0. P7.2: ZipIt-style merging "
              "(8) and same-image-correlation pairing (9) trail Mirror-exact by at least 10 points. Each 6-block size "
              "is scored separately.", ""]
        for size, c in chk.items():
            L += [f"### {size}: P7.1 {_verdict(c['P7.1'])}, P7.2 {_verdict(c['P7.2'])}"
                  f"{'' if c['complete'] else ' (incomplete: not every method has fit seeds 0, 1 and 2 and a bootstrap)'}",
                  "", f"Mirror-exact: {_ms(c['mirror'])}.", "",
                  "| # | baseline | seeds | baseline mean | Mirror minus baseline | 95% interval (seed 0) | P7.1 | P7.2 |",
                  "|---|---|---|---|---|---|---|---|"]
            for r in c["rows"]:
                if not r["n_seeds"]:
                    L.append(f"| {r['number']} | {LABEL[r['method']]} | 0 | not run | | | | |")
                    continue
                L.append(f"| {r['number']} | {LABEL[r['method']]} | {r['n_seeds']} | {_f(r['method_mean'])} | "
                         f"{_f(r['diff_mean'])} | [{_f(r['ci_low'])}, {_f(r['ci_high'])}] | {_verdict(r['p71_pass'])} | "
                         f"{'' if r['p72_pass'] is None else _verdict(r['p72_pass'])} |")
            L.append("")
        if gate is not None:
            L += ["## Gate B (best baseline, method-selection set only)", "",
                  f"Model `{gate['model']}`, size {gate['size']}, fit seed {gate['fit_seed']}: best **{gate['best']}**, "
                  f"second best **{gate['second_best']}**"
                  f"{'' if gate['complete'] else '; INCOMPLETE, missing ' + ', '.join(gate['missing_methods'])}. "
                  f"Rule: {gate['rule']}. Written to `gate_b.json`.", "",
                  "| rank | method | method-selection top-1 | hyperparameters |", "|---|---|---|---|"]
            for r in gate["ranking"]:
                L.append(f"| {r['rank']} | {LABEL[r['method']]} | {_f(r['msel_top1'])} | {json.dumps(r['hparams'])} |")
            L.append("")
    else:
        chk = p8_checks(df, main, tiny)
        L += ["## Predictions P8.1 and P8.2 (PREREG_NEXT_PHASE.md)", "",
              "Saving: whole-model parameters of the original minus those of the variant (basis changes counted), mean "
              "over fit seeds. P8.1: the MLP-only saving is more than half of the MLP-plus-attention saving. P8.2 (6 "
              "blocks): k-NN(MLP only) minus k-NN(MLP plus attention), mean over fit seeds, is at most 0.5.", ""]
        for span, c in chk.items():
            names = list(c["saving"])
            L += [f"### {span}: P8.1 {_verdict(c['P8.1'])}" + (f", P8.2 {_verdict(c['P8.2'])}" if c["P8.2"] != "n/a" else ""),
                  "", "| size | saving | val top-1 |", "|---|---|---|"]
            for s in names:
                L.append(f"| {s} | {_f(c['saving'][s], 0)} | {_ms(c['acc'][s])} |")
            if c["P8.2"] != "n/a":
                L.append(f"\nMLP only minus both: {_f(c['mlp_minus_both'])} points over fit seeds {c['seeds']}.")
            L.append("")
    if boot is not None and not boot.empty:
        L += ["## Paired bootstrap, Mirror-exact minus each method (validation, fit seed "
              f"{a.bootstrap_fit_seed}, {a.bootstrap_resamples:,} resamples)", "",
              "| model | size | method | n | difference | 95% interval | excludes 0 |", "|---|---|---|---|---|---|---|"]
        for _, r in boot.iterrows():
            L.append(f"| {r['model']} | {r['size']} | {LABEL.get(r['method'], r['method'])} | {int(r['n'])} | "
                     f"{_f(r['diff'])} | [{_f(r['ci_low'])}, {_f(r['ci_high'])}] | {'yes' if r['excludes_zero'] else 'no'} |")
        L.append("")
    L += ["## Notes", "",
          "- Methods 1 and 9 to 12 use one span operator A* (Mirror-exact machinery); for the pairing controls only the "
          "MLP neuron pairing changes, the attention keeps the fitted head maps. Their parameter counts equal "
          "Mirror-exact's (gap 0).",
          "- Mirror-local counts one d x d rotation per converted block and uses A_b for the attention output; it is an "
          "ablation and is not matched to the budget.",
          "- Mirror-exact's basis change at a span end that is not a model boundary is counted (d x d); "
          "params_total_absorbed in the CSV gives the count with it absorbed into the next dense block.",
          "- Compressor grid points are scored on the method-selection queries only; the chosen point is then scored on "
          "validation." + (" Gate B uses the method-selection top-1 of each method's chosen point, so methods with "
                           "larger grids get a small selection advantage on that set." if a.experiment == "MUST-07" else ""),
          "- Random pairing and random structure use seed random_seed + fit_seed, so their spread over fit seeds "
          "includes the random draw.",
          "- Biases, norms and embeddings are counted as dense; analytic GFLOPs exclude biases, norms and activations.",
          "", "Files: rows_<model>.csv, grid_<model>.csv, bootstrap.csv, preds/, info/, "
          + ("gate_b.json, " if a.experiment == "MUST-07" else "") + "meta.json."]
    text = "\n".join(L) + "\n"
    assert chr(0x2014) not in text
    path = out / "SUMMARY.md"
    path.write_text(text, encoding="utf-8")
    return path


def main_model_of(a) -> str:
    return a.main_model or ("tiny" if a.tiny else "dinov2_vitb14_reg")


def postprocess(a, out: Path, git, api_kind: str, budget_kind: str) -> dict:
    df = load_rows(out)
    boot = bootstrap_all(out, df, a.bootstrap_fit_seed, a.bootstrap_resamples) if df is not None else None
    gate = None
    if a.experiment == "MUST-07" and df is not None:
        gate = gate_b(df, main_model_of(a))
        if gate["ranking"]:
            gate.update({"git_hash": git["git_hash"], "git_dirty": git["git_dirty"],
                         "written": time.strftime("%Y-%m-%d %H:%M:%S")})
            (out / "gate_b.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    summary = write_summary(out, a, df, boot, gate, git, api_kind, budget_kind)
    return {"summary": str(summary), "gate_b": gate, "rows": df, "bootstrap": boot}


# ----------------------------------------------------------------------------------------------
def resolve_methods(spec: str, api: CompressorAPI) -> list:
    if spec in ("all", None):
        cand = list(MF.MIRROR_METHODS) + list(api.methods)
    elif spec == "mirror":
        cand = list(MF.MIRROR_METHODS)
    elif spec == "compressors":
        cand = list(api.methods)
    else:
        cand = [m.strip() for m in spec.split(",") if m.strip()]
    known = set(MF.MIRROR_METHODS) | set(api.methods)
    bad = [m for m in cand if m not in known]
    if bad:
        raise SystemExit(f"unknown or unavailable methods {bad} (compressors: {api.kind} {api.methods})")
    return sorted(dict.fromkeys(cand), key=lambda m: METHOD_NUMBER.get(m, 99))


def main(argv=None) -> dict:
    a = parse_args(argv)
    if a.top_k is None:
        a.top_k = 3 if a.tiny else 10
    if a.workers is None:
        a.workers = 1 if a.tiny else 8
    t_start = time.time()
    exp = EXPERIMENTS[a.experiment]
    out = Path(a.out) if a.out else REPO_ROOT / "results" / a.experiment / ("tiny" if a.tiny else "")
    out.mkdir(parents=True, exist_ok=True)
    if a.tiny:
        device = torch.device("cpu")
    else:
        device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type != "cuda":
            raise SystemExit("no GPU visible; use --tiny for CPU runs")
        torch.cuda.set_per_process_memory_fraction(float(a.gpu_frac))
    torch.backends.cuda.matmul.allow_tf32 = False     # fp32 features (flip_consistency_fp32) need full fp32
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    set_seed(0)
    git = git_info()
    api = compressor_api(force_shim=a.shim)
    budget_kind = MF.budget_api()[0]
    if a.postprocess_only:
        res = postprocess(a, out, git, api.kind, budget_kind)
        return {"out": str(out), **res}
    model_name = "tiny" if a.tiny else a.model
    adapter, proto, num_classes = V.build_runtime(a.model, a.tiny, device, a.cache_dir,
                                                  a.eval_batch or (32 if a.tiny else 512), a.stats_batch_size,
                                                  0 if a.tiny else a.num_workers)
    adapter.model.float().eval()
    L = adapter.depth
    sizes = [parse_size(s, L) for s in (a.sizes or (exp["tiny_sizes"] if a.tiny else exp["sizes"])).split(",") if s.strip()]
    methods = resolve_methods(a.methods or exp["methods"], api)
    seeds = _ints(a.fit_seeds or ("0" if a.tiny else "0,1,2"))
    sdir = Path(a.structure_dir) if a.structure_dir else (out / "structure" if a.tiny else REPO_ROOT / "results" / "MUST-07" / "structure")
    amp = "bf16" if device.type == "cuda" else None
    log(f"{a.experiment} {'TINY ' if a.tiny else ''}on {device}: model {model_name}, sizes {[s.name for s in sizes]}, "
        f"methods {methods}, fit seeds {seeds}; compressors {api.kind}, budget {budget_kind}; commit "
        f"{git['git_hash'][:10]}{' (dirty)' if git['git_dirty'] else ''}")
    rows = V.ResultsCsv(out / f"rows_{model_name}.csv", KEY)
    grid = V.ResultsCsv(out / f"grid_{model_name}.csv", GRID_KEY)
    ev = Evaluator(adapter, proto, device, amp, a.eval_batch or (32 if a.tiny else 512), a.preload, num_classes,
                   a.knn_k, a.knn_temperature, a.queries)
    ev.fp32_queries, ev.fp32_bank_per_class = a.fp32_queries, a.fp32_bank_per_class
    ctx = {"adapter": adapter, "device": device, "ev": ev, "inv": inventory(adapter), "git": git, "out": out,
           "model_name": model_name, "rows": rows, "grid": grid, "api": api, "depth": L,
           "skip_flip": set(_ints(a.skip_flip_seeds))}
    n_before = len(rows.rows)
    struct_files = {}
    if not a.structure_only:
        run_original(a, ctx)
    comp_methods = [m for m in methods if m in api.methods]
    for seed in seeds:
        pending = [(s, m) for s in sizes for m in methods if not rows.done((model_name, s.name, m, seed))]
        if not pending and not a.structure_only:
            log(f"fit seed {seed}: every row done")
            continue
        struct, spath = get_structure(a, adapter, proto, device, model_name, seed, sdir)
        struct_files[seed] = {"path": str(spath), "sha": sha1(spath)}
        if a.structure_only:
            continue
        ctx["struct_sha"] = struct_files[seed]["sha"]
        stats = None
        for size in sizes:
            budget = MF.span_budget(adapter.model, struct, range(size.k), size.components, model_name)
            bf = MF.budget_fields(budget)
            for m in methods:
                if rows.done((model_name, size.name, m, seed)):
                    continue
                if try_reuse(a, out, rows, model_name, size, m, seed, ctx["struct_sha"], L):
                    continue
                if m in MF.MIRROR_METHODS:
                    row = run_mirror(a, ctx, size, m, seed, struct, budget, bf)
                else:
                    if stats is None:
                        t = time.time()
                        stats = api.collect_stats(adapter, proto.fit_loader(seed), device, list(range(L)), comp_methods,
                                                  MF.COMPONENTS)
                        log(f"compressor statistics (fit seed {seed}): {time.time() - t:.1f} s")
                    row = run_compressor(a, ctx, size, m, seed, budget, bf, stats)
                if row.get("status") != "ok":         # every grid point missed the budget: logged in grid_<model>.csv,
                    log(f"{model_name} {size.name} {m} seed {seed}: NOT EVALUATED ({row.get('status')})")
                    continue                          # no row, so a corrected method is evaluated on the next run
                rows.add(row)
                if row.get("status") == "ok":
                    log(f"{model_name} {size.name} {m} seed {seed}: top-1 {row['val_top1']:.2f}, msel "
                        f"{row['msel_top1']:.2f}, flip {row['flip_consistency']:.2f}"
                        + (f" (fp32 {row['flip_consistency_fp32']:.2f})" if np.isfinite(row.get('flip_consistency_fp32', float('nan'))) else "")
                        + f", params {row['params_total']:,} (gap {100 * row['gap_rel']:.3f}%), {row['time_s']:.1f} s")
        del stats
        _free(device)
    res = postprocess(a, out, git, api.kind, budget_kind)
    wall = time.time() - t_start
    from mirror.adapters_timm import resolve_name

    try:
        model_meta = {model_name: (resolve_name(a.model) if not a.tiny else "tiny random timm ViT (CPU test)")}
    except KeyError:
        model_meta = {model_name: a.model}
    extra = {"sizes": [s.__dict__ for s in sizes], "methods": methods, "compressor_api": api.kind,
             "compressor_methods": list(api.methods), "budget_source": budget_kind, "structures": struct_files,
             "rows_added": len(rows.rows) - n_before, "rows_total": len(rows.rows),
             "images_featurised": ev.images_done, "peak_mem_gb_process": _peak(device),
             "data_sizes": {r: s.n for r, s in ev._sets.items()}, "amp": amp, "fp32_flip": a.fp32_flip}
    kw = dict(experiment=a.experiment, config=vars(a),
              seeds={"fit_seeds": seeds, "random_seed_base": a.random_seed, "bootstrap_seed": 0, "global": 0},
              models=model_meta, device=device, wall_seconds=wall, extra=extra)
    write_meta(out, **kw)
    write_meta(out / "meta", name=f"{model_name}_{time.strftime('%Y%m%d_%H%M%S')}.json", **kw)
    log(f"{len(rows.rows) - n_before} new rows ({len(rows.rows)} in rows_{model_name}.csv); summary {res['summary']}; "
        f"{wall / 60:.1f} min")
    return {"out": str(out), "rows_added": len(rows.rows) - n_before, "rows_total": len(rows.rows),
            "compressor_api": api.kind, "budget_source": budget_kind, **res}


if __name__ == "__main__":
    main()
