#!/usr/bin/env python
"""Symmetry-group measurement for any model and group (PROJECT_NEXT_PHASE.md 4.3 and 4.4; MUST-03).
Used by MUST-04, MUST-12, CORE-01 and CORE-04.

    python scripts/measure_groups.py --model dinov2_vitb14_reg --groups LR,UD,R90,R180,Klein --fit-seed 0 \
        --surgery UD --surgery-blocks 1-6 --experiment MUST-04 --out results/MUST-04/dinov2_vitb14_reg
    python scripts/measure_groups.py --checkpoint runs/ctrl/lr_s0/ep100.pt --arch controlled \
        --groups LR,UD,R90,Klein --experiment MUST-12 --out results/MUST-12/lr_s0
    python scripts/measure_groups.py --tiny --null-draws 3 --twin-null-draws 2 --surgery UD \
        --surgery-blocks 1-2 --out tiny_out/groups                                       # CPU, about a minute

What it does, for every requested group (LR, UD, R180: order 2; R90: cyclic of order 4; Klein: LR and UD):
  1. One paired pass [x; g(x)] over the fit images per generator (Klein reuses the LR and UD passes),
     collecting for the selected blocks: residual cross-moments C_l (channel-centred), the fc1
     cross-correlation R (fc1 outputs, as all earlier runs), qkv cross-moments for the heads, and, in the
     first pass, the fc1 and fc2 input second moments (activation-weighted shares). The hidden matchings
     run straight after the pass (mutual matching for involutions, Hungarian with the 1/2/4-cycle rule
     for R90) and everything is cached under <out>/cache/, so a rerun repeats no pass.
  2. Per layer: the operator (mirror/groups.py), its multiplicities, in-sample fit error against the
     identity and against random operators with the same character (null draws), the R90 error before
     and after enforcing rho^4 = I, the Klein error before and after enforcing commutation, and the same
     on held-out validation images (reporting only; --no-heldout to skip).
  3. Per block: twin, orbit and self-twin fractions (unthresholded and at 0.5, 0.6, 0.7, 0.8), breaking
     shares of fc1, fc2 (plain and activation-weighted) and attention under the group projection, the
     same for --null-draws draws of random operators with the same character plus random hidden
     permutations (and head structures) with the same cycle type (mean and 95% interval), and the free
     parameters and tax of the character formula. Block b reads layer b and writes layer b + 1: fc1 and
     attention inputs use rho_b, fc2 and attention outputs rho_(b+1) (as run_e1.py and M1).
  4. Twin-fraction null (PREREG_NEXT_PHASE.md Section 1): the token alignment of g is replaced by a random
     permutation of the patch grid with the same cycle type (prefix tokens keep their index); the same
     matching then gives the twin fraction (share in 4-cycles for R90). One forward per batch serves every
     draw in a chunk (--twin-null-mem-gb bounds the accumulators); --twin-null-groups, --twin-null-blocks.
  5. --surgery G1,G2: zero-shot k-NN after group-projection surgery of the MLPs of --surgery-blocks with
     per-layer operators (mirror.groups.apply_group_surgery), always alongside LR, with a paired bootstrap
     against LR on the validation queries. Paper protocol (bank = 100 train images per class, all 50k
     validation queries unless --queries, bf16 features); the original's k-NN comes from
     cache/knn_original/ as in stage1_ablate.py. Not available for --arch controlled.

Blocks are numbered from 1 in every output (block 1 = code index 0). Outputs (all resumable, rows already
present are skipped): <out>/layers.csv, blocks.csv, null_draws.csv, twin_null.csv, surgery.csv,
metrics.json (block-range summaries, ratios to null, twin-null intervals, surgery) and meta.json.

Data roles: the fit set (Protocol.fit_loader(fit seed), ImageNet train; controlled: its fit_loader) for
every statistic; validation only for held-out errors and k-NN reporting; nothing is chosen on validation.
"""
from __future__ import annotations

import sys

import torch

GPU_FRAC_DEFAULT = 0.2   # of an 80 GB H100: 16 GB; DINOv2-L with every statistic needs about 11 GB (estimate)


def _gpu_frac_from_argv(default: float) -> float:
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--gpu-frac" and i + 1 < len(argv):
            return float(argv[i + 1])
        if a.startswith("--gpu-frac="):
            return float(a.split("=", 1)[1])
    return default


if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(_gpu_frac_from_argv(GPU_FRAC_DEFAULT))  # cap before allocating

import argparse  # noqa: E402
import copy  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror import groups as GR  # noqa: E402
from mirror import variants as V  # noqa: E402
from mirror.group_stats import (attn_stats_from_numpy, collect_group_pass, collect_residual_moments,  # noqa: E402
                                collect_twin_null)
from mirror.meta import write_meta  # noqa: E402
from mirror.utils import REPO_ROOT, git_info, log, set_seed  # noqa: E402

LAYER_KEY = ["group", "layer"]
BLOCK_KEY = ["group", "block"]
DRAW_KEY = ["group", "block", "draw"]
TWIN_KEY = ["group", "draw", "block"]
SURGERY_KEY = ["group", "blocks"]
GROUP_INDEX = {g: i for i, g in enumerate(GR.GROUPS)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2_vitb14_reg", help="registry name (mirror/adapters_timm.py)")
    p.add_argument("--checkpoint", default=None, help="checkpoint of a model trained from scratch (with --arch)")
    p.add_argument("--arch", default=None, choices=("controlled",),
                   help="controlled: mirror.controlled.load_checkpoint_adapter and fit_loader")
    p.add_argument("--groups", default="LR,UD,R90,R180,Klein")
    p.add_argument("--fit-seed", type=int, default=0)
    p.add_argument("--fit-images", type=int, default=2000, help="fit images for --arch controlled")
    p.add_argument("--blocks", default="all", help="all, or 1-based ranges such as 1-6 or 1-3,7")
    p.add_argument("--null-draws", type=int, default=20)
    p.add_argument("--twin-null-draws", type=int, default=20)
    p.add_argument("--twin-null-groups", default="UD",
                   help="groups with a twin-fraction null (order-2 groups and R90; Klein is not supported). "
                        "Default UD, the one MUST-04 P4.2 tests: each draw costs one blossom matching per block")
    p.add_argument("--twin-null-blocks", default="1-6")
    p.add_argument("--twin-null-mem-gb", type=float, default=4.0, help="accumulator budget per chunk of draws")
    p.add_argument("--surgery", default="", help="groups for zero-shot surgery, e.g. UD (LR is always added)")
    p.add_argument("--surgery-blocks", default="1-6")
    p.add_argument("--queries", type=int, default=None, help="subsample validation queries (default: all)")
    p.add_argument("--summary-ranges", default="1-6,7-12", help="block ranges for metrics.json (plus both halves)")
    p.add_argument("--top-k", type=int, default=10, help="matching candidates per neuron (as run_e1.py)")
    p.add_argument("--workers", default="auto", help="processes for the matchings (auto: CPU count - 1)")
    p.add_argument("--no-heldout", action="store_true", help="skip held-out (validation) operator errors")
    p.add_argument("--no-attn", action="store_true", help="skip attention statistics and shares")
    p.add_argument("--reference-run", default=None,
                   help="run_e1.py run (e.g. runs/x2/perlayer_reg_s0) whose LR numbers the LR rows must reproduce; "
                        "the differences go to metrics.json (MUST-03 acceptance check)")
    p.add_argument("--seed", type=int, default=0, help="base seed of every null draw")
    p.add_argument("--experiment", default="MUST-03", help="experiment ID written to meta.json")
    p.add_argument("--out", required=True)
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"))
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--stats-batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gpu-frac", type=float, default=GPU_FRAC_DEFAULT,
                   help="torch.cuda.set_per_process_memory_fraction (read at import)")
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"),
                   help="auto: CPU with --tiny, else CUDA (required); cpu: force CPU (small controlled models)")
    p.add_argument("--tiny", action="store_true", help="tiny random ViT + synthetic data on CPU")
    return p.parse_args(argv)


class Csv(V.ResultsCsv):
    """variants.ResultsCsv (resumable by key) with a bulk add that writes the file once."""

    def add_many(self, rows: list) -> int:
        import pandas as pd

        new = [r for r in rows if not self.done(tuple(r[c] for c in self.key))]
        for r in new:
            self.rows.append(r)
            self._done.add(self._k(r))
        if new:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.rows).to_csv(self.path, index=False)
        return len(new)


def parse_blocks(spec: str, L: int) -> list:
    if str(spec).strip().lower() == "all":
        return list(range(L))
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(b - 1 for b in out if 1 <= b <= L)


def _free(device):
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def _workers(a, n_jobs: int) -> int:
    if str(a.workers) == "auto":
        w = 1 if a.tiny else max(1, (os.cpu_count() or 2) - 1)
    else:
        w = int(a.workers)
    return max(1, min(w, n_jobs))


def seed_for(base: int, group: str, draw: int, slot: int) -> int:
    """Null seeds: layer operators slot = layer, hidden permutations 3000 + block, heads 6000 + block."""
    return int(base) * 10_000_000 + GROUP_INDEX[group] * 1_000_000 + int(draw) * 10_000 + int(slot)


def twin_seed(base: int, group: str, draw: int) -> int:
    return int(base) * 10_000_000 + 9_000_000 + GROUP_INDEX[group] * 10_000 + int(draw)


def hungarian_many(R_list, workers: int) -> list:
    """R90 matchings in worker processes (the job is a library function, so any start method works)."""
    if workers > 1 and len(R_list) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(R_list))) as ex:
            return list(ex.map(GR.hungarian_c4, R_list))
    return [GR.hungarian_c4(R) for R in R_list]


# ----------------------------------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------------------------------
class Runtime:
    def __init__(self, a, device):
        self.a, self.device = a, device
        self.proto, self.num_classes, self.controlled = None, None, None
        if a.arch == "controlled":
            if not a.checkpoint:
                raise SystemExit("--arch controlled needs --checkpoint")
            import mirror.controlled as ctl   # written by the controlled-training work package

            self.controlled = ctl
            self.adapter = ctl.load_checkpoint_adapter(a.checkpoint, device)
            info = getattr(self.adapter, "info", {}) or {}
            if a.model != "dinov2_vitb14_reg":
                self.model_name = a.model
            else:   # e.g. controlled_vit_s8_LR_s0_e100 from the checkpoint's own record
                parts = [getattr(self.adapter, "model_name", "controlled")]
                for key, fmt in (("condition", "{}"), ("seed", "s{}"), ("epoch", "e{}")):
                    if info.get(key) is not None:
                        parts.append(fmt.format(info[key]))
                self.model_name = "_".join(str(p) for p in parts)
            self.checkpoint_info = {k: v for k, v in info.items() if isinstance(v, (str, int, float, bool))}
        else:
            self.adapter, self.proto, self.num_classes = V.build_runtime(
                a.model, a.tiny, device, a.cache_dir, a.batch_size, a.stats_batch_size, a.num_workers)
            self.model_name = "tiny" if a.tiny else a.model

    def fit_loader(self):
        if self.controlled is not None:
            return self.controlled.fit_loader(self.a.fit_seed, self.a.fit_images)
        return self.proto.fit_loader(self.a.fit_seed)

    def heldout_loader(self):
        if self.controlled is not None or self.a.no_heldout or not hasattr(self.proto, "heldout_loader"):
            return None
        return self.proto.heldout_loader()


# ----------------------------------------------------------------------------------------------
# Generator passes (cached)
# ----------------------------------------------------------------------------------------------
def _pass_path(out: Path, element: str, fit_seed: int) -> Path:
    return out / "cache" / f"pass_{element}_fit{fit_seed}.npz"


def _cache_ok(path: Path, want: dict) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as z:
            meta = json.loads(str(z["meta_json"]))
    except Exception:
        return False
    return all(meta.get(k) == v for k, v in want.items())


def run_generator_pass(rt: Runtime, element: str, blocks: list, attn: bool, need_inputs: bool, out: Path) -> None:
    """One paired pass for `element` plus its matchings, written to the cache (inputs moments too)."""
    a, dev = rt.a, rt.device
    t0 = time.time()
    log(f"pass {element}: blocks {blocks[0] + 1}..{blocks[-1] + 1}, fit seed {a.fit_seed}")
    gp = collect_group_pass(rt.adapter, rt.fit_loader(), dev, element, blocks, attn=attn, inputs=need_inputs)
    t_pass = time.time() - t0
    R_list = [gp.R[b] for b in blocks]
    gp.R = None
    t1 = time.time()
    if element == "R90":
        res = hungarian_many(R_list, _workers(a, len(R_list)))
        perm = np.stack([r["perm"] for r in res])
        perm_raw = np.stack([r["perm_raw"] for r in res])
        score = np.stack([r["score"] for r in res])
        self_score = np.stack([r["self_score"] for r in res])
        n_turned = np.asarray([r["n_turned"] for r in res])
    else:
        from mirror.matching import match_many

        ms = match_many(R_list, top_k=a.top_k, workers=_workers(a, len(R_list)))
        perm = np.stack([m.perm for m in ms])
        perm_raw = perm
        score = np.stack([m.score for m in ms])
        self_score = np.stack([np.diag(R).astype(np.float64) for R in R_list])
        n_turned = np.zeros(len(blocks), dtype=np.int64)
    del R_list
    t_match = time.time() - t1
    arrays = {"layers": np.asarray(gp.layers), "blocks": np.asarray(blocks), "perm": perm.astype(np.int64),
              "perm_raw": perm_raw.astype(np.int64), "score": score, "self_score": self_score,
              "n_turned": n_turned, **{f"res_{k}": np.asarray(v) for k, v in gp.residual.items()}}
    if gp.attn is not None:
        arrays.update({f"attn_{k}": np.asarray(v) for k, v in gp.attn.items()})
    meta = {"element": element, "blocks": [int(b) for b in blocks], "model": rt.model_name,
            "fit_seed": int(a.fit_seed), "n_images": int(gp.n_images), "top_k": int(a.top_k),
            "attn": bool(gp.attn is not None), "seconds_pass": round(t_pass, 1), "seconds_match": round(t_match, 1)}
    path = _pass_path(out, element, a.fit_seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, meta_json=np.asarray(json.dumps(meta)), **arrays)
    if gp.inputs is not None:
        np.savez(out / "cache" / f"inputs_fit{a.fit_seed}.npz", M1=gp.inputs["M1"],
                 M2=gp.inputs["M2"].astype(np.float32), n=gp.inputs["n"], blocks=np.asarray(blocks),
                 meta_json=np.asarray(json.dumps({"blocks": [int(b) for b in blocks], "model": rt.model_name,
                                                  "fit_seed": int(a.fit_seed)})))
    log(f"pass {element}: {gp.n_images} images, pass {t_pass:.1f} s, matching {t_match:.1f} s")
    del gp
    _free(dev)


def load_pass(out: Path, element: str, fit_seed: int) -> dict:
    with np.load(_pass_path(out, element, fit_seed)) as z:
        d = {k: z[k] for k in z.files}
    meta = json.loads(str(d.pop("meta_json")))
    res = {k[4:]: d.pop(k) for k in list(d) if k.startswith("res_")}
    attn = {k[5:]: d.pop(k) for k in list(d) if k.startswith("attn_")}
    out = {"meta": meta, "layers": [int(x) for x in d["layers"]], "blocks": [int(x) for x in d["blocks"]],
           "perm": d["perm"], "perm_raw": d["perm_raw"], "score": d["score"], "self_score": d["self_score"],
           "n_turned": d["n_turned"], "res": res}
    out["attn_stats"] = attn_stats_from_numpy(attn) if attn else None
    return out


# ----------------------------------------------------------------------------------------------
# Context shared by the analyses
# ----------------------------------------------------------------------------------------------
class Ctx:
    def __init__(self, rt: Runtime, out: Path, blocks: list, attn: bool):
        a, dev = rt.a, rt.device
        self.rt, self.a, self.dev, self.out, self.blocks, self.attn = rt, a, dev, out, blocks, attn
        self.layers = V.span_layers(blocks)
        self.d, self.h, self.L = rt.adapter.embed_dim, rt.adapter.mlp_hidden, rt.adapter.depth
        self.passes: dict = {}
        self.heldout: dict = {}
        from mirror.surgery import fold_mlp

        model = rt.adapter.model
        self.f_mlp = {b: fold_mlp(model.blocks[b]) for b in blocks}
        self.f_attn = {}
        if attn:
            from mirror.attention import fold_attn

            self.f_attn = {b: fold_attn(model.blocks[b]) for b in blocks}
        with np.load(out / "cache" / f"inputs_fit{a.fit_seed}.npz") as z:
            ib = [int(x) for x in z["blocks"]]
            self.M1 = {b: torch.as_tensor(z["M1"][ib.index(b)], dtype=torch.float64, device=dev) for b in blocks}
            self.M2 = {b: torch.as_tensor(z["M2"][ib.index(b)], device=dev) for b in blocks}

    def pass_(self, element: str) -> dict:
        if element not in self.passes:
            self.passes[element] = load_pass(self.out, element, self.a.fit_seed)
        return self.passes[element]

    def moments(self, element: str, l: int, heldout: bool = False):
        src = self.heldout.get(element) if heldout else self.pass_(element)
        if src is None:
            return None
        lay = src["layers"]
        res = src["res"]
        i = lay.index(l)
        return {"C": res["C"][i], "sq_x": float(res["sq_x"][i]), "sq_xt": float(res["sq_xt"][i]),
                "n": float(res["n"][i])}


# ----------------------------------------------------------------------------------------------
# Structure of one group
# ----------------------------------------------------------------------------------------------
def fitted_layer_ops(ctx: Ctx, group: str) -> dict:
    k = GR.kind(group)
    ops = {}
    for l in ctx.layers:
        if k == "c2":
            A, dp, dm = GR.fit_order2(ctx.moments(group, l)["C"])
            ops[l] = {"gens": [A], "raw": None, "d_plus": dp, "d_minus": dm}
        elif k == "c4":
            fc = GR.fit_c4(ctx.moments("R90", l)["C"])
            ops[l] = {"gens": [fc["rho"]], "raw": [fc["rho_raw"]], "m_plus": fc["m_plus"],
                      "m_minus": fc["m_minus"], "m_i_pairs": fc["m_i_pairs"]}
        else:
            Ca, Cb = ctx.moments("LR", l)["C"], ctx.moments("UD", l)["C"]
            kl = GR.fit_klein(Ca, Cb)
            ops[l] = {"gens": [kl["A_a"], kl["A_b"]], "raw": [GR.fit_order2(Ca)[0], GR.fit_order2(Cb)[0]],
                      "iterations": kl["iterations"], **{m: kl[m] for m in ("m_pp", "m_pm", "m_mp", "m_mm")}}
    return ops


def _elements(group: str) -> list:
    return list(GR.GROUP_GENERATORS[group])


def layer_errors(ctx: Ctx, group: str, l: int, gens, heldout: bool = False):
    """(sum over generators of e(rho), the same for the identity, sum ||Xt||^2)."""
    e = eI = sq = 0.0
    for el, rho in zip(_elements(group), gens):
        m = ctx.moments(el, l, heldout)
        if m is None:
            return None
        e += GR.moment_error(m["C"], m["sq_x"], m["sq_xt"], rho)
        eI += m["sq_x"] + m["sq_xt"] - 2.0 * float(np.trace(m["C"]))
        sq += m["sq_xt"]
    return e, eI, sq


def fitted_block_perms(ctx: Ctx, group: str) -> dict:
    k = GR.kind(group)
    out = {}
    if k in ("c2", "c4"):
        P = ctx.pass_(_elements(group)[0])
        for b in ctx.blocks:
            i = P["blocks"].index(b)
            out[b] = {"gens": [P["perm"][i]], "scores": [P["score"][i]], "n_turned": int(P["n_turned"][i]),
                      "n_raw_longer": int((GR.cycle_lengths(P["perm_raw"][i]) > (4 if k == "c4" else 2)).sum())}
        return out
    Pa, Pb = ctx.pass_("LR"), ctx.pass_("UD")
    for b in ctx.blocks:
        ia, ib = Pa["blocks"].index(b), Pb["blocks"].index(b)
        pa, pb = Pa["perm"][ia], Pb["perm"][ib]
        sa, sb = Pa["score"][ia], Pb["score"][ib]
        ssa, ssb = Pa["self_score"][ia], Pb["self_score"][ib]
        kc = GR.klein_commuting(pa, pb, sa - ssa, sb - ssb)
        na, nb = kc["perm_a"], kc["perm_b"]
        out[b] = {"gens": [na, nb], "scores": [np.where(na == pa, sa, ssa), np.where(nb == pb, sb, ssb)],
                  "n_turned": kc["n_units_changed"], "commute_share_before": kc["commute_share_before"],
                  "n_orbits_fixed": kc["n_orbits_fixed"]}
    return out


def fitted_heads(ctx: Ctx, group: str, b: int):
    from mirror.attention import pair_and_fit_heads

    k = GR.kind(group)
    if k == "klein":
        return tuple(fitted_heads(ctx, g, b) for g in ("LR", "UD"))
    P = ctx.pass_(_elements(group)[0])
    mom = P["attn_stats"].moments(P["blocks"].index(b))
    return pair_and_fit_heads(mom) if k == "c2" else GR.fit_heads_c4(mom)


def random_heads_like(group: str, heads, seed: int):
    from mirror.m1 import random_head_maps_like

    k = GR.kind(group)
    if k == "c2":
        return random_head_maps_like(heads, seed)
    if k == "c4":
        return GR.random_heads_c4_like(heads, seed)
    return (random_head_maps_like(heads[0], seed), random_head_maps_like(heads[1], seed + 1))


def attn_projection(group: str, f: dict, heads, gin, gout):
    k = GR.kind(group)
    if k == "c2":
        act = GR.attn_action_c2(heads, gin[0], gout[0])
        return GR.project_attn_group(f, act), act, {}
    if k == "c4":
        act = GR.attn_action_c4(heads, gin[0], gout[0])
        return GR.project_attn_group(f, act), act, {}
    a1 = GR.attn_action_c2(heads[0], gin[0], gout[0])
    a2 = GR.attn_action_c2(heads[1], gin[1], gout[1])
    e, info = GR.project_attn_klein(f, a1, a2, max_sweeps=50, tol=1e-9)
    return e, None, info


def head_summary(group: str, heads) -> dict:
    k = GR.kind(group)
    if k == "c2":
        return {"heads_pairs": int(heads.n_pairs), "heads_fixed": int(heads.n_self)}
    if k == "c4":
        lens = GR.cycle_lengths(heads["sigma"])
        return {"heads_fixed": int((lens == 1).sum()), "heads_2cycles": int((lens == 2).sum()),
                "heads_4cycles": int((lens == 4).sum()), "heads_turned": int(heads["n_turned"])}
    return {"heads_pairs_LR": int(heads[0].n_pairs), "heads_pairs_UD": int(heads[1].n_pairs)}


# ----------------------------------------------------------------------------------------------
# Analysis of one group
# ----------------------------------------------------------------------------------------------
def analyse_group(ctx: Ctx, group: str, git: dict) -> tuple:
    """Layer rows, block rows and per-draw null rows for one group."""
    a, d, h = ctx.a, ctx.d, ctx.h
    k = GR.kind(group)
    t0 = time.time()
    ops = fitted_layer_ops(ctx, group)
    perms = fitted_block_perms(ctx, group)
    heads = {b: fitted_heads(ctx, group, b) for b in ctx.blocks} if ctx.attn else {}
    D = None
    if ctx.attn:
        D = int(ctx.f_attn[ctx.blocks[0]]["Wq"].shape[0])
    n_draws = int(a.null_draws)
    e_null = {l: [] for l in ctx.layers}
    e_null_ho = {l: [] for l in ctx.layers}
    draw_rows, block_vals = [], {b: {} for b in ctx.blocks}
    for draw in range(-1, n_draws):
        if draw < 0:
            ops_k = {l: ops[l]["gens"] for l in ctx.layers}
            perms_k = {b: perms[b]["gens"] for b in ctx.blocks}
            heads_k = heads
        else:
            ops_k = {l: GR.random_operator_like(group, ops[l], d, seed_for(a.seed, group, draw, l))
                     for l in ctx.layers}
            perms_k = {b: GR.random_perm_like(group, perms[b]["gens"],
                                              np.random.default_rng(seed_for(a.seed, group, draw, 3000 + b)))
                       for b in ctx.blocks}
            heads_k = {b: random_heads_like(group, heads[b], seed_for(a.seed, group, draw, 6000 + b))
                       for b in ctx.blocks} if ctx.attn else {}
            for l in ctx.layers:
                e_null[l].append(layer_errors(ctx, group, l, ops_k[l])[0])
                ho = layer_errors(ctx, group, l, ops_k[l], heldout=True)
                if ho is not None:
                    e_null_ho[l].append(ho[0])
        for b in ctx.blocks:
            act = GR.block_action(group, ops_k[b], ops_k[b + 1], perms_k[b])
            sh = GR.mlp_group_shares(ctx.f_mlp[b], act, ctx.M1[b], ctx.M2[b])
            if ctx.attn:
                e, aact, ainfo = attn_projection(group, ctx.f_attn[b], heads_k[b], ops_k[b], ops_k[b + 1])
                sh.update(GR.attn_shares(ctx.f_attn[b], e))
                if draw < 0:
                    block_vals[b]["attn_sweeps"] = ainfo.get("sweeps", "")
                    block_vals[b]["attn_last_change"] = ainfo.get("last_change", "")
                    if aact is not None:
                        fa = GR.attn_free_params(aact, d, D)
                        block_vals[b].update({"free_attn": fa["matrices"], "dense_attn": fa["matrices_original"],
                                              "tax_attn": 1.0 - fa["matrices"] / fa["matrices_original"]})
            if draw < 0:
                block_vals[b].update({"fit": sh, "free": GR.mlp_free_params(act, h, d)})
            else:
                draw_rows.append({"group": group, "block": b + 1, "draw": draw, **sh})
    # ---- layer rows
    layer_rows = []
    for l in ctx.layers:
        e, eI, sq = layer_errors(ctx, group, l, ops[l]["gens"])
        row = {"group": group, "layer": l, "d": d,
               **{m: ops[l][m] for m in ("d_plus", "d_minus", "m_plus", "m_minus", "m_i_pairs", "m_pp", "m_pm",
                                          "m_mp", "m_mm", "iterations") if m in ops[l]},
               "e_rel": e / sq if sq > 0 else float("nan"), "e_ratio_I": e / eI if eI > 0 else float("nan"),
               "n_tokens": ctx.moments(_elements(group)[0], l)["n"]}
        if e_null[l]:
            en = float(np.mean(e_null[l]))
            row.update({"e_null_ratio_I": en / eI, "null_over_fit": en / e if e > 0 else float("inf")})
        if ops[l]["raw"] is not None:
            e_raw = layer_errors(ctx, group, l, ops[l]["raw"])[0]
            row["e_raw_ratio_I"] = e_raw / eI if eI > 0 else float("nan")
            row["e_enforced_over_raw"] = e / e_raw if e_raw > 0 else float("nan")
            if k == "klein":
                A1, A2 = ops[l]["raw"]
                row["commutator_raw"] = float(np.linalg.norm(A1 @ A2 - A2 @ A1) / (2.0 * np.sqrt(d)))
            else:
                R = ops[l]["raw"][0]
                row["raw_order4_defect"] = float(np.linalg.norm(np.linalg.matrix_power(R, 4) - np.eye(d)) / np.sqrt(d))
        ho = layer_errors(ctx, group, l, ops[l]["gens"], heldout=True)
        if ho is not None:
            eh, eIh, sqh = ho
            row.update({"heldout_e_rel": eh / sqh, "heldout_e_ratio_I": eh / eIh})
            if e_null_ho[l]:
                row["heldout_null_over_fit"] = float(np.mean(e_null_ho[l])) / eh if eh > 0 else float("inf")
            if ops[l]["raw"] is not None:
                row["heldout_e_raw_ratio_I"] = layer_errors(ctx, group, l, ops[l]["raw"], heldout=True)[0] / eIh
        row.update({"fit_seed": a.fit_seed, "git_hash": git["git_hash"]})
        layer_rows.append(row)
    # ---- block rows
    block_rows = []
    keys = ("s_fc1", "s_fc2", "s_mlp", "s_b1", "s_b2", "s_fc1_act", "s_fc2_act", "s_attn", "s_q", "s_k", "s_v", "s_o")
    for b in ctx.blocks:
        fit, free = block_vals[b]["fit"], block_vals[b]["free"]
        pb = perms[b]
        fr = GR.orbit_fractions(pb["gens"], pb["scores"])
        row = {"group": group, "block": b + 1, "h": h, "d": d, "layer_in": b, "layer_out": b + 1, **fr,
               "n_pairs": int(round(fr["frac_2"] * h / 2)) if k == "c2" else "",
               "n_self": int(round(fr["frac_fixed"] * h)), "n_turned": pb["n_turned"]}
        for key in ("commute_share_before", "n_orbits_fixed", "n_raw_longer"):
            if key in pb:
                row[key] = pb[key]
        nontriv = np.concatenate([s[g != np.arange(h)] for g, s in zip(pb["gens"], pb["scores"])])
        row["mean_score_nontrivial"] = float(nontriv.mean()) if nontriv.size else float("nan")
        for key in keys:
            if key not in fit:
                continue
            row[key] = fit[key]
            vals = [r[key] for r in draw_rows if r["block"] == b + 1 and key in r]
            ns = GR.null_summary(vals)
            row[f"{key}_null_mean"], row[f"{key}_null_lo"], row[f"{key}_null_hi"] = ns["mean"], ns["lo"], ns["hi"]
            row[f"{key}_over_null"] = fit[key] / ns["mean"] if ns["n"] and ns["mean"] > 0 else float("nan")
        row["null_draws"] = n_draws
        row.update(free)
        for key in ("free_attn", "dense_attn", "tax_attn", "attn_sweeps", "attn_last_change"):
            if key in block_vals[b]:
                row[key] = block_vals[b][key]
        if ctx.attn:
            row.update(head_summary(group, heads[b]))
        if k == "c2":
            from mirror.surgery import equivariant_param_count

            tA_in = int(round(float(np.trace(ops[b]["gens"][0]))))
            tA_out = int(round(float(np.trace(ops[b + 1]["gens"][0]))))
            row["free_fc1_section53"] = equivariant_param_count(h, d, row["n_self"], tA_in)
            row["free_fc2_section53"] = equivariant_param_count(h, d, row["n_self"], tA_out)
        row.update({"fit_seed": a.fit_seed, "null_seed_base": a.seed, "git_hash": git["git_hash"]})
        block_rows.append(row)
    log(f"{group}: {len(ctx.layers)} layers, {len(ctx.blocks)} blocks, {n_draws} null draws, "
        f"{time.time() - t0:.1f} s")
    return layer_rows, block_rows, draw_rows


# ----------------------------------------------------------------------------------------------
# Twin-fraction null
# ----------------------------------------------------------------------------------------------
def run_twin_null(ctx: Ctx, group: str, csv: "Csv", git: dict) -> int:
    a, dev, h = ctx.a, ctx.dev, ctx.h
    element = _elements(group)[0]
    blocks = [b for b in parse_blocks(a.twin_null_blocks, ctx.L) if b in ctx.blocks]
    if not blocks:
        return 0
    todo = [k for k in range(a.twin_null_draws) if not all(csv.done((group, k, b + 1)) for b in blocks)]
    if not todo:
        return 0
    base = GR.token_index(element, ctx.rt.adapter.grid_hw)
    per_draw = len(blocks) * h * h * 8 * 1.05
    chunk = max(1, int(a.twin_null_mem_gb * 2 ** 30 // per_draw))
    added = 0
    for s in range(0, len(todo), chunk):
        ks = todo[s:s + chunk]
        t0 = time.time()
        idxs = [GR.random_permutation_like(base, np.random.default_rng(twin_seed(a.seed, group, k))) for k in ks]
        Rs = collect_twin_null(ctx.rt.adapter, ctx.rt.fit_loader(), dev, element, idxs, blocks)
        _free(dev)
        t_pass = time.time() - t0
        jobs = [(k, b, Rs[i][b]) for i, k in enumerate(ks) for b in blocks]
        del Rs
        if GR.kind(group) == "c4":
            res = hungarian_many([R for _, _, R in jobs], _workers(a, len(jobs)))
            fr = [GR.orbit_fractions([r["perm"]], [r["score"]]) for r in res]
        else:
            from mirror.matching import match_many

            ms = match_many([R for _, _, R in jobs], top_k=a.top_k, workers=_workers(a, len(jobs)))
            fr = [GR.orbit_fractions([m.perm], [m.score]) for m in ms]
        added += csv.add_many([{"group": group, "draw": k, "block": b + 1, "element": element, **f,
                                "seed": twin_seed(a.seed, group, k), "fit_seed": a.fit_seed,
                                "git_hash": git["git_hash"]} for (k, b, _), f in zip(jobs, fr)])
        log(f"twin null {group}: draws {ks[0]}..{ks[-1]} (pass {t_pass:.1f} s, total {time.time() - t0:.1f} s)")
        del jobs
    return added


# ----------------------------------------------------------------------------------------------
# Surgery
# ----------------------------------------------------------------------------------------------
def run_surgery(ctx: Ctx, groups: list, blocks: list, csv: V.ResultsCsv, git: dict) -> dict:
    from mirror.bootstrap import paired_bootstrap
    from mirror.knn import evaluate_knn

    a, rt, dev = ctx.a, ctx.rt, ctx.dev
    amp = None if a.tiny else "bf16"
    bank, query = rt.proto.bank_loader(), rt.proto.query_loader(a.queries)
    labels = np.asarray(query.dataset.labels)[np.asarray(query.dataset.indices)]
    orig = V.original_knn(rt.adapter, bank, query, dev, rt.num_classes, rt.model_name, amp, a.cache_dir,
                          a.queries, a.tiny, git["git_hash"])
    tag = f"{blocks[0] + 1}-{blocks[-1] + 1}" if blocks == list(range(blocks[0], blocks[-1] + 1)) else \
        "_".join(str(b + 1) for b in blocks)
    preds = {}
    for g in ["LR"] + [x for x in groups if x != "LR"]:
        pred_path = ctx.out / f"surgery_pred_{g}_{tag}.npy"
        if csv.done((g, tag)) and pred_path.exists():
            preds[g] = np.load(pred_path)
            continue
        t0 = time.time()
        ops = fitted_layer_ops(ctx, g)
        perms = fitted_block_perms(ctx, g)
        m = copy.deepcopy(rt.adapter.model)
        acts = GR.apply_group_surgery(m, blocks, [ops[b]["gens"] for b in blocks], [ops[b + 1]["gens"] for b in blocks],
                                      [perms[b]["gens"] for b in blocks], group=g)
        r = evaluate_knn(rt.adapter, bank, query, dev, rt.num_classes, k=20, temperature=0.07, model=m, amp=amp)
        del m
        _free(dev)
        pred = r["pred"].numpy()
        np.save(pred_path, pred)
        preds[g] = pred
        free = [GR.mlp_free_params(act, ctx.h, ctx.d) for act in acts]
        kept = sum(f["free_fc1"] + f["free_fc2"] for f in free)
        dense = 2 * ctx.h * ctx.d * len(blocks)
        row = {"group": g, "blocks": tag, "n_blocks": len(blocks), "top1": r["top1"],
               "flip_consistency": r["flip_consistency"], "top1_original": orig["top1"],
               "flip_original": orig["flip_consistency"], "drop_points": orig["top1"] - r["top1"],
               "mlp_params_kept": kept, "mlp_params_dense": dense, "mlp_params_kept_frac": kept / dense,
               "n_query": r["n_query"], "n_bank": r["n_bank"], "original_source": orig["source"],
               "fit_seed": a.fit_seed, "git_hash": git["git_hash"], "time_s": round(time.time() - t0, 1)}
        if g != "LR":
            bs = paired_bootstrap(preds[g] == labels, preds["LR"] == labels)
            row.update({"diff_vs_LR": bs["diff"], "diff_vs_LR_ci_low": bs["ci_low"],
                        "diff_vs_LR_ci_high": bs["ci_high"], "bootstrap_n": bs["n"]})
        csv.add(row)
        log(f"surgery {g} blocks {tag}: top-1 {r['top1']:.2f} (drop {row['drop_points']:.2f}), "
            f"MLP params kept {row['mlp_params_kept_frac']:.3f}")
    return {"original": orig, "blocks": tag}


# ----------------------------------------------------------------------------------------------
# Reference check (LR against an earlier run_e1.py run)
# ----------------------------------------------------------------------------------------------
def reference_check(out: Path, ref: str, fit_seed: int) -> dict:
    """LR rows of this run against e1_layers.csv / e1_mlp.csv of an earlier run_e1.py run (same model and fit
    seed): d+ per layer, twin pairs, self-twins, mean twin correlation and breaking shares per block."""
    import pandas as pd

    res = Path(ref) / "results"
    meta = json.loads((res / "e1_meta.json").read_text(encoding="utf-8")) if (res / "e1_meta.json").exists() else {}
    em, el = pd.read_csv(res / "e1_mlp.csv"), pd.read_csv(res / "e1_layers.csv")
    blk, lay = pd.read_csv(out / "blocks.csv"), pd.read_csv(out / "layers.csv")
    blk, lay = blk[blk["group"] == "LR"], lay[lay["group"] == "LR"]
    m = blk.merge(em, on="block", suffixes=("", "_ref"))
    ml = lay.merge(el[["layer", "d_plus"]], on="layer", suffixes=("", "_ref"))
    out_d = {"reference": str(ref), "reference_fit_seed": meta.get("fit_seed"), "fit_seed": int(fit_seed),
             "blocks_compared": [int(b) for b in m["block"]], "layers_compared": [int(x) for x in ml["layer"]]}
    if len(ml):
        out_d["d_plus_max_abs_diff"] = int((ml["d_plus"].astype(int) - ml["d_plus_ref"].astype(int)).abs().max())
    if len(m):
        out_d["n_pairs_max_abs_diff"] = int((m["n_pairs"].astype(int) - m["n_pairs_ref"].astype(int)).abs().max())
        out_d["n_self_max_abs_diff"] = int((m["n_self"].astype(int) - m["n_self_ref"].astype(int)).abs().max())
        out_d["s_fc1_max_abs_diff"] = float((m["s_fc1"] - m["s1"]).abs().max())
        out_d["s_fc2_max_abs_diff"] = float((m["s_fc2"] - m["s2"]).abs().max())
        b1 = m[m["block"] == 1]
        if len(b1):
            r = b1.iloc[0]
            out_d["block1"] = {"n_pairs": int(r["n_pairs"]), "n_pairs_ref": int(r["n_pairs_ref"]),
                               "s_fc1": float(r["s_fc1"]), "s1_ref": float(r["s1"]),
                               "s_fc2": float(r["s_fc2"]), "s2_ref": float(r["s2"])}
    return out_d


# ----------------------------------------------------------------------------------------------
# metrics.json
# ----------------------------------------------------------------------------------------------
def summary_ranges(spec: str, L: int) -> dict:
    out = {}
    for part in str(spec).split(","):
        part = part.strip()
        if part:
            bl = parse_blocks(part, L)
            if bl:
                out[part] = bl
    half = L // 2
    out["first_half"] = list(range(half))
    out["second_half"] = list(range(half, L))
    return {k: v for k, v in out.items() if v}


def _nanmean(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(x)) if x.size and np.isfinite(x).any() else float("nan")


def build_metrics(out: Path, groups: list, blocks: list, L: int, a, extra: dict) -> dict:
    import pandas as pd

    def read(name):
        p = out / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    blk, lay, drw, twn, srg = (read(n) for n in ("blocks.csv", "layers.csv", "null_draws.csv", "twin_null.csv",
                                                  "surgery.csv"))
    ranges = summary_ranges(a.summary_ranges, L)
    tags = [f"t{int(round(100 * t))}" for t in GR.THRESHOLDS]
    res = {"model": extra.get("model"), "fit_seed": a.fit_seed, "groups": {}, "scores": {},
           "definitions": {
               "ratio_to_null": "mean over the range's blocks of (s_fc1 + s_fc2) / 2, divided by the mean over the "
                                "same blocks of the null means of (s_fc1 + s_fc2) / 2 (PREREG_NEXT_PHASE.md Section 1)",
               "twin_fraction_t50": "mean over blocks of the share of hidden units in a 2-cycle of the matched "
                                    "permutation whose twin correlation is at least 0.5",
               "share_4cycles": "R90: mean over blocks of the share of hidden units in 4-cycles of the Hungarian "
                                "matching after other cycle lengths are turned into fixed points",
               "twin_null": "same pipeline with the token alignment replaced by a random patch permutation with the "
                            "action's cycle type; mean and 2.5th and 97.5th percentiles over draws of the range mean",
               "null interval of a range": "2.5th and 97.5th percentiles over draws of the range mean",
               "blocks": "1-based (block 1 = code index 0)"}}
    for g in groups:
        gb = blk[blk["group"] == g] if len(blk) else blk
        if not len(gb):
            continue
        gl = lay[lay["group"] == g] if len(lay) else lay
        gd = drw[drw["group"] == g] if len(drw) else drw
        gt = twn[twn["group"] == g] if len(twn) else twn
        G = {"kind": GR.kind(g), "ranges": {}}
        for name, rb in ranges.items():
            rb1 = [b + 1 for b in rb if b in blocks]
            sub = gb[gb["block"].isin(rb1)]
            if not len(sub) or len(rb1) != len(rb):
                continue
            r = {"blocks": rb1, "n_blocks": int(len(sub))}
            for key in ("s_fc1", "s_fc2", "s_mlp", "s_fc1_act", "s_fc2_act", "s_attn"):
                if key not in sub:
                    continue
                r[f"{key}_mean"] = _nanmean(sub[key])
                if f"{key}_null_mean" in sub:
                    nm = _nanmean(sub[f"{key}_null_mean"])
                    r[f"{key}_null_mean"] = nm
                    r[f"{key}_ratio_to_null"] = r[f"{key}_mean"] / nm if nm and np.isfinite(nm) and nm > 0 else float("nan")
                    if len(gd) and key in gd:
                        per = gd[gd["block"].isin(rb1)].groupby("draw")[key].mean()
                        ns = GR.null_summary(per.values)
                        r[f"{key}_null_lo"], r[f"{key}_null_hi"], r[f"{key}_null_n"] = ns["lo"], ns["hi"], ns["n"]
            r["ratio_to_null"] = r.get("s_mlp_ratio_to_null", float("nan"))
            r["ratio_to_null_act"] = (0.5 * (r.get("s_fc1_act_mean", np.nan) + r.get("s_fc2_act_mean", np.nan))
                                      / (0.5 * (_nanmean(sub.get("s_fc1_act_null_mean", np.nan))
                                                + _nanmean(sub.get("s_fc2_act_null_mean", np.nan)))))
            for t in [""] + [f"_{x}" for x in tags]:
                for kind_ in ("2", "4", "fixed", "nontrivial"):
                    col = f"frac_{kind_}{t}"
                    if col in sub:
                        r[col] = _nanmean(sub[col])
            r["twin_fraction_t50"] = r.get("frac_2_t50")
            r["share_4cycles"] = r.get("frac_4")
            fsum = float((sub["free_fc1"] + sub["free_fc2"]).sum())
            r["free_mlp"] = fsum
            r["tax_mlp"] = 1.0 - fsum / float((2 * sub["h"] * sub["d"]).sum())
            if "free_attn" in sub and sub["free_attn"].notna().all():
                r["tax_attn"] = 1.0 - float(sub["free_attn"].sum()) / float(sub["dense_attn"].sum())
            span = sorted(set(rb) | {b + 1 for b in rb})
            ls = gl[gl["layer"].isin(span)] if len(gl) else gl
            for key in ("e_ratio_I", "null_over_fit", "heldout_e_ratio_I", "heldout_null_over_fit", "e_raw_ratio_I"):
                if len(ls) and key in ls:
                    r[f"{key}_mean"] = _nanmean(ls[key])
            if len(gt):
                tsub = gt[gt["block"].isin(rb1)]
                if len(tsub) and set(tsub["block"]) == set(rb1):
                    tn = {}
                    for col, obs_key in [("frac_2_t50", "twin_fraction_t50"), ("frac_4", "share_4cycles")] + \
                            [(f"frac_2_{x}", f"frac_2_{x}") for x in tags if x != "t50"] + [("frac_2", "frac_2")]:
                        if col not in tsub:
                            continue
                        per = tsub.groupby("draw")[col].mean()
                        per = per[tsub.groupby("draw")["block"].nunique() == len(rb1)]
                        ns = GR.null_summary(per.values)
                        obs = r.get(obs_key, float("nan"))
                        tn[obs_key] = {"observed": obs, "null_mean": ns["mean"], "null_lo": ns["lo"],
                                       "null_hi": ns["hi"], "n_draws": ns["n"],
                                       "within_null_95": bool(ns["n"] > 0 and ns["lo"] <= obs <= ns["hi"])}
                    r["twin_null"] = tn
            G["ranges"][name] = r
        res["groups"][g] = G
        sc = {}
        for name, r in G["ranges"].items():
            sc[f"ratio_to_null_{name}"] = r.get("ratio_to_null")
            sc[f"twin_fraction_t50_{name}"] = r.get("twin_fraction_t50")
            sc[f"share_4cycles_{name}"] = r.get("share_4cycles")
            sc[f"tax_mlp_{name}"] = r.get("tax_mlp")
            if "twin_null" in r:
                for key, v in r["twin_null"].items():
                    sc[f"twin_null_{key}_{name}"] = v
        res["scores"][g] = sc
    if len(srg):
        res["surgery"] = {str(r["group"]) + "_" + str(r["blocks"]): r for r in srg.to_dict("records")}
    res.update(extra)
    return res


# ----------------------------------------------------------------------------------------------
def main(argv=None) -> dict:
    a = parse_args(argv)
    t_start = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    if a.tiny or a.device == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_per_process_memory_fraction(float(a.gpu_frac))
        torch.cuda.reset_peak_memory_stats()
    else:
        raise SystemExit("no GPU visible; use --tiny for CPU runs")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    git = git_info()
    groups = GR.parse_groups(a.groups)
    surgery_groups = GR.parse_groups(a.surgery) if a.surgery.strip() else []
    if surgery_groups and "LR" not in groups:
        groups = ["LR"] + groups
    for g in surgery_groups:
        if g not in groups:
            groups.append(g)
    rt = Runtime(a, device)
    adapter = rt.adapter
    L = adapter.depth
    blocks = parse_blocks(a.blocks, L)
    if surgery_groups:
        if rt.controlled is not None:
            raise SystemExit("--surgery needs the k-NN protocol; not available for --arch controlled")
        sblocks = parse_blocks(a.surgery_blocks, L)
        blocks = sorted(set(blocks) | set(sblocks))
    attn = not a.no_attn
    log(f"measure_groups on {rt.model_name}: groups {groups}, blocks {blocks[0] + 1}..{blocks[-1] + 1} of {L}, "
        f"grid {adapter.grid_hw}, prefix tokens {adapter.n_prefix_tokens}, d {adapter.embed_dim}, "
        f"h {adapter.mlp_hidden}, device {device}, commit {git['git_hash'][:10]}{' (dirty)' if git['git_dirty'] else ''}")
    timings = {}

    # ---- passes
    want = {"blocks": [int(b) for b in blocks], "model": rt.model_name, "fit_seed": int(a.fit_seed),
            "top_k": int(a.top_k), "attn": bool(attn)}
    inputs_path = out / "cache" / f"inputs_fit{a.fit_seed}.npz"
    inputs_ok = _cache_ok(inputs_path, {"blocks": want["blocks"], "model": rt.model_name, "fit_seed": int(a.fit_seed)})
    for el in GR.generators_needed(groups):
        path = _pass_path(out, el, a.fit_seed)
        if _cache_ok(path, want) and inputs_ok:
            log(f"pass {el}: cached ({path})")
            continue
        t0 = time.time()
        run_generator_pass(rt, el, blocks, attn, need_inputs=not inputs_ok, out=out)
        inputs_ok = True
        timings[f"pass_{el}"] = round(time.time() - t0, 1)
    ctx = Ctx(rt, out, blocks, attn)
    ho_loader = rt.heldout_loader()
    if ho_loader is not None:
        for el in GR.generators_needed(groups):
            p = out / "cache" / f"heldout_{el}.npz"
            if p.exists():
                with np.load(p) as z:
                    lay = [int(x) for x in z["layers"]]
                    if set(ctx.layers) <= set(lay):
                        ctx.heldout[el] = {"layers": lay, "res": {k: z[k] for k in ("C", "G", "sq_x", "sq_xt", "n")}}
                        continue
            t0 = time.time()
            r = collect_residual_moments(adapter, ho_loader, device, el, ctx.layers)
            np.savez(p, layers=np.asarray(ctx.layers), **r)
            ctx.heldout[el] = {"layers": list(ctx.layers), "res": r}
            timings[f"heldout_{el}"] = round(time.time() - t0, 1)
            _free(device)

    # ---- analyses
    lay_csv = Csv(out / "layers.csv", LAYER_KEY)
    blk_csv = Csv(out / "blocks.csv", BLOCK_KEY)
    drw_csv = Csv(out / "null_draws.csv", DRAW_KEY)
    for g in groups:
        done = (all(blk_csv.done((g, b + 1)) for b in blocks) and all(lay_csv.done((g, l)) for l in ctx.layers))
        if done:
            log(f"{g}: rows present, skipped")
            continue
        t0 = time.time()
        lrows, brows, drows = analyse_group(ctx, g, git)
        lay_csv.add_many(lrows)
        drw_csv.add_many(drows)
        blk_csv.add_many(brows)
        timings[f"analyse_{g}"] = round(time.time() - t0, 1)
        _free(device)

    # ---- twin-fraction null
    if a.twin_null_draws > 0:
        tn_groups = [g for g in GR.parse_groups(a.twin_null_groups) if g in groups and GR.kind(g) != "klein"]
        tw_csv = Csv(out / "twin_null.csv", TWIN_KEY)
        for g in tn_groups:
            t0 = time.time()
            n = run_twin_null(ctx, g, tw_csv, git)
            if n:
                timings[f"twin_null_{g}"] = round(time.time() - t0, 1)

    # ---- surgery
    surgery_info = None
    if surgery_groups:
        t0 = time.time()
        surgery_info = run_surgery(ctx, surgery_groups, parse_blocks(a.surgery_blocks, L),
                                   V.ResultsCsv(out / "surgery.csv", SURGERY_KEY), git)
        timings["surgery"] = round(time.time() - t0, 1)

    peak = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
    extra = {"model": rt.model_name, "depth": L, "d": adapter.embed_dim, "h": adapter.mlp_hidden,
             "grid": list(adapter.grid_hw), "n_prefix_tokens": adapter.n_prefix_tokens,
             "blocks_measured": [b + 1 for b in blocks], "timings_s": timings,
             "peak_gpu_memory_bytes": peak, "git": git,
             "n_fit_images": ctx.pass_(GR.generators_needed(groups)[0])["meta"]["n_images"]}
    metrics = build_metrics(out, groups, blocks, L, a, extra)
    if surgery_info is not None:
        metrics["surgery_original"] = surgery_info["original"]
    if a.reference_run and "LR" in groups:
        metrics["reference_check"] = reference_check(out, a.reference_run, a.fit_seed)
        log(f"reference check against {a.reference_run}: {metrics['reference_check']}")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    models = {rt.model_name: ({"checkpoint": a.checkpoint, "arch": a.arch,
                               "checkpoint_info": getattr(rt, "checkpoint_info", {})} if a.arch else
                              {"registry": a.model, "note": "IDs and revisions: configs/models/*.yaml"})}
    write_meta(out, experiment=a.experiment, config=vars(a),
               seeds={"fit_seed": a.fit_seed, "null_seed_base": a.seed, "null_draws": a.null_draws,
                      "twin_null_draws": a.twin_null_draws},
               models=models, device=device, wall_seconds=time.time() - t_start,
               extra={"peak_gpu_memory_bytes": peak, "timings_s": timings, "groups": groups,
                      "blocks": [b + 1 for b in blocks]})
    log(f"done in {time.time() - t_start:.1f} s; metrics in {out / 'metrics.json'}")
    return metrics


if __name__ == "__main__":
    main()
