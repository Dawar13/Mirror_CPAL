#!/usr/bin/env python
"""MUST-05: twins are working mirror pairs (PROJECT_NEXT_PHASE.md, MUST-05; paper Fig. 1).

    python scripts/must05_twins.py --model dinov2_vitb14_reg                  # H100; fits the structure on fit0
    python scripts/must05_twins.py --model dinov2_vitb14_reg --fit-run runs/x2/perlayer_reg_s0
    python scripts/must05_twins.py --tiny --out tiny_out/MUST-05              # CPU test mode, about a minute

Blocks are numbered from 1 on the command line and in every output: block 1 is model.blocks[0] (the block
with 1,168 twin pairs in runs/x2/perlayer_reg_s0/results/e1_mlp.csv).

Structure (per-layer operators A_l, twin pairings, fit-set twin scores S_uv): --fit-run (a run_e1.py run), or
else a fresh fit on the fit set fit<--fit-seed> (2,000 ImageNet train images; variants.fit_structure, the
run_e1.py protocol) saved under --structure-dir (default <cache-dir>/structures/<model>_fit<seed>) and reused.

Data roles (Section 4.1): every choice is made on train images; validation is used only to measure.
  fit set (Protocol.fit_loader)   structure, same-image pairing, ranking of the strongest twins
  validation (query role)         tracking correlations (Protocol.query_loader(--track-images)) and the
                                  switch-off images (a fixed class-balanced subset, recorded by index)
  k-NN bank (train)               teaser patches (a fixed seeded subset of --teaser-images bank rows)

Parts (--parts, default all three; each is resumable: an output is reused when it exists and its signature
(<file>.sig.json, or inside the npz) shows the same settings and structure; --force recomputes):
 track   Mirror correlation of every pair on validation: corr(neuron i on x, neuron j on mirror(x) at the
         mirrored token), symmetrised over (i, j) and (j, i), over every token, for post-GELU (primary, --act)
         and pre-activations. Pair sets per block, all with the twin matching's pair count:
           twin        the fitted twin pairs;
           random      disjoint random pairs (matching.random_pairing);
           cosine      max-weight matching on plain cosine of the folded fc1 rows (variants.generic_pairings);
           same_image  the twin matcher (matching.match_neurons, top-k candidates) applied to the fit-set
                       same-image, same-token correlation with self-pairs scored 0 (see twins.same_image_pairings).
         Fit-set values of the same correlations are kept next to the validation ones.
 switch  Switch-off on --switch-images validation images and their mirrors. For pair (i, j) in block l:
         zero neuron i (after GELU, every token) on image x, and separately neuron j on mirror(x).
         Primary: at the output of block l + 1, cosine between Delta_i(x) A_(l+1) and Delta_j(mirror x)
         re-aligned (un-flipped), and the norm ratio ||Delta_j|| / ||A Delta_i||; A_(l+1) is the per-layer
         operator of the stream there, struct.A_layers[l + 1] (layer 0 = the input of block 1). Secondary:
         the same for the CLS token of the residual stream after the last block (before the final norm)
         with the last layer's operator. Deltas are channel-centred. Per pair the cosine pools every
         token of every image (twins.summarise_pairs). Pair types (--pair-types):
           twin            the --switch-pairs strongest twin pairs by fit-set S_uv;
           random          i drawn from all neurons, j a random neuron (not i, not i's twin);
           random_partner  the twins' i with a random j (same Delta_i as the twin rows).
         Block inputs are cached for the images and their mirrors; every forward runs blocks l to the end
         (to l + 1 only with --no-cls) on a batch holding the clean chunk and many ablated copies.
         Predictions, pooled over the blocks: twin median cosine >= 0.7; random median cosine <= 0.2;
         norm ratio in [0.8, 1.25] for >= 70% of twin pairs. Verdicts go to metrics.json (the CLS ones
         as secondary); the per-image sums stay in switch/block<l>.npz and the per-pair values in CSV.
 teaser  For the --teaser-pairs strongest twin pairs of --teaser-blocks (ranked on the fit set by the
         matching weight w_uv = 2 S_uv - S_uu - S_vv as scripts/f7_visuals.py does, or by S_uv with
         --teaser-rank-by score; w avoids near-symmetric look-alike pairs): the top --teaser-patches patches
         of neuron i and of neuron j (post-GELU, patch tokens, at most one patch per image, as
         scripts/f7_visuals.py) over --teaser-images bank images, as --crop x --crop crops centred on the
         token. Saved as teaser.npz, teaser_top_patches.csv, a raw PNG grid per block and a labelled
         teaser.png. Uses f7_visuals.BlockTracker and collect (bf16 autocast on CUDA, --teaser-amp).

Outputs in <out>/<model>/: tracking_pairs.csv, tracking_summary.csv, switch_images.csv, switch/block<l>.npz,
switch_pairs.csv, switch_summary.csv, teaser*.png, teaser.npz, teaser_top_patches.csv, fig_tracking.png,
fig_switch.png, metrics.json and meta.json (mirror/meta.py).
"""
from __future__ import annotations

import torch

DEFAULT_GPU_FRAC = 0.25
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(DEFAULT_GPU_FRAC)  # shared GPU: cap before anything is allocated

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mirror import twins as T  # noqa: E402
from mirror import variants as V  # noqa: E402
from mirror.data import make_loader  # noqa: E402
from mirror.meta import write_meta  # noqa: E402
from mirror.surgery import fold_mlp  # noqa: E402
from mirror.utils import REPO_ROOT, StageTimer, git_info, log, set_seed  # noqa: E402

EXPERIMENT = "MUST-05"
PARTS = ("track", "switch", "teaser")
TRACK_TYPES = ("twin", "random", "cosine", "same_image")
SWITCH_TYPES = ("twin", "random", "random_partner")
COS_TWIN, COS_RANDOM, BAND, BAND_SHARE = 0.7, 0.2, (0.8, 1.25), 0.7
DEFAULTS = {  # (full run, --tiny)
    "blocks": ("1-6", "1-3"), "track_images": (5000, 20), "switch_images": (500, 6),
    "switch_pairs": (200, 3), "teaser_blocks": ("1,3,6", "1,3"), "teaser_pairs": (8, 2),
    "teaser_patches": (16, 4), "teaser_images": (50000, 40),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2_vitb14_reg", help="registry name (mirror/adapters_timm.py)")
    p.add_argument("--fit-run", default=None, help="run_e1.py output directory (else a fresh fit on the fit set)")
    p.add_argument("--fit-seed", type=int, default=None,
                   help="fit set fit<seed> (Protocol.fit_loader; default 0; with --fit-run, the run's own fit seed)")
    p.add_argument("--structure-dir", default=None,
                   help="where a fresh fit is saved and reused (default <cache-dir>/structures/<model>_fit<seed>; "
                        "tiny: <out>/tiny/tiny_fit)")
    p.add_argument("--parts", default=",".join(PARTS), help=f"comma list from {PARTS}")
    p.add_argument("--blocks", default=None, help="blocks from 1, comma list or ranges like 1-6 (default 1-6; tiny 1-3)")
    p.add_argument("--act", choices=("post", "pre"), default="post",
                   help="activation of the primary tracking numbers and of the same-image pairing")
    p.add_argument("--track-images", type=int, default=None, help="validation images for tracking (default 5000)")
    p.add_argument("--switch-images", type=int, default=None, help="validation images for switch-off (default 500)")
    p.add_argument("--switch-pairs", type=int, default=None, help="pairs per type and block (default 200)")
    p.add_argument("--pair-types", default=",".join(SWITCH_TYPES), help=f"comma list from {SWITCH_TYPES}")
    p.add_argument("--no-cls", action="store_true", help="skip the secondary final-CLS measure (about 4x faster)")
    p.add_argument("--precision", choices=("fp32", "tf32", "bf16"), default="fp32",
                   help="switch-off forwards: fp32 (TF32 off, default), tf32 or bf16 autocast")
    p.add_argument("--chunk-images", type=int, default=4, help="images per switch-off chunk")
    p.add_argument("--abl-batch", type=int, default=256, help="image copies per switch-off forward")
    p.add_argument("--teaser-blocks", default=None, help="blocks for the teaser (default 1,3,6; tiny 1,3)")
    p.add_argument("--teaser-pairs", type=int, default=None, help="strongest pairs per teaser block (default 8)")
    p.add_argument("--teaser-patches", type=int, default=None, help="top patches per neuron (default 16)")
    p.add_argument("--teaser-images", type=int, default=None, help="bank images for the teaser (default 50000)")
    p.add_argument("--teaser-amp", choices=("bf16", "none"), default="bf16", help="teaser pass autocast on CUDA")
    p.add_argument("--teaser-rank-by", choices=("weight", "score"), default="weight",
                   help="teaser pairs: matching weight 2 S_uv - S_uu - S_vv (default, as f7_visuals.py) or S_uv")
    p.add_argument("--crop", type=int, default=42, help="crop side in pixels, centred on the token")
    p.add_argument("--out", default=str(REPO_ROOT / "results" / EXPERIMENT), help="results root; <out>/<model>/")
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"))
    p.add_argument("--top-k", type=int, default=10, help="matching candidates per neuron (as run_e1.py)")
    p.add_argument("--workers", type=int, default=4, help="processes for the blossom matchings")
    p.add_argument("--reference-run", default=None,
                   help="earlier run_e1.py run of this model and fit seed (e.g. runs/x2/perlayer_reg_s0): its pair "
                        "counts and d+ are compared with the structure used (recorded, never used to choose)")
    p.add_argument("--batch-size", type=int, default=32, help="images per batch (validation and bank passes)")
    p.add_argument("--stats-batch-size", type=int, default=16, help="images per batch on the fit set (as run_e1.py)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0, help="seed of image subsets and random pairs")
    p.add_argument("--gpu-frac", type=float, default=DEFAULT_GPU_FRAC, help="per-process GPU memory cap (H100 80 GB)")
    p.add_argument("--force", action="store_true", help="recompute parts whose outputs exist")
    p.add_argument("--tiny", action="store_true", help="tiny random timm ViT and synthetic images on the CPU")
    a = p.parse_args(argv)
    for key, (full, tiny) in DEFAULTS.items():
        if getattr(a, key) is None:
            setattr(a, key, tiny if a.tiny else full)
    return a


def sha_of(*arrays) -> str:
    h = hashlib.sha256()
    for x in arrays:
        h.update(np.ascontiguousarray(np.asarray(x, dtype=np.int64)).tobytes())
    return h.hexdigest()


class Ctx:
    def __init__(self, **kw):
        self.__dict__.update(kw)


# ----------------------------------------------------------------------------------------------
# Part 1: tracking
# ----------------------------------------------------------------------------------------------
def track_rows(c) -> pd.DataFrame:
    a, st, adapter, blocks = c.a, c.struct, c.adapter, c.blocks
    h = st.h
    workers = 1 if a.tiny else a.workers
    sets = {b: {"twin": T.twin_pairs(st.perms[b]),
                "random": T.matched_random_pairs(h, st.n_pairs(b), seed=a.seed * 1000 + 500 + b)} for b in blocks}
    log(f"tracking: cosine pairing of fc1 rows for blocks {[b + 1 for b in blocks]}")
    cos = T.cosine_pairings([fold_mlp(adapter.model.blocks[b])["W1"] for b in blocks],
                            [st.n_pairs(b) for b in blocks], a.top_k, workers)
    log(f"tracking: fit-set pass (fit{c.fit_seed}) for the same-image pairing and the fit-set values")
    fit = T.collect_act_stats(adapter, c.proto.fit_loader(c.fit_seed), c.device, blocks, desc="fit set")
    Qfit = [fit.same_corr(b, a.act).cpu().numpy() for b in blocks]
    same = T.same_image_pairings(Qfit, [st.n_pairs(b) for b in blocks], a.top_k, workers)
    del Qfit
    for b, pc, ps in zip(blocks, cos, same):
        sets[b]["cosine"], sets[b]["same_image"] = pc, ps
    vals = {}
    check = []
    for b in blocks:
        for kind in T.KINDS:
            R, Q = fit.mirror_corr(b, kind), fit.same_corr(b, kind)
            for t, P in sets[b].items():
                vals[(b, t, f"fit_mirror_corr_{kind}")] = T.pair_values(R, P)
                vals[(b, t, f"fit_same_corr_{kind}")] = T.pair_values(Q, P, symmetric=False)
            del R, Q
        tw = sets[b]["twin"]
        if len(tw):
            check.append(float(np.abs(vals[(b, "twin", "fit_mirror_corr_pre")] - st.match_scores[b][tw[:, 0]]).max()))
    c.notes["fit_twin_score_check_max_abs_diff"] = max(check) if check else None
    log(f"tracking: fit-set twin correlation (pre) against the structure's match scores, max abs diff "
        f"{c.notes['fit_twin_score_check_max_abs_diff']}")
    n_fit = fit.n_tokens(blocks[0])
    fit.free()
    del fit
    T_free(c.device)

    loader = c.proto.query_loader(a.track_images)
    idx = np.asarray(loader.dataset.indices)
    c.notes["track_images"] = {"n": int(len(idx)), "role": "validation (Protocol.query_loader)",
                               "indices_sha256": sha_of(idx)}
    log(f"tracking: validation pass over {len(idx)} images")
    val = T.collect_act_stats(adapter, loader, c.device, blocks, desc="validation")
    frames = []
    for b in blocks:
        for kind in T.KINDS:
            R, Q = val.mirror_corr(b, kind), val.same_corr(b, kind)
            for t, P in sets[b].items():
                vals[(b, t, f"val_mirror_corr_{kind}")] = T.pair_values(R, P)
                vals[(b, t, f"val_same_corr_{kind}")] = T.pair_values(Q, P, symmetric=False)
            del R, Q
        for t in TRACK_TYPES:
            P = sets[b][t]
            df = pd.DataFrame({"block": b + 1, "pair_type": t, "i": P[:, 0], "j": P[:, 1]})
            for col in sorted({k[2] for k in vals if k[0] == b and k[1] == t}):
                df[col] = vals[(b, t, col)]
            df["n_tokens_fit"], df["n_tokens_val"] = n_fit, val.n_tokens(b)
            frames.append(df)
    val.free()
    del val
    T_free(c.device)
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "model", c.model_name)
    out["seed"], out["fit_seed"], out["git_hash"] = a.seed, c.fit_seed, c.git["git_hash"]
    return out


def part_track(c) -> dict:
    path = c.out / "tracking_pairs.csv"
    a = c.a
    sig = {"model": c.model_name, "blocks": [b + 1 for b in c.blocks], "act": a.act, "track_images": a.track_images,
           "seed": a.seed, "top_k": a.top_k, "structure": c.struct_src.get("sha256")}
    if T.output_current(path, sig) and not a.force:
        log(f"tracking: {path} exists with the same settings, reused")
        df = pd.read_csv(path)
    else:
        df = track_rows(c)
        df.to_csv(path, index=False)
        T.write_signature(path, sig)
    act = c.a.act
    rows, per_block = [], {}
    for (b, t), g in df.groupby(["block", "pair_type"], sort=True):
        row = {"block": int(b), "pair_type": t}
        for col in (f"val_mirror_corr_{act}", f"val_mirror_corr_{'pre' if act == 'post' else 'post'}",
                    f"val_same_corr_{act}", f"fit_mirror_corr_{act}"):
            for k, v in T.describe(g[col]).items():
                row[f"{col}_{k}"] = v
        rows.append(row)
        per_block.setdefault(int(b), {})[t] = T.describe(g[f"val_mirror_corr_{act}"])
    summ = pd.DataFrame(rows)
    summ.to_csv(c.out / "tracking_summary.csv", index=False)
    plot_tracking(df, act, c.out / "fig_tracking.png", c.model_name)
    return {"act": act, "measure": f"validation mirror correlation (val_mirror_corr_{act}), symmetrised",
            "per_block": per_block, "pooled": {t: T.describe(df.loc[df.pair_type == t, f"val_mirror_corr_{act}"])
                                               for t in TRACK_TYPES if (df.pair_type == t).any()}}


# ----------------------------------------------------------------------------------------------
# Part 2: switch-off
# ----------------------------------------------------------------------------------------------
def switch_pairs_for(c, b: int, types) -> dict:
    a, st = c.a, c.struct
    tw, s = T.strongest_twin_pairs(st.perms[b], st.match_scores[b], a.switch_pairs)
    P = {"twin": tw,
         "random": T.random_neuron_pairs(st.perms[b], a.switch_pairs, seed=a.seed * 1000 + 700 + b),
         "random_partner": np.stack([tw[:, 0], T.random_partners_for(st.perms[b], tw[:, 0],
                                                                      seed=a.seed * 1000 + 800 + b)], axis=1)}
    return {t: P[t] for t in types}, s


def part_switch(c) -> dict:
    a, adapter, st = c.a, c.adapter, c.struct
    types = [t.strip() for t in a.pair_types.split(",") if t.strip()]
    bad = [t for t in types if t not in SWITCH_TYPES]
    if bad or "twin" not in types:
        raise SystemExit(f"--pair-types must include twin and come from {SWITCH_TYPES}")
    pool = np.asarray(c.proto.query_idx) if hasattr(c.proto, "query_idx") else np.arange(len(c.proto.val_labels))
    img_idx = T.balanced_subset(c.proto.val_labels, a.switch_images, seed=a.seed, pool=pool)
    pd.DataFrame({"val_index": img_idx, "label": np.asarray(c.proto.val_labels)[img_idx]}).to_csv(
        c.out / "switch_images.csv", index=False)
    images = None
    amp = "bf16" if a.precision == "bf16" else None
    L = adapter.depth
    sw = c.out / "switch"
    sw.mkdir(parents=True, exist_ok=True)
    done = []
    for b in c.blocks:
        if b + 1 >= L:
            log(f"switch-off: block {b + 1} is the last block (no block {b + 2}); skipped")
            continue
        P, s_twin = switch_pairs_for(c, b, types)
        sig = json.dumps({"model": c.model_name, "block": b + 1, "images": sha_of(img_idx), "pairs": sha_of(*P.values()),
                          "types": list(P), "precision": a.precision, "cls": not a.no_cls, "center": True,
                          "structure": c.struct_src.get("sha256")}, sort_keys=True)
        path = sw / f"block{b + 1}.npz"
        if path.exists() and not a.force:
            with np.load(path) as z:
                if str(z["signature"]) == sig:
                    log(f"switch-off: block {b + 1} done, reused")
                    done.append(b)
                    continue
        if images is None:
            images = np.asarray(c.proto.val[img_idx])
        t0 = time.time()
        with T.matmul_precision(a.precision):
            X, Xm = T.block_inputs(adapter, images, c.device, b, a.batch_size, amp=amp)
            res = T.switch_off_block(adapter, b, X, Xm, P, st.A_layers[b + 2], None if a.no_cls else st.A_layers[L],
                                     chunk_images=a.chunk_images, abl_batch=a.abl_batch, amp=amp)
        del X, Xm
        T_free(c.device)
        arrays = {"signature": np.asarray(sig), "img_idx": img_idx, "twin_fit_score": s_twin}
        for t, Pt in P.items():
            arrays[f"{t}_pairs"] = Pt
            for k, v in res[t].items():
                arrays[f"{t}_{k}"] = v
        tmp = sw / f"block{b + 1}.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
        log(f"switch-off: block {b + 1}, {sum(len(x) for x in P.values())} pairs x {len(img_idx)} images "
            f"in {time.time() - t0:.0f} s")
        done.append(b)
    return summarise_switch(c, done, types)


def summarise_switch(c, blocks, types) -> dict:
    frames = []
    for b in blocks:
        with np.load(c.out / "switch" / f"block{b + 1}.npz") as z:
            s_twin = z["twin_fit_score"]
            n_img = len(z["img_idx"])
            for t in types:
                if f"{t}_pairs" not in z.files:
                    continue
                P = z[f"{t}_pairs"]
                summ = T.summarise_pairs({k: z[f"{t}_{k}"] for k in T.SWITCH_KEYS})
                df = pd.DataFrame({"block": b + 1, "pair_type": t, "rank": np.arange(1, len(P) + 1),
                                   "i": P[:, 0], "j": P[:, 1]})
                df["twin_fit_score"] = s_twin[:len(P)] if t in ("twin", "random_partner") else np.nan
                for k, v in summ.items():
                    df[k] = v
                df["n_images"] = n_img
                frames.append(df)
    if not frames:
        return {"note": "no block had a following block"}
    df = pd.concat(frames, ignore_index=True)
    for pre in ("mid", "patch", "cls"):
        r = df[f"{pre}_norm_ratio"]
        df[f"{pre}_ratio_in_band"] = np.where(r.notna(), (r >= BAND[0]) & (r <= BAND[1]), np.nan)
    df.insert(0, "model", c.model_name)
    df["precision"], df["seed"], df["fit_seed"] = c.a.precision, c.a.seed, c.fit_seed
    df["git_hash"] = c.git["git_hash"]
    df.to_csv(c.out / "switch_pairs.csv", index=False)

    def stats(g, pre):
        band = g[f"{pre}_ratio_in_band"].dropna()
        return {"n_pairs": int(len(g)), "cos_median": T.describe(g[f"{pre}_cos"])["median"],
                "cos_img_median_median": T.describe(g[f"{pre}_cos_img_median"])["median"],
                "norm_ratio_median": T.describe(g[f"{pre}_norm_ratio"])["median"],
                "share_ratio_in_band": float(band.mean()) if len(band) else None}

    rows, per_block = [], {}
    for (b, t), g in df.groupby(["block", "pair_type"], sort=True):
        row = {"block": int(b), "pair_type": t}
        for pre in ("mid", "patch", "cls"):
            row.update({f"{pre}_{k}": v for k, v in stats(g, pre).items()})
        rows.append(row)
        per_block.setdefault(int(b), {})[t] = {pre: stats(g, pre) for pre in ("mid", "cls")}
    pd.DataFrame(rows).to_csv(c.out / "switch_summary.csv", index=False)
    pooled = {t: {pre: stats(df[df.pair_type == t], pre) for pre in ("mid", "patch", "cls")}
              for t in types if (df.pair_type == t).any()}

    def verdicts(pre):
        out = {"twin_median_cos": T.verdict(pooled["twin"][pre]["cos_median"], ">=", COS_TWIN),
               "twin_share_ratio_in_band": T.verdict(pooled["twin"][pre]["share_ratio_in_band"], ">=", BAND_SHARE)}
        if "random" in pooled:
            out["random_median_cos"] = T.verdict(pooled["random"][pre]["cos_median"], "<=", COS_RANDOM)
        return out

    v_main, v_cls = verdicts("mid"), (verdicts("cls") if not c.a.no_cls else {})
    for name, v in v_main.items():
        log(f"PREDICTION (primary, output of block l + 1) {name} {v['op']} {v['threshold']}: "
            f"{v['value']} -> {'PASS' if v['pass'] else 'FAIL' if v['pass'] is False else 'n/a'}")
    for name, v in v_cls.items():
        log(f"secondary (final CLS) {name} {v['op']} {v['threshold']}: {v['value']} -> "
            f"{'PASS' if v['pass'] else 'FAIL' if v['pass'] is False else 'n/a'}")
    plot_switch(df, c.out / "fig_switch.png", c.model_name)
    return {"blocks": [b + 1 for b in blocks], "pair_types": list(types), "n_images": int(df["n_images"].iloc[0]),
            "precision": c.a.precision, "pooling": "pairs of every block pooled (blocks listed above)",
            "band": list(BAND), "pooled": pooled, "per_block": per_block,
            "verdicts_primary": v_main, "verdicts_secondary_cls": v_cls,
            "all_primary_pass": all(v["pass"] is True for v in v_main.values())}


# ----------------------------------------------------------------------------------------------
# Part 3: teaser
# ----------------------------------------------------------------------------------------------
def load_f7(gpu_frac: float):
    """scripts/f7_visuals.py as a module (its tracker, top-patch selection and paired pass). It sets its own
    GPU cap at import, so ours is restored afterwards."""
    spec = importlib.util.spec_from_file_location("f7_visuals_for_must05", ROOT / "scripts" / "f7_visuals.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gpu_frac)
    return mod


def part_teaser(c) -> dict:
    a, adapter, st, proto = c.a, c.adapter, c.struct, c.proto
    path = c.out / "teaser.npz"
    sig = {"model": c.model_name, "blocks": [b + 1 for b in c.teaser_blocks], "pairs": a.teaser_pairs,
           "patches": a.teaser_patches, "images": a.teaser_images, "rank_by": a.teaser_rank_by, "crop": a.crop,
           "amp": a.teaser_amp, "seed": a.seed, "structure": c.struct_src.get("sha256")}
    if not T.output_current(path, sig) or a.force:
        f7 = load_f7(a.gpu_frac)
        bank = np.asarray(proto.index["bank_rows"])
        n = min(int(a.teaser_images), len(bank))
        vis_rows = np.sort(np.random.default_rng(a.seed).choice(bank, size=n, replace=False)).astype(np.int64)
        log(f"teaser: fit-set mirror scores (fit{c.fit_seed}) to rank pairs by {a.teaser_rank_by}")
        S_fit = V.collect_mirror_scores(adapter, proto.fit_loader(c.fit_seed), c.device, c.teaser_blocks)
        sel, trackers = {}, {}
        for b in c.teaser_blocks:
            pairs, s, w = T.rank_twin_pairs(st.perms[b], S_fit[b], a.teaser_pairs, by=a.teaser_rank_by)
            sel[b] = (pairs, s, w)
            rho = f7.random_partners(st.perms[b], seed=a.seed * 1000 + b + 1)
            tracked = pairs.reshape(-1).astype(np.int64)                    # u0, v0, u1, v1, ...
            trackers[b] = f7.BlockTracker(b + 1, st.perms[b], rho, tracked, a.teaser_patches, c.device)
        pin = c.device.type == "cuda"
        loader = make_loader(proto.train, proto.train_labels, vis_rows, a.batch_size,
                             0 if a.tiny else a.num_workers, pin)
        amp = a.teaser_amp if (c.device.type == "cuda" and a.teaser_amp != "none") else None
        log(f"teaser: {n} bank images, blocks {[b + 1 for b in c.teaser_blocks]}, amp {amp}")
        seen = f7.collect(adapter, loader, c.device, trackers, amp=amp, desc="teaser pass")
        if seen != n:
            raise RuntimeError(f"teaser pass saw {seen} images, expected {n}")
        grid_w, patch, S = adapter.grid_hw[1], adapter.patch_size, int(a.crop)
        arrays = {"vis_rows": vis_rows, "blocks": np.asarray([b + 1 for b in c.teaser_blocks]),
                  "crop": np.asarray(S), "patch": np.asarray(patch), "grid_w": np.asarray(grid_w)}
        table = []
        for b in c.teaser_blocks:
            pairs, s, w = sel[b]
            top = trackers[b].top.result()
            k = a.teaser_patches
            crops = np.full((len(pairs), 2, k, S, S, 3), 228, dtype=np.uint8)
            valid = np.zeros((len(pairs), 2, k), dtype=bool)
            for p, (u, v) in enumerate(pairs):
                for side, neuron in enumerate((u, v)):
                    col = 2 * p + side
                    for r in range(k):
                        vi = int(top["img"][col, r])
                        if vi < 0:
                            continue
                        row, pos = int(vis_rows[vi]), int(top["pos"][col, r])
                        crops[p, side, r] = T.token_crop(np.asarray(proto.train[row]), pos, grid_w, patch, S)
                        valid[p, side, r] = True
                        table.append({"block": b + 1, "rank": p + 1, "role": "i" if side == 0 else "j",
                                      "neuron": int(neuron), "twin": int(v if side == 0 else u), "top": r + 1,
                                      "train_row": row, "patch": pos, "patch_row": pos // grid_w,
                                      "patch_col": pos % grid_w, "activation": float(top["val"][col, r]),
                                      "twin_on_mirror": float(top["twin"][col, r]),
                                      "rand_on_mirror": float(top["rand"][col, r]), "fit_S_ij": float(s[p]),
                                      "fit_w_ij": float(w[p])})
            arrays[f"b{b + 1}_pairs"], arrays[f"b{b + 1}_fit_S"], arrays[f"b{b + 1}_fit_w"] = pairs, s, w
            arrays[f"b{b + 1}_crops"], arrays[f"b{b + 1}_valid"] = crops, valid
            for key in ("val", "img", "pos", "twin", "rand"):
                arrays[f"b{b + 1}_top_{key}"] = top[key]
        arrays["rank_by"] = np.asarray(a.teaser_rank_by)
        tdf = pd.DataFrame(table)
        tdf.insert(0, "model", c.model_name)
        tdf["seed"], tdf["git_hash"] = a.seed, c.git["git_hash"]
        tdf.to_csv(c.out / "teaser_top_patches.csv", index=False)
        tmp = c.out / "teaser.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
        T.write_signature(path, sig)
    else:
        log(f"teaser: {path} exists with the same settings, reused")
    with np.load(path) as z:
        z = {k: z[k] for k in z.files}
    figs = draw_teaser(z, c.out, c.model_name)
    tdf = pd.read_csv(c.out / "teaser_top_patches.csv")
    per_block = {}
    for b, g in tdf.groupby("block"):
        ok = g["activation"] > 0
        per_block[int(b)] = {
            "pairs": z[f"b{int(b)}_pairs"].tolist(), "fit_S": z[f"b{int(b)}_fit_S"].tolist(),
            "fit_w": z[f"b{int(b)}_fit_w"].tolist() if f"b{int(b)}_fit_w" in z else None,
            "mean_twin_on_mirror_over_activation": float((g.loc[ok, "twin_on_mirror"] / g.loc[ok, "activation"]).mean())
            if ok.any() else None,
            "mean_rand_on_mirror_over_activation": float((g.loc[ok, "rand_on_mirror"] / g.loc[ok, "activation"]).mean())
            if ok.any() else None}
    return {"n_images": int(len(z["vis_rows"])), "role": "k-NN bank rows (train), fixed seeded subset",
            "rank_by": str(z["rank_by"]) if "rank_by" in z else None,
            "vis_rows_sha256": sha_of(z["vis_rows"]), "crop": int(z["crop"]), "figures": [str(p) for p in figs],
            "per_block": per_block}


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_tracking(df, act, path: Path, model_name: str) -> None:
    plt = _plt()
    blocks = sorted(df["block"].unique())
    cols = min(3, len(blocks))
    rows = int(np.ceil(len(blocks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.8 * rows), squeeze=False, sharey=True)
    col = f"val_mirror_corr_{act}"
    for ax, b in zip(axes.ravel(), blocks):
        g = df[df.block == b]
        data = [g.loc[g.pair_type == t, col].dropna().to_numpy() for t in TRACK_TYPES]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(TRACK_TYPES) + 1), [t.replace("_", "\n") for t in TRACK_TYPES], fontsize=7)
        ax.axhline(0, color="0.7", lw=0.6)
        ax.set_title(f"block {b}", fontsize=9)
    for ax in axes.ravel()[len(blocks):]:
        ax.set_axis_off()
    axes[0, 0].set_ylabel(f"mirror correlation ({act}, validation)", fontsize=8)
    fig.suptitle(f"MUST-05 tracking, {model_name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_switch(df, path: Path, model_name: str) -> None:
    plt = _plt()
    types = [t for t in SWITCH_TYPES if (df.pair_type == t).any()]
    blocks = sorted(df["block"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    w = 0.8 / len(types)
    colours = {"twin": "#1f77b4", "random": "#999999", "random_partner": "#d62728"}
    for k, t in enumerate(types):
        pos = np.arange(len(blocks)) + (k - (len(types) - 1) / 2) * w
        for ax, col in ((axes[0], "mid_cos"), (axes[1], "mid_norm_ratio")):
            data = [df.loc[(df.block == b) & (df.pair_type == t), col].dropna().to_numpy() for b in blocks]
            bp = ax.boxplot(data, positions=pos, widths=w * 0.9, showfliers=False, patch_artist=True)
            for box in bp["boxes"]:
                box.set_facecolor(colours[t])
                box.set_alpha(0.6)
        axes[0].plot([], [], color=colours[t], lw=6, alpha=0.6, label=t)
    axes[0].axhline(COS_TWIN, color="k", ls="--", lw=0.7)
    axes[0].axhline(COS_RANDOM, color="0.5", ls=":", lw=0.7)
    axes[0].set_ylabel("cosine, output of block l + 1")
    axes[1].axhspan(*BAND, color="0.9")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("norm ratio ||D_j|| / ||A D_i||")
    for ax in axes:
        ax.set_xticks(range(len(blocks)), [str(b) for b in blocks])
        ax.set_xlabel("block l")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"MUST-05 switch-off, {model_name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def grid_canvas(crops, valid, gap: int = 2, mid: int = 14, row_gap: int = 6) -> np.ndarray:
    """Crops [P, 2, k, S, S, 3] as one uint8 canvas: one row per pair, i's crops, a gap, j's crops."""
    P, _, k, S = crops.shape[:4]
    group = k * S + (k - 1) * gap
    W = 2 * group + mid
    H = P * S + (P - 1) * row_gap if P else 1
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    for p in range(P):
        y = p * (S + row_gap)
        for side in range(2):
            x0 = side * (group + mid)
            for r in range(k):
                if valid[p, side, r]:
                    x = x0 + r * (S + gap)
                    canvas[y:y + S, x:x + S] = crops[p, side, r]
    return canvas


def draw_teaser(z: dict, out: Path, model_name: str) -> list:
    """teaser_block<l>.png: the raw crop grid of each block (exact pixels); teaser.png: all blocks with labels.
    The labelled figure is laid out in canvas pixels (100 px per inch), so crops are not resampled unevenly."""
    from PIL import Image

    plt = _plt()
    paths, canvases = [], []
    for b in z["blocks"].tolist():
        canvas = grid_canvas(z[f"b{b}_crops"], z[f"b{b}_valid"])
        p = out / f"teaser_block{b}.png"
        Image.fromarray(canvas).save(p)
        paths.append(p)
        canvases.append((b, canvas))
    if canvases:
        LM, RM, TT, GAP = 120, 10, 26, 16                                  # pixels
        Wt = LM + max(cv.shape[1] for _, cv in canvases) + RM
        Ht = sum(TT + cv.shape[0] + GAP for _, cv in canvases)
        fig = plt.figure(figsize=(Wt / 100, Ht / 100))
        S = int(z["crop"])
        y = 0
        for b, cv in canvases:
            y += TT
            h_, w_ = cv.shape[:2]
            ax = fig.add_axes((LM / Wt, 1 - (y + h_) / Ht, w_ / Wt, h_ / Ht))
            ax.imshow(cv, interpolation="nearest")
            ax.set_xlim(-0.5, w_ - 0.5)
            ax.set_ylim(h_ - 0.5, -0.5)
            ax.set_axis_off()
            pairs, s = z[f"b{b}_pairs"], z[f"b{b}_fit_S"]
            for p, (u, v) in enumerate(pairs):
                ax.text(-10, p * (S + 6) + S / 2, f"{u} / {v}\nS = {s[p]:.2f}", ha="right", va="center", fontsize=6)
            ax.text(0, -8, f"block {b}: neuron i (left) and its twin j (right), top patches, {model_name}",
                    ha="left", va="bottom", fontsize=8)
            y += h_ + GAP
        p = out / "teaser.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        paths.append(p)
    log(f"teaser: wrote {', '.join(str(x.name) for x in paths)}")
    return paths


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def T_free(device) -> None:
    import gc

    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def main(argv=None) -> dict:
    a = parse_args(argv)
    parts = [p.strip() for p in a.parts.split(",") if p.strip()]
    if any(p not in PARTS for p in parts):
        raise SystemExit(f"--parts must come from {PARTS}")
    t_start = time.time()
    timer = StageTimer()
    if a.tiny:
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
    set_seed(a.seed)
    git = git_info()
    model_name = "tiny" if a.tiny else a.model
    out = Path(a.out) / model_name
    out.mkdir(parents=True, exist_ok=True)
    log(f"{EXPERIMENT} on {model_name}, parts {parts}, device {device}, commit {git['git_hash'][:10]}"
        f"{' (dirty)' if git['git_dirty'] else ''}")

    with timer.stage("load"):
        adapter, proto, _ = V.build_runtime(a.model, a.tiny, device, a.cache_dir, a.batch_size, a.stats_batch_size,
                                            0 if a.tiny else a.num_workers)
        blocks = T.parse_blocks(a.blocks, adapter.depth)
        teaser_blocks = T.parse_blocks(a.teaser_blocks, adapter.depth)
        sdir = a.structure_dir or (out / "tiny_fit" if a.tiny else
                                   T.default_structure_dir(a.cache_dir, a.model, a.fit_seed or 0))
    with timer.stage("structure"):
        struct, src = T.get_structure(adapter=adapter, proto=proto, device=device, model_name=model_name,
                                      fit_run=a.fit_run, fit_seed=a.fit_seed, blocks=blocks + teaser_blocks,
                                      structure_dir=sdir, top_k=a.top_k, workers=1 if a.tiny else a.workers, log=log)
        if a.reference_run:
            src["reference_check"] = T.reference_check(struct, a.reference_run, sorted(set(blocks + teaser_blocks)))
            log(f"structure against {a.reference_run}: {src['reference_check']}")
    fit_seed = int(src["fit_seed"])
    log(f"structure: {src['kind']} {src['path']}; twin pairs per block "
        f"{ {b + 1: struct.n_pairs(b) for b in sorted(set(blocks + teaser_blocks))} }")
    c = Ctx(a=a, adapter=adapter, proto=proto, struct=struct, struct_src=src, device=device, out=out,
            model_name=model_name, blocks=blocks, teaser_blocks=teaser_blocks, git=git, fit_seed=fit_seed, notes={})

    metrics_path = out / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    metrics.update({"experiment": EXPERIMENT, "model": model_name, "fit_seed": fit_seed, "seed": a.seed,
                    "blocks": [b + 1 for b in blocks], "structure": src, "tiny": bool(a.tiny),
                    "twin_pairs_per_block": {b + 1: struct.n_pairs(b) for b in blocks}})
    if "track" in parts:
        with timer.stage("track"):
            metrics["tracking"] = part_track(c)
    if "switch" in parts:
        with timer.stage("switch"):
            metrics["switch_off"] = part_switch(c)
    if "teaser" in parts:
        with timer.stage("teaser"):
            metrics["teaser"] = part_teaser(c)
    metrics["notes"] = {**metrics.get("notes", {}), **c.notes}
    metrics["peak_memory_gb"] = T.peak_memory_gb(device)
    metrics["git"] = git
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    models = {model_name: ({"tiny": "TimmTinyAdapter(img_size=56, seed=0), random weights"} if a.tiny else
                           {"registry": _registry(a.model)})}
    write_meta(out, experiment=EXPERIMENT, config=vars(a),
               seeds={"global": a.seed, "fit": fit_seed, "random_pairs_tracking": "seed * 1000 + 500 + block0",
                      "random_pairs_switch": "seed * 1000 + 700 + block0",
                      "random_partners_switch": "seed * 1000 + 800 + block0",
                      "switch_images": a.seed, "teaser_images": a.seed},
               models=models, device=device, wall_seconds=time.time() - t_start,
               extra={"structure": src, "peak_memory_gb": metrics["peak_memory_gb"], "timing": timer.stages,
                      "parts": parts, "block_numbering": "1-based; block 1 = model.blocks[0]",
                      "data_roles": {"structure_and_choices": f"train fit{fit_seed}",
                                     "tracking": "validation (Protocol.query_loader)",
                                     "switch_off": "validation, class-balanced subset (switch_images.csv)",
                                     "teaser": "train k-NN bank rows, seeded subset"}})
    log(f"{EXPERIMENT} wrote {out} ({time.time() - t_start:.0f} s, peak GPU memory {metrics['peak_memory_gb']} GB)")
    return metrics


def _registry(name: str) -> dict:
    try:
        from mirror.adapters_timm import resolve_name

        return dict(resolve_name(name))
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


if __name__ == "__main__":
    main()
