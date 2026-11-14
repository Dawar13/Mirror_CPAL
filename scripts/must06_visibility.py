#!/usr/bin/env python
"""MUST-06: visibility test, can ordinary redundancy detectors see twins? (PROJECT_NEXT_PHASE.md, MUST-06;
paper Fig. 4a)

    python scripts/must06_visibility.py --model dinov2_vitb14_reg       # H100; fits the structure on fit0
    python scripts/must06_visibility.py --model dinov2_vitl14_reg
    python scripts/must06_visibility.py --model deit3_b16
    python scripts/must06_visibility.py --tiny --out tiny_out/MUST-06   # CPU test mode

Blocks are numbered from 1 (block 1 = model.blocks[0]); --blocks defaults to 1..6 for every model (DINOv2-L
has 24 blocks, so pass for example --blocks 1-12 to cover its first half). The structure (per-layer operators
and twin matching) comes from --fit-run or from a fresh fit on the fit set fit<--fit-seed>, shared with
MUST-05 under --structure-dir (default <cache-dir>/structures/<model>_fit<seed>).

For every twin pair (u, v) of the chosen blocks (the fit-set matching), measured on --images validation
images (Protocol.query_loader; never used to choose anything), over every token, for post-GELU activations
(primary, --act) and pre-activations:
 1. same-token Pearson correlation of h_u and h_v on the same image, against their mirror correlation (h_u on
    the image, h_v on the mirrored image at the mirrored token; symmetrised), the statistic the twins are
    matched on;
 2. the rank of the twin among each neuron's same-image correlated partners (self excluded; ties count
    against the twin), by |r| (primary, --partner-score) and by signed r; a pair counts as seen when the
    twin is in the top --top-partners of either neuron;
 3. singular values of the folded fc1 and fc2 (surgery.fold_mlp) before and after the mirror projection
    (left-right flip, per-layer operators A_l for fc1 and A_(l+1) for fc2, the twin pairing;
    surgery.project_mlp_two), with the effective rank exp(entropy of s / sum s) (Roy and Vetterli 2007
    (verify)) and the energy version (s^2 / sum s^2). A matched random structure (random involutions with
    the same d+, random pairing with the same pair count) is reported as context, not tested.
Matched random pairs (disjoint, same count) are reported next to the twins for 1 and 2.

Predictions (pooled over the chosen blocks): median same-token twin correlation <= 0.3; twin in the top 10
for <= 20% of pairs; effective rank changes by at most 2% (every fc1 and fc2 of the chosen blocks).

Outputs in <out>/<model>/: pairs.csv, blocks.csv, spectra.csv, spectra.npz, fig_correlations.png,
fig_spectra.png, metrics.json, meta.json. Resumable: the validation pass and the spectra are skipped when
pairs.csv or spectra.csv exist with the same settings and structure (their .sig.json files; --force
recomputes); metrics are always rebuilt from the CSVs.
"""
from __future__ import annotations

import torch

DEFAULT_GPU_FRAC = 0.15
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(DEFAULT_GPU_FRAC)  # shared GPU: cap before anything is allocated

import argparse  # noqa: E402
import hashlib  # noqa: E402
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
from mirror.matching import random_pairing  # noqa: E402
from mirror.meta import write_meta  # noqa: E402
from mirror.utils import REPO_ROOT, StageTimer, git_info, log, set_seed  # noqa: E402

EXPERIMENT = "MUST-06"
SAME_MAX, SEEN_MAX, ERANK_MAX = 0.3, 0.2, 0.02


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2_vitb14_reg", help="registry name (mirror/adapters_timm.py)")
    p.add_argument("--fit-run", default=None, help="run_e1.py output directory (else a fresh fit on the fit set)")
    p.add_argument("--fit-seed", type=int, default=None,
                   help="fit set fit<seed> (default 0; with --fit-run, the run's own fit seed)")
    p.add_argument("--structure-dir", default=None,
                   help="where a fresh fit is saved and reused (default <cache-dir>/structures/<model>_fit<seed>; "
                        "tiny: <out>/tiny/tiny_fit)")
    p.add_argument("--blocks", default=None, help="blocks from 1, comma list or ranges like 1-12 (default 1-6; tiny 1-3)")
    p.add_argument("--images", type=int, default=None, help="validation images (default 5000; tiny 20)")
    p.add_argument("--act", choices=("post", "pre"), default="post", help="activation of the tested numbers")
    p.add_argument("--top-partners", type=int, default=10, help="size of the partner list of the detector")
    p.add_argument("--partner-score", choices=("abs", "signed"), default="abs",
                   help="detector ranks partners by |r| (default) or by signed r")
    p.add_argument("--null-seeds", type=int, default=1, help="random structures for the spectrum context")
    p.add_argument("--out", default=str(REPO_ROOT / "results" / EXPERIMENT), help="results root; <out>/<model>/")
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"))
    p.add_argument("--top-k", type=int, default=10, help="matching candidates per neuron (fresh fit)")
    p.add_argument("--workers", type=int, default=4, help="processes for the blossom matchings")
    p.add_argument("--reference-run", default=None,
                   help="earlier run_e1.py run of this model and fit seed (e.g. runs/g/deit3_b16_s0): its pair counts "
                        "and d+ are compared with the structure used (recorded, never used to choose)")
    p.add_argument("--batch-size", type=int, default=32, help="images per batch on validation")
    p.add_argument("--stats-batch-size", type=int, default=16, help="images per batch on the fit set (as run_e1.py)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0, help="seed of the random pairs and random structures")
    p.add_argument("--gpu-frac", type=float, default=DEFAULT_GPU_FRAC, help="per-process GPU memory cap (H100 80 GB)")
    p.add_argument("--force", action="store_true", help="recompute outputs that exist")
    p.add_argument("--tiny", action="store_true", help="tiny random timm ViT and synthetic images on the CPU")
    a = p.parse_args(argv)
    if a.blocks is None:
        a.blocks = "1-3" if a.tiny else "1-6"
    if a.images is None:
        a.images = 20 if a.tiny else 5000
    return a


def free(device) -> None:
    import gc

    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------
# 1 and 2: correlations and the partner detector
# ----------------------------------------------------------------------------------------------
def pair_rows(a, adapter, proto, struct, blocks, device, model_name, notes) -> pd.DataFrame:
    loader = proto.query_loader(a.images)
    idx = np.asarray(loader.dataset.indices)
    notes["images"] = {"n": int(len(idx)), "role": "validation (Protocol.query_loader)",
                       "indices_sha256": hashlib.sha256(idx.astype(np.int64).tobytes()).hexdigest()}
    log(f"validation pass over {len(idx)} images, blocks {[b + 1 for b in blocks]}")
    st = T.collect_act_stats(adapter, loader, device, blocks, desc="validation")
    frames = []
    for b in blocks:
        sets = {"twin": T.twin_pairs(struct.perms[b]),
                "random": T.matched_random_pairs(struct.h, struct.n_pairs(b), seed=a.seed * 1000 + 500 + b)}
        cols = {t: {} for t in sets}
        for kind in T.KINDS:
            Q, R = st.same_corr(b, kind), st.mirror_corr(b, kind)
            for t, P in sets.items():
                cols[t][f"same_corr_{kind}"] = T.pair_values(Q, P, symmetric=False)
                cols[t][f"mirror_corr_{kind}"] = T.pair_values(R, P)
                for score in ("abs", "signed"):
                    ri, rj = T.partner_ranks(Q, P, absolute=score == "abs")
                    cols[t][f"rank_i_{score}_{kind}"], cols[t][f"rank_j_{score}_{kind}"] = ri, rj
            del Q, R
        for t, P in sets.items():
            df = pd.DataFrame({"block": b + 1, "pair_type": t, "i": P[:, 0], "j": P[:, 1]})
            df["fit_S"] = struct.match_scores[b][P[:, 0]] if t == "twin" else np.nan
            for k, v in cols[t].items():
                df[k] = v
            df["h"], df["n_tokens"] = struct.h, st.n_tokens(b)
            frames.append(df)
    st.free()
    del st
    free(device)
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "model", model_name)
    return out


def summarise_pairs(df: pd.DataFrame, act: str, k: int, score: str):
    """Per block (and pooled) summaries of the twin rows, with the matched random pairs as context."""
    def one(g):
        ri, rj = g[f"rank_i_{score}_{act}"], g[f"rank_j_{score}_{act}"]
        either, both = (ri <= k) | (rj <= k), (ri <= k) & (rj <= k)
        h = int(g["h"].iloc[0])
        return {"n_pairs": int(len(g)),
                "same_corr_median": T.describe(g[f"same_corr_{act}"])["median"],
                "same_corr_abs_median": T.describe(g[f"same_corr_{act}"].abs())["median"],
                "same_corr_q10": T.describe(g[f"same_corr_{act}"])["q10"],
                "same_corr_q90": T.describe(g[f"same_corr_{act}"])["q90"],
                "mirror_corr_median": T.describe(g[f"mirror_corr_{act}"])["median"],
                "share_seen_either": float(either.mean()) if len(g) else None,
                "share_seen_both": float(both.mean()) if len(g) else None,
                "share_neurons_seen": float(pd.concat([ri, rj]).le(k).mean()) if len(g) else None,
                "twin_rank_median": float(pd.concat([ri, rj]).median()) if len(g) else None,
                "chance_neuron": k / (h - 1)}

    rows = []
    for (b, t), g in df.groupby(["block", "pair_type"], sort=True):
        rows.append({"block": int(b), "pair_type": t, **one(g)})
    pooled = {t: one(g) for t, g in df.groupby("pair_type")}
    return pd.DataFrame(rows), pooled


# ----------------------------------------------------------------------------------------------
# 3: spectra
# ----------------------------------------------------------------------------------------------
def spectra(a, adapter, struct, blocks, model_name):
    rows, arrays = [], {}
    d = struct.d
    for b in blocks:
        blk = adapter.model.blocks[b]
        structures = [("mirror", 0, struct.A_layers[b], struct.A_layers[b + 1], struct.perms[b])]
        for s in range(a.null_seeds):
            seed = (a.seed + 1) * 100000 + 1000 * s + b
            structures.append(("random", s, V.random_operator(d, struct.d_plus[b], seed),
                               V.random_operator(d, struct.d_plus[b + 1], seed + 500),
                               random_pairing(struct.h, struct.n_pairs(b), seed=seed + 700)))
        for name, s, A_in, A_out, perm in structures:
            sp = T.mlp_spectra(blk, A_in, A_out, perm)
            for mat, r in sp.items():
                row = {"model": model_name, "block": b + 1, "matrix": mat, "structure": name, "null_seed": s,
                       "n_singular": int(len(r["before"])), "breaking_share": r["breaking_share"],
                       "sigma_max_before": float(r["before"].max()), "sigma_max_after": float(r["after"].max())}
                for kind in ("l1", "energy"):
                    e0, e1 = T.effective_rank(r["before"], kind), T.effective_rank(r["after"], kind)
                    row[f"erank_{kind}_before"], row[f"erank_{kind}_after"] = e0, e1
                    row[f"erank_{kind}_rel_change"] = (e1 - e0) / e0
                rows.append(row)
                tag = f"b{b + 1}_{mat}_{name}{s if name == 'random' else ''}"
                arrays[f"{tag}_before"], arrays[f"{tag}_after"] = r["before"], r["after"]
    return pd.DataFrame(rows), arrays


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_correlations(df, act, path: Path, model_name: str) -> None:
    plt = _plt()
    tw = df[df.pair_type == "twin"]
    blocks = sorted(tw["block"].unique())
    cols = min(3, len(blocks))
    rows = int(np.ceil(len(blocks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.6 * rows), squeeze=False, sharex=True)
    bins = np.linspace(-1, 1, 41)
    for ax, b in zip(axes.ravel(), blocks):
        g = tw[tw.block == b]
        ax.hist(g[f"same_corr_{act}"].dropna(), bins=bins, alpha=0.6, label="same image, same token")
        ax.hist(g[f"mirror_corr_{act}"].dropna(), bins=bins, alpha=0.6, label="mirror (matched on)")
        ax.set_title(f"block {b} ({len(g)} twin pairs)", fontsize=9)
    for ax in axes.ravel()[len(blocks):]:
        ax.set_axis_off()
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.supxlabel(f"correlation of twin activations ({act}, validation)", fontsize=9)
    fig.suptitle(f"MUST-06 visibility, {model_name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_spectra(arrays: dict, blocks1, path: Path, model_name: str) -> None:
    plt = _plt()
    cols = min(3, len(blocks1))
    rows = int(np.ceil(len(blocks1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.6 * rows), squeeze=False)
    for ax, b in zip(axes.ravel(), blocks1):
        for mat, colour in (("fc1", "#1f77b4"), ("fc2", "#d62728")):
            s0, s1 = arrays.get(f"b{b}_{mat}_mirror_before"), arrays.get(f"b{b}_{mat}_mirror_after")
            if s0 is None:
                continue
            ax.plot(np.arange(1, len(s0) + 1), s0, color=colour, lw=1.0, label=f"{mat} before")
            ax.plot(np.arange(1, len(s1) + 1), s1, color=colour, lw=1.0, ls="--", label=f"{mat} after")
        ax.set_yscale("log")
        ax.set_title(f"block {b}", fontsize=9)
    for ax in axes.ravel()[len(blocks1):]:
        ax.set_axis_off()
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.supxlabel("singular value index", fontsize=9)
    fig.suptitle(f"MUST-06 spectra before and after the mirror projection, {model_name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def main(argv=None) -> dict:
    a = parse_args(argv)
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
    log(f"{EXPERIMENT} on {model_name}, device {device}, commit {git['git_hash'][:10]}{' (dirty)' if git['git_dirty'] else ''}")

    with timer.stage("load"):
        adapter, proto, _ = V.build_runtime(a.model, a.tiny, device, a.cache_dir, a.batch_size, a.stats_batch_size,
                                            0 if a.tiny else a.num_workers)
        blocks = T.parse_blocks(a.blocks, adapter.depth)
        sdir = a.structure_dir or (out / "tiny_fit" if a.tiny else
                                   T.default_structure_dir(a.cache_dir, a.model, a.fit_seed or 0))
    with timer.stage("structure"):
        struct, src = T.get_structure(adapter=adapter, proto=proto, device=device, model_name=model_name,
                                      fit_run=a.fit_run, fit_seed=a.fit_seed, blocks=blocks, structure_dir=sdir,
                                      top_k=a.top_k, workers=1 if a.tiny else a.workers, log=log)
        if a.reference_run:
            src["reference_check"] = T.reference_check(struct, a.reference_run, blocks)
            log(f"structure against {a.reference_run}: {src['reference_check']}")
    fit_seed = int(src["fit_seed"])
    notes = {}

    pairs_csv = out / "pairs.csv"
    base_sig = {"model": model_name, "blocks": [b + 1 for b in blocks], "seed": a.seed,
                "structure": src.get("sha256")}
    pair_sig = {**base_sig, "images": a.images}
    with timer.stage("correlations"):
        if T.output_current(pairs_csv, pair_sig) and not a.force:
            log(f"{pairs_csv} exists with the same settings, reused")
            df = pd.read_csv(pairs_csv)
            prev = out / "metrics.json"
            if prev.exists():
                notes.update(json.loads(prev.read_text(encoding="utf-8")).get("notes", {}))
        else:
            df = pair_rows(a, adapter, proto, struct, blocks, device, model_name, notes)
            df["seed"], df["fit_seed"], df["git_hash"] = a.seed, fit_seed, git["git_hash"]
            df.to_csv(pairs_csv, index=False)
            T.write_signature(pairs_csv, pair_sig)
        blocks_df, pooled = summarise_pairs(df, a.act, a.top_partners, a.partner_score)
        blocks_df.insert(0, "model", model_name)
        blocks_df.to_csv(out / "blocks.csv", index=False)
        plot_correlations(df, a.act, out / "fig_correlations.png", model_name)

    spec_csv, spec_npz = out / "spectra.csv", out / "spectra.npz"
    with timer.stage("spectra"):
        spec_sig = {**base_sig, "null_seeds": a.null_seeds}
        if T.output_current(spec_csv, spec_sig) and spec_npz.exists() and not a.force:
            log(f"{spec_csv} exists with the same settings, reused")
            sdf = pd.read_csv(spec_csv)
            with np.load(spec_npz) as z:
                arrays = {k: z[k] for k in z.files}
        else:
            sdf, arrays = spectra(a, adapter, struct, blocks, model_name)
            sdf["seed"], sdf["fit_seed"], sdf["git_hash"] = a.seed, fit_seed, git["git_hash"]
            sdf.to_csv(spec_csv, index=False)
            np.savez_compressed(spec_npz, **arrays)
            T.write_signature(spec_csv, spec_sig)
        plot_spectra(arrays, sorted(sdf["block"].unique()), out / "fig_spectra.png", model_name)

    mir = sdf[sdf.structure == "mirror"]
    max_change = float(mir["erank_l1_rel_change"].abs().max()) if len(mir) else None
    tw = pooled.get("twin", {})
    verdicts = {
        "twin_median_same_token_corr": T.verdict(tw.get("same_corr_median"), "<=", SAME_MAX),
        "share_twin_in_top_partners": T.verdict(tw.get("share_seen_either"), "<=", SEEN_MAX),
        "max_abs_rel_change_effective_rank": T.verdict(max_change, "<=", ERANK_MAX),
    }
    for name, v in verdicts.items():
        log(f"PREDICTION {name} {v['op']} {v['threshold']}: {v['value']} -> "
            f"{'PASS' if v['pass'] else 'FAIL' if v['pass'] is False else 'n/a'}")
    per_block = {int(r.block): {k: getattr(r, k) for k in ("n_pairs", "same_corr_median", "mirror_corr_median",
                                                          "share_seen_either", "share_neurons_seen")}
                 for r in blocks_df[blocks_df.pair_type == "twin"].itertuples()}
    erank = {f"{int(r.block)}_{r.matrix}": {"before": r.erank_l1_before, "after": r.erank_l1_after,
                                            "rel_change": r.erank_l1_rel_change, "breaking_share": r.breaking_share}
             for r in mir.itertuples()}
    rnd = sdf[sdf.structure == "random"]
    metrics = {
        "experiment": EXPERIMENT, "model": model_name, "fit_seed": fit_seed, "seed": a.seed, "tiny": bool(a.tiny),
        "blocks": [b + 1 for b in blocks], "structure": src, "act": a.act,
        "detector": {"top_partners": a.top_partners, "score": a.partner_score,
                     "seen": "twin in the top list of either neuron of the pair"},
        "pooled": pooled, "per_block": per_block, "effective_rank_l1": erank,
        "effective_rank_random_structure_max_abs_rel_change": float(rnd["erank_l1_rel_change"].abs().max())
        if len(rnd) else None,
        "verdicts": verdicts, "all_pass": all(v["pass"] is True for v in verdicts.values()),
        "notes": notes, "peak_memory_gb": T.peak_memory_gb(device), "git": git,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    models = {model_name: ({"tiny": "TimmTinyAdapter(img_size=56, seed=0), random weights"} if a.tiny else
                           {"registry": _registry(a.model)})}
    write_meta(out, experiment=EXPERIMENT, config=vars(a),
               seeds={"global": a.seed, "fit": fit_seed, "random_pairs": "seed * 1000 + 500 + block0",
                      "random_structures": "(seed + 1) * 100000 + 1000 * null_seed + block0 (+500 A_out, +700 pairing)"},
               models=models, device=device, wall_seconds=time.time() - t_start,
               extra={"structure": src, "peak_memory_gb": metrics["peak_memory_gb"], "timing": timer.stages,
                      "block_numbering": "1-based; block 1 = model.blocks[0]",
                      "data_roles": {"structure": f"train fit{fit_seed}", "measurement": "validation (query role)"}})
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
