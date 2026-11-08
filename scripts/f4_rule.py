#!/usr/bin/env python
"""F4: symmetry versus low rank, single-block surgery and the P2 rule (PAPER_EXPERIMENTS.md F4,
PREREG_E2.md P2, PROJECT_CONTEXT.md Section 5.6; paper Figure 4).

    python scripts/f4_rule.py --model dinov2_vitb14_reg --fit-run runs/x2/perlayer_reg_s0 \
        --out runs/f4 --prereg-commit <hash of the commit that added PREREG_E2.md>
    python scripts/f4_rule.py --tiny --out tiny_out/f4                                      # CPU

For each block l (per-layer operators A_l for fc1 and A_(l+1) for fc2, the fit run's twins):
  1. s (mirror breaking share) and E_svd (energy SVD discards at the matched rank: per matrix
     r = floor(eq / (h + d)), eq the equivariant parameter count), weight-only and
     activation-weighted (W M^(1/2), M the second moment of the matrix's input over tokens).
     M is measured in one extra forward pass over --act-set (default: the protocol's held-out set,
     the first validation image of each class, 1,000 images, no labels used; "fit" uses the train
     fit images instead): for fc1 the LayerNorm output without affine (the input of the folded
     fc1), for fc2 the GELU output. For fc1 the approximation by the residual Gram G of the block's
     input layer (centred over channels; LayerNorm's per-token scaling ignored) is recorded as
     *_actG. With --act-set none the activation-weighted values use G for fc1 and weight-only fc2.
     Also recorded: the activation-weighted error of the plain SVD truncation actually applied.
  2. single-block surgery (only block l converted): mirror projection, and SVD at the matched ranks;
  3. k-NN top-1 of each (bank of 100 train images per class; validation queries, all 50k unless
     --queries; k = 20, T = 0.07, bf16 features) and its drop against the original.
Block-level s and E_svd are the means of the fc1 and fc2 fractions (both unit-free).

P2 (PREREG_E2.md): sign(E_svd - s), activation-weighted, predicts which method has the smaller
drop in at least 75% of block and model cases (DINOv2-reg, plain DINOv2, DeiT-III; all blocks). A
case where either difference is exactly 0 counts as a miss. The evaluation is recomputed over every
f4_<model>.csv in --out after each run and is final only when all three models are complete.

Outputs: <out>/f4_<model>.csv (one row per block; reruns skip blocks already present),
<out>/f4_p2.json, <out>/plots/fig4_rule.png. Non-tiny runs refuse to start unless the commit that
added PREREG_E2.md is known (git, or --prereg-commit on the VM, which has no git binary).
"""
from __future__ import annotations

import torch

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.08)  # shared GPU: cap before anything is allocated

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror import variants as V  # noqa: E402
from mirror.utils import REPO_ROOT, env_info, git_info, log, set_seed  # noqa: E402

SEED = 0
P2_THRESHOLD = 0.75                      # PREREG_E2.md P2, fixed
PREREG_MODELS = ("dinov2_vitb14_reg", "dinov2_vitb14", "deit3_b16")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2_vitb14_reg", help="registry name (mirror/adapters_timm.py)")
    p.add_argument("--fit-run", default=None, help="run_e1.py output directory for this model")
    p.add_argument("--out", default=str(REPO_ROOT / "runs" / "f4"))
    p.add_argument("--fit-seed", type=int, default=None, help="must agree with the fit run's fit seed")
    p.add_argument("--blocks", default=None, help="1-based comma list (default: all blocks)")
    p.add_argument("--queries", type=int, default=None,
                   help="validation queries, a multiple of 1000 (class-balanced); default all 50k")
    p.add_argument("--act-set", choices=("heldout", "fit", "none"), default="heldout",
                   help="images for the activation second moments (none: G approximation + weight-only fc2)")
    p.add_argument("--prereg-commit", default=None, help="commit that added PREREG_E2.md (VM has no git)")
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "cache"))
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--stats-batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--top-k", type=int, default=10, help="matching candidates (only for the --tiny synthetic fit)")
    p.add_argument("--tiny", action="store_true", help="tiny random ViT + synthetic data on CPU")
    return p.parse_args(argv)


def prereg_commit(a) -> str:
    if a.prereg_commit:
        return a.prereg_commit
    try:
        r = subprocess.run(["git", "log", "-n", "1", "--format=%H", "--", "PREREG_E2.md"], cwd=REPO_ROOT,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


# ----------------------------------------------------------------------------------------------
# P2 evaluation and figure
# ----------------------------------------------------------------------------------------------
def rule_hits(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The rule picks the winner: sign(E_svd - s) == sign(SVD drop - mirror drop), both non-zero."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return (np.sign(x) == np.sign(y)) & (x != 0) & (y != 0)


def _frac(df: pd.DataFrame, xcol: str) -> dict:
    ok = df[xcol].notna() & df["rule_y"].notna()
    hits = rule_hits(df.loc[ok, xcol], df.loc[ok, "rule_y"])
    n = int(ok.sum())
    return {"n_cases": n, "n_hits": int(hits.sum()), "fraction": float(hits.mean()) if n else float("nan")}


def evaluate_p2(out_dir: Path) -> dict:
    frames = [pd.read_csv(p) for p in sorted(out_dir.glob("f4_*.csv"))]
    if not frames:
        return {"status": "no data"}
    df = pd.concat(frames, ignore_index=True)
    present = sorted(df["model"].unique().tolist())
    complete = {m: bool(m in present and len(df[df["model"] == m]) == int(df[df["model"] == m]["depth"].iloc[0]))
                for m in PREREG_MODELS}
    pre = df[df["model"].isin(PREREG_MODELS)]
    primary_all = _frac(df, "rule_x")
    primary_pre = _frac(pre, "rule_x")
    if all(complete.values()):
        verdict = "PASS" if primary_pre["fraction"] >= P2_THRESHOLD else "FAIL"
    else:
        verdict = f"INCOMPLETE ({sum(complete.values())} of {len(PREREG_MODELS)} pre-registered models complete)"
    return {
        "hypothesis": "P2: sign(activation-weighted E_svd - s) predicts the smaller single-block k-NN drop "
                      f"in at least {int(P2_THRESHOLD * 100)}% of block and model cases",
        "threshold": P2_THRESHOLD, "verdict": verdict,
        "primary_prereg_models": primary_pre, "primary_all_models_run": primary_all,
        "per_model": {m: _frac(df[df["model"] == m], "rule_x") for m in present},
        "secondary": {
            "weight_only": _frac(df, "rule_x_weight"),
            "activation_weighted_plain_svd_error": _frac(df, "rule_x_plainsvd"),
            "fc1_G_approximation": _frac(df, "rule_x_G"),
        },
        "models_present": present, "complete": complete,
        "act_sources": sorted(df["act_source"].astype(str).unique().tolist()),
        "n_query": sorted(int(q) for q in df["n_query"].unique()),
        "prereg_commits": sorted(df["prereg_commit"].astype(str).unique().tolist()),
        "git_hashes": sorted(df["git_hash"].astype(str).unique().tolist()),
        "tie_rule": "a case with E_svd - s == 0 or SVD drop - mirror drop == 0 counts as a miss",
        "aggregation": "block value = mean of the fc1 and fc2 fractions",
    }


def plot_rule(out_dir: Path, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mirror.report import C_AQUA, C_BLUE, C_GRID, C_INK, C_MUTED, C_NULL, C_ORANGE

    frames = [pd.read_csv(p) for p in sorted(out_dir.glob("f4_*.csv"))]
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True).dropna(subset=["rule_x", "rule_y"])
    colours = [C_BLUE, C_ORANGE, C_AQUA, C_NULL, C_INK]
    markers = ["o", "s", "^", "D", "v"]
    fig, ax = plt.subplots(figsize=(6.2, 4.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    xm = max(float(df["rule_x"].abs().max()), 1e-6) * 1.15
    ym = max(float(df["rule_y"].abs().max()), 1e-6) * 1.15
    ax.fill_between([0, xm], 0, ym, color=C_GRID, alpha=0.45, linewidth=0)
    ax.fill_between([-xm, 0], -ym, 0, color=C_GRID, alpha=0.45, linewidth=0)
    ax.axhline(0, color=C_MUTED, linewidth=0.8)
    ax.axvline(0, color=C_MUTED, linewidth=0.8)
    for i, (model, g) in enumerate(df.groupby("model", sort=True)):
        ax.scatter(g["rule_x"], g["rule_y"], s=34, color=colours[i % len(colours)], marker=markers[i % len(markers)],
                   label=model, zorder=3, edgecolors="white", linewidths=0.6)
        for _, r in g.iterrows():
            ax.annotate(str(int(r["block"])), (r["rule_x"], r["rule_y"]), textcoords="offset points",
                        xytext=(4, 3), fontsize=7, color=C_MUTED)
    hits = rule_hits(df["rule_x"], df["rule_y"])
    ax.set_xlim(-xm, xm)
    ax.set_ylim(-ym, ym)
    ax.set_xlabel("E_svd - s (activation-weighted)", color=C_INK)
    ax.set_ylabel("SVD drop - mirror drop (k-NN points)", color=C_INK)
    ax.set_title(f"Symmetry versus low rank: the rule picks the winner in {int(hits.sum())} of {len(df)} cases",
                 fontsize=10, color=C_INK, loc="left")
    ax.text(0.98, 0.02, "shaded: rule correct", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=C_MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=C_MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
def main(argv=None) -> dict:
    a = parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    if a.tiny:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise SystemExit("no GPU visible; use --tiny for CPU runs")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    amp = None if a.tiny else "bf16"
    git = git_info()
    prereg = prereg_commit(a)
    if not prereg:
        if not a.tiny:
            raise SystemExit("PREREG_E2.md has no commit yet (or git is unavailable). Commit it before F4 runs "
                             "(PAPER_EXPERIMENTS.md X5) and pass --prereg-commit <hash> on the VM.")
        prereg = "none (tiny test)"
    model_name = "tiny" if a.tiny else a.model
    log(f"F4 on {model_name}, device {device}, commit {git['git_hash'][:10]}, PREREG_E2 commit {prereg[:10]}")

    adapter, proto, num_classes = V.build_runtime(a.model, a.tiny, device, a.cache_dir, a.batch_size,
                                                  a.stats_batch_size, a.num_workers)
    L, d, h = adapter.depth, adapter.embed_dim, adapter.mlp_hidden
    struct = V.load_or_make_structure(a.fit_run, a.tiny, adapter, proto, device, out, a.fit_seed, a.top_k)
    if (struct.depth, struct.d, struct.h) != (L, d, h):
        raise SystemExit(f"fit run shape {(struct.depth, struct.d, struct.h)} does not match the model")
    if not a.tiny and struct.meta.get("model") and struct.meta["model"] != a.model:
        raise SystemExit(f"fit run is for {struct.meta['model']}, not {a.model}")
    fit_seed = V.resolve_fit_seed(a.fit_seed, struct)
    blocks = list(range(L)) if not a.blocks else sorted({int(x) - 1 for x in a.blocks.split(",")})
    csv = V.ResultsCsv(out / f"f4_{model_name}.csv", ["model", "block"])
    todo = [b for b in blocks if not csv.done((model_name, b + 1))]
    t0 = time.time()
    if todo:
        moments, act_source = None, "fc1: G approximation; fc2: weight-only"
        if a.act_set != "none":
            loader = proto.heldout_loader() if a.act_set == "heldout" else proto.fit_loader(fit_seed)
            log(f"activation second moments over the {a.act_set} set, blocks {[b + 1 for b in todo]}")
            moments = V.collect_mlp_input_moments(adapter, loader, device, blocks=todo)
            moments["M1"] = {b: m.cpu() for b, m in moments["M1"].items()}
            moments["M2"] = {b: m.cpu() for b, m in moments["M2"].items()}
            act_source = f"measured ({a.act_set} set, {moments['n_images']} images)"
        bank, query = proto.bank_loader(), proto.query_loader(a.queries)
        orig = V.original_knn(adapter, bank, query, device, num_classes, model_name, amp, a.cache_dir,
                              a.queries, a.tiny, git["git_hash"])
        log(f"original top-1 {orig['top1']:.2f} ({orig['source']})")
        n_query = len(query.dataset)
        for b in todo:
            t = time.time()
            blk = adapter.model.blocks[b]
            G_b = struct.G[b] if struct.G is not None else None
            ms = V.block_rule_measures(blk, struct.A_layers[b], struct.A_layers[b + 1], struct.perms[b],
                                       M1=None if moments is None else moments["M1"][b],
                                       M2=None if moments is None else moments["M2"][b], M1_G=G_b)
            if moments is not None:
                s_act = 0.5 * (ms["s_fc1_act"] + ms["s_fc2_act"])
                E_act = 0.5 * (ms["E_fc1_act"] + ms["E_fc2_act"])
                E_plain = 0.5 * (ms["E_fc1_act_plainsvd"] + ms["E_fc2_act_plainsvd"])
            elif G_b is not None:
                s_act = 0.5 * (ms["s_fc1_actG"] + ms["s_fc2_weight"])
                E_act = 0.5 * (ms["E_fc1_actG"] + ms["E_fc2_weight"])
                E_plain = float("nan")
            else:
                raise SystemExit("no activation statistics: the fit run has no G and --act-set is none")
            s_G = 0.5 * (ms["s_fc1_actG"] + ms["s_fc2_weight"]) if G_b is not None else float("nan")
            E_G = 0.5 * (ms["E_fc1_actG"] + ms["E_fc2_weight"]) if G_b is not None else float("nan")
            mirror = V.mirror_spec("mirror", [b], [struct.A_layers[b]], [struct.A_layers[b + 1]],
                                   [struct.perms[b]], h, d)
            svd = V.svd_spec("svd", struct, [b])
            top1 = {}
            for name, spec in (("mirror", mirror), ("svd", svd)):
                m = V.variant_copy(adapter.model, spec)
                top1[name] = V.knn_top1(adapter, bank, query, device, num_classes, 20, 0.07, model=m, amp=amp)
                del m
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            drop_m, drop_s = orig["top1"] - top1["mirror"], orig["top1"] - top1["svd"]
            row = {"model": model_name, "block": b + 1, "depth": L, **ms,
                   "s_act": s_act, "E_act": E_act, "E_act_plainsvd": E_plain, "s_actG": s_G, "E_actG": E_G,
                   "act_source": act_source,
                   "mlp_params_mirror": mirror.params_kept, "mlp_params_svd": svd.params_kept,
                   "top1_original": orig["top1"], "top1_mirror": top1["mirror"], "top1_svd": top1["svd"],
                   "drop_mirror": drop_m, "drop_svd": drop_s,
                   "rule_x": E_act - s_act, "rule_x_weight": ms["E_weight"] - ms["s_weight"],
                   "rule_x_plainsvd": E_plain - s_act, "rule_x_G": E_G - s_G, "rule_y": drop_s - drop_m,
                   "n_query": n_query, "fit_seed": fit_seed, "fit_run": struct.source, "seed": SEED,
                   "git_hash": git["git_hash"], "git_dirty": git["git_dirty"], "prereg_commit": prereg,
                   "time_s": round(time.time() - t, 1)}
            row["rule_correct"] = bool(rule_hits([row["rule_x"]], [row["rule_y"]])[0])
            csv.add(row)
            log(f"block {b + 1}: s {s_act:.3f}, E_svd {E_act:.3f} (act.); drop mirror {drop_m:.2f}, "
                f"SVD {drop_s:.2f}; rule {'right' if row['rule_correct'] else 'wrong'}")
    else:
        log("all requested blocks already in the CSV")

    p2 = evaluate_p2(out)
    p2.update({"written_by": {"git": git, "seed": SEED, "args": vars(a), "env": env_info(device),
                              "date": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": round(time.time() - t0, 1)},
               "prereg_commit_this_run": prereg})
    (out / "f4_p2.json").write_text(json.dumps(p2, indent=2, default=str), encoding="utf-8")
    plot_rule(out, out / "plots" / "fig4_rule.png")
    pp = p2.get("primary_all_models_run", {})
    log(f"P2: {p2.get('verdict')}; all models run: {pp.get('n_hits')} of {pp.get('n_cases')} "
        f"({pp.get('fraction', float('nan')):.3f})")
    return p2


if __name__ == "__main__":
    main()
