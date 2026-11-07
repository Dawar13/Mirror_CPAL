"""E1 decision table, plots and the markdown report.

Aggregation over "blocks 1 to 6" (approved 2026-11-07):
  rows 1 and 2  mean over layers 1..6 (layer l = output of block l) of relative held-out errors
                e / ||Xt||^2, then the ratio of those means; every layer's value and the worst
                layer are reported as well;
  row 3         d- / d of the shared operator; below 0.1 blocks GO but does not trigger KILL;
  row 4         mean over blocks 1..6 of s1 and of s2 separately; the check uses the worse (max).
                GO also needs evidence that the structure is not trivial: d-/d >= 0.2 or a mean
                twin-neuron fraction >= evidence_twin_frac. Shares are shown next to a matched null;
  row 5         k-NN top-1 at k = 6 (surgery on blocks 1..6).
Outcome, in order: KILL if any KILL cell in rows 1, 4 or 5; GO if every row is GO;
"PIVOT: structure but little to save" if d-/d < 0.1; PIVOT if rows 1 to 4 are GO and the surgery
drop is above 10 points; otherwise AMBER (the table gives no verdict).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

GO, KILL, AMBER, INFO, MISSING = "GO", "KILL", "AMBER", "INFO", "MISSING"
PIVOT, PIVOT_LITTLE = "PIVOT", "PIVOT: structure but little to save"

# Categorical slots 1 to 3 of the reference palette (validated, light surface), plus neutral ink.
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_NULL, C_INK, C_MUTED, C_GRID = "#8a8985", "#0b0b0b", "#52514e", "#e4e3df"


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float, np.floating)) and math.isfinite(float(x))


def _ratio(a, b):
    if not (_finite(a) and _finite(b)):
        return float("nan")
    return float(a) / float(b) if b != 0 else float("inf")


def decide(layers_df, mlp_df, surgery_df, d_minus_frac: float, dcfg: dict) -> dict:
    lo, hi = int(dcfg["blocks"][0]), int(dcfg["blocks"][1])
    L = layers_df[(layers_df["layer"] >= lo) & (layers_df["layer"] <= hi)]
    M = mlp_df[(mlp_df["block"] >= lo) & (mlp_df["block"] <= hi)]
    rows = []

    e_A = float(L["rel_e_A_shared"].mean())
    e_null = float(L["rel_e_null_shared"].mean())
    e_I = float(L["rel_e_I"].mean())

    # Per-layer values behind rows 1 and 2, and the worst layer of each.
    per_layer = []
    for _, r in L.iterrows():
        per_layer.append({"layer": int(r["layer"]),
                          "r1": _ratio(r["rel_e_null_shared"], r["rel_e_A_shared"]),
                          "r2": _ratio(r["rel_e_A_shared"], r["rel_e_I"])})
    ok1 = [p for p in per_layer if not math.isnan(p["r1"])]
    ok2 = [p for p in per_layer if not math.isnan(p["r2"])]
    worst1 = min(ok1, key=lambda p: p["r1"]) if ok1 else None   # row 1: lower ratio is worse
    worst2 = max(ok2, key=lambda p: p["r2"]) if ok2 else None   # row 2: higher ratio is worse

    def worst_txt(w, key):
        return f"; worst layer {w['layer']}: {_fmt(w[key], '.3g')}" if w else ""

    r1 = _ratio(e_null, e_A)
    cell1 = GO if r1 >= dcfg["e_null_ratio_go"] else KILL if r1 < dcfg["e_null_ratio_kill"] else AMBER
    rows.append({"row": 1, "check": f"e(null) / e(A_shared), mean over layers{worst_txt(worst1, 'r1')}",
                 "value": r1, "go": f">= {dcfg['e_null_ratio_go']:g}", "kill": f"< {dcfg['e_null_ratio_kill']:g}",
                 "cell": cell1 if _finite(r1) or r1 == float("inf") else MISSING})

    r2 = _ratio(e_A, e_I)
    cell2 = GO if _finite(r2) and r2 <= dcfg["e_identity_ratio_go"] else INFO
    rows.append({"row": 2, "check": f"e(A_shared) / e(I), mean over layers{worst_txt(worst2, 'r2')}",
                 "value": r2, "go": f"<= {dcfg['e_identity_ratio_go']:g}", "kill": "informative only",
                 "cell": cell2})

    cell3 = GO if d_minus_frac >= dcfg["dminus_go"] else KILL if d_minus_frac < dcfg["dminus_kill"] else AMBER
    rows.append({"row": 3, "check": "d- / d (shared operator)", "value": d_minus_frac,
                 "go": f">= {dcfg['dminus_go']:g}",
                 "kill": f"< {dcfg['dminus_kill']:g}: blocks GO, no KILL (PIVOT: little to save)",
                 "cell": cell3})

    s1, s2 = float(M["s1"].mean()), float(M["s2"].mean())
    s1n, s2n = float(M["s1_null"].mean()), float(M["s2_null"].mean())
    s = max(s1, s2)
    twin = float(M["twin_frac"].mean())
    evidence = d_minus_frac >= dcfg["evidence_dminus"] or twin >= dcfg["evidence_twin_frac"]
    if not _finite(s):
        cell4 = MISSING
    elif s > dcfg["share_kill"]:
        cell4 = KILL
    elif s <= dcfg["share_go"] and evidence:
        cell4 = GO
    else:
        cell4 = AMBER
    note4 = "" if evidence or s > dcfg["share_go"] else "; low share is NOT evidence here (structure near trivial)"
    rows.append({"row": 4, "check": f"MLP breaking share, max(mean s1, mean s2) [s1 {s1:.3f}, s2 {s2:.3f}; "
                                    f"matched null s1 {s1n:.3f}, s2 {s2n:.3f}; twin fraction {twin:.3f}, "
                                    f"d-/d {d_minus_frac:.3f}{note4}]",
                 "value": s,
                 "go": f"<= {dcfg['share_go']:g}, and d-/d >= {dcfg['evidence_dminus']:g} or twin fraction >= "
                       f"{dcfg['evidence_twin_frac']:g}",
                 "kill": f"> {dcfg['share_kill']:g}", "cell": cell4})

    k = int(dcfg["surgery_k"])
    def top1(variant, kk):
        sel = surgery_df[(surgery_df["variant"] == variant) & (surgery_df["k"] == kk)]
        return float(sel["top1"].iloc[0]) if len(sel) else float("nan")

    orig, surg, svd = top1("original", 0), top1("surgery", k), top1("svd", k)
    drop, margin = orig - surg, surg - svd
    if not (_finite(drop) and _finite(margin)):
        cell5 = MISSING
    elif margin <= 0:
        cell5 = KILL
    elif drop <= dcfg["surgery_drop_go"] and margin >= dcfg["surgery_margin_go"]:
        cell5 = GO
    else:
        cell5 = AMBER
    rows.append({"row": 5, "check": f"Zero-shot surgery at k = {k}: k-NN drop [original {orig:.2f}, surgery "
                                    f"{surg:.2f}, SVD {svd:.2f}; margin over SVD {margin:.2f}]",
                 "value": drop, "go": f"drop <= {dcfg['surgery_drop_go']:g} and margin >= "
                                      f"{dcfg['surgery_margin_go']:g}",
                 "kill": "no better than SVD (margin <= 0)", "cell": cell5})

    cells = {r["row"]: r["cell"] for r in rows}
    kill_rows = [i for i in (1, 4, 5) if cells[i] == KILL]
    open_rows = [str(i) for i, c in cells.items() if c != GO and i != 3]
    if kill_rows:
        outcome, reason = KILL, f"KILL cell in row(s) {', '.join(map(str, kill_rows))}"
    elif all(c == GO for c in cells.values()):
        outcome, reason = GO, "every check is in its GO cell"
    elif cells[3] == KILL:
        outcome = PIVOT_LITTLE
        reason = (f"d-/d = {d_minus_frac:.3f} is below {dcfg['dminus_kill']:g}, which blocks GO but does not "
                  "trigger KILL" + (f"; also not GO: row(s) {', '.join(open_rows)}" if open_rows else ""))
    elif all(cells[i] == GO for i in (1, 2, 3, 4)) and _finite(drop) and drop > dcfg["surgery_drop_go"]:
        outcome, reason = PIVOT, (f"structure checks pass but the surgery drop is {drop:.2f} points "
                                  f"(above {dcfg['surgery_drop_go']:g}); run the 1-hour healing check")
    else:
        open_all = [str(i) for i, c in cells.items() if c != GO]
        outcome, reason = AMBER, (f"no rule fires: row(s) {', '.join(open_all)} not GO and no KILL cell "
                                  "in rows 1, 4, 5; needs a human decision")
    return {"rows": rows, "outcome": outcome, "reason": reason, "per_layer": per_layer,
            "worst": {"row1": worst1, "row2": worst2},
            "numbers": {"e_A_shared": e_A, "e_null_shared": e_null, "e_I": e_I, "s1": s1, "s2": s2,
                        "s1_null": s1n, "s2_null": s2n, "twin_frac": twin, "evidence": bool(evidence),
                        "top1_original": orig, "top1_surgery": surg, "top1_svd": svd,
                        "drop": drop, "margin_over_svd": margin, "d_minus_frac": d_minus_frac}}


# ----------------------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------------------
def _style(ax):
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_MUTED)
    ax.tick_params(colors=C_MUTED, labelsize=9)
    ax.xaxis.label.set_color(C_INK)
    ax.yaxis.label.set_color(C_INK)


def plot_equivariance(layers_df, path: Path, title_suffix: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = layers_df["layer"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor="#fcfcfb")
    ax = axes[0]
    ax.set_facecolor("#fcfcfb")
    series = [("rel_e_A_layer", "per-layer A_l", C_BLUE, "o", "-"),
              ("rel_e_A_shared", "shared A", C_ORANGE, "s", "-"),
              ("rel_e_I", "identity", C_AQUA, "^", "-"),
              ("rel_e_null_shared", "random involution (null)", C_NULL, "", "--")]
    for col, label, color, marker, ls in series:
        y = layers_df[col].to_numpy()
        ax.plot(x, y, ls, color=color, marker=marker or None, markersize=5, linewidth=2, label=label)
    ax.set_yscale("log")
    from matplotlib.ticker import FuncFormatter

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlabel("layer (0 = embeddings, l = output of block l)")
    ax.set_ylabel("held-out relative error  ||X~ - XA||² / ||X~||²")
    ax.set_title("Mirror operator error by depth" + title_suffix, fontsize=10, color=C_INK, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)

    ax = axes[1]
    ax.set_facecolor("#fcfcfb")
    ax.plot(x, layers_df["agree_shared"], "-", color=C_BLUE, marker="o", markersize=5, linewidth=2,
            label="agreement tr(A_l A_shared) / d")
    ax.plot(x, layers_df["dminus_frac"], "-", color=C_ORANGE, marker="s", markersize=5, linewidth=2,
            label="d- / d of A_l")
    ax.plot(x, layers_df["agree_null"], "--", color=C_NULL, linewidth=2, label="agreement with null")
    ax.set_xlabel("layer")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Operator agreement and odd share", fontsize=10, color=C_INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_breaking_share(mlp_df, path: Path, dcfg: dict, title_suffix: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = mlp_df["block"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor="#fcfcfb")
    ax = axes[0]
    ax.set_facecolor("#fcfcfb")
    ax.plot(x, mlp_df["s1"], "-", color=C_BLUE, marker="o", markersize=5, linewidth=2, label="fc1 (s1)")
    ax.plot(x, mlp_df["s2"], "-", color=C_ORANGE, marker="s", markersize=5, linewidth=2, label="fc2 (s2)")
    ax.plot(x, mlp_df["s1_null"], "--", color=C_NULL, linewidth=2, label="matched null, fc1")
    ax.plot(x, mlp_df["s2_null"], ":", color=C_NULL, linewidth=2, label="matched null, fc2")
    for thr, name in ((dcfg["share_go"], "GO"), (dcfg["share_kill"], "KILL")):
        ax.axhline(thr, color=C_MUTED, linewidth=0.8, linestyle=(0, (2, 3)))
        ax.annotate(f"{name} {thr:g}", (x.min(), thr), textcoords="offset points", xytext=(0, 3),
                    fontsize=8, color=C_MUTED)
    ax.set_xlabel("block")
    ax.set_ylabel("breaking share ||W - W_eq||² / ||W||²")
    ax.set_ylim(0, max(0.6, float(np.nanmax(mlp_df[["s1", "s2", "s1_null", "s2_null"]].to_numpy())) * 1.1))
    ax.set_title("MLP symmetry-breaking share by depth" + title_suffix, fontsize=10, color=C_INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    _style(ax)

    ax = axes[1]
    ax.set_facecolor("#fcfcfb")
    ax.plot(x, mlp_df["twin_frac"], "-", color=C_BLUE, marker="o", markersize=5, linewidth=2,
            label="twin-neuron fraction (neurons in pairs / all)")
    ax.plot(x, mlp_df["dminus_frac"], "-", color=C_ORANGE, marker="s", markersize=5, linewidth=2,
            label="d- / d (shared operator)")
    ax.plot(x, mlp_df["param_saving_frac"], "-", color=C_AQUA, marker="^", markersize=5, linewidth=2,
            label="implied MLP parameter saving")
    ax.axhline(dcfg["evidence_dminus"], color=C_MUTED, linewidth=0.8, linestyle=(0, (2, 3)))
    ax.annotate(f"evidence {dcfg['evidence_dminus']:g}", (x.min(), dcfg["evidence_dminus"]),
                textcoords="offset points", xytext=(0, 3), fontsize=8, color=C_MUTED)
    ax.set_xlabel("block")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Twin structure and odd share", fontsize=10, color=C_INK, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Markdown
# ----------------------------------------------------------------------------------------------
def _fmt(v, spec: str) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if math.isnan(v):
            return "n/a"
        if math.isinf(v):
            return "inf"
        return format(float(v), spec)
    return str(v)


def md_table(df, columns: list[tuple[str, str, str]]) -> str:
    """columns: (column name, header, format spec)."""
    head = "| " + " | ".join(h for _, h, _ in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    lines = [head, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c], f) for c, _, f in columns) + " |")
    return "\n".join(lines)


def write_report(path: Path, ctx: dict) -> None:
    d = ctx["decision"]
    meta = ctx["meta"]
    lines = []
    lines.append("# E1 report: Mirror Surgery go/kill on " + ctx["model_name"])
    lines.append("")
    if ctx.get("tiny"):
        lines.append("> **TINY TEST RUN.** Random tiny ViT and synthetic images. The numbers below test the "
                     "pipeline only and say nothing about DINOv2.")
        lines.append("")
    lines.append(f"**Outcome: {d['outcome']}.** {d['reason'][0].upper() + d['reason'][1:]}.")
    lines.append("")
    lines.append("## Run")
    lines.append("")
    lines.append(f"- Date: {meta['date']}; git commit `{meta['git']['git_hash']}`"
                 f"{' (uncommitted changes present)' if meta['git']['git_dirty'] else ''}; seed {meta['seed']}.")
    lines.append(f"- Config: `{meta['config_path']}`; device {meta['env'].get('gpu', meta['env']['device'])}; "
                 f"torch {meta['env']['torch']}.")
    lines.append(f"- Data: fit {meta['split_sizes']['fit']}, held-out {meta['split_sizes']['heldout']}, "
                 f"k-NN bank {meta['split_sizes']['bank']}, queries {meta['split_sizes']['query']} images "
                 "(each with its mirror).")
    lines.append(f"- Total time {meta['timing']['total_s'] / 60:.1f} min. Stages (s): "
                 + ", ".join(f"{k} {v:.0f}" for k, v in meta["timing"]["stages"].items()) + ".")
    lines.append("")

    lo, hi = ctx["decision_blocks"]
    lines.append(f"## Decision table (blocks {lo} to {hi})")
    lines.append("")
    lines.append("| Row | Check | Value | GO if | KILL if | Cell |")
    lines.append("|---|---|---|---|---|---|")
    for r in d["rows"]:
        lines.append(f"| {r['row']} | {r['check']} | {_fmt(r['value'], '.4g')} | {r['go']} | {r['kill']} | "
                     f"**{r['cell']}** |")
    lines.append("")
    w1, w2 = d["worst"]["row1"], d["worst"]["row2"]
    lines.append(f"Per-layer values behind rows 1 and 2 (layer l is the output of block l). Worst layer: row 1 "
                 f"layer {w1['layer'] if w1 else 'n/a'}, row 2 layer {w2['layer'] if w2 else 'n/a'}.")
    lines.append("")
    lines.append("| Layer | e(null) / e(A_shared) | e(A_shared) / e(I) |")
    lines.append("|---|---|---|")
    for p in d["per_layer"]:
        m1 = " (worst)" if w1 and p["layer"] == w1["layer"] else ""
        m2 = " (worst)" if w2 and p["layer"] == w2["layer"] else ""
        lines.append(f"| {p['layer']} | {_fmt(p['r1'], '.4g')}{m1} | {_fmt(p['r2'], '.4g')}{m2} |")
    lines.append("")
    lines.append(f"How blocks are combined: rows 1 and 2 use the mean over layers {lo} to {hi} of the relative "
                 f"held-out error e / ||X~||², then the ratio of those means. Row 4 averages s1 and s2 over "
                 f"blocks {lo} to {hi} and uses the larger; a low share counts as evidence only if d-/d is at "
                 f"least {ctx['evidence_dminus']:g} or the mean twin-neuron fraction is at least "
                 f"{ctx['evidence_twin_frac']:g}. Row 5 uses surgery on the first {ctx['surgery_k']} blocks. "
                 "Outcome, in order: KILL if any KILL cell in rows 1, 4 or 5; GO if all rows are GO; "
                 "PIVOT: structure but little to save, if d-/d is below the row 3 KILL limit; PIVOT if rows 1 "
                 "to 4 are GO and the drop is above the GO limit; otherwise AMBER (the table gives no verdict).")
    lines.append("")

    lines.append("## Residual stream: one mirror operator?")
    lines.append("")
    lines.append(f"Shared operator fitted on summed C over layers {ctx['shared_layers'][0]} to "
                 f"{ctx['shared_layers'][1]}: d = {ctx['d']}, d+ = {ctx['d_plus_shared']}, "
                 f"d- = {ctx['d_minus_shared']} (d-/d = {ctx['d_minus_shared'] / ctx['d']:.3f}). "
                 f"Directions with no signal (assigned +1): {ctx['n_zero_shared']}. Errors are relative, "
                 "e / ||X~||², on held-out images; the null is a random orthogonal involution with the same "
                 f"d+ (mean of {ctx['n_null']} seeds).")
    lines.append("")
    lines.append(md_table(ctx["layers_df"], [
        ("layer", "layer", "d"), ("d_minus", "d- (A_l)", "d"), ("agree_shared", "tr(A_l A)/d", ".3f"),
        ("rel_e_A_layer", "e(A_l)", ".4g"), ("rel_e_A_shared", "e(A)", ".4g"),
        ("rel_e_A_shared_norm", "e(A, scale-normalised fit)", ".4g"), ("rel_e_I", "e(I)", ".4g"),
        ("rel_e_null_shared", "e(null)", ".4g"), ("ratio_null_over_shared", "e(null)/e(A)", ".3g"),
    ]))
    lines.append("")
    lines.append("![equivariance vs depth](../plots/equivariance_vs_depth.png)")
    lines.append("")

    pos = ctx["pos"]
    lines.append("## Position embedding")
    lines.append("")
    lines.append(f"||mirror(P) - P A|| / ||P|| on the {ctx['grid'][0]} x {ctx['grid'][1]} patch grid "
                 "(P channel-centred like the tokens; the second column also removes the mean over positions).")
    lines.append("")
    lines.append("| Operator | channel-centred | also grid-centred |")
    lines.append("|---|---|---|")
    for name in pos["channel_centred"]:
        lines.append(f"| {name} | {pos['channel_centred'][name]:.4g} | {pos['grid_centred'][name]:.4g} |")
    lines.append("")
    lines.append(f"Layer 0 held-out error with the shared operator: e(A) = "
                 f"{float(ctx['layers_df'].iloc[0]['rel_e_A_shared']):.4g} (e(A_0) = "
                 f"{float(ctx['layers_df'].iloc[0]['rel_e_A_layer']):.4g}, e(I) = "
                 f"{float(ctx['layers_df'].iloc[0]['rel_e_I']):.4g}).")
    lines.append("")

    lines.append("## MLP twins and breaking share")
    lines.append("")
    lines.append("Neuron involution per block from the mirrored-versus-original fc1 correlations; breaking "
                 "share of the folded weights against the shared operator. Matched null (mean of "
                 f"{ctx['n_null']} seeds): a random involution with the same d+ and d- as the shared operator "
                 "and, per block, a random pairing with the same numbers of pairs and self-twins as the fitted "
                 "one. If the fitted structure is nearly trivial (few odd features, few pairs), the matched "
                 "null is nearly trivial too and its share is also low; a low share then is not evidence. "
                 "Twin fraction = neurons in pairs / all neurons. Evidence = d-/d at least "
                 f"{ctx['evidence_dminus']:g} or twin fraction at least {ctx['evidence_twin_frac']:g}.")
    lines.append("")
    lines.append(md_table(ctx["mlp_df"], [
        ("block", "block", "d"), ("n_pairs", "pairs", "d"), ("n_self", "self-twins", "d"),
        ("twin_frac", "twin fraction", ".3f"), ("dminus_frac", "d-/d", ".3f"),
        ("mean_score_pairs", "mean corr (pairs)", ".3f"), ("mean_score_self", "mean corr (self)", ".3f"),
        ("s1", "s1", ".3f"), ("s1_null", "s1 matched null", ".3f"),
        ("s2", "s2", ".3f"), ("s2_null", "s2 matched null", ".3f"), ("evidence", "evidence", "s"),
        ("param_saving_frac", "implied saving", ".3f"), ("svd_rank", "SVD rank", "d"),
    ]))
    lines.append("")
    lines.append("![breaking share vs depth](../plots/breaking_share_vs_depth.png)")
    lines.append("")

    lines.append("## Zero-shot surgery (k-NN, k = 20, T = 0.07)")
    lines.append("")
    lines.append("Surgery replaces the MLPs of the first k blocks by their nearest mirror-equivariant weights. "
                 "SVD keeps the same number of parameters as a rank-r factorisation. Random uses the null "
                 "operator and pairing. Top-1 and flip consistency are in percent.")
    lines.append("")
    lines.append(md_table(ctx["surgery_df"], [
        ("k", "k", "d"), ("variant", "variant", "s"), ("top1", "top-1", ".2f"),
        ("drop_points", "drop (points)", ".2f"), ("flip_consistency", "flip consistency", ".2f"),
        ("mlp_params_kept_frac", "MLP params kept (first k blocks)", ".3f"),
    ]))
    lines.append("")

    s = ctx["sanity"]
    lines.append("## Sanity checks in this run")
    lines.append("")
    lines.append(f"- Folding (block 0, one fit batch): max abs difference {s['fold_max_abs_diff']:.3g}.")
    lines.append(f"- Identity surgery on all blocks (A = I, P = I), CLS features of one batch: max abs "
                 f"difference {s['identity_surgery_max_abs_diff']:.3g}.")
    lines.append("")
    lines.append("All numbers come from `scripts/run_e1.py` with the config, seed and commit above; per-layer and "
                 "per-block values are in `results/e1_layers.csv`, `results/e1_mlp.csv` and "
                 "`results/e1_surgery.csv`.")
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
