#!/usr/bin/env python
"""Gate G2 / P3 report for the M3 healing pilot, built only from the pilot's output files.

    python scripts/m3_gate_report.py --runs runs/m3_pilot --out results/M3_PILOT_G2.md
Thresholds are read from PREREG_E2.md's P3 values (copied here as constants; committed 9804f06).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

P3_RANDOM, P3_SVD, P3_ORIG = 2.0, 3.0, 2.0  # PREREG_E2.md P3 (commit 9804f06)
METHODS = ("mirror", "svd", "random")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/m3_pilot")
    ap.add_argument("--out", default="results/M3_PILOT_G2.md")
    a = ap.parse_args(argv)
    runs = Path(a.runs)
    knn, summ = {}, {}
    for m in METHODS:
        knn[m] = json.loads((runs / m / "knn.json").read_text())
        summ[m] = json.loads((runs / m / "summary.json").read_text())
    orig = knn["mirror"]["original"]["top1"]
    healed = {m: knn[m]["healed"]["top1"] for m in METHODS}
    checks = [
        ("healed mirror minus healed random", healed["mirror"] - healed["random"], f">= {P3_RANDOM}",
         healed["mirror"] - healed["random"] >= P3_RANDOM),
        ("healed mirror minus healed SVD", healed["mirror"] - healed["svd"], f">= {P3_SVD}",
         healed["mirror"] - healed["svd"] >= P3_SVD),
        ("original minus healed mirror", orig - healed["mirror"], f"<= {P3_ORIG}", orig - healed["mirror"] <= P3_ORIG),
    ]
    p3 = all(c[3] for c in checks)
    g2 = checks[0][3] and checks[1][3]
    L = ["# M3 healing pilot: gate G2 and P3", "",
         f"**P3: {'PASS' if p3 else 'FAIL'}. Gate G2: {'PASS' if g2 else 'FAIL'}.** "
         "Pre-registered in PREREG_E2.md (commit 9804f06); thresholds unchanged.", "",
         "Setting: DINOv2 ViT-B/14 with registers, first 9 blocks converted, MLP plus attention, V-local; "
         "healing by feature distillation from the original model on 100k ImageNet train images, 2 epochs, "
         "batch 256, AdamW, cosine schedule, bf16, learning rate chosen per method on a held-out 5k train split. "
         "k-NN: 100k train bank, 50k validation queries, k = 20, T = 0.07, bf16 features.", "",
         "| Method | Zero-shot top-1 | Healed top-1 | Healed flip consistency | Held-out loss before | after | LR | git |",
         "|---|---|---|---|---|---|---|---|",
         f"| original | {orig:.2f} | | {knn['mirror']['original']['flip_consistency']:.2f} | | | | |"]
    for m in METHODS:
        fin = summ[m].get("final", {})
        ho0 = (fin.get("heldout_initial") or {}).get("total", float("nan"))
        ho1 = (fin.get("heldout_final") or {}).get("total", float("nan"))
        L.append(f"| {m} | {knn[m]['zero_shot']['top1']:.2f} | {healed[m]:.2f} | "
                 f"{knn[m]['healed']['flip_consistency']:.2f} | {ho0:.4g} | {ho1:.4g} | "
                 f"{summ[m]['lr_selection'].get('best_lr', float('nan')):.0e} | {str(knn[m].get('git_hash', ''))[:7]} |")
    L += ["", "| Pre-registered check | Value (points) | Threshold | Result |", "|---|---|---|---|"]
    L += [f"| {n} | {v:.2f} | {t} | {'PASS' if ok else 'FAIL'} |" for n, v, t, ok in checks]
    L += ["", "Notes:",
          f"- The largest margin over healed SVD that any method could reach at this setting is "
          f"{orig - healed['svd']:.2f} points (original minus healed SVD), below the 3.0 bar.",
          "- Held-out loss is the normalised feature-distillation loss on the 5k held-out train images "
          "(lower is closer to the original model).",
          "- Per CLAUDE.md, the healing grid (M3-grid) is not started after a failed gate; the framing goes back "
          "to Dawar.", ""]
    Path(a.out).write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
