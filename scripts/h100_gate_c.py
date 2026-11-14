#!/usr/bin/env python
"""Gate C and the CORE-01 epoch rule, scored mechanically on the GPU machine (PREREG_NEXT_PHASE.md, MUST-12).

Gate C: in the MUST-12 LR model (ViT-S/8, ImageNet-100, seed 0, final checkpoint), the LR ratio to null in blocks 1 to
6 is at most 0.8 and the LR twin fraction (threshold 0.5) in blocks 1 to 6 is at least 0.2.
Epoch rule: if the MUST-12 none run took over 25 minutes (wall time recorded in its meta.json), every CORE-01 run uses
60 epochs instead of 100.

Reads results/MUST-12/<run>/e<NNN>/metrics.json (scripts/measure_groups.py, newest checkpoint) and
runs/MUST-12/<run>/meta.json (scripts/train_controlled.py). Writes results/MUST-12/gate_c.json and prints one line per
item. Exit code 0 when gate C passes, 1 when it fails, 2 when an input is missing.

    python scripts/h100_gate_c.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RATIO_MAX, TWIN_MIN, MINUTES_MAX = 0.8, 0.2, 25.0


def final_metrics(results_dir: Path) -> tuple[Path, dict]:
    eps = []
    for d in results_dir.glob("e*"):
        m = re.fullmatch(r"e(\d+)", d.name)
        if m and (d / "metrics.json").exists():
            eps.append((int(m.group(1)), d))
    if not eps:
        raise FileNotFoundError(f"no measured checkpoint under {results_dir}")
    _, d = max(eps)
    return d, json.loads((d / "metrics.json").read_text())


def evaluate(lr_metrics: dict, none_meta: dict) -> dict:
    sc = (lr_metrics.get("scores") or {}).get("LR") or {}
    ratio, twin = sc.get("ratio_to_null_1-6"), sc.get("twin_fraction_t50_1-6")
    timing = ((none_meta.get("extra") or {}).get("timing") or {})
    minutes = timing.get("wall_minutes")
    ok_ratio = ratio is not None and ratio <= RATIO_MAX
    ok_twin = twin is not None and twin >= TWIN_MIN
    epochs = 60 if (minutes is not None and minutes > MINUTES_MAX) else 100
    return {"gate": "C", "pass": bool(ok_ratio and ok_twin),
            "lr_ratio_to_null_blocks_1_6": ratio, "bar_ratio_max": RATIO_MAX, "ratio_ok": bool(ok_ratio),
            "lr_twin_fraction_t50_blocks_1_6": twin, "bar_twin_min": TWIN_MIN, "twin_ok": bool(ok_twin),
            "must12_none_wall_minutes": minutes, "rule_minutes_max": MINUTES_MAX, "core01_epochs": epochs,
            "projected_minutes_100": timing.get("projected_minutes_100"),
            "steady_images_per_s": timing.get("steady_images_per_s")}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(REPO / "results" / "MUST-12"))
    p.add_argument("--runs", default=str(REPO / "runs" / "MUST-12"))
    p.add_argument("--arch", default="vit_s8")
    a = p.parse_args(argv)
    res, runs = Path(a.results), Path(a.runs)
    try:
        d, lr = final_metrics(res / f"{a.arch}_LR_s0")
        none_meta = json.loads((runs / f"{a.arch}_none_s0" / "meta.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"gate C: input missing ({e})")
        return 2
    out = evaluate(lr, none_meta)
    out.update({"lr_metrics": str(d / "metrics.json"), "none_meta": str(runs / f"{a.arch}_none_s0" / "meta.json"),
                "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    res.mkdir(parents=True, exist_ok=True)
    (res / "gate_c.json").write_text(json.dumps(out, indent=2))
    print(f"gate C {'PASS' if out['pass'] else 'FAIL'}: LR ratio to null (blocks 1-6) {out['lr_ratio_to_null_blocks_1_6']} "
          f"(bar <= {RATIO_MAX}), LR twin fraction {out['lr_twin_fraction_t50_blocks_1_6']} (bar >= {TWIN_MIN})")
    print(f"epoch rule: MUST-12 none run {out['must12_none_wall_minutes']} min (rule: > {MINUTES_MAX} min means 60) "
          f"-> CORE-01 epochs {out['core01_epochs']}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
