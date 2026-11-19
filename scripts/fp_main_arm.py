#!/usr/bin/env python
"""Print the main Mirror arm of DINOv2-B (docs/final_phase_design.md 4.5a), read from results/A01/decision.json:
mirror_exact_wspace when the A01 rule adopted A*_w, else mirror_exact_astar. Used inside job commands of
jobs/final_phase.yaml, so a job that needs the main arm does not have to exist twice.

    python scripts/fp_main_arm.py                        # mirror_exact_wspace or mirror_exact_astar
    python scripts/fp_main_arm.py --seed 1               # runs/final/main/dinov2_vitb14_reg/<arm>/s1
    python scripts/fp_main_arm.py --variant              # wspace or astar
Exit 2 (and nothing printed) while the decision file is missing or has no boolean "adopted".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def adopted(decision: Path) -> bool | None:
    try:
        d = json.loads(decision.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    v = d.get("adopted")
    return v if isinstance(v, bool) else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decision", default=str(ROOT / "results" / "A01" / "decision.json"))
    ap.add_argument("--model", default="dinov2_vitb14_reg")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--variant", action="store_true", help="print only wspace or astar")
    ap.add_argument("--root", default="runs/final/main")
    a = ap.parse_args(argv)
    v = adopted(Path(a.decision))
    if v is None:
        print(f"fp_main_arm: {a.decision} is missing or has no boolean 'adopted'", file=sys.stderr)
        return 2
    variant = "wspace" if v else "astar"
    if a.variant:
        print(variant)
    elif a.seed is None:
        print(f"mirror_exact_{variant}")
    else:
        print(f"{a.root}/{a.model}/mirror_exact_{variant}/s{a.seed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
