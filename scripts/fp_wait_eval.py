#!/usr/bin/env python
"""Wait until the evaluation server has written given passes for given run directories (a heal job ends when its
checkpoints are saved and queued; the decision jobs of jobs/final_phase.yaml must also wait for the evaluations).

    python scripts/fp_wait_eval.py --run-dir runs/final/main/dinov2_vitb14_reg/svdllm/s0 --passes bf16 [--step 780]

A pass counts as present when <run_dir>/eval/step<S>.json has a key equal to the pass name (all queries, full bank)
or, with --allow-subsets, starting with it (subsets carry _q<n> or _b<m>). The default step is the run's last
checkpoint (ckpt/step<S>.pt). Exit 0 when everything is present, 3 on timeout, 2 on a missing run directory.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path


def last_step(run: Path) -> int | None:
    steps = [int(m.group(1)) for p in (run / "ckpt").glob("step*.pt") if (m := re.fullmatch(r"step(\d+)\.pt", p.name))]
    return max(steps) if steps else None


def present(run: Path, step: int, passes: list, allow_subsets: bool) -> list:
    """The passes still missing for run/step."""
    path = run / "eval" / f"step{int(step)}.json"
    if not path.exists():
        return list(passes)
    try:
        res = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return list(passes)
    keys = set(res.get("passes", {}).keys()) | set(res.keys())
    missing = []
    for p in passes:
        ok = p in keys or (allow_subsets and any(k.startswith(p + "_") for k in keys))
        if not ok:
            missing.append(p)
    return missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", action="append", required=True)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--passes", default="bf16")
    ap.add_argument("--allow-subsets", action="store_true")
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("--timeout-hours", type=float, default=6.0)
    a = ap.parse_args(argv)
    passes = [p.strip() for p in a.passes.split(",") if p.strip()]
    runs = [Path(r) for r in a.run_dir]
    t0 = time.time()
    while True:
        waiting = {}
        for run in runs:
            if not run.exists():
                print(f"fp_wait_eval: {run} does not exist", flush=True)
                return 2
            step = a.step if a.step is not None else last_step(run)
            if step is None or not (run / "DONE").exists():
                waiting[str(run)] = "heal not finished"
                continue
            miss = present(run, step, passes, a.allow_subsets)
            if miss:
                waiting[str(run)] = f"step {step}: missing {miss}"
        if not waiting:
            print(f"fp_wait_eval: every pass present ({len(runs)} runs) after {time.time() - t0:.0f} s", flush=True)
            return 0
        if time.time() - t0 > a.timeout_hours * 3600:
            print(f"fp_wait_eval: timeout; still waiting for {json.dumps(waiting)}", flush=True)
            return 3
        time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
