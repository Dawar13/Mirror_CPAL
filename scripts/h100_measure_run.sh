#!/usr/bin/env bash
# Structure measurement (scripts/measure_groups.py) of every checkpoint of one controlled-training run, newest first
# (gate C and the CORE-01 tests read the final checkpoint; the earlier ones give the emergence curves),
# for the groups LR, UD, R90 and Klein (CORE-01, MUST-12; PROJECT_NEXT_PHASE.md Section 4.4). Finished checkpoints
# (metrics.json present) are skipped, so the job can be rerun after a preemption. Twin-fraction nulls are not drawn for
# controlled runs (the CORE-01 tests compare runs with each other and against the weight null).
#   bash scripts/h100_measure_run.sh <run_dir> <experiment> [groups]
#   e.g. bash scripts/h100_measure_run.sh runs/CORE-01/vit_s8_UD_s0 CORE-01
set -euo pipefail
cd "$(dirname "$0")/.."
run=${1:?run directory}; exp=${2:?experiment ID}; groups=${3:-LR,UD,R90,Klein}
name=$(basename "$run")
for ck in $(ls "$run"/ckpt_e*.pt 2>/dev/null | sort -r); do
  ep=$(basename "$ck" .pt); ep=${ep#ckpt_}
  out="results/$exp/$name/$ep"
  if [ -f "$out/metrics.json" ]; then echo "skip $out"; continue; fi
  echo "measure $ck -> $out"
  python scripts/measure_groups.py --checkpoint "$ck" --arch controlled --groups "$groups" --twin-null-draws 0 \
    --workers "${MW:-8}" --experiment "$exp" --out "$out"
done
