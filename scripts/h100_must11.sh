#!/usr/bin/env bash
# MUST-11 downstream tasks for the healed full models (PROJECT_NEXT_PHASE.md MUST-11): the original, healed
# Mirror-exact (all 3 seeds for the ImageNet linear probe; seed 0 for the rest) and the healed gate-B best baseline
# (seed 0). The best baseline's run folder is read from results/MUST-07/gate_b.json.
#   bash scripts/h100_must11.sh linear|transfer|ade20k
set -euo pipefail
cd "$(dirname "$0")/.."
task=${1:?linear, transfer or ade20k}
R=runs/MUST-09/dinov2_vitb14_reg
best=$(python -c "import json; print(json.load(open('results/MUST-07/gate_b.json'))['best'])")
V0="--healed-run vexact_s0=$R/mirror_exact/s0"
B0="--healed-run best_${best}_s0=$R/$best/s0"
case "$task" in
  linear)
    python scripts/tier2_linear.py --model dinov2_vitb14_reg --variants original $V0 \
      --healed-run vexact_s1=$R/mirror_exact/s1 --healed-run vexact_s2=$R/mirror_exact/s2 $B0 \
      --full-patch-view --out results/MUST-11/t2 ;;
  transfer)
    python scripts/tier2_transfer.py --model dinov2_vitb14_reg --variants original $V0 $B0 \
      --full-patch-view --out results/MUST-11/t2 ;;
  ade20k)
    python scripts/tier2_ade20k.py --model dinov2_vitb14_reg --variants original $V0 $B0 --mask-eq \
      --out results/MUST-11/t2_seg ;;
  *) echo "unknown task $task"; exit 2 ;;
esac
