#!/usr/bin/env bash
# MUST-09 heal of the gate-B baseline of a given rank (1 = best, 2 = second best) and seed, for job files.
# The method and its hyperparameters come from results/MUST-07/gate_b.json (chosen on the method-selection set only);
# seeds above 0 reuse the learning rate of that method's seed-0 run (PREREG_NEXT_PHASE.md, MUST-09). On success a
# marker runs/MUST-09/dinov2_vitb14_reg/gate_b_rank<rank>_s<seed>.done is written, which later jobs wait for.
#   bash scripts/h100_heal_gate.sh <rank> <seed>
set -euo pipefail
cd "$(dirname "$0")/.."
rank=${1:?rank (1 or 2)}; seed=${2:?seed}
G=results/MUST-07/gate_b.json
S=results/MUST-07/structure/dinov2_vitb14_reg_s0.npz
ROOT=runs/MUST-09/dinov2_vitb14_reg
method=$(python -c "import json,sys; g=json.load(open('$G')); r=sorted(g['ranking'], key=lambda x: int(x['rank'])); print(r[$rank - 1]['method'])")
extra=()
if [ "$seed" != "0" ]; then
  [ -f "$ROOT/$method/s0/metrics.json" ] || { echo "seed 0 of $method has not finished"; exit 3; }
  extra=(--lr-from "$ROOT/$method/s0")
fi
echo "gate B rank $rank = $method, seed $seed"
python scripts/must09_heal.py --from-gate-b "$G" --gate-rank "$rank" --seed "$seed" "${extra[@]}" \
  --span-structure "$S" --micro-batch 64 --curve-queries 10000 --gpu-frac 0.2
touch "$ROOT/gate_b_rank${rank}_s${seed}.done"
