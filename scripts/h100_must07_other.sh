#!/usr/bin/env bash
# MUST-07 on a second model (DeiT-III or DINOv2-L): 6 blocks MLP only and the full model, with Mirror-exact,
# Mirror-local, plain SVD, the gate-B best and second-best baselines and random structure (the reduced set agreed for
# the non-main models; PROJECT_NEXT_PHASE.md MUST-07 makes no prediction there). Gate B is read from gate_b.json.
#   bash scripts/h100_must07_other.sh <model>
set -euo pipefail
cd "$(dirname "$0")/.."
model=${1:?model}
G=results/MUST-07/gate_b.json
pick() { python -c "import json; g=json.load(open('$G')); print(g['$1'] or '')"; }
best=$(pick best); second=$(pick second_best)
methods="mirror_exact,mirror_local,svd,random_structure"
for m in $best $second; do
  case ",$methods," in *",$m,"*) ;; *) methods="$methods,$m" ;; esac
done
echo "$model: methods $methods"
python scripts/must07_compressors.py --model "$model" --fit-seeds 0 --sizes 6mlp,full --methods "$methods" \
  --fp32-flip exact --fp32-queries 5000 --fp32-bank-per-class 10
