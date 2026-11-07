#!/usr/bin/env bash
# E1b on the VM: the three pre-registered arms one after another (the GPU is shared with another
# project), each started only when enough GPU memory is free, then the extra plots for each.
#   HF_HOME=... nohup bash scripts/vm_run_e1b.sh > ~/e1b.log 2>&1 &
set -o pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
: "${HF_HOME:?set HF_HOME first}"
NEED_MB=${NEED_MB:-12000}

echo "== B0 tests $(date)"
mkdir -p runs
python -m pytest -q 2>&1 | tail -3 | tee runs/e1b_pytest.txt
grep -q " passed" runs/e1b_pytest.txt && ! grep -q "failed" runs/e1b_pytest.txt || exit 9

wait_for_gpu() {
  for i in $(seq 1 240); do   # up to 4 hours
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$free" -ge "$NEED_MB" ] && { echo "gpu free ${free} MiB"; return 0; }
    [ $((i % 10)) -eq 1 ] && echo "waiting for GPU memory: ${free} MiB free, need ${NEED_MB} $(date)"
    sleep 60
  done
  return 1
}

for arm in e1b_perlayer_b14 e1b_perlayer_b14reg e1b_shared16_b14; do
  mkdir -p runs/$arm/results
  wait_for_gpu || { echo "== SKIPPED $arm (no GPU memory) $(date)"; continue; }
  echo "== start $arm $(date)"
  if python scripts/run_e1.py --config configs/$arm.yaml --out-root runs/$arm > runs/$arm/results/e1_log.txt 2>&1 \
     && python scripts/plot_e1_extra.py --out-root runs/$arm > runs/$arm/results/extra_plots_log.txt 2>&1; then
    echo "== done $arm $(date)"; grep -E "OUTCOME" runs/$arm/results/e1_log.txt
  else
    echo "== FAILED $arm $(date)"; tail -5 runs/$arm/results/e1_log.txt
  fi
done
echo "== E1B_ALL_DONE $(date)"
