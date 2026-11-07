#!/usr/bin/env bash
# E1b on the VM: the three pre-registered arms in parallel (each capped at 20% of GPU memory),
# then the extra plots for each.  HF_HOME=... nohup bash scripts/vm_run_e1b.sh > ~/e1b.log 2>&1 &
set -o pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
: "${HF_HOME:?set HF_HOME first}"

echo "== B0 tests $(date)"
mkdir -p runs
python -m pytest -q 2>&1 | tail -3 | tee runs/e1b_pytest.txt
grep -q " passed" runs/e1b_pytest.txt && ! grep -q "failed" runs/e1b_pytest.txt || exit 9

pids=()
for arm in e1b_perlayer_b14 e1b_perlayer_b14reg e1b_shared16_b14; do
  mkdir -p runs/$arm/results
  echo "== start $arm $(date)"
  ( python scripts/run_e1.py --config configs/$arm.yaml --out-root runs/$arm > runs/$arm/results/e1_log.txt 2>&1 \
      && python scripts/plot_e1_extra.py --out-root runs/$arm > runs/$arm/results/extra_plots_log.txt 2>&1 \
      && echo "== done $arm $(date)" || echo "== FAILED $arm $(date)" ) &
  pids+=($!)
  sleep 20   # stagger start-up (model load, data loader workers)
done
wait "${pids[@]}"
echo "== E1B_ALL_DONE $(date)"
