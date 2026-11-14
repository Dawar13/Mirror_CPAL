#!/usr/bin/env bash
# The whole unattended run on the GPU machine, after scripts/h100_bootstrap.sh (PROJECT_NEXT_PHASE.md Section 9):
#   1. the data build (scripts/h100_data.sh) and job file A start together; A's jobs wait for the files they need;
#   2. when both MUST-12 pilot runs are measured, gate C and the epoch rule are scored (scripts/h100_gate_c.py,
#      results/MUST-12/gate_c.json). If gate C passes, job file B starts (CORE-01 grid, UP-01, UP-03) with the epoch
#      count the rule gives; with 100 epochs the pilot runs are linked in as CORE-01's seed-0 none and LR runs. If gate
#      C fails, B is not started (PREREG_NEXT_PHASE.md: report and decide with Dawar);
#   3. when A (and B, if started) are finished and no other process uses the GPU, job file Q (timings) runs alone.
# Every step is restart-safe: run the same command again after a preemption.
#   setsid nohup bash scripts/h100_overnight.sh > logs/overnight.log 2>&1 < /dev/null &
#   tail -f logs/overnight.log
# Settings: MW [8] neuron-matching processes per job, PAR_A [5], PAR_B [3], DATA_ARGS [--optional].
set -u
cd "$(dirname "$0")/.." || exit 1
[ -f "$HOME/.mirror_env.sh" ] && . "$HOME/.mirror_env.sh"
export MW="${MW:-8}"
mkdir -p logs
say() { echo "[$(date -u '+%F %T')] $*"; }

# PREREG_NEXT_PHASE.md must be committed before any GPU job (gate A).
if [ ! -f PREREG_NEXT_PHASE.md ] || head -1 PREREG_NEXT_PHASE.md | grep -q "DRAFT"; then
  say "PREREG_NEXT_PHASE.md is missing or still a draft: gate A not met, nothing started"; exit 2
fi

say "data build and job file A start"
nohup bash scripts/h100_data.sh ${DATA_ARGS:---optional} > logs/h100_data.log 2>&1 < /dev/null &
PAR=${PAR_A:-5} nohup bash scripts/h100_sched.sh jobs/h100_A_phase1.txt > logs/q_A.log 2>&1 < /dev/null &
pidA=$!

pidB=""
LRM=results/MUST-12/vit_s8_LR_s0     # final checkpoint e100 is measured first (scripts/h100_measure_run.sh)
until ls "$LRM"/e1*/metrics.json > /dev/null 2>&1 && [ -f runs/MUST-12/vit_s8_none_s0/last_done.flag ]; do
  if ! kill -0 "$pidA" 2> /dev/null; then say "job file A ended before the MUST-12 measurements: gate C not scored"; break; fi
  sleep 120
done
if ls "$LRM"/e1*/metrics.json > /dev/null 2>&1; then
  if python scripts/h100_gate_c.py; then
    E=$(python -c "import json; print(json.load(open('results/MUST-12/gate_c.json'))['core01_epochs'])")
    export CORE_EPOCHS="$E"
    if [ "$E" = "100" ]; then
      mkdir -p runs/CORE-01
      for c in none LR; do
        [ -e "runs/CORE-01/vit_s8_${c}_s0" ] || ln -s "../MUST-12/vit_s8_${c}_s0" "runs/CORE-01/vit_s8_${c}_s0"
      done
    fi
    say "gate C passed; job file B starts with CORE_EPOCHS=$E"
    PAR=${PAR_B:-3} nohup bash scripts/h100_sched.sh jobs/h100_B_core.txt > logs/q_B.log 2>&1 < /dev/null &
    pidB=$!
  else
    say "gate C FAILED or could not be scored (results/MUST-12/gate_c.json): job file B not started"
  fi
fi

while kill -0 "$pidA" 2> /dev/null || { [ -n "$pidB" ] && kill -0 "$pidB" 2> /dev/null; }; do sleep 120; done
say "job files A and B finished"
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2> /dev/null)" ]; do sleep 60; done
say "GPU quiet: job file Q (timings) starts"
PAR=1 bash scripts/h100_sched.sh jobs/h100_Q_latency.txt > logs/q_Q.log 2>&1 < /dev/null
say "OVERNIGHT_DONE"
# Optional self-pause (Jarvis Labs): only with AUTO_PAUSE=1, the jl CLI installed and JL_API_KEY and JL_MACHINE_ID set
# by the owner. Pause keeps everything under /home (results, runs, caches, venv) and stops compute billing; this
# script never destroys an instance. A grace period leaves time to pull results first.
if [ "${AUTO_PAUSE:-0}" = "1" ] && command -v jl > /dev/null && [ -n "${JL_MACHINE_ID:-}" ]; then
  say "auto-pause in ${PAUSE_GRACE_MIN:-30} min (touch logs/NO_PAUSE to cancel)"
  sleep $(( ${PAUSE_GRACE_MIN:-30} * 60 ))
  if [ -f logs/NO_PAUSE ]; then say "auto-pause cancelled (logs/NO_PAUSE)"; else
    say "pausing instance $JL_MACHINE_ID"; jl pause "$JL_MACHINE_ID" --yes; fi
fi
