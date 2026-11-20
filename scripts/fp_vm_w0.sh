#!/usr/bin/env bash
# W0 of the final phase on the H200 VM (FINAL_PHASE_PLAN.md v2, Section 8; docs/final_phase_design.md).
# Every step runs on its own and is restart-safe: a finished step leaves logs/final/w0/<step>.done and is skipped
# next time (FORCE=1 runs it again); the tools it calls resume by themselves (scripts/h100_data.sh,
# scripts/fp_cache.py). The coordinator calls the steps by hand or as W0 jobs of jobs/final_phase.yaml.
#
#   cd /home/Mirror_CPAL
#   bash scripts/fp_vm_w0.sh clock        # start the VM clock (logs/final/state/budget.json); bash only, run it first
#   read -rs HF_TOKEN && export HF_TOKEN  # typed into the ssh session, never written to the repo
#   bash scripts/fp_vm_w0.sh env          # scripts/h100_bootstrap.sh --skip-tests: venv, torch, HF login, GPU smoke
#   bash scripts/fp_vm_w0.sh heartbeat    # keeps the VM clock while no runner runs (detached; needs the manifest)
#   bash scripts/fp_vm_w0.sh gpu          # GPU check: name, memory, other processes, torch, a bf16 matmul
#   bash scripts/fp_vm_w0.sh prereg       # PREREG_FINAL_PHASE.md recorded and unchanged, else stop
#   bash scripts/fp_vm_w0.sh models       # weights under $HF_HOME: DINOv2-B/14, L/14, S/14 reg; DeiT-III-B/16; SigLIP 2 B/16
#   bash scripts/fp_vm_w0.sh data         # existing caches: val, train, msel and the disjointness check
#   bash scripts/fp_vm_w0.sh roles        # new roles fit3 and coco, the stream_train index, the role check
#   bash scripts/fp_vm_w0.sh roles_ext    # heal_ext (200k images at 256 px, for A11; about 40 GB, the long one)
#   bash scripts/fp_vm_w0.sh extra        # downstream data for A10 and P11.2: ADE20K and transfer sets, ImageNet-V2, Sketch
#   bash scripts/fp_vm_w0.sh tests        # the final-phase CPU tests on the VM (tests/test_fp_*.py)
#   bash scripts/fp_vm_w0.sh smoke        # GPU smoke test: --dry-run of scripts/fp_heal.py and scripts/fp_eval.py on the GPU
#   bash scripts/fp_vm_w0.sh timing       # one heal per model, then 3 concurrent heals -> results/final/w0_timing.json
#   bash scripts/fp_vm_w0.sh status       # which steps are done
#   bash scripts/fp_vm_w0.sh all          # clock env heartbeat gpu prereg models data roles tests smoke (stops at a failure)
#
# Settings (environment, defaults in brackets): HF_HOME [/home/hf], WORKERS [CPUs - 2], MANIFEST
# [jobs/final_phase.yaml], DELETE_SHARDS [1] (delete shards a role build downloaded, once decoded), BOOTSTRAP_ARGS,
# MODELS [dinov2_vitb14_reg,dinov2_vitl14_reg,dinov2_vits14_reg,deit3_b16,siglip2_b16],
# SMOKE_HEAL_CMD, SMOKE_EVAL_CMD (defaults below; flags to confirm against the heal and eval scripts (verify)),
# SMOKE_EXTRA_CMD (for example the A12b kernel's GPU correctness test), TIMING_CMD (a template with {model},
# {seed}, {steps} and {out}), TIMING_MODELS [dinov2_vitb14_reg deit3_b16 dinov2_vitl14_reg], TIMING_STEPS [780],
# FULL_STEPS [780], TIMING_UNITS [dinov2_vitb14_reg=1,deit3_b16=1,dinov2_vitl14_reg=3] (heal-units of one full heal,
# FINAL_PHASE_PLAN.md Section 9).
# Outputs: logs/final/w0/<step>.log and .done, results/data_check/models_final.json,
# results/data_check/final_roles.json, results/final/w0_timing.json.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
W0="$ROOT/logs/final/w0"; STATE="$ROOT/logs/final/state"
mkdir -p "$W0" "$STATE"
[ -f "$HOME/.mirror_env.sh" ] && . "$HOME/.mirror_env.sh"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then . .venv/bin/activate; fi
export HF_HOME="${HF_HOME:-/home/hf}"
PY="$(command -v python || command -v python3)"
ncpu=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc 2>/dev/null || echo 4)   # GNU nproc honours OMP_NUM_THREADS (the runner sets 4)
WORKERS="${WORKERS:-$(( ncpu > 3 ? ncpu - 2 : 1 ))}"
MANIFEST="${MANIFEST:-jobs/final_phase.yaml}"
DELETE_SHARDS="${DELETE_SHARDS:-1}"
MODELS="${MODELS:-dinov2_vitb14_reg,dinov2_vitl14_reg,dinov2_vits14_reg,deit3_b16,siglip2_b16}"
SMOKE_HEAL_CMD="${SMOKE_HEAL_CMD:-python scripts/fp_heal.py --dry-run --device cuda --out runs/final/_smoke/heal}"
SMOKE_EVAL_CMD="${SMOKE_EVAL_CMD:-python scripts/fp_eval.py --dry-run --device cuda --out runs/final/_smoke/eval}"
# not ${TIMING_CMD:-default}: the default's first } (in {model}) would end the expansion and append the rest
[ -n "${TIMING_CMD:-}" ] || TIMING_CMD='python scripts/fp_heal.py --model {model} --methods mirror_exact --seed {seed} --max-steps {steps} --out {out}'
TIMING_MODELS="${TIMING_MODELS:-dinov2_vitb14_reg deit3_b16 dinov2_vitl14_reg}"
TIMING_STEPS="${TIMING_STEPS:-780}"; FULL_STEPS="${FULL_STEPS:-780}"
TIMING_UNITS="${TIMING_UNITS:-dinov2_vitb14_reg=1,deit3_b16=1,dinov2_vitl14_reg=3}"
STEPS_ALL=(clock env heartbeat gpu prereg models data roles tests smoke)
STEPS_KNOWN=(clock env heartbeat gpu prereg models data roles roles_ext extra tests smoke timing)

say() { echo "[$(date -u '+%F %T')] $*"; }

run_step() {   # $1 step name, then the function to run; skips a done step unless FORCE=1
  local name=$1 fn=$2 t0 rc
  if [ -f "$W0/$name.done" ] && [ "${FORCE:-0}" != 1 ]; then
    say "$name: done already ($(head -1 "$W0/$name.done")); FORCE=1 runs it again"; return 0
  fi
  say "$name: start (log logs/final/w0/$name.log)"
  t0=$(date +%s)
  { echo "== $name $(date -u '+%F %T')"; } >> "$W0/$name.log"
  "$fn" >> "$W0/$name.log" 2>&1; rc=$?
  if [ $rc -eq 0 ]; then
    echo "ok $(date -u '+%F %T') in $(( $(date +%s) - t0 )) s" > "$W0/$name.done"
    say "$name: ok in $(( $(date +%s) - t0 )) s"
  else
    say "$name: FAILED rc=$rc after $(( $(date +%s) - t0 )) s; last lines of logs/final/w0/$name.log:"
    tail -15 "$W0/$name.log" | sed 's/^/    /'
  fi
  return $rc
}

# ---------------------------------------------------------------------------------------------------
step_clock() {   # bash only: the runner and fp_runner.py heartbeat continue this file
  if [ -f "$STATE/budget.json" ]; then echo "VM clock already started: $(cat "$STATE/budget.json")"; return 0; fi
  local now; now=$(date +%s)
  printf '{"w0_start_epoch": %s, "w0_start_utc": "%s", "counted_s": 0, "last_seen_epoch": %s, "boot_epoch": null, "pauses": [], "adjust_s": 0, "source": "scripts/fp_vm_w0.sh"}\n' \
    "$now" "$(date -u '+%F %T')" "$now" > "$STATE/budget.json"
  echo "VM clock started at $(date -u '+%F %T') UTC: $STATE/budget.json"
}

step_env() {
  # shellcheck disable=SC2086
  bash scripts/h100_bootstrap.sh --skip-tests --hf-home "$HF_HOME" ${BOOTSTRAP_ARGS:-}
}

step_heartbeat() {
  [ -f "$MANIFEST" ] || { echo "no $MANIFEST yet: start the heartbeat once the coordinator has written it"; return 1; }
  if [ -f "$STATE/heartbeat.lock" ] && command -v flock > /dev/null && ! flock -n "$STATE/heartbeat.lock" true; then
    echo "a heartbeat already runs"; return 0
  fi
  setsid nohup "$PY" scripts/fp_runner.py heartbeat "$MANIFEST" --interval 60 > "$ROOT/logs/final/heartbeat.out" 2>&1 < /dev/null &
  sleep 3
  "$PY" scripts/fp_runner.py clock "$MANIFEST"
}

step_gpu() {
  command -v nvidia-smi > /dev/null || { echo "nvidia-smi not found"; return 1; }
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version,utilization.gpu --format=csv
  local name apps
  name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  case "$name" in *H200*) echo "GPU: $name (as planned)" ;; *) echo "WARNING: expected an H200, found $name" ;; esac
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2> /dev/null)
  if [ -n "$apps" ]; then echo "WARNING: processes already on the GPU (the plan assumes it is ours alone):"; echo "$apps"; fi
  df -h "$ROOT" "$HF_HOME" | awk '!seen[$0]++'
  free -g | sed -n '1,2p'
  "$PY" - << 'PYEOF'
import time
import torch

assert torch.cuda.is_available(), "torch cannot see the GPU"
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, {p.name}, {p.total_memory / 2 ** 30:.1f} GiB, "
      f"capability {p.major}.{p.minor}")
x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    y = x @ x
torch.cuda.synchronize()
print(f"bf16 matmul: {2 * 8192 ** 3 * 10 / (time.time() - t0) / 1e12:.0f} TFLOP/s")
PYEOF
}

step_prereg() {
  env -u MIRROR_FINAL_ALLOW_NO_PREREG "$PY" - << 'PYEOF'
import json
import sys

sys.path.insert(0, ".")
from mirror.final.prereg import PreregError, require_prereg

try:
    st = require_prereg(allow_missing=False)
except PreregError as exc:
    print(f"STOP: {exc}")
    sys.exit(1)
print(json.dumps({k: st[k] for k in ("file", "sha256", "recorded_commit", "ok", "reason")}, indent=2))
PYEOF
}

step_models() {
  "$PY" scripts/prefetch_models.py --only "$MODELS" --out results/data_check/models_final.json
}

step_data() {
  bash scripts/h100_data.sh --only val,train,msel,check --workers "$WORKERS"
}

delete_flag() { [ "$DELETE_SHARDS" = 1 ] && echo "--delete-shards"; }

step_roles() {
  "$PY" scripts/fp_cache.py estimate || return 1
  # shellcheck disable=SC2046
  "$PY" scripts/fp_cache.py build --roles fit3,coco --workers "$WORKERS" $(delete_flag) || return 1
  "$PY" scripts/fp_cache.py index-stream || return 1
  "$PY" scripts/fp_cache.py check --require fit3,coco
}

step_roles_ext() {
  # shellcheck disable=SC2046
  "$PY" scripts/fp_cache.py build --roles heal_ext --workers "$WORKERS" $(delete_flag) || return 1
  "$PY" scripts/fp_cache.py check --require fit3,coco,heal_ext
}

step_extra() {
  bash scripts/h100_data.sh --only tier2,v2,sketch --workers "$WORKERS"
}

step_tests() {
  CUDA_VISIBLE_DEVICES="" MIRROR_FINAL_ALLOW_NO_PREREG=1 "$PY" -m pytest -q tests/test_fp_*.py
}

step_smoke() {
  local c rc=0
  mkdir -p runs/final/_smoke
  for c in "$SMOKE_HEAL_CMD" "$SMOKE_EVAL_CMD" "${SMOKE_EXTRA_CMD:-}"; do
    [ -n "$c" ] || continue
    echo "== $c"
    MIRROR_FINAL_ALLOW_NO_PREREG=1 bash -c "$c" || { echo "FAILED: $c"; rc=1; }
  done
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
  return $rc
}

timing_cmd() {   # $1 model, $2 seed, $3 tag
  local out="runs/final/_timing/${1}_${3}" c="$TIMING_CMD"
  c=${c//\{model\}/$1}; c=${c//\{seed\}/$2}; c=${c//\{steps\}/$TIMING_STEPS}; c=${c//\{out\}/$out}
  echo "$c"
}

elapsed() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.3f", b - a }'; }

step_timing() {
  local f="$W0/timing.tsv" m c t0 t1 i rc pids=()
  mkdir -p runs/final/_timing results/final
  printf "run\tmodel\tseconds\trc\n" > "$f"
  for m in $TIMING_MODELS; do                                  # one heal per model, alone on the GPU
    c=$(timing_cmd "$m" 0 alone)
    echo "== alone: $c"
    t0=$(date +%s.%N)
    bash -c "$c"; rc=$?
    t1=$(date +%s.%N)
    printf "alone\t%s\t%s\t%s\n" "$m" "$(elapsed "$t0" "$t1")" "$rc" >> "$f"
    [ $rc -eq 0 ] || return 1
  done
  m=$(echo "$TIMING_MODELS" | awk '{print $1}')                  # three concurrent heals of the first model
  t0=$(date +%s.%N)
  for i in 1 2 3; do
    c=$(timing_cmd "$m" "10$i" "concurrent$i")
    echo "== concurrent $i: $c"
    ( s=$(date +%s.%N); bash -c "$c"; r=$?; e=$(date +%s.%N)
      printf "concurrent%s\t%s\t%s\t%s\n" "$i" "$m" "$(elapsed "$s" "$e")" "$r" >> "$f" ) &
    pids+=($!)
  done
  for i in "${pids[@]}"; do wait "$i"; done
  t1=$(date +%s.%N)
  printf "concurrent_all\t%s\t%s\t0\n" "$m" "$(elapsed "$t0" "$t1")" >> "$f"
  cat "$f"
  "$PY" - "$f" "$TIMING_STEPS" "$FULL_STEPS" "$TIMING_UNITS" << 'PYEOF'
import csv
import json
import sys
import time

path, steps, full, units_txt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
units = {k: float(v) for k, v in (x.split("=") for x in units_txt.split(",") if x)}
rows = list(csv.DictReader(open(path), delimiter="\t"))
if any(r["rc"] != "0" for r in rows):
    sys.exit("a timing run failed: " + str([r for r in rows if r["rc"] != "0"]))
scale = full / steps
alone = {r["model"]: float(r["seconds"]) * scale / 3600 for r in rows if r["run"] == "alone"}
conc = [float(r["seconds"]) * scale / 3600 for r in rows if r["run"].startswith("concurrent") and r["run"][-1].isdigit()]
batch = next(float(r["seconds"]) * scale / 3600 for r in rows if r["run"] == "concurrent_all")
first = next(r["model"] for r in rows if r["run"].startswith("concurrent"))
u = units.get(first, 1.0)
out = {
    "hours_per_unit": sum(conc) / len(conc) / u, "concurrency": float(len(conc)),
    "hours_per_unit_alone": alone[first] / u,
    # how much sooner 3 heals end together than one after another (3 x alone / the batch's wall time)
    "throughput_gain": len(conc) * alone[first] / batch,
    "alone_hours_per_full_heal": alone, "concurrent_hours_per_full_heal": conc, "batch_hours": batch,
    "units": units, "measured_unit_ratio": {m: (alone[m] / alone[first]) for m in alone},
    "timing_steps": steps, "full_steps": full,
    "note": "wall time per heal scaled linearly from timing_steps to full_steps (fixed start-up costs make this an "
            "overestimate when timing_steps < full_steps); the runner uses hours_per_unit / concurrency as the wall "
            "time of one heal-unit when 3 heals run at once",
    "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
json.dump(out, open("results/final/w0_timing.json", "w"), indent=2)
print(json.dumps(out, indent=2))
PYEOF
}

status() {
  local s
  for s in "${STEPS_KNOWN[@]}"; do
    if [ -f "$W0/$s.done" ]; then echo "  $s: done ($(head -1 "$W0/$s.done"))"; else echo "  $s: not done"; fi
  done
  [ -f "$STATE/budget.json" ] && echo "  VM clock: $(cat "$STATE/budget.json")"
}

[ $# -ge 1 ] || { sed -n '2,32p' "$0"; exit 2; }
case "$1" in
  -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
  status) status; exit 0 ;;
  all)
    for s in "${STEPS_ALL[@]}"; do
      if [ "$s" = heartbeat ] && [ ! -f "$MANIFEST" ]; then say "heartbeat: $MANIFEST missing, not started"; continue; fi
      run_step "$s" "step_$s" || { say "stopped at $s"; exit 1; }
    done
    say "W0 steps done; next: bash scripts/fp_vm_w0.sh roles_ext, extra and timing (or let the runner run them)"
    exit 0 ;;
esac
for s in "$@"; do
  case " ${STEPS_KNOWN[*]} " in *" $s "*) ;; *) echo "unknown step $s (steps: ${STEPS_KNOWN[*]} status all)"; exit 2 ;; esac
  run_step "$s" "step_$s" || exit 1
done
