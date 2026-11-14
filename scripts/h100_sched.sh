#!/usr/bin/env bash
# Job scheduler for one H100 shared by 2 to 4 jobs of this project (PROJECT_NEXT_PHASE.md, Section 9).
# Successor of scripts/vm_sched.sh with the same job-file format, "<needed files> | <command>" with "@MiB"
# tokens, plus:
#   %lock=<name>   at most one running job holds <name>, across all job files (heals: %lock=heal)
#   %done=<file>   skip the job when <file> exists (reruns after a preemption redo only unfinished jobs)
# A job starts when its files exist, its locks are free, fewer than PAR of this file's jobs run, and
#   free GPU MiB - MiB reserved for jobs started less than WARMUP_S ago - its @MiB >= HEADROOM_MB.
# Per-job state (pid, exit code) lives in logs/<jobs>/state/, so the scheduler can be stopped and started
# again while its jobs keep running: it never starts a job twice and never kills anything (it starts
# every job in its own session; Ctrl-C or killing the scheduler leaves the jobs running). Jobs whose
# process vanished without an exit code (machine stopped) or that died from a signal are started again,
# up to MAX_ATTEMPTS launches; jobs that failed (exit code 1 to 127) only with RETRY_FAILED=1. The job
# file is read again at every step, so lines can be added while it runs. The rules are in
# scripts/h100_sched_core.py (tests: tests/test_h100_sched.py).
#
#   setsid nohup bash scripts/h100_sched.sh jobs/<file>.txt > logs/q_<file>.log 2>&1 < /dev/null &
#   bash scripts/h100_sched.sh --status jobs/<file>.txt     # every job: status and why it waits
#   bash scripts/h100_sched.sh --check jobs/<file>.txt      # parse the file only
#
# Settings (environment, defaults in brackets): PAR [3], NEED_MB [4000] (jobs without @MiB),
# HEADROOM_MB [8192], WARMUP_S [240], STAGGER [30] (pause after a start), POLL [20], GRACE_S [60],
# MAX_ATTEMPTS [3], RETRY_FAILED [0], GPU_LOG_S [60], SUMMARY_S [600].
# Outputs: logs/<jobs>/sched.log (start and end lines with exit codes), logs/<jobs>/job<i>.log (appended
# per attempt), logs/<jobs>/gpu.csv (GPU utilisation and memory every GPU_LOG_S), logs/<jobs>/state/.
# To stop one job for good: kill -TERM -<pid> (pid in logs/<jobs>/state/<key>.pid; the minus sign
# signals its whole process group), then touch logs/<jobs>/state/<key>.skip.
# To run a finished job again: delete its files in logs/<jobs>/state/ (or the whole folder).
set -u

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ROOT="${SCHED_ROOT:-$(cd "$(dirname "$SELF")/.." && pwd)}"
CORE="$(dirname "$SELF")/h100_sched_core.py"

# ---------------------------------------------------------------------------------------------------
# Runner (internal): one job, in its own session. Writes its pid first and its exit code last.
# ---------------------------------------------------------------------------------------------------
if [ "${1:-}" = "__run" ]; then
  state=$2; key=$3; log=$4
  st=$(awk '{sub(/^.*\) /, ""); print $20}' "/proc/$$/stat" 2>/dev/null || true)
  echo "$$ $st" > "$state/$key.pid.tmp" && mv -f "$state/$key.pid.tmp" "$state/$key.pid"
  cmd=$(cat "$state/$key.cmd")
  attempt=$(grep -o '"attempt": [0-9]*' "$state/$key.start" 2>/dev/null | grep -o '[0-9]*$')
  cd "$ROOT" || exit 1
  {
    echo "== start $(date -u '+%F %T') key $key attempt ${attempt:-?} host $(hostname) pid $$"
    echo "== cmd $cmd"
  } >> "$log"
  bash -c "$cmd" >> "$log" 2>&1 < /dev/null
  rc=$?
  echo "$rc" > "$state/$key.rc.tmp" && mv -f "$state/$key.rc.tmp" "$state/$key.rc"
  echo "== end rc=$rc $(date -u '+%F %T')" >> "$log"
  exit 0
fi

mode=run
case "${1:-}" in
  --status) mode=status; shift ;;
  --check) mode=check; shift ;;
  -h|--help|"") sed -n '2,32p' "$SELF"; exit 0 ;;
esac
JOBS="${1:?usage: bash scripts/h100_sched.sh [--status|--check] jobs/<file>.txt}"
[ -f "$JOBS" ] || { echo "no such job file: $JOBS"; exit 2; }
JOBS_ABS="$(cd "$(dirname "$JOBS")" && pwd)/$(basename "$JOBS")"
PAR="${PAR:-3}"; NEED_MB="${NEED_MB:-4000}"; HEADROOM_MB="${HEADROOM_MB:-8192}"; WARMUP_S="${WARMUP_S:-240}"
STAGGER="${STAGGER:-30}"; POLL="${POLL:-20}"; GRACE_S="${GRACE_S:-60}"; MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
RETRY_FAILED="${RETRY_FAILED:-0}"; GPU_LOG_S="${GPU_LOG_S:-60}"; SUMMARY_S="${SUMMARY_S:-600}"

name="$(basename "$JOBS" .txt)"
LOGDIR="$ROOT/logs/$name"; STATE="$LOGDIR/state"; LOCKS="$ROOT/logs/sched_locks"
mkdir -p "$STATE" "$LOCKS"
cd "$ROOT" || exit 1

# Environment for the jobs: the bootstrap's env file (HF_HOME, PATH) and the repo's venv.
if [ "${SCHED_NO_ENV:-0}" != 1 ]; then
  [ -f "$HOME/.mirror_env.sh" ] && . "$HOME/.mirror_env.sh"
  if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$ROOT/.venv/bin/activate" ]; then . "$ROOT/.venv/bin/activate"; fi
fi
PY="${SCHED_PYTHON:-$(command -v python3 || command -v python)}"
core_args=(--jobs "$JOBS_ABS" --root "$ROOT" --state "$STATE" --locks "$LOCKS" --default-mb "$NEED_MB"
           --par "$PAR" --headroom "$HEADROOM_MB" --warmup "$WARMUP_S" --grace "$GRACE_S"
           --max-attempts "$MAX_ATTEMPTS")
[ "$RETRY_FAILED" = 1 ] && core_args+=(--retry-failed)

gpu_free() {
  if [ -n "${SCHED_FAKE_FREE_MB:-}" ]; then echo "$SCHED_FAKE_FREE_MB"; return; fi
  local v
  v=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
  echo "${v:-0}"
}

gpu_log() {   # one CSV row: time, utilisation, memory, power, temperature, jobs of this file running
  local f="$LOGDIR/gpu.csv" q
  [ -f "$f" ] || echo "utc,gpu_util_pct,mem_util_pct,mem_used_mib,mem_free_mib,mem_total_mib,power_w,temp_c,running_jobs" > "$f"
  q=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free,memory.total,power.draw,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  [ -n "$q" ] || q=",,,,,,"
  echo "$(date -u '+%F %T'),$q,${1:-}" >> "$f"
}

say() { echo "$*" | tee -a "$LOGDIR/sched.log"; }

summary() {
  say "== status $(date -u '+%F %T') (free GPU memory $(gpu_free) MiB)"
  "$PY" "$CORE" status "${core_args[@]}" --free-mb "$(gpu_free)" 2>&1 | tee -a "$LOGDIR/sched.log"
}

if [ "$mode" = check ]; then exec "$PY" "$CORE" check "${core_args[@]}"; fi
if [ "$mode" = status ]; then
  "$PY" "$CORE" status "${core_args[@]}" --free-mb "$(gpu_free)"
  for f in "$STATE"/*.pid; do
    [ -f "$f" ] || continue
    k=$(basename "$f" .pid); p=$(cut -d' ' -f1 "$f")
    e=$(ps -o etime= -p "$p" 2>/dev/null | tr -d ' ')
    [ -f "$STATE/$k.rc" ] || echo "   $k: pid $p (elapsed ${e:-gone}; stop it with kill -TERM -$p)"
  done
  exit 0
fi

"$PY" "$CORE" check "${core_args[@]}" > /dev/null || { echo "the job file does not parse"; exit 2; }
have_flock=0; command -v flock > /dev/null && have_flock=1
if [ $have_flock = 1 ]; then
  exec 8> "$LOGDIR/sched.lock"
  flock -n 8 || { echo "another scheduler is running $JOBS (pid $(cat "$LOGDIR/sched.pid" 2>/dev/null))"; exit 3; }
else
  echo "warning: flock not found; no protection against two schedulers on one file or across files"
fi
echo $$ > "$LOGDIR/sched.pid"
trap 'say "== scheduler stopped $(date -u +%H:%M:%S); its jobs keep running (start it again to resume)"; exit 130' INT TERM HUP

launch() {   # $1 index, $2 key; the core has already written state/<key>.start and .cmd
  local idx=$1 key=$2 log="$LOGDIR/job$1.log"
  if command -v setsid > /dev/null; then
    setsid nohup bash "$SELF" __run "$STATE" "$key" "$log" > /dev/null 2>&1 < /dev/null 8>&- 9>&- &
  else
    nohup bash "$SELF" __run "$STATE" "$key" "$log" > /dev/null 2>&1 < /dev/null 8>&- 9>&- &
  fi
  say "== start $idx $(date -u +%H:%M:%S) $(head -c 400 "$STATE/$key.cmd")"
}

n=$("$PY" "$CORE" check "${core_args[@]}" | wc -l)
say "== scheduler $(date -u '+%F %T') pid $$ on $(hostname): $name, $n jobs, PAR $PAR, NEED_MB $NEED_MB, HEADROOM_MB $HEADROOM_MB, WARMUP_S $WARMUP_S"
summary
last_gpu=0; last_sum=$(date +%s)
while true; do
  free=$(gpu_free)
  if [ $have_flock = 1 ]; then exec 9> "$LOCKS/.mutex"; flock 9; fi
  out=$("$PY" "$CORE" step "${core_args[@]}" --free-mb "$free" 2>&1); crc=$?
  started=0; alldone=0; nrun=0
  [ $crc -eq 0 ] || say "== scheduler core error (rc $crc; fix the job file, the scheduler keeps polling): $out"
  while IFS= read -r line; do
    case "$line" in
      "END "*) say "== end ${line#END } at $(date -u +%H:%M:%S)" ;;
      "LOST "*) say "== lost ${line#LOST } at $(date -u +%H:%M:%S)" ;;
      "START "*) read -r idx key <<< "${line#START }"; launch "$idx" "$key"; started=1 ;;
      "COUNTS "*) nrun=$(grep -o 'running=[0-9]*' <<< "$line" | cut -d= -f2); nrun=${nrun:-0} ;;
      ALLDONE) alldone=1 ;;
    esac
  done <<< "$out"
  if [ $have_flock = 1 ]; then flock -u 9; exec 9>&-; fi
  now=$(date +%s)
  if [ $((now - last_gpu)) -ge "$GPU_LOG_S" ]; then gpu_log "$nrun"; last_gpu=$now; fi
  if [ $started -eq 0 ] && [ $((now - last_sum)) -ge "$SUMMARY_S" ]; then summary; last_sum=$now; fi
  [ $alldone -eq 1 ] && break
  if [ $started -eq 1 ]; then sleep "$STAGGER"; else sleep "$POLL"; fi
done
gpu_log 0
summary
say "== QUEUE_DONE $(date -u)"
