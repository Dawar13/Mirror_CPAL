#!/usr/bin/env bash
# File-aware job scheduler (replaces the strictly ordered vm_queue.sh when inputs arrive at different times).
# Each line of the jobs file: "<needed files, space separated> | <command>" ("-" when nothing is needed); "#" lines
# are comments. A job starts as soon as all its files exist, fewer than $PAR of this scheduler's jobs are running and
# at least $NEED_MB MiB of GPU memory are free; jobs may start out of order. Logs: logs/<jobs file name>/job<i>.log.
#   PAR=4 NEED_MB=4000 bash scripts/vm_sched.sh jobs/<file>.txt
set -u
JOBS="$1"; PAR="${PAR:-4}"; NEED_MB="${NEED_MB:-4000}"; STAGGER="${STAGGER:-60}"
name=$(basename "$JOBS" .txt); mkdir -p "logs/$name"
mapfile -t LINES < <(grep -vE '^\s*(#|$)' "$JOBS")
n=${#LINES[@]}; declare -A started
echo "== scheduler $(date -u +%H:%M:%S): $n jobs, PAR $PAR, NEED_MB $NEED_MB"
while [ "${#started[@]}" -lt "$n" ]; do
  launched=0
  for i in $(seq 0 $((n - 1))); do
    [ -n "${started[$i]:-}" ] && continue
    [ "$(jobs -rp | wc -l)" -ge "$PAR" ] && break
    need="${LINES[$i]%%|*}"; cmd="${LINES[$i]#*|}"
    ok=1; for f in $need; do [ "$f" = "-" ] && continue; [ -f "$f" ] || { ok=0; break; }; done
    [ $ok -eq 1 ] || continue
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$free" -ge "$NEED_MB" ] || break
    started[$i]=1; j=$((i + 1))
    echo "== start $j $(date -u +%H:%M:%S) $cmd"
    ( bash -c "$cmd" > "logs/$name/job$j.log" 2>&1; echo "== end $j rc=$? $(date -u +%H:%M:%S) $cmd" ) &
    launched=1; sleep "$STAGGER"
  done
  [ $launched -eq 1 ] || sleep 30
done
wait
echo "== QUEUE_DONE $(date -u)"
