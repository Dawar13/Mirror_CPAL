#!/usr/bin/env bash
# Job queue for the shared A100: run the commands in a jobs file (one per line, # for comments),
# at most $PAR at a time, each started only when at least $NEED_MB MiB of GPU memory is free.
#   PAR=3 NEED_MB=9000 HF_HOME=... setsid nohup bash scripts/vm_queue.sh jobs/stage0.txt > logs/queue_stage0.log 2>&1 &
cd "$(dirname "$0")/.."
source .venv/bin/activate
JOBS=$1
PAR=${PAR:-3}
NEED_MB=${NEED_MB:-9000}
name=$(basename "$JOBS" .txt)
mkdir -p logs/$name
i=0
while IFS= read -r line || [ -n "$line" ]; do
  [ -z "$line" ] && continue
  [[ "$line" =~ ^# ]] && continue
  i=$((i + 1))
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 20; done
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$NEED_MB" ]; do sleep 60; done
  echo "== start $i $(date -u +%H:%M:%S) $line"
  ( bash -c "$line" > logs/$name/job$i.log 2>&1; echo "== end $i rc=$? $(date -u +%H:%M:%S) $line" ) &
  sleep 45   # stagger model loading
done < "$JOBS"
wait
echo "== QUEUE_DONE $(date -u)"
