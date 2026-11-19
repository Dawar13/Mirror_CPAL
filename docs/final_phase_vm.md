# Phase G on the H200: exact commands

For FINAL_PHASE_PLAN.md v2, Section 8. Everything below is restart-safe: a command run twice skips finished work.

## 1. The machine (Dawar starts it)

- One H200 (141 GB). **Disk: 500 GB** (the caches take about 130 GB, the full-bank features of A10 about 110 GB, the
  healed checkpoints about 30 GB, and training shards are downloaded and deleted one at a time).
- As many CPU cores and as much host memory as the offer allows (32 cores or more and 256 GB or more if possible):
  the structure fits (A01, A02, A03) are CPU-bound matchings, and the new data roles decode JPEGs on the CPU.
- Repository at `/home/Mirror_CPAL`, Hugging Face cache at `/home/hf` (Jarvis Labs keeps only `/home` across a pause).

## 2. From the laptop (Git Bash, in the repository)

```
export H100_HOST=<ip> H100_PORT=<port> H100_USER=<user> H100_KEY=<ssh key>
FIRST=1 bash scripts/ship_h100.sh      # the committed code and .git (first time: everything)
bash scripts/fp_upload.sh              # structures, X2 operators, the MUST-09 checkpoint (about 1.7 GB, sha256-checked)
```

## 3. On the VM (ssh), once

```
cd /home/Mirror_CPAL
bash scripts/fp_vm_w0.sh clock         # starts the 15-hour VM clock (logs/final/state/budget.json)
read -rs HF_TOKEN && export HF_TOKEN   # a fresh token, typed, never written to the repository
bash scripts/fp_vm_w0.sh env           # venv, torch, Hugging Face login
bash scripts/fp_vm_w0.sh gpu           # GPU check
bash scripts/fp_vm_w0.sh prereg        # PREREG_FINAL_PHASE.md recorded and unchanged, else stop
. ~/.mirror_env.sh
setsid nohup python scripts/fp_runner.py run jobs/final_phase.yaml > logs/final/runner.out 2>&1 < /dev/null &
```

The runner then does the rest of W0 (models, data, roles, tests, the GPU smoke test, timing) and W1 to W5, with the
budget governor (no new heal after 13.5 VM-hours, everything stops at 15) and the Section 9 cut order.

## 4. Watching it

```
python scripts/fp_runner.py status jobs/final_phase.yaml     # every job's state and why it waits
python scripts/fp_runner.py project jobs/final_phase.yaml    # projected VM hours and the cuts it would make
tail -f logs/final/runner.log                                 # events
less results/STATUS.md                                        # the runner's section is rewritten every minute
tail logs/final/gpu.csv                                       # nvidia-smi every 30 s
```

Stop events (the runner stops launching and the coordinator reports): Gate P fails (results/A01/gate_p.json says
`"pass": false`), a job fails twice, a memory error persists, 13.5 VM-hours.

## 5. At the end

```
python scripts/collect_final_phase.py    # the runner's last job; writes results/FINAL_PHASE_REPORT.md
```

From the laptop: `bash scripts/pull_h100.sh` (tables, summaries, figures and logs only; checkpoints and features stay on
the VM). Then Dawar pauses or stops the machine; delete the token file first if others use it.
