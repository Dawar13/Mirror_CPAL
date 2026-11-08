# CLAUDE.md (Mirror_CPAL)

Read `PROJECT_CONTEXT.md` (v2) and `PAPER_EXPERIMENTS.md` before doing anything.

**Current phase: paper experiments (Stage 0 first).** E1 (KILL) and E1b (GO) are finished; keep their reports unchanged.

Rules:
- Never invent numbers, citations or results; mark anything unverified as (verify).
- Every reported number comes from a script in this repo with a fixed seed; log config, seed and git hash.
- Fit operators on ImageNet train images; evaluate on validation; healing data comes from train only.
- Pre-registered thresholds in PREREG_E2.md are fixed once committed. Stop and report at each gate.
- GPU is shared: cap memory per process (`torch.cuda.set_per_process_memory_fraction`), queue small jobs, run one healing job at a time.
- All downloads under `$HF_HOME` on the VM; nothing on the laptop.
- Keep `results/STATUS.md` current.
- Reports: British spelling, plain language, no em-dashes.
