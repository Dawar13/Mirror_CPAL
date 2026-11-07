# CLAUDE.md (Mirror_CPAL)

Read `PROJECT_CONTEXT.md` and `E1_INSTRUCTIONS.md` before doing anything.

**Current phase: A (code only).** The laptop has about 6 GB free and no GPU. Do not download any
model, dataset or weights here. Write all code and tests described in Phase A, run the tests on CPU
if the environment exists, then commit and push. Phase B runs on the H100 VM.

Rules:
- Never invent numbers, citations or results; mark anything unverified as (verify).
- Every reported number comes from a script in this repo with a fixed seed; log config, seed and git hash.
- Store statistics, not raw activations.
- Do not change E1 decision thresholds without asking.
- On the VM: all downloads under `$HF_HOME`; cap GPU memory with `torch.cuda.set_per_process_memory_fraction(0.2)` (the GPU is shared with another project).
- Reports: British spelling, plain language, no em-dashes.
