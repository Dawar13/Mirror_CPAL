# STATUS (paper experiments)

Updated 2026-11-08 (UTC times). Plan: `PAPER_EXPERIMENTS.md`; context: `PROJECT_CONTEXT.md` v2.

## Done
| Item | Result | Where |
|---|---|---|
| E1 (one operator, all layers) | KILL | `results/E1_REPORT.md` |
| E1b (per-layer, pre-registered) | primary GO; registers AMBER; shared 1 to 6 AMBER | `results/E1B_SUMMARY.md`, `runs/e1b_*` |
| X2 data: ImageNet train subsets | 106k at 224 px (fit0..2, bank), 105k at 256 px (heal, healval); train only | `cache/train_*` on the VM, `scripts/build_train_cache.py` |
| X1 SVD-collapse check | No bug: full rank exact, folding exact, ranks match (0.997 to 0.9999), fp32 only. Single-block SVD costs at most about 1.9 points in both models | `runs/x1/x1_svd_check.json` |
| X1b compounding | Registers model: 74.5 / 71.8 / 35.1 / 4.8% at k = 0 / 3 / 4 / 6 (sharp cliff at block 4) with steadily shrinking output norms; plain model degrades smoothly (58.8% at k = 6). Genuine effect, mechanism open (verify) | `runs/x1/x1b_svd_compound.json` |
| Gate G1 | Passed: not a bug, SVD numbers stand | |
| X4 full LayerNorm folding | Algebra correct: float64 check passes on the registers model (5.7e-6); fails narrowly on plain DINOv2 (2.0e-4). The fp32 bar of 1e-4 fails on both (9.8e-4 and 1.05e-2, relative about 5e-5 and 4e-4): rounding, amplified by high-norm tokens; DINOv2 interpolates position embeddings in fp32 (likely cause, verify) | `results/x4_fold_*.json` |
| X5 PREREG_E2.md | Committed 9804f06 before any gated run | `PREREG_E2.md` |
| Code: M1 attention conversion, M3 healing trainer, G adapters, X4 full LayerNorm folding | Written with CPU tests | `mirror/attention.py`, `mirror/heal.py`, `mirror/adapters_timm.py`, `mirror/fold.py` |

## Running (A100 available until about 13:00 UTC, shared with another project)
- X2: the three E1b arms under protocol v2, fit seed 0 (`runs/x2/*_s0`).
- Queue `jobs/stage1a.txt`: G (DeiT-III, SigLIP2, MAE at k = 6), then X3 seeds 1 and 2 for the registers and plain per-layer arms (k = 6).
- Being written: F1 to F3, A1 to A5 and F4, M1 run plus healing link, D.

## Next, in priority order
1. M1 zero-shot (MLP plus attention, V-local and V-exact).
2. M3 healing pilot and gate G2 (one job at a time).
3. F3 (P1), F1, F2, D (P5), A1 and A2 at k = 6.

## Will not fit on this A100 (needs another GPU later)
M3 grid, M2 latency (not yet implemented), Tier-2 evaluations, full F4, A3 to A5, DINOv2-S/L, V-exact full conversion healed, M4.

## Deviations from the plan
- k-NN features are computed in bf16 for every variant alike (speed on a shared GPU); fits and statistics stay in fp32 and float64.
- The GPU is shared with another project; memory caps of 6 to 12% per job, queue with a free-memory guard.
- X3 seeds 1 and 2 are run at k = 6 only; the exploratory shared 1 to 6 arm gets seed 0 only (time).
