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
| Code: M1 attention conversion, M3 healing trainer, G adapters, X4 full LayerNorm folding | Written with CPU tests | `mirror/attention.py`, `mirror/heal.py`, `mirror/adapters_timm.py`, `mirror/fold.py` |

## Running / next
- Stage 0 queue on the VM (`jobs/stage0.txt`): X4 folding checks on both DINOv2 models; X2 reruns of the three E1b arms under protocol v2 with fit seeds 0, 1, 2 (X3).
- `PREREG_E2.md` drafted (thresholds copied unchanged from PAPER_EXPERIMENTS.md); it is committed before the first F3, F4, G, D or M3 run.
- Then Stage 1: F1 to F4, F7, A1 to A5, M1 zero-shot, G, D.

## Deviations from the plan
- k-NN features are computed in bf16 for every variant alike (speed on a shared GPU); fits and statistics stay in fp32 and float64.
- The GPU is shared with another project; memory caps of 6 to 12% per job, queue with a free-memory guard.
