# STATUS (paper experiments)

Updated 2026-11-08 about 10:50 UTC. Plan: `PAPER_EXPERIMENTS.md`; context: `PROJECT_CONTEXT.md` v2;
pre-registration `PREREG_E2.md` committed as 9804f06 before any gated run. A100 available until about
13:00 UTC, shared with another project. All numbers below come from the files named; k-NN under the
paper protocol (operators fitted on ImageNet train, 100k train bank, all 50k validation queries, bf16 features).

## Stage 0
| Item | Result | Where |
|---|---|---|
| X1 SVD-collapse check, gate G1 | Passed: no bug (full rank exact, folding exact, ranks match, fp32). Registers model has a genuine cliff when SVD covers 4 or more blocks; mechanism open (verify) | `runs/x1/` |
| X2 data | 106k train images at 224 px (fit0..2, bank), 105k at 256 px (heal, healval) | VM `cache/` |
| X4 full LayerNorm folding | Algebra correct (float64 on the registers model 5.7e-6); fp32 bar of 1e-4 missed through rounding (registers 9.8e-4, plain 1.05e-2 absolute). In practice harmless: folded registers model k-NN 78.98 vs 78.94 | `results/x4_fold_*.json`, `runs/m1/m1_knn.csv` |
| X5 PREREG_E2.md | Committed 9804f06 | `PREREG_E2.md` |

## X2: E1b arms under the paper protocol (fit seed 0)
| Arm | Outcome | Original | Surgery k = 3 / 6 / 9 | SVD k = 3 / 6 / 9 | Random k = 6 |
|---|---|---|---|---|---|
| Plain DINOv2, per-layer (E1b primary) | **GO** | 78.71 | see csv / 74.68 / 73.26 | see csv / 66.23 / 36.63 | 3.66 |
| DINOv2-reg, per-layer (main model) | AMBER (row 4: 0.223 vs 0.2) | 78.94 | 78.35 / 77.09 / 76.36 | see csv / 8.77 / 5.83 | 3.98 |
| Plain, shared operator layers 1 to 6 | stopped after k = 6 and k = 3 (time) | 78.71 | see csv / 73.58 / - | 77.00 / 67.34 / - | 3.97 |
Seeds: registers model seed 2 running; seed 1 must be rerun (both copies stopped while freeing GPU memory).

## Findings and gates
- **P1 / gate G3: PASS, both bullets.** Registers model: late-layer error 1.30x early (limit 2). Plain model: masking the top 1% highest-norm tokens cuts late-layer error 2.27x (limit 2). `runs/f3/`
- **G generality (k = 6):** DeiT-III AMBER (rows 1 to 4 GO; surgery 81.55 vs SVD 80.67, margin 0.9); MAE strongest structure (row 1 = 40.8; k-NN baseline only 13.7%, not meaningful); SigLIP2 KILL (row 1 = 2.4; surgery 13.8 vs original 75.5). **P4:** DeiT-III and MAE bullets pass; the SigLIP2 bullet fails as worded (its raw breaking share 0.137 is below DINOv2-reg's 0.223 because its structure is near trivial; rows 1 to 3 show much weaker symmetry). `runs/g/`
- **M1 V-exact:** one operator for layers 0 to 6 has 216 odd directions (28%); exactness inside the span passes (relative about 2e-6); **full conversion (k = 12) gives 100.00% flip consistency in fp32**. k-NN of the attention variants running. `runs/m1/`

## Running now
- M3 healing pilot (gate G2, P3): mirror, then SVD, then random; about 45 min each at 1.74 s per step.
- M1 k-NN (V-local, V-exact, MLP-only, SVD, random at k = 6, 9, 12).
- X3: registers model seed 2; D data-free pilot (CPU).

## Will not fit on this A100 (needs another GPU)
X3 seed 1 rerun and plain-model seeds; A1 to A5; F4; M3 grid; M2 latency (not implemented); Tier-2 evaluations; DINOv2-S/L; V-exact healed; M4.

## Deviations from the plan
- k-NN features in bf16 for every variant alike; fits in fp32 and float64.
- GPU shared: memory caps 6 to 20% per job; jobs stopped or paused to keep healing within the A100 window (MAE k-NN, shared 1 to 6 arm at k = 9, X3 seed 1, A1/A2).
- X3 seeds 1 and 2 at k = 6 only.
