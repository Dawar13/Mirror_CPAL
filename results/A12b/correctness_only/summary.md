# A12b fused even/odd MLP kernel

**Pre-registered bar (A12b, at least 1.1x the original's throughput at batch 64 and 256): NOT SCORED (correctness test only (no timing)).**

Backend: triton (Triton 3.6.0); device cuda (NVIDIA H200); mode correctness_only.
Correctness: PASS (12 of 12 blocks; float32 and bf16 within the tolerances below; float64 layout check).
Exact mirror equivariance of the fused MLP (sign flip of the odd coordinates): float32 in 12 of 12 blocks, bf16 in 12 of 12 (plain even/odd MLP: float32 12, bf16 12).

## Correctness per block (errors: max |difference| / max |reference|)

| Block | pairs | self | fp32 fused | fp32 plain | fp32 tol | bf16 fused | bf16 plain | bf16 tol | layout f64 | pass |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1168 | 736 | 1.75e-06 | 1.75e-06 | 1.40e-05 | 4.46e-03 | 5.12e-03 | 1.02e-02 | 4.29e-15 | yes |
| 1 | 1116 | 840 | 1.79e-06 | 1.79e-06 | 1.43e-05 | 4.41e-03 | 5.04e-03 | 1.01e-02 | 4.47e-15 | yes |
| 2 | 1013 | 1046 | 1.78e-06 | 1.81e-06 | 1.44e-05 | 5.33e-03 | 5.41e-03 | 1.08e-02 | 4.19e-15 | yes |
| 3 | 963 | 1146 | 1.79e-06 | 1.79e-06 | 1.43e-05 | 4.30e-03 | 4.56e-03 | 9.12e-03 | 4.06e-15 | yes |
| 4 | 878 | 1316 | 1.98e-06 | 1.98e-06 | 1.58e-05 | 3.94e-03 | 4.84e-03 | 9.67e-03 | 4.00e-15 | yes |
| 5 | 672 | 1728 | 1.90e-06 | 1.90e-06 | 1.52e-05 | 3.89e-03 | 4.51e-03 | 9.02e-03 | 4.13e-15 | yes |
| 6 | 554 | 1964 | 2.10e-06 | 2.06e-06 | 1.65e-05 | 4.01e-03 | 5.39e-03 | 1.08e-02 | 4.26e-15 | yes |
| 7 | 406 | 2260 | 2.40e-06 | 2.40e-06 | 1.92e-05 | 4.00e-03 | 5.00e-03 | 9.99e-03 | 6.44e-15 | yes |
| 8 | 155 | 2762 | 2.16e-06 | 2.16e-06 | 1.73e-05 | 3.52e-03 | 3.80e-03 | 7.60e-03 | 4.19e-15 | yes |
| 9 | 93 | 2886 | 2.16e-06 | 2.12e-06 | 1.70e-05 | 4.37e-03 | 4.51e-03 | 9.03e-03 | 5.27e-15 | yes |
| 10 | 19 | 3034 | 2.23e-06 | 2.23e-06 | 1.78e-05 | 4.89e-03 | 4.38e-03 | 8.77e-03 | 4.81e-15 | yes |
| 11 | 17 | 3038 | 1.79e-06 | 1.79e-06 | 1.44e-05 | 3.65e-03 | 4.16e-03 | 8.32e-03 | 4.17e-15 | yes |

## What is fused

One Triton kernel per MLP computes both fc1 GEMMs (even part, K = ke; odd part, K = d - ko), the fc1 bias, the butterfly u +/- v, GELU and the butterfly ga +/- gb, and writes the fc2 operands once. fc2 runs as two torch GEMMs (cuBLAS); the output concatenation is left to torch.compile (fused with the residual addition). Attention is unchanged.

## Caveats

- Other GPU processes at the start of timing: none seen.
- Inputs are random images and tokens; timing does not depend on the weight values. Tile configurations of the kernel were chosen without measurement; the autotuner picked one per shape (metrics.json, autotune).

Files: metrics.json, correctness.csv, timing.csv (timing runs), meta.json.
