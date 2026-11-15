# MUST-07: zero-shot comparison at matched parameters

Written by `scripts/must07_compressors.py` on 2026-11-15 19:46:09, commit `805c231c9c` (uncommitted changes). Compressors: **real**; budget accounting: **real**.

Protocol: k-NN top-1 with the 100k-image train bank (100 per class) and the 50k validation images as queries (CLS after the final norm, cosine, k = 20, T = 0.07, bf16 features). Method selection: the 10k method-selection train images as queries against the same bank; every hyperparameter and gate B use only these. Flip consistency: share of validation images whose prediction equals that of the mirrored image, in bf16 (column flip) and, for full-model rows, in fp32 (column flip fp32; bf16 rounding breaks exact ties, so exact invariance is judged on the fp32 value). Parameters: the converted matrices with basis changes; budget = Mirror-exact's count for the same span and components; gap = (method - budget) / budget. GFLOPs: analytic, per image, 2 x multiply-adds. Values are mean +- standard deviation over fit seeds (n given).

## deit3_b16

Original: top-1 82.97, flip 96.38 (bf16), flip fp32 n/a, method-selection top-1 92.02, 86,585,320 parameters, 35.1 GFLOPs.

### 6mlp (blocks 1 to 6, mlp)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 81.06 (n = 1) | 90.79 | 94.81 | n/a | 19,537,952 | 0.00% | 77,811,720 (0.899) | 31.7 |  |
| 3 | plain truncated SVD | 81.29 (n = 1) | 90.75 | 94.99 | n/a | 19,541,760 | 0.02% | 77,815,528 (0.899) | 31.7 |  |
| 4 | SVD-LLM-style whitened SVD | 82.81 (n = 1) | 92.00 | 96.12 | n/a | 19,541,760 | 0.02% | 77,815,528 (0.899) | 31.7 | {"damp": 1e-06} |

### 6both (blocks 1 to 6, mlp+attn)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 81.43 (n = 1) | 91.11 | 97.33 | n/a | 28,802,208 | 0.00% | 72,920,200 (0.842) | 29.7 |  |
| 3 | plain truncated SVD | 78.59 (n = 1) | 88.61 | 93.31 | n/a | 28,803,072 | 0.00% | 72,921,064 (0.842) | 29.7 |  |
| 4 | SVD-LLM-style whitened SVD | 82.54 (n = 1) | 91.80 | 95.77 | n/a | 28,803,072 | 0.00% | 72,921,064 (0.842) | 29.7 | {"damp": 1e-06} |

## dino_b16

Original: top-1 70.48, flip 96.05 (bf16), flip fp32 n/a, method-selection top-1 72.82, 85,798,656 parameters, 35.1 GFLOPs.

### 6mlp (blocks 1 to 6, mlp)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 67.13 (n = 1) | 70.37 | 93.82 | n/a | 22,866,896 | 0.00% | 80,354,000 (0.937) | 33.0 |  |
| 3 | plain truncated SVD | 70.46 (n = 1) | 72.81 | 95.70 | n/a | 22,867,200 | 0.00% | 80,354,304 (0.937) | 33.0 |  |
| 4 | SVD-LLM-style whitened SVD | 70.51 (n = 1) | 72.98 | 95.72 | n/a | 22,867,200 | 0.00% | 80,354,304 (0.937) | 33.0 | {"damp": 1e-06} |

### 6both (blocks 1 to 6, mlp+attn)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 64.39 (n = 1) | 68.50 | 97.14 | n/a | 33,510,568 | 0.00% | 76,841,896 (0.896) | 31.6 |  |
| 3 | plain truncated SVD | 70.26 (n = 1) | 72.47 | 94.96 | n/a | 33,510,912 | 0.00% | 76,842,240 (0.896) | 31.6 |  |
| 4 | SVD-LLM-style whitened SVD | 70.54 (n = 1) | 72.92 | 95.95 | n/a | 33,510,912 | 0.00% | 76,842,240 (0.896) | 31.6 | {"damp": 1e-06} |

## dinov2_vitb14

Original: top-1 78.69, flip 96.35 (bf16), flip fp32 n/a, method-selection top-1 80.06, 86,580,480 parameters, 46.3 GFLOPs.

### 6mlp (blocks 1 to 6, mlp)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 74.42 (n = 1) | 76.19 | 90.72 | n/a | 17,357,792 | 0.00% | 75,626,720 (0.873) | 40.7 |  |
| 3 | plain truncated SVD | 67.74 (n = 1) | 68.67 | 84.11 | n/a | 17,360,640 | 0.02% | 75,629,568 (0.874) | 40.7 |  |
| 4 | SVD-LLM-style whitened SVD | 78.05 (n = 1) | 79.50 | 95.32 | n/a | 17,360,640 | 0.02% | 75,629,568 (0.874) | 40.7 | {"damp": 1e-06} |

### 6both (blocks 1 to 6, mlp+attn)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 73.54 (n = 1) | 75.11 | 92.29 | n/a | 25,993,004 | 0.00% | 70,106,156 (0.810) | 37.9 |  |
| 3 | plain truncated SVD | 12.18 (n = 1) | 12.99 | 23.99 | n/a | 25,995,264 | 0.01% | 70,108,416 (0.810) | 37.9 |  |
| 4 | SVD-LLM-style whitened SVD | 77.78 (n = 1) | 78.99 | 94.73 | n/a | 25,995,264 | 0.01% | 70,108,416 (0.810) | 37.9 | {"damp": 1e-06} |

## dinov2_vitb14_reg

Original: top-1 78.94, flip 96.32 (bf16), flip fp32 n/a, method-selection top-1 80.21, 86,583,552 parameters, 47.1 GFLOPs.

### 6mlp (blocks 1 to 6, mlp)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 76.59 (n = 1) | 78.05 | 91.70 | n/a | 17,034,432 | 0.00% | 75,306,432 (0.870) | 41.2 |  |
| 3 | plain truncated SVD | 9.36 (n = 1) | 10.60 | 42.91 | n/a | 17,038,080 | 0.02% | 75,310,080 (0.870) | 41.2 |  |
| 4 | SVD-LLM-style whitened SVD | 78.33 (n = 1) | 79.58 | 95.43 | n/a | 17,038,080 | 0.02% | 75,310,080 (0.870) | 41.2 | {"damp": 1e-06} |

### 6both (blocks 1 to 6, mlp+attn)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 77.41 (n = 1) | 78.83 | 95.02 | n/a | 25,668,000 | 0.00% | 69,784,224 (0.806) | 38.3 |  |
| 3 | plain truncated SVD | 5.84 (n = 1) | 6.41 | 29.79 | n/a | 25,668,096 | 0.00% | 69,784,320 (0.806) | 38.3 |  |
| 4 | SVD-LLM-style whitened SVD | 78.03 (n = 1) | 79.49 | 95.16 | n/a | 25,668,096 | 0.00% | 69,784,320 (0.806) | 38.3 | {"damp": 1e-06} |

## dinov2_vitl14_reg

Original: top-1 80.59, flip 97.74 (bf16), flip fp32 n/a, method-selection top-1 81.89, 304,372,736 parameters, 164.6 GFLOPs.

### 12mlp (blocks 1 to 12, mlp)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 79.37 (n = 1) | 80.74 | 94.48 | n/a | 61,608,988 | 0.00% | 265,318,428 (0.872) | 144.3 |  |
| 3 | plain truncated SVD | 79.07 (n = 1) | 79.98 | 94.80 | n/a | 61,614,080 | 0.01% | 265,323,520 (0.872) | 144.3 |  |
| 4 | SVD-LLM-style whitened SVD | 80.50 (n = 1) | 81.89 | 97.70 | n/a | 61,614,080 | 0.01% | 265,323,520 (0.872) | 144.3 | {"damp": 1e-06} |

### 12both (blocks 1 to 12, mlp+attn)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 76.50 (n = 1) | 77.88 | 94.68 | n/a | 93,675,004 | 0.00% | 247,052,796 (0.812) | 134.7 |  |
| 3 | plain truncated SVD | 63.91 (n = 1) | 65.51 | 76.84 | n/a | 93,675,520 | 0.00% | 247,053,312 (0.812) | 134.7 |  |
| 4 | SVD-LLM-style whitened SVD | 80.51 (n = 1) | 81.83 | 97.67 | n/a | 93,675,520 | 0.00% | 247,053,312 (0.812) | 134.7 | {"damp": 1e-06} |

## dinov2_vits14_reg

Original: top-1 74.47, flip 93.28 (bf16), flip fp32 n/a, method-selection top-1 75.86, 22,058,112 parameters, 12.5 GFLOPs.

### 6mlp (blocks 1 to 6, mlp)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 61.04 (n = 1) | 61.75 | 76.50 | n/a | 4,187,392 | 0.00% | 19,167,616 (0.869) | 10.9 |  |
| 3 | plain truncated SVD | 10.58 (n = 1) | 11.83 | 27.71 | n/a | 4,187,520 | 0.00% | 19,167,744 (0.869) | 10.9 |  |
| 4 | SVD-LLM-style whitened SVD | 68.37 (n = 1) | 70.42 | 86.65 | n/a | 4,187,520 | 0.00% | 19,167,744 (0.869) | 10.9 | {"damp": 1e-06} |

### 6both (blocks 1 to 6, mlp+attn)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 56.57 (n = 1) | 57.25 | 75.66 | n/a | 6,253,824 | 0.00% | 17,695,104 (0.802) | 10.2 |  |
| 3 | plain truncated SVD | 6.95 (n = 1) | 7.76 | 33.21 | n/a | 6,253,824 | 0.00% | 17,695,104 (0.802) | 10.2 |  |
| 4 | SVD-LLM-style whitened SVD | 59.96 (n = 1) | 62.63 | 78.13 | n/a | 6,253,824 | 0.00% | 17,695,104 (0.802) | 10.2 | {"damp": 1e-06} |

## mae_b16

Original: top-1 13.73, flip 54.54 (bf16), flip fp32 n/a, method-selection top-1 15.29, 85,798,656 parameters, 35.1 GFLOPs.

### 6mlp (blocks 1 to 6, mlp)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 8.67 (n = 1) | 9.28 | 51.94 | n/a | 16,216,916 | 0.00% | 73,704,020 (0.859) | 30.4 |  |
| 3 | plain truncated SVD | 3.81 (n = 1) | 4.21 | 38.80 | n/a | 16,220,160 | 0.02% | 73,707,264 (0.859) | 30.4 |  |
| 4 | SVD-LLM-style whitened SVD | 13.59 (n = 1) | 14.77 | 51.94 | n/a | 16,220,160 | 0.02% | 73,707,264 (0.859) | 30.4 | {"damp": 1e-06} |

### 6both (blocks 1 to 6, mlp+attn)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 9.89 (n = 1) | 10.56 | 54.66 | n/a | 24,190,636 | 0.00% | 67,521,964 (0.787) | 27.9 |  |
| 3 | plain truncated SVD | 4.26 (n = 1) | 4.70 | 40.43 | n/a | 24,192,000 | 0.01% | 67,523,328 (0.787) | 27.9 |  |
| 4 | SVD-LLM-style whitened SVD | 12.15 (n = 1) | 13.32 | 51.18 | n/a | 24,192,000 | 0.01% | 67,523,328 (0.787) | 27.9 | {"damp": 1e-06} |

## siglip2_b16

Original: top-1 75.54, flip 92.87 (bf16), flip fp32 n/a, method-selection top-1 77.93, 92,884,224 parameters, 34.9 GFLOPs.

### 6mlp (blocks 1 to 6, mlp)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 7.84 (n = 1) | 8.78 | 13.00 | n/a | 22,436,880 | 0.00% | 87,009,552 (0.937) | 32.6 |  |
| 3 | plain truncated SVD | 11.43 (n = 1) | 12.08 | 25.69 | n/a | 22,437,120 | 0.00% | 87,009,792 (0.937) | 32.6 |  |
| 4 | SVD-LLM-style whitened SVD | 74.25 (n = 1) | 76.78 | 91.26 | n/a | 22,437,120 | 0.00% | 87,009,792 (0.937) | 32.6 | {"damp": 1e-06} |

### 6both (blocks 1 to 6, mlp+attn)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 13.45 (n = 1) | 14.30 | 40.54 | n/a | 32,965,040 | 0.00% | 83,381,936 (0.898) | 31.2 |  |
| 3 | plain truncated SVD | 4.18 (n = 1) | 4.26 | 14.25 | n/a | 32,965,632 | 0.00% | 83,382,528 (0.898) | 31.2 |  |
| 4 | SVD-LLM-style whitened SVD | 72.90 (n = 1) | 75.43 | 90.24 | n/a | 32,965,632 | 0.00% | 83,382,528 (0.898) | 31.2 | {"damp": 1e-06} |

## Predictions P7.1 and P7.2 (PREREG_NEXT_PHASE.md)

Model `dinov2_vitb14_reg`. P7.1: Mirror-exact beats every baseline (methods 3 to 12) by at least 2 points (mean over fit seeds 0, 1 and 2) with the fit-seed-0 paired bootstrap interval above 0. P7.2: ZipIt-style merging (8) and same-image-correlation pairing (9) trail Mirror-exact by at least 10 points. Each 6-block size is scored separately.

### 6mlp: P7.1 not scored, P7.2 not scored (incomplete: not every method has fit seeds 0, 1 and 2 and a bootstrap)

Mirror-exact: 76.59 (n = 1).

| # | baseline | seeds | baseline mean | Mirror minus baseline | 95% interval (seed 0) | P7.1 | P7.2 |
|---|---|---|---|---|---|---|---|
| 3 | plain truncated SVD | 1 | 9.36 | 67.23 | [66.80, 67.64] | PASS |  |
| 4 | SVD-LLM-style whitened SVD | 1 | 78.33 | -1.74 | [-1.97, -1.52] | FAIL |  |
| 5 | ASVD-style activation scaling | 0 | not run | | | | |
| 6 | SliceGPT-style slicing | 0 | not run | | | | |
| 7 | FLAP-style pruning | 0 | not run | | | | |
| 8 | ZipIt-style merging | 0 | not run | | | | |
| 9 | pairing by same-image correlation | 0 | not run | | | | |
| 10 | pairing by weight cosine | 0 | not run | | | | |
| 11 | random pairing | 0 | not run | | | | |
| 12 | random structure | 0 | not run | | | | |

### 6both: P7.1 not scored, P7.2 not scored (incomplete: not every method has fit seeds 0, 1 and 2 and a bootstrap)

Mirror-exact: 77.41 (n = 1).

| # | baseline | seeds | baseline mean | Mirror minus baseline | 95% interval (seed 0) | P7.1 | P7.2 |
|---|---|---|---|---|---|---|---|
| 3 | plain truncated SVD | 1 | 5.84 | 71.57 | [71.16, 71.97] | PASS |  |
| 4 | SVD-LLM-style whitened SVD | 1 | 78.03 | -0.62 | [-0.82, -0.42] | FAIL |  |
| 5 | ASVD-style activation scaling | 0 | not run | | | | |
| 6 | SliceGPT-style slicing | 0 | not run | | | | |
| 7 | FLAP-style pruning | 0 | not run | | | | |
| 8 | ZipIt-style merging | 0 | not run | | | | |
| 9 | pairing by same-image correlation | 0 | not run | | | | |
| 10 | pairing by weight cosine | 0 | not run | | | | |
| 11 | random pairing | 0 | not run | | | | |
| 12 | random structure | 0 | not run | | | | |

## Gate B (best baseline, method-selection set only)

Model `dinov2_vitb14_reg`, size full, fit seed 0: best **None**, second best **None**; INCOMPLETE, missing svd, svdllm, asvd_scale, slicegpt, flap, zipit, pair_actcorr, pair_wcos, pair_random, random_structure. Rule: rank methods 3 to 12 (every non-Mirror method; Mirror-local is an ablation) by method-selection k-NN top-1 at the full-model size, each at its own method-selection-chosen hyperparameters; ties go to the lower MUST-07 method number. Written to `gate_b.json`.

| rank | method | method-selection top-1 | hyperparameters |
|---|---|---|---|

## Paired bootstrap, Mirror-exact minus each method (validation, fit seed 0, 10,000 resamples)

| model | size | method | n | difference | 95% interval | excludes 0 |
|---|---|---|---|---|---|---|
| dinov2_vits14_reg | 6both | plain truncated SVD | 50000 | 49.62 | [49.17, 50.08] | yes |
| dinov2_vits14_reg | 6both | SVD-LLM-style whitened SVD | 50000 | -3.39 | [-3.79, -2.98] | yes |
| dinov2_vits14_reg | 6mlp | plain truncated SVD | 50000 | 50.46 | [49.99, 50.91] | yes |
| dinov2_vits14_reg | 6mlp | SVD-LLM-style whitened SVD | 50000 | -7.33 | [-7.68, -6.98] | yes |
| dinov2_vitb14_reg | 6both | plain truncated SVD | 50000 | 71.57 | [71.16, 71.97] | yes |
| dinov2_vitb14_reg | 6both | SVD-LLM-style whitened SVD | 50000 | -0.62 | [-0.82, -0.42] | yes |
| dinov2_vitb14_reg | 6mlp | plain truncated SVD | 50000 | 67.23 | [66.80, 67.64] | yes |
| dinov2_vitb14_reg | 6mlp | SVD-LLM-style whitened SVD | 50000 | -1.74 | [-1.97, -1.52] | yes |
| deit3_b16 | 6mlp | plain truncated SVD | 50000 | -0.24 | [-0.45, -0.02] | yes |
| deit3_b16 | 6mlp | SVD-LLM-style whitened SVD | 50000 | -1.75 | [-1.95, -1.55] | yes |
| dinov2_vitl14_reg | 12both | plain truncated SVD | 50000 | 12.59 | [12.21, 12.96] | yes |
| dinov2_vitl14_reg | 12both | SVD-LLM-style whitened SVD | 50000 | -4.01 | [-4.28, -3.73] | yes |
| dinov2_vitl14_reg | 12mlp | plain truncated SVD | 50000 | 0.30 | [0.08, 0.52] | yes |
| dinov2_vitl14_reg | 12mlp | SVD-LLM-style whitened SVD | 50000 | -1.13 | [-1.33, -0.92] | yes |
| deit3_b16 | 6both | plain truncated SVD | 50000 | 2.83 | [2.58, 3.08] | yes |
| deit3_b16 | 6both | SVD-LLM-style whitened SVD | 50000 | -1.11 | [-1.30, -0.91] | yes |
| mae_b16 | 6both | plain truncated SVD | 50000 | 5.62 | [5.35, 5.90] | yes |
| mae_b16 | 6both | SVD-LLM-style whitened SVD | 50000 | -2.26 | [-2.52, -2.01] | yes |
| mae_b16 | 6mlp | plain truncated SVD | 50000 | 4.86 | [4.61, 5.11] | yes |
| mae_b16 | 6mlp | SVD-LLM-style whitened SVD | 50000 | -4.91 | [-5.19, -4.64] | yes |
| dino_b16 | 6both | plain truncated SVD | 50000 | -5.87 | [-6.16, -5.58] | yes |
| dino_b16 | 6both | SVD-LLM-style whitened SVD | 50000 | -6.15 | [-6.44, -5.87] | yes |
| dino_b16 | 6mlp | plain truncated SVD | 50000 | -3.33 | [-3.58, -3.07] | yes |
| dino_b16 | 6mlp | SVD-LLM-style whitened SVD | 50000 | -3.38 | [-3.63, -3.13] | yes |
| siglip2_b16 | 6mlp | plain truncated SVD | 50000 | -3.59 | [-3.92, -3.28] | yes |
| siglip2_b16 | 6mlp | SVD-LLM-style whitened SVD | 50000 | -66.42 | [-66.84, -65.98] | yes |
| siglip2_b16 | 6both | plain truncated SVD | 50000 | 9.27 | [8.95, 9.59] | yes |
| siglip2_b16 | 6both | SVD-LLM-style whitened SVD | 50000 | -59.45 | [-59.90, -59.00] | yes |
| dinov2_vitb14 | 6both | plain truncated SVD | 50000 | 61.36 | [60.91, 61.79] | yes |
| dinov2_vitb14 | 6both | SVD-LLM-style whitened SVD | 50000 | -4.24 | [-4.52, -3.98] | yes |
| dinov2_vitb14 | 6mlp | plain truncated SVD | 50000 | 6.67 | [6.34, 7.00] | yes |
| dinov2_vitb14 | 6mlp | SVD-LLM-style whitened SVD | 50000 | -3.64 | [-3.89, -3.38] | yes |

## Notes

- Methods 1 and 9 to 12 use one span operator A* (Mirror-exact machinery); for the pairing controls only the MLP neuron pairing changes, the attention keeps the fitted head maps. Their parameter counts equal Mirror-exact's (gap 0).
- Mirror-local counts one d x d rotation per converted block and uses A_b for the attention output; it is an ablation and is not matched to the budget.
- Mirror-exact's basis change at a span end that is not a model boundary is counted (d x d); params_total_absorbed in the CSV gives the count with it absorbed into the next dense block.
- Compressor grid points are scored on the method-selection queries only; the chosen point is then scored on validation. Gate B uses the method-selection top-1 of each method's chosen point, so methods with larger grids get a small selection advantage on that set.
- Random pairing and random structure use seed random_seed + fit_seed, so their spread over fit seeds includes the random draw.
- Biases, norms and embeddings are counted as dense; analytic GFLOPs exclude biases, norms and activations.

Files: rows_<model>.csv, grid_<model>.csv, bootstrap.csv, preds/, info/, gate_b.json, meta.json.
