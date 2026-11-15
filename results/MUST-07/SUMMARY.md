# MUST-07: zero-shot comparison at matched parameters

Written by `scripts/must07_compressors.py` on 2026-11-15 19:27:19, commit `805c231c9c` (uncommitted changes). Compressors: **real**; budget accounting: **real**.

Protocol: k-NN top-1 with the 100k-image train bank (100 per class) and the 50k validation images as queries (CLS after the final norm, cosine, k = 20, T = 0.07, bf16 features). Method selection: the 10k method-selection train images as queries against the same bank; every hyperparameter and gate B use only these. Flip consistency: share of validation images whose prediction equals that of the mirrored image, in bf16 (column flip) and, for full-model rows, in fp32 (column flip fp32; bf16 rounding breaks exact ties, so exact invariance is judged on the fp32 value). Parameters: the converted matrices with basis changes; budget = Mirror-exact's count for the same span and components; gap = (method - budget) / budget. GFLOPs: analytic, per image, 2 x multiply-adds. Values are mean +- standard deviation over fit seeds (n given).

## deit3_b16

Original: top-1 82.97, flip 96.38 (bf16), flip fp32 n/a, method-selection top-1 92.02, 86,585,320 parameters, 35.1 GFLOPs.

### 6mlp (blocks 1 to 6, mlp)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 81.06 (n = 1) | 90.79 | 94.81 | n/a | 19,537,952 | 0.00% | 77,811,720 (0.899) | 31.7 |  |
| 2 | Mirror-local (ablation) | 81.56 (n = 1) | 91.08 | 94.82 | n/a | 21,977,930 | 12.49% | 80,251,698 (0.927) | 32.6 |  |
| 3 | plain truncated SVD | 81.29 (n = 1) | 90.75 | 94.99 | n/a | 19,541,760 | 0.02% | 77,815,528 (0.899) | 31.7 |  |
| 4 | SVD-LLM-style whitened SVD | 82.81 (n = 1) | 92.00 | 96.12 | n/a | 19,541,760 | 0.02% | 77,815,528 (0.899) | 31.7 | {"damp": 1e-06} |
| 6 | SliceGPT-style slicing | 76.19 (n = 1) | 86.03 | 91.08 | n/a | 19,574,784 | 0.19% | 77,848,552 (0.899) | 31.7 | {"basis": "shared", "pca": "normed"} |
| 12 | random structure | 6.00 (n = 1) | 6.67 | 27.55 | n/a | 19,537,952 | 0.00% | 77,811,720 (0.899) | 31.7 |  |

### full (blocks 1 to 12, mlp+attn)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 72.85 (n = 1) | 82.94 | 99.26 | 100.00 | 70,562,208 | 0.00% | 72,212,872 (0.834) | 29.5 |  |
| 2 | Mirror-local (ablation) | 81.20 (n = 1) | 90.81 | 94.71 | n/a | 75,764,794 | 7.37% | 77,415,458 (0.894) | 31.5 |  |
| 3 | plain truncated SVD | 74.73 (n = 1) | 82.24 | 91.59 | n/a | 70,562,304 | 0.00% | 72,212,968 (0.834) | 29.5 |  |
| 4 | SVD-LLM-style whitened SVD | 80.54 (n = 1) | 88.90 | 95.26 | n/a | 70,562,304 | 0.00% | 72,212,968 (0.834) | 29.5 | {"damp": 1e-06} |
| 6 | SliceGPT-style slicing | 79.50 (n = 1) | 88.03 | 94.68 | n/a | 70,697,112 | 0.19% | 72,347,776 (0.836) | 29.5 | {"basis": "per_block", "pca": "stream"} |
| 12 | random structure | 12.52 (n = 1) | 13.74 | 96.19 | 100.00 | 70,562,208 | 0.00% | 72,212,872 (0.834) | 29.5 |  |
fp32 flip pass: 5,000 validation queries and their mirrors against 10,000 bank images (TF32 off).

## dinov2_vitb14_reg

Original: top-1 78.94, flip 96.32 (bf16), flip fp32 96.30, method-selection top-1 80.21, 86,583,552 parameters, 47.1 GFLOPs.

### 6mlp (blocks 1 to 6, mlp)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 76.60 +- 0.02 (n = 3) | 78.05 | 91.70 | n/a | 17,034,432 | 0.00% | 75,306,432 (0.870) | 41.2 |  |
| 2 | Mirror-local (ablation) | 77.12 (n = 1) | 78.61 | 92.35 | n/a | 20,001,574 | 17.42% | 78,273,574 (0.904) | 42.7 |  |
| 3 | plain truncated SVD | 9.37 +- 0.00 (n = 3) | 10.60 | 42.91 | n/a | 17,038,080 | 0.02% | 75,310,080 (0.870) | 41.2 |  |
| 4 | SVD-LLM-style whitened SVD | 78.35 +- 0.03 (n = 3) | 79.58 | 95.43 | n/a | 17,038,080 | 0.02% | 75,310,080 (0.870) | 41.2 | {"damp": 1e-06} |
| 5 | ASVD-style activation scaling | 76.81 +- 0.04 (n = 3) | 77.60 | 93.22 | n/a | 17,038,080 | 0.02% | 75,310,080 (0.870) | 41.2 | {"alpha": 1.0} |
| 6 | SliceGPT-style slicing | 31.41 +- 1.23 (n = 3) | 33.53 | 41.87 | n/a | 17,068,032 | 0.20% | 75,340,032 (0.870) | 41.2 | {"basis": "shared", "pca": "normed"} |
| 7 | FLAP-style pruning | 75.20 +- 0.07 (n = 3) | 76.85 | 92.72 | n/a | 17,035,776 | 0.01% | 75,307,776 (0.870) | 41.2 | {"allocation": "global"} |
| 8 | ZipIt-style merging | 75.67 +- 0.15 (n = 3) | 77.48 | 92.08 | n/a | 17,035,776 | 0.01% | 75,307,776 (0.870) | 41.2 | {"alpha": 0.0} |
| 9 | pairing by same-image correlation | 27.42 +- 0.81 (n = 3) | 29.10 | 40.80 | n/a | 17,034,432 | 0.00% | 75,306,432 (0.870) | 41.2 |  |
| 10 | pairing by weight cosine | 34.42 +- 0.36 (n = 3) | 35.14 | 47.26 | n/a | 17,034,432 | 0.00% | 75,306,432 (0.870) | 41.2 |  |
| 11 | random pairing | 22.40 +- 0.69 (n = 3) | 24.06 | 34.51 | n/a | 17,034,432 | 0.00% | 75,306,432 (0.870) | 41.2 |  |
| 12 | random structure | 2.31 +- 0.50 (n = 3) | 2.56 | 8.69 | n/a | 17,034,432 | 0.00% | 75,306,432 (0.870) | 41.2 |  |

### 6both (blocks 1 to 6, mlp+attn)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 77.34 +- 0.08 (n = 3) | 78.83 | 95.02 | n/a | 25,668,000 | 0.00% | 69,784,224 (0.806) | 38.3 |  |
| 2 | Mirror-local (ablation) | 65.90 (n = 1) | 67.46 | 79.90 | n/a | 28,440,198 | 10.80% | 72,556,422 (0.838) | 39.8 |  |
| 3 | plain truncated SVD | 5.87 +- 0.03 (n = 3) | 6.41 | 29.79 | n/a | 25,668,096 | 0.00% | 69,784,320 (0.806) | 38.3 |  |
| 4 | SVD-LLM-style whitened SVD | 78.02 +- 0.01 (n = 3) | 79.49 | 95.16 | n/a | 25,668,096 | 0.00% | 69,784,320 (0.806) | 38.3 | {"damp": 1e-06} |
| 5 | ASVD-style activation scaling | 74.41 +- 0.02 (n = 3) | 75.89 | 90.69 | n/a | 25,668,096 | 0.00% | 69,784,320 (0.806) | 38.3 | {"alpha": 1.0} |
| 6 | SliceGPT-style slicing | 75.60 +- 0.11 (n = 3) | 76.61 | 91.94 | n/a | 25,729,836 | 0.24% | 69,846,060 (0.807) | 38.3 | {"basis": "per_block", "pca": "normed"} |
| 7 | FLAP-style pruning | 70.29 +- 0.18 (n = 3) | 71.94 | 88.89 | n/a | 25,668,096 | 0.00% | 69,784,320 (0.806) | 37.8 | {"allocation": "uniform"} |
| 8 | ZipIt-style merging | 18.51 +- 1.20 (n = 3) | 21.13 | 40.30 | n/a | 25,668,096 | 0.00% | 69,784,320 (0.806) | 37.8 | {"alpha": 0.0} |
| 9 | pairing by same-image correlation | 31.31 +- 1.53 (n = 3) | 33.43 | 54.50 | n/a | 25,668,000 | 0.00% | 69,784,224 (0.806) | 38.3 |  |
| 10 | pairing by weight cosine | 39.85 +- 0.32 (n = 3) | 41.19 | 62.55 | n/a | 25,668,000 | 0.00% | 69,784,224 (0.806) | 38.3 |  |
| 11 | random pairing | 32.30 +- 3.30 (n = 3) | 35.92 | 56.87 | n/a | 25,668,000 | 0.00% | 69,784,224 (0.806) | 38.3 |  |
| 12 | random structure | 2.29 +- 0.42 (n = 3) | 2.87 | 13.73 | n/a | 25,668,000 | 0.00% | 69,784,224 (0.806) | 38.3 |  |

### 9both (blocks 1 to 9, mlp+attn)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 73.37 +- 0.11 (n = 3) | 74.94 | 94.03 | n/a | 43,442,640 | 0.00% | 66,325,200 (0.766) | 36.5 |  |
| 2 | Mirror-local (ablation) | 62.26 (n = 1) | 63.66 | 76.81 | n/a | 47,273,228 | 8.82% | 70,155,788 (0.810) | 38.5 |  |
| 3 | plain truncated SVD | 5.67 (n = 1) | 6.16 | 27.24 | n/a | 43,442,688 | 0.00% | 66,325,248 (0.766) | 36.5 |  |
| 4 | SVD-LLM-style whitened SVD | 77.35 (n = 1) | 78.77 | 94.26 | n/a | 43,442,688 | 0.00% | 66,325,248 (0.766) | 36.5 | {"damp": 1e-06} |
| 5 | ASVD-style activation scaling | 65.24 (n = 1) | 67.75 | 80.67 | n/a | 43,442,688 | 0.00% | 66,325,248 (0.766) | 36.5 | {"alpha": 1.0} |
| 6 | SliceGPT-style slicing | 72.48 (n = 1) | 73.92 | 88.57 | n/a | 43,522,176 | 0.18% | 66,404,736 (0.767) | 36.5 | {"basis": "per_block", "pca": "stream"} |
| 7 | FLAP-style pruning | 63.47 (n = 1) | 65.33 | 83.33 | n/a | 43,442,688 | 0.00% | 66,325,248 (0.766) | 35.9 | {"allocation": "uniform"} |
| 8 | ZipIt-style merging | 13.38 (n = 1) | 14.63 | 45.98 | n/a | 43,442,688 | 0.00% | 66,325,248 (0.766) | 35.9 | {"alpha": 0.0} |
| 9 | pairing by same-image correlation | 13.36 (n = 1) | 14.24 | 62.80 | n/a | 43,442,640 | 0.00% | 66,325,200 (0.766) | 36.5 |  |
| 10 | pairing by weight cosine | 16.94 (n = 1) | 18.61 | 62.87 | n/a | 43,442,640 | 0.00% | 66,325,200 (0.766) | 36.5 |  |
| 11 | random pairing | 11.07 (n = 1) | 12.23 | 56.40 | n/a | 43,442,640 | 0.00% | 66,325,200 (0.766) | 36.5 |  |
| 12 | random structure | 4.31 (n = 1) | 4.67 | 76.77 | n/a | 43,442,640 | 0.00% | 66,325,200 (0.766) | 36.5 |  |

### full (blocks 1 to 12, mlp+attn)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 53.81 +- 0.36 (n = 3) | 55.29 | 97.98 | 100.00 | 64,554,464 | 0.00% | 66,203,360 (0.765) | 36.4 |  |
| 2 | Mirror-local (ablation) | 62.10 (n = 1) | 63.26 | 76.43 | 74.22 | 69,374,408 | 7.47% | 71,023,304 (0.820) | 39.0 |  |
| 3 | plain truncated SVD | 5.17 (n = 1) | 5.67 | 25.82 | 31.54 | 64,555,008 | 0.00% | 66,203,904 (0.765) | 36.4 |  |
| 4 | SVD-LLM-style whitened SVD | 75.80 (n = 1) | 77.36 | 93.81 | 92.10 | 64,555,008 | 0.00% | 66,203,904 (0.765) | 36.4 | {"damp": 1e-06} |
| 5 | ASVD-style activation scaling | 57.09 (n = 1) | 59.70 | 74.76 | 69.96 | 64,555,008 | 0.00% | 66,203,904 (0.765) | 36.4 | {"alpha": 1.0} |
| 6 | SliceGPT-style slicing | 68.33 (n = 1) | 69.53 | 86.89 | 85.42 | 64,675,800 | 0.19% | 66,324,696 (0.766) | 36.5 | {"basis": "per_block", "pca": "stream"} |
| 7 | FLAP-style pruning | 65.12 (n = 1) | 67.00 | 85.90 | 84.62 | 64,555,008 | 0.00% | 66,203,904 (0.765) | 35.9 | {"allocation": "uniform"} |
| 8 | ZipIt-style merging | 13.58 (n = 1) | 15.17 | 42.08 | 45.62 | 64,555,008 | 0.00% | 66,203,904 (0.765) | 35.9 | {"alpha": 0.5} |
| 9 | pairing by same-image correlation | 12.17 (n = 1) | 13.88 | 94.57 | 100.00 | 64,554,464 | 0.00% | 66,203,360 (0.765) | 36.4 |  |
| 10 | pairing by weight cosine | 14.98 (n = 1) | 16.48 | 94.53 | 100.00 | 64,554,464 | 0.00% | 66,203,360 (0.765) | 36.4 |  |
| 11 | random pairing | 10.02 (n = 1) | 11.31 | 92.82 | 100.00 | 64,554,464 | 0.00% | 66,203,360 (0.765) | 36.4 |  |
| 12 | random structure | 4.78 (n = 1) | 5.08 | 93.44 | 100.00 | 64,554,464 | 0.00% | 66,203,360 (0.765) | 36.4 |  |
fp32 flip pass: 50,000 validation queries and their mirrors against 100,000 bank images (TF32 off).

## dinov2_vitl14_reg

Original: top-1 80.59, flip 97.74 (bf16), flip fp32 n/a, method-selection top-1 81.89, 304,372,736 parameters, 164.6 GFLOPs.

### 6mlp (blocks 1 to 6, mlp)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 80.36 (n = 1) | 81.67 | 96.94 | n/a | 29,536,180 | 0.00% | 283,577,268 (0.932) | 153.8 |  |
| 2 | Mirror-local (ablation) | 80.21 (n = 1) | 81.40 | 95.23 | n/a | 34,215,230 | 15.84% | 288,256,318 (0.947) | 156.2 |  |
| 3 | plain truncated SVD | 80.57 (n = 1) | 81.75 | 97.49 | n/a | 29,537,280 | 0.00% | 283,578,368 (0.932) | 153.8 |  |
| 4 | SVD-LLM-style whitened SVD | 80.54 (n = 1) | 81.95 | 97.79 | n/a | 29,537,280 | 0.00% | 283,578,368 (0.932) | 153.8 | {"damp": 1e-06} |
| 6 | SliceGPT-style slicing | 80.47 (n = 1) | 81.60 | 96.94 | n/a | 29,556,736 | 0.07% | 283,597,824 (0.932) | 153.8 | {"basis": "shared", "pca": "normed"} |
| 12 | random structure | 2.29 (n = 1) | 2.42 | 5.64 | n/a | 29,536,180 | 0.00% | 283,577,268 (0.932) | 153.8 |  |

### full (blocks 1 to 24, mlp+attn)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 30.17 (n = 1) | 31.51 | 95.91 | 100.00 | 246,026,620 | 0.00% | 248,409,468 (0.816) | 135.4 |  |
| 2 | Mirror-local (ablation) | 70.18 (n = 1) | 71.53 | 83.06 | n/a | 255,060,290 | 3.67% | 257,443,138 (0.846) | 140.1 |  |
| 3 | plain truncated SVD | 60.62 (n = 1) | 63.19 | 75.92 | n/a | 246,030,336 | 0.00% | 248,413,184 (0.816) | 135.4 |  |
| 4 | SVD-LLM-style whitened SVD | 80.54 (n = 1) | 81.79 | 97.30 | n/a | 246,030,336 | 0.00% | 248,413,184 (0.816) | 135.4 | {"damp": 1e-06} |
| 6 | SliceGPT-style slicing | 79.48 (n = 1) | 80.89 | 94.95 | n/a | 246,251,520 | 0.09% | 248,634,368 (0.817) | 135.5 | {"basis": "shared", "pca": "normed"} |
| 12 | random structure | 3.87 (n = 1) | 3.88 | 91.74 | 100.00 | 246,026,620 | 0.00% | 248,409,468 (0.816) | 135.4 |  |
fp32 flip pass: 5,000 validation queries and their mirrors against 10,000 bank images (TF32 off).

## Predictions P7.1 and P7.2 (PREREG_NEXT_PHASE.md)

Model `dinov2_vitb14_reg`. P7.1: Mirror-exact beats every baseline (methods 3 to 12) by at least 2 points (mean over fit seeds 0, 1 and 2) with the fit-seed-0 paired bootstrap interval above 0. P7.2: ZipIt-style merging (8) and same-image-correlation pairing (9) trail Mirror-exact by at least 10 points. Each 6-block size is scored separately.

### 6mlp: P7.1 FAIL, P7.2 FAIL

Mirror-exact: 76.60 +- 0.02 (n = 3).

| # | baseline | seeds | baseline mean | Mirror minus baseline | 95% interval (seed 0) | P7.1 | P7.2 |
|---|---|---|---|---|---|---|---|
| 3 | plain truncated SVD | 3 | 9.37 | 67.24 | [66.80, 67.64] | PASS |  |
| 4 | SVD-LLM-style whitened SVD | 3 | 78.35 | -1.75 | [-1.97, -1.52] | FAIL |  |
| 5 | ASVD-style activation scaling | 3 | 76.81 | -0.20 | [-0.49, -0.04] | FAIL |  |
| 6 | SliceGPT-style slicing | 3 | 31.41 | 45.19 | [44.03, 44.97] | PASS |  |
| 7 | FLAP-style pruning | 3 | 75.20 | 1.41 | [1.22, 1.71] | FAIL |  |
| 8 | ZipIt-style merging | 3 | 75.67 | 0.93 | [0.58, 1.05] | FAIL | FAIL |
| 9 | pairing by same-image correlation | 3 | 27.42 | 49.19 | [48.53, 49.47] | PASS | PASS |
| 10 | pairing by weight cosine | 3 | 34.42 | 42.19 | [42.12, 43.05] | PASS |  |
| 11 | random pairing | 3 | 22.40 | 54.20 | [53.18, 54.12] | PASS |  |
| 12 | random structure | 3 | 2.31 | 74.29 | [73.95, 74.72] | PASS |  |

### 6both: P7.1 FAIL, P7.2 PASS

Mirror-exact: 77.34 +- 0.08 (n = 3).

| # | baseline | seeds | baseline mean | Mirror minus baseline | 95% interval (seed 0) | P7.1 | P7.2 |
|---|---|---|---|---|---|---|---|
| 3 | plain truncated SVD | 3 | 5.87 | 71.47 | [71.16, 71.97] | PASS |  |
| 4 | SVD-LLM-style whitened SVD | 3 | 78.02 | -0.68 | [-0.82, -0.42] | FAIL |  |
| 5 | ASVD-style activation scaling | 3 | 74.41 | 2.92 | [2.71, 3.24] | PASS |  |
| 6 | SliceGPT-style slicing | 3 | 75.60 | 1.74 | [1.44, 1.94] | FAIL |  |
| 7 | FLAP-style pruning | 3 | 70.29 | 7.05 | [6.93, 7.58] | PASS |  |
| 8 | ZipIt-style merging | 3 | 18.51 | 58.83 | [57.47, 58.39] | PASS | PASS |
| 9 | pairing by same-image correlation | 3 | 31.31 | 46.03 | [45.01, 45.95] | PASS | PASS |
| 10 | pairing by weight cosine | 3 | 39.85 | 37.48 | [36.89, 37.81] | PASS |  |
| 11 | random pairing | 3 | 32.30 | 45.04 | [42.33, 43.27] | PASS |  |
| 12 | random structure | 3 | 2.29 | 75.05 | [74.28, 75.06] | PASS |  |

## Gate B (best baseline, method-selection set only)

Model `dinov2_vitb14_reg`, size full, fit seed 0: best **svdllm**, second best **slicegpt**. Rule: rank methods 3 to 12 (every non-Mirror method; Mirror-local is an ablation) by method-selection k-NN top-1 at the full-model size, each at its own method-selection-chosen hyperparameters; ties go to the lower MUST-07 method number. Written to `gate_b.json`.

| rank | method | method-selection top-1 | hyperparameters |
|---|---|---|---|
| 1 | SVD-LLM-style whitened SVD | 77.36 | {"damp": 1e-06} |
| 2 | SliceGPT-style slicing | 69.53 | {"basis": "per_block", "pca": "stream"} |
| 3 | FLAP-style pruning | 67.00 | {"allocation": "uniform"} |
| 4 | ASVD-style activation scaling | 59.70 | {"alpha": 1.0} |
| 5 | pairing by weight cosine | 16.48 | {} |
| 6 | ZipIt-style merging | 15.17 | {"alpha": 0.5} |
| 7 | pairing by same-image correlation | 13.88 | {} |
| 8 | random pairing | 11.31 | {} |
| 9 | plain truncated SVD | 5.67 | {} |
| 10 | random structure | 5.08 | {} |

## Paired bootstrap, Mirror-exact minus each method (validation, fit seed 0, 10,000 resamples)

| model | size | method | n | difference | 95% interval | excludes 0 |
|---|---|---|---|---|---|---|
| dinov2_vitb14_reg | full | Mirror-local (ablation) | 50000 | -8.22 | [-8.64, -7.82] | yes |
| dinov2_vitb14_reg | full | plain truncated SVD | 50000 | 48.71 | [48.24, 49.16] | yes |
| dinov2_vitb14_reg | full | SVD-LLM-style whitened SVD | 50000 | -21.92 | [-22.35, -21.50] | yes |
| dinov2_vitb14_reg | full | ASVD-style activation scaling | 50000 | -3.21 | [-3.65, -2.78] | yes |
| dinov2_vitb14_reg | full | SliceGPT-style slicing | 50000 | -14.45 | [-14.90, -14.02] | yes |
| dinov2_vitb14_reg | full | FLAP-style pruning | 50000 | -11.24 | [-11.67, -10.82] | yes |
| dinov2_vitb14_reg | full | ZipIt-style merging | 50000 | 40.30 | [39.83, 40.75] | yes |
| dinov2_vitb14_reg | full | pairing by same-image correlation | 50000 | 41.71 | [41.24, 42.16] | yes |
| dinov2_vitb14_reg | full | pairing by weight cosine | 50000 | 38.90 | [38.44, 39.35] | yes |
| dinov2_vitb14_reg | full | random pairing | 50000 | 43.86 | [43.39, 44.31] | yes |
| dinov2_vitb14_reg | full | random structure | 50000 | 49.10 | [48.64, 49.54] | yes |
| deit3_b16 | 6mlp | Mirror-local (ablation) | 50000 | -0.50 | [-0.67, -0.33] | yes |
| deit3_b16 | 6mlp | plain truncated SVD | 50000 | -0.24 | [-0.45, -0.02] | yes |
| deit3_b16 | 6mlp | SVD-LLM-style whitened SVD | 50000 | -1.75 | [-1.95, -1.55] | yes |
| deit3_b16 | 6mlp | SliceGPT-style slicing | 50000 | 4.86 | [4.59, 5.13] | yes |
| deit3_b16 | 6mlp | random structure | 50000 | 75.05 | [74.66, 75.43] | yes |
| deit3_b16 | full | Mirror-local (ablation) | 50000 | -8.35 | [-8.66, -8.04] | yes |
| deit3_b16 | full | plain truncated SVD | 50000 | -1.88 | [-2.20, -1.56] | yes |
| deit3_b16 | full | SVD-LLM-style whitened SVD | 50000 | -7.70 | [-8.01, -7.38] | yes |
| deit3_b16 | full | SliceGPT-style slicing | 50000 | -6.65 | [-6.97, -6.33] | yes |
| deit3_b16 | full | random structure | 50000 | 60.33 | [59.88, 60.78] | yes |
| dinov2_vitb14_reg | 6mlp | Mirror-local (ablation) | 50000 | -0.53 | [-0.74, -0.32] | yes |
| dinov2_vitb14_reg | 6mlp | plain truncated SVD | 50000 | 67.23 | [66.80, 67.64] | yes |
| dinov2_vitb14_reg | 6mlp | SVD-LLM-style whitened SVD | 50000 | -1.74 | [-1.97, -1.52] | yes |
| dinov2_vitb14_reg | 6mlp | ASVD-style activation scaling | 50000 | -0.26 | [-0.49, -0.04] | yes |
| dinov2_vitb14_reg | 6mlp | SliceGPT-style slicing | 50000 | 44.50 | [44.03, 44.97] | yes |
| dinov2_vitb14_reg | 6mlp | FLAP-style pruning | 50000 | 1.46 | [1.22, 1.71] | yes |
| dinov2_vitb14_reg | 6both | Mirror-local (ablation) | 50000 | 11.51 | [11.16, 11.86] | yes |
| dinov2_vitb14_reg | 6both | plain truncated SVD | 50000 | 71.57 | [71.16, 71.97] | yes |
| dinov2_vitb14_reg | 6both | SVD-LLM-style whitened SVD | 50000 | -0.62 | [-0.82, -0.42] | yes |
| dinov2_vitb14_reg | 6both | ASVD-style activation scaling | 50000 | 2.98 | [2.71, 3.24] | yes |
| dinov2_vitb14_reg | 6both | SliceGPT-style slicing | 50000 | 1.69 | [1.44, 1.94] | yes |
| dinov2_vitb14_reg | 6both | FLAP-style pruning | 50000 | 7.25 | [6.93, 7.58] | yes |
| dinov2_vitb14_reg | 6mlp | ZipIt-style merging | 50000 | 0.82 | [0.58, 1.05] | yes |
| dinov2_vitb14_reg | 6mlp | pairing by same-image correlation | 50000 | 49.01 | [48.53, 49.47] | yes |
| dinov2_vitb14_reg | 6mlp | pairing by weight cosine | 50000 | 42.58 | [42.12, 43.05] | yes |
| dinov2_vitb14_reg | 6mlp | random pairing | 50000 | 53.65 | [53.18, 54.12] | yes |
| dinov2_vitb14_reg | 6mlp | random structure | 50000 | 74.34 | [73.95, 74.72] | yes |
| dinov2_vitl14_reg | 6mlp | Mirror-local (ablation) | 50000 | 0.15 | [-0.00, 0.31] | no |
| dinov2_vitl14_reg | 6mlp | plain truncated SVD | 50000 | -0.21 | [-0.34, -0.07] | yes |
| dinov2_vitl14_reg | 6mlp | SVD-LLM-style whitened SVD | 50000 | -0.17 | [-0.30, -0.04] | yes |
| dinov2_vitl14_reg | 6mlp | SliceGPT-style slicing | 50000 | -0.11 | [-0.25, 0.04] | no |
| dinov2_vitl14_reg | 6mlp | random structure | 50000 | 78.07 | [77.70, 78.44] | yes |
| dinov2_vitl14_reg | full | Mirror-local (ablation) | 50000 | -40.01 | [-40.48, -39.55] | yes |
| dinov2_vitl14_reg | full | plain truncated SVD | 50000 | -30.45 | [-30.92, -29.98] | yes |
| dinov2_vitl14_reg | full | SVD-LLM-style whitened SVD | 50000 | -50.37 | [-50.84, -49.91] | yes |
| dinov2_vitl14_reg | full | SliceGPT-style slicing | 50000 | -49.31 | [-49.77, -48.84] | yes |
| dinov2_vitl14_reg | full | random structure | 50000 | 26.30 | [25.88, 26.71] | yes |
| dinov2_vitb14_reg | 6both | pairing by same-image correlation | 50000 | 45.49 | [45.01, 45.95] | yes |
| dinov2_vitb14_reg | 6both | pairing by weight cosine | 50000 | 37.35 | [36.89, 37.81] | yes |
| dinov2_vitb14_reg | 6both | random pairing | 50000 | 42.80 | [42.33, 43.27] | yes |
| dinov2_vitb14_reg | 6both | random structure | 50000 | 74.67 | [74.28, 75.06] | yes |
| dinov2_vitb14_reg | 6both | ZipIt-style merging | 50000 | 57.93 | [57.47, 58.39] | yes |
| dinov2_vitb14_reg | 9both | Mirror-local (ablation) | 50000 | 11.09 | [10.72, 11.45] | yes |
| dinov2_vitb14_reg | 9both | plain truncated SVD | 50000 | 67.68 | [67.25, 68.11] | yes |
| dinov2_vitb14_reg | 9both | SVD-LLM-style whitened SVD | 50000 | -4.00 | [-4.28, -3.72] | yes |
| dinov2_vitb14_reg | 9both | ASVD-style activation scaling | 50000 | 8.11 | [7.75, 8.46] | yes |
| dinov2_vitb14_reg | 9both | SliceGPT-style slicing | 50000 | 0.87 | [0.57, 1.18] | yes |
| dinov2_vitb14_reg | 9both | FLAP-style pruning | 50000 | 9.88 | [9.51, 10.24] | yes |
| dinov2_vitb14_reg | 9both | pairing by same-image correlation | 50000 | 59.99 | [59.54, 60.44] | yes |
| dinov2_vitb14_reg | 9both | pairing by weight cosine | 50000 | 56.41 | [55.95, 56.86] | yes |
| dinov2_vitb14_reg | 9both | random pairing | 50000 | 62.28 | [61.83, 62.71] | yes |
| dinov2_vitb14_reg | 9both | random structure | 50000 | 69.04 | [68.63, 69.46] | yes |
| dinov2_vitb14_reg | 9both | ZipIt-style merging | 50000 | 59.97 | [59.52, 60.41] | yes |

## Notes

- Methods 1 and 9 to 12 use one span operator A* (Mirror-exact machinery); for the pairing controls only the MLP neuron pairing changes, the attention keeps the fitted head maps. Their parameter counts equal Mirror-exact's (gap 0).
- Mirror-local counts one d x d rotation per converted block and uses A_b for the attention output; it is an ablation and is not matched to the budget.
- Mirror-exact's basis change at a span end that is not a model boundary is counted (d x d); params_total_absorbed in the CSV gives the count with it absorbed into the next dense block.
- Compressor grid points are scored on the method-selection queries only; the chosen point is then scored on validation. Gate B uses the method-selection top-1 of each method's chosen point, so methods with larger grids get a small selection advantage on that set.
- Random pairing and random structure use seed random_seed + fit_seed, so their spread over fit seeds includes the random draw.
- Biases, norms and embeddings are counted as dense; analytic GFLOPs exclude biases, norms and activations.

Files: rows_<model>.csv, grid_<model>.csv, bootstrap.csv, preds/, info/, gate_b.json, meta.json.
