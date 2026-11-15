# CORE-03: Mirror and low rank stack (zero-shot, full model)

Written by `scripts/core03_stacking.py` on 2026-11-15 19:43:17, commit `805c231c9c` (uncommitted changes); fit seed 0.

Protocol: k-NN top-1 with the 100k-image train bank and the 50k validation images (CLS after the final norm, cosine, k = 20, T = 0.07, bf16 features); every choice uses the 10k method-selection train images only. Parameters: the converted matrices of every block (MLP and attention) with basis changes (none at full conversion); fraction = parameters / dense count of the same matrices. Budget points: ceil(f x dense count). GFLOPs: analytic, per image, 2 x multiply-adds (MUST-07 runner).

## P3.1: FAIL

Rule: Mirror plus low rank strictly above low rank alone (validation top-1) at >= 3 of 4 budget points, on >= 2 of ['dinov2_vitb14_reg', 'dinov2_vitl14_reg', 'deit3_b16'], including one of ['dinov2_vitl14_reg', 'deit3_b16'].

Scored Mirror plus low rank: inner method **svd**. Low rank alone: the better of its methods on the method-selection set at each point.

| model | points scored | points above | status |
|---|---|---|---|
| dinov2_vitb14_reg | 4 | 0 (none) | fail |
| dinov2_vitl14_reg | 4 | 0 (none) | fail |
| deit3_b16 | 4 | 0 (none) | fail |

## deit3_b16

Original: top-1 82.97, method-selection 92.02, span 84,934,656 weights, 35.13 GFLOPs.
Mirror alone (its tax): 70,562,208 weights = 83.08% of the span, top-1 72.85, method-selection 82.94, flip fp32 100.00, 29.46 GFLOPs.

| point | budget | Mirror + low rank (scored) | weights | top-1 | msel | low rank alone (chosen) | weights | top-1 | msel | difference | 95% interval | above |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f0.85 | 72,194,458 | mirror_svd (Mirror-exact itself, below budget) | 70,562,208 | 72.85 | 82.94 | svdllm | 72,195,072 | 80.83 | 89.32 | -7.99 | [-8.30, -7.67] | no |
| f0.70 | 59,454,260 | mirror_svd | 59,455,438 | 56.86 | 63.21 | svdllm | 59,455,488 | 77.43 | 85.29 | -20.57 | [-20.97, -20.16] | no |
| f0.55 | 46,714,061 | mirror_svd | 46,714,332 | 40.80 | 44.43 | svdllm | 46,714,368 | 68.70 | 75.19 | -27.89 | [-28.35, -27.45] | no |
| f0.40 | 33,973,863 | mirror_svd | 33,974,676 | 20.58 | 23.05 | svdllm | 33,974,784 | 46.68 | 50.63 | -26.10 | [-26.56, -25.66] | no |

Every row (secondary curves included):

| point | curve | method | weights | fraction | gap | top-1 | msel | flip | flip fp32 | GFLOPs | check |
|---|---|---|---|---|---|---|---|---|---|---|---|
| original | original | original | 84,934,656 | 1.000 | 0.000% | 82.97 | 92.02 | 96.38 | n/a | 35.13 |  |
| mirror | mirror | mirror_exact | 70,562,208 | 0.831 | 0.000% | 72.85 | 82.94 | 99.26 | 100.00 | 29.46 |  |
| f0.85 | lowrank | svd | 72,195,072 | 0.850 | 0.001% | 75.41 | 83.43 | n/a | n/a | 30.11 | True |
| f0.85 | lowrank | svdllm | 72,195,072 | 0.850 | 0.001% | 80.83 | 89.32 | n/a | n/a | 30.11 | True |
| f0.85 | mirror_lowrank | mirror_svd | 70,562,208 | 0.831 | -2.261% | 72.85 | 82.94 | 99.26 | 100.00 | 29.46 |  |
| f0.85 | mirror_lowrank | mirror_svdllm | 70,562,208 | 0.831 | -2.261% | 72.85 | 82.94 | 99.26 | 100.00 | 29.46 |  |
| f0.70 | lowrank | svd | 59,455,488 | 0.700 | 0.002% | 65.31 | 71.41 | n/a | n/a | 25.09 | True |
| f0.70 | lowrank | svdllm | 59,455,488 | 0.700 | 0.002% | 77.43 | 85.29 | n/a | n/a | 25.09 | True |
| f0.70 | mirror_lowrank | mirror_svd | 59,455,438 | 0.700 | 0.002% | 56.86 | 63.21 | 98.55 | 100.00 | 25.09 | True |
| f0.70 | mirror_lowrank | mirror_svdllm | 59,455,438 | 0.700 | 0.002% | 64.87 | 72.65 | 98.70 | 100.00 | 25.09 | True |
| f0.55 | lowrank | svd | 46,714,368 | 0.550 | 0.001% | 36.55 | 39.07 | n/a | n/a | 20.07 | True |
| f0.55 | lowrank | svdllm | 46,714,368 | 0.550 | 0.001% | 68.70 | 75.19 | n/a | n/a | 20.07 | True |
| f0.55 | mirror_lowrank | mirror_svd | 46,714,332 | 0.550 | 0.001% | 40.80 | 44.43 | 97.43 | 99.98 | 20.07 | True |
| f0.55 | mirror_lowrank | mirror_svdllm | 46,714,332 | 0.550 | 0.001% | 55.20 | 60.99 | 97.86 | 100.00 | 20.07 | True |
| f0.40 | lowrank | svd | 33,974,784 | 0.400 | 0.003% | 13.23 | 14.53 | n/a | n/a | 15.05 | True |
| f0.40 | lowrank | svdllm | 33,974,784 | 0.400 | 0.003% | 46.68 | 50.63 | n/a | n/a | 15.05 | True |
| f0.40 | mirror_lowrank | mirror_svd | 33,974,676 | 0.400 | 0.002% | 20.58 | 23.05 | 96.57 | 100.00 | 15.05 | True |
| f0.40 | mirror_lowrank | mirror_svdllm | 33,974,676 | 0.400 | 0.002% | 37.89 | 41.65 | 96.96 | 100.00 | 15.05 | True |

## dinov2_vitb14_reg

Original: top-1 78.94, method-selection 80.21, span 84,934,656 weights, 47.08 GFLOPs.
Mirror alone (its tax): 64,554,464 weights = 76.00% of the span, top-1 53.88, method-selection 55.29, flip fp32 100.00, 36.44 GFLOPs.

| point | budget | Mirror + low rank (scored) | weights | top-1 | msel | low rank alone (chosen) | weights | top-1 | msel | difference | 95% interval | above |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f0.85 | 72,194,458 | mirror_svd (Mirror-exact itself, below budget) | 64,554,464 | 53.88 | 55.29 | svdllm | 72,195,072 | 77.91 | 79.51 | -24.03 | [-24.47, -23.61] | no |
| f0.70 | 59,454,260 | mirror_svd | 59,454,578 | 6.19 | 6.89 | svdllm | 59,455,488 | 74.02 | 75.45 | -67.83 | [-68.24, -67.41] | no |
| f0.55 | 46,714,061 | mirror_svd | 46,714,766 | 2.69 | 2.85 | svdllm | 46,717,440 | 60.98 | 62.40 | -58.29 | [-58.74, -57.86] | no |
| f0.40 | 33,973,863 | mirror_svd | 33,975,312 | 5.39 | 5.98 | svdllm | 33,974,016 | 15.55 | 17.20 | -10.16 | [-10.48, -9.83] | no |

Every row (secondary curves included):

| point | curve | method | weights | fraction | gap | top-1 | msel | flip | flip fp32 | GFLOPs | check |
|---|---|---|---|---|---|---|---|---|---|---|---|
| original | original | original | 84,934,656 | 1.000 | 0.000% | 78.94 | 80.21 | 96.32 | n/a | 47.08 |  |
| mirror | mirror | mirror_exact | 64,554,464 | 0.760 | 0.000% | 53.88 | 55.29 | 97.98 | 100.00 | 36.44 |  |
| f0.85 | lowrank | svd | 72,195,072 | 0.850 | 0.001% | 3.73 | 4.09 | n/a | n/a | 40.43 | True |
| f0.85 | lowrank | svdllm | 72,195,072 | 0.850 | 0.001% | 77.91 | 79.51 | n/a | n/a | 40.43 | True |
| f0.85 | mirror_lowrank | mirror_svd | 64,554,464 | 0.760 | -10.583% | 53.88 | 55.29 | 97.98 | 100.00 | 36.44 |  |
| f0.85 | mirror_lowrank | mirror_svdllm | 64,554,464 | 0.760 | -10.583% | 53.88 | 55.29 | 97.98 | 100.00 | 36.44 |  |
| f0.70 | lowrank | svd | 59,455,488 | 0.700 | 0.002% | 6.03 | 6.52 | n/a | n/a | 33.78 | True |
| f0.70 | lowrank | svdllm | 59,455,488 | 0.700 | 0.002% | 74.02 | 75.45 | n/a | n/a | 33.78 | True |
| f0.70 | mirror_lowrank | mirror_svd | 59,454,578 | 0.700 | 0.001% | 6.19 | 6.89 | 76.90 | 99.72 | 33.78 | True |
| f0.70 | mirror_lowrank | mirror_svdllm | 59,454,578 | 0.700 | 0.001% | 46.77 | 47.81 | 97.60 | 99.98 | 33.78 | True |
| f0.55 | lowrank | svd | 46,717,440 | 0.550 | 0.007% | 5.75 | 6.23 | n/a | n/a | 27.13 | True |
| f0.55 | lowrank | svdllm | 46,717,440 | 0.550 | 0.007% | 60.98 | 62.40 | n/a | n/a | 27.13 | True |
| f0.55 | mirror_lowrank | mirror_svd | 46,714,766 | 0.550 | 0.002% | 2.69 | 2.85 | 48.86 | 99.78 | 27.13 | True |
| f0.55 | mirror_lowrank | mirror_svdllm | 46,714,766 | 0.550 | 0.002% | 36.58 | 38.20 | 96.48 | 100.00 | 27.13 | True |
| f0.40 | lowrank | svd | 33,974,016 | 0.400 | 0.000% | 4.95 | 5.50 | n/a | n/a | 20.48 | True |
| f0.40 | lowrank | svdllm | 33,974,016 | 0.400 | 0.000% | 15.55 | 17.20 | n/a | n/a | 20.48 | True |
| f0.40 | mirror_lowrank | mirror_svd | 33,975,312 | 0.400 | 0.004% | 5.39 | 5.98 | 80.15 | 99.20 | 20.48 | True |
| f0.40 | mirror_lowrank | mirror_svdllm | 33,975,312 | 0.400 | 0.004% | 16.83 | 17.95 | 93.09 | 100.00 | 20.48 | True |

## dinov2_vitl14_reg

Original: top-1 80.59, method-selection 81.89, span 301,989,888 weights, 164.64 GFLOPs.
Mirror alone (its tax): 246,026,620 weights = 81.47% of the span, top-1 30.17, method-selection 31.51, flip fp32 100.00, 135.43 GFLOPs.

| point | budget | Mirror + low rank (scored) | weights | top-1 | msel | low rank alone (chosen) | weights | top-1 | msel | difference | 95% interval | above |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f0.85 | 256,691,405 | mirror_svd (Mirror-exact itself, below budget) | 246,026,620 | 30.17 | 31.51 | svdllm | 256,692,224 | 80.61 | 81.77 | -50.44 | [-50.90, -49.98] | no |
| f0.70 | 211,392,922 | mirror_svd | 211,396,432 | 12.03 | 13.47 | svdllm | 211,394,560 | 79.68 | 81.14 | -67.65 | [-68.07, -67.23] | no |
| f0.55 | 166,094,439 | mirror_svd | 166,095,336 | 9.79 | 11.12 | svdllm | 166,094,848 | 74.82 | 76.56 | -65.02 | [-65.45, -64.59] | no |
| f0.40 | 120,795,956 | mirror_svd | 120,797,312 | 9.69 | 10.69 | svdllm | 120,798,208 | 53.77 | 55.96 | -44.08 | [-44.53, -43.62] | no |

Every row (secondary curves included):

| point | curve | method | weights | fraction | gap | top-1 | msel | flip | flip fp32 | GFLOPs | check |
|---|---|---|---|---|---|---|---|---|---|---|---|
| original | original | original | 301,989,888 | 1.000 | 0.000% | 80.59 | 81.89 | 97.74 | n/a | 164.64 |  |
| mirror | mirror | mirror_exact | 246,026,620 | 0.815 | 0.000% | 30.17 | 31.51 | 95.91 | 100.00 | 135.43 |  |
| f0.85 | lowrank | svd | 256,692,224 | 0.850 | 0.000% | 67.51 | 69.87 | n/a | n/a | 141.00 | True |
| f0.85 | lowrank | svdllm | 256,692,224 | 0.850 | 0.000% | 80.61 | 81.77 | n/a | n/a | 141.00 | True |
| f0.85 | mirror_lowrank | mirror_svd | 246,026,620 | 0.815 | -4.155% | 30.17 | 31.51 | 95.91 | 100.00 | 135.43 |  |
| f0.85 | mirror_lowrank | mirror_svdllm | 246,026,620 | 0.815 | -4.155% | 30.17 | 31.51 | 95.91 | 100.00 | 135.43 |  |
| f0.70 | lowrank | svd | 211,394,560 | 0.700 | 0.001% | 22.37 | 24.54 | n/a | n/a | 117.35 | True |
| f0.70 | lowrank | svdllm | 211,394,560 | 0.700 | 0.001% | 79.68 | 81.14 | n/a | n/a | 117.35 | True |
| f0.70 | mirror_lowrank | mirror_svd | 211,396,432 | 0.700 | 0.002% | 12.03 | 13.47 | 94.05 | 100.00 | 117.35 | True |
| f0.70 | mirror_lowrank | mirror_svdllm | 211,396,432 | 0.700 | 0.002% | 21.79 | 23.14 | 94.43 | 100.00 | 117.35 | True |
| f0.55 | lowrank | svd | 166,094,848 | 0.550 | 0.000% | 10.29 | 11.10 | n/a | n/a | 93.71 | True |
| f0.55 | lowrank | svdllm | 166,094,848 | 0.550 | 0.000% | 74.82 | 76.56 | n/a | n/a | 93.71 | True |
| f0.55 | mirror_lowrank | mirror_svd | 166,095,336 | 0.550 | 0.001% | 9.79 | 11.12 | 94.32 | 100.00 | 93.71 | True |
| f0.55 | mirror_lowrank | mirror_svdllm | 166,095,336 | 0.550 | 0.001% | 19.50 | 20.97 | 93.05 | 100.00 | 93.71 | True |
| f0.40 | lowrank | svd | 120,798,208 | 0.400 | 0.002% | 6.80 | 7.34 | n/a | n/a | 70.06 | True |
| f0.40 | lowrank | svdllm | 120,798,208 | 0.400 | 0.002% | 53.77 | 55.96 | n/a | n/a | 70.06 | True |
| f0.40 | mirror_lowrank | mirror_svd | 120,797,312 | 0.400 | 0.001% | 9.69 | 10.69 | 94.93 | 99.98 | 70.06 | True |
| f0.40 | mirror_lowrank | mirror_svdllm | 120,797,312 | 0.400 | 0.001% | 13.34 | 14.67 | 91.57 | 100.00 | 70.06 | True |

## Notes

- Mirror plus low rank keeps Mirror-exact's exact flip invariance (fp32 flip column); a point whose budget is at or above Mirror-exact's own count is Mirror-exact itself, with fewer weights than the budget.
- Low rank alone spreads each budget over the matrices in proportion to Mirror-exact's per-matrix counts (mirror/budget.py, as MUST-07); Mirror plus low rank spreads it over the even and odd blocks in proportion to their Mirror-exact counts (mirror/stacking.py allocate_stack).
- The bootstrap intervals are reported beside each point; P3.1 itself compares the point estimates.

Files: rows_<model>.csv, curves.csv, p31.json, bootstrap.csv, preds/, info/, meta.json.
