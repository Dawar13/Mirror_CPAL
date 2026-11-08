# E1 report: Mirror Surgery go/kill on dinov2_vitb14_reg

**Outcome: AMBER.** No rule fires: row(s) 4 not GO and no KILL cell in rows 1, 4, 5; needs a human decision.

## Run

- Date: 2026-11-08 10:27:47; git commit `e8bf18d879ef505c5ba8d5c7fb89c18bf84886fd` (uncommitted changes present); seed 0.
- Config: `configs/x2_perlayer_reg.yaml`; device NVIDIA A100-SXM4-80GB; torch 2.11.0+cu128.
- Operator under test: one operator per layer (block b: fc1 against the operator of its input layer, fc2 against its output layer; nulls and SVD ranks matched per layer). Data split: cached order of appearance.
- Data: fit 2000, held-out 1000, k-NN bank 100000, queries 50000 images (each with its mirror); protocol v2: operators fitted on ImageNet train (fit seed 0), held-out errors and k-NN queries on validation, k-NN bank from train, k-NN features in bf16.
- Total time 119.0 min. Stages (s): load 4, sanity 1, fit_stats 82, fit_operators 1163, heldout_errors 45, position_check 2, mlp_matching 1288, breaking_share 0, surgery_knn 4535.

## Decision table (blocks 1 to 6)

| Row | Check | Value | GO if | KILL if | Cell |
|---|---|---|---|---|---|
| 1 | e(null) / e(A_l (per layer)), mean over layers; worst layer 6: 5.73 | 13.06 | >= 5 | < 3 | **GO** |
| 2 | e(A_l (per layer)) / e(I), mean over layers; worst layer 6: 0.362 | 0.1698 | <= 0.5 | informative only | **GO** |
| 3 | d- / d (mean of per-layer A_l over layers 1 to 6) | 0.2745 | >= 0.2 | < 0.1: blocks GO, no KILL (PIVOT: little to save) | **GO** |
| 4 | MLP breaking share, max(mean s1, mean s2) [s1 0.223, s2 0.177; matched null s1 0.427, s2 0.409; twin fraction 0.630, d-/d 0.275] | 0.2231 | <= 0.2, and d-/d >= 0.2 or twin fraction >= 0.2 | > 0.3 | **AMBER** |
| 5 | Zero-shot surgery at k = 6: k-NN drop [original 78.94, surgery 77.09, SVD 8.77; margin over SVD 68.33] | 1.842 | drop <= 10 and margin >= 5 | no better than SVD (margin <= 0) | **GO** |

Per-layer values behind rows 1 and 2 (layer l is the output of block l). Worst layer: row 1 layer 6, row 2 layer 6.

| Layer | e(null) / e(A_l) | e(A_l) / e(I) |
|---|---|---|
| 1 | 50.73 | 0.04064 |
| 2 | 23.37 | 0.09301 |
| 3 | 14.79 | 0.1483 |
| 4 | 11.49 | 0.2271 |
| 5 | 8.219 | 0.2827 |
| 6 | 5.734 (worst) | 0.3616 (worst) |

How blocks are combined: rows 1 and 2 use the mean over layers 1 to 6 of the relative held-out error e / ||X~||², then the ratio of those means. Row 4 averages s1 and s2 over blocks 1 to 6 and uses the larger; a low share counts as evidence only if d-/d is at least 0.2 or the mean twin-neuron fraction is at least 0.2. Row 5 uses surgery on the first 6 blocks. Outcome, in order: KILL if any KILL cell in rows 1, 4 or 5; GO if all rows are GO; PIVOT: structure but little to save, if d-/d is below the row 3 KILL limit; PIVOT if rows 1 to 4 are GO and the drop is above the GO limit; otherwise AMBER (the table gives no verdict).

## Residual stream: one mirror operator?

Shared operator fitted on summed C over layers 1 to 12: d = 768, d+ = 760, d- = 8 (d-/d = 0.010). Directions with no signal (assigned +1): 1. Errors are relative, e / ||X~||², on held-out images; the null is a random orthogonal involution with the same d+ (mean of 3 seeds).

| layer | d- (A_l) | tr(A_l A)/d | e(A_l) | e(A) | e(A, scale-normalised fit) | e(I) | e(null) | e(null)/e(A) |
|---|---|---|---|---|---|---|---|---|
| 0 | 368 | 0.044 | 0.006329 | 1.11 | 0.6665 | 1.101 | 1.121 | 1.01 |
| 1 | 279 | 0.273 | 0.03283 | 0.7914 | 0.3897 | 0.8078 | 0.8308 | 1.05 |
| 2 | 266 | 0.308 | 0.06866 | 0.6937 | 0.3173 | 0.7382 | 0.7615 | 1.1 |
| 3 | 226 | 0.411 | 0.09815 | 0.5987 | 0.2783 | 0.6617 | 0.6872 | 1.15 |
| 4 | 203 | 0.472 | 0.1108 | 0.4405 | 0.2516 | 0.4881 | 0.52 | 1.18 |
| 5 | 163 | 0.578 | 0.1377 | 0.4447 | 0.2819 | 0.487 | 0.52 | 1.17 |
| 6 | 128 | 0.671 | 0.1739 | 0.4173 | 0.3031 | 0.4809 | 0.5115 | 1.23 |
| 7 | 95 | 0.760 | 0.19 | 0.3804 | 0.3107 | 0.4317 | 0.4638 | 1.22 |
| 8 | 45 | 0.891 | 0.1088 | 0.1635 | 0.169 | 0.1901 | 0.2416 | 1.48 |
| 9 | 28 | 0.937 | 0.1273 | 0.1683 | 0.2042 | 0.1935 | 0.2405 | 1.43 |
| 10 | 11 | 0.983 | 0.1404 | 0.154 | 0.24 | 0.1755 | 0.2181 | 1.42 |
| 11 | 10 | 0.986 | 0.1355 | 0.1435 | 0.2408 | 0.1624 | 0.2015 | 1.4 |
| 12 | 8 | 0.997 | 0.07616 | 0.07689 | 0.2216 | 0.08949 | 0.1294 | 1.68 |

![equivariance vs depth](../plots/equivariance_vs_depth.png)

## Position embedding

||mirror(P) - P A|| / ||P|| on the 16 x 16 patch grid (P channel-centred like the tokens; the second column also removes the mean over positions).

| Operator | channel-centred | also grid-centred |
|---|---|---|
| A_shared | 0.8468 | 0.8701 |
| A_0 (layer 0) | 0.2126 | 0.1848 |
| identity | 1.052 | 1.082 |
| null | 1.061 | 1.09 |

Layer 0 held-out error with the shared operator: e(A) = 1.11 (e(A_0) = 0.006329, e(I) = 1.101).

## MLP twins and breaking share

Neuron involution per block from the mirrored-versus-original fc1 correlations; breaking share of the folded weights against each block's input-layer operator (fc1) and output-layer operator (fc2). Matched null (mean of 3 seeds): a random involution with the same d+ and d- as each layer's operator and, per block, a random pairing with the same numbers of pairs and self-twins as the fitted one. d-/d is that of the block's input-layer operator. If the fitted structure is nearly trivial (few odd features, few pairs), the matched null is nearly trivial too and its share is also low; a low share then is not evidence. Twin fraction = neurons in pairs / all neurons. Evidence = d-/d at least 0.2 or twin fraction at least 0.2.

| block | pairs | self-twins | twin fraction | d-/d | mean corr (pairs) | mean corr (self) | s1 | s1 matched null | s2 | s2 matched null | evidence | implied saving | SVD rank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1168 | 736 | 0.760 | 0.479 | 0.804 | 0.818 | 0.267 | 0.494 | 0.196 | 0.467 | yes | 0.481 | 310 |
| 2 | 1116 | 840 | 0.727 | 0.363 | 0.727 | 0.781 | 0.255 | 0.461 | 0.201 | 0.457 | yes | 0.460 | 330 |
| 3 | 1013 | 1046 | 0.660 | 0.346 | 0.684 | 0.806 | 0.226 | 0.448 | 0.186 | 0.430 | yes | 0.439 | 339 |
| 4 | 963 | 1146 | 0.627 | 0.294 | 0.634 | 0.791 | 0.231 | 0.423 | 0.192 | 0.410 | yes | 0.418 | 354 |
| 5 | 878 | 1316 | 0.572 | 0.264 | 0.623 | 0.802 | 0.204 | 0.398 | 0.170 | 0.376 | yes | 0.388 | 369 |
| 6 | 672 | 1728 | 0.438 | 0.212 | 0.662 | 0.858 | 0.155 | 0.338 | 0.117 | 0.312 | yes | 0.325 | 406 |
| 7 | 554 | 1964 | 0.361 | 0.167 | 0.701 | 0.899 | 0.103 | 0.285 | 0.078 | 0.258 | yes | 0.273 | 438 |
| 8 | 406 | 2260 | 0.264 | 0.124 | 0.769 | 0.928 | 0.066 | 0.221 | 0.042 | 0.176 | yes | 0.199 | 477 |
| 9 | 155 | 2762 | 0.101 | 0.059 | 0.790 | 0.929 | 0.041 | 0.101 | 0.015 | 0.083 | no | 0.093 | 551 |
| 10 | 93 | 2886 | 0.061 | 0.036 | 0.823 | 0.937 | 0.023 | 0.063 | 0.006 | 0.042 | no | 0.054 | 574 |
| 11 | 19 | 3034 | 0.012 | 0.014 | 0.834 | 0.927 | 0.006 | 0.020 | 0.002 | 0.019 | no | 0.020 | 601 |
| 12 | 17 | 3038 | 0.011 | 0.013 | 0.841 | 0.958 | 0.016 | 0.018 | 0.002 | 0.016 | no | 0.017 | 603 |

![breaking share vs depth](../plots/breaking_share_vs_depth.png)

## Zero-shot surgery (k-NN, k = 20, T = 0.07)

Surgery replaces the MLPs of the first k blocks by their nearest mirror-equivariant weights. SVD keeps the same number of parameters as a rank-r factorisation. Random uses the null operator and pairing. Top-1 and flip consistency are in percent.

| k | variant | top-1 | drop (points) | flip consistency | MLP params kept (first k blocks) |
|---|---|---|---|---|---|
| 0 | original | 78.94 | 0.00 | 96.28 | 1.000 |
| 3 | surgery | 78.35 | 0.59 | 94.25 | 0.540 |
| 3 | svd | 76.41 | 2.52 | 93.13 | 0.540 |
| 3 | random | 3.80 | 75.14 | 7.72 | 0.540 |
| 6 | surgery | 77.09 | 1.84 | 92.36 | 0.581 |
| 6 | svd | 8.77 | 70.17 | 42.55 | 0.581 |
| 6 | random | 3.98 | 74.95 | 15.73 | 0.581 |
| 9 | surgery | 76.36 | 2.57 | 90.71 | 0.658 |
| 9 | svd | 5.83 | 73.10 | 32.59 | 0.658 |
| 9 | random | 4.72 | 74.22 | 23.37 | 0.658 |

## Sanity checks in this run

- Folding (block 0, one fit batch): max abs difference 1.29e-05.
- Identity surgery on all blocks (A = I, P = I), CLS features of one batch: max abs difference 2.72e-05.

All numbers come from `scripts/run_e1.py` with the config, seed and commit above; per-layer and per-block values are in `results/e1_layers.csv`, `results/e1_mlp.csv` and `results/e1_surgery.csv`.
