# E1 report: Mirror Surgery go/kill on dinov2_vitb14

**Outcome: GO.** Every check is in its GO cell.

## Run

- Date: 2026-11-08 10:29:56; git commit `e8bf18d879ef505c5ba8d5c7fb89c18bf84886fd` (uncommitted changes present); seed 0.
- Config: `configs/x2_perlayer_plain.yaml`; device NVIDIA A100-SXM4-80GB; torch 2.11.0+cu128.
- Operator under test: one operator per layer (block b: fc1 against the operator of its input layer, fc2 against its output layer; nulls and SVD ranks matched per layer). Data split: cached order of appearance.
- Data: fit 2000, held-out 1000, k-NN bank 100000, queries 50000 images (each with its mirror); protocol v2: operators fitted on ImageNet train (fit seed 0), held-out errors and k-NN queries on validation, k-NN bank from train, k-NN features in bf16.
- Total time 119.3 min. Stages (s): load 12, sanity 3, fit_stats 59, fit_operators 1381, heldout_errors 43, position_check 2, mlp_matching 1023, breaking_share 3, surgery_knn 4616.

## Decision table (blocks 1 to 6)

| Row | Check | Value | GO if | KILL if | Cell |
|---|---|---|---|---|---|
| 1 | e(null) / e(A_l (per layer)), mean over layers; worst layer 6: 4.54 | 10.48 | >= 5 | < 3 | **GO** |
| 2 | e(A_l (per layer)) / e(I), mean over layers; worst layer 6: 0.381 | 0.1608 | <= 0.5 | informative only | **GO** |
| 3 | d- / d (mean of per-layer A_l over layers 1 to 6) | 0.2587 | >= 0.2 | < 0.1: blocks GO, no KILL (PIVOT: little to save) | **GO** |
| 4 | MLP breaking share, max(mean s1, mean s2) [s1 0.194, s2 0.168; matched null s1 0.416, s2 0.393; twin fraction 0.609, d-/d 0.259] | 0.1939 | <= 0.2, and d-/d >= 0.2 or twin fraction >= 0.2 | > 0.3 | **GO** |
| 5 | Zero-shot surgery at k = 6: k-NN drop [original 78.71, surgery 74.68, SVD 66.23; margin over SVD 8.45] | 4.028 | drop <= 10 and margin >= 5 | no better than SVD (margin <= 0) | **GO** |

Per-layer values behind rows 1 and 2 (layer l is the output of block l). Worst layer: row 1 layer 6, row 2 layer 6.

| Layer | e(null) / e(A_l) | e(A_l) / e(I) |
|---|---|---|
| 1 | 46.63 | 0.03425 |
| 2 | 19.45 | 0.08335 |
| 3 | 12.05 | 0.151 |
| 4 | 8.356 | 0.2079 |
| 5 | 6.243 | 0.2679 |
| 6 | 4.541 (worst) | 0.3809 (worst) |

How blocks are combined: rows 1 and 2 use the mean over layers 1 to 6 of the relative held-out error e / ||X~||², then the ratio of those means. Row 4 averages s1 and s2 over blocks 1 to 6 and uses the larger; a low share counts as evidence only if d-/d is at least 0.2 or the mean twin-neuron fraction is at least 0.2. Row 5 uses surgery on the first 6 blocks. Outcome, in order: KILL if any KILL cell in rows 1, 4 or 5; GO if all rows are GO; PIVOT: structure but little to save, if d-/d is below the row 3 KILL limit; PIVOT if rows 1 to 4 are GO and the drop is above the GO limit; otherwise AMBER (the table gives no verdict).

## Residual stream: one mirror operator?

Shared operator fitted on summed C over layers 1 to 12: d = 768, d+ = 760, d- = 8 (d-/d = 0.010). Directions with no signal (assigned +1): 1. Errors are relative, e / ||X~||², on held-out images; the null is a random orthogonal involution with the same d+ (mean of 3 seeds).

| layer | d- (A_l) | tr(A_l A)/d | e(A_l) | e(A) | e(A, scale-normalised fit) | e(I) | e(null) | e(null)/e(A) |
|---|---|---|---|---|---|---|---|---|
| 0 | 375 | 0.027 | 0.004798 | 1.17 | 0.4257 | 1.165 | 1.185 | 1.01 |
| 1 | 278 | 0.276 | 0.03721 | 1.082 | 0.4399 | 1.086 | 1.108 | 1.02 |
| 2 | 267 | 0.306 | 0.08722 | 1.031 | 0.481 | 1.046 | 1.071 | 1.04 |
| 3 | 217 | 0.435 | 0.1215 | 0.7855 | 0.3348 | 0.8052 | 0.8335 | 1.06 |
| 4 | 184 | 0.525 | 0.1634 | 0.6767 | 0.3306 | 0.7863 | 0.8142 | 1.2 |
| 5 | 148 | 0.621 | 0.1917 | 0.6241 | 0.3451 | 0.7157 | 0.745 | 1.19 |
| 6 | 98 | 0.752 | 0.1951 | 0.4438 | 0.3142 | 0.5122 | 0.5447 | 1.23 |
| 7 | 82 | 0.797 | 0.1899 | 0.3774 | 0.3153 | 0.4902 | 0.521 | 1.38 |
| 8 | 56 | 0.867 | 0.2215 | 0.3342 | 0.3766 | 0.4227 | 0.455 | 1.36 |
| 9 | 25 | 0.949 | 1.438 | 1.441 | 1.451 | 1.445 | 1.456 | 1.01 |
| 10 | 11 | 0.985 | 1.347 | 1.349 | 1.378 | 1.355 | 1.368 | 1.01 |
| 11 | 11 | 0.988 | 1.16 | 1.163 | 1.233 | 1.17 | 1.187 | 1.02 |
| 12 | 7 | 0.996 | 0.169 | 0.17 | 0.4784 | 0.1817 | 0.2176 | 1.28 |

![equivariance vs depth](../plots/equivariance_vs_depth.png)

## Position embedding

||mirror(P) - P A|| / ||P|| on the 16 x 16 patch grid (P channel-centred like the tokens; the second column also removes the mean over positions).

| Operator | channel-centred | also grid-centred |
|---|---|---|
| A_shared | 0.8089 | 0.8448 |
| A_0 (layer 0) | 0.2552 | 0.2441 |
| identity | 0.967 | 1.014 |
| null | 0.9809 | 1.026 |

Layer 0 held-out error with the shared operator: e(A) = 1.17 (e(A_0) = 0.004798, e(I) = 1.165).

## MLP twins and breaking share

Neuron involution per block from the mirrored-versus-original fc1 correlations; breaking share of the folded weights against each block's input-layer operator (fc1) and output-layer operator (fc2). Matched null (mean of 3 seeds): a random involution with the same d+ and d- as each layer's operator and, per block, a random pairing with the same numbers of pairs and self-twins as the fitted one. d-/d is that of the block's input-layer operator. If the fitted structure is nearly trivial (few odd features, few pairs), the matched null is nearly trivial too and its share is also low; a low share then is not evidence. Twin fraction = neurons in pairs / all neurons. Evidence = d-/d at least 0.2 or twin fraction at least 0.2.

| block | pairs | self-twins | twin fraction | d-/d | mean corr (pairs) | mean corr (self) | s1 | s1 matched null | s2 | s2 matched null | evidence | implied saving | SVD rank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1134 | 804 | 0.738 | 0.488 | 0.776 | 0.830 | 0.282 | 0.497 | 0.191 | 0.463 | yes | 0.480 | 309 |
| 2 | 1115 | 842 | 0.726 | 0.362 | 0.756 | 0.841 | 0.196 | 0.463 | 0.193 | 0.458 | yes | 0.460 | 330 |
| 3 | 1172 | 728 | 0.763 | 0.348 | 0.651 | 0.781 | 0.205 | 0.464 | 0.208 | 0.448 | yes | 0.456 | 329 |
| 4 | 930 | 1212 | 0.605 | 0.283 | 0.658 | 0.814 | 0.203 | 0.414 | 0.183 | 0.397 | yes | 0.406 | 359 |
| 5 | 778 | 1516 | 0.507 | 0.240 | 0.622 | 0.848 | 0.167 | 0.371 | 0.150 | 0.348 | yes | 0.360 | 386 |
| 6 | 479 | 2114 | 0.312 | 0.193 | 0.653 | 0.898 | 0.111 | 0.288 | 0.085 | 0.246 | yes | 0.266 | 437 |
| 7 | 290 | 2492 | 0.189 | 0.128 | 0.691 | 0.898 | 0.068 | 0.197 | 0.051 | 0.179 | no | 0.189 | 492 |
| 8 | 372 | 2328 | 0.242 | 0.107 | 0.734 | 0.921 | 0.058 | 0.201 | 0.042 | 0.176 | yes | 0.189 | 490 |
| 9 | 171 | 2730 | 0.111 | 0.073 | 0.778 | 0.922 | 0.036 | 0.121 | 0.017 | 0.084 | no | 0.103 | 540 |
| 10 | 85 | 2902 | 0.055 | 0.033 | 0.811 | 0.921 | 0.012 | 0.058 | 0.006 | 0.040 | no | 0.050 | 578 |
| 11 | 21 | 3030 | 0.014 | 0.014 | 0.755 | 0.916 | 0.005 | 0.021 | 0.003 | 0.021 | no | 0.021 | 601 |
| 12 | 15 | 3042 | 0.010 | 0.014 | 0.775 | 0.949 | 0.012 | 0.019 | 0.004 | 0.014 | no | 0.016 | 602 |

![breaking share vs depth](../plots/breaking_share_vs_depth.png)

## Zero-shot surgery (k-NN, k = 20, T = 0.07)

Surgery replaces the MLPs of the first k blocks by their nearest mirror-equivariant weights. SVD keeps the same number of parameters as a rank-r factorisation. Random uses the null operator and pairing. Top-1 and flip consistency are in percent.

| k | variant | top-1 | drop (points) | flip consistency | MLP params kept (first k blocks) |
|---|---|---|---|---|---|
| 0 | original | 78.71 | 0.00 | 96.36 | 1.000 |
| 3 | surgery | 77.36 | 1.35 | 93.38 | 0.534 |
| 3 | svd | 76.63 | 2.07 | 93.14 | 0.534 |
| 3 | random | 6.35 | 72.35 | 14.91 | 0.534 |
| 6 | surgery | 74.68 | 4.03 | 90.15 | 0.595 |
| 6 | svd | 66.23 | 12.48 | 82.69 | 0.594 |
| 6 | random | 3.66 | 75.04 | 14.19 | 0.595 |
| 9 | surgery | 73.26 | 5.45 | 87.45 | 0.677 |
| 9 | svd | 36.63 | 42.08 | 49.83 | 0.676 |
| 9 | random | 3.04 | 75.67 | 10.54 | 0.677 |

## Sanity checks in this run

- Folding (block 0, one fit batch): max abs difference 9.54e-06.
- Identity surgery on all blocks (A = I, P = I), CLS features of one batch: max abs difference 2.5e-05.

All numbers come from `scripts/run_e1.py` with the config, seed and commit above; per-layer and per-block values are in `results/e1_layers.csv`, `results/e1_mlp.csv` and `results/e1_surgery.csv`.
