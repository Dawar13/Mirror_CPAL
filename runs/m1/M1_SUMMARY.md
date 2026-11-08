# M1: attention conversion, zero-shot

Model `dinov2_vitb14_reg` (12 blocks, 12 heads of dimension 64, 4 registers). Operators and neuron pairings: `/home/dawar/Mirror_CPAL/runs/x2/perlayer_reg_s0/results/e1_operators.npz` (sha256 9ca69899622e). Attention statistics: fit seed 0 (2000 images, with their mirrors). Random-structure seed 1. Git `ced2c08928` (uncommitted changes). Written 2026-11-08 11:53:06.

Protocol: v2. Statistics and operators on train images; exactness probe, flip check and k-NN queries on validation; k-NN bank of train images. k-NN features in bf16 for every variant; exactness and flip check in fp32 with TF32 off.

## Head structure per block (V-local)

Pairing score: cosine of the best orthogonal map between head h on the original and head sigma(h) on the mirror (q and k together; 1 = exact). Head-map error: relative error of that map on q, k and on v, with the identity map (no head swap, B = C = I) in brackets. Breaking share s_all: share of the attention matrix energy (Wq, Wk, Wv, Wo) removed by the projection; random: the matched random structure. Kept: free matrix parameters of the equivariant attention over dense.

| block | self-heads | twin pairs | pairing score | error q,k (identity) | error v (identity) | s_all | random | MLP s1 / s2 | kept |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 12 | 0 | 0.981 | 0.039 (0.053) | 0.332 (0.460) | 0.283 | 0.494 | 0.267 / 0.196 | 0.515 |
| 2 | 10 | 1 | 0.944 | 0.113 (0.408) | 0.310 (1.167) | 0.113 | 0.446 | 0.255 / 0.201 | 0.548 |
| 3 | 10 | 1 | 0.931 | 0.139 (0.424) | 0.381 (1.138) | 0.154 | 0.445 | 0.226 / 0.186 | 0.576 |
| 4 | 12 | 0 | 0.922 | 0.156 (0.470) | 0.362 (0.769) | 0.134 | 0.368 | 0.231 / 0.192 | 0.642 |
| 5 | 10 | 1 | 0.928 | 0.143 (0.461) | 0.351 (0.843) | 0.091 | 0.350 | 0.204 / 0.170 | 0.647 |
| 6 | 10 | 1 | 0.926 | 0.148 (0.491) | 0.327 (0.884) | 0.087 | 0.323 | 0.155 / 0.117 | 0.681 |
| 7 | 10 | 1 | 0.923 | 0.154 (0.401) | 0.380 (0.743) | 0.083 | 0.276 | 0.103 / 0.078 | 0.719 |
| 8 | 10 | 1 | 0.951 | 0.098 (0.443) | 0.240 (0.679) | 0.038 | 0.237 | 0.066 / 0.042 | 0.772 |
| 9 | 12 | 0 | 0.937 | 0.125 (0.338) | 0.220 (0.410) | 0.065 | 0.109 | 0.041 / 0.015 | 0.895 |
| 10 | 10 | 1 | 0.960 | 0.080 (0.321) | 0.197 (0.407) | 0.045 | 0.127 | 0.023 / 0.006 | 0.862 |
| 11 | 12 | 0 | 0.960 | 0.081 (0.201) | 0.167 (0.182) | 0.005 | 0.023 | 0.006 / 0.002 | 0.970 |
| 12 | 12 | 0 | 0.962 | 0.076 (0.136) | 0.187 (0.249) | 0.005 | 0.020 | 0.016 / 0.002 | 0.973 |

A_out choice for Wo (the mid-block stream has no fitted operator): block-average s_all is 0.0931 with A_out = A_l and 0.0918 with A_out = A_l+1; kept **A_l+1**. The MLP keeps the E1b convention (fc1 with A_l, fc2 with A_l+1).

## V-exact: one operator per span

A* is fitted from the cross-moments of stream layers 0 to k, each divided by its energy tr(G_l). Agreement: tr(A* A_l)/d averaged over the span. Alignment: tr(A* C_l)/tr(A_l C_l) (fit data, 1 = as good as the per-layer operator). Shares: means over the span's blocks on the folded model. Exactness: largest relative error over the span's layers (fp32).

| k | d+ | d- | agreement | alignment (min) | attention s_all | MLP s1 / s2 | registers | max rel. error | pass |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 552 | 216 | 0.608 | 0.955 (0.896) | 0.161 | 0.258 / 0.232 | 0 1 2 3 (0 pairs) | 5.1e-06 | yes |
| 9 | 636 | 132 | 0.597 | 0.928 (0.886) | 0.134 | 0.228 / 0.198 | 0 1 2 3 (0 pairs) | 5.6e-06 | yes |
| 12 | 700 | 68 | 0.634 | 0.914 (0.877) | 0.117 | 0.198 / 0.158 | 0 1 2 3 (0 pairs) | 7.6e-06 | yes |

Embedding breaking shares (part of each embedding removed by the V-exact projection):

- k = 6: patch_embed.weight 0.076, patch_embed.bias 0.000, pos_embed.patches 0.035, pos_embed.cls 0.104, cls_token 0.104, registers 0.089
- k = 9: patch_embed.weight 0.195, patch_embed.bias 0.000, pos_embed.patches 0.045, pos_embed.cls 0.075, cls_token 0.075, registers 0.059
- k = 12: patch_embed.weight 0.290, patch_embed.bias 0.000, pos_embed.patches 0.062, pos_embed.cls 0.037, cls_token 0.037, registers 0.035

## Full conversion (k = 12, fp32)

With the final features projected onto the even subspace of A*: flip consistency 100.00% on 5000 validation queries (bank 10000 train images), **PASS** (must be 100%); largest relative CLS feature difference between an image and its mirror 9.8e-06. Without the even projection: flip consistency 96.22% (the features of the mirror are A* times the original's, not equal). Top-1 on this small subset protocol: 42.68% (not comparable with the main table).

## Zero-shot k-NN

Kept: matrix parameters of the converted blocks (MLP fc1, fc2 and attention Wq, Wk, Wv, Wo) over dense; SVD and the random structure are matched to Mirror V-local. Model: whole-model parameters after conversion (biases counted as dense). Flip consistency here is in bf16 over all queries.

| k | variant | top-1 | drop | flip consistency | kept (converted blocks) | model params |
|---|---|---|---|---|---|---|
| 0 | original | 78.94 | 0.00 | 96.28 | 1.000 | 86,583,552 (1.000) |
| 0 | folded original (sanity) | 78.98 | -0.04 | 96.26 | 1.000 | 86,583,552 (1.000) |
| 6 | Mirror V-local, MLP + attention | 76.02 | 2.91 | 91.32 | 0.588 | 69,095,592 (0.798) |
| 6 | Mirror V-exact, MLP + attention | 77.37 | 1.57 | 95.04 | 0.591 | 69,194,400 (0.799) |
| 6 | Mirror V-local, MLP only (A6) | 77.11 | 1.83 | 92.38 | 0.721 | 74,734,630 (0.863) |
| 6 | SVD at matched parameters, MLP + attention | 5.27 | 73.67 | 24.33 | 0.587 | 69,061,632 (0.798) |
| 6 | matched random structure, MLP + attention | 4.19 | 74.75 | 18.79 | 0.588 | 69,095,592 (0.798) |
| 9 | Mirror V-local, MLP + attention | 74.70 | 4.24 | 89.44 | 0.661 | 64,979,688 (0.750) |
| 9 | Mirror V-exact, MLP + attention | 73.34 | 5.59 | 93.99 | 0.673 | 65,735,376 (0.759) |
| 9 | Mirror V-local, MLP only (A6) | 76.37 | 2.57 | 90.73 | 0.772 | 72,065,992 (0.832) |
| 9 | SVD at matched parameters, MLP + attention | 6.49 | 72.44 | 37.45 | 0.660 | 64,932,864 (0.750) |
| 9 | matched random structure, MLP + attention | 3.62 | 75.32 | 17.75 | 0.661 | 64,979,688 (0.750) |
| 12 | Mirror V-local, MLP + attention | 74.84 | 4.09 | 89.38 | 0.735 | 64,090,370 (0.740) |
| 12 | Mirror V-exact, MLP + attention | 53.85 | 25.09 | 97.92 | 0.760 | 66,203,360 (0.765) |
| 12 | Mirror V-local, MLP only (A6) | 76.28 | 2.65 | 90.32 | 0.824 | 71,636,896 (0.827) |
| 12 | SVD at matched parameters, MLP + attention | 6.27 | 72.66 | 35.59 | 0.734 | 64,024,320 (0.739) |
| 12 | matched random structure, MLP + attention | 3.64 | 75.29 | 16.67 | 0.735 | 64,090,370 (0.740) |

## Notes

- One fit seed; headline numbers need 3 seeds (X3).
- V-exact uses the fully folded model (every norm an RMSNorm); for DINOv2 the position embedding interpolated to the 16 x 16 grid is written into the model first, so the converted model is tied to 224 px inputs. The folded original row checks that folding alone does not change k-NN.
- Registers: paired by the involution matching on register alignment scores averaged over the span's layers (layer 0 is the register parameters themselves); unpaired registers are projected to the even subspace.
- In bf16 the mirrored computation rounds differently, so the fully converted model's bf16 flip consistency can fall slightly below 100%; the exact check is the fp32 one above.
- SVD keeps biases dense and is matched on matrix parameters, like the E1b MLP comparison.

Files: m1_heads.csv, m1_heads_detail.csv, m1_ranks.csv, m1_vexact*.csv, m1_exactness.csv, m1_flipcheck.json, m1_knn.csv, m1_structure.npz, m1_meta.json (seed 0, git hash, config).
