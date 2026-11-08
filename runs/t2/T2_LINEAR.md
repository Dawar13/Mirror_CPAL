# Tier 2: ImageNet linear probe

Protocol (PAPER_EXPERIMENTS.md, Section 1, Tier 2): multinomial logistic regression on [CLS after the final norm, mean patch token after the final norm] (pooled embedding instead of CLS for models without one), features from bf16 forward passes, stored in float16. Trained on the 100k bank (100 ImageNet train images per class), tested on all 50k validation images. Weight decay chosen on a fixed class-stratified 10% split of the bank (never on validation), then refitted on the whole bank. Full-batch L-BFGS on standardised features.

The k-NN check column is k-NN top-1 (k = 20, T = 0.07) on the CLS (or pooled) half of the same cached features: it should sit close to the Tier-1 k-NN number of the same variant.

## dinov2_vitb14

| Variant | What it is | Top-1 | Top-5 | Change vs original (top-1) | k-NN check | Chosen wd | Model params | Notes |
|---|---|---|---|---|---|---|---|---|
| original | original | 80.68 | 95.79 | - | 78.70 | 0.00316 | 86,580,480 | - |
| mirror_mlp_k6 | Mirror Surgery, MLP only, V-local, k = 6 | 76.80 | 93.98 | -3.88 | 74.68 | 0.00316 | 75,120,970 | - |
| random_k6 | matched random structure, MLP only, k = 6 | 9.77 | 22.26 | -70.91 | 3.53 | 0.001 | 75,120,970 | - |
| svd_mlp_k6 | SVD at matched parameters, MLP only, k = 6 | 69.48 | 89.37 | -11.20 | 66.23 | 0.00316 | 75,099,648 | - |

## dinov2_vitb14_reg

| Variant | What it is | Top-1 | Top-5 | Change vs original (top-1) | k-NN check | Chosen wd | Model params | Notes |
|---|---|---|---|---|---|---|---|---|
| original | original | 80.99 | 95.97 | - | 78.94 | 0.00316 | 86,583,552 | - |
| mirror_mlp_k6 | Mirror Surgery, MLP only, V-local, k = 6 | 79.08 | 95.07 | -1.91 | 77.11 | 0.00316 | 74,734,630 | - |
| random_k6 | matched random structure, MLP only, k = 6 | 11.25 | 24.62 | -69.74 | 3.64 | 0.000316 | 74,734,630 | - |
| svd_mlp_k6 | SVD at matched parameters, MLP only, k = 6 | 20.92 | 40.62 | -60.07 | 8.77 | 0.001 | 74,718,720 | - |
| vexact_k6 | Mirror V-exact, MLP + attention, k = 6 | 79.51 | 95.24 | -1.48 | 77.37 | 0.00316 | 69,194,400 | - |
| healed_mirror_k9 | healed Mirror V-local, MLP + attention, k = 9 (M3) | 80.71 | 95.82 | -0.28 | 78.81 | 0.00316 | 64,979,688 | - |
| healed_random_k9 | healed random structure, MLP + attention, k = 9 (M3) | 74.99 | 92.79 | -5.99 | 73.28 | 0.00316 | 64,979,688 | - |
| healed_svd_k9 | healed SVD, MLP + attention, k = 9 (M3) | 79.30 | 95.26 | -1.69 | 77.57 | 0.00316 | 64,932,864 | - |
| vlocal_k9 | Mirror V-local, MLP + attention, k = 9 | 76.77 | 93.73 | -4.21 | 74.70 | 0.00316 | 64,979,688 | - |

## mae_b16

| Variant | What it is | Top-1 | Top-5 | Change vs original (top-1) | k-NN check | Chosen wd | Model params | Notes |
|---|---|---|---|---|---|---|---|---|
| original | original | 56.54 | 79.83 | - | 13.72 | 0.001 | 85,798,656 | - |

Source: `t2_linear.csv` and `probe/<model>/<variant>.json` in this folder. Commit(s): 8a7b6c4e42 (dirty), edc3bbc98b (dirty); seed(s): 0. Feature extraction and probe timings are in the CSV.
