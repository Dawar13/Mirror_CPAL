# Tier 2: ImageNet linear probe

Protocol (PAPER_EXPERIMENTS.md, Section 1, Tier 2): multinomial logistic regression on [CLS after the final norm, mean patch token after the final norm] (pooled embedding instead of CLS for models without one), features from bf16 forward passes, stored in float16. Trained on the 100k bank (100 ImageNet train images per class), tested on all 50k validation images. Weight decay chosen on a fixed class-stratified 10% split of the bank (never on validation), then refitted on the whole bank. Full-batch L-BFGS on standardised features.

The k-NN check column is k-NN top-1 (k = 20, T = 0.07) on the CLS (or pooled) half of the same cached features: it should sit close to the Tier-1 k-NN number of the same variant.

## tiny

| Variant | What it is | Top-1 | Top-5 | Change vs original (top-1) | k-NN check | Chosen wd | Model params | Notes |
|---|---|---|---|---|---|---|---|---|
| healed_svd_k4 | healed SVD, MLP + attention, k = 4 (M3) | 65.62 | 100.00 | - | 56.25 | 0.01 | 206,632 | wd at grid edge |

Source: `t2_linear.csv` and `probe/<model>/<variant>.json` in this folder. Commit(s): aeb57bed18; seed(s): 0. Feature extraction and probe timings are in the CSV.
