# Tier 2: ImageNet linear probe

Protocol (PAPER_EXPERIMENTS.md, Section 1, Tier 2): multinomial logistic regression on [CLS after the final norm, mean patch token after the final norm] (pooled embedding instead of CLS for models without one), features from bf16 forward passes, stored in float16. Trained on the 100k bank (100 ImageNet train images per class), tested on all 50k validation images. Weight decay chosen on a fixed class-stratified 10% split of the bank (never on validation), then refitted on the whole bank. Full-batch L-BFGS on standardised features.

The k-NN check column is k-NN top-1 (k = 20, T = 0.07) on the CLS (or pooled) half of the same cached features: it should sit close to the Tier-1 k-NN number of the same variant.

## dinov2_vitb14_reg

| Variant | What it is | Top-1 | Top-5 | Change vs original (top-1) | k-NN check | Chosen wd | Model params | Notes |
|---|---|---|---|---|---|---|---|---|
| original | original | 81.00 | 95.97 | - | 78.95 | 0.00316 | 86,583,552 | - |
| best_svdllm_s0 | healed model from a run directory (mirror.heal_structured.load_healed_model) | 79.70 | 95.28 | -1.30 | 78.05 | 0.001 | 65,349,120 | - |
| best_svdllm_s1 | healed model from a run directory (mirror.heal_structured.load_healed_model) | 79.77 | 95.31 | -1.23 | 78.07 | 0.001 | 65,349,120 | - |
| best_svdllm_s2 | healed model from a run directory (mirror.heal_structured.load_healed_model) | 79.81 | 95.52 | -1.19 | 78.08 | 0.00316 | 65,349,120 | - |
| vexact_s0 | healed model from a run directory (mirror.heal_structured.load_healed_model) | 80.12 | 95.55 | -0.88 | 78.35 | 0.00316 | 85,691,904 | - |
| vexact_s1 | healed model from a run directory (mirror.heal_structured.load_healed_model) | 80.05 | 95.57 | -0.95 | 78.33 | 0.00316 | 85,691,904 | - |
| vexact_s2 | healed model from a run directory (mirror.heal_structured.load_healed_model) | 80.17 | 95.55 | -0.84 | 78.31 | 0.00316 | 85,691,904 | - |

Source: `t2_linear.csv` and `probe/<model>/<variant>.json` in this folder. Commit(s): 7d05b6439c (dirty); seed(s): 0. Feature extraction and probe timings are in the CSV.
