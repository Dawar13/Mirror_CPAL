# Tier 2: ImageNet linear probe

Protocol (PAPER_EXPERIMENTS.md, Section 1, Tier 2): multinomial logistic regression on [CLS after the final norm, mean patch token after the final norm] (pooled embedding instead of CLS for models without one), features from bf16 forward passes, stored in float16. Trained on the 100k bank (100 ImageNet train images per class), tested on all 50k validation images. Weight decay chosen on a fixed class-stratified 10% split of the bank (never on validation), then refitted on the whole bank. Full-batch L-BFGS on standardised features.

The k-NN check column is k-NN top-1 (k = 20, T = 0.07) on the CLS (or pooled) half of the same cached features: it should sit close to the Tier-1 k-NN number of the same variant.

Mirror-exact feature views (--full-patch-view): view a, the variant's own row, is [CLS after the final norm, mean of the even-projected patch tokens], exactly flip-invariant; view b, rows <variant>_fullpatch, is [the same CLS, mean of the full patch tokens with the even projection bypassed] (PROJECT_NEXT_PHASE.md Section 4.5: even part for invariant use, full tokens for equivariant use). The odd part of the full patch mean changes sign under a flip, so view b is not invariant. Both come from one feature pass (columns feature_view, patch_mean_tokens, feature_flip_invariant).

## dinov2_vitb14_reg

| Variant | What it is | Top-1 | Top-5 | Change vs original (top-1) | k-NN check | Chosen wd | Model params | Notes |
|---|---|---|---|---|---|---|---|---|
| original | original | 81.00 | 95.97 | - | 78.95 | 0.00316 | 86,583,552 | - |
| best_svdllm_s0 | healed model from a run directory (mirror.heal_structured.load_healed_model) | 79.72 | 95.29 | -1.29 | 78.05 | 0.001 | 65,349,120 | - |
| vexact_s0 | healed model from a run directory (mirror.heal_structured.load_healed_model) | 80.10 | 95.56 | -0.90 | 78.31 | 0.00316 | 85,691,904 | - |
| vexact_s0_fullpatch | healed model from a run directory (mirror.heal_structured.load_healed_model) [view b: even CLS + mean of full patch tokens] | 80.09 | 95.57 | -0.92 | 78.31 | 0.00316 | 85,691,904 | - |
| vexact_s1 | healed model from a run directory (mirror.heal_structured.load_healed_model) | 80.04 | 95.58 | -0.96 | 78.35 | 0.00316 | 85,691,904 | - |
| vexact_s1_fullpatch | healed model from a run directory (mirror.heal_structured.load_healed_model) [view b: even CLS + mean of full patch tokens] | 80.06 | 95.56 | -0.94 | 78.35 | 0.00316 | 85,691,904 | - |
| vexact_s2 | healed model from a run directory (mirror.heal_structured.load_healed_model) | 80.19 | 95.56 | -0.82 | 78.35 | 0.00316 | 85,691,904 | - |
| vexact_s2_fullpatch | healed model from a run directory (mirror.heal_structured.load_healed_model) [view b: even CLS + mean of full patch tokens] | 80.13 | 95.57 | -0.87 | 78.35 | 0.00316 | 85,691,904 | - |

Source: `t2_linear.csv` and `probe/<model>/<variant>.json` in this folder. Commit(s): 805c231c9c (dirty); seed(s): 0. Feature extraction and probe timings are in the CSV.
