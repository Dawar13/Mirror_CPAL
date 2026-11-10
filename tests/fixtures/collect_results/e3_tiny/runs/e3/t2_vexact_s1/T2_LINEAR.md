# Tier 2: ImageNet linear probe

Protocol (PAPER_EXPERIMENTS.md, Section 1, Tier 2): multinomial logistic regression on [CLS after the final norm, mean patch token after the final norm] (pooled embedding instead of CLS for models without one), features from bf16 forward passes, stored in float16. Trained on the 100k bank (100 ImageNet train images per class), tested on all 50k validation images. Weight decay chosen on a fixed class-stratified 10% split of the bank (never on validation), then refitted on the whole bank. Full-batch L-BFGS on standardised features.

The k-NN check column is k-NN top-1 (k = 20, T = 0.07) on the CLS (or pooled) half of the same cached features: it should sit close to the Tier-1 k-NN number of the same variant.

Healed V-exact rows (healed_vexact_k<K>, scripts/heal_vexact.py): the same feature through the student's own final norm, which at full conversion holds the even head, so the feature of an image and of its mirror are equal in fp32 (bf16 forward passes round the mirrored computation differently). The loaded student is checked before probing (span exactness and image vs mirror feature, fp32); values and the feature statement are in the CSV columns vexact_* and notes.

## tiny

| Variant | What it is | Top-1 | Top-5 | Change vs original (top-1) | k-NN check | Chosen wd | Model params | Notes |
|---|---|---|---|---|---|---|---|---|
| healed_vexact_k4 | healed Mirror V-exact, MLP + attention, k = 4 (scripts/heal_vexact.py) | 78.12 | 100.00 | - | 68.75 | 0.01 | 209,000 | wd at grid edge; loaded student checked: span exactness 1.2e-06, image vs mirror feature 3.2e-07, PASS; even head, feature mirror-invariant |

Source: `t2_linear.csv` and `probe/<model>/<variant>.json` in this folder. Commit(s): aeb57bed18; seed(s): 0. Feature extraction and probe timings are in the CSV.
