# Flip-averaging baseline: exact flip invariance at test time

Question: exact mirror invariance can also be had by averaging the features of an image and of its mirror at test time (two forward passes per image). This table measures that baseline with the protocol of every other number, next to the healed V-exact model (exactly mirror-invariant in one forward pass).

Protocol. k-NN: bank of 100 ImageNet train images per class (100k), all 50k validation images as queries, CLS token after the final norm, cosine, k = 20, T = 0.07, bf16 forward passes unless the mode says fp32 (mirror.knn, as every Tier-1 number). Flip consistency: share of validation images whose k-NN prediction equals the prediction for their mirror image, measured. Identical features: share of validation images whose unit k-NN feature equals that of their mirror bit for bit. Linear probe: the Tier-2 recipe of scripts/tier2_linear.py ([CLS, mean patch token] after the final norm; logistic regression trained on the 100k bank, weight decay chosen on a 10% split of the bank, tested on the 50k validation images).

Flip averaging: f_avg(x) = (f(x) + f(mirror x)) / 2, for the bank and the queries alike (and for the probe's training and test images alike). The k-NN feature is averaged after the final norm and before the L2 normalisation, where the k-NN normalises every other feature; this is also the feature the healed V-exact student at full conversion was distilled to. The probe feature is averaged as a whole. The two forward passes use the same batches as the plain mode, so the average is the same for an image and its mirror bit for bit, in any precision.

Cost: forward passes per image (1 plain, 2 flip-averaged). GFLOPs per image: forward passes times the analytic cost of one forward pass (scripts/collect_results.analytic_cost, the cost model of the paper tables; analytic counts, not measured latency). Parameters: whole model, with the share of the original.

## dinov2_vitb14_reg

| Variant | What it is | Mode | k-NN top-1 | Change vs original (plain) | Flip consistency (%) | Identical features (%) | Linear probe top-1 | Forward passes per image | GFLOPs per image | Parameters (share of original) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| original | original | plain | 78.93 | - | 96.28 | 0.00 | 81.00 | 1 | 47.08 | 86,583,552 (1.000) | - |

Source: `flipavg.csv` and `rows/<model>/<variant>__<mode>.json` in this folder (predictions in `knn/`). Commit(s): c121ed09bb; seed(s): 0. Full notes, feature checks and timings are in the CSV.
