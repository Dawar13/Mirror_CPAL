# Flip-averaging baseline: exact flip invariance at test time

Question: exact mirror invariance can also be had by averaging the features of an image and of its mirror at test time (two forward passes per image). This table measures that baseline with the protocol of every other number, next to the healed V-exact model (exactly mirror-invariant in one forward pass).

Protocol. k-NN: bank of 100 ImageNet train images per class (100k), all 50k validation images as queries, CLS token after the final norm, cosine, k = 20, T = 0.07, bf16 forward passes unless the mode says fp32 (mirror.knn, as every Tier-1 number). Flip consistency: share of validation images whose k-NN prediction equals the prediction for their mirror image, measured. Identical features: share of validation images whose unit k-NN feature equals that of their mirror bit for bit. Linear probe: the Tier-2 recipe of scripts/tier2_linear.py ([CLS, mean patch token] after the final norm; logistic regression trained on the 100k bank, weight decay chosen on a 10% split of the bank, tested on the 50k validation images).

Flip averaging: f_avg(x) = (f(x) + f(mirror x)) / 2, for the bank and the queries alike (and for the probe's training and test images alike). The k-NN feature is averaged after the final norm and before the L2 normalisation, where the k-NN normalises every other feature; this is also the feature the healed V-exact student at full conversion was distilled to. The probe feature is averaged as a whole. The two forward passes use the same batches as the plain mode, so the average is the same for an image and its mirror bit for bit, in any precision.

Cost: forward passes per image (1 plain, 2 flip-averaged). GFLOPs per image: forward passes times the analytic cost of one forward pass (scripts/collect_results.analytic_cost, the cost model of the paper tables; analytic counts, not measured latency). Parameters: whole model, with the share of the original.

Tiny rows: CPU smoke test (tiny random ViT, synthetic images); not the protocol above.

## tiny

| Variant | What it is | Mode | k-NN top-1 | Change vs original (plain) | Flip consistency (%) | Identical features (%) | Linear probe top-1 | Forward passes per image | GFLOPs per image | Parameters (share of original) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| original | original | plain | 68.75 | - | 81.25 | 0.00 | 71.88 | 1 | 0.01 | 304,424 (1.000) | probe wd at grid edge |
| original | original | flipavg | 65.62 | -3.12 | 100.00 | 100.00 | 71.88 | 2 | 0.02 | 304,424 (1.000) | probe wd at grid edge |
| healed_vexact_k4 | healed Mirror V-exact, MLP + attention, k = 4 (scripts/heal_vexact.py) | plain | 78.12 | 9.38 | 100.00 | 0.00 | 75.00 | 1 | 0.00 | 208,776 (0.686) | checked after loading (fp32): span exactness 9.3e-07, image vs mirror probe feature 5.4e-07, PASS; healing run's fp32 flip check 100.00%; probe wd at grid edge |
| healed_vexact_k4 | healed Mirror V-exact, MLP + attention, k = 4 (scripts/heal_vexact.py) | plain_fp32 | 78.12 | 9.38 | 100.00 | 0.00 | - | 1 | 0.00 | 208,776 (0.686) | checked after loading (fp32): span exactness 9.3e-07, image vs mirror probe feature 5.4e-07, PASS; healing run's fp32 flip check 100.00%; fp32 forward passes, k-NN only |

### Exactly mirror-invariant options

- original (flipavg): k-NN 65.62, flip consistency 100.00%, linear probe 71.88, 2 forward passes per image, 0.02 GFLOPs per image, 304,424 parameters (1.000 of the original).
- healed_vexact_k4 (plain): k-NN 78.12, flip consistency 100.00%, linear probe 75.00, 1 forward pass per image, 0.00 GFLOPs per image, 208,776 parameters (0.686 of the original).
- healed_vexact_k4 (plain_fp32): k-NN 78.12, flip consistency 100.00%, linear probe -, 1 forward pass per image, 0.00 GFLOPs per image, 208,776 parameters (0.686 of the original).
- healed_vexact_k4 uses 0.301 of the GFLOPs per image and 0.686 of the parameters of the flip-averaged original; top-1 difference (V-exact minus flip-averaged original, points): k-NN 12.50, linear probe 3.12.

Source: `flipavg.csv` and `rows/<model>/<variant>__<mode>.json` in this folder (predictions in `knn/`). Commit(s): aeb57bed18; seed(s): 0. Full notes, feature checks and timings are in the CSV.
