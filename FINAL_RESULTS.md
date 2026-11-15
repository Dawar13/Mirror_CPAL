# FINAL_RESULTS: pre-registered predictions and cited numbers (next phase)

Written by `scripts/collect_next_phase.py` on 2026-11-15 20:29 UTC, code commit `6b4c66ab78` (uncommitted changes). Every number below is read from a result file written by a script of this repository and is listed with that file and the commit recorded in the run's meta.json. Nothing here is typed by hand; rerun the collector when new results arrive.

Pre-registration: `PREREG_NEXT_PHASE.md`, signed-off commit `e76fb17c3c` (sha256 of the current text `711b963039af`). Scored predictions: 40; PASS 17, FAIL 16, PENDING 7. Upside items reported: 2 of 7.

## Gates

| Gate | When | Condition | Status | Value | Evidence | Source |
|---|---|---|---|---|---|---|
| A | Before any new GPU job | PREREG_NEXT_PHASE.md committed after sign-off; MUST-00 clean | MET | sign-off commit e76fb17c3c; MUST-00: MATCH (rounded) 29, MATCH 9, CONTEXT 8, MISMATCH 2 | PREREG_NEXT_PHASE.md carries the sign-off line and was committed; MUST-00 audit present (mismatches reported in results/MUST-00/audit.md, accepted per results/STATUS.md) | `PREREG_NEXT_PHASE.md` (e76fb17c3c); `results/MUST-00/audit.json` (not recorded) |
| B | After MUST-07 | Best baseline chosen on the method-selection set and recorded (results/MUST-07/gate_b.json) before MUST-09 starts | MET | best svdllm, second slicegpt | ranking by method-selection k-NN at size full, fit seed 0; complete True | `results/MUST-07/gate_b.json` (805c231c9c) |
| C | After MUST-12 | LR model: LR ratio to null in blocks 1 to 6 at most 0.8 and LR twin fraction in blocks 1 to 6 at least 0.2 | NOT MET | ratio 0.4844, twin 0.0601 | scripts/h100_gate_c.py recorded the same values and verdict; CORE-01 epochs 100 | `results/MUST-12/vit_s8_LR_s0/e100/metrics.json` (e84e20ff35); `results/MUST-12/gate_c.json` (not recorded) |
| D | After MUST-09 | Mirror-exact mean at least best-baseline mean minus 1.0 | MET | Mirror-exact 78.34 vs svdllm 78.06 (difference +0.28; bar -1.0) |  | `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0/metrics.json` (52e29d1676); `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s1/metrics.json` (e84e20ff35) and 7 more |
| E | Before abstract registration (23 Nov) | MUST-04, MUST-05, MUST-07, MUST-12 done; CORE-01 running | PENDING | MUST-04 done; MUST-05 done; MUST-07 done; MUST-12 done; CORE-01 not started | files present (done = the experiment's scored outputs exist) | n/a |

## Pre-registered predictions

PENDING means an input is missing or incomplete (the reason is given). Failed predictions are failures and are reported as such.

| ID | Prediction | Value | Bar | n | Verdict | Source (commit) |
|---|---|---|---|---|---|---|
| MUST-01 | Predicted and achieved parameter savings agree to within 1 percentage point per layer, for all 7 measured models | 0 | <= 1.0 pp, 7 models | 144 blocks, 7 models | **PASS** | `results/MUST-01/law_per_layer.csv` (52e29d1676); `results/MUST-01/metrics.json` (52e29d1676) |
| MUST-02 | On 1,000 validation images in fp32, every component of Mirror-exact (full model) has maximum relative equivariance error below 1e-5, and flip consistency is 100% | max rel. error 2.51e-05 (block 11 act); flip 100.0000% | < 1e-5 and 100% | 1000 images, 198 components; flip on 50000 queries | **FAIL** | `results/MUST-02/dinov2_vitb14_reg/zeroshot/metrics.json` (e84e20ff35) |
| MUST-03 | Acceptance check of the generalised measurement code on dinov2_vitb14_reg with LR: the LR numbers of earlier runs are reproduced (twin pairs and breaking shares of block 1 within rounding of runs/x2/perlayer_reg_s0) | pairs 1168 vs 1168; \|ds_fc1\| 8.49e-09, \|ds_fc2\| 9.58e-09 | pairs equal; \|delta share\| < 0.005 | reference runs/x2/perlayer_reg_s0 | **PASS** | `results/MUST-04/dinov2_vitb14_reg/metrics.json` (e84e20ff35) |
| MUST-04 P4.1 | UD and R90 ratios to null in blocks 1 to 6 are at least 0.9 (LR's ratio is about 0.45) | min UD 0.4807 / min R90 0.7536 | >= 0.9 (UD and R90, every model) | 4 of 4 models | **FAIL** | `results/MUST-04/dinov2_vitb14_reg/metrics.json` (e84e20ff35); `results/MUST-04/dinov2_vitb14/metrics.json` (e84e20ff35) and 2 more |
| MUST-04 P4.2 | The UD twin fraction in blocks 1 to 6 is within the twin-fraction null's 95% interval | dinov2_vitb14_reg: 0.5600 in [0, 6.00e-04]; dinov2_vitb14: 0.5298 in [0.0016, 0.0035]; dinov2_vitl14_reg: 0.5458 in [0, 3.68e-04]; deit3_b16: 0.4630 in [0, 0] | inside [null 2.5%, null 97.5%] | 4 of 4 models | **FAIL** | `results/MUST-04/dinov2_vitb14_reg/metrics.json` (e84e20ff35); `results/MUST-04/dinov2_vitb14/metrics.json` (e84e20ff35) and 2 more |
| MUST-04 P4.3 | Zero-shot UD group-projection surgery on blocks 1 to 6 drops validation k-NN by more than 30 points, against 1.85 for LR | min UD drop 0.2320 | > 30 points | 4 of 4 models | **FAIL** | `results/MUST-04/dinov2_vitb14_reg/metrics.json` (e84e20ff35); `results/MUST-04/dinov2_vitb14/metrics.json` (e84e20ff35) and 2 more |
| MUST-05 P5.1 | Switch-off median cosine at least 0.7 for twin pairs and at most 0.2 for random pairs | twins 0.4875, random -9.22e-07 | >= 0.7 twins and <= 0.2 random | 1200 twin and 1200 random pairs, blocks [1, 2, 3, 4, 5, 6], 500 images | **FAIL** | `results/MUST-05/dinov2_vitb14_reg/switch_pairs.csv` (52e29d1676) |
| MUST-05 P5.2 | The norm ratio is between 0.8 and 1.25 for at least 70% of twin pairs | 0.9225 | >= 0.70 | 1200 twin and 1200 random pairs, blocks [1, 2, 3, 4, 5, 6], 500 images | **PASS** | `results/MUST-05/dinov2_vitb14_reg/switch_pairs.csv` (52e29d1676) |
| MUST-06 P6.1 | Median same-token Pearson correlation of twins on the same image at most 0.3 | 0.0516 | <= 0.3 | 5810 twin pairs | **PASS** | `results/MUST-06/dinov2_vitb14_reg/pairs.csv` (52e29d1676) |
| MUST-06 P6.2 | A twin is in its partner's top 10 same-image correlated neurons for at most 20% of pairs | 0.0812 | <= 0.20 | 5810 twin pairs | **PASS** | `results/MUST-06/dinov2_vitb14_reg/pairs.csv` (52e29d1676) |
| MUST-06 P6.3 | Effective rank (entropy-based) of fc1 and fc2 changes by at most 2% under group projection | 0.0864 | <= 0.02 for all 12 | 12 matrices | **FAIL** | `results/MUST-06/dinov2_vitb14_reg/pairs.csv` (52e29d1676) |
| MUST-07 P7.1 | On dinov2_vitb14_reg at 6 blocks, Mirror-exact beats every baseline on validation k-NN by at least 2 points, with the paired bootstrap interval excluding 0 | smallest margin -1.75 | >= 2.0 points and CI low > 0, every method, both sizes | 20 of 20 comparisons with 3 seeds | **FAIL** | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35); `results/MUST-07/bootstrap.csv` (805c231c9c) |
| MUST-07 P7.2 | ZipIt-style merging and same-image-correlation pairing trail Mirror-exact by at least 10 points at 6 blocks | smallest margin 0.9340 | >= 10.0 points | 4 of 4 comparisons with 3 seeds | **FAIL** | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| MUST-08 P8.1 | MLP-only saving is more than half of the MLP-plus-attention saving, at 6 blocks and at the full model | 6 blocks: 0.6713; full model: 0.6835 | > 0.5 at 6 blocks and at the full model | original model parameters from the original row | **PASS** | `results/MUST-08/rows_dinov2_vitb14_reg.csv` (805c231c9c) |
| MUST-08 P8.2 | k-NN(MLP only) minus k-NN(MLP plus attention) is at most 0.5 at 6 blocks (mean over fit seeds) | -0.7360 | <= 0.5 points | 3 fit seeds | **PASS** | `results/MUST-08/rows_dinov2_vitb14_reg.csv` (805c231c9c) |
| MUST-09 P9.1 | Mirror-exact is non-inferior to the best baseline: mean over seeds of Mirror-exact minus mean of the best baseline is above -0.5 points (validation k-NN) | 0.2793 | > -0.5 points | 3 + 3 seeds (best baseline: svdllm) | **PASS** | `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0/metrics.json` (52e29d1676); `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s1/metrics.json` (e84e20ff35) and 7 more |
| MUST-09 P9.2 | Only Mirror-exact has 100% flip consistency (scored as: Mirror-exact and the random-structure control reach 100% (fp32, rounding allowance); no compressor (best, second-best, plain SVD) does) | mirror_exact s0: 100.000%; mirror_exact s1: 100.000%; mirror_exact s2: 100.000%; random_structure s0: 100.000%; slicegpt s0: 95.432%; svd s0: 94.296%; svdllm s0: 95.804%; svdllm s1: 95.706%; svdllm s2: 95.732% | Mirror-exact and random structure 100%; compressors < 100% | fp32 pass per heal (50,000 queries each) | **PASS** | `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0/metrics.json` (52e29d1676); `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s1/metrics.json` (e84e20ff35) and 7 more |
| MUST-09 P9.3 | Random structure trails Mirror-exact by at least 3 points | 8.41 | >= 3.0 points | 3 seeds against 1 | **PASS** | `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0/metrics.json` (52e29d1676); `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s1/metrics.json` (e84e20ff35) and 7 more |
| MUST-10 P10.1 | Healed Mirror-exact (mean of 3 seeds) is within 1 point of flip averaging on validation k-NN, at under 45% of its FLOPs (analytic, one pass against two) | difference -0.66 points; FLOPs ratio 0.387 | \|difference\| <= 1.0 and FLOPs ratio < 0.45 | 3 healed seeds against 1 flip-averaged original | **PASS** | `results/MUST-10/flipavg.csv` (e84e20ff35); `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0/metrics.json` (52e29d1676) and 2 more |
| MUST-11 P11.1 | Healed Mirror-exact is within 1.0 point of the original on the ImageNet linear probe (mean of 3 seeds) and on ADE20K mIoU (seed 0) | probe -0.89; ADE20K -0.43 | both <= 1.0 | linear probe 3 seeds; ADE20K seed 0 | **PASS** | `results/MUST-11/t2/t2_linear.csv` (805c231c9c); `results/MUST-11/t2_seg/t2_ade20k.csv` (805c231c9c) |
| MUST-11 P11.2 | Mirror-exact with the symmetrised head gives 100% mask equivariance; the original model stays below 100% | original: raw 95.0309, sym 94.7413; best_svdllm_s0: raw 94.7590, sym 94.6249; vexact_s0: raw 98.9771, sym 99.9999 | Mirror-exact 100%; original < 100% | n/a | **PENDING** | `results/MUST-11/t2_seg/t2_ade20k.csv` (805c231c9c) |
| MUST-12 gate C | Gate C: in the LR model, the LR ratio to null in blocks 1 to 6 is at most 0.8 and the LR twin fraction in blocks 1 to 6 is at least 0.2 | ratio 0.4844, twin 0.0601 | ratio <= 0.8 and twin >= 0.2 | epoch 100 | **FAIL** | `results/MUST-12/vit_s8_LR_s0/e100/metrics.json` (e84e20ff35); `results/MUST-12/gate_c.json` (not recorded) |
| CORE-01 T1 | Dissociation: in every seed pair s = 0, 1, 2, the UD twin fraction (blocks 1 to 6, final) of UD run s minus that of LR run s is at least 0.2 (absolute) | min difference n/a | >= 0.2 in every pair | 0 of 3 seed pairs | **PENDING** | `results/MUST-12/vit_s8_LR_s0/e100/metrics.json` (e84e20ff35) |
| CORE-01 T2 | Specificity: the UD ratio to null is at most 0.7 in every UD run, and at least 0.9 in every none run and every LR run (blocks 1 to 6, final) | UD s0: n/a; UD s1: n/a; UD s2: n/a; none s0: 0.7026; none s1: n/a; none s2: n/a; LR s0: 0.6856; LR s1: n/a; LR s2: n/a | UD runs <= 0.7; none and LR runs >= 0.9 | 2 of 9 runs | **FAIL** | `results/MUST-12/vit_s8_none_s0/e100/metrics.json` (e84e20ff35); `results/MUST-12/vit_s8_LR_s0/e100/metrics.json` (e84e20ff35) |
| CORE-01 T3 | Rotation: in every R90 run, the share in 4-cycles (blocks 1 to 6, final) is at least 0.3, and at least 3 times the mean share in the none runs | R90 s0: n/a; R90 s1: n/a; none mean n/a | >= 0.3 and >= 3 x none mean, every R90 run | 2 R90 runs, 3 none runs | **PENDING** | `results/MUST-12/vit_s8_none_s0/e100/metrics.json` (e84e20ff35) |
| CORE-01 T4 | Emergence: in every LR run (for LR) and every UD run (for UD), the twin fraction of the trained group is higher at the final epoch than at epoch 5 | LR s0: 0.0846 -> 0.0601; LR s1: n/a -> n/a; LR s2: n/a -> n/a; UD s0: n/a -> n/a; UD s1: n/a -> n/a; UD s2: n/a -> n/a | final > epoch 5, every LR and UD run | 1 of 6 runs | **FAIL** | `results/MUST-12/vit_s8_LR_s0/e100/metrics.json` (e84e20ff35) |
| CORE-02 P2.1 | The achieved saving equals the prediction (identity check) | n/a | equal in every block | n/a | **PENDING** | n/a |
| CORE-02 P2.2 | Trained-group surgery on blocks 1 to 6 keeps validation top-1 within 3 points | n/a | <= 3.0 points | n/a | **PENDING** | n/a |
| CORE-02 P2.3 | Every wrong-group surgery loses at least 20 points | n/a | >= 20 points, every row | n/a | **PENDING** | n/a |
| CORE-03 P3.1 | Mirror plus low rank lies above low rank alone (the better of plain SVD and SVD-LLM) at at least 3 of 4 budget points on at least 2 of 3 models, including DINOv2-L or DeiT-III (zero-shot) | dinov2_vitb14_reg: 0 of 4 points above; dinov2_vitl14_reg: 0 of 4 points above; deit3_b16: 0 of 4 points above | >= 3 of 4 points on >= 2 of 3 models, one of them DINOv2-L or DeiT-III | 12 budget points | **FAIL** | `results/CORE-03/curves.csv` (805c231c9c) |
| CORE-04 P4.1 | Scale: the tax in blocks 1 to L/2 increases from DINOv2-S to g (either direction is reported) | s14 0.4002, b14 0.4185, l14 0.4140, g14 0.3440 | S < B < L < g | 4 of 4 sizes | **FAIL** | `results/CORE-04/structure/dinov2_vits14_reg/metrics.json` (e84e20ff35); `results/MUST-04/dinov2_vitb14_reg/metrics.json` (e84e20ff35) and 2 more |
| CORE-04 P4.2 | Models whose recipe is verified to have no flips have a lower LR tax than every flip-trained model of the same size | clip_b16: 0.3125; convnext_b: 0.2168; deit3_b16: 0.3487; dino_b16: 0.2495; dinov2_vitb14: 0.4048; dinov2_vitb14_reg: 0.4185; dinov2_vitg14_reg: 0.3440; dinov2_vitl14_reg: 0.4140; dinov2_vits14_reg: 0.4002; dinov3_b16: 0.4002; eva02_b14: 0.1647; mae_b16: 0.4482; resnet50: 0.2794; siglip2_b16: 0.2309 | below every flip-trained model of its size | 1 verified no-flip model(s): clip_b16 | **FAIL** | `results/CORE-04/structure/clip_b16/metrics.json` (not recorded); `results/MUST-04/deit3_b16/metrics.json` (not recorded) and 5 more |
| CORE-04 P4.3 | CNNs: twin structure appears in ConvNeXt's pointwise MLPs (LR ratio to null at most 0.8 and twin fraction at least 0.2 in the first half of the stages' pointwise MLPs) | ratio 0.5034, twin 0.2319 | ratio <= 0.8 and twin >= 0.2 | n/a | **PASS** | `results/CORE-04/structure/convnext_b/metrics.json` (e84e20ff35) |
| CORE-05 P5.1 | The activation-weighted breaking share is at most half the weight-energy breaking share in blocks 1 to 6 | 0.5803 | <= 0.5 | 12 matrices | **FAIL** | `results/CORE-05/measures_dinov2_vitb14_reg.csv` (805c231c9c) |
| CORE-05 P5.2 | Over training (CORE-01 LR and UD runs), breaking energy in the top-variance band falls faster than in the bottom band | n/a | top fall > bottom fall | n/a | **PENDING** | n/a |
| CORE-05 P5.3 | The out-of-sample rule is correct on at least 75% of layer-model cases on DINOv2-L-reg, DeiT-III and MAE | 0.6667 | >= 0.75 | 48 layer-model cases | **FAIL** | `results/CORE-05/rule_score.json` (805c231c9c) |
| CORE-06 P6.1 | Species from the even part is within 1 point of the full feature | original: 90.01 vs 89.96; vexact_s0: 89.42 vs 89.45; vexact_s1: 89.47 vs 89.52; vexact_s2: 89.37 vs 89.44 | <= 1.0 points, every model | 4 model(s) | **PASS** | `results/CORE-06/probes.csv` (805c231c9c) |
| CORE-06 P6.2 | Facing direction from the odd part is at least 85% | original: 94.99; vexact_s0: 99.19; vexact_s1: 99.21; vexact_s2: 99.25 | >= 85%, every model | 4 model(s) | **PASS** | `results/CORE-06/probes.csv` (805c231c9c) |
| CORE-06 P6.3 | Facing direction from healed Mirror-exact's even part is at most 60% on natural test images, and exactly 50% on the mirror-augmented test set | vexact_s0: 50.84 / 50.00; vexact_s1: 50.29 / 50.00; vexact_s2: 50.57 / 50.00 | <= 60% and = 50% | 3 model(s) | **PASS** | `results/CORE-06/probes.csv` (805c231c9c) |
| CORE-07 P7.1 | Mirror-exact's drop from ImageNet to ImageNet-V2 (matched frequency, linear probe) is within 1 point of the original's drop | drops: Mirror-exact 9.17 (mean of 3), original 8.62 | <= 1.0 points | 3 Mirror-exact seed(s) | **PASS** | `results/CORE-07/shift_rows.csv` (805c231c9c) |
| UP-01 | UD tax rises with p (0.1, 0.25, 0.5) | n/a | none (reported) | n/a | **PENDING** | n/a |
| UP-02 | Breaking share stays below 0.05 throughout training from a UD-equivariant start, at accuracy within 1 point of the UD runs | n/a | none (reported) | n/a | **PENDING** | n/a |
| UP-03 | The trained-group tax differs between ViT-S and ViT-Ti; either direction is reported | n/a | none (reported) | n/a | **PENDING** | n/a |
| UP-04 | Tied-from-scratch reaches about the same accuracy at the reduced parameter count | n/a | none (reported) | n/a | **PENDING** | n/a |
| UP-05 | A speed claim needs more than 1.1x the original's speed; otherwise it is a limitation | n/a | none (reported) | n/a | **PENDING** | n/a |
| UP-06 | The extra drop from INT8 weight-only quantisation and from ToMe (r = 8) is the same for both models within 0.5 points | see details | none (reported) | n/a | **REPORTED** | `results/UP-06/comparison.json` (e84e20ff35) |
| UP-07 | Pairing from weights alone agrees with the data-driven pairing on at least 70% of pairs | see details | none (reported) | n/a | **REPORTED** | `results/UP-07/up07.json` (e84e20ff35) |

## Scoring details

For each prediction: the scored quantity as computed, the reason for a PENDING verdict, and the values per model, run, seed or seed pair.

### MUST-01: PASS

- Prediction: Predicted and achieved parameter savings agree to within 1 percentage point per layer, for all 7 measured models.
- Scored quantity: largest |predicted - achieved| saving over every layer (fc1, fc2) and block (MLP), in percentage points.
- Bar: <= 1.0 pp, 7 models. Value: 0. n: 144 blocks, 7 models.
- Sources: `results/MUST-01/law_per_layer.csv` (52e29d1676); `results/MUST-01/metrics.json` (52e29d1676); pre-registration commit e76fb17c3c.
- Notes: structure synthetic with the recorded counts (arithmetic check only) for dinov2_vitl14_reg, dinov2_vits14_reg.

| model | n_blocks | fit_seeds | max_abs_diff_pp | structure | verdict |
|---|---|---|---|---|---|
| deit3_b16 | 12 | [0] | 0 | fitted operators and pairings | PASS |
| dinov2_vitb14 | 36 | [0, 1, 2] | 0 | fitted operators and pairings | PASS |
| dinov2_vitb14_reg | 36 | [0, 1, 2] | 0 | fitted operators and pairings | PASS |
| dinov2_vitl14_reg | 24 | [0] | 0 | synthetic structure with the recorded counts | PASS |
| dinov2_vits14_reg | 12 | [0] | 0 | synthetic structure with the recorded counts | PASS |
| mae_b16 | 12 | [0] | 0 | fitted operators and pairings | PASS |
| siglip2_b16 | 12 | [0] | 0 | fitted operators and pairings | PASS |

### MUST-02: FAIL

- Prediction: On 1,000 validation images in fp32, every component of Mirror-exact (full model) has maximum relative equivariance error below 1e-5, and flip consistency is 100%.
- Scored quantity: max over images and components of ||T out(x) - out(mirror x)|| / ||out(mirror x)|| (fp32, zero-shot, fit seed 0); k-NN flip consistency with fp32 features on all 50,000 validation images (rounding allowance).
- Bar: < 1e-5 and 100%. Value: max rel. error 2.51e-05 (block 11 act); flip 100.0000%. n: 1000 images, 198 components; flip on 50000 queries.
- Sources: `results/MUST-02/dinov2_vitb14_reg/zeroshot/metrics.json` (e84e20ff35); pre-registration commit e76fb17c3c.
- Notes: flip check: 100% raw (no disagreement).

| variant | dtype | max_rel_tensor | worst_component | max_rel_token | n_components | components_verdict | flip_consistency_fp32 | n_disagree | n_query | flip_verdict | flip_consistency_bf16 | n_images | git_hash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zeroshot | float32 | 2.51e-05 | block 11 act | 1.80e-04 | 198 | FAIL | 100.00 | 0 | 50,000 | PASS | 98.05 | 1000 | e84e20ff35bb1ac4c3e7e3406fba5d633fb3c9f9 |
| zeroshot_fp64 | float64 | 5.77e-14 | block 4 v | 3.39e-13 | 198 | PASS | n/a | n/a | n/a | PENDING | n/a | 1000 | 52e29d16766d80933db138b9414e4f69a81e4b7e |

### MUST-03: PASS

- Prediction: Acceptance check of the generalised measurement code on dinov2_vitb14_reg with LR: the LR numbers of earlier runs are reproduced (twin pairs and breaking shares of block 1 within rounding of runs/x2/perlayer_reg_s0).
- Scored quantity: block 1: twin pairs equal; |s_fc1 - s1_ref| and |s_fc2 - s2_ref| below 0.005 (half a unit of the 2-decimal values quoted, 0.27 and 0.49); the synthetic-network tests are the unit tests, not scored here.
- Bar: pairs equal; |delta share| < 0.005. Value: pairs 1168 vs 1168; |ds_fc1| 8.49e-09, |ds_fc2| 9.58e-09. n: reference runs/x2/perlayer_reg_s0.
- Sources: `results/MUST-04/dinov2_vitb14_reg/metrics.json` (e84e20ff35); pre-registration commit e76fb17c3c.

| reference | reference_fit_seed | fit_seed | blocks_compared | layers_compared | d_plus_max_abs_diff | n_pairs_max_abs_diff | n_self_max_abs_diff | s_fc1_max_abs_diff | s_fc2_max_abs_diff | block1 |
|---|---|---|---|---|---|---|---|---|---|---|
| runs/x2/perlayer_reg_s0 | 0 | 0 | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] | 0 | 0 | 0 | 4.51e-08 | 1.49e-08 | {"n_pairs": 1168, "n_pairs_ref": 1168, "s_fc1": 0.2670600646 |

### MUST-04 P4.1: FAIL

- Prediction: UD and R90 ratios to null in blocks 1 to 6 are at least 0.9 (LR's ratio is about 0.45).
- Scored quantity: ratio to null, blocks 1 to 6 (mean breaking share of fc1 and fc2 over the blocks / mean of the null means); per model, fit seed 0; claim passes only if it holds on all four models (PREREG Section 9, item 2).
- Bar: >= 0.9 (UD and R90, every model). Value: min UD 0.4807 / min R90 0.7536. n: 4 of 4 models.
- Sources: `results/MUST-04/dinov2_vitb14_reg/metrics.json` (e84e20ff35); `results/MUST-04/dinov2_vitb14/metrics.json` (e84e20ff35); `results/MUST-04/dinov2_vitl14_reg/metrics.json` (e84e20ff35); `results/MUST-04/deit3_b16/metrics.json` (e84e20ff35); pre-registration commit e76fb17c3c.

| model | ratio_LR | twin_LR | ratio_UD | twin_UD | ratio_R90 | twin_R90 | ratio_R180 | twin_R180 | tax_LR_1-6 | ud_twin_observed | ud_twin_null_mean | ud_twin_null_lo | ud_twin_null_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov2_vitb14_reg | 0.4791 | 0.5677 | 0.5129 | 0.5600 | 0.8494 | 0.0474 | 0.5272 | 0.6228 | 0.4185 | 0.5600 | 1.63e-04 | 0 | 6.00e-04 |
| dinov2_vitb14 | 0.4474 | 0.5374 | 0.4956 | 0.5298 | 0.8366 | 0.0543 | 0.5081 | 0.5732 | 0.4048 | 0.5298 | 0.0023 | 0.0016 | 0.0035 |
| dinov2_vitl14_reg | 0.5020 | 0.5432 | 0.5114 | 0.5458 | 0.7555 | 0.0182 | 0.5287 | 0.5917 | 0.4452 | 0.5458 | 7.73e-05 | 0 | 3.68e-04 |
| deit3_b16 | 0.4501 | 0.4301 | 0.4807 | 0.4630 | 0.7536 | 0.0862 | 0.4807 | 0.5047 | 0.3487 | 0.4630 | 0 | 0 | 0 |

### MUST-04 P4.2: FAIL

- Prediction: The UD twin fraction in blocks 1 to 6 is within the twin-fraction null's 95% interval.
- Scored quantity: UD twin fraction (threshold 0.5), blocks 1 to 6, against the 2.5th and 97.5th percentiles of the twin-fraction null (random patch permutation with the same cycle type); per model, fit seed 0; claim passes only if it holds on all four models (PREREG Section 9, item 2).
- Bar: inside [null 2.5%, null 97.5%]. Value: dinov2_vitb14_reg: 0.5600 in [0, 6.00e-04]; dinov2_vitb14: 0.5298 in [0.0016, 0.0035]; dinov2_vitl14_reg: 0.5458 in [0, 3.68e-04]; deit3_b16: 0.4630 in [0, 0]. n: 4 of 4 models.
- Sources: `results/MUST-04/dinov2_vitb14_reg/metrics.json` (e84e20ff35); `results/MUST-04/dinov2_vitb14/metrics.json` (e84e20ff35); `results/MUST-04/dinov2_vitl14_reg/metrics.json` (e84e20ff35); `results/MUST-04/deit3_b16/metrics.json` (e84e20ff35); pre-registration commit e76fb17c3c.

| model | ratio_LR | twin_LR | ratio_UD | twin_UD | ratio_R90 | twin_R90 | ratio_R180 | twin_R180 | tax_LR_1-6 | ud_twin_observed | ud_twin_null_mean | ud_twin_null_lo | ud_twin_null_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov2_vitb14_reg | 0.4791 | 0.5677 | 0.5129 | 0.5600 | 0.8494 | 0.0474 | 0.5272 | 0.6228 | 0.4185 | 0.5600 | 1.63e-04 | 0 | 6.00e-04 |
| dinov2_vitb14 | 0.4474 | 0.5374 | 0.4956 | 0.5298 | 0.8366 | 0.0543 | 0.5081 | 0.5732 | 0.4048 | 0.5298 | 0.0023 | 0.0016 | 0.0035 |
| dinov2_vitl14_reg | 0.5020 | 0.5432 | 0.5114 | 0.5458 | 0.7555 | 0.0182 | 0.5287 | 0.5917 | 0.4452 | 0.5458 | 7.73e-05 | 0 | 3.68e-04 |
| deit3_b16 | 0.4501 | 0.4301 | 0.4807 | 0.4630 | 0.7536 | 0.0862 | 0.4807 | 0.5047 | 0.3487 | 0.4630 | 0 | 0 | 0 |

### MUST-04 P4.3: FAIL

- Prediction: Zero-shot UD group-projection surgery on blocks 1 to 6 drops validation k-NN by more than 30 points, against 1.85 for LR.
- Scored quantity: original top-1 minus top-1 after UD surgery (per-layer operators, MLP only, blocks 1 to 6), validation k-NN; per model, fit seed 0; claim passes only if it holds on all four models (PREREG Section 9, item 2).
- Bar: > 30 points. Value: min UD drop 0.2320. n: 4 of 4 models.
- Sources: `results/MUST-04/dinov2_vitb14_reg/metrics.json` (e84e20ff35); `results/MUST-04/dinov2_vitb14/metrics.json` (e84e20ff35); `results/MUST-04/dinov2_vitl14_reg/metrics.json` (e84e20ff35); `results/MUST-04/deit3_b16/metrics.json` (e84e20ff35); pre-registration commit e76fb17c3c.

| model | ratio_LR | twin_LR | ratio_UD | twin_UD | ratio_R90 | twin_R90 | ratio_R180 | twin_R180 | tax_LR_1-6 | ud_twin_observed | ud_twin_null_mean | ud_twin_null_lo | ud_twin_null_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov2_vitb14_reg | 0.4791 | 0.5677 | 0.5129 | 0.5600 | 0.8494 | 0.0474 | 0.5272 | 0.6228 | 0.4185 | 0.5600 | 1.63e-04 | 0 | 6.00e-04 |
| dinov2_vitb14 | 0.4474 | 0.5374 | 0.4956 | 0.5298 | 0.8366 | 0.0543 | 0.5081 | 0.5732 | 0.4048 | 0.5298 | 0.0023 | 0.0016 | 0.0035 |
| dinov2_vitl14_reg | 0.5020 | 0.5432 | 0.5114 | 0.5458 | 0.7555 | 0.0182 | 0.5287 | 0.5917 | 0.4452 | 0.5458 | 7.73e-05 | 0 | 3.68e-04 |
| deit3_b16 | 0.4501 | 0.4301 | 0.4807 | 0.4630 | 0.7536 | 0.0862 | 0.4807 | 0.5047 | 0.3487 | 0.4630 | 0 | 0 | 0 |

### MUST-05 P5.1: FAIL

- Prediction: Switch-off median cosine at least 0.7 for twin pairs and at most 0.2 for random pairs.
- Scored quantity: median over pairs of the switch-off cosine at the output of block l+1 (per-pair cosine pooled over images and tokens), pooled over the 200 strongest twin pairs (and 200 random pairs) of each of blocks 1 to 6, dinov2_vitb14_reg, 500 validation images.
- Bar: >= 0.7 twins and <= 0.2 random. Value: twins 0.4875, random -9.22e-07. n: 1200 twin and 1200 random pairs, blocks [1, 2, 3, 4, 5, 6], 500 images.
- Sources: `results/MUST-05/dinov2_vitb14_reg/switch_pairs.csv` (52e29d1676); pre-registration commit e76fb17c3c.

| block | twin_median_cos | random_median_cos | twin_share_ratio_in_band | n_twin | n_random |
|---|---|---|---|---|---|
| 1 | 0.5194 | 5.26e-04 | 0.8600 | 200 | 200 |
| 2 | 0.4878 | 6.38e-04 | 0.8550 | 200 | 200 |
| 3 | 0.4806 | 3.32e-04 | 0.9300 | 200 | 200 |
| 4 | 0.4296 | 5.65e-04 | 0.9550 | 200 | 200 |
| 5 | 0.4579 | -6.18e-04 | 0.9800 | 200 | 200 |
| 6 | 0.5411 | -0.0018 | 0.9550 | 200 | 200 |

### MUST-05 P5.2: PASS

- Prediction: The norm ratio is between 0.8 and 1.25 for at least 70% of twin pairs.
- Scored quantity: share of twin pairs (blocks 1 to 6 pooled) with ||Delta_j|| / ||A Delta_i|| in [0.8, 1.25].
- Bar: >= 0.70. Value: 0.9225. n: 1200 twin and 1200 random pairs, blocks [1, 2, 3, 4, 5, 6], 500 images.
- Sources: `results/MUST-05/dinov2_vitb14_reg/switch_pairs.csv` (52e29d1676); pre-registration commit e76fb17c3c.

| block | twin_median_cos | random_median_cos | twin_share_ratio_in_band | n_twin | n_random |
|---|---|---|---|---|---|
| 1 | 0.5194 | 5.26e-04 | 0.8600 | 200 | 200 |
| 2 | 0.4878 | 6.38e-04 | 0.8550 | 200 | 200 |
| 3 | 0.4806 | 3.32e-04 | 0.9300 | 200 | 200 |
| 4 | 0.4296 | 5.65e-04 | 0.9550 | 200 | 200 |
| 5 | 0.4579 | -6.18e-04 | 0.9800 | 200 | 200 |
| 6 | 0.5411 | -0.0018 | 0.9550 | 200 | 200 |

### MUST-06 P6.1: PASS

- Prediction: Median same-token Pearson correlation of twins on the same image at most 0.3.
- Scored quantity: median over twin pairs of the same-token Pearson correlation (post-GELU); blocks 1 to 6 pooled; scored on dinov2_vitb14_reg (the claim), reported per model for the other two.
- Bar: <= 0.3. Value: 0.0516. n: 5810 twin pairs.
- Sources: `results/MUST-06/dinov2_vitb14_reg/pairs.csv` (52e29d1676); pre-registration commit e76fb17c3c.
- Notes: other models (reported): dinov2_vitl14_reg: PASS; deit3_b16: PASS.

| model | git_hash | median_same_corr | median_mirror_corr | share_seen | n_twin_pairs | blocks | max_abs_erank_change | n_matrices | P6.1 | P6.2 | P6.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov2_vitb14_reg | 52e29d16766d80933db138b9414e4f69a81e4b7e | 0.0516 | 0.5513 | 0.0812 | 5810 | [1, 2, 3, 4, 5, 6] | 0.0864 | 12 | PASS | PASS | FAIL |
| dinov2_vitl14_reg | e84e20ff35bb1ac4c3e7e3406fba5d633fb3c9f9 | 0.0870 | 0.5773 | 0.1043 | 7255 | [1, 2, 3, 4, 5, 6] | 0.2161 | 12 | PASS | PASS | FAIL |
| deit3_b16 | e84e20ff35bb1ac4c3e7e3406fba5d633fb3c9f9 | 0.0331 | 0.5350 | 0.0388 | 4385 | [1, 2, 3, 4, 5, 6] | 0.1764 | 12 | PASS | PASS | FAIL |

### MUST-06 P6.2: PASS

- Prediction: A twin is in its partner's top 10 same-image correlated neurons for at most 20% of pairs.
- Scored quantity: share of twin pairs where the twin is in the top 10 |r| partners of either neuron; blocks 1 to 6 pooled; scored on dinov2_vitb14_reg (the claim), reported per model for the other two.
- Bar: <= 0.20. Value: 0.0812. n: 5810 twin pairs.
- Sources: `results/MUST-06/dinov2_vitb14_reg/pairs.csv` (52e29d1676); pre-registration commit e76fb17c3c.
- Notes: other models (reported): dinov2_vitl14_reg: PASS; deit3_b16: PASS.

| model | git_hash | median_same_corr | median_mirror_corr | share_seen | n_twin_pairs | blocks | max_abs_erank_change | n_matrices | P6.1 | P6.2 | P6.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov2_vitb14_reg | 52e29d16766d80933db138b9414e4f69a81e4b7e | 0.0516 | 0.5513 | 0.0812 | 5810 | [1, 2, 3, 4, 5, 6] | 0.0864 | 12 | PASS | PASS | FAIL |
| dinov2_vitl14_reg | e84e20ff35bb1ac4c3e7e3406fba5d633fb3c9f9 | 0.0870 | 0.5773 | 0.1043 | 7255 | [1, 2, 3, 4, 5, 6] | 0.2161 | 12 | PASS | PASS | FAIL |
| deit3_b16 | e84e20ff35bb1ac4c3e7e3406fba5d633fb3c9f9 | 0.0331 | 0.5350 | 0.0388 | 4385 | [1, 2, 3, 4, 5, 6] | 0.1764 | 12 | PASS | PASS | FAIL |

### MUST-06 P6.3: FAIL

- Prediction: Effective rank (entropy-based) of fc1 and fc2 changes by at most 2% under group projection.
- Scored quantity: largest |relative change| of exp(entropy of s / sum s) over the 12 matrices (fc1, fc2 of blocks 1 to 6); holds if every matrix satisfies it.
- Bar: <= 0.02 for all 12. Value: 0.0864. n: 12 matrices.
- Sources: `results/MUST-06/dinov2_vitb14_reg/pairs.csv` (52e29d1676); pre-registration commit e76fb17c3c.
- Notes: other models (reported): dinov2_vitl14_reg: FAIL; deit3_b16: FAIL.

| model | git_hash | median_same_corr | median_mirror_corr | share_seen | n_twin_pairs | blocks | max_abs_erank_change | n_matrices | P6.1 | P6.2 | P6.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov2_vitb14_reg | 52e29d16766d80933db138b9414e4f69a81e4b7e | 0.0516 | 0.5513 | 0.0812 | 5810 | [1, 2, 3, 4, 5, 6] | 0.0864 | 12 | PASS | PASS | FAIL |
| dinov2_vitl14_reg | e84e20ff35bb1ac4c3e7e3406fba5d633fb3c9f9 | 0.0870 | 0.5773 | 0.1043 | 7255 | [1, 2, 3, 4, 5, 6] | 0.2161 | 12 | PASS | PASS | FAIL |
| deit3_b16 | e84e20ff35bb1ac4c3e7e3406fba5d633fb3c9f9 | 0.0331 | 0.5350 | 0.0388 | 4385 | [1, 2, 3, 4, 5, 6] | 0.1764 | 12 | PASS | PASS | FAIL |

### MUST-07 P7.1: FAIL

- Prediction: On dinov2_vitb14_reg at 6 blocks, Mirror-exact beats every baseline on validation k-NN by at least 2 points, with the paired bootstrap interval excluding 0.
- Scored quantity: for each of methods 3 to 12 and each 6-block size (MLP only; MLP plus attention): mean over fit seeds 0, 1, 2 of Mirror-exact minus mean of the method; 95% paired bootstrap interval on the fit-seed-0 per-image predictions (10,000 resamples).
- Bar: >= 2.0 points and CI low > 0, every method, both sizes. Value: smallest margin -1.75. n: 20 of 20 comparisons with 3 seeds.
- Sources: `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35); `results/MUST-07/bootstrap.csv` (805c231c9c); pre-registration commit e76fb17c3c.

| size | method | seeds_mirror | seeds_method | mirror_mean | method_mean | diff_mean | boot_diff_s0 | ci_low | ci_high | boot_n | P7.1 | P7.2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6mlp | svd | [0, 1, 2] | [0, 1, 2] | 76.60 | 9.37 | 67.24 | 67.23 | 66.80 | 67.64 | 50,000 | PASS |  |
| 6mlp | svdllm | [0, 1, 2] | [0, 1, 2] | 76.60 | 78.35 | -1.75 | -1.74 | -1.97 | -1.52 | 50,000 | FAIL |  |
| 6mlp | asvd_scale | [0, 1, 2] | [0, 1, 2] | 76.60 | 76.81 | -0.2040 | -0.2640 | -0.4920 | -0.0360 | 50,000 | FAIL |  |
| 6mlp | slicegpt | [0, 1, 2] | [0, 1, 2] | 76.60 | 31.41 | 45.19 | 44.50 | 44.03 | 44.97 | 50,000 | PASS |  |
| 6mlp | flap | [0, 1, 2] | [0, 1, 2] | 76.60 | 75.20 | 1.41 | 1.46 | 1.22 | 1.71 | 50,000 | FAIL |  |
| 6mlp | zipit | [0, 1, 2] | [0, 1, 2] | 76.60 | 75.67 | 0.9340 | 0.8180 | 0.5820 | 1.05 | 50,000 | FAIL | FAIL |
| 6mlp | pair_actcorr | [0, 1, 2] | [0, 1, 2] | 76.60 | 27.42 | 49.19 | 49.01 | 48.53 | 49.47 | 50,000 | PASS | PASS |
| 6mlp | pair_wcos | [0, 1, 2] | [0, 1, 2] | 76.60 | 34.42 | 42.19 | 42.58 | 42.12 | 43.05 | 50,000 | PASS |  |
| 6mlp | pair_random | [0, 1, 2] | [0, 1, 2] | 76.60 | 22.40 | 54.20 | 53.65 | 53.18 | 54.12 | 50,000 | PASS |  |
| 6mlp | random_structure | [0, 1, 2] | [0, 1, 2] | 76.60 | 2.31 | 74.29 | 74.34 | 73.95 | 74.72 | 50,000 | PASS |  |
| 6both | svd | [0, 1, 2] | [0, 1, 2] | 77.34 | 5.87 | 71.47 | 71.57 | 71.16 | 71.97 | 50,000 | PASS |  |
| 6both | svdllm | [0, 1, 2] | [0, 1, 2] | 77.34 | 78.02 | -0.6840 | -0.6160 | -0.8180 | -0.4160 | 50,000 | FAIL |  |
| 6both | asvd_scale | [0, 1, 2] | [0, 1, 2] | 77.34 | 74.41 | 2.92 | 2.98 | 2.71 | 3.24 | 50,000 | PASS |  |
| 6both | slicegpt | [0, 1, 2] | [0, 1, 2] | 77.34 | 75.60 | 1.74 | 1.69 | 1.44 | 1.94 | 50,000 | FAIL |  |
| 6both | flap | [0, 1, 2] | [0, 1, 2] | 77.34 | 70.29 | 7.05 | 7.25 | 6.93 | 7.58 | 50,000 | PASS |  |
| 6both | zipit | [0, 1, 2] | [0, 1, 2] | 77.34 | 18.51 | 58.83 | 57.93 | 57.47 | 58.39 | 50,000 | PASS | PASS |
| 6both | pair_actcorr | [0, 1, 2] | [0, 1, 2] | 77.34 | 31.31 | 46.03 | 45.49 | 45.01 | 45.95 | 50,000 | PASS | PASS |
| 6both | pair_wcos | [0, 1, 2] | [0, 1, 2] | 77.34 | 39.85 | 37.48 | 37.35 | 36.89 | 37.81 | 50,000 | PASS |  |
| 6both | pair_random | [0, 1, 2] | [0, 1, 2] | 77.34 | 32.30 | 45.04 | 42.80 | 42.33 | 43.27 | 50,000 | PASS |  |
| 6both | random_structure | [0, 1, 2] | [0, 1, 2] | 77.34 | 2.29 | 75.05 | 74.67 | 74.28 | 75.06 | 50,000 | PASS |  |

### MUST-07 P7.2: FAIL

- Prediction: ZipIt-style merging and same-image-correlation pairing trail Mirror-exact by at least 10 points at 6 blocks.
- Scored quantity: mean over fit seeds 0, 1, 2 of Mirror-exact minus mean of zipit and of pair_actcorr, both 6-block sizes.
- Bar: >= 10.0 points. Value: smallest margin 0.9340. n: 4 of 4 comparisons with 3 seeds.
- Sources: `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35); pre-registration commit e76fb17c3c.

| size | method | seeds_mirror | seeds_method | mirror_mean | method_mean | diff_mean | boot_diff_s0 | ci_low | ci_high | boot_n | P7.1 | P7.2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6mlp | svd | [0, 1, 2] | [0, 1, 2] | 76.60 | 9.37 | 67.24 | 67.23 | 66.80 | 67.64 | 50,000 | PASS |  |
| 6mlp | svdllm | [0, 1, 2] | [0, 1, 2] | 76.60 | 78.35 | -1.75 | -1.74 | -1.97 | -1.52 | 50,000 | FAIL |  |
| 6mlp | asvd_scale | [0, 1, 2] | [0, 1, 2] | 76.60 | 76.81 | -0.2040 | -0.2640 | -0.4920 | -0.0360 | 50,000 | FAIL |  |
| 6mlp | slicegpt | [0, 1, 2] | [0, 1, 2] | 76.60 | 31.41 | 45.19 | 44.50 | 44.03 | 44.97 | 50,000 | PASS |  |
| 6mlp | flap | [0, 1, 2] | [0, 1, 2] | 76.60 | 75.20 | 1.41 | 1.46 | 1.22 | 1.71 | 50,000 | FAIL |  |
| 6mlp | zipit | [0, 1, 2] | [0, 1, 2] | 76.60 | 75.67 | 0.9340 | 0.8180 | 0.5820 | 1.05 | 50,000 | FAIL | FAIL |
| 6mlp | pair_actcorr | [0, 1, 2] | [0, 1, 2] | 76.60 | 27.42 | 49.19 | 49.01 | 48.53 | 49.47 | 50,000 | PASS | PASS |
| 6mlp | pair_wcos | [0, 1, 2] | [0, 1, 2] | 76.60 | 34.42 | 42.19 | 42.58 | 42.12 | 43.05 | 50,000 | PASS |  |
| 6mlp | pair_random | [0, 1, 2] | [0, 1, 2] | 76.60 | 22.40 | 54.20 | 53.65 | 53.18 | 54.12 | 50,000 | PASS |  |
| 6mlp | random_structure | [0, 1, 2] | [0, 1, 2] | 76.60 | 2.31 | 74.29 | 74.34 | 73.95 | 74.72 | 50,000 | PASS |  |
| 6both | svd | [0, 1, 2] | [0, 1, 2] | 77.34 | 5.87 | 71.47 | 71.57 | 71.16 | 71.97 | 50,000 | PASS |  |
| 6both | svdllm | [0, 1, 2] | [0, 1, 2] | 77.34 | 78.02 | -0.6840 | -0.6160 | -0.8180 | -0.4160 | 50,000 | FAIL |  |
| 6both | asvd_scale | [0, 1, 2] | [0, 1, 2] | 77.34 | 74.41 | 2.92 | 2.98 | 2.71 | 3.24 | 50,000 | PASS |  |
| 6both | slicegpt | [0, 1, 2] | [0, 1, 2] | 77.34 | 75.60 | 1.74 | 1.69 | 1.44 | 1.94 | 50,000 | FAIL |  |
| 6both | flap | [0, 1, 2] | [0, 1, 2] | 77.34 | 70.29 | 7.05 | 7.25 | 6.93 | 7.58 | 50,000 | PASS |  |
| 6both | zipit | [0, 1, 2] | [0, 1, 2] | 77.34 | 18.51 | 58.83 | 57.93 | 57.47 | 58.39 | 50,000 | PASS | PASS |
| 6both | pair_actcorr | [0, 1, 2] | [0, 1, 2] | 77.34 | 31.31 | 46.03 | 45.49 | 45.01 | 45.95 | 50,000 | PASS | PASS |
| 6both | pair_wcos | [0, 1, 2] | [0, 1, 2] | 77.34 | 39.85 | 37.48 | 37.35 | 36.89 | 37.81 | 50,000 | PASS |  |
| 6both | pair_random | [0, 1, 2] | [0, 1, 2] | 77.34 | 32.30 | 45.04 | 42.80 | 42.33 | 43.27 | 50,000 | PASS |  |
| 6both | random_structure | [0, 1, 2] | [0, 1, 2] | 77.34 | 2.29 | 75.05 | 74.67 | 74.28 | 75.06 | 50,000 | PASS |  |

### MUST-08 P8.1: PASS

- Prediction: MLP-only saving is more than half of the MLP-plus-attention saving, at 6 blocks and at the full model.
- Scored quantity: saving = original model parameters minus Mirror-exact model parameters (mean over fit seeds); ratio MLP-only saving / MLP-plus-attention saving.
- Bar: > 0.5 at 6 blocks and at the full model. Value: 6 blocks: 0.6713; full model: 0.6835. n: original model parameters from the original row.
- Sources: `results/MUST-08/rows_dinov2_vitb14_reg.csv` (805c231c9c); pre-registration commit e76fb17c3c.

| span | saving_mlp_only | saving_mlp_attn | ratio | seeds | P8.1 |
|---|---|---|---|---|---|
| 6 blocks | 11284334.67 | 16810744.00 | 0.6713 | [0, 1, 2] / [0, 1, 2] | PASS |
| full model | 13949550.67 | 20410212.00 | 0.6835 | [0, 1, 2] / [0, 1, 2] | PASS |

### MUST-08 P8.2: PASS

- Prediction: k-NN(MLP only) minus k-NN(MLP plus attention) is at most 0.5 at 6 blocks (mean over fit seeds).
- Scored quantity: mean over fit seeds 0, 1, 2 of validation k-NN, Mirror-exact 6mlp minus 6both.
- Bar: <= 0.5 points. Value: -0.7360. n: 3 fit seeds.
- Sources: `results/MUST-08/rows_dinov2_vitb14_reg.csv` (805c231c9c); pre-registration commit e76fb17c3c.

| size | seed0 | seed1 | seed2 |
|---|---|---|---|
| 6mlp | 76.59 | 76.60 | 76.62 |
| 6attn | 77.40 | 77.44 | 77.42 |
| 6both | 77.41 | 77.26 | 77.35 |
| fullmlp | 52.53 | 52.57 | 52.56 |
| fullattn | 62.67 | 62.14 | 62.40 |
| full | 53.88 | 53.42 | 54.13 |

### MUST-09 P9.1: PASS

- Prediction: Mirror-exact is non-inferior to the best baseline: mean over seeds of Mirror-exact minus mean of the best baseline is above -0.5 points (validation k-NN).
- Scored quantity: mean over healing seeds 0, 1, 2 of final validation k-NN (bf16, 50,000 queries, step 780), Mirror-exact minus the gate-B best baseline (flip-symmetrised targets).
- Bar: > -0.5 points. Value: 0.2793. n: 3 + 3 seeds (best baseline: svdllm).
- Sources: `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0/metrics.json` (52e29d1676); `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s1/metrics.json` (e84e20ff35); `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s2/metrics.json` (e84e20ff35); `runs/MUST-09/dinov2_vitb14_reg/random_structure/s0/metrics.json` (52e29d1676); `runs/MUST-09/dinov2_vitb14_reg/slicegpt/s0/metrics.json` (e84e20ff35); `runs/MUST-09/dinov2_vitb14_reg/svd/s0/metrics.json` (e84e20ff35) and 3 more; pre-registration commit e76fb17c3c.

| method | targets | seed | knn_top1 | n_query | flip_bf16 | flip_fp32 | n_query_fp32 | n_disagree_fp32 | invariance | knn_top1_fp32 | lr | params_frac | params_model |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mirror_exact | sym | 0 | 78.32 | 50,000 | 99.74 | 100.00 | 50,000 | 0 | PASS | 78.37 | 1.00e-04 | 0.7646 | 66,203,360 |
| mirror_exact | sym | 1 | 78.35 | 50,000 | 99.71 | 100.00 | 50,000 | 0 | PASS | 78.34 | 1.00e-04 | 0.7646 | 66,203,360 |
| mirror_exact | sym | 2 | 78.35 | 50,000 | 99.72 | 100.00 | 50,000 | 0 | PASS | 78.32 | 1.00e-04 | 0.7646 | 66,203,360 |
| random_structure | sym | 0 | 69.93 | 50,000 | 99.34 | 100.00 | 50,000 | 0 | PASS | 69.96 | 1.00e-04 | 0.7646 | 66,203,360 |
| slicegpt | sym | 0 | 77.89 | 50,000 | 95.40 | 95.43 | 50,000 | 2284 | PENDING | 77.92 | 1.00e-04 | 0.7660 | 66,324,696 |
| svd | sym | 0 | 77.02 | 50,000 | 94.23 | 94.30 | 50,000 | 2852 | PENDING | 76.99 | 1.00e-04 | 0.7646 | 66,203,904 |
| svdllm | sym | 0 | 78.05 | 50,000 | 95.81 | 95.80 | 50,000 | 2098 | PENDING | 78.07 | 1.00e-04 | 0.7646 | 66,203,904 |
| svdllm | sym | 1 | 78.07 | 50,000 | 95.64 | 95.71 | 50,000 | 2147 | PENDING | 78.07 | 1.00e-04 | 0.7646 | 66,203,904 |
| svdllm | sym | 2 | 78.06 | 50,000 | 95.74 | 95.73 | 50,000 | 2134 | PENDING | 78.08 | 1.00e-04 | 0.7646 | 66,203,904 |

### MUST-09 P9.2: PASS

- Prediction: Only Mirror-exact has 100% flip consistency (scored as: Mirror-exact and the random-structure control reach 100% (fp32, rounding allowance); no compressor (best, second-best, plain SVD) does).
- Scored quantity: fp32 k-NN flip consistency on all 50,000 validation images, every heal; the rounding allowance applies to the exact-invariance side; a compressor fails the prediction if its raw fp32 share is 100%.
- Bar: Mirror-exact and random structure 100%; compressors < 100%. Value: mirror_exact s0: 100.000%; mirror_exact s1: 100.000%; mirror_exact s2: 100.000%; random_structure s0: 100.000%; slicegpt s0: 95.432%; svd s0: 94.296%; svdllm s0: 95.804%; svdllm s1: 95.706%; svdllm s2: 95.732%. n: fp32 pass per heal (50,000 queries each).
- Sources: `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0/metrics.json` (52e29d1676); `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s1/metrics.json` (e84e20ff35); `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s2/metrics.json` (e84e20ff35); `runs/MUST-09/dinov2_vitb14_reg/random_structure/s0/metrics.json` (52e29d1676); `runs/MUST-09/dinov2_vitb14_reg/slicegpt/s0/metrics.json` (e84e20ff35); `runs/MUST-09/dinov2_vitb14_reg/svd/s0/metrics.json` (e84e20ff35) and 3 more; pre-registration commit e76fb17c3c.
- Notes: mirror_exact s0: 100% raw (no disagreement); mirror_exact s1: 100% raw (no disagreement); mirror_exact s2: 100% raw (no disagreement); random_structure s0: 100% raw (no disagreement); svdllm s0: fp32 95.804%; svdllm s1: fp32 95.706%; svdllm s2: fp32 95.732%; slicegpt s0: fp32 95.432%; svd s0: fp32 94.296%.

| method | targets | seed | knn_top1 | n_query | flip_bf16 | flip_fp32 | n_query_fp32 | n_disagree_fp32 | invariance | knn_top1_fp32 | lr | params_frac | params_model |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mirror_exact | sym | 0 | 78.32 | 50,000 | 99.74 | 100.00 | 50,000 | 0 | PASS | 78.37 | 1.00e-04 | 0.7646 | 66,203,360 |
| mirror_exact | sym | 1 | 78.35 | 50,000 | 99.71 | 100.00 | 50,000 | 0 | PASS | 78.34 | 1.00e-04 | 0.7646 | 66,203,360 |
| mirror_exact | sym | 2 | 78.35 | 50,000 | 99.72 | 100.00 | 50,000 | 0 | PASS | 78.32 | 1.00e-04 | 0.7646 | 66,203,360 |
| random_structure | sym | 0 | 69.93 | 50,000 | 99.34 | 100.00 | 50,000 | 0 | PASS | 69.96 | 1.00e-04 | 0.7646 | 66,203,360 |
| slicegpt | sym | 0 | 77.89 | 50,000 | 95.40 | 95.43 | 50,000 | 2284 | PENDING | 77.92 | 1.00e-04 | 0.7660 | 66,324,696 |
| svd | sym | 0 | 77.02 | 50,000 | 94.23 | 94.30 | 50,000 | 2852 | PENDING | 76.99 | 1.00e-04 | 0.7646 | 66,203,904 |
| svdllm | sym | 0 | 78.05 | 50,000 | 95.81 | 95.80 | 50,000 | 2098 | PENDING | 78.07 | 1.00e-04 | 0.7646 | 66,203,904 |
| svdllm | sym | 1 | 78.07 | 50,000 | 95.64 | 95.71 | 50,000 | 2147 | PENDING | 78.07 | 1.00e-04 | 0.7646 | 66,203,904 |
| svdllm | sym | 2 | 78.06 | 50,000 | 95.74 | 95.73 | 50,000 | 2134 | PENDING | 78.08 | 1.00e-04 | 0.7646 | 66,203,904 |

### MUST-09 P9.3: PASS

- Prediction: Random structure trails Mirror-exact by at least 3 points.
- Scored quantity: mean over seeds 0, 1, 2 of Mirror-exact minus random structure (seed 0), final validation k-NN.
- Bar: >= 3.0 points. Value: 8.41. n: 3 seeds against 1.
- Sources: `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0/metrics.json` (52e29d1676); `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s1/metrics.json` (e84e20ff35); `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s2/metrics.json` (e84e20ff35); `runs/MUST-09/dinov2_vitb14_reg/random_structure/s0/metrics.json` (52e29d1676); `runs/MUST-09/dinov2_vitb14_reg/slicegpt/s0/metrics.json` (e84e20ff35); `runs/MUST-09/dinov2_vitb14_reg/svd/s0/metrics.json` (e84e20ff35) and 3 more; pre-registration commit e76fb17c3c.

| method | targets | seed | knn_top1 | n_query | flip_bf16 | flip_fp32 | n_query_fp32 | n_disagree_fp32 | invariance | knn_top1_fp32 | lr | params_frac | params_model |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mirror_exact | sym | 0 | 78.32 | 50,000 | 99.74 | 100.00 | 50,000 | 0 | PASS | 78.37 | 1.00e-04 | 0.7646 | 66,203,360 |
| mirror_exact | sym | 1 | 78.35 | 50,000 | 99.71 | 100.00 | 50,000 | 0 | PASS | 78.34 | 1.00e-04 | 0.7646 | 66,203,360 |
| mirror_exact | sym | 2 | 78.35 | 50,000 | 99.72 | 100.00 | 50,000 | 0 | PASS | 78.32 | 1.00e-04 | 0.7646 | 66,203,360 |
| random_structure | sym | 0 | 69.93 | 50,000 | 99.34 | 100.00 | 50,000 | 0 | PASS | 69.96 | 1.00e-04 | 0.7646 | 66,203,360 |
| slicegpt | sym | 0 | 77.89 | 50,000 | 95.40 | 95.43 | 50,000 | 2284 | PENDING | 77.92 | 1.00e-04 | 0.7660 | 66,324,696 |
| svd | sym | 0 | 77.02 | 50,000 | 94.23 | 94.30 | 50,000 | 2852 | PENDING | 76.99 | 1.00e-04 | 0.7646 | 66,203,904 |
| svdllm | sym | 0 | 78.05 | 50,000 | 95.81 | 95.80 | 50,000 | 2098 | PENDING | 78.07 | 1.00e-04 | 0.7646 | 66,203,904 |
| svdllm | sym | 1 | 78.07 | 50,000 | 95.64 | 95.71 | 50,000 | 2147 | PENDING | 78.07 | 1.00e-04 | 0.7646 | 66,203,904 |
| svdllm | sym | 2 | 78.06 | 50,000 | 95.74 | 95.73 | 50,000 | 2134 | PENDING | 78.08 | 1.00e-04 | 0.7646 | 66,203,904 |

### MUST-10 P10.1: PASS

- Prediction: Healed Mirror-exact (mean of 3 seeds) is within 1 point of flip averaging on validation k-NN, at under 45% of its FLOPs (analytic, one pass against two).
- Scored quantity: |mean over seeds 0 to 2 of healed Mirror-exact k-NN (MUST-09) minus flip-averaged original k-NN (MUST-10)|, and healed GFLOPs per image / flip-averaging GFLOPs per image.
- Bar: |difference| <= 1.0 and FLOPs ratio < 0.45. Value: difference -0.66 points; FLOPs ratio 0.387. n: 3 healed seeds against 1 flip-averaged original.
- Sources: `results/MUST-10/flipavg.csv` (e84e20ff35); `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0/metrics.json` (52e29d1676); `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s1/metrics.json` (e84e20ff35); `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s2/metrics.json` (e84e20ff35); pre-registration commit e76fb17c3c.

| experiment | model | variant | mode | forwards_per_image | knn_top1 | knn_amp | flip_consistency | knn_top1_fp32 | flip_consistency_fp32 | feature_flip_max_abs_unit_fp32 | feature_identical_pct_fp32 | probe_top1 | probe_top5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MUST-10 | dinov2_vitb14_reg | original | plain | 1 | 78.95 | bf16 | 96.28 | n/a | n/a | n/a | n/a | 81.00 | 95.97 |
| MUST-10 | dinov2_vitb14_reg | original | flipavg | 2 | 79.00 | bf16 | 100.00 | 79.01 | 100.00 | 0 | 100.00 | 81.13 | 95.99 |
| MUST-10 | dinov2_vitb14_reg | healed | plain | 1 | 78.32 | bf16 | 99.74 | 78.37 | 100.00 | 2.32e-06 | 0 | 80.10 | 95.56 |

### MUST-11 P11.1: PASS

- Prediction: Healed Mirror-exact is within 1.0 point of the original on the ImageNet linear probe (mean of 3 seeds) and on ADE20K mIoU (seed 0).
- Scored quantity: |mean of vexact_s0..s2 linear-probe top-1 minus the original's| and |vexact_s0 mIoU minus the original's|.
- Bar: both <= 1.0. Value: probe -0.89; ADE20K -0.43. n: linear probe 3 seeds; ADE20K seed 0.
- Sources: `results/MUST-11/t2/t2_linear.csv` (805c231c9c); `results/MUST-11/t2_seg/t2_ade20k.csv` (805c231c9c); pre-registration commit e76fb17c3c.

### MUST-11 P11.2: PENDING

- Prediction: Mirror-exact with the symmetrised head gives 100% mask equivariance; the original model stays below 100%.
- Scored quantity: share of pixels with mask(mirror x) = mirror(mask x), fp32, ADE20K validation; Mirror-exact (vexact_s0) symmetrised head with the rounding allowance; the original below 100% with every recorded head.
- Bar: Mirror-exact 100%; original < 100%. Value: original: raw 95.0309, sym 94.7413; best_svdllm_s0: raw 94.7590, sym 94.6249; vexact_s0: raw 98.9771, sym 99.9999. n: n/a.
- Sources: `results/MUST-11/t2_seg/t2_ade20k.csv` (805c231c9c); pre-registration commit e76fb17c3c.
- Notes: Mirror-exact symmetrised head: 99.9999% raw, 422 disagreement(s); their top-two margins are not recorded, so the rounding allowance (PREREG Section 1) cannot be applied.

### MUST-12 gate C: FAIL

- Prediction: Gate C: in the LR model, the LR ratio to null in blocks 1 to 6 is at most 0.8 and the LR twin fraction in blocks 1 to 6 is at least 0.2.
- Scored quantity: MUST-12 ViT-S/8 LR run, seed 0, final checkpoint: LR ratio to null and LR twin fraction (t 0.5), blocks 1 to 6.
- Bar: ratio <= 0.8 and twin >= 0.2. Value: ratio 0.4844, twin 0.0601. n: epoch 100.
- Sources: `results/MUST-12/vit_s8_LR_s0/e100/metrics.json` (e84e20ff35); `results/MUST-12/gate_c.json` (not recorded); pre-registration commit e76fb17c3c.

### CORE-01 T1: PENDING

- Prediction: Dissociation: in every seed pair s = 0, 1, 2, the UD twin fraction (blocks 1 to 6, final) of UD run s minus that of LR run s is at least 0.2 (absolute).
- Scored quantity: UD twin fraction (t 0.5), blocks 1 to 6, last checkpoint: UD run s minus LR run s, pairs s = 0, 1, 2.
- Bar: >= 0.2 in every pair. Value: min difference n/a. n: 0 of 3 seed pairs.
- Sources: `results/MUST-12/vit_s8_LR_s0/e100/metrics.json` (e84e20ff35); pre-registration commit e76fb17c3c.

| seed_pair | UD_run_UD_twin | LR_run_UD_twin | difference | verdict | UD_run_epoch | LR_run_epoch |
|---|---|---|---|---|---|---|
| 0 | n/a | 0.1988 | n/a | PENDING | n/a | 100 |
| 1 | n/a | n/a | n/a | PENDING | n/a | n/a |
| 2 | n/a | n/a | n/a | PENDING | n/a | n/a |

### CORE-01 T2: FAIL

- Prediction: Specificity: the UD ratio to null is at most 0.7 in every UD run, and at least 0.9 in every none run and every LR run (blocks 1 to 6, final).
- Scored quantity: UD ratio to null, blocks 1 to 6, last checkpoint.
- Bar: UD runs <= 0.7; none and LR runs >= 0.9. Value: UD s0: n/a; UD s1: n/a; UD s2: n/a; none s0: 0.7026; none s1: n/a; none s2: n/a; LR s0: 0.6856; LR s1: n/a; LR s2: n/a. n: 2 of 9 runs.
- Sources: `results/MUST-12/vit_s8_none_s0/e100/metrics.json` (e84e20ff35); `results/MUST-12/vit_s8_LR_s0/e100/metrics.json` (e84e20ff35); pre-registration commit e76fb17c3c.

| run | UD_ratio_to_null | bar | verdict | epoch |
|---|---|---|---|---|
| UD s0 | n/a | <= 0.7 | PENDING | n/a |
| UD s1 | n/a | <= 0.7 | PENDING | n/a |
| UD s2 | n/a | <= 0.7 | PENDING | n/a |
| none s0 | 0.7026 | >= 0.9 | FAIL | 100 |
| none s1 | n/a | >= 0.9 | PENDING | n/a |
| none s2 | n/a | >= 0.9 | PENDING | n/a |
| LR s0 | 0.6856 | >= 0.9 | FAIL | 100 |
| LR s1 | n/a | >= 0.9 | PENDING | n/a |
| LR s2 | n/a | >= 0.9 | PENDING | n/a |

### CORE-01 T3: PENDING

- Prediction: Rotation: in every R90 run, the share in 4-cycles (blocks 1 to 6, final) is at least 0.3, and at least 3 times the mean share in the none runs.
- Scored quantity: R90 share of hidden units in 4-cycles, blocks 1 to 6, last checkpoint; against 3 x the mean over the three none runs.
- Bar: >= 0.3 and >= 3 x none mean, every R90 run. Value: R90 s0: n/a; R90 s1: n/a; none mean n/a. n: 2 R90 runs, 3 none runs.
- Sources: `results/MUST-12/vit_s8_none_s0/e100/metrics.json` (e84e20ff35); pre-registration commit e76fb17c3c.

| run | share_4cycles | none_mean | ratio_to_none | verdict |
|---|---|---|---|---|
| R90 s0 | n/a | n/a | n/a | PENDING |
| R90 s1 | n/a | n/a | n/a | PENDING |

### CORE-01 T4: FAIL

- Prediction: Emergence: in every LR run (for LR) and every UD run (for UD), the twin fraction of the trained group is higher at the final epoch than at epoch 5.
- Scored quantity: twin fraction (t 0.5) of the trained group, blocks 1 to 6 (the range of T1 to T3), last checkpoint against epoch 5.
- Bar: final > epoch 5, every LR and UD run. Value: LR s0: 0.0846 -> 0.0601; LR s1: n/a -> n/a; LR s2: n/a -> n/a; UD s0: n/a -> n/a; UD s1: n/a -> n/a; UD s2: n/a -> n/a. n: 1 of 6 runs.
- Sources: `results/MUST-12/vit_s8_LR_s0/e100/metrics.json` (e84e20ff35); pre-registration commit e76fb17c3c.

| run | group | epoch5 | final | final_epoch | verdict |
|---|---|---|---|---|---|
| LR s0 | LR | 0.0846 | 0.0601 | 100 | FAIL |
| LR s1 | LR | n/a | n/a | n/a | PENDING |
| LR s2 | LR | n/a | n/a | n/a | PENDING |
| UD s0 | UD | n/a | n/a | n/a | PENDING |
| UD s1 | UD | n/a | n/a | n/a | PENDING |
| UD s2 | UD | n/a | n/a | n/a | PENDING |

### CORE-02 P2.1: PENDING

- Prediction: The achieved saving equals the prediction (identity check).
- Scored quantity: per block 1 to 6 of every scored model (LR+UD, R90) and group: achieved free-parameter counts of fc1 and fc2 (compressed form) equal to the character-formula counts; reconstruction error below 1e-4; prediction timestamp before the surgery timestamp.
- Bar: equal in every block. Value: n/a. n: n/a.
- Sources: n/a; pre-registration commit e76fb17c3c.
- Notes: results/CORE-02/surgery.csv missing (predictions.json missing too).

### CORE-02 P2.2: PENDING

- Prediction: Trained-group surgery on blocks 1 to 6 keeps validation top-1 within 3 points.
- Scored quantity: |top-1 drop| after surgery with the model's training group (Klein for LR+UD, R90), ImageNet-100 validation, every scored model.
- Bar: <= 3.0 points. Value: n/a. n: n/a.
- Sources: n/a; pre-registration commit e76fb17c3c.
- Notes: results/CORE-02/surgery.csv missing (predictions.json missing too).

### CORE-02 P2.3: PENDING

- Prediction: Every wrong-group surgery loses at least 20 points.
- Scored quantity: top-1 drop after surgery with a group that is neither the training group nor one of its subgroups, every scored model.
- Bar: >= 20 points, every row. Value: n/a. n: n/a.
- Sources: n/a; pre-registration commit e76fb17c3c.
- Notes: results/CORE-02/surgery.csv missing (predictions.json missing too).

### CORE-03 P3.1: FAIL

- Prediction: Mirror plus low rank lies above low rank alone (the better of plain SVD and SVD-LLM) at at least 3 of 4 budget points on at least 2 of 3 models, including DINOv2-L or DeiT-III (zero-shot).
- Scored quantity: per model and budget point: validation top-1 of the scored Mirror-plus-low-rank point strictly above the scored low-rank-alone point (the better of plain SVD and SVD-LLM by method-selection k-NN, as curves.csv marks it); zero-shot.
- Bar: >= 3 of 4 points on >= 2 of 3 models, one of them DINOv2-L or DeiT-III. Value: dinov2_vitb14_reg: 0 of 4 points above; dinov2_vitl14_reg: 0 of 4 points above; deit3_b16: 0 of 4 points above. n: 12 budget points.
- Sources: `results/CORE-03/curves.csv` (805c231c9c); pre-registration commit e76fb17c3c.

| model | point | mirror_lowrank | lowrank | lowrank_method | above | model_verdict |
|---|---|---|---|---|---|---|
| dinov2_vitb14_reg | f0.40 | 5.39 | 15.55 | svdllm | no | n/a |
| dinov2_vitb14_reg | f0.55 | 2.69 | 60.98 | svdllm | no | n/a |
| dinov2_vitb14_reg | f0.70 | 6.19 | 74.02 | svdllm | no | n/a |
| dinov2_vitb14_reg | f0.85 | 53.88 | 77.91 | svdllm | no | n/a |
| dinov2_vitl14_reg | f0.40 | 9.69 | 53.77 | svdllm | no | n/a |
| dinov2_vitl14_reg | f0.55 | 9.79 | 74.82 | svdllm | no | n/a |
| dinov2_vitl14_reg | f0.70 | 12.03 | 79.68 | svdllm | no | n/a |
| dinov2_vitl14_reg | f0.85 | 30.17 | 80.61 | svdllm | no | n/a |
| deit3_b16 | f0.40 | 20.58 | 46.68 | svdllm | no | n/a |
| deit3_b16 | f0.55 | 40.80 | 68.70 | svdllm | no | n/a |
| deit3_b16 | f0.70 | 56.86 | 77.43 | svdllm | no | n/a |
| deit3_b16 | f0.85 | 72.85 | 80.83 | svdllm | no | n/a |
| dinov2_vitb14_reg | n/a | n/a | n/a | n/a | n/a | FAIL |
| dinov2_vitl14_reg | n/a | n/a | n/a | n/a | n/a | FAIL |
| deit3_b16 | n/a | n/a | n/a | n/a | n/a | FAIL |

### CORE-04 P4.1: FAIL

- Prediction: Scale: the tax in blocks 1 to L/2 increases from DINOv2-S to g (either direction is reported).
- Scored quantity: LR tax (character formula, MLP), blocks 1 to L/2, DINOv2 S, B, L, g with registers; increases means strictly increasing at every step.
- Bar: S < B < L < g. Value: s14 0.4002, b14 0.4185, l14 0.4140, g14 0.3440. n: 4 of 4 sizes.
- Sources: `results/CORE-04/structure/dinov2_vits14_reg/metrics.json` (e84e20ff35); `results/MUST-04/dinov2_vitb14_reg/metrics.json` (e84e20ff35); `results/MUST-04/dinov2_vitl14_reg/metrics.json` (e84e20ff35); `results/CORE-04/structure/dinov2_vitg14_reg/metrics.json` (e84e20ff35); pre-registration commit e76fb17c3c.
- Notes: mixed direction (reported either way, PREREG Section 6).

| model | tax_first_half |
|---|---|
| dinov2_vits14_reg | 0.4002 |
| dinov2_vitb14_reg | 0.4185 |
| dinov2_vitl14_reg | 0.4140 |
| dinov2_vitg14_reg | 0.3440 |

### CORE-04 P4.2: FAIL

- Prediction: Models whose recipe is verified to have no flips have a lower LR tax than every flip-trained model of the same size.
- Scored quantity: LR tax, blocks 1 to L/2; verified no-flip: recipe_flip 'no' without a 'probable' or '(verify)' confidence in configs/models; same size: same width and depth.
- Bar: below every flip-trained model of its size. Value: clip_b16: 0.3125; convnext_b: 0.2168; deit3_b16: 0.3487; dino_b16: 0.2495; dinov2_vitb14: 0.4048; dinov2_vitb14_reg: 0.4185; dinov2_vitg14_reg: 0.3440; dinov2_vitl14_reg: 0.4140; dinov2_vits14_reg: 0.4002; dinov3_b16: 0.4002; eva02_b14: 0.1647; mae_b16: 0.4482; resnet50: 0.2794; siglip2_b16: 0.2309. n: 1 verified no-flip model(s): clip_b16.
- Sources: `results/CORE-04/structure/clip_b16/metrics.json` (not recorded); `results/MUST-04/deit3_b16/metrics.json` (not recorded); `results/CORE-04/structure/dino_b16/metrics.json` (not recorded); `results/MUST-04/dinov2_vitb14/metrics.json` (not recorded); `results/MUST-04/dinov2_vitb14_reg/metrics.json` (not recorded); `results/CORE-04/structure/eva02_b14/metrics.json` (not recorded) and 1 more; pre-registration commit e76fb17c3c.

| model | recipe_flip | verified_no_flip | size | tax_first_half |
|---|---|---|---|---|
| clip_b16 | no | yes | 768 x 12 | 0.3125 |
| convnext_b | yes | no | stage widths 128, 256, 512, 1024 x 36 | 0.2168 |
| deit3_b16 | yes | no | 768 x 12 | 0.3487 |
| dino_b16 | yes | no | 768 x 12 | 0.2495 |
| dinov2_vitb14 | yes | no | 768 x 12 | 0.4048 |
| dinov2_vitb14_reg | yes | no | 768 x 12 | 0.4185 |
| dinov2_vitg14_reg | yes | no | 1536 x 40 | 0.3440 |
| dinov2_vitl14_reg | yes | no | 1024 x 24 | 0.4140 |
| dinov2_vits14_reg | yes | no | 384 x 12 | 0.4002 |
| dinov3_b16 | no | no | 768 x 12 | 0.4002 |
| eva02_b14 | yes | no | 768 x 12 | 0.1647 |
| mae_b16 | yes | no | 768 x 12 | 0.4482 |
| resnet50 | yes | no | stage output channels 256, 512, 1024, 2048 (bottleneck widths 64, 128, 256, 512; expansion 4) x 16 | 0.2794 |
| siglip2_b16 | unknown | no | 768 x 12 | 0.2309 |

### CORE-04 P4.3: PASS

- Prediction: CNNs: twin structure appears in ConvNeXt's pointwise MLPs (LR ratio to null at most 0.8 and twin fraction at least 0.2 in the first half of the stages' pointwise MLPs).
- Scored quantity: LR ratio to null and LR twin fraction (t 0.5), first half of ConvNeXt-B's pointwise MLPs.
- Bar: ratio <= 0.8 and twin >= 0.2. Value: ratio 0.5034, twin 0.2319. n: n/a.
- Sources: `results/CORE-04/structure/convnext_b/metrics.json` (e84e20ff35); pre-registration commit e76fb17c3c.

### CORE-05 P5.1: FAIL

- Prediction: The activation-weighted breaking share is at most half the weight-energy breaking share in blocks 1 to 6.
- Scored quantity: dinov2_vitb14_reg, operator A*_L: mean of s_cov over fc1 and fc2 of blocks 1 to 6 over the mean of s_w (other models reported).
- Bar: <= 0.5. Value: 0.5803. n: 12 matrices.
- Sources: `results/CORE-05/measures_dinov2_vitb14_reg.csv` (805c231c9c); pre-registration commit e76fb17c3c.

| model | s_w | s_cov | ratio | n_matrices | verdict |
|---|---|---|---|---|---|
| dinov2_vitb14_reg | 0.2536 | 0.1472 | 0.5803 | 12 | FAIL |
| dinov2_vitl14_reg | 0.2302 | 0.1236 | 0.5369 | 12 | FAIL |
| deit3_b16 | 0.2067 | 0.1413 | 0.6834 | 12 | FAIL |
| mae_b16 | 0.2672 | 0.1475 | 0.5521 | 12 | FAIL |

### CORE-05 P5.2: PENDING

- Prediction: Over training (CORE-01 LR and UD runs), breaking energy in the top-variance band falls faster than in the bottom band.
- Scored quantity: per run, within-band breaking share (mean over fc1 and fc2 of blocks 1 to 6): fall from the first to the last checkpoint, top band against bottom band; every LR and UD run.
- Bar: top fall > bottom fall. Value: n/a. n: n/a.
- Sources: n/a; pre-registration commit e76fb17c3c.
- Notes: results/CORE-05/ckpt_bands.csv missing.

### CORE-05 P5.3: FAIL

- Prediction: The out-of-sample rule is correct on at least 75% of layer-model cases on DINOv2-L-reg, DeiT-III and MAE.
- Scored quantity: share of correct cases (every block of the three models, pooled) of the frozen rule against the validation ground truth.
- Bar: >= 0.75. Value: 0.6667. n: 48 layer-model cases.
- Sources: `results/CORE-05/rule_score.json` (805c231c9c); pre-registration commit e76fb17c3c.
- Notes: rule frozen 2026-11-15T16:36:17Z, scored 2026-11-15T20:21:19Z.

### CORE-06 P6.1: PASS

- Prediction: Species from the even part is within 1 point of the full feature.
- Scored quantity: |species top-1 of the full CLS minus the even CLS|, natural CUB test images, original and every healed Mirror-exact seed.
- Bar: <= 1.0 points, every model. Value: original: 90.01 vs 89.96; vexact_s0: 89.42 vs 89.45; vexact_s1: 89.47 vs 89.52; vexact_s2: 89.37 vs 89.44. n: 4 model(s).
- Sources: `results/CORE-06/probes.csv` (805c231c9c); pre-registration commit e76fb17c3c.

| variant | species_even | species_full | verdict |
|---|---|---|---|
| original | 89.96 | 90.01 | PASS |
| vexact_s0 | 89.45 | 89.42 | PASS |
| vexact_s1 | 89.52 | 89.47 | PASS |
| vexact_s2 | 89.44 | 89.37 | PASS |

### CORE-06 P6.2: PASS

- Prediction: Facing direction from the odd part is at least 85%.
- Scored quantity: facing accuracy from the odd part (CLS, or mean patch token when the odd CLS's training-split CV accuracy is below 85%), natural test images, every model probed.
- Bar: >= 85%, every model. Value: original: 94.99; vexact_s0: 99.19; vexact_s1: 99.21; vexact_s2: 99.25. n: 4 model(s).
- Sources: `results/CORE-06/probes.csv` (805c231c9c); pre-registration commit e76fb17c3c.

| variant | feature | odd_cls_cv_train | facing_odd | verdict |
|---|---|---|---|---|
| original | odd_cls | 95.68 | 94.99 | PASS |
| vexact_s0 | odd_cls | 99.14 | 99.19 | PASS |
| vexact_s1 | odd_cls | 99.12 | 99.21 | PASS |
| vexact_s2 | odd_cls | 99.08 | 99.25 | PASS |

### CORE-06 P6.3: PASS

- Prediction: Facing direction from healed Mirror-exact's even part is at most 60% on natural test images, and exactly 50% on the mirror-augmented test set.
- Scored quantity: facing accuracy from the even CLS of every healed Mirror-exact seed; mirror-augmented: 50% with every differing pair a near-tie (relative logit margin < 1e-4).
- Bar: <= 60% and = 50%. Value: vexact_s0: 50.84 / 50.00; vexact_s1: 50.29 / 50.00; vexact_s2: 50.57 / 50.00. n: 3 model(s).
- Sources: `results/CORE-06/probes.csv` (805c231c9c); pre-registration commit e76fb17c3c.

| variant | facing_even_natural | facing_even_mirror_augmented | pairs_differ_not_near_tie | verdict |
|---|---|---|---|---|
| vexact_s0 | 50.84 | 50.00 | 0 | PASS |
| vexact_s1 | 50.29 | 50.00 | 0 | PASS |
| vexact_s2 | 50.57 | 50.00 | 0 | PASS |

### CORE-07 P7.1: PASS

- Prediction: Mirror-exact's drop from ImageNet to ImageNet-V2 (matched frequency, linear probe) is within 1 point of the original's drop.
- Scored quantity: |mean over healed Mirror-exact seeds of (ImageNet top-1 - ImageNet-V2 top-1) minus the original's|, same linear probe.
- Bar: <= 1.0 points. Value: drops: Mirror-exact 9.17 (mean of 3), original 8.62. n: 3 Mirror-exact seed(s).
- Sources: `results/CORE-07/shift_rows.csv` (805c231c9c); pre-registration commit e76fb17c3c.

| model | variant | role | val_top1 | val_top1_recorded_by_probe | n_val | v2_top1 | v2_top5 | n_v2 | v2_drop | sketch_top1 | sketch_top5 | n_sketch | sketch_drop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov2_vitb14_reg | original | original | 81.00 | 81.00 | 50,000 | 72.38 | 91.50 | 10,000 | 8.62 | 52.77 | 75.47 | 50,889 | 28.23 |
| dinov2_vitb14_reg | vexact_s0 | mirror_exact | 80.10 | 80.10 | 50,000 | 70.95 | 90.63 | 10,000 | 9.15 | 49.31 | 71.32 | 50,889 | 30.79 |
| dinov2_vitb14_reg | vexact_s1 | mirror_exact | 80.04 | 80.04 | 50,000 | 70.90 | 90.67 | 10,000 | 9.14 | 49.45 | 71.50 | 50,889 | 30.59 |
| dinov2_vitb14_reg | vexact_s2 | mirror_exact | 80.19 | 80.19 | 50,000 | 70.97 | 90.61 | 10,000 | 9.22 | 49.39 | 71.65 | 50,889 | 30.80 |
| dinov2_vitb14_reg | best_svdllm_s0 | best_baseline | 79.73 | 79.73 | 50,000 | 70.31 | 90.22 | 10,000 | 9.42 | 45.17 | 66.95 | 50,889 | 34.56 |

### UP-01: PENDING

- Prediction: UD tax rises with p (0.1, 0.25, 0.5).
- Scored quantity: reported as measured, not scored (PREREG Section 4).
- Bar: none (reported). Value: n/a. n: n/a.
- Sources: n/a; pre-registration commit e76fb17c3c.
- Notes: results/UP-01/ has no finished output.

| p | run | UD_tax_1-6 | epoch |
|---|---|---|---|
| 0.1000 | vit_s8_UD_p0.1_s0 | n/a | n/a |
| 0.2500 | vit_s8_UD_p0.25_s0 | n/a | n/a |
| 0.5000 | CORE-01 UD runs (mean) | n/a | n/a |

### UP-02: PENDING

- Prediction: Breaking share stays below 0.05 throughout training from a UD-equivariant start, at accuracy within 1 point of the UD runs.
- Scored quantity: reported as measured, not scored (PREREG Section 4).
- Bar: none (reported). Value: n/a. n: n/a.
- Sources: n/a; pre-registration commit e76fb17c3c.
- Notes: results/UP-02/ has no finished output.

### UP-03: PENDING

- Prediction: The trained-group tax differs between ViT-S and ViT-Ti; either direction is reported.
- Scored quantity: reported as measured, not scored (PREREG Section 4).
- Bar: none (reported). Value: n/a. n: n/a.
- Sources: n/a; pre-registration commit e76fb17c3c.
- Notes: results/UP-03/ has no finished output.

| condition | group | ViT-Ti_tax_1-6 | ViT-S_tax_1-6_mean |
|---|---|---|---|
| none | LR | n/a | 0.1562 |
| LR | LR | n/a | 0.1446 |
| UD | UD | n/a | n/a |

### UP-04: PENDING

- Prediction: Tied-from-scratch reaches about the same accuracy at the reduced parameter count.
- Scored quantity: reported as measured, not scored (PREREG Section 4).
- Bar: none (reported). Value: n/a. n: n/a.
- Sources: n/a; pre-registration commit e76fb17c3c.
- Notes: results/UP-04/ has no finished output.

### UP-05: PENDING

- Prediction: A speed claim needs more than 1.1x the original's speed; otherwise it is a limitation.
- Scored quantity: reported as measured, not scored (PREREG Section 4).
- Bar: none (reported). Value: n/a. n: n/a.
- Sources: n/a; pre-registration commit e76fb17c3c.
- Notes: results/UP-05/ has no finished output (no script yet).

### UP-06: REPORTED

- Prediction: The extra drop from INT8 weight-only quantisation and from ToMe (r = 8) is the same for both models within 0.5 points.
- Scored quantity: reported as measured, not scored (PREREG Section 4).
- Bar: none (reported). Value: see details. n: n/a.
- Sources: `results/UP-06/comparison.json` (e84e20ff35); pre-registration commit e76fb17c3c.

| bar_points | reading | prediction_holds |
|---|---|---|
| 0.5000 | point estimate \|drop(mirror_exact) - drop(original)\| <= bar | yes |

### UP-07: REPORTED

- Prediction: Pairing from weights alone agrees with the data-driven pairing on at least 70% of pairs.
- Scored quantity: reported as measured, not scored (PREREG Section 4).
- Bar: none (reported). Value: see details. n: n/a.
- Sources: `results/UP-07/up07.json` (e84e20ff35); pre-registration commit e76fb17c3c.

| tiny | verdict.bar | verdict.scored_block | reproduction.data_fc1 | reproduction.weights_fc1 | reproduction.stored_data_fc1 | reproduction.stored_weights_fc1 | reproduction.stored_file | reproduction.same_operators_file | provenance.model | provenance.fit_run | provenance.npz_sha256 | provenance.fit_seed | provenance.seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no | 0.7000 | 1 | 0.4658 | 0.4409 | 0.4658 | 0.4409 | runs/d/d_datafree.json | yes | dinov2_vitb14_reg | /home/ubuntu/Mirror_CPAL/runs/x2/perlayer_reg_s0/results/e1_operators.npz | 9ca69899622e2d2f9af4c7be180b9bdd1c8567ac9520622d1af36932b43965f6 | 0 | 0 |

## Numbers the paper will cite (new phase)

Key, value and source file (with the commit recorded by the run). Values are unrounded in PAPER_NUMBERS.json.

| Key | Value | Unit | n | Description | Source |
|---|---|---|---|---|---|
| core04.dinov2_vitb14_reg.LR.tax_first_half | 0.4185 |  |  | CORE-04: LR tax, blocks 1 to L/2, dinov2_vitb14_reg | `results/MUST-04/dinov2_vitb14_reg/metrics.json` (e84e20ff35) |
| core04.dinov2_vitb14_reg.original_knn | 78.95 | % |  | CORE-04: original k-NN top-1, dinov2_vitb14_reg (paper protocol) | `results/CORE-04/scale/knn.csv` (e84e20ff35) |
| core04.dinov2_vitg14_reg.LR.tax_first_half | 0.3440 |  |  | CORE-04: LR tax, blocks 1 to L/2, dinov2_vitg14_reg | `results/CORE-04/structure/dinov2_vitg14_reg/metrics.json` (e84e20ff35) |
| core04.dinov2_vitg14_reg.original_knn | 80.35 | % |  | CORE-04: original k-NN top-1, dinov2_vitg14_reg (paper protocol) | `results/CORE-04/scale/knn.csv` (e84e20ff35) |
| core04.dinov2_vitl14_reg.LR.tax_first_half | 0.4140 |  |  | CORE-04: LR tax, blocks 1 to L/2, dinov2_vitl14_reg | `results/MUST-04/dinov2_vitl14_reg/metrics.json` (e84e20ff35) |
| core04.dinov2_vitl14_reg.original_knn | 80.58 | % |  | CORE-04: original k-NN top-1, dinov2_vitl14_reg (paper protocol) | `results/CORE-04/scale/knn.csv` (e84e20ff35) |
| core04.dinov2_vits14_reg.LR.tax_first_half | 0.4002 |  |  | CORE-04: LR tax, blocks 1 to L/2, dinov2_vits14_reg | `results/CORE-04/structure/dinov2_vits14_reg/metrics.json` (e84e20ff35) |
| core04.dinov2_vits14_reg.original_knn | 74.47 | % |  | CORE-04: original k-NN top-1, dinov2_vits14_reg (paper protocol) | `results/CORE-04/scale/knn.csv` (e84e20ff35) |
| core05.deit3_b16.act_over_weight_1-6 | 0.6834 |  | 12 | CORE-05: activation-weighted over weight-energy breaking share (A*_L), blocks 1 to 6, deit3_b16 | `results/CORE-05/measures_deit3_b16.csv` (805c231c9c) |
| core05.dinov2_vitb14_reg.act_over_weight_1-6 | 0.5803 |  | 12 | CORE-05: activation-weighted over weight-energy breaking share (A*_L), blocks 1 to 6, dinov2_vitb14_reg | `results/CORE-05/measures_dinov2_vitb14_reg.csv` (805c231c9c) |
| core05.dinov2_vitl14_reg.act_over_weight_1-6 | 0.5369 |  | 12 | CORE-05: activation-weighted over weight-energy breaking share (A*_L), blocks 1 to 6, dinov2_vitl14_reg | `results/CORE-05/measures_dinov2_vitl14_reg.csv` (805c231c9c) |
| core05.mae_b16.act_over_weight_1-6 | 0.5521 |  | 12 | CORE-05: activation-weighted over weight-energy breaking share (A*_L), blocks 1 to 6, mae_b16 | `results/CORE-05/measures_mae_b16.csv` (805c231c9c) |
| core05.p53.share_correct | 0.6667 |  | 48 | CORE-05 P5.3: share of correct layer-model cases (frozen rule) | `results/CORE-05/rule_score.json` (805c231c9c) |
| core07.drop_difference | 0.5473 | points | 3 | CORE-07: Mirror-exact minus original drop to ImageNet-V2 | `results/CORE-07/shift_rows.csv` (805c231c9c) |
| gate_b.best | svdllm |  |  | Gate B: best baseline (method-selection k-NN, full model) | `results/MUST-07/gate_b.json` (805c231c9c) |
| gate_b.second_best | slicegpt |  |  | Gate B: second-best baseline | `results/MUST-07/gate_b.json` (805c231c9c) |
| must01.max_abs_diff_pp | 0 | percentage points | 144 | MUST-01: largest \|predicted - achieved\| saving, any layer, any model | `results/MUST-01/law_per_layer.csv` (52e29d1676) |
| must01.n_blocks | 144 |  |  | MUST-01: blocks compared (all models and fit seeds) | `results/MUST-01/law_per_layer.csv` (52e29d1676) |
| must02.zeroshot.flip_bf16 | 98.05 | % |  | MUST-02: k-NN flip consistency, bf16 (reported alongside) | `results/MUST-02/dinov2_vitb14_reg/zeroshot/metrics.json` (e84e20ff35) |
| must02.zeroshot.flip_fp32 | 100.00 | % | 50,000 | MUST-02: k-NN flip consistency, fp32 features | `results/MUST-02/dinov2_vitb14_reg/zeroshot/metrics.json` (e84e20ff35) |
| must02.zeroshot.max_rel_tensor | 2.51e-05 |  | 1000 | MUST-02: largest relative equivariance error, any component, zero-shot Mirror-exact, fp32 | `results/MUST-02/dinov2_vitb14_reg/zeroshot/metrics.json` (e84e20ff35) |
| must03.block1.n_pairs | 1168 |  |  | MUST-03: block 1 LR twin pairs, new code (reference 1168) | `results/MUST-04/dinov2_vitb14_reg/metrics.json` (e84e20ff35) |
| must04.deit3_b16.LR.ratio_to_null_1-6 | 0.4501 |  |  | MUST-04: LR ratio to null, blocks 1 to 6, deit3_b16 | `results/MUST-04/deit3_b16/metrics.json` (e84e20ff35) |
| must04.deit3_b16.LR.surgery_drop | 1.39 | points | 50,000 | MUST-04: k-NN drop after LR surgery, blocks 1 to 6, deit3_b16 | `results/MUST-04/deit3_b16/metrics.json` (e84e20ff35) |
| must04.deit3_b16.LR.tax_1-6 | 0.3487 |  |  | MUST-04: LR tax (character formula), MLP, blocks 1 to 6, deit3_b16 | `results/MUST-04/deit3_b16/metrics.json` (e84e20ff35) |
| must04.deit3_b16.R180.ratio_to_null_1-6 | 0.4807 |  |  | MUST-04: R180 ratio to null, blocks 1 to 6, deit3_b16 | `results/MUST-04/deit3_b16/metrics.json` (e84e20ff35) |
| must04.deit3_b16.R90.ratio_to_null_1-6 | 0.7536 |  |  | MUST-04: R90 ratio to null, blocks 1 to 6, deit3_b16 | `results/MUST-04/deit3_b16/metrics.json` (e84e20ff35) |
| must04.deit3_b16.UD.ratio_to_null_1-6 | 0.4807 |  |  | MUST-04: UD ratio to null, blocks 1 to 6, deit3_b16 | `results/MUST-04/deit3_b16/metrics.json` (e84e20ff35) |
| must04.deit3_b16.UD.surgery_drop | 2.03 | points | 50,000 | MUST-04: k-NN drop after UD surgery, blocks 1 to 6, deit3_b16 | `results/MUST-04/deit3_b16/metrics.json` (e84e20ff35) |
| must04.deit3_b16.UD.twin_t50_1-6 | 0.4630 |  | 20 | MUST-04: UD twin fraction (t 0.5), blocks 1 to 6, deit3_b16 | `results/MUST-04/deit3_b16/metrics.json` (e84e20ff35) |
| must04.dinov2_vitb14.LR.ratio_to_null_1-6 | 0.4474 |  |  | MUST-04: LR ratio to null, blocks 1 to 6, dinov2_vitb14 | `results/MUST-04/dinov2_vitb14/metrics.json` (e84e20ff35) |
| must04.dinov2_vitb14.LR.surgery_drop | 4.01 | points | 50,000 | MUST-04: k-NN drop after LR surgery, blocks 1 to 6, dinov2_vitb14 | `results/MUST-04/dinov2_vitb14/metrics.json` (e84e20ff35) |
| must04.dinov2_vitb14.LR.tax_1-6 | 0.4048 |  |  | MUST-04: LR tax (character formula), MLP, blocks 1 to 6, dinov2_vitb14 | `results/MUST-04/dinov2_vitb14/metrics.json` (e84e20ff35) |
| must04.dinov2_vitb14.R180.ratio_to_null_1-6 | 0.5081 |  |  | MUST-04: R180 ratio to null, blocks 1 to 6, dinov2_vitb14 | `results/MUST-04/dinov2_vitb14/metrics.json` (e84e20ff35) |
| must04.dinov2_vitb14.R90.ratio_to_null_1-6 | 0.8366 |  |  | MUST-04: R90 ratio to null, blocks 1 to 6, dinov2_vitb14 | `results/MUST-04/dinov2_vitb14/metrics.json` (e84e20ff35) |
| must04.dinov2_vitb14.UD.ratio_to_null_1-6 | 0.4956 |  |  | MUST-04: UD ratio to null, blocks 1 to 6, dinov2_vitb14 | `results/MUST-04/dinov2_vitb14/metrics.json` (e84e20ff35) |
| must04.dinov2_vitb14.UD.surgery_drop | 6.08 | points | 50,000 | MUST-04: k-NN drop after UD surgery, blocks 1 to 6, dinov2_vitb14 | `results/MUST-04/dinov2_vitb14/metrics.json` (e84e20ff35) |
| must04.dinov2_vitb14.UD.twin_t50_1-6 | 0.5298 |  | 20 | MUST-04: UD twin fraction (t 0.5), blocks 1 to 6, dinov2_vitb14 | `results/MUST-04/dinov2_vitb14/metrics.json` (e84e20ff35) |
| must04.dinov2_vitb14_reg.LR.ratio_to_null_1-6 | 0.4791 |  |  | MUST-04: LR ratio to null, blocks 1 to 6, dinov2_vitb14_reg | `results/MUST-04/dinov2_vitb14_reg/metrics.json` (e84e20ff35) |
| must04.dinov2_vitb14_reg.LR.surgery_drop | 1.84 | points | 50,000 | MUST-04: k-NN drop after LR surgery, blocks 1 to 6, dinov2_vitb14_reg | `results/MUST-04/dinov2_vitb14_reg/metrics.json` (e84e20ff35) |
| must04.dinov2_vitb14_reg.LR.tax_1-6 | 0.4185 |  |  | MUST-04: LR tax (character formula), MLP, blocks 1 to 6, dinov2_vitb14_reg | `results/MUST-04/dinov2_vitb14_reg/metrics.json` (e84e20ff35) |
| must04.dinov2_vitb14_reg.R180.ratio_to_null_1-6 | 0.5272 |  |  | MUST-04: R180 ratio to null, blocks 1 to 6, dinov2_vitb14_reg | `results/MUST-04/dinov2_vitb14_reg/metrics.json` (e84e20ff35) |
| must04.dinov2_vitb14_reg.R90.ratio_to_null_1-6 | 0.8494 |  |  | MUST-04: R90 ratio to null, blocks 1 to 6, dinov2_vitb14_reg | `results/MUST-04/dinov2_vitb14_reg/metrics.json` (e84e20ff35) |
| must04.dinov2_vitb14_reg.UD.ratio_to_null_1-6 | 0.5129 |  |  | MUST-04: UD ratio to null, blocks 1 to 6, dinov2_vitb14_reg | `results/MUST-04/dinov2_vitb14_reg/metrics.json` (e84e20ff35) |
| must04.dinov2_vitb14_reg.UD.surgery_drop | 3.63 | points | 50,000 | MUST-04: k-NN drop after UD surgery, blocks 1 to 6, dinov2_vitb14_reg | `results/MUST-04/dinov2_vitb14_reg/metrics.json` (e84e20ff35) |
| must04.dinov2_vitb14_reg.UD.twin_t50_1-6 | 0.5600 |  | 20 | MUST-04: UD twin fraction (t 0.5), blocks 1 to 6, dinov2_vitb14_reg | `results/MUST-04/dinov2_vitb14_reg/metrics.json` (e84e20ff35) |
| must04.dinov2_vitl14_reg.LR.ratio_to_null_1-6 | 0.5020 |  |  | MUST-04: LR ratio to null, blocks 1 to 6, dinov2_vitl14_reg | `results/MUST-04/dinov2_vitl14_reg/metrics.json` (e84e20ff35) |
| must04.dinov2_vitl14_reg.LR.surgery_drop | 0.3680 | points | 50,000 | MUST-04: k-NN drop after LR surgery, blocks 1 to 6, dinov2_vitl14_reg | `results/MUST-04/dinov2_vitl14_reg/metrics.json` (e84e20ff35) |
| must04.dinov2_vitl14_reg.LR.tax_1-6 | 0.4452 |  |  | MUST-04: LR tax (character formula), MLP, blocks 1 to 6, dinov2_vitl14_reg | `results/MUST-04/dinov2_vitl14_reg/metrics.json` (e84e20ff35) |
| must04.dinov2_vitl14_reg.R180.ratio_to_null_1-6 | 0.5287 |  |  | MUST-04: R180 ratio to null, blocks 1 to 6, dinov2_vitl14_reg | `results/MUST-04/dinov2_vitl14_reg/metrics.json` (e84e20ff35) |
| must04.dinov2_vitl14_reg.R90.ratio_to_null_1-6 | 0.7555 |  |  | MUST-04: R90 ratio to null, blocks 1 to 6, dinov2_vitl14_reg | `results/MUST-04/dinov2_vitl14_reg/metrics.json` (e84e20ff35) |
| must04.dinov2_vitl14_reg.UD.ratio_to_null_1-6 | 0.5114 |  |  | MUST-04: UD ratio to null, blocks 1 to 6, dinov2_vitl14_reg | `results/MUST-04/dinov2_vitl14_reg/metrics.json` (e84e20ff35) |
| must04.dinov2_vitl14_reg.UD.surgery_drop | 0.2320 | points | 50,000 | MUST-04: k-NN drop after UD surgery, blocks 1 to 6, dinov2_vitl14_reg | `results/MUST-04/dinov2_vitl14_reg/metrics.json` (e84e20ff35) |
| must04.dinov2_vitl14_reg.UD.twin_t50_1-6 | 0.5458 |  | 20 | MUST-04: UD twin fraction (t 0.5), blocks 1 to 6, dinov2_vitl14_reg | `results/MUST-04/dinov2_vitl14_reg/metrics.json` (e84e20ff35) |
| must05.random_median_cos | -9.22e-07 |  | 1200 | MUST-05: switch-off median cosine, random pairs | `results/MUST-05/dinov2_vitb14_reg/switch_pairs.csv` (52e29d1676) |
| must05.twin_median_cos | 0.4875 |  | 1200 | MUST-05: switch-off median cosine, twin pairs, blocks 1 to 6 | `results/MUST-05/dinov2_vitb14_reg/switch_pairs.csv` (52e29d1676) |
| must05.twin_share_norm_in_band | 0.9225 |  | 1200 | MUST-05: share of twin pairs with norm ratio in [0.8, 1.25] | `results/MUST-05/dinov2_vitb14_reg/switch_pairs.csv` (52e29d1676) |
| must06.deit3_b16.max_abs_erank_change | 0.1764 |  | 4385 | MUST-06: largest \|relative change\| of effective rank under projection, blocks 1 to 6, deit3_b16 | `results/MUST-06/deit3_b16/spectra.csv` (e84e20ff35) |
| must06.deit3_b16.median_same_corr | 0.0331 |  | 4385 | MUST-06: median same-token twin correlation, blocks 1 to 6, deit3_b16 | `results/MUST-06/deit3_b16/pairs.csv` (e84e20ff35) |
| must06.deit3_b16.share_seen | 0.0388 |  | 4385 | MUST-06: share of twins in the top-10 same-image partners, blocks 1 to 6, deit3_b16 | `results/MUST-06/deit3_b16/pairs.csv` (e84e20ff35) |
| must06.dinov2_vitb14_reg.max_abs_erank_change | 0.0864 |  | 5810 | MUST-06: largest \|relative change\| of effective rank under projection, blocks 1 to 6, dinov2_vitb14_reg | `results/MUST-06/dinov2_vitb14_reg/spectra.csv` (52e29d1676) |
| must06.dinov2_vitb14_reg.median_same_corr | 0.0516 |  | 5810 | MUST-06: median same-token twin correlation, blocks 1 to 6, dinov2_vitb14_reg | `results/MUST-06/dinov2_vitb14_reg/pairs.csv` (52e29d1676) |
| must06.dinov2_vitb14_reg.share_seen | 0.0812 |  | 5810 | MUST-06: share of twins in the top-10 same-image partners, blocks 1 to 6, dinov2_vitb14_reg | `results/MUST-06/dinov2_vitb14_reg/pairs.csv` (52e29d1676) |
| must06.dinov2_vitl14_reg.max_abs_erank_change | 0.2161 |  | 7255 | MUST-06: largest \|relative change\| of effective rank under projection, blocks 1 to 6, dinov2_vitl14_reg | `results/MUST-06/dinov2_vitl14_reg/spectra.csv` (e84e20ff35) |
| must06.dinov2_vitl14_reg.median_same_corr | 0.0870 |  | 7255 | MUST-06: median same-token twin correlation, blocks 1 to 6, dinov2_vitl14_reg | `results/MUST-06/dinov2_vitl14_reg/pairs.csv` (e84e20ff35) |
| must06.dinov2_vitl14_reg.share_seen | 0.1043 |  | 7255 | MUST-06: share of twins in the top-10 same-image partners, blocks 1 to 6, dinov2_vitl14_reg | `results/MUST-06/dinov2_vitl14_reg/pairs.csv` (e84e20ff35) |
| must06.invisible_claim | FAIL |  |  | MUST-06: the claim 'invisible' (all three predictions on dinov2_vitb14_reg) | `results/MUST-06/dinov2_vitb14_reg/pairs.csv` (52e29d1676) |
| must07.deit3_b16.6mlp.mirror_exact.top1 | 81.06 | % | 1 | MUST-07: validation k-NN, mirror_exact, 6mlp, deit3_b16 (mean over fit seeds) | `results/MUST-07/rows_deit3_b16.csv` (5edf2980f0) |
| must07.deit3_b16.full.mirror_exact.top1 | 72.85 | % | 1 | MUST-07: validation k-NN, mirror_exact, full, deit3_b16 (mean over fit seeds) | `results/MUST-07/rows_deit3_b16.csv` (5edf2980f0) |
| must07.deit3_b16.original.original.top1 | 82.97 | % | 1 | MUST-07: validation k-NN, original, original, deit3_b16 (mean over fit seeds) | `results/MUST-07/rows_deit3_b16.csv` (5edf2980f0) |
| must07.dinov2_vitb14_reg.6both.asvd_scale.top1 | 74.41 | % | 3 | MUST-07: validation k-NN, asvd_scale, 6both, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitb14_reg.6both.flap.top1 | 70.29 | % | 3 | MUST-07: validation k-NN, flap, 6both, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitb14_reg.6both.mirror_exact.top1 | 77.34 | % | 3 | MUST-07: validation k-NN, mirror_exact, 6both, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitb14_reg.6both.mirror_local.top1 | 65.90 | % | 1 | MUST-07: validation k-NN, mirror_local, 6both, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitb14_reg.6both.mirror_minus_asvd_scale | 2.92 | points | 3 | MUST-07: Mirror-exact minus asvd_scale, 6both, mean over fit seeds 0 to 2 | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35); `results/MUST-07/bootstrap.csv` (805c231c9c) |
| must07.dinov2_vitb14_reg.6both.mirror_minus_flap | 7.05 | points | 3 | MUST-07: Mirror-exact minus flap, 6both, mean over fit seeds 0 to 2 | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35); `results/MUST-07/bootstrap.csv` (805c231c9c) |
| must07.dinov2_vitb14_reg.6both.mirror_minus_pair_actcorr | 46.03 | points | 3 | MUST-07: Mirror-exact minus pair_actcorr, 6both, mean over fit seeds 0 to 2 | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35); `results/MUST-07/bootstrap.csv` (805c231c9c) |
| must07.dinov2_vitb14_reg.6both.mirror_minus_pair_random | 45.04 | points | 3 | MUST-07: Mirror-exact minus pair_random, 6both, mean over fit seeds 0 to 2 | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35); `results/MUST-07/bootstrap.csv` (805c231c9c) |
| must07.dinov2_vitb14_reg.6both.mirror_minus_pair_wcos | 37.48 | points | 3 | MUST-07: Mirror-exact minus pair_wcos, 6both, mean over fit seeds 0 to 2 | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35); `results/MUST-07/bootstrap.csv` (805c231c9c) |
| must07.dinov2_vitb14_reg.6both.mirror_minus_random_structure | 75.05 | points | 3 | MUST-07: Mirror-exact minus random_structure, 6both, mean over fit seeds 0 to 2 | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35); `results/MUST-07/bootstrap.csv` (805c231c9c) |
| must07.dinov2_vitb14_reg.6both.mirror_minus_slicegpt | 1.74 | points | 3 | MUST-07: Mirror-exact minus slicegpt, 6both, mean over fit seeds 0 to 2 | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35); `results/MUST-07/bootstrap.csv` (805c231c9c) |
| must07.dinov2_vitb14_reg.6both.mirror_minus_svd | 71.47 | points | 3 | MUST-07: Mirror-exact minus svd, 6both, mean over fit seeds 0 to 2 | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35); `results/MUST-07/bootstrap.csv` (805c231c9c) |
| must07.dinov2_vitb14_reg.6both.mirror_minus_svdllm | -0.6840 | points | 3 | MUST-07: Mirror-exact minus svdllm, 6both, mean over fit seeds 0 to 2 | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35); `results/MUST-07/bootstrap.csv` (805c231c9c) |
| must07.dinov2_vitb14_reg.6both.mirror_minus_zipit | 58.83 | points | 3 | MUST-07: Mirror-exact minus zipit, 6both, mean over fit seeds 0 to 2 | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35); `results/MUST-07/bootstrap.csv` (805c231c9c) |
| must07.dinov2_vitb14_reg.6both.pair_actcorr.top1 | 31.31 | % | 3 | MUST-07: validation k-NN, pair_actcorr, 6both, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitb14_reg.6both.pair_random.top1 | 32.30 | % | 3 | MUST-07: validation k-NN, pair_random, 6both, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitb14_reg.6both.pair_wcos.top1 | 39.85 | % | 3 | MUST-07: validation k-NN, pair_wcos, 6both, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitb14_reg.6both.random_structure.top1 | 2.29 | % | 3 | MUST-07: validation k-NN, random_structure, 6both, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitb14_reg.6both.slicegpt.top1 | 75.60 | % | 3 | MUST-07: validation k-NN, slicegpt, 6both, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitb14_reg.6both.svd.top1 | 5.87 | % | 3 | MUST-07: validation k-NN, svd, 6both, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitb14_reg.6both.svdllm.top1 | 78.02 | % | 3 | MUST-07: validation k-NN, svdllm, 6both, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitb14_reg.6both.zipit.top1 | 18.51 | % | 3 | MUST-07: validation k-NN, zipit, 6both, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitb14_reg.6mlp.asvd_scale.top1 | 76.81 | % | 3 | MUST-07: validation k-NN, asvd_scale, 6mlp, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitb14_reg.6mlp.flap.top1 | 75.20 | % | 3 | MUST-07: validation k-NN, flap, 6mlp, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitb14_reg.6mlp.mirror_exact.top1 | 76.60 | % | 3 | MUST-07: validation k-NN, mirror_exact, 6mlp, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitb14_reg.6mlp.mirror_local.top1 | 77.12 | % | 1 | MUST-07: validation k-NN, mirror_local, 6mlp, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitb14_reg.6mlp.mirror_minus_asvd_scale | -0.2040 | points | 3 | MUST-07: Mirror-exact minus asvd_scale, 6mlp, mean over fit seeds 0 to 2 | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35); `results/MUST-07/bootstrap.csv` (805c231c9c) |
| must07.dinov2_vitb14_reg.6mlp.mirror_minus_flap | 1.41 | points | 3 | MUST-07: Mirror-exact minus flap, 6mlp, mean over fit seeds 0 to 2 | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35); `results/MUST-07/bootstrap.csv` (805c231c9c) |
| must07.dinov2_vitb14_reg.6mlp.mirror_minus_pair_actcorr | 49.19 | points | 3 | MUST-07: Mirror-exact minus pair_actcorr, 6mlp, mean over fit seeds 0 to 2 | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35); `results/MUST-07/bootstrap.csv` (805c231c9c) |
| must07.dinov2_vitb14_reg.6mlp.mirror_minus_pair_random | 54.20 | points | 3 | MUST-07: Mirror-exact minus pair_random, 6mlp, mean over fit seeds 0 to 2 | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35); `results/MUST-07/bootstrap.csv` (805c231c9c) |
| must07.dinov2_vitb14_reg.6mlp.mirror_minus_pair_wcos | 42.19 | points | 3 | MUST-07: Mirror-exact minus pair_wcos, 6mlp, mean over fit seeds 0 to 2 | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35); `results/MUST-07/bootstrap.csv` (805c231c9c) |
| must07.dinov2_vitb14_reg.6mlp.mirror_minus_random_structure | 74.29 | points | 3 | MUST-07: Mirror-exact minus random_structure, 6mlp, mean over fit seeds 0 to 2 | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35); `results/MUST-07/bootstrap.csv` (805c231c9c) |
| must07.dinov2_vitb14_reg.6mlp.mirror_minus_slicegpt | 45.19 | points | 3 | MUST-07: Mirror-exact minus slicegpt, 6mlp, mean over fit seeds 0 to 2 | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35); `results/MUST-07/bootstrap.csv` (805c231c9c) |
| must07.dinov2_vitb14_reg.6mlp.mirror_minus_svd | 67.24 | points | 3 | MUST-07: Mirror-exact minus svd, 6mlp, mean over fit seeds 0 to 2 | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35); `results/MUST-07/bootstrap.csv` (805c231c9c) |
| must07.dinov2_vitb14_reg.6mlp.mirror_minus_svdllm | -1.75 | points | 3 | MUST-07: Mirror-exact minus svdllm, 6mlp, mean over fit seeds 0 to 2 | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35); `results/MUST-07/bootstrap.csv` (805c231c9c) |
| must07.dinov2_vitb14_reg.6mlp.mirror_minus_zipit | 0.9340 | points | 3 | MUST-07: Mirror-exact minus zipit, 6mlp, mean over fit seeds 0 to 2 | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35); `results/MUST-07/bootstrap.csv` (805c231c9c) |
| must07.dinov2_vitb14_reg.6mlp.pair_actcorr.top1 | 27.42 | % | 3 | MUST-07: validation k-NN, pair_actcorr, 6mlp, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitb14_reg.6mlp.pair_random.top1 | 22.40 | % | 3 | MUST-07: validation k-NN, pair_random, 6mlp, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitb14_reg.6mlp.pair_wcos.top1 | 34.42 | % | 3 | MUST-07: validation k-NN, pair_wcos, 6mlp, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitb14_reg.6mlp.random_structure.top1 | 2.31 | % | 3 | MUST-07: validation k-NN, random_structure, 6mlp, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitb14_reg.6mlp.slicegpt.top1 | 31.41 | % | 3 | MUST-07: validation k-NN, slicegpt, 6mlp, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitb14_reg.6mlp.svd.top1 | 9.37 | % | 3 | MUST-07: validation k-NN, svd, 6mlp, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitb14_reg.6mlp.svdllm.top1 | 78.35 | % | 3 | MUST-07: validation k-NN, svdllm, 6mlp, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitb14_reg.6mlp.zipit.top1 | 75.67 | % | 3 | MUST-07: validation k-NN, zipit, 6mlp, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitb14_reg.9both.mirror_exact.top1 | 73.37 | % | 3 | MUST-07: validation k-NN, mirror_exact, 9both, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitb14_reg.full.mirror_exact.top1 | 53.81 | % | 3 | MUST-07: validation k-NN, mirror_exact, full, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitb14_reg.original.original.top1 | 78.94 | % | 1 | MUST-07: validation k-NN, original, original, dinov2_vitb14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| must07.dinov2_vitl14_reg.6mlp.mirror_exact.top1 | 80.36 | % | 1 | MUST-07: validation k-NN, mirror_exact, 6mlp, dinov2_vitl14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitl14_reg.csv` (d572a53fe9) |
| must07.dinov2_vitl14_reg.full.mirror_exact.top1 | 30.17 | % | 1 | MUST-07: validation k-NN, mirror_exact, full, dinov2_vitl14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitl14_reg.csv` (d572a53fe9) |
| must07.dinov2_vitl14_reg.original.original.top1 | 80.59 | % | 1 | MUST-07: validation k-NN, original, original, dinov2_vitl14_reg (mean over fit seeds) | `results/MUST-07/rows_dinov2_vitl14_reg.csv` (d572a53fe9) |
| must08.6_blocks.saving_ratio | 0.6713 |  |  | MUST-08: MLP-only saving over MLP-plus-attention saving, 6 blocks | `results/MUST-08/rows_dinov2_vitb14_reg.csv` (805c231c9c) |
| must08.6blocks.mlp_minus_both | -0.7360 | points | 3 | MUST-08: k-NN Mirror-exact 6mlp minus 6both, mean over fit seeds 0 to 2 | `results/MUST-08/rows_dinov2_vitb14_reg.csv` (805c231c9c) |
| must08.full_model.saving_ratio | 0.6835 |  |  | MUST-08: MLP-only saving over MLP-plus-attention saving, full model | `results/MUST-08/rows_dinov2_vitb14_reg.csv` (805c231c9c) |
| must09.mirror_exact.knn_mean | 78.34 | % | 3 | MUST-09: healed Mirror-exact, validation k-NN, mean of seeds 0 to 2 | `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0/metrics.json` (52e29d1676); `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s1/metrics.json` (e84e20ff35) and 7 more |
| must09.mirror_exact.knn_std | 0.0208 | points | 3 | MUST-09: healed Mirror-exact, k-NN standard deviation over seeds | `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0/metrics.json` (52e29d1676); `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s1/metrics.json` (e84e20ff35) and 7 more |
| must09.mirror_exact.params_frac | 0.7646 |  |  | MUST-09: healed Mirror-exact, share of the original's parameters (all basis changes counted) | `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0/metrics.json` (52e29d1676); `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s1/metrics.json` (e84e20ff35) and 7 more |
| must09.mirror_minus_best | 0.2793 | points | 3 | MUST-09: Mirror-exact minus best baseline (seed means) | `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0/metrics.json` (52e29d1676); `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s1/metrics.json` (e84e20ff35) and 7 more |
| must09.mirror_minus_random | 8.41 | points |  | MUST-09: Mirror-exact (seed mean) minus random structure (seed 0) | `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0/metrics.json` (52e29d1676); `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s1/metrics.json` (e84e20ff35) and 7 more |
| must09.svdllm.knn_mean | 78.06 | % | 3 | MUST-09: healed best baseline (svdllm), mean of seeds 0 to 2 | `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0/metrics.json` (52e29d1676); `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s1/metrics.json` (e84e20ff35) and 7 more |
| must10.flipavg.knn | 79.00 | % |  | MUST-10: flip-averaged original, validation k-NN (bf16) | `results/MUST-10/flipavg.csv` (e84e20ff35) |
| must10.gflops_ratio | 0.3870 |  |  | MUST-10: healed Mirror-exact GFLOPs over flip-averaging GFLOPs (analytic) | `results/MUST-10/flipavg.csv` (e84e20ff35) |
| must10.mirror_minus_flipavg | -0.6600 | points |  | MUST-10: healed Mirror-exact (mean of 3 seeds) minus flip-averaged original, validation k-NN | `results/MUST-10/flipavg.csv` (e84e20ff35); `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0/metrics.json` (52e29d1676) and 2 more |
| must10.original.knn | 78.95 | % |  | MUST-10: original model, validation k-NN (bf16) | `results/MUST-10/flipavg.csv` (e84e20ff35) |
| must11.ade20k.original | 47.59 | % |  | MUST-11: ADE20K mIoU, original | `results/MUST-11/t2_seg/t2_ade20k.csv` (805c231c9c) |
| must11.ade20k.vexact_s0 | 47.16 | % |  | MUST-11: ADE20K mIoU, healed Mirror-exact seed 0 | `results/MUST-11/t2_seg/t2_ade20k.csv` (805c231c9c) |
| must11.maskeq.original.raw | 95.03 | % |  | MUST-11: mask equivariance, original, raw head (fp32) | `results/MUST-11/t2_seg/t2_ade20k.csv` (805c231c9c) |
| must11.maskeq.vexact_s0.sym | 100.00 | % |  | MUST-11: mask equivariance, healed Mirror-exact seed 0, symmetrised head (fp32) | `results/MUST-11/t2_seg/t2_ade20k.csv` (805c231c9c) |
| must11.probe.original | 81.00 | % |  | MUST-11: ImageNet linear probe, original | `results/MUST-11/t2/t2_linear.csv` (805c231c9c) |
| must11.probe.vexact_mean | 80.11 | % | 3 | MUST-11: ImageNet linear probe, healed Mirror-exact, mean of 3 seeds | `results/MUST-11/t2/t2_linear.csv` (805c231c9c) |
| must12.LR.ratio_to_null_1-6 | 0.4844 |  |  | MUST-12: LR ratio to null, blocks 1 to 6, LR pilot | `results/MUST-12/vit_s8_LR_s0/e100/metrics.json` (e84e20ff35) |
| must12.LR.twin_t50_1-6 | 0.0601 |  |  | MUST-12: LR twin fraction (t 0.5), blocks 1 to 6, LR pilot | `results/MUST-12/vit_s8_LR_s0/e100/metrics.json` (e84e20ff35) |
| must12.core01_epochs | 100 |  |  | CORE-01 epochs by the MUST-12 rule | `runs/MUST-12/vit_s8_none_s0/meta.json` (52e29d1676) |
| must12.none_wall_minutes | 11.76 | minutes |  | MUST-12: wall time of the none pilot run (epoch rule: over 25 minutes means 60 epochs) | `runs/MUST-12/vit_s8_none_s0/meta.json` (52e29d1676) |

## Failed checks (reported as failures)

### New phase

| ID | Prediction | Value | Bar | Source |
|---|---|---|---|---|
| MUST-02 | On 1,000 validation images in fp32, every component of Mirror-exact (full model) has maximum relative equivariance error below 1e-5, and flip consistency is 100% | max rel. error 2.51e-05 (block 11 act); flip 100.0000% | < 1e-5 and 100% | `results/MUST-02/dinov2_vitb14_reg/zeroshot/metrics.json` (e84e20ff35) |
| MUST-04 P4.1 | UD and R90 ratios to null in blocks 1 to 6 are at least 0.9 (LR's ratio is about 0.45) | min UD 0.4807 / min R90 0.7536 | >= 0.9 (UD and R90, every model) | `results/MUST-04/dinov2_vitb14_reg/metrics.json` (e84e20ff35); `results/MUST-04/dinov2_vitb14/metrics.json` (e84e20ff35); `results/MUST-04/dinov2_vitl14_reg/metrics.json` (e84e20ff35) and 1 more |
| MUST-04 P4.2 | The UD twin fraction in blocks 1 to 6 is within the twin-fraction null's 95% interval | dinov2_vitb14_reg: 0.5600 in [0, 6.00e-04]; dinov2_vitb14: 0.5298 in [0.0016, 0.0035]; dinov2_vitl14_reg: 0.5458 in [0, 3.68e-04]; deit3_b16: 0.4630 in [0, 0] | inside [null 2.5%, null 97.5%] | `results/MUST-04/dinov2_vitb14_reg/metrics.json` (e84e20ff35); `results/MUST-04/dinov2_vitb14/metrics.json` (e84e20ff35); `results/MUST-04/dinov2_vitl14_reg/metrics.json` (e84e20ff35) and 1 more |
| MUST-04 P4.3 | Zero-shot UD group-projection surgery on blocks 1 to 6 drops validation k-NN by more than 30 points, against 1.85 for LR | min UD drop 0.2320 | > 30 points | `results/MUST-04/dinov2_vitb14_reg/metrics.json` (e84e20ff35); `results/MUST-04/dinov2_vitb14/metrics.json` (e84e20ff35); `results/MUST-04/dinov2_vitl14_reg/metrics.json` (e84e20ff35) and 1 more |
| MUST-05 P5.1 | Switch-off median cosine at least 0.7 for twin pairs and at most 0.2 for random pairs | twins 0.4875, random -9.22e-07 | >= 0.7 twins and <= 0.2 random | `results/MUST-05/dinov2_vitb14_reg/switch_pairs.csv` (52e29d1676) |
| MUST-06 P6.3 | Effective rank (entropy-based) of fc1 and fc2 changes by at most 2% under group projection | 0.0864 | <= 0.02 for all 12 | `results/MUST-06/dinov2_vitb14_reg/pairs.csv` (52e29d1676) |
| MUST-07 P7.1 | On dinov2_vitb14_reg at 6 blocks, Mirror-exact beats every baseline on validation k-NN by at least 2 points, with the paired bootstrap interval excluding 0 | smallest margin -1.75 | >= 2.0 points and CI low > 0, every method, both sizes | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35); `results/MUST-07/bootstrap.csv` (805c231c9c) |
| MUST-07 P7.2 | ZipIt-style merging and same-image-correlation pairing trail Mirror-exact by at least 10 points at 6 blocks | smallest margin 0.9340 | >= 10.0 points | `results/MUST-07/rows_dinov2_vitb14_reg.csv` (e84e20ff35) |
| MUST-12 gate C | Gate C: in the LR model, the LR ratio to null in blocks 1 to 6 is at most 0.8 and the LR twin fraction in blocks 1 to 6 is at least 0.2 | ratio 0.4844, twin 0.0601 | ratio <= 0.8 and twin >= 0.2 | `results/MUST-12/vit_s8_LR_s0/e100/metrics.json` (e84e20ff35); `results/MUST-12/gate_c.json` (not recorded) |
| CORE-01 T2 | Specificity: the UD ratio to null is at most 0.7 in every UD run, and at least 0.9 in every none run and every LR run (blocks 1 to 6, final) | UD s0: n/a; UD s1: n/a; UD s2: n/a; none s0: 0.7026; none s1: n/a; none s2: n/a; LR s0: 0.6856; LR s1: n/a; LR s2: n/a | UD runs <= 0.7; none and LR runs >= 0.9 | `results/MUST-12/vit_s8_none_s0/e100/metrics.json` (e84e20ff35); `results/MUST-12/vit_s8_LR_s0/e100/metrics.json` (e84e20ff35) |
| CORE-01 T4 | Emergence: in every LR run (for LR) and every UD run (for UD), the twin fraction of the trained group is higher at the final epoch than at epoch 5 | LR s0: 0.0846 -> 0.0601; LR s1: n/a -> n/a; LR s2: n/a -> n/a; UD s0: n/a -> n/a; UD s1: n/a -> n/a; UD s2: n/a -> n/a | final > epoch 5, every LR and UD run | `results/MUST-12/vit_s8_LR_s0/e100/metrics.json` (e84e20ff35) |
| CORE-03 P3.1 | Mirror plus low rank lies above low rank alone (the better of plain SVD and SVD-LLM) at at least 3 of 4 budget points on at least 2 of 3 models, including DINOv2-L or DeiT-III (zero-shot) | dinov2_vitb14_reg: 0 of 4 points above; dinov2_vitl14_reg: 0 of 4 points above; deit3_b16: 0 of 4 points above | >= 3 of 4 points on >= 2 of 3 models, one of them DINOv2-L or DeiT-III | `results/CORE-03/curves.csv` (805c231c9c) |
| CORE-04 P4.1 | Scale: the tax in blocks 1 to L/2 increases from DINOv2-S to g (either direction is reported) | s14 0.4002, b14 0.4185, l14 0.4140, g14 0.3440 | S < B < L < g | `results/CORE-04/structure/dinov2_vits14_reg/metrics.json` (e84e20ff35); `results/MUST-04/dinov2_vitb14_reg/metrics.json` (e84e20ff35); `results/MUST-04/dinov2_vitl14_reg/metrics.json` (e84e20ff35) and 1 more |
| CORE-04 P4.2 | Models whose recipe is verified to have no flips have a lower LR tax than every flip-trained model of the same size | clip_b16: 0.3125; convnext_b: 0.2168; deit3_b16: 0.3487; dino_b16: 0.2495; dinov2_vitb14: 0.4048; dinov2_vitb14_reg: 0.4185; dinov2_vitg14_reg: 0.3440; dinov2_vitl14_reg: 0.4140; dinov2_vits14_reg: 0.4002; dinov3_b16: 0.4002; eva02_b14: 0.1647; mae_b16: 0.4482; resnet50: 0.2794; siglip2_b16: 0.2309 | below every flip-trained model of its size | `results/CORE-04/structure/clip_b16/metrics.json` (not recorded); `results/MUST-04/deit3_b16/metrics.json` (not recorded); `results/CORE-04/structure/dino_b16/metrics.json` (not recorded) and 4 more |
| CORE-05 P5.1 | The activation-weighted breaking share is at most half the weight-energy breaking share in blocks 1 to 6 | 0.5803 | <= 0.5 | `results/CORE-05/measures_dinov2_vitb14_reg.csv` (805c231c9c) |
| CORE-05 P5.3 | The out-of-sample rule is correct on at least 75% of layer-model cases on DINOv2-L-reg, DeiT-III and MAE | 0.6667 | >= 0.75 | `results/CORE-05/rule_score.json` (805c231c9c) |

### Earlier phase (from results/paper/numbers.json)

| Check | Key | Value | Bar | Verdict | Source |
|---|---|---|---|---|---|
| F4 energy rule (P2): predicts the single-block winner | p2 | 0.4444 | >= 0.75 | FAIL | `runs/f4/f4_p2.json` (8a7b6c4e42) |
| Weights-only twin pairing (P5 part 2): agreement with the data-driven pairing | p5_part2 | 0.4658 | >= 0.7 | FAIL | `runs/d/d_datafree.json` (6fd06a579e) |
| Weights-only twin pairing, fully data-free (P5 part 2, strict) | p5_part2_strict | 0.4409 | >= 0.7 | FAIL | `runs/d/d_datafree.json` (6fd06a579e) |
| gate G4: below the bar, claim C6 is dropped | g4 | n/a | P5 passes | FAIL | `runs/d/d_datafree.json` (6fd06a579e) |
| gate G2: P3.a and P3.b (else stop and discuss the framing) | m3.g2 | n/a | both pass | FAIL | `T_healing_p3` (94b2e19d8b) |
| all three P3 bullets | m3.p3 | n/a | all pass | FAIL | `T_healing_p3` (94b2e19d8b) |
| healed Mirror minus healed SVD | m3.p3.b | 1.24 | >= 3 | FAIL | `T_healing` (94b2e19d8b) |
| SigLIP2 breaking share minus DINOv2-reg breaking share (row 4 definition, fit seed 0) | p4.siglip2 | -0.0861 | > 0 | FAIL | `runs/g/siglip2_b16_s0;runs/x2/perlayer_reg_s0` (98faa6454f) |
| P5 (parts 1 and 2 as D2 defines them) | p5 | n/a | both parts pass | FAIL | `runs/d/d_datafree.json` (6fd06a579e) |
| X4 folding check (plain, float32) | x4.plain.float32 | n/a |  | FAIL | `results/x4_fold_dinov2_vitb14.json` (e8bf18d879) |
| X4 folding check (plain, float64) | x4.plain.float64 | n/a |  | FAIL | `results/x4_fold_dinov2_vitb14_fp64.json` (aabbbfa3bc) |
| X4 folding check (reg, float32) | x4.reg.float32 | n/a |  | FAIL | `results/x4_fold_dinov2_vitb14_reg.json` (e8bf18d879) |

## Earlier-phase numbers the paper cites (MUST-00 audit of PROJECT_NEXT_PHASE.md Section 3)

As audited in `results/MUST-00/audit.json`; CONTEXT and MISMATCH rows need the corrected wording of `results/MUST-00/audit.md` before they are cited.

| # | Topic | Claim | Value found | Source | Audit |
|---|---|---|---|---|---|
| 1 | Original k-NN top-1 | 78.94 | 78.9359986782074 in all three fit-seed runs; the RTX 5090 runs measure 78.932 | runs/x2/perlayer_reg_s0/results/e1_surgery.csv:2 (k=0, original); numbers.json main.reg.original.top1; results/paper/tables/T_main_zeroshot.md:14 | MATCH (rounded) |
| 2 | Twins | Block 1: 1,168 twin pairs and 736 self-twins of 3,072 neurons (76% in pairs) | n_pairs 1168, n_self 736, h 3072, twin_frac 0.7604166666666666 (fit seed 0). Fit seed 1: 1171 / 730 (0.762); fit seed 2: 1167 / 738 (0.760) | runs/x2/perlayer_reg_s0/results/e1_mlp.csv:2 (block 1); results/paper/tables/T_structure_per_layer.md:35 (seed 1: line 53) | MATCH |
| 3 | Twins | Blocks 1 to 6: 63% in pairs | twin_frac_mean 0.630859375 (3 fit seeds, std 0.00066) | results/paper/tables/T_structure_summary.csv row dinov2_vitb14_reg,1-6; T_f1_ranges.md:11; numbers.json f1.reg.first_half.twin | MATCH (rounded) |
| 4 | Twins | Blocks 7 to 12: 13% | twin_frac_mean 0.13494646990740738 (3 fit seeds) | results/paper/tables/T_structure_summary.csv row dinov2_vitb14_reg,7-12; T_f1_ranges.md:12 | MATCH (rounded) |
| 5 | Twins | Mean twin correlation 0.80 | Block 1 mean twin correlation 0.8039 (fit seed 0), 0.8021 (seed 1), 0.8023 (seed 2). Mean over blocks 1 to 6: 0.689; blocks 7 to 12: 0.793; all 12 blocks: 0.741 (fit seed 0, computed by this audit from the per-block values) | runs/x2/perlayer_reg_s0/results/e1_mlp.csv:2 column mean_score_pairs; T_structure_per_layer.md:35 to 46 (twin corr column); T_structure_per_layer.csv | CONTEXT |
| 6 | Odd directions | 279 of 768 at layer 1, 128 at layer 6, 8 at layer 12 | d_minus 279 (layer 1), 128 (layer 6), 8 (layer 12), d = 768, for fit seeds 0 and 2; fit seed 1: 281, 128, 8 | runs/x2/perlayer_reg_s0/results/e1_layers.csv rows layer=1, 6, 12 (column d_minus); T_structure_per_layer.md:35, 40, 46 | MATCH |
| 7 | Tax | 0.42 of MLP weights in blocks 1 to 6 | tax_formula_mean 0.4186713842698085 (3 fit seeds, std 0.0002); realised saving identical | results/paper/tables/T_structure_summary.csv row dinov2_vitb14_reg,1-6; T_f1_ranges.md:11; numbers.json f1.reg.first_half.tax | MATCH (rounded) |
| 8 | Tax | 0.11 in blocks 7 to 12 | tax_formula_mean 0.10940306863667053 (3 fit seeds) | results/paper/tables/T_structure_summary.csv row dinov2_vitb14_reg,7-12; T_f1_ranges.md:12 | MATCH (rounded) |
| 9 | Breaking share | About 0.19 to 0.22 in early blocks, against about 0.42 to 0.43 for matched random structure | Registers model, mean fc1 share over blocks 1 to 6: s1 0.2230 (null 0.4271), 3 fit seeds. Plain DINOv2: 0.1938 (null 0.4160). Per block (registers, fit seed 0) s1 runs from 0.155 (block 6) to 0.267 (block 1) | results/paper/tables/T_structure_summary.csv columns s1_mean, s1_null_mean, rows dinov2_vitb14_reg,1-6 and dinov2_vitb14,1-6; T_f1_ranges.md:9, 11; numbers.json f1.reg.first_half.s1, f1.plain.first_half.s1 | CONTEXT |
| 10 | Breaking share | Block 1 fc1: 0.27 against 0.49 | s1 0.2670600731187647, s1_null 0.49425580842939026 (fit seed 0); seeds 1 and 2: 0.266 and 0.267, null 0.495 and 0.494 | runs/x2/perlayer_reg_s0/results/e1_mlp.csv:2 columns s1, s1_null; T_structure_per_layer.md:35 | MATCH (rounded) |
| 11 | Operator fit | 10.6x better than a random involution at layer 1 | 10.56639473986125 is the E1b primary arm: plain DINOv2 (dinov2_vitb14), E1b protocol (39,000-image bank, 8,000 queries, original 77.35), mean over layers 1 to 6, not layer 1. Registers model, paper protocol: mean over layers 1 to 6 is 13.06 / 13.07 / 13.07 (fit seeds 0, 1, 2); layer 1 alone 50.73 / 50.88 / 50.74 | runs/e1b_perlayer_b14/results/e1_meta.json decision.rows[0].value; PROJECT_CONTEXT.md:27; registers: runs/x2/perlayer_reg_s0/results/e1_meta.json decision.rows[0].value and decision.per_layer[0].r1; T_structure_per_layer.md:35 | CONTEXT |
| 12 | Operator fit | e(A)/e(I) = 0.16 | 0.16011080803711272 is the E1b primary arm (plain DINOv2, E1b protocol), mean over layers 1 to 6. Registers model, paper protocol: 0.1698 / 0.1698 / 0.1697 (mean over layers 1 to 6, fit seeds 0, 1, 2); layer 1 alone 0.0406 for all three seeds | runs/e1b_perlayer_b14/results/e1_meta.json decision.rows[1].value; runs/x2/perlayer_reg_s0/results/e1_meta.json decision.rows[1].value and decision.per_layer[0].r2 | CONTEXT |
| 13 | Operator drift | Neighbouring layers agree at 0.80 (chance 0.49) | agree_full 0.8023 (null 0.4881) fit seed 0; 0.8022 (0.4873) seed 1; 0.8024 (0.4875) seed 2 | results/paper/tables/T_f2_drift.md:12 to 14; numbers.json f2.perlayer_reg_s0.agree and .agree_null; results/paper/scripts/f2_dinov2/f2_neighbours.csv | MATCH (rounded) |
| 14 | Artefact tokens | Removing the top 1% by norm cuts late-layer mirror error 2.27x | 2.2722933297201893: masking cut over layers 9 to 11 for plain DINOv2 (mean error 1.315 with all tokens, 0.579 without the top 1%) | runs/f3/f3_dinov2_vitb14.json p1.value (= p1.masking_cut_late); T_prereg.md:35; results/STATUS.md:13 | CONTEXT |
| 15 | Artefact tokens | registers model 1.30x | 1.2964747230982945: late-layer error (layers 9 to 11) over early-layer error (layers 1 to 6), registers model. The masking cut for the registers model is 0.81 (masking slightly raises its error) | runs/f3/f3_dinov2_vitb14_reg.json p1.value, p1.late_over_early, p1.masking_cut_late; T_prereg.md:36 | CONTEXT |
| 16 | Artefact tokens | Pre-registered pass | P1.plain PASS (bar >= 2), P1.registers PASS (bar <= 2), G3 PASS | results/paper/tables/T_prereg.md:35 to 37; runs/f3/f3_gate_G3.json | MATCH |
| 17 | Energy rule (F4) | 16 of 36 correct, needed 27. Pre-registered fail | n_hits 16, n_cases 36, fraction 0.4444, threshold 0.75 (27 of 36), verdict FAIL | runs/f4/f4_p2.json primary_prereg_models; T_prereg.md:38; numbers.json p2.value | MATCH |
| 18 | Zero-shot surgery, 6 blocks | Mirror (per-layer, MLP) 77.09 ± 0.003 (3 fit seeds) | mean 77.09133227666219, sample std (n - 1) 0.0030559478279285263, n = 3; per seed 77.094, 77.088, 77.092 | runs/x2/perlayer_reg_s{0,1,2}/results/e1_surgery.csv row k=6, variant=surgery (line 6 in s0); numbers.json main.reg.k6.mirror_mlp.top1 and .top1_std; T_main_zeroshot.md:24 | MATCH (rounded) |
| 19 | Zero-shot surgery, 6 blocks | plain SVD 8.77 | 8.76933311422666 (mean of 8.768, 8.784, 8.756) | runs/x2/perlayer_reg_s{0,1,2}/results/e1_surgery.csv row k=6, variant=svd; numbers.json main.reg.k6.svd_mlp.top1; T_main_zeroshot.md:25 | MATCH (rounded) |
| 20 | Zero-shot surgery, 6 blocks | random structure 3.95 | 3.945999965071678 (3 fit seeds, std 0.11) | runs/x2/perlayer_reg_s{0,1,2}/results/e1_surgery.csv row k=6, variant=random; numbers.json main.reg.k6.random_mlp.top1; T_main_zeroshot.md:26 | MATCH (rounded) |
| 21 | Ablations | Wrong operator 3.88 | A1 random_operator, k = 6: 3.877333303292593 ± 0.441 (s0 3.916, s1 4.298, s2 3.418) | runs/a/dinov2_vitb14_reg/A1_s{0,1,2}.csv line 6 (variant=random_operator, k=6); T_ablations.md:20; numbers.json a.reg.A1.random_operator.k6.top1 | MATCH (rounded) |
| 22 | Ablations | random pairing 28.96 | A2 random_pairing, k = 6: 28.95866632461548 ± 3.498 (s0 29.646, s1 32.062, s2 25.168) | runs/a/dinov2_vitb14_reg/A2_s{0,1,2}.csv line 3 (variant=random_pairing, k=6); T_ablations.md:31 | MATCH (rounded) |
| 23 | Ablations | pairing by weight cosine 24.69 | A2 generic, k = 6: 24.685999751091003 ± 0.047 (s0 24.652, s1 24.740, s2 24.666) | runs/a/dinov2_vitb14_reg/A2_s{0,1,2}.csv line 5 (variant=generic, k=6); T_ablations.md:33 | MATCH (rounded) |
| 24 | Ablations | 100 fit images 77.03 | A3 n100, k = 6, fit seed 0: 77.03199982643127 | runs/a/dinov2_vitb14_reg/A3_s0.csv:2; T_ablations.md:42; numbers.json a.reg.A3.n100.k6.top1 | MATCH (rounded) |
| 25 | Exact version | 6 blocks, MLP plus attention: 77.37 | V-exact, MLP + attention, k = 6: 77.36799716949463 (fit seed 0, n = 1) | runs/m1/m1_knn.csv:5 (k=6, variant=vexact); numbers.json m1.k6.vexact.top1 | MATCH (rounded) |
| 26 | Exact version | All 12 blocks, before healing: 53.85 | 53.850001096725464 (k = 12, V-exact, fit seed 0) | runs/m1/m1_knn.csv:15; runs/m3_vexact/k12/knn.json zero_shot.top1 | MATCH |
| 27 | Attention | 130 of 144 heads mirror onto themselves; 7 twin-head pairs | n_self_total 130, n_pairs_total 7 (130 + 2 x 7 = 144); per block 12, 10, 10, 12, 10, 10, 10, 10, 12, 10, 12, 12 | runs/m1/m1_heads.csv; T_attention.md:11 to 22; numbers.json m1.heads.n_self_total, m1.heads.n_pairs_total | MATCH |
| 28 | Attention | Block 1 attention breaking share 0.28 against 0.49 random | s_all 0.2831163669660666, s_all_random 0.4939001148308489 | runs/m1/m1_heads.csv:2 (block 1); T_attention.md:11 | MATCH (rounded) |
| 29 | Healing pilot, 9 blocks | Mirror 78.81, SVD 77.57, random 73.28 | healed top-1: Mirror 78.81399989128113, SVD 77.57399678230286, random 73.27799797058105 | runs/m3_pilot/{mirror,svd,random}/knn.json healed.top1; T_healing.md:13 to 15 | MATCH (rounded) |
| 30 | Healing pilot, 9 blocks | Pre-registered 3-point bar failed (it needed 80.57, above the original) | P3.b = 1.24 against >= 3: FAIL; 77.574 + 3 = 80.574 (computed), above the original 78.936 | T_healing.md:22; numbers.json m3.p3.b.value, m3.p3.b.verdict | MATCH (rounded) |
| 31 | Full conversion, healed | 78.31 | 78.31199765205383 (A100, git 5fb53c0744, healing seed 0) | runs/m3_vexact/k12/knn.json healed.top1; T_healing.md (V-exact table, k = 12 row) | MATCH (rounded) |
| 32 | Full conversion, healed | rerun of the same seed on another GPU: 78.34 | 78.33799719810486 (RTX 5090, git 6f35281c33) | runs/m3_vexact_5090/k12/knn.json healed.top1; numbers.json e3.v0.healed | MATCH (rounded) |
| 33 | Full conversion, healed | 100% flip consistency | flip_consistency 100.0 in the fp32 check (TF32 off, 5,000 validation queries, 10,000-image train bank). Paper-protocol k-NN (bf16, 50,000 queries): 99.76399540901184 (5090 rerun 99.742) | runs/m3_vexact/k12/flipcheck.json flip_consistency, n_query, n_bank, dtype; runs/m3_vexact/k12/knn.json healed.flip_consistency | CONTEXT |
| 34 | Full conversion, healed | 76.5% of parameters kept | model_params_frac 0.7646182036976261 (66,203,360 of 86,583,552); converted matrices 0.760 | runs/m3_vexact/k12/summary.json method_info.model_params_frac; T_healing.md (V-exact table, k = 12 row) | MATCH (rounded) |
| 35 | Downstream, 6 blocks healed exact | Linear probe 80.86 against 80.99 | healed_vexact_k6 80.858; original 80.988 | runs/t2/t2_linear.csv:6 to 7; T_tier2.md:21 to 22 | MATCH (rounded) |
| 36 | Downstream, 6 blocks healed exact | ADE20K 47.51 against 47.59 | mIoU healed_vexact_k6 47.50705421032515; original 47.592269358673775 (8,000 iterations, batch 16, crop 448) | runs/t2_seg/t2_ade20k.csv:2 to 3; T_tier2.md:80 | MATCH (rounded) |
| 37 | Downstream, 6 blocks healed exact | transfer within about 0.5 points | Healed V-exact k = 6 against the original: CIFAR-100 90.54 vs 91.40 (0.86 below), DTD 83.19 vs 83.56 (0.37), Pets 95.67 vs 95.99 (0.33), Flowers-102 99.69 vs 99.69 (0.00, ceiling). The 0.51 figure is the healed V-local k = 9 pilot (CIFAR-100 90.89 vs 91.40) | runs/t2/t2_transfer.csv:2, 3, 5, 9, 10, 16, 17, 23, 24; T_tier2.md:43, 45, 50, 57, 64; results/STATUS.md:36, 39 | MISMATCH |
| 38 | Generality | DeiT-III and MAE pass | P4.deit3.all PASS (share 0.184, ratio 10.66); P4.mae.all PASS (share 0.232, ratio 40.76). Decision-table outcome AMBER for both | results/paper/tables/T_generality_p4.csv; T_generality.md:17 to 18, 27 to 32 | MATCH |
| 39 | Generality | DINOv2-L has the strongest structure | Fit ratio e(null)/e(A) 79.41, the highest of all models (next MAE 40.76). On other measures it is not top: e(A)/e(I) 0.079 (MAE 0.048), tax over the first half 0.414 (MAE 0.448), breaking share 0.252 | runs/g/dinov2_l_reg_s0/results/e1_meta.json decision.rows[0].value; T_generality.md:21; T_f1_ranges.md:15, 21 | MATCH |
| 40 | Generality | SigLIP2 weak, as predicted | Outcome KILL; fit ratio 2.38, d-/d 0.138, twin fraction 0.285. Pre-registered P4 sub-check (SigLIP2 breaking share above DINOv2-reg's): 0.137 vs 0.223, FAIL | results/paper/tables/T_generality_p4.csv row P4.siglip2; T_generality.md:19, 33; results/STATUS.md:23 | MISMATCH |
| 41 | Generality | DINOv2-S loses 18 points under surgery | original 74.45600032806396, surgery 56.03399872779846 at k = 6: drop 18.42 (A100 run 56.00) | runs/g/dinov2_s_reg_s0/results/e1_surgery.csv:2 to 3; T_generality.md:20 | MATCH (rounded) |
| 42 | Generality | On DINOv2-L and DeiT-III, plain SVD stays within 0.7 to 0.9 points of Mirror | DINOv2-L: Mirror 79.256 vs SVD 78.568, margin 0.688 (k = 12 of 24); DeiT-III: 81.548 vs 80.670, margin 0.878 (k = 6) | runs/g/dinov2_l_reg_s0/results/e1_surgery.csv:3 to 4; runs/g/deit3_b16_s0/results/e1_surgery.csv:3 to 4; T_generality.md:17, 21 | MATCH (rounded) |
| 43 | Weights only | Operator from weights matches the data-driven one at 0.90 (pass) | p5.part1.value 0.8999972812677988, PASS (bar 0.8; chance 0.652); d+ 392 from weights vs 400 from data | runs/d/d_datafree.json p5.part1; T_prereg.md:56 | MATCH (rounded) |
| 44 | Weights only | pairing from weights 47% (needed 70%, fail) | p5.part2.value 0.4657534246575342 (given the data-driven A_0), FAIL; fully data-free 0.4409, FAIL | runs/d/d_datafree.json p5.part2, p5.part2_strict; T_prereg.md:57 to 58 | MATCH (rounded) |
| 45 | Speed | 0.70 to 0.91x of the original's speed | A100 (runs/m2_clean), bf16, torch.compile, batch 256, Mirror variants against dense in the same implementation: 0.6992 (V-exact k = 12) to 0.9100 (V-local MLP only k = 6). RTX 5090: 0.6346 to 0.8919. A100 batch 1, compiled: 0.495 to 0.843 | results/paper/tables/T_latency.csv (source runs/m2_clean/m2_latency.csv), rows batch=256, mode=compile; T_latency_5090.csv (runs/m2_5090/m2_latency.csv); results/STATUS.md:45, 48 | MATCH (rounded) |
| 46 | Speed | 88.9 ms against 75.7 ms | RTX 5090 profile, batch 256, compiled: V-exact k = 12 median 88.93957 ms; dense 75.68418 ms (ratio 0.851) | runs/m2_profile/m2_profile.csv; T_latency_profile.md:12, 24; numbers.json m2p.reg.vexact.k12.b256.compile.latency | CONTEXT |
| 47 | SVD collapse check (gate G1) | Not a bug | G1 PASS (full-rank SVD reproduces the model, max CLS difference 5.12e-05; ranks within 1% of budget) | results/paper/tables/T_prereg.md:61; runs/x1/x1_svd_check.json; results/STATUS.md:12 | MATCH |
| 48 | Counting error | one d x d basis change per block, not counted: about 3.5M extra parameters at 6 blocks | 6 x 768 x 768 = 3,538,944 (computed). Recorded as rotation_params 3538944 for Mirror V-local MLP only, k = 6, and excluded from model_params | results/paper/tables/T_main_zeroshot.csv rows k=6, variant=mirror_mlp, mirror_mlp_m1, vlocal (column rotation_params); results/paper/tables/cost_model.md:13 | MATCH (rounded) |

## Scoring conventions

- A claim over several models, runs or seed pairs is FAIL as soon as one available unit fails, PASS when every unit is present and passes, otherwise PENDING.
- "Within x points" is read literally: |difference| <= x.
- Exact invariance or equivariance (MUST-02, MUST-09 P9.2, MUST-11 P11.2): PASS at a raw 100%, or when every disagreement has a recorded relative top-two margin below 1e-4 (PREREG Section 1); if disagreements exist and their margins are not recorded the item is PENDING. In P9.2 the allowance applies to the exactly invariant models only; a compressor fails the prediction only at a raw fp32 share of 100%.
- "Final" (CORE-01, MUST-12) is the latest measured checkpoint, and must equal the run's planned epoch count (runs/<exp>/<run>/meta.json config.epochs).
- MUST-03: twin pairs must be equal and breaking shares within 0.005 (half a unit of the quoted 2-decimal values).
- CORE-02: a surgery group is wrong unless all its elements are in the trained group (LR+UD trains the Klein group, R90 the rotations); "equals" in P2.1 uses the MUST-01 tolerance of 1 percentage point.
- CORE-04 P4.1: "increases from S to g" means strictly increasing across S, B, L and g.

## Inputs read

134 files; their sha256 are in PAPER_NUMBERS.json (`inputs`).

