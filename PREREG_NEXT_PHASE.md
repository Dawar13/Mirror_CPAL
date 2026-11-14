# ACTIVE: signed off by Dawar on 2026-11-14 (about 18:00 UTC), including every choice of Section 9. Thresholds are fixed from the commit that adds this line.

# PREREG_NEXT_PHASE: pre-registered predictions for the remaining paper experiments

Source of every prediction below: PROJECT_NEXT_PHASE.md (version 1, 14 November 2026), Sections 5 to 8. This file
copies those predictions and thresholds, and adds only the operational definitions needed to score them without
judgement after the fact (marked **Operational**). Where PROJECT_NEXT_PHASE.md leaves a choice open, the choice is
listed in Section 9 for sign-off.

Rules (PROJECT_NEXT_PHASE.md, Section 0 and Section 13):
- Once signed off, the commit that adds this file is the pre-registration record. Every new report and every
  `meta.json` states that commit hash (`mirror/meta.py` records the file's sha256 and the last commit that touched it).
- No threshold, model, block range or test below changes after that commit. If something must change, it is logged in
  `notes/decisions.md` with the reason, and the paper reports it.
- Failed predictions are reported, in the paper, as failures.
- No GPU job of the new phase starts before this commit (gate A).

## 1. Shared definitions

**Blocks.** "Block 1" is the first transformer block (code index 0); "blocks 1 to 6" are code indices 0 to 5. For a
model with L blocks, "the first half" is blocks 1 to L/2.

**Models.** Main model: DINOv2 ViT-B/14 with registers (`dinov2_vitb14_reg`). Hub or timm IDs and revisions are
recorded in `configs/models/*.yaml` before any run.

**Data roles** (PROJECT_NEXT_PHASE.md 4.1, all ImageNet-train roles disjoint by image ID, checked by
`scripts/check_disjoint.py` and logged):
- fit sets fit0, fit1, fit2: 2,000 train images each (2 per class), as in all earlier runs;
- k-NN bank: 100,000 train images (100 per class);
- heal set: 100,000 train images; LR-selection set: 5,000 train images;
- method-selection set (new): 10,000 train images (10 per class), drawn from train shards that no other role uses;
- validation: all 50,000 ImageNet-val images, for reporting only. No choice of method, hyperparameter, learning
  rate or "best baseline" uses validation.
- controlled training: ImageNet-100 (CMC class list, `data/imagenet100_classes.txt`), 72 x 72 uint8, train split for
  training, its validation split for reporting.

**k-NN top-1** (4.6): bank as above, queries the 50k validation images (or the 10k method-selection images for
selection), CLS after the final norm (the even part for Mirror-exact at full conversion), cosine, k = 20, T = 0.07.

**Flip consistency:** share of validation images whose k-NN prediction equals the prediction for the mirrored image.
**Operational:** every claim of exact invariance (MUST-02, MUST-09 P9.2, MUST-11 P11.2) is scored with fp32 features
on all 50,000 validation images (bank features also fp32). bf16 rounding breaks exact ties: the earlier V-exact k = 12
run gave 100% in the fp32 check on 5,000 queries and 99.76% with bf16 features on 50,000 (MUST-00 audit). The bf16
value is reported alongside.
**Operational, rounding allowance for "100%":** an exact-invariance or exact-equivariance prediction (100% flip
consistency, 100% mask equivariance) passes if every disagreement is a near-tie that fp32 rounding can flip: for k-NN,
the disagreeing query's top-two class votes differ by less than 1e-4 of its total vote; for segmentation masks, the
disagreeing pixel's top-two logits differ by less than 1e-4 of their absolute value. The raw share, the number of
disagreements and their margins are reported. Any disagreement with a larger margin is a failure.

**Breaking share** of a matrix W under a group action: ||W - W_eq||^2 / ||W||^2, with W_eq the group projection
(4.4), on the folded weights (norm affine and LayerScale absorbed). Activation-weighted version:
||(W - W_eq) Sigma^(1/2)||^2 / ||W Sigma^(1/2)||^2, Sigma the input covariance on the fit set.

**Null** (4.4): 20 seeded draws of a random orthogonal action with the same character (eigenvalue multiplicities) and
a random hidden permutation with the same cycle type. Reported as mean and 95% interval (2.5th and 97.5th percentiles).

**Ratio to null.** **Operational:** for a block range, the mean breaking share of fc1 and fc2 over the blocks in the
range, divided by the mean of the corresponding null means. Attention breaking shares are reported, not tested.

**Twin fraction.** **Operational:** the share of MLP hidden units that sit in a 2-cycle of the matched permutation
sigma(g) (mutual matching on the symmetrised cross-correlation, as in all earlier runs) whose twin correlation is at
least 0.5. Reported also at 0.6, 0.7 and 0.8; the tests use 0.5. For a block range, the mean over its blocks.

**Twin-fraction null.** **Operational (for sign-off, Section 9):** the twin fraction obtained by the same pipeline when
the token alignment of the group action is replaced by a random permutation of the patch grid with the same cycle type
(for UD on an even grid: a random fixed-point-free involution of the patch positions; prefix tokens keep their index),
20 seeded draws, mean and 95% interval.

**Share in 4-cycles** (R90): the share of MLP hidden units in 4-cycles of the Hungarian matching, after cycles of
length other than 1, 2 and 4 are turned into fixed points.

**Predicted tax** (the law, 4.4): free = (1/|G|) sum_g tr(rho(g)) fix(sigma(g)); tax = 1 - free / (d h).

**Statistics** (4.9): zero-shot headline numbers use 3 fit seeds (mean and standard deviation); healed headline
numbers use 3 training seeds; differences between methods on the same images use a paired bootstrap over validation
images (10,000 resamples, 95% percentile interval, `mirror/bootstrap.py`, seed 0). Every number is reported with n.

## 2. Phase 1 (Must)

### MUST-01. The law (correctness check)
- Prediction: predicted and achieved parameter savings agree to within 1 percentage point per layer, for all 7
  measured models. This is an identity when the code is right: a failure means a bug, to be fixed before any use.
- Unit test (must pass before use): on synthetic exactly equivariant layers for LR, R90 and the Klein group, the
  formula equals a brute-force count of free parameters.

### MUST-02. Exactness
- Prediction: on 1,000 validation images in fp32, every component of Mirror-exact (full model) has maximum relative
  equivariance error below 1e-5, and flip consistency is 100%.
- **Operational:** maximum over images of ||T out(x) - out(mirror x)|| / ||out(mirror x)|| at the output of each
  component in each block, with T the action expected at that component (`docs/exactness.md`). Zero-shot model, fit
  seed 0; the healed seed-0 model is also measured and reported against the same bar.

### MUST-03. Generalised measurement code
- Must pass before MUST-04, MUST-12 and CORE-01 use it: on synthetic exactly equivariant networks, breaking share zero
  (to rounding) and correct orbit counts; on random networks, breaking share at the null level; on
  `dinov2_vitb14_reg` with LR, the LR numbers of earlier runs are reproduced (twin pairs and breaking shares of block 1
  within rounding of `runs/x2/perlayer_reg_s0`).

### MUST-04. Wrong-symmetry controls on released models
Models: `dinov2_vitb14_reg`, `dinov2_vitb14`, DINOv2-L/14 with registers, DeiT-III B/16. Fit seed 0, all layers.
- P4.1: UD and R90 ratios to null in blocks 1 to 6 are at least 0.9 (LR's ratio is about 0.45).
- P4.2: the UD twin fraction in blocks 1 to 6 is within the twin-fraction null's 95% interval.
- P4.3: zero-shot UD group-projection surgery on blocks 1 to 6 drops validation k-NN by more than 30 points, against
  1.85 for LR (78.94 minus 77.09).
- **Operational:** each prediction is scored per model and passes for the claim only if it holds on all four models;
  per-model results are reported either way. P4.3: per-layer UD operators, MLP only, blocks 1 to 6, the same setting
  as the LR 77.09 row. R180 is measured and reported, not tested.

### MUST-05. Twins are working mirror pairs
`dinov2_vitb14_reg`, fit seed 0, blocks 1 to 6, the 200 strongest twin pairs per block against 200 random pairs per
block, 500 validation images.
- P5.1: switch-off median cosine (primary measure, output of block l+1, after re-aligning tokens) at least 0.7 for
  twin pairs and at most 0.2 for random pairs.
- P5.2: the norm ratio is between 0.8 and 1.25 for at least 70% of twin pairs.
- **Operational:** medians and shares pooled over the 1,200 twin pairs (and 1,200 random pairs) of blocks 1 to 6;
  per-block values reported.

### MUST-06. Visibility
Twin pairs in blocks 1 to 6 of `dinov2_vitb14_reg`, DINOv2-L/14 with registers and DeiT-III.
- P6.1: median same-token Pearson correlation of twins on the same image at most 0.3.
- P6.2: a twin is in its partner's top 10 same-image correlated neurons for at most 20% of pairs.
- P6.3: effective rank (entropy-based) of fc1 and fc2 changes by at most 2% under group projection.
- **Operational:** P6.1 and P6.2 pooled over blocks 1 to 6 per model; P6.3 holds if every one of the 12 matrices
  (fc1 and fc2 of blocks 1 to 6) satisfies it, per model. Scored on each model; the claim "invisible" stands if all
  three predictions hold on `dinov2_vitb14_reg` and is reported per model for the other two.
- If it fails: the paper drops the word "invisible" and keeps "different axis", resting on CORE-03 (Section 8).

### MUST-07. Zero-shot comparison against strong compressors
Methods (matched parameters, 4.7): Mirror-exact; Mirror-local (ablation, basis changes counted); plain SVD;
SVD-LLM-style whitening; ASVD-style scaling; SliceGPT-style slicing; FLAP-style pruning; ZipIt-style merging; pairing
by same-image activation correlation tied with the fitted operator; pairing by weight cosine; random pairing; random
structure. Sizes: 6 blocks MLP only; 6 blocks MLP plus attention; 9 blocks; full model.
- P7.1: on `dinov2_vitb14_reg` at 6 blocks, Mirror-exact beats every baseline on validation k-NN by at least 2
  points, with the paired bootstrap interval excluding 0.
- P7.2: ZipIt-style merging and same-image-correlation pairing trail Mirror-exact by at least 10 points at 6 blocks.
- On DINOv2-L and DeiT-III: no prediction; reported as measured.
- **Operational:** "every baseline" means methods 3 to 12 (Mirror-local is an ablation, not a baseline). "At 6
  blocks" means both 6-block sizes, scored separately; the difference is between means over fit seeds 0, 1 and 2, and
  the bootstrap uses the fit-seed-0 per-image predictions. Hyperparameters of every method are chosen on the
  method-selection set only.
- Gate B: the best and second-best baselines are chosen by method-selection k-NN at the full-model size on
  `dinov2_vitb14_reg`, and recorded in `results/MUST-07/gate_b.json` before MUST-09 starts.

### MUST-08. Matched component ablation
- P8.1: the MLP carries most of the saving: MLP-only saving is more than half of the MLP-plus-attention saving, at 6
  blocks and at the full model.
- P8.2: adding attention costs at most 0.5 points at 6 blocks: k-NN(MLP only) minus k-NN(MLP plus attention) is at
  most 0.5 (mean over fit seeds).

### MUST-09. Healed full-model comparison, identical recipe
Runs: Mirror-exact 3 seeds; best baseline 3 seeds; second-best baseline 1 seed; plain SVD 1 seed (3 if within 1 point
of Mirror-exact); random structure 1 seed. Recipe 4.8 for every method, learning rate chosen per method on the
LR-selection set.
- **Operational:** the learning rate is chosen once per method, by the 100-step trials of its seed-0 run (grid {1e-5,
  5e-5, 1e-4}; the 2e-4 check trial is recorded and never chosen); seeds 1 and 2 reuse it. Validation k-NN is logged at
  steps 0 and 780 on all 50,000 images and at steps 100 and 300 on a fixed class-balanced subset of 10,000 validation
  images (reporting only). Micro-batch 64 instead of 32 (the same batch of 256; only the summation order changes).
  The structure (A*, pairings, head maps, register pairing) of every run is the canonical fit-seed-0 span structure of
  `results/MUST-07/structure/`, so every method's budget follows the same structure.
- P9.1: Mirror-exact is non-inferior to the best baseline: mean over seeds of Mirror-exact minus mean of the best
  baseline is above -0.5 points (validation k-NN).
- P9.2: only Mirror-exact has 100% flip consistency. **Operational (for sign-off, Section 9):** the random-structure
  control is built the same way as Mirror-exact (one shared involution over the whole model, here a random one), so it
  is also exactly invariant by construction; the prediction is scored as "Mirror-exact and the random-structure control
  reach 100% (fp32, rounding allowance of Section 1); no compressor (best, second-best, plain SVD) does".
- P9.3: random structure trails Mirror-exact by at least 3 points.
- Gate D: Mirror-exact mean at least best-baseline mean minus 1.0. If it fails, everything else keeps running and the
  framing is discussed (Section 8).

### MUST-10. Against test-time flip averaging
- P10.1: healed Mirror-exact (mean of 3 seeds) is within 1 point of flip averaging on validation k-NN, at under 45% of
  its FLOPs (analytic, one pass against two).
- Linear probe, latency and memory are reported, not tested.

### MUST-11. Downstream tasks
- P11.1: healed Mirror-exact is within 1.0 point of the original on the ImageNet linear probe (mean of 3 seeds) and on
  ADE20K mIoU (seed 0).
- P11.2: Mirror-exact with the symmetrised head gives 100% mask equivariance; the original model stays below 100%.

### MUST-12. Controlled-training pilot, gate C
ViT-S/8 on ImageNet-100 at 64 x 64, seed 0, conditions none and LR.
- Gate C: in the LR model, the LR ratio to null in blocks 1 to 6 is at most 0.8 and the LR twin fraction in blocks 1 to
  6 is at least 0.2. If it fails: report, and decide with Dawar (proposed fix: rerun the pilot at 96 x 96 with
  ViT-S/16 before the grid).
- Epoch rule (fixed before the grid): if the MUST-12 run takes over 25 minutes, every CORE-01 run uses 60 epochs
  instead of 100, with checkpoints at epochs 0, 2, 5, 10, 25, 50 and 60.

## 3. Phase 2 (Core)

### CORE-01. Controlled training
Conditions and seeds: none (3), LR (3), UD (3), LR+UD (2), R90 (2). Measured at every checkpoint for LR, UD, R90 and
Klein. "Final" means the last checkpoint.
- T1 Dissociation: in every seed pair s = 0, 1, 2, the UD twin fraction (blocks 1 to 6, final) of UD run s minus that
  of LR run s is at least 0.2 (absolute). If T1 fails, the cause claim drops (Section 8).
- T2 Specificity: the UD ratio to null is at most 0.7 in every UD run, and at least 0.9 in every none run and every LR
  run (blocks 1 to 6, final).
- T3 Rotation: in every R90 run, the share in 4-cycles (blocks 1 to 6, final) is at least 0.3, and at least 3 times
  the mean share in the none runs.
- T4 Emergence: in every LR run (for LR) and every UD run (for UD), the twin fraction of the trained group is higher
  at the final epoch than at epoch 5. The rise over training is reported as a curve.
- The predicted structure matrix (PROJECT_NEXT_PHASE.md CORE-01) is reported cell by cell; only T1 to T4 are tests.

### CORE-02. The law predicts the refund for new groups
- Before any surgery, the predicted tax for the LR+UD and R90 models is written to `results/CORE-02/predictions.json`
  with a timestamp, and committed.
- P2.1: the achieved saving equals the prediction (identity check).
- P2.2: trained-group surgery on blocks 1 to 6 keeps validation top-1 within 3 points.
- P2.3: every wrong-group surgery loses at least 20 points.

### CORE-03. Two axes: Mirror and low rank stack
- P3.1: Mirror plus low rank lies above low rank alone (the better of plain SVD and SVD-LLM) at at least 3 of 4
  budget points (about 85%, 70%, 55% and 40% of the converted span) on at least 2 of 3 models, including DINOv2-L or
  DeiT-III. Zero-shot; one healed point at 55% on `dinov2_vitb14_reg` is reported.

### CORE-04. Generality, scale, recipes
- P4.1 (scale): the tax in blocks 1 to L/2 increases from DINOv2-S to g. Either direction is reported (Section 8).
- P4.2 (recipe): models whose recipe is verified to have no flips have a lower LR tax than every flip-trained model of
  the same size.
- P4.3 (CNNs): twin structure appears in ConvNeXt's pointwise MLPs. **Operational:** LR ratio to null at most 0.8 and
  twin fraction at least 0.2 in the first half of the stages' pointwise MLPs.

### CORE-05. Why surgery is cheap
- P5.1: the activation-weighted breaking share is at most half the weight-energy breaking share in blocks 1 to 6.
- P5.2: over training (CORE-01 LR and UD runs), breaking energy in the top-variance band falls faster than in the
  bottom band.
- P5.3: the out-of-sample rule, designed on `dinov2_vitb14_reg` only and frozen before scoring, is correct on at least
  75% of layer-model cases on DINOv2-L-reg, DeiT-III and MAE. Reported next to the failed F4 rule.

### CORE-06. Orientation is kept
- P6.1: species from the even part is within 1 point of the full feature.
- P6.2: facing direction from the odd part is at least 85%.
- P6.3: facing direction from healed Mirror-exact's even part is at most 60% on natural test images, and exactly 50% on
  the mirror-augmented test set.

### CORE-07. Distribution shift
- P7.1: Mirror-exact's drop from ImageNet to ImageNet-V2 (matched frequency, linear probe) is within 1 point of the
  original's drop.

## 4. Phase 3 (Upside), reported, not gating
- UP-01: UD tax rises with p (0.1, 0.25, 0.5).
- UP-02: breaking share stays below 0.05 throughout training from a UD-equivariant start, at accuracy within 1 point
  of the UD runs.
- UP-03: the trained-group tax differs between ViT-S and ViT-Ti; either direction is reported.
- UP-04: tied-from-scratch reaches about the same accuracy at the reduced parameter count.
- UP-05: a speed claim needs more than 1.1x the original's speed; otherwise it is a limitation.
- UP-06: the extra drop from INT8 weight-only quantisation and from ToMe (r = 8) is the same for both models within 0.5
  points.
- UP-07: pairing from weights alone agrees with the data-driven pairing on at least 70% of pairs.

## 5. Gates (PROJECT_NEXT_PHASE.md Section 8)
| Gate | When | Condition | If it fails |
|---|---|---|---|
| A | Before any new GPU job | this file committed after sign-off; MUST-00 clean | fix first |
| B | After MUST-07 | best baseline chosen on the method-selection set and recorded | MUST-09 does not start until recorded |
| C | After MUST-12 | Section 2, MUST-12 | report; decide with Dawar |
| D | After MUST-09 | Mirror-exact mean at least best-baseline mean minus 1.0 | keep running; discuss framing |
| E | Before abstract registration (23 Nov) | MUST-04, MUST-05, MUST-07, MUST-12 done; CORE-01 running | freeze title and abstract with Dawar |

## 6. Fallback framings (decided in advance; PROJECT_NEXT_PHASE.md Section 11)
Copied unchanged: CORE-01 T1 fails, so the cause claim drops; LR twins without augmentation but UD twins only with UD
training, so "the symmetry of the training distribution creates the tax"; MUST-06 fails, so drop "invisible"; CORE-03
fails, so soften to "different structure from low rank"; gate D fails, so headline exact equivariance, zero-shot
strength and 100 images; CORE-04 scale goes the other way, so report "smaller models pay less".

## 7. What is already known
Section 3 of PROJECT_NEXT_PHASE.md, as corrected by the MUST-00 audit (`results/MUST-00/audit.md`). Known values are
listed so that no bar can be said to have been set after seeing a new result.

## 8. Deviations
Any deviation from this file (a changed recipe, a failed job rerun with different settings, a dropped run) is logged
in `notes/decisions.md` with date, reason and effect, and listed in the final report.

## 9. Choices for sign-off (PROJECT_NEXT_PHASE.md leaves these open)
1. Twin-fraction null for MUST-04 P4.2: random patch-grid permutation with the action's cycle type (Section 1).
2. MUST-04: a prediction passes only if it holds on all four models.
3. MUST-07 P7.1: "at 6 blocks" means both 6-block sizes, scored separately.
4. MUST-09 targets: every method distils to the same targets as Mirror-exact (flip-symmetrised, as in the earlier
   V-exact runs); a raw-target run of the best baseline is added if time allows and reported as a check.
5. MUST-09 trainable set: for every method, the converted blocks' structured parameters plus the embeddings (as in
   E3); Mirror-exact re-projects its embeddings every step.
6. E3 (PREREG_E3.md) is superseded by MUST-09. If the South Korea machine comes back, its finished E3 arm is pulled and
   reported as earlier work; no E3 run continues.
7. Mirror-exact basis changes: counted at a span end that is not the model boundary (d x d), as PROJECT_NEXT_PHASE.md
   4.5 says, although an implementation could absorb it into the next dense block; the absorbed count is reported too.
8. MUST-09 P9.2 scored as "Mirror-exact and random structure at 100%, no compressor" (Section 2), because random
   structure is exactly invariant by construction.
9. MUST-09: learning rate chosen once per method (seed 0) and reused for seeds 1 and 2; curve points at steps 100 and
   300 on a 10,000-image validation subset; micro-batch 64. These keep each heal near 0.5 to 0.8 H100-hours.
10. Canonical structures: for `dinov2_vitb14_reg` (fit seeds 0, 1, 2) and DeiT-III (fit seed 0), the per-layer operators
    and pairings are the files of the earlier runs (`runs/x2/perlayer_reg_s*/results/e1_operators.npz`,
    `runs/g/deit3_b16_s0/results/e1_operators.npz`, behind 77.09 and the E3 work), uploaded to the H100; the span
    operators A* and head maps are computed from them once by `scripts/must07_compressors.py --structure-only` and
    stored in `results/MUST-07/structure/`. Every new-phase script reads that file. DINOv2-L is fitted fresh.
11. Gates B, C and D: when a gate passes, report it and continue without waiting; stop only when a gate fails. (The
    plan says "stop and report at every gate"; waiting on a passed gate would leave the rented GPU idle.)
