# PREREG_FINAL_PHASE.md: pre-registered predictions and decision rules for the final phase

The thresholds in Section 1 are copied unchanged from FINAL_PHASE_PLAN.md v2, Section 6 (owner: Dawar, 19 November
2026). This file is committed on its own; that commit is the pre-registration record, and Dawar's approval of it is
the sign-off. After the commit nothing in this file changes. Its sha256 and commit are recorded in
`results/final/prereg_record.json` and quoted in `results/STATUS.md`; every Phase G script refuses to run if the file
has changed (`mirror/final/prereg.py`).

Section 2 fixes the operational definitions needed to score each rule mechanically. Where the plan leaves a choice
open, the choice is made here, before any Phase G run, and the reason is given. Section 3 lists the scoring
conventions. Anything not listed is scored by the plain reading of the plan.

## 1. Pre-registered thresholds (FINAL_PHASE_PLAN.md v2, Section 6, unchanged)

| ID | Prediction or decision rule | PASS bar |
|---|---|---|
| Gate P | The new pipeline reproduces MUST-09 | Mirror (current A*) and SVD-LLM means within ±0.15 of 78.34 and 78.06. Otherwise stop and report |
| A01 | Adopt A*_w as the main operator | Zero-shot k-NN ≥ 58.84 (current 53.84 + 5) **and** healed mean (2 seeds) ≥ the current operator's mean in the same pipeline + 0.15. Otherwise keep the current A* and report A*_w as an ablation |
| A02-1 | d− stable across ImageNet fits | Within ±15% of fit set 0 |
| A02-2 | Odd subspaces agree | Energy-weighted mean cos² of principal angles ≥ 0.8 for every pair of ImageNet fits |
| A02-3 | Twins agree | ≥ 70% of strong pairs in blocks 1 to 6 shared between any two ImageNet fits |
| A02-4 | Head maps agree | ≥ 90% of head assignments (self or swap) identical across fits |
| A02-5 | Healed accuracy is stable across fits | Spread over ImageNet fit sets ≤ 0.2 k-NN |
| A02-6 | The fit domain does not matter | COCO-fit healed k-NN within 0.3 of the ImageNet-fit mean |
| A03-1 | Controls C1 to C3 find no structure | Twin fraction in blocks 1 to 6 ≤ 0.05 **and** ratio to null ≥ 0.9 |
| A03-2 | The real pairing matters after healing | True operator + random pairing ≤ Mirror − 3.0 k-NN |
| A04-1 | It is not just distillation | Both dense students ≤ Mirror − 0.5 k-NN |
| A04-2 | Healing without surgery | Report (it defines the ceiling of the recipe) |
| A05-1 | Non-inferiority to every baseline | Lower 95% bound of (Mirror − best baseline) > −0.3 k-NN. Superiority is claimed only if the lower bound is > 0 |
| A06-1 | Stacking beats low rank alone | Mirror + SVD ≥ SVD-LLM + 0.5 at ≥ 2 of 3 budgets, with the interval above 0 at 0.50 |
| A06-2 | The count formula predicts the stacking gain | Measured gain within 50% of the pre-written prediction at ≥ 2 of 3 budgets |
| A06-3 | Price of exactness | Report the k-NN difference between the 9-block and the full conversion |
| A07 | Built-in invariance beats brute force at equal compute | Mirror k-NN ≥ the best flip-averaged model at ≤ equal measured compute + 0.5 |
| A08-1 | DeiT-III-B | Mirror top-1 ≥ SVD-LLM − 0.5 (mean of 3); 100% float32 flip consistency; stored parameters equal the formula |
| A08-2 | DINOv2-L (indicative, 1 seed) | Mirror k-NN ≥ SVD-LLM − 0.5; 100% float32 flip consistency |
| A09-1 | Exact in float64 | Maximum relative error ≤ 1e-10 at every tensor |
| A09-2 | Numerically exact in float32 | Equivariance error ≤ 2x the model's own float32-vs-float64 deviation at every tensor; 100% flip consistency for both forms |
| A09-3 | bfloat16 | Report for both forms (target: 100% for the even/odd form) |
| A10 | Standard metrics | Mirror within 1.0 of the original on full-bank k-NN and on the linear protocol |
| A11 | Healing ceiling | Report. Target: Mirror within 0.2 of the original |
| A12b | Fused kernel | ≥ 1.1x the original's throughput at batch 64 and 256 (bf16, compiled) earns a speed claim; otherwise a limitation |
| A15a | SigLIP 2 | Report the zero-shot k-NN with whitened projection against the current collapse |
| A15b | Cause probe | Rotation ratio to null in blocks 1 to 6 falls by ≥ 0.1 after the fine-tune (from about 0.82); report either way |

## 2. Operational definitions

### 2.1 Common to every rule

- **Main model:** DINOv2-B/14 with registers (`dinov2_vitb14_reg`), 86,583,552 parameters counted as in MUST-09.
- **k-NN** (the MUST-09 protocol, notes/paper/method_spec.md Section 7): bank of 100,000 ImageNet-train images
  (100 per class, cache role `bank`); queries are all 50,000 ImageNet-validation images; feature is the class
  token after the final norm (after the even head for exactly invariant models), L2-normalised; cosine
  similarity; k = 20; votes weighted exp(similarity / 0.07); top-1 in %. Forward passes in bf16 autocast.
- **Full-bank k-NN** (A10): the same with all 1,281,167 ImageNet-train images as the bank.
- **Flip consistency** (float32): the share of validation images whose k-NN prediction for the image equals that
  for its mirror image, with the bank, the queries and the mirrored queries all computed in float32 with TF32
  off. For a classifier head (DeiT-III), the share whose predicted class is the same. "100%" holds when no image
  disagrees, or when every disagreeing image is a near-tie: its relative top-two margin (k-NN: top-two class votes
  over the total vote; classifier: top-two logits over their absolute value) is below 1e-4 in both the image and its
  mirror image (the convention of PREREG_NEXT_PHASE.md Section 1, read in its stricter form).
- **Healing recipe:** identical to MUST-09 (notes/paper/method_spec.md Section 6): the 100,000-image heal set,
  780 optimiser steps of 256 images, AdamW (weight decay 0.05 on matrices), 5% linear warm-up then cosine to 0,
  random resized crops (scale 0.35 to 1) and flips, bf16 autocast, normalised MSE on the final class token, the
  final patch tokens and every block output, flip-averaged final targets, block targets as MUST-09 assigns them
  per method. Within a healing seed every method sees the identical batches in the identical order. Healing
  seeds are 0, 1 and 2, used in that order (a method with n seeds uses seeds 0 to n−1).
- **Learning rates:** existing methods keep their MUST-09 choice (1e-4 for every method; this includes Mirror-exact
  with any operator or fit set, and the random-structure and random-pairing controls). New methods (full
  SVD-LLM, IO-SVD, the dense students, the original distilled, Mirror + SVD-LLM, the 9-block Mirror, the A15b
  probe) choose from {1e-4, 2e-4} by the held-out distillation loss after a 200-step trial (own warm-up and cosine
  schedule over 200 steps, healing seed 0) on the 5,000-image LR-selection set (centre view); the lower loss wins
  and ties go to 1e-4. Validation images are never used for any choice.
- **Sizes:** a baseline at the main size has between 0% and +1% more parameters than Mirror's 66,203,360 (rounding
  towards more parameters for the baseline). A06 sizes are fractions of 86,583,552 with the same tolerance.
- **Means** are arithmetic means of seed-level top-1 values.

### 2.2 Gate P and A01

- **Gate P:** the means over healing seeds 0, 1 and 2 of Mirror-exact with the current operator (A*_12 of
  `results/MUST-07/structure/dinov2_vitb14_reg_s0.npz`) and of SVD-LLM (whitening only, the MUST-09 gate-B
  hyperparameters) are within ±0.15 k-NN of 78.34 and 78.06 respectively. Both must hold. If either fails, Phase G
  stops and reports before any other heal is launched.
- **A*_w** is fitted in **matched mode**: the operator that minimises the projection damage of the fully folded
  model (MLP matrices with their twin pairings, plus the patch-embedding, position-grid, class-token and register
  terms, each normalised by its own weight energy) among symmetric orthogonal involutions with A 1 = 1 and the
  same number of odd directions as the current operator (d− = 68), alternating with weight-space re-pairing that
  keeps each block's number of twin pairs (at most 3 rounds; a new pairing is accepted only if it lowers the
  damage; stop when fewer than 1% of neurons change partner). Head maps are the fitted ones, as now. Reason: with
  d− and the pair counts fixed, the converted model has exactly the current size, so the A01 comparison is not
  confounded by size. The unconstrained fit ("free mode": the plain sign rule) is fitted and reported as a
  diagnostic only; it is not healed.
- **A01 zero-shot:** the k-NN of the A*_w full conversion before healing (step 0 of its healing run).
- **A01 healed:** the mean over healing seeds 0 and 1 of the A*_w heals, against the mean over the same seeds 0 and 1
  of the current operator's Gate P heals (paired by seed).
- If A*_w is adopted, a third A*_w seed (seed 2) is healed, and every later Mirror run uses the weight-space fit
  ("the adopted operator"): for the 9-block point it is fitted the same way over blocks 1 to 9 with the d− of A*_9;
  for DeiT-III-B and DINOv2-L it is fitted on each model's own structure; for the A02 fit sets on each fit set's own
  structure. Otherwise every later run uses the activation-fitted A* of the relevant structure.

### 2.3 A02 (stability)

- Fits: ImageNet-train fit sets 0, 1, 2 (the existing roles) and 3 (a new role of 2 images per class, disjoint from
  every other role), and one fit on 2,000 COCO images (Places365 if COCO is unavailable). Every fit uses the same
  pipeline (`mirror_family.fit_span_structure`).
- A02-1: |d−(fit k) − d−(fit 0)| ≤ 0.15 d−(fit 0) for k = 1, 2, 3, where d− is that of the full-model stream
  operator A*_12.
- A02-2: for fits a and b, with S_a the symmetrised, centred, energy-normalised cross-moment sum that defines
  A*_12 of fit a, Π_a the projector on A_a's odd subspace and E_a = Π_a |S_a| Π_a, ρ(a→b) = tr(Π_b E_a) / tr(E_a);
  the agreement is (ρ(a→b) + ρ(b→a)) / 2 and must be ≥ 0.8 for every pair of ImageNet fits.
- A02-3: a strong pair is a matched twin pair whose symmetrised twin correlation is ≥ 0.5; agreement(a→b) is the
  share of a's strong pairs in blocks 1 to 6 that b also matches as a pair; the mean of both directions must be
  ≥ 0.70 for every pair of ImageNet fits.
- A02-4: the share of heads, over all blocks, with the same assignment in both fits (self-mapped, or swapped with
  the same partner) must be ≥ 0.90 for every pair of ImageNet fits.
- A02-5: Mirror-exact (the adopted operator type) healed with healing seed 0 from fit sets 0, 1 and 2; spread =
  max − min of the three k-NN values ≤ 0.2.
- A02-6: the COCO-fit heal (healing seed 0) is within 0.3 of the mean of the three ImageNet-fit heals of A02-5.

### 2.4 A03 (controls)

- The discovery pipeline is the MUST-04 measurement (notes/paper/method_spec.md Section 8) run on modified paired
  inputs: C1 pairs each image with the mirror of a different image (a fixed seeded derangement); C2 skips the
  re-alignment of the mirrored token grid; C3 replaces the mirror by a fixed seeded random involutive permutation
  of the pixels inside every patch (the same permutation in every patch and channel). The twin fraction is the
  MUST-04 statistic over blocks 1 to 6. **A03-1 scores the split-half ratio to null:** the operators are fitted on
  one half of the fit images and the twin pairings on the other half, then the MUST-04 ratio (blocks 1 to 6, 20 null
  draws) is computed for that structure. Reason: the in-sample MUST-04 ratio is biased below 1 for any fitted
  structure, including one fitted to noise, because the operator fit and the twin matching read the same
  cross-moments (on an exactly equivariant synthetic network the three controls give 0.87 to 0.89 in sample and
  0.91 to 1.00 split-half, against 0.00 split-half for the real mirror; tests/test_fp_controls.py). The in-sample
  ratio is reported next to it for every row, including the real mirror, so the size of this bias in the earlier
  MUST-04 ratios is on record.
- A03-2: H1 (the operator of the main Mirror arm with a random pairing of the fitted pair counts, pairing seed 1,
  healing seed 0) ≤ the main Mirror arm's 3-seed mean − 3.0.

### 2.5 A04 to A07

- A04-1: the width-pruned student (FLAP-style, the gate-B allocation "uniform", main size) and the block-dropped
  student (3 of 12 blocks removed greedily by method-selection k-NN; its size is reported) each, healed with seed
  0, ≤ the main Mirror arm's 3-seed mean − 0.5.
- A05-1: the baselines are SVD-LLM, full SVD-LLM, IO-SVD, SliceGPT-style and plain SVD at the main size. The best
  baseline is the one with the highest mean healed k-NN. The interval is the A13 hierarchical paired bootstrap of
  (Mirror − best baseline) (10,000 resamples, bootstrap seed 0, 95% percentile interval). Mirror's runs are the main
  arm (fit set 0, healing seeds 0 to 2) and the A02 heals from fit sets 1 and 2 (healing seed 0), so structure
  variance enters the interval; the baseline's runs are its healing seeds. Each resample draws Mirror's fit sets
  with replacement, then healing seeds within each drawn fit set, and each method's healing seeds independently
  of the other method's; validation images are drawn once per resample and shared by both methods (paired). The
  point estimate is the mean over fit sets of the mean over each fit set's healing seeds. Other rules that say
  "the main Mirror arm" use the 3-seed mean of fit set 0.
- A06: sizes 0.65, 0.50 and 0.40 of 86,583,552. A06-1 uses the seed-0 point estimates at each budget (the mean of
  seeds 0 and 1 at 0.50), and the interval at 0.50 is the A13 bootstrap over images and those two seeds.
  A06-2: the predicted gain at budget B is acc_pred(B) − acc_SVD(B), with acc_pred(B) = acc_SVD(B / 0.765) − δ,
  δ = (distilled original − healed Mirror at 0.765, 3-seed mean), and acc_SVD interpolated linearly in log size
  between the measured SVD-LLM points (0.40, 0.50, 0.65, 0.765) and the distilled original at size 1.0; the
  measured gain is acc(Mirror + SVD, B) − acc(SVD-LLM, B); "within 50%" means |measured − predicted| ≤ 0.5 |predicted|.
- A07: measured compute is time per image at batch 256 (bf16, torch.compile, median of 100 runs), two passes for a
  flip-averaged model, and for Mirror its faster stored form. The candidates are the flip-averaged SVD-LLM models
  of A06, the flip-averaged DINOv2-S/14 reg and the flip-averaged original; the rule compares the main Mirror arm's
  3-seed mean with the best k-NN among candidates whose measured compute is at most Mirror's, plus 0.5.

### 2.6 A08 to A15

- A08-1 (DeiT-III-B/16): top-1 of the model's own classifier on the 50,000 validation images, means over healing
  seeds 0 to 2. The classifier reads the even part of the class token; it stays frozen during healing for every
  DeiT-III method, as the final norm affine does. Healing adds one loss term: the normalised MSE between the
  student's logits and the flip-averaged teacher logits, weight 1, normaliser fixed on the same 1,024 calibration
  images as the other terms. Learning rate: each method's DINOv2-B rate and the other value of {1e-4, 2e-4} are
  compared by a 200-step trial (Section 2.1 rule). "Stored parameters equal the formula": the parameter count of
  the even/odd form equals the equivariant-map count formula exactly (0 percentage points, as MUST-01).
- A08-2 (DINOv2-L/14 reg): healing seed 0, one heal each.
- A09: the components and precisions of FINAL_PHASE_PLAN.md A09, on 1,000 validation images; errors and deviations
  are the maximum over the images (and their mirror images) of the per-tensor relative norm. The float64 model is
  the stored model's weights re-projected once in float64, so it is exact up to float64 rounding; A09-1 is scored
  on it. "The model's own float32-versus-float64 deviation" (A09-2) is the per-tensor relative difference between
  the stored model run in float32 and that float64 model on the same images. It contains both kinds of float32
  rounding (stored weights and arithmetic), and the 2x factor is then exactly the triangle-inequality allowance for
  the rounding in the image and in its mirror, so only a real break of equivariance can exceed it. The deviation
  with the stored float32 weights run in float64 is reported as a diagnostic. Flip consistency as Section 2.1, for
  both forms.
- A10: |Mirror (3-seed mean) − original| ≤ 1.0 on full-bank k-NN and on the linear protocol. The linear protocol
  follows the DINOv2 linear evaluation on cached centre-crop features of the full training set (no training
  augmentation) with hyperparameters chosen on a held-out 10% of the training features (never on validation); both
  departures from DINOv2's own code are recorded in the results.
- P11.2 (MUST-11, previous phase, still pending): re-measured in A10 on the same model it was measured on, the
  healed Mirror-exact seed 0 of MUST-09 (checkpoint copied from the laptop), with the MUST-11 ADE20K head recipe and
  seed, now logging the top-two logit margin of every pixel whose mask changes under mirroring; scored by
  PREREG_NEXT_PHASE.md Section 1 with the near-tie read in both copies (pixel of the image and the matching pixel of
  the mirror image).
- A12b: throughput at batch 64 and at batch 256, each at least 1.1 times the original's, same device and settings.
- A15a: the whitened projection keeps the fitted operator and pairings and replaces the plain Frobenius metric by
  the activation-weighted one, ||(W − X) M^(1/2)||_F, with M the input second moment on the fit images plus a ridge of
  1e-6 times its mean diagonal (the repository's SVD-LLM value); compared with the plain projection in the
  configuration of the existing collapse (blocks 1 to 6 with A*_6, MLP only and MLP plus attention, bf16 k-NN).
- A15b: the R90 ratio to null in blocks 1 to 6 after the 780-step rotation-invariance fine-tune minus before ≤ −0.1.
  The fine-tune follows the healing recipe except its targets: the student (a copy of the original model, all
  blocks and embeddings trainable) sees rot90(x), and its targets are the teacher's outputs on x with the student's
  patch grid rotated back, without flip averaging (the probe tests rotation pairing alone). Both ratios are measured
  with the MUST-04 code on the same fit images.

## 3. Scoring conventions

- A rule over several models, seeds or pairs fails as soon as one available unit fails; it passes only when every
  unit is present and passes; otherwise it is PENDING with the reason.
- "Within x" is read literally: |difference| ≤ x.
- Every scored number is read from a result file written by a script of this repository and listed with the file
  and the commit recorded in its meta.json.
- Rules cut by the budget governor or the cut order (FINAL_PHASE_PLAN.md Section 9) are reported as NOT RUN with
  the reason, never as passed.
- Every deviation from this file is logged in notes/decisions.md with its reason and reported in the paper.
