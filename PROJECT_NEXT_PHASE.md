# PROJECT_NEXT_PHASE.md: The Symmetry Tax (CPAL 2027, paper 1), remaining work

Version 1, 14 November 2026. Owner: Dawar. Executor: Claude Code on one rented H100.

This file defines every experiment still to run for the Mirror paper, in three phases: **Phase 1 (Must)**, **Phase 2 (Core)** and **Phase 3 (Upside)**. For the remaining work it supersedes earlier experiment plans (PAPER_EXPERIMENTS.md and the job lists). PROJECT_CONTEXT.md still holds the background. If the two disagree about what to run next, this file wins until Dawar changes it.

Old IDs (E1, E1b, X1 to X3, F1 to F4, A1 to A5, M1 to M3, G, D) refer to finished work. New IDs in this file are `MUST-xx`, `CORE-xx` and `UP-xx`.

---

## 0. How to use this file

1. Read Sections 1 to 4 fully before running anything.
2. Do `MUST-00` first: check that the numbers in Section 3 match the files in `results/`. Report any mismatch before new runs.
3. Write `PREREG_NEXT_PHASE.md` from the predictions and thresholds in Sections 5 to 7. Commit it before the first new GPU job, and put the commit hash in every new report. Thresholds never change after that commit. If something must change, log it in `notes/decisions.md` with the reason, and the paper reports it.
4. Time the first job of each kind (one heal, one controlled training run, one structure measurement). Then re-plan the budget in `results/STATUS.md` from real timings before continuing.
5. Stop and report to Dawar at every gate in Section 8.

---

## 1. The paper in one paragraph

> A symmetry in the training data is not free. A vision transformer trained on mirror-symmetric images learns most of its early features twice: one neuron for a pattern and a twin neuron for its mirror image. A character formula from representation theory gives the exact size of this duplication, the **symmetry tax**, for any finite symmetry group. Controlled training shows that the symmetry of the training distribution creates the tax: models trained with upside-down flips acquire upside-down twins, which released models, never trained that way, do not have. Rank-based and correlation-based compressors cannot see the tax, so it is a second axis of parsimony that stacks with low rank. And it can be refunded in closed form from about 100 images, turning a released DINOv2 into an exactly mirror-equivariant model with about a quarter fewer parameters at near-original accuracy.

**CPAL message:** parsimony has two axes, rank and symmetry, and current compressors only see the first.

**Working titles** (final wording after results):
- *The Symmetry Tax: Vision Transformers Pay for Their Training Symmetries in Duplicate Neurons, and the Bill Can Be Refunded*
- *Every Symmetry Has a Price: Measuring, Predicting and Refunding the Symmetry Tax in Vision Foundation Models*

**Abstract skeleton** (numbers in brackets come from results only):
> Data augmentation teaches a network a symmetry, but what does that cost in parameters? We show that vision foundation models trained with horizontal flips store most early features twice, as mirror-twin neurons: [x]% of early MLP weights in DINOv2 are fixed by their twins. A character formula gives the exact size of this symmetry tax for any finite group. Controlled training shows that the symmetry of the training distribution creates the tax: models trained with vertical flips acquire vertical twins, which released models lack. The tax is invisible to rank-based compression and stacks with it. We refund it in closed form from 100 images: DINOv2 becomes exactly mirror-equivariant with [y]% fewer parameters at [z] points below the original, [matching or beating] the strongest compressor under identical healing, and replacing test-time flip averaging at [w] of its cost.

---

## 2. The six claims and how they connect

Every experiment below serves one of these claims. In the paper, each section opens with the prediction made by the section before it and closes with a number the next section uses.

| # | Claim | Why it comes here | Main figure or table | Experiments |
|---|---|---|---|---|
| 1 | **The tax is real.** Flip-trained ViTs store early features as working mirror twins, specific to the trained symmetry | Raises the question | Fig. 1 (teaser), Fig. 2 (tax by depth and model) | Done, plus MUST-04, MUST-05, CORE-04 |
| 2 | **It has an exact size.** A character formula counts the free parameters of a layer that respects a group; trained networks sit near that subspace only for the group they were trained with | Turns the observation into a quantity | Fig. 2 inset, Table 4 | MUST-01, CORE-02 |
| 3 | **The training distribution creates it.** Train with a symmetry and its twins appear; without it, they do not | Explains Claim 1 and predicts Claim 1's control result | Fig. 3 (twin matrix and emergence curves) | MUST-12, CORE-01, CORE-02, UP-01 to UP-04 |
| 4 | **Compressors cannot see it.** Twin redundancy is not low rank and not same-image correlation; it stacks with low rank | Explains the SVD collapse and predicts stacking | Fig. 4 (visibility and stacking curves) | MUST-06, MUST-07, CORE-03, CORE-05 |
| 5 | **It can be refunded exactly.** Closed-form surgery, exact equivariance, fair comparison, useful features | Delivers the number Claim 2 promised | Fig. 5, Tables 1 to 3 | MUST-02, MUST-07 to MUST-11, CORE-06, CORE-07, UP-05, UP-06 |
| 6 | **Where it stops.** Artefact tokens, model size, training recipes | Sets the scope honestly | Fig. 2, Table 4, Limitations | Done, plus CORE-04 |

**How the pieces lock together:**
- The law predicts how large the refund can be, and the refund achieves it.
- The cause predicts that released models have no upside-down twins, and MUST-04 confirms it.
- Invisibility explains why plain SVD can collapse while Mirror does not, and predicts that stacking works (CORE-03).
- The mechanism (CORE-05) explains why surgery is cheap: the asymmetric part of the weights sits in directions the data rarely uses.
- Artefact tokens show where the network's own computation stops being symmetric.

---

## 3. What is already established (do not rerun unless stated)

These come from earlier reports. `MUST-00` checks each against its results file before the paper cites it. All are DINOv2-B with registers (`dinov2_vitb14_reg`) under the paper protocol (Section 4.6) unless stated.

| Topic | Result |
|---|---|
| Original k-NN top-1 | 78.94 |
| Twins | Block 1: 1,168 twin pairs and 736 self-twins of 3,072 neurons (76% in pairs). Blocks 1 to 6: 63% in pairs. Blocks 7 to 12: 13%. Mean twin correlation 0.80 |
| Odd directions | 279 of 768 at layer 1, 128 at layer 6, 8 at layer 12 |
| Tax | 0.42 of MLP weights in blocks 1 to 6; 0.11 in blocks 7 to 12 |
| Breaking share | About 0.19 to 0.22 in early blocks, against about 0.42 to 0.43 for matched random structure (block 1 fc1: 0.27 against 0.49) |
| Operator fit | 10.6x better than a random involution at layer 1; e(A)/e(I) = 0.16 |
| Operator drift | Neighbouring layers agree at 0.80 (chance 0.49) |
| Artefact tokens | Removing the top 1% by norm cuts late-layer mirror error 2.27x; registers model 1.30x. Pre-registered pass |
| Energy rule (F4) | 16 of 36 correct, needed 27. Pre-registered **fail**; report it |
| Zero-shot surgery, 6 blocks | Mirror (per-layer, MLP) 77.09 ± 0.003 (3 fit seeds); plain SVD 8.77; random structure 3.95 |
| Ablations | Wrong operator 3.88; random pairing 28.96; pairing by weight cosine 24.69; 100 fit images 77.03 |
| Exact version | 6 blocks, MLP plus attention: 77.37 (**different configuration** from 77.09, so the two cannot be compared; MUST-08 fixes this). All 12 blocks, before healing: 53.85 |
| Attention | 130 of 144 heads mirror onto themselves; 7 twin-head pairs. Block 1 attention breaking share 0.28 against 0.49 random |
| Healing pilot, 9 blocks | Mirror 78.81, SVD 77.57, random 73.28. Pre-registered 3-point bar failed (it needed 80.57, above the original) |
| Full conversion, healed | 78.31 (rerun of the same seed on another GPU: 78.34). 100% flip consistency. 76.5% of parameters kept |
| Downstream, 6 blocks healed exact | Linear probe 80.86 against 80.99; ADE20K 47.51 against 47.59; transfer within about 0.5 points |
| Generality | DeiT-III and MAE pass; DINOv2-L has the strongest structure; SigLIP2 weak, as predicted; DINOv2-S loses 18 points under surgery. On DINOv2-L and DeiT-III, plain SVD stays within 0.7 to 0.9 points of Mirror |
| Weights only | Operator from weights matches the data-driven one at 0.90 (pass); pairing from weights 47% (needed 70%, **fail**) |
| Speed | 0.70 to 0.91x of the original's speed; 88.9 ms against 75.7 ms |
| SVD collapse check (gate G1) | Not a bug |
| Counting error | The per-layer version needs one d x d basis change per block, which was not counted: about 3.5M extra parameters at 6 blocks. **Mirror-exact is the main method from now on**; Mirror-local is an ablation |

---

## 4. Definitions and shared protocol

### 4.1 Data splits (all disjoint, logged by image ID)
- **Fit sets:** 3 sets of 2,000 ImageNet-train images (fit seeds 0, 1, 2). Used for operators, twins and activation statistics.
- **k-NN bank:** 100 images per class from ImageNet-train (100k).
- **Heal set:** 100k ImageNet-train images.
- **LR-selection set:** 5k ImageNet-train images.
- **Method-selection set (new):** 10k ImageNet-train images, used as k-NN queries (against the bank) to choose the best baseline and every hyperparameter. Validation is never used for selection.
- **Validation:** all 50k ImageNet-val images, reporting only.
- **Controlled-training data:** ImageNet-100 (the standard 100-class subset used by CMC, Tian et al. 2019; check the class list), resized to 72 x 72 and stored as uint8 arrays. Train on the train split, report on its val split.

### 4.2 Models

Record the exact hub or timm IDs, revisions and the training augmentation recipe in `configs/models/*.yaml`. For the recipe, include a quoted sentence and a link to the paper or code. Never fill it in from memory.

| Role | Model (verify ID) | Flips in training recipe? |
|---|---|---|
| Main | DINOv2 ViT-B/14 with registers | Yes (verify) |
| Main | DINOv2 ViT-B/14 plain | Yes |
| Scale | DINOv2 ViT-S/14, L/14, g/14, with registers | Yes |
| Supervised | DeiT-III ViT-B/16 (the checkpoint already used) | Yes (verify) |
| Masked reconstruction | MAE ViT-B/16 | Yes (verify) |
| Language-supervised | SigLIP2 ViT-B/16 | Verify |
| Language-supervised | OpenAI CLIP ViT-B/16 | The paper reportedly uses random square crops only (verify the exact sentence) |
| Self-supervised, older | DINO v1 ViT-B/16 | Yes (verify) |
| Optional | DINOv3 ViT-B/16 (gated licence; include only if accessible) | Verify |
| Optional | EVA-02 ViT-B/14 | Verify |
| CNN | ConvNeXt-B (supervised, ImageNet-1k) | Yes (verify) |
| CNN | ResNet-50 (ImageNet-1k) | Yes (verify) |

### 4.3 Symmetry actions on tokens

The patch grid is G x G (16 x 16 for DINOv2 at 224 px). CLS and register tokens are global and keep their index.
- **LR (left-right flip):** the image is flipped horizontally; token (r, c) maps to (r, G-1-c).
- **UD (upside-down flip):** the image is flipped vertically; token (r, c) maps to (G-1-r, c).
- **R90 (rotation by 90 degrees):** the image is rotated by 90 degrees counter-clockwise; token (r, c) maps to (G-1-c, r). Check the convention numerically with a toy image before use.
- **R180:** R90 applied twice.
- **Klein group {e, LR, UD, LR·UD}:** both flips together.

### 4.4 Measurement (one code path for every model and group)
- **Operator per generator g, per layer.** Fit an orthogonal map ρ(g) from tokens of the original image to the aligned tokens of g(image).
  - Order 2 (LR, UD): use the existing closed-form involution fit.
  - R90: fit by orthogonal Procrustes, then enforce ρ^4 = I by rounding the rotation angles of its real Schur form to multiples of 90 degrees.
  - Report the fit error before and after enforcement, e(ρ)/e(I), and the ratio against a random operator with the same eigenvalue multiplicities.
- **Hidden permutation σ(g)** for each MLP (and each attention layer at head level), by matching neuron i on the original to neuron j on g(image) at aligned tokens.
  - Involutions: the existing mutual matching.
  - R90: one-to-one Hungarian matching. Keep cycles of length 1, 2 and 4; turn any other cycle into fixed points, and report how many were turned.
  - Klein group: the two involution matchings. Report how well they commute; orbits come from the group they generate.
  - Report twin, orbit and self-twin fractions at correlation thresholds 0.5, 0.6, 0.7 and 0.8, plus the matching itself.
- **Group projection (the surgery, generalised).** For a layer W from input space (action ρ) to hidden units (action P = permutation σ), the projection is W_eq = (1/|G|) Σ_g P(g) W ρ(g)^T. Biases and the output layer are projected the same way. For |G| = 2 this is exactly the existing Mirror Surgery.
- **Breaking share:** ||W - W_eq||² / ||W||². Also report the activation-weighted version ||(W - W_eq) Σ^(1/2)||² / ||W Σ^(1/2)||², where Σ is the input activation covariance on the fit set.
- **Null:** the same quantities for 20 random draws of an orthogonal action with the same character (same eigenvalue multiplicities) and a random permutation with the same cycle type. Report the mean and the 95% interval.
- **Character formula (the law).** A layer that respects the group has free parameters
  `free = (1/|G|) Σ_g tr(ρ(g)) · fix(σ(g))`,
  where fix(σ(g)) counts the hidden units that g leaves in place. The bias contributes the number of hidden orbits. The tax is 1 - free / (d · h).
  - For LR this reduces to the existing (h·d + f(d₊ - d₋))/2.
  - The formula itself is textbook representation theory. Its empirical content is that trained networks sit close to this subspace (low breaking share) only for the group they were trained with. The paper says so plainly.

### 4.5 Methods
- **Mirror-exact (main):** one shared operator A* over the converted span. For a span starting at block 1 this includes the patch embedding (flipped kernels), position embeddings (A* pos(r, c) = pos(r, G-1-c)), CLS (even), registers (matched among themselves), LayerNorm (A* fixes the all-ones direction; γ and β are folded into the next weights with tied biases) and LayerScale (folded into the adjacent weights). Tokens live in the A* basis inside the span, so the basis change is paid once at each end of the span (zero ends when the whole model is converted).
- **Mirror-local:** one operator per layer; ablation only. Its per-block basis changes are counted.
- **Final feature:**
  - invariant use: the even part of the final CLS;
  - equivariant use: the full CLS and patch tokens.
  Report both where relevant.

### 4.6 Evaluation (unchanged unless stated)
- **k-NN top-1:** the 100k bank; queries are the 50k validation images; CLS after the final norm; cosine; k = 20; T = 0.07.
- **Flip consistency:** the share of validation images whose prediction equals the prediction for the mirrored image. Also report the maximum absolute feature difference.
- **Tier-2 tasks:**
  - ImageNet linear probe on [CLS, mean patch token], trained on the bank;
  - transfer linear probes on CIFAR-100, Flowers-102, Pets and DTD;
  - ADE20K linear segmentation: frozen backbone, linear head on last-layer patch tokens, 448 x 448 crops, 8,000 iterations, batch 16, mIoU.
- **Cost:** parameters (whole model and converted span, including every basis change), analytic FLOPs including rotations, peak memory, and latency (batch 1, 64 and 256; bf16; torch.compile; median of 100 runs after warm-up). Each is reported separately; none stands in for another.

### 4.7 Matched budgets
Every compressor matches Mirror-exact's parameter count over the same converted matrices to within 1%, with all basis changes counted. Where a method cannot hit the budget exactly, round toward giving the baseline more parameters, and print the gap.

### 4.8 Healing recipe (identical for every method)
- **Teacher and student:** the teacher is the original model; the student is the compressed model, trained only in its own structured parameters:
  - Mirror: shared parameters, so exact symmetry survives every step;
  - SVD: the low-rank factors;
  - slicing: the sliced weights;
  - pruning: the surviving weights;
  - merging: the merged weights.
- **Loss:** normalised MSE on the final CLS and patch tokens, plus the outputs of the converted blocks.
- **Data and schedule:** the heal set, random resized crop with flips, 2 epochs, batch 256, AdamW, cosine schedule, bf16.
- **Learning rate:** chosen per method from {1e-5, 5e-5, 1e-4} on the LR-selection set.
- **Logging:** accuracy at steps 0, 100, 300 and 780.

### 4.9 Statistics
- **Zero-shot headline numbers:** 3 fit seeds, mean ± std.
- **Healed headline numbers:** 3 independent training seeds, mean ± std.
- **Differences between methods on the same images:** paired bootstrap 95% interval (10,000 resamples over validation images).
- Always report n.

---

## 5. Phase 1: MUST (without these the paper is not a safe accept)

### MUST-00. Sync and audit (CPU)
Confirm every number in Section 3 against `results/`, list the file behind each, and flag mismatches. Confirm the model IDs and recipes in `configs/models/`. **GPU-h: 0.**

### MUST-01. The law, written and checked on existing data (CPU)
- **Claim:** 2.
- **Reviewer question:** "Is the 42% a number you fitted, or a law?"
- **Method:** implement the character formula of Section 4.4. Unit-test it on a synthetic network built to be exactly equivariant for LR, R90 and the Klein group (the formula must equal a brute-force count of free parameters). Then compute the predicted tax per layer for all 7 measured models from their fitted operators and fixed-unit counts, and compare it with the parameter saving surgery actually achieved.
- **Prediction:** the predicted and achieved savings agree to within 1 percentage point per layer. This is an identity when the code is right, so it is a correctness check, not a finding.
- **Outputs:** `results/MUST-01/`; Fig. 2 inset.
- **GPU-h: 0.**

### MUST-02. Exactness proposition and per-component audit
- **Claim:** 5.
- **Reviewer question:** "Is exact equivariance proven or only measured? Do GELU, LayerNorm and softmax really commute with your mirror?"
- **Method:**
  1. Write the proposition and proof sketch (`docs/exactness.md`).
  2. For each component of Mirror-exact, state the condition and why it holds:
     - patch embedding;
     - position embeddings;
     - CLS and registers;
     - LayerNorm (A* fixes the all-ones vector; γ and β are folded; tied biases);
     - LayerScale (folded);
     - Q and K (the same per-head orthogonal map, so attention scores are preserved; softmax commutes with the token permutation);
     - V and O;
     - head permutation;
     - MLP: fc1 tied rows, tied biases, GELU acting on permuted units, fc2 tied columns;
     - residual addition (the same A* on both branches);
     - final norm and even projection.
  3. On 1,000 validation images in fp32, measure the maximum relative error of the equivariance relation at the output of each component in every block.
- **Prediction:** every component below 1e-5 in fp32, and 100% flip consistency.
- **Outputs:** Appendix table; `results/MUST-02/`.
- **GPU-h: 0.15.**

### MUST-03. Generalised measurement code and tests
- **Purpose:** the UD, R90 and Klein measurement code that MUST-04, MUST-12 and CORE-01 need.
- **Method:** implement Section 4.4 for all groups.
- **Tests:**
  - on synthetic exactly-equivariant networks: zero breaking share and the correct orbit counts;
  - on random networks: breaking share at the null level;
  - on DINOv2-B-reg with LR: reproduces the existing LR numbers.
- **GPU-h: 0.1.**

### MUST-04. Wrong-symmetry controls on released models
- **Claims:** 1 and 3.
- **Reviewer question:** "Is this a generic redundancy that any transformation would reveal?"
- **Method:** for DINOv2-B-reg, DINOv2-B, DINOv2-L-reg and DeiT-III, measure UD, R90 and R180 exactly as LR was measured (operator fit, twins or orbits, breaking shares against the null, all layers, fit seed 0). Run zero-shot group-projection surgery for UD on blocks 1 to 6 as a destructive control.
- **Predictions:**
  - UD and R90 breaking shares in blocks 1 to 6 are at least 0.9 x their nulls, against LR's ratio of about 0.45;
  - UD twin fraction is within the null's 95% interval;
  - UD surgery on 6 blocks drops k-NN by more than 30 points, against 1.85 for LR.
- **Outputs:** Fig. 2 panel; `results/MUST-04/`.
- **GPU-h: 0.4.**

### MUST-05. Twins are working mirror pairs
- **Claim:** 1.
- **Reviewer question:** "Are twins real computational units, or an artefact of matching?"
- **Method:**
  1. **Tracking:** the correlation between neuron i on images and twin j on mirrored images at mirrored tokens. Compare with random pairs, pairs matched by weight cosine, and pairs matched by same-image activation correlation.
  2. **Switch-off test:** on 500 validation images, zero neuron i (after GELU, all tokens) on image x, and separately zero twin j on the mirrored image. Primary measure, at the output of block l+1 after re-aligning tokens: cosine between A_(l+1) Δ_i(x) and Δ_j(mirror x), plus the ratio of their norms. Secondary measure: the same at the final CLS. Use the 200 strongest pairs per block for blocks 1 to 6, against 200 random pairs.
  3. **Teaser:** for the 8 strongest pairs in blocks 1, 3 and 6, the top 16 activating patches from 50k training images, cropped at 42 x 42 around the token, for i and for j.
- **Predictions:**
  - switch-off median cosine at least 0.7 for twins and at most 0.2 for random pairs;
  - norm ratio between 0.8 and 1.25 for at least 70% of twin pairs.
- **Outputs:** Fig. 1; `results/MUST-05/`.
- **GPU-h: 0.5.**

### MUST-06. Visibility test: can ordinary redundancy detectors see twins?
- **Claim:** 4.
- **Reviewer question:** "Is this just low rank, or ordinary correlated neurons, in disguise?"
- **Method:** for twin pairs in blocks 1 to 6 of DINOv2-B-reg, DINOv2-L-reg and DeiT-III:
  1. same-token Pearson correlation of h_i and h_j on the same image, against their correlation on mirrored tokens (0.80);
  2. how often a twin is among a neuron's top 10 same-image correlated partners;
  3. singular value spectra and effective rank (entropy) of fc1 and fc2 before and after group projection.
- **Predictions:**
  - median same-token twin correlation at most 0.3;
  - a twin is in a neuron's top 10 same-image partners for at most 20% of pairs;
  - effective rank changes by at most 2% under projection.
- **If it fails:** the paper drops the word "invisible" and keeps "different axis", resting on CORE-03.
- **Outputs:** Fig. 4a; `results/MUST-06/`.
- **GPU-h: 0.1.**

### MUST-07. Zero-shot comparison against strong compressors and redundancy controls
- **Claims:** 4 and 5.
- **Reviewer question:** "You only beat plain SVD."
- **Methods, all at matched parameters (Section 4.7) over the same matrices:**
  1. Mirror-exact.
  2. Mirror-local (ablation).
  3. Plain truncated SVD.
  4. Activation-aware SVD in the SVD-LLM style (whitening by the Cholesky factor of the input activation covariance).
  5. ASVD-style activation scaling (if quick to add).
  6. SliceGPT-style PCA slicing of the residual stream over the span. Fold the LayerNorm mean subtraction and LayerScale first; follow the SliceGPT LayerNorm recipe (check the paper).
  7. Structured MLP width pruning, FLAP-style: activation-fluctuation importance with bias compensation.
  8. Correlation merging, ZipIt-style within one model: merge the neurons most correlated on the same image.
  9. Pairing by same-image activation correlation, tied with the fitted operator.
  10. Pairing by weight cosine (done; recompute under the corrected budget).
  11. Random pairing (done; recompute).
  12. Random structure: random involution with matched d± plus random pairing.
- **Sizes:** 6 blocks MLP only; 6 blocks MLP plus attention; 9 blocks; full model.
- **Models:** DINOv2-B-reg (all sizes), DINOv2-L-reg and DeiT-III (6 blocks and full model).
- **Selection:** the best baseline is chosen by k-NN on the method-selection set, at the full-model size, on DINOv2-B-reg.
- **Predictions:**
  - On DINOv2-B-reg at 6 blocks, Mirror-exact beats every baseline on validation k-NN by at least 2 points, paired bootstrap interval excluding 0.
  - Methods 8 and 9 trail Mirror by at least 10 points at 6 blocks.
  - On DINOv2-L and DeiT-III, no prediction of superiority; report as measured.
- **Outputs:** Fig. 5 (zero-shot curves), Table 3 rows; `results/MUST-07/`.
- **GPU-h: 1.5.**

### MUST-08. Matched component ablation
- **Claim:** 5.
- **Reviewer question:** "Where does the redundancy live, and does converting attention cost accuracy?"
- **Method:** Mirror-exact on blocks 1 to 6 and on the full model: MLP only, attention only, both. Report parameters, FLOPs, k-NN and flip consistency. This replaces the invalid 77.09 against 77.37 comparison.
- **Prediction:** the MLP carries most of the saving; adding attention costs at most 0.5 points at 6 blocks.
- **GPU-h: 0.3.**

### MUST-09. Healed full-model comparison, identical recipe
- **Claim:** 5.
- **Reviewer questions:** "One lucky seed." "SVD just needed training." "Healing favours you."
- **Runs (full model, matched parameters, Section 4.8):**

| Method | Seeds |
|---|---|
| Mirror-exact | 3 (0, 1, 2) |
| Best baseline from MUST-07 | 3 |
| Second-best baseline from MUST-07 | 1 |
| Plain SVD | 1; raise to 3 if within 1 point of Mirror |
| Random structure | 1 |

- **Report:** accuracy at steps 0, 100, 300 and 780; final k-NN, flip consistency, maximum feature difference under flip, parameters, FLOPs, peak memory, latency.
- **Predictions:**
  - Mirror-exact is non-inferior to the best baseline: the difference in seed means is above -0.5 points;
  - only Mirror-exact has 100% flip consistency;
  - random structure trails Mirror by at least 3 points.
- **Outputs:** Table 1; recovery-curve figure; `results/MUST-09/`.
- **GPU-h: about 4.5** (about 0.5 per heal; time the first one).

### MUST-10. Against test-time flip averaging
- **Claim:** 5.
- **Reviewer question:** "Exact invariance is trivial: average the predictions for an image and its mirror."
- **Method:** the original model with the features of x and mirror(x) averaged (two passes, all parameters), against healed Mirror-exact (one pass, 76.5% of parameters, even part). Compare k-NN, linear probe, FLOPs, latency and memory.
- **Prediction:** Mirror-exact is within 1 point of flip-averaging accuracy at under 45% of its FLOPs.
- **GPU-h: 0.2.**

### MUST-11. Downstream tasks for the healed full models
- **Claim:** 5.
- **Reviewer question:** "k-NN alone is not enough; do the features survive?"
- **Models:** the original, healed Mirror-exact (all 3 seeds for linear probe and k-NN; seed 0 for the rest), and healed best baseline (seed 0).
- **Tasks:**
  - ImageNet linear probe;
  - the 4 transfer probes;
  - ADE20K mIoU;
  - mask equivariance: the share of pixels where mask(mirror x) equals mirror(mask x). Report it for the raw linear head and for a symmetrised head (head(z) + head(A* z))/2, which is exactly equivariant on Mirror-exact.
- **Predictions:**
  - Mirror-exact is within 1.0 point of the original on linear probe and ADE20K;
  - Mirror-exact with the symmetrised head gives 100% mask equivariance; the original stays below 100%.
- **Outputs:** Table 2; `results/MUST-11/`.
- **GPU-h: 1.2.**

### MUST-12. Controlled-training pilot (gate for CORE-01)
- **Claim:** 3.
- **Method:** ViT-S/8 on ImageNet-100 at 64 x 64 (Section 6, CORE-01 recipe), seed 0, two conditions: none and LR. Time the run. Measure LR and UD structure at the end.
- **Gate C (Section 8):** in the LR model, LR breaking share in blocks 1 to 6 is at most 0.8 x null and the twin fraction is at least 0.2.
- **GPU-h: 0.4.**

**Phase 1 total: about 9.4 GPU-hours.**

---

## 6. Phase 2: CORE (what makes the paper spectacular)

### CORE-01. Controlled training: the training distribution creates the tax (the centrepiece)
- **Claim:** 3.
- **Reviewer questions:** "Correlation is not cause." "Augmentation is known to raise equivariance (Bruintjes et al. 2023); what is new?"
- **Why upside-down flips are the clean lever:** natural images are already roughly left-right symmetric, so LR twins may appear without any augmentation (Bouchacourt et al. 2021 found that much invariance comes from the data itself). Upside-down images are rare in natural data, so UD twins can only come from training.
- **Recipe (identical for every run; only the augmentation changes):**
  - ViT-S/8 at 64 x 64 (8 x 8 patches plus CLS), learned position embeddings, no registers;
  - supervised cross-entropy with label smoothing 0.1;
  - AdamW, lr 1e-3 at batch 512, weight decay 0.05, 10 warm-up epochs, cosine schedule;
  - 100 epochs (reduce to 60 for every run if the MUST-12 run takes over 25 minutes; fix this before the grid);
  - random resized crop to 64 from the stored 72 x 72 images (scale 0.35 to 1);
  - no colour jitter, mixup or cutmix;
  - bf16 and torch.compile;
  - augmentation done on the GPU from uint8 arrays, so data loading does not bottleneck.
- **Conditions:**

| Run | Augmentation | Seeds |
|---|---|---|
| none | none | 3 |
| LR | left-right flip, p = 0.5 | 3 |
| UD | upside-down flip, p = 0.5 | 3 |
| LR+UD | both, independently, p = 0.5 each (Klein group) | 2 |
| R90 | uniform over 0, 90, 180 and 270 degrees | 2 |

- **Checkpoints:** epochs 0, 2, 5, 10, 25, 50 and 100 for every run.
- **Measured on every checkpoint, for every group (LR, UD, R90, Klein):** operator fit, twin and orbit fractions, breaking shares (fc1, fc2, attention) against nulls, the predicted tax, and validation accuracy on original, LR-flipped, UD-flipped and rotated validation images.
- **Predicted structure matrix (final checkpoint, blocks 1 to 6):**

| Trained with | LR twins | UD twins | R90 orbits of 4 |
|---|---|---|---|
| none | above null, below the LR run | at null | at null |
| LR | high | at null | at null |
| UD | above null (from the data) | high | at null |
| LR+UD | high | high | at null (Klein orbits of 4 instead) |
| R90 | above null | at null | high |

- **Pre-registered tests:**
  1. **Dissociation:** UD twin fraction in UD runs minus UD twin fraction in LR runs is at least 0.2 (absolute), in all 3 seed pairs.
  2. **Specificity:** UD breaking share is at most 0.7 x null in UD runs, and at least 0.9 x null in none and LR runs.
  3. **Rotation:** the share of hidden units in 4-cycles in R90 runs is at least 0.3, and at least 3x the share in none runs.
  4. **Emergence:** in LR and UD runs, the twin fraction for the trained group rises over training, and is higher at epoch 100 than at epoch 5 in every seed.
- **If test 1 fails:** the cause claim drops (Section 11).
- **Outputs:** Fig. 3 (matrix heatmap and emergence curves); `results/CORE-01/`.
- **GPU-h: about 4.0** (13 runs plus measurement; run 2 to 3 at a time).

### CORE-02. The law predicts the refund for new groups (predict, then verify)
- **Claim:** 2.
- **Reviewer question:** "The character formula is textbook."
- **Method:**
  1. Before any surgery, write the predicted tax for the LR+UD and R90 models into the results file, with its timestamp.
  2. Then run zero-shot group-projection surgery on blocks 1 to 6 for the trained group and for every wrong group, on the final checkpoints.
- **Predictions:**
  - the achieved saving equals the prediction (identity check);
  - trained-group surgery keeps accuracy within 3 points;
  - every wrong-group surgery loses at least 20 points.
- **GPU-h: 0.3.**

### CORE-03. Two axes of parsimony: Mirror and low rank stack
- **Claim:** 4.
- **Reviewer question:** "If they are different kinds of redundancy, show it."
- **Method:** inside Mirror-exact, apply truncated SVD separately to the even block and the odd block of each projected matrix, at several ranks. Build full accuracy-against-parameter curves at 4 budget points (about 85%, 70%, 55% and 40% of the converted span) for: low rank alone (the best of plain SVD and SVD-LLM), Mirror alone (one point, its tax), and Mirror plus low rank. Run zero-shot on DINOv2-B-reg, DINOv2-L-reg and DeiT-III, plus one healed point at 55% on DINOv2-B-reg.
- **Prediction:** Mirror plus low rank lies above low rank alone at at least 3 of 4 budget points on at least 2 of 3 models, including one of DINOv2-L or DeiT-III where SVD alone is competitive.
- **Outputs:** Fig. 4b; `results/CORE-03/`.
- **GPU-h: 1.0.**

### CORE-04. Generality, scale and training recipes
- **Claims:** 1 and 6.
- **Reviewer questions:** "Only DINOv2-B?" "Does it grow or shrink with size?" "Is the flip in the recipe really what matters?"
- **Method:**
  - For every model in Section 4.2: LR structure (operator fit, twin fraction, breaking share against null, tax by depth), plus UD structure as a control.
  - For every ViT with LR breaking share at most 0.8 x null in blocks 1 to L/2: zero-shot Mirror-exact surgery on the first half of the blocks against plain SVD and the best MUST-07 baseline. MAE uses the linear probe as its accuracy measure.
  - CNNs: structure only, on the pointwise MLP layers of ConvNeXt and the 1 x 1 convolutions of ResNet-50.
  - DINOv2 S, B, L and g: report the tax against width, and add their original k-NN under the paper protocol to the accuracy-against-parameter plot, so the refunded B model can be placed against the released family.
- **Predictions:**
  - **Scale:** the tax in blocks 1 to L/2 increases from S to g. If it falls instead, report it; Bruintjes et al. (2023) found smaller models learn more equivariance, so either direction is informative.
  - **Recipe:** models whose recipe is verified to have no flips have a lower LR tax than every flip-trained model of the same size.
  - **CNNs:** the twin structure appears in ConvNeXt's pointwise MLPs.
- **Outputs:** Fig. 2 (all models), Table 4; `results/CORE-04/`.
- **GPU-h: 1.5.**

### CORE-05. Why surgery is cheap and why SVD sometimes collapses
- **Claims:** 3 and 4.
- **Reviewer question:** "Mirror throws away more weight energy than SVD yet loses less. Why? And why does plain SVD collapse only on DINOv2-B?"
- **Hypothesis:** symmetric training data symmetrises exactly the weights it uses. The leftover asymmetry is noise from the random starting weights, in directions the data rarely visits. SVD instead removes directions the next layers rely on whenever the weight spectrum is flat.
- **Method:**
  1. For all MUST-07 models and all converted blocks, compare the weight-energy breaking share with the activation-weighted breaking share.
  2. Split the breaking energy into three bands of input directions by activation variance (top 10%, middle 80%, bottom 10%): for the released models, and across CORE-01 checkpoints for the LR and UD runs.
  3. Measure the flatness of each model's weight and activation spectra (effective rank) and relate it to plain SVD's zero-shot loss.
  4. Out-of-sample rule: the activation-weighted rule, "Mirror beats SVD at equal parameters when the activation-weighted breaking share is below the activation-weighted energy SVD discards", is designed on DINOv2-B-reg only and scored on DINOv2-L-reg, DeiT-III and MAE. Report it next to the failed F4 rule.
- **Predictions:**
  - the activation-weighted breaking share is at most half the weight-energy one in blocks 1 to 6;
  - over training, the breaking energy in the top-variance band falls faster than in the bottom band;
  - the out-of-sample rule is correct on at least 75% of layer-model cases.
- **Outputs:** Fig. 4c; `results/CORE-05/`.
- **GPU-h: 0.3.**

### CORE-06. "What" against "which way": orientation is kept, not destroyed
- **Claim:** 5.
- **Reviewer question:** "Exact invariance throws away orientation information."
- **Method:** on CUB-200-2011, define the direction a bird faces from the part annotations: the sign of (beak x minus tail x) when both are visible and they are more than 10% of the box width apart (check the part IDs). On the original model (even and odd parts via the final-layer operator) and on healed Mirror-exact, fit logistic-regression probes for species and for facing direction on the even part, the odd part and the full CLS. Report on the natural test images, and on a test set that adds every image's mirror with its facing label flipped.
- **Predictions:**
  - species from the even part is within 1 point of the full feature;
  - facing direction from the odd part is at least 85%;
  - facing direction from Mirror-exact's even part is at most 60% on natural images, and exactly 50% on the mirror-augmented test set (by construction).
- **If the final odd part is too small to carry facing direction:** use the odd part of the mean patch token, and say so.
- **Outputs:** small figure or table; `results/CORE-06/`.
- **GPU-h: 0.3.**

### CORE-07. Distribution shift
- **Claim:** 5.
- **Reviewer question:** "Does surgery hurt robustness?"
- **Method:** ImageNet-V2 (matched frequency) with the ImageNet linear probe, for the original, healed Mirror-exact and the healed best baseline. Add ImageNet-Sketch if disk space allows.
- **Prediction:** Mirror-exact's drop under shift is within 1 point of the original's drop.
- **GPU-h: 0.2.**

**Phase 2 total: about 7.6 GPU-hours.**

---

## 7. Phase 3: UPSIDE (cheap, could open something new)

### UP-01. Dose response
UD flip at p = 0.1 and p = 0.25, 1 seed each, CORE-01 recipe.
- **Prediction:** UD tax rises with p.
- **GPU-h: 0.6.**

### UP-02. Symmetric start
One UD run whose weights start exactly UD-equivariant: initial weights projected by Section 4.4 with a fixed canonical operator.
- **Prediction:** the breaking share stays below 0.05 throughout training, at accuracy within 1 point of the UD runs. This would show the symmetric network was reachable all along. Relevant theory: training on symmetric data keeps symmetric parameters symmetric (Nordenfors et al. 2024; Gerken and Kessel 2024; check the exact statements).
- **GPU-h: 0.35.**

### UP-03. Width, causally
ViT-Ti/8 with none, LR and UD, 1 seed each.
- **Prediction:** the tax for the trained group is higher in ViT-S than in ViT-Ti (or the reverse; either is reported and connects to CORE-04's scale result).
- **GPU-h: 0.5.**

### UP-04. Tied from scratch
A ViT-S/8 trained with UD flips with weights constrained to UD-equivariant form from the start (shared parameters only). Compare its accuracy and parameter count with the unconstrained UD runs and with UD runs refunded by surgery and healed for 1 epoch.
- **Prediction:** about the same accuracy at the reduced parameter count. Prior work on equivariant ViTs from scratch exists (Flopping for FLOPs; check the venue), so this is support, not a claim.
- **GPU-h: 0.35.**

### UP-05. Speed attempt
Twin units stored in a pre-permuted layout so no shuffle is needed at run time; tokens kept in the A* basis for the whole model; the butterfly (u + v, u - v) as one fused operation. Time it at batch 1, 64 and 256.
- **Report:** whatever it gives. Above 1.1x the original's speed counts as a speed claim; otherwise it is a limitation. Memory and parameter savings are reported regardless.
- **GPU-h: 0.3.**

### UP-06. Stacking with other compression axes
- INT8 weight-only per-channel quantisation of healed Mirror-exact and of the original.
- ToMe token merging (r = 8) on both.
- **Prediction:** the extra accuracy drop from each is the same for both models to within 0.5 points, so Mirror composes with quantisation and token merging.
- **GPU-h: 0.3.**

### UP-07. Pairing from weights alone, second attempt (CPU)
Match fc1 rows against fc1 rows mapped by the weights-only operator, jointly with the fc2 columns, under the involution constraint.
- **Bar:** at least 70% agreement with the data-driven pairing (the same bar as before). If it passes, add a short data-free paragraph; if not, report both attempts.
- **GPU-h: 0.05.**

### UP-08. Higher-resolution replication (only if extra compute is bought)
CORE-01's none, LR and UD conditions at 128 x 128 with ViT-S/16, 2 seeds each. This removes the "toy scale" objection to the centrepiece.
- **GPU-h: about 8 to 10** (estimate; time one epoch first).

**Phase 3 total (without UP-08): about 2.5 GPU-hours.**

---

## 8. Gates (stop and report to Dawar)

| Gate | When | Condition | If it fails |
|---|---|---|---|
| A | Before any new GPU job | PREREG_NEXT_PHASE.md committed; MUST-00 clean | Fix first |
| B | After MUST-07 | Best baseline chosen on the method-selection set and recorded | Do not start MUST-09 until recorded |
| C | After MUST-12 | LR model: LR breaking share at most 0.8 x null in blocks 1 to 6 and twin fraction at least 0.2 | Report. Proposed fix: rerun the pilot at 96 x 96 with ViT-S/16 before the grid; decide with Dawar |
| D | After MUST-09 | Mirror-exact mean at least best-baseline mean minus 1.0 | Keep running everything else; discuss framing (Section 11) |
| E | Before abstract registration (23 Nov) | MUST-04, MUST-05, MUST-07, MUST-12 done; CORE-01 running | Freeze title and abstract with Dawar from what is in hand |

---

## 9. Compute, concurrency and order

**Estimates (single-job H100-hours; time the first job of each kind and re-plan):**

| Phase | GPU-h |
|---|---|
| Phase 1, Must | about 9.4 |
| Phase 2, Core | about 7.6 |
| Phase 3, Upside (without UP-08) | about 2.5 |
| Summed | about 19.5 |

The working limit is about 18 H100-hours. Heals use about 5 to 6 GB, ViT-S/8 training about 10 GB, and measurement jobs are small. So run 2 to 3 jobs at once, mainly MUST-09 heals alongside CORE-01 runs. Keep GPU utilisation above 85% and leave 8 GB of headroom. With that overlap, the expected wall time on one H100 is about 15 to 17 hours.

**Order:**
1. CPU: MUST-00, MUST-01, PREREG_NEXT_PHASE.md, CUB part IDs, model recipes, ImageNet-100 preprocessing, novelty search.
2. Short decisive jobs: MUST-03, MUST-02, MUST-04, MUST-05, MUST-06, MUST-07 (then gate B), MUST-08, MUST-10, MUST-12 (then gate C).
3. Long jobs together: MUST-09 heals and CORE-01 runs.
4. On finished models: MUST-11, CORE-02, CORE-03, CORE-04, CORE-05, CORE-06, CORE-07.
5. Upside, in order: UP-02, UP-01, UP-06, UP-03, UP-04, UP-05, UP-07.

**Cut order if timings run long:** UP-05, UP-07, UP-03, UP-04, UP-06, UP-01, the second-best baseline heal in MUST-09, ImageNet-Sketch, the CNNs in CORE-04, CORE-04's optional models (DINOv3, EVA-02).

**Never cut:**
- MUST-02, MUST-04, MUST-05, MUST-07;
- the Mirror and best-baseline seeds in MUST-09;
- MUST-11's linear probe and ADE20K;
- MUST-12;
- CORE-01's none, LR and UD conditions;
- CORE-03 zero-shot.

---

## 10. Reviewer questions and where the paper answers them

The reviewer lenses follow CPAL's subject areas: low-dimensional structure; symmetry and equivariance; pruning, quantisation and distillation; implicit bias and learning dynamics; neuroscience-inspired parsimony; and evaluation methodology.

| # | Likely question | Answer in the paper | Experiments |
|---|---|---|---|
| 1 | "Mirrored neuron copies are known (Olah et al. 2020, naturally occurring equivariance; Lenc and Vedaldi 2015; Bökman and Kahl 2023)." | Yes, qualitatively and in CNNs. New here: the count in foundation ViTs with a law, the cause-and-effect split, invisibility to rank compression with stacking, and an exact refund on released models. Cite all three prominently | MUST-04, CORE-01, MUST-06, CORE-03 |
| 2 | "Augmentation raising learned equivariance is known (Bruintjes et al. 2023; Bouchacourt et al. 2021)." | We measure parameter-level structure and its specificity (the UD dissociation), not equivariance error, and connect it to a compressible parameter count | CORE-01, MUST-04 |
| 3 | "The formula is textbook." | Stated as such. The finding is that trained networks sit near this subspace only for the trained group, and the formula predicts refunds before surgery for groups not yet tested | MUST-01, CORE-02 |
| 4 | "This is low rank in disguise." | Twins are not same-image correlated, projection does not reduce rank, and the two stack | MUST-06, CORE-03 |
| 5 | "Weak baselines." | 12 methods at matched parameters, including SVD-LLM, SliceGPT, FLAP-style pruning and within-model merging | MUST-07, MUST-09 |
| 6 | "The SVD collapse looks like a bug or a one-off." | Bug check passed (G1). Shown on three models, and explained by spectral flatness | MUST-07, CORE-05 |
| 7 | "Parameter matching is unfair." | Every basis change counted; Mirror-exact as the main method; gaps printed | Section 4.7, MUST-07 |
| 8 | "Healing favours your method." | One recipe, per-method learning-rate selection on a held-out split, recovery curves from step 0, zero-shot reported alongside | MUST-09 |
| 9 | "Single seeds." | 3 fit seeds zero-shot, 3 training seeds healed, paired bootstrap | Section 4.9 |
| 10 | "Exact invariance is trivial by flip averaging." | Same guarantee at under half the compute and three quarters of the parameters | MUST-10 |
| 11 | "Invariance destroys orientation information." | The global feature can be invariant (even part) or equivariant (full); orientation sits in the odd part; dense outputs are exactly equivariant | CORE-06, MUST-11 |
| 12 | "Is exactness proven?" | A proposition, plus a per-component numerical audit at every block | MUST-02 |
| 13 | "No speed-up, so no practical value." | Parameter and memory savings, an exact guarantee, composition with quantisation and token merging; speed reported honestly | MUST-09, UP-05, UP-06 |
| 14 | "The controlled study is toy scale." | Same prediction confirmed in released models (MUST-04, CORE-04 recipe test); higher-resolution replication if compute allows | CORE-04, UP-08 |
| 15 | "Cherry-picked layers or models." | Pre-registered thresholds; every model reported; failed checks reported | PREREG, Section 3 |
| 16 | "Why do late layers lose it?" | Artefact tokens; registers restore it (pre-registered pass) | Done |
| 17 | "Does surgery hurt robustness?" | ImageNet-V2 (and Sketch) | CORE-07 |
| 18 | "Twins are an artefact of the matching procedure." | Nulls with matched character, wrong-group controls, the switch-off test | MUST-04, MUST-05 |
| 19 | "Only self-supervised ViT-B." | 10 or more models: self-supervised, supervised, masked, language-supervised, CNN; S to g | CORE-04 |
| 20 | "Why not train an equivariant model from scratch?" | That costs the whole pre-training; the refund applies to released checkpoints with 100 images and a short heal. Small-scale tied-from-scratch arm included | UP-04 |
| 21 | "What is the parsimony principle?" | Two axes, rank and symmetry; in description-length terms, a symmetric layer needs only its free parameters, which the law counts | Claims 2 and 4 |
| 22 | "How much data is needed?" | 100 images; the operator is readable from weights alone; pairing from weights is reported as a failure (or a pass, if UP-07 succeeds) | Done, UP-07 |

---

## 11. Fallback framings (decided in advance)

| If | Then the paper says |
|---|---|
| CORE-01 test 1 fails (no UD twins after UD training) | The cause claim drops. The paper rests on discovery, law, invisibility and refund, and reports the controlled study as a negative result |
| LR twins appear without augmentation, but UD twins only with UD training | "The symmetry of the training distribution creates the tax"; augmentation is one way to make the distribution symmetric |
| MUST-06 fails (twins correlated on the same image) | Drop "invisible"; keep "different axis" on CORE-03 |
| CORE-03 fails | Soften to "different structure from low rank"; keep MUST-06 |
| Gate D fails (best baseline beats Mirror after healing) | Headline: exact equivariance, zero-shot strength, 100 images. "Best compressor" was never the claim |
| CORE-04 scale goes the other way | Report "smaller models pay less", connected to Bruintjes et al. |

---

## 12. Deliverables

- One folder per experiment, `results/<ID>/`, with `metrics.json`, CSV tables, plots and `meta.json` (commit, prereg hash, config, seeds, model revisions, GPU, wall time).
- `FINAL_RESULTS.md`: every pre-registered prediction marked pass or fail with its number, plus every number the paper will cite and its source file.
- `PAPER_NUMBERS.json`: the same numbers in machine-readable form.
- Figures and tables built by scripts only:

| Item | Content | Source |
|---|---|---|
| Fig. 1 | Teaser: twin patches; one-line tax | MUST-05 |
| Fig. 2 | Tax by depth across models and sizes; wrong-group controls; artefact inset | MUST-04, CORE-04, done |
| Fig. 3 | Controlled training: structure matrix and emergence curves | CORE-01 |
| Fig. 4 | (a) visibility; (b) stacking curves; (c) why surgery is cheap | MUST-06, CORE-03, CORE-05 |
| Fig. 5 | Accuracy against parameters: all methods, zero-shot and healed, with the DINOv2 family | MUST-07, MUST-09, CORE-04 |
| Table 1 | Healed full-model comparison | MUST-09, MUST-10 |
| Table 2 | Downstream and robustness | MUST-11, CORE-07 |
| Table 3 | Ablations: operator, pairing, fit size, components, projection strength | Done, MUST-07, MUST-08 |
| Table 4 | Generality and the law across models and groups | CORE-02, CORE-04 |
| Appendix | Exactness audit; law derivation; pre-registration pass and fail list; failed checks (F4, weights-only pairing) | MUST-02, MUST-01, all |

---

## 13. Rules

- **No invented numbers.** Every number in the paper comes from a file in `results/`. Model IDs, recipes, citations and baseline details are checked or marked (verify).
- **Honesty:** failed pre-registered checks are reported in the paper, including F4 and the weights-only pairing.
- **Data hygiene:** fit, bank, heal, LR-selection, method-selection and validation sets stay disjoint, checked by image ID and logged. Validation is never used for any choice.
- **Reproducibility:** fixed seeds; every run writes `meta.json`; plots come from scripts.
- **GPU etiquette:** check `nvidia-smi` before each job, cap memory per process, keep 8 GB headroom, never kill processes this project did not start.
- **Secrets:** use a fresh Hugging Face token on the machine; never commit tokens, keys or weights.
- **Writing:** British spelling, plain language, no em dashes, no emoji.
