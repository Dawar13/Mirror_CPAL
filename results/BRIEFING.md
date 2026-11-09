# Briefing: The Symmetry Tax and Mirror Surgery (project Mirror_CPAL)

Prepared on 9 November 2026 from the repository `Mirror_CPAL`, for Dawar to upload into a claude.ai chat.

**Note for the Claude reading this file.** You have no access to the repository. This file is meant to be self-contained: every term is defined where it first appears (and again in the glossary, Section 13), and every number is copied from a file in the repository, with that file named in brackets, for example (results/STATUS.md). Where I did simple arithmetic on two quoted numbers, it is marked "(arithmetic)". Anything that is not established by the repository is marked "(verify)". Please do not add numbers or citations that are not here; if Dawar asks for one, say it needs checking against the repository. Paths are relative to the repository root.

**Target venue.** CPAL 2027 Proceedings Track, deadline 5 December 2026 AoE (PMLR, 9 pages); fallback is the Recent Spotlight Track on 18 January 2027 (PROJECT_CONTEXT.md). Planned working title: "The Symmetry Tax: Vision Foundation Models Learn Their Features Twice, and Mirror Surgery Refunds It" (PROJECT_CONTEXT.md).

---

## Contents

0. Four direct answers (the questions Dawar asked)
1. One-paragraph summary
2. The idea from scratch
3. What Mirror Surgery is, precisely
4. Why low rank fails where mirror works (claim C4)
5. What healing is
6. Is the paper about healing?
7. Why is healing "not working"?
8. Pre-registration and the verdict words
9. Full history with every statistic
10. Compute and logistics
11. Strong signals and weak or negative signals
12. What is left
13. Glossary
14. Suggested questions to ask

---

## 0. Four direct answers

**What did "E1 failed (KILL)" mean?** E1 was the first go/no-go experiment (7 November 2026). It tested one specific design: a *single* mirror operator shared by all 12 layers of plain DINOv2 ViT-B/14. Its decision table, fixed in advance, returned KILL because three rows landed in KILL cells: the shared operator was barely better than a random one (ratio 1.10 against a GO bar of 5), the MLP weights were no closer to symmetric under it than under a matched random structure (breaking share 0.307, matched random 0.307), and converting 6 blocks wrecked the model (k-NN top-1 from 78.21 to 14.10, while SVD at the same size kept 72.45) (results/E1_REPORT.md). The cause: layers 9 to 11 of plain DINOv2 do not line up under any mirror, and the shared fit summed unnormalised statistics over all layers, so those layers dominated and the shared operator collapsed to nearly the identity (8 odd directions out of 768) (results/E1_DIAGNOSTIC.md, PROJECT_CONTEXT.md). KILL applied to that design only. A per-layer version then passed every check in a pre-registered rerun (E1b, GO), and the project continued with one operator per layer (results/E1B_SUMMARY.md). Details in Section 9.2 to 9.4.

**What is healing?** A short fine-tune after surgery. The converted model (the "student") is trained for 2 epochs on 100,000 ImageNet training images to reproduce the original model's (the "teacher's") internal features block by block and its final features, while the symmetric structure is re-imposed after every training step so the parameter saving and the symmetry are kept (configs/m3_pilot.yaml, mirror/heal.py). Section 5.

**Is the whole paper about healing?** No. The paper is about a finding (flip-trained vision transformers store many features twice, as mirror twins, and this can be measured and counted: "the symmetry tax") and a method that removes the duplication from a released checkpoint with no training at all ("zero-shot Mirror Surgery"), including a fully converted model that is exactly mirror-symmetric. Healing is a standard supporting step that makes heavy conversion usable. Section 6.

**Why is healing "not working"?** Healing itself works very well: the fully converted model goes from 53.85 to 78.31 k-NN top-1 (original 78.94) (runs/m3_vexact/k12/SUMMARY.md). What failed is a pre-registered *comparison*: at 9 converted blocks, healed Mirror had to beat healed SVD by at least 3.0 points; it won by 1.24 (78.81 against 77.57) (results/M3_PILOT_G2.md). Healing rescues SVD too (from 6.49 to 77.57), and nobody can get much above the original model, so a 3-point lead was out of reach: Mirror would have needed 80.57 (arithmetic: 77.57 + 3.0), above the original's 78.94. Section 7.

---

## 1. One-paragraph summary

The project asks whether large vision transformers trained with left-right flip augmentation store many of their features twice, once for each mirror orientation, and whether that duplication can be removed from a released checkpoint without retraining. For the main model, DINOv2 ViT-B/14 with registers, the answer is yes in the first half of the network: in blocks 1 to 6 about 63% of MLP neurons sit in mirror-twin pairs and a closed-form count says 0.419 of those blocks' MLP weights are redundant (results/paper/tables/T_f1_ranges.md). "Mirror Surgery" ties each twin pair together so it is stored and computed once. With no retraining, converting the MLPs of the first 6 blocks keeps k-NN top-1 at 77.09 (original 78.94), while SVD at the same parameter count falls to 8.77 and a random-structure control to 3.95 (results/paper/tables/T_main_zeroshot.md). Converting all 12 blocks with one shared operator ("V-exact") gives a model that is exactly mirror-invariant (100% flip consistency in fp32) with 0.765 of the parameters; zero-shot it drops to 53.85, and a short distillation ("healing") brings it to 78.31, replicated at 78.34 on a second GPU (runs/m3_vexact/k12/SUMMARY.md, runs/m3_vexact_5090/k12/SUMMARY.md). Many pre-registered checks passed (artefact tokens P1, generality P4 for DeiT-III and MAE, structure beats random after healing P3.a), but three matter as negatives: the healing gate G2 failed on the SVD condition (margin 1.24, bar 3.0), the energy rule P2 failed (16 of 36 cases right, bar 75%), and weight-only twin pairing P5 part 2 failed (46.6%, bar 70%); also, the cheaper models are not faster in wall-clock time with plain PyTorch (measured 0.70x to 0.91x of the dense model's speed at batch 256) (results/STATUS.md). As of 9 November 2026 a job list is ready for a rented RTX 5090 in South Korea, with zero-shot strong baselines first (whether it has started is not recorded in the repository (verify)), and a new pre-registered full-conversion test ("E3") is proposed and waiting for Dawar's decision (jobs/stage22_korea.txt, results/STATUS.md).

**Key numbers at a glance** (registers model `dinov2_vitb14_reg` unless stated; k-NN top-1 in %, paper protocol)

| What | Number | Source |
|---|---|---|
| Original model, k-NN top-1 / linear probe | 78.94 / 80.99 | results/STATUS.md |
| Zero-shot, 6 blocks MLP only: Mirror / SVD / random | 77.09 / 8.77 / 3.95 | T_main_zeroshot.md |
| Same, plain DINOv2: Mirror / SVD / random (original 78.71) | 74.65 / 66.14 / 3.48 | T_main_zeroshot.md |
| Symmetry tax, blocks 1 to 6 (share of MLP weights removable) | 0.419 (plain 0.405) | T_f1_ranges.md |
| Healed, 9 blocks MLP + attention: Mirror / SVD / random | 78.81 / 77.57 / 73.28 | results/M3_PILOT_G2.md |
| V-exact full conversion: zero-shot / healed / fp32 flip consistency | 53.85 / 78.31 / 100% | runs/m3_vexact/k12/SUMMARY.md |
| V-exact full conversion, parameters kept (whole model) | 0.765 | runs/m3_vexact/k12/SUMMARY.md |
| Measured speed vs dense, batch 256 compiled, Mirror variants | 0.70x to 0.91x | results/STATUS.md |

(T_*.md files are in results/paper/tables/.)

---

## 2. The idea from scratch

### 2.1 Flip augmentation

When large vision models are trained, each training image is often shown either as it is or mirrored left to right, at random. This is called flip augmentation. It teaches the model that a left-facing dog and a right-facing dog are both dogs. The project's premise is that the models studied (DINOv2 and others) were trained this way (PROJECT_CONTEXT.md, Section 1); the exact augmentation of each model is not recorded in the repository (verify per model).

### 2.2 Why a network trained with flips learns many things twice: an everyday analogy

Imagine a glove factory with one drawing for the left glove and a separate drawing for the right glove, made by two teams who never compared notes. A right glove is just a mirrored left glove, so the two drawings end up as near mirror images of each other, yet both are stored in full. You could keep one drawing plus the instruction "mirror it" and lose almost nothing. The duplicated drawing is a tax you pay for storage; keeping one drawing and the rule is a refund.

A vision transformer trained with flips is in the same position. A detector for "dog's head turned left" is useful, and so is one for "dog's head turned right". Training learns both, separately, and because nothing forces them to share weights, the network stores two independent copies whose weights are (almost) mirror images of each other. The project calls this duplication **the symmetry tax**, and the method that removes it **Mirror Surgery** (it "refunds" the tax).

A second picture helps with "even" and "odd". Take a pair of twin detectors L (left-facing) and R (right-facing). Instead of storing L and R, store their sum S = L + R and their difference D = L - R. Under a mirror, L and R swap places, so S stays exactly the same (it is **even**) and D changes sign (it is **odd**). Anything that respects the mirror can be described using even parts and odd parts only, and that description has no duplication.

### 2.3 The precise version: the mirror operator A_l

Some vocabulary first.
- A **vision transformer (ViT)** cuts an image into small square patches (14 x 14 pixels for DINOv2), turns each patch into a vector called a **token**, and passes the tokens through a stack of **blocks** (12 for the ViT-B models here). Each block has an **attention** layer (tokens exchange information) and an **MLP** (a two-layer network applied to each token separately).
- The tokens are carried along a **residual stream**: each block reads the stream and adds its output back to it. Each token vector has width d = 768 in the ViT-B models; "layer l" means the stream after block l, and layer 0 is the stream entering block 1 (results/E1_REPORT.md, README.md).
- Besides patch tokens there is a **CLS token** (a summary token whose final value is used as the image's feature) and, in the "with registers" models, 4 **register tokens** (extra scratch tokens) (results/paper/tables/cost_model.md: 261 tokens per image = 256 patches plus 5 prefix tokens).

Now take an image and its mirror. Run both through the model. At layer l, let X be the token features of the original image (one row per token) and X~ the features of the mirrored image, with the token positions flipped back so that row i in both refers to the same piece of the scene. If the model were perfectly mirror-symmetric, there would be a fixed d x d matrix A_l with X~ = X A_l at every token. The project fits this matrix from data. It is constrained to be:
- **symmetric and orthogonal** (a rigid rotation or reflection of the feature space), and
- an **involution**, A_l^2 = I: mirroring twice gives back the original.

Such a matrix has only two possible eigenvalues, +1 and -1. The directions with eigenvalue +1 are the **even directions** (unchanged by a mirror; there are d+ of them) and those with -1 are the **odd directions** (sign-flipped; there are d- of them), with d+ + d- = d (PROJECT_CONTEXT.md, Section 1).

The fit has a closed form: sum the cross-moment C = sum over tokens of X^T X~, take its symmetric part (C + C^T)/2 = V Lambda V^T, and set A = V sign(Lambda) V^T. This is the exact best symmetric orthogonal involution in the least-squares sense (PROJECT_CONTEXT.md, Section 5.1).

How good is the fit? For the registers model at layer 1, the relative held-out error e(A_l) = ||X~ - X A_l||^2 / ||X~||^2 is 0.0328, against 0.8078 for doing nothing (the identity, A = I) and 1.6656 for a random involution with the same d+ (results/paper/tables/T_structure_per_layer.md, fit seed 0). The number of odd directions falls with depth: layer 1 has d+ 489 and d- 279, layer 6 has 640 and 128, layer 12 has 760 and 8 (same table).

### 2.4 Mirror-twin neurons

An MLP has a first weight matrix (fc1, from d = 768 to h = 3,072 hidden neurons), a GELU non-linearity, and a second matrix (fc2, back to 768). Each hidden neuron j has an input weight vector w_j (a row of fc1). Neuron j is the **mirror twin** of neuron i if, on a mirrored image, neuron i fires the way neuron j fires on the original (at mirrored positions). In weight terms this is w_j = A_l w_i: one neuron's weights are the mirror image of the other's (PROJECT_CONTEXT.md, Section 1). Neurons with no twin that pair with themselves are **self-twins**.

Twins are found by measuring, on 2,000 images and their mirrors, the correlation between each neuron's activation on mirrored images and every neuron's activation on originals, and then solving a maximum-weight matching (each neuron paired at most once) (E1_INSTRUCTIONS.md, A5). In block 1 of the registers model this gives 1,168 twin pairs and 736 self-twins (twin fraction 0.760), with mean twin correlation 0.804 (T_structure_per_layer.md, fit seed 0). The twin fraction falls with depth to 0.011 in block 12 (same table).

### 2.5 The symmetry tax, counted

If a weight matrix is exactly mirror-symmetric, it has fewer free numbers than a general matrix. For an h x d map the count of free parameters is (h d + tr(P) tr(A)) / 2, where tr(P) = f is the number of self-twins and tr(A) = d+ - d-. So the redundant share is 1/2 - f (d+ - d-) / (2 h d) (PROJECT_CONTEXT.md, Section 5.3). Why: a twin pair needs d numbers in place of 2d (a half saving), and a self-twin can only use the even directions (d+ numbers in place of d).

Measured tax (share of MLP weights that can be removed), mean over blocks (T_f1_ranges.md):

| Model | Blocks 1 to 6 (first half) | Blocks 7 to 12 (second half) |
|---|---|---|
| DINOv2-reg (3 fit seeds) | 0.419 | 0.109 |
| DINOv2 plain (3 fit seeds) | 0.405 | 0.095 |
| DeiT-III | 0.349 | 0.017 |
| MAE | 0.448 | 0.342 |
| SigLIP2 | 0.231 | 0.011 |
| DINOv2-S/14 reg | 0.400 | 0.127 |
| DINOv2-L/14 reg (24 blocks: 1 to 12, 13 to 24) | 0.414 | 0.064 |

The formula and the saving that surgery actually achieves agree to rounding level (largest difference 1.11e-16), but this checks bookkeeping rather than the theory, because both come from the same counts (results/paper/tables/ext_f1_dinov2_F1_SUMMARY.md).

---

## 3. What Mirror Surgery is, precisely

### 3.1 The steps

1. **Fold the norms.** Before touching the MLP, the LayerNorm scale and shift and the LayerScale factor are folded into the MLP weights so that the projection acts on the weights that really multiply the stream; the outputs are unchanged by this (E1_INSTRUCTIONS.md, A6; results/E1_REPORT.md: folding difference 6.68e-06). Residual tokens are channel-centred and the all-ones direction is assigned +1 so the operator commutes with LayerNorm (README.md).
2. **Fit the mirror operators** A_l from activations of 2,000 ImageNet training images and their mirrors (Section 2.3). The paper protocol uses three different "fit seeds" (three sets of 2,000 train images) (PAPER_EXPERIMENTS.md, Section 1).
3. **Pair twin neurons** in each MLP from the same images (Section 2.4).
4. **Project the weights onto the exactly mirror-symmetric form.** With P the neuron pairing (a permutation that swaps each twin pair and fixes self-twins), the projection is W_eq = (W + P W A)/2 for fc1, and similarly for fc2 and the biases (PROJECT_CONTEXT.md, Section 5.2). In words: each neuron's weight vector is replaced by the average of itself and the mirror image of its twin's weights, so afterwards the twins are exact mirror images. The share of weight energy removed by this averaging is the **breaking share** s = ||W - W_eq||^2 / ||W||^2: the part of the weights that broke the symmetry. For a random matrix it is about 0.5 (E1_INSTRUCTIONS.md, A9); the matched random structure (the "null") gives 0.427 on average over blocks 1 to 6 of the registers model (T_f1_ranges.md); for block 1 of the registers model it is 0.267 for fc1 (null 0.494) and 0.196 for fc2 (null 0.467) (T_structure_per_layer.md, fit seed 0).
5. **Compute each twin pair once.** In even/odd coordinates (the input split into its even part x+ and odd part x-), a twin pair has weights [a, b] and [a, -b]. Compute u = a . x+ and v = b . x- once, then the two neurons' pre-activations are u + v and u - v (a "butterfly"). A self-twin has b = 0 and reads only x+ (GELU is not an odd function, so a self-twin cannot use odd inputs) (PROJECT_CONTEXT.md, Section 5.2).

### 3.2 An important correction: surgery ties weights, it does not switch them off

It is easy to picture surgery as deleting or zeroing weights. It does not do that. It **ties** twin weights together: after projection, one neuron's weights are an exact, fixed function (the mirror) of its twin's, so the pair is stored once (as a and b) and computed once (u and v, then u + v and u - v). That tying is where the parameter saving comes from. What is genuinely thrown away is only the small symmetry-breaking difference between the twins (the breaking share), not whole neurons. Both neurons of a pair still exist and still fire.

### 3.3 Attention

Under a mirror, the 12 attention heads of a block either map to themselves ("self-heads") or swap in pairs ("twin heads") (PROJECT_CONTEXT.md, Section 5.4; this structure is discussed by Ong and Bokman, NeurIPS 2026, arXiv 2608.30417, listed in PROJECT_CONTEXT.md Section 6). Each head gets small internal maps (B_h for queries and keys, C_h for values), and the query, key, value and output weights are projected onto the symmetric form in the same averaging way. Measured on the registers model: 130 self-heads and 7 twin pairs over the 144 heads (12 blocks x 12 heads); block 1's attention breaking share is 0.283 against 0.494 for matched random structure (results/paper/INDEX.md, T_attention.md).

### 3.4 V-local and V-exact: two designs

The residual stream causes a subtlety. A block's output is added to its input. If layer l-1 uses operator A_(l-1) and layer l uses a different A_l, the skip path carries features that mirror one way while the block writes features that mirror another way, so the network as a whole is only approximately symmetric (PROJECT_CONTEXT.md, Section 5.5; E1B_PREREG.md, "Known limits").

- **V-local** (local operators): one operator per layer, as fitted. Most accurate per layer, only approximately mirror-equivariant overall, and an efficient implementation needs one d x d basis rotation at each block boundary (counted in the FLOPs). Can be applied to the MLP only, or to the MLP plus attention.
- **V-exact** (exact): one operator A* for the whole converted span of blocks, fitted with each layer's statistics normalised to equal energy (this normalisation is exactly what E1 lacked). Inside the span the model is exactly mirror-equivariant: the mirrored image's features equal A* times the original's, to within 1e-4 (measured 5.1e-06 to 7.6e-06) (T_attention.md). No basis rotations are needed. If the span starts at block 1, the patch embedding, position embeddings, CLS token and registers are projected too (PAPER_EXPERIMENTS.md, M1).
- **Full conversion** means V-exact over all 12 blocks (k = 12). An "even head" then keeps only the even part of the final feature, which A* leaves unchanged. Result: the model gives *the same answer for an image and its mirror, guaranteed by construction*. Measured: 100% flip consistency in fp32 with the even head (largest relative CLS difference 9.8e-06), against 96.22% without it (T_attention.md).

**Equivariant** means "transforms predictably" (mirror the input, the features change by A); **invariant** means "does not change at all" (mirror the input, the answer is identical). V-exact is equivariant inside the span; with full conversion and the even head, the output is invariant.

### 3.5 Zero-shot, k, and what is converted

- **Zero-shot**: surgery with no retraining at all; the projected weights are used as they are.
- **k**: the number of converted blocks, always the first k blocks (k = 3, 6, 9 or 12).
- "MLP only" converts only the MLPs of those blocks; "MLP + attention" converts both.

### 3.6 Baselines and controls

Every comparison is at a matched parameter count (within 1% of the converted blocks' count) (PAPER_EXPERIMENTS.md, Section 1).
- **SVD at matched parameters (low rank):** each weight matrix is replaced by its best rank-r approximation (two thin matrices), with r chosen so the parameter count matches Mirror's (E1_INSTRUCTIONS.md, A6). This is the standard simple compression baseline.
- **Random structure (control):** a random involution with the same d+ and d- as the fitted operator, plus a random pairing with the same numbers of pairs and self-twins, projected in exactly the same way (PAPER_EXPERIMENTS.md, M3-pilot). It has exactly the same shape and parameter count as Mirror Surgery, so if Mirror beats it, the *fitted* structure matters, not just the act of tying weights.
- **MLP width pruning at matched parameters:** appears in the latency measurements (M2) only (T_latency.md).
- **Strong baselines (code ready, not yet run):** activation-aware SVD ("SVD-LLM style whitening" in the code; SVD-LLM is not cited in PROJECT_CONTEXT.md (verify)), SliceGPT-style PCA slicing (SliceGPT, arXiv 2401.15024, listed in PROJECT_CONTEXT.md Section 6), and activation-weighted MLP width pruning (scripts/strong_baselines.py).

### 3.7 How models are scored

- **Tier 1 (every variant):** **k-NN top-1**: each validation image's CLS feature is compared (cosine similarity) with a memory bank of 100 labelled ImageNet train images per class (100,000), the 20 nearest vote (weighted, temperature 0.07), and top-1 accuracy is measured on all 50,000 validation images. Also **flip consistency** (the share of images whose predicted class equals that of their mirror) and parameter and FLOP counts (PAPER_EXPERIMENTS.md, Section 1). Features in bf16 for every variant alike (results/STATUS.md).
- **Tier 2 (final variants):** ImageNet **linear probe** (logistic regression on [CLS, mean patch token]), **transfer** linear probes on CIFAR-100, Flowers-102, Oxford-IIIT Pets and DTD, and **ADE20K** semantic segmentation with a linear head (mIoU), plus latency (PAPER_EXPERIMENTS.md, Section 1).
- **Protocol v1 versus v2:** E1 and E1b used ImageNet validation images for everything (fit 2,000; held-out 1,000; bank 39,000; queries 8,000). The paper protocol (v2) fits on ImageNet train, uses the 100k train bank and all 50k validation queries. Numbers from the two protocols are not comparable (results/paper/tables/T_e1_history.md); for example the plain DINOv2 original scores 77.35 under v1 in E1b and 78.71 under v2.

---

## 4. Why low rank fails where mirror works (claim C4)

**The idea as planned.** SVD compression keeps the top r singular directions of a weight matrix and discards the rest; it works well when a few directions carry most of the energy (a steep spectrum). The project's hypothesis is that DINOv2's training spreads information evenly across directions (a flat weight spectrum), so any rank cut throws away a lot, while Mirror projection keeps full rank in both the even and odd parts and only removes the symmetry-breaking part (PROJECT_CONTEXT.md, Sections 1 and 5.6). The pre-registered rule P2 turned this into a test: at matched parameters, SVD discards energy E_svd and Mirror discards energy s, so Mirror should win in a block exactly when s < E_svd (activation-weighted versions as the main measure) (PREREG_E2.md).

**What the data show.**

1. *Mirror clearly beats same-size SVD when several DINOv2 blocks are converted* (T_main_zeroshot.md, MLP only):

| Model | k | Mirror | SVD | Random | Original |
|---|---|---|---|---|---|
| DINOv2-reg | 3 | 78.35 | 76.41 | 3.80 | 78.94 |
| DINOv2-reg | 6 | 77.09 | 8.77 | 3.95 | 78.94 |
| DINOv2-reg | 9 | 76.36 | 5.83 | 4.72 | 78.94 |
| DINOv2 plain | 3 | 77.36 | 76.63 | 6.35 | 78.71 |
| DINOv2 plain | 6 | 74.65 | 66.14 | 3.48 | 78.71 |
| DINOv2 plain | 9 | 73.26 | 36.63 | 3.04 | 78.71 |

2. *But not because SVD throws away more weight energy per matrix.* At matched parameters, SVD discards 0.127 (fc1) and 0.137 (fc2) of the weight energy in blocks 1 to 6 of the registers model (results/paper/tables/T_x_checks.md), while Mirror's breaking shares are larger, 0.223 and 0.177 (T_f1_ranges.md, ext_f1_dinov2_F1_SUMMARY.md). In the activation-weighted single-block test (F4), Mirror's discarded energy is larger than SVD's in most early DINOv2 blocks, yet Mirror's k-NN drop is smaller (results/paper/tables/T_f4.md). So the pre-registered rule P2 got only 16 of 36 cases right (0.444; bar 0.75): DeiT-III 9 of 12, plain DINOv2 3 of 12, DINOv2-reg 4 of 12. P2 FAILED (T_f4.md). Note that single-block drops are all below 1 point (largest 0.79), so each case compares small differences (T_f4.md).

3. *SVD's damage compounds across blocks on the registers model.* Converting one block at a time with SVD is harmless (largest single-block drop 1.55 points on a small protocol), but stacking them collapses: 74.45 with no blocks, 73.70, 73.10, 71.75 for 1 to 3 blocks, then 35.10, 8.95, 4.80 for 4 to 6 blocks. Plain DINOv2 falls more gently: 74.25 down to 58.75 at 6 blocks (T_x_checks.md, X1b, small protocol). Gate G1 checked that this collapse is not a bug (full-rank SVD gives identical outputs, ranks match the budget, no fp16 overflow; PASS) (T_x_checks.md, T_prereg.md).

4. *It depends on the model.* On DINOv2-L (k = 12 of 24) SVD survives at 78.57 against Mirror's 79.26; on DeiT-III (k = 6) SVD gets 80.67 against Mirror's 81.55 (T_generality.md).

**Bottom line for C4.** "At equal parameters Mirror beats low rank" holds strongly on DINOv2-B (especially the registers model) and weakly elsewhere. "A simple energy rule predicts which wins, per layer" failed as pre-registered. The "flat spectrum" premise has not been measured directly in the repository beyond the energy numbers above (verify), and why SVD errors compound so badly on DINOv2-B but not on DINOv2-L or DeiT-III is an open question (verify). How C4 is stated in the paper is one of Dawar's open decisions (results/STATUS.md). Secondary readings of the rule that were not pre-registered score higher (weight-only 25 of 36; activation-weighted with plain SVD error 30 of 36; fc1 approximation 22 of 36) (T_f4.md), but they cannot replace the pre-registered verdict.

---

## 5. What healing is

### 5.1 Why surgery loses accuracy

A real network is only *approximately* mirror-symmetric. Projection keeps the symmetric part of the weights and throws away the symmetry-breaking part (the breaking share). For a few blocks the loss is small (6 blocks MLP only: 77.09 against 78.94). When everything is converted with one operator the loss is large: V-exact at k = 12 scores 53.85 zero-shot against 78.94 (T_main_zeroshot.md), because one operator A* must serve all 12 layers and the late layers have few odd directions (A* for 12 blocks has only 68 odd directions) (T_attention.md).

### 5.2 What healing does

Healing is a short distillation fine-tune:
- **Teacher:** the original model, frozen. **Student:** the converted model (mirror/heal.py).
- **Loss:** normalised mean squared error between student and teacher on (a) the final CLS token, (b) the final patch tokens and (c) the residual-stream output of every converted block, each divided by the teacher's mean square (measured once on 1,024 training images and then fixed), summed with weight 1 each (mirror/heal.py, configs/m3_pilot.yaml).
- **Data:** 100,000 ImageNet *train* images, disjoint from the fit and k-NN bank images; random resized crops (scale 0.35 to 1.0) to 224 pixels and random horizontal flips (PAPER_EXPERIMENTS.md, M3-pilot; configs/m3_pilot.yaml). Validation images are never used for healing (CLAUDE.md).
- **Budget:** 2 epochs, batch 256, which is 780 steps and 199,680 images seen; AdamW with weight decay 0.05, 5% linear warm-up then cosine decay, bf16 (runs/m3_vexact/k12/SUMMARY.md, configs/m3_pilot.yaml).
- **Learning rate:** chosen separately for each method from {1e-5, 5e-5, 1e-4} by training 100 steps with each and taking the lowest distillation loss on 5,000 held-out *train* images (never on validation) (configs/m3_pilot.yaml). All three pilot arms chose 1e-4; V-exact chose 5e-5 at k = 6 and 1e-4 at k = 12 (T_healing.md). (1e-4 is the largest value in the grid; the repository does not flag this.)
- **Structure kept exact:** after every optimiser step, the converted weights are projected back onto their symmetric form, so the student stays a Mirror model with the same saving and the same exact symmetry (mirror/heal.py: "constraints ... applied ... once before training and after every optimiser step"). For SVD the low-rank factors are trained directly (T_healing.md). At the end of V-exact healing the constraint residuals are about 1e-8 (runs/m3_vexact/k12/SUMMARY.md).

Short form: same budget, data, optimiser and learning-rate grid for every method, and the student is never allowed to leave its compressed form.

This is the standard "recovery fine-tuning" step used after pruning or low-rank compression in general practice; the repository does not cite a specific prior method for it (verify any named method before writing one into the paper).

### 5.3 Why it is needed

- It makes heavy compression usable: full conversion (exact symmetry, 0.765 of the parameters) goes from 53.85 to 78.31 (runs/m3_vexact/k12/SUMMARY.md).
- Claim C5 in the plan says that "after equal-budget healing" Mirror beats SVD, SliceGPT-style slicing and width pruning, and beats random structure (PROJECT_CONTEXT.md, Section 4). Comparing methods only zero-shot could be unfair to methods that recover well with a little training, so a healed comparison is the fair test.

### 5.4 The V-exact recipe differs from the pilot (important)

V-exact healing used two extra choices that the M3 pilot arms did not (results/STATUS.md, "Deviations"; runs/m3_vexact_5090/k12/SUMMARY.md):
1. The embeddings (patch embedding, position grid, CLS, registers; 653,568 extra trainable entries) are trained and re-projected every step; the pilot trains only the converted blocks.
2. The targets are **symmetrised**: the teacher's block outputs are centred and averaged with their mirrored, A*-mapped versions; at full conversion the final target is the average of the teacher's features for the image and its mirror. This removes the part of the target an exactly symmetric student cannot fit; in expectation under random flips it changes the loss only by a constant.

So V-exact results should not be compared one-for-one with the pilot arms, and V-exact's held-out losses are not comparable with the pilot's (T_healing.md).

---

## 6. Is the paper about healing?

**No.** The claims (PROJECT_CONTEXT.md, Section 4), in plain words, with their status as of 9 November 2026:

| Claim | Plain words | Status |
|---|---|---|
| C1, the symmetry tax | Early and middle blocks of flip-trained ViTs are nearly mirror-symmetric with per-layer operators; most MLP neurons there are mirror twins; a formula counts the redundancy | **Supported.** E1b GO; paper protocol GO on 3 seeds (plain), AMBER on 3 seeds (registers, only because the fc1 breaking share is 0.223 against a 0.2 bar); tax 0.419 / 0.405 (results/STATUS.md, T_f1_ranges.md) |
| C2, the operator drifts with depth | One operator cannot serve all layers; neighbouring layers agree on their high-energy part | **Supported.** E1 KILL itself; one operator for all layers collapses (k = 6: 20.48 against 77.08 per-layer); neighbour agreement 0.802 (null 0.488), 0.961 energy-weighted (T_ablations.md, T_f2_drift.md) |
| C3, artefact tokens break the symmetry | In plain DINOv2 the late-layer failure comes from high-norm artefact tokens; registers or masking restore it | **Supported.** P1 both bullets PASS (2.27x, 1.30x), gate G3 PASS (T_prereg.md) |
| C4, symmetry is not low rank | At equal parameters Mirror beats low rank where the spectrum is flat, and a simple rule predicts the winner per layer | **Split.** First half strongly supported on DINOv2-B; the rule (P2) FAILED, 16 of 36 (Section 4) |
| C5, Mirror Surgery works | Released checkpoints become symmetric, cheaper networks (MLP and attention); after equal-budget healing they beat SVD, SliceGPT-style slicing and pruning, and a random-structure control | **Partly.** Zero-shot: yes, by large margins on DINOv2. Exact full conversion: works (78.31, 100% flip). Healed: beats random (P3.a PASS, +5.54) and stays near the original (P3.c PASS), but not SVD by 3 points (P3.b FAIL, so G2 FAIL); strong baselines not yet run; no wall-clock speed-up |
| C6 (stretch), data-free readout | The operator and twins can be read from the weights alone | **Partial at best.** Layer-0 operator from weights PASS (0.90); twin pairing from weights FAIL (46.6%); gate G4 FAIL, so C6 is dropped unless Dawar decides otherwise (T_prereg.md, runs/d/D_DATAFREE_REPORT.md) |
| C7, generality | The effect appears in DeiT-III and MAE, weaker in SigLIP2 | **Mostly supported.** P4 PASS for DeiT-III and MAE; SigLIP2 structure weak as predicted (KILL verdict), but its pre-registered sub-check FAILED as worded (Section 9.9) |

Healing appears only inside C5, as the fair way to compare compressed models after a short recovery. It is a supporting step, not the contribution. The contributions are the measurement (C1 to C3, C7), the explanation versus low rank (C4), and the method (C5: zero-shot conversion, exact full conversion).

---

## 7. Why is healing "not working"?

**Healing works.** The problem is one pre-registered comparison.

### 7.1 Healing recovers almost everything, for every method

M3 pilot (registers model, first 9 blocks converted, MLP plus attention, V-local, matched parameters) (results/M3_PILOT_G2.md, T_healing.md):

| Method | Zero-shot top-1 | Healed top-1 | Recovered (arithmetic) | Healed flip consistency | Held-out loss before / after | Healed linear probe |
|---|---|---|---|---|---|---|
| Original | 78.94 | | | 96.28 | | 80.99 |
| Mirror | 74.70 | **78.81** | 4.11 | 95.98 | 3.639 / 0.863 | 80.71 |
| SVD | 6.49 | 77.57 | 71.08 | 94.12 | 53.108 / 2.130 | 79.30 |
| Random structure | 3.62 | 73.28 | 69.66 | 88.67 | 25.065 / 3.878 | 74.99 |

(The held-out loss is the distillation loss on 5,000 held-out train images; lower means closer to the original.)

V-exact (one operator over the span, exactly equivariant; different recipe, Section 5.4) (T_healing.md, SUMMARY files):

| Setting | Zero-shot | Healed | Original on that GPU | Parameters kept (whole model) |
|---|---|---|---|---|
| k = 6 (RTX 5090) | 77.37 | 78.79 | 78.93 | 0.799 |
| k = 12, full conversion (A100) | 53.85 | **78.31** | 78.94 | 0.765 |
| k = 12, full conversion again (RTX 5090) | 53.83 | **78.34** | 78.93 | 0.765 |

### 7.2 What failed: the comparison with healed SVD (gate G2)

Pre-registered test P3 (PREREG_E2.md), at k = 9:

| Check | Value | Bar | Verdict |
|---|---|---|---|
| P3.a healed Mirror minus healed random | +5.54 | at least 2.0 | PASS |
| P3.b healed Mirror minus healed SVD | +1.24 | at least 3.0 | **FAIL** |
| P3.c original minus healed Mirror | 0.12 | at most 2.0 | PASS |
| Gate G2 (P3.a and P3.b) | | both | **FAIL** |

(results/M3_PILOT_G2.md, T_prereg.md)

**Why it failed.**
1. *Healing rescues SVD too.* Zero-shot SVD at k = 9 is broken (6.49), but 780 steps of distillation bring it to 77.57. The large zero-shot gap (74.70 against 6.49) mostly closes.
2. *There is a ceiling.* The student learns to copy the teacher, so it can hardly beat the original (78.94). To beat healed SVD by 3.0 points, Mirror needed 80.57 (arithmetic: 77.57 + 3.0), which is above the original. The largest margin any method could reach at this setting was 1.36 points (original minus healed SVD) (results/M3_PILOT_G2.md). The bar was effectively unreachable once SVD healed this well.
3. *Mirror is still ahead.* Mirror wins on every measure, just by less than the bar: k-NN by 1.24, linear probe by 1.41, transfer CIFAR-100 by 2.22 and DTD by 2.66 points (T_tier2.md, information only; the verdict stays on k-NN as pre-registered). It also keeps flip consistency closest to the original and ends with the lowest distillation loss (table above).

**What passed.** P3.a shows the fitted structure matters after healing (random structure with the same shape and budget heals to only 73.28). P3.c shows healed Mirror is essentially lossless at 9 blocks (0.12 below the original).

**Consequence.** By the project's rules (CLAUDE.md: stop and report at each gate), the full healing grid (M3-grid: all baselines at k = 6 and 9, three seeds) was put on hold and the framing went back to Dawar (results/STATUS.md). That is why it can look as if "healing is not working": the gate stopped the healing programme, not a failure of healing.

---

## 8. Pre-registration and the verdict words

**Pre-registration** means writing down, before running an experiment, exactly what will be measured and what result counts as success or failure, and committing that file to git so the time stamp proves it came first. Thresholds are never changed after the commit (CLAUDE.md). Files and commits: E1_INSTRUCTIONS.md (decision table for E1, commit 153b21c), E1B_PREREG.md (commit 93bd945), PREREG_E2.md (commit 9804f06) (T_prereg.md).

### 8.1 Decision table verdicts (E1, E1b and the generality runs)

Five rows over blocks 1 to 6 (E1_INSTRUCTIONS.md, B4; E1B_PREREG.md):

| Row | Check | GO if | KILL if |
|---|---|---|---|
| 1 | e(null) / e(A): how much better the fitted operator is than a matched random involution | at least 5 | below 3 |
| 2 | e(A) / e(I): fitted operator against doing nothing | at most 0.5 | informative only ("INFO") |
| 3 | d- / d: share of odd directions | at least 0.2 | below 0.1 (then "PIVOT: little to save") |
| 4 | MLP breaking share (larger of fc1 and fc2 means) | at most 0.2 (with enough odd directions or twins) | above 0.3 |
| 5 | Zero-shot surgery at k = 6: k-NN drop, and margin over same-size SVD | drop at most 10 and margin at least 5 | no better than SVD |

- **GO:** every row in its GO cell.
- **KILL:** any KILL cell in rows 1, 4 or 5.
- **PIVOT:** rows 1 to 4 GO but the surgery drop is above 10 points (structure is there, but it costs too much accuracy); or "PIVOT: little to save" if d-/d is below 0.1.
- **AMBER:** the table gives no verdict (some row is neither GO nor KILL); a human decides.

### 8.2 PASS and FAIL

Single pre-registered checks (P1 to P5) are PASS or FAIL against one threshold.

### 8.3 Gates (stop and report to Dawar)

From PAPER_EXPERIMENTS.md, Section 0.3:
- **G1** (after X1): if SVD's collapse on the registers model is a bug, fix it and rerun all SVD numbers. **PASS** (no bug).
- **G2** (after the healing pilot): healed Mirror must beat healed random by at least 2 points and healed SVD by at least 3. **FAIL** (SVD part).
- **G3** (after F3): if neither registers nor masking restores the late layers, drop C3. **PASS**.
- **G4** (after the data-free pilot): if data-free agreement is below the bar, drop C6. **FAIL**.

### 8.4 Pre-registered hypotheses P1 to P5 (PREREG_E2.md)

- **P1 (artefacts, C3):** plain DINOv2: excluding the top 1% highest-norm tokens cuts the mean error over layers 9 to 11 by at least 2x; registers model: mean error over layers 9 to 11 at most 2x that over layers 1 to 6.
- **P2 (rule, C4):** the sign of (activation-weighted E_svd minus activation-weighted s) predicts the winner of single-block surgery in at least 75% of block and model cases.
- **P3 (healing, G2):** at k = 9, healed Mirror at least 2.0 above healed random, at least 3.0 above healed SVD, within 2.0 of the original.
- **P4 (generality, C7):** DeiT-III and MAE: mean MLP breaking share at most 0.25 over the first half and e(null)/e(A) at least 5; SigLIP2's breaking share higher than DINOv2-reg's.
- **P5 (data-free, C6):** layer-0 operator from weights agrees with the data-driven one (mean cosine of principal angles at least 0.8); weight-only twin pairing in block 1 recovers at least 70% of data-driven pairs.

All results are in Section 9.15.

---

## 9. Full history with every statistic

### 9.1 Timeline (from `git log`; commit hashes in brackets; dates are commit dates)

- **7 November 2026.** E1 code written and tested on the laptop (153b21c); decision rules fixed (aa77384). E1 run on the friend's A100 (report dated 16:38): **KILL** (c74827f, 09581d0). Post-hoc per-layer diagnostic (29b1ad6). E1b pre-registered (93bd945).
- **8 November.** E1b primary arm **GO** (26f578d); E1b complete, registers and shared-operator arms AMBER (f79f934). Paper plan v2 (b890211). X1 SVD check, no bug (a3cb824, 7456d50). Stage 0 code: paper protocol, attention conversion, healing trainer (e8bf18d). PREREG_E2 committed (9804f06). Generality, data-free, findings and ablation code (98faa64 to 1f32b99). M1 attention run (ced2c08). First paper-protocol results (f737522). **M3 healing pilot complete: G2 FAIL** (8f294b9). Paper data package (ab5dae8). Latency and Tier-2 code (8a7b6c4, edc3bbc). Seeds, ablations, linear probes (66d8ef9). F4 **P2 FAIL**, M2 (contended) (623e623). End of A100 session STATUS (eaf25db).
- **Night of 8 to 9 November.** Clean latency re-run, ADE20K code and V-exact healing code (e8c6954, d9e8ca0, 5fb53c0; commit dates just after midnight on 9 November). V-exact full conversion healed on the A100 (its summary is time-stamped 2026-11-08 21:14:07; committed as 2f81626). The A100 went offline at about 21:40 UTC on 8 November (results/STATUS.md).
- **9 November.** Moved to a rented RTX 5090 at about 04:30 UTC (b95ccd5). DINOv2-L and DINOv2-S generality (f966623, cf2db3b). V-exact k = 6 healed (48447ad). Paper figures (ee53d5f). Tier 2 for healed V-exact k = 6 (4bb5c05). Consistency audit (9c58cf6). ADE20K on the 5090 (2039ef8). Flip-averaging baseline code (c121ed0). Strong baselines code (1fbe7c2). **V-exact k = 12 replicated on the 5090** (87b0dbe). F7 visuals script and resume list (fceaf2f). 5090 switched off at about 09:50 UTC, storage kept; STATUS "paused" (5cd29af). Job list for a 5090 in South Korea (6f2c918).

### 9.2 E1: one shared operator (protocol v1, KILL)

**Question** (E1_INSTRUCTIONS.md): does plain DINOv2 ViT-B/14 contain *one* mirror operator for the whole residual stream and mirror-twin MLP neurons, so that projecting early MLPs onto symmetric weights costs little accuracy and clearly less than same-size SVD?

**Design.** One operator fitted from cross-moments summed over layers 1 to 12; MLP-only surgery on the first k blocks; fit 2,000, held-out 1,000, bank 39,000, queries 8,000, all from ImageNet validation; seed 0; A100; 28.1 minutes (results/E1_REPORT.md).

**Decision table** (results/E1_REPORT.md):

| Row | Value | Cell |
|---|---|---|
| 1. e(null) / e(A_shared) | 1.102 (worst layer 1: 1.02) | KILL |
| 2. e(A_shared) / e(I) | 0.9376 | INFO |
| 3. d- / d | 0.01042 | KILL |
| 4. MLP breaking share | 0.3072 (s1 0.307, s2 0.256; matched null 0.307, 0.308) | KILL |
| 5. k = 6 drop | 64.11 (original 78.21, surgery 14.10, SVD 72.45; margin -58.35) | KILL |

**Outcome: KILL** (KILL cells in rows 1, 4, 5).

Shared operator: d+ 760, d- 8 out of 768. Per-layer errors with each layer's *own* best operator were above 1 at layers 9, 10, 11 (1.46, 1.37, 1.18): no operator at all lines those layers up (results/E1_REPORT.md, results/E1_DIAGNOSTIC.md).

Zero-shot surgery in E1 (k-NN top-1 / flip consistency, %) (results/E1_REPORT.md):

| k | Shared-operator surgery | SVD | Random |
|---|---|---|---|
| 0 (original) | 78.21 / 96.58 | | |
| 3 | 62.99 / 80.63 | 76.93 / 94.58 | 68.59 / 85.90 |
| 6 | 14.10 / 34.26 | 72.45 / 89.43 | 36.83 / 54.40 |
| 9 | 11.28 / 27.08 | 55.80 / 71.70 | 22.76 / 38.28 |

**Why KILL, in plain words.** Layers 9 to 11 of plain DINOv2 do not line up under any mirror. Their statistics are large, and the E1 specification summed the raw (unnormalised) statistics of all 12 layers, so those layers dominated the shared fit and pulled it almost to the identity, with only 8 odd directions. With almost no odd directions, the "symmetric" projection was meaningless and surgery destroyed the model. PROJECT_CONTEXT.md calls the pooling of unnormalised layers a "spec flaw". A scale-normalised shared fit, shown only as a diagnostic in the report, had much lower errors in early layers (layer 1: 0.4194 against 1.063) (results/E1_REPORT.md).

Other E1 facts: the layer-0 position-embedding check gives ||mirror(P) - P A_0|| / ||P|| = 0.256 for the layer-0 operator against 0.967 for the identity (results/E1_REPORT.md). Folding check 6.68e-06; identity-surgery check 2.81e-05 (same).

### 9.3 E1 per-layer diagnostic (post hoc, no verdict)

Designed after seeing E1, so it carries no verdict (results/E1_DIAGNOSTIC.md, commit 29b1ad6). Same data, one operator per layer:

| Row | Shared (E1) | Per-layer (diagnostic) | GO limit |
|---|---|---|---|
| 1. e(null)/e(A) | 1.10 | 10.58 (worst layer 6: 4.55) | at least 5 |
| 2. e(A)/e(I) | 0.938 | 0.160 | at most 0.5 |
| 3. d-/d | 0.010 | 0.259 (0.129 to 0.362) | at least 0.2 |
| 4. s1, s2 (null) | 0.307, 0.256 (0.307, 0.308) | 0.194, 0.169 (0.416, 0.394) | at most 0.2 |
| 5. k = 6 drop; margin | 64.11; -58.35 | 4.29; +8.87 | at most 10; at least 5 |

Per-layer zero-shot surgery (top-1, flip consistency in brackets): k = 3: 76.70 (93.15) against SVD 76.00 (93.38) and random 5.00 (12.83), MLP saving 46.6%; k = 6: 73.93 (89.54) against 65.06 (82.59) and 3.81 (14.14), saving 40.5%; original 78.21 (96.58). Block 1: 74% of neurons in pairs, mean twin correlation about 0.77. d-/d falls from 0.49 at layer 0 to 0.13 at layer 6 and about 0.01 from layer 10 (results/E1_DIAGNOSTIC.md).

### 9.4 E1b: per-layer operators, pre-registered (protocol v1, GO)

Pre-registered in E1B_PREREG.md (commit 93bd945) before any run, same thresholds as E1, fresh data split (split seed 1), seed 0, fp32, A100 shared with another project (memory capped at 12% and 6%) (results/E1B_SUMMARY.md).

| Arm | Role | Outcome | Rows not GO |
|---|---|---|---|
| Per-layer, plain ViT-B/14 | primary | **GO** | none |
| Per-layer, ViT-B/14 with registers | secondary | AMBER | 4 |
| One shared operator fitted on layers 1 to 6, plain | exploratory | AMBER | 1, 4 |

Decision table (results/E1B_SUMMARY.md, T_prereg.md):

| Row | Primary | Registers | Shared 1 to 6 |
|---|---|---|---|
| 1. e(null)/e(A) | 10.57 (worst 4.55) | 13.18 (worst 5.74) | 4.39 (worst 3.44) |
| 2. e(A)/e(I) | 0.160 | 0.169 | 0.373 |
| 3. d-/d | 0.260 | 0.276 | 0.225 |
| 4. breaking share (null) | 0.194 (0.416) | 0.223 (0.427) | 0.244 (0.392) |
| 5. k = 6 drop, margin over SVD | 3.62, +9.31 | 1.77, +68.26 | 4.75, +6.64 |

Zero-shot surgery, k-NN top-1 (results/E1B_SUMMARY.md):

| k | Arm | Original | Mirror | SVD | Random |
|---|---|---|---|---|---|
| 3 | primary | 77.35 | 75.74 | 76.06 | 6.58 |
| 6 | primary | 77.35 | 73.73 | 64.41 | 3.78 |
| 9 | primary | 77.35 | 71.69 | 34.80 | 3.35 |
| 3 | registers | 78.11 | 77.35 | 75.81 | 3.20 |
| 6 | registers | 78.11 | 76.34 | 8.08 | 3.31 |
| 9 | registers | 78.11 | 75.79 | 5.84 | 4.63 |
| 3 | shared 1 to 6 | 77.35 | 76.30 | 76.56 | 18.68 |
| 6 | shared 1 to 6 | 77.35 | 72.60 | 65.96 | 4.16 |
| 9 | shared 1 to 6 | 77.35 | 46.59 | 26.79 | 3.81 |

What E1b established (results/E1B_SUMMARY.md): the per-layer result replicates on fresh data (rows 1 to 4 match the diagnostic to about 0.01); Mirror degrades gently while same-size SVD does not (k = 3: SVD ahead by 0.3; k = 6: Mirror ahead by 9.3; k = 9: by 36.9); random structure destroys the model; with registers, layers 9 to 11 line up well (per-layer error 0.13 to 0.14, against 1.17 to 1.45 without registers), pointing to artefact tokens as the cause of E1's failure; a shared operator for layers 1 to 6 is close but weaker (layer 1 error 0.213 against 0.037 per-layer). About 40% of MLP parameters are removed in converted blocks (40.5%, primary). Flip consistency drops from 96.1% to 89.7% at k = 6 (per-layer surgery is not exactly equivariant). Caveats: registers arm narrowly AMBER (0.223 against 0.2); SVD collapse on the registers model flagged for checking (done later as X1, no bug); single seed.

### 9.5 Stage 0 of the paper experiments: X1 to X5 (protocol v2)

**X1 and gate G1: is SVD's collapse on the registers model a bug?** No. Full-rank SVD reproduces the model (max CLS difference 5.12e-05 registers, 3.39e-05 plain); folding exact (4.29e-06, 6.29e-06); SVD ranks within 1% of the budget (fc1 0.9979, fc2 0.9973 registers; 0.9985, 0.9973 plain); features finite; float32, no fp16. **G1 PASS** (T_x_checks.md, T_prereg.md). X1b compounding numbers are in Section 4, point 3.

**X2: E1b arms rerun under the paper protocol.** Plain DINOv2: **GO on all 3 fit seeds**; registers model: **AMBER on all 3 fit seeds** (row 4 only, 0.223) (results/STATUS.md, T_generality.md). Per seed (T_generality.md):

| Model | Fit seed | Row 1 | Row 2 | Row 3 | Row 4 | Row 5 drop | Mirror k = 6 | SVD k = 6 | Margin |
|---|---|---|---|---|---|---|---|---|---|
| DINOv2-reg | 0 | 13.06 | 0.170 | 0.275 | 0.223 | 1.84 | 77.09 | 8.77 | 68.33 |
| DINOv2-reg | 1 | 13.07 | 0.170 | 0.275 | 0.223 | 1.85 | 77.09 | 8.78 | 68.30 |
| DINOv2-reg | 2 | 13.07 | 0.170 | 0.275 | 0.223 | 1.84 | 77.09 | 8.76 | 68.34 |
| DINOv2 | 0 | 10.48 | 0.161 | 0.259 | 0.194 | 4.03 | 74.68 | 66.23 | 8.45 |
| DINOv2 | 1 | 10.49 | 0.161 | 0.259 | 0.194 | 4.14 | 74.57 | 66.05 | 8.52 |
| DINOv2 | 2 | 10.48 | 0.161 | 0.259 | 0.194 | 4.02 | 74.69 | 66.12 | 8.57 |

Main zero-shot rows, MLP only (T_main_zeroshot.md; top-1, with flip consistency and kept fraction):

| Model | k | Variant | Top-1 | Flip | MLP kept | Whole-model GFLOPs saved |
|---|---|---|---|---|---|---|
| DINOv2-reg | 0 | Original | 78.94 | 96.28 | 1.000 | 0% |
| DINOv2-reg | 0 | Folded original (X4) | 78.98 | 96.26 | 1.000 | 0% |
| DINOv2-reg | 3 | Mirror / SVD / random | 78.35 / 76.41 / 3.80 | 94.25 / 93.13 / 7.72 | 0.540 | 5.3% (SVD 7.2%) |
| DINOv2-reg | 6 | Mirror (n = 3) | 77.09 ± 0.00 | 92.31 | 0.581 | 9.2% |
| DINOv2-reg | 6 | SVD (n = 3) | 8.77 ± 0.01 | 42.47 | 0.581 | 13.2% |
| DINOv2-reg | 6 | Random (n = 3) | 3.95 ± 0.11 | 15.22 | 0.581 | 9.2% |
| DINOv2-reg | 9 | Mirror / SVD / random | 76.36 / 5.83 / 4.72 | 90.71 / 32.59 / 23.37 | 0.658 | 10.2% (SVD 16.1%) |
| DINOv2 | 0 | Original | 78.71 | 96.36 | 1.000 | 0% |
| DINOv2 | 3 | Mirror / SVD / random | 77.36 / 76.63 / 6.35 | 93.38 / 93.14 / 14.91 | 0.534 | 5.3% |
| DINOv2 | 6 | Mirror / SVD / random (n = 3) | 74.65 ± 0.07 / 66.14 ± 0.09 / 3.48 ± 0.20 | 90.15 / 82.65 / 13.19 | 0.595 | 8.8% |
| DINOv2 | 9 | Mirror / SVD / random | 73.26 / 36.63 / 3.04 | 87.45 / 49.83 / 10.54 | 0.677 | 9.3% |

(SVD counts fewer FLOPs than Mirror at the same parameter count mainly because per-layer Mirror pays one d x d basis rotation per converted block, which V-exact and SVD do not need (results/paper/tables/cost_model.md).) The exploratory shared-operator arm under v2 is partial (fit seed 0 only): k = 6 surgery 73.58, SVD 67.34, random 3.97 (runs/x2/shared16_plain_s0/results/e1_surgery.csv).

**X3: fit seeds.** Mirror V-local, MLP only, k = 6, registers: 77.09 ± 0.003 over fit seeds 0, 1, 2 (per seed 77.094, 77.088, 77.092) (results/paper/INDEX.md, T_ablations.md). Only k = 6 has three seeds; every other headline number has n = 1 so far (results/STATUS.md, "Deviations").

**A note on two versions of the same number (the "quote X2" rule).** The ablation script and some later runs rebuild the k = 6 MLP-only model from the operators saved in float32 and get 77.11 / 77.04 / 77.10 (mean 77.08), differing from X2 by at most 0.044 points. STATUS.md says: quote X2 (77.09). This briefing follows that rule; ablation tables keep their own values because they are compared with each other (results/STATUS.md, T_ablations.md).

**X4: full LayerNorm folding (SliceGPT-style).** On the registers model the algebra is exact in float64 (max difference 5.72e-06, PASS against the 1e-4 bar) but float32 misses the bar through rounding (9.81e-04); plain DINOv2 misses in both (1.05e-02 float32, 1.97e-04 float64). Folded model k-NN 78.98 against 78.94 (T_x_checks.md, results/STATUS.md).

**X5: pre-registration.** PREREG_E2.md committed as 9804f06 before any gated run (results/STATUS.md).

### 9.6 Stage 1 findings: F1, F2, F3, F4 (F7 pending)

**F1, symmetry tax.** See Section 2.5. Extra columns (T_f1_ranges.md):

| Model | Blocks | Twin fraction | s1 (null) | d-/d |
|---|---|---|---|---|
| DINOv2 | 1 to 6 | 0.609 | 0.194 (0.416) | 0.259 |
| DINOv2 | 7 to 12 | 0.103 | 0.032 (0.103) | 0.042 |
| DINOv2-reg | 1 to 6 | 0.631 | 0.223 (0.427) | 0.275 |
| DINOv2-reg | 7 to 12 | 0.135 | 0.043 (0.118) | 0.043 |
| DeiT-III | 1 to 6 | 0.476 | 0.184 (0.365) | 0.218 |
| DeiT-III | 7 to 12 | 0.007 | 0.006 (0.023) | 0.007 |
| MAE | 1 to 6 | 0.665 | 0.220 (0.452) | 0.344 |
| MAE | 7 to 12 | 0.354 | 0.123 (0.344) | 0.252 |
| SigLIP2 | 1 to 6 | 0.285 | 0.137 (0.252) | 0.138 |
| SigLIP2 | 7 to 12 | 0.001 | 0.009 (0.012) | 0.010 |
| DINOv2-S reg | 1 to 6 | 0.575 | 0.198 (0.407) | 0.270 |
| DINOv2-S reg | 7 to 12 | 0.157 | 0.042 (0.136) | 0.053 |
| DINOv2-L reg | 1 to 12 | 0.579 | 0.252 (0.418) | 0.291 |
| DINOv2-L reg | 13 to 24 | 0.065 | 0.025 (0.069) | 0.032 |

Per-layer detail for DINOv2-reg (fit seed 0; T_structure_per_layer.md): twin fraction by block 0.760, 0.727, 0.660, 0.627, 0.572, 0.438, 0.361, 0.264, 0.101, 0.061, 0.012, 0.011; tax by block 0.481, 0.460, 0.439, 0.418, 0.388, 0.325, 0.273, 0.199, 0.093, 0.054, 0.020, 0.017. MAE is unusual: its structure stays strong into the second half (tax 0.342 in blocks 7 to 12).

**F2, operator drift (C2).** Agreement between neighbouring layers' operators, tr(A_l A_(l+1))/d, mean over pairs (T_f2_drift.md):

| Model | Agreement (null) | Energy-weighted | Principal-angle mean cosine (null) |
|---|---|---|---|
| DINOv2 (3 seeds) | 0.806 (0.508) | 0.937 | 0.973 (0.891) |
| DINOv2-reg (3 seeds) | 0.802 (0.488) | 0.961 | 0.972 (0.885) |
| DeiT-III | 0.859 (0.613) | 0.986 | 0.985 (0.919) |
| MAE | 0.862 (0.164) | 0.976 | 0.973 (0.782) |
| SigLIP2 | 0.839 (0.724) | 0.899 | 0.977 (0.946) |

Reading: neighbours agree far above chance, especially on the high-energy part, but they are not identical; that is why one operator for a long span (V-exact) costs accuracy. The V-exact span operators agree with the per-layer ones at 0.608 (k = 6), 0.597 (k = 9), 0.634 (k = 12) (T_attention.md).

**F3, artefact tokens (P1, gate G3, claim C3).** Plain DINOv2: mean error over layers 9 to 11 is 1.315 with all tokens and 0.579 after excluding the top 1% highest-norm tokens, a cut of **2.27x** (bar 2x): PASS; late layers are 9.91x worse than early layers. Registers model: late over early **1.30x** (bar at most 2x): PASS (late 0.134, early 0.104). **G3 PASS** (runs/f3/f3_dinov2_vitb14.json, runs/f3/f3_dinov2_vitb14_reg.json, runs/f3/f3_gate_G3.json). Two information-only readings from the same files: excluding tokens above 3x the median norm brings layers 9 to 11 of plain DINOv2 to 0.184, 0.169, 0.172 (from 1.44, 1.35, 1.16); and the share of top-1% high-norm positions in the mirrored image that sit at the mirrored positions of the original's high-norm tokens falls from 0.976 at layer 0 to about 0.21 at layers 9 to 11. In words: in plain DINOv2, the artefact tokens appear at positions that do not mirror, which breaks the symmetry; registers remove them (Darcet et al., "Vision Transformers Need Registers", arXiv 2309.16588, listed in PROJECT_CONTEXT.md Section 6).

**F4, symmetry against low rank (P2).** See Section 4. **P2 FAIL**, 16 of 36 (T_f4.md).

**F7, visuals (teaser figure).** Code ready (scripts/f7_visuals.py), not yet run; queued (jobs/stage22_korea.txt).

### 9.7 M1: attention conversion, V-local and V-exact (zero-shot)

Registers model, fit seed 0 (T_main_zeroshot.md, results/paper/INDEX.md):

| k | Variant | Top-1 | Flip | Kept (converted matrices) | Whole-model params | GFLOPs saved (model / converted) |
|---|---|---|---|---|---|---|
| 6 | Mirror V-local, MLP + attention | 76.02 | 91.32 | 0.588 | 69,095,592 | 15.5% / 31.1% |
| 6 | Mirror V-exact, MLP + attention | 77.37 | 95.04 | 0.591 | 69,194,400 | 19.3% / 38.8% |
| 6 | SVD, MLP + attention | 5.27 | 24.33 | 0.587 | 69,061,632 | 19.4% / 39.0% |
| 6 | Random, MLP + attention | 4.19 | 18.79 | 0.588 | 69,095,592 | 15.5% / 31.1% |
| 9 | Mirror V-local, MLP + attention | 74.70 | 89.44 | 0.661 | 64,979,688 | 18.1% / 24.2% |
| 9 | Mirror V-exact, MLP + attention | 73.34 | 93.99 | 0.673 | 65,735,376 | 23.1% / 31.0% |
| 9 | SVD, MLP + attention | 6.49 | 37.45 | 0.660 | 64,932,864 | 24.0% / 32.2% |
| 9 | Random, MLP + attention | 3.62 | 17.75 | 0.661 | 64,979,688 | 18.1% / 24.2% |
| 12 | Mirror V-local, MLP only | 76.28 | 90.32 | 0.824 | 71,636,896 | 8.7% / 8.8% |
| 12 | Mirror V-local, MLP + attention | 74.84 | 89.38 | 0.735 | 64,090,370 | 17.1% / 17.2% |
| 12 | Mirror V-exact, MLP + attention | 53.85 | 97.92 | 0.760 | 66,203,360 | 22.6% / 22.7% |
| 12 | SVD, MLP + attention | 6.27 | 35.59 | 0.734 | 64,024,320 | 25.0% / 25.1% |
| 12 | Random, MLP + attention | 3.64 | 16.67 | 0.735 | 64,090,370 | 17.1% / 17.2% |

Original: 78.94 top-1, 86,583,552 parameters, 47.08 analytic GFLOPs per image (T_main_zeroshot.md, cost_model.md). V-exact span operators: k = 6: d+ 552, d- 216; k = 9: 636, 132; k = 12: 700, 68; exactness inside the span 5.1e-06, 5.6e-06, 7.6e-06 (bar 1e-4, all pass) (T_attention.md). Budget note: V-exact keeps 1.91% (k = 9) and 3.49% (k = 12) more parameters than its SVD baseline, outside the plan's 1% rule, because SVD and random were matched to V-local (results/paper/INDEX.md).

**A6 (MLP only against MLP + attention)** and **A7 (V-local against V-exact)**, zero-shot, come from these rows: at k = 6, MLP only 77.09 (X2), V-local MLP + attention 76.02, V-exact 77.37; at k = 9: 76.36, 74.70, 73.34; at k = 12: 76.28, 74.84, 53.85. V-exact has the highest flip consistency at every k (T_main_zeroshot.md). Converting attention too saves more FLOPs (15.5% against 9.2% at k = 6) for about one point of accuracy; V-exact wins at k = 6 and loses badly at k = 12 before healing.

Per-block attention structure (T_attention.md): self-heads per block 12, 10, 10, 12, 10, 10, 10, 10, 12, 10, 12, 12 (the rest form one twin pair); attention breaking share falls from 0.283 (block 1; random 0.494) to 0.005 (blocks 11 and 12).

### 9.8 Ablations A1 to A5 (zero-shot, registers model, MLP only)

Mean over fit seeds where n = 3 (T_ablations.md). "Params" is MLP weights kept relative to the default; rows not within 1% are not parameter-matched.

**A1, which operator** (k = 6, n = 3 | k = 9, n = 1):

| Operator | k = 6 | k = 9 | Params |
|---|---|---|---|
| Per-layer (default) | 77.08 ± 0.03 (quote X2: 77.09) | 76.35 (quote X2: 76.36) | 1.00 |
| Shared over the span (energy-normalised) | 76.37 ± 0.06 | 67.67 | 1.00 / 1.01 |
| Shared over all layers (the E1 design) | 20.48 ± 0.13 | 15.32 | 1.17 / 1.13, not matched |
| Identity A = I (all features even, odd weights zeroed) | 24.63 ± 0.19 | 16.71 | 1.18 / 1.14, not matched |
| Random operator | 3.88 ± 0.44 | 3.03 | 1.00 |
| SVD | 8.77 ± 0.01 | 5.82 | 1.00 |
| Random structure | 3.70 ± 0.67 | 4.40 | 1.00 |

**A2, which pairing:**

| Pairing | k = 6 | k = 9 | Params |
|---|---|---|---|
| Mirror matching (default) | 77.08 ± 0.03 | 76.35 | 1.00 |
| Random pairing, same counts | 28.96 ± 3.50 | 12.59 | 1.00 |
| All self-twins (no pairs) | 13.73 ± 0.04 | 12.05 | 1.20 / 1.17, not matched |
| Generic similarity (plain weight cosine, not mirror) | 24.69 ± 0.05 | 13.81 | 1.00 |
| Plain Hungarian, no involution constraint (approximate projection) | 77.27 ± 0.05 | 76.53 | 0.98 / 0.99, not matched |
| Plain Hungarian with exact cycle projection | 77.20 ± 0.02 | 76.39 | 0.98 / 0.99, not matched |

**A3, fit-set size** (k = 6, seed 0): 100 images 77.03; 500: 77.18; 2,000: 77.09; 6,000 (all three fit sets pooled; the plan asked for 10,000): 77.11.

**A4, which blocks** (k = 6, seed 0): first 6: 77.11; last 6 (blocks 7 to 12): 78.67; lowest breaking share first: picks the same blocks 7 to 12, so 78.67. Not parameter-matched: the last 6 blocks keep 1.53x the MLP weights of the first 6 (0.89 against 0.58 of their MLP weights), so "last beats first" partly reflects more weights kept.

**A5, projection strength** alpha (W becomes W - alpha (W - W_eq)), k = 6, seed 0: alpha 0: 78.94; 0.25: 78.89; 0.5: 78.43; 0.75: 77.99; 1: 77.11. Only alpha = 1 has the Mirror budget; the others keep dense weights.

**Readings:** the wrong operator or the wrong pairing destroys the model even with more weights kept; a pairing that ignores the mirror (generic cosine) also fails, so it is the mirror redundancy specifically, not any redundancy; Hungarian matching without the involution constraint matches Mirror; 100 fit images already suffice (results/STATUS.md).

**More seeds, from STATUS.md** (raw rows were on the VM, not yet pulled to the laptop): "A4 and A5 for fit seeds 1 and 2 agree with seed 0 (last 6 blocks 78.63 / 78.64 vs first 6 77.04 / 77.10; A5 falls smoothly from 78.94 at alpha 0 to 77.04 / 77.10 at alpha 1)" (results/STATUS.md). The job lists still include A4/A5 for fit seed 2 (jobs/stage22_korea.txt), so which seeds are complete needs checking (verify). STATUS also quotes A1 and A2 at k = 9 from the A100 log (for example per-layer 76.37, shared over the span 67.68, generic 13.86), within about 0.05 of the 5090 rerun values in the table above.

### 9.9 G: generality across models (zero-shot, P4)

Per-layer operators, decision table over each model's first half, surgery (MLP only) at k = half the depth (T_generality.md):

| Model | Outcome | Row 1 | Row 2 | Row 3 | Row 4 | Row 5 drop | Original | Mirror | SVD | Margin |
|---|---|---|---|---|---|---|---|---|---|---|
| DINOv2-reg (seed 0) | AMBER | 13.06 | 0.170 | 0.275 | 0.223 | 1.84 | 78.94 | 77.09 | 8.77 | 68.33 |
| DINOv2 (seed 0) | GO | 10.48 | 0.161 | 0.259 | 0.194 | 4.03 | 78.71 | 74.68 | 66.23 | 8.45 |
| DeiT-III B/16 (supervised) | AMBER | 10.66 | 0.215 | 0.218 | 0.184 | 1.41 | 82.95 | 81.55 | 80.67 | 0.88 |
| MAE B/16 | AMBER | 40.76 | 0.048 | 0.344 | 0.232 | 5.57 | 13.72 | 8.14 | 3.52 | 4.62 |
| SigLIP2 B/16 (trained with text) | KILL | 2.38 | 0.613 | 0.138 | 0.137 | 61.70 | 75.53 | 13.83 | 9.13 | 4.70 |
| DINOv2-S/14 reg | PIVOT | 20.32 | 0.116 | 0.270 | 0.198 | 18.42 | 74.46 | 56.03 | 10.24 | 45.79 |
| DINOv2-L/14 reg (k = 12 of 24) | AMBER | 79.41 | 0.079 | 0.291 | 0.252 | 1.35 | 80.60 | 79.26 | 78.57 | 0.69 |

Notes:
- DeiT-III is AMBER only because its margin over SVD (0.88) is below 5; its structure passes (T_generality.md).
- MAE's CLS token gives weak k-NN (original 13.72), so the plan measures MAE with the linear probe instead (original 56.54) (PAPER_EXPERIMENTS.md, T_tier2.md). The repository's MAE run is partial (no meta file); a complete MAE run on the 5090 gave AMBER with original 13.73, surgery 8.13, SVD 3.55, random 1.54, but those files were not yet pulled (results/STATUS.md).
- DINOv2-S: strong structure (row 1 20.32) but surgery drops 18.42 points, hence PIVOT; the 5090 run reproduces the A100 run (56.03 against 56.00) (results/STATUS.md).
- DINOv2-L: the strongest structure of all (row 1 79.41), surgery loses only 1.35, but SVD also survives (78.57), so the margin is only 0.69; the dramatic SVD collapse seen on DINOv2-B does not happen on DINOv2-L (results/STATUS.md).

**P4 checks** (T_prereg.md): DeiT-III breaking share 0.184 (at most 0.25) PASS, ratio 10.66 (at least 5) PASS; MAE 0.232 PASS, 40.76 PASS; **SigLIP2's breaking share higher than DINOv2-reg's: FAIL** (0.137 against 0.223). Why the SigLIP2 sub-check failed: SigLIP2 has so little structure (row 1 2.38, d-/d 0.138, twin fraction 0.285) that its matched random null is also low (0.252), and a near-trivial structure has little to break; its weakness shows in rows 1 to 3 instead (results/STATUS.md: "structure near trivial; rows 1 to 3 much weaker"). The prediction that SigLIP2 is weaker came true in substance (rows 1 to 3), but the pre-registered measure (the breaking share) did not capture it.

### 9.10 D: data-free readout (P5, gate G4, claim C6)

(runs/d/D_DATAFREE_REPORT.md, T_prereg.md)
- **D1, layer-0 operator from the weights only** (flipped patch-embedding kernels and mirrored position embeddings): mean cosine of principal angles with the data-driven even subspace 0.900 (bar 0.8): **PASS**; chance level 0.652; d+ 392 from weights against 400 from data.
- **D2, twin pairing from weights only** in block 1, given the data-driven operator: recovers 0.466 of the data-driven pairs (bar 0.7): **FAIL**; with the fully data-free operator: 0.441: FAIL. Recovery is higher in later blocks (between 0.592 in block 2 and 0.895 in block 11), but P5 was pre-registered on block 1.
- **P5 FAIL, G4 FAIL**: C6 is dropped unless Dawar decides otherwise. A "partial C6" framing (operator yes, pairing no) is an open writing decision (Section 12).
- D3 (propagating operators to later layers from weights) is exploratory with no bar (runs/d/D_DATAFREE_REPORT.md).

### 9.11 M2: efficient implementation and latency

A real even/odd implementation was built (twin butterflies, block-structured attention, basis rotations for V-local) and checked against the dense projected model: float64 16 of 16 and fp32 16 of 16 pass; bf16 0 of 16 pass the 1e-3 bar, but the differences are at the level of the dense model's own bf16 rounding (0.41 to 4.93 times the noise floor) (runs/m2_clean/M2_SUMMARY.md). Clean timings on the A100 (runs/m2_clean/, T_latency.md), speed-up against the dense model in the same implementation, bf16, torch.compile, batch 256 (analytic FLOP speed-up in brackets):

| Variant | k = 6 | k = 9 | k = 12 |
|---|---|---|---|
| Mirror V-local, MLP + attention | 0.853x (1.162x) | 0.777x (1.193x) | 0.701x (1.178x) |
| Mirror V-exact, MLP + attention | 0.845x (1.239x) | 0.777x (1.301x) | 0.699x (1.292x) |
| Mirror V-local, MLP only | 0.910x (1.101x) | 0.858x (1.114x) | 0.801x (1.096x) |
| SVD at matched parameters | 0.975x (1.241x) | 0.937x (1.316x) | 0.915x (1.334x) |
| MLP width pruning at matched parameters | 1.133x (1.241x) | 1.168x (1.315x) | 1.169x (1.332x) |

At batch 1 (compiled) Mirror is slower still: for example V-exact k = 6 0.706x, V-local k = 12 0.579x (T_latency.md). **Conclusion: no wall-clock gain with plain PyTorch operations**; the extra gathers, butterflies and smaller matrix multiplications cost more than they save, and a fused kernel would be needed for a latency claim (results/STATUS.md). One probable timing outlier (Mirror MLP only, k = 6, batch 64: 1.247x, above its analytic 1.101x) is kept but not used (T_latency.md). The first M2 run was contended and is superseded. A profiling script (scripts/m2_profile.py) exists but has not run (T_latency_profile.md: no rows).

### 9.12 M3 pilot and gate G2

See Section 7 for the table and verdicts. Extra facts (T_healing.md): all three arms ran 780 of 780 steps; free matrix parameters 42,097,128 (Mirror and random) against 42,050,304 (SVD), so matched; Mirror and random train their dense tensors (63,763,200 entries) and are re-projected each step, SVD trains its factors (42,112,512 entries); one seed (seed 0), commit 94b2e19.

### 9.13 V-exact healing at k = 6 and k = 12, and the cross-GPU replication

| Run | GPU | Zero-shot | Healed | Original (same GPU) | Flip, bf16 k-NN (zero-shot / healed) | fp32 flip check | Exactness after healing | LR | Held-out loss before / after | Kept (converted / whole) | Training time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| k = 6 | RTX 5090 | 77.37 | 78.79 | 78.93 | 94.99 / 96.38 | | 4.6e-06 | 5e-05 | 1.1558 / 0.2918 | 0.591 / 0.799 | 2,949 s |
| k = 12 | A100 | 53.85 | 78.31 | 78.94 | 97.92 / 99.76 | 100.00% (pass) | 1.4e-05 | 1e-04 | 7.3170 / 1.3677 | 0.760 / 0.765 | 4,799 s |
| k = 12 | RTX 5090 | 53.83 | 78.34 | 78.93 | 98.04 / 99.74 | 100.00% (pass) | 8.7e-06 | 1e-04 | 7.3170 / 1.3469 | 0.760 / 0.765 | 2,471 s |

(runs/m3_vexact/k6/SUMMARY.md, runs/m3_vexact/k12/SUMMARY.md, runs/m3_vexact_5090/k12/SUMMARY.md)

Details: in k = 12, the converted blocks keep 64,554,464 of 84,934,656 matrix parameters and the whole model has 66,203,360 of 86,583,552; peak GPU memory 5.44 GB; largest relative CLS difference between an image and its mirror after healing 6.1e-06 (A100) and 4.8e-06 (5090). The bf16 flip consistency can fall slightly below 100% because bf16 rounds the mirrored computation differently; the exact check is the fp32 one (SUMMARY files). The A100's healed k = 12 checkpoint was lost with that machine (results/paper/INDEX.md), so k = 12 was healed again on the 5090, which doubles as the replication: 78.34 against 78.31. Caveats: one seed per run; no matched healed baseline at k = 12 yet; V-exact recipe differs from the pilot (Section 5.4).

### 9.14 Tier 2: linear probes, transfer and ADE20K

**ImageNet linear probe** (top-1, %) (T_tier2.md):

| Model | Variant | k | Linear top-1 | Drop vs original |
|---|---|---|---|---|
| DINOv2 | Original | 0 | 80.68 | |
| DINOv2 | Mirror V-local, MLP only | 6 | 76.80 | 3.88 |
| DINOv2 | SVD, MLP only | 6 | 69.48 | 11.20 |
| DINOv2 | Random, MLP only | 6 | 9.77 | 70.91 |
| DINOv2-reg | Original | 0 | 80.99 | |
| DINOv2-reg | Mirror V-local, MLP only | 6 | 79.08 | 1.91 |
| DINOv2-reg | SVD, MLP only | 6 | 20.92 | 60.07 |
| DINOv2-reg | Random, MLP only | 6 | 11.25 | 69.74 |
| DINOv2-reg | Mirror V-exact (zero-shot) | 6 | 79.51 | 1.48 |
| DINOv2-reg | Mirror V-exact, healed | 6 | 80.86 | 0.13 |
| DINOv2-reg | Mirror V-local, MLP + attention (zero-shot) | 9 | 76.77 | 4.21 |
| DINOv2-reg | Mirror V-local, healed | 9 | 80.71 | 0.28 |
| DINOv2-reg | SVD, healed | 9 | 79.30 | 1.69 |
| DINOv2-reg | Random, healed | 9 | 74.99 | 5.99 |
| MAE | Original | 0 | 56.54 | |

**Transfer linear probes**, DINOv2-reg, top-1 (%) (T_tier2.md, results/STATUS.md):

| Variant | CIFAR-100 | DTD | Flowers-102 | Pets |
|---|---|---|---|---|
| Original | 91.40 | 83.56 | 99.69 | 95.99 |
| V-exact k = 6, zero-shot | 85.21 | 81.65 | 99.69 | 95.72 |
| V-exact k = 6, healed | 90.54 | 83.19 | 99.69 | 95.67 |
| V-local k = 9, zero-shot | 80.89 | 80.37 | 99.59 | 94.74 |
| V-local k = 9, healed (Mirror) | 90.89 | 83.35 | 99.69 | 95.58 |
| SVD k = 9, healed | 88.67 | 80.69 | 99.69 | 95.50 |
| Random k = 9, healed | 84.47 | 77.39 | 99.56 | 94.30 |

Healed Mirror is within about 0.5 points of the original on all four (largest drop 0.51, on CIFAR-100; STATUS.md rounds this to "within 0.5") and ahead of or level with healed SVD and ahead of healed random. Flowers-102 is at ceiling (several variants tie with the original exactly), so it does not separate methods. Several probes chose a weight decay at the edge of the grid (results/paper/INDEX.md).

**ADE20K linear segmentation** (mIoU, %; frozen backbone, linear head, 448 px crops, 8,000 iterations, batch 16; RTX 5090) (runs/t2_seg/t2_ade20k.csv, runs/t2_seg/T2_ADE20K.md, results/STATUS.md):

| Variant | mIoU | Change vs original |
|---|---|---|
| Original | 47.59 | |
| V-exact k = 6, healed | 47.51 | -0.09 |
| V-exact k = 6, zero-shot | 47.19 | -0.40 |
| Mirror V-local, MLP only, k = 6 | 46.88 | -0.72 |
| SVD at matched parameters, MLP only, k = 6 | 9.84 | -37.75 |

ADE20K for the three healed pilot models and for healed V-exact k = 12, and a rerun of the original (to measure run-to-run spread), are still to run or to pull (Section 12).

### 9.15 Every pre-registered check in one table

(T_prereg.md, results/STATUS.md)

| Check | Value | Bar | Verdict |
|---|---|---|---|
| E1 (shared operator) | rows 1, 4, 5 in KILL cells | decision table | KILL |
| E1b primary (per-layer, plain) | all rows GO | decision table | GO |
| E1b registers | row 4: 0.223 | decision table | AMBER |
| E1b shared 1 to 6 | rows 1 and 4 not GO | decision table | AMBER |
| G1 SVD collapse is not a bug | four checks clean | | PASS |
| P1 plain: masking cuts late-layer error | 2.27x | at least 2x | PASS |
| P1 registers: late against early error | 1.30x | at most 2x | PASS |
| G3 | | one P1 bullet passes | PASS |
| P2 energy rule | 16 of 36 (0.444) | at least 0.75 | FAIL |
| P3.a healed Mirror minus healed random | +5.54 | at least 2.0 | PASS |
| P3.b healed Mirror minus healed SVD | +1.24 | at least 3.0 | FAIL |
| P3.c original minus healed Mirror | 0.12 | at most 2.0 | PASS |
| G2 (P3.a and P3.b) | | both | FAIL |
| P4 DeiT-III (share, ratio) | 0.184, 10.66 | at most 0.25, at least 5 | PASS |
| P4 MAE (share, ratio) | 0.232, 40.76 | at most 0.25, at least 5 | PASS |
| P4 SigLIP2 share above DINOv2-reg's | 0.137 against 0.223 | higher | FAIL |
| P5 part 1, layer-0 operator from weights | 0.90 | at least 0.8 | PASS |
| P5 part 2, weight-only twin pairing | 46.6% | at least 70% | FAIL |
| G4 | | P5 passes | FAIL |

---

## 10. Compute and logistics

- **Friend's A100** (NVIDIA A100-SXM4-80GB), shared with another project, so every job capped its GPU memory (6% to 20% per job) and some jobs were paused or reordered (results/E1B_SUMMARY.md, results/STATUS.md). E1, E1b, all of Stage 0 and most of Stage 1 and 2 ran there, including the M3 pilot and V-exact k = 12 healing. It went offline at about 21:40 UTC on 8 November (results/STATUS.md). Its healed k = 12 checkpoint was lost with it (results/paper/INDEX.md); about 2 GB of large artefacts (operators, statistics, healed checkpoints) are on the laptop under `artifacts/a100/` (results/STATUS.md).
- **Rented RTX 5090 (vast.ai), Estonia.** Work moved there at about 04:30 UTC on 9 November with the same code, torch build, seeds and ImageNet subsets; memory caps were scaled by 2.5 so each job kept the same limit in GB as on the 80 GB A100 (caps never change results). The laptop's upload speed was about 60 KB/s, so jobs waited for their inputs (results/STATUS.md, jobs/). It ran DINOv2-S and L, V-exact k = 6 healing, Tier 2 for healed V-exact k = 6, ADE20K and the k = 12 replication. It was switched off at about 09:50 UTC on 9 November while Dawar travelled, with storage kept (results/STATUS.md).
- **GPU taken, copy to South Korea.** After the stop, the Estonian machine's GPU was taken by another renter; its `/workspace` was copied with vast.ai's instance copy to a new RTX 5090 in South Korea (instance 59906003), rented on 9 November (jobs/stage22_korea.txt).
- **Replication across GPUs.** Results run on both GPUs agree to within 0.03 points (results/STATUS.md): V-exact k = 12 healed 78.34 against 78.31, zero-shot 53.83 against 53.85; original 78.93 against 78.94; DINOv2-S surgery 56.03 against 56.00.
- **One operational incident.** Shipping code to the A100 at about 19:01 UTC overwrote a just-written ablation CSV, losing 7 rows; the values survived in the log, the ship step was fixed to skip `runs/`, `results/` and `logs/`, an audit found no other loss, and the rerun reproduced the logged values (results/STATUS.md).

---

## 11. Strong signals and weak or negative signals

### 11.1 Strong signals

| Signal | Numbers | Why it matters |
|---|---|---|
| The structure is real and replicates | E1b GO on fresh data; paper protocol GO x3 (plain), AMBER x3 (registers, row 4 only); 77.09 ± 0.003 over 3 fit seeds (results/STATUS.md, INDEX.md) | C1 rests on pre-registered, multi-seed evidence, not one lucky run |
| Large zero-shot margins over same-size SVD and random on DINOv2 | k = 6 registers 77.09 / 8.77 / 3.95; plain 74.65 / 66.14 / 3.48 (T_main_zeroshot.md) | Training-free compression that keeps accuracy where low rank collapses |
| The fitted structure, not just the shape, matters | random operator 3.88, random pairing 28.96, generic cosine pairing 24.69 against 77.08 (T_ablations.md); healed random 73.28 against healed Mirror 78.81 (P3.a PASS) | Answers the "any constraint would do" objection, before and after healing |
| Cheap to fit | 100 fit images give 77.03 (T_ablations.md) | Practical; data needs are tiny |
| Exact full conversion works after healing, and replicates | 78.31 / 78.34 against 78.94; 100% fp32 flip; 0.765 of parameters (SUMMARY files) | A released foundation model turned into an exactly mirror-invariant one in one forward pass, near-lossless |
| Healed Mirror near-lossless downstream | linear 80.71 vs 80.99; transfer drops at most 0.51; ADE20K V-exact k = 6 healed 47.51 vs 47.59 (T_tier2.md, t2_ade20k.csv) | Not only k-NN; features stay useful |
| Artefact-token explanation confirmed | P1 2.27x and 1.30x, G3 PASS (T_prereg.md) | C3 is a clean, pre-registered finding that links to the registers literature |
| Generality | P4 PASS on DeiT-III and MAE; DINOv2-L structure strongest (row 1 79.41) (T_generality.md) | The tax is not a DINOv2 quirk |
| Layer-0 operator readable from weights | 0.90 against chance 0.65 (D report) | Partial support for a data-free story |

### 11.2 Weak or negative signals

In the last column, items marked "(repo)" are planned in the repository (STATUS.md, job lists, PAPER_EXPERIMENTS.md or the E3 proposal); items marked "(suggestion)" are mine, not decided by anyone.

| Signal | Numbers | What it means | Mitigation |
|---|---|---|---|
| **G2 FAIL** | healed Mirror over healed SVD +1.24, bar 3.0; ceiling: max reachable 1.36 (results/M3_PILOT_G2.md) | The headline "beats SVD after healing" cannot be claimed as pre-registered; healing closes most of the zero-shot gap | Report as is; or the proposed E3 at full conversion with matched healed baselines; or recentre on the tax, exact conversion and zero-shot results; Dawar's decision (repo) |
| **No wall-clock speed-up** | analytic 1.10x to 1.30x, measured 0.70x to 0.91x at batch 256; pruning is faster (1.13x to 1.17x) (results/STATUS.md) | A "cheaper" claim has to be about parameters and FLOPs, not speed; STATUS says a fused kernel would be needed for a latency claim | M2 latency and profile on a quiet GPU, queued last (repo); state savings as parameters and FLOPs, kernel as future work (suggestion) |
| **P2 FAIL** | 16 of 36 (T_f4.md) | C4's predictive rule is wrong; energy accounting underrates Mirror in early DINOv2 blocks | How C4 is stated is an open decision (repo); report P2 as failed and state C4 as an empirical comparison plus the X1b compounding observation (suggestion) |
| **Margins depend on the model** | DINOv2-L margin over SVD 0.69; DeiT-III 0.88; DINOv2-S surgery 56.03 vs 74.46 (PIVOT) (T_generality.md) | The dramatic SVD collapse is mainly a DINOv2-B effect; the small model loses more | Report as measured (suggestion); nothing further is planned in the repo |
| **SigLIP2 sub-check FAIL** | 0.137 against 0.223 (T_prereg.md) | The pre-registered measure (breaking share) did not capture SigLIP2's weaker structure | Report as failed, with rows 1 to 3 as the evidence of weak structure (suggestion) |
| **Weight-only twin pairing FAIL** | 46.6% (bar 70%) (T_prereg.md) | C6 is dropped under the G4 rule unless Dawar decides otherwise | "C6 as partial" is an open writing decision (repo) |
| **Only weak baselines so far** | SVD and random only; strong baselines code ready, not run (results/STATUS.md) | Reviewers will expect activation-aware SVD, SliceGPT-style slicing and pruning | Zero-shot strong baselines first in the Korea job list, about 1 GPU-hour (repo) |
| **One healing seed** | every healed number n = 1 (T_healing.md) | Margins of about 1 point need error bars | E3 proposes a second V-exact healing seed; 3-seed headline numbers planned (PAPER_EXPERIMENTS.md Stage 4) (repo) |
| **V-exact recipe differs from the pilot** | trained embeddings, symmetrised targets (results/STATUS.md) | V-exact numbers are not directly comparable with the pilot arms | E3 proposes healed SVD with the better of two target recipes (repo) |
| **MAE low absolute k-NN** | original 13.72 (T_generality.md) | MAE's k-NN drop is not comparable with the other models' | The plan measures MAE with the linear probe (original 56.54) (repo); a linear probe of converted MAE is not in the tables yet |
| **Registers model narrowly AMBER** | row 4 0.223 against 0.2 on all 3 seeds | A pre-registered row is not GO on the main model | Report it next to the plain model's GO (suggestion) |
| **Fewer seeds than planned** | only k = 6 has 3 fit seeds; ablations mostly seed 0 (results/STATUS.md) | Headline numbers with n = 1 are not final | Stage 4 reruns (repo); ablation seeds 1 and 2 queued (repo) |
| **Budget mismatches in a few rows** | V-exact +1.91% (k = 9) and +3.49% (k = 12) above SVD; A4 not matched (results/paper/INDEX.md) | Comparisons in those rows slightly favour Mirror | Flagged in the tables (repo); rematch any row used for a claim (suggestion) |
| **X4 fp32 bar missed** | 9.81e-04 against 1e-4 (registers) (T_x_checks.md) | Rounding, not algebra (float64 passes on the registers model) | Report the float64 check alongside (suggestion) |
| **Healing LR at the grid edge** | all pilot arms chose 1e-4, the largest value (T_healing.md) | Healing might be under-tuned for some arm (my observation, not flagged in the repo) | Consider a wider grid in E3 (suggestion, verify) |

---

## 12. What is left

### 12.1 Queued now on the South Korea RTX 5090 (jobs/stage22_korea.txt, in this order)

1. **Zero-shot strong baselines**: activation-aware SVD, SliceGPT-style slicing, activation-weighted MLP pruning, each at the Mirror V-local MLP-only budget, k = 6 and 9 (scripts/strong_baselines.py). Zero-shot only, so not part of the healing grid that G2 put on hold.
2. **Flip-averaging baseline**: exact invariance by averaging the features of an image and its mirror at test time (two forward passes), measured against the healed V-exact model, which is exactly invariant in one pass with fewer parameters (scripts/flip_avg_baseline.py). This answers the reviewer objection that exact invariance is trivial.
3. **ADE20K, linear probe and transfer probes for healed V-exact k = 12**.
4. **F7 teaser figure** (twin-pair visuals) (scripts/f7_visuals.py).
5. **Ablation seeds 1 and 2 at k = 9** (A1, A2), and A4/A5 for fit seed 2.
6. **ADE20K for the three healed pilot models** (Mirror, SVD, random at k = 9).
7. **ADE20K rerun of the original** to measure run-to-run spread.
8. **Last, alone on the GPU: M2 latency and profile** on the 5090.

**Results pending from the old Estonian disk** (results/STATUS.md, "Not done yet"): the MAE complete run (AMBER: original 13.73, surgery 8.13, SVD 3.55, random 1.54), A4/A5 fit seed 1, A1 at k = 9 fit seed 1, and whatever finished after 09:26 UTC (possibly ADE20K of healed V-exact k = 12, its probes, the flip-averaging baseline). The Korea instance is a copy of that disk (jobs/stage22_korea.txt), so these should be there but have not yet been pulled to the laptop or checked (verify).

### 12.2 Proposed, awaiting Dawar: a new pre-registered test "E3"

A new pre-registered healing test at full conversion (k = 12) with matched healed baselines: healed SVD (using the better of two target recipes, the pilot's raw targets or V-exact's symmetrised targets, so the comparison is fair to SVD), healed random structure, and a second V-exact healing seed, reusing the E2 bars (at least 2.0 over random, at least 3.0 over SVD, within 2.0 of the original). The idea is to test where compression is heaviest and the comparison is most informative, rather than at k = 9 where the ceiling made the SVD bar unreachable (results/STATUS.md, decision 1: "a new pre-registered healing test where the ceiling does not bind"). Whether the ceiling binds at k = 12 is not known until healed SVD at k = 12 is measured (verify). Files for E3 are being prepared separately and are not described here.

### 12.3 Writing decisions waiting for Dawar (results/STATUS.md)

1. **Framing after G2**: report as is; run a new pre-registered healing test (E3); or recentre the paper on the symmetry tax, exact conversion and zero-shot results.
2. **How C4 is stated** after P2 failed (the energy rule underrates Mirror in early DINOv2 blocks).
3. **C6 as partial** (operator from weights passes, pairing fails) or dropped (G4 rule).
4. Next GPU and its job list (now the Korea instance).

### 12.4 From the original plan, not yet done (PAPER_EXPERIMENTS.md, results/STATUS.md)

M3 grid (on hold after G2), M4 (symmetry plus low-rank composition), 3-seed reruns of all headline numbers, the final FINAL_RESULTS.md, and a novelty search on OpenReview in the final week. Plan timeline: experiments by 26 November, writing 22 November to 3 December, submit 5 December (PROJECT_CONTEXT.md).

---

## 13. Glossary

- **A_l, A\***: the mirror operator of layer l (per-layer), and the single operator shared over a converted span (V-exact). A d x d symmetric orthogonal matrix with A^2 = I.
- **Activation-weighted**: a weight measure multiplied by the input activation covariance to the power 1/2, so directions the data actually uses count more (PROJECT_CONTEXT.md, 5.6).
- **ADE20K**: a scene-segmentation dataset with 150 classes (train 20,210, validation 2,000 images in the copy used) (runs/t2_seg/T2_ADE20K.md).
- **AdamW, cosine schedule, warm-up**: a standard optimiser; the learning rate rises linearly for the first 5% of steps and then decays along a cosine curve.
- **AMBER, GO, KILL, PIVOT, PASS, FAIL, INFO**: verdict words, Section 8.
- **Analytic versus measured speed-up**: FLOP ratio from counting multiply-adds, against real timing on a GPU.
- **Artefact tokens (high-norm tokens)**: a few patch tokens in late layers of plain DINOv2 with very large norm, used by the model as scratch space; registers remove them (Darcet et al., arXiv 2309.16588).
- **bf16, fp32, float64, TF32**: number formats, from least to most precise (bf16 is 16-bit, fp32 32-bit, float64 64-bit). TF32 is a reduced-precision GPU mode for fp32 maths, switched off for exactness checks.
- **Block**: one transformer layer: attention plus MLP, each added to the residual stream.
- **Breaking share s**: share of a weight matrix's energy that projection removes, ||W - W_eq||^2 / ||W||^2.
- **Butterfly**: computing u and v once and outputting u + v and u - v.
- **C1 to C7**: the paper's claims, Section 6.
- **CIFAR-100, DTD, Flowers-102, Oxford-IIIT Pets**: small image-classification datasets used for transfer probes (100 classes of small images; textures; flowers; cat and dog breeds).
- **CLS token**: a special token whose final value is used as the whole image's feature.
- **d, h, d+, d-**: model width (768 for ViT-B), MLP hidden width (3,072), numbers of even and odd directions.
- **DINOv2, DINOv2-reg**: self-supervised vision transformers; "-reg" has 4 register tokens. Main model `dinov2_vitb14_reg`, contrast `dinov2_vitb14`. S, B, L are small, base, large sizes; /14 is the patch size.
- **DeiT-III, MAE, SigLIP2**: other vision transformers used for generality: DeiT-III supervised on labels, MAE self-supervised by masked reconstruction, SigLIP2 trained with text (PROJECT_CONTEXT.md, C7).
- **Distillation (teacher, student)**: training a model (student) to copy another model's (teacher's) outputs.
- **e(A), e(I), e(null)**: relative held-out error of the mirror fit with the fitted operator, the identity, and a matched random involution.
- **E1, E1b, E2, E3**: the first go/no-go test (KILL), its per-layer pre-registered rerun (GO), the paper experiments pre-registered in PREREG_E2.md, and the proposed new full-conversion test.
- **Energy-weighted agreement**: agreement of two operators measured on the directions where the activations have most energy.
- **Equivariant / invariant**: mirror the input and the features change in a known way (by A) / do not change at all.
- **Even / odd directions**: directions unchanged / sign-flipped by the mirror operator.
- **Even head**: at full conversion, keeping only the even part of the final feature so the output is invariant.
- **Fit seed**: which set of 2,000 ImageNet train images was used to fit operators and twins (seeds 0, 1, 2). Separate from the run seed (0 throughout).
- **Flip augmentation**: randomly mirroring training images left to right.
- **Flip consistency**: share of validation images whose predicted class equals that of their mirror.
- **Folding (LayerNorm folding)**: absorbing normalisation scales and shifts and LayerScale into neighbouring weights without changing outputs.
- **G1 to G4**: gates, Section 8.3.
- **GELU**: the smooth non-linearity in the MLP; it is not an odd function.
- **GFLOPs**: billions of floating-point operations per image (2 x multiply-adds).
- **Healing**: short distillation fine-tune after surgery with the structure kept exact, Section 5.
- **Held-out loss**: the distillation loss on 5,000 held-out train images.
- **Hungarian matching**: an assignment algorithm; here a pairing without the constraint that pairs be mutual.
- **ImageNet (train, val)**: the 1,000-class image dataset; operators and healing use train images, all accuracy is measured on validation.
- **Involution**: a map that is its own inverse (doing it twice gives the identity), such as a mirror.
- **k**: number of converted blocks (the first k).
- **k-NN top-1**: accuracy of classifying each validation image by a vote of its 20 nearest neighbours in a labelled bank (100 train images per class), cosine similarity, temperature 0.07.
- **LayerScale**: a learned per-channel scale on each block's output in DINOv2.
- **Linear probe**: logistic regression trained on frozen features; measures how much class information the features hold.
- **Matched null / random structure**: a random involution with the same d+ and d- and a random pairing with the same counts, projected the same way; the control.
- **mIoU**: mean intersection-over-union across classes, the standard segmentation score.
- **MLP (fc1, fc2)**: the two-layer per-token network inside each block.
- **P (neuron involution)**: the permutation that swaps each twin pair and fixes self-twins.
- **P1 to P5**: pre-registered hypotheses, Section 8.4.
- **Pre-registration**: committing the test and thresholds to git before running, Section 8.
- **Principal angles**: angles between two subspaces; a mean cosine near 1 means the subspaces nearly coincide.
- **Projection**: replacing weights by their nearest exactly symmetric version, W_eq = (W + P W A)/2.
- **Protocol v1 / v2**: E1 and E1b's all-validation splits / the paper protocol (fit on train, 100k train bank, 50k validation queries, bf16 features).
- **Registers**: extra learnable tokens (4 in DINOv2-reg) that absorb global computation.
- **Residual stream**: the running token vectors to which every block adds its output.
- **Self-twin, twin pair, twin fraction**: a neuron matched with itself; two neurons that are mirror images; the share of neurons in pairs.
- **SVD (low rank), E_svd**: approximating a matrix by its top-r singular directions; E_svd is the energy discarded.
- **Symmetry tax**: the share of weights that is redundant because of mirror duplication, 1/2 - f (d+ - d-) / (2 h d) per matrix.
- **Tier 1, Tier 2**: the cheap metrics for every variant (k-NN, flip consistency, counts) and the fuller evaluation for final variants (linear probe, transfer, ADE20K, latency).
- **torch.compile, eager**: PyTorch with and without graph compilation; compiled is the fair timing mode.
- **V-local, V-exact**: per-layer operators (approximately symmetric overall) / one operator per span (exactly symmetric), Section 3.4.
- **ViT**: vision transformer.
- **X1 to X5, F1 to F7, A1 to A7, M1 to M4, D1 to D3, G**: experiment IDs from PAPER_EXPERIMENTS.md: Stage 0 checks, findings, ablations, method experiments (attention, latency, healing, composition), data-free readout, generality.
- **Zero-shot**: surgery with no retraining at all.

---

## 14. Suggested questions to ask the chat

1. Walk me through the closed-form fit A = V sign(Lambda) V^T: why does taking the sign of the eigenvalues give the best mirror operator?
2. Using block 1 of DINOv2-reg (1,168 pairs, 736 self-twins, d+ 489 at layer 1), work through the symmetry-tax formula by hand and explain each term.
3. Why exactly does a self-twin neuron have to ignore the odd directions? What would go wrong with an odd GELU input?
4. Explain, with a two-block toy example, why per-layer operators cannot make a residual network exactly equivariant, and why one operator per span can.
5. Why did E1's shared fit collapse to almost the identity? Would the energy-normalised fit used in V-exact have rescued E1, given that layers 9 to 11 still do not line up in plain DINOv2?
6. In plain words, why can healing rescue SVD from 6.49 to 77.57? What does that suggest about what SVD's damage really is?
7. Is the ceiling argument against G2 airtight? Could a student ever beat its teacher in k-NN after distillation, and what would that mean?
8. If I run E3 at full conversion, what result would I need for the comparison with healed SVD to be meaningful, and what could make E3 fail for reasons unrelated to the structure?
9. P2 failed: Mirror throws away more weight energy than SVD in early blocks yet loses less accuracy. What are plausible explanations, and which experiment in the repo's style would test them?
10. Why might SVD compound so badly on DINOv2-B with registers but not on DINOv2-L or DeiT-III?
11. Why are the Mirror models slower in practice despite fewer FLOPs, and what would a fused kernel need to do to turn the analytic 1.2x to 1.3x into real speed?
12. How should I present the flip-averaging baseline so that the healed V-exact model's exact invariance looks like a real contribution, not a trick?
13. Given G2's result, which framing gives the strongest honest paper: "Mirror beats SVD", "the symmetry tax exists and can be refunded exactly", or something else? What would each require from the remaining experiments?
14. What should the paper say about SigLIP2, given that the structure is weak but the pre-registered sub-check failed?
15. Which three numbers would a sceptical reviewer attack first, and how do the repo's data answer them?
