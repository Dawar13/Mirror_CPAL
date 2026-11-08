# PAPER EXPERIMENTS: everything needed for a complete CPAL submission

Read PROJECT_CONTEXT.md (v2) first. Experiment IDs below are used in file names, reports and the paper.

## 0. Overview

### 0.1 Dependency graph
```
STAGE 0 (blocking, about 1.5 h)
  X1 SVD-collapse check   X2 new eval protocol + rerun E1b arms   X3 seeds   X4 full LayerNorm folding   X5 PREREG_E2.md committed
        |
        v
STAGE 1 (all in parallel)
  F: findings (F1 tax, F2 drift, F3 artefacts, F4 symmetry vs low-rank rule, F7 visuals)
  A: zero-shot ablations (A1 to A5)
  M1: attention conversion (V-local and V-exact)
  G: generality across models (zero-shot)
  D: data-free readout pilot
        |
        v  (needs M1)
STAGE 2
  M2: efficient implementation + latency      A6 MLP-only vs MLP+attention      A7 V-local vs V-exact
  M3-pilot: healing pilot -> GATE G2 -> M3-grid: full healing grid with all baselines
        |
        v  (needs M3)
STAGE 3
  M4: symmetry + low-rank composition    V-exact full conversion healed (100% flip consistency)    Tier-2 evaluations on final variants
        |
        v
STAGE 4: 3-seed reruns of headline numbers, final tables and figures, FINAL_RESULTS.md
```

### 0.2 Must-have versus nice-to-have
- **Must-have:** X1 to X5, F1 to F4, F7, A1 to A6, M1 (V-local), M2, M3 (pilot and grid), G (DeiT-III, MAE, SigLIP2), Tier-2 evaluations, 3-seed headline.
- **Nice-to-have, in this order:** V-exact full conversion, D (data-free), M4, G (DINOv2-S and L), A7.

### 0.3 Gates (stop and report to Dawar before continuing)
- **G1 (after X1):** if the SVD collapse on the registers model is a bug, fix it and rerun every SVD number before any comparison.
- **G2 (after M3-pilot, pre-registered):** healed Mirror Surgery must beat a healed random-structure control by at least 2 k-NN points, and healed SVD by at least 3. Otherwise the structure is not what matters after healing; stop and discuss the framing.
- **G3 (after F3):** if neither registers nor masking restores the late layers, drop claim C3.
- **G4 (after the D pilot):** if layer-0 data-free agreement is below the pre-registered bar, drop C6.

### 0.4 Compute (estimates; shared GPU)
| Block | GPU-hours |
|---|---|
| Stage 0 | 1.5 |
| F1 to F4, F7 | 3 |
| A1 to A5 | 2 |
| G | 3 |
| D | 1 |
| M1 | 2 |
| M2, A6, A7 | 1.5 |
| M3 pilot | 3 |
| M3 grid | 8 |
| M4, V-exact full | 2.5 |
| Tier-2 evaluations | 4 |
| 3-seed headline | 2 |
| **Total** | **about 33** |

If time is short, trim in this order: ASVD at k = 6, SliceGPT at k = 6, DINOv2-S/L, M4.

### 0.5 Timeline
- 8 to 10 Nov: Stage 0; start Stage 1.
- 11 to 14 Nov: finish Stage 1; M2; M3 pilot and gate G2.
- 15 to 20 Nov: M3 grid, M4, V-exact full conversion, Tier-2 evaluations.
- 21 to 24 Nov: 3-seed reruns, final tables and figures, FINAL_RESULTS.md.
- 22 Nov to 3 Dec: writing, in parallel. 4 to 5 Dec: final checks and submission.

---

## 1. Evaluation suite (shared by every experiment)
- **Fit data:** operators and twins are fitted on ImageNet **train** images: 3 fixed seeds, 2,000 images each, unlabelled.
- **Tier 1 (every variant):**
  - k-NN top-1: memory bank of 100 train images per class (100k, fixed list, disjoint from fit and healing images); queries are the full validation set (50k); CLS after final norm, cosine, k = 20, T = 0.07.
  - Flip consistency on validation: same prediction for an image and its mirror.
  - Parameter count and analytic FLOPs for the whole model and for the converted blocks.
- **Tier 2 (final variants only):**
  - ImageNet linear probe: logistic regression on [CLS, mean patch token], trained on the 100k bank, tested on validation.
  - Transfer linear probes: CIFAR-100, Oxford Flowers-102, Oxford-IIIT Pets, DTD.
  - ADE20K linear segmentation, short protocol: frozen backbone, linear head on last-layer patch tokens, 448 x 448 crops, 8,000 iterations, batch 16, mIoU. The same protocol for every variant.
  - Latency (from M2).
- **Matched budgets:** every baseline matches the parameter count of the converted blocks to within 1%. FLOPs are reported separately and include basis rotations.

---

## STAGE 0 (blocking)

**X1, SVD collapse check.** On the registers model, recompute same-size SVD at k = 6 and 9. Checks:
1. rank = full gives identical outputs;
2. register tokens and LayerScale are handled the same way as in the plain model;
3. ranks match the stated parameter counts;
4. no fp16 overflow.

Report the cause, and rerun the E1b SVD arms if needed (gate G1).

**X2, new protocol.** Implement the train-fit / val-eval protocol (Section 1). Rerun the E1b arms (plain, registers, shared 1 to 6) under it so all numbers are comparable. Keep the E1b reports as they are.

**X3, seeds.** Three fit seeds for every headline number; report mean and standard deviation.

**X4, full LayerNorm folding (SliceGPT-style).** Fold LayerNorm centring and affine parameters, plus LayerScale, so every norm becomes RMSNorm on a mean-zero residual stream that commutes with any orthogonal A. Test: the folded model's outputs equal the original's to within 1e-4 (fp32), on 256 images.

**X5, pre-registration.** Write `PREREG_E2.md` with the hypotheses below and commit it before running F3, F4, G, D or M3. Record the commit hash in every later report. Thresholds can be edited before the commit, never after.
- **P1 (artefacts):**
  - plain DINOv2: excluding the top 1% highest-norm tokens cuts the mean e(A_l) over layers 9 to 11 by at least 2x;
  - registers model: mean e(A_l) over layers 9 to 11 is at most 2x its mean over layers 1 to 6.
- **P2 (rule):** the sign of (activation-weighted E_svd minus activation-weighted s) predicts which method has the smaller k-NN drop in single-block surgery in at least 75% of block and model cases.
- **P3 (healing, gate G2):** at k = 9 (registers model, MLP plus attention, V-local), all of:
  - healed Mirror Surgery is at least 2.0 points above the healed random-structure control;
  - it is at least 3.0 points above healed SVD;
  - it is within 2.0 points of the original.
- **P4 (generality):**
  - DeiT-III and MAE: mean MLP breaking share at most 0.25 over their first half of blocks, and e(null)/e(A) at least 5;
  - SigLIP2's breaking share is higher than DINOv2-reg's.
- **P5 (data-free):**
  - the layer-0 data-free operator's even subspace agrees with the data-driven one (mean cosine of principal angles at least 0.8);
  - weight-only twin pairing in block 1 recovers at least 70% of the data-driven pairs.

---

## STAGE 1 (parallel)

### Track F: findings
**F1, symmetry tax.** For both models, all 12 layers, 3 seeds, report per layer:
- d+ and d-;
- twin pairs and self-twins, with the histogram of twin scores;
- breaking shares of fc1 and fc2 against the matched null;
- the tax predicted by the formula in Section 5.3 versus the parameter saving actually achieved.

Paper: Figure 2 (tax against depth, both models).

**F2, operator drift.**
- Matrix of agreement tr(A_l A_m)/d for all layer pairs.
- Agreement restricted to each layer's top-energy subspace (95% of activation energy).
- Principal angles between the even subspaces of neighbouring layers.

Paper: Figure 3a.

**F3, artefact tokens (P1).** Plain DINOv2:
- find high-norm tokens per layer (top 1% by norm, and above 3x the median);
- e(A_l) with and without them;
- the fraction of artefact positions in the mirrored image that sit at mirrored positions of artefacts in the original.

Registers model: e(A_l) for all layers. Paper: Figure 3b.

**F4, symmetry versus low rank (P2).** For each block l, models DINOv2-reg, plain DINOv2 and DeiT-III:
1. Compute s and E_svd at matched parameters, weight-only and activation-weighted.
2. Run single-block surgery (only block l converted) with mirror projection and with SVD at matched parameters.
3. Record each method's k-NN drop.

Report how often the rule picks the winner, with a scatter plot of (E_svd - s) against (SVD drop - mirror drop). Paper: Figure 4.

**F7, visuals.**
- For blocks 1, 3 and 6, the 8 strongest twin pairs: the top-activating train patches for each neuron of a pair (the twin's patches should be mirror images).
- Layer-0 patch-embedding filters paired with their mirrored partners.

Paper: Figure 1 (teaser).

### Track A: zero-shot ablations (k-NN, k = 6 and 9, 3 seeds, registers model)
- **A1, operator:**
  - per-layer;
  - shared over the span (energy-normalised);
  - shared over all layers (the E1 design);
  - identity A = I (forces all features even, so odd weights are zeroed);
  - random involution with matched d+ and d-.
- **A2, pairing:**
  - mirror matching;
  - random pairing with matched counts;
  - all self-twins (no pairs);
  - generic-similarity pairing (pairs by plain weight cosine, not mirror; tests whether any redundancy would do);
  - plain Hungarian without the involution constraint.
- **A3, fit-set size:** 100, 500, 2,000 and 10,000 images, plus 0 if D succeeds.
- **A4, block selection at matched parameters:** first k, last k, or lowest breaking share first.
- **A5, projection strength:** W <- W - alpha (W - W_eq), alpha in {0, 0.25, 0.5, 0.75, 1}.

Paper: Table 3.

### M1: attention conversion
- **Head pairing per layer:** score each pair of heads (h, h') by the Procrustes fit between [q; k] of head h on originals and head h' on mirrored-unflipped images; involution matching over the 12 heads.
- **Head maps:** fit B_h on [q; k] and C_h on v (orthogonal; involutions for self-heads). Project W_Q, W_K, W_V, W_O and biases onto the equivariant form (Section 5.4) by averaging over each orbit.
- **Report:** self-heads versus twin pairs per layer, and attention breaking shares.
- **Variants:**
  - V-local: per-layer A_l.
  - V-exact: one A* over the converted span. If the span starts at block 1, also project the patch embedding (A* W_pe = W_pe with flipped kernels), the position embeddings (A* pos(r, c) = pos(r, W-1-c)), the CLS token (even) and the registers (matched among themselves).
- **Exactness test (V-exact):** inside the converted span, mirrored-image features equal A* times the original features to within 1e-4. A fully converted model must reach 100% flip consistency.
- **Zero-shot k-NN** for k in {3, 6, 9, 12}, MLP plus attention.

### G: generality (zero-shot)
- **Models** (check exact timm or Hugging Face names):
  - DeiT-III ViT-B/16 (supervised);
  - MAE ViT-B/16;
  - SigLIP2 ViT-B/16 (expected to be weaker);
  - DINOv2 ViT-S/14 and ViT-L/14 with registers (nice-to-have).
- **For each:** the E1b rows 1 to 4, then zero-shot surgery at k = L/2 against SVD.
- **MAE:** its CLS gives weak k-NN, so use the Tier-2 ImageNet linear probe as its accuracy measure.

Paper: Table 2 (tests P4).

### D: data-free readout (pilot, then decide)
- **D1:** layer-0 operator from weights only: flipped patch-embedding kernels plus mirrored position embeddings, using the same sign-of-cross-covariance formula. Compare with the data-driven A_0 (P5).
- **D2:** twin pairing from weights only, given A: match rows of W1 against rows of W1 A by cosine. Agreement with the data-driven pairing.
- **D3 (stretch):** propagate operators through blocks 1 to 3 from weight relations only, then zero-shot surgery with the data-free operators.

---

## STAGE 2 (needs M1)

**M2, efficient implementation and latency.**
- Implement the converted blocks in even/odd coordinates:
  - MLP: twin butterfly (u + v, u - v); self-twins read even features only;
  - attention: block-structured projections;
  - V-local: one basis rotation at each block boundary, folded into weights where possible.
- Check it matches the projected dense model (max abs difference under 1e-3 in bf16).
- Measure parameters, analytic FLOPs including rotations, and latency and throughput (batch 1, 64 and 256; bf16; torch.compile; median of 100 runs after warm-up).
- Compare with the original, SVD and width pruning at matched parameters.
- Report honestly; the real speed-up is expected to be lower than the FLOP saving.

Paper: Table 1 columns.

**A6:** MLP-only versus MLP plus attention conversion, at the same k. **A7:** V-local versus V-exact: accuracy, flip consistency and latency.

**M3-pilot, healing (gate G2, tests P3).**
- **Recipe (the same for every method):**
  - student = converted model, trained in its constrained form so the symmetry stays exact (train only the shared parameters); teacher = original model;
  - loss: normalised MSE on final-layer CLS and patch tokens, plus the outputs of the converted blocks;
  - data: 100k ImageNet train images, disjoint from the fit and k-NN bank images; random resized crop with flips;
  - 2 epochs, batch 256, AdamW, cosine schedule, bf16;
  - learning rate chosen per method from {1e-5, 5e-5, 1e-4} on a held-out 5k train split, never on validation.
- **Pilot runs:** k = 9, registers model, MLP plus attention, V-local:
  - Mirror Surgery;
  - SVD at matched parameters;
  - random-structure control: a random involution with matched d+ and d-, plus a random pairing with matched counts, projected and then healed with the same budget.
- **Stop and report at gate G2.**

**M3-grid (only if G2 passes).**
- **Methods at matched parameters:**
  - Mirror V-local;
  - Mirror V-exact;
  - SVD;
  - activation-aware SVD (whitened, SVD-LLM style);
  - SliceGPT-style PCA slicing of the residual stream over the span;
  - structured MLP width pruning (activation-weighted importance).
- **Levels:** k = 6 and 9. V-exact is also run at k = 12 (full conversion).
- **Seeds:** 1 for the grid; 3 for the headline (Mirror against the best baseline at k = 9).

Paper: Figure 5 (accuracy against parameters, zero-shot and healed) and Table 1.

---

## STAGE 3 (needs M3)
- **M4, composition:** inside the converted blocks, apply SVD to the even and odd blocks to remove 60% and 70% of parameters. Compare with SVD alone at matched parameters, zero-shot and after healing at one level.
- **V-exact full conversion, healed:** all 12 blocks of DINOv2-reg converted, giving an exactly mirror-equivariant model with 100% flip consistency. Report accuracy and cost. (Nice-to-have, high impact.)
- **Tier-2 evaluations** on the final variants:
  - the original;
  - the best Mirror variant at k = 6, 9 and 12;
  - the best baseline at k = 9.

---

## STAGE 4: finish
- Rerun all headline numbers with 3 seeds.
- Build every table and figure from scripts:
  - Fig 1 teaser (F7);
  - Fig 2 tax (F1);
  - Fig 3 drift and artefacts (F2, F3);
  - Fig 4 rule (F4);
  - Fig 5 compression curves (M3);
  - Table 1 main (M2, M3, Tier 2);
  - Table 2 generality (G);
  - Table 3 ablations (A1 to A7);
  - Table 4 data-free (D, if it survives).
- Write `results/FINAL_RESULTS.md`: every number with its source file, seed and commit hash, every pre-registered check marked pass or fail, and every deviation from this plan explained.

## Kickoff prompt for Claude Code
"Read PROJECT_CONTEXT.md and PAPER_EXPERIMENTS.md. Do Stage 0 first: X1 to X4, then write
PREREG_E2.md (X5) and stop so I can review it before you commit. After I approve, run Stage 1
tracks in parallel through a job queue that respects the GPU memory cap, report at each gate, and
keep results/STATUS.md updated with what has run, what passed and what is next. Never change
pre-registered thresholds after committing them."
