# FINAL_PHASE_PLAN.md: Mirror Surgery, final experiments (CPAL 2027)

Version 2, 19 November 2026. Owner: Dawar.

**Executor:** Claude Code. Coding is done on the laptop. GPU work runs on one H200 VM, which Dawar starts.

**This file is self-contained.** It replaces every earlier plan and open-items list. Section 3 maps every suggestion received so far (AI reviewer, GPT review, the open-items notes) to an item here, so nothing else needs to be read to run it. The only other things Claude Code needs are already in the repository:
- `CLAUDE.md`;
- the `mirror/` library;
- `notes/paper/method_spec.md` and `notes/paper/numbers.md`;
- `results/` (in particular `results/MUST-09` for the reproduction gate);
- `scripts/collect_next_phase.py`, `PAPER_NUMBERS.json` and `notes/decisions.md`.

**Deadlines:**
- abstract registration: 23 November (verify the date on OpenReview);
- results freeze: 28 November;
- submission: 5 December 2026, AoE.

**Hard ceiling:** Module A must finish within **15 wall-clock hours of H200 VM time**, setup included. Section 9 explains how this is enforced. Module B is not planned.

---

## 0. How to use this file

1. Read Sections 1 to 7 before writing any code.
2. **Phase C (coding; no GPU; no model or dataset downloads on the laptop).**
   - Build everything in Section 7 and pass the CPU tests.
   - Commit `PREREG_FINAL_PHASE.md` (the thresholds in Section 6) in its own commit.
   - Stop and report.
3. **Phase G (GPU), only after Dawar provides the VM.** Run the waves in Section 8 within the 15-hour ceiling.
4. **Rules that never change:**
   - Every number in the paper comes from a file in `results/`.
   - Thresholds never change after the pre-registration commit.
   - Every deviation goes into `notes/decisions.md` with the reason.

---

## 1. Where the paper stands

**Main model:** DINOv2-B/14 with registers (`dinov2_vitb14_reg`, 86.6M parameters).

**Healed full conversion (MUST-09).** Mirror Surgery and SVD-LLM each used 3 healing seeds; for Mirror all three share one fitted structure.

| Method | k-NN | Notes |
|---|---|---|
| Original | 78.95 | |
| Mirror Surgery | 78.34 ± 0.02 | |
| SVD-LLM (whitening step only) | 78.06 ± 0.01 | |
| SliceGPT-style (1 seed) | 77.89 | |
| Plain SVD (1 seed) | 77.02 | |
| Random structure | 69.93 | |
| Two-pass flip averaging | 79.00 | |

- **Before healing:** Mirror 53.84 against SVD-LLM 75.80.
- **Exactness:** 100% flip consistency in float32. The bfloat16 healed model agrees for 99.73% of images.
- **Cost:**
  - parameters 0.765 of the original (66.2M);
  - analytic FLOPs 0.774;
  - throughput 0.72x the original (even/odd form, plain PyTorch).
- **Healing:** 780 steps of 256 images on 100k ImageNet-train images. Every method chose the top of the learning-rate grid (1e-4). One heal took 27 minutes on a shared H200, mostly data loading and evaluation.

**Other results:**
- **Downstream:** Mirror matches or beats SVD-LLM on the linear probe, four transfer datasets, ADE20K, ImageNet-V2 and Sketch.
- **CUB:** the even part predicts species and is exactly at chance on facing direction; the odd part predicts facing direction at 99.2%.
- **Structure in 14 models:** strong in the early blocks. Up-down structure is about as strong as left-right. CLIP and DINOv3 have it without flip training.
- **Earlier causal pilot:** small ViTs trained from scratch on ImageNet-100 formed almost no twins even with left-right flips (twin share 0.060 at 64 px and 0.065 at 96 px).
- **Pre-registration record:** 47 rows. Of the 40 core predictions, 17 passed, 16 failed and 7 could not be scored; the other 7 rows are optional UP items.

**Weaknesses this plan must remove:**
1. Healed results exist for one model at one size.
2. The error bar ignores which images were used to fit the structure.
3. Only half of SVD-LLM was run; IO-SVD is missing.
4. "Is it just distillation?" is only partly answered.
5. There is a single compression point, not a curve.
6. There is no control showing the discovery pipeline does not invent structure.
7. The exactness wording is loose: the float32 audit error of 2.51e-5 failed a 1e-5 bar.
8. Flip averaging is not compared at equal compute.
9. The k-NN uses a 100k bank, so it is not comparable with published numbers.
10. The model is slower than the original.
11. The cause of the twins is post hoc.

---

## 2. The story and the claim-evidence map

The paper tells one story: **discovery, then mechanism, then method, then representation, then consequence, then evidence.**

| # | Claim | Evidence |
|---|---|---|
| 1 | **Discovery.** Pretrained vision transformers store many early features twice, as mirror twins | 14-model structure (done); stability across 5 independent fits (A02); the pipeline finds nothing in null controls (A03) |
| 2 | **Mechanism.** One mirror operator on the residual stream splits it into even and odd directions; a twin's weights are the mirror of its partner's | Lemma 1 and Section 3 of the draft (done); operator agreement across fits (A02) |
| 3 | **Method.** Mirror Surgery: closed-form operator, projection onto exactly equivariant weights, short healing | Propositions 1 to 3 (done); operator choice (A01); exactness audit with precise definitions (A09) |
| 4 | **Representation.** The even part encodes what the object is; the odd part encodes which way it faces | CUB (done) |
| 5 | **Consequence.** An exactly mirror-invariant model with 23.5% fewer parameters, on the accuracy-size frontier, and it stacks with low rank | A05, A06, A07, A11, A12 |
| 6 | **Generality and rigour.** Holds on three models, against nine baselines and controls at matched size and identical healing, with stage ablations, standard metrics and honest statistics | A04, A05, A08, A10, A13 |

**Hypothesis, not a claim:** early pairing reflects natural-image statistics (Lemma 2). The paper says "we characterise and exploit the structure; we do not establish its cause". A15b provides one causal probe.

---

## 3. Coverage map: every suggestion received, and where it is handled

### 3.1 AI reviewer's six recommendations

| # | Recommendation | Where | Decision |
|---|---|---|---|
| 1 | Heal a second model (DeiT-III-B or DINOv2-L) | A08 | Both: DeiT-III-B with 3 seeds (standard top-1), DINOv2-L with 1 seed (indicative) |
| 2 | Heal from more than one fitted structure | A02, A13 | Heals from fit sets 1 and 2 and from a COCO fit; structure variance enters the error bar |
| 3 | Full SVD-LLM, including its update step | A05 | Run, 3 seeds |
| 4 | Dense 66M model distilled with the same 780 steps | A04 | Width-pruned and block-dropped dense students, plus the original distilled ("healing without surgery") |
| 5 | Fused kernel for the even/odd form | A12b | Best-effort Triton kernel (laptop work, 0.3 h of GPU timing). Speed claim only if ≥ 1.1x; otherwise a profiled, honest limitation |
| 6 | Test the Fourier explanation with 90-degree rotations | A15b, A14 | Modified. Training small ViTs from scratch is not repeated, because the earlier pilot formed no twins even with left-right flips. Instead, a short rotation-invariance fine-tune of DINOv2-B tests whether rotation pairing rises in the early blocks. The Fourier figure is averaged over 1,000 ImageNet images (A14) |

### 3.2 GPT review priorities

| Priority | Where |
|---|---|
| P1. Robustness of the mirror operator across samples | A02 |
| P2. Null controls (random transformation, random pairing, shuffled data, random structure) | A03, A04 |
| P3. Full pipeline on a second architecture | A08 |
| P4. Apples-to-apples comparison at several compression ratios | A05, A06 |
| P5. Flip averaging at equal compute | A07 |
| P6. Stage ablation (surgery, symmetry, healing) | A04 |
| P7. Seeds and statistics | A05 (3 seeds), A13 |
| P8. Mathematical audit of exact equivariance | A09 |
| P9. Origin of the twins, without overclaiming | A14 (wording), A15b (probe) |
| P10. Systems story | A12, A12b |
| P11. One sharp story | Section 2, A14 |
| "What not to do" | Respected: no new datasets, no new compression method (A01 only changes how the operator is fitted, in closed form), no speed claim without measurement, no large causal study |

### 3.3 Open-items notes (items 1 to 14 and decisions)

| Item | Where |
|---|---|
| 1 to 6 | As in Section 3.1 |
| 7. Full-bank k-NN | A10 |
| 8. MUST-11 P11.2 pixel margins | A10 |
| 9. IO-SVD baseline | A05 |
| 10. ImageNet classification with the DINOv2 linear protocol | A10 |
| 11. A* after healing from fit sets 1 and 2; A* fitted on a non-ImageNet set | A02 |
| 12. Stability of B_h, C_h and head pairings across fit seeds | A02 |
| 13. SigLIP 2 projected in the whitened basis | A15a |
| 14. LEE before and after conversion | Not run: flip consistency is the exact metric, and LEE measures approximate equivariance, so it says nothing new about an exactly equivariant model |
| 14. Dihedral (left-right plus up-down) conversion of the early blocks | Module B (not planned) |
| Teaser sharpness (Fig. 1a) | A14: regenerate the tiles on the VM (minutes) |
| Fourier figure over ~1,000 ImageNet images | A14 (CPU) |
| Anonymous author block, no acknowledgements | A14 |
| Already answered in the draft by the 17 Nov review response (A* fit-set size, head pairing per block table, SigLIP 2 diagnosis, the attention step of the exactness proof, related work on REViT, Equi-ViT and IO-SVD) | A14: keep these in the rewrite |

---

## 4. MODULE A (the complete plan)

Each item lists the question it answers, what to run and the outputs. The pass bars are in Section 6 and are pre-registered.

### A01. Lock the recipe: damage-minimising operator, plus a reproduction gate

- **Question:** can the accuracy lost before healing (53.84) be cut in a principled way, and does that lift the healed number?
- **Idea.** Today A* is fitted to match activations. But the damage comes from the weight change that projection makes.
  - For fixed twin pairings P_l, minimising Σ_l (‖W1_l A − P_l W1_l‖² + ‖A W2_l − W2_l P_l‖²) over symmetric orthogonal A is the same as maximising tr(A·M), with M = Σ_l sym(W1_lᵀ P_l W1_l + W2_l P_l W2_lᵀ).
  - Add the patch-embedding and position-embedding terms. Normalise each layer's term by its weight energy, as the activation fit does.
  - The solution has the Lemma 1 closed form: A = V sign(Λ) Vᵀ for the eigen-decomposition of M.
  - Alternate pairing and refitting for at most 3 rounds, stopping when fewer than 1% of the pairings change.
  - Attention head maps are refitted given A, as now.
- **Runs:**
  1. Fit A*_w and run zero-shot full conversion.
  2. Heal A*_w-Mirror for 2 seeds.
  3. **Reproduction gate (Gate P):** rebuild Mirror with the current A* and SVD-LLM in the new optimised pipeline, 3 seeds each. These runs double as A05's main seeds.
  4. If A*_w is adopted, heal one more A*_w seed so the main Mirror arm has 3 seeds. All later Mirror runs then use the adopted operator.
- **Outputs:** `results/A01/`: before-healing and healed k-NN for both operators; d+ and d− of A*_w; agreement with the current A*.

### A02. Stability of the operator and the twins across independent fits

- **Question:** "Did you find this operator only because of the images you happened to use?"
- **Fits:**
  - ImageNet-train fit sets 0, 1 and 2 (already fitted), plus a new disjoint set 3, 2,000 images each;
  - one fit on 2,000 COCO images (fall back to Places365 if COCO is not available).
- **Measure, per fit:**
  - d+ and d− of the stream operator, and the per-layer d−;
  - the eigenvalue spectrum of sym(C);
  - operator agreement: ‖A_a − A_b‖_F² / (2d), and the energy-weighted mean cos² of the principal angles between odd subspaces;
  - twin agreement: the share of strong pairs (correlation at least 0.5) in blocks 1 to 6 found by both fits;
  - agreement of the head maps (self or swap), and of B_h and C_h (relative Frobenius distance);
  - zero-shot k-NN of the full conversion.
- **Heals:** one seed each from fit sets 1 and 2 and from the COCO fit. Fit set 0 already has 3 seeds from A01.
- **Outputs:** `results/A02/`, Table "Operator stability", Figure "spectra and twin agreement across fits".

### A03. Discovery null controls: does the pipeline invent structure?

Run the full discovery pipeline (operator fit, then pairing) on inputs that contain no mirror relation.
- **C1, shuffled correspondence:** pair the tokens of image a with the mirrored, re-aligned tokens of a different image b (a random derangement of the images).
- **C2, no re-alignment:** compare the token at (r, c) in the original with the token at (r, c) in the mirrored image, without flipping the grid back.
- **C3, non-symmetry transformation:** replace the mirror with a fixed random involutive permutation of the pixels inside every patch. It keeps pixel statistics but is not a symmetry of natural images.
- **C4, random operator null:** a random involution with matched d+ and d−, plus a random pairing, 20 draws. This exists already; report it systematically.

For each control, report:
- d−;
- the shape of the eigenvalue spectrum;
- the twin fraction in blocks 1 to 6;
- the ratio to null;
- zero-shot k-NN after surgery, where that is defined.

**Healed control:** H1 uses the true operator with a random pairing, 1 seed. The healed random structure (random operator and random pairing) is rerun in the new pipeline as part of A04.

**Outputs:** `results/A03/` and Table "Discovery controls".

### A04. Stage ablation: what surgery, symmetry and healing each contribute

Produce one table with these rows, all DINOv2-B and all healed with the identical recipe unless stated:

| Row | Symmetry enforced | Compression | Healed | Source |
|---|---|---|---|---|
| Original | no | no | no | done |
| Original distilled with the same recipe ("healing without surgery") | no | no | yes | **new, 1 seed** |
| Surgery, before healing | yes | yes | no | A01 |
| Surgery + healing (main) | yes | yes | yes | A01 / A05 |
| Random structure + healing | random | yes | yes | rerun in the new pipeline, 1 seed |
| True operator + random pairing + healing | partial | yes | yes | A03 H1 |
| Compression without symmetry + healing | no | yes | yes | A05 (SVD-LLM, full SVD-LLM, IO-SVD, SliceGPT, plain SVD) |
| Width-pruned dense student + healing | no | yes | yes | **new, 1 seed** |
| Block-dropped dense student + healing | no | yes | yes | **new, 1 seed** |

**The dense students:**
- **Width-pruned:** FLAP-style fluctuation scores. Prune MLP hidden units and attention heads to 66.2M (±1%, rounding towards more parameters for the baseline), with bias compensation.
- **Block-dropped:** remove 3 of the 12 blocks, chosen greedily by the smallest k-NN loss on the method-selection set. That gives about 65.3M (75.4%); print the gap.

### A05. Complete baseline set at matched size, 3 seeds

**Methods:** all at 66.2M (±1%, rounding towards more parameters for the baseline), on the same matrices after the same LayerNorm folding, healed with the identical recipe.

| Method | Seeds |
|---|---|
| Mirror Surgery (the operator chosen in A01) | 3 |
| SVD-LLM, whitening only (current baseline) | 3 |
| **Full SVD-LLM** (whitening plus sequential low-rank update; port from the official SVD-LLM code, verify the repository) | 3 |
| **IO-SVD** (arXiv 2605.15626, input-output whitening with adaptive rank; code at github.com/mint-vu/IO-SVD; adapt to ViT linear layers; allocate rank to match total parameters) | 3 |
| SliceGPT-style | 2 |
| Plain SVD | 1 |
| Random structure, dense students, original distilled | 1 (A04) |

**Learning rate:**
- New methods (full SVD-LLM, IO-SVD, dense students, original distilled) pick their rate from {1e-4, 2e-4} with 200-step runs, using the held-out distillation loss on the LR-selection set. Every method in the earlier phase chose the top of the old grid (1e-4), so the grid moves up.
- Existing methods keep their recorded choice; the paper cites the earlier check at 2e-4 for Mirror and SVD-LLM.
- Validation images are never used for any choice.

**Pairing:** within each seed, every method sees the identical batches in the identical order, so all comparisons are paired.

### A06. Accuracy-size frontier and stacking with low rank

**Sizes**, as fractions of the full model's parameters: 0.765 (from A05), 0.65, 0.50 and 0.40.

| Arm | Sizes | Exactly invariant? |
|---|---|---|
| SVD-LLM (whitening) | 0.65, 0.50, 0.40 (0.765 from A05) | no |
| Mirror, full | 0.765, from A05 | yes |
| **Mirror + SVD-LLM**: SVD-LLM applied inside the even and odd blocks of the converted model, with rank allocated by whitened energy and parameters matched | 0.65, 0.50, 0.40 | yes |
| Mirror on the first 9 blocks only (**price of exactness**) | 0.766 | no (partial) |

- **Seeds:** 1 per point, plus 1 extra seed at 0.50 for SVD-LLM and for Mirror + SVD-LLM.
- **The 9-block point** is almost the same size as the full conversion (0.766 against 0.765) but leaves the last 3 blocks unconverted. The accuracy difference between the two is what exactness costs.
- **Upper anchor of the curve:** the original distilled with the same recipe (A04), at size 1.0. The zero-shot points at 0.87 and 0.81 from the existing table are shown for reference only.
- **Prediction, written into the pre-registration before any run:**
  - acc(Mirror + SVD, B) = acc_SVD(B / 0.765) − δ, where δ = (distilled original − healed Mirror at 0.765);
  - acc_SVD is interpolated on a log-size axis between the measured SVD-LLM points and the distilled original.
- **Outputs:** `results/A06/` and the frontier figure, with exactly invariant points marked.

### A07. Exact invariance built in, against flip averaging at equal compute

- **Question:** "Why change the model, when running it on an image and its mirror and averaging gives invariance?"
- **Runs** (evaluation only, using A05 and A06 checkpoints):
  - two-pass flip averaging of the SVD-LLM model whose doubled measured FLOPs is closest to Mirror's single pass (about the 0.40 point);
  - two-pass flip averaging of the released DINOv2-S/14 reg;
  - two-pass flip averaging of the original (79.00, the ceiling).
- **Compare:** Mirror (1 pass) and Mirror + SVD at matched compute.
- **Metrics:** k-NN, linear probe, flip consistency, analytic FLOPs and measured throughput.

### A08. Second and third models: the full pipeline

The pipeline, for each model:
1. fit the operator;
2. find the even/odd split;
3. find the twins and head maps;
4. run surgery;
5. heal;
6. evaluate;
7. measure flip consistency;
8. check the parameter saving (formula against stored).

**DeiT-III-B/16** (timm; verify the exact checkpoint ID and its flip augmentation). It is supervised and has its own classifier.
- The head reads the even part of the class token, so the classifier input is ½(I + A*) applied to the class token.
- Healing uses the same loss plus a logit-distillation term (state it in the pre-registration).
- **Learning rates:** reuse each method's DINOv2-B choice, with one 200-step check at the other grid value.
- **Metric:** standard ImageNet top-1 from its own head, plus k-NN.
- **Methods:** Mirror ×3, SVD-LLM ×3, random structure ×1.

**DINOv2-L/14 reg** (24 blocks, d = 1024). Runs last among the main items, and is on the cut list.
- **Methods:** Mirror ×1 and SVD-LLM ×1, reported as indicative.
- **Metrics:** k-NN, flip consistency, parameter saving.

**Outputs:** `results/A08/` and Table "Other models".

### A09. Exactness audit with precise definitions

**Three levels, never conflated:**
- **E1, exact in real arithmetic:** Proposition 3, a theorem.
- **E2, numerically exact:** the equivariance error is no larger than the rounding error the model itself makes at that precision.
- **E3, bitwise identical:** features or predictions are identical bit for bit.

**For each component**, state the theoretical condition and measure the maximum relative equivariance error. Components:
- patch embedding;
- position embeddings;
- class and register tokens;
- folded LayerNorm;
- attention Q, K, V and O, and the head maps;
- MLP fc1, GELU, fc2 and biases;
- residual additions;
- final norm;
- even projection;
- DeiT classifier head.

Measure in float64, float32 and bfloat16, for both the dense-stored and the even/odd forms, on 1,000 validation images.

**Rounding floor:** run the original model and the Mirror model in float32 and in float64 on the same images, and record the per-tensor relative deviation. The MUST-02 value (2.51e-5) is judged against this floor.

**Also report:**
- flip consistency in float32 and bfloat16, for both forms;
- a bitwise check of the even/odd form. A sign flip is exact in floating point; the only expected source of difference is the order of summation in attention.

**Outputs:** appendix table and `results/A09/`. The paper states which level holds where.

### A10. Standard metrics, comparable with published numbers

- **Full-bank k-NN:** the bank is all 1.28M ImageNet-train images, k = 20, T = 0.07. Run it for:
  - the original;
  - Mirror (3 seeds);
  - SVD-LLM (3 seeds);
  - full SVD-LLM and IO-SVD (1 seed each).
- **DINOv2 linear protocol** on full ImageNet-train features. Verify the protocol details: which tokens are concatenated, the hyperparameter grid, the augmentation. Run it for the original, Mirror (3 seeds) and SVD-LLM (3 seeds).
- **Downstream for SVD-LLM seeds 1 and 2:** linear probe, ImageNet-V2 and deduplicated Sketch. This matches Mirror's 3 seeds.
- **P11.2:** rerun the segmentation evaluation with the top-two pixel margins logged. This closes the last pending prediction.

### A11. Healing ceiling (headline accuracy)

- **Question:** is the 0.61-point gap to the original a limit of the method, or of the short healing budget?
- **Why we suspect the budget:** the current healing curve plateaus between steps 300 and 780 on 100k images. That points to the number of distinct images, not the number of steps.
- **Runs:**
  - extended healing at about 3x the images seen (about 2,340 steps of 256), drawn from a larger ImageNet-train subset that is disjoint from the fit, selection and k-NN-bank sets, with the same schedule shape;
  - Mirror (the A01 operator) ×1 and SVD-LLM ×1 under the same budget.
- **Report:** k-NN, full-bank k-NN and the linear protocol.

### A12. Systems table and profiling

**Models:** the original, Mirror (dense-stored and even/odd), SVD-LLM, IO-SVD, the dense students, and flip averaging.

**Report:**
- parameters and parameter memory (MB in bf16);
- analytic FLOPs;
- measured throughput at batch 1, 64 and 256 (bf16, torch.compile, median of 100 runs after warm-up);
- latency;
- peak memory.

**Profile the even/odd form:**
- the share of time in GEMMs, in butterflies and permutes, and in the rest;
- achieved TFLOP/s per GEMM shape;
- the effect of the awkward sizes (700 and 68 are not multiples of 8).

**Output:** the systems table, plus a short note on why the even/odd form is slower in plain PyTorch.

### A12b. Fused even/odd kernel (best effort)

- **What:** a Triton kernel for the even/odd MLP.
  - Pad 700→704 and 68→72;
  - store twins in a pre-permuted layout;
  - fuse butterfly + GELU + butterfly;
  - keep the stream in the A* basis.
  - MLP first; attention only if time allows.
- **Order:** built only after every other Phase C task passes, so it never delays Phase G.
- **Checks:** outputs must match the even/odd reference within float32 rounding.
- **GPU:** timing takes about 0.3 h, inside the ceiling.
- **Claim:** a speed claim only if the bar in Section 6 is met. Otherwise the paper reports the profiled limitation (A12) and names the kernel as future work.

### A13. Statistics protocol (pre-registered, replaces post hoc intervals)

- **Hierarchical paired bootstrap:**
  - resample validation images, paired across methods;
  - where available, also resample fit seeds and healing seeds;
  - 10,000 resamples.
- **Report:**
  - mean ± standard deviation over seeds;
  - the 95% interval of each difference;
  - the effect size (difference divided by the pooled seed standard deviation).
- **Previous phase:** numbers computed post hoc stay labelled as post hoc.

### A14. Writing and figures (Claude Code drafts, Dawar edits)

1. Rebuild the paper around the Section 2 story and its claim-evidence map.
2. **Abstract:**
   - lead with the discovery;
   - then the method;
   - then the strongest measured consequences;
   - mention pre-registration in one clause.

   Register the abstract on 23 November using only measured claims.
3. Write the exactness definitions (A09) into the method section.
4. Present the cause as a hypothesis. Never write that flip augmentation causes the twins. Cite the earlier pilot (no twins from scratch) and A15b.
5. **Limitations:**
   - speed, unless A12b passes;
   - SwiGLU models have few twins;
   - bfloat16 behaviour;
   - the converted model is tied to its position grid;
   - the cause is untested beyond A15b.
6. **Appendix:** the full pre-registration record, worded as "of 40 core predictions, 17 passed, 16 failed and 7 could not be scored (47 rows including 7 optional items)", plus every new prediction from this phase.
7. **Keep in the rewrite** what the draft already answers:
   - the A* fit-set size;
   - the head pairing per block table;
   - the SigLIP 2 diagnosis;
   - the attention step of the exactness proof;
   - related work on REViT, Equi-ViT and IO-SVD.
8. **Figures:**
   - regenerate the Fig. 1 teaser tiles at full resolution on the VM (minutes);
   - average the Fourier figure (Lemma 2) over about 1,000 ImageNet images instead of 9 sample photos (CPU);
   - build the new frontier, stability and controls figures (Section 10).
9. Keep the author block anonymous and omit acknowledgements.

### A15. Scope probes (last in the queue, first to cut)

- **A15a. SigLIP 2 in the whitened basis.**
  - Project SigLIP 2 B/16 in the basis whitened by its activation covariance, zero-shot.
  - Question: does the collapse disappear? This explains the one model where surgery fails.
  - Structure and zero-shot only, about 0.3 h.
- **A15b. Cause probe.**
  - Fine-tune DINOv2-B for 780 steps with a rotation-invariance distillation target: the student sees rot90(x), and the target is the teacher on x.
  - Measure 90-degree orbit structure (ratio to null, share of neurons in 4-cycles) in blocks 1 to 6 before and after.
  - Prediction: if data symmetry shapes pairing, rotation structure rises. Report either way, as evidence rather than proof.
  - About 1 heal-unit.

---

## 5. MODULE B: not planned

Module A is the complete plan. These items are listed for the record. They run only if Dawar asks after seeing Module A's results, and they sit outside the 15-hour ceiling.

| ID | What |
|---|---|
| B01 | Up-down symmetry in blocks 1 to 6 (Klein group; generators must commute; late blocks left-right only), gated on zero-shot cost |
| B02 | DINOv2-S/14 reg, healed (small-model scope) |
| B03 | Extra seeds: frontier points, IO-SVD on DeiT, DINOv2-L to 3 seeds |
| B04 | Downstream tasks for DeiT-III-B and DINOv2-L |
| B05 | Extended healing: extra seeds, and the original distilled under the extended budget |
| B06 | Fused kernel for attention, if A12b covers MLPs only |

---

## 6. Pre-registration thresholds (copy into PREREG_FINAL_PHASE.md)

Dawar confirms these by approving the pre-registration commit. After that they never change.

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

---

## 7. Phase C: coding tasks (laptop; CPU only; tiny random ViTs for tests)

Extend the existing `mirror/` library. Keep the current code paths working, because Gate P reproduces MUST-09 with them.

1. `mirror/fit_weightspace.py`: the A*_w fit (A01), with alternation and the stopping rule.
2. `mirror/stability.py`: operator distance, principal angles (energy-weighted), twin agreement, head-map agreement, B_h and C_h distances (A02).
3. `mirror/controls.py`: C1 shuffled correspondence, C2 no re-alignment, C3 within-patch involutive pixel permutation, C4 random-operator nulls (A03).
4. **Baselines:**
   - `baselines/svdllm_full.py`: whitening plus sequential update;
   - `baselines/iosvd.py`: port and adapt;
   - `baselines/dense_students.py`: FLAP-style width pruning to a target count, and greedy block dropping;
   - `baselines/mirror_plus_svd.py`: SVD inside the even and odd blocks, with rank allocation.

   Every baseline reports its exact parameter count and its gap to the target.
5. `heal/multistudent.py`:
   - one process, one data stream, one teacher forward per batch;
   - N students updated in lockstep, each with its own optimiser, learning rate and structure-preserving step (re-projection for Mirror, factor updates for low rank, masks for pruning);
   - GPU-resident uint8 data with GPU random-resized-crop and flip;
   - bf16 autocast, torch.compile, fused AdamW;
   - checkpoints at steps 0, 100, 300 and 780 (and the extended budget), with asynchronous evaluation hooks;
   - a rotation-invariance target option for A15b.
6. **Evaluation:**
   - `eval/knn_fast.py`: GPU-resident bank and validation set, with a full-bank option of 1.28M;
   - `eval/linear_protocol.py`: DINOv2 linear evaluation on cached features;
   - `eval/flipavg.py`: two-pass flip averaging for any checkpoint.

   Reuse the existing downstream code.
7. Model adapters for DeiT-III-B (with the even-projected head and top-1 evaluation), DINOv2-L/14 reg and SigLIP 2 B/16 (A15a). All must pass the exactness test on tiny random versions.
8. `audit/exactness.py`: per-component error at float64, float32 and bfloat16, for both forms, plus the rounding floor and the bitwise check (A09).
9. `systems/bench.py`: throughput, latency, memory and a torch-profiler summary (A12).
10. `stats/hbootstrap.py`: the hierarchical paired bootstrap (A13).
11. `run/runner.py` with a `jobs/final_phase.yaml` manifest:
    - jobs with dependencies, estimated GPU memory and priority;
    - keeps the GPU at ≥ 85% utilisation with ≥ 10 GB headroom;
    - resumable after a crash;
    - logs nvidia-smi every 30 s;
    - writes `results/STATUS.md`;
    - includes the budget governor (Section 8) and the cut order (Section 9).
12. `data/cache.py`:
    - decode ImageNet subsets once (nvjpeg on the GPU, or multi-process CPU) into uint8 memmaps on the VM's local disk;
    - fit sets 0 to 3, the COCO fit set, the heal set (100k), the extended heal set, the LR-selection (5k) and method-selection (10k) sets, the k-NN bank (100k), the full train set for full-bank k-NN and the linear protocol, and validation (50k);
    - splits disjoint by image ID, with the IDs logged.
13. **Figure scripts (A14):**
    - regenerate the teaser tiles from the model (runs on the VM);
    - average the Fourier figure over about 1,000 ImageNet images (CPU-capable);
    - build the frontier, stability and controls figures from `results/`.
14. **CPU tests:**
    - the A*_w closed form against brute-force optimisation on random small matrices;
    - alternation monotonicity;
    - stability metrics on known operators;
    - controls producing nulls on synthetic equivariant networks;
    - matched parameter counts for every baseline;
    - Mirror + SVD exactness;
    - lockstep multi-student training equal to separate single-student runs (same seed, same batches);
    - DeiT, DINOv2-L and SigLIP adapters exact on tiny random models;
    - the audit on tiny models in float64 (≤ 1e-12);
    - runner dependency resolution, resume, governor and cut order;
    - bootstrap against a known analytic case.
15. **Pre-registration:** write `PREREG_FINAL_PHASE.md` from Section 6 and commit it alone. Record its hash in `results/STATUS.md`. All Phase G scripts refuse to run if it has changed.
16. **A12b fused kernel, last:** only after items 1 to 15 pass. Write a CPU reference test of the fused logic; the GPU correctness test runs in W0.

**Stop and report:**
- what is ready, and the test results;
- anything marked (verify);
- estimated GPU memory per student for each model;
- whether A12b is ready;
- the exact commands for Phase G.

---

## 8. Phase G: GPU plan

### Principles

1. **Never leave the GPU idle.** The runner packs light jobs (structure fits, audits, evaluations, profiling) alongside healing.
2. **Keep the data on the GPU.** Decode once, then crop and flip on the GPU. The old heal took 27 minutes, mostly on data loading and evaluation. The pure compute of one DINOv2-B heal is estimated at 3 to 4 minutes (verify on the first run).
3. **Multi-student healing.** One process per (model, seed) trains several methods on the same batches with a single teacher forward. Pack as many students as memory allows, keeping 10 GB of headroom.
4. **Measure, then plan.** Time the first heal of each kind. Write a re-plan to `results/STATUS.md` before launching the next wave.
5. **Checkpoints stay on the VM.** Pull tables, metrics and small artefacts to the laptop with sha256 checks; the laptop disk is full.
6. **Budget governor.** The runner records VM hours from the start of W0. At 13.5 hours it launches no new heals and only finishes running jobs, evaluations and the report. At 15 hours everything stops, and Claude Code tells Dawar the VM can be paused.

### Waves

The runner overlaps waves wherever dependencies allow.

| Wave | Content | Estimated wall clock |
|---|---|---|
| W0 | Environment; GPU check; model downloads (DINOv2-B reg, DINOv2-L reg, DINOv2-S reg, DeiT-III-B, SigLIP 2 B/16); data caches; tests on the VM; GPU smoke test; A12b GPU correctness test if built; pre-registration hash check; timing of one heal per model and of 3 concurrent heals | 30 to 45 min |
| W1 | A01 (A*_w fit, zero-shot, 2 heals) and Gate P (Mirror ×3, SVD-LLM ×3). In parallel: A02 fits and structure, A03 structure controls, A09 audit, A12 systems and A12b timing for DINOv2-B. Apply Gate P and the A01 rule automatically; post a status note | about 1 h |
| W2 | LR selection for the new methods, then the A05 matrix (multi-student per seed), A04 rows, A02 heals and A03 H1 | 1.5 to 2.5 h |
| W3 | A06 frontier, and A07 flip-averaging evaluations as checkpoints land | 1 to 1.5 h |
| W4 | A08 DeiT-III-B (fit, structure, LR check, heals), with A09 and A12 for it; teaser regeneration | 1.5 to 2 h |
| W5 | A10 standard metrics, P11.2, A11 extended healing; then DINOv2-L; then A15a and A15b | 1.5 to 2.5 h |
| End | Write `results/FINAL_PHASE_REPORT.md` (every prediction with PASS, FAIL or PENDING, plus headline numbers and deviations); update `PAPER_NUMBERS.json` through the collector; stop and wait for Dawar | |

### Stop and report

Claude Code stops and reports when any of these happens:
- Gate P fails;
- a job fails twice;
- a memory error persists after concurrency is reduced;
- the budget governor reaches 13.5 hours;
- Module A is complete.

---

## 9. Budget: hard ceiling of 15 VM-hours

The ceiling covers everything from VM start to the final report: setup, healing, evaluation and pulling results.

**Heal-units.** One unit is one DINOv2-B heal (780 steps). A DINOv2-L heal counts as about 3 units, an A11 extended heal as about 3, and a 200-step learning-rate check as about 0.25.

| Item | Heal-units |
|---|---|
| A01 + Gate P (+1 if A*_w is adopted) | 9 |
| A02 | 3 |
| A03 | 1 |
| A04 | 4 |
| A05 | 9, plus about 2.5 for learning-rate selection |
| A06 | 9 |
| A08 DeiT-III-B | 7.5 |
| A08 DINOv2-L | 6 |
| A11 | 6 |
| A15b | 1 |
| **Total** | **about 58** |

Other GPU work comes to about 4.5 hours: setup and caching, structure fits, controls, audit, systems, kernel timing, full-bank k-NN, the linear protocol, downstream evaluation, flip-averaging evaluation, A15a and the teaser. About 3 hours of it runs alongside healing.

**Expected wall clock:**
- With the Section 8 optimisations (data on the GPU, several students per process): about 6 to 9 hours.
- Without them, but with 3 heals at once as on 15 November: about 10 to 12 hours.
- Both fit under 15. The ceiling depends on at least 3 heals running at once, so W0 checks this first.

**How the ceiling is enforced:**
1. After W0, Claude Code projects the total from the measured timings. If the projection is above 13.5 hours, it cuts in this order before launching anything, and logs the cuts:
   1. A15b and A15a (−1 unit and about 0.3 h);
   2. DINOv2-L (−6 units);
   3. A11 extended healing for SVD-LLM, then for Mirror (−3 each);
   4. the extra seeds at 0.50 in A06 (−2);
   5. the second SliceGPT seed and the third IO-SVD seed (−1 each);
   6. the DeiT random-structure control and the third DeiT seeds (−3).
2. The budget governor (Section 8) stops new heals at 13.5 hours and stops everything at 15.

**Never cut:**
- Gate P and the A01 decision;
- A02, A03 and A04;
- the main A05 arms (Mirror, SVD-LLM, full SVD-LLM and IO-SVD, at least 2 seeds each);
- the A06 stacking points at 1 seed;
- DeiT-III-B with 2 seeds per arm;
- A09, A12, A13, and A10 for the main models.

---

## 10. Deliverables

**Per experiment:** `results/<ID>/` containing metrics.json, CSV tables, plots and meta.json. meta.json records:
- commit and pre-registration hash;
- config and seeds;
- model revisions;
- GPU, wall time and peak memory.

**Tables:**
- T1: main healed comparison on DINOv2-B (all methods, seeds, intervals);
- T2: stage ablation;
- T3: other models;
- T4: operator stability;
- T5: discovery controls;
- T6: systems;
- T7: exactness audit;
- T8: standard metrics.

**Figures:**
- F1: frontier, with stacking and flip-averaging points (exactly invariant points marked);
- F2: stability (spectra and twin agreement across fits);
- F3: controls (real versus control twin fractions);
- F4: healing curves for every method;
- the regenerated teaser;
- the averaged Fourier figure;
- the existing structure and CUB figures.

**Reports:** `results/FINAL_PHASE_REPORT.md`, and an updated `PAPER_NUMBERS.json` (no derived numbers outside the collector).

---

## 11. Standing rules

- Dawar starts and destroys GPU instances. Claude Code never creates, starts, resumes or destroys one.
- Downloads go under `$HF_HOME` on the VM, never on the laptop.
- Use a fresh Hugging Face token on the VM. Never commit tokens, keys or weights. Delete the token file at the end if others use the machine.
- Check nvidia-smi before each job. Never kill processes this project did not start.
- No invented numbers. Model IDs, recipes, baseline code and citations are checked or marked (verify).
- Validation images are never used for any choice. All splits are disjoint and logged by image ID.
- Writing: British spelling, plain language, no em dashes, no emoji.
- Before submission: make the repository private or anonymise it (check CPAL's double-blind rules; verify).

---

## 12. Timeline

| Date | Work |
|---|---|
| 19 to 21 Nov | Phase C coding and tests; pre-registration commit; A12b last |
| 21 or 22 Nov | Phase G, W0 to W2 (aim: main table, controls and stability before 23 Nov) |
| 23 Nov | Abstract registration from measured claims only |
| 23 to 25 Nov | W3 to W5 |
| 26 Nov | Module A report |
| 26 to 28 Nov | Buffer: figures, tables, and any rerun that fits inside the ceiling |
| 28 Nov | Results freeze |
| 28 Nov to 4 Dec | Writing, figures, adversarial review, anonymity check |
| 5 Dec | Submit |

---

## 13. Kickoff prompts for Claude Code

### Prompt 1: Phase C (coding)

```
Read FINAL_PHASE_PLAN.md in full. It is self-contained. Then read CLAUDE.md,
notes/paper/method_spec.md, notes/paper/numbers.md and the mirror/ library.

We are in Phase C: coding only. Do not use a GPU and do not download any model or dataset on this
laptop. Build every item in Section 7, in order, with the CPU tests listed there, using tiny random
ViTs. Build A12b (the fused kernel) last, and only after everything else passes. Keep the existing
code paths working, because Gate P reproduces MUST-09 with them.

Write PREREG_FINAL_PHASE.md from Section 6 and commit it on its own. Put its hash in
results/STATUS.md. Write jobs/final_phase.yaml for waves W0 to W5, with the budget governor and the
cut order from Sections 8 and 9.

Use a task list. When everything passes, stop and report:
- what is ready, and the test results;
- anything marked (verify);
- estimated GPU memory per student for each model;
- whether A12b is ready;
- the exact commands for Phase G.
```

### Prompt 2: Phase G (GPU)

```
Dawar has started an H200 VM: <instance details>. HF_HOME is <path>. Follow FINAL_PHASE_PLAN.md
Section 8. First verify that the PREREG_FINAL_PHASE.md commit exists and is unchanged; if not, stop.

Run W0, time one heal per model and 3 concurrent heals, and write a re-plan to results/STATUS.md.
Module A has a hard ceiling of 15 VM-hours from the start of W0. If the projection exceeds 13.5
hours, apply the Section 9 cut order before launching anything. Then run W1 to W5 through the runner
with the maximum safe concurrency, with the budget governor on.

Apply Gate P and the A01 rule automatically as pre-registered, and log every decision in
notes/decisions.md. Keep STATUS.md current. Pull tables and metrics, not checkpoints, to the laptop
with sha256 checks. Stop at any stop event in Section 8.

When Module A is complete, write results/FINAL_PHASE_REPORT.md and update PAPER_NUMBERS.json through
the collector, then stop and wait.
```
