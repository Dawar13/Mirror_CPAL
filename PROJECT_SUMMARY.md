# Mirror Surgery for CPAL 2027: full project summary

Status on 18 November 2026. Every number below is copied from a result file in this repository. The main sources
are `FINAL_RESULTS.md` and `PAPER_NUMBERS.json` (written by `scripts/collect_next_phase.py`), `notes/paper/numbers.md`
(the sourced numbers sheet for the paper) and `results/STATUS.md`. Each section names its source files. Anything
not yet checked is marked (verify).

---

## 1. Where things stand

- **All GPU work for the paper is finished.** The last run was on a Jarvis Labs H200 on 15 November. The instance
  was deleted that night on your order, after every result table had been pulled and checked by sha256. No GPU
  instance is running and none is billing.
- **Pre-registered scorecard for the final phase:** 40 predictions. 17 passed, 16 failed and 7 are pending. Two
  further "upside" items were reported without a pass/fail bar.
- **Main result:** after healing, Mirror Surgery matches the strongest compressor we tested (SVD-LLM) at the same
  size: 78.34 against 78.06 ImageNet k-NN, three seeds each. It is also exactly mirror-invariant in float32, and
  it matches or beats SVD-LLM on every downstream task.
- **Main weaknesses:**
  - before healing, SVD-LLM beats Mirror Surgery at every size;
  - only one model (DINOv2-B/14 with registers) has been healed;
  - the smaller even/odd form runs slower than the original;
  - the "natural-image symmetry" explanation of early pairing is post hoc.
- **The paper:** draft 3 is written, titled "Learned Twice, Stored Once: Mirror Surgery for Vision Transformers".
  - CPAL 2027 template, anonymous;
  - main text exactly 9 pages (the strict limit), 20 pages in all;
  - PDF at `paper/Learned_Twice_Stored_Once_draft3.pdf`.
- **Ratings so far:**
  - an independent review agent scored draft v1 at 2.5 of 5 as written, and about 3.5 once the writing was fixed;
    its fixes went into v2;
  - my own estimate was 3 of 5, or about 3.5 for CPAL in particular;
  - reaching 4 needs a little more evidence (Section 11).
- **Deadlines:** abstract registration on 23 November, paper on 5 December 2026 (AoE). The fallback is the Recent
  Spotlight Track on 18 January 2027.
- **Waiting on you:** whether to run the roughly 5 GPU-hour "reach 4" plan (Section 11). It needs a pre-registration
  addendum committed first, and you starting a GPU.

---

## 2. The idea in plain words

Vision transformers are usually trained on photos that are randomly mirrored left to right. To recognise a
left-facing and a right-facing object, a model grows two detectors, one the mirror image of the other ("mirror
twins"). It stores both in full, so part of the model is a duplicate. We call that duplicate the **symmetry tax**.

**Mirror Surgery** removes the duplicate from an already-trained model, without retraining from scratch. It has
four steps:
1. Fit a **mirror operator**: an exact description of how the model's internal features change when the input
   image is mirrored.
2. Find the **twin neurons**: pairs where one neuron responds to the mirrored image the way its twin responds to the
   original.
3. **Project the weights** onto the nearest exactly mirror-symmetric weights, so each twin pair becomes an exact
   mirror pair that can be stored and computed once.
4. **Heal** with a short distillation from the original model (780 steps), because one operator for the whole
   network costs accuracy.

The result is a model that gives exactly the same prediction for an image and its mirror image (in float32), with
23.5% fewer parameters in its even/odd form.

---

## 3. Timeline

| Date (2026) | Phase | Hardware | Outcome |
|---|---|---|---|
| 7 Nov | **E1**: one mirror operator shared by all 12 layers (plain DINOv2-B) | Friend's shared A100 80 GB (GCP) | **KILL**. The shared operator had only 8 of 768 odd directions. Surgery on 6 blocks gave 14.10 k-NN against 78.21 for the original; same-size SVD gave 72.45. The flaw was in the design: the fit pooled layers with very different energy. |
| 7 to 8 Nov | **E1b**: one operator per layer, pre-registered (commit 93bd945), fresh images | A100 | **GO** on the primary arm (plain DINOv2-B). The registers model and the operator shared over blocks 1 to 6 came out AMBER. |
| 8 to 9 Nov | **E2** paper experiments (PREREG_E2.md): structure, artefact tokens, symmetry against low rank, healing pilot, other models, weights-only readout | A100, then RTX 5090 (vast.ai, Estonia, then South Korea) | Mixed. Gate G2 failed on a bar that no student could meet (Section 6). |
| 9 Nov | **E3** (PREREG_E3.md, signed off): matched healed baselines at full conversion | South Korea 5090 | Partly run. The instance stopped answering that night. E3 was later replaced by MUST-09. |
| 14 Nov | **Next phase** adopted (PROJECT_NEXT_PHASE.md). PREREG_NEXT_PHASE.md signed off (commit e76fb17). | Laptop; H200 574587 set up, then deleted by you | Gate A met. |
| 15 Nov | **H200 run**: every MUST, CORE and UP experiment | Jarvis Labs H200 141 GB, instance 574820 | Jobs ran from 06:56 to 20:18 UTC. Instance deleted at about 20:30 UTC, about 14.4 hours after boot (check the bill). |
| 16 Nov | Pulled tables only (laptop storage full); scored everything | Laptop | 17 pass, 16 fail, 7 pending. |
| 17 Nov | **Paper**: v1, independent review, v2, drafts 2 and 3 | Laptop | Draft 3 committed (a50dbe4). Your commit e303491 ("drafts added") is pushed to origin/main. |

---

## 4. Method and the maths in the paper

Main model: DINOv2 ViT-B/14 with registers (`dinov2_vitb14_reg`, 86,583,552 parameters).

- **Mirror operator (Lemma 1).** Take the token features x of an image and x̃ of its mirror image at mirrored
  positions, and let C = Σ xᵀx̃. Write the symmetrised matrix ½(C + Cᵀ) = V Λ Vᵀ. Then A = V sign(Λ) Vᵀ is the exact
  least-squares best fit among symmetric orthogonal matrices (A² = I).
  - Directions with eigenvalue +1 are **even** (unchanged by a mirror).
  - Directions with eigenvalue −1 are **odd** (they change sign).
- **One operator for the stream.** In a residual network, exact end-to-end equivariance needs one shared operator A*
  for the residual stream. For the full model, A* has d₊ = 700 even and d₋ = 68 odd directions.
  - **Mirror-exact** (called V-exact earlier) uses A* and is exact.
  - **Mirror-local** (called V-local earlier) uses one operator per layer. It is more accurate before healing but
    only approximately equivariant.
- **Projection (Proposition 1).** Let P be the permutation that swaps each twin pair. The update
  W ← ½(W + P W A*) is the orthogonal projection onto the equivariant weights, that is, the nearest equivariant
  weights.
- **Count and tax (Proposition 2).** An equivariant h × d map has h₊d₊ + h₋d₋ = ½(hd + tr P · tr A*) free
  parameters. The redundant share, the "tax", is therefore ½ − tr P tr A / (2hd). This formula matched the
  implemented saving in every one of 144 checks over seven models (MUST-01, difference 0 percentage points;
  two of the models use synthetic structures with the recorded counts).
- **Exactness (Proposition 3).** After projection, x̃ₗ = xₗ A* at every layer. The class token passed through the even
  projection ½(I + A*) is therefore exactly invariant.
- **Power spectrum (Lemma 2).** For any real image, the magnitude of the left-right mirrored spectrum equals that of
  the up-down mirrored spectrum. This is the post hoc reason why early layers pair for up-down flips as much as for
  left-right ones.
- **Attention.** Under a mirror, attention heads map to themselves or swap in pairs. Of 144 heads, 130 map to
  themselves and 7 pairs swap.
- **Healing recipe** (identical for every method; `runs/MUST-09/.../meta.json`):
  - data: 100,000 ImageNet train images, 2 epochs, 780 steps of 256 images;
  - optimiser: AdamW (weight decay 0.05), 5% warm-up then cosine decay;
  - augmentation: crops at scale 0.35 to 1.0, flips with p = 0.5;
  - loss: normalised MSE on the CLS token, the patch tokens and every block's output, with flip-symmetrised targets;
  - learning rate: chosen from {1e-5, 5e-5, 1e-4} by held-out loss on 5,000 other train images. Every heal chose
    1e-4, the top of the grid.
- **Evaluation protocol:**
  - k-NN: ImageNet validation top-1, bank of 100,000 train images (100 per class), 50,000 queries, cosine distance,
    k = 20, T = 0.07, features in bf16;
  - flip consistency: measured in fp32;
  - data hygiene: operators are fitted on train images, evaluation uses validation, healing uses train only.

---

## 5. Results

### 5.1 The symmetry tax (structure)

Sources: `results/MUST-00/audit.md`, `results/MUST-04`, `results/CORE-04`, `notes/paper/numbers.md` T5.

**Main model (fit seed 0 unless stated)**

| Quantity | Value |
|---|---|
| Block 1: neurons in twin pairs | 1,168 pairs and 736 self-twins of 3,072 neurons (76% in pairs; the same within 0.2 points on 3 fit seeds) |
| Twin share, blocks 1 to 6 / 7 to 12 | 63% / 13% |
| Odd directions of the stream at layers 1, 6, 12 | 279, 128, 8 (of 768) |
| Tax (share of MLP weights that are mirror copies), blocks 1 to 6 / 7 to 12 | 0.42 / 0.11 |
| Breaking share fc1, blocks 1 to 6 (registers model) | 0.223, against 0.427 for matched random structure |
| Operator fit, mean over layers 1 to 6: e(null)/e(A) | 13.06 (layer 1 alone: 50.7) |
| Operator fit: e(A)/e(I) | 0.17 |
| Neighbouring layers' operators agree | 0.80 (chance 0.49) |
| Attention heads mapping to themselves / twin-head pairs | 130 of 144 / 7 |

**Artefact tokens (E2, P1 passed).** In plain DINOv2, the late-layer mirror error comes from high-norm artefact
tokens. Masking the top 1% of tokens by norm cuts that error 2.27×. In the registers model, late-layer error is only
1.30× the early-layer error.

**Four symmetries (MUST-04, four models).** Mean "ratio to null" in blocks 1 to 6, where 1 is chance and lower means
stronger pairing:

| Left-right | Up-down | 180° rotation | 90° rotation |
|---|---|---|---|
| 0.47 | 0.51 | 0.52 | 0.82 |

This was the big surprise:
- up-down structure is nearly as strong as left-right, even though no model was trained with up-down flips;
- the up-down twin share in blocks 1 to 6 is 0.46 to 0.56, far outside its null interval (upper bound at most
  0.0035);
- zero-shot up-down surgery on blocks 1 to 6 costs only 0.23 to 6.08 points.

The pre-registered predictions (MUST-04 P4.1 to P4.3) said the opposite, so all three failed.

**14 models (CORE-04).** LR tax in the first half of the blocks (`results/CORE-04`, `results/MUST-04`):

| Model | Flip-trained? | LR ratio to null | LR twin share | LR tax |
|---|---|---|---|---|
| DINOv2-S/14 reg | yes | 0.45 | 0.48 | 0.400 |
| DINOv2-B/14 reg | yes | 0.48 | 0.57 | 0.419 |
| DINOv2-B/14 | yes | 0.45 | 0.54 | 0.405 |
| DINOv2-L/14 reg | yes | 0.51 | 0.50 | 0.414 |
| DINOv2-g/14 reg (SwiGLU) | yes | 0.77 | 0.06 | 0.344 |
| DINOv3-B/16 | probably not | 0.45 | 0.53 | 0.400 |
| DINO v1 B/16 | yes | 0.42 | 0.28 | 0.250 |
| DeiT-III B/16 | yes | 0.45 | 0.43 | 0.349 |
| MAE B/16 | yes | 0.50 | 0.60 | 0.448 |
| CLIP B/16 | no (verified) | 0.51 | 0.33 | 0.313 |
| SigLIP 2 B/16 | unknown | 0.56 | 0.24 | 0.231 |
| EVA-02 B/14 (SwiGLU) | yes | 0.62 | 0.05 | 0.165 |
| ConvNeXt-B | yes | 0.50 | 0.23 | 0.217 |
| ResNet-50 | yes | 0.44 | 0.31 | 0.279 |

What the 14 models show:
- **Depth.** The structure is concentrated in the early blocks. Odd directions collapse with depth: 11 of the 12
  transformers end with at most 13, while MAE keeps 191.
- **Training without flips.** Two models trained without flips have early structure as strong as the rest: CLIP
  (ratio 0.51) and DINOv3 (0.45). This is why the paper says early pairing reflects image statistics more than
  augmentation (post hoc).
- **SwiGLU models** (EVA-02, DINOv2-g) have few twins.
- **CNNs.** Twin structure also appears in ConvNeXt (CORE-04 P4.3 passed).
- **Scale.** The tax does not rise with model size: S 0.400, B 0.419, L 0.414, g 0.344 (P4.1 failed).

**How twins behave (MUST-05, MUST-06).**
- On the same image, twins are almost uncorrelated: median correlation 0.05. On mirrored images it is 0.55.
- So twins are not redundant on any single image. They cover the two orientations.
- A twin is in its partner's top-10 same-image correlates for only 8% of pairs.
- Twin norms match within 0.8 to 1.25 for 92% of pairs (passed).
- The "switch-off" test failed: median cosine 0.49 for twin pairs against a bar of 0.7 (random pairs: about 0).

### 5.2 Before healing (zero-shot), main model

Source: `results/MUST-07/rows_dinov2_vitb14_reg.csv`, `bootstrap.csv`. Validation k-NN at matched parameters, mean of
3 fit seeds where available. The original scores 78.94 in this runner.

| Method | Blocks 1 to 6, MLP | Blocks 1 to 6, MLP + attention | 9 blocks | Full model |
|---|---|---|---|---|
| **Mirror-exact** | 76.60 | 77.34 | 73.37 | 53.81 |
| SVD-LLM style (whitened SVD) | **78.35** | **78.02** | **77.35** | **75.80** |
| ASVD style | 76.81 | 74.41 | 65.24 | 57.09 |
| SliceGPT style | 31.41 | 75.60 | 72.48 | 68.33 |
| FLAP style | 75.20 | 70.29 | 63.47 | 65.12 |
| ZipIt style (merging) | 75.67 | 18.51 | 13.38 | 13.58 |
| Plain SVD | 9.37 | 5.87 | 5.67 | 5.17 |
| Random pairing | 22.40 | 32.30 | 11.07 | 10.02 |
| Random structure | 2.31 | 2.29 | 4.31 | 4.78 |
| Parameters kept (fraction of model) | 0.870 | 0.806 | 0.766 | 0.765 |

What the table shows:
- **Plain SVD collapses on DINOv2.** This was the original "symmetry beats low rank" story. In the earlier phase,
  Mirror-local on the 6-block MLPs scored 77.09 (3 fit seeds, ± 0.003) against 8.77 for plain SVD and 3.95 for
  random structure.
- **Whitened SVD (SVD-LLM) does not collapse,** and it beats Mirror Surgery before healing at every size. This is the
  most important negative result. MUST-07 P7.1 failed: the smallest margin is −1.75.
- **Gate B** (best baseline, chosen on the separate method-selection set): SVD-LLM first, SliceGPT second.
- **Ablations (earlier phase, 6 blocks).** The right structure matters a great deal:
  - wrong (random) operator: 3.88;
  - random pairing: 28.96;
  - pairing by weight cosine: 24.69;
  - fitting on only 100 images still gives 77.03.
- **MUST-08 (passed).** MLP-only conversion gives about two thirds of the saving of MLP plus attention (0.67 at
  6 blocks, 0.68 for the full model).

### 5.3 After healing: the main result (MUST-09)

Source: `runs/MUST-09/dinov2_vitb14_reg/*/s*/metrics.json`, `results/MUST-09/summary.csv`. 780 steps for every method,
same recipe, same size (0.765 of the parameters, 0.774 of the FLOPs).

| Method | Seeds | k-NN (bf16) | Flip consistency (fp32) | Before healing |
|---|---|---|---|---|
| Original | n/a | 78.95 | 96.30 | n/a |
| **Mirror-exact** | 3 | **78.34 ± 0.02** (78.32, 78.35, 78.35) | **100.000%** on every seed | 53.84 |
| SVD-LLM (best baseline) | 3 | 78.06 ± 0.01 (78.05, 78.07, 78.06) | 95.7 to 95.8% | 75.80 |
| SliceGPT (second best) | 1 | 77.89 | 95.43% | 68.29 |
| Plain SVD | 1 | 77.02 | 94.30% | 5.20 |
| Random structure | 1 | 69.93 | 100.000% | 4.78 |

What the table shows:
- **Mirror-exact minus SVD-LLM:** +0.28 points.
  - Paired bootstrap 95% interval +0.09 to +0.47 (computed afterwards on the laptop and labelled post hoc;
    `results/paper_stats/healed_bootstrap.json`, `scripts/paper_bootstrap_healed.py`).
  - The pre-registered test was non-inferiority (P9.1, bar −0.5), and it passed.
- **Random structure trails by 8.41 points** (P9.3 passed), so the fitted structure matters and the healing alone
  does not explain the result.
- **Only the exactly equivariant models reach 100% flip consistency** (P9.2 passed).
- **Gate D was met.**
- **Precision caveat:** in bf16 the healed model agrees with itself on 99.73% of images, not 100%.
- **Seed caveat:** all three healing seeds share one fitted structure (fit seed 0).

### 5.4 Invariance, cost and speed (MUST-10, UP-06, latency)

Sources: `results/MUST-10`, `results/MUST-09/latency.csv`, `results/UP-06`.

- **Flip averaging.** Running the original on the image and its mirror and averaging the features scores 79.00,
  but needs two passes.
  - Healed Mirror Surgery is 0.66 below it (mean of 3 seeds; seed 0: −0.68, 95% interval −0.87 to −0.51).
  - It needs 0.387 of the FLOPs (analytic). P10.1 passed.
- **Two forms of the healed model:**
  - **Dense stored** (projected weights kept as full matrices): runs at the original's speed (about 1.00×) and has
    exact invariance, but saves almost no parameters (0.997).
  - **Even/odd form** (twins computed once): 0.765 of the parameters and 0.774 of the FLOPs, but it runs at
    0.72 to 0.76 of the original's throughput (batch 64 and 256, H200, bf16, compiled). With plain PyTorch kernels
    the smaller matrices and extra memory moves lose the arithmetic saving. **No speed claim is made.**
  - Flip averaging runs at about 0.50 to 0.52 of the original's throughput.
- **Earlier profile (RTX 5090).** The Mirror forms do 8.8 to 23.1% fewer GEMM multiply-adds, but their GEMMs run at
  104 to 161 TFLOP/s against 206 for dense. A fused kernel would be needed for a real speed-up.
- **Quantisation and token merging (UP-06).**
  - INT8 weights cost Mirror Surgery 0.006 points (original 0.004).
  - Token merging (ToMe) costs it 0.18 points (original 0.34).
  - Neither keeps exact invariance: flip consistency falls to 99.1% and 96.8%.

### 5.5 Downstream tasks (MUST-11, CORE-07)

Sources: `results/MUST-11/t2/*.csv`, `results/MUST-11/t2_seg/t2_ade20k.csv`, `results/CORE-07/shift_rows.csv`.

| Task | Original | Mirror-exact | SVD-LLM (seed 0) |
|---|---|---|---|
| ImageNet linear probe | 81.00 | 80.11 ± 0.07 (3 seeds) | 79.72 |
| CIFAR-100 (seed 0) | 91.42 | 90.18 | 89.40 |
| DTD (seed 0) | 83.51 | 83.35 | 82.07 |
| Oxford Pets (seed 0) | 95.94 | 95.69 | 95.53 |
| Flowers-102 (seed 0, at ceiling) | 99.69 | 99.69 | 99.64 |
| ADE20K mIoU (seed 0, linear head) | 47.59 | 47.16 | 46.79 |
| ImageNet-V2 drop (lower is better) | 8.62 | 9.17 (3 seeds) | 9.42 |
| ImageNet-Sketch, near-duplicates removed | 58.61 | 55.03 (3 seeds) | 50.43 |

What the table shows:
- **P11.1 passed:** Mirror Surgery is within 1 point of the original on the linear probe (−0.89) and on ADE20K
  (−0.43).
- **Gaps within noise:** the Pets and Flowers gaps to SVD-LLM. Transfer and ADE20K are single runs.
- **ImageNet-V2 (CORE-07 P7.1 passed):** the drop is within the pre-registered 1-point margin of the original's.
  The paired interval of the difference (+0.08 to +1.02) shows the drop is nevertheless reliably a little larger.
- **ImageNet-Sketch:** it contains flipped and rotated near-duplicates, which an invariant model scores alike, so
  the deduplicated numbers are the ones to cite.
- **Segmentation mask equivariance:**
  - with a head averaged over the image and its mirror, Mirror Surgery gives the mirrored mask for all but 422 of
    487.7 million validation pixels (99.99991%);
  - its raw head scores 98.98%, and the original model 95.03%;
  - P11.2 stays **pending**, because the top-two margins of those 422 pixels were not recorded.

### 5.6 What the even and odd parts encode (CORE-06, all passed)

Source: `results/CORE-06/probes.csv`. CUB-200-2011 birds, linear probes on the class token, mean of 3 healed seeds.

| Feature | Species | Facing direction (natural test set) | Facing direction (test set with every image and its mirror) |
|---|---|---|---|
| Even part, healed | 89.47 | 50.57 | **50.00** (exactly chance, as it must be) |
| Odd part, healed (68 of 768 directions) | 3.71 | **99.21** | 99.21 |
| Full feature, healed | 89.42 | 95.36 | 95.31 |
| Original, even part (last-layer operator) | 89.96 | 86.70 | 86.31 |
| Original, odd part | 25.01 | 94.99 | 95.31 |

The healed model cleanly separates *what* the object is (even part) from *which way it faces* (odd part). The
original model mixes the two.

### 5.7 Other models before healing (earlier phase and CORE-04)

- **DeiT-III B/16:** AMBER. Mirror 81.55 against plain SVD 80.67 at 6 blocks.
- **DINOv2-L/14 reg:** AMBER. It has the strongest operator fit of all models (79.4), but plain SVD also survives
  there (79.26 against 78.57).
- **DINOv2-S/14 reg:** PIVOT. Surgery gives 56.03 against 74.46 for the original.
- **MAE:** AMBER. It has the strongest structure by several measures, but a weak original k-NN (13.73).
- **SigLIP 2:** KILL. Its structure is weak, as predicted, but the pre-registered sub-check (its breaking share above
  DINOv2-reg's) failed: 0.137 against 0.223.
- **Mirror plus low rank (CORE-03).** Stacking low rank inside the even and odd blocks never beat low rank alone
  (0 of 4 budget points on each of 3 models; P3.1 failed).

---

## 6. Pre-registration record

### 6.1 Earlier phases

| Check | Value | Bar | Verdict |
|---|---|---|---|
| E1 (shared operator, 12 layers) | rows 1, 4, 5 in the KILL cell | see E1_INSTRUCTIONS.md | **KILL** |
| E1b primary (per-layer, plain DINOv2) | fit 10.57, e(A)/e(I) 0.160, d₋/d 0.260, breaking 0.194 (null 0.416), 6-block drop 3.62 and margin over SVD +9.31 | five GO rows | **GO** |
| G1: SVD collapse is not a bug | four checks clean | | PASS |
| P1: artefact tokens (plain / registers) | 2.27× / 1.30× | ≥ 2× / ≤ 2× | PASS, gate G3 PASS |
| P2: energy rule predicts the per-block winner | 16 of 36 (0.44) | ≥ 0.75 | FAIL |
| P3.a: healed Mirror minus healed random (9 blocks) | +5.54 | ≥ 2.0 | PASS |
| P3.b: healed Mirror minus healed plain SVD | +1.24 | ≥ 3.0 | FAIL (passing would have needed 80.57, above the original's 78.94) |
| P3.c: original minus healed Mirror | 0.12 | ≤ 2.0 | PASS |
| Gate G2 (P3.a and P3.b) | | both | FAIL; E3 and then MUST-09 followed |
| P4: DeiT-III and MAE structure | pass | | PASS |
| P4: SigLIP 2 breaking share above DINOv2-reg's | 0.137 against 0.223 | higher | FAIL |
| P5 part 1: layer-0 operator from weights alone | 0.90 | ≥ 0.8 | PASS |
| P5 part 2: twin pairing from weights alone | 46.6% | ≥ 70% | FAIL (gate G4: data-free claim dropped) |

### 6.2 Final phase (PREREG_NEXT_PHASE.md, sign-off commit e76fb17)

**Gates:**

| Gate | Status | Detail |
|---|---|---|
| A | met | sign-off committed; MUST-00 audit accepted (38 match, 8 need context, 2 mismatch) |
| B | met | best baseline SVD-LLM, second SliceGPT |
| C | **not met** | LR-trained small ViT: twin share 0.060 (bar 0.2) at 64 px, and 0.065 at 96 px |
| D | met | Mirror-exact 78.34 against SVD-LLM 78.06 (bar: at least −1.0) |
| E | pending | condition included "CORE-01 running", which was dropped after gate C |

**Passed (17):**
- MUST-01: tax formula exact;
- MUST-03: measurement code reproduces the earlier runs;
- MUST-05 P5.2: twin norms match;
- MUST-06 P6.1 and P6.2: twins uncorrelated on the same image;
- MUST-08 P8.1 and P8.2;
- MUST-09 P9.1, P9.2 and P9.3: the main result;
- MUST-10 P10.1: flip averaging;
- MUST-11 P11.1: probe and ADE20K;
- CORE-04 P4.3: ConvNeXt twins;
- CORE-06 P6.1, P6.2 and P6.3: CUB;
- CORE-07 P7.1: ImageNet-V2.

**Failed (16):**
- **MUST-02:** fp32 component error 2.51e-5, bar 1e-5. It is 5.8e-14 in float64, and flip consistency is 100%.
- **MUST-04 P4.1, P4.2 and P4.3:** up-down structure is as strong as left-right.
- **MUST-05 P5.1:** switch-off cosine 0.49, bar 0.7.
- **MUST-06 P6.3:** effective rank changed by 8.6%, bar 2%.
- **MUST-07 P7.1 and P7.2:** zero-shot Mirror does not beat every baseline by 2 points. It loses to SVD-LLM, and
  ZipIt trails by only 0.93 points at 6 MLP blocks.
- **MUST-12 gate C:** both sizes.
- **CORE-01 T2 and T4:** scored on the pilot runs only.
- **CORE-03 P3.1:** stacking.
- **CORE-04 P4.1:** tax does not rise with scale.
- **CORE-04 P4.2:** CLIP (no flips) has a higher tax than DINO v1 and EVA-02.
- **CORE-05 P5.1:** ratio 0.58, bar 0.5.
- **CORE-05 P5.3:** the frozen rule picks the less harmful method on 32 of 48 held-out blocks (0.667), bar 0.75.
  The rule was committed in 70a47df before scoring.

**Pending (7):**
- **MUST-11 P11.2:** pixel margins not recorded.
- **CORE-01 T1 and T3; CORE-02 P2.1 to P2.3; CORE-05 P5.2:** all depend on the CORE-01 controlled-training grid,
  which was dropped when gate C failed.

**Reported without a bar:**
- UP-06: INT8 and ToMe (Section 5.4).
- UP-07: pairing from weights alone agrees on 60.1% of block-1 pairs, below the 70% guide.
- UP-01 to UP-05 were not run.

The paper reports every one of these, including all 16 failures (Appendix A of the draft).

---

## 7. How the story changed

| Topic | Original thesis (PROJECT_CONTEXT.md v2, 8 November) | What the data now support |
|---|---|---|
| Cause of the twins | Flip augmentation creates them | Early pairing is as strong for up-down as for left-right, and as strong in CLIP and DINOv3 (trained without flips). The paper offers natural-image symmetry as a **post hoc** explanation (Lemma 2). The planned causal test (CORE-01) could not run: small ViTs trained from scratch never formed enough twins (gate C). |
| Symmetry against low rank | Mirror beats low rank where the spectrum is flat | True against plain SVD (which collapses), **false against whitened SVD-LLM** before healing. The rule that was meant to predict the winner failed twice (P2: 0.44; CORE-05 P5.3: 0.667). |
| Method claim | Healed Mirror Surgery beats SVD, SliceGPT and width pruning | It **matches** the best compressor after healing (+0.28, interval excludes 0, post hoc), with exact invariance as a bonus no compressor has. |
| Speed | Cleaner speed-up from the even/odd form | No speed-up with plain PyTorch; the even/odd form runs at about 0.72× the original. |
| Data-free readout | Stretch claim | Dropped (gate G4). |
| Interpretability | Not planned | New and clean: the even part encodes the species, the odd part the facing direction (CUB). |

The paper's framing is now: **pretrained ViTs spend much of their early capacity on mirror twins. Mirror Surgery
turns that duplication into an exactly equivariant model that, after a short distillation, matches the strongest
structured compressor we tested.**

---

## 8. The paper (draft 3)

**Title:** "Learned Twice, Stored Once: Mirror Surgery for Vision Transformers".

**Format:**
- CPAL 2027 template (`cpal_2027.sty`), anonymous submission with line numbers, natbib with unsrtnat (set by the
  style file).
- CPAL limit: **9 pages of main text including figures, strictly enforced** (over-length papers are not reviewed).
  References, acknowledgements and appendices are extra and unlimited.
- Draft 3's main text ends at the bottom of page 9; 20 pages in total.

**Outline:**

| Section | Content |
|---|---|
| Abstract | Hedged: matches SVD-LLM with the bootstrap interval, exact in float32, the two limits stated, pre-registration with all 16 failures |
| 1 Introduction | Figure 1 (method diagram) at the top of page 2, contribution bullets |
| 2 Related work | Learned symmetry; making pretrained models equivariant (credits Bökman et al. 2025 and Nordström et al. 2025); structured compression |
| 3 Where the symmetry tax lives | Twins figure, operator (Lemma 1), twins and tax, closeness to null, 14 models, symmetry-by-depth figure, early structure not specific to LR or to flips, power-spectrum lemma |
| 4 Mirror Surgery | One stream operator, projection (Prop. 1), even/odd coordinates and count (Prop. 2), exactness (Prop. 3), healing |
| 5 Experiments | Setup, baselines, healed table, healing/size figure, invariance and cost, downstream table, before healing, CUB figure |
| 6 Limitations and outlook | What did not work, scope, outlook |
| Appendix A | Pre-registration history and the full prediction table |
| Appendix B | Structure in all models (table, block-by-block figure, full twin grid) |
| Appendix C | Healing curves, before-healing table and figure, other models, components, efficiency |
| Appendix D | Implementation details (data, baselines, downstream protocols) |
| Appendix E | Proofs |

**Figures (8, all from scripts, bright colour-blind-safe Okabe-Ito palette in `scripts/paper_style.py`):**

| # | Figure | Where | Script |
|---|---|---|---|
| 1 | Method overview (TikZ) | main | `scripts/make_paper_fig_method.py` |
| 2 | Twin examples (compact) | main | `scripts/make_paper_fig_twins.py` |
| 3 | Structure across symmetries and depth | main | `scripts/make_paper_fig_symmetry.py` |
| 4 | Healing curves and accuracy against size | main | `scripts/make_paper_fig_results.py` |
| 5 | What even and odd features encode (CUB) | main | `scripts/make_paper_fig_cub.py` |
| 6 | Structure block by block | appendix | `scripts/make_paper_fig_blocks.py` |
| 7 | Full twin grid (12 pairs, blocks 1, 3, 6) | appendix | `scripts/make_paper_fig_twins_full.py` |
| 8 | Accuracy before healing | appendix | `scripts/make_paper_fig_zeroshot.py` |

**Tables:**
- main text: the healed comparison and downstream tasks;
- appendix: pre-registration, structure, healing curves, before healing, other models;
- the appendix tables are generated by `scripts/make_paper_prereg_table.py`, `make_paper_structure_table.py` and
  `make_paper_core04_table.py`.

**References:** `paper/reference.bib` has 69 entries. An agent checked each one against an authoritative source
(arXiv, publisher, PMLR, CVF, OpenReview); four were corrected and none had to be removed. The record is in
`paper/references_verification.md`.

**Writing history:**

| Commit | Draft | Main changes |
|---|---|---|
| 2e4569f | v1 | First full draft, verified bibliography, figures and generated tables |
| b359550 | v2 | Independent review applied: size and speed as a trade-off, exactness in float32 only, post hoc explanation marked, "matches" not "beats", baseline differences disclosed, Bökman 2025 credited; post hoc paired bootstrap |
| bdaaafb | Draft 2 | New title; propositions with proofs and the power-spectrum lemma; method diagram; healing and size figure; before-healing table moved to the appendix |
| a50dbe4 | Draft 3 | Bright palette; method diagram as Figure 1; CUB result as a figure; three appendix figures |
| e303491 | (yours) | "drafts added": PDF, template files; pushed to origin/main |

**Supporting notes:**
- `notes/paper/numbers.md`: every number with its source file, plus 16 cautions about mixed pipelines;
- `notes/paper/method_spec.md`: the method as implemented;
- `notes/paper/review_v1.md`: the independent review.

---

## 9. Compute and infrastructure

| Machine | When | Used for | End |
|---|---|---|---|
| Friend's shared A100 80 GB (GCP) | 7 to 8 Nov | E1, E1b, much of E2 | Offline from 8 Nov 21:40 UTC |
| vast.ai RTX 5090, Estonia (59831199) | 9 Nov | E2 continued | GPU taken by another renter; disk copied to Korea |
| vast.ai RTX 5090, South Korea (59906003) | 9 Nov | E2, start of E3, latency on a second device | Unreachable from the night of 9 to 10 Nov |
| Jarvis Labs H200 (574587) | 14 Nov | Setup only | Deleted by you the same evening |
| Jarvis Labs H200 141 GB (574820) | 15 Nov | The whole final phase | Deleted at about 20:30 UTC on your order |

Notes on the final run:
- **Cost:** about 14.4 hours from boot to deletion. At the 14 November price of about 435 INR per hour that is
  roughly 6,300 INR (verify against the bill).
- **What went wrong, and the fixes:**
  - **Thread oversubscription** until 10:12 UTC. Fixed with a thread cap.
  - **Serial tails** (MUST-07, ADE20K, CORE-04, CORE-05) were split into parallel processes. Each split was logged
    and gave identical rows.
  - **Host-memory overload** at 15:49 UTC, which caused one OOM kill. Fixed by loading images from memory-mapped
    caches (`--preload loader`, identical rows).
  - **CORE-03 bug:** an empty even or odd part crashed the code. Fixed in 805c231 with a regression test.
- Every operational decision is in `notes/decisions.md`.

**Standing rules:**
- you start GPUs; I never create, start or resume instances with the CLI;
- an instance is destroyed only on your order;
- on a GPU used by this project alone, 2 or 3 heals may run at once;
- downloads go under `$HF_HOME` on the VM, never on the laptop.

---

## 10. What is on the laptop

- **Kept in git:** all result tables, summaries, figures, logs, `FINAL_RESULTS.md`, `PAPER_NUMBERS.json`, and the
  paper source and PDF.
- **Kept but not in git:**
  - the 9 healed MUST-09 checkpoints and their k-NN prediction files (`runs/MUST-09`, 2.7 GB);
  - the fitted structures (`results/MUST-07/structure` and others).
  - Deleting these would mean redoing GPU work, so ask first.
- **Not kept** (your choice, because laptop storage is full; all can be regenerated on a GPU): feature files,
  caches, MUST-12 pilot checkpoints, and the k-NN prediction arrays of the other experiments.
- **Git:** the working tree is clean and `main` is level with `origin/main`.

---

## 11. Open items and decisions waiting for you

1. **The "reach 4" plan (not approved yet, nothing started).** About 5 GPU-hours in one session:

   | Experiment | What it fixes | GPU time (estimate) |
   |---|---|---|
   | Heal DeiT-III with Mirror Surgery and SVD-LLM | "only one model was healed" | about 2 h |
   | Heal from two more structure fits (fit seeds 1 and 2) | the error bars ignore variation in the fitted structure | about 1 h |
   | Train small ViTs with 90-degree rotation augmentation, pre-registered | turns the natural-image explanation into a tested claim; most likely route to 4.5 | about 1.5 h |
   | Ratio to null against a structure fitted to shuffled patches | "the measure is an artefact of fitting" | about 0.5 h |
   | Re-run segmentation with pixel margins logged | settles P11.2, the last pending prediction that can be settled | about 10 min |
   | Optional, laptop only: fused even/odd kernel | if it reaches the original's speed, the parameter saving becomes a deployment gain | none |

   Before anything runs:
   - a pre-registration addendum must be committed;
   - the job files must be tested on the laptop;
   - you must start the GPU.

   The bolder option discussed earlier was Klein-group surgery for early layers plus a full mechanism test, about
   one H200 day and 4 to 6 days of coding. It is probably too much before 5 December.
2. **Abstract registration on 23 November.** It needs the final title and abstract. Gate E cannot be met as written
   (CORE-01 was dropped), so it should be closed with a line in `notes/decisions.md`.
3. **Anonymity check.** The repository's `origin/main` now holds the paper PDF and source, and the memory notes say
   the repository is public. Check CPAL's rules on public preprints and code during double-blind review (verify),
   and decide whether to make the repository private or anonymise it until decisions are out.
4. **Values marked "(derived)" in `notes/paper/numbers.md`** were computed outside the collector. Under the
   CLAUDE.md rule, any of them cited in the paper should be added to `scripts/collect_next_phase.py` before
   submission.
5. **Operational definitions to confirm.** Some definitions logged on 15 November are marked "to confirm with Dawar"
   (CORE-02, CORE-03, CORE-05, CORE-06, CORE-07, UP-06 and UP-07 scoring details). They were fixed before the runs;
   they only need your sign-off.
6. **Novelty re-check** on OpenReview (ICLR 2027 submissions) in the final week, as planned in `PROJECT_CONTEXT.md`.
   The closest prior work found so far:
   - Bökman and Kahl 2023;
   - Bökman et al. 2025 ("Flopping for FLOPs");
   - Nordström et al. 2025;
   - Shahverdi et al. ICML 2026;
   - Rahman et al. CVPR 2026.

   The risk was judged medium-low (`docs/novelty_check_2026-11.md`).
7. **`PROJECT_CONTEXT.md` is out of date.** It is still v2 of 8 November, with the original claims C1 to C7. Its
   claims list no longer matches the findings in Section 7; it could be updated or marked historical.

---

## 12. Where to find things

| What | Where |
|---|---|
| Scored predictions, gates, cited numbers | `FINAL_RESULTS.md`, `PAPER_NUMBERS.json` |
| Numbers for the paper with sources and cautions | `notes/paper/numbers.md` |
| Running status | `results/STATUS.md` |
| Decisions and deviations | `notes/decisions.md` |
| Pre-registrations | `E1B_PREREG.md`, `PREREG_E2.md`, `PREREG_E3.md`, `PREREG_NEXT_PHASE.md` |
| Plans | `PROJECT_CONTEXT.md` (v2, dated), `PAPER_EXPERIMENTS.md`, `PROJECT_NEXT_PHASE.md` |
| E1 and E1b reports (unchanged) | `results/E1_REPORT.md`, `results/E1B_SUMMARY.md` |
| Per-experiment results | `results/MUST-*`, `results/CORE-*`, `results/UP-*` |
| Healed checkpoints (git-ignored) | `runs/MUST-09/dinov2_vitb14_reg/` |
| Paper source | `paper/main.tex`, `paper/sections/*.tex`, `paper/figures/` |
| Paper PDF | `paper/Learned_Twice_Stored_Once_draft3.pdf` |
| Figure and table scripts | `scripts/make_paper_*.py`, `scripts/paper_style.py`, `scripts/paper_bootstrap_healed.py` |
| Library code | `mirror/` (operators, twins, surgery, attention, baselines, healing, evaluation) |
| Build the PDF | in `paper/`: pdflatex, bibtex, pdflatex twice |
