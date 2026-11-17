# Review of the CPAL 2027 draft, version 1

Independent read-only review, 17 November 2026. Read: `paper/main.tex`, `paper/sections/*.tex`, `paper/main.pdf` (17 pages) and `paper/main.log`. Checked against: `notes/paper/numbers.md`, `notes/paper/method_spec.md`, `FINAL_RESULTS.md`, `paper/references_verification.md`, plus `results/MUST-09/curves.csv`, `runs/MUST-09/dinov2_vitb14_reg/*/s0/metrics.json` (learning-rate trials, timing), `results/MUST-04/dinov2_vitb14_reg/blocks.csv` (absolute breaking shares), `results/CORE-04/structure/{convnext_b,resnet50}/layers.csv`, `results/MUST-05/dinov2_vitb14_reg/meta.json`, `configs/models/*.yaml` and the five `scripts/make_paper_*.py`. From MUST-12 only the blocks 1 to 6 values in numbers.md T9a were used. Line numbers refer to the `.tex` file named.

**Overall.** Almost every number in the tables traces to a result file and is rounded correctly. I recomputed the four-symmetry means (0.47, 0.51, 0.52, 0.82), every cell of Tables 1, 3, 4, 6 and 7, the step-100 and step-300 seed means (78.02, 78.38; 77.50, 78.10) and the CUB means. The main problems are framing, not arithmetic. The text claims more than the data show on size and speed, on exactness, on "beats SVD-LLM" and on the natural-image explanation. Two statements also contradict each other.

## 0. Ten most important issues

1. Size and speed framing (C2, P2, P3). The 23.5% saving exists only in the even/odd form, which runs at 0.72x the original's speed. "Twice the throughput" uses the dense stored model, which saves nothing. "39% of the cost" is an analytic FLOP count.
2. "Exact" and "never changes" hold only in float32 (P1, C13). In bf16, which is used for every headline accuracy and every speed number, the model disagrees with itself on 0.27% of images after healing and 2% before. MUST-02 failed its 1e-5 bar. Int8 and ToMe break the invariance.
3. The natural-image explanation is post hoc but is stated as a conclusion in the abstract, the intro and a section heading (C7, C8). Its evidence is confounded (CLIP, DINOv3) or underpowered (pilots), and the related prediction CORE-04 P4.2 failed.
4. Contradiction (C1). structure.tex:76 says late blocks are cheap to convert, while method.tex:58-59 and experiments.tex:139-140 say late blocks are hurt most.
5. The word "beats" or "leads" against SVD-LLM (C4, P5). The pre-registered test was non-inferiority, there is no bootstrap, all heals share one fitted structure, and the downstream gaps are single-seed and at noise level.
6. Undisclosed differences in healing and baselines (C5, C6). Block targets differ by method. SVD-LLM is the whitening step only, with no search. SliceGPT and ZipIt are simplified. ASVD, FLAP and ZipIt were never healed. The learning rate sits at the edge of the grid.
7. The even/odd block-diagonal form is not credited to Bökman et al. 2025, whose implementation does give real speed-ups (R1).
8. The ratio to null is lower in late blocks than in early ones for 10 of 12 transformers, which looks like it contradicts "concentrated in the first half". No absolute breaking shares are given, and "1 is chance" uses a random rather than a fitted null (C10).
9. Pre-registration history (C19, C20). The exploratory phase, and the reuse of its fitted structures, is not disclosed. The appendix text disagrees with the table on what is PENDING. UP-05 (the speed bar) can be scored but is left PENDING.
10. Number fixes (N1 to N9): 0.2% should be 0.25%; 53.84 against 53.81 and 78.95 against 78.94 come from different pipelines; the Table 2 drop is 9.42 where 9.41 follows from the table; 0.73 against 0.72; ConvNeXt's "first" count is measured inside the block; the R180 values are not in Appendix B; "144 blocks" includes synthetic and repeated blocks; Table 5 has an escaping bug.

## 1. Numbers

### 1a. Mismatches, wrong or unsupported values (most important first)

- **N1** experiments.tex:14. "in practice within 0.2%" is wrong: SliceGPT at 6 blocks is +0.2409% over budget (numbers.md T4a). Fix: "within 0.25%".
- **N2** Tables 1 and 3 give different "before healing" values for the same seed-0 structure: Mirror Surgery 53.84 against 53.81 ± 0.36 (seed 0 alone is 53.88; MUST-02 in float32 gives 53.96), SliceGPT 68.29 against 68.33, plain SVD 5.20 against 5.17. Table 1 uses step 0 of the healing run (curves.csv) and Table 3 uses the MUST-07 runner. Fix: add to the Table 1 caption "Before: step 0 of the healing run; the compressor runner of Table 3 gives 53.88 for the same structure".
- **N3** The original model is 78.95 in Table 1 (experiments.tex:36) and 78.94 in the Table 3 caption (experiments.tex:110). These are two pipelines (78.954 against 78.938, numbers.md caution 4). Fix: one sentence in Setup saying so.
- **N4** Table 2, SVD-LLM row: the drop is 9.42, but ImageNet 79.72 minus IN-V2 70.31 gives 9.41. The drop uses the CORE-07 probe (79.726) and the ImageNet column uses MUST-11 (79.716). Fix: use 79.73 in the ImageNet column, or footnote the two probe runs.
- **N5** experiments.tex:67 says "0.73 of the original's throughput", but appendix.tex:62-64 prints 5,897 and 8,137, which give 0.72 (MUST-10 run). The 0.73 comes from the MUST-09 run (0.726). Fix: "0.72".
- **N6** structure_table.tex, ConvNeXt "first" = 57. This is space `dw0` (the depthwise-conv output inside block 1), not "the stream at the input of the first block"; ConvNeXt's stem output is not measured. ResNet-50's "first" (15) is out of d = 64, but only the last d is printed. Fix in `scripts/make_paper_structure_table.py`: use `out0` for ConvNeXt or footnote it, and print d for both the first and the last space.
- **N7** structure.tex:69-70 says 180-degree structure fades in all four models "(Appendix B)", but Table 6 shows only LR and UD. The values exist (numbers.md T5a: 0.53 to 0.60, 0.51 to 0.60, 0.57 to 0.66, 0.48 to 0.70). Fix: add R180 columns or give the values inline.
- **N8** Figure 1 caption, intro.tex:9-10, "42% of the MLP weights are redundant". This is tax_1-6 = 0.4185, computed with the per-layer operators, and removing those weights costs 1.84 points before healing (MUST-04 LR surgery). Fix: "tying the twins would remove 42% of the MLP weights of the first six blocks (per-layer operators)". Also say that the rows are the strongest pairs (the teaser is ranked by weight, MUST-05 meta.json).
- **N9** intro.tex:36-37 and method.tex:48-49, "144 measured blocks of seven models". The 144 rows include three fit seeds of the same 12 blocks for two models, and synthetic structures for DINOv2-L and DINOv2-S (36 rows; numbers.md T8). It is an identity check of the implementation, not a measurement. Fix: in Section 4, "for every converted block (144 checks over seven models, two with synthetic structures) the implemented count equals the formula". Drop the claim from the contribution list.

### 1b. Mixed protocols that must be stated

- **P1** Precision. Every headline k-NN number is bf16; flip consistency is float32. In bf16 the healed model agrees with itself on 99.73% of images (128 to 143 disagreements per seed) and the unhealed one on 97.9 to 98.1% (973 disagreements, MUST-02). Speed is also measured in bf16. abstract.tex:12-13 and intro.tex:38 combine a float32 property with a bf16 speed. Fix: in Setup, "accuracies are computed in bf16 and flip consistency in float32, where the model is exact; bf16 rounding breaks the symmetry on 0.27% of validation images after healing (2% before)". Add "in float32" to the abstract.
- **P2** "39% of the cost" (intro.tex:42) is the analytic FLOP ratio of the even/odd form (36.44 against 94.16 GFLOPs, P10.1). "Twice the throughput" (abstract.tex:13, experiments.tex:65) is measured on the dense stored model at 47.08 GFLOPs. The even/odd form measures 5,897 against 4,220 images per second, which is 1.40x flip averaging or 72% of its cost. Fix: "at 39% of its FLOPs (analytic count) and about half its measured cost".
- **P3** Table 1: Params 0.765 and Speed 1.00 sit in one row but describe different objects. The dense stored model holds 86.28M entries (0.997) and runs at 1.00; the even/odd form is 0.765 and runs at 0.72. Also, the SVD-LLM, SVD and random-control speeds come from the MUST-09 benchmark run, while flip averaging (0.52) and Mirror (1.00) come from MUST-10. Fix: split Mirror Surgery into two rows and name both runs in the caption.
- **P4** experiments.tex:63-64 gives 0.66 (mean of three seeds) next to −0.68 with a CI (seed 0). Fix: "0.66 points below (mean of three seeds; seed 0: 0.68, 95% CI 0.51 to 0.87)".
- **P5** Table 2 and experiments.tex:93-95 compare Mirror's three-seed mean (80.11) with SVD-LLM's seed 0 (79.72). Transfer and ADE20K are single runs. The Pets (0.16) and Flowers (0.05) gaps are within noise, and several transfer probes chose a weight decay at the edge of the grid (numbers.md T2). Fix: change "leads" to "matches or exceeds", say "one seed" in the text, and remove the bold on Pets and Flowers.
- **P6** experiments.tex:57 "from the first evaluation at step 100 onwards". Step 0 is also an evaluation, and steps 100 and 300 use 10,000 queries (appendix.tex:29-30). Fix: "from step 100 (first check after step 0, on a 10,000-image subset) onwards".
- **P7** core04_table.tex gives 76.59 and 77.41 for DINOv2-B reg (fit seed 0), while Table 3 gives three-seed means of 76.60 and 77.34. Fix: add "fit seed 0" to the caption.
- **P8** experiments.tex:135 "37 to 49 points" holds only for the six-block columns; at 9 blocks the gap is 56 to 60. Fix: "at six blocks, 37 to 49 points".
- **P9** experiments.tex:100-103 compares Mirror's symmetrised head (99.99991%) with the original's raw head (95.03%; symmetrised 94.74%). Mirror's raw head scores 98.98%. P11.2 is PENDING because 422 pixels differ and their margins were not stored. Fix: "for all but 422 of the 487.7 million pixels of the 2,000 validation images (raw head 98.98%; original 95.03% raw, 94.74% symmetrised)".
- **P10** CORE-07 P7.1 passes on the point estimate (+0.55 against a 1-point margin), but its 95% interval is [0.08, 1.02]: Mirror's IN-V2 drop is reliably larger than the original's. Fix: give the interval in experiments.tex:97-98.

### 1c. Sourcing gaps

- Figure 1's "50,000 ImageNet training images" is correct (`results/MUST-05/dinov2_vitb14_reg/meta.json`, teaser_images) but is not in numbers.md. Add it.
- The odd-direction counts in Table 6 for the ten CORE-04 models come from layers.csv through the script. That is fine, but numbers.md has no row for them. Add one.

## 2. Claims (method and experiments against method_spec.md and code)

- **C1 Contradiction.** structure.tex:76 says "Late blocks are nearly invariant, so the same conversion costs them little". Against it: method.tex:58-59 ("costs accuracy, especially in late blocks"), experiments.tex:139-140 ("late blocks ... are hurt by the projection") and Table 3 (73.37 at 9 blocks, 53.81 full). Fix: "Late blocks are nearly invariant, so they hold little to collect, and forcing on them a single operator fitted mostly to early layers costs accuracy (Section 5)."
- **C2 Size and speed.** Wrong or misleading statements:
  - intro.tex:36 says "stores 23.5% fewer parameters", and abstract.tex:8 and method.tex:5 say "stores each twin pair once". The healed checkpoint stores dense projected weights.
  - related.tex:22 says it "reduces the parameter count rather than increasing the cost". The reduced form is 28% slower than the original.
  - discussion.tex:18-19, "depends on the kernels used", hides that the even/odd form is slower at every batch size (0.54x at batch 1).
  - UP-05 (pre-registered: a speed claim needs more than 1.1x) is listed PENDING although the data exist.

  Fix: in the abstract and Scope, "The 23.5% saving is realised by an even/odd implementation that in our code runs at 0.72x the original's throughput; stored densely, the model runs at full speed but saves nothing." Score UP-05 as not met (0.72x).
- **C3** "closed-form conversion" (abstract.tex:7, intro.tex:22, 35). The conversion needs activations from 2,000 images, a blossom matching per MLP, a dynamic programme to pair heads and polar fits, and then 780 distillation steps before it is competitive. Fix: "a training-free conversion (an eigendecomposition and a matching per layer), followed by a short distillation".
- **C4 Wording of the main result.** abstract.tex:11 says "leads it", intro.tex:40 "matches or beats", discussion.tex:30 "matches". The pre-registered P9.1 is a non-inferiority test (above −0.5). No significance test of the +0.28 exists (numbers.md T1b), and the healed spread excludes the fit variation (fit-seed sd 0.36 before healing). Fix: use "matches" everywhere, with "(+0.28 points; pre-registered as non-inferiority)". Optionally compute the paired bootstrap from the stored `step780_*.npz` files and label it post hoc.
- **C5 "Same recipe" and "fair"** (experiments.tex:18-19, 53-54; method.tex:62-63). Four facts are missing:
  1. Block targets differ: Mirror and the random control get A*-symmetrised teacher outputs, SVD and SVD-LLM raw outputs, SliceGPT centred outputs.
  2. All methods, not only Mirror, get flip-averaged final targets.
  3. Every method chose 1e-4, the top of the learning-rate grid. A 2e-4 check would not have won for Mirror, SVD-LLM, SliceGPT or the random control, but would have for plain SVD (`lr_extra` in each `metrics.json`). This is a useful defence, so state it.
  4. "the other methods once" (experiments.tex:20) implies every method was healed, but ASVD, FLAP and ZipIt were not.

  Fix: one sentence each in Setup, and replace "fair in the sense that matters for the claim" with this list.
- **C6 Baselines are partial reimplementations** (method_spec §9 and §10 item 10):
  - SVD-LLM is the whitening step only, with no parameter update and a single grid point. This contradicts "hyperparameters are chosen" (experiments.tex:15).
  - SliceGPT takes its PCA statistics from the original model and does not recompute them after slicing.
  - ZipIt merges units within one model.
  - FLAP zeroes heads.

  Fix: call them "SVD-LLM (whitening only)", "SliceGPT-style" and "ZipIt-style (within one model)" in tables and text, and list the differences in Appendix D.
- **C7 Post hoc explanation stated as a finding.** The places are abstract.tex:6-7 ("so early pairing follows the symmetries of natural images rather than the training augmentation"), intro.tex:33-34 ("therefore tracks"), the structure.tex:52 heading and intro.tex:54-55 ("does not need flips at all"). Section 3 itself says the explanation is not tested (structure.tex:68-69). The evidence is weak:
  - Only two models were trained without flips, and both are confounded. CLIP differs in data, objective, QuickGELU and pre-norm (configs/models/clip_b16.yaml notes). DINOv3-B's flip setting is inferred from its teacher and ViT-L configs; the ViT-B config is not public.
  - The pre-registered CORE-04 P4.2 failed, and MUST-04 P4.1 to P4.3 predicted the opposite of the UD result.

  Fix: abstract, "... present even in models trained without flips, which suggests that early pairing reflects the symmetries of natural images more than the training augmentation". Change the heading to "Early structure is not specific to left-right flips or to flip augmentation". Say that the explanation was formed after these predictions failed. Change structure.tex:59 to "whose released teacher configurations switch flips off (the ViT-B configuration is not public)".
- **C8** intro.tex:53-54 ("failed where we tested it") and discussion.tex:12-15. The controlled study never ran. Both pilots (one seed each) developed few twins (twin fraction 0.06 to 0.07), and the flip-trained pilot had the lower LR ratio in both sizes (0.48 against 0.55, and 0.48 against 0.52). Fix: "Our controlled test of this cause could not be run: in two small pilot ViTs neither the flip-trained nor the flip-free model formed many twins, so the study stopped at its pre-registered gate."
- **C9 Second-order argument** (structure.tex:61-63). "Second-order image statistics cannot tell the two mirrors apart" is true only for stationary statistics, that is, the power spectrum. Position-dependent second-order statistics can tell them apart (top against bottom, Torralba and Oliva p. 398; references_verification.md:148-150). Fix: "the power spectrum cannot tell ...", and add that a patch filter shared across positions sees statistics pooled over positions. Also, R180 is LR followed by UD, so R180 structure follows from LR and UD structure in a shared basis and is not independent evidence. Say so at structure.tex:55 and soften abstract.tex:4-5.
- **C10 What the ratio to null means.** In Table 6 the late LR ratio is lower (stronger) than the early one for 10 of 12 transformers (DINOv2-B reg: 0.31 late against 0.48 early), yet structure.tex:44 says the structure is "concentrated in the first half". The cause is that late structure is nearly trivial, so the fitted and null shares are both tiny (block 12 fc2: 0.002 against 0.016, blocks.csv). Early blocks are far from exact: block 1 loses 27% (fc1) and 20% (fc2) of its weight energy to projection, against about 48% for the null. That makes discussion.tex:22, "nearly symmetric", too strong. Also, "1 is chance" (Figure 2 caption, structure.tex:24-25) compares with random structures, not with structures fitted on a non-symmetry; fitting alone may lower the ratio (pilots at epoch 5: 0.92). Fix: two sentences explaining the ratio, the absolute block-1 shares, and the twin-fraction null (a structure fitted after a random patch permutation gives at most 0.0035 against 0.46 to 0.56; P4.2).
- **C11 Pairing is not universal.** abstract.tex:2 ("Many MLP neurons ...") and structure.tex:49 ("far from chance in all 14 models") overstate it. DINOv2-g (twin fraction 0.06) and EVA-02 (0.05) are exactly the two SwiGLU models (`mlp_type` in their configs). With a gated unit silu(x1)·x2, a single unit can itself be odd, so twin pairs need not form. structure.tex:50 instead blames g's 20-block first half, which is untested. Fix: "in the 12 models with GELU or ReLU units; the two SwiGLU models pair few units, plausibly because a gated unit can itself be odd (verify)".
- **C12 How general the method is.** abstract.tex:7 says "a pretrained transformer" and intro.tex:35 "a pretrained ViT". The code refuses SwiGLU, rotary embeddings, q/k-norm and `norm_pre` (method_spec §4), so 4 of the 14 measured models cannot be converted (DINOv2-g, CLIP, DINOv3, EVA-02). The converted model is also tied to 224 px, because the position grid is materialised; 448 px needed re-projection. Fix: name both limits in Scope, and say "a ViT with GELU MLPs and learned position embeddings".
- **C13 Exactness caveats.** method.tex:53-55: the float32 error of 2.5e-5 is above the pre-registered 1e-5 bar (MUST-02 FAIL), but the main text does not say so. experiments.tex:68-70 says Mirror Surgery "composes" with int8 and ToMe, yet both break the invariance (float32 flip consistency 99.09% and 96.80%, numbers.md T3; UP-06 is seed 0 only). Fix: add the bar, and "neither keeps the exact invariance (flip consistency 99.1% and 96.8%)".
- **C14** related.tex:22 says "exact rather than approximate equivariance", but group averaging, cited in the same paragraph, is also exact. Fix: "like group averaging it is exact, but costs one pass instead of two". related.tex:24 says "needs no retraining", but 780 distillation steps are needed. Fix: "needs no training from scratch".
- **C15 CUB comparison** (experiments.tex:150, 170-172):
  - The original's odd part uses A_L with d_- = 8 while Mirror's has 68 dimensions, so "mixes the two" compares parts of different sizes from different operators. State the dimensions, or redo the original's split with A* (a CPU probe).
  - Chance level on the mirror-augmented set is guaranteed by construction. Say "as it must be".
  - intro.tex:45, "carry no information": natural images give 50.57%. Say "essentially none".
- **C16** structure.tex:74, "the structure is left-right symmetric in every model", says something else than intended. Fix: "left-right structure is present in every model".
- **C17** discussion.tex:24-25 raises "vision-only and image-text models in deep left-right pairing", which no results section discusses (it shows only in the late ratios of Table 6, 0.55 and 0.58, and the colours of Figure 2b). Fix: add one sentence with these numbers to Section 3, or drop it.
- **C18** discussion.tex:10-12: the "per-layer rule that we froze in advance" appears only here. Fix: add one clause of context, or move it to Appendix A.
- **C19 Timing of the pre-registration.** appendix.tex:4 ("Before any of the experiments in this paper ran") and intro.tex:47-48 overstate it. An exploratory phase (E1, E1b, X2) came first, and Mirror Surgery reloads the per-layer operators and twin pairings fitted in run X2 (method_spec §1; MUST-03 checks that they reproduce). Fix: "The pre-registration followed an exploratory phase on DINOv2-B in which we found the twins. Every number reported here comes from runs started after it was committed; structures fitted in the exploratory phase were reused after a reproduction check (MUST-03)."
- **C20** appendix.tex:6-9 calls CORE-01 PENDING, but Table 5 scores CORE-01 T2 and T4 as FAIL (on the pilots), and P11.2 is PENDING without a stated reason. Fix: say both.
- **C21** appendix.tex:71, "the paper uses the first [fit set]". Table 3 and MUST-01 use all three. Fix: "healed models and structure measurements use the first; error bars before healing use all three".
- **C22 Things a reviewer will ask for in Setup:**
  - The k-NN bank is a 100k subset, so the original scores 78.95, below the published full-bank DINOv2 k-NN (verify the published figure). State this before the abstract's 78.34% is read as weak.
  - Healing cost: the heal stage took 27 min for Mirror s0 and 60 min for SVD-LLM s0 on a shared H200 (`timing` in metrics.json).
  - "Zero-shot" means "without healing". That reads oddly in vision; see S1.

## 3. Citations

- **R1 Missing credit: bokman2025flopping.** The abstract of arXiv 2502.05169 splits features into mirror-symmetric and mirror-antisymmetric parts, making linear layers block-diagonal at half the FLOPs, with wall-clock gains. method.tex:37-49 presents the same even/odd form without citing it, and related.tex:23-24 says only that it trains from scratch. Fix: cite it in "Even and odd coordinates" and say what is new here: a learned operator with odd directions in a LayerNorm residual stream, head pairing, and conversion of released models. Note that their kernels give a speed-up and ours do not yet.
- **R2** shahverdi2026identifiable (related.tex:11-12), "and observe this in small models trained from scratch". The abstract names no experiments, only that the structure is "consistently observed in practice". Mark it (verify) against the full text, or drop the clause.
- **R3** rahman2026tunable (related.tex:19-20). The abstract supports "projecting the weights into a designed subspace" of any pretrained architecture, with soft (bounded) equivariance; fine-tuning is not in the abstract (verify). Use it as the "approximate" contrast in C14.
- **R4** bokman2023investigating (related.tex:9-10), "showed that CNNs trained to be flip-invariant are close to ...". They trained with flip augmentation and gave evidence by merging with the mirrored copy. Fix: "found evidence that CNNs trained with flip augmentation are close to ...".
- **R5** vanderschaaf1996modelling (structure.tex:64-65). Only its abstract was checked. It supports "power varies with orientation" (fine as used), not a horizontal or vertical excess. Keep it paired with Torralba and Oliva.
- **R6** wang2019learning (experiments.tex:99-101). The dataset README says "we augment the data set by flipping and rotating the images". Check that the NeurIPS paper says it too; otherwise cite the README (verify). Also report the deduplicated Sketch result, which already exists (Mirror 55.03 mean against SVD-LLM 50.43, numbers.md T7): the lead survives deduplication.
- **R7** touvron2022deit, "DeiT III's fine-tuning rotates by at most 30 degrees" (structure.tex:57). The source is timm's RandAugment setting (configs/models/deit3_b16.yaml), not the paper (verify), or cite timm.
- **R8** olah2020naturally (related.tex:7), "rotated or mirrored curve detectors". The article shows rotated curve detectors and mirrored units, for example left- and right-facing heads. Fix: "rotated curve detectors and mirrored pairs of units".
- **R9** Useful addition: nordstrom2025quick (dihedral and octic ViTs, already in the bib) in the Outlook on the four-element group. Correct as used: radford2021learning (random crop only), caron2021emerging (k = 20), wu2018unsupervised (weighted k-NN; its k = 200 is rightly not claimed), and the dataset citations.

## 4. Style

**S1 Terms that are undefined, defined late or confusing**
- "Symmetry tax" is never given as a number, although Figure 1 (42%) and Table 5 use one. Add to Section 3: "We call the share of an MLP's weights that tying twins would remove, 1 minus free over dense parameters, its symmetry tax."
- "Breaking share" and "ratio to null" (structure.tex:21-26) are defined inside a long chain. Rewrite: "To measure how close the weights are to this form, we project each weight matrix onto it and record the share of its energy that the projection removes, s. A share of 0 means exact symmetry. A random structure also removes energy, so we divide s by its mean over 20 random structures of the same size. A ratio to null of 1 means no closer than chance; lower means closer."
- "Even head" (method.tex:34) clashes with attention heads and with the segmentation head. Rename it "even projection".
- "Twins" means all matched pairs in the text (1,168 at structure.tex:48) but pairs with correlation of at least 0.5 in Table 6. Fix: "matched pairs" against "strong twins (correlation at least 0.5)".
- "Zero-shot", meaning without healing, reads as CLIP-style classification to a vision reader. Fix: "before healing" or "training-free".
- "Random involutions" (structure.tex:24). Fix: "random mirror operators".
- "Character formula of representation theory" (method.tex:46-47). Fix: "the standard count of equivariant linear maps", or drop.
- "A head symmetrised with A*" (experiments.tex:101) is not explained. Fix: "a segmentation head averaged over the image and its mirror".
- Table 5 uses code names: Mirror-exact, vexact_s0, best_svdllm_s0, dinov2_vitb14_reg, "gate C", "switch-off median cosine", "tax" and runs/... paths. Map these to paper names in `scripts/make_paper_prereg_table.py` and add a three-line glossary above the table.

**S2 Ambiguous or awkward sentences**
- experiments.tex:52-53: "the other compressors disagree with themselves on 4 to 6% of mirrored validation images, and so does the original model on 3.7%." Rewrite: "The other compressors change their prediction on 4 to 6% of validation images when the image is mirrored; the original model does so on 3.7%."
- structure.tex:58-59: "CLIP ... has a left-right ratio of 0.51, in the middle of the flip-trained models, and so does DINOv3 (0.45)". Rewrite: "CLIP, whose only augmentation is a random crop, has a left-right ratio of 0.51, within the range of the flip-trained models (0.42 to 0.77). DINOv3, trained without flips according to its released teacher configurations, has 0.45."
- intro.tex:33: "just as strong in CLIP and DINOv3". CLIP's twin fraction (0.33) is well below that of the two DINOv2-B models (0.54 and 0.57). Rewrite: "of similar strength in CLIP and DINOv3".
- experiments.tex:53-54: "The comparison is fair in the sense that matters for the claim". Replace with the list in C5.
- appendix.tex:20: "real and usable, not only measurable". Rewrite: "Early up-down structure can also be used:".
- discussion.tex:22-24: "nearly symmetric ... the four-element group of all three symmetries". Rewrite: "about as close to up-down and 180-degree symmetry ... the group generated by left-right and up-down flips, which contains the 180-degree rotation".
- discussion.tex:29: "an exactly flip-equivariant model". Rewrite: "an exactly flip-equivariant backbone whose class token is flip-invariant".
- Naming: "DINOv2-B" refers to the registers model in abstract.tex:9 and intro.tex:39-40, but Table 6 uses the same name for the model without registers. Use "DINOv2-B/14 with registers", or define a short name once. "ImageNet k-NN" at intro.tex:40 should be "$k$-NN".
- Title and intro.tex:21 ("collects it"): mildly catchy rather than plain. Optional.

**S3 Repetition.** Each of these should appear once in the body plus once in the abstract:
- 78.34 against 78.06 appears in the abstract, intro and Section 5.
- "Before healing it loses to SVD-LLM" appears four times (abstract.tex:14-15, intro.tex:52-53, experiments.tex:138-141, discussion.tex:5-8).
- "Pairing does not need flips" appears three times.

**S4 Hard paragraphs**
- structure.tex:4-31 ("Measuring mirror structure") packs the operator fit, twins, the equivariance condition, breaking share, ratio to null, the other symmetries, the data and the model list into one paragraph. Split it into four short paragraphs.
- method.tex:29-32 (attention): add "B_h is a rotation inside the head's query and key space".

**S5 Typesetting**
- Equation 2 is 20 pt too wide (main.log), and its number drops to its own line on page 5. Fix: split it over two lines.
- Table 5 prints `\{}|ds_fc1\{}|` and `\{}|difference\{}|` (MUST-03 and MUST-10 rows) because the generator escapes the markdown `\|` again. Fix: unescape `\|` to `|` before LaTeX escaping. The table is also 15 pt too wide.
- The healing-curve table (appendix.tex:32-46) has no caption or label.
- British spelling and no em-dashes: no problems found.

## 5. Structure and page budget

The main text ends two lines into page 9 (main.pdf), so about 0.9 page is free. The fixes above add roughly 20 to 30 lines, so nothing has to go. If space is needed, cut in this order (least loss first):
1. intro.tex:52-55, "The failures are informative ..." (5 lines). It repeats Section 6 and contains C8.
2. structure.tex:73-77, "What this means for compression" (5 lines, contains C1). Replace it with one sentence opening Section 4.
3. discussion.tex:27-30, Conclusion (4 lines). It repeats the abstract; fold one sentence into the Outlook.
4. intro.tex:28-50, contribution bullets (about 20 lines). Cut each to two lines, drop numbers repeated from the abstract, and drop the "144 blocks" clause (saves about 7 lines).
5. Table 4 (CUB, about 10 lines with caption). Replace it with two sentences of numbers and move the table to the appendix, or merge it into Table 2 as two columns.
6. related.tex:26-34, the structured-compression paragraph. Cut it to four lines.
7. Only if needed: move Table 3's 9-block column to Appendix C. Keep the pairing-control rows; they carry the "structure matters" argument.

Do not shorten Setup or Scope; both need to grow (C5, C6, C12, C22).

## 6. Reviewer view

**Five strongest objections, ranked, with the cheapest fix**

1. **The compression buys nothing at deployment.** The 23.5% saving exists only in a form that is 28% slower than the original and slower than SVD-LLM at batch 256 (0.76 against 0.94). The fast model is the dense one, which saves nothing. At a parsimony venue this is the first thing a reviewer will hit, and the abstract's "twice the throughput" and "39% of the cost" will read as spin once Table 1's caption is read. *Writing:* state the trade-off in the abstract, split Table 1 into two rows, and score UP-05. *Small experiment:* one fused GEMM per even/odd block (concatenate the even and odd products), or report the stored size of the even/odd checkpoint.
2. **The win is small and rests on thin support.** +0.28 k-NN rests on one model, one size and one fitted structure, against simplified baselines that were healed with different block targets. Downstream gaps are single runs. *Writing:* C4 to C6. *Small experiments:* a paired bootstrap from the stored predictions (CPU, minutes); heal Mirror Surgery from fit seeds 1 and 2 (about two GPU-hours); SVD-LLM probes for seeds 1 and 2.
3. **The structure story is post hoc, and its key evidence is confounded or underpowered.** The two no-flip models are CLIP and DINOv3-B, with an inferred recipe. The pilots barely paired, R180 follows from LR and UD, and the ratio to null behaves counter-intuitively in late blocks with no fitted null. *Writing:* C7 to C11. *Small experiment:* the ratio to null for a structure fitted after a random patch permutation, on the four MUST-04 models (the code already exists for the twin-fraction null).
4. **Novelty against Bökman et al. (2023, 2025) is not drawn.** The even/odd block-diagonal form, the permutation picture and the halved cost already exist. *Writing:* R1, plus a short paragraph on what is new: post-hoc conversion of released models, a learned operator with odd directions in a LayerNorm residual stream, head pairing, and the four-symmetry and 14-model measurements.
5. **"Exact" holds only in float32, at 224 px, without quantisation or token merging.** The headline bf16 pipeline disagrees on 0.27% of images, and the pre-registered float32 bar failed. *Writing:* P1, C12, C13. *Optional:* report float32 k-NN for the healed models next to bf16 (the values exist: 78.345 against 78.075 for SVD-LLM).

**Score: 2.5 out of 5 as written (borderline reject).** The 14-model measurements, the up-down finding and the open pre-registration are a real contribution. But the compression result is a 0.28-point gain from a model that is slower once its parameters are actually saved, and several headline sentences (speed, exactness, "beats", the natural-image cause) claim more than the data show. With the writing fixes alone I would expect about 3.5.
