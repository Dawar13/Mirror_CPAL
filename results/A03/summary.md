# A03 discovery null controls: dinov2_vitb14_reg

**Rule A03-1 (pre-registered): FAIL.** For each of C1, C2 and C3: twin fraction in blocks 1 to 6 at most 0.05 and split-half ratio to null at least 0.9 (PREREG_FINAL_PHASE.md Section 2.4).

- C1: PASS (twin fraction 0.004, split-half ratio to null 0.977)
- C2: PASS (twin fraction 0.000, split-half ratio to null 1.027)
- C3: FAIL (twin fraction 0.079, split-half ratio to null 0.922)

Diagnostic, not the pre-registered score: the same rule with the same-sample MUST-04 ratio gives FAIL (C1 0.970, C2 1.044, C3 0.921).

Fit set: /home/Mirror_CPAL/cache/train224_uint8.npy rows fit0 (2000 images, stored order). Range: blocks 1 to 6 (1-based). Null: 20 draws, base seed 0. Control seed 0.

| control | twin fraction | ratio to null | split-half ratio | d- of A* | spectrum sum abs | odd share | zero-shot k-NN |
|---|---|---|---|---|---|---|---|
| mirror | 0.568 | 0.479 | 0.480 | 68 | 0.865 | 0.053 | 53.86 |
| C1 | 0.004 | 0.970 | 0.977 | 338 | 0.353 | 0.029 | 2.66 |
| C2 | 0.000 | 1.044 | 1.027 | 4 | 0.506 | 0.001 | 12.52 |
| C3 | 0.079 | 0.921 | 0.922 | 199 | 0.598 | 0.134 | 5.16 |
| C4 | 0.004 | 1.000 |  | 68 |  |  | 4.79 |

Original model k-NN: 78.95 (computed, cached to /home/Mirror_CPAL/cache/knn_original/dinov2_vitb14_reg_v2_bf16.json).

Reference against results/MUST-07/structure/dinov2_vitb14_reg_s0.npz: {"operator_agreement": 0.9999999992174667, "span_agreement": 0.9999999998379083, "pair_recovery": 1.0, "twin_recovery": NaN, "d_plus_span_refit": 700, "d_plus_span_canonical": 700}.

## How to read this

- Twin fraction: share of hidden units in a matched pair whose symmetrised correlation is at least 0.5, mean over the range's blocks. Ratio to null: mean breaking share of the real MLP weights under the row's fitted structure, divided by the mean over the same blocks of the null's mean (random involutions with the fitted d+, random pairings with the fitted cycle type); 1 means no better than random. Both exactly as MUST-04.
- C4 is the null itself: its ratio is each draw against the mean of the other draws, and its twin fraction scores the random pairings with the reference's correlations.
- The same-sample ratio of any fitted structure sits below 1 even without a mirror relation, because the matching and the operator are fitted to the same cross-moments (about 0.88 on the synthetic equivariant network of the CPU tests). The split-half ratio fits the operators on one half of the fit images and the pairings on the other; it removes that effect.
- Prefix tokens (CLS, registers) keep their index in every row, and C1 keeps the true re-alignment, so any part of the tokens that does not depend on the image stays mirror paired in C1; C2 and C3 keep features that do not depend on position (C2) or on the pixel order inside a patch (C3).
- d- and the spectrum: an involution fit always returns an involution, so d- alone does not show structure. The sum of absolute eigenvalues of the span matrix (1 for an exact symmetry at every layer) and the odd share (absolute mass on negative eigenvalues) do.
- Zero-shot k-NN: full Mirror-exact surgery with the row's own structure (C4: the random-structure method), no healing.
