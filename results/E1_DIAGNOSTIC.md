# E1 diagnostic: why the pre-registered test says KILL, and what per-layer operators show

**Status.** The pre-registered E1 decision is **KILL** (rows 1, 4 and 5; see `E1_REPORT.md`, commit
`c74827f`). This note is a **post-hoc diagnostic** (commit `29b1ad6`, `scripts/e1_diag_perlayer.py`).
It does not change that decision. It was designed after seeing the E1 numbers, so it needs a
pre-registered confirmation before any claim rests on it.

Model DINOv2 ViT-B/14, ImageNet-1k validation (fit 2,000, held-out 1,000, k-NN bank 39,000,
queries 8,000), seed 0, A100-SXM4-80GB, full fp32.

## 1. What failed: the single shared operator

E1 fits one mirror operator from the cross-moments summed over layers 1 to 12. That sum is dominated
by layers 9 to 11, where no operator lines the mirrored tokens up with the original ones: relative
held-out error is above 1.0 for every operator, including each layer's own best fit
(`e1_layers.csv`, column `rel_e_A_layer`: 1.46, 1.37, 1.18). The shared fit therefore collapses
almost to the identity (d+ = 760, d- = 8), and everything downstream inherits that:

- row 1: e(null) / e(A_shared) = 1.10 (KILL below 3);
- row 4: breaking shares 0.307 and 0.256, equal to their matched nulls (0.307, 0.308);
- row 5: surgery on blocks 1 to 6 drops k-NN top-1 from 78.21 to 14.10, against 72.45 for
  rank-matched SVD and 36.83 for random-involution surgery.

A likely cause, **not yet verified**: DINOv2 without registers is known to produce a few
high-norm "artefact" tokens in late layers, and their positions need not move consistently under
a mirror. (verify: count high-norm tokens per layer in original and mirrored images.)

## 2. What holds: one operator per layer

Each layer has its own excellent mirror operator A_l (fit on the fit split, scored on held-out
images). Applying the same aggregation as the decision table (mean over layers or blocks 1 to 6)
to the per-layer operators:

| Row | Check | Shared operator (pre-registered) | Per-layer operators (diagnostic) | GO limit |
|---|---|---|---|---|
| 1 | e(null) / e(A) | 1.10 | 10.58 (worst layer 6: 4.55) | >= 5 |
| 2 | e(A) / e(I) | 0.938 | 0.160 | <= 0.5 |
| 3 | d- / d | 0.010 | 0.259 (0.129 to 0.362) | >= 0.2 |
| 4 | breaking share s1, s2 (matched null) | 0.307, 0.256 (0.307, 0.308) | 0.194, 0.169 (0.416, 0.394) | <= 0.2 |
| 5 | surgery k = 6: drop; margin over SVD | 64.11; -58.35 | 4.29; +8.87 | <= 10; >= 5 |

Sources: rows 1 to 3 from `e1_layers.csv` (columns `rel_e_null_layer`, `rel_e_A_layer`,
`rel_e_I`, `dminus_frac`); rows 4 and 5 from `e1_diag_perlayer.csv` and
`e1_diag_perlayer_surgery.csv`. In the diagnostic, fc1 of block b is tested against the operator
of its input layer and fc2 against its output layer; nulls have the same d+ per layer and the same
numbers of pairs and self-twins; the SVD rank is matched to the per-layer parameter count.

Zero-shot surgery with per-layer operators (k-NN top-1 and flip consistency, %):

| k | Per-layer surgery | Rank-matched SVD | Matched random structure | MLP parameter saving (first k blocks) |
|---|---|---|---|---|
| 3 | 76.70 (93.15) | 76.00 (93.38) | 5.00 (12.83) | 46.6% |
| 6 | 73.93 (89.54) | 65.06 (82.59) | 3.81 (14.14) | 40.5% |

Original model: 78.21 (96.58).

Other observations:
- **Mirror-twin neurons are real.** In block 1, 74% of MLP neurons sit in mirror pairs. Their
  activations correlate with their twin's at about 0.77 on average, while their correlation with
  themselves is near zero (`plots/twin_scores.png`).
- **The odd share falls with depth.** d- / d of A_l goes from 0.49 at layer 0 to 0.13 at layer 6
  and about 0.01 from layer 10 on; the twin fraction falls from 0.74 (block 1) to 0.01 (block 12).
  Savings, if real, are concentrated in the first half of the network.
- **Position embedding.** ||mirror(P) - P A_0|| / ||P|| = 0.256, against 0.967 for the identity.
- Layer 0 is near exact (e(A_0) = 0.0047) partly by construction: the patch embedding is linear.

## 3. Caveats before anything is claimed

1. **Post hoc.** The per-layer variant was chosen after seeing E1. It needs a pre-registered rerun
   (same thresholds) on a second fit split or seed, and ideally on DINOv2 ViT-B with registers.
2. **One operator through depth is rejected in this model.** Consecutive A_l differ, while
   section 4.2 of `PROJECT_CONTEXT.md` notes that an exactly equivariant residual network needs
   the same operator on the span of the stream. Per-layer surgery only makes each MLP map the input mirror to the
   output mirror; it does not make the whole network exactly equivariant (flip consistency after
   surgery is 89.5%, not 100%). Whether the per-layer operators agree on the high-energy part of
   the stream is untested (an energy-weighted agreement would answer it).
3. At k = 3 the margin over SVD is only 0.70 points; the gap opens at k = 6.

## 4. Decisions needed (not taken here)

- Whether to pre-register an "E1b" with per-layer operators (and, separately, a variant that
  excludes layers 9 to 11 from the shared fit) before E2 and E3.
- Whether to check the artefact-token hypothesis with `dinov2_vitb14_reg`.
