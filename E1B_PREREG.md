# E1b pre-registration: per-layer mirror operators

Written on 2026-11-07 and committed **before any E1b run**; the commit that adds this file is the
pre-registration record. Thresholds are unchanged from `E1_INSTRUCTIONS.md` (B4) and the decisions
of 2026-11-07.

## Why E1b exists

E1 tested one mirror operator shared by all layers (fitted on layers 1 to 12) and returned
**KILL** (rows 1, 4, 5; `results/E1_REPORT.md`). A post-hoc diagnostic
(`results/E1_DIAGNOSTIC.md`) found that **one operator per layer** would pass every row. That
variant was chosen after seeing the data, so E1b tests it on fresh data with the rule fixed in advance.

## Arms

| Arm | Config | Model | Operator | Data split |
|---|---|---|---|---|
| **Primary** | `configs/e1b_perlayer_b14.yaml` | DINOv2 ViT-B/14 | one per layer | seed 1 |
| Secondary | `configs/e1b_perlayer_b14reg.yaml` | DINOv2 ViT-B/14 with 4 registers | one per layer | seed 1 |
| Exploratory | `configs/e1b_shared16_b14.yaml` | DINOv2 ViT-B/14 | one shared, fitted on layers 1 to 6 | seed 1 |

Split seed 1: within each class the 50 validation images are shuffled with seed 1 before the
2 / 1 / 39 / 8 split, so fit, held-out, bank and query images all differ from E1 (apart from
chance overlap). Everything else is as in E1: seed 0, fp32, k-NN k = 20, T = 0.07, surgery on the
first k in {6, 3, 9} blocks.

## Method in per-layer mode

- A_l is fitted separately for each layer on the fit split; errors are measured on held-out images.
- Block b reads layer b-1 and writes layer b (1-based): fc1 is tested against the input-layer
  operator and fc2 against the output-layer operator, so a projected MLP maps the input mirror to
  the output mirror.
- Nulls: random involutions with the same d+ as each layer's operator, and random pairings with the
  same numbers of pairs and self-twins per block (3 seeds; the first is used for random surgery).
- SVD baseline: rank matched to the equivariant parameter count of each matrix separately.

## Decision rule (unchanged)

Rows 1 to 5 over blocks 1 to 6 with the E1 thresholds and aggregation. In per-layer mode rows 1
and 2 use each layer's own operator and its matched null, and row 3 is the mean over layers 1 to 6
of d- / d of A_l. Outcome, in order: KILL if any KILL cell in rows 1, 4 or 5; GO if all rows are
GO; "PIVOT: structure but little to save" if d- / d < 0.1; PIVOT if rows 1 to 4 are GO and the
surgery drop is above 10 points; otherwise AMBER.

## What the result means (decided now)

- **Primary GO**: proceed to E2 and E3 with the per-layer design; the paper is reframed around
  per-layer mirror structure (the single-operator claim is dropped).
- **Primary KILL**: the mirror-surgery line stops (firm no-go).
- **Primary PIVOT or AMBER**: human decision, informed by the secondary and exploratory arms.
- The secondary and exploratory arms are reported in full but do not override the primary.

## Known limits, stated in advance

- Per-layer surgery does not make the whole network exactly mirror-equivariant, because the
  residual connection carries the input operator while the MLP writes with the output operator.
  Flip consistency below 100% is expected.
- If layers 9 to 11 again show errors above 1 for every operator, the registers arm indicates
  whether DINOv2's high-norm artefact tokens are the cause (not yet verified).
