# E1b summary: per-layer mirror surgery

**Pre-registered verdict: GO.** The primary arm (one mirror operator per layer, DINOv2 ViT-B/14,
fresh data split) passes all five checks with the original thresholds. Under the rule fixed in
`E1B_PREREG.md` (commit `93bd945`, before any E1b run), the project proceeds to E2 and E3 with the
per-layer design, and the single-operator claim is dropped.

Runs: 7 to 8 November 2026, friend's A100-SXM4-80GB (shared with another project, so memory was
capped at 12% and 6%; this does not change results), seed 0, split seed 1, fp32. Reports, tables
and plots for each arm are in `runs/<arm>/`.

## Outcomes

| Arm | Role | Outcome | Rows not GO |
|---|---|---|---|
| Per-layer, ViT-B/14 (`runs/e1b_perlayer_b14`) | **primary** | **GO** | none |
| Per-layer, ViT-B/14 with registers (`runs/e1b_perlayer_b14reg`) | secondary | AMBER | 4 |
| Shared operator from layers 1 to 6, ViT-B/14 (`runs/e1b_shared16_b14`) | exploratory | AMBER | 1, 4 |
| E1 for reference: shared operator from layers 1 to 12 (`results/E1_REPORT.md`) | | KILL | 1, 4, 5 |

## Decision table, blocks 1 to 6

| Row (GO limit) | Primary | Registers | Shared 1 to 6 |
|---|---|---|---|
| 1. e(null) / e(A), >= 5 | **10.57** (worst layer 4.55) | 13.18 (worst 5.74) | 4.39 (worst 3.44) |
| 2. e(A) / e(I), <= 0.5 | **0.160** | 0.169 | 0.373 |
| 3. d- / d, >= 0.2 | **0.260** | 0.276 | 0.225 |
| 4. breaking share, <= 0.2 (matched null) | **0.194** (0.416) | 0.223 (0.427) | 0.244 (0.392) |
| 5. surgery k = 6: drop <= 10, margin over SVD >= 5 | **3.62, +9.31** | 1.77, +68.26 | 4.75, +6.64 |

## Zero-shot surgery (k-NN top-1, %)

| Blocks operated on | Arm | Original | Mirror surgery | Rank-matched SVD | Matched random |
|---|---|---|---|---|---|
| 3 | primary | 77.35 | 75.74 | 76.06 | 6.58 |
| 6 | primary | 77.35 | **73.73** | 64.41 | 3.78 |
| 9 | primary | 77.35 | 71.69 | 34.80 | 3.35 |
| 3 | registers | 78.11 | 77.35 | 75.81 | 3.20 |
| 6 | registers | 78.11 | 76.34 | 8.08 | 3.31 |
| 9 | registers | 78.11 | 75.79 | 5.84 | 4.63 |
| 3 | shared 1 to 6 | 77.35 | 76.30 | 76.56 | 18.68 |
| 6 | shared 1 to 6 | 77.35 | 72.60 | 65.96 | 4.16 |
| 9 | shared 1 to 6 | 77.35 | 46.59 | 26.79 | 3.81 |

Implied MLP parameter saving in blocks 1 to 6 with per-layer operators: 40.5% (primary).

## What we learned

1. **The per-layer result replicates on fresh data.** Rows 1 to 4 of the primary arm match the
   post-hoc E1 diagnostic to within about 0.01, on different fit, held-out, bank and query images.
2. **Mirror surgery degrades gently; same-size SVD does not.** The advantage grows with depth
   (k = 3: SVD slightly ahead by 0.3 points; k = 6: surgery ahead by 9.3; k = 9: by 36.9). Random
   structure with the same counts destroys the model, so the fitted structure is specific.
3. **The late-layer failure in E1 comes from DINOv2's artefact tokens.** With registers, layers 9
   to 11 align well (per-layer error 0.13 to 0.14, against 1.17 to 1.45 without registers).
4. **One operator for the early half is close but weaker.** Fitted on layers 1 to 6 it has
   d- / d = 0.225 and passes rows 2, 3 and 5, but not rows 1 and 4; per-layer operators fit
   clearly better (layer 1 error 0.037 against 0.213).

## Caveats

- **Registers arm is AMBER, narrowly.** fc1's breaking share is 0.223 against the 0.2 limit (its
  matched null is 0.427); every other row is GO. It does not override the primary under the
  pre-registered rule, but the paper should report it.
- **SVD collapses on the registers model** (8.08% at k = 6, against 64.41% without registers at a
  similar budget). This is unexpectedly large and should be checked before it is quoted (verify).
- **Not exactly equivariant.** Per-layer surgery makes each MLP map the input mirror to the output
  mirror, but the residual connection carries the input operator, so flip consistency drops from
  96.1% to 89.7% (primary, k = 6). An exactly equivariant block-diagonal network (E2) needs a way
  to handle operators that change between layers; this is open.
- **k = 3 is a tie with SVD.** The case for mirror surgery rests on cutting 6 or more blocks.
- Single seed per arm; the row 1 worst layer in the primary arm (layer 6, 4.55) is below 5,
  although the mean passes.

## Next decisions (not taken here)

- E2 design: block-diagonal implementation per layer, and how to bridge operators between layers.
- E3: healing distillation against SVD, SliceGPT and width pruning at equal FLOPs, using per-layer
  operators and k = 6.
- Check the SVD collapse on the registers model, and consider using the registers model as the main
  model (it is cleaner in late layers).
