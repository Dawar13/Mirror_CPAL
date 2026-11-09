# DRAFT, not active until Dawar signs off

# PREREG_E3: matched healed baselines at full conversion (k = 12)

Nothing in this file has been run on a GPU. Once signed off, the commit
that adds the signed-off file is the pre-registration record: every later E3 report states that commit hash, and no
bar, arm or rule below changes after it. The jobs file `jobs/e3_k12.txt` must not be launched before that commit.

## 1. Why E3

Gate G2 (PREREG_E2.md, P3) failed. At k = 9 (registers model, MLP plus attention, Mirror V-local), healed with the M3
pilot recipe (sources: `runs/m3_pilot/<method>/knn.json`, collected in `results/paper/tables/T_healing.md`):

| model | k-NN top-1 (v2 protocol) |
|---|---|
| original | 78.94 |
| healed Mirror | 78.81 |
| healed SVD | 77.57 |
| healed random-structure control | 73.28 |

P3.a (Mirror minus random, +5.54, bar 2.0) passed; P3.b (Mirror minus SVD, +1.24, bar 3.0) failed. The second bar
could not be met by any student distilled from this teacher: healed SVD ended 1.37 points below the original, so
passing would have needed healed Mirror at 80.57 or more, 1.63 points above the original model it is distilled from.

At full conversion the student starts far from the teacher, so healing has real work to do. V-exact with all 12
blocks converted scores 53.85 k-NN zero-shot and 78.31 after healing on the A100 (`runs/m3_vexact/k12/knn.json`), and
53.83 and 78.34 on the RTX 5090 (`runs/m3_vexact_5090/k12/knn.json`). There is no healed baseline at k = 12 with the
same parameter budget and recipe, so this headline cannot yet be credited to the mirror structure. E3 adds those
baselines and a second V-exact seed.

## 2. What is known before E3 runs, and what is not

Known (already measured; listed so that nobody can say the bars were set after seeing E3 results):

| quantity | value | source |
|---|---|---|
| V-exact k = 12, seed 0, RTX 5090: original / zero-shot / healed k-NN | 78.93 / 53.83 / 78.34 | `runs/m3_vexact_5090/k12/knn.json` |
| same run: fp32 flip check, exactness inside the span | 100%, 8.7e-06 (tol 1e-4) | `flipcheck.json`, `exactness.json` |
| same run: learning-rate trials (held-out loss after 100 steps) | 1e-5: 2.67153; 5e-5: 2.22745; 1e-4: 2.16720 (chosen) | `lr_selection/lr_selection.json` |
| V-exact k = 12, seed 0, A100: original / zero-shot / healed k-NN | 78.94 / 53.85 / 78.31 | `runs/m3_vexact/k12/knn.json` |
| V-exact kept matrix parameters (all 12 blocks) | 64,554,464 of 84,934,656 (0.760); MLP 42,693,344, attention 21,861,120 | `runs/m3_vexact_5090/k12/summary.json` |
| A* at k = 12 | d+ 700, d- 68; register pairing identity (no register pairs) | `runs/m1/m1_structure.npz` (sha256 9a10a17e...) |
| M3 pilot chosen learning rate | 1e-4 for Mirror, SVD and random (the largest rate of the grid) | `runs/m3_pilot/<method>/lr_selection/lr_selection.json` |
| M3 pilot budget (blocks 1 to 9) | Mirror keeps 42,097,128 of 63,700,992 matrix parameters; healed SVD 77.57 at that budget | `runs/m3_pilot/mirror/summary.json`, `T_healing.md` |

Unknown (not run): every E3 arm below (healed SVD with either target, healed random structure at k = 12, V-exact
seed 1), the zero-shot k-NN of the new arms, every 2e-4 trial, and the Tier-2 linear probe of every E3 student.

## 3. Arms

All arms: `dinov2_vitb14_reg`, all 12 blocks converted, MLP and attention. One shared recipe, the one of
`configs/m3_vexact.yaml` (itself the M3 pilot's budget):

- data: the 100,000 ImageNet train images of the healing cache (`heal_rows`); 5,000 other train images
  (`healval_rows`) only for the learning-rate choice; nothing from validation;
- 2 epochs, 780 steps of 256 images, AdamW (weight decay 0.05), 5% linear warm-up then cosine decay to 0, bf16
  autocast, random resized crops (scale 0.35 to 1.0) to 224 px with flips (p = 0.5);
- loss: normalised MSE on the final-norm CLS token, the final-norm patch tokens and the output of each of the 12
  blocks, weight 1 each, normalisers fixed before training on 1,024 train images;
- learning rate chosen from {1e-5, 5e-5, 1e-4} by the held-out loss after 100-step trials (same seed per trial);
- trainable set: the converted blocks and the embeddings (patch embedding, the 16 x 16 position grid with its CLS
  row, CLS token, registers); the final norm affine and the mask token stay frozen; the norm and LayerScale
  parameters inside the blocks are absorbed into the weights and frozen;
- healing seed 0 unless stated.

| arm | what | config and command | output |
|---|---|---|---|
| 1 | V-exact, seed 0 (exists, not rerun) | `configs/m3_vexact.yaml --k 12` | `runs/m3_vexact_5090/k12/` (A100 copy: `runs/m3_vexact/k12/`) |
| 2 | V-exact, healing seed 1 (same A*, pairings, head maps; the seed sets data order, crops, flips) | `python scripts/heal_vexact.py --config configs/e3_vexact_s1.yaml` | `runs/e3/vexact_s1/k12/` |
| 3a | SVD at matched parameters, raw teacher targets (as the pilot's SVD arm) | `python scripts/heal.py --config configs/e3_svd_raw.yaml` | `runs/e3/svd_raw/svd/` |
| 3b | SVD at matched parameters, V-exact final targets (Section 5) | `python scripts/heal.py --config configs/e3_svd_sym.yaml` | `runs/e3/svd_sym/svd/` |
| 4 | random-structure control: the V-exact construction and recipe with a random A* (d+ 700, d- 68, A 1 = 1), random neuron pairings with the fitted numbers of pairs and self-twins per block, random head maps with the fitted head pairs and self-head traces, random register pairing with the fitted number of pairs (zero, so identity); seed 1 | `python scripts/heal_vexact.py --config configs/e3_random.yaml` | `runs/e3/random/k12/` |

Arm 4 must be exactly equivariant under its own random structure: the exactness check (tolerance 1e-4) and the fp32
flip check (100%) run as for V-exact. Its targets are symmetrised with the random A*, as V-exact's are with the fitted
one. Random and fitted operators agree to tr(A_random A*)/d = 0.678, which is the chance level for two involutions
with d+ 700 of 768 (0.677; `scripts/e3_budget.py`): most of the stream is even under both, so the control tests
which 68-dimensional odd subspace and which pairings are used, at equal dimensions.

## 4. Budget matching

The SVD arms take, per block and per matrix, the largest rank r whose factorisation r (rows + cols) does not exceed
the number of free parameters V-exact keeps in that matrix (the rule of the pilot's SVD arm). The per-block counts come
from `mirror.heal_vexact.vexact_block_params`, the accounting that `vexact_param_counts` sums to the 64,554,464 above.
MLP and attention are matched separately in every block. Table from `python scripts/e3_budget.py` on the files the
V-exact run used (shortfall: how far SVD is below V-exact):

| block | MLP ranks (fc1, fc2) | MLP V-exact | MLP SVD | shortfall | attention ranks (q, k, v, o) | attention V-exact | attention SVD | shortfall |
|---|---|---|---|---|---|---|---|---|
| 1 | 367, 367 | 2,824,448 | 2,818,560 | 0.21% | 255, 255, 236, 236 | 1,510,816 | 1,508,352 | 0.16% |
| 2 | 376, 376 | 2,890,176 | 2,887,680 | 0.09% | 257, 257, 235, 235 | 1,514,608 | 1,511,424 | 0.21% |
| 3 | 393, 393 | 3,020,368 | 3,018,240 | 0.07% | 273, 273, 255, 255 | 1,627,104 | 1,622,016 | 0.31% |
| 4 | 401, 401 | 3,083,568 | 3,079,680 | 0.13% | 301, 301, 293, 293 | 1,828,080 | 1,824,768 | 0.18% |
| 5 | 415, 415 | 3,191,008 | 3,187,200 | 0.12% | 289, 289, 281, 281 | 1,754,768 | 1,751,040 | 0.21% |
| 6 | 449, 449 | 3,451,392 | 3,448,320 | 0.09% | 294, 294, 281, 281 | 1,768,672 | 1,766,400 | 0.13% |
| 7 | 468, 468 | 3,600,544 | 3,594,240 | 0.18% | 296, 296, 289, 289 | 1,799,008 | 1,797,120 | 0.10% |
| 8 | 493, 493 | 3,787,616 | 3,786,240 | 0.04% | 299, 299, 303, 303 | 1,852,096 | 1,849,344 | 0.15% |
| 9 | 534, 534 | 4,104,880 | 4,101,120 | 0.09% | 331, 331, 331, 331 | 2,037,904 | 2,033,664 | 0.21% |
| 10 | 544, 544 | 4,183,248 | 4,177,920 | 0.13% | 310, 310, 317, 317 | 1,927,936 | 1,926,144 | 0.09% |
| 11 | 556, 556 | 4,276,784 | 4,270,080 | 0.16% | 341, 341, 348, 348 | 2,118,800 | 2,116,608 | 0.10% |
| 12 | 557, 557 | 4,279,312 | 4,277,760 | 0.04% | 343, 343, 347, 347 | 2,121,328 | 2,119,680 | 0.08% |
| total | | 42,693,344 | 42,647,040 | | | 21,861,120 | 21,826,560 | |

SVD keeps 64,473,600 matrix parameters in all, 0.13% fewer than V-exact; every block is within 0.31%. The run records
the same numbers in `summary.json` (`method_info.vexact_match`) and checks them against the pilot's own rule. The
random arm keeps exactly V-exact's counts in every block (same d+, numbers of pairs and self-heads, self-head traces).

Not counted, as in E2 and M1: biases and embeddings. They train dense in the SVD arms and projected (fewer free
entries) in V-exact and the random arm, which favours SVD slightly.

## 5. Targets of the SVD arms (3a and 3b), and the 3b decision

V-exact's recipe uses symmetrised targets (`mirror/heal_vexact.py`, `VExactTeacher`). That could help or hurt a student
that is not mirror-equivariant, so SVD is healed with both target types and compared at its better one.

- 3a, raw: the teacher's own final-norm CLS and patch tokens and its own block outputs, as in the M3 pilot.
- 3b, V-exact final targets: the final CLS and patch targets are the flip average of the teacher's final-norm
  features, (f(x) + align(f(mirror x))) / 2 (patch grid aligned), and the fixed loss normalisers are measured on these
  targets, the CLS one included, exactly as in V-exact at full conversion. The teacher runs on every batch and its
  mirror, as for V-exact.
- 3b block targets: **the teacher's own block outputs, not centred and not symmetrised with A*.** Reasons:
  1. Centring: V-exact's block targets are centred over channels because its student is LayerNorm-folded and its
     stream is the centred stream. The SVD student keeps LayerNorm; its stream is the original, uncentred one, so a
     centred target would ask it to learn the folding instead of the function.
  2. Symmetrising with A*: write the teacher's block output as t = T + R, with T the part that has the student's
     symmetry (T(mirror x) = M[T(x)], M = align then A*) and R the rest. Averaged over an image and its mirror, the
     loss against t equals the loss against T plus a constant plus a cross term proportional to
     <s(x) - M[s(mirror x)], R(x)>. For an exactly equivariant student s the cross term is zero, so for V-exact
     symmetrising only removes a part it cannot fit. For the SVD student the cross term is not zero: with symmetrised
     targets it is pulled towards A*-equivariance in every block and asked to discard the teacher's own asymmetric
     part. That would put the method under test into the baseline's targets and could handicap it.
  3. The final flip average has no such issue: it is the mirror-invariant part of the teacher's output, a target any
     student can use, and it is the one change in V-exact's targets that could plausibly make the teacher signal
     better (two views per image). Giving it to SVD as well removes that possible advantage from V-exact.

The comparison uses the better of 3a and 3b (higher healed k-NN on validation). Choosing by the test metric favours
the baseline; this is deliberate.

## 6. Primary metric

k-NN top-1 with the v2 protocol of PREREG_E2.md: CLS token after the student's own final norm (for V-exact: RMSNorm,
even head, affine), features in bf16 for every arm alike, k = 20, T = 0.07, bank of 100 ImageNet train images per
class (100,000), queries all 50,000 validation images. Value: `knn.json`, `healed.top1`, written by each run.
V-exact (arm 1) is the seed 0 RTX 5090 run, 78.34, since E3 runs on an RTX 5090; the A100 run (78.31) is reported
next to it.

## 7. Checks and bars (proposed: the E2 bars, unchanged, to avoid tuning)

Let V0 and V1 be the healed k-NN of V-exact seed 0 (arm 1) and seed 1 (arm 2), R that of arm 4, S = max(3a, 3b) that of
the better SVD arm, and s = |V0 - V1| the seed difference.

| check | statement | bar |
|---|---|---|
| E3.a | V0 - R | >= 2.0 k-NN points |
| E3.b | V0 - S | >= 3.0 k-NN points |
| seed condition | s is reported; a check counts as passed only if its difference is also larger than s | difference > s |
| gate G5 | E3.a and E3.b both pass (with the seed condition) | both |

- If G5 passes: at full conversion, healing does not close the gap to a matched structure-free baseline, and the k = 12
  headline can be credited to the mirror structure.
- If G5 fails: report as is, stop, and discuss the framing with Dawar. No new arms, rates, seeds or budgets inside E3.
- Also reported, without changing the verdict: both differences with V1 in place of V0, and with the mean of V0 and
  V1; if a verdict differs between seeds, the report says so. For E3.b, the bar means S must be at most 75.34 when
  V0 = 78.34.
- Only V-exact has two seeds; the baselines have one each, so s is the only seed-noise estimate and is assumed to
  apply to them too.

Validity conditions (fixed now; an arm that fails one is invalid, the bug is fixed and only that arm is rerun, and the
report lists it as a deviation): every run completes 780 of 780 steps without divergence; arms 2 and 4 pass the
exactness check (tolerance 1e-4) and the fp32 flip check (100%); the SVD arms' numerical ranks equal their target
ranks; every block is matched within 1% (Section 4).

## 8. Secondary learning-rate check (pre-registered; never changes a chosen rate)

In the M3 pilot every arm chose 1e-4, the largest rate of the grid, and so did V-exact at k = 12, so healing may be
under-tuned. The primary grid stays {1e-5, 5e-5, 1e-4} (V-exact seed 0 stays comparable and nothing already known is
re-tuned). In addition:

- every E3 arm runs one more 100-step selection trial at 2e-4 after its grid, with the same seed, data order,
  augmentation, budget and 5,000 held-out train images (`lr_selection.extra_lrs`, `mirror.heal.probe_extra_lrs`);
  the result goes to `lr_extra.json` in the run folder and the chosen rate is not changed;
- V-exact k = 12 seed 0 (already healed) gets the same check with `--lr-probe-only`: the grid trials are rerun next to
  2e-4 on the E3 GPU (so the comparison is within one run), and the rerun grid is compared with the grid recorded in
  `runs/m3_vexact_5090/k12/lr_selection/lr_selection.json`; output `runs/e3/lr_probe_vexact_s0/k12/lr_extra.json`;
- 2e-4 "would have won" for an arm if its held-out loss is below that of every grid rate (ties go to the grid);
- if 2e-4 would have won for any arm (including V-exact seed 0), the E3 report says so and flags the main comparison
  as possibly under-tuned. The full budget is not rerun at 2e-4 inside E3.

## 9. Secondary measures (reported, not gated)

- ImageNet linear probe (Tier 2, `scripts/tier2_linear.py`, as for the other healed models) of each E3 student; for
  V-exact seed 0, the k = 12 probe from `jobs/stage22_korea.txt` (`runs/t2`). Original: 80.99.
- Flip consistency: the bf16 k-NN flip consistency of every arm (`knn.json`); the fp32 flip check of arms 2 and 4
  (must be 100%).
- Zero-shot k-NN of every arm (`knn.json`, `zero_shot`).
- Parameter counts: kept matrix parameters per block and in total (Section 4), whole-model counts, trainable entries.
- Held-out distillation loss before and after healing (comparable only between arms with the same targets).
- Time per run and the 2e-4 results of Section 8.

## 10. Compute

`jobs/e3_k12.txt` (scheduler `scripts/vm_sched.sh`): one line chains the four healing runs and the seed 0 learning-rate
probe, so only one healing job runs at a time; four Tier-2 lines start as each run finishes. Estimate on an RTX 5090,
from the V-exact k = 12 run (4,703 s in all: about 1,405 s for the three selection trials, 2,471 s for the full run,
722 s k-NN, 100 s checks): SVD raw about 4,200 s (the teacher sees each batch once), SVD with V-exact targets about
5,100 s, V-exact seed 1 and the random control about 5,200 s each (including the 2e-4 trial, about 470 s), the seed 0
probe about 1,900 s. About 6.0 GPU-hours for the healing line and 0.4 for Tier 2: **about 6.4 GPU-hours.**

## 11. Fixed choices and known differences

- Hardware: arm 1 was healed on another RTX 5090 (and on the A100: 78.31 against 78.34); E3 runs on the RTX 5090 in
  South Korea, whose storage was copied from the first one. Same code path, data and cached original score.
- Recipe: every E3 arm uses the V-exact recipe (embeddings train; trainable set as in Section 3), not the pilot's
  (converted blocks only). The SVD arms write the 224-px position grid into the model before training, as V-exact
  does; this does not change the function at 224 px.
- Code: the E3 options (`structure.mode span`, `student.*`, `targets.final`, `structure.random_span`,
  `lr_selection.extra_lrs`, `--lr-probe-only`) are off by default; with them off, tiny CPU runs of
  `scripts/heal.py` and `scripts/heal_vexact.py` give outputs identical to the code before E3. Tests:
  `tests/test_e3.py`.
- Every output records the config, seed and git hash.

## 12. Points for Dawar to decide before signing

1. Seed condition: as written, a check needs its difference to exceed s. A stricter reading is that the margin over
   the bar must exceed s (difference >= bar + s). Choose one.
2. The overall budget is close to the pilot's (V-exact keeps 0.765 of the model's parameters at k = 12; the pilot's
   V-local k = 9 kept 42,097,128 of 63,700,992 matrix parameters in blocks 1 to 9), where healed SVD reached 77.57.
   E3.b passes only if healed SVD ends at 75.34 or lower. Whether this is a fair test of the claim, or E3 should
   report the difference without a bar, is a framing decision.
3. Gate name (G5) and whether E3 replaces G2 in the paper or sits next to it.
