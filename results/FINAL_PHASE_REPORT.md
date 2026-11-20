# FINAL_PHASE_REPORT: pre-registered rules of the final phase, scored

Written by `scripts/collect_final_phase.py` (`mirror/final/collect.py`) on 2026-11-20 17:54 UTC, code commit `6299d2c838` (uncommitted changes). Every number below is read from a result file written by a script of this repository and is listed with that file and the commit recorded in the run's meta.json. Nothing here is typed by hand; rerun the collector when new results arrive.

Pre-registration: `PREREG_FINAL_PHASE.md`, commit `fd83c955d7`, sha256 of the current text `e3dfa5b50aa3`: unchanged since the recorded pre-registration commit.

Scored rules: 23; PASS 13, FAIL 5, PENDING 2, NOT RUN 3. Report-only rules: 5 (3 reported).

Main Mirror arm: `main/dinov2_vitb14_reg/mirror_exact_astar` (from results/A01/decision.json ("adopted": false)).

Jobs not run (runner state `logs/final/state`): a10_fullbank (skipped by hand (.skip file)); a12_bench_deit (not started: budget governor (13.5 VM-hours)); a12b_timing (not started: budget governor (13.5 VM-hours)); a15a_siglip (cut by the Section 9 cut order (rank 1)); a15b_measure (cut by the Section 9 cut order (rank 1)); heal_a11_mirror_exact (cut by the Section 9 cut order (rank 4)); heal_a15b (cut by the Section 9 cut order (rank 1)); heal_l_s0 (cut by the Section 9 cut order (rank 2)).

## Rules

PENDING means an input is missing or incomplete (the reason is given). NOT RUN means an input was cut by the Section 9 order or the budget governor; it is never counted as passed. Failed rules are failures and are reported as such.

| ID | Rule | Value | Bar | n | Verdict | Sources (commit) |
|---|---|---|---|---|---|---|
| Gate P | The new pipeline reproduces MUST-09 | Mirror (current A*) 78.352; SVD-LLM 78.083 | Mirror (current A*) and SVD-LLM means within ±0.15 of 78.34 and 78.06. Otherwise stop and report | 3 + 3 seeds | **PASS** | `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/eval/step780.json` (21f6ef2682) and 10 more |
| A01 | Adopt A*_w as the main operator | zero-shot 7.19; healed A*_w 76.910 vs A* 78.358 (difference -1.448) | Zero-shot k-NN ≥ 58.84 (current 53.84 + 5) **and** healed mean (2 seeds) ≥ the current operator's mean in the same pipeline + 0.15. Otherwise keep the current A* and report A*_w as an ablation | zero-shot 1 run; 2 + 2 healed seeds | **FAIL** | `runs/final/main/dinov2_vitb14_reg/mirror_exact_wspace/s0/meta.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_wspace/s0/eval/step0.json` (21f6ef2682) and 8 more |
| A02-1 | d− stable across ImageNet fits | 0.0147 | Within ±15% of fit set 0 | 3 of 3 fits against fit 0 | **PASS** | `results/A02/metrics.json` (21f6ef2682) |
| A02-2 | Odd subspaces agree | 0.9990 | Energy-weighted mean cos² of principal angles ≥ 0.8 for every pair of ImageNet fits | 6 of 6 pairs | **PASS** | `results/A02/metrics.json` (21f6ef2682) |
| A02-3 | Twins agree | 0.9533 | ≥ 70% of strong pairs in blocks 1 to 6 shared between any two ImageNet fits | 6 of 6 pairs | **PASS** | `results/A02/metrics.json` (21f6ef2682) |
| A02-4 | Head maps agree | 1.00 | ≥ 90% of head assignments (self or swap) identical across fits | 6 of 6 pairs | **PASS** | `results/A02/metrics.json` (21f6ef2682) |
| A02-5 | Healed accuracy is stable across fits | 0.0220 | Spread over ImageNet fit sets ≤ 0.2 k-NN | 3 heals (seed 0) | **PASS** | `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/eval/step780.json` (21f6ef2682) and 4 more |
| A02-6 | The fit domain does not matter | -0.0020 | COCO-fit healed k-NN within 0.3 of the ImageNet-fit mean | 1 COCO heal against 3 | **PASS** | `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/eval/step780.json` (21f6ef2682) and 6 more |
| A03-1 | Controls C1 to C3 find no structure | C1: twins 0.0036, split-half 0.9765; C2: twins 2.17e-04, split-half 1.03; C3: twins 0.0794, split-half 0.9218 | Twin fraction in blocks 1 to 6 ≤ 0.05 **and** ratio to null ≥ 0.9 | 3 controls | **FAIL** | `results/A03/metrics.json` (21f6ef2682) |
| A03-2 | The real pairing matters after healing | -1.34 | True operator + random pairing ≤ Mirror − 3.0 k-NN | 1 heal against the 3-seed mean | **FAIL** | `runs/final/main/dinov2_vitb14_reg/pair_random_h1/s0/metrics.json` (87f1d9b1e6); `runs/final/main/dinov2_vitb14_reg/pair_random_h1/s0/eval/step780.json` (87f1d9b1e6) and 6 more |
| A04-1 | It is not just distillation | width-pruned (flap) +0.06; block-dropped (blockdrop) -3.10 | Both dense students ≤ Mirror − 0.5 k-NN | 2 students (seed 0) against the 3-seed mean | **FAIL** | `runs/final/main/dinov2_vitb14_reg/flap/s0/metrics.json` (11e50a1526); `runs/final/main/dinov2_vitb14_reg/flap/s0/eval/step780.json` (11e50a1526) and 8 more |
| A04-2 | Healing without surgery | 78.95 | Report (it defines the ceiling of the recipe) | 1 heal | **REPORTED** | `runs/final/main/dinov2_vitb14_reg/original/s0/metrics.json` (11e50a1526); `runs/final/main/dinov2_vitb14_reg/original/s0/eval/step780.json` (11e50a1526) |
| A05-1 | Non-inferiority to every baseline | Mirror - svdllm +0.271 (point estimate), 95% interval [+0.081, +0.465]; superiority claimed (lower end > 0) | Lower 95% bound of (Mirror − best baseline) > −0.3 k-NN. Superiority is claimed only if the lower bound is > 0 | Mirror 5 runs over 3 fit sets; svdllm 3 seeds; 50,000 images; 10,000 resamples | **PASS** | `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/eval/step780.json` (21f6ef2682) and 32 more |
| A06-1 | Stacking beats low rank alone | 0.65: +1.64; 0.50: +2.75; 0.40: +4.13; interval at 0.50 [+2.481, +3.005] (PASS) | Mirror + SVD ≥ SVD-LLM + 0.5 at ≥ 2 of 3 budgets, with the interval above 0 at 0.50 | 3 of 3 budgets scored; at least 2 must pass (PASS) | **PASS** | `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f065/s0/metrics.json` (579ada6c2a); `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f065/s0/eval/step780.json` (c7c76556a8) and 14 more |
| A06-2 | The count formula predicts the stacking gain | 0.65: measured +1.64 vs predicted +1.26; 0.50: measured +2.75 vs predicted +2.51; 0.40: measured +4.13 vs predicted +5.96 | Measured gain within 50% of the pre-written prediction at ≥ 2 of 3 budgets | 3 of 3 budgets scored; at least 2 must pass | **PASS** | `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f065/s0/metrics.json` (579ada6c2a); `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f065/s0/eval/step780.json` (c7c76556a8) and 28 more |
| A06-3 | Price of exactness | 0.4280 | Report the k-NN difference between the 9-block and the full conversion | 1 heal against the 3-seed mean | **REPORTED** | `runs/final/a06/dinov2_vitb14_reg/mirror_partial_k9/s0/metrics.json` (579ada6c2a); `runs/final/a06/dinov2_vitb14_reg/mirror_partial_k9/s0/eval/step780.json` (66cebdae87) and 6 more |
| A07 | Built-in invariance beats brute force at equal compute | Mirror 78.352 vs best eligible DINOv2-S/14 reg, flip-averaged 74.594 (difference +3.758) | Mirror k-NN ≥ the best flip-averaged model at ≤ equal measured compute + 0.5 | 5 candidates, 1 eligible with a value | **PASS** | `runs/final/a06/dinov2_vitb14_reg/svdllm_f065/s0/metrics.json` (21f6ef2682); `runs/final/a06/dinov2_vitb14_reg/svdllm_f065/s0/eval/step780.json` (547224c624) and 13 more |
| A08-1 | DeiT-III-B | top-1 Mirror 82.695 vs SVD-LLM 82.883 (difference -0.187); flip: PASS, PASS, PASS; parameters: largest \|stored - formula\| 0 | Mirror top-1 ≥ SVD-LLM − 0.5 (mean of 3); 100% float32 flip consistency; stored parameters equal the formula | 3 + 3 seeds | **PASS** | `runs/final/a08/deit3_b16/mirror_exact/s0/eval/top1_step780.json` (66cebdae87); `runs/final/a08/deit3_b16/mirror_exact/s1/eval/top1_step780.json` (66cebdae87) and 11 more |
| A08-2 | DINOv2-L (indicative, 1 seed) | Mirror n/a vs SVD-LLM n/a (difference n/a); fp32 flip: runs/final/a08/dinov2_vitl14_reg/mirror_exact/s0: cut by the Section 9 cut order (rank 2) (job heal_l_s0) | Mirror k-NN ≥ SVD-LLM − 0.5; 100% float32 flip consistency | 1 + 1 heals (indicative) | **NOT RUN** | n/a |
| A09-1 | Exact in float64 | n/a | Maximum relative error ≤ 1e-10 at every tensor | n/a | **PENDING** | n/a |
| A09-2 | Numerically exact in float32 | n/a | Equivariance error ≤ 2x the model's own float32-vs-float64 deviation at every tensor; 100% flip consistency for both forms | n/a | **PENDING** | n/a |
| A09-3 | bfloat16 | n/a | Report for both forms (target: 100% for the even/odd form) | n/a | **PENDING** | n/a |
| A10 | Standard metrics | n/a | Mirror within 1.0 of the original on full-bank k-NN and on the linear protocol | n/a | **NOT RUN** | n/a |
| A11 | Healing ceiling | n/a | Report. Target: Mirror within 0.2 of the original | 1 heal per arm | **REPORTED** | `runs/final/a11/dinov2_vitb14_reg/svdllm_ext/s0/metrics.json` (87f1d9b1e6); `runs/final/a11/dinov2_vitb14_reg/svdllm_ext/s0/eval/step2340.json` (87f1d9b1e6) |
| A12b | Fused kernel | batch 64: 0.804x; batch 256: 0.833x | ≥ 1.1x the original's throughput at batch 64 and 256 (bf16, compiled) earns a speed claim; otherwise a limitation | 2 batch sizes | **FAIL** | `results/A12b/metrics.json` (6299d2c838) |
| A15a | SigLIP 2 | n/a | Report the zero-shot k-NN with whitened projection against the current collapse | n/a | **NOT RUN** | n/a |
| A15b | Cause probe | n/a | Rotation ratio to null in blocks 1 to 6 falls by ≥ 0.1 after the fine-tune (from about 0.82); report either way | n/a | **NOT RUN** | n/a |
| P11.2 | P11.2 (MUST-11, previous phase, re-measured in A10): Mirror-exact with the symmetrised head gives 100% mask equivariance; the original model stays below 100% | Mirror-exact (symmetrised head) 99.99991%: 99.9999% raw, 415 disagreement(s), all near-ties in both copies (largest relative margin 9.98e-06); original 95.0309%, 94.7413% | Mirror-exact 100% (every differing pixel a near-tie in both copies, PREREG_NEXT_PHASE.md Section 1); original < 100% | 487,666,546 pixels | **PASS** | `runs/final/A10/p112/t2_ade20k.csv` (21f6ef2682); `runs/final/A10/p112/seg/dinov2_vitb14_reg/vexact_s0_maskeq.json` (not recorded) |

## Scoring details

### Gate P: PASS

- Rule: The new pipeline reproduces MUST-09.
- Bar: Mirror (current A*) and SVD-LLM means within ±0.15 of 78.34 and 78.06. Otherwise stop and report.
- Scored quantity: mean over healing seeds 0, 1, 2 of the final bf16 k-NN (50,000 queries), Mirror with the current A* and SVD-LLM, against 78.34 and 78.06.
- Value: Mirror (current A*) 78.352; SVD-LLM 78.083. n: 3 + 3 seeds.
- Sources: `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/eval/step780.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s1/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s1/eval/step780.json` (7d05b6439c); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s2/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s2/eval/step780.json` (7d05b6439c); `runs/final/main/dinov2_vitb14_reg/svdllm/s0/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/svdllm/s0/eval/step780.json` (21f6ef2682) and 4 more.
- Notes: mirror_exact_astar: mean 78.352 against 78.34 +- 0.15 (PASS); svdllm: mean 78.083 against 78.06 +- 0.15 (PASS).

| arm | seed | knn_top1 | mean | target | verdict |
|---|---|---|---|---|---|
| mirror_exact_astar | 0 | 78.35 | 78.35 | 78.34 | PASS |
| mirror_exact_astar | 1 | 78.36 | 78.35 | 78.34 | PASS |
| mirror_exact_astar | 2 | 78.34 | 78.35 | 78.34 | PASS |
| svdllm | 0 | 78.08 | 78.08 | 78.06 | PASS |
| svdllm | 1 | 78.07 | 78.08 | 78.06 | PASS |
| svdllm | 2 | 78.10 | 78.08 | 78.06 | PASS |

### A01: FAIL

- Rule: Adopt A*_w as the main operator.
- Bar: Zero-shot k-NN ≥ 58.84 (current 53.84 + 5) **and** healed mean (2 seeds) ≥ the current operator's mean in the same pipeline + 0.15. Otherwise keep the current A* and report A*_w as an ablation.
- Scored quantity: zero-shot k-NN of the A*_w full conversion (step 0 of its heal) >= 58.84, and the mean final k-NN of A*_w over healing seeds 0 and 1 >= the current A*'s mean over the same seeds + 0.15.
- Value: zero-shot 7.19; healed A*_w 76.910 vs A* 78.358 (difference -1.448). n: zero-shot 1 run; 2 + 2 healed seeds.
- Sources: `runs/final/main/dinov2_vitb14_reg/mirror_exact_wspace/s0/meta.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_wspace/s0/eval/step0.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_wspace/s0/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_wspace/s0/eval/step780.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_wspace/s1/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_wspace/s1/eval/step780.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/eval/step780.json` (21f6ef2682) and 2 more.
- Notes: zero-shot 7.19 (FAIL); healed A*_w minus A* -1.448 (FAIL); adopted: no (main arm mirror_exact_astar); main arm used for the later rules: mirror_exact_astar, from results/A01/decision.json ("adopted": false).

| part | value | bar | verdict | d_plus | d_minus | zero_shot_fit_run | weighted_agreement |
|---|---|---|---|---|---|---|---|
| zero-shot (step 0) | 7.19 | >= 58.84 | FAIL | n/a | n/a | n/a | n/a |
| healed A*_w minus A* (seeds 0 and 1) | -1.45 | >= 0.15 | FAIL | n/a | n/a | n/a | n/a |
| operator current | n/a | n/a | n/a | 700 | 68 | n/a | 1.00 |
| operator wspace_matched | n/a | n/a | n/a | 700 | 68 | n/a | n/a |

### A02-1: PASS

- Rule: d− stable across ImageNet fits.
- Bar: Within ±15% of fit set 0.
- Scored quantity: |d-(fit k) - d-(fit 0)| / d-(fit 0) of the full-model operator A*_12, k = 1, 2, 3.
- Value: 0.0147. n: 3 of 3 fits against fit 0.
- Sources: `results/A02/metrics.json` (21f6ef2682).

| fit | d_minus | d_minus_fit0 | rel_change | verdict |
|---|---|---|---|---|
| fit1 | 69.00 | 68.00 | 0.0147 | PASS |
| fit2 | 68.00 | 68.00 | 0 | PASS |
| fit3 | 68.00 | 68.00 | 0 | PASS |

### A02-2: PASS

- Rule: Odd subspaces agree.
- Bar: Energy-weighted mean cos² of principal angles ≥ 0.8 for every pair of ImageNet fits.
- Scored quantity: energy-weighted odd-subspace agreement (rho(a->b) + rho(b->a)) / 2, every pair of ImageNet fits 0 to 3.
- Value: 0.9990. n: 6 of 6 pairs.
- Sources: `results/A02/metrics.json` (21f6ef2682).

| pair | weighted_agreement | verdict | note |
|---|---|---|---|
| fit0-fit1 | 0.9990 | PASS |  |
| fit0-fit2 | 0.9990 | PASS |  |
| fit0-fit3 | 0.9990 | PASS |  |
| fit1-fit2 | 0.9990 | PASS |  |
| fit1-fit3 | 0.9990 | PASS |  |
| fit2-fit3 | 0.9990 | PASS |  |

### A02-3: PASS

- Rule: Twins agree.
- Bar: ≥ 70% of strong pairs in blocks 1 to 6 shared between any two ImageNet fits.
- Scored quantity: share of strong pairs (blocks 1 to 6) shared, mean of both directions, every pair of ImageNet fits.
- Value: 0.9533. n: 6 of 6 pairs.
- Sources: `results/A02/metrics.json` (21f6ef2682).

| pair | twin_mean | verdict | note |
|---|---|---|---|
| fit0-fit1 | 0.9562 | PASS |  |
| fit0-fit2 | 0.9533 | PASS |  |
| fit0-fit3 | 0.9697 | PASS |  |
| fit1-fit2 | 0.9552 | PASS |  |
| fit1-fit3 | 0.9608 | PASS |  |
| fit2-fit3 | 0.9649 | PASS |  |

### A02-4: PASS

- Rule: Head maps agree.
- Bar: ≥ 90% of head assignments (self or swap) identical across fits.
- Scored quantity: share of heads with the same assignment (self or swap), every pair of ImageNet fits.
- Value: 1.00. n: 6 of 6 pairs.
- Sources: `results/A02/metrics.json` (21f6ef2682).

| pair | head_share_same | verdict | note |
|---|---|---|---|
| fit0-fit1 | 1.00 | PASS |  |
| fit0-fit2 | 1.00 | PASS |  |
| fit0-fit3 | 1.00 | PASS |  |
| fit1-fit2 | 1.00 | PASS |  |
| fit1-fit3 | 1.00 | PASS |  |
| fit2-fit3 | 1.00 | PASS |  |

### A02-5: PASS

- Rule: Healed accuracy is stable across fits.
- Bar: Spread over ImageNet fit sets ≤ 0.2 k-NN.
- Scored quantity: max - min of the final k-NN of Mirror-exact (adopted operator) healed with seed 0 from fit sets 0, 1 and 2.
- Value: 0.0220. n: 3 heals (seed 0).
- Sources: `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/eval/step780.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_fit1/s0/metrics.json` (87f1d9b1e6); `runs/final/main/dinov2_vitb14_reg/mirror_exact_fit1/s0/eval/step780.json` (87f1d9b1e6); `runs/final/main/dinov2_vitb14_reg/mirror_exact_fit2/s0/metrics.json` (87f1d9b1e6); `runs/final/main/dinov2_vitb14_reg/mirror_exact_fit2/s0/eval/step780.json` (87f1d9b1e6).

| fit_set | knn_top1 | structure | note |
|---|---|---|---|
| fit0 | 78.35 | results/MUST-07/structure/dinov2_vitb14_reg_s0.npz |  |
| fit1 | 78.34 | {'adopted': {'wspace': 'results/A02/wspace_fit1/structure/dinov2_vitb14_reg_wspace_matched.npz', 'astar': 'results/A02/fits/dinov2_vitb14_reg_fit1.npz'}} |  |
| fit2 | 78.37 | {'adopted': {'wspace': 'results/A02/wspace_fit2/structure/dinov2_vitb14_reg_wspace_matched.npz', 'astar': 'results/A02/fits/dinov2_vitb14_reg_fit2.npz'}} |  |

### A02-6: PASS

- Rule: The fit domain does not matter.
- Bar: COCO-fit healed k-NN within 0.3 of the ImageNet-fit mean.
- Scored quantity: |COCO-fit heal (seed 0) - mean of the three ImageNet-fit heals of A02-5|.
- Value: -0.0020. n: 1 COCO heal against 3.
- Sources: `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/eval/step780.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_fit1/s0/metrics.json` (87f1d9b1e6); `runs/final/main/dinov2_vitb14_reg/mirror_exact_fit1/s0/eval/step780.json` (87f1d9b1e6); `runs/final/main/dinov2_vitb14_reg/mirror_exact_fit2/s0/metrics.json` (87f1d9b1e6); `runs/final/main/dinov2_vitb14_reg/mirror_exact_fit2/s0/eval/step780.json` (87f1d9b1e6); `runs/final/main/dinov2_vitb14_reg/mirror_exact_coco/s0/metrics.json` (87f1d9b1e6); `runs/final/main/dinov2_vitb14_reg/mirror_exact_coco/s0/eval/step780.json` (87f1d9b1e6).

| fit_set | knn_top1 | structure | note |
|---|---|---|---|
| fit0 | 78.35 | results/MUST-07/structure/dinov2_vitb14_reg_s0.npz |  |
| fit1 | 78.34 | {'adopted': {'wspace': 'results/A02/wspace_fit1/structure/dinov2_vitb14_reg_wspace_matched.npz', 'astar': 'results/A02/fits/dinov2_vitb14_reg_fit1.npz'}} |  |
| fit2 | 78.37 | {'adopted': {'wspace': 'results/A02/wspace_fit2/structure/dinov2_vitb14_reg_wspace_matched.npz', 'astar': 'results/A02/fits/dinov2_vitb14_reg_fit2.npz'}} |  |
| coco | 78.35 | {'adopted': {'wspace': 'results/A02/wspace_coco/structure/dinov2_vitb14_reg_wspace_matched.npz', 'astar': 'results/A02/fits/dinov2_vitb14_reg_coco.npz'}} |  |

### A03-1: FAIL

- Rule: Controls C1 to C3 find no structure.
- Bar: Twin fraction in blocks 1 to 6 ≤ 0.05 **and** ratio to null ≥ 0.9.
- Scored quantity: for each of C1, C2 and C3: twin fraction in blocks 1 to 6 (MUST-04 statistic) <= 0.05 and split-half ratio to null (blocks 1 to 6, 20 null draws) >= 0.9.
- Value: C1: twins 0.0036, split-half 0.9765; C2: twins 2.17e-04, split-half 1.03; C3: twins 0.0794, split-half 0.9218. n: 3 controls.
- Sources: `results/A03/metrics.json` (21f6ef2682).

| control | twin_fraction_1_6 | ratio_in_sample | ratio_split_half | verdict |
|---|---|---|---|---|
| mirror | 0.5677 | 0.4791 | 0.4795 | reference (not scored) |
| C1 | 0.0036 | 0.9700 | 0.9765 | PASS |
| C2 | 2.17e-04 | 1.04 | 1.03 | PASS |
| C3 | 0.0794 | 0.9209 | 0.9218 | FAIL |

### A03-2: FAIL

- Rule: The real pairing matters after healing.
- Bar: True operator + random pairing ≤ Mirror − 3.0 k-NN.
- Scored quantity: final k-NN of H1 (main operator, random pairing, healing seed 0) <= the main Mirror arm's 3-seed mean - 3.0.
- Value: -1.34. n: 1 heal against the 3-seed mean.
- Sources: `runs/final/main/dinov2_vitb14_reg/pair_random_h1/s0/metrics.json` (87f1d9b1e6); `runs/final/main/dinov2_vitb14_reg/pair_random_h1/s0/eval/step780.json` (87f1d9b1e6); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/eval/step780.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s1/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s1/eval/step780.json` (7d05b6439c); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s2/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s2/eval/step780.json` (7d05b6439c).

| arm | knn_top1 |
|---|---|
| H1 | 77.01 |
| main Mirror (mean) | 78.35 |

### A04-1: FAIL

- Rule: It is not just distillation.
- Bar: Both dense students ≤ Mirror − 0.5 k-NN.
- Scored quantity: final k-NN of the width-pruned (FLAP-style) and the block-dropped dense students, seed 0, each <= the main Mirror arm's 3-seed mean - 0.5.
- Value: width-pruned (flap) +0.06; block-dropped (blockdrop) -3.10. n: 2 students (seed 0) against the 3-seed mean.
- Sources: `runs/final/main/dinov2_vitb14_reg/flap/s0/metrics.json` (11e50a1526); `runs/final/main/dinov2_vitb14_reg/flap/s0/eval/step780.json` (11e50a1526); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/eval/step780.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s1/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s1/eval/step780.json` (7d05b6439c); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s2/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s2/eval/step780.json` (7d05b6439c) and 2 more.

| student | knn_top1 | minus_mirror | params_frac | verdict | note |
|---|---|---|---|---|---|
| width-pruned (flap) | 78.41 | 0.0560 | 0.7646 | FAIL |  |
| block-dropped (blockdrop) | 75.25 | -3.10 | 0.7544 | PASS |  |

### A04-2: REPORTED

- Rule: Healing without surgery.
- Bar: Report (it defines the ceiling of the recipe).
- Scored quantity: final k-NN of the original distilled with the same recipe (healing without surgery), seed 0.
- Value: 78.95. n: 1 heal.
- Sources: `runs/final/main/dinov2_vitb14_reg/original/s0/metrics.json` (11e50a1526); `runs/final/main/dinov2_vitb14_reg/original/s0/eval/step780.json` (11e50a1526).
- Notes: distilled minus released original +0.01.

| row | knn_top1 |
|---|---|
| original distilled | 78.95 |
| original (released, same evaluation) | 78.94 |
| main Mirror arm (mean) | 78.35 |

### A05-1: PASS

- Rule: Non-inferiority to every baseline.
- Bar: Lower 95% bound of (Mirror − best baseline) > −0.3 k-NN. Superiority is claimed only if the lower bound is > 0.
- Scored quantity: lower end of the 95% interval of (Mirror - best baseline): A13 hierarchical paired bootstrap; Mirror = main arm (fit set 0, seeds 0 to 2) and the A02 heals from fit sets 1 and 2; best baseline = highest mean healed k-NN among SVD-LLM, full SVD-LLM, IO-SVD, SliceGPT and plain SVD.
- Value: Mirror - svdllm +0.271 (point estimate), 95% interval [+0.081, +0.465]; superiority claimed (lower end > 0). n: Mirror 5 runs over 3 fit sets; svdllm 3 seeds; 50,000 images; 10,000 resamples.
- Sources: `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/eval/step780.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s1/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s1/eval/step780.json` (7d05b6439c); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s2/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s2/eval/step780.json` (7d05b6439c); `runs/final/main/dinov2_vitb14_reg/mirror_exact_fit1/s0/metrics.json` (87f1d9b1e6); `runs/final/main/dinov2_vitb14_reg/mirror_exact_fit1/s0/eval/step780.json` (87f1d9b1e6) and 26 more.
- Notes: bootstrap (run-weighted) mean difference +0.271, effect size 21.95.

| method | seed | knn_top1 | state | note |
|---|---|---|---|---|
| Mirror (astar) fit set 0 | 0 | 78.35 | ok | results/MUST-07/structure/dinov2_vitb14_reg_s0.npz |
| Mirror (astar) fit set 0 | 1 | 78.36 | ok | results/MUST-07/structure/dinov2_vitb14_reg_s0.npz |
| Mirror (astar) fit set 0 | 2 | 78.34 | ok | results/MUST-07/structure/dinov2_vitb14_reg_s0.npz |
| Mirror (astar) fit set 1 | 0 | 78.34 | ok | {'adopted': {'wspace': 'results/A02/wspace_fit1/structure/dinov2_vitb14_reg_wspace_matched.npz', 'astar': 'results/A02/fits/dinov2_vitb14_reg_fit1.npz'}} |
| Mirror (astar) fit set 2 | 0 | 78.37 | ok | {'adopted': {'wspace': 'results/A02/wspace_fit2/structure/dinov2_vitb14_reg_wspace_matched.npz', 'astar': 'results/A02/fits/dinov2_vitb14_reg_fit2.npz'}} |
| svdllm | 0 | 78.08 | ok |  |
| svdllm | 1 | 78.07 | ok |  |
| svdllm | 2 | 78.10 | ok |  |
| svdllm_full | 0 | 76.45 | ok |  |
| svdllm_full | 1 | 76.76 | ok |  |
| svdllm_full | 2 | 76.06 | ok |  |
| iosvd | 0 | 0.5440 | ok |  |
| iosvd | 1 | 0.4000 | ok |  |
| iosvd | 2 | 0.6580 | ok |  |
| slicegpt | 0 | 77.86 | ok |  |
| slicegpt | 1 | 77.85 | ok |  |
| svd | 0 | 77.00 | ok |  |

### A06-1: PASS

- Rule: Stacking beats low rank alone.
- Bar: Mirror + SVD ≥ SVD-LLM + 0.5 at ≥ 2 of 3 budgets, with the interval above 0 at 0.50.
- Scored quantity: Mirror + SVD-LLM minus SVD-LLM at 0.65, 0.50 and 0.40 of the full size (seed 0; mean of seeds 0 and 1 at 0.50) >= 0.5 at 2 or more budgets, and the 95% interval at 0.50 (A13 bootstrap over images and seeds 0 and 1) above 0.
- Value: 0.65: +1.64; 0.50: +2.75; 0.40: +4.13; interval at 0.50 [+2.481, +3.005] (PASS). n: 3 of 3 budgets scored; at least 2 must pass (PASS).
- Sources: `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f065/s0/metrics.json` (579ada6c2a); `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f065/s0/eval/step780.json` (c7c76556a8); `runs/final/a06/dinov2_vitb14_reg/svdllm_f065/s0/metrics.json` (21f6ef2682); `runs/final/a06/dinov2_vitb14_reg/svdllm_f065/s0/eval/step780.json` (547224c624); `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f050/s0/metrics.json` (579ada6c2a); `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f050/s0/eval/step780.json` (66cebdae87); `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f050/s1/metrics.json` (7d05b6439c); `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f050/s1/eval/step780.json` (7d05b6439c) and 8 more.

| budget | mirror_svdllm | svdllm | difference | verdict | note |
|---|---|---|---|---|---|
| 0.65 | 78.20 | 76.57 | 1.64 | PASS |  |
| 0.50 | 76.26 | 73.51 | 2.75 | PASS |  |
| 0.40 | 71.60 | 67.47 | 4.13 | PASS |  |

### A06-2: PASS

- Rule: The count formula predicts the stacking gain.
- Bar: Measured gain within 50% of the pre-written prediction at ≥ 2 of 3 budgets.
- Scored quantity: measured gain acc(Mirror + SVD, B) - acc(SVD-LLM, B) within 50% of the pre-registered prediction acc_pred(B) - acc_SVD(B) (mirror.final.baselines.sizes) at 2 or more budgets.
- Value: 0.65: measured +1.64 vs predicted +1.26; 0.50: measured +2.75 vs predicted +2.51; 0.40: measured +4.13 vs predicted +5.96. n: 3 of 3 budgets scored; at least 2 must pass.
- Sources: `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f065/s0/metrics.json` (579ada6c2a); `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f065/s0/eval/step780.json` (c7c76556a8); `runs/final/a06/dinov2_vitb14_reg/svdllm_f065/s0/metrics.json` (21f6ef2682); `runs/final/a06/dinov2_vitb14_reg/svdllm_f065/s0/eval/step780.json` (547224c624); `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f050/s0/metrics.json` (579ada6c2a); `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f050/s0/eval/step780.json` (66cebdae87); `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f050/s1/metrics.json` (7d05b6439c); `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f050/s1/eval/step780.json` (7d05b6439c) and 22 more.

| budget | predicted_gain | measured_gain | abs_error | allowed | verdict | note |
|---|---|---|---|---|---|---|
| 0.65 | 1.26 | 1.64 | 0.3782 | 0.6289 | PASS | n/a |
| 0.50 | 2.51 | 2.75 | 0.2347 | 1.26 | PASS | n/a |
| 0.40 | 5.96 | 4.13 | 1.84 | 2.98 | PASS | n/a |
| inputs | n/a | n/a | n/a | n/a | n/a | distilled original 78.95, healed Mirror at 0.765 78.35, SVD-LLM at 0.765 78.08 (3-seed mean) |

### A06-3: REPORTED

- Rule: Price of exactness.
- Bar: Report the k-NN difference between the 9-block and the full conversion.
- Scored quantity: final k-NN of Mirror on the first 9 blocks (seed 0) minus the full conversion (main arm).
- Value: 0.4280. n: 1 heal against the 3-seed mean.
- Sources: `runs/final/a06/dinov2_vitb14_reg/mirror_partial_k9/s0/metrics.json` (579ada6c2a); `runs/final/a06/dinov2_vitb14_reg/mirror_partial_k9/s0/eval/step780.json` (66cebdae87); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/eval/step780.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s1/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s1/eval/step780.json` (7d05b6439c); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s2/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s2/eval/step780.json` (7d05b6439c).
- Notes: paired by healing seed 0: +0.428.

| row | knn_top1 |
|---|---|
| 9-block Mirror (seed 0) | 78.78 |
| full conversion (main arm mean) | 78.35 |
| full conversion (main arm seed 0) | 78.35 |

### A07: PASS

- Rule: Built-in invariance beats brute force at equal compute.
- Bar: Mirror k-NN ≥ the best flip-averaged model at ≤ equal measured compute + 0.5.
- Scored quantity: main Mirror arm's 3-seed mean k-NN >= the best k-NN among flip-averaged candidates (A06 SVD-LLM models, DINOv2-S/14 reg, the original) whose measured time per image at batch 256 (two passes) is at most Mirror's faster stored form, + 0.5.
- Value: Mirror 78.352 vs best eligible DINOv2-S/14 reg, flip-averaged 74.594 (difference +3.758). n: 5 candidates, 1 eligible with a value.
- Sources: `runs/final/a06/dinov2_vitb14_reg/svdllm_f065/s0/metrics.json` (21f6ef2682); `runs/final/a06/dinov2_vitb14_reg/svdllm_f065/s0/eval/step780.json` (547224c624); `runs/final/a06/dinov2_vitb14_reg/svdllm_f050/s0/metrics.json` (21f6ef2682); `runs/final/a06/dinov2_vitb14_reg/svdllm_f050/s0/eval/step780.json` (7d05b6439c); `runs/final/a06/dinov2_vitb14_reg/svdllm_f040/s0/metrics.json` (21f6ef2682); `runs/final/a06/dinov2_vitb14_reg/svdllm_f040/s0/eval/step780.json` (7d05b6439c); `runs/final/a07/dinov2_vits14_reg/eval/original.json` (21f6ef2682); `runs/final/a07/dinov2_vitb14_reg/eval/original.json` (21f6ef2682) and 7 more.

| candidate | knn_top1 | ms_per_image_b256 | timing | eligible | note |
|---|---|---|---|---|---|
| Mirror (mirror_exact_astar, mirror_dense) | 78.35 | 0.1219 | measured (faster stored form) | n/a | n/a |
| SVD-LLM 0.65 (seed 0), flip-averaged | 76.68 | 0.2447 | measured (flipavg_svdllm_f065) | no |  |
| SVD-LLM 0.50 (seed 0), flip-averaged | 73.69 | 0.2140 | measured (flipavg_svdllm_f050) | no |  |
| SVD-LLM 0.40 (seed 0), flip-averaged | 67.91 | 0.2056 | measured (flipavg_svdllm_f040) | no |  |
| DINOv2-S/14 reg, flip-averaged | 74.59 | 0.0974 | measured (flipavg_dinov2_vits14_reg) | yes |  |
| original DINOv2-B/14 reg, flip-averaged | 79.00 | 0.2492 | measured (flipavg_original) | no |  |

### A08-1: PASS

- Rule: DeiT-III-B.
- Bar: Mirror top-1 ≥ SVD-LLM − 0.5 (mean of 3); 100% float32 flip consistency; stored parameters equal the formula.
- Scored quantity: DeiT-III-B: mean top-1 (own classifier, 50,000 images, seeds 0 to 2) of Mirror >= SVD-LLM - 0.5; 100% float32 flip consistency (predicted class) of Mirror; stored parameters equal the count formula (0 percentage points).
- Value: top-1 Mirror 82.695 vs SVD-LLM 82.883 (difference -0.187); flip: PASS, PASS, PASS; parameters: largest |stored - formula| 0. n: 3 + 3 seeds.
- Sources: `runs/final/a08/deit3_b16/mirror_exact/s0/eval/top1_step780.json` (66cebdae87); `runs/final/a08/deit3_b16/mirror_exact/s1/eval/top1_step780.json` (66cebdae87); `runs/final/a08/deit3_b16/mirror_exact/s2/eval/top1_step780.json` (7d05b6439c); `results/A08/deit3_b16/params_check.json` (66cebdae87); `runs/final/a08/deit3_b16/mirror_exact/s0/metrics.json` (66cebdae87); `runs/final/a08/deit3_b16/mirror_exact/s1/metrics.json` (66cebdae87); `runs/final/a08/deit3_b16/mirror_exact/s2/metrics.json` (66cebdae87); `runs/final/a08/deit3_b16/svdllm/s0/metrics.json` (66cebdae87) and 5 more.

| part | value | verdict | note |
|---|---|---|---|
| top-1 difference | -0.1873 | PASS |  |
| fp32 flip | n/a | PASS | s0: 100% raw (no disagreement) |
| fp32 flip | n/a | PASS | s1: 100% raw (no disagreement) |
| fp32 flip | n/a | PASS | s2: 100% raw (no disagreement) |
| parameters | n/a | PASS | largest \|stored - formula\| 0 |
| released DeiT-III-B top-1 (reported) | 83.68 |  | runs/final/a08/deit3_b16/_original/eval/top1_original.json |

### A08-2: NOT RUN

- Rule: DINOv2-L (indicative, 1 seed).
- Bar: Mirror k-NN ≥ SVD-LLM − 0.5; 100% float32 flip consistency.
- Scored quantity: DINOv2-L/14 reg, seed 0: Mirror k-NN >= SVD-LLM k-NN - 0.5; 100% float32 flip consistency.
- Value: Mirror n/a vs SVD-LLM n/a (difference n/a); fp32 flip: runs/final/a08/dinov2_vitl14_reg/mirror_exact/s0: cut by the Section 9 cut order (rank 2) (job heal_l_s0). n: 1 + 1 heals (indicative).
- Sources: n/a.
- Notes: runs/final/a08/dinov2_vitl14_reg/mirror_exact/s0: cut by the Section 9 cut order (rank 2) (job heal_l_s0); runs/final/a08/dinov2_vitl14_reg/svdllm/s0: cut by the Section 9 cut order (rank 2) (job heal_l_s0); runs/final/a08/dinov2_vitl14_reg/mirror_exact/s0: cut by the Section 9 cut order (rank 2) (job heal_l_s0).

### A09-1: PENDING

- Rule: Exact in float64.
- Bar: Maximum relative error ≤ 1e-10 at every tensor.
- Scored quantity: largest relative equivariance error at any tensor, float64 (re-projected exact weights), both forms, 1,000 validation images.
- Value: n/a. n: n/a.
- Sources: n/a.
- Notes: results/A09/dinov2_vitb14_reg/mirror_exact_astar_s<seed>/ (the healed main arm's audit) missing.

### A09-2: PENDING

- Rule: Numerically exact in float32.
- Bar: Equivariance error ≤ 2x the model's own float32-vs-float64 deviation at every tensor; 100% flip consistency for both forms.
- Scored quantity: every float32 tensor: equivariance error <= 2 x the model's own float32-versus-float64 deviation (exact floor); and float32 flip consistency 100% for the dense and even/odd forms.
- Value: n/a. n: n/a.
- Sources: n/a.
- Notes: results/A09/dinov2_vitb14_reg/mirror_exact_astar_s<seed>/ (the healed main arm's audit) missing.

### A09-3: PENDING

- Rule: bfloat16.
- Bar: Report for both forms (target: 100% for the even/odd form).
- Scored quantity: bfloat16: largest error and flip consistency for both forms (target 100% for even/odd).
- Value: n/a. n: n/a.
- Sources: n/a.
- Notes: results/A09/dinov2_vitb14_reg/mirror_exact_astar_s<seed>/ (the healed main arm's audit) missing.

### A10: NOT RUN

- Rule: Standard metrics.
- Bar: Mirror within 1.0 of the original on full-bank k-NN and on the linear protocol.
- Scored quantity: |main Mirror arm's 3-seed mean - original| on the full-bank k-NN (1,281,167 train images) and on the DINOv2 linear protocol (validation top-1 of the classifier chosen on the held-out 10%).
- Value: n/a. n: n/a.
- Sources: n/a.
- Notes: the full-bank summary (runs/final/A10/**/summary.json) missing: skipped by hand (.skip file) (job a10_fullbank).

### A11: REPORTED

- Rule: Healing ceiling.
- Bar: Report. Target: Mirror within 0.2 of the original.
- Scored quantity: extended healing (about 3x the images): Mirror and SVD-LLM k-NN, full-bank k-NN and linear protocol, against the original (target: Mirror within 0.2).
- Value: n/a. n: 1 heal per arm.
- Sources: `runs/final/a11/dinov2_vitb14_reg/svdllm_ext/s0/metrics.json` (87f1d9b1e6); `runs/final/a11/dinov2_vitb14_reg/svdllm_ext/s0/eval/step2340.json` (87f1d9b1e6).
- Notes: runs/final/a11/dinov2_vitb14_reg/mirror_exact_ext/s0: cut by the Section 9 cut order (rank 4) (job heal_a11_mirror_exact).

| arm | knn_top1 | steps | note |
|---|---|---|---|
| Mirror | n/a | n/a | runs/final/a11/dinov2_vitb14_reg/mirror_exact_ext/s0: cut by the Section 9 cut order (rank 4) (job heal_a11_mirror_exact) |
| SVD-LLM | 78.33 | 2340 |  |
| original (same evaluation) | 78.94 | n/a | n/a |

### A12b: FAIL

- Rule: Fused kernel.
- Bar: ≥ 1.1x the original's throughput at batch 64 and 256 (bf16, compiled) earns a speed claim; otherwise a limitation.
- Scored quantity: throughput of the fused even/odd kernel over the original's at batch 64 and 256 (bf16, compiled, same device); >= 1.1 at both earns a speed claim, otherwise a limitation.
- Value: batch 64: 0.804x; batch 256: 0.833x. n: 2 batch sizes.
- Sources: `results/A12b/metrics.json` (6299d2c838).
- Notes: reported as a limitation (no speed claim).

### A15a: NOT RUN

- Rule: SigLIP 2.
- Bar: Report the zero-shot k-NN with whitened projection against the current collapse.
- Scored quantity: SigLIP 2 B/16 zero-shot k-NN with the whitened projection against the plain projection (the CORE-04 collapse), per size.
- Value: n/a. n: n/a.
- Sources: n/a.
- Notes: results/A15a/metrics.json missing: cut by the Section 9 cut order (rank 1) (job a15a_siglip).

### A15b: NOT RUN

- Rule: Cause probe.
- Bar: Rotation ratio to null in blocks 1 to 6 falls by ≥ 0.1 after the fine-tune (from about 0.82); report either way.
- Scored quantity: R90 ratio to null in blocks 1 to 6 after the rot90 fine-tune minus before <= -0.1.
- Value: n/a. n: n/a.
- Sources: n/a.
- Notes: results/A15b/metrics.json missing: cut by the Section 9 cut order (rank 1) (job a15b_measure).

### P11.2: PASS

- Rule: P11.2 (MUST-11, previous phase, re-measured in A10): Mirror-exact with the symmetrised head gives 100% mask equivariance; the original model stays below 100%.
- Bar: Mirror-exact 100% (every differing pixel a near-tie in both copies, PREREG_NEXT_PHASE.md Section 1); original < 100%.
- Scored quantity: share of pixels with mask(mirror x) = mirror(mask x), fp32, ADE20K validation: Mirror-exact (healed seed 0 of MUST-09, symmetrised head) with the near-tie rule read in both copies; the original below 100% with every recorded head.
- Value: Mirror-exact (symmetrised head) 99.99991%: 99.9999% raw, 415 disagreement(s), all near-ties in both copies (largest relative margin 9.98e-06); original 95.0309%, 94.7413%. n: 487,666,546 pixels.
- Sources: `runs/final/A10/p112/t2_ade20k.csv` (21f6ef2682); `runs/final/A10/p112/seg/dinov2_vitb14_reg/vexact_s0_maskeq.json` (not recorded).

| model | mask_eq_sym | n_differ | max_rel_margin_both_copies | n_not_near_tie | verdict | file | shares |
|---|---|---|---|---|---|---|---|
| vexact_s0 | 100.00 | 415 | 9.98e-06 | 0 | PASS | runs/final/A10/p112/t2_ade20k.csv | n/a |
| original | n/a | n/a | n/a | n/a | PASS | runs/final/A10/p112/t2_ade20k.csv | [95.03088797073237, 94.74133499409656] |

## Headline numbers

Key, value and source file with the commit recorded by the run; unrounded in `results/final/numbers.json` and in the `final_phase` section of `PAPER_NUMBERS.json`.

| Key | Value | Unit | n | Description | Source |
|---|---|---|---|---|---|
| a01.adopted | no |  |  | A01: A*_w adopted as the main operator | `runs/final/main/dinov2_vitb14_reg/mirror_exact_wspace/s0/meta.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_wspace/s0/eval/step0.json` (21f6ef2682) and 8 more |
| a01.healed.astar_mean_s01 | 78.36 | % | 2 | A01: healed current A*, mean of seeds 0 and 1 | `runs/final/main/dinov2_vitb14_reg/mirror_exact_wspace/s0/meta.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_wspace/s0/eval/step0.json` (21f6ef2682) and 8 more |
| a01.healed.wspace_mean_s01 | 76.91 | % | 2 | A01: healed A*_w, mean of seeds 0 and 1 | `runs/final/main/dinov2_vitb14_reg/mirror_exact_wspace/s0/meta.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_wspace/s0/eval/step0.json` (21f6ef2682) and 8 more |
| a01.wspace.d_minus | 68 |  |  | A01: d- of A*_w (matched mode) | `results/A01/metrics.json` (8b5fa38c7e) |
| a01.zero_shot | 7.19 | % |  | A01: zero-shot k-NN of the A*_w full conversion (step 0 of its heal, bf16, 50,000 queries) | `runs/final/main/dinov2_vitb14_reg/mirror_exact_wspace/s0/meta.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_wspace/s0/eval/step0.json` (21f6ef2682) |
| a02.coco_minus_imagenet | -0.0020 | points |  | A02-6: COCO-fit heal minus the mean of the ImageNet-fit heals | `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/eval/step780.json` (21f6ef2682) and 6 more |
| a02.d_minus.fit0 | 68 |  |  | A02: d- of A*_12, fit set 0 | `results/A02/metrics.json` (21f6ef2682) |
| a02.head_share_same.min | 1.00 |  | 6 | A02: smallest head_share_same over the pairs of ImageNet fits | `results/A02/metrics.json` (21f6ef2682) |
| a02.heal_spread | 0.0220 | points | 3 | A02-5: spread (max - min) of the healed k-NN over fit sets 0 to 2 | `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/eval/step780.json` (21f6ef2682) and 4 more |
| a02.twin_mean.min | 0.9533 |  | 6 | A02: smallest twin_mean over the pairs of ImageNet fits | `results/A02/metrics.json` (21f6ef2682) |
| a02.weighted_agreement.min | 0.9990 |  | 6 | A02: smallest weighted_agreement over the pairs of ImageNet fits | `results/A02/metrics.json` (21f6ef2682) |
| a03.C1.ratio_in_sample | 0.9700 |  |  | A03: C1 in-sample ratio to null, blocks 1 to 6 (reported beside the scored split-half ratio) | `results/A03/metrics.json` (21f6ef2682) |
| a03.C1.ratio_split_half | 0.9765 |  |  | A03: C1 split-half ratio to null, blocks 1 to 6 | `results/A03/metrics.json` (21f6ef2682) |
| a03.C1.twin_fraction_1_6 | 0.0036 |  |  | A03: C1 twin fraction, blocks 1 to 6 | `results/A03/metrics.json` (21f6ef2682) |
| a03.C2.ratio_in_sample | 1.04 |  |  | A03: C2 in-sample ratio to null, blocks 1 to 6 (reported beside the scored split-half ratio) | `results/A03/metrics.json` (21f6ef2682) |
| a03.C2.ratio_split_half | 1.03 |  |  | A03: C2 split-half ratio to null, blocks 1 to 6 | `results/A03/metrics.json` (21f6ef2682) |
| a03.C2.twin_fraction_1_6 | 2.17e-04 |  |  | A03: C2 twin fraction, blocks 1 to 6 | `results/A03/metrics.json` (21f6ef2682) |
| a03.C3.ratio_in_sample | 0.9209 |  |  | A03: C3 in-sample ratio to null, blocks 1 to 6 (reported beside the scored split-half ratio) | `results/A03/metrics.json` (21f6ef2682) |
| a03.C3.ratio_split_half | 0.9218 |  |  | A03: C3 split-half ratio to null, blocks 1 to 6 | `results/A03/metrics.json` (21f6ef2682) |
| a03.C3.twin_fraction_1_6 | 0.0794 |  |  | A03: C3 twin fraction, blocks 1 to 6 | `results/A03/metrics.json` (21f6ef2682) |
| a03.h1_minus_mirror | -1.34 | points |  | A03-2: H1 (true operator, random pairing) minus the main Mirror arm's mean | `runs/final/main/dinov2_vitb14_reg/pair_random_h1/s0/metrics.json` (87f1d9b1e6); `runs/final/main/dinov2_vitb14_reg/pair_random_h1/s0/eval/step780.json` (87f1d9b1e6) and 6 more |
| a04.blockdrop.knn | 75.25 | % |  | A04: block-dropped (blockdrop) dense student, final k-NN (seed 0) | `runs/final/main/dinov2_vitb14_reg/blockdrop/s0/metrics.json` (11e50a1526); `runs/final/main/dinov2_vitb14_reg/blockdrop/s0/eval/step780.json` (11e50a1526) |
| a04.blockdrop.params_frac | 0.7544 |  |  | A04: block-dropped (blockdrop) student, share of the original's parameters | `runs/final/main/dinov2_vitb14_reg/blockdrop/s0/metrics.json` (11e50a1526); `runs/final/main/dinov2_vitb14_reg/blockdrop/s0/eval/step780.json` (11e50a1526) |
| a04.flap.knn | 78.41 | % |  | A04: width-pruned (flap) dense student, final k-NN (seed 0) | `runs/final/main/dinov2_vitb14_reg/flap/s0/metrics.json` (11e50a1526); `runs/final/main/dinov2_vitb14_reg/flap/s0/eval/step780.json` (11e50a1526) |
| a04.flap.params_frac | 0.7646 |  |  | A04: width-pruned (flap) student, share of the original's parameters | `runs/final/main/dinov2_vitb14_reg/flap/s0/metrics.json` (11e50a1526); `runs/final/main/dinov2_vitb14_reg/flap/s0/eval/step780.json` (11e50a1526) |
| a04.original_distilled.knn | 78.95 | % |  | A04-2: original distilled with the healing recipe, final k-NN | `runs/final/main/dinov2_vitb14_reg/original/s0/metrics.json` (11e50a1526); `runs/final/main/dinov2_vitb14_reg/original/s0/eval/step780.json` (11e50a1526) |
| a05.best_baseline | svdllm |  |  | A05-1: best baseline (highest mean healed k-NN) | `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/eval/step780.json` (21f6ef2682) and 32 more |
| a05.iosvd.mean | 0.5340 | % | 3 | A05: IO-SVD, mean healed k-NN over its seeds | `runs/final/main/dinov2_vitb14_reg/iosvd/s0/metrics.json` (11e50a1526); `runs/final/main/dinov2_vitb14_reg/iosvd/s0/eval/step780.json` (11e50a1526) and 4 more |
| a05.mirror_minus_best.ci_high | 0.4649 | points |  | A05-1: upper end of the 95% interval | `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/eval/step780.json` (21f6ef2682) and 32 more |
| a05.mirror_minus_best.ci_low | 0.0809 | points |  | A05-1: lower end of the 95% interval (A13 bootstrap, 10,000 resamples, seed 0) | `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/eval/step780.json` (21f6ef2682) and 32 more |
| a05.mirror_minus_best.point | 0.2713 | points |  | A05-1: Mirror minus best baseline, point estimate | `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/eval/step780.json` (21f6ef2682) and 32 more |
| a05.mirror_point | 78.35 | % | 5 | A05-1: Mirror point estimate (mean over fit sets 0 to 2 of the fit-set means) | `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/eval/step780.json` (21f6ef2682) and 32 more |
| a05.slicegpt.mean | 77.85 | % | 2 | A05: SliceGPT-style, mean healed k-NN over its seeds | `runs/final/main/dinov2_vitb14_reg/slicegpt/s0/metrics.json` (11e50a1526); `runs/final/main/dinov2_vitb14_reg/slicegpt/s0/eval/step780.json` (579ada6c2a) and 2 more |
| a05.superiority | yes |  |  | A05-1: superiority claimed (lower end above 0) | `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/eval/step780.json` (21f6ef2682) and 32 more |
| a05.svd.mean | 77.00 | % | 1 | A05: plain SVD, mean healed k-NN over its seeds | `runs/final/main/dinov2_vitb14_reg/svd/s0/metrics.json` (11e50a1526); `runs/final/main/dinov2_vitb14_reg/svd/s0/eval/step780.json` (579ada6c2a) |
| a05.svdllm.mean | 78.08 | % | 3 | A05: SVD-LLM (whitening only), mean healed k-NN over its seeds | `runs/final/main/dinov2_vitb14_reg/svdllm/s0/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/svdllm/s0/eval/step780.json` (21f6ef2682) and 4 more |
| a05.svdllm_full.mean | 76.42 | % | 3 | A05: full SVD-LLM, mean healed k-NN over its seeds | `runs/final/main/dinov2_vitb14_reg/svdllm_full/s0/metrics.json` (11e50a1526); `runs/final/main/dinov2_vitb14_reg/svdllm_full/s0/eval/step780.json` (11e50a1526) and 4 more |
| a06.f040.mirror_svdllm | 71.60 | % | 1 | A06: Mirror + SVD-LLM at 0.4 of the full size, final k-NN | `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f040/s0/metrics.json` (579ada6c2a); `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f040/s0/eval/step780.json` (66cebdae87) |
| a06.f040.predicted_gain | 5.96 | points |  | A06-2: pre-registered predicted stacking gain at 0.4 | `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f040/s0/metrics.json` (579ada6c2a); `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f040/s0/eval/step780.json` (66cebdae87) and 16 more |
| a06.f040.svdllm | 67.47 | % | 1 | A06: SVD-LLM at 0.4 of the full size, final k-NN | `runs/final/a06/dinov2_vitb14_reg/svdllm_f040/s0/metrics.json` (21f6ef2682); `runs/final/a06/dinov2_vitb14_reg/svdllm_f040/s0/eval/step780.json` (7d05b6439c) |
| a06.f050.diff_ci_high | 3.01 | points |  | A06-1: upper end of the same interval | `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f065/s0/metrics.json` (579ada6c2a); `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f065/s0/eval/step780.json` (c7c76556a8) and 14 more |
| a06.f050.diff_ci_low | 2.48 | points |  | A06-1: lower end of the 95% interval of Mirror + SVD-LLM minus SVD-LLM at 0.50 | `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f065/s0/metrics.json` (579ada6c2a); `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f065/s0/eval/step780.json` (c7c76556a8) and 14 more |
| a06.f050.mirror_svdllm | 76.26 | % | 2 | A06: Mirror + SVD-LLM at 0.5 of the full size, final k-NN | `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f050/s0/metrics.json` (579ada6c2a); `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f050/s0/eval/step780.json` (66cebdae87) and 2 more |
| a06.f050.predicted_gain | 2.51 | points |  | A06-2: pre-registered predicted stacking gain at 0.5 | `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f050/s0/metrics.json` (579ada6c2a); `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f050/s0/eval/step780.json` (66cebdae87) and 20 more |
| a06.f050.svdllm | 73.51 | % | 2 | A06: SVD-LLM at 0.5 of the full size, final k-NN | `runs/final/a06/dinov2_vitb14_reg/svdllm_f050/s0/metrics.json` (21f6ef2682); `runs/final/a06/dinov2_vitb14_reg/svdllm_f050/s0/eval/step780.json` (7d05b6439c) and 2 more |
| a06.f065.mirror_svdllm | 78.20 | % | 1 | A06: Mirror + SVD-LLM at 0.65 of the full size, final k-NN | `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f065/s0/metrics.json` (579ada6c2a); `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f065/s0/eval/step780.json` (c7c76556a8) |
| a06.f065.predicted_gain | 1.26 | points |  | A06-2: pre-registered predicted stacking gain at 0.65 | `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f065/s0/metrics.json` (579ada6c2a); `runs/final/a06/dinov2_vitb14_reg/mirror_svdllm_f065/s0/eval/step780.json` (c7c76556a8) and 16 more |
| a06.f065.svdllm | 76.57 | % | 1 | A06: SVD-LLM at 0.65 of the full size, final k-NN | `runs/final/a06/dinov2_vitb14_reg/svdllm_f065/s0/metrics.json` (21f6ef2682); `runs/final/a06/dinov2_vitb14_reg/svdllm_f065/s0/eval/step780.json` (547224c624) |
| a06.price_of_exactness | 0.4280 | points |  | A06-3: 9-block Mirror minus the full conversion (main arm mean) | `runs/final/a06/dinov2_vitb14_reg/mirror_partial_k9/s0/metrics.json` (579ada6c2a); `runs/final/a06/dinov2_vitb14_reg/mirror_partial_k9/s0/eval/step780.json` (66cebdae87) and 6 more |
| a07.mirror_minus_best_flipavg | 3.76 | points |  | A07: Mirror (3-seed mean) minus the best flip-averaged model at or below Mirror's measured compute | `runs/final/a06/dinov2_vitb14_reg/svdllm_f065/s0/metrics.json` (21f6ef2682); `runs/final/a06/dinov2_vitb14_reg/svdllm_f065/s0/eval/step780.json` (547224c624) and 13 more |
| a07.mirror_ms_per_image_b256 | 0.1219 | ms |  | A07: Mirror's faster stored form, time per image at batch 256 (bf16, compiled) | `results/A12/dinov2_vitb14_reg/systems_table.csv` (6299d2c838) |
| a08.deit3.mirror_top1_mean | 82.70 | % | 3 | A08-1: DeiT-III-B Mirror, classifier top-1, mean of seeds 0 to 2 | `runs/final/a08/deit3_b16/mirror_exact/s0/metrics.json` (66cebdae87); `runs/final/a08/deit3_b16/mirror_exact/s0/eval/top1_step780.json` (66cebdae87) and 4 more |
| a08.deit3.original_top1 | 83.68 | % |  | A08: released DeiT-III-B/16, classifier top-1 (bf16) | `runs/final/a08/deit3_b16/_original/eval/top1_original.json` (21f6ef2682) |
| a08.deit3.svdllm_top1_mean | 82.88 | % | 3 | A08-1: DeiT-III-B SVD-LLM, classifier top-1, mean of seeds 0 to 2 | `runs/final/a08/deit3_b16/svdllm/s0/metrics.json` (66cebdae87); `runs/final/a08/deit3_b16/svdllm/s0/eval/top1_step780.json` (66cebdae87) and 4 more |
| a12b.speed_b256 | 0.8330 |  |  | A12b: fused kernel throughput over the original's, batch 256 | `results/A12b/metrics.json` (6299d2c838) |
| a12b.speed_b64 | 0.8044 |  |  | A12b: fused kernel throughput over the original's, batch 64 | `results/A12b/metrics.json` (6299d2c838) |
| gate_p.mirror_exact_astar.knn_mean | 78.35 | % | 3 | Gate P: final bf16 k-NN, mean of seeds 0 to 2, mirror_exact_astar | `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s0/eval/step780.json` (21f6ef2682) and 4 more |
| gate_p.svdllm.knn_mean | 78.08 | % | 3 | Gate P: final bf16 k-NN, mean of seeds 0 to 2, svdllm | `runs/final/main/dinov2_vitb14_reg/svdllm/s0/metrics.json` (21f6ef2682); `runs/final/main/dinov2_vitb14_reg/svdllm/s0/eval/step780.json` (21f6ef2682) and 4 more |
| p11_2.vexact_s0.mask_eq_sym | 100.00 | % |  | P11.2 (re-measured): mask equivariance, healed Mirror-exact seed 0, symmetrised head, fp32 | `runs/final/A10/p112/t2_ade20k.csv` (21f6ef2682) |
| p11_2.vexact_s0.max_rel_margin | 9.98e-06 |  |  | P11.2: largest relative top-two margin of the differing pixels (larger of the two copies) | `runs/final/A10/p112/t2_ade20k.csv` (21f6ef2682) |

## Deviations logged after the pre-registration commit

From `notes/decisions.md` (rows absent at the pre-registration commit fd83c955d7).

| 2026-11-20 | Dawar confirmed the operational choices of PREREG_FINAL_PHASE.md (fd83c95): A01 matched mode, A03-1 on the split-half ratio, the A09-2 floor against the re-projected float64 model, and the smaller readings of its Section 2. The pre-registration is signed off as committed; nothing in it changes from here | Dawar's reply "confirm" | Phase G may start once the VM is up | Dawar |
| 2026-11-20 | Manifest compressed for wall clock (scripts/fp_make_jobs.py): Gate P (current A* and SVD-LLM, 3 seeds) no longer waits for the CPU-bound A*_w fit; the A*_w heals are their own jobs after the fit and Gate P; every heal that does not use the A01 decision (the seed-0 baseline matrix, the other A05 seeds, the A06 SVD-LLM points, A11 SVD-LLM) starts as soon as Gate P passes instead of after the A01 decision; H1 moved into the A02 job and the A06 Mirror points into their own job (both use the adopted operator); float32 k-NN passes only on seed 0 of each Mirror arm, Mirror + SVD-LLM at 0.50 and DINOv2-L (DeiT-III's float32 check is its classifier pass); no step-0 evaluation where no rule uses it (A02, A06, A11, DINOv2-L, A15b); the linear protocol trains in TF32. 81 jobs, 61.5 heal-units, unchanged science | Dawar asked to compress GPU time as much as possible | Gate P still runs before any other heal (PREREG Section 2.2) | Claude (operational) |
| 2026-11-20 | Phase G VM (Jarvis H200, jl-vm-581628, 143,771 MiB, 28 cores, 295 GB RAM, 485 GB disk; ssh ubuntu@217.18.55.50 with the vast key; user ubuntu with sudo). Two setup changes on the VM only: /home made group-writable for ubuntu (root:ubuntu 775) so scripts/ship_h100.sh can stage its archives there; MIRROR_MEM_SCALE removed from ~/.mirror_env.sh after the bootstrap, because the final-phase memory fractions are already H200 fractions (scripts/fp_make_jobs.py: (GB + 6) / 141) and the bootstrap's 80 GB rescaling (0.570) would have capped a 45 GB heal near 29 GB. Older scripts with a 0.25 cap get 35 GB instead of 20 GB, harmless on a GPU used only by this project. VM clock started 07:18:52 UTC | Machine checks; scripts/h100_bootstrap.sh step 5 | No result changes; memory caps as designed for the H200 | Claude |
| 2026-11-20 | Phase G start-up fixes, all found on the VM or by two review agents before the code ran (no result touched): (1) set_per_process_memory_fraction needs a device index in torch 2.11 (every heal and the eval server would have crashed; 68160fd); (2) GNU nproc honours OMP_NUM_THREADS, so under the runner the data build had 2 workers of 28 (8b5fa38; the build restarted, the runner logged one failure); (3) TIMING_CMD's ${VAR:-default} ended at the } of {model} and appended text, so w0_timing failed twice and the runner stopped; reset and resumed (24da351); (4) the W0 timing now starts clean, a DeiT or ViT-L timing failure no longer blocks Gate P, and a full heal is estimated as wall time + remaining steps x measured s/step (start-up was scaled by 13, which would have inflated the projection and cut jobs early; 3b71e63); (5) the eval server sends a failed request back to the queue at most twice, and restarts itself after a sticky CUDA error, so a decision job never waits 6 h for a lost evaluation; (6) A10 deletes each streamed train shard after use (about 147 GB otherwise); (7) A12 peak memory now counts the timed model's weights, input and activations only (other models in the process were included; the process figure is kept as peak_memory_process_gb). Not changed: the short A01/A02/A03 fits are kind cpu but use up to about 5 GB of GPU memory unseen by the scheduler (they run 20 s to 3 min) | VM logs; review agents | Run continues; A12 memory numbers comparable across models | Claude |
| 2026-11-20 | Dawar (11:10 UTC, 3.9 VM-hours used): finish within 6 to 7 more hours, not up to the 15 h ceiling. Measured rate from the first five heals 0.20 VM-hours per heal-unit; everything would end near 12.6 VM-hours. Manifest budget set in scripts/fp_make_jobs.py: target 10.75 VM-hours, no new heal after 10.25, hard stop 11.25; the governor cuts in the Section 9 order (projection: A15a/b, DINOv2-L, A11 Mirror; A11 SVD-LLM was already at step 1940 of 2340 and finishes as an unpaired data point). Scheduling interventions earlier the same morning, all without effect on any result: the runner packs greedily by declared GPU memory, so after Gate P and again after A01 evaluations and small heals took the room the 61 GB seed-0 baseline job (heal_main_s0, the critical path: its learning-rate selection gates seeds 1 and 2) needed; a15a_siglip (8 min in), heal_main_s1_slicegpt (70 s), a07_flipavg_svdllm (30 s), a07_probe_mirror (40 s) and a09_audit_b (24 min) were stopped and reset (their partial outputs in runs/final/_superseded/), and the other jobs were held with .skip files until heal_main_s0 started at 11:09:44, then released | Dawar's time limit; runner logs | DINOv2-L, A11 and A15 drop out unless the measured rate improves; every never-cut item stays | Dawar / Claude |
| 2026-11-20 | IO-SVD diverges under the pre-registered recipe: its 200-step trial chose 1e-4 (held-out 1.79 -> 1.88; 2e-4 diverged to 68.8), and in the seed-0 heal (heal_main_s0) its loss rose from 1.62 (step 20) to 401 (step 100) once warm-up ended, while the other seven students trained normally. Dawar chose: keep the pre-registered IO-SVD runs and report them as they are (diverged), and add one clearly labelled post-hoc run at 5e-5, seed 0 (job heal_posthoc_iosvd, group posthoc, experiment A05-posthoc); it enters no pre-registered score | heal_main_s0 log; Dawar's answer | The main table reports IO-SVD as diverged at the pre-registered rate, with the post-hoc 5e-5 result beside it, labelled | Dawar |
| 2026-11-20 | Full SVD-LLM port fixed and rerun (Dawar's choice, 13:35 UTC). The first runs used one warm-up and cosine over both stages of SVD-LLM's update (U for steps 0 to 389, then V): V started with a fresh optimiser state at about half the peak rate and no warm-up, the loss jumped (seed 0: 2.04 at step 380 to 11.2 at step 440; seed 1: 2.00 to 4.34) and the student ended worse than it began (held-out 2.83 -> 3.11); its 200-step trial worsened at both rates. The official method runs the two stages as separate fine-tunes, so each stage now has its own warm-up (the run's 5%) and cosine (mirror/final/baselines/svdllm_full.py stage_schedule "own"; the lockstep trainer takes the student's rate from the hook's lr_at). The rate trial is redone (heal_svdllm_full_s0) and seeds 1 and 2 rerun (heal_svdllm_full_s1, heal_main_s2); the first runs (seeds 0 and 1, and seed 2 stopped at step 320) are kept in runs/final/_superseded/svdllm_full_shared_schedule/ | heal_main_s0, heal_main_s1 logs | Full SVD-LLM is compared as the official two-stage method; no other student changes | Dawar / Claude |
| 2026-11-20 | Post-hoc IO-SVD at 5e-5 extended to seeds 1 and 2 (Dawar, 14:15 UTC): seed 0 reached k-NN 78.71 against Mirror's 78.35 (3-seed mean), the strongest baseline at the main size, so it gets a 3-seed mean. Still post hoc, outside every pre-registered score (A05-1's best baseline is chosen among the pre-registered runs) | step780 evaluations | The paper reports the post-hoc IO-SVD 3-seed mean beside the pre-registered table | Dawar |
| 2026-11-20 | Full SVD-LLM fix abandoned; back to the pre-registered port (Dawar, 14:25 UTC). With each stage's own warm-up the V stage reached the full peak rate and diverged harder (seed 0: loss 2.10 at step 380 to 152 at step 460, held-out 2.83 -> 13.13; its new rate trial 2.83 -> 15.1 at 1e-4), so the cause is not the missing warm-up: training SVD-LLM's factors directly with AdamW at these rates is unstable, while the official code updates them through LoRA adapters, which this port does not reproduce. Default stage_schedule back to "shared"; the original seed 0 and 1 runs (k-NN 76.45 and 76.76) are restored as the full SVD-LLM results and seed 2 reruns with the same port (heal_main_s2); the per-stage runs (seed 0 complete, seed 1 stopped after 3 min) are kept in runs/final/_superseded/svdllm_full_own_schedule/. Full SVD-LLM is below whitening-only SVD-LLM (78.08) either way, so A05-1's best baseline does not change | heal_svdllm_full_s0 log | Report full SVD-LLM as pre-registered, with the port's limitation stated | Dawar / Claude |
| 2026-11-20 | 15:30 UTC, last three heals running: the runner admits jobs by measured free GPU memory (declared_budget off, a new job's declared GB reserved for 600 s instead of 240 s), so the evaluations, above all the 45 GB A10 full-bank pass over 1.28M training images, start beside the last heals instead of after them (67 GB was free while the declared budget was full) | runner status, gpu.csv | Shorter tail; each job keeps its own memory cap | Claude |
| 2026-11-20 | A10 full-bank pass cut to five models (16:30 UTC): it ran at 44 images/s over 10 models while sharing the GPU with five processes (both A09 float64 audits among them), about 8 hours for 1.28M images. A10's pre-registered rule needs only Mirror against the original, so the pass now runs the original, Mirror seeds 0 to 2 and SVD-LLM seed 0 (context); SVD-LLM seeds 1 and 2, full SVD-LLM, the diverged pre-registered IO-SVD and A11 SVD-LLM were dropped. The first attempt (61,440 images in 25 min) is kept in runs/final/_superseded/A10_fullbank_10models/ | a10_fullbank log | A10 is scored as pre-registered; fewer context rows | Claude |
| 2026-11-20 | Hard stop raised from 11.25 to 11.75 VM-hours (Dawar, 16:40 UTC): the A10 full-bank pass, the A12 benchmarks (each alone on the GPU; A07's compute matching uses them) and the report would end near 11.5 VM-hours | runner projection | Run ends about 18:35 to 18:50 UTC | Dawar |

## Tables

Under `results/final/tables/` (CSV with source columns, and markdown): T1_main_comparison, T2_stage_ablation, T3_other_models, T4_operator_stability, T4_operator_stability_pairs, T5_discovery_controls, T6_systems.

## Scoring conventions

- Section 3 of the pre-registration: a rule over several units fails as soon as one available unit fails, passes only when every unit is present and passes, and is otherwise PENDING with the reason; a rule whose needed input was cut is NOT RUN, never passed.
- "At least k of n" rules (A06-1, A06-2) pass once k units pass and fail once k can no longer be reached.
- "Within x" is |difference| <= x; values within 1e-9 of a bar count as equal to the bar.
- Healed values: final-step bf16 k-NN (step 780), all 50,000 validation queries, 100k bank.
- 100% flip consistency: no disagreement, or every disagreement a near-tie in both copies (relative top-two margin below 1e-4 in the image and its mirror). When only the one-copy margin is recorded, the unit is PENDING unless a one-copy margin already reaches 1e-4.
- A05-1: point estimate = mean over fit sets of the fit-set means; interval = mirror.final.stats.hbootstrap (10,000 resamples, seed 0, 95% percentile, images paired, seeds drawn independently per method).
- A07: candidates whose measured time per image at batch 256 (flip averaging: two passes) is at most Mirror's faster stored form; Mirror must be at least 0.5 above the best of them.

## Inputs read

240 files; their sha256 are in `results/final/numbers.json` (`inputs`).

