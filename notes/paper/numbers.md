# Numbers for the CPAL 2027 paper (Mirror Surgery)

Compiled on 2026-11-17 in a read-only pass over the result files of this repository. Every value is read from, or computed by plain arithmetic (mean, sample standard deviation, difference, ratio, count) over, the file named under its table. Values marked **(derived)** were computed for this note by a throwaway script outside the repository; they are not stored in any committed result file, so under the CLAUDE.md rule they should be added to `scripts/collect_next_phase.py` before they are cited. Nothing here is typed from memory or from prose. Main model: `dinov2_vitb14_reg` (DINOv2 ViT-B/14 with registers).

Conventions (from `results/MUST-07/SUMMARY.md` and `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0/meta.json`, config.knn): k-NN = ImageNet validation top-1 (%) with a 100,000-image ImageNet-train bank (100 per class), 50,000 validation queries, cosine, k = 20, T = 0.07, bf16 features unless marked fp32 (TF32 off). "msel" = the 10,000 method-selection train images used as queries (all hyperparameter choices and gate B). sd = sample standard deviation (n - 1) over seeds. Flip consistency = share of validation images whose prediction equals that of the mirrored image. GFLOPs are analytic, per image.

## 0. Provenance: dates, hardware, commits

Collector: `PAPER_NUMBERS.json` and `FINAL_RESULTS.md` written by scripts/collect_next_phase.py at 2026-11-15T20:29:32Z, commit 6b4c66ab78 (dirty: True); counts {'scored': 40, 'PASS': 17, 'FAIL': 16, 'PENDING': 7, 'REPORTED': 2}. Pre-registration `PREREG_NEXT_PHASE.md`, sign-off commit e76fb17c3c (from `FINAL_RESULTS.md` line 5).

| Experiment | Date (UTC) | Commit in meta.json | Dirty | GPU | File |
|---|---|---|---|---|---|
| MUST-01 | 2026-11-15T08:02:15Z | 52e29d1676 | True | not recorded | `results/MUST-01/meta.json` |
| MUST-02 fp32 | 2026-11-15T08:49:38Z | e84e20ff35 | True | NVIDIA H200 | `results/MUST-02/dinov2_vitb14_reg/zeroshot/meta.json` |
| MUST-02 fp64 | 2026-11-15T07:55:10Z | 52e29d1676 | False | NVIDIA H200 | `results/MUST-02/dinov2_vitb14_reg/zeroshot_fp64/meta.json` |
| MUST-04 b14-reg | 2026-11-15T10:25:18Z | e84e20ff35 | True | NVIDIA H200 | `results/MUST-04/dinov2_vitb14_reg/meta.json` |
| MUST-04 b14 | 2026-11-15T10:57:50Z | e84e20ff35 | True | NVIDIA H200 | `results/MUST-04/dinov2_vitb14/meta.json` |
| MUST-04 l14-reg | 2026-11-15T11:44:20Z | e84e20ff35 | True | NVIDIA H200 | `results/MUST-04/dinov2_vitl14_reg/meta.json` |
| MUST-04 deit3 | 2026-11-15T10:36:28Z | e84e20ff35 | True | NVIDIA H200 | `results/MUST-04/deit3_b16/meta.json` |
| MUST-05 | 2026-11-15T07:54:02Z | 52e29d1676 | False | NVIDIA H200 | `results/MUST-05/dinov2_vitb14_reg/meta.json` |
| MUST-06 b14-reg | 2026-11-15T08:18:41Z | 52e29d1676 | True | NVIDIA H200 | `results/MUST-06/dinov2_vitb14_reg/meta.json` |
| MUST-07 (final merge) | 2026-11-15T19:12:21Z | 805c231c9c | True | NVIDIA H200 | `results/MUST-07/meta.json` |
| MUST-08 | 2026-11-15T19:27:16Z | 805c231c9c | True | NVIDIA H200 | `results/MUST-08/meta.json` |
| MUST-09 mirror_exact s0 | 2026-11-15T08:40:18Z | e84e20ff35 | True | NVIDIA H200 | `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0/meta.json` |
| MUST-09 mirror_exact s1 | 2026-11-15T09:44:39Z | e84e20ff35 | True | NVIDIA H200 | `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s1/meta.json` |
| MUST-09 mirror_exact s2 | 2026-11-15T10:26:17Z | e84e20ff35 | True | NVIDIA H200 | `runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s2/meta.json` |
| MUST-09 random_structure s0 | 2026-11-15T08:53:33Z | e84e20ff35 | True | NVIDIA H200 | `runs/MUST-09/dinov2_vitb14_reg/random_structure/s0/meta.json` |
| MUST-09 slicegpt s0 | 2026-11-15T14:59:56Z | 805c231c9c | True | NVIDIA H200 | `runs/MUST-09/dinov2_vitb14_reg/slicegpt/s0/meta.json` |
| MUST-09 svd s0 | 2026-11-15T10:07:49Z | e84e20ff35 | True | NVIDIA H200 | `runs/MUST-09/dinov2_vitb14_reg/svd/s0/meta.json` |
| MUST-09 svdllm s0 | 2026-11-15T14:02:37Z | 805c231c9c | True | NVIDIA H200 | `runs/MUST-09/dinov2_vitb14_reg/svdllm/s0/meta.json` |
| MUST-09 svdllm s1 | 2026-11-15T16:58:34Z | 805c231c9c | True | NVIDIA H200 | `runs/MUST-09/dinov2_vitb14_reg/svdllm/s1/meta.json` |
| MUST-09 svdllm s2 | 2026-11-15T17:00:40Z | 805c231c9c | True | NVIDIA H200 | `runs/MUST-09/dinov2_vitb14_reg/svdllm/s2/meta.json` |
| MUST-10 | 2026-11-15T20:17:01Z | 805c231c9c | True | NVIDIA H200 | `results/MUST-10/meta.json` |
| UP-06 | 2026-11-15T11:46:31Z | e84e20ff35 | True | NVIDIA H200 | `results/UP-06/meta.json` |
| UP-07 | 2026-11-15T10:26:57Z | e84e20ff35 | True | not recorded | `results/UP-07/meta.json` |
| CORE-03 | 2026-11-15T19:43:17Z | 805c231c9c | True | NVIDIA H200 | `results/CORE-03/meta.json` |
| CORE-04 (final) | 2026-11-15T19:46:10Z | 805c231c9c | True | NVIDIA H200 | `results/CORE-04/meta.json` |
| CORE-05 measures | 2026-11-15T15:00:57Z | 805c231c9c | True | NVIDIA H200 | `results/CORE-05/meta_measures.json` |
| CORE-05 rule frozen | 2026-11-15T16:36:17Z | 805c231c9c | True | NVIDIA H200 | `results/CORE-05/meta_freeze.json` |
| CORE-05 scored | 2026-11-15T20:21:19Z | 805c231c9c | True | NVIDIA H200 | `results/CORE-05/meta_score.json` |
| CORE-06 | 2026-11-15T20:09:34Z | 805c231c9c | True | NVIDIA H200 | `results/CORE-06/meta.json` |
| CORE-07 | 2026-11-15T17:09:27Z | 805c231c9c | True | NVIDIA H200 | `results/CORE-07/meta.json` |
| MUST-12 vit_s16_LR_s0 | 2026-11-15T12:52:24Z | d572a53fe9 | True | NVIDIA H200 | `runs/MUST-12/vit_s16_LR_s0/meta.json` |
| MUST-12 vit_s16_none_s0 | 2026-11-15T12:51:44Z | d572a53fe9 | True | NVIDIA H200 | `runs/MUST-12/vit_s16_none_s0/meta.json` |
| MUST-12 vit_s8_LR_s0 | 2026-11-15T07:48:31Z | 52e29d1676 | False | NVIDIA H200 | `runs/MUST-12/vit_s8_LR_s0/meta.json` |
| MUST-12 vit_s8_none_s0 | 2026-11-15T07:26:16Z | 52e29d1676 | False | NVIDIA H200 | `runs/MUST-12/vit_s8_none_s0/meta.json` |

Hardware and software (env block of the meta files above): NVIDIA H200, CUDA 12.8, torch 2.11.0+cu128, Python 3.10.12, numpy 2.2.6, Linux-5.15.0-202-generic-x86_64-with-glibc2.35. All GPU work above is dated 2026-11-15.

| Commit | Commit time (git log, committer local time) |
|---|---|
| e76fb17c3c | 2026-11-14T23:33:51+05:30 |
| 52e29d1676 | 2026-11-15T11:43:33+05:30 |
| e84e20ff35 | 2026-11-15T13:58:06+05:30 |
| 5edf2980f0 | 2026-11-15T17:34:59+05:30 |
| d572a53fe9 | 2026-11-15T17:41:59+05:30 |
| 805c231c9c | 2026-11-15T19:20:21+05:30 |
| 70a47df49e | 2026-11-15T22:07:27+05:30 |
| 6b4c66ab78 | 2026-11-16T01:12:34+05:30 |

MUST-11 meta files (`results/MUST-11/t2/t2_linear_meta_dinov2_vitb14_reg.json`, `t2_transfer_meta_...json`, `results/MUST-11/t2_seg/t2_ade20k_meta_...json`) carry commit 805c231c9c and NVIDIA H200 but no date_utc; the per-row `date` column of the MUST-11 CSVs gives 2026-11-15 14:24 to 18:15.

## T1. Healed comparison (MUST-09)

### T1a. Every heal (validation k-NN after 780 steps)

| Method | Seed | k-NN bf16 | k-NN fp32 | Flip bf16 | Flip fp32 (disagreements of 50,000) | Params (fraction) | GFLOPs (fraction) | Steps | LR | k-NN at step 0 | Commit (metrics.json) | Commit (meta.json) | Started (meta) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mirror_exact | 0 | 78.316 | 78.372 | 99.744 | 100.000 (0) | 66,203,360 (0.7646) | 36.44 (0.7740) | 780 | 0.0001 | 53.84 | 52e29d1676 | e84e20ff35 | 2026-11-15T08:40:18Z |
| mirror_exact | 1 | 78.352 | 78.340 | 99.714 | 100.000 (0) | 66,203,360 (0.7646) | 36.44 (0.7740) | 780 | 0.0001 | 53.84 | e84e20ff35 | e84e20ff35 | 2026-11-15T09:44:39Z |
| mirror_exact | 2 | 78.352 | 78.324 | 99.724 | 100.000 (0) | 66,203,360 (0.7646) | 36.44 (0.7740) | 780 | 0.0001 | 53.84 | e84e20ff35 | e84e20ff35 | 2026-11-15T10:26:17Z |
| svdllm | 0 | 78.048 | 78.070 | 95.812 | 95.804 (2098) | 66,203,904 (0.7646) | 36.44 (0.7740) | 780 | 0.0001 | 75.80 | e84e20ff35 | 805c231c9c | 2026-11-15T14:02:37Z |
| svdllm | 1 | 78.074 | 78.070 | 95.640 | 95.706 (2147) | 66,203,904 (0.7646) | 36.44 (0.7740) | 780 | 0.0001 | 75.80 | 805c231c9c | 805c231c9c | 2026-11-15T16:58:34Z |
| svdllm | 2 | 78.060 | 78.084 | 95.736 | 95.732 (2134) | 66,203,904 (0.7646) | 36.44 (0.7740) | 780 | 0.0001 | 75.80 | 805c231c9c | 805c231c9c | 2026-11-15T17:00:40Z |
| slicegpt | 0 | 77.886 | 77.916 | 95.400 | 95.432 (2284) | 66,324,696 (0.7660) | 36.50 (0.7754) | 780 | 0.0001 | 68.29 | e84e20ff35 | 805c231c9c | 2026-11-15T14:59:56Z |
| svd | 0 | 77.018 | 76.992 | 94.234 | 94.296 (2852) | 66,203,904 (0.7646) | 36.44 (0.7740) | 780 | 0.0001 | 5.20 | e84e20ff35 | e84e20ff35 | 2026-11-15T10:07:49Z |
| random_structure | 0 | 69.934 | 69.964 | 99.340 | 100.000 (0) | 66,203,360 (0.7646) | 36.44 (0.7740) | 780 | 0.0001 | 4.78 | 52e29d1676 | e84e20ff35 | 2026-11-15T08:53:33Z |

Source: `runs/MUST-09/dinov2_vitb14_reg/<method>/s<seed>/metrics.json (knn_bf16, knn_fp32, cost, total_steps, lr, curve[0], git_hash)`; `runs/MUST-09/dinov2_vitb14_reg/<method>/s<seed>/meta.json (git, date_utc)`; `disagreement counts: runs/MUST-09/dinov2_vitb14_reg/<method>/s<seed>/knn/step780_fp32.npz (pred != pred_flip, derived count; identical to FINAL_RESULTS.md P9.2 table)`; `results/MUST-09/summary.csv (same values)`.

### T1b. Per method, mean ± sd over healing seeds

| Method | n seeds | k-NN bf16 | k-NN fp32 | Flip bf16 | Flip fp32 | Params fraction | GFLOPs fraction | Healing steps |
|---|---|---|---|---|---|---|---|---|
| mirror_exact | 3 | 78.340 ± 0.021 | 78.345 ± 0.024 | 99.727 ± 0.015 | 100.000 ± 0.000 | 0.7646 | 0.7740 | 780 |
| svdllm | 3 | 78.061 ± 0.013 | 78.075 ± 0.008 | 95.729 ± 0.086 | 95.747 ± 0.051 | 0.7646 | 0.7740 | 780 |
| slicegpt | 1 | 77.886 | 77.916 | 95.400 | 95.432 | 0.7660 | 0.7754 | 780 |
| svd | 1 | 77.018 | 76.992 | 94.234 | 94.296 | 0.7646 | 0.7740 | 780 |
| random_structure | 1 | 69.934 | 69.964 | 99.340 | 100.000 | 0.7646 | 0.7740 | 780 |
| original | n/a | 78.954 | 78.918 (MUST-07 runner) | 96.280 | 96.302 (MUST-07 runner) | 1.0000 (86,583,552) | 1.0000 (47.08 GFLOPs) | n/a |

- Mirror-exact minus SVD-LLM, seed means: bf16 +0.2793 (PAPER_NUMBERS must09.mirror_minus_best +0.2793); fp32 +0.2707 **(derived)**.
- Mirror-exact (seed mean) minus random structure (seed 0): 8.4060 (PAPER_NUMBERS must09.mirror_minus_random 8.4060).
- Original model: bf16 k-NN 78.954 and flip 96.280 from metrics.json knn_original (cached file `cache/knn_original/dinov2_vitb14_reg_v2_bf16.json` on the VM). The fp32 original numbers come from the MUST-07 runner (`results/MUST-07/rows_dinov2_vitb14_reg.csv`, original row: n_val_fp32 50000, n_bank_fp32 100000); the same runner gives the original bf16 k-NN as 78.938, not 78.954.
- **Paired bootstrap for healed Mirror-exact minus healed SVD-LLM: not available.** No script computed one (searched results/, runs/MUST-09, FINAL_RESULTS.md, scripts/collect_next_phase.py). The per-image predictions needed for it exist: `runs/MUST-09/dinov2_vitb14_reg/{mirror_exact,svdllm}/s{0,1,2}/knn/step780_{bf16,fp32}.npz` (arrays pred, pred_flip, labels, 50,000 each); `mirror/bootstrap.py` is the routine MUST-10 used.
- Healing recipe (meta.json config, identical across methods): 100,000 ImageNet-train images, 2 epochs, batch 256 (micro-batch 64), 780 steps, weight decay 0.05, warm-up 0.05 of steps, crop 224 with scale [0.35, 1.0], horizontal flip p 0.5, flip-symmetrised targets ("sym"). LR chosen from [1e-05, 5e-05, 0.0001] by held-out loss after 100 steps on 5,000 held-out train images; every heal chose 1e-4 (summary.csv lr column).
- Training-curve caution: in `results/MUST-09/curves.csv` steps 0 and 780 use all 50,000 queries, steps 100 and 300 only 10,000 (n_query, full_validation False); `results/paper_next/tables/table1_healed.md` puts them side by side.

### T1c. Latency (MUST-09 healed models, seed 0)

| Model | Form | Batch | Median ms | p10 to p90 ms | Images/s | Throughput / original (derived) | Peak GB |
|---|---|---|---|---|---|---|---|
| original | original | 1 | 2.04 | 2.00 to 2.14 | 491 | 1.000 | 1.40 |
| original | original | 64 | 7.92 | 7.83 to 8.45 | 8077 | 1.000 | 1.53 |
| original | original | 256 | 31.05 | 30.45 to 33.03 | 8244 | 1.000 | 2.08 |
| mirror_exact | healed_as_stored | 1 | 2.05 | 2.02 to 2.18 | 488 | 0.994 | 1.25 |
| mirror_exact | healed_as_stored | 64 | 7.86 | 7.79 to 8.65 | 8147 | 1.009 | 1.44 |
| mirror_exact | healed_as_stored | 256 | 30.54 | 29.97 to 33.15 | 8382 | 1.017 | 2.21 |
| mirror_exact | healed_even_odd | 1 | 3.80 | 3.72 to 3.92 | 263 | 0.536 | 1.12 |
| mirror_exact | healed_even_odd | 64 | 10.91 | 10.90 to 10.92 | 5866 | 0.726 | 1.38 |
| mirror_exact | healed_even_odd | 256 | 40.68 | 40.21 to 41.34 | 6293 | 0.763 | 2.16 |
| svdllm | healed_as_stored | 1 | 3.94 | 3.90 to 3.99 | 254 | 0.518 | 0.55 |
| svdllm | healed_as_stored | 64 | 12.81 | 12.78 to 12.85 | 4997 | 0.619 | 0.73 |
| svdllm | healed_as_stored | 256 | 32.92 | 32.09 to 34.35 | 7776 | 0.943 | 1.34 |
| svd | healed_as_stored | 1 | 4.23 | 4.18 to 4.28 | 236 | 0.482 | 0.48 |
| svd | healed_as_stored | 64 | 12.81 | 12.78 to 12.84 | 4997 | 0.619 | 0.70 |
| svd | healed_as_stored | 256 | 32.85 | 32.07 to 34.53 | 7793 | 0.945 | 1.34 |
| random_structure | healed_as_stored | 1 | 2.01 | 1.97 to 2.16 | 496 | 1.012 | 1.01 |
| random_structure | healed_as_stored | 64 | 7.80 | 7.74 to 8.68 | 8206 | 1.016 | 1.27 |
| random_structure | healed_as_stored | 256 | 30.68 | 30.09 to 32.60 | 8345 | 1.012 | 2.04 |
| random_structure | healed_even_odd | 1 | 3.76 | 3.70 to 3.94 | 266 | 0.542 | 0.89 |
| random_structure | healed_even_odd | 64 | 10.90 | 10.88 to 10.93 | 5872 | 0.727 | 1.14 |
| random_structure | healed_even_odd | 256 | 40.64 | 39.98 to 41.57 | 6299 | 0.764 | 1.92 |

Protocol (latency.json): bf16 weights and inputs, torch.compile mode default, warm-up 20, 100 timed iterations, batches [1, 64, 256]; device NVIDIA H200; measured 2026-11-15 20:11 to 20:15 at commit 805c231c9c. **Caution:** the nvidia-smi line stored with each measurement reads 99% utilisation (mirror_exact: "NVIDIA H200, 99 %, 5989 MiB, 143771 MiB, 1875 MHz"; svdllm: "NVIDIA H200, 99 %, 2875 MiB, 143771 MiB, 1710 MHz"); whether other jobs shared the GPU is not recorded (verify), but other meta files carry nearby times (CORE-05 truth 20:08:43, CORE-06 20:09:34, MUST-10 latency 20:17:01, CORE-05 score 20:21:19 UTC). Single seed-0 model per method, no repeat runs. "original" was timed inside the mirror_exact job. SliceGPT has no latency file (not available). "healed_as_stored" runs the projected weights as dense matrices (params_stored 86,281,728 in summary.csv), so it does not realise the parameter saving; "healed_even_odd" is the efficient even/odd form.

Source: `results/MUST-09/latency.csv`; `runs/MUST-09/dinov2_vitb14_reg/{mirror_exact,svdllm,svd,random_structure}/s0/latency.json`.

## T2. Downstream (MUST-11)

| Variant | Top-1 | Top-5 | Chosen wd | Feature view | Flip-invariant feature | k-NN check | model_params column | Commit | Date |
|---|---|---|---|---|---|---|---|---|---|
| original | 81.004 | 95.972 | 0.00316 | not set (recipe t2v1; 1,536-dim) | n/a | 78.954 | 86,583,552 | 805c231c9c | 2026-11-15 14:28:47 |
| vexact_s0 | 80.104 | 95.558 | 0.00316 | a: even CLS + mean of even-projected patch tokens | True | 78.312 | 85,691,904 | 805c231c9c | 2026-11-15 15:02:42 |
| vexact_s1 | 80.042 | 95.580 | 0.00316 | a: even CLS + mean of even-projected patch tokens | True | 78.352 | 85,691,904 | 805c231c9c | 2026-11-15 15:46:03 |
| vexact_s2 | 80.186 | 95.556 | 0.00316 | a: even CLS + mean of even-projected patch tokens | True | 78.354 | 85,691,904 | 805c231c9c | 2026-11-15 16:46:42 |
| vexact_s0_fullpatch | 80.088 | 95.572 | 0.00316 | b: even CLS + mean of full patch tokens (even projection bypassed) | False | 78.312 | 85,691,904 | 805c231c9c | 2026-11-15 15:18:34 |
| vexact_s1_fullpatch | 80.060 | 95.564 | 0.00316 | b: even CLS + mean of full patch tokens (even projection bypassed) | False | 78.352 | 85,691,904 | 805c231c9c | 2026-11-15 16:10:10 |
| vexact_s2_fullpatch | 80.130 | 95.568 | 0.00316 | b: even CLS + mean of full patch tokens (even projection bypassed) | False | 78.354 | 85,691,904 | 805c231c9c | 2026-11-15 17:08:13 |
| best_svdllm_s0 | 79.716 | 95.286 | 0.001 | not set (recipe t2v1; 1,536-dim) | n/a | 78.048 | 65,349,120 | 805c231c9c | 2026-11-15 17:45:29 |

- Mirror-exact (view a, 3 seeds): 80.111 ± 0.072; minus original -0.893 (PAPER_NUMBERS must11.probe.vexact_mean 80.1107). View b: 80.093 ± 0.035 **(derived)**. SVD-LLM seed 0 minus original -1.288 **(derived)**.
- Probe protocol (CSV): 100000 bank images for training (90000 + 10000 for the weight-decay choice), 50000 validation images, feature dim 1536, wd grid 1e-7,1e-6,1e-5,1e-4,1e-3,1e-2, bf16 features stored as float16.
- Do not cite the model_params column of the MUST-11 CSVs as the model size: it is 85,691,904 for Mirror-exact here, against 66,203,360 in MUST-09 (different counting).

Source: `results/MUST-11/t2/t2_linear.csv`.

| Dataset | Train / test | Original | Mirror-exact s0 (view a) | Mirror-exact s0 (view b) | SVD-LLM s0 | Mirror minus original (derived) | SVD-LLM minus original (derived) | wd at grid edge |
|---|---|---|---|---|---|---|---|---|
| cifar100 | 50000 / 10000 | 91.42 | 90.18 | 90.14 | 89.40 | -1.24 | -2.02 | none |
| dtd | 3760 / 1880 | 83.51 | 83.35 | 83.24 | 82.07 | -0.16 | -1.44 | best_svdllm_s0, vexact_s0 |
| pets | 3680 / 3669 | 95.94 | 95.69 | 95.50 | 95.53 | -0.25 | -0.41 | best_svdllm_s0 |
| flowers102 | 2040 / 6149 | 99.69 | 99.69 | 99.69 | 99.64 | +0.00 | -0.05 | original, best_svdllm_s0, vexact_s0, vexact_s0_fullpatch |

Transfer is linear-probe top-1 (%), seed 0 only (healed seed 0 of each method); no seed spread is available.

Source: `results/MUST-11/t2/t2_transfer.csv`.

| Variant | mIoU | aAcc | mAcc | Mask equivariance raw (%) | Mask equivariance symmetrised (%) | Pixels differing raw / sym | Sym logit diff (rel.) | Operator used |
|---|---|---|---|---|---|---|---|---|
| original | 47.59 | 80.98 | 59.47 | 95.0309 | 94.74133 | 24,232,674 / 25,644,765 | 0.801 | A_L: last-layer per-layer operator (layer 12) |
| vexact_s0 | 47.16 | 81.02 | 58.89 | 98.9771 | 99.99991 | 4,988,566 / 422 | 2.45e-05 | A* of the model's even head (full tokens: exactly A*-equivariant before the final affine) |
| best_svdllm_s0 | 46.79 | 80.87 | 58.57 | 94.7590 | 94.62487 | 25,558,433 / 26,212,718 | 0.615 | A_L: last-layer per-layer operator (layer 12) |

- Mirror-exact s0 minus original mIoU -0.43; SVD-LLM minus original -0.80 **(derived)**.
- Recipe (CSV): linear head, 8000 iterations, batch 16, crop 448, lr 0.001, wd 0.0001, 20210 train / 2000 val images; mask equivariance over 487,666,546 pixels of 2000 images in fp32 (no autocast, TF32 off). The original run resumed from iteration 5000. Seed 0 only. The original and SVD-LLM rows are measured with the per-layer operator A_L of layer 12, Mirror-exact with its own A*.
- P11.2 stays PENDING: 422 pixels differ under the symmetrised head and their top-two margins were not recorded (`FINAL_RESULTS.md`, MUST-11 P11.2). PAPER_NUMBERS rounds the symmetrised value to "100.00"; the raw value is 99.99991.

Source: `results/MUST-11/t2_seg/t2_ade20k.csv`; `results/MUST-11/t2_seg/seg/dinov2_vitb14_reg/{original,vexact_s0,best_svdllm_s0}_maskeq.json (same values)`.

## T3. Flip averaging (MUST-10), INT8 and ToMe (UP-06)

| Model | Mode | Passes | k-NN bf16 | k-NN fp32 | Flip bf16 | Flip fp32 | Probe top-1 | Probe top-5 | GFLOPs/image | Params (fraction) |
|---|---|---|---|---|---|---|---|---|---|---|
| original | plain | 1 | 78.954 | n/a | 96.280 | n/a | 81.004 | 95.972 | 47.08 | 86,583,552 (1.0000) |
| original | flipavg | 2 | 79.000 | 79.008 | 100.000 | 100.000 | 81.126 | 95.990 | 94.16 | 86,583,552 (1.0000) |
| healed | plain | 1 | 78.316 | 78.372 | 99.744 | 100.000 | 80.102 | 95.562 | 36.44 | 66,203,360 (0.7646) |

- Healed seed 0 minus flip-averaged original: k-NN -0.684, 95% paired bootstrap [-0.868, -0.506] (50000 images, 10000 resamples, seed 0); probe -1.024 [-1.200, -0.852]; GFLOPs ratio 0.3870; parameter ratio 0.7646.
- Scored P10.1 uses the 3-seed mean: -0.6600 points (PAPER_NUMBERS must10.mirror_minus_flipavg). No bootstrap exists for the 3-seed version.
- The healed probe row is seed 0 only (healed_run column).

Source: `results/MUST-10/flipavg.csv`; `results/MUST-10/comparison_dinov2_vitb14_reg_s0.json`; `results/MUST-10/MUST10.md`.

| Form | Batch | Median ms | Images/s | Throughput / original plain (derived) |
|---|---|---|---|---|
| original_plain | 1 | 2.06 | 485 | 1.000 |
| original_plain | 64 | 7.87 | 8137 | 1.000 |
| original_plain | 256 | 31.07 | 8240 | 1.000 |
| original_flipavg_batched | 1 | 2.13 | 470 | 0.969 |
| original_flipavg_batched | 64 | 15.16 | 4220 | 0.519 |
| original_flipavg_batched | 256 | 60.92 | 4202 | 0.510 |
| original_flipavg_two_passes | 1 | 3.87 | 259 | 0.533 |
| original_flipavg_two_passes | 64 | 15.78 | 4056 | 0.498 |
| original_flipavg_two_passes | 256 | 62.54 | 4094 | 0.497 |
| healed_as_stored | 1 | 1.98 | 504 | 1.040 |
| healed_as_stored | 64 | 7.84 | 8167 | 1.004 |
| healed_as_stored | 256 | 30.82 | 8306 | 1.008 |
| healed_even_odd | 1 | 3.74 | 267 | 0.551 |
| healed_even_odd | 64 | 10.85 | 5897 | 0.725 |
| healed_even_odd | 256 | 40.71 | 6288 | 0.763 |

Same protocol and caveat as T1c (bf16, compiled, H200, 99% utilisation recorded; commit 805c231c9c; measured from 20:17:01 UTC per meta.json).

Source: `results/MUST-10/latency.csv`; `results/MUST-10/meta.json`.

| Variant | k-NN bf16 | Flip bf16 | Disagreements bf16 | k-NN fp32 | Flip fp32 | Disagreements fp32 | Tokens after last block |
|---|---|---|---|---|---|---|---|
| original | 78.954 | 96.280 | 1860 | n/a | n/a | n/a | 261 |
| original+int8 | 78.950 | 96.266 | 1867 | n/a | n/a | n/a | 261 |
| original+tome | 78.618 | 95.036 | 2482 | n/a | n/a | n/a | 165 |
| mirror_exact | 78.316 | 99.744 | 128 | 78.372 | 100.000 | 0 | 261 |
| mirror_exact+int8 | 78.310 | 99.104 | 448 | 78.312 | 99.090 | 455 | 261 |
| mirror_exact+tome | 78.138 | 96.838 | 1581 | 78.154 | 96.798 | 1601 | 165 |

| Form | Extra drop, original | Extra drop, Mirror-exact | Difference | 95% CI of difference | 95% CI drop original | 95% CI drop Mirror-exact | Within 0.5 |
|---|---|---|---|---|---|---|---|
| int8 | 0.004 | 0.006 | +0.002 | [-0.102, 0.102] | [-0.054, 0.062] | [-0.080, 0.090] | True |
| tome | 0.336 | 0.178 | -0.158 | [-0.338, 0.024] | [0.202, 0.470] | [0.046, 0.314] | True |

Bar 0.5 points on the point estimate; prediction holds: True. One healed model (seed 0). INT8 is simulated weight-only per output channel (48 matrices); ToMe r = 8 per block, 261 to 165 tokens; neither is mirror-equivariant, so flip consistency drops (summary.md caveats). Date 2026-11-15T11:46:31Z, commit e84e20ff35.

Source: `results/UP-06/rows.csv`; `results/UP-06/comparison.json`; `results/UP-06/summary.md`.

## T4. Zero-shot compressors at matched parameters (MUST-07, MUST-08)

### T4a. dinov2_vitb14_reg (sizes: 6mlp = blocks 1 to 6 MLP; 6both = blocks 1 to 6 MLP + attention; 9both; full = 12 blocks MLP + attention)

| Size | # | Method | n fit seeds | Val k-NN mean | sd | msel top-1 (seed 0) | Flip bf16 (seed 0) | Flip fp32 (n queries) | Params in span (seed 0) | Gap to budget | Model params (fraction), seed 0 | GFLOPs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| original | 0 | original | 1 | 78.94 | n/a | 80.21 | 96.32 | 96.30 (50000) | 84,934,656 | n/a | 86,583,552 (1.000) | 47.1 |
| 6mlp | 1 | Mirror-exact | 3 | 76.60 | 0.02 | 78.05 | 91.70 | n/a | 17,034,432 | 0.0000% | 75,306,432 (0.870) | 41.2 |
| 6mlp | 2 | Mirror-local (ablation) | 1 | 77.12 | n/a | 78.61 | 92.35 | n/a | 20,001,574 | 17.4185% | 78,273,574 (0.904) | 42.7 |
| 6mlp | 3 | plain SVD | 3 | 9.37 | 0.00 | 10.60 | 42.91 | n/a | 17,038,080 | 0.0214% | 75,310,080 (0.870) | 41.2 |
| 6mlp | 4 | SVD-LLM-style | 3 | 78.35 | 0.03 | 79.58 | 95.43 | n/a | 17,038,080 | 0.0214% | 75,310,080 (0.870) | 41.2 |
| 6mlp | 5 | ASVD-style | 3 | 76.81 | 0.04 | 77.60 | 93.22 | n/a | 17,038,080 | 0.0214% | 75,310,080 (0.870) | 41.2 |
| 6mlp | 6 | SliceGPT-style | 3 | 31.41 | 1.23 | 33.53 | 41.87 | n/a | 17,068,032 | 0.1972% | 75,340,032 (0.870) | 41.2 |
| 6mlp | 7 | FLAP-style | 3 | 75.20 | 0.07 | 76.85 | 92.72 | n/a | 17,035,776 | 0.0079% | 75,307,776 (0.870) | 41.2 |
| 6mlp | 8 | ZipIt-style | 3 | 75.67 | 0.15 | 77.48 | 92.08 | n/a | 17,035,776 | 0.0079% | 75,307,776 (0.870) | 41.2 |
| 6mlp | 9 | pairing by same-image corr. | 3 | 27.42 | 0.81 | 29.10 | 40.80 | n/a | 17,034,432 | 0.0000% | 75,306,432 (0.870) | 41.2 |
| 6mlp | 10 | pairing by weight cosine | 3 | 34.42 | 0.36 | 35.14 | 47.26 | n/a | 17,034,432 | 0.0000% | 75,306,432 (0.870) | 41.2 |
| 6mlp | 11 | random pairing | 3 | 22.40 | 0.69 | 24.06 | 34.51 | n/a | 17,034,432 | 0.0000% | 75,306,432 (0.870) | 41.2 |
| 6mlp | 12 | random structure | 3 | 2.31 | 0.50 | 2.56 | 8.69 | n/a | 17,034,432 | 0.0000% | 75,306,432 (0.870) | 41.2 |
| 6both | 1 | Mirror-exact | 3 | 77.34 | 0.08 | 78.83 | 95.02 | n/a | 25,668,000 | 0.0000% | 69,784,224 (0.806) | 38.3 |
| 6both | 2 | Mirror-local (ablation) | 1 | 65.90 | n/a | 67.46 | 79.90 | n/a | 28,440,198 | 10.8002% | 72,556,422 (0.838) | 39.8 |
| 6both | 3 | plain SVD | 3 | 5.87 | 0.03 | 6.41 | 29.79 | n/a | 25,668,096 | 0.0004% | 69,784,320 (0.806) | 38.3 |
| 6both | 4 | SVD-LLM-style | 3 | 78.02 | 0.01 | 79.49 | 95.16 | n/a | 25,668,096 | 0.0004% | 69,784,320 (0.806) | 38.3 |
| 6both | 5 | ASVD-style | 3 | 74.41 | 0.02 | 75.89 | 90.69 | n/a | 25,668,096 | 0.0004% | 69,784,320 (0.806) | 38.3 |
| 6both | 6 | SliceGPT-style | 3 | 75.60 | 0.11 | 76.61 | 91.94 | n/a | 25,729,836 | 0.2409% | 69,846,060 (0.807) | 38.3 |
| 6both | 7 | FLAP-style | 3 | 70.29 | 0.18 | 71.94 | 88.89 | n/a | 25,668,096 | 0.0004% | 69,784,320 (0.806) | 37.8 |
| 6both | 8 | ZipIt-style | 3 | 18.51 | 1.20 | 21.13 | 40.30 | n/a | 25,668,096 | 0.0004% | 69,784,320 (0.806) | 37.8 |
| 6both | 9 | pairing by same-image corr. | 3 | 31.31 | 1.53 | 33.43 | 54.50 | n/a | 25,668,000 | 0.0000% | 69,784,224 (0.806) | 38.3 |
| 6both | 10 | pairing by weight cosine | 3 | 39.85 | 0.32 | 41.19 | 62.55 | n/a | 25,668,000 | 0.0000% | 69,784,224 (0.806) | 38.3 |
| 6both | 11 | random pairing | 3 | 32.30 | 3.30 | 35.92 | 56.87 | n/a | 25,668,000 | 0.0000% | 69,784,224 (0.806) | 38.3 |
| 6both | 12 | random structure | 3 | 2.29 | 0.42 | 2.87 | 13.73 | n/a | 25,668,000 | 0.0000% | 69,784,224 (0.806) | 38.3 |
| 9both | 1 | Mirror-exact | 3 | 73.37 | 0.11 | 74.94 | 94.03 | n/a | 43,442,640 | 0.0000% | 66,325,200 (0.766) | 36.5 |
| 9both | 2 | Mirror-local (ablation) | 1 | 62.26 | n/a | 63.66 | 76.81 | n/a | 47,273,228 | 8.8176% | 70,155,788 (0.810) | 38.5 |
| 9both | 3 | plain SVD | 1 | 5.67 | n/a | 6.16 | 27.24 | n/a | 43,442,688 | 0.0001% | 66,325,248 (0.766) | 36.5 |
| 9both | 4 | SVD-LLM-style | 1 | 77.35 | n/a | 78.77 | 94.26 | n/a | 43,442,688 | 0.0001% | 66,325,248 (0.766) | 36.5 |
| 9both | 5 | ASVD-style | 1 | 65.24 | n/a | 67.75 | 80.67 | n/a | 43,442,688 | 0.0001% | 66,325,248 (0.766) | 36.5 |
| 9both | 6 | SliceGPT-style | 1 | 72.48 | n/a | 73.92 | 88.57 | n/a | 43,522,176 | 0.1831% | 66,404,736 (0.767) | 36.5 |
| 9both | 7 | FLAP-style | 1 | 63.47 | n/a | 65.33 | 83.33 | n/a | 43,442,688 | 0.0001% | 66,325,248 (0.766) | 35.9 |
| 9both | 8 | ZipIt-style | 1 | 13.38 | n/a | 14.63 | 45.98 | n/a | 43,442,688 | 0.0001% | 66,325,248 (0.766) | 35.9 |
| 9both | 9 | pairing by same-image corr. | 1 | 13.36 | n/a | 14.24 | 62.80 | n/a | 43,442,640 | 0.0000% | 66,325,200 (0.766) | 36.5 |
| 9both | 10 | pairing by weight cosine | 1 | 16.94 | n/a | 18.61 | 62.87 | n/a | 43,442,640 | 0.0000% | 66,325,200 (0.766) | 36.5 |
| 9both | 11 | random pairing | 1 | 11.07 | n/a | 12.23 | 56.40 | n/a | 43,442,640 | 0.0000% | 66,325,200 (0.766) | 36.5 |
| 9both | 12 | random structure | 1 | 4.31 | n/a | 4.67 | 76.77 | n/a | 43,442,640 | 0.0000% | 66,325,200 (0.766) | 36.5 |
| full | 1 | Mirror-exact | 3 | 53.81 | 0.36 | 55.29 | 97.98 | 100.00 (50000) | 64,554,464 | 0.0000% | 66,203,360 (0.765) | 36.4 |
| full | 2 | Mirror-local (ablation) | 1 | 62.10 | n/a | 63.26 | 76.43 | 74.22 (5000) | 69,374,408 | 7.4665% | 71,023,304 (0.820) | 39.0 |
| full | 3 | plain SVD | 1 | 5.17 | n/a | 5.67 | 25.82 | 31.54 (5000) | 64,555,008 | 0.0008% | 66,203,904 (0.765) | 36.4 |
| full | 4 | SVD-LLM-style | 1 | 75.80 | n/a | 77.36 | 93.81 | 92.10 (5000) | 64,555,008 | 0.0008% | 66,203,904 (0.765) | 36.4 |
| full | 5 | ASVD-style | 1 | 57.09 | n/a | 59.70 | 74.76 | 69.96 (5000) | 64,555,008 | 0.0008% | 66,203,904 (0.765) | 36.4 |
| full | 6 | SliceGPT-style | 1 | 68.33 | n/a | 69.53 | 86.89 | 85.42 (5000) | 64,675,800 | 0.1880% | 66,324,696 (0.766) | 36.5 |
| full | 7 | FLAP-style | 1 | 65.12 | n/a | 67.00 | 85.90 | 84.62 (5000) | 64,555,008 | 0.0008% | 66,203,904 (0.765) | 35.9 |
| full | 8 | ZipIt-style | 1 | 13.58 | n/a | 15.17 | 42.08 | 45.62 (5000) | 64,555,008 | 0.0008% | 66,203,904 (0.765) | 35.9 |
| full | 9 | pairing by same-image corr. | 1 | 12.17 | n/a | 13.88 | 94.57 | 100.00 (5000) | 64,554,464 | 0.0000% | 66,203,360 (0.765) | 36.4 |
| full | 10 | pairing by weight cosine | 1 | 14.98 | n/a | 16.48 | 94.53 | 100.00 (5000) | 64,554,464 | 0.0000% | 66,203,360 (0.765) | 36.4 |
| full | 11 | random pairing | 1 | 10.02 | n/a | 11.31 | 92.82 | 100.00 (5000) | 64,554,464 | 0.0000% | 66,203,360 (0.765) | 36.4 |
| full | 12 | random structure | 1 | 4.78 | n/a | 5.08 | 93.44 | 100.00 (5000) | 64,554,464 | 0.0000% | 66,203,360 (0.765) | 36.4 |

Seeds: Mirror-exact has fit seeds 0 to 2 at every size; the other methods have 3 fit seeds at 6mlp and 6both and seed 0 only at 9both and full. The fp32 flip pass used 50,000 queries for full-size Mirror-exact and the original, 5,000 queries (10,000-image bank) for the other full-size rows (n_val_fp32 / n_bank_fp32 columns). Model params of Mirror-exact differ slightly between fit seeds (75,306,432 / 75,285,460 / 75,305,760 at 6mlp, MUST-08 rows); the table shows seed 0.

Source: `results/MUST-07/rows_dinov2_vitb14_reg.csv (means and sd derived from its per-seed rows; they match results/MUST-07/SUMMARY.md)`.

### T4b. Paired bootstrap, Mirror-exact minus method, fit seed 0 (50,000 images, 10,000 resamples), dinov2_vitb14_reg

| Method | 6mlp | 6both | 9both | full |
|---|---|---|---|---|
| Mirror-local (ablation) | -0.53 [-0.74, -0.32] | +11.51 [11.16, 11.86] | +11.09 [10.72, 11.45] | -8.22 [-8.64, -7.82] |
| plain SVD | +67.23 [66.80, 67.64] | +71.57 [71.16, 71.97] | +67.68 [67.25, 68.11] | +48.71 [48.24, 49.16] |
| SVD-LLM-style | -1.74 [-1.97, -1.52] | -0.62 [-0.82, -0.42] | -4.00 [-4.28, -3.72] | -21.92 [-22.35, -21.50] |
| ASVD-style | -0.26 [-0.49, -0.04] | +2.98 [2.71, 3.24] | +8.11 [7.75, 8.46] | -3.21 [-3.65, -2.78] |
| SliceGPT-style | +44.50 [44.03, 44.97] | +1.69 [1.44, 1.94] | +0.87 [0.57, 1.18] | -14.45 [-14.90, -14.02] |
| FLAP-style | +1.46 [1.22, 1.71] | +7.25 [6.93, 7.58] | +9.88 [9.51, 10.24] | -11.24 [-11.67, -10.82] |
| ZipIt-style | +0.82 [0.58, 1.05] | +57.93 [57.47, 58.39] | +59.97 [59.52, 60.41] | +40.30 [39.83, 40.75] |
| pairing by same-image corr. | +49.01 [48.53, 49.47] | +45.49 [45.01, 45.95] | +59.99 [59.54, 60.44] | +41.71 [41.24, 42.16] |
| pairing by weight cosine | +42.58 [42.12, 43.05] | +37.35 [36.89, 37.81] | +56.41 [55.95, 56.86] | +38.90 [38.44, 39.35] |
| random pairing | +53.65 [53.18, 54.12] | +42.80 [42.33, 43.27] | +62.28 [61.83, 62.71] | +43.86 [43.39, 44.31] |
| random structure | +74.34 [73.95, 74.72] | +74.67 [74.28, 75.06] | +69.04 [68.63, 69.46] | +49.10 [48.64, 49.54] |

P7.1 (bar: at least +2.0 on the 3-seed mean with CI above 0, every method, both 6-block sizes) fails; smallest margin -1.75 (SVD-LLM, 6mlp). P7.2 (ZipIt and same-image pairing trail by at least 10) fails at 6mlp: ZipIt margin 0.934 (`FINAL_RESULTS.md`, MUST-07 P7.1 and P7.2 tables).

Source: `results/MUST-07/bootstrap.csv`.

### T4c. Gate B ranking (full size, fit seed 0, method-selection k-NN only)

| Rank | Method | msel top-1 | Chosen hyperparameters | Gap |
|---|---|---|---|---|
| 1 | SVD-LLM-style | 77.36 | {"damp": 1e-06} | 0.0008% |
| 2 | SliceGPT-style | 69.53 | {"basis": "per_block", "pca": "stream"} | 0.1880% |
| 3 | FLAP-style | 67.00 | {"allocation": "uniform"} | 0.0008% |
| 4 | ASVD-style | 59.70 | {"alpha": 1.0} | 0.0008% |
| 5 | pairing by weight cosine | 16.48 | {} | 0.0000% |
| 6 | ZipIt-style | 15.17 | {"alpha": 0.5} | 0.0008% |
| 7 | pairing by same-image corr. | 13.88 | {} | 0.0000% |
| 8 | random pairing | 11.31 | {} | 0.0000% |
| 9 | plain SVD | 5.67 | {} | 0.0008% |
| 10 | random structure | 5.08 | {} | 0.0000% |

Source: `results/MUST-07/gate_b.json`.

### T4d. MUST-08: MLP-only against MLP + attention (Mirror-exact, dinov2_vitb14_reg, fit seeds 0 to 2)

| Size | Val k-NN seeds 0 / 1 / 2 | Mean ± sd | msel (s0) | Flip bf16 (s0) | Flip fp32 (n) | Params in span (s0) | Model params, seed mean | Saving vs original | Saving / original params (derived) |
|---|---|---|---|---|---|---|---|---|---|
| 6mlp | 76.59 / 76.60 / 76.62 | 76.60 ± 0.02 | 78.05 | 91.70 | n/a | 17,034,432 | 75,299,217 | 11,284,335 | 0.1303 |
| 6attn | 77.40 / 77.44 / 77.42 | 77.42 ± 0.02 | 78.92 | 92.54 | n/a | 9,223,392 | 81,646,967 | 4,936,585 | 0.0570 |
| 6both | 77.41 / 77.26 / 77.35 | 77.34 ± 0.08 | 78.83 | 95.02 | n/a | 25,668,000 | 69,772,808 | 16,810,744 | 0.1942 |
| fullmlp | 52.53 / 52.57 / 52.56 | 52.55 ± 0.02 | 54.28 | 64.67 | 60.94 (5000) | 42,693,344 | 72,634,001 | 13,949,551 | 0.1611 |
| fullattn | 62.67 / 62.14 / 62.40 | 62.40 ± 0.26 | 64.64 | 77.27 | 73.72 (5000) | 21,861,120 | 80,122,891 | 6,460,661 | 0.0746 |
| full | 53.88 / 53.42 / 54.13 | 53.81 ± 0.36 | 55.29 | 97.98 | 100.00 (50000) | 64,554,464 | 66,173,340 | 20,410,212 | 0.2357 |

- P8.1 saving ratios: 6 blocks 0.6713, full model 0.6835 (bar > 0.5, PASS). P8.2: 6mlp minus 6both -0.7360 points (bar <= 0.5, PASS). Original model params 86,583,552.

Source: `results/MUST-08/rows_dinov2_vitb14_reg.csv`; `results/MUST-08/SUMMARY.md`.

### T4e. MUST-07 on DINOv2-L/14 reg (6mlp = blocks 1 to 6 of 24; full = 24 blocks), fit seed 0 only

| Size | # | Method | n fit seeds | Val k-NN mean | sd | msel top-1 (seed 0) | Flip bf16 (seed 0) | Flip fp32 (n queries) | Params in span (seed 0) | Gap to budget | Model params (fraction), seed 0 | GFLOPs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| original | 0 | original | 1 | 80.59 | n/a | 81.89 | 97.74 | n/a | 301,989,888 | n/a | 304,372,736 (1.000) | 164.6 |
| 6mlp | 1 | Mirror-exact | 1 | 80.36 | n/a | 81.67 | 96.94 | n/a | 29,536,180 | 0.0000% | 283,577,268 (0.932) | 153.8 |
| 6mlp | 2 | Mirror-local (ablation) | 1 | 80.21 | n/a | 81.40 | 95.23 | n/a | 34,215,230 | 15.8418% | 288,256,318 (0.947) | 156.2 |
| 6mlp | 3 | plain SVD | 1 | 80.57 | n/a | 81.75 | 97.49 | n/a | 29,537,280 | 0.0037% | 283,578,368 (0.932) | 153.8 |
| 6mlp | 4 | SVD-LLM-style | 1 | 80.54 | n/a | 81.95 | 97.79 | n/a | 29,537,280 | 0.0037% | 283,578,368 (0.932) | 153.8 |
| 6mlp | 6 | SliceGPT-style | 1 | 80.47 | n/a | 81.60 | 96.94 | n/a | 29,556,736 | 0.0696% | 283,597,824 (0.932) | 153.8 |
| 6mlp | 12 | random structure | 1 | 2.29 | n/a | 2.42 | 5.64 | n/a | 29,536,180 | 0.0000% | 283,577,268 (0.932) | 153.8 |
| full | 1 | Mirror-exact | 1 | 30.17 | n/a | 31.51 | 95.91 | 100.00 (5000) | 246,026,620 | 0.0000% | 248,409,468 (0.816) | 135.4 |
| full | 2 | Mirror-local (ablation) | 1 | 70.18 | n/a | 71.53 | 83.06 | n/a | 255,060,290 | 3.6718% | 257,443,138 (0.846) | 140.1 |
| full | 3 | plain SVD | 1 | 60.62 | n/a | 63.19 | 75.92 | n/a | 246,030,336 | 0.0015% | 248,413,184 (0.816) | 135.4 |
| full | 4 | SVD-LLM-style | 1 | 80.54 | n/a | 81.79 | 97.30 | n/a | 246,030,336 | 0.0015% | 248,413,184 (0.816) | 135.4 |
| full | 6 | SliceGPT-style | 1 | 79.48 | n/a | 80.89 | 94.95 | n/a | 246,251,520 | 0.0914% | 248,634,368 (0.817) | 135.5 |
| full | 12 | random structure | 1 | 3.87 | n/a | 3.88 | 91.74 | 100.00 (5000) | 246,026,620 | 0.0000% | 248,409,468 (0.816) | 135.4 |

| Size | Mirror-exact minus | Difference | 95% CI | Excludes 0 |
|---|---|---|---|---|
| 6mlp | Mirror-local (ablation) | +0.15 | [-0.00, 0.31] | False |
| 6mlp | plain SVD | -0.21 | [-0.34, -0.07] | True |
| 6mlp | SVD-LLM-style | -0.17 | [-0.30, -0.04] | True |
| 6mlp | SliceGPT-style | -0.11 | [-0.25, 0.04] | False |
| 6mlp | random structure | +78.07 | [77.70, 78.44] | True |
| full | Mirror-local (ablation) | -40.01 | [-40.48, -39.55] | True |
| full | plain SVD | -30.45 | [-30.92, -29.98] | True |
| full | SVD-LLM-style | -50.37 | [-50.84, -49.91] | True |
| full | SliceGPT-style | -49.31 | [-49.77, -48.84] | True |
| full | random structure | +26.30 | [25.88, 26.71] | True |

Source: `results/MUST-07/rows_dinov2_vitl14_reg.csv`; `results/MUST-07/bootstrap.csv`.

### T4e. MUST-07 on DeiT-III B/16, fit seed 0 only

| Size | # | Method | n fit seeds | Val k-NN mean | sd | msel top-1 (seed 0) | Flip bf16 (seed 0) | Flip fp32 (n queries) | Params in span (seed 0) | Gap to budget | Model params (fraction), seed 0 | GFLOPs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| original | 0 | original | 1 | 82.97 | n/a | 92.02 | 96.38 | n/a | 84,934,656 | n/a | 86,585,320 (1.000) | 35.1 |
| 6mlp | 1 | Mirror-exact | 1 | 81.06 | n/a | 90.79 | 94.81 | n/a | 19,537,952 | 0.0000% | 77,811,720 (0.899) | 31.7 |
| 6mlp | 2 | Mirror-local (ablation) | 1 | 81.56 | n/a | 91.08 | 94.82 | n/a | 21,977,930 | 12.4884% | 80,251,698 (0.927) | 32.6 |
| 6mlp | 3 | plain SVD | 1 | 81.29 | n/a | 90.75 | 94.99 | n/a | 19,541,760 | 0.0195% | 77,815,528 (0.899) | 31.7 |
| 6mlp | 4 | SVD-LLM-style | 1 | 82.81 | n/a | 92.00 | 96.12 | n/a | 19,541,760 | 0.0195% | 77,815,528 (0.899) | 31.7 |
| 6mlp | 6 | SliceGPT-style | 1 | 76.19 | n/a | 86.03 | 91.08 | n/a | 19,574,784 | 0.1885% | 77,848,552 (0.899) | 31.7 |
| 6mlp | 12 | random structure | 1 | 6.00 | n/a | 6.67 | 27.55 | n/a | 19,537,952 | 0.0000% | 77,811,720 (0.899) | 31.7 |
| full | 1 | Mirror-exact | 1 | 72.85 | n/a | 82.94 | 99.26 | 100.00 (5000) | 70,562,208 | 0.0000% | 72,212,872 (0.834) | 29.5 |
| full | 2 | Mirror-local (ablation) | 1 | 81.20 | n/a | 90.81 | 94.71 | n/a | 75,764,794 | 7.3730% | 77,415,458 (0.894) | 31.5 |
| full | 3 | plain SVD | 1 | 74.73 | n/a | 82.24 | 91.59 | n/a | 70,562,304 | 0.0001% | 72,212,968 (0.834) | 29.5 |
| full | 4 | SVD-LLM-style | 1 | 80.54 | n/a | 88.90 | 95.26 | n/a | 70,562,304 | 0.0001% | 72,212,968 (0.834) | 29.5 |
| full | 6 | SliceGPT-style | 1 | 79.50 | n/a | 88.03 | 94.68 | n/a | 70,697,112 | 0.1912% | 72,347,776 (0.836) | 29.5 |
| full | 12 | random structure | 1 | 12.52 | n/a | 13.74 | 96.19 | 100.00 (5000) | 70,562,208 | 0.0000% | 72,212,872 (0.834) | 29.5 |

| Size | Mirror-exact minus | Difference | 95% CI | Excludes 0 |
|---|---|---|---|---|
| 6mlp | Mirror-local (ablation) | -0.50 | [-0.67, -0.33] | True |
| 6mlp | plain SVD | -0.24 | [-0.45, -0.02] | True |
| 6mlp | SVD-LLM-style | -1.75 | [-1.95, -1.55] | True |
| 6mlp | SliceGPT-style | +4.86 | [4.59, 5.13] | True |
| 6mlp | random structure | +75.05 | [74.66, 75.43] | True |
| full | Mirror-local (ablation) | -8.35 | [-8.66, -8.04] | True |
| full | plain SVD | -1.88 | [-2.20, -1.56] | True |
| full | SVD-LLM-style | -7.70 | [-8.01, -7.38] | True |
| full | SliceGPT-style | -6.65 | [-6.97, -6.33] | True |
| full | random structure | +60.33 | [59.88, 60.78] | True |

Source: `results/MUST-07/rows_deit3_b16.csv`; `results/MUST-07/bootstrap.csv`.

## T5. Structure across models (MUST-04, CORE-04)

### T5a. MUST-04: ratio to null and twin fraction (threshold 0.5) by group and range, fit seed 0

| Model | Group | Ratio 1-6 | Twin 1-6 | Ratio 7-12 | Twin 7-12 | Ratio first half | Twin first half | Ratio second half | Twin second half | R90 share in 4-cycles 1-6 | Tax (MLP) first half |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov2_vitb14_reg | LR | 0.4791 | 0.5677 | 0.3061 | 0.1197 | 0.4791 | 0.5677 | 0.3061 | 0.1197 |  | 0.4185 |
| dinov2_vitb14_reg | UD | 0.5129 | 0.5600 | 0.6385 | 0.0908 | 0.5129 | 0.5600 | 0.6385 | 0.0908 |  | 0.4251 |
| dinov2_vitb14_reg | R180 | 0.5272 | 0.6228 | 0.6045 | 0.1228 | 0.5272 | 0.6228 | 0.6045 | 0.1228 |  | 0.4559 |
| dinov2_vitb14_reg | R90 | 0.8494 | 0.0474 | 0.8074 | 0.0302 | 0.8494 | 0.0474 | 0.8074 | 0.0302 | 0.1367 | 0.5975 |
| dinov2_vitb14 | LR | 0.4474 | 0.5374 | 0.2773 | 0.0878 | 0.4474 | 0.5374 | 0.2773 | 0.0878 |  | 0.4048 |
| dinov2_vitb14 | UD | 0.4956 | 0.5298 | 0.6518 | 0.0595 | 0.4956 | 0.5298 | 0.6518 | 0.0595 |  | 0.4171 |
| dinov2_vitb14 | R180 | 0.5081 | 0.5732 | 0.5998 | 0.0857 | 0.5081 | 0.5732 | 0.5998 | 0.0857 |  | 0.4465 |
| dinov2_vitb14 | R90 | 0.8366 | 0.0543 | 0.7610 | 0.0241 | 0.8366 | 0.0543 | 0.7610 | 0.0241 | 0.1395 | 0.5929 |
| dinov2_vitl14_reg | LR | 0.5020 | 0.5432 | 0.5268 | 0.4630 | 0.5135 | 0.5031 | 0.3388 | 0.0565 |  | 0.4140 |
| dinov2_vitl14_reg | UD | 0.5114 | 0.5458 | 0.5930 | 0.4242 | 0.5497 | 0.4850 | 0.7036 | 0.0392 |  | 0.4220 |
| dinov2_vitl14_reg | R180 | 0.5287 | 0.5917 | 0.6045 | 0.4749 | 0.5651 | 0.5333 | 0.6605 | 0.0564 |  | 0.4521 |
| dinov2_vitl14_reg | R90 | 0.7555 | 0.0182 | 0.9152 | 0.0586 | 0.8287 | 0.0384 | 0.8049 | 0.0137 | 0.1160 | 0.5989 |
| deit3_b16 | LR | 0.4501 | 0.4301 | 0.3149 | 0.0058 | 0.4501 | 0.4301 | 0.3149 | 0.0058 |  | 0.3487 |
| deit3_b16 | UD | 0.4807 | 0.4630 | 0.7086 | 0.0205 | 0.4807 | 0.4630 | 0.7086 | 0.0205 |  | 0.3897 |
| deit3_b16 | R180 | 0.4807 | 0.5047 | 0.7017 | 0.0196 | 0.4807 | 0.5047 | 0.7017 | 0.0196 |  | 0.4146 |
| deit3_b16 | R90 | 0.7536 | 0.0862 | 0.7599 | 0.0079 | 0.7536 | 0.0862 | 0.7599 | 0.0079 | 0.1187 | 0.5755 |

First half = blocks 1 to 6 for the 12-block models and 1 to 12 for dinov2_vitl14_reg (24 blocks); second half = the rest.

| Model | UD twin fraction 1-6 | Twin-null mean | Twin-null 95% interval | Null draws | Inside null |
|---|---|---|---|---|---|
| dinov2_vitb14_reg | 0.5600 | 0.000163 | [0.00000, 0.00060] | 20 | False |
| dinov2_vitb14 | 0.5298 | 0.00234 | [0.00163, 0.00349] | 20 | False |
| dinov2_vitl14_reg | 0.5458 | 7.73e-05 | [0.00000, 0.00037] | 20 | False |
| deit3_b16 | 0.4630 | 0 | [0.00000, 0.00000] | 20 | False |

| Model | Surgery (MLP, per-layer operators) | Original k-NN | k-NN after | Drop | Flip after | MLP params kept | UD minus LR, 95% CI |
|---|---|---|---|---|---|---|---|
| dinov2_vitb14_reg | LR_1-6 | 78.954 | 77.110 | 1.844 | 92.36 | 0.5815 |  |
| dinov2_vitb14_reg | UD_1-6 | 78.954 | 75.324 | 3.630 | 90.88 | 0.5749 | -1.786 [-2.028, -1.546] |
| dinov2_vitb14 | LR_1-6 | 78.694 | 74.686 | 4.008 | 90.22 | 0.5952 |  |
| dinov2_vitb14 | UD_1-6 | 78.694 | 72.618 | 6.076 | 87.95 | 0.5829 | -2.068 [-2.320, -1.816] |
| dinov2_vitl14_reg | LR_1-6 | 80.576 | 80.208 | 0.368 | 95.23 | 0.5548 |  |
| dinov2_vitl14_reg | UD_1-6 | 80.576 | 80.344 | 0.232 | 97.18 | 0.5513 | +0.136 [-0.016, 0.290] |
| deit3_b16 | LR_1-6 | 82.964 | 81.574 | 1.390 | 94.84 | 0.6513 |  |
| deit3_b16 | UD_1-6 | 82.964 | 80.930 | 2.034 | 94.13 | 0.6103 | -0.644 [-0.836, -0.454] |

Source: `results/MUST-04/<model>/metrics.json (scores, groups.UD.ranges.1-6.twin_null, surgery)`; `per-block values: results/MUST-04/<model>/blocks.csv`.

### T5b. CORE-04: LR and UD structure in 14 models (fit seed 0)

| Model | File | Blocks | First half | recipe_flip (configs/models) | LR ratio FH | LR twin FH | LR tax FH | UD ratio FH | UD twin FH | LR ratio SH | LR twin SH | UD ratio SH | UD twin SH | Commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clip_b16 | `results/CORE-04/structure/clip_b16/metrics.json` | 12 | 1 to 6 | no | 0.5056 | 0.3253 | 0.3125 | 0.5232 | 0.3507 | 0.5451 | 0.0005 | 0.8173 | 0.0023 | e84e20ff35 |
| convnext_b | `results/CORE-04/structure/convnext_b/metrics.json` | 36 | 1 to 18 | yes | 0.5034 | 0.2319 | 0.2168 | 0.5834 | 0.2104 | 0.7413 | 0.0031 | 0.7025 | 0.0030 | e84e20ff35 |
| deit3_b16 | `results/MUST-04/deit3_b16/metrics.json` | 12 | 1 to 6 | yes | 0.4501 | 0.4301 | 0.3487 | 0.4807 | 0.4630 | 0.3149 | 0.0058 | 0.7086 | 0.0205 | e84e20ff35 |
| dino_b16 | `results/CORE-04/structure/dino_b16/metrics.json` | 12 | 1 to 6 | yes | 0.4215 | 0.2797 | 0.2495 | 0.4920 | 0.3748 | 0.3745 | 0.0000 | 0.8320 | 0.0020 | e84e20ff35 |
| dinov2_vitb14 | `results/MUST-04/dinov2_vitb14/metrics.json` | 12 | 1 to 6 | yes | 0.4474 | 0.5374 | 0.4048 | 0.4956 | 0.5298 | 0.2773 | 0.0878 | 0.6518 | 0.0595 | e84e20ff35 |
| dinov2_vitb14_reg | `results/MUST-04/dinov2_vitb14_reg/metrics.json` | 12 | 1 to 6 | yes | 0.4791 | 0.5677 | 0.4185 | 0.5129 | 0.5600 | 0.3061 | 0.1197 | 0.6385 | 0.0908 | e84e20ff35 |
| dinov2_vitg14_reg | `results/CORE-04/structure/dinov2_vitg14_reg/metrics.json` | 40 | 1 to 20 | yes | 0.7672 | 0.0563 | 0.3440 | 0.8009 | 0.0538 | 0.6173 | 0.0041 | 0.8546 | 0.0005 | e84e20ff35 |
| dinov2_vitl14_reg | `results/MUST-04/dinov2_vitl14_reg/metrics.json` | 24 | 1 to 12 | yes | 0.5135 | 0.5031 | 0.4140 | 0.5497 | 0.4850 | 0.3388 | 0.0565 | 0.7036 | 0.0392 | e84e20ff35 |
| dinov2_vits14_reg | `results/CORE-04/structure/dinov2_vits14_reg/metrics.json` | 12 | 1 to 6 | yes | 0.4546 | 0.4835 | 0.4002 | 0.4490 | 0.5254 | 0.2961 | 0.1257 | 0.5939 | 0.1057 | e84e20ff35 |
| dinov3_b16 | `results/CORE-04/structure/dinov3_b16/metrics.json` | 12 | 1 to 6 | no (probable) | 0.4523 | 0.5330 | 0.4002 | 0.4810 | 0.5451 | 0.3367 | 0.0576 | 0.6574 | 0.0686 | e84e20ff35 |
| eva02_b14 | `results/CORE-04/structure/eva02_b14/metrics.json` | 12 | 1 to 6 | yes | 0.6168 | 0.0485 | 0.1647 | 0.6928 | 0.0544 | 0.3517 | 0.0000 | 0.8129 | 0.0000 | e84e20ff35 |
| mae_b16 | `results/CORE-04/structure/mae_b16/metrics.json` | 12 | 1 to 6 | yes | 0.5046 | 0.5972 | 0.4482 | 0.5120 | 0.6171 | 0.3840 | 0.2847 | 0.5629 | 0.1998 | e84e20ff35 |
| resnet50 | `results/CORE-04/structure/resnet50/metrics.json` | 16 | 1 to 8 | yes | 0.4434 | 0.3066 | 0.2794 | 0.4665 | 0.2954 | 0.5214 | 0.0898 | 0.7167 | 0.0593 | e84e20ff35 |
| siglip2_b16 | `results/CORE-04/structure/siglip2_b16/metrics.json` | 12 | 1 to 6 | unknown | 0.5585 | 0.2413 | 0.2309 | 0.5743 | 0.2878 | 0.5786 | 0.0002 | 0.8089 | 0.0020 | e84e20ff35 |

FH = first half, SH = second half of the blocks (for ConvNeXt-B and ResNet-50, of the stages' pointwise MLPs / bottlenecks). The four MUST-04 models have no folder under results/CORE-04/structure; the collector reads them from results/MUST-04. recipe_flip is the `recipe_flip` field of `configs/models/<model>.yaml`; dinov3_b16 carries `recipe_flip_confidence: probable`, so only clip_b16 counts as verified no-flip (CORE-04 P4.2).

Source: `results/CORE-04/structure/<model>/metrics.json`; `results/MUST-04/<model>/metrics.json`; `configs/models/<model>.yaml`.

| Model | Original k-NN | Flip bf16 | Params | Depth | Width |
|---|---|---|---|---|---|
| dinov2_vits14_reg | 74.474 | 93.28 | 22,058,112 | 12 | 384 |
| dinov2_vitb14_reg | 78.954 | 96.28 | 86,583,552 | 12 | 768 |
| dinov2_vitl14_reg | 80.576 | 97.74 | 304,372,736 | 24 | 1024 |
| dinov2_vitg14_reg | 80.346 | 98.23 | 1,136,486,912 | 40 | 1536 |

Source: `results/CORE-04/scale/knn.csv`.

### T5c. CORE-04 zero-shot surgery: Mirror-exact against plain SVD and SVD-LLM at matched parameters (fit seed 0)

| Model | Size | Original | Mirror-exact | Plain SVD | SVD-LLM | Mirror minus SVD (95% CI) | Mirror minus SVD-LLM (95% CI) | Model params fraction | Mirror flip bf16 |
|---|---|---|---|---|---|---|---|---|---|
| deit3_b16 | 6mlp | 82.97 | 81.06 | 81.29 | 82.81 | -0.24 [-0.45, -0.02] | -1.75 [-1.95, -1.55] | 0.8987 | 94.81 |
| deit3_b16 | 6both | 82.97 | 81.43 | 78.59 | 82.54 | +2.83 [2.58, 3.08] | -1.11 [-1.30, -0.91] | 0.8422 | 97.33 |
| dino_b16 | 6mlp | 70.48 | 67.13 | 70.46 | 70.51 | -3.33 [-3.58, -3.07] | -3.38 [-3.63, -3.13] | 0.9365 | 93.82 |
| dino_b16 | 6both | 70.48 | 64.39 | 70.26 | 70.54 | -5.87 [-6.16, -5.58] | -6.15 [-6.44, -5.87] | 0.8956 | 97.14 |
| dinov2_vitb14 | 6mlp | 78.69 | 74.42 | 67.74 | 78.05 | +6.67 [6.34, 7.00] | -3.64 [-3.89, -3.38] | 0.8735 | 90.72 |
| dinov2_vitb14 | 6both | 78.69 | 73.54 | 12.18 | 77.78 | +61.36 [60.91, 61.79] | -4.24 [-4.52, -3.98] | 0.8097 | 92.29 |
| dinov2_vitb14_reg | 6mlp | 78.94 | 76.59 | 9.36 | 78.33 | +67.23 [66.80, 67.64] | -1.74 [-1.97, -1.52] | 0.8698 | 91.70 |
| dinov2_vitb14_reg | 6both | 78.94 | 77.41 | 5.84 | 78.03 | +71.57 [71.16, 71.97] | -0.62 [-0.82, -0.42] | 0.8060 | 95.02 |
| dinov2_vitl14_reg | 12mlp | 80.59 | 79.37 | 79.07 | 80.50 | +0.30 [0.08, 0.52] | -1.13 [-1.33, -0.92] | 0.8717 | 94.48 |
| dinov2_vitl14_reg | 12both | 80.59 | 76.50 | 63.91 | 80.51 | +12.59 [12.21, 12.96] | -4.01 [-4.28, -3.73] | 0.8117 | 94.68 |
| dinov2_vits14_reg | 6mlp | 74.47 | 61.04 | 10.58 | 68.37 | +50.46 [49.99, 50.91] | -7.33 [-7.68, -6.98] | 0.8690 | 76.50 |
| dinov2_vits14_reg | 6both | 74.47 | 56.57 | 6.95 | 59.96 | +49.62 [49.17, 50.08] | -3.39 [-3.79, -2.98] | 0.8022 | 75.66 |
| mae_b16 | 6mlp | 13.73 | 8.67 | 3.81 | 13.59 | +4.86 [4.61, 5.11] | -4.91 [-5.19, -4.64] | 0.8590 | 51.94 |
| mae_b16 | 6both | 13.73 | 9.89 | 4.26 | 12.15 | +5.62 [5.35, 5.90] | -2.26 [-2.52, -2.01] | 0.7870 | 54.66 |
| siglip2_b16 | 6mlp | 75.54 | 7.84 | 11.43 | 74.25 | -3.59 [-3.92, -3.28] | -66.42 [-66.84, -65.98] | 0.9368 | 13.00 |
| siglip2_b16 | 6both | 75.54 | 13.45 | 4.18 | 72.90 | +9.27 [8.95, 9.59] | -59.45 [-59.90, -59.00] | 0.8977 | 40.54 |

Not run (status.csv, eligible by the 0.8 ratio rule but unsupported by the surgery code): dinov2_vitg14_reg (SwiGLU MLP); clip_b16 (norm_pre LayerNorm before block 1); dinov3_b16 (Eva blocks with rotary position embeddings); eva02_b14 (Eva blocks with rotary position embeddings, also SwiGLU).

MAE scores low on k-NN for every variant (original 13.73), so CORE-04 also ran a linear probe on MAE (linear_mae_b16.csv):

| Size | Method | Linear top-1 | Top-5 | Chosen wd |
|---|---|---|---|---|
| original | original | 56.532 | 79.812 | 0.001 |
| 6mlp | mirror_exact | 46.482 | 71.312 | 0.001 |
| 6mlp | svd | 22.170 | 42.190 | 0.001 |
| 6mlp | svdllm | 54.128 | 77.714 | 0.001 |
| 6both | mirror_exact | 48.520 | 72.958 | 0.001 |
| 6both | svd | 16.768 | 33.956 | 0.001 |
| 6both | svdllm | 52.476 | 76.334 | 0.001 |

Source: `results/CORE-04/surgery/rows_<model>.csv`; `results/CORE-04/surgery/bootstrap.csv`; `results/CORE-04/surgery/status.csv`; `results/CORE-04/surgery/linear_mae_b16.csv`.

## T6. Interpretability: even and odd features on CUB (CORE-06)

### T6a. species accuracy (%)

| Features | Test set | original | vexact_s0 | vexact_s1 | vexact_s2 | Mirror-exact mean ± sd (derived) |
|---|---|---|---|---|---|---|
| even_cls | natural | 89.96 | 89.45 | 89.52 | 89.44 | 89.47 ± 0.05 |
| even_cls | mirror_augmented | 90.06 | 89.45 | 89.52 | 89.44 | 89.47 ± 0.05 |
| odd_cls | natural | 25.01 | 3.80 | 3.71 | 3.62 | 3.71 ± 0.09 |
| odd_cls | mirror_augmented | 25.41 | 3.42 | 3.39 | 3.31 | 3.37 ± 0.05 |
| full_cls | natural | 90.01 | 89.42 | 89.47 | 89.37 | 89.42 ± 0.05 |
| full_cls | mirror_augmented | 90.05 | 89.36 | 89.43 | 89.33 | 89.37 ± 0.05 |

### T6b. facing accuracy (%)

| Features | Test set | original | vexact_s0 | vexact_s1 | vexact_s2 | Mirror-exact mean ± sd (derived) |
|---|---|---|---|---|---|---|
| even_cls | natural | 86.70 | 50.84 | 50.29 | 50.57 | 50.57 ± 0.27 |
| even_cls | mirror_augmented | 86.31 | 50.00 | 50.00 | 50.00 | 50.00 ± 0.00 |
| odd_cls | natural | 94.99 | 99.19 | 99.21 | 99.25 | 99.21 ± 0.03 |
| odd_cls | mirror_augmented | 95.31 | 99.18 | 99.19 | 99.25 | 99.21 ± 0.04 |
| full_cls | natural | 96.33 | 95.38 | 95.46 | 95.25 | 95.36 ± 0.10 |
| full_cls | mirror_augmented | 96.42 | 95.30 | 95.40 | 95.24 | 95.31 ± 0.08 |
| even_patch | natural | 83.38 | 50.74 | 50.07 | 50.74 | 50.52 ± 0.39 |
| even_patch | mirror_augmented | 82.91 | 50.00 | 50.00 | 50.00 | 50.00 ± 0.00 |
| odd_patch | natural | 96.63 | 99.32 | 99.27 | 99.30 | 99.30 ± 0.03 |
| odd_patch | mirror_augmented | 96.72 | 99.27 | 99.20 | 99.27 | 99.25 ± 0.04 |
| full_patch | natural | 96.87 | 97.29 | 97.23 | 97.31 | 97.27 ± 0.04 |
| full_patch | mirror_augmented | 96.82 | 97.26 | 97.24 | 97.29 | 97.26 ± 0.02 |

Data (`results/CORE-06/SUMMARY.md`): CUB-200-2011, 5,994 train / 5,794 test images for species; facing labels on 5,443 train / 5,307 test images; mirror-augmented test sets double the test images. The original model's even/odd split uses A_L; the healed models use their exact A*. Weight decay chosen by cross-validation on the training split. CUB overlaps ImageNet (caveat in SUMMARY.md).

- Mirror-augmented facing from the even CLS: original 86.31 (pairs predicted alike 26.10%, differing pairs not near-tie 3922); vexact_s0 50.00 (pairs predicted alike 100.00%, differing pairs not near-tie 0); vexact_s1 50.00 (pairs predicted alike 100.00%, differing pairs not near-tie 0); vexact_s2 50.00 (pairs predicted alike 100.00%, differing pairs not near-tie 0).
- P6.1, P6.2, P6.3 all PASS (`FINAL_RESULTS.md`).

Source: `results/CORE-06/probes.csv`; `results/CORE-06/SUMMARY.md`.

## T7. Distribution shift (CORE-07)

| Variant | Role | ImageNet val (probe) | ImageNet-V2 MF | V2 drop | Sketch | Sketch drop | Sketch dedup | Sketch dedup drop | Chosen wd |
|---|---|---|---|---|---|---|---|---|---|
| original | original | 81.004 | 72.38 | 8.624 | 52.77 | 28.23 | 58.61 | 22.40 | 0.00316 |
| vexact_s0 | mirror_exact | 80.102 | 70.95 | 9.152 | 49.31 | 30.79 | 54.95 | 25.15 | 0.00316 |
| vexact_s1 | mirror_exact | 80.044 | 70.90 | 9.144 | 49.45 | 30.59 | 55.11 | 24.94 | 0.00316 |
| vexact_s2 | mirror_exact | 80.188 | 70.97 | 9.218 | 49.39 | 30.80 | 55.04 | 25.15 | 0.00316 |
| best_svdllm_s0 | best_baseline | 79.726 | 70.31 | 9.416 | 45.17 | 34.56 | 50.43 | 29.29 | 0.001 |

- n: ImageNet val 50000, ImageNet-V2 10000, Sketch 50889 (dedup 33009).
- P7.1: drop original 8.624, Mirror-exact mean 9.171 (sd 0.041, 3 seeds), difference +0.547, bootstrap 95% CI [0.083, 1.015] (10000 resamples); verdict PASS.
- SVD-LLM seed 0 (reported, not scored): V2 drop 9.416, minus original +0.792, CI [0.244, 1.344]. Sketch drop: original 28.23, Mirror-exact mean 30.73, SVD-LLM 34.56.
- Caveat (SUMMARY.md): ImageNet-Sketch pads classes with flipped and rotated copies, which an exactly invariant model scores alike; the dedup columns drop one image of each detected near-duplicate pair (heuristic threshold 0.02).
- The CORE-07 probes for the healed models were retrained here; their ImageNet top-1 (80.102 / 80.044 / 80.188; SVD-LLM 79.726) differ slightly from MUST-11 t2_linear.csv (80.104 / 80.042 / 80.186; 79.716).

Source: `results/CORE-07/shift_rows.csv`; `results/CORE-07/metrics.json`; `results/CORE-07/SUMMARY.md`.

## T8. Exactness and the law (MUST-01, MUST-02)

| Model | Blocks compared | Fit seeds | Structure | Max abs(predicted - achieved) (pp) | Max diff vs recorded run (pp) |
|---|---|---|---|---|---|
| dinov2_vitb14 | 36 | 0, 1, 2 | fitted operators and pairings | 0 | 8.33e-15 |
| dinov2_vitb14_reg | 36 | 0, 1, 2 | fitted operators and pairings | 0 | 8.33e-15 |
| deit3_b16 | 12 | 0 | fitted operators and pairings | 0 | 8.33e-15 |
| dinov2_vitl14_reg | 24 | 0 | synthetic structure with the recorded counts | 0 | 9.71e-15 |
| dinov2_vits14_reg | 12 | 0 | synthetic structure with the recorded counts | 0 | 9.02e-15 |
| mae_b16 | 12 | 0 | fitted operators and pairings | 0 | 5.55e-15 |
| siglip2_b16 | 12 | 0 | fitted operators and pairings | 0 | 8.33e-15 |

Overall max 0.0 pp over 144 block rows, tolerance 1.0 pp, pass True. For dinov2_vitl14_reg and dinov2_vits14_reg the structure is synthetic with the recorded counts (arithmetic check only).

| Model | Fit seed | k | MLP params kept, recorded | Achieved | Diff (pp) |
|---|---|---|---|---|---|
| dinov2_vitb14 | 0 | 3 | 0.534401 | 0.534401 | 0.0 |
| dinov2_vitb14 | 0 | 6 | 0.595236 | 0.595236 | 0.0 |
| dinov2_vitb14 | 0 | 9 | 0.676694 | 0.676694 | 0.0 |
| dinov2_vitb14 | 1 | 6 | 0.594855 | 0.594855 | 0.0 |
| dinov2_vitb14 | 2 | 6 | 0.595053 | 0.595053 | 0.0 |
| dinov2_vitb14_reg | 0 | 3 | 0.539918 | 0.539918 | 0.0 |
| dinov2_vitb14_reg | 0 | 6 | 0.581481 | 0.581481 | 0.0 |
| dinov2_vitb14_reg | 0 | 9 | 0.658148 | 0.658148 | 0.0 |
| dinov2_vitb14_reg | 1 | 6 | 0.581100 | 0.581100 | 0.0 |
| dinov2_vitb14_reg | 2 | 6 | 0.581405 | 0.581405 | 0.0 |
| deit3_b16 | 0 | 6 | 0.651288 | 0.651288 | 0.0 |
| dinov2_vitl14_reg | 0 | 12 | 0.586002 | 0.586002 | 0.0 |
| dinov2_vits14_reg | 0 | 6 | 0.599829 | 0.599829 | 0.0 |
| mae_b16 | 0 | 6 | 0.551819 | 0.551819 | 0.0 |
| siglip2_b16 | 0 | 6 | 0.769120 | 0.769120 | 0.0 |

Source: `results/MUST-01/metrics.json`; `results/MUST-01/law_per_layer.csv`; `results/MUST-01/law_surgery_k.csv`; `results/MUST-01/summary.md`.

| Variant | dtype | Max rel. error (whole tensor) | Worst component | Max rel. error (per token) | Worst (token) | Components above 1e-5 | k-NN flip fp32 | Disagreements | Max unit-feature diff | k-NN flip bf16 (disagreements) | k-NN top-1 fp32 | Images (component audit) | Commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zeroshot | float32 | 2.51e-05 | block 11 act | 0.00018 | block 11 fc2 | 116 of 198 | 100.000 | 0 of 50000 | 4.28e-06 | 98.054 (973) | 53.960 | 1000 | e84e20ff35 |
| zeroshot_fp64 | float64 | 5.77e-14 | block 4 v | 3.39e-13 | block 11 fc2 | 0 of 198 | n/a | n/a | n/a | n/a | n/a | 1000 | 52e29d1676 |

| Space | Max rel. error fp32 | Max rel. error fp64 |
|---|---|---|
| attention | 1.73e-05 | 3.44e-14 |
| head-k | 1.55e-05 | 2.67e-14 |
| head-out | 1.45e-05 | 3.07e-14 |
| head-q | 1.57e-05 | 3.07e-14 |
| head-v | 2.5e-05 | 5.77e-14 |
| hidden | 2.51e-05 | 4.34e-14 |
| invariant | 1.55e-05 | 3.39e-14 |
| patch | 9.02e-07 | 2.85e-15 |
| stream | 2.33e-05 | 4e-14 |

Zero-shot Mirror-exact, full model (k = 12, d+ 700, d- 68), fit seed 0; tolerance 1e-5; 1,000 validation images for the component audit; the fp32 flip check uses all 50,000 validation queries against the 100,000-image bank. MUST-02 FAILS on the fp32 component bar (2.51e-05) while flip consistency is 100%.

Healed models (MUST-09):

| Method | Seed | Exactness max rel. error inside span (64 images, fp32, tol 1e-4) | CLS flip rel. max (fp32, 64 images) | Max unit-feature flip diff (fp32, 50,000) |
|---|---|---|---|---|
| mirror_exact | 0 | 9.31e-06 | 6.53e-06 | 2.32e-06 |
| mirror_exact | 1 | 8.69e-06 | 4.04e-06 | 2.94e-06 |
| mirror_exact | 2 | 9.81e-06 | 4.29e-06 | 3.2e-06 |
| svdllm | 0 | n/a | 0.225 | 0.131 |
| svdllm | 1 | n/a | 0.222 | 0.153 |
| svdllm | 2 | n/a | 0.21 | 0.136 |
| slicegpt | 0 | n/a | 0.305 | 0.156 |
| svd | 0 | n/a | 0.305 | 0.127 |
| random_structure | 0 | 6.54e-06 | 7.83e-06 | 4.3e-06 |

Source: `results/MUST-02/dinov2_vitb14_reg/{zeroshot,zeroshot_fp64}/metrics.json`; `runs/MUST-09/dinov2_vitb14_reg/<method>/s<seed>/metrics.json (exactness, cls_flip_fp32_probe, knn_fp32)`.

## T9. Negative results (one line each)

| Item | Prediction | Value | Verdict | Source |
|---|---|---|---|---|
| MUST-04 P4.1 | UD and R90 ratio to null in blocks 1 to 6 >= 0.9 on all 4 models | UD 0.4807 to 0.5129, R90 0.7536 to 0.8494 (T5a) | FAIL | `results/MUST-04/<model>/metrics.json` |
| MUST-04 P4.2 | UD twin fraction 1-6 inside the twin-null 95% interval | 0.4630 to 0.5600 against null upper bounds of at most 0.0035 (T5a) | FAIL | same |
| MUST-04 P4.3 | Zero-shot UD surgery on blocks 1 to 6 drops k-NN by > 30 points | UD drops 0.232 (L-reg) to 6.076 (DINOv2-B) points (T5a) | FAIL | same |
| MUST-05 P5.1 | Switch-off median cosine >= 0.7 (twins), <= 0.2 (random) | twins 0.4875, random -9.22e-07 (1,200 pairs each, blocks 1 to 6, 500 images); P5.2 norm-ratio share 0.9225 passes | FAIL | `results/MUST-05/dinov2_vitb14_reg/switch_pairs.csv` (recomputed: median of mid_cos) |
| MUST-06 P6.3 | Effective rank of fc1/fc2 changes by <= 2% under projection | largest absolute change 0.0864 (b14-reg; 2 of 12 matrices above 0.02), 0.2161 (L-reg), 0.1764 (DeiT-III); energy-based erank 0.1031 (b14-reg) | FAIL | `results/MUST-06/<model>/spectra.csv` (erank_l1_rel_change, structure mirror, blocks 1 to 6) |
| MUST-07 P7.1 | Mirror-exact beats every baseline at 6 blocks by >= 2 points | smallest margin -1.75 (SVD-LLM, 6mlp); other margins under 2: ASVD -0.20, FLAP +1.41, ZipIt +0.93 (6mlp); SVD-LLM -0.68, SliceGPT +1.74 (6both) | FAIL | `results/MUST-07/rows_dinov2_vitb14_reg.csv`, `bootstrap.csv` |
| MUST-07 P7.2 | ZipIt and same-image pairing trail by >= 10 points | ZipIt 6mlp margin 0.934; the other three margins 46.03 to 58.83 | FAIL | same |
| Gate C (MUST-12) | LR-trained pilot: LR ratio to null 1-6 <= 0.8 and LR twin fraction 1-6 >= 0.2 | ViT-S/8 64 px: ratio 0.4844, twin 0.0601; ViT-S/16 96 px: ratio 0.4752, twin 0.0647 (ratio passes, twin fails) | FAIL (both) | `results/MUST-12/gate_c.json`, `results/MUST-12/gate_c_vit_s16.json` |
| CORE-01 | Controlled-training grid (T1 to T4) | dropped after gate C; T1 and T3 PENDING, T2 and T4 FAIL on the pilot runs only (below) | dropped | `results/STATUS.md` lines 36 to 41; `FINAL_RESULTS.md` |
| CORE-03 P3.1 | Mirror + low rank above low rank alone at >= 3 of 4 budget points on >= 2 of 3 models | 0 of 4 points on each of dinov2_vitb14_reg, dinov2_vitl14_reg, deit3_b16 (T9b) | FAIL | `results/CORE-03/p31.json`, `curves.csv` |
| CORE-04 P4.1 | LR tax in blocks 1 to L/2 rises S < B < L < g | S 0.4002, B 0.4185, L 0.4140, g 0.3440 (T5b) | FAIL | `results/CORE-04/structure/*/metrics.json`, `results/MUST-04/*/metrics.json` |
| CORE-04 P4.2 | Verified no-flip models have lower LR tax than every flip-trained model of the same size | clip_b16 (only verified no-flip) 0.3125 is above dino_b16 0.2495, eva02_b14 0.1647 (and siglip2 0.2309, flip unknown) (T5b) | FAIL | same; `configs/models/*.yaml` |
| CORE-05 P5.1 | Activation-weighted breaking share <= half the weight-energy share, blocks 1 to 6 | ratio 0.5803 (b14-reg, operator A*_L); L-reg 0.5369, DeiT-III 0.6834, MAE 0.5521; only L-reg with the per-layer operator (0.4442) and star6 (0.4571) is below 0.5 | FAIL | `results/CORE-05/measures_<model>.csv` (recomputed) |
| CORE-05 P5.3 | Frozen out-of-sample rule correct on >= 75% of cases | 32 of 48 = 0.6667; per model dinov2_vitl14_reg 13/24, deit3_b16 11/12, mae_b16 8/12; rule frozen 2026-11-15T16:36:17Z, scored 2026-11-15T20:21:19Z | FAIL | `results/CORE-05/rule_score.json` |
| CORE-05 P5.2 | Band split over CORE-01 training | not available (results/CORE-05/ckpt_bands.csv missing; depends on CORE-01) | PENDING | `FINAL_RESULTS.md` |
| UP-07 (reported) | Weights-only pairing agrees with data-driven pairing on >= 70% of pairs (block 1) | weights_joint 0.6010 (702 of 1168); given the data operator 0.6361 | below bar | `results/UP-07/up07.json` |

### T9a. MUST-12 controlled pilots, blocks 1 to 6 only

Per the instruction, nothing from blocks 7 to 12 or the "second_half" range of the MUST-12 pilot models was opened, computed or reported (the extraction scripts read only the "_1-6" score keys, plus "first_half" once to confirm that for these 12-block models it is the same range).

| Run (seed 0) | Epoch | LR ratio 1-6 | LR twin 1-6 | UD ratio 1-6 | UD twin 1-6 | R90 4-cycle share 1-6 | LR tax 1-6 |
|---|---|---|---|---|---|---|---|
| ViT-S/8, 64 px, none | 5 | 0.9215 | 0.0762 | 0.9326 | 0.1910 | 0.0087 | 0.2535 |
| ViT-S/8, 64 px, none | 100 | 0.5465 | 0.0618 | 0.7026 | 0.1823 | 0.0126 | 0.1562 |
| ViT-S/8, 64 px, LR flips | 5 | 0.9198 | 0.0846 | 0.9321 | 0.2014 | 0.0104 | 0.2611 |
| ViT-S/8, 64 px, LR flips | 100 | 0.4844 | 0.0601 | 0.6856 | 0.1988 | 0.0148 | 0.1446 |
| ViT-S/16, 96 px, none | 5 | 0.9245 | 0.0623 | 0.9341 | 0.1534 | 0.0082 | 0.3005 |
| ViT-S/16, 96 px, none | 100 | 0.5200 | 0.0734 | 0.6832 | 0.1860 | 0.0178 | 0.1893 |
| ViT-S/16, 96 px, LR flips | 5 | 0.9252 | 0.0686 | 0.9354 | 0.1691 | 0.0056 | 0.3018 |
| ViT-S/16, 96 px, LR flips | 100 | 0.4752 | 0.0647 | 0.6634 | 0.1782 | 0.0130 | 0.1714 |

| Run | Image px | Epochs | Val top-1 | Val top-1, LR-flipped | UD-flipped | R90-rotated | n val | n train | Wall minutes | Commit | Started |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vit_s8_none_s0 | 64 | 100 | 55.1000 | 55.0000 | 32.8200 | 30.2000 | 5000 | 126689 | 11.76 | 52e29d1676 | 2026-11-15T07:26:16Z |
| vit_s8_LR_s0 | 64 | 100 | 56.0000 | 56.1400 | 33.7400 | 30.2600 | 5000 | 126689 | 19.56 | 52e29d1676 | 2026-11-15T07:48:31Z |
| vit_s16_none_s0 | 96 | 100 | 52.8400 | 52.1200 | 31.1800 | 27.9600 | 5000 | 126689 | 45.68 | d572a53fe9 | 2026-11-15T12:51:44Z |
| vit_s16_LR_s0 | 96 | 100 | 51.7000 | 51.8600 | 31.6200 | 27.7000 | 5000 | 126689 | 45.81 | d572a53fe9 | 2026-11-15T12:52:24Z |

CORE-01 pilot values (blocks 1 to 6, epoch 100): T2 UD ratio none s0 0.7026 and LR s0 0.6856 (bar >= 0.9, FAIL); T4 LR twin fraction LR s0 epoch 5 0.0846 to epoch 100 0.0601 (must rise, FAIL). Gate C epoch rule: ViT-S/8 none wall 11.76 min gives 100 epochs; ViT-S/16 none wall 45.68 min gives 60 epochs (rule 25.0 min), although both ViT-S/16 pilots trained for 100.

Source: `results/MUST-12/<run>/{e000,e005,e100}/metrics.json (scores, keys *_1-6 only)`; `results/MUST-12/gate_c.json`; `results/MUST-12/gate_c_vit_s16.json`; `runs/MUST-12/<run>/meta.json (config, extra.final_val, extra.data, extra.timing)`.

### T9b. CORE-03 curve points (zero-shot, full model, fit seed 0, validation k-NN)

| Model | Point | Span kept | SVD alone | SVD-LLM alone (scored low rank) | Mirror + SVD (scored) | Mirror + SVD-LLM (not scored) | Top-1 | Scored diff (95% CI) | Note |
|---|---|---|---|---|---|---|---|---|---|
| dinov2_vitb14_reg | original | 1.000 |  |  |  |  | 78.94 |  |  |
| dinov2_vitb14_reg | Mirror-exact alone | 0.760 |  |  |  |  | 53.88 |  |  |
| dinov2_vitb14_reg | f0.85 | 0.850 | 3.73 | 77.91 | 53.88 | 53.88 |  | -24.03 [-24.47, -23.61] | Mirror-exact itself (below budget) |
| dinov2_vitb14_reg | f0.70 | 0.700 | 6.03 | 74.02 | 6.19 | 46.77 |  | -67.83 [-68.24, -67.41] |  |
| dinov2_vitb14_reg | f0.55 | 0.550 | 5.75 | 60.98 | 2.69 | 36.58 |  | -58.29 [-58.74, -57.86] |  |
| dinov2_vitb14_reg | f0.40 | 0.400 | 4.95 | 15.55 | 5.39 | 16.83 |  | -10.16 [-10.48, -9.83] |  |
| dinov2_vitl14_reg | original | 1.000 |  |  |  |  | 80.59 |  |  |
| dinov2_vitl14_reg | Mirror-exact alone | 0.815 |  |  |  |  | 30.17 |  |  |
| dinov2_vitl14_reg | f0.85 | 0.850 | 67.51 | 80.61 | 30.17 | 30.17 |  | -50.44 [-50.90, -49.98] | Mirror-exact itself (below budget) |
| dinov2_vitl14_reg | f0.70 | 0.700 | 22.37 | 79.68 | 12.03 | 21.79 |  | -67.65 [-68.07, -67.23] |  |
| dinov2_vitl14_reg | f0.55 | 0.550 | 10.29 | 74.82 | 9.79 | 19.50 |  | -65.02 [-65.45, -64.59] |  |
| dinov2_vitl14_reg | f0.40 | 0.400 | 6.80 | 53.77 | 9.69 | 13.34 |  | -44.08 [-44.53, -43.62] |  |
| deit3_b16 | original | 1.000 |  |  |  |  | 82.97 |  |  |
| deit3_b16 | Mirror-exact alone | 0.831 |  |  |  |  | 72.85 |  |  |
| deit3_b16 | f0.85 | 0.850 | 75.41 | 80.83 | 72.85 | 72.85 |  | -7.99 [-8.30, -7.67] | Mirror-exact itself (below budget) |
| deit3_b16 | f0.70 | 0.700 | 65.31 | 77.43 | 56.86 | 64.87 |  | -20.57 [-20.97, -20.16] |  |
| deit3_b16 | f0.55 | 0.550 | 36.55 | 68.70 | 40.80 | 55.20 |  | -27.89 [-28.35, -27.45] |  |
| deit3_b16 | f0.40 | 0.400 | 13.23 | 46.68 | 20.58 | 37.89 |  | -26.10 [-26.56, -25.66] |  |

The scored inner method is plain SVD (p31.json score_inner "svd"). Mirror + SVD-LLM is above SVD-LLM alone at one point (dinov2_vitb14_reg f0.40: 16.83 against 15.55); it is not the scored curve.

Source: `results/CORE-03/curves.csv`; `results/CORE-03/bootstrap.csv`; `results/CORE-03/p31.json`; `results/CORE-03/SUMMARY.md`.

## T10. Existing figures (results/paper_next/figures)

| Figure | What it shows | Usable as is? | Data file(s) in results/paper_next/ |
|---|---|---|---|
| `fig1_twins.png` | Grid of top-activating ImageNet-train patches for 4 strongest twin pairs in each of blocks 1, 3 and 6 of DINOv2-B reg (6 patches for neuron i, 6 for its twin j). Title: "the LR tax is 42% of MLP weights in blocks 1 to 6". In block 1 several pairs show mirrored edge patterns (for example pair 1); by block 6 the mirroring is hard to see by eye. | Usable after layout fixes: large empty band under the title, many patches padded with grey at image borders, small low-resolution crops. | data/fig1_teaser.npz, data/fig1_meta.csv (teaser from results/MUST-05/dinov2_vitb14_reg/teaser.npz); PAPER_NUMBERS must04.dinov2_vitb14_reg.LR.tax_1-6 |
| `fig2_tax_depth.png` | (a) LR tax by relative block depth for all 14 models (4 coloured, 10 grey); (b) breaking share over the null mean per block for LR, UD, R90, R180 on DINOv2-B reg with the null 95% band; (c) ratio to null in blocks 1 to 6 for the 4 MUST-04 models with the 0.9 P4.1 bar. | Not usable as is: legends sit on the curves in (a) and (b), grey models are labelled with code names, the grey bundle in (a) is unreadable. Panel (c) is clean and shows the P4.1 failure (UD and R180 as low as LR). | data/fig2_blocks.csv, data/table4_generality.csv |
| `fig3_controlled.png` | (a) Structure matrix at the final checkpoint, blocks 1 to 6: rows "none" and "LR flips", columns LR twins, UD twins, R90 and Klein orbits of 4 (values 0.06, 0.18, 0.01, 0.00 and 0.06, 0.20, 0.01, 0.00); (b) LR twin share in blocks 1 to 6 against epoch for the two pilots, from 0.4798 at epoch 0 to 0.0618 (none) and 0.0601 (LR) at epoch 100 (values from results/MUST-12/vit_s8_*/e*/metrics.json, *_1-6 keys). | Usable only as a negative-result or appendix figure, with a corrected caption: it shows the ViT-S/8 64 px pilots (seed 0, two conditions), not the planned CORE-01 grid; "mean over seeds" in the panel title is one seed. Shows blocks 1 to 6 only. | data/fig3_matrix.csv, data/fig3_emergence.csv (data/fig3_val_acc.csv is not plotted) |
| `fig4_visibility_stacking.png` | (a) Histograms of pair correlations in blocks 1 to 6 of DINOv2-B reg: twins on the mirrored image (broad, centred above 0.5), twins on the same image and random pairs (both near 0; MUST-06 medians over 5,810 twin pairs: mirrored 0.5513, same image 0.0516, from results/MUST-06/dinov2_vitb14_reg/pairs.csv); (b) zero-shot k-NN against parameters kept for low rank alone and Mirror plus low rank on DeiT-III, DINOv2-B reg and DINOv2-L reg, with Mirror alone as stars. | Panel (a) is usable. Panel (b) needs the legend moved (it covers the DINOv2-L star and curves) and the model named in (a). Planned panel (c) (bands over training) is missing: data/fig4c_bands.csv does not exist. | data/fig4a_pairs.csv (data/fig4a_erank.csv present, not plotted), data/fig4b_stacking.csv |
| `fig5_accuracy_params.png` | (a) Zero-shot validation k-NN against parameters for every MUST-07 method on DINOv2-B reg (Mirror-exact, SVD-LLM, plain SVD, random structure highlighted; other methods thin grey; original as a star); (b) healed full models at about 66M parameters against the released DINOv2 S, B, L and g (reg) on a log axis. | Usable with light clean-up: in (a) the thin grey lines cross because each method is joined across sizes, and Mirror-local sits at its own unmatched parameter counts; in (b) the healed points overlap and the log axis has sparse labels. | data/fig5_points.csv; PAPER_NUMBERS gate_b.best |

All five were written by scripts/make_figures_next.py at 2026-11-15T20:30:27Z (commit 6b4c66ab78); PDF versions sit next to each PNG. Other data files in results/paper_next/data feed the tables: table1_healed.csv, table1_flipavg.csv, table2_downstream.csv, table2_shift.csv, must07_rows.csv, table3_must07_p7.csv, table3_must08.csv, table3_earlier_ablations.csv, table4_generality.csv, appendix_law.csv, appendix_exactness.csv, must09_curves.csv.

Source: `results/paper_next/figures/*.png`; `results/paper_next/FIGURES.md`; `results/paper_next/figures_meta.json`; `scripts/make_figures_next.py (plotting code read to confirm fig3 uses *_1-6 keys only)`.

## Inconsistencies and cautions found while compiling

1. MUST-09 provenance: metrics.json and meta.json record different commits for 4 of 9 heals (mirror_exact s0 and random_structure s0: 52e29d1676 against e84e20ff35; slicegpt s0 and svdllm s0: e84e20ff35 against 805c231c9c; T1a). FINAL_RESULTS.md cites the metrics.json commits.
2. MUST-10 headline: FINAL_RESULTS.md and PAPER_NUMBERS give -0.66 (3-seed mean against flip averaging); results/MUST-10/MUST10.md and results/STATUS.md give -0.68 (seed 0). The bootstrap interval exists for seed 0 only.
3. Mask equivariance: PAPER_NUMBERS must11.maskeq.vexact_s0.sym is printed as 100.00 in FINAL_RESULTS.md, but the file value is 99.99991 with 422 differing pixels and P11.2 is PENDING.
4. Original-model k-NN differs by pipeline: 78.954 (MUST-09, MUST-10, CORE-04 scale, cached) against 78.938 (MUST-07 and MUST-08 runner; STATUS.md quotes 78.94); DINOv2-L reg 80.576 (MUST-04, CORE-04 scale) against 80.588 (MUST-07 runner, also CORE-04 surgery rows); DeiT-III 82.964 (MUST-04) against 82.974 (MUST-07).
5. Linear probe of the same healed models: MUST-11 t2_linear.csv 80.104 / 80.042 / 80.186 (SVD-LLM 79.716) against CORE-07 and MUST-10 80.102 / 80.044 / 80.188 (79.726). The k-NN check inside t2_linear.csv (78.312 and 78.354 for seeds 0 and 2) differs from MUST-09 (78.316, 78.352).
6. Model size: the model_params column of the MUST-11 CSVs gives 85,691,904 for Mirror-exact, MUST-09 gives 66,203,360.
7. results/MUST-07/SUMMARY.md says the full-size fp32 flip pass used 50,000 queries and a 100,000-image bank; rows_dinov2_vitb14_reg.csv shows that only for Mirror-exact and the original; the other full-size rows used 5,000 queries and a 10,000-image bank.
8. FINAL_RESULTS.md cites results/MUST-06/<model>/pairs.csv for P6.3, but the effective-rank values are in spectra.csv (PAPER_NUMBERS cites spectra.csv).
9. FINAL_RESULTS.md CORE-04 P4.2 lists the commits of its metrics.json sources as "(not recorded)"; P4.1 lists e84e20ff35 for the same files.
10. CORE-06: SUMMARY.md header and metrics.json say commit e84e20ff35, meta.json says 805c231c9c; FINAL_RESULTS.md cites probes.csv at 805c231c9c.
11. results/MUST-07/rows_deit3_b16.csv and results/CORE-04/surgery/rows_dinov2_vits14_reg.csv record commit 5edf2980f0, which none of the matching meta files records (MUST-07 deit3 meta: e84e20ff35, d572a53fe9; CORE-04 vits14 meta: d572a53fe9).
12. results/MUST-12/gate_c_vit_s16.json sets 60 CORE-01 epochs by the wall-time rule, but both ViT-S/16 pilots trained 100 epochs (runs meta). Gate C fails either way.
13. FINAL_RESULTS.md shows for UP-07 only the earlier-phase reproduction values (0.4658, 0.4409); the new primary value in up07.json is 0.6010 (weights_joint, block 1), also below the 0.70 bar.
14. results/paper_next/tables/table1_healed.md places steps 100 and 300 (10,000 queries) next to steps 0 and 780 (50,000 queries).
15. fig3_controlled.png says "mean over seeds" but holds one seed per condition (ViT-S/8 pilots only).
16. Latency: no SliceGPT measurement; the stored nvidia-smi lines read 99% utilisation; single runs. Treat speed numbers as indicative.

Checked and consistent: every MUST-09 per-seed k-NN, flip and parameter value in FINAL_RESULTS.md; all MUST-07, MUST-08, MUST-04, CORE-04, CORE-05 P5.1, CORE-06, CORE-07 and MUST-05 headline values; MUST-12 gate C and CORE-01 T2/T4 values (blocks 1 to 6). The CORE-05 rule hashes 68ab3dff1811 (rule content) and 480dd983e6a3 (file sha256, `results/CORE-05/rule_committed.flag`) refer to different things and agree with each other.

