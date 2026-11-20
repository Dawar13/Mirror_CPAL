# T1: main healed comparison, DINOv2-B/14 reg (final bf16 k-NN, 50,000 validation queries; interval: A13 bootstrap of Mirror minus the arm, 10,000 resamples, seed 0)

Written by `scripts/collect_final_phase.py`; every row's source file and commit are in `T1_main_comparison.csv` (columns source_file and git_hash).

| arm | folder | role | seeds | n | s0 | s1 | s2 | mean | sd | flip_fp32_min | params_model | params_frac | lr | mirror_minus_arm | ci_low | ci_high | p_le_0 | effect_size |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mirror, A*_w | mirror_exact_wspace | A01 ablation | 0,1 | 2 | 76.93 | 76.89 | n/a | 76.91 | 0.0339 | 100.00 | 66,203,360 | 0.7646 | 1.00e-04 | n/a | n/a | n/a | n/a | n/a |
| Mirror, current A* | mirror_exact_astar | main arm | 0,1,2 | 3 | 78.35 | 78.36 | 78.34 | 78.35 | 0.0120 | 100.00 | 66,203,360 | 0.7646 | 1.00e-04 | n/a | n/a | n/a | n/a | n/a |
| Mirror from fit1 | mirror_exact_fit1 | A02 heal | 0 | 1 | 78.34 | n/a | n/a | 78.34 | n/a | n/a | 66,114,564 | 0.7636 | 1.00e-04 | n/a | n/a | n/a | n/a | n/a |
| Mirror from fit2 | mirror_exact_fit2 | A02 heal | 0 | 1 | 78.37 | n/a | n/a | 78.37 | n/a | n/a | 66,202,096 | 0.7646 | 1.00e-04 | n/a | n/a | n/a | n/a | n/a |
| Mirror from coco | mirror_exact_coco | A02 heal | 0 | 1 | 78.35 | n/a | n/a | 78.35 | n/a | n/a | 65,872,540 | 0.7608 | 1.00e-04 | n/a | n/a | n/a | n/a | n/a |
| SVD-LLM (whitening only) | svdllm | baseline (A05) | 0,1,2 | 3 | 78.08 | 78.07 | 78.10 | 78.08 | 0.0136 | n/a | 66,203,904 | 0.7646 | 1.00e-04 | 0.2713 | 0.0809 | 0.4649 | 0.0028 | 21.95 |
| full SVD-LLM | svdllm_full | baseline (A05) | 0,1,2 | 3 | 76.45 | 76.76 | 76.06 | 76.42 | 0.3477 | n/a | 66,203,904 | 0.7646 | 1.00e-04 | 1.93 | 1.55 | 2.31 | 0 | 9.60 |
| IO-SVD | iosvd | baseline (A05) | 0,1,2 | 3 | 0.5440 | 0.4000 | 0.6580 | 0.5340 | 0.1293 | n/a | 66,204,672 | 0.7646 | 1.00e-04 | 77.82 | 77.44 | 78.20 | 0 | 1034.18 |
| SliceGPT-style | slicegpt | baseline (A05) | 0,1 | 2 | 77.86 | 77.85 | n/a | 77.85 | 0.0028 | n/a | 66,324,696 | 0.7660 | 1.00e-04 | 0.5000 | 0.2916 | 0.7020 | 0 | 47.65 |
| plain SVD | svd | baseline (A05) | 0 | 1 | 77.00 | n/a | n/a | 77.00 | n/a | n/a | 66,203,904 | 0.7646 | 1.00e-04 | 1.35 | 1.14 | 1.57 | 0 | 116.38 |
| Random structure | random_structure | control (A04) | 0 | 1 | 70.10 | n/a | n/a | 70.10 | n/a | n/a | 66,203,360 | 0.7646 | 1.00e-04 | n/a | n/a | n/a | n/a | n/a |
| H1: true operator, random pairing | pair_random_h1 | control (A03) | 0 | 1 | 77.01 | n/a | n/a | 77.01 | n/a | n/a | 66,203,360 | 0.7646 | 1.00e-04 | n/a | n/a | n/a | n/a | n/a |
| Width-pruned dense student | flap | dense student (A04) | 0 | 1 | 78.41 | n/a | n/a | 78.41 | n/a | n/a | 66,203,904 | 0.7646 | 2.00e-04 | n/a | n/a | n/a | n/a | n/a |
| Block-dropped dense student | blockdrop | dense student (A04) | 0 | 1 | 75.25 | n/a | n/a | 75.25 | n/a | n/a | 65,315,328 | 0.7544 | 2.00e-04 | n/a | n/a | n/a | n/a | n/a |
| Original distilled | original | healing without surgery (A04) | 0 | 1 | 78.95 | n/a | n/a | 78.95 | n/a | n/a | 86,583,552 | 1.00 | 1.00e-04 | n/a | n/a | n/a | n/a | n/a |

