# T3: other models (A08): DeiT-III-B/16 (classifier top-1 and k-NN) and DINOv2-L/14 reg (k-NN, indicative)

Written by `scripts/collect_final_phase.py`; every row's source file and commit are in `T3_other_models.csv` (columns source_file and git_hash).

| arm | folder | role | seeds | n | s0 | s1 | s2 | mean | sd | flip_fp32_min | params_model | params_frac | lr | classifier_top1_mean | classifier_n | model |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mirror | mirror_exact | deit3_b16 | 0,1,2 | 3.00 | 82.09 | 82.06 | 82.06 | 82.07 | 0.0185 | n/a | 72212872.00 | 0.8340 | 2.00e-04 | 82.70 | 3.00 | deit3_b16 |
| SVD-LLM | svdllm | deit3_b16 | 0,1,2 | 3.00 | 82.20 | 82.17 | 82.16 | 82.18 | 0.0178 | n/a | 72212968.00 | 0.8340 | 2.00e-04 | 82.88 | 3.00 | deit3_b16 |
| Random structure | random_structure | deit3_b16 | 0 | 1.00 | 79.70 | n/a | n/a | 79.70 | n/a | n/a | 72212872.00 | 0.8340 | 2.00e-04 | 80.13 | 1.00 | deit3_b16 |
| Mirror | mirror_exact | dinov2_vitl14_reg |  | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | dinov2_vitl14_reg |
| SVD-LLM | svdllm | dinov2_vitl14_reg |  | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | dinov2_vitl14_reg |
| Original (released) | _original | deit3_b16 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 83.68 | 1.00 | deit3_b16 |

