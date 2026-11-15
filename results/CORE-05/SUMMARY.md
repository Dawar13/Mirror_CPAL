# CORE-05: why surgery is cheap

Commit 805c231c9c (dirty); PREREG_NEXT_PHASE.md sha256 711b963039af. Statistics on fit set 0 (ImageNet train); single-block k-NN on the method-selection queries (design) and validation (scoring).

| Prediction | Verdict |
|---|---|
| P5.1: the activation-weighted breaking share is at most half the weight-energy breaking share in blocks 1 to 6 | FAIL |
| P5.2: over training (CORE-01 LR and UD runs), breaking energy in the top-variance band falls faster than in the bottom band | n/a |
| P5.3: the out-of-sample rule, designed on dinov2_vitb14_reg only and frozen before scoring, is correct on at least 75% of layer-model cases on DINOv2-L-reg, DeiT-III and MAE | FAIL |

## P5.1: activation-weighted against weight-energy breaking share, blocks 1 to 6 (fc1, fc2)

| Model | Operator | s_w | s_cov | s_cov / s_w | s_mom / s_w | Pass |
|---|---|---|---|---|---|---|
| dinov2_vitb14_reg | perlayer | 0.200 | 0.115 | 0.574 | 0.384 | no |
| dinov2_vitb14_reg | star | 0.254 | 0.147 | 0.580 | 0.341 | no |
| dinov2_vitb14_reg | star6 | 0.245 | 0.137 | 0.560 | 0.328 | no |
| dinov2_vitl14_reg | perlayer | 0.222 | 0.099 | 0.444 | 0.312 | yes |
| dinov2_vitl14_reg | star | 0.230 | 0.124 | 0.537 | 0.337 | no |
| dinov2_vitl14_reg | star6 | 0.210 | 0.096 | 0.457 | 0.309 | yes |
| deit3_b16 | perlayer | 0.157 | 0.112 | 0.712 | 0.538 | no |
| deit3_b16 | star | 0.207 | 0.141 | 0.683 | 0.470 | no |
| deit3_b16 | star6 | 0.192 | 0.128 | 0.666 | 0.489 | no |
| mae_b16 | perlayer | 0.226 | 0.133 | 0.589 | 0.430 | no |
| mae_b16 | star | 0.267 | 0.148 | 0.552 | 0.366 | no |
| mae_b16 | star6 | 0.265 | 0.148 | 0.561 | 0.388 | no |

## P5.2: band split over training (CORE-01)

| Run | Group | Epochs | Fall top | Fall bottom | Pass |
|---|---|---|---|---|---|

## P5.3: the out-of-sample rule

Frozen 2026-11-15T16:36:17Z on dinov2_vitb14_reg: rule mom_plain (second moment M, SVD energy plain truncation applied), rule sha256 68ab3dff1811.
Design candidates (method-selection truth): cov_plain 6 of 12; cov_white 4 of 12; mom_plain 7 of 12; mom_white 4 of 12.
Scored: 32 of 48 correct (0.667), bar 0.75: FAIL. Per model: dinov2_vitl14_reg 13 of 24; deit3_b16 11 of 12; mae_b16 8 of 12.
Secondary: method-selection truth 33 of 48; feature-distortion truth 42 of 48; F4-style rule on the same cases 26 of 48.
Earlier F4 rule (PREREG_E2.md P2, /home/ubuntu/Mirror_CPAL/runs/f4/f4_p2.json): 16 of 36, FAIL.

## Flatness and plain SVD (reported)

| Model | erank W (1-6) | erank W Sigma^1/2 (1-6) | erank input (1-6) | SVD loss 6mlp | SVD loss full |
|---|---|---|---|---|---|
| dinov2_vitb14_reg | 0.815 | 0.629 | 0.698 | 69.57 | 73.77 |
| dinov2_vitl14_reg | 0.566 | 0.351 | 0.499 | 0.02 | 19.97 |
| deit3_b16 | 0.669 | 0.594 | 0.696 | 1.68 | 8.25 |
| mae_b16 | 0.840 | 0.662 | 0.664 | - | - |

Files: measures_<model>.csv, truth_<model>.csv, rule_frozen.json, rule_score.json, ckpt_bands.csv, spectra/, preds/, metrics.json, meta.json.
