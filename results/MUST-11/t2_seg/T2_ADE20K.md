# Tier 2: ADE20K linear segmentation

Protocol (PAPER_EXPERIMENTS.md, Section 1, Tier 2; script `scripts/tier2_ade20k.py`, recipe t2seg-v1): frozen backbone (bf16 autocast), last-layer patch tokens after the final norm (CLS and registers excluded), head = BatchNorm then a 1 x 1 convolution to 150 classes. Training: random 448 x 448 crops of train images resized to a shorter side of 448 px, random horizontal flip, batch 16, 8,000 iterations, AdamW (lr 0.001, weight decay 0.0001, 300 warm-up iterations, cosine), cross-entropy on bilinearly upsampled logits, label 0 ignored. The crop schedule and the head initialisation come from the seed, so every variant sees the same crops. Evaluation: validation images resized to a shorter side of 448 px, 448 px sliding windows (stride 298), logits averaged over windows and upsampled to the original label resolution; single scale, no flip. mIoU, aAcc (pixel accuracy) and mAcc (mean class accuracy) in %.

Data: Hugging Face `merve/scene_parse_150` at revision `b83d71b405b3` (parquet copy of scene_parse_150; train 20,210 images, validation 2,000) (verify on VM).

## dinov2_vitb14_reg

| Variant | What it is | mIoU | aAcc | mAcc | Change vs original (mIoU) | Notes |
|---|---|---|---|---|---|---|
| original | original | 47.59 | 80.98 | 59.47 | - | training resumed at iteration 5000 |
| best_svdllm_s0 | healed model from a run directory (mirror.heal_structured.load_healed_model) | 46.79 | 80.87 | 58.57 | -0.80 | - |
| vexact_s0 | healed model from a run directory (mirror.heal_structured.load_healed_model) | 47.16 | 81.02 | 58.89 | -0.43 | - |

## Mask equivariance (MUST-11)

Share of pixels where the mask of the mirrored image equals the mirrored mask (all pixels at label resolution), fp32 without autocast, TF32 off, with a flip-symmetric window layout (the mirrored image uses the mirrored windows), so only the model and the head are tested. Raw: the trained linear head. Symmetrised: (head(z) + head(T z)) / 2 with T applying the operator in the last column to the normalised final tokens before the final affine. Mirror-exact models with an even head use their full tokens (even projection bypassed) unless the run used --even-tokens; their tokens are then exactly A*-equivariant, so the symmetrised head is exact and the raw head is not. mIoU columns: the same pass scored on the original images (fp32), beside the bf16 protocol mIoU of the main table.

| Model | Variant | Raw head (%) | Pixels differing | Symmetrised head (%) | Pixels differing | mIoU raw / symmetrised (fp32) | Images | Operator |
|---|---|---|---|---|---|---|---|---|
| dinov2_vitb14_reg | original | 95.0309 | 24,232,674 | 94.7413 | 25,644,765 | 47.58 / 47.57 | 2000 | A_L: last-layer per-layer operator (layer 12) from /home/ubuntu/Mirror_CPAL/runs/x2/perlayer_reg_s0/results/e1_operators.npz |
| dinov2_vitb14_reg | best_svdllm_s0 | 94.7590 | 25,558,433 | 94.6249 | 26,212,718 | 46.74 / 46.74 | 2000 | A_L: last-layer per-layer operator (layer 12) from /home/ubuntu/Mirror_CPAL/runs/x2/perlayer_reg_s0/results/e1_operators.npz |
| dinov2_vitb14_reg | vexact_s0 | 98.9771 | 4,988,566 | 99.9999 | 422 | 47.13 / 47.15 | 2000 | A* of the model's even head (full tokens: exactly A*-equivariant before the final affine) |

Source: `t2_ade20k.csv` and `seg/<model>/<variant>.json` in this folder. Commit(s): 805c231c9c (dirty); seed(s): 0. Timings and peak GPU memory are in the CSV.
