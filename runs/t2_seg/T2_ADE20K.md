# Tier 2: ADE20K linear segmentation

Protocol (PAPER_EXPERIMENTS.md, Section 1, Tier 2; script `scripts/tier2_ade20k.py`, recipe t2seg-v1): frozen backbone (bf16 autocast), last-layer patch tokens after the final norm (CLS and registers excluded), head = BatchNorm then a 1 x 1 convolution to 150 classes. Training: random 448 x 448 crops of train images resized to a shorter side of 448 px, random horizontal flip, batch 16, 8,000 iterations, AdamW (lr 0.001, weight decay 0.0001, 300 warm-up iterations, cosine), cross-entropy on bilinearly upsampled logits, label 0 ignored. The crop schedule and the head initialisation come from the seed, so every variant sees the same crops. Evaluation: validation images resized to a shorter side of 448 px, 448 px sliding windows (stride 298), logits averaged over windows and upsampled to the original label resolution; single scale, no flip. mIoU, aAcc (pixel accuracy) and mAcc (mean class accuracy) in %.

Data: Hugging Face `merve/scene_parse_150` at revision `b83d71b405b3` (parquet copy of scene_parse_150; train 20,210 images, validation 2,000) (verify on VM).

## dinov2_vitb14_reg

| Variant | What it is | mIoU | aAcc | mAcc | Change vs original (mIoU) | Notes |
|---|---|---|---|---|---|---|
| original | original | 47.59 | 80.98 | 59.47 | - | - |
| healed_vexact_k6 | healed Mirror V-exact, MLP + attention, k = 6 (scripts/heal_vexact.py) | 47.51 | 80.96 | 59.30 | -0.09 | rebuilt at 448 px; max relative error inside the span 4.0e-06 |
| mirror_mlp_k6 | Mirror Surgery, MLP only, V-local, k = 6 | 46.88 | 80.67 | 58.41 | -0.72 | - |
| svd_mlp_k6 | SVD at matched parameters, MLP only, k = 6 | 9.84 | 57.19 | 13.60 | -37.75 | - |
| vexact_k6 | Mirror V-exact, MLP + attention, k = 6 | 47.19 | 80.83 | 58.89 | -0.40 | rebuilt at 448 px; max relative error inside the span 4.8e-06 (224 px build run at 448 px: 4.3e-06) |

V-exact: `mirror.m1.build_vexact` projects the position embedding on the grid in use, and DINOv2 re-interpolates its position embedding when the input grid changes (bicubic with an offset, not flip-symmetric). The model built at 224 px for M1 is therefore not exactly equivariant at 448 px. Here V-exact is rebuilt with the same recipe at the segmentation resolution (same A*, pairings, head maps and register pairing; the position grid of 448 px inputs materialised and projected), and every forward pass is exactly one 448 x 448 window. The relative errors in the notes are measured on two training crops in fp32 (M1 bar 0.0001).

Source: `t2_ade20k.csv` and `seg/<model>/<variant>.json` in this folder. Commit(s): 6f35281c33, b95ccd5886; seed(s): 0. Timings and peak GPU memory are in the CSV.
