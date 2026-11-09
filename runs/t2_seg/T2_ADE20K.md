# Tier 2: ADE20K linear segmentation

Protocol (PAPER_EXPERIMENTS.md, Section 1, Tier 2; script `scripts/tier2_ade20k.py`, recipe t2seg-v1): frozen backbone (bf16 autocast), last-layer patch tokens after the final norm (CLS and registers excluded), head = BatchNorm then a 1 x 1 convolution to 150 classes. Training: random 448 x 448 crops of train images resized to a shorter side of 448 px, random horizontal flip, batch 16, 8,000 iterations, AdamW (lr 0.001, weight decay 0.0001, 300 warm-up iterations, cosine), cross-entropy on bilinearly upsampled logits, label 0 ignored. The crop schedule and the head initialisation come from the seed, so every variant sees the same crops. Evaluation: validation images resized to a shorter side of 448 px, 448 px sliding windows (stride 298), logits averaged over windows and upsampled to the original label resolution; single scale, no flip. mIoU, aAcc (pixel accuracy) and mAcc (mean class accuracy) in %.

Data: Hugging Face `merve/scene_parse_150` at revision `b83d71b405b3` (parquet copy of scene_parse_150; train 20,210 images, validation 2,000) (verify on VM).

## dinov2_vitb14_reg

| Variant | What it is | mIoU | aAcc | mAcc | Change vs original (mIoU) | Notes |
|---|---|---|---|---|---|---|
| original | original | 47.59 | 80.98 | 59.47 | - | - |

Source: `t2_ade20k.csv` and `seg/<model>/<variant>.json` in this folder. Commit(s): b95ccd5886; seed(s): 0. Timings and peak GPU memory are in the CSV.
