# CORE-07: distribution shift

Commit 7d05b6439c (dirty); PREREG_NEXT_PHASE.md sha256 711b963039af.

**P7.1** (Mirror-exact's drop from ImageNet to ImageNet-V2 (matched frequency, linear probe) is within 1 point of the original's drop): **PASS**

Drop to ImageNet-V2: original 8.62; Mirror-exact 9.17 (mean of 3 seeds); difference +0.55 points (bound 1.0); bootstrap 95% interval [+0.08, +1.02] (10000 resamples, seed 0).

| Variant | Role | ImageNet val | ImageNet-V2 MF | drop | Sketch | drop | Probe |
|---|---|---|---|---|---|---|---|
| original | original | 81.00 | 72.38 | 8.62 | 52.77 | 28.23 | /home/Mirror_CPAL/runs/final/A10/t2/probe/dinov2_vitb14_reg/original.json |
| vexact_s0 | mirror_exact | 80.12 | 71.04 | 9.08 | 49.37 | 30.75 | /home/Mirror_CPAL/runs/final/A10/t2/probe/dinov2_vitb14_reg/vexact_s0.json |
| vexact_s1 | mirror_exact | 80.05 | 70.82 | 9.23 | 49.41 | 30.65 | /home/Mirror_CPAL/runs/final/A10/t2/probe/dinov2_vitb14_reg/vexact_s1.json |
| vexact_s2 | mirror_exact | 80.17 | 70.96 | 9.21 | 49.36 | 30.80 | /home/Mirror_CPAL/runs/final/A10/t2/probe/dinov2_vitb14_reg/vexact_s2.json |
| best_svdllm_s0 | best_baseline | 79.70 | 70.32 | 9.38 | 45.15 | 34.55 | /home/Mirror_CPAL/runs/final/A10/t2/probe/dinov2_vitb14_reg/best_svdllm_s0.json |
| best_svdllm_s1 | best_baseline | 79.77 | 70.27 | 9.50 | 45.28 | 34.49 | /home/Mirror_CPAL/runs/final/A10/t2/probe/dinov2_vitb14_reg/best_svdllm_s1.json |
| best_svdllm_s2 | best_baseline | 79.81 | 70.52 | 9.29 | 46.12 | 33.70 | /home/Mirror_CPAL/runs/final/A10/t2/probe/dinov2_vitb14_reg/best_svdllm_s2.json |

Notes:
- ImageNet-Sketch pads some classes with flipped and rotated copies of other images (configs/data/shift_sets.yaml); an exactly invariant model scores such a flipped pair alike, so Sketch numbers are reported with this caveat and also without one image of each detected near-duplicate pair
- the probe of each model is the tier2_linear ImageNet probe trained on the bank; nothing is fitted on validation or on the shifted sets
- Sketch near-duplicate pairs (heuristic, threshold 0.02): 32992 ({'identity': 3586, 'flip_lr': 16152, 'flip_ud': 6707, 'rot90': 25, 'rot180': 6475, 'rot270': 17, 'transpose': 15, 'antitranspose': 15}); 28877 at half and 44621 at twice the threshold. Top-1 without one image of each pair: column sketch_dedup_top1 of shift_rows.csv.

Files: shift_rows.csv, metrics.json, preds/, features/, meta.json.
