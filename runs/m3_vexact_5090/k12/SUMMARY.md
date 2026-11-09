# V-exact healing, k = 12 (full conversion)

Model `dinov2_vitb14_reg` (12 blocks); blocks 1 to 12 converted with one operator A* (d+ 700, d- 68), MLP and attention, exactly mirror-equivariant over the span with the even head (final features of an image and its mirror equal). A* and register pairing: `/workspace/Mirror_CPAL/runs/m1/m1_structure.npz`; neuron pairings: `/workspace/Mirror_CPAL/runs/x2/perlayer_reg_s0/results/e1_operators.npz`; head maps: `/workspace/Mirror_CPAL/runs/m1/m1_structure.npz`. Seed 0; git `6f35281c33`. Written 2026-11-09 09:26:20.

Recipe: the M3 pilot's budget, data, optimiser, schedule and learning-rate grid, with the differences listed under "Deviations from the M3 pilot" below. 780 steps of 256 images (199680 images seen, 100000 healing images, train split), AdamW, cosine schedule, bf16 autocast; learning rate 1e-04 chosen from {1e-05, 5e-05, 1e-04} on 5000 held-out train images. Targets: block outputs centred, symmetrised with A*; final target symmetrised. Embeddings: trained, re-projected every step.

| model | k-NN top-1 | drop (points) | flip consistency (bf16) |
|---|---|---|---|
| original | 78.93 | | 96.28 |
| V-exact, zero-shot | 53.83 | 25.11 | 98.04 |
| V-exact, healed | 78.34 | 0.59 | 99.74 |

k-NN: v2 protocol (100000 bank, 50000 queries), k = 20, T = 0.07, bf16 features.

Held-out distillation loss (5000 held-out train images, V-exact targets): before 7.3170, after 1.3469.
Exactness after healing (float32, 64 validation images): largest relative error inside the span 8.7e-06 (tolerance 0.0001): **PASS**.
Flip check after healing (float32, TF32 off; 5000 validation queries, bank 10000 train images): flip consistency 100.00%, **PASS** (must be 100%); largest relative CLS feature difference between an image and its mirror 4.8e-06.
Constraint checks at the end of training (0 = exact): MirrorMLPConstraint 2.6e-08; MirrorAttnConstraint 1.0e-08; VExactEmbeddings embedding share 2.5e-16.

Cost: the converted blocks keep 64,554,464 of 84,934,656 matrix parameters (0.760); whole model 66,203,360 parameters (0.765 of 86,583,552; biases and embeddings counted dense, as in M1). Trainable while healing: 85,671,168 parameters. Final run 2471 s of training, peak GPU memory 5.44 GB; whole script 4703 s.

## Deviations from the M3 pilot

The M3 pilot (scripts/heal.py, configs/m3_pilot.yaml) trains only the converted blocks, against the teacher's own outputs. This run differs in two ways (a description; the recipe is as configured):

1. Embeddings trained and re-projected: cls_token, pos_embed, register_tokens, patch_embed.proj.weight, patch_embed.proj.bias train (train_scope "all") and are re-projected with A* after every optimiser step, so they stay exactly equivariant. Extra trainable parameters compared with the pilot, read from this run: 653,568 (of 85,671,168 trainable).
2. Symmetrised targets: the teacher's block outputs are centred over channels and symmetrised with A* (the mean of the output for the image and the aligned, A*-mapped output for its mirror); the teacher runs on every batch and on its mirror. At full conversion the final targets (CLS and patch tokens) are the flip average of the teacher's final-norm features for the image and its mirror, and the loss normalisers (fixed) are measured on these targets, the CLS one included. Under random flips (p = 0.5) the expected loss equals that of the raw targets up to a constant; the change removes the part an exactly equivariant student cannot fit.

Notes:
- One seed. Held-out losses use the V-exact targets (centred, symmetrised), so they are not comparable with the M3 pilot's.
- In bf16 the mirrored computation rounds differently, so the bf16 flip consistency of the fully converted model can fall slightly below 100%; the exact check is the fp32 one.

Files: summary.json, knn.json, exactness.json, flipcheck.json, final/ (metrics.json, train_log.csv, heldout_log.csv, trained_blocks.pt), lr_selection/.
