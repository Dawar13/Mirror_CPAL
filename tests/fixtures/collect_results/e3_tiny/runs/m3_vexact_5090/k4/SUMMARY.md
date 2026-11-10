# V-exact healing, k = 4 (full conversion) (TINY TEST, random weights and synthetic data)

Model `tiny random timm ViT` (4 blocks); blocks 1 to 4 converted with one operator A* (d+ 36, d- 28), MLP and attention, exactly mirror-equivariant over the span with the even head (final features of an image and its mirror equal). A* and register pairing: `tiny random span operator (smoke test)`; neuron pairings: `tiny random structure (smoke test)`; head maps: `tiny random head structure (smoke test)`. Seed 0; git `aeb57bed18`. Written 2026-11-09 23:50:19.

Recipe: the M3 pilot's budget, data, optimiser, schedule and learning-rate grid, with the differences listed under "Deviations from the M3 pilot" below. 4 steps of 8 images (32 images seen, 32 healing images, train split), AdamW, cosine schedule, fp32; learning rate 1e-04 chosen from {1e-04, 1e-03} on 8 held-out train images. Targets: block outputs centred, symmetrised with A*; final target symmetrised. Embeddings: trained, re-projected every step.

| model | k-NN top-1 | drop (points) | flip consistency (fp32) |
|---|---|---|---|
| original | 56.25 | | 37.50 |
| V-exact, zero-shot | 62.50 | -6.25 | 100.00 |
| V-exact, healed | 62.50 | -6.25 | 100.00 |

k-NN: tiny synthetic protocol (32 bank, 16 queries), k = 20, T = 0.07, fp32 features.

Held-out distillation loss (8 held-out train images, V-exact targets): before 1.2722, after 1.0666.
Exactness after healing (float32, 8 validation images): largest relative error inside the span 9.3e-07 (tolerance 0.0001): **PASS**.
Flip check after healing (float32, TF32 off; 16 validation queries, bank 32 train images): flip consistency 100.00%, **PASS** (must be 100%); largest relative CLS feature difference between an image and its mirror 5.4e-07.
Constraint checks at the end of training (0 = exact): MirrorMLPConstraint 4.4e-08; MirrorAttnConstraint 1.8e-08; VExactEmbeddings embedding share 3.8e-16.

Cost: the converted blocks keep 100,960 of 196,608 matrix parameters (0.514); whole model 208,776 parameters (0.686 of 304,424; biases and embeddings counted dense, as in M1). Trainable while healing: 237,760 parameters. Final run 1 s of training; whole script 15 s.

## Deviations from the M3 pilot

The M3 pilot (scripts/heal.py, configs/m3_pilot.yaml) trains only the converted blocks, against the teacher's own outputs. This run differs in two ways (a description; the recipe is as configured):

1. Embeddings trained and re-projected: cls_token, pos_embed, patch_embed.proj.weight, patch_embed.proj.bias train (train_scope "all") and are re-projected with A* after every optimiser step, so they stay exactly equivariant. Extra trainable parameters compared with the pilot, read from this run: 38,848 (of 237,760 trainable).
2. Symmetrised targets: the teacher's block outputs are centred over channels and symmetrised with A* (the mean of the output for the image and the aligned, A*-mapped output for its mirror); the teacher runs on every batch and on its mirror. At full conversion the final targets (CLS and patch tokens) are the flip average of the teacher's final-norm features for the image and its mirror, and the loss normalisers (fixed) are measured on these targets, the CLS one included. Under random flips (p = 0.5) the expected loss equals that of the raw targets up to a constant; the change removes the part an exactly equivariant student cannot fit.

Notes:
- One seed. Held-out losses use the V-exact targets (centred, symmetrised), so they are not comparable with the M3 pilot's.
- In bf16 the mirrored computation rounds differently, so the bf16 flip consistency of the fully converted model can fall slightly below 100%; the exact check is the fp32 one.

Files: summary.json, knn.json, exactness.json, flipcheck.json, final/ (metrics.json, train_log.csv, heldout_log.csv, trained_blocks.pt), lr_selection/.
