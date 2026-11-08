# V-exact healing, k = 4 (full conversion) (TINY TEST, random weights and synthetic data)

Model `tiny random timm ViT` (4 blocks); blocks 1 to 4 converted with one operator A* (d+ 36, d- 28), MLP and attention, exactly mirror-equivariant over the span with the even head (final features of an image and its mirror equal). A* and register pairing: `tiny random span operator (smoke test)`; neuron pairings: `tiny random structure (smoke test)`; head maps: `tiny random head structure (smoke test)`. Seed 0; git `53c8aada92`. Written 2026-11-09 00:49:21.

Recipe (as the M3 pilot): 4 steps of 8 images (32 images seen, 32 healing images, train split), AdamW, cosine schedule, fp32; learning rate 1e-04 chosen from {1e-04, 1e-03} on 8 held-out train images. Targets: block outputs centred, symmetrised with A*; final target symmetrised. Embeddings: trained, re-projected every step.

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

Cost: the converted blocks keep 100,960 of 196,608 matrix parameters (0.514); whole model 208,776 parameters (0.686 of 304,424; biases and embeddings counted dense, as in M1). Trainable while healing: 237,760 parameters. Final run 1 s of training; whole script 19 s.

Notes:
- One seed. Held-out losses use the V-exact targets (centred, symmetrised), so they are not comparable with the M3 pilot's.
- In bf16 the mirrored computation rounds differently, so the bf16 flip consistency of the fully converted model can fall slightly below 100%; the exact check is the fp32 one.

Files: summary.json, knn.json, exactness.json, flipcheck.json, final/ (metrics.json, train_log.csv, heldout_log.csv, trained_blocks.pt), lr_selection/.
