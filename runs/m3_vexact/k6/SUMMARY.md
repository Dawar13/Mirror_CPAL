# V-exact healing, k = 6

Model `dinov2_vitb14_reg` (12 blocks); blocks 1 to 6 converted with one operator A* (d+ 552, d- 216), MLP and attention, exactly mirror-equivariant over the span. A* and register pairing: `/workspace/Mirror_CPAL/runs/m1/m1_structure.npz`; neuron pairings: `/workspace/Mirror_CPAL/runs/x2/perlayer_reg_s0/results/e1_operators.npz`; head maps: `/workspace/Mirror_CPAL/runs/m1/m1_structure.npz`. Seed 0; git `b95ccd5886`. Written 2026-11-09 07:21:21.

Recipe (as the M3 pilot): 780 steps of 256 images (199680 images seen, 100000 healing images, train split), AdamW, cosine schedule, bf16 autocast; learning rate 5e-05 chosen from {1e-05, 5e-05, 1e-04} on 5000 held-out train images. Targets: block outputs centred, symmetrised with A*; final target raw (no even head). Embeddings: trained, re-projected every step.

| model | k-NN top-1 | drop (points) | flip consistency (bf16) |
|---|---|---|---|
| original | 78.93 | | 96.28 |
| V-exact, zero-shot | 77.37 | 1.56 | 94.99 |
| V-exact, healed | 78.79 | 0.14 | 96.38 |

k-NN: v2 protocol (100000 bank, 50000 queries), k = 20, T = 0.07, bf16 features.

Held-out distillation loss (5000 held-out train images, V-exact targets): before 1.1558, after 0.2918.
Exactness after healing (float32, 64 validation images): largest relative error inside the span 4.6e-06 (tolerance 0.0001): **PASS**.
Constraint checks at the end of training (0 = exact): MirrorMLPConstraint 2.2e-08; MirrorAttnConstraint 9.7e-09; VExactEmbeddings embedding share 2.1e-16.

Cost: the converted blocks keep 25,078,176 of 42,467,328 matrix parameters (0.591); whole model 69,194,400 parameters (0.799 of 86,583,552; biases and embeddings counted dense, as in M1). Trainable while healing: 43,162,368 parameters. Final run 2949 s of training, peak GPU memory 4.16 GB; whole script 4823 s.

Notes:
- One seed. Held-out losses use the V-exact targets (centred, symmetrised), so they are not comparable with the M3 pilot's.
- In bf16 the mirrored computation rounds differently, so the bf16 flip consistency of the fully converted model can fall slightly below 100%; the exact check is the fp32 one.

Files: summary.json, knn.json, exactness.json, final/ (metrics.json, train_log.csv, heldout_log.csv, trained_blocks.pt), lr_selection/.
