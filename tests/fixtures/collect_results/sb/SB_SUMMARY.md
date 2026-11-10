# Strong baselines, zero-shot (M3-grid methods)

Written by `scripts/strong_baselines.py` on 2026-11-09 23:47:27, commit `aeb57bed18`. Model `tiny`.

Each method compresses the MLPs of blocks 1 to k and nothing else, at the parameter budget of Mirror V-local, MLP only, at the same k: the MLP weight count of the converted blocks (biases excluded). Statistics come from the fit run's 10 train fit images (no mirrored copies); k-NN uses the paper protocol (40 train bank images, 20 validation queries, k = 20, T = 0.07, fp32 on CPU features).

- **asvd**: activation-aware SVD in the SVD-LLM style. Each folded fc1 and fc2 is whitened by the Cholesky factor of its input second moment (with a small ridge), truncated and un-whitened. Ranks are those of the plain SVD baseline (largest rank whose factors fit each matrix's Mirror count).
- **slicegpt**: SliceGPT-style PCA slicing. Per block, the principal directions of the residual stream entering the MLP (folded RMSNorm form, mirror/fold.py) are computed; fc1 reads only the top d_in of them and fc2 writes only into the top d_out, with d_in = floor(Mirror fc1 count / h) and d_out = floor(Mirror fc2 count / h). The residual stream, the norms, attention and the hidden width are not sliced. The change of stream basis between blocks (one d x d matrix per block in an implementation) is not counted, as Mirror V-local's basis rotations are not counted in its MLP budget; it is recorded as basis_change_params.
- **prune_act**: structured width pruning. Neurons are scored by mean |activation| on the fit images times the norm of their fc2 column (LayerScale included); per block the floor(Mirror block count / (2 d)) highest-scoring neurons are kept and the others removed.

Every allocation rounds down, so a baseline never holds more MLP weights than Mirror. Ratio = MLP weights kept over the Mirror budget (the column params_vs_mirror_first_k of the CSV).

| method | k | fit seed | top-1 | drop | flip consistency | MLP weights kept | Mirror budget | ratio | within 1% |
|---|---|---|---|---|---|---|---|---|---|
| asvd | 1 | 0 | 50.00 | 20.00 | 70.00 | 16,000 | 16,448 | 0.9728 | NO |
| slicegpt | 1 | 0 | 60.00 | 10.00 | 45.00 | 16,384 | 16,448 | 0.9961 | yes |
| prune_act | 1 | 0 | 65.00 | 5.00 | 55.00 | 16,384 | 16,448 | 0.9961 | yes |
| asvd | 2 | 0 | 45.00 | 25.00 | 55.00 | 32,000 | 32,936 | 0.9716 | NO |
| slicegpt | 2 | 0 | 45.00 | 25.00 | 45.00 | 32,768 | 32,936 | 0.9949 | yes |
| prune_act | 2 | 0 | 60.00 | 10.00 | 45.00 | 32,768 | 32,936 | 0.9949 | yes |

Original model: top-1 70.00, flip consistency 45.00 (same protocol; drop = original minus method).

Mirror V-local, MLP only and plain SVD at the same k and fit seed are in `runs/a/tiny/A1_s<fit seed>.csv` (variants per_layer and svd); the canonical Mirror values are those of X2 in `results/paper` (`T_main_zeroshot`).

Caveats: statistics are taken from the original model for every block (not from a model whose earlier blocks are already compressed); slicegpt is evaluated in the absorbed form of the unfolded model, which computes the same function as the sliced folded model (tested in float64) and shares the numerics of the other variants; prune_act has no bias compensation for the removed neurons.
