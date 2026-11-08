# Analytic cost model

Function `analytic_cost` in `scripts/collect_results.py`. These are analytic counts, not measured latency.

## Assumptions

- Analytic counts at 224 px, not measured latency (M2 measures latency; not run).
- MACs = multiply-accumulates per image; GFLOPs per image = 2 x MACs / 1e9.
- Counted: patch embedding n_patches x (in_chans p^2) x d; per block and token: attention projections 4 d^2 (qkv 3 d^2 plus proj d^2) and MLP 2 h d; per block and image: attention token mixing 2 T^2 d (q k^T and attention times v), which surgery does not change.
- Not counted: biases, norms, softmax, GELU, LayerScale, residual additions, the twin butterfly additions (u + v, u - v) and the k-NN head.
- Mirror MLP: each matrix costs its equivariant free-parameter count (h d + tr(P) tr(A)) / 2 per token (eq_params_fc1, eq_params_fc2 of e1_mlp.csv; mlp_eq_fc1, mlp_eq_fc2 of m1_ranks.csv), i.e. every free parameter is used once per token in even/odd coordinates.
- Mirror attention (M1): the free-parameter counts of Wq, Wk, Wv, Wo per block (attn_free_q/k/v/o of m1_ranks.csv for V-local; attn_free_matrices and mlp_free of m1_vexact_blocks.csv for V-exact, which depend on the span k).
- Basis rotations: every variant with per-layer operators (Mirror MLP only, V-local, and the random controls, which use random per-layer involutions) pays one d x d rotation per converted block (d^2 MACs per token): the stream is rotated from the basis of A_(l-1) to that of A_l at each converted block, the rotation into the first basis is folded into the linear patch embedding and the one after the last converted block into the next dense block. V-exact (one operator for the span) and SVD need none. Rotation matrices are d^2 stored parameters each; they are reported in rotation_params and are not included in model_params (which follows m1_knn.csv).
- SVD: r (in + out) per matrix (ranks svd_rank and svd_rank_fc2 of e1_mlp.csv; svd_r_* of m1_ranks.csv).
- Whole-model parameters: the dense model's total minus the dense matrix parameters of the converted matrices plus the kept ones (biases stay dense), as in m1_knn.csv.

## Inputs per model

- DINOv2-reg: d 768, h 3072, depth 12, T 261 tokens (256 patches plus 5 prefix), patch 14, in_chans 3, dense parameters 86,583,552.
  - d, h, depth: runs/x2/perlayer_reg_s0/results/e1_mlp.csv
  - in_chans: assumption: RGB input
  - T: n_tokens / split_sizes.heldout (runs/x2/perlayer_reg_s0/results/e1_layers.csv, runs/x2/perlayer_reg_s0/results/e1_meta.json)
  - grid: runs/f3/f3_dinov2_vitb14_reg.json data.grid_hw
  - patch: crop 224 (runs/x2/perlayer_reg_s0/results/e1_meta.json config.data.crop) / grid width
  - model_params_dense: runs/m1/m1_knn.csv (k = 0, original, model_params)
- DINOv2: d 768, h 3072, depth 12, T 257 tokens (256 patches plus 1 prefix), patch 14, in_chans 3, dense parameters 86,580,480.
  - d, h, depth: runs/x2/perlayer_plain_s0/results/e1_mlp.csv
  - in_chans: assumption: RGB input
  - T: n_tokens / split_sizes.heldout (runs/x2/perlayer_plain_s0/results/e1_layers.csv, runs/x2/perlayer_plain_s0/results/e1_meta.json)
  - grid: runs/f3/f3_dinov2_vitb14.json data.grid_hw
  - patch: crop 224 (runs/x2/perlayer_plain_s0/results/e1_meta.json config.data.crop) / grid width
  - model_params_dense: runs/m1/m1_knn.csv original model_params of dinov2_vitb14_reg minus n_registers x d (runs/m1/m1_meta.json: n_registers 4); the two models differ only by the register tokens (verify against the hub model)

## Checks against counts recorded by the runs

- 75 parameter counts recomputed from the per-block CSVs and compared with the totals the runs recorded (mlp_params_kept_frac of e1_surgery.csv; mlp_params_kept, attn_params_kept, model_params of m1_knn.csv): largest relative difference 0.00e+00 (`cost_model_checks.csv`).
- DINOv2-reg dense: 47.08 GFLOPs per image (analytic).
- DINOv2 dense: 46.32 GFLOPs per image (analytic).
