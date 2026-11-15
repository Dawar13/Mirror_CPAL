# MUST-08: matched component ablation of Mirror-exact

Written by `scripts/must07_compressors.py` on 2026-11-15 19:27:15, commit `805c231c9c` (uncommitted changes). Compressors: **real**; budget accounting: **real**.

Protocol: k-NN top-1 with the 100k-image train bank (100 per class) and the 50k validation images as queries (CLS after the final norm, cosine, k = 20, T = 0.07, bf16 features). Method selection: the 10k method-selection train images as queries against the same bank; every hyperparameter and gate B use only these. Flip consistency: share of validation images whose prediction equals that of the mirrored image, in bf16 (column flip) and, for full-model rows, in fp32 (column flip fp32; bf16 rounding breaks exact ties, so exact invariance is judged on the fp32 value). Parameters: the converted matrices with basis changes; budget = Mirror-exact's count for the same span and components; gap = (method - budget) / budget. GFLOPs: analytic, per image, 2 x multiply-adds. Values are mean +- standard deviation over fit seeds (n given).

## dinov2_vitb14_reg

Original: top-1 78.94, flip 96.32 (bf16), flip fp32 96.16, method-selection top-1 80.21, 86,583,552 parameters, 47.1 GFLOPs.

### 6mlp (blocks 1 to 6, mlp)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 76.60 +- 0.02 (n = 3) | 78.05 | 91.70 | n/a | 17,034,432 | 0.00% | 75,306,432 (0.870) | 41.2 |  |

### 6attn (blocks 1 to 6, attn)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 77.42 +- 0.02 (n = 3) | 78.92 | 92.54 | n/a | 9,223,392 | 0.00% | 81,651,168 (0.943) | 44.5 |  |

### 6both (blocks 1 to 6, mlp+attn)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 77.34 +- 0.08 (n = 3) | 78.83 | 95.02 | n/a | 25,668,000 | 0.00% | 69,784,224 (0.806) | 38.3 |  |

### fullmlp (blocks 1 to 12, mlp)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 52.55 +- 0.02 (n = 3) | 54.28 | 64.67 | 60.94 | 42,693,344 | 0.00% | 72,653,792 (0.839) | 39.8 |  |
fp32 flip pass: 5,000 validation queries and their mirrors against 10,000 bank images (TF32 off).

### fullattn (blocks 1 to 12, attn)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 62.40 +- 0.26 (n = 3) | 64.64 | 77.27 | 73.72 | 21,861,120 | 0.00% | 80,133,120 (0.926) | 43.7 |  |
fp32 flip pass: 5,000 validation queries and their mirrors against 10,000 bank images (TF32 off).

### full (blocks 1 to 12, mlp+attn)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 53.81 +- 0.36 (n = 3) | 55.29 | 97.98 | 100.00 | 64,554,464 | 0.00% | 66,203,360 (0.765) | 36.4 |  |
fp32 flip pass: 50,000 validation queries and their mirrors against 100,000 bank images (TF32 off).

## Predictions P8.1 and P8.2 (PREREG_NEXT_PHASE.md)

Saving: whole-model parameters of the original minus those of the variant (basis changes counted), mean over fit seeds. P8.1: the MLP-only saving is more than half of the MLP-plus-attention saving. P8.2 (6 blocks): k-NN(MLP only) minus k-NN(MLP plus attention), mean over fit seeds, is at most 0.5.

### 6 blocks: P8.1 PASS, P8.2 PASS

| size | saving | val top-1 |
|---|---|---|
| 6mlp | 11284335 | 76.60 +- 0.02 (n = 3) |
| 6attn | 4936585 | 77.42 +- 0.02 (n = 3) |
| 6both | 16810744 | 77.34 +- 0.08 (n = 3) |

MLP only minus both: -0.74 points over fit seeds [0, 1, 2].

### full model: P8.1 PASS

| size | saving | val top-1 |
|---|---|---|
| fullmlp | 13949551 | 52.55 +- 0.02 (n = 3) |
| fullattn | 6460661 | 62.40 +- 0.26 (n = 3) |
| full | 20410212 | 53.81 +- 0.36 (n = 3) |

## Notes

- Methods 1 and 9 to 12 use one span operator A* (Mirror-exact machinery); for the pairing controls only the MLP neuron pairing changes, the attention keeps the fitted head maps. Their parameter counts equal Mirror-exact's (gap 0).
- Mirror-local counts one d x d rotation per converted block and uses A_b for the attention output; it is an ablation and is not matched to the budget.
- Mirror-exact's basis change at a span end that is not a model boundary is counted (d x d); params_total_absorbed in the CSV gives the count with it absorbed into the next dense block.
- Compressor grid points are scored on the method-selection queries only; the chosen point is then scored on validation.
- Random pairing and random structure use seed random_seed + fit_seed, so their spread over fit seeds includes the random draw.
- Biases, norms and embeddings are counted as dense; analytic GFLOPs exclude biases, norms and activations.

Files: rows_<model>.csv, grid_<model>.csv, bootstrap.csv, preds/, info/, meta.json.
