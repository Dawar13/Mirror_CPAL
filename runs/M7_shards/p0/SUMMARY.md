# MUST-07: zero-shot comparison at matched parameters

Written by `scripts/must07_compressors.py` on 2026-11-15 17:53:24, commit `805c231c9c` (uncommitted changes). Compressors: **real**; budget accounting: **real**.

Protocol: k-NN top-1 with the 100k-image train bank (100 per class) and the 50k validation images as queries (CLS after the final norm, cosine, k = 20, T = 0.07, bf16 features). Method selection: the 10k method-selection train images as queries against the same bank; every hyperparameter and gate B use only these. Flip consistency: share of validation images whose prediction equals that of the mirrored image, in bf16 (column flip) and, for full-model rows, in fp32 (column flip fp32; bf16 rounding breaks exact ties, so exact invariance is judged on the fp32 value). Parameters: the converted matrices with basis changes; budget = Mirror-exact's count for the same span and components; gap = (method - budget) / budget. GFLOPs: analytic, per image, 2 x multiply-adds. Values are mean +- standard deviation over fit seeds (n given).

## dinov2_vitb14_reg

Original: top-1 78.94, flip 96.32 (bf16), flip fp32 96.30, method-selection top-1 80.21, 86,583,552 parameters, 47.1 GFLOPs.

### 6both (blocks 1 to 6, mlp+attn)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 9 | pairing by same-image correlation | 31.93 (n = 1) | 33.43 | 54.50 | n/a | 25,668,000 | 0.00% | 69,784,224 (0.806) | 38.3 |  |
| 10 | pairing by weight cosine | 40.07 (n = 1) | 41.19 | 62.55 | n/a | 25,668,000 | 0.00% | 69,784,224 (0.806) | 38.3 |  |
| 11 | random pairing | 34.61 (n = 1) | 35.92 | 56.87 | n/a | 25,668,000 | 0.00% | 69,784,224 (0.806) | 38.3 |  |
| 12 | random structure | 2.74 (n = 1) | 2.87 | 13.73 | n/a | 25,668,000 | 0.00% | 69,784,224 (0.806) | 38.3 |  |

### 9both (blocks 1 to 9, mlp+attn)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 9 | pairing by same-image correlation | 13.36 (n = 1) | 14.24 | 62.80 | n/a | 43,442,640 | 0.00% | 66,325,200 (0.766) | 36.5 |  |
| 10 | pairing by weight cosine | 16.94 (n = 1) | 18.61 | 62.87 | n/a | 43,442,640 | 0.00% | 66,325,200 (0.766) | 36.5 |  |
| 11 | random pairing | 11.07 (n = 1) | 12.23 | 56.40 | n/a | 43,442,640 | 0.00% | 66,325,200 (0.766) | 36.5 |  |
| 12 | random structure | 4.31 (n = 1) | 4.67 | 76.77 | n/a | 43,442,640 | 0.00% | 66,325,200 (0.766) | 36.5 |  |

## Predictions P7.1 and P7.2 (PREREG_NEXT_PHASE.md)

Model `dinov2_vitb14_reg`. P7.1: Mirror-exact beats every baseline (methods 3 to 12) by at least 2 points (mean over fit seeds 0, 1 and 2) with the fit-seed-0 paired bootstrap interval above 0. P7.2: ZipIt-style merging (8) and same-image-correlation pairing (9) trail Mirror-exact by at least 10 points. Each 6-block size is scored separately.

### 6mlp: P7.1 not scored, P7.2 not scored (incomplete: not every method has fit seeds 0, 1 and 2 and a bootstrap)

Mirror-exact: not run.

| # | baseline | seeds | baseline mean | Mirror minus baseline | 95% interval (seed 0) | P7.1 | P7.2 |
|---|---|---|---|---|---|---|---|
| 3 | plain truncated SVD | 0 | not run | | | | |
| 4 | SVD-LLM-style whitened SVD | 0 | not run | | | | |
| 5 | ASVD-style activation scaling | 0 | not run | | | | |
| 6 | SliceGPT-style slicing | 0 | not run | | | | |
| 7 | FLAP-style pruning | 0 | not run | | | | |
| 8 | ZipIt-style merging | 0 | not run | | | | |
| 9 | pairing by same-image correlation | 0 | not run | | | | |
| 10 | pairing by weight cosine | 0 | not run | | | | |
| 11 | random pairing | 0 | not run | | | | |
| 12 | random structure | 0 | not run | | | | |

### 6both: P7.1 not scored, P7.2 not scored (incomplete: not every method has fit seeds 0, 1 and 2 and a bootstrap)

Mirror-exact: not run.

| # | baseline | seeds | baseline mean | Mirror minus baseline | 95% interval (seed 0) | P7.1 | P7.2 |
|---|---|---|---|---|---|---|---|
| 3 | plain truncated SVD | 0 | not run | | | | |
| 4 | SVD-LLM-style whitened SVD | 0 | not run | | | | |
| 5 | ASVD-style activation scaling | 0 | not run | | | | |
| 6 | SliceGPT-style slicing | 0 | not run | | | | |
| 7 | FLAP-style pruning | 0 | not run | | | | |
| 8 | ZipIt-style merging | 0 | not run | | | | |
| 9 | pairing by same-image correlation | 1 | 31.93 | n/a | [n/a, n/a] | FAIL | FAIL |
| 10 | pairing by weight cosine | 1 | 40.07 | n/a | [n/a, n/a] | FAIL |  |
| 11 | random pairing | 1 | 34.61 | n/a | [n/a, n/a] | FAIL |  |
| 12 | random structure | 1 | 2.74 | n/a | [n/a, n/a] | FAIL |  |

## Gate B (best baseline, method-selection set only)

Model `dinov2_vitb14_reg`, size full, fit seed 0: best **None**, second best **None**; INCOMPLETE, missing svd, svdllm, asvd_scale, slicegpt, flap, zipit, pair_actcorr, pair_wcos, pair_random, random_structure. Rule: rank methods 3 to 12 (every non-Mirror method; Mirror-local is an ablation) by method-selection k-NN top-1 at the full-model size, each at its own method-selection-chosen hyperparameters; ties go to the lower MUST-07 method number. Written to `gate_b.json`.

| rank | method | method-selection top-1 | hyperparameters |
|---|---|---|---|

## Notes

- Methods 1 and 9 to 12 use one span operator A* (Mirror-exact machinery); for the pairing controls only the MLP neuron pairing changes, the attention keeps the fitted head maps. Their parameter counts equal Mirror-exact's (gap 0).
- Mirror-local counts one d x d rotation per converted block and uses A_b for the attention output; it is an ablation and is not matched to the budget.
- Mirror-exact's basis change at a span end that is not a model boundary is counted (d x d); params_total_absorbed in the CSV gives the count with it absorbed into the next dense block.
- Compressor grid points are scored on the method-selection queries only; the chosen point is then scored on validation. Gate B uses the method-selection top-1 of each method's chosen point, so methods with larger grids get a small selection advantage on that set.
- Random pairing and random structure use seed random_seed + fit_seed, so their spread over fit seeds includes the random draw.
- Biases, norms and embeddings are counted as dense; analytic GFLOPs exclude biases, norms and activations.

Files: rows_<model>.csv, grid_<model>.csv, bootstrap.csv, preds/, info/, gate_b.json, meta.json.
