# MUST-07: zero-shot comparison at matched parameters

Written by `scripts/must07_compressors.py` on 2026-11-15 16:28:19, commit `805c231c9c` (uncommitted changes). Compressors: **real**; budget accounting: **real**.

Protocol: k-NN top-1 with the 100k-image train bank (100 per class) and the 50k validation images as queries (CLS after the final norm, cosine, k = 20, T = 0.07, bf16 features). Method selection: the 10k method-selection train images as queries against the same bank; every hyperparameter and gate B use only these. Flip consistency: share of validation images whose prediction equals that of the mirrored image, in bf16 (column flip) and, for full-model rows, in fp32 (column flip fp32; bf16 rounding breaks exact ties, so exact invariance is judged on the fp32 value). Parameters: the converted matrices with basis changes; budget = Mirror-exact's count for the same span and components; gap = (method - budget) / budget. GFLOPs: analytic, per image, 2 x multiply-adds. Values are mean +- standard deviation over fit seeds (n given).

## dinov2_vitb14_reg

Original: top-1 78.94, flip 96.32 (bf16), flip fp32 96.30, method-selection top-1 80.21, 86,583,552 parameters, 47.1 GFLOPs.

### 9both (blocks 1 to 9, mlp+attn)

| # | method | val top-1 | msel top-1 (seed 0) | flip | flip fp32 | params (span) | gap | model params | GFLOPs | chosen hyperparameters |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Mirror-exact | 73.35 (n = 1) | 74.94 | 94.03 | n/a | 43,442,640 | 0.00% | 66,325,200 (0.766) | 36.5 |  |
| 2 | Mirror-local (ablation) | 62.26 (n = 1) | 63.66 | 76.81 | n/a | 47,273,228 | 8.82% | 70,155,788 (0.810) | 38.5 |  |
| 3 | plain truncated SVD | 5.67 (n = 1) | 6.16 | 27.24 | n/a | 43,442,688 | 0.00% | 66,325,248 (0.766) | 36.5 |  |
| 4 | SVD-LLM-style whitened SVD | 77.35 (n = 1) | 78.77 | 94.26 | n/a | 43,442,688 | 0.00% | 66,325,248 (0.766) | 36.5 | {"damp": 1e-06} |

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
| 9 | pairing by same-image correlation | 0 | not run | | | | |
| 10 | pairing by weight cosine | 0 | not run | | | | |
| 11 | random pairing | 0 | not run | | | | |
| 12 | random structure | 0 | not run | | | | |

## Gate B (best baseline, method-selection set only)

Model `dinov2_vitb14_reg`, size full, fit seed 0: best **None**, second best **None**; INCOMPLETE, missing svd, svdllm, asvd_scale, slicegpt, flap, zipit, pair_actcorr, pair_wcos, pair_random, random_structure. Rule: rank methods 3 to 12 (every non-Mirror method; Mirror-local is an ablation) by method-selection k-NN top-1 at the full-model size, each at its own method-selection-chosen hyperparameters; ties go to the lower MUST-07 method number. Written to `gate_b.json`.

| rank | method | method-selection top-1 | hyperparameters |
|---|---|---|---|

## Paired bootstrap, Mirror-exact minus each method (validation, fit seed 0, 10,000 resamples)

| model | size | method | n | difference | 95% interval | excludes 0 |
|---|---|---|---|---|---|---|
| dinov2_vitb14_reg | 9both | Mirror-local (ablation) | 50000 | 11.09 | [10.72, 11.45] | yes |
| dinov2_vitb14_reg | 9both | plain truncated SVD | 50000 | 67.68 | [67.25, 68.11] | yes |
| dinov2_vitb14_reg | 9both | SVD-LLM-style whitened SVD | 50000 | -4.00 | [-4.28, -3.72] | yes |

## Notes

- Methods 1 and 9 to 12 use one span operator A* (Mirror-exact machinery); for the pairing controls only the MLP neuron pairing changes, the attention keeps the fitted head maps. Their parameter counts equal Mirror-exact's (gap 0).
- Mirror-local counts one d x d rotation per converted block and uses A_b for the attention output; it is an ablation and is not matched to the budget.
- Mirror-exact's basis change at a span end that is not a model boundary is counted (d x d); params_total_absorbed in the CSV gives the count with it absorbed into the next dense block.
- Compressor grid points are scored on the method-selection queries only; the chosen point is then scored on validation. Gate B uses the method-selection top-1 of each method's chosen point, so methods with larger grids get a small selection advantage on that set.
- Random pairing and random structure use seed random_seed + fit_seed, so their spread over fit seeds includes the random draw.
- Biases, norms and embeddings are counted as dense; analytic GFLOPs exclude biases, norms and activations.

Files: rows_<model>.csv, grid_<model>.csv, bootstrap.csv, preds/, info/, gate_b.json, meta.json.
