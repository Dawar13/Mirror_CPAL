# MUST-07: zero-shot comparison at matched parameters

Written by `scripts/must07_compressors.py` on 2026-11-15 14:21:44, commit `805c231c9c` (uncommitted changes). Compressors: **real**; budget accounting: **real**.

Protocol: k-NN top-1 with the 100k-image train bank (100 per class) and the 50k validation images as queries (CLS after the final norm, cosine, k = 20, T = 0.07, bf16 features). Method selection: the 10k method-selection train images as queries against the same bank; every hyperparameter and gate B use only these. Flip consistency: share of validation images whose prediction equals that of the mirrored image, in bf16 (column flip) and, for full-model rows, in fp32 (column flip fp32; bf16 rounding breaks exact ties, so exact invariance is judged on the fp32 value). Parameters: the converted matrices with basis changes; budget = Mirror-exact's count for the same span and components; gap = (method - budget) / budget. GFLOPs: analytic, per image, 2 x multiply-adds. Values are mean +- standard deviation over fit seeds (n given).

No rows yet.
