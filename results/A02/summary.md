# A02: stability across fits (dinov2_vitb14_reg)

- **A02-1** (d- within 15% of fit 0): PASS
- **A02-2** (odd subspaces agree (energy weighted, >= 0.8)): PASS, smallest 0.999
- **A02-3** (twins agree (>= 0.70 of strong pairs, blocks 1 to 6)): PASS, smallest 0.953
- **A02-4** (head maps agree (>= 0.90 of assignments)): PASS, smallest 1.000
- A02-5 and A02-6 are decided from the healed runs.

## Fits

| fit | group | d+ | d- | pairs | strong pairs (blocks 1-6) | twin fraction (1-6) | head pairs | zero-shot top-1 |
|---|---|---|---|---|---|---|---|---|
| fit0 | imagenet | 700 | 68 | 7,054 | 5232 | 0.568 | 7 | 53.88 |
| fit1 | imagenet | 699 | 69 | 7,065 | 5246 | 0.569 | 7 | 53.42 |
| fit2 | imagenet | 700 | 68 | 7,054 | 5234 | 0.568 | 7 | 54.13 |
| fit3 | imagenet | 700 | 68 | 7,055 | 5235 | 0.568 | 7 | 53.98 |
| coco | coco | 695 | 73 | 7,036 | 5173 | 0.561 | 7 | 55.73 |

## Pairs of fits

| a | b | distance | rho a->b | rho b->a | weighted | cos^2 (unweighted) | twins a->b | twins b->a | twins mean | Jaccard | heads same | B rel. dist. | C rel. dist. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fit0 | fit1 | 0.0049 | 0.999 | 0.999 | 0.999 | 0.993 | 0.958 | 0.955 | 0.956 | 0.914 | 1.000 | 0.020 | 0.019 |
| fit0 | fit2 | 0.0028 | 0.999 | 0.999 | 0.999 | 0.992 | 0.954 | 0.953 | 0.953 | 0.908 | 1.000 | 0.015 | 0.018 |
| fit0 | fit3 | 0.0040 | 0.999 | 0.999 | 0.999 | 0.989 | 0.970 | 0.969 | 0.970 | 0.939 | 1.000 | 0.016 | 0.017 |
| fit0 | coco | 0.0246 | 0.992 | 0.990 | 0.991 | 0.967 | 0.889 | 0.893 | 0.891 | 0.792 | 1.000 | 0.062 | 0.067 |
| fit1 | fit2 | 0.0049 | 0.999 | 0.999 | 0.999 | 0.993 | 0.954 | 0.956 | 0.955 | 0.912 | 1.000 | 0.020 | 0.019 |
| fit1 | fit3 | 0.0050 | 0.999 | 0.999 | 0.999 | 0.993 | 0.960 | 0.962 | 0.961 | 0.923 | 1.000 | 0.017 | 0.017 |
| fit1 | coco | 0.0232 | 0.992 | 0.990 | 0.991 | 0.964 | 0.885 | 0.890 | 0.888 | 0.787 | 1.000 | 0.058 | 0.066 |
| fit2 | fit3 | 0.0026 | 0.999 | 0.999 | 0.999 | 0.993 | 0.965 | 0.965 | 0.965 | 0.930 | 1.000 | 0.015 | 0.016 |
| fit2 | coco | 0.0241 | 0.992 | 0.990 | 0.991 | 0.969 | 0.895 | 0.897 | 0.896 | 0.800 | 1.000 | 0.057 | 0.063 |
| fit3 | coco | 0.0241 | 0.992 | 0.990 | 0.991 | 0.969 | 0.888 | 0.892 | 0.890 | 0.792 | 1.000 | 0.056 | 0.064 |

Definitions: mirror/stability.py (PREREG_FINAL_PHASE.md Section 2.3). Distance is ||A_a - A_b||_F^2 / (2 d) for the full-model operator A*_L; rho is the energy-weighted share of one fit's odd subspace inside the other's; a strong pair has a symmetrised twin correlation of at least 0.5.
