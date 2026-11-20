# A01: weight-space operator A*_w (dinov2_vitb14_reg, full model)

Zero-shot k-NN was not run (--skip-knn).

Source structure: `/home/Mirror_CPAL/results/MUST-07/structure/dinov2_vitb14_reg_s0.npz` (sha256 13f19ba0f75bf047).

| operator | d+ | d- | pairs (span) | damage | J | distance to A* | cos^2 (unweighted) | rho(A*_w -> A*) | pairs kept | params | zero-shot top-1 | flip (bf16) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| current | 700 | 68 | 7,054 | 2.6857 | 10.6286 | 0.0000 | 1.000 | 1.000 | 1.000 | n/a | n/a | n/a |
| wspace_matched | 700 | 68 | 7,054 | 1.8714 | 12.2572 | 0.2086 | 0.411 | 0.661 | 0.182 | n/a | n/a | n/a |

Damage is the weighted sum of the terms' breaking shares ||X - X_eq||^2 / ||X||^2 (MLP blocks, patch embedding, position grid, class token, registers; mirror/fit_weightspace.py); J = tr(A M) and damage = (number of terms - J) / 2. rho is the energy-weighted share of A*_w's odd subspace inside the current A*'s (mirror/stability.py).

## Alternation

| mode | round | J after re-pairing | J | damage | d- | neurons changed | share changed |
|---|---|---|---|---|---|---|---|
| matched | 0 | 10.62860 | 11.25707 | 2.37146 | 68 | 0 | 0.00% |
| matched | 1 | 11.85927 | 12.06294 | 1.96853 | 68 | 12762 | 34.62% |
| matched | 2 | 12.13393 | 12.19523 | 1.90238 | 68 | 6906 | 18.73% |
| matched | 3 | 12.22579 | 12.25720 | 1.87140 | 68 | 5221 | 14.16% |

Round 0 fits A on the source pairings (its first J column is the current operator's J). The free mode is a diagnostic only (not healed).
