# A01: weight-space operator A*_w (dinov2_vitb14_reg, full model)

Zero-shot k-NN was not run (--skip-knn).

Source structure: `/home/Mirror_CPAL/results/MUST-07/structure/dinov2_vitb14_reg_s0.npz` (sha256 13f19ba0f75bf047).

| operator | d+ | d- | pairs (span) | damage | J | distance to A* | cos^2 (unweighted) | rho(A*_w -> A*) | pairs kept | params | zero-shot top-1 | flip (bf16) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| current | 700 | 68 | 7,054 | 2.6857 | 10.6286 | 0.0000 | 1.000 | 1.000 | 1.000 | n/a | n/a | n/a |
| wspace_free | 765 | 3 | 24 | 0.4473 | 15.1054 | 0.1711 | 0.886 | 0.859 | 0.002 | n/a | n/a | n/a |

Damage is the weighted sum of the terms' breaking shares ||X - X_eq||^2 / ||X||^2 (MLP blocks, patch embedding, position grid, class token, registers; mirror/fit_weightspace.py); J = tr(A M) and damage = (number of terms - J) / 2. rho is the energy-weighted share of A*_w's odd subspace inside the current A*'s (mirror/stability.py).

## Alternation

| mode | round | J after re-pairing | J | damage | d- | neurons changed | share changed |
|---|---|---|---|---|---|---|---|
| free | 0 | 10.62860 | 11.25759 | 2.37120 | 71 | 0 | 0.00% |
| free | 1 | 13.03313 | 14.54657 | 0.72671 | 5 | 12868 | 34.91% |
| free | 2 | 15.08037 | 15.10487 | 0.44756 | 3 | 1582 | 4.29% |
| free | 3 | 15.10545 | 15.10545 | 0.44728 | 3 | 6 | 0.02% |

Round 0 fits A on the source pairings (its first J column is the current operator's J). The free mode is a diagnostic only (not healed).
