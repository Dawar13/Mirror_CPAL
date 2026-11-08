# Track D: data-free readout pilot (P5, gate G4)

## Verdict

- P5 part 1, layer-0 even subspace: mean cosine of principal angles 0.900 (bar 0.8): **PASS**. Chance level (random involutions with the same d+): 0.652. d+ 392 from the weights against 400 from data; overlap 0.829; tr(A A^w)/d 0.664.
- P5 part 2, block-1 pairing from the weights given the data-driven A_0: recovers 0.466 of the data-driven pairs (bar 0.7): **FAIL**.
- Fully data-free chain (pairing given A_0^w): recovers 0.441: **FAIL**.
- P5 as D2 defines it (parts 1 and 2): **FAIL**; strict reading (part 1 and the fully data-free chain): **FAIL**. Gate G4: not passed: claim C6 is dropped unless Dawar decides otherwise.

PREREG_E2.md does not say whether the block-1 pairing may use the data-driven A_0 ("given A" in PAPER_EXPERIMENTS.md D2) or must use A_0^w; both are shown so the reading can be chosen without looking at which one passes.

## Provenance

- Model: `dinov2_vitb14_reg`; operators: `runs/x2/perlayer_reg_s0/results/e1_operators.npz` (sha256 9ca69899622e2d2f), operator run git n/a, fit seed n/a, protocol n/a.
- This script: git 6fd06a579e259af724f74a4d85afe2e0067eb3de (dirty), seed 0, date 2026-11-08 10:46:43, CPU float64.
- PREREG_E2.md commit: 9804f0670ea630e03b1bde285943f7d735b90ccd; file sha256 8c30d76dc75227e7.
- Layer-0 map affine check (weights against the adapter's own layer-0 tokens): max relative difference 1.62e-16. Stored operators cleaned to exact involutions (largest eigenvalue deviation 4.59e-08).

## D1: layer-0 operator from the weights

C_w = sigma^2 HW J W F W^T J + sum over tokens of J k_t kt_t^T J (isotropic N(0, sigma^2 I) prior on the normalised pixels of a patch; k = bias, position, CLS and registers), channel-centred as in the data path. rho is the resulting content share of the token energy. The primary row (sigma^2 = 1) is the only one used for P5; the others are sensitivity checks. Agreement is with the data-driven A_0.

| variant | rho | d+ (w) | d+ (data) | mean cos even | mean cos odd | overlap even | tr(A A^w)/d |
|---|---|---|---|---|---|---|---|
| primary: isotropic prior, sigma^2 = 1 | 0.839 | 392 | 400 | 0.900 | 0.893 | 0.829 | 0.664 |
| sweep | 0.000 | 640 | 400 | 0.961 | 0.877 | 0.581 | 0.229 |
| sweep | 0.100 | 393 | 400 | 0.970 | 0.968 | 0.932 | 0.876 |
| sweep | 0.250 | 393 | 400 | 0.944 | 0.940 | 0.893 | 0.795 |
| sweep: equal energy | 0.500 | 392 | 400 | 0.922 | 0.917 | 0.859 | 0.726 |
| sweep | 0.750 | 392 | 400 | 0.906 | 0.900 | 0.836 | 0.680 |
| sweep | 0.900 | 392 | 400 | 0.896 | 0.889 | 0.823 | 0.652 |
| sweep | 1.000 | 474 | 400 | 0.920 | 0.891 | 0.736 | 0.542 |
| position term only (no bias, no prefix), sigma^2 = 1 | 0.998 | 388 | 400 | 0.901 | 0.895 | 0.820 | 0.656 |
| null: random involution, same d+ (mean of 3) | n/a | 400 | 400 | 0.652 | 0.622 | 0.521 | 0.002 |

Mean cosine is over min(d+ (w), d+ (data)) angles; overlap = ||U^T U^w||_F^2 / max(d+), which also penalises a dimension mismatch.

## D2: twin pairing from the weights

S[i, j] = cos(w_i, A w_j) over folded fc1 rows, then the E1b involution matching (top-k 10). Recovery = data-driven pairs found / data-driven pairs. Chance = expected recovery of a random pairing with the same number of pairs.

| block | A used | pairs (data) | pairs (w) | recovery | precision | self-twins kept | chance | mean S, data pairs | mean S, data self |
|---|---|---|---|---|---|---|---|---|---|
| 1 | data A_b | 1168 | 959 | 0.466 | 0.567 | 0.844 | 0.000 | 0.481 | 0.279 |
| 2 | data A_b | 1116 | 1100 | 0.592 | 0.601 | 0.817 | 0.000 | 0.487 | 0.528 |
| 3 | data A_b | 1013 | 1001 | 0.651 | 0.658 | 0.894 | 0.000 | 0.511 | 0.625 |
| 4 | data A_b | 963 | 929 | 0.622 | 0.645 | 0.901 | 0.000 | 0.447 | 0.665 |
| 5 | data A_b | 878 | 799 | 0.628 | 0.690 | 0.924 | 0.000 | 0.462 | 0.738 |
| 6 | data A_b | 672 | 612 | 0.670 | 0.735 | 0.974 | 0.000 | 0.500 | 0.825 |
| 7 | data A_b | 554 | 522 | 0.764 | 0.810 | 0.988 | 0.000 | 0.584 | 0.903 |
| 8 | data A_b | 406 | 398 | 0.815 | 0.832 | 0.996 | 0.000 | 0.653 | 0.944 |
| 9 | data A_b | 155 | 152 | 0.852 | 0.868 | 0.999 | 0.000 | 0.682 | 0.957 |
| 10 | data A_b | 93 | 92 | 0.892 | 0.902 | 1.000 | 0.000 | 0.681 | 0.971 |
| 11 | data A_b | 19 | 20 | 0.895 | 0.850 | 0.999 | 0.000 | 0.719 | 0.992 |
| 12 | data A_b | 17 | 15 | 0.824 | 0.933 | 1.000 | 0.000 | 0.507 | 0.981 |
| 1 | data-free A_0^w | 1168 | 938 | 0.441 | 0.549 | 0.845 | 0.000 | 0.473 | 0.254 |
| 1 | null: random involution | 1168 | 1335 | 0.000 | 0.000 | 0.125 | 0.000 | 0.007 | 0.039 |

## D3 (stretch): propagation to later layers

Moment model per token: attention replaced by its value-output map (attention mixing taken as the identity), MLP with independent Gaussian pre-activations and twins swapping under the mirror, no correlation between the MLP output and the stream it reads. The layer-0 state is the D1 primary one (sigma^2 = 1). Agreement with the data-driven A_l (l = output of block l). The propagation is only useful where it beats copying A_0^w forward. The scale-check rows show how much the answer depends on the absolute stream energy set at layer 0. Exploratory: no pre-registered bar.

| layer | operator | mean cos even | mean cos odd | tr(A A^w)/d | d+ (w) | d+ (data) |
|---|---|---|---|---|---|---|
| 1 | propagated, weight-only pairing | 0.930 | 0.888 | 0.624 | 445 | 489 |
| 1 | propagated, data-driven pairing | 0.936 | 0.896 | 0.653 | 450 | 489 |
| 1 | scale check: data-driven pairing, equal-energy layer 0 | 0.925 | 0.877 | 0.633 | 457 | 489 |
| 1 | baseline: A_0^w copied forward | 0.879 | 0.830 | 0.364 | 392 | 489 |
| 1 | context: data-driven A_(l-1) | 0.878 | 0.825 | 0.377 | 400 | 489 |
| 2 | propagated, weight-only pairing | 0.938 | 0.891 | 0.668 | 470 | 502 |
| 2 | propagated, data-driven pairing | 0.941 | 0.893 | 0.704 | 482 | 502 |
| 2 | scale check: data-driven pairing, equal-energy layer 0 | 0.919 | 0.850 | 0.651 | 494 | 502 |
| 2 | baseline: A_0^w copied forward | 0.879 | 0.822 | 0.326 | 392 | 502 |
| 2 | context: data-driven A_(l-1) | 0.960 | 0.926 | 0.797 | 489 | 502 |
| 3 | propagated, weight-only pairing | 0.917 | 0.814 | 0.562 | 506 | 542 |
| 3 | propagated, data-driven pairing | 0.921 | 0.823 | 0.584 | 509 | 542 |
| 3 | scale check: data-driven pairing, equal-energy layer 0 | 0.875 | 0.702 | 0.492 | 540 | 542 |
| 3 | baseline: A_0^w copied forward | 0.862 | 0.761 | 0.163 | 392 | 542 |
| 3 | context: data-driven A_(l-1) | 0.953 | 0.895 | 0.691 | 502 | 542 |

Pairing in later blocks from the propagated data-free operator:

| block | recovery | precision | self-twins kept | pairs (w) | pairs (data) |
|---|---|---|---|---|---|
| 2 | 0.481 | 0.548 | 0.818 | 980 | 1116 |
| 3 | 0.552 | 0.638 | 0.904 | 876 | 1013 |

## Caveats

- The isotropic prior on patch content is the only data-free choice made; natural patches are far from isotropic, so D1 tests whether the embedding's own mirror structure dominates.
- The D1 weighting (sigma^2 = 1.0: unit-variance isotropic prior on normalised pixels) was fixed in code before any run on real weights; the rho sweep (with rho = 0.5, equal energy) and the data-calibrated rho are sensitivity checks only. Real patches are correlated, so the prior understates the content energy that passes through low-frequency kernels; the calibrated row shows by how much.
- The mean-cosine measure has a high chance level when d+ is near d/2 (see the null row); the overlap and trace columns are harder to satisfy by chance.
- Directions that neither the data nor the weights reach (zero eigenvalues of C) are set to +1 in both operators, which adds a few shared even directions.
- Block 1 means model.blocks[0]; its fc1 is compared under A_0 (the block input), as in E1b.
