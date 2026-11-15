# UP-07: pairing from weights alone, second attempt

## Verdict

- Primary (weights only; fc1 rows and fc2 columns scored jointly; A_0^w for fc1 and the propagated A_1^w for fc2): block 1 recovers 0.601 of the data-driven pairs (702 of 1168; bar 0.7): **FAIL**.
- First attempt (fc1 rows only, runs/d): 0.466 given the data-driven A_0 (the 46.6% of P5 part 2) and 0.441 with A_0^w.
- The same joint rule given the data-driven operators (A_0 for fc1, A_1 for fc2; the reading of the first attempt's 46.6%): 0.636 (743 of 1168): **FAIL** against the same bar.
- Primary rule pooled over blocks 1 to 12: 0.382 of the data-driven pairs (2694 of 7054); mean over blocks 0.216. Reported, not scored.
- Reproduction of the first attempt in block 1 (same code path, fc1 rows only): data-driven A_0 0.466 (stored 0.466), A_0^w 0.441 (stored 0.441); layer-0 operator agreement (P5 part 1) 0.900 (first attempt 0.900).
- Against every stored fc1-only row of the first attempt (15 rows, runs/d/d_datafree.json; same operators file: yes), the largest difference in recovery is 0.0000.

## Recovery by block

Recovery = data-driven pairs found / data-driven pairs (mirror.datafree.pairing_agreement, the first attempt's measure). Chance = expected recovery of a random pairing with the primary rule's number of pairs.

| block | pairs (data) | weights_joint | weights_fc1 | weights_fc2 | shared_joint | data_joint | data_fc1 | pairs found (primary) | chance |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1168 | 0.601 | 0.441 | 0.603 | 0.531 | 0.636 | 0.466 | 1059 | 0.00022 |
| 2 | 1116 | 0.626 | 0.481 | 0.545 | 0.427 | 0.651 | 0.592 | 1053 | 0.00022 |
| 3 | 1013 | 0.631 | 0.552 | 0.492 | 0.367 | 0.674 | 0.651 | 937 | 0.00020 |
| 4 | 963 | 0.466 | 0.493 | 0.317 | 0.135 | 0.661 | 0.622 | 656 | 0.00014 |
| 5 | 878 | 0.192 | 0.310 | 0.057 | 0.059 | 0.666 | 0.628 | 230 | 0.00005 |
| 6 | 672 | 0.024 | 0.051 | 0.013 | 0.022 | 0.690 | 0.670 | 23 | 0.00000 |
| 7 | 554 | 0.022 | 0.036 | 0.007 | 0.032 | 0.767 | 0.764 | 14 | 0.00000 |
| 8 | 406 | 0.012 | 0.022 | 0.002 | 0.027 | 0.793 | 0.815 | 7 | 0.00000 |
| 9 | 155 | 0.019 | 0.032 | 0.000 | 0.065 | 0.806 | 0.852 | 6 | 0.00000 |
| 10 | 93 | 0.000 | 0.011 | 0.000 | 0.097 | 0.871 | 0.892 | 0 | 0.00000 |
| 11 | 19 | 0.000 | 0.105 | 0.000 | 0.105 | 1.000 | 0.895 | 1 | 0.00000 |
| 12 | 17 | 0.000 | 0.000 | 0.000 | 0.118 | 0.882 | 0.824 | 4 | 0.00000 |

Variants: weights_joint: weights-only operators, fc1 and fc2 jointly (primary); weights_fc1: weights-only operators, fc1 rows only (first attempt's rule); weights_fc2: weights-only operators, fc2 columns only; shared_joint: A_0^w on both sides in every block, jointly; data_joint: data-driven operators, fc1 and fc2 jointly; data_fc1: data-driven A_in, fc1 rows only (first attempt's D2).

Pooled over the requested blocks:

| variant | pairs found | data-driven pairs | pooled recovery | mean over blocks |
|---|---|---|---|---|
| weights_joint | 2694 | 7054 | 0.382 | 0.216 |
| weights_fc1 | 2429 | 7054 | 0.344 | 0.211 |
| weights_fc2 | 2179 | 7054 | 0.309 | 0.170 |
| shared_joint | 1718 | 7054 | 0.244 | 0.165 |
| data_joint | 4825 | 7054 | 0.684 | 0.758 |
| data_fc1 | 4464 | 7054 | 0.633 | 0.722 |

## Weight-only operators against the data-driven ones

Layer l = the stream after block l (layer 0 = block 1's input). Mean cosine of the principal angles between the +1 eigenspaces and tr(A A^w)/d (mirror.datafree.compare_operators); the copied row uses A_0^w at every layer.

| layer | d+ (data) | d+ (weights) | mean cos even | mean cos odd | tr(A A^w)/d | tr, A_0^w copied |
|---|---|---|---|---|---|---|
| 0 | 400 | 392 | 0.900 | 0.893 | 0.664 | 0.664 |
| 1 | 489 | 445 | 0.930 | 0.888 | 0.624 | 0.364 |
| 2 | 502 | 470 | 0.938 | 0.891 | 0.668 | 0.326 |
| 3 | 542 | 506 | 0.917 | 0.814 | 0.562 | 0.163 |
| 4 | 565 | 545 | 0.891 | 0.706 | 0.502 | 0.090 |
| 5 | 605 | 581 | 0.885 | 0.590 | 0.452 | 0.056 |
| 6 | 640 | 615 | 0.899 | 0.515 | 0.494 | 0.046 |
| 7 | 673 | 644 | 0.921 | 0.465 | 0.570 | 0.040 |
| 8 | 723 | 688 | 0.961 | 0.399 | 0.726 | 0.032 |
| 9 | 740 | 730 | 0.973 | 0.299 | 0.850 | 0.029 |
| 10 | 757 | 755 | 0.988 | 0.174 | 0.942 | 0.023 |
| 11 | 758 | 759 | 0.990 | 0.185 | 0.955 | 0.023 |
| 12 | 760 | 758 | 0.992 | 0.210 | 0.957 | 0.023 |

## Provenance

- Model `dinov2_vitb14_reg`; data-driven pairing `/home/ubuntu/Mirror_CPAL/runs/x2/perlayer_reg_s0/results/e1_operators.npz` (sha256 9ca69899622e2d2f, fit seed 0); git a3a7c0ce51ad336edab31f7fc3f177600cca23a4 (dirty); seed 0; alpha 0.5; top-k 10; chain pairing fc1; 4 matching workers; 122.4 min on the CPU.

## Caveats

- alpha = 0.5 (equal weight of the fc1 and fc2 cosines) was fixed in code before any run on real weights; no other alpha is scored.
- The fc2 operator A_(b)^w comes from the first attempt's moment model (attention replaced by its value-output map, independent Gaussian pre-activations), propagated with the weight-only fc1 pairing; its error grows with depth (see the operator table).
- The data-driven pairing is itself an estimate from one fit set (fit seed 0); it is the reference here, not the truth, so 100% is not the ceiling of a perfect weight-only rule.
- Block 1 is scored, as in the first attempt; the other blocks and the pooled recovery are reported, not scored.
- PREREG_NEXT_PHASE.md does not say whether 'weights alone' may use the data-driven operators; the primary row uses none, the data_joint row is shown for comparison with the first attempt's 46.6%.
