# UP-06: Mirror stacked with INT8 quantisation and token merging

## Verdict

Prediction (PREREG_NEXT_PHASE.md, UP-06): the extra drop from each method is the same for both models within 0.5 points. **HOLDS**.

- INT8 weight-only, per output channel: extra drop 0.00 for the original, 0.01 for healed Mirror-exact; difference 0.00 points (95% interval -0.10 to 0.10, n = 50000): **PASS** (bar 0.5; interval inside the bar: yes).
- ToMe: extra drop 0.34 for the original, 0.18 for healed Mirror-exact; difference -0.16 points (95% interval -0.34 to 0.02, n = 50000): **PASS** (bar 0.5; interval inside the bar: yes).

## Rows

Validation k-NN top-1 (bank of 100000 train images, 50000 validation queries, k = 20, T = 0.07, bf16 features); flip consistency = share of validation images whose prediction equals that of the mirrored image; fp32 columns with TF32 off.

| variant | top-1 | flip consistency | disagreements | max unit-feature diff | top-1 fp32 | flip consistency fp32 | disagreements fp32 | tokens after last block |
|---|---|---|---|---|---|---|---|---|
| original | 78.95 | 96.280 | 1860 | 0.154133 | n/a | n/a | n/a | 261 |
| original+int8 | 78.95 | 96.266 | 1867 | 0.154923 | n/a | n/a | n/a | 261 |
| original+tome | 78.62 | 95.036 | 2482 | 0.161985 | n/a | n/a | n/a | 165 |
| mirror_exact | 78.32 | 99.744 | 128 | 0.009984 | 78.37 | 100.000 | 0 | 261 |
| mirror_exact+int8 | 78.31 | 99.104 | 448 | 0.029211 | 78.31 | 99.090 | 455 | 261 |
| mirror_exact+tome | 78.14 | 96.838 | 1581 | 0.188606 | 78.15 | 96.798 | 1601 | 165 |

## Provenance

- Healed run: `/home/ubuntu/Mirror_CPAL/runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0` (method mirror_exact, healing seed 0); model dinov2_vitb14_reg; git e84e20ff35bb1ac4c3e7e3406fba5d633fb3c9f9 (dirty); seed 0; protocol v2 (100k train bank, 50k validation).
- INT8: 48 matrices, 84934656 weights, largest |W - W_hat| / s = 0.5000 (bound 0.5); symmetric, per output channel, [-127, 127], fp32 scale max|W_row|/127, round half to even; weights dequantised in place (simulated weight-only INT8).
- ToMe: r = 8 per block, 5 prefix tokens (CLS and registers) never merged, proportional attention; tokens after each block 253, 245, 237, 229, 221, 213, 205, 197, 189, 181, 173, 165 (from 261).
- Mirror-exact weight constraints after INT8 (largest relative violation; about 1e-8 before): MirrorMLPConstraint 2.03e-02; MirrorAttnConstraint 1.53e-02.

## Caveats

- INT8 is simulated: weights are replaced by their dequantised INT8 values and run in the usual float kernels; the accuracy effect is that of the rounding, no INT8 kernel or speed is measured.
- ToMe is not mirror-equivariant (its alternating A/B partition swaps under a left-right flip of an even-width grid), so Mirror-exact + ToMe need not be exactly invariant; INT8 rounding of a dense, non-diagonal A* basis also breaks the exact weight constraints slightly. Both are measured in the flip-consistency columns.
- Prefix tokens (CLS and registers) are protected from merging; the official ToMe code protects the class token only, and registers are protected here in the same way.
- One healed Mirror-exact model (healing seed 0); other seeds need another invocation.
- The verdict uses the point estimate of the difference in extra drops; the bootstrap interval is reported next to it and is not part of the verdict.
