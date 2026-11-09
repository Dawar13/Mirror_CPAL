# M2: efficient mirror-converted blocks and latency

Model `dinov2_vitb14_reg` (12 blocks, d 768, h 3072, 12 heads, 261 tokens per image at 224 px). Structure: `/workspace/m2run/runs/m1/m1_structure.npz`, operators `/workspace/m2run/runs/x2/perlayer_reg_s0/results/e1_operators.npz`. Git `6f2c9181a8`, seed 0. Device: NVIDIA GeForce RTX 5090, torch 2.11.0+cu128. Written 2026-11-09 16:41:53.

**Measured speed-ups are expected to be smaller than the FLOP savings.** The converted blocks replace a few large matrix multiplications by more, smaller ones (even and odd parts), plus butterfly additions, gathers and concatenations that move memory but do no multiply-adds; token mixing (attention over tokens) is unchanged; at batch 1 the GPU is limited by kernel launches rather than arithmetic; and V-local pays a d x d basis rotation per converted block. The FLOP column below is analytic; the latency columns are measured.

## Equivalence checks

Each efficient model against the dense model it was built from, on random normal images (seed 0). float64: both built in float64 (criterion: max relative error of the CLS feature under 1e-6). Dropped: relative size (Frobenius) of the non-equivariant part of the folded weights that the even/odd form leaves out; it is zero in exact arithmetic. LayerNorm folding centres every write, which keeps the weights equivariant only if each operator fixes the all-ones direction exactly; largest |A 1 - 1| / sqrt(d) of the operators used: 3.3e-07 (per-layer), 1.2e-11 (span operators) (fitted operators stored in float32; this is what the dropped part measures). fp32 (TF32 off) and bf16: max abs difference of the L2-normalised CLS feature (the k-NN feature), under 1e-5 and 1e-3. Noise floor: the dense model in bf16 against itself in fp32 (same feature). Raw: the CLS feature before normalisation (absolute; largest entry in brackets).

| variant | k | float64 rel. | dropped (float64) | fp32 unit | fp32 raw | bf16 unit | bf16 noise floor | bf16 / noise | pass (64 / 32 / bf16) |
|---|---|---|---|---|---|---|---|---|---|
| dense (folded original, same implementation) | 0 | 1.1e-14 | 0.0e+00 | 3.7e-07 | 9.2e-06 (6.1) | 3.3e-03 | 4.2e-03 | 0.80 | yes / yes / NO |
| Mirror V-local, MLP + attention | 6 | 5.0e-10 | 6.7e-11 | 8.2e-07 | 2.0e-05 (4.4) | 9.8e-03 | 2.9e-03 | 3.40 | yes / yes / NO |
| Mirror V-exact, MLP + attention | 6 | 3.2e-14 | 1.2e-15 | 1.1e-06 | 2.7e-05 (5.7) | 5.9e-03 | 2.6e-03 | 2.27 | yes / yes / NO |
| Mirror V-local, MLP only | 6 | 4.4e-10 | 6.7e-11 | 8.7e-07 | 2.0e-05 (5.3) | 7.3e-03 | 4.8e-03 | 1.51 | yes / yes / NO |
| SVD at matched parameters | 6 | 1.2e-14 | 0.0e+00 | 9.9e-07 | 2.6e-05 (5.0) | 7.5e-03 | 9.3e-03 | 0.81 | yes / yes / NO |
| MLP width pruning at matched parameters | 6 | 4.8e-15 | 0.0e+00 | 2.9e-07 | 7.3e-06 (4.3) | 2.4e-03 | 3.9e-03 | 0.61 | yes / yes / NO |
| Mirror V-local, MLP + attention | 9 | 6.8e-10 | 8.1e-11 | 6.8e-07 | 1.7e-05 (4.3) | 6.4e-03 | 2.8e-03 | 2.28 | yes / yes / NO |
| Mirror V-exact, MLP + attention | 9 | 3.2e-14 | 1.2e-15 | 8.6e-07 | 2.2e-05 (4.3) | 6.3e-03 | 2.5e-03 | 2.55 | yes / yes / NO |
| Mirror V-local, MLP only | 9 | 4.4e-10 | 8.1e-11 | 8.6e-07 | 2.0e-05 (5.2) | 7.8e-03 | 3.3e-03 | 2.36 | yes / yes / NO |
| SVD at matched parameters | 9 | 7.0e-15 | 0.0e+00 | 3.1e-07 | 8.3e-06 (3.4) | 5.7e-03 | 1.2e-02 | 0.47 | yes / yes / NO |
| MLP width pruning at matched parameters | 9 | 3.7e-15 | 0.0e+00 | 2.9e-07 | 7.3e-06 (3.1) | 2.6e-03 | 3.3e-03 | 0.79 | yes / yes / NO |
| Mirror V-local, MLP + attention | 12 | 1.1e-09 | 8.1e-11 | 7.9e-07 | 2.0e-05 (4.3) | 1.1e-02 | 5.5e-03 | 1.96 | yes / yes / NO |
| Mirror V-exact, MLP + attention | 12 | 2.6e-14 | 1.1e-15 | 8.7e-07 | 2.3e-05 (4.3) | 1.0e-02 | 5.3e-03 | 1.90 | yes / yes / NO |
| Mirror V-local, MLP only | 12 | 1.1e-09 | 8.1e-11 | 7.5e-07 | 1.7e-05 (5.2) | 6.4e-03 | 7.1e-03 | 0.91 | yes / yes / NO |
| SVD at matched parameters | 12 | 3.3e-14 | 0.0e+00 | 9.4e-07 | 2.0e-05 (3.2) | 3.9e-03 | 4.6e-03 | 0.84 | yes / yes / NO |
| MLP width pruning at matched parameters | 12 | 3.0e-15 | 0.0e+00 | 1.8e-07 | 4.4e-06 (3.3) | 1.9e-03 | 3.5e-03 | 0.56 | yes / yes / NO |

Outcome: float64: 16 of 16 pass; fp32: 16 of 16 pass; bf16: 0 of 16 pass. In bf16 the efficient-versus-dense difference is 0.47 to 3.40 times the dense model's own bf16 rounding (noise floor 2.5e-03 to 1.2e-02). The bf16 bar of 1e-3 is below that rounding level, so a bf16 failure here reflects bf16 arithmetic, not the implementation; the float64 and fp32 rows are the equivalence checks.

## Parameters and analytic cost

Params: whole model, M1 convention (original count minus the dense matrices of the converted blocks plus the matrices the efficient model stores; biases and embeddings as in the original). Kept: stored matrix parameters of the converted blocks over dense. M1 formula: the count m1_knn.csv uses for this variant. Rotations: stored rotation matrices (d x d each, not in Params). GFLOPs per image: analytic, rotations included; executed adds the alignment padding.

| k | variant | params | kept | M1 formula | d x d rotations per token | GFLOPs | saving vs dense | GFLOPs executed |
|---|---|---|---|---|---|---|---|---|
| 0 | original (hub model as released) | 86,583,552 | 1.000 | 86,583,552 | 0 | 47.08 | 0.0% | 47.08 |
| 0 | dense (folded original, same implementation) | 86,583,552 | 1.000 | 86,583,552 | 0 | 47.08 | 0.0% | 47.08 |
| 6 | Mirror V-local, MLP + attention | 70,472,166 | 0.621 | 69,095,592 | 6 | 40.52 | 13.9% | 40.62 |
| 6 | Mirror V-exact, MLP + attention | 69,194,400 | 0.591 | 69,194,400 | 0 | 38.00 | 19.3% | 38.04 |
| 6 | Mirror V-local, MLP only | 74,734,630 | 0.721 | 74,734,630 | 6 | 42.74 | 9.2% | 42.82 |
| 6 | SVD at matched parameters | 69,061,632 | 0.587 | 69,061,632 | 0 | 37.93 | 19.4% | 38.07 |
| 6 | MLP width pruning at matched parameters | 69,091,584 | 0.588 | n/a | 0 | 37.95 | 19.4% | 37.96 |
| 9 | Mirror V-local, MLP + attention | 66,678,872 | 0.688 | 64,979,688 | 9 | 39.46 | 16.2% | 39.65 |
| 9 | Mirror V-exact, MLP + attention | 65,735,376 | 0.673 | 65,735,376 | 0 | 36.20 | 23.1% | 36.44 |
| 9 | Mirror V-local, MLP only | 72,065,992 | 0.772 | 72,065,992 | 9 | 42.27 | 10.2% | 42.41 |
| 9 | SVD at matched parameters | 64,932,864 | 0.660 | 64,932,864 | 0 | 35.78 | 24.0% | 35.98 |
| 9 | MLP width pruning at matched parameters | 64,972,032 | 0.661 | n/a | 0 | 35.80 | 24.0% | 35.82 |
| 12 | Mirror V-local, MLP + attention | 65,879,016 | 0.756 | 64,090,370 | 12 | 39.97 | 15.1% | 40.21 |
| 12 | Mirror V-exact, MLP + attention | 66,203,360 | 0.760 | 66,203,360 | 0 | 36.44 | 22.6% | 36.78 |
| 12 | Mirror V-local, MLP only | 71,636,896 | 0.824 | 71,636,896 | 12 | 42.97 | 8.7% | 43.15 |
| 12 | SVD at matched parameters | 64,024,320 | 0.734 | 64,024,320 | 0 | 35.30 | 25.0% | 35.59 |
| 12 | MLP width pruning at matched parameters | 64,079,616 | 0.735 | n/a | 0 | 35.33 | 25.0% | 35.37 |

## Latency, compile (bf16, median of 100 runs after 20 warm-up runs)

Speed-up: dense latency over the variant's latency (dense = folded original in the same implementation); FLOP speed-up: dense analytic multiply-adds over the variant's. Original: the hub model as released.

| k | variant | ms (batch 1) | ms (batch 64) | ms (batch 256) | speed-up (b1) | speed-up (b64) | speed-up (b256) | img/s (b256) | FLOP speed-up | mode used |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | original (hub model as released) | 1.12 | 18.98 | 75.84 | 0.93 | 1.02 | 1.01 | 3375 | 1.00 | compile |
| 0 | dense (folded original, same implementation) | 1.04 | 19.41 | 76.32 | 1.00 | 1.00 | 1.00 | 3354 | 1.00 | compile |
| 6 | Mirror V-local, MLP + attention | 1.47 | 22.73 | 93.68 | 0.70 | 0.85 | 0.81 | 2733 | 1.16 | compile |
| 6 | Mirror V-exact, MLP + attention | 1.55 | 22.90 | 96.60 | 0.67 | 0.85 | 0.79 | 2650 | 1.24 | compile |
| 6 | Mirror V-local, MLP only | 1.39 | 21.18 | 85.58 | 0.75 | 0.92 | 0.89 | 2991 | 1.10 | compile |
| 6 | SVD at matched parameters | 1.49 | 18.96 | 76.51 | 0.70 | 1.02 | 1.00 | 3346 | 1.24 | compile |
| 6 | MLP width pruning at matched parameters | 1.11 | 16.20 | 64.96 | 0.93 | 1.20 | 1.17 | 3941 | 1.24 | compile |
| 9 | Mirror V-local, MLP + attention | 1.69 | 25.45 | 106.25 | 0.61 | 0.76 | 0.72 | 2409 | 1.19 | compile |
| 9 | Mirror V-exact, MLP + attention | 1.74 | 24.64 | 102.50 | 0.60 | 0.79 | 0.74 | 2498 | 1.30 | compile |
| 9 | Mirror V-local, MLP only | 1.55 | 22.73 | 92.63 | 0.67 | 0.85 | 0.82 | 2764 | 1.11 | compile |
| 9 | SVD at matched parameters | 1.71 | 19.67 | 79.18 | 0.61 | 0.99 | 0.96 | 3233 | 1.32 | compile |
| 9 | MLP width pruning at matched parameters | 1.12 | 15.40 | 62.40 | 0.92 | 1.26 | 1.22 | 4103 | 1.32 | compile |
| 12 | Mirror V-local, MLP + attention | 1.89 | 28.88 | 120.28 | 0.55 | 0.67 | 0.63 | 2128 | 1.18 | compile |
| 12 | Mirror V-exact, MLP + attention | 2.00 | 21.82 | 88.82 | 0.52 | 0.89 | 0.86 | 2882 | 1.29 | compile |
| 12 | Mirror V-local, MLP only | 1.67 | 24.77 | 100.91 | 0.62 | 0.78 | 0.76 | 2537 | 1.10 | compile |
| 12 | SVD at matched parameters | 1.89 | 20.97 | 84.64 | 0.55 | 0.93 | 0.90 | 3025 | 1.33 | compile |
| 12 | MLP width pruning at matched parameters | 1.16 | 15.64 | 63.45 | 0.90 | 1.24 | 1.20 | 4035 | 1.33 | compile |

## Latency, eager (bf16, median of 100 runs after 20 warm-up runs)

Speed-up: dense latency over the variant's latency (dense = folded original in the same implementation); FLOP speed-up: dense analytic multiply-adds over the variant's. Original: the hub model as released.

| k | variant | ms (batch 1) | ms (batch 64) | ms (batch 256) | speed-up (b1) | speed-up (b64) | speed-up (b256) | img/s (b256) | FLOP speed-up | mode used |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | original (hub model as released) | 1.82 | 20.28 | 81.74 | 0.93 | 1.03 | 1.11 | 3132 | 1.00 | eager |
| 0 | dense (folded original, same implementation) | 1.70 | 20.93 | 90.69 | 1.00 | 1.00 | 1.00 | 2823 | 1.00 | eager |
| 6 | Mirror V-local, MLP + attention | 2.81 | 24.30 | 107.89 | 0.61 | 0.86 | 0.84 | 2373 | 1.16 | eager |
| 6 | Mirror V-exact, MLP + attention | 3.27 | 24.28 | 109.68 | 0.52 | 0.86 | 0.83 | 2334 | 1.24 | eager |
| 6 | Mirror V-local, MLP only | 2.41 | 22.57 | 100.48 | 0.71 | 0.93 | 0.90 | 2548 | 1.10 | eager |
| 6 | SVD at matched parameters | 2.26 | 20.89 | 92.76 | 0.75 | 1.00 | 0.98 | 2760 | 1.24 | eager |
| 6 | MLP width pruning at matched parameters | 1.73 | 17.60 | 78.41 | 0.99 | 1.19 | 1.16 | 3265 | 1.24 | eager |
| 9 | Mirror V-local, MLP + attention | 3.35 | 26.42 | 117.14 | 0.51 | 0.79 | 0.77 | 2185 | 1.19 | eager |
| 9 | Mirror V-exact, MLP + attention | 3.96 | 26.72 | 119.45 | 0.43 | 0.78 | 0.76 | 2143 | 1.30 | eager |
| 9 | Mirror V-local, MLP only | 2.75 | 23.55 | 105.31 | 0.62 | 0.89 | 0.86 | 2431 | 1.11 | eager |
| 9 | SVD at matched parameters | 2.54 | 21.50 | 95.88 | 0.67 | 0.97 | 0.95 | 2670 | 1.32 | eager |
| 9 | MLP width pruning at matched parameters | 1.71 | 16.85 | 75.56 | 1.00 | 1.24 | 1.20 | 3388 | 1.32 | eager |
| 12 | Mirror V-local, MLP + attention | 3.84 | 28.84 | 127.29 | 0.44 | 0.73 | 0.71 | 2011 | 1.18 | eager |
| 12 | Mirror V-exact, MLP + attention | 4.66 | 28.74 | 129.81 | 0.36 | 0.73 | 0.70 | 1972 | 1.29 | eager |
| 12 | Mirror V-local, MLP only | 3.06 | 25.01 | 110.91 | 0.56 | 0.84 | 0.82 | 2308 | 1.10 | eager |
| 12 | SVD at matched parameters | 2.81 | 22.67 | 100.56 | 0.61 | 0.92 | 0.90 | 2546 | 1.33 | eager |
| 12 | MLP width pruning at matched parameters | 1.73 | 16.64 | 75.28 | 0.99 | 1.26 | 1.20 | 3401 | 1.33 | eager |

## What is folded and what is not

- All norms are RMSNorm on the mean-zero stream (full LayerNorm folding, X4); LayerScale and the norm affines are folded into the weights. The same holds for the dense, SVD and pruning baselines, so the comparison with `dense` isolates the structure; `original` also differs by the hub code (LayerNorm, LayerScale, position interpolation at every forward, its own attention kernel).
- V-exact: the rotation into the A* basis is folded into the patch embedding, CLS token, registers and position embedding; blocks after the span are rewritten in that basis (free); the rotation out is applied to the CLS feature only (d x d per image; with the even head of full conversion only the d+ even rows). No per-token rotation.
- V-local and MLP only: the rotation into the first basis is folded into the embeddings and the blocks after the span are rewritten in the last basis (free); one d x d rotation of the stream per converted block is computed explicitly (counted in FLOPs and latency), plus the CLS rotation out.
- V-local attention output: M1 projected Wo with A_(l+1). That operator is not the block's input operator A_l, so Wo cannot stay structured in the block's basis without a second rotation. `vlocal` stores Wo dense in that basis (the rotation folded into it: d x H dh multiply-adds, more than Wo's free parameters, which is why its kept fraction exceeds M1's formula); `vlocal_r2` (if run) keeps Wo structured and rotates the attention output (a second d^2 per token).
- Self-heads: their head-internal bases (eigenvectors of B and C) are folded into the q, k, v and Wo weights. Twin heads are computed once as u + v and u - v.
- Alignment: structured, low-rank and pruned GEMM dimensions are zero-padded to multiples of 8; padding changes no output and is excluded from Params and GFLOPs (included in GFLOPs executed).

## Caveats

- Measured on one device (NVIDIA GeForce RTX 5090) in one session; latencies are medians of single forwards (CUDA events on a GPU), no CUDA graphs. Device state before and after timing (nvidia-smi): NVIDIA GeForce RTX 5090, 0 %, 4 MiB, 32607 MiB, 195 MHz; NVIDIA GeForce RTX 5090, 100 %, 1800 MiB, 32607 MiB, 2947 MHz. The GPU is shared: other processes running at the same time would inflate latencies.
- torch.compile default mode (no autotuning); the efficient models use plain PyTorch ops (no custom kernels). The butterfly, gathers and concatenations are fused by the compiler but still cost memory traffic.
- SVD and pruning weights are for timing only (their accuracy comes from M1 and M3); pruning keeps neurons by weight norms, not activations. Both are matched to Mirror V-local's matrix parameters (M1 formula).
- One structure (fit seed of the M1 run); latency does not depend on the fitted values, only on the shapes.

Files: m2_latency.csv, m2_checks.csv, m2_meta.json (config, seed, git hash, environment).
