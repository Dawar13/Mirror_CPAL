# M2: efficient mirror-converted blocks and latency

Model `dinov2_vitb14_reg` (12 blocks, d 768, h 3072, 12 heads, 261 tokens per image at 224 px). Structure: `/home/dawar/Mirror_CPAL/runs/m1/m1_structure.npz`, operators `/home/dawar/Mirror_CPAL/runs/x2/perlayer_reg_s0/results/e1_operators.npz`. Git `8a7b6c4e42` (uncommitted changes), seed 0. Device: NVIDIA A100-SXM4-80GB, torch 2.11.0+cu128. Written 2026-11-08 16:25:29.

**Measured speed-ups are expected to be smaller than the FLOP savings.** The converted blocks replace a few large matrix multiplications by more, smaller ones (even and odd parts), plus butterfly additions, gathers and concatenations that move memory but do no multiply-adds; token mixing (attention over tokens) is unchanged; at batch 1 the GPU is limited by kernel launches rather than arithmetic; and V-local pays a d x d basis rotation per converted block. The FLOP column below is analytic; the latency columns are measured.

## Equivalence checks

Each efficient model against the dense model it was built from, on validation images [826, 2048, 3761, 13488, 15390, 25554, 31844, 42525] (seed 0). float64: both built in float64 (criterion: max relative error of the CLS feature under 1e-6). Dropped: relative size (Frobenius) of the non-equivariant part of the folded weights that the even/odd form leaves out; it is zero in exact arithmetic. LayerNorm folding centres every write, which keeps the weights equivariant only if each operator fixes the all-ones direction exactly; largest |A 1 - 1| / sqrt(d) of the operators used: 3.3e-07 (per-layer), 1.2e-11 (span operators) (fitted operators stored in float32; this is what the dropped part measures). fp32 (TF32 off) and bf16: max abs difference of the L2-normalised CLS feature (the k-NN feature), under 1e-5 and 1e-3. Noise floor: the dense model in bf16 against itself in fp32 (same feature). Raw: the CLS feature before normalisation (absolute; largest entry in brackets).

| variant | k | float64 rel. | dropped (float64) | fp32 unit | fp32 raw | bf16 unit | bf16 noise floor | bf16 / noise | pass (64 / 32 / bf16) |
|---|---|---|---|---|---|---|---|---|---|
| dense (folded original, same implementation) | 0 | 4.6e-15 | 0.0e+00 | 1.4e-06 | 4.0e-05 (3.8) | 4.2e-03 | 3.0e-03 | 1.40 | yes / yes / NO |
| Mirror V-local, MLP + attention | 6 | 6.4e-10 | 6.7e-11 | 4.1e-06 | 1.1e-04 (4.1) | 9.0e-03 | 3.8e-03 | 2.39 | yes / yes / NO |
| Mirror V-local, Wo structured + 2nd rotation | 6 | 6.1e-10 | 6.7e-11 | 3.6e-06 | 9.4e-05 (4.1) | 1.0e-02 | 3.8e-03 | 2.67 | yes / yes / NO |
| Mirror V-exact, MLP + attention | 6 | 1.6e-14 | 1.2e-15 | 2.5e-06 | 6.7e-05 (3.8) | 1.2e-02 | 2.5e-03 | 4.93 | yes / yes / NO |
| Mirror V-local, MLP only | 6 | 7.2e-10 | 6.7e-11 | 4.2e-06 | 1.1e-04 (3.5) | 7.5e-03 | 3.0e-03 | 2.46 | yes / yes / NO |
| SVD at matched parameters | 6 | 5.5e-15 | 0.0e+00 | 1.8e-06 | 4.7e-05 (5.8) | 1.0e-02 | 1.4e-02 | 0.70 | yes / yes / NO |
| MLP width pruning at matched parameters | 6 | 3.6e-15 | 0.0e+00 | 3.3e-07 | 8.8e-06 (5.1) | 2.0e-03 | 4.9e-03 | 0.41 | yes / yes / NO |
| Mirror V-local, MLP + attention | 9 | 7.7e-10 | 8.1e-11 | 1.8e-06 | 4.6e-05 (4.2) | 9.3e-03 | 3.0e-03 | 3.06 | yes / yes / NO |
| Mirror V-local, Wo structured + 2nd rotation | 9 | 7.9e-10 | 8.1e-11 | 2.3e-06 | 5.2e-05 (4.2) | 8.3e-03 | 3.0e-03 | 2.71 | yes / yes / NO |
| Mirror V-exact, MLP + attention | 9 | 2.2e-14 | 1.1e-15 | 2.2e-06 | 5.9e-05 (3.7) | 5.4e-03 | 2.6e-03 | 2.11 | yes / yes / NO |
| Mirror V-local, MLP only | 9 | 7.6e-10 | 8.1e-11 | 2.3e-06 | 6.5e-05 (3.7) | 5.2e-03 | 3.3e-03 | 1.56 | yes / yes / NO |
| SVD at matched parameters | 9 | 3.3e-15 | 0.0e+00 | 3.0e-07 | 8.3e-06 (3.7) | 1.8e-03 | 2.6e-03 | 0.71 | yes / yes / NO |
| MLP width pruning at matched parameters | 9 | 2.6e-15 | 0.0e+00 | 3.1e-07 | 8.1e-06 (3.9) | 3.6e-03 | 4.1e-03 | 0.86 | yes / yes / NO |
| Mirror V-local, MLP + attention | 12 | 7.7e-10 | 8.1e-11 | 3.9e-06 | 1.0e-04 (4.2) | 7.4e-03 | 4.3e-03 | 1.73 | yes / yes / NO |
| Mirror V-local, Wo structured + 2nd rotation | 12 | 8.0e-10 | 8.1e-11 | 3.0e-06 | 7.7e-05 (4.2) | 9.6e-03 | 4.3e-03 | 2.25 | yes / yes / NO |
| Mirror V-exact, MLP + attention | 12 | 1.4e-14 | 1.1e-15 | 2.9e-06 | 7.6e-05 (9.3) | 1.4e-02 | 7.7e-03 | 1.78 | yes / yes / NO |
| Mirror V-local, MLP only | 12 | 7.8e-10 | 8.1e-11 | 3.3e-06 | 8.7e-05 (3.6) | 7.1e-03 | 3.1e-03 | 2.27 | yes / yes / NO |
| SVD at matched parameters | 12 | 4.2e-15 | 0.0e+00 | 1.3e-06 | 2.6e-05 (3.1) | 3.9e-03 | 3.0e-03 | 1.31 | yes / yes / NO |
| MLP width pruning at matched parameters | 12 | 2.4e-15 | 0.0e+00 | 3.0e-07 | 7.6e-06 (3.5) | 2.1e-03 | 3.4e-03 | 0.62 | yes / yes / NO |

Outcome: float64: 19 of 19 pass; fp32: 19 of 19 pass; bf16: 0 of 19 pass. In bf16 the efficient-versus-dense difference is 0.41 to 4.93 times the dense model's own bf16 rounding (noise floor 2.5e-03 to 1.4e-02). The bf16 bar of 1e-3 is below that rounding level, so a bf16 failure here reflects bf16 arithmetic, not the implementation; the float64 and fp32 rows are the equivalence checks.

## Parameters and analytic cost

Params: whole model, M1 convention (original count minus the dense matrices of the converted blocks plus the matrices the efficient model stores; biases and embeddings as in the original). Kept: stored matrix parameters of the converted blocks over dense. M1 formula: the count m1_knn.csv uses for this variant. Rotations: stored rotation matrices (d x d each, not in Params). GFLOPs per image: analytic, rotations included; executed adds the alignment padding.

| k | variant | params | kept | M1 formula | d x d rotations per token | GFLOPs | saving vs dense | GFLOPs executed |
|---|---|---|---|---|---|---|---|---|
| 0 | original (hub model as released) | 86,583,552 | 1.000 | 86,583,552 | 0 | 47.08 | 0.0% | 47.08 |
| 0 | dense (folded original, same implementation) | 86,583,552 | 1.000 | 86,583,552 | 0 | 47.08 | 0.0% | 47.08 |
| 6 | Mirror V-local, MLP + attention | 70,472,166 | 0.621 | 69,095,592 | 6 | 40.52 | 13.9% | 40.62 |
| 6 | Mirror V-local, Wo structured + 2nd rotation | 69,095,592 | 0.588 | 69,095,592 | 12 | 41.65 | 11.5% | 41.77 |
| 6 | Mirror V-exact, MLP + attention | 69,194,400 | 0.591 | 69,194,400 | 0 | 38.00 | 19.3% | 38.04 |
| 6 | Mirror V-local, MLP only | 74,734,630 | 0.721 | 74,734,630 | 6 | 42.74 | 9.2% | 42.82 |
| 6 | SVD at matched parameters | 69,061,632 | 0.587 | 69,061,632 | 0 | 37.93 | 19.4% | 38.07 |
| 6 | MLP width pruning at matched parameters | 69,091,584 | 0.588 | n/a | 0 | 37.95 | 19.4% | 37.96 |
| 9 | Mirror V-local, MLP + attention | 66,678,872 | 0.688 | 64,979,688 | 9 | 39.46 | 16.2% | 39.65 |
| 9 | Mirror V-local, Wo structured + 2nd rotation | 64,979,688 | 0.661 | 64,979,688 | 18 | 41.34 | 12.2% | 41.56 |
| 9 | Mirror V-exact, MLP + attention | 65,735,376 | 0.673 | 65,735,376 | 0 | 36.20 | 23.1% | 36.44 |
| 9 | Mirror V-local, MLP only | 72,065,992 | 0.772 | 72,065,992 | 9 | 42.27 | 10.2% | 42.41 |
| 9 | SVD at matched parameters | 64,932,864 | 0.660 | 64,932,864 | 0 | 35.78 | 24.0% | 35.98 |
| 9 | MLP width pruning at matched parameters | 64,972,032 | 0.661 | n/a | 0 | 35.80 | 24.0% | 35.82 |
| 12 | Mirror V-local, MLP + attention | 65,879,016 | 0.756 | 64,090,370 | 12 | 39.97 | 15.1% | 40.21 |
| 12 | Mirror V-local, Wo structured + 2nd rotation | 64,090,370 | 0.735 | 64,090,370 | 24 | 42.73 | 9.2% | 43.01 |
| 12 | Mirror V-exact, MLP + attention | 66,203,360 | 0.760 | 66,203,360 | 0 | 36.44 | 22.6% | 36.78 |
| 12 | Mirror V-local, MLP only | 71,636,896 | 0.824 | 71,636,896 | 12 | 42.97 | 8.7% | 43.15 |
| 12 | SVD at matched parameters | 64,024,320 | 0.734 | 64,024,320 | 0 | 35.30 | 25.0% | 35.59 |
| 12 | MLP width pruning at matched parameters | 64,079,616 | 0.735 | n/a | 0 | 35.33 | 25.0% | 35.37 |

## Latency, compile (bf16, median of 100 runs after 20 warm-up runs)

Speed-up: dense latency over the variant's latency (dense = folded original in the same implementation); FLOP speed-up: dense analytic multiply-adds over the variant's. Original: the hub model as released.

| k | variant | ms (batch 1) | ms (batch 64) | ms (batch 256) | speed-up (b1) | speed-up (b64) | speed-up (b256) | img/s (b256) | FLOP speed-up | mode used |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | original (hub model as released) | 1.38 | 92.11 | 347.96 | 0.93 | 0.93 | 0.92 | 736 | 1.00 | compile |
| 0 | dense (folded original, same implementation) | 1.29 | 85.84 | 320.31 | 1.00 | 1.00 | 1.00 | 799 | 1.00 | compile |
| 6 | Mirror V-local, MLP + attention | 6.37 | 77.71 | 326.05 | 0.20 | 1.10 | 0.98 | 785 | 1.16 | compile |
| 6 | Mirror V-local, Wo structured + 2nd rotation | 1.72 | 104.07 | 416.19 | 0.75 | 0.82 | 0.77 | 615 | 1.13 | compile |
| 6 | Mirror V-exact, MLP + attention | 6.15 | 80.44 | 316.35 | 0.21 | 1.07 | 1.01 | 809 | 1.24 | compile |
| 6 | Mirror V-local, MLP only | 1.49 | 71.77 | 296.27 | 0.86 | 1.20 | 1.08 | 864 | 1.10 | compile |
| 6 | SVD at matched parameters | 7.49 | 71.09 | 281.72 | 0.17 | 1.21 | 1.14 | 909 | 1.24 | compile |
| 6 | MLP width pruning at matched parameters | 1.22 | 59.78 | 240.82 | 1.06 | 1.44 | 1.33 | 1063 | 1.24 | compile |
| 9 | Mirror V-local, MLP + attention | 7.52 | 91.43 | 348.47 | 0.17 | 0.94 | 0.92 | 735 | 1.19 | compile |
| 9 | Mirror V-local, Wo structured + 2nd rotation | 7.20 | 76.95 | 470.64 | 0.18 | 1.12 | 0.68 | 544 | 1.14 | compile |
| 9 | Mirror V-exact, MLP + attention | 6.49 | 86.91 | 301.47 | 0.20 | 0.99 | 1.06 | 849 | 1.30 | compile |
| 9 | Mirror V-local, MLP only | 4.23 | 52.45 | 208.39 | 0.31 | 1.64 | 1.54 | 1228 | 1.11 | compile |
| 9 | SVD at matched parameters | 4.34 | 44.85 | 193.94 | 0.30 | 1.91 | 1.65 | 1320 | 1.32 | compile |
| 9 | MLP width pruning at matched parameters | 1.20 | 35.38 | 149.95 | 1.07 | 2.43 | 2.14 | 1707 | 1.32 | compile |
| 12 | Mirror V-local, MLP + attention | 4.48 | 62.43 | 257.26 | 0.29 | 1.38 | 1.25 | 995 | 1.18 | compile |
| 12 | Mirror V-local, Wo structured + 2nd rotation | 6.92 | 67.58 | 166.71 | 0.19 | 1.27 | 1.92 | 1536 | 1.10 | compile |
| 12 | Mirror V-exact, MLP + attention | 6.83 | 93.24 | 360.04 | 0.19 | 0.92 | 0.89 | 711 | 1.29 | compile |
| 12 | Mirror V-local, MLP only | 1.75 | 80.65 | 321.73 | 0.74 | 1.06 | 1.00 | 796 | 1.10 | compile |
| 12 | SVD at matched parameters | 1.87 | 88.72 | 285.82 | 0.69 | 0.97 | 1.12 | 896 | 1.33 | compile |
| 12 | MLP width pruning at matched parameters | 1.22 | 55.35 | 226.84 | 1.06 | 1.55 | 1.41 | 1129 | 1.33 | compile |

## Latency, eager (bf16, median of 100 runs after 20 warm-up runs)

Speed-up: dense latency over the variant's latency (dense = folded original in the same implementation); FLOP speed-up: dense analytic multiply-adds over the variant's. Original: the hub model as released.

| k | variant | ms (batch 1) | ms (batch 64) | ms (batch 256) | speed-up (b1) | speed-up (b64) | speed-up (b256) | img/s (b256) | FLOP speed-up | mode used |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | original (hub model as released) | 8.73 | 101.94 | 389.53 | 0.92 | 0.84 | 0.86 | 657 | 1.00 | eager |
| 0 | dense (folded original, same implementation) | 8.06 | 86.05 | 336.48 | 1.00 | 1.00 | 1.00 | 761 | 1.00 | eager |
| 6 | Mirror V-local, MLP + attention | 11.76 | 115.04 | 466.90 | 0.69 | 0.75 | 0.72 | 548 | 1.16 | eager |
| 6 | Mirror V-local, Wo structured + 2nd rotation | 12.46 | 146.89 | 583.46 | 0.65 | 0.59 | 0.58 | 439 | 1.13 | eager |
| 6 | Mirror V-exact, MLP + attention | 11.01 | 109.02 | 474.95 | 0.73 | 0.79 | 0.71 | 539 | 1.24 | eager |
| 6 | Mirror V-local, MLP only | 7.45 | 100.57 | 366.25 | 1.08 | 0.86 | 0.92 | 699 | 1.10 | eager |
| 6 | SVD at matched parameters | 9.24 | 89.33 | 346.76 | 0.87 | 0.96 | 0.97 | 738 | 1.24 | eager |
| 6 | MLP width pruning at matched parameters | 8.79 | 79.12 | 287.95 | 0.92 | 1.09 | 1.17 | 889 | 1.24 | eager |
| 9 | Mirror V-local, MLP + attention | 15.23 | 127.64 | 526.27 | 0.53 | 0.67 | 0.64 | 486 | 1.19 | eager |
| 9 | Mirror V-local, Wo structured + 2nd rotation | 13.20 | 173.50 | 691.04 | 0.61 | 0.50 | 0.49 | 370 | 1.14 | eager |
| 9 | Mirror V-exact, MLP + attention | 13.36 | 117.46 | 474.50 | 0.60 | 0.73 | 0.71 | 540 | 1.30 | eager |
| 9 | Mirror V-local, MLP only | 7.48 | 67.74 | 278.42 | 1.08 | 1.27 | 1.21 | 919 | 1.11 | eager |
| 9 | SVD at matched parameters | 7.41 | 58.31 | 240.10 | 1.09 | 1.48 | 1.40 | 1066 | 1.32 | eager |
| 9 | MLP width pruning at matched parameters | 4.16 | 45.06 | 188.07 | 1.94 | 1.91 | 1.79 | 1361 | 1.32 | eager |
| 12 | Mirror V-local, MLP + attention | 12.33 | 92.12 | 323.08 | 0.65 | 0.93 | 1.04 | 792 | 1.18 | eager |
| 12 | Mirror V-local, Wo structured + 2nd rotation | 16.70 | 114.54 | 574.93 | 0.48 | 0.75 | 0.59 | 445 | 1.10 | eager |
| 12 | Mirror V-exact, MLP + attention | 17.48 | 150.29 | 572.41 | 0.46 | 0.57 | 0.59 | 447 | 1.29 | eager |
| 12 | Mirror V-local, MLP only | 9.37 | 124.84 | 513.66 | 0.86 | 0.69 | 0.66 | 498 | 1.10 | eager |
| 12 | SVD at matched parameters | 7.56 | 94.12 | 353.00 | 1.07 | 0.91 | 0.95 | 725 | 1.33 | eager |
| 12 | MLP width pruning at matched parameters | 4.65 | 67.19 | 273.99 | 1.73 | 1.28 | 1.23 | 934 | 1.33 | eager |

## What is folded and what is not

- All norms are RMSNorm on the mean-zero stream (full LayerNorm folding, X4); LayerScale and the norm affines are folded into the weights. The same holds for the dense, SVD and pruning baselines, so the comparison with `dense` isolates the structure; `original` also differs by the hub code (LayerNorm, LayerScale, position interpolation at every forward, its own attention kernel).
- V-exact: the rotation into the A* basis is folded into the patch embedding, CLS token, registers and position embedding; blocks after the span are rewritten in that basis (free); the rotation out is applied to the CLS feature only (d x d per image; with the even head of full conversion only the d+ even rows). No per-token rotation.
- V-local and MLP only: the rotation into the first basis is folded into the embeddings and the blocks after the span are rewritten in the last basis (free); one d x d rotation of the stream per converted block is computed explicitly (counted in FLOPs and latency), plus the CLS rotation out.
- V-local attention output: M1 projected Wo with A_(l+1). That operator is not the block's input operator A_l, so Wo cannot stay structured in the block's basis without a second rotation. `vlocal` stores Wo dense in that basis (the rotation folded into it: d x H dh multiply-adds, more than Wo's free parameters, which is why its kept fraction exceeds M1's formula); `vlocal_r2` (if run) keeps Wo structured and rotates the attention output (a second d^2 per token).
- Self-heads: their head-internal bases (eigenvectors of B and C) are folded into the q, k, v and Wo weights. Twin heads are computed once as u + v and u - v.
- Alignment: structured, low-rank and pruned GEMM dimensions are zero-padded to multiples of 8; padding changes no output and is excluded from Params and GFLOPs (included in GFLOPs executed).

## Caveats

- Measured on one device (NVIDIA A100-SXM4-80GB) in one session; latencies are medians of single forwards (CUDA events on a GPU), no CUDA graphs. Device state before and after timing (nvidia-smi): NVIDIA A100-SXM4-80GB, 100 %, 27921 MiB, 81920 MiB, 1395 MHz; NVIDIA A100-SXM4-80GB, 100 %, 21854 MiB, 81920 MiB, 1395 MHz. The GPU is shared: other processes running at the same time would inflate latencies.
- torch.compile default mode (no autotuning); the efficient models use plain PyTorch ops (no custom kernels). The butterfly, gathers and concatenations are fused by the compiler but still cost memory traffic.
- SVD and pruning weights are for timing only (their accuracy comes from M1 and M3); pruning keeps neurons by weight norms, not activations. Both are matched to Mirror V-local's matrix parameters (M1 formula).
- One structure (fit seed of the M1 run); latency does not depend on the fitted values, only on the shapes.

Files: m2_latency.csv, m2_checks.csv, m2_meta.json (config, seed, git hash, environment).
