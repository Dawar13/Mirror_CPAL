# M2: efficient mirror-converted blocks and latency

Model `dinov2_vitb14_reg` (12 blocks, d 768, h 3072, 12 heads, 261 tokens per image at 224 px). Structure: `/home/dawar/Mirror_CPAL/runs/m1/m1_structure.npz`, operators `/home/dawar/Mirror_CPAL/runs/x2/perlayer_reg_s0/results/e1_operators.npz`. Git `ced98c85c9` (uncommitted changes), seed 0. Device: NVIDIA A100-SXM4-80GB, torch 2.11.0+cu128. Written 2026-11-08 18:36:03.

**Measured speed-ups are expected to be smaller than the FLOP savings.** The converted blocks replace a few large matrix multiplications by more, smaller ones (even and odd parts), plus butterfly additions, gathers and concatenations that move memory but do no multiply-adds; token mixing (attention over tokens) is unchanged; at batch 1 the GPU is limited by kernel launches rather than arithmetic; and V-local pays a d x d basis rotation per converted block. The FLOP column below is analytic; the latency columns are measured.

## Equivalence checks

Each efficient model against the dense model it was built from, on validation images [826, 2048, 3761, 13488, 15390, 25554, 31844, 42525] (seed 0). float64: both built in float64 (criterion: max relative error of the CLS feature under 1e-6). Dropped: relative size (Frobenius) of the non-equivariant part of the folded weights that the even/odd form leaves out; it is zero in exact arithmetic. LayerNorm folding centres every write, which keeps the weights equivariant only if each operator fixes the all-ones direction exactly; largest |A 1 - 1| / sqrt(d) of the operators used: 3.3e-07 (per-layer), 1.2e-11 (span operators) (fitted operators stored in float32; this is what the dropped part measures). fp32 (TF32 off) and bf16: max abs difference of the L2-normalised CLS feature (the k-NN feature), under 1e-5 and 1e-3. Noise floor: the dense model in bf16 against itself in fp32 (same feature). Raw: the CLS feature before normalisation (absolute; largest entry in brackets).

| variant | k | float64 rel. | dropped (float64) | fp32 unit | fp32 raw | bf16 unit | bf16 noise floor | bf16 / noise | pass (64 / 32 / bf16) |
|---|---|---|---|---|---|---|---|---|---|
| dense (folded original, same implementation) | 0 | 4.6e-15 | 0.0e+00 | 1.4e-06 | 4.0e-05 (3.8) | 4.2e-03 | 3.0e-03 | 1.40 | yes / yes / NO |
| Mirror V-local, MLP + attention | 6 | 6.4e-10 | 6.7e-11 | 4.1e-06 | 1.1e-04 (4.1) | 9.0e-03 | 3.8e-03 | 2.39 | yes / yes / NO |
| Mirror V-exact, MLP + attention | 6 | 1.6e-14 | 1.2e-15 | 2.5e-06 | 6.7e-05 (3.8) | 1.2e-02 | 2.5e-03 | 4.93 | yes / yes / NO |
| Mirror V-local, MLP only | 6 | 7.2e-10 | 6.7e-11 | 4.2e-06 | 1.1e-04 (3.5) | 7.5e-03 | 3.0e-03 | 2.46 | yes / yes / NO |
| SVD at matched parameters | 6 | 5.5e-15 | 0.0e+00 | 1.8e-06 | 4.7e-05 (5.8) | 1.0e-02 | 1.4e-02 | 0.70 | yes / yes / NO |
| MLP width pruning at matched parameters | 6 | 3.6e-15 | 0.0e+00 | 3.3e-07 | 8.8e-06 (5.1) | 2.0e-03 | 4.9e-03 | 0.41 | yes / yes / NO |
| Mirror V-local, MLP + attention | 9 | 7.7e-10 | 8.1e-11 | 1.8e-06 | 4.6e-05 (4.2) | 9.3e-03 | 3.0e-03 | 3.06 | yes / yes / NO |
| Mirror V-exact, MLP + attention | 9 | 2.2e-14 | 1.1e-15 | 2.2e-06 | 5.9e-05 (3.7) | 5.4e-03 | 2.6e-03 | 2.11 | yes / yes / NO |
| Mirror V-local, MLP only | 9 | 7.6e-10 | 8.1e-11 | 2.3e-06 | 6.5e-05 (3.7) | 5.2e-03 | 3.3e-03 | 1.56 | yes / yes / NO |
| SVD at matched parameters | 9 | 3.3e-15 | 0.0e+00 | 3.0e-07 | 8.3e-06 (3.7) | 1.8e-03 | 2.6e-03 | 0.71 | yes / yes / NO |
| MLP width pruning at matched parameters | 9 | 2.6e-15 | 0.0e+00 | 3.1e-07 | 8.1e-06 (3.9) | 3.6e-03 | 4.1e-03 | 0.86 | yes / yes / NO |
| Mirror V-local, MLP + attention | 12 | 7.7e-10 | 8.1e-11 | 3.9e-06 | 1.0e-04 (4.2) | 7.4e-03 | 4.3e-03 | 1.73 | yes / yes / NO |
| Mirror V-exact, MLP + attention | 12 | 1.4e-14 | 1.1e-15 | 2.9e-06 | 7.6e-05 (9.3) | 1.4e-02 | 7.7e-03 | 1.78 | yes / yes / NO |
| Mirror V-local, MLP only | 12 | 7.8e-10 | 8.1e-11 | 3.3e-06 | 8.7e-05 (3.6) | 7.1e-03 | 3.1e-03 | 2.27 | yes / yes / NO |
| SVD at matched parameters | 12 | 4.2e-15 | 0.0e+00 | 1.3e-06 | 2.6e-05 (3.1) | 3.9e-03 | 3.0e-03 | 1.31 | yes / yes / NO |
| MLP width pruning at matched parameters | 12 | 2.4e-15 | 0.0e+00 | 3.0e-07 | 7.6e-06 (3.5) | 2.1e-03 | 3.4e-03 | 0.62 | yes / yes / NO |

Outcome: float64: 16 of 16 pass; fp32: 16 of 16 pass; bf16: 0 of 16 pass. In bf16 the efficient-versus-dense difference is 0.41 to 4.93 times the dense model's own bf16 rounding (noise floor 2.5e-03 to 1.4e-02). The bf16 bar of 1e-3 is below that rounding level, so a bf16 failure here reflects bf16 arithmetic, not the implementation; the float64 and fp32 rows are the equivalence checks.

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
| 0 | original (hub model as released) | 3.88 | 26.63 | 102.48 | 0.92 | 1.01 | 1.00 | 2498 | 1.00 | compile |
| 0 | dense (folded original, same implementation) | 3.56 | 26.77 | 101.97 | 1.00 | 1.00 | 1.00 | 2511 | 1.00 | compile |
| 6 | Mirror V-local, MLP + attention | 4.39 | 31.60 | 119.59 | 0.81 | 0.85 | 0.85 | 2141 | 1.16 | compile |
| 6 | Mirror V-exact, MLP + attention | 5.05 | 31.54 | 120.66 | 0.71 | 0.85 | 0.85 | 2122 | 1.24 | compile |
| 6 | Mirror V-local, MLP only | 4.23 | 21.47 | 112.05 | 0.84 | 1.25 | 0.91 | 2285 | 1.10 | compile |
| 6 | SVD at matched parameters | 4.71 | 28.41 | 104.54 | 0.76 | 0.94 | 0.98 | 2449 | 1.24 | compile |
| 6 | MLP width pruning at matched parameters | 3.24 | 23.37 | 90.02 | 1.10 | 1.15 | 1.13 | 2844 | 1.24 | compile |
| 9 | Mirror V-local, MLP + attention | 5.23 | 35.06 | 131.21 | 0.68 | 0.76 | 0.78 | 1951 | 1.19 | compile |
| 9 | Mirror V-exact, MLP + attention | 5.54 | 34.95 | 131.22 | 0.64 | 0.77 | 0.78 | 1951 | 1.30 | compile |
| 9 | Mirror V-local, MLP only | 4.85 | 31.25 | 118.88 | 0.74 | 0.86 | 0.86 | 2153 | 1.11 | compile |
| 9 | SVD at matched parameters | 5.23 | 29.34 | 108.77 | 0.68 | 0.91 | 0.94 | 2354 | 1.32 | compile |
| 9 | MLP width pruning at matched parameters | 3.50 | 22.68 | 87.31 | 1.02 | 1.18 | 1.17 | 2932 | 1.32 | compile |
| 12 | Mirror V-local, MLP + attention | 6.16 | 36.94 | 145.46 | 0.58 | 0.72 | 0.70 | 1760 | 1.18 | compile |
| 12 | Mirror V-exact, MLP + attention | 7.21 | 37.65 | 145.84 | 0.49 | 0.71 | 0.70 | 1755 | 1.29 | compile |
| 12 | Mirror V-local, MLP only | 5.50 | 34.32 | 127.28 | 0.65 | 0.78 | 0.80 | 2011 | 1.10 | compile |
| 12 | SVD at matched parameters | 6.57 | 30.89 | 111.48 | 0.54 | 0.87 | 0.91 | 2296 | 1.33 | compile |
| 12 | MLP width pruning at matched parameters | 3.68 | 22.78 | 87.24 | 0.97 | 1.18 | 1.17 | 2934 | 1.33 | compile |

## Latency, eager (bf16, median of 100 runs after 20 warm-up runs)

Speed-up: dense latency over the variant's latency (dense = folded original in the same implementation); FLOP speed-up: dense analytic multiply-adds over the variant's. Original: the hub model as released.

| k | variant | ms (batch 1) | ms (batch 64) | ms (batch 256) | speed-up (b1) | speed-up (b64) | speed-up (b256) | img/s (b256) | FLOP speed-up | mode used |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | original (hub model as released) | 6.01 | 30.50 | 115.35 | 0.97 | 1.09 | 1.09 | 2219 | 1.00 | eager |
| 0 | dense (folded original, same implementation) | 5.85 | 33.10 | 125.77 | 1.00 | 1.00 | 1.00 | 2035 | 1.00 | eager |
| 6 | Mirror V-local, MLP + attention | 9.04 | 44.46 | 169.44 | 0.65 | 0.74 | 0.74 | 1511 | 1.16 | eager |
| 6 | Mirror V-exact, MLP + attention | 10.35 | 45.36 | 169.41 | 0.57 | 0.73 | 0.74 | 1511 | 1.24 | eager |
| 6 | Mirror V-local, MLP only | 7.70 | 38.78 | 145.99 | 0.76 | 0.85 | 0.86 | 1753 | 1.10 | eager |
| 6 | SVD at matched parameters | 7.67 | 34.02 | 129.87 | 0.76 | 0.97 | 0.97 | 1971 | 1.24 | eager |
| 6 | MLP width pruning at matched parameters | 5.46 | 29.46 | 112.07 | 1.07 | 1.12 | 1.12 | 2284 | 1.24 | eager |
| 9 | Mirror V-local, MLP + attention | 10.76 | 50.87 | 193.58 | 0.54 | 0.65 | 0.65 | 1322 | 1.19 | eager |
| 9 | Mirror V-exact, MLP + attention | 12.91 | 53.69 | 202.42 | 0.45 | 0.62 | 0.62 | 1265 | 1.30 | eager |
| 9 | Mirror V-local, MLP only | 8.80 | 42.83 | 158.50 | 0.67 | 0.77 | 0.79 | 1615 | 1.11 | eager |
| 9 | SVD at matched parameters | 8.05 | 36.75 | 135.31 | 0.73 | 0.90 | 0.93 | 1892 | 1.32 | eager |
| 9 | MLP width pruning at matched parameters | 5.64 | 28.93 | 108.28 | 1.04 | 1.14 | 1.16 | 2364 | 1.32 | eager |
| 12 | Mirror V-local, MLP + attention | 12.84 | 57.21 | 215.89 | 0.46 | 0.58 | 0.58 | 1186 | 1.18 | eager |
| 12 | Mirror V-exact, MLP + attention | 15.32 | 60.78 | 233.16 | 0.38 | 0.54 | 0.54 | 1098 | 1.29 | eager |
| 12 | Mirror V-local, MLP only | 10.14 | 45.15 | 172.15 | 0.58 | 0.73 | 0.73 | 1487 | 1.10 | eager |
| 12 | SVD at matched parameters | 9.63 | 37.40 | 139.84 | 0.61 | 0.89 | 0.90 | 1831 | 1.33 | eager |
| 12 | MLP width pruning at matched parameters | 5.99 | 28.08 | 106.67 | 0.98 | 1.18 | 1.18 | 2400 | 1.33 | eager |

## What is folded and what is not

- All norms are RMSNorm on the mean-zero stream (full LayerNorm folding, X4); LayerScale and the norm affines are folded into the weights. The same holds for the dense, SVD and pruning baselines, so the comparison with `dense` isolates the structure; `original` also differs by the hub code (LayerNorm, LayerScale, position interpolation at every forward, its own attention kernel).
- V-exact: the rotation into the A* basis is folded into the patch embedding, CLS token, registers and position embedding; blocks after the span are rewritten in that basis (free); the rotation out is applied to the CLS feature only (d x d per image; with the even head of full conversion only the d+ even rows). No per-token rotation.
- V-local and MLP only: the rotation into the first basis is folded into the embeddings and the blocks after the span are rewritten in the last basis (free); one d x d rotation of the stream per converted block is computed explicitly (counted in FLOPs and latency), plus the CLS rotation out.
- V-local attention output: M1 projected Wo with A_(l+1). That operator is not the block's input operator A_l, so Wo cannot stay structured in the block's basis without a second rotation. `vlocal` stores Wo dense in that basis (the rotation folded into it: d x H dh multiply-adds, more than Wo's free parameters, which is why its kept fraction exceeds M1's formula); `vlocal_r2` (if run) keeps Wo structured and rotates the attention output (a second d^2 per token).
- Self-heads: their head-internal bases (eigenvectors of B and C) are folded into the q, k, v and Wo weights. Twin heads are computed once as u + v and u - v.
- Alignment: structured, low-rank and pruned GEMM dimensions are zero-padded to multiples of 8; padding changes no output and is excluded from Params and GFLOPs (included in GFLOPs executed).

## Caveats

- Measured on one device (NVIDIA A100-SXM4-80GB) in one session; latencies are medians of single forwards (CUDA events on a GPU), no CUDA graphs. Device state before and after timing (nvidia-smi): NVIDIA A100-SXM4-80GB, 27 %, 784 MiB, 81920 MiB, 1155 MHz; NVIDIA A100-SXM4-80GB, 25 %, 2459 MiB, 81920 MiB, 1410 MHz. The GPU is shared: other processes running at the same time would inflate latencies.
- torch.compile default mode (no autotuning); the efficient models use plain PyTorch ops (no custom kernels). The butterfly, gathers and concatenations are fused by the compiler but still cost memory traffic.
- SVD and pruning weights are for timing only (their accuracy comes from M1 and M3); pruning keeps neurons by weight norms, not activations. Both are matched to Mirror V-local's matrix parameters (M1 formula).
- One structure (fit seed of the M1 run); latency does not depend on the fitted values, only on the shapes.

Files: m2_latency.csv, m2_checks.csv, m2_meta.json (config, seed, git hash, environment).
