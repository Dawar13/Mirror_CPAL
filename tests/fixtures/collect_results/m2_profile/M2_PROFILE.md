# M2 profile: where the time goes in the converted models (TINY TEST, random weights, CPU operator times, not a measurement)

Model `tiny random timm ViT` (4 blocks, d 64, 17 tokens per image), batch 2, bf16. Structure `tiny planted structure`, operators `n/a`. Git `53c8aada92`, seed 0. Device: cpu, torch 2.11.0+cpu. Written 2026-11-09 00:51:18.

Method. The models are the ones `scripts/m2_latency.py` times, built by its own code (same structure and operator files, alignment 8, bf16). Per variant, k and mode: latency is the median of 2 forwards (wall clock, perf_counter, after 1 warm-up forwards, no profiler attached); then 2 more forwards were recorded with torch.profiler and the time of every GPU kernel (including memcpy and memset) was summed and divided by the number of forwards. Kernels are sorted into categories by name (`classify_kernel` in `scripts/m2_profile.py`). Idle time is latency minus kernel time (the GPU waiting for work), so for every row the latency gap to dense equals the sum of the category changes plus the idle change. Nothing was optimised.

On CPU there are no GPU kernels: every number below is CPU operator self time from the profiler. This run only checks that the pipeline works.

## Main findings (eager, batch 2)

- Mirror rows slower than dense: 4 of 6 (speed 0.62 to 1.05x of dense, analytic FLOP ratio 1.11 to 1.66x).
- GEMM kernels: 12.97 to 20.73 ms per forward in the Mirror rows against 16.70 ms for dense; Mirror rows spending more time in GEMMs than dense: 2 of 6, although they execute 8.9 to 34.2% fewer GEMM multiply-adds (analytic). Achieved GEMM throughput 0.0 TFLOP/s against 0.0 for dense; at the dense rate their GEMMs would take 10.99 to 15.21 ms.
- Non-GEMM, non-attention kernels (fused Triton, elementwise, normalisation, memory movement, other): 6.39 to 10.08 ms per forward against 8.24 ms for dense (change -1.85 to +1.84 ms).
- Attention kernels (token mixing, unchanged by the conversion): 0.69 to 2.42 ms against 1.36 ms for dense.
- Kernel launches per forward: 769 to 1201 against 650 for dense; idle time (latency minus cpu operator time): -1.44 to 11.59 ms against -4.00 ms.
- Largest single contributor to each slower Mirror row's gap: Mirror V-exact, MLP + attention k = 2: idle (+5.09 of 4.22 ms); Mirror V-local, MLP + attention k = 2: idle (+5.51 of 9.09 ms); Mirror V-local, MLP only k = 2: idle (+15.59 of 13.76 ms); Mirror V-exact, MLP + attention k = 4: idle (+7.53 of 7.76 ms).
- SVD at matched parameters, k = 2: speed 0.68x of dense; GEMM 18.30 ms (+1.60), other kernels -2.10 ms.
- MLP width pruning at matched parameters, k = 2: speed 1.28x of dense; GEMM 11.47 ms (-5.23), other kernels -3.90 ms.
- SVD at matched parameters, k = 4: speed 0.98x of dense; GEMM 13.34 ms (-3.36), other kernels -2.94 ms.
- MLP width pruning at matched parameters, k = 4: speed 1.17x of dense; GEMM 11.66 ms (-5.04), other kernels -3.11 ms.

## Per variant

Eager (mode used for dense: eager):

- Mirror V-exact, MLP + attention, k = 2: 26.52 ms per forward against 22.30 ms for dense (0.84x the dense speed; analytic FLOP ratio 1.24x). CPU operator time 25.43 ms against 26.30 ms (-0.87 ms); by category (ms): elementwise +0.43, GEMM +0.15, other +0.06, memory movement -0.12, attention -0.34, normalisation -1.05. Idle time (latency minus kernel time) +5.09 ms. Kernel launches per forward 914 against 650. GEMM throughput 0.0 against 0.0 TFLOP/s; at the dense model's throughput its GEMMs would take 13.88 ms, measured 16.85 ms.
- Mirror V-local, MLP + attention, k = 2: 31.39 ms per forward against 22.30 ms for dense (0.71x the dense speed; analytic FLOP ratio 1.18x). CPU operator time 29.88 ms against 26.30 ms (+3.59 ms); by category (ms): GEMM +4.03, attention +1.06, elementwise +0.30, other -0.35, memory movement -0.53, normalisation -0.92. Idle time (latency minus kernel time) +5.51 ms. Kernel launches per forward 834 against 650. GEMM throughput 0.0 against 0.0 TFLOP/s; at the dense model's throughput its GEMMs would take 14.40 ms, measured 20.73 ms.
- Mirror V-local, MLP only, k = 2: 36.06 ms per forward against 22.30 ms for dense (0.62x the dense speed; analytic FLOP ratio 1.11x). CPU operator time 24.47 ms against 26.30 ms (-1.83 ms); by category (ms): elementwise +0.16, other -0.51, memory movement -0.62, normalisation -0.89. Idle time (latency minus kernel time) +15.59 ms. Kernel launches per forward 769 against 650. GEMM throughput 0.0 against 0.0 TFLOP/s; at the dense model's throughput its GEMMs would take 15.21 ms, measured 16.70 ms.
- SVD at matched parameters, k = 2: 33.01 ms per forward against 22.30 ms for dense (0.68x the dense speed; analytic FLOP ratio 1.26x). CPU operator time 25.80 ms against 26.30 ms (-0.50 ms); by category (ms): GEMM +1.60, elementwise -0.07, other -0.52, memory movement -0.56, normalisation -0.99. Idle time (latency minus kernel time) +11.21 ms. Kernel launches per forward 800 against 650. GEMM throughput 0.0 against 0.0 TFLOP/s; at the dense model's throughput its GEMMs would take 13.57 ms, measured 18.30 ms.
- MLP width pruning at matched parameters, k = 2: 17.47 ms per forward against 22.30 ms for dense (1.28x the dense speed; analytic FLOP ratio 1.26x). CPU operator time 17.17 ms against 26.30 ms (-9.13 ms); by category (ms): elementwise -0.45, attention -0.52, memory movement -0.76, normalisation -1.05, other -1.11, GEMM -5.23. Idle time (latency minus kernel time) +4.30 ms. Kernel launches per forward 650 against 650. GEMM throughput 0.0 against 0.0 TFLOP/s; at the dense model's throughput its GEMMs would take 13.24 ms, measured 11.47 ms.
- Mirror V-exact, MLP + attention, k = 4: 30.06 ms per forward against 22.30 ms for dense (0.74x the dense speed; analytic FLOP ratio 1.66x). CPU operator time 26.53 ms against 26.30 ms (+0.24 ms); by category (ms): other +1.09, elementwise +1.02, memory movement +0.60, attention -0.53, normalisation -0.88, GEMM -1.08. Idle time (latency minus kernel time) +7.53 ms. Kernel launches per forward 1201 against 650. GEMM throughput 0.0 against 0.0 TFLOP/s; at the dense model's throughput its GEMMs would take 10.99 ms, measured 15.62 ms.
- Mirror V-local, MLP + attention, k = 4: 21.73 ms per forward against 22.30 ms for dense (1.03x the dense speed; analytic FLOP ratio 1.44x). CPU operator time 22.19 ms against 26.30 ms (-4.11 ms); by category (ms): elementwise +0.64, memory movement -0.07, normalisation -0.33, attention -0.67, GEMM -3.73. Idle time (latency minus kernel time) +3.54 ms. Kernel launches per forward 1019 against 650. GEMM throughput 0.0 against 0.0 TFLOP/s; at the dense model's throughput its GEMMs would take 12.18 ms, measured 12.97 ms.
- Mirror V-local, MLP only, k = 4: 21.18 ms per forward against 22.30 ms for dense (1.05x the dense speed; analytic FLOP ratio 1.26x). CPU operator time 22.62 ms against 26.30 ms (-3.68 ms); by category (ms): elementwise +0.47, memory movement +0.09, other -0.23, attention -0.41, normalisation -0.90, GEMM -2.69. Idle time (latency minus kernel time) +2.56 ms. Kernel launches per forward 879 against 650. GEMM throughput 0.0 against 0.0 TFLOP/s; at the dense model's throughput its GEMMs would take 13.73 ms, measured 14.00 ms.
- SVD at matched parameters, k = 4: 22.66 ms per forward against 22.30 ms for dense (0.98x the dense speed; analytic FLOP ratio 1.72x). CPU operator time 20.00 ms against 26.30 ms (-6.30 ms); by category (ms): elementwise -0.08, other -0.33, attention -0.63, memory movement -0.75, normalisation -1.14, GEMM -3.36. Idle time (latency minus kernel time) +6.66 ms. Kernel launches per forward 950 against 650. GEMM throughput 0.0 against 0.0 TFLOP/s; at the dense model's throughput its GEMMs would take 10.43 ms, measured 13.34 ms.
- MLP width pruning at matched parameters, k = 4: 19.03 ms per forward against 22.30 ms for dense (1.17x the dense speed; analytic FLOP ratio 1.70x). CPU operator time 18.14 ms against 26.30 ms (-8.15 ms); by category (ms): attention +0.28, elementwise -0.55, memory movement -0.71, normalisation -1.04, other -1.09, GEMM -5.04. Idle time (latency minus kernel time) +4.89 ms. Kernel launches per forward 650 against 650. GEMM throughput 0.0 against 0.0 TFLOP/s; at the dense model's throughput its GEMMs would take 9.70 ms, measured 11.66 ms.

## CPU operator time per forward by category (eager, ms)

Share: category time over total kernel time. GPU busy: kernel time over latency.

| k | variant | latency | speed vs dense | kernel time | GPU busy | launches | GEMM | attention | fused Triton | elementwise | normalisation | memory movement | other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | dense (folded original, same implementation) | 22.30 | 1.00 | 26.30 | 118% | 650 | 16.70 (64%) | 1.36 (5%) | 0.00 (0%) | 1.48 (6%) | 2.13 (8%) | 1.95 (7%) | 2.68 (10%) |
| 2 | Mirror V-exact, MLP + attention | 26.52 | 0.84 | 25.43 | 96% | 914 | 16.85 (66%) | 1.02 (4%) | 0.00 (0%) | 1.91 (7%) | 1.08 (4%) | 1.83 (7%) | 2.74 (11%) |
| 2 | Mirror V-local, MLP + attention | 31.39 | 0.71 | 29.88 | 95% | 834 | 20.73 (69%) | 2.42 (8%) | 0.00 (0%) | 1.78 (6%) | 1.21 (4%) | 1.42 (5%) | 2.33 (8%) |
| 2 | Mirror V-local, MLP only | 36.06 | 0.62 | 24.47 | 68% | 769 | 16.70 (68%) | 1.39 (6%) | 0.00 (0%) | 1.64 (7%) | 1.24 (5%) | 1.33 (5%) | 2.17 (9%) |
| 2 | SVD at matched parameters | 33.01 | 0.68 | 25.80 | 78% | 800 | 18.30 (71%) | 1.41 (5%) | 0.00 (0%) | 1.40 (5%) | 1.14 (4%) | 1.38 (5%) | 2.16 (8%) |
| 2 | MLP width pruning at matched parameters | 17.47 | 1.28 | 17.17 | 98% | 650 | 11.47 (67%) | 0.84 (5%) | 0.00 (0%) | 1.02 (6%) | 1.08 (6%) | 1.19 (7%) | 1.58 (9%) |
| 4 | Mirror V-exact, MLP + attention | 30.06 | 0.74 | 26.53 | 88% | 1201 | 15.62 (59%) | 0.83 (3%) | 0.00 (0%) | 2.50 (9%) | 1.25 (5%) | 2.55 (10%) | 3.77 (14%) |
| 4 | Mirror V-local, MLP + attention | 21.73 | 1.03 | 22.19 | 102% | 1019 | 12.97 (58%) | 0.69 (3%) | 0.00 (0%) | 2.12 (10%) | 1.81 (8%) | 1.88 (8%) | 2.72 (12%) |
| 4 | Mirror V-local, MLP only | 21.18 | 1.05 | 22.62 | 107% | 879 | 14.00 (62%) | 0.95 (4%) | 0.00 (0%) | 1.95 (9%) | 1.23 (5%) | 2.03 (9%) | 2.45 (11%) |
| 4 | SVD at matched parameters | 22.66 | 0.98 | 20.00 | 88% | 950 | 13.34 (67%) | 0.73 (4%) | 0.00 (0%) | 1.40 (7%) | 0.99 (5%) | 1.20 (6%) | 2.35 (12%) |
| 4 | MLP width pruning at matched parameters | 19.03 | 1.17 | 18.14 | 95% | 650 | 11.66 (64%) | 1.63 (9%) | 0.00 (0%) | 0.93 (5%) | 1.09 (6%) | 1.24 (7%) | 1.59 (9%) |

Change against dense (eager, ms per forward; the columns add up to the latency gap):

| k | variant | gap | GEMM | attention | fused Triton | elementwise | normalisation | memory movement | other | idle |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | Mirror V-exact, MLP + attention | +4.22 | +0.15 | -0.34 | +0.00 | +0.43 | -1.05 | -0.12 | +0.06 | +5.09 |
| 2 | Mirror V-local, MLP + attention | +9.09 | +4.03 | +1.06 | +0.00 | +0.30 | -0.92 | -0.53 | -0.35 | +5.51 |
| 2 | Mirror V-local, MLP only | +13.76 | -0.00 | +0.03 | +0.00 | +0.16 | -0.89 | -0.62 | -0.51 | +15.59 |
| 2 | SVD at matched parameters | +10.71 | +1.60 | +0.05 | +0.00 | -0.07 | -0.99 | -0.56 | -0.52 | +11.21 |
| 2 | MLP width pruning at matched parameters | -4.83 | -5.23 | -0.52 | +0.00 | -0.45 | -1.05 | -0.76 | -1.11 | +4.30 |
| 4 | Mirror V-exact, MLP + attention | +7.76 | -1.08 | -0.53 | +0.00 | +1.02 | -0.88 | +0.60 | +1.09 | +7.53 |
| 4 | Mirror V-local, MLP + attention | -0.57 | -3.73 | -0.67 | +0.00 | +0.64 | -0.33 | -0.07 | +0.04 | +3.54 |
| 4 | Mirror V-local, MLP only | -1.12 | -2.69 | -0.41 | +0.00 | +0.47 | -0.90 | +0.09 | -0.23 | +2.56 |
| 4 | SVD at matched parameters | +0.36 | -3.36 | -0.63 | +0.00 | -0.08 | -1.14 | -0.75 | -0.33 | +6.66 |
| 4 | MLP width pruning at matched parameters | -3.26 | -5.04 | +0.28 | +0.00 | -0.55 | -1.04 | -0.71 | -1.09 | +4.89 |

## GEMM efficiency

Analytic GEMM work: multiply-adds executed by matrix products per forward (patch embedding, every weight matrix with its alignment padding, rotations; token mixing excluded, it runs inside the attention kernel), from `mirror.efficient.fast_vit_cost`. Throughput: 2 x that work over the measured GEMM kernel time. At dense rate: the GEMM time this work would take at the dense model's measured throughput.

| mode | k | variant | GEMM ms | GEMM launches | analytic GEMM GFLOP | TFLOP/s | at dense rate (ms) |
|---|---|---|---|---|---|---|---|
| eager | 0 | dense (folded original, same implementation) | 16.70 | 37 | 0.0 | 0.0 | 16.70 |
| eager | 2 | Mirror V-exact, MLP + attention | 16.85 | 68 | 0.0 | 0.0 | 13.88 |
| eager | 2 | Mirror V-local, MLP + attention | 20.73 | 68 | 0.0 | 0.0 | 14.40 |
| eager | 2 | Mirror V-local, MLP only | 16.70 | 60 | 0.0 | 0.0 | 15.21 |
| eager | 2 | SVD at matched parameters | 18.30 | 75 | 0.0 | 0.0 | 13.57 |
| eager | 2 | MLP width pruning at matched parameters | 11.47 | 37 | 0.0 | 0.0 | 13.24 |
| eager | 4 | Mirror V-exact, MLP + attention | 15.62 | 96 | 0.0 | 0.0 | 10.99 |
| eager | 4 | Mirror V-local, MLP + attention | 12.97 | 96 | 0.0 | 0.0 | 12.18 |
| eager | 4 | Mirror V-local, MLP only | 14.00 | 80 | 0.0 | 0.0 | 13.73 |
| eager | 4 | SVD at matched parameters | 13.34 | 113 | 0.0 | 0.0 | 10.43 |
| eager | 4 | MLP width pruning at matched parameters | 11.66 | 37 | 0.0 | 0.0 | 9.70 |

## Converted sub-modules timed alone

Each converted block's attention and MLP, and the V-local stream rotation, timed alone on a bf16 input of 2 x 17 tokens x 64 (median of 2 runs after 1 warm-up runs, wall clock, perf_counter), against the same block of the dense model. Sums over the timed blocks (ms). Remainder: end-to-end gap minus the summed sub-module changes (norms, residual adds, the embedding and head, and fusion across module boundaries in compile mode, which isolated timing cannot see).

| mode | k | variant | blocks | MLP | MLP dense | MLP change | attention | attention dense | attention change | rotations | sum of changes | end-to-end gap | remainder |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eager | 2 | Mirror V-exact, MLP + attention | 0,1 | 5.40 | 3.31 | +2.09 | 6.76 | 3.74 | +3.02 | n/a | +5.11 | +4.22 | -0.88 |
| eager | 2 | Mirror V-local, MLP + attention | 0,1 | 8.60 | 3.31 | +5.30 | 9.51 | 3.74 | +5.77 | 1.57 | +12.63 | +9.09 | -3.54 |
| eager | 2 | Mirror V-local, MLP only | 0,1 | 5.51 | 3.31 | +2.20 | n/a | n/a | n/a | 0.85 | +3.05 | +13.76 | +10.71 |
| eager | 2 | SVD at matched parameters | 0,1 | 4.67 | 3.31 | +1.36 | 7.08 | 3.74 | +3.34 | n/a | +4.70 | +10.71 | +6.01 |
| eager | 2 | MLP width pruning at matched parameters | 0,1 | 1.66 | 3.31 | -1.65 | n/a | n/a | n/a | n/a | -1.65 | -4.83 | -3.19 |
| eager | 4 | Mirror V-exact, MLP + attention | 0,1,2,3 | 11.99 | 7.17 | +4.82 | 13.89 | 7.55 | +6.34 | n/a | +11.15 | +7.76 | -3.39 |
| eager | 4 | Mirror V-local, MLP + attention | 0,1,2,3 | 10.26 | 7.17 | +3.09 | 11.09 | 7.55 | +3.53 | 1.51 | +8.13 | -0.57 | -8.70 |
| eager | 4 | Mirror V-local, MLP only | 0,1,2,3 | 11.79 | 7.17 | +4.62 | n/a | n/a | n/a | 2.27 | +6.89 | -1.12 | -8.01 |
| eager | 4 | SVD at matched parameters | 0,1,2,3 | 9.91 | 7.17 | +2.74 | 11.76 | 7.55 | +4.21 | n/a | +6.95 | +0.36 | -6.59 |
| eager | 4 | MLP width pruning at matched parameters | 0,1,2,3 | 3.79 | 7.17 | -3.38 | n/a | n/a | n/a | n/a | -3.38 | -3.26 | +0.12 |

- eager, Mirror V-exact, MLP + attention, k = 2: converted MLPs +2.09 ms (50% of the gap), attention +3.02 ms (71% of the gap); end-to-end gap +4.22 ms.
- eager, Mirror V-local, MLP + attention, k = 2: converted MLPs +5.30 ms (58% of the gap), attention +5.77 ms (63% of the gap), rotations +1.57 ms (17% of the gap); end-to-end gap +9.09 ms.
- eager, Mirror V-local, MLP only, k = 2: converted MLPs +2.20 ms (16% of the gap), rotations +0.85 ms (6% of the gap); end-to-end gap +13.76 ms.
- eager, SVD at matched parameters, k = 2: converted MLPs +1.36 ms (13% of the gap), attention +3.34 ms (31% of the gap); end-to-end gap +10.71 ms.
- eager, MLP width pruning at matched parameters, k = 2: converted MLPs -1.65 ms; end-to-end gap -4.83 ms.
- eager, Mirror V-exact, MLP + attention, k = 4: converted MLPs +4.82 ms (62% of the gap), attention +6.34 ms (82% of the gap); end-to-end gap +7.76 ms.
- eager, Mirror V-local, MLP + attention, k = 4: converted MLPs +3.09 ms, attention +3.53 ms, rotations +1.51 ms; end-to-end gap -0.57 ms.
- eager, Mirror V-local, MLP only, k = 4: converted MLPs +4.62 ms, rotations +2.27 ms; end-to-end gap -1.12 ms.
- eager, SVD at matched parameters, k = 4: converted MLPs +2.74 ms (759% of the gap), attention +4.21 ms (1167% of the gap); end-to-end gap +0.36 ms.
- eager, MLP width pruning at matched parameters, k = 4: converted MLPs -3.38 ms; end-to-end gap -3.26 ms.

Time against analytic work, per part (sums over the timed blocks): time ratio = converted over dense time; work ratio = converted over dense executed multiply-adds per token (rotations: d x d against nothing in dense).

| mode | k | variant | part | time ratio | work ratio |
|---|---|---|---|---|---|
| eager | 2 | Mirror V-exact, MLP + attention | attention | 1.81 | 0.58 |
| eager | 2 | Mirror V-exact, MLP + attention | mlp | 1.63 | 0.61 |
| eager | 2 | Mirror V-local, MLP + attention | attention | 2.54 | 0.66 |
| eager | 2 | Mirror V-local, MLP + attention | mlp | 2.60 | 0.56 |
| eager | 2 | Mirror V-local, MLP + attention | rotation | n/a | n/a |
| eager | 2 | Mirror V-local, MLP only | mlp | 1.67 | 0.56 |
| eager | 2 | Mirror V-local, MLP only | rotation | n/a | n/a |
| eager | 2 | SVD at matched parameters | attention | 1.89 | 0.50 |
| eager | 2 | SVD at matched parameters | mlp | 1.41 | 0.59 |
| eager | 2 | MLP width pruning at matched parameters | mlp | 0.50 | 0.27 |
| eager | 4 | Mirror V-exact, MLP + attention | attention | 1.84 | 0.58 |
| eager | 4 | Mirror V-exact, MLP + attention | mlp | 1.67 | 0.60 |
| eager | 4 | Mirror V-local, MLP + attention | attention | 1.47 | 0.67 |
| eager | 4 | Mirror V-local, MLP + attention | mlp | 1.43 | 0.56 |
| eager | 4 | Mirror V-local, MLP + attention | rotation | n/a | n/a |
| eager | 4 | Mirror V-local, MLP only | mlp | 1.64 | 0.56 |
| eager | 4 | Mirror V-local, MLP only | rotation | n/a | n/a |
| eager | 4 | SVD at matched parameters | attention | 1.56 | 0.50 |
| eager | 4 | SVD at matched parameters | mlp | 1.38 | 0.59 |
| eager | 4 | MLP width pruning at matched parameters | mlp | 0.53 | 0.26 |

## Caveats

- One device and one session; the GPU is shared. nvidia-smi before: cpu; after: cpu. Other processes running at the same time would inflate latencies; kernel times are less affected than latency but are not immune (shared memory bandwidth and clocks).
- Latency and the profile come from different forwards of the same compiled model; the profiler adds CPU overhead, which does not change kernel durations but can change idle time. Idle can be negative if kernels ran slower under the profiler than in the timed runs.
- Categories come from kernel names. In eager mode the casts to and from float32 inside RMSNorm are copy kernels and count as memory movement (in dense and converted models alike); in compile mode they are fused into Triton kernels. A Triton GEMM template (max-autotune, not used here) would count as GEMM.
- Sub-modules timed alone do not see fusion with their neighbours (compile mode) or the cost of the norms and residual adds around them; their sum need not equal the end-to-end gap.
- torch.compile default mode, no CUDA graphs, plain PyTorch ops as in M2. This is a measurement of that implementation, not of what a fused kernel could do.
- One structure (the M1 fit seed); latency depends on the shapes, not the fitted values.

Files: m2_profile.csv, m2_submodules.csv, m2_triton.csv, m2_top_kernels.csv, top_kernels/, m2_profile_meta.json.
