# M2 profile: where the time goes in the converted models

Model `dinov2_vitb14_reg` (12 blocks, d 768, 261 tokens per image), batch 256, bf16. Structure `/workspace/m2run/runs/m1/m1_structure.npz`, operators `/workspace/m2run/runs/x2/perlayer_reg_s0/results/e1_operators.npz`. Git `6f2c9181a8`, seed 0. Device: NVIDIA GeForce RTX 5090, torch 2.11.0+cu128. Written 2026-11-09 16:23:22.

Method. The models are the ones `scripts/m2_latency.py` times, built by its own code (same structure and operator files, alignment 8, bf16). Per variant, k and mode: latency is the median of 100 forwards (CUDA events, after 20 warm-up forwards, no profiler attached); then 10 more forwards were recorded with torch.profiler and the time of every GPU kernel (including memcpy and memset) was summed and divided by the number of forwards. Kernels are sorted into categories by name (`classify_kernel` in `scripts/m2_profile.py`). Idle time is latency minus kernel time (the GPU waiting for work), so for every row the latency gap to dense equals the sum of the category changes plus the idle change. Nothing was optimised.

## Main findings (compile, batch 256)

- Mirror rows slower than dense: 6 of 6 (speed 0.63 to 0.88x of dense, analytic FLOP ratio 1.10 to 1.29x).
- GEMM kernels: 58.91 to 92.84 ms per forward in the Mirror rows against 55.37 ms for dense; Mirror rows spending more time in GEMMs than dense: 6 of 6, although they execute 8.8 to 23.1% fewer GEMM multiply-adds (analytic). Achieved GEMM throughput 104.0 to 161.3 TFLOP/s against 206.1 for dense; at the dense rate their GEMMs would take 42.57 to 50.49 ms.
- Non-GEMM, non-attention kernels (fused Triton, elementwise, normalisation, memory movement, other): 13.90 to 19.99 ms per forward against 12.85 ms for dense (change +1.05 to +7.14 ms).
- Attention kernels (token mixing, unchanged by the conversion): 6.82 to 9.07 ms against 6.78 ms for dense.
- Kernel launches per forward: 128 to 227 against 99 for dense; idle time (latency minus gpu kernel time): 0.28 to 2.09 ms against 0.68 ms.
- Largest single contributor to each slower Mirror row's gap: Mirror V-exact, MLP + attention k = 6: GEMM (+15.58 of 20.30 ms); Mirror V-local, MLP + attention k = 6: GEMM (+14.34 of 18.03 ms); Mirror V-local, MLP only k = 6: GEMM (+8.61 of 9.90 ms); Mirror V-exact, MLP + attention k = 12: fused Triton (+7.14 of 13.26 ms); Mirror V-local, MLP + attention k = 12: GEMM (+37.48 of 44.78 ms); Mirror V-local, MLP only k = 12: GEMM (+22.88 of 25.37 ms).
- SVD at matched parameters, k = 6: speed 0.99x of dense; GEMM 54.40 ms (-0.96), other kernels +2.53 ms.
- MLP width pruning at matched parameters, k = 6: speed 1.16x of dense; GEMM 46.25 ms (-9.11), other kernels -0.97 ms.
- SVD at matched parameters, k = 12: speed 0.89x of dense; GEMM 59.11 ms (+3.75), other kernels +4.46 ms.
- MLP width pruning at matched parameters, k = 12: speed 1.20x of dense; GEMM 42.99 ms (-12.37), other kernels -1.61 ms.

## Main findings (eager, batch 256)

- Mirror rows slower than dense: 6 of 6 (speed 0.69 to 0.90x of dense, analytic FLOP ratio 1.10 to 1.29x).
- GEMM kernels: 49.21 to 55.53 ms per forward in the Mirror rows against 55.36 ms for dense; Mirror rows spending more time in GEMMs than dense: 1 of 6, although they execute 8.8 to 23.1% fewer GEMM multiply-adds (analytic). Achieved GEMM throughput 177.4 to 191.5 TFLOP/s against 206.1 for dense; at the dense rate their GEMMs would take 42.57 to 50.49 ms.
- Non-GEMM, non-attention kernels (fused Triton, elementwise, normalisation, memory movement, other): 38.93 to 73.23 ms per forward against 27.56 ms for dense (change +11.37 to +45.67 ms).
- Attention kernels (token mixing, unchanged by the conversion): 6.88 to 6.97 ms against 6.82 ms for dense.
- Kernel launches per forward: 251 to 519 against 179 for dense; idle time (latency minus gpu kernel time): 0.20 to 1.43 ms against 0.24 ms.
- Largest single contributor to each slower Mirror row's gap: Mirror V-exact, MLP + attention k = 6: memory movement (+13.24 of 19.40 ms); Mirror V-local, MLP + attention k = 6: elementwise (+10.46 of 17.90 ms); Mirror V-local, MLP only k = 6: elementwise (+8.22 of 10.45 ms); Mirror V-exact, MLP + attention k = 12: memory movement (+28.22 of 39.85 ms); Mirror V-local, MLP + attention k = 12: memory movement (+20.85 of 37.39 ms); Mirror V-local, MLP only k = 12: elementwise (+12.28 of 20.98 ms).
- SVD at matched parameters, k = 6: speed 0.97x of dense; GEMM 53.54 ms (-1.82), other kernels +4.75 ms.
- MLP width pruning at matched parameters, k = 6: speed 1.15x of dense; GEMM 45.31 ms (-10.06), other kernels -2.18 ms.
- SVD at matched parameters, k = 12: speed 0.89x of dense; GEMM 55.83 ms (+0.46), other kernels +8.99 ms.
- MLP width pruning at matched parameters, k = 12: speed 1.20x of dense; GEMM 42.42 ms (-12.95), other kernels -3.36 ms.

## Per variant

Compile (mode used for dense: compile):

- Mirror V-exact, MLP + attention, k = 6: 95.99 ms per forward against 75.68 ms for dense (0.79x the dense speed; analytic FLOP ratio 1.24x). GPU kernel time 95.70 ms against 75.00 ms (+20.70 ms); by category (ms): GEMM +15.58, fused Triton +4.09, attention +1.03. Idle time (latency minus kernel time) -0.40 ms. Kernel launches per forward 165 against 99. GEMM throughput 128.2 against 206.1 TFLOP/s; at the dense model's throughput its GEMMs would take 44.14 ms, measured 70.95 ms.
- Mirror V-local, MLP + attention, k = 6: 93.72 ms per forward against 75.68 ms for dense (0.81x the dense speed; analytic FLOP ratio 1.16x). GPU kernel time 93.12 ms against 75.00 ms (+18.12 ms); by category (ms): GEMM +14.34, fused Triton +2.79, attention +0.99. Idle time (latency minus kernel time) -0.08 ms. Kernel launches per forward 140 against 99. GEMM throughput 140.0 against 206.1 TFLOP/s; at the dense model's throughput its GEMMs would take 47.35 ms, measured 69.70 ms.
- Mirror V-local, MLP only, k = 6: 85.59 ms per forward against 75.68 ms for dense (0.88x the dense speed; analytic FLOP ratio 1.10x). GPU kernel time 84.85 ms against 75.00 ms (+9.85 ms); by category (ms): GEMM +8.61, fused Triton +1.10, attention +0.14. Idle time (latency minus kernel time) +0.05 ms. Kernel launches per forward 128 against 99. GEMM throughput 161.3 against 206.1 TFLOP/s; at the dense model's throughput its GEMMs would take 50.08 ms, measured 63.97 ms.
- SVD at matched parameters, k = 6: 76.72 ms per forward against 75.68 ms for dense (0.99x the dense speed; analytic FLOP ratio 1.24x). GPU kernel time 76.56 ms against 75.00 ms (+1.56 ms); by category (ms): attention +2.46, fused Triton +0.06, GEMM -0.96. Idle time (latency minus kernel time) -0.53 ms. Kernel launches per forward 153 against 99. GEMM throughput 167.3 against 206.1 TFLOP/s; at the dense model's throughput its GEMMs would take 44.17 ms, measured 54.40 ms.
- MLP width pruning at matched parameters, k = 6: 65.06 ms per forward against 75.68 ms for dense (1.16x the dense speed; analytic FLOP ratio 1.24x). GPU kernel time 64.92 ms against 75.00 ms (-10.08 ms); by category (ms): memory movement +0.33, attention +0.16, fused Triton -1.46, GEMM -9.11. Idle time (latency minus kernel time) -0.54 ms. Kernel launches per forward 104 against 99. GEMM throughput 196.2 against 206.1 TFLOP/s; at the dense model's throughput its GEMMs would take 44.04 ms, measured 46.25 ms.
- Mirror V-exact, MLP + attention, k = 12: 88.94 ms per forward against 75.68 ms for dense (0.85x the dense speed; analytic FLOP ratio 1.29x). GPU kernel time 87.97 ms against 75.00 ms (+12.97 ms); by category (ms): fused Triton +7.14, GEMM +3.54, attention +2.29. Idle time (latency minus kernel time) +0.28 ms. Kernel launches per forward 227 against 99. GEMM throughput 148.9 against 206.1 TFLOP/s; at the dense model's throughput its GEMMs would take 42.57 ms, measured 58.91 ms.
- Mirror V-local, MLP + attention, k = 12: 120.46 ms per forward against 75.68 ms for dense (0.63x the dense speed; analytic FLOP ratio 1.18x). GPU kernel time 118.60 ms against 75.00 ms (+43.59 ms); by category (ms): GEMM +37.48, fused Triton +3.95, attention +2.17. Idle time (latency minus kernel time) +1.18 ms. Kernel launches per forward 182 against 99. GEMM throughput 104.0 against 206.1 TFLOP/s; at the dense model's throughput its GEMMs would take 46.84 ms, measured 92.84 ms.
- Mirror V-local, MLP only, k = 12: 101.06 ms per forward against 75.68 ms for dense (0.75x the dense speed; analytic FLOP ratio 1.10x). GPU kernel time 98.97 ms against 75.00 ms (+23.96 ms); by category (ms): GEMM +22.88, fused Triton +1.05. Idle time (latency minus kernel time) +1.41 ms. Kernel launches per forward 158 against 99. GEMM throughput 133.0 against 206.1 TFLOP/s; at the dense model's throughput its GEMMs would take 50.49 ms, measured 78.25 ms.
- SVD at matched parameters, k = 12: 84.83 ms per forward against 75.68 ms for dense (0.89x the dense speed; analytic FLOP ratio 1.33x). GPU kernel time 83.21 ms against 75.00 ms (+8.21 ms); by category (ms): attention +4.58, GEMM +3.75, fused Triton -0.12. Idle time (latency minus kernel time) +0.93 ms. Kernel launches per forward 203 against 99. GEMM throughput 143.3 against 206.1 TFLOP/s; at the dense model's throughput its GEMMs would take 41.10 ms, measured 59.11 ms.
- MLP width pruning at matched parameters, k = 12: 63.29 ms per forward against 75.68 ms for dense (1.20x the dense speed; analytic FLOP ratio 1.33x). GPU kernel time 61.02 ms against 75.00 ms (-13.98 ms); by category (ms): memory movement +1.23, attention -0.13, fused Triton -2.71, GEMM -12.37. Idle time (latency minus kernel time) +1.59 ms. Kernel launches per forward 104 against 99. GEMM throughput 195.6 against 206.1 TFLOP/s; at the dense model's throughput its GEMMs would take 40.82 ms, measured 42.99 ms.

Eager (mode used for dense: eager):

- Mirror V-exact, MLP + attention, k = 6: 109.39 ms per forward against 89.99 ms for dense (0.82x the dense speed; analytic FLOP ratio 1.24x). GPU kernel time 108.97 ms against 89.75 ms (+19.22 ms); by category (ms): memory movement +13.24, elementwise +10.53, attention +0.08, GEMM -4.61. Idle time (latency minus kernel time) +0.18 ms. Kernel launches per forward 354 against 179. GEMM throughput 179.2 against 206.1 TFLOP/s; at the dense model's throughput its GEMMs would take 44.14 ms, measured 50.75 ms.
- Mirror V-local, MLP + attention, k = 6: 107.89 ms per forward against 89.99 ms for dense (0.83x the dense speed; analytic FLOP ratio 1.16x). GPU kernel time 107.32 ms against 89.75 ms (+17.57 ms); by category (ms): elementwise +10.46, memory movement +9.89, attention +0.11, GEMM -2.87. Idle time (latency minus kernel time) +0.33 ms. Kernel launches per forward 295 against 179. GEMM throughput 185.9 against 206.1 TFLOP/s; at the dense model's throughput its GEMMs would take 47.35 ms, measured 52.50 ms.
- Mirror V-local, MLP only, k = 6: 100.45 ms per forward against 89.99 ms for dense (0.90x the dense speed; analytic FLOP ratio 1.10x). GPU kernel time 99.69 ms against 89.75 ms (+9.94 ms); by category (ms): elementwise +8.22, memory movement +3.18, attention +0.06, GEMM -1.49. Idle time (latency minus kernel time) +0.51 ms. Kernel launches per forward 251 against 179. GEMM throughput 191.5 against 206.1 TFLOP/s; at the dense model's throughput its GEMMs would take 50.07 ms, measured 53.88 ms.
- SVD at matched parameters, k = 6: 92.91 ms per forward against 89.99 ms for dense (0.97x the dense speed; analytic FLOP ratio 1.24x). GPU kernel time 92.68 ms against 89.75 ms (+2.93 ms); by category (ms): memory movement +2.42, elementwise +2.33, GEMM -1.82. Idle time (latency minus kernel time) -0.02 ms. Kernel launches per forward 239 against 179. GEMM throughput 170.0 against 206.1 TFLOP/s; at the dense model's throughput its GEMMs would take 44.17 ms, measured 53.54 ms.
- MLP width pruning at matched parameters, k = 6: 78.46 ms per forward against 89.99 ms for dense (1.15x the dense speed; analytic FLOP ratio 1.24x). GPU kernel time 77.51 ms against 89.75 ms (-12.24 ms); by category (ms): normalisation -0.05, memory movement -0.14, elementwise -2.02, GEMM -10.06. Idle time (latency minus kernel time) +0.71 ms. Kernel launches per forward 177 against 179. GEMM throughput 200.3 against 206.1 TFLOP/s; at the dense model's throughput its GEMMs would take 44.04 ms, measured 45.31 ms.
- Mirror V-exact, MLP + attention, k = 12: 129.84 ms per forward against 89.99 ms for dense (0.69x the dense speed; analytic FLOP ratio 1.29x). GPU kernel time 129.38 ms against 89.75 ms (+39.63 ms); by category (ms): memory movement +28.22, elementwise +17.45, attention +0.12, GEMM -6.16. Idle time (latency minus kernel time) +0.22 ms. Kernel launches per forward 519 against 179. GEMM throughput 178.3 against 206.1 TFLOP/s; at the dense model's throughput its GEMMs would take 42.57 ms, measured 49.21 ms.
- Mirror V-local, MLP + attention, k = 12: 127.38 ms per forward against 89.99 ms for dense (0.71x the dense speed; analytic FLOP ratio 1.18x). GPU kernel time 127.17 ms against 89.75 ms (+37.42 ms); by category (ms): memory movement +20.85, elementwise +17.38, attention +0.15, GEMM -0.95. Idle time (latency minus kernel time) -0.04 ms. Kernel launches per forward 407 against 179. GEMM throughput 177.4 against 206.1 TFLOP/s; at the dense model's throughput its GEMMs would take 46.84 ms, measured 54.42 ms.
- Mirror V-local, MLP only, k = 12: 110.98 ms per forward against 89.99 ms for dense (0.81x the dense speed; analytic FLOP ratio 1.10x). GPU kernel time 109.54 ms against 89.75 ms (+19.79 ms); by category (ms): elementwise +12.28, memory movement +7.34, GEMM +0.17, attention +0.07, normalisation -0.06. Idle time (latency minus kernel time) +1.19 ms. Kernel launches per forward 321 against 179. GEMM throughput 187.4 against 206.1 TFLOP/s; at the dense model's throughput its GEMMs would take 50.49 ms, measured 55.53 ms.
- SVD at matched parameters, k = 12: 100.57 ms per forward against 89.99 ms for dense (0.89x the dense speed; analytic FLOP ratio 1.33x). GPU kernel time 99.20 ms against 89.75 ms (+9.45 ms); by category (ms): memory movement +4.64, elementwise +4.63, GEMM +0.46, normalisation -0.07, attention -0.21. Idle time (latency minus kernel time) +1.13 ms. Kernel launches per forward 295 against 179. GEMM throughput 151.7 against 206.1 TFLOP/s; at the dense model's throughput its GEMMs would take 41.10 ms, measured 55.83 ms.
- MLP width pruning at matched parameters, k = 12: 75.23 ms per forward against 89.99 ms for dense (1.20x the dense speed; analytic FLOP ratio 1.33x). GPU kernel time 73.44 ms against 89.75 ms (-16.31 ms); by category (ms): attention -0.09, memory movement -0.47, elementwise -2.84, GEMM -12.95. Idle time (latency minus kernel time) +1.54 ms. Kernel launches per forward 175 against 179. GEMM throughput 198.3 against 206.1 TFLOP/s; at the dense model's throughput its GEMMs would take 40.81 ms, measured 42.42 ms.

## GPU kernel time per forward by category (compile, ms)

Share: category time over total kernel time. GPU busy: kernel time over latency. M2 clean: the latency `runs/m2_clean` reported for the same variant, k, batch and mode.

| k | variant | latency | M2 clean | speed vs dense | kernel time | GPU busy | launches | GEMM | attention | fused Triton | elementwise | normalisation | memory movement | other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | dense (folded original, same implementation) | 75.68 | 101.97 | 1.00 | 75.00 | 99% | 99 | 55.37 (74%) | 6.78 (9%) | 12.85 (17%) | 0.00 (0%) | 0.00 (0%) | 0.00 (0%) | 0.00 (0%) |
| 6 | Mirror V-exact, MLP + attention | 95.99 | 120.66 | 0.79 | 95.70 | 100% | 165 | 70.95 (74%) | 7.81 (8%) | 16.94 (18%) | 0.00 (0%) | 0.00 (0%) | 0.00 (0%) | 0.00 (0%) |
| 6 | Mirror V-local, MLP + attention | 93.72 | 119.59 | 0.81 | 93.12 | 99% | 140 | 69.70 (75%) | 7.77 (8%) | 15.64 (17%) | 0.00 (0%) | 0.00 (0%) | 0.00 (0%) | 0.00 (0%) |
| 6 | Mirror V-local, MLP only | 85.59 | 112.05 | 0.88 | 84.85 | 99% | 128 | 63.97 (75%) | 6.93 (8%) | 13.96 (16%) | 0.00 (0%) | 0.00 (0%) | 0.00 (0%) | 0.00 (0%) |
| 6 | SVD at matched parameters | 76.72 | 104.54 | 0.99 | 76.56 | 100% | 153 | 54.40 (71%) | 9.25 (12%) | 12.91 (17%) | 0.00 (0%) | 0.00 (0%) | 0.00 (0%) | 0.00 (0%) |
| 6 | MLP width pruning at matched parameters | 65.06 | 90.02 | 1.16 | 64.92 | 100% | 104 | 46.25 (71%) | 6.94 (11%) | 11.40 (18%) | 0.00 (0%) | 0.00 (0%) | 0.33 (1%) | 0.00 (0%) |
| 12 | Mirror V-exact, MLP + attention | 88.94 | 145.84 | 0.85 | 87.97 | 99% | 227 | 58.91 (67%) | 9.07 (10%) | 19.99 (23%) | 0.00 (0%) | 0.00 (0%) | 0.00 (0%) | 0.00 (0%) |
| 12 | Mirror V-local, MLP + attention | 120.46 | 145.46 | 0.63 | 118.60 | 98% | 182 | 92.84 (78%) | 8.95 (8%) | 16.80 (14%) | 0.00 (0%) | 0.00 (0%) | 0.00 (0%) | 0.00 (0%) |
| 12 | Mirror V-local, MLP only | 101.06 | 127.28 | 0.75 | 98.97 | 98% | 158 | 78.25 (79%) | 6.82 (7%) | 13.90 (14%) | 0.00 (0%) | 0.00 (0%) | 0.00 (0%) | 0.00 (0%) |
| 12 | SVD at matched parameters | 84.83 | 111.48 | 0.89 | 83.21 | 98% | 203 | 59.11 (71%) | 11.36 (14%) | 12.73 (15%) | 0.00 (0%) | 0.00 (0%) | 0.00 (0%) | 0.00 (0%) |
| 12 | MLP width pruning at matched parameters | 63.29 | 87.24 | 1.20 | 61.02 | 96% | 104 | 42.99 (70%) | 6.65 (11%) | 10.14 (17%) | 0.00 (0%) | 0.00 (0%) | 1.23 (2%) | 0.00 (0%) |

Change against dense (compile, ms per forward; the columns add up to the latency gap):

| k | variant | gap | GEMM | attention | fused Triton | elementwise | normalisation | memory movement | other | idle |
|---|---|---|---|---|---|---|---|---|---|---|
| 6 | Mirror V-exact, MLP + attention | +20.30 | +15.58 | +1.03 | +4.09 | +0.00 | +0.00 | +0.00 | -0.00 | -0.40 |
| 6 | Mirror V-local, MLP + attention | +18.03 | +14.34 | +0.99 | +2.79 | +0.00 | +0.00 | +0.00 | -0.00 | -0.08 |
| 6 | Mirror V-local, MLP only | +9.90 | +8.61 | +0.14 | +1.10 | +0.00 | +0.00 | +0.00 | -0.00 | +0.05 |
| 6 | SVD at matched parameters | +1.04 | -0.96 | +2.46 | +0.06 | +0.00 | +0.00 | +0.00 | -0.00 | -0.53 |
| 6 | MLP width pruning at matched parameters | -10.62 | -9.11 | +0.16 | -1.46 | +0.00 | +0.00 | +0.33 | -0.00 | -0.54 |
| 12 | Mirror V-exact, MLP + attention | +13.26 | +3.54 | +2.29 | +7.14 | +0.00 | +0.00 | +0.00 | -0.00 | +0.28 |
| 12 | Mirror V-local, MLP + attention | +44.78 | +37.48 | +2.17 | +3.95 | +0.00 | +0.00 | +0.00 | -0.00 | +1.18 |
| 12 | Mirror V-local, MLP only | +25.37 | +22.88 | +0.03 | +1.05 | +0.00 | +0.00 | +0.00 | -0.00 | +1.41 |
| 12 | SVD at matched parameters | +9.14 | +3.75 | +4.58 | -0.12 | +0.00 | +0.00 | +0.00 | -0.00 | +0.93 |
| 12 | MLP width pruning at matched parameters | -12.39 | -12.37 | -0.13 | -2.71 | +0.00 | +0.00 | +1.23 | -0.00 | +1.59 |

## GPU kernel time per forward by category (eager, ms)

Share: category time over total kernel time. GPU busy: kernel time over latency. M2 clean: the latency `runs/m2_clean` reported for the same variant, k, batch and mode.

| k | variant | latency | M2 clean | speed vs dense | kernel time | GPU busy | launches | GEMM | attention | fused Triton | elementwise | normalisation | memory movement | other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | dense (folded original, same implementation) | 89.99 | 125.77 | 1.00 | 89.75 | 100% | 179 | 55.36 (62%) | 6.82 (8%) | 0.00 (0%) | 10.78 (12%) | 6.44 (7%) | 10.34 (12%) | 0.00 (0%) |
| 6 | Mirror V-exact, MLP + attention | 109.39 | 169.41 | 0.82 | 108.97 | 100% | 354 | 50.75 (47%) | 6.90 (6%) | 0.00 (0%) | 21.30 (20%) | 6.43 (6%) | 23.59 (22%) | 0.00 (0%) |
| 6 | Mirror V-local, MLP + attention | 107.89 | 169.44 | 0.83 | 107.32 | 99% | 295 | 52.50 (49%) | 6.93 (6%) | 0.00 (0%) | 21.24 (20%) | 6.41 (6%) | 20.24 (19%) | 0.00 (0%) |
| 6 | Mirror V-local, MLP only | 100.45 | 145.99 | 0.90 | 99.69 | 99% | 251 | 53.88 (54%) | 6.88 (7%) | 0.00 (0%) | 19.00 (19%) | 6.41 (6%) | 13.52 (14%) | 0.00 (0%) |
| 6 | SVD at matched parameters | 92.91 | 129.87 | 0.97 | 92.68 | 100% | 239 | 53.54 (58%) | 6.79 (7%) | 0.00 (0%) | 13.11 (14%) | 6.47 (7%) | 12.77 (14%) | 0.00 (0%) |
| 6 | MLP width pruning at matched parameters | 78.46 | 112.07 | 1.15 | 77.51 | 99% | 177 | 45.31 (58%) | 6.85 (9%) | 0.00 (0%) | 8.76 (11%) | 6.39 (8%) | 10.20 (13%) | 0.00 (0%) |
| 12 | Mirror V-exact, MLP + attention | 129.84 | 233.16 | 0.69 | 129.38 | 100% | 519 | 49.21 (38%) | 6.94 (5%) | 0.00 (0%) | 28.23 (22%) | 6.44 (5%) | 38.56 (30%) | 0.00 (0%) |
| 12 | Mirror V-local, MLP + attention | 127.38 | 215.89 | 0.71 | 127.17 | 100% | 407 | 54.42 (43%) | 6.97 (5%) | 0.00 (0%) | 28.16 (22%) | 6.44 (5%) | 31.20 (25%) | 0.00 (0%) |
| 12 | Mirror V-local, MLP only | 110.98 | 172.15 | 0.81 | 109.54 | 99% | 321 | 55.53 (51%) | 6.89 (6%) | 0.00 (0%) | 23.06 (21%) | 6.38 (6%) | 17.69 (16%) | 0.00 (0%) |
| 12 | SVD at matched parameters | 100.57 | 139.84 | 0.89 | 99.20 | 99% | 295 | 55.83 (56%) | 6.62 (7%) | 0.00 (0%) | 15.40 (16%) | 6.37 (6%) | 14.99 (15%) | 0.00 (0%) |
| 12 | MLP width pruning at matched parameters | 75.23 | 106.67 | 1.20 | 73.44 | 98% | 175 | 42.42 (58%) | 6.73 (9%) | 0.00 (0%) | 7.94 (11%) | 6.48 (9%) | 9.88 (13%) | 0.00 (0%) |

Change against dense (eager, ms per forward; the columns add up to the latency gap):

| k | variant | gap | GEMM | attention | fused Triton | elementwise | normalisation | memory movement | other | idle |
|---|---|---|---|---|---|---|---|---|---|---|
| 6 | Mirror V-exact, MLP + attention | +19.40 | -4.61 | +0.08 | +0.00 | +10.53 | -0.01 | +13.24 | -0.00 | +0.18 |
| 6 | Mirror V-local, MLP + attention | +17.90 | -2.87 | +0.11 | +0.00 | +10.46 | -0.03 | +9.89 | -0.00 | +0.33 |
| 6 | Mirror V-local, MLP only | +10.45 | -1.49 | +0.06 | +0.00 | +8.22 | -0.03 | +3.18 | -0.00 | +0.51 |
| 6 | SVD at matched parameters | +2.92 | -1.82 | -0.03 | +0.00 | +2.33 | +0.03 | +2.42 | +0.00 | -0.02 |
| 6 | MLP width pruning at matched parameters | -11.53 | -10.06 | +0.03 | +0.00 | -2.02 | -0.05 | -0.14 | -0.00 | +0.71 |
| 12 | Mirror V-exact, MLP + attention | +39.85 | -6.16 | +0.12 | +0.00 | +17.45 | +0.00 | +28.22 | +0.00 | +0.22 |
| 12 | Mirror V-local, MLP + attention | +37.39 | -0.95 | +0.15 | +0.00 | +17.38 | -0.00 | +20.85 | +0.00 | -0.04 |
| 12 | Mirror V-local, MLP only | +20.98 | +0.17 | +0.07 | +0.00 | +12.28 | -0.06 | +7.34 | +0.00 | +1.19 |
| 12 | SVD at matched parameters | +10.58 | +0.46 | -0.21 | +0.00 | +4.63 | -0.07 | +4.64 | +0.00 | +1.13 |
| 12 | MLP width pruning at matched parameters | -14.76 | -12.95 | -0.09 | +0.00 | -2.84 | +0.04 | -0.47 | +0.00 | +1.54 |

## GEMM efficiency

Analytic GEMM work: multiply-adds executed by matrix products per forward (patch embedding, every weight matrix with its alignment padding, rotations; token mixing excluded, it runs inside the attention kernel), from `mirror.efficient.fast_vit_cost`. Throughput: 2 x that work over the measured GEMM kernel time. At dense rate: the GEMM time this work would take at the dense model's measured throughput.

| mode | k | variant | GEMM ms | GEMM launches | analytic GEMM GFLOP | TFLOP/s | at dense rate (ms) |
|---|---|---|---|---|---|---|---|
| compile | 0 | dense (folded original, same implementation) | 55.37 | 49 | 11409.2 | 206.1 | 55.37 |
| compile | 6 | Mirror V-exact, MLP + attention | 70.95 | 74 | 9095.5 | 128.2 | 44.14 |
| compile | 6 | Mirror V-local, MLP + attention | 69.70 | 74 | 9757.0 | 140.0 | 47.35 |
| compile | 6 | Mirror V-local, MLP only | 63.97 | 67 | 10318.9 | 161.3 | 50.08 |
| compile | 6 | SVD at matched parameters | 54.40 | 85 | 9102.9 | 167.3 | 44.17 |
| compile | 6 | MLP width pruning at matched parameters | 46.25 | 49 | 9075.8 | 196.2 | 44.04 |
| compile | 12 | Mirror V-exact, MLP + attention | 58.91 | 97 | 8771.8 | 148.9 | 42.57 |
| compile | 12 | Mirror V-local, MLP + attention | 92.84 | 96 | 9651.9 | 104.0 | 46.84 |
| compile | 12 | Mirror V-local, MLP only | 78.25 | 84 | 10404.7 | 133.0 | 50.49 |
| compile | 12 | SVD at matched parameters | 59.11 | 119 | 8469.1 | 143.3 | 41.10 |
| compile | 12 | MLP width pruning at matched parameters | 42.99 | 47 | 8410.8 | 195.6 | 40.82 |
| eager | 0 | dense (folded original, same implementation) | 55.36 | 49 | 11409.2 | 206.1 | 55.36 |
| eager | 6 | Mirror V-exact, MLP + attention | 50.75 | 74 | 9095.5 | 179.2 | 44.14 |
| eager | 6 | Mirror V-local, MLP + attention | 52.50 | 74 | 9757.0 | 185.9 | 47.35 |
| eager | 6 | Mirror V-local, MLP only | 53.88 | 68 | 10318.9 | 191.5 | 50.07 |
| eager | 6 | SVD at matched parameters | 53.54 | 85 | 9102.9 | 170.0 | 44.17 |
| eager | 6 | MLP width pruning at matched parameters | 45.31 | 49 | 9075.8 | 200.3 | 44.04 |
| eager | 12 | Mirror V-exact, MLP + attention | 49.21 | 98 | 8771.8 | 178.3 | 42.57 |
| eager | 12 | Mirror V-local, MLP + attention | 54.42 | 98 | 9651.9 | 177.4 | 46.84 |
| eager | 12 | Mirror V-local, MLP only | 55.53 | 85 | 10404.7 | 187.4 | 50.49 |
| eager | 12 | SVD at matched parameters | 55.83 | 120 | 8469.1 | 151.7 | 41.10 |
| eager | 12 | MLP width pruning at matched parameters | 42.42 | 48 | 8410.8 | 198.3 | 40.81 |

## Fused Triton kernels by what they fuse (compile)

From the operations Inductor names in each kernel (e.g. `triton_poi_fused_add_cat_gelu_sub`): norm = mean, pow or rsqrt; cat/gather/pad = cat, index, gather, scatter, pad, clone, permute, slice or select; activation = GELU and similar; pointwise = none of these (adds, multiplies, dtype casts). ms per forward (launches).

| mode | k | variant | activation | cat/gather/pad | cat/gather/pad + activation | norm | norm + cat/gather/pad | pointwise |
|---|---|---|---|---|---|---|---|---|
| compile | 0 | dense (folded original, same implementation) | 6.28 (12) | 0 | 0 | 6.26 (23) | 0.32 (2) | 0 |
| compile | 6 | Mirror V-exact, MLP + attention | 3.14 (6) | 2.93 (28) | 4.69 (16) | 2.99 (11) | 3.18 (14) | 0.00 (1) |
| compile | 6 | Mirror V-local, MLP + attention | 3.14 (6) | 1.64 (4) | 4.67 (16) | 2.72 (10) | 3.47 (15) | 0.00 (1) |
| compile | 6 | Mirror V-local, MLP only | 3.14 (6) | 0 | 4.70 (16) | 2.64 (10) | 3.47 (15) | 0.00 (1) |
| compile | 6 | SVD at matched parameters | 6.29 (12) | 0 | 0 | 6.30 (23) | 0.32 (2) | 0 |
| compile | 6 | MLP width pruning at matched parameters | 4.58 (14) | 0 | 0 | 6.50 (23) | 0.32 (2) | 0 |
| compile | 12 | Mirror V-exact, MLP + attention | 0 | 5.67 (53) | 8.68 (34) | 0 | 5.64 (25) | 0.00 (1) |
| compile | 12 | Mirror V-local, MLP + attention | 0 | 2.79 (7) | 8.22 (34) | 0.23 (1) | 5.57 (26) | 0.00 (1) |
| compile | 12 | Mirror V-local, MLP only | 0 | 0 | 8.16 (34) | 0.22 (1) | 5.52 (25) | 0.00 (1) |
| compile | 12 | SVD at matched parameters | 6.19 (12) | 0 | 0 | 6.26 (23) | 0.29 (2) | 0 |
| compile | 12 | MLP width pruning at matched parameters | 3.91 (13) | 0 | 0 | 5.94 (22) | 0.29 (2) | 0 |

## Converted sub-modules timed alone

Each converted block's attention and MLP, and the V-local stream rotation, timed alone on a bf16 input of 256 x 261 tokens x 768 (median of 50 runs after 10 warm-up runs, CUDA events), against the same block of the dense model. Sums over the timed blocks (ms). Remainder: end-to-end gap minus the summed sub-module changes (norms, residual adds, the embedding and head, and fusion across module boundaries in compile mode, which isolated timing cannot see).

| mode | k | variant | blocks | MLP | MLP dense | MLP change | attention | attention dense | attention change | rotations | sum of changes | end-to-end gap | remainder |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| compile | 6 | Mirror V-exact, MLP + attention | 0,1,2,3,4,5 | 30.90 | 20.67 | +10.23 | 23.77 | 12.42 | +11.34 | n/a | +21.58 | +20.30 | -1.27 |
| eager | 6 | Mirror V-exact, MLP + attention | 0,1,2,3,4,5 | 29.06 | 21.34 | +7.73 | 24.06 | 12.90 | +11.15 | n/a | +18.88 | +19.40 | +0.52 |
| compile | 6 | Mirror V-local, MLP + attention | 0,1,2,3,4,5 | 27.88 | 20.67 | +7.21 | 20.05 | 12.42 | +7.63 | 2.53 | +17.37 | +18.03 | +0.66 |
| eager | 6 | Mirror V-local, MLP + attention | 0,1,2,3,4,5 | 28.39 | 21.34 | +7.05 | 20.49 | 12.90 | +7.59 | 2.46 | +17.10 | +17.90 | +0.80 |
| compile | 6 | Mirror V-local, MLP only | 0,1,2,3,4,5 | 27.83 | 20.67 | +7.16 | n/a | n/a | n/a | 2.54 | +9.69 | +9.90 | +0.21 |
| eager | 6 | Mirror V-local, MLP only | 0,1,2,3,4,5 | 28.34 | 21.34 | +7.01 | n/a | n/a | n/a | 2.46 | +9.47 | +10.45 | +0.99 |
| compile | 6 | SVD at matched parameters | 0,1,2,3,4,5 | 16.80 | 20.67 | -3.86 | 16.55 | 12.42 | +4.13 | n/a | +0.27 | +1.04 | +0.77 |
| eager | 6 | SVD at matched parameters | 0,1,2,3,4,5 | 17.20 | 21.34 | -4.14 | 19.40 | 12.90 | +6.50 | n/a | +2.36 | +2.92 | +0.56 |
| compile | 6 | MLP width pruning at matched parameters | 0,1,2,3,4,5 | 9.20 | 20.67 | -11.47 | n/a | n/a | n/a | n/a | -11.47 | -10.62 | +0.85 |
| eager | 6 | MLP width pruning at matched parameters | 0,1,2,3,4,5 | 8.87 | 21.34 | -12.47 | n/a | n/a | n/a | n/a | -12.47 | -11.53 | +0.94 |
| compile | 12 | Mirror V-exact, MLP + attention | 0,1,2,3,4,5,6,7,8,9,10,11 | 45.33 | 41.34 | +3.98 | 34.24 | 24.87 | +9.38 | n/a | +13.36 | +13.26 | -0.11 |
| eager | 12 | Mirror V-exact, MLP + attention | 0,1,2,3,4,5,6,7,8,9,10,11 | 58.17 | 42.74 | +15.42 | 49.41 | 25.90 | +23.52 | n/a | +38.94 | +39.85 | +0.91 |
| compile | 12 | Mirror V-local, MLP + attention | 0,1,2,3,4,5,6,7,8,9,10,11 | 62.33 | 41.34 | +20.99 | 43.45 | 24.87 | +18.58 | 5.07 | +44.65 | +44.78 | +0.13 |
| eager | 12 | Mirror V-local, MLP + attention | 0,1,2,3,4,5,6,7,8,9,10,11 | 57.87 | 42.74 | +15.13 | 42.43 | 25.90 | +16.53 | 4.95 | +36.61 | +37.39 | +0.78 |
| compile | 12 | Mirror V-local, MLP only | 0,1,2,3,4,5,6,7,8,9,10,11 | 62.35 | 41.34 | +21.00 | n/a | n/a | n/a | 5.09 | +26.09 | +25.37 | -0.72 |
| eager | 12 | Mirror V-local, MLP only | 0,1,2,3,4,5,6,7,8,9,10,11 | 57.88 | 42.74 | +15.14 | n/a | n/a | n/a | 4.95 | +20.09 | +20.98 | +0.90 |
| compile | 12 | SVD at matched parameters | 0,1,2,3,4,5,6,7,8,9,10,11 | 38.13 | 41.34 | -3.21 | 36.16 | 24.87 | +11.29 | n/a | +8.08 | +9.14 | +1.06 |
| eager | 12 | SVD at matched parameters | 0,1,2,3,4,5,6,7,8,9,10,11 | 39.21 | 42.74 | -3.53 | 39.59 | 25.90 | +13.70 | n/a | +10.16 | +10.58 | +0.42 |
| compile | 12 | MLP width pruning at matched parameters | 0,1,2,3,4,5,6,7,8,9,10,11 | 28.51 | 41.34 | -12.83 | n/a | n/a | n/a | n/a | -12.83 | -12.39 | +0.44 |
| eager | 12 | MLP width pruning at matched parameters | 0,1,2,3,4,5,6,7,8,9,10,11 | 27.25 | 42.74 | -15.49 | n/a | n/a | n/a | n/a | -15.49 | -14.76 | +0.73 |

- compile, Mirror V-exact, MLP + attention, k = 6: converted MLPs +10.23 ms (50% of the gap), attention +11.34 ms (56% of the gap); end-to-end gap +20.30 ms.
- eager, Mirror V-exact, MLP + attention, k = 6: converted MLPs +7.73 ms (40% of the gap), attention +11.15 ms (57% of the gap); end-to-end gap +19.40 ms.
- compile, Mirror V-local, MLP + attention, k = 6: converted MLPs +7.21 ms (40% of the gap), attention +7.63 ms (42% of the gap), rotations +2.53 ms (14% of the gap); end-to-end gap +18.03 ms.
- eager, Mirror V-local, MLP + attention, k = 6: converted MLPs +7.05 ms (39% of the gap), attention +7.59 ms (42% of the gap), rotations +2.46 ms (14% of the gap); end-to-end gap +17.90 ms.
- compile, Mirror V-local, MLP only, k = 6: converted MLPs +7.16 ms (72% of the gap), rotations +2.54 ms (26% of the gap); end-to-end gap +9.90 ms.
- eager, Mirror V-local, MLP only, k = 6: converted MLPs +7.01 ms (67% of the gap), rotations +2.46 ms (24% of the gap); end-to-end gap +10.45 ms.
- compile, SVD at matched parameters, k = 6: converted MLPs -3.86 ms (-373% of the gap), attention +4.13 ms (398% of the gap); end-to-end gap +1.04 ms.
- eager, SVD at matched parameters, k = 6: converted MLPs -4.14 ms (-142% of the gap), attention +6.50 ms (223% of the gap); end-to-end gap +2.92 ms.
- compile, MLP width pruning at matched parameters, k = 6: converted MLPs -11.47 ms; end-to-end gap -10.62 ms.
- eager, MLP width pruning at matched parameters, k = 6: converted MLPs -12.47 ms; end-to-end gap -11.53 ms.
- compile, Mirror V-exact, MLP + attention, k = 12: converted MLPs +3.98 ms (30% of the gap), attention +9.38 ms (71% of the gap); end-to-end gap +13.26 ms.
- eager, Mirror V-exact, MLP + attention, k = 12: converted MLPs +15.42 ms (39% of the gap), attention +23.52 ms (59% of the gap); end-to-end gap +39.85 ms.
- compile, Mirror V-local, MLP + attention, k = 12: converted MLPs +20.99 ms (47% of the gap), attention +18.58 ms (42% of the gap), rotations +5.07 ms (11% of the gap); end-to-end gap +44.78 ms.
- eager, Mirror V-local, MLP + attention, k = 12: converted MLPs +15.13 ms (40% of the gap), attention +16.53 ms (44% of the gap), rotations +4.95 ms (13% of the gap); end-to-end gap +37.39 ms.
- compile, Mirror V-local, MLP only, k = 12: converted MLPs +21.00 ms (83% of the gap), rotations +5.09 ms (20% of the gap); end-to-end gap +25.37 ms.
- eager, Mirror V-local, MLP only, k = 12: converted MLPs +15.14 ms (72% of the gap), rotations +4.95 ms (24% of the gap); end-to-end gap +20.98 ms.
- compile, SVD at matched parameters, k = 12: converted MLPs -3.21 ms (-35% of the gap), attention +11.29 ms (124% of the gap); end-to-end gap +9.14 ms.
- eager, SVD at matched parameters, k = 12: converted MLPs -3.53 ms (-33% of the gap), attention +13.70 ms (129% of the gap); end-to-end gap +10.58 ms.
- compile, MLP width pruning at matched parameters, k = 12: converted MLPs -12.83 ms; end-to-end gap -12.39 ms.
- eager, MLP width pruning at matched parameters, k = 12: converted MLPs -15.49 ms; end-to-end gap -14.76 ms.

Time against analytic work, per part (sums over the timed blocks): time ratio = converted over dense time; work ratio = converted over dense executed multiply-adds per token (rotations: d x d against nothing in dense).

| mode | k | variant | part | time ratio | work ratio |
|---|---|---|---|---|---|
| compile | 6 | Mirror V-exact, MLP + attention | attention | 1.91 | 0.61 |
| compile | 6 | Mirror V-exact, MLP + attention | mlp | 1.50 | 0.58 |
| eager | 6 | Mirror V-exact, MLP + attention | attention | 1.86 | 0.61 |
| eager | 6 | Mirror V-exact, MLP + attention | mlp | 1.36 | 0.58 |
| compile | 6 | Mirror V-local, MLP + attention | attention | 1.61 | 0.70 |
| compile | 6 | Mirror V-local, MLP + attention | mlp | 1.35 | 0.59 |
| compile | 6 | Mirror V-local, MLP + attention | rotation | n/a | n/a |
| eager | 6 | Mirror V-local, MLP + attention | attention | 1.59 | 0.70 |
| eager | 6 | Mirror V-local, MLP + attention | mlp | 1.33 | 0.59 |
| eager | 6 | Mirror V-local, MLP + attention | rotation | n/a | n/a |
| compile | 6 | Mirror V-local, MLP only | mlp | 1.35 | 0.59 |
| compile | 6 | Mirror V-local, MLP only | rotation | n/a | n/a |
| eager | 6 | Mirror V-local, MLP only | mlp | 1.33 | 0.59 |
| eager | 6 | Mirror V-local, MLP only | rotation | n/a | n/a |
| compile | 6 | SVD at matched parameters | attention | 1.33 | 0.61 |
| compile | 6 | SVD at matched parameters | mlp | 0.81 | 0.59 |
| eager | 6 | SVD at matched parameters | attention | 1.50 | 0.61 |
| eager | 6 | SVD at matched parameters | mlp | 0.81 | 0.59 |
| compile | 6 | MLP width pruning at matched parameters | mlp | 0.45 | 0.38 |
| eager | 6 | MLP width pruning at matched parameters | mlp | 0.42 | 0.38 |
| compile | 12 | Mirror V-exact, MLP + attention | attention | 1.38 | 0.78 |
| compile | 12 | Mirror V-exact, MLP + attention | mlp | 1.10 | 0.76 |
| eager | 12 | Mirror V-exact, MLP + attention | attention | 1.91 | 0.78 |
| eager | 12 | Mirror V-exact, MLP + attention | mlp | 1.36 | 0.76 |
| compile | 12 | Mirror V-local, MLP + attention | attention | 1.75 | 0.80 |
| compile | 12 | Mirror V-local, MLP + attention | mlp | 1.51 | 0.74 |
| compile | 12 | Mirror V-local, MLP + attention | rotation | n/a | n/a |
| eager | 12 | Mirror V-local, MLP + attention | attention | 1.64 | 0.80 |
| eager | 12 | Mirror V-local, MLP + attention | mlp | 1.35 | 0.74 |
| eager | 12 | Mirror V-local, MLP + attention | rotation | n/a | n/a |
| compile | 12 | Mirror V-local, MLP only | mlp | 1.51 | 0.74 |
| compile | 12 | Mirror V-local, MLP only | rotation | n/a | n/a |
| eager | 12 | Mirror V-local, MLP only | mlp | 1.35 | 0.74 |
| eager | 12 | Mirror V-local, MLP only | rotation | n/a | n/a |
| compile | 12 | SVD at matched parameters | attention | 1.45 | 0.74 |
| compile | 12 | SVD at matched parameters | mlp | 0.92 | 0.74 |
| eager | 12 | SVD at matched parameters | attention | 1.53 | 0.74 |
| eager | 12 | SVD at matched parameters | mlp | 0.92 | 0.74 |
| compile | 12 | MLP width pruning at matched parameters | mlp | 0.69 | 0.60 |
| eager | 12 | MLP width pruning at matched parameters | mlp | 0.64 | 0.60 |

## Caveats

- One device and one session; the GPU is shared. nvidia-smi before: NVIDIA GeForce RTX 5090, 0 %, 4 MiB, 32607 MiB, 195 MHz; after: NVIDIA GeForce RTX 5090, 99 %, 2236 MiB, 32607 MiB, 2647 MHz. Other processes running at the same time would inflate latencies; kernel times are less affected than latency but are not immune (shared memory bandwidth and clocks).
- Latency and the profile come from different forwards of the same compiled model; the profiler adds CPU overhead, which does not change kernel durations but can change idle time. Idle can be negative if kernels ran slower under the profiler than in the timed runs.
- Categories come from kernel names. In eager mode the casts to and from float32 inside RMSNorm are copy kernels and count as memory movement (in dense and converted models alike); in compile mode they are fused into Triton kernels. A Triton GEMM template (max-autotune, not used here) would count as GEMM.
- Sub-modules timed alone do not see fusion with their neighbours (compile mode) or the cost of the norms and residual adds around them; their sum need not equal the end-to-end gap.
- torch.compile default mode, no CUDA graphs, plain PyTorch ops as in M2. This is a measurement of that implementation, not of what a fused kernel could do.
- One structure (the M1 fit seed); latency depends on the shapes, not the fitted values.
- Self-check: summed kernel time over the span from the first kernel's start to the last one's end is at most 0.998 over all rows (above 1 would mean overlapping kernels or double counting; record_function ranges were excluded: 0 at most per row).

Files: m2_profile.csv, m2_submodules.csv, m2_triton.csv, m2_top_kernels.csv, top_kernels/, m2_profile_meta.json.
