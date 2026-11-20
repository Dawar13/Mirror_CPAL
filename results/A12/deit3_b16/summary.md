# A12 systems table

bf16 weights and inputs, torch.compile (default), median of 100 runs after 20 warm-up runs; parameters as the method counts them and as stored; memory of the stored entries in bf16 (MB = 10^6 bytes).

| Model | Params (method) | Stored entries | MB (bf16) | GFLOPs | img/s b1 | img/s b64 | img/s b256 | ms b1 | peak GB |
|---|---|---|---|---|---|---|---|---|---|
| original | 86,585,320 | 86,585,320 | 173.2 | 35.13 | 497.3 | 10,627.2 | 11,000.7 | 2.011 | 0.74 |
| mirror_dense | 72,212,872 | 87,138,280 | 174.3 | 29.46 | 456.9 | 10,628.0 | 10,992.4 | 2.189 | 0.74 |
| mirror_even_odd | 72,212,872 | 72,491,184 | 145.0 | 29.46 | 263.1 | 7,973.5 | 8,689.6 | 3.802 | 0.93 |
| svdllm | 72,212,968 | 72,212,968 | 144.4 | 29.46 | 220.6 | 5,794.5 | 9,881.2 | 4.532 | 0.82 |

## Where the time goes (torch.profiler)

- mirror_even_odd: rest 40.8%, butterflies and permutes 40.8%, attention 13.2%, GEMM 5.1% (total 28.595 ms per forward, cuda kernel time (torch.profiler))
- original: rest 56.9%, butterflies and permutes 25.9%, attention 10.1%, GEMM 7.1% (total 22.543 ms per forward, cuda kernel time (torch.profiler))

## GEMM shapes (each timed alone)

- mirror_even_odd: M 50432, N 8, K 48 (x1 per forward; N and K multiples of 8: True and True): 1.36 TFLOP/s
- mirror_even_odd: M 50432, N 48, K 8 (x1 per forward; N and K multiples of 8: True and True): 1.38 TFLOP/s
- mirror_even_odd: M 50432, N 24, K 48 (x1 per forward; N and K multiples of 8: True and True): 4.09 TFLOP/s
- mirror_even_odd: M 50432, N 48, K 48 (x1 per forward; N and K multiples of 8: True and True): 8.24 TFLOP/s
- mirror_even_odd: M 50432, N 56, K 48 (x1 per forward; N and K multiples of 8: True and True): 9.61 TFLOP/s
- mirror_even_odd: M 50432, N 48, K 56 (x1 per forward; N and K multiples of 8: True and True): 9.73 TFLOP/s
- mirror_even_odd: M 50432, N 64, K 48 (x1 per forward; N and K multiples of 8: True and True): 11.00 TFLOP/s
- mirror_even_odd: M 50432, N 48, K 112 (x1 per forward; N and K multiples of 8: True and True): 18.99 TFLOP/s
- mirror_even_odd: M 50432, N 160, K 48 (x1 per forward; N and K multiples of 8: True and True): 27.63 TFLOP/s
- mirror_even_odd: M 50432, N 176, K 48 (x1 per forward; N and K multiples of 8: True and True): 30.12 TFLOP/s
- mirror_even_odd: M 50432, N 48, K 176 (x1 per forward; N and K multiples of 8: True and True): 30.29 TFLOP/s
- mirror_even_odd: M 50432, N 48, K 184 (x1 per forward; N and K multiples of 8: True and True): 31.42 TFLOP/s

## Alignment of the even/odd form

- align 1, batch 64: 3,954.2 img/s (16.186 ms)
- align 1, batch 256: 4,603.6 img/s (55.609 ms)
- align 8, batch 64: 7,965.3 img/s (8.035 ms)
- align 8, batch 256: 8,651.6 img/s (29.590 ms)

Files: systems_rows.csv, systems_table.csv, profile.json, gemm.csv, align.csv, align_gemm.csv, metrics.json, meta.json.
