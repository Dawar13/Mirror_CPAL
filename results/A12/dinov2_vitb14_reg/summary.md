# A12 systems table

bf16 weights and inputs, torch.compile (default), median of 100 runs after 20 warm-up runs; parameters as the method counts them and as stored; memory of the stored entries in bf16 (MB = 10^6 bytes).

| Model | Params (method) | Stored entries | MB (bf16) | GFLOPs | img/s b1 | img/s b64 | img/s b256 | ms b1 | peak GB |
|---|---|---|---|---|---|---|---|---|---|
| original | 86,583,552 | 86,583,552 | 173.2 | 47.08 | 482.9 | 7,981.9 | 8,105.6 | 2.071 | 0.90 |
| mirror_dense | 66,203,360 | 86,281,728 | 172.6 | 36.44 | 500.5 | 7,966.8 | 8,204.7 | 1.998 | 1.19 |
| mirror_even_odd | 66,203,360 | 66,460,104 | 132.9 | 36.44 | 266.2 | 5,841.8 | 6,213.0 | 3.756 | 1.22 |
| svdllm | 66,203,904 | 65,349,120 | 130.7 | 36.44 | 231.2 | 4,856.0 | 7,674.9 | 4.325 | 0.98 |
| iosvd | 66,204,672 | 73,607,424 | 147.2 | 36.44 | 221.4 | 5,357.2 | 7,169.6 | 4.516 | 0.94 |
| flap | 66,203,904 | 65,333,740 | 130.7 | 36.44 | 257.3 | 6,617.6 | 7,185.5 | 3.886 | 1.14 |
| blockdrop | 86,583,552 | 64,460,544 | 128.9 | 47.08 | 594.5 | 10,435.3 | 10,647.4 | 1.682 | 1.15 |
| flipavg_original | 86,583,552 | 86,583,552 | 173.2 | 94.16 | 267.2 | 3,984.0 | 4,013.0 | 3.742 | 1.29 |
| svdllm_f040 | 34,633,728 | 33,778,944 | 67.6 | 19.96 | 260.7 | 5,046.1 | 9,782.1 | 3.836 | 0.88 |
| svdllm_f050 | 43,292,160 | 42,437,376 | 84.9 | 24.48 | 275.0 | 6,233.0 | 9,336.4 | 3.637 | 0.88 |
| svdllm_f065 | 56,282,112 | 55,427,328 | 110.9 | 31.26 | 253.1 | 4,280.4 | 8,270.9 | 3.952 | 0.93 |
| dinov2_vits14_reg | 22,058,112 | 22,058,112 | 44.1 | 12.46 | 485.3 | 17,794.4 | 20,717.6 | 2.061 | 0.45 |
| flipavg_svdllm_f040 | 34,633,728 | 33,778,944 | 67.6 | 39.92 | 134.5 | 2,548.4 | 4,863.9 | 7.436 | 1.24 |
| flipavg_svdllm_f050 | 43,292,160 | 42,437,376 | 84.9 | 48.96 | 143.9 | 3,159.3 | 4,672.0 | 6.948 | 1.27 |
| flipavg_svdllm_f065 | 56,282,112 | 55,427,328 | 110.9 | 62.52 | 134.9 | 2,163.0 | 4,086.0 | 7.414 | 1.31 |
| flipavg_dinov2_vits14_reg | 22,058,112 | 22,058,112 | 44.1 | 24.91 | 251.7 | 9,180.5 | 10,270.4 | 3.973 | 0.64 |

## Where the time goes (torch.profiler)

- mirror_even_odd: butterflies and permutes 47.1%, rest 35.2%, attention 14.2%, GEMM 3.5% (total 39.983 ms per forward, cuda kernel time (torch.profiler))
- original: rest 54.8%, butterflies and permutes 25.4%, attention 14.8%, GEMM 5.0% (total 30.721 ms per forward, cuda kernel time (torch.profiler))

## GEMM shapes (each timed alone)

- mirror_even_odd: M 66816, N 72, K 8 (x2 per forward; N and K multiples of 8: True and True): 2.81 TFLOP/s
- mirror_even_odd: M 66816, N 24, K 72 (x2 per forward; N and K multiples of 8: True and True): 8.29 TFLOP/s
- mirror_even_odd: M 66816, N 72, K 24 (x2 per forward; N and K multiples of 8: True and True): 8.32 TFLOP/s
- mirror_even_odd: M 66816, N 40, K 72 (x1 per forward; N and K multiples of 8: True and True): 13.86 TFLOP/s
- mirror_even_odd: M 66816, N 72, K 48 (x1 per forward; N and K multiples of 8: True and True): 16.51 TFLOP/s
- mirror_even_odd: M 66816, N 48, K 72 (x1 per forward; N and K multiples of 8: True and True): 16.84 TFLOP/s
- mirror_even_odd: M 66816, N 72, K 80 (x1 per forward; N and K multiples of 8: True and True): 27.71 TFLOP/s
- mirror_even_odd: M 66816, N 96, K 72 (x1 per forward; N and K multiples of 8: True and True): 32.95 TFLOP/s
- mirror_even_odd: M 66816, N 72, K 96 (x1 per forward; N and K multiples of 8: True and True): 33.33 TFLOP/s
- mirror_even_odd: M 66816, N 72, K 120 (x1 per forward; N and K multiples of 8: True and True): 41.19 TFLOP/s
- mirror_even_odd: M 66816, N 136, K 72 (x1 per forward; N and K multiples of 8: True and True): 46.60 TFLOP/s
- mirror_even_odd: M 66816, N 72, K 144 (x1 per forward; N and K multiples of 8: True and True): 49.60 TFLOP/s

## Alignment of the even/odd form

- align 1, batch 64: 2,468.4 img/s (25.927 ms)
- align 1, batch 256: 2,769.1 img/s (92.448 ms)
- align 8, batch 64: 5,826.1 img/s (10.985 ms)
- align 8, batch 256: 6,205.3 img/s (41.255 ms)

Files: systems_rows.csv, systems_table.csv, profile.json, gemm.csv, align.csv, align_gemm.csv, metrics.json, meta.json.
