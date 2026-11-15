# Zero-shot comparison at matched parameters (MUST-07), DINOv2-B reg

Validation k-NN top-1, mean ± std over fit seeds (n). Sizes: 6mlp = blocks 1 to 6, MLP only; 6both = blocks 1 to 6, MLP and attention; 9both; full = every block.

| Method | 6mlp | 6both | 9both | full | diff 6mlp (CI s0) | diff 6both (CI s0) |
|---|---|---|---|---|---|---|
| Original model |  |  |  |  |  |  |
| Mirror-exact | 76.60 ± 0.02 (n = 3) | 77.34 ± 0.08 (n = 3) | 73.37 ± 0.11 (n = 3) | 53.81 ± 0.36 (n = 3) |  |  |
| Mirror-local (ablation) | 77.12 (n = 1) | 65.90 (n = 1) | 62.26 (n = 1) | 62.10 (n = 1) |  |  |
| Plain SVD | 9.37 ± 0.00 (n = 3) | 5.87 ± 0.03 (n = 3) | 5.67 (n = 1) | 5.17 (n = 1) | 67.24 [66.80, 67.64] | 71.47 [71.16, 71.97] |
| SVD-LLM-style whitening | 78.35 ± 0.03 (n = 3) | 78.02 ± 0.01 (n = 3) | 77.35 (n = 1) | 75.80 (n = 1) | -1.75 [-1.97, -1.52] | -0.68 [-0.82, -0.42] |
| ASVD-style scaling | 76.81 ± 0.04 (n = 3) | 74.41 ± 0.02 (n = 3) | 65.24 (n = 1) | 57.09 (n = 1) | -0.20 [-0.49, -0.04] | 2.92 [2.71, 3.24] |
| SliceGPT-style slicing | 31.41 ± 1.23 (n = 3) | 75.60 ± 0.11 (n = 3) | 72.48 (n = 1) | 68.33 (n = 1) | 45.19 [44.03, 44.97] | 1.74 [1.44, 1.94] |
| FLAP-style pruning | 75.20 ± 0.07 (n = 3) | 70.29 ± 0.18 (n = 3) | 63.47 (n = 1) | 65.12 (n = 1) | 1.41 [1.22, 1.71] | 7.05 [6.93, 7.58] |
| ZipIt-style merging | 75.67 ± 0.15 (n = 3) | 18.51 ± 1.20 (n = 3) | 13.38 (n = 1) | 13.58 (n = 1) | 0.93 [0.58, 1.05] | 58.83 [57.47, 58.39] |
| Pairing by same-image correlation | 27.42 ± 0.81 (n = 3) | 31.31 ± 1.53 (n = 3) | 13.36 (n = 1) | 12.17 (n = 1) | 49.19 [48.53, 49.47] | 46.03 [45.01, 45.95] |
| Pairing by weight cosine | 34.42 ± 0.36 (n = 3) | 39.85 ± 0.32 (n = 3) | 16.94 (n = 1) | 14.98 (n = 1) | 42.19 [42.12, 43.05] | 37.48 [36.89, 37.81] |
| Random pairing | 22.40 ± 0.69 (n = 3) | 32.30 ± 3.30 (n = 3) | 11.07 (n = 1) | 10.02 (n = 1) | 54.20 [53.18, 54.12] | 45.04 [42.33, 43.27] |
| Random structure | 2.31 ± 0.50 (n = 3) | 2.29 ± 0.42 (n = 3) | 4.31 (n = 1) | 4.78 (n = 1) | 74.29 [73.95, 74.72] | 75.05 [74.28, 75.06] |

Sources: `data/must07_rows.csv`, `data/table3_must07_p7.csv`
