# Earlier ablations: operator, pairing, fit size, block choice and projection strength (mean ± std over fit seeds)

Earlier-phase protocol (Mirror-local operators); see results/paper/INDEX.md for each setting.

| Model | Ablation | Variant | Blocks k | k-NN top-1 | Parameters vs Mirror | Matched |
|---|---|---|---|---|---|---|
| DINOv2-reg | A1 | per_layer | 6 | 77.08 ± 0.03 (n = 3) | 1.00 | yes |
| DINOv2-reg | A1 | shared_span | 6 | 76.37 ± 0.06 (n = 3) | 1.00 | yes |
| DINOv2-reg | A1 | shared_all | 6 | 20.48 ± 0.13 (n = 3) | 1.17 | no |
| DINOv2-reg | A1 | identity | 6 | 24.63 ± 0.19 (n = 3) | 1.18 | no |
| DINOv2-reg | A1 | random_operator | 6 | 3.88 ± 0.44 (n = 3) | 1.00 | yes |
| DINOv2-reg | A1 | svd | 6 | 8.77 ± 0.01 (n = 3) | 1.00 | yes |
| DINOv2-reg | A1 | random_structure | 6 | 3.70 ± 0.67 (n = 3) | 1.00 | yes |
| DINOv2-reg | A1 | per_layer | 9 | 76.34 ± 0.02 (n = 2) | 1.00 | yes |
| DINOv2-reg | A1 | shared_span | 9 | 67.78 ± 0.17 (n = 2) | 1.01 | yes |
| DINOv2-reg | A1 | shared_all | 9 | 15.22 ± 0.13 (n = 2) | 1.13 | no |
| DINOv2-reg | A1 | identity | 9 | 16.57 ± 0.19 (n = 2) | 1.14 | no |
| DINOv2-reg | A1 | random_operator | 9 | 3.84 ± 1.15 (n = 2) | 1.00 | yes |
| DINOv2-reg | A1 | svd | 9 | 5.83 ± 0.01 (n = 2) | 1.00 | yes |
| DINOv2-reg | A1 | random_structure | 9 | 4.63 ± 0.32 (n = 2) | 1.00 | yes |
| DINOv2-reg | A2 | mirror | 6 | 77.08 ± 0.03 (n = 3) | 1.00 | yes |
| DINOv2-reg | A2 | random_pairing | 6 | 28.96 ± 3.50 (n = 3) | 1.00 | yes |
| DINOv2-reg | A2 | self_twins | 6 | 13.73 ± 0.04 (n = 3) | 1.20 | no |
| DINOv2-reg | A2 | generic | 6 | 24.69 ± 0.05 (n = 3) | 1.00 | yes |
| DINOv2-reg | A2 | hungarian | 6 | 77.27 ± 0.05 (n = 3) | 0.98 | no |
| DINOv2-reg | A2 | hungarian_cycle | 6 | 77.20 ± 0.02 (n = 3) | 0.98 | no |
| DINOv2-reg | A2 | mirror | 9 | 76.34 ± 0.02 (n = 2) | 1.00 | yes |
| DINOv2-reg | A2 | random_pairing | 9 | 12.59 (n = 1) | 1.00 | yes |
| DINOv2-reg | A2 | self_twins | 9 | 12.05 (n = 1) | 1.17 | no |
| DINOv2-reg | A2 | generic | 9 | 13.81 (n = 1) | 1.00 | yes |
| DINOv2-reg | A2 | hungarian | 9 | 76.53 (n = 1) | 0.99 | no |
| DINOv2-reg | A2 | hungarian_cycle | 9 | 76.39 (n = 1) | 0.99 | no |
| DINOv2-reg | A3 | n100 | 6 | 77.03 (n = 1) | 1.00 | yes |
| DINOv2-reg | A3 | n500 | 6 | 77.18 (n = 1) | 1.00 | yes |
| DINOv2-reg | A3 | n2000 | 6 | 77.09 (n = 1) | 1.00 | yes |
| DINOv2-reg | A3 | n6000 | 6 | 77.11 (n = 1) | 1.00 | yes |
| DINOv2-reg | A4 | first | 6 | 77.08 ± 0.04 (n = 2) | 1.00 | yes |
| DINOv2-reg | A4 | last | 6 | 78.65 ± 0.02 (n = 2) | 1.53 | no |
| DINOv2-reg | A4 | lowest_share | 6 | 78.65 ± 0.02 (n = 2) | 1.53 | no |
| DINOv2-reg | A5 | alpha0.00 | 6 | 78.94 ± 0.00 (n = 2) | 1.72 | no |
| DINOv2-reg | A5 | alpha0.25 | 6 | 78.87 ± 0.03 (n = 2) | 1.72 | no |
| DINOv2-reg | A5 | alpha0.50 | 6 | 78.43 ± 0.01 (n = 2) | 1.72 | no |
| DINOv2-reg | A5 | alpha0.75 | 6 | 77.96 ± 0.03 (n = 2) | 1.72 | no |
| DINOv2-reg | A5 | alpha1.00 | 6 | 77.08 ± 0.04 (n = 2) | 1.00 | yes |

Sources: `data/table3_earlier_ablations.csv (from results/paper/tables/T_ablations.csv)`
