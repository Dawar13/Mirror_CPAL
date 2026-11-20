# T2: stage ablation (A04), DINOv2-B/14 reg, final bf16 k-NN

Written by `scripts/collect_final_phase.py`; every row's source file and commit are in `T2_stage_ablation.csv` (columns source_file and git_hash).

| row | symmetry | compression | healed | knn_top1 | minus_main | params_frac | note |
|---|---|---|---|---|---|---|---|
| Original | no | no | no | 78.94 | 0.5860 | n/a |  |
| Original distilled (healing without surgery) | no | no | yes | 78.95 | 0.5960 | 1.00 |  |
| Surgery, before healing | yes | yes | no | 53.84 | -24.51 | n/a |  |
| Surgery + healing (main arm, 3-seed mean) | yes | yes | yes | 78.35 | 0 | n/a |  |
| Random structure + healing | random | yes | yes | 70.10 | -8.25 | 0.7646 |  |
| True operator + random pairing + healing (H1) | partial | yes | yes | 77.01 | -1.34 | 0.7646 |  |
| Compression without symmetry + healing: SVD-LLM (whitening only) | no | yes | yes | 78.08 | -0.2693 | n/a |  |
| Compression without symmetry + healing: full SVD-LLM | no | yes | yes | 76.42 | -1.93 | n/a |  |
| Compression without symmetry + healing: IO-SVD | no | yes | yes | 0.5340 | -77.82 | n/a |  |
| Compression without symmetry + healing: SliceGPT-style | no | yes | yes | 77.85 | -0.4980 | n/a |  |
| Compression without symmetry + healing: plain SVD | no | yes | yes | 77.00 | -1.35 | n/a |  |
| Width-pruned dense student + healing | no | yes | yes | 78.41 | 0.0560 | 0.7646 |  |
| Block-dropped dense student + healing | no | yes | yes | 75.25 | -3.10 | 0.7544 |  |

