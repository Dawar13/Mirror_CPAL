# T5: discovery controls (A03); A03-1 scores the split-half ratio, the in-sample ratio is shown beside it

Written by `scripts/collect_final_phase.py`; every row's source file and commit are in `T5_discovery_controls.csv` (columns source_file and git_hash).

| control | label | d_minus | twin_fraction_1_6 | ratio_in_sample_1_6 | ratio_split_half_1_6 | spectrum_abs_sum | odd_share | zero_shot_knn | healed_knn |
|---|---|---|---|---|---|---|---|---|---|
| mirror | real left-right mirror (reference) | 68.00 | 0.5677 | 0.4791 | 0.4795 | 0.8648 | 0.0529 | 53.86 | n/a |
| C1 | shuffled correspondence: the mirror of a different image (seeded derangement), re-aligned | 338.00 | 0.0036 | 0.9700 | 0.9765 | 0.3535 | 0.0286 | 2.66 | n/a |
| C2 | no re-alignment: token (r, c) of x against token (r, c) of mirror(x) | 4.00 | 2.17e-04 | 1.04 | 1.03 | 0.5057 | 9.49e-04 | 12.52 | n/a |
| C3 | random involutive pixel permutation inside every patch (same in every patch and channel) | 199.00 | 0.0794 | 0.9209 | 0.9218 | 0.5980 | 0.1343 | 5.16 | n/a |
| C4 | random-operator null: random involutions with the fitted d+ and d- (A 1 = 1) and random pairings with the fitted cycle type (the reference's null draws) | 68.00 | 0.0041 | 1.00 | n/a | n/a | n/a | 4.79 | n/a |
| H1 | true operator, random pairing, healed (seed 0) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 77.01 |

