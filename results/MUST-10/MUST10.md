# MUST-10: healed Mirror-exact against test-time flip averaging

k-NN: 100k train bank, 50k validation queries, k = 20, T = 0.07; top-1 with bf16 features; flip consistency also with fp32 features for the bank and every query (the measure of exact invariance). Probe: scripts/tier2_linear.py recipe. GFLOPs: analytic, per image (forward passes times one pass).

| model | mode | forward passes | k-NN top-1 | flip consistency (fp32) | flip consistency (k-NN precision) | linear probe top-1 | GFLOPs per image | parameters (share) |
|---|---|---|---|---|---|---|---|---|
| original | plain | 1 | 78.95 | - | 96.280 | 81.00 | 47.08 | 86,583,552 (1.000) |
| original | flipavg | 2 | 79.00 | 100.000 | 100.000 | 81.13 | 94.16 | 86,583,552 (1.000) |
| healed | plain | 1 | 78.32 | 100.000 | 99.744 | 80.10 | 36.44 | 66,203,360 (0.765) |

Healed minus flip-averaged original: k-NN -0.68 points, probe -1.02 points; GFLOPs ratio 0.387, parameter ratio 0.765.
Paired bootstrap (knn, n = 50000, 10000 resamples): -0.68 points, 95% interval -0.87 to -0.51.
Paired bootstrap (probe, n = 50000, 10000 resamples): -1.02 points, 95% interval -1.20 to -0.85.
