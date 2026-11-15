# MUST-01: the law against the achieved saving

Commit 52e29d1676 (uncommitted changes present). Written by scripts/must01_law.py; every number below is read from law_per_layer.csv and law_surgery_k.csv in this folder.

Predicted: free parameters of fc1 and fc2 from the character formula (mirror/groups.py) with the block's operators and twin matching. Achieved: entries of the weight tensors of the even/odd MLP (mirror/efficient.py) built from random weights projected onto the same structure. The difference is in percentage points of the dense MLP weights. Pre-registered bar: 1 point per block.

| Model | Fit seeds | Blocks | Structure | Max abs diff, MLP (pp) | Max abs diff, fc1 / fc2 (pp) | Max abs diff vs recorded (pp) | Biases match | Counts match the run's CSV |
|---|---|---|---|---|---|---|---|---|
| dinov2_vitb14 | 0, 1, 2 | 36 | fitted operators and pairings | 0.00e+00 | 0.00e+00 / 0.00e+00 | 8.33e-15 | yes | yes |
| dinov2_vitb14_reg | 0, 1, 2 | 36 | fitted operators and pairings | 0.00e+00 | 0.00e+00 / 0.00e+00 | 8.33e-15 | yes | yes |
| deit3_b16 | 0 | 12 | fitted operators and pairings | 0.00e+00 | 0.00e+00 / 0.00e+00 | 8.33e-15 | yes | yes |
| dinov2_vitl14_reg | 0 | 24 | synthetic structure with the recorded counts | 0.00e+00 | 0.00e+00 / 0.00e+00 | 9.71e-15 | yes | yes |
| dinov2_vits14_reg | 0 | 12 | synthetic structure with the recorded counts | 0.00e+00 | 0.00e+00 / 0.00e+00 | 9.02e-15 | yes | yes |
| mae_b16 | 0 | 12 | fitted operators and pairings | 0.00e+00 | 0.00e+00 / 0.00e+00 | 5.55e-15 | yes | yes |
| siglip2_b16 | 0 | 12 | fitted operators and pairings | 0.00e+00 | 0.00e+00 / 0.00e+00 | 8.33e-15 | yes | yes |

Largest absolute per-block difference over every model: 0.00e+00 percentage points (within the 1-point bar).

For dinov2_vitl14_reg, dinov2_vits14_reg the fitted operators and pairings are not on this machine, so the structure is synthetic with the recorded counts (d+ per layer, twin pairs and self-twins per block). For these models both numbers derive from the same recorded counts: the check covers the arithmetic of the two counting routes, not the stored structure.

Surgery rows recorded by the runs (blocks 1 to k): MLP parameters kept, recorded against achieved here.

| Model | Fit seed | k | Recorded | Achieved | Diff (pp) |
|---|---|---|---|---|---|
| dinov2_vitb14 | 0 | 3 | 0.534401 | 0.534401 | 0.00e+00 |
| dinov2_vitb14 | 0 | 6 | 0.595236 | 0.595236 | 0.00e+00 |
| dinov2_vitb14 | 0 | 9 | 0.676694 | 0.676694 | 0.00e+00 |
| dinov2_vitb14 | 1 | 6 | 0.594855 | 0.594855 | 0.00e+00 |
| dinov2_vitb14 | 2 | 6 | 0.595053 | 0.595053 | 0.00e+00 |
| dinov2_vitb14_reg | 0 | 3 | 0.539918 | 0.539918 | 0.00e+00 |
| dinov2_vitb14_reg | 0 | 6 | 0.581481 | 0.581481 | 0.00e+00 |
| dinov2_vitb14_reg | 0 | 9 | 0.658148 | 0.658148 | 0.00e+00 |
| dinov2_vitb14_reg | 1 | 6 | 0.581100 | 0.581100 | 0.00e+00 |
| dinov2_vitb14_reg | 2 | 6 | 0.581405 | 0.581405 | 0.00e+00 |
| deit3_b16 | 0 | 6 | 0.651288 | 0.651288 | 0.00e+00 |
| dinov2_vitl14_reg | 0 | 12 | 0.586002 | 0.586002 | 0.00e+00 |
| dinov2_vits14_reg | 0 | 6 | 0.599829 | 0.599829 | 0.00e+00 |
| mae_b16 | 0 | 6 | 0.551819 | 0.551819 | 0.00e+00 |
| siglip2_b16 | 0 | 6 | 0.769120 | 0.769120 | 0.00e+00 |

The formula and the module count agree by construction when the code is right, so this is a correctness check, not a finding (PROJECT_NEXT_PHASE.md, MUST-01).
