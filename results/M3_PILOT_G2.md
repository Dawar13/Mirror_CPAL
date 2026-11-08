# M3 healing pilot: gate G2 and P3

**P3: FAIL. Gate G2: FAIL.** Pre-registered in PREREG_E2.md (commit 9804f06); thresholds unchanged.

Setting: DINOv2 ViT-B/14 with registers, first 9 blocks converted, MLP plus attention, V-local; healing by feature distillation from the original model on 100k ImageNet train images, 2 epochs, batch 256, AdamW, cosine schedule, bf16, learning rate chosen per method on a held-out 5k train split. k-NN: 100k train bank, 50k validation queries, k = 20, T = 0.07, bf16 features.

| Method | Zero-shot top-1 | Healed top-1 | Healed flip consistency | Held-out loss before | after | LR | git |
|---|---|---|---|---|---|---|---|
| original | 78.94 | | 96.28 | | | | |
| mirror | 74.70 | 78.81 | 95.98 | 3.639 | 0.8628 | 1e-04 | 94b2e19 |
| svd | 6.49 | 77.57 | 94.12 | 53.11 | 2.13 | 1e-04 | 94b2e19 |
| random | 3.62 | 73.28 | 88.67 | 25.07 | 3.878 | 1e-04 | 94b2e19 |

| Pre-registered check | Value (points) | Threshold | Result |
|---|---|---|---|
| healed mirror minus healed random | 5.54 | >= 2.0 | PASS |
| healed mirror minus healed SVD | 1.24 | >= 3.0 | FAIL |
| original minus healed mirror | 0.12 | <= 2.0 | PASS |

Notes:
- The largest margin over healed SVD that any method could reach at this setting is 1.36 points (original minus healed SVD), below the 3.0 bar.
- Held-out loss is the normalised feature-distillation loss on the 5k held-out train images (lower is closer to the original model).
- Per CLAUDE.md, the healing grid (M3-grid) is not started after a failed gate; the framing goes back to Dawar.
