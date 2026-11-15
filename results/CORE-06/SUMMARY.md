# CORE-06: orientation is kept, not destroyed

Commit e84e20ff35 (dirty); seed 0; PREREG_NEXT_PHASE.md sha256 711b963039af.

CUB-200-2011: 5994 train and 5794 test images; facing labels on 5443 train and 5307 test images (share facing right: train 0.507, test 0.497). Facing convention: sign = sign(x_beak - x_tail); +1 = beak right of tail (bird faces right); probe class 1 = faces right, 0 = faces left; images without a label (a part not visible, or parts closer than 10% of the box width) are left out of every facing probe. CUB images overlap with ImageNet (Caltech warning); the comparisons are within each model.

| Prediction | Verdict | Detail |
|---|---|---|
| P6.1 species from the even part is within 1 point of the full feature | PASS | original: even_cls 89.96, full_cls 90.01, full_minus_even 0.05, pass True; vexact_s0: even_cls 89.45, full_cls 89.42, full_minus_even -0.03, pass True; vexact_s1: even_cls 89.52, full_cls 89.47, full_minus_even -0.05, pass True; vexact_s2: even_cls 89.44, full_cls 89.37, full_minus_even -0.07, pass True |
| P6.2 facing direction from the odd part is at least 85% | PASS | original: feature odd_cls, fallback_to_mean_patch False, odd_cls_cv_top1_train 95.68, facing_acc_natural 94.99, facing_acc_mirror_augmented 95.31, odd_cls_natural 94.99, odd_patch_natural 96.63, pass True; vexact_s0: feature odd_cls, fallback_to_mean_patch False, odd_cls_cv_top1_train 99.14, facing_acc_natural 99.19, facing_acc_mirror_augmented 99.18, odd_cls_natural 99.19, odd_patch_natural 99.32, pass True; vexact_s1: feature odd_cls, fallback_to_mean_patch False, odd_cls_cv_top1_train 99.12, facing_acc_natural 99.21, facing_acc_mirror_augmented 99.19, odd_cls_natural 99.21, odd_patch_natural 99.27, pass True; vexact_s2: feature odd_cls, fallback_to_mean_patch False, odd_cls_cv_top1_train 99.08, facing_acc_natural 99.25, facing_acc_mirror_augmented 99.25, odd_cls_natural 99.25, odd_patch_natural 99.30, pass True |
| P6.3 facing from healed Mirror-exact's even part is at most 60% on natural test images and exactly 50% on the mirror-augmented test set | PASS | vexact_s0: natural 50.84, mirror_augmented 50.00, mirror_augmented_raw_exactly_50 True, n_pairs_differ 0, n_pairs_differ_not_near_tie 0, pass_natural True, pass_exact50 True, pass True; vexact_s1: natural 50.29, mirror_augmented 50.00, mirror_augmented_raw_exactly_50 True, n_pairs_differ 0, n_pairs_differ_not_near_tie 0, pass_natural True, pass_exact50 True, pass True; vexact_s2: natural 50.57, mirror_augmented 50.00, mirror_augmented_raw_exactly_50 True, n_pairs_differ 0, n_pairs_differ_not_near_tie 0, pass_natural True, pass_exact50 True, pass True |

Operational definitions: P6.1: |full CLS - even CLS| species top-1 on natural test images <= 1.0, on every model probed (original with A_L, each healed Mirror-exact seed). P6.2: facing accuracy on natural test images >= 85.0% from the odd CLS, or from the odd mean patch token when the odd CLS's training-split CV accuracy is below 85.0% (decided before scoring); on every model probed. P6.3: even CLS, natural <= 60.0%; mirror-augmented 50% with every differing pair a near-tie (relative logit margin < 0.0001); every healed seed.

## All probes (accuracy %, weight decay chosen by cross-validation on the training split)

| Variant | Task | Features | Test set | Accuracy | n | wd | CV (train) | Pairs predicted alike |
|---|---|---|---|---|---|---|---|---|
| original | species | even_cls | natural | 89.96 | 5794 | 0.01 | 89.29 | - |
| original | species | even_cls | mirror_augmented | 90.06 | 11588 | 0.01 | 89.29 | 98.00 |
| original | species | odd_cls | natural | 25.01 | 5794 | 1e-05 | 25.39 | - |
| original | species | odd_cls | mirror_augmented | 25.41 | 11588 | 1e-05 | 25.39 | 21.83 |
| original | species | full_cls | natural | 90.01 | 5794 | 0.001 | 89.27 | - |
| original | species | full_cls | mirror_augmented | 90.05 | 11588 | 0.001 | 89.27 | 97.17 |
| original | facing | even_cls | natural | 86.70 | 5307 | 0.01 | 84.99 | - |
| original | facing | even_cls | mirror_augmented | 86.31 | 10614 | 0.01 | 84.99 | 26.10 |
| original | facing | odd_cls | natural | 94.99 | 5307 | 1 | 95.68 | - |
| original | facing | odd_cls | mirror_augmented | 95.31 | 10614 | 1 | 95.68 | 5.95 |
| original | facing | full_cls | natural | 96.33 | 5307 | 0.01 | 96.27 | - |
| original | facing | full_cls | mirror_augmented | 96.42 | 10614 | 0.01 | 96.27 | 4.33 |
| original | facing | even_patch | natural | 83.38 | 5307 | 0.01 | 80.91 | - |
| original | facing | even_patch | mirror_augmented | 82.91 | 10614 | 0.01 | 80.91 | 32.22 |
| original | facing | odd_patch | natural | 96.63 | 5307 | 0.1 | 96.42 | - |
| original | facing | odd_patch | mirror_augmented | 96.72 | 10614 | 0.1 | 96.42 | 2.34 |
| original | facing | full_patch | natural | 96.87 | 5307 | 0.1 | 96.55 | - |
| original | facing | full_patch | mirror_augmented | 96.82 | 10614 | 0.1 | 96.55 | 2.92 |
| vexact_s0 | species | even_cls | natural | 89.45 | 5794 | 0.01 | 88.69 | - |
| vexact_s0 | species | even_cls | mirror_augmented | 89.45 | 11588 | 0.01 | 88.69 | 100.00 |
| vexact_s0 | species | odd_cls | natural | 3.80 | 5794 | 0.001 | 3.69 | - |
| vexact_s0 | species | odd_cls | mirror_augmented | 3.42 | 11588 | 0.001 | 3.69 | 0.00 |
| vexact_s0 | species | full_cls | natural | 89.42 | 5794 | 0.01 | 88.57 | - |
| vexact_s0 | species | full_cls | mirror_augmented | 89.36 | 11588 | 0.01 | 88.57 | 98.96 |
| vexact_s0 | facing | even_cls | natural | 50.84 | 5307 | 0.001 | 49.53 | - |
| vexact_s0 | facing | even_cls | mirror_augmented | 50.00 | 10614 | 0.001 | 49.53 | 100.00 |
| vexact_s0 | facing | odd_cls | natural | 99.19 | 5307 | 0.001 | 99.14 | - |
| vexact_s0 | facing | odd_cls | mirror_augmented | 99.18 | 10614 | 0.001 | 99.14 | 0.02 |
| vexact_s0 | facing | full_cls | natural | 95.38 | 5307 | 0.01 | 95.59 | - |
| vexact_s0 | facing | full_cls | mirror_augmented | 95.30 | 10614 | 0.01 | 95.59 | 4.24 |
| vexact_s0 | facing | even_patch | natural | 50.74 | 5307 | 0.1 | 51.04 | - |
| vexact_s0 | facing | even_patch | mirror_augmented | 50.00 | 10614 | 0.1 | 51.04 | 100.00 |
| vexact_s0 | facing | odd_patch | natural | 99.32 | 5307 | 0.001 | 98.93 | - |
| vexact_s0 | facing | odd_patch | mirror_augmented | 99.27 | 10614 | 0.001 | 98.93 | 0.32 |
| vexact_s0 | facing | full_patch | natural | 97.29 | 5307 | 0.01 | 97.26 | - |
| vexact_s0 | facing | full_patch | mirror_augmented | 97.26 | 10614 | 0.01 | 97.26 | 3.00 |
| vexact_s1 | species | even_cls | natural | 89.52 | 5794 | 0.01 | 88.76 | - |
| vexact_s1 | species | even_cls | mirror_augmented | 89.52 | 11588 | 0.01 | 88.76 | 100.00 |
| vexact_s1 | species | odd_cls | natural | 3.71 | 5794 | 0.001 | 3.70 | - |
| vexact_s1 | species | odd_cls | mirror_augmented | 3.39 | 11588 | 0.001 | 3.70 | 0.00 |
| vexact_s1 | species | full_cls | natural | 89.47 | 5794 | 0.01 | 88.61 | - |
| vexact_s1 | species | full_cls | mirror_augmented | 89.43 | 11588 | 0.01 | 88.61 | 98.96 |
| vexact_s1 | facing | even_cls | natural | 50.29 | 5307 | 1 | 49.97 | - |
| vexact_s1 | facing | even_cls | mirror_augmented | 50.00 | 10614 | 1 | 49.97 | 100.00 |
| vexact_s1 | facing | odd_cls | natural | 99.21 | 5307 | 0.0001 | 99.12 | - |
| vexact_s1 | facing | odd_cls | mirror_augmented | 99.19 | 10614 | 0.0001 | 99.12 | 0.08 |
| vexact_s1 | facing | full_cls | natural | 95.46 | 5307 | 0.001 | 95.59 | - |
| vexact_s1 | facing | full_cls | mirror_augmented | 95.40 | 10614 | 0.001 | 95.59 | 5.92 |
| vexact_s1 | facing | even_patch | natural | 50.07 | 5307 | 0.01 | 51.24 | - |
| vexact_s1 | facing | even_patch | mirror_augmented | 50.00 | 10614 | 0.01 | 51.24 | 100.00 |
| vexact_s1 | facing | odd_patch | natural | 99.27 | 5307 | 0.01 | 99.01 | - |
| vexact_s1 | facing | odd_patch | mirror_augmented | 99.20 | 10614 | 0.01 | 99.01 | 0.28 |
| vexact_s1 | facing | full_patch | natural | 97.23 | 5307 | 0.01 | 97.17 | - |
| vexact_s1 | facing | full_patch | mirror_augmented | 97.24 | 10614 | 0.01 | 97.17 | 3.00 |
| vexact_s2 | species | even_cls | natural | 89.44 | 5794 | 0.01 | 88.66 | - |
| vexact_s2 | species | even_cls | mirror_augmented | 89.44 | 11588 | 0.01 | 88.66 | 100.00 |
| vexact_s2 | species | odd_cls | natural | 3.62 | 5794 | 0.001 | 3.80 | - |
| vexact_s2 | species | odd_cls | mirror_augmented | 3.31 | 11588 | 0.001 | 3.80 | 0.00 |
| vexact_s2 | species | full_cls | natural | 89.37 | 5794 | 0.01 | 88.56 | - |
| vexact_s2 | species | full_cls | mirror_augmented | 89.33 | 11588 | 0.01 | 88.56 | 99.10 |
| vexact_s2 | facing | even_cls | natural | 50.57 | 5307 | 0.001 | 49.72 | - |
| vexact_s2 | facing | even_cls | mirror_augmented | 50.00 | 10614 | 0.001 | 49.72 | 100.00 |
| vexact_s2 | facing | odd_cls | natural | 99.25 | 5307 | 0.001 | 99.08 | - |
| vexact_s2 | facing | odd_cls | mirror_augmented | 99.25 | 10614 | 0.001 | 99.08 | 0.04 |
| vexact_s2 | facing | full_cls | natural | 95.25 | 5307 | 0.01 | 95.55 | - |
| vexact_s2 | facing | full_cls | mirror_augmented | 95.24 | 10614 | 0.01 | 95.55 | 4.20 |
| vexact_s2 | facing | even_patch | natural | 50.74 | 5307 | 0.1 | 50.96 | - |
| vexact_s2 | facing | even_patch | mirror_augmented | 50.00 | 10614 | 0.1 | 50.96 | 100.00 |
| vexact_s2 | facing | odd_patch | natural | 99.30 | 5307 | 0.001 | 98.97 | - |
| vexact_s2 | facing | odd_patch | mirror_augmented | 99.27 | 10614 | 0.001 | 98.97 | 0.32 |
| vexact_s2 | facing | full_patch | natural | 97.31 | 5307 | 0.01 | 97.23 | - |
| vexact_s2 | facing | full_patch | mirror_augmented | 97.29 | 10614 | 0.01 | 97.23 | 2.94 |

## Invariance checks (test images against their mirrors; forward precision fp32)

- original: cls_even_rel_max 4.14e-01, cls_odd_rel_max 3.25e+00, cls_odd_rel_median 1.81e+00, cls_odd_norm_share_median 1.14e-02, patch_even_rel_max 4.59e-01, patch_odd_rel_max 3.67e+00, patch_odd_rel_median 1.26e+00, patch_odd_norm_share_median 8.43e-03
- vexact_s0: cls_even_rel_max 6.36e-06, cls_odd_rel_max 1.96e-04, cls_odd_rel_median 2.42e-05, cls_odd_norm_share_median 3.22e-03, patch_even_rel_max 1.15e-05, patch_odd_rel_max 4.56e-05, patch_odd_rel_median 7.45e-06, patch_odd_norm_share_median 3.57e-03
- vexact_s1: cls_even_rel_max 6.38e-06, cls_odd_rel_max 1.98e-04, cls_odd_rel_median 2.43e-05, cls_odd_norm_share_median 3.20e-03, patch_even_rel_max 8.28e-06, patch_odd_rel_max 4.57e-05, patch_odd_rel_median 6.88e-06, patch_odd_norm_share_median 3.61e-03
- vexact_s2: cls_even_rel_max 6.35e-06, cls_odd_rel_max 2.12e-04, cls_odd_rel_median 2.40e-05, cls_odd_norm_share_median 3.24e-03, patch_even_rel_max 1.84e-05, patch_odd_rel_max 3.85e-05, patch_odd_rel_median 6.54e-06, patch_odd_norm_share_median 3.60e-03

Files: probes.csv, probe/*.json, features/*.npz, metrics.json, meta.json.
