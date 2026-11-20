# A09 addendum to FINAL_PHASE_REPORT.md (20 November 2026)

`results/FINAL_PHASE_REPORT.md` lists A09-1 to A09-3 as PENDING with no value. The audits did run (jobs `a09_audit_b`
and `a09_audit_deit`, commit 7d05b6439c), but they wrote to `results/A09/<model>/` while the collector looks for
`results/A09/<model>/mirror_exact_<variant>_s<seed>/`. The VM was destroyed after the pull, and the collector cannot
be rerun on the laptop without the k-NN arrays the bootstrap needs, so the generated report is left unchanged and the
audits' own scoring (`scripts/fp_a09_audit.py`, the same pre-registered rules) is copied here from
`results/A09/dinov2_vitb14_reg/summary.md` and `results/A09/deit3_b16/summary.md`.

| Rule | DINOv2-B/14 reg (Mirror A*, seed 0) | DeiT-III-B/16 (Mirror, seed 0) |
|---|---|---|
| A09-1, exact in float64 (largest relative error at every tensor ≤ 1e-10) | PASS: 6.85e-14 (dense block 4 v) | PASS: 3.35e-13 (even/odd block 1 attention probabilities) |
| A09-2, numerically exact in float32 (error ≤ 2x the model's own float32-versus-float64 deviation at every tensor; 100% float32 flip consistency for both forms) | FAIL: 373 of 374 tensors within the bar; the one exception is the dense embedding prefix tokens at ratio 2.0; float32 flip consistency 100% for both forms (0 of 50,000 images differ) | FAIL: 375 of 376 tensors; the same tensor at ratio 2.0 |
| A09-3, bfloat16 (report; target 100% for the even/odd form) | dense 99.758% (121 of 50,000 differ), even/odd 99.738% (131 differ); largest error 0.112 (dense), 0.0278 (even/odd) | largest error 0.395 (dense), 0.0513 (even/odd); flip consistency not recorded by the audit |

Bitwise level (E3, DINOv2-B): 0% of images give bit-identical features. The sign flip alone is exact in the MLP and the
attention (100% identical entries); the differences come from summation order (the patch-embedding convolution sums the
flipped kernel window in another order, and attention sums over keys in mirrored order).

The DeiT-III classifier's own float32 check (A08-1, `runs/final/a08/deit3_b16/mirror_exact/s{0,1,2}/eval/top1_step780.json`)
gives 100% flip consistency with 0 of 50,000 images differing at every seed.
