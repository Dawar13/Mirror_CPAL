# Tier 2: transfer linear probes

Same features and probe as the ImageNet linear probe (T2_LINEAR.md): [CLS (or pooled) after the final norm, mean patch token after the final norm], bf16 forward passes, float16 cache, multinomial logistic regression on standardised features (L-BFGS). Weight decay chosen on a fixed class-stratified 10% split of the training images, never on the test split; refitted on all training images and scored once on the test split. Images: resize 256 (bicubic), centre crop 224.

## dinov2_vitb14_reg: top-1 (mean per-class top-1)

| Variant | cifar100 | flowers102 | pets | dtd |
|---|---|---|---|---|
| original | 91.40 (91.40) | 99.69 (99.73) | 95.99 (95.96) | 83.56 (83.56) |
| healed_vexact_k6 | 90.54 (90.54) | 99.69 (99.73) | 95.67 (95.63) | 83.19 (83.19) |
| vexact_k6 | 85.21 (85.21) | 99.69 (99.73) | 95.72 (95.68) | 81.65 (81.65) |
| healed_mirror_k9 | 90.89 (90.89) | 99.69 (99.73) | 95.58 (95.55) | 83.35 (83.35) |
| healed_random_k9 | 84.47 (84.47) | 99.56 (99.55) | 94.30 (94.26) | 77.39 (77.39) |
| healed_svd_k9 | 88.67 (88.67) | 99.69 (99.73) | 95.50 (95.47) | 80.69 (80.69) |
| vlocal_k9 | 80.89 (80.89) | 99.59 (99.62) | 94.74 (94.68) | 80.37 (80.37) |

- healed_vexact_k6: feature [CLS, mean patch token] after the student's own final norm (RMSNorm, affine; no even head at k = 6 < 12), the forward of the k-NN feature of scripts/heal_vexact.py; exactly equivariant over blocks 1..6 only, so not mirror-invariant; checks after loading (64 v2 check images, float32): span exactness max rel 4.6e-06 (tol 0.0001; 4.6e-06 recorded after healing), image vs mirror probe feature max rel 4.2e-01: PASS; Tier-1 k-NN of the healed run 78.79 (bf16 features).

Datasets (Hugging Face ids, verify on VM): cifar100: uoft-cs/cifar100; flowers102: dpdl-benchmark/oxford_flowers102; pets: timm/oxford-iiit-pet; dtd: tanganke/dtd.
Source: `t2_transfer.csv` and `probe/<model>/<dataset>/<variant>.json`. Commit(s): 8a7b6c4e42 (dirty), b95ccd5886; seed(s): 0.
