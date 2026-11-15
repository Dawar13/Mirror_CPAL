# Tier 2: transfer linear probes

Same features and probe as the ImageNet linear probe (T2_LINEAR.md): [CLS (or pooled) after the final norm, mean patch token after the final norm], bf16 forward passes, float16 cache, multinomial logistic regression on standardised features (L-BFGS). Weight decay chosen on a fixed class-stratified 10% split of the training images, never on the test split; refitted on all training images and scored once on the test split. Images: resize 256 (bicubic), centre crop 224.

## dinov2_vitb14_reg: top-1 (mean per-class top-1)

| Variant | cifar100 | flowers102 | pets | dtd |
|---|---|---|---|---|
| original | 91.42 (91.42) | 99.69 (99.73) | 95.94 (95.91) | 83.51 (83.51) |
| best_svdllm_s0 | 89.40 (89.40) | 99.64 (99.67) | 95.53 (95.49) | 82.07 (82.07) |
| vexact_s0 | 90.18 (90.18) | 99.69 (99.73) | 95.69 (95.66) | 83.35 (83.35) |
| vexact_s0_fullpatch | 90.14 (90.14) | 99.69 (99.73) | 95.50 (95.47) | 83.24 (83.24) |

- best_svdllm_s0: healed run /home/ubuntu/Mirror_CPAL/runs/MUST-09/dinov2_vitb14_reg/svdllm/s0 (mirror.heal_structured.load_healed_model); method svdllm; seed 0; position embedding as loaded (257 rows; the original model has 1370), used as the model uses it at 224 px.
- vexact_s0: healed run /home/ubuntu/Mirror_CPAL/runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0 (mirror.heal_structured.load_healed_model); method mirror_exact; seed 0; exact operator: loader info['A_star']; position embedding as loaded (257 rows; the original model has 1370), used as the model uses it at 224 px.
- vexact_s0_fullpatch: feature view b: [CLS after the final norm (even part), mean of the full patch tokens (final RMSNorm then affine, even projection bypassed)]; not flip-invariant (the odd part of the patch mean changes sign under a flip); model as in view a: healed run /home/ubuntu/Mirror_CPAL/runs/MUST-09/dinov2_vitb14_reg/mirror_exact/s0 (mirror.heal_structured.load_healed_model); method mirror_exact; seed 0; exact operator: loader info['A_star']; position embedding as loaded (257 rows; the original model has 1370), used as the model uses it at 224 px.

Datasets (Hugging Face ids, verify on VM): cifar100: uoft-cs/cifar100; flowers102: dpdl-benchmark/oxford_flowers102; pets: timm/oxford-iiit-pet; dtd: tanganke/dtd.
Source: `t2_transfer.csv` and `probe/<model>/<dataset>/<variant>.json`. Commit(s): 805c231c9c (dirty); seed(s): 0.
