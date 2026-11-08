# PREREG_E2: pre-registered hypotheses for the paper experiments

Transcribed from `PAPER_EXPERIMENTS.md` (X5) without changing any threshold. This file is committed
before F3, F4, G, D or M3 run; the commit that adds it is the pre-registration record, and every
later report states that commit hash. Thresholds are never edited after the commit.

Main model: DINOv2 ViT-B/14 with registers (`dinov2_vitb14_reg`); contrast model: plain
`dinov2_vitb14`. Protocol v2 (PAPER_EXPERIMENTS.md Section 1): operators fitted on ImageNet train
(fit seeds 0, 1, 2; 2,000 images each), held-out errors and k-NN queries on validation, k-NN bank of
100 train images per class. "Layers 1 to 6" means the outputs of blocks 1 to 6.

## P1 (artefacts; tests claim C3; gate G3)
- Plain DINOv2: excluding the top 1% highest-norm tokens cuts the mean e(A_l) over layers 9 to 11
  by at least 2x.
- Registers model: the mean e(A_l) over layers 9 to 11 is at most 2x its mean over layers 1 to 6.
- Gate G3: if neither registers nor masking restores the late layers, claim C3 is dropped.

## P2 (symmetry versus low rank; tests claim C4)
- The sign of (activation-weighted E_svd minus activation-weighted s) predicts which method has the
  smaller k-NN drop in single-block surgery in at least 75% of block and model cases
  (models: DINOv2-reg, plain DINOv2, DeiT-III; all blocks).

## P3 (healing; gate G2)
At k = 9 (registers model, MLP plus attention, V-local), all of:
- healed Mirror Surgery is at least 2.0 k-NN points above the healed random-structure control;
- it is at least 3.0 points above healed SVD;
- it is within 2.0 points of the original.
Gate G2: if the first two fail, the structure is not what matters after healing; stop and discuss.

## P4 (generality; tests claim C7)
- DeiT-III and MAE: mean MLP breaking share at most 0.25 over their first half of blocks, and
  e(null)/e(A) at least 5.
- SigLIP2's breaking share is higher than DINOv2-reg's.

## P5 (data-free readout; tests claim C6; gate G4)
- The layer-0 data-free operator's even subspace agrees with the data-driven one (mean cosine of
  principal angles at least 0.8).
- Weight-only twin pairing in block 1 recovers at least 70% of the data-driven pairs.
- Gate G4: below the bar, claim C6 is dropped.

## Fixed analysis choices (stated now so they cannot be tuned later)
- Breaking shares and e(A) use the same definitions and aggregation as E1b (mean over the stated
  layers or blocks; matched nulls with the same d+ and d-, and the same numbers of pairs and
  self-twins).
- "Mean MLP breaking share" in P4 is the larger of mean s1 and mean s2 over the first half of
  blocks, exactly as in the E1b decision row 4.
- k-NN: k = 20, T = 0.07, CLS after the final norm (pooled embedding for SigLIP2, which has no CLS),
  features computed in bf16 for every variant alike.
