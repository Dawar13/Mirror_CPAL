# MUST-00: audit of Section 3 of PROJECT_NEXT_PHASE.md

Date 2026-11-14. Repository HEAD `a5094b4` (working tree as found). CPU only; read-only apart from this file and `audit.json`.

## Verdict

Section 3 has 21 rows; split into single claims they give 48 checks. **38 claims match (9 exactly, 29 after rounding), 2 mismatch and 8 need context; none is missing from the files.** The two mismatches: the healed 6-block exact model is not within about 0.5 points on transfer (CIFAR-100 is 0.86 below; the 0.51 belongs to the healed 9-block V-local pilot), and "SigLIP2 weak, as predicted" hides a pre-registered FAIL (P4 SigLIP2 sub-check). The context cases: the operator fit figures 10.6x and 0.16 come from E1b on plain DINOv2 under the older protocol and are means over layers 1 to 6, not layer 1 (registers model, paper protocol: 13.1x and 0.17 over layers 1 to 6, 50.7x and 0.041 at layer 1); the breaking-share range 0.19 to 0.22 spans two models (registers alone: 0.223 against 0.427); the 0.80 twin correlation is block 1 only; the artefact-token 2.27x is plain DINOv2 and the registers 1.30x is a different ratio; the 100% flip consistency is the fp32 check on 5,000 images (99.76% under the paper protocol); and 88.9 against 75.7 ms is the RTX 5090 profile while 0.70 to 0.91x is the A100. Every other number, including the 77.09 ± 0.003 std, the ablation rows and the 3.5M rotation count, is backed by a file. `configs/models/` does not exist yet, so the model IDs and recipes could not be confirmed.

## How the check was done

- Each value was read from the raw run file (CSV row or JSON key) and cross-checked against `results/paper/numbers.json` and the collector tables in `results/paper/tables/`. Values marked (computed) are simple arithmetic on file values done for this audit.
- Model, protocol and seed were checked in each run's metadata. Paper protocol here means a 100,000-image train bank, 50,000 validation queries, k = 20, T = 0.07 and bf16 features, as in the X2 runs; every X2, M1, M3, G and A k-NN number used it, and the ImageNet linear probe used the same 100k bank and 50k validation images. E1b used a 39,000-image bank and 8,000 queries.
- Line numbers refer to the files as they stood on 2026-11-14.

## Table

| # | Topic | Claim as written in Section 3 | Value found | Source | Verdict | Note |
|---|---|---|---|---|---|---|
| 1 | Original k-NN top-1 | 78.94 | 78.9359986782074 in all three fit-seed runs; the RTX 5090 runs measure 78.932 | runs/x2/perlayer_reg_s0/results/e1_surgery.csv:2 (k=0, original); numbers.json main.reg.original.top1; results/paper/tables/T_main_zeroshot.md:14 | **MATCH (rounded)** | Registers model, paper protocol. Older protocols gave other originals (E1, plain: 78.21; E1b registers arm: 78.11). |
| 2 | Twins | Block 1: 1,168 twin pairs and 736 self-twins of 3,072 neurons (76% in pairs) | n_pairs 1168, n_self 736, h 3072, twin_frac 0.7604166666666666 (fit seed 0). Fit seed 1: 1171 / 730 (0.762); fit seed 2: 1167 / 738 (0.760) | runs/x2/perlayer_reg_s0/results/e1_mlp.csv:2 (block 1); results/paper/tables/T_structure_per_layer.md:35 (seed 1: line 53) | **MATCH** | Fit seed 0 only; the counts differ slightly by fit seed, so name the seed. |
| 3 | Twins | Blocks 1 to 6: 63% in pairs | twin_frac_mean 0.630859375 (3 fit seeds, std 0.00066) | results/paper/tables/T_structure_summary.csv row dinov2_vitb14_reg,1-6; T_f1_ranges.md:11; numbers.json f1.reg.first_half.twin | **MATCH (rounded)** | Mean of the per-block twin fractions over blocks 1 to 6. |
| 4 | Twins | Blocks 7 to 12: 13% | twin_frac_mean 0.13494646990740738 (3 fit seeds) | results/paper/tables/T_structure_summary.csv row dinov2_vitb14_reg,7-12; T_f1_ranges.md:12 | **MATCH (rounded)** | Shown as 0.135 in T_f1_ranges.md; 13.49% rounds to 13%. |
| 5 | Twins | Mean twin correlation 0.80 | Block 1 mean twin correlation 0.8039 (fit seed 0), 0.8021 (seed 1), 0.8023 (seed 2). Mean over blocks 1 to 6: 0.689; blocks 7 to 12: 0.793; all 12 blocks: 0.741 (fit seed 0, computed by this audit from the per-block values) | runs/x2/perlayer_reg_s0/results/e1_mlp.csv:2 column mean_score_pairs; T_structure_per_layer.md:35 to 46 (twin corr column); T_structure_per_layer.csv | **CONTEXT** | 0.80 holds for block 1 only. Placed after the block-range figures it reads as a general mean; over blocks 1 to 6 it is 0.69. |
| 6 | Odd directions | 279 of 768 at layer 1, 128 at layer 6, 8 at layer 12 | d_minus 279 (layer 1), 128 (layer 6), 8 (layer 12), d = 768, for fit seeds 0 and 2; fit seed 1: 281, 128, 8 | runs/x2/perlayer_reg_s0/results/e1_layers.csv rows layer=1, 6, 12 (column d_minus); T_structure_per_layer.md:35, 40, 46 | **MATCH** | Registers model, paper protocol, fit seed 0. Model-specific: plain DINOv2 gives 278, 98, 7 (runs/x2/perlayer_plain_s0). Name the fit seed. |
| 7 | Tax | 0.42 of MLP weights in blocks 1 to 6 | tax_formula_mean 0.4186713842698085 (3 fit seeds, std 0.0002); realised saving identical | results/paper/tables/T_structure_summary.csv row dinov2_vitb14_reg,1-6; T_f1_ranges.md:11; numbers.json f1.reg.first_half.tax | **MATCH (rounded)** | Mean of the per-block tax (fc1 and fc2 averaged) over blocks 1 to 6. |
| 8 | Tax | 0.11 in blocks 7 to 12 | tax_formula_mean 0.10940306863667053 (3 fit seeds) | results/paper/tables/T_structure_summary.csv row dinov2_vitb14_reg,7-12; T_f1_ranges.md:12 | **MATCH (rounded)** |  |
| 9 | Breaking share | About 0.19 to 0.22 in early blocks, against about 0.42 to 0.43 for matched random structure | Registers model, mean fc1 share over blocks 1 to 6: s1 0.2230 (null 0.4271), 3 fit seeds. Plain DINOv2: 0.1938 (null 0.4160). Per block (registers, fit seed 0) s1 runs from 0.155 (block 6) to 0.267 (block 1) | results/paper/tables/T_structure_summary.csv columns s1_mean, s1_null_mean, rows dinov2_vitb14_reg,1-6 and dinov2_vitb14,1-6; T_f1_ranges.md:9, 11; numbers.json f1.reg.first_half.s1, f1.plain.first_half.s1 | **CONTEXT** | The 0.19 and 0.42 ends come from plain DINOv2 (E1b gave the same: 0.194, null 0.416). For the registers model the figure is 0.223 against 0.427, which misses the pre-registered 0.2 bar (decision row 4 AMBER on all 3 fit seeds). |
| 10 | Breaking share | Block 1 fc1: 0.27 against 0.49 | s1 0.2670600731187647, s1_null 0.49425580842939026 (fit seed 0); seeds 1 and 2: 0.266 and 0.267, null 0.495 and 0.494 | runs/x2/perlayer_reg_s0/results/e1_mlp.csv:2 columns s1, s1_null; T_structure_per_layer.md:35 | **MATCH (rounded)** |  |
| 11 | Operator fit | 10.6x better than a random involution at layer 1 | 10.56639473986125 is the E1b primary arm: plain DINOv2 (dinov2_vitb14), E1b protocol (39,000-image bank, 8,000 queries, original 77.35), mean over layers 1 to 6, not layer 1. Registers model, paper protocol: mean over layers 1 to 6 is 13.06 / 13.07 / 13.07 (fit seeds 0, 1, 2); layer 1 alone 50.73 / 50.88 / 50.74 | runs/e1b_perlayer_b14/results/e1_meta.json decision.rows[0].value; PROJECT_CONTEXT.md:27; registers: runs/x2/perlayer_reg_s0/results/e1_meta.json decision.rows[0].value and decision.per_layer[0].r1; T_structure_per_layer.md:35 | **CONTEXT** | Other model, older protocol and other aggregation. 'At layer 1' is wrong for every source: 10.6 is a mean over layers 1 to 6. Plain DINOv2 under the paper protocol: 10.48 (mean) and 46.6 (layer 1). |
| 12 | Operator fit | e(A)/e(I) = 0.16 | 0.16011080803711272 is the E1b primary arm (plain DINOv2, E1b protocol), mean over layers 1 to 6. Registers model, paper protocol: 0.1698 / 0.1698 / 0.1697 (mean over layers 1 to 6, fit seeds 0, 1, 2); layer 1 alone 0.0406 for all three seeds | runs/e1b_perlayer_b14/results/e1_meta.json decision.rows[1].value; runs/x2/perlayer_reg_s0/results/e1_meta.json decision.rows[1].value and decision.per_layer[0].r2 | **CONTEXT** | Same issue as the row above. |
| 13 | Operator drift | Neighbouring layers agree at 0.80 (chance 0.49) | agree_full 0.8023 (null 0.4881) fit seed 0; 0.8022 (0.4873) seed 1; 0.8024 (0.4875) seed 2 | results/paper/tables/T_f2_drift.md:12 to 14; numbers.json f2.perlayer_reg_s0.agree and .agree_null; results/paper/scripts/f2_dinov2/f2_neighbours.csv | **MATCH (rounded)** | tr(A_l A_(l+1))/d, mean over neighbouring layer pairs. Plain DINOv2: 0.806 (0.508). |
| 14 | Artefact tokens | Removing the top 1% by norm cuts late-layer mirror error 2.27x | 2.2722933297201893: masking cut over layers 9 to 11 for plain DINOv2 (mean error 1.315 with all tokens, 0.579 without the top 1%) | runs/f3/f3_dinov2_vitb14.json p1.value (= p1.masking_cut_late); T_prereg.md:35; results/STATUS.md:13 | **CONTEXT** | Plain DINOv2, not the registers model. The row does not name the model, so under the Section 3 header it reads as the registers model. |
| 15 | Artefact tokens | registers model 1.30x | 1.2964747230982945: late-layer error (layers 9 to 11) over early-layer error (layers 1 to 6), registers model. The masking cut for the registers model is 0.81 (masking slightly raises its error) | runs/f3/f3_dinov2_vitb14_reg.json p1.value, p1.late_over_early, p1.masking_cut_late; T_prereg.md:36 | **CONTEXT** | A different ratio from the 2.27x it sits beside; it is not a masking cut. |
| 16 | Artefact tokens | Pre-registered pass | P1.plain PASS (bar >= 2), P1.registers PASS (bar <= 2), G3 PASS | results/paper/tables/T_prereg.md:35 to 37; runs/f3/f3_gate_G3.json | **MATCH** |  |
| 17 | Energy rule (F4) | 16 of 36 correct, needed 27. Pre-registered fail | n_hits 16, n_cases 36, fraction 0.4444, threshold 0.75 (27 of 36), verdict FAIL | runs/f4/f4_p2.json primary_prereg_models; T_prereg.md:38; numbers.json p2.value | **MATCH** | 36 cases are 3 models (DINOv2, DINOv2-reg, DeiT-III) x 12 blocks; the registers model alone scores 4 of 12. |
| 18 | Zero-shot surgery, 6 blocks | Mirror (per-layer, MLP) 77.09 ± 0.003 (3 fit seeds) | mean 77.09133227666219, sample std (n - 1) 0.0030559478279285263, n = 3; per seed 77.094, 77.088, 77.092 | runs/x2/perlayer_reg_s{0,1,2}/results/e1_surgery.csv row k=6, variant=surgery (line 6 in s0); numbers.json main.reg.k6.mirror_mlp.top1 and .top1_std; T_main_zeroshot.md:24 | **MATCH (rounded)** | The std is right (0.003 points, fit-seed spread only). The ablation rebuild of the same setting gives 77.08 ± 0.03 (T_ablations); STATUS says quote X2. Its model_params (74,730,315) exclude the 3,538,944 rotation parameters (see Counting error). |
| 19 | Zero-shot surgery, 6 blocks | plain SVD 8.77 | 8.76933311422666 (mean of 8.768, 8.784, 8.756) | runs/x2/perlayer_reg_s{0,1,2}/results/e1_surgery.csv row k=6, variant=svd; numbers.json main.reg.k6.svd_mlp.top1; T_main_zeroshot.md:25 | **MATCH (rounded)** |  |
| 20 | Zero-shot surgery, 6 blocks | random structure 3.95 | 3.945999965071678 (3 fit seeds, std 0.11) | runs/x2/perlayer_reg_s{0,1,2}/results/e1_surgery.csv row k=6, variant=random; numbers.json main.reg.k6.random_mlp.top1; T_main_zeroshot.md:26 | **MATCH (rounded)** |  |
| 21 | Ablations | Wrong operator 3.88 | A1 random_operator, k = 6: 3.877333303292593 ± 0.441 (s0 3.916, s1 4.298, s2 3.418) | runs/a/dinov2_vitb14_reg/A1_s{0,1,2}.csv line 6 (variant=random_operator, k=6); T_ablations.md:20; numbers.json a.reg.A1.random_operator.k6.top1 | **MATCH (rounded)** | This is the random operator. Other wrong operators score higher but keep more weights: identity 24.63 (1.18x), one operator for all layers 20.48 (1.17x). |
| 22 | Ablations | random pairing 28.96 | A2 random_pairing, k = 6: 28.95866632461548 ± 3.498 (s0 29.646, s1 32.062, s2 25.168) | runs/a/dinov2_vitb14_reg/A2_s{0,1,2}.csv line 3 (variant=random_pairing, k=6); T_ablations.md:31 | **MATCH (rounded)** | Large seed spread (std 3.5); quote it with the std. |
| 23 | Ablations | pairing by weight cosine 24.69 | A2 generic, k = 6: 24.685999751091003 ± 0.047 (s0 24.652, s1 24.740, s2 24.666) | runs/a/dinov2_vitb14_reg/A2_s{0,1,2}.csv line 5 (variant=generic, k=6); T_ablations.md:33 | **MATCH (rounded)** |  |
| 24 | Ablations | 100 fit images 77.03 | A3 n100, k = 6, fit seed 0: 77.03199982643127 | runs/a/dinov2_vitb14_reg/A3_s0.csv:2; T_ablations.md:42; numbers.json a.reg.A3.n100.k6.top1 | **MATCH (rounded)** | One fit seed. Compare with the same script's 2,000-image value (77.086), not with X2's 77.09. |
| 25 | Exact version | 6 blocks, MLP plus attention: 77.37 | V-exact, MLP + attention, k = 6: 77.36799716949463 (fit seed 0, n = 1) | runs/m1/m1_knn.csv:5 (k=6, variant=vexact); numbers.json m1.k6.vexact.top1 | **MATCH (rounded)** | Paper protocol (runs/m1/m1_meta.json: bank 100000, query 50000). One fit seed, against three for 77.09. |
| 26 | Exact version | All 12 blocks, before healing: 53.85 | 53.850001096725464 (k = 12, V-exact, fit seed 0) | runs/m1/m1_knn.csv:15; runs/m3_vexact/k12/knn.json zero_shot.top1 | **MATCH** |  |
| 27 | Attention | 130 of 144 heads mirror onto themselves; 7 twin-head pairs | n_self_total 130, n_pairs_total 7 (130 + 2 x 7 = 144); per block 12, 10, 10, 12, 10, 10, 10, 10, 12, 10, 12, 12 | runs/m1/m1_heads.csv; T_attention.md:11 to 22; numbers.json m1.heads.n_self_total, m1.heads.n_pairs_total | **MATCH** | Fit seed 0. |
| 28 | Attention | Block 1 attention breaking share 0.28 against 0.49 random | s_all 0.2831163669660666, s_all_random 0.4939001148308489 | runs/m1/m1_heads.csv:2 (block 1); T_attention.md:11 | **MATCH (rounded)** |  |
| 29 | Healing pilot, 9 blocks | Mirror 78.81, SVD 77.57, random 73.28 | healed top-1: Mirror 78.81399989128113, SVD 77.57399678230286, random 73.27799797058105 | runs/m3_pilot/{mirror,svd,random}/knn.json healed.top1; T_healing.md:13 to 15 | **MATCH (rounded)** | k = 9, MLP plus attention, Mirror arm is V-local (per-layer), one seed. |
| 30 | Healing pilot, 9 blocks | Pre-registered 3-point bar failed (it needed 80.57, above the original) | P3.b = 1.24 against >= 3: FAIL; 77.574 + 3 = 80.574 (computed), above the original 78.936 | T_healing.md:22; numbers.json m3.p3.b.value, m3.p3.b.verdict | **MATCH (rounded)** |  |
| 31 | Full conversion, healed | 78.31 | 78.31199765205383 (A100, git 5fb53c0744, healing seed 0) | runs/m3_vexact/k12/knn.json healed.top1; T_healing.md (V-exact table, k = 12 row) | **MATCH (rounded)** | One healing seed. |
| 32 | Full conversion, healed | rerun of the same seed on another GPU: 78.34 | 78.33799719810486 (RTX 5090, git 6f35281c33) | runs/m3_vexact_5090/k12/knn.json healed.top1; numbers.json e3.v0.healed | **MATCH (rounded)** |  |
| 33 | Full conversion, healed | 100% flip consistency | flip_consistency 100.0 in the fp32 check (TF32 off, 5,000 validation queries, 10,000-image train bank). Paper-protocol k-NN (bf16, 50,000 queries): 99.76399540901184 (5090 rerun 99.742) | runs/m3_vexact/k12/flipcheck.json flip_consistency, n_query, n_bank, dtype; runs/m3_vexact/k12/knn.json healed.flip_consistency | **CONTEXT** | The 100% is the fp32 exactness check on a 5k subset, not flip consistency as Section 4.6 defines it (all 50k validation images), which is 99.76% in bf16. |
| 34 | Full conversion, healed | 76.5% of parameters kept | model_params_frac 0.7646182036976261 (66,203,360 of 86,583,552); converted matrices 0.760 | runs/m3_vexact/k12/summary.json method_info.model_params_frac; T_healing.md (V-exact table, k = 12 row) | **MATCH (rounded)** | Whole model; biases and embeddings counted dense. |
| 35 | Downstream, 6 blocks healed exact | Linear probe 80.86 against 80.99 | healed_vexact_k6 80.858; original 80.988 | runs/t2/t2_linear.csv:6 to 7; T_tier2.md:21 to 22 | **MATCH (rounded)** | Healed model probed on the RTX 5090, original on the A100 (different git hashes); one seed. |
| 36 | Downstream, 6 blocks healed exact | ADE20K 47.51 against 47.59 | mIoU healed_vexact_k6 47.50705421032515; original 47.592269358673775 (8,000 iterations, batch 16, crop 448) | runs/t2_seg/t2_ade20k.csv:2 to 3; T_tier2.md:80 | **MATCH (rounded)** | One run each; the planned ADE20K rerun of the original (run-to-run spread) has not run, so the 0.09 gap has no error bar. |
| 37 | Downstream, 6 blocks healed exact | transfer within about 0.5 points | Healed V-exact k = 6 against the original: CIFAR-100 90.54 vs 91.40 (0.86 below), DTD 83.19 vs 83.56 (0.37), Pets 95.67 vs 95.99 (0.33), Flowers-102 99.69 vs 99.69 (0.00, ceiling). The 0.51 figure is the healed V-local k = 9 pilot (CIFAR-100 90.89 vs 91.40) | runs/t2/t2_transfer.csv:2, 3, 5, 9, 10, 16, 17, 23, 24; T_tier2.md:43, 45, 50, 57, 64; results/STATUS.md:36, 39 | **MISMATCH** | For the row's model the largest drop is 0.86, not about 0.5. |
| 38 | Generality | DeiT-III and MAE pass | P4.deit3.all PASS (share 0.184, ratio 10.66); P4.mae.all PASS (share 0.232, ratio 40.76). Decision-table outcome AMBER for both | results/paper/tables/T_generality_p4.csv; T_generality.md:17 to 18, 27 to 32 | **MATCH** | They pass the pre-registered structure check P4, not the full decision table (both AMBER). Paper protocol, fit seed 0. |
| 39 | Generality | DINOv2-L has the strongest structure | Fit ratio e(null)/e(A) 79.41, the highest of all models (next MAE 40.76). On other measures it is not top: e(A)/e(I) 0.079 (MAE 0.048), tax over the first half 0.414 (MAE 0.448), breaking share 0.252 | runs/g/dinov2_l_reg_s0/results/e1_meta.json decision.rows[0].value; T_generality.md:21; T_f1_ranges.md:15, 21 | **MATCH** | True for the operator fit ratio; name the measure. DINOv2-L/14 with registers, k = 12 of 24, one fit seed. |
| 40 | Generality | SigLIP2 weak, as predicted | Outcome KILL; fit ratio 2.38, d-/d 0.138, twin fraction 0.285. Pre-registered P4 sub-check (SigLIP2 breaking share above DINOv2-reg's): 0.137 vs 0.223, FAIL | results/paper/tables/T_generality_p4.csv row P4.siglip2; T_generality.md:19, 33; results/STATUS.md:23 | **MISMATCH** | The structure is weak, but 'as predicted' contradicts the recorded pre-registered verdict (FAIL). The weakness shows in decision rows 1 to 3, which were not the pre-registered measure. |
| 41 | Generality | DINOv2-S loses 18 points under surgery | original 74.45600032806396, surgery 56.03399872779846 at k = 6: drop 18.42 (A100 run 56.00) | runs/g/dinov2_s_reg_s0/results/e1_surgery.csv:2 to 3; T_generality.md:20 | **MATCH (rounded)** | DINOv2-S/14 with registers, MLP-only V-local surgery, one fit seed. |
| 42 | Generality | On DINOv2-L and DeiT-III, plain SVD stays within 0.7 to 0.9 points of Mirror | DINOv2-L: Mirror 79.256 vs SVD 78.568, margin 0.688 (k = 12 of 24); DeiT-III: 81.548 vs 80.670, margin 0.878 (k = 6) | runs/g/dinov2_l_reg_s0/results/e1_surgery.csv:3 to 4; runs/g/deit3_b16_s0/results/e1_surgery.csv:3 to 4; T_generality.md:17, 21 | **MATCH (rounded)** |  |
| 43 | Weights only | Operator from weights matches the data-driven one at 0.90 (pass) | p5.part1.value 0.8999972812677988, PASS (bar 0.8; chance 0.652); d+ 392 from weights vs 400 from data | runs/d/d_datafree.json p5.part1; T_prereg.md:56 | **MATCH (rounded)** | Layer-0 (input) operator only: mean cosine of principal angles of the even subspace. Registers model, fit seed 0 operators. |
| 44 | Weights only | pairing from weights 47% (needed 70%, fail) | p5.part2.value 0.4657534246575342 (given the data-driven A_0), FAIL; fully data-free 0.4409, FAIL | runs/d/d_datafree.json p5.part2, p5.part2_strict; T_prereg.md:57 to 58 | **MATCH (rounded)** | Block 1 only, as pre-registered. |
| 45 | Speed | 0.70 to 0.91x of the original's speed | A100 (runs/m2_clean), bf16, torch.compile, batch 256, Mirror variants against dense in the same implementation: 0.6992 (V-exact k = 12) to 0.9100 (V-local MLP only k = 6). RTX 5090: 0.6346 to 0.8919. A100 batch 1, compiled: 0.495 to 0.843 | results/paper/tables/T_latency.csv (source runs/m2_clean/m2_latency.csv), rows batch=256, mode=compile; T_latency_5090.csv (runs/m2_5090/m2_latency.csv); results/STATUS.md:45, 48 | **MATCH (rounded)** | Holds for the A100 at batch 256 only; say so. The reference is the dense model in the same implementation (0.995x the plain original). |
| 46 | Speed | 88.9 ms against 75.7 ms | RTX 5090 profile, batch 256, compiled: V-exact k = 12 median 88.93957 ms; dense 75.68418 ms (ratio 0.851) | runs/m2_profile/m2_profile.csv; T_latency_profile.md:12, 24; numbers.json m2p.reg.vexact.k12.b256.compile.latency | **CONTEXT** | Different GPU from the 0.70 to 0.91 range (A100) and one variant only (V-exact, k = 12). The 5090 latency run gives 0.859x for the same variant. |
| 47 | SVD collapse check (gate G1) | Not a bug | G1 PASS (full-rank SVD reproduces the model, max CLS difference 5.12e-05; ranks within 1% of budget) | results/paper/tables/T_prereg.md:61; runs/x1/x1_svd_check.json; results/STATUS.md:12 | **MATCH** |  |
| 48 | Counting error | one d x d basis change per block, not counted: about 3.5M extra parameters at 6 blocks | 6 x 768 x 768 = 3,538,944 (computed). Recorded as rotation_params 3538944 for Mirror V-local MLP only, k = 6, and excluded from model_params | results/paper/tables/T_main_zeroshot.csv rows k=6, variant=mirror_mlp, mirror_mlp_m1, vlocal (column rotation_params); results/paper/tables/cost_model.md:13 | **MATCH (rounded)** | Counted, the Mirror k = 6 MLP-only model keeps 20,001,574 converted MLP parameters against 16,446,720 for its SVD baseline (21.6% more; fit seed 0), and 78,269,259 against 74,712,320 for the whole model (4.8% more). The timed M2 implementation stores 4,128,768 (7 x d^2) for V-local k = 6 and 589,824 for V-exact k = 6 (T_latency.csv). |

## Proposed corrected wording for Section 3

For every Section 3 row that is not a clean match. Rows marked "precision only" are matches where naming the seed or the budget consequence avoids a later misreading.

**Header sentence** (line 60)

- Now: All are DINOv2-B with registers (`dinov2_vitb14_reg`) under the paper protocol (Section 4.6) unless stated.
- Proposed: All are DINOv2-B with registers (`dinov2_vitb14_reg`) under the paper protocol (Section 4.6), fit seed 0 and one run, unless a model, seed count or device is stated.

**Twins** (line 65)

- Now: Block 1: 1,168 twin pairs and 736 self-twins of 3,072 neurons (76% in pairs). Blocks 1 to 6: 63% in pairs. Blocks 7 to 12: 13%. Mean twin correlation 0.80
- Proposed: Block 1 (fit seed 0): 1,168 twin pairs and 736 self-twins of 3,072 neurons (76% in pairs), mean twin correlation 0.80. Blocks 1 to 6: 63% in pairs, mean twin correlation 0.69. Blocks 7 to 12: 13% in pairs (pair fractions are means over 3 fit seeds)

**Odd directions (precision only)** (line 66)

- Now: 279 of 768 at layer 1, 128 at layer 6, 8 at layer 12
- Proposed: 279 of 768 at layer 1, 128 at layer 6, 8 at layer 12 (fit seed 0; fit seed 1 gives 281 at layer 1)

**Breaking share** (line 68)

- Now: About 0.19 to 0.22 in early blocks, against about 0.42 to 0.43 for matched random structure (block 1 fc1: 0.27 against 0.49)
- Proposed: fc1 share, mean over blocks 1 to 6 (3 fit seeds): 0.22 against 0.43 for matched random structure, just above the pre-registered 0.2 bar (AMBER); plain DINOv2: 0.19 against 0.42. Block 1 fc1: 0.27 against 0.49

**Operator fit** (line 69)

- Now: 10.6x better than a random involution at layer 1; e(A)/e(I) = 0.16
- Proposed: Held-out error 13.1x lower than a matched random involution and e(A)/e(I) = 0.17, both means over layers 1 to 6 (3 fit seeds); at layer 1 alone 50.7x to 50.9x and 0.041. (10.6x and 0.16 are E1b: plain DINOv2, older protocol, mean over layers 1 to 6)

**Artefact tokens** (line 71)

- Now: Removing the top 1% by norm cuts late-layer mirror error 2.27x; registers model 1.30x. Pre-registered pass
- Proposed: Plain DINOv2: removing the top 1% highest-norm tokens cuts the mean mirror error over layers 9 to 11 by 2.27x. Registers model: the late-layer error (layers 9 to 11) is only 1.30x the early-layer error (layers 1 to 6). Both pre-registered P1 bullets pass (G3 pass)

**Full conversion, healed** (line 78)

- Now: 78.31 (rerun of the same seed on another GPU: 78.34). 100% flip consistency. 76.5% of parameters kept
- Proposed: 78.31 (rerun of the same seed on another GPU: 78.34; one healing seed). Flip consistency 100% in the fp32 check (5,000 validation queries) and 99.76% in the bf16 k-NN run on all 50,000 validation images. 76.5% of the whole model's parameters kept

**Downstream, 6 blocks healed exact** (line 79)

- Now: Linear probe 80.86 against 80.99; ADE20K 47.51 against 47.59; transfer within about 0.5 points
- Proposed: Linear probe 80.86 against 80.99; ADE20K 47.51 against 47.59 (one run each, no rerun spread yet); transfer within 0.86 points (CIFAR-100 90.54 against 91.40; DTD 0.37 and Pets 0.33 below; Flowers-102 at ceiling). Healed model measured on the RTX 5090, originals on the A100

**Generality** (line 80)

- Now: DeiT-III and MAE pass; DINOv2-L has the strongest structure; SigLIP2 weak, as predicted; DINOv2-S loses 18 points under surgery. On DINOv2-L and DeiT-III, plain SVD stays within 0.7 to 0.9 points of Mirror
- Proposed: DeiT-III and MAE pass the pre-registered structure check P4 (both AMBER overall); DINOv2-L has the best operator fit (fit ratio 79.4, k = 12 of 24); SigLIP2 has weak structure (fit ratio 2.4, KILL), but its pre-registered P4 sub-check failed (breaking share 0.137, below DINOv2-reg's 0.223, not above); DINOv2-S loses 18.4 points under surgery. On DINOv2-L and DeiT-III, plain SVD stays within 0.69 and 0.88 points of Mirror. All fit seed 0

**Speed** (line 82)

- Now: 0.70 to 0.91x of the original's speed; 88.9 ms against 75.7 ms
- Proposed: Measured against the dense model in the same implementation (bf16, torch.compile, batch 256): 0.70x to 0.91x on the A100 and 0.63x to 0.89x on the RTX 5090. Example (RTX 5090 profile, V-exact k = 12): 88.9 ms against 75.7 ms for dense

**Counting error (precision only)** (line 84)

- Now: about 3.5M extra parameters at 6 blocks
- Proposed: 3.54M extra parameters at 6 blocks (6 x 768 x 768; recorded as rotation_params but left out of model_params). Counted, Mirror-local MLP-only at k = 6 keeps 21.6% more converted parameters than its SVD baseline (4.8% of the whole model), far outside the 1% budget rule of Section 4.7

## configs/models/

Not present at audit time: `configs/` holds only run configs (e1, e1b, x2, g, m3, e3). Section 4.2 asks for `configs/models/*.yaml` with hub IDs, revisions and quoted training recipes; that half of MUST-00 is still open.

## Other observations

- With the per-block basis changes counted, Mirror-local MLP-only at k = 6 keeps 20.0M converted MLP parameters against 16.4M for its SVD baseline (21.6% more; whole model 78,269,259 against 74,712,320, 4.8% more), so the 77.09 against 8.77 comparison is not parameter-matched under the Section 4.7 rule. V-exact needs no basis change inside the span.
- Several headline numbers are single runs: the exact 6-block 77.37, the 100-image ablation, the healing pilot, the full conversion (one healing seed, replicated on a second GPU), every Tier-2 number and every generality model.
- `results/STATUS.md` line 36 is the origin of "within 0.51"; it describes the healed V-local 9-block pilot, while line 39 gives the healed V-exact 6-block transfer values.
