# STATUS (paper experiments)

Updated 2026-11-09 about 07:30 UTC (13:00 IST). The A100 went offline at about 21:40 UTC on 8 November; work moved to a rented RTX 5090 (vast.ai, Estonia) at about 04:30 UTC on 9 November. Same code, same torch build (2.11.0+cu128), same seeds and the same ImageNet subsets (rebuilt deterministically); results that ran on both GPUs agree to within 0.03 points.
committed as 9804f06 before any gated run. Every number below is in `results/paper/` (tables, `numbers.json`
with source file, seed and commit for each value); raw outputs in `runs/`; large artefacts (operators,
statistics, healed checkpoints, 2 GB) in `artifacts/a100/` on the laptop only. Protocol: operators fitted on
ImageNet train (fit seeds 0, 1, 2), k-NN with a 100k train bank and 50k validation queries, bf16 features.

## Pre-registered checks
| Check | Value | Bar | Verdict |
|---|---|---|---|
| G1 SVD collapse is not a bug (X1, X1b) | four checks clean | | PASS |
| P1 plain: masking top 1% tokens cuts late-layer error | 2.27x | >= 2x | PASS |
| P1 registers: late-layer error vs early | 1.30x | <= 2x | PASS |
| G3 | | one P1 bullet passes | PASS |
| P2 energy rule predicts single-block winner (F4, 3 models x 12 blocks) | 16 of 36 (0.44) | >= 0.75 | FAIL |
| P3.a healed mirror minus healed random (k-NN) | +5.54 | >= 2.0 | PASS |
| P3.b healed mirror minus healed SVD (k-NN) | +1.24 | >= 3.0 | FAIL |
| P3.c original minus healed mirror | 0.12 | <= 2.0 | PASS |
| G2 (P3.a and P3.b) | | both | FAIL: healing grid not started; framing goes to Dawar |
| P4 DeiT-III, MAE | pass | | PASS |
| P4 SigLIP2 breaking share above DINOv2-reg | 0.137 vs 0.223 | higher | FAIL (structure near trivial; rows 1 to 3 much weaker) |
| P5 part 1 layer-0 operator from weights | 0.90 | >= 0.8 | PASS (chance about 0.65 to 0.73) |
| P5 part 2 weight-only twin pairing | 46.6% | >= 70% | FAIL |

## Headline results (registers model unless stated; original k-NN 78.94, linear probe 80.99)
- E1b decision table under the paper protocol: plain DINOv2 GO on all 3 seeds; registers model AMBER on all 3 seeds (row 4 only).
- Zero-shot surgery, 6 blocks, MLP only (Mirror V-local, MLP only; canonical source X2, `T_main_zeroshot`): mirror 77.09 / 77.09 / 77.09 (fit seeds 0, 1, 2); SVD 8.8; random 3 to 4. The ablation script rebuilds this setting from the float32 operator file and gives 77.11 / 77.04 / 77.10 (M1 and the linear probe give 77.11 at fit seed 0), a numerical difference of at most 0.044 points; quote X2. Linear probe: mirror 79.08, SVD 20.92, random 11.25.
- V-exact (exactly mirror-equivariant over the span), 6 blocks MLP plus attention: k-NN 77.37, linear 79.51; full conversion (12 blocks) 100% flip consistency, k-NN 53.9.
- Healing, 9 blocks, MLP plus attention: mirror 78.81 k-NN / 80.71 linear; SVD 77.57 / 79.30; random 73.28 / 74.99.
- Plain DINOv2, 6 blocks: mirror 74.65 vs SVD 66.14 (k-NN, mean of 3 fit seeds; fit seed 0 alone 74.68 vs 66.23); linear 76.80 vs 69.48 (fit seed 0).
- Generality: DeiT-III AMBER (mirror 81.55 vs SVD 80.67); MAE strongest structure (fit ratio 40.8); SigLIP2 KILL (weak structure, as predicted).
- Ablations (k = 6): wrong operator collapses (shared over all layers 20.5, identity 24.5, random 3 to 4); wrong pairing collapses (random 25 to 32, self-twins 13.7, generic weight-cosine 24.7); Hungarian without the involution constraint matches mirror (77.2 to 77.3); 100 fit images suffice (77.03).
- X4 folding: algebra exact in float64 on the registers model; fp32 bar missed through rounding; folded model k-NN 78.98.
- Transfer probes (linear, 4 datasets) complete: healed mirror within 0.5 points of the original on CIFAR-100, DTD, Flowers and Pets, ahead of healed SVD and healed random (`runs/t2/t2_transfer.csv`). Flowers-102 is at ceiling (original 99.69%, several variants tie with it exactly), so it does not separate the methods.
- V-exact full conversion, healed (`runs/m3_vexact/k12/`): all 12 blocks exactly mirror-equivariant, k-NN 78.31 (original 78.94, zero-shot 53.85), fp32 flip consistency 100% (exactness 1.4e-05, tolerance 1e-4), 0.765 of the parameters. One seed; no matched healed baseline at k = 12.
- V-exact k = 6 healed (5090, `runs/m3_vexact/k6/`): k-NN 78.79 (original 78.93 on this GPU, zero-shot 77.37), exactness 4.6e-06; 0.799 of the parameters.
- ADE20K linear segmentation (Tier 2, 5090, `runs/t2_seg/`, 8,000 iterations, batch 16, 448 crops, full validation): original mIoU 47.59; V-exact k = 6 (zero-shot, exactly equivariant over blocks 1 to 6) 47.19. Other variants running.
- G, DINOv2-L/14 reg (5090, k = 12 of 24): AMBER. Structure strongest of all models (fit ratio 79.4), surgery 79.26 vs original 80.60, but SVD at matched parameters also survives (78.57), so the margin over SVD is only 0.69. The large SVD collapse seen on DINOv2-B does not happen on DINOv2-L.
- G, DINOv2-S/14 reg (5090): PIVOT, surgery 56.03 vs original 74.46, SVD 10.24; reproduces the A100 run (56.00).
- Ablations, more seeds and k = 9 (raw rows in `runs/a/`, on the VM until pulled): A4 and A5 for fit seeds 1 and 2 agree with seed 0 (last 6 blocks 78.63 / 78.64 vs first 6 77.04 / 77.10; A5 falls smoothly from 78.94 at alpha 0 to 77.04 / 77.10 at alpha 1). A2 at k = 9, seed 0: mirror 76.37, Hungarian 76.52, random pairing 12.59, self-twins 12.06, generic weight-cosine 13.86. A1 at k = 9, seed 0: per-layer 76.37, shared over the span 67.68, shared over all layers 15.34, identity 16.74, random operator 3.03, SVD 5.83.
- M2 latency, clean re-run (`runs/m2_clean/`, GPU otherwise quiet apart from one small process of another project): the even/odd implementation matches the dense model it is built from (float64 16 of 16, fp32 16 of 16; bf16 differences are at the bf16 rounding level). Analytic saving 1.10x to 1.30x, but measured speed against the dense reference at batch 256 (compiled) is 0.70x to 0.91x for every mirror variant, so **no wall-clock gain with plain PyTorch ops**. SVD at matched parameters is also slower (0.92x to 0.98x); MLP width pruning at matched parameters is faster (1.13x to 1.17x). The extra gathers, butterflies and smaller GEMMs cost more than they save; a fused kernel would be needed for a latency claim.

## Not done yet
- M3 grid (on hold after G2), M4.
- Strong baselines, zero-shot (asvd, slicegpt, prune_act at the Mirror V-local MLP-only budget, k = 6 and 9): code and CPU tests ready (`scripts/strong_baselines.py`, `mirror/baselines.py`, `tests/test_strong_baselines.py`), not run; waits for approval to use the 5090.
- Running on the 5090 (07:30 UTC): ADE20K (MLP-only mirror and SVD at k = 6), V-exact k = 12 healed again (second device; the A100 checkpoint was lost with that VM), linear and transfer probes of healed V-exact k = 6, A1/A2 k = 9 seed 0 (A1 done, matches the A100 log to 0.03), ADE20K original rerun (GPU run-to-run spread). Waiting for uploads from the laptop (slow uplink, about 60 KB/s): fit seed 1 and 2 operators (A4/A5 and A1/A2 k = 9 for seeds 1 and 2), then the M3-pilot checkpoints (ADE20K healed variants). Not yet possible on the 5090: ADE20K of healed V-exact (code being added), M2 profile (needs a quiet GPU; another project's job runs on the rented instance).

## Decisions waiting for Dawar
1. Framing after G2 (report as is; a new pre-registered healing test where the ceiling does not bind; or recentre on the symmetry tax, exact conversion and zero-shot results).
2. P2 failed: the energy rule underrates mirror projection in early DINOv2 blocks; decide how C4 is stated.
3. Next GPU and its job list.

## Deviations from the plan
- k-NN features in bf16 for every variant alike; fits in fp32 and float64.
- GPU shared with another project: memory caps 6 to 20% per job; some jobs paused or reordered.
- X3 seeds at k = 6 only; A3 to A5 seed 0 only; ablations at k = 6 only.
- A3 largest fit set is 6,000 images (all fit seeds pooled), not 10,000.
- A1 at k = 9, seed 0 was rerun: shipping code to the VM at about 19:01 UTC unpacked the committed copy of `runs/a/dinov2_vitb14_reg/A1_s0.csv` over the file the job had just written, losing its 7 k = 9 rows (values are in `logs/stage8_more/job1.log`). The ship step now skips `runs/`, `results/` and `logs/`; an audit of every ablation CSV against its log found no other loss. The rerun reproduces the logged values exactly so far.
- V-exact healing differs from the M3-pilot recipe in two ways (independent code review, 2026-11-09): the embeddings (patch embedding, position grid, CLS, registers; about 0.65M entries) are trained and re-projected every step, while the pilot arms train only the converted blocks; and the targets are symmetrised (equal in expectation under random flips, lower variance), at full conversion the final target is the flip average, with the CLS normaliser measured on it. Compare V-exact with the pilot arms with this in mind.
- Per-process GPU memory caps on the 5090 are scaled by 2.5 (`MIRROR_MEM_SCALE`) so each job keeps the same limit in GB as on the 80 GB A100; caps never change results.
- M2: the first timings (`runs/m2/`) were contended and are superseded by `runs/m2_clean/`; the VM tree was exported from commit ced98c8 without git, so its metadata flags it as dirty.
- Consistency audit of the paper package (2026-11-09, `scripts/collect_results.py`): every MLP-only number is quoted from X2; the M1, ablation and Tier-2 rebuilds of that setting reload the float32 operator file and differ by up to 0.04 points (numerical), and are marked. A4 block selection and some A1/A2 alternatives are not parameter-matched (last 6 blocks keep 1.53x the MLP weights of first 6; ratios in `T_ablations`). M2 Mirror V-local, MLP only, k = 6, batch 64 compiled (1.247x) is a probable timing outlier, kept and flagged. The M2 analytic speed-up of V-local is below the collector's because the timed implementation stores Wo dense. Flowers-102 is at ceiling.
