# STATUS (paper experiments)

Updated 2026-11-10 about 04:00 UTC. **The RTX 5090 in South Korea (vast.ai instance 59906003) stopped answering some time after 18:45 UTC on 9 November (connection refused on both ssh routes at 03:55 UTC on 10 November); its disk holds the caches, the E3 run in progress and the first finished E3 arm (healed SVD, raw targets, not yet pulled). Dawar to check the vast.ai console (balance, instance state) and restart it.** History: the A100 went offline at about 21:40 UTC on 8 November; work moved to a 5090 in Estonia at about 04:30 UTC on 9 November; that GPU was taken by another renter after a stop, so its /workspace was copied with vast.ai's instance copy to the South Korea 5090 (9 November, about 15:40 to 19:00 UTC). Same code, torch build, seeds and ImageNet subsets throughout; results run on two GPUs agree to within 0.03 points. Every result that was on the Estonian disk has been pulled and committed (7821764).
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
| E3 / G5 (PREREG_E3.md, k = 12 matched healed baselines) | | E3.a >= 2.0, E3.b >= 3.0, seed condition | signed off 2026-11-09 (aeb57be); running: 1 of 4 arms finished on the GPU, not yet pulled |
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
- Transfer probes (linear, 4 datasets) complete: healed mirror within 0.51 points of the original on CIFAR-100, DTD, Flowers and Pets (CIFAR-100 0.51 below), ahead of healed SVD and healed random (`runs/t2/t2_transfer.csv`). Flowers-102 is at ceiling (original 99.69%, several variants tie with it exactly), so it does not separate the methods.
- V-exact full conversion, healed (`runs/m3_vexact/k12/`): all 12 blocks exactly mirror-equivariant, k-NN 78.31 (original 78.94, zero-shot 53.85), fp32 flip consistency 100% (exactness 1.4e-05, tolerance 1e-4), 0.765 of the parameters. One seed; no matched healed baseline at k = 12.
- V-exact full conversion healed again on the 5090 (`runs/m3_vexact_5090/k12/`): k-NN 78.34 (A100 run 78.31), zero-shot 53.83 (53.85), fp32 flip consistency 100%: the headline replicates on a second device.
- Healed V-exact k = 6, Tier 2 (5090): linear probe 80.86 (original 80.99); transfer CIFAR-100 90.54 (91.40), DTD 83.19 (83.56), Flowers 99.69 (99.69, at ceiling), Pets 95.67 (95.99).
- ADE20K mIoU (5090, `runs/t2_seg/`): original 47.59; healed V-exact k = 6 47.51; V-exact k = 6 zero-shot 47.19; Mirror V-local MLP only k = 6 46.88; SVD at matched parameters, MLP only, k = 6 9.84.
- V-exact k = 6 healed (5090, `runs/m3_vexact/k6/`): k-NN 78.79 (original 78.93 on this GPU, zero-shot 77.37), exactness 4.6e-06; 0.799 of the parameters.
- G, DINOv2-L/14 reg (5090, k = 12 of 24): AMBER. Structure strongest of all models (fit ratio 79.4), surgery 79.26 vs original 80.60, but SVD at matched parameters also survives (78.57), so the margin over SVD is only 0.69. The large SVD collapse seen on DINOv2-B does not happen on DINOv2-L.
- G, DINOv2-S/14 reg (5090): PIVOT, surgery 56.03 vs original 74.46, SVD 10.24; reproduces the A100 run (56.00).
- Ablations, more seeds and k = 9 (raw rows in `runs/a/`): A4 and A5 for fit seed 1 agree with seed 0 (seed 2 still to run) (last 6 blocks 78.63 / 78.64 vs first 6 77.04 / 77.10; A5 falls smoothly from 78.94 at alpha 0 to 77.04 / 77.10 at alpha 1). A2 at k = 9, seed 0: mirror 76.37, Hungarian 76.52, random pairing 12.59, self-twins 12.06, generic weight-cosine 13.86. A1 at k = 9, seed 0: per-layer 76.37, shared over the span 67.68, shared over all layers 15.34, identity 16.74, random operator 3.03, SVD 5.83.
- M2 latency, clean re-run (`runs/m2_clean/`, GPU otherwise quiet apart from one small process of another project): the even/odd implementation matches the dense model it is built from (float64 16 of 16, fp32 16 of 16; bf16 differences are at the bf16 rounding level). Analytic saving 1.10x to 1.30x, but measured speed against the dense reference at batch 256 (compiled) is 0.70x to 0.91x for every mirror variant, so **no wall-clock gain with plain PyTorch ops**. SVD at matched parameters is also slower (0.92x to 0.98x); MLP width pruning at matched parameters is faster (1.13x to 1.17x). The extra gathers, butterflies and smaller GEMMs cost more than they save; a fused kernel would be needed for a latency claim.

- M2 profile (RTX 5090, quiet GPU, batch 256, bf16, `runs/m2_profile/`): Mirror rows execute 8.8 to 23.1% fewer GEMM multiply-adds than dense, but their GEMMs run at 104 to 161 TFLOP/s against 206 for dense (compiled), and extra memory movement and elementwise kernels add 1 to 7 ms per forward (compiled; 11 to 46 ms eager). At the dense GEMM rate their GEMMs would take 42.6 to 50.5 ms against 55.4 ms: the speed-up is there in the arithmetic, the plain-PyTorch kernels lose it.
- M2 latency on a second device (RTX 5090, `runs/m2_5090/`, same arguments as `runs/m2_clean/`): Mirror variants 0.63x to 0.89x of dense at batch 256 compiled (A100 0.70x to 0.91x); MLP width pruning 1.17x to 1.22x; equivalence float64 and fp32 16 of 16.
- MAE complete run (5090, `runs/g/mae_b16_s0/`): AMBER; original 13.73 k-NN, surgery 8.13, SVD 3.55, random 1.54 (all four arms; replaces the A100 partial run).
- Healed V-exact k = 12, ImageNet linear probe (`runs/t2/`): 80.10 (original 80.99).
- E3 signed off (PREREG_E3.md, aeb57be): healed SVD (raw and flip-averaged targets, ranks matched per block within 0.31%), healed random structure, V-exact healing seed 1, learning-rate check at 2e-4. Started 18:11 UTC on 9 November on the South Korea 5090 from a separate folder (`/workspace/e3run`); the healed SVD arm with raw targets finished at about 18:45 UTC, the rest was running when the instance stopped answering.

## Not done yet
- M3 grid (on hold after G2), M4.
- Strong baselines, zero-shot (asvd, slicegpt, prune_act at the Mirror V-local MLP-only budget, k = 6 and 9): code and CPU tests ready (`scripts/strong_baselines.py`, `mirror/baselines.py`, `tests/test_strong_baselines.py`), not run; waits for approval to use the 5090.
- On the South Korea 5090's disk, not yet pulled: E3 healed SVD with raw targets (finished) and whatever else E3 finished before the instance stopped.
- Still to run (`jobs/e3_k12.txt` from `/workspace/e3run`, then `jobs/stage22_korea.txt` from `/workspace/Mirror_CPAL` after shipping the current code; every job skips finished work): the rest of E3 (SVD with flip-averaged targets, V-exact seed 1, random structure, learning-rate probe of seed 0, Tier 2 of each); zero-shot strong baselines (asvd, slicegpt, prune_act, k = 6 and 9); flip-averaging baseline (only the original's plain row is done); ADE20K and transfer of healed V-exact k = 12; F7 teaser visuals; A2 k = 9 seed 1 (only the mirror row is done); A1/A2 k = 9 and A4/A5 for fit seed 2; ADE20K of the three healed M3-pilot models (checkpoints now on the instance); ADE20K rerun of the original (run-to-run spread). M2 latency and profile are done.
- The healing grid stays on hold after G2; E3 is the pre-registered follow-up (signed off). Zero-shot strong baselines: queued first in `jobs/stage22_korea.txt` (recommended to Dawar; zero-shot only).

## Decisions waiting for Dawar
1. Framing after G2 (report as is; a new pre-registered healing test where the ceiling does not bind; or recentre on the symmetry tax, exact conversion and zero-shot results).
2. P2 failed: the energy rule underrates mirror projection in early DINOv2 blocks; decide how C4 is stated.
3. The South Korea 5090 is unreachable: check the vast.ai console (balance and instance state) and restart it; send the new ssh command. Keep the Estonian instance until the Korea one is back and verified; it is no longer needed for results (all pulled), only as a second copy of the caches and of the other project's folder.

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
- MAE: the committed MAE run is the complete 5090 run (all four arms); the A100 partial run it replaces (no random arm; values within 0.04) stays in git history.
- M2 on the 5090: the profile and the latency run were run from a separate folder (`/workspace/m2run`, code at 6f2c918) while the copy of the main folder was still in progress; the latency run's equivalence checks used random images (the validation cache had not been copied yet); the A100 run's checks used validation images.
- E3 runs from a separate folder (`/workspace/e3run`, code at aeb57be, caches linked from the main folder) so that the copy could not overwrite its files.
- E3 validity, SVD ranks: the healing script's own rank check (mirror/heal.py, LowRankMLP.check) multiplies the factors in float32 and so reads full rank (for example 768 against a target of 310 in the M3 pilot) although the factorisation is low rank by construction; the collector therefore takes the float64 rank recorded by the Tier-2 rebuild of each SVD arm (runs/e3/t2_svd_*), so the SVD arms' validity is confirmed only once those probes have run. The check was left unchanged so that all arms are recorded by the same code.
