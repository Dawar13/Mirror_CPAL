# STATUS (paper experiments)

Updated 2026-11-08 about 18:45 UTC (00:15 IST on 9 November); the A100 session continues. Pre-registration `PREREG_E2.md`
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
- Zero-shot surgery, 6 blocks, MLP: mirror 77.11 / 77.04 / 77.10 (3 seeds); SVD 8.8; random 3 to 4. Linear probe: mirror 79.08, SVD 20.92, random 11.25.
- V-exact (exactly mirror-equivariant over the span), 6 blocks MLP plus attention: k-NN 77.37, linear 79.51; full conversion (12 blocks) 100% flip consistency, k-NN 53.9.
- Healing, 9 blocks, MLP plus attention: mirror 78.81 k-NN / 80.71 linear; SVD 77.57 / 79.30; random 73.28 / 74.99.
- Plain DINOv2, 6 blocks: mirror 74.68 vs SVD 66.23 (k-NN); linear 76.80 vs 69.48.
- Generality: DeiT-III AMBER (mirror 81.55 vs SVD 80.67); MAE strongest structure (fit ratio 40.8); SigLIP2 KILL (weak structure, as predicted).
- Ablations (k = 6): wrong operator collapses (shared over all layers 20.5, identity 24.5, random 3 to 4); wrong pairing collapses (random 25 to 32, self-twins 13.7, generic weight-cosine 24.7); Hungarian without the involution constraint matches mirror (77.2 to 77.3); 100 fit images suffice (77.03).
- X4 folding: algebra exact in float64 on the registers model; fp32 bar missed through rounding; folded model k-NN 78.98.
- Transfer probes (linear, 4 datasets) complete: healed mirror within 0.5 points of the original on CIFAR-100, DTD, Flowers and Pets, ahead of healed SVD and healed random (`runs/t2/t2_transfer.csv`).
- M2 latency, clean re-run (`runs/m2_clean/`, GPU otherwise quiet apart from one small process of another project): the even/odd implementation matches the dense model it is built from (float64 16 of 16, fp32 16 of 16; bf16 differences are at the bf16 rounding level). Analytic saving 1.10x to 1.30x, but measured speed against the dense reference at batch 256 (compiled) is 0.70x to 0.91x for every mirror variant, so **no wall-clock gain with plain PyTorch ops**. SVD at matched parameters is also slower (0.92x to 0.98x); MLP width pruning at matched parameters is faster (1.13x to 1.17x). The extra gathers, butterflies and smaller GEMMs cost more than they save; a fused kernel would be needed for a latency claim.

## Not done yet
- M3 grid (on hold after G2), M4.
- Running or queued on the A100 (2026-11-08 evening): A1/A2 at k = 9 (seeds 0, 1, 2), A4/A5 seeds 1 and 2, DINOv2-S and DINOv2-L generality; code being written for ADE20K linear segmentation. V-exact healing is wired and CPU-tested but not yet run (`scripts/heal_vexact.py`, `configs/m3_vexact.yaml`, job list `jobs/stage9_vexact.txt`: k = 12, then 6).

## Decisions waiting for Dawar
1. Framing after G2 (report as is; a new pre-registered healing test where the ceiling does not bind; or recentre on the symmetry tax, exact conversion and zero-shot results).
2. P2 failed: the energy rule underrates mirror projection in early DINOv2 blocks; decide how C4 is stated.
3. Next GPU and its job list.

## Deviations from the plan
- k-NN features in bf16 for every variant alike; fits in fp32 and float64.
- GPU shared with another project: memory caps 6 to 20% per job; some jobs paused or reordered.
- X3 seeds at k = 6 only; A3 to A5 seed 0 only; ablations at k = 6 only.
- A3 largest fit set is 6,000 images (all fit seeds pooled), not 10,000.
- M2: the first timings (`runs/m2/`) were contended and are superseded by `runs/m2_clean/`; the VM tree was exported from commit ced98c8 without git, so its metadata flags it as dirty.
