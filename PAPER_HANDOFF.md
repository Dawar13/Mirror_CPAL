# Paper handoff: where everything is (final phase, 20 November 2026)

For whoever writes the CPAL 2027 paper from the final-phase results. Every number below comes from a file in this
repository, written by a script with a fixed seed; each table row names its source file and commit.

## Start here

1. `results/FINAL_PHASE_REPORT.md`: every pre-registered rule, its threshold, the measured value and the verdict,
   with sources. Rules are fixed in `PREREG_FINAL_PHASE.md` (committed before any GPU run, fd83c95).
2. `results/final/A09_addendum.md`: the A09 exactness audits ran, but the report shows them as PENDING because of a
   path mismatch; the verdicts are here.
3. `results/STATUS.md`, section "Phase G": a one-screen summary of the run, the verdicts and what was cut.
4. `notes/decisions.md`, rows dated 2026-11-20: every deviation and intervention. The paper must disclose the ones
   marked below.

## Numbers and tables

- `results/final/tables/T1` to `T6` (`.md` to read, `.csv` with source file and commit per row): T1 main comparison
  (DINOv2-B, all arms, bootstrap intervals), T2 stage ablation, T3 other models (DeiT-III), T4 operator stability
  (A02), T5 discovery controls (A03), T6 systems (A12).
- `results/final/numbers.json`: every number the report quotes, machine-readable.
- Item folders `results/A01` to `results/A14` (`summary.md` and `metrics.json` in each); per-run metrics and
  evaluations in `runs/final/<group>/<model>/<method>/s<seed>/` (`metrics.json`, `eval/step780.json`).

## Figures

`paper/figures/`: `fig_frontier`, `fig_stability`, `fig_controls`, `fig_curves`, `fig_fourier`, `fig_twins_hires`
(teaser, with 48 tiles in `fig_twins_hires_tiles/`). Each has `.pdf`, `.png` and a `.json` with every plotted value,
its inputs (sha256) and the caption. Re-draw on a laptop: `python scripts/fp_fig_<name>.py` (teaser:
`--from-tiles results/A14/teaser/tiles.npz`; Fourier: `--from-maps results/A14/fourier/fourier_maps.npz`). Earlier
figures and the previous draft: `results/paper_next/`, `paper/Learned_Twice_Stored_Once_draft3.pdf`. Template:
`cpal_2027.tex`, `paper/cpal_2027.sty`; bibliography `reference.bib`.

## Headline results (DINOv2-B/14 reg unless stated; k-NN top-1, 50,000 validation queries, bf16)

| Result | Value | Rule |
|---|---|---|
| Mirror (3 seeds) vs SVD-LLM (3 seeds), same size | 78.35 vs 78.08; difference +0.27, 95% interval [+0.08, +0.47] | A05-1 PASS, superiority |
| Mirror + SVD-LLM vs SVD-LLM at 0.65 / 0.50 / 0.40 of the size | +1.64 / +2.75 / +4.13 | A06-1 PASS; A06-2 PASS |
| Exactness: float32 flip consistency | 100% (k-NN, DeiT-III classifier with 0 of 50,000 differing, ADE20K masks 99.9999% with all differences near-ties) | A08-1, A09, P11.2 |
| Float64 equivariance error | 6.85e-14 (DINOv2-B), 3.35e-13 (DeiT-III) | A09-1 PASS |
| Stability across fit sets and domain | spread 0.022; COCO fit -0.002 | A02-1 to A02-6 PASS |
| DeiT-III-B top-1 (3 seeds) | Mirror 82.70 vs SVD-LLM 82.88 (-0.19) | A08-1 PASS |
| Against flip averaging at equal compute | +3.76 | A07 PASS |
| Reproduction of the earlier MUST-09 numbers | 78.35 / 78.08 | Gate P PASS |

## Results the paper must report honestly

- FAIL: A01 (the weight-space operator A*_w: zero-shot 7.19, healed -1.45; report as an ablation that supports
  fitting on activations), A03-1 (control C3 twin fraction 0.079), A03-2 (random pairing costs only 1.34, bar 3.0),
  A04-1 (FLAP, a width-pruned dense student, 78.41 on seed 0, not 0.5 below Mirror), A09-2 (one tensor at exactly
  2x the rounding floor), A12b (fused kernel 0.80x and 0.83x: no speed claim).
- Post hoc, outside the pre-registered scores: IO-SVD diverged at the pre-registered learning rate (seeds 0.54, 0.40,
  0.66); at 5e-5 it reaches 78.72 over 3 seeds (`runs/final/posthoc/`), 0.37 above Mirror. Disclose both.
- Full SVD-LLM is a weak port (3 seeds 76.45, 76.76, 76.06): the official LoRA-based update was not reproduced; a fix
  tried during the run diverged and was abandoned (`notes/decisions.md`).
- Not run: DINOv2-L (A08-2), A15a/b and A11's Mirror arm, cut in the pre-registered order for time; A10 full-bank
  k-NN and linear protocol (its data path was too slow). A11's SVD-LLM arm alone: 78.33 after 2,340 steps.
- Suggested framing: lead with exactness and with stacking on low rank (the frontier figure), not with single-size
  accuracy superiority over every compressor.

## Large files (Git LFS)

Run `git lfs install` once, then `git lfs pull` after cloning, to get the large files stored through Git LFS:

- `artifacts/final_phase/mirror_exact_astar/s{0,1,2}/step780.pt`: the three healed Mirror A* checkpoints
  (329 MB each; a `.sha256` file beside each). They load with the operator structure
  `results/MUST-07/structure/dinov2_vitb14_reg_s0.npz` and the run's `spec.json` in
  `runs/final/main/dinov2_vitb14_reg/mirror_exact_astar/s<seed>/`.
- `results/MUST-07/structure/*.npz`: the fitted operator structures of all ten models (128 to 439 MB).
- `results/e1_operators.npz` and `runs/x2/perlayer_reg_s{0,1,2}/results/e1_operators.npz`: the E1 operators and
  the A02 fit inputs.

Not in the repository: every other healed checkpoint (deleted with the GPU instance), the CORE-07 feature caches,
the previous phase's healed states (`runs/MUST-09`) and the earlier A100 artefacts. None is needed to write the
paper; every number and figure above comes from committed files.
