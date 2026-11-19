# Open items and suggested experiments

## Checked against the data while rewriting
- All plotted numbers are read from Mirror_CPAL `results/` (see `data/*.csv`, `source` column).
- Table numbers were carried over from draft 3; spot-checked against `results/MUST-09/summary.csv`,
  `results/MUST-07`, `results/CORE-06/probes.csv` and `results/paper_next/data/*.csv`. They match.
- Pre-registration counts: 47 rows; 40 core (17 PASS, 16 FAIL, 7 PENDING) and 7 optional UP items.
  The draft said "40 scored predictions". The rewrite says "of 40 core predictions, 17 passed,
  16 failed and 7 could not be scored", which is what the CSV shows.

## Experiments that would most strengthen the paper (by expected reviewer impact)
1. **Heal a second model** (DeiT III-B/16 or DINOv2-L/14 reg). Today every healed result is one model at one size.
   This is the most likely reviewer complaint.
2. **Structure-fit seeds for healing.** The three healing seeds share one fitted structure, so the +0.28 bootstrap
   interval omits structure variance. Heal from fit seeds 1 and 2 (already fitted for the zero-shot table).
3. **Full SVD-LLM** (whitening + sequential update), or say clearly why it was omitted. Reviewers know SVD-LLM's update step.
4. **Dense distillation baseline at 66M parameters**, e.g. a narrower ViT distilled with the same 780 steps.
   This answers "is it just distillation?" The random-structure control answers part of this, but not all.
5. **Fused even/odd kernel.** The 0.72x throughput is a visible weakness. A Triton kernel for the block-diagonal
   matmuls, even on MLPs only, would turn the FLOPs saving into a speed claim.
6. **Causal test of the Fourier explanation.** Train a small ViT with 90-degree rotation augmentation and check
   whether 90-degree pairing appears (the ImageNet-100 grid failed its gate; a larger model or longer schedule may pass).
7. **Full-bank k-NN** for the original and healed models, so numbers are comparable with published DINOv2 results.
8. MUST-11 P11.2 is pending only because margins were not stored; rerunning that evaluation with margins would close it.

## Added after the review of 2026-11-17 (see RESPONSE_TO_REVIEW.md)
Already answered from existing results (no new runs): A* fit-set size (A3) and fit seeds, head pairing per block
(new Table 5), SigLIP 2 diagnosis, the explicit attention step of the exactness proof, related work (REViT, Equi-ViT, IO-SVD).
New runs the reviewer asks for, in priority order:
9. **IO-SVD** baseline at the same budget, healed with the same recipe (code: github.com/mint-vu/IO-SVD). Together with
   item 3 this is the reviewer's main stated reservation.
10. **Fine-tuned ImageNet classification** (linear head, DINOv2 protocol) for original / Mirror Surgery / SVD-LLM.
11. **A\* after healing**: heal from fit sets 1 and 2 (same as item 2), and fit A\* on a non-ImageNet set (e.g. Places or COCO) to test domain shift.
12. **Stability of B_h, C_h and head pairings** across fit seeds 0-2 (cheap: structure-only fits).
13. **SigLIP 2 hypothesis**: project in the whitened basis and see whether the collapse disappears.
14. Optional: LEE (Lie derivative) before/after conversion; dihedral (LR + UD) conversion of the early blocks.

## Things to decide or confirm
- The teaser (Fig. 1a) crops two patches per neuron from `fig_twins.png`. If the raw `fig1_teaser.npz` exists
  on the GPU box, regenerating from it would give sharper tiles.
- Fig. A3 (Fourier) uses nine scikit-image sample photos. Averaging over ~1,000 ImageNet images would make
  panel (d) smoother and more convincing; the LR = UD identity holds exactly either way.
- Author block is anonymous; acknowledgements omitted for blind review.
