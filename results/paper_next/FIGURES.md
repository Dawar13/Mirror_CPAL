# Figures and tables of the next phase

Written by `scripts/make_figures_next.py` on 2026-11-15 20:30 UTC (code commit `6b4c66ab78`) from `results/paper_next/data` and PAPER_NUMBERS.json only. Skipped items had no input; no placeholder data is drawn.

## Figures

- **fig1_twins**: fig1_twins.png, fig1_twins.pdf. Top activating patches (50k ImageNet-train images) of the strongest twin pairs in blocks 1, 3, 6. Sources: data/fig1_teaser.npz, PAPER_NUMBERS.json must04.dinov2_vitb14_reg.LR.tax_1-6.
- **fig2_tax_depth**: fig2_tax_depth.png, fig2_tax_depth.pdf. Tax by depth across models (a); breaking share relative to the matched null for the trained (LR) and wrong groups (b, c). Sources: data/fig2_blocks.csv, data/table4_generality.csv. Not drawn: artefact-token inset (earlier-phase result, results/paper).
- **fig3_controlled**: fig3_controlled.png, fig3_controlled.pdf. Controlled training (ViT-S/8, ImageNet-100): structure by training symmetry (a) and its emergence (b); thin lines are single seeds. Sources: data/fig3_matrix.csv, data/fig3_emergence.csv.
- **fig4_visibility_stacking**: fig4_visibility_stacking.png, fig4_visibility_stacking.pdf. (a) Twins correlate on the mirrored image but not on the same image; (b) Mirror stacks with low rank; (c) breaking energy by input-variance band over training. Sources: data/fig4a_pairs.csv, data/fig4b_stacking.csv, data/fig4c_bands.csv. Not drawn: panel c (inputs missing).
- **fig5_accuracy_params**: fig5_accuracy_params.png, fig5_accuracy_params.pdf. Accuracy against parameters: every method at matched parameters, zero-shot (hollow) and healed (filled, mean and standard deviation over seeds), with the released DINOv2 family. Sources: data/fig5_points.csv, PAPER_NUMBERS.json gate_b.best.

## Tables

- **table1_healed**: table1_healed.md, table1_healed.csv, table1_healed.tex. Healed full-model comparison (identical recipe) and test-time flip averaging Sources: data/table1_healed.csv, data/table1_flipavg.csv.
- **table2_downstream**: table2_downstream.md, table2_downstream.csv, table2_downstream.tex. Downstream tasks, robustness and mask equivariance of the healed full models Sources: data/table2_downstream.csv, data/table2_shift.csv.
- **table3a_compressors**: table3a_compressors.md, table3a_compressors.csv, table3a_compressors.tex. Zero-shot comparison at matched parameters (MUST-07), DINOv2-B reg Sources: data/must07_rows.csv, data/table3_must07_p7.csv.
- **table3b_components**: table3b_components.md, table3b_components.csv, table3b_components.tex. Matched component ablation of Mirror-exact (MUST-08) Sources: data/table3_must08.csv.
- **table3c_earlier_ablations**: table3c_earlier_ablations.md, table3c_earlier_ablations.csv, table3c_earlier_ablations.tex. Earlier ablations: operator, pairing, fit size, block choice and projection strength (mean ± std over fit seeds) Sources: data/table3_earlier_ablations.csv (from results/paper/tables/T_ablations.csv).
- **table4_generality**: table4_generality.md, table4_generality.csv, table4_generality.tex. Generality and the law across models and groups Sources: data/table4_generality.csv, data/appendix_law.csv.
- **table4b_refund_new_groups**: skipped (no CORE-02 surgery rows).
- **appendix_a1_exactness**: appendix_a1_exactness.md, appendix_a1_exactness.csv, appendix_a1_exactness.tex. Exactness audit of Mirror-exact, per component (largest over blocks) Sources: data/appendix_exactness.csv.
- **appendix_a2_law**: appendix_a2_law.md, appendix_a2_law.csv, appendix_a2_law.tex. The law: predicted against achieved parameter saving (MUST-01) Sources: data/appendix_law.csv.
- **appendix_a3_prereg**: appendix_a3_prereg.md, appendix_a3_prereg.csv, appendix_a3_prereg.tex. Pre-registration: every prediction with its verdict Sources: PAPER_NUMBERS.json predictions.
- **appendix_a4_failed**: appendix_a4_failed.md, appendix_a4_failed.csv, appendix_a4_failed.tex. Failed checks, reported as failures Sources: PAPER_NUMBERS.json failed_checks.
