# Final-phase baselines: full SVD-LLM, IO-SVD and the dense students

Written 19 November 2026 (Phase C, baselines agent). Code: `mirror/final/baselines/svdllm_full.py`, `iosvd.py`,
`dense_students.py`. Tests: `tests/test_fp_baselines.py` (12 tests, CPU, tiny random ViT). Plan items: A05 (full
SVD-LLM, IO-SVD) and A04 (width-pruned and block-dropped dense students).

All three compress the same matrices as every other baseline (fc1, fc2, wq, wk, wv, wo of the chosen blocks, folded
weights: norm affines and LayerScale folded with `mirror.surgery.fold_mlp` and `mirror.attention.fold_attn`, written
back in absorbed form), at the same `mirror.budget.SpanBudget`, with the same rule (`budget.gap_ok`: 0 <= gap <= 1%,
rounding towards more parameters for the baseline).

## 1. Sources read (19 November 2026)

| Source | URL | Version or commit | What was read |
|---|---|---|---|
| SVD-LLM paper | https://arxiv.org/abs/2403.07378 | v5, 16 March 2025 (versions v1 12 Mar 2024, v2 15 Mar 2024, v3 1 Apr 2024, v4 28 May 2024) | abstract, version history |
| SVD-LLM paper, HTML | https://arxiv.org/html/2403.07378v5 | v5 | Section 3.2, Algorithm 3, Tables 4, 5 and 15, Appendix A.10 |
| SVD-LLM paper, v1 | https://arxiv.org/abs/2403.07378v1 | v1 | abstract ("layer-wise closed-form model parameter update") |
| SVD-LLM code | https://github.com/AIoT-MLSys-Lab/SVD-LLM | commit 7538cca98880ff312a79a16252aab5a7b480fbe9 (28 August 2025, the latest; listed by https://api.github.com/repos/AIoT-MLSys-Lab/SVD-LLM/commits?per_page=5) | README.md, SVDLLM.py, utils/LoRA.py, read in full from `https://raw.githubusercontent.com/AIoT-MLSys-Lab/SVD-LLM/7538cca98880ff312a79a16252aab5a7b480fbe9/<file>` |
| IO-SVD paper | https://arxiv.org/abs/2605.15626 | v1, 15 May 2026 (the only version) | abstract |
| IO-SVD paper, HTML | https://arxiv.org/html/2605.15626v1 | v1 | Sections 3.1 to 3.3, Eq. 1 to 14, Algorithms 1 and 2, Appendices C to E, Table 4 |
| IO-SVD code | https://github.com/mint-vu/IO-SVD | none: the repository is empty | the page says "This repository is empty"; https://api.github.com/repos/mint-vu/IO-SVD reports size 0, created 2026-05-15T04:53:07Z; https://api.github.com/repos/mint-vu/IO-SVD/commits answers HTTP 409 (empty repository); issue #1 (18 May 2026, https://api.github.com/repos/mint-vu/IO-SVD/issues?state=all) also reports the link as broken |
| FLAP | not re-read here | (verify) | `mirror/baselines.py` records arXiv 2312.11983 and github.com/CASIA-IVA-Lab/FLAP (lib/prune.py, lib/layerwrapper.py) without a commit |

## 2. Full SVD-LLM (`svdllm_full.py`)

### 2.1 What SVD-LLM's second component actually is

* **Paper v5 (ICLR 2025), Section 3.2 and Algorithm 3, "Parameter Update with Sequential Low-rank Approximation".**
  After whitening, each compressed matrix is stored as two factors W'_u [out, r] and W'_v [r, in]. Line 4: "Fix all
  W'_v, fine-tune all W'_u" with LoRA (W'_u <- W'_u + B_u A_u); line 5: "Fix all W'_u, fine-tune all W'_v"
  (W'_v <- W'_v + B_v A_v). LoRA is applied to each factor, not to the product, so the rank stays r. Reason given:
  fine-tuning both at once "will not guarantee a decrease in fine-tuning loss"; Table 15 (Appendix A.10) shows joint
  LoRA worse (80.23 against 31.79 perplexity at 80% on LLaMA-7B, as read from the HTML; the table caption's labels
  look swapped (verify)). Table 5: the order changes little. Data: Alpaca,
  50K samples. The paper gives no LoRA hyperparameters.
* **README Step 2 (commit above):** "We first update the compressed weight matrix U and then V with LoRA
  fine-tuning", `python LoRA.py --prune_model ... --data_path yahma/alpaca-cleaned --lora_r 8 --num_epochs 2
  --learning_rate 1e-4 --batch_size 64`. `utils/LoRA.py` is one PEFT LoRA training (alpha 16, dropout 0.05, 100
  warm-up steps, `bias="none"`); its default target modules list both the `*_u_proj` and `*_v_proj` layers, so the
  two rounds are two runs with different target modules. The scripting of the two rounds is not in the repository.
* **The older component still in the code:** `SVDLLM.py --step 2` runs `whitening_local_update`, the v1 paper's
  "layer-wise closed-form model parameter update": layers in order; calibration inputs reach layer i through the
  already compressed layers (`inps = outs` after the compressed layer is re-run); inside layer i the forward hooks
  sit on the original linear layers, so the target is the original layer's output; for each linear,
  x = X V_r^T Sigma_r (V_r = VT_r S^-1, the whitened right factor), U^T <- lstsq(x, Y), factors
  (U Sigma^1/2, Sigma^1/2 V_r). It uses `--updating_nsamples` (16) sequences, separate from the 256 whitening
  sequences. The current README does not mention it.

Conclusion: the published second component is a gradient fine-tuning stage, so it is a training hook here. The
closed-form update is available as an option (off by default).

### 2.2 Mapping

| Official | Here |
|---|---|
| `whitening()`: S = chol(X X^T), SVD of W S, W'_u = U_r Sigma^1/2, W'_v = Sigma^1/2 V_r^T S^-1 | `compress` calls `mirror.baselines.compress(..., "svdllm", ...)` unchanged (same factors, ridge lam = 1e-6 mean diag M), so the compressed start equals the whitening-only baseline exactly (tested); every entry gets `U_key`, `V_key` |
| LoRA_u then LoRA_v (Algorithm 3) | `training_hooks(info)` -> `[SequentialFactorSchedule]`; `begin_step(step, total, model)`: steps [0, round(split total)) train every `LowRankLinear.U` (= W'_u) with every `.V` frozen, then the reverse; `order` "u_first" (default) or "v_first"; `split` 0.5 (390 of 780 steps, one epoch of the heal set each; 100 of a 200-step learning-rate trial) |
| `whitening_local_update` (v1) | `closed_form_update` with `hparams={"closed_form_update": True}` and `calib={"adapter", "loader", "device"}`: U' = W M_c V^T (V M_c V^T)^-1 (normal equations of the same least squares; tested equal to the official lstsq on the token matrix), M_c from the fit images through the updated earlier blocks and the original block |

### 2.3 Deviations

1. No LoRA adapters: each round trains the factor itself (LoRA's B A is a rank-8 restriction of that update). The
   factorisation keeps rank r either way.
2. One healing run with the identical data, batches, 780 steps and learning-rate rule as every method (one warm-up
   plus cosine); the official code runs two separate LoRA trainings with their own schedules on Alpaca with a
   language-modelling loss.
3. Biases of the converted matrices and the embeddings train in both rounds, as for every method (LoRA.py trains no
   bias; LLaMA has none).
4. Inherited from the whitening baseline: the ridge (the folded norm outputs are mean-zero over channels, so X X^T is
   singular; the official code shifts by the smallest eigenvalue only when Cholesky fails) and the matched
   per-matrix ranks instead of one compression ratio.
5. Closed-form option: the update images are the given calibration loader (official: 16 separate sequences); the
   target excludes the bias (the bias is unchanged; the official OPT path includes it in the target).

### 2.4 Trainer contract for the hook (multi-student trainer)

* Build the optimiser over both factor groups, that is call `configure_trainable` before the first `begin_step`
  (`frozen_parameters` of the hook is empty). Building the optimiser from `requires_grad` after `begin_step(0, ...)`
  would leave V out of it for the whole run.
* Call `begin_step(step, total, model)` at the start of every optimiser step, before the forward pass, with `total`
  the run's step budget.
* Zero gradients with `set_to_none=True` (mirror/heal.py does). `begin_step` also sets the frozen group's `.grad` to
  None; AdamW skips parameters without a gradient (no update, no weight decay). Tested with AdamW and weight decay.
* `torch.compile` will recompile once at the switch (requires_grad changes).

## 3. IO-SVD (`iosvd.py`)

### 3.1 Status

The paper exists and was read. The official repository exists but is empty, so nothing was ported from code; the
implementation follows the paper and marks what the paper leaves open (verify).

### 3.2 What the paper states (W [m, n] = [d_out, d_in])

* Objective (Eq. 4): dJ ~ 1/2 ||C^1/2 (W - W^) R^1/2||_F^2, R = E_t[x x^T] (uncentred), C = E_t[J^T H J] with H the
  Hessian of the KL to the uncompressed model's top-K next-token probabilities; damping R + lam_R I, C + lam_C I.
* C in practice (Appendix C): H = A A^T, probes v_j = A e_j, g = d(v_j^T z)/dh via backward hooks, C = (1/N) sum g g^T,
  a full d_out x d_out matrix.
* Truncation (Eq. 5, 7, 8): B = C^1/2 W R^1/2 = U S V^T, W^ = C^-1/2 U_r S_r (R^-1/2 V_r)^T.
* Rank allocation (Eq. 9, 10, 13, 14; Algorithms 1, 2): score I_i = |u_i^T G~ v_i s_i|, G~ = C^-1/2 (dL_cal/dW) R^-1/2;
  one min-heap over every matrix's current tail component; pop the lowest, drop it, gain storage_gain(r) (0 above
  r* + 1, m n - r* (m + n) at r* + 1, m + n below; r* = floor(m n / (m + n))); push the next tail while
  r > r_min = ceil(eta r*); stop when the removed parameters reach the target; a matrix with r > r* keeps W.

### 3.3 Adaptation to the ViT, and choices the paper does not fix

| Item | Here | Status |
|---|---|---|
| R | `collect_stats` (methods ["svdllm"]): M1, M2, Nin, Mctx over the fit images through the original model | paper does not say original or partly compressed model (verify) |
| Output metric | DINOv2 has no classifier. C is the Gauss-Newton matrix of the healing objective's final terms: normalised MSE on the final-norm CLS token (weight 1 / (d mean CLS^2)) and patch tokens (1 / (P d mean patch^2)), per image; `output="cls"` uses the CLS term only | adaptation |
| C estimate | `collect_extra_stats`: Gaussian probes over those outputs (unbiased; default 8 per batch) or `probes="exact"` (one per output entry); gradients at each linear layer's output in an absorbed copy (the folded matrix's own output); the gradient at token t includes every path through attention | tested equal to an explicit Jacobian; cross-token treatment not stated in the paper (verify) |
| Score | default `second_order`: s_i^2 (dropping component i raises the Eq. 4 objective by s_i^2 / 2). The paper's first-order |g s| needs a calibration loss with a non-zero gradient at the uncompressed weights; the paper never defines L_cal, its own KL objective has zero gradient there, and DINOv2 has no task head. `first_order` is available with `collect_extra_stats(..., loss_fn=...)` (gradient tested equal to autograd's) | departure for DINOv2 (verify) |
| lam_R, lam_C | 1e-6 x mean diagonal (as our svdllm ridge; C of fc2 and wo is singular along the all-ones direction, which no LayerNorm reader sees) | not given (verify) |
| eta | 0.1 by default; grid {0.1, 0.5} (`HPARAM_GRID`, choose on the method-selection set as in MUST-07); r_min at least 1 | not given (verify) |
| Square roots | symmetric eigendecomposition; the truncated W^ and the scores do not depend on this choice | not given |
| Factor split | A = C^-1/2 U_r S_r^1/2, V = S_r^1/2 V_r^T R^-1/2 (balanced, as SVD-LLM's code) | not given (verify) |
| Budget | greedy against `budget.total`, stopping before the removal that would go below it, so 0 <= gap < max(m + n) (+0.006% at DINOv2-B); the paper stops once the removed amount reaches its target | rounding towards more parameters, as the rule requires |
| Integer tie | when m n / (m + n) is an integer (768 x 768: r* = 384), a matrix at r = r* costs m n either way and stays dense; the paper's "dense iff r > r*" would truncate it for no saving | tie convention |
| `allocation="matched"` | svdllm's per-matrix ranks with IO whitening (an ablation, not the method) | ours |

Not implemented: the loss-aware 8-bit remapping of factor rows, half-prune quantisation and KV-cache compression; no
method in this comparison is quantised.

Dense matrices keep W and get full-rank given factors, so `make_trainable(..., lowrank_factors="given")` always works;
such a matrix then trains as a full-rank factorisation (r (m + n) entries for its m n count). `dense_as_width(info,
n_heads)` rewrites those entries as "width, every index kept", so they train as plain dense matrices with exactly
m n entries (tested). Whether IO-SVD leaves any DINOv2-B matrix dense is only known after the run (the log line and
`per_matrix[*]["dense_storage"]` say).

## 4. Dense students (`dense_students.py`, A04)

### 4.1 Width-pruned: `width_pruned(adapter, budget, stats, hparams=None)`

Wraps `mirror.baselines.compress(adapter.model, "flap", ...)` with the gate-B choice `{"allocation": "uniform"}`
(`results/MUST-07/gate_b.json`: msel top-1 67.00, params_total 64,555,008 against 64,554,464). Confirmed from the code
and the tests:

* fluctuation scores (sample variance of each fc2 or attention-projection input channel times ||W[:, j]||^2; a head
  scores the sum of its channels);
* MLP hidden units removed (smaller `nn.Linear`), whole heads removed in dense form (rows of wq, wk, wv and their
  biases, columns of wo set to zero);
* budget: `_flap_uniform` keeps round(H x attention share) heads per block and fills each block's share with
  neurons, at most one neuron (2 d weights) above the budget; within 1%;
* bias compensation on fc2 and on the attention projection (mean of the removed inputs);
* zeroed heads are not counted (`per_matrix` params = kept heads x dh x d; `structure_check` realised count equals
  `params_matrices`); `make_trainable` stores wq, wk, wv as `IndexedLinear` (axis out) and wo as `IndexedLinear`
  (axis in) over the kept heads only, so the removed heads stay exactly zero while training (tested after AdamW
  steps), and the student holds exactly `params_matrices` matrix entries (tested).

Caveat: a block that keeps no head would fail `make_trainable`; `info["min_heads_kept"]` reports the smallest count
and a warning is logged at zero. MUST-09 never healed FLAP, so this has not run at DINOv2-B size (verify on the VM).

### 4.2 Block-dropped: `block_dropped(adapter, n_drop, score_fn, max_candidates=None, *, budget=None, target=None)`

Greedy: each round scores every remaining candidate's removal with `score_fn(model)` (higher is better; on the VM the
k-NN top-1 of the 10k method-selection queries against the 100k bank, the bank embedded by the shortened model) and
removes the best; ties go to the lower block index. Every candidate score, the order and the original network's
score are recorded. Candidates are light views (shared weights, own blocks container; the original is untouched,
tested). `max_candidates=k`: round 1 tries all blocks, later rounds only the k remaining blocks with the best round-1
scores. Score evaluations for 3 of 12 blocks: 1 + 12 + 11 + 10 = 34 (all candidates) or 1 + 12 + 4 + 4 = 21
(`max_candidates=4`); the time per evaluation is to be measured on the VM (verify).

Count: the original model's parameters (hub position table included, as for the 66,203,360 target) minus every
parameter of the removed blocks. Arithmetic from the DINOv2-B shapes: one block holds 7,089,408 parameters
(7,077,888 matrix weights, 6,912 biases, 3,072 norm and 1,536 LayerScale entries), so 3 blocks give 65,315,328
(0.7544 of 86,583,552), 1.34% below the target. The VM run prints the measured count; `eligible` is False by
construction and the gap is reported.

`blockdrop_trainable(model, info, adapter) -> (model, constraints, spec)`: position grid materialised (as
`make_trainable` does for every method), every block absorbed (norm affines and LayerScale folded into the weights;
the same function, tested), `BlockDropTrainable` freezes the absorbed affines, LayerScale, final norm affine, mask
token and any head, so blocks and embeddings train (run with `train_scope="all"`). `spec` (JSON) holds `kept`,
`order`, `block_map` {student j: teacher kept[j]} (block-output targets: the teacher's raw output of block kept[j]),
the counts and `base`. `load_blockdrop(adapter, spec, state_dict)` rebuilds it (tested equal).

## 5. Public API

```python
# svdllm_full
METHOD = "svdllm_full"; DEFAULT_HPARAMS = {"damp": 1e-6, "order": "u_first", "split": 0.5, "closed_form_update": False}
compress(model, blocks, components, budget, stats, hparams=None, calib=None, *, verbose=True) -> (model, info)
training_hooks(info) -> [SequentialFactorSchedule]      # begin_step(step, total, model), check, describe
closed_form_update(m, original, info, calib, ridge=0.0) -> list      # used by compress when enabled
stats_needed(components) -> set                          # collect_stats names (those of "svdllm")

# iosvd
METHOD = "iosvd"; DEFAULT_HPARAMS = {"damp_in": 1e-6, "damp_out": 1e-6, "eta": 0.1, "score": "second_order",
                                     "allocation": "adaptive"}; HPARAM_GRID (eta 0.1, 0.5)
collect_extra_stats(adapter, loader, device, blocks, components, *, n_probes=8, probes="gaussian", output="final",
                    loss_fn=None, seed=0, stats_device=None, to_cpu=True) -> IOExtraStats
compress(model, blocks, components, budget, stats, extra, hparams=None, *, verbose=True) -> (model, info)
adaptive_ranks(shapes, scores, total_target, eta) -> dict;  io_factorise(W, R, C, damp_in, damp_out) -> dict
dense_as_width(info, n_heads) -> info

# dense_students
width_pruned(adapter, budget, stats, hparams=None, *, verbose=True) -> (model, info)
block_dropped(adapter, n_drop, score_fn, max_candidates=None, *, budget=None, target=None, score_original=True,
              verbose=True) -> (model, info)
blockdrop_trainable(model, info, adapter, *, absorb=True) -> (model, constraints, spec)
load_blockdrop(adapter, spec, state_dict) -> nn.Module;  blockdrop_skeleton(adapter, spec)
model_level(adapter_or_model, budget, params_total, step=0) -> dict;  drop_blocks(model, kept, copy_weights=True)
```

Every builder reports `params_model`, `params_frac`, `target`, `eligible`. Naming: for the three compressors `gap` is
the compressor contract's matrix-level gap ((params_total - budget_total) / budget_total) and `gap_model` the gap to
the whole-model target; for block dropping `gap` = `gap_model`.

## 6. Wiring for the coordinator and the heal agent

1. Statistics: one `mirror.baselines.collect_stats(adapter, fit_loader, device, blocks, methods=["svdllm", "flap"],
   components=("mlp", "attn"))` serves all three; IO-SVD also needs `iosvd.collect_extra_stats(...)` on the fit images
   (DINOv2-B: C holds about 1.19 GB in float64 for 12 blocks, by arithmetic; the backward graph is kept across probes,
   so use a loader batch of about 16 to 32 images (verify memory on the VM)).
2. Students: `svdllm_full` and `iosvd` -> `heal_structured.make_trainable(model, info, adapter,
   lowrank_factors="given")` (optionally `iosvd.dense_as_width(info, 12)` first); constraints + `svdllm_full.
   training_hooks(info)` for full SVD-LLM; `width_pruned` -> `make_trainable(model, info, adapter)`; block dropping ->
   `blockdrop_trainable`. All unfolded, so block targets are raw (`distill_teacher` gives "raw" for these specs);
   block dropping needs `TargetMode.block_map` from `spec["block_map"]`.
3. The multi-student trainer must call `begin_step(step, total, model)` when a constraint has it (Section 2.4).
4. Registry method names (design doc Section 4.1): `svdllm_full`, `iosvd`, `flap` (the width-pruned student),
   `blockdrop`.
5. Learning rate: the new methods choose from {1e-4, 2e-4} with 200-step trials; the schedule then splits at step 100.

## 7. Items marked (verify)

* SVD-LLM: which update order produced the paper's main tables (Table 5's "W'_v first" row at 20% equals the main
  7.73, while Algorithm 3 and the README say U first); how the authors scripted the two LoRA rounds; the per-round
  LoRA hyperparameters (README command against LoRA.py defaults).
* IO-SVD: R from the original or a partly compressed model; lam_R, lam_C; eta; K (not applicable); L_cal; the factor
  split; cross-token terms in the probe gradients. The repository is empty, so none of these can be checked in code.
* FLAP source commit; zero-head blocks at DINOv2-B size; block-dropping evaluation time; collect_extra_stats memory
  per batch on the VM.

## 8. Tests

`.venv/Scripts/python.exe -m pytest -q tests/test_fp_baselines.py` (with `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`): 12
tests on the tiny float64 ViT: budget rule and model-level counts for all three compressors; full SVD-LLM start equal
to "svdllm"; given factors reach `LowRankLinear` unchanged; the schedule's requires_grad toggling and an AdamW loop in
which only the scheduled factor group moves; the closed-form update equal to the official lstsq, unchanged block 0,
never a larger error; IO-SVD equal to SVD-LLM with an identity output metric, optimal in the weighted norm, C equal
to an explicit Jacobian, G equal to autograd, Algorithms 1 and 2 on hand-made cases, trainable form equal to the
compressed model, `dense_as_width` counts; width pruning's zeroed heads stay zero and are not counted; block dropping
picks the least harmful blocks in order, reports its negative gap, and its trainable form and reload match.
