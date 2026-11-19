# Final phase: code design and shared interfaces (Phase C)

Written 19 November 2026 for the parallel build of FINAL_PHASE_PLAN.md v2, Section 7. Every agent reads this file
first. The plan is the authority for what to run; this file fixes how the pieces fit, who owns which file and the
interfaces between them.

## 0. Rules for every agent (laptop, Phase C)

1. **No GPU, no model or dataset downloads.** Never call `torch.hub.load`, `timm.create_model(..., pretrained=True)`,
   `datasets.load_dataset` or `hf_hub_download` on the laptop. Tests use tiny random models. Reading public web
   pages (papers, GitHub source) to verify a method is allowed; never put the user's name, email or project details
   into a request.
2. **The laptop is short of RAM** (11.7 GB, often under 1 GB free) and runs several agents at once. Before running
   Python, set `export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`. Run only your own tests:
   `.venv/Scripts/python.exe -m pytest -q tests/test_fp_<yours>*.py`. Never run the whole suite (12 minutes,
   700+ tests); the coordinator does. Keep test models tiny (embed dim 32 to 64, depth 2 to 4, 28 to 56 px images).
3. **Do not install packages.** If something is missing, say so in your report.
4. **Only edit the files you own** (Section 2). Everything else is read-only. If you need a change in an existing
   file, put a helper in your own files or report the needed change.
5. **No git commits or pushes.** The coordinator commits.
6. **Keep the existing code paths working.** Gate P reproduces MUST-09 with them: `mirror/heal.py`,
   `mirror/heal_structured.py`, `mirror/heal_vexact.py`, `mirror/mirror_family.py`, `mirror/baselines.py`,
   `mirror/budget.py`, `scripts/must09_heal.py` stay unchanged.
7. **No invented numbers or facts.** Model IDs, recipes, baseline algorithms and citations are checked against a
   source or marked "(verify)". Record every source URL you used in your module docstring.
8. **Style** for docstrings, comments and reports: British spelling, plain language, no em dashes, no emoji.
9. Windows laptop, Git Bash. The venv is `.venv/Scripts/python.exe` (Python 3.12, torch 2.11 CPU, timm 1.0.31,
   numpy 2.4, scipy 1.15, networkx, scikit-learn, PyYAML). No Triton on the laptop.

## 1. Layout

```
mirror/fit_weightspace.py            A01 weight-space operator A*_w with alternation
mirror/stability.py                  A02 agreement metrics between fits
mirror/controls.py                   A03 discovery null controls C1 to C4
mirror/final/prereg.py               pre-registration guard, final_meta (coordinator, done)
mirror/final/baselines/              svdllm_full.py, iosvd.py, dense_students.py, mirror_plus_svd.py, sizes.py
mirror/final/heal/                   multistudent.py (trainer), students.py (registry), targets.py, data_gpu.py
mirror/final/eval/                   knn_fast.py, fullbank.py, linear_protocol.py, flipavg.py
mirror/final/audit/exactness.py      A09
mirror/final/adapters.py             DeiT-III head and logits, DINOv2-L and SigLIP 2 helpers, A15a projection
mirror/final/systems/bench.py        A12
mirror/final/stats/hbootstrap.py     A13
mirror/final/run/runner.py           runner, governor, cut order
mirror/final/data/cache.py           data roles, new caches, full-train stream
scripts/fp_*.py                      one command-line entry point per job type (see Section 5)
tests/test_fp_*.py                   CPU tests, one file group per agent
jobs/final_phase.yaml                the Phase G manifest (infra agent drafts the schema, coordinator fills it)
```

Package names avoid the existing modules: `mirror/baselines.py` exists, so the new baselines live in
`mirror/final/baselines/`.

## 2. Ownership

| Agent | Owns (creates) | May also edit |
|---|---|---|
| structure | `mirror/fit_weightspace.py`, `mirror/stability.py`, `mirror/controls.py`, `scripts/fp_a01_weightspace.py`, `scripts/fp_a02_stability.py`, `scripts/fp_a03_controls.py`, `scripts/fp_structure_fit.py`, `tests/test_fp_weightspace.py`, `tests/test_fp_stability.py`, `tests/test_fp_controls.py` | none |
| baselines | `mirror/final/baselines/svdllm_full.py`, `iosvd.py`, `dense_students.py`, `docs/final_baselines.md`, `tests/test_fp_baselines.py` | none |
| stacking | `mirror/final/baselines/mirror_plus_svd.py`, `mirror/final/baselines/sizes.py`, `mirror/final/baselines/partial_span.py`, `tests/test_fp_stacking.py` | none |
| heal | `mirror/final/heal/*.py`, `scripts/fp_heal.py`, `tests/test_fp_heal*.py` | none |
| eval | `mirror/final/eval/*.py`, `scripts/fp_eval*.py`, `scripts/fp_fullbank.py`, `tests/test_fp_eval*.py` | `scripts/tier2_ade20k.py` (margin logging only; defaults unchanged) |
| audit | `mirror/final/audit/exactness.py`, `mirror/final/adapters.py`, `mirror/final/systems/bench.py`, `scripts/fp_a09_audit.py`, `scripts/fp_a12_bench.py`, `scripts/fp_a15a_siglip.py`, `tests/test_fp_audit.py`, `tests/test_fp_adapters.py`, `tests/test_fp_bench.py` | none |
| infra | `mirror/final/run/runner.py`, `mirror/final/data/cache.py`, `mirror/final/stats/hbootstrap.py`, `scripts/fp_runner.py`, `scripts/fp_cache.py`, `scripts/fp_vm_w0.sh`, `jobs/final_phase.schema.md`, `tests/test_fp_runner.py`, `tests/test_fp_cache.py`, `tests/test_fp_hbootstrap.py` | none |
| coordinator | `mirror/final/prereg.py`, `PREREG_FINAL_PHASE.md`, this file, `jobs/final_phase.yaml`, `results/STATUS.md`, `notes/decisions.md`, figure scripts, the A12b kernel | anything, at integration |

## 3. Conventions (from notes/paper/method_spec.md; read it)

- Blocks are 0-based in code; "block 1" in prose is index 0. Stream layer 0 is the token sequence entering code
  block 0; stream layer l (l >= 1) is the output of code block l-1 (method_spec.md calls it "the output of block l").
- `fc1.weight` W1 is [h, d] and `fc2.weight` W2 is [d, h]; tokens are row vectors, so a mirror acts as x -> x A with
  A a symmetric orthogonal involution (A A = I). The neuron involution P is stored as an int64 permutation `perm`
  with perm[perm] = arange(h).
- Equivariance of an MLP: P W1 = W1 A_in, P b1 = b1, A_out W2 = W2 P, A_out b2 = b2 (mirror/surgery.py).
- Every stream operator fixes the all-ones vector (A 1 = 1): the folded stream is centred, so the fit is done on the
  centred subspace (C <- Pc C Pc, Pc = I - 11^T/d) and the null direction gets +1 (mirror/fit.py fit_involution).
- The canonical structures are `SpanStructure` files (mirror/mirror_family.py: fit_span_structure,
  save_span_structure, load_span_structure). On the laptop: results/MUST-07/structure/dinov2_vitb14_reg_s{0,1,2}.npz,
  deit3_b16_s0.npz, dinov2_vitl14_reg_s0.npz (git-ignored, about 128 MB each; the VM gets them by upload).
  **Every new operator fit (A01 A*_w, A02 fit set 3 and COCO, A03 controls) writes a SpanStructure with
  save_span_structure**, so `mirror.mirror_family.build_mirror_variant` and the healing code read it unchanged.
  An A*_w structure is the fit-seed structure with `A_star[L]` replaced by A*_w, `perms` replaced by the final
  pairings, `reg_perm[L]` re-matched under A*_w, and `meta["operator_fit"]` describing the fit.
- Compressor contract (mirror/baselines.py, section "Methods at the Mirror-exact budget"; mirror/heal_structured.py
  docstring): `compress(...) -> (model, info)` with `info["per_matrix"]` entries of kind lowrank, width, slice or
  merge, `info["tensors"]`, `info["params_total"]`, `info["budget_total"]`, `info["gap"]`. New low-rank baselines
  emit kind "lowrank" with given factors (`U_key`, `V_key`), so `heal_structured.make_trainable(...,
  lowrank_factors="given")` turns them into trainable students without new code.
- Budgets: `mirror.budget.mirror_exact_budget(...)` at full conversion gives the main size: 66,203,360 model
  parameters (0.7646 of 86,583,552), 64,554,464 in the converted matrices. Baselines are eligible when
  0 <= (params - budget) / budget <= 1% (`budget.gap_ok`), that is, rounding towards more parameters for the baseline.
- Every run writes `meta.json` with `mirror.final.prereg.write_final_meta` and calls
  `mirror.final.prereg.require_prereg()` before touching the GPU. Tests set `MIRROR_FINAL_ALLOW_NO_PREREG=1`.

## 4. Interfaces between agents

### 4.1 Students (heal agent owns `mirror/final/heal/students.py`)

```python
@dataclass
class TargetMode:
    final: str = "symmetrised"        # "symmetrised" (flip average of the teacher's final-norm tokens) or "raw"
    blocks: str = "raw"               # "symmetrised" (centred, then (c(x) + align(c(mirror x)) A)/2), "centred", "raw"
    A: np.ndarray | None = None       # the student's stream operator (symmetrised block targets only)
    reg_perm: np.ndarray | None = None
    k: int | None = None              # span size: block i is symmetrised only if i < k (VExactTeacher semantics)
    block_map: dict | None = None     # student block index -> teacher block index (block dropping); None = identity
    logits: bool = False              # DeiT-III: add a normalised-MSE term on the flip-averaged teacher logits
    input_transform: str | None = None  # "rot90" for A15b: the student sees rot90(x); targets come from x

@dataclass
class StudentBuild:
    method: str
    model: nn.Module                  # the trainable form (after make_trainable or the module's own equivalent)
    constraints: list                 # mirror/heal.py constraint protocol; optional extra hook begin_step(step, total, model)
    spec: dict                        # JSON, enough to rebuild the student (see load_student)
    structure: dict | None            # mirror family: A_star, perms, maps_after, reg_perm, blocks
    targets: TargetMode
    params: dict                      # params_model, params_frac, budget_total, gap, matrices, ...
    train_scope: str = "all"          # with constraints' frozen_parameters, as in MUST-09
```

`build_student(method, ctx, **hp) -> StudentBuild` dispatches on the method name. Methods (Section 6 of the plan):
`mirror_exact` (any structure file, including A*_w), `random_structure`, `pair_random` (A03 H1: true operator, random
pairing), `svd`, `svdllm`, `slicegpt`, `flap` (the width-pruned dense student), `original` (healing without
surgery), `mirror_partial` (blocks 0..k-1 converted, the rest dense and trainable; A06 price of exactness, k = 9),
`rot90_probe` (A15b), and, from the other agents' modules: `svdllm_full`, `iosvd`, `blockdrop`, `mirror_svdllm`.
Targets reproduce MUST-09 exactly: `mirror.heal_structured.distill_teacher` decides the mode for the existing methods.

`load_student(run_dir, step=None, device="cpu") -> (adapter_or_None, model, info)` rebuilds a checkpoint for the
evaluation agent.

### 4.2 Baseline builders (baselines and stacking agents)

Each new method exposes a pure function the registry calls; the coordinator wires anything missing at integration.

```python
# mirror/final/baselines/svdllm_full.py
def compress(model, blocks, components, budget, stats, hparams=None, calib=None) -> tuple[nn.Module, dict]
def training_hooks(info) -> list      # constraint-protocol objects (e.g. the sequential factor schedule), may be []

# mirror/final/baselines/iosvd.py
def collect_extra_stats(adapter, loader, device, blocks, components) -> object   # whatever IO-SVD needs beyond collect_stats
def compress(model, blocks, components, budget, stats, extra, hparams=None) -> tuple[nn.Module, dict]

# mirror/final/baselines/dense_students.py
def width_pruned(adapter, budget, stats, hparams=None) -> tuple[nn.Module, dict]      # wraps baselines.compress("flap")
def block_dropped(adapter, n_drop, score_fn, max_candidates=None) -> tuple[nn.Module, dict]
def blockdrop_trainable(model, info, adapter) -> tuple[nn.Module, list, dict]          # (model, constraints, spec)
def load_blockdrop(adapter, spec, state_dict) -> nn.Module

# mirror/final/baselines/sizes.py (stacking agent)
def budget_for_fraction(adapter_or_model, frac, components=("mlp", "attn"), blocks=None) -> object   # a SpanBudget
def full_model_params(adapter_or_model) -> int

# mirror/final/baselines/mirror_plus_svd.py (stacking agent)
def build(adapter, structure, frac, stats, hparams=None) -> tuple[nn.Module, list, dict, dict]   # model, constraints, spec, info
def load(adapter, spec, state_dict) -> nn.Module

# mirror/final/baselines/partial_span.py (stacking agent)
def build_partial(adapter, structure, k, components=("mlp", "attn")) -> tuple[nn.Module, list, dict, dict]
```

Every builder reports `params_model`, `params_frac`, `target`, `gap` (relative to the target) and `eligible`.

### 4.3 Shared teacher pass (heal agent)

One teacher forward per micro-batch: the original model on [x; mirror(x)] when any student needs flip-symmetrised
targets (otherwise on x), capturing the residual stream after every block and the final-norm output (and the
logits for DeiT-III). Each student's targets are derived from these tensors exactly as
`mirror.heal_vexact.VExactTeacher.distill_targets` derives them (reuse its `stream_target` and `final_targets`).
The fixed loss normalisers are measured per distinct target mode on the same 1,024 calibration images as MUST-09.

### 4.4 Checkpoints and evaluation requests (heal writes, eval reads)

Run directory of one student: `runs/final/<experiment>/<model>/<method>[_<variant>]/s<seed>/` with `spec.json`
(+ `structure.npz` for the mirror family), `ckpt/step<S>.pt` (student state dict, fp32), `train_log.csv`,
`heldout.json`, `metrics.json` and `meta.json`. After saving a checkpoint the trainer writes
`runs/final/_queue/<timestamp>_<id>.json`:
`{"run_dir": ..., "step": S, "passes": ["bf16", "fp32"?], "n_query": 50000 or 10000, "priority": int}`.
The evaluation server takes requests from the queue, writes `<run_dir>/eval/step<S>.json` and
`<run_dir>/knn/step<S>_<bf16|fp32>.npz` (arrays pred, pred_flip, labels; the MUST-09 format), and moves the request
to `runs/final/_queue/done/`.

### 4.5a Arms and their run directories (manifest, gates and collector all use this table)

Run directory: `runs/final/<group>/<model>/<method>[_<variant>]/s<seed>/` (scripts/fp_heal.py writes it from the job
spec's group, model, method, variant and seed). Model names: `dinov2_vitb14_reg`, `deit3_b16`, `dinov2_vitl14_reg`.

| Arm | group | method | variant | seeds |
|---|---|---|---|---|
| Mirror, current A* (Gate P) | main | mirror_exact | astar | 0, 1, 2 |
| SVD-LLM, whitening only (Gate P, A05) | main | svdllm | | 0, 1, 2 |
| Mirror, A*_w (A01) | main | mirror_exact | wspace | 0, 1 (and 2 if adopted) |
| Full SVD-LLM | main | svdllm_full | | 0, 1, 2 |
| IO-SVD | main | iosvd | | 0, 1, 2 |
| SliceGPT-style | main | slicegpt | | 0, 1 |
| Plain SVD | main | svd | | 0 |
| Random structure | main | random_structure | | 0 |
| H1: main operator, random pairing | main | pair_random | h1 (structure "adopted": the main arm's operator) | 0 |
| Width-pruned dense student | main | flap | | 0 |
| Block-dropped dense student | main | blockdrop | | 0 |
| Original distilled | main | original | | 0 |
| A02 heals | main | mirror_exact | fit1, fit2, coco (the adopted operator type fitted on that fit set; spec.json records the file) | 0 |
| A06 SVD-LLM | a06 | svdllm | f065, f050, f040 | 0 (and 1 at f050) |
| A06 Mirror + SVD-LLM | a06 | mirror_svdllm | f065, f050, f040 | 0 (and 1 at f050) |
| A06 Mirror on 9 blocks | a06 | mirror_partial | k9 | 0 |
| A08 DeiT-III-B | a08 | mirror_exact, svdllm, random_structure | | 0, 1, 2; 0, 1, 2; 0 |
| A08 DINOv2-L | a08 | mirror_exact, svdllm | | 0 |
| A11 extended healing | a11 | mirror_exact, svdllm | ext | 0 |
| A15b cause probe | a15 | rot90_probe | | 0 |
| Learning-rate trials | lr | any | lr1e-04, lr2e-04 | 0 |

Students whose structure depends on the A01 decision use `structure: adopted` with the job's
`adopted_structures: {wspace, astar}`, or their own `structure: {adopted: {wspace, astar}}` (mirror/final/heal/
students.py resolve_structure); the file actually used is in each run's spec.json.
"The main Mirror arm" is `main/<model>/mirror_exact_wspace` if results/A01/decision.json says `"adopted": true`,
otherwise `main/<model>/mirror_exact_astar`. Gate files: `results/A01/gate_p.json` (`"pass": true|false`) and
`results/A01/decision.json` (`"adopted": true|false`), written by `scripts/fp_gate_p.py` and
`scripts/fp_a01_decide.py`.

### 4.5 Results

Each experiment writes `results/<ID>/` (A01 ... A15b): CSV tables, `summary.md`, `metrics.json` and `meta.json`.
Checkpoints and features stay on the VM under `runs/final/` and `cache/`. Only tables, summaries, figures and logs are
pulled to the laptop.

## 5. Scripts (one entry point per job type)

| Script | Owner | Job |
|---|---|---|
| `scripts/fp_cache.py` | infra | build or check data roles; stream the full train set |
| `scripts/fp_structure_fit.py` | structure | fit a SpanStructure on any fit role (fit3, coco) |
| `scripts/fp_a01_weightspace.py` | structure | fit A*_w, write its structure, zero-shot full conversion |
| `scripts/fp_a02_stability.py` | structure | agreement metrics across fits |
| `scripts/fp_a03_controls.py` | structure | C1 to C4 |
| `scripts/fp_heal.py` | heal | one multi-student process: several (method, variant) students, one seed, one model |
| `scripts/fp_eval_server.py`, `scripts/fp_eval.py` | eval | k-NN queue server and one-off evaluation |
| `scripts/fp_fullbank.py` | eval | full-train features for several models in one pass, full-bank k-NN, linear protocol |
| `scripts/fp_a09_audit.py` | audit | exactness audit |
| `scripts/fp_a12_bench.py` | audit | systems table and profile |
| `scripts/fp_a15a_siglip.py` | audit | SigLIP 2 whitened projection, zero-shot |
| `scripts/fp_runner.py` | infra | the runner over jobs/final_phase.yaml |

Every script: `--dry-run` (CPU, tiny model, no data) for the tests, `--out`, and resume (skip finished work).

## 6. GPU efficiency targets (Phase G)

- Data on the GPU: the 100k heal set (uint8, 256 px, 19.7 GB) and the evaluation bank and validation set (15 + 7.5 GB)
  are copied to the GPU once per process; crops and flips run on the GPU (mirror/heal.py augment_batch). Larger sets
  (the extended heal set) stream from pinned host memory with a prefetch thread.
- One teacher forward per micro-batch for all students; students run one after another inside the micro-batch loop,
  so peak activation memory is one student's.
- bf16 autocast, fused AdamW, optional `torch.compile` per student (off by default for students with irregular
  shapes); no `.item()` or host sync inside the step except the logged loss every `log_every` steps.
- Evaluation is asynchronous: the trainer only saves checkpoints and queues requests.
- The runner keeps the GPU at 85% utilisation or more with at least 10 GB free, and stops new heals at 13.5 VM-hours.
