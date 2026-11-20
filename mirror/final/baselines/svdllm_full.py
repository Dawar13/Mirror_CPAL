"""Full SVD-LLM at the Mirror-exact budget (FINAL_PHASE_PLAN.md, A05): truncation-aware data whitening plus SVD-LLM's
second component, "parameter update with sequential low-rank approximation".

Sources (read on 19 November 2026; also listed in docs/final_baselines.md)
  paper  X. Wang, Y. Zheng, Z. Wan, M. Zhang, "SVD-LLM: Truncation-aware Singular Value Decomposition for Large
         Language Model Compression", ICLR 2025, arXiv 2403.07378.
         https://arxiv.org/abs/2403.07378 (v5, 16 March 2025, the current version) and
         https://arxiv.org/html/2403.07378v5 : Section 3.2, Algorithm 3, Table 5 and Appendix A.10;
         https://arxiv.org/abs/2403.07378v1 (12 March 2024): its abstract calls the second component a
         "layer-wise closed-form model parameter update".
  code   https://github.com/AIoT-MLSys-Lab/SVD-LLM at commit 7538cca98880ff312a79a16252aab5a7b480fbe9 (28 August
         2025): README.md, SVDLLM.py (profle_svdllm, whitening, whitening_local_update, class local_update) and
         utils/LoRA.py.

What the official second component is
  * Paper v5, Section 3.2 and Algorithm 3: after whitening, every compressed matrix is stored as W'_u [out, r] and
    W'_v [r, in]. Line 4: "Fix all W'_v, fine-tune all W'_u" with LoRA (W'_u <- W'_u + B_u A_u); line 5: "Fix all
    W'_u, fine-tune all W'_v" (W'_v <- W'_v + B_v A_v). The LoRA terms are merged into the factors, so the rank stays
    r. The paper's reason for the sequential order: fine-tuning both factors at once "will not guarantee a decrease in
    fine-tuning loss" (Appendix A.10 / Table 15: joint LoRA is worse). Table 5 reports that the order (U first or V
    first) changes little; as read, its "W'_v first" row at 20% equals the main result (7.73) while Algorithm 3 and
    the README put U first (verify which order produced the main tables). Data: Alpaca, 50K samples; the LoRA
    hyperparameters are not given in the paper.
  * README Step 2 (same commit): "We first update the compressed weight matrix U and then V with LoRA fine-tuning",
    command `python LoRA.py --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64` (data
    yahma/alpaca-cleaned). utils/LoRA.py itself runs one PEFT LoRA training (alpha 16, dropout 0.05, warm-up 100
    steps); its default target modules list both the *_u_proj and the *_v_proj layers, so the two rounds are two runs
    with different target modules (the scripting of the two rounds is not in the repository).
  * So the published second component is a gradient fine-tuning stage. It is implemented here as a training hook
    (SequentialFactorSchedule), and compress() itself is the whitening step.
  * SVDLLM.py also keeps the v1 component: `--step 2` runs whitening_local_update, a closed-form least-squares update
    of the left factor. Layers are processed in order; the calibration inputs reach layer i through the already
    compressed layers 0..i-1 (`inps = outs` after the compressed layer is re-run); inside layer i the hooks sit on the
    original linear layers, so target = original layer output on those inputs; for each linear with truncated
    (U_r, Sigma_r, V_r = VT_r S^-1): x = X V_r^T Sigma_r, U^T <- lstsq(x, Y), factors (U Sigma^1/2, Sigma^1/2 V_r).
    The update set is `--updating_nsamples` (default 16) sequences, separate from the 256 whitening sequences. The
    current README does not document this path. It is available here with hparams closed_form_update=True (default
    False, so the default is the published v5 method).

Mapping to this repository
  whitening       mirror.baselines.compress(..., "svdllm", ...) unchanged: L = chol(M + lam I), lam = damp * mean
                  diag M (ridge: the folded norm outputs are mean-zero over channels), W L = U S V^T, factors
                  U_f = U_r S_r^1/2 (= W'_u) and V_f = S_r^1/2 V_r^T L^-1 (= W'_v), the official whitening() split.
                  Ranks: the matched per-matrix allocation of mirror.budget (as the whitening-only baseline), so the
                  compressed start is identical to "svdllm" and only the training differs.
  given factors   every per_matrix entry carries U_key and V_key ("<b>.<name>.U" / ".V" in info["tensors"]), so
                  heal_structured.make_trainable(..., lowrank_factors="given") builds LowRankLinear(U = W'_u, V = W'_v).
                  A matrix stored dense by the allocation (rank r with r (m + n) >= m n; never at DINOv2-B sizes)
                  gets full-rank balanced SVD factors, so the given-factor form exists (it then holds r (m + n)
                  trainable entries for an m n count, as the existing balanced conversion does).
  LoRA_u, LoRA_v  SequentialFactorSchedule.begin_step(step, total, model): steps [0, split * total) train every
                  LowRankLinear.U of the converted matrices with every .V frozen; steps [split * total, total) the
                  reverse ("u_first", Algorithm 3); "v_first" swaps them (Table 5).
  closed form     closed_form_update(): the official whitening_local_update on our matrices, in the normal-equation
                  form U' = W M_c V^T (V M_c V^T)^-1 (the same least-squares solution as lstsq over all tokens, and
                  independent of how Sigma is split), M_c = sum x x^T of the matrix's input over the calibration
                  images with blocks before b already compressed and updated, and block b itself original.

Deviations (also in docs/final_baselines.md)
  1. No LoRA adapters: each round trains the factor itself (a full update of W'_u or W'_v; LoRA's B A is a rank-8
     restriction of that update). The low-rank structure is kept either way (U [out, r] times V [r, in]).
  2. One healing run for both rounds: the identical data, batches, steps (780) and learning-rate rule as every other
     method (one warm-up plus cosine over the whole run), split at split * total (default half: one epoch of the
     100k heal set per round). The official code runs two separate LoRA trainings, each with its own schedule, on
     Alpaca with a language-modelling loss; here the loss is the shared distillation objective.
  3. Biases of the converted matrices and the embeddings train in both rounds, as for every method (LoRA.py trains
     no bias, bias="none"; LLaMA has none).
  4. Inherited from the whitening baseline: the ridge, the matched per-matrix ranks (not one uniform ratio).
  5. closed_form_update (off by default): the update images are the calibration loader given (the official default
     is 16 separate sequences); the target excludes the bias (the bias stays dense and unchanged; for OPT the official
     target includes it).

Trainer contract for the hook (the multi-student trainer must honour it)
  * Build the optimiser over both factor groups (configure_trainable before the first begin_step; frozen_parameters
    returns nothing for this hook).
  * Call begin_step(step, total, model) at the start of every optimiser step, before the forward pass, with `total`
    the run's step budget (780; 200 for a learning-rate trial).
  * Zero gradients with set_to_none=True (mirror/heal.py does). begin_step also sets the frozen group's .grad to
    None, and AdamW skips parameters without a gradient, so a frozen factor gets no update and no weight decay.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from mirror import baselines as B

METHOD = "svdllm_full"
ORDERS = ("u_first", "v_first")
DEFAULT_HPARAMS = {"damp": B.DEFAULT_DAMP, "order": "u_first", "split": 0.5, "closed_form_update": False,
                   "stage_schedule": "shared"}
# stage_schedule "shared" (the pre-registered port, the default): one warm-up and cosine over the whole run, the
# second factor starting with a fresh optimiser state at about half the peak rate; in Phase G its loss jumped after
# step 390 (2.0 -> 11.2) and the student ended slightly worse than it began. "own" (tried on 20 November 2026 and
# abandoned): each stage with its own warm-up and cosine, as two separate fine-tunes; the second stage then reaches
# the full peak rate and diverged harder (2.1 -> 152). Both train the factors directly with AdamW; the official code
# updates them through LoRA adapters, which this port does not reproduce.
STAGE_SCHEDULES = ("own", "shared")
HPARAM_GRID = [dict(DEFAULT_HPARAMS)]           # one point, as the whitening-only "svdllm" (gate B)
RECORD_PATHS = {"fc1": "mlp.fc1", "fc2": "mlp.fc2", "wq": "attn.qkv.q", "wk": "attn.qkv.k", "wv": "attn.qkv.v",
                "wo": "attn.proj"}          # where heal_structured.make_trainable puts each matrix
SOURCES = {
    "paper": "https://arxiv.org/abs/2403.07378 (v5, ICLR 2025); https://arxiv.org/html/2403.07378v5 Section 3.2, "
             "Algorithm 3; v1 abstract https://arxiv.org/abs/2403.07378v1",
    "code": "https://github.com/AIoT-MLSys-Lab/SVD-LLM commit 7538cca98880ff312a79a16252aab5a7b480fbe9 "
            "(README.md Step 2, SVDLLM.py whitening / whitening_local_update / local_update, utils/LoRA.py)",
}

__all__ = ["METHOD", "DEFAULT_HPARAMS", "HPARAM_GRID", "SOURCES", "resolve_hparams", "stats_needed", "compress",
           "training_hooks", "SequentialFactorSchedule", "closed_form_update"]


def resolve_hparams(hparams: dict | None) -> dict:
    hp = {**DEFAULT_HPARAMS, **(hparams or {})}
    bad = set(hp) - set(DEFAULT_HPARAMS)
    if bad:
        raise ValueError(f"{METHOD}: unknown hyperparameters {sorted(bad)} (allowed {sorted(DEFAULT_HPARAMS)})")
    if hp["order"] not in ORDERS:
        raise ValueError(f"{METHOD}: order must be one of {ORDERS}, not {hp['order']!r}")
    if not 0.0 <= float(hp["split"]) <= 1.0:
        raise ValueError(f"{METHOD}: split must lie in [0, 1]")
    hp["split"], hp["damp"] = float(hp["split"]), float(hp["damp"])
    hp["closed_form_update"] = bool(hp["closed_form_update"])
    return hp


def stats_needed(components) -> set:
    """Statistics of mirror.baselines.collect_stats this method reads (those of "svdllm")."""
    return B.needs_v2(["svdllm"], components)


# ----------------------------------------------------------------------------------------------
# Compression (the whitening step, plus the optional v1 closed-form update)
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def _full_rank_factors(W: torch.Tensor) -> tuple:
    U, S, Vh = torch.linalg.svd(W.to(torch.float64), full_matrices=False)
    sq = S.sqrt()
    return (U * sq).to(torch.float32).cpu(), (sq[:, None] * Vh).to(torch.float32).cpu()


def compress(model: nn.Module, blocks, components, budget, stats, hparams: dict | None = None, calib=None,
             *, verbose: bool = True) -> tuple:
    """(compressed deep copy, info) for full SVD-LLM at `budget` (a mirror.budget.SpanBudget for these blocks and
    components). `model` is the original network (adapter.model); it and `stats` (mirror.baselines.collect_stats with
    methods ["svdllm"]) are not modified.

    The returned model is exactly the whitening-only "svdllm" compressor's (same ranks, same factors), unless
    hparams closed_form_update=True, which then needs calib = {"adapter", "loader", "device"} (fit images; see
    closed_form_update). info follows the compressor contract of mirror.baselines.compress (per_matrix entries of
    kind "lowrank" with U_key and V_key; tensors; params_total, budget_total, gap, gap_ok) and adds hparams (order,
    split), "schedule" (what training_hooks(info) will do), the model-level counts (params_model, params_frac,
    target, gap_model, eligible) and "sources"."""
    from .dense_students import model_level

    hp = resolve_hparams(hparams)
    m, info = B.compress(model, "svdllm", blocks, components, budget, stats, {"damp": hp["damp"]},
                         keep_tensors=True, verbose=False)
    tensors = info["tensors"]
    for pm in info["per_matrix"]:
        key = f"{int(pm['block'])}.{pm['name']}"
        pm["U_key"], pm["V_key"] = key + ".U", key + ".V"
        if pm.get("dense_storage"):
            tensors[key + ".U"], tensors[key + ".V"] = _full_rank_factors(B._read_matrix(m, pm["block"], pm["name"]))
            pm["factors_note"] = ("stored dense by the allocation; full-rank balanced SVD factors given so that the "
                                  "given-factor trainable form exists")
    upd = None
    if hp["closed_form_update"]:
        if calib is None:
            raise ValueError("closed_form_update needs calib={'adapter': ..., 'loader': ..., 'device': ...}")
        upd = closed_form_update(m, model, info, calib)
    first, second = ("U", "V") if hp["order"] == "u_first" else ("V", "U")
    info.update({
        "method": METHOD, "hparams": dict(hp), "base_method": "svdllm",
        "schedule": {"type": "SequentialFactorSchedule", "order": hp["order"], "split": hp["split"],
                     "stage_schedule": hp["stage_schedule"],
                     "rounds": [f"train every {first} factor, {second} frozen: steps [0, round(split * total))",
                                f"train every {second} factor, {first} frozen: steps [round(split * total), total)"],
                     "note": "applied by training_hooks(info) during healing (paper v5 Algorithm 3)"},
        "closed_form_update": upd, "sources": dict(SOURCES),
        "model_form": info["model_form"] + "; factors are SVD-LLM's W'_u = U_r S_r^1/2 and W'_v = S_r^1/2 V_r^T L^-1",
    })
    info.update(model_level(model, budget, info["params_total"], info["gap_step"]))
    if verbose:
        from mirror.utils import log

        log(f"{METHOD} {hp}: matrices {info['params_total']:,} against budget {info['budget_total']:,} (gap "
            f"{100 * info['gap']:+.4f}%); model {info['params_model']:,} against target {info['target']:,} (gap "
            f"{100 * info['gap_model']:+.4f}%, eligible {info['eligible']})")
    return m, info


def _matrix(f: dict, name: str) -> torch.Tensor:
    return f[{"fc1": "W1", "fc2": "W2", "wq": "Wq", "wk": "Wk", "wv": "Wv", "wo": "Wo"}[name]]


@torch.no_grad()
def closed_form_update(m: nn.Module, original: nn.Module, info: dict, calib, ridge: float = 0.0) -> list:
    """SVDLLM.py's whitening_local_update (paper v1's "layer-wise closed-form update") on the compressed model `m`
    (in place; info["tensors"] U factors replaced). For each converted block b in order: the calibration images run
    through m's embeddings and its already updated blocks before b, then through the ORIGINAL block b (absorbed form,
    so its linear layers see the folded inputs); M_c = sum x x^T of each matrix's input is accumulated in float64 and
    the left factor is refitted with the right factor fixed: U' = W M_c V^T (V M_c V^T + ridge I)^-1, W the original
    folded weight. Dense-storage matrices are left as they are. Returns one record per updated matrix with the
    relative output error on the calibration inputs, sqrt(tr(D M_c D^T) / tr(W M_c W^T)) for D = W - U V, before
    and after (after <= before: U' is the least-squares optimum for this V).

    calib: {"adapter": the original model's adapter (layer0_tokens and normalisation), "loader": iterable of
    (uint8 images, labels) as for collect_stats, "device": torch device (default the model's)}."""
    from mirror.attention import absorb_attn, fold_attn, write_absorbed_attn
    from mirror.data import normalize_for
    from mirror.surgery import absorb_block, fold_mlp, write_absorbed_mlp

    adapter, loader = calib["adapter"], calib["loader"]
    device = calib.get("device") or next(m.parameters()).device
    tensors = info["tensors"]
    by_block: dict = {}
    for pm in info["per_matrix"]:
        if pm["kind"] == "lowrank" and not pm.get("dense_storage"):
            by_block.setdefault(int(pm["block"]), []).append(pm)
    records = []
    for b in sorted(by_block):
        ref = copy.deepcopy(original.blocks[b])
        absorb_block(ref)
        absorb_attn(ref)
        ref.eval()
        names = {pm["name"] for pm in by_block[b]}
        slots = {"fc1": ref.mlp.fc1, "fc2": ref.mlp.fc2, "qkv": ref.attn.qkv, "proj": ref.attn.proj}
        want = ({"fc1"} & names) | ({"fc2"} & names) | ({"qkv"} if names & {"wq", "wk", "wv"} else set()) | \
               ({"proj"} if "wo" in names else set())
        mom: dict = {}
        handles = []

        def pre(_mod, args, key):
            X = args[0].reshape(-1, args[0].shape[-1]).to(torch.float64)
            mom[key] = mom.get(key, 0.0) + X.T @ X

        for key in want:
            handles.append(slots[key].register_forward_pre_hook(lambda mod, args, key=key: pre(mod, args, key)))
        n_img = 0
        try:
            for imgs, _ in loader:
                x = normalize_for(adapter, imgs.to(device, non_blocking=True))
                t = adapter.layer0_tokens(x, m)
                for i in range(b):
                    t = m.blocks[i](t)
                ref(t)
                n_img += int(x.shape[0])
        finally:
            for hd in handles:
                hd.remove()
        if n_img == 0:
            raise ValueError("the calibration loader is empty")
        blk = m.blocks[b]
        orig_mlp = fold_mlp(original.blocks[b]) if names & {"fc1", "fc2"} else None
        orig_att = fold_attn(original.blocks[b]) if names & {"wq", "wk", "wv", "wo"} else None
        new_mlp = fold_mlp(blk) if orig_mlp is not None else None
        new_att = fold_attn(blk) if orig_att is not None else None
        for pm in by_block[b]:
            name = pm["name"]
            W = _matrix(orig_mlp if name in ("fc1", "fc2") else orig_att, name)
            Mc = mom[{"fc1": "fc1", "fc2": "fc2", "wo": "proj"}.get(name, "qkv")].to(W.device)
            U = tensors[pm["U_key"]].to(device=W.device, dtype=torch.float64)
            V = tensors[pm["V_key"]].to(device=W.device, dtype=torch.float64)
            K = V @ Mc @ V.T
            K = 0.5 * (K + K.T)
            if ridge > 0:
                K = K + float(ridge) * float(torch.diagonal(K).mean()) * torch.eye(K.shape[0], dtype=K.dtype,
                                                                                    device=K.device)
            rhs = W @ Mc @ V.T
            U_new = torch.linalg.solve(K, rhs.T).T
            base = float(torch.einsum("ij,jk,ik->", W, Mc, W))

            def rel_err(Wc):
                D = W - Wc
                return (max(float(torch.einsum("ij,jk,ik->", D, Mc, D)), 0.0) / base) ** 0.5 if base > 0 else 0.0

            before, W_new = rel_err(U @ V), U_new @ V
            after = rel_err(W_new)
            target = new_mlp if name in ("fc1", "fc2") else new_att
            target[{"fc1": "W1", "fc2": "W2", "wq": "Wq", "wk": "Wk", "wv": "Wv", "wo": "Wo"}[name]] = W_new
            tensors[pm["U_key"]] = U_new.to(torch.float32).cpu()
            rec = {"block": b, "name": name, "rank": int(pm["rank"]), "rel_output_err_before": before,
                   "rel_output_err_after": after,
                   "u_rel_change": float((U_new - U).norm() / U.norm().clamp_min(1e-300)), "n_images": n_img}
            pm["closed_form_update"] = rec
            records.append(rec)
        if new_mlp is not None:
            write_absorbed_mlp(blk, **new_mlp)
        if new_att is not None:
            write_absorbed_attn(blk, **new_att)
    return records


# ----------------------------------------------------------------------------------------------
# The sequential low-rank update as a training hook
# ----------------------------------------------------------------------------------------------
class SequentialFactorSchedule:
    """Constraint-protocol object (mirror/heal.py) with the extra hook begin_step(step, total, model): SVD-LLM's
    "parameter update with sequential low-rank approximation" (paper v5 Algorithm 3) inside the shared healing run.

    order "u_first": steps [0, boundary) train every LowRankLinear.U (W'_u) of the converted matrices with every .V
    (W'_v) frozen; steps [boundary, total) the reverse. "v_first" swaps the two. boundary = round(split * total),
    clamped to [0, total]. Nothing is frozen for the whole run (frozen_parameters is empty, so the optimiser holds
    both groups) and nothing is projected (__call__ does nothing); the structure is kept by the factorised modules."""

    def __init__(self, entries, order: str = "u_first", split: float = 0.5, stage_schedule: str = "own"):
        if order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}")
        if stage_schedule not in STAGE_SCHEDULES:
            raise ValueError(f"stage_schedule must be one of {STAGE_SCHEDULES}")
        self.stage_schedule = stage_schedule
        if not 0.0 <= float(split) <= 1.0:
            raise ValueError("split must lie in [0, 1]")
        self.entries = [(int(e["block"]), str(e["name"])) for e in entries]
        if not self.entries:
            raise ValueError("no low-rank matrices to schedule")
        self.order, self.split = order, float(split)
        self.last = None

    # -- schedule -----------------------------------------------------------------------------------------------
    def boundary(self, total: int) -> int:
        return min(max(int(round(self.split * int(total))), 0), int(total))

    def trained_factor(self, step: int, total: int) -> str:
        first, second = ("U", "V") if self.order == "u_first" else ("V", "U")
        return first if int(step) < self.boundary(total) else second

    def modules(self, model: nn.Module) -> list:
        from mirror.heal import LowRankLinear

        out = []
        for b, name in self.entries:
            mod = model.blocks[b].get_submodule(RECORD_PATHS[name])
            if not isinstance(mod, LowRankLinear):
                raise TypeError(f"block {b} {name} is {type(mod).__name__}, not LowRankLinear; build the student with "
                                "heal_structured.make_trainable(..., lowrank_factors='given') first")
            out.append(mod)
        return out

    def lr_at(self, step: int, base_lr: float, total: int, warmup: int, min_lr_ratio: float = 0.0) -> float:
        """Learning rate of step `step` for this student (the lockstep trainer uses it instead of the run's).
        "own": each stage [0, boundary) and [boundary, total) is a run of its own: linear warm-up over the run's
        warm-up fraction of the stage, then cosine to the end of the stage. "shared": the run's schedule."""
        from mirror.heal import lr_at as run_lr_at

        total, step = int(total), int(step)
        if self.stage_schedule == "shared":
            return run_lr_at(step, base_lr, total, int(warmup), min_lr_ratio)
        b = self.boundary(total)
        start, n = (0, b) if step < b else (b, total - b)
        if n <= 0:                                  # split 0 or 1: one stage, the run's schedule
            return run_lr_at(step, base_lr, total, int(warmup), min_lr_ratio)
        w = int(round(n * int(warmup) / total)) if total > 0 else 0
        return run_lr_at(step - start, base_lr, n, min(w, n), min_lr_ratio)

    def factor_parameters(self, model: nn.Module) -> dict:
        mods = self.modules(model)
        return {"U": [mod.U for mod in mods], "V": [mod.V for mod in mods]}

    @torch.no_grad()
    def begin_step(self, step: int, total: int, model: nn.Module) -> dict:
        """Set requires_grad for this step: the trained factor group on, the other off (its .grad set to None)."""
        train = self.trained_factor(step, total)
        groups = self.factor_parameters(model)
        for which, params in groups.items():
            on = which == train
            for p in params:
                p.requires_grad_(on)
                if not on:
                    p.grad = None
        self.last = {"step": int(step), "total": int(total), "train": train, "boundary": self.boundary(total)}
        return dict(self.last)

    # -- constraint protocol ------------------------------------------------------------------------------------
    def frozen_parameters(self, model: nn.Module) -> list:
        return []

    def __call__(self, model: nn.Module) -> None:
        return None

    @torch.no_grad()
    def check(self, model: nn.Module) -> dict:
        mods = self.modules(model)
        return {"ok": True, "n_matrices": len(mods), "ranks": [int(mod.rank) for mod in mods], "order": self.order,
                "split": self.split, "last_step": self.last,
                "requires_grad": {"U": sorted({bool(mod.U.requires_grad) for mod in mods}),
                                  "V": sorted({bool(mod.V.requires_grad) for mod in mods})}}

    def describe(self) -> dict:
        return {"type": "SequentialFactorSchedule", "order": self.order, "split": self.split,
                "n_matrices": len(self.entries),
                "rule": "round 1 trains every W'_u (U) with W'_v (V) frozen, round 2 the reverse (u_first); "
                        "the split is at round(split * total) of the shared healing run",
                "source": SOURCES["paper"]}


def training_hooks(info: dict) -> list:
    """The hooks a healing run of `info`'s student needs: one SequentialFactorSchedule over every low-rank matrix,
    with the order and split from info["hparams"] (defaults u_first, 0.5)."""
    hp = {**DEFAULT_HPARAMS, **(info.get("hparams") or {})}
    entries = [pm for pm in info["per_matrix"] if pm["kind"] == "lowrank"]
    return [SequentialFactorSchedule(entries, order=hp["order"], split=float(hp["split"]),
                                     stage_schedule=str(hp["stage_schedule"]))]
