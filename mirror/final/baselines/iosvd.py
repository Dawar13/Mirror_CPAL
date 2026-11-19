"""IO-SVD at the Mirror-exact budget (FINAL_PHASE_PLAN.md, A05): input-output whitened SVD with adaptive rank,
adapted to the ViT linear layers fc1, fc2, wq, wk, wv, wo of the folded model.

Sources (read on 19 November 2026; also listed in docs/final_baselines.md)
  paper  A. Abbasi, C. Thrash, H. Qin, H. Pirsiavash, S. Kolouri, "IO-SVD: Input-Output Whitened SVD for
         Adaptive-Rank LLM Compression", arXiv 2605.15626 (v1, 15 May 2026, the only version).
         https://arxiv.org/abs/2605.15626 , https://arxiv.org/html/2605.15626v1 : Sections 3.1 to 3.3, Eq. 1 to 14,
         Algorithms 1 and 2, Appendices C to E, Table 4.
  code   https://github.com/mint-vu/IO-SVD exists ("Official repository for the paper ...", created 15 May 2026) but
         holds no files: on 19 November 2026 the GitHub API reports size 0 and its commits endpoint answers HTTP 409
         (empty repository); issue #1 (18 May 2026) also reports the link as broken. Nothing was ported from code;
         everything below follows the paper, and what the paper leaves open is marked (verify).

The method as the paper states it (W [m, n] = [d_out, d_in], h = W x)
  objective  dJ ~ 1/2 || C^1/2 (W - W^) R^1/2 ||_F^2 (Eq. 4), R = E_t[x_t x_t^T] (uncentred input moment),
             C = E_t[J_t^T H_t J_t] (output curvature: H_t the Hessian of the KL to the uncompressed model's
             prediction over its top-K tokens, J_t = dz_t / dh_t), both damped: R + lam_R I, C + lam_C I.
  C in practice (Appendix C): H = A A^T with probes v_j = A e_j; g = d(v_j^T z)/dh by reverse-mode products and
             backward hooks; C = (1/N) sum_t sum_j g g^T, a full d_out x d_out matrix.
  truncation B = C^1/2 W R^1/2 (Eq. 5), B = U S V^T, W^ = C^-1/2 U_r S_r (R^-1/2 V_r)^T (Eq. 7, 8), stored as two
             factors (Eq. 11; the split is not stated).
  rank       scores I_i = |g_i s_i|, g_i = u_i^T G~ v_i, G~ = C^-1/2 G R^-1/2, G = dL_cal/dW (Eq. 9, 10); one min-heap
             over every matrix's current tail component; pop the lowest score, drop it, gain storage_gain(r) (Eq. 13,
             14: 0 while r > r* + 1, m n - r* (m + n) at r = r* + 1, m + n for r <= r*, r* = floor(m n / (m + n))),
             push the next tail if r > r_min = ceil(eta r*); stop when the removed parameters reach the target;
             a matrix with r > r* keeps its dense W (Algorithms 1, 2).

What is implemented here, and every departure
  * Same matrices, folding and budget as every baseline: the folded weights (norm affines and LayerScale folded,
    mirror.surgery.fold_mlp, mirror.attention.fold_attn), written back in absorbed form into a copy of the unfolded
    model, as mirror.baselines does. R comes from mirror.baselines.collect_stats (methods ["svdllm"]): M1 (fc1),
    M2 (fc2), Nin (wq, wk, wv), Mctx (wo), over the fit images through the original model (whether the paper takes
    R from the original or a partly compressed model is not stated (verify)).
  * Output metric for a ViT (adaptation). DINOv2 has no classifier, so the KL over top-K tokens has no direct
    analogue. C is the Gauss-Newton matrix of the healing objective's final terms: the normalised MSE on the
    final-norm CLS token and on the patch tokens (weights 1 / (d mean CLS^2) and 1 / (P d mean patch^2) per image,
    as mirror/heal.py distill_terms with batch normalisers), i.e. H = a diagonal weight per output entry.
    collect_extra_stats estimates C = (1/N_tokens) sum_t g_t g_t^T with g_t the gradient, at token t, of a random
    Gaussian probe of those outputs (unbiased; probes="exact" uses one probe per output entry and is exact), taken
    with respect to the output of each linear layer in the absorbed model (the folded matrix's own output). The
    gradient at token t includes every path through attention (the CLS output depends on all tokens); the paper
    does not say how it treats cross-token terms (verify). output="cls" uses the CLS term only.
  * Score. The paper's first-order score needs a calibration loss with a non-zero gradient at the uncompressed
    weights. The paper never defines L_cal (verify); its own objective (Eq. 1, a KL to the uncompressed model) has
    zero gradient there, and DINOv2 has no task head. Default score "second_order": s_i^2, which is twice the
    increase of the paper's own objective (Eq. 4, 6) when component i is dropped, so it is comparable across
    matrices. score "first_order" (|g_i s_i|, Eq. 10) is available when collect_extra_stats gets a loss_fn
    (for example a classifier's cross-entropy on train labels).
  * Unstated constants (verify): lam_R = damp_in x mean diag R and lam_C = damp_out x mean diag C (1e-6 each, the
    ridge of our svdllm; the folded norm outputs are mean-zero over channels, so R is singular, and every stream
    write is invisible along the all-ones direction, so C of fc2 and wo is singular too); eta (0.1; grid {0.1, 0.5});
    r_min at least 1; square roots by symmetric eigendecomposition (the truncated W^ does not depend on the choice);
    factor split A = C^-1/2 U_r S_r^1/2 [m, r], V = S_r^1/2 V_r^T R^-1/2 [r, n] (balanced, as SVD-LLM's code).
  * Budget. The greedy runs over all converted matrices against budget.total (the Mirror-matched count, basis
    changes included) and rounds towards more parameters: it stops before the removal that would take the total
    below the budget (the paper stops once the removed amount reaches its target), so 0 <= gap < max(m + n)
    (+0.006% at DINOv2-B). At the integer tie r = r* = m n / (m + n) (every 768 x 768 matrix of DINOv2-B) the
    matrix stays dense, since the factors would cost the same; the paper's "dense iff r > r*" would truncate it for
    no saving. allocation "matched" (an ablation, not the method) keeps svdllm's per-matrix ranks.
  * Not implemented: the loss-aware 8-bit remapping of factor rows (Section 3.3), half-prune quantisation and KV
    cache compression; no method here is quantised.
  * Dense matrices keep W and get full-rank balanced SVD factors as given factors, so
    heal_structured.make_trainable(..., lowrank_factors="given") always works (such a matrix then trains as a
    full-rank factorisation holding r (m + n) entries for its m n count); dense_as_width(info, n_heads) rewrites
    those entries so that they train as plain dense matrices with exactly m n entries.
"""
from __future__ import annotations

import copy
import heapq
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from mirror import baselines as B
from mirror.budget import allocate, allocation_totals, gap_of, gap_ok, normalise_components

METHOD = "iosvd"
SCORES = ("second_order", "first_order")
ALLOCATIONS = ("adaptive", "matched")
OUTPUTS = ("final", "cls")
PROBES = ("gaussian", "exact")
DEFAULT_HPARAMS = {"damp_in": 1e-6, "damp_out": 1e-6, "eta": 0.1, "score": "second_order", "allocation": "adaptive"}
HPARAM_GRID = [{**DEFAULT_HPARAMS, "eta": 0.1}, {**DEFAULT_HPARAMS, "eta": 0.5}]
SOURCES = {
    "paper": "https://arxiv.org/abs/2605.15626 (v1, 15 May 2026); https://arxiv.org/html/2605.15626v1 Eq. 1-14, "
             "Algorithms 1-2, Appendices C-E",
    "code": "https://github.com/mint-vu/IO-SVD exists but is empty on 19 November 2026 (API size 0; commits HTTP 409); "
            "nothing ported from code",
}
_FOLD_KEY = {"fc1": "W1", "fc2": "W2", "wq": "Wq", "wk": "Wk", "wv": "Wv", "wo": "Wo"}

__all__ = ["METHOD", "DEFAULT_HPARAMS", "HPARAM_GRID", "SOURCES", "IOExtraStats", "collect_extra_stats",
           "resolve_hparams", "stats_needed", "io_factorise", "adaptive_ranks", "compress", "dense_as_width"]


# ----------------------------------------------------------------------------------------------
# Output-side statistics
# ----------------------------------------------------------------------------------------------
@dataclass
class IOExtraStats:
    blocks: list
    components: tuple
    C: dict = field(default_factory=dict)    # "<b>.<name>" -> [out, out] float64, sum over (token, probe) of g g^T
    G: dict = field(default_factory=dict)    # "<b>.<name>" -> [out, in] float64, dL_cal/dW summed over the batches
    n_tokens: int = 0
    n_images: int = 0
    n_probes: int = 0
    c_count: int = 0                         # C / c_count = mean over tokens of J^T H J
    info: dict = field(default_factory=dict)

    def C_mean(self, key: str) -> torch.Tensor:
        if key not in self.C:
            raise KeyError(f"output statistic C[{key!r}] missing; collect it with collect_extra_stats for this block "
                           "and component")
        return self.C[key] / max(int(self.c_count), 1)


def stats_needed(components) -> set:
    """Input statistics (mirror.baselines.collect_stats) IO-SVD reads: those of "svdllm" (M1, M2, Nin, Mctx)."""
    return B.needs_v2(["svdllm"], components)


def collect_extra_stats(adapter, loader, device, blocks, components, *, n_probes: int = 8, probes: str = "gaussian",
                        output: str = "final", loss_fn=None, seed: int = 0, stats_device=None,
                        to_cpu: bool = True) -> IOExtraStats:
    """Output curvature C per converted matrix (and, with loss_fn, the calibration gradient G), from one pass of
    `loader` (fit images, (uint8 images, labels) batches as for collect_stats) through an absorbed copy of
    adapter.model (norm affines and LayerScale folded into the weights of `blocks`; the same function), in the
    model's dtype without autocast, sums in float64.

    probes "gaussian": n_probes probes xi ~ N(0, I) per batch over the output entries (seeded, drawn on the CPU);
    "exact": one probe per output entry (exact; for tests and tiny models). output "final": the final-norm CLS token
    and patch tokens, weighted as the healing loss; "cls": the CLS token only. loss_fn(adapter, model, x, labels) ->
    scalar, run on the absorbed copy from the normalised images x, enables score "first_order"."""
    from mirror.attention import absorb_attn, attn_dims
    from mirror.data import normalize_for
    from mirror.surgery import absorb_block

    if probes not in PROBES:
        raise ValueError(f"probes must be one of {PROBES}")
    if output not in OUTPUTS:
        raise ValueError(f"output must be one of {OUTPUTS}")
    comps = normalise_components(components)
    blocks = sorted({int(b) for b in blocks})
    L = len(adapter.model.blocks)
    if not blocks or blocks[0] < 0 or blocks[-1] >= L:
        raise ValueError(f"blocks must lie in [0, {L - 1}]")
    m = copy.deepcopy(adapter.model).eval()
    m.requires_grad_(False)
    for b in blocks:
        absorb_block(m.blocks[b])
        absorb_attn(m.blocks[b])
    sdev = device if stats_device is None else stats_device
    st = IOExtraStats(blocks=blocks, components=comps, n_probes=int(n_probes) if probes == "gaussian" else 0)
    mode = {"what": None}
    inputs: dict = {}
    handles = []

    def add(table: dict, key: str, value: torch.Tensor) -> None:
        table[key] = value if key not in table else table[key] + value

    def attach(module: nn.Module, parts: list, in_key: str) -> None:
        def fwd(_mod, args, out):
            if mode["what"] == "G":
                inputs[in_key] = args[0].detach()
            if not out.requires_grad:
                return

            def on_grad(g):
                for key, sl in parts:
                    gs = g if sl is None else g[..., sl]
                    gg = gs.reshape(-1, gs.shape[-1]).to(device=sdev, dtype=torch.float64)
                    if mode["what"] == "C":
                        add(st.C, key, gg.T @ gg)
                    elif mode["what"] == "G":
                        X = inputs[in_key].reshape(-1, inputs[in_key].shape[-1]).to(device=sdev, dtype=torch.float64)
                        add(st.G, key, gg.T @ X)

            out.register_hook(on_grad)

        handles.append(module.register_forward_hook(fwd))

    for b in blocks:
        blk = m.blocks[b]
        if "mlp" in comps:
            attach(blk.mlp.fc1, [(f"{b}.fc1", None)], f"{b}.fc1")
            attach(blk.mlp.fc2, [(f"{b}.fc2", None)], f"{b}.fc2")
        if "attn" in comps:
            H, dh = attn_dims(blk)
            D = H * dh
            attach(blk.attn.qkv, [(f"{b}.wq", slice(0, D)), (f"{b}.wk", slice(D, 2 * D)),
                                  (f"{b}.wv", slice(2 * D, 3 * D))], f"{b}.qkv")
            attach(blk.attn.proj, [(f"{b}.wo", None)], f"{b}.proj")
    gen = torch.Generator().manual_seed(int(seed))
    n_pre = int(adapter.n_prefix_tokens)

    def run(x):
        t = adapter.layer0_tokens(x, m)
        for blk in m.blocks:
            t = blk(t)
        return m.norm(t)

    n_img = n_tok = 0
    weights_seen = []
    try:
        for imgs, labels in loader:
            x = normalize_for(adapter, imgs.to(device, non_blocking=True)).detach().requires_grad_(True)
            with torch.enable_grad():
                final = run(x)
                cls, patch = final[:, 0], final[:, n_pre:]
                d, P = int(cls.shape[-1]), int(patch.shape[1])
                terms = [(cls, 1.0 / (d * float(cls.detach().double().pow(2).mean())))]
                if output == "final":
                    terms.append((patch, 1.0 / (P * d * float(patch.detach().double().pow(2).mean()))))
                weights_seen.append([w for _, w in terms])
                mode["what"] = "C"
                if probes == "gaussian":
                    for p in range(int(n_probes)):
                        phi = 0.0
                        for t, w in terms:
                            xi = torch.randn(t.shape, generator=gen, dtype=t.dtype).to(t.device)
                            phi = phi + math.sqrt(w) * (xi * t).sum()
                        torch.autograd.grad(phi, x, retain_graph=p < int(n_probes) - 1)
                else:
                    coords = [(i, k) for i, (t, _) in enumerate(terms) for k in range(t[0].numel())]
                    for c, (i, k) in enumerate(coords):
                        t, w = terms[i]
                        phi = math.sqrt(w) * t.reshape(t.shape[0], -1)[:, k].sum()
                        torch.autograd.grad(phi, x, retain_graph=c < len(coords) - 1)
                mode["what"] = None
                del final, cls, patch, terms
            if loss_fn is not None:
                mode["what"] = "G"
                with torch.enable_grad():
                    xg = x.detach().requires_grad_(True)
                    loss = loss_fn(adapter, m, xg, labels.to(device) if torch.is_tensor(labels) else labels)
                    torch.autograd.grad(loss, xg)
                mode["what"] = None
                inputs.clear()
            n_img += int(x.shape[0])
            n_tok += int(x.shape[0]) * int(adapter.n_tokens)
    finally:
        mode["what"] = None
        for hd in handles:
            hd.remove()
    if n_img == 0:
        raise ValueError("the loader is empty")
    st.n_images, st.n_tokens = n_img, n_tok
    st.c_count = n_tok * (int(n_probes) if probes == "gaussian" else 1)
    if to_cpu:
        st.C = {k: v.cpu() for k, v in st.C.items()}
        st.G = {k: v.cpu() for k, v in st.G.items()}
    st.info = {"output": output, "probes": probes, "n_probes": st.n_probes, "seed": int(seed),
               "first_order_loss": None if loss_fn is None else getattr(loss_fn, "__name__", repr(loss_fn)),
               "n_images": n_img, "n_tokens": n_tok, "blocks": [b + 1 for b in blocks], "components": list(comps),
               "model_form": "absorbed copy of the original (norm affines and LayerScale folded into the weights of "
                             "the converted blocks; the same function); gradients taken at each linear layer's output",
               "weights": "CLS 1 / (d mean CLS^2), patch 1 / (P d mean patch^2) per image, per batch",
               "precision": "model dtype forward and backward, float64 sums", "stats_device": str(sdev),
               "mean_weights_first_batch": weights_seen[0] if weights_seen else None}
    return st


# ----------------------------------------------------------------------------------------------
# Factorisation and rank allocation
# ----------------------------------------------------------------------------------------------
def resolve_hparams(hparams: dict | None) -> dict:
    hp = {**DEFAULT_HPARAMS, **(hparams or {})}
    bad = set(hp) - set(DEFAULT_HPARAMS)
    if bad:
        raise ValueError(f"{METHOD}: unknown hyperparameters {sorted(bad)} (allowed {sorted(DEFAULT_HPARAMS)})")
    if hp["score"] not in SCORES:
        raise ValueError(f"{METHOD}: score must be one of {SCORES}")
    if hp["allocation"] not in ALLOCATIONS:
        raise ValueError(f"{METHOD}: allocation must be one of {ALLOCATIONS}")
    if not 0.0 <= float(hp["eta"]) <= 1.0:
        raise ValueError(f"{METHOD}: eta must lie in [0, 1]")
    for k in ("damp_in", "damp_out", "eta"):
        hp[k] = float(hp[k])
    return hp


def _sym_roots(M: torch.Tensor, damp: float) -> tuple:
    """((M + lam I)^1/2, (M + lam I)^-1/2, lam) for a symmetric PSD M, lam = damp * mean(diag M), by eigh."""
    M = 0.5 * (M + M.T)
    n = M.shape[0]
    md = float(torch.diagonal(M).mean())
    lam = float(damp) * (md if md > 0 else 1.0)
    ev, Q = torch.linalg.eigh(M + lam * torch.eye(n, dtype=M.dtype, device=M.device))
    ev = ev.clamp_min(lam if lam > 0 else torch.finfo(M.dtype).tiny)
    s = ev.sqrt()
    return (Q * s) @ Q.T, (Q / s) @ Q.T, lam


def io_factorise(W: torch.Tensor, R: torch.Tensor, C: torch.Tensor, damp_in: float = 1e-6,
                 damp_out: float = 1e-6) -> dict:
    """IO-whitened SVD of W [m, n] (float64): B = C^1/2 W R^1/2 = U S V^T (Eq. 5). Returns U, S, Vh, the square
    roots and their inverses, and the two ridges. reconstruct() turns it into factors at a rank."""
    W = W.to(torch.float64)
    Rs, Ris, lam_r = _sym_roots(R.to(W), damp_in)
    Cs, Cis, lam_c = _sym_roots(C.to(W), damp_out)
    U, S, Vh = torch.linalg.svd(Cs @ W @ Rs, full_matrices=False)
    return {"U": U, "S": S, "Vh": Vh, "Cs": Cs, "Cis": Cis, "Rs": Rs, "Ris": Ris, "ridge_in": lam_r,
            "ridge_out": lam_c}


def _reconstruct(f: dict, r: int) -> tuple:
    """(W^, A, V) at rank r: W^ = C^-1/2 U_r S_r V_r^T R^-1/2 (Eq. 8), A = C^-1/2 U_r S_r^1/2,
    V = S_r^1/2 V_r^T R^-1/2."""
    sq = f["S"][:r].sqrt()
    A = (f["Cis"] @ f["U"][:, :r]) * sq
    V = sq[:, None] * (f["Vh"][:r] @ f["Ris"])
    return A @ V, A, V


def _scores(f: dict, score: str, G: torch.Tensor | None) -> torch.Tensor:
    S = f["S"]
    if score == "second_order":
        return S * S
    Gt = f["Cis"] @ G.to(S) @ f["Ris"]                      # whitened gradient (Eq. 9)
    g = (f["U"] * (Gt @ f["Vh"].T)).sum(0)                  # u_i^T G~ v_i
    return (g * S).abs()                                    # Eq. 10


def adaptive_ranks(shapes, scores, total_target: int, eta: float = 0.1) -> dict:
    """The greedy truncation of Algorithms 1 and 2 at a parameter budget, rounding towards more parameters.

    shapes: [(m, n)] per matrix; scores: per matrix, one score per singular component in descending-singular-value
    order (length min(m, n)). Every matrix starts at full rank (dense, m n parameters). A min-heap holds each
    matrix's current tail component (score, matrix index; ties to the lower index); the lowest is popped and dropped
    and the matrix's next tail is pushed while r > r_min = max(1, ceil(eta r*)). Storage follows Eq. 14: m n while
    r (m + n) >= m n, else r (m + n), so the gains are 0 above r* + 1, m n - r* (m + n) at r* + 1 and m + n below
    (r* = floor(m n / (m + n))). Tie convention: when m n / (m + n) is an integer (768 x 768: r* = 384), a matrix at
    r = r* costs m n either way and is kept dense (as mirror.budget does), where the paper's "dense iff r > r*" would
    store the truncated factors at the same count. The loop stops before the removal that would take the total below
    total_target (or when every matrix is at its floor). Returns ranks, dense flags, total, r_star, r_min, the number
    of removed components and the reason for stopping."""
    k = len(shapes)
    sc = [torch.nan_to_num(torch.as_tensor(s, dtype=torch.float64).cpu(), nan=float("inf")).tolist() for s in scores]
    r = [min(int(a), int(b)) for a, b in shapes]
    if any(len(sc[j]) < r[j] for j in range(k)):
        raise ValueError("each matrix needs one score per singular component")
    rstar = [(int(a) * int(b)) // (int(a) + int(b)) for a, b in shapes]
    rmin = [max(1, int(math.ceil(float(eta) * rs))) for rs in rstar]

    def params(j: int, rr: int) -> int:
        a, b = int(shapes[j][0]), int(shapes[j][1])
        return a * b if rr * (a + b) >= a * b else rr * (a + b)

    total = sum(params(j, r[j]) for j in range(k))
    heap = [(sc[j][r[j] - 1], j) for j in range(k) if r[j] > rmin[j]]
    heapq.heapify(heap)
    removed, stop = 0, "every matrix at its floor"
    while heap:
        s, j = heapq.heappop(heap)
        gain = params(j, r[j]) - params(j, r[j] - 1)
        if total - gain < int(total_target):
            stop = "the next removal would go below the budget"
            break
        r[j] -= 1
        total -= gain
        removed += 1
        if r[j] > rmin[j]:
            heapq.heappush(heap, (sc[j][r[j] - 1], j))
    dense = [r[j] * (int(shapes[j][0]) + int(shapes[j][1])) >= int(shapes[j][0]) * int(shapes[j][1]) for j in range(k)]
    return {"ranks": r, "dense": dense, "total": int(total), "r_star": rstar, "r_min": rmin, "removed": removed,
            "stop": stop}


# ----------------------------------------------------------------------------------------------
# Compression
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def compress(model: nn.Module, blocks, components, budget, stats, extra: IOExtraStats, hparams: dict | None = None,
             *, verbose: bool = True) -> tuple:
    """(compressed deep copy, info) for IO-SVD at `budget` (mirror.budget.SpanBudget for these blocks and
    components). `model` is the original network (adapter.model); `stats` holds the input moments
    (mirror.baselines.collect_stats with methods ["svdllm"]); `extra` the output curvature (collect_extra_stats).
    None of them is modified. info follows the compressor contract of mirror.baselines.compress (per_matrix kind
    "lowrank" with U_key and V_key, tensors, params_total, budget_total, gap, gap_ok, gap_step) and adds per-matrix
    r_star, r_min, the share of IO-whitened energy discarded and the predicted increase of the Eq. 4 objective,
    "allocation_trace", the model-level counts (params_model, params_frac, target, gap_model, eligible), the
    statistics' info and "sources"."""
    from mirror.attention import fold_attn, write_absorbed_attn
    from mirror.surgery import fold_mlp, write_absorbed_mlp

    from .dense_students import model_level

    hp = resolve_hparams(hparams)
    comps = normalise_components(components)
    blocks = [int(b) for b in blocks]
    if list(budget.blocks) != blocks or tuple(budget.components) != comps:
        raise ValueError(f"budget is for blocks {budget.blocks} and {budget.components}, not {blocks} and {comps}")
    if hp["score"] == "first_order" and not extra.G:
        raise ValueError("score 'first_order' needs the calibration gradient: collect_extra_stats(..., loss_fn=...)")
    n_tok = max(int(stats.n_tokens), 1)
    cache = {"block": None, "f": None}             # one block's folded float64 weights at a time

    def folded(b: int) -> dict:
        if cache["block"] != b:
            f = {}
            if "mlp" in comps:
                f.update(fold_mlp(model.blocks[b]))
            if "attn" in comps:
                f.update(fold_attn(model.blocks[b]))
            cache["block"], cache["f"] = b, f
        return cache["f"]

    def factor(mb) -> dict:
        b, name = int(mb.block), mb.name
        W = folded(b)[_FOLD_KEY[name]]
        mom, _, key = B._input_stats(name, b)
        R = B._stat(stats, mom, key).to(device=W.device, dtype=torch.float64) / n_tok
        C = extra.C_mean(f"{b}.{name}").to(device=W.device, dtype=torch.float64)
        return io_factorise(W, R, C, hp["damp_in"], hp["damp_out"])

    # pass 1: spectra and scores (factorisations are recomputed in pass 2, so memory stays at one matrix)
    shapes, scores, spectra = [], [], []
    for mb in budget.matrices:
        f = factor(mb)
        G = extra.G.get(f"{mb.block}.{mb.name}") if hp["score"] == "first_order" else None
        if hp["score"] == "first_order" and G is None:
            raise KeyError(f"calibration gradient G[{mb.block}.{mb.name}] missing")
        shapes.append(tuple(int(x) for x in mb.shape))
        scores.append(_scores(f, hp["score"], G).cpu())
        spectra.append((f["S"] ** 2).cpu())
        del f
    if hp["allocation"] == "adaptive":
        trace = adaptive_ranks(shapes, scores, int(budget.total), hp["eta"])
        ranks, dense = trace["ranks"], trace["dense"]
        targets = [None] * len(shapes)
    else:
        al = allocate("svdllm", budget)
        by = {(a["block"], a["name"]): a for a in al}
        ranks = [int(by[(mb.block, mb.name)]["rank"]) for mb in budget.matrices]
        dense = [bool(by[(mb.block, mb.name)]["dense_storage"]) for mb in budget.matrices]
        targets = [int(by[(mb.block, mb.name)]["target"]) for mb in budget.matrices]
        trace = {"ranks": ranks, "dense": dense, "note": "matched: svdllm's per-matrix ranks (mirror.budget)"}
    # pass 2: write the compressed weights and the given factors, block by block
    order = [int(mb.block) for mb in budget.matrices]
    if order != sorted(order):
        raise ValueError("budget.matrices must be grouped by block (mirror.budget builds them that way)")
    m = copy.deepcopy(model)
    tensors, per = {}, []
    new: dict = {}
    predicted = 0.0

    def write(b: int) -> None:
        blk = m.blocks[b]
        if "mlp" in comps:
            write_absorbed_mlp(blk, new["W1"], new["b1"], new["W2"], new["b2"])
        if "attn" in comps:
            write_absorbed_attn(blk, **{k: new[k] for k in ("Wq", "Wk", "Wv", "bq", "bk", "bv", "Wo", "bo")})

    current = None
    for j, mb in enumerate(budget.matrices):
        b, name = int(mb.block), mb.name
        if b != current:
            if current is not None:
                write(current)
            current, new = b, dict(folded(b))
        o, i = shapes[j]
        key = f"{b}.{name}"
        e2 = spectra[j]
        tot = float(e2.sum())
        if dense[j]:
            W = folded(b)[_FOLD_KEY[name]]
            U, S, Vh = torch.linalg.svd(W.to(torch.float64), full_matrices=False)
            sq = S.sqrt()
            A, V = U * sq, sq[:, None] * Vh
            r, params, disc, inc = min(o, i), o * i, 0.0, 0.0
            ridge_in = ridge_out = None
        else:
            f = factor(mb)
            r = int(ranks[j])
            W_new, A, V = _reconstruct(f, r)
            new[_FOLD_KEY[name]] = W_new
            params = r * (o + i)
            disc = float(e2[r:].sum()) / tot if tot > 0 else 0.0
            inc = 0.5 * float(e2[r:].sum())
            ridge_in, ridge_out = f["ridge_in"], f["ridge_out"]
            del f
        predicted += inc
        tensors[key + ".U"] = A.to(torch.float32).cpu()
        tensors[key + ".V"] = V.to(torch.float32).cpu()
        per.append({"block": b, "name": name, "kind": "lowrank", "shape": [o, i], "rank": int(r),
                    "dense_storage": bool(dense[j]), "params": int(params), "target": targets[j],
                    "U_key": key + ".U", "V_key": key + ".V", "r_star": (o * i) // (o + i),
                    "r_min": None if hp["allocation"] != "adaptive" else int(trace["r_min"][j]),
                    "io_discarded_energy": disc, "predicted_objective_increase": inc, "score": hp["score"],
                    "ridge_in": ridge_in, "ridge_out": ridge_out})
    if current is not None:
        write(current)
    m.eval()
    tot = allocation_totals(per)
    gap = gap_of(tot["total"], budget.total)
    step = int(max(sum(mb.shape) for mb in budget.matrices))
    step_rel = step / int(budget.total)
    ok = gap_ok(gap, step_rel)
    note = ("within the rule (0 <= gap <= 1%)" if ok and gap <= 0.01 else
            "within one allocation step (1% not reachable at this size)" if ok else
            "shortfall or overshoot: see allocation_trace")
    info = {"method": METHOD, "hparams": dict(hp), "components": comps, "blocks": blocks,
            "params_matrices": tot["matrices"], "params_basis_change": tot["basis_change"],
            "params_total": tot["total"], "budget_total": int(budget.total), "gap": float(gap), "gap_ok": bool(ok),
            "gap_step": step, "gap_reachable": bool(step_rel <= 0.01), "gap_note": note,
            "params_biases": int(budget.biases_dense), "budget_matrices": int(budget.matrix_total),
            "budget_basis_change": int(budget.basis_change), "budget_variant": budget.variant, "per_matrix": per,
            "basis": [], "tensors": tensors, "allocation_trace": {k: v for k, v in trace.items() if k != "ranks"},
            "predicted_objective_increase": predicted,
            "model_form": "absorbed weights in the unfolded model (norm affine and LayerScale folded); factors "
                          "A = C^-1/2 U_r S_r^1/2 and V = S_r^1/2 V_r^T R^-1/2",
            "statistics": {"input": dict(stats.info), "output": dict(extra.info)}, "sources": dict(SOURCES)}
    info.update(model_level(model, budget, tot["total"], step))
    if verbose:
        from mirror.utils import log

        log(f"{METHOD} {hp}: matrices {tot['total']:,} against budget {int(budget.total):,} (gap {100 * gap:+.4f}%; "
            f"{note}); model {info['params_model']:,} against target {info['target']:,} (gap "
            f"{100 * info['gap_model']:+.4f}%); {sum(dense)} dense matrices")
    return m, info


def dense_as_width(info: dict, n_heads: int) -> dict:
    """Optional, before heal_structured.make_trainable: rewrite every dense-storage "lowrank" entry (a matrix the
    allocation kept dense) as a "width" entry that keeps every index, so the student trains that matrix as a plain
    dense matrix with m n entries instead of a full-rank factorisation with r (m + n). Weights and counts are
    unchanged; mirror.baselines.structure_check and make_trainable accept the result (dense nn.Linear for an MLP
    whose fc1 and fc2 are both dense, IndexedLinear over every index otherwise). Works for any low-rank compressor's
    info (svdllm_full as well). Modifies info in place and returns it."""
    for pm in info["per_matrix"]:
        if pm.get("kind") != "lowrank" or not pm.get("dense_storage"):
            continue
        o, i = (int(x) for x in pm["shape"])
        name = pm["name"]
        if name in ("fc1", "fc2"):
            n = o if name == "fc1" else i
            pm.update(kind="width", unit="neuron", n=n, kept=list(range(n)))
        else:
            n = o if name in ("wq", "wk", "wv") else i
            pm.update(kind="width", unit="head", n=n, kept=list(range(n)), heads=int(n_heads),
                      heads_kept=list(range(int(n_heads))))
        pm["params"] = o * i
        pm["dense_form"] = "width with every index kept (dense trainable matrix)"
    return info
