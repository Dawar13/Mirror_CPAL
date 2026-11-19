"""DINOv2 linear evaluation on cached features (A10, FINAL_PHASE_PLAN.md v2), plus the Tier-2 probe on GPU-held data.

The DINOv2 protocol, as read on 19 November 2026 in facebookresearch/dinov2 at commit
7764ea0f912e53c92e82eb78a2a1631e92725fc8 (the head of main, and the hub code of notes/paper/method_spec.md):
  dinov2/eval/linear.py (last changed in 81b2b6419385a321287de91e00282ef7cbd26f94)
      https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/eval/linear.py
  dinov2/eval/utils.py (ModelWithIntermediateLayers), dinov2/models/vision_transformer.py (get_intermediate_layers),
  dinov2/data/transforms.py, dinov2/eval/metrics.py, dinov2/eval/setup.py, dinov2/configs/ssl_default_config.yaml,
  dinov2/run/submit.py, README.md (same commit).
- Features: ModelWithIntermediateLayers calls get_intermediate_layers(images, n=max(n_last_blocks_list) = 4,
  return_class_token=True) with norm=True under autocast and inference_mode: each of the last 4 block outputs is passed
  through the model's final norm; class token = out[:, 0]; patch tokens = out[:, 1 + num_register_tokens:] (registers
  excluded). create_linear_input(x_tokens_list, use_n_blocks, use_avgpool) concatenates the class tokens of the last
  use_n_blocks blocks (in block order) and, with use_avgpool, the mean of the LAST block's patch tokens; .float().
- Grid: n_last_blocks in [1, 4] x avgpool in [False, True] x 13 learning rates [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4,
  1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1] = 52 linear classifiers (nn.Linear(out_dim, 1000), weight N(0, 0.01), bias
  0), each lr scaled by scale_lr: lr * (batch_size * world_size) / 256.
- Optimiser: one torch.optim.SGD over all classifiers (one parameter group each), momentum 0.9, weight_decay 0;
  CosineAnnealingLR(T_max = max_iter, eta_min = 0) stepped every iteration; loss = the SUM over classifiers of the
  mean cross-entropy, so every classifier sees the same batches and gets its own gradient.
- Length and batch: epochs 10 x epoch_length 1250 = max_iter 12,500 iterations; batch_size 128 per GPU. The launcher
  (dinov2/run/submit.py) defaults to --ngpus 8 --nodes 1 and the README runs the evaluation "on a single node", so the
  global batch is 1,024 (lr x 4) and 12,500 x 1,024 images = 10 passes over the 1.28M train images. The GPU count of
  the published runs is not stated in the code (verify).
- Training augmentation (make_classification_train_transform): RandomResizedCrop(224, bicubic; torchvision's default
  scale (0.08, 1) and ratio (3/4, 4/3), no scale argument is passed), RandomHorizontalFlip(0.5), ImageNet mean and
  std; sampler SHARDED_INFINITE. Evaluation transform: Resize(256, bicubic), CenterCrop(224).
- Autocast dtype: config compute_precision.teacher.backbone.mixed_precision.param_dtype (fp16 in
  ssl_default_config.yaml; the configs used for the released models are not checked here (verify)).
- Selection: every classifier is evaluated on the validation set (top-1, micro average: MetricType.MEAN_ACCURACY maps
  to "micro"), every 1,250 iterations and at the end; the reported number is the best classifier's top-1, the first
  maximum in grid order (n = 1 before 4, avgpool False before True, lr ascending).

Ours (run_linear_protocol) keeps the features, the grid, the lr scaling (global batch 1,024 by default), SGD with
momentum 0.9 and no weight decay, the cosine schedule, 12,500 iterations, the initialisation and the summed loss, and
trains all 52 classifiers in parallel on the same batches (one weight matrix per feature type, the learning rates as a
vector; tests check the update against torch.optim.SGD with CosineAnnealingLR). Documented deviations:
  1. cached features, no augmentation: each train image is the 224 px centre crop of the 256 px resize (as every
     cache of this project), not a RandomResizedCrop with a flip; the features are computed once (bf16 autocast, the
     project's precision, instead of DINOv2's fp16 default) and stored as float16;
  2. selection on a held-out 10% of the TRAINING features (class-stratified, as scripts/tier2_linear.py), never on
     validation: the grid is trained on the other 90%, the best hold-out top-1 picks (n, avgpool, lr) (ties: first
     in DINOv2's grid order), then the grid is trained again on all training features (refit "grid"; "chosen" trains
     the chosen classifier only) and the chosen classifier's validation top-1 is the reported number. DINOv2's own
     rule (best validation top-1 over the refitted grid) is also recorded, labelled, only for comparison with
     published numbers; it is never used for a choice;
  3. data order: a fresh seeded permutation of the training positions per pass, batches running on across passes
     (DINOv2: its sharded infinite sampler); the classifiers are scored once, at the end of training.

Also here: tier2_probe, the Tier-2 linear probe of scripts/tier2_linear.py (logistic regression on [CLS, mean patch
token], L-BFGS, weight decay chosen on a 10% split of the 100k bank, top-1 on the 50k validation images), run on the
GPU-held images of knn_fast.EvalData, optionally with the flip-averaged feature (MUST-10, A07).
"""
from __future__ import annotations

import importlib.util
import math
import sys
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_LINEAR = {
    "source": f"https://github.com/facebookresearch/dinov2/blob/{DINOV2_COMMIT}/dinov2/eval/linear.py",
    "commit": DINOV2_COMMIT,
    "linear_py_last_commit": "81b2b6419385a321287de91e00282ef7cbd26f94",
    "n_last_blocks_list": [1, 4],
    "avgpool": [False, True],
    "base_learning_rates": [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1],
    "lr_scaling": "lr = base_lr * batch_size * world_size / 256 (scale_lr)",
    "batch_size_per_gpu": 128,
    "gpus_default": "8 per node, 1 node (dinov2/run/submit.py defaults; the published runs' count is not stated: "
                    "verify)",
    "optimizer": "SGD, momentum 0.9, weight_decay 0",
    "schedule": "CosineAnnealingLR(T_max = max_iter, eta_min = 0), stepped every iteration",
    "iterations": "epochs 10 x epoch_length 1250 = 12,500",
    "init": "weight N(0, 0.01), bias 0",
    "loss": "sum over classifiers of nn.CrossEntropyLoss (mean over the batch)",
    "features": "get_intermediate_layers(n=4, return_class_token=True, norm=True): class tokens of the last n blocks "
                "(n in {1, 4}) after the final norm, optionally with the mean of the last block's patch tokens "
                "(registers excluded)",
    "train_augmentation": "RandomResizedCrop(224, bicubic, torchvision default scale (0.08, 1)), "
                          "RandomHorizontalFlip(0.5)",
    "eval_transform": "Resize(256, bicubic), CenterCrop(224)",
    "autocast": "teacher backbone param_dtype of the config (fp16 in ssl_default_config.yaml)",
    "selection": "best validation top-1 (micro) over all classifiers, first maximum in grid order",
}
DEVIATIONS = [
    "cached features without augmentation: 224 px centre crops of the 256 px resize, no RandomResizedCrop or flip; "
    "features computed once in bf16 autocast (DINOv2 default fp16) and stored as float16",
    "hyperparameters (n_last_blocks, avgpool, lr) chosen on a class-stratified 10% hold-out of the TRAINING features "
    "(grid trained on the other 90%), then refitted on all training features; DINOv2 picks the best classifier on the "
    "validation set (recorded separately, for comparison only)",
    "data order: seeded permutations of the training positions, batches running on across passes; classifiers scored "
    "once at the end of training (DINOv2 also scores every 1,250 iterations)",
]
N_LAST_STORED = 4                     # class tokens of the last 4 blocks are stored, then the mean patch token


# ----------------------------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------------------------
def last_block_outputs(adapter, x: torch.Tensor, model=None, n_last: int = N_LAST_STORED) -> list:
    """The residual stream after each of the last `n_last` blocks (before the final norm), oldest first: the
    adapter's layer0_tokens then the blocks, as adapter.forward_tokens (DINOv2: prepare_tokens_with_masks, then the
    blocks, as get_intermediate_layers)."""
    m = adapter.model if model is None else model
    t = adapter.layer0_tokens(x, m)
    L = len(m.blocks)
    outs = []
    for i, blk in enumerate(m.blocks):
        t = blk(t)
        if i >= L - int(n_last):
            outs.append(t)
    return outs


def linear_feature(adapter, x: torch.Tensor, model=None, n_last: int = N_LAST_STORED) -> torch.Tensor:
    """[class token of block L-n_last, ..., class token of block L-1, mean patch token of block L-1], every token after
    the model's final norm (for a Mirror model the final norm holds the even head), float32 [B, (n_last + 1) d]. Every
    DINOv2 classifier input is a contiguous column range of this vector (feature_types)."""
    m = adapter.model if model is None else model
    p = int(adapter.n_prefix_tokens)
    normed = [m.norm(o) for o in last_block_outputs(adapter, x, m, n_last)]
    return torch.cat([o[:, 0].float() for o in normed] + [normed[-1][:, p:].float().mean(dim=1)], dim=-1)


def feature_types(d: int, n_last_blocks_list=(1, 4), avgpool_list=(False, True), n_stored: int = N_LAST_STORED) -> list:
    """[(n_last_blocks, avgpool, start column, stop column)] in DINOv2's grid order, for the stored layout of
    linear_feature."""
    out = []
    for n in n_last_blocks_list:
        if not 1 <= int(n) <= n_stored:
            raise ValueError(f"n_last_blocks {n} is outside 1..{n_stored}")
        for avg in avgpool_list:
            start, stop = (n_stored - int(n)) * d, n_stored * d + (d if avg else 0)
            out.append((int(n), bool(avg), int(start), int(stop)))
    return out


# ----------------------------------------------------------------------------------------------
# The grid, trained in parallel
# ----------------------------------------------------------------------------------------------
@dataclass
class LinearConfig:
    n_last_blocks_list: tuple = (1, 4)
    avgpool_list: tuple = (False, True)
    base_lrs: tuple = tuple(DINOV2_LINEAR["base_learning_rates"])
    batch_size: int = 1024             # global batch (128 per GPU x 8 GPUs)
    iterations: int = 12500            # 10 epochs x 1250
    momentum: float = 0.9
    weight_decay: float = 0.0
    init_std: float = 0.01
    holdout_frac: float = 0.1
    seed: int = 0
    refit: str = "grid"                # "grid", "chosen" or "none"
    eval_chunk: int = 8192
    tf32: bool = False
    log_every: int = 1000
    extra: dict = field(default_factory=dict)

    def lrs(self) -> list:
        return [float(lr) * self.batch_size / 256.0 for lr in self.base_lrs]


class LinearGrid:
    """Every classifier of the grid: one weight matrix [D_t, G * C] per feature type t (G learning rates), trained by
    SGD with momentum on the summed loss, exactly as one torch.optim.SGD over separate nn.Linear classifiers with a
    CosineAnnealingLR schedule (the update is done by hand so the learning rate can be a vector)."""

    def __init__(self, d: int, num_classes: int, cfg: LinearConfig, device, types=None, lrs=None, seed=None,
                 dtype=torch.float32):
        self.cfg, self.C, self.device, self.dtype = cfg, int(num_classes), torch.device(device), dtype
        self.types = list(types) if types is not None else feature_types(d, cfg.n_last_blocks_list, cfg.avgpool_list)
        self.lrs = torch.tensor(list(lrs) if lrs is not None else cfg.lrs(), dtype=dtype)   # initial lr per classifier
        G = len(self.lrs)
        gen = torch.Generator().manual_seed(int(cfg.seed if seed is None else seed))
        self.W, self.b, self.bW, self.bb = [], [], [], []
        for (_, _, s, e) in self.types:
            w = torch.randn(e - s, G * self.C, generator=gen, dtype=torch.float64) * float(cfg.init_std)
            self.W.append(w.to(device=self.device, dtype=dtype))
            self.b.append(torch.zeros(G * self.C, device=self.device, dtype=dtype))
            self.bW.append(None)
            self.bb.append(None)
        self.lrs = self.lrs.to(self.device)
        self.G = G

    @torch.no_grad()
    def step(self, xb: torch.Tensor, yb: torch.Tensor, it: int, total: int) -> float:
        """One SGD step at iteration `it` of `total` (cosine factor (1 + cos(pi it / total)) / 2). Returns the summed
        loss of the batch (mean cross-entropy summed over classifiers) at logging iterations, else 0."""
        B, G, C = xb.shape[0], self.G, self.C
        lr = self.lrs * (0.5 * (1.0 + math.cos(math.pi * it / total)))
        ar = torch.arange(B, device=xb.device)
        loss = 0.0
        for t, (_, _, s, e) in enumerate(self.types):
            x = xb[:, s:e]
            logits = torch.addmm(self.b[t], x, self.W[t]).view(B, G, C)
            logp = torch.log_softmax(logits, dim=-1)
            loss += float(-logp[ar, :, yb].sum()) / B if (it % max(int(self.cfg.log_every), 1) == 0) else 0.0
            g = logp.exp_()
            g[ar, :, yb] -= 1.0
            g = g.view(B, G * C).div_(B)
            dW = x.T @ g
            db = g.sum(dim=0)
            if self.cfg.weight_decay:
                dW.add_(self.W[t], alpha=float(self.cfg.weight_decay))
            if self.bW[t] is None:
                self.bW[t], self.bb[t] = dW.clone(), db.clone()
            else:
                self.bW[t].mul_(self.cfg.momentum).add_(dW)
                self.bb[t].mul_(self.cfg.momentum).add_(db)
            D = e - s
            self.W[t].view(D, G, C).sub_(lr.view(1, G, 1) * self.bW[t].view(D, G, C))
            self.b[t].view(G, C).sub_(lr.view(G, 1) * self.bb[t].view(G, C))
        return loss

    @torch.no_grad()
    def logits(self, t: int, xb: torch.Tensor) -> torch.Tensor:
        s, e = self.types[t][2], self.types[t][3]
        return torch.addmm(self.b[t], xb[:, s:e], self.W[t]).view(xb.shape[0], self.G, self.C)

    def names(self) -> list:
        lrs = [float(v) for v in self.lrs.cpu()]
        return [f"n{n}_avgpool{int(a)}_lr{lr:.6g}" for (n, a, _, _) in self.types for lr in lrs]


def _as_tensor(X, device, chunk: int = 65536) -> torch.Tensor:
    """X (a tensor, an array or a read-only memmap) as a tensor on `device`, copied in row chunks (a 10 GB memmap is
    never held twice in host memory)."""
    if isinstance(X, torch.Tensor):
        return X.to(device)
    arr = np.asarray(X)
    first = torch.from_numpy(np.array(arr[:1]))
    out = torch.empty(arr.shape, dtype=first.dtype, device=device)
    for s in range(0, arr.shape[0], int(chunk)):
        out[s:s + int(chunk)] = torch.from_numpy(np.array(arr[s:s + int(chunk)])).to(device)
    return out


class BatchStream:
    """Batches of training positions: seeded permutations of `positions`, running on across passes."""

    def __init__(self, positions, batch_size: int, seed: int, device):
        self.pos = torch.as_tensor(np.asarray(positions, dtype=np.int64), device=device)
        self.bs, self.device = int(batch_size), device
        self.gen = torch.Generator().manual_seed(int(seed))
        self.buf = self.pos[:0]

    def next(self) -> torch.Tensor:
        while self.buf.shape[0] < self.bs:
            perm = torch.randperm(self.pos.shape[0], generator=self.gen).to(self.pos.device)
            self.buf = torch.cat([self.buf, self.pos[perm]])
        out, self.buf = self.buf[:self.bs], self.buf[self.bs:]
        return out


def train_grid(X: torch.Tensor, y: torch.Tensor, positions, d: int, num_classes: int, cfg: LinearConfig, device,
               types=None, lrs=None, seed=None, log=None, dtype=torch.float32) -> LinearGrid:
    """Train the grid (or a sub-grid: `types`, `lrs`) on the rows `positions` of X (a float16 or float32 tensor on its
    feature device; batches are gathered there and moved to `device`)."""
    seed = int(cfg.seed if seed is None else seed)
    grid = LinearGrid(d, num_classes, cfg, device, types, lrs, seed, dtype)
    stream = BatchStream(positions, cfg.batch_size, seed + 1, X.device)
    T = int(cfg.iterations)
    t0 = time.time()
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
    try:
        for it in range(T):
            idx = stream.next()
            xb = X.index_select(0, idx).to(device, non_blocking=True).to(grid.dtype)
            yb = y.index_select(0, idx).to(device, non_blocking=True)
            loss = grid.step(xb, yb, it, T)
            if log is not None and cfg.log_every and it % int(cfg.log_every) == 0:
                log(f"    linear grid ({len(grid.types)} feature types x {grid.G} lrs): iteration {it}/{T}, summed "
                    f"loss {loss:.3f}, {time.time() - t0:.0f} s")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return grid


@torch.no_grad()
def evaluate_grid(grid: LinearGrid, X: torch.Tensor, y: torch.Tensor, positions=None, chunk: int = 8192) -> dict:
    """Top-1 and top-5 (%) of every classifier on the rows `positions` (default all) of X, in grid.names() order."""
    n_rows = X.shape[0] if positions is None else len(positions)
    pos = None if positions is None else torch.as_tensor(np.asarray(positions, dtype=np.int64), device=X.device)
    T, G = len(grid.types), grid.G
    c1 = torch.zeros(T, G, dtype=torch.float64)
    c5 = torch.zeros(T, G, dtype=torch.float64)
    k5 = min(5, grid.C)
    for s in range(0, n_rows, int(chunk)):
        idx = torch.arange(s, min(s + int(chunk), n_rows), device=X.device) if pos is None else pos[s:s + int(chunk)]
        xb = X.index_select(0, idx).to(grid.device).to(grid.dtype)
        yb = y.index_select(0, idx).to(grid.device)
        for t in range(T):
            top = grid.logits(t, xb).topk(k5, dim=2).indices
            c1[t] += (top[:, :, 0] == yb[:, None]).sum(0).double().cpu()
            c5[t] += (top == yb[:, None, None]).any(dim=2).sum(0).double().cpu()
    return {"top1": (100.0 * c1 / max(n_rows, 1)).reshape(-1).tolist(),
            "top5": (100.0 * c5 / max(n_rows, 1)).reshape(-1).tolist(), "n": int(n_rows)}


@torch.no_grad()
def predict_one(grid: LinearGrid, flat: int, X: torch.Tensor, chunk: int = 8192) -> np.ndarray:
    """Top-1 predictions of classifier number `flat` (grid.names() order) on every row of X."""
    t, g = divmod(int(flat), grid.G)
    out = []
    for s in range(0, X.shape[0], int(chunk)):
        xb = X[s:s + int(chunk)].to(grid.device).to(grid.dtype)
        out.append(grid.logits(t, xb)[:, g].argmax(dim=1).cpu())
    return torch.cat(out).numpy()


def stratified_holdout(labels, frac: float = 0.1, seed: int = 0) -> tuple:
    """Fixed class-stratified split of positions 0..N-1: about `frac` of every class held out (at least one per class
    with two or more images). The algorithm of scripts/tier2_linear.stratified_holdout (tested equal)."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    hold = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        if len(idx) < 2:
            continue
        n_hold = min(max(int(round(frac * len(idx))), 1), len(idx) - 1)
        hold.append(rng.permutation(idx)[:n_hold])
    hold = np.sort(np.concatenate(hold)) if hold else np.zeros(0, dtype=np.int64)
    train = np.setdiff1d(np.arange(len(labels)), hold)
    return train.astype(np.int64), hold.astype(np.int64)


def run_linear_protocol(Xtr, ytr, Xval, yval, d: int, num_classes: int, cfg: LinearConfig | None = None, device="cpu",
                        feature_device=None, log=None) -> dict:
    """The protocol of the module docstring on cached features Xtr [N, 5d] and Xval [M, 5d] (linear_feature layout,
    float16 or float32 arrays or tensors). Returns the record (grid trials, choice, validation top-1, deviations) and
    the chosen classifier's validation predictions under "val_pred"."""
    cfg = cfg or LinearConfig()
    device = torch.device(device)
    fdev = torch.device(feature_device) if feature_device is not None else device
    t0 = time.time()
    X = _as_tensor(Xtr, fdev)
    y = torch.as_tensor(np.asarray(ytr), dtype=torch.long, device=fdev)
    Xv = _as_tensor(Xval, fdev)
    yv = torch.as_tensor(np.asarray(yval), dtype=torch.long, device=fdev)
    if X.shape[1] != (N_LAST_STORED + 1) * d or Xv.shape[1] != X.shape[1]:
        raise ValueError(f"features have {X.shape[1]} and {Xv.shape[1]} columns; the layout needs "
                         f"{(N_LAST_STORED + 1) * d} (d = {d})")
    tr, ho = stratified_holdout(y.cpu().numpy(), cfg.holdout_frac, cfg.seed)
    if len(ho) == 0:
        raise ValueError("the hold-out split is empty (every class needs two or more training images)")
    say = log or (lambda m: None)
    say(f"  linear protocol: {len(tr):,} training rows, {len(ho):,} hold-out rows, {Xv.shape[0]:,} validation rows; "
        f"{len(feature_types(d, cfg.n_last_blocks_list, cfg.avgpool_list))} feature types x {len(cfg.base_lrs)} lrs, "
        f"{cfg.iterations} iterations of {cfg.batch_size}")
    sel_grid = train_grid(X, y, tr, d, num_classes, cfg, device, log=log)
    ho_ev = evaluate_grid(sel_grid, X, y, ho, cfg.eval_chunk)
    names = sel_grid.names()
    flat_cfg = [(n, a, base, lr) for (n, a, _, _) in sel_grid.types for base, lr in zip(cfg.base_lrs, cfg.lrs())]
    best = max(range(len(names)), key=lambda i: (round(ho_ev["top1"][i], 9), -i))       # first maximum in grid order
    val_sel = evaluate_grid(sel_grid, Xv, yv, None, cfg.eval_chunk)
    trials = [{"name": names[i], "n_last_blocks": flat_cfg[i][0], "avgpool": flat_cfg[i][1],
               "base_lr": flat_cfg[i][2], "lr": flat_cfg[i][3], "holdout_top1": ho_ev["top1"][i],
               "val_top1_selection_split": val_sel["top1"][i]} for i in range(len(names))]
    chosen = dict(trials[best])
    say(f"  linear protocol: chosen {chosen['name']} (hold-out top-1 {chosen['holdout_top1']:.2f})")
    out = {"protocol": "DINOv2 linear evaluation on cached features", "dinov2": DINOV2_LINEAR, "deviations": DEVIATIONS,
           "config": asdict(cfg), "lrs": cfg.lrs(), "chosen": chosen, "n_train": int(X.shape[0]),
           "n_selection_train": int(len(tr)), "n_holdout": int(len(ho)), "n_val": int(Xv.shape[0]), "d": int(d),
           "num_classes": int(num_classes), "val_top1_chosen_selection_split": val_sel["top1"][best]}
    if cfg.refit == "none":
        fin, fin_index, fin_ev = sel_grid, best, val_sel
        out["refit"] = "none: the classifier trained on the 90% split is reported"
    else:
        all_rows = np.arange(X.shape[0])
        if cfg.refit == "grid":
            fin = train_grid(X, y, all_rows, d, num_classes, cfg, device, log=log)
            fin_index = best
        elif cfg.refit == "chosen":
            t = best // sel_grid.G
            fin = train_grid(X, y, all_rows, d, num_classes, cfg, device, types=[sel_grid.types[t]],
                             lrs=[chosen["lr"]], log=log)
            fin_index = 0
        else:
            raise ValueError(f"refit must be grid, chosen or none, not {cfg.refit!r}")
        fin_ev = evaluate_grid(fin, Xv, yv, None, cfg.eval_chunk)
        out["refit"] = f"{cfg.refit}: trained again on all {X.shape[0]:,} training rows"
        if cfg.refit == "grid":
            for i, tr_ in enumerate(trials):
                tr_["val_top1_refit"] = fin_ev["top1"][i]
            dv = max(range(len(names)), key=lambda i: (round(fin_ev["top1"][i], 9), -i))
            out["val_top1_dinov2_rule"] = fin_ev["top1"][dv]
            out["dinov2_rule_classifier"] = names[dv]
            out["dinov2_rule_note"] = ("best validation top-1 over the refitted grid, DINOv2's own selection; for "
                                       "comparison with published numbers only, never used for a choice")
    out["val_top1"] = fin_ev["top1"][fin_index]
    out["val_top5"] = fin_ev["top5"][fin_index]
    out["trials"] = trials
    out["val_pred"] = predict_one(fin, fin_index, Xv, cfg.eval_chunk)
    out["val_labels"] = yv.cpu().numpy()
    out["seconds"] = round(time.time() - t0, 1)
    say(f"  linear protocol: validation top-1 {out['val_top1']:.2f} ({out['refit']}; {out['seconds']:.0f} s)")
    return out


# ----------------------------------------------------------------------------------------------
# Tier-2 probe (scripts/tier2_linear.py) on GPU-held images
# ----------------------------------------------------------------------------------------------
def load_tier2(gpu_frac: float | None = None):
    """scripts/tier2_linear.py as a module. Its import sets a 0.25 GPU memory cap; `gpu_frac` re-applies the
    caller's cap afterwards."""
    from mirror.utils import REPO_ROOT

    key = "_fp_eval_tier2_linear"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, REPO_ROOT / "scripts" / "tier2_linear.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            del sys.modules[key]
            raise
    if gpu_frac is not None and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(gpu_frac))
    return sys.modules[key]


@torch.no_grad()
def features_f16(adapter, model, images, feature_fn, amp, batch_size: int, device, idx=None) -> np.ndarray:
    """feature_fn(adapter, x, model) for every image, float16 [N, D] (scripts/tier2_linear.extract_features: bf16
    autocast on CUDA, float16 storage, an error if a value leaves the float16 range)."""
    from mirror.data import normalize_for

    from .knn_fast import autocast_for

    m = adapter.model if model is None else model
    m.eval()
    out = []
    lim = np.finfo(np.float16).max
    for imgs in images.batches(idx, batch_size, device):
        with autocast_for(amp, device):
            f = feature_fn(adapter, normalize_for(adapter, imgs), m)
        f = f.float()
        if not torch.isfinite(f).all():
            raise FloatingPointError("non-finite features")
        if float(f.abs().max()) > lim:
            raise OverflowError("feature values exceed the float16 range")
        out.append(f.to(torch.float16).cpu())
    return torch.cat(out).numpy()


def tier2_probe(adapter, model, data, device, *, flipavg: bool = False, amp: str | None = "bf16",
                batch_size: int = 256, seed: int = 0, wd_grid=None, gpu_frac: float | None = None, log=None) -> dict:
    """The Tier-2 probe of scripts/tier2_linear.py (run_probe, knn_check, unchanged) on [CLS, mean patch token] of the
    bank (training) and validation images of `data` (knn_fast.EvalData), or on its flip average (A07). Returns the
    scores and the validation predictions ("val_pred")."""
    from .flipavg import probe_feature, probe_feature_flipavg

    T2 = load_tier2(gpu_frac)
    fn = probe_feature_flipavg if flipavg else probe_feature
    t0 = time.time()
    Xb = features_f16(adapter, model, data.bank, fn, amp, batch_size, device)
    Xv = features_f16(adapter, model, data.val, fn, amp, batch_size, device)
    yb, yv = data.bank.labels.numpy(), data.val.labels.numpy()
    wd = T2.DEFAULT_WD_GRID if wd_grid is None else wd_grid
    grid = [float(v) for v in wd.split(",") if v.strip()] if isinstance(wd, str) else [float(v) for v in wd]
    solver = {"max_iter": 1000, "history_size": 20, "tol_grad": 1e-6}
    pr = T2.run_probe(Xb, yb, Xv, yv, data.num_classes, grid, 0.1, seed, True, device, solver)
    d_first = int(adapter.embed_dim)
    kc = T2.knn_check(Xb, yb, Xv, yv, d_first, data.num_classes, device)
    mean, std = (torch.as_tensor(pr[k]).to(device) for k in ("mean", "std"))
    Xt = T2.standardise(torch.as_tensor(Xv).to(device), mean, std).float()
    W, b = (torch.as_tensor(pr[k]).to(device).float() for k in ("W", "b"))
    pred = (Xt @ W + b).argmax(1).cpu().numpy()
    test, sel = pr["test"], pr["selection"]
    if log is not None:
        log(f"  Tier-2 probe{' (flip-averaged)' if flipavg else ''}: top-1 {test['top1']:.2f} (wd "
            f"{sel['chosen_wd']:.3g}), k-NN check {kc:.2f}")
    return {"top1": test["top1"], "top5": test["top5"], "chosen_wd": sel["chosen_wd"],
            "wd_at_grid_edge": sel["wd_at_grid_edge"], "knn_check_top1": kc, "n_bank": int(Xb.shape[0]),
            "n_val": int(Xv.shape[0]), "flip_average": bool(flipavg),
            "recipe": T2.FEATURE_RECIPE + ("; flip-averaged: (p(x) + p(mirror x)) / 2" if flipavg else ""),
            "amp": amp or "fp32", "wd_grid": grid, "seed": int(seed), "seconds": round(time.time() - t0, 1),
            "selection": sel, "val_pred": pred, "val_labels": yv}
