"""Fast k-NN evaluation for the final phase: the MUST-09 protocol with the images held on the GPU.

Protocol (unchanged from MUST-09: mirror/knn.py, scripts/must09_heal.py (CurveLogger, the fp32 pass),
scripts/flip_avg_baseline.knn_eval; notes/paper/method_spec.md Section 7):
- bank: the 100k "bank" role of cache/train224_uint8.npy (cache/train_index.npz `bank_rows`, `bank_labels`: 100
  ImageNet train images per class, resize 256 bicubic, centre crop 224), in the stored order, as
  mirror.protocol.Protocol.bank_loader;
- queries: all 50k validation images (mirror.data.load_val_cache), or for intermediate curve points the subset of
  MUST-09, mirror.protocol.Protocol.query_loader(n): the first n / 1000 images of every class in dataset order
  (make_splits(labels, {"q": n // 1000})). Here n // (number of classes): the same rule for ImageNet;
- feature: adapter.cls_feature(x, model), the CLS token after the final norm (for Mirror models the final norm holds
  the even head, so this is the even part), computed per batch exactly as mirror.knn.extract_features (the uint8 batch
  normalised with the adapter's statistics, the mirror taken on the normalised batch, the forward pass under the pass's
  autocast, F.normalize(f.float())); cosine similarity, k = 20, votes exp(sim / 0.07), argmax, computed exactly as
  mirror.knn.knn_predict (the same query chunks of 1,024 and the same operations). With the same batch size the
  predictions equal mirror.knn.evaluate_knn's bit for bit (tests/test_fp_eval_knn.py);
- passes: "bf16" (bf16 autocast for every forward pass: bank, queries and mirrored queries; the k-NN itself in fp32 with
  TF32 off, as MUST-09 ran with compute.tf32 false), "fp32" (no autocast, TF32 off for matmuls and convolutions: the
  measure for exact-invariance claims, PREREG_NEXT_PHASE.md Section 1), each optionally on fewer queries and a smaller
  bank (the first m bank images of every class, in bank order), and "bf16_flipavg" / "fp32_flipavg" (the flip-averaged
  feature of MUST-10: (f(x) + f(mirror x)) / 2 on the CLS after the final norm, before the L2 normalisation, for the
  bank and the queries alike; see flipavg.py);
- flip consistency: the share of queries whose prediction for x equals the prediction for mirror(x), bank fixed.
  For every disagreeing query the relative top-two vote margin (v1 - v2) / sum(v) of both copies is recorded
  (PREREG_NEXT_PHASE.md Section 1: a disagreement counts as a near-tie if the top-two class votes differ by less than
  1e-4 of the total vote). `max_vote_margin_rel` (a key scripts/collect_next_phase.find_margin reads) takes, for each
  disagreeing query, the larger margin of its two copies (both copies must be near-ties); the smaller one is kept as
  `max_vote_margin_rel_one_copy` (the convention of scripts/core06_orientation.py, a near-tie on one of the copies).
  The gap between the k-th and (k+1)-th similarity of each listed disagreement is recorded as well: rounding can also
  swap the 20th and 21st neighbours, which the vote margin does not describe.

Data on the device: ImageSet holds uint8 images as one [N, 3, H, W] tensor, copied once per process (bank 15.05 GB
and validation 7.53 GB at 224 px), on the GPU or in pinned host memory; batches are sliced on the device.

Outputs of evaluate_request (used by scripts/fp_eval.py and scripts/fp_eval_server.py): <out>/eval/<tag>.json (tag
step<S>, or "original" for a released model), <out>/knn/<tag>_<key>.npz with int16 arrays pred, pred_flip, labels
(the MUST-09 format of scripts/must09_heal.save_predictions; a query subset adds query_rows) and
<out>/eval/<tag>_meta.json (mirror.final.prereg.write_final_meta). Result keys: the pass name for all queries and the
full bank ("bf16", "fp32", ...), with the suffix _q<n> for a query subset and _b<m> for a bank of m images per class
("bf16_q10000", "fp32_q10000_b20"). A finished key with the same settings (protocol, pass, k, T, query and bank counts,
data, checkpoint sha256) is skipped on a rerun, so a stopped evaluation resumes where it stopped.

Cost per pass: n_bank + 2 n_query forward passes (bf16: 200k images for the full protocol, 120k with 10k queries); a
flip-averaged pass forwards 2 n_bank + 2 n_query images: the averaged feature of the mirrored query is the same sum of
the same two forward passes (mirror(mirror x) = x exactly), so it is not recomputed, and its flip consistency is 100%
by construction (recorded as such).

Sources: this repository only (mirror/knn.py, mirror/protocol.py, scripts/must09_heal.py, scripts/flip_avg_baseline.py,
scripts/must10_flipavg.py, PREREG_NEXT_PHASE.md Section 1, docs/final_phase_design.md Section 4.4).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mirror.data import make_splits, mirror, normalize_for, synthetic_dataset

PROTOCOL = "fp-knn-v1"
K, TEMPERATURE = 20, 0.07
KNN_CHUNK = 1024                      # query chunk of mirror.knn.knn_predict
PASSES = ("bf16", "fp32", "bf16_flipavg", "fp32_flipavg")
NEAR_TIE_REL = 1e-4                   # PREREG_NEXT_PHASE.md Section 1, rounding allowance
MAX_LISTED = 1000                     # disagreements listed one by one in the JSON
DEFAULT_LOADER = "mirror.final.heal.students:load_student"
REQUEST_DEFAULTS = {"run_dir": None, "step": None, "passes": ["bf16"], "n_query": None, "priority": 0,
                    "out_dir": None, "original": False, "model": None, "fp32_n_query": None,
                    "fp32_bank_per_class": None, "with_original": ["bf16"], "flip_average": False, "cost": None,
                    "experiment": None, "id": None,
                    "ckpt": None, "tags": None, "created_utc": None}     # informational keys of the trainer's EvalQueue


def _log(msg: str) -> None:
    from mirror.utils import log

    log(msg)


# ----------------------------------------------------------------------------------------------
# Passes, precision and small helpers
# ----------------------------------------------------------------------------------------------
def parse_pass(name: str) -> tuple:
    """(amp, flip_average) of a pass name: "bf16" -> ("bf16", False), "fp32_flipavg" -> (None, True)."""
    if name not in PASSES:
        raise ValueError(f"unknown pass {name!r}; known passes: {', '.join(PASSES)}")
    base, _, rest = name.partition("_")
    return ("bf16" if base == "bf16" else None), rest == "flipavg"


def result_key(pass_name: str, n_query: int | None = None, n_query_total: int | None = None,
               bank_per_class: int | None = None) -> str:
    """"bf16" for all queries and the full bank; "_q<n>" for a query subset; "_b<m>" for a smaller bank."""
    key = pass_name
    if n_query is not None and n_query_total is not None and int(n_query) < int(n_query_total):
        key += f"_q{int(n_query)}"
    if bank_per_class is not None:
        key += f"_b{int(bank_per_class)}"
    return key


@contextlib.contextmanager
def strict_fp32():
    """TF32 off for matmuls and convolutions and the 'highest' float32 matmul precision (MUST-09's compute.tf32 false
    and its fp32 pass); the previous settings are restored afterwards."""
    prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
            torch.get_float32_matmul_precision())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev[0], prev[1]
        torch.set_float32_matmul_precision(prev[2])


def autocast_for(amp: str | None, device) -> contextlib.AbstractContextManager:
    """bf16 autocast on CUDA only (CPU runs stay in the model's dtype), as mirror.knn.extract_features."""
    if amp == "bf16" and torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def jsonable(o):
    """JSON-friendly copy: numpy and torch scalars and arrays, Paths; non-finite floats become None."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, torch.Tensor):
        o = o.detach().cpu().numpy()
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return repr(o)


def write_json_atomic(path, obj) -> Path:
    """Write JSON through a temporary file and os.replace, so a reader never sees half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(jsonable(obj), indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def sha256_file(path, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def save_predictions(path, pred, pred_flip, labels, query_rows=None) -> Path:
    """The MUST-09 prediction file (int16 pred, pred_flip, labels; scripts/must09_heal.save_predictions), written
    atomically; a query subset also stores its validation rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"pred": np.asarray(pred).astype(np.int16), "pred_flip": np.asarray(pred_flip).astype(np.int16),
              "labels": np.asarray(labels).astype(np.int16)}
    if query_rows is not None:
        arrays["query_rows"] = np.asarray(query_rows, dtype=np.int64)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)
    return path


# ----------------------------------------------------------------------------------------------
# Images held on the device
# ----------------------------------------------------------------------------------------------
class ImageSet:
    """uint8 images as one contiguous [N, 3, H, W] tensor on its storage device (the GPU, or pinned host memory),
    with int64 labels (kept on the CPU). Batches are sliced (or gathered) on the storage device and moved to the
    compute device; their content and layout equal a DataLoader batch of mirror.data.ArrayImageDataset."""

    def __init__(self, images: torch.Tensor, labels, name: str = ""):
        if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"ImageSet needs uint8 [N, 3, H, W], got {images.dtype} {tuple(images.shape)}")
        self.images = images.contiguous()
        self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.long).cpu()
        if self.labels.shape[0] != self.images.shape[0]:
            raise ValueError(f"{self.labels.shape[0]} labels for {self.images.shape[0]} images")
        self.name = name

    @classmethod
    def from_array(cls, array, rows=None, labels=None, device="cpu", pin: bool = False, chunk: int = 1024,
                   name: str = "", log=None) -> "ImageSet":
        """Copy rows of a uint8 [M, H, W, 3] array (a memmap) to `device` in chunks. `labels` are the labels of the
        selected rows (in `rows` order)."""
        rows = np.arange(int(array.shape[0]), dtype=np.int64) if rows is None else np.asarray(rows, dtype=np.int64)
        if labels is None:
            raise ValueError("labels are needed")
        n, H, W = len(rows), int(array.shape[1]), int(array.shape[2])
        dev = torch.device(device)
        pin = bool(pin and dev.type == "cpu" and torch.cuda.is_available())
        out = torch.empty((n, 3, H, W), dtype=torch.uint8, device=dev, pin_memory=pin)
        t0 = time.time()
        for s in range(0, n, int(chunk)):
            r = rows[s:s + int(chunk)]
            block = torch.from_numpy(np.ascontiguousarray(array[r]))
            out[s:s + len(r)].copy_(block.to(dev).permute(0, 3, 1, 2))
        if log is not None:
            log(f"{name or 'images'}: {n:,} images ({out.numel() / 2 ** 30:.2f} GiB) on {dev}"
                f"{' (pinned)' if pin else ''} in {time.time() - t0:.0f} s")
        return cls(out, labels, name)

    def __len__(self) -> int:
        return int(self.images.shape[0])

    @property
    def nbytes(self) -> int:
        return int(self.images.numel())

    def labels_at(self, idx=None) -> torch.Tensor:
        return self.labels if idx is None else self.labels[torch.as_tensor(np.asarray(idx, dtype=np.int64))]

    def batches(self, idx=None, batch_size: int = 512, device=None):
        """uint8 [b, 3, H, W] batches on `device`, in order (all images, or the positions `idx`)."""
        dev = self.images.device if device is None else torch.device(device)
        if idx is None:
            for s in range(0, len(self), int(batch_size)):
                yield self.images[s:s + int(batch_size)].to(dev, non_blocking=True)
            return
        it = torch.as_tensor(np.asarray(idx, dtype=np.int64), device=self.images.device)
        for s in range(0, len(it), int(batch_size)):
            yield self.images.index_select(0, it[s:s + int(batch_size)]).to(dev, non_blocking=True)


class EvalData:
    """The k-NN bank and the validation queries, held once per process."""

    def __init__(self, bank: ImageSet, val: ImageSet, num_classes: int, source: str):
        self.bank, self.val, self.num_classes, self.source = bank, val, int(num_classes), str(source)

    @classmethod
    def from_cache(cls, cache_dir, device="cuda", pin: bool = True, chunk: int = 1024, log=None) -> "EvalData":
        """cache/train224_uint8.npy rows `bank_rows` (labels `bank_labels`, cache/train_index.npz) and all validation
        images (mirror.data.load_val_cache)."""
        from mirror.data import load_val_cache

        cache = Path(cache_dir)
        train = np.load(cache / "train224_uint8.npy", mmap_mode="r")
        with np.load(cache / "train_index.npz") as z:
            rows = np.asarray(z["bank_rows"], dtype=np.int64)
            labels = np.asarray(z["bank_labels"], dtype=np.int64)
        val, val_labels, _, meta = load_val_cache(cache)
        bank = ImageSet.from_array(train, rows, labels, device, pin, chunk, "k-NN bank", log)
        vset = ImageSet.from_array(val, None, np.asarray(val_labels, dtype=np.int64), device, pin, chunk,
                                   "validation", log)
        n_cls = int(meta.get("n_classes") or (int(max(labels.max(), val_labels.max())) + 1))
        return cls(bank, vset, n_cls, f"cache {cache.as_posix()} (bank: train_index bank_rows, {len(bank)} images; "
                                      f"queries: validation, {len(vset)} images)")

    @classmethod
    def synthetic(cls, num_classes: int = 4, bank_per_class: int = 8, val_per_class: int = 4, img_size: int = 56,
                  seed: int = 0, device="cpu") -> "EvalData":
        """Random uint8 images with a weak class tint (mirror.data.synthetic_dataset), split per class into a bank
        and queries (tests and --dry-run)."""
        images, labels = synthetic_dataset(num_classes, bank_per_class + val_per_class, img_size, seed=seed)
        sp = make_splits(labels, {"bank": bank_per_class, "val": val_per_class})
        bank = ImageSet.from_array(images, sp["bank"], labels[sp["bank"]], device, name="synthetic bank")
        val = ImageSet.from_array(images, sp["val"], labels[sp["val"]], device, name="synthetic queries")
        return cls(bank, val, num_classes, f"synthetic(classes={num_classes}, bank={bank_per_class}/class, "
                                           f"queries={val_per_class}/class, {img_size} px, seed={seed})")

    def query_idx(self, n_query: int | None):
        """None (all queries) or the MUST-09 subset: the first n_query // num_classes images of every class in
        dataset order (mirror.protocol.Protocol.query_loader)."""
        if n_query is None or int(n_query) >= len(self.val):
            return None
        per = int(n_query) // self.num_classes
        if per < 1:
            raise ValueError(f"n_query {n_query} is below one image per class ({self.num_classes} classes)")
        return make_splits(self.val.labels.numpy(), {"q": per})["q"]

    def bank_idx(self, per_class: int | None):
        """None (the full bank) or the first `per_class` bank images of every class, in bank order."""
        if per_class is None:
            return None
        return make_splits(self.bank.labels.numpy(), {"b": int(per_class)})["b"]


# ----------------------------------------------------------------------------------------------
# Features and the k-NN
# ----------------------------------------------------------------------------------------------
def _cls_feature(adapter, x, model):
    return adapter.cls_feature(x, model)


@torch.no_grad()
def extract(adapter, model, images: ImageSet, idx=None, batch_size: int = 512, device=None, amp=None,
            flip: bool = False, feature_fn=None, normalise: bool = True) -> torch.Tensor:
    """Features of the images (positions `idx`, default all) in order, as mirror.knn.extract_features: normalise the
    uint8 batch with the adapter's statistics, mirror the normalised batch if `flip`, run feature_fn (default
    adapter.cls_feature) under the pass's autocast, then F.normalize(f.float()) (skipped with normalise=False)."""
    m = adapter.model if model is None else model
    m.eval()
    device = images.images.device if device is None else torch.device(device)
    fn = feature_fn or _cls_feature
    out = []
    for imgs in images.batches(idx, batch_size, device):
        x = normalize_for(adapter, imgs)
        if flip:
            x = mirror(x)
        with autocast_for(amp, device):
            f = fn(adapter, x, m)
        f = f.float()
        out.append(F.normalize(f, dim=-1) if normalise else f)
    return torch.cat(out)


@torch.no_grad()
def extract_flipavg(adapter, model, images: ImageSet, idx=None, batch_size: int = 512, device=None, amp=None,
                    feature_fn=None) -> torch.Tensor:
    """Unit features of (f(x) + f(mirror x)) / 2 with f = feature_fn (default cls_feature): two forward passes of the
    same batch inside the pass's autocast, averaged in float32 and then L2-normalised, exactly as
    scripts/flip_avg_baseline.FlipAveragedAdapter inside mirror.knn.extract_features."""
    m = adapter.model if model is None else model
    m.eval()
    device = images.images.device if device is None else torch.device(device)
    fn = feature_fn or _cls_feature
    out = []
    for imgs in images.batches(idx, batch_size, device):
        x = normalize_for(adapter, imgs)
        with autocast_for(amp, device):
            f = 0.5 * (fn(adapter, x, m).float() + fn(adapter, mirror(x), m).float())
        out.append(F.normalize(f.float(), dim=-1))
    return torch.cat(out)


def vote_margin_rel(votes: torch.Tensor) -> torch.Tensor:
    """(largest vote - second largest vote) / total vote, per row (PREREG_NEXT_PHASE.md Section 1)."""
    if votes.shape[1] < 2:
        return torch.full((votes.shape[0],), float("inf"), device=votes.device)
    t2 = votes.topk(2, dim=1).values
    return (t2[:, 0] - t2[:, 1]) / votes.sum(dim=1).clamp_min(1e-30)


@torch.no_grad()
def knn_vote(bank_feats: torch.Tensor, bank_labels: torch.Tensor, query_feats: torch.Tensor, num_classes: int,
             k: int = K, temperature: float = TEMPERATURE, chunk: int = KNN_CHUNK) -> tuple:
    """(predictions, relative top-two vote margins) for unit features. The predictions are computed exactly as
    mirror.knn.knn_predict (same chunks, same operations)."""
    k = min(int(k), bank_feats.shape[0])
    preds, margins = [], []
    for s in range(0, query_feats.shape[0], int(chunk)):
        sim = query_feats[s:s + int(chunk)] @ bank_feats.T
        top_sim, top_idx = sim.topk(k, dim=1)
        weights = (top_sim / temperature).exp()
        votes = torch.zeros(sim.shape[0], num_classes, device=sim.device, dtype=weights.dtype)
        votes.scatter_add_(1, bank_labels[top_idx], weights)
        preds.append(votes.argmax(dim=1))
        margins.append(vote_margin_rel(votes))
    return torch.cat(preds), torch.cat(margins)


@torch.no_grad()
def kth_gap(bank_feats: torch.Tensor, query_feats: torch.Tensor, k: int = K) -> torch.Tensor:
    """Similarity of the k-th minus the (k+1)-th nearest bank image, per query (nan if the bank is too small)."""
    if bank_feats.shape[0] <= k or query_feats.shape[0] == 0:
        return torch.full((query_feats.shape[0],), float("nan"))
    top = (query_feats @ bank_feats.T).topk(int(k) + 1, dim=1).values
    return (top[:, k - 1] - top[:, k]).cpu()


def flip_summary(pred, pred_flip, labels, m_x, m_m, diff) -> dict:
    """Top-1, flip consistency and the near-tie record of the disagreements (CPU tensors in; `diff` is the largest
    absolute difference between the unit features of each query and of its mirror)."""
    agree = pred == pred_flip
    dis = ~agree
    nd = int(dis.sum())
    strict = torch.maximum(m_x, m_m)[dis]
    lenient = torch.minimum(m_x, m_m)[dis]
    return {"top1": 100.0 * float((pred == labels).float().mean()),
            "top1_mirrored_queries": 100.0 * float((pred_flip == labels).float().mean()),
            "flip_consistency": 100.0 * float(agree.float().mean()), "n_disagree": nd,
            "max_vote_margin_rel": float(strict.max()) if nd else None,
            "max_vote_margin_rel_one_copy": float(lenient.max()) if nd else None,
            "n_disagree_not_near_tie": int((strict >= NEAR_TIE_REL).sum()), "near_tie_rel": NEAR_TIE_REL,
            "feature_flip_max_abs": float(diff.max()) if diff.numel() else None,
            "feature_identical_pct": 100.0 * float((diff == 0).float().mean()) if diff.numel() else None}


@torch.no_grad()
def run_pass(adapter, model, data: EvalData, pass_name: str, *, n_query: int | None = None,
             bank_per_class: int | None = None, batch_size: int = 512, k: int = K,
             temperature: float = TEMPERATURE, device=None, bank_feats: torch.Tensor | None = None, feature_fn=None,
             chunk: int = KNN_CHUNK, max_listed: int = MAX_LISTED) -> dict:
    """One k-NN pass of `model` (adapter.model if None): bank features (or the given ones), queries and mirrored
    queries, predictions, flip consistency and the record of disagreements. Returns {"summary", "pred",
    "pred_flip", "labels", "query_rows", "bank_feats", "disagreements"}."""
    amp, flipavg = parse_pass(pass_name)
    device = data.bank.images.device if device is None else torch.device(device)
    model = adapter.model if model is None else model
    qidx, bidx = data.query_idx(n_query), data.bank_idx(bank_per_class)
    t0 = time.time()
    with strict_fp32():
        if bank_feats is None:
            if flipavg:
                bank_feats = extract_flipavg(adapter, model, data.bank, bidx, batch_size, device, amp, feature_fn)
            else:
                bank_feats = extract(adapter, model, data.bank, bidx, batch_size, device, amp, feature_fn=feature_fn)
        bl = data.bank.labels_at(bidx).to(bank_feats.device)
        if flipavg:
            qf = extract_flipavg(adapter, model, data.val, qidx, batch_size, device, amp, feature_fn)
            qf_flip = qf
        else:
            qf = extract(adapter, model, data.val, qidx, batch_size, device, amp, feature_fn=feature_fn)
            qf_flip = extract(adapter, model, data.val, qidx, batch_size, device, amp, flip=True, feature_fn=feature_fn)
        pred, m_x = knn_vote(bank_feats, bl, qf, data.num_classes, k, temperature, chunk)
        if flipavg:
            pred_flip, m_m = pred, m_x
        else:
            pred_flip, m_m = knn_vote(bank_feats, bl, qf_flip, data.num_classes, k, temperature, chunk)
        labels = data.val.labels_at(qidx)
        pred_c, pred_flip_c, m_x_c, m_m_c = pred.cpu(), pred_flip.cpu(), m_x.cpu(), m_m.cpu()
        diff = (qf_flip - qf).abs().amax(dim=1).cpu()
        summary = flip_summary(pred_c, pred_flip_c, labels, m_x_c, m_m_c, diff)
        rows = np.arange(len(data.val)) if qidx is None else np.asarray(qidx)
        dis = torch.nonzero(pred_c != pred_flip_c).flatten()[:int(max_listed)]
        listed = []
        if len(dis):
            dd = dis.to(qf.device)
            gx, gm = kth_gap(bank_feats, qf[dd], k), kth_gap(bank_feats, qf_flip[dd], k)
            for j, i in enumerate(dis.tolist()):
                listed.append({"query_row": int(rows[i]), "label": int(labels[i]), "pred": int(pred_c[i]),
                               "pred_flip": int(pred_flip_c[i]), "vote_margin_rel": float(m_x_c[i]),
                               "vote_margin_rel_mirrored": float(m_m_c[i]), "kth_sim_gap": float(gx[j]),
                               "kth_sim_gap_mirrored": float(gm[j])})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    fw = 2 if flipavg else 1
    summary.update({
        "pass": pass_name, "amp": amp or "fp32", "flip_average": flipavg, "n_bank": int(bank_feats.shape[0]),
        "n_query": int(qf.shape[0]), "k": int(k), "temperature": float(temperature), "batch_size": int(batch_size),
        "forward_passes_per_image": fw, "seconds": round(time.time() - t0, 2),
        "precision": ("bf16 autocast for every forward pass" if amp else "fp32 forward passes, no autocast")
                     + ", TF32 off; k-NN in fp32",
        "bank_subset": None if bank_per_class is None else f"first {int(bank_per_class)} bank images per class",
        "query_subset": None if qidx is None else (f"first {int(n_query) // data.num_classes} validation images "
                                                   "per class in dataset order (MUST-09 rule)"),
        "n_disagree_listed": len(listed)})
    if flipavg:
        summary["flip_consistency_by_construction"] = True
        summary["note"] = ("flip-averaged feature (f(x) + f(mirror x)) / 2 on the CLS after the final norm, before the "
                           "L2 normalisation, for the bank and the queries; the mirrored query has the same feature "
                           "(the same two forward passes), so flip consistency is 100% by construction")
    return {"summary": summary, "pred": pred_c, "pred_flip": pred_flip_c, "labels": labels,
            "query_rows": None if qidx is None else np.asarray(qidx), "bank_feats": bank_feats,
            "disagreements": listed}


# ----------------------------------------------------------------------------------------------
# Evaluator: data held once, the original model's bank features and results cached per model
# ----------------------------------------------------------------------------------------------
def model_fingerprint(model) -> str:
    """Short id of a model's weights (names, shapes and the first values of the first tensors)."""
    h = hashlib.sha1()
    for i, (name, p) in enumerate(model.state_dict().items()):
        if i >= 6:
            break
        h.update(name.encode())
        h.update(str(tuple(p.shape)).encode())
        h.update(p.detach().reshape(-1)[:256].float().cpu().numpy().tobytes())
    return h.hexdigest()[:12]


class KnnEvaluator:
    """Holds the data on the device and the settings; caches the original model's bank features and k-NN results
    per model (in memory, and on disk under cache_dir/knn_original/ when cache_dir is set)."""

    def __init__(self, data: EvalData, device, batch_size: int = 512, k: int = K, temperature: float = TEMPERATURE,
                 chunk: int = KNN_CHUNK, cache_dir=None, max_cached_banks: int = 6, log=None):
        self.data, self.device = data, torch.device(device)
        self.batch_size, self.k, self.temperature, self.chunk = int(batch_size), int(k), float(temperature), int(chunk)
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.max_cached_banks = int(max_cached_banks)
        self.log = log or _log
        self._banks: dict = {}
        self._orig: dict = {}

    def counts(self, pass_name: str, n_query=None, bank_per_class=None) -> tuple:
        qidx, bidx = self.data.query_idx(n_query), self.data.bank_idx(bank_per_class)
        nq = len(self.data.val) if qidx is None else len(qidx)
        nb = len(self.data.bank) if bidx is None else len(bidx)
        return nq, nb, result_key(pass_name, nq, len(self.data.val), bank_per_class)

    def settings(self, pass_name: str, n_query=None, bank_per_class=None, checkpoint_sha256=None) -> dict:
        nq, nb, _ = self.counts(pass_name, n_query, bank_per_class)
        return {"protocol": PROTOCOL, "pass": pass_name, "k": self.k, "temperature": self.temperature,
                "n_query": nq, "n_bank": nb, "data": self.data.source, "checkpoint_sha256": checkpoint_sha256}

    def run(self, adapter, model, pass_name: str, n_query=None, bank_per_class=None, bank_feats=None,
            feature_fn=None) -> dict:
        return run_pass(adapter, model, self.data, pass_name, n_query=n_query, bank_per_class=bank_per_class,
                        batch_size=self.batch_size, k=self.k, temperature=self.temperature, device=self.device,
                        bank_feats=bank_feats, feature_fn=feature_fn, chunk=self.chunk)

    def original(self, adapter, model_key: str, pass_name: str, n_query=None, bank_per_class=None) -> dict:
        """The original model's (adapter.model) summary for this pass, computed once per model and cached; its bank
        features are cached too, so another query count costs only the queries."""
        nq, nb, key = self.counts(pass_name, n_query, bank_per_class)
        settings = self.settings(pass_name, n_query, bank_per_class)
        if (model_key, key) in self._orig:
            return self._orig[(model_key, key)]
        disk = None
        if self.cache_dir is not None:
            disk = self.cache_dir / "knn_original" / _safe(model_key) / f"{key}.json"
            rec = read_json(disk)
            if rec and rec.get("settings") == jsonable(settings):
                self._orig[(model_key, key)] = rec["summary"]
                return rec["summary"]
        bkey = (model_key, pass_name, bank_per_class)
        r = self.run(adapter, adapter.model, pass_name, n_query, bank_per_class, bank_feats=self._banks.get(bkey))
        self._banks[bkey] = r["bank_feats"]
        while len(self._banks) > self.max_cached_banks:
            self._banks.pop(next(iter(self._banks)))
        summary = r["summary"]
        self._orig[(model_key, key)] = summary
        if disk is not None:
            save_predictions(disk.with_suffix(".npz"), r["pred"], r["pred_flip"], r["labels"], r["query_rows"])
            write_json_atomic(disk, {"settings": settings, "summary": summary, "model_key": model_key})
        self.log(f"original {model_key} [{key}]: k-NN {summary['top1']:.2f}, flip consistency "
                 f"{summary['flip_consistency']:.3f}% ({summary['seconds']:.0f} s; cached)")
        return summary


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))[:120]


# ----------------------------------------------------------------------------------------------
# Requests (docs/final_phase_design.md Section 4.4)
# ----------------------------------------------------------------------------------------------
def normalise_request(req: dict) -> dict:
    """Validated request with defaults. Fields: run_dir, step, passes (list or comma string), n_query (None or 50000
    = all, 10000 = the MUST-09 subset), priority, and optionally out_dir (default run_dir), original (true: the
    released model `model`, written to out_dir), model, fp32_n_query and fp32_bank_per_class (smaller fp32 passes),
    with_original (passes of the original model to add for reference: list, true = the same passes, false = none;
    default ["bf16"]), flip_average (true: add the _flipavg twin of every pass), cost (true: analytic GFLOPs; default
    on when a flip-averaged pass is asked for), experiment, id."""
    if not isinstance(req, dict):
        raise ValueError("a request must be a JSON object")
    unknown = sorted(set(req) - set(REQUEST_DEFAULTS))
    r = {**REQUEST_DEFAULTS, **req}
    passes = r["passes"]
    if isinstance(passes, str):
        passes = [p.strip() for p in passes.split(",") if p.strip()]
    passes = [str(p) for p in (passes or [])]
    if r.get("flip_average"):
        passes = passes + [f"{p}_flipavg" for p in passes if not p.endswith("_flipavg")]
    passes = list(dict.fromkeys(passes))
    if not passes:
        raise ValueError("the request names no pass")
    for p in passes:
        parse_pass(p)
    r["passes"] = passes
    for key in ("step", "n_query", "fp32_n_query", "fp32_bank_per_class", "priority"):
        if r.get(key) is not None:
            r[key] = int(r[key])
    if r["original"]:
        if not r.get("model") or not r.get("out_dir"):
            raise ValueError("an original-model request needs model and out_dir")
    elif not r.get("run_dir"):
        raise ValueError("a request needs run_dir (or original: true with model and out_dir)")
    wo = r.get("with_original")
    r["with_original"] = (list(passes) if wo is True else [] if not wo else
                          [str(p) for p in ([wo] if isinstance(wo, str) else wo)])
    for p in r["with_original"]:
        parse_pass(p)
    if r.get("cost") is None:
        r["cost"] = any(parse_pass(p)[1] for p in passes)
    if unknown:
        r["unknown_fields"] = unknown
    return r


def checkpoint_path(run_dir, step) -> Path | None:
    """<run_dir>/ckpt/step<S>.pt (docs/final_phase_design.md Section 4.4)."""
    if run_dir is None or step is None:
        return None
    return Path(run_dir) / "ckpt" / f"step{int(step)}.pt"


def model_name_of(info, run_dir, req: dict):
    """The model name from the loader's info, the run's spec.json, or the request."""
    for d in (info, (info or {}).get("spec") if isinstance(info, dict) else None):
        if isinstance(d, dict):
            for key in ("model", "model_name"):
                if d.get(key):
                    return str(d[key])
    spec = read_json(Path(run_dir) / "spec.json") if run_dir else None
    if isinstance(spec, dict):
        for key in ("model", "model_name"):
            if spec.get(key):
                return str(spec[key])
        ad = spec.get("adapter")
        if isinstance(ad, dict) and ad.get("name"):
            return str(ad["name"])
    return req.get("model")


def adapter_key(run_dir):
    """Cache key of a run's original-model adapter: its spec.json "adapter" entry (or None)."""
    spec = read_json(Path(run_dir) / "spec.json") if run_dir else None
    ad = spec.get("adapter") if isinstance(spec, dict) else None
    return "adapter:" + json.dumps(jsonable(ad), sort_keys=True) if ad else None


def _accepts(fn, name: str) -> bool:
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind == p.VAR_KEYWORD for p in params.values())


def resolve_callable(spec):
    """A callable, or "module:function" imported on first use."""
    if spec is None or callable(spec):
        return spec
    import importlib

    mod, _, fn = str(spec).partition(":")
    return getattr(importlib.import_module(mod), fn)


def default_build_original(model_name: str, device):
    """The released model's adapter (mirror.adapters_timm.build_model_adapter; DINOv2 through torch.hub under
    $HF_HOME). VM only: it downloads weights."""
    from mirror.adapters_timm import build_model_adapter

    ad = build_model_adapter(str(model_name), device=device)
    ad.model.float().eval()
    return ad


def load_for_request(r: dict, device, loader=None, build_original=None, adapters: dict | None = None) -> tuple:
    """(adapter, model, info, model name) for a request. The student comes from loader(run_dir, step, device)
    (mirror.final.heal.students.load_student); when it returns no adapter, the original model's adapter is built from
    the model name (cached in `adapters`)."""
    adapters = {} if adapters is None else adapters
    build_original = resolve_callable(build_original) or default_build_original

    def original_adapter(name):
        if not name:
            raise ValueError("the model name is unknown (no adapter from the loader, no model in spec.json or the "
                             "request)")
        if name not in adapters:
            adapters[name] = build_original(name, device)
        return adapters[name]

    if r["original"]:
        ad = original_adapter(r["model"])
        return ad, ad.model, {"original": True, "model": r["model"]}, r["model"]
    loader = resolve_callable(loader or DEFAULT_LOADER)
    akey = adapter_key(r["run_dir"])
    kw = {"adapter": adapters[akey]} if akey is not None and akey in adapters and _accepts(loader, "adapter") else {}
    ad, model, info = loader(r["run_dir"], r["step"], device, **kw)   # load_student reuses a given original adapter
    info = info if isinstance(info, dict) else {}
    name = model_name_of(info, r["run_dir"], r)
    if ad is None:
        ad = original_adapter(name)
    elif akey is not None:
        adapters.setdefault(akey, ad)
    if next(ad.model.parameters()).device != torch.device(device):
        ad.model.to(device)                                  # the original runs for the reference passes
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return ad, model, info, name


def evaluate_request(ev: KnnEvaluator, req: dict, loader=None, build_original=None, adapters: dict | None = None,
                     log=None, should_stop=None) -> dict:
    """Run the passes of one request that are not done yet and write <out>/eval/<tag>.json, the prediction files and
    <out>/eval/<tag>_meta.json. Returns the record. `should_stop()` is checked between passes (a stop leaves the
    finished passes on disk)."""
    from mirror.final.prereg import write_final_meta
    from mirror.utils import git_info

    log = log or ev.log
    t_start = time.time()
    r = normalise_request(req)
    run = None if r["original"] else Path(r["run_dir"])
    out = Path(r["out_dir"]) if r.get("out_dir") else run
    tag = (f"step{r['step']}" if r["step"] is not None else ("original" if r["original"] else "final"))
    path = out / "eval" / f"{tag}.json"
    rec = read_json(path) or {}
    ck = Path(r["ckpt"]) if r.get("ckpt") and Path(r["ckpt"]).is_file() else checkpoint_path(run, r["step"])
    ck_sha = sha256_file(ck) if ck is not None and ck.is_file() else None
    experiment = r.get("experiment") or (r.get("tags") or {}).get("experiment") or "fp_eval"

    def counts_of(p):
        fp32 = p.startswith("fp32")
        nq = r["fp32_n_query"] if fp32 and r["fp32_n_query"] is not None else r["n_query"]
        bpc = r["fp32_bank_per_class"] if fp32 else None
        return nq, bpc, ev.counts(p, nq, bpc)[2]

    plan = []
    for p in r["passes"]:
        nq, bpc, key = counts_of(p)
        plan.append((p, nq, bpc, key, ev.settings(p, nq, bpc, ck_sha)))
    rec.setdefault("passes", {})
    todo = [x for x in plan if not _done(rec, out, x[3], x[4])]
    orig_todo = [] if r["original"] else [(p, *counts_of(p)) for p in r["with_original"]
                                          if counts_of(p)[2] not in (rec.get("original_reference") or {})]
    loaded = None
    cost = rec.get("cost")
    if todo or orig_todo or (r["cost"] and cost is None):
        loaded = load_for_request(r, ev.device, loader, build_original, adapters)
    if loaded is not None:
        adapter, model, info, name = loaded
        model_key = f"{name or 'model'}-{model_fingerprint(adapter.model)}"
        rec.update({"model": name, "model_key": model_key,
                    "loader_info": {k: info.get(k) for k in ("method", "seed", "family", "variant", "step")
                                    if isinstance(info, dict) and k in info}})
        if r["cost"] and cost is None:
            from .flipavg import analytic_cost

            cost = analytic_cost(adapter, model, info=info, run_dir=run)
            rec["cost"] = cost
    rec.update({"format": PROTOCOL, "tag": tag, "run_dir": None if run is None else str(run), "step": r["step"],
                "original": bool(r["original"]), "checkpoint": None if ck is None else str(ck),
                "checkpoint_sha256": ck_sha, "data": ev.data.source})
    for p, nq, bpc, key, settings in todo:
        if should_stop is not None and should_stop():
            log(f"stop requested: {path} keeps {sorted(rec['passes'])}")
            break
        res = ev.run(adapter, model, p, nq, bpc)
        pp = save_predictions(out / "knn" / f"{tag}_{key}.npz", res["pred"], res["pred_flip"], res["labels"],
                              res["query_rows"])
        s = dict(res["summary"])
        if cost and cost.get("gflops_per_forward") is not None:
            s["gflops_per_image"] = s["forward_passes_per_image"] * float(cost["gflops_per_forward"])
        rec["passes"][key] = {**s, "settings": settings, "predictions": pp.relative_to(out).as_posix(),
                              "disagreements": res["disagreements"], "git_hash": git_info().get("git_hash"),
                              "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        write_json_atomic(path, rec)
        log(f"{out.name} {tag} [{key}]: k-NN top-1 {s['top1']:.2f}, flip consistency {s['flip_consistency']:.3f}% "
            f"({s['n_query']} queries, {s['n_bank']} bank images, {s['seconds']:.0f} s)")
        del res
    if loaded is not None and orig_todo:
        rec.setdefault("original_reference", {})
        for p, nq, bpc, key in orig_todo:
            if should_stop is not None and should_stop():
                break
            rec["original_reference"][key] = ev.original(adapter, model_key, p, nq, bpc)
    rec["request"] = {k: v for k, v in r.items()}
    write_json_atomic(path, rec)
    if loaded is not None or todo:
        write_final_meta(out / "eval", name=f"{tag}_meta.json", experiment=experiment,
                         config={"request": r, "batch_size": ev.batch_size, "k": ev.k, "temperature": ev.temperature,
                                 "knn_chunk": ev.chunk, "protocol": PROTOCOL, "data": ev.data.source},
                         seeds={"note": "the k-NN evaluation has no random parts"},
                         models={"model": rec.get("model"), "model_key": rec.get("model_key"),
                                 "run_dir": rec.get("run_dir"), "step": r["step"], "checkpoint_sha256": ck_sha},
                         device=ev.device, wall_seconds=time.time() - t_start,
                         extra={"passes_run": [x[3] for x in todo], "eval_json": str(path)})
    if loaded is not None:
        del loaded
    return rec


def _done(rec: dict, out: Path, key: str, settings: dict) -> bool:
    p = (rec.get("passes") or {}).get(key)
    return bool(p) and p.get("settings") == jsonable(settings) and (out / str(p.get("predictions", ""))).is_file()


def free_memory(device) -> None:
    import gc

    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def auto_gpu_fraction(data_bytes: int, reserve_gb: float = 12.0, device=0) -> float:
    """Per-process GPU memory cap that holds the data plus `reserve_gb` (models, features, activations)."""
    total = torch.cuda.get_device_properties(device).total_memory
    return float(min(0.95, max(0.05, (data_bytes + reserve_gb * 2 ** 30) / total)))


def cache_bytes(cache_dir) -> int:
    """Bytes the bank and the validation images take on the device (read from the memmap headers)."""
    cache = Path(cache_dir)
    with np.load(cache / "train_index.npz") as z:
        n_bank = int(len(z["bank_rows"]))
    val = np.load(cache / "val224_uint8.npy", mmap_mode="r")
    train = np.load(cache / "train224_uint8.npy", mmap_mode="r")
    per = int(np.prod(train.shape[1:]))
    return n_bank * per + int(np.prod(val.shape))


def make_evaluator(*, dry_run: bool = False, cache_dir="cache", device=None, batch_size: int = 512,
                   data_on: str = "device", gpu_frac: float | None = None, reserve_gb: float = 12.0, cache_root=None,
                   log=None) -> tuple:
    """(evaluator, device, GPU cap or None, pre-registration status) for scripts/fp_eval.py and
    scripts/fp_eval_server.py. On a GPU the memory cap is set first (default: the images plus `reserve_gb` over the
    device's memory), then mirror.final.prereg.require_prereg() runs, then the images are copied once. A dry run uses
    the CPU, synthetic images and no disk cache, and records the pre-registration status without requiring it."""
    from mirror.final.prereg import require_prereg

    log = log or _log
    if dry_run:
        st = require_prereg(allow_missing=True)
        data = EvalData.synthetic(num_classes=4, bank_per_class=8, val_per_class=4, img_size=56, seed=0)
        return KnnEvaluator(data, "cpu", batch_size, log=log), torch.device("cpu"), None, st
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    frac = None
    if device.type == "cuda":
        frac = gpu_frac if gpu_frac is not None else auto_gpu_fraction(
            cache_bytes(cache_dir) if data_on == "device" else 0, reserve_gb, device)
        from ...utils import cuda_index
        torch.cuda.set_per_process_memory_fraction(float(frac), cuda_index(device))
    st = require_prereg()
    on = device if data_on == "device" else torch.device("cpu")
    data = EvalData.from_cache(cache_dir, on, pin=data_on == "host", log=log)
    return KnnEvaluator(data, device, batch_size, cache_dir=cache_root, log=log), device, frac, st


# ----------------------------------------------------------------------------------------------
# Tiny models for --dry-run and tests
# ----------------------------------------------------------------------------------------------
def tiny_original(model_name: str = "tiny", device="cpu"):
    """The tiny random timm ViT of the tests (mirror.adapters.TimmTinyAdapter, 56 px, float32)."""
    from mirror.adapters import TimmTinyAdapter

    ad = TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float32)
    return ad.to(device=torch.device(device))


def tiny_loader(run_dir, step, device="cpu"):
    """Dry-run stand-in for mirror.final.heal.students.load_student: the tiny original with its weights moved by a
    small seeded amount (seed = step), so different steps give different models."""
    import copy

    ad = tiny_original("tiny", device)
    model = copy.deepcopy(ad.model)
    g = torch.Generator().manual_seed(1000 + int(step or 0))
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.02 * p.detach().std().nan_to_num(0.0) * torch.randn(p.shape, generator=g).to(p.device))
    return ad, model, {"model": "tiny", "method": "dry_run", "step": step}
