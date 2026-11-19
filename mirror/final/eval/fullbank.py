"""Full-train features and the full-bank k-NN (A10, FINAL_PHASE_PLAN.md v2: "Standard metrics, comparable with
published numbers").

1. stream_features: one pass over all ImageNet-train images (1,281,167; the image stream of the infra agent's
   mirror.final.data.cache, get_role("stream_train").batches, coded here against any iterator of
   (uint8 [B, 224, 224, 3], labels, ids); ids may be integers with several columns, e.g. (shard, row), or strings)
   computes, for SEVERAL models on each batch, (a) the k-NN feature (the CLS token after the final norm,
   adapter.cls_feature; for Mirror models the even part) and (b) the DINOv2 linear-protocol feature
   (linear_protocol.linear_feature: class tokens of the last 4 blocks and the mean patch token of the last block, each
   after the final norm, 5 d numbers), from ONE forward pass per model and batch (the k-NN feature is the last class
   token of (b), the same operations as cls_feature; adapters whose cls_feature differs get their own pass). bf16
   autocast as the k-NN protocol. Stored as float16 .npy memmaps per model (<out>/<name>/train_<kind>.npy), with
   <out>/train_labels.npy and <out>/train_ids.npy (int64 [N] or [N, c], or fixed-width strings) shared by all
   models. Progress is
   saved per model every few batches (<out>/progress.json), so an interrupted stream resumes at the last saved image.
   val_features does the same for the 50k validation images (plus the k-NN feature of every mirrored image, for flip
   consistency).
2. fullbank_knn: k-NN of the 50k validation queries (and of their mirrors) against all 1.28M train features: cosine
   (features L2-normalised in float32), k = 20, T = 0.07, votes exp(sim / T), argmax; the bank is read in chunks and
   each query keeps a running top-20 that is merged exactly with each chunk's top-20 (topk of the union), so memory is
   bounded by one chunk and the result is the top-20 of all similarities (ties aside), sorted, with the votes formed as
   mirror.knn.knn_predict forms them. Flip consistency and the vote-margin record as knn_fast.
3. The linear protocol on the stored features: linear_protocol.run_linear_protocol (fp_fullbank.py).

Flip-averaged models (A07) average both features over the image and its mirror (flipavg.py); their mirrored
validation feature equals the plain one by construction and is copied.

Cost (estimates, verify on the first run): per model one forward of every train image (1.28M) and two of every
validation image (100k), so for k models about 1.38M k forward passes; storage per ViT-B model: 1.28M x 768 x 2 B =
1.97 GB (k-NN) and 1.28M x 3,840 x 2 B = 9.84 GB (linear); the k-NN reads the bank once per model.

Sources: this repository (mirror/knn.py, knn_fast.py, linear_protocol.py; docs/final_phase_design.md) and the DINOv2
code cited in linear_protocol.py.
"""
from __future__ import annotations

import inspect
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mirror.data import mirror, normalize_for

from .knn_fast import (K, TEMPERATURE, autocast_for, flip_summary, jsonable, read_json, save_predictions,
                       strict_fp32, vote_margin_rel, write_json_atomic)
from .linear_protocol import N_LAST_STORED, linear_feature

KINDS = ("knn", "linear")
FP16_MAX = float(np.finfo(np.float16).max)


def standard_token_path(adapter) -> bool:
    """True when adapter.cls_feature is norm(forward_tokens(x))[:, 0] over the standard token path, so the k-NN
    feature can be read from the linear-protocol pass (mirror.adapters.ViTAdapter and its DINOv2 and tiny subclasses;
    mirror.adapters_timm.TimmPretrainedAdapter with feature "cls")."""
    from mirror.adapters import ViTAdapter

    cls = type(adapter)
    if getattr(adapter, "feature", "cls") == "pooled":
        return False
    fwd_ok = cls.forward_tokens is ViTAdapter.forward_tokens
    cls_ok = cls.cls_feature is ViTAdapter.cls_feature or cls.__name__ == "TimmPretrainedAdapter"
    return bool(fwd_ok and cls_ok)


class FeatureModel:
    """One model of the streaming pass: name, adapter, model, the feature kinds to store and flip averaging."""

    def __init__(self, name: str, adapter, model=None, kinds=KINDS, flipavg: bool = False, amp: str | None = "bf16",
                 info: dict | None = None):
        self.name, self.adapter = str(name), adapter
        self.model = adapter.model if model is None else model
        self.kinds = tuple(k for k in KINDS if k in tuple(kinds))
        if not self.kinds:
            raise ValueError(f"{name}: no feature kind among {KINDS}")
        self.flipavg, self.amp, self.info = bool(flipavg), amp, dict(info or {})
        self.d = int(adapter.embed_dim)
        self.token_path = standard_token_path(adapter)
        if "linear" in self.kinds and not self.token_path:
            raise ValueError(f"{name}: the linear protocol needs class tokens on the standard token path")

    def dim(self, kind: str) -> int:
        return self.d if kind == "knn" else (N_LAST_STORED + 1) * self.d

    def _one(self, x: torch.Tensor, kinds) -> dict:
        out = {}
        if "linear" in kinds or self.token_path:
            lin = linear_feature(self.adapter, x, self.model)
            if "linear" in kinds:
                out["linear"] = lin
            if "knn" in kinds:
                out["knn"] = lin[:, (N_LAST_STORED - 1) * self.d:N_LAST_STORED * self.d]
        if "knn" in kinds and "knn" not in out:
            out["knn"] = self.adapter.cls_feature(x, self.model).float()
        return out

    @torch.no_grad()
    def features(self, x: torch.Tensor, kinds=None) -> dict:
        """float32 features of normalised images x under the model's autocast; flip-averaged if asked."""
        kinds = self.kinds if kinds is None else tuple(kinds)
        with autocast_for(self.amp, x.device):
            a = self._one(x, kinds)
            if self.flipavg:
                b = self._one(mirror(x), kinds)
                a = {k: 0.5 * (a[k].float() + b[k].float()) for k in a}
        return {k: v.float() for k, v in a.items()}

    def describe(self) -> dict:
        return {"name": self.name, "kinds": list(self.kinds), "flip_average": self.flipavg, "amp": self.amp or "fp32",
                "d": self.d, "linear_dim": self.dim("linear"), "knn_from_linear_pass": self.token_path,
                "info": {k: v for k, v in self.info.items() if isinstance(v, (str, int, float, bool)) or v is None}}


def to_nchw_uint8(images, device) -> torch.Tensor:
    """A stream batch (numpy or torch, uint8 [B, H, W, 3] or [B, 3, H, W]) as a contiguous uint8 [B, 3, H, W] tensor
    on `device`."""
    t = images if isinstance(images, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(images))
    if t.dtype != torch.uint8 or t.ndim != 4:
        raise ValueError(f"stream images must be uint8 [B, H, W, 3], got {t.dtype} {tuple(t.shape)}")
    t = t.to(device, non_blocking=True)
    if t.shape[-1] == 3 and t.shape[1] != 3:
        t = t.permute(0, 3, 1, 2)
    return t.contiguous()


def _open(path: Path, shape: tuple, dtype) -> np.memmap:
    """Open an existing .npy memmap of this shape and dtype for update, or create it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        mm = np.load(path, mmap_mode="r+")
        if mm.shape == tuple(shape) and mm.dtype == np.dtype(dtype):
            return mm
        del mm
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=tuple(shape))


def _np(a) -> np.ndarray:
    """Labels or ids from a stream batch (list, numpy, or a torch tensor on any device) as a numpy array."""
    return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)


def _f16(t: torch.Tensor, what: str) -> np.ndarray:
    if not torch.isfinite(t).all():
        raise FloatingPointError(f"{what}: non-finite features")
    if float(t.abs().max()) > FP16_MAX:
        raise OverflowError(f"{what}: feature values exceed the float16 range")
    return t.to(torch.float16).cpu().numpy()


def skip_images(it, start: int):
    """Drop the first `start` images of an iterator of (images, labels, ids) batches (for streams without a start
    argument); the batch that straddles `start` is cut."""
    seen = 0
    for images, labels, ids in it:
        b = len(labels)
        if seen + b <= start:
            seen += b
            continue
        lo = max(start - seen, 0)
        seen += b
        yield images[lo:], labels[lo:], ids[lo:]


def make_stream_factory(stream_fn, batch_size: int | None = None):
    """factory(start) -> iterator for a stream function: passes start= and batch_size= when the function takes
    them, else skips images itself."""
    params = inspect.signature(stream_fn).parameters

    def factory(start: int = 0):
        kw = {}
        if batch_size is not None and "batch_size" in params:
            kw["batch_size"] = int(batch_size)
        if "start" in params:
            return iter(stream_fn(start=int(start), **kw))
        return skip_images(iter(stream_fn(**kw)), int(start))

    return factory


def stream_features(models: list, stream_factory, n_total: int, out_dir, device, *, micro_batch: int = 256,
                    flush_every: int = 20, log=None) -> dict:
    """Features of every streamed train image for every model (module docstring). Resumes from <out>/progress.json.
    Returns the progress record."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    say = log or (lambda m: None)
    prog_path = out / "progress.json"
    prog = read_json(prog_path) or {}
    if prog.get("n_train") not in (None, int(n_total)):
        raise ValueError(f"{prog_path} was written for {prog['n_train']} train images, not {n_total}")
    prog["n_train"] = int(n_total)
    done = prog.setdefault("train", {})
    stores = {}
    for fm in models:
        for kind in fm.kinds:
            stores[(fm.name, kind)] = _open(out / fm.name / f"train_{kind}.npy", (int(n_total), fm.dim(kind)),
                                            np.float16)
        done.setdefault(fm.name, 0)
    labels_mm = _open(out / "train_labels.npy", (int(n_total),), np.int64)
    ids_mm = np.load(out / "train_ids.npy", mmap_mode="r+") if (out / "train_ids.npy").exists() else None
    done.setdefault("_labels_ids", 0)
    start = min([done[fm.name] for fm in models] + [done["_labels_ids"]])
    if start >= n_total:
        say(f"train features: every model is done ({n_total:,} images)")
        return prog
    say(f"train features: {len(models)} model(s), resuming at image {start:,} of {n_total:,}")
    pos, t0, n_batches = start, time.time(), 0

    def save_progress():
        for mm in stores.values():
            mm.flush()
        labels_mm.flush()
        if ids_mm is not None:
            ids_mm.flush()
        write_json_atomic(prog_path, prog)

    for images, labels, ids in stream_factory(start):
        b = len(labels)
        if b == 0:
            continue
        if pos + b > n_total:
            raise ValueError(f"the stream gave more than {n_total} images")
        x8 = None
        for fm in models:
            lo = max(done[fm.name] - pos, 0)
            if lo >= b:
                continue
            if x8 is None:
                x8 = to_nchw_uint8(images, device)
            for s in range(lo, b, int(micro_batch)):
                e = min(s + int(micro_batch), b)
                feats = fm.features(normalize_for(fm.adapter, x8[s:e]))
                for kind in fm.kinds:
                    stores[(fm.name, kind)][pos + s:pos + e] = _f16(feats[kind], f"{fm.name} {kind}")
            done[fm.name] = pos + b
        lo = max(done["_labels_ids"] - pos, 0)
        if lo < b:
            labels_mm[pos + lo:pos + b] = _np(labels).astype(np.int64)[lo:]
            ids_np = _np(ids)
            if ids_mm is None:                                   # ints (any number of columns) or strings
                width = max(64, int(ids_np.dtype.itemsize // 4))
                dt = np.int64 if ids_np.dtype.kind in "iu" else np.dtype(f"<U{width}")
                ids_mm = _open(out / "train_ids.npy", (int(n_total),) + tuple(ids_np.shape[1:]), dt)
            if ids_mm.dtype.kind == "U" and ids_np.dtype.kind == "U" and ids_np.dtype.itemsize > ids_mm.dtype.itemsize:
                raise ValueError("an image id is longer than the id store's width")
            ids_mm[pos + lo:pos + b] = ids_np[lo:]
            done["_labels_ids"] = pos + b
        pos += b
        n_batches += 1
        if n_batches % int(flush_every) == 0:
            save_progress()
            rate = (pos - start) / max(time.time() - t0, 1e-9)
            say(f"  train features: {pos:,}/{n_total:,} images, {rate:.0f} img/s per pass over {len(models)} model(s)")
        if pos >= n_total:
            break
    save_progress()
    if pos < n_total:
        raise RuntimeError(f"the stream ended at image {pos:,} of {n_total:,}; rerun to resume")
    say(f"train features: done, {n_total - start:,} images in {time.time() - t0:.0f} s")
    return prog


def val_features(models: list, images, out_dir, device, *, batch_size: int = 256, flip_knn: bool = True,
                 log=None) -> dict:
    """Validation features (`images`: a knn_fast.ImageSet of the 50k validation images, labels included) for every
    model, and the k-NN feature of every mirrored image. Resumes per model (progress.json "val")."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    say = log or (lambda m: None)
    prog_path = out / "progress.json"
    prog = read_json(prog_path) or {}
    vdone = prog.setdefault("val", {})
    n = len(images)
    np.save(out / "val_labels.npy", images.labels.numpy().astype(np.int64))
    for fm in models:
        if vdone.get(fm.name) == n:
            continue
        t0 = time.time()
        stores = {kind: _open(out / fm.name / f"val_{kind}.npy", (n, fm.dim(kind)), np.float16) for kind in fm.kinds}
        flip = None
        if flip_knn and "knn" in fm.kinds:
            flip = _open(out / fm.name / "val_knn_flip.npy", (n, fm.d), np.float16)
        pos = 0
        for x8 in images.batches(None, batch_size, device):
            x = normalize_for(fm.adapter, x8)
            feats = fm.features(x)
            e = pos + x.shape[0]
            for kind in fm.kinds:
                stores[kind][pos:e] = _f16(feats[kind], f"{fm.name} val {kind}")
            if flip is not None:
                fk = feats["knn"] if fm.flipavg else fm.features(mirror(x), kinds=("knn",))["knn"]
                flip[pos:e] = _f16(fk, f"{fm.name} val mirrored knn")
            pos = e
        for mm in list(stores.values()) + ([flip] if flip is not None else []):
            mm.flush()
        vdone[fm.name] = n
        write_json_atomic(prog_path, prog)
        say(f"validation features {fm.name}: {n:,} images ({time.time() - t0:.0f} s)")
    return prog


# ----------------------------------------------------------------------------------------------
# k-NN against a large bank, chunked, with an exact top-k merge
# ----------------------------------------------------------------------------------------------
def _unit(a, device) -> torch.Tensor:
    t = a if isinstance(a, torch.Tensor) else torch.from_numpy(np.array(a, copy=True))     # memmaps are read-only
    return F.normalize(t.to(device).float(), dim=-1)


@torch.no_grad()
def topk_chunked(bank, queries: dict, k: int = K, device="cpu", bank_chunk: int = 131072, query_chunk: int = 2048,
                 log=None) -> dict:
    """{name: (similarities [Q, k], bank rows [Q, k])}, sorted in decreasing similarity, of the unit query features
    (`queries`: name -> unit float32 tensor on `device`) against `bank` (an [N, d] array or memmap, any float dtype;
    every chunk is L2-normalised in float32). Each chunk's top-k is merged with the running top-k by a topk of their
    union, so the result is the top-k of all N similarities."""
    device = torch.device(device)
    N = int(bank.shape[0])
    k = min(int(k), N)
    state = {name: (torch.full((q.shape[0], k), float("-inf"), device=device),
                    torch.full((q.shape[0], k), -1, dtype=torch.long, device=device)) for name, q in queries.items()}
    t0 = time.time()
    for s in range(0, N, int(bank_chunk)):
        b = _unit(bank[s:s + int(bank_chunk)], device)
        kk = min(k, b.shape[0])
        for name, q in queries.items():
            rv, ri = state[name]
            for qs in range(0, q.shape[0], int(query_chunk)):
                qe = min(qs + int(query_chunk), q.shape[0])
                v, i = (q[qs:qe] @ b.T).topk(kk, dim=1)
                cv = torch.cat([rv[qs:qe], v], dim=1)
                ci = torch.cat([ri[qs:qe], i + s], dim=1)
                tv, tp = cv.topk(k, dim=1)
                rv[qs:qe] = tv
                ri[qs:qe] = ci.gather(1, tp)
        if log is not None and (s // int(bank_chunk)) % 4 == 0:
            log(f"  full-bank k-NN: bank rows {min(s + int(bank_chunk), N):,}/{N:,} ({time.time() - t0:.0f} s)")
    return state


@torch.no_grad()
def vote(top_sim: torch.Tensor, top_idx: torch.Tensor, bank_labels: torch.Tensor, num_classes: int,
         temperature: float = TEMPERATURE, chunk: int = 4096) -> tuple:
    """(predictions, relative top-two vote margins) from sorted top-k similarities, formed as
    mirror.knn.knn_predict forms them (weights exp(sim / T), scatter_add of the weights to the neighbours' labels)."""
    preds, margins = [], []
    for s in range(0, top_sim.shape[0], int(chunk)):
        ts, ti = top_sim[s:s + int(chunk)], top_idx[s:s + int(chunk)]
        weights = (ts / temperature).exp()
        votes = torch.zeros(ts.shape[0], num_classes, device=ts.device, dtype=weights.dtype)
        votes.scatter_add_(1, bank_labels[ti], weights)
        preds.append(votes.argmax(dim=1))
        margins.append(vote_margin_rel(votes))
    return torch.cat(preds), torch.cat(margins)


def fullbank_knn(out_dir, name: str, num_classes: int, device="cpu", k: int = K, temperature: float = TEMPERATURE,
                 bank_chunk: int = 131072, query_chunk: int = 2048, log=None) -> dict:
    """k-NN of the stored validation features (and their mirrors) of model `name` against its stored train features;
    writes <out>/<name>/fullbank_knn.json and fullbank_knn_pred.npz (MUST-09 format). Skips a finished model."""
    out = Path(out_dir)
    mdir = out / name
    res_path = mdir / "fullbank_knn.json"
    bank = np.load(mdir / "train_knn.npy", mmap_mode="r")
    settings = {"protocol": "fp-fullbank-knn-v1", "k": int(k), "temperature": float(temperature),
                "n_bank": int(bank.shape[0]), "train_knn_bytes": int(os.path.getsize(mdir / "train_knn.npy"))}
    prev = read_json(res_path)
    if prev and prev.get("settings") == jsonable(settings) and (mdir / "fullbank_knn_pred.npz").is_file():
        return prev
    device = torch.device(device)
    t0 = time.time()
    labels = torch.from_numpy(np.load(out / "train_labels.npy")).to(device)
    yv = torch.from_numpy(np.load(out / "val_labels.npy"))
    with strict_fp32():
        q = {"x": _unit(np.load(mdir / "val_knn.npy", mmap_mode="r"), device)}
        has_flip = (mdir / "val_knn_flip.npy").is_file()
        if has_flip:
            q["mirror"] = _unit(np.load(mdir / "val_knn_flip.npy", mmap_mode="r"), device)
        top = topk_chunked(bank, q, k, device, bank_chunk, query_chunk, log)
        pred, m_x = vote(*top["x"], labels, num_classes, temperature)
        if has_flip:
            pred_f, m_m = vote(*top["mirror"], labels, num_classes, temperature)
            diff = (q["mirror"] - q["x"]).abs().amax(dim=1).cpu()
        else:
            pred_f, m_m, diff = pred, m_x, torch.zeros(pred.shape[0])
    summary = flip_summary(pred.cpu(), pred_f.cpu(), yv, m_x.cpu(), m_m.cpu(), diff)
    if not has_flip:
        for key in ("flip_consistency", "n_disagree", "max_vote_margin_rel", "max_vote_margin_rel_one_copy",
                    "n_disagree_not_near_tie", "feature_flip_max_abs", "feature_identical_pct",
                    "top1_mirrored_queries"):
            summary[key] = None
    pp = save_predictions(mdir / "fullbank_knn_pred.npz", pred.cpu(), pred_f.cpu(), yv)
    rec = {"model": name, **summary, "n_bank": int(bank.shape[0]), "n_query": int(pred.shape[0]), "k": int(k),
           "temperature": float(temperature), "settings": settings, "predictions": pp.name,
           "features": "stored float16 features (bf16 autocast), L2-normalised in float32; cosine; k-NN in fp32, "
                       "TF32 off", "seconds": round(time.time() - t0, 1)}
    write_json_atomic(res_path, rec)
    if log is not None:
        log(f"full-bank k-NN {name}: top-1 {rec['top1']:.2f} against {rec['n_bank']:,} train images"
            + (f", flip consistency {rec['flip_consistency']:.3f}%" if has_flip else ""))
    return rec
