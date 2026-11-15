#!/usr/bin/env python
"""CORE-07: distribution shift (PROJECT_NEXT_PHASE.md CORE-07, PREREG_NEXT_PHASE.md P7.1).

    python scripts/core07_shift.py --model dinov2_vitb14_reg --must09-dir runs/MUST-09/dinov2_vitb14_reg \
        --mirror-seeds 0,1,2 --gate-b results/MUST-07/gate_b.json --probe-from results/MUST-11/t2
    python scripts/core07_shift.py --tiny --out tiny_out/core07              # synthetic folders, tiny ViT, CPU

What it measures: the ImageNet linear probe (scripts/tier2_linear.py: logistic regression on [CLS, mean patch token]
after the final norm, trained on the 100k bank, weight decay chosen on a split of the bank) applied unchanged to
ImageNet-V2 matched frequency and, if present, ImageNet-Sketch. Models: the original, healed Mirror-exact
(runs/MUST-09/<model>/mirror_exact/s<seed>, variant names vexact_s<seed> as in MUST-11) and the healed gate-B best
baseline (its method read from results/MUST-07/gate_b.json, chosen there on the method-selection set; variant name
best_<method>_s<seed>). Each model's drop is ImageNet validation top-1 minus shifted top-1, with the same probe.

Probe and features (scripts/tier2_linear.py by import, nothing re-implemented):
  - If --probe-from (default results/MUST-11/t2) holds tier2_linear's probe of the same variant name, whose recorded
    feature sources (weights sha256) equal this run's, that probe (W, b, standardisation) is used as it is, with its
    own feature recipe, precision and columns (the variant's own row, view a: for Mirror-exact the exactly
    flip-invariant [even CLS, mean of even-projected patch tokens]). Its cached validation features are reused when
    they match; otherwise they are recomputed.
  - Otherwise the probe is trained here exactly as tier2_linear trains it (extract_features on the bank and the
    validation images, run_probe) and stored under <out>/probe/ in tier2_linear's layout.
  - Shifted-set features use the probe's recipe, autocast (bf16 by default, as tier2_linear) and storage dtype.
Images: resize 256 (bicubic), centre crop 224 (as every ImageNet cache), decoded once into --shift-cache-dir.

Label mapping (configs/data/shift_sets.yaml):
  ImageNet-V2: folder names are the integer ImageNet-1k class index (label = int(folder name), as the official
  loader); a torchvision ImageFolder would sort the names as strings, which is wrong. Expected 1000 folders x 10.
  ImageNet-Sketch: zip of WNID folders; label = rank of the WNID in the sorted ImageNet-1k list
  (mirror.controlled.imagenet1k_wnids, timm's list; the Hugging Face ImageNet label order). Only members ending in
  ".JPEG" are used, as the dataset's own loader does. The Sketch paper pads classes with fewer than 50 images with
  flipped and rotated copies of other images in the class: an exactly mirror-invariant model gives identical outputs
  on a flipped pair, so those are not independent test points. The script counts near-duplicate pairs under the 8
  flips and rotations of a 32 x 32 grey thumbnail (a heuristic, threshold --dup-thr on the mean absolute difference)
  and reports Sketch top-1 on all images (primary) and with one image of each detected pair removed.

P7.1 (operational, this script's choices, listed in the report): drop(model) = top-1 on ImageNet validation (all
50,000) minus top-1 on ImageNet-V2 matched frequency (all 10,000), the same probe for both. Mirror-exact's drop is the
mean over the healed seeds given (3 for the headline). Pass: |drop(Mirror-exact) - drop(original)| <= 1.0 point. A
bootstrap interval of the difference (validation and V2 images resampled independently, paired across models,
10,000 resamples, seed 0) is reported, not tested. Sketch and the best baseline are reported, not tested.

Outputs (resumable): <out>/features/<model>/<variant>_<set>.npz, <out>/preds/<model>/<variant>_<set>.npz,
<out>/probe/<model>/... (probes trained here), <out>/shift_rows.csv, <out>/metrics.json, <out>/SUMMARY.md,
<out>/meta.json.
"""
from __future__ import annotations

import sys

import torch


def _early_gpu_frac(argv, default: float = 0.2) -> float:
    for i, s in enumerate(argv):
        if s == "--gpu-frac" and i + 1 < len(argv):
            return float(argv[i + 1])
        if s.startswith("--gpu-frac="):
            return float(s.split("=", 1)[1])
    return default


if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(_early_gpu_frac(sys.argv[1:]))  # shared GPU: cap before allocating

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.cub import decode_to_cache, file_sha256  # noqa: E402
from mirror.data import make_loader  # noqa: E402
from mirror.meta import write_meta  # noqa: E402
from mirror.utils import REPO_ROOT, git_info, log, set_seed  # noqa: E402

EXPERIMENT = "CORE-07"
P71_MAX_GAP = 1.0
V2_FOLDER = "imagenetv2-matched-frequency-format-val"
IMAGE_SUFFIXES = (".jpeg", ".jpg", ".png")


def load_tier2():
    key = "_core07_tier2_linear"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, REPO_ROOT / "scripts" / "tier2_linear.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


T2 = load_tier2()


def parse_args(argv=None):
    hf = os.environ.get("HF_HOME", "/home/hf")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2_vitb14_reg")
    p.add_argument("--variants", default="original", help="tier2_linear variant names (comma list; may be empty)")
    p.add_argument("--healed-run", action="append", default=None, metavar="[NAME=]DIR", help="as tier2_linear")
    p.add_argument("--must09-dir", default=None,
                   help="runs/MUST-09/<model>: adds vexact_s<seed> (mirror_exact) and best_<method>_s<seed> runs")
    p.add_argument("--mirror-seeds", default="0,1,2")
    p.add_argument("--gate-b", default="results/MUST-07/gate_b.json", help="gate-B record naming the best baseline")
    p.add_argument("--best-seeds", default="0")
    p.add_argument("--probe-from", default="results/MUST-11/t2", help="tier2_linear output folder to reuse probes from")
    p.add_argument("--no-probe-reuse", action="store_true")
    p.add_argument("--v2-root", default=os.path.join(hf, "extracted", V2_FOLDER))
    p.add_argument("--sketch-zip", default=os.path.join(hf, "extracted", "ImageNet-Sketch.zip"))
    p.add_argument("--no-sketch", action="store_true")
    p.add_argument("--shift-cache-dir", default=str(REPO_ROOT / "cache" / "shift"))
    p.add_argument("--cache-dir", default="cache", help="ImageNet caches (bank and validation)")
    p.add_argument("--out", default=None, help="default results/CORE-07 (tiny: tiny_out/core07)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--resize", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--decode-workers", type=int, default=16)
    p.add_argument("--amp", default="bf16", choices=["bf16", "none"], help="for probes trained here (as tier2)")
    p.add_argument("--feature-dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--wd-grid", default=None, help="probes trained here (default tier2_linear's)")
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--dup-thr", type=float, default=0.02,
                   help="Sketch near-duplicate threshold: mean |difference| of 32 x 32 grey thumbnails, 0..1")
    p.add_argument("--bootstrap-resamples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-frac", type=float, default=0.2, help="per-process GPU memory cap (read at import)")
    p.add_argument("--device", default=None)
    p.add_argument("--tiny", action="store_true")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------------------------
# Shifted test sets
# ----------------------------------------------------------------------------------------------
def v2_index(root) -> tuple:
    """([("file", path)], labels) for an ImageNet-V2 folder: label = int(folder name), never ImageFolder's string
    order. Files sorted by (label, name)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found (scripts/h100_data.sh stage v2)")
    items = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if not re.fullmatch(r"\d+", d.name):
            raise ValueError(f"{d}: ImageNet-V2 folders are named by the integer class index")
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES:
                items.append((int(d.name), f.name, f))
    items.sort(key=lambda t: (t[0], t[1]))
    return [("file", str(f)) for _, _, f in items], np.asarray([lab for lab, _, _ in items], dtype=np.int64)


def sketch_index(zip_path, wnids) -> tuple:
    """([("zip", zip, member)], labels): members ending in ".JPEG" (as the dataset's loader), label = index of the
    parent folder's WNID in `wnids` (sorted ImageNet-1k order)."""
    pos = {w: i for i, w in enumerate(wnids)}
    items, unknown = [], set()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".JPEG"):
                continue
            wnid = name.rstrip("/").split("/")[-2].lower()
            if wnid not in pos:
                unknown.add(wnid)
                continue
            items.append((pos[wnid], name))
    if unknown:
        raise ValueError(f"{len(unknown)} Sketch folders are not ImageNet-1k WNIDs, e.g. {sorted(unknown)[:3]}")
    items.sort()
    return [("zip", str(zip_path), n) for _, n in items], np.asarray([lab for lab, _ in items], dtype=np.int64)


def build_shift_cache(name: str, sources: list, labels: np.ndarray, cache_dir: Path, resize: int, crop: int,
                      workers: int, extra: dict) -> dict:
    """<cache_dir>/<name>{crop}_uint8.npy, _labels.npy and _meta.json; reused when the meta matches."""
    cache_dir = Path(cache_dir)
    npy = cache_dir / f"{name}{crop}_uint8.npy"
    meta_path = cache_dir / f"{name}_meta.json"
    listing = json.dumps([s[1:] for s in sources])
    import hashlib

    want = {"n_images": len(sources), "resize": int(resize), "crop": int(crop),
            "listing_sha256": hashlib.sha256(listing.encode()).hexdigest(),
            "labels_sha256": hashlib.sha256(np.asarray(labels, dtype=np.int64).tobytes()).hexdigest()}
    if meta_path.exists() and npy.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("complete") and all(meta.get(k) == v for k, v in want.items()):
            return meta
    t0 = time.time()
    decode_to_cache(sources, npy, resize, crop, workers)
    np.save(cache_dir / f"{name}_labels.npy", np.asarray(labels, dtype=np.int64))
    meta = {**want, **extra, "complete": True, "seconds": round(time.time() - t0, 1),
            "transform": f"resize {resize} (bicubic), centre crop {crop} (mirror.data.preprocess_pil)"}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_shift(name: str, cache_dir: Path, crop: int) -> tuple:
    cache_dir = Path(cache_dir)
    return (np.load(cache_dir / f"{name}{crop}_uint8.npy", mmap_mode="r"), np.load(cache_dir / f"{name}_labels.npy"),
            json.loads((cache_dir / f"{name}_meta.json").read_text(encoding="utf-8")))


DIHEDRAL = ("identity", "flip_lr", "flip_ud", "rot90", "rot180", "rot270", "transpose", "antitranspose")


def _dihedral(T, k: str):
    import torch as th

    return {"identity": T, "flip_lr": th.flip(T, [-1]), "flip_ud": th.flip(T, [-2]), "rot90": th.rot90(T, 1, [-2, -1]),
            "rot180": th.rot90(T, 2, [-2, -1]), "rot270": th.rot90(T, 3, [-2, -1]), "transpose": T.transpose(-1, -2),
            "antitranspose": th.rot90(T, 2, [-2, -1]).transpose(-1, -2)}[k]


@torch.no_grad()
def near_duplicates(images, labels, thr: float = 0.02, size: int = 32, batch: int = 1024) -> dict:
    """Pairs (i < j, same class) whose 32 x 32 grey thumbnails differ by less than `thr` (mean absolute difference,
    0..1) under one of the 8 flips and rotations. Returns the pairs with their transform, counts per transform, the
    counts at thresholds thr/2 and 2 thr, and a keep mask that drops the later image of each pair (a heuristic)."""
    import torch.nn.functional as F

    labels = np.asarray(labels)
    thumbs = np.empty((len(labels), size, size), dtype=np.float32)
    for s in range(0, len(labels), batch):
        x = torch.as_tensor(np.array(images[s:s + batch])).permute(0, 3, 1, 2).float().mean(1, keepdim=True) / 255.0
        thumbs[s:s + batch] = F.adaptive_avg_pool2d(x, size)[:, 0].numpy()
    pairs, keep = [], np.ones(len(labels), dtype=bool)
    dists_all = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        if len(idx) < 2:
            continue
        T = torch.as_tensor(thumbs[idx])
        flat = T.reshape(len(idx), -1)
        best = torch.full((len(idx), len(idx)), float("inf"))
        arg = torch.zeros((len(idx), len(idx)), dtype=torch.long)
        for k, name in enumerate(DIHEDRAL):
            D = torch.cdist(_dihedral(T, name).reshape(len(idx), -1), flat, p=1) / flat.shape[1]
            better = D < best
            best = torch.where(better, D, best)
            arg = torch.where(better, torch.full_like(arg, k), arg)
        iu = torch.triu_indices(len(idx), len(idx), 1)
        d = torch.minimum(best[iu[0], iu[1]], best[iu[1], iu[0]]).numpy()
        a_ = torch.where(best[iu[0], iu[1]] <= best[iu[1], iu[0]], arg[iu[0], iu[1]], arg[iu[1], iu[0]]).numpy()
        dists_all.append(d)
        for p, (i, j) in enumerate(zip(iu[0].tolist(), iu[1].tolist())):
            if d[p] < thr:
                pairs.append((int(idx[i]), int(idx[j]), DIHEDRAL[int(a_[p])], float(d[p])))
                keep[idx[j]] = False
    d_all = np.concatenate(dists_all) if dists_all else np.zeros(0)
    by = {n: sum(1 for p in pairs if p[2] == n) for n in DIHEDRAL}
    return {"thr": thr, "thumb": size, "n_pairs": len(pairs), "by_transform": by,
            "n_pairs_at_half_thr": int((d_all < thr / 2).sum()), "n_pairs_at_double_thr": int((d_all < 2 * thr).sum()),
            "n_dropped": int((~keep).sum()), "keep": keep, "pairs": pairs[:2000],
            "note": "heuristic: thumbnails of the 224 px crops; a centre crop of a flipped image differs from the flip "
                    "of the centre crop by at most one pixel column"}


# ----------------------------------------------------------------------------------------------
# Variants, probes and features
# ----------------------------------------------------------------------------------------------
def role_of(name: str) -> str:
    if name == "original":
        return "original"
    if name.startswith(("vexact", "mirror")):
        return "mirror_exact"
    if name.startswith("best"):
        return "best_baseline"
    return "other"


def auto_runs(a) -> list:
    """--must09-dir: vexact_s<seed> = <dir>/mirror_exact/s<seed>, best_<method>_s<seed> = <dir>/<method>/s<seed>
    (method from --gate-b); only folders that exist (missing ones are logged)."""
    if not a.must09_dir:
        return []
    base = T2._resolve(a.must09_dir)
    specs = []
    for s in [int(x) for x in str(a.mirror_seeds).split(",") if x.strip()]:
        d = base / "mirror_exact" / f"s{s}"
        if d.is_dir():
            specs.append(f"vexact_s{s}={d}")
        else:
            log(f"missing healed Mirror-exact run {d}; skipped")
    gb = T2._resolve(a.gate_b)
    if gb.exists():
        best = json.loads(gb.read_text(encoding="utf-8")).get("best")
        for s in [int(x) for x in str(a.best_seeds).split(",") if x.strip()]:
            d = base / str(best) / f"s{s}"
            if best and d.is_dir():
                specs.append(f"best_{best}_s{s}={d}")
            else:
                log(f"missing healed best-baseline run {d}; skipped")
    else:
        log(f"{gb} not found: no best baseline")
    return specs


def reusable_probe(probe_dir, model_name: str, vname: str, sources: dict):
    """tier2_linear's probe of `vname` in `probe_dir` if its recorded sources equal `sources`, else None."""
    if probe_dir is None:
        return None
    pj = Path(probe_dir) / "probe" / model_name / f"{vname}.json"
    pz = pj.with_name(f"{vname}_probe.npz")
    if not (pj.exists() and pz.exists()):
        return None
    j = json.loads(pj.read_text(encoding="utf-8"))
    if T2._jsonable(j.get("sources")) != T2._jsonable(sources):
        log(f"  {vname}: probe in {pj} was trained on other weights (sources differ); not reused")
        return None
    fm = (j.get("feature_meta") or {}).get("bank") or {}
    with np.load(pz) as z:
        W, b, mean, std = z["W"], z["b"], z["mean"], z["std"]
    return {"W": W, "b": b, "mean": mean, "std": std, "columns": j.get("feature_columns"),
            "recipe": fm.get("recipe", T2.FEATURE_RECIPE), "amp": fm.get("amp", "bf16"),
            "dtype": fm.get("dtype", "float16"), "img_size": fm.get("img_size"), "val_top1_recorded": j["row"]["top1"],
            "chosen_wd": j["row"]["chosen_wd"], "source": str(pj), "sha256": T2.sha256(pz),
            "features_dir": Path(probe_dir) / "features" / model_name}


def feature_fn_for(recipe: str):
    if recipe == T2.FEATURE_RECIPE_PARTS:
        return T2.probe_feature_parts
    if recipe == T2.FEATURE_RECIPE:
        return T2.probe_feature
    raise ValueError(f"unknown tier2_linear feature recipe {recipe!r}")


@torch.no_grad()
def predict(probe: dict, X: np.ndarray, device) -> np.ndarray:
    """Top-5 class indices [N, 5] of the probe on raw features X (columns selected first)."""
    cols = probe.get("columns")
    X = X if cols is None else X[:, np.asarray(cols)]
    W = torch.as_tensor(probe["W"], device=device, dtype=torch.float32)
    b = torch.as_tensor(probe["b"], device=device, dtype=torch.float32)
    mean = torch.as_tensor(probe["mean"], device=device, dtype=torch.float32)
    std = torch.as_tensor(probe["std"], device=device, dtype=torch.float32)
    out = []
    for s in range(0, X.shape[0], 16384):
        Xs = T2.standardise(torch.as_tensor(X[s:s + 16384], device=device), mean, std)
        out.append(torch.addmm(b, Xs, W).topk(min(5, W.shape[1]), dim=1).indices.cpu().numpy())
    return np.concatenate(out)


def topk_stats(top5: np.ndarray, y: np.ndarray, keep=None) -> dict:
    y = np.asarray(y)
    if keep is not None:
        top5, y = top5[keep], y[keep]
    return {"top1": 100.0 * float((top5[:, 0] == y).mean()), "top5": 100.0 * float((top5 == y[:, None]).any(1).mean()),
            "n": int(len(y))}


def drop_bootstrap(cv_a, cs_a, cv_b, cs_b, n_resamples: int = 10_000, seed: int = 0) -> dict:
    """Difference of drops, (val_a - shift_a) - (val_b - shift_b), in points, with a 95% percentile interval:
    validation and shifted images resampled independently, the same resample for both models (paired)."""
    dv = np.asarray(cv_a, dtype=np.float64) - np.asarray(cv_b, dtype=np.float64)
    ds = np.asarray(cs_a, dtype=np.float64) - np.asarray(cs_b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples)
    for s in range(0, n_resamples, 100):
        m = min(100, n_resamples - s)
        iv = rng.integers(0, len(dv), size=(m, len(dv)))
        is_ = rng.integers(0, len(ds), size=(m, len(ds)))
        stats[s:s + m] = dv[iv].mean(1) - ds[is_].mean(1)
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return {"diff": 100.0 * float(dv.mean() - ds.mean()), "ci_low": 100.0 * float(lo), "ci_high": 100.0 * float(hi),
            "n_val": int(len(dv)), "n_shift": int(len(ds)), "n_resamples": int(n_resamples), "seed": int(seed)}


# ----------------------------------------------------------------------------------------------
# Tiny synthetic data
# ----------------------------------------------------------------------------------------------
def write_tiny_shift(root: Path, n_classes: int = 4, per_class: int = 3, size: int = 64, seed: int = 0) -> dict:
    """A V2-shaped folder (integer folder names) and a Sketch-shaped zip (WNID folders, .JPEG members; one
    flipped copy per class) with random images."""
    from PIL import Image

    from mirror.controlled import imagenet1k_wnids

    rng = np.random.default_rng(seed)
    v2 = root / V2_FOLDER
    for c in range(n_classes):
        (v2 / str(c)).mkdir(parents=True, exist_ok=True)
        for k in range(per_class):
            img = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
            Image.fromarray(img).save(v2 / str(c) / f"{k:03d}.jpeg", quality=95)
    wnids = imagenet1k_wnids()[:n_classes]
    zp = root / "ImageNet-Sketch.zip"
    import io

    with zipfile.ZipFile(zp, "w") as zf:
        for c, w in enumerate(wnids):
            first = None
            for k in range(per_class):
                img = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
                if k == per_class - 1 and first is not None:
                    img = first[:, ::-1].copy()                       # a flipped copy, as the Sketch padding
                first = img if first is None else first
                buf = io.BytesIO()
                Image.fromarray(img).save(buf, format="PNG")         # PNG data under a .JPEG name, as some Sketch files
                zf.writestr(f"sketch/{w}/{w}_{k}.JPEG", buf.getvalue())
            zf.writestr(f"sketch/{w}/notes.txt", "ignored")
    return {"v2": v2, "sketch": zp}


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def main(argv=None) -> dict:
    t0 = time.time()
    a = parse_args(argv)
    tiny = a.tiny
    out = Path(a.out) if a.out else REPO_ROOT / ("tiny_out/core07" if tiny else "results/CORE-07")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() and not tiny else "cpu"))
    T2.apply_memory_cap(a.gpu_frac)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    set_seed(a.seed)
    git = git_info()
    model_name = "tiny" if tiny else a.model
    grid = [float(x) for x in (a.wd_grid or T2.DEFAULT_WD_GRID).split(",") if x.strip()]
    solver = {"max_iter": a.max_iter, "history_size": 20, "tol_grad": 1e-6}

    # ---- variants (tier2_linear's factory)
    paths = {"operators": None, "m1_structure": None, "heal_dir": None, "vexact_dir": None, "vexact_checkpoint": None,
             "vexact_config": None}
    adapter = T2.build_adapter(a.model, tiny, device, a.img_size)
    if tiny:
        from mirror import variants as V

        data = T2.TinyData(4, 20, 8, seed=a.seed, batch_size=16)
        V.load_or_make_structure(None, True, adapter, V.TinyProtocol(), device, out)
        paths["operators"] = str(out / "tiny_fit" / "results" / "e1_operators.npz")
        names = ["original", "mirror_mlp_k2"]
        healed_specs = []
        shift_src = write_tiny_shift(out / "tiny_shift")
        v2_root, sketch_zip, resize, crop = shift_src["v2"], shift_src["sketch"], 64, 56
        probe_dir = None
    else:
        data = T2.ProtocolData(T2._resolve(a.cache_dir), a.batch_size, a.num_workers, device.type == "cuda")
        names = [n.strip() for n in a.variants.split(",") if n.strip()]
        healed_specs = list(a.healed_run or []) + auto_runs(a)
        v2_root, sketch_zip, resize, crop = Path(a.v2_root), Path(a.sketch_zip), a.resize, a.img_size
        probe_dir = None if a.no_probe_reuse else T2._resolve(a.probe_from)
    extra, dirs = T2.healed_run_variants(healed_specs)
    variants = [T2.parse_variant(n) for n in names] + extra
    paths["healed_runs"] = {k: str(p) for k, p in dirs.items()}
    factory = T2.make_factory(adapter, SimpleNamespace(model=a.model, heal_config=None, random_seed=1), tiny, paths)
    for v in variants:
        factory.sources(v)                          # every input exists before any work starts
    log(f"CORE-07: model {model_name}, variants {[v.name for v in variants]}, out {out}, device {device}, commit "
        f"{git['git_hash'][:10]}")

    # ---- shifted sets
    sets = {}
    src, lab = v2_index(v2_root)
    counts = np.bincount(lab, minlength=1 if tiny else 1000)
    v2_check = {"n_images": int(len(lab)), "n_folders": int((counts > 0).sum()), "min_per_class": int(counts.min()),
                "max_per_class": int(counts.max()), "expected": "1000 folders 0..999, 10 images each, 10,000 in all"}
    if not tiny and (len(lab) != 10_000 or (counts != 10).any()):
        log(f"WARNING: ImageNet-V2 layout differs from the expected one: {v2_check}")
    meta = build_shift_cache("imagenetv2_mf", src, lab, Path(a.shift_cache_dir) if not tiny else out / "shift_cache",
                             resize, crop, 1 if tiny else a.decode_workers,
                             {"root": str(v2_root), "label_rule": "int(folder name)", "check": v2_check})
    sets["v2"] = {"name": "imagenetv2_mf", "meta": meta}
    if not a.no_sketch and Path(sketch_zip).exists():
        from mirror.controlled import imagenet1k_wnids

        wnids = imagenet1k_wnids()
        src, lab = sketch_index(sketch_zip, wnids)
        n_cls = int(len(np.unique(lab)))
        meta = build_shift_cache("imagenet_sketch", src, lab, Path(a.shift_cache_dir) if not tiny else out / "shift_cache",
                                 resize, crop, 1 if tiny else a.decode_workers,
                                 {"zip": str(sketch_zip), "label_rule": "rank of the WNID folder in sorted ImageNet-1k",
                                  "n_classes_present": n_cls, "zip_bytes": Path(sketch_zip).stat().st_size})
        sets["sketch"] = {"name": "imagenet_sketch", "meta": meta}
    elif not a.no_sketch:
        log(f"ImageNet-Sketch not found at {sketch_zip}; skipped")
    scache = out / "shift_cache" if tiny else Path(a.shift_cache_dir)
    loaded = {}
    for key, s in sets.items():
        imgs, labels, m = load_shift(s["name"], scache, crop)
        loaded[key] = (imgs, labels, m)
    dup = None
    if "sketch" in loaded:
        dup = near_duplicates(loaded["sketch"][0], loaded["sketch"][1], a.dup_thr)
        log(f"Sketch near-duplicate pairs under flips and rotations (thr {a.dup_thr}): {dup['n_pairs']} "
            f"{dup['by_transform']}")

    # ---- per variant: probe, validation and shifted predictions
    rows, preds = [], {}
    for v in variants:
        sources = factory.sources(v)
        probe = reusable_probe(probe_dir, model_name, v.name, sources)
        model = None
        info = None

        def get_model():
            nonlocal model, info
            if model is None:
                model, info = factory.build(v)
            return model

        if probe is not None:
            log(f"=== {v.name}: probe reused from {probe['source']}")
        recipe = probe["recipe"] if probe else T2.FEATURE_RECIPE
        amp_s = probe["amp"] if probe else a.amp
        dtype_s = probe["dtype"] if probe else a.feature_dtype
        fdtype = np.float16 if dtype_s == "float16" else np.float32
        amp = None if amp_s == "none" else amp_s
        fn = feature_fn_for(recipe)
        expect = {"model": model_name, "variant": v.name, "recipe": recipe, "amp": amp_s, "dtype": dtype_s,
                  "sources": sources, "img_size": adapter.img_size}

        def feats(split, loader, extra_expect=None, search=()):
            n = len(loader.dataset)
            for d in search:
                hit = T2.cached_features(T2.feature_path(d, model_name, v.name, split) if d != "PROBE" else
                                         probe["features_dir"] / f"{v.name}_{split}.npz",
                                         {**expect, **(extra_expect or {}), "split": split, "n": n})
                if hit is not None:
                    return hit[0], hit[1]
            fp = T2.feature_path(out, model_name, v.name, split)
            ts = time.time()
            X, y = T2.extract_features(adapter, loader, device, get_model(), amp, fdtype, tag=split, feature_fn=fn)
            T2.save_features(fp, X, y, np.arange(n), {**expect, **(extra_expect or {}), "split": split, "n": int(n),
                                                       "dim": int(X.shape[1]), "git_hash": git["git_hash"],
                                                       "seconds": round(time.time() - ts, 1)})
            return X, y

        if probe is None:
            pj = T2.probe_json_path(out, model_name, v.name)
            pz = pj.with_name(f"{v.name}_probe.npz")
            if pj.exists() and pz.exists() and T2._jsonable(json.loads(pj.read_text())["sources"]) == T2._jsonable(sources):
                probe = reusable_probe(out, model_name, v.name, sources)
                probe["source"] = f"trained here earlier ({pj})"
                log(f"=== {v.name}: probe {probe['source']}")
            else:
                log(f"=== {v.name}: training the probe here (tier2_linear's run_probe)")
                Xb, yb = feats("bank", data.loader("bank"), search=(out,))
                Xv, yv = feats("val", data.loader("val"), search=(out,))
                pr = T2.run_probe(Xb, yb, Xv, yv, data.num_classes, grid, 0.1, a.seed, True, device, solver)
                pz.parent.mkdir(parents=True, exist_ok=True)
                np.savez(pz, W=pr["W"], b=pr["b"], mean=pr["mean"], std=pr["std"],
                         holdout_positions=pr["holdout_positions"])
                row = {"top1": pr["test"]["top1"], "chosen_wd": pr["selection"]["chosen_wd"]}
                T2.write_json(pj, {"row": row, "selection": pr["selection"], "final": pr["final"], "test": pr["test"],
                                   "sources": sources, "feature_meta": {"bank": {**expect, "split": "bank"},
                                                                        "val": {**expect, "split": "val"}},
                                   "git": git, "trained_by": "scripts/core07_shift.py (tier2_linear.run_probe)"})
                probe = reusable_probe(out, model_name, v.name, sources)
                probe["source"] = f"trained here ({pj})"
        Xv, yv = feats("val", data.loader("val"), search=(("PROBE",) if probe.get("features_dir") and probe_dir and
                                                           str(probe["source"]).startswith(str(probe_dir)) else ()) + (out,))
        top_v = predict(probe, Xv, device)
        res = {"val": topk_stats(top_v, yv)}
        preds[v.name] = {"val": (top_v[:, 0] == yv)}
        for key, (imgs, labels, m) in loaded.items():
            loader = make_loader(imgs, labels, np.arange(len(labels)), a.batch_size if not tiny else 16,
                                 0 if tiny else a.num_workers, device.type == "cuda")
            X, y = feats(key, loader, {"shift_listing_sha256": m["listing_sha256"]}, search=(out,))
            top = predict(probe, X, device)
            res[key] = topk_stats(top, y)
            preds[v.name][key] = (top[:, 0] == y)
            if key == "sketch" and dup is not None:
                res["sketch_dedup"] = topk_stats(top, y, dup["keep"])
            pp = out / "preds" / model_name / f"{v.name}_{key}.npz"
            pp.parent.mkdir(parents=True, exist_ok=True)
            np.savez(pp, top5=top.astype(np.int16), labels=y.astype(np.int16))
        np.savez(out / "preds" / model_name / f"{v.name}_val.npz", top5=top_v.astype(np.int16), labels=yv.astype(np.int16))
        row = {"model": model_name, "variant": v.name, "role": role_of(v.name), "val_top1": res["val"]["top1"],
               "val_top1_recorded_by_probe": probe.get("val_top1_recorded"), "n_val": res["val"]["n"]}
        for key in loaded:
            row.update({f"{key}_top1": res[key]["top1"], f"{key}_top5": res[key]["top5"], f"n_{key}": res[key]["n"],
                        f"{key}_drop": res["val"]["top1"] - res[key]["top1"]})
        if "sketch_dedup" in res:
            row.update({"sketch_dedup_top1": res["sketch_dedup"]["top1"], "n_sketch_dedup": res["sketch_dedup"]["n"],
                        "sketch_dedup_drop": res["val"]["top1"] - res["sketch_dedup"]["top1"]})
        row.update({"probe_source": probe["source"], "probe_sha256": probe["sha256"], "feature_recipe": recipe,
                    "amp": amp_s, "feature_dtype": dtype_s, "chosen_wd": probe.get("chosen_wd"),
                    "git_hash": git["git_hash"], "git_dirty": git["git_dirty"], "seed": a.seed})
        rows.append(row)
        log(f"  {v.name}: val {res['val']['top1']:.2f}" + "".join(f", {k} {res[k]['top1']:.2f}" for k in loaded))
        if model is not None and model is not adapter.model:
            del model
        T2.free_memory(device)

    import pandas as pd

    pd.DataFrame(rows).to_csv(out / "shift_rows.csv", index=False)
    scoring = score_p71(rows, preds, a.bootstrap_resamples, a.seed)
    from mirror.meta import prereg_info

    metrics = {"experiment": EXPERIMENT, "P7.1": scoring, "rows": rows, "git": git, "prereg": prereg_info(),
               "sets": {k: s["meta"] for k, s in sets.items()},
               "sketch_near_duplicates": None if dup is None else {k: v for k, v in dup.items() if k != "keep"},
               "notes": ["ImageNet-Sketch pads some classes with flipped and rotated copies of other images "
                         "(configs/data/shift_sets.yaml); an exactly invariant model scores such a flipped pair "
                         "alike, so Sketch numbers are reported with this caveat and also without one image of each "
                         "detected near-duplicate pair",
                         "the probe of each model is the tier2_linear ImageNet probe trained on the bank; nothing is "
                         "fitted on validation or on the shifted sets"]}
    T2.write_json(out / "metrics.json", metrics)
    write_summary(out, metrics)
    write_meta(out, experiment=EXPERIMENT, config=vars(a), seeds={"global": a.seed, "bootstrap": a.seed},
               models={model_name: {"variants": [v.name for v in variants], "healed_runs": paths["healed_runs"]}},
               device=device, wall_seconds=time.time() - t0, extra={"verdict": scoring["verdict"]})
    log(f"P7.1: {scoring['verdict']}")
    return {"out": str(out), "rows": rows, "metrics": metrics}


def score_p71(rows: list, preds: dict, n_resamples: int, seed: int) -> dict:
    by = {r["variant"]: r for r in rows}
    orig = by.get("original")
    mir = [r for r in rows if r["role"] == "mirror_exact"]
    best = [r for r in rows if r["role"] == "best_baseline"]
    out = {"prediction": "Mirror-exact's drop from ImageNet to ImageNet-V2 (matched frequency, linear probe) is within "
                         "1 point of the original's drop",
           "operational": f"drop = validation top-1 - V2 top-1 (same probe); Mirror-exact = mean over healed seeds; "
                          f"pass if |drop(Mirror-exact) - drop(original)| <= {P71_MAX_GAP}",
           "verdict": "n/a"}
    if orig is None or not mir or "v2_drop" not in orig:
        out["missing"] = "original or Mirror-exact rows, or ImageNet-V2"
        return out
    d_o = orig["v2_drop"]
    d_m = float(np.mean([r["v2_drop"] for r in mir]))
    diff = d_m - d_o
    cm_val = np.mean([preds[r["variant"]]["val"] for r in mir], axis=0)
    cm_v2 = np.mean([preds[r["variant"]]["v2"] for r in mir], axis=0)
    boot = drop_bootstrap(cm_val, cm_v2, preds["original"]["val"], preds["original"]["v2"], n_resamples, seed)
    out.update({"drop_original": d_o, "drop_mirror_exact_mean": d_m,
                "drop_mirror_exact_by_seed": {r["variant"]: r["v2_drop"] for r in mir},
                "drop_mirror_exact_std": float(np.std([r["v2_drop"] for r in mir], ddof=1)) if len(mir) > 1 else None,
                "n_seeds": len(mir), "diff_mirror_minus_original": diff, "bootstrap": boot,
                "pass": bool(abs(diff) <= P71_MAX_GAP), "verdict": "PASS" if abs(diff) <= P71_MAX_GAP else "FAIL",
                "complete": len(mir) >= 3})
    for key in ("sketch",):
        if f"{key}_drop" in orig:
            out[f"{key}_reported"] = {"drop_original": orig[f"{key}_drop"],
                                      "drop_mirror_exact_mean": float(np.mean([r[f"{key}_drop"] for r in mir])),
                                      "drop_best_baseline": [r[f"{key}_drop"] for r in best]}
    out["best_baseline_reported"] = [{"variant": r["variant"], "v2_drop": r["v2_drop"],
                                      "diff_vs_original": r["v2_drop"] - d_o,
                                      "bootstrap": drop_bootstrap(preds[r["variant"]]["val"], preds[r["variant"]]["v2"],
                                                                  preds["original"]["val"], preds["original"]["v2"],
                                                                  n_resamples, seed)} for r in best]
    return out


def write_summary(out: Path, metrics: dict) -> Path:
    s = metrics["P7.1"]
    rows = metrics["rows"]
    lines = ["# CORE-07: distribution shift", "",
             f"Commit {metrics['git']['git_hash'][:10]}{' (dirty)' if metrics['git']['git_dirty'] else ''}; "
             f"PREREG_NEXT_PHASE.md sha256 {str(metrics['prereg'].get('sha256'))[:12]}.", "",
             f"**P7.1** ({s['prediction']}): **{s['verdict']}**" + ("" if s.get("complete", False) else " (fewer than "
                                                                      "3 Mirror-exact seeds or missing rows)"), ""]
    if "drop_original" in s:
        b = s["bootstrap"]
        lines += [f"Drop to ImageNet-V2: original {s['drop_original']:.2f}; Mirror-exact {s['drop_mirror_exact_mean']:.2f} "
                  f"(mean of {s['n_seeds']} seed{'' if s['n_seeds'] == 1 else 's'}); difference {s['diff_mirror_minus_original']:+.2f} points (bound "
                  f"{P71_MAX_GAP}); bootstrap 95% interval [{b['ci_low']:+.2f}, {b['ci_high']:+.2f}] "
                  f"({b['n_resamples']} resamples, seed {b['seed']}).", ""]
    keys = [k for k in ("v2", "sketch") if any(f"{k}_top1" in r for r in rows)]
    head = "| Variant | Role | ImageNet val | " + " | ".join(
        f"{'ImageNet-V2 MF' if k == 'v2' else 'Sketch'} | drop" for k in keys) + " | Probe |"
    lines += [head, "|" + "---|" * (3 + 2 * len(keys) + 1)]
    for r in rows:
        cells = [r["variant"], r["role"], f"{r['val_top1']:.2f}"]
        for k in keys:
            cells += [f"{r.get(f'{k}_top1', float('nan')):.2f}", f"{r.get(f'{k}_drop', float('nan')):.2f}"]
        cells.append(str(r["probe_source"]))
        lines.append("| " + " | ".join(cells) + " |")
    nd = metrics.get("sketch_near_duplicates")
    lines += ["", "Notes:"] + [f"- {n}" for n in metrics["notes"]]
    if nd:
        lines.append(f"- Sketch near-duplicate pairs (heuristic, threshold {nd['thr']}): {nd['n_pairs']} "
                     f"({nd['by_transform']}); {nd['n_pairs_at_half_thr']} at half and {nd['n_pairs_at_double_thr']} at "
                     f"twice the threshold. Top-1 without one image of each pair: column sketch_dedup_top1 of "
                     "shift_rows.csv.")
    lines += ["", "Files: shift_rows.csv, metrics.json, preds/, features/, meta.json."]
    path = out / "SUMMARY.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
