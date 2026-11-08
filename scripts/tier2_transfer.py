#!/usr/bin/env python
"""Tier 2: transfer linear probes (PAPER_EXPERIMENTS.md, Section 1, Tier 2): CIFAR-100, Oxford
Flowers-102, Oxford-IIIT Pets and DTD, with the same features and probe recipe as
scripts/tier2_linear.py.

    export HF_HOME=/home/dawar/hf
    python scripts/tier2_transfer.py --model dinov2_vitb14_reg --variants original,vexact_k6,vlocal_k9 --out runs/t2
    python scripts/tier2_transfer.py --model dinov2_vitb14_reg --datasets cifar100 --variants original --out runs/t2
    python scripts/tier2_transfer.py --tiny --out tiny_out/t2      # CPU smoke test (synthetic parquet, no download)

Data (downloaded on the VM only, under $HF_HOME). Hugging Face dataset ids below were checked on the
Hugging Face API on 2026-11-08 (not gated, parquet files; split sizes from the dataset cards); the
splits have not been compared with the official split files (verify on VM):
  cifar100    uoft-cs/cifar100                 train 50,000 / test 10,000; columns img, fine_label (100)
  flowers102  dpdl-benchmark/oxford_flowers102 train 1,020 + validation 1,020 / test 6,149; image, label (102)
  pets        timm/oxford-iiit-pet             train (trainval) 3,680 / test 3,669; image, label (37)
  dtd         tanganke/dtd                     train 3,760 (train + val) / test 1,880; image, label (47);
                                               which of DTD's ten official splits this is (verify on VM)
The parquet shards are fetched with huggingface_hub.snapshot_download (repo_type "dataset", only the
needed splits; the same route scripts/build_cache.py used for ImageNet) and decoded once with the
ImageNet cache transform (mirror.data.preprocess_pil: resize 256 bicubic, centre crop 224) into
uint8 arrays under <cache-dir>/transfer/<dataset>/ (about 9 GB for CIFAR-100, under 2 GB for the
others). --loader datasets uses datasets.load_dataset instead (cache under $HF_HOME/datasets).

Probe (identical to the ImageNet probe): features = [CLS (or pooled) after the final norm, mean patch
token after the final norm], bf16 forward passes, float16 cache; multinomial logistic regression on
standardised features, full-batch L-BFGS; the weight decay is chosen on a fixed class-stratified 10%
split of the TRAINING images (train + validation merged where a dataset has an official validation
split), the probe is refitted on all training images and scored once on the test split. Reports
top-1, top-5 and mean per-class top-1 (the usual metric for Flowers and Pets).

Outputs under --out: t2_transfer.csv (one row per model, dataset and variant), T2_TRANSFER.md,
probe/<model>/<dataset>/<variant>.json, features/<model>/<dataset>/<variant>_<train|test>.npz and
t2_transfer_meta_<model>.json. Reruns skip cached images, features and finished probes.
"""
from __future__ import annotations

import torch

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.25)  # shared GPU: cap before anything is allocated

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.data import make_loader, synthetic_dataset  # noqa: E402
from mirror.utils import REPO_ROOT, StageTimer, env_info, git_info, log, set_seed  # noqa: E402


def _load_t2():
    key = "_t2_tier2_linear"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, REPO_ROOT / "scripts" / "tier2_linear.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


T2 = _load_t2()

DATASETS = {
    "cifar100": {"hf_id": "uoft-cs/cifar100", "train": ["train"], "test": "test", "image": "img",
                 "label": "fine_label", "num_classes": 100, "name": "CIFAR-100", "note": "(verify on VM)"},
    "flowers102": {"hf_id": "dpdl-benchmark/oxford_flowers102", "train": ["train", "validation"], "test": "test",
                   "image": "image", "label": "label", "num_classes": 102, "name": "Oxford Flowers-102",
                   "note": "official 1,020 / 1,020 / 6,149 split sizes (verify on VM)"},
    "pets": {"hf_id": "timm/oxford-iiit-pet", "train": ["train"], "test": "test", "image": "image", "label": "label",
             "num_classes": 37, "name": "Oxford-IIIT Pets", "note": "train = official trainval (verify on VM)"},
    "dtd": {"hf_id": "tanganke/dtd", "train": ["train"], "test": "test", "image": "image", "label": "label",
            "num_classes": 47, "name": "DTD", "note": "3,760 / 1,880: one official split, train + val (verify on VM)"},
}
SPLITS = ("train", "test")
CACHE_FORMAT = 1


# ----------------------------------------------------------------------------------------------
# Image sources
# ----------------------------------------------------------------------------------------------
def parquet_files_for_split(files, split: str) -> list:
    """Parquet shards of one split, by file name: '<split>-00000-of-00001.parquet' or '<split>.parquet'."""
    out = [str(f) for f in files if Path(f).name.startswith(f"{split}-") or Path(f).name == f"{split}.parquet"]
    return sorted(out)


def hub_parquet_files(hf_id: str, splits) -> dict:
    """Download (or reuse from $HF_HOME) only the parquet shards of `splits`; {split: [paths]}."""
    from huggingface_hub import snapshot_download

    pats = [f"*{s}-*.parquet" for s in splits] + [f"*{s}.parquet" for s in splits]
    local = snapshot_download(repo_id=hf_id, repo_type="dataset", allow_patterns=pats)
    files = sorted(Path(local).rglob("*.parquet"))
    out = {s: parquet_files_for_split(files, s) for s in splits}
    for s, fs in out.items():
        if not fs:
            raise FileNotFoundError(f"{hf_id}: no parquet shard for split {s!r} in {local} (verify the layout on VM)")
    return out


def iter_parquet(files, image_col: str, label_col: str, batch_rows: int = 256):
    """Yield (list of encoded images, list of labels) in file and row order."""
    import pyarrow.parquet as pq

    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=batch_rows, columns=[image_col, label_col]):
            yield batch.column(image_col).to_pylist(), batch.column(label_col).to_pylist()


def iter_datasets_lib(hf_id: str, split: str, image_col: str, label_col: str, batch_rows: int = 256):
    """Same as iter_parquet through datasets.load_dataset (cache under $HF_HOME/datasets)."""
    import datasets

    ds = datasets.load_dataset(hf_id, split=split)
    ds = ds.cast_column(image_col, datasets.Image(decode=False))
    for batch in ds.iter(batch_size=batch_rows):
        yield batch[image_col], batch[label_col]


def _image_bytes(item) -> bytes:
    if isinstance(item, (bytes, bytearray)):
        return bytes(item)
    if isinstance(item, dict):
        if item.get("bytes"):
            return item["bytes"]
        if item.get("path") and os.path.exists(item["path"]):
            return Path(item["path"]).read_bytes()
    raise ValueError("image entry has neither bytes nor a readable path")


def decode_one(args) -> np.ndarray:
    """Encoded image -> uint8 [crop, crop, 3] with the ImageNet cache transform (mirror.data.preprocess_pil)."""
    from PIL import Image

    from mirror.data import preprocess_pil

    data, resize, crop = args
    return preprocess_pil(Image.open(io.BytesIO(data)), resize, crop)


def build_split_cache(batches, n_total: int, out_dir: Path, split: str, resize: int, crop: int,
                      workers: int = 8) -> dict:
    """Decode `batches` (from iter_parquet / iter_datasets_lib) into <out_dir>/<split>_<crop>_uint8.npy
    and <split>_labels.npy, written atomically. Returns counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{split}_{crop}_uint8.npy"
    partial = out_dir / f"{split}_{crop}_uint8.partial.npy"
    images = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint8, shape=(n_total, crop, crop, 3))
    labels = np.empty(n_total, dtype=np.int64)
    pool = None
    if workers > 1:
        import multiprocessing as mp

        pool = mp.get_context("spawn" if os.name == "nt" else "fork").Pool(workers)
    pos, t0 = 0, time.time()
    try:
        for imgs, labs in batches:
            jobs = [(_image_bytes(im), resize, crop) for im in imgs]
            arrs = pool.map(decode_one, jobs, chunksize=8) if pool else [decode_one(j) for j in jobs]
            n = len(arrs)
            if pos + n > n_total:
                raise RuntimeError(f"more rows than the expected {n_total}")
            images[pos:pos + n] = np.stack(arrs)
            labels[pos:pos + n] = np.asarray(labs, dtype=np.int64)
            pos += n
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    if pos != n_total:
        raise RuntimeError(f"decoded {pos} rows, expected {n_total}")
    images.flush()
    del images
    os.replace(partial, final)
    np.save(out_dir / f"{split}_labels.npy", labels)
    return {"n": int(pos), "seconds": round(time.time() - t0, 1)}


def _count_rows(files) -> int:
    import pyarrow.parquet as pq

    return int(sum(pq.ParquetFile(f).metadata.num_rows for f in files))


def ensure_dataset_cache(name: str, entry: dict, cache_root: Path, loader: str = "parquet", resize: int = 256,
                         crop: int = 224, workers: int = 8, files: dict | None = None) -> dict:
    """Build (once) the uint8 cache of one dataset; return {split: (images memmap, labels)} plus meta.

    Training splits listed in entry["train"] are concatenated in that order into "train". `files`
    ({hf split: [parquet paths]}) skips the download (tests)."""
    out = Path(cache_root) / name
    meta_path = out / "meta.json"
    want = {"format": CACHE_FORMAT, "hf_id": entry["hf_id"], "train_splits": entry["train"], "test_split": entry["test"],
            "image": entry["image"], "label": entry["label"], "resize": resize, "crop": crop}
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if not (meta.get("complete") and all(meta.get(k) == v for k, v in want.items())):
        hf_splits = list(entry["train"]) + [entry["test"]]
        t0 = time.time()
        if loader == "parquet":
            files = files or hub_parquet_files(entry["hf_id"], hf_splits)
            parts = {"train": [f for s in entry["train"] for f in files[s]], "test": files[entry["test"]]}
            counts = {k: _count_rows(v) for k, v in parts.items()}
            sources = {k: [os.path.basename(f) for f in v] for k, v in parts.items()}
            make = {k: (lambda v=v: iter_parquet(v, entry["image"], entry["label"])) for k, v in parts.items()}
        elif loader == "datasets":
            import datasets

            def chain(splits):
                for s in splits:
                    yield from iter_datasets_lib(entry["hf_id"], s, entry["image"], entry["label"])

            n = {s: len(datasets.load_dataset(entry["hf_id"], split=s)) for s in hf_splits}
            counts = {"train": sum(n[s] for s in entry["train"]), "test": n[entry["test"]]}
            sources = {"train": list(entry["train"]), "test": [entry["test"]]}
            make = {"train": lambda: chain(entry["train"]), "test": lambda: chain([entry["test"]])}
        else:
            raise ValueError("loader must be parquet or datasets")
        info = {}
        for split in SPLITS:
            log(f"  {name}: decoding {split} ({counts[split]} images, resize {resize}, crop {crop})")
            info[split] = build_split_cache(make[split](), counts[split], out, split, resize, crop, workers)
        labels = {s: np.load(out / f"{s}_labels.npy") for s in SPLITS}
        for s in SPLITS:
            if labels[s].min() < 0 or labels[s].max() >= entry["num_classes"]:
                raise ValueError(f"{name} {s}: labels outside 0..{entry['num_classes'] - 1}")
        missing = sorted(set(range(entry["num_classes"])) - set(np.unique(labels["train"]).tolist()))
        if missing:
            raise ValueError(f"{name}: classes {missing[:10]} have no training image")
        meta = {**want, "complete": True, "counts": {s: int(len(labels[s])) for s in SPLITS}, "loader": loader,
                "sources": sources, "decode": info, "seconds": round(time.time() - t0, 1),
                "interpolation": "bicubic", "note": entry.get("note", "")}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    data = {s: (np.load(out / f"{s}_{crop}_uint8.npy", mmap_mode="r"), np.load(out / f"{s}_labels.npy")) for s in SPLITS}
    return {"data": data, "meta": meta, "dir": str(out)}


# ----------------------------------------------------------------------------------------------
# Tiny mode: a synthetic dataset written as a local parquet file (exercises decoding; no download)
# ----------------------------------------------------------------------------------------------
def write_tiny_parquet(root: Path, num_classes: int = 3, per_train: int = 10, per_test: int = 4, size: int = 40,
                       seed: int = 0) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, per, s in (("train", per_train, seed), ("test", per_test, seed + 1)):
        imgs, labels = synthetic_dataset(num_classes, per, size, seed=s)
        enc = []
        for im in imgs:
            buf = io.BytesIO()
            Image.fromarray(im).save(buf, format="PNG")
            enc.append({"bytes": buf.getvalue(), "path": None})
        table = pa.table({"image": enc, "label": labels.astype(np.int64)})
        path = root / f"{split}-00000-of-00001.parquet"
        pq.write_table(table, path)
        files[split] = [str(path)]
    return files


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dinov2_vitb14_reg")
    p.add_argument("--variants", default="original", help="variant names as in scripts/tier2_linear.py")
    p.add_argument("--datasets", default=",".join(DATASETS), help=f"comma-separated subset of {list(DATASETS)}")
    p.add_argument("--out", default=None, help="output directory (default runs/t2; tiny_out/t2 with --tiny)")
    p.add_argument("--cache-dir", default="cache", help="decoded images go to <cache-dir>/transfer/<dataset>/")
    p.add_argument("--loader", default="parquet", choices=["parquet", "datasets"])
    p.add_argument("--decode-workers", type=int, default=8)
    p.add_argument("--operators", default=None)
    p.add_argument("--m1-structure", default=None)
    p.add_argument("--heal-dir", default=None)
    p.add_argument("--heal-config", default=None)
    p.add_argument("--random-seed", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--amp", default="bf16", choices=["bf16", "none"])
    p.add_argument("--feature-dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--wd-grid", default=T2.DEFAULT_WD_GRID)
    p.add_argument("--no-wd-refine", action="store_true")
    p.add_argument("--holdout-frac", type=float, default=0.1)
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--history-size", type=int, default=20)
    p.add_argument("--tol-grad", type=float, default=1e-6)
    p.add_argument("--probe-tf32", action="store_true")
    p.add_argument("--skip-probe", action="store_true")
    p.add_argument("--force-features", action="store_true")
    p.add_argument("--force-probe", action="store_true")
    p.add_argument("--memory-fraction", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--tiny", action="store_true", help="tiny random ViT, synthetic parquet dataset, CPU")
    return p.parse_args(argv)


CSV_COLUMNS = ["model", "dataset", "variant", "label", "family", "method", "k", "top1", "top5", "mean_per_class_top1",
               "chosen_wd", "wd_at_grid_edge", "holdout_top1_at_chosen", "final_hit_iter_limit", "feature_dim",
               "n_train", "n_test", "num_classes", "hf_id", "t_extract_s", "t_probe_s", "git_hash", "git_dirty", "seed",
               "date"]


def write_tables(out: Path) -> list:
    import pandas as pd

    rows = []
    for p in sorted((Path(out) / "probe").glob("*/*/*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if "transfer_row" in r:
            rows.append(r["transfer_row"])
    if not rows:
        return rows
    rows.sort(key=lambda r: (r["model"], r["dataset"], r["k"], r["family"] != "original", r["variant"]))
    df = pd.DataFrame(rows)
    df = df[[c for c in CSV_COLUMNS if c in df.columns] + [c for c in df.columns if c not in CSV_COLUMNS]]
    df.to_csv(Path(out) / "t2_transfer.csv", index=False)
    lines = ["# Tier 2: transfer linear probes", "",
             "Same features and probe as the ImageNet linear probe (T2_LINEAR.md): [CLS (or pooled) after the "
             "final norm, mean patch token after the final norm], bf16 forward passes, float16 cache, multinomial "
             "logistic regression on standardised features (L-BFGS). Weight decay chosen on a fixed "
             "class-stratified 10% split of the training images, never on the test split; refitted on all "
             "training images and scored once on the test split. Images: resize 256 (bicubic), centre crop 224.", ""]
    for model in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == model]
        variants = []
        for r in rs:
            if r["variant"] not in variants:
                variants.append(r["variant"])
        dsets = [d for d in DATASETS if any(r["dataset"] == d for r in rs)] + sorted(
            {r["dataset"] for r in rs} - set(DATASETS))
        lines += [f"## {model}: top-1 (mean per-class top-1)", "",
                  "| Variant | " + " | ".join(dsets) + " |", "|---|" + "---|" * len(dsets)]
        for v in variants:
            cells = []
            for d in dsets:
                r = next((r for r in rs if r["variant"] == v and r["dataset"] == d), None)
                cells.append("-" if r is None else f"{r['top1']:.2f} ({r['mean_per_class_top1']:.2f})")
            lines.append(f"| {v} | " + " | ".join(cells) + " |")
        lines.append("")
    git = sorted({str(r["git_hash"])[:10] + (" (dirty)" if r.get("git_dirty") else "") for r in rows})
    lines += ["Datasets (Hugging Face ids, verify on VM): " + "; ".join(
        f"{d}: {DATASETS[d]['hf_id']}" for d in DATASETS if any(r['dataset'] == d for r in rows)) + ".",
              f"Source: `t2_transfer.csv` and `probe/<model>/<dataset>/<variant>.json`. Commit(s): {', '.join(git)}; "
              f"seed(s): {', '.join(sorted({str(r['seed']) for r in rows}))}.", ""]
    (Path(out) / "T2_TRANSFER.md").write_text("\n".join(lines), encoding="utf-8")
    return rows


def main(argv=None) -> dict:
    a = parse_args(argv)
    tiny = a.tiny
    variants = T2.parse_variants(a.variants)
    model_name = "tiny" if tiny else a.model
    out = Path(a.out) if a.out else REPO_ROOT / ("tiny_out/t2" if tiny else "runs/t2")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() and not tiny else "cpu"))
    T2.apply_memory_cap(a.memory_fraction)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    set_seed(a.seed)
    git = git_info()
    timer = StageTimer()
    if not tiny and not os.environ.get("HF_HOME"):
        log("WARNING: HF_HOME is not set; Hugging Face downloads go to its default cache, not $HF_HOME")
    amp = None if a.amp == "none" else a.amp
    fdtype = np.float16 if a.feature_dtype == "float16" else np.float32
    grid = [float(x) for x in a.wd_grid.split(",") if x.strip()]
    solver = {"max_iter": a.max_iter, "history_size": a.history_size, "tol_grad": a.tol_grad}
    settings = {"grid": grid, "refine": not a.no_wd_refine, "holdout_frac": a.holdout_frac, "seed": a.seed,
                "solver": solver, "tf32": a.probe_tf32}
    cache_root = (out if tiny else T2._resolve(a.cache_dir)) / "transfer"

    with timer.stage("datasets"):
        if tiny:
            files = write_tiny_parquet(out / "tiny_parquet")
            registry = {"tiny_synth": {"hf_id": "local/tiny", "train": ["train"], "test": "test", "image": "image",
                                       "label": "label", "num_classes": 3, "name": "tiny synthetic"}}
            dsets = {"tiny_synth": ensure_dataset_cache("tiny_synth", registry["tiny_synth"], cache_root, "parquet",
                                                        resize=64, crop=56, workers=1, files=files)}
        else:
            names = [d.strip() for d in a.datasets.split(",") if d.strip()]
            bad = set(names) - set(DATASETS)
            if bad:
                raise ValueError(f"unknown datasets {sorted(bad)}; known {list(DATASETS)}")
            registry = {d: DATASETS[d] for d in names}
            dsets = {d: ensure_dataset_cache(d, registry[d], cache_root, a.loader, workers=a.decode_workers)
                     for d in names}
    for d, ds in dsets.items():
        log(f"{d}: {ds['meta']['counts']} images in {ds['dir']}")

    with timer.stage("load"):
        adapter = T2.build_adapter(a.model, tiny, device, a.img_size)
        paths = T2.resolve_paths(a)
        if tiny and paths["operators"] is None and any(v.family in ("mlp", "m1") for v in variants):
            from mirror import variants as V

            V.load_or_make_structure(None, True, adapter, V.TinyProtocol(), device, out)
            paths["operators"] = str(out / "tiny_fit" / "results" / "e1_operators.npz")
        factory = T2.VariantFactory(adapter, a.model, tiny, paths["operators"], paths["m1_structure"],
                                    paths["heal_dir"], a.heal_config, a.random_seed)
        kind = T2.feature_kind(adapter)
    pin = device.type == "cuda"
    bs = min(a.batch_size, 16) if tiny else a.batch_size
    nw = 0 if tiny else a.num_workers

    def loader(d, split):
        imgs, labs = dsets[d]["data"][split]
        return make_loader(imgs, labs, np.arange(len(labs)), bs, nw, pin)

    results, run_timing = [], {}
    for v in variants:
        vt = StageTimer()
        log(f"=== {v.name}: {v.label}")
        sources = factory.sources(v)
        model, info = None, None
        feats, t_ext = {}, {}
        for d in dsets:
            t_ext[d] = 0.0
            for split in SPLITS:
                fp = out / "features" / model_name / d / f"{v.name}_{split}.npz"
                expect = {"model": model_name, "variant": v.name, "recipe": T2.FEATURE_RECIPE, "amp": a.amp,
                          "dtype": a.feature_dtype, "sources": sources, "img_size": adapter.img_size, "dataset": d,
                          "hf_id": registry[d]["hf_id"], "split": split, "n": int(len(dsets[d]["data"][split][1]))}
                hit = None if a.force_features else T2.cached_features(fp, expect)
                if hit is not None:
                    feats[(d, split)] = hit
                    continue
                if model is None:
                    with vt.stage(f"build {v.name}"):
                        model, info = factory.build(v)
                t0 = time.time()
                with vt.stage(f"extract {v.name} {d} {split}"):
                    X, y = T2.extract_features(adapter, loader(d, split), device, model, amp, fdtype, tag=f"{d} {split}")
                t_ext[d] += time.time() - t0
                meta = {**expect, "dim": int(X.shape[1]), "feature_kind": kind, "git_hash": git["git_hash"],
                        "seed": a.seed, "seconds": round(time.time() - t0, 2)}
                T2.save_features(fp, X, y, np.arange(len(y)), meta)
                feats[(d, split)] = (X, y, meta)
        if model is not None and model is not adapter.model:
            del model
        T2.free_memory(device)
        vj = out / "probe" / model_name / f"{v.name}_variant.json"      # shared with scripts/tier2_linear.py
        if info is None:
            info = json.loads(vj.read_text(encoding="utf-8"))["info"] if vj.exists() else {
                "variant": v.name, "label": v.label, "family": v.family, "method": v.method, "k": v.k}
        else:
            T2.write_json(vj, {"info": info, "sources": sources, "git": git})
        if a.skip_probe:
            run_timing[v.name] = vt.stages
            continue
        for d in dsets:
            pj = out / "probe" / model_name / d / f"{v.name}.json"
            if pj.exists() and not a.force_probe and t_ext[d] == 0.0:
                prev = json.loads(pj.read_text(encoding="utf-8"))
                if prev.get("probe_settings") == T2._jsonable(settings):
                    log(f"  {d}: probe done before, top-1 {prev['transfer_row']['top1']:.2f}; skipping")
                    results.append(prev["transfer_row"])
                    continue
            (Xt, yt, mt), (Xs, ys, ms) = feats[(d, "train")], feats[(d, "test")]
            torch.backends.cuda.matmul.allow_tf32 = bool(a.probe_tf32)
            t0 = time.time()
            with vt.stage(f"probe {v.name} {d}"):
                pr = T2.run_probe(Xt, yt, Xs, ys, registry[d]["num_classes"], grid, a.holdout_frac, a.seed,
                                  not a.no_wd_refine, device, solver)
            torch.backends.cuda.matmul.allow_tf32 = False
            sel, fin, test = pr["selection"], pr["final"], pr["test"]
            log(f"  {v.name} on {d}: top-1 {test['top1']:.2f}, mean per-class {test['mean_per_class_top1']:.2f} "
                f"(wd {sel['chosen_wd']:.3g})")
            row = {"model": model_name, "dataset": d, "variant": v.name, "label": v.label, "family": v.family,
                   "method": v.method, "k": v.k, "top1": test["top1"], "top5": test["top5"],
                   "mean_per_class_top1": test["mean_per_class_top1"], "chosen_wd": sel["chosen_wd"],
                   "wd_at_grid_edge": sel["wd_at_grid_edge"], "holdout_top1_at_chosen": sel["holdout_top1_at_chosen"],
                   "final_hit_iter_limit": fin["hit_limit"], "feature_dim": int(Xt.shape[1]), "n_train": int(len(yt)),
                   "n_test": int(len(ys)), "num_classes": registry[d]["num_classes"], "hf_id": registry[d]["hf_id"],
                   "t_extract_s": round(t_ext[d], 1), "t_probe_s": round(time.time() - t0, 1),
                   "git_hash": git["git_hash"], "git_dirty": git["git_dirty"], "seed": a.seed,
                   "date": time.strftime("%Y-%m-%d %H:%M:%S")}
            T2.write_json(pj, {"transfer_row": row, "selection": sel, "final": fin, "test": test, "variant_info": info,
                               "sources": sources, "dataset_meta": dsets[d]["meta"],
                               "feature_meta": {"train": mt, "test": ms}, "probe_settings": settings,
                               "git": git, "env": env_info(device)})
            results.append(row)
            write_tables(out)
        run_timing[v.name] = vt.stages
        del feats
        T2.free_memory(device)

    rows = write_tables(out)
    T2.write_json(out / f"t2_transfer_meta_{model_name}.json", {
        "experiment": "Tier 2 transfer linear probes", "model": model_name, "variants": [v.name for v in variants],
        "datasets": {d: {**registry[d], "cache": dsets[d]["meta"]} for d in dsets}, "args": vars(a), "paths": paths,
        "git": git, "env": env_info(device), "seed": a.seed, "feature_recipe": T2.FEATURE_RECIPE,
        "timing": {"stages": timer.stages, "variants": run_timing, "total_s": timer.total()}})
    log(f"done in {timer.total() / 60:.1f} min; {len(rows)} rows in {out / 't2_transfer.csv'}")
    return {"out": str(out), "rows": results, "all_rows": rows}


if __name__ == "__main__":
    main()
