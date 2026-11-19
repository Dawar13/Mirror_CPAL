#!/usr/bin/env python
"""Final phase A10: full-train features for several models in one pass, the full-bank k-NN and the DINOv2 linear
protocol (FINAL_PHASE_PLAN.md v2, A10; docs/final_phase_design.md Section 5).

    python scripts/fp_fullbank.py --out runs/final/A10/fullbank \
        --model original=original:dinov2_vitb14_reg \
        --model mirror_s0=runs/final/A05/dinov2_vitb14_reg/mirror_exact/s0@780 \
        --model svdllm_s0=runs/final/A05/dinov2_vitb14_reg/svdllm/s0@780 \
        --model iosvd_s0=runs/final/A05/dinov2_vitb14_reg/iosvd/s0@780 --knn-only iosvd_s0
    python scripts/fp_fullbank.py --out runs/final/A10/fullbank --stage knn,linear        # later stages only
    python scripts/fp_fullbank.py --dry-run --out tiny_out/fp_fullbank                    # CPU, tiny, synthetic

--model NAME=SOURCE (repeatable): SOURCE is original:<model name> (a released model, --original-builder, default
mirror.adapters_timm.build_model_adapter), or a run directory with an optional @<step> (loaded with --loader, default
mirror.final.heal.students:load_student(run_dir, step, device)). --knn-only NAMES stores only the k-NN feature for
those models; --flipavg NAMES makes those models flip-averaged (A07).
Stages (--stage, default all, each resumable):
  features  one streaming pass over the train images for all models at once (mirror.final.eval.fullbank.
            stream_features; the stream is --stream, default role:stream_train = mirror.final.data.cache.get_role(
            "stream_train", --cache-dir).batches, which needs cache/stream_train_index.npz from scripts/fp_cache.py;
            a resume starts at the right shard), then the 50k validation images and their mirrors (val_features);
  knn       full-bank k-NN of the validation queries and their mirrors, k = 20, T = 0.07 (fullbank_knn);
  linear    the DINOv2 linear protocol on the stored features (mirror.final.eval.linear_protocol.run_linear_protocol;
            its docstring records the verified DINOv2 recipe and our deviations).
Outputs under --out: <name>/train_{knn,linear}.npy and val_{knn,knn_flip,linear}.npy (float16 memmaps),
train_labels.npy, train_ids.npy, val_labels.npy, progress.json, <name>/fullbank_knn.json and fullbank_knn_pred.npz,
<name>/linear_protocol.json and linear_pred.npz, summary.json, summary.csv, meta.json.
GPU: the memory cap is set first (--gpu-frac, default automatic: validation images plus --reserve-gb), then
mirror.final.prereg.require_prereg().
"""
from __future__ import annotations

import argparse
import csv
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.final.eval import fullbank as FB  # noqa: E402
from mirror.final.eval import knn_fast as KF  # noqa: E402
from mirror.final.eval.linear_protocol import LinearConfig, run_linear_protocol  # noqa: E402
from mirror.final.prereg import require_prereg, write_final_meta  # noqa: E402
from mirror.utils import log  # noqa: E402

STAGES = ("features", "knn", "linear")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="runs/final/A10/fullbank")
    p.add_argument("--model", action="append", default=None, metavar="NAME=SOURCE")
    p.add_argument("--knn-only", default="", help="comma list of model names without linear features")
    p.add_argument("--flipavg", default="", help="comma list of model names to flip-average")
    p.add_argument("--stage", default=",".join(STAGES))
    p.add_argument("--stream", default="role:stream_train",
                   help="role:stream_train (mirror.final.data.cache) or module:function of a generator")
    p.add_argument("--n-train", type=int, default=None, help="train images in the stream (if the module cannot say)")
    p.add_argument("--stream-batch", type=int, default=512, help="batch size asked of the stream")
    p.add_argument("--stream-workers", type=int, default=None, help="decode processes of the stream (its default)")
    p.add_argument("--micro-batch", type=int, default=256, help="forward batch per model")
    p.add_argument("--flush-every", type=int, default=20, help="stream batches between saved progress")
    p.add_argument("--cache-dir", default="cache", help="validation cache (mirror.data.load_val_cache)")
    p.add_argument("--val-on", default="device", choices=["device", "host"])
    p.add_argument("--loader", default=KF.DEFAULT_LOADER)
    p.add_argument("--original-builder", default=None)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--bank-chunk", type=int, default=131072)
    p.add_argument("--query-chunk", type=int, default=2048)
    p.add_argument("--linear-batch", type=int, default=1024)
    p.add_argument("--linear-iters", type=int, default=12500)
    p.add_argument("--linear-refit", default="grid", choices=["grid", "chosen", "none"])
    p.add_argument("--linear-tf32", action="store_true", help="TF32 matmuls for the classifiers (faster)")
    p.add_argument("--holdout-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0, help="linear protocol: hold-out split, initialisation, data order")
    p.add_argument("--feature-device", default=None, help="where the training features sit (default: the GPU)")
    p.add_argument("--gpu-frac", type=float, default=None)
    p.add_argument("--reserve-gb", type=float, default=16.0)
    p.add_argument("--device", default=None)
    p.add_argument("--dry-run", action="store_true", help="CPU, tiny models, a synthetic stream")
    return p.parse_args(argv)


def parse_models(specs, knn_only: str, flipavg: str) -> list:
    """[(name, source dict)] from NAME=SOURCE strings."""
    ko = {s.strip() for s in knn_only.split(",") if s.strip()}
    fa = {s.strip() for s in flipavg.split(",") if s.strip()}
    out, seen = [], set()
    for spec in specs or []:
        name, sep, src = str(spec).partition("=")
        if not sep or not name or not src:
            raise SystemExit(f"--model {spec!r}: expected NAME=SOURCE")
        if name in seen:
            raise SystemExit(f"--model: two models named {name!r}")
        seen.add(name)
        if src.startswith("original:"):
            source = {"original": src.split(":", 1)[1]}
        else:
            run, step = src, None
            head, at, tail = src.rpartition("@")
            if at and tail.isdigit():
                run, step = head, int(tail)
            source = {"run_dir": run, "step": step}
        kinds = ("knn",) if name in ko else FB.KINDS
        out.append((name, {**source, "kinds": kinds, "flipavg": name in fa}))
    unknown = (ko | fa) - seen
    if unknown:
        raise SystemExit(f"--knn-only/--flipavg name models not given with --model: {sorted(unknown)}")
    return out


def load_models(specs: list, device, loader, build_original) -> list:
    adapters: dict = {}
    build_original = KF.resolve_callable(build_original) or KF.default_build_original
    loader = KF.resolve_callable(loader)
    models = []
    for name, s in specs:
        if "original" in s:
            mname = s["original"]
            if mname not in adapters:
                adapters[mname] = build_original(mname, device)
            ad, model, info = adapters[mname], adapters[mname].model, {"original": True, "model": mname}
        else:
            ad, model, info = loader(s["run_dir"], s["step"], device)
            info = info if isinstance(info, dict) else {}
            mname = KF.model_name_of(info, s["run_dir"], {})
            if ad is None:
                if mname not in adapters:
                    adapters[mname] = build_original(mname, device)
                ad = adapters[mname]
            model = model.to(device).eval()                  # the adapter's own (original) model is not run here
            info = {**{k: info.get(k) for k in ("method", "seed", "family") if k in info}, "model": mname,
                    "run_dir": s["run_dir"], "step": s["step"]}
        for prm in model.parameters():
            prm.requires_grad_(False)
        models.append(FB.FeatureModel(name, ad, model, s["kinds"], s["flipavg"], "bf16", info))
    return models


def synthetic_stream(n: int, num_classes: int, img_size: int, seed: int = 0):
    """Dry run: a deterministic synthetic train stream; returns (stream_fn(start, batch_size), labels)."""
    from mirror.data import synthetic_dataset

    per = -(-n // num_classes)
    images, labels = synthetic_dataset(num_classes, per, img_size, seed=seed)
    images, labels = images[:n], labels[:n]

    def stream_train(start: int = 0, batch_size: int = 16):
        for s in range(int(start), n, int(batch_size)):
            e = min(s + int(batch_size), n)
            rows = np.arange(s, e, dtype=np.int64)
            yield images[s:e], labels[s:e], np.stack([rows // 32, rows % 32], 1)      # (shard, row), as the infra

    return stream_train, labels


def role_stream_factory(cache_dir, batch_size: int, workers: int | None = None):
    """(factory(start), n_train) over the infra agent's stream: mirror.final.data.cache.get_role("stream_train")
    .batches (uint8 [B, 224, 224, 3], labels, ids (shard, row)), in stream order. Role.batches ignores `start` for
    the stream, so a resume starts at the shard that holds image `start` (stream_train_index.npz shard_rows, file
    order) and skips the images before it inside that shard."""
    from mirror.final.data import cache as C

    role = C.get_role("stream_train", cache_dir)
    shard_rows = None
    try:
        with np.load(Path(cache_dir) / C.STREAM_INDEX) as z:
            if "shard_rows" in z.files:
                shard_rows = np.asarray(z["shard_rows"], dtype=np.int64)
    except OSError:
        shard_rows = None
    kw = {} if workers is None else {"workers": int(workers)}

    def factory(start: int = 0):
        if not start or shard_rows is None:
            return FB.skip_images(iter(role.batches(batch_size=int(batch_size), **kw)), int(start))
        ends = np.cumsum(shard_rows)
        i = int(np.searchsorted(ends, int(start), side="right"))
        first = int(ends[i - 1]) if i > 0 else 0
        it = role.batches(batch_size=int(batch_size), start_shard=i, **kw)
        return FB.skip_images(iter(it), int(start) - first)

    return factory, len(role)


def stream_source(a):
    """(stream factory, n_train): "role:stream_train" (default) is the infra agent's stream; anything else is a
    "module:function" generator taking start= and batch_size= (or not: then images are skipped)."""
    if a.stream.startswith("role:"):
        factory, n = role_stream_factory(a.cache_dir, a.stream_batch, a.stream_workers)
        return factory, int(a.n_train or n)
    mod_name, _, fn_name = a.stream.partition(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    n = a.n_train
    if n is None:
        for attr in ("N_TRAIN", "n_train", "train_size", "train_count"):
            v = getattr(mod, attr, None)
            if v is not None:
                n = int(v() if callable(v) else v)
                break
    if n is None:
        raise SystemExit(f"{a.stream}: the train image count is unknown; pass --n-train")
    return FB.make_stream_factory(fn, a.stream_batch), int(n)


def write_summary(out: Path, models: list) -> list:
    rows = []
    for fm in models:
        k = KF.read_json(out / fm.name / "fullbank_knn.json") or {}
        lp = KF.read_json(out / fm.name / "linear_protocol.json") or {}
        rows.append({"name": fm.name, "model": fm.info.get("model"), "run_dir": fm.info.get("run_dir"),
                     "step": fm.info.get("step"), "flip_average": fm.flipavg, "kinds": "+".join(fm.kinds),
                     "fullbank_knn_top1": k.get("top1"), "fullbank_flip_consistency": k.get("flip_consistency"),
                     "n_bank": k.get("n_bank"), "n_query": k.get("n_query"),
                     "linear_val_top1": lp.get("val_top1"), "linear_val_top5": lp.get("val_top5"),
                     "linear_chosen": (lp.get("chosen") or {}).get("name"),
                     "linear_val_top1_dinov2_rule": lp.get("val_top1_dinov2_rule"),
                     "linear_n_train": lp.get("n_train")})
    KF.write_json_atomic(out / "summary.json", {"rows": rows})
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["name"])
        w.writeheader()
        w.writerows(rows)
    return rows


def main(argv=None) -> dict:
    a = parse_args(argv)
    t0 = time.time()
    stages = [s.strip() for s in a.stage.split(",") if s.strip()]
    bad = set(stages) - set(STAGES)
    if bad:
        raise SystemExit(f"unknown stage(s) {sorted(bad)}; known {STAGES}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frac = None
    if a.dry_run:
        device = torch.device("cpu")
        st = require_prereg(allow_missing=True)
        if not a.model:
            a.model = ["original=original:tiny", "student=dryrun@1", "original_flipavg=original:tiny"]
            a.flipavg = a.flipavg or "original_flipavg"
            a.knn_only = a.knn_only or "original_flipavg"
        loader = KF.tiny_loader
        build_original = lambda name, dev: KF.tiny_original(name, dev)  # noqa: E731
        num_classes, n_train = 4, 96
        stream_fn, _ = synthetic_stream(n_train, num_classes, 56, seed=3)
        factory = FB.make_stream_factory(stream_fn, 16)
        val = KF.EvalData.synthetic(num_classes=4, bank_per_class=1, val_per_class=8, img_size=56, seed=5).val
        lcfg = LinearConfig(batch_size=32, iterations=60, holdout_frac=a.holdout_frac, seed=a.seed,
                            refit=a.linear_refit, log_every=20, base_lrs=(1e-3, 1e-2, 0.1))
    else:
        device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if device.type == "cuda":
            vb = int(np.prod(np.load(Path(a.cache_dir) / "val224_uint8.npy", mmap_mode="r").shape)) \
                if a.val_on == "device" else 0
            frac = a.gpu_frac if a.gpu_frac is not None else KF.auto_gpu_fraction(vb, a.reserve_gb, device)
            torch.cuda.set_per_process_memory_fraction(float(frac), device)
        st = require_prereg()
        loader, build_original = a.loader, a.original_builder
        lcfg = LinearConfig(batch_size=a.linear_batch, iterations=a.linear_iters, holdout_frac=a.holdout_frac,
                            seed=a.seed, refit=a.linear_refit, tf32=a.linear_tf32)
        factory = n_train = val = None
        num_classes = 1000
    specs = parse_models(a.model, a.knn_only, a.flipavg)
    if not specs:
        raise SystemExit("give at least one --model NAME=SOURCE")
    models = load_models(specs, device, loader, build_original)
    if frac is not None:
        torch.cuda.set_per_process_memory_fraction(float(frac), device)       # after imports that set their own cap
    log(f"fp_fullbank: {len(models)} model(s) {[m.name for m in models]}, stages {stages}, device {device}")
    timing = {}
    if "features" in stages:
        ts = time.time()
        if not a.dry_run:
            factory, n_train = stream_source(a)
            from mirror.data import load_val_cache

            vimg, vlab, _, _ = load_val_cache(a.cache_dir)
            val = KF.ImageSet.from_array(vimg, None, np.asarray(vlab, dtype=np.int64),
                                         device if a.val_on == "device" else "cpu", pin=a.val_on == "host",
                                         name="validation", log=log)
        FB.stream_features(models, factory, n_train, out, device, micro_batch=a.micro_batch,
                           flush_every=a.flush_every, log=log)
        FB.val_features(models, val, out, device, batch_size=a.micro_batch, log=log)
        del val
        KF.free_memory(device)
        timing["features"] = round(time.time() - ts, 1)
    if "knn" in stages:
        ts = time.time()
        for fm in models:
            FB.fullbank_knn(out, fm.name, num_classes, device, a.k, a.temperature, a.bank_chunk, a.query_chunk, log)
        timing["knn"] = round(time.time() - ts, 1)
    if "linear" in stages:
        ts = time.time()
        fdev = a.feature_device or device
        for fm in models:
            if "linear" not in fm.kinds:
                continue
            path = out / fm.name / "linear_protocol.json"
            settings = {"config": KF.jsonable(lcfg.__dict__), "n_train_bytes":
                        (out / fm.name / "train_linear.npy").stat().st_size}
            prev = KF.read_json(path)
            if prev and prev.get("settings") == KF.jsonable(settings):
                log(f"linear protocol {fm.name}: done before (top-1 {prev['val_top1']:.2f}); skipping")
                continue
            log(f"linear protocol {fm.name}")
            res = run_linear_protocol(np.load(out / fm.name / "train_linear.npy", mmap_mode="r"),
                                      np.load(out / "train_labels.npy"),
                                      np.load(out / fm.name / "val_linear.npy", mmap_mode="r"),
                                      np.load(out / "val_labels.npy"), fm.d, num_classes, lcfg, device, fdev, log)
            KF.save_predictions(out / fm.name / "linear_pred.npz", res.pop("val_pred"), np.zeros(0),
                                res.pop("val_labels"))
            KF.write_json_atomic(path, {**res, "model": fm.name, "settings": settings,
                                        "predictions": "linear_pred.npz (pred, labels; pred_flip empty)"})
            KF.free_memory(device)
        timing["linear"] = round(time.time() - ts, 1)
    rows = write_summary(out, models)
    write_final_meta(out, experiment="A10 full bank and linear protocol", config=vars(a),
                     seeds={"linear_protocol": a.seed}, models={m.name: m.describe() for m in models}, device=device,
                     wall_seconds=time.time() - t0,
                     extra={"stages": stages, "timing": timing, "gpu_frac": frac, "prereg_ok": st.get("ok"),
                            "prereg_allowed_without_check": bool(st.get("allowed_without_check")),
                            "linear_config": lcfg.__dict__})
    log(f"fp_fullbank: done in {time.time() - t0:.0f} s; {out / 'summary.csv'}")
    return {"rows": rows, "out": str(out), "timing": timing}


if __name__ == "__main__":
    main()
