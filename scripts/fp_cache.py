#!/usr/bin/env python
"""Final-phase data roles (mirror/final/data/cache.py): build, check, stream, estimate disk.

    python scripts/fp_cache.py estimate                         # disk per role (no network)
    python scripts/fp_cache.py build --roles fit3,coco --workers 24 --delete-shards
    python scripts/fp_cache.py build --roles heal_ext --workers 24 --delete-shards
    python scripts/fp_cache.py index-stream                     # stream_train IDs and labels (label columns only)
    python scripts/fp_cache.py check                            # -> results/data_check/final_roles.json
    python scripts/fp_cache.py roles                            # every role, old and new, through the accessor
    python scripts/fp_cache.py stream --shards 0:2 --workers 24 # throughput of the full-train stream (timing only)
    python scripts/fp_cache.py --dry-run [--out DIR]            # synthetic shards, whole pipeline, CPU, no downloads

Every build is restart-safe: a complete role is skipped, an interrupted one resumes after its last finished shard.
Downloads go under $HF_HOME (VM only; ImageNet-1k needs the Hugging Face token). Builds compare the first
--parity images of every role with the torchvision transform and stop on any difference. --delete-shards removes
shards this build downloaded once their rows are decoded (the train cache's own shards are kept).
Exit code 1 when a check fails.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.final.data import cache as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _roles(text: str) -> list:
    names = [x.strip() for x in text.split(",") if x.strip()]
    bad = [n for n in names if n not in C.NEW_ROLES]
    if bad:
        raise SystemExit(f"unknown roles {bad}; new roles: {sorted(C.NEW_ROLES)}")
    return names


def cmd_build(a) -> int:
    if not os.environ.get("HF_HOME"):
        print("WARNING: HF_HOME is not set; downloads go to the default Hugging Face cache", flush=True)
    names = _roles(a.roles)
    cache = Path(a.cache_dir)
    imagenet = [n for n in names if n in ("fit3", "heal_ext")]
    out = {}
    if imagenet:
        out.update(C.build_imagenet_roles(imagenet, C.imagenet_source(cache), cache, a.workers, a.delete_shards,
                                          a.parity))
    if "coco" in names:
        out["coco"] = C.build_coco(C.coco_source(), cache, workers=a.workers, delete_shards=a.delete_shards,
                                   parity=a.parity)
    for n, m in out.items():
        print(f"{n}: n={m.get('n')} shards={m.get('n_shards')} sha256={str(m.get('selection_sha256'))[:16]} "
              f"parity={m.get('parity')}")
    return 0


def cmd_index_stream(a) -> int:
    C.build_stream_index(C.imagenet_source(a.cache_dir), a.cache_dir, threads=a.threads)
    return 0


def cmd_check(a) -> int:
    rep = C.check_roles(a.cache_dir, a.out, require=tuple(x for x in a.require.split(",") if x))
    for name, r in rep["roles"].items():
        print(f"{name:12s} n={r['n']:8d} unique={r['unique']:8d} per class {r['per_class_min']}..{r['per_class_max']}")
    print(("PASS" if rep["ok"] else "FAIL: " + "; ".join(rep["problems"])) + f" -> {a.out}")
    return 0 if rep["ok"] else 1


def cmd_roles(a) -> int:
    for r in C.list_roles(a.cache_dir):
        if r["present"]:
            print(f"{r['role']:12s} n={r['n']:8d} {r['size']} px {r['dataset']}"
                  + (" (streamed)" if r["streamed"] else ""))
        else:
            print(f"{r['role']:12s} missing ({r['why']})")
    return 0


def cmd_stream(a) -> int:
    lo, _, hi = a.shards.partition(":")
    src = C.imagenet_source(a.cache_dir)
    t0, n = time.time(), 0
    for x, y, ids in C.stream_batches(src, a.batch_size, a.workers, int(lo or 0), int(hi) if hi else None,
                                      delete_after=a.delete, log=print, log_every_s=30):
        n += len(y)
    dt = time.time() - t0
    rec = {"images": n, "seconds": round(dt, 1), "images_per_s": round(n / max(dt, 1e-9), 1),
           "workers": a.workers, "shards": a.shards, "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(json.dumps(rec))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rec, indent=2))
    return 0


def cmd_estimate(a) -> int:
    rows = C.disk_estimates(a.cache_dir)
    print(C.format_estimates(rows))
    return 0


def dry_run(out_dir: Path | None) -> int:
    """The whole pipeline on synthetic shards: 4 classes, tiny images, scaled role sizes, no network."""
    tmp = None
    if out_dir is None:
        tmp = tempfile.mkdtemp(prefix="fp_cache_dry_")
        out_dir = Path(tmp)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        n_cls = 4
        C.write_synthetic_shards(out_dir / "imagenet", n_shards=14, rows_per_shard=12, n_classes=n_cls, seed=0)
        C.write_synthetic_shards(out_dir / "coco", n_shards=3, rows_per_shard=8, n_classes=1, seed=1, labels=False,
                                 image_id=True, fmt="JPEG")
        src = C.LocalParquetSource(out_dir / "imagenet", dataset="imagenet_train", cached=[])
        coco = C.LocalParquetSource(out_dir / "coco", dataset="coco_train", label_col=None, extra_cols=("image_id",))
        cache = out_dir / "cache"
        chain = (C.scaled_spec(C.FIT3, per_class=1, n=n_cls, resize=32, crop=28),
                 C.scaled_spec(C.HEAL_EXT, per_class=3, n=3 * n_cls, resize=32, crop=32))
        built = C.build_imagenet_roles(["fit3", "heal_ext"], src, cache, workers=0, parity=0, chain=chain,
                                       n_classes=n_cls, used_need=4, msel_per_class=2)
        coco_spec = C.scaled_spec(C.COCO, n=6, resize=32, crop=28)
        built["coco"] = C.build_coco(coco, cache, spec=coco_spec, workers=0, parity=0)
        C.build_stream_index(src, cache)
        n_stream = sum(len(src.labels(f)) for f in src.files())
        rep = C.check_roles(cache, out_dir / "final_roles.json",
                            expected={"fit3": n_cls, "heal_ext": 3 * n_cls, "coco": 6, "stream_train": n_stream},
                            require=("fit3", "heal_ext", "coco", "stream_train"))
        n = sum(len(y) for _, y, _ in C.stream_batches(src, batch_size=10, workers=0, start_shard=0, stop_shard=3,
                                                       resize=32, crop=28))
        print(json.dumps({"built": {k: v["n"] for k, v in built.items()}, "check_ok": rep["ok"],
                          "problems": rep["problems"], "streamed_images": n}, indent=2))
        print(C.format_estimates(C.disk_estimates(cache)))
        return 0 if rep["ok"] else 1
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="synthetic shards, whole pipeline, CPU, no downloads")
    p.add_argument("--out", default=None, help="dry run: keep its files here")
    p.add_argument("--cache-dir", default=str(ROOT / "cache"))
    sub = p.add_subparsers(dest="cmd")
    b = sub.add_parser("build")
    b.add_argument("--roles", required=True, help="comma-separated: fit3, heal_ext, coco")
    b.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    b.add_argument("--delete-shards", action="store_true")
    b.add_argument("--parity", type=int, default=32, help="images compared with the torchvision transform")
    s = sub.add_parser("index-stream")
    s.add_argument("--threads", type=int, default=8)
    c = sub.add_parser("check")
    c.add_argument("--out", default=str(ROOT / "results" / "data_check" / "final_roles.json"))
    c.add_argument("--require", default="fit3,coco")
    sub.add_parser("roles")
    t = sub.add_parser("stream")
    t.add_argument("--shards", default="0:2", help="start:stop shard indices")
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    t.add_argument("--delete", action="store_true")
    t.add_argument("--out", default=None)
    sub.add_parser("estimate")
    a = p.parse_args(argv)
    if a.dry_run:
        return dry_run(Path(a.out) if a.out else None)
    handlers = {"build": cmd_build, "index-stream": cmd_index_stream, "check": cmd_check, "roles": cmd_roles,
                "stream": cmd_stream, "estimate": cmd_estimate}
    if a.cmd not in handlers:
        p.print_help()
        return 2
    return handlers[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
