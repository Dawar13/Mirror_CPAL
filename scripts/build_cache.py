#!/usr/bin/env python
"""B1: download the ImageNet-1k validation parquet shards (under $HF_HOME) and build the cache.

    export HF_HOME=/big_disk/hf
    python scripts/build_cache.py --config configs/e1.yaml

Writes cache/val224_uint8.npy (about 7.5 GB), cache/val_labels.npy, cache/splits.npz, cache/meta.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mirror.data import build_val_cache  # noqa: E402
from mirror.utils import REPO_ROOT, load_config, log  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "e1.yaml"))
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if not os.environ.get("HF_HOME"):
        log("warning: HF_HOME is not set; Hugging Face will download to its default cache")
    meta = build_val_cache(cfg)
    log(json.dumps({k: v for k, v in meta.items() if k != "source_files"}, indent=2))
    expect = {"fit": 2000, "heldout": 1000, "bank": 39000, "query": 8000}
    if meta["split_sizes"] != expect:
        log(f"warning: split sizes {meta['split_sizes']} differ from the plan {expect}")


if __name__ == "__main__":
    main()
