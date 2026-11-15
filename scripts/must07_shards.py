"""Split MUST-07 rows of one model over parallel processes, then merge them back (bookkeeping only).

must07_compressors.py keeps rows_<model>.csv in memory and rewrites it after every row, so two processes must
never share an output folder. A shard is a run with its own --out folder; `seed` copies the original-model row
(and its predictions and the label file) from the main folder into each shard so the shard does not evaluate
the original again; `merge` copies every new row, grid row, prediction and info file from the shards into the
main folder (a row already in the main folder is never replaced) and writes the CSVs atomically. Afterwards run
`python scripts/must07_compressors.py --postprocess-only` to rebuild bootstrap.csv, gate_b.json and SUMMARY.md.

    python scripts/must07_shards.py seed  --model dinov2_vitb14_reg runs/M7_shards/z0_6both runs/M7_shards/c1 ...
    python scripts/must07_shards.py merge --model dinov2_vitb14_reg runs/M7_shards/*
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
KEY = ["model", "size", "method", "fit_seed"]                 # as scripts/must07_compressors.py
GRID_KEY = ["model", "size", "method", "fit_seed", "hparams"]


def _p(s: str) -> Path:
    p = Path(s)
    return p if p.is_absolute() else REPO_ROOT / p


def _keys(df: pd.DataFrame, cols: list) -> list:
    return [tuple(str(int(v)) if isinstance(v, float) and v == int(v) else str(v) for v in r)
            for r in df[cols].itertuples(index=False)]


def _atomic_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def seed(main: Path, model: str, shards: list) -> None:
    rows = pd.read_csv(main / f"rows_{model}.csv")
    orig = rows[(rows["size"] == "original") & (rows["method"] == "original")]
    if len(orig) != 1:
        raise SystemExit(f"expected one original row in {main}/rows_{model}.csv, found {len(orig)}")
    for sh in shards:
        dst = sh / f"rows_{model}.csv"
        if dst.exists():
            print(f"{sh}: rows file exists, left as it is")
            continue
        (sh / "preds" / model / "original").mkdir(parents=True, exist_ok=True)
        for rel in (Path("preds") / model / "labels.npz", Path("preds") / model / "original" / "original_s-1.npz"):
            shutil.copy2(main / rel, sh / rel)
        orig.to_csv(dst, index=False)
        print(f"{sh}: seeded with the original row")


def merge(main: Path, model: str, shards: list) -> None:
    rows_path, grid_path = main / f"rows_{model}.csv", main / f"grid_{model}.csv"
    rows = pd.read_csv(rows_path)
    grid = pd.read_csv(grid_path) if grid_path.exists() else None
    have, have_g = set(_keys(rows, KEY)), set(_keys(grid, GRID_KEY)) if grid is not None else set()
    add, add_g, n_files = [], [], 0
    for sh in shards:
        f = sh / f"rows_{model}.csv"
        if not f.exists():
            print(f"{sh}: no rows file, skipped")
            continue
        df = pd.read_csv(f)
        df = df[df["size"] != "original"]
        new = [i for i, k in zip(df.index, _keys(df, KEY)) if k not in have]
        dup = len(df) - len(new)
        for i in new:
            r = df.loc[i]
            have.add(_keys(df.loc[[i]], KEY)[0])
            for kind, ext in (("preds", "npz"), ("info", "json")):
                src = sh / kind / model / str(r["size"]) / f"{r['method']}_s{int(r['fit_seed'])}.{ext}"
                if src.exists():
                    dst = main / kind / model / str(r["size"]) / src.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    n_files += 1
        add.append(df.loc[new])
        g = sh / f"grid_{model}.csv"
        if g.exists():
            gd = pd.read_csv(g)
            keep = [i for i, k in zip(gd.index, _keys(gd, GRID_KEY)) if k not in have_g]
            have_g.update(_keys(gd.loc[keep], GRID_KEY))
            add_g.append(gd.loc[keep])
        for m in sorted((sh / "meta").glob("*.json")) if (sh / "meta").exists() else []:
            dst = main / "meta" / f"shard_{sh.name}_{m.name}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(m, dst)
        print(f"{sh}: {len(new)} new rows, {dup} already present")
    n_new = sum(len(a) for a in add)
    if n_new:
        _atomic_csv(pd.concat([rows] + add, ignore_index=True), rows_path)
    if add_g and sum(len(a) for a in add_g):
        _atomic_csv(pd.concat(([grid] if grid is not None else []) + add_g, ignore_index=True), grid_path)
    print(f"merged {n_new} rows and {n_files} prediction or info files into {rows_path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=("seed", "merge"))
    p.add_argument("shards", nargs="+")
    p.add_argument("--model", required=True)
    p.add_argument("--main", default="results/MUST-07", help="the main MUST-07 output folder")
    a = p.parse_args(argv)
    main_dir, shards = _p(a.main), [_p(s) for s in a.shards]
    if any(s.resolve() == main_dir.resolve() for s in shards):
        raise SystemExit("a shard folder cannot be the main folder")
    for s in shards:
        s.mkdir(parents=True, exist_ok=True)
    (seed if a.action == "seed" else merge)(main_dir, a.model, shards)
    return 0


if __name__ == "__main__":
    sys.exit(main())
