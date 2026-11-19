#!/usr/bin/env python
"""Teaser tiles (paper Figure 1): the mirror-twin patches of the earlier teaser, regenerated at full resolution.

    python scripts/fp_fig_teaser.py                                   # VM: model, caches and the original images
    python scripts/fp_fig_teaser.py --select recompute                # VM: also re-find the top patches (50k images)
    python scripts/fp_fig_teaser.py --source cache --no-model         # no originals, no model: cache pixels only
    python scripts/fp_fig_teaser.py --from-tiles results/A14/teaser/tiles.npz    # re-draw from saved tiles (laptop)
    .venv/Scripts/python.exe scripts/fp_fig_teaser.py --dry-run       # CPU: tiny random ViT and random images

What is reused. The earlier teaser (--teaser; default: results/MUST-05/dinov2_vitb14_reg/teaser.npz, written by
scripts/must05_twins.py part "teaser", or its copy results/paper_next/data/fig1_teaser.npz, whichever exists; both
are git-ignored, so one must be uploaded to the VM, about 3 MB) fixes the choice: the
blocks (1, 3, 6), the 8 strongest twin pairs of each block (ranked on fit set 0 by the matching weight
w = 2 S_ij - S_ii - S_jj), their twin correlations S_ij, the 50,000 k-NN bank images it searched (vis_rows: rows of
cache/train224_uint8.npy, the train memmap that also holds the fit sets) and, per neuron, the 16 images with the
largest activation (top_img, at most one patch per image) with the token of that maximum (top_pos). Nothing is
re-ranked: the same pairs, images and tokens are drawn (--select stored, the default).

What is new. (1) The model is run again (DINOv2-B/14 reg, the same paired pass of scripts/f7_visuals.py collect
with bf16 autocast on CUDA, as MUST-05): with --select stored on the images of the drawn tiles, to check that each
neuron's largest activation in each image is at the stored token and has the stored value (verify.csv); with
--select recompute over all 50,000 images, to re-find the top patches (agreement with the stored choice is printed).
--no-model skips it. (2) Each tile is cut from the original ImageNet image at its own resolution instead of from the
224-pixel cache: the token's 3 x 3 patch window (42 x 42 cache pixels, the earlier crop) is mapped back through the
cache's preprocessing (shorter side resized to 256 with bicubic interpolation, centre crop 224; the geometry of
mirror/final/data/cache.py resize_centre_crop) to a box in the original image, which is resampled once (Lanczos) to
--tile-px pixels (default 168 = 4 x 42). Parts of the window outside the original image are filled with light grey
(228), as before. The originals are the rows of the ImageNet-1k parquet shards recorded for the cache
(cache/train_meta.json shards_used, repo and revision; cache/train_index.npz src224: shard and row of every cache
row), read from the local Hugging Face cache under $HF_HOME and downloaded there if missing (VM only; --no-download
forbids it). A tile whose original is unavailable falls back to the cache pixels, enlarged by pixel repetition
(printed, and marked in tiles.csv). --source cache uses the cache pixels for every tile.

Outputs: <out>/<stem>.pdf and .png (300 dpi; stem fig_twins_hires): the rows --rows (block:pair rank, default
1:1,1:2,3:1,6:1, the rows of scripts/make_paper_fig_twins.py), --figure-patches tiles per neuron, neuron i on the
left, its twin j on the right, the twin correlation S_ij at the right edge; <out>/<stem>_tiles/: one PNG per tile
of the figure rows (for the method diagram); <results>/A14/teaser/: tiles.npz (every tile of --pairs pairs and
--patches patches per neuron, with provenance), tiles.csv, verify.csv, metrics.json and meta.json
(mirror.final.prereg.write_final_meta). Style: scripts/paper_style.py (apply(), C); text in ink colours; tiles
embedded at their pixel size in the PDF (no resampling by matplotlib). No randomness.

GPU: the memory cap (--gpu-frac) is set before anything is allocated, then mirror.final.prereg.require_prereg().
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(1, str(ROOT))

STEM = "fig_twins_hires"
FILL = 228                               # light grey outside the image (scripts/f7_visuals.py CROP_FILL)
DEFAULT_TEASERS = (ROOT / "results" / "MUST-05" / "dinov2_vitb14_reg" / "teaser.npz",   # git-ignored (results/**/*.npz):
                   ROOT / "results" / "paper_next" / "data" / "fig1_teaser.npz")       # upload one of them to the VM
DEFAULT_ROWS = "1:1,1:2,3:1,6:1"         # (block, 1-based pair rank): scripts/make_paper_fig_twins.py ROWS

# ---------------------------------------------------------------------------------------------- layout only (no data)
FIG_W = 6.0
LEFT, RIGHT, TOP, BOTTOM = 0.55, 0.42, 0.20, 0.02     # inches (as scripts/make_paper_fig_twins_full.py)
PATCH_GAP, SIDE_GAP, ROW_GAP, BLOCK_GAP = 0.025, 0.22, 0.025, 0.10
FS, FS_TITLE = 8, 8.5


def _early_gpu_frac(argv, default: float = 0.15) -> float:
    for i, s in enumerate(argv):
        if s == "--gpu-frac" and i + 1 < len(argv):
            return float(argv[i + 1])
        if s.startswith("--gpu-frac="):
            return float(s.split("=", 1)[1])
    return default


# ---------------------------------------------------------------------------------------------- small helpers
def git_head() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
                              timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        return head + (" (uncommitted changes)" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(p).as_posix()


def sha256_of(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


# ---------------------------------------------------------------------------------------------- geometry
def cache_geometry(w: int, h: int, resize: int = 256, crop: int = 224) -> dict:
    """Where the centre crop of the cache lies in the original image: torchvision Resize(resize) (shorter side, the
    longer side int(resize * long / short)) then CenterCrop(crop) (offsets int(round(...))), exactly the arithmetic
    of mirror/final/data/cache.py resize_centre_crop. Returns the resized size, the crop offsets and the scale from
    resized to original pixels."""
    short, long = (w, h) if w <= h else (h, w)
    new_short, new_long = resize, int(resize * long / short)
    new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    top = int(round((new_h - crop) / 2.0))
    left = int(round((new_w - crop) / 2.0))
    return {"w": int(w), "h": int(h), "new_w": new_w, "new_h": new_h, "left": left, "top": top,
            "sx": w / new_w, "sy": h / new_h}


def resize_centre_crop(img, resize: int, crop: int):
    """The cache preprocessing on a PIL image (PIL bicubic resize of the shorter side, centre crop)."""
    from PIL import Image

    g = cache_geometry(img.size[0], img.size[1], resize, crop)
    if (g["new_w"], g["new_h"]) != img.size:
        img = img.resize((g["new_w"], g["new_h"]), Image.BICUBIC)
    return img.crop((g["left"], g["top"], g["left"] + crop, g["top"] + crop))


def token_box(pos: int, grid_w: int, patch: int, size: int) -> tuple:
    """(x0, y0, x1, y1) in cache pixels of the size x size window centred on patch token `pos` (row-major), the
    window of mirror/twins.py token_crop."""
    r, c = divmod(int(pos), int(grid_w))
    y0 = r * patch + (patch - size) // 2
    x0 = c * patch + (patch - size) // 2
    return x0, y0, x0 + size, y0 + size


def box_in_original(box: tuple, g: dict) -> tuple:
    """The cache-pixel box mapped to original-image pixels (edges map to edges)."""
    x0, y0, x1, y1 = box
    return ((x0 + g["left"]) * g["sx"], (y0 + g["top"]) * g["sy"], (x1 + g["left"]) * g["sx"],
            (y1 + g["top"]) * g["sy"])


def crop_original(img, box_o: tuple, tile_px: int, fill: int = FILL) -> np.ndarray:
    """The region box_o (floats, original pixels) resampled once to tile_px x tile_px (Lanczos); parts outside the
    image are `fill`."""
    from PIL import Image, ImageOps

    w, h = img.size
    m = int(math.ceil(max(box_o[2] - box_o[0], box_o[3] - box_o[1]))) + 2
    need = box_o[0] < 0 or box_o[1] < 0 or box_o[2] > w or box_o[3] > h
    src, off = (ImageOps.expand(img, border=m, fill=(fill,) * 3), m) if need else (img, 0)
    out = src.resize((tile_px, tile_px), Image.LANCZOS,
                     box=(box_o[0] + off, box_o[1] + off, box_o[2] + off, box_o[3] + off))
    return np.asarray(out.convert("RGB"), dtype=np.uint8)


def crop_cache(img: np.ndarray, box: tuple, fill: int = FILL) -> np.ndarray:
    """The box from a cache image [H, W, 3] (mirror/twins.py token_crop)."""
    x0, y0, x1, y1 = box
    S = x1 - x0
    out = np.full((S, S, 3), fill, dtype=np.uint8)
    H, W = img.shape[:2]
    ys, xs, ye, xe = max(y0, 0), max(x0, 0), min(y1, H), min(x1, W)
    if ye > ys and xe > xs:
        out[ys - y0:ye - y0, xs - x0:xe - x0] = img[ys:ye, xs:xe]
    return out


def enlarge(tile: np.ndarray, size: int) -> np.ndarray:
    """Pixel repetition to size x size (exact when size is a multiple of the tile's side)."""
    S = tile.shape[0]
    if S == size:
        return tile
    idx = (np.arange(size) * S) // size
    return tile[idx][:, idx]


# ---------------------------------------------------------------------------------------------- originals
class ParquetOriginals:
    """Original JPEG bytes of train224 cache rows: cache/train_meta.json (shards_used, repo, revision) and
    cache/train_index.npz src224 (index into shards_used, row in the shard). Shards come from the local Hugging
    Face cache, or are downloaded under $HF_HOME when allowed (VM only)."""

    def __init__(self, cache_dir: Path, allow_download: bool, notes: list):
        meta = json.loads((Path(cache_dir) / "train_meta.json").read_text(encoding="utf-8"))
        with np.load(Path(cache_dir) / "train_index.npz") as z:
            self.src = np.asarray(z["src224"], dtype=np.int64)
        self.shards = list(meta["shards_used"])
        self.repo, self.revision = meta.get("repo", "ILSVRC/imagenet-1k"), meta.get("revision")
        self.allow_download, self.notes = allow_download, notes
        self.source = f"{self.repo}@{self.revision} parquet shards (cache/train_meta.json, train_index.npz src224)"
        if not os.environ.get("HF_HOME"):
            notes.append("HF_HOME is not set: shards are looked up in the default Hugging Face cache")

    def _local(self, name: str) -> tuple:
        """(local path or None, whether it was downloaded now)."""
        from huggingface_hub import hf_hub_download, try_to_load_from_cache

        p = try_to_load_from_cache(self.repo, name, revision=self.revision, repo_type="dataset")
        if isinstance(p, str) and Path(p).exists():
            return p, False
        if not self.allow_download:
            return None, False
        return hf_hub_download(self.repo, name, repo_type="dataset", revision=self.revision), True

    def fetch(self, rows) -> dict:
        import pyarrow.parquet as pq

        by_shard: dict = {}
        for r in sorted({int(x) for x in rows}):
            si, rr = (int(v) for v in self.src[r])
            by_shard.setdefault(si, []).append((r, rr))
        out, downloaded = {}, 0
        print(f"originals: {sum(len(v) for v in by_shard.values())} images in {len(by_shard)} shards")
        for si, items in sorted(by_shard.items()):
            name = self.shards[si]
            try:
                path, fresh = self._local(name)
                downloaded += int(fresh)
            except Exception as e:  # noqa: BLE001
                self.notes.append(f"shard {name}: {type(e).__name__}: {e}")
                path = None
            if path is None:
                self.notes.append(f"shard {name} not available locally (--no-download): {len(items)} tiles from "
                                  "the cache")
                continue
            col = pq.read_table(path, columns=["image"]).column("image")
            for r, rr in items:
                out[r] = col[rr].as_py()["bytes"]
        print(f"originals: {len(out)} read; {downloaded} shards downloaded now, the others were in the local cache")
        return out


class MemoryOriginals:
    """Original image bytes held in memory (the dry run)."""

    def __init__(self, data: dict):
        self.data, self.source = data, "synthetic originals in memory (dry run)"

    def fetch(self, rows) -> dict:
        return {int(r): self.data[int(r)] for r in rows if int(r) in self.data}


# ---------------------------------------------------------------------------------------------- the earlier teaser
def load_teaser(path: Path) -> dict:
    with np.load(path) as z:
        T = {k: z[k] for k in z.files}
    need = ["vis_rows", "blocks", "crop", "patch", "grid_w"]
    for b in T.get("blocks", []):
        need += [f"b{int(b)}_{k}" for k in ("pairs", "fit_S", "top_img", "top_pos", "top_val")]
    miss = [k for k in need if k not in T]
    if miss:
        raise SystemExit(f"{path}: missing {miss} (expected a scripts/must05_twins.py teaser.npz)")
    return T


def selection(T: dict, blocks: list, n_pairs: int, n_patches: int, tops: dict | None = None) -> list:
    """One entry per tile: block, pair rank (1-based), side (i or j), patch rank (1-based), neuron, twin, image
    index into vis_rows, cache row, token and stored activation. `tops` replaces the stored top lists."""
    out = []
    for b in blocks:
        pairs, S = T[f"b{b}_pairs"], T[f"b{b}_fit_S"]
        top = tops[b] if tops is not None else {k: T[f"b{b}_top_{k}"] for k in ("img", "pos", "val")}
        for p in range(min(n_pairs, len(pairs))):
            for side in range(2):
                col = 2 * p + side
                for r in range(min(n_patches, top["img"].shape[1])):
                    img = int(top["img"][col, r])
                    if img < 0:
                        continue
                    out.append({"block": int(b), "pair": p + 1, "side": "ij"[side], "rank": r + 1,
                                "neuron": int(pairs[p, side]), "twin": int(pairs[p, 1 - side]), "img": img,
                                "row": int(T["vis_rows"][img]), "pos": int(top["pos"][col, r]),
                                "val": float(top["val"][col, r]), "fit_S": float(S[p])})
    return out


# ---------------------------------------------------------------------------------------------- model pass
def load_f7(gpu_frac: float | None):
    """scripts/f7_visuals.py as a module (the tracker and the paired pass of the earlier teaser). It sets its own GPU
    cap at import, so ours is restored afterwards (as scripts/must05_twins.py does)."""
    import torch

    spec = importlib.util.spec_from_file_location("f7_visuals_for_fp_teaser", ROOT / "scripts" / "f7_visuals.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if gpu_frac is not None and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gpu_frac)
    return mod


def track(f7, adapter, images, rows: np.ndarray, T: dict, blocks: list, k: int, device, amp, batch: int) -> dict:
    """Top-k patches (at most one per image) of every tracked neuron of the teaser's pairs over `rows`, with
    f7_visuals.BlockTracker and collect (the paired pass of MUST-05). Image indices refer to `rows`."""
    from mirror.data import make_loader

    trackers = {}
    for b in blocks:
        pairs = np.asarray(T[f"b{b}_pairs"], dtype=np.int64)
        h = adapter.mlp_hidden
        perm = np.arange(h)
        perm[pairs[:, 0]], perm[pairs[:, 1]] = pairs[:, 1], pairs[:, 0]
        rho = f7.random_partners(perm, seed=1000 + b)                 # only the "rand" extra depends on it
        trackers[b - 1] = f7.BlockTracker(b, perm, rho, pairs.reshape(-1), int(k), device)
    loader = make_loader(images, np.zeros(len(images), dtype=np.int64), rows, batch, 0, False)
    seen = f7.collect(adapter, loader, device, trackers, amp=amp, desc="teaser pass")
    if seen != len(rows):
        raise RuntimeError(f"the pass saw {seen} images, expected {len(rows)}")
    return {b: trackers[b - 1].top.result() for b in blocks}


def verify(sel: list, f7, adapter, images, T: dict, blocks: list, device, amp, batch: int) -> list:
    """Each tile's neuron on its image: the arg-max token and the activation, recomputed (all images of the tiles
    ranked, k = their number), against the stored token and value."""
    rows = np.array(sorted({e["row"] for e in sel}), dtype=np.int64)
    res = track(f7, adapter, images, rows, T, blocks, len(rows), device, amp, batch)
    where = {int(r): i for i, r in enumerate(rows)}
    out = []
    for e in sel:
        col = 2 * (e["pair"] - 1) + (0 if e["side"] == "i" else 1)
        top = res[e["block"]]
        hit = np.flatnonzero(top["img"][col] == where[e["row"]])
        if len(hit) == 0:
            out.append({**e, "val_recomputed": None, "pos_recomputed": None, "pos_match": False, "rel_diff": None})
            continue
        v, pp = float(top["val"][col, hit[0]]), int(top["pos"][col, hit[0]])
        out.append({**e, "val_recomputed": v, "pos_recomputed": pp, "pos_match": pp == e["pos"],
                    "rel_diff": abs(v - e["val"]) / max(abs(e["val"]), 1e-12)})
    return out


# ---------------------------------------------------------------------------------------------- tiles
def make_tiles(sel: list, T: dict, cache_images, originals, tile_px: int, resize: int, crop: int,
               notes: list) -> tuple:
    S, patch, grid_w = int(T["crop"]), int(T["patch"]), int(T["grid_w"])
    raw = originals.fetch([e["row"] for e in sel]) if originals is not None else {}
    tiles, info = [], []
    from PIL import Image

    decoded: dict = {}
    for e in sel:
        box = token_box(e["pos"], grid_w, patch, S)
        rec = {"box_cache": box}
        data = raw.get(e["row"])
        if data is not None:
            if e["row"] not in decoded:
                decoded[e["row"]] = Image.open(io.BytesIO(data)).convert("RGB")
            img = decoded[e["row"]]
            g = cache_geometry(img.size[0], img.size[1], resize, crop)
            bo = box_in_original(box, g)
            tile = crop_original(img, bo, tile_px)
            rec.update(source="original", orig_w=img.size[0], orig_h=img.size[1], box_original=bo,
                       native_px=round(bo[2] - bo[0], 2))
        else:
            tile = enlarge(crop_cache(np.asarray(cache_images[e["row"]]), box), tile_px)
            rec.update(source="cache", orig_w=None, orig_h=None, box_original=None, native_px=S)
        tiles.append(tile)
        info.append(rec)
    n_cache = sum(r["source"] == "cache" for r in info)
    if originals is not None and n_cache:
        notes.append(f"{n_cache} of {len(info)} tiles fell back to the cache pixels (original not available)")
    return np.stack(tiles) if tiles else np.zeros((0, tile_px, tile_px, 3), np.uint8), info


# ---------------------------------------------------------------------------------------------- figure
def parse_rows(spec: str) -> list:
    out = []
    for part in [x.strip() for x in str(spec).split(",") if x.strip()]:
        b, _, p = part.partition(":")
        out.append((int(b), int(p or 1)))
    return out


def draw(sel: list, tiles: np.ndarray, rows: list, n_patch: int, out: Path, stem: str, synthetic: bool) -> tuple:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from paper_style import C, apply

    by = {}
    for e, t in zip(sel, tiles):
        by[(e["block"], e["pair"], e["side"], e["rank"])] = (e, t)
    use = [(b, p) for b, p in rows if any(k[0] == b and k[1] == p for k in by)]
    if not use:
        raise SystemExit(f"none of the rows {rows} has tiles")
    k = n_patch
    patch = (FIG_W - LEFT - RIGHT - SIDE_GAP - (2 * k - 2) * PATCH_GAP) / (2 * k)
    blocks = [b for b, _ in use]
    n_blocks = len(dict.fromkeys(blocks))
    grid_h = len(use) * patch + (len(use) - n_blocks) * ROW_GAP + (n_blocks - 1) * BLOCK_GAP
    fig_h = TOP + grid_h + BOTTOM
    apply()
    plt.rcParams["savefig.bbox"] = "standard"
    fig = plt.figure(figsize=(FIG_W, fig_h))

    def ax_at(x_in, y_top_in):
        return fig.add_axes([x_in / FIG_W, (fig_h - y_top_in - patch) / fig_h, patch / FIG_W, patch / fig_h])

    x_left = [LEFT + c * (patch + PATCH_GAP) for c in range(k)]
    x_twin0 = LEFT + k * patch + (k - 1) * PATCH_GAP + SIDE_GAP
    x_right = [x_twin0 + c * (patch + PATCH_GAP) for c in range(k)]
    x_end = x_right[-1] + patch
    y, prev, span, written = TOP, None, {}, []
    tile_dir = out / f"{stem}_tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    for b, p in use:
        if prev is not None:
            y += BLOCK_GAP + ROW_GAP if b != prev else ROW_GAP
        prev = b
        S_ij = None
        for side, xs in (("i", x_left), ("j", x_right)):
            for c, x in enumerate(xs):
                ax = ax_at(x, y)
                ax.set_axis_off()
                got = by.get((b, p, side, c + 1))
                if got is None:
                    continue
                e, t = got
                S_ij = e["fit_S"]
                ax.imshow(t, interpolation="none")                    # embedded at its own pixel size in the PDF
                name = tile_dir / f"block{b}_pair{p}_{side}_{c + 1}.png"
                Image.fromarray(t).save(name)
                written.append(name)
        if S_ij is not None:
            fig.text((x_end + 0.08) / FIG_W, (fig_h - y - patch / 2) / fig_h, f"{S_ij:.2f}", ha="left",
                     va="center", fontsize=FS, color=C["ink2"])
        lo, hi = span.get(b, (y, y))
        span[b] = (min(lo, y), max(hi, y + patch))
        y += patch
    for b, (lo, hi) in span.items():
        fig.text((LEFT - 0.08) / FIG_W, (fig_h - (lo + hi) / 2) / fig_h, f"block {b}", ha="right", va="center",
                 fontsize=FS, color=C["ink"])
    y_title = (fig_h - TOP + 0.05) / fig_h
    w_side = k * patch + (k - 1) * PATCH_GAP
    fig.text((LEFT + w_side / 2) / FIG_W, y_title, "neuron $i$", ha="center", va="bottom", fontsize=FS_TITLE,
             color=C["ink"])
    fig.text((x_twin0 + w_side / 2) / FIG_W, y_title, "its mirror twin $j$", ha="center", va="bottom",
             fontsize=FS_TITLE, color=C["ink"])
    fig.text((x_end + 0.08) / FIG_W, y_title, "corr.", ha="left", va="bottom", fontsize=FS, color=C["ink2"])
    if synthetic:
        fig.text(0.005, 0.005, "synthetic data (dry run)", ha="left", va="bottom", fontsize=6, color=C["grey"])
    out.mkdir(parents=True, exist_ok=True)
    pdf, png = out / f"{stem}.pdf", out / f"{stem}.png"
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png, written, use


# ---------------------------------------------------------------------------------------------- dry run
def make_dry_inputs(work: Path, seed: int = 0, n_images: int = 24, img_size: int = 56, resize: int = 64) -> dict:
    """Synthetic originals of random sizes, their cache images (the cache preprocessing at resize/crop
    `resize`/`img_size`), a tiny random ViT and an earlier teaser in the MUST-05 format built with the same tracker
    on the cache images (values from a seeded generator; they mean nothing)."""
    import torch
    from PIL import Image

    from mirror.controls import tiny_adapter

    g = np.random.default_rng(seed)
    originals, cache = {}, np.zeros((n_images, img_size, img_size, 3), dtype=np.uint8)
    for i in range(n_images):
        w, h = int(g.integers(70, 112)), int(g.integers(60, 96))
        yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
        base = np.zeros((h, w, 3))
        for _ in range(3):
            fy, fx, ph = g.uniform(0.5, 4), g.uniform(0.5, 4), g.uniform(0, 2 * np.pi)
            base += g.uniform(0.2, 0.6, 3) * np.sin(2 * np.pi * (fy * yy + fx * xx) + ph)[..., None]
        arr = np.clip(128 + 70 * base + 8 * g.normal(size=(h, w, 3)), 0, 255).astype(np.uint8)
        im = Image.fromarray(arr)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        originals[i] = buf.getvalue()
        cache[i] = np.asarray(resize_centre_crop(im, resize, img_size), dtype=np.uint8)
    work = Path(work)
    (work / "cache").mkdir(parents=True, exist_ok=True)
    np.save(work / "cache" / "train224_uint8.npy", cache)
    adapter = tiny_adapter(img_size=img_size, embed_dim=32, depth=3, num_heads=2, reg_tokens=2, seed=seed,
                           dtype=torch.float32)
    f7 = load_f7(None)
    h = adapter.mlp_hidden
    perm = g.permutation(h)
    blocks = [1, 2]
    T = {"vis_rows": np.arange(n_images, dtype=np.int64), "blocks": np.array(blocks), "crop": np.array(42),
         "patch": np.array(14), "grid_w": np.array(img_size // 14), "rank_by": np.array("weight")}
    for b in blocks:
        T[f"b{b}_pairs"] = perm[4 * b: 4 * b + 4].reshape(2, 2).astype(np.int64)
        T[f"b{b}_fit_S"] = np.sort(g.uniform(0.6, 0.95, 2))[::-1]
        T[f"b{b}_fit_w"] = T[f"b{b}_fit_S"] - 0.1
    tops = track(f7, adapter, cache, T["vis_rows"], T, blocks, 4, "cpu", None, 8)
    for b in blocks:
        for key in ("val", "img", "pos", "twin", "rand"):
            T[f"b{b}_top_{key}"] = tops[b][key]
    tpath = work / "teaser.npz"
    np.savez(tpath, **T)
    return {"teaser": tpath, "cache": work / "cache", "originals": MemoryOriginals(originals), "adapter": adapter,
            "f7": f7, "resize": resize, "crop": img_size}


# ---------------------------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teaser", default=None,
                   help="the earlier teaser (default: the first of results/MUST-05/dinov2_vitb14_reg/teaser.npz and "
                        "its copy results/paper_next/data/fig1_teaser.npz that exists)")
    p.add_argument("--model", default="dinov2_vitb14_reg")
    p.add_argument("--cache-dir", default=str(ROOT / "cache"), help="train224_uint8.npy, train_index.npz, train_meta.json")
    p.add_argument("--source", choices=("parquet", "cache"), default="parquet",
                   help="tiles from the original images (parquet shards) or from the cache pixels")
    p.add_argument("--no-download", action="store_true", help="use only shards already in the local HF cache")
    p.add_argument("--select", choices=("stored", "recompute"), default="stored",
                   help="tiles of the stored choice (default) or of top patches re-found over vis_rows")
    p.add_argument("--no-model", action="store_true", help="skip the model pass (no check of the stored choice)")
    p.add_argument("--blocks", default=None, help="1-based blocks (default: every block of the teaser file)")
    p.add_argument("--pairs", type=int, default=4, help="pairs per block in tiles.npz")
    p.add_argument("--patches", type=int, default=6, help="patches per neuron in tiles.npz")
    p.add_argument("--rows", default=DEFAULT_ROWS, help="figure rows, block:pair rank (1-based)")
    p.add_argument("--figure-patches", type=int, default=6, help="patches per neuron in the figure")
    p.add_argument("--tile-px", type=int, default=168, help="tile side in pixels (default 4 x 42)")
    p.add_argument("--resize", type=int, default=256, help="the cache's resize of the shorter side")
    p.add_argument("--crop", type=int, default=224, help="the cache's centre crop")
    p.add_argument("--batch-size", type=int, default=32, help="images per batch of the model pass (MUST-05: 32)")
    p.add_argument("--amp", choices=("bf16", "none"), default="bf16", help="autocast of the model pass on CUDA")
    p.add_argument("--device", default=None)
    p.add_argument("--gpu-frac", type=float, default=0.15)
    p.add_argument("--stem", default=STEM)
    p.add_argument("--from-tiles", default=None, help="re-draw from a tiles.npz written earlier")
    p.add_argument("--out", default=None, help="figure folder (default paper/figures; dry run: <work>/figures)")
    p.add_argument("--results", default=str(ROOT / "results"), help="results root (writes <results>/A14/teaser)")
    p.add_argument("--dry-run", action="store_true", help="CPU, tiny random ViT and random images")
    p.add_argument("--work", default=None, help="folder for the dry run's synthetic inputs (default: a new temp dir)")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    a = parse_args(argv_list)
    t0 = time.time()
    notes: list = []
    out = Path(a.out) if a.out else ROOT / "paper" / "figures"
    results = Path(a.results)

    if a.from_tiles:
        with np.load(a.from_tiles, allow_pickle=False) as z:
            tiles = z["tiles"]
            sel = json.loads(str(z["selection_json"]))
        pdf, png, written, used = draw(sel, tiles, parse_rows(a.rows), a.figure_patches, out, a.stem, False)
        print(f"re-drawn from {a.from_tiles}: {rel(pdf)}, {rel(png)}, {len(written)} tile PNGs")
        return {"pdf": pdf, "png": png, "tiles": written, "notes": notes}

    dry = None
    if a.dry_run:
        os.environ.setdefault("MIRROR_FINAL_ALLOW_NO_PREREG", "1")
        work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="fp_fig_teaser_"))
        dry = make_dry_inputs(work)
        a.teaser, a.cache_dir, a.resize, a.crop = str(dry["teaser"]), str(dry["cache"]), dry["resize"], dry["crop"]
        out = Path(a.out) if a.out else work / "figures"
        results = work / "results"
        a.rows, a.pairs, a.patches, a.figure_patches = "1:1,1:2,2:1", 2, 4, 4
        print(f"dry run: synthetic inputs in {work}")
    if a.teaser is None:
        found = [p for p in DEFAULT_TEASERS if p.exists()]
        if not found:
            raise SystemExit("the earlier teaser is missing: upload results/MUST-05/dinov2_vitb14_reg/teaser.npz (git-ignored, "
                             "about 3 MB, from the laptop) or pass --teaser")
        a.teaser = str(found[0])
    T = load_teaser(Path(a.teaser))
    blocks = [int(b) for b in T["blocks"]] if a.blocks is None else [int(x) for x in a.blocks.split(",")]
    cache_path = Path(a.cache_dir) / "train224_uint8.npy"
    if not cache_path.exists():
        raise SystemExit(f"{cache_path} not found (the VM cache; on the laptop use --dry-run or --from-tiles)")
    cache_images = np.load(cache_path, mmap_mode="r")
    if cache_images.shape[1] != a.crop:
        raise SystemExit(f"cache images are {cache_images.shape[1]} px, --crop is {a.crop}")

    # model pass: check the stored choice, or re-find the top patches
    use_model = not a.no_model
    f7 = adapter = None
    device, amp = "cpu", None
    tops, agreement, checks = None, None, []
    if use_model:
        import torch

        if dry is not None:
            adapter, f7 = dry["adapter"], dry["f7"]
        else:
            if not torch.cuda.is_available():
                raise SystemExit("the model pass needs the VM GPU (or pass --no-model, or --dry-run on the laptop)")
            device = a.device or "cuda"
            if torch.device(device).type == "cuda":
                torch.cuda.set_per_process_memory_fraction(float(a.gpu_frac))
            from mirror.final.prereg import require_prereg

            require_prereg()
            from mirror.adapters_timm import build_model_adapter

            adapter = build_model_adapter(a.model, device=device)
            adapter.model.float().eval()
            f7 = load_f7(float(a.gpu_frac))
            amp = a.amp if (torch.device(device).type == "cuda" and a.amp != "none") else None
        if a.select == "recompute":
            k = int(T[f"b{blocks[0]}_top_img"].shape[1])
            tops = track(f7, adapter, cache_images, np.asarray(T["vis_rows"], dtype=np.int64), T, blocks, k, device,
                         amp, a.batch_size)
            same = tot = 0
            for b in blocks:
                st_img, st_pos = T[f"b{b}_top_img"], T[f"b{b}_top_pos"]
                ok = (tops[b]["img"] == st_img) & (tops[b]["pos"] == st_pos) & (st_img >= 0)
                same += int(ok.sum())
                tot += int((st_img >= 0).sum())
            agreement = same / tot if tot else float("nan")
            print(f"recomputed top patches: {same} of {tot} (image, token) entries equal the stored choice")
    elif a.select == "recompute":
        raise SystemExit("--select recompute needs the model (drop --no-model)")
    sel = selection(T, blocks, a.pairs, a.patches, tops)
    if use_model:
        checks = verify(sel, f7, adapter, cache_images, T, blocks, device, amp, a.batch_size)
        n_pos = sum(bool(c["pos_match"]) for c in checks)
        diffs = [c["rel_diff"] for c in checks if c["rel_diff"] is not None]
        print(f"model check: {n_pos} of {len(checks)} tiles have the neuron's largest activation at the drawn token; "
              f"largest relative difference to the stored value {max(diffs) if diffs else float('nan'):.2e}")
        if n_pos < len(checks):
            notes.append(f"{len(checks) - n_pos} tiles: the recomputed arg-max token differs from the stored one "
                         "(near-ties at bf16 are possible; see verify.csv)")

    # tiles
    originals = None
    if a.source == "parquet":
        if dry is not None:
            originals = dry["originals"]
        else:
            try:
                originals = ParquetOriginals(Path(a.cache_dir), not a.no_download, notes)
            except (OSError, KeyError, ValueError) as e:
                notes.append(f"originals unavailable ({type(e).__name__}: {e}): every tile from the cache pixels")
    tiles, info = make_tiles(sel, T, cache_images, originals, a.tile_px, a.resize, a.crop, notes)
    pdf, png, written, used = draw(sel, tiles, parse_rows(a.rows), a.figure_patches, out, a.stem, dry is not None)

    # data products
    import pandas as pd

    data_dir = results / "A14" / "teaser"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows_tab = [{**e, **{k: (json.dumps(v) if isinstance(v, tuple) else v) for k, v in r.items()}}
                for e, r in zip(sel, info)]
    pd.DataFrame(rows_tab).to_csv(data_dir / "tiles.csv", index=False)
    if checks:
        pd.DataFrame(checks).to_csv(data_dir / "verify.csv", index=False)
    np.savez_compressed(data_dir / "tiles.npz", tiles=tiles, selection_json=np.asarray(json.dumps(sel)),
                        info_json=np.asarray(json.dumps(info, default=list)))
    natives = [r["native_px"] for r in info if r["source"] == "original"]
    metrics = {"teaser": rel(Path(a.teaser)), "teaser_sha256": sha256_of(Path(a.teaser)), "blocks": blocks,
               "select": a.select, "source": a.source,
               "originals": getattr(originals, "source", None), "n_tiles": len(sel),
               "n_from_original": len(natives), "native_px_min": min(natives) if natives else None,
               "native_px_median": float(np.median(natives)) if natives else None, "tile_px": a.tile_px,
               "model_check": None if not checks else {
                   "n": len(checks), "pos_match": sum(bool(c["pos_match"]) for c in checks),
                   "max_rel_diff": max((c["rel_diff"] for c in checks if c["rel_diff"] is not None), default=None)},
               "recompute_agreement": agreement, "figure_rows": used, "notes": notes, "dry_run": dry is not None}
    (data_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str), encoding="utf-8")
    try:
        from mirror.final.prereg import write_final_meta

        write_final_meta(data_dir, experiment="A14-teaser", config=vars(a), seeds={"note": "no random step"},
                         models={a.model: "tiny random ViT (dry run)" if dry is not None else a.model},
                         device=device, wall_seconds=time.time() - t0, extra={"notes": notes})
    except Exception as e:  # noqa: BLE001  (the figure does not depend on it)
        notes.append(f"meta.json not written ({type(e).__name__}: {e})")

    print(f"git: {git_head()}")
    print(f"teaser: {rel(Path(a.teaser))}; blocks {blocks}; select {a.select}; source {a.source} "
          f"({getattr(originals, 'source', 'cache pixels')})")
    print(f"tiles: {len(sel)} ({len(natives)} from originals; native window "
          f"{metrics['native_px_min']} to {max(natives) if natives else None} px, median {metrics['native_px_median']};"
          f" drawn at {a.tile_px} px)")
    for (b, p) in used:
        e = next(x for x in sel if x["block"] == b and x["pair"] == p)
        print(f"  figure row: block {b}, pair {p}: neurons {e['neuron'] if e['side'] == 'i' else e['twin']} and "
              f"{e['twin'] if e['side'] == 'i' else e['neuron']}, twin correlation {e['fit_S']:.3f}")
    for n in notes:
        print(f"note: {n}")
    print(f"\nwrote {rel(pdf)}, {rel(png)}, {len(written)} tile PNGs in {rel(out / (a.stem + '_tiles'))}; "
          f"data in {rel(data_dir)}")
    return {"pdf": pdf, "png": png, "tiles": written, "notes": notes, "metrics": metrics, "selection": sel,
            "info": info, "checks": checks, "data_dir": data_dir}


if __name__ == "__main__":
    if "--dry-run" not in sys.argv[1:] and "--from-tiles" not in " ".join(sys.argv[1:]):
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(_early_gpu_frac(sys.argv[1:]))   # shared GPU: cap first
    main()
