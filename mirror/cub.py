"""CUB-200-2011 for CORE-06 (facing direction from beak and tail) and a small image-cache builder shared with
scripts/core07_shift.py.

Annotation files (configs/data/cub_parts.yaml, read from the dataset README; paths relative to CUB_200_2011/):
    images.txt               <image_id> <image_name>
    train_test_split.txt     <image_id> <is_training_image>          1 = train, 0 = test
    image_class_labels.txt   <image_id> <class_id>                   class ids 1..200 (returned 0-based)
    bounding_boxes.txt       <image_id> <x> <y> <width> <height>     pixels of the original image
    parts/part_locs.txt      <image_id> <part_id> <x> <y> <visible>  part ids 1..15; beak 2, tail 14
Parts that are not visible are written as x = y = 0 with visible = 0, so the filter is on `visible`, never on the
coordinates.

Facing direction (PROJECT_NEXT_PHASE.md CORE-06; configs/data/cub_parts.yaml core06_facing_rule):
    sign = sign(x_beak - x_tail), only when both parts have visible == 1 and |x_beak - x_tail| > 0.1 * box width;
    otherwise the image has no facing label (sign 0).
    Convention (a choice, not from a source): sign +1 means the beak is to the right of the tail in the image (the
    bird faces right); as a probe class, 1 = faces right, 0 = faces left.
A horizontal mirror maps x to W - x (or W - 1 - x, depending on the pixel origin): the sign flips and |dx| is
unchanged, so the 10% test gives the same answer for an image and its mirror. Resizing and centre cropping are
monotone in x, so the label of the original image is the label of the preprocessed image.

Mirror-augmented test set (CORE-06): every test image with a facing label, plus its mirror with the facing label
flipped (species unchanged). A feature that is exactly mirror-invariant gives the same prediction for an image and
its mirror, so exactly one of the two is right: accuracy exactly 50% by construction (P6.3).

Images: resize 256 (bicubic) and centre crop 224, as the ImageNet caches (mirror.data.preprocess_pil), stored as
uint8 [N, S, S, 3] in images.txt order. `decode_to_cache` writes any list of image sources (files or zip members)
the same way; scripts/core07_shift.py uses it for ImageNet-V2 and ImageNet-Sketch.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np

BEAK_ID = 2
TAIL_ID = 14
MIN_FRAC = 0.1
N_PARTS = 15
LEFT_RIGHT_PAIRS = ((7, 11), (8, 12), (9, 13))      # left/right eye, leg, wing (swap under a mirror; unused here)
FACING_CONVENTION = ("sign = sign(x_beak - x_tail); +1 = beak right of tail (bird faces right); probe class 1 = "
                     "faces right, 0 = faces left; images without a label (a part not visible, or parts closer than "
                     "10% of the box width) are left out of every facing probe")

__all__ = ["BEAK_ID", "TAIL_ID", "MIN_FRAC", "FACING_CONVENTION", "load_annotations", "facing_sign",
           "facing_class", "mirror_x", "mirror_augmented", "decode_to_cache", "build_cub_cache", "load_cub_cache",
           "write_synthetic_cub", "file_sha256"]


# ----------------------------------------------------------------------------------------------
# Annotations
# ----------------------------------------------------------------------------------------------
def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _id_table(path: Path, n_values: int, dtype=float) -> dict:
    """{image_id: tuple of n_values} from a whitespace table whose first column is the image id."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 1 + n_values:
            raise ValueError(f"{path}: expected {1 + n_values} columns, got {len(parts)} in {line!r}")
        out[int(parts[0])] = tuple(dtype(p) for p in parts[1:])
    return out


def load_annotations(root) -> dict:
    """Every annotation CORE-06 needs, as arrays in images.txt order (row i = the i-th line of images.txt).

    Returns image_id [N] int64, name [N] (relative paths under images/), is_train [N] bool, label [N] int64 (0-based
    class), bbox [N, 4] float64 (x, y, w, h), parts [N, 15, 3] float64 (x, y, visible; part id p at index p - 1),
    files (sha256 of each annotation file)."""
    root = Path(root)
    ids, names = [], []
    for line in (root / "images.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        i, name = line.strip().split(" ", 1)
        ids.append(int(i))
        names.append(name.strip())
    ids = np.asarray(ids, dtype=np.int64)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"{root / 'images.txt'}: duplicate image ids")
    split = _id_table(root / "train_test_split.txt", 1, int)
    cls = _id_table(root / "image_class_labels.txt", 1, int)
    box = _id_table(root / "bounding_boxes.txt", 4, float)
    missing = [int(i) for i in ids if int(i) not in split or int(i) not in cls or int(i) not in box]
    if missing:
        raise ValueError(f"{len(missing)} image ids lack a split, class or box, e.g. {missing[:5]}")
    pos = {int(i): k for k, i in enumerate(ids)}
    parts = np.zeros((len(ids), N_PARTS, 3), dtype=np.float64)
    seen = np.zeros((len(ids), N_PARTS), dtype=bool)
    raw = np.loadtxt(root / "parts" / "part_locs.txt", dtype=np.float64, ndmin=2)
    if raw.shape[1] != 5:
        raise ValueError(f"part_locs.txt: expected 5 columns, got {raw.shape[1]}")
    for img_id, pid, x, y, vis in raw:
        k, p = pos[int(img_id)], int(pid) - 1
        if not 0 <= p < N_PARTS:
            raise ValueError(f"part id {int(pid)} outside 1..{N_PARTS}")
        parts[k, p] = (x, y, vis)
        seen[k, p] = True
    if not seen.all():
        raise ValueError(f"part_locs.txt misses {int((~seen).sum())} (image, part) entries")
    files = {f: file_sha256(root / f) for f in ("images.txt", "train_test_split.txt", "image_class_labels.txt",
                                               "bounding_boxes.txt", "parts/part_locs.txt")}
    if (root / "parts" / "parts.txt").exists():
        files["parts/parts.txt"] = file_sha256(root / "parts" / "parts.txt")
    return {"image_id": ids, "name": names,
            "is_train": np.asarray([split[int(i)][0] == 1 for i in ids]),
            "label": np.asarray([cls[int(i)][0] - 1 for i in ids], dtype=np.int64),
            "bbox": np.asarray([box[int(i)] for i in ids], dtype=np.float64),
            "parts": parts, "files": files}


def check_part_names(root) -> dict:
    """parts/parts.txt against the IDs CORE-06 uses (beak 2, tail 14); {ok, beak, tail} (verify on the VM)."""
    p = Path(root) / "parts" / "parts.txt"
    if not p.exists():
        return {"ok": None, "note": "parts/parts.txt not found"}
    names = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            i, name = line.strip().split(" ", 1)
            names[int(i)] = name.strip()
    return {"ok": names.get(BEAK_ID) == "beak" and names.get(TAIL_ID) == "tail", "beak": names.get(BEAK_ID),
            "tail": names.get(TAIL_ID), "n_parts": len(names)}


def facing_sign(parts: np.ndarray, bbox_w: np.ndarray, beak: int = BEAK_ID, tail: int = TAIL_ID,
                min_frac: float = MIN_FRAC) -> np.ndarray:
    """int8 [N]: +1 (beak right of tail), -1 (beak left of tail) or 0 (no label: beak or tail not visible, or
    |x_beak - x_tail| <= min_frac * box width)."""
    parts = np.asarray(parts, dtype=np.float64)
    w = np.asarray(bbox_w, dtype=np.float64)
    xb, vb = parts[:, beak - 1, 0], parts[:, beak - 1, 2]
    xt, vt = parts[:, tail - 1, 0], parts[:, tail - 1, 2]
    dx = xb - xt
    ok = (vb == 1) & (vt == 1) & (np.abs(dx) > min_frac * w)
    return np.where(ok, np.sign(dx), 0).astype(np.int8)


def facing_class(sign: np.ndarray) -> np.ndarray:
    """Probe class of a facing sign: 1 = faces right (+1), 0 = faces left (-1); -1 for images without a label."""
    s = np.asarray(sign)
    return np.where(s > 0, 1, np.where(s < 0, 0, -1)).astype(np.int64)


def mirror_x(x, width, pixel_centres: bool = True):
    """x after a horizontal mirror of an image `width` pixels wide: W - 1 - x for pixel-centre coordinates
    (origin at the first pixel's centre), W - x for edge coordinates. Only the sign of x_beak - x_tail matters for
    CORE-06, and both conventions flip it."""
    x = np.asarray(x, dtype=np.float64)
    return (width - 1 - x) if pixel_centres else (width - x)


def mirror_augmented(rows: np.ndarray, facing: np.ndarray, species: np.ndarray | None = None) -> dict:
    """The mirror-augmented test set from test `rows` with facing signs `facing` (rows without a label dropped):
    every image once as it is and once mirrored, the mirrored copy with its facing sign flipped.

    Returns row [2n] (image row), mirrored [2n] bool, facing_sign [2n], facing [2n] (probe class), species [2n]
    (unchanged; -1 if not given), pair [2n] (index of the natural copy's entry)."""
    rows = np.asarray(rows, dtype=np.int64)
    f = np.asarray(facing, dtype=np.int64)
    keep = f != 0
    rows, f = rows[keep], f[keep]
    sp = np.full(len(rows), -1, dtype=np.int64) if species is None else np.asarray(species, dtype=np.int64)[keep]
    n = len(rows)
    sign = np.concatenate([f, -f])
    return {"row": np.concatenate([rows, rows]), "mirrored": np.concatenate([np.zeros(n, bool), np.ones(n, bool)]),
            "facing_sign": sign, "facing": facing_class(sign), "species": np.concatenate([sp, sp]),
            "pair": np.concatenate([np.arange(n), np.arange(n)])}


# ----------------------------------------------------------------------------------------------
# Image cache
# ----------------------------------------------------------------------------------------------
_ZIPS: dict = {}


def _read_source(src) -> bytes:
    """src = ("file", path) or ("zip", zip path, member name); each worker process keeps its zip open."""
    if src[0] == "file":
        with open(src[1], "rb") as fh:
            return fh.read()
    if src[0] == "zip":
        zf = _ZIPS.get(src[1])
        if zf is None:
            zf = _ZIPS[src[1]] = zipfile.ZipFile(src[1])
        return zf.read(src[2])
    raise ValueError(f"unknown source kind {src[0]!r}")


def _decode(args):
    from PIL import Image

    from .data import preprocess_pil

    src, resize, crop = args
    with Image.open(io.BytesIO(_read_source(src))) as img:
        return preprocess_pil(img, resize, crop)


def decode_to_cache(sources: list, out_npy, resize: int = 256, crop: int = 224, workers: int = 8,
                    chunk: int = 512) -> Path:
    """Decode `sources` (see _read_source) with resize (bicubic) + centre crop, in order, into a uint8
    [N, crop, crop, 3] .npy file. Written to <name>.partial.npy and renamed at the end, so a partial file never
    looks finished."""
    out_npy = Path(out_npy)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    partial = out_npy.with_name(out_npy.stem + ".partial.npy")
    arr = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint8, shape=(len(sources), crop, crop, 3))
    pool = None
    if workers > 1 and len(sources) > 1:
        import multiprocessing as mp

        pool = mp.get_context("spawn" if os.name == "nt" else "fork").Pool(workers)
    try:
        for s in range(0, len(sources), chunk):
            jobs = [(src, resize, crop) for src in sources[s:s + chunk]]
            out = pool.map(_decode, jobs, chunksize=8) if pool else [_decode(j) for j in jobs]
            arr[s:s + len(out)] = np.stack(out)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    arr.flush()
    del arr
    os.replace(partial, out_npy)
    return out_npy


def build_cub_cache(root, cache_dir, resize: int = 256, crop: int = 224, workers: int = 8) -> dict:
    """cache_dir/cub{crop}_uint8.npy (images.txt order) and cub_meta.json; skipped when the meta says complete
    for the same annotation files and preprocessing. Returns the meta."""
    root, cache_dir = Path(root), Path(cache_dir)
    npy, meta_path = cache_dir / f"cub{crop}_uint8.npy", cache_dir / "cub_meta.json"
    ann = load_annotations(root)
    want = {"root": str(root), "resize": int(resize), "crop": int(crop), "n_images": len(ann["name"]),
            "annotation_sha256": ann["files"]}
    if meta_path.exists() and npy.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("complete") and all(meta.get(k) == v for k, v in want.items() if k != "root"):
            return meta
    sources = [("file", str(root / "images" / n)) for n in ann["name"]]
    decode_to_cache(sources, npy, resize, crop, workers)
    meta = {**want, "complete": True, "order": "images.txt", "interpolation": "bicubic",
            "transform": f"resize {resize} (shorter side, bicubic), centre crop {crop} (mirror.data.preprocess_pil)"}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_cub_cache(cache_dir, crop: int = 224) -> np.ndarray:
    return np.load(Path(cache_dir) / f"cub{crop}_uint8.npy", mmap_mode="r")


# ----------------------------------------------------------------------------------------------
# Synthetic CUB (tests and --tiny)
# ----------------------------------------------------------------------------------------------
def write_synthetic_cub(root, n_classes: int = 4, per_class: int = 8, size: int = 64, seed: int = 0) -> Path:
    """A tiny CUB_200_2011-shaped folder with random PNG images and annotation files. Half of each class is
    train. The beak and tail are placed so that about a third of the images face right, a third face left and
    the rest have no label (a part not visible, or the parts closer than 10% of the box width). The images carry
    a weak horizontal gradient whose direction follows the facing sign, so a mirror-sensitive feature can read
    it. Returns the root."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "parts").mkdir(parents=True, exist_ok=True)
    lines = {k: [] for k in ("images", "split", "cls", "box", "parts")}
    img_id = 0
    for c in range(n_classes):
        cdir = root / "images" / f"{c + 1:03d}.Synthetic_{c}"
        cdir.mkdir(parents=True, exist_ok=True)
        for k in range(per_class):
            img_id += 1
            name = f"{cdir.name}/img_{img_id:05d}.png"
            kind = img_id % 3                      # 0: faces right, 1: faces left, 2: no label
            ramp = np.linspace(0, 60, size)[None, :, None]
            base = rng.integers(0, 160, size=(size, size, 3)).astype(np.float64) + 20 * c
            img = base + (ramp if kind == 0 else ramp[:, ::-1] if kind == 1 else 0)
            Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(root / "images" / name)
            bx, by, bw, bh = 4.0, 6.0, float(size - 8), float(size - 12)
            xb, xt, vb, vt = (bx + 0.8 * bw, bx + 0.2 * bw, 1, 1) if kind == 0 else \
                             (bx + 0.2 * bw, bx + 0.8 * bw, 1, 1) if kind == 1 else \
                             ((bx + 0.5 * bw, bx + 0.45 * bw, 1, 1) if k % 2 else (0.0, bx + 0.3 * bw, 0, 1))
            lines["images"].append(f"{img_id} {name}")
            lines["split"].append(f"{img_id} {int(k < per_class // 2)}")
            lines["cls"].append(f"{img_id} {c + 1}")
            lines["box"].append(f"{img_id} {bx} {by} {bw} {bh}")
            for p in range(1, N_PARTS + 1):
                if p == BEAK_ID:
                    x, y, v = xb, 20.0, vb
                elif p == TAIL_ID:
                    x, y, v = xt, 30.0, vt
                else:
                    x, y, v = 0.0, 0.0, 0
                lines["parts"].append(f"{img_id} {p} {x:.1f} {y:.1f} {v}")
    (root / "images.txt").write_text("\n".join(lines["images"]) + "\n", encoding="utf-8")
    (root / "train_test_split.txt").write_text("\n".join(lines["split"]) + "\n", encoding="utf-8")
    (root / "image_class_labels.txt").write_text("\n".join(lines["cls"]) + "\n", encoding="utf-8")
    (root / "bounding_boxes.txt").write_text("\n".join(lines["box"]) + "\n", encoding="utf-8")
    (root / "parts" / "part_locs.txt").write_text("\n".join(lines["parts"]) + "\n", encoding="utf-8")
    names = ["back", "beak", "belly", "breast", "crown", "forehead", "left eye", "left leg", "left wing", "nape",
             "right eye", "right leg", "right wing", "tail", "throat"]
    (root / "parts" / "parts.txt").write_text("\n".join(f"{i + 1} {n}" for i, n in enumerate(names)) + "\n",
                                              encoding="utf-8")
    return root
