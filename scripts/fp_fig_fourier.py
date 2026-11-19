#!/usr/bin/env python
"""Lemma 2, illustrated: magnitude spectra of about 1,000 ImageNet training images (CPU only, no model).

    python scripts/fp_fig_fourier.py                                     # VM: cache/train224_uint8.npy, 1 per class
    python scripts/fp_fig_fourier.py --from-maps results/A14/fourier/fourier_maps.npz      # re-draw (laptop)
    .venv/Scripts/python.exe scripts/fp_fig_fourier.py --dry-run          # synthetic images, temporary folder

Lemma 2 (paper/sections/structure.tex): for any real image f, |F(f_LR)| = |F(f_UD)| at every frequency, where f_LR
and f_UD are its left-right and up-down mirrors, because F(f_LR)(kx, ky) = F(f)(-kx, ky), F(f_UD)(kx, ky) =
F(f)(kx, -ky) and |F(f)(-k)| = |F(f)(k)| for a real image. The discrete flip x -> N - 1 - x is the flip x -> -x
followed by a circular shift, which changes only the phase, so the identity holds exactly for the discrete Fourier
transform too. A 90-degree rotation maps one frequency axis onto the other, so it changes the spectrum of any image
whose power differs between the two axes.

Data (training images only; nothing is chosen on validation): the uint8 cache --images (default
<cache-dir>/train224_uint8.npy, ImageNet train, resize 256 and centre crop 224, as every other final-phase cache) and
its index --index (default <cache-dir>/train_index.npz). Rows: --per-class images of every class of the role --role
(default bank, the k-NN bank role: 100 train images per class; one per class gives 1,000 images), drawn with
numpy.random.default_rng(--seed); or --n rows drawn uniformly from the role (or from the whole array without an
index). The rows and their sha256 are saved.

Computation, float64 throughout: luminance Y = 0.299 R + 0.587 G + 0.114 B; the image mean subtracted; four
versions of each image: itself, its left-right mirror, its up-down mirror and its 90-degree rotation
(numpy.rot90, counter-clockwise); each multiplied by a separable Hann window (exactly symmetric, so it is unchanged
by both mirrors and by the rotation, and windowing commutes with all three; --window none skips it); magnitudes
|numpy.fft.fft2|. Per image: the largest |difference| between the left-right and up-down magnitude spectra, and
between the rotated and the original magnitude spectra, both divided by the image's largest magnitude. Over the
images: the mean magnitude spectrum of each version.

Figure (6.0 x 1.92 inches; scripts/paper_style.py): (a) log10 of the mean magnitude spectrum of the originals;
(b) log10 of the ratio of the mean magnitude spectra of the left-right and up-down mirrors; (c) the same for the
90-degree rotation against the original; (b) and (c) share one diverging scale (vermilion below 0, blue above), so
(b) is flat at 0; (d) per image, the largest difference relative to the peak, as two histograms on a log axis. The
zero frequency is masked in the maps (the mean is removed). Writes <out>/fig_fourier.pdf, .png (300 dpi) and
fig_fourier.json; data in <results>/A14/fourier/: fourier_maps.npz (mean spectra, per-image differences, rows),
metrics.json (every reported number) and meta.json (mirror.final.prereg.write_final_meta). Prints every reported
number and a caption that says what each panel shows. The only random step is the seeded choice of rows.

--dry-run: synthetic uint8 images (a few classes, with more structure along one axis than the other) and an index
in the real formats, in a temporary folder (or --work); outputs go there, marked "synthetic data".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import LogLocator, MaxNLocator, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(1, str(ROOT))
from paper_style import C, apply  # noqa: E402

STEM = "fig_fourier"
LUMA = (0.299, 0.587, 0.114)            # ITU-R BT.601 luma weights
VERSIONS = ("orig", "lr", "ud", "r90")
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#EEF4F9", C["blue"], "#00304D"])
DIV = LinearSegmentedColormap.from_list("div", [C["terracotta"], "#F2F2F2", C["blue"]])
BAND = (0.05, 0.45)                     # cycles per pixel: the axis profiles of the anisotropy summary

# ---------------------------------------------------------------------------------------------- layout only (no data)
FIG_W, FIG_H = 6.0, 1.92
MAP_B, MAP_S = 0.58, 1.02               # bottom and side of the maps (inches)
A_L, B_L, C_L = 0.40, 1.72, 3.02        # left edges of the maps
CB = (1.72, 0.30, 2.32, 0.05)           # shared colour bar of (b) and (c): left, bottom, width, height
D_AX = (4.62, MAP_B, 1.30, MAP_S)
FS, FS_TITLE = 8, 8


# ---------------------------------------------------------------------------------------------- helpers
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


def symmetric_hann(n: int) -> np.ndarray:
    """A Hann window that is exactly symmetric in floating point (h[i] == h[n - 1 - i])."""
    h = np.hanning(n)
    return 0.5 * (h + h[::-1])


def window2d(n: int, kind: str) -> np.ndarray:
    if kind == "none":
        return np.ones((n, n))
    h = symmetric_hann(n)
    return np.outer(h, h)


def luminance(imgs: np.ndarray) -> np.ndarray:
    """uint8 [B, H, W, 3] -> float64 [B, H, W]."""
    x = np.asarray(imgs, dtype=np.float64)
    return LUMA[0] * x[..., 0] + LUMA[1] * x[..., 1] + LUMA[2] * x[..., 2]


def versions(f: np.ndarray) -> dict:
    """The image itself, its left-right and up-down mirrors and its 90-degree rotation ([..., H, W])."""
    return {"orig": f, "lr": f[..., :, ::-1], "ud": f[..., ::-1, :], "r90": np.rot90(f, 1, axes=(-2, -1))}


def magnitude_spectra(f: np.ndarray, window: np.ndarray) -> dict:
    """|fft2| of each version of mean-removed images f [B, N, N] (float64), each multiplied by `window`."""
    return {k: np.abs(np.fft.fft2(v * window, axes=(-2, -1))) for k, v in versions(f).items()}


def lemma_differences(mags: dict) -> tuple:
    """Per image: max |M_LR - M_UD| / max M and max |M_R90 - M| / max M."""
    peak = mags["orig"].reshape(len(mags["orig"]), -1).max(axis=1)
    peak = np.where(peak > 0, peak, 1.0)
    e_m = np.abs(mags["lr"] - mags["ud"]).reshape(len(peak), -1).max(axis=1) / peak
    e_r = np.abs(mags["r90"] - mags["orig"]).reshape(len(peak), -1).max(axis=1) / peak
    return e_m, e_r


def choose_rows(images_path: Path, index_path: Path | None, role: str, per_class: int | None, n: int | None,
                seed: int, notes: list) -> tuple:
    arr = np.load(images_path, mmap_mode="r")
    if arr.ndim != 4 or arr.shape[-1] != 3 or arr.dtype != np.uint8:
        raise SystemExit(f"{images_path}: expected uint8 [N, H, W, 3], got {arr.dtype} {arr.shape}")
    rng = np.random.default_rng(seed)
    how = ""
    if index_path is not None and Path(index_path).exists():
        with np.load(index_path) as z:
            if f"{role}_rows" not in z.files:
                raise SystemExit(f"{index_path} has no {role}_rows")
            pool = np.asarray(z[f"{role}_rows"], dtype=np.int64)
            labels = np.asarray(z[f"{role}_labels"], dtype=np.int64) if f"{role}_labels" in z.files else None
        if n is None and labels is not None:
            k = int(per_class or 1)
            rows = np.concatenate([rng.choice(pool[labels == c], size=min(k, int((labels == c).sum())),
                                              replace=False) for c in np.unique(labels)])
            how = f"{k} per class of role {role} ({len(np.unique(labels))} classes), default_rng({seed})"
        else:
            m = min(int(n or 1000), len(pool))
            rows = rng.choice(pool, size=m, replace=False)
            how = f"{m} rows of role {role} drawn uniformly, default_rng({seed})"
    else:
        notes.append(f"no index at {index_path}: rows drawn from the whole array")
        m = min(int(n or 1000), len(arr))
        rows = rng.choice(len(arr), size=m, replace=False)
        how = f"{m} rows of the whole array drawn uniformly, default_rng({seed})"
    rows = np.sort(rows.astype(np.int64))
    return arr, rows, how


def compute(arr, rows: np.ndarray, window_kind: str, batch: int) -> dict:
    H, W = arr.shape[1:3]
    N = min(H, W)
    if H != W:
        y0, x0 = (H - N) // 2, (W - N) // 2
    win = window2d(N, window_kind)
    sums = {k: np.zeros((N, N)) for k in VERSIONS}
    e_m, e_r = [], []
    for s in range(0, len(rows), batch):
        imgs = np.asarray(arr[rows[s:s + batch]])
        if H != W:
            imgs = imgs[:, y0:y0 + N, x0:x0 + N]
        f = luminance(imgs)
        f = f - f.mean(axis=(-2, -1), keepdims=True)     # one scalar per image, shared by all four versions
        mags = magnitude_spectra(f, win)
        a, b = lemma_differences(mags)
        e_m.append(a)
        e_r.append(b)
        for k in VERSIONS:
            sums[k] += mags[k].sum(axis=0)
    means = {k: np.fft.fftshift(v / len(rows)) for k, v in sums.items()}
    return {"means": means, "e_mirror": np.concatenate(e_m), "e_rot": np.concatenate(e_r), "N": N,
            "crop_to_square": H != W}


def summarise(maps: dict) -> dict:
    """Every reported number, from the mean spectra and the per-image differences."""
    m = maps["means"]
    N = m["orig"].shape[0]
    c = N // 2                                                          # zero frequency after fftshift
    mask = np.ones((N, N), dtype=bool)
    mask[c, c] = False
    with np.errstate(divide="ignore", invalid="ignore"):
        lr_ud = np.log10(m["lr"] / m["ud"])
        rot = np.log10(m["r90"] / m["orig"])
    k = np.fft.fftshift(np.fft.fftfreq(N))
    band = (np.abs(k) >= BAND[0]) & (np.abs(k) <= BAND[1])
    on_kx = m["orig"][c, band].mean()                                   # k_y = 0: changes along the horizontal
    on_ky = m["orig"][band, c].mean()                                   # k_x = 0: changes along the vertical
    em, er = maps["e_mirror"], maps["e_rot"]
    return {"n_images": int(len(em)), "size": int(N),
            "lr_ud_max_abs_log10_ratio": float(np.nanmax(np.abs(lr_ud[mask]))),
            "lr_ud_max_rel_diff_mean_spectra": float(np.max(np.abs(m["lr"] - m["ud"])) / float(m["orig"].max())),
            "rot_max_abs_log10_ratio": float(np.nanmax(np.abs(rot[mask]))),
            "rot_p995_abs_log10_ratio": float(np.nanpercentile(np.abs(rot[mask]), 99.5)),
            "per_image_mirror_max": float(em.max()), "per_image_mirror_median": float(np.median(em)),
            "per_image_mirror_exact_zero": int((em == 0).sum()),
            "per_image_rot_min": float(er.min()), "per_image_rot_median": float(np.median(er)),
            "per_image_rot_max": float(er.max()),
            "axis_band_cycles_per_pixel": list(BAND), "mean_mag_on_kx_axis": float(on_kx),
            "mean_mag_on_ky_axis": float(on_ky), "kx_over_ky": float(on_kx / on_ky)}


# ---------------------------------------------------------------------------------------------- drawing
def draw(maps: dict, s: dict, out: Path, synthetic: bool) -> tuple:
    apply()
    plt.rcParams.update({"savefig.bbox": "standard", "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1,
                         "axes.labelsize": FS})
    m = maps["means"]
    N = m["orig"].shape[0]
    c = N // 2
    half = 0.5 / N
    ext = (-0.5 - half, 0.5 - half, 0.5 - half, -0.5 - half)            # origin "upper": k_y grows downwards
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    def map_ax(left):
        ax = fig.add_axes([left / FIG_W, MAP_B / FIG_H, MAP_S / FIG_W, MAP_S / FIG_H])
        ax.grid(False)
        return ax

    def masked(a):
        a = np.array(a, dtype=np.float64)
        a[c, c] = np.nan
        return a

    with np.errstate(divide="ignore", invalid="ignore"):
        logm = masked(np.log10(m["orig"]))
        lr_ud = masked(np.log10(m["lr"] / m["ud"]))
        rot = masked(np.log10(m["r90"] / m["orig"]))
    ax_a, ax_b, ax_c = map_ax(A_L), map_ax(B_L), map_ax(C_L)
    seq = SEQ.copy()
    seq.set_bad(C["paper"])
    lo, hi = np.nanpercentile(logm, 1), np.nanmax(logm)
    ax_a.imshow(logm, cmap=seq, vmin=lo, vmax=hi, extent=ext, origin="upper", interpolation="nearest")
    L = max(float(s["rot_p995_abs_log10_ratio"]), 1e-3)
    L = float(np.ceil(L * 20) / 20)                                       # round up to 0.05 decades
    div = DIV.copy()
    div.set_bad(C["paper"])
    norm = Normalize(-L, L)
    ax_b.imshow(lr_ud, cmap=div, norm=norm, extent=ext, origin="upper", interpolation="nearest")
    ax_c.imshow(rot, cmap=div, norm=norm, extent=ext, origin="upper", interpolation="nearest")
    for ax in (ax_a, ax_b, ax_c):
        ax.set_xticks([-0.4, 0, 0.4], ["-0.4", "0", "0.4"])
        ax.set_yticks([-0.4, 0, 0.4], ["-0.4", "0", "0.4"])
        ax.tick_params(length=2, pad=1.5)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(C["ink2"])
            sp.set_linewidth(0.5)
    ax_a.set_xlabel("$k_x$ (cycles per pixel)", labelpad=1)
    ax_a.set_ylabel("$k_y$", labelpad=1)
    for ax in (ax_b, ax_c):
        ax.tick_params(labelleft=False, labelbottom=False)
    ax_a.set_title("(a) originals", loc="left", fontsize=FS_TITLE, pad=4)
    ax_b.set_title("(b) left-right / up-down", loc="left", fontsize=FS_TITLE, pad=4)
    ax_c.set_title("(c) rotated / original", loc="left", fontsize=FS_TITLE, pad=4)
    ax_b.text(0.5, 0.5, "0 at every\nfrequency\n(largest "
              f"{s['lr_ud_max_abs_log10_ratio']:.0e})", transform=ax_b.transAxes, ha="center", va="center",
              fontsize=FS - 1.5, color=C["ink2"], linespacing=1.15)
    cax = fig.add_axes([CB[0] / FIG_W, CB[1] / FIG_H, CB[2] / FIG_W, CB[3] / FIG_H])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIV), cax=cax, orientation="horizontal",
                      ticks=[-L, 0, L])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=FS - 1.5, length=2, width=0.5, pad=1)
    cb.ax.set_xticklabels([f"{-L:g}", "0", f"{L:g}"])
    cb.set_label("log$_{10}$ ratio of mean |F|", fontsize=FS - 1.5, labelpad=1)

    # (d) per-image differences
    ax_d = fig.add_axes([D_AX[0] / FIG_W, D_AX[1] / FIG_H, D_AX[2] / FIG_W, D_AX[3] / FIG_H])
    em, er = maps["e_mirror"], maps["e_rot"]
    pos = np.concatenate([em[em > 0], er[er > 0]])
    lo_e = 10.0 ** np.floor(np.log10(pos.min())) if len(pos) else 1e-17
    hi_e = 10.0 ** np.ceil(np.log10(max(pos.max(), lo_e * 10))) if len(pos) else 1.0
    bins = np.logspace(np.log10(lo_e), np.log10(hi_e), 46)
    for vals, col, lab in ((em, C["blue"], "left-right vs up-down"), (er, C["terracotta"], "rotated vs original")):
        v = np.where(vals > 0, vals, lo_e)                                  # exact zeros drawn at the left edge
        ax_d.hist(v, bins=bins, color=col, alpha=0.85, lw=0, label=lab)
    ax_d.set_xscale("log")
    ax_d.set_xlim(lo_e, hi_e)
    ax_d.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax_d.xaxis.set_minor_locator(NullLocator())
    ax_d.set_xlabel("largest difference / peak", labelpad=1)
    ax_d.set_ylabel("images", labelpad=1)
    ax_d.set_title("(d) per image", loc="left", fontsize=FS_TITLE, pad=4)
    ax_d.grid(axis="x", visible=False)
    handles = [Line2D([], [], color=C["blue"], lw=4, label="left-right vs\nup-down"),
               Line2D([], [], color=C["terracotta"], lw=4, label="rotated vs\noriginal")]
    ax_d.legend(handles=handles, loc="upper center", fontsize=FS - 1.5, handlelength=0.8, handletextpad=0.4,
                borderaxespad=0.1, labelspacing=0.4, ncol=1)
    ax_d.set_ylim(0, ax_d.get_ylim()[1] * 1.6)
    ax_d.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    if synthetic:
        fig.text(0.995, 0.995, "synthetic data (dry run)", ha="right", va="top", fontsize=6, color=C["grey"])
    out.mkdir(parents=True, exist_ok=True)
    pdf, png = out / f"{STEM}.pdf", out / f"{STEM}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png, L


def caption(s: dict, how: str, window: str, L: float) -> str:
    more = "horizontal" if s["kx_over_ky"] > 1 else "vertical"
    ratio = s["kx_over_ky"] if s["kx_over_ky"] > 1 else 1.0 / s["kx_over_ky"]
    return (f"Lemma 2 on {s['n_images']:,} ImageNet training images ({how}; luminance, mean removed"
            + (", Hann window" if window == "hann" else "") + f", {s['size']} x {s['size']} pixels). "
            "(a) Mean magnitude spectrum of the images, log scale, zero frequency at the centre. (b) Ratio of the mean "
            "magnitude spectra of the left-right and the up-down mirror images, log scale: it is 0 at every "
            f"frequency (largest |log10 ratio| {s['lr_ud_max_abs_log10_ratio']:.1e}), and so is the difference for "
            f"every single image (largest {s['per_image_mirror_max']:.1e} of the image's peak magnitude). (c) The same "
            "ratio for the image rotated by 90 degrees against the original, on the same colour scale (from "
            f"{-L:g} to {L:g} decades): rotation swaps the two frequency axes, and the mean magnitude along the "
            f"{more} frequency axis is {ratio:.2f} times that along the other ({BAND[0]:g} to {BAND[1]:g} cycles per "
            "pixel), so the rotated spectrum differs. (d) Per image, the largest difference between the two "
            "magnitude spectra relative to the image's peak magnitude: rounding error for the two mirrors "
            f"(at most {s['per_image_mirror_max']:.1e}), a real difference for the rotation (median "
            f"{s['per_image_rot_median']:.2f}).")


# ---------------------------------------------------------------------------------------------- dry run
def make_dry_inputs(work: Path, seed: int = 0, n_classes: int = 6, per_class: int = 4, size: int = 32) -> tuple:
    """Synthetic uint8 images [N, size, size, 3] with more structure along one axis, and an index with a bank role
    (values from a seeded generator; they mean nothing). Returns (images path, index path)."""
    g = np.random.default_rng(seed)
    cache = Path(work) / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    n = n_classes * per_class
    y, x = np.mgrid[0:size, 0:size] / size
    imgs = np.zeros((n, size, size, 3), dtype=np.uint8)
    for i in range(n):
        ky, kx = g.integers(1, 6), g.integers(1, 3)
        base = 0.6 * np.sin(2 * np.pi * (ky * y + g.uniform())) + 0.25 * np.sin(2 * np.pi * (kx * x + g.uniform()))
        base = base[..., None] + 0.2 * g.normal(size=(size, size, 3))
        imgs[i] = np.clip(127 + 80 * base, 0, 255).astype(np.uint8)
    np.save(cache / "train224_uint8.npy", imgs)
    labels = np.repeat(np.arange(n_classes), per_class)
    np.savez(cache / "train_index.npz", bank_rows=np.arange(n), bank_labels=labels)
    return cache / "train224_uint8.npy", cache / "train_index.npz"


# ---------------------------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", default=str(ROOT / "cache"))
    p.add_argument("--images", default=None, help="uint8 [N, H, W, 3] .npy (default <cache-dir>/train224_uint8.npy)")
    p.add_argument("--index", default=None, help="role index .npz (default <cache-dir>/train_index.npz)")
    p.add_argument("--role", default="bank", help="role of the index (<role>_rows, <role>_labels)")
    p.add_argument("--per-class", type=int, default=1, help="images per class of the role (default 1)")
    p.add_argument("--n", type=int, default=None, help="instead: this many rows drawn uniformly from the role")
    p.add_argument("--seed", type=int, default=0, help="seed of the row choice")
    p.add_argument("--window", choices=("hann", "none"), default="hann")
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--from-maps", default=None, help="re-draw from a fourier_maps.npz written earlier")
    p.add_argument("--out", default=None, help="figure folder (default paper/figures; dry run: <work>/figures)")
    p.add_argument("--results", default=str(ROOT / "results"), help="results root (writes <results>/A14/fourier)")
    p.add_argument("--dry-run", action="store_true", help="synthetic images in a temporary folder")
    p.add_argument("--work", default=None, help="folder for the dry run's synthetic inputs (default: a new temp dir)")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    a = parse_args(argv)
    t0 = time.time()
    notes: list = []
    out = Path(a.out) if a.out else ROOT / "paper" / "figures"
    results = Path(a.results)
    images = Path(a.images) if a.images else Path(a.cache_dir) / "train224_uint8.npy"
    index = Path(a.index) if a.index else Path(a.cache_dir) / "train_index.npz"
    if a.dry_run:
        work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="fp_fig_fourier_"))
        images, index = make_dry_inputs(work)
        out = Path(a.out) if a.out else work / "figures"
        results = work / "results"
        print(f"dry run: synthetic inputs in {work}")
    data_dir = results / "A14" / "fourier"
    if a.from_maps:
        with np.load(a.from_maps) as z:
            maps = {"means": {k: z[f"mean_{k}"] for k in VERSIONS}, "e_mirror": z["e_mirror"], "e_rot": z["e_rot"],
                    "N": int(z["mean_orig"].shape[0])}
            rows, how, window = z["rows"], str(z["how"]), str(z["window"])
        notes.append(f"re-drawn from {a.from_maps}")
    else:
        if not images.exists():
            raise SystemExit(f"{images} not found (the VM cache; on the laptop use --dry-run or --from-maps)")
        arr, rows, how = choose_rows(images, index if index.exists() else None, a.role,
                                     a.per_class, a.n, a.seed, notes)
        window = a.window
        maps = compute(arr, rows, window, a.batch)
        if maps["crop_to_square"]:
            notes.append("images are not square: the central square was used")
    s = summarise(maps)
    pdf, png, L = draw(maps, s, out, a.dry_run)
    rows_sha = hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()
    if not a.from_maps:
        data_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(data_dir / "fourier_maps.npz", **{f"mean_{k}": v for k, v in maps["means"].items()},
                            e_mirror=maps["e_mirror"], e_rot=maps["e_rot"], rows=np.asarray(rows, dtype=np.int64),
                            how=np.asarray(how), window=np.asarray(window))
        metrics = {"experiment": "A14 Fourier figure (Lemma 2)", "images": str(images), "index": str(index),
                   "rows_how": how, "rows_sha256": rows_sha, "window": window, "luma": LUMA, **s,
                   "dry_run": bool(a.dry_run)}
        (data_dir / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
        try:
            from mirror.final.prereg import write_final_meta

            write_final_meta(data_dir, experiment="A14-fourier", config=vars(a), seeds={"rows": a.seed},
                             models={}, wall_seconds=time.time() - t0,
                             extra={"rows_sha256": rows_sha, "rows_how": how, "notes": notes})
        except Exception as e:  # noqa: BLE001  (the figure does not depend on it)
            notes.append(f"meta.json not written ({type(e).__name__}: {e})")

    print(f"git: {git_head()}")
    print(f"images: {rel(images)}; rows: {how}; sha256 {rows_sha[:16]}; window {window}")
    for k, v in s.items():
        print(f"  {k}: {v}")
    for n in notes:
        print(f"note: {n}")
    cap = caption(s, how, window, L)
    print(f"\nCaption: {cap}")
    meta = {"figure": STEM, "git": git_head(), "argv": list(sys.argv[1:] if argv is None else argv),
            "dry_run": bool(a.dry_run), "rows_sha256": rows_sha, "rows_how": how, "window": window,
            "colour_limit_decades": L, "numbers": s, "notes": notes, "caption": cap,
            "data": None if a.from_maps else rel(data_dir / "fourier_maps.npz")}
    mpath = out / f"{STEM}.json"
    mpath.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"\nwrote {rel(pdf)}, {rel(png)} and {rel(mpath)}" + ("" if a.from_maps else f"; data in {rel(data_dir)}"))
    return {"pdf": pdf, "png": png, "meta": mpath, "notes": notes, "numbers": s, "data_dir": data_dir}


if __name__ == "__main__":
    main()
