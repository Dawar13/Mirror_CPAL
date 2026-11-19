"""Final-phase paper figures (scripts/fp_fig_*.py): CPU only, synthetic inputs in the real formats, no downloads.

What is checked:
  * every script's --dry-run writes a non-empty PDF and PNG (and its JSON record) from synthetic inputs written in the
    formats of the producing scripts (fp_a02_stability.py, fp_a03_controls.py, fp_heal.py with knn_fast.py records,
    fp_eval.py A07 records, the train224 cache and index, a MUST-05 teaser file);
  * the plotted values are the values in the input files (means and spreads recomputed here from the fixtures);
  * missing optional inputs (no A07 flip-averaged results, methods without runs, no A03 metrics.json, no A02
    stability.csv or metrics.json, missing twin agreements) are skipped with a note instead of an error;
  * the Fourier identity of Lemma 2 holds to rounding on random images (both window choices, odd and even sizes),
    while a 90-degree rotation changes the spectrum of an anisotropic image;
  * the teaser geometry: the cache's centre crop maps back to the right box of the original image, and a tile cut
    from the original agrees with the cache pixels it replaces.
Run: .venv/Scripts/python.exe -m pytest -q tests/test_fp_figures.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_fp_figures_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


CTL = load_script("fp_fig_controls")
STB = load_script("fp_fig_stability")
FRT = load_script("fp_fig_frontier")
CRV = load_script("fp_fig_curves")
FOU = load_script("fp_fig_fourier")


def assert_figure(res):
    for key in ("pdf", "png"):
        p = Path(res[key])
        assert p.exists() and p.stat().st_size > 1000, p
    assert Path(res["pdf"]).read_bytes()[:5] == b"%PDF-"
    assert Path(res["png"]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------------------------- F3 controls
def test_controls_dry_run(tmp_path):
    res = CTL.main(["--dry-run", "--work", str(tmp_path)])
    assert_figure(res)
    df = pd.read_csv(tmp_path / "results" / "A03" / "controls.csv")
    rows = {r["control"]: r for r in res["rows"]}
    assert set(rows) == {"mirror", "C1", "C2", "C3", "C4"}
    for _, r in df.iterrows():
        got = rows[r["control"]]
        assert got["twin"] == pytest.approx(r["twin_fraction_t50_1-6"])
        assert got["same"] == pytest.approx(r["ratio_to_null_1-6"])
        if r["control"] != "C4":
            assert got["split"] == pytest.approx(r["ratio_to_null_split_1-6"])
    meta = json.loads(Path(res["meta"]).read_text(encoding="utf-8"))
    assert meta["bars"] == {"twin_max": 0.05, "ratio_min": 0.9} and meta["dry_run"] is True


def test_controls_without_metrics(tmp_path):
    res_dir = CTL.make_dry_inputs(tmp_path)
    (res_dir / "A03" / "metrics.json").unlink()
    res = CTL.main(["--results", str(res_dir), "--out", str(tmp_path / "fig")])
    assert_figure(res)
    assert any("metrics.json not found" in n for n in res["notes"])
    c4 = next(r for r in res["rows"] if r["control"] == "C4")
    assert c4["twin_lo"] is None and c4["same_lo"] is not None      # interval from the CSV, twin range skipped


# ---------------------------------------------------------------------------------------------- F2 stability
def test_stability_dry_run(tmp_path):
    res = STB.main(["--dry-run", "--work", str(tmp_path)])
    assert_figure(res)
    d = res["data"]
    pairs = pd.read_csv(tmp_path / "results" / "A02" / "pairs.csv")
    order = d["order"]
    for _, r in pairs.iterrows():
        i, j = order.index(r["a"]), order.index(r["b"])
        assert d["mats"]["weighted_agreement"][max(i, j), min(i, j)] == pytest.approx(r["weighted_agreement"])
        assert d["mats"]["twin_mean"][max(i, j), min(i, j)] == pytest.approx(r["twin_mean"])
    with np.load(tmp_path / "results" / "A02" / "spectra.npz") as z:
        lam = z["S_eig"][0]
    s0 = d["spectra"][str(np.load(tmp_path / "results" / "A02" / "spectra.npz")["names"][0])]
    assert s0["n_neg"] == int((lam < 0).sum()) and s0["even"][0] == pytest.approx(lam.max())


def test_stability_missing_optional(tmp_path):
    res_dir = STB.make_dry_inputs(tmp_path)
    (res_dir / "A02" / "stability.csv").unlink()
    (res_dir / "A02" / "metrics.json").unlink()
    pairs = pd.read_csv(res_dir / "A02" / "pairs.csv")
    pairs.loc[pairs["b"] == "coco", "twin_mean"] = np.nan                   # a fit without sidecar twin scores
    pairs.to_csv(res_dir / "A02" / "pairs.csv", index=False)
    res = STB.main(["--results", str(res_dir), "--out", str(tmp_path / "fig")])
    assert_figure(res)
    assert any("stability.csv" in n for n in res["notes"]) and any("bar not in metrics.json" in n
                                                                    for n in res["notes"])
    M = res["data"]["mats"]["twin_mean"]
    assert np.isnan(M[res["data"]["order"].index("coco"), 0])


# ---------------------------------------------------------------------------------------------- F1 frontier
def _seed_values(runs: Path, rel: str, key: str = "bf16") -> list:
    out = []
    for p in sorted((runs / rel).glob("s*/eval/step780.json")):
        out.append(json.loads(p.read_text(encoding="utf-8"))["passes"][key]["top1"])
    return out


def test_frontier_dry_run(tmp_path):
    res = FRT.main(["--dry-run", "--work", str(tmp_path), "--with-prediction"])
    assert_figure(res)
    pts = {p["key"]: p for p in res["points"]}
    runs = tmp_path / "runs" / "final"
    m = _seed_values(runs, "main/dinov2_vitb14_reg/mirror_exact_astar")
    assert len(m) == 3
    assert pts["main/mirror_exact_astar"]["y"] == pytest.approx(np.mean(m))
    assert pts["main/mirror_exact_astar"]["sd"] == pytest.approx(np.std(m, ddof=1))
    assert pts["main/mirror_exact_astar"]["x"] == pytest.approx(0.7646)
    assert pts["a06/svdllm_f050"]["n"] == 2 and pts["a06/svdllm_f040"]["sd"] is None
    stars = [p for p in res["points"] if p["series"] == "flipavg"]
    assert {p["label"] for p in stars} == {"SVD-LLM 0.40", "DINOv2-S/14 reg", "original"}
    orig = next(p for p in stars if p["label"] == "original")
    assert orig["x"] == pytest.approx(2.0)                                   # two passes of the original itself
    preds = [p for p in res["points"] if p["series"] == "prediction"]
    assert len(preds) == 3


def test_frontier_without_a07_and_redraw(tmp_path):
    res = FRT.main(["--dry-run", "--dry-no-a07", "--work", str(tmp_path)])
    assert_figure(res)
    assert not [p for p in res["points"] if p["series"] == "flipavg"]
    assert any("no flip-averaged results" in n for n in res["notes"])
    again = FRT.main(["--from-values", str(res["meta"]), "--out", str(tmp_path / "redraw")])
    assert_figure(again)
    assert [p["y"] for p in again["points"]] == [p["y"] for p in res["points"]]


def test_frontier_missing_arm(tmp_path):
    runs, results = FRT.make_dry_inputs(tmp_path, with_a07=False)
    import shutil

    shutil.rmtree(runs / "a06" / "dinov2_vitb14_reg" / "mirror_partial_k9")
    res = FRT.main(["--runs", str(runs), "--results", str(results), "--out", str(tmp_path / "fig")])
    assert_figure(res)
    assert any("mirror_partial_k9: no run directory" in n for n in res["notes"])


# ---------------------------------------------------------------------------------------------- F4 healing curves
def test_curves_dry_run(tmp_path):
    res = CRV.main(["--dry-run", "--work", str(tmp_path)])
    assert_figure(res)
    info = res["info"]
    assert info["subset_steps"] == [100, 300] and info["full_queries"] == 50000
    mirror = next(s for s in res["series"] if s["key"] == "mirror")
    runs = tmp_path / "runs" / "final" / "main" / "dinov2_vitb14_reg" / "mirror_exact_astar"
    for row in mirror["rows"]:
        vals = []
        for p in sorted(runs.glob(f"s*/eval/step{row['step']}.json")):
            rec = json.loads(p.read_text(encoding="utf-8"))
            vals.append(next(v["top1"] for v in rec["passes"].values()))
        assert row["mean"] == pytest.approx(np.mean(vals)) and row["min"] == pytest.approx(min(vals))
        assert row["max"] == pytest.approx(max(vals))
    assert mirror["rows"][1]["keys"] == ["bf16_q10000"] and mirror["rows"][0]["keys"] == ["bf16"]
    methods = {s["method"] for s in res["series"]}
    assert "pair_random_h1" in methods                                       # design doc 4.5a: H1's directory
    assert not any(m.startswith("mirror_exact_fit") for m in methods)        # A02 heals are not methods


def test_curves_missing_methods(tmp_path):
    res = CRV.main(["--dry-run", "--work", str(tmp_path), "--dry-skip", "iosvd,blockdrop"])
    assert_figure(res)
    keys = {s["key"] for s in res["series"]}
    assert "iosvd" not in keys and "blockdrop" not in keys and "mirror" in keys
    assert sum("no run directory" in n for n in res["notes"]) == 2


# ---------------------------------------------------------------------------------------------- Fourier (Lemma 2)
@pytest.mark.parametrize("n", [32, 33])
@pytest.mark.parametrize("window", ["hann", "none"])
def test_fourier_identity_random_images(n, window):
    g = np.random.default_rng(n)
    f = g.normal(size=(6, n, n))
    f -= f.mean(axis=(-2, -1), keepdims=True)
    mags = FOU.magnitude_spectra(f, FOU.window2d(n, window))
    e_m, e_r = FOU.lemma_differences(mags)
    assert e_m.max() < 1e-12                                                 # rounding only (about 1e-15)
    assert e_r.min() > 1e-3                                                  # white noise: rotation changes |F|
    # the identity itself: |F(LR f)|(kx, ky) = |F(f)|(-kx, ky)
    F = np.abs(np.fft.fft2(f * FOU.window2d(n, window), axes=(-2, -1)))
    neg = (-np.arange(n)) % n
    assert np.allclose(mags["lr"], F[..., :, neg], rtol=0, atol=1e-10 * F.max())
    # the window is unchanged by both mirrors and by the rotation, exactly
    w = FOU.window2d(n, window)
    assert np.array_equal(w, w[:, ::-1]) and np.array_equal(w, w[::-1, :]) and np.array_equal(w, np.rot90(w))


def test_fourier_rotation_sees_anisotropy():
    n = 32
    y = np.arange(n)[:, None] * np.ones((1, n))
    f = np.sin(2 * np.pi * 3 * y / n)[None]                                  # horizontal stripes: power on k_y
    f = f - f.mean()
    e_m, e_r = FOU.lemma_differences(FOU.magnitude_spectra(f, FOU.window2d(n, "hann")))
    assert e_m.max() < 1e-12 and e_r.min() > 0.5


def test_fourier_without_cache_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit, match="not found"):                       # the laptop: no cache, no work
        FOU.main(["--cache-dir", str(tmp_path), "--out", str(tmp_path / "fig")])
    assert not (tmp_path / "fig").exists()


def test_fourier_dry_run_and_redraw(tmp_path):
    res = FOU.main(["--dry-run", "--work", str(tmp_path)])
    assert_figure(res)
    s = res["numbers"]
    assert s["n_images"] == 6 and s["per_image_mirror_max"] < 1e-12 and s["lr_ud_max_abs_log10_ratio"] < 1e-12
    assert s["per_image_rot_median"] > 0.01
    maps = res["data_dir"] / "fourier_maps.npz"
    assert maps.exists() and (res["data_dir"] / "metrics.json").exists()
    again = FOU.main(["--from-maps", str(maps), "--out", str(tmp_path / "redraw"),
                      "--results", str(tmp_path / "redraw_results")])
    assert_figure(again)
    assert again["numbers"] == pytest.approx(s)


# ---------------------------------------------------------------------------------------------- teaser
@pytest.fixture(scope="module")
def TSR():
    return load_script("fp_fig_teaser")


def test_teaser_geometry(TSR):
    from PIL import Image

    g = np.random.default_rng(0)
    for w, h in ((375, 500), (500, 333), (640, 427), (256, 300)):
        geo = TSR.cache_geometry(w, h, 256, 224)
        assert min(geo["new_w"], geo["new_h"]) == 256
        # the whole crop maps to the centred region of the original with the right aspect
        bo = TSR.box_in_original((0, 0, 224, 224), geo)
        assert 0 <= bo[0] < bo[2] <= w + 1e-9 and 0 <= bo[1] < bo[3] <= h + 1e-9
        assert (bo[2] - bo[0]) == pytest.approx(224 * w / geo["new_w"])
        assert (bo[0] + bo[2]) / 2 == pytest.approx(w / 2, abs=w / geo["new_w"])
        assert (bo[1] + bo[3]) / 2 == pytest.approx(h / 2, abs=h / geo["new_h"])
    # a smooth original: the full-resolution tile of a token window, brought back to 42 px, matches the cache crop
    w, h = 300, 240
    yy, xx = np.mgrid[0:h, 0:w]
    arr = np.stack([127 + 80 * np.sin(xx / 23.0 + c) * np.cos(yy / 31.0 - c) for c in range(3)], -1)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    cache = np.asarray(TSR.resize_centre_crop(img, 256, 224))
    try:
        from mirror.final.data.cache import resize_centre_crop as infra_crop
    except Exception:  # noqa: BLE001  (another agent's module; the geometry above is checked on its own)
        infra_crop = None
    if infra_crop is not None:
        assert np.array_equal(cache, np.asarray(infra_crop(img, 256, 224)))
    geo = TSR.cache_geometry(w, h, 256, 224)
    for pos in (0, 17, 135, 255):
        box = TSR.token_box(pos, 16, 14, 42)
        ref = TSR.crop_cache(cache, box).astype(np.float64)
        tile = TSR.crop_original(img, TSR.box_in_original(box, geo), 168)
        back = np.asarray(Image.fromarray(tile).resize((42, 42), Image.BICUBIC), dtype=np.float64)
        inside = ref[..., 0] != TSR.FILL
        assert np.abs(back - ref)[inside].mean() < 4.0, pos                 # out of 255: the same picture
        other = TSR.crop_cache(cache, TSR.token_box((pos + 97) % 256, 16, 14, 42)).astype(np.float64)
        assert np.abs(other - ref).mean() > 10.0                            # a different window is far away
    # pixel repetition of the cache fallback is exact
    t = g.integers(0, 255, (42, 42, 3), dtype=np.uint8)
    big = TSR.enlarge(t, 168)
    assert np.array_equal(big[::4, ::4], t)


def test_teaser_defaults_exit_cleanly_without_inputs(TSR, tmp_path, monkeypatch):
    monkeypatch.setattr(TSR, "DEFAULT_TEASERS", (tmp_path / "a.npz", tmp_path / "b.npz"))
    with pytest.raises(SystemExit, match="upload"):                          # no earlier teaser: say what to upload
        TSR.main(["--cache-dir", str(tmp_path), "--out", str(tmp_path / "fig")])
    monkeypatch.setattr(TSR, "DEFAULT_TEASERS", (tmp_path / "a.npz", ROOT / "tests" / "no_such.npz"))
    np.savez(tmp_path / "a.npz", vis_rows=np.arange(2), blocks=np.array([1]), crop=np.array(42),
             patch=np.array(14), grid_w=np.array(16), b1_pairs=np.zeros((1, 2), int), b1_fit_S=np.ones(1),
             b1_top_img=np.zeros((2, 1), int), b1_top_pos=np.zeros((2, 1), int), b1_top_val=np.ones((2, 1)))
    with pytest.raises(SystemExit, match="train224_uint8.npy not found"):    # found it; no cache: stop, no model
        TSR.main(["--cache-dir", str(tmp_path), "--out", str(tmp_path / "fig")])
    assert not (tmp_path / "fig").exists()


def test_teaser_dry_run_and_redraw(TSR, tmp_path, monkeypatch):
    monkeypatch.setenv("MIRROR_FINAL_ALLOW_NO_PREREG", "1")
    res = TSR.main(["--dry-run", "--work", str(tmp_path)])
    assert_figure(res)
    assert res["tiles"] and all(Path(p).stat().st_size > 0 for p in res["tiles"])
    assert all(r["source"] == "original" for r in res["info"])
    assert res["checks"] and all(c["pos_match"] for c in res["checks"])
    assert max(c["rel_diff"] for c in res["checks"]) < 1e-5
    T = np.load(tmp_path / "teaser.npz")
    e = res["selection"][0]
    col = 2 * (e["pair"] - 1) + (0 if e["side"] == "i" else 1)
    assert e["pos"] == int(T[f"b{e['block']}_top_pos"][col, e["rank"] - 1])   # the stored choice, unchanged
    tiles = res["data_dir"] / "tiles.npz"
    again = TSR.main(["--from-tiles", str(tiles), "--rows", "1:1,2:1", "--figure-patches", "4",
                      "--out", str(tmp_path / "redraw")])
    assert_figure(again)
