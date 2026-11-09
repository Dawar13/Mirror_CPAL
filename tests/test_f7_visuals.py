"""F7 visuals (scripts/f7_visuals.py): CPU tests with the tiny random ViT and synthetic uint8 images."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mirror.attention import attn_dims
from mirror.data import make_loader
from mirror.m1 import build_vexact, random_head_structure, random_involution_fixing_ones
from mirror.matching import random_pairing

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location("f7_visuals_test", ROOT / "scripts" / "f7_visuals.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


f7 = load_script()


def uint8_loader(n=12, size=56, seed=0, batch_size=5):
    imgs = np.random.default_rng(seed).integers(0, 256, size=(n, size, size, 3), dtype=np.uint8)
    return make_loader(imgs, np.zeros(n, dtype=np.int64), np.arange(n), batch_size)


def track(adapter, perms, blocks, n_pairs=5, k=3):
    """Trackers for the first n_pairs pairs (by index) of each 1-based block, as main() builds them."""
    h = adapter.mlp_hidden
    sel, all_pairs, rhos, trackers = {}, {}, {}, {}
    for b in blocks:
        perm = perms[b - 1]
        ranked = f7.rank_pairs(perm, np.zeros(h), np.zeros(h))       # equal scores: ordered by u
        all_pairs[b], sel[b] = ranked, ranked.head(n_pairs).reset_index(drop=True)
        rhos[b] = f7.random_partners(perm, seed=b)
        trackers[b - 1] = f7.BlockTracker(b, perm, rhos[b], f7.tracked_neurons(sel[b]), k, "cpu", chunk=50)
    return sel, all_pairs, rhos, trackers


# ---------------------------------------------------------------- the CLI
def test_tiny_cli_writes_all_files(tmp_path):
    out = tmp_path / "f7"
    meta = f7.main(["--tiny", "--out", str(out), "--pairs", "3", "--topk", "4", "--teaser-pairs", "1"])
    for name in ("f7_pairs.csv", "f7_blocks.csv", "f7_all_pairs.csv", "f7_top_patches.csv", "f7_layer0_filters.csv",
                 "f7_layer0.npz", "f7_crops.npz", "f7_meta.json", "F7_SUMMARY.md"):
        assert (out / name).is_file(), name
    for stem in ("fig0_teaser", "figF7_twins_full", "figF7_layer0_filters"):
        for ext in ("pdf", "png"):
            assert (out / "figures" / f"{stem}.{ext}").stat().st_size > 1000, stem

    saved = json.loads((out / "f7_meta.json").read_text(encoding="utf-8"))
    assert saved["seed"] == 0 and saved["tiny"] is True and "git_hash" in saved["git"]
    assert saved["config"]["pairs"] == 3 and saved["blocks"] == [1, 2, 4]
    assert saved["images"]["source"] == "bank" and saved["images"]["n"] == 40
    assert meta["outputs"] and all(Path(p).exists() for p in meta["outputs"])

    pairs = pd.read_csv(out / "f7_pairs.csv")
    assert len(pairs) == 9 and list(pairs["block"].unique()) == [1, 2, 4]
    assert pairs.groupby("block")["rank"].apply(list).tolist() == [[1, 2, 3]] * 3
    for col in ("corr_twin", "corr_rand", "corr_twin_vu", "corr_twin_uv"):
        assert pairs[col].between(-1, 1).all(), col
    assert (pairs["git_hash"] == saved["git"]["git_hash"]).all()
    every = pd.read_csv(out / "f7_all_pairs.csv")
    assert every.groupby("block")["chosen"].sum().tolist() == [3, 3, 3]
    blocks = pd.read_csv(out / "f7_blocks.csv")
    assert (blocks["n_pairs"].to_numpy() == every.groupby("block").size().to_numpy()).all()
    # the per-pair correlation and the per-neuron one are the same quantity computed two ways
    assert blocks["partner_vs_diag_max_abs_diff"].max() < 1e-10

    top = pd.read_csv(out / "f7_top_patches.csv")
    for _, g in top.groupby(["block", "rank", "role"]):
        assert np.all(np.diff(g["activation"].to_numpy()) <= 0)
        assert g["vis_index"].is_unique                                  # one patch per image
    with np.load(out / "f7_crops.npz") as z:
        assert z["b1_crops"].shape == (3, 2, 4, 42, 42, 3) and z["b1_valid"].all()
        assert len(z["vis_rows"]) == 40

    l0 = pd.read_csv(out / "f7_layer0_filters.csv")
    assert set(l0["operator"]) == {"A_0", "random involution"} and set(l0["parity"]) == {"even", "odd"}
    text = (out / "F7_SUMMARY.md").read_text(encoding="utf-8")
    assert "TINY MODE" in text and chr(0x2014) not in text and chr(0x2013) not in text    # no em or en dashes

    # re-drawing from the saved outputs alone gives the same figures (no model, no images)
    re_dir = tmp_path / "replot"
    info = f7.main(["--replot", str(out), "--fig-dir", str(re_dir), "--teaser-pairs", "1"])
    assert len(info["outputs"]) == 6 and (out / "f7_replot_meta.json").is_file()
    for stem in ("fig0_teaser", "figF7_twins_full", "figF7_layer0_filters"):
        assert (re_dir / f"{stem}.png").read_bytes() == (out / "figures" / f"{stem}.png").read_bytes(), stem


# ---------------------------------------------------------------- mirror check
def test_mirror_check_is_one_for_exact_twins_and_lower_for_random_partners(tiny64):
    """An exactly mirror-equivariant tiny ViT (V-exact over all blocks): every twin pair has correlation 1 in
    both directions, the twin is each neuron's best partner, and on the top patches the twin's mirror
    activation equals the neuron's own; random partners are clearly lower. The unconverted model is not
    exact for the same pairing."""
    d, h, L = tiny64.embed_dim, tiny64.mlp_hidden, tiny64.depth
    H, dh = attn_dims(tiny64.model.blocks[0])
    A = random_involution_fixing_ones(d, d // 2 + 4, seed=3)
    perms = [random_pairing(h, h // 3, seed=10 + b) for b in range(L)]
    maps = [random_head_structure(H, dh, (1, 2, 0, 1)[b], seed=20 + b) for b in range(L)]
    exact, _ = build_vexact(tiny64, A, list(range(L)), perms, maps)

    blocks = [1, 3]
    sel, all_pairs, rhos, trackers = track(tiny64, perms, blocks)
    assert f7.collect(tiny64, uint8_loader(), "cpu", trackers, model=exact) == 12
    for b in blocks:
        rows, brow, every = f7.analyse_block(trackers[b - 1], sel[b], perms[b - 1], rhos[b], all_pairs[b])
        assert len(rows) == 5 and brow["n_tokens"] == 12 * 16
        for r in rows:
            assert r["corr_twin_vu"] > 1 - 1e-9 and r["corr_twin_uv"] > 1 - 1e-9
            assert r["twin_rank_u"] == 1 and r["twin_rank_v"] == 1
            assert abs(r["top_ratio_twin"] - 1.0) < 1e-9
            assert r["corr_rand"] < 0.5 and r["max_other_u"] < 0.9 and r["max_other_v"] < 0.9
        assert every["corr_twin"].min() > 1 - 1e-9                    # every pair, not only the chosen ones
        assert every["corr_rand"].max() < 0.5 and abs(every["corr_rand"].median()) < 0.2
        assert brow["frac_paired_neurons_twin_gt_rand"] == 1.0
        assert brow["partner_vs_diag_max_abs_diff"] < 1e-10

    sel0, all0, rhos0, trackers0 = track(tiny64, perms, blocks)
    f7.collect(tiny64, uint8_loader(), "cpu", trackers0)
    _, _, every0 = f7.analyse_block(trackers0[0], sel0[1], perms[0], rhos0[1], all0[1])
    assert every0["corr_twin"].max() < 0.5


# ---------------------------------------------------------------- top-k selection
def test_top_patches_on_planted_activations():
    rng = np.random.default_rng(0)
    N, P, n, k = 23, 9, 4, 5
    acts = rng.uniform(0.0, 1.0, size=(N, P, n))
    extra = rng.normal(size=(N, P, n))
    planted = {}
    for j in range(n):
        imgs = rng.choice(N, size=k, replace=False)
        poss = rng.integers(0, P, size=k)
        acts[imgs, poss, j] = 20.0 - np.arange(k)
        # a second high patch in the best image: larger than the runner-up image, but one patch per image
        acts[imgs[0], (poss[0] + 1) % P, j] = 19.5
        planted[j] = (imgs, poss)
    tp = f7.TopPatches(n, k, extras=("x",))
    for s, e in ((0, 7), (7, 8), (8, 20), (20, 23)):                    # uneven batches
        tp.update(torch.tensor(acts[s:e]), s, {"x": torch.tensor(extra[s:e])})
    res = tp.result()
    for j, (imgs, poss) in planted.items():
        assert res["img"][j].tolist() == imgs.tolist()
        assert res["pos"][j].tolist() == poss.tolist()
        assert np.allclose(res["val"][j], 20.0 - np.arange(k))
        assert np.allclose(res["x"][j], extra[imgs, poss, j])

    # unplanted data: the same as a brute-force per-image maximum
    acts = rng.normal(size=(N, P, n))
    tp = f7.TopPatches(n, k)
    for s in range(0, N, 6):
        tp.update(torch.tensor(acts[s:s + 6]), s)
    res = tp.result()
    best = acts.max(1)
    for j in range(n):
        order = np.argsort(-best[:, j])[:k]
        assert res["img"][j].tolist() == order.tolist()
        assert res["pos"][j].tolist() == acts[order, :, j].argmax(1).tolist()

    # fewer images than k: the empty slots are marked with image -1
    tp = f7.TopPatches(2, 4)
    tp.update(torch.ones(3, 5, 2), 0)
    assert (tp.result()["img"] >= 0).sum(1).tolist() == [3, 3]


# ---------------------------------------------------------------- helpers
def test_rank_pairs_by_matching_weight_and_random_partners():
    perm = np.array([1, 0, 3, 2, 4, 6, 5])
    scores = np.array([0.9, 0.9, 0.8, 0.8, 0.5, 0.95, 0.95])          # S[i, perm i]
    self_corr = np.array([0.9, 0.8, -0.5, -0.6, 0.5, 0.0, 0.1])       # S[i, i]
    by_w = f7.rank_pairs(perm, scores, self_corr, "weight")
    assert list(zip(by_w["u"], by_w["v"])) == [(2, 3), (5, 6), (0, 1)]
    assert np.allclose(by_w["fit_w"], [2.7, 1.8, 0.1]) and by_w["rank"].tolist() == [1, 2, 3]
    by_s = f7.rank_pairs(perm, scores, self_corr, "score")
    assert list(zip(by_s["u"], by_s["v"])) == [(5, 6), (0, 1), (2, 3)]
    for seed in range(5):
        rho = f7.random_partners(perm, seed)
        assert np.all(rho != np.arange(7)) and np.all(rho != perm)
    assert np.array_equal(f7.random_partners(perm, 3), f7.random_partners(perm, 3))


def test_patch_crop_geometry():
    img = np.random.default_rng(0).integers(0, 256, size=(42, 42, 3), dtype=np.uint8)
    assert np.array_equal(f7.patch_crop(img, 4, 3, 14, 1), img)          # centre patch of a 3 x 3 grid
    corner = f7.patch_crop(img, 0, 3, 14, 1, fill=7)
    assert np.array_equal(corner[14:, 14:], img[:28, :28])
    assert (corner[:14] == 7).all() and (corner[:, :14] == 7).all()
    assert np.array_equal(f7.patch_crop(img, 8, 3, 14, 0), img[28:, 28:])


def test_layer0_filters_of_an_exactly_equivariant_embedding():
    """A_0 W = W F exactly: even principal filters are symmetric, odd ones antisymmetric, the breaking share is 0;
    a random involution does not split the filters."""
    rng = np.random.default_rng(0)
    d, p = 16, 6
    A = random_involution_fixing_ones(d, 9, seed=1)
    W0 = rng.normal(size=(d, 3 * p * p))
    W = (0.5 * (W0 + A @ f7.flip_kernels(W0, p))).reshape(d, 3, p, p)
    l0 = f7.layer0_analysis(W, A, null_seed=0)
    assert l0["breaking_share"] < 1e-20 and l0["breaking_share_null"] > 0.05
    ev, od = l0["fitted"]["even"], l0["fitted"]["odd"]
    assert ev["dim"] == 9 and od["dim"] == 7 and ev["zero"].sum() == 1     # the all-ones direction, centred away
    assert np.allclose(ev["sym_fractions"][~ev["zero"]], 1.0) and np.allclose(od["sym_fractions"], 0.0, atol=1e-12)
    assert abs(ev["sym_fraction"] - 1) < 1e-12 and od["sym_fraction"] < 1e-12
    assert abs(ev["energy_share"] + od["energy_share"] - 1) < 1e-12
    assert 0.05 < l0["null"]["even"]["sym_fraction"] < 0.95 and 0.05 < l0["null"]["odd"]["sym_fraction"] < 0.95
    table = f7.layer0_table(l0)
    assert len(table) == 2 * d and table["numerically_zero"].sum() >= 1
    rgb = f7.filter_rgb(ev["filters"][0], p)
    assert rgb.shape == (p, p, 3) and np.allclose(rgb, rgb[:, ::-1])         # a symmetric filter looks symmetric
