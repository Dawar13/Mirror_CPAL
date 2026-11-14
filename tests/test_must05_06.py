"""MUST-05 (scripts/must05_twins.py) and MUST-06 (scripts/must06_visibility.py) with mirror/twins.py:
CPU tests with the tiny random ViT, synthetic data and exactly mirror-equivariant tiny models."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mirror import twins as T
from mirror.adapters import TimmTinyAdapter, ViTAdapter, _randomise_affine
from mirror.attention import attn_dims
from mirror.data import make_loader
from mirror.m1 import build_vexact, random_head_structure, random_involution_fixing_ones
from mirror.matching import random_pairing

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m05 = load_script("must05_twins")
m06 = load_script("must06_visibility")


class RegTiny(TimmTinyAdapter):
    """Tiny float64 timm ViT with 2 register tokens on a 3 x 3 patch grid (odd width: the centre column maps
    to itself), so the code paths for registers and another grid size are exercised."""

    def __init__(self):
        from timm.models.vision_transformer import VisionTransformer

        torch.manual_seed(0)
        model = VisionTransformer(img_size=42, patch_size=14, embed_dim=32, depth=3, num_heads=4,
                                  init_values=1e-5, reg_tokens=2)
        _randomise_affine(model, 0)
        ViTAdapter.__init__(self, model.double(), img_size=42, patch_size=14,
                            n_prefix_tokens=int(model.num_prefix_tokens))


def exact_model(adapter, seed=0):
    """V-exact conversion of every block with one operator A and random twin pairings: the residual stream
    of the mirrored image, un-flipped, is exactly the original's times A at every layer."""
    d, h, L = adapter.embed_dim, adapter.mlp_hidden, adapter.depth
    H, dh = attn_dims(adapter.model.blocks[0])
    A = random_involution_fixing_ones(d, d // 2 + 4, seed=3 + seed)
    perms = [random_pairing(h, h // 3, seed=10 + b + seed) for b in range(L)]
    maps = [random_head_structure(H, dh, 1, seed=20 + b + seed) for b in range(L)]
    exact, _ = build_vexact(adapter, A, list(range(L)), perms, maps)
    return exact, A, perms


def uint8_images(n, size, seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(n, size, size, 3), dtype=np.uint8)


# ---------------------------------------------------------------- the CLIs (tiny mode)
@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("must")
    m5 = m05.main(["--tiny", "--out", str(root / "m05")])
    sdir = root / "m05" / "tiny" / "tiny_fit"
    # an "earlier run" of the same structure in the run_e1.py CSV layout, with one block's count changed
    from mirror.variants import load_fit_run

    st = load_fit_run(sdir)
    ref = root / "ref" / "results"
    ref.mkdir(parents=True)
    pd.DataFrame({"block": [1, 2, 3], "n_pairs": [st.n_pairs(0), st.n_pairs(1), st.n_pairs(2) + 1]}).to_csv(
        ref / "e1_mlp.csv", index=False)
    pd.DataFrame({"layer": range(len(st.d_plus)), "d_plus": st.d_plus}).to_csv(ref / "e1_layers.csv", index=False)
    m6 = m06.main(["--tiny", "--out", str(root / "m06"), "--structure-dir", str(sdir),
                   "--reference-run", str(root / "ref")])
    return root, m5, m6


def test_must05_tiny_cli_writes_everything(runs):
    root, metrics, _ = runs
    out = root / "m05" / "tiny"
    for name in ("tracking_pairs.csv", "tracking_summary.csv", "switch_images.csv", "switch_pairs.csv",
                 "switch_summary.csv", "teaser.npz", "teaser_top_patches.csv", "teaser.png", "teaser_block1.png",
                 "teaser_block3.png", "fig_tracking.png", "fig_switch.png", "metrics.json", "meta.json"):
        assert (out / name).is_file(), name
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["experiment"] == "MUST-05" and "git_hash" in meta["git"] and meta["config"]["tiny"] is True
    assert meta["seeds"]["fit"] == 0 and meta["wall_seconds"] > 0
    saved = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert saved["blocks"] == [1, 2, 3] and saved["structure"]["kind"] == "fresh fit"
    assert saved["notes"]["fit_twin_score_check_max_abs_diff"] < 1e-6     # the fit pass reproduces the matching scores

    tr = pd.read_csv(out / "tracking_pairs.csv")
    assert set(tr["pair_type"]) == {"twin", "random", "cosine", "same_image"}
    sizes = tr.groupby(["block", "pair_type"]).size().unstack()
    assert (sizes.nunique(axis=1) == 1).all()                              # every pair set has the twin count
    for col in ("val_mirror_corr_post", "val_mirror_corr_pre", "val_same_corr_post", "fit_mirror_corr_pre"):
        assert tr[col].between(-1, 1).all(), col
    for (_, t), g in tr.groupby(["block", "pair_type"]):                   # pairs are disjoint within a set
        assert len(set(g["i"]) | set(g["j"])) == 2 * len(g)

    sw = pd.read_csv(out / "switch_pairs.csv")
    assert len(sw) == 3 * 3 * 3 and set(sw["pair_type"]) == {"twin", "random", "random_partner"}
    assert sw["mid_cos"].between(-1, 1).all() and (sw["mid_norm_ratio"] > 0).all() and sw["cls_cos"].notna().all()
    assert (sw.loc[sw.pair_type == "random_partner", "i"].to_numpy() == sw.loc[sw.pair_type == "twin", "i"].to_numpy()).all()
    tw = sw[sw.pair_type == "twin"]
    assert (tw.groupby("block")["twin_fit_score"].apply(lambda s: s.is_monotonic_decreasing)).all()
    v = metrics["switch_off"]["verdicts_primary"]
    assert set(v) == {"twin_median_cos", "twin_share_ratio_in_band", "random_median_cos"}
    assert all(x["pass"] in (True, False) for x in v.values())
    imgs = pd.read_csv(out / "switch_images.csv")
    assert len(imgs) == 6 and imgs["label"].value_counts().max() - imgs["label"].value_counts().min() <= 1

    with np.load(out / "teaser.npz") as z:
        assert z["b1_crops"].shape == (2, 2, 4, 42, 42, 3) and z["b1_valid"].all()
        assert len(z["vis_rows"]) == 40 and z["blocks"].tolist() == [1, 3]
        assert str(z["rank_by"]) == "weight" and np.all(np.diff(z["b1_fit_w"]) <= 0)    # strongest first
    top = pd.read_csv(out / "teaser_top_patches.csv")
    for _, g in top.groupby(["block", "rank", "role"]):
        assert np.all(np.diff(g["activation"].to_numpy()) <= 0) and g["train_row"].is_unique
    text = "".join((out / n).read_text(encoding="utf-8") for n in ("metrics.json", "meta.json"))
    assert chr(0x2014) not in text and chr(0x2013) not in text


def test_must05_resumes_without_recomputing(runs):
    root, metrics, _ = runs
    out = root / "m05" / "tiny"
    stamps = {p: p.stat().st_mtime_ns for p in [out / "tracking_pairs.csv", out / "teaser.npz",
                                                *sorted((out / "switch").glob("block*.npz"))]}
    again = m05.main(["--tiny", "--out", str(root / "m05")])
    assert all(p.stat().st_mtime_ns == t for p, t in stamps.items())
    assert again["structure"]["reused"] is True
    assert again["switch_off"]["pooled"] == metrics["switch_off"]["pooled"]


def test_must06_tiny_cli_reuses_the_structure(runs):
    root, _, metrics = runs
    out = root / "m06" / "tiny"
    for name in ("pairs.csv", "blocks.csv", "spectra.csv", "spectra.npz", "fig_correlations.png",
                 "fig_spectra.png", "metrics.json", "meta.json"):
        assert (out / name).is_file(), name
    assert metrics["structure"]["reused"] is True and metrics["blocks"] == [1, 2, 3]
    chk = metrics["structure"]["reference_check"]
    assert chk["d_plus_equal"] is True and chk["n_pairs_equal"] is False
    assert [r["fresh"] == r["reference"] for r in chk["n_pairs"].values()] == [True, True, False]
    assert set(metrics["verdicts"]) == {"twin_median_same_token_corr", "share_twin_in_top_partners",
                                        "max_abs_rel_change_effective_rank"}
    pairs = pd.read_csv(out / "pairs.csv")
    m5 = pd.read_csv(root / "m05" / "tiny" / "tracking_pairs.csv")
    tw5 = m5[m5.pair_type == "twin"].set_index(["block", "i", "j"])
    tw6 = pairs[pairs.pair_type == "twin"].set_index(["block", "i", "j"])
    assert tw5.index.equals(tw6.index)                                     # same twins, same validation images
    assert np.allclose(tw5["val_same_corr_post"], tw6["same_corr_post"])
    assert np.allclose(tw5["val_mirror_corr_pre"], tw6["mirror_corr_pre"])
    h = int(pairs["h"].iloc[0])
    for c in [c for c in pairs if c.startswith("rank_")]:
        assert pairs[c].between(1, h - 1).all(), c
    sp = pd.read_csv(out / "spectra.csv")
    assert set(sp["structure"]) == {"mirror", "random"} and len(sp) == 3 * 2 * 2
    assert (sp["erank_l1_before"] <= sp["n_singular"] + 1e-9).all()
    assert meta_ok(out / "meta.json", "MUST-06")


def meta_ok(path, exp):
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    return m["experiment"] == exp and m["config"]["tiny"] is True and "git_hash" in m["git"]


# ---------------------------------------------------------------- switch-off on exact models
@pytest.mark.parametrize("which", ["plain", "registers"])
def test_switch_off_is_exact_for_exact_twins(which, tiny64):
    adapter = tiny64 if which == "plain" else RegTiny()
    exact, A, perms = exact_model(adapter)
    size = adapter.img_size
    imgs = uint8_images(5, size)
    b = 0
    X, Xm = T.block_inputs(adapter, imgs, "cpu", b, batch_size=2, model=exact)
    tw, _ = T.strongest_twin_pairs(perms[b], np.zeros(adapter.mlp_hidden), 5)
    pairs = {"twin": tw, "random": T.random_neuron_pairs(perms[b], 5, seed=1),
             "random_partner": np.stack([tw[:, 0], T.random_partners_for(perms[b], tw[:, 0], seed=2)], 1)}
    res = T.switch_off_block(adapter, b, X, Xm, pairs, A, A, model=exact, chunk_images=2, abl_batch=12,
                             progress=False)
    s = T.summarise_pairs(res["twin"])
    for pre in ("mid", "patch", "cls"):
        assert np.allclose(s[f"{pre}_cos"], 1.0, atol=1e-9), pre
        assert np.allclose(s[f"{pre}_cos_img_median"], 1.0, atol=1e-9), pre
        assert np.allclose(s[f"{pre}_norm_ratio"], 1.0, atol=1e-9), pre
    assert (s["mid_rms_i"] > 0).all()
    for t in ("random", "random_partner"):
        r = T.summarise_pairs(res[t])
        assert np.abs(r["mid_cos"]).max() < 0.5, t
    # the unconverted model is not equivariant for this pairing
    X0, Xm0 = T.block_inputs(adapter, imgs, "cpu", b, batch_size=2)
    res0 = T.switch_off_block(adapter, b, X0, Xm0, {"twin": tw}, A, None, chunk_images=2, abl_batch=12, progress=False)
    s0 = T.summarise_pairs(res0["twin"])
    assert np.abs(s0["mid_cos"]).max() < 0.5 and np.isnan(s0["cls_cos"]).all()


def test_switch_off_batching_does_not_change_the_result(tiny64):
    exact, A, perms = exact_model(tiny64, seed=1)
    imgs = uint8_images(5, 56, seed=1)
    X, Xm = T.block_inputs(tiny64, imgs, "cpu", 1, batch_size=3)
    pairs = {"random": T.random_neuron_pairs(perms[1], 4, seed=5)}
    r1 = T.switch_off_block(tiny64, 1, X, Xm, pairs, A, A, chunk_images=1, abl_batch=3, progress=False)
    r2 = T.switch_off_block(tiny64, 1, X, Xm, pairs, A, A, chunk_images=5, abl_batch=64, progress=False)
    for k in T.SWITCH_KEYS:
        assert np.allclose(r1["random"][k], r2["random"][k], rtol=1e-9, atol=1e-14), k
    with pytest.raises(ValueError):
        T.switch_off_block(tiny64, tiny64.depth - 1, X, Xm, pairs, A, A, progress=False)


def test_summarise_pairs_arithmetic():
    dot = np.array([[1.0, 2.0, 0.0], [-1.0, -1.0, -1.0]])
    ni = np.array([[1.0, 4.0, 0.0], [1.0, 1.0, 1.0]])
    nj = np.array([[1.0, 1.0, 4.0], [4.0, 4.0, 4.0]])
    res = {f"{p}_{k}": v for p in ("mid", "patch", "cls") for k, v in (("dot", dot), ("ni", ni), ("nj", nj))}
    s = T.summarise_pairs(res)
    assert np.isclose(s["mid_cos"][0], 3.0 / np.sqrt(5.0 * 6.0))
    assert np.isclose(s["mid_cos_img_median"][0], 1.0)                  # images 1 and 2 only (image 3 has ni = 0)
    assert np.isclose(s["mid_norm_ratio"][0], np.sqrt(6.0 / 5.0))
    assert np.isclose(s["mid_cos"][1], -0.5) and np.isclose(s["mid_norm_ratio"][1], 2.0)


# ---------------------------------------------------------------- activation statistics
def test_act_stats_match_brute_force_correlations():
    rng = np.random.default_rng(0)
    h, n = 6, 37
    z = rng.normal(size=(n, h))
    zt = 0.7 * z[:, [1, 0, 2, 4, 3, 5]] + 0.3 * rng.normal(size=(n, h))
    st = T.ActStats([0], h, "cpu", kinds=("pre",), chunk=5)
    for s, e in ((0, 10), (10, 11), (11, 37)):                           # uneven batches and chunks
        st.update(0, "pre", torch.tensor(z[s:e])[None], torch.tensor(zt[s:e])[None])
    full = np.corrcoef(np.concatenate([zt, z], axis=1).T)
    R = st.mirror_corr(0, "pre").numpy()
    Q = st.same_corr(0, "pre").numpy()
    assert np.allclose(R, full[:h, h:], atol=1e-12)                      # R[i, j] = corr(zt_i, z_j)
    assert np.allclose(Q, full[h:, h:], atol=1e-12)
    assert st.n_tokens(0) == n
    assert np.isclose(T.pair_values(R, [[0, 1]])[0], 0.5 * (full[0, h + 1] + full[1, h]))
    assert np.isclose(T.pair_values(st.mirror_corr(0, "pre"), [[0, 1]])[0], T.pair_values(R, [[0, 1]])[0])


@pytest.mark.parametrize("which", ["plain", "registers"])
def test_tracking_statistics_of_exact_twins(which, tiny64):
    adapter = tiny64 if which == "plain" else RegTiny()
    exact, A, perms = exact_model(adapter)
    imgs = uint8_images(12, adapter.img_size)
    loader = make_loader(imgs, np.zeros(12, dtype=np.int64), np.arange(12), 5)
    st = T.collect_act_stats(adapter, loader, "cpu", [0, 2], model=exact)
    assert st.n_tokens(0) == 12 * adapter.n_tokens
    for b in (0, 2):
        tw = T.twin_pairs(perms[b])
        rp = T.matched_random_pairs(adapter.mlp_hidden, len(tw), seed=1)
        for kind in T.KINDS:
            R = st.mirror_corr(b, kind)
            assert T.pair_values(R, tw).min() > 1 - 1e-9, (b, kind)
            assert np.median(np.abs(T.pair_values(R, rp))) < 0.3, (b, kind)
            assert np.allclose(np.diag(st.same_corr(b, kind).numpy()), 1.0)


# ---------------------------------------------------------------- detector, pairings, spectra
def test_partner_ranks_with_ties_and_signs():
    Q = np.array([[1.0, 0.5, -0.9, 0.5],
                  [0.5, 1.0, 0.1, 0.2],
                  [-0.9, 0.1, 1.0, 0.3],
                  [0.5, 0.2, 0.3, 1.0]])
    ri, rj = T.partner_ranks(Q, [[0, 1], [1, 2]])
    assert ri.tolist() == [2, 3] and rj.tolist() == [1, 2]                # the tie 0.5 = 0.5 counts against the twin
    ri, rj = T.partner_ranks(Q, [[0, 1], [1, 2]], absolute=True)
    assert ri.tolist() == [3, 3] and rj.tolist() == [1, 3]
    ri, _ = T.partner_ranks(torch.tensor(Q), [[0, 2]], absolute=True)
    assert ri.tolist() == [1]


def test_same_image_pairing_recovers_planted_pairs_and_matches_the_count():
    rng = np.random.default_rng(0)
    h = 12
    Q = 0.05 * rng.normal(size=(h, h))
    Q = 0.5 * (Q + Q.T)
    for a, b, r in ((0, 5, 0.9), (1, 7, 0.8), (2, 3, 0.7), (4, 11, 0.6)):
        Q[a, b] = Q[b, a] = r
    np.fill_diagonal(Q, 1.0)
    (p3,), (p6,) = T.same_image_pairings([Q], [3]), T.same_image_pairings([Q], [6])
    assert p3.tolist() == [[0, 5], [1, 7], [2, 3]]
    assert len(p6) == 6 and len(set(p6.ravel())) == 12 and {(0, 5), (1, 7), (2, 3), (4, 11)} <= set(map(tuple, p6))


def test_rank_twin_pairs_matches_f7_ranking():
    f7 = load_script("f7_visuals")
    rng = np.random.default_rng(0)
    h = 30
    perm = random_pairing(h, 9, seed=4)
    S = rng.uniform(-1, 1, size=(h, h))
    S = 0.5 * (S + S.T)
    for by in ("weight", "score"):
        pairs, s, w = T.rank_twin_pairs(perm, S, 5, by=by)
        ref = f7.rank_pairs(perm, S[np.arange(h), perm], np.diag(S), by).head(5)
        assert pairs.tolist() == ref[["u", "v"]].to_numpy().tolist(), by
        assert np.allclose(s, ref["fit_S_uv"]) and np.allclose(w, ref["fit_w"])
    pairs, s = T.strongest_twin_pairs(perm, S[np.arange(h), perm], 5)
    assert pairs.tolist() == T.rank_twin_pairs(perm, S, 5, by="score")[0].tolist()


def test_pair_helpers():
    perm = np.array([1, 0, 3, 2, 4, 6, 5])
    pairs, s = T.strongest_twin_pairs(perm, np.array([0.5, 0.5, 0.9, 0.9, 1.0, 0.7, 0.7]), 2)
    assert pairs.tolist() == [[2, 3], [5, 6]] and np.allclose(s, [0.9, 0.7])
    assert T.twin_pairs(perm).tolist() == [[0, 1], [2, 3], [5, 6]]
    for seed in range(5):
        j = T.random_partners_for(perm, np.arange(7), seed)
        assert np.all(j != np.arange(7)) and np.all(j != perm)
        rp = T.random_neuron_pairs(perm, 5, seed)
        assert len(set(rp[:, 0])) == 5 and np.all(rp[:, 1] != perm[rp[:, 0]])
    mp = T.matched_random_pairs(20, 6, seed=3)
    assert len(mp) == 6 and len(set(mp.ravel())) == 12
    assert T.parse_blocks("1-3,6", 12) == [0, 1, 2, 5]
    with pytest.raises(SystemExit):
        T.parse_blocks("0,2", 12)


def test_output_signatures(tmp_path):
    out = tmp_path / "x.csv"
    sig = {"blocks": [1, 2], "images": 20, "structure": "abc"}
    assert not T.output_current(out, sig)
    out.write_text("a\n1\n")
    assert not T.output_current(out, sig)                                # no signature yet
    T.write_signature(out, sig)
    assert T.output_current(out, dict(reversed(list(sig.items()))))    # key order does not matter
    assert not T.output_current(out, {**sig, "images": 40})


def test_balanced_subset():
    labels = np.repeat(np.arange(10), 7)
    np.random.default_rng(0).shuffle(labels)
    for n in (5, 10, 23, 70):
        idx = T.balanced_subset(labels, n, seed=1)
        counts = np.bincount(labels[idx], minlength=10)
        assert len(idx) == n and len(set(idx)) == n and counts.max() - counts[counts > 0].min() <= 1
        assert (counts > 0).sum() == min(n, 10)
    assert np.array_equal(T.balanced_subset(labels, 23, seed=1), T.balanced_subset(labels, 23, seed=1))
    pool = np.arange(0, 70, 2)
    assert set(T.balanced_subset(labels, 15, seed=0, pool=pool)) <= set(pool)


def test_effective_rank_and_exact_projection_spectra(tiny64):
    assert np.isclose(T.effective_rank(np.ones(7)), 7.0)
    assert np.isclose(T.effective_rank([3.0, 0.0, 0.0]), 1.0)
    p = np.array([2.0, 1.0, 1.0]) / 4
    assert np.isclose(T.effective_rank([2.0, 1.0, 1.0]), np.exp(-(p * np.log(p)).sum()))
    assert np.isclose(T.effective_rank([1.0, 1.0], "energy"), 2.0)
    exact, A, perms = exact_model(tiny64)
    for b in (0, 3):
        sp = T.mlp_spectra(exact.blocks[b], A, A, perms[b])        # already in the subspace: nothing changes
        for r in sp.values():
            assert r["breaking_share"] < 1e-20 and np.allclose(r["before"], r["after"], atol=1e-12)
        sp0 = T.mlp_spectra(tiny64.model.blocks[b], A, A, perms[b])
        assert all(r["breaking_share"] > 0.1 for r in sp0.values())


def test_token_crop_matches_f7_and_other_patch_sizes():
    f7 = load_script("f7_visuals")
    img = np.random.default_rng(0).integers(0, 256, size=(56, 56, 3), dtype=np.uint8)
    for pos in range(16):
        assert np.array_equal(T.token_crop(img, pos, 4, 14, 42), f7.patch_crop(img, pos, 4, 14, 1))
    big = np.random.default_rng(1).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    c = T.token_crop(big, 5, 4, 16, 42)                                  # patch (1, 1): pixels 16..31, centre 24
    assert c.shape == (42, 42, 3) and np.array_equal(c, big[3:45, 3:45])
