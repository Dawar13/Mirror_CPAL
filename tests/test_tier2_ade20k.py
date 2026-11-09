"""Tier 2 ADE20K linear segmentation (scripts/tier2_ade20k.py): CPU tests with tiny models and synthetic
segmentation data. Nothing here downloads data or weights."""
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mirror.adapters import Dinov2Adapter, TimmTinyAdapter, ViTAdapter
from mirror.attention import attn_dims
from mirror.data import normalize_for
from mirror.m1 import build_vexact, random_head_structure, random_involution_fixing_ones
from mirror.matching import random_pairing

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    key = f"{name}_segtest"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


S = load_script("tier2_ade20k")
REQUESTED = "original,healed_mirror_k9,healed_svd_k9,healed_random_k9,vexact_k6,mirror_mlp_k6,svd_mlp_k6"


# ---------------------------------------------------------------- variant names and defaults
def test_requested_variant_names_resolve_and_protocol_defaults():
    a = S.parse_args(["--model", "dinov2_vitb14_reg", "--variants", REQUESTED, "--out", "runs/t2_seg"])
    got = {v.name: (v.family, v.method, v.k) for v in S.T2.parse_variants(a.variants)}
    assert got == {"original": ("original", "original", 0), "healed_mirror_k9": ("healed", "mirror", 9),
                   "healed_svd_k9": ("healed", "svd", 9), "healed_random_k9": ("healed", "random", 9),
                   "vexact_k6": ("m1", "vexact", 6), "mirror_mlp_k6": ("mlp", "mirror", 6),
                   "svd_mlp_k6": ("mlp", "svd", 6)}
    assert all(v.label and "{k}" not in v.label for v in S.T2.parse_variants(a.variants))
    # The protocol of PAPER_EXPERIMENTS.md: 448 px crops, 8,000 iterations, batch 16.
    assert (a.crop, a.iters, a.batch_size, a.eval_stride) == (448, 8000, 16, 298)
    assert a.crop % 14 == 0 and a.eval_stride <= a.crop
    assert a.hf_id == "merve/scene_parse_150" and a.hf_revision == S.KNOWN_REVISIONS["merve/scene_parse_150"]
    assert S.parse_args(["--hf-id", "danjacobellis/scene_parse_150"]).hf_revision == \
        S.KNOWN_REVISIONS["danjacobellis/scene_parse_150"]
    t = S.parse_args(["--tiny"])
    assert (t.crop, t.batch_size, t.num_workers) == (56, 4, 0)
    for bad in ("vexact", "healed_vexact_k0", "mirror_k6", "svd_mlp_k0"):    # healed_vexact_k<K> (K >= 1) is valid
        with pytest.raises(ValueError):
            S.T2.parse_variants(bad)


# ---------------------------------------------------------------- labels, loss and metrics
def test_label_mapping_and_ignore_index():
    lut = S.label_lut(150)
    assert lut[0] == 255 and lut[1] == 0 and lut[150] == 149 and lut[151] == 255 and lut[255] == 255
    assert np.array_equal(lut[1:151], np.arange(150))
    small = S.label_lut(4)
    assert list(small[:6]) == [255, 0, 1, 2, 3, 255]

    # The loss ignores label 255: changing the logits there changes neither the loss nor any gradient.
    g = torch.Generator().manual_seed(0)
    lab = torch.randint(0, 3, (2, 4, 4), generator=g)
    lab[0, :2] = 255
    logits = torch.randn(2, 3, 4, 4, generator=g, requires_grad=True)
    loss = S.seg_loss(logits, lab)
    assert torch.allclose(loss, F.cross_entropy(logits, lab, ignore_index=255))
    loss.backward()
    assert logits.grad[0, :, :2].abs().max() == 0 and logits.grad[0, :, 2:].abs().max() > 0
    other = logits.detach().clone()
    other[0, :, :2] = 100.0 * torch.randn(3, 2, 4, generator=g)
    assert torch.allclose(S.seg_loss(other, lab), loss.detach())
    # A batch without labelled pixels gives 0 (not NaN) and zero gradients.
    z = torch.randn(1, 3, 4, 4, requires_grad=True)
    l0 = S.seg_loss(z, torch.full((1, 4, 4), 255))
    l0.backward()
    assert float(l0.detach()) == 0.0 and torch.isfinite(z.grad).all() and z.grad.abs().max() == 0

    # The confusion matrix skips ignored pixels whatever the prediction there.
    conf = torch.zeros(9, dtype=torch.int64)
    S.confusion_update(conf, torch.tensor([0, 1, 2, 2]), torch.tensor([0, 255, 2, 255]), 3)
    assert conf.view(3, 3).tolist() == [[1, 0, 0], [0, 0, 0], [0, 0, 1]]


def test_miou_on_planted_predictions():
    label = torch.tensor([0, 0, 1, 1, 2, 2, 255, 255, 0])
    pred = torch.tensor([0, 1, 1, 1, 0, 2, 3, 0, 0])
    conf = torch.zeros(16, dtype=torch.int64)
    S.confusion_update(conf, pred, label, 4)
    m = S.seg_metrics(conf.view(4, 4).numpy())
    # class 0: TP 2, label 3, pred 3 -> IoU 2/4; class 1: TP 2, label 2, pred 3 -> 2/3;
    # class 2: TP 1, label 2, pred 1 -> 1/2; class 3 only predicted on an ignored pixel -> empty union, left out.
    assert np.allclose(m["iou"][:3], [0.5, 2 / 3, 0.5]) and np.isnan(m["iou"][3])
    assert m["mIoU"] == pytest.approx(100 * (0.5 + 2 / 3 + 0.5) / 3)
    assert m["aAcc"] == pytest.approx(100 * 5 / 7)
    assert m["mAcc"] == pytest.approx(100 * (2 / 3 + 1 + 0.5) / 3)
    assert (m["n_pixels"], m["n_classes_scored"], m["n_classes_in_labels"]) == (7, 3, 3)
    # A class predicted on a labelled pixel but absent from the labels scores IoU 0 (and counts).
    conf2 = torch.zeros(25, dtype=torch.int64)
    S.confusion_update(conf2, torch.tensor([0, 4]), torch.tensor([0, 0]), 5)
    m2 = S.seg_metrics(conf2.view(5, 5).numpy())
    assert m2["iou"][4] == 0.0 and m2["mIoU"] == pytest.approx(25.0) and m2["mAcc"] == pytest.approx(50.0)
    # Perfect predictions.
    y = torch.randint(0, 6, (500,), generator=torch.Generator().manual_seed(1))
    conf3 = torch.zeros(36, dtype=torch.int64)
    S.confusion_update(conf3, y, y, 6)
    m3 = S.seg_metrics(conf3.view(6, 6).numpy())
    assert m3["mIoU"] == pytest.approx(100.0) and m3["aAcc"] == pytest.approx(100.0) and m3["mAcc"] == pytest.approx(100.0)


# ---------------------------------------------------------------- training crops
def _coord_image(h, w):
    """Image whose red and green channels hold the row and column; label = a function of position."""
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    img = np.stack([yy, xx, np.full_like(yy, 7)], axis=-1).astype(np.uint8)
    lab = (((yy // 7) + (xx // 5)) % 4 + 1).astype(np.uint8)
    return img, lab


def test_crop_and_flip_keep_image_and_label_aligned():
    lut = S.label_lut(4)
    img, lab = _coord_image(60, 90)
    for uy, ux, flip in [(0.0, 0.0, False), (0.999, 0.999, False), (0.3, 0.7, True), (0.5, 0.1, True)]:
        ci, cl = S.crop_flip(img, lab, 56, uy, ux, flip, lut)
        assert ci.shape == (3, 56, 56) and ci.dtype == torch.uint8 and cl.shape == (56, 56) and cl.dtype == torch.uint8
        ys, xs = ci[0].numpy().astype(int), ci[1].numpy().astype(int)
        assert np.array_equal(cl.numpy(), lut[lab[ys, xs]])            # every label sits on its own pixel
        if flip:
            assert (np.diff(xs, axis=1) == -1).all()                    # columns run right to left
        else:
            assert (np.diff(xs, axis=1) == 1).all()
    ci, _ = S.crop_flip(img, lab, 56, 0.0, 0.0, False, lut)
    assert ci[0, 0, 0] == 0 and ci[1, 0, 0] == 0
    ci, _ = S.crop_flip(img, lab, 56, 0.999, 0.999, False, lut)
    assert ci[0, -1, -1] == 59 and ci[1, -1, -1] == 89                 # bottom-right corner reachable
    # An image smaller than the crop is padded with ignored labels.
    small_img, small_lab = _coord_image(40, 70)
    ci, cl = S.crop_flip(small_img, small_lab, 56, 0.5, 0.5, False, lut)
    assert (cl[40:] == 255).all() and (ci[:, 40:] == 0).all() and (cl[:40] != 255).all()


def test_schedule_is_seeded_and_a_prefix_of_longer_runs():
    a = S.make_schedule(10, 35, seed=0)
    for e in range(3):                                                  # each epoch is a permutation
        assert sorted(a["index"][10 * e:10 * e + 10].tolist()) == list(range(10))
    b = S.make_schedule(10, 20, seed=0)
    for k in ("index", "uy", "ux", "flip"):
        assert np.array_equal(a[k][:20], b[k])
    c = S.make_schedule(10, 35, seed=1)
    assert not np.array_equal(a["index"], c["index"]) and not np.array_equal(a["uy"], c["uy"])


# ---------------------------------------------------------------- sliding-window evaluation
def _coverage(h, w, crop, wins):
    cover = np.zeros((h, w), dtype=np.int64)
    for y, x in wins:
        assert 0 <= y <= h - crop and 0 <= x <= w - crop
        cover[y:y + crop, x:x + crop] += 1
    return cover


def test_sliding_window_evaluation_covers_every_pixel_exactly_once():
    # A tiling (stride = crop, sizes multiples of the crop): every pixel in exactly one window.
    for h, w in [(448, 448), (448, 896), (896, 1344)]:
        wins = S.sliding_windows(h, w, 448, 448)
        assert len(wins) == (h // 448) * (w // 448)
        assert (_coverage(h, w, 448, wins) == 1).all()
    # The overlapping layout used for evaluation: every pixel covered, no duplicate windows, the last
    # row and column touch the bottom and right edges.
    for h, w, crop, stride in [(448, 597, 448, 298), (448, 1500, 448, 298), (1000, 448, 448, 298), (448, 449, 448, 298),
                               (56, 70, 56, 42), (56, 56, 56, 42), (70, 140, 56, 14)]:
        wins = S.sliding_windows(h, w, crop, stride)
        assert len(set(wins)) == len(wins) and wins[0] == (0, 0) and (h - crop, w - crop) in wins
        assert (_coverage(h, w, crop, wins) >= 1).all()
    with pytest.raises(ValueError):
        S.sliding_windows(448, 600, 448, 449)                            # a stride above the crop leaves gaps
    # After averaging, every pixel carries total weight exactly one: an identity "model" returns the input
    # unchanged, including an image smaller than the window (padded, then cropped back).
    g = torch.Generator().manual_seed(0)
    for h, w, stride in [(56, 70, 42), (56, 56, 42), (70, 140, 14), (40, 70, 42), (112, 168, 56)]:
        x = torch.randn(1, 3, h, w, generator=g)
        out, cnt = S.slide_logits(lambda xb: xb, x, 56, stride, 3, batch=2)
        assert out.shape == x.shape and torch.allclose(out, x, rtol=1e-6, atol=1e-6)
        ph, pw = max(h, 56), max(w, 56)
        ref = _coverage(ph, pw, 56, S.sliding_windows(ph, pw, 56, stride))[:h, :w]
        assert np.array_equal(cnt.numpy(), ref)
    out, cnt = S.slide_logits(lambda xb: xb, torch.randn(1, 3, 112, 168, generator=g), 56, 56, 3)
    assert (cnt == 1).all()                                              # stride = crop: each pixel once


# ---------------------------------------------------------------- the head learns
def _avgpool_features(x_uint8):
    """Stand-in backbone: the mean colour of every 14 x 14 patch, [B, 3, H/14, W/14]."""
    return F.avg_pool2d(x_uint8.float() / 255.0, 14)


def test_head_trains_on_separable_synthetic_problem():
    C = 4
    lut = S.label_lut(C)
    tr_i, tr_l = S.synthetic_seg_dataset(24, C, [(56, 56)], seed=0, block=28, noise=6.0)
    va_i, va_l = S.synthetic_seg_dataset(6, C, [(56, 70), (70, 56), (56, 56)], seed=5, block=14, noise=6.0)
    train, val = S.RaggedStore.from_arrays(tr_i, tr_l), S.RaggedStore.from_arrays(va_i, va_l)
    cfg = S.TrainCfg(iters=150, batch=8, lr=0.05, weight_decay=1e-4, warmup=5, log_every=50, ckpt_every=1000)
    untrained = S.LinearSegHead(3, C, bn=True, seed=0)
    base = S.evaluate(untrained, _avgpool_features, val, 56, 14, lut, C, "cpu", eval_batch=4)
    head = S.LinearSegHead(3, C, bn=True, seed=0)
    res = S.train_head(head, _avgpool_features, train, 56, lut, cfg, "cpu", seed=0)
    assert res["history"][-1]["loss"] < 0.5 * res["history"][0]["loss"] and res["start_iter"] == 0
    ev = S.evaluate(head, _avgpool_features, val, 56, 14, lut, C, "cpu", eval_batch=4)
    assert ev["mIoU"] > 90.0 and ev["aAcc"] > 95.0, ev
    assert base["mIoU"] < 50.0
    assert ev["n_images"] == 6 and ev["confusion"].sum() == sum(int((lut[lb] != 255).sum()) for lb in va_l)


def test_training_resumes_from_checkpoint_identically(tmp_path):
    C = 4
    lut = S.label_lut(C)
    imgs, labs = S.synthetic_seg_dataset(10, C, [(56, 70), (60, 56)], seed=2)
    store = S.RaggedStore.from_arrays(imgs, labs)
    cfg = S.TrainCfg(iters=12, batch=4, lr=0.02, warmup=2, log_every=3, ckpt_every=5)
    settings = {"run": "test", "seed": 0}
    ref = S.LinearSegHead(3, C, seed=0)
    r_ref = S.train_head(ref, _avgpool_features, store, 56, lut, cfg, "cpu", seed=0)

    calls = {"n": 0}

    def crashing(x):
        calls["n"] += 1
        if calls["n"] == 8:
            raise KeyboardInterrupt("simulated crash at iteration 8")
        return _avgpool_features(x)

    ck = tmp_path / "ckpt.pt"
    with pytest.raises(KeyboardInterrupt):
        S.train_head(S.LinearSegHead(3, C, seed=0), crashing, store, 56, lut, cfg, "cpu", seed=0, ckpt_path=ck,
                     settings=settings)
    assert ck.exists() and torch.load(ck, weights_only=True)["t"] == 5
    head = S.LinearSegHead(3, C, seed=0)
    r = S.train_head(head, _avgpool_features, store, 56, lut, cfg, "cpu", seed=0, ckpt_path=ck, settings=settings)
    assert r["start_iter"] == 5
    for (n1, t1), (n2, t2) in zip(ref.state_dict().items(), head.state_dict().items()):
        assert n1 == n2 and torch.equal(t1, t2), n1
    assert [h["loss"] for h in r["history"]] == [h["loss"] for h in r_ref["history"]]
    # A checkpoint made with other settings is not used (the run above left one at iteration 10).
    assert torch.load(ck, weights_only=True)["t"] == 10
    r2 = S.train_head(S.LinearSegHead(3, C, seed=0), _avgpool_features, store, 56, lut, cfg, "cpu", seed=0,
                      ckpt_path=ck, settings={"run": "other"})
    assert r2["start_iter"] == 0


# ---------------------------------------------------------------- data cache
def test_cache_decodes_parquet_once_with_raw_labels(tmp_path):
    sizes = ((56, 70), (64, 56), (60, 84), (56, 56))
    files = S.write_tiny_seg_parquet(tmp_path / "pq", num_classes=4, n_train=8, n_val=4, seed=3, sizes=sizes)
    entry = dict(S.TINY_ENTRY)
    data = S.ensure_seg_cache(entry, tmp_path / "cache", 56, workers=1, files=files)
    tr, va = data["train"], data["val"]
    assert len(tr) == 8 and len(va) == 4 and data["meta"]["complete"] and data["meta"]["counts"] == {"train": 8, "val": 4}
    ref_tr = S.synthetic_seg_dataset(8, 4, sizes, seed=3)
    ref_va = S.synthetic_seg_dataset(4, 4, sizes, seed=4)
    for i in range(8):
        im, lb = tr.image(i), tr.label(i)
        assert min(im.shape[:2]) == 56 and lb.shape == im.shape[:2]          # train label at the image size
        assert set(np.unique(lb)) <= set(np.unique(ref_tr[1][i]))             # nearest: no new values
        if ref_tr[1][i].shape == (56, 56):
            assert np.array_equal(lb, ref_tr[1][i]) and np.array_equal(im, ref_tr[0][i])
    for i in range(4):
        assert min(va.image(i).shape[:2]) == 56
        assert np.array_equal(va.label(i), ref_va[1][i])                      # validation labels: original size
        assert tuple(va.index["orig_h"][[i]]) + tuple(va.index["orig_w"][[i]]) == ref_va[1][i].shape
    assert data["meta"]["decode"]["val"]["label_resolution"] == "original"
    # The store pickles without its memmaps (DataLoader workers map the files themselves).
    blob = pickle.dumps(tr)
    assert len(blob) < 20000
    assert np.array_equal(pickle.loads(blob).image(2), tr.image(2))
    # A second call reuses the cache.
    mt = (tmp_path / "cache" / "tiny_seg" / "train_images.u8").stat().st_mtime_ns
    S.ensure_seg_cache(entry, tmp_path / "cache", 56, workers=1, files=files)
    assert (tmp_path / "cache" / "tiny_seg" / "train_images.u8").stat().st_mtime_ns == mt
    # Label values above the class count are refused.
    with pytest.raises(ValueError, match="above"):
        S.ensure_seg_cache({**entry, "num_classes": 2, "cache_name": "x"}, tmp_path / "cache", 56, workers=1, files=files)


# ---------------------------------------------------------------- features
class _MockDinoAdapter(Dinov2Adapter):
    """Dinov2Adapter around the DINOv2 mock of tests/test_fold.py (no hub download)."""

    def __init__(self, model, img_size):
        ViTAdapter.__init__(self, model, img_size=img_size, patch_size=model.patch_size,
                            n_prefix_tokens=1 + model.num_register_tokens)


def _structure(adapter, seed=0):
    d, h, L = adapter.embed_dim, adapter.mlp_hidden, adapter.depth
    H, dh = attn_dims(adapter.model.blocks[0])
    A = random_involution_fixing_ones(d, d // 2 + 4, seed=seed + 3)
    perms = [random_pairing(h, h // 3, seed=seed + 10 + b) for b in range(L)]
    maps = [random_head_structure(H, dh, 1 + b % 2, seed=seed + 20 + b) for b in range(L)]
    return A, perms, maps


def test_feature_map_is_last_layer_patch_tokens_after_final_norm():
    ad = TimmTinyAdapter(img_size=56, seed=0)
    x = torch.randint(0, 256, (2, 3, 56, 56), dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    f = S.make_feature_fn(ad, ad.model, "cpu", amp="bf16")(x)                # CPU: no autocast
    assert f.shape == (2, ad.embed_dim, 4, 4) and f.dtype == torch.float32
    with torch.no_grad():
        t = ad.model.norm(ad.forward_tokens(normalize_for(ad, x)))
    assert torch.equal(f[:, :, 1, 2], t[:, ad.n_prefix_tokens + 1 * 4 + 2])   # row-major patch grid
    # DINOv2 layout with registers: CLS and the 4 registers are dropped, at a larger grid too.
    from test_fold import make_mock

    mock = make_mock(num_register_tokens=4, img_size=70)
    mad = _MockDinoAdapter(mock, 84)
    xm = torch.randint(0, 256, (2, 3, 84, 84), dtype=torch.uint8, generator=torch.Generator().manual_seed(1))
    fm = S.make_feature_fn(mad, mock, "cpu")(xm)
    with torch.no_grad():
        tm = mock.norm(mad.forward_tokens(normalize_for(mad, xm)))
    assert fm.shape == (2, 48, 6, 6) and mad.n_prefix_tokens == 5
    assert torch.allclose(fm, tm[:, 5:].float().transpose(1, 2).reshape(2, 48, 6, 6))


# ---------------------------------------------------------------- V-exact at the segmentation resolution
def test_vexact_rebuilt_at_segmentation_resolution_is_exact_and_the_low_resolution_build_is_not():
    from test_fold import make_mock

    mock = make_mock(num_register_tokens=4, img_size=70)       # positions stored for 5 x 5 (like 37 x 37)
    ad_lo = _MockDinoAdapter(mock, 56)                         # stands for 224 px: 4 x 4 grid
    ad_seg = S.adapter_at(ad_lo, 84)                           # stands for 448 px: 6 x 6 grid
    assert ad_seg.grid_hw == (6, 6) and ad_lo.grid_hw == (4, 4) and ad_seg.model is mock
    A, perms, maps = _structure(ad_lo, seed=7)
    k, reg_perm = 2, [1, 0, 2, 3]
    g = torch.Generator().manual_seed(3)
    x_seg = torch.randn(2, 3, 84, 84, dtype=torch.float64, generator=g)
    x_lo = torch.randn(2, 3, 56, 56, dtype=torch.float64, generator=g)
    m_seg, info = build_vexact(ad_seg, A, list(range(k)), perms[:k], maps[:k], reg_perm=reg_perm)
    assert info["pos_materialised"] and m_seg.pos_embed.shape[1] == 1 + 36
    ok = S.span_exactness(ad_seg, m_seg, A, reg_perm, x_seg, k)
    assert ok["max_rel_err_inside_span"] < 1e-10 and ok["pass"]
    # The build for the smaller grid is exact at its own size but not after DINOv2 re-interpolates it.
    m_lo, _ = build_vexact(ad_lo, A, list(range(k)), perms[:k], maps[:k], reg_perm=reg_perm)
    assert S.span_exactness(ad_lo, m_lo, A, reg_perm, x_lo, k)["max_rel_err_inside_span"] < 1e-10
    bad = S.span_exactness(ad_seg, m_lo, A, reg_perm, x_seg, k)
    assert bad["max_rel_err_inside_span"] > 1e-6 and not bad["pass"]
    with pytest.raises(ValueError):
        S.adapter_at(ad_lo, 80)                                  # not a multiple of the patch size

    # vexact_checks, as main() calls it: the contrast model is built at the contrast size and run at the
    # segmentation size (a stand-in for the VariantFactory with the same structure).
    from types import SimpleNamespace

    built = []

    def make_variant(name, kk, ad, Sx, ops):
        built.append((name, kk, ad.img_size))
        return build_vexact(ad, Sx["A_star"][kk], list(range(kk)), perms[:kk], maps[:kk], reg_perm=Sx["reg_perm"][kk])[0]

    Sx = {"A_star": {k: A}, "reg_perm": {k: np.asarray(reg_perm)}}
    fake = SimpleNamespace(m1_structure=lambda kk: (Sx, None, {}), m1=lambda: SimpleNamespace(make_variant=make_variant))
    chk = S.vexact_checks(fake, ad_seg, m_seg, SimpleNamespace(k=k), x_seg, contrast_px=56)
    assert built == [("vexact", k, 56)] and chk["built_px"] == chk["eval_px"] == 84
    assert chk["at_build_resolution"]["pass"] and chk["contrast"]["built_px"] == 56
    assert chk["contrast"]["max_rel_err_inside_span"] == pytest.approx(bad["max_rel_err_inside_span"], rel=1e-6)
    assert "contrast" not in S.vexact_checks(fake, ad_seg, m_seg, SimpleNamespace(k=k), x_seg, contrast_px=None)


# ---------------------------------------------------------------- CLI
def _tiny_files(tmp_path):
    from test_tier2 import _tiny_m1_structure, _tiny_operators

    ops = _tiny_operators(tmp_path / "ops.npz")
    return ops, _tiny_m1_structure(tmp_path / "m1_structure.npz", ops)


FAST = ["--iters", "6", "--log-every", "3", "--ckpt-every", "4"]


def test_cli_tiny_resume_skips_finished_variant(tmp_path, monkeypatch):
    ops, _ = _tiny_files(tmp_path)
    out = tmp_path / "t2_seg"
    args = ["--tiny", "--variants", "original,mirror_mlp_k2", "--out", str(out), "--operators", str(ops), *FAST]
    res = S.main(args)
    assert [r["variant"] for r in res["rows"]] == ["original", "mirror_mlp_k2"]
    vdir = out / "seg" / "tiny"
    for v in ("original", "mirror_mlp_k2"):
        for suffix in (".json", "_head.pt", "_confusion.npy", "_train_log.csv"):
            assert (vdir / f"{v}{suffix}").is_file()
        assert not (vdir / f"{v}_ckpt.pt").exists()
    import pandas as pd

    df = pd.read_csv(out / "t2_ade20k.csv")
    assert list(df["variant"]) == ["original", "mirror_mlp_k2"]
    for col in ("mIoU", "aAcc", "mAcc", "iterations", "seed", "git_hash", "t_train_s", "t_eval_s"):
        assert col in df.columns and df[col].notna().all()
    assert (df["iterations"] == 6).all() and (df["seed"] == 0).all()
    assert "mIoU" in (out / "T2_ADE20K.md").read_text(encoding="utf-8")
    row = json.loads((vdir / "original.json").read_text(encoding="utf-8"))
    assert len(row["metrics"]["per_class_iou"]) == 4 and row["settings"]["iters"] == 6

    mt = {p: p.stat().st_mtime_ns for p in vdir.iterdir()}

    def boom(*a, **k):
        raise AssertionError("recomputed")

    monkeypatch.setattr(S, "train_head", boom)
    monkeypatch.setattr(S, "evaluate", boom)
    monkeypatch.setattr(S.T2.VariantFactory, "build", boom)
    res2 = S.main(args)                                            # both variants skipped
    assert [r["mIoU"] for r in res2["rows"]] == [r["mIoU"] for r in res["rows"]]
    assert all(p.stat().st_mtime_ns == t for p, t in mt.items())
    with pytest.raises(AssertionError, match="recomputed"):        # another setting: not reused
        S.main(["--tiny", "--variants", "original", "--out", str(out), "--operators", str(ops), "--iters", "7",
                "--log-every", "3"])
    # Only the result file lost: the trained head is reused and only the evaluation reruns.
    monkeypatch.undo()
    monkeypatch.setattr(S, "train_head", boom)
    (vdir / "original.json").unlink()
    res3 = S.main(args)
    assert [r["mIoU"] for r in res3["rows"]] == [r["mIoU"] for r in res["rows"]]


def test_cli_tiny_vexact_checks_and_other_families(tmp_path):
    ops, m1s = _tiny_files(tmp_path)
    out = tmp_path / "t2_seg"
    names = ["original", "vexact_k2", "vlocal_k2", "svd_mlp_k2", "random_k2"]
    res = S.main(["--tiny", "--variants", ",".join(names), "--out", str(out), "--operators", str(ops),
                  "--m1-structure", str(m1s), *FAST])
    rows = {r["variant"]: r for r in res["rows"]}
    assert list(rows) == names
    assert all(np.isfinite(r["mIoU"]) and 0 <= r["mIoU"] <= 100 for r in rows.values())
    vx = rows["vexact_k2"]
    assert vx["vexact_built_px"] == 56 and vx["vexact_max_rel_err"] < 1e-4      # fp32, tiny timm ViT
    js = json.loads((out / "seg" / "tiny" / "vexact_k2.json").read_text(encoding="utf-8"))
    assert js["vexact_checks"]["at_build_resolution"]["pass"]
    assert "skipped" in js["vexact_checks"]["contrast"]                          # timm: fixed position grid
    assert rows["original"]["vexact_max_rel_err"] is None
    assert rows["svd_mlp_k2"]["converted_params_kept"] < rows["svd_mlp_k2"]["converted_params_dense"]
    assert "rebuilt at 56 px" in (out / "T2_ADE20K.md").read_text(encoding="utf-8")
