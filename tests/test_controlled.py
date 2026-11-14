"""Controlled training (mirror/controlled.py, scripts/train_controlled.py, scripts/build_imagenet100.py).

CPU only: synthetic data, a 3-block test ViT at the real 64 px / patch 8 layout (65 tokens).
"""
import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

from mirror import controlled as C  # noqa: E402


@pytest.fixture(scope="module")
def build_script():
    # Imported by its plain name so spawned decoding workers (Windows) can import it too.
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        yield importlib.import_module("build_imagenet100")
    finally:
        sys.path.remove(str(ROOT / "scripts"))


def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_cli", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------------
# Class list
# ----------------------------------------------------------------------------------------------
def test_class_list_has_100_unique_imagenet_ids():
    text = C.CLASS_LIST.read_text(encoding="ascii")
    assert "github.com/HobbitLong/CMC" in text            # header names the source
    wnids = C.read_class_list()
    assert len(wnids) == 100 and len(set(wnids)) == 100
    order = C.imagenet1k_wnids()
    assert set(wnids) <= set(order)
    m = C.in100_label_map(wnids, order)
    assert list(m["hf_index"]) == sorted(m["hf_index"]) and len(set(m["hf_index"].tolist())) == 100
    assert m["wnids"] == sorted(wnids)                    # label order = sorted WordNet IDs
    assert (m["lut"] >= 0).sum() == 100
    assert sorted(m["lut"][m["lut"] >= 0].tolist()) == list(range(100))
    for lab, h in enumerate(m["hf_index"]):
        assert m["lut"][h] == lab and order[h] == m["wnids"][lab]


def test_ckpt_epochs_for_60_and_100():
    assert C.ckpt_epochs_for(100) == [0, 2, 5, 10, 25, 50, 100]
    assert C.ckpt_epochs_for(60) == [0, 2, 5, 10, 25, 50, 60]
    assert C.ckpt_epochs_for(3, [0, 1, 2]) == [0, 1, 2, 3]


# ----------------------------------------------------------------------------------------------
# Symmetry conventions with a marker image
# ----------------------------------------------------------------------------------------------
def _marker(size=64):
    """A 16-wide, 8-tall red bar in the top-left corner (asymmetric, so a transpose would show)."""
    x = torch.zeros(3, size, size, dtype=torch.uint8)
    x[0, 0:8, 0:16] = 255
    return x


def _bbox(img):
    ys, xs = torch.nonzero(img[0] > 127, as_tuple=True)
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def test_view_conventions_marker():
    x = _marker()
    S = 64
    expect = {   # (row min, row max, col min, col max) of the bar after the view
        "orig": (0, 7, 0, 15),
        "LR": (0, 7, S - 16, S - 1),            # top-right
        "UD": (S - 8, S - 1, 0, 15),            # bottom-left
        "LRUD": (S - 8, S - 1, S - 16, S - 1),  # bottom-right
        "R180": (S - 8, S - 1, S - 16, S - 1),
        "R90": (S - 16, S - 1, 0, 7),           # counter-clockwise: top-left corner -> bottom-left, bar upright
        "R270": (0, 15, S - 8, S - 1),          # clockwise: top-left -> top-right
    }
    for v, box in expect.items():
        assert _bbox(C.apply_view(x, v)) == box, v
    assert torch.equal(C.apply_view(C.apply_view(x, "R90"), "R270"), x)
    assert torch.equal(C.apply_view(C.apply_view(x, "R90"), "R90"), C.apply_view(x, "R180"))
    assert torch.equal(C.apply_view(x, None), x)


def test_view_token_maps_match_project_convention():
    """Token (r, c) of the original lands where PROJECT_NEXT_PHASE.md 4.3 says, with its patch content
    transformed by the same action."""
    G, P = 8, 8
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 256, (3, G * P, G * P), generator=g, dtype=torch.uint8)

    def patches(img):   # [G*G, 3, P, P], row-major tokens
        return img.reshape(3, G, P, G, P).permute(1, 3, 0, 2, 4).reshape(G * G, 3, P, P)

    for v in ("LR", "UD", "LRUD", "R90", "R180", "R270"):
        perm = C.view_token_permutation(v, G)
        po, pt = patches(x), patches(C.apply_view(x, v))
        for i in range(G * G):
            assert torch.equal(pt[perm[i]], C.apply_view(po[i], v)), (v, i)
    r, c = 2, 5
    assert C.view_token_permutation("R90", G)[r * G + c] == (G - 1 - c) * G + r
    assert C.view_token_permutation("LR", G)[r * G + c] == r * G + (G - 1 - c)
    assert C.view_token_permutation("UD", G)[r * G + c] == (G - 1 - r) * G + c


def test_sampling_statistics_per_condition():
    n = 40000
    for cond, spec in C.CONDITIONS.items():
        s = C.sample_symmetry(n, cond, torch.Generator().manual_seed(1))
        lr, ud, k = s[:, 0].double(), s[:, 1].double(), s[:, 2]
        assert abs(lr.mean().item() - spec["p_lr"]) < 0.01, cond
        assert abs(ud.mean().item() - spec["p_ud"]) < 0.01, cond
        if spec["rot90"]:
            freq = torch.bincount(k, minlength=4).double() / n
            assert torch.all((freq - 0.25).abs() < 0.01), cond
        else:
            assert int(k.abs().sum()) == 0, cond
        if cond == "LRUD":   # independent flips: each of the four combinations about a quarter
            combo = torch.bincount((2 * s[:, 0] + s[:, 1]), minlength=4).double() / n
            assert torch.all((combo - 0.25).abs() < 0.01)
    # Every condition makes the same draws: same crops, and LR flags shared by LR and LRUD.
    c1, s1 = C.epoch_aug_params(500, 3, 7, "LR")
    c2, s2 = C.epoch_aug_params(500, 3, 7, "LRUD")
    c3, _ = C.epoch_aug_params(500, 3, 7, "none")
    assert torch.equal(c1, c2) and torch.equal(c1, c3) and torch.equal(s1[:, 0], s2[:, 0])
    assert torch.all(c1[:, 4] == 0)                     # the crop step never flips
    area = c1[:, 2] * c1[:, 3] / (72 * 72)
    assert area.min() > 0.33 and area.max() <= 1.0


def test_augment_train_marker_every_condition():
    """Full-image crop, then the condition's symmetry: the bar ends up where each image's draw says."""
    S, out = 72, 64
    img = torch.zeros(S, S, 3, dtype=torch.uint8)
    img[0:9, 0:18, 0] = 255
    for cond in C.CONDITIONS:
        n = 64
        x = img.unsqueeze(0).repeat(n, 1, 1, 1)
        crops = torch.tensor([[0.0, 0.0, S, S, 0.0]], dtype=torch.float64).repeat(n, 1)
        sym = C.sample_symmetry(n, cond, torch.Generator().manual_seed(5))
        y = C.augment_train(x, crops, sym, out, cond)
        base = C.augment_train(x[:1], crops[:1], torch.zeros(1, 3, dtype=torch.int64), out, None)[0]
        assert y.shape == (n, 3, out, out) and y.dtype == torch.float32
        names = {(0, 0, 0): "orig", (1, 0, 0): "LR", (0, 1, 0): "UD", (1, 1, 0): "LRUD",
                 (0, 0, 1): "R90", (0, 0, 2): "R180", (0, 0, 3): "R270"}
        seen = set()
        for i in range(n):
            v = names[tuple(int(t) for t in sym[i])]
            seen.add(v)
            assert torch.equal(y[i], C.apply_view(base, v)), (cond, i, v)
            red = y[i, 0] > y[i, 0].mean()                       # where the bar is
            top = red[: out // 2].sum() > red[out // 2:].sum()
            left = red[:, : out // 2].sum() > red[:, out // 2:].sum()
            assert (top, left) == {"orig": (True, True), "LR": (True, False), "UD": (False, True),
                                   "LRUD": (False, False), "R90": (False, True), "R180": (False, False),
                                   "R270": (True, False)}[v], (cond, v)
        allowed = {"none": {"orig"}, "LR": {"orig", "LR"}, "UD": {"orig", "UD"},
                   "LRUD": {"orig", "LR", "UD", "LRUD"}, "R90": {"orig", "R90", "R180", "R270"},
                   "UD_p0.1": {"orig", "UD"}, "UD_p0.25": {"orig", "UD"}}[cond]
        assert seen <= allowed, (cond, seen)


# ----------------------------------------------------------------------------------------------
# Models and adapter
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("arch,dim,heads,n_params", [("vit_s8", 384, 6, 21432292), ("vit_ti8", 192, 3, 5407780)])
def test_real_architectures(arch, dim, heads, n_params):
    torch.manual_seed(0)
    model = C.build_model(arch, 100)
    assert C.count_params(model) == n_params
    ad = C.ControlledAdapter(model, arch)
    assert ad.grid_hw == (8, 8) and ad.n_prefix_tokens == 1 and ad.n_tokens == 65
    assert ad.depth == 12 and ad.embed_dim == dim and ad.mlp_hidden == 4 * dim
    assert model.blocks[0].attn.num_heads == heads
    seen = []
    h = ad.blocks[0].register_forward_hook(lambda m, a, o: seen.append(tuple(o.shape)))
    with torch.no_grad():
        ad.cls_feature(torch.randn(2, 3, 64, 64))
    h.remove()
    assert seen == [(2, 65, dim)]
    assert tuple(ad.pos_embed_grid().shape) == (8, 8, dim)


# ----------------------------------------------------------------------------------------------
# Training: save, resume, reproduce exactly
# ----------------------------------------------------------------------------------------------
def _cfg(tmp, run, **kw):
    base = dict(exp="TEST", arch="vit_test8", condition="LRUD", seed=3, epochs=3, ckpt_epochs=(0, 1, 2, 3),
                batch_size=16, lr=3.2e-2, warmup_epochs=1.0, device="cpu", amp="none", compile="off",
                eval_batch=16, save_every_min=0.0, log_every=0, cache_dir=str(tmp / "cache"),
                run_dir=str(tmp / run))
    base.update(kw)
    return C.TrainConfig(**base)


def _csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("controlled")
    C.write_synthetic_cache(tmp / "cache", num_classes=10, train_per_class=8, val_per_class=4, seed=0)
    out_a = C.train(_cfg(tmp, "A"))                                  # uninterrupted
    stop1 = C.train(_cfg(tmp, "B", stop_after_steps=7))              # 5 steps per epoch: mid-epoch 2
    stop2 = C.train(_cfg(tmp, "B", stop_after_steps=10))             # resume, stop at an epoch boundary
    out_b = C.train(_cfg(tmp, "B"))                                  # resume to the end
    return {"tmp": tmp, "A": tmp / "A", "B": tmp / "B", "out_a": out_a, "stops": (stop1, stop2), "out_b": out_b}


def test_resume_reproduces_uninterrupted_run(trained):
    A, B = trained["A"], trained["B"]
    assert trained["stops"][0]["stopped"] and trained["stops"][0]["step"] == 7
    assert trained["stops"][1]["stopped"] and trained["stops"][1]["step"] == 10
    assert not trained["out_b"]["stopped"]
    la = torch.load(A / "last.pt", weights_only=True)
    lb = torch.load(B / "last.pt", weights_only=True)
    assert la["step"] == lb["step"] == 15
    for k in la["model"]:
        assert torch.equal(la["model"][k], lb["model"][k]), k
    sa, sb = la["optimizer"]["state"], lb["optimizer"]["state"]
    assert sa.keys() == sb.keys()
    for i in sa:
        for k in sa[i]:
            assert torch.equal(sa[i][k], sb[i][k]), (i, k)
    ea = [e for e, _ in C.list_checkpoints(A)]
    assert ea == [e for e, _ in C.list_checkpoints(B)] == [0, 1, 2, 3]
    for e, p in C.list_checkpoints(A):
        ca, cb = torch.load(p, weights_only=True), torch.load(B / p.name, weights_only=True)
        assert ca["val"] == cb["val"] and ca["epoch"] == e
        assert all(torch.equal(ca["model"][k], cb["model"][k]) for k in ca["model"])
    cols = ["epoch", "step", "lr_end", "train_loss", "train_top1", "grad_norm_mean"]
    ta, tb = _csv(A / "train_log.csv"), _csv(B / "train_log.csv")
    assert [[r[c] for c in cols] for r in ta] == [[r[c] for c in cols] for r in tb]
    assert [r["resumed"] for r in tb] == ["0", "1", "0"]
    vcols = ["epoch", "step", "top1_orig", "top1_LR", "top1_UD", "top1_R90", "n_val"]
    va, vb = _csv(A / "val_acc.csv"), _csv(B / "val_acc.csv")
    assert [[r[c] for c in vcols] for r in va] == [[r[c] for c in vcols] for r in vb]
    assert [r["epoch"] for r in va] == ["0", "1", "2", "3"]
    # Training changed the weights, and the run's record is complete.
    c0 = torch.load(A / "ckpt_e000.pt", weights_only=True)["model"]
    assert not torch.equal(c0["blocks.0.mlp.fc1.weight"], la["model"]["blocks.0.mlp.fc1.weight"])
    meta = json.loads((A / "meta.json").read_text())
    assert meta["extra"]["status"] == "complete" and meta["experiment"] == "TEST"
    assert meta["extra"]["timing"]["epochs_run"] == 3 and "projected_minutes_60" in meta["extra"]["timing"]


def test_resume_from_older_last_state(trained):
    """A hard kill can leave a last.pt older than the logs and checkpoints (it is not written every
    epoch). Resuming drops the later log rows, redoes the work and ends identical to the clean run."""
    import shutil

    tmp = trained["tmp"]
    C.train(_cfg(tmp, "D", stop_after_steps=5))                       # end of epoch 1, saved
    shutil.copy(tmp / "D" / "last.pt", tmp / "D_last_e1.pt")
    C.train(_cfg(tmp, "D", stop_after_steps=12))                      # epoch 2 logged and checkpointed
    assert [r["epoch"] for r in _csv(tmp / "D" / "train_log.csv")] == ["1", "2"]
    shutil.copy(tmp / "D_last_e1.pt", tmp / "D" / "last.pt")          # the later state is lost
    C.train(_cfg(tmp, "D"))
    la = torch.load(trained["A"] / "last.pt", weights_only=True)["model"]
    ld = torch.load(tmp / "D" / "last.pt", weights_only=True)["model"]
    assert all(torch.equal(la[k], ld[k]) for k in la)
    assert [r["epoch"] for r in _csv(tmp / "D" / "train_log.csv")] == ["1", "2", "3"]
    assert [r["epoch"] for r in _csv(tmp / "D" / "val_acc.csv")] == ["0", "1", "2", "3"]
    cols = ["train_loss", "train_top1", "grad_norm_mean"]
    assert ([[r[c] for c in cols] for r in _csv(trained["A"] / "train_log.csv")]
            == [[r[c] for c in cols] for r in _csv(tmp / "D" / "train_log.csv")])


def test_finished_run_is_left_alone_and_settings_are_checked(trained):
    tmp = trained["tmp"]
    before = (trained["A"] / "last.pt").stat().st_mtime_ns
    out = C.train(_cfg(tmp, "A"))
    assert not out["stopped"] and (trained["A"] / "last.pt").stat().st_mtime_ns == before
    with pytest.raises(ValueError, match="different settings"):
        C.train(_cfg(tmp, "A", lr=1e-2))


def test_condition_changes_only_the_symmetry(trained):
    """Same seed, other condition: same initial weights (epoch 0 checkpoint)."""
    tmp = trained["tmp"]
    C.train(_cfg(tmp, "C", condition="none", epochs=1, ckpt_epochs=(0,)))
    c_a = torch.load(trained["A"] / "ckpt_e000.pt", weights_only=True)["model"]
    c_c = torch.load(tmp / "C" / "ckpt_e000.pt", weights_only=True)["model"]
    assert all(torch.equal(c_a[k], c_c[k]) for k in c_a)


# ----------------------------------------------------------------------------------------------
# Interface for the measurement code
# ----------------------------------------------------------------------------------------------
def test_checkpoint_adapter_with_knn_and_hooks(trained):
    from mirror.adapters import ViTAdapter
    from mirror.knn import extract_features
    from mirror.stats import run_paired

    cache = trained["tmp"] / "cache"
    ad = C.load_checkpoint_adapter(trained["A"] / "ckpt_e003.pt", "cpu")
    assert isinstance(ad, ViTAdapter)
    assert ad.grid_hw == (8, 8) and ad.n_prefix_tokens == 1 and ad.n_tokens == 65 and ad.depth == 3
    assert ad.info["epoch"] == 3 and ad.info["condition"] == "LRUD" and ad.num_classes == 10
    assert not ad.model.training and not any(p.requires_grad for p in ad.model.parameters())

    loader = C.val_loader(None, batch_size=16, cache_dir=cache)
    feats, labels = extract_features(ad, loader, "cpu")
    assert feats.shape == (40, ad.embed_dim)
    assert torch.allclose(feats.norm(dim=1), torch.ones(40), atol=1e-5)
    assert torch.equal(labels, torch.from_numpy(np.load(cache / C.VAL_LABELS)))

    imgs, _ = next(iter(loader))
    assert imgs.dtype == torch.uint8 and imgs.shape == (16, 3, 64, 64)
    x = (imgs.float() / 255 - torch.tensor(ad.mean).view(1, 3, 1, 1)) / torch.tensor(ad.std).view(1, 3, 1, 1)
    with torch.no_grad():
        assert torch.allclose(ad.logits(x), ad.model(x), atol=1e-5)
        assert torch.allclose(ad.cls_feature(x), ad.model.forward_features(x)[:, 0], atol=1e-5)

    shapes = []
    hooks = [b.register_forward_hook(lambda m, a, o: shapes.append(tuple(o.shape))) for b in ad.blocks]
    with torch.no_grad():
        ad.cls_feature(x)
    for h in hooks:
        h.remove()
    assert shapes == [(16, 65, ad.embed_dim)] * ad.depth

    layers = {}
    run_paired(ad, C.fit_loader(0, n=20, batch_size=10, cache_dir=cache), "cpu",
               residual_cb=lambda layer, X, Xt: layers.setdefault(layer, (X.shape, Xt.shape)))
    assert sorted(layers) == list(range(ad.depth + 1))
    assert all(s[0] == (10, 65, ad.embed_dim) and s[1] == s[0] for s in layers.values())


def test_loaders(trained):
    cache = trained["tmp"] / "cache"
    labels = np.load(cache / C.TRAIN_LABELS)
    r0, r1 = C.fit_rows(labels, 0, 20), C.fit_rows(labels, 1, 20)
    assert len(r0) == 20 and len(np.intersect1d(r0, r1)) == 0
    assert np.array_equal(r0, C.fit_rows(labels, 0, 20))
    assert np.array_equal(np.bincount(labels[r0], minlength=10), np.full(10, 2))
    assert len(C.fit_rows(labels, 0, 25)) == 25
    base = torch.cat([b for b, _ in C.val_loader(None, batch_size=8, cache_dir=cache)])
    raw = torch.from_numpy(np.load(cache / C.VAL_FILE))
    assert torch.equal(base, raw[:, 4:68, 4:68].permute(0, 3, 1, 2))
    for v in ("LR", "UD", "R90"):
        tv = torch.cat([b for b, _ in C.val_loader(v, batch_size=8, cache_dir=cache)])
        assert torch.equal(tv, C.apply_view(base, v)), v
    assert len(C.val_loader(None, cache_dir=cache, n=10).dataset) == 10


def test_train_script_tiny(tmp_path):
    mod = _load_script("train_controlled")
    out = mod.main(["--tiny", "--condition", "UD", "--out-root", str(tmp_path), "--epochs", "2"])
    run = tmp_path / "CORE-01" / "vit_test8_UD_s0"
    assert not out["stopped"]
    assert [e for e, _ in C.list_checkpoints(run)] == [0, 1, 2]
    for f in ("last.pt", "train_log.csv", "val_acc.csv", "meta.json", "last_done.flag"):
        assert (run / f).exists(), f
    meta = json.loads((run / "meta.json").read_text())
    assert meta["config"]["condition"] == "UD" and meta["seeds"]["train"] == 0
    assert meta["git"]["git_hash"]


# ----------------------------------------------------------------------------------------------
# ImageNet-100 build
# ----------------------------------------------------------------------------------------------
def test_decode_matches_torchvision(build_script):
    import io

    from PIL import Image
    from torchvision import transforms as T

    rng = np.random.default_rng(0)
    tf = T.Compose([T.Resize(72, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(72)])
    for w, h in [(500, 375), (375, 500), (300, 300), (73, 200), (50, 40), (999, 101), (72, 72)]:
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        ours = build_script.decode_resize_crop(buf.getvalue(), 72)
        ref = np.asarray(tf(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")), dtype=np.uint8)
        assert ours.shape == (72, 72, 3) and np.array_equal(ours, ref), (w, h)


def test_build_imagenet100_resumable(build_script, tmp_path):
    import pyarrow.parquet as pq

    bs = build_script
    lm = C.in100_label_map(C.read_class_list(), C.imagenet1k_wnids())
    src = bs.make_synthetic_shards(tmp_path / "src", lm["hf_index"], n_train_shards=3, rows_per_shard=40,
                                   n_val_rows=30)
    common = ["--source", "local", "--local-dir", str(src), "--no-strict", "--no-verify-classes"]
    # One go, with two decoding processes (the pool path).
    full = tmp_path / "full"
    bs.main(common + ["--cache-dir", str(full), "--workers", "2", "--download-workers", "2"])
    # Interrupted after one shard, then resumed.
    part = tmp_path / "part"
    out = bs.main(common + ["--cache-dir", str(part), "--workers", "1", "--max-shards", "1"])
    assert out == {"stopped": True}
    assert len(list((part / "in100_parts" / "train").glob("*.npz"))) == 1
    bs.main(common + ["--cache-dir", str(part), "--workers", "1"])
    for f in (C.TRAIN_FILE, C.VAL_FILE, C.TRAIN_LABELS, C.VAL_LABELS):
        assert np.array_equal(np.load(full / f), np.load(part / f)), f
    meta = json.loads((full / C.META_FILE).read_text())
    assert meta["extra"]["complete"] and meta["experiment"] == "IN100-BUILD"
    assert not (full / "in100_parts" / "train").exists()            # parts removed after assembly

    # Every stored image is the decoded source row, with the right label.
    d = C.load_in100(full)
    idx = np.load(full / C.INDEX_FILE)
    shards = sorted((src / "data").glob("train-*.parquet"))
    assert list(idx["train_shards"]) == [p.name for p in shards]
    assert len(d["train_labels"]) == len(idx["train_src"]) > 0
    tables = [pq.read_table(p) for p in shards]
    for i, (si, row) in enumerate(idx["train_src"]):
        rec = tables[si].slice(int(row), 1).to_pylist()[0]
        assert lm["lut"][rec["label"]] == d["train_labels"][i]
        assert np.array_equal(d["train_images"][i], bs.decode_resize_crop(rec["image"]["bytes"], 72))
    # Every subset image of the shards was taken, and nothing else.
    n_subset = sum(int((lm["lut"][t.column("label").to_numpy()] >= 0).sum()) for t in tables)
    assert n_subset == len(d["train_labels"])
    assert set(np.unique(d["val_labels"])) <= set(range(100))
    assert list(idx["wnids"]) == lm["wnids"]


def test_parse_classes_py(build_script):
    text = 'IMAGENET2012_CLASSES = OrderedDict(\n    {\n        "n01440764": "tench, Tinca tinca",\n' \
           '        "n01443537": "goldfish, Carassius auratus",\n    }\n)\n'
    assert build_script.parse_classes_py(text) == ["n01440764", "n01443537"]
