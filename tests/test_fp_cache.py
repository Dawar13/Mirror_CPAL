"""mirror/final/data/cache.py and scripts/fp_cache.py: new data roles, accessor, checks and the full-train stream,
on fake shards (no downloads)."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from mirror.final.data import cache as C

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    key = f"_test_fp_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


@pytest.fixture(scope="module")
def btc():
    return _load("build_train_cache")


def synthetic_labels(n_shards, per_shard, n_classes, seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.6, 1.4, n_classes)
    p /= p.sum()
    files = [f"data/train-{i:05d}-of-{n_shards:05d}.parquet" for i in range(n_shards)]
    return files, {f: rng.choice(n_classes, size=per_shard, p=p).tolist() for f in files}


# ----------------------------------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------------------------------
def test_constants_match_existing_builder(btc):
    assert C.IMAGENET_REVISION == btc.REVISION and C.IMAGENET_REPO == btc.REPO
    assert C.USED_PER_CLASS == btc.need_per_class() == 211
    assert (C.MSEL_PER_CLASS, C.MSEL_SEED) == (btc.MSEL_PER_CLASS, btc.MSEL_SEED)
    assert (C.FIT3.per_class, C.FIT3.n, C.FIT3.crop) == (2, 2000, 224)
    assert (C.HEAL_EXT.per_class, C.HEAL_EXT.n, C.HEAL_EXT.crop) == (200, 200_000, 256)
    assert (C.COCO.n, C.COCO.crop) == (2000, 224)
    assert len({C.FIT3.seed, C.HEAL_EXT.seed, C.COCO.seed, 0, 1}) == 5


def test_selection_functions_equal_build_train_cache(btc):
    files, labels = synthetic_labels(40, 900, 20, seed=3)
    used, counts = C.select_used_shards(files, labels.__getitem__, 211, n_classes=20)
    b_used, b_counts = btc.select_used_shards(files, labels.__getitem__, 211, n_classes=20)
    assert used == b_used and np.array_equal(counts, b_counts)
    for start, per, seed in ((len(used), 10, 1), (len(used) + 2, 2, 2), (3, 7, 5)):
        assert C.select_block(files, start, labels.__getitem__, per, seed, n_classes=20) == \
            btc.select_msel(files, start, labels.__getitem__, per, seed, n_classes=20)


def test_new_roles_disjoint_balanced_and_deterministic(btc):
    n_cls = 20
    files, labels = synthetic_labels(60, 900, n_cls, seed=4)
    plan = C.plan_imagenet(files, labels.__getitem__, n_classes=n_cls)
    used, msel = plan["_used"], plan["_msel"]
    fit3, he = plan["fit3"], plan["heal_ext"]
    assert fit3["first_shard_index"] == len(used) + len(msel)
    assert he["first_shard_index"] == fit3["first_shard_index"] + len(fit3["shards"])
    for p, per in ((fit3, 2), (he, 200)):
        cls = np.array([c for *_, c in p["picks"]])
        assert np.array_equal(np.bincount(cls, minlength=n_cls), np.full(n_cls, per))
        for si, r, c in p["picks"]:
            assert labels[p["shards"][si]][r] == c
    # every role of the existing builder and msel, by (shard, row) ID
    roles = btc.select_roles(used, labels, 0, n_classes=n_cls)
    ids = {name: C.id_keys(C.make_ids(v, used)) for name, v in roles.items()}
    m_pool, m_picks = btc.select_msel(files, len(used), labels.__getitem__, n_classes=n_cls)
    assert m_pool == msel
    ids["msel"] = C.id_keys(C.make_ids(m_picks, m_pool))
    ids["fit3"] = C.id_keys(C.make_ids(fit3["picks"], fit3["shards"]))
    ids["heal_ext"] = C.id_keys(C.make_ids(he["picks"], he["shards"]))
    names = sorted(ids)
    for i, a in enumerate(names):
        assert len(np.unique(ids[a])) == len(ids[a])
        for b in names[i + 1:]:
            assert not np.intersect1d(ids[a], ids[b]).size, (a, b)
    assert len(np.union1d(ids["heal"], ids["heal_ext"])) == 300 * n_cls
    # determinism; the existing caches' shard lists give the same plan; another seed gives other images
    again = C.plan_imagenet(files, labels.__getitem__, used=used, msel_shards=msel, n_classes=n_cls)
    assert again["fit3"]["picks"] == fit3["picks"] and again["heal_ext"]["picks"] == he["picks"]
    other = C.plan_imagenet(files, labels.__getitem__, n_classes=n_cls,
                            chain=(C.scaled_spec(C.FIT3, seed=99), C.HEAL_EXT))
    assert other["fit3"]["shards"] == fit3["shards"] and other["fit3"]["picks"] != fit3["picks"]
    only = C.plan_imagenet(files, labels.__getitem__, n_classes=n_cls, upto="fit3")
    assert "heal_ext" not in only and only["fit3"]["picks"] == fit3["picks"]
    with pytest.raises(RuntimeError, match="first shards"):
        C.plan_imagenet(files, labels.__getitem__, used=used[1:], msel_shards=msel, n_classes=n_cls)
    with pytest.raises(RuntimeError, match="right after"):
        C.plan_imagenet(files, labels.__getitem__, used=used, msel_shards=[files[-1]], n_classes=n_cls)


def test_plan_coco():
    sizes = {f"data/train-{i:05d}-of-00040-abc.parquet": 2900 + i for i in range(40)}
    files = sorted(sizes)
    p = C.plan_coco(files, sizes.__getitem__)
    assert p["shards"] == files[:2] and p["pool_rows"] == 2900 + 2901
    keys = [(si, r) for si, r, _ in p["picks"]]
    assert len(keys) == 2000 == len(set(keys)) and keys == sorted(keys)
    assert all(r < sizes[p["shards"][si]] for si, r in keys) and all(c == -1 for *_, c in p["picks"])
    assert C.plan_coco(files, sizes.__getitem__)["picks"] == p["picks"]
    assert C.plan_coco(files, sizes.__getitem__, C.scaled_spec(C.COCO, seed=7))["picks"] != p["picks"]
    assert C.make_ids(p["picks"][:1], p["shards"]).tolist() == [[0, keys[0][1]]]


# ----------------------------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------------------------
def _encode(arr, mode, fmt):
    from PIL import Image

    img = Image.fromarray(arr)
    if mode != "RGB":
        img = img.convert(mode)
    buf = io.BytesIO()
    img.save(buf, format=fmt, **({"quality": 90} if fmt == "JPEG" else {}))
    return buf.getvalue()


@pytest.mark.parametrize("size", [(500, 375), (375, 500), (256, 256), (257, 255), (100, 80), (1000, 300),
                                  (224, 224), (299, 1001), (513, 340)])
def test_preprocessing_identical_to_build_train_cache(btc, size):
    rng = np.random.default_rng(size[0] * 7 + size[1])
    w, h = size
    arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    arr[: h // 2] = arr[: h // 2] // 3                          # some structure, not only noise
    cases = [("RGB", "JPEG"), ("RGB", "PNG"), ("L", "JPEG"), ("RGBA", "PNG"), ("P", "PNG"), ("CMYK", "JPEG")]
    for mode, fmt in cases:
        data = _encode(arr, mode, fmt)
        for crop in (224, 256):
            mine = C.decode_image(data, 256, crop)
            ref = btc._decode((data, 256, crop))
            assert mine.dtype == np.uint8 and mine.shape == (crop, crop, 3)
            assert np.array_equal(mine, ref), (size, mode, fmt, crop)
    assert np.array_equal(C.decode_reference(data, 256, 224), btc._decode((data, 256, 224)))


def test_parity_check_reports_and_raises():
    rng = np.random.default_rng(0)
    data = [_encode(rng.integers(0, 256, (40, 60, 3), dtype=np.uint8), "RGB", "PNG") for _ in range(3)]
    dec = np.stack([C.decode_image(d, 32, 28) for d in data])
    assert C.parity_check(data, dec, 32, 28) == {"checked": 3, "identical": True}
    dec[1, 0, 0, 0] ^= 1
    with pytest.raises(RuntimeError, match="differs"):
        C.parity_check(data, dec, 32, 28)


# ----------------------------------------------------------------------------------------------
# Building roles from fake shards
# ----------------------------------------------------------------------------------------------
N_CLS = 4
CHAIN = (C.scaled_spec(C.FIT3, per_class=1, n=N_CLS, resize=32, crop=28),
         C.scaled_spec(C.HEAL_EXT, per_class=3, n=3 * N_CLS, resize=32, crop=32))
COCO_SPEC = C.scaled_spec(C.COCO, n=6, resize=32, crop=28)


@pytest.fixture(scope="module")
def shards(tmp_path_factory):
    d = tmp_path_factory.mktemp("shards")
    C.write_synthetic_shards(d / "imagenet", n_shards=14, rows_per_shard=12, n_classes=N_CLS, seed=0)
    C.write_synthetic_shards(d / "coco", n_shards=3, rows_per_shard=8, n_classes=1, seed=1, labels=False,
                             image_id=True, fmt="JPEG")
    return d


def _src(shards, **kw):
    return C.LocalParquetSource(shards / "imagenet", dataset="imagenet_train", **kw)


def _coco(shards, **kw):
    return C.LocalParquetSource(shards / "coco", dataset="coco_train", label_col=None, extra_cols=("image_id",), **kw)


def _build(shards, cache, src=None, **kw):
    out = C.build_imagenet_roles(["fit3", "heal_ext"], src or _src(shards), cache, workers=0, parity=0, chain=CHAIN,
                                 n_classes=N_CLS, used_need=4, msel_per_class=2, **kw)
    out["coco"] = C.build_coco(_coco(shards), cache, spec=COCO_SPEC, workers=0, parity=0)
    return out


def _bytes_at(folder, shard, row):
    return C.read_rows(Path(folder) / shard, [row])["images"][0]


def test_build_roles_from_fake_shards(shards, tmp_path):
    cache = tmp_path / "cache"
    metas = _build(shards, cache)
    for spec in (*CHAIN, COCO_SPEC):
        role = C.get_role(spec.name, cache)
        folder = shards / ("coco" if spec.name == "coco" else "imagenet")
        assert metas[spec.name]["complete"] and len(role) == spec.n
        assert isinstance(role.images, np.memmap) and role.images.shape == (spec.n, spec.crop, spec.crop, 3)
        with np.load(cache / spec.index_file) as z:
            shard_names = [str(s) for s in z["shards"]]
            src = z["source"]
        nums = {C.shard_number(s): s for s in shard_names}
        for i in (0, spec.n - 1):
            num, row = role.ids[i]
            assert nums[num] == shard_names[src[i, 0]] and row == src[i, 1]
            assert np.array_equal(role.images[i], C.decode_image(_bytes_at(folder, nums[num], row), 32, spec.crop))
        if spec.name != "coco":
            labs = C.LocalParquetSource(folder).labels
            assert all(labs(nums[n])[r] == lab for (n, r), lab in zip(role.ids.tolist(), role.labels.tolist()))
            assert np.array_equal(np.bincount(role.labels, minlength=N_CLS), np.full(N_CLS, spec.per_class))
    coco = C.get_role("coco", cache)
    assert (coco.labels == -1).all() and len(set(coco.meta["array_image_id"].tolist())) == 6
    # a second build skips everything and fetches nothing
    src = _src(shards)
    again = C.build_imagenet_roles(["fit3", "heal_ext"], src, cache, workers=0, parity=0, chain=CHAIN,
                                   n_classes=N_CLS, used_need=4, msel_per_class=2)
    assert src.fetched == [] and again["fit3"]["selection_sha256"] == metas["fit3"]["selection_sha256"]


class _Flaky(C.LocalParquetSource):
    def __init__(self, *a, fail_at=2, **kw):
        super().__init__(*a, **kw)
        self.fail_at = fail_at

    def fetch(self, f):
        if len(self.fetched) + 1 == self.fail_at:
            self.fetched.append(f)
            raise OSError("network down")
        return super().fetch(f)


def test_build_resumes_after_interruption(shards, tmp_path):
    clean = tmp_path / "clean"
    _build(shards, clean)
    broken = tmp_path / "broken"
    spec = CHAIN[1]
    with pytest.raises(OSError, match="network"):
        C.build_imagenet_roles(["heal_ext"], _Flaky(shards / "imagenet", fail_at=2), broken, workers=0, parity=0,
                               chain=CHAIN, n_classes=N_CLS, used_need=4, msel_per_class=2)
    partial = broken / (spec.images_file + ".partial.npy")
    prog = json.loads((broken / (spec.images_file + ".progress.json")).read_text())
    assert partial.exists() and len(prog["shards_done"]) == 1
    src = _src(shards)
    C.build_imagenet_roles(["heal_ext"], src, broken, workers=0, parity=0, chain=CHAIN, n_classes=N_CLS,
                           used_need=4, msel_per_class=2)
    assert len(src.fetched) == len(json.loads((clean / spec.meta_file).read_text())["shards"]) - 1
    assert np.array_equal(np.load(broken / spec.images_file), np.load(clean / spec.images_file))
    with np.load(broken / spec.index_file) as a, np.load(clean / spec.index_file) as b:
        assert all(np.array_equal(a[k], b[k]) for k in a.files)
    assert not partial.exists()


def test_delete_shards_only_removes_downloaded_files(shards, tmp_path):
    plan = C.plan_imagenet(_src(shards).files(), _src(shards).labels, chain=CHAIN, n_classes=N_CLS, used_need=4,
                           msel_per_class=2)
    pool = plan["heal_ext"]["shards"]
    src = _src(shards, cached=pool[:1])                          # first shard was on disk before the build
    C.build_imagenet_roles(["heal_ext"], src, tmp_path, workers=0, parity=0, chain=CHAIN, n_classes=N_CLS,
                           used_need=4, msel_per_class=2, delete_shards=True)
    assert src.released == pool[1:]


# ----------------------------------------------------------------------------------------------
# Accessor for every role, and the checks
# ----------------------------------------------------------------------------------------------
def _fake_old_cache(cache, shards):
    """Existing caches in their real file layout, with tiny role sizes, from the same fake shards."""
    src = _src(shards)
    files = src.files()
    plan = C.plan_imagenet(files, src.labels, chain=CHAIN, n_classes=N_CLS, used_need=4, msel_per_class=2)
    used = plan["_used"]
    per = {"fit0": 1, "fit1": 1, "fit2": 1, "bank": 1, "healval": 0, "heal": 0}
    roles = {k: [] for k in per}
    rng = np.random.default_rng(0)
    pairs = [(si, r, lab) for si, f in enumerate(used) for r, lab in enumerate(src.labels(f))]
    order = rng.permutation(len(pairs))
    take = iter(order)
    for k in ("fit0", "fit1", "fit2", "bank"):
        roles[k] = [pairs[next(take)] for _ in range(2)]
    roles["healval"] = [pairs[next(take)] for _ in range(2)]
    roles["heal"] = [pairs[next(take)] for _ in range(3)]
    idx, g224, g256 = {}, [], []
    for k in ("fit0", "fit1", "fit2", "bank"):
        idx[f"{k}_rows"] = np.arange(len(g224), len(g224) + len(roles[k]))
        idx[f"{k}_labels"] = np.array([c for *_, c in roles[k]])
        g224 += roles[k]
    for k in ("healval", "heal"):
        idx[f"{k}_rows"] = np.arange(len(g256), len(g256) + len(roles[k]))
        idx[f"{k}_labels"] = np.array([c for *_, c in roles[k]])
        g256 += roles[k]
    idx["src224"] = np.array([(si, r) for si, r, _ in g224])
    idx["src256"] = np.array([(si, r) for si, r, _ in g256])
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(cache / "train_index.npz", **idx)
    (cache / "train_meta.json").write_text(json.dumps({"complete": True, "shards_used": used}))
    np.save(cache / "train224_uint8.npy", rng.integers(0, 256, (len(g224), 8, 8, 3), dtype=np.uint8))
    np.save(cache / "heal256_uint8.npy", rng.integers(0, 256, (len(g256), 8, 8, 3), dtype=np.uint8))
    m_pool, m_picks = C.select_block(files, len(used), src.labels, 2, 1, n_classes=N_CLS)
    np.savez(cache / "msel_index.npz", msel_rows=np.arange(len(m_picks)),
             msel_labels=np.array([c for *_, c in m_picks]), msel_source=np.array([(s, r) for s, r, _ in m_picks]),
             msel_shards=np.array(m_pool))
    np.save(cache / "msel224_uint8.npy", rng.integers(0, 256, (len(m_picks), 8, 8, 3), dtype=np.uint8))
    np.save(cache / "val224_uint8.npy", rng.integers(0, 256, (5, 8, 8, 3), dtype=np.uint8))
    np.save(cache / "val_labels.npy", np.arange(5) % N_CLS)
    (cache / "train_shard_labels.json").write_text(json.dumps({f: src.labels(f) for f in files}))
    return roles, used, m_pool


def test_accessor_covers_every_role(shards, tmp_path):
    cache = tmp_path / "cache"
    roles, used, m_pool = _fake_old_cache(cache, shards)
    _build(shards, cache)
    C.build_stream_index(_src(shards), cache)
    seen = set()
    for name in C.ALL_ROLES:
        r = C.get_role(name, cache)
        images, labels, ids = C.role_arrays(name, cache)
        assert len(labels) == len(ids) == len(r) and ids.shape[1] == 2 and labels.dtype == np.int64
        seen.add(name)
        if name in roles:
            assert isinstance(images, np.memmap)                 # a view of the shared memmap, not a copy
            exp = [(C.shard_number(used[si]), rr) for si, rr, _ in roles[name]]
            assert ids.tolist() == [list(e) for e in exp]
            assert labels.tolist() == [c for *_, c in roles[name]]
            batches = list(r.batches(batch_size=1))
            assert len(batches) == len(r) and batches[0][0].shape == (1, 8, 8, 3)
    assert seen == set(C.ALL_ROLES)
    assert C.get_role("stream_train", cache).images is None and C.get_role("val", cache).ids[:, 0].max() == -1
    assert {r["role"] for r in C.list_roles(cache) if r["present"]} == set(C.ALL_ROLES)
    # the existing caches' shard lists are picked up, and the new roles still match a plan from scratch
    used2, msel2 = C.existing_shard_lists(cache)
    assert used2 == used and msel2 == m_pool
    exp = {"fit3": N_CLS, "heal_ext": 3 * N_CLS, "coco": 6, "stream_train": 14 * 12}
    rep = C.check_roles(cache, tmp_path / "rep.json", expected=exp, require=("fit3", "heal_ext", "coco"))
    assert rep["ok"], rep["problems"]
    assert all(v == 0 for v in rep["pairwise_overlaps"].values()) and "fit3|heal_ext" in rep["pairwise_overlaps"]
    assert rep["stream_train"]["roles"]["heal_ext"] == {"inside": 3 * N_CLS, "labels_agree": 3 * N_CLS}
    assert rep["label_check"]["fit3"] == {"checked": N_CLS, "mismatches": 0}
    assert json.loads((tmp_path / "rep.json").read_text())["ok"]


def test_check_detects_overlap_and_label_errors(shards, tmp_path):
    cache = tmp_path / "cache"
    _fake_old_cache(cache, shards)
    _build(shards, cache)
    spec = CHAIN[0]
    with np.load(cache / spec.index_file) as z:
        idx = {k: z[k].copy() for k in z.files}
    heal_id = C.get_role("heal", cache).ids[0]
    idx["ids"][0] = heal_id                                      # fit3 image 0 becomes heal image 0
    np.savez(cache / spec.index_file, **idx)
    rep = C.check_roles(cache, None, expected={}, require=())
    assert not rep["ok"] and rep["pairwise_overlaps"]["heal|fit3"] == 1
    assert rep["label_check"]["fit3"]["checked"] == N_CLS
    idx["ids"][0] = idx["ids"][1]                                # duplicate inside the role
    np.savez(cache / spec.index_file, **idx)
    rep = C.check_roles(cache, None, expected={}, require=())
    assert any("duplicate" in p for p in rep["problems"])
    assert any("required role" in p for p in C.check_roles(tmp_path / "empty", None, require=("fit3",))["problems"])


# ----------------------------------------------------------------------------------------------
# Full-train stream
# ----------------------------------------------------------------------------------------------
def _expected_stream(src, files):
    ids, labs = [], []
    for f in files:
        lab = src.labels(f)
        ids += [(C.shard_number(f), r) for r in range(len(lab))]
        labs += lab
    return np.array(ids), np.array(labs)


def test_stream_order_ids_and_batches(shards):
    src = _src(shards)
    files = src.files()
    batches = list(C.stream_batches(src, batch_size=10, workers=0, resize=32, crop=28, chunk=5))
    ids = np.concatenate([b[2] for b in batches])
    labs = np.concatenate([b[1] for b in batches])
    e_ids, e_labs = _expected_stream(src, files)
    assert np.array_equal(ids, e_ids) and np.array_equal(labs, e_labs)
    assert [len(b[1]) for b in batches] == [10] * 16 + [8]       # 168 images; batches cross shard boundaries
    x = batches[2][0]                                            # images 20..29: shard 1, rows 8..11 then shard 2
    assert x.dtype == np.uint8 and x.shape == (10, 28, 28, 3) and x.flags["C_CONTIGUOUS"]
    for j in (0, 9):
        num, row = batches[2][2][j]
        assert np.array_equal(x[j], C.decode_image(_bytes_at(shards / "imagenet", files[num], row), 32, 28))
    # stream_train_index.npz holds the same IDs and labels in the same order
    with np.load(_index(shards)) as z:
        assert np.array_equal(z["ids"], ids) and np.array_equal(z["labels"], labs)


def _index(shards):
    out = shards / "stream_cache"
    if not (out / C.STREAM_INDEX).exists():
        C.build_stream_index(_src(shards), out)
    return out / C.STREAM_INDEX


def test_stream_resume_alignment_and_deletion(shards):
    files = _src(shards).files()
    src = _src(shards, cached=files[:3])
    batches = list(C.stream_batches(src, batch_size=8, workers=0, start_shard=2, stop_shard=6, resize=32, crop=28,
                                    align_to_shards=True, delete_after=True))
    assert [len(b[1]) for b in batches] == [8, 4] * 4               # never crosses a shard boundary
    nums = [set(b[2][:, 0].tolist()) for b in batches]
    assert all(len(s) == 1 for s in nums) and [min(s) for s in nums][::2] == [2, 3, 4, 5]
    assert src.fetched == files[2:6] and src.released == files[3:6]  # only shards it downloaded itself
    # stopping early ends the download thread and leaves nothing running
    src2 = _src(shards)
    it = C.stream_batches(src2, batch_size=4, workers=0, resize=32, crop=28)
    first = next(it)
    it.close()
    assert len(first[1]) == 4 and len(src2.fetched) <= 3


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="worker processes: run on the VM (Linux)")
def test_stream_with_worker_processes(shards):
    src = _src(shards)
    serial = list(C.stream_batches(src, batch_size=7, workers=0, stop_shard=4, resize=32, crop=28))
    pooled = list(C.stream_batches(_src(shards), batch_size=7, workers=2, stop_shard=4, resize=32, crop=28, chunk=3))
    assert len(serial) == len(pooled)
    for a, b in zip(serial, pooled):
        assert all(np.array_equal(u, v) for u, v in zip(a, b))


# ----------------------------------------------------------------------------------------------
# Disk estimates and the command line
# ----------------------------------------------------------------------------------------------
def test_disk_estimates(tmp_path):
    rows = {r["role"]: r for r in C.disk_estimates(tmp_path)}
    assert rows["fit3"]["decoded_gb"] == pytest.approx(2000 * 224 * 224 * 3 / 1e9, abs=0.01)
    assert rows["heal_ext"]["decoded_gb"] == pytest.approx(39.32, abs=0.01)
    assert rows["coco"]["shards"] == 2 and rows["stream_train"]["shards"] == 294
    assert rows["stream_train"]["decoded_gb"] == 0.0
    assert 80 <= rows["heal_ext"]["shards"] <= 95
    text = C.format_estimates(list(rows.values()))
    assert "heal_ext" in text and "new decoded caches: 39.9 GB" in text


def test_fp_cache_dry_run(capsys):
    FP = _load("fp_cache")
    assert FP.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert '"check_ok": true' in out and "stream_train" in out
