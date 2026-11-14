"""scripts/build_train_cache.py: the refactored selection reproduces the original one exactly, and the
method-selection set (--msel) is disjoint from every other role (scripts/check_disjoint.py)."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    key = f"_test_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


BTC = _load("build_train_cache")
CHK = _load("check_disjoint")


def original_selection(files, shard_labels, seed):
    """Frozen copy of the selection in scripts/build_train_cache.py at commit a5094b4 (steps 1 and 2 of
    main(), labels already cached), used to build the A100 and 5090 caches. Do not edit."""
    ROLES_224 = [("fit0", 2), ("fit1", 2), ("fit2", 2), ("bank", 100)]
    ROLES_256 = [("healval", 5), ("heal", 100)]
    need = sum(n for _, n in ROLES_224 + ROLES_256)
    counts = np.zeros(1000, dtype=np.int64)
    used = []
    for f in files:
        used.append(f)
        counts += np.bincount(np.asarray(shard_labels[f]), minlength=1000)
        if counts.min() >= need:
            break
    if counts.min() < need:
        raise RuntimeError("not enough images per class in the train shards")
    rng = np.random.default_rng(seed)
    per_class = {c: [] for c in range(1000)}
    for si, f in enumerate(used):
        for r, lab in enumerate(shard_labels[f]):
            per_class[lab].append((si, r))
    roles = {name: [] for name, _ in ROLES_224 + ROLES_256}
    for c in range(1000):
        items = per_class[c]
        order = rng.permutation(len(items))
        pos = 0
        for name, n in ROLES_224 + ROLES_256:
            roles[name] += [(items[i][0], items[i][1], c) for i in order[pos:pos + n]]
            pos += n
    group224 = [x for name, _ in ROLES_224 for x in roles[name]]
    group256 = [x for name, _ in ROLES_256 for x in roles[name]]
    idx, start = {}, 0
    for name, n in ROLES_224:
        k = n * 1000
        idx[f"{name}_rows"] = np.arange(start, start + k)
        idx[f"{name}_labels"] = np.array([c for _, _, c in group224[start:start + k]])
        start += k
    start = 0
    for name, n in ROLES_256:
        k = n * 1000
        idx[f"{name}_rows"] = np.arange(start, start + k)
        idx[f"{name}_labels"] = np.array([c for _, _, c in group256[start:start + k]])
        start += k
    idx["src224"] = np.array([(si, r) for si, r, _ in group224])
    idx["src256"] = np.array([(si, r) for si, r, _ in group256])
    return used, roles, group224, group256, idx


def synthetic_shards(n_shards=90, per_shard=4300, seed=123, skew=True):
    """Shard names in the hub's naming and label lists; class frequencies vary (like ImageNet)."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.8, 1.2, 1000) if skew else np.ones(1000)
    p /= p.sum()
    files = [f"data/train-{i:05d}-of-{n_shards:05d}.parquet" for i in range(n_shards)]
    labels = {f: rng.choice(1000, size=per_shard, p=p).tolist() for f in files}
    return files, labels


@pytest.fixture(scope="module")
def shards():
    return synthetic_shards()


@pytest.fixture(scope="module")
def selection(shards):
    files, labels = shards
    used, counts = BTC.select_used_shards(files, labels.__getitem__, BTC.need_per_class())
    roles = BTC.select_roles(used, labels, 0)
    g224, g256 = BTC.role_groups(roles)
    return used, counts, roles, g224, g256, BTC.build_index(g224, g256)


# ----------------------------------------------------------------------------------------------
# Existing roles: identical to the original code
# ----------------------------------------------------------------------------------------------
def test_need_is_211():
    assert BTC.need_per_class() == 211
    assert BTC.ROLES_224 == [("fit0", 2), ("fit1", 2), ("fit2", 2), ("bank", 100)]
    assert BTC.ROLES_256 == [("healval", 5), ("heal", 100)]


def test_selection_matches_original(shards, selection):
    files, labels = shards
    used, counts, roles, g224, g256, idx = selection
    o_used, o_roles, o_g224, o_g256, o_idx = original_selection(files, labels, 0)
    assert used == o_used and 50 < len(used) < len(files)       # stops early, leaves shards for msel
    assert counts.min() >= 211
    assert {k: [tuple(map(int, x)) for x in v] for k, v in roles.items()} == \
           {k: [tuple(map(int, x)) for x in v] for k, v in o_roles.items()}
    assert g224 == o_g224 and g256 == o_g256
    assert sorted(idx) == sorted(o_idx)
    for k in idx:
        assert idx[k].dtype == o_idx[k].dtype and idx[k].shape == o_idx[k].shape, k
        assert np.array_equal(idx[k], o_idx[k]), k


@pytest.mark.parametrize("seed", [1, 7])
def test_selection_matches_original_other_seeds(seed):
    files, labels = synthetic_shards(n_shards=80, per_shard=4400, seed=seed + 10, skew=False)
    used, _ = BTC.select_used_shards(files, labels.__getitem__, BTC.need_per_class())
    g224, g256 = BTC.role_groups(BTC.select_roles(used, labels, seed))
    _, _, o_g224, o_g256, _ = original_selection(files, labels, seed)
    assert g224 == o_g224 and g256 == o_g256


def test_extra_cached_labels_do_not_change_selection(shards, selection):
    """--msel fetches labels of later shards into the same JSON cache; the used shards stay the same."""
    files, labels = shards
    used = selection[0]
    seen = []

    def labels_of(f):
        seen.append(f)
        return labels[f]

    used2, _ = BTC.select_used_shards(files, labels_of, BTC.need_per_class())
    assert used2 == used and seen == used                     # only the used shards are read


def test_selection_fingerprint_is_stable(selection):
    """Frozen value from the original code on this synthetic input (numpy default_rng stream). If it
    fails on a machine, its numpy draws differ, and a cache built there would differ from the A100's."""
    used, *_, idx = selection
    fp = BTC.selection_fingerprint(idx, used)
    assert fp == BTC.selection_fingerprint({k: v.copy() for k, v in idx.items()}, list(used))
    assert fp == FROZEN_FINGERPRINT


FROZEN_FINGERPRINT = "b22c21ef8797e8d501972614d14517c9a5514299293ee61bde6a5f3da1b3bb13"   # original code, numpy 2.4.4


# ----------------------------------------------------------------------------------------------
# Method-selection set
# ----------------------------------------------------------------------------------------------
def test_msel_disjoint_and_balanced(shards, selection):
    files, labels = shards
    used, _, roles, *_ = selection
    msel_shards, picks = BTC.select_msel(files, len(used), labels.__getitem__)
    assert msel_shards == files[len(used):len(used) + len(msel_shards)]   # the next shards, in file order
    assert not set(msel_shards) & set(used)
    assert len(picks) == 10_000
    assert np.array_equal(np.bincount([c for *_, c in picks], minlength=1000), np.full(1000, 10))
    for si, r, c in picks:                                      # labels come from the shard
        assert labels[msel_shards[si]][r] == c
    ids = {name: [(used[si], r) for si, r, _ in v] for name, v in roles.items()}
    ids["msel"] = [(msel_shards[si], r) for si, r, _ in picks]
    res = CHK.check_roles(ids)
    assert res["ok"], res
    assert set(res["overlaps"]) >= {f"{a}|msel" for a in ("bank", "fit0", "fit1", "fit2", "heal", "healval")}


def test_msel_is_deterministic_and_seeded(shards, selection):
    files, labels = shards
    n_used = len(selection[0])
    a = BTC.select_msel(files, n_used, labels.__getitem__)
    b = BTC.select_msel(files, n_used, labels.__getitem__, seed=1)
    c = BTC.select_msel(files, n_used, labels.__getitem__, seed=2)
    assert a == b and a[1] != c[1]


def test_msel_fails_without_enough_shards(shards, selection):
    files, labels = shards
    with pytest.raises(RuntimeError):
        BTC.select_msel(files[:len(selection[0]) + 1], len(selection[0]), labels.__getitem__, per_class=10)


# ----------------------------------------------------------------------------------------------
# check_disjoint
# ----------------------------------------------------------------------------------------------
def test_check_roles_detects_overlap_and_duplicates():
    ok = CHK.check_roles({"a": [("s0", 1), ("s0", 2)], "b": [("s1", 1)]})
    assert ok["ok"] and ok["overlaps"] == {"a|b": 0}
    clash = CHK.check_roles({"a": [("s0", 1), ("s0", 2)], "b": [("s0", 2)]})
    assert not clash["ok"] and clash["overlaps"]["a|b"] == 1
    dup = CHK.check_roles({"a": [("s0", 1), ("s0", 1)], "b": []})
    assert not dup["ok"] and dup["roles"]["a"] == {"n": 2, "unique": 1}


def _write_cache(tmp_path, shards, selection, msel=True, corrupt=False):
    files, labels = shards
    used, _, _, _, _, idx = selection
    idx = {k: v.copy() for k, v in idx.items()}
    if corrupt:                                  # heal image 0 becomes bank image 0
        bank0 = idx["src224"][idx["bank_rows"][0]]
        idx["src256"][idx["heal_rows"][0]] = bank0
        idx["heal_labels"][0] = idx["bank_labels"][0]
    np.savez(tmp_path / "train_index.npz", **idx)
    meta = {"complete": True, "seed": 0, "shards_used": used,
            "selection_sha256": BTC.selection_fingerprint(idx, used)}
    (tmp_path / "train_meta.json").write_text(json.dumps(meta))
    (tmp_path / "train_shard_labels.json").write_text(json.dumps(labels))
    (tmp_path / "meta.json").write_text(json.dumps(
        {"complete": True, "n_images": 50000,
         "source_files": [f"validation-{i:05d}-of-00014.parquet" for i in range(14)]}))
    if msel:
        ms, picks = BTC.select_msel(files, len(used), labels.__getitem__)
        np.savez(tmp_path / "msel_index.npz", **BTC.msel_index(ms, picks))


def test_check_disjoint_run_passes(tmp_path, shards, selection):
    _write_cache(tmp_path, shards, selection)
    rep = CHK.run(tmp_path, tmp_path / "out" / "disjoint.json")
    assert rep["ok"], rep["problems"]
    saved = json.loads((tmp_path / "out" / "disjoint.json").read_text())
    assert saved["ok"] and saved["roles"]["msel"]["n"] == 10_000
    assert saved["roles"]["bank"]["per_class_min"] == 100
    assert all(v == 0 for v in saved["pairwise_overlaps"].values())
    assert all(v["mismatches"] == 0 and v["checked"] == saved["roles"][k]["n"]
               for k, v in saved["label_check"].items())


def test_check_disjoint_run_fails_on_overlap(tmp_path, shards, selection):
    _write_cache(tmp_path, shards, selection, corrupt=True)
    rep = CHK.run(tmp_path, tmp_path / "disjoint.json")
    assert not rep["ok"]
    assert rep["pairwise_overlaps"]["bank|heal"] == 1


def test_check_disjoint_requires_msel(tmp_path, shards, selection):
    _write_cache(tmp_path, shards, selection, msel=False)
    assert not CHK.run(tmp_path, tmp_path / "d.json")["ok"]
    assert CHK.run(tmp_path, tmp_path / "d.json", allow_missing_msel=True)["ok"]


# ----------------------------------------------------------------------------------------------
# Writing the msel files and reading them through mirror.protocol.Protocol.msel_loader
# ----------------------------------------------------------------------------------------------
def _tiny_parquet(path, labels, size=(40, 30), seed=0):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    rng = np.random.default_rng(seed)
    imgs = []
    for _ in labels:
        arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        imgs.append({"bytes": buf.getvalue(), "path": None})
    pq.write_table(pa.table({"image": imgs, "label": labels}), path)


def test_write_msel_is_read_by_protocol(tmp_path):
    from mirror.protocol import Protocol

    n_cls = 3
    labels = {f"data/train-{i:05d}-of-00004.parquet": [i % n_cls, (i + 1) % n_cls, (i + 2) % n_cls, 0]
              for i in range(4)}
    files = sorted(labels)
    local = {}
    for i, f in enumerate(files):
        local[f] = tmp_path / f"shard{i}.parquet"
        _tiny_parquet(local[f], labels[f], seed=i)
    shards, picks = BTC.select_msel(files, 2, labels.__getitem__, per_class=2, n_classes=n_cls)
    assert set(shards) <= set(files[2:])
    meta = BTC.write_msel(tmp_path, shards, picks, lambda name: str(local[name]), workers=1)
    assert meta["complete"] and meta["n"] == 6 and meta["per_class_min"] == 2
    images = np.load(tmp_path / "msel224_uint8.npy")
    assert images.shape == (6, 224, 224, 3) and images.dtype == np.uint8
    with np.load(tmp_path / "msel_index.npz") as z:
        assert z["msel_labels"].dtype == np.int64 and z["msel_source"].shape == (6, 2)
        assert [str(s) for s in z["msel_shards"]] == list(shards)
    # same pixels as the train224 transform applied to the stored image bytes
    import pyarrow.parquet as pq

    si, r, _ = picks[0]
    raw = pq.read_table(local[shards[si]], columns=["image"]).column("image")[r].as_py()["bytes"]
    assert np.array_equal(images[0], BTC._decode((raw, 256, 224)))
    proto = Protocol.__new__(Protocol)                  # only the attributes msel_loader uses
    proto.cache, proto._msel, proto.bs, proto.nw, proto.pin = tmp_path, None, 4, 0, False
    batches = list(proto.msel_loader())
    got = np.concatenate([np.asarray(b[1]) for b in batches])
    assert got.tolist() == [c for *_, c in picks]


# ----------------------------------------------------------------------------------------------
# scripts/prefetch_models.py (model-weight stage of scripts/h100_data.sh): the plan only, no download
# ----------------------------------------------------------------------------------------------
def test_prefetch_plan_covers_registry_and_dinov2_family():
    from mirror.adapters_timm import MODEL_REGISTRY

    PM = _load("prefetch_models")
    items = PM.plan(MODEL_REGISTRY, extra_timm=["resnet50.a1_in1k"])
    names = {it["name"] for it in items}
    assert set(MODEL_REGISTRY) <= names
    ids = [(it["kind"], it["id"]) for it in items]
    assert len(ids) == len(set(ids))                               # registry and family lists deduplicated
    hub = {it["id"] for it in items if it["kind"] == "torch_hub"}
    assert {f"dinov2_vit{s}14{r}" for s in "sblg" for r in ("", "_reg")} <= hub
    assert ("timm", "resnet50.a1_in1k") in ids
    for name, e in MODEL_REGISTRY.items():
        it = next(i for i in items if i["name"] == name)
        assert it["kind"] == {"timm": "timm", "torch_hub": "torch_hub"}.get(e["source"], it["kind"])
    odd = PM.plan({"x": {"source": "onnx"}, "y": {"source": "hf", "hf_id": "org/model"}})
    assert odd[0]["kind"] == "skip" and odd[1] == {"name": "y", "kind": "hf", "id": "org/model", "revision": None}
    only = PM.plan(MODEL_REGISTRY, only={"dinov2_vitg14_reg"})
    assert [it["id"] for it in only] == ["dinov2_vitg14_reg"]


def test_prefetch_dry_run_downloads_nothing(capsys):
    PM = _load("prefetch_models")
    assert PM.main(["--dry-run", "--no-dinov2-g"]) == 0
    out = capsys.readouterr().out
    assert "torch_hub" in out and "vitg14" not in out
