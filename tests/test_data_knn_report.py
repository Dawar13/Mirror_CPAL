"""Splits, cache building from parquet, k-NN and the decision logic."""
import io

import numpy as np
import pandas as pd
import pytest
import torch

from mirror.data import build_cache_from_parquet, load_val_cache, make_splits, synthetic_dataset
from mirror.knn import knn_predict
from mirror.report import decide

PLAN = {"fit": 2, "heldout": 1, "bank": 3, "query": 2}


def test_splits_follow_order_of_appearance():
    labels = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0])
    s = make_splits(labels, PLAN)
    idx0 = np.flatnonzero(labels == 0)
    assert set(idx0[:2]) <= set(s["fit"]) and idx0[2] in s["heldout"]
    assert set(idx0[3:6]) <= set(s["bank"]) and set(idx0[6:8]) <= set(s["query"])
    allidx = np.concatenate(list(s.values()))
    assert len(allidx) == len(set(allidx)) == 16
    assert all(np.all(np.diff(v) > 0) for v in s.values())


def test_seeded_splits_are_fresh_balanced_and_deterministic():
    labels = np.repeat(np.arange(4), 8)
    a, b = make_splits(labels, PLAN, seed=1), make_splits(labels, PLAN, seed=1)
    base = make_splits(labels, PLAN)
    assert all(np.array_equal(a[k], b[k]) for k in PLAN)                     # deterministic
    assert any(not np.array_equal(a[k], base[k]) for k in PLAN)              # differs from E1's split
    for k, n in PLAN.items():
        assert np.bincount(labels[a[k]], minlength=4).tolist() == [n] * 4    # still class-balanced
    assert len(np.concatenate(list(a.values()))) == len(set(np.concatenate(list(a.values()))))


def test_splits_reject_small_classes():
    with pytest.raises(ValueError):
        make_splits(np.array([0, 0, 1]), PLAN)


def test_build_cache_from_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from PIL import Image

    rng = np.random.default_rng(0)
    rows = []
    for i in range(16):
        w, h = int(rng.integers(40, 90)), int(rng.integers(40, 90))
        buf = io.BytesIO()
        mode = "L" if i == 3 else "RGB"  # one greyscale image, as in real ImageNet
        arr = rng.integers(0, 255, (h, w) if mode == "L" else (h, w, 3), dtype=np.uint8)
        Image.fromarray(arr, mode).save(buf, format="JPEG")
        rows.append({"image": {"bytes": buf.getvalue(), "path": f"{i}.jpg"}, "label": i % 2})
    files = []
    for k in range(2):
        f = tmp_path / f"validation-0000{k}-of-00002.parquet"
        pq.write_table(pa.Table.from_pylist(rows[8 * k:8 * k + 8]), f)
        files.append(str(f))
    meta = build_cache_from_parquet(files, tmp_path / "cache", PLAN, resize=36, crop=32, workers=1)
    images, labels, splits, meta2 = load_val_cache(tmp_path / "cache", crop=32)
    assert images.shape == (16, 32, 32, 3) and images.dtype == np.uint8
    assert labels.tolist() == [i % 2 for i in range(16)]
    assert meta["split_sizes"] == {"fit": 4, "heldout": 2, "bank": 6, "query": 4} == meta2["split_sizes"]
    assert set(splits) == {"fit", "heldout", "bank", "query"}


def test_synthetic_dataset_shapes():
    imgs, labels = synthetic_dataset(3, 5, 28, seed=0)
    assert imgs.shape == (15, 28, 28, 3) and imgs.dtype == np.uint8
    assert np.bincount(labels).tolist() == [5, 5, 5]


def test_knn_predict_weighted_votes():
    bank = torch.nn.functional.normalize(torch.tensor([[1.0, 0], [0.9, 0.1], [0, 1.0], [0.1, 0.9]]), dim=1)
    labels = torch.tensor([0, 0, 1, 1])
    q = torch.nn.functional.normalize(torch.tensor([[1.0, 0.05], [0.05, 1.0]]), dim=1)
    assert knn_predict(bank, labels, q, num_classes=2, k=4).tolist() == [0, 1]


DCFG = {"blocks": [1, 2], "e_null_ratio_go": 5.0, "e_null_ratio_kill": 3.0, "e_identity_ratio_go": 0.5,
        "dminus_go": 0.2, "dminus_kill": 0.1, "share_go": 0.2, "share_kill": 0.3, "surgery_k": 2,
        "surgery_drop_go": 10.0, "surgery_margin_go": 5.0, "evidence_dminus": 0.2, "evidence_twin_frac": 0.2}


def frames(e_A=0.01, e_null=0.2, e_I=0.1, s=0.1, orig=80.0, surg=75.0, svd=60.0, twin=0.5, e_A_layer2=None):
    e_A_layers = [e_A, e_A, e_A if e_A_layer2 is None else e_A_layer2, e_A]
    layers = pd.DataFrame({"layer": [0, 1, 2, 3], "rel_e_A_shared": e_A_layers,
                           "rel_e_null_shared": [e_null] * 4, "rel_e_I": [e_I] * 4})
    mlp = pd.DataFrame({"block": [1, 2, 3], "s1": [s] * 3, "s2": [s] * 3, "s1_null": [0.5] * 3,
                        "s2_null": [0.5] * 3, "twin_frac": [twin] * 3})
    surgery = pd.DataFrame({"k": [0, 2, 2], "variant": ["original", "surgery", "svd"], "top1": [orig, surg, svd]})
    return layers, mlp, surgery


def test_decision_go_pivot_kill_amber():
    assert decide(*frames(), 0.3, DCFG)["outcome"] == "GO"
    assert decide(*frames(surg=60.0, svd=40.0), 0.3, DCFG)["outcome"] == "PIVOT"
    assert decide(*frames(e_null=0.02), 0.3, DCFG)["outcome"] == "KILL"          # row 1
    assert decide(*frames(s=0.35), 0.3, DCFG)["outcome"] == "KILL"               # row 4
    assert decide(*frames(svd=76.0), 0.3, DCFG)["outcome"] == "KILL"             # row 5
    assert decide(*frames(e_null=0.04), 0.3, DCFG)["outcome"] == "AMBER"         # ratio 4: between


def test_decision_little_to_save():
    out = decide(*frames(), 0.05, DCFG)                                           # d-/d < 0.1, many pairs
    assert out["rows"][2]["cell"] == "KILL"
    assert out["outcome"] == "PIVOT: structure but little to save"
    assert decide(*frames(s=0.35), 0.05, DCFG)["outcome"] == "KILL"              # KILL still wins


def test_decision_low_share_needs_evidence():
    # Low share, few pairs and d-/d < 0.2: the share is not evidence, so row 4 is not GO.
    out = decide(*frames(twin=0.05), 0.15, DCFG)
    assert out["rows"][3]["cell"] == "AMBER" and out["outcome"] == "AMBER"
    assert "NOT evidence" in out["rows"][3]["check"]
    # Many pairs is enough evidence even with small d-/d.
    assert decide(*frames(twin=0.5), 0.15, DCFG)["rows"][3]["cell"] == "GO"
    # Large d-/d is enough evidence even with few pairs.
    assert decide(*frames(twin=0.05), 0.3, DCFG)["rows"][3]["cell"] == "GO"


def test_decision_reports_per_layer_and_worst():
    out = decide(*frames(e_A_layer2=0.05), 0.3, DCFG)  # layer 2 is the bad one
    assert [p["layer"] for p in out["per_layer"]] == [1, 2]
    assert out["worst"]["row1"]["layer"] == 2 and out["worst"]["row2"]["layer"] == 2
    assert "worst layer 2" in out["rows"][0]["check"]


def test_matched_null_has_fitted_counts():
    from mirror.nulls import matched_null

    nl = matched_null(d=20, d_plus=13, n_pairs_per_block=[0, 5, 31], h=64, seed=3)
    assert (nl.d_plus, nl.d_minus) == (13, 7)
    assert nl.n_pairs == [0, 5, 31] and nl.n_self == [64, 54, 2]
    assert all(np.array_equal(p[p], np.arange(64)) for p in nl.perms)
