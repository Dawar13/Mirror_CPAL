"""The block-drop scorer (mirror/final/heal/blockdrop_score.py) on a tiny model: it scores any model with the
method-selection k-NN (synthetic images without the cache), and the block-dropped student built through the registry
with it removes the requested number of blocks and keeps a valid block map."""
import os
import types

import numpy as np
import pytest
import torch

os.environ.setdefault("MIRROR_FINAL_ALLOW_NO_PREREG", "1")

from mirror.adapters import TimmTinyAdapter  # noqa: E402
from mirror.final.heal.blockdrop_score import msel_scorer  # noqa: E402


def _ctx(tmp_path, adapter):
    return types.SimpleNamespace(adapter=adapter, device="cpu", cache_dir=str(tmp_path / "no_cache"))


def test_scorer_returns_a_percentage_and_is_deterministic(tmp_path):
    ad = TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float32)
    score = msel_scorer(_ctx(tmp_path, ad))
    a, b = score(ad.model), score(ad.model)
    assert 0.0 <= a <= 100.0 and a == b
    assert "never validation" in score.describe["queries"]


def test_block_dropped_student_through_the_registry(tmp_path):
    pytest.importorskip("mirror.final.baselines.dense_students")
    from mirror.final.baselines import dense_students as DS

    ad = TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float32)
    score = msel_scorer(_ctx(tmp_path, ad))
    model, info = DS.block_dropped(ad, 1, score)
    depth = len(ad.model.blocks)
    assert len(model.blocks) == depth - 1
    kept = [int(b) for b in info["kept"]]
    assert sorted(set(range(depth)) - set(kept)) == sorted(int(b) for b in info["dropped"])
    assert len(kept) == depth - 1 and len(set(kept)) == depth - 1
    assert all(0 <= int(k) < depth for k in kept)
    assert np.isfinite(info.get("params_model", 0))
