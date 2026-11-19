"""scripts/fp_a08_top1.py dry run: a tiny DeiT-III, synthetic images, both passes written with predictions."""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MIRROR_FINAL_ALLOW_NO_PREREG", "1")
ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("fp_a08_top1", ROOT / "scripts" / "fp_a08_top1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dry_run_writes_both_passes(tmp_path):
    mod = _load()
    assert mod.main(["--dry-run", "--out", str(tmp_path), "--passes", "bf16,fp32"]) == 0
    res = json.loads((tmp_path / "eval" / "top1_original.json").read_text())
    for p in ("bf16", "fp32"):
        assert 0.0 <= res[p]["top1"] <= 100.0 and res[p]["n"] == 40
        assert 0.0 <= res[p]["flip_consistency"] <= 100.0
        with np.load(tmp_path / "top1" / f"original_{p}.npz") as z:
            assert z["pred"].shape == z["pred_flip"].shape == z["labels"].shape == (40,)
    assert (tmp_path / "meta.json").exists()


def test_unknown_pass_is_refused():
    mod = _load()
    try:
        mod._passes("bf16,fp16")
    except SystemExit as e:
        assert "fp16" in str(e)
    else:
        raise AssertionError("fp16 should be refused")
