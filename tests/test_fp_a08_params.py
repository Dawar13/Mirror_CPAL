"""scripts/fp_a08_params.py dry run: on a tiny DeiT-III the even/odd form stores exactly the formula's count."""
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("MIRROR_FINAL_ALLOW_NO_PREREG", "1")
ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_counts_match(tmp_path):
    spec = importlib.util.spec_from_file_location("fp_a08_params", ROOT / "scripts" / "fp_a08_params.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--dry-run", "--out", str(tmp_path)]) == 0
    rec = json.loads((tmp_path / "params_check.json").read_text())
    assert rec["params_stored"] == rec["params_formula"] > 0 and rec["max_abs_diff_pp"] == 0.0
    assert rec["verdict"] == "PASS"
