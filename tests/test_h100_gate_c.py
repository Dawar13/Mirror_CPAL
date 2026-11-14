"""scripts/h100_gate_c.py: gate C and the CORE-01 epoch rule (PREREG_NEXT_PHASE.md, MUST-12)."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("h100_gate_c", ROOT / "scripts" / "h100_gate_c.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write(tmp, ratio, twin, minutes, epochs=(0, 50, 100)):
    res, runs = tmp / "results", tmp / "runs"
    for e in epochs:
        d = res / "vit_s8_LR_s0" / f"e{e:03d}"
        d.mkdir(parents=True)
        # older checkpoints carry values that would flip the verdict, so reading the newest one matters
        r, t = (ratio, twin) if e == max(epochs) else (5.0, 0.0)
        (d / "metrics.json").write_text(json.dumps({"scores": {"LR": {"ratio_to_null_1-6": r,
                                                                       "twin_fraction_t50_1-6": t}}}))
    (runs / "vit_s8_none_s0").mkdir(parents=True)
    (runs / "vit_s8_none_s0" / "meta.json").write_text(json.dumps({"extra": {"timing": {"wall_minutes": minutes}}}))
    return ["--results", str(res), "--runs", str(runs)]


@pytest.mark.parametrize("ratio,twin,minutes,rc,epochs", [
    (0.45, 0.6, 18.0, 0, 100),     # pass, fast pilot
    (0.80, 0.20, 25.0, 0, 100),    # both bars inclusive; exactly 25 minutes keeps 100 epochs
    (0.81, 0.6, 30.0, 1, 60),      # ratio fails; slow pilot -> 60 epochs
    (0.45, 0.19, 12.0, 1, 100),    # twin fraction fails
])
def test_gate_c(tmp_path, ratio, twin, minutes, rc, epochs):
    G = load()
    args = write(tmp_path, ratio, twin, minutes)
    assert G.main(args) == rc
    out = json.loads((tmp_path / "results" / "gate_c.json").read_text())
    assert out["pass"] == (rc == 0) and out["core01_epochs"] == epochs
    assert out["lr_metrics"].replace("\\", "/").endswith("e100/metrics.json")     # the newest checkpoint is read


def test_gate_c_missing_inputs(tmp_path):
    G = load()
    assert G.main(["--results", str(tmp_path / "r"), "--runs", str(tmp_path / "x")]) == 2
