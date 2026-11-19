"""scripts/fp_wait_eval.py: waits for the evaluation records of finished heals."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("fp_wait_eval", ROOT / "scripts" / "fp_wait_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(tmp_path, name, passes, done=True):
    run = tmp_path / name
    (run / "ckpt").mkdir(parents=True)
    for s in (0, 100, 780):
        (run / "ckpt" / f"step{s}.pt").write_bytes(b"x")
    if done:
        (run / "DONE").write_text("ok")
    if passes is not None:
        (run / "eval").mkdir()
        (run / "eval" / "step780.json").write_text(json.dumps({"passes": {p: {"top1": 1.0} for p in passes}}))
    return run


def test_present_and_missing(tmp_path):
    mod = _load()
    a = _run(tmp_path, "a", ["bf16", "fp32"])
    b = _run(tmp_path, "b", ["bf16_q10000"])
    assert mod.last_step(a) == 780
    assert mod.main(["--run-dir", str(a), "--passes", "bf16,fp32", "--poll", "0.01", "--timeout-hours", "0"]) == 0
    assert mod.present(b, 780, ["bf16"], allow_subsets=False) == ["bf16"]
    assert mod.present(b, 780, ["bf16"], allow_subsets=True) == []
    c = _run(tmp_path, "c", None, done=False)
    assert mod.main(["--run-dir", str(c), "--poll", "0.01", "--timeout-hours", "0"]) == 3
    assert mod.main(["--run-dir", str(tmp_path / "nope"), "--poll", "0.01", "--timeout-hours", "0"]) == 2
