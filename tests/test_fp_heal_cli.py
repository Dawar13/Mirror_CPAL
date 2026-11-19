"""scripts/fp_heal.py --dry-run (CPU, tiny random ViT, synthetic uint8 images): the design-doc layout of every
student folder and the evaluation queue, the learning-rate selection, a resumed job equal to an uninterrupted one,
refusal to overwrite a saved state without --resume, and the skip of a finished job."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from mirror.final.heal import multistudent as MS
from mirror.final.heal.students import load_student

ROOT = Path(__file__).resolve().parents[1]
STUDENTS = ("mirror_exact", "svdllm", "original")


def load_fp_heal():
    spec = importlib.util.spec_from_file_location("fp_heal_cli_test", ROOT / "scripts" / "fp_heal.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_dry_run_layout_resume_and_skip(tmp_path, monkeypatch):
    monkeypatch.setenv("MIRROR_FINAL_ALLOW_NO_PREREG", "1")
    FP = load_fp_heal()
    a = FP.main(["--dry-run", "--out-root", str(tmp_path / "a")])
    exp = tmp_path / "a" / "dryrun" / "tiny"
    for m in STUDENTS:
        rd = exp / m / "s0"
        for f in ("spec.json", "ckpt/step0.pt", "ckpt/step2.pt", "ckpt/step4.pt", "train_log.csv", "heldout_log.csv",
                  "heldout.json", "metrics.json", "meta.json", "DONE"):
            assert (rd / f).is_file(), (m, f)
        met = json.loads((rd / "metrics.json").read_text())
        assert met["complete"] and met["steps_done"] == 4 and met["method"] == m and met["lr"] == a["lrs"][m]
        meta = json.loads((rd / "meta.json").read_text())
        assert meta["experiment"] == "dryrun" and meta["seeds"]["heal"] == 0 and "prereg" in meta
        assert meta["extra"]["prereg"].get("allowed_without_check") or meta["prereg"]["ok"]
        adapter, model, info = load_student(rd)                               # the latest step
        assert info["step"] == 4 and info["method"] == m
    assert (exp / "mirror_exact" / "s0" / "structure.npz").is_file()
    sel = json.loads((exp / "svdllm" / "s0" / "lr_selection" / "lr_selection.json").read_text())
    assert sel["lrs"] == [1e-4, 1e-3] and a["lrs"]["svdllm"] == sel["best_lr"] and sel["budget_steps"] == 2
    for lr in ("1e-04", "1e-03"):                     # the trials, in group "lr" (design doc 4.5a)
        trial = tmp_path / "a" / "lr" / "tiny" / f"svdllm_lr{lr}" / "s0"
        assert (trial / "train_log.csv").is_file() and (trial / "heldout.json").is_file()
        assert str(trial) in [t["trial_dir"] for t in sel["trials"]]
    reqs = [json.loads(p.read_text()) for p in (tmp_path / "a" / "_queue").glob("*.json")]
    assert sorted((Path(r["run_dir"]).parent.name, r["step"]) for r in reqs) == sorted(
        (m, s) for m in STUDENTS for s in (0, 2, 4))
    job = Path(a["job_dir"])
    assert (job / "summary.json").is_file() and (job / "meta.json").is_file() and not (job / "train_state.pt").exists()
    assert json.loads((job / "DONE").read_text())["diverged"] == []

    # interrupted in the hook of step 2 (the training state of step 2 is saved just before it)
    orig = MS.save_checkpoint

    def failing(model, path, step, meta=None):
        if int(step) == 2:
            raise KeyboardInterrupt
        return orig(model, path, step, meta)

    monkeypatch.setattr(MS, "save_checkpoint", failing)
    with pytest.raises(KeyboardInterrupt):
        FP.main(["--dry-run", "--out-root", str(tmp_path / "b")])
    monkeypatch.setattr(MS, "save_checkpoint", orig)
    assert not any((tmp_path / "b" / "dryrun" / "tiny" / m / "s0" / "DONE").exists() for m in STUDENTS)
    with pytest.raises(SystemExit, match="--resume"):
        FP.main(["--dry-run", "--out-root", str(tmp_path / "b")])          # a saved state is not overwritten
    b = FP.main(["--dry-run", "--out-root", str(tmp_path / "b"), "--resume"])
    assert b["lrs"] == a["lrs"]
    for m in STUDENTS:
        sa = torch.load(exp / m / "s0" / "ckpt" / "step4.pt", weights_only=True)["state_dict"]
        ckb = tmp_path / "b" / "dryrun" / "tiny" / m / "s0" / "ckpt" / "step4.pt"
        sb = torch.load(ckb, weights_only=True)["state_dict"]
        assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa), m   # bit for bit
        mb = json.loads((tmp_path / "b" / "dryrun" / "tiny" / m / "s0" / "metrics.json").read_text())
        assert mb["resumed_from_step"] == 2 and mb["steps_done"] == 4
    # a finished job is skipped
    again = FP.main(["--dry-run", "--out-root", str(tmp_path / "a")])
    assert again["skipped"]
