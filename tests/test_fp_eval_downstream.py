"""Downstream evaluation of final-phase run directories (scripts/fp_eval_downstream.py): the bridge between
mirror.heal_structured.load_healed_model (MUST-09 layout) and load_student (final-phase layout). CPU only, a stand-in
loader with the tiny random ViT."""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import torch

import mirror.heal_structured as HS
from mirror.final.eval import knn_fast as KF

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    key = f"{name}_fp_eval_downstream_test"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


D = load_script("fp_eval_downstream")


def test_bridge_keeps_must09_runs_and_routes_final_phase_runs(tmp_path):
    m9 = tmp_path / "must09"
    m9.mkdir()
    (m9 / "spec.json").write_text(json.dumps({"format": D.MUST09_FORMAT, "state_file": "healed_state.pt"}),
                                  encoding="utf-8")
    (m9 / "healed_state.pt").write_bytes(b"x")
    new = tmp_path / "final" / "svdllm" / "s1"
    (new / "ckpt").mkdir(parents=True)
    (new / "spec.json").write_text(json.dumps({"method": "svdllm"}), encoding="utf-8")
    assert D.must09_style(m9) and not D.must09_style(new)
    assert D.final_phase_layout(new) and not D.final_phase_layout(m9)
    model = KF.tiny_original().model
    seen = []

    def original(run_dir, device, adapter=None, check=True):
        return "must09 loader", run_dir

    def load_student(run_dir, step, device):
        seen.append(step)
        return None, model, {"method": "svdllm", "seed": 1}

    bridge = D.make_bridge(original, load_student, default_step=780, steps={str(new): 300})
    assert bridge(m9, "cpu")[0] == "must09 loader"
    ad, m, info = bridge(new, "cpu")
    assert seen == [300] and m is model and "load_student" in info["loaded_by"] and info["method"] == "svdllm"
    assert (ad.depth, ad.embed_dim, ad.n_prefix_tokens, ad.patch_size) == (4, 64, 1, 14)    # stand-in adapter
    assert [c[0] for c in bridge.calls] == ["must09", "load_student"]


def test_tier2_linear_probe_of_a_final_phase_run_through_the_bridge(tmp_path, monkeypatch):
    monkeypatch.setattr(HS, "load_healed_model", HS.load_healed_model)          # restored after the test
    run = tmp_path / "runs" / "final" / "A05" / "tiny" / "svdllm" / "s1"
    (run / "ckpt").mkdir(parents=True)
    torch.save({"step": 780}, run / "ckpt" / "step780.pt")
    steps = []

    def load_student(run_dir, step, device):
        steps.append(step)
        ad, model, _ = KF.tiny_loader(run_dir, 11, device)
        return ad, model, {"method": "svdllm", "seed": 1}

    out = tmp_path / "t2"
    res = D.main(["--script", "tier2_linear", "--step", "780", "--", "--tiny", "--variants", "",
                  "--healed-run", f"best_svdllm_s1={run}", "--out", str(out)], load_student=load_student)
    assert steps == [780] and [c[0] for c in res["bridge_calls"]] == ["load_student"]
    rows = pd.read_csv(out / "t2_linear.csv")
    r = rows[rows["variant"] == "best_svdllm_s1"].iloc[0]
    assert 0 <= float(r["top1"]) <= 100
    meta = json.loads((out / "fp_eval_downstream_meta.json").read_text(encoding="utf-8"))
    assert meta["config"]["script"] == "tier2_linear" and meta["extra"]["bridge_calls"][0][2] == 780
