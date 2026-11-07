"""Test 8: `run_e1.py --tiny` runs end to end and writes every output file; B2 sanity passes."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = [
    "results/e1_layers.csv", "results/e1_mlp.csv", "results/e1_surgery.csv", "results/e1_meta.json",
    "results/e1_operators.npz", "results/E1_REPORT.md",
    "plots/equivariance_vs_depth.png", "plots/breaking_share_vs_depth.png",
]


def run(script, out_root):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--tiny", "--out-root", str(out_root),
                           "--config", str(ROOT / "configs" / "e1.yaml")],
                          capture_output=True, text=True, timeout=600)


def test_run_e1_tiny_writes_all_outputs(tmp_path):
    proc = run("run_e1.py", tmp_path)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    for rel in OUTPUTS:
        assert (tmp_path / rel).is_file(), rel
    report = (tmp_path / "results/E1_REPORT.md").read_text(encoding="utf-8")
    outcomes = ("GO", "PIVOT", "PIVOT: structure but little to save", "KILL", "AMBER")
    assert any(f"**Outcome: {o}.**" in report for o in outcomes)
    assert "matched null" in report and "twin fraction" in report and "(worst)" in report
    assert "—" not in report  # no em-dashes
    meta = json.loads((tmp_path / "results/e1_meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 0 and "git_hash" in meta["git"]
    assert meta["sanity"]["fold_max_abs_diff"] < 1e-4


def test_sanity_b2_tiny_passes(tmp_path):
    proc = run("sanity_b2.py", tmp_path)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    out = json.loads((tmp_path / "results/sanity_b2.json").read_text(encoding="utf-8"))
    assert out["fold_pass"] and out["identity_pass"]
