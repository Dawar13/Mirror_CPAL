"""Tests for scripts/collect_results.py (the paper data package). CPU only, no downloads.

The collector runs on the laptop's current result files into a temporary folder; the tests check
that the main tables exist, that every number in numbers.json that was read directly from a file
equals the value in that file, and that missing experiments are marked "not run" (or "pending").
A second run on a nearly empty copy of the inputs checks that missing inputs are handled.
"""
from __future__ import annotations

import importlib.util
import json
import math
import shutil
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MAIN_TABLES = ["T_main_zeroshot", "T_healing", "T_generality", "T_structure_per_layer", "T_prereg", "T_x_checks",
               "T_e1_history", "T_attention"]


def _load_collector():
    spec = importlib.util.spec_from_file_location("collect_results", ROOT / "scripts" / "collect_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CR = _load_collector()


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    out = tmp_path_factory.mktemp("paper")
    res = CR.main(["--out", str(out), "--skip-scripts"])
    return out, res


def _section(index_text: str, heading_prefix: str) -> str:
    parts = index_text.split("\n## ")
    for p in parts:
        if p.startswith(heading_prefix):
            return p
    raise AssertionError(f"no INDEX section starting with {heading_prefix!r}")


def _close(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=0.0)
    return a == b


# ------------------------------------------------------------------------------------------------
def test_main_tables_exist(package):
    out, _ = package
    for name in MAIN_TABLES:
        assert (out / "tables" / f"{name}.csv").is_file(), name
        assert (out / "tables" / f"{name}.md").is_file(), name
    for f in ("INDEX.md", "numbers.json", "collector_meta.json", "tables/cost_model.md"):
        assert (out / f).is_file(), f
    for p in list((out / "tables").glob("*.md")) + [out / "INDEX.md"]:
        assert "—" not in p.read_text(encoding="utf-8"), f"em-dash in {p.name}"
    idx = (out / "INDEX.md").read_text(encoding="utf-8")
    for exp in ("E1:", "E1b:", "X1:", "X1b:", "X2:", "X3:", "X4:", "F1:", "F2:", "F3:", "G:", "M1:", "M3:", "D:",
                "A:", "F4:", "M2:", "Tier 2:", "Deviations from the plan", "Numbers for the paper"):
        _section(idx, exp)


def test_numbers_equal_their_sources(package):
    out, _ = package
    numbers = json.loads((out / "numbers.json").read_text(encoding="utf-8"))
    assert numbers, "numbers.json is empty"
    checked = 0
    for key, e in numbers.items():
        for field in ("value", "source_file", "seed", "git_hash"):
            assert field in e, (key, field)
        loc = e.get("locator")
        if not loc:
            assert e.get("derived"), key
            continue
        assert _close(CR.resolve_locator(ROOT, loc), e["value"]), (key, loc, e["value"])
        checked += 1
    assert checked >= 50, f"only {checked} numbers carry a locator"


def test_numbers_spot_checks_against_csvs(package):
    out, _ = package
    numbers = json.loads((out / "numbers.json").read_text(encoding="utf-8"))
    knn_path = ROOT / "runs" / "m1" / "m1_knn.csv"
    if knn_path.is_file() and "m1.k6.vexact.top1" in numbers:
        knn = pd.read_csv(knn_path)
        sel = knn[(knn["k"] == 6) & (knn["variant"] == "vexact")]
        assert numbers["m1.k6.vexact.top1"]["value"] == float(sel["top1"].iloc[0])
        assert numbers["m1.k6.vexact.top1"]["source_file"] == "runs/m1/m1_knn.csv"
    # a mean over fit seeds equals the mean of the per-run CSV values
    vals = []
    for p in sorted((ROOT / "runs" / "x2").glob("perlayer_reg_s*/results/e1_surgery.csv")):
        s = pd.read_csv(p)
        sel = s[(s["k"] == 6) & (s["variant"] == "surgery")]
        if len(sel):
            vals.append(float(sel["top1"].iloc[0]))
    if vals and "main.reg.k6.mirror_mlp.top1" in numbers:
        assert math.isclose(numbers["main.reg.k6.mirror_mlp.top1"]["value"], sum(vals) / len(vals), rel_tol=1e-12)
    main = pd.read_csv(out / "tables" / "T_main_zeroshot.csv")
    row = main[(main["model"] == "dinov2_vitb14_reg") & (main["k"] == 6) & (main["variant"] == "mirror_mlp")]
    if vals:
        assert int(row["n_seeds"].iloc[0]) == len(vals)


def test_missing_experiments_marked_not_run(package):
    out, res = package
    idx = (out / "INDEX.md").read_text(encoding="utf-8")
    absent = {"A:": not any((ROOT / "runs" / "a").glob("*/A*_s*.csv")),
              "F4:": not (ROOT / "runs" / "f4" / "f4_p2.json").is_file(),
              "M2:": not (ROOT / "runs" / "m2").exists(),
              "Tier 2:": True}
    for heading, missing in absent.items():
        if missing:
            assert "not run" in _section(idx, heading), heading
    prereg = pd.read_csv(out / "tables" / "T_prereg.csv")
    if absent["F4:"]:
        assert prereg.loc[prereg["id"] == "P2", "verdict"].iloc[0] == "not run"
    # runs planned in jobs/*.txt whose folders hold no results are listed as not run
    for d in (ROOT / "runs" / "x2").glob("*"):
        if d.is_dir() and not any((d / "results").glob("e1_*.csv")):
            assert f"`runs/x2/{d.name}`" in idx and "not run" in _section(idx, "X2:")
    # zero-shot variants that no run produced are rows with status "not run"
    main = pd.read_csv(out / "tables" / "T_main_zeroshot.csv")
    m1_meta = ROOT / "runs" / "m1" / "m1_meta.json"
    m1_model = json.loads(m1_meta.read_text(encoding="utf-8")).get("model") if m1_meta.is_file() else None
    for model in ("dinov2_vitb14_reg", "dinov2_vitb14"):
        if model != m1_model:
            sel = main[(main["model"] == model) & (main["variant"] == "vlocal")]
            assert len(sel) and (sel["status"] == "not run").all(), model
    assert set(main["status"]) <= {"ok", "not run"}
    # healing methods without files are not run, and P3 is pending until all three have healed k-NN
    heal = pd.read_csv(out / "tables" / "T_healing.csv")
    p3 = pd.read_csv(out / "tables" / "T_healing_p3.csv")
    for m in ("mirror", "svd", "random"):
        has = (ROOT / "runs" / "m3_pilot" / m).exists()
        if not has:
            assert heal.loc[heal["method"] == m, "status"].iloc[0] == "not run"
    healed = heal["healed_top1"].notna().all() if "healed_top1" in heal else False
    if not healed:
        assert p3.loc[p3["check"] == "P3.b", "verdict"].iloc[0] in ("pending", "FAIL", "PASS")


def test_cost_model_reproduces_recorded_counts(package):
    out, _ = package
    checks = pd.read_csv(out / "tables" / "cost_model_checks.csv")
    if len(checks):
        assert (checks["rel_diff"] <= 1e-12).all(), checks[checks["rel_diff"] > 1e-12]
    # unit check of the formula on a toy architecture
    arch = {"d": 4, "h": 8, "depth": 3, "T": 5, "n_patches": 4, "patch": 2, "in_chans": 3, "model_params_dense": 1000}
    dense = CR.analytic_cost(arch, [])
    block = 5 * (4 * 16 + 2 * 8 * 4) + 2 * 25 * 4
    assert dense["macs_model"] == 4 * 3 * 4 * 4 + 3 * block
    assert dense["gflops_saved_model_pct"] == 0.0
    conv = CR.analytic_cost(arch, [{"mlp": 30, "attn": 40, "rot": 1}])
    assert conv["macs_converted"] == 5 * 70 + 2 * 25 * 4 + 5 * 16
    assert conv["matrix_params_kept"] == 70 and conv["rotation_params"] == 16
    assert conv["model_params"] == 1000 - (4 * 16 + 2 * 8 * 4) + 70


def test_missing_inputs_are_handled(tmp_path):
    """A copy of the repo with almost no results: the collector must not fail, and must say "not run"
    or "pending"; with only the mirror healing run present, P3.a and P3.b stay pending."""
    root = tmp_path / "root"
    (root / "runs").mkdir(parents=True)
    (root / "results").mkdir()
    for name in ("PREREG_E2.md", "PAPER_EXPERIMENTS.md"):
        if (ROOT / name).is_file():
            shutil.copy2(ROOT / name, root / name)
    src = ROOT / "runs" / "m3_pilot" / "mirror"
    have_mirror = (src / "knn.json").is_file() or (src / "summary.json").is_file()
    if have_mirror:
        dst = root / "runs" / "m3_pilot" / "mirror"
        dst.mkdir(parents=True)
        for f in ("knn.json", "summary.json"):
            if (src / f).is_file():
                shutil.copy2(src / f, dst / f)
        if (src / "final" / "metrics.json").is_file():
            (dst / "final").mkdir()
            shutil.copy2(src / "final" / "metrics.json", dst / "final" / "metrics.json")
    out = tmp_path / "out"
    CR.main(["--root", str(root), "--out", str(out), "--skip-scripts"])
    for name in MAIN_TABLES:
        assert (out / "tables" / f"{name}.md").is_file(), name
    idx = (out / "INDEX.md").read_text(encoding="utf-8")
    for heading in ("E1:", "M1:", "D:", "A:", "F4:"):
        assert "not run" in _section(idx, heading), heading
    heal = pd.read_csv(out / "tables" / "T_healing.csv")
    assert heal.loc[heal["method"] == "svd", "status"].iloc[0] == "not run"
    assert heal.loc[heal["method"] == "random", "status"].iloc[0] == "not run"
    p3 = pd.read_csv(out / "tables" / "T_healing_p3.csv").set_index("check")
    assert p3.loc["P3.a", "verdict"] == "pending"
    assert p3.loc["P3.b", "verdict"] == "pending"
    if have_mirror:
        assert p3.loc["P3.c", "verdict"] in ("PASS", "FAIL")
    numbers = json.loads((out / "numbers.json").read_text(encoding="utf-8"))
    for key, e in numbers.items():
        if e.get("locator"):
            assert _close(CR.resolve_locator(root, e["locator"]), e["value"]), key
