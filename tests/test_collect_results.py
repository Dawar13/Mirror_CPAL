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
               "T_e1_history", "T_attention", "T_ablations", "T_tier2", "T_latency", "T_f4"]


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
              "F4:": not (ROOT / "runs" / "f4" / "f4_p2.json").is_file() and not any((ROOT / "runs" / "f4").glob("f4_*.csv")),
              "M2:": not (ROOT / "runs" / "m2" / "m2_latency.csv").is_file(),
              # ADE20K segmentation has no script or results yet, so the Tier 2 section always names something not run
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


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_new_result_types_from_synthetic_files(tmp_path):
    """M2 latency, F4/P2, Tier-2 linear and transfer probes, ablation seeds and the healing linear columns, read
    from files in the formats scripts/m2_latency.py, f4_rule.py, tier2_linear.py, tier2_transfer.py and
    stage1_ablate.py write. All numbers below are synthetic test fixtures, not results."""
    root = tmp_path / "root"
    (root / "results").mkdir(parents=True)
    for name in ("PREREG_E2.md", "PAPER_EXPERIMENTS.md"):
        if (ROOT / name).is_file():
            shutil.copy2(ROOT / name, root / name)
    m, g = "dinov2_vitb14_reg", "fixturehash"
    lat = []
    for v, k, ms, flop in (("original", 0, 10.0, 1.0), ("dense", 0, 9.0, 1.0), ("vlocal", 6, 8.0, 1.2),
                           ("vexact", 6, 7.5, 1.25)):
        lat.append({"model": m, "variant": v, "label": v, "k": k, "params_model": 1000 - 10 * k, "gflops": 40.0 / flop,
                    "gflops_exec": 40.0 / flop, "kept_frac_converted": 1.0, "rotation_params": 0, "n_rotations": 0,
                    "macs_per_image": 20.0 / flop, "batch": 64, "mode": "compile", "mode_used": "compile",
                    "compile_ok": True, "latency_ms_median": ms, "latency_ms_p10": ms, "latency_ms_p90": ms,
                    "throughput_img_s": 64e3 / ms, "git_hash": g, "seed": 0, "speedup_vs_dense": 9.0 / ms,
                    "speedup_vs_original": 10.0 / ms, "flop_speedup_vs_dense": flop})
    _write_csv(root / "runs" / "m2" / "m2_latency.csv", lat)
    _write_csv(root / "runs" / "m2" / "m2_checks.csv", [
        {"variant": "vlocal", "k": 6, "dtype": dt, key: val, "noise_floor_unit": 1e-4, "criterion": f"{key} < {thr:g}",
         "threshold": thr, "pass": val < thr, "git_hash": g, "seed": 0}
        for dt, key, val, thr in (("float64", "rel_raw", 1e-9, 1e-6), ("float32", "max_abs_unit", 2e-6, 1e-5),
                                  ("bfloat16", "max_abs_unit", 5e-3, 1e-3))])
    (root / "runs" / "m2" / "m2_meta.json").write_text(json.dumps({"model": m, "seed": 0, "git": {"git_hash": g},
                                                                    "date": "fixture"}), encoding="utf-8")
    _write_csv(root / "runs" / "f4" / f"f4_{m}.csv", [
        {"model": m, "block": b, "depth": 12, "s_act": 0.2, "E_act": e, "rule_x": e - 0.2, "drop_mirror": 1.0,
         "drop_svd": dsvd, "rule_y": dsvd - 1.0, "rule_correct": (e - 0.2) * (dsvd - 1.0) > 0, "s_weight": 0.3,
         "E_weight": 0.1, "fit_seed": 0, "seed": 0, "git_hash": g, "prereg_commit": "fixture"}
        for b, e, dsvd in ((1, 0.3, 2.0), (2, 0.1, 3.0))])
    (root / "runs" / "f4" / "f4_p2.json").write_text(json.dumps({
        "hypothesis": "P2 fixture", "threshold": 0.75, "verdict": "INCOMPLETE (1 of 3 pre-registered models complete)",
        "primary_prereg_models": {"n_cases": 2, "n_hits": 1, "fraction": 0.5},
        "primary_all_models_run": {"n_cases": 2, "n_hits": 1, "fraction": 0.5},
        "per_model": {m: {"n_cases": 2, "n_hits": 1, "fraction": 0.5}},
        "secondary": {"weight_only": {"n_cases": 2, "n_hits": 2, "fraction": 1.0}},
        "complete": {m: False}, "prereg_commits": ["fixture"], "git_hashes": [g],
        "written_by": {"git": {"git_hash": g}, "seed": 0}}), encoding="utf-8")
    (root / "runs" / "f4" / "plots").mkdir()
    (root / "runs" / "f4" / "plots" / "fig4_rule.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    t2 = [("original", "original", "original", 0, 60.0), ("healed_mirror_k9", "healed", "mirror", 9, 58.0),
          ("healed_svd_k9", "healed", "svd", 9, 55.0), ("healed_random_k9", "healed", "random", 9, 50.0)]
    _write_csv(root / "runs" / "t2" / "t2_linear.csv", [
        {"model": m, "variant": v, "label": v, "family": f, "method": me, "k": k, "top1": t, "top5": t + 10,
         "mean_per_class_top1": t, "knn_check_top1": t - 2, "chosen_wd": 0.001, "wd_at_grid_edge": v == "healed_svd_k9",
         "final_hit_iter_limit": False, "model_params": 1000, "git_hash": g, "seed": 0, "date": "fixture"}
        for v, f, me, k, t in t2])
    _write_csv(root / "runs" / "t2" / "t2_transfer.csv", [
        {"model": m, "dataset": "pets", "variant": v, "label": v, "family": f, "method": me, "k": k, "top1": t,
         "top5": t + 5, "mean_per_class_top1": t - 1, "chosen_wd": 0.01, "wd_at_grid_edge": False, "n_train": 100,
         "n_test": 50, "num_classes": 37, "hf_id": "fixture", "git_hash": g, "seed": 0}
        for v, f, me, k, t in (("original", "original", "original", 0, 90.0), ("vexact_k6", "m1", "vexact", 6, 85.0))])
    for fs, top in ((0, 20.0), (1, 22.0)):
        rows = [{"model": m, "ablation": "A2", "variant": "generic", "k": 6, "fit_seed": fs, "top1": top,
                 "flip_consistency": 30.0, "drop_points": 60.0 - top, "mlp_params_kept_frac": 0.5, "params_exact": True,
                 "reused_from": "", "info": json.dumps({"pairing": "fixture pairing"}), "git_hash": g, "seed": 0}]
        if fs == 0:
            rows.append({"model": m, "ablation": "A2", "variant": "hungarian", "k": 6, "fit_seed": 0, "top1": 30.0,
                         "flip_consistency": 40.0, "drop_points": 30.0, "mlp_params_kept_frac": 0.5, "params_exact": False,
                         "reused_from": "A1/per_layer/k6/s0",
                         "info": json.dumps({"pairing": "fixture", "projection": "not exact"}), "git_hash": g, "seed": 0})
        _write_csv(root / "runs" / "a" / m / f"A2_s{fs}.csv", rows)
    hm = root / "runs" / "m3_pilot" / "mirror"
    hm.mkdir(parents=True)
    (hm / "knn.json").write_text(json.dumps({"seed": 0, "git_hash": g, "original": {"top1": 60.5, "flip_consistency": 90.0},
                                             "healed": {"top1": 57.0, "flip_consistency": 88.0}}), encoding="utf-8")
    (hm / "summary.json").write_text(json.dumps({"config": {"blocks": list(range(9)), "model": {"name": m}}}),
                                     encoding="utf-8")

    out = tmp_path / "out"
    CR.main(["--root", str(root), "--out", str(out), "--skip-scripts"])
    tab = out / "tables"
    lat_t = pd.read_csv(tab / "T_latency.csv")
    v = lat_t[(lat_t["variant"] == "vlocal") & (lat_t["k"] == 6)].iloc[0]
    assert math.isclose(v["speedup_vs_dense"], 9.0 / 8.0) and math.isclose(v["measured_over_analytic"], (9.0 / 8.0) / 1.2)
    assert str(v["checks_pass"]) == "False" and "bfloat16: FAIL" in v["checks"]
    chk = pd.read_csv(tab / "T_latency_checks.csv")
    assert set(chk["pass"].astype(str)) == {"True", "False"}
    assert len(pd.read_csv(tab / "T_f4.csv")) == 2
    pre = pd.read_csv(tab / "T_prereg.csv").set_index("id")
    assert pre.loc["P2", "value"] == 0.5 and str(pre.loc["P2", "verdict"]).startswith("INCOMPLETE")
    assert (out / "figures" / "ext_f4_fig4_rule.png").is_file()
    t2t = pd.read_csv(tab / "T_tier2.csv").set_index("variant")
    assert math.isclose(t2t.loc["healed_mirror_k9", "margin_svd"], 58.0 - 55.0)
    assert math.isclose(t2t.loc["healed_mirror_k9", "margin_random"], 58.0 - 50.0)
    assert math.isclose(t2t.loc["healed_svd_k9", "drop_vs_original"], 60.0 - 55.0)
    tx = pd.read_csv(tab / "T_tier2_transfer.csv").set_index("variant")
    assert math.isclose(tx.loc["vexact_k6", "drop_vs_original"], 5.0)
    seeds = pd.read_csv(tab / "T_ablations_seeds.csv").set_index("variant")
    assert seeds.loc["generic", "n_seeds"] == 2 and math.isclose(seeds.loc["generic", "top1_mean"], 21.0)
    assert "A1/per_layer/k6/s0" in str(seeds.loc["hungarian", "note"])
    heal = pd.read_csv(tab / "T_healing.csv").set_index("method")
    assert heal.loc["mirror", "linear_top1"] == 58.0 and heal.loc["mirror", "healed_top1"] == 57.0
    p3 = pd.read_csv(tab / "T_healing_p3.csv").set_index("check")
    assert p3.loc["P3.b", "verdict"] == "pending"          # the linear probe never decides P3
    idx = (out / "INDEX.md").read_text(encoding="utf-8")
    m2 = _section(idx, "M2:")
    assert "measured" in m2 and "failing: vlocal k = 6 bfloat16" in m2
    assert "P2" in _section(idx, "F4:") and "pets" in _section(idx, "Tier 2:")
    numbers = json.loads((out / "numbers.json").read_text(encoding="utf-8"))
    located = [e for e in numbers.values() if e.get("locator")]
    assert located
    for e in located:
        assert _close(CR.resolve_locator(root, e["locator"]), e["value"]), e
