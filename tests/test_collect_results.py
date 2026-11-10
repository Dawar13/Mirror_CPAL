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
               "T_e1_history", "T_attention", "T_ablations", "T_tier2", "T_latency", "T_latency_profile", "T_f4",
               "T_e3", "T_strong_baselines", "T_flipavg", "T_f7"]
# Real outputs of the scripts, copied without their binary files (heads, checkpoints, confusion matrices):
#   t2_seg/      runs/t2_seg_smoke (scripts/tier2_ade20k.py, 50-iteration smoke test on the A100, 20 val images)
#   m3_vexact/   scripts/heal_vexact.py --config configs/m3_vexact.yaml --tiny (k4: full conversion of the
#                4-block tiny model) and --tiny --k 2 (CPU, random weights, synthetic data)
#   m2_profile/  scripts/m2_profile.py --tiny --k 2,4 (CPU operator self times)
# They test the file formats only; none of their numbers is a result.
FIX = ROOT / "tests" / "fixtures" / "collect_results"


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
    for exp in ("E1:", "E1b:", "X1:", "X1b:", "X2:", "X3:", "X4:", "F1:", "F2:", "F3:", "G:", "M1:", "M3:", "E3:", "D:",
                "A:", "SB:", "F4:", "M2:", "Tier 2:", "Flip averaging:", "F7:", "Deviations from the plan",
                "Numbers for the paper"):
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
              "M2:": not (ROOT / "runs" / "m2" / "m2_latency.csv").is_file()
              or not (ROOT / "runs" / "m2_profile" / "m2_profile.csv").is_file(),
              "Tier 2:": not all((ROOT / "runs" / p).is_file() for p in ("t2/t2_linear.csv", "t2/t2_transfer.csv",
                                                                         "t2_seg/t2_ade20k.csv"))}
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
    for heading in ("E1:", "M1:", "D:", "A:", "F4:", "SB:", "Flip averaging:", "F7:"):
        assert "not run" in _section(idx, heading), heading
    # E3 without its pre-registration file, and the second-device M2 run: not run, nothing fails
    assert "not run: no `PREREG_E3.md`" in _section(idx, "E3:")
    assert "Second device (`runs/m2_5090/`): not run" in _section(idx, "M2:")
    for name in ("T_e3", "T_e3_checks", "T_strong_baselines", "T_flipavg", "T_f7", "T_latency_5090", "T_latency_devices",
                 "T_tier2_seg_rerun"):
        assert (out / "tables" / f"{name}.csv").is_file(), name
    g5 = pd.read_csv(out / "tables" / "T_e3_checks.csv").set_index("id")
    assert g5.loc["G5", "verdict"] == "pending" and g5.loc["E3.a", "verdict"] == "pending"
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


# ------------------------------------------------------------------------------------------------
# New sources: ADE20K segmentation, V-exact healing, M2 profile, DINOv2-S/L generality, ablations at k = 9
# ------------------------------------------------------------------------------------------------
def _bare_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    (root / "runs").mkdir(parents=True)
    (root / "results").mkdir()
    for name in ("PREREG_E2.md", "PAPER_EXPERIMENTS.md"):
        if (ROOT / name).is_file():
            shutil.copy2(ROOT / name, root / name)
    return root


def _run(root: Path, tmp_path: Path, name: str = "out"):
    out = tmp_path / name
    res = CR.main(["--root", str(root), "--out", str(out), "--skip-scripts"])
    idx = (out / "INDEX.md").read_text(encoding="utf-8")
    numbers = json.loads((out / "numbers.json").read_text(encoding="utf-8"))
    for key, e in numbers.items():
        if e.get("locator"):
            assert _close(CR.resolve_locator(root, e["locator"]), e["value"]), key
        else:
            assert e.get("derived"), key
    for p in list((out / "tables").glob("*.md")) + [out / "INDEX.md"]:
        assert "—" not in p.read_text(encoding="utf-8"), p.name
    return out, res, idx, numbers


def test_tier2_seg_present(tmp_path):
    root = _bare_root(tmp_path)
    shutil.copytree(FIX / "t2_seg", root / "runs" / "t2_seg")
    # a smoke-test folder with other numbers next to it must never be read
    shutil.copytree(FIX / "t2_seg", root / "runs" / "t2_seg_smoke")
    decoy = root / "runs" / "t2_seg_smoke" / "t2_ade20k.csv"
    d = pd.read_csv(decoy)
    d["mIoU"] = d["mIoU"] + 50.0
    d.to_csv(decoy, index=False)
    out, res, idx, numbers = _run(root, tmp_path)
    src = pd.read_csv(FIX / "t2_seg" / "t2_ade20k.csv").set_index("variant")
    seg = pd.read_csv(out / "tables" / "T_tier2_seg.csv").set_index("variant")
    assert set(seg.index) == set(src.index)
    for v in src.index:
        for c in ("mIoU", "aAcc", "mAcc"):
            assert math.isclose(seg.loc[v, c], src.loc[v, c], rel_tol=1e-12), (v, c)
        assert int(seg.loc[v, "model_params"]) == int(src.loc[v, "model_params"])
        assert str(seg.loc[v, "per_class_matches_miou"]) == "True", v
        assert "partial validation" in str(seg.loc[v, "note"])           # the smoke test scored 20 of 2000 images
    assert math.isclose(seg.loc["vexact_k6", "delta_miou_vs_original"],
                        src.loc["vexact_k6", "mIoU"] - src.loc["original", "mIoU"], rel_tol=1e-12)
    assert math.isclose(seg.loc["vexact_k6", "vexact_max_rel_err"], src.loc["vexact_k6", "vexact_max_rel_err"])
    cls = pd.read_csv(out / "tables" / "T_tier2_seg_classes.csv")
    vj = json.loads((FIX / "t2_seg" / "seg" / "dinov2_vitb14_reg" / "vexact_k6.json").read_text(encoding="utf-8"))
    assert len(cls) == 2 * len(vj["metrics"]["per_class_iou"])
    c0 = cls[(cls["variant"] == "vexact_k6") & (cls["class_index"] == 0)].iloc[0]
    assert math.isclose(c0["iou"], vj["metrics"]["per_class_iou"][0]) and int(c0["ade20k_label"]) == 1
    sec = _section(idx, "Tier 2:")
    assert "ADE20K, DINOv2-reg vexact_k6: mIoU" in sec and "T_tier2_seg" in sec
    assert "ADE20K linear segmentation: not run" not in sec
    e = numbers["t2seg.reg.vexact_k6.miou"]
    assert e["source_file"] == "runs/t2_seg/t2_ade20k.csv" and math.isclose(e["value"], src.loc["vexact_k6", "mIoU"])
    assert e["git_hash"] == src.loc["vexact_k6", "git_hash"] and e["seed"] == 0
    assert not any("t2_seg_smoke" in json.dumps(v) for v in numbers.values())
    assert "ADE20K" in (out / "tables" / "T_tier2.md").read_text(encoding="utf-8")
    assert "- [x] `runs/t2_seg/t2_ade20k.csv`" in idx


def test_tier2_seg_absent(tmp_path):
    root = _bare_root(tmp_path)
    shutil.copytree(FIX / "t2_seg", root / "runs" / "t2_seg_smoke")      # only the smoke folder: must not count
    out, res, idx, numbers = _run(root, tmp_path)
    assert res["tables"]["T_tier2_seg"].rows == [] and res["tables"]["T_tier2_seg_classes"].rows == []
    assert "ADE20K linear segmentation: not run" in _section(idx, "Tier 2:")
    assert "- [ ] `runs/t2_seg/t2_ade20k.csv`" in idx
    assert not any(k.startswith("t2seg.") for k in numbers)


def test_vexact_healing_present(tmp_path):
    root = _bare_root(tmp_path)
    shutil.copytree(FIX / "m3_vexact", root / "runs" / "m3_vexact")
    (root / "jobs").mkdir()
    shutil.copy2(ROOT / "jobs" / "stage9_vexact.txt", root / "jobs" / "stage9_vexact.txt")
    out, res, idx, numbers = _run(root, tmp_path)
    tv = res["tables"]["T_healing_vexact"]
    assert sorted(r["k"] for r in tv.rows) == [2, 4, 6, 12]           # fixtures k = 2, 4; planned k = 6, 12
    for k in (6, 12):
        assert tv.find(k=k)["status"] == "not run"
    j = {n: json.loads((FIX / "m3_vexact" / "k4" / n).read_text(encoding="utf-8"))
         for n in ("knn.json", "exactness.json", "flipcheck.json", "summary.json", "final/metrics.json",
                   "lr_selection/lr_selection.json")}
    r4 = tv.find(k=4)
    assert r4["full_conversion"] and r4["tiny"] and "TINY" in r4["status"] and r4["status"].startswith("complete")
    assert r4["healed_top1"] == j["knn.json"]["healed"]["top1"]
    assert r4["zero_shot_top1"] == j["knn.json"]["zero_shot"]["top1"]
    assert r4["healed_flip"] == j["knn.json"]["healed"]["flip_consistency"]
    assert r4["flip_fp32"] == j["flipcheck.json"]["flip_consistency"] and r4["flip_fp32_pass"] is True
    assert r4["exact_max_rel"] == j["exactness.json"]["max_rel_inside"] and r4["exact_pass"] is True
    assert r4["chosen_lr"] == j["lr_selection/lr_selection.json"]["best_lr"]
    assert r4["heldout_loss_before"] == j["final/metrics.json"]["heldout_initial"]["total"]
    assert r4["heldout_loss_after"] == j["final/metrics.json"]["heldout_final"]["total"]
    assert r4["kept_frac_converted"] == j["summary.json"]["method_info"]["converted_kept_frac"]
    assert r4["git_hash"] == j["knn.json"]["git_hash"] and r4["seed"] == 0
    assert r4["block_targets"] == j["summary.json"]["teacher"]["block_targets"]
    r2 = tv.find(k=2)
    assert r2["full_conversion"] is False and r2.get("flip_fp32") is None and r2["status"].startswith("complete")
    md = (out / "tables" / "T_healing.md").read_text(encoding="utf-8")
    assert "V-exact healing" in md and "not comparable with the M3 pilot's" in md
    sec = _section(idx, "M3:")
    assert "V-exact healing" in sec and "k = 12: not run" in sec          # the model depth is unknown in this root
    assert "k = 4 (full conversion): status complete" in sec
    e = numbers["m3vx.k4.healed"]
    assert e["locator"]["file"] == "runs/m3_vexact/k4/knn.json" and e["git_hash"] == j["knn.json"]["git_hash"]
    assert numbers["m3vx.k4.flip_fp32"]["locator"]["file"] == "runs/m3_vexact/k4/flipcheck.json"
    assert "m3vx.k2.flip_fp32" not in numbers
    assert any("TINY smoke-test run" in d for d in res["meta"]["deviations"])
    assert "- [x] `runs/m3_vexact/k4/knn.json`" in idx and "- [ ] `runs/m3_vexact/k12/knn.json`" in idx
    # a run that has only finished its learning-rate selection
    lr_only = root / "runs" / "m3_vexact" / "k3" / "lr_selection"
    lr_only.mkdir(parents=True)
    shutil.copy2(FIX / "m3_vexact" / "k2" / "lr_selection" / "lr_selection.json", lr_only / "lr_selection.json")
    _, res2, _, _ = _run(root, tmp_path, "out2")
    assert res2["tables"]["T_healing_vexact"].find(k=3)["status"] == "learning-rate selection only"


def test_vexact_healing_absent(tmp_path):
    root = _bare_root(tmp_path)
    out, res, idx, numbers = _run(root, tmp_path)                       # no jobs, no config, no runs
    assert res["tables"]["T_healing_vexact"].rows == []
    assert "V-exact healing: not run" in _section(idx, "M3:")
    (root / "jobs").mkdir()
    shutil.copy2(ROOT / "jobs" / "stage9_vexact.txt", root / "jobs" / "stage9_vexact.txt")
    out, res, idx, numbers = _run(root, tmp_path, "out2")             # planned in the job list, not run
    tv = res["tables"]["T_healing_vexact"]
    assert {r["k"]: r["status"] for r in tv.rows} == {6: "not run", 12: "not run"}
    sec = _section(idx, "M3:")
    assert "k = 6: not run" in sec and "k = 12: not run" in sec
    assert not any(k.startswith("m3vx.") for k in numbers)


def test_m2_profile_present(tmp_path):
    root = _bare_root(tmp_path)
    shutil.copytree(FIX / "m2_profile", root / "runs" / "m2_profile")
    out, res, idx, numbers = _run(root, tmp_path)
    prof = pd.read_csv(FIX / "m2_profile" / "m2_profile.csv")
    sub = pd.read_csv(FIX / "m2_profile" / "m2_submodules.csv")
    tp = res["tables"]["T_latency_profile"]
    keys = ["model", "variant", "k", "mode", "batch"]
    assert len(tp.rows) == prof.groupby(keys).ngroups
    dense = prof[(prof["variant"] == "dense")].iloc[0]
    for r in tp.rows:
        sel = prof[(prof["variant"] == r["variant"]) & (prof["k"] == r["k"]) & (prof["mode"] == r["mode"])]
        assert r["latency_ms_median"] == float(sel["latency_ms_median"].iloc[0])
        assert r["ms_gemm"] == float(sel[sel["category"] == "gemm"]["ms_per_forward"].iloc[0])
        assert r["share_memory"] == float(sel[sel["category"] == "memory"]["share"].iloc[0])
        assert math.isclose(r["gap_ms"], r["latency_ms_median"] - float(dense["latency_ms_median"]), abs_tol=1e-12)
        assert abs(r["gap_check"]) < 1e-9                                 # category changes + idle change = gap
    b = int(dense["batch"])
    v = tp.find(model="tiny", variant="vlocal", k=2, mode="eager", batch=b)
    s = sub[(sub["variant"] == "vlocal") & (sub["k"] == 2)]
    for part in ("mlp", "attention", "rotation"):
        assert math.isclose(v[f"{part}_delta_ms"], float(s[s["part"] == part]["delta_ms"].sum()), rel_tol=1e-12)
    assert math.isclose(v["sub_remainder_ms"], v["gap_ms"] - v["sub_sum_delta_ms"], rel_tol=1e-12, abs_tol=1e-12)
    assert tp.find(model="tiny", variant="vexact", k=2, mode="eager", batch=b).get("rotation_delta_ms") is None
    md = (out / "tables" / "T_latency_profile.md").read_text(encoding="utf-8")
    assert "Gap to dense split into categories" in md and "Sub-module split" in md
    sec = _section(idx, "M2:")
    assert "Profile (why the converted models are slower than dense)" in sec and "TINY smoke test" in sec
    assert "vlocal k = 2 (eager)" in sec and "gap to dense" in sec
    e = numbers[f"m2p.tiny.vlocal.k2.b{b}.eager.d_gemm"]
    assert e["locator"]["file"] == "runs/m2_profile/m2_profile.csv" and e["git_hash"] == dense["git_hash"]
    assert any(d.startswith("M2 profile:") for d in res["meta"]["deviations"])
    assert "- [x] `runs/m2_profile/m2_profile.csv`" in idx


def test_m2_profile_absent(tmp_path):
    root = _bare_root(tmp_path)
    out, res, idx, numbers = _run(root, tmp_path)
    assert res["tables"]["T_latency_profile"].rows == []
    assert "Profile (scripts/m2_profile.py): not run" in _section(idx, "M2:")
    assert "- [ ] `runs/m2_profile/m2_profile.csv`" in idx
    assert (out / "tables" / "T_latency_profile.md").is_file()


def _g_fixture(root: Path, name: str, model: str, depth: int) -> None:
    """A G run in scripts/run_e1.py's format, built from the real DeiT-III run (synthetic: model renamed and, for a
    24-block model, blocks 1 to 12 repeated as 13 to 24 and the k = 6 surgery rows relabelled k = 12)."""
    src = ROOT / "runs" / "g" / "deit3_b16_s0" / "results"
    dst = root / "runs" / "g" / name / "results"
    dst.mkdir(parents=True)
    lay, mlp, sur = (pd.read_csv(src / f"e1_{n}.csv") for n in ("layers", "mlp", "surgery"))
    meta = json.loads((src / "e1_meta.json").read_text(encoding="utf-8"))
    meta["config"]["model"]["name"] = model
    if depth == 24:
        extra = lay[lay["layer"] >= 1].copy()
        extra["layer"] += 12
        lay = pd.concat([lay, extra], ignore_index=True)
        extra = mlp.copy()
        extra["block"] += 12
        mlp = pd.concat([mlp, extra], ignore_index=True)
        sur.loc[sur["k"] == 6, "k"] = 12
        sur.loc[sur["k"] == 12, "top1"] += 0.5                              # so k = 12 differs from the source rows
        meta["config"]["decision"].update({"blocks": [1, 12], "surgery_k": 12})
        meta["config"]["surgery"]["ks"] = [12]
    lay.to_csv(dst / "e1_layers.csv", index=False)
    mlp.to_csv(dst / "e1_mlp.csv", index=False)
    sur.to_csv(dst / "e1_surgery.csv", index=False)
    (dst / "e1_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_generality_dinov2_s_and_l(tmp_path):
    if not (ROOT / "runs" / "g" / "deit3_b16_s0" / "results" / "e1_meta.json").is_file():
        pytest.skip("needs runs/g/deit3_b16_s0 to build the fixture")
    root = _bare_root(tmp_path)
    _g_fixture(root, "dinov2_s_reg_s0", "dinov2_vits14_reg", 12)
    _g_fixture(root, "dinov2_l_reg_s0", "dinov2_vitl14_reg", 24)
    out, res, idx, numbers = _run(root, tmp_path)
    tg = res["tables"]["T_generality"]
    s, l_ = tg.find(model="dinov2_vits14_reg", fit_seed=0), tg.find(model="dinov2_vitl14_reg", fit_seed=0)
    assert s["model_label"] == "DINOv2-S/14 reg" and l_["model_label"] == "DINOv2-L/14 reg"
    assert (s["k_half"], s["depth"], s["blocks"]) == (6, 12, "1-6")
    assert (l_["k_half"], l_["depth"], l_["blocks"]) == (12, 24, "1-12")
    sur = pd.read_csv(root / "runs" / "g" / "dinov2_l_reg_s0" / "results" / "e1_surgery.csv")
    assert l_["top1_surgery_half"] == float(sur[(sur["k"] == 12) & (sur["variant"] == "surgery")]["top1"].iloc[0])
    ss = res["tables"]["T_structure_summary"]
    assert ss.find(model="dinov2_vitl14_reg", blocks="1-12") and ss.find(model="dinov2_vitl14_reg", blocks="13-24")
    assert len([r for r in res["tables"]["T_structure_per_layer"].rows if r["model"] == "dinov2_vitl14_reg"]) == 25
    sec = _section(idx, "G:")
    assert "DINOv2-L/14 reg (fit seed 0" in sec and "k = 12: surgery" in sec and "24 blocks, k = 12" in sec
    assert numbers["g.reg_l.s0.surgery"]["locator"]["where"] == {"k": 12, "variant": "surgery"}
    assert (out / "figures" / "fig_structure_summary.png").is_file() and (out / "figures" / "fig_eA_vs_depth.png").is_file()
    assert "- [x] `runs/g/dinov2_l_reg_s0/results/e1_surgery.csv`" in idx


def test_generality_dinov2_s_and_l_absent(tmp_path):
    root = _bare_root(tmp_path)
    (root / "jobs").mkdir()
    shutil.copy2(ROOT / "jobs" / "stage8_more.txt", root / "jobs" / "stage8_more.txt")
    shutil.copytree(ROOT / "configs", root / "configs")                 # the configs' base chain
    out, res, idx, numbers = _run(root, tmp_path)
    sec = _section(idx, "G:")
    assert "not run" in sec and "DINOv2-S/14 reg (planned: `runs/g/dinov2_s_reg_s0`)" in sec
    assert "DINOv2-L/14 reg (planned: `runs/g/dinov2_l_reg_s0`)" in sec
    assert "- [ ] `runs/g/dinov2_l_reg_s0/results/e1_meta.json`" in idx


def test_ablations_every_k_and_fit_seed(tmp_path):
    src = ROOT / "runs" / "a" / "dinov2_vitb14_reg"
    if not (src / "A1_s0.csv").is_file() or not (src / "A4_s0.csv").is_file():
        pytest.skip("needs the ablation CSVs under runs/a/dinov2_vitb14_reg")
    root = _bare_root(tmp_path)
    dst = root / "runs" / "a" / "dinov2_vitb14_reg"
    dst.mkdir(parents=True)
    for f in src.glob("A*_s*.csv"):
        shutil.copy2(f, dst / f.name)
    # synthetic rows: A1 and A2 at k = 9 for fit seeds 0, 1, 2 (copies of the k = 6 rows with other numbers)
    want = {}
    for abl in ("A1", "A2"):
        for fs in (0, 1, 2):
            p = dst / f"{abl}_s{fs}.csv"
            if not p.is_file():
                pytest.skip(f"needs {p.name}")
            d = pd.read_csv(p)
            d = d[d["k"] != 9]          # real k = 9 rows (once they exist) are replaced by the synthetic ones
            k9 = d[d["k"] == 6].copy()
            k9["k"] = 9
            k9["top1"] = k9["top1"] - 1.0 - 0.25 * fs
            k9["drop_points"] = k9["top1_original"] - k9["top1"]
            for _, r in k9.iterrows():
                want.setdefault((abl, r["variant"]), []).append(float(r["top1"]))
            pd.concat([d, k9], ignore_index=True).to_csv(p, index=False)
    # A4 and A5 for fit seeds 1 and 2 (copies of the seed 0 rows)
    for abl in ("A4", "A5"):
        d = pd.read_csv(dst / f"{abl}_s0.csv")
        for fs in (1, 2):
            e = d.copy()
            e["fit_seed"] = fs
            e["top1"] = e["top1"] + 0.1 * fs
            e.to_csv(dst / f"{abl}_s{fs}.csv", index=False)
    # one row present in two files: kept once, reported as a deviation
    pd.read_csv(dst / "A4_s1.csv").head(1).to_csv(dst / "A4_s1b.csv", index=False)
    out, res, idx, numbers = _run(root, tmp_path)
    tas, ta, tx = (res["tables"][n] for n in ("T_ablations_seeds", "T_ablations", "T_ablations_a2_extra"))
    m = "dinov2_vitb14_reg"
    for (abl, var), vals in want.items():
        r = tas.find(model=m, ablation=abl, variant=var, k=9)
        assert r["n_seeds"] == 3 and r["fit_seeds"] == "0,1,2", (abl, var)
        mean = sum(vals) / 3
        assert math.isclose(r["top1_mean"], mean, rel_tol=1e-12)
        assert math.isclose(r["top1_std"], math.sqrt(sum((x - mean) ** 2 for x in vals) / 2), rel_tol=1e-9)
        assert r["top1_per_seed"].count("s") == 3
        assert tas.find(model=m, ablation=abl, variant=var, k=6)["n_seeds"] >= 1
        for fs in (0, 1, 2):
            assert ta.find(model=m, ablation=abl, variant=var, k=9, fit_seed=fs) is not None
    a4 = tas.find(model=m, ablation="A4", variant=str(pd.read_csv(dst / "A4_s0.csv")["variant"].iloc[0]), k=6)
    assert a4["n_seeds"] == 3 and a4["fit_seeds"] == "0,1,2"
    assert len([r for r in ta.rows if r["ablation"] == "A4" and r["fit_seed"] == 1]) == len(pd.read_csv(dst / "A4_s1.csv"))
    assert any("appears in more than one CSV" in d for d in res["meta"]["deviations"])
    for var in ("hungarian", "generic"):
        if ("A2", var) in want:
            for k in (6, 9):
                assert tx.find(model=m, variant=var, k=k, fit_seed=1) is not None
            assert f"a.reg.A2x.{var}.k9.s1.top1" in numbers and f"a.reg.A2x.{var}.k6.s1.top1" in numbers
    sec = _section(idx, "A:")
    assert "A1 (DINOv2-reg, k = 6)" in sec and "A1 (DINOv2-reg, k = 9)" in sec and "A4 (DINOv2-reg, k = 6)" in sec
    gap = [line for line in sec.splitlines() if line.startswith("- Not yet run against the plan")]
    assert not gap or ("A1 has" not in gap[0] and "A2 has" not in gap[0])
    e = numbers["a.reg.A1.per_layer.k9.top1"]
    assert math.isclose(e["value"], sum(want[("A1", "per_layer")]) / 3, rel_tol=1e-12) and e["fit_seed"] == "0,1,2"
    seeds = pd.read_csv(out / "tables" / "T_ablations_seeds.csv")
    assert set(seeds["k"]) == {6, 9}
    # no row twice: one per (ablation, variant, k, fit seed), one seed summary per (ablation, variant, k)
    keys = [(r["model"], r["ablation"], r["variant"], r["k"], r["fit_seed"]) for r in ta.rows]
    assert len(keys) == len(set(keys))
    skeys = [(r["model"], r["ablation"], r["variant"], r["k"]) for r in tas.rows]
    assert len(skeys) == len(set(skeys))
    for r in tas.rows:
        assert r["n_seeds"] == len([x for x in ta.rows if (x["model"], x["ablation"], x["variant"], x["k"]) ==
                                    (r["model"], r["ablation"], r["variant"], r["k"])])


# ------------------------------------------------------------------------------------------------
# Consistency rules: one canonical number per setting, parameter ratios, one label per method, flags
# ------------------------------------------------------------------------------------------------
def test_canonical_headline_rule_unit():
    """mark_canonical_main: an X2 row is canonical; an M1 row of the same setting is not (with the per-seed
    difference); an M1 row at a k that X2 did not run stays canonical. Synthetic numbers."""
    m = "dinov2_vitb14_reg"
    T = CR.Table("T_main_zeroshot", "t", ["model", "k", "variant"], [])

    def rec(top1, fs):
        return {"top1": top1, "src": f"runs/x2/perlayer_reg_s{fs}/results/e1_surgery.csv", "seed": 0,
                "where": {"k": 6, "variant": "surgery"}}

    x2 = T.add({"model": m, "k": 6, "variant": "mirror_mlp", "status": "ok", "experiment": "X2", "n_seeds": 2,
                "top1_mean": 70.5, "_per_seed": {0: rec(70.0, 0), 1: rec(71.0, 1)}})
    m1 = T.add({"model": m, "k": 6, "variant": "mirror_mlp_m1", "status": "ok", "experiment": "M1", "n_seeds": 1,
                "top1_mean": 70.25, "_per_seed": {0: rec(70.25, 0)}})
    only = T.add({"model": m, "k": 12, "variant": "mirror_mlp_m1", "status": "ok", "experiment": "M1", "n_seeds": 1,
                  "top1_mean": 60.0, "_per_seed": {0: rec(60.0, 0)}})
    gone = T.add({"model": m, "k": 12, "variant": "mirror_mlp", "status": "not run", "experiment": "X2"})
    CR.mark_canonical_main(T)
    assert x2["canonical"] == "yes" and only["canonical"] == "yes" and gone["canonical"] == ""
    assert m1["canonical"] == "no" and math.isclose(m1["minus_canonical"], 0.25)
    assert "X2" in m1["remark"] and m1["remark_id"] >= 1 and "only source" in only["remark"]
    v, x = CR.canonical_seed_value(T, m, 6, "mirror", 1)
    assert v == 71.0 and x["src"].endswith("perlayer_reg_s1/results/e1_surgery.csv")
    assert CR.canonical_seed_value(T, m, 6, "mirror", 2) == (None, None)


def _abl_row(m, abl, var, k, fs, top1, ratio, blocks="1 2 3 4 5 6", reused="", n_fit=2000):
    return {"model": m, "ablation": abl, "variant": var, "k": k, "fit_seed": fs, "blocks": blocks, "top1": top1,
            "flip_consistency": 90.0, "drop_points": 80.0 - top1, "mlp_params_kept_frac": 0.58 * ratio,
            "params_vs_mirror_first_k": ratio, "params_exact": True, "n_fit_images": n_fit, "n_query": 50000,
            "reused_from": reused, "info": "{}", "git_hash": "fixturehash", "seed": 0}


def test_ablations_canonical_and_parameter_ratio_synthetic(tmp_path):
    """build_ablations with T_main_zeroshot: the default cell gets the X2 value of its fit seed and the difference;
    every row carries its parameter ratio and whether it is matched; A4 and A1/A2 budget notes are written.
    All numbers are synthetic fixtures."""
    root = _bare_root(tmp_path)
    m = "dinov2_vitb14_reg"
    d = root / "runs" / "a" / m
    for fs, pl in ((0, 77.11), (1, 77.04)):
        _write_csv(d / f"A1_s{fs}.csv", [_abl_row(m, "A1", "per_layer", 6, fs, pl, 1.0),
                                         _abl_row(m, "A1", "svd", 6, fs, 8.77, 0.999),
                                         _abl_row(m, "A1", "identity", 6, fs, 24.5, 1.18)])
    _write_csv(d / "A4_s0.csv", [_abl_row(m, "A4", "first", 6, 0, 77.11, 1.0, reused="A1/per_layer/k6/s0"),
                                 _abl_row(m, "A4", "last", 6, 0, 78.67, 1.53, blocks="7 8 9 10 11 12"),
                                 _abl_row(m, "A4", "lowest_share", 6, 0, 78.67, 1.53, blocks="7 8 9 10 11 12",
                                          reused="A4/last/k6/s0")])
    _write_csv(d / "A3_s0.csv", [_abl_row(m, "A3", "n2000", 6, 0, 77.09, 1.0),
                                 _abl_row(m, "A3", "n100", 6, 0, 77.03, 0.998, n_fit=100)])
    Tm = CR.Table("T_main_zeroshot", "t", ["model", "k", "variant"], [])
    per = {fs: {"top1": v, "src": f"runs/x2/perlayer_reg_s{fs}/results/e1_surgery.csv", "seed": 0,
                "where": {"k": 6, "variant": "surgery"}} for fs, v in ((0, 77.094), (1, 77.088))}
    per_svd = {fs: {"top1": 8.77, "src": f"runs/x2/perlayer_reg_s{fs}/results/e1_surgery.csv", "seed": 0,
                    "where": {"k": 6, "variant": "svd"}} for fs in (0, 1)}
    Tm.add({"model": m, "k": 6, "variant": "mirror_mlp", "status": "ok", "experiment": "X2", "_per_seed": per})
    Tm.add({"model": m, "k": 6, "variant": "svd_mlp", "status": "ok", "experiment": "X2", "_per_seed": per_svd})
    T, S, X, files, dups = CR.build_ablations(CR.Paths(root), Tm)
    pl0 = T.find(model=m, ablation="A1", variant="per_layer", k=6, fit_seed=0)
    assert pl0["canonical_top1"] == 77.094 and math.isclose(pl0["minus_canonical"], 77.11 - 77.094)
    assert pl0["_loc"]["canonical_top1"]["file"] == "runs/x2/perlayer_reg_s0/results/e1_surgery.csv"
    assert T.find(model=m, ablation="A4", variant="first", k=6, fit_seed=0)["canonical_top1"] == 77.094
    assert T.find(model=m, ablation="A3", variant="n2000", k=6, fit_seed=0)["canonical_top1"] == 77.094
    assert T.find(model=m, ablation="A3", variant="n100", k=6, fit_seed=0).get("canonical_top1") is None
    assert T.find(model=m, ablation="A1", variant="identity", k=6, fit_seed=0).get("canonical_top1") is None
    assert T.find(model=m, ablation="A1", variant="svd", k=6, fit_seed=1)["minus_canonical"] == 0
    s_pl = S.find(model=m, ablation="A1", variant="per_layer", k=6)
    assert math.isclose(s_pl["canonical_top1_mean"], (77.094 + 77.088) / 2)
    assert "quote X2" in s_pl["note"]
    assert "identical to X2" in S.find(model=m, ablation="A1", variant="svd", k=6)["note"]
    # parameter ratio on every row; matched within 1%
    for r in T.rows:
        assert r["params_vs_mirror_first_k"] is not None
        assert r["params_matched"] == (abs(r["params_vs_mirror_first_k"] - 1) <= 0.01)
    last = S.find(model=m, ablation="A4", variant="last", k=6)
    assert last["params_matched"] is False and math.isclose(last["params_vs_first_k_mean"], 1.53)
    assert "not parameter-matched" in last["note"]
    assert "A4/last" in S.find(model=m, ablation="A4", variant="lowest_share", k=6)["note"]
    notes = " ".join(S.notes)
    assert "A4 (DINOv2-reg, k = 6) is not parameter-matched" in notes and "same blocks as last" in notes
    assert "A1 identity 1.18x" in notes and "still collapse" in notes and "Canonical number" in notes
    assert "params vs default" in [h for h, _, _ in T.md_cols] and "params vs default" in [h for h, _, _ in S.md_cols]


def test_canonical_headline_and_ratios_real(package):
    """On the repository's own results: the headline is the X2 value everywhere it is quoted; same-setting rows
    from other scripts are marked; the ablation tables carry the parameter ratio of every row."""
    out, res = package
    x2 = {}
    for p in sorted((ROOT / "runs" / "x2").glob("perlayer_reg_s*/results/e1_surgery.csv")):
        s = pd.read_csv(p)
        sel = s[(s["k"] == 6) & (s["variant"] == "surgery")]
        if len(sel):
            x2[int(p.parent.parent.name.rsplit("_s", 1)[1])] = float(sel["top1"].iloc[0])
    if not x2:
        pytest.skip("needs the X2 runs of the registers model")
    T = res["tables"]
    main = T["T_main_zeroshot"]
    head = main.find(model="dinov2_vitb14_reg", k=6, variant="mirror_mlp")
    assert head["canonical"] == "yes" and head["experiment"] == "X2"
    assert math.isclose(head["top1_mean"], sum(x2.values()) / len(x2), rel_tol=1e-12)
    numbers = json.loads((out / "numbers.json").read_text(encoding="utf-8"))
    for key in ("main.reg.k6.mirror_mlp.top1", "main.reg.k6.mirror_mlp.top1_mean"):
        if key in numbers:
            assert all(f.startswith("runs/x2/") for f in str(numbers[key]["source_file"]).split(";")), key
            assert math.isclose(numbers[key]["value"], head["top1_mean"], rel_tol=1e-12)
    dup = main.find(model="dinov2_vitb14_reg", k=6, variant="mirror_mlp_m1")
    if dup is not None and dup["status"] == "ok":
        assert dup["canonical"] == "no" and "X2" in dup["remark"]
        assert "not canonical" in _section((out / "INDEX.md").read_text(encoding="utf-8"), "M1:")
    abl = T["T_ablations"]
    if abl.rows:
        src = {}
        for f in (ROOT / "runs" / "a").glob("*/A*_s*.csv"):
            for _, r in pd.read_csv(f).iterrows():
                src[(r["model"], r["ablation"], r["variant"], int(r["k"]), int(r["fit_seed"]))] = r
        for r in abl.rows:
            s = src[(r["model"], r["ablation"], r["variant"], r["k"], r["fit_seed"])]
            assert math.isclose(r["params_vs_mirror_first_k"], float(s["params_vs_mirror_first_k"]), rel_tol=1e-12)
        for fs, v in x2.items():
            r = abl.find(model="dinov2_vitb14_reg", ablation="A1", variant="per_layer", k=6, fit_seed=fs)
            if r is not None:
                assert r["canonical_top1"] == v and math.isclose(r["minus_canonical"], r["top1"] - v, abs_tol=1e-12)
        md = (out / "tables" / "T_ablations.md").read_text(encoding="utf-8")
        assert "params vs default" in md and "matched (1%)" in md and "X2 (canonical)" in md
        a4 = T["T_ablations_seeds"].find(model="dinov2_vitb14_reg", ablation="A4", variant="last", k=6)
        if a4 is not None and a4["params_vs_first_k_mean"] > 1.01:
            assert a4["params_matched"] is False
            assert any("A4" in n and "not parameter-matched" in n for n in T["T_ablations_seeds"].notes)
    t2 = T["T_tier2"].find(model="dinov2_vitb14_reg", variant="mirror_mlp_k6")
    if t2 is not None and 0 in x2:
        assert t2["tier1_knn_top1"] == x2[0] and t2["tier1_same_model"].startswith("no")


def test_one_label_per_method(package):
    out, res = package
    banned = ("Mirror per-layer", "Mirror Surgery, MLP only", "Mirror, MLP only", "SVD at the MLP-only budget",
              "SVD at the V-local budget")
    for p in list((out / "tables").glob("T_*.md")) + [out / "INDEX.md"]:
        text = p.read_text(encoding="utf-8")
        for b in banned:
            assert b not in text, (p.name, b)
    main = res["tables"]["T_main_zeroshot"]
    labels = {r["variant"]: r["variant_label"] for r in main.rows if r["k"]}
    assert labels.get("mirror_mlp") == labels.get("mirror_mlp_m1") == CR.LABEL_MIRROR_MLP
    for t in ("T_tier2", "T_tier2_transfer", "T_tier2_seg"):
        for r in res["tables"][t].rows:
            fm = (r.get("family"), r.get("method"))
            if fm in CR.METHOD_LABEL:
                assert r["label"] == CR.METHOD_LABEL[fm], (t, r["variant"])
    for r in res["tables"]["T_latency"].rows:
        if r["variant"] in CR.M2_LABEL:
            assert r["label"] == CR.M2_LABEL[r["variant"]]
    assert CR.method_label("mlp", "mirror", "anything") == CR.METHOD_LABEL[("m1", "mlp_only")] == CR.LABEL_MIRROR_MLP


def test_timing_outlier_and_ceiling_flags(tmp_path):
    """A converted variant measured faster than its FLOPs allow is flagged (kept); a transfer dataset whose
    variants tie with an original at ceiling is flagged. Synthetic numbers."""
    root = _bare_root(tmp_path)
    m, g = "dinov2_vitb14_reg", "fixturehash"
    lat = []
    for v, k, b, sp, flop in (("dense", 0, 64, 1.0, 1.0), ("mlp_only", 6, 1, 0.84, 1.10), ("mlp_only", 6, 64, 1.25, 1.10),
                              ("mlp_only", 6, 256, 0.91, 1.10)):
        lat.append({"model": m, "variant": v, "label": v, "k": k, "params_model": 1000, "gflops": 40.0 / flop,
                    "macs_per_image": 20.0 / flop, "batch": b, "mode": "compile", "mode_used": "compile",
                    "latency_ms_median": 1.0 / sp, "throughput_img_s": 1.0, "git_hash": g, "seed": 0,
                    "speedup_vs_dense": sp, "speedup_vs_original": sp, "flop_speedup_vs_dense": flop})
    _write_csv(root / "runs" / "m2" / "m2_latency.csv", lat)
    rows = [("original", "original", "original", 0, 99.69), ("a_k6", "m1", "vexact", 6, 99.69),
            ("b_k9", "healed", "mirror", 9, 99.69), ("c_k9", "healed", "random", 9, 99.56)]
    _write_csv(root / "runs" / "t2" / "t2_transfer.csv", [
        {"model": m, "dataset": ds, "variant": v, "label": v, "family": f, "method": me, "k": k,
         "top1": t if ds == "flowers102" else t - 10 - k, "n_train": 2040, "n_test": 6149, "git_hash": g, "seed": 0}
        for ds in ("flowers102", "dtd") for v, f, me, k, t in rows])
    out, res, idx, numbers = _run(root, tmp_path)
    L = res["tables"]["T_latency"]
    flagged = [r for r in L.rows if r.get("timing_flag")]
    assert [(r["variant"], r["batch"]) for r in flagged] == [("mlp_only", 64)]
    assert "0.840x at batch 1" in flagged[0]["timing_note"] and "0.910x at batch 256" in flagged[0]["timing_note"]
    assert len(L.rows) == 4 and "Probable timing outlier" in _section(idx, "M2:")
    X = res["tables"]["T_tier2_transfer"]
    assert all(r.get("dataset_at_ceiling") for r in X.rows if r["dataset"] == "flowers102")
    assert not any(r.get("dataset_at_ceiling") for r in X.rows if r["dataset"] == "dtd")
    assert any(n.startswith("flowers102") and "does not separate the methods" in n for n in X.notes)
    assert "flowers102 is at ceiling" in _section(idx, "Tier 2:")


# ------------------------------------------------------------------------------------------------
# E3 (PREREG_E3.md), strong baselines, flip averaging, second-device M2, ADE20K rerun, F7
# ------------------------------------------------------------------------------------------------
# Real outputs of the scripts' --tiny CPU modes (paths rewritten to tiny_out/, binaries dropped):
#   e3_tiny/runs/   scripts/heal.py --config configs/e3_svd_{raw,sym}.yaml --tiny; scripts/heal_vexact.py --config
#                   configs/e3_vexact_s1.yaml / e3_random.yaml / m3_vexact.yaml --tiny (and --lr-probe-only); the Tier-2
#                   probes of scripts/tier2_linear.py --tiny on each (runs/e3/t2_*, runs/t2)
#   sb/             scripts/strong_baselines.py --tiny --ks 1,2
#   t1_flipavg/     scripts/flip_avg_baseline.py --tiny --variants original --vexact-dir <tiny m3_vexact_5090> --vexact-fp32
#   f7/             scripts/f7_visuals.py --tiny (f7_blocks.csv, f7_meta.json, F7_SUMMARY.md)
# They test the file formats only; none of their numbers is a result.
def _root(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    (root / "runs").mkdir(parents=True)
    (root / "results").mkdir()
    for f in ("PREREG_E2.md", "PAPER_EXPERIMENTS.md", "PREREG_E3.md"):
        if (ROOT / f).is_file():
            shutil.copy2(ROOT / f, root / f)
    return root


def _wj(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


M_ = "dinov2_vitb14_reg"
CFG_ = {"model": {"name": M_}, "structure": {"operators": "runs/x2/perlayer_reg_s0/results/e1_operators.npz"}}


def _lr_extra(extra_loss=2.3, flag=False, ref=False):
    lx = {"chosen_lr": 1e-4, "grid_lrs": [1e-5, 5e-5, 1e-4],
          "grid": [{"lr": 1e-5, "heldout_loss": 2.6}, {"lr": 5e-5, "heldout_loss": 2.2}, {"lr": 1e-4, "heldout_loss": 2.1}],
          "extra_lrs": [2e-4], "extra": [{"lr": 2e-4, "heldout_loss": extra_loss, "diverged": False}],
          "extra_would_win": flag, "budget_steps": 100, "seed": 0, "git_hash": "fixturehash"}
    if ref:
        lx["reference"] = {"run": "runs/m3_vexact_5090/k12", "same_best_lr": True, "max_rel_diff_heldout_loss": 1e-6}
    return lx


def _vexact_arm(d: Path, top1, seed=0, method="vexact", steps=780, lx=None, mlp=1000, attn=500):
    """A healing run in scripts/heal_vexact.py's format (synthetic numbers)."""
    knn = {"protocol": "v2", "amp": "bf16", "seed": seed, "git_hash": "fixturehash",
           "original": {"top1": 72.0, "flip_consistency": 96.0}, "zero_shot": {"top1": 50.0, "flip_consistency": 98.0},
           "healed": {"top1": top1, "flip_consistency": 99.7}}
    met = {"steps_done": steps, "total_steps": 780, "diverged": False, "seed": seed, "git_hash": "fixturehash",
           "heldout_initial": {"total": 3.0}, "heldout_final": {"total": 1.0}}
    summ = {"method": method, "k": 12, "depth": 12, "blocks": list(range(12)), "seed": seed, "git_hash": "fixturehash",
            "config": CFG_, "env": {"gpu": "fixture GPU"},
            "method_info": {"mlp_params_kept": mlp, "attn_params_kept": attn, "converted_params_kept": mlp + attn,
                            "converted_params_dense": 3000, "model_params": 2000, "model_params_original": 4000,
                            "model_params_frac": 0.5},
            "teacher": {"block_targets": "centred, symmetrised with A*", "final_target": "symmetrised"}}
    if method == "vexact_random":
        summ["random_structure"] = {"seed": 1, "agreement_with_fitted_A": 0.678}
    _wj(d / "knn.json", knn)
    _wj(d / "final" / "metrics.json", met)
    _wj(d / "summary.json", summ)
    _wj(d / "exactness.json", {"tol": 1e-4, "max_rel_inside": 9e-6, "pass": True, "seed": seed})
    _wj(d / "flipcheck.json", {"flip_consistency": 100.0, "pass": True, "n_query": 5000, "seed": seed})
    _wj(d / "lr_selection" / "lr_selection.json", {"best_lr": 1e-4, "seed": seed, "budget_steps": 100,
                                                   "trials": [{"lr": 1e-4, "heldout_loss": 2.1}]})
    (d / "SUMMARY.md").write_text("fixture\n", encoding="utf-8")
    if lx is not None:
        _wj(d / "lr_extra.json", lx)


def _svd_arm(d: Path, top1, steps=780, lx=None, final="raw"):
    """A healing run in scripts/heal.py's format (synthetic numbers). Its own MLP rank check reads full rank (float32
    product, as mirror/heal.py LowRankMLP.check records it); the attention check is float64 and equal."""
    _wj(d / "knn.json", {"protocol": "v2", "amp": "bf16", "seed": 0, "git_hash": "fixturehash",
                         "original": {"top1": 72.0, "flip_consistency": 96.0},
                         "zero_shot": {"top1": 30.0, "flip_consistency": 80.0},
                         "healed": {"top1": top1, "flip_consistency": 97.0}})
    _wj(d / "final" / "metrics.json", {
        "steps_done": steps, "total_steps": 780, "diverged": False, "seed": 0, "git_hash": "fixturehash",
        "heldout_initial": {"total": 3.0}, "heldout_final": {"total": 1.2},
        "constraint_checks": {"0:LowRankMLP": {"numerical_ranks": [[768, 768]], "target_ranks": [[310, 327]]},
                              "1:LowRankAttention": {"numerical_ranks": [[195, 195, 194, 206]],
                                                     "target_ranks": [[195, 195, 194, 206]]}}})
    _wj(d / "summary.json", {"method": "svd", "blocks": list(range(12)), "seed": 0, "git_hash": "fixturehash",
                             "config": CFG_, "env": {"gpu": "fixture GPU"},
                             "method_info": {"mlp_params_kept": 995, "attn_params_kept": 498, "mlp_params_dense": 2000,
                                             "attn_params_dense": 1000,
                                             "vexact_match": {"max_block_rel_gap_mlp": 0.002, "max_block_rel_gap_attn": 0.003}},
                             "targets": {"type": final, "final_target": final, "block_targets": "raw"}})
    _wj(d / "lr_selection" / "lr_selection.json", {"best_lr": 1e-4, "seed": 0, "budget_steps": 100,
                                                   "trials": [{"lr": 1e-4, "heldout_loss": 2.1}]})
    if lx is not None:
        _wj(d / "lr_extra.json", lx)


def _t2(folder: Path, rows: list, variants: dict) -> None:
    """t2_linear.csv rows and probe/<model>/<variant>_variant.json files in scripts/tier2_linear.py's format."""
    _write_csv(folder / "t2_linear.csv", rows)
    for var, info in variants.items():
        _wj(folder / "probe" / M_ / f"{var}_variant.json", {"info": info, "sources": {}})


def _t2row(var, fam, meth, k, top1):
    return {"model": M_, "variant": var, "label": var, "family": fam, "method": meth, "k": k, "top1": top1, "top5": 90.0,
            "knn_check_top1": top1 - 2, "chosen_wd": 0.001, "wd_at_grid_edge": False, "model_params": 2000 if k else 4000,
            "git_hash": "fixturehash", "seed": 0, "date": "fixture"}


RANKS64 = {"0:LowRankMLP": {"numerical_ranks": [[310, 327]], "target_ranks": [[310, 327]],
                            "note": "ranks of U V formed in float64 from the factors"},
           "1:LowRankAttention": {"numerical_ranks": [[195, 195, 194, 206]], "target_ranks": [[195, 195, 194, 206]]}}


def _e3_root(tmp_path, name, V0, V1, R, S3a, S3b, raw_steps=780, sym_extra=2.3, sym_flag=False) -> Path:
    """Every E3 arm at k = 12 with the given healed k-NN values (synthetic fixtures, not results)."""
    root = _root(tmp_path, name)
    r = root / "runs"
    _vexact_arm(r / "m3_vexact_5090" / "k12", V0)
    _wj(r / "e3" / "lr_probe_vexact_s0" / "k12" / "lr_extra.json", _lr_extra(ref=True))
    _vexact_arm(r / "e3" / "vexact_s1" / "k12", V1, seed=1, lx=_lr_extra())
    _vexact_arm(r / "e3" / "random" / "k12", R, method="vexact_random", lx=_lr_extra())
    _svd_arm(r / "e3" / "svd_raw" / "svd", S3a, steps=raw_steps, lx=_lr_extra())
    _svd_arm(r / "e3" / "svd_sym" / "svd", S3b, lx=_lr_extra(sym_extra, sym_flag), final="flip_average")
    _t2(r / "t2", [_t2row("original", "original", "original", 0, 75.0), _t2row("healed_vexact_k12", "healed_vexact", "vexact", 12, 74.0)],
        {"healed_vexact_k12": {"vexact_run": "/workspace/Mirror_CPAL/runs/m3_vexact_5090/k12"}})   # as the VM records it
    _t2(r / "e3" / "t2_vexact_s1", [_t2row("healed_vexact_k12", "healed_vexact", "vexact", 12, 73.9)],
        {"healed_vexact_k12": {"vexact_run": "runs/e3/vexact_s1/k12"}})
    _t2(r / "e3" / "t2_random", [_t2row("healed_vexact_k12", "healed_vexact", "vexact", 12, 70.0)],
        {"healed_vexact_k12": {"vexact_run": "runs/e3/random/k12"}})
    for arm, top in (("svd_raw", 71.0), ("svd_sym", 71.5)):
        _t2(r / "e3" / f"t2_{arm}", [_t2row("healed_svd_k12", "healed", "svd", 12, top)],
            {"healed_svd_k12": {"constraint_checks": RANKS64}})
        _wj(r / "e3" / f"t2_{arm}" / "probe" / M_ / "healed_svd_k12_variant.json",
            {"info": {"constraint_checks": RANKS64}, "sources": {"heal_run": f"runs/e3/{arm}/svd"}})
    return root


def test_e3_check_rule_unit():
    """PREREG_E3.md Sections 7 and 12: difference >= bar AND difference > s; a missed bar fails whatever s is."""
    assert CR.e3_check(3.334, 2.0, 0.218) == ("PASS", True, True)
    assert CR.e3_check(1.898, 3.0, 0.218) == ("FAIL", False, True)
    assert CR.e3_check(3.334, 2.0, 3.338) == ("FAIL", True, False)          # the seed condition binds
    assert CR.e3_check(3.334, 2.0, 3.334) == ("FAIL", True, False)          # equal to s is not larger than s
    assert CR.e3_check(3.334, 2.0, None) == ("pending", True, None)        # bar met, s unknown
    assert CR.e3_check(1.5, 2.0, None) == ("FAIL", False, None)            # bar missed: fails before s is known
    assert CR.e3_check(None, 2.0, 0.1) == ("pending", None, None)
    assert CR.e3_check(70.338 - 67.338, 3.0, 0.218)[0] == "PASS"           # exactly at the bar, float rounding absorbed
    assert CR.e3_check(70.336 - 67.338, 3.0, 0.218)[0] == "FAIL"           # one query (0.002 points) short
    assert CR.lr_txt(2e-4) == "2e-4" and CR.lr_txt(1e-5) == "1e-5"


def test_e3_g5_hand_computed(tmp_path):
    """Gate G5 on synthetic arms, values computed by hand:
    V0 70.338, V1 70.120 -> s = 0.218; R 67.004 -> E3.a = 3.334 >= 2.0 and > s: PASS;
    3a 67.900, 3b 68.440 -> S = 68.440 (3b) -> E3.b = 1.898 < 3.0: FAIL; G5 FAIL.
    With V1: E3.a 3.116 (would PASS), E3.b 1.680 (would FAIL); with the mean 70.229: 3.225, 1.789."""
    root = _e3_root(tmp_path, "s1", 70.338, 70.120, 67.004, 67.900, 68.440)
    out, res, idx, numbers = _run(root, tmp_path, "out_s1")
    C = res["tables"]["T_e3_checks"]
    a, b, s, g = (C.find(id=i) for i in ("E3.a", "E3.b", "E3.seed", "G5"))
    assert math.isclose(s["value"], 0.218, abs_tol=1e-9)
    assert math.isclose(a["value"], 3.334, abs_tol=1e-9) and a["verdict"] == "PASS" and a["meets_bar"] and a["exceeds_s"]
    assert math.isclose(b["value"], 1.898, abs_tol=1e-9) and b["verdict"] == "FAIL" and b["S_arm"] == "3b"
    assert math.isclose(a["max_baseline"], 68.338, abs_tol=1e-9) and math.isclose(b["max_baseline"], 67.338, abs_tol=1e-9)
    assert g["verdict"] == "FAIL" and "report as is, stop" in g["note"]
    for cid, v, verdict in (("E3.a (V1)", 3.116, "would PASS"), ("E3.b (V1)", 1.680, "would FAIL"),
                            ("E3.a (mean)", 3.225, "would PASS"), ("E3.b (mean)", 1.789, "would FAIL")):
        r = C.find(id=cid)
        assert math.isclose(r["value"], v, abs_tol=1e-9) and r["verdict"] == verdict, cid
    assert C.find(id="E3.lr")["verdict"] == "no"
    for arm in ("1", "2", "3a", "3b", "4"):
        assert C.find(id=f"E3.valid.{arm}")["verdict"] == "PASS", (arm, C.find(id=f"E3.valid.{arm}")["note"])
    # arms: linear probes from the right runs, parameter ratios, the float64 rank check of the Tier-2 rebuild
    T = res["tables"]["T_e3"]
    v0, raw, rnd = T.find(arm="1"), T.find(arm="3a"), T.find(arm="4")
    assert v0["healed_top1"] == 70.338 and v0["linear_top1"] == 74.0 and v0["linear_same_run"] is True
    assert v0["extra_would_win"] is False and v0["probe_same_best_lr"] is True
    assert raw["linear_top1"] == 71.0 and raw["rebuild_mlp_ranks_equal"] is True and raw["attn_ranks_equal"] is True
    assert raw["mlp_ranks_equal_recorded"] is False                     # the healing run's float32 check, not used
    assert math.isclose(raw["kept_vs_vexact"], (995 + 498) / 1500) and raw["converted_params_dense"] == 3000
    assert rnd["random_structure_seed"] == 1 and rnd["linear_top1"] == 70.0
    assert T.find(arm="1 (A100)")["status"] == "not run"
    # T_prereg, INDEX, numbers
    pre = pd.read_csv(out / "tables" / "T_prereg.csv").set_index("id")
    assert pre.loc["G5", "verdict"] == "FAIL" and pre.loc["G5", "prereg_file"] == "PREREG_E3.md"
    assert pre.loc["E3.a", "verdict"] == "PASS" and pre.loc["E3.b", "verdict"] == "FAIL"
    sec = _section(idx, "E3:")
    assert "G5: gate G5" in sec and "**FAIL**" in sec and "E3.a (V1) +3.116 (would PASS)" in sec
    assert numbers["e3.g5.verdict"]["value"] == "FAIL" and math.isclose(numbers["e3.a.value"]["value"], 3.334, abs_tol=1e-9)
    assert numbers["e3.v0.healed"]["locator"]["file"] == "runs/m3_vexact_5090/k12/knn.json"
    assert numbers["e3.v1.linear"]["locator"]["file"] == "runs/e3/t2_vexact_s1/t2_linear.csv"
    # T_tier2 takes the Tier-1 value of the healed V-exact probe from the run it rebuilt (runs/m3_vexact_5090)
    t2 = res["tables"]["T_tier2"].find(model=M_, variant="healed_vexact_k12")
    assert t2["tier1_knn_top1"] == 70.338 and t2["healing_run"] == "runs/m3_vexact_5090/k12"


def test_e3_g5_pass_at_the_bar_and_seed_condition(tmp_path):
    """S = 67.338 puts E3.b exactly at the bar (70.338 - 67.338 = 3.000): PASS, so G5 PASS. With V1 = 67.000 the
    seed difference s = 3.338 exceeds both differences: E3.a and E3.b FAIL on the seed condition, G5 FAIL."""
    root = _e3_root(tmp_path, "s2", 70.338, 70.120, 67.004, 67.100, 67.338)
    out, res, idx, numbers = _run(root, tmp_path, "out_s2")
    C = res["tables"]["T_e3_checks"]
    assert C.find(id="E3.b")["verdict"] == "PASS" and math.isclose(C.find(id="E3.b")["value"], 3.0, abs_tol=1e-9)
    assert C.find(id="G5")["verdict"] == "PASS" and "credited to the mirror structure" in C.find(id="G5")["note"]
    root = _e3_root(tmp_path, "s3", 70.338, 67.000, 67.004, 67.100, 67.338)
    out, res, idx, numbers = _run(root, tmp_path, "out_s3")
    C = res["tables"]["T_e3_checks"]
    a, b = C.find(id="E3.a"), C.find(id="E3.b")
    assert math.isclose(C.find(id="E3.seed")["value"], 3.338, abs_tol=1e-9)
    assert (a["verdict"], a["meets_bar"], a["exceeds_s"]) == ("FAIL", True, False)
    assert (b["verdict"], b["meets_bar"], b["exceeds_s"]) == ("FAIL", True, False)
    assert C.find(id="G5")["verdict"] == "FAIL"


def test_e3_validity_and_learning_rate_check(tmp_path):
    """Arm 3a stops at 779 of 780 steps: invalid (Section 7), so E3.b stays pending although its value passes; the 2e-4
    trial of arm 3b beats the grid (2.0 < 2.1) while the run's own flag says no: E3.lr flags it, the mismatch is noted."""
    root = _e3_root(tmp_path, "s4", 70.338, 70.120, 67.004, 67.100, 67.338, raw_steps=779, sym_extra=2.0, sym_flag=False)
    out, res, idx, numbers = _run(root, tmp_path, "out_s4")
    C = res["tables"]["T_e3_checks"]
    assert C.find(id="E3.valid.3a")["verdict"] == "FAIL" and "steps 779 of 780" in C.find(id="E3.valid.3a")["note"]
    b = C.find(id="E3.b")
    assert b["verdict"] == "pending" and "fail a validity condition" in b["note"] and "computed value PASS" in b["note"]
    assert C.find(id="E3.a")["verdict"] == "PASS" and C.find(id="G5")["verdict"] == "pending"
    lr = C.find(id="E3.lr")
    assert lr["verdict"] == "yes: 3b" and lr["value"] == 1 and "possibly under-tuned" in lr["note"]
    assert "differs from the run's own" in lr["note"]
    sym = res["tables"]["T_e3"].find(arm="3b")
    assert sym["extra_would_win"] is True and sym["extra_would_win_recorded"] is False and sym["grid_best_heldout_loss"] == 2.1
    dev = res["meta"]["deviations"]
    assert any("E3 arm 3a fails a validity condition" in d for d in dev)
    assert any("possibly under-tuned" in d for d in dev)


def test_e3_from_tiny_runs(tmp_path):
    """The real output formats of scripts/heal.py, heal_vexact.py (incl. --lr-probe-only) and tier2_linear.py (--tiny):
    every arm is read, the Tier-2 rebuild is matched to its run, the 2e-4 flag is recomputed, and no verdict is given
    on TINY inputs."""
    root = _root(tmp_path, "tiny")
    shutil.copytree(FIX / "e3_tiny" / "runs", root / "runs", dirs_exist_ok=True)
    out, res, idx, numbers = _run(root, tmp_path, "out_tiny")
    T, C = res["tables"]["T_e3"], res["tables"]["T_e3_checks"]
    assert res["ctx"]["e3_info"]["k"] == 4
    for arm, rel in (("1", "m3_vexact_5090/k4"), ("2", "e3/vexact_s1/k4"), ("3a", "e3/svd_raw/svd"), ("3b", "e3/svd_sym/svd"),
                     ("4", "e3/random/k4")):
        r = T.find(arm=arm)
        knn = json.loads((root / "runs" / rel / "knn.json").read_text(encoding="utf-8"))
        assert r["healed_top1"] == knn["healed"]["top1"] and "TINY" in r["status"], arm
        lin = r.get("linear_top1")
        assert lin is not None and r["linear_same_run"] is True, arm
        lx = root / "runs" / (rel if arm != "1" else "e3/lr_probe_vexact_s0/k4") / "lr_extra.json"
        rec = json.loads(lx.read_text(encoding="utf-8"))
        assert r["extra_would_win"] == rec["extra_would_win"] and r["extra_win_matches_recorded"] is True, arm
    assert T.find(arm="3a")["rebuild_mlp_ranks_equal"] is True and T.find(arm="3a")["mlp_ranks_equal_recorded"] is False
    assert T.find(arm="4")["random_structure_seed"] == 1
    assert T.find(arm="1")["probe_same_best_lr"] is True
    for cid in ("E3.a", "E3.b", "G5"):
        assert C.find(id=cid)["verdict"] == "pending", cid
    assert "TINY smoke-test inputs" in C.find(id="E3.a")["note"]
    assert any("TINY smoke-test run" in d and "E3 arm" in d for d in res["meta"]["deviations"])
    assert "- [x] `runs/e3/svd_raw/svd/lr_extra.json`" in idx


def test_e3_absent(tmp_path):
    root = _root(tmp_path, "none")
    out, res, idx, numbers = _run(root, tmp_path, "out_none")
    T, C = res["tables"]["T_e3"], res["tables"]["T_e3_checks"]
    assert {r["status"] for r in T.rows} == {"not run"} and len(T.rows) == len(CR.E3_ARMS)
    assert {C.find(id=i)["verdict"] for i in ("E3.a", "E3.b", "E3.seed", "G5", "E3.lr")} == {"pending"}
    sec = _section(idx, "E3:")
    if (ROOT / "PREREG_E3.md").is_file():
        assert "Arm 3a" in sec and "not run (no files under `runs/e3/svd_raw/svd/`" in sec
        assert "E3.a V0 - R >= 2.0" in sec and "E3.b V0 - S >= 3.0" in sec
    assert not any(k.startswith("e3.") and k.endswith(".healed") for k in numbers)
    assert "- [ ] `runs/e3/random/k12/knn.json`" in idx


def _sb_row(k, m, top1, kept, budget, ratio, within=True):
    info = {"baseline": m, "mirror_budget": budget, "within_1pct": within, "bias_params": 10, "allocation": [[1, 1]]}
    if m == "slicegpt":
        info["basis_change_params"] = k * 768 * 768
    return {"model": M_, "ablation": "SB", "variant": m, "k": k, "fit_seed": 0, "top1": top1, "flip_consistency": 90.0,
            "top1_original": 79.0, "flip_original": 96.0, "drop_points": 79.0 - top1, "mlp_params_kept": kept,
            "mlp_params_dense": 2 * 3072 * 768 * k, "params_vs_mirror_first_k": ratio, "n_query": 50000, "n_fit_images": 2000,
            "seed": 0, "git_hash": "fixturehash", "git_dirty": False, "info": json.dumps(info)}


def test_strong_baselines_against_canonical_x2(tmp_path):
    """build_strong_baselines with a hand-built T_main_zeroshot: the canonical X2 Mirror and SVD rows of the same k and fit
    seed sit next to each strong baseline, with the margins and budget ratios. Synthetic numbers."""
    root = _root(tmp_path, "sb")
    d, k, budget = 768, 6, 30_000_000
    _write_csv(root / "runs" / "sb" / "SB_s0.csv", [
        _sb_row(6, "asvd", 70.0, 29_900_000, budget, 29_900_000 / budget),
        _sb_row(6, "slicegpt", 50.0, 29_950_000, budget, 29_950_000 / budget),
        _sb_row(6, "prune_act", 60.0, 29_000_000, budget, 29_000_000 / budget, within=False),
        _sb_row(9, "asvd", 65.0, 44_000_000, 44_100_000, 44_000_000 / 44_100_000)])
    Tm = CR.Table("T_main_zeroshot", "t", ["model", "k", "variant"], [])
    src = "runs/x2/perlayer_reg_s0/results/e1_surgery.csv"

    def rec(top1, var, cost):
        return {"top1": top1, "flip": 91.0, "drop": 79.0 - top1, "src": src, "where": {"k": k, "variant": var},
                "git": "x2hash", "seed": 0, "fit_seed": 0, "cost": cost}

    Tm.add({"model": M_, "k": 0, "variant": "original", "status": "ok", "experiment": "X2",
            "_per_seed": {0: {"top1": 79.0, "src": src, "where": {"k": 0, "variant": "original"}}}})
    Tm.add({"model": M_, "k": 6, "variant": "mirror_mlp", "status": "ok", "experiment": "X2",
            "_per_seed": {0: rec(77.0, "surgery", {"matrix_params_kept": budget + 6 * 4 * d * d, "rotation_params": 6 * d * d})}})
    Tm.add({"model": M_, "k": 6, "variant": "svd_mlp", "status": "ok", "experiment": "X2",
            "_per_seed": {0: rec(9.0, "svd", {"matrix_params_kept": 29_800_000 + 6 * 4 * d * d, "rotation_params": 0})}})
    T, files = CR.build_strong_baselines(CR.Paths(root), Tm, {M_: {"d": d}})
    mir, svd = T.find(model=M_, k=6, fit_seed=0, method="mirror"), T.find(model=M_, k=6, fit_seed=0, method="svd")
    assert mir["top1"] == 77.0 and mir["mlp_params_kept"] == budget and mir["ratio_to_mirror"] == 1.0
    assert mir["basis_params_not_counted"] == 6 * d * d and mir["mlp_params_kept_cost_model"] == budget
    assert svd["mlp_params_kept"] == 29_800_000 and math.isclose(svd["ratio_to_mirror"], 29_800_000 / budget)
    asvd = T.find(model=M_, k=6, fit_seed=0, method="asvd")
    assert math.isclose(asvd["mirror_minus_this"], 7.0) and math.isclose(asvd["this_minus_svd"], 61.0)
    assert asvd["_loc"]["top1"]["file"] == "runs/sb/SB_s0.csv" and asvd["drop_points"] == 9.0
    sl = T.find(model=M_, k=6, fit_seed=0, method="slicegpt")
    assert sl["basis_params_not_counted"] == 6 * d * d and sl["within_1pct"] is True
    assert T.find(model=M_, k=6, fit_seed=0, method="prune_act")["within_1pct"] is False
    assert T.find(model=M_, k=9, fit_seed=0, method="mirror")["status"] == "not run"
    assert T.find(model=M_, k=9, fit_seed=0, method="asvd")["top1"] == 65.0
    assert [r["method"] for r in T.rows if r["k"] == 6] == ["mirror", "svd", "asvd", "slicegpt", "prune_act"]


def test_strong_baselines_tiny_format(tmp_path):
    root = _root(tmp_path, "sbtiny")
    shutil.copytree(FIX / "sb", root / "runs" / "sb")
    out, res, idx, numbers = _run(root, tmp_path, "out_sbtiny")
    T = res["tables"]["T_strong_baselines"]
    src = pd.read_csv(FIX / "sb" / "SB_s0.csv")
    sb = [r for r in T.rows if r["experiment"] == "SB"]
    assert len(sb) == len(src)
    for r in sb:
        s = src[(src["variant"] == r["method"]) & (src["k"] == r["k"])].iloc[0]
        assert r["top1"] == float(s["top1"]) and r["ratio_to_mirror"] == float(s["params_vs_mirror_first_k"])
        assert r["mirror_budget"] == json.loads(s["info"])["mirror_budget"]
    assert {r["status"] for r in T.rows if r["experiment"] != "SB"} == {"not run"}    # no X2 rows for the tiny model
    assert "SB:" in idx and "Activation-aware SVD" in _section(idx, "SB:")
    off = [r for r in sb if not r["within_1pct"]]
    assert off and any("Strong baselines:" in d and "within 1%" in d for d in res["meta"]["deviations"])


def test_flipavg_tiny_format(tmp_path):
    root = _root(tmp_path, "fa")
    shutil.copytree(FIX / "t1_flipavg", root / "runs" / "t1_flipavg")
    out, res, idx, numbers = _run(root, tmp_path, "out_fa")
    T = res["tables"]["T_flipavg"]
    src = pd.read_csv(FIX / "t1_flipavg" / "flipavg.csv")
    assert len(T.rows) == len(src)
    o, fa = T.find(model="tiny", variant="original", mode="plain"), T.find(model="tiny", variant="original", mode="flipavg")
    vx = T.find(model="tiny", variant="healed_vexact_k4", mode="plain")
    assert o["rel_cost"] == 1.0 and math.isclose(fa["rel_cost"], fa["gflops"] / o["gflops"]) and fa["forwards_per_image"] == 2
    assert fa["exactly_invariant"] and vx["exactly_invariant"] and not o["exactly_invariant"]
    assert math.isclose(vx["minus_flipavg_knn"], vx["knn_top1"] - fa["knn_top1"])
    assert math.isclose(vx["minus_flipavg_probe"], vx["probe_top1"] - fa["probe_top1"])
    assert math.isclose(vx["cost_vs_flipavg"], vx["gflops"] / fa["gflops"])
    assert math.isclose(vx["params_vs_flipavg"], vx["params"] / fa["params"])
    assert vx["label"] == CR.METHOD_LABEL[("healed_vexact", "vexact")]
    sec = _section(idx, "Flip averaging:")
    assert "against the flip-averaged original" in sec and "TINY" in sec
    assert any(d.startswith("Flip averaging:") for d in res["meta"]["deviations"])


def test_latency_second_device(tmp_path):
    if not all((ROOT / "runs" / d / "m2_latency.csv").is_file() for d in ("m2_clean", "m2_5090")):
        pytest.skip("needs runs/m2_clean and runs/m2_5090")
    root = _root(tmp_path, "m2b")
    for d in ("m2_clean", "m2_5090"):
        shutil.copytree(ROOT / "runs" / d, root / "runs" / d)
    out, res, idx, numbers = _run(root, tmp_path, "out_m2b")
    L1, L2, D = (res["tables"][n] for n in ("T_latency", "T_latency_5090", "T_latency_devices"))
    lat2 = pd.read_csv(ROOT / "runs" / "m2_5090" / "m2_latency.csv")
    assert len(L2.rows) == len(lat2) and len(res["tables"]["T_latency_checks_5090"].rows) > 0
    meta2 = json.loads((ROOT / "runs" / "m2_5090" / "m2_meta.json").read_text(encoding="utf-8"))
    assert any(meta2["env"]["gpu"] in n for n in L2.notes)
    for r in D.rows:
        a = L1.find(model=r["model"], variant=r["variant"], k=r["k"], batch=r["batch"], mode=r["mode"])
        b = L2.find(model=r["model"], variant=r["variant"], k=r["k"], batch=r["batch"], mode=r["mode"])
        assert r["speedup_1"] == a["speedup_vs_dense"] and r["speedup_2"] == b["speedup_vs_dense"]
        assert math.isclose(r["speedup_diff"], b["speedup_vs_dense"] - a["speedup_vs_dense"])
    assert len(D.rows) == len(lat2)
    sec = _section(idx, "M2:")
    assert "Second device: `runs/m2_5090/`" in sec and "measured speed-up against dense" in sec
    if str(meta2.get("check_images", "")).startswith("random"):
        assert "fell back to random images" in sec
        assert any("runs/m2_5090/" in d and "random normal images" in d for d in res["meta"]["deviations"])


def test_tier2_seg_rerun_and_healed_rows(tmp_path):
    """ADE20K: healed k = 9 and V-exact k = 12 rows with margins, and a rerun of the original giving the run-to-run
    spread (mIoU of the rerun: synthetic, the first run's plus 0.07)."""
    root = _root(tmp_path, "seg")
    shutil.copytree(FIX / "t2_seg", root / "runs" / "t2_seg")
    f = root / "runs" / "t2_seg" / "t2_ade20k.csv"
    df = pd.read_csv(f)
    o = df[df["variant"] == "original"].iloc[0]
    extra = []
    for var, fam, meth, k, miou in (("healed_mirror_k9", "healed", "mirror", 9, 40.0), ("healed_svd_k9", "healed", "svd", 9, 38.5),
                                    ("healed_random_k9", "healed", "random", 9, 30.25),
                                    ("healed_vexact_k12", "healed_vexact", "vexact", 12, 39.0)):
        r = o.copy()
        r["variant"], r["family"], r["method"], r["k"], r["mIoU"] = var, fam, meth, k, miou
        extra.append(r)
    pd.concat([df, pd.DataFrame(extra)], ignore_index=True).to_csv(f, index=False)
    rr = o.copy()
    rr["mIoU"] = o["mIoU"] + 0.07
    _write_csv(root / "runs" / "t2_seg_rerun" / "t2_ade20k.csv", [rr.to_dict()])
    out, res, idx, numbers = _run(root, tmp_path, "out_seg")
    T, R = res["tables"]["T_tier2_seg"], res["tables"]["T_tier2_seg_rerun"]
    hm = T.find(model=o["model"], variant="healed_mirror_k9")
    assert math.isclose(hm["margin_svd_miou"], 1.5) and math.isclose(hm["margin_random_miou"], 9.75)
    assert T.find(model=o["model"], variant="healed_vexact_k12")["mIoU"] == 39.0
    assert len(R.rows) == 1 and math.isclose(R.rows[0]["diff_mIoU"], 0.07, abs_tol=1e-9)
    first = T.find(model=o["model"], variant="original")
    assert math.isclose(first["rerun_minus_first_mIoU"], 0.07, abs_tol=1e-9) and first["rerun_mIoU"] == float(rr["mIoU"])
    assert any(n.startswith("Run-to-run spread") for n in T.notes)
    sec = _section(idx, "Tier 2:")
    assert "ADE20K rerun (run-to-run spread)" in sec and "healed_mirror_k9" in sec
    assert numbers[f"t2segrerun.{CR.MODEL_SHORT.get(o['model'], o['model'])}.original.diff"]["value"] == R.rows[0]["diff_mIoU"]


def test_f7_tiny_format(tmp_path):
    root = _root(tmp_path, "f7")
    shutil.copytree(FIX / "f7", root / "runs" / "f7")
    (root / "runs" / "f7" / "figures").mkdir()
    (root / "runs" / "f7" / "figures" / "fig0_teaser.pdf").write_bytes(b"%PDF-fixture")
    out, res, idx, numbers = _run(root, tmp_path, "out_f7")
    B, L0 = res["tables"]["T_f7"], res["tables"]["T_f7_layer0"]
    src = pd.read_csv(FIX / "f7" / "f7_blocks.csv")
    assert len(B.rows) == len(src)
    for _, s in src.iterrows():
        r = B.find(model=s["model"], block=int(s["block"]))
        assert r["twin_r_median"] == s["pairs_twin_corr_median"] and r["rand_r_median"] == s["pairs_rand_corr_median"]
    meta = json.loads((FIX / "f7" / "f7_meta.json").read_text(encoding="utf-8"))
    assert len(L0.rows) == 4
    assert L0.find(operator="A_0 (fitted)", parity="even")["sym_fraction"] == meta["layer0"]["fitted_even_sym_fraction"]
    sec = _section(idx, "F7:")
    assert "TINY smoke test" in sec and "fig0_teaser.pdf" in sec and "Block 1: twin correlation median" in sec


def test_make_figures_e3_and_teaser(tmp_path):
    """scripts/make_figures.py draws the E3 figure from the collector's tables and lists the F7 teaser files."""
    root = _e3_root(tmp_path, "fig", 70.338, 70.120, 67.004, 67.900, 68.440)
    out, res, idx, numbers = _run(root, tmp_path, "out_fig")
    spec = importlib.util.spec_from_file_location("make_figures_t", ROOT / "scripts" / "make_figures.py")
    mf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mf)
    figs = tmp_path / "figs"
    figs.mkdir()
    (figs / "fig0_teaser.pdf").write_bytes(b"%PDF-fixture")
    mf.main(["--tables", str(out / "tables"), "--out", str(figs)])
    assert (figs / "fig8_e3.pdf").is_file() and (figs / "fig8_e3.png").is_file()
    md = (figs / "FIGURES.md").read_text(encoding="utf-8")
    assert "Figure 8. E3" in md and "E3.b (V0 minus the better SVD arm): +1.90, FAIL" in md and "gate G5: FAIL" in md
    assert "Figure 0 (teaser, F7)" in md and "fig0_teaser.pdf" in md
    assert "—" not in md
