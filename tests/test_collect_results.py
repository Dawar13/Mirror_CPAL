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
               "T_e1_history", "T_attention", "T_ablations", "T_tier2", "T_latency", "T_latency_profile", "T_f4"]
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
