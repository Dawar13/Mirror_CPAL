"""scripts/collect_next_phase.py and scripts/make_figures_next.py on the tiny fixture tree tests/fixtures/next_phase/
(synthetic values, files shaped like the real outputs of measure_groups.py, must05_twins.py, must06_visibility.py,
must07_compressors.py, must09_heal.py, must10_flipavg.py, train_controlled.py and h100_gate_c.py). The fixture is the
all-PASS case; each test copies it and edits one input to get FAIL or PENDING. CPU only."""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures" / "next_phase"


def _load(name: str):
    key = f"_test_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, REPO / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


CNP = _load("collect_next_phase")
FIG = _load("make_figures_next")


def tree(tmp_path: Path, name: str = "tree") -> Path:
    root = tmp_path / name
    shutil.copytree(FIX, root)
    return root


def collect(root: Path):
    payload = CNP.main(["--root", str(root)])
    return {p["id"]: p for p in payload["predictions"]}, {g["gate"]: g for g in payload["gates"]}, payload


def jedit(path: Path, fn) -> None:
    obj = json.loads(path.read_text(encoding="utf-8"))
    fn(obj)
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")


def write_teaser(root: Path) -> None:
    rng = np.random.default_rng(0)
    arr = {"blocks": np.array([1, 3]), "crop": np.array(42), "patch": np.array(14), "grid_w": np.array(16),
           "vis_rows": np.arange(10)}
    for b in (1, 3):
        arr[f"b{b}_pairs"] = np.array([[0, 1], [2, 3]])
        arr[f"b{b}_fit_S"] = np.array([0.9, 0.8])
        arr[f"b{b}_crops"] = rng.integers(0, 255, size=(2, 2, 4, 42, 42, 3), dtype=np.uint8)
        arr[f"b{b}_valid"] = np.ones((2, 2, 4), dtype=bool)
    np.savez_compressed(root / "results" / "MUST-05" / "dinov2_vitb14_reg" / "teaser.npz", **arr)


# ------------------------------------------------------------------------------------------------ helpers
def test_exact_invariance_rule():
    assert CNP.exact_invariance(100.0, 0, None)[0] == CNP.PASS
    assert CNP.exact_invariance(99.99, 5, {"max_margin_rel": 2e-5})[0] == CNP.PASS
    assert CNP.exact_invariance(99.99, 5, {})[0] == CNP.PENDING
    assert CNP.exact_invariance(99.99, 5, {"margins_rel": [1e-5, 2e-4]})[0] == CNP.FAIL
    assert CNP.exact_invariance(None, None, None)[0] == CNP.PENDING
    assert CNP.combine([CNP.PASS, CNP.PENDING]) == CNP.PENDING
    assert CNP.combine([CNP.PASS, CNP.PENDING, CNP.FAIL]) == CNP.FAIL
    assert CNP.combine([CNP.PASS, CNP.PASS]) == CNP.PASS


# ------------------------------------------------------------------------------------------------ full fixture
def test_fixture_all_pass_and_outputs(tmp_path):
    root = tree(tmp_path)
    P, G, payload = collect(root)
    for pid in ("MUST-01", "MUST-03", "MUST-04 P4.1", "MUST-04 P4.2", "MUST-04 P4.3", "MUST-05 P5.1", "MUST-05 P5.2",
                "MUST-06 P6.1", "MUST-06 P6.2", "MUST-06 P6.3", "MUST-07 P7.1", "MUST-07 P7.2", "MUST-09 P9.1",
                "MUST-09 P9.2", "MUST-09 P9.3", "MUST-10 P10.1", "MUST-12 gate C", "CORE-01 T1", "CORE-01 T2",
                "CORE-01 T3", "CORE-01 T4", "CORE-02 P2.1", "CORE-02 P2.2", "CORE-02 P2.3", "CORE-03 P3.1",
                "CORE-05 P5.1", "CORE-05 P5.2"):
        assert P[pid]["verdict"] == "PASS", (pid, P[pid]["reason"])
    for pid in ("MUST-02", "MUST-08 P8.1", "MUST-11 P11.1", "CORE-05 P5.3", "CORE-06 P6.1", "CORE-07 P7.1"):
        assert P[pid]["verdict"] == "PENDING" and "missing" in P[pid]["reason"]
    assert all(P[f"UP-0{i}"]["verdict"] in ("PENDING", "REPORTED") for i in range(1, 8))
    assert {g: G[g]["status"] for g in "ABCDE"} == {"A": "MET", "B": "MET", "C": "MET", "D": "MET", "E": "MET"}
    # sources carry the run's commit (from meta.json) and every prediction the prereg commit
    assert P["MUST-04 P4.1"]["sources"][0]["git_hash"].startswith("f1x7ure")
    assert all("prereg_commit" in p for p in payload["predictions"])
    assert payload["prereg"]["file"] == "PREREG_NEXT_PHASE.md"
    # earlier-phase failures: F4 and the weights-only pairing
    keys = [f["key"] for f in payload["failed_checks"]["earlier_phase"]]
    assert keys[:2] == ["p2", "p5_part2"]
    md = (root / "FINAL_RESULTS.md").read_text(encoding="utf-8")
    assert "\u2014" not in md
    for needle in ("## Gates", "## Pre-registered predictions", "## Numbers the paper will cite",
                   "## Failed checks (reported as failures)", "F4 energy rule", "MUST-07 P7.1", "CORE-01 T1"):
        assert needle in md
    js = json.loads((root / "PAPER_NUMBERS.json").read_text(encoding="utf-8"))
    assert js["numbers"]["must09.mirror_minus_best"]["value"] == pytest.approx(78.30 - 78.40)
    assert js["numbers"]["must04.dinov2_vitb14_reg.UD.surgery_drop"]["sources"][0]["file"] == \
        "results/MUST-04/dinov2_vitb14_reg/metrics.json"
    data = root / "results" / "paper_next" / "data"
    for name in ("fig2_blocks", "fig3_matrix", "fig3_emergence", "fig4a_pairs", "fig4b_stacking", "fig4c_bands",
                 "fig5_points", "must07_rows", "table1_healed", "table4_generality", "table4_core02"):
        assert (data / f"{name}.csv").is_file(), name


def test_prereg_quotes_match_signed_off_text(tmp_path):
    P, _, payload = collect(tree(tmp_path))
    if not payload["prereg"].get("sha256"):
        pytest.skip("PREREG_NEXT_PHASE.md not available")
    missing = [pid for pid, p in P.items() if p["quote"] and p["quote_found"] is not True]
    assert not missing, missing


# ------------------------------------------------------------------------------------------------ MUST-04 P4.1
def test_must04_p41_all_four_models_rule(tmp_path):
    # one model below the bar: FAIL for the claim, whatever the others show
    root = tree(tmp_path, "fail")
    jedit(root / "results/MUST-04/deit3_b16/metrics.json",
          lambda m: m["scores"]["UD"].__setitem__("ratio_to_null_1-6", 0.85))
    P, _, _ = collect(root)
    assert P["MUST-04 P4.1"]["verdict"] == "FAIL"
    assert P["MUST-04 P4.2"]["verdict"] == "PASS"
    per = {d["model"]: d.get("P4.1") for d in P["MUST-04 P4.1"]["details"]}
    assert per["deit3_b16"] == "FAIL" and per["dinov2_vitb14_reg"] == "PASS"
    # one model not measured: PENDING (the claim needs all four)
    root = tree(tmp_path, "pending")
    shutil.rmtree(root / "results/MUST-04/dinov2_vitb14")
    P, G, _ = collect(root)
    assert P["MUST-04 P4.1"]["verdict"] == "PENDING"
    assert "dinov2_vitb14" in P["MUST-04 P4.1"]["reason"]
    assert P["MUST-04 P4.3"]["verdict"] == "PENDING"
    assert G["E"]["status"] == "PENDING"
    # one missing and one failing: FAIL
    jedit(root / "results/MUST-04/deit3_b16/metrics.json",
          lambda m: m["scores"]["R90"].__setitem__("ratio_to_null_1-6", 0.5))
    P, _, _ = collect(root)
    assert P["MUST-04 P4.1"]["verdict"] == "FAIL"


def test_must04_p42_twin_null_interval(tmp_path):
    root = tree(tmp_path)
    jedit(root / "results/MUST-04/dinov2_vitl14_reg/metrics.json",
          lambda m: m["scores"]["UD"]["twin_null_twin_fraction_t50_1-6"].__setitem__("observed", 0.2))
    P, _, _ = collect(root)
    assert P["MUST-04 P4.2"]["verdict"] == "FAIL"


# ------------------------------------------------------------------------------------------------ MUST-07 P7.1
def _rows(root):
    return root / "results/MUST-07/rows_dinov2_vitb14_reg.csv"


def test_must07_p71_seeds_bootstrap_and_both_sizes(tmp_path):
    P, _, _ = collect(tree(tmp_path, "base"))
    det = P["MUST-07 P7.1"]["details"]
    assert len(det) == 20 and {d["size"] for d in det} == {"6mlp", "6both"}       # 10 baselines x 2 sizes
    assert all(d["P7.1"] == "PASS" for d in det)
    # a missing fit seed of one method at one 6-block size: PENDING
    root = tree(tmp_path, "seed")
    df = pd.read_csv(_rows(root))
    df = df[~((df["method"] == "flap") & (df["size"] == "6both") & (df["fit_seed"] == 2))]
    df.to_csv(_rows(root), index=False)
    P, _, _ = collect(root)
    assert P["MUST-07 P7.1"]["verdict"] == "PENDING"
    assert "flap" in P["MUST-07 P7.1"]["reason"]
    # the three-seed mean margin below 2 points at the second 6-block size: FAIL
    root = tree(tmp_path, "margin")
    df = pd.read_csv(_rows(root))
    me = df[(df["method"] == "mirror_exact") & (df["size"] == "6both")].set_index("fit_seed")["val_top1"]
    sel = (df["method"] == "slicegpt") & (df["size"] == "6both")
    df.loc[sel, "val_top1"] = df.loc[sel, "fit_seed"].map(me) - 1.5
    df.to_csv(_rows(root), index=False)
    P, _, _ = collect(root)
    assert P["MUST-07 P7.1"]["verdict"] == "FAIL"
    row = next(d for d in P["MUST-07 P7.1"]["details"] if d["method"] == "slicegpt" and d["size"] == "6both")
    assert row["diff_mean"] == pytest.approx(1.5) and row["P7.1"] == "FAIL"
    # bootstrap interval (fit seed 0) including zero: FAIL although the mean margin is large
    root = tree(tmp_path, "ci")
    b = pd.read_csv(root / "results/MUST-07/bootstrap.csv")
    b.loc[(b["method"] == "svdllm") & (b["size"] == "6mlp"), "ci_low"] = -0.1
    b.to_csv(root / "results/MUST-07/bootstrap.csv", index=False)
    P, _, _ = collect(root)
    assert P["MUST-07 P7.1"]["verdict"] == "FAIL"
    assert P["MUST-07 P7.2"]["verdict"] == "PASS"


# ------------------------------------------------------------------------------------------------ MUST-09 P9.2
def _heal(root, method, seed):
    return root / f"runs/MUST-09/dinov2_vitb14_reg/{method}/s{seed}/metrics.json"


def test_must09_p92_rounding_allowance(tmp_path):
    P, _, _ = collect(tree(tmp_path, "base"))
    assert P["MUST-09 P9.2"]["verdict"] == "PASS"
    assert "near-ties" in P["MUST-09 P9.2"]["reason"]
    # the one disagreement is not a near-tie: FAIL
    root = tree(tmp_path, "wide")
    jedit(_heal(root, "mirror_exact", 1), lambda m: m["knn_fp32"].__setitem__("max_margin_rel", 5e-4))
    P, _, _ = collect(root)
    assert P["MUST-09 P9.2"]["verdict"] == "FAIL"
    # margins not recorded: the allowance cannot be applied, PENDING
    root = tree(tmp_path, "nomargin")
    jedit(_heal(root, "mirror_exact", 1), lambda m: m["knn_fp32"].pop("max_margin_rel"))
    P, _, _ = collect(root)
    assert P["MUST-09 P9.2"]["verdict"] == "PENDING"
    assert "not recorded" in P["MUST-09 P9.2"]["reason"]
    # a compressor at a raw 100%: FAIL ("no compressor does")
    root = tree(tmp_path, "svd100")
    jedit(_heal(root, "svd", 0), lambda m: m["knn_fp32"].__setitem__("flip_consistency", 100.0))
    P, _, _ = collect(root)
    assert P["MUST-09 P9.2"]["verdict"] == "FAIL"


def test_must09_gate_d_and_p91(tmp_path):
    root = tree(tmp_path)
    for s in (0, 1, 2):
        jedit(_heal(root, "svdllm", s), lambda m: m["knn_bf16"].__setitem__("knn_top1", m["knn_bf16"]["knn_top1"] + 1.0))
    P, G, _ = collect(root)
    assert P["MUST-09 P9.1"]["verdict"] == "FAIL"           # -1.1 points
    assert G["D"]["status"] == "NOT MET"


# ------------------------------------------------------------------------------------------------ CORE-01 T1
def _final(root, cond, seed, ep=100):
    return root / f"results/CORE-01/vit_s8_{cond}_s{seed}/e{ep:03d}/metrics.json"


def test_core01_t1_every_seed_pair(tmp_path):
    P, _, _ = collect(tree(tmp_path, "base"))
    det = P["CORE-01 T1"]["details"]
    assert [d["seed_pair"] for d in det] == [0, 1, 2]
    assert all(d["difference"] == pytest.approx(0.42) for d in det)
    # one seed pair below 0.2: FAIL
    root = tree(tmp_path, "pair2")
    jedit(_final(root, "UD", 2), lambda m: m["scores"]["UD"].__setitem__("twin_fraction_t50_1-6", 0.2))
    P, _, _ = collect(root)
    assert P["CORE-01 T1"]["verdict"] == "FAIL"
    assert [d["verdict"] for d in P["CORE-01 T1"]["details"]] == ["PASS", "PASS", "FAIL"]
    # the pair is matched by seed: lowering the LR run of seed 1 changes pair 1 only
    root = tree(tmp_path, "lrseed")
    jedit(_final(root, "LR", 1), lambda m: m["scores"]["UD"].__setitem__("twin_fraction_t50_1-6", 0.3))
    P, _, _ = collect(root)
    assert [d["verdict"] for d in P["CORE-01 T1"]["details"]] == ["PASS", "FAIL", "PASS"]
    # the final checkpoint of one run not measured yet (only epoch 5): PENDING
    root = tree(tmp_path, "nofinal")
    shutil.rmtree(_final(root, "UD", 1).parent)
    P, _, _ = collect(root)
    assert P["CORE-01 T1"]["verdict"] == "PENDING"
    assert P["CORE-01 T1"]["details"][1]["difference"] is None


# ------------------------------------------------------------------------------------------------ gate C
def test_gate_c(tmp_path):
    root = tree(tmp_path, "fail")
    jedit(root / "results/MUST-12/vit_s8_LR_s0/e100/metrics.json",
          lambda m: m["scores"]["LR"].__setitem__("ratio_to_null_1-6", 0.85))
    P, G, _ = collect(root)
    assert P["MUST-12 gate C"]["verdict"] == "FAIL" and G["C"]["status"] == "NOT MET"
    assert "differs" in G["C"]["evidence"]                   # gate_c.json still records the old value
    root = tree(tmp_path, "missing")
    shutil.rmtree(root / "results/MUST-12")
    P, G, _ = collect(root)
    assert P["MUST-12 gate C"]["verdict"] == "PENDING" and G["C"]["status"] == "PENDING"


# ------------------------------------------------------------------------------------------------ CORE-02
def test_core02_trained_wrong_and_subgroup(tmp_path):
    root = tree(tmp_path, "wrong")
    path = root / "results/CORE-02/surgery.csv"
    df = pd.read_csv(path)
    df.loc[df["relation"] == "subgroup", "drop_points"] = 12.0           # subgroups are reported, not scored
    df.to_csv(path, index=False)
    P, _, _ = collect(root)
    assert P["CORE-02 P2.3"]["verdict"] == "PASS" and P["CORE-02 P2.2"]["verdict"] == "PASS"
    df.loc[(df["run"] == "vit_s8_R90_s0") & (df["group"] == "UD"), "drop_points"] = 15.0
    df.to_csv(path, index=False)
    P, _, _ = collect(root)
    assert P["CORE-02 P2.3"]["verdict"] == "FAIL"
    law = root / "results/CORE-02/law_per_block.csv"
    lw = pd.read_csv(law)
    lw.loc[0, "free_fc1_achieved"] = lw.loc[0, "free_fc1_predicted"] + 1
    lw.to_csv(law, index=False)
    P, _, _ = collect(root)
    assert P["CORE-02 P2.1"]["verdict"] == "FAIL"


# ------------------------------------------------------------------------------------------------ figures
def test_figures_from_fixtures(tmp_path):
    root = tree(tmp_path)
    write_teaser(root)
    collect(root)
    entries = FIG.main(["--root", str(root)])
    made = {e["name"] for e in entries if e["status"] == "made"}
    figs = root / "results/paper_next/figures"
    for name in ("fig1_twins", "fig2_tax_depth", "fig3_controlled", "fig4_visibility_stacking", "fig5_accuracy_params"):
        assert name in made, name
        assert (figs / f"{name}.png").stat().st_size > 0 and (figs / f"{name}.pdf").stat().st_size > 0
    tabs = root / "results/paper_next/tables"
    for name in ("table1_healed", "table3a_compressors", "table4_generality", "appendix_a2_law", "appendix_a3_prereg",
                 "appendix_a4_failed"):
        for ext in ("md", "csv", "tex"):
            assert (tabs / f"{name}.{ext}").is_file(), (name, ext)
    assert "\\toprule" in (tabs / "table1_healed.tex").read_text(encoding="utf-8")
    index = (root / "results/paper_next/FIGURES.md").read_text(encoding="utf-8")
    assert "table2_downstream**: skipped" in index                      # no MUST-11 or CORE-07 in the fixture
    assert "table4b_refund_new_groups" in made


def test_figures_skipped_without_inputs(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    P, _, _ = collect(root)
    assert all(p["verdict"] == "PENDING" for p in P.values())
    entries = FIG.main(["--root", str(root)])
    figs = [e for e in entries if e["kind"] == "figure"]
    assert len(figs) == 5 and all(e["status"] == "skipped" for e in figs)
    fig_dir = root / "results/paper_next/figures"
    assert not fig_dir.exists() or not any(fig_dir.glob("*.png"))
    assert "skipped" in (root / "results/paper_next/FIGURES.md").read_text(encoding="utf-8")
