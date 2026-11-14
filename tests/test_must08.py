"""MUST-08 (scripts/must08_components.py, a mode of scripts/must07_compressors.py): matched component ablation of
Mirror-exact, tiny random ViT and synthetic data on CPU."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_m08_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


W = load_script("must08_components")
M = load_script("must07_compressors")
SIZES = ["2mlp", "2attn", "2both", "fullmlp", "fullattn", "full"]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("must08")
    sdir, out7, out8 = base / "structure", base / "m07", base / "m08"
    r7 = M.main(["--tiny", "--methods", "mirror_exact", "--sizes", "2mlp,2both,full", "--out", str(out7),
                 "--structure-dir", str(sdir)])
    r8 = W.main(["--tiny", "--out", str(out8), "--structure-dir", str(sdir), "--reuse-from", str(out7)])
    return out7, out8, r7, r8


def test_must08_rows_reuse_and_costs(runs):
    out7, out8, r7, r8 = runs
    df = pd.read_csv(out8 / "rows_tiny.csv")
    me = df[(df["method"] == "mirror_exact") & (df["status"] == "ok")].set_index("size")
    assert sorted(me.index) == sorted(SIZES)
    # Rows MUST-07 already scored with the same structure are copied, with their predictions; the rest are new.
    reused = me["reused_from"].fillna("").astype(str)
    assert {s for s in SIZES if reused[s]} == {"2mlp", "2both", "full"}
    for s in SIZES:
        assert M.preds_path(out8, "tiny", s, "mirror_exact", 0).is_file()
    df7 = pd.read_csv(out7 / "rows_tiny.csv").set_index(["size", "method"])
    assert me.loc["2both", "val_top1"] == df7.loc[("2both", "mirror_exact"), "val_top1"]
    # Components add up: MLP-only plus attention-only matrices equal both; at the full model (no basis change) the
    # FLOP and parameter savings add up too.
    for span, (a, b, c) in {"2": ("2mlp", "2attn", "2both"), "full": ("fullmlp", "fullattn", "full")}.items():
        assert me.loc[a, "params_matrices"] + me.loc[b, "params_matrices"] == me.loc[c, "params_matrices"]
        assert me.loc[a, "params_basis_change"] == me.loc[b, "params_basis_change"] == me.loc[c, "params_basis_change"]
        assert (me.loc[[a, b, c], "gap"] == 0).all()
    orig = df[df["method"] == "original"].iloc[0]
    sav = {s: orig["macs"] - me.loc[s, "macs"] for s in ("fullmlp", "fullattn", "full")}
    assert sav["fullmlp"] + sav["fullattn"] == sav["full"]
    psav = {s: orig["model_params"] - me.loc[s, "model_params"] for s in ("fullmlp", "fullattn", "full")}
    assert psav["fullmlp"] + psav["fullattn"] == psav["full"]
    # Flip consistency in fp32 for every full-model row; only MLP plus attention at the full model claims exactness.
    for s in SIZES:
        assert np.isfinite(me.loc[s, "flip_consistency_fp32"]) == s.startswith("full")
        assert bool(me.loc[s, "claims_exact_invariance"]) == (s == "full")
    assert me.loc["full", "flip_consistency_fp32"] == 100.0
    meta = json.loads((out8 / "meta.json").read_text())
    assert meta["experiment"] == "MUST-08" and meta["config"]["experiment"] == "MUST-08"
    text = (out8 / "SUMMARY.md").read_text(encoding="utf-8")
    assert "P8.1" in text and "P8.2" in text and chr(0x2014) not in text and "Gate B" not in text
    assert not (out8 / "gate_b.json").exists()
    assert r8["rows_added"] == 7                     # 6 sizes and the original


def test_must08_resume_and_wrapper_guard(runs):
    out7, out8, _, _ = runs
    res = W.main(["--tiny", "--out", str(out8), "--structure-dir", str(out8.parent / "structure"),
                  "--reuse-from", str(out7)])
    assert res["rows_added"] == 0
    with pytest.raises(SystemExit):
        W.main(["--experiment", "MUST-07", "--tiny", "--out", str(out8)])


def test_p8_checks_logic():
    rows = [{"model": "m", "method": "original", "size": "original", "fit_seed": -1, "status": "ok",
             "model_params": 1000, "val_top1": 80.0}]
    for seed in (0, 1, 2):
        for s, (p, acc) in {"6mlp": (800, 77.0), "6attn": (900, 76.0), "6both": (700, 76.8),
                            "fullmlp": (600, 70.0), "fullattn": (850, 60.0), "full": (500, 55.0)}.items():
            rows.append({"model": "m", "method": "mirror_exact", "size": s, "fit_seed": seed, "status": "ok",
                         "model_params": p, "val_top1": acc + 0.1 * seed})
    chk = M.p8_checks(pd.DataFrame(rows), "m", tiny=False)
    six, full = chk["6 blocks"], chk["full model"]
    assert six["P8.1"] is True and full["P8.1"] is True                   # 200 > 150 and 400 > 250
    assert six["mlp_minus_both"] == pytest.approx(0.2) and six["P8.2"] is True and six["seeds"] == [0, 1, 2]
    rows[1]["val_top1"] = 90.0                                            # seed 0, 6mlp: now 13.2 points above both
    assert M.p8_checks(pd.DataFrame(rows), "m", tiny=False)["6 blocks"]["P8.2"] is False
