"""CORE-02 (scripts/core02_refund.py). CPU only: synthetic ImageNet-100 stand-in, tiny controlled checkpoints.

  * the predict-only step writes predictions.json (UTC timestamps, structure and checkpoint hashes) and no surgery;
  * the surgery step adds one row per model and group, the law rows, scores, summary and meta;
  * prediction equals achieved saving: the compressed form of the surgery output has exactly the predicted free
    parameters, on the tiny models and on synthetic exactly equivariant layers for LR, R90 and Klein (where both
    also equal a brute-force count), and it reproduces the surgery output;
  * on an exactly UD-equivariant tiny model, UD surgery leaves the outputs unchanged and every wrong-group surgery
    changes them;
  * the surgery step refuses a structure that differs from the predicted one, and a rerun skips finished rows.
"""
import copy
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mirror import controlled as C
from mirror import groups as GR

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ["LR", "UD", "R90", "Klein"]
TINY = ["--tiny", "--conditions", "UD,LRUD,R90", "--groups", ",".join(GROUPS)]


def load_script():
    spec = importlib.util.spec_from_file_location("core02_refund_t", ROOT / "scripts" / "core02_refund.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("core02")
    mod = load_script()
    pred = mod.main(TINY + ["--predict-only", "--out", str(out)])
    snapshot = {"surgery_exists": (out / "surgery.csv").exists(),
                "predictions": json.loads((out / "predictions.json").read_text())}
    res = mod.main(TINY + ["--out", str(out)])
    return mod, out, pred, snapshot, res


def test_predict_only_comes_first(run):
    _, out, pred, snap, _ = run
    assert not snap["surgery_exists"]
    p = snap["predictions"]
    assert p["experiment"] == "CORE-02" and p["created_utc"].endswith("Z")
    assert set(p["models"]) == {"vit_test8_UD_s0", "vit_test8_LRUD_s0", "vit_test8_R90_s0"}
    m = p["models"]["vit_test8_LRUD_s0"]
    assert m["trained_group"] == "Klein" and m["scored"] and len(m["structure_sha256"]) == 64
    assert {g: m["groups"][g]["relation"] for g in GROUPS} == {"LR": "subgroup", "UD": "subgroup", "R90": "wrong",
                                                              "Klein": "trained"}
    r90 = p["models"]["vit_test8_R90_s0"]["groups"]
    assert {g: r90[g]["relation"] for g in GROUPS} == {"LR": "wrong", "UD": "wrong", "R90": "trained",
                                                       "Klein": "wrong"}
    assert [r["block"] for r in m["groups"]["R90"]["per_block"]] == [1, 2, 3]     # tiny model: 3 blocks
    assert all(0.0 < r["tax_mlp"] < 1.0 for g in GROUPS for r in m["groups"][g]["per_block"])
    assert (out / "meta_predict.json").exists()


def test_surgery_outputs_and_scores(run):
    _, out, _, snap, res = run
    for f in ("predictions.json", "surgery.csv", "law_per_block.csv", "scores.json", "summary.md", "meta.json"):
        assert (out / f).exists(), f
    srg = pd.read_csv(out / "surgery.csv")
    assert len(srg) == 3 * len(GROUPS)
    assert (srg["surgery_utc"] > srg["predicted_utc"]).all()
    assert srg["counts_equal_all_blocks"].all() and (srg["recon_max"] < 1e-4).all()
    assert np.allclose(srg["tax_predicted"], srg["tax_achieved"], atol=0, rtol=0)
    sc = res["scores"]
    assert sc["P2.1"]["pass"] is True and sc["P2.1"]["complete"]
    assert set(sc["models"]) == {"vit_test8_LRUD_s0", "vit_test8_R90_s0"}           # UD model reported only
    assert set(sc["models"]["vit_test8_LRUD_s0"]["P2.3"]["drops"]) == {"R90"}
    assert set(sc["models"]["vit_test8_LRUD_s0"]["subgroup_drops"]) == {"LR", "UD"}
    assert set(sc["models"]["vit_test8_R90_s0"]["P2.3"]["drops"]) == {"LR", "UD", "Klein"}
    meta = json.loads((out / "meta.json").read_text())
    assert meta["experiment"] == "CORE-02" and meta["prereg"]["sha256"] and meta["git"]["git_hash"]
    # snapshot of predictions unchanged by the surgery step
    assert json.loads((out / "predictions.json").read_text()) == snap["predictions"]


def test_prediction_equals_achieved_per_block(run):
    _, out, _, _, _ = run
    law = pd.read_csv(out / "law_per_block.csv")
    assert len(law) == 3 * len(GROUPS) * 3
    for k in ("fc1", "fc2", "b1", "b2"):
        assert (law[f"free_{k}_predicted"] == law[f"free_{k}_achieved"]).all(), k
    assert (law["diff_pp"] == 0).all() and law["counts_equal"].all()


def test_ud_exact_model_ud_surgery_keeps_outputs_wrong_groups_change_them(run):
    mod, out, _, _, _ = run
    st = mod.load_structure(out / "structure" / "vit_test8_UD_s0__e002.npz")
    model, _ = C.load_checkpoint_model(out / "tiny_runs" / "CORE-01" / "vit_test8_UD_s0" / "ckpt_e002.pt")
    x = torch.randn(8, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    blocks = [0, 1, 2]
    gaps = {}
    with torch.no_grad():
        y0 = model(x)
        for g in GROUPS:
            m = copy.deepcopy(model)
            GR.apply_group_surgery(m, blocks, [st["ops"][g][b] for b in blocks], [st["ops"][g][b + 1] for b in blocks],
                                   [st["perms"][g][b] for b in blocks], group=g)
            gaps[g] = float((m(x) - y0).norm() / y0.norm())
    assert gaps["UD"] < 1e-5
    assert all(gaps[g] > 1e-2 for g in ("LR", "R90", "Klein"))
    orig = np.load(out / "preds" / "vit_test8_UD_s0__original.npz")["orig"]
    assert np.array_equal(np.load(out / "preds" / "vit_test8_UD_s0__UD.npy"), orig)  # same predictions


def _synthetic_actions():
    """Exactly equivariant structures for LR (c2), R90 (c4) and Klein on small layers (h = 12, d = 6)."""
    rng = np.random.default_rng(0)
    d, h = 6, 12
    out = {}
    A1 = GR.random_c2(d, 4, 1)
    A2 = GR.random_c2(d, 3, 2)
    perm = GR.random_permutation_like(np.array([1, 0, 3, 2, 5, 4, 6, 7, 8, 9, 10, 11]), rng)
    out["LR"] = GR.block_action("LR", [A1], [A2], [perm])
    R1 = GR.random_c4(d, 2, 2, 1, 3)
    R2 = GR.random_c4(d, 1, 1, 2, 4)
    base = np.array([1, 2, 3, 0, 5, 6, 7, 4, 9, 8, 10, 11])          # two 4-cycles, a 2-cycle, two fixed
    out["R90"] = GR.block_action("R90", [R1], [R2], [GR.random_permutation_like(base, rng)])
    a_in, b_in = GR.random_klein(d, 2, 2, 1, 1, 5)
    a_out, b_out = GR.random_klein(d, 3, 1, 1, 1, 6)
    pa = np.array([1, 0, 3, 2, 5, 4, 6, 7, 9, 8, 10, 11])
    pb = np.array([2, 3, 0, 1, 4, 5, 7, 6, 9, 8, 10, 11])
    na, nb = GR.random_klein_action_like(pa, pb, rng)
    out["Klein"] = GR.block_action("Klein", [a_in, b_in], [a_out, b_out], [na, nb])
    return out, h, d


def test_compressed_count_equals_formula_and_brute_force():
    mod = load_script()
    acts, h, d = _synthetic_actions()
    g = torch.Generator().manual_seed(0)
    for name, act in acts.items():
        W1, b1 = torch.randn(h, d, generator=g, dtype=torch.float64), torch.randn(h, generator=g, dtype=torch.float64)
        W2, b2 = torch.randn(d, h, generator=g, dtype=torch.float64), torch.randn(d, generator=g, dtype=torch.float64)
        e = GR.project_mlp_group(W1, b1, W2, b2, act)
        cm = mod.compressed_mlp(e["W1"].numpy(), e["b1"].numpy(), e["W2"].numpy(), e["b2"].numpy(), act)
        f = GR.mlp_free_params(act, h, d)
        assert cm["free_fc1"] == f["free_fc1"] == GR.constraint_nullity(act, h, d, "fc1"), name
        assert cm["free_fc2"] == f["free_fc2"] == GR.constraint_nullity(act, h, d, "fc2"), name
        assert cm["free_b1"] == f["free_b1"] and cm["free_b2"] == f["free_b2"], name
        assert max(cm["recon_fc1"], cm["recon_fc2"], cm["recon_b1"], cm["recon_b2"]) < 1e-12, name
        # weights that are not equivariant are not reproduced: the check detects a broken surgery
        bad = mod.compressed_mlp(W1.numpy(), b1.numpy(), W2.numpy(), b2.numpy(), act)
        assert bad["recon_fc1"] > 0.1 and bad["recon_fc2"] > 0.1, name


def test_relations():
    mod = load_script()
    assert mod.relation("Klein", "Klein") == "trained"
    assert mod.relation("UD", "Klein") == "subgroup" and mod.relation("R180", "R90") == "subgroup"
    assert mod.relation("R90", "Klein") == "wrong" and mod.relation("LR", "R90") == "wrong"
    assert mod.relation("UD", "LR") == "wrong" and mod.relation("LR", None) == "wrong"


def test_refuses_changed_structure_and_rerun_skips(run, tmp_path):
    mod, out, _, _, _ = run
    n_rows = len(pd.read_csv(out / "surgery.csv"))
    before = {p.name: p.stat().st_mtime_ns for p in (out / "structure").glob("*.npz")}
    mod.main(TINY + ["--out", str(out)])                                           # nothing left to do
    assert len(pd.read_csv(out / "surgery.csv")) == n_rows
    assert {p.name: p.stat().st_mtime_ns for p in (out / "structure").glob("*.npz")} == before
    other = tmp_path / "copy"
    shutil.copytree(out, other)
    p = json.loads((other / "predictions.json").read_text())
    p["models"]["vit_test8_R90_s0"]["structure_sha256"] = "0" * 64
    (other / "predictions.json").write_text(json.dumps(p))
    (other / "surgery.csv").unlink()
    with pytest.raises(SystemExit, match="differs from predictions.json"):
        mod.main(TINY + ["--out", str(other)])
