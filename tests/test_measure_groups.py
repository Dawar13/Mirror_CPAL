"""scripts/measure_groups.py --tiny end to end (CPU, synthetic data, about a minute).

  * every output exists (layers, blocks, null draws, twin null, surgery, metrics.json, meta.json);
  * the LR path reproduces the existing pipeline: operators (d+), twin pairs and breaking shares equal
    variants.fit_structure + surgery.mlp_breaking_shares_two, and the cached LR matching equals its pairs;
  * the character-formula counts equal the Section 5.3 counts for every order-2 row;
  * metrics.json carries the ratio to null (PREREG definition, recomputed here from blocks.csv), the twin
    fraction, the twin-fraction null interval, the R90 share in 4-cycles and the surgery rows;
  * a rerun skips every finished row and every cached pass.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location("measure_groups", ROOT / "scripts" / "measure_groups.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ARGS = ["--tiny", "--groups", "LR,UD,R90,Klein", "--null-draws", "3", "--twin-null-draws", "2",
        "--twin-null-groups", "UD,R90", "--twin-null-blocks", "1", "--summary-ranges", "1,1-2",
        "--surgery", "UD", "--surgery-blocks", "1-2", "--experiment", "MUST-03-test"]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("groups")
    mod = load_script()
    metrics = mod.main(ARGS + ["--out", str(out)])
    return mod, out, metrics


def test_outputs_exist(run):
    _, out, metrics = run
    for name in ("layers.csv", "blocks.csv", "null_draws.csv", "twin_null.csv", "surgery.csv", "metrics.json",
                 "meta.json"):
        assert (out / name).exists(), name
    meta = json.loads((out / "meta.json").read_text())
    assert meta["experiment"] == "MUST-03-test" and meta["config"]["tiny"] is True
    assert meta["seeds"]["fit_seed"] == 0
    blk = pd.read_csv(out / "blocks.csv")
    assert set(blk["group"]) == {"LR", "UD", "R90", "Klein"}
    assert sorted(blk[blk["group"] == "UD"]["block"]) == [1, 2, 3, 4]          # 1-based blocks
    drw = pd.read_csv(out / "null_draws.csv")
    assert len(drw) == 4 * 4 * 3
    assert metrics["peak_gpu_memory_bytes"] is None                              # CPU run


def test_lr_reproduces_existing_pipeline(run):
    from mirror import variants as V
    from mirror.adapters import TimmTinyAdapter
    from mirror.surgery import mlp_breaking_shares_two

    _, out, _ = run
    ad = TimmTinyAdapter(img_size=56, seed=0)
    st = V.fit_structure(ad, V.TinyProtocol().fit_loader(0), "cpu", top_k=10, workers=1)
    lay = pd.read_csv(out / "layers.csv")
    lay = lay[lay["group"] == "LR"].sort_values("layer")
    assert lay["d_plus"].astype(int).tolist() == st.d_plus
    blk = pd.read_csv(out / "blocks.csv")
    blk = blk[blk["group"] == "LR"].sort_values("block")
    with np.load(out / "cache" / "pass_LR_fit0.npz") as z:
        perms = z["perm"]
    for b in range(ad.depth):
        assert np.array_equal(perms[b], st.perms[b])
        row = blk[blk["block"] == b + 1].iloc[0]
        assert int(row["n_pairs"]) == st.n_pairs(b) and int(row["n_self"]) == st.n_self(b)
        sh = mlp_breaking_shares_two(ad.model.blocks[b], st.A_layers[b], st.A_layers[b + 1], st.perms[b])
        assert row["s_fc1"] == pytest.approx(sh["s1"], rel=1e-12)
        assert row["s_fc2"] == pytest.approx(sh["s2"], rel=1e-12)
        assert row["s_b1"] == pytest.approx(sh["sb1"], rel=1e-10)


def test_character_counts_match_section53(run):
    _, out, _ = run
    blk = pd.read_csv(out / "blocks.csv")
    c2 = blk[blk["group"].isin(["LR", "UD"])]
    assert (c2["free_fc1"] == c2["free_fc1_section53"]).all()
    assert (c2["free_fc2"] == c2["free_fc2_section53"]).all()
    assert ((blk["tax_mlp"] > 0) & (blk["tax_mlp"] < 1)).all()


def test_metrics_scoring_fields(run):
    _, out, metrics = run
    blk = pd.read_csv(out / "blocks.csv")
    ud = blk[(blk["group"] == "UD") & blk["block"].isin([1, 2])]
    want = ud["s_mlp"].mean() / ud["s_mlp_null_mean"].mean()
    r = metrics["groups"]["UD"]["ranges"]["1-2"]
    assert r["ratio_to_null"] == pytest.approx(want, rel=1e-12)
    assert r["twin_fraction_t50"] == pytest.approx(ud["frac_2_t50"].mean())
    tn = metrics["groups"]["UD"]["ranges"]["1"]["twin_null"]["twin_fraction_t50"]
    assert tn["n_draws"] == 2 and tn["null_lo"] <= tn["null_mean"] <= tn["null_hi"]
    assert "share_4cycles" in metrics["groups"]["R90"]["ranges"]["1"]["twin_null"]
    assert metrics["scores"]["R90"]["share_4cycles_first_half"] is not None
    assert "ratio_to_null_first_half" in metrics["scores"]["Klein"]
    srg = pd.read_csv(out / "surgery.csv")
    assert set(srg["group"]) == {"LR", "UD"}
    assert np.isfinite(srg[srg["group"] == "UD"]["diff_vs_LR"]).all()
    assert "UD_1-2" in metrics["surgery"]


def test_reference_check(run, tmp_path):
    """--reference-run: the LR rows against e1_mlp.csv / e1_layers.csv of a run_e1.py run (here one built from
    variants.fit_structure on the same tiny model, so every difference is zero)."""
    from mirror import variants as V
    from mirror.adapters import TimmTinyAdapter
    from mirror.surgery import mlp_breaking_shares_two

    mod, out, _ = run
    ad = TimmTinyAdapter(img_size=56, seed=0)
    st = V.fit_structure(ad, V.TinyProtocol().fit_loader(0), "cpu", top_k=10, workers=1)
    res = tmp_path / "ref" / "results"
    res.mkdir(parents=True)
    pd.DataFrame({"layer": range(ad.depth + 1), "d_plus": st.d_plus}).to_csv(res / "e1_layers.csv", index=False)
    rows = []
    for b in range(ad.depth):
        sh = mlp_breaking_shares_two(ad.model.blocks[b], st.A_layers[b], st.A_layers[b + 1], st.perms[b])
        rows.append({"block": b + 1, "n_pairs": st.n_pairs(b), "n_self": st.n_self(b), "s1": sh["s1"], "s2": sh["s2"]})
    pd.DataFrame(rows).to_csv(res / "e1_mlp.csv", index=False)
    rc = mod.reference_check(out, str(tmp_path / "ref"), 0)
    assert rc["d_plus_max_abs_diff"] == 0 and rc["n_pairs_max_abs_diff"] == 0 and rc["n_self_max_abs_diff"] == 0
    assert rc["s_fc1_max_abs_diff"] < 1e-12 and rc["s_fc2_max_abs_diff"] < 1e-12
    assert rc["blocks_compared"] == [1, 2, 3, 4]


def test_must01_tiny(tmp_path):
    """scripts/must01_law.py --tiny: formula against the module count, with and without an operators npz."""
    spec = importlib.util.spec_from_file_location("must01_law", ROOT / "scripts" / "must01_law.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.main(["--tiny", "--out", str(tmp_path)])
    assert m["pass"] and m["max_abs_diff_pp"] == 0.0
    df = pd.read_csv(tmp_path / "law_per_layer.csv")
    assert set(df["structure"]) == {"fitted operators and pairings", "synthetic structure with the recorded counts"}
    assert df["counts_match_csv"].all() and df["bias_match"].all()
    assert (df["diff_pp_recorded"].abs() < 1e-9).all()
    assert (tmp_path / "summary.md").exists() and (tmp_path / "meta.json").exists()


def test_controlled_checkpoint_path(tmp_path, monkeypatch):
    """--checkpoint --arch controlled with a synthetic ImageNet-100 cache and a tiny controlled ViT (CPU)."""
    ctl = pytest.importorskip("mirror.controlled")
    import torch

    from mirror import groups as GR

    for view in ("LR", "UD", "R90", "R180", "R270"):      # both modules use the same token conventions
        assert np.array_equal(GR.token_index(view, 8), ctl.view_token_permutation(view, 8))
    cache = tmp_path / "in100"
    ctl.write_synthetic_cache(cache, num_classes=10, train_per_class=8, val_per_class=2)
    monkeypatch.setenv("IN100_CACHE_DIR", str(cache))
    torch.manual_seed(0)
    model = ctl.build_model("vit_test8", num_classes=10)
    ckpt = tmp_path / "ckpt_e005.pt"
    torch.save({"format": ctl.CKPT_FORMAT, "arch": "vit_test8", "condition": "LR", "seed": 0, "epoch": 5,
                "img_size": 64, "patch_size": 8, "num_classes": 10,
                "model": {k: v.detach().clone() for k, v in model.state_dict().items()}}, ckpt)
    out = tmp_path / "out"
    m = load_script().main(["--checkpoint", str(ckpt), "--arch", "controlled", "--device", "cpu",
                            "--groups", "LR,UD,R90", "--fit-images", "20", "--null-draws", "2",
                            "--twin-null-draws", "1", "--twin-null-blocks", "1", "--summary-ranges", "1-2",
                            "--experiment", "MUST-12-test", "--out", str(out)])
    assert m["model"] == "controlled_vit_test8_LR_s0_e5" and m["grid"] == [8, 8] and m["n_fit_images"] == 20
    blk = pd.read_csv(out / "blocks.csv")
    assert set(blk["group"]) == {"LR", "UD", "R90"} and sorted(blk["block"].unique()) == [1, 2, 3]
    assert "ratio_to_null" in m["groups"]["UD"]["ranges"]["1-2"]
    meta = json.loads((out / "meta.json").read_text())
    assert meta["models"]["controlled_vit_test8_LR_s0_e5"]["checkpoint_info"]["condition"] == "LR"


def test_rerun_skips_finished_work(run):
    mod, out, _ = run
    before = {p.name: p.stat().st_mtime_ns for p in (out / "cache").glob("pass_*.npz")}
    n_rows = {n: len(pd.read_csv(out / n)) for n in ("layers.csv", "blocks.csv", "twin_null.csv", "surgery.csv")}
    mod.main(ARGS + ["--out", str(out)])
    after = {p.name: p.stat().st_mtime_ns for p in (out / "cache").glob("pass_*.npz")}
    assert before == after
    for n, k in n_rows.items():
        assert len(pd.read_csv(out / n)) == k, n
