"""Tier 2 healed V-exact variant (healed_vexact_k<K> in scripts/tier2_linear.py and scripts/tier2_transfer.py):
CPU tests with tiny scripts/heal_vexact.py runs (tiny random ViT, 4 blocks: k = 4 is the full conversion with
the even head, k = 2 a partial span)."""
import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mirror.data import mirror, normalize_for, synthetic_dataset
from mirror.heal_vexact import build_vexact_student

ROOT = Path(__file__).resolve().parents[1]
CONFIG = str(ROOT / "configs" / "m3_vexact.yaml")
FAST = ["--max-iter", "60", "--wd-grid", "1e-3,1e-2", "--no-wd-refine"]


def load_script(name):
    key = f"{name}_vxt2test"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod          # dataclasses need the module registered while it executes
        spec.loader.exec_module(mod)
    return sys.modules[key]


T2 = load_script("tier2_linear")


@pytest.fixture(scope="module")
def vexact_runs(tmp_path_factory):
    """Tiny V-exact healing runs for k = 4 (full conversion) and k = 2, plus the in-memory healed students
    (captured where scripts/heal_vexact.py hands them to its final k-NN)."""
    root = tmp_path_factory.mktemp("m3_vexact")
    hv = load_script("heal_vexact")
    captured = {}
    knn = hv.heal_cli.evaluate_final_knn

    def capture(cfg, teacher, student, make_student, device, out, git):
        captured[int(cfg["k"])] = copy.deepcopy(student)
        return knn(cfg, teacher, student, make_student, device, out, git)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(hv.heal_cli, "evaluate_final_knn", capture)
        for k in (4, 2):
            hv.main(["--config", CONFIG, "--tiny", "--k", str(k), "--lr", "1e-3", "--max-steps", "3",
                     "--out-dir", str(root)])
    assert set(captured) == {4, 2}
    return root, captured


def images(n_per_class=3, seed=5):
    ad = T2.build_adapter("tiny", True, "cpu")
    raw = torch.from_numpy(synthetic_dataset(2, n_per_class, 56, seed=seed)[0]).permute(0, 3, 1, 2)
    return normalize_for(ad, raw)


def flip_rel(ad, model, x) -> float:
    with torch.no_grad():
        f = T2.probe_feature(ad, x, model).double()
        ft = T2.probe_feature(ad, mirror(x), model).double()
    return float((ft - f).abs().max() / f.abs().max())


# ---------------------------------------------------------------- the variant is the healed student
def test_variant_is_the_healed_student(vexact_runs):
    root, captured = vexact_runs
    ad = T2.build_adapter("tiny", True, "cpu")
    fac = T2.VariantFactory(ad, "tiny", True, vexact_dir=root)
    x = images()
    for k in (4, 2):
        v = T2.parse_variant(f"healed_vexact_k{k}")
        m, info = fac.build(v)
        # Identical outputs and state to the student scripts/heal_vexact.py healed and evaluated.
        with torch.no_grad():
            got, ref = T2.probe_feature(ad, x, m), T2.probe_feature(ad, x, captured[k])
        assert (got - ref).abs().max().item() == 0.0
        sd, sd_ref = m.state_dict(), captured[k].state_dict()
        assert set(sd) == set(sd_ref) and all(torch.equal(sd[n], sd_ref[n]) for n in sd)
        ck = info["checkpoint_check"]
        assert ck["all_equal"] and ck["scope"] == "all" and ck["n_tensors"] == ck["n_model_tensors"]
        assert ck["blocks"] == list(range(k)) and info["even_head"] == (k == 4)
        assert info["family"] == "healed_vexact" and info["k"] == k and info["train_embeddings"]
        # The checks after loading reproduce the values heal_vexact.py recorded after healing (same images,
        # same model, CPU), and pass.
        vc = info["vexact_checks"]
        assert vc["pass"] and vc["span_exact_pass"] and vc["full_conversion"] == (k == 4)
        rec = json.loads((root / f"k{k}" / "exactness.json").read_text())["max_rel_inside"]
        assert vc["span_exact_max_rel_inside"] == rec
        assert vc["recorded_after_healing"]["exactness.json:max_rel_inside"] == rec
        if k == 4:
            fl = json.loads((root / "k4" / "flipcheck.json").read_text())
            assert vc["probe_feature_flip"]["knn_cls_feature_rel_max"] == fl["cls_feature_rel_max"]
            assert vc["probe_feature_flip"]["rel_max"] < 1e-5
        # Parameter counts as heal_vexact.py reports them.
        summ = json.loads((root / f"k{k}" / "summary.json").read_text())
        for key in ("converted_params_kept", "converted_params_dense", "model_params"):
            assert info[key] == summ["method_info"][key], key
        assert "even head" in info["feature"] if k == 4 else "no even head" in info["feature"]
        # Teeth: the same build without the checkpoint (zero-shot V-exact) gives other features.
        st = fac.heal_vexact().load_vexact_structure(summ["config"], ad, k)
        zs, _, _ = build_vexact_student(ad, st["A_star"], k, st["perms"], st["heads"], st["reg_perm"])
        with torch.no_grad():
            assert (T2.probe_feature(ad, x, zs) - got).abs().max().item() > 1e-4


def test_full_conversion_features_are_mirror_invariant(vexact_runs):
    root, _ = vexact_runs
    ad = T2.build_adapter("tiny", True, "cpu")
    fac = T2.VariantFactory(ad, "tiny", True, vexact_dir=root)
    x = images(seed=11)
    full, _ = fac.build(T2.parse_variant("healed_vexact_k4"))
    assert flip_rel(ad, full, x) < 1e-5                       # [CLS, mean patch] of x and mirror(x) agree (fp32)
    # The same through the probe's own extraction (float32 features, no autocast on CPU).
    from mirror.data import make_loader

    imgs, labels = synthetic_dataset(2, 3, 56, seed=11)
    loader = make_loader(imgs, labels, np.arange(len(labels)), batch_size=4)
    flipped = make_loader(np.ascontiguousarray(imgs[:, :, ::-1]), labels, np.arange(len(labels)), batch_size=4)
    X, _ = T2.extract_features(ad, loader, "cpu", full, amp=None, dtype=np.float32)
    Xm, _ = T2.extract_features(ad, flipped, "cpu", full, amp=None, dtype=np.float32)
    assert np.abs(Xm - X).max() / np.abs(X).max() < 1e-5
    # Teeth: a partial span (blocks 3 and 4 unconverted, no even head) and the original are not invariant.
    part, _ = fac.build(T2.parse_variant("healed_vexact_k2"))
    assert flip_rel(ad, part, x) > 1e-3 and flip_rel(ad, ad.model, x) > 1e-3


# ---------------------------------------------------------------- CLI: linear probe and transfer
def test_cli_linear_and_transfer_record_feature_and_checks(vexact_runs, tmp_path):
    root, _ = vexact_runs
    out = tmp_path / "t2"
    names = ["original", "healed_vexact_k4", "healed_vexact_k2"]
    res = T2.main(["--tiny", "--variants", ",".join(names), "--out", str(out), "--vexact-dir", str(root), *FAST])
    rows = {r["variant"]: r for r in res["rows"]}
    assert list(rows) == names
    r4, r2 = rows["healed_vexact_k4"], rows["healed_vexact_k2"]
    assert r4["family"] == "healed_vexact" and r4["method"] == "vexact" and r4["k"] == 4
    assert r4["vexact_checks_pass"] and r4["vexact_even_head"] and r4["vexact_feature_flip_rel_max"] < 1e-5
    assert r4["vexact_span_exact_rel_max"] < 1e-4 and "even head y (I + A*)/2" in r4["notes"] and "PASS" in r4["notes"]
    assert r2["vexact_checks_pass"] and not r2["vexact_even_head"] and r2["vexact_feature_flip_rel_max"] > 1e-3
    assert "no even head" in r2["notes"] and "notes" not in rows["original"]
    assert "check images, float32)" in r4["notes"] and "(fp32 features)" in r4["notes"]    # tiny k-NN ran on CPU
    assert r4["feature_dim"] == 128 and 0 <= r4["top1"] <= 100
    df = pd.read_csv(out / "t2_linear.csv")
    assert set(df["variant"]) == set(names) and df["git_hash"].nunique() == 1
    by = df.set_index("variant")
    assert by.loc["healed_vexact_k4", "vexact_checks_pass"] and pd.isna(by.loc["original", "notes"])
    assert "even head" in by.loc["healed_vexact_k4", "notes"]
    md = (out / "T2_LINEAR.md").read_text(encoding="utf-8")
    assert chr(0x2014) not in md and "healed_vexact_k4" in md and "even head, feature mirror-invariant" in md
    vj = json.loads((out / "probe" / "tiny" / "healed_vexact_k4.json").read_text())
    assert vj["sources"]["checkpoint"].endswith("trained_blocks.pt") and len(vj["sources"]["checkpoint_sha256"]) == 64
    # Rerun: cached features and probes are reused; the row (with its checks) comes back unchanged.
    res2 = T2.main(["--tiny", "--variants", "healed_vexact_k4", "--out", str(out), "--vexact-dir", str(root), *FAST])
    assert res2["rows"][0]["notes"] == r4["notes"] and res2["rows"][0]["top1"] == r4["top1"]
    # Transfer probes: the same variant, feature statement and checks.
    TR = load_script("tier2_transfer")
    tout = tmp_path / "tr"
    rt = TR.main(["--tiny", "--variants", "healed_vexact_k4", "--out", str(tout), "--vexact-dir", str(root), *FAST])
    t = rt["rows"][0]
    assert t["variant"] == "healed_vexact_k4" and t["vexact_checks_pass"] and t["vexact_even_head"]
    assert t["vexact_feature_flip_rel_max"] < 1e-5 and "even head" in t["notes"]
    dft = pd.read_csv(tout / "t2_transfer.csv")
    assert list(dft["variant"]) == ["healed_vexact_k4"] and "notes" in dft.columns
    tmd = (tout / "T2_TRANSFER.md").read_text(encoding="utf-8")
    assert chr(0x2014) not in tmd and "- healed_vexact_k4: feature [CLS, mean patch token]" in tmd


# ---------------------------------------------------------------- failures are loud
def test_missing_checkpoint_and_wrong_load_fail_clearly(vexact_runs, tmp_path):
    root, _ = vexact_runs
    ad = T2.build_adapter("tiny", True, "cpu")
    v4, v2, v3 = (T2.parse_variant(f"healed_vexact_k{k}") for k in (4, 2, 3))
    with pytest.raises(FileNotFoundError, match="--vexact-dir"):
        T2.VariantFactory(ad, "tiny", True).sources(v4)
    with pytest.raises(FileNotFoundError, match=r"not found\. Heal it first .*heal_vexact\.py .*--k 3"):
        T2.VariantFactory(ad, "tiny", True, vexact_dir=root).sources(v3)
    with pytest.raises(FileNotFoundError, match="trained_blocks.pt not found"):    # the CLI stops before extracting
        T2.main(["--tiny", "--variants", "original,healed_vexact_k3", "--out", str(tmp_path / "t2"),
                 "--vexact-dir", str(root), *FAST])
    assert not (tmp_path / "t2" / "features").exists()
    ck4 = root / "k4" / "final" / "trained_blocks.pt"
    with pytest.raises(FileNotFoundError, match="not found"):
        T2.VariantFactory(ad, "tiny", True, vexact_checkpoint=tmp_path / "nope.pt").sources(v4)
    # One checkpoint file cannot serve two variants; a checkpoint of another k is refused.
    with pytest.raises(ValueError, match="--vexact-checkpoint"):
        T2.main(["--tiny", "--variants", "healed_vexact_k4,healed_vexact_k2", "--out", str(tmp_path / "t2"),
                 "--vexact-checkpoint", str(ck4), *FAST])
    with pytest.raises(ValueError, match="k = 4, not 2"):
        T2.VariantFactory(ad, "tiny", True, vexact_checkpoint=ck4).build(v2)
    m, info = T2.VariantFactory(ad, "tiny", True, vexact_checkpoint=ck4).build(v4)    # the override itself works
    assert info["checkpoint"] == str(ck4) and info["vexact_checks"]["pass"]
    # A student built with another structure (here the span operator of another seed) does not pass silently.
    bad = tmp_path / "bad"
    shutil.copytree(root / "k4", bad / "k4")
    s = json.loads((bad / "k4" / "summary.json").read_text())
    s["config"]["seed"] = 1
    (bad / "k4" / "summary.json").write_text(json.dumps(s))
    with pytest.raises(RuntimeError, match="not exactly mirror-equivariant"):
        T2.VariantFactory(ad, "tiny", True, vexact_dir=bad).build(v4)
