"""Tier 2 linear probe (scripts/tier2_linear.py): CPU tests with the tiny random ViT and synthetic data."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mirror.adapters import TimmTinyAdapter
from mirror.attention import attn_dims
from mirror.data import make_loader, make_splits, normalize_for, synthetic_dataset
from mirror.m1 import fit_span_operator, random_head_structure, random_involution_fixing_ones
from mirror.matching import random_pairing

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    import sys

    key = f"{name}_t2test"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod          # dataclasses need the module registered while it executes
    spec.loader.exec_module(mod)
    return mod


T2 = load_script("tier2_linear")
FAST = ["--max-iter", "60", "--wd-grid", "1e-3,1e-2", "--no-wd-refine"]


def blobs(n_classes=6, per=40, dim=16, sep=4.0, seed=0, sample_seed=None):
    """Gaussian classes: means from `seed`, samples from `sample_seed` (independent draws per set)."""
    means = np.random.default_rng(seed).standard_normal((n_classes, dim)) * sep
    rng = np.random.default_rng(1000 + seed if sample_seed is None else sample_seed)
    y = np.repeat(np.arange(n_classes), per)
    X = means[y] + rng.standard_normal((len(y), dim))
    return X.astype(np.float16), y.astype(np.int64)


# ---------------------------------------------------------------- variant names
def test_requested_variant_names_resolve():
    names = ("original,mirror_mlp_k6,vexact_k6,vlocal_k9,svd_mlp_k6,random_k6,healed_mirror_k9,healed_svd_k9,"
             "healed_random_k9")
    got = {v.name: (v.family, v.method, v.k) for v in T2.parse_variants(names)}
    assert got == {"original": ("original", "original", 0), "mirror_mlp_k6": ("mlp", "mirror", 6),
                   "vexact_k6": ("m1", "vexact", 6), "vlocal_k9": ("m1", "vlocal", 9), "svd_mlp_k6": ("mlp", "svd", 6),
                   "random_k6": ("mlp", "random", 6), "healed_mirror_k9": ("healed", "mirror", 9),
                   "healed_svd_k9": ("healed", "svd", 9), "healed_random_k9": ("healed", "random", 9)}
    extra = {v.name: (v.family, v.method, v.k) for v in T2.parse_variants("random_mlp_k3,svd_attn_k12,random_attn_k6")}
    assert extra == {"random_mlp_k3": ("mlp", "random", 3), "svd_attn_k12": ("m1", "svd", 12),
                     "random_attn_k6": ("m1", "random", 6)}
    assert [v.name for v in T2.parse_variants("original, original,vexact_k6")] == ["original", "vexact_k6"]
    for v in T2.parse_variants(names):
        assert v.label and "{k}" not in v.label
    vx = {v.name: (v.family, v.method, v.k) for v in T2.parse_variants("healed_vexact_k12,healed_vexact_k6")}
    assert vx == {"healed_vexact_k12": ("healed_vexact", "vexact", 12), "healed_vexact_k6": ("healed_vexact", "vexact", 6)}
    for bad in ("mirror_k6", "vexact", "healed_vexact", "healed_vexact_k0", "healed_vlocal_k9", "svd_mlp_k0", "Original", ""):
        with pytest.raises(ValueError):
            T2.parse_variants(bad)


# ---------------------------------------------------------------- features
def test_feature_extraction_shapes_and_content():
    ad = TimmTinyAdapter(img_size=56, seed=0)
    images, labels = synthetic_dataset(3, 5, 56, seed=1)
    loader = make_loader(images, labels, np.arange(len(labels)), batch_size=4)
    X, y = T2.extract_features(ad, loader, "cpu", amp="bf16")
    d = ad.embed_dim
    assert X.shape == (15, 2 * d) and X.dtype == np.float16 and y.dtype == np.int64
    assert np.array_equal(y, labels)
    # First half: CLS after the final norm (= the Tier-1 k-NN feature); second half: mean patch token.
    x = normalize_for(ad, torch.from_numpy(images).permute(0, 3, 1, 2))
    with torch.no_grad():
        t = ad.model.norm(ad.forward_tokens(x))
        cls = ad.cls_feature(x)
    assert np.abs(X[:, :d].astype(np.float32) - cls.numpy()).max() < 1e-2
    patch = t[:, ad.n_prefix_tokens:].mean(dim=1).numpy()
    assert np.abs(X[:, d:].astype(np.float32) - patch).max() < 1e-2
    X32, _ = T2.extract_features(ad, loader, "cpu", dtype=np.float32)
    assert X32.dtype == np.float32 and np.abs(X32 - X.astype(np.float32)).max() < 1e-2
    assert T2.feature_kind(ad) == "cls"
    assert T2.feature_kind(SimpleNamespace(feature="pooled", has_cls_token=False)) == "pooled"


# ---------------------------------------------------------------- probe
def test_probe_learns_separable_problem():
    Xb, yb = blobs(seed=0)
    Xv, yv = blobs(per=15, seed=0, sample_seed=77)          # same classes, fresh samples
    res = T2.run_probe(Xb, yb, Xv, yv, 6, [1e-4, 1e-3, 1e-2], holdout_frac=0.1, seed=0, device="cpu",
                       solver={"max_iter": 200})
    assert res["test"]["top1"] >= 95.0 and res["test"]["top5"] == 100.0
    assert res["selection"]["chosen_wd"] in [t["wd"] for t in res["selection"]["trials"]]
    assert res["W"].shape == (16, 6) and res["b"].shape == (6,)
    # Shuffled labels: the probe cannot beat chance by much on held-out data.
    ys = np.random.default_rng(2).permutation(yb)
    res2 = T2.run_probe(Xb, ys, Xv, yv, 6, [1e-2], holdout_frac=0.1, seed=0, refine=False, device="cpu",
                        solver={"max_iter": 100})
    assert res2["test"]["top1"] < 50.0


def test_fit_logreg_matches_direct_minimiser():
    """Converged L-BFGS solution has (near) zero gradient of the stated objective."""
    Xb, yb = blobs(n_classes=3, per=30, dim=5, sep=1.0, seed=3)
    X = torch.as_tensor(Xb, dtype=torch.float32)
    y = torch.as_tensor(yb)
    wd = 1e-2
    r = T2.fit_logreg(X, y, 3, wd, max_iter=500, tol_grad=1e-7)
    W, b = r["W"].clone().requires_grad_(True), r["b"].clone().requires_grad_(True)
    loss = torch.nn.functional.cross_entropy(X @ W + b, y) + 0.5 * wd * (W * W).sum()
    loss.backward()
    assert W.grad.abs().max() < 1e-4 and b.grad.abs().max() < 1e-4 and not r["hit_limit"]
    # Warm start from the solution changes nothing.
    r2 = T2.fit_logreg(X, y, 3, wd, init=(r["W"], r["b"]), max_iter=500, tol_grad=1e-7)
    assert (r2["W"] - r["W"]).abs().max() < 1e-3


def test_weight_decay_split_is_disjoint_and_never_uses_the_test_set():
    _, yb = blobs(n_classes=5, per=20)
    tr, ho = T2.stratified_holdout(yb, 0.1, seed=0)
    assert len(np.intersect1d(tr, ho)) == 0 and np.array_equal(np.union1d(tr, ho), np.arange(len(yb)))
    assert all((yb[ho] == c).sum() == 2 for c in range(5))                     # 10% of 20 per class
    tr2, ho2 = T2.stratified_holdout(yb, 0.1, seed=0)
    assert np.array_equal(ho, ho2)                                             # fixed split
    assert not np.array_equal(ho, T2.stratified_holdout(yb, 0.1, seed=1)[1])
    # The choice of wd depends on the bank only: replacing the test set changes nothing in it.
    Xb, yb = blobs(seed=4, sep=1.0)
    Xv, yv = blobs(per=10, seed=4, sep=1.0, sample_seed=55)
    junk = np.random.default_rng(9).standard_normal(Xv.shape).astype(np.float16)
    kw = dict(holdout_frac=0.1, seed=0, device="cpu", solver={"max_iter": 80})
    r1 = T2.run_probe(Xb, yb, Xv, yv, 6, [1e-4, 1e-3, 1e-2], **kw)
    r2 = T2.run_probe(Xb, yb, junk, np.zeros_like(yv), 6, [1e-4, 1e-3, 1e-2], **kw)
    strip = lambda s: [{k: v for k, v in t.items() if k != "seconds"} for t in s["trials"]]  # noqa: E731
    assert strip(r1["selection"]) == strip(r2["selection"])
    assert r1["selection"]["chosen_wd"] == r2["selection"]["chosen_wd"]
    assert np.array_equal(r1["holdout_positions"], r2["holdout_positions"])
    assert r1["holdout_positions"].max() < len(yb)                             # positions into the bank
    # select_weight_decay has no test-set argument at all.
    import inspect

    params = set(inspect.signature(T2.select_weight_decay).parameters)
    assert not {"Xv", "yv", "X_test", "y_test"} & params


def test_protocol_data_reads_bank_from_train_and_test_from_validation(tmp_path):
    """ProtocolData on a miniature fake cache: the bank comes from the train array (bank rows), the test
    set from the validation array, so the weight-decay split (positions into the bank) cannot touch it."""
    rng = np.random.default_rng(0)
    n_tr, n_val = 12, 6
    np.save(tmp_path / "train224_uint8.npy", rng.integers(0, 256, (n_tr, 224, 224, 3), dtype=np.uint8))
    lab = np.arange(n_tr) % 3
    idx = {f"fit{s}_rows": np.array([s]) for s in range(3)}
    idx.update({f"fit{s}_labels": lab[[s]] for s in range(3)})
    idx.update({"bank_rows": np.arange(3, n_tr), "bank_labels": lab[3:]})
    np.savez(tmp_path / "train_index.npz", **idx)
    np.save(tmp_path / "val224_uint8.npy", rng.integers(0, 256, (n_val, 224, 224, 3), dtype=np.uint8))
    np.save(tmp_path / "val_labels.npy", np.arange(n_val) % 3)
    np.savez(tmp_path / "splits.npz", query=np.arange(n_val))
    (tmp_path / "meta.json").write_text(json.dumps({"complete": True, "n_classes": 3}))
    d = T2.ProtocolData(tmp_path, batch_size=4, num_workers=0, pin=False)
    bank, val = d.loader("bank"), d.loader("val")
    assert bank.dataset.images is d.proto.train and val.dataset.images is d.proto.val
    assert np.array_equal(d.rows("bank"), np.arange(3, n_tr)) and np.array_equal(d.rows("val"), np.arange(n_val))
    ys = np.concatenate([y.numpy() for _, y in bank])
    assert np.array_equal(ys, lab[3:])
    tr, ho = T2.stratified_holdout(ys, 0.34, seed=0)
    assert ho.max() < len(bank.dataset) and len(np.intersect1d(tr, ho)) == 0


# ---------------------------------------------------------------- CLI: resume
def test_cli_tiny_resume_skips_cached_features(tmp_path, monkeypatch):
    out = tmp_path / "t2"
    args = ["--tiny", "--variants", "original,mirror_mlp_k2", "--out", str(out), *FAST]
    res = T2.main(args)
    assert [r["variant"] for r in res["rows"]] == ["original", "mirror_mlp_k2"]
    for v in ("original", "mirror_mlp_k2"):
        for s in ("bank", "val"):
            assert (out / "features" / "tiny" / f"{v}_{s}.npz").is_file()
    with np.load(out / "features" / "tiny" / "original_bank.npz") as z:
        meta = json.loads(str(z["meta_json"]))
        assert z["X"].dtype == np.float16 and z["X"].shape == (80, 128) and meta["recipe"] == T2.FEATURE_RECIPE
    mt = {p: p.stat().st_mtime_ns for p in (out / "features").rglob("*.npz")}
    pj = (out / "probe" / "tiny" / "original.json").stat().st_mtime_ns

    def boom(*a, **k):
        raise AssertionError("features were recomputed")

    monkeypatch.setattr(T2, "extract_features", boom)
    res2 = T2.main(args)
    assert [r["top1"] for r in res2["rows"]] == [r["top1"] for r in res["rows"]]
    assert all(p.stat().st_mtime_ns == t for p, t in mt.items())
    assert (out / "probe" / "tiny" / "original.json").stat().st_mtime_ns == pj        # probe skipped too
    # A setting that changes the features (no autocast) invalidates the cache.
    with pytest.raises(AssertionError, match="recomputed"):
        T2.main(["--tiny", "--variants", "original", "--out", str(out), "--amp", "none", *FAST])
    monkeypatch.undo()
    # --force-probe refits from the cached features and gives the same numbers (fixed seed).
    res3 = T2.main(args + ["--force-probe"])
    assert [r["top1"] for r in res3["rows"]] == [r["top1"] for r in res["rows"]]


# ---------------------------------------------------------------- CLI: every variant family
def _tiny_operators(path):
    ad = TimmTinyAdapter(img_size=56, seed=0)
    d, h, L = ad.embed_dim, ad.mlp_hidden, ad.depth
    rng = np.random.default_rng(0)
    A = np.stack([random_involution_fixing_ones(d, 30 + l, seed=l) for l in range(L + 1)]).astype(np.float32)
    C = np.stack([a + 0.3 * rng.standard_normal((d, d)) for a in A]).astype(np.float32)
    G = np.stack([np.eye(d) * (1 + l) for l in range(L + 1)]).astype(np.float32)
    perms = np.stack([random_pairing(h, 80, seed=b) for b in range(L)]).astype(np.int32)
    np.savez_compressed(path, A_layers=A, C=C, G=G, perms=perms)
    return path


def _tiny_m1_structure(path, ops_path, ks=(2,)):
    """An m1_structure.npz in the layout scripts/m1_attention.py writes (random head maps)."""
    m1 = load_script("m1_attention")
    ad = TimmTinyAdapter(img_size=56, seed=0)
    H, dh = attn_dims(ad.model.blocks[0])
    ops = m1.load_operators(ops_path, ad.depth, ad.embed_dim, ad.mlp_hidden)
    maps = []
    for b in range(ad.depth):
        hm = random_head_structure(H, dh, 1, seed=40 + b)
        hm.scores = np.zeros((H, H))
        maps.append(hm)
    S = {"maps": maps, "a_out_offset": 1, "n_reg": 0,
         "A_star": {k: fit_span_operator(ops["C"], range(k + 1), ops["G"])[0] for k in ks},
         "reg_perm": {k: np.zeros(0, dtype=np.int64) for k in ks}, "reg_scores": {k: np.zeros((0, 0)) for k in ks}}
    a = SimpleNamespace(model="tiny", tiny=True, fit_seed=0, random_seed=1, register_margin=0.0)
    m1._save_structure(path, S, a, ops, {"git_hash": "test"})
    return path


def test_cli_tiny_all_variant_families(tmp_path):
    ops = _tiny_operators(tmp_path / "ops.npz")
    s = _tiny_m1_structure(tmp_path / "m1_structure.npz", ops)
    heal_cli = load_script("heal")
    for method in ("mirror", "svd", "random"):
        heal_cli.main(["--config", str(ROOT / "configs" / "m3_pilot.yaml"), "--tiny", "--method", method,
                       "--out-dir", str(tmp_path / "heal")])
    names = ["original", "mirror_mlp_k2", "svd_mlp_k2", "random_k2", "vlocal_k2", "vexact_k2", "svd_attn_k2",
             "random_attn_k2", "healed_mirror_k3", "healed_svd_k3", "healed_random_k3"]
    out = tmp_path / "t2"
    res = T2.main(["--tiny", "--variants", ",".join(names), "--out", str(out), "--operators", str(ops),
                   "--m1-structure", str(s), "--heal-dir", str(tmp_path / "heal"), *FAST])
    rows = {r["variant"]: r for r in res["rows"]}
    assert list(rows) == names
    import pandas as pd

    df = pd.read_csv(out / "t2_linear.csv")
    assert set(df["variant"]) == set(names) and df["git_hash"].nunique() == 1 and (df["seed"] == 0).all()
    assert (df["feature_dim"] == 128).all() and df["top1"].between(0, 100).all() and (df["top5"] >= df["top1"]).all()
    md = (out / "T2_LINEAR.md").read_text(encoding="utf-8")
    assert chr(0x2014) not in md and all(n in md for n in names)          # no em-dashes in reports
    # Matched budgets as in M1: SVD keeps no more than Mirror V-local; random keeps the same.
    assert rows["svd_attn_k2"]["converted_params_kept"] <= rows["vlocal_k2"]["converted_params_kept"]
    assert rows["random_attn_k2"]["converted_params_kept"] == rows["vlocal_k2"]["converted_params_kept"]
    assert rows["random_k2"]["converted_params_kept"] == rows["mirror_mlp_k2"]["converted_params_kept"]
    assert rows["original"]["model_params"] > rows["vlocal_k2"]["model_params"]
    # Healed students: constraints hold after loading, and the checkpoint really was loaded.
    for method in ("mirror", "random"):
        info = json.loads((out / "probe" / "tiny" / f"healed_{method}_k3.json").read_text())["variant_info"]
        checks = info["constraint_checks"]
        assert max(c["max_rel_violation"] for c in checks.values()) < 1e-5
    info = json.loads((out / "probe" / "tiny" / "healed_svd_k3.json").read_text())["variant_info"]
    for c in info["constraint_checks"].values():
        assert c["numerical_ranks"] == c["target_ranks"]
    ad = T2.build_adapter("tiny", True, "cpu")
    fac = T2.VariantFactory(ad, "tiny", True, ops, s, tmp_path / "heal")
    v = T2.parse_variant("healed_svd_k3")
    healed, _ = fac.build(v)
    cfg = json.loads((tmp_path / "heal" / "svd" / "summary.json").read_text())["config"]
    st = fac.heal().load_structure(cfg, ad)
    zero_shot, _ = fac.heal().method_setup("svd", ad, st, [0, 1, 2], cfg)[0]()
    x = normalize_for(ad, torch.from_numpy(synthetic_dataset(2, 2, 56, seed=3)[0]).permute(0, 3, 1, 2))
    with torch.no_grad():
        assert (T2.probe_feature(ad, x, healed) - T2.probe_feature(ad, x, zero_shot)).abs().max() > 1e-4
    # Missing inputs fail early with a clear message.
    with pytest.raises(FileNotFoundError, match="--heal-dir"):
        T2.VariantFactory(ad, "tiny", True, ops, s, None).sources(T2.parse_variant("healed_mirror_k3"))
    with pytest.raises(ValueError, match="--ks including 3"):
        T2.VariantFactory(ad, "tiny", True, ops, s, None).build(T2.parse_variant("vexact_k3"))


# ---------------------------------------------------------------- transfer probes (scripts/tier2_transfer.py)
def test_transfer_parquet_split_names():
    TR = load_script("tier2_transfer")
    files = ["a/cifar100/train-00000-of-00001.parquet", "a/cifar100/test-00000-of-00001.parquet",
             "b/data/validation-00000-of-00001.parquet", "b/data/test-00001-of-00006.parquet",
             "b/data/test-00000-of-00006.parquet", "c/train.parquet", "c/trainval-00000-of-00001.parquet"]
    assert TR.parquet_files_for_split(files, "train") == ["a/cifar100/train-00000-of-00001.parquet", "c/train.parquet"]
    assert TR.parquet_files_for_split(files, "test") == ["a/cifar100/test-00000-of-00001.parquet",
                                                         "b/data/test-00000-of-00006.parquet",
                                                         "b/data/test-00001-of-00006.parquet"]
    assert TR.parquet_files_for_split(files, "validation") == ["b/data/validation-00000-of-00001.parquet"]
    assert set(TR.DATASETS) == {"cifar100", "flowers102", "pets", "dtd"}
    assert {d["num_classes"] for d in TR.DATASETS.values()} == {100, 102, 37, 47}


def test_transfer_cli_tiny_decodes_caches_and_resumes(tmp_path, monkeypatch):
    TR = load_script("tier2_transfer")
    out = tmp_path / "t2"
    args = ["--tiny", "--variants", "original,mirror_mlp_k2", "--out", str(out), *FAST]
    res = TR.main(args)
    rows = res["rows"]
    assert [(r["dataset"], r["variant"]) for r in rows] == [("tiny_synth", "original"), ("tiny_synth", "mirror_mlp_k2")]
    assert all(r["n_train"] == 30 and r["n_test"] == 12 and r["num_classes"] == 3 for r in rows)
    meta = json.loads((out / "transfer" / "tiny_synth" / "meta.json").read_text())
    assert meta["complete"] and meta["counts"] == {"train": 30, "test": 12} and meta["crop"] == 56
    imgs = np.load(out / "transfer" / "tiny_synth" / "train_56_uint8.npy")
    assert imgs.shape == (30, 56, 56, 3) and imgs.dtype == np.uint8
    import pandas as pd

    df = pd.read_csv(out / "t2_transfer.csv")
    assert len(df) == 2 and df["git_hash"].nunique() == 1
    assert chr(0x2014) not in (out / "T2_TRANSFER.md").read_text(encoding="utf-8")
    # Rerun: no decoding, no feature extraction, probes skipped.
    monkeypatch.setattr(TR, "build_split_cache", lambda *a, **k: (_ for _ in ()).throw(AssertionError("decoded")))
    monkeypatch.setattr(TR.T2, "extract_features", lambda *a, **k: (_ for _ in ()).throw(AssertionError("extracted")))
    res2 = TR.main(args)
    assert [r["top1"] for r in res2["rows"]] == [r["top1"] for r in rows]


# ---------------------------------------------------------------- --healed-run (MUST-11): any healed model
def _fake_heal_structured(monkeypatch, make, calls):
    """Stand-in for mirror.heal_structured (written separately): load_healed_model(run_dir, device) ->
    (adapter, model, info) from `make()`."""
    import sys
    import types

    def load_healed_model(run_dir, device):
        calls.append(Path(run_dir))
        return make()

    fake = types.ModuleType("mirror.heal_structured")
    fake.load_healed_model = load_healed_model
    monkeypatch.setitem(sys.modules, "mirror.heal_structured", fake)


def _exact_tiny(seed=5, even_head=True):
    """(adapter, Mirror-exact tiny model with every block converted, structure) in float32."""
    from mirror.m1 import build_vexact

    ad = TimmTinyAdapter(img_size=56, seed=0)
    d, h, L = ad.embed_dim, ad.mlp_hidden, ad.depth
    H, dh = attn_dims(ad.model.blocks[0])
    A = random_involution_fixing_ones(d, d // 2 + 4, seed=seed)
    perms = [random_pairing(h, h // 3, seed=seed + 10 + b) for b in range(L)]
    maps = [random_head_structure(H, dh, 1, seed=seed + 20 + b) for b in range(L)]
    model, _ = build_vexact(ad, A, list(range(L)), perms, maps, even_head=even_head)
    return ad, model, A


def test_healed_run_variant_names():
    vs, dirs = T2.healed_run_variants(["runs/must09/vexact_s0", "svd one=runs/must09/svd_s0,x=runs/y"])
    assert [(v.name, v.family, v.method, v.k) for v in vs] == [
        ("healedrun_vexact_s0", "healed_run", "run", 0), ("svd_one", "healed_run", "run", 0), ("x", "healed_run", "run", 0)]
    assert dirs["svd_one"] == T2._resolve("runs/must09/svd_s0") and all(v.label for v in vs)
    with pytest.raises(ValueError, match="two runs named"):
        T2.healed_run_variants(["a/k12", "b/k12"])
    a = T2.parse_args(["--variants", "", "--healed-run", "runs/a"])
    assert [v.name for v in T2.variants_from_args(a)] == ["healedrun_a"]
    a = T2.parse_args(["--variants", "original"])
    assert [v.name for v in T2.variants_from_args(a)] == ["original"]          # unchanged without --healed-run
    with pytest.raises(ValueError, match="no variants given"):
        T2.variants_from_args(T2.parse_args(["--variants", ""]))
    with pytest.raises(ValueError, match="clash"):
        T2.variants_from_args(T2.parse_args(["--variants", "original", "--healed-run", "original=runs/a"]))
    # The info layout of mirror.heal_structured.load_healed_model: the mirror structure under info["structure"].
    hm = random_head_structure(4, 16, 1, seed=0)
    info = {"method": "mirror_exact", "spec": {"blocks": [0, 1]}, "constraints": [object()],
            "structure": {"A_star": np.eye(64), "reg_perm": np.array([1, 0]), "maps_after": [hm], "blocks": [0, 1, 2]}}
    assert T2._info_value(info, "blocks") == [0, 1, 2] and T2._info_value(info, "method") == "mirror_exact"
    assert T2._info_value(info, "k") is None and T2.healed_run_operator(info, None)[1] == "loader info['A_star']"
    s = T2.small_info(info)
    assert s["constraints"] == ["object"] and s["structure"]["maps_after"] == ["HeadMaps"]
    assert s["structure"]["A_star"] == "array of shape [64, 64]" and s["structure"]["reg_perm"] == [1, 0]
    json.dumps(s)


def test_cli_linear_and_transfer_with_a_healed_run(tmp_path, monkeypatch):
    calls = []
    _fake_heal_structured(monkeypatch, lambda: _exact_tiny()[:2] + ({"method": "mirror_exact", "seed": 0,
                                                                       "big": np.zeros(500)},), calls)
    run = tmp_path / "runs" / "vexact_s0"
    (run / "final").mkdir(parents=True)
    torch.save({"x": torch.zeros(3)}, run / "final" / "student.pt")
    out = tmp_path / "t2"
    res = T2.main(["--tiny", "--variants", "original", "--healed-run", f"vx={run}", "--out", str(out), *FAST])
    rows = {r["variant"]: r for r in res["rows"]}
    assert list(rows) == ["original", "vx"] and calls == [run]
    r = rows["vx"]
    assert r["family"] == "healed_run" and 0 <= r["top1"] <= 100
    assert "load_healed_model" in r["notes"] and "even head" in r["notes"] and "method mirror_exact" in r["notes"]
    vj = json.loads((out / "probe" / "tiny" / "vx_variant.json").read_text())
    assert vj["sources"]["weights_sha256"] == {"final/student.pt": T2.sha256(run / "final" / "student.pt")}
    assert vj["info"]["healed_info"]["big"] == "array of shape [500]" and vj["info"]["healed_k"] == 4
    assert vj["info"]["position_grid"] is None and vj["info"]["healed_run_checks"] is None      # 224 px: no check images
    assert "feature_view" not in r and vj["info"]["even_head"]                 # default: view a only, as before
    # The features of the exact model are mirror-invariant (the even head), so the probe sees the same thing.
    ad, model, _ = _exact_tiny()
    x = normalize_for(ad, torch.from_numpy(synthetic_dataset(2, 2, 56, seed=3)[0]).permute(0, 3, 1, 2))
    with torch.no_grad():
        f, ft = T2.probe_feature(ad, x, model), T2.probe_feature(ad, x.flip(-1), model)
        p, pt = T2.probe_feature_parts(ad, x, model), T2.probe_feature_parts(ad, x.flip(-1), model)
        p0 = T2.probe_feature_parts(ad, x, ad.model)
    assert float((f - ft).abs().max() / f.abs().max()) < 1e-5
    # Parts: [even CLS, even patch mean] is probe_feature (view a); the full patch mean (view b) is not invariant;
    # for a model without an even head the last part equals the second.
    assert torch.equal(p[:, :128], f) and float((pt[:, 128:] - p[:, 128:]).abs().max() / p[:, 128:].abs().max()) > 1e-3
    assert torch.equal(p0[:, 64:128], p0[:, 128:])
    # --full-patch-view: both views from one feature pass, view b under <variant>_fullpatch.
    out_v = tmp_path / "t2v"
    res = T2.main(["--tiny", "--variants", "original", "--healed-run", f"vx={run}", "--full-patch-view", "--out",
                   str(out_v), *FAST])
    rows = {r["variant"]: r for r in res["rows"]}
    assert list(rows) == ["original", "vx", "vx_fullpatch"]
    a_, b_ = rows["vx"], rows["vx_fullpatch"]
    assert a_["feature_flip_invariant"] is True and a_["patch_mean_tokens"] == "even-projected"
    assert b_["feature_flip_invariant"] is False and b_["patch_mean_tokens"] == "full" and b_["feature_dim"] == 128
    assert b_["notes"].startswith("feature view b") and "view b" in b_["label"] and "feature_view" not in rows["original"]
    assert a_["top1"] == r["top1"]                                              # view a is the default feature
    with np.load(out_v / "features" / "tiny" / "vx_bank.npz") as z:
        assert z["X"].shape[1] == 192 and json.loads(str(z["meta_json"]))["recipe"] == T2.FEATURE_RECIPE_PARTS
    md = (out_v / "T2_LINEAR.md").read_text(encoding="utf-8")
    assert "Mirror-exact feature views (--full-patch-view)" in md and chr(0x2014) not in md
    # Transfer: --variants may be empty when a healed run is given; default name healedrun_<folder>.
    TR = load_script("tier2_transfer")
    res = TR.main(["--tiny", "--variants", "", "--healed-run", str(run), "--out", str(tmp_path / "tr"), *FAST])
    assert [(r["dataset"], r["variant"]) for r in res["rows"]] == [("tiny_synth", "healedrun_vexact_s0")]
    assert "load_healed_model" in res["rows"][0]["notes"] and len(calls) == 3
    res = TR.main(["--tiny", "--variants", "", "--healed-run", str(run), "--full-patch-view", "--out",
                   str(tmp_path / "trv"), *FAST])
    assert [r["variant"] for r in res["rows"]] == ["healedrun_vexact_s0", "healedrun_vexact_s0_fullpatch"]
    assert [r["feature_flip_invariant"] for r in res["rows"]] == [True, False]
    # A missing run directory fails before any work.
    with pytest.raises(FileNotFoundError, match="healed run directory"):
        T2.main(["--tiny", "--variants", "", "--healed-run", str(tmp_path / "nope"), "--out", str(out), *FAST])
