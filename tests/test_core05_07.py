"""CORE-05 (scripts/core05_why_cheap.py), CORE-06 (scripts/core06_orientation.py, mirror/cub.py) and CORE-07
(scripts/core07_shift.py): CPU tests with synthetic data and tiny random ViTs. The three end-to-end tiny runs take
about a minute each on a laptop CPU."""
import copy
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from mirror import cub as CUB
from mirror import variants as V
from mirror.adapters import TimmTinyAdapter
from mirror.matching import random_pairing

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_c0507_test", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


C5 = load_script("core05_why_cheap")
C6 = load_script("core06_orientation")
C7 = load_script("core07_shift")


def rand_psd(n, seed, rank=None):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(rank or 3 * n, n, generator=g, dtype=torch.float64)
    X = X * torch.linspace(3.0, 0.05, n, dtype=torch.float64)          # spread variances, so bands differ
    return X.T @ X / X.shape[0]


# ================================================================ CORE-05: pure measures
def test_band_split_sums_to_total():
    g = torch.Generator().manual_seed(0)
    W = torch.randn(12, 20, generator=g, dtype=torch.float64)
    W_eq = W + 0.3 * torch.randn(12, 20, generator=g, dtype=torch.float64)
    S = rand_psd(20, 1)
    lam, Vec = C5.sorted_eig(S)
    assert torch.all(lam[:-1] >= lam[1:])                                 # descending
    b = C5.band_split(W, W_eq, lam, Vec)
    tot = float(((W - W_eq) ** 2).sum())
    parts = sum(b[f"{k}_breaking_energy"] for k in C5.BANDS)
    assert abs(parts - tot) <= 1e-10 * tot
    assert abs(b["breaking_energy_total"] - tot) <= 1e-10 * tot
    assert abs(sum(b[f"{k}_share_of_breaking"] for k in C5.BANDS) - 1) < 1e-12
    assert abs(sum(b[f"{k}_variance_share"] for k in C5.BANDS) - 1) < 1e-12
    assert (b["top_n"], b["mid_n"], b["bottom_n"]) == (2, 16, 2)            # 10% / 80% / 10% of 20 directions
    assert b["top_variance_share"] > b["bottom_variance_share"]
    # the eigen-basis activation-weighted share equals the direct formula (PREREG Section 1)
    assert abs(b["s_cov_eig"] - V.weighted_share(W, W_eq, S)) < 1e-10


def test_matrix_measures_properties():
    g = torch.Generator().manual_seed(2)
    W = torch.randn(16, 10, generator=g, dtype=torch.float64)
    S = rand_psd(10, 3)
    M = S + torch.outer(torch.ones(10, dtype=torch.float64), torch.ones(10, dtype=torch.float64)) * 0.1
    eq = 60
    mm = C5.matrix_measures(W, W, S, M, eq)                                 # an exactly equivariant matrix
    assert mm["s_w"] == 0 and mm["s_cov"] == 0 and mm["s_mom"] == 0
    assert mm["svd_rank"] == 3 and mm["svd_params"] >= eq                  # ceil(60 / 26) = 3: SVD gets at least eq
    W_eq = W + 0.2 * torch.randn(16, 10, generator=g, dtype=torch.float64)
    mm = C5.matrix_measures(W, W_eq, S, M, eq)
    assert mm["E_cov_white"] <= mm["E_cov_plain"] + 1e-12                 # whitened truncation is optimal in that norm
    for k in ("erank_w", "erank_act", "erank_input"):
        assert 0 < mm[k] <= 1 + 1e-12
    assert C5.svd_rank(10_000, 50, 60) == 50                               # capped at full rank


def test_rule_hits_and_spearman():
    assert C5.rule_hits([1, -1, 0, 1, -2], [2, -3, 1, 0, 1]).tolist() == [True, True, False, False, False]
    assert abs(C5.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-12
    assert abs(C5.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12


def test_input_stats_layernorm_inputs():
    ad = TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float64)
    proto = V.TinyProtocol()
    st, n_img = C5.collect_stats(ad, proto.fit_loader(0), "cpu", [0, 2], attn=True)
    s = st.get("fc1", 0)
    assert s["n"] == n_img * ad.n_tokens
    ones = torch.ones(ad.embed_dim, dtype=torch.float64)
    # LayerNorm outputs sum to zero over channels, so the covariance annihilates the all-ones direction
    assert float(s["mean"].sum().abs()) < 1e-9 and float((s["Sigma"] @ ones).abs().max()) < 1e-9
    assert float(torch.linalg.eigvalsh(s["Sigma"]).min()) > -1e-9
    assert torch.allclose(s["M"] - torch.outer(s["mean"], s["mean"]), s["Sigma"])
    assert st.get("fc2", 2)["Sigma"].shape == (ad.mlp_hidden, ad.mlp_hidden)
    assert st.get("proj", 2)["n"] == n_img * ad.n_tokens


def test_residual_evaluator_equals_full_forward():
    ad = TimmTinyAdapter(img_size=56, seed=1, randomise=True, dtype=torch.float64)
    L, d, h = ad.depth, ad.embed_dim, ad.mlp_hidden
    A = V.random_involution_fixing_ones(d, 40, seed=3)
    struct = SimpleNamespace(A_star={L: A}, perms=[random_pairing(h, 90, seed=b) for b in range(L)])
    var = C5.single_block_variants(ad, struct, [1, 3])
    assert set(var) == {"b2_mirror", "b2_svd", "b4_mirror", "b4_svd"}
    assert var["b2_svd"][2]["params"] >= var["b2_mirror"][2]["params"]   # rounding gives the baseline more
    x = torch.randn(3, 3, 56, 56, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    out = C5.ResidualEvaluator(ad, var)(x)
    ref = F.normalize(ad.cls_feature(x).float(), dim=-1)
    assert torch.equal(out["original"], ref)
    for name, (b, mod, _) in var.items():
        m2 = copy.deepcopy(ad.model)
        m2.blocks[b] = copy.deepcopy(mod)
        full = F.normalize(ad.cls_feature(x, m2).float(), dim=-1)
        assert torch.allclose(out[name], full, atol=1e-6, rtol=0), name
        assert not torch.allclose(out[name], ref, atol=1e-6)               # the surgery changed something


# ================================================================ CORE-05: the frozen rule comes first
def _tables(model, depth, seed):
    rng = np.random.default_rng(seed)
    mrows, trows = [], [{"model": model, "block": 0, "method": "original", "drop_val": 0.0, "drop_msel": 0.0,
                         "distortion_val": 0.0}]
    for b in range(1, depth + 1):
        for mat in ("fc1", "fc2"):
            r = {"model": model, "op": "star", "block": b, "matrix": mat}
            r.update({c: float(rng.uniform(0.01, 0.3)) for c in ("s_cov", "s_mom", "E_cov_plain", "E_cov_white",
                                                                 "E_mom_plain", "E_mom_white")})
            mrows.append(r)
            mrows.append({**r, "op": "perlayer"})
        for mth in ("mirror", "svd"):
            trows.append({"model": model, "block": b, "method": mth, "drop_val": float(rng.uniform(0, 1)),
                          "drop_msel": float(rng.uniform(0, 1)), "distortion_val": float(rng.uniform(0, 1))})
    return pd.DataFrame(mrows), pd.DataFrame(trows)


def test_rule_is_frozen_before_scoring(tmp_path):
    a = C5.parse_args(["--tiny", "--out", str(tmp_path)])
    git = {"git_hash": "test", "git_dirty": False}
    with pytest.raises(SystemExit):                                         # no frozen rule: scoring refuses
        C5.stage_score(a, tmp_path, git)
    m, t = _tables("tiny", 4, 0)
    m.to_csv(tmp_path / "measures_tiny.csv", index=False)
    t.to_csv(tmp_path / "truth_tiny.csv", index=False)
    rec = C5.stage_freeze(a, tmp_path, git)                                 # only the design model's files exist
    frozen = tmp_path / "rule_frozen.json"
    assert frozen.exists() and set(rec["design_files_sha256"]) == {"measures_tiny.csv", "truth_tiny.csv"}
    assert rec["rule"]["id"] in {c["id"] for c in C5.RULE_CANDIDATES} and rec["design_complete"]
    before = frozen.read_bytes()
    m2, t2 = _tables("tiny_s1", 4, 1)                                       # scoring data appear only now
    m2.to_csv(tmp_path / "measures_tiny_s1.csv", index=False)
    t2.to_csv(tmp_path / "truth_tiny_s1.csv", index=False)
    C5.stage_freeze(a, tmp_path, git)                                       # a second freeze does not rewrite it
    assert frozen.read_bytes() == before
    res = C5.stage_score(a, tmp_path, git)
    assert res["rule_frozen_sha256"] == hashlib.sha256(before).hexdigest()
    assert res["complete"] and res["primary"]["n_cases"] == 4 and res["verdict"] in ("PASS", "FAIL")
    assert frozen.read_bytes() == before                                    # scoring never touches the rule
    # the frozen rule decides by itself: recompute its hits by hand
    rule = rec["rule"]
    mm = m2[(m2.op == "star")].groupby("block")[[rule["s_col"], rule["E_col"]]].mean()
    tt = t2[t2.method != "original"].pivot(index="block", columns="method", values="drop_val")
    x = mm[rule["E_col"]] - mm[rule["s_col"]]
    y = tt["svd"] - tt["mirror"]
    assert res["primary"]["n_hits"] == int(C5.rule_hits(x.values, y.loc[x.index].values).sum())
    # editing the frozen rule is detected
    j = json.loads(before)
    j["rule"]["E_col"] = "E_w"
    frozen.write_text(json.dumps(j), encoding="utf-8")
    with pytest.raises(SystemExit):
        C5.stage_score(a, tmp_path, git)


def test_core05_tiny_all(tmp_path):
    res = C5.main(["--tiny", "--stage", "all", "--out", str(tmp_path)])
    m = res["metrics"]
    for k in ("P5.1", "P5.2", "P5.3"):
        assert m[k]["verdict"] != "n/a", k
    for f in ("measures_tiny.csv", "measures_tiny_s1.csv", "truth_tiny.csv", "truth_tiny_s1.csv", "rule_frozen.json",
              "rule_score.json", "ckpt_bands.csv", "metrics.json", "SUMMARY.md", "meta.json"):
        assert (tmp_path / f).exists(), f
    meas = pd.read_csv(tmp_path / "measures_tiny.csv")
    star = meas[meas.op == "star"]
    tot = star[[f"{b}_breaking_energy" for b in C5.BANDS]].sum(1)
    assert np.allclose(tot, star["breaking_energy_total"], rtol=1e-10)
    score = json.loads((tmp_path / "rule_score.json").read_text())
    frozen = json.loads((tmp_path / "rule_frozen.json").read_text())
    assert score["rule_frozen_utc"] == frozen["frozen_utc"] <= score["scored_utc"]
    assert set(frozen["design_files_sha256"]) == {"measures_tiny.csv", "truth_tiny.csv"}
    ck = pd.read_csv(tmp_path / "ckpt_bands.csv")
    assert set(ck["group"]) == {"LR", "UD"} and set(ck["epoch"]) == {0, 1}
    truth = pd.read_csv(tmp_path / "truth_tiny.csv")
    assert len(truth) == 1 + 2 * 4 and (truth[truth.method == "svd"]["params"].values >=
                                        truth[truth.method == "svd"]["mirror_params"].values).all()


# ================================================================ CORE-06: CUB facing labels
def test_facing_sign_filters():
    w = np.full(7, 100.0)
    parts = np.zeros((7, 15, 3))
    #            beak x, beak vis, tail x, tail vis
    cases = [(80, 1, 20, 1),     # faces right
             (20, 1, 80, 1),     # faces left
             (0, 0, 80, 1),      # beak not visible (x = 0 written for invisible parts)
             (80, 1, 0, 0),      # tail not visible
             (60, 1, 50, 1),     # |dx| = 10 = 10% of the box width: not more than 10%, no label
             (61, 1, 50, 1),     # 11% apart: faces right
             (50, 1, 61, 1)]     # 11% apart: faces left
    for i, (xb, vb, xt, vt) in enumerate(cases):
        parts[i, CUB.BEAK_ID - 1] = (xb, 10, vb)
        parts[i, CUB.TAIL_ID - 1] = (xt, 10, vt)
    s = CUB.facing_sign(parts, w)
    assert s.tolist() == [1, -1, 0, 0, 0, 1, -1]
    assert CUB.facing_class(s).tolist() == [1, 0, -1, -1, -1, 1, 0]
    # a horizontal mirror flips the sign and keeps the 10% test
    m = parts.copy()
    for p in (CUB.BEAK_ID - 1, CUB.TAIL_ID - 1):
        m[:, p, 0] = np.where(m[:, p, 2] == 1, CUB.mirror_x(m[:, p, 0], 120), 0)
    assert CUB.facing_sign(m, w).tolist() == [-1, 1, 0, 0, 0, -1, 1]


def test_synthetic_cub_annotations(tmp_path):
    root = CUB.write_synthetic_cub(tmp_path / "CUB_200_2011", n_classes=3, per_class=6, size=40)
    ann = CUB.load_annotations(root)
    assert len(ann["name"]) == 18 and ann["parts"].shape == (18, 15, 3) and ann["label"].max() == 2
    assert ann["is_train"].sum() == 9
    s = CUB.facing_sign(ann["parts"], ann["bbox"][:, 2])
    kind = ann["image_id"] % 3                                             # the generator's design
    assert (s[kind == 0] == 1).all() and (s[kind == 1] == -1).all() and (s[kind == 2] == 0).all()
    assert CUB.check_part_names(root)["ok"]


def test_mirror_augmented_flips_facing():
    ma = CUB.mirror_augmented(np.array([5, 7, 9]), np.array([1, 0, -1]), species=np.array([3, 4, 6]))
    assert ma["row"].tolist() == [5, 9, 5, 9]
    assert ma["mirrored"].tolist() == [False, False, True, True]
    assert ma["facing_sign"].tolist() == [1, -1, -1, 1]
    assert ma["facing"].tolist() == [1, 0, 0, 1]
    assert ma["species"].tolist() == [3, 6, 3, 6]
    assert ma["pair"].tolist() == [0, 1, 0, 1]


def test_invariant_feature_gives_exactly_half_on_mirror_augmented():
    rng = np.random.default_rng(0)
    n, d = 200, 16
    y = rng.integers(0, 2, n)
    X_tr = rng.standard_normal((n, d)).astype(np.float32) + 1.5 * y[:, None].astype(np.float32)
    probe = C6.fit_final(X_tr, y, 2, 1e-3, "cpu", {"max_iter": 200})
    X_te = rng.standard_normal((80, d)).astype(np.float32)
    y_te = rng.integers(0, 2, 80)
    X_te += 1.5 * y_te[:, None]
    lg = C6.logits_of(probe, X_te)
    nat = C6.score_natural(lg, y_te)
    assert nat["acc"] > 70                                                # the feature carries the label
    # an exactly invariant feature: the mirror's feature equals the image's, the label is flipped
    ma = C6.score_mirror_augmented(lg, C6.logits_of(probe, X_te.copy()), y_te, 1 - y_te)
    assert ma["acc"] == 50.0 and ma["pairs_same_prediction"] == 100.0 and ma["n_pairs_differ"] == 0
    assert ma["n_pairs_differ_not_near_tie"] == 0
    # an exactly sign-flipping (odd) feature with a bias-free probe keeps its natural accuracy
    probe0 = dict(probe, b=torch.zeros_like(probe["b"]), mean=torch.zeros_like(probe["mean"]))
    l0, l1 = C6.logits_of(probe0, X_te), C6.logits_of(probe0, -X_te)
    ma2 = C6.score_mirror_augmented(l0, l1, y_te, 1 - y_te)
    assert ma2["acc"] == C6.score_natural(l0, y_te)["acc"] and ma2["pairs_same_prediction"] == 0.0


def test_stratified_folds_and_cv_on_train_only():
    y = np.repeat(np.arange(5), 7)
    f = C6.stratified_folds(y, 5, seed=0)
    assert set(f.tolist()) == set(range(5))
    for c in range(5):
        assert len(set(f[y == c].tolist())) == 5                          # each class spread over the folds
    rng = np.random.default_rng(1)
    X = rng.standard_normal((35, 6)).astype(np.float32) + y[:, None]
    cv = C6.cv_select_wd(X, y, 5, [1e-3, 1e-1], 5, 0, "cpu", {"max_iter": 100})
    assert cv["chosen_wd"] in (1e-3, 1e-1) and len(cv["trials"]) == 2 and all(t["n"] == 35 for t in cv["trials"])


def test_final_parts_layernorm_convention():
    ad = TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float32)
    A = V.random_involution_fixing_ones(ad.embed_dim, 40, seed=0)
    parts = C6.layernorm_parts(ad.model, A, "test")
    x = torch.randn(2, 3, 56, 56, generator=torch.Generator().manual_seed(0))
    t = ad.forward_tokens(x)
    f = parts(t, ad.n_prefix_tokens)
    own = ad.model.norm(t)
    assert torch.allclose(f[:, 0] + f[:, 1], own[:, 0], atol=1e-5)        # full = even + odd = the model's feature
    assert torch.allclose(f[:, 2] + f[:, 3], own[:, ad.n_prefix_tokens:].mean(1), atol=1e-5)
    assert parts.d_minus == ad.embed_dim - 40


def test_core06_tiny_pipeline(tmp_path):
    res = C6.main(["--tiny", "--out", str(tmp_path)])
    sc = res["metrics"]["scoring"]
    p63 = sc["P6.3"]["per_model"]["exact_tiny"]
    assert p63["mirror_augmented"] == 50.0 and p63["n_pairs_differ_not_near_tie"] == 0 and p63["pass_exact50"]
    inv = res["metrics"]["invariance"]["exact_tiny"]
    assert inv["cls_even_rel_max"] < 1e-4 and inv["cls_odd_rel_max"] < 1e-4    # fp32: exact up to rounding
    df = pd.read_csv(tmp_path / "probes.csv")
    assert set(df["variant"]) == {"original", "exact_tiny"} and set(df["task"]) == {"species", "facing"}
    assert (tmp_path / "features" / "exact_tiny_natural.npz").exists() and (tmp_path / "SUMMARY.md").exists()
    ff = df[(df.variant == "exact_tiny") & (df.task == "facing") & (df.features == "even_patch")
            & (df.test_set == "mirror_augmented")]
    assert float(ff["acc"].iloc[0]) == 50.0                                # the even patch mean is invariant too


# ================================================================ CORE-07: shifted sets
def _png(path, seed, size=20):
    from PIL import Image

    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)).save(path, format="PNG")


def test_v2_labels_are_int_folder_names(tmp_path):
    for k, name in enumerate(["0", "1", "2", "10"]):
        (tmp_path / name).mkdir()
        _png(tmp_path / name / f"img{k}.jpeg", k)
    (tmp_path / "10" / "readme.txt").write_text("not an image")
    src, lab = C7.v2_index(tmp_path)
    assert lab.tolist() == [0, 1, 2, 10]                                   # ImageFolder would give 0, 1, 3, 2
    assert Path(src[-1][1]).parent.name == "10"
    (tmp_path / "train").mkdir()
    with pytest.raises(ValueError):
        C7.v2_index(tmp_path)


def test_sketch_labels_by_wnid_rank(tmp_path):
    zp = tmp_path / "s.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        for w in ("n00000003", "n00000001"):
            zf.writestr(f"sketch/{w}/{w}_0.JPEG", b"x")
            zf.writestr(f"sketch/{w}/{w}_1.png", b"x")                      # not .JPEG: ignored, as the loader does
    src, lab = C7.sketch_index(zp, ["n00000001", "n00000002", "n00000003"])
    assert lab.tolist() == [0, 2] and all(s[2].endswith(".JPEG") for s in src)
    with pytest.raises(ValueError):
        C7.sketch_index(zp, ["n00000001"])


def test_drop_bootstrap():
    rng = np.random.default_rng(0)
    cv, cs = rng.integers(0, 2, 300), rng.integers(0, 2, 100)
    same = C7.drop_bootstrap(cv, cs, cv, cs, n_resamples=200)
    assert same["diff"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    cv2, cs2 = rng.integers(0, 2, 300), rng.integers(0, 2, 100)
    r = C7.drop_bootstrap(cv2, cs2, cv, cs, n_resamples=500)
    manual = 100 * ((cv2.mean() - cs2.mean()) - (cv.mean() - cs.mean()))
    assert abs(r["diff"] - manual) < 1e-9 and r["ci_low"] <= r["diff"] <= r["ci_high"]


def test_core07_tiny_runs_on_synthetic_folders(tmp_path):
    res = C7.main(["--tiny", "--out", str(tmp_path)])
    rows = {r["variant"]: r for r in res["rows"]}
    assert set(rows) == {"original", "mirror_mlp_k2"}
    for r in rows.values():
        assert r["n_v2"] == 12 and r["n_sketch"] == 12 and np.isfinite(r["v2_drop"])
    lab = np.load(tmp_path / "shift_cache" / "imagenetv2_mf_labels.npy")
    assert lab.tolist() == sorted(np.repeat(np.arange(4), 3).tolist())     # int(folder name) labels
    nd = res["metrics"]["sketch_near_duplicates"]
    assert nd["n_pairs"] == 4 and nd["by_transform"]["flip_lr"] == 4       # one flipped copy per class
    p = res["metrics"]["P7.1"]
    assert p["verdict"] in ("PASS", "FAIL") and p["n_seeds"] == 1 and not p["complete"]
    assert (tmp_path / "shift_rows.csv").exists() and (tmp_path / "SUMMARY.md").exists()
