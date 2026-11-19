"""A09 exactness audit (mirror/final/audit/exactness.py, scripts/fp_a09_audit.py): CPU tests on tiny random ViTs
(registers paired; a DeiT-III with its classifier), float64 where exactness is tested (errors at most 1e-12)."""
import copy
import importlib.util
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MIRROR_FINAL_ALLOW_NO_PREREG", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from mirror.attention import attn_dims  # noqa: E402
from mirror.data import make_loader, synthetic_dataset  # noqa: E402
from mirror.final import adapters as FA  # noqa: E402
from mirror.final.audit import exactness as EX  # noqa: E402
from mirror.m1 import build_vexact, random_head_structure, random_involution_fixing_ones  # noqa: E402
from mirror.matching import random_pairing  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
S = 28
TOL = 1e-12
REG = np.array([1, 0])
DENSE_ROWS = {"patch_embed", "tokens0", "tokens0_prefix", "tokens0_patches", "norm1", "q", "k", "v", "attn_logits",
              "attn_probs", "head_out", "attn_proj", "ls1", "resid_attn", "norm2", "fc1", "act", "fc2", "ls2",
              "block_out", "final_rms", "even_proj", "final_affine", "cls_feature"}
EO_ROWS = {"patch_embed", "tokens0", "tokens0_prefix", "tokens0_patches", "norm1", "q", "k", "v", "attn_logits",
           "attn_probs", "head_out", "attn_proj", "resid_attn", "norm2", "fc1", "act", "fc2", "block_out",
           "final_rms_cls", "cls_feature"}


def imgs(n=2, seed=0):
    return torch.randn(n, 3, S, S, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


def exact_model(ad, seed=0, reg=None):
    d, h, L = ad.embed_dim, ad.mlp_hidden, ad.depth
    H, dh = attn_dims(ad.model.blocks[0])
    A = random_involution_fixing_ones(d, d // 2 + 3, seed=seed + 3)
    perms = [random_pairing(h, h // 3, seed=seed + 10 + b) for b in range(L)]
    maps = [random_head_structure(H, dh, (1, 2)[b % 2], seed=seed + 20 + b) for b in range(L)]
    model, info = build_vexact(ad, A, range(L), perms, maps, reg_perm=reg, even_head=True)
    return model, EX.AuditStructure(A=A, perms=perms, maps=info["maps_after"], reg_perm=reg)


@pytest.fixture(scope="module")
def reg64():
    ad = FA.tiny_dinov2_reg(img_size=S, patch=7, dim=32, depth=2, heads=4, reg_tokens=2)
    model, st = exact_model(ad, reg=REG)
    return ad, model, st


def rows_by(rows, form, precision):
    return {(r["block"], r["component"]): r for r in rows if r["form"] == form and r["precision"] == precision}


# ---------------------------------------------------------------- float64: both forms at rounding level
def test_float64_audit_both_forms_every_component(reg64):
    ad, model, st = reg64
    res = EX.audit_forms(ad, model, st, [imgs(2), imgs(1, seed=1)], forms=("dense", "even_odd"),
                         precisions=("float64",), align=8)
    for form, want in (("dense", DENSE_ROWS), ("even_odd", EO_ROWS)):
        got = rows_by(res["rows"], form, "float64")
        assert {c for _, c in got} == want, (form, want ^ {c for _, c in got})
        for (b, c), r in got.items():
            assert r["in_span"] and r["n_images"] == 3, (form, b, c)
            assert r["eq_rel_tensor_max"] < TOL and r["eq_rel_token_max"] < TOL, (form, b, c, r["eq_rel_tensor_max"])
    groups = {r["plan_component"] for r in res["rows"]}
    for g in EX.COMPONENTS:
        if EX.COMPONENTS[g]["rows"] and g != "DeiT classifier head":
            assert g in groups, g


def test_must02_agreement_dense_float64(reg64):
    """The dense float64 numbers agree with the MUST-02 audit on the same model and images."""
    spec = importlib.util.spec_from_file_location("must02_fp_test", ROOT / "scripts" / "must02_exactness.py")
    m02 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m02                       # its dataclass looks its module up
    spec.loader.exec_module(m02)
    ad, model, st = reg64
    x = [imgs(2, seed=3)]
    old = {(r["block"], r["component"]): r for r in m02.audit_model(
        ad, model, m02.ExactStructure(A=st.A, k=st.k, reg_perm=st.reg_perm, perms=st.perms, maps=st.maps), x, "cpu")}
    new = rows_by(EX.audit_forms(ad, model, st, x, forms=("dense",), precisions=("float64",))["rows"], "dense",
                  "float64")
    common = set(old) & set(new)
    assert len(common) >= 30
    for key in common:
        assert math.isclose(old[key]["rel_tensor_max"], new[key]["eq_rel_tensor_max"], rel_tol=1e-6, abs_tol=1e-18), key


# ---------------------------------------------------------------- lockstep floor and the scores
def test_floor_float32_and_scores(reg64):
    ad, model64, st = reg64
    model = copy.deepcopy(model64).float()                    # a model stored in float32, as on the VM
    res = EX.audit_forms(ad, model, st, [imgs(2, seed=4)], forms=("dense", "even_odd"),
                         precisions=("float64", "float64_exact", "float32"), original=ad.model, align=8)
    rows = res["rows"]
    for form in ("dense", "even_odd"):
        ex = rows_by(rows, form, "float64_exact")
        assert max(r["eq_rel_tensor_max"] for r in ex.values()) < TOL                       # E1
        assert max(r["dev_rel_tensor_max"] for r in ex.values()) > 1e-10                     # weights moved (fp32 rounding)
        f64 = rows_by(rows, form, "float64")
        assert max(r["eq_rel_tensor_max"] for r in f64.values()) > 1e-12                    # fp32 weight rounding shows
        assert all(math.isnan(r["dev_rel_tensor_max"]) for r in f64.values())               # the reference itself
        f32 = rows_by(rows, form, "float32")
        devs = [r["dev_rel_tensor_max"] for r in f32.values()]
        assert all(math.isfinite(v) for v in devs) and np.mean([v > 0 for v in devs]) > 0.8
        assert all(math.isfinite(r["dev_exact_rel_tensor_max"]) for r in f32.values())        # the second floor
        assert all(r["a09_2_pass_exact_floor"] in (True, False) for r in f32.values() if r["in_span"])
        assert all(r["a09_2_pass"] in (True, False) for r in f32.values() if r["in_span"])
    orig = rows_by(rows, "original", "float32")
    assert orig and all(math.isnan(r["eq_rel_tensor_max"]) for r in orig.values())
    assert all(math.isfinite(r["dev_rel_tensor_max"]) for r in orig.values())
    sc = EX.score_a09(rows)
    assert sc["A09-1"]["verdict"] == "PASS" and sc["A09-1"]["max_rel_error"] < TOL
    assert sc["A09-2"]["verdict"] in ("PENDING", "FAIL")              # no flip check given
    ok = {"dense/float32": {"pass_with_near_ties": True}, "even_odd/float32": {"pass_with_near_ties": True}}
    sc2 = EX.score_a09(rows, ok)
    assert sc2["A09-2"]["verdict"] == ("PASS" if sc2["A09-2"]["tensors_pass"] else "FAIL")
    bad = {**ok, "even_odd/float32": {"pass_with_near_ties": False}}
    assert EX.score_a09(rows, bad)["A09-2"]["verdict"] == "FAIL"
    sx = EX.score_a09(rows, ok, floor="exact")["A09-2"]
    assert sx["floor"] == "exact" and "tensors_pass_cast_floor" in sx
    assert sx["tensors_pass_cast_floor"] == sc2["A09-2"]["tensors_pass"]


def test_score_rules_on_synthetic_rows():
    def row(form, prec, eq, dev, dev_exact=None, comp="fc1"):
        dev_exact = dev if dev_exact is None else dev_exact
        return {"form": form, "precision": prec, "in_span": True, "space": "stream", "stage": "block", "block": 1,
                "component": comp, "eq_rel_tensor_max": eq, "dev_rel_tensor_max": dev,
                "ratio_eq_to_dev": eq / dev if dev else float("nan"),
                "a09_2_pass": (eq <= 2 * dev) if prec == "float32" else None,
                "dev_exact_rel_tensor_max": dev_exact,
                "ratio_eq_to_dev_exact": eq / dev_exact if dev_exact else float("nan"),
                "a09_2_pass_exact_floor": (eq <= 2 * dev_exact) if prec == "float32" else None}

    rows = [row("dense", "float64_exact", 3e-16, 1e-9), row("dense", "float32", 3e-7, 2e-7),
            row("dense", "bf16_autocast", 1e-2, 5e-3)]
    flip = {"dense/float32": {"pass": True}, "dense/bf16_autocast": {"flip_consistency": 99.5, "n_disagree": 3}}
    for floor in ("exact", "cast"):                      # "exact" is the pre-registered reading (the default)
        sc = EX.score_a09(rows, flip, floor=floor)
        assert sc["A09-1"]["verdict"] == "PASS" and sc["A09-2"]["verdict"] == "PASS"
        assert sc["A09-3"]["by_form"]["dense/bf16_autocast"]["flip_consistency"] == 99.5
    assert EX.score_a09(rows, flip)["A09-2"]["floor"] == "exact"
    rows[0]["eq_rel_tensor_max"] = 2e-10
    rows[1].update(eq_rel_tensor_max=5e-7, a09_2_pass=False, ratio_eq_to_dev=2.5, a09_2_pass_exact_floor=False,
                   ratio_eq_to_dev_exact=2.5)
    sc = EX.score_a09(rows, flip)
    assert sc["A09-1"]["verdict"] == "FAIL" and sc["A09-2"]["verdict"] == "FAIL"
    assert sc["A09-2"]["n_tensors_fail"] == 1
    # weight rounding counted in the exact floor only: the cast reading fails, the pre-registered reading passes
    rows[1].update(eq_rel_tensor_max=5e-7, dev_rel_tensor_max=2e-7, a09_2_pass=False, ratio_eq_to_dev=2.5,
                   dev_exact_rel_tensor_max=3e-7, a09_2_pass_exact_floor=True, ratio_eq_to_dev_exact=5e-7 / 3e-7)
    assert EX.score_a09(rows, flip)["A09-2"]["tensors_pass"] is True
    assert EX.score_a09(rows, flip, floor="cast")["A09-2"]["tensors_pass"] is False


# ---------------------------------------------------------------- localisation and weight conditions
def test_broken_weight_is_localised(reg64):
    ad, model, st = reg64
    broken = copy.deepcopy(model)
    with torch.no_grad():
        broken.blocks[1].mlp.fc2.weight.add_(0.05 * torch.randn_like(broken.blocks[1].mlp.fc2.weight))
    got = rows_by(EX.audit_forms(ad, broken, st, [imgs(2, seed=5)], forms=("dense",),
                                 precisions=("float64",))["rows"], "dense", "float64")
    assert max(r["eq_rel_tensor_max"] for (b, _), r in got.items() if b == 1) < TOL
    for c in ("norm1", "q", "attn_probs", "attn_proj", "norm2", "fc1", "act"):
        assert got[(2, c)]["eq_rel_tensor_max"] < TOL, c
    assert got[(2, "fc2")]["eq_rel_tensor_max"] > 1e-3 and got[(None, "cls_feature")]["eq_rel_tensor_max"] > 1e-6
    w = {r["component"]: r["rel_residual"] for r in EX.weight_conditions(ad, broken, st)}
    assert w["block 2 fc2.weight"] > 1e-3
    assert max(v for k, v in w.items() if k != "block 2 fc2.weight") < TOL


def test_weight_conditions_and_reprojection(reg64):
    ad, model, st = reg64
    w = EX.weight_conditions(ad, model, st)
    names = {r["component"] for r in w}
    assert {"patch_embed.weight", "patch_embed.bias", "pos_embed.grid", "pos_embed.cls", "pos_embed.registers",
            "cls_token", "registers", "even_head.P", "block 1 Wq", "block 2 fc1.weight", "block 1 head maps"} <= names
    assert max(r["rel_residual"] for r in w) < TOL
    m32 = copy.deepcopy(model).float()
    r32 = max(r["rel_residual"] for r in EX.weight_conditions(ad, m32, st))
    assert 1e-10 < r32 < 1e-5                                                  # float32 storage rounding
    ex = EX.reproject_exact(m32, ad, st)
    assert next(ex.parameters()).dtype == torch.float64
    assert max(r["rel_residual"] for r in EX.weight_conditions(ad, ex, st)) < 1e-13


# ---------------------------------------------------------------- even/odd internals and the bitwise check
def test_even_odd_parts_equal_module_forward(reg64):
    ad, model, st = reg64
    fv = EX.build_even_odd(copy.deepcopy(model).float(), st, align=8)
    t = fv.embed(imgs(2).float())
    blk = fv.blocks[0]
    x = blk.norm1(t)
    assert torch.equal(EX.mirror_attention_parts(blk.attn, x)["out"], blk.attn(x))
    y = blk.norm2(t + blk.attn(x))
    assert torch.equal(EX.mirror_mlp_parts(blk.mlp, y)["out"], blk.mlp(y))


@pytest.fixture
def one_thread():
    """Bitwise equality of grid-flipped token rows depends on how the CPU matrix products split their work (with 3 or
    more threads some rows take another kernel path); earlier tests may also change the thread count. Pin one thread
    for the bitwise test and restore it afterwards."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_bitwise_check_and_snapping(reg64, one_thread):
    ad, model, st = reg64
    fv = EX.build_even_odd(copy.deepcopy(model).float(), st, align=8)
    res = EX.bitwise_check(fv, imgs(2, seed=6), dp=st.d_plus, grid_hw=ad.grid_hw, n_prefix=ad.n_prefix_tokens,
                           reg_perm=REG, reg_start=1)
    assert {"patch_embed", "tokens0_prefix", "tokens0_patches", "block 1 attn", "block 2 out", "cls_feature"} <= set(
        res["stages"])
    for r in res["controlled_blocks"]:
        assert r["norm1"] == 1.0 and r["q"] == 1.0                      # a sign flip of the odd coordinates is exact
        assert r["attn_sign_only"] == 1.0 and r["mlp_sign_only"] == 1.0
        assert r["attn_matched_key_order"] >= r["attn_natural_key_order"]
    assert res["why"] and res["why"][0].startswith("first stage")
    assert res["conditions"]["torch_threads"] == 1 and res["conditions"]["device"] == "cpu"
    assert "snapped" in res and res["snapped"]["stages"]
    sn = EX.snap_embedding_exact(fv, st.d_plus, ad.grid_hw, REG)
    again = EX.bitwise_check(sn, imgs(2, seed=6), dp=st.d_plus, grid_hw=ad.grid_hw, n_prefix=ad.n_prefix_tokens,
                             reg_perm=REG, reg_start=1, controlled=False, snapped=False)
    assert all(v["all_identical"] for v in again["parameters"].values()), again["parameters"]
    assert again["stages"]["tokens0_prefix"]["all_identical"]          # class and registers exact once snapped


# ---------------------------------------------------------------- flip checks with a k-NN hook
def test_feature_flip_check_with_knn_hook(reg64):
    ad, model, st = reg64
    images, labels = synthetic_dataset(3, 5, S, seed=7)
    ld = make_loader(images, labels, np.arange(len(labels)), 5)
    from mirror.data import normalize

    def norm(x):
        return normalize(x, torch.float64, ad.mean, ad.std)

    feat = lambda x: ad.cls_feature(x, model)                                # noqa: E731
    bf, by = EX.compute_features(feat, ld, norm, "float64", "cpu")
    knn = EX.make_knn_fn(bf, by, 3, k=5)
    r = EX.feature_flip_check(feat, ld, norm, precision="float64", device="cpu", knn_fn=knn)
    assert r["flip_consistency"] == 100.0 and r["pass"] and r["max_rel_feature_diff_per_image"] < TOL
    assert len(r["pred"]) == len(r["pred_flip"]) == len(r["labels"]) == 15

    calls = {"n": 0}

    def tie_knn(q):                                                            # disagree once, on a near tie
        calls["n"] += 1
        pred = torch.zeros(q.shape[0], dtype=torch.long)
        if calls["n"] == 2:
            pred[0] = 1
        return pred, torch.full((q.shape[0],), 1e-6, dtype=torch.float64)

    r2 = EX.feature_flip_check(feat, ld, norm, precision="float64", device="cpu", knn_fn=tie_knn)
    assert r2["n_disagree"] == 1 and not r2["pass"] and r2["pass_with_near_ties"]


def test_deit_classifier_in_audit_and_flip():
    ad = FA.tiny_deit3(img_size=S, patch=7, dim=32, depth=2, heads=4, num_classes=5)
    model, st = exact_model(ad, seed=9)
    head = model.head
    res = EX.audit_forms(ad, model, st, [imgs(2, seed=8)], forms=("dense", "even_odd"), precisions=("float64",),
                         head=head, align=8)
    for form in ("dense", "even_odd"):
        lg = rows_by(res["rows"], form, "float64")[(None, "logits")]
        assert lg["in_span"] and lg["eq_rel_tensor_max"] < TOL
    images, labels = synthetic_dataset(3, 3, S, seed=9)
    ld = make_loader(images, labels, np.arange(len(labels)), 3)
    from mirror.data import normalize
    r = EX.feature_flip_check(lambda x: head(ad.cls_feature(x, model)), ld,
                              lambda x: normalize(x, torch.float64, ad.mean, ad.std), precision="float64",
                              classifier=True)
    assert r["flip_consistency"] == 100.0 and r["pass"]


# ---------------------------------------------------------------- the script
def test_a09_script_dry_run(tmp_path):
    spec = importlib.util.spec_from_file_location("fp_a09_test", ROOT / "scripts" / "fp_a09_audit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "a09"
    res = mod.main(["--dry-run", "--out", str(out)])
    assert res["scores"]["A09-1"]["verdict"] == "PASS"
    assert set(res["flip"]) == {"dense/float32", "dense/bf16_autocast", "even_odd/float32", "even_odd/bf16_autocast"}
    assert res["bitwise"]["float32"]["why"]
    for f in ("components.csv", "weights.csv", "flip.json", "bitwise.json", "metrics.json", "meta.json", "summary.md",
              "knn/dense_float32.npz"):
        assert (out / f).exists(), f
    again = mod.main(["--dry-run", "--out", str(out)])                       # resume: every step reused
    assert again["scores"]["A09-1"]["verdict"] == res["scores"]["A09-1"]["verdict"]
    assert math.isclose(again["scores"]["A09-1"]["max_rel_error"], res["scores"]["A09-1"]["max_rel_error"],
                        rel_tol=1e-9)                                       # read back from components.csv
