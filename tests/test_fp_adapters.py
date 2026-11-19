"""Final-phase model adapters (mirror/final/adapters.py) and A15a (scripts/fp_a15a_siglip.py): tiny random DeiT-III,
DINOv2-L analogue (registers) and SigLIP 2 models built by timm with pretrained=False, float64 where exactness is
tested. Nothing here downloads weights or data."""
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MIRROR_FINAL_ALLOW_NO_PREREG", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from mirror import mirror_family as MF  # noqa: E402
from mirror.adapters import ViTAdapter  # noqa: E402
from mirror.attention import attn_dims, project_attn  # noqa: E402
from mirror.data import make_loader, mirror, synthetic_dataset  # noqa: E402
from mirror.final import adapters as FA  # noqa: E402
from mirror.final.audit.exactness import AuditStructure, build_even_odd  # noqa: E402
from mirror.m1 import (build_vexact, random_head_structure, random_involution_fixing_ones,  # noqa: E402
                       stream_equivariance)
from mirror.matching import random_pairing  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
S = 28                      # image size of the tiny models (patch 7: a 4 x 4 grid)


def imgs(n=3, seed=0, size=S):
    return torch.randn(n, 3, size, size, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


def loader(n_classes=3, per_class=4, size=S, seed=1, bs=6):
    images, labels = synthetic_dataset(n_classes, per_class, size, seed=seed)
    return make_loader(images, labels, np.arange(len(labels)), bs)


def random_structure(ad, seed=0, n_head_pairs=(1, 2)):
    d, h, L = ad.embed_dim, ad.mlp_hidden, ad.depth
    H, dh = attn_dims(ad.model.blocks[0])
    A = random_involution_fixing_ones(d, d // 2 + 3, seed=seed + 3)
    perms = [random_pairing(h, h // 3, seed=seed + 10 + b) for b in range(L)]
    maps = [random_head_structure(H, dh, n_head_pairs[b % len(n_head_pairs)], seed=seed + 20 + b) for b in range(L)]
    return A, perms, maps


def max_stream_err(ad, model, x, A, k=None, reg_perm=None, reg_start=1):
    rows = stream_equivariance(ad, model, x, A, reg_perm, reg_start)
    k = len(model.blocks) if k is None else k
    return max(r["rel_max"] for r in rows if r["layer"] <= k)


@pytest.fixture(scope="module")
def deit():
    return FA.tiny_deit3(img_size=S, patch=7, dim=32, depth=2, heads=4, num_classes=5)


# ---------------------------------------------------------------- DeiT-III
def test_deit3_layout_head_path_and_logits(deit):
    lay = FA.model_layout(deit)
    assert lay["registers"] == 0 and lay["no_embed_class"] and lay["class_token"] and lay["layerscale"]
    assert lay["n_tokens"] == 1 + 16 and lay["pos_embed_rows"] == 16 and lay["pos_prefix_rows"] == 0
    facts = FA.deit_head_path(deit.model)
    assert facts["fc_norm"] == "Identity" and facts["global_pool"] == "token" and facts["attn_pool"] is None
    x = imgs()
    assert torch.equal(FA.deit_logits(deit, x), deit.model(x))              # the model's own classifier path
    fa = FA.flip_averaged_logits(deit, x)
    assert torch.allclose(fa, 0.5 * (deit.model(x) + deit.model(mirror(x))), atol=1e-12)
    chk = FA.check_against_config(deit, {"architecture": {"blocks": 2, "width": 32, "heads": 4, "mlp_hidden": 128,
                                                          "registers": 0, "cls_token": 1, "patch_size": 7,
                                                          "layerscale": True}})
    assert chk["ok"], chk["checks"]
    real = FA.check_against_config(deit, ROOT / "configs" / "models" / "deit3_b16.yaml")
    assert not real["ok"] and {c["field"] for c in real["checks"] if not c["ok"]} >= {"blocks", "width"}


def test_deit3_full_conversion_even_classifier_exact(deit):
    A, perms, maps = random_structure(deit)
    L = deit.depth
    model, info = build_vexact(deit, A, range(L), perms, maps, even_head=False)
    facts = FA.install_even_classifier(model, A)
    assert facts["even_head"] == "installed"
    assert FA.install_even_classifier(model, A)["even_head"] == "already installed"
    with pytest.raises(ValueError):
        FA.install_even_classifier(model, random_involution_fixing_ones(deit.embed_dim, 10, seed=99))
    x = imgs()
    assert max_stream_err(deit, model, x, A) < 1e-10
    lg, lgm = FA.deit_logits(deit, x, model), FA.deit_logits(deit, mirror(x), model)
    assert float((lg - lgm).abs().max() / lg.abs().max()) < 1e-12
    r = FA.top1_eval(deit, model, loader(), return_preds=True)
    assert r["flip_consistency"] == 100.0 and r["n_disagree"] == 0 and r["pass_with_near_ties"]
    assert len(r["pred"]) == r["n"] == 12
    # the even/odd form with the same classifier gives the same logits
    st = AuditStructure(A=A, perms=perms, maps=info["maps_after"])
    eo = FA.EvenOddClassifier(build_even_odd(model, st, align=8), model.head)
    assert float((eo(x) - lg).abs().max() / lg.abs().max()) < 1e-10


def test_deit3_fitted_structure_conversion_budget_and_even_odd(deit):
    """The existing structure fit, conversion and budget code on DeiT-III (no registers, no_embed_class)."""
    images, labels = synthetic_dataset(3, 4, S, seed=2)
    ss = MF.fit_span_structure(deit, make_loader(images, labels, np.arange(8), 4), "cpu", model_name="tiny_deit3",
                               top_k=3, workers=1)
    L = deit.depth
    assert len(ss.reg_perm[L]) == 0
    model, info = MF.build_mirror_variant(deit, "mirror_exact", range(L), ("mlp", "attn"), ss)
    assert info["gap"] == 0 and info["full"] and info["build"]["even_head"]
    x = imgs()
    A = ss.A_star[L]
    assert max_stream_err(deit, model, x, A) < 1e-10
    st = AuditStructure(A=A, perms=info["structure"]["perms"], maps=info["structure"]["heads_after"])
    fv = build_even_odd(model, st, align=8)
    ref = deit.cls_feature(x, model)
    assert float((fv(x) - ref).abs().max() / ref.abs().max()) < 1e-10
    from mirror.efficient import fast_vit_cost
    c = fast_vit_cost(fv, deit.n_tokens, 16)
    assert c["matrix_params_converted"] == info["params_matrices"]       # stored form = the formula (A08-1)


# ---------------------------------------------------------------- DINOv2-L analogue (4 registers)
def test_dinov2_reg_analogue_registers_paired_exact():
    ad = FA.tiny_dinov2_reg(img_size=S, patch=7, dim=32, depth=2, heads=4, reg_tokens=4)
    assert FA.model_layout(ad)["registers"] == 4 and ad.n_prefix_tokens == 5
    A, perms, maps = random_structure(ad, seed=4)
    reg = np.array([1, 0, 2, 3])
    model, info = build_vexact(ad, A, range(ad.depth), perms, maps, reg_perm=reg, even_head=True)
    x = imgs(seed=5)
    assert max_stream_err(ad, model, x, A, reg_perm=reg) < 1e-10
    f, ft = ad.cls_feature(x, model), ad.cls_feature(mirror(x), model)
    assert float((f - ft).abs().max() / f.abs().max()) < 1e-12
    st = AuditStructure(A=A, perms=perms, maps=info["maps_after"], reg_perm=reg)
    fv = build_even_odd(model, st, align=8)
    assert float((fv(x) - f).abs().max() / f.abs().max()) < 1e-10
    from mirror.budget import mirror_exact_budget
    b = mirror_exact_budget(ad.model, A, list(range(ad.depth)), perms, maps)
    assert b.total > 0 and b.basis_change == 0


def test_dinov2_hub_layout_mock_interpolated_grid():
    """The DINOv2 hub layout (registers after the class token, no register positions, a stored position grid of
    another size interpolated with offset 0.1): the mock of tests/test_fold.py, converted exactly."""
    spec = importlib.util.spec_from_file_location("fold_mock_fp", ROOT / "tests" / "test_fold.py")
    mock = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mock
    spec.loader.exec_module(mock)
    m = mock.make_mock(num_register_tokens=4, img_size=84)               # stored 6 x 6 grid, used at 4 x 4

    class MockAdapter(ViTAdapter):
        def layer0_tokens(self, x, model=None):
            return self._m(model).prepare_tokens_with_masks(x)

    ad = MockAdapter(m, img_size=56, patch_size=14, n_prefix_tokens=5)
    A, perms, maps = random_structure(ad, seed=6)
    reg = np.array([0, 2, 1, 3])
    model, info = build_vexact(ad, A, range(ad.depth), perms, maps, reg_perm=reg, even_head=True)
    assert info["pos_materialised"]
    x = imgs(seed=7, size=56)
    assert max_stream_err(ad, model, x, A, reg_perm=reg) < 1e-10
    st = AuditStructure(A=A, perms=perms, maps=info["maps_after"], reg_perm=reg)
    fv = build_even_odd(model, st, align=8)
    f = ad.cls_feature(x, model)
    assert float((fv(x) - f).abs().max() / f.abs().max()) < 1e-10


# ---------------------------------------------------------------- SigLIP 2 and the whitened projection
def test_siglip2_feature_and_whitened_projection_exact():
    ad = FA.tiny_siglip2(img_size=S, patch=7, dim=32, depth=4, heads=4)
    facts = FA.siglip_feature_path(ad.model)
    assert not facts["class_token"] and facts["global_pool"] == "map" and facts["attn_pool"] == "AttentionPoolLatent"
    x = imgs()
    f = ad.cls_feature(x)
    assert torch.equal(f, ad.model.forward_head(ad.model.norm(ad.forward_tokens(x)), pre_logits=True))
    assert torch.allclose(f, ad.model(x), atol=1e-12)                    # num_classes 0: the head is Identity
    A, perms, maps = random_structure(ad, seed=8)
    k = 2
    mom = FA.collect_projection_moments(ad, loader(), "cpu", range(k))
    assert mom["info"]["n_images"] == 12 and set(mom["blocks"][0]) == {"qkv", "o", "fc1", "fc2"}
    from mirror.heal import MirrorMLPConstraint

    for comps in (("mlp",), ("mlp", "attn")):
        mw, iw = FA.build_whitened_vexact(ad, A, range(k), perms[:k], maps[:k], mom, components=comps)
        mp, _ = build_vexact(ad, A, range(k), perms[:k], maps[:k], components=comps)
        for m in (mw, mp):
            assert MirrorMLPConstraint(list(range(k)), [A] * k, [A] * k, perms[:k]).check(m)["max_rel_violation"] < 1e-12
            if comps == ("mlp", "attn"):          # MLP only leaves attention unprojected: not exact (as CORE-04 6mlp)
                assert max_stream_err(ad, m, x, A, k=k, reg_start=0) < 1e-10
        for r in iw["weighted"]:
            assert r["weighted_error"] <= r["weighted_error_plain"] + 1e-12, r
        assert mw(x).shape == mp(x).shape == f.shape
    assert {r["matrix"] for r in iw["weighted"]} >= {"patch_embed", "fc1", "fc2", "Wq", "Wk", "Wv", "Wo"}


def _equivariant_basis(R_out, R_in):
    Fp, Fm = FA.Involution(R_out).basis()
    Ep, Em = FA.Involution(R_in).basis()
    cols = [(Fa[:, i:i + 1] @ Ea[:, j:j + 1].T).reshape(-1) for Fa, Ea in ((Fp, Ep), (Fm, Em))
            for i in range(Fa.shape[1]) for j in range(Ea.shape[1])]
    return torch.stack(cols, 1)


@pytest.mark.parametrize("perm_out", [False, True])
def test_weighted_projection_is_the_constrained_minimiser(perm_out):
    g = torch.Generator().manual_seed(0)
    n_out, n_in = 6, 5
    R_out = random_pairing(n_out, 2, seed=1) if perm_out else random_involution_fixing_ones(n_out, 3, seed=1)
    R_in = random_involution_fixing_ones(n_in, 2, seed=2)
    W = torch.randn(n_out, n_in, dtype=torch.float64, generator=g)
    Z = torch.randn(50, n_in, dtype=torch.float64, generator=g) + 0.3
    M = Z.T @ Z / 50
    X, _, inf = FA.weighted_projection(W, R_out, R_in, M, damp=0)
    Ro, Ri = FA.Involution(R_out), FA.Involution(R_in)
    assert float((Ri.right(X) - Ro.left(X)).abs().max()) < 1e-13              # X R_in = R_out X
    Phi = _equivariant_basis(R_out, R_in)                                       # brute force: least squares
    L = torch.linalg.cholesky(M)
    K = torch.kron(torch.eye(n_out, dtype=torch.float64), L.T)
    Xb = (Phi @ torch.linalg.lstsq(K @ Phi, K @ W.reshape(-1)).solution).reshape(n_out, n_in)
    assert float((Xb - X).abs().max()) < 1e-12
    base = FA.weighted_error(W, X, M)
    for s in range(5):                                                          # equivariant moves only increase it
        D = (Phi @ torch.randn(Phi.shape[1], dtype=torch.float64, generator=g)).reshape(n_out, n_in)
        assert FA.weighted_error(W, X + 1e-3 * D, M) > base
    assert inf["weighted_error"] <= inf["weighted_error_plain"] + 1e-15
    plain = Ro.project_plain(W, Ri)
    Xi, _, _ = FA.weighted_projection(W, R_out, R_in, torch.eye(n_in, dtype=torch.float64), damp=0)
    assert float((Xi - plain).abs().max()) < 1e-13                              # M = I: the plain projection
    Ms = 0.5 * (M + Ri.left(Ri.right(M)))                                       # M R_in = R_in M: N+- = 0
    Xs, _, _ = FA.weighted_projection(W, R_out, R_in, Ms, damp=0)
    assert float((Xs - plain).abs().max()) < 1e-12


def test_weighted_projection_with_bias_is_joint_minimiser():
    g = torch.Generator().manual_seed(3)
    n_out, n_in = 5, 4
    R_out, R_in = random_involution_fixing_ones(n_out, 3, seed=4), random_pairing(n_in, 1, seed=5)
    W = torch.randn(n_out, n_in, dtype=torch.float64, generator=g)
    b = torch.randn(n_out, dtype=torch.float64, generator=g)
    Z = torch.randn(60, n_in, dtype=torch.float64, generator=g) + 0.5
    M, mu = Z.T @ Z / 60, Z.mean(0)
    X, c, _ = FA.weighted_projection(W, R_out, R_in, M, bias=b, mean=mu, damp=0)
    Ro, Ri = FA.Involution(R_out), FA.Involution(R_in)
    assert float((Ri.right(X) - Ro.left(X)).abs().max()) < 1e-13
    assert float((Ro.left(c[:, None])[:, 0] - c).abs().max()) < 1e-13
    Zt = torch.cat([Z, torch.ones(60, 1, dtype=torch.float64)], 1)            # brute force on [u; 1]
    Phi = _equivariant_basis(R_out, _aug_dense(R_in, n_in))
    Mt = Zt.T @ Zt / 60
    L = torch.linalg.cholesky(Mt)
    Wt = torch.cat([W, b[:, None]], 1)
    K = torch.kron(torch.eye(n_out, dtype=torch.float64), L.T)
    Xb = (Phi @ torch.linalg.lstsq(K @ Phi, K @ Wt.reshape(-1)).solution).reshape(n_out, n_in + 1)
    assert float((Xb[:, :-1] - X).abs().max()) < 1e-11 and float((Xb[:, -1] - c).abs().max()) < 1e-11
    X2, c2, _ = FA.weighted_projection(W, R_out, R_in, M, bias=b, mean=mu, damp=1e-6)  # damped: still equivariant
    assert float((Ri.right(X2) - Ro.left(X2)).abs().max()) < 1e-13


def _aug_dense(perm, n):
    P = np.zeros((n + 1, n + 1))
    P[np.arange(n), np.asarray(perm)] = 1.0
    P[n, n] = 1.0
    return P


def test_head_involution_and_patch_flip_conventions():
    H, dh, d = 4, 3, 6
    hm = random_head_structure(H, dh, 1, seed=11)
    R = FA.head_involution(hm, "B")
    assert np.allclose(R, R.T) and np.allclose(R @ R, np.eye(H * dh))
    A = random_involution_fixing_ones(d, 4, seed=12)
    g = torch.Generator().manual_seed(13)
    f = {k: torch.randn(H * dh, d, dtype=torch.float64, generator=g) for k in ("Wq", "Wk", "Wv")}
    f.update({k: torch.randn(H * dh, dtype=torch.float64, generator=g) for k in ("bq", "bk", "bv")})
    f["Wo"], f["bo"] = torch.randn(d, H * dh, dtype=torch.float64, generator=g), torch.randn(d, dtype=torch.float64,
                                                                                            generator=g)
    e = project_attn(**f, A_in=A, A_out=A, head_maps=hm, gauge=False)        # plain projection, fitted frame
    Rb, Rc, At = (torch.as_tensor(m) for m in (R, FA.head_involution(hm, "C"), A))
    assert float((e["Wq"] @ At - Rb @ e["Wq"]).abs().max()) < 1e-12          # W A* = R W
    assert float((e["Wv"] @ At - Rc @ e["Wv"]).abs().max()) < 1e-12
    assert float((e["Wo"] @ Rc - At @ e["Wo"]).abs().max()) < 1e-12          # Wo R_C = A* Wo
    W = torch.randn(5, 3, 4, 4, dtype=torch.float64, generator=g)
    p = FA.patch_flip_perm(3, 4)
    assert torch.equal(W.reshape(5, -1)[:, torch.as_tensor(p)], W.flip(-1).reshape(5, -1))


def test_moments_match_baseline_statistics():
    from mirror.baselines import collect_stats

    ad = FA.tiny_siglip2(img_size=S, patch=7, dim=32, depth=2, heads=4)
    ld = loader()
    mom = FA.collect_projection_moments(ad, ld, "cpu", [0, 1], patch=False)
    st = collect_stats(ad, ld, "cpu", [0, 1], methods=("svdllm",), components=("mlp", "attn"))
    for b in (0, 1):
        for mine, theirs in (("fc1", st.M1[b]), ("fc2", st.M2[b]), ("qkv", st.Nin[b]), ("o", st.Mctx[b])):
            M = mom["blocks"][b][mine]["M"]
            assert float((M - theirs).abs().max() / theirs.abs().max()) < 1e-10, (b, mine)


# ---------------------------------------------------------------- the A15a script
def test_a15a_script_dry_run(tmp_path):
    spec = importlib.util.spec_from_file_location("fp_a15a_test", ROOT / "scripts" / "fp_a15a_siglip.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "a15a"
    res = mod.main(["--dry-run", "--out", str(out)])
    names = {r["name"] for r in res["rows"]}
    assert names == {"original", "2mlp_plain", "2mlp_whitened", "2both_plain", "2both_whitened"}
    assert set(res["comparison"]) == {"2mlp", "2both"}
    both = {n: v for n, v in res["exactness"].items() if "both" in n}
    assert len(both) == 2 and all(v["max_rel_inside"] < 1e-4 for v in both.values())   # float32, MLP and attention
    for f in ("summary.csv", "weighted.csv", "metrics.json", "meta.json", "summary.md"):
        assert (out / f).exists(), f
    again = mod.main(["--dry-run", "--out", str(out)])                                # resume: nothing recomputed
    assert len(again["rows"]) == 5
