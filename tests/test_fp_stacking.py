"""Final phase, stacking agent: mirror/final/baselines/sizes.py, mirror_plus_svd.py and partial_span.py.

CPU tests with a tiny random timm ViT (embed 64, depth 4, 4 heads, 56 px; classifier removed so, as DINOv2, every
parameter belongs to the backbone) in float64. The fitted tiny structure is made DINOv2-like for the size tests: a
random span operator with d+ = 56 of 64 (A 1 = 1), 50 twin pairs per block and one twin head pair per block (random
head maps; surgery gauges the pair to B = C = I), so Mirror-exact keeps about 0.75 of the model and the A06 fractions
0.65, 0.50 and 0.40 lie below it, as on DINOv2-B. Exactness holds for any valid structure."""
import copy
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from mirror import baselines as B
from mirror import budget as BU
from mirror import mirror_family as MF
from mirror import stacking as ST
from mirror.adapters import TimmTinyAdapter
from mirror.attention import HeadMaps
from mirror.data import make_loader, mirror, synthetic_dataset
from mirror.final.baselines import mirror_plus_svd as MS
from mirror.final.baselines import partial_span as PS
from mirror.final.baselines import sizes as SZ
from mirror.heal import configure_trainable
from mirror.heal_vexact import span_exactness
from mirror.m1 import feature_invariance, random_head_structure, random_involution_fixing_ones, stream_equivariance

ROOT = Path(__file__).resolve().parents[1]
FRACS = (0.65, 0.50, 0.40)
DINO_B = {"d": 768, "h": 3072, "H": 12, "dh": 64, "depth": 12, "full_params": SZ.DINOV2_B_REG_PARAMS}


# ---------------------------------------------------------------- fixtures
def _thin(perm, n_keep: int) -> np.ndarray:
    """The first n_keep twin pairs of an involution (by smaller index); the other neurons become self-twins."""
    p = np.asarray(perm)
    out = np.arange(len(p))
    kept = 0
    for i in range(len(p)):
        j = int(p[i])
        if j > i and kept < n_keep:
            out[i], out[j] = j, i
            kept += 1
    return out


@pytest.fixture(scope="module")
def ad():
    a = TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float64)
    a.model.reset_classifier(0)
    return a


@pytest.fixture(scope="module")
def fit_loader():
    images, labels = synthetic_dataset(4, 12, 56, seed=3)
    return make_loader(images, labels, np.arange(16), 8)


@pytest.fixture(scope="module")
def struct(ad, fit_loader):
    st = MF.fit_span_structure(ad, fit_loader, "cpu", model_name="tiny", top_k=3, workers=1, pairings=False)
    L, d = ad.depth, ad.embed_dim
    A_star = {k: random_involution_fixing_ones(d, 56, seed=100 + k) for k in range(1, L + 1)}
    perms = [_thin(p, 50) for p in st.perms]
    heads = [random_head_structure(4, 16, 1, seed=200 + b, self_traces=[(12, 10), (10, 12)]) for b in range(L)]
    return dataclasses.replace(st, A_star=A_star, perms=perms, heads=heads)


@pytest.fixture(scope="module")
def stats(ad, fit_loader):
    return B.collect_stats(ad, fit_loader, "cpu", list(range(ad.depth)), ["svdllm"], MF.COMPONENTS)


@pytest.fixture(scope="module")
def mirror_exact(ad, struct):
    return MF.build_mirror_variant(ad, "mirror_exact", range(ad.depth), MF.COMPONENTS, struct)


@pytest.fixture(scope="module")
def built(ad, struct, stats):
    return {f: MS.build(ad, struct, f, stats) for f in FRACS}


def probe(n=3, seed=5):
    return torch.randn(n, 3, 56, 56, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def exactness(ad, model, A, reg_perm, k=None):
    xs = [probe(2, 5), probe(2, 6)]
    ex = span_exactness(ad, model, xs, A, ad.depth if k is None else k, reg_perm, tol=1e-10)
    inv = max(feature_invariance(ad, model, x)["rel_max"] for x in xs)
    return ex, inv


# ---------------------------------------------------------------- sizes
def test_full_model_params_and_targets(ad):
    n = sum(p.numel() for p in ad.model.parameters())
    assert SZ.full_model_params(ad) == SZ.full_model_params(ad.model) == n == 239_424
    from mirror.fold import fold_copy

    with pytest.raises(ValueError):
        SZ.full_model_params(fold_copy(ad.model, final_norm_affine=False)[0])     # a converted copy is refused
    # DINOv2-B with registers (counts from notes/paper/method_spec.md Section 5)
    assert SZ.full_model_params(DINO_B) == 86_583_552
    assert SZ.converted_dense_params(DINO_B) == 84_934_656
    assert SZ.target_params(DINO_B, 0.65) == 56_279_309                    # ceil(56,279,308.8)
    assert SZ.target_params(DINO_B, 0.50) == 43_291_776                    # exact, not rounded up by float noise
    assert SZ.target_params(ad, 0.40) == math.ceil(0.4 * 239_424)


def test_budget_for_fraction_at_dinov2_scale():
    """At DINOv2-B sizes the SVD-LLM allocation of mirror/budget.py lands within [0, +1%] of every A06 point."""
    outside = 86_583_552 - 84_934_656
    for f in FRACS:
        bud = SZ.budget_for_fraction(DINO_B, f)
        assert bud.target_model == SZ.target_params(DINO_B, f) and bud.total == bud.target_model - outside
        assert bud.matrix_total == bud.total and bud.basis_change == 0 and len(bud.matrices) == 72
        t = BU.matrix_targets(bud)
        assert all(t[(m.block, m.name)] == m.mirror_exact for m in bud.matrices)   # shares are the targets
        ratios = {round(m.mirror_exact / m.dense, 4) for m in bud.matrices}
        assert max(ratios) - min(ratios) < 1e-3                                     # one compression ratio
        alloc = BU.allocate("svdllm", bud)
        tot = BU.allocation_totals(alloc)
        gap = BU.gap_of(tot["total"], bud.total)
        assert 0 <= gap <= 0.01
        size = SZ.student_size(bud, tot["matrices"], tot["basis_change"])
        assert size["eligible"] and size["params_model"] >= size["target"] and 0 <= size["gap"] <= gap
        assert abs(size["params_frac"] - f) < 1e-4
    # proportions from a reference budget (the MUST-07 rule when it is Mirror-exact's)
    ref = SZ.budget_for_fraction(DINO_B, 0.65)
    ref = dataclasses.replace(ref, matrices=[dataclasses.replace(m, mirror_exact=m.dense * (2 if m.name == "fc1"
                                                                                           else 1))
                                             for m in ref.matrices])
    bud = SZ.budget_for_fraction(DINO_B, 0.50, proportions=ref)
    fc1 = next(m for m in bud.matrices if m.name == "fc1")
    fc2 = next(m for m in bud.matrices if m.name == "fc2")
    assert abs(fc1.mirror_exact / fc2.mirror_exact - 2.0) < 1e-4 and bud.matrix_total == bud.total
    with pytest.raises(ValueError):
        SZ.budget_for_fraction(DINO_B, 0.01)                               # below what lies outside the matrices


def test_budget_for_fraction_gives_eligible_svdllm_students(ad, stats):
    for f in FRACS:
        bud = SZ.budget_for_fraction(ad, f)
        m, inf = B.compress(ad.model, "svdllm", bud.blocks, bud.components, bud, stats, verbose=False)
        assert inf["gap_ok"] and 0 <= inf["gap"] <= 0.01 and inf["budget_total"] == bud.total
        size = SZ.student_size(bud, inf["params_matrices"], inf["params_basis_change"])
        assert size["eligible"] and size["target"] <= size["params_model"] <= 1.01 * size["target"]
        chk = B.structure_check(m, inf)
        assert chk["ok"] and chk["params_realised"] == inf["params_matrices"]


def test_a06_prediction_on_hand_made_numbers():
    svd = {0.40: 70.0, 0.50: 74.0, 0.65: 76.5, 0.765: 78.0}
    orig, mir = 79.0, 78.3
    delta = orig - mir

    def lin(s, s0, a0, s1, a1):
        return a0 + (math.log(s) - math.log(s0)) / (math.log(s1) - math.log(s0)) * (a1 - a0)

    want = {0.65: lin(0.65 / 0.765, 0.765, 78.0, 1.0, 79.0) - delta,
            0.50: lin(0.50 / 0.765, 0.65, 76.5, 0.765, 78.0) - delta,
            0.40: lin(0.40 / 0.765, 0.50, 74.0, 0.65, 76.5) - delta}
    for b, w in want.items():
        assert abs(SZ.acc_pred(b, svd, orig, mir) - w) < 1e-12
        assert abs(SZ.predicted_gain(b, svd, orig, mir) - (w - svd[b])) < 1e-12
    assert abs(SZ.acc_pred(0.65 * 0.765, svd, orig, mir) - (76.5 - delta)) < 1e-9     # lands on a measured point
    assert abs(SZ.acc_svd_interp(1.0, svd, orig) - orig) < 1e-12 and SZ.acc_svd_interp(0.40, svd, orig) == 70.0
    assert SZ.acc_pred(0.40, svd, orig, mir) < SZ.acc_pred(0.50, svd, orig, mir) < SZ.acc_pred(0.65, svd, orig, mir)
    assert abs(SZ.acc_pred(0.5, svd, orig, orig) - SZ.acc_svd_interp(0.5 / 0.765, svd, orig)) < 1e-12   # delta 0
    for bad in (0.39, 1.01):
        with pytest.raises(ValueError):
            SZ.acc_svd_interp(bad, svd, orig)
    with pytest.raises(ValueError):
        SZ.predicted_gain(0.45, svd, orig, mir)                            # no measured SVD-LLM point there
    assert SZ.gain_within(1.0, 2.0) and SZ.gain_within(3.0, 2.0) and not SZ.gain_within(0.9, 2.0)
    assert not SZ.gain_within(3.1, 2.0)


# ---------------------------------------------------------------- bases, whitening, allocation
def test_head_sign_basis_is_head_local_and_diagonalises(mirror_exact):
    s = mirror_exact[1]["structure"]
    for hm in list(s["heads_after"]) + list(s["heads"]):                   # gauged (B = C = I on the pair) and not
        assert hm.n_pairs == 1
        for which in ("B", "C"):
            hb = MS.HeadSignBasis(hm, which)
            Q, P = hb.Q, ST.head_involution(hm, which)
            assert np.abs(Q.T @ Q - np.eye(hb.n)).max() < 1e-12
            sign = np.r_[np.ones(hb.n_plus), -np.ones(hb.n_minus)]
            assert np.abs(Q.T @ P @ Q - np.diag(sign)).max() < 1e-12
            X = hm.B if which == "B" else hm.C
            assert hb.trace == int(round(sum(np.trace(X[h]) for h in hm.self_heads)))
            heads_hit = [{int(r) // hm.dh for r in np.flatnonzero(np.abs(Q[:, c]) > 1e-14)} for c in range(hb.n)]
            pair = set(hm.pairs[0])
            assert all(len(h) == 1 or h == pair for h in heads_hit)          # each column lives on one head or the pair
    # Teeth: the pre-gauge maps of the pair are not the identity.
    assert np.abs(s["heads"][0].B[s["heads"][0].pairs[0][0]] - np.eye(16)).max() > 0.1


def test_symmetrised_moment_has_the_same_blocks_and_no_cross_block():
    g = torch.Generator().manual_seed(1)
    X = torch.randn(200, 12, dtype=torch.float64, generator=g)
    M = X.T @ X
    perm = np.array([3, 1, 5, 0, 4, 2, 6, 7, 11, 9, 10, 8])                 # pairs (0,3), (2,5), (8,11)
    for basis in (ST.PermSignBasis(perm), ST.DenseSignBasis(random_involution_fixing_ones(12, 7, seed=3))):
        E = torch.as_tensor(basis.dense())
        R = E @ torch.diag(torch.as_tensor(np.r_[np.ones(basis.n_plus), -np.ones(basis.n_minus)])) @ E.T
        Msym = 0.5 * (M + R @ M @ R)
        for a, b in zip(ST.sign_moments(basis, M), ST.sign_moments(basis, Msym)):
            assert rel(a, b) < 1e-13
        rot = E.T @ Msym @ E
        assert float(rot[:basis.n_plus, basis.n_plus:].abs().max()) < 1e-10 * float(rot.abs().max())
        assert float((E.T @ M @ E)[:basis.n_plus, basis.n_plus:].abs().max()) > 1e-3        # teeth


def test_whitened_truncation_is_the_svdllm_arithmetic():
    g = torch.Generator().manual_seed(0)
    Fb = torch.randn(9, 6, dtype=torch.float64, generator=g)
    X = torch.randn(40, 6, dtype=torch.float64, generator=g)
    M = X.T @ X
    U, S, Vh, L, _ = MS.whitened_svd(Fb, M)
    for r in (2, 4, 6):
        Ft = MS._truncate(U, S, Vh, L, r)
        Wb = B.lowrank_factors("svdllm", Fb, r, M=M)[0]
        assert rel(Ft, Wb) < 1e-12
        for mode in ("balanced", "whitened"):
            Uf, Vf = MS._factor_pair(Ft, r, U, S, Vh, L, mode)
            assert Uf.shape == (9, r) and Vf.shape == (6, r) and rel(Uf @ Vf.T, Ft) < 1e-11
    assert rel(MS._truncate(U, S, Vh, L, 6), Fb) < 1e-12                    # full rank reproduces the block
    sv = torch.linalg.svdvals(Fb @ MS.whitened_svd_factor(M, 6, B.DEFAULT_DAMP, Fb))
    assert rel(sv, S) < 1e-12                                              # pass-1 spectra = pass-2 decomposition


def test_allocate_by_energy_greedy_order_and_dense_switch():
    a = {"shape": (3, 2), "sv2": [9.0, 1.0], "norm": 10.0}                 # rank cap floor(6/5) = 1, then dense
    b = {"shape": (4, 1), "sv2": [4.0], "norm": 4.0}                        # cap 0: one dense unit (4 parameters)
    al = MS.allocate_by_energy([a, b], 4)
    assert (al[0]["rank"], al[1]["dense_storage"], al[1]["params"]) == (0, True, 4)
    al = MS.allocate_by_energy([a, b], 5)
    assert al[0]["rank"] == 1 and not al[0]["dense_storage"] and al[1]["dense_storage"]
    assert sum(x["params"] for x in al) == 9 and abs(al[0]["discarded_energy"] - 0.1) < 1e-12
    al = MS.allocate_by_energy([a, b], 10)
    assert al[0]["dense_storage"] and al[0]["params"] == 6 and al[0]["rank"] == 2 and al[1]["params"] == 4
    al = MS.allocate_by_energy([a, b], 5, energy="absolute")               # 9/5 beats 4/4: a first
    assert al[0]["rank"] == 1 and al[1]["params"] == 0
    al = MS.allocate_by_energy([a, b], 5, dense_switch=False)              # ranks only: b's 1/5 beats a's 0.9/5
    assert not any(x["dense_storage"] for x in al) and al[0]["rank"] == 0 and al[1]["rank"] == 1
    assert al[1]["params"] == 5
    al = MS.allocate_by_energy([a, b], 0, min_rank=1)
    assert al[0]["rank"] == 1 and al[1]["rank"] == 0                       # min_rank stops at the cap
    al = MS.allocate_by_energy([a, b], 100)
    assert all(x["dense_storage"] for x in al) and sum(x["params"] for x in al) == 10   # everything, below budget


# ---------------------------------------------------------------- Mirror + SVD-LLM
def test_full_rank_reproduces_mirror_exact(ad, struct, stats, mirror_exact):
    m0, info0 = mirror_exact
    x = probe(2, 9)
    for ranks in ("full", "dense"):
        model, cons, spec, info = MS.build(ad, struct, 0.65, stats, {"ranks": ranks})
        for b in range(ad.depth):
            new, old = MS._modules_of(model.blocks[b]), ST._read(m0.blocks[b], MF.COMPONENTS)
            biases = MS._biases(old)
            for name in MS.MATRICES:
                W, bb = new[name].dense(torch.float64)
                assert rel(W, old[name]) < 1e-10 and rel(bb, biases[name]) < 1e-10, (ranks, b, name)
        with torch.no_grad():
            assert rel(ad.cls_feature(x, model), ad.cls_feature(x, m0)) < 1e-10
        if ranks == "dense":
            assert info["params_matrices"] == info0["params_matrices"] == info["mirror_exact_matrices"]
            assert info["params_model"] == info["mirror_exact_params_model"]
        else:
            assert info["params_matrices"] > info0["params_matrices"]       # full-rank factors cost more
            assert all(e["rank"] == e["full"] and not e["dense_storage"] for e in info["allocation"])
    with pytest.raises(ValueError):
        MS.build(ad, struct, 0.5, None)                                   # the real method needs statistics


@pytest.mark.parametrize("frac", FRACS)
def test_counts_within_one_percent_and_exact(ad, built, frac):
    model, cons, spec, info = built[frac]
    N = SZ.full_model_params(ad)
    assert info["target"] == SZ.target_params(ad, frac) and info["params_full"] == N
    assert info["eligible"] and 0 <= info["gap_matrices"] <= 0.01 and 0 <= info["gap"] <= 0.01
    assert info["params_model"] == N - info["span_dense"] + info["params_matrices"] + info["params_basis_change"]
    assert info["params_model"] - info["target"] <= info["gap_step"]
    realised = sum(MS._modules_of(model.blocks[b])[n].counted_params() for b in info["blocks"] for n in MS.MATRICES)
    assert realised == info["params_matrices"] == sum(r["params"] for r in info["per_matrix"])
    assert sum(r["mlp"] + r["attn"] for r in info["block_costs"]) == info["params_matrices"]
    assert any(e["rank"] < e["full"] for e in info["allocation"])          # something was truncated
    assert info["params_matrices"] < info["mirror_exact_matrices"]
    for e in info["allocation"]:
        m, n = e["shape"]
        assert 0 <= e["rank"] <= min(m, n)
        assert e["params"] == (m * n if e["dense_storage"] else e["rank"] * (m + n))
        if not e["dense_storage"]:
            assert e["rank"] <= (m * n) // (m + n)                          # never factorised past break-even
    A, reg = info["structure"]["A_star"], info["structure"]["reg_perm"]
    ex, inv = exactness(ad, model, A, reg)
    assert ex["pass"] and ex["max_rel_inside"] < 1e-10 and inv < 1e-10
    chk = cons[0].check(model)
    assert chk["ok"] and chk["max_offdiag"] < 1e-12 and chk["max_bias_odd"] < 1e-12
    json.dumps(spec)                                                       # the spec is JSON


def test_module_paths_agree_and_only_counted_entries_train(built):
    model, cons, spec, info = built[0.50]
    n_factor, paths = 0, set()
    with torch.no_grad():
        for b in info["blocks"]:
            for name, mod in MS._modules_of(model.blocks[b]).items():
                x = torch.randn(3, 5, mod.in_features, dtype=torch.float64, generator=torch.Generator().manual_seed(b))
                W, bb = mod.dense()
                flag = mod.use_dense
                paths.add(flag)
                try:
                    for use in (False, True):                                 # factor path and dense product
                        mod.use_dense = use
                        assert rel(mod(x), x @ W.T + bb) < 1e-12, (b, name, use)
                finally:
                    mod.use_dense = flag
                n_factor += sum(p.numel() for n_, p in mod.named_parameters() if n_ != "b_e")
    assert n_factor == info["params_matrices"]                                # trainable matrix entries = the count
    assert paths == {False, True}                                             # both paths occur in a real build


def test_ranks_fall_with_the_budget(built):
    tot = {f: sum(e["rank"] for e in built[f][3]["allocation"]) for f in FRACS}
    assert tot[0.65] > tot[0.50] > tot[0.40]
    m40 = built[0.40][3]
    assert m40["params_frac"] < built[0.50][3]["params_frac"] < built[0.65][3]["params_frac"]


def test_exact_after_gradient_steps_on_the_factors(ad, struct, stats):
    model, cons, spec, info = MS.build(ad, struct, 0.50, stats)
    blocks = info["blocks"]
    trainable = configure_trainable(model, blocks, "all", cons)
    names = {n for n, _ in trainable}
    assert any(".U_e" in n for n in names) and any(".V_" in n or ".W_" in n for n in names)
    assert any(n.endswith(".b_e") for n in names) and "pos_embed" in names and "cls_token" in names
    assert not any("norm.affine" in n or ".ls1." in n or ".ls2." in n for n in names)
    assert not any(n.startswith("mirror_bases") for n, _ in model.named_parameters())
    for c in cons:
        c.prepare(model) if hasattr(c, "prepare") else None
    ref = copy.deepcopy(model)
    opt = torch.optim.AdamW([p for _, p in trainable], lr=1e-2, weight_decay=0.05)
    target = torch.randn(2, ad.embed_dim, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    for step in range(3):
        opt.zero_grad()
        x = probe(2, 20 + step)
        x = torch.where(torch.rand(2, 1, 1, 1, generator=torch.Generator().manual_seed(step)) < 0.5, x, mirror(x))
        loss = (ad.cls_feature(x, model) - target).pow(2).mean() + 0.1 * model.blocks[1].mlp.fc1(
            ad.layer0_tokens(x, model)).pow(2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for c in cons:
                c(model)                                                   # embeddings re-projected
    x = probe(2, 7)
    with torch.no_grad():
        assert rel(ad.cls_feature(x, model), ad.cls_feature(x, ref)) > 1e-6     # the steps changed the model
    changed = [n for (n, p), (_, q) in zip(model.named_parameters(), ref.named_parameters()) if not torch.equal(p, q)]
    assert any(".U_e" in n for n in changed) and any(".V_e" in n for n in changed) and "pos_embed" in changed
    ex, inv = exactness(ad, model, info["structure"]["A_star"], info["structure"]["reg_perm"])
    assert ex["pass"] and ex["max_rel_inside"] < 1e-10 and inv < 1e-10
    chk = cons[0].check(model)
    assert chk["ok"] and chk["max_offdiag"] < 1e-12                        # every block kept its kind and rank
    assert cons[1].check(model)["max_share"] < 1e-20                       # embeddings exactly projected


def test_load_export_and_refine(ad, built, tmp_path):
    model, cons, spec, info = built[0.40]
    model = copy.deepcopy(model)
    sd = model.state_dict()
    assert "mirror_bases.A__Qp" in sd and "mirror_bases.P0__src_p" in sd
    assert not any("__" in k and not k.startswith("mirror_bases.") for k in sd)   # each basis stored once
    back = MS.load(ad, json.loads(json.dumps(spec)), sd)
    x = probe(2, 11)
    with torch.no_grad():
        f0 = ad.cls_feature(x, model)
        assert rel(ad.cls_feature(x, back), f0) < 1e-13
        dense = MS.export_dense(model)
        assert all(isinstance(MS._modules_of(dense.blocks[b])["fc1"], torch.nn.Linear) for b in info["blocks"])
        assert not hasattr(dense, "mirror_bases")
        assert rel(ad.cls_feature(x, dense), f0) < 1e-10
    A, reg = info["structure"]["A_star"], info["structure"]["reg_perm"]
    ex, inv = exactness(ad, dense, A, reg)
    assert ex["max_rel_inside"] < 1e-10 and inv < 1e-10
    # refine_bases: bases rounded through float32 lose exactness; refining restores it, moving them by rounding only.
    with torch.no_grad():
        for name, buf in model.mirror_bases.named_buffers():
            if buf.is_floating_point():
                buf.copy_(buf.float().double())
    ex_bad, _ = exactness(ad, model, A, reg)
    assert ex_bad["max_rel_inside"] > 1e-9
    moved = MS.refine_bases(model, info["structure"])
    assert 0 < moved < 1e-6
    ex_ok, inv_ok = exactness(ad, model, A, reg)
    assert ex_ok["max_rel_inside"] < 1e-10 and inv_ok < 1e-10
    # The structure file round trip (mirror family format).
    from mirror.heal_structured import load_structure

    st = load_structure(MS.save_structure(tmp_path / "structure.npz", info["structure"]))
    assert np.array_equal(st["A_star"], info["structure"]["A_star"]) and st["blocks"] == info["blocks"]
    lc = MS.constraints_for(ad, spec, st)                                 # checks on the loaded student
    assert lc[0].check(back)["ok"] and lc[0].check(back)["max_offdiag"] < 1e-12
    assert lc[1].check(back)["max_share"] < 1e-20


# ---------------------------------------------------------------- partial span
def test_partial_span_exact_in_the_span_and_later_blocks_train(ad, struct):
    k, L = 2, ad.depth
    model, cons, spec, info = PS.build_partial(ad, struct, k)
    N = SZ.full_model_params(ad)
    assert info["params_model"] == N - info["span_dense"] + info["params_matrices"] + info["params_basis_change"]
    assert info["params_basis_change"] == ad.embed_dim ** 2                # one d x d change at the span end
    full = MF.build_mirror_variant(ad, "mirror_exact", range(L), MF.COMPONENTS, struct)[1]
    assert info["target"] == N - full["span_dense"] + full["params_total"]
    A = info["structure"]["A_star"]
    rows = stream_equivariance(ad, model, probe(2, 5), A)
    assert max(r["rel_max"] for r in rows if r["layer"] <= k) < 1e-10
    assert min(r["rel_max"] for r in rows if r["layer"] > k) > 1e-6       # teeth: the dense blocks are not
    trainable = configure_trainable(model, list(range(k)), "all", cons)
    names = {n for n, _ in trainable}
    for b in range(k, L):                                                  # every parameter of a later block trains
        assert {f"blocks.{b}.{n}" for n, _ in model.blocks[b].named_parameters()} <= names
    for b in range(k):
        assert f"blocks.{b}.ls1.gamma" not in names and f"blocks.{b}.mlp.fc1.weight" in names
    assert "norm.affine.weight" not in names and "pos_embed" in names
    for c in cons:
        c.prepare(model)
    ref = copy.deepcopy(model)
    opt = torch.optim.AdamW([p for _, p in trainable], lr=1e-2)
    for step in range(2):
        opt.zero_grad()
        ad.cls_feature(probe(2, 30 + step), model).pow(2).sum().backward()
        if step == 0:
            for b in range(k, L):
                g = [p.grad for p in model.blocks[b].parameters() if p.grad is not None]
                assert g and max(float(x.abs().max()) for x in g) > 0
        opt.step()
        with torch.no_grad():
            for c in cons:
                c(model)
    for b in range(k, L):
        assert not torch.equal(model.blocks[b].mlp.fc1.weight, ref.blocks[b].mlp.fc1.weight)
    rows = stream_equivariance(ad, model, probe(2, 6), A)
    assert max(r["rel_max"] for r in rows if r["layer"] <= k) < 1e-10
    assert cons[0].check(model)["max_rel_violation"] < 1e-12 and cons[1].check(model)["max_rel_violation"] < 1e-12
    back = PS.load_partial(ad, json.loads(json.dumps(spec)), model.state_dict())
    x = probe(2, 8)
    with torch.no_grad():
        assert rel(ad.cls_feature(x, back), ad.cls_feature(x, model)) < 1e-13
    with pytest.raises(ValueError):
        PS.build_partial(ad, struct, L)                                    # k = depth is the full conversion


def test_partial_span_reproduces_the_must07_9both_count():
    """From the canonical DINOv2-B structure (fit seed 0): 9 blocks = 66,325,200 parameters (0.766), the full
    conversion 66,203,360 (0.7646), as results/MUST-07 and notes/paper/method_spec.md Section 5 report."""
    path = ROOT / "results" / "MUST-07" / "structure" / "dinov2_vitb14_reg_s0.npz"
    if not path.exists():
        pytest.skip("canonical structure file not on this machine (git-ignored)")
    with np.load(path) as z:
        A = z["A_star"]
        A9, A12 = A[8].copy(), A[11].copy()
        del A
        perms = [np.asarray(p, dtype=np.int64) for p in z["perms"]]
        heads = [HeadMaps(sigma=z["sigma"][b], B=z["B"][b], C=z["C"][b]) for b in range(12)]
        reg = {9: np.asarray(z["reg_perm"][8], dtype=np.int64), 12: np.asarray(z["reg_perm"][11], dtype=np.int64)}
    st = MF.SpanStructure(model="dinov2_vitb14_reg", fit_seed=0, A_layers=[], perms=perms, heads=heads,
                          A_star={9: A9, 12: A12}, reg_perm=reg)
    res = PS.partial_span_params(DINO_B, st, 9)
    assert res["params_matrices"] == 42_852_816 and res["params_basis_change"] == 589_824
    assert res["params_model"] == 66_325_200 and abs(res["params_frac"] - 0.7660253993737748) < 1e-12
    assert res["target"] == 66_203_360 and res["target_matrices"] == 64_554_464
    assert res["eligible"] and abs(res["gap"] - (66_325_200 - 66_203_360) / 66_203_360) < 1e-15
