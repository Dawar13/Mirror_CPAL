"""Final-phase baselines (mirror/final/baselines: svdllm_full, iosvd, dense_students): CPU only, tiny random ViT.

On the tiny timm ViT (float64, d 64, h 256, 4 heads of 16, 4 blocks) with synthetic images:
  * full SVD-LLM: the compressed start equals the whitening-only "svdllm" (same budget rule, kind lowrank with given
    factors); the given-factor trainable form computes the compressed model; the sequential schedule toggles
    requires_grad as specified and, in a real optimiser loop, only the scheduled factor group changes; the optional
    closed-form update equals the official lstsq solution, leaves block 0 unchanged and never increases the error;
  * IO-SVD: with an identity output metric it equals SVD-LLM's truncation; its truncation is optimal in the weighted
    norm; the output curvature equals an explicit Jacobian computation; the calibration gradient equals autograd's;
    the greedy allocation follows Algorithms 1 and 2 (storage gain, floor, stop rule); compress meets the budget rule
    and its trainable form computes the compressed model; dense_as_width makes dense matrices train with their own
    count;
  * width pruning meets the budget rule, does not count the zeroed heads and keeps them at zero while training;
  * block dropping picks the least harmful blocks for a synthetic score, reports its gap below the target, and its
    trainable form and reload compute the shortened model.
"""
import copy
import dataclasses
import hashlib
import json

import numpy as np
import pytest
import torch

from mirror import baselines as B
from mirror import budget as BU
from mirror.adapters import TimmTinyAdapter
from mirror.attention import absorb_attn
from mirror.data import normalize_for
from mirror.final.baselines import dense_students as DS
from mirror.final.baselines import iosvd as IO
from mirror.final.baselines import svdllm_full as SF
from mirror.heal import LowRankLinear, configure_trainable
from mirror.heal_structured import IndexedLinear, dense_weight, make_trainable
from mirror.m1 import random_head_structure, random_involution_fixing_ones
from mirror.matching import random_pairing
from mirror.surgery import absorb_block, fold_mlp

L = 4
COMPS = ("mlp", "attn")


def structure(seed=0):
    A = random_involution_fixing_ones(64, 37, seed=seed + 1)
    perms = [random_pairing(256, 256 // 3 + 5 * b, seed=seed * 100 + 50 + b) for b in range(L)]
    heads = [random_head_structure(4, 16, 1 + (b % 2), seed=seed * 100 + 90 + b) for b in range(L)]
    return A, perms, heads


def uint8_loader(n=6, seed=2, split=3):
    imgs = torch.from_numpy(np.random.default_rng(seed).integers(0, 256, (n, 3, 56, 56), dtype=np.uint8))
    lab = torch.arange(n) % 3
    return [(imgs[:split], lab[:split]), (imgs[split:], lab[split:])]


def model_hash(model) -> str:
    h = hashlib.sha1()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rel_feat(ad, x, a, b) -> float:
    with torch.no_grad():
        fa, fb = ad.cls_feature(x, a), ad.cls_feature(x, b)
    return float((fa - fb).abs().max() / fb.abs().max())


@pytest.fixture(scope="module")
def ad():
    return TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float64)


@pytest.fixture(scope="module")
def loader():
    return uint8_loader()


@pytest.fixture(scope="module")
def st(ad, loader):
    return B.collect_stats(ad, loader, "cpu", range(L), methods=["svdllm", "flap"], components=COMPS)


@pytest.fixture(scope="module")
def bud(ad):
    A, perms, heads = structure()
    return BU.mirror_exact_budget(ad, A, list(range(L)), perms, heads, COMPS)


@pytest.fixture(scope="module")
def ex(ad, loader):
    return IO.collect_extra_stats(ad, loader, "cpu", range(L), COMPS, n_probes=2, seed=0)


@pytest.fixture(scope="module")
def x64():
    return torch.randn(2, 3, 56, 56, generator=torch.Generator().manual_seed(0), dtype=torch.float64)


def check_rule(info, bud):
    """The budget rule of mirror.budget.gap_ok: 0 <= gap <= 1%, or within one step where 1% is unreachable; the
    whole-model count and target as model_level defines them."""
    assert info["gap"] >= 0 and info["gap_ok"], (info["gap"], info.get("gap_note"))
    assert info["gap"] <= max(0.01, info["gap_step"] / info["budget_total"]) + 1e-12
    if info["gap_reachable"]:                                    # true on these budgets: 0 <= gap <= 1% itself
        assert info["gap"] <= 0.01
    assert info["params_total"] - info["budget_total"] < info["gap_step"]
    assert info["eligible"] and info["gap_model"] >= 0
    assert info["params_model"] == info["params_original"] - bud.dense_total + info["params_total"]
    assert info["target"] == info["params_original"] - bud.dense_total + bud.total


# ================================================================ full SVD-LLM
def test_svdllm_full_start_equals_whitening_and_meets_budget(ad, st, bud):
    h0 = model_hash(ad.model)
    m, info = SF.compress(ad.model, list(range(L)), COMPS, bud, st, verbose=False)
    ref, _ = B.compress(ad.model, "svdllm", list(range(L)), COMPS, bud, st, verbose=False)
    check_rule(info, bud)
    assert info["target"] == info["params_original"] - bud.dense_total + bud.total
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), ref.state_dict().values()))
    assert model_hash(ad.model) == h0
    for pm in info["per_matrix"]:
        assert pm["kind"] == "lowrank" and pm["U_key"] in info["tensors"] and pm["V_key"] in info["tensors"]
        assert info["tensors"][pm["U_key"]].shape[1] == pm["rank"] == info["tensors"][pm["V_key"]].shape[0]
    sc = B.structure_check(m, info)
    assert sc["ok"] and sc["params_realised"] == info["params_matrices"]
    assert SF.training_hooks(info)[0].describe()["order"] == "u_first"
    with pytest.raises(ValueError):
        SF.compress(ad.model, list(range(L)), COMPS, bud, st, {"order": "sideways"}, verbose=False)
    with pytest.raises(ValueError):
        SF.compress(ad.model, list(range(L)), COMPS, bud, st, {"closed_form_update": True}, verbose=False)


def test_svdllm_full_given_factors_and_sequential_schedule(ad, st, bud, x64):
    m, info = SF.compress(ad.model, list(range(L)), COMPS, bud, st, verbose=False)
    ref = copy.deepcopy(m)
    with pytest.raises(TypeError):                                    # not yet in trainable form
        SF.training_hooks(info)[0].begin_step(0, 4, m)
    mt, cons, spec = make_trainable(m, info, ad, lowrank_factors="given")
    assert spec["lowrank_factors"] == "given"
    for pm in info["per_matrix"]:
        mod = mt.blocks[pm["block"]].get_submodule(SF.RECORD_PATHS[pm["name"]])
        assert isinstance(mod, LowRankLinear) and mod.rank == pm["rank"]
        assert torch.equal(mod.U.detach(), info["tensors"][pm["U_key"]].to(mod.U))     # W'_u, not re-balanced
        assert torch.equal(mod.V.detach(), info["tensors"][pm["V_key"]].to(mod.V))     # W'_v
    assert rel_feat(ad, x64, mt, ref) < 1e-6                       # float32 factors of a float64 model
    # the schedule: U first, V second, split at round(0.5 * total)
    hk = SF.training_hooks(info)[0]
    groups = hk.factor_parameters(mt)
    assert hk.begin_step(0, 10, mt)["train"] == "U" and hk.boundary(10) == 5
    assert hk.boundary(780) == 390 and hk.boundary(200) == 100       # the heal run and a learning-rate trial
    assert all(p.requires_grad for p in groups["U"]) and not any(p.requires_grad for p in groups["V"])
    for p in groups["U"]:
        p.grad = torch.ones_like(p)
    assert hk.begin_step(5, 10, mt)["train"] == "V"
    assert all(p.grad is None and not p.requires_grad for p in groups["U"])
    assert all(p.requires_grad for p in groups["V"])
    vf = SF.SequentialFactorSchedule(info["per_matrix"], order="v_first", split=0.25)
    assert vf.trained_factor(0, 8) == "V" and vf.trained_factor(1, 8) == "V" and vf.trained_factor(2, 8) == "U"
    assert SF.SequentialFactorSchedule(info["per_matrix"], split=0.0).trained_factor(0, 8) == "V"
    # in a real optimiser loop (mirror/heal.py's trainable set and zeroing), only the scheduled group changes
    hooks = SF.training_hooks(info)
    trainable = configure_trainable(mt, list(range(L)), "all", cons + hooks)
    names = {n for n, _ in trainable}
    assert any(n.endswith(".U") for n in names) and any(n.endswith(".V") for n in names)
    opt = torch.optim.AdamW([p for _, p in trainable], lr=1e-2, weight_decay=0.05)
    total = 4
    for step in range(total):
        hooks[0].begin_step(step, total, mt)
        before = {k: [p.detach().clone() for p in v] for k, v in groups.items()}
        opt.zero_grad(set_to_none=True)
        (ad.cls_feature(x64, mt) ** 2).mean().backward()
        opt.step()
        moved = {k: any(not torch.equal(a, p.detach()) for a, p in zip(before[k], v)) for k, v in groups.items()}
        assert moved == ({"U": True, "V": False} if step < 2 else {"U": False, "V": True}), (step, moved)


def test_closed_form_update_matches_official_lstsq(ad, st, bud, loader):
    calib = {"adapter": ad, "loader": loader, "device": "cpu"}
    m, info = SF.compress(ad.model, list(range(L)), COMPS, bud, st, {"closed_form_update": True}, calib=calib,
                          verbose=False)
    recs = info["closed_form_update"]
    assert len(recs) == len(info["per_matrix"])
    assert all(r["rel_output_err_after"] <= r["rel_output_err_before"] + 1e-12 for r in recs)
    assert max(r["u_rel_change"] for r in recs if r["block"] == 0) < 1e-3       # block 0: whitening is already optimal
    assert max(r["u_rel_change"] for r in recs if r["block"] > 0) > 1e-4        # later blocks see compressed inputs
    assert B.structure_check(m, info)["ok"]
    check_rule(info, bud)
    # independent check for block 2 fc1 and wo: inputs through the updated blocks 0, 1 and the original block 2
    ref = copy.deepcopy(ad.model.blocks[2])
    absorb_block(ref)
    absorb_attn(ref)
    cap = {"fc1": [], "proj": []}
    hs = [ref.mlp.fc1.register_forward_pre_hook(lambda mod, a: cap["fc1"].append(a[0].reshape(-1, 64))),
          ref.attn.proj.register_forward_pre_hook(lambda mod, a: cap["proj"].append(a[0].reshape(-1, 64)))]
    with torch.no_grad():
        for imgs, _ in loader:
            t = ad.layer0_tokens(normalize_for(ad, imgs), m)
            for i in range(2):
                t = m.blocks[i](t)
            ref(t)
    for h in hs:
        h.remove()
    from mirror.attention import fold_attn

    for name, slot, W in (("fc1", "fc1", fold_mlp(ad.model.blocks[2])["W1"]),
                          ("wo", "proj", fold_attn(ad.model.blocks[2])["Wo"])):
        X = torch.cat(cap[slot])
        V = info["tensors"][f"2.{name}.V"].double()
        U_off = torch.linalg.lstsq(X @ V.T, X @ W.T).solution.T          # SVDLLM.py add_batch_update_u, in our factors
        U_got = info["tensors"][f"2.{name}.U"].double()
        assert float((U_got - U_off).norm() / U_off.norm()) < 1e-5, name


# ================================================================ IO-SVD
def test_io_identity_output_metric_equals_svdllm_and_is_optimal():
    g = torch.Generator().manual_seed(3)
    W = torch.randn(12, 9, generator=g, dtype=torch.float64)
    X = torch.randn(200, 9, generator=g, dtype=torch.float64) @ torch.randn(9, 9, generator=g, dtype=torch.float64)
    M = X.T @ X
    r = 4
    f = IO.io_factorise(W, M / 200, torch.eye(12, dtype=torch.float64), 1e-6, 0.0)
    W_io, A, V = IO._reconstruct(f, r)
    W_ll, _, _, _ = B.lowrank_factors("svdllm", W, r, M=M, damp=1e-6)
    assert float((W_io - W_ll).norm() / W_ll.norm()) < 1e-9
    assert float((A @ V - W_io).norm() / W_io.norm()) < 1e-12
    # a non-trivial output metric: the IO truncation minimises || C^1/2 (W - W^) R^1/2 || at rank r
    Y = torch.randn(300, 12, generator=g, dtype=torch.float64) @ torch.randn(12, 12, generator=g, dtype=torch.float64)
    C = Y.T @ Y / 300
    f = IO.io_factorise(W, M / 200, C, 1e-6, 1e-6)
    W_io, _, _ = IO._reconstruct(f, r)

    def werr(Wh):
        return float((f["Cs"] @ (W - Wh) @ f["Rs"]).norm() ** 2)

    assert abs(werr(W_io) - float((f["S"][r:] ** 2).sum())) < 1e-8 * float((f["S"] ** 2).sum())
    assert werr(W_io) <= werr(W_ll) + 1e-9
    assert werr(W_io) <= werr(B.lowrank_factors("svd", W, r)[0]) + 1e-9


def test_adaptive_ranks_follow_algorithms_1_and_2():
    s_small = torch.linspace(1.0, 0.1, 10) * 0.01
    s_large = torch.linspace(1.0, 0.1, 10)
    shapes = [(10, 10), (10, 10)]                       # r* = 5; storage gain 0 above r* + 1, then 20 per rank
    out = IO.adaptive_ranks(shapes, [s_small, s_large], 140, eta=0.1)
    assert out["ranks"] == [2, 10] and out["dense"] == [False, True] and out["total"] == 140
    assert out["r_star"] == [5, 5] and out["r_min"] == [1, 1]
    out = IO.adaptive_ranks(shapes, [s_small, s_large], 135, eta=0.1)
    assert out["ranks"] == [2, 10] and out["total"] == 140      # rounds towards more parameters (gap 5 < step 20)
    out = IO.adaptive_ranks(shapes, [s_small, s_large], 140, eta=0.5)
    assert out["r_min"] == [3, 3] and out["ranks"] == [3, 4] and out["total"] == 140   # floor, then the other matrix
    out = IO.adaptive_ranks(shapes, [s_small, s_large], 10, eta=0.5)
    assert out["ranks"] == [3, 3] and out["stop"] == "every matrix at its floor"
    out = IO.adaptive_ranks([(4, 4), (4, 4)], [torch.ones(4), torch.ones(4)], 16 + 8, eta=0.0)
    assert out["ranks"] == [1, 2] and out["total"] == 24          # ties go to the lower matrix index
    assert out["dense"] == [False, True]                          # integer tie r = r* = 2 costs m n: kept dense
    out = IO.adaptive_ranks([(5, 3)], [torch.tensor([3.0, 2.0, 1.0])], 15, eta=0.0)
    assert out["ranks"] == [2] and out["dense"] == [True] and out["total"] == 15        # r* = 1: rank 2 is dense


def test_output_curvature_equals_explicit_jacobian(ad):
    one = uint8_loader(n=2, seed=5, split=2)[:1]
    exa = IO.collect_extra_stats(ad, one, "cpu", range(L), COMPS, probes="exact", output="cls")
    m = copy.deepcopy(ad.model)
    for blk in m.blocks:
        absorb_block(blk)
        absorb_attn(blk)
    x = normalize_for(ad, one[0][0])
    with torch.no_grad():
        cls = ad.cls_feature(x, m)
    w = 1.0 / (64 * float(cls.pow(2).mean()))
    for b, mod, h in ((1, m.blocks[1].mlp.fc1, 256), (2, m.blocks[2].attn.proj, 64)):
        with torch.no_grad():
            base = []
            hd = mod.register_forward_hook(lambda md, a, o: base.append(o))
            ad.cls_feature(x, m)
            hd.remove()
        shape = base[0].shape

        def f(delta):
            hd2 = mod.register_forward_hook(lambda md, a, o: o + delta)
            try:
                return ad.cls_feature(x, m)
            finally:
                hd2.remove()

        J = torch.autograd.functional.jacobian(f, torch.zeros(shape, dtype=torch.float64))    # [B, d, B, T, h]
        Cx = sum(w * torch.einsum("kth,kti->hi", J[i, :, i], J[i, :, i]) for i in range(shape[0]))
        key = f"{b}.{'fc1' if h == 256 else 'wo'}"
        assert float((exa.C[key] - Cx).norm() / Cx.norm()) < 1e-9, key
    assert exa.c_count == exa.n_tokens == 2 * ad.n_tokens
    # the Gaussian estimate is a symmetric positive semi-definite matrix of the right size for every matrix
    g = IO.collect_extra_stats(ad, one, "cpu", [1], ("mlp",), n_probes=3, output="final")
    assert set(g.C) == {"1.fc1", "1.fc2"} and g.c_count == 3 * g.n_tokens
    ev = torch.linalg.eigvalsh(g.C["1.fc1"])
    assert float(ev.min()) > -1e-10 * float(ev.max())


def test_first_order_gradient_equals_autograd(ad):
    two = uint8_loader(n=4, seed=6, split=2)

    def loss_fn(adapter, model, x, labels):
        return (adapter.cls_feature(x, model) ** 2).sum() + 0.1 * labels.double().sum()

    exg = IO.collect_extra_stats(ad, two, "cpu", [1], COMPS, n_probes=1, loss_fn=loss_fn)
    m = copy.deepcopy(ad.model)
    absorb_block(m.blocks[1])
    absorb_attn(m.blocks[1])
    for p in (m.blocks[1].mlp.fc2.weight, m.blocks[1].attn.qkv.weight):
        p.requires_grad_(True)
    for imgs, lab in two:
        loss_fn(ad, m, normalize_for(ad, imgs), lab).backward()
    Wg = m.blocks[1].attn.qkv.weight.grad
    assert float((exg.G["1.fc2"] - m.blocks[1].mlp.fc2.weight.grad).norm() / exg.G["1.fc2"].norm()) < 1e-10
    assert float((exg.G["1.wk"] - Wg[64:128]).norm() / Wg[64:128].norm()) < 1e-10


def test_iosvd_compress_budget_structure_and_trainable_form(ad, st, bud, ex, x64):
    h0 = model_hash(ad.model)
    m, info = IO.compress(ad.model, list(range(L)), COMPS, bud, st, ex, verbose=False)
    check_rule(info, bud)
    assert model_hash(ad.model) == h0
    sc = B.structure_check(m, info)
    assert sc["ok"] and sc["params_realised"] == info["params_matrices"]
    assert all(p["kind"] == "lowrank" and p["U_key"] in info["tensors"] for p in info["per_matrix"])
    assert len({p["rank"] for p in info["per_matrix"]}) > 2                       # adaptive: ranks differ
    again, info2 = IO.compress(ad.model, list(range(L)), COMPS, bud, st, ex, verbose=False)
    assert model_hash(again) == model_hash(m)
    mt, _, _ = make_trainable(copy.deepcopy(m), info, ad, lowrank_factors="given")
    assert rel_feat(ad, x64, mt, m) < 1e-6
    mm, im = IO.compress(ad.model, list(range(L)), COMPS, bud, st, ex, {"allocation": "matched"}, verbose=False)
    ref = {(a["block"], a["name"]): a["rank"] for a in BU.allocate("svdllm", bud)}
    assert all(p["rank"] == ref[(p["block"], p["name"])] for p in im["per_matrix"])
    check_rule(im, bud)
    with pytest.raises(ValueError):
        IO.compress(ad.model, list(range(L)), COMPS, bud, st, ex, {"score": "first_order"}, verbose=False)
    full = BU.dense_budget(ad, list(range(L)), COMPS)                          # dense budget: nothing truncated
    md, idn = IO.compress(ad.model, list(range(L)), COMPS, full, st, ex, verbose=False)
    assert all(p["dense_storage"] for p in idn["per_matrix"]) and idn["gap"] == 0.0
    assert rel_feat(ad, x64, md, ad.model) < 1e-12
    mdt, _, _ = make_trainable(md, idn, ad, lowrank_factors="given")            # full-rank given factors exist
    assert rel_feat(ad, x64, mdt, ad.model) < 1e-6


def test_dense_as_width_trains_dense_matrices_with_their_own_count(ad, st, ex, x64):
    from mirror.heal_structured import module_macs

    full = BU.dense_budget(ad, list(range(L)), COMPS)
    mixed = dataclasses.replace(full, total=full.total - 3000)                   # some matrices factored, most dense
    m, info = IO.compress(ad.model, list(range(L)), COMPS, mixed, st, ex, verbose=False)
    n_dense = sum(p["dense_storage"] for p in info["per_matrix"])
    assert 0 < n_dense < len(info["per_matrix"]) and info["gap_ok"]
    ref = copy.deepcopy(m)
    IO.dense_as_width(info, n_heads=4)
    assert sum(p["kind"] == "width" for p in info["per_matrix"]) == n_dense
    sc = B.structure_check(m, info)
    assert sc["ok"] and sc["params_realised"] == info["params_matrices"]
    mt, cons, _ = make_trainable(m, info, ad, lowrank_factors="given")
    held = sum(module_macs(mt.blocks[b].get_submodule(p)) for b in range(L)
               for p in ("mlp.fc1", "mlp.fc2", "attn.qkv", "attn.proj"))
    assert held == info["params_matrices"]                                         # exactly the counted entries
    assert rel_feat(ad, x64, mt, ref) < 1e-6 and cons[0].check(mt)["ok"]


# ================================================================ dense students
def test_width_pruned_budget_and_zeroed_heads_stay_zero(ad, st, bud, x64):
    m, info = DS.width_pruned(ad, bud, st, verbose=False)
    check_rule(info, bud)
    assert info["hparams"]["allocation"] == "uniform" and info["structure_check"]["ok"]
    for pm in info["per_matrix"]:
        assert pm["kind"] == "width"
        if pm["name"] in ("wq", "wk", "wv", "wo"):
            assert pm["params"] == pm["heads"] * 16 * 64                         # removed heads not counted
    assert any(r["heads"] < 4 for r in info["pruning"])
    ref = copy.deepcopy(m)
    mt, cons, _ = make_trainable(m, info, ad)
    assert rel_feat(ad, x64, mt, ref) < 1e-10
    stored = 0
    for b in range(L):
        blk = mt.blocks[b]
        assert isinstance(blk.attn.qkv.q, IndexedLinear) and isinstance(blk.attn.proj, IndexedLinear)
        stored += blk.mlp.fc1.weight.numel() + blk.mlp.fc2.weight.numel() + blk.attn.proj.weight.numel()
        stored += sum(getattr(blk.attn.qkv, n).weight.numel() for n in "qkv")
    assert stored == info["params_matrices"]
    trainable = configure_trainable(mt, list(range(L)), "all", cons)
    opt = torch.optim.AdamW([p for _, p in trainable], lr=1e-2, weight_decay=0.05)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        (ad.cls_feature(x64, mt) ** 2).mean().backward()
        opt.step()
    for pm in info["per_matrix"]:
        if pm["name"] not in ("wq", "wk", "wv", "wo"):
            continue
        blk = mt.blocks[pm["block"]]
        mod = blk.attn.proj if pm["name"] == "wo" else getattr(blk.attn.qkv, pm["name"][1])
        W, bias = dense_weight(mod)
        gone = [h for h in range(4) if h not in pm["heads_kept"]]
        for h in gone:
            sl = slice(16 * h, 16 * (h + 1))
            if pm["name"] == "wo":
                assert float(W[:, sl].abs().max()) == 0.0
            else:
                assert float(W[sl].abs().max()) == 0.0 and float(bias[sl].abs().max()) == 0.0


@pytest.fixture(scope="module")
def ad_drop():
    """Block 2 nearly the identity (LayerScale 1e-6), block 0 small (1e-2): the least harmful blocks, in that order."""
    a = TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float64)
    with torch.no_grad():
        for b, s in ((2, 1e-6), (0, 1e-2)):
            a.model.blocks[b].ls1.gamma.mul_(s)
            a.model.blocks[b].ls2.gamma.mul_(s)
    return a


def test_block_dropped_greedy_order_and_gap(ad_drop, bud, x64):
    with torch.no_grad():
        ref = ad_drop.cls_feature(x64)
    calls = []

    def score_fn(model):
        calls.append(len(model.blocks))
        with torch.no_grad():
            return -float((ad_drop.cls_feature(x64, model) - ref).norm())

    h0 = model_hash(ad_drop.model)
    m, info = DS.block_dropped(ad_drop, 2, score_fn, budget=bud, verbose=False)
    assert info["order"] == [2, 0] and info["kept"] == [1, 3] and len(m.blocks) == 2
    assert [len(r["candidates"]) for r in info["rounds"]] == [4, 3] and calls == [4, 3, 3, 3, 3, 2, 2, 2]
    assert info["score_original"] == 0.0
    assert max(c["score"] for c in info["rounds"][0]["candidates"]) == info["rounds"][0]["score_after"]
    assert model_hash(ad_drop.model) == h0 and len(ad_drop.model.blocks) == L       # the views changed nothing
    n_orig = sum(p.numel() for p in ad_drop.model.parameters())
    per = sum(p.numel() for p in ad_drop.model.blocks[0].parameters())
    assert info["params_model"] == n_orig - 2 * per and info["block_params"] == per
    tgt = n_orig - bud.dense_total + bud.total
    assert info["target"] == tgt and info["gap"] == pytest.approx((n_orig - 2 * per - tgt) / tgt)
    assert info["gap"] < 0 and info["eligible"] is False
    with torch.no_grad():                                                            # the copy runs the kept blocks
        t = ad_drop.layer0_tokens(x64)
        for b in info["kept"]:
            t = ad_drop.model.blocks[b](t)
        assert float((ad_drop.model.norm(t)[:, 0] - ad_drop.cls_feature(x64, m)).abs().max()) == 0.0
    _, few = DS.block_dropped(ad_drop, 2, score_fn, max_candidates=1, score_original=False, verbose=False)
    assert [len(r["candidates"]) for r in few["rounds"]] == [4, 1] and few["order"] == [2, 0]
    assert few["gap"] is None and few["eligible"] is None
    # DINOv2 keeps its blocks in an nn.ModuleList and has n_blocks: both survive a drop, and the adapter path runs it
    dl = copy.deepcopy(ad_drop.model)
    dl.blocks = torch.nn.ModuleList(list(dl.blocks))
    dl.n_blocks = L
    view = DS.drop_blocks(dl, [0, 3], copy_weights=False)
    assert isinstance(view.blocks, torch.nn.ModuleList) and view.n_blocks == 2 and dl.n_blocks == L
    assert len(dl.blocks) == L and view.blocks[1] is dl.blocks[3]
    with torch.no_grad():
        assert torch.isfinite(ad_drop.cls_feature(x64, view)).all()


def test_blockdrop_trainable_form_and_reload(ad_drop, x64):
    def score_fn(model):
        with torch.no_grad():
            return -float(ad_drop.cls_feature(x64, model).norm())

    m, info = DS.block_dropped(ad_drop, 1, score_fn, verbose=False)
    ref = copy.deepcopy(m)
    mt, cons, spec = DS.blockdrop_trainable(m, info, ad_drop)
    assert json.loads(json.dumps(spec))["block_map"] == {str(j): b for j, b in enumerate(info["kept"])}
    assert spec["teacher_blocks"] == info["kept"] and len(mt.blocks) == L - 1
    assert rel_feat(ad_drop, x64, mt, ref) < 1e-12                    # absorbed form computes the same function
    trainable = configure_trainable(mt, list(range(L - 1)), "all", cons)
    names = {n for n, _ in trainable}
    assert {"patch_embed.proj.weight", "pos_embed", "cls_token", "blocks.0.mlp.fc1.weight",
            "blocks.0.attn.qkv.bias"} <= names
    assert not names & {"norm.weight", "norm.bias", "blocks.0.norm1.weight", "blocks.0.ls2.gamma", "head.weight"}
    assert cons[0].check(mt)["ok"]
    with torch.no_grad():
        for p in mt.blocks[0].mlp.fc1.parameters():
            p.add_(0.01)
    back = DS.load_blockdrop(ad_drop, spec, mt.state_dict())
    assert rel_feat(ad_drop, x64, back, mt) == 0.0 and not any(p.requires_grad for p in back.parameters())
