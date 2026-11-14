"""Compressors at the Mirror-exact budget (mirror/baselines.py METHODS_V2, compress, structure_check): CPU only.

On the tiny random ViT (float64, d 64, h 256, 4 heads of 16, 4 blocks) and synthetic images:
  * collect_stats equals collect_baseline_stats on the shared statistics and direct computations on the original
    model for the new ones; its memory estimate is exact;
  * every method and grid point meets the budget rule (gap >= 0 and <= 1%, or within one step where the tiny model's
    granularity makes 1% unreachable), passes structure_check, leaves the input model and the statistics unchanged
    and is deterministic;
  * at the full-size budget every method reproduces the original model (SliceGPT per-block given room for its
    shortcut matrices, since at the dense budget it must pay for them by slicing);
  * svdllm equals the existing asvd on the MLP; slicegpt (MLP only, pca "stream") equals the existing slicegpt;
  * low-rank factors reproduce the written weights; asvd_scale follows its formula;
  * FLAP keeps the top fluctuation scores and its bias compensation is exact for a constant neuron or head;
  * ZipIt's greedy matching (with repeated matching alpha) and merge rules, exact for duplicated neurons;
  * the SliceGPT dense form equals the sliced network rebuilt from the info tensors;
  * structure_check catches violations.
"""
import copy
import dataclasses
import hashlib

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mirror import baselines as B
from mirror import budget as BU
from mirror.adapters import TimmTinyAdapter
from mirror.attention import fold_attn
from mirror.data import normalize_for
from mirror.m1 import random_head_structure, random_involution_fixing_ones
from mirror.matching import random_pairing
from mirror.surgery import fold_mlp, svd_truncate

L = 4


def structure(d=64, h=256, seed=0):
    A = random_involution_fixing_ones(d, d // 2 + 5, seed=seed + 1)
    perms = [random_pairing(h, h // 3 + 5 * b, seed=seed * 100 + 50 + b) for b in range(L)]
    heads = [random_head_structure(4, 16, 1 + (b % 2), seed=seed * 100 + 90 + b) for b in range(L)]
    return A, perms, heads


def uint8_loader(n=6, seed=2):
    imgs = torch.from_numpy(np.random.default_rng(seed).integers(0, 256, (n, 3, 56, 56), dtype=np.uint8))
    return [(imgs[:3], torch.zeros(3, dtype=torch.long)), (imgs[3:], torch.zeros(n - 3, dtype=torch.long))], imgs


@pytest.fixture(scope="module")
def ad():
    return TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float64)


@pytest.fixture(scope="module")
def st(ad):
    loader, _ = uint8_loader()
    return B.collect_stats(ad, loader, "cpu", range(L))


@pytest.fixture(scope="module")
def x64():
    return torch.randn(3, 3, 56, 56, generator=torch.Generator().manual_seed(0), dtype=torch.float64)


def model_hash(model) -> str:
    h = hashlib.sha1()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def stats_hash(st) -> str:
    h = hashlib.sha1()
    for name in B.V2_STATS:
        for k, v in sorted(getattr(st, name).items()):
            h.update(f"{name}{k}".encode())
            h.update(v.contiguous().numpy().tobytes())
    return h.hexdigest()


def max_diff(ad, x, m) -> float:
    """Largest difference of the final-norm CLS feature and of the last stream centred over channels (a folded
    model, as SliceGPT with attention returns, carries the original stream minus its channel mean)."""
    def centred(t):
        return t - t.mean(-1, keepdim=True)

    with torch.no_grad():
        a = (centred(ad.forward_tokens(x)) - centred(ad.forward_tokens(x, m))).abs().max().item()
        b = (ad.cls_feature(x) - ad.cls_feature(x, m)).abs().max().item()
    return max(a, b)


# ---------------------------------------------------------------- statistics
def test_collect_stats_matches_existing_and_direct(ad, st):
    loader, imgs = uint8_loader()
    old = B.collect_baseline_stats(ad, loader, "cpu", range(L))
    for name in ("M1", "M2", "abs_g", "S"):
        for b in range(L):
            assert torch.equal(getattr(st, name)[b], getattr(old, name)[b]), (name, b)
    assert st.n_tokens == old.n_tokens == 6 * ad.n_tokens and st.n_images == 6
    d, h, D, H, dh = 64, 256, 64, 4, 16
    ctx, qs = {}, {}
    hooks = [blk.attn.proj.register_forward_pre_hook(lambda m_, a, b=b: ctx.__setitem__(b, a[0].reshape(-1, D)))
             for b, blk in enumerate(ad.model.blocks)]
    hooks += [blk.attn.qkv.register_forward_hook(lambda m_, a, o, b=b: qs.__setitem__(b, o[..., :D].reshape(-1, D)))
              for b, blk in enumerate(ad.model.blocks)]

    def close(a, b_, tol=1e-9):
        return (a - b_).abs().max().item() <= tol * max(b_.abs().max().item(), 1e-30)

    try:
        with torch.no_grad():
            t = ad.layer0_tokens(normalize_for(ad, imgs))
            for b, blk in enumerate(ad.model.blocks):
                X = t.reshape(-1, d)
                Xc = X - X.mean(1, keepdim=True)
                n = F.layer_norm(X, (d,), eps=blk.norm1.eps)
                assert close(st.Sin[b], Xc.T @ Xc) and close(st.Nin[b], n.T @ n) and close(st.abs_n[b], n.abs().sum(0))
                mid = t + blk.ls1(blk.attn(blk.norm1(t)))
                u = F.layer_norm(mid.reshape(-1, d), (d,), eps=blk.norm2.eps)
                g = blk.mlp.act(blk.mlp.fc1(blk.norm2(mid))).reshape(-1, h)
                assert close(st.abs_u[b], u.abs().sum(0))
                assert close(st.sum_g[b], g.sum(0)) and close(st.sq_g[b], (g * g).sum(0))
                c, q = ctx[b], qs[b]
                assert close(st.Mctx[b], c.T @ c) and close(st.sum_ctx[b], c.sum(0))
                assert close(st.sq_ctx[b], (c * c).sum(0)) and close(st.abs_ctx[b], c.abs().sum(0))
                assert close(st.sum_q[b], q.sum(0))
                for hh in range(H):
                    sl = slice(hh * dh, (hh + 1) * dh)
                    assert close(st.Mq_h[b][hh], q[:, sl].T @ q[:, sl])
                    assert close(st.Mctx_h[b][hh], c[:, sl].T @ c[:, sl])
                t = blk(t)
            X = t.reshape(-1, d)
            Xc = X - X.mean(1, keepdim=True)
            n = F.layer_norm(X, (d,), eps=ad.model.norm.eps)
            assert close(st.Sin[L], Xc.T @ Xc) and close(st.Nin[L], n.T @ n)
    finally:
        for hd in hooks:
            hd.remove()
    held = sum(v.numel() for name in B.V2_STATS for v in getattr(st, name).values())
    assert 8 * held == B.stats_memory_bytes(BU.model_dims(ad), L, B.METHODS_V2, ("mlp", "attn"))
    only = B.collect_stats(ad, loader, "cpu", [1], methods=["flap"], components=("mlp",))
    assert set(only.sum_g) == {1} and not only.M2 and not only.Nin and not only.Mctx


# ---------------------------------------------------------------- budget rule, structure, no side effects
CASES = [(("mlp",), [0, 1]), (("mlp", "attn"), [0, 1]), (("mlp", "attn"), [0, 1, 2, 3]), (("attn",), [1, 2])]


@pytest.mark.parametrize("comps,blocks", CASES, ids=[f"{'+'.join(c)}-{len(b)}" for c, b in CASES])
def test_every_method_meets_the_budget_and_structure(ad, st, x64, comps, blocks):
    A, perms, heads = structure()
    bud = BU.mirror_exact_budget(ad, A, blocks, perms, heads, comps)
    h0, s0 = model_hash(ad.model), stats_hash(st)
    for m in B.METHODS_V2:
        assert 1 <= len(B.HPARAM_GRIDS[m]) <= 3
        for hp in B.HPARAM_GRIDS[m]:
            model, info = B.compress(ad.model, m, blocks, comps, bud, st, hp, verbose=False)
            tag = (m, hp, comps, blocks)
            assert info["gap_ok"] and info["gap"] >= 0, (tag, info["gap"], info["gap_note"])
            assert info["gap"] <= max(0.01, info["gap_step"] / info["budget_total"]) + 1e-12, tag
            if info["gap_reachable"]:
                assert info["gap"] <= 0.01, tag
            assert info["params_total"] == info["params_matrices"] + info["params_basis_change"]
            assert info["budget_total"] == bud.total and info["params_biases"] == bud.biases_dense
            assert sorted((p["block"], p["name"]) for p in info["per_matrix"]) == \
                sorted((x.block, x.name) for x in bud.matrices)
            assert all(p["kind"] in ("lowrank", "width", "slice", "merge") for p in info["per_matrix"])
            sc = B.structure_check(model, info)
            assert sc["ok"], (tag, [c for c in sc["checks"] if not c["ok"]][:3])
            assert sc["params_realised"] == info["params_matrices"]
            assert model_hash(ad.model) == h0 and stats_hash(st) == s0, tag      # nothing modified
            d = max_diff(ad, x64, model)
            assert np.isfinite(d) and d > 1e-6, tag                               # it did compress
            if m in ("zipit", "flap", "slicegpt") or hp:
                continue
            again, _ = B.compress(ad.model, m, blocks, comps, bud, st, hp, verbose=False)
            assert model_hash(again) == model_hash(model), tag                    # deterministic


def test_compress_rejects_a_mismatched_budget(ad, st):
    A, perms, heads = structure()
    bud = BU.mirror_exact_budget(ad, A, [0, 1], perms, heads)
    with pytest.raises(ValueError):
        B.compress(ad.model, "svd", [0, 1, 2], ("mlp", "attn"), bud, st)
    with pytest.raises(ValueError):
        B.compress(ad.model, "svd", [0, 1], ("mlp",), bud, st)
    with pytest.raises(ValueError):
        B.compress(ad.model, "asvd_scale", [0, 1], ("mlp", "attn"), bud, st, {"rank": 3})


# ---------------------------------------------------------------- full size reproduces the original
@pytest.mark.parametrize("comps", [("mlp", "attn"), ("mlp",)])
def test_full_size_budget_reproduces_the_original(ad, st, x64, comps):
    blocks = list(range(L))
    bud = BU.dense_budget(ad, blocks, comps)
    for m in B.METHODS_V2:
        for hp in B.HPARAM_GRIDS[m]:
            b_ = bud
            if m == "slicegpt" and hp["basis"] == "per_block":
                # SliceGPT pays for its shortcut (or rotation) matrices; give it room for them so that d_s = d
                lay = BU._slice_layout(bud, "per_block")
                extra = lay["basis_params"](64)
                b_ = dataclasses.replace(bud, basis_change=extra, total=bud.total + extra)
            model, info = B.compress(ad.model, m, blocks, comps, b_, st, hp, verbose=False)
            assert info["gap"] == 0.0 or (m == "flap" and info["gap"] <= 0.0), (m, hp, info["gap"])
            assert max_diff(ad, x64, model) < 1e-8, (m, hp)


# ---------------------------------------------------------------- agreement with the existing methods
def test_svdllm_equals_existing_asvd_on_the_mlp(ad, st):
    A, perms, heads = structure()
    blocks = [0, 1, 2]
    loader, _ = uint8_loader()
    old = B.collect_baseline_stats(ad, loader, "cpu", blocks)
    bud = BU.mirror_exact_budget(ad, A, blocks, perms, None, ("mlp",))
    model, info = B.compress(ad.model, "svdllm", blocks, ("mlp",), bud, st, verbose=False)
    ranks = {(p["block"], p["name"]): p["rank"] for p in info["per_matrix"]}
    alloc = [(ranks[(b, "fc1")], ranks[(b, "fc2")]) for b in blocks]
    ref, _ = B.baseline_copy(ad.model, "asvd", blocks, alloc, old)
    for b in blocks:
        for name in ("fc1", "fc2"):
            assert torch.equal(getattr(model.blocks[b].mlp, name).weight, getattr(ref.blocks[b].mlp, name).weight)
            assert torch.equal(getattr(model.blocks[b].mlp, name).bias, getattr(ref.blocks[b].mlp, name).bias)


def test_slicegpt_mlp_only_stream_pca_equals_existing_slicegpt(ad, st):
    A, perms, heads = structure()
    blocks = [0, 1]
    loader, _ = uint8_loader()
    old = B.collect_baseline_stats(ad, loader, "cpu", blocks)
    bud = BU.mirror_exact_budget(ad, A, blocks, perms, None, ("mlp",))
    model, info = B.compress(ad.model, "slicegpt", blocks, ("mlp",), bud, st, {"basis": "per_block", "pca": "stream"},
                             verbose=False)
    ds = info["per_matrix"][0]["d_s"]
    assert info["params_basis_change"] == 64 * 64 * len(blocks)           # one rotation, plus the exit at block 2
    ref, _ = B.baseline_copy(ad.model, "slicegpt", blocks, [(ds, ds)] * len(blocks), old)
    for b in blocks:
        for name in ("fc1", "fc2"):
            assert (getattr(model.blocks[b].mlp, name).weight - getattr(ref.blocks[b].mlp, name).weight).abs().max() \
                < 1e-12


# ---------------------------------------------------------------- low-rank details
def test_lowrank_factors_and_asvd_scale_formula(ad, st):
    A, perms, heads = structure()
    blocks = [0, 1]
    bud = BU.mirror_exact_budget(ad, A, blocks, perms, heads)
    for m, hp in (("svd", {}), ("svdllm", {}), ("asvd_scale", {"alpha": 0.25})):
        model, info = B.compress(ad.model, m, blocks, ("mlp", "attn"), bud, st, hp, verbose=False)
        for p in info["per_matrix"]:
            W = B._read_matrix(model, p["block"], p["name"])
            U, V = info["tensors"][f"{p['block']}.{p['name']}.U"], info["tensors"][f"{p['block']}.{p['name']}.V"]
            assert U.shape[1] == p["rank"] == V.shape[0]
            assert (U.double() @ V.double() - W).abs().max() < 1e-5 * W.abs().max()   # factors kept in float32
            orig = fold_mlp(ad.model.blocks[p["block"]]) if p["name"] in ("fc1", "fc2") else \
                fold_attn(ad.model.blocks[p["block"]])
            Wo = orig[{"fc1": "W1", "fc2": "W2", "wq": "Wq", "wk": "Wk", "wv": "Wv", "wo": "Wo"}[p["name"]]]
            if m == "svd":
                assert (W - svd_truncate(Wo, p["rank"])).abs().max() < 1e-12
            if m == "asvd_scale":
                mom, absn, key = B._input_stats(p["name"], p["block"])
                s = (getattr(st, absn)[key] / st.n_tokens).pow(0.25)
                Uu, S, Vh = torch.linalg.svd(Wo * s[None, :], full_matrices=False)
                r = p["rank"]
                assert (W - ((Uu[:, :r] * S[:r]) @ Vh[:r]) / s[None, :]).abs().max() < 1e-10 * Wo.abs().max()


# ---------------------------------------------------------------- FLAP
def test_flap_keeps_top_scores_and_compensates_bias(ad, st, x64):
    A, perms, heads = structure()
    blocks = [0, 1]
    bud = BU.mirror_exact_budget(ad, A, blocks, perms, heads, ("mlp",))
    model, info = B.compress(ad.model, "flap", blocks, ("mlp",), bud, st, {"allocation": "uniform"}, verbose=False)
    ref = copy.deepcopy(ad.model)
    n = st.n_tokens
    for b in blocks:
        f = fold_mlp(ad.model.blocks[b])
        var = (st.sq_g[b] - st.sum_g[b] ** 2 / n) / (n - 1)
        score = var * f["W2"].pow(2).sum(0)
        p = next(q for q in info["per_matrix"] if q["block"] == b and q["name"] == "fc2")
        keep = sorted(torch.argsort(-score, stable=True)[:p["n"]].tolist())
        assert p["kept"] == keep
        gone = sorted(set(range(256)) - set(keep))
        b2 = f["b2"] + f["W2"][:, gone] @ (st.sum_g[b][gone] / n)
        assert (model.blocks[b].mlp.fc2.bias - b2).abs().max() < 1e-12
        # the pruned MLP equals the dense MLP with each removed neuron replaced by its mean output
        W2 = f["W2"].clone()
        W2[:, gone] = 0
        B.write_absorbed_mlp(ref.blocks[b], f["W1"], f["b1"], W2, b2)
    with torch.no_grad():
        assert (ad.cls_feature(x64, model) - ad.cls_feature(x64, ref)).abs().max() < 1e-10


def test_flap_bias_compensation_is_exact_for_constant_units(st, x64):
    """A neuron with a constant output and a head with a constant context have zero fluctuation; FLAP removes them
    first and its bias compensation makes the pruned model equal to the original."""
    ad2 = TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float64)
    with torch.no_grad():
        ad2.model.blocks[0].mlp.fc1.weight[7] = 0.0                       # neuron 7: GELU(b1_7), constant
        ad2.model.blocks[0].attn.qkv.weight[2 * 64 + 16:2 * 64 + 32] = 0.0  # head 1: v = b_v, context constant
    loader, _ = uint8_loader()
    st2 = B.collect_stats(ad2, loader, "cpu", [0], methods=["flap"])
    full = BU.dense_budget(ad2, [0], ("mlp",))
    bud = dataclasses.replace(full, total=full.total - 2 * 64)           # one neuron fewer
    model, info = B.compress(ad2.model, "flap", [0], ("mlp",), bud, st2, {"allocation": "global"}, verbose=False)
    assert 7 not in info["per_matrix"][0]["kept"] and info["per_matrix"][0]["n"] == 255
    assert max_diff(ad2, x64, model) < 1e-10
    full = BU.dense_budget(ad2, [0], ("attn",))
    bud = dataclasses.replace(full, total=full.total - 4 * 16 * 64)      # one head fewer
    model, info = B.compress(ad2.model, "flap", [0], ("attn",), bud, st2, {"allocation": "global"}, verbose=False)
    assert info["per_matrix"][0]["heads_kept"] == [0, 2, 3]
    assert B.structure_check(model, info)["ok"]
    assert max_diff(ad2, x64, model) < 1e-10


# ---------------------------------------------------------------- ZipIt
def test_zipit_greedy_matching():
    C = torch.tensor([[1, .9, .1, .2], [.9, 1, .3, .1], [.1, .3, 1, .8], [.2, .1, .8, 1]], dtype=torch.float64)
    assert B.zipit_groups(C, 2, 0.0) == [[0, 1], [2, 3]]
    assert B.zipit_groups(C, 3, 1.0) == [[0, 1], [2], [3]]
    assert B.zipit_groups(C, 1, 0.0) == [[0, 1, 2, 3]]
    assert B.zipit_groups(C, 4, 0.5) == [[0], [1], [2], [3]]
    # repeated matching: the merged pair {0, 1} scores alpha * min(0.85, 0.8) against feature 2
    C = torch.tensor([[1, .9, .85, .1], [.9, 1, .8, .1], [.85, .8, 1, .5], [.1, .1, .5, 1]], dtype=torch.float64)
    assert B.zipit_groups(C, 2, 1.0) == [[0, 1, 2], [3]]
    assert B.zipit_groups(C, 2, 0.1) == [[0, 1], [2, 3]]
    M = torch.randn(50, 6, dtype=torch.float64)
    Cm = B.corr_from_moments(M.T @ M, M.sum(0), 50)
    assert (Cm - torch.corrcoef(M.T)).abs().max() < 1e-12


def test_zipit_merge_rules(ad, st):
    A, perms, heads = structure()
    blocks = [0, 1]
    bud = BU.mirror_exact_budget(ad, A, blocks, perms, heads)
    model, info = B.compress(ad.model, "zipit", blocks, ("mlp", "attn"), bud, st, {"alpha": 0.1}, verbose=False)
    for b in blocks:
        f, fa = fold_mlp(ad.model.blocks[b]), fold_attn(ad.model.blocks[b])
        pm = {p["name"]: p for p in info["per_matrix"] if p["block"] == b}
        g = pm["fc1"]["groups"]
        assert sorted(i for grp in g for i in grp) == list(range(256)) and len(g) == pm["fc1"]["n"]
        mlp = model.blocks[b].mlp
        for k, grp in enumerate(g):
            assert (mlp.fc1.weight[k] - f["W1"][grp].mean(0)).abs().max() < 1e-12
            assert abs(float(mlp.fc1.bias[k] - f["b1"][grp].mean())) < 1e-12
            assert (mlp.fc2.weight[:, k] - f["W2"][:, grp].sum(1)).abs().max() < 1e-12
        D = 64
        qkv, proj = model.blocks[b].attn.qkv, model.blocks[b].attn.proj
        for grp in pm["wq"]["groups"]:
            r = grp[0]
            assert (qkv.weight[r] - fa["Wq"][grp].mean(0)).abs().max() < 1e-12
            assert (qkv.weight[D + r] - fa["Wk"][grp].sum(0)).abs().max() < 1e-12
            assert all(float(qkv.weight[i].abs().max()) == 0 for i in grp[1:])
        for grp in pm["wv"]["groups"]:
            r = grp[0]
            assert (qkv.weight[2 * D + r] - fa["Wv"][grp].mean(0)).abs().max() < 1e-12
            assert (proj.weight[:, r] - fa["Wo"][:, grp].sum(1)).abs().max() < 1e-12
        assert all(len({i // 16 for i in grp}) == 1 for grp in pm["wq"]["groups"] + pm["wv"]["groups"])   # in-head


def test_zipit_is_exact_for_duplicated_neurons(x64):
    ad2 = TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float64)
    with torch.no_grad():
        fc1 = ad2.model.blocks[1].mlp.fc1
        fc1.weight[9] = fc1.weight[4]
        fc1.bias[9] = fc1.bias[4]
    loader, _ = uint8_loader()
    st2 = B.collect_stats(ad2, loader, "cpu", [1], methods=["zipit"], components=("mlp",))
    full = BU.dense_budget(ad2, [1], ("mlp",))
    bud = dataclasses.replace(full, total=full.total - 2 * 64)
    model, info = B.compress(ad2.model, "zipit", [1], ("mlp",), bud, st2, verbose=False)
    assert [4, 9] in info["per_matrix"][0]["groups"]
    assert max_diff(ad2, x64, model) < 1e-10


# ---------------------------------------------------------------- SliceGPT: dense form equals the sliced network
def _sliced_block_forward(blk, xs, Qa, Qm, Qn, H=4, dh=16, d=64):
    """Run one SlicedBlock in sliced coordinates from its weights and bases (what a healing rebuild would do)."""
    def rms(z):
        return z / torch.sqrt(z.pow(2).sum(-1, keepdim=True) / d + blk.norm1.eps)

    qkv, proj, fc1, fc2 = blk.attn.qkv, blk.attn.proj, blk.mlp.fc1, blk.mlp.fc2
    Wqkv_s, Wo_s, bo_s = qkv.weight @ Qa, Qm.T @ proj.weight, proj.bias @ Qm
    W1_s, W2_s, b2_s = fc1.weight @ Qm, Qn.T @ fc2.weight, fc2.bias @ Qn
    Bn, T, _ = xs.shape
    y = (rms(xs) @ Wqkv_s.T + qkv.bias).reshape(Bn, T, 3, H, dh).permute(2, 0, 3, 1, 4)
    att = torch.softmax(y[0] @ y[1].transpose(-1, -2) / dh ** 0.5, dim=-1) @ y[2]
    a = att.transpose(1, 2).reshape(Bn, T, H * dh) @ Wo_s.T + bo_s
    xm = xs @ (Qa.T @ Qm) + a                                             # attention shortcut, d_s x d_s
    m = blk.mlp.act(rms(xm) @ W1_s.T + fc1.bias) @ W2_s.T + b2_s
    return xm @ (Qm.T @ Qn) + m                                           # MLP shortcut


def test_slicegpt_dense_form_equals_the_sliced_network(ad, st, x64):
    A, perms, heads = structure()
    blocks = [0, 1]
    bud = BU.mirror_exact_budget(ad, A, blocks, perms, heads)
    model, info = B.compress(ad.model, "slicegpt", blocks, ("mlp", "attn"), bud, st, verbose=False)
    T = info["tensors"]
    ds = info["per_matrix"][0]["d_s"]
    assert ds < 64 and all(isinstance(model.blocks[b], B.SlicedBlock) for b in blocks)
    names = {(p["block"], p["name"]): p["basis_key"] for p in info["per_matrix"]}
    with torch.no_grad():
        t = ad.layer0_tokens(x64, model)
        dense = t
        for b in blocks:
            dense = model.blocks[b](dense)
        xs = t @ T[names[(0, "wq")]]                                      # entry: embeddings rotated and sliced
        for b in blocks:
            xs = _sliced_block_forward(model.blocks[b], xs, T[names[(b, "wq")]], T[names[(b, "wo")]],
                                       T[names[(b, "fc2")]])
        back = xs @ T[names[(blocks[-1], "fc2")]].T                        # exit map d_s x d (counted)
    assert (back - dense).abs().max() < 1e-10
    assert info["params_basis_change"] == 2 * len(blocks) * ds * ds + ds * 64


# ---------------------------------------------------------------- structure_check is not vacuous
def test_structure_check_catches_violations(ad, st):
    A, perms, heads = structure()
    bud = BU.mirror_exact_budget(ad, A, [0, 1], perms, heads)
    model, info = B.compress(ad.model, "svdllm", [0, 1], ("mlp", "attn"), bud, st, verbose=False)
    assert B.structure_check(model, info)["ok"]
    with torch.no_grad():
        model.blocks[0].mlp.fc1.weight.add_(1e-3 * torch.randn_like(model.blocks[0].mlp.fc1.weight))
    assert not B.structure_check(model, info)["ok"]
    model, info = B.compress(ad.model, "flap", [0, 1], ("mlp", "attn"), bud, st, verbose=False)
    gone = sorted(set(range(4)) - set(next(p for p in info["per_matrix"] if p["name"] == "wq" and p["block"] == 0)
                                      ["heads_kept"]))
    assert gone
    with torch.no_grad():
        model.blocks[0].attn.qkv.weight[gone[0] * 16, 0] = 1.0
    assert not B.structure_check(model, info)["ok"]
    model, info = B.compress(ad.model, "slicegpt", [0, 1], ("mlp", "attn"), bud, st, verbose=False)
    with torch.no_grad():
        model.blocks[1].mlp.fc1.weight.add_(1e-3)
    assert not B.structure_check(model, info)["ok"]
