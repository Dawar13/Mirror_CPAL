"""M1 runnable experiments: V-exact construction, registers, attention constraints in healing, and the
--tiny CLI paths of scripts/m1_attention.py and scripts/heal.py (CPU, tiny random models)."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mirror.adapters import Dinov2Adapter, TimmTinyAdapter, ViTAdapter, _randomise_affine
from mirror.attention import apply_attention_surgery, attention_free_params, attn_branch, attn_dims
from mirror.data import make_loader
from mirror.fit import fit_involution
from mirror.fold import fold_copy
from mirror.heal import HealConfig, HealData, MirrorMLPConstraint, heal
from mirror.heal_attn import LowRankAttention, LowRankQKV, MirrorAttnConstraint
from mirror.knn import evaluate_knn
from mirror.m1 import (RegisterScores, apply_attention_svd, build_vexact, feature_invariance, fit_span_operator,
                       match_registers, materialise_pos_embed, matched_random_structure,
                       random_head_maps_like, random_head_structure, random_involution_fixing_ones,
                       stream_equivariance)
from mirror.matching import random_pairing
from mirror.surgery import apply_surgery_two

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_cli", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def structure(adapter, seed=0, n_pairs_heads=(1, 2, 0, 1)):
    d, h, L = adapter.embed_dim, adapter.mlp_hidden, adapter.depth
    H, dh = attn_dims(adapter.model.blocks[0])
    A = random_involution_fixing_ones(d, d // 2 + 4, seed=seed + 3)
    perms = [random_pairing(h, h // 3, seed=seed + 10 + b) for b in range(L)]
    maps = [random_head_structure(H, dh, n_pairs_heads[b % len(n_pairs_heads)], seed=seed + 20 + b) for b in range(L)]
    return A, perms, maps


def images(n, size=56, seed=0, dtype=torch.float64):
    return torch.randn(n, 3, size, size, dtype=dtype, generator=torch.Generator().manual_seed(seed))


def span_errors(rows, k):
    inside = max(r["rel_max"] for r in rows if r["layer"] <= k)
    outside = [r["rel_max"] for r in rows if r["layer"] > k]
    return inside, outside


# ---------------------------------------------------------------- span operator
def test_fit_span_operator_energy_normalised_and_fixes_ones():
    rng = np.random.default_rng(0)
    d, n = 12, 4
    Pc = np.eye(d) - 1.0 / d
    X = [rng.standard_normal((200, d)) @ Pc for _ in range(n)]
    Xt = [x @ random_involution_fixing_ones(d, 7, seed=5) + 0.1 * rng.standard_normal((200, d)) @ Pc for x in X]
    C = np.stack([x.T @ xt for x, xt in zip(X, Xt)]).astype(np.float32)
    G = np.stack([x.T @ x for x in X]).astype(np.float32)
    A, info = fit_span_operator(C, range(n), G)
    assert np.abs(A @ A - np.eye(d)).max() < 1e-12 and np.abs(A - A.T).max() < 1e-14
    assert np.abs(A @ np.ones(d) - np.ones(d)).max() < 1e-10           # A* 1 = 1 exactly (re-centred C)
    assert info["d_plus"] + info["d_minus"] == d
    # Each layer is divided by its energy: scaling one layer's statistics leaves A* unchanged.
    C2, G2 = C.copy(), G.copy()
    C2[1] *= 1000.0
    G2[1] *= 1000.0
    A2, _ = fit_span_operator(C2, range(n), G2)
    assert np.abs(A2 - A).max() < 1e-6
    # Same as fit.fit_involution on the normalised, centred sum.
    S = sum(Pc @ C[l].astype(np.float64) @ Pc / np.trace(G[l].astype(np.float64)) for l in range(n))
    assert np.abs(fit_involution(S)[0] - A).max() < 1e-8


# ---------------------------------------------------------------- V-exact on the tiny model
@pytest.mark.parametrize("k", [1, 2, 3])
def test_vexact_tiny_is_exactly_equivariant_over_the_span(tiny64, k):
    A, perms, maps = structure(tiny64)
    x = images(3)
    before = {n: v.clone() for n, v in tiny64.model.state_dict().items()}
    m, info = build_vexact(tiny64, A, list(range(k)), perms[:k], maps[:k])
    assert all(torch.equal(v, before[n]) for n, v in tiny64.model.state_dict().items())   # input untouched
    rows = stream_equivariance(tiny64, m, x, A)
    inside, outside = span_errors(rows, k)
    assert inside < 1e-10, rows                                      # mirrored features = A* x original
    assert outside and min(outside) > 1e-6                           # teeth: unconverted blocks break it
    assert set(info["embedding_shares"]) >= {"patch_embed.weight", "pos_embed.patches", "cls_token"}
    # The original (folded, unprojected) model is not equivariant even at the embeddings.
    folded, _ = fold_copy(tiny64.model, final_norm_affine=False)
    assert stream_equivariance(tiny64, folded, x, A)[0]["rel_max"] > 1e-3


def test_vexact_full_conversion_flip_consistency_100(tiny64):
    A, perms, maps = structure(tiny64, seed=1)
    L = tiny64.depth
    m, _ = build_vexact(tiny64, A, list(range(L)), perms, maps, even_head=True)
    x = images(4, seed=2)
    rows = stream_equivariance(tiny64, m, x, A)
    assert max(r["rel_max"] for r in rows) < 1e-10                   # every layer, embeddings to the last block
    assert feature_invariance(tiny64, m, x)["rel_max"] < 1e-10       # CLS feature of the mirror = original's
    # k-NN flip consistency on random inputs: 100%.
    rng = np.random.default_rng(3)
    imgs = rng.integers(0, 256, size=(48, 56, 56, 3), dtype=np.uint8)
    labels = np.repeat(np.arange(4), 12)
    bank = make_loader(imgs, labels, np.arange(32), 16)
    query = make_loader(imgs, labels, np.arange(32, 48), 16)
    res = evaluate_knn(tiny64, bank, query, "cpu", 4, k=5, model=m)
    assert res["flip_consistency"] == 100.0
    # Teeth: without the even head the features of the mirror are not equal to the original's.
    m2, _ = build_vexact(tiny64, A, list(range(L)), perms, maps, even_head=False)
    assert feature_invariance(tiny64, m2, x)["rel_max"] > 1e-3


def _timm_with_registers(n_reg=2, seed=0):
    from timm.models.vision_transformer import VisionTransformer

    torch.manual_seed(seed)
    model = VisionTransformer(img_size=56, patch_size=14, embed_dim=64, depth=3, num_heads=4, init_values=1e-5,
                              reg_tokens=n_reg)
    _randomise_affine(model, seed)
    with torch.no_grad():
        g = torch.Generator().manual_seed(seed + 1)
        for p in (model.cls_token, model.reg_token, model.pos_embed):
            p.copy_(0.5 * torch.randn(p.shape, generator=g) + 0.2)
    model = model.double()
    ad = TimmTinyAdapter.__new__(TimmTinyAdapter)
    ViTAdapter.__init__(ad, model, img_size=56, patch_size=14, n_prefix_tokens=int(model.num_prefix_tokens))
    return ad


@pytest.mark.parametrize("reg_perm", [[1, 0], [0, 1]])
def test_vexact_timm_registers_exact(reg_perm):
    ad = _timm_with_registers()
    A, perms, maps = structure(ad, seed=4)
    x = images(3, seed=5)
    m, info = build_vexact(ad, A, [0, 1], perms[:2], maps[:2], reg_perm=reg_perm)
    inside, outside = span_errors(stream_equivariance(ad, m, x, A, reg_perm=reg_perm), 2)
    assert inside < 1e-10 and min(outside) > 1e-6
    assert "registers" in info["embedding_shares"] and "pos_embed.registers" in info["embedding_shares"]
    # Checking with the other register pairing fails: the pairing is really used.
    other = [1, 0] if reg_perm == [0, 1] else [0, 1]
    assert stream_equivariance(ad, m, x, A, reg_perm=other)[0]["rel_max"] > 1e-3


class _MockDinoAdapter(Dinov2Adapter):
    def __init__(self, model, img_size):
        ViTAdapter.__init__(self, model, img_size=img_size, patch_size=model.patch_size,
                            n_prefix_tokens=1 + model.num_register_tokens)


@pytest.mark.parametrize("reg_perm", [[1, 0, 2, 3], [0, 1, 2, 3], [3, 2, 1, 0]])
def test_vexact_dinov2_mock_registers_and_interpolated_positions(reg_perm):
    from test_fold import make_mock

    # Position embedding stored for a 5 x 5 grid, images give 4 x 4: DINOv2's bicubic interpolation
    # with interpolate_offset (not flip-symmetric) is materialised first.
    mock = make_mock(num_register_tokens=4, img_size=70)
    ad = _MockDinoAdapter(mock, 56)
    x = images(3, seed=6)
    m0 = copy.deepcopy(mock)
    assert materialise_pos_embed(m0, ad) and m0.pos_embed.shape[1] == 1 + 16
    assert (ad.cls_feature(x, m0) - ad.cls_feature(x)).abs().max().item() == 0.0
    A, perms, maps = structure(ad, seed=7)
    L = ad.depth
    for k in (2, L):
        m, info = build_vexact(ad, A, list(range(k)), perms[:k], maps[:k], reg_perm=reg_perm, even_head=(k == L))
        assert info["pos_materialised"]
        inside, outside = span_errors(stream_equivariance(ad, m, x, A, reg_perm=reg_perm), k)
        assert inside < 1e-10
        if k < L:
            assert min(outside) > 1e-6
        else:
            assert feature_invariance(ad, m, x)["rel_max"] < 1e-10


def test_register_matching_recovers_planted_pair():
    rng = np.random.default_rng(0)
    d, R = 32, 4
    A = random_involution_fixing_ones(d, 18, seed=1)
    reg = rng.standard_normal((R, d))
    reg -= reg.mean(1, keepdims=True)
    reg[1] = reg[0] @ A                       # registers 0 and 1 are mirror twins; 2 and 3 are not
    sc = RegisterScores({"k": (A, 0)}, 1, R, start=1)
    tok = torch.tensor(np.concatenate([np.zeros((1, d)), reg, rng.standard_normal((5, d))]))[None].repeat(3, 1, 1)
    sc.update(0, tok, tok)                    # registers are constants: identical in an image and its mirror
    S = sc.cosines("k", [0])
    assert abs(S[0, 1] - 1.0) < 1e-12
    perm = match_registers(S)
    assert list(perm[:2]) == [1, 0] and perm[2] == 2 and perm[3] == 3
    assert list(match_registers(S, margin=10.0)) == [0, 1, 2, 3]   # no pair clears a large margin


# ---------------------------------------------------------------- random structure
def test_matched_random_structure_matches_heal_script_and_counts(tiny64):
    heal_cli = load_script("heal")
    d, h, L = tiny64.embed_dim, tiny64.mlp_hidden, tiny64.depth
    A_layers = [random_involution_fixing_ones(d, 30 + 2 * l, seed=l) for l in range(L + 1)]
    perms = [random_pairing(h, 40 + b, seed=b) for b in range(L)]
    _, _, maps = structure(tiny64)
    st = {"A_layers": A_layers, "perms": perms, "mode": "per_layer", "heads": maps, "a_out_offset": 1}
    ours = matched_random_structure(A_layers, perms, seed=1, fix_ones=True, heads=maps)
    theirs = heal_cli.matched_random_structure(st, tiny64, seed=1, fix_ones=True)
    for a, b in zip(ours["A_layers"], theirs["A_layers"]):
        assert np.array_equal(a, b)
    for a, b in zip(ours["perms"], theirs["perms"]):
        assert np.array_equal(a, b)
    for a, b, hm in zip(ours["heads"], theirs["heads"], maps):
        assert np.array_equal(a.sigma, b.sigma) and np.array_equal(a.B, b.B)
        a.check()
        assert a.n_pairs == hm.n_pairs
        same = attention_free_params(d, a, A_layers[0], A_layers[1]) == attention_free_params(d, hm, A_layers[0], A_layers[1])
        assert same                                                  # matched free-parameter count
    r = random_head_maps_like(maps[1], seed=3)
    assert not np.array_equal(r.sigma, maps[1].sigma) or not np.allclose(r.B, maps[1].B)


# ---------------------------------------------------------------- healing with attention constraints
def _heal_data(n_train=16, n_held=8, seed=0):
    rng = np.random.default_rng(seed)
    imgs = rng.integers(0, 256, size=(n_train + n_held, 64, 64, 3), dtype=np.uint8)
    rows = rng.permutation(n_train + n_held)
    return HealData(imgs, train_rows=rows[:n_train], heldout_rows=rows[n_train:])


def _cfg(blocks, **kw):
    c = dict(blocks=blocks, seed=0, lr=1e-3, max_steps=4, batch_size=8, micro_batch=4, crop_size=56,
             warmup_steps=1, calib_images=8, log_every=0, prefetch=0)
    c.update(kw)
    return HealConfig(**c)


def _mirror_attn_student(adapter, blocks, offset=1, seed=0):
    d, h = adapter.embed_dim, adapter.mlp_hidden
    H, dh = attn_dims(adapter.model.blocks[0])
    A = {l: random_involution_fixing_ones(d, 24 + 4 * l, seed=100 + 10 * seed + l) for l in range(adapter.depth + 1)}
    A_in, A_mlp_out = [A[b] for b in blocks], [A[b + 1] for b in blocks]
    A_att_out = [A[b + offset] for b in blocks]
    perms = [random_pairing(h, 90, seed=200 + b) for b in blocks]
    heads = [random_head_structure(H, dh, 1 + b % 2, seed=300 + b) for b in blocks]
    student = copy.deepcopy(adapter.model)
    apply_surgery_two(student, blocks, A_in, A_mlp_out, perms)
    maps = apply_attention_surgery(student, blocks, A_in, A_att_out, heads, gauge=True)
    cons = [MirrorMLPConstraint(blocks, A_in, A_mlp_out, perms), MirrorAttnConstraint(blocks, A_in, A_att_out, maps)]
    return student, cons, (A_in, A_att_out)


def test_attention_constraint_keeps_branch_exactly_equivariant(tiny64):
    blocks = [1, 2]
    data = _heal_data()
    student, cons, (A_in, A_out) = _mirror_attn_student(tiny64, blocks)
    w0 = {b: student.blocks[b].attn.qkv.weight.clone() for b in blocks}
    n1 = {b: (student.blocks[b].norm1.weight.clone(), student.blocks[b].ls1.gamma.clone()) for b in blocks}
    res = heal(tiny64, student, data, _cfg(blocks, decay_1d=True), cons, None)
    x = torch.randn(3, 17, tiny64.embed_dim, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    for i, b in enumerate(blocks):
        blk = student.blocks[b]
        Ai, Ao = torch.tensor(A_in[i]), torch.tensor(A_out[i])
        assert (attn_branch(blk, x @ Ai) - attn_branch(blk, x) @ Ao).abs().max().item() < 1e-10
        assert (blk.attn.qkv.weight - w0[b]).abs().max().item() > 1e-5          # training moved the weights
        assert torch.equal(blk.norm1.weight, n1[b][0]) and torch.equal(blk.ls1.gamma, n1[b][1])   # frozen
        assert torch.equal(blk.norm1.weight, torch.ones_like(blk.norm1.weight))
    assert res["constraint_checks"]["1:MirrorAttnConstraint"]["max_rel_violation"] < 1e-12
    assert not any(".norm1." in n or ".ls1." in n for n in res["trainable_names"])
    # Teeth: the same training with only the MLP constraint breaks the attention equivariance.
    free, cons2, _ = _mirror_attn_student(tiny64, blocks)
    heal(tiny64, free, data, _cfg(blocks), cons2[:1], None)
    blk = free.blocks[1]
    Ai, Ao = torch.tensor(A_in[0]), torch.tensor(A_out[0])
    assert (attn_branch(blk, x @ Ai) - attn_branch(blk, x) @ Ao).abs().max().item() > 1e-6


def test_attention_constraint_projection_matches_surgery(tiny64):
    """Called on the freshly converted student, the constraint changes nothing (idempotent)."""
    student, cons, _ = _mirror_attn_student(tiny64, [0, 3], offset=0)
    before = {k: v.clone() for k, v in student.state_dict().items()}
    cons[1].prepare(student)
    with torch.no_grad():
        cons[1](student)
    for k, v in student.state_dict().items():
        assert (v - before[k]).abs().max().item() < 1e-12, k
    with pytest.raises(ValueError):
        cons[1].prepare(copy.deepcopy(tiny64.model))                 # not absorbed: refused


def test_low_rank_attention_keeps_exact_ranks(tiny64, images64):
    blocks, ranks = [0, 2], [(10, 12, 9, 11), (5, 6, 7, 8)]
    student = copy.deepcopy(tiny64.model)
    apply_attention_svd(student, blocks, ranks)
    dense = copy.deepcopy(student)
    con = LowRankAttention(blocks, ranks)
    con.prepare(student)
    con.prepare(student)                                             # idempotent
    assert isinstance(student.blocks[0].attn.qkv, LowRankQKV)
    assert (tiny64.cls_feature(images64, dense) - tiny64.cls_feature(images64, student)).abs().max().item() < 1e-10
    u0 = student.blocks[0].attn.qkv.q.U.clone()
    res = heal(tiny64, student, _heal_data(), _cfg(blocks), [con], None)
    assert res["constraint_checks"]["0:LowRankAttention"]["numerical_ranks"] == [list(r) for r in ranks]
    assert (student.blocks[0].attn.qkv.q.U - u0).abs().max().item() > 1e-5
    W = student.blocks[2].attn.qkv.weight
    assert W.shape == (3 * 64, 64) and student.blocks[2].attn.qkv.bias.shape == (3 * 64,)


# ---------------------------------------------------------------- CLI: scripts/m1_attention.py --tiny
def _tiny_operators(path, seed=0):
    ad = TimmTinyAdapter(img_size=56, seed=0)
    d, h, L = ad.embed_dim, ad.mlp_hidden, ad.depth
    rng = np.random.default_rng(seed)
    A = np.stack([random_involution_fixing_ones(d, 30 + l, seed=l) for l in range(L + 1)]).astype(np.float32)
    C = np.stack([a + 0.3 * rng.standard_normal((d, d)) for a in A]).astype(np.float32)
    G = np.stack([np.eye(d) * (1 + l) for l in range(L + 1)]).astype(np.float32)
    perms = np.stack([random_pairing(h, 80, seed=b) for b in range(L)]).astype(np.int32)
    np.savez_compressed(path, A_layers=A, C=C, G=G, perms=perms)
    return path


def test_m1_cli_tiny_runs_and_resumes(tmp_path):
    m1 = load_script("m1_attention")
    ops = _tiny_operators(tmp_path / "ops.npz")
    out = tmp_path / "m1"
    args = ["--tiny", "--operators", str(ops), "--out", str(out), "--ks", "2,4"]
    res = m1.main(args)
    for f in ("m1_structure.npz", "m1_heads.csv", "m1_heads_detail.csv", "m1_ranks.csv", "m1_vexact.csv",
              "m1_vexact_blocks.csv", "m1_vexact_layers.csv", "m1_exactness.csv", "m1_flipcheck.json",
              "m1_knn.csv", "m1_attn_stats.npz", "m1_meta.json", "m1_a_out_choice.json", "M1_SUMMARY.md"):
        assert (out / f).is_file(), f
    meta = json.loads((out / "m1_meta.json").read_text())
    assert meta["seed"] == 0 and meta["fit_seed"] == 0 and len(meta["git"]["git_hash"]) >= 7
    assert all(meta["exactness_pass"].values()) and meta["flip_check_pass"] is True
    flip = json.loads((out / "m1_flipcheck.json").read_text())
    assert flip["even_head"]["flip_consistency"] == 100.0
    knn = res["knn"]
    assert len(knn) == 2 + 2 * 5 and set(knn["variant"]) == {"original", "folded", "vlocal", "vexact", "mlp_only",
                                                             "svd", "random"}
    assert knn["git_hash"].nunique() == 1
    # Matched budgets: SVD never keeps more than Mirror V-local; the random structure keeps the same.
    for k in (2, 4):
        kept = knn[knn["k"] == k].set_index("variant")["converted_params_kept"]
        assert kept["svd"] <= kept["vlocal"] and kept["random"] == kept["vlocal"]
        assert kept["mlp_only"] > kept["vlocal"]
    folded = knn[knn["variant"] == "folded"].iloc[0]
    orig = knn[knn["variant"] == "original"].iloc[0]
    assert folded["top1"] == orig["top1"]
    text = (out / "M1_SUMMARY.md").read_text(encoding="utf-8")
    assert "\u2014" not in text and "A_out choice" in text and "PASS" in text
    # Resume: nothing is recomputed and no row is duplicated.
    mtime = (out / "m1_structure.npz").stat().st_mtime_ns
    res2 = m1.main(args)
    assert len(res2["knn"]) == len(knn)
    assert (out / "m1_structure.npz").stat().st_mtime_ns == mtime
    # The structure file feeds scripts/heal.py (head maps and the A_out choice).
    heal_cli = load_script("heal")
    cfg = {"structure": {"heads": str(out / "m1_structure.npz"), "attn_a_out": "auto"}}
    ad = TimmTinyAdapter(img_size=56, seed=0)
    hs = heal_cli.load_heads(cfg, ad, str(ops))
    with np.load(out / "m1_structure.npz") as z:
        assert hs["a_out_offset"] == int(z["a_out_offset"])
        assert all(np.array_equal(hm.sigma, s) for hm, s in zip(hs["heads"], z["sigma"]))
    cfg["structure"]["attn_a_out"] = "same"
    assert heal_cli.load_heads(cfg, ad, str(ops))["a_out_offset"] == 0


def test_m1_cli_tiny_fits_its_own_operators(tmp_path):
    """Default --tiny path: operators and twin pairings fitted in-run (run_e1 layout), then M1."""
    m1 = load_script("m1_attention")
    out = tmp_path / "m1"
    res = m1.main(["--tiny", "--out", str(out), "--ks", "2", "--variants", "folded,vexact"])
    with np.load(out / "tiny_operators.npz") as z:
        assert z["A_layers"].shape == (5, 64, 64) and z["perms"].shape == (4, 256) and z["C"].shape == (5, 64, 64)
    assert res["meta"]["exactness_pass"] == {2: True}
    assert list(res["knn"]["variant"]) == ["original", "folded", "vexact"]


# ---------------------------------------------------------------- CLI: scripts/heal.py --tiny with attention
def test_heal_cli_tiny_with_attention_and_knn(tmp_path):
    heal_cli = load_script("heal")
    for method in ("mirror", "svd", "random"):
        s = heal_cli.main(["--config", str(ROOT / "configs" / "m3_pilot.yaml"), "--tiny", "--method", method,
                           "--out-dir", str(tmp_path)])
        info, checks = s["method_info"], s["final"]["constraint_checks"]
        assert info["attention"] and info["attn_params_kept"] <= info["attn_params_dense"]
        if method == "svd":
            assert info["attn_params_kept"] <= info["attn_params_mirror"]
            got = checks["1:LowRankAttention"]
            assert got["numerical_ranks"] == got["target_ranks"]
        else:
            assert checks["1:MirrorAttnConstraint"]["max_rel_violation"] < 1e-6     # float32 model
        k = s["knn"]
        assert {"original", "zero_shot", "healed"} <= set(k) and 0 <= k["healed"]["top1"] <= 100
        assert (tmp_path / method / "knn.json").exists()
        assert not any(".norm1." in n or ".ls1." in n for n in s["final"]["trainable_names"])
