"""M2: efficient even/odd implementation (mirror/efficient.py) and the --tiny CLI of scripts/m2_latency.py.

CPU, float64, the tiny random timm ViT (fixture tiny64) and planted structures (random involutions
with A 1 = 1, random neuron pairings, random head maps), projected with the existing M1 builders."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mirror.adapters import Dinov2Adapter, ViTAdapter
from mirror.attention import apply_attention_surgery, attention_free_params, attn_dims
from mirror.efficient import (Basis, BlockSpec, build_fast_vit, build_mirror_attention, build_mirror_mlp,
                              fast_vit_cost, prune_keep)
from mirror.fold import fold_layernorm_full
from mirror.heal_attn import read_attn
from mirror.m1 import (build_vexact, materialise_pos_embed, mlp_mirror_params, random_head_structure,
                       random_involution_fixing_ones)
from mirror.matching import random_pairing
from mirror.surgery import apply_surgery_two, equivariant_param_count, perm_trace, svd_truncate

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-10


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_cli", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tokens(n, T, d, seed=0):
    return torch.randn(n, T, d, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


def Qt(basis):
    return torch.as_tensor(basis.Q, dtype=torch.float64)


def folded(model, adapter):
    m = copy.deepcopy(model)
    materialise_pos_embed(m, adapter)
    fold_layernorm_full(m, final_norm_affine=False)
    return m


def dense_specs(L):
    return [BlockSpec() for _ in range(L)]


# ---------------------------------------------------------------- MLP
@pytest.mark.parametrize("align", [1, 8])
@pytest.mark.parametrize("d_plus_in,d_plus_out,n_pairs", [(37, 37, 90), (37, 29, 90), (40, 33, 128), (21, 50, 0)])
def test_mirror_mlp_equals_projected_dense_and_counts(tiny64, align, d_plus_in, d_plus_out, n_pairs):
    d, h = tiny64.embed_dim, tiny64.mlp_hidden
    A_in = random_involution_fixing_ones(d, d_plus_in, seed=1)
    A_out = A_in if d_plus_out == d_plus_in else random_involution_fixing_ones(d, d_plus_out, seed=2)
    perm = random_pairing(h, n_pairs, seed=3)
    m = copy.deepcopy(tiny64.model)
    apply_surgery_two(m, [1], [A_in], [A_out], [perm])
    blk = m.blocks[1]
    bi, bo = Basis.of(A_in), Basis.of(A_out)
    mod = build_mirror_mlp(blk.mlp.fc1.weight, blk.mlp.fc1.bias, blk.mlp.fc2.weight, blk.mlp.fc2.bias, blk.mlp.act,
                           bi, bo, A_in, A_out, perm, align=align, tol=TOL)
    x = tokens(3, 17, d)
    ref = blk.mlp(x) @ Qt(bo)                       # projected dense MLP, output in the output basis
    assert (mod(x @ Qt(bi)) - ref).abs().max().item() < TOL
    # Parameter counts: (h d + tr(P) tr(A)) / 2 per matrix, h+ d+ + h- d- multiply-adds.
    f = perm_trace(perm)
    want1 = (h * d + f * (2 * d_plus_in - d)) // 2
    want2 = (h * d + f * (2 * d_plus_out - d)) // 2
    assert mod.cost["fc1_params"] == want1 == equivariant_param_count(h, d, f, 2 * d_plus_in - d)
    assert mod.cost["fc2_params"] == want2
    assert mod.cost["matrix_params"] == mod.cost["macs_token"] == want1 + want2
    assert mod.cost["fc1_params"] == (n_pairs + f) * d_plus_in + n_pairs * (d - d_plus_in)      # h+ d+ + h- d-
    stored = sum(p.numel() for p in (mod.E1, mod.O1, mod.E2, mod.O2))
    assert stored == mod.cost["macs_token_exec"]
    if align == 1:
        assert stored == want1 + want2              # no padding: stored = free parameters
    else:
        assert stored >= want1 + want2


def test_mirror_mlp_rejects_unprojected_weights(tiny64):
    d, h = tiny64.embed_dim, tiny64.mlp_hidden
    A = random_involution_fixing_ones(d, 37, seed=1)
    perm = random_pairing(h, 90, seed=3)
    blk = tiny64.model.blocks[1]                    # not projected: the structure check must fail
    with pytest.raises(ValueError, match="not equivariant"):
        build_mirror_mlp(blk.mlp.fc1.weight, blk.mlp.fc1.bias, blk.mlp.fc2.weight, blk.mlp.fc2.bias, blk.mlp.act,
                         Basis.of(A), Basis.of(A), A, A, perm)
    with pytest.raises(ValueError, match="does not match"):
        B = random_involution_fixing_ones(d, 30, seed=9)
        build_mirror_mlp(blk.mlp.fc1.weight, blk.mlp.fc1.bias, blk.mlp.fc2.weight, blk.mlp.fc2.bias, blk.mlp.act,
                         Basis.of(B), Basis.of(A), A, A, perm)


# ---------------------------------------------------------------- attention
@pytest.mark.parametrize("align", [1, 8])
@pytest.mark.parametrize("n_pairs", [0, 1, 2])
@pytest.mark.parametrize("out", ["same", "fold", "rotate"])
def test_mirror_attention_equals_projected_dense_and_counts(tiny64, align, n_pairs, out):
    d = tiny64.embed_dim
    H, dh = attn_dims(tiny64.model.blocks[0])
    A_in = random_involution_fixing_ones(d, 35, seed=4)
    A_out = A_in if out == "same" else random_involution_fixing_ones(d, 30, seed=5)
    hm = random_head_structure(H, dh, n_pairs, seed=6 + n_pairs)
    m = copy.deepcopy(tiny64.model)
    maps = apply_attention_surgery(m, [2], [A_in], [A_out], [hm])
    blk = m.blocks[2]
    bi = Basis.of(A_in)
    mod, rot = build_mirror_attention(read_attn(blk), H, dh, blk.attn.scale, bi, A_in, A_out, maps[0], align,
                                      wo_mode="rotate" if out == "rotate" else "fold", tol=TOL)
    x = tokens(3, 17, d, seed=1)
    y = mod(x @ Qt(bi))
    if rot is not None:
        y = y @ rot
    assert (y - blk.attn(x) @ Qt(bi)).abs().max().item() < TOL    # same function, output in the block basis
    assert (rot is not None) == (out == "rotate")
    free = attention_free_params(d, maps[0], A_in, A_out)
    assert mod.cost["qkv_params"] == free["Wq"] + free["Wk"] + free["Wv"]
    if out == "fold":
        assert mod.cost["out_params"] == d * H * dh                    # Wo dense (rotation folded into it)
    else:
        assert mod.cost["out_params"] == free["Wo"]
        assert mod.cost["matrix_params"] == free["matrices"]
    if align == 1:
        n = sum(p.numel() for name, p in mod.named_parameters() if name in ("We", "Wodd", "Oe", "Oo", "Wd"))
        assert n == mod.cost["matrix_params"]


def test_twin_heads_computed_once():
    """With only twin pairs and A_in = A_out, q, k, v and Wo cost exactly half of the dense matrices."""
    from mirror.adapters import TimmTinyAdapter

    ad = TimmTinyAdapter(img_size=56, seed=1, randomise=True, dtype=torch.float64)
    d = ad.embed_dim
    H, dh = attn_dims(ad.model.blocks[0])
    A = random_involution_fixing_ones(d, 41, seed=7)
    hm = random_head_structure(H, dh, H // 2, seed=8)
    m = copy.deepcopy(ad.model)
    maps = apply_attention_surgery(m, [0], [A], [A], [hm])
    mod, _ = build_mirror_attention(read_attn(m.blocks[0]), H, dh, m.blocks[0].attn.scale, Basis.of(A), A, A,
                                    maps[0], tol=TOL)
    assert mod.cost["matrix_params"] == 2 * H * dh * d                  # half of 4 H dh d


# ---------------------------------------------------------------- whole model
def planted(adapter, seed=0):
    d, h, L = adapter.embed_dim, adapter.mlp_hidden, adapter.depth
    H, dh = attn_dims(adapter.model.blocks[0])
    A = random_involution_fixing_ones(d, d // 2 + 6, seed=seed + 3)
    A_layers = [random_involution_fixing_ones(d, d // 2 - 4 + 3 * l, seed=seed + 50 + l) for l in range(L + 1)]
    perms = [random_pairing(h, h // 3 + 4 * b, seed=seed + 10 + b) for b in range(L)]
    maps = [random_head_structure(H, dh, (1, 2, 0, 1)[b % 4], seed=seed + 20 + b) for b in range(L)]
    return A, A_layers, perms, maps


@pytest.mark.parametrize("align", [1, 8])
@pytest.mark.parametrize("k", [2, 4])
def test_full_tiny_vexact_equals_dense_projected(tiny64, images64, k, align):
    A, _, perms, maps = planted(tiny64)
    L, d, h = tiny64.depth, tiny64.embed_dim, tiny64.mlp_hidden
    m, info = build_vexact(tiny64, A, list(range(k)), perms[:k], maps[:k], even_head=(k == L))
    specs = [BlockSpec(("mirror", A, A, info["maps_after"][b]), ("mirror", A, A, perms[b])) if b < k else BlockSpec()
             for b in range(L)]
    basis = Basis.of(A)
    fv = build_fast_vit(m, specs, [basis] * (L + 1), align=align, tol=TOL)
    assert (fv(images64) - tiny64.cls_feature(images64, m)).abs().max().item() < TOL
    assert fv.info["n_rotations"] == 0 and fv.info["n_attn_rotations"] == 0       # V-exact: no rotations
    if k == L:
        assert fv.head.m == basis.dp                                              # even head: even rows only
    T, n_patches = tiny64.n_tokens, tiny64.grid_hw[0] * tiny64.grid_hw[1]
    c = fast_vit_cost(fv, T, n_patches)
    want = sum(mlp_mirror_params(h, d, perms[b], A, A) + attention_free_params(d, maps[b], A, A)["matrices"]
               for b in range(k))
    assert c["matrix_params_converted"] == want
    assert c["macs_rot"] == 0
    dense = build_fast_vit(folded(tiny64.model, tiny64), dense_specs(L))
    cd = fast_vit_cost(dense, T, n_patches)
    assert c["macs"] < cd["macs"]
    assert c["macs"] == cd["macs"] - T * (c["matrix_params_converted_dense"] - want) + c["macs_head"]


def test_full_dinov2_mock_vexact_registers_equals_dense_projected():
    from test_fold import make_mock

    class MockAd(Dinov2Adapter):
        def __init__(self, model, img_size):
            ViTAdapter.__init__(self, model, img_size=img_size, patch_size=model.patch_size,
                                n_prefix_tokens=1 + model.num_register_tokens)

    ad = MockAd(make_mock(num_register_tokens=4, img_size=70), 56)          # positions interpolated 5x5 -> 4x4
    A, _, perms, maps = planted(ad, seed=1)
    L = ad.depth
    x = torch.randn(3, 3, 56, 56, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    for k in (2, L):
        m, info = build_vexact(ad, A, list(range(k)), perms[:k], maps[:k], reg_perm=[1, 0, 3, 2], even_head=(k == L))
        specs = [BlockSpec(("mirror", A, A, info["maps_after"][b]), ("mirror", A, A, perms[b])) if b < k
                 else BlockSpec() for b in range(L)]
        fv = build_fast_vit(m, specs, [Basis.of(A)] * (L + 1), align=8, tol=TOL)
        assert (fv(x) - ad.cls_feature(x, m)).abs().max().item() < TOL
    dense = build_fast_vit(folded(ad.model, ad), dense_specs(L))
    assert (dense(x) - ad.cls_feature(x)).abs().max().item() < TOL


@pytest.mark.parametrize("variant", ["vlocal_same", "vlocal_fold", "vlocal_rotate", "mlp_only"])
def test_full_tiny_vlocal_equals_dense_projected(tiny64, images64, variant):
    _, A, perms, maps = planted(tiny64, seed=2)
    L, d, h = tiny64.depth, tiny64.embed_dim, tiny64.mlp_hidden
    k = 3
    off = 0 if variant == "vlocal_same" else 1
    blocks = list(range(k))
    m = copy.deepcopy(tiny64.model)
    apply_surgery_two(m, blocks, [A[b] for b in blocks], [A[b + 1] for b in blocks], perms[:k])
    if variant != "mlp_only":
        after = apply_attention_surgery(m, blocks, [A[b] for b in blocks], [A[b + off] for b in blocks], maps[:k])
    ref = tiny64.cls_feature(images64, m)                   # projected dense (unfolded, as M1 builds it)
    bases = [Basis.of(A[l]) for l in range(k + 1)] + [Basis.of(A[k]) for _ in range(L - k)]
    specs = []
    for b in range(L):
        if b >= k:
            specs.append(BlockSpec())
            continue
        attn = ("dense",) if variant == "mlp_only" else ("mirror", A[b], A[b + off], after[b])
        specs.append(BlockSpec(attn, ("mirror", A[b], A[b + 1], perms[b])))
    fv = build_fast_vit(folded(m, tiny64), specs, bases, align=8, wo_mode="rotate" if variant == "vlocal_rotate"
                        else "fold", tol=1e-9)
    assert (fv(images64) - ref).abs().max().item() < TOL
    assert fv.info["n_rotations"] == k                      # one stream rotation per converted block
    assert fv.info["n_attn_rotations"] == (k if variant == "vlocal_rotate" else 0)
    c = fast_vit_cost(fv, tiny64.n_tokens, 16)
    assert c["macs_rot"] == tiny64.n_tokens * d * d * (k + fv.info["n_attn_rotations"])
    assert c["macs_head"] == d * d                          # rotation of the CLS feature out of A_k's basis


def test_dense_fast_vit_equals_original(tiny64, images64):
    fv = build_fast_vit(folded(tiny64.model, tiny64), dense_specs(tiny64.depth))
    assert (fv(images64) - tiny64.cls_feature(images64)).abs().max().item() < TOL
    c = fast_vit_cost(fv, tiny64.n_tokens, 16)
    d, h, L, T = tiny64.embed_dim, tiny64.mlp_hidden, tiny64.depth, tiny64.n_tokens
    assert c["macs"] == 16 * 3 * 14 * 14 * d + L * (T * (4 * d * d + 2 * h * d) + 2 * T * T * d)   # cost_model.md


def test_svd_and_prune_baselines_equal_their_dense_versions(tiny64, images64):
    base = folded(tiny64.model, tiny64)
    L, d, h = tiny64.depth, tiny64.embed_dim, tiny64.mlp_hidden
    att_r, mlp_r, h_keep = [(5, 6, 7, 9), (11, 3, 8, 10)], [(10, 12), (17, 9)], [50, 83]
    for align in (1, 8):
        svd = build_fast_vit(base, [BlockSpec(("svd", att_r[b]), ("svd", mlp_r[b])) if b < 2 else BlockSpec()
                                    for b in range(L)], align=align)
        ref = copy.deepcopy(base)
        with torch.no_grad():
            for b in range(2):
                blk = ref.blocks[b]
                W = blk.attn.qkv.weight
                for i, r in enumerate(att_r[b][:3]):
                    W[i * d:(i + 1) * d] = svd_truncate(W[i * d:(i + 1) * d].clone(), r)
                blk.attn.proj.weight.copy_(svd_truncate(blk.attn.proj.weight.clone(), att_r[b][3]))
                blk.mlp.fc1.weight.copy_(svd_truncate(blk.mlp.fc1.weight.clone(), mlp_r[b][0]))
                blk.mlp.fc2.weight.copy_(svd_truncate(blk.mlp.fc2.weight.clone(), mlp_r[b][1]))
        assert (svd(images64) - tiny64.cls_feature(images64, ref)).abs().max().item() < TOL
        c = fast_vit_cost(svd, tiny64.n_tokens, 16)
        assert c["matrix_params_converted"] == sum(sum(att_r[b]) * (d + d) + sum(mlp_r[b]) * (h + d) for b in range(2))

        pr = build_fast_vit(base, [BlockSpec(("dense",), ("prune", h_keep[b])) if b < 2 else BlockSpec()
                                   for b in range(L)], align=align)
        ref = copy.deepcopy(base)
        with torch.no_grad():
            for b in range(2):
                mlp = ref.blocks[b].mlp
                keep = prune_keep(mlp.fc1.weight, mlp.fc2.weight, h_keep[b])
                drop = torch.ones(h, dtype=torch.bool)
                drop[keep] = False
                mlp.fc2.weight[:, drop] = 0.0
        assert (pr(images64) - tiny64.cls_feature(images64, ref)).abs().max().item() < TOL
        c = fast_vit_cost(pr, tiny64.n_tokens, 16)
        assert c["matrix_params_converted"] == sum(4 * d * d + 2 * h_keep[b] * d for b in range(2))


# ---------------------------------------------------------------- CLI
def test_m2_cli_tiny_runs(tmp_path):
    m2 = load_script("m2_latency")
    res = m2.main(["--tiny", "--out", str(tmp_path), "--k", "2,4", "--batches", "1,2", "--warmup", "1",
                   "--iters", "2"])
    lat = pd.read_csv(tmp_path / "m2_latency.csv")
    chk = pd.read_csv(tmp_path / "m2_checks.csv")
    assert (tmp_path / "M2_SUMMARY.md").exists() and (tmp_path / "m2_meta.json").exists()
    want = {"original", "dense", "vlocal", "vexact", "mlp_only", "svd", "prune"}
    assert want <= set(lat["variant"])
    assert set(lat["batch"]) == {1, 2}
    assert lat["latency_ms_median"].gt(0).all() and lat["git_hash"].notna().all() and (lat["seed"] == 0).all()
    # Efficient equals projected dense in float64 and fp32 for every converted variant.
    for dt in ("float64", "float32"):
        sub = chk[chk["dtype"] == dt]
        assert {"vlocal", "vexact", "mlp_only", "svd", "prune", "dense"} <= set(sub["variant"])
        assert sub["pass"].astype(bool).all(), sub
    bf = chk[chk["dtype"] == "bfloat16"]
    assert len(bf) and bf["noise_floor_unit"].gt(0).all() and "ratio_to_noise" in bf
    # Analytic FLOPs: every converted variant is cheaper than dense; V-exact pays no rotation.
    one = lat[(lat["batch"] == 1)].drop_duplicates(["variant", "k"])
    dense_macs = float(one[one["variant"] == "dense"]["macs_per_image"].iloc[0])
    conv = one[~one["variant"].isin(["original", "dense"])]
    assert (conv["macs_per_image"] < dense_macs).all()
    vx = one[one["variant"] == "vexact"]
    assert (vx["n_rotations"] == 0).all() and (vx["n_attn_rotations"] == 0).all()     # no per-token rotation
    assert (vx["rotation_macs"] <= 64 * 64).all()                                       # only the CLS feature
    assert (one[one["variant"] == "vlocal"]["n_rotations"] == one[one["variant"] == "vlocal"]["k"]).all()
    meta = json.loads((tmp_path / "m2_meta.json").read_text())
    assert meta["tiny"] and meta["seed"] == 0 and meta["git"]["git_hash"]
    text = (tmp_path / "M2_SUMMARY.md").read_text(encoding="utf-8")
    assert "smaller than the FLOP savings" in text and chr(0x2014) not in text
    assert res["out"] == str(tmp_path)
