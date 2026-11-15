"""CPU tests for CORE-04: mirror/adapters_extra.py, mirror/cnn_groups.py and scripts/core04_generality.py.

Random weights only (pretrained=False), tiny architectures, synthetic data: no downloads.
"""
import copy
import importlib.util
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # nothing here may reach the hub

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from mirror import adapters_extra as AX
from mirror import cnn_groups as CG
from mirror import groups as GR
from mirror.adapters import TimmTinyAdapter, _randomise_affine
from mirror.variants import TinyProtocol

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Tiny versions of every ViT family (timm model arguments overridden; random weights).
TINY_VIT = {
    "clip_b16": dict(img_size=64, embed_dim=64, depth=2, num_heads=2),
    "dino_b16": dict(img_size=64, embed_dim=64, depth=2, num_heads=2),
    "dinov3_b16": dict(img_size=64, embed_dim=64, depth=2, num_heads=2),
    "eva02_b14": dict(img_size=56, embed_dim=64, depth=2, num_heads=2, ref_feat_shape=(4, 4)),
}
# name -> (n_prefix, mlp kind, rope)
EXPECT = {"clip_b16": (1, "gelu", False), "dino_b16": (1, "gelu", False), "dinov3_b16": (5, "gelu", True),
          "eva02_b14": (1, "swiglu", True)}


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_t04_{name}", os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tiny_vit(name, seed=0):
    kw = dict(TINY_VIT[name])
    img = kw.pop("img_size")
    torch.manual_seed(seed)
    ad = AX.build_vit_adapter(AX.EXTRA_REGISTRY[name], img_size=img, pretrained=False, **kw)
    _randomise_affine(ad.model, seed)
    return ad, img


@torch.no_grad()
def _randomise_bn(model, seed=0):
    g = torch.Generator().manual_seed(seed + 777)
    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            n = mod.num_features
            mod.weight.copy_(1.0 + 0.3 * torch.randn(n, generator=g))
            mod.bias.copy_(0.1 * torch.randn(n, generator=g))
            mod.running_mean.copy_(0.1 * torch.randn(n, generator=g))
            mod.running_var.copy_(1.0 + 0.5 * torch.rand(n, generator=g))


def tiny_convnext(input_size=64):
    torch.manual_seed(0)
    ad = AX.build_cnn_adapter(AX.EXTRA_REGISTRY["convnext_b"], pretrained=False, input_size=input_size,
                              depths=(1, 1, 2, 1), dims=(8, 16, 16, 32))
    _randomise_affine(ad.model, 0)
    return ad


def tiny_resnet(input_size=65):
    torch.manual_seed(0)
    ad = AX.build_cnn_adapter(AX.EXTRA_REGISTRY["resnet50"], pretrained=False, input_size=input_size,
                              layers=(1, 1, 1, 1), channels=(4, 8, 8, 16), stem_width=8)
    _randomise_bn(ad.model)
    return ad


# ----------------------------------------------------------------------------------------------
# Registry and hook
# ----------------------------------------------------------------------------------------------
def test_registry_ids_match_model_yaml():
    for short, entry in AX.EXTRA_REGISTRY.items():
        y = yaml.safe_load(open(os.path.join(ROOT, entry["yaml"]), encoding="utf-8"))
        assert y["short_name"] == short
        src = y["source"]
        if "timm_name" in entry:
            assert src["timm_name"] == entry["timm_name"], short
        else:
            assert (src["hub_repo"], src["entry_point"]) == (entry["hub_repo"], entry["hub_name"]), short
        assert "(verify" in entry["note"]
        assert short in AX.SURGERY_SUPPORT


def test_adapters_timm_hook_falls_back(monkeypatch):
    import mirror.adapters_timm as T

    assert T.resolve_name("clip_b16") is AX.EXTRA_REGISTRY["clip_b16"]
    assert T.resolve_name("vit_base_patch16_224.dino") is AX.EXTRA_REGISTRY["dino_b16"]
    assert T.resolve_name("dinov2_vitg14_reg") is AX.EXTRA_REGISTRY["dinov2_vitg14_reg"]
    assert T.resolve_name("deit3_b16") is T.MODEL_REGISTRY["deit3_b16"]       # existing names unchanged
    with pytest.raises(KeyError):
        T.resolve_name("not_a_model")
    calls = []
    monkeypatch.setattr(AX, "build_vit_adapter",
                        lambda entry, img_size=224, pretrained=True: calls.append((entry["timm_name"], img_size,
                                                                                  pretrained)) or "adapter")
    assert T.build_model_adapter("dino_b16", img_size=64, pretrained=False) == "adapter"
    assert calls == [("vit_base_patch16_224.dino", 64, False)]
    monkeypatch.undo()
    with pytest.raises(ValueError, match="CNN"):
        T.build_model_adapter("convnext_b", pretrained=False)


# ----------------------------------------------------------------------------------------------
# Tiny ViT adapters
# ----------------------------------------------------------------------------------------------
@pytest.fixture(scope="module", params=sorted(TINY_VIT))
def vit(request):
    ad, img = tiny_vit(request.param)
    x = torch.randn(2, 3, img, img, generator=torch.Generator().manual_seed(1))
    return request.param, ad, x


def test_vit_layout(vit):
    name, ad, _ = vit
    n_prefix, kind, rope = EXPECT[name]
    assert ad.grid_hw == (4, 4) and ad.n_prefix_tokens == n_prefix
    assert ad.embed_dim == 64 and ad.depth == 2
    assert (AX.is_swiglu(ad.model.blocks[0].mlp)) is (kind == "swiglu")
    assert ad.mlp_hidden == AX.mlp_hidden_of(ad.model.blocks[0].mlp)
    assert getattr(ad, "uses_rope", False) is rope
    assert len(ad.mean) == 3 and len(ad.std) == 3
    if name == "clip_b16":
        assert type(ad.model.blocks[0].mlp.act).__name__ == "QuickGELU"
        assert isinstance(ad.model.norm_pre, nn.LayerNorm)
        assert ad.mean == pytest.approx((0.48145466, 0.4578275, 0.40821073))


@torch.no_grad()
def test_vit_tokens_reproduce_forward_features(vit):
    _, ad, x = vit
    m = ad.model
    ref = m.forward_features(x)
    out = m.norm(ad.forward_tokens(x))
    assert out.shape == ref.shape
    assert float((out - ref).abs().max()) < 1e-5
    assert ad.cls_feature(x).shape == (2, 64)


@torch.no_grad()
def test_vit_patch_tokens_are_row_major(vit):
    """A bright patch at grid cell (r, c) changes token n_prefix + r * W + c most (layer 0 tokens)."""
    _, ad, _ = vit
    P, (H, W) = ad.patch_size, ad.grid_hw
    base = torch.zeros(1, 3, ad.img_size, ad.img_size)
    t0 = ad.layer0_tokens(base)
    for r, c in ((0, 1), (2, 3), (3, 0)):
        x = base.clone()
        x[..., r * P:(r + 1) * P, c * P:(c + 1) * P] = 3.0
        d = (ad.layer0_tokens(x) - t0)[0].norm(dim=-1)
        assert int(d[ad.n_prefix_tokens:].argmax()) == r * W + c


def test_rope_models_pass_the_flip_check(vit):
    name, ad, _ = vit
    if not EXPECT[name][2]:
        return
    for g in ("LR", "UD"):
        r = AX.rope_flip_check_adapter(ad, g)
        assert r["ok"] and r["n_neither"] == 0 and r["n_odd"] > 0, r
        assert r["max_rel_logit_err"] < 1e-6


def test_rope_check_rejects_mixed_frequencies():
    """A pair rotating with w_x c + w_y r (both non-zero) is neither even nor odd under the mirror."""
    G, dh = 4, 4
    r, c = np.divmod(np.arange(G * G), G)
    theta = np.stack([0.7 * c + 0.4 * r, 0.9 * r], axis=1)            # pair 0 mixed, pair 1 axial
    ang = np.repeat(theta, 2, axis=1)                                  # interleaved layout
    emb = torch.as_tensor(np.concatenate([np.sin(ang), np.cos(ang)], axis=1))
    res = AX.rope_flip_check(emb, (G, G), half=False, element="LR")
    assert res["n_neither"] == 1 and not res["ok"]
    axial = np.repeat(np.stack([0.7 * c, 0.9 * r], axis=1), 2, axis=1)
    emb2 = torch.as_tensor(np.concatenate([np.sin(axial), np.cos(axial)], axis=1))
    assert AX.rope_flip_check(emb2, (G, G), half=False, element="LR")["ok"]


# ----------------------------------------------------------------------------------------------
# SwiGLU
# ----------------------------------------------------------------------------------------------
class _Dinov2SwiGLUReference(nn.Module):
    """The forward of dinov2/layers/swiglu_ffn.py SwiGLUFFN: w12 packed (gate first), w3 out."""

    def __init__(self, d, h):
        super().__init__()
        self.w12 = nn.Linear(d, 2 * h)
        self.w3 = nn.Linear(h, d)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(torch.nn.functional.silu(x1) * x2)


class _UnpackedReference(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w1, self.w2, self.w3 = nn.Linear(d, h), nn.Linear(d, h), nn.Linear(h, d)

    def forward(self, x):
        return self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))


@pytest.mark.parametrize("kind", ["dinov2_w12", "unpacked", "timm_swiglu", "timm_glumlp"])
def test_swiglu_conversion_is_the_same_function(kind):
    from timm.layers import GluMlp, SwiGLU

    torch.manual_seed(0)
    d, h = 12, 20
    if kind == "dinov2_w12":
        ref = _Dinov2SwiGLUReference(d, h)
    elif kind == "unpacked":
        ref = _UnpackedReference(d, h)
    elif kind == "timm_swiglu":
        ref = SwiGLU(d, h, norm_layer=nn.LayerNorm)
        _randomise_affine(ref, 0)
        nn.init.normal_(ref.fc1_g.weight)
    else:
        ref = GluMlp(d, 2 * h, act_layer=nn.SiLU, gate_last=False)
    ref = ref.double()
    conv = AX.swiglu_from_module(copy.deepcopy(ref))
    x = torch.randn(3, 5, d, dtype=torch.float64)
    assert conv.hidden_features == h
    assert AX.swiglu_conversion_error(ref, conv, x) < 1e-12


class _Block(nn.Module):
    """Pre-norm MLP branch with LayerScale as Eva's gamma_2: gamma_2 * mlp(norm2(x))."""

    def __init__(self, mlp, d):
        super().__init__()
        self.norm2 = nn.LayerNorm(d, eps=1e-6)
        self.mlp = mlp
        self.gamma_2 = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.gamma_2 * self.mlp(self.norm2(x))


def _swiglu_block(d=10, h=14, seed=0):
    g = torch.Generator().manual_seed(seed)
    lin = lambda i, o: nn.Linear(i, o)  # noqa: E731
    mlp = AX.SwiGLUMLP(lin(d, h), lin(d, h), lin(h, d), norm=nn.LayerNorm(h, eps=1e-6))
    blk = _Block(mlp, d).double()
    with torch.no_grad():
        for p in blk.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float64) * (0.5 if p.dim() > 1 else 0.3))
        for p in (blk.norm2.weight, blk.mlp.norm.weight, blk.gamma_2):           # affines and LayerScale near 1
            p.add_(1.0)
    return blk


@torch.no_grad()
def test_swiglu_fold_is_exact_and_projection_is_equivariant():
    d, h = 10, 14
    blk = _swiglu_block(d, h)
    X = torch.randn(7, d, dtype=torch.float64)
    ref = blk(X)
    f = AX.fold_vit_mlp(blk)
    assert f["kind"] == "swiglu" and len(f["W1"]) == 2
    folded = copy.deepcopy(blk)
    AX.write_absorbed_vit_mlp(folded, f["W1"], f["b1"], f["W2"], f["b2"])
    assert float((folded(X) - ref).abs().max()) < 1e-12                       # folding keeps the function
    A_in = GR.random_c2(d, 6, seed=1)                                           # involutions fixing the ones vector
    A_out = GR.random_c2(d, 4, seed=2)
    perm = GR.random_permutation_like(np.array([1, 0, 3, 2, 5, 4] + list(range(6, h))), np.random.default_rng(3))
    e = CG.project_unit(f, A_in, A_out, perm)
    proj = copy.deepcopy(blk)
    AX.write_absorbed_vit_mlp(proj, e["W1"], e["b1"], e["W2"], e["b2"])
    A_in_t, A_out_t = torch.as_tensor(A_in), torch.as_tensor(A_out)
    lhs, rhs = proj(X @ A_in_t), proj(X) @ A_out_t                             # MLP(x A_in) = MLP(x) A_out
    assert float((lhs - rhs).abs().max()) < 1e-10
    sh = CG.unit_shares(f, e)
    assert 0.0 < sh["s_fc1"] < 1.0 and 0.0 < sh["s_fc2"] < 1.0
    again = CG.project_unit(e, A_in, A_out, perm)                              # a projection: idempotent
    assert CG.unit_shares(e, again)["s_fc1"] < 1e-20 and CG.unit_shares(e, again)["s_fc2"] < 1e-20


def test_swiglu_free_parameters_equal_brute_force():
    d, h = 5, 6
    A_in, A_out = GR.random_c2(d, 3, seed=4), GR.random_c2(d, 2, seed=5)
    perm = np.array([1, 0, 3, 2, 4, 5])
    f = {"W1": [torch.zeros(h, d, dtype=torch.float64)] * 2, "b1": [torch.zeros(h, dtype=torch.float64)] * 2,
         "W2": torch.zeros(d, h, dtype=torch.float64), "b2": torch.zeros(d, dtype=torch.float64)}
    fr = CG.unit_free(f, A_in, A_out, perm)
    act = GR.block_action("LR", [A_in], [A_out], [perm])
    assert fr["free_fc1"] == 2 * GR.constraint_nullity(act, h, d, "fc1")
    assert fr["free_fc2"] == GR.constraint_nullity(act, h, d, "fc2")
    assert fr["dense_fc1"] == 2 * h * d
    assert fr["tax_mlp"] == pytest.approx(1 - (fr["free_fc1"] + fr["free_fc2"]) / (3 * h * d))


def test_gelu_fold_equals_surgery_fold(tiny64):
    from mirror.surgery import fold_mlp

    blk = tiny64.model.blocks[1]
    a, b = AX.fold_vit_mlp(blk), fold_mlp(blk)
    assert len(a["W1"]) == 1
    for k in ("W1", "b1"):
        assert torch.equal(a[k][0], b[k])
    for k in ("W2", "b2"):
        assert torch.equal(a[k], b[k])


# ----------------------------------------------------------------------------------------------
# CNNs: exact flip at the chosen sizes
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def _symmetrise_kernels(model):
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            mod.weight.copy_(0.5 * (mod.weight + mod.weight.flip(-1)))


@torch.no_grad()
@pytest.mark.parametrize("family,good,bad", [("resnet", 33, 32), ("convnext", 64, 56)])
def test_flip_is_exact_only_at_the_chosen_sizes(family, good, bad):
    """With flip-symmetric kernels every feature map is exactly flip-equivariant at the chosen input size (ResNet
    32 k + 1, ConvNeXt 32 k), and not at the other one (stride-2 parity)."""
    ad = tiny_resnet(65) if family == "resnet" else tiny_convnext(64)
    ad.model.double()
    _symmetrise_kernels(ad.model)
    spec = CG.resnet_spec(ad) if family == "resnet" else CG.convnext_spec(ad)

    def errs(size):
        x = torch.randn(2, 3, size, size, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
        return CG.tiny_check_flip(spec, x, "LR")

    e_good, e_bad = errs(good), errs(bad)
    assert max(e_good.values()) < 1e-12, e_good
    assert max(e_bad.values()) > 1e-3, e_bad
    assert AX.flip_exact_size(family, good) and not AX.flip_exact_size(family, bad)


def test_cnn_adapter_refuses_inexact_size_and_resizes():
    with pytest.raises(ValueError, match="flip is not exact"):
        tiny_resnet(64)
    ad = tiny_resnet(65)
    x = ad.prepare(torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8), "cpu")
    assert x.shape == (2, 3, 65, 65)
    u8 = torch.randint(0, 256, (1, 3, 64, 64), dtype=torch.uint8)
    flip_then = ad.prepare(u8.flip(-1), "cpu")
    then_flip = ad.prepare(u8, "cpu").flip(-1)
    assert float((flip_then - then_flip).abs().max()) < 1e-5                   # bilinear resize commutes with flip


def test_cnn_folds_are_exact():
    cn, rn = tiny_convnext(), tiny_resnet()
    blk = cn.model.stages[0].blocks[0]
    x = torch.randn(3, 2, 2, 8)                                                # NHWC tokens entering the norm
    f = CG.fold_convnext_block(blk)
    with torch.no_grad():
        ref = blk.mlp(blk.norm(x)) * blk.gamma
        u = torch.nn.functional.layer_norm(x.double(), (8,), None, None, blk.norm.eps)
        got = torch.nn.functional.gelu(u @ f["W1"][0].T + f["b1"][0]) @ f["W2"].T + f["b2"]
    assert float((got.float() - ref).abs().max()) < 1e-5
    bb = rn.model.layer2[0]
    f = CG.fold_bottleneck(bb)
    xin = torch.randn(2, bb.conv1.in_channels, 3, 3)
    with torch.no_grad():
        ref1 = bb.bn1(bb.conv1(xin))
        got1 = torch.einsum("oc,bchw->bohw", f["W1"][0].float(), xin) + f["b1"][0].float()[None, :, None, None]
    assert float((got1 - ref1).abs().max()) < 1e-5


# ----------------------------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("family", ["convnext", "resnet"])
def test_lr_measurement_runs_end_to_end_on_tiny_cnns(tmp_path, family):
    ad = tiny_convnext() if family == "convnext" else tiny_resnet()
    spec = CG.convnext_spec(ad, "tiny_" + family) if family == "convnext" else CG.resnet_spec(ad, "tiny_" + family)
    proto = TinyProtocol(img_size=64)
    out = tmp_path / family
    m = CG.run_measurement(spec, lambda: proto.fit_loader(0), "cpu", out, groups=("LR",), null_draws=2, top_k=3,
                           workers=1, log=lambda *_: None)
    import pandas as pd

    blk = pd.read_csv(out / "blocks.csv")
    assert len(blk) == len(spec.units) and set(blk["group"]) == {"LR"}
    for col in ("s_fc1", "s_fc2", "s_mlp_null_mean", "tax_mlp", "frac_2_t50"):
        assert blk[col].notna().all(), col
    assert ((blk["s_fc1"] >= 0) & (blk["s_fc1"] <= 1)).all() and ((blk["tax_mlp"] > 0) & (blk["tax_mlp"] < 1)).all()
    r = m["groups"]["LR"]["ranges"]["first_half"]
    assert np.isfinite(r["ratio_to_null"]) and 0 <= r["twin_fraction_t50"] <= 1 and 0 < r["tax_mlp"] < 1
    assert "stages_1-2" in m["groups"]["LR"]["ranges"]
    lay = pd.read_csv(out / "layers.csv")
    assert lay["d_plus"].between(0, lay["d"]).all()
    if family == "resnet":                                     # two hidden layers per bottleneck, both reported
        assert "frac_2_t50_h1" in blk and "frac_2_t50_h2" in blk
        assert not lay["centred"].any()
    m2 = CG.run_measurement(spec, lambda: proto.fit_loader(0), "cpu", out, groups=("LR",), null_draws=2, top_k=3,
                            workers=1, log=lambda *_: None)                  # resumable: everything cached
    assert m2["timings_s"] == {} and m2["groups"]["LR"]["ranges"]["first_half"] == r


def test_engine_finds_planted_identity_structure(tmp_path):
    """An exactly flip-invariant network (symmetric kernels): every fitted operator is the identity and every
    hidden unit is its own twin (no pairs)."""
    ad = tiny_convnext()
    _symmetrise_kernels(ad.model)
    spec = CG.convnext_spec(ad, "sym")
    r = CG.collect_pass(spec, TinyProtocol(img_size=64).fit_loader(0), "cpu", "LR", inputs=False)
    for k, sp in r["space"].items():
        A, dp, dm = GR.fit_order2(sp["C"], center=spec.centre_of(k))
        assert dm == 0 and np.allclose(A, np.eye(A.shape[0]), atol=1e-8), k
    match = CG.match_hidden(r["R"], top_k=3, workers=1)
    for k, mm in match.items():
        assert (mm["perm"] == np.arange(len(mm["perm"]))).all(), k


def test_engine_matches_measure_groups_on_the_tiny_vit(tmp_path):
    """Same model, data, fits, matchings and null seeds: the engine reproduces scripts/measure_groups.py."""
    import pandas as pd

    mg = _load("measure_groups")
    mg.main(["--tiny", "--groups", "LR", "--null-draws", "2", "--twin-null-draws", "0", "--no-attn", "--no-heldout",
             "--workers", "1", "--top-k", "3", "--out", str(tmp_path / "mg")])
    ad = TimmTinyAdapter(img_size=56, seed=0)
    proto = TinyProtocol()
    CG.run_measurement(CG.vit_spec(ad, name="tiny"), lambda: proto.fit_loader(0), "cpu", tmp_path / "eng",
                       groups=("LR",), null_draws=2, top_k=3, workers=1, log=lambda *_: None)
    a, b = pd.read_csv(tmp_path / "mg" / "blocks.csv"), pd.read_csv(tmp_path / "eng" / "blocks.csv")
    m = a.merge(b, on=["group", "block"], suffixes=("_mg", "_en"))
    assert len(m) == 4
    for c in ("s_fc1", "s_fc2", "s_b1", "s_b2", "s_fc1_act", "s_mlp_null_mean", "frac_2_t50", "free_fc1", "free_fc2"):
        assert float((m[c + "_mg"] - m[c + "_en"]).abs().max()) < 1e-12, c
    assert float((m["s_fc2_act_mg"] - m["s_fc2_act_en"]).abs().max()) < 1e-6   # measure_groups stores M2 in float32
    ja, jb = (json.loads((tmp_path / d / "metrics.json").read_text()) for d in ("mg", "eng"))
    for key in ("ratio_to_null", "twin_fraction_t50", "tax_mlp"):
        assert ja["groups"]["LR"]["ranges"]["first_half"][key] == pytest.approx(
            jb["groups"]["LR"]["ranges"]["first_half"][key], abs=1e-12)


def test_engine_runs_on_tiny_swiglu_rope_vit(tmp_path):
    ad, img = tiny_vit("eva02_b14")
    spec = CG.vit_spec(ad, name="tiny_eva")
    assert {u.kind for u in spec.units} == {"swiglu"}
    proto = TinyProtocol()
    m = CG.run_measurement(spec, lambda: proto.fit_loader(0), "cpu", tmp_path, groups=("LR", "UD"), null_draws=2,
                           top_k=3, workers=1, log=lambda *_: None)
    for g in ("LR", "UD"):
        r = m["groups"][g]["ranges"]["first_half"]
        assert np.isfinite(r["ratio_to_null"]) and 0 < r["tax_mlp"] < 1


# ----------------------------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------------------------
def test_model_records_come_from_yaml():
    c4 = _load("core04_generality")
    clip, v3, sig = (c4.model_record(n, False) for n in ("clip_b16", "dinov3_b16", "siglip2_b16"))
    assert clip["recipe_flip"] == "no" and clip["recipe_verified"]
    assert v3["recipe_flip"] == "no" and not v3["recipe_verified"]
    assert sig["recipe_flip"] == "unknown"
    b_reg = c4.model_record("dinov2_vitb14_reg", False)
    assert clip["size_class"] == v3["size_class"] == b_reg["size_class"] == "768x12"
    assert c4.model_record("convnext_b", False)["size_class"] == "CNN"
    assert set(c4.MODELS) >= set(AX.EXTRA_REGISTRY)


def test_scoring_rules():
    c4 = _load("core04_generality")
    order = ("s", "b", "l", "g")
    rows = [{"name": n, "LR_tax_mlp_first_half": t} for n, t in zip(order, (0.30, 0.35, 0.40, 0.42))]
    assert c4.score_p41(rows, order)["verdict"] == "PASS"
    rows[2]["LR_tax_mlp_first_half"] = 0.33
    r = c4.score_p41(rows, order)
    assert r["verdict"] == "FAIL" and r["direction"] == "mixed"
    assert c4.score_p41(rows[:3], order)["verdict"] == "not scored"
    base = dict(size_class="768x12", recipe_verified=True)
    rows = [{"name": "clip", "recipe_flip": "no", "LR_tax_mlp_first_half": 0.20, **base},
            {"name": "v3", "recipe_flip": "no", "LR_tax_mlp_first_half": 0.35, **base, "recipe_verified": False},
            {"name": "dino", "recipe_flip": "yes", "LR_tax_mlp_first_half": 0.30, **base},
            {"name": "deit", "recipe_flip": "yes", "LR_tax_mlp_first_half": 0.25, **base},
            {"name": "sig", "recipe_flip": "unknown", "LR_tax_mlp_first_half": 0.10, **base},
            {"name": "cnn", "recipe_flip": "yes", "LR_tax_mlp_first_half": 0.05, "size_class": "CNN",
             "recipe_verified": True}]
    p42 = c4.score_p42(rows, False)
    assert p42["verdict"] == "PASS"                                           # 0.20 < min(0.30, 0.25); CNN excluded
    assert p42["secondary (with probable no-flip)"]["verdict"] == "FAIL"      # DINOv3 0.35 is not below 0.25
    rows.append({"name": "mae", "recipe_flip": "yes", "LR_tax_mlp_first_half": None, **base})
    assert c4.score_p42(rows, False)["verdict"] == "incomplete"               # a flip-trained peer not measured yet
    rows[-1]["LR_tax_mlp_first_half"] = 0.15
    assert c4.score_p42(rows, False)["verdict"] == "FAIL"
    conv = [{"name": "cx", "LR_ratio_to_null_first_half": 0.7, "LR_twin_fraction_t50_first_half": 0.25}]
    assert c4.score_p43(conv, "cx")["verdict"] == "PASS"
    conv[0]["LR_twin_fraction_t50_first_half"] = 0.1
    assert c4.score_p43(conv, "cx")["verdict"] == "FAIL"
    assert c4.score_p43(conv, "missing")["verdict"] == "not scored"


def test_runner_tiny_end_to_end(tmp_path):
    c4 = _load("core04_generality")
    argv = ["--tiny", "--out", str(tmp_path), "--null-draws", "2", "--top-k", "3",
            "--models", "tiny_eva,tiny_convnext,tiny_resnet", "--steps", "structure,surgery,score"]
    res = c4.main(argv)
    assert {v["status"] for v in res["structure"].values()} == {"measured"}
    for f in ("summary.csv", "tax_by_depth.csv", "scores.json", "SUMMARY.md", "meta.json"):
        assert (tmp_path / f).exists(), f
    for n in ("tiny_eva", "tiny_convnext", "tiny_resnet"):
        assert (tmp_path / "structure" / n / "metrics.json").exists()
        assert (tmp_path / "structure" / n / "meta.json").exists()
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["experiment"] == "CORE-04" and meta["prereg"]["file"] == "PREREG_NEXT_PHASE.md"
    sc = json.loads((tmp_path / "scores.json").read_text())
    assert sc["P4.3"]["verdict"] in ("PASS", "FAIL") and sc["P4.1"]["verdict"] == "not scored"
    eva = json.loads((tmp_path / "structure" / "tiny_eva" / "metrics.json").read_text())
    assert eva["checks"]["rope_flip"]["LR"]["ok"] is True
    import pandas as pd

    st = pd.read_csv(tmp_path / "surgery" / "status.csv")
    assert set(st["model"]) == {"tiny_eva"}                                   # CNNs are structure only
    res2 = c4.main(argv[:-1] + ["structure,score"])                           # resumable
    assert {v["status"] for v in res2["structure"].values()} == {"done"}


def test_runner_tiny_surgery_and_linear_probe(tmp_path):
    """Surgery step through scripts/must07_compressors.py (Mirror-exact and plain SVD, first half, both sizes) and the
    MAE-style linear probe on the rebuilt models, forced on the tiny GELU ViT (its ratio is above 0.8)."""
    import pandas as pd

    c4 = _load("core04_generality")
    c4.main(["--tiny", "--out", str(tmp_path), "--null-draws", "2", "--top-k", "3", "--models", "tiny",
             "--steps", "structure,surgery,score", "--tiny-force-surgery"])
    st = pd.read_csv(tmp_path / "surgery" / "status.csv")
    assert st.iloc[0]["eligible"] and str(st.iloc[0]["status"]).startswith("run: must07 sizes 2mlp,2both")
    rows = pd.read_csv(tmp_path / "surgery" / "rows_tiny.csv")
    got = set(zip(rows["size"], rows["method"]))
    assert {("2mlp", "mirror_exact"), ("2mlp", "svd"), ("2both", "mirror_exact"), ("2both", "svd")} <= got
    ok = rows[rows["method"] != "original"]
    assert (ok["status"] == "ok").all() and (ok["gap_rel"] >= -1e-12).all()
    lin = pd.read_csv(tmp_path / "surgery" / "linear_tiny.csv")
    assert len(lin) == 5 and lin["top1"].between(0, 100).all()
    assert set(lin["method"]) == {"original", "mirror_exact", "svd"}
