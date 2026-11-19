"""A12b fused even/odd MLP kernel (mirror/final/systems/kernel.py, scripts/fp_a12b_kernel.py). CPU only.

The pure PyTorch reference of the fused computation against mirror/efficient.py's even/odd MLP and the dense projected
MLP (float64 to 1e-12, float32 to float32 rounding), with alignment padding, odd numbers of self-twins and a block
without pairs; exact sign-flip equivariance of the reference (float64, float32, bfloat16); the builders, the backend
fallback, the custom op's CPU path and fake implementation; the FastViT swap on a tiny converted model; the bar rule;
the script's dry run. The Triton path needs CUDA and Triton and is skipped here: its GPU test is the script's
--correctness-only run in W0."""
import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MIRROR_FINAL_ALLOW_NO_PREREG", "1")

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from mirror.efficient import Basis, build_dense_mlp, build_mirror_mlp, fast_vit_cost  # noqa: E402
from mirror.final.systems import kernel as K  # noqa: E402
from mirror.m1 import random_involution_fixing_ones  # noqa: E402
from mirror.matching import random_pairing  # noqa: E402
from mirror.surgery import project_mlp_two  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# (d, h, d+, twin pairs): d+ never a multiple of 8; self-twins 35 (odd), 75 (odd, no pairs), 1 (odd), 4 with ke
# clipped at d (d+ = 37 -> 40 = d), 70.
CASES = [(40, 75, 23, 20), (40, 75, 23, 0), (48, 97, 33, 48), (40, 64, 37, 30), (32, 96, 17, 13)]


def equivariant_mlp(d, h, dp, n_pairs, seed=0):
    """Random weights projected in float64 onto the equivariant subspace of (A, perm)."""
    A = random_involution_fixing_ones(d, dp, seed=seed + 1)
    perm = random_pairing(h, n_pairs, seed=seed + 2)
    g = torch.Generator().manual_seed(seed)
    kw = dict(generator=g, dtype=torch.float64)
    W = project_mlp_two(torch.randn(h, d, **kw) / d ** 0.5, 0.1 * torch.randn(h, **kw),
                        torch.randn(d, h, **kw) / h ** 0.5, 0.1 * torch.randn(d, **kw), A, A, perm)
    return W, A, perm


def dense(x, W):
    return F.linear(F.gelu(F.linear(x, W["W1"], W["b1"])), W["W2"], W["b2"])


def even_odd(W, A, perm, align):
    B = Basis.of(A)
    mm = build_mirror_mlp(W["W1"], W["b1"], W["W2"], W["b2"], nn.GELU(), B, B, A, A, perm, align=align, tol=1e-9)
    return mm, torch.as_tensor(B.Q, dtype=torch.float64), B.dp


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).abs().max() / b.abs().max())


def tokens(*shape, seed=0, dtype=torch.float64):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64).to(dtype)


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("fp_a12b_kernel_test", ROOT / "scripts" / "fp_a12b_kernel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- reference against the even/odd and dense MLPs
@pytest.mark.parametrize("align", [1, 8])
@pytest.mark.parametrize("d,h,dp,n_pairs", CASES)
def test_reference_equals_even_odd_and_dense_float64(d, h, dp, n_pairs, align):
    W, A, perm = equivariant_mlp(d, h, dp, n_pairs)
    mm, Q, _ = even_odd(W, A, perm, align)
    fm = K.build_fused_mlp(mm)
    if not K.available():
        assert fm.backend == "reference"
    fm = K.build_fused_mlp(mm, backend="reference")
    x = tokens(3, 11, d)
    y = fm(x @ Q)
    assert y.dtype == torch.float64 and y.shape == (3, 11, d)
    assert rel(y, dense(x, W) @ Q) < 1e-12                     # the dense projected MLP, output in the A* basis
    assert rel(y, mm(x @ Q)) < 1e-12                           # mirror/efficient.py's even/odd MLP
    lay = fm.layout
    assert (lay["n_pairs"], lay["n_self"]) == (n_pairs, h - 2 * n_pairs)
    assert fm.npad % 8 == 0 and fm.nspad % 8 == 0 and fm.npad >= n_pairs and fm.nspad >= h - 2 * n_pairs
    assert fm.d == d and fm.ke <= d and fm.ko + lay["KO"] == d
    if align == 8:                                            # 700 -> 704 and 68 -> 72 in the plan's sizes
        assert (fm.ke % 8 == 0 or fm.ke == d) and fm.ko % 8 == 0
    assert fm.cost["matrix_params"] == mm.cost["matrix_params"]
    assert fm.cost["macs_token_exec"] == sum(int(p.numel()) for p in (fm.E1, fm.O1, fm.E2, fm.O2))


def test_hidden_padding_is_optional():
    W, A, perm = equivariant_mlp(40, 75, 23, 21)
    mm, Q, _ = even_odd(W, A, perm, 1)
    fm1 = K.build_fused_mlp(mm, backend="reference", align_hidden=1)
    fm8 = K.build_fused_mlp(mm, backend="reference", align_hidden=8)
    assert (fm1.npad, fm1.nspad) == (21, 33) and (fm8.npad, fm8.nspad) == (24, 40)
    x = tokens(5, 40) @ Q
    assert rel(fm8(x), fm1(x)) < 1e-14 and rel(fm1(x), mm(x)) < 1e-12


@pytest.mark.parametrize("d,h,dp,n_pairs", CASES[:3])
def test_reference_float32_within_rounding(d, h, dp, n_pairs):
    W, A, perm = equivariant_mlp(d, h, dp, n_pairs)
    mm, Q, _ = even_odd(W, A, perm, 8)
    x32 = (tokens(4, 9, d) @ Q).float()
    truth = K.build_fused_mlp(copy.deepcopy(mm).float().double(), backend="reference")(x32.double())
    err_fused = rel(K.build_fused_mlp(mm, backend="reference").float()(x32), truth)
    err_plain = rel(copy.deepcopy(mm).float()(x32), truth)
    assert err_fused <= max(8 * err_plain, 64 * 2.0 ** -24) and err_fused < 1e-5


def test_build_from_dense_weights():
    W, A, perm = equivariant_mlp(48, 97, 33, 48, seed=5)
    fm = K.build_fused_mlp_from_dense(W["W1"], W["b1"], W["W2"], W["b2"], A, perm, backend="reference")
    Q = torch.as_tensor(Basis.of(A).Q, dtype=torch.float64)
    x = tokens(2, 7, 48, seed=1)
    assert rel(fm(x @ Q), dense(x, W) @ Q) < 1e-12
    noise = 0.1 * torch.randn(W["W1"].shape, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    with pytest.raises(ValueError, match="not equivariant"):         # unprojected weights are refused
        K.build_fused_mlp_from_dense(W["W1"] + noise, W["b1"], W["W2"], W["b2"], A, perm, backend="reference")


# ---------------------------------------------------------------- exact equivariance of the reference
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("align", [1, 8])
@pytest.mark.parametrize("d,h,dp,n_pairs", CASES)
def test_reference_sign_flip_is_exact(d, h, dp, n_pairs, align, dtype):
    """Negating the odd input coordinates leaves the even outputs unchanged and negates the odd outputs, bit for bit
    (torch.equal: +0 equals -0), in every precision: the twins swap exactly."""
    W, A, perm = equivariant_mlp(d, h, dp, n_pairs, seed=3)
    mm, Q, dpp = even_odd(W, A, perm, align)
    assert dpp == dp
    fm = K.build_fused_mlp(mm, backend="reference").to(dtype)
    x = (tokens(4, 13, d, seed=2) @ Q).to(dtype)
    s = K.sign_vector(d, dp, dtype=dtype)
    y, ym = fm(x), fm(x * s)
    assert torch.equal(ym, y * s)
    if n_pairs:
        assert not torch.equal(ym, y)                           # the check is not vacuous
    else:
        assert not bool(y[..., dp:].any())                      # no twins: no odd channel, odd outputs exactly 0


# ---------------------------------------------------------------- builders, backends, custom op
def test_rejects_other_activations_and_modules():
    W, A, perm = equivariant_mlp(40, 75, 23, 20)
    B = Basis.of(A)
    tanh = build_mirror_mlp(W["W1"], W["b1"], W["W2"], W["b2"], nn.GELU(approximate="tanh"), B, B, A, A, perm,
                            align=8)
    with pytest.raises(ValueError, match="exact"):
        K.build_fused_mlp(tanh)
    with pytest.raises(TypeError):
        K.build_fused_mlp(build_dense_mlp(W["W1"], W["b1"], W["W2"], W["b2"], nn.GELU()))
    with pytest.raises(ValueError, match="backend"):
        K.resolve_backend("cuda")


@pytest.mark.skipif(K.available(), reason="CUDA and Triton are present")
def test_cpu_backend_falls_back_cleanly():
    st = K.triton_status()
    assert st["ok"] is False and st["reason"] and not K.available()
    assert K.resolve_backend("auto") == "reference" and K.resolve_backend("reference") == "reference"
    with pytest.raises(RuntimeError, match="not available"):
        K.resolve_backend("triton")
    W, A, perm = equivariant_mlp(40, 75, 23, 20)
    mm, Q, _ = even_odd(W, A, perm, 8)
    fm = K.build_fused_mlp(mm)
    assert fm.backend == "reference" and K.autotune_report() == {}
    fm.backend = "triton"                                       # a Triton module never falls back silently
    with pytest.raises(RuntimeError, match="CUDA"):
        fm(tokens(2, 40) @ Q)


def test_custom_op_cpu_path_and_fake():
    W, A, perm = equivariant_mlp(40, 75, 23, 20)
    mm, Q, _ = even_odd(W, A, perm, 8)
    fm = K.build_fused_mlp(mm, backend="reference").float()
    op = K.fc1_op()
    assert op is K.fc1_op() and "fc1_butterfly" in str(op)
    x2 = (tokens(19, 40) @ Q).float()
    args = (x2, fm.E1.detach(), fm.b1.detach(), fm.O1.detach(), fm.ko, fm.npad)
    h, dd = op(*args)
    href, dref = K.fc1_reference(*args)
    assert h.shape == (19, fm.npad + fm.nspad) and dd.shape == (19, fm.npad)
    assert torch.equal(h, href) and torch.equal(dd, dref)
    if hasattr(torch.library, "opcheck"):                       # schema, fake (meta) implementation, tracing
        torch.library.opcheck(op, args)


@pytest.mark.skipif(not K.available(), reason="needs CUDA and Triton (W0 runs scripts/fp_a12b_kernel.py "
                                              "--correctness-only)")
def test_triton_matches_reference_on_gpu():
    W, A, perm = equivariant_mlp(64, 160, 37, 50)
    mm, Q, dp = even_odd(W, A, perm, 8)
    fm = K.build_fused_mlp(mm, backend="triton").float().cuda()
    ref = K.build_fused_mlp(mm, backend="reference").float().cuda()
    x = (tokens(300, 64) @ Q).float().cuda()
    assert rel(fm(x), ref(x)) < 1e-5
    fb = copy.deepcopy(fm).to(torch.bfloat16)
    assert rel(fb(x.to(torch.bfloat16)), ref(x)) < 3e-2


# ---------------------------------------------------------------- whole model
@pytest.fixture(scope="module")
def tiny_converted():
    from mirror.adapters import TimmTinyAdapter
    from mirror.attention import attn_dims
    from mirror.final.audit.exactness import AuditStructure
    from mirror.m1 import build_vexact, random_head_structure

    ad = TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float64)
    d, h, L = ad.embed_dim, ad.mlp_hidden, ad.depth
    H, dh = attn_dims(ad.model.blocks[0])
    A = random_involution_fixing_ones(d, 37, seed=3)                      # d+ = 37: not a multiple of 8
    perms = [random_pairing(h, 61 + 20 * b, seed=10 + b) for b in range(L)]
    maps = [random_head_structure(H, dh, (1, 2)[b % 2], seed=20 + b) for b in range(L)]
    model, info = build_vexact(ad, A, list(range(L)), perms, maps, even_head=True)
    return ad, model, AuditStructure(A=A, perms=perms, maps=info["maps_after"])


def test_fused_fast_vit_equals_even_odd_and_converted_model(tiny_converted):
    from mirror.final.audit.exactness import build_even_odd

    ad, model, st = tiny_converted
    fv = build_even_odd(model, st, align=8)
    fz = K.fuse_fast_vit(fv)
    assert all(isinstance(b.mlp, K.FusedMirrorMLP) for b in fz.blocks)
    assert not any(isinstance(b.mlp, K.FusedMirrorMLP) for b in fv.blocks)        # the input is left untouched
    assert all(b.rot_t is None for b in fz.blocks) and fz.info["fused_mlps"] == len(fz.blocks)  # one basis
    x = torch.randn(2, 3, 56, 56, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    y = fv(x)
    assert rel(fz(x), y) < 1e-12
    assert rel(ad.cls_feature(x, model), y) < 1e-10                             # the dense-stored converted model
    hy = K.fused_even_odd(model, st, attention="dense")
    assert rel(hy(x), y) < 1e-10 and hy.info["attention"] == "dense"
    eo = K.fused_even_odd(model, st, attention="even_odd")
    assert rel(eo(x), y) < 1e-12
    T, P = ad.n_tokens, ad.grid_hw[0] * ad.grid_hw[1]
    assert fast_vit_cost(fz, T, P)["macs"] == fast_vit_cost(fv, T, P)["macs"]  # same analytic cost


# ---------------------------------------------------------------- script
def test_bar_rule(script):
    a = SimpleNamespace(dry_run=False, correctness_only=False)

    def rows(o64, f64, o256, f256, mode="compile"):
        out = []
        for b, o, f in ((64, o64, f64), (256, o256, f256)):
            out.append({"level": "model", "form": "original", "batch": b, "throughput_img_s": o,
                        "mode_used": "compile"})
            out.append({"level": "model", "form": "even_odd_fused", "batch": b, "throughput_img_s": f,
                        "mode_used": mode})
        return out

    met = script.score_bar(rows(100, 111, 100, 115), True, "triton", a)
    assert met["met"] is True and met["status"] == "met" and met["ratios"][64] == pytest.approx(1.11)
    low = script.score_bar(rows(100, 109, 100, 130), True, "triton", a)            # batch 64 below 1.1
    assert low["met"] is False and low["status"] == "not met"
    for r, ok, be in ((rows(100, 120, 100, 120, mode="eager"), True, "triton"), (rows(100, 120, 100, 120), False,
                      "triton"), (rows(100, 120, 100, 120), True, "reference"), (rows(100, 120, 100, 120)[:2], True,
                                                                                  "triton")):
        res = script.score_bar(r, ok, be, a)
        assert res["met"] is None and res["status"].startswith("not scored")
    assert script.score_bar(rows(100, 120, 100, 120), True, "triton",
                            SimpleNamespace(dry_run=False, correctness_only=True))["met"] is None
    eager = script.score_bar(rows(100, 120, 100, 120, mode="eager"), True, "triton", a)
    assert eager["ratios"] == {}                                                   # eager ratios never reach the bar

    # the keys the final collector reads
    k = script.headline_keys(rows(100, 111, 100, 115), met, a)
    assert k["scope"] == "whole_model" and k["bar_met"] is True
    assert k["speed_vs_original_b64"] == pytest.approx(1.11) and k["speed_vs_original_b256"] == pytest.approx(1.15)
    assert script.headline_keys(rows(100, 109, 100, 130), low, a)["bar_met"] is False
    stack = [{"level": "mlp_stack", "form": f, "batch": b, "throughput_img_s": 100.0, "mode_used": "compile"}
             for f in ("dense", "even_odd", "fused") for b in (64, 256)]
    only = script.headline_keys(stack, script.score_bar(stack, True, "triton", a), a)
    assert only["scope"] == "mlp_only" and only["bar_met"] is None
    assert only["speed_vs_original_b64"] is None and only["speed_vs_original_b256"] is None
    none = script.headline_keys([], script.score_bar([], True, "triton", a), a)
    assert none["scope"] == "none" and none["bar_met"] is None and none["speed_vs_original_b64"] is None


def test_script_dry_run(script, tmp_path):
    out = tmp_path / "a12b"
    res = script.main(["--dry-run", "--out", str(out)])
    assert res["ok"] and res["correctness"]["pass"] and res["backend"] == "reference"
    assert res["bar"]["met"] is None and res["bar"]["status"].startswith("not scored")
    import json

    saved = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert list(saved)[:6] == ["experiment", "speed_vs_original_b64", "speed_vs_original_b256", "bar_met",
                               "bar_status", "scope"]
    assert saved["scope"] == "dry_run" and saved["bar_met"] is None
    assert saved["speed_vs_original_b64"] is None and saved["speed_vs_original_b256"] is None
    for f in ("metrics.json", "timing.csv", "correctness.csv", "summary.md", "meta.json"):
        assert (out / f).exists(), f
    forms = {(r["level"], r["form"]) for r in res["timing_rows"]}
    assert forms == {("mlp_stack", "dense"), ("mlp_stack", "even_odd"), ("mlp_stack", "fused"), ("model", "original"),
                     ("model", "even_odd"), ("model", "even_odd_fused"), ("model", "hybrid_fused")}
    blocks = res["correctness"]["blocks"]
    assert len(blocks) == 4 and all(b["pass"] and b["equiv_exact_fused_fp32"] and b["equiv_exact_fused_bf16"]
                                    for b in blocks)
    assert res["correctness"]["model"]["pass"]
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert "NOT SCORED" in text and "not measurements" in text and chr(0x2014) not in text   # no em dash
