"""A12 systems benchmark (mirror/final/systems/bench.py, scripts/fp_a12_bench.py): only the plumbing is testable on
the CPU, so timings come from a fake timer; models are tiny random ViTs."""
import copy
import importlib.util
import math
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MIRROR_FINAL_ALLOW_NO_PREREG", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

from mirror.adapters import TimmTinyAdapter  # noqa: E402
from mirror.attention import attn_dims  # noqa: E402
from mirror.data import mirror  # noqa: E402
from mirror.final.audit.exactness import AuditStructure, build_even_odd  # noqa: E402
from mirror.final.systems import bench as BE  # noqa: E402
from mirror.heal_vexact import vexact_block_params  # noqa: E402
from mirror.m1 import build_vexact, random_head_structure, random_involution_fixing_ones  # noqa: E402
from mirror.matching import random_pairing  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class FakeTimer:
    """Returns the given milliseconds in turn and counts the timed calls (it still runs the function)."""

    def __init__(self, values):
        self.values, self.i, self.calls = list(values), 0, 0

    def __call__(self, fn, x):
        fn(x)
        self.calls += 1
        v = self.values[self.i % len(self.values)]
        self.i += 1
        return float(v)


class Counter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 4)
        self.n = 0

    def forward(self, x):
        self.n += 1
        return self.lin(x.mean(dim=(2, 3)))


@pytest.fixture(scope="module")
def tiny():
    ad = TimmTinyAdapter(img_size=56, seed=0, randomise=True, dtype=torch.float32)
    d, h, L = ad.embed_dim, ad.mlp_hidden, ad.depth
    H, dh = attn_dims(ad.model.blocks[0])
    A = random_involution_fixing_ones(d, 37, seed=3)                       # d+ = 37: not a multiple of 8
    perms = [random_pairing(h, 61 + b, seed=10 + b) for b in range(L)]
    maps = [random_head_structure(H, dh, (1, 2)[b % 2], seed=20 + b) for b in range(L)]
    model, info = build_vexact(ad, A, range(L), perms, maps, even_head=True)
    st = AuditStructure(A=A, perms=perms, maps=info["maps_after"])
    return ad, model, st


def test_measure_and_time_module_with_fake_timer():
    mod = Counter()
    t = FakeTimer([5, 1, 3, 2, 4])
    r = BE.measure(mod, torch.zeros(2, 3, 8, 8), warmup=3, iters=5, timer=t)
    assert r["latency_ms_median"] == 3.0 and r["latency_ms_min"] == 1.0 and r["n_timed"] == 5
    assert mod.n == 3 + 5 and t.calls == 5
    out = BE.time_module(Counter(), 4, img_px=8, device="cpu", dtype=torch.float32, compile=False, warmup=1, iters=3,
                         timer=FakeTimer([2.0]))
    assert out["latency_ms_median"] == 2.0 and out["throughput_img_s"] == pytest.approx(4 * 1e3 / 2.0)
    assert out["mode_used"] == "eager" and out["peak_memory_gb"] is None and out["batch"] == 4


def test_param_memory_flip_average_and_feature_modules(tiny):
    ad, model, st = tiny
    feat = BE.FeatureModule(ad, model)
    x = torch.randn(2, 3, 56, 56, generator=torch.Generator().manual_seed(0))
    fa, fb = BE.FlipAverage(feat), BE.FlipAverage(feat, batched=True)
    want = 0.5 * (ad.cls_feature(x, model) + ad.cls_feature(mirror(x), model))
    assert torch.allclose(fa(x), want, atol=1e-6) and torch.allclose(fb(x), want, atol=1e-5)
    pm = BE.param_memory(model)
    n_par = sum(p.numel() for p in model.parameters())
    assert pm["params_stored"] == n_par and pm["float_buffers"] == model.norm.even.P.numel()
    assert pm["param_mb_bf16"] == pytest.approx(2 * pm["entries_stored"] / 1e6)
    fv = build_even_odd(model, st, align=8)
    pe = BE.param_memory(fv)
    assert pe["entries_stored"] < pm["entries_stored"]                    # the even/odd form stores less
    assert BE.param_memory(fa)["entries_stored"] == BE.param_memory(feat)["entries_stored"]


def test_analytic_costs_agree(tiny):
    ad, model, st = tiny
    arch = BE.model_arch(ad)
    dense = BE.structure_cost(ad)
    c0 = BE.analytic_cost(arch, [])
    assert dense["gflops"] == pytest.approx(c0["gflops_model_dense"]) and dense["params_model"] == arch["model_params_dense"]
    blocks = vexact_block_params(ad.embed_dim, ad.mlp_hidden, st.A, st.perms, st.maps)
    c = BE.analytic_cost(arch, [{"mlp": b["mlp"], "attn": b["attn"], "rot": 0} for b in blocks])
    fv = build_even_odd(model, st, align=8)
    from mirror.efficient import fast_vit_cost
    H, W = ad.grid_hw
    fc = fast_vit_cost(fv, ad.n_tokens, H * W)
    assert fc["macs"] - fc["macs_head"] == c["macs_model"]               # the same accounting (CLS rotation aside)
    assert fc["macs_exec"] >= fc["macs"]                                  # padding is executed, not counted
    eo = BE.even_odd_cost(fv, ad)
    assert eo["gflops"] == pytest.approx(2 * fc["macs"] / 1e9)


def test_gemm_shapes_and_alignment(tiny):
    ad, model, st = tiny
    T = ad.n_tokens
    s1 = BE.gemm_shapes(build_even_odd(model, st, align=1), T, 3)
    s8 = BE.gemm_shapes(build_even_odd(model, st, align=8), T, 3)
    names = {r["where"].rsplit(".", 1)[-1] for r in s1}
    assert {"We", "Wodd", "Oe", "Oo", "E1", "O1", "E2", "O2"} <= names
    assert all(r["M"] == 3 * T and r["flops"] == 2 * r["M"] * r["N"] * r["K"] for r in s1)
    mirror8 = [r for r in s8 if r["where"].rsplit(".", 1)[-1] in ("We", "Wodd", "Oe", "Oo", "E1", "O1", "E2", "O2")]
    assert all(r["N_mult8"] and r["K_mult8"] for r in mirror8)            # align 8 pads every GEMM dimension
    assert not all(r["N_mult8"] and r["K_mult8"] for r in s1)             # d+ = 37 is not a multiple of 8
    dense = BE.gemm_shapes(BE.FeatureModule(ad, ad.model), T, 3)
    assert {(r["N"], r["K"]) for r in dense} >= {(3 * 64, 64), (64, 64), (256, 64), (64, 256)}
    t = BE.gemm_table(s8[:3], device="cpu", dtype=torch.float32, warmup=1, iters=2, timer=FakeTimer([1.0]))
    assert all(r["tflops"] == pytest.approx(2 * r["M"] * r["N"] * r["K"] / 1e-3 / 1e12) for r in t)


def test_group_categories():
    cats = {"gemm": (6.0, 10), "memory": (1.0, 5), "elementwise": (1.0, 4), "triton": (0.5, 2), "attention": (1.0, 1),
            "norm": (0.3, 3), "other": (0.2, 1)}
    g = BE.group_categories(cats)
    assert g["ms"]["GEMM"] == 6.0 and g["ms"]["butterflies and permutes"] == 2.5 and g["ms"]["rest"] == pytest.approx(0.5)
    assert sum(g["share"].values()) == pytest.approx(1.0) and g["total_ms"] == pytest.approx(10.0)


def test_run_benchmark_resume_and_table(tiny):
    ad, model, st = tiny
    orig = BE.BenchEntry("original", BE.FeatureModule(ad, copy.deepcopy(ad.model)), "original",
                         params_analytic=100, analytic={"gflops": 1.0, "source": "test"})
    eo = BE.BenchEntry("mirror_even_odd", build_even_odd(model, st, align=8), "mirror_even_odd", params_analytic=70,
                       analytic={"gflops": 0.8})
    fa = BE.BenchEntry("flipavg", BE.FlipAverage(orig.module), "flip_average", passes=2, params_analytic=100,
                       analytic=orig.analytic)
    timer = FakeTimer([4.0, 2.0])
    rows = BE.run_benchmark([orig, eo, fa], (1, 2), img_px=56, device="cpu", dtype=torch.bfloat16, compile=False,
                            warmup=1, iters=2, timer=timer)
    assert len(rows) == 6 and {r["name"] for r in rows} == {"original", "mirror_even_odd", "flipavg"}
    r = next(r for r in rows if r["name"] == "flipavg" and r["batch"] == 2)
    assert r["gflops_analytic"] == pytest.approx(2.0) and r["latency_ms_median"] == pytest.approx(3.0)
    assert r["throughput_img_s"] == pytest.approx(2 * 1e3 / 3.0) and r["dtype"] == "bfloat16"
    done = {(x["name"], x["batch"]) for x in rows}
    assert BE.run_benchmark([orig, eo, fa], (1, 2), img_px=56, device="cpu", compile=False, timer=timer,
                            done=done) == []
    table = BE.systems_table(rows)
    t = {x["name"]: x for x in table}
    assert t["flipavg"]["speed_vs_original_b1"] == pytest.approx(1.0) and t["original"]["params_analytic"] == 100
    assert set(t["mirror_even_odd"]) >= {"throughput_b1", "throughput_b2", "latency_ms_b1", "param_mb_bf16"}


def test_profile_summary_cpu(tiny):
    ad, model, st = tiny
    p = BE.profile_summary(build_even_odd(model, st, align=8), 1, img_px=56, device="cpu", dtype=torch.float32,
                           compile=False, iters=1, warmup=1)
    assert "cpu operator self time" in p["time_source"]
    assert p["groups"]["total_ms"] > 0 and sum(p["groups"]["share"].values()) == pytest.approx(1.0)
    assert "GEMM" in p["groups"]["ms"] and p["top_kernels"]


def test_a12_script_dry_run(tmp_path):
    spec = importlib.util.spec_from_file_location("fp_a12_test", ROOT / "scripts" / "fp_a12_bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "a12"
    res = mod.main(["--dry-run", "--out", str(out)])
    names = {r["name"] for r in res["rows"]}
    assert names == {"original", "mirror_dense", "mirror_even_odd", "svd_r8", "flipavg_original"}
    assert len(res["rows"]) == 10 and set(res["profile"]) == {"mirror_even_odd", "original"}
    assert {r["align"] for r in res["align"]["rows"]} == {1, 8} and res["gemm"]
    for f in ("systems_rows.csv", "systems_table.csv", "profile.json", "gemm.csv", "align.csv", "align_gemm.csv",
              "summary.md", "metrics.json", "meta.json"):
        assert (out / f).exists(), f
    t = {x["name"]: x for x in res["table"]}
    assert t["mirror_even_odd"]["entries_stored"] < t["mirror_dense"]["entries_stored"]
    assert t["flipavg_original"]["gflops_analytic"] == pytest.approx(2 * t["original"]["gflops_analytic"])
    again = mod.main(["--dry-run", "--out", str(out)])                     # resume: rows read back, none re-timed
    assert len(again["rows"]) == 10 and all(math.isfinite(float(r["latency_ms_median"])) for r in again["rows"])
