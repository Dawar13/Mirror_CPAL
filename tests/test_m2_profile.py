"""scripts/m2_profile.py: kernel-name categories, event aggregation, derived gaps, and the --tiny CPU path."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from torch.autograd import DeviceType

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_cli", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def prof():
    return load_script("m2_profile")


# Representative names as torch.profiler reports them (cuBLAS, CUTLASS, cuDNN, flash and memory-efficient
# attention, PyTorch native kernels, Inductor Triton kernels, runtime copies) and CPU operator names.
NAMES = [
    ("ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_tn", "gemm"),
    ("sm80_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize160x128x32_stage4_warpsize2x2x1_tensor16x8x16_kernel", "gemm"),
    ("void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_64x4_tn_align8>"
     "(cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_64x4_tn_align8::Params)", "gemm"),
    ("void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, false, __nv_bfloat16>"
     "(cublasLt::cublasSplitKParams<float>, float const*)", "gemm"),
    ("void gemmk1_kernel<int, float, 256, 5, false, false, false, false, cublasGemvTensorStridedBatched<"
     "__nv_bfloat16 const>, float, 0>(cublasGemmk1Params<float, float const, float, 0>)", "gemm"),
    ("sm80_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32_nhwckrsc_nchw_tilesize128x128x32_stage4", "gemm"),
    ("triton_tem_fused_addmm_0", "gemm"),
    ("aten::addmm", "gemm"),
    ("aten::_slow_conv2d_forward", "gemm"),
    ("void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, "
     "cutlass::bfloat16_t, Flash_kernel_traits<64, 128, 128, 4, cutlass::bfloat16_t> >, false, false, false, "
     "false, true, true, false, false>(pytorch_flash::Flash_fwd_params)", "attention"),
    ("fmha_cutlassF_bf16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::bfloat16_t, "
     "cutlass::arch::Sm80, true, 64, 128, 64, true, true>::Params)", "attention"),
    ("cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16_knob_3_64x64x64_4x1x1_kernel0_0", "attention"),
    ("aten::_scaled_dot_product_flash_attention_for_cpu", "attention"),
    ("triton_poi_fused_add_cat_gelu_sub_3", "triton"),
    ("triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_4", "triton"),
    ("triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_7", "triton"),
    ("void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl(at::TensorIteratorBase&, "
     "at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#4}::operator()() const::"
     "{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::GeluCUDAKernelImpl(at::"
     "TensorIteratorBase&, at::native::GeluType)::{lambda()#2}::operator()() const::{lambda()#4}::operator()() "
     "const::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>)", "elementwise"),
    ("void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c10::BFloat16>, "
     "std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>)",
     "elementwise"),
    ("void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>, "
     "4, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, "
     "at::native::memory::LoadWithCast<2>, at::native::memory::StoreWithCast<1> >(int, "
     "at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>)", "elementwise"),
    ("void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<"
     "c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor<float> > >"
     "(at::TensorIteratorBase&, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, "
     "at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, ...)", "elementwise"),
    ("aten::gelu", "elementwise"),
    ("aten::sub", "elementwise"),
    ("void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<c10::BFloat16, float, true>"
     "(int, float, c10::BFloat16 const*, c10::BFloat16 const*, c10::BFloat16 const*, float*, float*, "
     "c10::BFloat16*)", "norm"),
    ("void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, "
     "float>, unsigned int, float, 4, 4> >(at::native::ReduceOp<float, at::native::MeanOps<float, float, float, "
     "float>, unsigned int, float, 4, 4>)", "norm"),
    ("void at::native::vectorized_elementwise_kernel<4, at::native::rsqrt_kernel_cuda(at::TensorIteratorBase&)::"
     "{lambda()#2}::operator()() const::{lambda()#4}::operator()() const::{lambda(float)#1}, "
     "std::array<char*, 2ul> >(int, ...)", "norm"),
    ("aten::_fused_rms_norm", "norm"),
    ("aten::mean", "norm"),
    ("void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::"
     "direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::"
     "operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, ...)::{lambda(int)#1}>(int, ...)",
     "memory"),
    ("void at::native::(anonymous namespace)::CatArrayBatchedCopy_aligned16_contig<at::native::(anonymous "
     "namespace)::OpaqueType<2u>, unsigned int, 3, 128, 1>(...)", "memory"),
    ("void at::native::(anonymous namespace)::indexSelectLargeIndex<c10::BFloat16, long, unsigned int, 2, 2, -2, "
     "true>(at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, ...)", "memory"),
    ("void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<...>>(long, ...)", "memory"),
    ("void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<c10::BFloat16>, "
     "std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>)", "memory"),
    ("void cudnn::engines_precompiled::nchwToNhwcKernel<__nv_bfloat16, __nv_bfloat16, float, false, true, "
     "(cudnnKernelDataType_t)2>(cudnn::engines_precompiled::nchw2nhwc_params_t<float>, ...)", "memory"),
    ("Memcpy DtoD (Device -> Device)", "memory"),
    ("Memset (Device)", "memory"),
    ("aten::cat", "memory"),
    ("aten::index_select", "memory"),
    ("aten::_to_copy", "memory"),
    ("aten::constant_pad_nd", "memory"),
    ("aten::empty", "other"),
    ("void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, "
     "at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, "
     "unsigned int, float, 4> >(...)", "other"),
    ("some_unknown_kernel_xyz", "other"),
]


@pytest.mark.parametrize("name,cat", NAMES)
def test_classify_kernel(prof, name, cat):
    assert prof.classify_kernel(name) == cat


def test_categories_complete(prof):
    assert set(prof.CATEGORIES) == set(prof.CATEGORY_LABELS) == set(prof.SHORT)
    assert {prof.classify_kernel(n) for n, _ in NAMES} <= set(prof.CATEGORIES)


def test_triton_tag(prof):
    assert prof.triton_tag("triton_poi_fused_add_cat_gelu_sub_3") == "cat/gather/pad + activation"
    assert prof.triton_tag("triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_4") == "norm"
    assert prof.triton_tag("triton_poi_fused_constant_pad_nd_index_select_1") == "cat/gather/pad"
    assert prof.triton_tag("triton_poi_fused__to_copy_add_mul_0") == "pointwise"
    assert prof.triton_tag("triton_poi_fused_gelu_2") == "activation"
    assert prof.triton_tag("triton_") == "unnamed"


def _ev(name, dev, start, end, annotation=False, self_cpu=0.0):
    return SimpleNamespace(name=name, device_type=dev, time_range=SimpleNamespace(start=start, end=end),
                           is_user_annotation=annotation, self_cpu_time_total=self_cpu)


def test_aggregate_events_gpu_and_cpu(prof):
    cuda, cpu = DeviceType.CUDA, DeviceType.CPU
    events = [
        _ev("aten::linear", cpu, 0, 50, self_cpu=5.0),                       # CPU op: ignored on the GPU path
        _ev("## Call CompiledFxGraph abc ##", cuda, 0, 100, annotation=True),  # GPU annotation range: excluded
        _ev("Torch-Compiled Region: 0/0", cuda, 0, 100),                      # name-filtered even without the flag
        _ev("ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_tn", cuda, 10, 40),
        _ev("ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_tn", cuda, 50, 70),
        _ev("triton_poi_fused_add_cat_gelu_sub_3", cuda, 70, 80),
    ]
    df, info = prof.aggregate_events(events, gpu=True)
    got = dict(zip(df["name"], zip(df["total_us"], df["calls"])))
    assert got == {"ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_tn": (50.0, 2),
                   "triton_poi_fused_add_cat_gelu_sub_3": (10.0, 1)}
    assert info["annotations_skipped"] == 2
    assert info["kernel_span_us"] == 70.0 and info["kernel_sum_over_span"] == pytest.approx(60 / 70)
    df_cpu, info_cpu = prof.aggregate_events(events, gpu=False)
    assert list(df_cpu["name"]) == ["aten::linear"] and float(df_cpu["total_us"].iloc[0]) == 5.0
    assert "cpu" in info_cpu["time_source"]


def test_kernel_tables(prof):
    kern = pd.DataFrame({"name": ["ampere_bf16_s16816gemm_x", "triton_poi_fused_add_cat_0", "Memcpy DtoD (Device -> Device)",
                                  "void pytorch_flash::flash_fwd_kernel<x>(y)"],
                         "total_us": [6000.0, 2000.0, 1000.0, 1000.0], "calls": [8, 4, 2, 2]})
    cats, top, allk = prof.kernel_tables(kern, iters=2, top=3)
    assert cats["gemm"] == (3.0, 4.0) and cats["triton"] == (1.0, 2.0)
    assert cats["memory"] == (0.5, 1.0) and cats["attention"] == (0.5, 1.0) and cats["other"] == (0.0, 0.0)
    assert sum(ms for ms, _ in cats.values()) == pytest.approx(5.0)
    assert allk["share"].sum() == pytest.approx(1.0)
    assert list(top["rank"]) == [1, 2, 3] and top["category"].iloc[0] == "gemm"
    assert top.loc[top["category"] == "triton", "triton_tag"].iloc[0] == "cat/gather/pad"


def test_add_derived_gap_decomposition(prof):
    base = {"mode": "compile", "batch": 4, "macs_per_image": 100.0, "gemm_gflop": 10.0}
    rows = []
    for v, lat, cats in (("dense", 10.0, [5, 1, 2, 0.5, 0.5, 0.5, 0.1]), ("vexact", 12.5, [5.5, 1, 3, 0.4, 0.6, 0.9, 0.1])):
        r = {**base, "variant": v, "k": 0 if v == "dense" else 6, "latency_ms_median": lat,
             "kernel_ms": float(sum(cats)), "launches": 10.0}
        r.update({f"ms_{c}": float(x) for c, x in zip(prof.CATEGORIES, cats)})
        r["macs_per_image"] = 100.0 if v == "dense" else 80.0
        r["gemm_gflop"] = 10.0 if v == "dense" else 8.0
        rows.append(r)
    df = prof.add_derived(pd.DataFrame(rows))
    vx = df[df["variant"] == "vexact"].iloc[0]
    parts = sum(vx[f"d_{c}"] for c in prof.CATEGORIES) + vx["d_idle_ms"]
    assert vx["gap_ms"] == pytest.approx(2.5) and parts == pytest.approx(vx["gap_ms"])
    assert vx["speedup_vs_dense"] == pytest.approx(10 / 12.5) and vx["flop_speedup_vs_dense"] == pytest.approx(1.25)
    assert vx["gemm_ms_at_dense_rate"] == pytest.approx(5.0 * 0.8)


def test_tiny_cli_writes_all_files(prof, tmp_path):
    variants = ["dense", "vexact", "vlocal", "mlp_only", "svd", "prune"]
    res = prof.main(["--tiny", "--out", str(tmp_path), "--variants", ",".join(variants), "--k", "2",
                     "--batch", "2", "--modes", "eager", "--warmup", "1", "--iters", "2", "--prof-iters", "2",
                     "--sub-warmup", "1", "--sub-iters", "2"])
    for f in ("m2_profile.csv", "m2_submodules.csv", "m2_triton.csv", "m2_top_kernels.csv", "M2_PROFILE.md",
              "m2_profile_meta.json"):
        assert (tmp_path / f).is_file(), f
    for v in variants:
        assert (tmp_path / "top_kernels" / f"{v}_k{0 if v == 'dense' else 2}_eager.csv").is_file(), v
    p = pd.read_csv(tmp_path / "m2_profile.csv")
    assert set(p["variant"]) == set(variants)
    assert p.groupby(["variant", "k", "mode"])["category"].apply(set).map(lambda s: s == set(prof.CATEGORIES)).all()
    for col in ("ms_per_forward", "share", "kernel_count", "latency_ms_median", "delta_ms_vs_dense", "git_hash", "seed"):
        assert col in p.columns, col
    assert (p["seed"] == 0).all() and p["git_hash"].notna().all()
    assert p["time_source"].str.contains("cpu").all()
    shares = p.groupby(["variant", "k", "mode"])["share"].sum()
    assert ((shares - 1.0).abs() < 1e-9).all()
    assert (p[(p["variant"] == "dense") & (p["category"] == "gemm")]["ms_per_forward"] > 0).all()
    # Gap = sum of category changes + idle change, for every row.
    runs = res["runs"]
    for _, r in runs.iterrows():
        parts = sum(r[f"d_{c}"] for c in prof.CATEGORIES) + r["d_idle_ms"]
        assert parts == pytest.approx(r["gap_ms"], abs=1e-9)
    s = pd.read_csv(tmp_path / "m2_submodules.csv")
    got = {(v, part) for v, part in zip(s["variant"], s["part"])}
    assert {("vexact", "mlp"), ("vexact", "attention"), ("vlocal", "mlp"), ("vlocal", "attention"),
            ("vlocal", "rotation"), ("mlp_only", "mlp"), ("mlp_only", "rotation"), ("svd", "mlp"),
            ("svd", "attention"), ("prune", "mlp")} <= got
    assert ("prune", "attention") not in got and ("mlp_only", "attention") not in got and ("vexact", "rotation") not in got
    assert set(s["block"]) == {0, 1} and s["ms_median"].gt(0).all() and s["dense_ms_median"].ge(0).all()
    assert (s[s["part"] == "rotation"]["dense_ms_median"] == 0).all()
    top = pd.read_csv(tmp_path / "m2_top_kernels.csv")
    assert top.groupby(["variant", "k", "mode"])["rank"].max().le(30).all()
    text = (tmp_path / "M2_PROFILE.md").read_text(encoding="utf-8")
    assert chr(0x2014) not in text and chr(0x2013) not in text
    assert "TINY TEST" in text and "## Converted sub-modules timed alone" in text and "## GEMM efficiency" in text
    meta = json.loads((tmp_path / "m2_profile_meta.json").read_text())
    assert meta["tiny"] and meta["seed"] == 0 and meta["git"]["git_hash"]
    assert set(meta["nvidia_smi"]) == {"before", "after"} and meta["batch"] == 2
