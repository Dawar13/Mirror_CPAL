"""A12b: a fused kernel for the even/odd MLP of a converted model (FINAL_PHASE_PLAN.md v2, A12b and Section 6;
PREREG_FINAL_PHASE.md: "throughput at batch 64 and at batch 256, each at least 1.1 times the original's, same device
and settings"). The timing and the GPU correctness test are in scripts/fp_a12b_kernel.py.

What is computed. The even/odd MLP of mirror/efficient.py (MirrorMLP; notes/paper/method_spec.md Sections 2 and 5),
with the residual stream kept in the eigenbasis of A* in every layer (one basis for all layers, as
mirror.efficient.build_fast_vit and mirror.final.audit.exactness.build_even_odd build the converted model). A token is
[x+, x-] (the d+ even coordinates, then the d- odd ones), the mirror is a sign flip of the odd coordinates and no
block needs a basis change.

Layout (pre-permuted and padded; taken from a MirrorMLP built with align 8, the pair and self-twin widths re-padded
to multiples of `align_hidden`, default 16, so that the row strides of h and dd are 32-byte multiples and Triton
compiles one specialisation for every block). Hidden units are ordered [twin pairs | self-twins], and the two twins
i and j = perm[i] of pair p share column p (one row a+ and one row a- per pair), so pairs are contiguous and nothing
is gathered:
  E1 [npad + nspad, ke]  even rows: a+ of every pair, then c+ of every self-twin; reads x[..., :ke] (ke = d+ rounded
                         up to a multiple of 8, 700 -> 704; the extra columns read odd coordinates against zero weights)
  b1 [npad + nspad]      fc1 bias (one per pair, shared by its twins)
  O1 [npad, d - ko]      odd rows: a- of every pair; reads x[..., ko:] (ko = d+ rounded down to a multiple of 8, so
                         68 -> 72 columns; the first d+ - ko columns read even coordinates against zero weights)
  E2 [ke_out, npad + nspad], b2 [ke_out]   even output rows (p+ of the pairs, q+ of the self-twins), zero beyond d+
  O2 [d - ko_out, npad]  odd output rows (p- of the pairs), zero rows before the odd coordinates
Padding pairs and self-twins are all zero (GELU(0) = 0, so they add nothing).

Per token (exact erf GELU, as nn.GELU()):
  u  = x[:ke] E1^T + b1                    every hidden column
  v  = x[ko:] O1^T                         pair columns
  ga = GELU(u_p + v),  gb = GELU(u_p - v)  twin i and twin j of pair p (first butterfly, then GELU)
  h  = [ga + gb | GELU(u_s)],  dd = ga - gb   (second butterfly)
  y  = [(h E2^T + b2)[:d+] | (dd O2^T)[cut:]]

Fused (Triton, one kernel per MLP): the even and the odd fc1 GEMM, the fc1 bias, the first butterfly, GELU and the
second butterfly. The hidden activations are written once, already in the fc2 layout (h and dd). Not fused: fc2 (two
GEMMs through torch F.linear, i.e. cuBLAS; fusion buys nothing there because h and dd are already the GEMM operands)
and the concatenation of the even and odd outputs (under torch.compile, Inductor fuses it with the residual addition
that follows). Attention is unchanged (MLP first; a fused attention is left for later, plan item B06).

Backends. "triton": CUDA tensors in bfloat16, float16 or float32 (float32 uses IEEE fp32 products, no TF32, for the
correctness test). "reference": pure PyTorch, any device and dtype, the same layout, padding and order of operations
(float32 accumulation for 16-bit inputs, bias added after the accumulation, butterflies and GELU in float32, h and dd
rounded to the input dtype before fc2). `available()` says whether the Triton path can run; builders take
backend="auto" (Triton when available, else the reference). Triton is imported lazily, only by `available()` /
`triton_status()` and only when CUDA is available, so this module imports and runs on a CPU-only machine.

Exactness. Negating the odd input coordinates negates v exactly (every product and partial sum changes sign; round
to nearest is symmetric), so the twins swap exactly, h is unchanged and dd changes sign exactly. In the reference the
mirror equivariance therefore holds bit for bit (up to the sign of exact zeros), as the plan says a sign flip does
(A09). The Triton launch disables the contraction of a multiply and an add into one FMA (enable_fp_fusion=False):
with it, the second butterfly could round twin i and twin j differently. Whether the GPU output is exactly
equivariant is measured by the script, not assumed.

torch.compile. The Triton launch sits inside the custom op mirror_a12b::fc1_butterfly (torch.library.custom_op, with a
fake implementation for tracing), so a compiled model calls it as one opaque kernel without a graph break.

Triton API, checked on 20 November 2026 against the Triton 3.6 source that torch 2.11 pins (triton==3.6.0 in the
wheel metadata) and against torch 2.11's Inductor, which emits libdevice.erf, tl.dot(..., input_precision=...) and the
enable_fp_fusion option: https://raw.githubusercontent.com/triton-lang/triton/release/3.6.x/python/triton/runtime/jit.py
(annotations containing "constexpr", strings included, are constexprs; compile options such as enable_fp_fusion are
accepted as launch keywords; a kernel may reference only modules, Triton builtins, libdevice stubs and jitted
functions as globals). Not run on a GPU here: the version on the VM and the kernel itself are checked in W0 (verify).
The tile configurations below are untested guesses for an H200; the autotuner picks one per shape (verify).

Public API: available, triton_status, resolve_backend, fc1_reference, fc2_apply, fc1_triton, fc1_op, FusedMirrorMLP,
build_fused_mlp, build_fused_mlp_from_dense, fuse_fast_vit, fused_even_odd, autotune_report, sign_vector.
"""
from __future__ import annotations

import copy
import importlib.util

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...efficient import Basis, BlockSpec, FastViT, MirrorMLP, build_fast_vit, build_mirror_mlp, ceil_to

__all__ = [
    "available", "triton_status", "resolve_backend", "fc1_reference", "fc2_apply", "fc1_triton", "fc1_op",
    "FusedMirrorMLP", "build_fused_mlp", "build_fused_mlp_from_dense", "fuse_fast_vit", "fused_even_odd",
    "autotune_report", "sign_vector", "OP_QUALNAME",
]

OP_QUALNAME = "mirror_a12b::fc1_butterfly"
BACKENDS = ("triton", "reference")

# Filled by _load_triton() (module globals, because the kernel resolves `tl` and `libdevice` from this module).
triton = None
tl = None
libdevice = None
_KERNELS: dict | None = None
_LOAD_ERROR: str | None = None
_STATUS: dict | None = None
_OP = None


# ----------------------------------------------------------------------------------------------
# Reference (pure PyTorch, any device)
# ----------------------------------------------------------------------------------------------
def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def fc1_reference(x2: torch.Tensor, E1: torch.Tensor, b1: torch.Tensor, O1: torch.Tensor, ko: int,
                  npad: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The fused fc1 in PyTorch: (h [M, npad + nspad], dd [M, npad]) in the dtype of x2 [M, d].

    Same operations as the Triton kernel: u = x[:, :ke] E1^T then + b1, v = x[:, ko:] O1^T, ga = GELU(u_p + v),
    gb = GELU(u_p - v), h = [ga + gb | GELU(u_s)], dd = ga - gb. 16-bit inputs are computed in float32 (products of
    16-bit values are exact in float32, accumulation in float32) and h, dd are rounded to the input dtype."""
    ct = _compute_dtype(x2.dtype)
    ke, ko, npad = int(E1.shape[1]), int(ko), int(npad)
    u = torch.matmul(x2[:, :ke].to(ct), E1.to(ct).t()) + b1.to(ct)
    v = torch.matmul(x2[:, ko:].to(ct), O1.to(ct).t())
    up = u[:, :npad]
    ga = F.gelu(up + v)
    gb = F.gelu(up - v)
    h = torch.cat([ga + gb, F.gelu(u[:, npad:])], dim=1)
    return h.to(x2.dtype), (ga - gb).to(x2.dtype)


def fc2_apply(h: torch.Tensor, dd: torch.Tensor, E2: torch.Tensor, b2: torch.Tensor, O2: torch.Tensor, dpo: int,
              cut: int) -> torch.Tensor:
    """fc2 of the even/odd MLP (torch GEMMs): [ (h E2^T + b2)[:, :d+] | (dd O2^T)[:, cut:] ]."""
    ye = F.linear(h, E2, b2)
    yo = F.linear(dd, O2)
    return torch.cat([ye[:, :int(dpo)], yo[:, int(cut):]], dim=1)


def sign_vector(d: int, dp: int, dtype=torch.float32, device=None) -> torch.Tensor:
    """+1 on the d+ even coordinates, -1 on the odd ones: the mirror in the A* basis."""
    s = torch.ones(int(d), dtype=dtype, device=device)
    s[int(dp):] = -1
    return s


# ----------------------------------------------------------------------------------------------
# Triton (loaded lazily)
# ----------------------------------------------------------------------------------------------
def _fc1_kernel(x_ptr, e1_ptr, b1_ptr, o1_ptr, h_ptr, d_ptr, M, NE, NP,
                stride_xm, stride_e1, stride_o1, stride_hm, stride_dm,
                KE: tl.constexpr, KO: tl.constexpr, KO_OFF: tl.constexpr, IEEE: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_KO: tl.constexpr,
                GROUP_M: tl.constexpr):
    # Jitted by _load_triton (it is a plain function until then). One program computes a BLOCK_M x BLOCK_N tile of
    # the hidden layer. Tiles over twin-pair columns (n < NP) run the even and the odd GEMM, the butterflies and
    # GELU; the other tiles run the even GEMM and GELU. Rows and weight columns wrap (modulo) so that the loads in
    # the main loops need no mask; the stores are masked.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(NE, BLOCK_N)
    width = GROUP_M * num_pid_n
    group_id = pid // width
    first_m = group_id * GROUP_M
    group_m = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % width) % group_m)
    pid_n = (pid % width) // group_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rows = (offs_m % M).to(tl.int64)
    x_rows = x_ptr + rows[:, None] * stride_xm

    # u = x[:, :KE] E1^T + b1
    offs_k = tl.arange(0, BLOCK_K)
    e_cols = e1_ptr + (offs_n % NE)[None, :] * stride_e1
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, KE, BLOCK_K):
        kk = k0 + offs_k
        if KE % BLOCK_K == 0:
            a = tl.load(x_rows + kk[None, :])
            b = tl.load(e_cols + kk[:, None])
        else:
            a = tl.load(x_rows + kk[None, :], mask=kk[None, :] < KE, other=0.0)
            b = tl.load(e_cols + kk[:, None], mask=kk[:, None] < KE, other=0.0)
        if IEEE:
            acc_u = tl.dot(a, b, acc_u, input_precision="ieee")
        else:
            acc_u = tl.dot(a, b, acc_u)
    bias = tl.load(b1_ptr + (offs_n % NE)).to(tl.float32)
    u = acc_u + bias[None, :]

    out_rows = offs_m.to(tl.int64)[:, None]
    row_ok = (offs_m < M)[:, None]
    h_ptrs = h_ptr + out_rows * stride_hm + offs_n[None, :]
    h_mask = row_ok & (offs_n < NE)[None, :]
    if pid_n * BLOCK_N < NP:
        # v = x[:, KO_OFF:] O1^T on the pair columns, then butterfly, GELU, butterfly.
        offs_ko = tl.arange(0, BLOCK_KO)
        o_cols = o1_ptr + (offs_n % NP)[None, :] * stride_o1
        acc_v = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, KO, BLOCK_KO):
            kk = k0 + offs_ko
            if KO % BLOCK_KO == 0:
                a = tl.load(x_rows + KO_OFF + kk[None, :])
                b = tl.load(o_cols + kk[:, None])
            else:
                a = tl.load(x_rows + KO_OFF + kk[None, :], mask=kk[None, :] < KO, other=0.0)
                b = tl.load(o_cols + kk[:, None], mask=kk[:, None] < KO, other=0.0)
            if IEEE:
                acc_v = tl.dot(a, b, acc_v, input_precision="ieee")
            else:
                acc_v = tl.dot(a, b, acc_v)
        pi = u + acc_v
        pj = u - acc_v
        ga = pi * 0.5 * (1.0 + libdevice.erf(pi * 0.7071067811865476))
        gb = pj * 0.5 * (1.0 + libdevice.erf(pj * 0.7071067811865476))
        is_pair = (offs_n < NP)[None, :]
        if (pid_n + 1) * BLOCK_N <= NP:
            hv = ga + gb
        else:
            gu = u * 0.5 * (1.0 + libdevice.erf(u * 0.7071067811865476))
            hv = tl.where(is_pair, ga + gb, gu)
        tl.store(h_ptrs, hv.to(h_ptr.dtype.element_ty), mask=h_mask)
        d_ptrs = d_ptr + out_rows * stride_dm + offs_n[None, :]
        tl.store(d_ptrs, (ga - gb).to(d_ptr.dtype.element_ty), mask=row_ok & is_pair)
    else:
        gu = u * 0.5 * (1.0 + libdevice.erf(u * 0.7071067811865476))
        tl.store(h_ptrs, gu.to(h_ptr.dtype.element_ty), mask=h_mask)


def _configs(triton_mod, kind: str) -> list:
    """Autotune candidates. "exact" (float32, IEEE products): one small tile, correctness only. "fast" (16-bit): tile
    shapes for Hopper chosen without measurement; the autotuner times them per (M, NE, NP) (verify on the VM)."""
    C = triton_mod.Config
    if kind == "exact":
        return [C({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "BLOCK_KO": 16, "GROUP_M": 8}, num_warps=4,
                  num_stages=2)]
    tiles = ((128, 128, 64, 16, 8, 3), (128, 128, 64, 32, 8, 3), (128, 128, 64, 16, 4, 4), (128, 64, 64, 16, 4, 4),
             (64, 128, 64, 16, 4, 4), (128, 128, 32, 32, 4, 4), (64, 64, 64, 16, 4, 3), (128, 64, 32, 32, 4, 5))
    return [C({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "BLOCK_KO": bko, "GROUP_M": 8}, num_warps=w, num_stages=s)
            for bm, bn, bk, bko, w, s in tiles]


def _load_triton() -> dict:
    """Import Triton, bind `tl` and `libdevice` for the kernel and build the autotuned kernels (once)."""
    global triton, tl, libdevice, _KERNELS, _LOAD_ERROR
    if _KERNELS is not None:
        return _KERNELS
    if _LOAD_ERROR is not None:
        raise RuntimeError(_LOAD_ERROR)
    try:
        import triton as _triton
        import triton.language as _tl
        try:
            from triton.language.extra import libdevice as _ld        # Triton 3.x, as torch 2.11's Inductor uses it
        except ImportError:
            _ld = _tl.math                                            # older Triton: erf lives in tl.math
        if not hasattr(_ld, "erf"):
            raise ImportError("no erf in Triton's libdevice or tl.math")
        triton, tl, libdevice = _triton, _tl, _ld
        jit = _triton.jit(_fc1_kernel)
        key = ["M", "NE", "NP"]
        kernels = {"fast": _triton.autotune(configs=_configs(_triton, "fast"), key=key)(jit),
                   "exact": _triton.autotune(configs=_configs(_triton, "exact"), key=key)(jit)}
    except Exception as e:  # noqa: BLE001
        _LOAD_ERROR = f"{type(e).__name__}: {e}"
        raise RuntimeError(_LOAD_ERROR) from e
    _KERNELS = kernels
    return kernels


def triton_status() -> dict:
    """{"ok", "reason", "cuda", "triton_version", "erf"}; computed once per process (Triton is imported here, and
    only when CUDA is available)."""
    global _STATUS
    if _STATUS is not None:
        return dict(_STATUS)
    st = {"ok": False, "reason": "", "cuda": bool(torch.cuda.is_available()), "triton_version": None, "erf": None}
    if not st["cuda"]:
        st["reason"] = "CUDA is not available"
    elif importlib.util.find_spec("triton") is None:
        st["reason"] = "Triton is not installed"
    else:
        try:
            _load_triton()
            st.update(ok=True, triton_version=getattr(triton, "__version__", None),
                      erf=f"{getattr(libdevice, '__name__', '?')}.erf")
        except Exception as e:  # noqa: BLE001
            st["reason"] = f"Triton could not be loaded: {e}"
    _STATUS = st
    return dict(st)


def available() -> bool:
    """True when the Triton path can run here (CUDA and a working Triton)."""
    return bool(triton_status()["ok"])


def resolve_backend(backend: str = "auto") -> str:
    """"auto" -> "triton" when available, else "reference"; "triton" raises when Triton cannot run here."""
    if backend not in ("auto",) + BACKENDS:
        raise ValueError(f"backend must be auto, triton or reference, not {backend!r}")
    if backend == "reference":
        return "reference"
    st = triton_status()
    if st["ok"]:
        return "triton"
    if backend == "triton":
        raise RuntimeError(f"the Triton backend is not available: {st['reason']}")
    return "reference"


def fc1_triton(x2: torch.Tensor, E1: torch.Tensor, b1: torch.Tensor, O1: torch.Tensor, ko: int,
               npad: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The fused fc1 on the GPU (one Triton launch): (h, dd) as fc1_reference."""
    kernels = _load_triton()
    if not x2.is_cuda:
        raise RuntimeError("fc1_triton needs CUDA tensors")
    dt = x2.dtype
    if dt not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError(f"fc1_triton supports float32, bfloat16 and float16, not {dt}")
    if not (E1.dtype == b1.dtype == O1.dtype == dt):
        raise TypeError("x and the fc1 weights must have the same dtype")
    if x2.dim() != 2:
        raise ValueError("x must be [tokens, d]")
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    E1, b1, O1 = E1.contiguous(), b1.contiguous(), O1.contiguous()
    M, d = (int(s) for s in x2.shape)
    NE, KE = (int(s) for s in E1.shape)
    NP, KO, ko = int(npad), int(O1.shape[1]), int(ko)
    if int(O1.shape[0]) != NP or NP > NE or KE > d or ko + KO != d or int(b1.shape[0]) != NE:
        raise ValueError(f"inconsistent fc1 layout: x {tuple(x2.shape)}, E1 {tuple(E1.shape)}, O1 {tuple(O1.shape)}, "
                         f"ko {ko}, npad {NP}")
    h = torch.empty((M, NE), dtype=dt, device=x2.device)
    dd = torch.empty((M, NP), dtype=dt, device=x2.device)
    if M == 0:
        return h, dd
    kern = kernels["exact" if dt == torch.float32 else "fast"]
    cdiv = triton.cdiv

    def grid(meta):
        return (cdiv(M, meta["BLOCK_M"]) * cdiv(NE, meta["BLOCK_N"]),)

    with torch.cuda.device(x2.device):
        kern[grid](x2, E1, b1, O1, h, dd, M, NE, NP, x2.stride(0), E1.stride(0), max(int(O1.stride(0)), 1),
                   h.stride(0), max(int(dd.stride(0)), 1), KE=KE, KO=KO, KO_OFF=ko, IEEE=dt == torch.float32,
                   enable_fp_fusion=False)
    return h, dd


def _fc1_impl(x, E1, b1, O1, ko, npad):
    if x.is_cuda and available():
        return fc1_triton(x, E1, b1, O1, ko, npad)
    return fc1_reference(x, E1, b1, O1, ko, npad)


def _op_exists(ns: str, name: str) -> bool:
    try:
        getattr(getattr(torch.ops, ns), name)
        return True
    except (AttributeError, RuntimeError):
        return False


def fc1_op():
    """The custom op mirror_a12b::fc1_butterfly(x, E1, b1, O1, ko, npad) -> (h, dd), registered on first use: the
    Triton kernel for CUDA tensors when available, the reference otherwise; a fake implementation for tracing."""
    global _OP
    if _OP is not None:
        return _OP
    ns, name = OP_QUALNAME.split("::")
    if not _op_exists(ns, name):
        @torch.library.custom_op(OP_QUALNAME, mutates_args=())
        def _fc1_butterfly(x: torch.Tensor, E1: torch.Tensor, b1: torch.Tensor, O1: torch.Tensor, ko: int,
                           npad: int) -> tuple[torch.Tensor, torch.Tensor]:
            return _fc1_impl(x, E1, b1, O1, ko, npad)

        @_fc1_butterfly.register_fake
        def _fc1_butterfly_fake(x, E1, b1, O1, ko, npad):
            return x.new_empty((x.shape[0], E1.shape[0])), x.new_empty((x.shape[0], npad))
    _OP = getattr(getattr(torch.ops, ns), name).default
    return _OP


def autotune_report() -> dict:
    """The configuration the autotuner chose per key (M, NE, NP), for each kernel that ran (empty without Triton)."""
    if _KERNELS is None:
        return {}
    out = {}
    for kind, k in _KERNELS.items():
        cache = getattr(k, "cache", None) or {}
        out[kind] = {str(key): str(cfg) for key, cfg in cache.items()}
    return out


# ----------------------------------------------------------------------------------------------
# Module and builders
# ----------------------------------------------------------------------------------------------
class FusedMirrorMLP(nn.Module):
    """The even/odd MLP with fc1 fused (see the module docstring). Inputs and outputs are tokens in the A* basis,
    shape [..., d]. `backend` is "triton" (CUDA tensors only; it never falls back silently) or "reference"."""

    def __init__(self, E1, b1, O1, E2, b2, O2, *, ke: int, ko: int, npad: int, dpo: int, cut: int, backend: str,
                 cost: dict | None = None, layout: dict | None = None):
        super().__init__()
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, not {backend!r}")
        for name, t in (("E1", E1), ("b1", b1), ("O1", O1), ("E2", E2), ("b2", b2), ("O2", O2)):
            setattr(self, name, nn.Parameter(t.detach().clone().contiguous(), requires_grad=False))
        self.ke, self.ko, self.npad, self.dpo, self.cut = int(ke), int(ko), int(npad), int(dpo), int(cut)
        self.d = self.ko + int(self.O1.shape[1])
        self.backend = backend
        self.cost = dict(cost or {})
        self.layout = dict(layout or {})
        self._check()
        if backend == "triton":
            fc1_op()                    # register the custom op now, so torch.compile never traces its registration

    def _check(self) -> None:
        NE, KE = (int(s) for s in self.E1.shape)
        ok = (KE == self.ke <= self.d and tuple(self.b1.shape) == (NE,) and int(self.O1.shape[0]) == self.npad <= NE
              and int(self.E2.shape[1]) == NE and int(self.O2.shape[1]) == self.npad
              and tuple(self.b2.shape) == (int(self.E2.shape[0]),) and self.dpo <= int(self.E2.shape[0])
              and self.cut <= int(self.O2.shape[0]) and self.dpo + int(self.O2.shape[0]) - self.cut == self.d)
        if not ok:
            raise ValueError("inconsistent even/odd MLP layout: " + self.extra_repr())

    @property
    def nspad(self) -> int:
        return int(self.E1.shape[0]) - self.npad

    def fc1(self, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.backend == "triton":
            if not x2.is_cuda:
                raise RuntimeError("this FusedMirrorMLP uses the Triton backend, which needs CUDA tensors; build it "
                                   "with backend='reference' for the CPU")
            op = _OP if _OP is not None else fc1_op()
            return op(x2, self.E1, self.b1, self.O1, self.ko, self.npad)
        return fc1_reference(x2, self.E1, self.b1, self.O1, self.ko, self.npad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        h, dd = self.fc1(x2)
        y = fc2_apply(h, dd, self.E2, self.b2, self.O2, self.dpo, self.cut)
        return y.reshape(*lead, y.shape[-1])

    def extra_repr(self) -> str:
        return (f"d={self.d}, ke={self.ke}, ko={self.ko}, npad={self.npad}, nspad={self.nspad}, dpo={self.dpo}, "
                f"cut={self.cut}, backend={self.backend}")


def _rows(M: torch.Tensor, n: int) -> torch.Tensor:
    out = M.new_zeros(int(n), M.shape[1])
    out[:M.shape[0]] = M
    return out


def _cols(M: torch.Tensor, n: int) -> torch.Tensor:
    out = M.new_zeros(M.shape[0], int(n))
    out[:, :M.shape[1]] = M
    return out


def _vec(v: torch.Tensor, n: int) -> torch.Tensor:
    out = v.new_zeros(int(n))
    out[:v.shape[0]] = v
    return out


def _is_exact_gelu(act: nn.Module) -> bool:
    return isinstance(act, nn.GELU) and getattr(act, "approximate", "none") == "none"


@torch.no_grad()
def build_fused_mlp(mlp: MirrorMLP, backend: str = "auto", align_hidden: int = 16) -> FusedMirrorMLP:
    """FusedMirrorMLP computing the same function as `mlp` (mirror.efficient.MirrorMLP, any alignment), in its dtype
    and on its device. The pair and self-twin widths are padded with zero units to multiples of `align_hidden`."""
    if not isinstance(mlp, MirrorMLP):
        raise TypeError(f"expected mirror.efficient.MirrorMLP, got {type(mlp).__name__}")
    if not _is_exact_gelu(mlp.act):
        raise ValueError(f"the fused kernel computes the exact (erf) GELU; this MLP uses {mlp.act!r}")
    E1, b1, O1, E2, b2, O2 = (t.detach() for t in (mlp.E1, mlp.b1, mlp.O1, mlp.E2, mlp.b2, mlp.O2))
    npad = int(mlp.npad)
    nspad = int(E1.shape[0]) - npad
    a = max(1, int(align_hidden))
    np2, ns2 = ceil_to(npad, a), ceil_to(nspad, a)
    if (np2, ns2) != (npad, nspad):
        E1 = torch.cat([_rows(E1[:npad], np2), _rows(E1[npad:], ns2)], 0)
        b1 = torch.cat([_vec(b1[:npad], np2), _vec(b1[npad:], ns2)])
        O1 = _rows(O1, np2)
        E2 = torch.cat([_cols(E2[:, :npad], np2), _cols(E2[:, npad:], ns2)], 1)
        O2 = _cols(O2, np2)
    exec_ = int(E1.numel() + O1.numel() + E2.numel() + O2.numel())
    cost = dict(mlp.cost)
    cost.update({"macs_token_exec": exec_, "stored_params": exec_ + int(b1.numel() + b2.numel()), "fused": True})
    layout = {"d": int(mlp.ko + O1.shape[1]), "ke": int(mlp.ke), "ko": int(mlp.ko), "KO": int(O1.shape[1]),
              "npad": np2, "nspad": ns2, "dpo": int(mlp.dpo), "cut": int(mlp.cut),
              "n_pairs": cost.get("n_pairs"), "n_self": cost.get("n_self"), "align_hidden": a}
    return FusedMirrorMLP(E1, b1, O1, E2, b2, O2, ke=int(mlp.ke), ko=int(mlp.ko), npad=np2, dpo=int(mlp.dpo),
                          cut=int(mlp.cut), backend=resolve_backend(backend), cost=cost, layout=layout)


@torch.no_grad()
def build_fused_mlp_from_dense(W1, b1, W2, b2, A, perm, act: nn.Module | None = None, basis: Basis | None = None,
                               align: int = 8, align_hidden: int = 16, backend: str = "auto",
                               tol: float = 1e-5) -> FusedMirrorMLP:
    """From exactly equivariant dense weights (P W1 = W1 A, P b1 = b1, A W2 = W2 P, A b2 = b2; fc1.weight [h, d],
    fc2.weight [d, h]), the operator A, the pairing `perm` and the stream basis (default eig(A)): the module reads
    and writes tokens x Q. Built through mirror.efficient.build_mirror_mlp (float64, structure checked to `tol`)."""
    basis = basis if basis is not None else Basis.of(A)
    mm = build_mirror_mlp(W1, b1, W2, b2, act if act is not None else nn.GELU(), basis, basis, A, A, perm,
                          align=align, tol=tol)
    return build_fused_mlp(mm, backend=backend, align_hidden=align_hidden)


@torch.no_grad()
def fuse_fast_vit(fv: FastViT, backend: str = "auto", align_hidden: int = 16, copy_model: bool = True) -> FastViT:
    """A FastViT whose MirrorMLP blocks are replaced by FusedMirrorMLP (a deep copy unless copy_model is False).
    Attention, norms, embedding and head are unchanged."""
    out = copy.deepcopy(fv) if copy_model else fv
    be = resolve_backend(backend)
    n = 0
    for blk in out.blocks:
        if isinstance(blk.mlp, MirrorMLP):
            blk.mlp = build_fused_mlp(blk.mlp, backend=be, align_hidden=align_hidden)
            n += 1
    out.info = dict(getattr(out, "info", {}) or {})
    out.info.update({"fused_mlps": n, "fused_backend": be})
    return out


@torch.no_grad()
def fused_even_odd(model: nn.Module, st, align: int = 8, backend: str = "auto", attention: str = "even_odd",
                   align_hidden: int = 16, tol: float = 1e-5) -> FastViT:
    """The even/odd form of a fully converted model (as mirror.final.audit.exactness.build_even_odd: one basis
    eig(A*) for every layer, Wo structured) with fused MLPs. attention="dense" keeps attention dense-stored in the
    A* basis (rotation folded into its weights: the same function, without the attention parameter saving).
    `st` needs A, perms and maps (head maps after the twin gauge), e.g. an AuditStructure."""
    if attention not in ("even_odd", "dense"):
        raise ValueError("attention must be even_odd or dense")
    L = len(model.blocks)
    if len(st.perms) < L or (attention == "even_odd" and len(st.maps) < L):
        raise ValueError(f"the fused form needs the full conversion ({L} blocks)")
    A = st.A
    if attention == "even_odd":
        specs = [BlockSpec(("mirror", A, A, st.maps[b]), ("mirror", A, A, st.perms[b])) for b in range(L)]
    else:
        specs = [BlockSpec(("dense",), ("mirror", A, A, st.perms[b])) for b in range(L)]
    fv = build_fast_vit(model, specs, [Basis.of(A)] * (L + 1), align=int(align), wo_mode="fold", tol=tol)
    out = fuse_fast_vit(fv, backend=backend, align_hidden=align_hidden, copy_model=False)
    out.info["attention"] = attention
    return out
