# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors

"""OSS Benchmark for attention kernel implementations.

Compares four kernel implementations:
  1. CuteDSL GDPA  - CuTe DSL GDPA kernel with gelu activation
  2. Triton GDPA   - Triton GDPA kernel with gelu activation
  3. CUTLASS FMHA  - NVIDIA CUTLASS Blackwell FMHA (softmax, via fbgemm-gpu-genai)
  4. CuteDSL FA4   - Tri Dao's Flash Attention v4 (softmax)

Setup:
    bash ads_mkl/ops/cute_dsl/gdpa/scripts/setup_benchmark_env.sh

Usage:
    conda activate gdpa-bench
    python ads_mkl/ops/cute_dsl/gdpa/scripts/benchmark.py bench-fwd
    python ads_mkl/ops/cute_dsl/gdpa/scripts/benchmark.py bench-bwd
    python ads_mkl/ops/cute_dsl/gdpa/scripts/benchmark.py bench-fwd-bwd

Benchmark shapes (head_dim=128, num_heads=4, total_dim=512):
  Set 1 (equal QKV length): B=1152, H=4, d=128, Q=KV=[256..2048], sparsity 0.5 & 1.0
  Set 2 (fixed KV length):  B=1152, H=4, d=128, KV=256, Q=[256..2048], sparsity 1.0

Requirements:
    - NVIDIA Blackwell GPU (B200/B100/GB200)
    - CUDA 12.8+
    - PyTorch 2.10+ with triton
    - fbgemm-gpu-genai >= 1.5.0  (for CUTLASS FMHA)
    - facebookexperimental/triton (for Triton GDPA, provides tl.async_task)
    - flash-attn >= 2.8.0       (for CuteDSL FA4, https://github.com/Dao-AILab/flash-attention)

Kernels are enabled conditionally based on available packages. The benchmark
runs with whatever subset of kernels is importable.
"""

import os
import sys

import click
import torch
import triton

BASE_MODULE = "ads_mkl"
current_path = os.path.dirname(os.path.abspath(__file__))
relative_prefix = current_path[: current_path.find(BASE_MODULE)]
if relative_prefix not in sys.path:
    sys.path.insert(0, relative_prefix)

# Internal kernel imports
from ads_mkl.ops.cute_dsl.gdpa.src.interface import (
    flash_attn_varlen_func as cutedsl_gdpa_varlen_func,
    precompute_for_varline_load_balancing,
)
from ads_mkl.ops.cute_dsl.gdpa.utils.utils import generate_jagged_data

HAS_TRITON_GDPA = False
HAS_CUTLASS_FMHA = False
HAS_FA4 = False

try:
    from ads_mkl.ops.cute_dsl.gdpa.triton.triton_generalized_dot_product_attention import (
        generalized_dot_product_attention as triton_gdpa_func,
    )

    # Triton GDPA requires tl.async_task (Meta's internal Triton fork).
    # Check at import time to avoid verbose compilation errors at runtime.
    if not hasattr(triton.language, "async_task"):
        print(
            "Warning: Triton GDPA requires tl.async_task "
            "(Meta's internal Triton fork). Skipping."
        )
    else:
        HAS_TRITON_GDPA = True
except Exception as e:
    print(f"Warning: Triton GDPA not available: {e}")

# OSS kernel imports (optional)
try:
    # Pre-load cuBLAS libraries needed by fbgemm_gpu_experimental_gen_ai.so.
    # The .so links against these but doesn't bundle them; we need to load
    # the system libraries before importing.
    # NOTE: Do NOT preload system NCCL here — PyPI torch bundles its own
    # nvidia-nccl-cu12 and the system version may be too old (missing symbols
    # like ncclMemFree). Let torch use its bundled NCCL.
    import ctypes
    import glob as _glob

    for _lib_name in ["libcublasLt.so.*[0-9]", "libcublas.so.*[0-9]"]:
        for _cuda_dir in [
            "/usr/local/cuda-12.9/lib64",
            "/usr/local/cuda-12.8/lib64",
            "/usr/local/cuda/lib64",
            "/usr/local/fbcode/platform010/lib",
        ]:
            _matches = sorted(_glob.glob(os.path.join(_cuda_dir, _lib_name)))
            if _matches:
                ctypes.CDLL(_matches[0], mode=ctypes.RTLD_GLOBAL)
                break

    from fbgemm_gpu.experimental.gen_ai.attention.cutlass_blackwell_fmha import (
        cutlass_blackwell_fmha_func,
    )

    HAS_CUTLASS_FMHA = True
except Exception as e:
    print(
        "Warning: CUTLASS Blackwell FMHA not available. "
        f"Install fbgemm-gpu-genai to enable. ({e.__class__.__name__}: {e})"
    )

try:
    from flash_attn.cute import flash_attn_varlen_func as fa4_varlen_func

    HAS_FA4 = True
except ImportError as e:
    # flash_attn.__init__.py tries to import the compiled CUDA extension
    # (flash_attn_2_cuda) at top level, which fails when installed with
    # FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE. Bypass the top-level init
    # and import the pure-Python CuTe DSL submodule directly.
    if "flash_attn_2_cuda" in str(e):
        import subprocess
        import types

        # Clear partially-loaded flash_attn modules
        for _key in list(sys.modules.keys()):
            if _key.startswith("flash_attn"):
                del sys.modules[_key]
        # Locate the package directory via pip
        _result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "flash-attn"],
            capture_output=True,
            text=True,
        )
        _location = None
        for _line in _result.stdout.split("\n"):
            if _line.startswith("Location:"):
                _location = _line.split(":", 1)[1].strip()
                break
        if _location:
            _fa_path = os.path.join(_location, "flash_attn")
            _dummy = types.ModuleType("flash_attn")
            _dummy.__path__ = [_fa_path]
            _dummy.__package__ = "flash_attn"
            sys.modules["flash_attn"] = _dummy
            try:
                from flash_attn.cute import flash_attn_varlen_func as fa4_varlen_func

                HAS_FA4 = True
            except Exception as e2:
                print(
                    f"Warning: CuteDSL FA4 not available ({e2}). "
                    "Try: FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE "
                    "pip install git+https://github.com/Dao-AILab/flash-attention.git "
                    "--no-build-isolation"
                )
        else:
            print(f"Warning: CuteDSL FA4 not available ({e})")
    else:
        print(
            "Warning: CuteDSL FA4 not available. "
            "Install: FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE "
            "pip install git+https://github.com/Dao-AILab/flash-attention.git "
            "--no-build-isolation "
            f"({e})"
        )


@click.group()
def cli() -> None:
    torch.manual_seed(2025)


# Default hyper-parameters
B = 1152
H = 4
# Per-head dimension (total hidden dim = H * DIM = 4 * 128 = 512)
DIM = 128
DTYPE = torch.bfloat16
ACTIVATION = "fast_gelu"
SEQ_LENGTHS = [256, 512, 1024, 2048]
# Benchmark timing: warmup and repetition durations in ms
WARMUP_MS = 100
REP_MS = 2000


def get_provider_lines():
    """Build provider lists based on available kernels."""
    line_vals = ["cutedsl_gdpa"]
    line_names = ["CuteDSL_GDPA"]
    styles = [("blue", "-")]

    if HAS_TRITON_GDPA:
        line_vals.append("triton_gdpa")
        line_names.append("Triton_GDPA")
        styles.append(("green", "-"))

    if HAS_CUTLASS_FMHA:
        line_vals.append("cutlass_fmha")
        line_names.append("CUTLASS_FMHA")
        styles.append(("red", "-"))

    if HAS_FA4:
        line_vals.append("cutedsl_fa4")
        line_names.append("CuteDSL_FA4")
        styles.append(("orange", "-"))

    return line_vals, line_names, styles


def compute_flops(
    q_offsets: torch.Tensor,
    kv_offsets: torch.Tensor,
    nheads: int,
    headdim: int,
) -> int:
    """Compute forward FLOPS for jagged attention (QK + PV matmuls)."""
    q_lengths = (q_offsets[1:] - q_offsets[:-1]).float()
    kv_lengths = (kv_offsets[1:] - kv_offsets[:-1]).float()
    # 2 * q * kv * d for QK matmul + 2 * q * kv * d for PV matmul = 4 * q * kv * d
    return int(nheads * 4 * headdim * (q_lengths * kv_lengths).sum().item())


# ---------------------------------------------------------------------------
# Forward-only kernel dispatch
# ---------------------------------------------------------------------------
def run_fwd(
    provider,
    jagged_q,
    jagged_k,
    jagged_v,
    q_offsets,
    kv_offsets,
    max_len,
    dff,
    activation,
    quantiles,
):
    """Run forward pass and return (ms, min_ms, max_ms)."""
    if provider == "cutedsl_gdpa":
        # Precompute tile mappings for persistent scheduling
        tile_to_batch_q, tile_to_head_q, tile_to_block_q = (
            precompute_for_varline_load_balancing(jagged_q, q_offsets, kv_offsets)
        )

        return triton.testing.do_bench(
            lambda: cutedsl_gdpa_varlen_func(
                q=jagged_q,
                k=jagged_k,
                v=jagged_v,
                cu_seqlens_q=q_offsets,
                cu_seqlens_k=kv_offsets,
                max_seqlen_q=max_len,
                max_seqlen_k=dff,
                prefer_persistent=True,
                tile_to_batch_q=tile_to_batch_q,
                tile_to_head_q=tile_to_head_q,
                tile_to_block_q=tile_to_block_q,
                activation=activation,
            ),
            quantiles=quantiles,
            warmup=WARMUP_MS,
            rep=REP_MS,
        )
    elif provider == "triton_gdpa":
        return triton.testing.do_bench(
            lambda: triton_gdpa_func(
                query=jagged_q,
                key=jagged_k,
                value=jagged_v,
                query_offset=q_offsets,
                key_offset=kv_offsets,
                output_offset=q_offsets,
                max_seq_len_q=max_len,
                max_seq_len_kv=dff,
                activation=activation,
                qk_scale=1,
            ),
            quantiles=quantiles,
            warmup=WARMUP_MS,
            rep=REP_MS,
        )
    elif provider == "cutlass_fmha":
        return triton.testing.do_bench(
            lambda: cutlass_blackwell_fmha_func(
                q=jagged_q,
                k=jagged_k,
                v=jagged_v,
                cu_seqlens_q=q_offsets,
                cu_seqlens_k=kv_offsets,
                max_seq_len_q=max_len,
                max_seq_len_k=dff,
            ),
            quantiles=quantiles,
            warmup=WARMUP_MS,
            rep=REP_MS,
        )
    elif provider == "cutedsl_fa4":
        return triton.testing.do_bench(
            lambda: fa4_varlen_func(
                q=jagged_q,
                k=jagged_k,
                v=jagged_v,
                cu_seqlens_q=q_offsets,
                cu_seqlens_k=kv_offsets,
            ),
            quantiles=quantiles,
            warmup=WARMUP_MS,
            rep=REP_MS,
        )


def _extract_output(result):
    """Extract the output tensor from a kernel result (may be a tuple)."""
    if isinstance(result, tuple):
        return result[0]
    return result


def run_bwd(
    provider,
    jagged_q,
    jagged_k,
    jagged_v,
    q_offsets,
    kv_offsets,
    max_len,
    dff,
    activation,
    quantiles,
):
    """Run forward + backward pass, timing only the backward."""
    # Forward pass to get output
    if provider == "cutedsl_gdpa":
        # Precompute tile mappings for persistent scheduling
        tile_to_batch_q, tile_to_head_q, tile_to_block_q = (
            precompute_for_varline_load_balancing(jagged_q, q_offsets, kv_offsets)
        )
        tile_to_batch_k, tile_to_head_k, tile_to_block_k = (
            precompute_for_varline_load_balancing(
                jagged_q, q_offsets, kv_offsets, is_bwd=True
            )
        )

        # Forward with persistent scheduling via autograd
        result = cutedsl_gdpa_varlen_func(
            q=jagged_q,
            k=jagged_k,
            v=jagged_v,
            cu_seqlens_q=q_offsets,
            cu_seqlens_k=kv_offsets,
            max_seqlen_q=max_len,
            max_seqlen_k=dff,
            prefer_persistent=True,
            tile_to_batch_q=tile_to_batch_q,
            tile_to_head_q=tile_to_head_q,
            tile_to_block_q=tile_to_block_q,
            tile_to_batch_k=tile_to_batch_k,
            tile_to_head_k=tile_to_head_k,
            tile_to_block_k=tile_to_block_k,
            activation=activation,
        )
        out = _extract_output(result)

        dout = torch.randn_like(out)

        # Warmup: backward via autograd (triggers persistent backward compilation)
        out.backward(dout, retain_graph=True)
        torch.cuda.synchronize()

        jagged_q.grad = None
        jagged_k.grad = None
        jagged_v.grad = None

        def bwd():
            if jagged_q.grad is not None:
                jagged_q.grad = None
                jagged_k.grad = None
                jagged_v.grad = None
            out.backward(dout, retain_graph=True)

        return triton.testing.do_bench(
            bwd, quantiles=quantiles, warmup=WARMUP_MS, rep=REP_MS
        )
    elif provider == "triton_gdpa":
        out = _extract_output(
            triton_gdpa_func(
                query=jagged_q,
                key=jagged_k,
                value=jagged_v,
                query_offset=q_offsets,
                key_offset=kv_offsets,
                output_offset=q_offsets,
                max_seq_len_q=max_len,
                max_seq_len_kv=dff,
                activation=activation,
                qk_scale=1,
            )
        )
    elif provider == "cutlass_fmha":
        out = _extract_output(
            cutlass_blackwell_fmha_func(
                q=jagged_q,
                k=jagged_k,
                v=jagged_v,
                cu_seqlens_q=q_offsets,
                cu_seqlens_k=kv_offsets,
                max_seq_len_q=max_len,
                max_seq_len_k=dff,
            )
        )
    elif provider == "cutedsl_fa4":
        out = _extract_output(
            fa4_varlen_func(
                q=jagged_q,
                k=jagged_k,
                v=jagged_v,
                cu_seqlens_q=q_offsets,
                cu_seqlens_k=kv_offsets,
            )
        )

    dout = torch.randn_like(out)

    def bwd():
        if jagged_q.grad is not None:
            jagged_q.grad = None
            jagged_k.grad = None
            jagged_v.grad = None
        out.backward(dout, retain_graph=True)

    return triton.testing.do_bench(
        bwd, quantiles=quantiles, warmup=WARMUP_MS, rep=REP_MS
    )


def run_fwd_bwd(
    provider,
    jagged_q,
    jagged_k,
    jagged_v,
    q_offsets,
    kv_offsets,
    max_len,
    dff,
    activation,
    quantiles,
):
    """Time the combined forward + backward pass."""
    dout = torch.randn(
        int(q_offsets[-1].item()),
        jagged_q.shape[1],
        jagged_q.shape[2],
        dtype=jagged_q.dtype,
        device=jagged_q.device,
    )

    # Precompute tile mappings for persistent scheduling (cutedsl_gdpa only)
    if provider == "cutedsl_gdpa":
        tile_to_batch_q, tile_to_head_q, tile_to_block_q = (
            precompute_for_varline_load_balancing(jagged_q, q_offsets, kv_offsets)
        )
        tile_to_batch_k, tile_to_head_k, tile_to_block_k = (
            precompute_for_varline_load_balancing(
                jagged_q, q_offsets, kv_offsets, is_bwd=True
            )
        )

    def fwd_bwd():
        if jagged_q.grad is not None:
            jagged_q.grad = None
            jagged_k.grad = None
            jagged_v.grad = None
        if provider == "cutedsl_gdpa":
            out = _extract_output(
                cutedsl_gdpa_varlen_func(
                    q=jagged_q,
                    k=jagged_k,
                    v=jagged_v,
                    cu_seqlens_q=q_offsets,
                    cu_seqlens_k=kv_offsets,
                    max_seqlen_q=max_len,
                    max_seqlen_k=dff,
                    prefer_persistent=True,
                    tile_to_batch_q=tile_to_batch_q,
                    tile_to_head_q=tile_to_head_q,
                    tile_to_block_q=tile_to_block_q,
                    tile_to_batch_k=tile_to_batch_k,
                    tile_to_head_k=tile_to_head_k,
                    tile_to_block_k=tile_to_block_k,
                    activation=activation,
                )
            )
        elif provider == "triton_gdpa":
            out = _extract_output(
                triton_gdpa_func(
                    query=jagged_q,
                    key=jagged_k,
                    value=jagged_v,
                    query_offset=q_offsets,
                    key_offset=kv_offsets,
                    output_offset=q_offsets,
                    max_seq_len_q=max_len,
                    max_seq_len_kv=dff,
                    activation=activation,
                    qk_scale=1,
                )
            )
        elif provider == "cutlass_fmha":
            out = _extract_output(
                cutlass_blackwell_fmha_func(
                    q=jagged_q,
                    k=jagged_k,
                    v=jagged_v,
                    cu_seqlens_q=q_offsets,
                    cu_seqlens_k=kv_offsets,
                    max_seq_len_q=max_len,
                    max_seq_len_k=dff,
                )
            )
        elif provider == "cutedsl_fa4":
            out = _extract_output(
                fa4_varlen_func(
                    q=jagged_q,
                    k=jagged_k,
                    v=jagged_v,
                    cu_seqlens_q=q_offsets,
                    cu_seqlens_k=kv_offsets,
                )
            )
        out.backward(dout, retain_graph=False)

    return triton.testing.do_bench(
        fwd_bwd, quantiles=quantiles, warmup=WARMUP_MS, rep=REP_MS
    )


# ---------------------------------------------------------------------------
# Set 1: Equal QKV length, sparsity 0.5 and 1.0
# ---------------------------------------------------------------------------
def build_set1_configs(mode):
    line_vals, line_names, styles = get_provider_lines()
    configs = []
    for sparsity in [0.5, 1.0]:
        configs.append(
            triton.testing.Benchmark(
                x_names=["seq_length"],
                x_vals=SEQ_LENGTHS,
                line_arg="provider",
                line_vals=line_vals,
                line_names=line_names,
                styles=styles,
                ylabel="TFLOPS",
                plot_name=f"set1-{mode}-B{B}-H{H}-d{DIM}-QeqKV-sparsity{sparsity}",
                args={
                    "B": B,
                    "H": H,
                    "dim": DIM,
                    "sparsity": sparsity,
                    "activation": ACTIVATION,
                    "dtype": DTYPE,
                    "fixed_kv": 0,
                    "mode": mode,
                },
            )
        )
    return configs


# ---------------------------------------------------------------------------
# Set 2: Fixed KV=256, varying Q length, sparsity 1.0
# ---------------------------------------------------------------------------
def build_set2_configs(mode):
    line_vals, line_names, styles = get_provider_lines()
    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=["seq_length"],
            x_vals=SEQ_LENGTHS,
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_names,
            styles=styles,
            ylabel="TFLOPS",
            plot_name=f"set2-{mode}-B{B}-H{H}-d{DIM}-KV256-sparsity1.0",
            args={
                "B": B,
                "H": H,
                "dim": DIM,
                "sparsity": 1.0,
                "activation": ACTIVATION,
                "dtype": DTYPE,
                "fixed_kv": 256,
                "mode": mode,
            },
        )
    )
    return configs


def build_all_configs(mode):
    return build_set1_configs(mode) + build_set2_configs(mode)


@triton.testing.perf_report(build_all_configs("fwd"))
def benchmark_fwd(
    seq_length, provider, B, H, dim, sparsity, activation, dtype, fixed_kv, mode
):
    return _run_benchmark(
        seq_length, provider, B, H, dim, sparsity, activation, dtype, fixed_kv, "fwd"
    )


@triton.testing.perf_report(build_all_configs("bwd"))
def benchmark_bwd(
    seq_length, provider, B, H, dim, sparsity, activation, dtype, fixed_kv, mode
):
    return _run_benchmark(
        seq_length, provider, B, H, dim, sparsity, activation, dtype, fixed_kv, "bwd"
    )


@triton.testing.perf_report(build_all_configs("fwd_bwd"))
def benchmark_fwd_bwd(
    seq_length, provider, B, H, dim, sparsity, activation, dtype, fixed_kv, mode
):
    return _run_benchmark(
        seq_length,
        provider,
        B,
        H,
        dim,
        sparsity,
        activation,
        dtype,
        fixed_kv,
        "fwd_bwd",
    )


def _run_benchmark(
    seq_length, provider, B, H, dim, sparsity, activation, dtype, fixed_kv, mode
):
    # dff = KV length per batch element
    # fixed_kv > 0 means KV is fixed at that value (Set 2)
    # fixed_kv == 0 means KV = Q length (Set 1)
    dff = fixed_kv if fixed_kv > 0 else seq_length
    requires_grad = mode in ("bwd", "fwd_bwd")

    jagged_q, jagged_k, jagged_v, q_offsets, kv_offsets, lengths, max_len = (
        generate_jagged_data(
            max_len=seq_length,
            B=B,
            H=H,
            dim=dim,
            dff=dff,
            sparsity=sparsity,
            dtype=dtype,
            requires_grad=requires_grad,
        )
    )
    # Reset default device since generate_jagged_data sets it to "cuda"
    # which can cause issues with kernels that do CPU-side computation.
    torch.set_default_device(None)
    quantiles = [0.5, 0.2, 0.8]

    # Use fwd flops for both fwd and bwd reporting (bwd is ~2.5x fwd flops
    # but we report fwd-equivalent TFLOPS for consistency)
    total_flops = compute_flops(q_offsets, kv_offsets, H, dim)
    if mode == "bwd":
        total_flops = int(total_flops * 2.5)
    elif mode == "fwd_bwd":
        total_flops = int(total_flops * 3.5)

    try:
        if mode == "fwd":
            ms, min_ms, max_ms = run_fwd(
                provider,
                jagged_q,
                jagged_k,
                jagged_v,
                q_offsets,
                kv_offsets,
                max_len,
                dff,
                activation,
                quantiles,
            )
        elif mode == "bwd":
            ms, min_ms, max_ms = run_bwd(
                provider,
                jagged_q,
                jagged_k,
                jagged_v,
                q_offsets,
                kv_offsets,
                max_len,
                dff,
                activation,
                quantiles,
            )
        elif mode == "fwd_bwd":
            ms, min_ms, max_ms = run_fwd_bwd(
                provider,
                jagged_q,
                jagged_k,
                jagged_v,
                q_offsets,
                kv_offsets,
                max_len,
                dff,
                activation,
                quantiles,
            )
    except Exception as e:
        err_msg = str(e).split("\n")[0][:200]
        print(
            f"Error running {provider} ({mode}) at seq_length={seq_length}: {err_msg}"
        )
        return float("nan"), float("nan"), float("nan")

    def perf(ms):
        return total_flops * 1e-12 / (ms * 1e-3)

    return perf(ms), perf(max_ms), perf(min_ms)


@cli.command()
def bench_fwd() -> None:
    """Run forward pass benchmark."""
    benchmark_fwd.run(print_data=True)


@cli.command()
def bench_bwd() -> None:
    """Run backward pass benchmark."""
    benchmark_bwd.run(print_data=True)


@cli.command()
def bench_fwd_bwd() -> None:
    """Run combined forward + backward benchmark."""
    benchmark_fwd_bwd.run(print_data=True)


@cli.command()
def bench_all() -> None:
    """Run forward, backward, and combined benchmarks."""
    print("=" * 60)
    print("FORWARD PASS")
    print("=" * 60)
    benchmark_fwd.run(print_data=True)
    print()
    print("=" * 60)
    print("BACKWARD PASS")
    print("=" * 60)
    benchmark_bwd.run(print_data=True)
    print()
    print("=" * 60)
    print("FORWARD + BACKWARD")
    print("=" * 60)
    benchmark_fwd_bwd.run(print_data=True)


if __name__ == "__main__":
    cli()
