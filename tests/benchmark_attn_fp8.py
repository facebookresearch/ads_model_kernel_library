# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# @nolint
# pyre-unsafe

"""Minimal FP8 E4M3 benchmark for GDPA (varlen mode only).

Usage:
    python benchmark_attn_fp8.py           # Run forward benchmark (default)
    python benchmark_attn_fp8.py backward  # Run backward benchmark
"""

import argparse
import time

import torch
from ads_mkl.ops.cute_dsl.gdpa.src.interface import (
    cutedsl_generalized_dot_product_attention,
    flash_attn_varlen_func,
)
from torch.profiler import profile
from triton.testing import do_bench


def flops(
    batch, nheads, seqlen_q, seqlen_k, headdim, headdim_v=None, causal=False, mode="fwd"
):
    """
    Calculate FLOPs for attention.

    Forward pass: 2 * batch * nheads * seqlen_q * seqlen_k * (headdim + headdim_v)
    - QK^T matmul: batch * nheads * seqlen_q * seqlen_k * headdim * 2 (mul + add)
    - softmax(QK^T)V matmul: batch * nheads * seqlen_q * seqlen_k * headdim_v * 2

    Backward pass is approximately 2.5x forward pass FLOPs.
    """
    if headdim_v is None:
        headdim_v = headdim

    assert mode in ["fwd", "bwd", "fwd_bwd"]

    if causal:
        avg_seqlen = (max(0, seqlen_k - seqlen_q) + seqlen_k) / 2
    else:
        avg_seqlen = seqlen_k

    f = batch * nheads * 2 * seqlen_q * avg_seqlen * (headdim + headdim_v)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


def compute_error_metrics(pred: torch.Tensor, ref: torch.Tensor) -> dict:
    """Compute error metrics between prediction and reference."""
    pred_f32 = pred.float()
    ref_f32 = ref.float()
    abs_err = torch.abs(pred_f32 - ref_f32)

    atol, rtol = 0.2, 0.1
    tolerance = atol + rtol * torch.abs(ref_f32)
    is_acceptable = abs_err <= tolerance
    mismatch = (~is_acceptable).sum().item()
    total = pred.numel()

    return {
        "mean_abs_err": abs_err.mean().item(),
        "max_abs_err": abs_err.max().item(),
        "mismatch_pct": 100.0 * mismatch / total,
    }


def benchmark_fp8(activation: str = "fast_gelu") -> None:
    """Run FP8 E4M3 benchmark with accuracy verification."""
    device = "cuda"
    generate_traces = True  # Set to True to generate perfdoctor traces
    batch_size = 256
    seqlen_q = 3072
    seqlen_k = 3072
    nheads = 4
    headdim = 128

    print(
        f"Config: {batch_size=}, {seqlen_q=}, {seqlen_k=}, {nheads=}, {headdim=}, {activation=}"
    )

    # Create cu_seqlens for varlen mode
    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * seqlen_q, seqlen_q, device=device, dtype=torch.int32
    )
    cu_seqlens_k = torch.arange(
        0, (batch_size + 1) * seqlen_k, seqlen_k, device=device, dtype=torch.int32
    )
    total_q = batch_size * seqlen_q
    total_k = batch_size * seqlen_k

    # Create FP32 reference tensors (small values to avoid FP8 overflow)
    q_fp32 = (
        torch.randn(total_q, nheads, headdim, device=device, dtype=torch.float32) * 0.2
    )
    k_fp32 = (
        torch.randn(total_k, nheads, headdim, device=device, dtype=torch.float32) * 0.2
    )
    v_fp32 = (
        torch.rand(total_k, nheads, headdim, device=device, dtype=torch.float32) * 0.2
        - 0.1
    )

    # Convert to FP8 and BF16
    q_fp8 = q_fp32.to(torch.float8_e4m3fn)
    k_fp8 = k_fp32.to(torch.float8_e4m3fn)
    v_fp8 = v_fp32.to(torch.float8_e4m3fn)

    q_bf16 = q_fp32.to(torch.bfloat16)
    k_bf16 = k_fp32.to(torch.bfloat16)
    v_bf16 = v_fp32.to(torch.bfloat16)

    qk_scale = 1.0 / (headdim**0.5)

    # Run FP8
    print("\nRunning FP8 E4M3...")
    result_fp8 = flash_attn_varlen_func(
        q_fp8,
        k_fp8,
        v_fp8,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        softmax_scale=qk_scale,
        activation=activation,
    )
    torch.cuda.synchronize()
    out_fp8 = (result_fp8[0] if isinstance(result_fp8, tuple) else result_fp8).to(
        torch.bfloat16
    )

    # Run BF16 reference
    print("Running BF16 reference...")
    result_bf16 = flash_attn_varlen_func(
        q_bf16,
        k_bf16,
        v_bf16,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        softmax_scale=qk_scale,
        activation=activation,
    )
    torch.cuda.synchronize()
    out_bf16 = result_bf16[0] if isinstance(result_bf16, tuple) else result_bf16

    # Validate
    has_nan_fp8 = torch.isnan(out_fp8).any()
    has_nan_bf16 = torch.isnan(out_bf16).any()

    if has_nan_fp8 or has_nan_bf16:
        print(f"\n❌ FAILED: FP8 has NaN: {has_nan_fp8}, BF16 has NaN: {has_nan_bf16}")
        return

    metrics = compute_error_metrics(out_fp8, out_bf16)
    print(f"\n{'=' * 80}")
    print("FP8 E4M3 vs BF16 Accuracy")
    print(f"{'=' * 80}")
    print(f"  Mean Absolute Error: {metrics['mean_abs_err']:.6f}")
    print(f"  Max Absolute Error: {metrics['max_abs_err']:.6f}")
    print(f"  Mismatch Rate: {metrics['mismatch_pct']:.2f}%")

    if metrics["mismatch_pct"] < 1.0:
        print(f"\n✅ PASSED (mismatch: {metrics['mismatch_pct']:.2f}%)")
    else:
        print(f"\n⚠️  WARNING: High mismatch rate ({metrics['mismatch_pct']:.2f}%)")

    # Benchmark
    print(f"\n{'=' * 80}")
    print("Performance Benchmark")
    print(f"{'=' * 80}")

    # Calculate FLOPs for forward pass
    nFLOPS_fwd = flops(
        batch_size,
        nheads,
        seqlen_q,
        seqlen_k,
        headdim,
        headdim,
        causal=False,
        mode="fwd",
    )

    time_fp8 = do_bench(
        lambda: flash_attn_varlen_func(
            q_fp8,
            k_fp8,
            v_fp8,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            activation=activation,
        ),
        warmup=200,
        rep=2000,
    )
    time_bf16 = do_bench(
        lambda: flash_attn_varlen_func(
            q_bf16,
            k_bf16,
            v_bf16,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            activation=activation,
        ),
        warmup=200,
        rep=2000,
    )

    # Convert time from ms to seconds for TFLOPS calculation
    time_fp8_s = time_fp8 * 1e-3
    time_bf16_s = time_bf16 * 1e-3
    tflops_fp8 = nFLOPS_fwd / time_fp8_s * 1e-12
    tflops_bf16 = nFLOPS_fwd / time_bf16_s * 1e-12

    print(f"  FP8:  {time_fp8:.3f} ms, {tflops_fp8:.1f} TFLOPS")
    print(f"  BF16: {time_bf16:.3f} ms, {tflops_bf16:.1f} TFLOPS")
    print(f"  Speedup: {time_bf16 / time_fp8:.2f}x")
    print(f"{'=' * 80}")

    # Generate perfdoctor traces for performance investigation
    if generate_traces:
        print(f"\n{'=' * 80}")
        print("Generating perfdoctor traces...")
        print(f"{'=' * 80}")

        # Warmup before profiling FP8
        for _ in range(3):
            flash_attn_varlen_func(
                q_fp8,
                k_fp8,
                v_fp8,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                activation=activation,
            )
        torch.cuda.synchronize()

        # Profile FP8 forward pass
        with profile() as p_fp8:
            for _ in range(10):
                flash_attn_varlen_func(
                    q_fp8,
                    k_fp8,
                    v_fp8,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    activation=activation,
                )

        mf_path_prefix = "manifold://pyper_traces/"
        timestamp = int(time.time())
        mf_path_fp8 = f"tree/efficient_module_suite/gdpa_fp8_fwd_{batch_size}x{seqlen_q}x{nheads}x{headdim}_{timestamp}.json"
        p_fp8.export_chrome_trace(mf_path_prefix + mf_path_fp8)
        trace_url_fp8 = f"https://www.internalfb.com/intern/perfdoctor/trace_view?filepath={mf_path_fp8}.gz&bucket=pyper_traces"

        # Warmup before profiling BF16
        for _ in range(3):
            flash_attn_varlen_func(
                q_bf16,
                k_bf16,
                v_bf16,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                activation=activation,
            )
        torch.cuda.synchronize()

        # Profile BF16 forward pass
        with profile() as p_bf16:
            for _ in range(10):
                flash_attn_varlen_func(
                    q_bf16,
                    k_bf16,
                    v_bf16,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    activation=activation,
                )

        mf_path_bf16 = f"tree/efficient_module_suite/gdpa_bf16_fwd_{batch_size}x{seqlen_q}x{nheads}x{headdim}_{timestamp}.json"
        p_bf16.export_chrome_trace(mf_path_prefix + mf_path_bf16)
        trace_url_bf16 = f"https://www.internalfb.com/intern/perfdoctor/trace_view?filepath={mf_path_bf16}.gz&bucket=pyper_traces"

        print(f"\nPerfdoctor Traces:")
        print(f"  FP8 Forward:  {trace_url_fp8}")
        print(f"  BF16 Forward: {trace_url_bf16}")
        print(f"{'=' * 80}")


def benchmark_fp8_backward(activation: str = "gelu"):
    """Run FP8 E4M3 backward pass benchmark with accuracy verification."""
    device = "cuda"
    generate_traces = True
    batch_size = 256
    seqlen_q = 3072
    seqlen_k = 3072
    nheads = 4
    headdim = 128

    print(f"\n{'=' * 80}")
    print("Backward Pass Benchmark (FP8 and BF16)")
    print(f"{'=' * 80}")
    print(
        f"Config: {batch_size=}, {seqlen_q=}, {seqlen_k=}, {nheads=}, {headdim=}, {activation=}"
    )

    # Create cu_seqlens for varlen mode
    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * seqlen_q, seqlen_q, device=device, dtype=torch.int32
    )
    cu_seqlens_k = torch.arange(
        0, (batch_size + 1) * seqlen_k, seqlen_k, device=device, dtype=torch.int32
    )
    total_q = batch_size * seqlen_q
    total_k = batch_size * seqlen_k

    # Create FP32 reference tensors (small values to avoid FP8 overflow)
    q_fp32 = (
        torch.randn(total_q, nheads, headdim, device=device, dtype=torch.float32) * 0.2
    )
    k_fp32 = (
        torch.randn(total_k, nheads, headdim, device=device, dtype=torch.float32) * 0.2
    )
    v_fp32 = (
        torch.rand(total_k, nheads, headdim, device=device, dtype=torch.float32) * 0.2
        - 0.1
    )

    qk_scale = 1.0 / (headdim**0.5)

    # Create gradient of output
    dout_fp32 = (
        torch.randn(total_q, nheads, headdim, device=device, dtype=torch.float32) * 0.1
    )
    dout_bf16 = dout_fp32.to(torch.bfloat16)
    dout_fp8 = dout_fp32.to(torch.float8_e4m3fn)

    # BF16 tensors for backward testing
    q_bf16 = q_fp32.to(torch.bfloat16).requires_grad_(True)
    k_bf16 = k_fp32.to(torch.bfloat16).requires_grad_(True)
    v_bf16 = v_fp32.to(torch.bfloat16).requires_grad_(True)

    # FP8 tensors for backward testing
    q_fp8 = q_fp32.to(torch.float8_e4m3fn).requires_grad_(True)
    k_fp8 = k_fp32.to(torch.float8_e4m3fn).requires_grad_(True)
    v_fp8 = v_fp32.to(torch.float8_e4m3fn).requires_grad_(True)

    # Run BF16 forward + backward
    print("\nRunning BF16 forward + backward...")
    out_bf16, lse_bf16 = cutedsl_generalized_dot_product_attention(
        q_bf16,
        k_bf16,
        v_bf16,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        softmax_scale=qk_scale,
        activation=activation,
    )
    torch.cuda.synchronize()

    dq_bf16, dk_bf16, dv_bf16 = torch.autograd.grad(
        out_bf16, (q_bf16, k_bf16, v_bf16), dout_bf16, retain_graph=True
    )
    torch.cuda.synchronize()
    print("BF16 backward pass completed successfully!")

    # Run FP8 forward + backward
    # Note: Use BF16 dout for FP8 backward since gradients are computed in BF16
    print("Running FP8 forward + backward...")
    out_fp8, lse_fp8 = cutedsl_generalized_dot_product_attention(
        q_fp8,
        k_fp8,
        v_fp8,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        softmax_scale=qk_scale,
        activation=activation,
    )
    torch.cuda.synchronize()

    # For FP8 inputs, use BF16 dout since the backward pass uses BF16 gradients
    dq_fp8, dk_fp8, dv_fp8 = torch.autograd.grad(
        out_fp8, (q_fp8, k_fp8, v_fp8), dout_bf16, retain_graph=True
    )
    torch.cuda.synchronize()

    # Validate and compare gradients
    print(f"\n{'=' * 80}")
    print("Gradient Validation (FP8 vs BF16)")
    print(f"{'=' * 80}")

    for name, fp8_grad, bf16_grad in [
        ("dQ", dq_fp8, dq_bf16),
        ("dK", dk_fp8, dk_bf16),
        ("dV", dv_fp8, dv_bf16),
    ]:
        # Convert to float for validation (FP8 doesn't support isnan/isinf)
        fp8_grad_f32 = fp8_grad.float()
        bf16_grad_f32 = bf16_grad.float()
        has_nan_fp8 = torch.isnan(fp8_grad_f32).any()
        has_nan_bf16 = torch.isnan(bf16_grad_f32).any()
        has_inf_fp8 = torch.isinf(fp8_grad_f32).any()
        has_inf_bf16 = torch.isinf(bf16_grad_f32).any()

        # Compare FP8 vs BF16 gradients
        metrics = compute_error_metrics(fp8_grad_f32, bf16_grad_f32)
        threshold = 15.0
        status = (
            "PASS"
            if not has_nan_fp8
            and not has_inf_fp8
            and metrics["mismatch_pct"] < threshold
            else "FAIL"
        )

        print(
            f"  {name}: nan={has_nan_fp8}, inf={has_inf_fp8} | "
            f"mean_err={metrics['mean_abs_err']:.6f}, max_err={metrics['max_abs_err']:.6f}, "
            f"mismatch={metrics['mismatch_pct']:.2f}% [{status}]"
        )

    print("FP8 backward completed without crash!")
    # return  # Skip benchmarking for now

    # Benchmark backward pass
    print(f"\n{'=' * 80}")
    print("Backward Pass Performance")
    print(f"{'=' * 80}")

    nFLOPS_bwd = flops(
        batch_size,
        nheads,
        seqlen_q,
        seqlen_k,
        headdim,
        headdim,
        causal=False,
        mode="bwd",
    )

    def bench_backward_bf16():
        q = q_fp32.to(torch.bfloat16).requires_grad_(True)
        k = k_fp32.to(torch.bfloat16).requires_grad_(True)
        v = v_fp32.to(torch.bfloat16).requires_grad_(True)
        out, _ = cutedsl_generalized_dot_product_attention(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            softmax_scale=qk_scale,
            activation=activation,
        )
        torch.autograd.grad(out, (q, k, v), dout_bf16)

    def bench_backward_fp8():
        q = q_fp32.to(torch.float8_e4m3fn).requires_grad_(True)
        k = k_fp32.to(torch.float8_e4m3fn).requires_grad_(True)
        v = v_fp32.to(torch.float8_e4m3fn).requires_grad_(True)
        out, _ = cutedsl_generalized_dot_product_attention(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            softmax_scale=qk_scale,
            activation=activation,
        )
        # Use BF16 dout for FP8 backward since gradients are computed in BF16
        torch.autograd.grad(out, (q, k, v), dout_bf16)

    time_bwd_bf16 = do_bench(bench_backward_bf16, warmup=50, rep=200)
    time_bwd_fp8 = do_bench(bench_backward_fp8, warmup=50, rep=200)

    time_bwd_bf16_s = time_bwd_bf16 * 1e-3
    time_bwd_fp8_s = time_bwd_fp8 * 1e-3
    tflops_bwd_bf16 = nFLOPS_bwd / time_bwd_bf16_s * 1e-12
    tflops_bwd_fp8 = nFLOPS_bwd / time_bwd_fp8_s * 1e-12

    print(f"  BF16 (fwd+bwd): {time_bwd_bf16:.3f} ms, {tflops_bwd_bf16:.1f} TFLOPS")
    print(f"  FP8 (fwd+bwd):  {time_bwd_fp8:.3f} ms, {tflops_bwd_fp8:.1f} TFLOPS")
    print(f"  Speedup: {time_bwd_bf16 / time_bwd_fp8:.2f}x")
    print(f"{'=' * 80}")

    # Generate traces for backward pass
    if generate_traces:
        print(f"\n{'=' * 80}")
        print("Generating backward pass traces...")
        print(f"{'=' * 80}")

        mf_path_prefix = "manifold://pyper_traces/"
        timestamp = int(time.time())

        # Warmup and profile BF16 backward
        for _ in range(3):
            bench_backward_bf16()
        torch.cuda.synchronize()

        mf_path_bwd_bf16 = f"tree/efficient_module_suite/gdpa_bf16_bwd_{batch_size}x{seqlen_q}x{nheads}x{headdim}_{timestamp}.json"
        with profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
            on_trace_ready=lambda p: p.export_chrome_trace(
                mf_path_prefix + mf_path_bwd_bf16
            ),
        ):
            for _ in range(10):
                bench_backward_bf16()
            torch.cuda.synchronize()

        trace_url_bwd_bf16 = f"https://www.internalfb.com/intern/perfdoctor/trace_view?filepath={mf_path_bwd_bf16}.gz&bucket=pyper_traces"

        # Warmup and profile FP8 backward
        for _ in range(3):
            bench_backward_fp8()
        torch.cuda.synchronize()

        mf_path_bwd_fp8 = f"tree/efficient_module_suite/gdpa_fp8_bwd_{batch_size}x{seqlen_q}x{nheads}x{headdim}_{timestamp}.json"
        with profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
            on_trace_ready=lambda p: p.export_chrome_trace(
                mf_path_prefix + mf_path_bwd_fp8
            ),
        ):
            for _ in range(10):
                bench_backward_fp8()
            torch.cuda.synchronize()

        trace_url_bwd_fp8 = f"https://www.internalfb.com/intern/perfdoctor/trace_view?filepath={mf_path_bwd_fp8}.gz&bucket=pyper_traces"

        print(f"\nPerfdoctor Traces (Backward):")
        print(f"  BF16 Backward: {trace_url_bwd_bf16}")
        print(f"  FP8 Backward:  {trace_url_bwd_fp8}")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FP8 E4M3 benchmark for GDPA (varlen mode only)."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="forward",
        choices=["forward", "backward"],
        help="Benchmark mode: 'forward' (default) or 'backward'",
    )
    parser.add_argument(
        "--activation",
        default="gelu",
        choices=["relu", "gelu"],
        help="Activation function: 'relu', or 'gelu'",
    )
    args = parser.parse_args()

    if args.mode == "forward":
        benchmark_fp8(activation=args.activation)
    else:
        benchmark_fp8_backward(activation=args.activation)
