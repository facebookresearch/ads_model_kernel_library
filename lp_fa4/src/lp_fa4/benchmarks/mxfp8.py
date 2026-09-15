# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from collections.abc import Callable, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from lp_fa4 import __version__ as lp_fa4_version
from lp_fa4.cute import (
    mxfp8_flash_attn_varlen_backward,
    mxfp8_flash_attn_varlen_forward,
    Mxfp8VarlenMeta,
    Mxfp8VarlenTensor,
    quantize_mxfp8_varlen,
)


def _parse_lengths(raw: str | None, count: int, default: int) -> tuple[int, ...]:
    if raw is None:
        return (default,) * count
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("length lists must contain positive integers")
    return values


def _measure(
    fn: Callable[[], object],
    *,
    warmup_ms: float,
    measure_ms: float,
    launches_per_sample: int,
) -> list[float]:
    def sample() -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(launches_per_sample):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / launches_per_sample

    fn()
    warmup_elapsed = 0.0
    while warmup_elapsed < warmup_ms:
        warmup_elapsed += sample() * launches_per_sample
    samples: list[float] = []
    measured_elapsed = 0.0
    while measured_elapsed < measure_ms:
        latency = sample()
        samples.append(latency)
        measured_elapsed += latency * launches_per_sample
    return samples


def _summary(
    samples_ms: Sequence[float],
    flop: int,
) -> dict[str, float | int | list[float]]:
    mean_ms = statistics.fmean(samples_ms)
    return {
        "samples": len(samples_ms),
        "samples_ms": list(samples_ms),
        "mean_ms": mean_ms,
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "pstdev_ms": statistics.pstdev(samples_ms),
        "tflops": flop / (mean_ms * 1.0e9),
    }


def _attention_work(
    q_lengths: Sequence[int],
    k_lengths: Sequence[int],
    *,
    broadcast_q: bool,
) -> int:
    if broadcast_q:
        return q_lengths[0] * sum(k_lengths)
    return sum(q_length * k_length for q_length, k_length in zip(q_lengths, k_lengths))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark LP-FA4 MXFP8 kernels")
    parser.add_argument("--mode", choices=("fwd", "bwd", "both"), default="both")
    parser.add_argument("--batch-size", type=int, required=True)
    q_shape = parser.add_mutually_exclusive_group(required=True)
    q_shape.add_argument("--seqlen-q", type=int)
    q_shape.add_argument("--q-lengths")
    k_shape = parser.add_mutually_exclusive_group(required=True)
    k_shape.add_argument("--seqlen-k", type=int)
    k_shape.add_argument("--k-lengths")
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--broadcast-q", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-ms", type=float, default=1000.0)
    parser.add_argument("--measure-ms", type=float, default=5000.0)
    parser.add_argument("--launches-per-sample", type=int, default=5)
    parser.add_argument("--scaling-mode", choices=("rceil", "floor"), default="rceil")
    parser.add_argument("--dq-accum-dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument(
        "--p-scale-mode", choices=("dynamic", "const_p"), default="const_p"
    )
    parser.add_argument("--output", type=Path)
    return parser


def _resolve_lengths(
    args: argparse.Namespace,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    k_lengths = _parse_lengths(args.k_lengths, args.batch_size, args.seqlen_k)
    q_count = 1 if args.broadcast_q else len(k_lengths)
    q_lengths = _parse_lengths(args.q_lengths, q_count, args.seqlen_q)
    if len(k_lengths) != args.batch_size:
        raise ValueError("K length count must equal --batch-size")
    if len(q_lengths) != q_count:
        raise ValueError("Q length count does not match the selected attention mode")
    return q_lengths, k_lengths


def _environment() -> dict[str, Any]:
    major, minor = torch.cuda.get_device_capability()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "lp_fa4": _package_version("lp-fa4", fallback=lp_fa4_version),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cutlass_dsl": _package_version("nvidia-cutlass-dsl"),
        "quack_kernels": _package_version("quack-kernels"),
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": f"{major}.{minor}",
    }


def _package_version(distribution: str, *, fallback: str = "unknown") -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return fallback


def _run(args: argparse.Namespace) -> dict[str, Any]:
    q_lengths, k_lengths = _resolve_lengths(args)
    torch.manual_seed(args.seed)
    shape_q = (sum(q_lengths), args.num_heads, 128)
    shape_k = (sum(k_lengths), args.num_heads, 128)
    q = torch.randn(shape_q, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(shape_k, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(shape_k, dtype=torch.bfloat16, device="cuda")
    dout_tokens = sum(q_lengths) * (args.batch_size if args.broadcast_q else 1)
    dout = torch.randn(
        (dout_tokens, args.num_heads, 128),
        dtype=torch.bfloat16,
        device="cuda",
    )
    q_meta = Mxfp8VarlenMeta.from_lengths(q_lengths, device="cuda")
    k_meta = Mxfp8VarlenMeta.from_lengths(k_lengths, device="cuda")
    orientations = (
        ("hdim", "hdim", "seq") if args.mode == "fwd" else ("both", "both", "both")
    )
    q_mx, k_mx, v_mx = (
        quantize_mxfp8_varlen(
            tensor,
            meta,
            orientations=orientation,
            scaling_mode=args.scaling_mode,
        )
        for tensor, meta, orientation in zip(
            (q, k, v),
            (q_meta, k_meta, k_meta),
            orientations,
        )
    )
    return _run_modes(args, q_mx, k_mx, v_mx, q_meta, k_meta, dout)


def _run_modes(
    args: argparse.Namespace,
    q: Mxfp8VarlenTensor,
    k: Mxfp8VarlenTensor,
    v: Mxfp8VarlenTensor,
    q_meta: Mxfp8VarlenMeta,
    k_meta: Mxfp8VarlenMeta,
    dout: torch.Tensor,
) -> dict[str, Any]:
    work = _attention_work(
        _lengths_from_meta(q_meta),
        _lengths_from_meta(k_meta),
        broadcast_q=args.broadcast_q,
    )
    common = {"broadcast_q": args.broadcast_q}
    out, lse = mxfp8_flash_attn_varlen_forward(
        q, k, v, q_meta, k_meta, return_lse=True, **common
    )
    if lse is None:
        raise RuntimeError("backward setup requires LSE")
    results: dict[str, Any] = {}
    if args.mode in ("fwd", "both"):

        def run_fwd() -> object:
            return mxfp8_flash_attn_varlen_forward(
                q, k, v, q_meta, k_meta, return_lse=False, **common
            )

        samples = _measure_fn(run_fwd, args)
        results["fwd"] = _summary(samples, 4 * args.num_heads * 128 * work)
    if args.mode in ("bwd", "both"):
        dq_dtype = torch.float16 if args.dq_accum_dtype == "fp16" else torch.float32

        def run_bwd() -> object:
            return mxfp8_flash_attn_varlen_backward(
                q,
                k,
                v,
                out,
                dout,
                lse,
                q_meta,
                k_meta,
                dq_accum_dtype=dq_dtype,
                p_scale_mode=args.p_scale_mode,
                output_mxfp8_dkv=True,
                **common,
            )

        samples = _measure_fn(run_bwd, args)
        results["bwd"] = _summary(samples, 10 * args.num_heads * 128 * work)
    return _record(args, q_meta, k_meta, results)


def _measure_fn(fn: Callable[[], object], args: argparse.Namespace) -> list[float]:
    return _measure(
        fn,
        warmup_ms=args.warmup_ms,
        measure_ms=args.measure_ms,
        launches_per_sample=args.launches_per_sample,
    )


def _lengths_from_meta(meta: Mxfp8VarlenMeta) -> tuple[int, ...]:
    offsets = meta.cpu_cu_seqlens.tolist()
    return tuple(end - start for start, end in zip(offsets, offsets[1:]))


def _record(
    args: argparse.Namespace,
    q_meta: Mxfp8VarlenMeta,
    k_meta: Mxfp8VarlenMeta,
    results: dict[str, Any],
) -> dict[str, Any]:
    return {
        "argv": sys.argv[1:],
        "environment": _environment(),
        "seed": args.seed,
        "mode": args.mode,
        "num_heads": args.num_heads,
        "head_dim": 128,
        "broadcast_q": args.broadcast_q,
        "q_lengths": _lengths_from_meta(q_meta),
        "k_lengths": _lengths_from_meta(k_meta),
        "cu_seqlens_q": q_meta.cu_seqlens.tolist(),
        "cu_seqlens_k": k_meta.cu_seqlens.tolist(),
        "cu_seqlens_sf_q": q_meta.cu_seqlens_sf.tolist(),
        "cu_seqlens_sf_k": k_meta.cu_seqlens_sf.tolist(),
        "scaling_mode": args.scaling_mode,
        "dq_accum_dtype": args.dq_accum_dtype,
        "p_scale_mode": args.p_scale_mode,
        "bwd_output_mxfp8_dkv": args.mode in ("bwd", "both"),
        "warmup_ms": args.warmup_ms,
        "measure_ms": args.measure_ms,
        "launches_per_sample": args.launches_per_sample,
        "results": results,
    }


def main() -> None:
    args = _build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        raise RuntimeError(f"MXFP8 requires SM10x; found {major}.{minor}")
    if min(args.warmup_ms, args.measure_ms, args.launches_per_sample) <= 0:
        raise ValueError("timing windows and launches per sample must be positive")
    record = _run(args)
    rendered = json.dumps(record, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(f"{rendered}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
