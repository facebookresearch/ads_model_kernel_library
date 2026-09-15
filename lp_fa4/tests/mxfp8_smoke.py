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
import math

import torch
import torch.nn.functional as F
from lp_fa4.cute import (
    mxfp8_flash_attn_varlen_backward,
    mxfp8_flash_attn_varlen_forward,
    Mxfp8VarlenMeta,
    quantize_mxfp8_varlen,
)


def _quality(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual_fp32 = actual.float().flatten()
    expected_fp32 = expected.float().flatten()
    if not bool(torch.isfinite(actual_fp32).all()) or not bool(
        torch.isfinite(expected_fp32).all()
    ):
        raise RuntimeError("quality inputs must contain only finite values")
    cosine = F.cosine_similarity(actual_fp32, expected_fp32, dim=0).item()
    signal = torch.linalg.vector_norm(expected_fp32)
    noise = torch.linalg.vector_norm(actual_fp32 - expected_fp32)
    sqnr = 20.0 * math.log10((signal / noise).item())
    return cosine, sqnr


def _dequant_mxfp8_output(
    payload: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    expected_shape = (*payload.shape[:-1], payload.shape[-1] // 32)
    if tuple(scales.shape) != expected_shape:
        raise RuntimeError(
            f"MXFP8 output scales must have shape {expected_shape}, got {tuple(scales.shape)}"
        )
    expanded_scales = torch.exp2(scales.float() - 127.0).repeat_interleave(
        32,
        dim=-1,
    )
    return payload.float() * expanded_scales


def _check_quality(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    min_cosine: float,
    min_sqnr: float,
) -> None:
    cosine, sqnr = _quality(actual, expected)
    print(f"{name}: cosine={cosine:.6f}, SQNR={sqnr:.2f} dB")
    if (
        not math.isfinite(cosine)
        or not math.isfinite(sqnr)
        or cosine <= min_cosine
        or sqnr <= min_sqnr
    ):
        raise RuntimeError(f"{name} failed MXFP8 numerical thresholds")


def _reference_sequence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_ref, k_ref, v_ref = (
        tensor.float().detach().requires_grad_(True) for tensor in (q, k, v)
    )
    out_ref = (
        F.scaled_dot_product_attention(
            q_ref.transpose(0, 1).unsqueeze(0),
            k_ref.transpose(0, 1).unsqueeze(0),
            v_ref.transpose(0, 1).unsqueeze(0),
        )
        .squeeze(0)
        .transpose(0, 1)
    )
    out_ref.backward(dout.float())
    return out_ref, q_ref.grad, k_ref.grad, v_ref.grad


def _reference_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    lengths: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    start = 0
    for length in lengths:
        end = start + length
        outputs.append(
            _reference_sequence(
                q[start:end],
                k[start:end],
                v[start:end],
                dout[start:end],
            )
        )
        start = end
    return tuple(
        torch.cat([sequence[index] for sequence in outputs]) for index in range(4)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scaling-mode",
        choices=("rceil", "floor"),
        default="rceil",
    )
    parser.add_argument("--lengths", default="127,257")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        raise RuntimeError(
            f"MXFP8 LP-FA4 requires SM10x; found compute capability {major}.{minor}"
        )

    torch.manual_seed(0)
    lengths = tuple(int(value) for value in args.lengths.split(","))
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("--lengths must contain positive comma-separated integers")
    num_heads, head_dim = 16, 128
    shape = (sum(lengths), num_heads, head_dim)
    q, k, v = (
        torch.randn(shape, dtype=torch.bfloat16, device="cuda") for _ in range(3)
    )
    dout = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    metadata = Mxfp8VarlenMeta.from_lengths(lengths, device="cuda")
    q_mx, k_mx, v_mx = (
        quantize_mxfp8_varlen(
            tensor,
            metadata,
            scaling_mode=args.scaling_mode,
        )
        for tensor in (q, k, v)
    )

    out, lse = mxfp8_flash_attn_varlen_forward(
        q_mx,
        k_mx,
        v_mx,
        metadata,
        metadata,
    )
    if lse is None:
        raise RuntimeError("forward did not return LSE for backward")
    dq, dk_mx, dv_mx, sf_dk, sf_dv = mxfp8_flash_attn_varlen_backward(
        q_mx,
        k_mx,
        v_mx,
        out,
        dout,
        lse,
        metadata,
        metadata,
        output_mxfp8_dkv=True,
    )
    torch.cuda.synchronize()

    if dk_mx.dtype != torch.float8_e4m3fn or dv_mx.dtype != torch.float8_e4m3fn:
        raise RuntimeError("fused dK/dV payloads must use E4M3")
    if sf_dk.dtype != torch.uint8 or sf_dv.dtype != torch.uint8:
        raise RuntimeError("fused dK/dV scales must use E8M0 byte storage")
    dk = _dequant_mxfp8_output(dk_mx, sf_dk)
    dv = _dequant_mxfp8_output(dv_mx, sf_dv)

    out_ref, dq_ref, dk_ref, dv_ref = _reference_varlen(
        q,
        k,
        v,
        dout,
        lengths,
    )
    thresholds = {
        "out": (out, out_ref, 0.98, 8.0),
        "dq": (dq, dq_ref, 0.97, 10.0),
        "dk": (dk, dk_ref, 0.97, 8.0),
        "dv": (dv, dv_ref, 0.97, 8.0),
    }
    start = 0
    for index, length in enumerate(lengths):
        end = start + length
        for name, (actual, expected, min_cosine, min_sqnr) in thresholds.items():
            _check_quality(
                f"sequence {index} {name}",
                actual[start:end],
                expected[start:end],
                min_cosine,
                min_sqnr,
            )
        start = end
    for name, (actual, expected, min_cosine, min_sqnr) in thresholds.items():
        _check_quality(name, actual, expected, min_cosine, min_sqnr)


if __name__ == "__main__":
    main()
