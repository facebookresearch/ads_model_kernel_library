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

import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import torch
from lp_fa4.cute.interface import _flash_attn_bwd, _flash_attn_fwd
from quack.blockscaled import pack_scale_2d_to_blocked_contig
from quack.blockscaled.quantize import to_mx


Mxfp8Orientation = Literal["hdim", "seq", "both"]
Mxfp8ScalingMode = Literal["rceil", "floor"]
Mxfp8PScaleMode = Literal["dynamic", "const_p"]
_SCALE_TILE = 128
_SCALE_VECTOR = 32
_E4M3_MAX = 448.0
_F32_MANTISSA_BITS = 23
_F32_MIN_NORMAL = 2.0**-126
_F32_MANTISSA_MASK = (1 << _F32_MANTISSA_BITS) - 1


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _cumulative(values: Sequence[int]) -> tuple[int, ...]:
    offsets = [0]
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("sequence lengths must be nonnegative integers")
        offsets.append(offsets[-1] + value)
    return tuple(offsets)


def _offset_values(offsets: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in offsets.detach().cpu().tolist())


def _lengths_from_offsets(offsets: Sequence[int]) -> tuple[int, ...]:
    if len(offsets) < 2 or offsets[0] != 0:
        raise ValueError("sequence offsets must start at zero and contain a batch")
    lengths = tuple(end - start for start, end in zip(offsets, offsets[1:]))
    if any(length < 0 for length in lengths):
        raise ValueError("sequence offsets must be monotonic")
    return lengths


@dataclass(frozen=True)
class Mxfp8VarlenMeta:
    """Packed-token and independently padded MXFP8 scale offsets."""

    cu_seqlens: torch.Tensor
    cu_seqlens_sf: torch.Tensor
    max_seqlen: int
    cpu_cu_seqlens: torch.Tensor

    def __post_init__(self) -> None:
        _validate_offsets(self.cu_seqlens, "cu_seqlens")
        _validate_offsets(self.cu_seqlens_sf, "cu_seqlens_sf")
        _validate_offsets(
            self.cpu_cu_seqlens,
            "cpu_cu_seqlens",
            require_cpu=True,
        )
        if self.cu_seqlens_sf.device != self.cu_seqlens.device:
            raise ValueError("sequence and scale offsets must be colocated")
        raw = _offset_values(self.cpu_cu_seqlens)
        if _offset_values(self.cu_seqlens) != raw:
            raise ValueError("CPU and device sequence offsets must match")
        lengths = _lengths_from_offsets(raw)
        expected_sf = _cumulative(
            tuple(_round_up(length, _SCALE_TILE) for length in lengths)
        )
        if _offset_values(self.cu_seqlens_sf) != expected_sf:
            raise ValueError("scale offsets must pad every sequence to 128 tokens")
        if self.max_seqlen != max(lengths):
            raise ValueError("max_seqlen must equal the largest sequence length")

    @classmethod
    def from_lengths(
        cls,
        lengths: Sequence[int],
        *,
        device: torch.device | str,
    ) -> Mxfp8VarlenMeta:
        if len(lengths) == 0:
            raise ValueError("at least one sequence length is required")
        raw = _cumulative(lengths)
        padded = _cumulative(
            tuple(_round_up(length, _SCALE_TILE) for length in lengths)
        )
        cpu_raw = torch.tensor(raw, dtype=torch.int32, device="cpu")
        return cls(
            cu_seqlens=cpu_raw.to(device=device),
            cu_seqlens_sf=torch.tensor(padded, dtype=torch.int32, device=device),
            max_seqlen=max(lengths, default=0),
            cpu_cu_seqlens=cpu_raw,
        )

    @classmethod
    def from_cu_seqlens(
        cls,
        cu_seqlens: torch.Tensor,
        *,
        cpu_cu_seqlens: torch.Tensor | None = None,
    ) -> Mxfp8VarlenMeta:
        _validate_offsets(cu_seqlens, "cu_seqlens")
        cpu_offsets = (
            cu_seqlens.detach().cpu() if cpu_cu_seqlens is None else cpu_cu_seqlens
        )
        _validate_offsets(cpu_offsets, "cpu_cu_seqlens", require_cpu=True)
        if _offset_values(cu_seqlens) != _offset_values(cpu_offsets):
            raise ValueError("CPU and device sequence offsets must match")
        lengths = _lengths_from_offsets(_offset_values(cpu_offsets))
        return cls.from_lengths(lengths, device=cu_seqlens.device)

    @property
    def total_tokens(self) -> int:
        return int(self.cpu_cu_seqlens[-1])

    @property
    def padded_tokens(self) -> int:
        lengths = _lengths_from_offsets(_offset_values(self.cpu_cu_seqlens))
        return sum(_round_up(length, _SCALE_TILE) for length in lengths)

    @property
    def batch_size(self) -> int:
        return self.cu_seqlens.shape[0] - 1


@dataclass(frozen=True)
class Mxfp8VarlenTensor:
    """An E4M3 tensor quantized in either or both attention orientations."""

    hdim: torch.Tensor | None = None
    hdim_scale: torch.Tensor | None = None
    seq: torch.Tensor | None = None
    seq_scale: torch.Tensor | None = None


def _validate_offsets(
    offsets: torch.Tensor,
    name: str,
    *,
    require_cpu: bool = False,
) -> None:
    if (
        offsets.dtype != torch.int32
        or offsets.ndim != 1
        or offsets.shape[0] < 2
        or not offsets.is_contiguous()
    ):
        raise ValueError(f"{name} must be contiguous int32 with at least two entries")
    if require_cpu and offsets.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    _lengths_from_offsets(_offset_values(offsets))


def _metadata_segments(meta: Mxfp8VarlenMeta) -> tuple[tuple[int, int], ...]:
    offsets = _offset_values(meta.cpu_cu_seqlens)
    return tuple(zip(offsets, offsets[1:]))


def _quantize_block32(
    tensor: torch.Tensor,
    scaling_mode: Mxfp8ScalingMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _quack_supports_scaling_mode():
        return to_mx(tensor, _SCALE_VECTOR, scaling_mode)
    if scaling_mode == "floor":
        return to_mx(tensor, _SCALE_VECTOR)
    return _quantize_block32_rceil(tensor)


def _quantize_block32_rceil(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_shape = tensor.shape
    blocks = tensor.reshape(
        *original_shape[:-1],
        original_shape[-1] // _SCALE_VECTOR,
        _SCALE_VECTOR,
    )
    max_abs = torch.amax(torch.abs(blocks), dim=-1, keepdim=True).float()
    ratio_bits = (max_abs / _E4M3_MAX).view(torch.int32)
    exponent = (
        torch.bitwise_right_shift(
            ratio_bits,
            _F32_MANTISSA_BITS,
        )
        & 0xFF
    )
    has_fraction = (ratio_bits & _F32_MANTISSA_MASK) != 0
    scale_bytes = torch.clamp(
        exponent + has_fraction.to(torch.int32),
        max=0xFF,
    ).to(torch.uint8)
    scale_bytes = torch.where(torch.isnan(max_abs), 0xFF, scale_bytes)
    scale = torch.bitwise_left_shift(
        scale_bytes.to(torch.int32),
        _F32_MANTISSA_BITS,
    ).view(torch.float32)
    scale = torch.clamp(scale, min=_F32_MIN_NORMAL)
    quantized = torch.clamp(
        blocks.float() / scale,
        min=-_E4M3_MAX,
        max=_E4M3_MAX,
    )
    return (
        quantized.to(torch.float8_e4m3fn).reshape(original_shape),
        scale_bytes.view(torch.float8_e8m0fnu).squeeze(-1),
    )


@lru_cache(maxsize=1)
def _quack_supports_scaling_mode() -> bool:
    return "scaling_mode" in inspect.signature(to_mx).parameters


def _pad_hdim_scale(
    scale: torch.Tensor,
    padded_length: int,
) -> torch.Tensor:
    length, num_heads, scale_dim = scale.shape
    padded = torch.zeros(
        (num_heads, padded_length, scale_dim),
        dtype=torch.uint8,
        device=scale.device,
    )
    padded[:, :length] = scale.permute(1, 0, 2).contiguous().view(torch.uint8)
    return padded.view(scale.dtype)


def _quantize_hdim(
    tensor: torch.Tensor,
    meta: Mxfp8VarlenMeta,
    scaling_mode: Mxfp8ScalingMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    qdata, scale = _quantize_block32(tensor, scaling_mode)
    padded_scales = [
        _pad_hdim_scale(scale[start:end], _round_up(end - start, _SCALE_TILE))
        for start, end in _metadata_segments(meta)
    ]
    logical_scale = torch.cat(padded_scales, dim=1)
    blocked_scale = pack_scale_2d_to_blocked_contig(logical_scale)
    return qdata, blocked_scale.reshape(-1, 16).contiguous()


def _quantize_seq_segment(
    segment: torch.Tensor,
    scaling_mode: Mxfp8ScalingMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    length, num_heads, head_dim = segment.shape
    if length == 0:
        return (
            segment.to(torch.float8_e4m3fn),
            torch.empty(
                (num_heads, head_dim, 0),
                dtype=torch.float8_e8m0fnu,
                device=segment.device,
            ),
        )
    padded_length = _round_up(length, _SCALE_TILE)
    padded = torch.zeros(
        (num_heads, head_dim, padded_length),
        dtype=segment.dtype,
        device=segment.device,
    )
    padded[..., :length] = segment.permute(1, 2, 0)
    qdata, scale = _quantize_block32(padded.contiguous(), scaling_mode)
    return qdata[..., :length].permute(2, 0, 1).contiguous(), scale


def _quantize_seq(
    tensor: torch.Tensor,
    meta: Mxfp8VarlenMeta,
    scaling_mode: Mxfp8ScalingMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    quantized_and_scales = [
        _quantize_seq_segment(tensor[start:end], scaling_mode)
        for start, end in _metadata_segments(meta)
    ]
    qdata = torch.cat([pair[0] for pair in quantized_and_scales], dim=0)
    logical_scale = torch.cat([pair[1] for pair in quantized_and_scales], dim=2)
    blocked_scale = pack_scale_2d_to_blocked_contig(logical_scale)
    num_heads = tensor.shape[1]
    return qdata, blocked_scale.reshape(num_heads * 32, -1).contiguous()


def _validate_source_tensor(
    tensor: torch.Tensor,
    meta: Mxfp8VarlenMeta,
) -> None:
    if tensor.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("MXFP8 source tensors must be BF16 or FP32")
    if tensor.ndim != 3 or tensor.shape[-1] != 128 or tensor.shape[1] < 1:
        raise ValueError("MXFP8 source tensors must have shape (tokens, heads, 128)")
    if tensor.shape[0] != meta.total_tokens:
        raise ValueError("source token count must match the sequence offsets")
    if not tensor.is_contiguous() or tensor.device != meta.cu_seqlens.device:
        raise ValueError("source tensors must be contiguous and colocated with offsets")
    if meta.padded_tokens == 0:
        raise ValueError("MXFP8 quantization requires at least one token")


def quantize_mxfp8_varlen(
    tensor: torch.Tensor,
    meta: Mxfp8VarlenMeta,
    *,
    orientations: Mxfp8Orientation = "both",
    scaling_mode: Mxfp8ScalingMode = "rceil",
) -> Mxfp8VarlenTensor:
    """Quantize packed D128 MHA data without crossing sequence boundaries."""
    _validate_source_tensor(tensor, meta)
    if orientations not in ("hdim", "seq", "both"):
        raise ValueError("orientations must be hdim, seq, or both")
    if scaling_mode not in ("rceil", "floor"):
        raise ValueError("scaling_mode must be rceil or floor")
    hdim, hdim_scale = (
        _quantize_hdim(tensor, meta, scaling_mode)
        if orientations in ("hdim", "both")
        else (None, None)
    )
    seq, seq_scale = (
        _quantize_seq(tensor, meta, scaling_mode)
        if orientations in ("seq", "both")
        else (None, None)
    )
    return Mxfp8VarlenTensor(hdim, hdim_scale, seq, seq_scale)


def _validate_operand(
    operand: Mxfp8VarlenTensor,
    meta: Mxfp8VarlenMeta,
    *,
    num_heads: int,
    require_hdim: bool,
    require_seq: bool,
) -> None:
    expected = (meta.total_tokens, num_heads, 128)
    for payload, scale, orientation in (
        (operand.hdim, operand.hdim_scale, "hdim"),
        (operand.seq, operand.seq_scale, "seq"),
    ):
        required = require_hdim if orientation == "hdim" else require_seq
        if not required:
            continue
        if payload is None or scale is None or payload.shape != expected:
            raise ValueError(f"missing or invalid {orientation} MXFP8 operand")
        if payload.dtype != torch.float8_e4m3fn or not payload.is_contiguous():
            raise TypeError(f"{orientation} payload must be contiguous E4M3")
        e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
        if scale.dtype not in (torch.uint8, e8m0_dtype):
            raise TypeError(f"{orientation} scale must contain E8M0 bytes")
        if payload.device != meta.cu_seqlens.device:
            raise ValueError(f"{orientation} payload must be colocated with offsets")
        if scale.device != payload.device or not scale.is_contiguous():
            raise ValueError(f"{orientation} scale must be contiguous and colocated")
    if require_hdim and operand.hdim_scale.shape != (
        num_heads * meta.padded_tokens // 4,
        16,
    ):
        raise ValueError("invalid head-dimension MXFP8 scale shape")
    if require_seq and operand.seq_scale.shape != (
        num_heads * 32,
        meta.padded_tokens // 8,
    ):
        raise ValueError("invalid sequence-axis MXFP8 scale shape")


def _validate_attention_metadata(
    q_meta: Mxfp8VarlenMeta,
    k_meta: Mxfp8VarlenMeta,
    *,
    broadcast_q: bool,
) -> None:
    expected_q_batch = 1 if broadcast_q else k_meta.batch_size
    if q_meta.batch_size != expected_q_batch:
        raise ValueError("Q and K metadata have incompatible batch sizes")
    if q_meta.cu_seqlens.device != k_meta.cu_seqlens.device:
        raise ValueError("Q and K metadata must be on the same device")


def mxfp8_flash_attn_varlen_forward(
    q: Mxfp8VarlenTensor,
    k: Mxfp8VarlenTensor,
    v: Mxfp8VarlenTensor,
    q_meta: Mxfp8VarlenMeta,
    k_meta: Mxfp8VarlenMeta,
    *,
    softmax_scale: float | None = None,
    broadcast_q: bool = False,
    return_lse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run noncausal packed D128 MHA forward on prequantized operands."""
    _validate_attention_metadata(q_meta, k_meta, broadcast_q=broadcast_q)
    num_heads = q.hdim.shape[1] if q.hdim is not None else 0
    _validate_operand(
        q, q_meta, num_heads=num_heads, require_hdim=True, require_seq=False
    )
    _validate_operand(
        k, k_meta, num_heads=num_heads, require_hdim=True, require_seq=False
    )
    _validate_operand(
        v, k_meta, num_heads=num_heads, require_hdim=False, require_seq=True
    )
    out, lse, _p, _row_max = _flash_attn_fwd(
        q.hdim,
        k.hdim,
        v.seq,
        cu_seqlens_q=q_meta.cu_seqlens,
        cu_seqlens_k=k_meta.cu_seqlens,
        max_seqlen_q=q_meta.max_seqlen,
        max_seqlen_k=k_meta.max_seqlen,
        softmax_scale=softmax_scale,
        return_lse=return_lse,
        sfq=q.hdim_scale,
        sfk=k.hdim_scale,
        sfv=v.seq_scale,
        cu_seqlens_sf_q=q_meta.cu_seqlens_sf,
        cu_seqlens_sf_k=k_meta.cu_seqlens_sf,
        broadcast_q=broadcast_q,
        cpu_cu_seqlens_k=k_meta.cpu_cu_seqlens,
    )
    return out, lse


def mxfp8_flash_attn_varlen_backward(
    q: Mxfp8VarlenTensor,
    k: Mxfp8VarlenTensor,
    v: Mxfp8VarlenTensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    q_meta: Mxfp8VarlenMeta,
    k_meta: Mxfp8VarlenMeta,
    *,
    softmax_scale: float | None = None,
    broadcast_q: bool = False,
    dq_accum_dtype: torch.dtype = torch.float32,
    p_scale_mode: Mxfp8PScaleMode = "const_p",
    output_mxfp8_dkv: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
):
    """Run MXFP8 backward; optionally emit fused E4M3 dK/dV and E8M0 scales."""
    _validate_attention_metadata(q_meta, k_meta, broadcast_q=broadcast_q)
    num_heads = q.hdim.shape[1] if q.hdim is not None else 0
    _validate_operand(
        q, q_meta, num_heads=num_heads, require_hdim=True, require_seq=True
    )
    _validate_operand(
        k, k_meta, num_heads=num_heads, require_hdim=True, require_seq=True
    )
    _validate_operand(
        v, k_meta, num_heads=num_heads, require_hdim=True, require_seq=False
    )
    result = _flash_attn_bwd(
        q.hdim,
        k.hdim,
        v.hdim,
        out,
        dout,
        lse,
        softmax_scale=softmax_scale,
        cu_seqlens_q=q_meta.cu_seqlens,
        cu_seqlens_k=k_meta.cu_seqlens,
        max_seqlen_q=q_meta.max_seqlen,
        max_seqlen_k=k_meta.max_seqlen,
        dq_accum_dtype=dq_accum_dtype,
        mxfp8_internal_scale_mode=p_scale_mode,
        sfq=q.hdim_scale,
        sfk=k.hdim_scale,
        sfv=v.hdim_scale,
        q_dk=q.seq,
        sfq_dk=q.seq_scale,
        k_dq=k.seq,
        sfk_dq=k.seq_scale,
        cu_seqlens_sf_q=q_meta.cu_seqlens_sf,
        cu_seqlens_sf_k=k_meta.cu_seqlens_sf,
        broadcast_q=broadcast_q,
        cpu_cu_seqlens_k=k_meta.cpu_cu_seqlens,
        output_mxfp8_dkv=output_mxfp8_dkv,
    )
    return result
