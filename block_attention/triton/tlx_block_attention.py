# pyre-unsafe
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

import math
import types
import typing

import torch
import triton  # @manual=//triton:triton
import triton.language as tl  # @manual=//triton:triton
import triton.language.extra.tlx as tlx  # @manual=//triton:triton
from ads_mkl.ops.cute_dsl.block_attention.triton.rotary_utils import (
    apply_rotary_pos_emb_jagged,
)
from torch.utils.flop_counter import (
    _unpack_flash_attention_nested_shapes,
    register_flop_formula,
    sdpa_backward_flop_count,
    sdpa_flop_count,
)

try:
    # Prefer native meta-shape helper when available.
    from torch.utils.flop_counter import (  # type: ignore[attr-defined]
        _unpack_nested_shapes as _unpack_nested_shapes_meta,
    )
except Exception:

    def _unpack_nested_shapes_meta(**kwargs):
        # Fallback keeps this module self-contained without external ads_mkl imports.
        return _unpack_flash_attention_nested_shapes(**kwargs)


from triton.tools.tensor_descriptor import TensorDescriptor


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def binary_search_le(ptr, value, length):
    """Find the largest index where ptr[index] <= value"""
    low = 0
    high = length - 1
    result = 0
    for _ in range(32):  # Enough for any reasonable size
        mid = (low + high) // 2
        mid_val = tl.load(ptr + mid)
        # If mid_val <= value, this could be our answer, search right
        go_right = mid_val <= value
        result = tl.where(go_right, mid, result)
        low = tl.where(go_right, mid + 1, low)
        high = tl.where(go_right, high, mid - 1)
    OF = tl.load(ptr + result).to(tl.int32)
    return result, OF


@triton.jit
def _allocate_buffers(
    BLOCK_D,
    HEAD_DIM,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    SMEM_BUFFER,
    TMEM_BUFFER,
):
    # Descriptors are now passed from outside the kernel

    q_tile = tlx.local_alloc(
        (BLOCK_D, HEAD_DIM), desc_q.dtype, tl.constexpr(SMEM_BUFFER)
    )
    k_tile = tlx.local_alloc(
        (BLOCK_D, HEAD_DIM), desc_q.dtype, tl.constexpr(SMEM_BUFFER)
    )
    v_tile = tlx.local_alloc(
        (BLOCK_D, HEAD_DIM), desc_q.dtype, tl.constexpr(SMEM_BUFFER)
    )
    p_tile = tlx.local_alloc(
        (BLOCK_D, BLOCK_D), desc_q.dtype, tl.constexpr(SMEM_BUFFER)
    )
    TMEMqk = tlx.local_alloc(
        (BLOCK_D, BLOCK_D), tl.float32, tl.constexpr(TMEM_BUFFER), tlx.storage_kind.tmem
    )
    TMEMpv = tlx.local_alloc(
        (BLOCK_D, HEAD_DIM),
        tl.float32,
        tl.constexpr(TMEM_BUFFER),
        tlx.storage_kind.tmem,
    )
    return (
        q_tile,
        k_tile,
        v_tile,
        p_tile,
        TMEMqk,
        TMEMpv,
        desc_q,
        desc_k,
        desc_v,
        desc_o,
    )


@triton.jit
def _block_attention(  # noqa: C901
    query,
    key,
    value,
    out,
    offsets,
    BLOCK_PER_PROGRAM,
    BLOCK_PER_BATCH,
    stride_seq,
    stride_head,
    stride_dim,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    TOTAL_LEN_Q,
    sm_scale,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_BATCHES: tl.constexpr,
    SMEM_BUFFER: tl.constexpr,
    TMEM_BUFFER: tl.constexpr,
):
    LOG2_E = 1.4426950408889634  # = 1 / ln(2)
    tid = tl.program_id(0)
    start = tl.load(BLOCK_PER_PROGRAM + tid).to(tl.int32)
    end = tl.load(BLOCK_PER_PROGRAM + tid + 1).to(tl.int32)
    iters = end - start

    TOTAL_PER_HEAD = tl.load(BLOCK_PER_BATCH + NUM_BATCHES).to(tl.int32)
    global_hidx = start // TOTAL_PER_HEAD
    start = start % TOTAL_PER_HEAD
    offIDX, left = binary_search_le(BLOCK_PER_BATCH, start, NUM_BATCHES)
    batch_start = tl.load(offsets + offIDX).to(tl.int32)

    q_tile, k_tile, v_tile, p_tile, TMEMqk, TMEMpv, desc_q, desc_k, desc_v, desc_out = (
        _allocate_buffers(
            BLOCK_D,
            HEAD_DIM,
            desc_q,
            desc_k,
            desc_v,
            desc_o,
            SMEM_BUFFER,
            TMEM_BUFFER,
        )
    )

    qk_SMEM_free = tlx.alloc_barriers(SMEM_BUFFER, 1)  # type: ignore[arg-type]
    qk_SMEM_full = tlx.alloc_barriers(SMEM_BUFFER, 1)  # type: ignore[arg-type]

    pv_SMEM_full = tlx.alloc_barriers(SMEM_BUFFER, 1)  # type: ignore[arg-type]
    pv_SMEM_free = tlx.alloc_barriers(SMEM_BUFFER, 1)  # type: ignore[arg-type]

    v_SMEM_free = tlx.alloc_barriers(SMEM_BUFFER, 1)  # type: ignore[arg-type]
    v_SMEM_full = tlx.alloc_barriers(SMEM_BUFFER, 1)  # type: ignore[arg-type]

    qk_TMEM_free = tlx.alloc_barriers(TMEM_BUFFER, 1)  # type: ignore[arg-type]
    qk_TMEM_full = tlx.alloc_barriers(TMEM_BUFFER, 1)  # type: ignore[arg-type]

    pv_TMEM_free = tlx.alloc_barriers(TMEM_BUFFER, 1)  # type: ignore[arg-type]
    pv_TMEM_full = tlx.alloc_barriers(TMEM_BUFFER, 1)  # type: ignore[arg-type]

    with tlx.async_tasks():
        # qk load
        with tlx.async_task("default", num_warps=1, registers=48):
            q_current = ((start - left) * BLOCK_D).to(tl.int32) + batch_start
            Z_next = tl.load(offsets + offIDX + 1).to(tl.int32)
            offset_idx = offIDX
            hidx = global_hidx

            for i in range(iters):
                buff_idx = i % SMEM_BUFFER
                buff_phase = i // SMEM_BUFFER

                tlx.barrier_wait(qk_SMEM_free[buff_idx], (buff_phase & 1) ^ 1)
                tlx.barrier_expect_bytes(
                    qk_SMEM_full[buff_idx], 2 * 2 * BLOCK_D * HEAD_DIM
                )  # BF16 2bytes
                tlx.async_descriptor_load(
                    desc_q,
                    q_tile[buff_idx],
                    [q_current, hidx * HEAD_DIM],
                    qk_SMEM_full[buff_idx],
                )
                ##k
                tlx.async_descriptor_load(
                    desc_k,
                    k_tile[buff_idx],
                    [q_current, hidx * HEAD_DIM],
                    qk_SMEM_full[buff_idx],
                )

                # v
                tlx.barrier_wait(v_SMEM_free[buff_idx], (buff_phase & 1) ^ 1)
                tlx.barrier_expect_bytes(v_SMEM_full[buff_idx], 2 * BLOCK_D * HEAD_DIM)
                tlx.async_descriptor_load(
                    desc_v,
                    v_tile[buff_idx],
                    [q_current, hidx * HEAD_DIM],
                    v_SMEM_full[buff_idx],
                )

                q_current += BLOCK_D
                if q_current >= Z_next:
                    q_current = Z_next % TOTAL_LEN_Q
                    offset_idx += 1
                if offset_idx == NUM_BATCHES:
                    offset_idx = 0
                    hidx += 1
                Z_next = tl.load(offsets + offset_idx + 1).to(tl.int32)

        # q@k MMA
        with tlx.async_task(num_warps=1, registers=48):
            q_current = ((start - left) * BLOCK_D).to(tl.int32) + batch_start
            offset_idx = offIDX
            Z_next = tl.load(offsets + offset_idx + 1).to(tl.int32)
            hidx = global_hidx

            for i in range(iters):
                buff_idx = i % SMEM_BUFFER
                buff_phase = i // SMEM_BUFFER

                TMEM_PHASE = i // TMEM_BUFFER
                tmem_idx = i % TMEM_BUFFER
                _step = TMEM_PHASE & 1  # noqa: F841

                tlx.barrier_wait(qk_SMEM_full[buff_idx], (buff_phase & 1))
                k_tile_T = tlx.local_trans(k_tile[buff_idx])

                tlx.barrier_wait(qk_TMEM_free[tmem_idx], (TMEM_PHASE & 1) ^ 1)
                tlx.async_dot(
                    q_tile[buff_idx],
                    k_tile_T,
                    TMEMqk[tmem_idx],
                    use_acc=False,
                    mBarriers=[qk_SMEM_free[buff_idx], qk_TMEM_full[tmem_idx]],
                )

                q_current += BLOCK_D
                if q_current >= Z_next:
                    q_current = Z_next % TOTAL_LEN_Q
                    offset_idx += 1
                if offset_idx == NUM_BATCHES:
                    offset_idx = 0
                    hidx += 1
                Z_next = tl.load(offsets + offset_idx + 1).to(tl.int32)

        # Softmax load P, V
        with tlx.async_task(num_warps=4, registers=120):
            q_current = ((start - left) * BLOCK_D).to(tl.int32) + batch_start
            Z_next = tl.load(offsets + offIDX + 1).to(tl.int32)
            offset_idx = offIDX
            hidx = global_hidx

            for i in range(iters):
                buff_idx = i % SMEM_BUFFER
                buff_phase = i // SMEM_BUFFER
                TMEM_PHASE = i // TMEM_BUFFER
                tmem_idx = i % TMEM_BUFFER
                _step = TMEM_PHASE & 1  # noqa: F841

                tlx.barrier_wait(qk_TMEM_full[tmem_idx], (TMEM_PHASE & 1))
                result = tlx.local_load(TMEMqk[tmem_idx])
                tlx.barrier_arrive(qk_TMEM_free[tmem_idx])

                # ===== Mask =====
                if Z_next < (q_current + BLOCK_D):
                    block_size = Z_next - q_current
                    row_indices = tl.arange(0, BLOCK_D)
                    col_indices = tl.arange(0, BLOCK_D)
                    valid_mask = (row_indices[:, None] < block_size) & (
                        col_indices[None, :] < block_size
                    )
                    result = tl.where(valid_mask, result, -1.0e9)

                ## Load P
                # Scale factor for log2-based exp
                result = result * (sm_scale * LOG2_E)
                max_val = tl.max(result, axis=1)[:, None]
                result = result - max_val
                result = tl.math.exp2(result)  # Use exp2 instead of exp (faster on GPU)
                sum_val = tl.sum(result, axis=1)[:, None]
                P = result / sum_val

                tlx.barrier_wait(pv_SMEM_free[buff_idx], (buff_phase & 1) ^ 1)
                tlx.local_store(p_tile[buff_idx], P.to(tlx.dtype_of(desc_v)))
                tlx.fence_async_shared()
                tlx.barrier_arrive(pv_SMEM_full[buff_idx])

                q_current += BLOCK_D
                if q_current >= Z_next:
                    q_current = Z_next % TOTAL_LEN_Q
                    offset_idx += 1
                if offset_idx == NUM_BATCHES:
                    offset_idx = 0
                    hidx += 1
                Z_next = tl.load(offsets + offset_idx + 1).to(tl.int32)

        # p@v MMA
        with tlx.async_task(num_warps=1, registers=48):
            q_current = ((start - left) * BLOCK_D).to(tl.int32) + batch_start
            Z_next = tl.load(offsets + offIDX + 1).to(tl.int32)
            offset_idx = offIDX
            hidx = global_hidx

            for i in range(iters):
                buff_idx = i % SMEM_BUFFER
                buff_phase = i // SMEM_BUFFER
                TMEM_PHASE = i // TMEM_BUFFER
                tmem_idx = i % TMEM_BUFFER
                _step = TMEM_PHASE & 1  # noqa: F841

                tlx.barrier_wait(pv_SMEM_full[buff_idx], (buff_phase & 1))
                tlx.barrier_wait(pv_TMEM_free[tmem_idx], (TMEM_PHASE & 1) ^ 1)

                tlx.barrier_wait(v_SMEM_full[buff_idx], (buff_phase & 1))

                tlx.async_dot(
                    p_tile[buff_idx],
                    v_tile[buff_idx],
                    TMEMpv[tmem_idx],
                    use_acc=False,
                    mBarriers=[
                        v_SMEM_free[buff_idx],
                        pv_TMEM_full[tmem_idx],
                        pv_SMEM_free[buff_idx],
                    ],
                )

                q_current += BLOCK_D
                if q_current >= Z_next:
                    q_current = Z_next % TOTAL_LEN_Q
                    offset_idx += 1
                if offset_idx == NUM_BATCHES:
                    offset_idx = 0
                    hidx += 1
                Z_next = tl.load(offsets + offset_idx + 1).to(tl.int32)

        # Epilogue
        with tlx.async_task(num_warps=8, registers=200):
            q_current = ((start - left) * BLOCK_D).to(tl.int32) + batch_start
            Z_next = tl.load(offsets + offIDX + 1).to(tl.int32)
            offset_idx = offIDX
            hidx = global_hidx

            for i in range(iters):
                TMEM_PHASE = i // TMEM_BUFFER
                tmem_idx = i % TMEM_BUFFER

                tlx.barrier_wait(pv_TMEM_full[tmem_idx], TMEM_PHASE & 1)

                PV = tlx.local_load(TMEMpv[tmem_idx])
                tlx.barrier_arrive(pv_TMEM_free[tmem_idx])
                fin = PV.to(tlx.dtype_of(desc_out))

                if Z_next >= (q_current + BLOCK_D):
                    desc_out.store([q_current, hidx * HEAD_DIM], fin)
                else:
                    # the only time this should be necessary is bleeding into next program
                    # Otherwise should be syncronous
                    # if i == iters - 1:
                    block_size = Z_next - q_current
                    offs_m = (q_current + tl.arange(0, BLOCK_D)).to(tl.int64)
                    offs_d = (hidx * HEAD_DIM + tl.arange(0, HEAD_DIM)).to(tl.int64)

                    mask = tl.arange(0, BLOCK_D)[:, None] < block_size

                    out_ptrs = (
                        out
                        + offs_m[:, None] * stride_seq
                        + offs_d[None, :] * stride_dim
                    )
                    tl.store(out_ptrs, fin, mask=mask)
                    # else:
                    #     desc_out.store([q_current, hidx * HEAD_DIM], fin)

                q_current += BLOCK_D
                if q_current >= Z_next:
                    q_current = Z_next % TOTAL_LEN_Q
                    offset_idx += 1
                if offset_idx == NUM_BATCHES:
                    offset_idx = 0
                    hidx += 1
                Z_next = tl.load(offsets + offset_idx + 1).to(tl.int32)


def get_batches_old(offsets, BLOCK_D, NUM_SMS, HEADS):
    # Vectorized block calculation
    sizes = offsets[1:] - offsets[:-1]
    num_blocks = (sizes + BLOCK_D - 1) // BLOCK_D
    BLOCK_PER_BATCH = torch.cat(
        [
            torch.zeros(1, dtype=num_blocks.dtype, device=offsets.device),
            num_blocks.cumsum(0),
        ]
    )
    total_blocks = BLOCK_PER_BATCH[-1] * HEADS
    # Vectorized program partitioning
    if total_blocks <= NUM_SMS:
        BLOCK_PER_PROGRAM = torch.arange(0, total_blocks + 1, device=offsets.device)
        re = 1
    else:
        re = total_blocks // NUM_SMS
        rem = total_blocks % NUM_SMS
        base = torch.full((NUM_SMS,), re, device=offsets.device)
        base[:rem] += 1
        BLOCK_PER_PROGRAM = torch.cat(
            [torch.zeros(1, dtype=base.dtype, device=offsets.device), base.cumsum(0)]
        )
    return BLOCK_PER_BATCH, total_blocks, BLOCK_PER_PROGRAM, re


# Fall back to using gpu offsets which costs a streamSyncronize
def get_batches_no_cpu(offsets, BLOCK_D: int, NUM_SMS: int, HEADS: int):
    device = offsets.device
    dtype = offsets.dtype

    # Vectorized block calculation
    sizes = offsets[1:] - offsets[:-1]
    num_blocks = (sizes + BLOCK_D - 1) // BLOCK_D

    # Optimization 1: Pre-allocate + cumsum with out parameter (avoids torch.cat)
    batch_count = num_blocks.numel()
    BLOCK_PER_BATCH = torch.empty(batch_count + 1, dtype=dtype, device=device)
    BLOCK_PER_BATCH[0] = 0
    torch.cumsum(num_blocks, dim=0, out=BLOCK_PER_BATCH[1:])

    total_blocks = (BLOCK_PER_BATCH[-1] * HEADS).item()

    # Vectorized program partitioning
    if total_blocks <= NUM_SMS:
        BLOCK_PER_PROGRAM = torch.arange(NUM_SMS + 1, device=device, dtype=torch.int64)
        re = 1
    else:
        # Optimization 2: Use divmod for single operation
        re, rem = divmod(total_blocks, NUM_SMS)

        # Optimization 3: Direct closed-form computation
        # Instead of: full → slice assign → cumsum → cat
        # Use formula: BLOCK_PER_PROGRAM[i] = i * re + min(i, rem)
        BLOCK_PER_PROGRAM = torch.empty(NUM_SMS + 1, dtype=torch.int64, device=device)
        BLOCK_PER_PROGRAM[0] = 0
        indices = torch.arange(1, NUM_SMS + 1, device=device, dtype=torch.int64)
        BLOCK_PER_PROGRAM[1:] = indices * re + torch.clamp(indices, max=rem)

    return BLOCK_PER_BATCH, total_blocks, BLOCK_PER_PROGRAM, re


def get_batches(
    offsets, BLOCK_D: int, NUM_SMS: int, HEADS: int, cpu_offsets: torch.Tensor
):
    if cpu_offsets is None:
        return get_batches_no_cpu(offsets, BLOCK_D, NUM_SMS, HEADS)
    device = offsets.device
    dtype = offsets.dtype
    # ===== ALL scalar computation on CPU (no GPU sync) =====
    sizes_cpu = cpu_offsets[1:] - cpu_offsets[:-1]
    num_blocks_cpu = (sizes_cpu + BLOCK_D - 1) // BLOCK_D
    batch_count = num_blocks_cpu.numel()  # Python int from CPU tensor metadata
    total_blocks = (num_blocks_cpu.sum() * HEADS).item()  # Python int from CPU tensor
    # ===== GPU tensor construction =====
    sizes = offsets[1:] - offsets[:-1]
    num_blocks = (sizes + BLOCK_D - 1) // BLOCK_D
    BLOCK_PER_BATCH = torch.zeros(batch_count + 1, dtype=dtype, device=device)
    torch.cumsum(num_blocks, dim=0, out=BLOCK_PER_BATCH[1:])
    # ===== Program partitioning (all Python int math, no GPU sync) =====
    if total_blocks <= NUM_SMS:
        BLOCK_PER_PROGRAM = torch.arange(NUM_SMS + 1, device=device, dtype=torch.int64)
        re = 1
    else:
        re, rem = divmod(total_blocks, NUM_SMS)
        indices = torch.arange(NUM_SMS + 1, device=device, dtype=torch.int64)
        BLOCK_PER_PROGRAM = indices * re + torch.clamp(indices, max=rem)
    return BLOCK_PER_BATCH, total_blocks, BLOCK_PER_PROGRAM, re


def expect_contiguous(x: torch.Tensor) -> torch.Tensor:
    if x is not None and not x.is_contiguous():
        return x.contiguous()
    return x


@torch.library.custom_op("ads_mkl::tlx_block_attention", mutates_args=())
def tlx_block_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    offsets: torch.Tensor,
    cpu_offset: typing.Optional[torch.Tensor],
    include_projection_rotary: bool = False,
    x: typing.Optional[torch.Tensor] = None,
    w: typing.Optional[torch.Tensor] = None,
    sinu_pos: typing.Optional[torch.Tensor] = None,
    qhead: typing.Optional[int] = None,
    kvhead: typing.Optional[int] = None,
    max_seq_len: typing.Optional[int] = 0,
    bias: typing.Optional[torch.Tensor] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Block attention forward pass.

    When fused=False (default): Standard block attention on q, k, v tensors.
    When fused=True: Runs fused projection + rotary to get q, k, v, then attention.
        Requires: x, w, sinu_pos, qhead, kvhead (and optionally bias, max_seq_len)

    Returns:
        (out, BLOCK_PER_BATCH, BLOCK_PER_PROGRAM, num_blocks, q, k, v)
        - q, k, v are None when fused=False (already provided as input)
        - q, k, v are the intermediate tensors when fused=True (needed for backward)
    """
    # If fused, run projection + rotary to get q, k, v
    if include_projection_rotary:
        assert x is not None, "x is required when fused=True"
        assert w is not None, "w is required when fused=True"
        assert sinu_pos is not None, "sinu_pos is required when fused=True"
        assert qhead is not None, "qhead is required when fused=True"
        assert kvhead is not None, "kvhead is required when fused=True"

        qkv = torch.matmul(x, w)
        if bias is not None:
            qkv.add_(bias)
        head_dim = x.shape[1] // qhead
        d_model = qhead * head_dim
        kv_d_model = kvhead * head_dim
        query = qkv[:, :d_model].reshape(-1, qhead, head_dim).contiguous()
        key = (
            qkv[:, d_model : d_model + kv_d_model]
            .reshape(-1, kvhead, head_dim)
            .contiguous()
        )
        value = (
            qkv[:, d_model + kv_d_model :].reshape(-1, kvhead, head_dim).contiguous()
        )
        del qkv
        query, key, value = apply_rotary_pos_emb_jagged(query, key, value, sinu_pos)

    BLOCK_D = 64
    HEAD_DIM = query.shape[2]
    NUM_HEADS = query.shape[1]
    TOTAL_LEN_Q = query.shape[0]
    NUM_BATCHES = offsets.shape[0] - 1
    assert HEAD_DIM == 64 or HEAD_DIM == 128
    if cpu_offset is not None:
        assert cpu_offset.is_cpu

    query = expect_contiguous(query)
    key = expect_contiguous(key)
    value = expect_contiguous(value)

    device = torch.device("cuda:0")  # Use your target device index
    NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count
    # print("NUM_SMS", NUM_SMS)
    BLOCK_PER_BATCH, num_blocks, BLOCK_PER_PROGRAM, ITERS = get_batches(
        offsets, BLOCK_D, NUM_SMS, NUM_HEADS, cpu_offset
    )

    assert BLOCK_PER_PROGRAM.device == DEVICE
    assert BLOCK_PER_BATCH.device == DEVICE
    assert offsets.device == DEVICE

    _NUM_PROGRAM = min(NUM_SMS, num_blocks)  # noqa: F841
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    out = expect_contiguous(torch.empty_like(query))
    stride_seq = out.stride(0)
    stride_head = out.stride(1)
    stride_dim = out.stride(2)

    # Create tensor descriptors outside the kernel
    desc_q = TensorDescriptor(
        query,
        shape=[TOTAL_LEN_Q, HEAD_DIM * NUM_HEADS],
        strides=[HEAD_DIM * NUM_HEADS, 1],
        block_shape=[BLOCK_D, HEAD_DIM],
    )
    desc_k = TensorDescriptor(
        key,
        shape=[TOTAL_LEN_Q, HEAD_DIM * NUM_HEADS],
        strides=[HEAD_DIM * NUM_HEADS, 1],
        block_shape=[BLOCK_D, HEAD_DIM],
    )
    desc_v = TensorDescriptor(
        value,
        shape=[TOTAL_LEN_Q, HEAD_DIM * NUM_HEADS],
        strides=[HEAD_DIM * NUM_HEADS, 1],
        block_shape=[BLOCK_D, HEAD_DIM],
    )
    desc_o = TensorDescriptor(
        out,
        shape=[TOTAL_LEN_Q, HEAD_DIM * NUM_HEADS],
        strides=[HEAD_DIM * NUM_HEADS, 1],
        block_shape=[BLOCK_D, HEAD_DIM],
    )

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    kern_kwargs = {
        "BLOCK_D": tl.constexpr(64),
        "HEAD_DIM": tl.constexpr(HEAD_DIM),
        "NUM_BATCHES": tl.constexpr(NUM_BATCHES),
        "SMEM_BUFFER": tl.constexpr(3),
        "TMEM_BUFFER": tl.constexpr(2),
    }

    _block_attention[(min(NUM_SMS, num_blocks), 1)](
        query,
        key,
        value,
        out,
        offsets,
        BLOCK_PER_PROGRAM,
        BLOCK_PER_BATCH,
        stride_seq,
        stride_head,
        stride_dim,
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        TOTAL_LEN_Q,
        sm_scale,
        **kern_kwargs,
    )

    # Return q, k, v only when fused (needed for backward)
    # When not fused, return empty tensors (custom ops can't return None)
    if include_projection_rotary:
        # Return detached aux tensors so autograd does not materialize grad buffers
        # for q/k/v output slots.
        return (
            out,
            BLOCK_PER_BATCH,
            BLOCK_PER_PROGRAM,
            num_blocks,
            query.detach(),
            key.detach(),
            value.detach(),
        )
    else:
        empty_q = None
        empty_k = None
        empty_v = None
        return (
            out,
            BLOCK_PER_BATCH,
            BLOCK_PER_PROGRAM,
            num_blocks,
            empty_q,
            empty_k,
            empty_v,
        )


@torch.fx.wrap
def block_attention_api(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    offsets: torch.Tensor,
    cpu_offsets: torch.Tensor,
) -> torch.Tensor:
    return tlx_block_attention(query, key, value, offsets, cpu_offsets)[0]


def block_attention_setup_context(ctx, inputs, output):
    (
        query,
        key,
        value,
        offsets,
        cpu_offsets,
        include_projection_rotary,
        x,
        w,
        sinu_pos,
        qhead,
        kvhead,
        max_seq_len,
        bias,
    ) = inputs
    (out, BLOCK_PER_BATCH, BLOCK_PER_PROGRAM, num_blocks, q_fused, k_fused, v_fused) = (
        output
    )

    # Only fused path has aux q/k/v outputs where we want undefined grads to stay None.
    if include_projection_rotary:
        ctx.set_materialize_grads(False)

    # Mark q/k/v outputs as non-differentiable so autograd does NOT create
    # zero-filled gradient accumulators (FillFunctor<BFloat16>) for them.
    # Only 'out' needs gradients. q/k/v are auxiliary outputs needed only
    # for setup_context to save them for backward.
    # See: https://docs.pytorch.org/docs/stable/generated/torch.autograd.function.FunctionCtx.mark_non_differentiable.html
    if include_projection_rotary and q_fused is not None:
        ctx.mark_non_differentiable(q_fused, k_fused, v_fused)

    # When fused=True, q/k/v come from output (computed via projection+rotary)
    # When fused=False, q/k/v are the original inputs
    if include_projection_rotary:
        query = q_fused
        key = k_fused
        value = v_fused

    BLOCK_D = 64
    HEAD_DIM = query.shape[2]
    NUM_HEADS = query.shape[1]
    TOTAL_LEN_Q = query.shape[0]
    NUM_BATCHES = offsets.shape[0] - 1
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    # Save tensors needed for backward
    if include_projection_rotary:
        # Save detached q/k/v in ctx to avoid carrying autograd view/history metadata.
        query_saved = query.detach()
        key_saved = key.detach()
        value_saved = value.detach()
        # For fused mode: save x, w for projection backward gemms.
        ctx.save_for_backward(
            x,
            w,
            sinu_pos,
            offsets,
            query_saved,
            key_saved,
            value_saved,
            BLOCK_PER_BATCH,
            BLOCK_PER_PROGRAM,
        )
        ctx.qhead = qhead
        ctx.kvhead = kvhead
        ctx.bias = bias
    else:
        # For non-fused mode: just save q, k, v
        ctx.save_for_backward(
            query, key, value, offsets, BLOCK_PER_BATCH, BLOCK_PER_PROGRAM
        )

    ctx.include_projection_rotary = include_projection_rotary
    ctx.num_blocks = num_blocks
    ctx.sm_scale = sm_scale
    ctx.TOTAL_LEN_Q = TOTAL_LEN_Q
    ctx.NUM_BATCHES = NUM_BATCHES
    ctx.BLOCK_D = BLOCK_D
    ctx.HEAD_DIM = HEAD_DIM
    ctx.NUM_HEADS = NUM_HEADS

    ## num_blocks # of blocks to be processed
    ## BLOCK_PER_BATCH # of blocks per batch
    ## BLOCK_PER_PROGRAM # of blocks per pid (148 SMs on B200)

    pass


@torch.library.register_fake("ads_mkl::tlx_block_attention")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    offsets: torch.Tensor,
    cpu_offset: typing.Optional[torch.Tensor],
    include_projection_rotary: bool = False,
    x: typing.Optional[torch.Tensor] = None,
    w: typing.Optional[torch.Tensor] = None,
    sinu_pos: typing.Optional[torch.Tensor] = None,
    qhead: typing.Optional[int] = None,
    kvhead: typing.Optional[int] = None,
    max_seq_len: typing.Optional[int] = 0,
    bias: typing.Optional[torch.Tensor] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    device = torch.device("cuda:0")  # Use your target device index
    NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count

    if (
        include_projection_rotary
        and x is not None
        and qhead is not None
        and kvhead is not None
    ):
        M = x.shape[0]
        head_dim = x.shape[1] // qhead
        out = torch.empty((M, qhead, head_dim), dtype=x.dtype, device=x.device)
        q = torch.empty((M, qhead, head_dim), dtype=x.dtype, device=x.device)
        k = torch.empty((M, kvhead, head_dim), dtype=x.dtype, device=x.device)
        v = torch.empty((M, kvhead, head_dim), dtype=x.dtype, device=x.device)
        return (
            out,
            torch.zeros_like(offsets),
            torch.empty(NUM_SMS + 1, dtype=torch.int64, device=device),
            NUM_SMS,
            q,
            k,
            v,
        )
    else:
        empty_q = None
        empty_k = None
        empty_v = None
        return (
            torch.zeros_like(query),
            torch.zeros_like(offsets),
            torch.empty(NUM_SMS + 1, dtype=torch.int64, device=device),
            NUM_SMS,
            empty_q,
            empty_k,
            empty_v,
        )


@register_flop_formula(torch.ops.ads_mkl.tlx_block_attention, get_raw=True)
def jfa_forward_flop(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    offsets: torch.Tensor,
    cpu_offset: typing.Optional[torch.Tensor],
    include_projection_rotary: bool = False,
    x: typing.Optional[torch.Tensor] = None,
    w: typing.Optional[torch.Tensor] = None,
    sinu_pos: typing.Optional[torch.Tensor] = None,
    qhead: typing.Optional[int] = None,
    kvhead: typing.Optional[int] = None,
    max_seq_len: typing.Optional[int] = 0,
    bias: typing.Optional[torch.Tensor] = None,
    *args: typing.Any,
    **kwargs: typing.Any,
) -> int:
    """Count flops for block attention forward.

    When fused=True, we also count the projection GEMM flops from
    _tlx_fused_projection_rotary since nested custom ops don't trigger
    separate flop counting via FlopCounterMode dispatch.
    """
    # When fused, q/k/v are computed internally - use placeholder shapes for counting
    projection_flops = 0
    if (
        include_projection_rotary
        and x is not None
        and qhead is not None
        and kvhead is not None
    ):
        M = x.shape[0]
        head_dim = x.shape[1] // qhead
        query_for_shapes = torch.empty(
            (M, qhead, head_dim), device=x.device, dtype=x.dtype
        )
        key_for_shapes = torch.empty(
            (M, kvhead, head_dim), device=x.device, dtype=x.dtype
        )
        value_for_shapes = torch.empty(
            (M, kvhead, head_dim), device=x.device, dtype=x.dtype
        )
        # Add projection GEMM flops: x @ w -> 2 * M * K * N
        if w is not None:
            m, k = x.shape
            k2, n = w.shape
            projection_flops = 2 * m * k * n
    else:
        query_for_shapes = query
        key_for_shapes = key
        value_for_shapes = value

    fn = (
        _unpack_nested_shapes_meta
        if query_for_shapes.is_meta
        else _unpack_flash_attention_nested_shapes
    )
    maxlen = 1 if query_for_shapes.is_meta else int(offsets.max().item())
    sizes = fn(
        query=query_for_shapes,
        key=key_for_shapes,
        value=value_for_shapes,
        cum_seq_q=offsets,
        cum_seq_k=offsets,
        max_q=maxlen,
        max_k=maxlen,
    )

    sizes = (
        (
            query_shape,
            (_b2, _h2, 64, _d2),
            (_b3, _h3, 64, d_v),
            grad_out_shape,
        )
        for (
            query_shape,
            (_b2, _h2, s_k, _d2),
            (_b3, _h3, _s3, d_v),
            grad_out_shape,
        ) in sizes
    )
    attention_flops = sum(
        sdpa_flop_count(query_shape, key_shape, value_shape)
        for query_shape, key_shape, value_shape, _ in sizes
    )
    return projection_flops + attention_flops


##Backward
@triton.jit
def _allocate_buffers_bwd(
    BLOCK_D,
    HEAD_DIM,
    desc_q,
    desc_k,
    desc_v,
    SMEM_BUFFER,
    TMEM_BUFFER,
    desc_grad,
):
    # Descriptors are now passed from outside the kernel

    q_tile = tlx.local_alloc(
        (BLOCK_D, HEAD_DIM), desc_q.dtype, tl.constexpr(SMEM_BUFFER)
    )
    k_tile = tlx.local_alloc(
        (BLOCK_D, HEAD_DIM), desc_q.dtype, tl.constexpr(SMEM_BUFFER)
    )
    v_tile = tlx.local_alloc(
        (BLOCK_D, HEAD_DIM), desc_q.dtype, tl.constexpr(SMEM_BUFFER)
    )
    p_tile = tlx.local_alloc(
        (BLOCK_D, BLOCK_D), desc_q.dtype, tl.constexpr(SMEM_BUFFER)
    )
    p_tile_fp32 = tlx.local_alloc((BLOCK_D, BLOCK_D), tl.float32, tl.constexpr(1))
    DS_tile = tlx.local_alloc(
        (BLOCK_D, BLOCK_D), desc_q.dtype, tl.constexpr(SMEM_BUFFER)
    )
    grad_tile = tlx.local_alloc(
        (BLOCK_D, HEAD_DIM), desc_q.dtype, tl.constexpr(SMEM_BUFFER)
    )
    TMEMqk = tlx.local_alloc(
        (BLOCK_D, BLOCK_D), tl.float32, tl.constexpr(TMEM_BUFFER), tlx.storage_kind.tmem
    )

    TMEMdv = tlx.local_alloc(
        (BLOCK_D, HEAD_DIM),
        tl.float32,
        tl.constexpr(TMEM_BUFFER),
        tlx.storage_kind.tmem,
    )

    TMEMdp = tlx.local_alloc(
        (BLOCK_D, BLOCK_D),
        tl.float32,
        tl.constexpr(TMEM_BUFFER),
        tlx.storage_kind.tmem,
    )

    TMEMdq = tlx.local_alloc(
        (BLOCK_D, HEAD_DIM),
        tl.float32,
        tl.constexpr(TMEM_BUFFER),
        tlx.storage_kind.tmem,
    )

    TMEMdk = tlx.local_alloc(
        (BLOCK_D, HEAD_DIM),
        tl.float32,
        tl.constexpr(TMEM_BUFFER),
        tlx.storage_kind.tmem,
    )

    return (
        q_tile,
        k_tile,
        v_tile,
        p_tile,
        p_tile_fp32,
        DS_tile,
        grad_tile,
        TMEMqk,
        TMEMdv,
        TMEMdq,
        TMEMdk,
        TMEMdp,
        desc_q,
        desc_k,
        desc_v,
        desc_grad,
    )


@triton.jit
def _block_attention_backward(  # noqa: C901
    grad,
    grad_q,
    grad_k,
    grad_v,
    query,
    key,
    value,
    offsets,
    BLOCK_PER_PROGRAM,
    BLOCK_PER_BATCH,
    stride_seq,
    stride_head,
    stride_dim,
    desc_q,
    desc_k,
    desc_v,
    desc_grad_q,
    desc_grad_k,
    desc_grad_v,
    desc_grad,
    TOTAL_LEN_Q,
    sm_scale,
    # Optional: for fused rotary backward with contiguous output
    sinu_pos,  # [M, HEAD_DIM] sin in first half, cos in second half
    desc_sinu_pos,  # tensor descriptor for sinu_pos
    grad_out_contiguous,  # [M, N] contiguous output buffer (when FUSED_ROTARY=True)
    desc_grad_out,  # tensor descriptor for grad_out_contiguous
    QHEAD: tl.constexpr,  # number of query heads
    KVHEAD: tl.constexpr,  # number of kv heads
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_BATCHES: tl.constexpr,
    SMEM_BUFFER: tl.constexpr,
    TMEM_BUFFER: tl.constexpr,
    FUSED_ROTARY: tl.constexpr = False,  # Flag to enable fused rotary backward
):
    LOG2_E = 1.4426950408889634  # = 1 / ln(2)
    tid = tl.program_id(0)
    start = tl.load(BLOCK_PER_PROGRAM + tid).to(tl.int32)
    end = tl.load(BLOCK_PER_PROGRAM + tid + 1).to(tl.int32)
    iters = end - start

    TOTAL_PER_HEAD = tl.load(BLOCK_PER_BATCH + NUM_BATCHES).to(tl.int32)
    global_hidx = start // TOTAL_PER_HEAD
    start = start % TOTAL_PER_HEAD
    offIDX, left = binary_search_le(BLOCK_PER_BATCH, start, NUM_BATCHES)
    batch_start = tl.load(offsets + offIDX).to(tl.int32)

    (
        q_tile,
        k_tile,
        v_tile,
        p_tile,
        p_tile_fp32,
        DS_tile,
        grad_tile,
        TMEMqk,
        TMEMdv,
        TMEMdq,
        TMEMdk,
        TMEMdp,
        desc_q,
        desc_k,
        desc_v,
        desc_grad,
    ) = _allocate_buffers_bwd(
        BLOCK_D,
        HEAD_DIM,
        desc_q,
        desc_k,
        desc_v,
        SMEM_BUFFER,
        TMEM_BUFFER,
        desc_grad,
    )

    k_SMEM_free = tlx.alloc_barriers(SMEM_BUFFER, 2)  # type: ignore[arg-type]
    k_SMEM_full = tlx.alloc_barriers(SMEM_BUFFER, 1)  # type: ignore[arg-type]

    q_SMEM_free = tlx.alloc_barriers(SMEM_BUFFER, 2)  # type: ignore[arg-type]
    q_SMEM_full = tlx.alloc_barriers(SMEM_BUFFER, 1)  # type: ignore[arg-type]

    v_SMEM_free = tlx.alloc_barriers(SMEM_BUFFER, 1)  # type: ignore[arg-type]
    v_SMEM_full = tlx.alloc_barriers(SMEM_BUFFER, 1)  # type: ignore[arg-type]

    qk_TMEM_free = tlx.alloc_barriers(TMEM_BUFFER, 1)  # type: ignore[arg-type]
    qk_TMEM_full = tlx.alloc_barriers(TMEM_BUFFER, 1)  # type: ignore[arg-type]

    grad_SMEM_free = tlx.alloc_barriers(SMEM_BUFFER, 2)  # type: ignore[arg-type]
    grad_SMEM_full = tlx.alloc_barriers(SMEM_BUFFER, 1)  # type: ignore[arg-type]

    p_SMEM_free = tlx.alloc_barriers(SMEM_BUFFER, 2)  # type: ignore[arg-type]
    p_SMEM_full = tlx.alloc_barriers(SMEM_BUFFER, 1)  # type: ignore[arg-type]

    p_fp32_SMEM_free = tlx.alloc_barriers(1, 1)  # type: ignore[arg-type]
    p_fp32_SMEM_full = tlx.alloc_barriers(1, 1)  # type: ignore[arg-type]

    dS_SMEM_free = tlx.alloc_barriers(SMEM_BUFFER, 1)  # type: ignore[arg-type]
    dS_SMEM_full = tlx.alloc_barriers(SMEM_BUFFER, 1)  # type: ignore[arg-type]

    dv_TMEM_free = tlx.alloc_barriers(TMEM_BUFFER, 1)  # type: ignore[arg-type]
    dv_TMEM_full = tlx.alloc_barriers(TMEM_BUFFER, 1)  # type: ignore[arg-type]

    dp_TMEM_free = tlx.alloc_barriers(TMEM_BUFFER, 1)  # type: ignore[arg-type]
    dp_TMEM_full = tlx.alloc_barriers(TMEM_BUFFER, 1)  # type: ignore[arg-type]

    dq_TMEM_free = tlx.alloc_barriers(TMEM_BUFFER, 1)  # type: ignore[arg-type]
    dq_TMEM_full = tlx.alloc_barriers(TMEM_BUFFER, 1)  # type: ignore[arg-type]
    dk_TMEM_free = tlx.alloc_barriers(TMEM_BUFFER, 1)  # type: ignore[arg-type]
    dk_TMEM_full = tlx.alloc_barriers(TMEM_BUFFER, 1)  # type: ignore[arg-type]

    # Allocate sinu_pos tile and barriers only if fused rotary is enabled
    if FUSED_ROTARY:
        HALF_DIM: tl.constexpr = HEAD_DIM // 2
        # Use single-buffered sinu staging to reduce SMEM pressure in fused-rotary bwd.
        sinu_tile = tlx.local_alloc((BLOCK_D, HEAD_DIM), desc_q.dtype, tl.constexpr(1))
        sinu_SMEM_free = tlx.alloc_barriers(1, 1)  # type: ignore[arg-type]
        sinu_SMEM_full = tlx.alloc_barriers(1, 1)  # type: ignore[arg-type]

    with tlx.async_tasks():
        # Load
        with tlx.async_task("default", num_warps=1, registers=48):
            q_current = ((start - left) * BLOCK_D).to(tl.int32) + batch_start
            offset_idx = offIDX
            Z_next = tl.load(offsets + offset_idx + 1).to(tl.int32)
            hidx = global_hidx

            for i in range(iters):
                buff_idx = i % SMEM_BUFFER
                buff_phase = i // SMEM_BUFFER

                ##q
                tlx.barrier_wait(q_SMEM_free[buff_idx], (buff_phase & 1) ^ 1)
                tlx.barrier_expect_bytes(
                    q_SMEM_full[buff_idx], 2 * BLOCK_D * HEAD_DIM
                )  # BF16 2bytes
                tlx.async_descriptor_load(
                    desc_q,
                    q_tile[buff_idx],
                    [q_current, hidx * HEAD_DIM],
                    q_SMEM_full[buff_idx],
                )
                ##k
                tlx.barrier_wait(k_SMEM_free[buff_idx], (buff_phase & 1) ^ 1)
                tlx.barrier_expect_bytes(
                    k_SMEM_full[buff_idx], 2 * BLOCK_D * HEAD_DIM
                )  # BF16 2bytes
                tlx.async_descriptor_load(
                    desc_k,
                    k_tile[buff_idx],
                    [q_current, hidx * HEAD_DIM],
                    k_SMEM_full[buff_idx],
                )

                ## do
                tlx.barrier_wait(grad_SMEM_free[buff_idx], (buff_phase & 1) ^ 1)
                tlx.barrier_expect_bytes(
                    grad_SMEM_full[buff_idx], 2 * BLOCK_D * HEAD_DIM
                )  # BF16 2bytes
                tlx.async_descriptor_load(
                    desc_grad,
                    grad_tile[buff_idx],
                    [q_current, hidx * HEAD_DIM],
                    grad_SMEM_full[buff_idx],
                )

                ## V
                tlx.barrier_wait(v_SMEM_free[buff_idx], (buff_phase & 1) ^ 1)
                tlx.barrier_expect_bytes(
                    v_SMEM_full[buff_idx], 2 * BLOCK_D * HEAD_DIM
                )  # BF16 2bytes
                tlx.async_descriptor_load(
                    desc_v,
                    v_tile[buff_idx],
                    [q_current, hidx * HEAD_DIM],
                    v_SMEM_full[buff_idx],
                )

                ## sinu_pos (for fused rotary backward) - load [BLOCK_D, HEAD_DIM]
                if FUSED_ROTARY:
                    sinu_buff_idx = 0
                    sinu_buff_phase = i
                    tlx.barrier_wait(
                        sinu_SMEM_free[sinu_buff_idx], (sinu_buff_phase & 1) ^ 1
                    )
                    tlx.barrier_expect_bytes(
                        sinu_SMEM_full[sinu_buff_idx], 2 * BLOCK_D * HEAD_DIM
                    )  # BF16 2bytes
                    tlx.async_descriptor_load(
                        desc_sinu_pos,
                        sinu_tile[sinu_buff_idx],
                        [q_current, 0],  # sinu_pos has shape [M, HEAD_DIM]
                        sinu_SMEM_full[sinu_buff_idx],
                    )

                q_current += BLOCK_D
                if q_current >= Z_next:
                    q_current = Z_next % TOTAL_LEN_Q
                    offset_idx += 1
                if offset_idx == NUM_BATCHES:
                    offset_idx = 0
                    hidx += 1
                Z_next = tl.load(offsets + offset_idx + 1).to(tl.int32)

        # MMA
        with tlx.async_task(num_warps=1, registers=48):
            q_current = ((start - left) * BLOCK_D).to(tl.int32) + batch_start
            offset_idx = offIDX
            Z_next = tl.load(offsets + offset_idx + 1).to(tl.int32)
            hidx = global_hidx

            for i in range(iters):
                buff_idx = i % SMEM_BUFFER
                buff_phase = i // SMEM_BUFFER

                TMEM_PHASE = i // TMEM_BUFFER
                tmem_idx = i % TMEM_BUFFER

                tlx.barrier_wait(k_SMEM_full[buff_idx], (buff_phase & 1))
                tlx.barrier_wait(q_SMEM_full[buff_idx], (buff_phase & 1))
                k_tile_T = tlx.local_trans(k_tile[buff_idx])

                tlx.barrier_wait(qk_TMEM_free[tmem_idx], (TMEM_PHASE & 1) ^ 1)
                tlx.async_dot(
                    q_tile[buff_idx],
                    k_tile_T,
                    TMEMqk[tmem_idx],
                    use_acc=False,
                    mBarriers=[
                        k_SMEM_free[buff_idx],
                        q_SMEM_free[buff_idx],
                        qk_TMEM_full[tmem_idx],
                    ],
                )

                q_current += BLOCK_D
                if q_current >= Z_next:
                    q_current = Z_next % TOTAL_LEN_Q
                    offset_idx += 1
                if offset_idx == NUM_BATCHES:
                    offset_idx = 0
                    hidx += 1
                Z_next = tl.load(offsets + offset_idx + 1).to(tl.int32)

        # Softmax load P
        with tlx.async_task(num_warps=4, registers=120):
            q_current = ((start - left) * BLOCK_D).to(tl.int32) + batch_start
            Z_next = tl.load(offsets + offIDX + 1).to(tl.int32)
            offset_idx = offIDX
            hidx = global_hidx

            for i in range(iters):
                buff_idx = i % SMEM_BUFFER
                buff_phase = i // SMEM_BUFFER
                TMEM_PHASE = i // TMEM_BUFFER
                tmem_idx = i % TMEM_BUFFER

                tlx.barrier_wait(qk_TMEM_full[tmem_idx], (TMEM_PHASE & 1))
                result = tlx.local_load(TMEMqk[tmem_idx])
                tlx.barrier_arrive(qk_TMEM_free[tmem_idx])

                # ===== Mask =====
                if Z_next < (q_current + BLOCK_D):
                    block_size = Z_next - q_current
                    row_indices = tl.arange(0, BLOCK_D)
                    col_indices = tl.arange(0, BLOCK_D)
                    valid_mask = (row_indices[:, None] < block_size) & (
                        col_indices[None, :] < block_size
                    )
                    result = tl.where(valid_mask, result, -1.0e9)

                ## Load P
                # Scale factor for log2-based exp
                result = result * (sm_scale * LOG2_E)
                max_val = tl.max(result, axis=1)[:, None]
                result = result - max_val
                result = tl.math.exp2(result)  # Use exp2 instead of exp (faster on GPU)
                sum_val = tl.sum(result, axis=1)[:, None]
                P = result
                P = result / sum_val  ## Softmax

                # Zero out invalid rows/cols in P for partial blocks
                if Z_next < (q_current + BLOCK_D):
                    block_size = Z_next - q_current
                    row_indices = tl.arange(0, BLOCK_D)
                    col_indices = tl.arange(0, BLOCK_D)
                    valid_mask = (row_indices[:, None] < block_size) & (
                        col_indices[None, :] < block_size
                    )
                    P = tl.where(valid_mask, P, 0.0)

                P_bf16 = P.to(tlx.dtype_of(desc_v))
                tlx.fence_async_shared()

                tlx.barrier_wait(p_SMEM_free[buff_idx], (buff_phase & 1) ^ 1)
                p_fp32_buff_idx = 0
                p_fp32_buff_phase = i

                tlx.barrier_wait(
                    p_fp32_SMEM_free[p_fp32_buff_idx], (p_fp32_buff_phase & 1) ^ 1
                )
                tlx.local_store(p_tile[buff_idx], P_bf16)
                tlx.local_store(p_tile_fp32[p_fp32_buff_idx], P)
                tlx.fence_async_shared()
                tlx.barrier_arrive(p_SMEM_full[buff_idx])
                tlx.barrier_arrive(p_fp32_SMEM_full[p_fp32_buff_idx])

                q_current += BLOCK_D
                if q_current >= Z_next:
                    q_current = Z_next % TOTAL_LEN_Q
                    offset_idx += 1
                if offset_idx == NUM_BATCHES:
                    offset_idx = 0
                    hidx += 1
                Z_next = tl.load(offsets + offset_idx + 1).to(tl.int32)

        # DV
        with tlx.async_task(num_warps=1, registers=48):
            q_current = ((start - left) * BLOCK_D).to(tl.int32) + batch_start
            Z_next = tl.load(offsets + offIDX + 1).to(tl.int32)
            offset_idx = offIDX
            hidx = global_hidx

            for i in range(iters):
                buff_idx = i % SMEM_BUFFER
                buff_phase = i // SMEM_BUFFER
                TMEM_PHASE = i // TMEM_BUFFER
                tmem_idx = i % TMEM_BUFFER

                tlx.barrier_wait(grad_SMEM_full[buff_idx], (buff_phase & 1))
                tlx.barrier_wait(p_SMEM_full[buff_idx], (buff_phase & 1))

                # async_dot(P_tile[buf_id]^T, dO_tile[buf_id], dV_tmem[tmem_id])
                pT = tlx.local_trans(p_tile[buff_idx])  # Goes through TMA
                tlx.fence_async_shared()

                tlx.barrier_wait(dv_TMEM_free[tmem_idx], (TMEM_PHASE & 1) ^ 1)
                tlx.async_dot(
                    pT,
                    grad_tile[buff_idx],
                    TMEMdv[tmem_idx],
                    use_acc=False,
                    mBarriers=[
                        p_SMEM_free[buff_idx],
                        grad_SMEM_free[buff_idx],
                        dv_TMEM_full[tmem_idx],
                    ],
                )

                q_current += BLOCK_D
                if q_current >= Z_next:
                    q_current = Z_next % TOTAL_LEN_Q
                    offset_idx += 1
                if offset_idx == NUM_BATCHES:
                    offset_idx = 0
                    hidx += 1
                Z_next = tl.load(offsets + offset_idx + 1).to(tl.int32)

        # Di,DS
        with tlx.async_task(num_warps=4, registers=120):
            q_current = ((start - left) * BLOCK_D).to(tl.int32) + batch_start
            Z_next = tl.load(offsets + offIDX + 1).to(tl.int32)
            offset_idx = offIDX
            hidx = global_hidx

            for i in range(iters):
                buff_idx = i % SMEM_BUFFER
                buff_phase = i // SMEM_BUFFER
                TMEM_PHASE = i // TMEM_BUFFER
                tmem_idx = i % TMEM_BUFFER

                p_fp32_buff_idx = 0
                p_fp32_buff_phase = i

                tlx.barrier_wait(p_SMEM_full[buff_idx], (buff_phase & 1))
                tlx.barrier_wait(v_SMEM_full[buff_idx], (buff_phase & 1))
                tlx.barrier_wait(
                    p_fp32_SMEM_full[p_fp32_buff_idx], p_fp32_buff_phase & 1
                )
                ## Do something
                # dP = dO_tile[buf_id] @ V_tile[buf_id]^T
                tlx.fence_async_shared()
                pLocal = tlx.local_load(p_tile_fp32[p_fp32_buff_idx])
                tlx.fence_async_shared()
                vT = tlx.local_trans(v_tile[buff_idx])
                tlx.fence_async_shared()

                # 115502
                # if tl.program_id(0) == 4 and i == 32:
                #     tl.device_print("vT max:", tl.max(tl.abs(vvT)))
                tlx.barrier_wait(grad_SMEM_full[buff_idx], (buff_phase & 1))
                tlx.barrier_wait(dp_TMEM_free[tmem_idx], (TMEM_PHASE & 1) ^ 1)
                tlx.async_dot(
                    grad_tile[buff_idx],  # BLOCK_D, HEAD_DIM
                    vT,  # HEAD_DIM, BLOCK_D
                    TMEMdp[tmem_idx],  # BLOCK_D, BLOCK_D
                    use_acc=False,
                    mBarriers=[dp_TMEM_full[tmem_idx]],
                )
                tlx.barrier_wait(dp_TMEM_full[tmem_idx], (TMEM_PHASE & 1))

                DP = tlx.local_load(TMEMdp[tmem_idx])
                tlx.tcgen05_commit(dp_TMEM_free[tmem_idx])
                tlx.tcgen05_commit(v_SMEM_free[buff_idx])
                tlx.tcgen05_commit(grad_SMEM_free[buff_idx])
                tlx.tcgen05_commit(p_SMEM_free[buff_idx])
                tlx.tcgen05_commit(p_fp32_SMEM_free[p_fp32_buff_idx])

                Di = tl.sum(pLocal * DP, axis=1)[:, None]
                dS = pLocal * (DP - Di)
                dS = dS * sm_scale
                dS = dS.to(tlx.dtype_of(desc_v))

                tlx.fence_async_shared()
                tlx.barrier_wait(dS_SMEM_free[buff_idx], (buff_phase & 1) ^ 1)
                tlx.local_store(DS_tile[buff_idx], dS)
                tlx.fence_async_shared()
                tlx.barrier_arrive(dS_SMEM_full[buff_idx])
                # Di = rowsum(P ⊙ dP)
                # dS = P ⊙ (dP - Di)

                q_current += BLOCK_D
                if q_current >= Z_next:
                    q_current = Z_next % TOTAL_LEN_Q
                    offset_idx += 1
                if offset_idx == NUM_BATCHES:
                    offset_idx = 0
                    hidx += 1
                Z_next = tl.load(offsets + offset_idx + 1).to(tl.int32)

        ## MMA Dq, Dk
        with tlx.async_task(num_warps=1, registers=48):
            q_current = ((start - left) * BLOCK_D).to(tl.int32) + batch_start
            Z_next = tl.load(offsets + offIDX + 1).to(tl.int32)
            offset_idx = offIDX
            hidx = global_hidx
            for i in range(iters):
                buff_idx = i % SMEM_BUFFER
                buff_phase = i // SMEM_BUFFER
                TMEM_PHASE = i // TMEM_BUFFER
                tmem_idx = i % TMEM_BUFFER

                # dQ = dS @ K
                # dK = dS^T @ Q
                tlx.barrier_wait(dS_SMEM_full[buff_idx], (buff_phase & 1))
                tlx.barrier_wait(k_SMEM_full[buff_idx], (buff_phase & 1))
                tlx.barrier_wait(q_SMEM_full[buff_idx], (buff_phase & 1))
                tlx.barrier_wait(dq_TMEM_free[tmem_idx], (TMEM_PHASE & 1) ^ 1)
                tlx.barrier_wait(dk_TMEM_free[tmem_idx], (TMEM_PHASE & 1) ^ 1)

                # dQ = dS @ K
                tlx.async_dot(
                    DS_tile[buff_idx],
                    k_tile[buff_idx],
                    TMEMdq[tmem_idx],
                    use_acc=False,
                    mBarriers=[
                        dq_TMEM_full[tmem_idx],
                        k_SMEM_free[buff_idx],
                    ],
                )

                # dK = dS^T @ Q
                dsT_view = tlx.local_trans(DS_tile[buff_idx])
                tlx.async_dot(
                    dsT_view,
                    q_tile[buff_idx],
                    TMEMdk[tmem_idx],
                    use_acc=False,
                    mBarriers=[
                        dk_TMEM_full[tmem_idx],
                        q_SMEM_free[buff_idx],
                        dS_SMEM_free[buff_idx],
                    ],
                )

                q_current += BLOCK_D
                if q_current >= Z_next:
                    q_current = Z_next % TOTAL_LEN_Q
                    offset_idx += 1
                if offset_idx == NUM_BATCHES:
                    offset_idx = 0
                    hidx += 1
                Z_next = tl.load(offsets + offset_idx + 1).to(tl.int32)

        # Epilogue
        with tlx.async_task(num_warps=8, registers=200):
            q_current = ((start - left) * BLOCK_D).to(tl.int32) + batch_start
            Z_next = tl.load(offsets + offIDX + 1).to(tl.int32)
            offset_idx = offIDX
            hidx = global_hidx

            for i in tl.range(iters):
                TMEM_PHASE = i // TMEM_BUFFER
                tmem_idx = i % TMEM_BUFFER
                buff_idx = i % SMEM_BUFFER
                buff_phase = i // SMEM_BUFFER

                # Load sin/cos if fused rotary is enabled
                if FUSED_ROTARY:
                    sinu_buff_idx = 0
                    sinu_buff_phase = i
                    tlx.barrier_wait(sinu_SMEM_full[sinu_buff_idx], sinu_buff_phase & 1)
                    sinu_local = tlx.local_load(sinu_tile[sinu_buff_idx])
                    tlx.barrier_arrive(sinu_SMEM_free[sinu_buff_idx])
                    # sinu_pos layout per row: [sin[0:64], cos[64:128]]
                    # Shape after load: [BLOCK_D, HEAD_DIM] = [BLOCK_D, 128]
                    # Keep in bf16 to match baseline rotary kernel

                    # Reshape to [BLOCK_D, 2, HALF_DIM] where:
                    #   dim1=0 -> first half (sin)
                    #   dim1=1 -> second half (cos)
                    sinu_reshaped = sinu_local.reshape(BLOCK_D, 2, HALF_DIM)
                    # Transpose to [BLOCK_D, HALF_DIM, 2] so last dim is 2 for split
                    sinu_trans = sinu_reshaped.trans(0, 2, 1)  # [BLOCK_D, HALF_DIM, 2]
                    # split() splits on last dim (size 2)
                    # sin_local gets [:,:,0], cos_local gets [:,:,1]
                    sin_local, cos_local = sinu_trans.split()
                    # Negate sin for conjugate (backward pass)
                    neg_sin = -sin_local

                ## DV
                tlx.barrier_wait(dv_TMEM_full[tmem_idx], TMEM_PHASE & 1)

                dvLocal = tlx.local_load(TMEMdv[tmem_idx])
                tlx.barrier_arrive(dv_TMEM_free[tmem_idx])

                # Apply rotary conjugate to dv if fused
                if FUSED_ROTARY:
                    dv0, dv1 = dvLocal.reshape(BLOCK_D, HALF_DIM, 2).split()
                    dvLocal = tl.interleave(
                        dv0 * cos_local - dv1 * neg_sin, dv1 * cos_local + dv0 * neg_sin
                    )
                    dv_col_offset = (QHEAD + KVHEAD) * HEAD_DIM + hidx * HEAD_DIM
                    # Store bf16 to contiguous buffer
                    dvLocal = dvLocal.to(tlx.dtype_of(desc_grad_out))
                    if Z_next >= (q_current + BLOCK_D):
                        desc_grad_out.store([q_current, dv_col_offset], dvLocal)
                    else:
                        block_size = Z_next - q_current
                        offs_m = (q_current + tl.arange(0, BLOCK_D)).to(tl.int64)
                        offs_d = (dv_col_offset + tl.arange(0, HEAD_DIM)).to(tl.int64)
                        mask = tl.arange(0, BLOCK_D)[:, None] < block_size
                        N_TOTAL: tl.constexpr = (QHEAD + 2 * KVHEAD) * HEAD_DIM
                        out_ptrs = (
                            grad_out_contiguous
                            + offs_m[:, None] * N_TOTAL
                            + offs_d[None, :]
                        )
                        tl.store(out_ptrs, dvLocal, mask=mask)
                else:
                    dvLocal = dvLocal.to(tlx.dtype_of(desc_grad_v))
                    if Z_next >= (q_current + BLOCK_D):
                        desc_grad_v.store([q_current, hidx * HEAD_DIM], dvLocal)
                    else:
                        block_size = Z_next - q_current
                        offs_m = (q_current + tl.arange(0, BLOCK_D)).to(tl.int64)
                        offs_d = (hidx * HEAD_DIM + tl.arange(0, HEAD_DIM)).to(tl.int64)
                        mask = tl.arange(0, BLOCK_D)[:, None] < block_size
                        out_ptrs = (
                            grad_v
                            + offs_m[:, None] * stride_seq
                            + offs_d[None, :] * stride_dim
                        )
                        tl.store(out_ptrs, dvLocal, mask=mask)

                ## DQ
                tlx.barrier_wait(dq_TMEM_full[tmem_idx], TMEM_PHASE & 1)
                dqLocal = tlx.local_load(TMEMdq[tmem_idx])
                tlx.barrier_arrive(dq_TMEM_free[tmem_idx])

                # Apply rotary conjugate to dq if fused
                if FUSED_ROTARY:
                    dq0, dq1 = dqLocal.reshape(BLOCK_D, HALF_DIM, 2).split()
                    dqLocal = tl.interleave(
                        dq0 * cos_local - dq1 * neg_sin, dq1 * cos_local + dq0 * neg_sin
                    )
                    dq_col_offset = hidx * HEAD_DIM
                    dqLocal = dqLocal.to(tlx.dtype_of(desc_grad_out))
                    if Z_next >= (q_current + BLOCK_D):
                        desc_grad_out.store([q_current, dq_col_offset], dqLocal)
                    else:
                        block_size = Z_next - q_current
                        offs_m = (q_current + tl.arange(0, BLOCK_D)).to(tl.int64)
                        offs_d = (dq_col_offset + tl.arange(0, HEAD_DIM)).to(tl.int64)
                        mask = tl.arange(0, BLOCK_D)[:, None] < block_size
                        N_TOTAL: tl.constexpr = (QHEAD + 2 * KVHEAD) * HEAD_DIM
                        out_ptrs = (
                            grad_out_contiguous
                            + offs_m[:, None] * N_TOTAL
                            + offs_d[None, :]
                        )
                        tl.store(out_ptrs, dqLocal, mask=mask)
                else:
                    dqLocal = dqLocal.to(tlx.dtype_of(desc_grad_q))
                    if Z_next >= (q_current + BLOCK_D):
                        desc_grad_q.store([q_current, hidx * HEAD_DIM], dqLocal)
                    else:
                        block_size = Z_next - q_current
                        offs_m = (q_current + tl.arange(0, BLOCK_D)).to(tl.int64)
                        offs_d = (hidx * HEAD_DIM + tl.arange(0, HEAD_DIM)).to(tl.int64)
                        mask = tl.arange(0, BLOCK_D)[:, None] < block_size
                        out_ptrs = (
                            grad_q
                            + offs_m[:, None] * stride_seq
                            + offs_d[None, :] * stride_dim
                        )
                        tl.store(out_ptrs, dqLocal, mask=mask)

                ## DK
                tlx.barrier_wait(dk_TMEM_full[tmem_idx], TMEM_PHASE & 1)
                dkLocal = tlx.local_load(TMEMdk[tmem_idx])
                tlx.barrier_arrive(dk_TMEM_free[tmem_idx])

                # Apply rotary conjugate to dk if fused
                if FUSED_ROTARY:
                    dk0, dk1 = dkLocal.reshape(BLOCK_D, HALF_DIM, 2).split()
                    dkLocal = tl.interleave(
                        dk0 * cos_local - dk1 * neg_sin, dk1 * cos_local + dk0 * neg_sin
                    )
                    dk_col_offset = QHEAD * HEAD_DIM + hidx * HEAD_DIM
                    dkLocal = dkLocal.to(tlx.dtype_of(desc_grad_out))
                    if Z_next >= (q_current + BLOCK_D):
                        desc_grad_out.store([q_current, dk_col_offset], dkLocal)
                    else:
                        block_size = Z_next - q_current
                        offs_m = (q_current + tl.arange(0, BLOCK_D)).to(tl.int64)
                        offs_d = (dk_col_offset + tl.arange(0, HEAD_DIM)).to(tl.int64)
                        mask = tl.arange(0, BLOCK_D)[:, None] < block_size
                        N_TOTAL: tl.constexpr = (QHEAD + 2 * KVHEAD) * HEAD_DIM
                        out_ptrs = (
                            grad_out_contiguous
                            + offs_m[:, None] * N_TOTAL
                            + offs_d[None, :]
                        )
                        tl.store(out_ptrs, dkLocal, mask=mask)
                else:
                    dkLocal = dkLocal.to(tlx.dtype_of(desc_grad_k))
                    if Z_next >= (q_current + BLOCK_D):
                        desc_grad_k.store([q_current, hidx * HEAD_DIM], dkLocal)
                    else:
                        block_size = Z_next - q_current
                        offs_m = (q_current + tl.arange(0, BLOCK_D)).to(tl.int64)
                        offs_d = (hidx * HEAD_DIM + tl.arange(0, HEAD_DIM)).to(tl.int64)
                        mask = tl.arange(0, BLOCK_D)[:, None] < block_size
                        out_ptrs = (
                            grad_k
                            + offs_m[:, None] * stride_seq
                            + offs_d[None, :] * stride_dim
                        )
                        tl.store(out_ptrs, dkLocal, mask=mask)

                q_current += BLOCK_D
                if q_current >= Z_next:
                    q_current = Z_next % TOTAL_LEN_Q
                    offset_idx += 1
                if offset_idx == NUM_BATCHES:
                    offset_idx = 0
                    hidx += 1
                Z_next = tl.load(offsets + offset_idx + 1).to(tl.int32)

    pass


@torch.library.custom_op("ads_mkl::tlx_block_attention_bwd", mutates_args=())
def block_attention_backward(
    do: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    offsets: torch.Tensor,
    BLOCK_PER_BATCH: torch.Tensor,
    BLOCK_PER_PROGRAM: torch.Tensor,
    num_blocks: int,
    sinu_pos: typing.Optional[torch.Tensor] = None,
    fused_rotary: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    BLOCK_D = 64
    HEAD_DIM = query.shape[2]
    NUM_HEADS = query.shape[1]
    TOTAL_LEN_Q = query.shape[0]
    NUM_BATCHES = offsets.shape[0] - 1
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    device = torch.device("cuda:0")  # Use your target device index
    NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count
    _NUM_PROGRAM = min(NUM_SMS, num_blocks)  # noqa: F841
    stride_seq = query.stride(0)
    stride_head = query.stride(1)
    stride_dim = query.stride(2)

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    kern_kwargs = {
        "BLOCK_D": tl.constexpr(64),
        "HEAD_DIM": tl.constexpr(HEAD_DIM),
        "NUM_BATCHES": tl.constexpr(NUM_BATCHES),
        "SMEM_BUFFER": tl.constexpr(2),
        "TMEM_BUFFER": tl.constexpr(2),
    }

    def expect_contiguous(x: torch.Tensor) -> torch.Tensor:
        if x is not None and not x.is_contiguous():
            return x.contiguous()
        return x

    # Determine qhead and kvhead from query shape
    QHEAD = query.shape[1]
    KVHEAD = key.shape[1]  # Assuming key/value have same number of heads

    # Allocate output tensors
    if fused_rotary:
        # Allocate contiguous output buffer [M, N] where N = qhead*head_dim + 2*kvhead*head_dim
        N_OUT = QHEAD * HEAD_DIM + 2 * KVHEAD * HEAD_DIM
        grad_out_contiguous = expect_contiguous(
            torch.empty(TOTAL_LEN_Q, N_OUT, device=query.device, dtype=query.dtype)
        )
        # dq, dk, dv not used when fused_rotary=True
        dq = None
        dk = None
        dv = None
    else:
        grad_out_contiguous = None
        dq = expect_contiguous(torch.empty_like(query))
        dk = expect_contiguous(torch.empty_like(key))
        dv = expect_contiguous(torch.empty_like(value))

    # Create tensor descriptors outside the kernel
    desc_q = TensorDescriptor(
        query,
        shape=[TOTAL_LEN_Q, HEAD_DIM * NUM_HEADS],
        strides=[HEAD_DIM * NUM_HEADS, 1],
        block_shape=[BLOCK_D, HEAD_DIM],
    )
    desc_k = TensorDescriptor(
        key,
        shape=[TOTAL_LEN_Q, HEAD_DIM * NUM_HEADS],
        strides=[HEAD_DIM * NUM_HEADS, 1],
        block_shape=[BLOCK_D, HEAD_DIM],
    )
    desc_v = TensorDescriptor(
        value,
        shape=[TOTAL_LEN_Q, HEAD_DIM * NUM_HEADS],
        strides=[HEAD_DIM * NUM_HEADS, 1],
        block_shape=[BLOCK_D, HEAD_DIM],
    )

    desc_grad_q = None
    desc_grad_k = None
    desc_grad_v = None
    if not fused_rotary:
        desc_grad_q = TensorDescriptor(
            dq,
            shape=[TOTAL_LEN_Q, HEAD_DIM * NUM_HEADS],
            strides=[HEAD_DIM * NUM_HEADS, 1],
            block_shape=[BLOCK_D, HEAD_DIM],
        )
        desc_grad_k = TensorDescriptor(
            dk,
            shape=[TOTAL_LEN_Q, HEAD_DIM * NUM_HEADS],
            strides=[HEAD_DIM * NUM_HEADS, 1],
            block_shape=[BLOCK_D, HEAD_DIM],
        )
        desc_grad_v = TensorDescriptor(
            dv,
            shape=[TOTAL_LEN_Q, HEAD_DIM * NUM_HEADS],
            strides=[HEAD_DIM * NUM_HEADS, 1],
            block_shape=[BLOCK_D, HEAD_DIM],
        )

    desc_grad = TensorDescriptor(
        do,
        shape=[TOTAL_LEN_Q, HEAD_DIM * NUM_HEADS],
        strides=[HEAD_DIM * NUM_HEADS, 1],
        block_shape=[BLOCK_D, HEAD_DIM],
    )

    # Create sinu_pos tensor descriptor if fused rotary is enabled
    desc_sinu_pos = None
    desc_grad_out = None
    if fused_rotary and sinu_pos is not None:
        desc_sinu_pos = TensorDescriptor(
            sinu_pos,
            shape=[TOTAL_LEN_Q, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[BLOCK_D, HEAD_DIM],
        )
        # Create descriptor for contiguous bf16 output buffer
        N_OUT = QHEAD * HEAD_DIM + 2 * KVHEAD * HEAD_DIM
        desc_grad_out = TensorDescriptor(
            grad_out_contiguous,
            shape=[TOTAL_LEN_Q, N_OUT],
            strides=[N_OUT, 1],
            block_shape=[BLOCK_D, HEAD_DIM],
        )

    _block_attention_backward[(min(NUM_SMS, num_blocks), 1)](
        do,
        dq,
        dk,
        dv,
        query,
        key,
        value,
        offsets,
        BLOCK_PER_PROGRAM,
        BLOCK_PER_BATCH,
        stride_seq,
        stride_head,
        stride_dim,
        desc_q,
        desc_k,
        desc_v,
        desc_grad_q,
        desc_grad_k,
        desc_grad_v,
        desc_grad,
        TOTAL_LEN_Q,
        sm_scale,
        sinu_pos,
        desc_sinu_pos,
        grad_out_contiguous,
        desc_grad_out,
        QHEAD=QHEAD,
        KVHEAD=KVHEAD,
        FUSED_ROTARY=fused_rotary,
        **kern_kwargs,
    )

    if fused_rotary:
        return (grad_out_contiguous, None, None, None, None, None)
    else:
        return (dq, dk, dv, None, None, None)


@torch.library.register_fake("ads_mkl::tlx_block_attention_bwd")
def __(
    do: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    offsets: torch.Tensor,
    BLOCK_PER_BATCH: torch.Tensor,
    BLOCK_PER_PROGRAM: torch.Tensor,
    num_blocks: int,
    sinu_pos: typing.Optional[torch.Tensor] = None,
    fused_rotary: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    QHEAD = query.shape[1]
    KVHEAD = key.shape[1]
    HEAD_DIM = query.shape[2]
    M = query.shape[0]
    N_OUT = QHEAD * HEAD_DIM + 2 * KVHEAD * HEAD_DIM
    if fused_rotary and sinu_pos is not None:
        return (
            torch.zeros(M, N_OUT, device=query.device, dtype=query.dtype),
            None,
            None,
            None,
            None,
            None,
        )
    return (
        torch.zeros_like(query),
        torch.zeros_like(key),
        torch.zeros_like(value),
        None,
        None,
        None,
    )


@register_flop_formula(torch.ops.ads_mkl.tlx_block_attention_bwd, get_raw=True)
def jfa_backward_flop(
    do: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    offsets: torch.Tensor,
    BLOCK_PER_BATCH: torch.Tensor,
    BLOCK_PER_PROGRAM: torch.Tensor,
    num_blocks: int,
    sinu_pos: typing.Optional[torch.Tensor] = None,
    fused_rotary: bool = False,
    *args: typing.Any,
    **kwargs: typing.Any,
) -> int:
    """Count flops for block attention backward.

    Includes SDPA backward flops. When fused_rotary=True, also includes
    projection backward GEMMs (dx = dz @ w and dw = x.T @ dz).
    """
    fn = (
        _unpack_nested_shapes_meta
        if query.is_meta
        else _unpack_flash_attention_nested_shapes
    )
    maxlen = 1 if query.is_meta else int(offsets.max().item())
    shapes = fn(
        query=query,
        key=key,
        value=value,
        grad_out=do,
        cum_seq_q=offsets,
        cum_seq_k=offsets,
        max_q=maxlen,
        max_k=maxlen,
    )

    shapes = (
        (
            query_shape,
            (_b2, _h2, 64, _d2),
            (_b3, _h3, 64, d_v),
            grad_out_shape,
        )
        for (
            query_shape,
            (_b2, _h2, s_k, _d2),
            (_b3, _h3, _s3, d_v),
            grad_out_shape,
        ) in shapes
    )
    sdpa_flops = sum(
        sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape)
        for query_shape, key_shape, value_shape, grad_out_shape in shapes
    )

    return sdpa_flops


def tlx_block_attention_backward(ctx, *grad_outputs):
    do = grad_outputs[0]
    # print(do.shape)
    num_blocks = ctx.num_blocks
    include_projection_rotary = ctx.include_projection_rotary

    if include_projection_rotary:
        # bf16 fused mode: saved (x, w, sinu_pos, offsets, q, k, v, BLOCK_PER_BATCH, BLOCK_PER_PROGRAM)
        (
            x,
            w,
            sinu_pos,
            offsets,
            query,
            key,
            value,
            BLOCK_PER_BATCH,
            BLOCK_PER_PROGRAM,
        ) = ctx.saved_tensors

        # Step 1: Block attention backward with fused rotary
        dz, _, _, _, _, _ = block_attention_backward(
            do,
            query,
            key,
            value,
            offsets,
            BLOCK_PER_BATCH,
            BLOCK_PER_PROGRAM,
            num_blocks,
            sinu_pos=sinu_pos,
            fused_rotary=True,
        )

        # Step 2: Projection backward (gemms)
        dx = torch.matmul(dz, w.t())
        dw = torch.matmul(x.t(), dz)
        dbias = dz.sum(dim=0) if ctx.bias is not None else None

        # Return gradients matching input order:
        # (query, key, value, offsets, cpu_offset, fused, x, w, sinu_pos, qhead, kvhead, max_seq_len, bias)
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            dx,
            dw,
            None,
            None,
            None,
            None,
            dbias,
        )
    else:
        # Non-fused mode: saved (q, k, v, offsets, BLOCK_PER_BATCH, BLOCK_PER_PROGRAM)
        query, key, value, offsets, BLOCK_PER_BATCH, BLOCK_PER_PROGRAM = (
            ctx.saved_tensors
        )

        dq, dk, dv, _, _, _ = block_attention_backward(
            do,
            query,
            key,
            value,
            offsets,
            BLOCK_PER_BATCH,
            BLOCK_PER_PROGRAM,
            num_blocks,
        )
        # Return gradients matching input order (13 elements):
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


if not isinstance(
    tlx_block_attention, types.FunctionType
):  # In case of duplicate registration, `@custom_triton_op` returns the base function
    tlx_block_attention.register_autograd(
        tlx_block_attention_backward,
        setup_context=block_attention_setup_context,
    )


# =============================================================================
# Public API for fused projection + rotary + block attention
# =============================================================================


@torch.fx.wrap
def fused_projection_rotary_blockattn(
    x: torch.Tensor,
    w: torch.Tensor,
    sinu_pos: torch.Tensor,
    offsets: torch.Tensor,
    cpu_offset: typing.Optional[torch.Tensor],
    qhead: int,
    kvhead: int,
    max_seq_len: typing.Optional[int] = 0,
    bias: typing.Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Public API for unified fused projection + rotary + block attention.

    This combines:
    - Forward: tlx_fused_projection_rotary + tlx_block_attention
    - Backward: block_attention_backward(fused_rotary=True) + projection gemms

    The backward pass applies rotary conjugate directly in the attention backward
    kernel epilogue, avoiding a separate rotary backward kernel.
    """

    out, _, _, _, _, _, _ = tlx_block_attention(
        query=None,
        key=None,
        value=None,
        offsets=offsets,
        cpu_offset=cpu_offset,
        include_projection_rotary=True,
        x=x,
        w=w,
        sinu_pos=sinu_pos,
        qhead=qhead,
        kvhead=kvhead,
        max_seq_len=max_seq_len,
        bias=bias,
    )
    return out
