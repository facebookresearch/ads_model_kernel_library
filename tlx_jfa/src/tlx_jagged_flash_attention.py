# pyre-ignore-all-errors
import math
import types
from functools import lru_cache
from typing import Generator, Optional, Tuple

import torch
import triton  # @manual=//triton:triton
import triton.language as tl  # @manual=//triton:triton
import triton.language.extra.tlx as tlx  # @manual=//triton:triton
from bwd_1cta import _attn_bwd_ws, _get_autotune_bwd_kernel
from kernel_common import (
    _get_bufidx_phase,
    _join_n,
    _reduce_or,
    _split_n,
    bwd_calculate_num_steps,
    bwd_lookup_offsets,
)
from register_helpers import custom_register_kernel
from tlx_math import _fma_f32x2, _mul_f32x2, _sub_f32x2
from torch.nn import functional as F
from torch.utils.flop_counter import (
    _unpack_flash_attention_nested_shapes,
    register_flop_formula,
    sdpa_backward_flop_count,
    sdpa_flop_count,
)
from triton.runtime.jit import JITFunction
from triton.tools.tensor_descriptor import TensorDescriptor  # @manual=//triton:triton
from utils import should_use_i64_idx

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# A/B switch for the 2-CTA (cluster) backward path. When True, the backward
# launch routes broadcast_q + HEAD_DIM=128 + G=1 + LB-on tiles (sliding window
# included) through the collaborative-MMA `_attn_bwd_ws_2cta` kernel; all other
# cases (and False) fall back to the general 1-CTA `_attn_bwd_ws` in `bwd_1cta`.
# The CLC-vs-persistent choice for the 2-CTA kernel lives in the autotune config
# (`configs_bwd_2cta_tlx`), not here. Flip to False to use only the 1-CTA path.
JFA_BWD_USE_2CTA: bool = True


@lru_cache
def get_num_sms() -> Optional[int]:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties("cuda").multi_processor_count


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    BLOCK_D = nargs["BLOCK_D"]
    if not isinstance(nargs["Q"], TensorDescriptor):
        return
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    nargs["Q"].block_shape = [BLOCK_M_SPLIT, BLOCK_D]
    nargs["K"].block_shape = [BLOCK_N, BLOCK_D]
    nargs["V"].block_shape = [BLOCK_N, BLOCK_D]
    nargs["Out"].block_shape = [BLOCK_M_SPLIT, BLOCK_D]


@lru_cache
def get_cuda_autotune_config_fwd():
    return [
        triton.Config(
            {
                "BLOCK_M": BM,
                "BLOCK_N": BN,
                "NUM_BUFFERS_Q": bq,
                "NUM_BUFFERS_KV": bkv,
                "NUM_BUFFERS_QK": bqk,
                # "NUM_BUFFERS_O": bo,
                "NUM_MMA_GROUPS": mma_g,
                "NUM_MMA_SLICES": mma_s,
                "NUM_REGS_SFM": rsfm,
                "NUM_REGS_MMA": rmma,
                "NUM_REGS_LOAD": rload,
                "NUM_REGS_EPILOG": repi,
                "USE_CLC": True,
                "RESCALE_OPT": ropt,
            },
            num_warps=4,
            num_stages=0,
            pre_hook=_host_descriptor_pre_hook,
        )
        for BM in [256]  # 128 or 256
        for BN in [128]
        for bq in [1]
        for bkv in [3]
        for bqk in [1]  # in tmem
        # for bo in [1]  # in tmem
        for mma_g in [2]
        for mma_s in [2]
        for rsfm in [176]
        for rmma in [24, 32]
        for rload in [24, 32]
        for repi in [24, 32]
        for ropt in [True]
    ]


@triton.jit  # pragma: no cover
def _compute_seq_len(
    tile_idx,
    n_tile_num,
    Q_offsets,
    K_offsets,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    BROADCAST_Q: tl.constexpr,
    ENABLE_LOAD_BALANCING: tl.constexpr,
    BLOCK_M: tl.constexpr,
    valid_tiles_b,
    valid_tiles_m_start,
    valid_tiles_m_end,
    valid_tiles_h,
    sm_offsets,
    cur_sm_offset,
):
    if ENABLE_LOAD_BALANCING:
        off_h = tl.load(valid_tiles_h + cur_sm_offset).to(tl.int64)
        off_z = tl.load(valid_tiles_b + cur_sm_offset).to(tl.int64)
        if not BROADCAST_Q:
            begin_q = tl.load(valid_tiles_m_start + cur_sm_offset).to(tl.int64)
            end_q = tl.load(valid_tiles_m_end + cur_sm_offset).to(tl.int64)
            start_m = 0
        else:
            begin_q = tl.load(Q_offsets)
            end_q = tl.load(Q_offsets + 1)
            m_start_abs = tl.load(valid_tiles_m_start + cur_sm_offset).to(tl.int64)
            start_m = (m_start_abs - begin_q) // BLOCK_M
    else:
        off_hz = tile_idx // n_tile_num
        off_z = off_hz // H
        off_h = off_hz % H
        if not BROADCAST_Q:
            off_q_z = off_z
        else:
            off_q_z = 0
        begin_q = tl.load(Q_offsets + off_q_z)
        end_q = tl.load(Q_offsets + off_q_z + 1)
        start_m = tile_idx % n_tile_num

    qlen = end_q - begin_q
    qlen = tl.minimum(qlen, N_CTX)

    begin_k = tl.load(K_offsets + off_z)
    end_k = tl.load(K_offsets + off_z + 1)
    klen = end_k - begin_k

    return begin_q, end_q, begin_k, end_k, qlen, klen, off_z, off_h, start_m


def compute_balanced_tiles(
    cpu_query_offsets: torch.Tensor,
    cpu_key_offsets: torch.Tensor,
    bs: int,
    BLOCK_M: int,
    BLOCK_N: int,
    H: int,
    broadcast_q: bool,
):
    assert cpu_query_offsets.is_cpu
    assert cpu_key_offsets.is_cpu

    # Determine query start/end for each batch
    if broadcast_q:
        query_starts = torch.full((bs,), cpu_query_offsets[0].item(), dtype=torch.int32)
        query_ends = torch.full((bs,), cpu_query_offsets[1].item(), dtype=torch.int32)
    else:
        query_starts = cpu_query_offsets[:-1]
        query_ends = cpu_query_offsets[1:]

    # Compute tiles per example
    qlens = query_ends - query_starts
    tiles_per_example = (qlens + BLOCK_M - 1) // BLOCK_M  # cdiv

    # Compute workload per example
    klens = cpu_key_offsets[1:] - cpu_key_offsets[:-1]
    workloads = (klens + BLOCK_N - 1) // BLOCK_N  # cdiv

    # Create tensors for all tiles: (b, h, m_start, m_end, workload)
    # For each batch
    batch_indices = torch.arange(bs)
    # For each head
    head_indices = torch.arange(H)

    # Generate all (batch, head, tile_idx) combinations
    # First expand by heads: for each batch, repeat H times
    batch_expanded = torch.repeat_interleave(batch_indices, H * tiles_per_example)
    head_expanded = head_indices.repeat(tiles_per_example.sum())

    # Generate tile indices for each (batch, head) pair
    # For each batch, we need tiles_per_example[b] tiles
    tile_offsets = torch.arange(tiles_per_example.sum())
    cumsum_tiles = tiles_per_example.cumsum(0)
    tile_start_indices = torch.cat([torch.tensor([0]), cumsum_tiles[:-1]])
    tile_local_indices = tile_offsets - torch.repeat_interleave(
        tile_start_indices, tiles_per_example
    )

    # Compute m_start and m_end for each tile
    batch_for_tiles = torch.repeat_interleave(batch_indices, tiles_per_example)
    m_starts = query_starts[batch_for_tiles] + tile_local_indices * BLOCK_M
    m_ends = torch.minimum(m_starts + BLOCK_M, query_ends[batch_for_tiles])
    m_starts_expanded = torch.repeat_interleave(m_starts, H)
    m_ends_expanded = torch.repeat_interleave(m_ends, H)

    # Workload for each tile
    workload_expanded = torch.repeat_interleave(workloads, H * tiles_per_example)

    # Sort by workload in descending order
    sorted_indices = torch.argsort(workload_expanded, descending=True)
    batch_sorted = batch_expanded[sorted_indices]
    head_sorted = head_expanded[sorted_indices]
    m_start_sorted = m_starts_expanded[sorted_indices]
    m_end_sorted = m_ends_expanded[sorted_indices]

    # Dispatch to SMs using zigzag pattern
    num_sm = get_num_sms()
    total_tiles = len(batch_sorted)
    tile_indices = torch.arange(total_tiles)
    cycle = tile_indices // num_sm
    pos_in_cycle = tile_indices % num_sm
    sm_indices = torch.where(cycle % 2 == 0, pos_in_cycle, num_sm - 1 - pos_in_cycle)

    # Sort by SM index to group tiles by SM
    sm_sorted_indices = torch.argsort(sm_indices, stable=True)
    final_batch = batch_sorted[sm_sorted_indices]
    final_head = head_sorted[sm_sorted_indices]
    final_m_start = m_start_sorted[sm_sorted_indices]
    final_m_end = m_end_sorted[sm_sorted_indices]

    # Compute offsets for each SM
    sm_indices_sorted = sm_indices[sm_sorted_indices]
    offsets = torch.cat(
        [
            torch.tensor([0]),
            torch.cumsum(torch.bincount(sm_indices_sorted, minlength=num_sm), dim=0),
        ]
    )

    # Send all scheduling-related metadata to GPU side
    return (
        final_m_start.pin_memory().to(
            device="cuda", dtype=torch.int32, non_blocking=True
        ),
        final_m_end.pin_memory().to(
            device="cuda", dtype=torch.int32, non_blocking=True
        ),
        final_batch.pin_memory().to(
            device="cuda", dtype=torch.int32, non_blocking=True
        ),
        final_head.pin_memory().to(device="cuda", dtype=torch.int32, non_blocking=True),
        offsets.pin_memory().to(device="cuda", dtype=torch.int32, non_blocking=True),
    )


@triton.jit  # pragma: no cover
def _softmax_inner_iter(
    cid,
    accum_cnt_qk,
    m_i,
    l_i,
    qk_scale,
    klen,
    start_n,
    start_m,
    qk_fulls,
    qk_tiles,
    alpha_empties,
    alpha_fulls,
    alpha_tiles,
    p_fulls,
    p_tiles,
    out_dtype,
    BLOCK_M_SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    NUM_MMA_SLICES: tl.constexpr,
    RESCALE_OPT: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    APPLY_MASK: tl.constexpr,
):
    # Per-iteration body of the softmax warpgroup. APPLY_MASK is constexpr so
    # the bulk vs tail loops generate different code: bulk emits no mask `where`
    # at all, tail emits the klen-bound mask.
    _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)
    tlx.barrier_wait(qk_fulls[cid], qk_phase)
    qk = tlx.local_load(qk_tiles[cid])
    if APPLY_MASK or WINDOW_SIZE is not None:
        offs_m = start_m * BLOCK_M + cid * BLOCK_M_SPLIT + tl.arange(0, BLOCK_M_SPLIT)
    if APPLY_MASK:
        offs_n = start_n + tl.arange(0, BLOCK_N)
        masks = offs_n[None, :] < klen
        if WINDOW_SIZE is not None:
            window_mask = tl.abs(offs_m[:, None] - offs_n[None, :]) <= WINDOW_SIZE
            masks &= window_mask
        qk = tl.where(masks, qk, -1.0e9)
    elif WINDOW_SIZE is not None:
        offs_n = start_n + tl.arange(0, BLOCK_N)
        window_mask = tl.abs(offs_m[:, None] - offs_n[None, :]) <= WINDOW_SIZE
        qk = tl.where(window_mask, qk, -1.0e9)

    if RESCALE_OPT:
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
    else:
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)

    if RESCALE_OPT:
        alpha_ = (m_i - m_ij) * qk_scale
        alpha = tl.math.exp2(alpha_)
        rescale_mask = alpha_ >= -8.0
        alpha = tl.where(rescale_mask, 1.0, alpha)
        m_ij = tl.where(rescale_mask, m_i, m_ij)
    else:
        alpha = tl.math.exp2(m_i - m_ij)
    tlx.barrier_wait(alpha_empties[cid], qk_phase ^ 1)
    tlx.local_store(alpha_tiles[cid * BLOCK_N], alpha[:, None])
    tlx.barrier_arrive(alpha_fulls[cid])

    if RESCALE_OPT:
        m_scaled = m_ij * qk_scale
        qk = _fma_f32x2(qk, qk_scale, -m_scaled[:, None])
    else:
        qk = _fma_f32x2(qk, qk_scale, -m_ij[:, None])
    qks = _split_n(qk, NUM_MMA_SLICES)
    ps = ()
    for slice_id in tl.static_range(0, NUM_MMA_SLICES):
        p_bufIdx = cid * NUM_MMA_GROUPS * NUM_MMA_SLICES + NUM_MMA_SLICES + slice_id
        p_i = tl.math.exp2(qks[slice_id])
        tlx.local_store(p_tiles[p_bufIdx], p_i.to(out_dtype))
        tlx.barrier_arrive(p_fulls[slice_id + cid * NUM_MMA_SLICES])
        ps = ps + (p_i,)

    p = _join_n(ps)
    l_ij = tl.sum(p, 1)
    l_i = l_i * alpha + l_ij
    m_i = m_ij
    accum_cnt_qk += 1
    return m_i, l_i, accum_cnt_qk


@lru_cache
def _get_autotune_fwd_kernel(kernel: JITFunction) -> JITFunction:
    return triton.autotune(
        configs=get_cuda_autotune_config_fwd(),
        key=["N_CTX", "HEAD_DIM", "H", "G"],
    )(kernel)


@triton.jit  # pragma: no cover
# Triton TR001: launched through _get_autotune_fwd_kernel.
def _attn_fwd_ws(  # noqa: C901, TR001
    Q,
    Q_offsets,
    K,
    K_offsets,
    V,
    Out,
    M,
    sm_scale,
    stride_qh,
    stride_kh,
    stride_oh,
    stride_mh,
    Z,  # Batch size
    H,  # number of q heads.
    G,  # number of q head in each group. number of k v head will be H//G
    N_CTX,
    total_len_q,
    total_len_kv,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    BLOCK_D: tl.constexpr,  #
    USE_ON_DEVICE_TMA: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_BUFFERS_QK: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    NUM_MMA_SLICES: tl.constexpr,
    BROADCAST_Q: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    ENABLE_LOAD_BALANCING: tl.constexpr,
    NUM_REGS_SFM: tl.constexpr,
    NUM_REGS_MMA: tl.constexpr,
    NUM_REGS_LOAD: tl.constexpr,
    NUM_REGS_EPILOG: tl.constexpr,
    valid_tiles_b,
    valid_tiles_m_start,
    valid_tiles_m_end,
    valid_tiles_h,
    sm_offsets,
    USE_I64_IDX: tl.constexpr,
    USE_CLC: tl.constexpr,
    RESCALE_OPT: tl.constexpr = True,
):
    # Load balancing cannot work with CLC due to SM allocation
    LOAD_BALANCING_VALID: tl.constexpr = ENABLE_LOAD_BALANCING and not USE_CLC
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // 2

    n_tile_num = tl.cdiv(N_CTX, BLOCK_M)
    prog_id = tl.program_id(0)
    if USE_I64_IDX:
        prog_id = prog_id.to(tl.int64)
    num_progs = tl.num_programs(0)
    tile_idx = prog_id
    if LOAD_BALANCING_VALID:
        sm_start = tl.load(sm_offsets + prog_id)
        sm_end = tl.load(sm_offsets + prog_id + 1)

        tiles_per_sm = sm_end - sm_start
    else:
        total_tiles = n_tile_num * Z * H

        tiles_per_sm = total_tiles // num_progs
        if prog_id < total_tiles % num_progs:
            tiles_per_sm += 1

        # unused declarations to make compiler happy
        sm_start = -1
        sm_end = -1

    # on-device TMA
    if USE_ON_DEVICE_TMA:
        desc_q = tl.make_tensor_descriptor(
            Q,
            shape=[total_len_q, HEAD_DIM * H],
            strides=[HEAD_DIM * H, 1],
            block_shape=[BLOCK_M_SPLIT, BLOCK_D],
        )
        desc_k = tl.make_tensor_descriptor(
            K,
            shape=[total_len_kv, HEAD_DIM * H // G],
            strides=[HEAD_DIM * H // G, 1],
            block_shape=[BLOCK_N, BLOCK_D],
        )
        desc_v = tl.make_tensor_descriptor(
            V,
            shape=[total_len_kv, HEAD_DIM * H // G],
            strides=[HEAD_DIM * H // G, 1],
            block_shape=[BLOCK_N, BLOCK_D],
        )
    else:
        desc_q = Q
        desc_k = K
        desc_v = V

    # allocate SMEM buffers and barriers
    q_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_D), tlx.dtype_of(desc_q), NUM_MMA_GROUPS * NUM_BUFFERS_Q
    )
    kv_tiles = tlx.local_alloc((BLOCK_N, BLOCK_D), tlx.dtype_of(desc_k), NUM_BUFFERS_KV)
    o_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_D), tlx.dtype_of(desc_v), NUM_MMA_GROUPS
    )

    q_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_Q)
    q_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_Q)
    kv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    kv_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    o_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    o_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)

    # allocate TMEM buffers and barriers
    qk_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_N), tl.float32, NUM_MMA_GROUPS, tlx.storage_kind.tmem
    )
    # Shared buffer for QK, P and Alpha, l, and m.
    # A single QK buffer is split evenly:
    #   - First half  : stores P
    #   - Second half  : stores Alpha, l, and m
    #     QK : |                              BLK_M/2 * BLOCK_N * fp32                  |
    #     P:                                                |  BLK_M/2 * BLOCK_N * fp16 |
    #  Alpha : |BLK_M/2*1*fp32|
    #     l :                 |BLK_M/2*1*fp32|
    #     m :                                |BLK_M/2*1*fp32|
    p_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_N // NUM_MMA_SLICES),
        tlx.dtype_of(desc_v),
        NUM_MMA_GROUPS * NUM_MMA_SLICES * 2,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )
    alpha_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, 1),
        tl.float32,
        BLOCK_N * NUM_MMA_GROUPS * NUM_BUFFERS_QK,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )
    l_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, 1),
        tl.float32,
        BLOCK_N * NUM_MMA_GROUPS * NUM_BUFFERS_QK,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )
    m_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, 1),
        tl.float32,
        BLOCK_N * NUM_MMA_GROUPS * NUM_BUFFERS_QK,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )

    acc_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_D), tl.float32, NUM_MMA_GROUPS, tlx.storage_kind.tmem
    )

    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    qk_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    p_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_MMA_SLICES)
    acc_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    acc_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)

    alpha_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    alpha_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    l_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)

    if USE_CLC:
        clc_context = tlx.clc_create_context(4 + NUM_MMA_GROUPS)

    with tlx.async_tasks():
        # correction group
        with tlx.async_task("default"):
            accum_cnt = 0
            accum_cnt_temp = 0
            clc_phase_consumer = 0
            clc_phase_producer = 1
            phase = 0
            has_more_tile = True
            i = 0
            while has_more_tile:
                # initialize offsets
                begin_q, end_q, begin_k, end_k, qlen, klen, off_z, off_h, start_m = (
                    _compute_seq_len(
                        tile_idx,
                        n_tile_num,
                        Q_offsets,
                        K_offsets,
                        H,
                        N_CTX,
                        BROADCAST_Q,
                        LOAD_BALANCING_VALID,
                        BLOCK_M,
                        valid_tiles_b,
                        valid_tiles_m_start,
                        valid_tiles_m_end,
                        valid_tiles_h,
                        sm_offsets,
                        sm_start + i,
                    )
                )
                out_offset = off_h.to(tl.int64) * stride_oh
                if start_m * BLOCK_M < qlen:
                    lo, hi = 0, klen
                    if WINDOW_SIZE is not None:
                        lo = max(
                            lo, ((start_m * BLOCK_M - WINDOW_SIZE) // BLOCK_N) * BLOCK_N
                        )
                        hi = min(hi, (start_m + 1) * BLOCK_M + WINDOW_SIZE)
                    for start_n in tl.range(lo, hi, BLOCK_N):
                        start_n = tl.multiple_of(start_n, BLOCK_N)
                        _, phase = _get_bufidx_phase(accum_cnt, 1)
                        for cid in tl.static_range(0, NUM_MMA_GROUPS):
                            # -- update output accumulator --
                            tlx.barrier_wait(alpha_fulls[cid], phase)
                            # Use alpha[0] for cid=0, and alpha[HEAD_DIM] for cid=1
                            alpha_1 = tlx.local_load(alpha_tiles[cid * BLOCK_N])
                            tlx.barrier_arrive(alpha_empties[cid])
                            # Ballot skip: when alpha was forced to 1.0
                            # in the softmax warp (no row needs rescale), skip the
                            # entire TMEM load/mul/store loop.
                            if RESCALE_OPT:
                                pred = alpha_1 < 1.0
                                ballot_result = tlx.vote_ballot_sync(0xFFFFFFFF, pred)
                                should_rescale = ballot_result != 0
                                should_rescale_red = tl.reduce(
                                    should_rescale, axis=0, combine_fn=_reduce_or
                                )
                                should_rescale_scalar = tl.reshape(
                                    should_rescale_red, ()
                                )
                            if not RESCALE_OPT or should_rescale_scalar:
                                for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                                    subslice = tlx.subslice(
                                        acc_tiles[cid],
                                        BLOCK_D * slice_id // NUM_MMA_SLICES,
                                        BLOCK_D // NUM_MMA_SLICES,
                                    )
                                    acc = tlx.local_load(subslice)
                                    # acc = acc * alpha_1
                                    acc = _mul_f32x2(acc, alpha_1)
                                    tlx.local_store(subslice, acc)
                            tlx.barrier_arrive(acc_fulls[cid])
                        accum_cnt += 1

                    _, phase = _get_bufidx_phase(accum_cnt_temp, 1)
                    for cid in tl.static_range(0, NUM_MMA_GROUPS):
                        # epilogue
                        tlx.barrier_wait(l_fulls[cid], phase)
                        # Use l[1]/l[1+HEAD_DIM] and m[2][2 + HEAD_DIM]
                        # to disambigulate from alpha[0]/alpha[HEAD_DIM]
                        l_i_epilogue = tlx.local_load(l_tiles[cid * BLOCK_N + 1])
                        m = tlx.local_load(m_tiles[cid * BLOCK_N + 2])
                        tlx.barrier_arrive(qk_empties[cid])
                        # When RESCALE_OPT is on, m_tiles holds UNSCALED row-max.
                        # The bwd kernel reads logsumexp = (m * sm_scale * log2_e)
                        # + log2(l), so we scale here.
                        if RESCALE_OPT:
                            m = m * sm_scale * 1.44269504
                        m += tl.math.log2(l_i_epilogue)
                        offs_m = (
                            start_m * BLOCK_M
                            + cid * BLOCK_M_SPLIT
                            + tl.arange(0, BLOCK_M_SPLIT)
                        )
                        if BROADCAST_Q:
                            begin_o = qlen * off_z
                            m_ptrs = M + begin_o + offs_m + off_h * stride_mh
                        else:
                            m_ptrs = M + begin_q + offs_m + off_h * stride_mh
                        tl.store(
                            m_ptrs, tl.reshape(m, [BLOCK_M_SPLIT]), mask=offs_m < qlen
                        )

                        tlx.barrier_wait(acc_empties[cid], phase)
                        tlx.barrier_wait(o_empties[cid], phase ^ 1)
                        scale = 1 / l_i_epilogue
                        for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                            subslice = tlx.subslice(
                                acc_tiles[cid],
                                HEAD_DIM * slice_id // NUM_MMA_SLICES,
                                HEAD_DIM // NUM_MMA_SLICES,
                            )
                            acc = tlx.local_load(subslice)
                            acc = _mul_f32x2(acc, scale)
                            acc = acc.to(tlx.dtype_of(Out))
                            subslice_o = tlx.local_slice(
                                o_tiles[cid],
                                [0, HEAD_DIM * slice_id // NUM_MMA_SLICES],
                                [BLOCK_M_SPLIT, HEAD_DIM // NUM_MMA_SLICES],
                            )
                            tlx.local_store(subslice_o, acc)
                        tlx.barrier_arrive(o_fulls[cid])
                    accum_cnt_temp += 1
                i += 1
                if USE_CLC:
                    if USE_I64_IDX:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer).to(
                            tl.int64
                        )
                    else:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx != -1
                else:
                    if not ENABLE_LOAD_BALANCING:
                        tile_idx += num_progs
                    has_more_tile = i < tiles_per_sm

        # softmax groups
        with tlx.async_task(
            num_warps=4, registers=NUM_REGS_SFM, replicate=NUM_MMA_GROUPS
        ):
            accum_cnt_qk = 0
            clc_phase_consumer = 0
            clc_phase_producer = 1
            phase = 0
            has_more_tile = True
            i = 0
            while has_more_tile:
                # initialize offsets
                begin_q, end_q, begin_k, end_k, qlen, klen, off_z, off_h, start_m = (
                    _compute_seq_len(
                        tile_idx,
                        n_tile_num,
                        Q_offsets,
                        K_offsets,
                        H,
                        N_CTX,
                        BROADCAST_Q,
                        LOAD_BALANCING_VALID,
                        BLOCK_M,
                        valid_tiles_b,
                        valid_tiles_m_start,
                        valid_tiles_m_end,
                        valid_tiles_h,
                        sm_offsets,
                        sm_start + i,
                    )
                )
                # initialize pointer to m and l
                m_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) - float("inf")
                l_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) + 1.0
                acc = tl.zeros([BLOCK_M_SPLIT, BLOCK_D], dtype=tl.float32)
                qk_scale = sm_scale
                qk_scale *= 1.44269504  # 1/log(2)

                cid = tlx.async_task_replica_id()
                if start_m * BLOCK_M < qlen:
                    lo, hi = 0, klen
                    if WINDOW_SIZE is not None:
                        lo = max(
                            lo, ((start_m * BLOCK_M - WINDOW_SIZE) // BLOCK_N) * BLOCK_N
                        )
                        hi = min(hi, (start_m + 1) * BLOCK_M + WINDOW_SIZE)
                    # Loop peeling: for the no-window (PMA) case, split the KV
                    # loop into a bulk pass with no klen mask and a 0-or-1
                    # iteration tail with the mask. Removes the per-iteration
                    # `if start_n + BLOCK_N >= klen` branch + dead `tl.where`
                    # from the hot loop. Window mode keeps a single loop because
                    # the window mask must be applied every iteration.
                    if WINDOW_SIZE is None:
                        hi_aligned = (klen // BLOCK_N) * BLOCK_N
                        for start_n in tl.range(lo, hi_aligned, BLOCK_N):
                            start_n = tl.multiple_of(start_n, BLOCK_N)
                            m_i, l_i, accum_cnt_qk = _softmax_inner_iter(
                                cid,
                                accum_cnt_qk,
                                m_i,
                                l_i,
                                qk_scale,
                                klen,
                                start_n,
                                start_m,
                                qk_fulls,
                                qk_tiles,
                                alpha_empties,
                                alpha_fulls,
                                alpha_tiles,
                                p_fulls,
                                p_tiles,
                                tlx.dtype_of(desc_v),
                                BLOCK_M_SPLIT=BLOCK_M_SPLIT,
                                BLOCK_M=BLOCK_M,
                                BLOCK_N=BLOCK_N,
                                NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                                NUM_MMA_SLICES=NUM_MMA_SLICES,
                                RESCALE_OPT=RESCALE_OPT,
                                WINDOW_SIZE=WINDOW_SIZE,
                                APPLY_MASK=False,
                            )
                        for start_n in tl.range(hi_aligned, klen, BLOCK_N):
                            start_n = tl.multiple_of(start_n, BLOCK_N)
                            m_i, l_i, accum_cnt_qk = _softmax_inner_iter(
                                cid,
                                accum_cnt_qk,
                                m_i,
                                l_i,
                                qk_scale,
                                klen,
                                start_n,
                                start_m,
                                qk_fulls,
                                qk_tiles,
                                alpha_empties,
                                alpha_fulls,
                                alpha_tiles,
                                p_fulls,
                                p_tiles,
                                tlx.dtype_of(desc_v),
                                BLOCK_M_SPLIT=BLOCK_M_SPLIT,
                                BLOCK_M=BLOCK_M,
                                BLOCK_N=BLOCK_N,
                                NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                                NUM_MMA_SLICES=NUM_MMA_SLICES,
                                RESCALE_OPT=RESCALE_OPT,
                                WINDOW_SIZE=WINDOW_SIZE,
                                APPLY_MASK=True,
                            )
                    else:
                        for start_n in tl.range(lo, hi, BLOCK_N):
                            start_n = tl.multiple_of(start_n, BLOCK_N)
                            m_i, l_i, accum_cnt_qk = _softmax_inner_iter(
                                cid,
                                accum_cnt_qk,
                                m_i,
                                l_i,
                                qk_scale,
                                klen,
                                start_n,
                                start_m,
                                qk_fulls,
                                qk_tiles,
                                alpha_empties,
                                alpha_fulls,
                                alpha_tiles,
                                p_fulls,
                                p_tiles,
                                tlx.dtype_of(desc_v),
                                BLOCK_M_SPLIT=BLOCK_M_SPLIT,
                                BLOCK_M=BLOCK_M,
                                BLOCK_N=BLOCK_N,
                                NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                                NUM_MMA_SLICES=NUM_MMA_SLICES,
                                RESCALE_OPT=RESCALE_OPT,
                                WINDOW_SIZE=WINDOW_SIZE,
                                APPLY_MASK=True,
                            )

                    # prepare l_i for the epilog
                    # Use l[1]/l[1+HEAD_DIM] and m[2][2 + HEAD_DIM]
                    # to disambigulate from alpha[0]/alpha[HEAD_DIM]
                    tlx.local_store(l_tiles[cid * BLOCK_N + 1], l_i[:, None])
                    tlx.local_store(m_tiles[cid * BLOCK_N + 2], m_i[:, None])
                    tlx.barrier_arrive(l_fulls[cid])
                i += 1
                if USE_CLC:
                    if USE_I64_IDX:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer).to(
                            tl.int64
                        )
                    else:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx != -1
                else:
                    if not ENABLE_LOAD_BALANCING:
                        tile_idx += num_progs
                    has_more_tile = i < tiles_per_sm

        # mma group
        with tlx.async_task(num_warps=1, registers=NUM_REGS_MMA):
            accum_cnt_kv = 0
            accum_cnt_qk = 0
            accum_cnt_q = 0
            clc_phase_consumer = 0
            j = 0
            has_more_tile = True
            while has_more_tile:
                # initialize offsets
                begin_q, end_q, begin_k, end_k, qlen, klen, off_z, off_h, start_m = (
                    _compute_seq_len(
                        tile_idx,
                        n_tile_num,
                        Q_offsets,
                        K_offsets,
                        H,
                        N_CTX,
                        BROADCAST_Q,
                        LOAD_BALANCING_VALID,
                        BLOCK_M,
                        valid_tiles_b,
                        valid_tiles_m_start,
                        valid_tiles_m_end,
                        valid_tiles_h,
                        sm_offsets,
                        sm_start + j,
                    )
                )
                if start_m * BLOCK_M < qlen:
                    q_bufIdx, q_phase = _get_bufidx_phase(accum_cnt_q, NUM_BUFFERS_Q)
                    k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                    v_bufIdx, v_phase = _get_bufidx_phase(
                        accum_cnt_kv + 1, NUM_BUFFERS_KV
                    )

                    # wait for the K buffer to be populated by the producer
                    tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)

                    # wait for the Q buffer to be populated by the producer
                    tlx.barrier_wait(q_fulls[q_bufIdx], q_phase)

                    lo, hi = 0, klen
                    if WINDOW_SIZE is not None:
                        lo = max(
                            lo, ((start_m * BLOCK_M - WINDOW_SIZE) // BLOCK_N) * BLOCK_N
                        )
                        hi = min(hi, (start_m + 1) * BLOCK_M + WINDOW_SIZE)

                    # -- compute q0 @ k ----
                    k_tile = tlx.local_trans(kv_tiles[k_bufIdx])
                    tlx.barrier_wait(qk_empties[0], q_phase ^ 1)
                    tlx.async_dot(
                        q_tiles[0],
                        k_tile,
                        qk_tiles[0],
                        use_acc=False,
                        mBarriers=[qk_fulls[0]],
                    )

                    # -- compute q1 @ k ----
                    tlx.barrier_wait(q_fulls[q_bufIdx + NUM_BUFFERS_Q], q_phase)
                    tlx.barrier_wait(qk_empties[1], q_phase ^ 1)
                    tlx.async_dot(
                        q_tiles[1],
                        k_tile,
                        qk_tiles[1],
                        use_acc=False,
                        mBarriers=[qk_fulls[1], kv_empties[k_bufIdx]],
                    )

                    _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)

                    # -- compute p0 @ v ----
                    # wait for the V buffer to be populated by the producer
                    tlx.barrier_wait(kv_fulls[v_bufIdx], v_phase)
                    tlx.barrier_wait(acc_fulls[0], qk_phase)
                    # Use p[NUM_MMA_SLICES + slice_id] for cid=0, and
                    # p[NUM_MMA_GROUPS * NUM_MMA_SLICES + NUM_MMA_SLICES + slice_id] for cid=1
                    for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                        tlx.barrier_wait(
                            p_fulls[slice_id + 0 * NUM_MMA_SLICES], qk_phase
                        )
                        kv_slice = tlx.local_slice(
                            kv_tiles[v_bufIdx],
                            [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                            [BLOCK_N // NUM_MMA_SLICES, HEAD_DIM],
                        )
                        p_bufIdx = NUM_MMA_SLICES + slice_id
                        tlx.async_dot(
                            p_tiles[p_bufIdx],
                            kv_slice,
                            acc_tiles[0],
                            use_acc=slice_id > 0,
                            force_async=True,
                        )

                    acc1_init = False

                    for i in tl.range(lo + BLOCK_N, hi, BLOCK_N):
                        start_n = tl.multiple_of(i, BLOCK_N)
                        v_bufIdx_prev = v_bufIdx
                        qk_phase_prev = qk_phase

                        accum_cnt_qk += 1
                        accum_cnt_kv += 2
                        k_bufIdx, k_phase = _get_bufidx_phase(
                            accum_cnt_kv, NUM_BUFFERS_KV
                        )
                        v_bufIdx, v_phase = _get_bufidx_phase(
                            accum_cnt_kv + 1, NUM_BUFFERS_KV
                        )

                        # -- compute q0 @ k ----
                        # wait for the K buffer to be populated by the producer
                        tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)
                        k_tile = tlx.local_trans(kv_tiles[k_bufIdx])
                        _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)

                        tlx.async_dot(
                            q_tiles[0],
                            k_tile,
                            qk_tiles[0],
                            use_acc=False,
                            mBarriers=[qk_fulls[0]],
                        )

                        # -- compute p1 @ v from the previous iteration----
                        tlx.barrier_wait(acc_fulls[1], qk_phase_prev)
                        for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                            tlx.barrier_wait(
                                p_fulls[slice_id + 1 * NUM_MMA_SLICES], qk_phase_prev
                            )
                            kv_slice = tlx.local_slice(
                                kv_tiles[v_bufIdx_prev],
                                [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                                [BLOCK_N // NUM_MMA_SLICES, HEAD_DIM],
                            )
                            p_bufIdx = (
                                1 * NUM_MMA_GROUPS * NUM_MMA_SLICES
                                + NUM_MMA_SLICES
                                + slice_id
                            )
                            use_acc = acc1_init if slice_id == 0 else True
                            mBarriers = (
                                [kv_empties[v_bufIdx_prev]]
                                if slice_id == NUM_MMA_SLICES - 1
                                else []
                            )
                            tlx.async_dot(
                                p_tiles[p_bufIdx],
                                kv_slice,
                                acc_tiles[1],
                                use_acc=use_acc,
                                mBarriers=mBarriers,
                                force_async=True,
                            )

                        acc1_init = True

                        # -- compute q1 @ k ----
                        tlx.async_dot(
                            q_tiles[1],
                            k_tile,
                            qk_tiles[1],
                            use_acc=False,
                            mBarriers=[qk_fulls[1], kv_empties[k_bufIdx]],
                        )

                        # -- compute p0 @ v ----
                        # wait for the V buffer to be populated by the producer
                        tlx.barrier_wait(kv_fulls[v_bufIdx], v_phase)

                        tlx.barrier_wait(acc_fulls[0], qk_phase)
                        for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                            tlx.barrier_wait(
                                p_fulls[slice_id + 0 * NUM_MMA_SLICES], qk_phase
                            )
                            # Use p[1] for cid=0, and p[3] for cid=1
                            kv_slice = tlx.local_slice(
                                kv_tiles[v_bufIdx],
                                [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                                [BLOCK_N // NUM_MMA_SLICES, HEAD_DIM],
                            )
                            p_bufIdx = NUM_MMA_SLICES + slice_id
                            tlx.async_dot(
                                p_tiles[p_bufIdx],
                                kv_slice,
                                acc_tiles[0],
                                use_acc=True,
                                force_async=True,
                            )

                    tlx.tcgen05_commit(q_empties[q_bufIdx])
                    tlx.tcgen05_commit(q_empties[q_bufIdx + NUM_BUFFERS_Q])
                    tlx.tcgen05_commit(acc_empties[0])

                    # -- compute p1 @ v ----
                    tlx.barrier_wait(acc_fulls[1], qk_phase)
                    for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                        tlx.barrier_wait(p_fulls[slice_id + NUM_MMA_SLICES], qk_phase)
                        # Use p[1] for cid=0, and p[3] for cid=1
                        kv_slice = tlx.local_slice(
                            kv_tiles[v_bufIdx],
                            [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                            [BLOCK_N // NUM_MMA_SLICES, HEAD_DIM],
                        )
                        p_bufIdx = (
                            1 * NUM_MMA_GROUPS * NUM_MMA_SLICES
                            + NUM_MMA_SLICES
                            + slice_id
                        )
                        use_acc = acc1_init if slice_id == 0 else True
                        mBarriers = (
                            [acc_empties[1], kv_empties[v_bufIdx]]
                            if slice_id == NUM_MMA_SLICES - 1
                            else []
                        )
                        tlx.async_dot(
                            p_tiles[p_bufIdx],
                            kv_slice,
                            acc_tiles[1],
                            use_acc=use_acc,
                            mBarriers=mBarriers,
                            force_async=True,
                        )

                    accum_cnt_qk += 1
                    accum_cnt_kv += 2
                    accum_cnt_q += 1
                j += 1
                if USE_CLC:
                    if USE_I64_IDX:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer).to(
                            tl.int64
                        )
                    else:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx != -1
                else:
                    if not ENABLE_LOAD_BALANCING:
                        tile_idx += num_progs
                    has_more_tile = j < tiles_per_sm

        # load
        with tlx.async_task(num_warps=1, registers=NUM_REGS_LOAD):
            accum_cnt_kv = 0
            accum_cnt_q = 0
            clc_phase_consumer = 0
            i = 0
            has_more_tile = True
            while has_more_tile:
                # initialize offsets
                begin_q, end_q, begin_k, end_k, qlen, klen, off_z, off_h, start_m = (
                    _compute_seq_len(
                        tile_idx,
                        n_tile_num,
                        Q_offsets,
                        K_offsets,
                        H,
                        N_CTX,
                        BROADCAST_Q,
                        LOAD_BALANCING_VALID,
                        BLOCK_M,
                        valid_tiles_b,
                        valid_tiles_m_start,
                        valid_tiles_m_end,
                        valid_tiles_h,
                        sm_offsets,
                        sm_start + i,
                    )
                )
                off_h_kv = off_h // G
                q_offset = off_h.to(tl.int64) * stride_qh
                kv_offset = off_h_kv.to(tl.int64) * stride_kh

                if start_m * BLOCK_M < qlen:
                    lo, hi = 0, klen
                    if WINDOW_SIZE is not None:
                        lo = max(
                            lo, ((start_m * BLOCK_M - WINDOW_SIZE) // BLOCK_N) * BLOCK_N
                        )
                        hi = min(hi, (start_m + 1) * BLOCK_M + WINDOW_SIZE)
                    # load q0
                    q_bufIdx, q_phase = _get_bufidx_phase(accum_cnt_q, NUM_BUFFERS_Q)
                    tlx.barrier_wait(q_empties[q_bufIdx], q_phase ^ 1)
                    tlx.barrier_expect_bytes(
                        q_fulls[q_bufIdx], 2 * BLOCK_M_SPLIT * BLOCK_D
                    )  # float16
                    tlx.async_descriptor_load(
                        desc_q,
                        q_tiles[q_bufIdx],
                        [
                            (begin_q + start_m * BLOCK_M).to(tl.int32),
                            (q_offset).to(tl.int32),
                        ],
                        q_fulls[q_bufIdx],
                    )

                    # loop over loading k, v
                    k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                    # wait for the K buffer to be released by the consumer
                    k_empty = tlx.local_view(kv_empties, k_bufIdx)
                    tlx.barrier_wait(k_empty, k_phase ^ 1)

                    # load K
                    k_full = tlx.local_view(kv_fulls, k_bufIdx)
                    k_tile = tlx.local_view(kv_tiles, k_bufIdx)
                    tlx.barrier_expect_bytes(k_full, 2 * BLOCK_N * BLOCK_D)  # float16
                    start_n = lo
                    tlx.async_descriptor_load(
                        desc_k,
                        k_tile,
                        [
                            (begin_k + start_n).to(tl.int32),
                            (kv_offset).to(tl.int32),
                        ],
                        k_full,
                    )

                    # load q1
                    q_bufIdx += NUM_BUFFERS_Q
                    tlx.barrier_wait(q_empties[q_bufIdx], q_phase ^ 1)
                    tlx.barrier_expect_bytes(
                        q_fulls[q_bufIdx], 2 * BLOCK_M_SPLIT * BLOCK_D
                    )  # float16
                    tlx.async_descriptor_load(
                        desc_q,
                        q_tiles[q_bufIdx],
                        [
                            (begin_q + start_m * BLOCK_M + BLOCK_M_SPLIT).to(tl.int32),
                            (q_offset).to(tl.int32),
                        ],
                        q_fulls[q_bufIdx],
                    )

                    v_bufIdx, v_phase = _get_bufidx_phase(
                        accum_cnt_kv + 1, NUM_BUFFERS_KV
                    )
                    # wait for the V buffer to be released by the consumer
                    v_empty = tlx.local_view(kv_empties, v_bufIdx)
                    tlx.barrier_wait(v_empty, v_phase ^ 1)
                    # load V
                    v_full = tlx.local_view(kv_fulls, v_bufIdx)
                    v_tile = tlx.local_view(kv_tiles, v_bufIdx)
                    tlx.barrier_expect_bytes(v_full, 2 * BLOCK_N * BLOCK_D)  # float16
                    tlx.async_descriptor_load(
                        desc_v,
                        v_tile,
                        [(begin_k + start_n).to(tl.int32), (kv_offset).to(tl.int32)],
                        v_full,
                    )

                    accum_cnt_kv += 2

                    for start_n in tl.range(lo + BLOCK_N, hi, BLOCK_N):
                        start_n = tl.multiple_of(start_n, BLOCK_N)
                        k_bufIdx, k_phase = _get_bufidx_phase(
                            accum_cnt_kv, NUM_BUFFERS_KV
                        )
                        # wait for the K buffer to be released by the consumer
                        k_empty = tlx.local_view(kv_empties, k_bufIdx)
                        tlx.barrier_wait(k_empty, k_phase ^ 1)
                        # load K
                        k_full = tlx.local_view(kv_fulls, k_bufIdx)
                        k_tile = tlx.local_view(kv_tiles, k_bufIdx)
                        tlx.barrier_expect_bytes(
                            k_full, 2 * BLOCK_N * BLOCK_D
                        )  # float16
                        tlx.async_descriptor_load(
                            desc_k,
                            k_tile,
                            [
                                (begin_k + start_n).to(tl.int32),
                                (kv_offset).to(tl.int32),
                            ],
                            k_full,
                        )

                        v_bufIdx, v_phase = _get_bufidx_phase(
                            accum_cnt_kv + 1, NUM_BUFFERS_KV
                        )
                        # wait for the V buffer to be released by the consumer
                        v_empty = tlx.local_view(kv_empties, v_bufIdx)
                        tlx.barrier_wait(v_empty, v_phase ^ 1)
                        # load V
                        v_full = tlx.local_view(kv_fulls, v_bufIdx)
                        v_tile = tlx.local_view(kv_tiles, v_bufIdx)
                        tlx.barrier_expect_bytes(
                            v_full, 2 * BLOCK_N * BLOCK_D
                        )  # float16
                        tlx.async_descriptor_load(
                            desc_v,
                            v_tile,
                            [
                                (begin_k + start_n).to(tl.int32),
                                (kv_offset).to(tl.int32),
                            ],
                            v_full,
                        )

                        accum_cnt_kv += 2
                    accum_cnt_q += 1
                i += 1
                if USE_CLC:
                    if USE_I64_IDX:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer).to(
                            tl.int64
                        )
                    else:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx != -1
                else:
                    if not ENABLE_LOAD_BALANCING:
                        tile_idx += num_progs
                    has_more_tile = i < tiles_per_sm

        # epilog group
        with tlx.async_task(num_warps=1, registers=NUM_REGS_EPILOG):
            accum_cnt = 0
            clc_phase_consumer = 0
            clc_phase_producer = 1
            i = 0
            has_more_tile = True
            while has_more_tile:
                if USE_CLC:
                    tlx.clc_producer(clc_context, clc_phase_producer)
                    clc_phase_producer = clc_phase_producer ^ 1
                # initialize offsets
                begin_q, end_q, begin_k, end_k, qlen, klen, off_z, off_h, start_m = (
                    _compute_seq_len(
                        tile_idx,
                        n_tile_num,
                        Q_offsets,
                        K_offsets,
                        H,
                        N_CTX,
                        BROADCAST_Q,
                        LOAD_BALANCING_VALID,
                        BLOCK_M,
                        valid_tiles_b,
                        valid_tiles_m_start,
                        valid_tiles_m_end,
                        valid_tiles_h,
                        sm_offsets,
                        sm_start + i,
                    )
                )
                out_offset = off_h.to(tl.int64) * stride_oh
                if not BROADCAST_Q:
                    begin_o = begin_q
                    end_o = end_q
                else:
                    begin_o = qlen * off_z
                    end_o = qlen * (off_z + 1)
                _, phase = _get_bufidx_phase(accum_cnt, 1)
                if start_m * BLOCK_M < qlen:
                    desc_o = tl.make_tensor_descriptor(
                        Out,
                        shape=[end_o.to(tl.int32), HEAD_DIM * H],
                        strides=[HEAD_DIM * H, 1],
                        block_shape=[BLOCK_M_SPLIT, BLOCK_D],
                    )
                    for cid in tl.static_range(0, NUM_MMA_GROUPS):
                        tlx.barrier_wait(o_fulls[cid], phase)
                        tlx.fence_async_shared()
                        tlx.async_descriptor_store(
                            desc_o,
                            o_tiles[cid],
                            [
                                (begin_o + start_m * BLOCK_M + cid * BLOCK_M_SPLIT).to(
                                    tl.int32
                                ),
                                (out_offset).to(tl.int32),
                            ],
                        )
                        tlx.async_descriptor_store_wait(0)
                        tlx.barrier_arrive(o_empties[cid])
                    accum_cnt += 1
                i += 1
                if USE_CLC:
                    if USE_I64_IDX:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer).to(
                            tl.int64
                        )
                    else:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx != -1
                else:
                    if not ENABLE_LOAD_BALANCING:
                        tile_idx += num_progs
                    has_more_tile = i < tiles_per_sm


@triton.jit  # pragma: no cover
# Triton TR001: launched through _get_autotune_bwd_preprocess_kernel.
def _attn_bwd_preprocess(  # noqa: TR001
    O,
    O_offsets,
    DO,  #
    Delta,  #
    stride_lh,
    H,
    N_CTX,  #
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid0 = tl.program_id(0).to(tl.int64)
    start_m = pid0 * BLOCK_M
    off_m = start_m + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1).to(tl.int64)
    off_d = tl.arange(0, BLOCK_D)
    off_h = off_hz % H
    off_z = off_hz // H
    begin_o = tl.load(O_offsets + off_z).to(tl.int64)
    end_o = tl.load(O_offsets + off_z + 1).to(tl.int64)
    olen = end_o - begin_o
    olen = tl.minimum(olen, N_CTX)
    if start_m > olen:
        return
    stride_om = HEAD_DIM * H
    o_offsets = (
        begin_o * stride_om
        + off_h * HEAD_DIM
        + off_m[:, None] * stride_om
        + off_d[None, :]
    )
    o_mask = (off_m[:, None] < olen) & (off_d[None, :] < HEAD_DIM)
    # load
    o = tl.load(O + o_offsets, mask=o_mask, other=0.0)
    do = tl.load(DO + o_offsets, mask=o_mask, other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_h * stride_lh + begin_o + off_m, delta, mask=off_m < olen)


def _bwd_host_descriptor_pre_hook_tlx_2cta(nargs):
    # 2-CTA (cluster) descriptor block shapes, in the kernel's flat 2D jagged
    # layout over [total_tokens, HEAD_DIM * H]. Q/dO/dQ split along the head
    # dimension (HEAD_DIM // NUM_CTAS); the transposed B-operand descriptors
    # (kt/qt/dot) split along the seq dimension (BLOCK_* // NUM_CTAS).
    BLOCK_M1 = nargs["BLOCK_M1"]
    BLOCK_N1 = nargs["BLOCK_N1"]
    HEAD_DIM = nargs["HEAD_DIM"]
    EPILOGUE_SUBTILE = nargs["EPILOGUE_SUBTILE"]
    NUM_CTAS = nargs["NUM_CTAS"]

    nargs["desc_q"].block_shape = [BLOCK_M1, HEAD_DIM // NUM_CTAS]
    nargs["desc_do"].block_shape = [BLOCK_M1, HEAD_DIM // NUM_CTAS]
    nargs["desc_v"].block_shape = [BLOCK_N1, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N1, HEAD_DIM]
    nargs["desc_dq"].block_shape = [BLOCK_M1 // NUM_CTAS, HEAD_DIM // EPILOGUE_SUBTILE]
    # Transposed B-operand descriptors over the same tensors.
    nargs["desc_kt"].block_shape = [BLOCK_N1 * NUM_CTAS, HEAD_DIM // NUM_CTAS]
    nargs["desc_qt"].block_shape = [BLOCK_M1 // NUM_CTAS, HEAD_DIM]
    nargs["desc_dot"].block_shape = [BLOCK_M1 // NUM_CTAS, HEAD_DIM]


def _make_bwd_2cta_config(
    enable_clc: bool, persistent: bool, epilogue_subtile: int
) -> triton.Config:
    return triton.Config(
        {
            "BLOCK_M1": 128,
            "BLOCK_N1": 128,
            "NUM_BUFFERS_KV": 1,
            "NUM_BUFFERS_Q": 1,
            "NUM_BUFFERS_DO": 1,
            "NUM_BUFFERS_DS": 1,
            "NUM_BUFFERS_TMEM": 1,
            "EPILOGUE_SUBTILE": epilogue_subtile,
            "DKV_STORE_NCOL": 64,
            "NUM_CTAS": 2,
            "ENABLE_CLC": enable_clc,
            "PERSISTENT": persistent,
        },
        num_warps=8,
        num_stages=1,
        pre_hook=_bwd_host_descriptor_pre_hook_tlx_2cta,
        ctas_per_cga=(2, 1, 1),
    )


# 2-CTA backward autotune candidates. Both modes are correct; the autotuner
# picks the faster per (N_CTX, HEAD_DIM, H, G). The CLC-vs-persistent choice is
# expressed here (in the config) rather than via a global.
#   - persistent (static round-robin): EPILOGUE_SUBTILE=8, the benchmarked winner
#     on PMA shapes (~+6-12% bwd vs 1-CTA).
#   - CLC (Cluster Launch Control work-stealing): must use EPILOGUE_SUBTILE=16
#     because the launch-control context needs ~32B more SMEM (ES=8 overflows the
#     SMEM limit by 8B), which makes it slower than persistent on these shapes.
configs_bwd_2cta_tlx = [
    _make_bwd_2cta_config(enable_clc=False, persistent=True, epilogue_subtile=8),
    _make_bwd_2cta_config(enable_clc=True, persistent=False, epilogue_subtile=16),
]


@lru_cache
def _get_autotune_bwd_2cta_kernel(kernel: JITFunction) -> JITFunction:
    return triton.autotune(
        configs=configs_bwd_2cta_tlx,
        key=["N_CTX", "HEAD_DIM", "H", "G"],
        restore_value=["dQ"],
    )(kernel)


configs_bwd_preproc_tlx = [
    triton.Config(
        {
            "BLOCK_M": BM,
        },
        num_warps=w,
        num_stages=s,
    )
    for BM in [128]  # 128 or 256
    for w in [8]
    for s in [2]
]


@lru_cache
def get_cuda_autotune_config_bwd_preproc() -> list[triton.Config]:
    return configs_bwd_preproc_tlx


@lru_cache
def _get_autotune_bwd_preprocess_kernel(kernel: JITFunction) -> JITFunction:
    return triton.autotune(
        configs=get_cuda_autotune_config_bwd_preproc(),
        key=["N_CTX", "HEAD_DIM", "H"],
    )(kernel)


# =============================================================================
# 2-CTA (cluster, collaborative-MMA) backward path.
#
# Ported from the TLX tutorial `blackwell_fa_ws_pipelined_persistent.py`
# (`_bwd_load_2cta` / `_bwd_mma_dots_2cta` / `_bwd_compute_inner_loop` 2-CTA
# branch + relay task), adapted to the kernel's flat 2D jagged TMA
# descriptors and broadcast_q addressing. Scoped to broadcast_q + HEAD_DIM=128
# + G=1 + load-balancing-on; sliding window is supported. Non-persistent, CLC
# disabled.
#
# Two CTAs in a cluster (cluster_cta_rank 0 / 1) collaboratively process two
# adjacent N-blocks of the *same* (batch, head). All MMAs are issued by the
# leader (rank 0) with `two_ctas=True`; the accumulator M dimension is split
# 64+64 across the two SMs. dS is staged in TMEM, then the two CTAs exchange
# halves over DSMEM so each can compute its half of dQ.
#
# Because the MMAs are collaborative, BOTH CTAs must always run the full
# pipeline (no `if start_n < klen` guard). Sentinel/padding tiles and jagged
# out-of-bounds rows are zeroed via P-masking (`offs_n < klen`); dK/dV stores
# use on-device descriptors bounded by `end_k` so OOB stores are dropped.
# =============================================================================


@triton.jit  # pragma: no cover
def _bwd_mma_dots_2cta(
    blk_idx,
    num_steps,
    kv_buf_id,
    kv_phase,
    k_tiles,
    v_tiles,
    q_tiles,
    do_tiles,
    qk_tiles,
    qk_fulls,
    qk_empties,
    p_tiles,
    p_fulls,
    dp_tiles,
    dp_fulls,
    dp_empties,
    dv_tiles,
    dv_fulls,
    dv_empties,
    dk_tiles,
    dk_fulls,
    dk_empties,
    dq_tiles,
    dq_fulls,
    dq_empties,
    ds_tiles,
    ds_fulls,
    dsT_tmem_tiles,
    dsT_tmem_fulls,
    do_fulls,
    do_empties,
    q_fulls,
    q_empties,
    k_mma_done,
    qt_tiles,
    dot_tiles,
    kt_tiles,
    qt_fulls,
    qt_empties,
    dot_fulls,
    dot_empties,
    kt_fulls,
    kt_empties,
    k_fulls,
    v_fulls,
    ds_empties,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_DO: tl.constexpr,
    NUM_BUFFERS_TMEM: tl.constexpr,
    NUM_BUFFERS_DS: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    DQ_BUF_OFFSET: tl.constexpr = 0,
    P_BUF_OFFSET: tl.constexpr = 0,
):
    """2-CTA MMA dot sequence: prolog + main loop + epilog (S -> dK -> dP ->
    dQ -> dV). All dots use two_ctas=True. Verbatim port of the tutorial."""
    tlx.barrier_wait(k_fulls[kv_buf_id], kv_phase)
    tlx.barrier_wait(v_fulls[kv_buf_id], kv_phase)

    q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
    do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
    tmem_buf_id, tmem_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)

    # Dot 1: qkT = tl.dot(k, qT)
    tlx.barrier_wait(qt_fulls[q_buf_id], q_phase)
    tlx.barrier_wait(qk_empties[tmem_buf_id], tmem_phase ^ 1)
    qT = tlx.local_trans(qt_tiles[q_buf_id])
    tlx.async_dot(
        k_tiles[kv_buf_id],
        qT,
        qk_tiles[tmem_buf_id],
        use_acc=False,
        mBarriers=[qk_fulls[tmem_buf_id], qt_empties[q_buf_id]],
        two_ctas=True,
    )

    # Dot 2: dpT = tl.dot(v, tl.trans(do))
    tlx.barrier_wait(dot_fulls[do_buf_id], do_phase)
    doT = tlx.local_trans(dot_tiles[do_buf_id])
    tlx.async_dot(
        v_tiles[kv_buf_id],
        doT,
        dp_tiles[tmem_buf_id],
        use_acc=False,
        mBarriers=[dp_fulls[tmem_buf_id], dot_empties[do_buf_id]],
        two_ctas=True,
    )

    # Dot 3: dv += tl.dot(ppT, do)
    tlx.barrier_wait(do_fulls[do_buf_id], do_phase)
    tlx.barrier_wait(p_fulls[tmem_buf_id], tmem_phase)
    tlx.barrier_wait(dv_empties[kv_buf_id], kv_phase ^ 1)
    tlx.async_dot(
        p_tiles[tmem_buf_id + P_BUF_OFFSET],
        do_tiles[do_buf_id],
        dv_tiles[kv_buf_id],
        use_acc=False,
        mBarriers=[do_empties[do_buf_id]],
        two_ctas=True,
    )
    blk_idx += 1

    # Main loop: S -> dK -> dP -> dQ -> dV
    tlx.barrier_wait(dk_empties[kv_buf_id], kv_phase ^ 1)
    # kt is loaded once per n-block and reused across the whole m-loop, so wait on
    # it once here (like k_fulls/v_fulls) instead of every iteration.
    tlx.barrier_wait(kt_fulls[kv_buf_id], kv_phase)
    for j in range(1, num_steps):
        q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
        tmem_buf_id, tmem_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_TMEM)

        tlx.barrier_wait(qt_fulls[q_buf_id], q_phase)
        tlx.barrier_wait(qk_empties[tmem_buf_id], tmem_phase ^ 1)
        prev_tmem_buf_id, prev_tmem_phase = _get_bufidx_phase(
            blk_idx - 1, NUM_BUFFERS_TMEM
        )
        tlx.barrier_wait(dq_empties[prev_tmem_buf_id], prev_tmem_phase ^ 1)
        qT = tlx.local_trans(qt_tiles[q_buf_id])
        tlx.async_dot(
            k_tiles[kv_buf_id],
            qT,
            qk_tiles[tmem_buf_id],
            use_acc=False,
            mBarriers=[qk_fulls[tmem_buf_id], qt_empties[q_buf_id]],
            two_ctas=True,
        )

        prev_blk_idx = blk_idx - 1
        q_buf_id_prev, q_phase_prev = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_Q)
        tmem_buf_id_prev, tmem_phase_prev = _get_bufidx_phase(
            prev_blk_idx, NUM_BUFFERS_TMEM
        )
        ds_buf_id_prev, ds_phase_prev = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_DS)

        # Dot 4: dk += tl.dot(dsT, q) (read dsT from TMEM)
        tlx.barrier_wait(q_fulls[q_buf_id_prev], q_phase_prev)
        tlx.barrier_wait(dsT_tmem_fulls[ds_buf_id_prev], ds_phase_prev)
        tlx.async_dot(
            dsT_tmem_tiles[ds_buf_id_prev],
            q_tiles[q_buf_id_prev],
            dk_tiles[kv_buf_id],
            use_acc=(j - 1) > 0,
            mBarriers=[q_empties[q_buf_id_prev], dp_empties[ds_buf_id_prev]],
            two_ctas=True,
        )

        do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
        tlx.barrier_wait(dot_fulls[do_buf_id], do_phase)
        tlx.barrier_wait(dp_empties[tmem_buf_id], tmem_phase ^ 1)
        doT = tlx.local_trans(dot_tiles[do_buf_id])
        tlx.async_dot(
            v_tiles[kv_buf_id],
            doT,
            dp_tiles[tmem_buf_id],
            use_acc=False,
            mBarriers=[dp_fulls[tmem_buf_id], dot_empties[do_buf_id]],
            two_ctas=True,
        )
        # Dot 5: dq = tl.dot(tl.trans(dsT), k)
        # dq_empties[prev] was already waited before Dot 1 (qk aliases the dq TMEM
        # region) and kt_fulls is now waited once before the loop, so neither needs
        # re-waiting here.
        tlx.barrier_wait(ds_fulls[ds_buf_id_prev], ds_phase_prev)
        dsT_view = tlx.local_trans(ds_tiles[ds_buf_id_prev])
        tlx.async_dot(
            dsT_view,
            kt_tiles[kv_buf_id],
            dq_tiles[tmem_buf_id_prev + DQ_BUF_OFFSET],
            use_acc=False,
            mBarriers=[dq_fulls[tmem_buf_id_prev], ds_empties[ds_buf_id_prev]],
            two_ctas=True,
        )
        # Dot 3: dv += tl.dot(ppT, do)
        tlx.barrier_wait(do_fulls[do_buf_id], do_phase)
        tlx.barrier_wait(p_fulls[tmem_buf_id], tmem_phase)
        tlx.async_dot(
            p_tiles[tmem_buf_id + P_BUF_OFFSET],
            do_tiles[do_buf_id],
            dv_tiles[kv_buf_id],
            use_acc=True,
            mBarriers=[do_empties[do_buf_id]],
            two_ctas=True,
        )
        blk_idx += 1

    tlx.tcgen05_commit(dv_fulls[kv_buf_id], two_ctas=True)

    # Epilog: dk += dsT @ q ; dq = trans(dsT) @ k
    prev_blk_idx = blk_idx - 1
    q_buf_id, q_phase = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_Q)
    tmem_buf_id, tmem_phase = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_TMEM)
    ds_buf_id, ds_phase = _get_bufidx_phase(prev_blk_idx, NUM_BUFFERS_DS)
    tlx.barrier_wait(q_fulls[q_buf_id], q_phase)
    tlx.barrier_wait(dsT_tmem_fulls[ds_buf_id], ds_phase)
    tlx.async_dot(
        dsT_tmem_tiles[ds_buf_id],
        q_tiles[q_buf_id],
        dk_tiles[kv_buf_id],
        use_acc=num_steps > 1,
        mBarriers=[q_empties[q_buf_id], dk_fulls[kv_buf_id], dp_empties[ds_buf_id]],
        two_ctas=True,
    )

    tlx.barrier_wait(ds_fulls[ds_buf_id], ds_phase)
    tlx.barrier_wait(dq_empties[tmem_buf_id], tmem_phase ^ 1)
    dsT_view = tlx.local_trans(ds_tiles[ds_buf_id])
    tlx.barrier_wait(kt_fulls[kv_buf_id], kv_phase)
    tlx.async_dot(
        dsT_view,
        kt_tiles[kv_buf_id],
        dq_tiles[tmem_buf_id + DQ_BUF_OFFSET],
        use_acc=False,
        mBarriers=[dq_fulls[tmem_buf_id], ds_empties[ds_buf_id]],
        two_ctas=True,
    )
    tlx.tcgen05_commit(k_mma_done[kv_buf_id], two_ctas=True)
    tlx.tcgen05_commit(kt_empties[kv_buf_id], two_ctas=True)

    return blk_idx


@triton.jit  # pragma: no cover
def _bwd_load_2cta(  # noqa: C901
    blk_idx,
    begin_q,
    begin_o,
    begin_k,
    off_h2,
    off_h_kv,
    stride_qh,
    stride_kh,
    off_chz,
    start_n,
    num_steps,
    tile_count,
    desc_k,
    desc_v,
    desc_q,
    desc_do,
    desc_kt,
    desc_qt,
    desc_dot,
    M,
    D,
    k_tiles,
    v_tiles,
    q_tiles,
    do_tiles,
    sM_tiles,
    sD_tiles,
    k_empties,
    q_fulls,
    q_empties,
    do_fulls,
    do_empties,
    m_fulls,
    m_empties,
    d_fulls,
    d_empties,
    k_fulls,
    v_fulls,
    kt_tiles,
    kt_fulls,
    kt_empties,
    qt_tiles,
    qt_fulls,
    qt_empties,
    dot_tiles,
    dot_fulls,
    dot_empties,
    cluster_cta_rank,
    is_leader,
    K_BYTES_PER_ELEM: tl.constexpr,
    V_BYTES_PER_ELEM: tl.constexpr,
    Q_BYTES_PER_ELEM: tl.constexpr,
    DO_BYTES_PER_ELEM: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_DO: tl.constexpr,
    M_STAGE: tl.constexpr,
    D_STAGE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_CTAS: tl.constexpr,
):
    HD_SPLIT: tl.constexpr = HEAD_DIM // NUM_CTAS
    M_SPLIT: tl.constexpr = BLOCK_M1 // NUM_CTAS
    kv_col = (off_h_kv * stride_kh).to(tl.int32)
    q_col = (off_h2 * stride_qh).to(tl.int32)
    rank_hd = cluster_cta_rank * HD_SPLIT
    rank_m = cluster_cta_rank * M_SPLIT

    curr_m = 0
    step_m = BLOCK_M1

    # Load K — both CTAs load their own N-block (clustered barrier).
    kv_buf_id, kv_phase = _get_bufidx_phase(tile_count, NUM_BUFFERS_KV)
    tlx.barrier_wait(k_empties[kv_buf_id], kv_phase ^ 1)
    if is_leader:
        tlx.barrier_expect_bytes(
            k_fulls[kv_buf_id], K_BYTES_PER_ELEM * BLOCK_N1 * HEAD_DIM * NUM_CTAS
        )
    tlx.async_descriptor_load(
        desc_k,
        k_tiles[kv_buf_id],
        [(begin_k + start_n).to(tl.int32), kv_col],
        k_fulls[kv_buf_id],
        two_ctas=tl.constexpr(True),
    )

    # Load V
    if is_leader:
        tlx.barrier_expect_bytes(
            v_fulls[kv_buf_id], V_BYTES_PER_ELEM * BLOCK_N1 * HEAD_DIM * NUM_CTAS
        )
    tlx.async_descriptor_load(
        desc_v,
        v_tiles[kv_buf_id],
        [(begin_k + start_n).to(tl.int32), kv_col],
        v_fulls[kv_buf_id],
        two_ctas=tl.constexpr(True),
    )

    # Load Qt [M_SPLIT, HEAD_DIM] per CTA (for dots 1,2).
    q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
    tlx.barrier_wait(qt_empties[q_buf_id], q_phase ^ 1)
    if is_leader:
        tlx.barrier_expect_bytes(
            qt_fulls[q_buf_id], Q_BYTES_PER_ELEM * BLOCK_M1 * HEAD_DIM
        )
    tlx.async_descriptor_load(
        desc_qt,
        qt_tiles[q_buf_id],
        [(begin_q + curr_m + rank_m).to(tl.int32), q_col],
        qt_fulls[q_buf_id],
        two_ctas=tl.constexpr(True),
    )

    # Load M (raw bulk copy)
    m_buf_id, m_phase = _get_bufidx_phase(blk_idx, M_STAGE)
    tlx.barrier_wait(m_empties[m_buf_id], m_phase ^ 1)
    tlx.barrier_expect_bytes(m_fulls[m_buf_id], 4 * BLOCK_M1)
    tlx.async_load(
        M + off_chz + curr_m, sM_tiles[m_buf_id], bulk=True, barrier=m_fulls[m_buf_id]
    )

    # Load dO [BLOCK_M1, HD_SPLIT] per CTA.
    do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)
    tlx.barrier_wait(do_empties[do_buf_id], do_phase ^ 1)
    if is_leader:
        tlx.barrier_expect_bytes(
            do_fulls[do_buf_id], DO_BYTES_PER_ELEM * BLOCK_M1 * HEAD_DIM
        )
    tlx.async_descriptor_load(
        desc_do,
        do_tiles[do_buf_id],
        [(begin_o + curr_m).to(tl.int32), q_col + rank_hd],
        do_fulls[do_buf_id],
        two_ctas=tl.constexpr(True),
    )
    # Load dOt [M_SPLIT, HEAD_DIM] per CTA (for dots 1,2).
    tlx.barrier_wait(dot_empties[do_buf_id], do_phase ^ 1)
    if is_leader:
        tlx.barrier_expect_bytes(
            dot_fulls[do_buf_id], DO_BYTES_PER_ELEM * BLOCK_M1 * HEAD_DIM
        )
    tlx.async_descriptor_load(
        desc_dot,
        dot_tiles[do_buf_id],
        [(begin_o + curr_m + rank_m).to(tl.int32), q_col],
        dot_fulls[do_buf_id],
        two_ctas=tl.constexpr(True),
    )

    # Load D (delta) (raw bulk copy)
    d_buf_id, d_phase = _get_bufidx_phase(blk_idx, D_STAGE)
    tlx.barrier_wait(d_empties[d_buf_id], d_phase ^ 1)
    tlx.barrier_expect_bytes(d_fulls[d_buf_id], 4 * BLOCK_M1)
    tlx.async_load(
        D + off_chz + curr_m, sD_tiles[d_buf_id], bulk=True, barrier=d_fulls[d_buf_id]
    )

    # Load Kt (B for dQ = dS @ K), [BLOCK_N1*NUM_CTAS, HD_SPLIT] per CTA.
    tlx.barrier_wait(kt_empties[kv_buf_id], kv_phase ^ 1)
    lower_start_n = start_n - cluster_cta_rank * BLOCK_N1
    if is_leader:
        tlx.barrier_expect_bytes(
            kt_fulls[kv_buf_id], K_BYTES_PER_ELEM * BLOCK_N1 * HEAD_DIM * NUM_CTAS
        )
    tlx.async_descriptor_load(
        desc_kt,
        kt_tiles[kv_buf_id],
        [(begin_k + lower_start_n).to(tl.int32), kv_col + rank_hd],
        kt_fulls[kv_buf_id],
        two_ctas=tl.constexpr(True),
    )

    curr_m += step_m
    blk_idx += 1

    for _ in range(1, num_steps):
        q_buf_id, q_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_Q)
        do_buf_id, do_phase = _get_bufidx_phase(blk_idx, NUM_BUFFERS_DO)

        tlx.barrier_wait(qt_empties[q_buf_id], q_phase ^ 1)
        if is_leader:
            tlx.barrier_expect_bytes(
                qt_fulls[q_buf_id], Q_BYTES_PER_ELEM * BLOCK_M1 * HEAD_DIM
            )
        tlx.async_descriptor_load(
            desc_qt,
            qt_tiles[q_buf_id],
            [(begin_q + curr_m + rank_m).to(tl.int32), q_col],
            qt_fulls[q_buf_id],
            two_ctas=tl.constexpr(True),
        )

        tlx.barrier_wait(dot_empties[do_buf_id], do_phase ^ 1)
        if is_leader:
            tlx.barrier_expect_bytes(
                dot_fulls[do_buf_id], DO_BYTES_PER_ELEM * BLOCK_M1 * HEAD_DIM
            )
        tlx.async_descriptor_load(
            desc_dot,
            dot_tiles[do_buf_id],
            [(begin_o + curr_m + rank_m).to(tl.int32), q_col],
            dot_fulls[do_buf_id],
            two_ctas=tl.constexpr(True),
        )

        prev_q_buf_id, prev_q_phase = _get_bufidx_phase(blk_idx - 1, NUM_BUFFERS_Q)
        tlx.barrier_wait(q_empties[prev_q_buf_id], prev_q_phase ^ 1)
        if is_leader:
            tlx.barrier_expect_bytes(
                q_fulls[prev_q_buf_id], Q_BYTES_PER_ELEM * BLOCK_M1 * HEAD_DIM
            )
        tlx.async_descriptor_load(
            desc_q,
            q_tiles[prev_q_buf_id],
            [(begin_q + curr_m - step_m).to(tl.int32), q_col + rank_hd],
            q_fulls[prev_q_buf_id],
            two_ctas=tl.constexpr(True),
        )

        m_buf_id, m_phase = _get_bufidx_phase(blk_idx, M_STAGE)
        tlx.barrier_wait(m_empties[m_buf_id], m_phase ^ 1)
        tlx.barrier_expect_bytes(m_fulls[m_buf_id], 4 * BLOCK_M1)
        tlx.async_load(
            M + off_chz + curr_m,
            sM_tiles[m_buf_id],
            bulk=True,
            barrier=m_fulls[m_buf_id],
        )

        tlx.barrier_wait(do_empties[do_buf_id], do_phase ^ 1)
        if is_leader:
            tlx.barrier_expect_bytes(
                do_fulls[do_buf_id], DO_BYTES_PER_ELEM * BLOCK_M1 * HEAD_DIM
            )
        tlx.async_descriptor_load(
            desc_do,
            do_tiles[do_buf_id],
            [(begin_o + curr_m).to(tl.int32), q_col + rank_hd],
            do_fulls[do_buf_id],
            two_ctas=tl.constexpr(True),
        )

        d_buf_id, d_phase = _get_bufidx_phase(blk_idx, D_STAGE)
        tlx.barrier_wait(d_empties[d_buf_id], d_phase ^ 1)
        tlx.barrier_expect_bytes(d_fulls[d_buf_id], 4 * BLOCK_M1)
        tlx.async_load(
            D + off_chz + curr_m,
            sD_tiles[d_buf_id],
            bulk=True,
            barrier=d_fulls[d_buf_id],
        )

        curr_m += step_m
        blk_idx += 1

    # Load q_tiles for the last M-block (epilog dk will consume).
    last_q_buf_id, last_q_phase = _get_bufidx_phase(blk_idx - 1, NUM_BUFFERS_Q)
    tlx.barrier_wait(q_empties[last_q_buf_id], last_q_phase ^ 1)
    if is_leader:
        tlx.barrier_expect_bytes(
            q_fulls[last_q_buf_id], Q_BYTES_PER_ELEM * BLOCK_M1 * HEAD_DIM
        )
    tlx.async_descriptor_load(
        desc_q,
        q_tiles[last_q_buf_id],
        [(begin_q + curr_m - step_m).to(tl.int32), q_col + rank_hd],
        q_fulls[last_q_buf_id],
        two_ctas=tl.constexpr(True),
    )

    return blk_idx


@triton.jit  # pragma: no cover
def _bwd_compute_2cta_inner(
    start_n,
    klen,
    qlen,
    curr_m,
    blk_idx,
    num_steps,
    qk_fulls,
    qk_tiles,
    qk_empties,
    p_tiles,
    p_fulls,
    dp_empties,
    dp_fulls,
    dp_tiles,
    ds_tiles,
    dsT_tmem_tiles,
    dsT_tmem_fulls,
    sM_tiles,
    sD_tiles,
    m_fulls,
    m_empties,
    d_fulls,
    d_empties,
    ds_xchg_tiles,
    ds_peer_fulls,
    ds_empties,
    cluster_cta_rank,
    do_out_dtype,
    q_out_dtype,
    sm_scale,
    NUM_BUFFERS_TMEM: tl.constexpr,
    NUM_BUFFERS_DS: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    M_STAGE: tl.constexpr,
    D_STAGE: tl.constexpr,
    LN2: tl.constexpr,
    WINDOW_SIZE: tl.constexpr = None,
    P_BUF_OFFSET: tl.constexpr = 0,
):
    M_SPLIT: tl.constexpr = BLOCK_M1 // NUM_CTAS
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    for it in range(num_steps):
        blk = blk_idx + it
        cur_m = curr_m + it * BLOCK_M1
        tmem_buf_id, tmem_phase = _get_bufidx_phase(blk, NUM_BUFFERS_TMEM)
        ds_buf_id, _ = _get_bufidx_phase(blk, NUM_BUFFERS_DS)
        m_buf_id, m_phase = _get_bufidx_phase(blk, M_STAGE)
        d_buf_id, d_phase = _get_bufidx_phase(blk, D_STAGE)

        tlx.barrier_wait(qk_fulls[tmem_buf_id], tmem_phase)
        tlx.barrier_wait(m_fulls[m_buf_id], m_phase)

        offs_m = cur_m + tl.arange(0, BLOCK_M1)
        qkT = tlx.local_load(qk_tiles[tmem_buf_id])
        m = tlx.local_load(sM_tiles[m_buf_id])

        # Scale the recomputed score on the fp32 accumulator (K is fed native,
        # matching the 1-CTA _bwd_softmax_iter). tl.minimum bounds the exp2
        # argument for masked/padding columns whose stored LSE is 0 -- pre-scaling
        # K in bf16 instead rounds a large QK^T positive on the row-max column and
        # overflows exp2 to +inf -> NaN grads at high activation magnitude.
        qkT = qkT * (sm_scale / LN2)
        pT = tl.math.exp2(tl.minimum(_sub_f32x2(qkT, m[None, :]), 0.0))
        # Jagged mask: zero out out-of-range key rows (handles sentinel/padding
        # cluster tiles and the boundary K block) and out-of-range query cols.
        pmask = (offs_n[:, None] < klen) & (offs_m[None, :] < qlen)
        if WINDOW_SIZE is not None:
            pmask &= tl.abs(offs_m[None, :] - offs_n[:, None]) <= WINDOW_SIZE
        pT = tl.where(pmask, pT, 0.0)

        ppT = pT.to(do_out_dtype)
        tlx.local_store(p_tiles[tmem_buf_id + P_BUF_OFFSET], ppT)
        # P aliases the QK TMEM region, so qk_empties (which frees that region for
        # reuse) must be signaled after the P store, not before. local_store->TMEM
        # auto-emits tcgen05.wait::st, so p_fulls already observes the completed
        # store; no manual wait needed.
        tlx.barrier_arrive(qk_empties[tmem_buf_id], 1, remote_cta_rank=0)
        tlx.barrier_arrive(p_fulls[tmem_buf_id], 1, remote_cta_rank=0)

        # dS = pT * (dpT - Di)
        tlx.barrier_wait(dp_fulls[tmem_buf_id], tmem_phase)
        dpT = tlx.local_load(dp_tiles[tmem_buf_id])
        tlx.barrier_wait(d_fulls[d_buf_id], d_phase)
        Di = tlx.local_load(sD_tiles[d_buf_id])
        tlx.barrier_arrive(m_empties[m_buf_id])
        tlx.barrier_arrive(d_empties[d_buf_id])
        dsT = _mul_f32x2(pT, _sub_f32x2(dpT, Di[None, :]))
        dsT = dsT.to(q_out_dtype)
        tlx.local_store(dsT_tmem_tiles[ds_buf_id], dsT)
        # dsT aliases the dP TMEM region; dp_empties is arrived after the DSMEM
        # exchange below. local_store->TMEM auto-emits tcgen05.wait::st, so both
        # dsT_tmem_fulls and the TMEM read-back below observe the completed store;
        # no manual wait needed.
        tlx.barrier_arrive(dsT_tmem_fulls[ds_buf_id], 1, remote_cta_rank=0)

        # DSMEM exchange: split dS M-columns into own/peer halves; keep own
        # half locally, ship peer half to peer's ds_tiles.
        _, ds_phase = _get_bufidx_phase(blk, NUM_BUFFERS_DS)
        tlx.barrier_wait(ds_empties[ds_buf_id], ds_phase ^ 1)
        peer_rank = 1 - cluster_cta_rank
        if cluster_cta_rank == 0:
            own_tmem = tlx.local_slice(
                dsT_tmem_tiles[ds_buf_id], [0, 0], [BLOCK_N1, M_SPLIT]
            )
            peer_tmem = tlx.local_slice(
                dsT_tmem_tiles[ds_buf_id], [0, M_SPLIT], [BLOCK_N1, M_SPLIT]
            )
            own_smem = tlx.local_slice(ds_tiles[ds_buf_id], [0, 0], [BLOCK_N1, M_SPLIT])
        else:
            own_tmem = tlx.local_slice(
                dsT_tmem_tiles[ds_buf_id], [0, M_SPLIT], [BLOCK_N1, M_SPLIT]
            )
            peer_tmem = tlx.local_slice(
                dsT_tmem_tiles[ds_buf_id], [0, 0], [BLOCK_N1, M_SPLIT]
            )
            own_smem = tlx.local_slice(
                ds_tiles[ds_buf_id], [BLOCK_N1, 0], [BLOCK_N1, M_SPLIT]
            )
        own_data = tlx.local_load(own_tmem)
        tlx.local_store(own_smem, own_data)
        peer_data = tlx.local_load(peer_tmem)
        tlx.barrier_arrive(dp_empties[tmem_buf_id], 1, remote_cta_rank=0)
        tlx.local_store(ds_xchg_tiles[ds_buf_id], peer_data)
        tlx.fence("async_shared")
        tlx.barrier_expect_bytes(ds_peer_fulls[ds_buf_id], 2 * BLOCK_N1 * M_SPLIT)
        tlx.async_remote_shmem_copy(
            dst=own_smem,
            src=ds_xchg_tiles[ds_buf_id],
            remote_cta_rank=peer_rank,
            barrier=ds_peer_fulls[ds_buf_id],
        )
    return curr_m + num_steps * BLOCK_M1, blk_idx + num_steps


@triton.jit  # pragma: no cover
# Triton TR001: launched through _get_autotune_bwd_2cta_kernel.
def _attn_bwd_ws_2cta(  # noqa: C901, TR001
    desc_q: TensorDescriptor,
    Q_offsets: tl.tensor,
    desc_k: TensorDescriptor,
    K_offsets: tl.tensor,
    desc_v: TensorDescriptor,
    desc_do: TensorDescriptor,
    Out_offsets: tl.tensor,
    desc_dq: TensorDescriptor,
    desc_kt: TensorDescriptor,
    desc_qt: TensorDescriptor,
    desc_dot: TensorDescriptor,
    dQ: tl.tensor,
    dK: tl.tensor,
    dV: tl.tensor,
    sm_scale: float,
    M: tl.tensor,
    D: tl.tensor,
    stride_km: int,
    stride_qh: int,
    stride_kh: int,
    stride_mh: int,
    stride_d: int,
    Z,
    H: int,
    G: int,
    N_CTX_KV,
    tile_to_batch: tl.tensor,
    tile_to_head: tl.tensor,
    tile_to_block: tl.tensor,
    total_valid_tiles: int,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_DO: tl.constexpr,
    NUM_BUFFERS_DS: tl.constexpr,
    NUM_BUFFERS_TMEM: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    DKV_STORE_NCOL: tl.constexpr,
    BROADCAST_Q: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    USE_I64_IDX: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    ENABLE_CLC: tl.constexpr,
    PERSISTENT: tl.constexpr,
) -> None:
    tl.static_assert(NUM_CTAS == 2, "2-CTA kernel requires NUM_CTAS == 2")
    tl.static_assert(HEAD_DIM == BLOCK_D)
    # begin_o below handles both forms, but non-broadcast q with ragged sequence
    # lengths makes begin_o (= begin_q) an unaligned TMA coordinate and faults at
    # runtime with a misaligned address. Fail at compile time until the
    # descriptor coordinates are aligned for that case.
    tl.static_assert(BROADCAST_Q, "2-CTA path is scoped to broadcast_q")

    Q_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_q))
    K_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_k))
    V_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_v))
    DO_BYTES_PER_ELEM: tl.constexpr = tlx.size_of(tlx.dtype_of(desc_do))

    M_STAGE: tl.constexpr = 1
    D_STAGE: tl.constexpr = 2
    DQ_STORE_M: tl.constexpr = BLOCK_M1 // NUM_CTAS
    DQ_SLICE_N: tl.constexpr = HEAD_DIM // EPILOGUE_SUBTILE
    DS_ROWS: tl.constexpr = BLOCK_N1 * NUM_CTAS
    DS_COLS: tl.constexpr = BLOCK_M1 // NUM_CTAS
    P_BUF_IDX: tl.constexpr = 1
    DQ_BUF_IDX: tl.constexpr = 0
    LN2: tl.constexpr = 0.6931471824645996

    prog_id = tl.program_id(0)
    if USE_I64_IDX:
        prog_id = prog_id.to(tl.int64)
    tile_idx = prog_id

    # Persistent (static round-robin) bounds. Both CTAs of a cluster advance
    # tile_idx by num_progs (even, since the grid is cluster-aligned), so each
    # cluster keeps processing adjacent N-block pairs of the same (batch, head)
    # and both ranks stop on the same iteration (total_valid_tiles is even).
    num_progs = tl.num_programs(0)
    total_tiles_2cta = total_valid_tiles
    if USE_I64_IDX:
        num_progs = num_progs.to(tl.int64)
        total_tiles_2cta = total_tiles_2cta.to(tl.int64)

    cluster_cta_rank = tlx.cluster_cta_rank()
    is_leader = cluster_cta_rank == 0

    # =====================================================================
    # Barriers
    # =====================================================================
    k_mma_done = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    k_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    q_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    do_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    do_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    m_fulls = tlx.alloc_barriers(num_barriers=M_STAGE)
    m_empties = tlx.alloc_barriers(num_barriers=M_STAGE)
    d_fulls = tlx.alloc_barriers(num_barriers=D_STAGE)
    d_empties = tlx.alloc_barriers(num_barriers=D_STAGE)
    ds_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM, arrive_count=NUM_CTAS)
    dsT_tmem_fulls = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS_DS, arrive_count=NUM_CTAS
    )

    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    qk_empties = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS_TMEM, arrive_count=NUM_CTAS
    )
    p_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM, arrive_count=NUM_CTAS)
    dp_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dq_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dq_empties = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS_TMEM, arrive_count=NUM_CTAS
    )
    dv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    dv_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV, arrive_count=NUM_CTAS)
    dk_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    dk_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV, arrive_count=NUM_CTAS)
    # dp_empties needs arrivals from both MMA (Dot 4) and compute (after the
    # DSMEM exchange) before Dot 2 can overwrite dp.
    dp_empties = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS_TMEM, arrive_count=NUM_CTAS + 1
    )

    k_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    v_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    kt_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    kt_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    qt_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    qt_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    dot_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    dot_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    ds_peer_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DS)
    ds_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DS)

    # =====================================================================
    # SMEM + TMEM buffers
    # =====================================================================
    k_tiles = tlx.local_alloc(
        (BLOCK_N1, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS_KV
    )
    v_tiles = tlx.local_alloc(
        (BLOCK_N1, HEAD_DIM), tlx.dtype_of(desc_v), NUM_BUFFERS_KV
    )
    q_tiles = tlx.local_alloc(
        (BLOCK_M1, HEAD_DIM // NUM_CTAS), tlx.dtype_of(desc_q), NUM_BUFFERS_Q
    )
    do_tiles = tlx.local_alloc(
        (BLOCK_M1, HEAD_DIM // NUM_CTAS), tlx.dtype_of(desc_do), NUM_BUFFERS_DO
    )
    ds_tiles = tlx.local_alloc((DS_ROWS, DS_COLS), tlx.dtype_of(desc_q), NUM_BUFFERS_DS)

    kt_tiles = tlx.local_alloc(
        (BLOCK_N1 * NUM_CTAS, HEAD_DIM // NUM_CTAS),
        tlx.dtype_of(desc_k),
        NUM_BUFFERS_KV,
    )
    qt_tiles = tlx.local_alloc(
        (BLOCK_M1 // NUM_CTAS, HEAD_DIM), tlx.dtype_of(desc_q), NUM_BUFFERS_Q
    )
    dot_tiles = tlx.local_alloc(
        (BLOCK_M1 // NUM_CTAS, HEAD_DIM), tlx.dtype_of(desc_do), NUM_BUFFERS_DO
    )
    ds_xchg_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1 // NUM_CTAS), tlx.dtype_of(desc_q), NUM_BUFFERS_DS
    )

    sM_tiles = tlx.local_alloc((BLOCK_M1,), tl.float32, M_STAGE)
    sD_tiles = tlx.local_alloc((BLOCK_M1,), tl.float32, D_STAGE)

    dq_store_buf = tlx.local_alloc((DQ_STORE_M, DQ_SLICE_N), tlx.dtype_of(desc_dq), 2)
    # dK/dV epilogue staging (reuse k/v SMEM to fit budget).
    sdv_store_buf = tlx.local_alloc(
        (BLOCK_N1, DKV_STORE_NCOL), tlx.dtype_of(desc_v), NUM_BUFFERS_KV, reuse=v_tiles
    )
    sdk_store_buf = tlx.local_alloc(
        (BLOCK_N1, DKV_STORE_NCOL), tlx.dtype_of(desc_k), NUM_BUFFERS_KV, reuse=k_tiles
    )

    # S/P/dQ share TMEM. P offset to column 64; dQ at column 0.
    qk_p_storage_alias = tlx.storage_alias_spec(storage=tlx.storage_kind.tmem)
    qk_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1),
        tl.float32,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=qk_p_storage_alias,
    )
    p_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1),
        tlx.dtype_of(desc_do),
        2,
        tlx.storage_kind.tmem,
        reuse=qk_p_storage_alias,
    )
    # dP and dS (TMEM) share a slot (sequential lifetime).
    dp_dq_storage_alias = tlx.storage_alias_spec(storage=tlx.storage_kind.tmem)
    dp_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1),
        tl.float32,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=dp_dq_storage_alias,
    )
    dsT_tmem_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1),
        tlx.dtype_of(desc_q),
        NUM_BUFFERS_DS,
        tlx.storage_kind.tmem,
        reuse=dp_dq_storage_alias,
    )
    dv_tiles = tlx.local_alloc(
        (BLOCK_N1, HEAD_DIM), tl.float32, NUM_BUFFERS_KV, tlx.storage_kind.tmem
    )
    dk_tiles = tlx.local_alloc(
        (BLOCK_N1, HEAD_DIM), tl.float32, NUM_BUFFERS_KV, tlx.storage_kind.tmem
    )
    # dQ at column 0; split along M (rows).
    dq_tiles = tlx.local_alloc(
        (BLOCK_M1 // NUM_CTAS, HEAD_DIM),
        tl.float32,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=qk_p_storage_alias,
    )
    dp_dq_storage_alias.set_buffer_overlap(
        tlx.reuse_group(
            dp_tiles,
            dsT_tmem_tiles,
            group_type=tlx.reuse_group_type.shared,
        )
    )

    # CLC: each of the 5 warp groups (compute, reduction, mma, load, relay) on
    # both CTAs is a consumer; the reduction task is the producer.
    NUM_CLC_CONSUMERS: tl.constexpr = 5 * NUM_CTAS
    if ENABLE_CLC:
        clc_context = tlx.clc_create_context(NUM_CLC_CONSUMERS)

    with tlx.async_tasks():
        # ---- compute (default) ----
        with tlx.async_task("default"):
            tile_idx_l = tile_idx
            blk_idx = 0
            kv_tile_idx = 0
            clc_phase_consumer = 0
            has_more_tile = True
            do_out_dtype = tlx.dtype_of(desc_do)
            q_out_dtype = tlx.dtype_of(desc_q)
            DKV_STORE_ITERS: tl.constexpr = HEAD_DIM // DKV_STORE_NCOL
            while has_more_tile:
                off_z, off_h, off_h_kv, off_q_z, pid = bwd_lookup_offsets(
                    tile_to_batch,
                    tile_to_head,
                    tile_to_block,
                    G,
                    tile_idx_l,
                    BROADCAST_Q,
                )
                begin_q = tl.load(Q_offsets + off_q_z)
                end_q = tl.load(Q_offsets + off_q_z + 1)
                qlen = end_q - begin_q
                begin_k = tl.load(K_offsets + off_z)
                end_k = tl.load(K_offsets + off_z + 1)
                klen = end_k - begin_k
                if not BROADCAST_Q:
                    begin_o = begin_q
                else:
                    begin_o = qlen * off_z
                start_n = pid * BLOCK_N1
                off_h2 = off_h.to(tl.int64)
                kv_offset = off_h_kv.to(tl.int64) * stride_kh

                num_steps, _start_m = bwd_calculate_num_steps(
                    qlen, start_n, BLOCK_M1, BLOCK_N1, WINDOW_SIZE
                )

                _curr_m, blk_idx = _bwd_compute_2cta_inner(
                    start_n,
                    klen,
                    qlen,
                    0,
                    blk_idx,
                    num_steps,
                    qk_fulls,
                    qk_tiles,
                    qk_empties,
                    p_tiles,
                    p_fulls,
                    dp_empties,
                    dp_fulls,
                    dp_tiles,
                    ds_tiles,
                    dsT_tmem_tiles,
                    dsT_tmem_fulls,
                    sM_tiles,
                    sD_tiles,
                    m_fulls,
                    m_empties,
                    d_fulls,
                    d_empties,
                    ds_xchg_tiles,
                    ds_peer_fulls,
                    ds_empties,
                    cluster_cta_rank,
                    do_out_dtype,
                    q_out_dtype,
                    sm_scale,
                    NUM_BUFFERS_TMEM=NUM_BUFFERS_TMEM,
                    NUM_BUFFERS_DS=NUM_BUFFERS_DS,
                    BLOCK_M1=BLOCK_M1,
                    BLOCK_N1=BLOCK_N1,
                    NUM_CTAS=NUM_CTAS,
                    M_STAGE=M_STAGE,
                    D_STAGE=D_STAGE,
                    LN2=LN2,
                    WINDOW_SIZE=WINDOW_SIZE,
                    P_BUF_OFFSET=P_BUF_IDX,
                )

                # Epilogue: dV / dK stores (on-device descriptors bounded by end_k).
                kv_buf_id, kv_phase = _get_bufidx_phase(kv_tile_idx, NUM_BUFFERS_KV)
                desc_dv = tl.make_tensor_descriptor(
                    dV,
                    shape=[end_k.to(tl.int32), HEAD_DIM * H],
                    strides=[HEAD_DIM * H, 1],
                    block_shape=[BLOCK_N1, DKV_STORE_NCOL],
                )
                desc_dk = tl.make_tensor_descriptor(
                    dK,
                    shape=[end_k.to(tl.int32), HEAD_DIM * H],
                    strides=[HEAD_DIM * H, 1],
                    block_shape=[BLOCK_N1, DKV_STORE_NCOL],
                )
                tlx.barrier_wait(dv_fulls[kv_buf_id], kv_phase)
                for slice_id in tl.static_range(DKV_STORE_ITERS):
                    dv_slice = tlx.local_slice(
                        dv_tiles[kv_buf_id],
                        [0, slice_id * DKV_STORE_NCOL],
                        [BLOCK_N1, DKV_STORE_NCOL],
                    )
                    dv = tlx.local_load(dv_slice)
                    tlx.local_store(
                        sdv_store_buf[kv_buf_id], dv.to(tlx.dtype_of(desc_dv))
                    )
                    tlx.fence_async_shared()
                    tlx.async_descriptor_store(
                        desc_dv,
                        sdv_store_buf[kv_buf_id],
                        [
                            (begin_k + start_n).to(tl.int32),
                            (kv_offset + slice_id * DKV_STORE_NCOL).to(tl.int32),
                        ],
                    )
                    tlx.async_descriptor_store_wait(0)
                tlx.barrier_arrive(dv_empties[kv_buf_id], 1, remote_cta_rank=0)

                tlx.barrier_wait(dk_fulls[kv_buf_id], kv_phase)
                tlx.barrier_wait(k_mma_done[kv_buf_id], kv_phase)
                for slice_id in tl.static_range(DKV_STORE_ITERS):
                    dk_slice = tlx.local_slice(
                        dk_tiles[kv_buf_id],
                        [0, slice_id * DKV_STORE_NCOL],
                        [BLOCK_N1, DKV_STORE_NCOL],
                    )
                    dk = tlx.local_load(dk_slice)
                    dk *= sm_scale
                    tlx.local_store(
                        sdk_store_buf[kv_buf_id], dk.to(tlx.dtype_of(desc_dk))
                    )
                    tlx.fence_async_shared()
                    tlx.async_descriptor_store(
                        desc_dk,
                        sdk_store_buf[kv_buf_id],
                        [
                            (begin_k + start_n).to(tl.int32),
                            (kv_offset + slice_id * DKV_STORE_NCOL).to(tl.int32),
                        ],
                    )
                    tlx.async_descriptor_store_wait(0)
                tlx.barrier_arrive(k_empties[kv_buf_id])
                tlx.barrier_arrive(dk_empties[kv_buf_id], 1, remote_cta_rank=0)
                kv_tile_idx += 1

                if ENABLE_CLC:
                    tile_idx_l = tlx.clc_consumer(
                        clc_context, clc_phase_consumer, multi_ctas=True
                    )
                    if USE_I64_IDX:
                        tile_idx_l = tile_idx_l.to(tl.int64)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx_l != -1
                elif PERSISTENT:
                    tile_idx_l += num_progs
                    has_more_tile = tile_idx_l < total_tiles_2cta
                else:
                    has_more_tile = False

        # ---- reduction (dQ) — also the CLC producer ----
        with tlx.async_task(num_warps=4, registers=88):
            tile_idx_l = tile_idx
            blk_idx = 0
            clc_phase_consumer = 0
            clc_phase_producer = 1
            has_more_tile = True
            dq_m_offset = cluster_cta_rank * DQ_STORE_M
            while has_more_tile:
                off_z, off_h, off_h_kv, off_q_z, pid = bwd_lookup_offsets(
                    tile_to_batch,
                    tile_to_head,
                    tile_to_block,
                    G,
                    tile_idx_l,
                    BROADCAST_Q,
                )
                begin_q = tl.load(Q_offsets + off_q_z)
                end_q = tl.load(Q_offsets + off_q_z + 1)
                qlen = end_q - begin_q
                start_n = pid * BLOCK_N1
                off_h2 = off_h.to(tl.int64)
                q_col = (off_h2 * stride_qh).to(tl.int32)
                num_steps, _start_m = bwd_calculate_num_steps(
                    qlen, start_n, BLOCK_M1, BLOCK_N1, WINDOW_SIZE
                )
                for it in range(num_steps):
                    blk = blk_idx + it
                    cur_m = it * BLOCK_M1
                    tmem_buf_id, tmem_phase = _get_bufidx_phase(blk, NUM_BUFFERS_TMEM)
                    tlx.barrier_wait(dq_fulls[tmem_buf_id], tmem_phase)
                    dq_full = tlx.local_load(dq_tiles[tmem_buf_id + DQ_BUF_IDX])
                    tlx.barrier_arrive(dq_empties[tmem_buf_id], 1, remote_cta_rank=0)
                    # dq = sm_scale * (dsT @ K). K is now native (was pre-scaled by
                    # sm_scale/LN2, which this scaled back by LN2); match the 1-CTA
                    # dq epilogue and scale by sm_scale directly.
                    dq_full = dq_full * sm_scale
                    dq_slices = _split_n(dq_full, EPILOGUE_SUBTILE)
                    for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                        dq_smem = dq_store_buf[slice_id % 2]
                        tlx.async_descriptor_store_wait(1)
                        tlx.local_store(
                            dq_smem, dq_slices[slice_id].to(tlx.dtype_of(desc_dq))
                        )
                        tlx.fence_async_shared()
                        tlx.async_descriptor_store(
                            desc_dq,
                            dq_smem,
                            [
                                (begin_q + cur_m + dq_m_offset).to(tl.int32),
                                q_col + slice_id * DQ_SLICE_N,
                            ],
                            store_reduce="add",
                        )
                blk_idx += num_steps
                tlx.async_descriptor_store_wait(0)
                if ENABLE_CLC:
                    tlx.clc_producer(clc_context, clc_phase_producer, multi_ctas=True)
                    clc_phase_producer = clc_phase_producer ^ 1
                    tile_idx_l = tlx.clc_consumer(
                        clc_context, clc_phase_consumer, multi_ctas=True
                    )
                    if USE_I64_IDX:
                        tile_idx_l = tile_idx_l.to(tl.int64)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx_l != -1
                elif PERSISTENT:
                    tile_idx_l += num_progs
                    has_more_tile = tile_idx_l < total_tiles_2cta
                else:
                    has_more_tile = False

        # ---- mma ----
        with tlx.async_task(num_warps=1, registers=88):
            tile_idx_l = tile_idx
            blk_idx = 0
            kv_tile_idx = 0
            clc_phase_consumer = 0
            has_more_tile = True
            while has_more_tile:
                off_z, off_h, off_h_kv, off_q_z, pid = bwd_lookup_offsets(
                    tile_to_batch,
                    tile_to_head,
                    tile_to_block,
                    G,
                    tile_idx_l,
                    BROADCAST_Q,
                )
                begin_q = tl.load(Q_offsets + off_q_z)
                end_q = tl.load(Q_offsets + off_q_z + 1)
                qlen = end_q - begin_q
                start_n = pid * BLOCK_N1
                num_steps, _start_m = bwd_calculate_num_steps(
                    qlen, start_n, BLOCK_M1, BLOCK_N1, WINDOW_SIZE
                )
                if is_leader:
                    kv_buf_id, kv_phase = _get_bufidx_phase(kv_tile_idx, NUM_BUFFERS_KV)
                    blk_idx = _bwd_mma_dots_2cta(
                        blk_idx,
                        num_steps,
                        kv_buf_id,
                        kv_phase,
                        k_tiles,
                        v_tiles,
                        q_tiles,
                        do_tiles,
                        qk_tiles,
                        qk_fulls,
                        qk_empties,
                        p_tiles,
                        p_fulls,
                        dp_tiles,
                        dp_fulls,
                        dp_empties,
                        dv_tiles,
                        dv_fulls,
                        dv_empties,
                        dk_tiles,
                        dk_fulls,
                        dk_empties,
                        dq_tiles,
                        dq_fulls,
                        dq_empties,
                        ds_tiles,
                        ds_fulls,
                        dsT_tmem_tiles,
                        dsT_tmem_fulls,
                        do_fulls,
                        do_empties,
                        q_fulls,
                        q_empties,
                        k_mma_done,
                        qt_tiles,
                        dot_tiles,
                        kt_tiles,
                        qt_fulls,
                        qt_empties,
                        dot_fulls,
                        dot_empties,
                        kt_fulls,
                        kt_empties,
                        k_fulls,
                        v_fulls,
                        ds_empties,
                        NUM_BUFFERS_Q=NUM_BUFFERS_Q,
                        NUM_BUFFERS_DO=NUM_BUFFERS_DO,
                        NUM_BUFFERS_TMEM=NUM_BUFFERS_TMEM,
                        NUM_BUFFERS_DS=NUM_BUFFERS_DS,
                        BLOCK_N1=BLOCK_N1,
                        DQ_BUF_OFFSET=DQ_BUF_IDX,
                        P_BUF_OFFSET=P_BUF_IDX,
                    )
                kv_tile_idx += 1
                if ENABLE_CLC:
                    tile_idx_l = tlx.clc_consumer(
                        clc_context, clc_phase_consumer, multi_ctas=True
                    )
                    if USE_I64_IDX:
                        tile_idx_l = tile_idx_l.to(tl.int64)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx_l != -1
                elif PERSISTENT:
                    tile_idx_l += num_progs
                    has_more_tile = tile_idx_l < total_tiles_2cta
                else:
                    has_more_tile = False

        # ---- load ----
        with tlx.async_task(num_warps=1, registers=88):
            tile_idx_l = tile_idx
            blk_idx = 0
            kv_tile_idx = 0
            clc_phase_consumer = 0
            has_more_tile = True
            while has_more_tile:
                off_z, off_h, off_h_kv, off_q_z, pid = bwd_lookup_offsets(
                    tile_to_batch,
                    tile_to_head,
                    tile_to_block,
                    G,
                    tile_idx_l,
                    BROADCAST_Q,
                )
                begin_q = tl.load(Q_offsets + off_q_z)
                end_q = tl.load(Q_offsets + off_q_z + 1)
                qlen = end_q - begin_q
                begin_k = tl.load(K_offsets + off_z)
                end_k = tl.load(K_offsets + off_z + 1)
                klen = end_k - begin_k
                if not BROADCAST_Q:
                    begin_o = begin_q
                else:
                    begin_o = qlen * off_z
                start_n = pid * BLOCK_N1
                off_h2 = off_h.to(tl.int64)
                off_chz = off_h2 * stride_mh + begin_o
                num_steps, _start_m = bwd_calculate_num_steps(
                    qlen, start_n, BLOCK_M1, BLOCK_N1, WINDOW_SIZE
                )
                blk_idx = _bwd_load_2cta(
                    blk_idx,
                    begin_q,
                    begin_o,
                    begin_k,
                    off_h2,
                    off_h_kv,
                    stride_qh,
                    stride_kh,
                    off_chz,
                    start_n,
                    num_steps,
                    kv_tile_idx,
                    desc_k,
                    desc_v,
                    desc_q,
                    desc_do,
                    desc_kt,
                    desc_qt,
                    desc_dot,
                    M,
                    D,
                    k_tiles,
                    v_tiles,
                    q_tiles,
                    do_tiles,
                    sM_tiles,
                    sD_tiles,
                    k_empties,
                    q_fulls,
                    q_empties,
                    do_fulls,
                    do_empties,
                    m_fulls,
                    m_empties,
                    d_fulls,
                    d_empties,
                    k_fulls,
                    v_fulls,
                    kt_tiles,
                    kt_fulls,
                    kt_empties,
                    qt_tiles,
                    qt_fulls,
                    qt_empties,
                    dot_tiles,
                    dot_fulls,
                    dot_empties,
                    cluster_cta_rank,
                    is_leader,
                    K_BYTES_PER_ELEM=K_BYTES_PER_ELEM,
                    V_BYTES_PER_ELEM=V_BYTES_PER_ELEM,
                    Q_BYTES_PER_ELEM=Q_BYTES_PER_ELEM,
                    DO_BYTES_PER_ELEM=DO_BYTES_PER_ELEM,
                    BLOCK_M1=BLOCK_M1,
                    BLOCK_N1=BLOCK_N1,
                    NUM_BUFFERS_KV=NUM_BUFFERS_KV,
                    NUM_BUFFERS_Q=NUM_BUFFERS_Q,
                    NUM_BUFFERS_DO=NUM_BUFFERS_DO,
                    M_STAGE=M_STAGE,
                    D_STAGE=D_STAGE,
                    HEAD_DIM=HEAD_DIM,
                    NUM_CTAS=NUM_CTAS,
                )
                kv_tile_idx += 1
                if ENABLE_CLC:
                    tile_idx_l = tlx.clc_consumer(
                        clc_context, clc_phase_consumer, multi_ctas=True
                    )
                    if USE_I64_IDX:
                        tile_idx_l = tile_idx_l.to(tl.int64)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx_l != -1
                elif PERSISTENT:
                    tile_idx_l += num_progs
                    has_more_tile = tile_idx_l < total_tiles_2cta
                else:
                    has_more_tile = False

        # ---- relay: wait peer DSMEM, signal ds_fulls to leader ----
        with tlx.async_task(num_warps=1, registers=40):
            tile_idx_l = tile_idx
            blk_idx = 0
            clc_phase_consumer = 0
            has_more_tile = True
            while has_more_tile:
                off_z, off_h, off_h_kv, off_q_z, pid = bwd_lookup_offsets(
                    tile_to_batch,
                    tile_to_head,
                    tile_to_block,
                    G,
                    tile_idx_l,
                    BROADCAST_Q,
                )
                begin_q = tl.load(Q_offsets + off_q_z)
                end_q = tl.load(Q_offsets + off_q_z + 1)
                qlen = end_q - begin_q
                start_n = pid * BLOCK_N1
                num_steps, _start_m = bwd_calculate_num_steps(
                    qlen, start_n, BLOCK_M1, BLOCK_N1, WINDOW_SIZE
                )
                for it in range(num_steps):
                    blk = blk_idx + it
                    ds_buf_id_relay, ds_phase_relay = _get_bufidx_phase(
                        blk, NUM_BUFFERS_DS
                    )
                    tlx.barrier_wait(ds_peer_fulls[ds_buf_id_relay], ds_phase_relay)
                    tlx.fence("async_shared")
                    tlx.barrier_arrive(ds_fulls[ds_buf_id_relay], 1, remote_cta_rank=0)
                blk_idx += num_steps
                if ENABLE_CLC:
                    tile_idx_l = tlx.clc_consumer(
                        clc_context, clc_phase_consumer, multi_ctas=True
                    )
                    if USE_I64_IDX:
                        tile_idx_l = tile_idx_l.to(tl.int64)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx_l != -1
                elif PERSISTENT:
                    tile_idx_l += num_progs
                    has_more_tile = tile_idx_l < total_tiles_2cta
                else:
                    has_more_tile = False


def expect_contiguous(x: torch.Tensor) -> torch.Tensor:
    if x is not None and not x.is_contiguous():
        return x.contiguous()
    return x


@torch.library.custom_op("ads_mkl::tlx_jagged_flash_attention", mutates_args=())
def tlx_jagged_flash_attention(  # noqa: C901
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_offset: torch.Tensor,
    key_offset: torch.Tensor,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    output_offset: torch.Tensor | None = None,
    sm_scale: float | None = None,
    window_size: int | None = None,
    broadcast_q: bool = False,
    use_on_device_tma: bool = False,
    cpu_query_offset: torch.Tensor | None = None,
    cpu_key_offset: torch.Tensor | None = None,
    ad_to_request_offset: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
    HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    extra_kern_args = {}
    ad_to_request_offset = expect_contiguous(ad_to_request_offset)
    if ad_to_request_offset is not None:
        raise NotImplementedError(
            "ad_to_request_offset selects the D-tiled IKBO forward, which is not "
            "part of this package. Call without ad_to_request_offset and with "
            "head_dim <= 128."
        )

    # The forward issues a single BLOCK_D = next_pow2(head_dim); at head_dim > 128
    # that is >= 256 and the q/kv/o SMEM exceeds the ~227KB SM100 cap (head_dim=256
    # needs ~459KB) -> allocation failure before launch. head_dim > 128 requires the
    # D-tiled forward, which is not part of this package.
    assert HEAD_DIM_Q <= 128, (
        f"tlx_jagged_flash_attention supports head_dim<=128, got "
        f"{HEAD_DIM_Q}; head_dim>128 exceeds the SM100 SMEM cap in the non-D-tiled "
        f"path."
    )

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM_Q)

    BLOCK_D = max(triton.next_power_of_2(HEAD_DIM_Q), 16)

    if broadcast_q:
        BATCH: int = key_offset.size(0) - 1
    else:
        BATCH: int = query_offset.size(0) - 1

    q = expect_contiguous(query)
    k = expect_contiguous(key)
    v = expect_contiguous(value)
    o = torch.empty(
        (
            BATCH * q.shape[0] if broadcast_q else q.shape[0],
            q.shape[1],
            HEAD_DIM_Q,
        ),
        device=q.device,
        dtype=q.dtype,
    )
    # intermediate tensors
    if broadcast_q:
        M = torch.empty(
            (query.shape[1], BATCH * query.shape[0]),
            device=query.device,
            dtype=torch.float32,
        )
    else:
        M = torch.empty(
            (query.shape[1], query.shape[0]), device=query.device, dtype=torch.float32
        )
    N_CTX = max_seq_len_q
    H = q.shape[1]
    G = q.shape[1] // k.shape[1]
    assert q.shape[1] % k.shape[1] == 0
    y_dim_q = q.shape[0]
    y_dim_kv = k.shape[0]
    x_dim_kv = HEAD_DIM_Q * H // G
    x_dim_q = HEAD_DIM_Q * H

    if not use_on_device_tma:
        dummy_block = [1, 1]
        desc_q = TensorDescriptor(
            q,
            shape=[y_dim_q, x_dim_q],
            strides=[q.stride(0), q.stride(2)],
            block_shape=dummy_block,
        )
        desc_v = TensorDescriptor(
            v,
            shape=[y_dim_kv, x_dim_kv],
            strides=[v.stride(0), v.stride(2)],
            block_shape=dummy_block,
        )
        desc_k = TensorDescriptor(
            k,
            shape=[y_dim_kv, x_dim_kv],
            strides=[k.stride(0), k.stride(2)],
            block_shape=dummy_block,
        )

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    NUM_SMS = get_num_sms() or 1000000

    if cpu_query_offset is None or cpu_key_offset is None or window_size is not None:
        # Sliding window load balancing is not supported yet.
        enable_load_balancing = False
        (
            valid_tiles_m_start,
            valid_tiles_m_end,
            valid_tiles_b,
            valid_tiles_h,
            sm_offsets,
        ) = None, None, None, None, None
    else:
        enable_load_balancing = True
        (
            valid_tiles_m_start,
            valid_tiles_m_end,
            valid_tiles_b,
            valid_tiles_h,
            sm_offsets,
        ) = compute_balanced_tiles(
            cpu_query_offset, cpu_key_offset, BATCH, 256, 128, H, broadcast_q
        )

    def grid(META):
        total_ctas = triton.cdiv(max_seq_len_q, META["BLOCK_M"]) * BATCH * H
        if not META["USE_CLC"]:
            if META["ENABLE_LOAD_BALANCING"]:
                total_ctas = valid_tiles_b.shape[0]
            total_ctas = min(NUM_SMS, total_ctas)

        return (total_ctas, 1, 1)

    kernel_fn = _get_autotune_fwd_kernel(_attn_fwd_ws)
    kernel_fn[grid](
        q if use_on_device_tma else desc_q,
        query_offset,
        k if use_on_device_tma else desc_k,
        key_offset,
        v if use_on_device_tma else desc_v,
        o,
        M,
        sm_scale,
        q.stride(1),  # query stride H
        k.stride(1),  # key stride H
        o.stride(1),  # output stride H
        M.stride(0),  # M stride H
        BATCH,
        H,
        G,
        N_CTX=N_CTX,
        total_len_q=q.shape[0],  # B*M
        total_len_kv=k.shape[0],  # B*M
        HEAD_DIM=HEAD_DIM_K,  #
        BLOCK_D=BLOCK_D,
        USE_ON_DEVICE_TMA=use_on_device_tma,
        BROADCAST_Q=broadcast_q,
        WINDOW_SIZE=window_size,
        ENABLE_LOAD_BALANCING=enable_load_balancing,
        valid_tiles_b=valid_tiles_b,
        valid_tiles_m_start=valid_tiles_m_start,
        valid_tiles_m_end=valid_tiles_m_end,
        valid_tiles_h=valid_tiles_h,
        sm_offsets=sm_offsets,
        USE_I64_IDX=should_use_i64_idx(q, k, v, o),
        **extra_kern_args,
    )
    return o, M


# FLOP counting functions
def _unpack_nested_shapes_meta(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_out: Optional[torch.Tensor] = None,
    cum_seq_q: torch.Tensor,
    cum_seq_k: torch.Tensor,
    max_q: int,
    max_k: int,
) -> Generator[
    Tuple[
        Tuple[int, int, int, int],
        Tuple[int, int, int, int],
        Tuple[int, int, int, int],
        Optional[Tuple[int, int, int, int]],
    ],
    None,
    None,
]:
    _, h_q, d_q = query.shape
    _, h_k, d_k = key.shape
    _, h_v, d_v = value.shape

    b = cum_seq_q.size(0) - 1

    avg_seq_len_q = query.size(0) // b
    avg_seq_len_k = key.size(0) // b

    for _ in range(b):
        new_query_shape = (1, h_q, avg_seq_len_q, d_q)
        new_key_shape = (1, h_k, avg_seq_len_k, d_k)
        new_value_shape = (1, h_v, avg_seq_len_k, d_v)
        new_grad_out_shape = new_query_shape if grad_out is not None else None
        yield new_query_shape, new_key_shape, new_value_shape, new_grad_out_shape

    return


@register_flop_formula(torch.ops.ads_mkl.tlx_jagged_flash_attention, get_raw=True)
def jagged_flash_attention_forward_flop(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_offset: torch.Tensor,
    key_offset: torch.Tensor,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    output_offset: torch.Tensor | None = None,
    sm_scale: float | None = None,
    window_size: int | None = None,
    broadcast_q: bool = False,
    use_on_device_tma: bool = False,
    cpu_query_offset: torch.Tensor | None = None,
    cpu_key_offset: torch.Tensor | None = None,
    ad_to_request_offset: torch.Tensor | None = None,
    *args,
    **kwargs,
) -> int:
    """Count flops for the forward."""
    fused_qkv = key is None and value is None
    fused_kv = key is not None and value is None
    if fused_qkv:
        HEAD_DIM = query.shape[-1] // 3
        query, key, value = query.split(HEAD_DIM, dim=-1)
    elif fused_kv:
        HEAD_DIM = key.shape[-1] // 2
        key, value = key.split(HEAD_DIM, dim=-1)
    bs_q = query_offset.size(0) - 1
    bs_k = key_offset.size(0) - 1
    if ad_to_request_offset is not None:
        sizes = (
            (
                (
                    1,
                    query.shape[1],
                    int(query_offset[i + 1] - query_offset[i]),
                    query.shape[2],
                ),
                (
                    1,
                    key.shape[1],
                    int(
                        key_offset[ad_to_request_offset[i] + 1]
                        - key_offset[ad_to_request_offset[i]]
                    ),
                    key.shape[2],
                ),
                (
                    1,
                    value.shape[1],
                    int(
                        key_offset[ad_to_request_offset[i] + 1]
                        - key_offset[ad_to_request_offset[i]]
                    ),
                    value.shape[2],
                ),
                None,
            )
            for i in range(bs_q)
        )
    elif bs_q != bs_k:
        # broadcast q bs to k bs
        assert bs_k % bs_q == 0
        assert broadcast_q
        query_length = query_offset[1]
        query_offset = torch.arange(bs_k + 1, device=query.device) * query_length
        query = query.repeat_interleave(bs_k // bs_q, dim=0)
        if query.is_meta:
            sizes = _unpack_nested_shapes_meta(
                query=query,
                key=key,
                value=value,
                cum_seq_q=query_offset,
                cum_seq_k=key_offset,
                max_q=max_seq_len_q,
                max_k=max_seq_len_kv,
            )
        else:
            sizes = _unpack_flash_attention_nested_shapes(
                query=query,
                key=key,
                value=value,
                cum_seq_q=query_offset,
                cum_seq_k=key_offset,
                max_q=max_seq_len_q,
                max_k=max_seq_len_kv,
            )
    elif query.is_meta:
        sizes = _unpack_nested_shapes_meta(
            query=query,
            key=key,
            value=value,
            cum_seq_q=query_offset,
            cum_seq_k=key_offset,
            max_q=max_seq_len_q,
            max_k=max_seq_len_kv,
        )
    else:
        sizes = _unpack_flash_attention_nested_shapes(
            query=query,
            key=key,
            value=value,
            cum_seq_q=query_offset,
            cum_seq_k=key_offset,
            max_q=max_seq_len_q,
            max_k=max_seq_len_kv,
        )
    if window_size is not None:
        # replace number of keys and values
        sizes = (
            (
                query_shape,
                (_b2, _h2, 2 * window_size + 1, _d2),
                (_b3, _h3, 2 * window_size + 1, d_v),
                grad_out_shape,
            )
            for (
                query_shape,
                (_b2, _h2, s_k, _d2),
                (_b3, _h3, _s3, d_v),
                grad_out_shape,
            ) in sizes
        )
    return sum(
        sdpa_flop_count(query_shape, key_shape, value_shape)
        for query_shape, key_shape, value_shape, _ in sizes
    )


def _jagged_flash_attention_setup_context(ctx, inputs, output):
    (
        query,
        key,
        value,
        query_offset,
        key_offset,
        max_seq_len_q,
        max_seq_len_kv,
        output_offset,
        sm_scale,
        window_size,
        broadcast_q,
        _,
        _,
        cpu_key_offset,
        ad_to_request_offset,
    ) = inputs
    o, M = output
    HEAD_DIM_Q = query.shape[-1]
    HEAD_DIM_K = key.shape[-1]
    BLOCK_D = max(triton.next_power_of_2(HEAD_DIM_K), 32)
    if broadcast_q:
        BATCH = key_offset.size(0) - 1
    else:
        BATCH = query_offset.size(0) - 1
    nheads = query.shape[1]
    nheads_k = key.shape[1]
    ctx.save_for_backward(
        query, key, value, o, M, query_offset, key_offset, output_offset
    )
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM_Q)
    ctx.sm_scale = sm_scale
    ctx.HEAD_DIM = HEAD_DIM_K
    ctx.BLOCK_D = BLOCK_D
    ctx.BATCH = BATCH
    ctx.N_HEAD = nheads
    ctx.G = nheads // nheads_k
    ctx.window_size = window_size
    ctx.broadcast_q = broadcast_q
    ctx.N_CTX = max_seq_len_q
    ctx.N_CTX_KV = max_seq_len_kv
    ctx.BROADCAST_Q = broadcast_q
    ctx.WINDOW_SIZE = window_size
    ctx.cpu_key_offset = cpu_key_offset
    ctx.ad_to_request_offset = ad_to_request_offset


def precompute_static_bwd_varlen_tiles(
    num_head: int,
    k_offsets: torch.Tensor,
    n_block_size: int,
    device: torch.device,
    num_sms: int,
    num_ctas: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Precompute tile-to-(batch, head, block) lookup tables for backward varlen.

    Uses block-innermost ordering for L2 cache sharing: consecutive tile indices
    share the same (batch, head) pair, so adjacent CTAs access the same K/V data.

    When ``num_ctas > 1`` (cluster / collaborative-MMA path) each (batch, head)'s
    N-block count is padded up to a multiple of ``num_ctas`` with sentinel tiles
    (``block = n_blocks`` so ``start_n >= klen`` and the tile's contributions are
    masked out). This keeps every cluster-aligned group of ``num_ctas`` tile
    indices within the same (batch, head) and over adjacent N-blocks, which is the
    collaborative-MMA pairing requirement.

    Returns (tile_to_batch, tile_to_head, tile_to_block, total_tiles).
    """
    cpu_offsets = k_offsets.cpu()
    num_batch = len(cpu_offsets) - 1

    tile_batch = []
    tile_head = []
    tile_block = []
    for batch in range(num_batch):
        seqlen = int(cpu_offsets[batch + 1] - cpu_offsets[batch])
        n_blocks = (seqlen + n_block_size - 1) // n_block_size
        # Pad each (batch, head)'s block count to a multiple of num_ctas so
        # cluster pairs never straddle a head boundary. Sentinel tiles use
        # block == n_blocks (their start_n lands at/after klen).
        n_blocks_padded = n_blocks
        if num_ctas > 1 and n_blocks % num_ctas != 0:
            n_blocks_padded = ((n_blocks + num_ctas - 1) // num_ctas) * num_ctas
        for head in range(num_head):
            for block in range(n_blocks_padded):
                tile_batch.append(batch)
                tile_head.append(head)
                tile_block.append(block)

    total_tiles = len(tile_batch)
    if total_tiles == 0:
        empty = torch.zeros(1, dtype=torch.int32, device=device)
        return empty, empty, empty, 0

    t_batch = (
        torch.tensor(tile_batch, dtype=torch.int32)
        .pin_memory()
        .to(device=device, non_blocking=True)
    )
    t_head = (
        torch.tensor(tile_head, dtype=torch.int32)
        .pin_memory()
        .to(device=device, non_blocking=True)
    )
    t_block = (
        torch.tensor(tile_block, dtype=torch.int32)
        .pin_memory()
        .to(device=device, non_blocking=True)
    )
    return t_batch, t_head, t_block, total_tiles


@torch.library.custom_op("ads_mkl::tlx_jagged_flash_attention_bwd", mutates_args=())
def tlx_jagged_flash_attention_backward(  # noqa: C901
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    M: torch.Tensor,
    sm_scale: float,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    BATCH: int,
    N_HEAD: int,
    G: int,
    N_CTX: int,
    N_CTX_KV: int,
    HEAD_DIM: int,
    BLOCK_D: int,
    output_offset: torch.Tensor | None = None,
    window_size: int | None = None,
    broadcast_q: bool = False,
    cpu_key_offset: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert do.is_contiguous()
    assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
    assert G == 1
    dq_dtype = (
        torch.float32 if broadcast_q else q.dtype
    )  # to ensure the precision under atomic_add
    dq = torch.zeros(q.shape, device=q.device, dtype=dq_dtype)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    def preproc_grid(meta):
        return (
            triton.cdiv(N_CTX, meta["BLOCK_M"]),  # tiles along N (K/V)
            BATCH * N_HEAD,
        )

    delta = torch.empty_like(M)
    do = expect_contiguous(do)
    if output_offset is None:
        output_offset = q_offsets

    kernel_fn = _get_autotune_bwd_preprocess_kernel(_attn_bwd_preprocess)
    kernel_fn[preproc_grid](
        o,
        output_offset,
        do,  #
        delta,  #
        M.stride(0),
        N_HEAD,
        N_CTX,  #
        BLOCK_D=BLOCK_D,
        HEAD_DIM=HEAD_DIM,  #
    )

    dummy_block = [1, 1]
    desc_k = TensorDescriptor(
        k,
        shape=[k.shape[0], HEAD_DIM * N_HEAD],
        strides=[HEAD_DIM * N_HEAD, 1],
        block_shape=dummy_block,
    )
    desc_v = TensorDescriptor(
        v,
        shape=[v.shape[0], HEAD_DIM * N_HEAD],
        strides=[HEAD_DIM * N_HEAD, 1],
        block_shape=dummy_block,
    )
    desc_q = TensorDescriptor(
        q,
        shape=[q.shape[0], HEAD_DIM * N_HEAD],
        strides=[HEAD_DIM * N_HEAD, 1],
        block_shape=dummy_block,
    )
    desc_do = TensorDescriptor(
        do,
        shape=[do.shape[0], HEAD_DIM * N_HEAD],
        strides=[HEAD_DIM * N_HEAD, 1],
        block_shape=dummy_block,
    )
    desc_dq = TensorDescriptor(
        dq,
        shape=[dq.shape[0], HEAD_DIM * N_HEAD],
        strides=[HEAD_DIM * N_HEAD, 1],
        block_shape=dummy_block,
    )

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    NUM_SMS = (
        get_num_sms() or 1000000
    )  # if num sms is None, use a large number so that it is a no-op

    # Route the PMA case (broadcast_q, head_dim 128, single query group, no
    # sliding window, load balancing on) through the 2-CTA collaborative-MMA
    # backward. Everything else falls back to the general 1-CTA kernel, which
    # handles all shapes. Hardcoded via JFA_BWD_USE_2CTA for A/B testing.
    use_2cta = (
        JFA_BWD_USE_2CTA
        and broadcast_q
        and HEAD_DIM == 128
        and G == 1
        and window_size is None
        and cpu_key_offset is not None
    )
    num_ctas_bwd = 2 if use_2cta else 1

    # Software load balancing: precompute tile-to-(batch,head,block) lookup tables
    if cpu_key_offset is not None:
        # Use BLOCK_N1=128 as default (autotuner may pick different, but 128 is
        # the only value in configs_bwd_tlx so this is safe)
        enable_load_balancing = True
        BLOCK_N1_DEFAULT = 128
        tile_to_batch, tile_to_head, tile_to_block, lb_total_tiles = (
            precompute_static_bwd_varlen_tiles(
                N_HEAD,
                k_offsets,
                BLOCK_N1_DEFAULT,
                q.device,
                NUM_SMS,
                num_ctas=num_ctas_bwd,
            )
        )
    else:
        enable_load_balancing = False
        # Dummy tensors (unused when ENABLE_LB=False)
        tile_to_batch = None
        tile_to_head = None
        tile_to_block = None
        lb_total_tiles = 0

    if use_2cta:
        # Transposed B-operand descriptors over the same tensors. Block shapes
        # are set by _bwd_host_descriptor_pre_hook_tlx_2cta. K is passed native
        # (the 2-CTA kernel scales the score in-register, like the 1-CTA path).
        desc_kt = TensorDescriptor(
            k,
            shape=[k.shape[0], HEAD_DIM * N_HEAD],
            strides=[HEAD_DIM * N_HEAD, 1],
            block_shape=dummy_block,
        )
        desc_qt = TensorDescriptor(
            q,
            shape=[q.shape[0], HEAD_DIM * N_HEAD],
            strides=[HEAD_DIM * N_HEAD, 1],
            block_shape=dummy_block,
        )
        desc_dot = TensorDescriptor(
            do,
            shape=[do.shape[0], HEAD_DIM * N_HEAD],
            strides=[HEAD_DIM * N_HEAD, 1],
            block_shape=dummy_block,
        )

        # Cluster grid. ctas_per_cga=(2,1,1) (from the config) pairs adjacent
        # program ids into clusters. Non-persistent: one program per (already
        # cluster-padded) tile. Persistent: grid sized to the SM count (rounded
        # down to a cluster multiple) and each cluster round-robins over tiles.
        def grid_2cta(meta):
            if meta["PERSISTENT"] and not meta["ENABLE_CLC"]:
                ncta = (NUM_SMS // meta["NUM_CTAS"]) * meta["NUM_CTAS"]
                return (min(ncta, lb_total_tiles), 1, 1)
            return (lb_total_tiles, 1, 1)

        kernel_fn = _get_autotune_bwd_2cta_kernel(_attn_bwd_ws_2cta)
        kernel_fn[grid_2cta](
            desc_q,
            q_offsets,
            desc_k,
            k_offsets,
            desc_v,
            desc_do,
            output_offset,
            desc_dq,
            desc_kt,
            desc_qt,
            desc_dot,
            dq,
            dk,
            dv,
            sm_scale,
            M,
            delta,
            k.stride(0),
            q.stride(1),
            k.stride(1),
            M.stride(0),
            q.stride(2),
            BATCH,
            N_HEAD,
            G,
            N_CTX_KV=N_CTX_KV,
            tile_to_batch=tile_to_batch,
            tile_to_head=tile_to_head,
            tile_to_block=tile_to_block,
            total_valid_tiles=lb_total_tiles,
            BLOCK_D=BLOCK_D,
            HEAD_DIM=HEAD_DIM,
            BROADCAST_Q=broadcast_q,
            WINDOW_SIZE=window_size,
            USE_I64_IDX=should_use_i64_idx(q, k, v, o),
        )

        if broadcast_q:
            dq = dq.to(q.dtype)
        return dq, dk, dv

    def grid(meta):
        if enable_load_balancing:
            if meta["ENABLE_CLC"]:
                return (lb_total_tiles, 1, 1)
            else:
                return (min(NUM_SMS, lb_total_tiles), 1, 1)
        else:
            if meta["ENABLE_CLC"]:
                return (
                    triton.cdiv(N_CTX_KV, meta["BLOCK_N1"]) * BATCH * N_HEAD,
                    1,
                    1,
                )
            else:
                return (
                    min(
                        NUM_SMS,
                        triton.cdiv(N_CTX_KV, meta["BLOCK_N1"]) * BATCH * N_HEAD,
                    ),
                    1,
                    1,
                )

    kernel_fn = _get_autotune_bwd_kernel(_attn_bwd_ws)
    kernel_fn[grid](
        desc_q,
        q_offsets,
        desc_k,
        k_offsets,
        desc_v,
        desc_do,
        output_offset,
        desc_dq,
        dq,
        dk,
        dv,  #
        sm_scale,
        M,
        delta,  #
        k.stride(0),
        q.stride(1),
        k.stride(1),
        M.stride(0),
        q.stride(2),
        BATCH,
        N_HEAD,
        G,
        N_CTX_KV=N_CTX_KV,  # max_seq_len_kv
        tile_to_batch=tile_to_batch,
        tile_to_head=tile_to_head,
        tile_to_block=tile_to_block,
        total_valid_tiles=lb_total_tiles,
        BLOCK_D=BLOCK_D,
        HEAD_DIM=HEAD_DIM,  #
        BROADCAST_Q=broadcast_q,
        WINDOW_SIZE=window_size,
        USE_I64_IDX=should_use_i64_idx(q, k, v, o),
        ENABLE_LB=enable_load_balancing,
    )

    if broadcast_q:
        dq = dq.to(q.dtype)

    return dq, dk, dv


@register_flop_formula(torch.ops.ads_mkl.tlx_jagged_flash_attention_bwd, get_raw=True)
def jfa_backward_flop(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    M: torch.Tensor,
    sm_scale: float,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    BATCH: int,
    N_HEAD: int,
    G: int,
    N_CTX: int,
    N_CTX_KV: int,
    HEAD_DIM: int,
    BLOCK_D: int,
    output_offset: torch.Tensor | None = None,
    window_size: int | None = None,
    broadcast_q: bool = False,
    *args,
    **kwargs,
) -> int:
    """Count flops for the backward."""
    bs_q = q_offsets.size(0) - 1
    bs_k = k_offsets.size(0) - 1
    if bs_q != bs_k:
        # broadcast q bs to k bs
        assert bs_k % bs_q == 0
        assert broadcast_q
        query_length = q_offsets[1]
        q_offsets = torch.arange(bs_k + 1, device=q.device) * query_length
        q = q.repeat_interleave(bs_k // bs_q, dim=0)
    if q.is_meta:
        shapes = _unpack_nested_shapes_meta(
            query=q,
            key=k,
            value=v,
            grad_out=do,
            cum_seq_q=q_offsets,
            cum_seq_k=k_offsets,
            max_q=N_CTX,
            max_k=N_CTX_KV,
        )
    else:
        shapes = _unpack_flash_attention_nested_shapes(
            query=q,
            key=k,
            value=v,
            grad_out=do,
            cum_seq_q=q_offsets,
            cum_seq_k=k_offsets,
            max_q=N_CTX,
            max_k=N_CTX_KV,
        )
    if window_size is not None:
        # replace number of keys and values
        shapes = (
            (
                query_shape,
                (_b2, _h2, 2 * window_size + 1, _d2),
                (_b3, _h3, 2 * window_size + 1, d_v),
                grad_out_shape,
            )
            for (
                query_shape,
                (_b2, _h2, s_k, _d2),
                (_b3, _h3, _s3, d_v),
                grad_out_shape,
            ) in shapes
        )
    return sum(
        sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape)
        for query_shape, key_shape, value_shape, grad_out_shape in shapes
    )


def _tlx_jagged_flash_attention_backward(ctx, do, _):
    if ctx.ad_to_request_offset is not None:
        raise RuntimeError("tlx_jagged_flash_attention IKBO backward is not supported")
    query, key, value, o, M, query_offset, key_offset, output_offset = ctx.saved_tensors
    dq, dk, dv = tlx_jagged_flash_attention_backward(
        do,
        query,
        key,
        value,
        o,
        M,
        ctx.sm_scale,
        query_offset,
        key_offset,
        BATCH=ctx.BATCH,
        N_HEAD=ctx.N_HEAD,
        G=ctx.G,
        N_CTX=ctx.N_CTX,
        N_CTX_KV=ctx.N_CTX_KV,
        HEAD_DIM=ctx.HEAD_DIM,
        BLOCK_D=ctx.BLOCK_D,
        output_offset=output_offset,
        window_size=ctx.WINDOW_SIZE,
        broadcast_q=ctx.BROADCAST_Q,
        cpu_key_offset=ctx.cpu_key_offset,
    )
    return (dq, dk, dv, *((None,) * 12))


if not isinstance(
    tlx_jagged_flash_attention, types.FunctionType
):  # In case of duplicate registration, `@custom_triton_op` returns the base function
    tlx_jagged_flash_attention.register_autograd(
        _tlx_jagged_flash_attention_backward,
        setup_context=_jagged_flash_attention_setup_context,
    )


@torch.jit.script_if_tracing
@custom_register_kernel("ads_mkl::tlx_jagged_flash_attention", "cpu")
def cpu_jagged_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_offset: torch.Tensor,
    key_offset: torch.Tensor,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    output_offset: torch.Tensor | None = None,
    sm_scale: float | None = None,
    window_size: int | None = None,
    broadcast_q: bool = False,
    use_on_device_tma: bool = False,
    cpu_query_offset: torch.Tensor | None = None,
    cpu_key_offset: torch.Tensor | None = None,
    ad_to_request_offset: torch.Tensor | None = None,
) -> torch.Tensor:
    if not broadcast_q:
        M = torch.zeros(
            (query.shape[1], query.shape[0]),
            device=query.device,
            dtype=torch.float32,
        )
    else:
        BATCH = key_offset.size(0) - 1
        M = torch.zeros(
            (query.shape[1], BATCH * query.shape[0]),
            device=query.device,
            dtype=torch.float32,
        )
    assert not broadcast_q, "broadcast_q unsupported"
    HEAD_DIM_Q = query.shape[-1]

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM_Q)

    bs = query_offset.size(0) - 1
    o = torch.zeros_like(query)

    for i in range(bs):
        q_start = query_offset[i]
        q_end = query_offset[i + 1]
        q_i = query[q_start:q_end, :, :].transpose(0, 1)  # h, n, d

        if ad_to_request_offset is None:
            kv_i = i
        else:
            kv_i = ad_to_request_offset[i]
        kv_start = key_offset[kv_i]
        kv_end = key_offset[kv_i + 1]
        k_i = key[kv_start:kv_end, :, :].transpose(0, 1)
        v_i = value[kv_start:kv_end, :, :].transpose(0, 1)  # h, m, d

        s = torch.matmul(q_i, k_i.transpose(-2, -1))  # h, n, m
        s *= sm_scale

        if window_size is not None:
            n, m = q_i.shape[1], k_i.shape[1]

            s = torch.where(
                (
                    torch.abs(
                        torch.arange(n, device=s.device)[:, None]
                        - torch.arange(m, device=s.device)[None, :]
                    )
                    <= window_size
                )[None, :, :],
                s,
                float("-inf"),
            ).to(dtype=s.dtype, device=s.device)
        p = F.softmax(s, dim=-1)
        out = torch.matmul(p, v_i)  # h, n, m @ h m, d = h, n, d
        out = out.transpose(0, 1).contiguous()
        o[q_start:q_end, :, :] = out

    return o, M


@torch.library.register_fake("ads_mkl::tlx_jagged_flash_attention")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_offset: torch.Tensor,
    key_offset: torch.Tensor,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    output_offset: torch.Tensor | None = None,
    sm_scale: float | None = None,
    window_size: int | None = None,
    broadcast_q: bool = False,
    use_on_device_tma: bool = False,
    cpu_query_offset: torch.Tensor | None = None,
    cpu_key_offset: torch.Tensor | None = None,
    ad_to_request_offset: torch.Tensor | None = None,
):
    if not broadcast_q:
        M = torch.zeros(
            (query.shape[1], query.shape[0]),
            device=query.device,
            dtype=torch.float32,
        )
        return torch.zeros_like(query), M
    else:
        BATCH = key_offset.size(0) - 1
        M = torch.zeros(
            (query.shape[1], BATCH * query.shape[0]),
            device=query.device,
            dtype=torch.float32,
        )
        return torch.zeros(
            (BATCH * query.shape[0], query.shape[1], query.shape[2]),
            dtype=query.dtype,
            device=query.device,
        ), M


@torch.library.register_fake("ads_mkl::tlx_jagged_flash_attention_bwd")
def __(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    M: torch.Tensor,
    sm_scale: float,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    BATCH: int,
    N_HEAD: int,
    G: int,
    N_CTX: int,
    N_CTX_KV: int,
    HEAD_DIM: int,
    BLOCK_D: int,
    output_offset: torch.Tensor | None = None,
    window_size: int | None = None,
    broadcast_q: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)


@torch.fx.wrap
def jagged_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_offset: torch.Tensor,
    key_offset: torch.Tensor,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    output_offset: torch.Tensor | None = None,
    sm_scale: float | None = None,
    window_size: int | None = None,
    broadcast_q: bool = False,
    use_on_device_tma: bool = False,
) -> torch.Tensor:
    if torch.jit.is_tracing() or torch.jit.is_scripting():
        return cpu_jagged_flash_attention(
            query=query,
            key=key,
            value=value,
            query_offset=query_offset,
            key_offset=key_offset,
            max_seq_len_q=max_seq_len_q,
            max_seq_len_kv=max_seq_len_kv,
            output_offset=output_offset,
            sm_scale=sm_scale,
            window_size=window_size,
            broadcast_q=broadcast_q,
            use_on_device_tma=use_on_device_tma,
        )[0]
    query = expect_contiguous(query)
    key = expect_contiguous(key)
    value = expect_contiguous(value)
    return torch.ops.ads_mkl.tlx_jagged_flash_attention(
        query=query,
        key=key,
        value=value,
        query_offset=query_offset,
        key_offset=key_offset,
        max_seq_len_q=max_seq_len_q,
        max_seq_len_kv=max_seq_len_kv,
        output_offset=output_offset,
        sm_scale=sm_scale,
        window_size=window_size,
        broadcast_q=broadcast_q,
        use_on_device_tma=use_on_device_tma,
    )[0]
