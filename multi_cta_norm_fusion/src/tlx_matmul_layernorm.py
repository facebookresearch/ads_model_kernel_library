# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors

import os
from functools import lru_cache
from typing import Any, Dict

import torch
import triton  # @manual=//triton:triton
import triton.language as tl  # @manual=//triton:triton
import triton.language.extra.tlx as tlx  # @manual=//triton:triton
from torch._inductor.runtime.triton_compat import libdevice
from torch.utils.flop_counter import register_flop_formula
from triton.tools.tensor_descriptor import TensorDescriptor  # @manual=//triton:triton


def _make_pre_hook(
    block_m: int, block_k: int, block_n: int, num_mma_groups: int, epilogue_subtile: int
):
    """Create a pre_hook closure for the given block sizes."""
    block_m_split = block_m // num_mma_groups

    def hook(nargs: Dict[str, Any]) -> None:
        nargs["a_desc"].block_shape = [block_m_split, block_k]
        nargs["b_desc"].block_shape = [block_k, block_n]
        nargs["out_desc"].block_shape = [block_m_split, block_n // epilogue_subtile]

    return hook


def _make_configs(N: int):  # noqa: C901
    """Generate autotune configs sweeping BLOCK_N, MMA groups, and tile sizes."""
    if os.environ.get("ADS_MKL_DISABLE_AUTOTUNE", "0") == "1":
        assert N % 256 == 0
        return [
            triton.Config(
                {
                    "BLOCK_M": 256,
                    "BLOCK_K": 64,
                    "BLOCK_N": 256,
                    "NUM_SMEM_BUFFERS": 3,
                    "NUM_TMEM_BUFFERS": 1,
                    "GROUP_SIZE_M": 8,
                    "NUM_MMA_GROUPS": 2,
                    "EPILOGUE_SUBTILE": 4,
                    "INTERLEAVE_EPILOGUE": 0,
                },
                num_warps=4,
                num_stages=1,
                ctas_per_cga=(N // 256, 1, 1),
                pre_hook=_make_pre_hook(
                    256,
                    64,
                    256,
                    2,
                    4,
                ),
            )
        ]
    configs = []
    # Each BLOCK_N implies a specific num_ctas = N // BLOCK_N at runtime.
    # We generate configs for all supported num_ctas values that BLOCK_N could produce.
    # The prune function filters to the correct one based on actual N.
    for block_n in [128, 256]:
        # local_alloc buffer count must be power of 2
        if N % block_n != 0:
            continue
        num_ctas = N // block_n
        dsmem_slots = num_ctas * 2
        if dsmem_slots & (dsmem_slots - 1) != 0:
            continue
        if num_ctas > 8:
            continue
        for block_m in [128, 256]:
            for block_k in [64, 128]:
                for num_mma_groups in [1, 2]:
                    # Hardware constraint: BLOCK_M / NUM_MMA_GROUPS <= 128
                    if block_m // num_mma_groups > 128:
                        continue
                    for gm in [8]:
                        for num_smem in [3, 4]:
                            for num_tmem in [1, 2]:
                                for subtile in [1, 4, 8]:
                                    for num_warps in [4, 8]:
                                        if block_n % subtile != 0:
                                            continue
                                        for interleave in [0, 1]:
                                            # INTERLEAVE_EPILOGUE requires NUM_MMA_GROUPS == 2
                                            if interleave and num_mma_groups != 2:
                                                continue
                                            configs.append(
                                                triton.Config(
                                                    {
                                                        "BLOCK_M": block_m,
                                                        "BLOCK_K": block_k,
                                                        "BLOCK_N": block_n,
                                                        "NUM_SMEM_BUFFERS": num_smem,
                                                        "NUM_TMEM_BUFFERS": num_tmem,
                                                        "GROUP_SIZE_M": gm,
                                                        "NUM_MMA_GROUPS": num_mma_groups,
                                                        "EPILOGUE_SUBTILE": subtile,
                                                        "INTERLEAVE_EPILOGUE": interleave,
                                                    },
                                                    num_warps=num_warps,
                                                    num_stages=1,
                                                    ctas_per_cga=(num_ctas, 1, 1),
                                                    pre_hook=_make_pre_hook(
                                                        block_m,
                                                        block_k,
                                                        block_n,
                                                        num_mma_groups,
                                                        subtile,
                                                    ),
                                                )
                                            )
    return configs


def _prune_configs(configs, named_args, **kwargs):
    """Keep only configs where BLOCK_N evenly divides N and ctas_per_cga matches."""
    N = named_args["N"]
    M = named_args["M"]
    pruned = []
    for c in configs:
        block_n = c.kwargs["BLOCK_N"]
        block_m = c.kwargs["BLOCK_M"]
        if N % block_n != 0:
            continue
        num_ctas = N // block_n
        if c.ctas_per_cga[0] != num_ctas:
            continue
        # local_alloc buffer count must be power of 2
        dsmem_slots = num_ctas * 2  # NUM_DSMEM_BUFFERS=2
        if dsmem_slots & (dsmem_slots - 1) != 0:
            continue
        # BLOCK_M must not exceed M (avoid OOB with MMA groups)
        if block_m > M:
            continue
        pruned.append(c)
    return pruned


# ============================================================================
# Unified CLC Persistent Kernel with NUM_MMA_GROUPS
# ============================================================================


@triton.jit
def fused_matmul_layernorm_kernel(  # noqa: C901
    a_desc: Any,
    b_desc: Any,
    out_desc: Any,
    ln_weight_ptr,
    ln_bias_ptr,
    mean_ptr,
    rstd_ptr,
    eps: float,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    INTERLEAVE_EPILOGUE: tl.constexpr,
    SKIP_NORM: tl.constexpr = False,
) -> None:
    NUM_REDUCTION_CTAS: tl.constexpr = N // BLOCK_N
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS
    SLICE_N: tl.constexpr = BLOCK_N // EPILOGUE_SUBTILE
    NUM_DSMEM_BUFFERS: tl.constexpr = 2
    NUM_WARP_GROUPS: tl.constexpr = 4
    NUM_CLC_CONSUMERS: tl.constexpr = NUM_WARP_GROUPS * NUM_REDUCTION_CTAS

    # ---- Allocations ----
    # A buffers: per MMA group (each loads [BLOCK_M_SPLIT, BLOCK_K])
    buffers_A = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_K), tl.bfloat16, NUM_SMEM_BUFFERS * NUM_MMA_GROUPS
    )
    # B buffers: shared across groups
    buffers_B = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.bfloat16, NUM_SMEM_BUFFERS)
    # TMEM: per MMA group
    tmem_buffers = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_N),
        tl.float32,
        NUM_TMEM_BUFFERS * NUM_MMA_GROUPS,
        tlx.storage_kind.tmem,
    )
    # DSMEM reduction: per-group rows (BLOCK_M_SPLIT), since norm is row-independent
    sum_reduction_buf = tlx.local_alloc(
        (BLOCK_M_SPLIT, 1), tl.float32, NUM_REDUCTION_CTAS * NUM_DSMEM_BUFFERS
    )
    sum_sq_reduction_buf = tlx.local_alloc(
        (BLOCK_M_SPLIT, 1), tl.float32, NUM_REDUCTION_CTAS * NUM_DSMEM_BUFFERS
    )

    # ---- Barriers ----
    # A barriers: per group
    A_smem_empty_bars = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1
    )
    A_smem_full_bars = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1
    )
    # B barriers: shared
    B_smem_empty_bars = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS, arrive_count=1
    )
    B_smem_full_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    # TMEM barriers: per group
    tmem_full_bars = tlx.alloc_barriers(
        num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1
    )
    tmem_empty_bars = tlx.alloc_barriers(
        num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=EPILOGUE_SUBTILE
    )
    # Reduction barriers
    reduction_barriers = tlx.alloc_barriers(num_barriers=NUM_DSMEM_BUFFERS)
    consumer_barriers = tlx.alloc_barriers(
        num_barriers=NUM_DSMEM_BUFFERS, arrive_count=NUM_REDUCTION_CTAS - 1
    )

    # ---- CLC context ----
    clc_context = tlx.clc_create_context(NUM_CLC_CONSUMERS)

    with tlx.async_tasks():
        # ---- Task 0: Load (TMA) — CLC persistent ----
        with tlx.async_task(num_warps=1, registers=24):
            cta_cluster_rank = tlx.cluster_cta_rank()
            tile_id = tl.program_id(axis=0) // NUM_REDUCTION_CTAS
            k_tiles = tl.cdiv(K, BLOCK_K)
            num_m_tiles = tl.cdiv(M, BLOCK_M)
            offs_bn = cta_cluster_rank * BLOCK_N

            load_phase_a = 0
            load_phase_b = 0
            processed_k_iters = 0
            clc_phase_consumer = 0

            while tile_id != -1:
                if tile_id < num_m_tiles:
                    offs_am = tile_id * BLOCK_M
                    for k in range(0, k_tiles):
                        offs_k = k * BLOCK_K

                        # Load A sub-tiles for each MMA group
                        for g in tl.static_range(NUM_MMA_GROUPS):
                            a_buf = (
                                g * NUM_SMEM_BUFFERS
                                + (processed_k_iters + k) % NUM_SMEM_BUFFERS
                            )
                            tlx.barrier_wait(A_smem_empty_bars[a_buf], load_phase_a ^ 1)
                            tlx.barrier_expect_bytes(
                                A_smem_full_bars[a_buf],
                                2 * BLOCK_M_SPLIT * BLOCK_K,
                            )
                            tlx.async_descriptor_load(
                                a_desc,
                                buffers_A[a_buf],
                                [offs_am + g * BLOCK_M_SPLIT, offs_k],
                                A_smem_full_bars[a_buf],
                            )

                        # Load B (shared across groups)
                        b_buf = (processed_k_iters + k) % NUM_SMEM_BUFFERS
                        tlx.barrier_wait(B_smem_empty_bars[b_buf], load_phase_b ^ 1)
                        tlx.barrier_expect_bytes(
                            B_smem_full_bars[b_buf],
                            2 * BLOCK_K * BLOCK_N,
                        )
                        tlx.async_descriptor_load(
                            b_desc,
                            buffers_B[b_buf],
                            [offs_k, offs_bn],
                            B_smem_full_bars[b_buf],
                        )

                        load_phase_a = load_phase_a ^ (
                            (processed_k_iters + k) % NUM_SMEM_BUFFERS
                            == NUM_SMEM_BUFFERS - 1
                        )
                        load_phase_b = load_phase_b ^ (b_buf == NUM_SMEM_BUFFERS - 1)

                    processed_k_iters += k_tiles

                tile_id = tlx.clc_consumer(
                    clc_context, clc_phase_consumer, multi_ctas=True
                )
                tile_id = tile_id // NUM_REDUCTION_CTAS if tile_id != -1 else -1
                clc_phase_consumer = clc_phase_consumer ^ 1

            tlx.cluster_barrier()

        # ---- Task 1: MMA — CLC persistent ----
        with tlx.async_task(num_warps=1, registers=24):
            tile_id = tl.program_id(axis=0) // NUM_REDUCTION_CTAS
            k_tiles = tl.cdiv(K, BLOCK_K)
            num_m_tiles = tl.cdiv(M, BLOCK_M)

            dot_phase_b = 0
            dot_phase_a = 0
            tmem_write_phase = 1
            cur_tmem_buf = 0
            processed_k_iters = 0
            clc_phase_consumer = 0

            while tile_id != -1:
                if tile_id < num_m_tiles:
                    # Wait for TMEM empty for all groups
                    for g in tl.static_range(NUM_MMA_GROUPS):
                        acc_buf = g * NUM_TMEM_BUFFERS + cur_tmem_buf
                        tlx.barrier_wait(tmem_empty_bars[acc_buf], tmem_write_phase)
                    tmem_write_phase = tmem_write_phase ^ (
                        cur_tmem_buf == NUM_TMEM_BUFFERS - 1
                    )

                    for k in range(0, k_tiles):
                        b_buf = (processed_k_iters + k) % NUM_SMEM_BUFFERS
                        # Wait B once per K-tile
                        tlx.barrier_wait(B_smem_full_bars[b_buf], dot_phase_b)

                        # Dot for each MMA group
                        for g in tl.static_range(NUM_MMA_GROUPS):
                            a_buf = (
                                g * NUM_SMEM_BUFFERS
                                + (processed_k_iters + k) % NUM_SMEM_BUFFERS
                            )
                            acc_buf = g * NUM_TMEM_BUFFERS + cur_tmem_buf

                            tlx.barrier_wait(A_smem_full_bars[a_buf], dot_phase_a)
                            # Last group signals both A and B consumed
                            if g == NUM_MMA_GROUPS - 1:
                                tlx.async_dot(
                                    buffers_A[a_buf],
                                    buffers_B[b_buf],
                                    tmem_buffers[acc_buf],
                                    use_acc=k > 0,
                                    mBarriers=[
                                        A_smem_empty_bars[a_buf],
                                        B_smem_empty_bars[b_buf],
                                    ],
                                    out_dtype=tl.float32,
                                )
                            else:
                                tlx.async_dot(
                                    buffers_A[a_buf],
                                    buffers_B[b_buf],
                                    tmem_buffers[acc_buf],
                                    use_acc=k > 0,
                                    mBarriers=[A_smem_empty_bars[a_buf]],
                                    out_dtype=tl.float32,
                                )

                        dot_phase_b = dot_phase_b ^ (b_buf == NUM_SMEM_BUFFERS - 1)
                        dot_phase_a = dot_phase_a ^ (
                            (processed_k_iters + k) % NUM_SMEM_BUFFERS
                            == NUM_SMEM_BUFFERS - 1
                        )

                    # Wait for last B to be consumed before signaling TMEM full
                    last_b_buf = (processed_k_iters + k_tiles - 1) % NUM_SMEM_BUFFERS
                    last_dot_phase_b = dot_phase_b ^ (
                        last_b_buf == NUM_SMEM_BUFFERS - 1
                    )
                    tlx.barrier_wait(B_smem_empty_bars[last_b_buf], last_dot_phase_b)

                    # Signal TMEM full for all groups
                    for g in tl.static_range(NUM_MMA_GROUPS):
                        acc_buf = g * NUM_TMEM_BUFFERS + cur_tmem_buf
                        tlx.barrier_arrive(tmem_full_bars[acc_buf], 1)

                    cur_tmem_buf = (cur_tmem_buf + 1) % NUM_TMEM_BUFFERS
                    processed_k_iters += k_tiles

                tile_id = tlx.clc_consumer(
                    clc_context, clc_phase_consumer, multi_ctas=True
                )
                tile_id = tile_id // NUM_REDUCTION_CTAS if tile_id != -1 else -1
                clc_phase_consumer = clc_phase_consumer ^ 1

            tlx.cluster_barrier()

        # ---- Task 2: Epilogue — CLC persistent + producer ----
        with tlx.async_task("default"):
            cta_cluster_rank = tlx.cluster_cta_rank()
            tile_id = tl.program_id(axis=0) // NUM_REDUCTION_CTAS
            num_m_tiles = tl.cdiv(M, BLOCK_M)

            cross_cta_bytes: tl.constexpr = (
                BLOCK_M_SPLIT * tlx.size_of(tl.float32) * (NUM_REDUCTION_CTAS - 1)
            )
            merged_cross_cta_bytes: tl.constexpr = 2 * cross_cta_bytes

            tmem_read_phase = 0
            cur_tmem_buf = 0
            clc_phase_producer = 1
            clc_phase_consumer = 0
            buffer_id = 0
            producer_phase = 0
            consumer_phase = 1

            while tile_id != -1:
                tlx.clc_producer(clc_context, clc_phase_producer, multi_ctas=True)
                clc_phase_producer = clc_phase_producer ^ 1

                if tile_id < num_m_tiles:
                    if INTERLEAVE_EPILOGUE:
                        # ============================================================
                        # Interleaved epilogue (requires NUM_MMA_GROUPS == 2)
                        # Alternates TMA stores between group 0 and group 1 to
                        # distribute memory traffic across different M-row addresses.
                        # ============================================================
                        acc_buf_0 = 0 * NUM_TMEM_BUFFERS + cur_tmem_buf
                        acc_buf_1 = 1 * NUM_TMEM_BUFFERS + cur_tmem_buf
                        acc_tmem_0 = tmem_buffers[acc_buf_0]
                        acc_tmem_1 = tmem_buffers[acc_buf_1]
                        offs_am_0 = tile_id * BLOCK_M + 0 * BLOCK_M_SPLIT
                        offs_am_1 = tile_id * BLOCK_M + 1 * BLOCK_M_SPLIT

                        # Wait for both groups' TMEM to be ready
                        tlx.barrier_wait(tmem_full_bars[acc_buf_0], tmem_read_phase)
                        tlx.barrier_wait(tmem_full_bars[acc_buf_1], tmem_read_phase)

                        if SKIP_NORM:
                            # Interleaved matmul-only stores: g0s0, g1s0, g0s1, g1s1...
                            sub0 = tlx.local_slice(
                                acc_tmem_0, [0, 0], [BLOCK_M_SPLIT, SLICE_N]
                            )
                            result0 = tlx.local_load(sub0)
                            tlx.barrier_arrive(tmem_empty_bars[acc_buf_0], 1)
                            out_desc.store(
                                [offs_am_0, cta_cluster_rank * BLOCK_N],
                                result0.to(tl.bfloat16),
                            )

                            sub1 = tlx.local_slice(
                                acc_tmem_1, [0, 0], [BLOCK_M_SPLIT, SLICE_N]
                            )
                            result1 = tlx.local_load(sub1)
                            tlx.barrier_arrive(tmem_empty_bars[acc_buf_1], 1)
                            out_desc.store(
                                [offs_am_1, cta_cluster_rank * BLOCK_N],
                                result1.to(tl.bfloat16),
                            )

                            for s in tl.static_range(1, EPILOGUE_SUBTILE):
                                sub0 = tlx.local_slice(
                                    acc_tmem_0,
                                    [0, s * SLICE_N],
                                    [BLOCK_M_SPLIT, SLICE_N],
                                )
                                result0 = tlx.local_load(sub0)
                                tlx.barrier_arrive(tmem_empty_bars[acc_buf_0], 1)
                                out_desc.store(
                                    [
                                        offs_am_0,
                                        cta_cluster_rank * BLOCK_N + s * SLICE_N,
                                    ],
                                    result0.to(tl.bfloat16),
                                )

                                sub1 = tlx.local_slice(
                                    acc_tmem_1,
                                    [0, s * SLICE_N],
                                    [BLOCK_M_SPLIT, SLICE_N],
                                )
                                result1 = tlx.local_load(sub1)
                                tlx.barrier_arrive(tmem_empty_bars[acc_buf_1], 1)
                                out_desc.store(
                                    [
                                        offs_am_1,
                                        cta_cluster_rank * BLOCK_N + s * SLICE_N,
                                    ],
                                    result1.to(tl.bfloat16),
                                )
                        else:
                            # ---- Pass 1: accumulate sums for group 0 ----
                            local_sum_0 = tl.zeros((BLOCK_M_SPLIT, 1), dtype=tl.float32)
                            local_sum_sq_0 = tl.zeros(
                                (BLOCK_M_SPLIT, 1), dtype=tl.float32
                            )
                            for s in tl.static_range(EPILOGUE_SUBTILE):
                                sub = tlx.local_slice(
                                    acc_tmem_0,
                                    [0, s * SLICE_N],
                                    [BLOCK_M_SPLIT, SLICE_N],
                                )
                                acc_slice = tlx.local_load(sub)
                                local_sum_0 += tl.sum(acc_slice, axis=1, keep_dims=True)
                                local_sum_sq_0 += tl.sum(
                                    acc_slice * acc_slice, axis=1, keep_dims=True
                                )

                            # ---- DSMEM exchange for group 0 ----
                            buf_id_g0 = buffer_id
                            tlx.barrier_wait(
                                consumer_barriers[buf_id_g0], consumer_phase
                            )
                            tlx.barrier_expect_bytes(
                                reduction_barriers[buf_id_g0],
                                size=merged_cross_cta_bytes,
                            )
                            buf_offset_0 = buf_id_g0 * NUM_REDUCTION_CTAS
                            tlx.local_store(
                                sum_reduction_buf[buf_offset_0 + cta_cluster_rank],
                                local_sum_0,
                            )
                            tlx.local_store(
                                sum_sq_reduction_buf[buf_offset_0 + cta_cluster_rank],
                                local_sum_sq_0,
                            )
                            for i in tl.static_range(NUM_REDUCTION_CTAS):
                                if cta_cluster_rank != i:
                                    tlx.async_remote_shmem_store(
                                        dst=sum_reduction_buf[
                                            buf_offset_0 + cta_cluster_rank
                                        ],
                                        src=local_sum_0,
                                        remote_cta_rank=i,
                                        barrier=reduction_barriers[buf_id_g0],
                                    )
                                    tlx.async_remote_shmem_store(
                                        dst=sum_sq_reduction_buf[
                                            buf_offset_0 + cta_cluster_rank
                                        ],
                                        src=local_sum_sq_0,
                                        remote_cta_rank=i,
                                        barrier=reduction_barriers[buf_id_g0],
                                    )
                            tlx.barrier_wait(
                                reduction_barriers[buf_id_g0], phase=producer_phase
                            )
                            global_sum_0 = tl.zeros(
                                (BLOCK_M_SPLIT, 1), dtype=tl.float32
                            )
                            global_sum_sq_0 = tl.zeros(
                                (BLOCK_M_SPLIT, 1), dtype=tl.float32
                            )
                            for i in tl.static_range(NUM_REDUCTION_CTAS):
                                global_sum_0 += tlx.local_load(
                                    tlx.local_view(sum_reduction_buf, buf_offset_0 + i)
                                )
                                global_sum_sq_0 += tlx.local_load(
                                    tlx.local_view(
                                        sum_sq_reduction_buf, buf_offset_0 + i
                                    )
                                )
                            mean_0 = global_sum_0 / N
                            var_0 = global_sum_sq_0 / N - mean_0 * mean_0
                            rstd_0 = libdevice.rsqrt(var_0 + eps)

                            # Advance buffer_id after g0 exchange
                            buffer_id = buffer_id ^ 1
                            if buffer_id == 0:
                                consumer_phase = consumer_phase ^ 1
                                producer_phase = producer_phase ^ 1

                            # ---- Pass 1: accumulate sums for group 1 ----
                            local_sum_1 = tl.zeros((BLOCK_M_SPLIT, 1), dtype=tl.float32)
                            local_sum_sq_1 = tl.zeros(
                                (BLOCK_M_SPLIT, 1), dtype=tl.float32
                            )
                            for s in tl.static_range(EPILOGUE_SUBTILE):
                                sub = tlx.local_slice(
                                    acc_tmem_1,
                                    [0, s * SLICE_N],
                                    [BLOCK_M_SPLIT, SLICE_N],
                                )
                                acc_slice = tlx.local_load(sub)
                                local_sum_1 += tl.sum(acc_slice, axis=1, keep_dims=True)
                                local_sum_sq_1 += tl.sum(
                                    acc_slice * acc_slice, axis=1, keep_dims=True
                                )

                            # ---- DSMEM exchange for group 1 ----
                            buf_id_g1 = buffer_id
                            tlx.barrier_wait(
                                consumer_barriers[buf_id_g1], consumer_phase
                            )
                            tlx.barrier_expect_bytes(
                                reduction_barriers[buf_id_g1],
                                size=merged_cross_cta_bytes,
                            )
                            buf_offset_1 = buf_id_g1 * NUM_REDUCTION_CTAS
                            tlx.local_store(
                                sum_reduction_buf[buf_offset_1 + cta_cluster_rank],
                                local_sum_1,
                            )
                            tlx.local_store(
                                sum_sq_reduction_buf[buf_offset_1 + cta_cluster_rank],
                                local_sum_sq_1,
                            )
                            for i in tl.static_range(NUM_REDUCTION_CTAS):
                                if cta_cluster_rank != i:
                                    tlx.async_remote_shmem_store(
                                        dst=sum_reduction_buf[
                                            buf_offset_1 + cta_cluster_rank
                                        ],
                                        src=local_sum_1,
                                        remote_cta_rank=i,
                                        barrier=reduction_barriers[buf_id_g1],
                                    )
                                    tlx.async_remote_shmem_store(
                                        dst=sum_sq_reduction_buf[
                                            buf_offset_1 + cta_cluster_rank
                                        ],
                                        src=local_sum_sq_1,
                                        remote_cta_rank=i,
                                        barrier=reduction_barriers[buf_id_g1],
                                    )
                            tlx.barrier_wait(
                                reduction_barriers[buf_id_g1], phase=producer_phase
                            )
                            global_sum_1 = tl.zeros(
                                (BLOCK_M_SPLIT, 1), dtype=tl.float32
                            )
                            global_sum_sq_1 = tl.zeros(
                                (BLOCK_M_SPLIT, 1), dtype=tl.float32
                            )
                            for i in tl.static_range(NUM_REDUCTION_CTAS):
                                global_sum_1 += tlx.local_load(
                                    tlx.local_view(sum_reduction_buf, buf_offset_1 + i)
                                )
                                global_sum_sq_1 += tlx.local_load(
                                    tlx.local_view(
                                        sum_sq_reduction_buf, buf_offset_1 + i
                                    )
                                )
                            mean_1 = global_sum_1 / N
                            var_1 = global_sum_sq_1 / N - mean_1 * mean_1
                            rstd_1 = libdevice.rsqrt(var_1 + eps)

                            # Advance buffer_id after g1 exchange
                            buffer_id = buffer_id ^ 1
                            if buffer_id == 0:
                                consumer_phase = consumer_phase ^ 1
                                producer_phase = producer_phase ^ 1

                            # ---- Pass 2: interleaved normalize+store ----
                            # First slice: g0s0, g1s0
                            sub0 = tlx.local_slice(
                                acc_tmem_0, [0, 0], [BLOCK_M_SPLIT, SLICE_N]
                            )
                            acc0 = tlx.local_load(sub0)
                            tlx.barrier_arrive(tmem_empty_bars[acc_buf_0], 1)
                            ln_w_0 = tl.load(
                                ln_weight_ptr
                                + cta_cluster_rank * BLOCK_N
                                + tl.arange(0, SLICE_N)
                            ).to(tl.float32)
                            ln_b_0 = tl.load(
                                ln_bias_ptr
                                + cta_cluster_rank * BLOCK_N
                                + tl.arange(0, SLICE_N)
                            ).to(tl.float32)
                            norm0 = (acc0 - mean_0) * rstd_0
                            out0 = (norm0 * ln_w_0[None, :] + ln_b_0[None, :]).to(
                                tl.bfloat16
                            )
                            out_desc.store(
                                [offs_am_0, cta_cluster_rank * BLOCK_N], out0
                            )

                            sub1 = tlx.local_slice(
                                acc_tmem_1, [0, 0], [BLOCK_M_SPLIT, SLICE_N]
                            )
                            acc1 = tlx.local_load(sub1)
                            tlx.barrier_arrive(tmem_empty_bars[acc_buf_1], 1)
                            norm1 = (acc1 - mean_1) * rstd_1
                            out1 = (norm1 * ln_w_0[None, :] + ln_b_0[None, :]).to(
                                tl.bfloat16
                            )
                            out_desc.store(
                                [offs_am_1, cta_cluster_rank * BLOCK_N], out1
                            )

                            # Remaining slices: interleaved g0, g1
                            for s in tl.static_range(1, EPILOGUE_SUBTILE):
                                ln_w_s = tl.load(
                                    ln_weight_ptr
                                    + cta_cluster_rank * BLOCK_N
                                    + s * SLICE_N
                                    + tl.arange(0, SLICE_N)
                                ).to(tl.float32)
                                ln_b_s = tl.load(
                                    ln_bias_ptr
                                    + cta_cluster_rank * BLOCK_N
                                    + s * SLICE_N
                                    + tl.arange(0, SLICE_N)
                                ).to(tl.float32)

                                sub0 = tlx.local_slice(
                                    acc_tmem_0,
                                    [0, s * SLICE_N],
                                    [BLOCK_M_SPLIT, SLICE_N],
                                )
                                acc0 = tlx.local_load(sub0)
                                tlx.barrier_arrive(tmem_empty_bars[acc_buf_0], 1)
                                norm0 = (acc0 - mean_0) * rstd_0
                                out0 = (norm0 * ln_w_s[None, :] + ln_b_s[None, :]).to(
                                    tl.bfloat16
                                )
                                out_desc.store(
                                    [
                                        offs_am_0,
                                        cta_cluster_rank * BLOCK_N + s * SLICE_N,
                                    ],
                                    out0,
                                )

                                sub1 = tlx.local_slice(
                                    acc_tmem_1,
                                    [0, s * SLICE_N],
                                    [BLOCK_M_SPLIT, SLICE_N],
                                )
                                acc1 = tlx.local_load(sub1)
                                tlx.barrier_arrive(tmem_empty_bars[acc_buf_1], 1)
                                norm1 = (acc1 - mean_1) * rstd_1
                                out1 = (norm1 * ln_w_s[None, :] + ln_b_s[None, :]).to(
                                    tl.bfloat16
                                )
                                out_desc.store(
                                    [
                                        offs_am_1,
                                        cta_cluster_rank * BLOCK_N + s * SLICE_N,
                                    ],
                                    out1,
                                )

                            # Store mean/rstd for both groups
                            if cta_cluster_rank == 0:
                                row_offsets_0 = (
                                    tile_id * BLOCK_M
                                    + 0 * BLOCK_M_SPLIT
                                    + tl.arange(0, BLOCK_M_SPLIT)
                                )
                                tl.store(
                                    mean_ptr + row_offsets_0,
                                    tl.reshape(mean_0, (BLOCK_M_SPLIT,)),
                                )
                                tl.store(
                                    rstd_ptr + row_offsets_0,
                                    tl.reshape(rstd_0, (BLOCK_M_SPLIT,)),
                                )
                                row_offsets_1 = (
                                    tile_id * BLOCK_M
                                    + 1 * BLOCK_M_SPLIT
                                    + tl.arange(0, BLOCK_M_SPLIT)
                                )
                                tl.store(
                                    mean_ptr + row_offsets_1,
                                    tl.reshape(mean_1, (BLOCK_M_SPLIT,)),
                                )
                                tl.store(
                                    rstd_ptr + row_offsets_1,
                                    tl.reshape(rstd_1, (BLOCK_M_SPLIT,)),
                                )

                            # Arrive consumer barriers for both groups' DSMEM buffers
                            for i in tl.static_range(NUM_REDUCTION_CTAS):
                                if cta_cluster_rank != i:
                                    tlx.barrier_arrive(
                                        consumer_barriers[buf_id_g0],
                                        arrive_count=1,
                                        remote_cta_rank=i,
                                    )
                                    tlx.barrier_arrive(
                                        consumer_barriers[buf_id_g1],
                                        arrive_count=1,
                                        remote_cta_rank=i,
                                    )

                    else:
                        # ============================================================
                        # Sequential epilogue (original path)
                        # Process each MMA group fully before the next.
                        # ============================================================
                        for g in tl.static_range(NUM_MMA_GROUPS):
                            acc_buf = g * NUM_TMEM_BUFFERS + cur_tmem_buf
                            acc_tmem = tmem_buffers[acc_buf]
                            tlx.barrier_wait(tmem_full_bars[acc_buf], tmem_read_phase)

                            if SKIP_NORM:
                                for s in tl.static_range(EPILOGUE_SUBTILE):
                                    sub = tlx.local_slice(
                                        acc_tmem,
                                        [0, s * SLICE_N],
                                        [BLOCK_M_SPLIT, SLICE_N],
                                    )
                                    result = tlx.local_load(sub)
                                    tlx.barrier_arrive(tmem_empty_bars[acc_buf], 1)
                                    out_tile = result.to(tl.bfloat16)
                                    offs_am = tile_id * BLOCK_M + g * BLOCK_M_SPLIT
                                    offs_bn = cta_cluster_rank * BLOCK_N + s * SLICE_N
                                    out_desc.store([offs_am, offs_bn], out_tile)
                            else:
                                # ---- Pass 1: accumulate sums ----
                                local_sum = tl.zeros(
                                    (BLOCK_M_SPLIT, 1), dtype=tl.float32
                                )
                                local_sum_sq = tl.zeros(
                                    (BLOCK_M_SPLIT, 1), dtype=tl.float32
                                )
                                for s in tl.static_range(EPILOGUE_SUBTILE):
                                    sub = tlx.local_slice(
                                        acc_tmem,
                                        [0, s * SLICE_N],
                                        [BLOCK_M_SPLIT, SLICE_N],
                                    )
                                    acc_slice = tlx.local_load(sub)
                                    local_sum += tl.sum(
                                        acc_slice, axis=1, keep_dims=True
                                    )
                                    local_sum_sq += tl.sum(
                                        acc_slice * acc_slice, axis=1, keep_dims=True
                                    )

                                # ---- DSMEM exchange ----
                                tlx.barrier_wait(
                                    consumer_barriers[buffer_id], consumer_phase
                                )
                                tlx.barrier_expect_bytes(
                                    reduction_barriers[buffer_id],
                                    size=merged_cross_cta_bytes,
                                )
                                buf_offset = buffer_id * NUM_REDUCTION_CTAS
                                tlx.local_store(
                                    sum_reduction_buf[buf_offset + cta_cluster_rank],
                                    local_sum,
                                )
                                tlx.local_store(
                                    sum_sq_reduction_buf[buf_offset + cta_cluster_rank],
                                    local_sum_sq,
                                )
                                for i in tl.static_range(NUM_REDUCTION_CTAS):
                                    if cta_cluster_rank != i:
                                        tlx.async_remote_shmem_store(
                                            dst=sum_reduction_buf[
                                                buf_offset + cta_cluster_rank
                                            ],
                                            src=local_sum,
                                            remote_cta_rank=i,
                                            barrier=reduction_barriers[buffer_id],
                                        )
                                        tlx.async_remote_shmem_store(
                                            dst=sum_sq_reduction_buf[
                                                buf_offset + cta_cluster_rank
                                            ],
                                            src=local_sum_sq,
                                            remote_cta_rank=i,
                                            barrier=reduction_barriers[buffer_id],
                                        )
                                tlx.barrier_wait(
                                    reduction_barriers[buffer_id],
                                    phase=producer_phase,
                                )

                                global_sum = tl.zeros(
                                    (BLOCK_M_SPLIT, 1), dtype=tl.float32
                                )
                                global_sum_sq = tl.zeros(
                                    (BLOCK_M_SPLIT, 1), dtype=tl.float32
                                )
                                for i in tl.static_range(NUM_REDUCTION_CTAS):
                                    global_sum += tlx.local_load(
                                        tlx.local_view(
                                            sum_reduction_buf, buf_offset + i
                                        )
                                    )
                                    global_sum_sq += tlx.local_load(
                                        tlx.local_view(
                                            sum_sq_reduction_buf, buf_offset + i
                                        )
                                    )

                                mean = global_sum / N
                                var = global_sum_sq / N - mean * mean
                                rstd = libdevice.rsqrt(var + eps)

                                # ---- Pass 2: normalize + store ----
                                for s in tl.static_range(EPILOGUE_SUBTILE):
                                    sub = tlx.local_slice(
                                        acc_tmem,
                                        [0, s * SLICE_N],
                                        [BLOCK_M_SPLIT, SLICE_N],
                                    )
                                    acc_slice = tlx.local_load(sub)
                                    tlx.barrier_arrive(tmem_empty_bars[acc_buf], 1)
                                    ln_w_slice = tl.load(
                                        ln_weight_ptr
                                        + cta_cluster_rank * BLOCK_N
                                        + s * SLICE_N
                                        + tl.arange(0, SLICE_N)
                                    ).to(tl.float32)
                                    ln_b_slice = tl.load(
                                        ln_bias_ptr
                                        + cta_cluster_rank * BLOCK_N
                                        + s * SLICE_N
                                        + tl.arange(0, SLICE_N)
                                    ).to(tl.float32)
                                    normalized = (acc_slice - mean) * rstd
                                    out_slice = (
                                        normalized * ln_w_slice[None, :]
                                        + ln_b_slice[None, :]
                                    )
                                    out_slice = out_slice.to(tl.bfloat16)
                                    offs_am = tile_id * BLOCK_M + g * BLOCK_M_SPLIT
                                    offs_bn = cta_cluster_rank * BLOCK_N + s * SLICE_N
                                    out_desc.store([offs_am, offs_bn], out_slice)

                                if cta_cluster_rank == 0:
                                    row_offsets = (
                                        tile_id * BLOCK_M
                                        + g * BLOCK_M_SPLIT
                                        + tl.arange(0, BLOCK_M_SPLIT)
                                    )
                                    mean_1d = tl.reshape(mean, (BLOCK_M_SPLIT,))
                                    rstd_1d = tl.reshape(rstd, (BLOCK_M_SPLIT,))
                                    tl.store(mean_ptr + row_offsets, mean_1d)
                                    tl.store(rstd_ptr + row_offsets, rstd_1d)

                                for i in tl.static_range(NUM_REDUCTION_CTAS):
                                    if cta_cluster_rank != i:
                                        tlx.barrier_arrive(
                                            consumer_barriers[buffer_id],
                                            arrive_count=1,
                                            remote_cta_rank=i,
                                        )

                                buffer_id = buffer_id ^ 1
                                if buffer_id == 0:
                                    consumer_phase = consumer_phase ^ 1
                                    producer_phase = producer_phase ^ 1

                    tmem_read_phase = tmem_read_phase ^ (
                        cur_tmem_buf == NUM_TMEM_BUFFERS - 1
                    )
                    cur_tmem_buf = (cur_tmem_buf + 1) % NUM_TMEM_BUFFERS

                tile_id = tlx.clc_consumer(
                    clc_context, clc_phase_consumer, multi_ctas=True
                )
                tile_id = tile_id // NUM_REDUCTION_CTAS if tile_id != -1 else -1
                clc_phase_consumer = clc_phase_consumer ^ 1

            tlx.cluster_barrier()

        # ---- Task 3: Dummy (pad warps to multiple of 4) ----
        with tlx.async_task(num_warps=2, registers=24):
            tile_id = tl.program_id(axis=0) // NUM_REDUCTION_CTAS
            clc_phase_consumer = 0
            while tile_id != -1:
                tile_id = tlx.clc_consumer(
                    clc_context, clc_phase_consumer, multi_ctas=True
                )
                tile_id = tile_id // NUM_REDUCTION_CTAS if tile_id != -1 else -1
                clc_phase_consumer = clc_phase_consumer ^ 1
            tlx.cluster_barrier()


@lru_cache
def get_matmul_layernorm_kernel(N: int):
    return triton.autotune(
        configs=_make_configs(N),
        prune_configs_by={"early_config_prune": _prune_configs},
        key=["M", "N", "K"],
    )(fused_matmul_layernorm_kernel)


# ============================================================================
# Host-side launcher
# ============================================================================


def _matmul_layernorm_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused matmul + layernorm: out = layernorm(x @ w, ln_weight, ln_bias)."""
    assert x.is_contiguous() and w.is_contiguous()
    M_val, K_val = x.shape
    K2, N_val = w.shape
    assert K_val == K2
    assert ln_weight.shape[0] == N_val
    assert ln_bias.shape[0] == N_val

    out = torch.empty((M_val, N_val), device=x.device, dtype=x.dtype)
    mean = torch.empty((M_val,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((M_val,), device=x.device, dtype=torch.float32)

    dummy_block = [1, 1]
    a_desc = TensorDescriptor(x, x.shape, x.stride(), dummy_block)
    b_desc = TensorDescriptor(w, w.shape, w.stride(), dummy_block)
    out_desc = TensorDescriptor(out, out.shape, out.stride(), dummy_block)

    def alloc_fn(size: int, alignment: int, _):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    def grid(META):
        num_reduction_ctas = N_val // META["BLOCK_N"]
        num_m_tiles = triton.cdiv(M_val, META["BLOCK_M"])
        return (num_m_tiles * num_reduction_ctas,)

    get_matmul_layernorm_kernel(N_val)[grid](
        a_desc,
        b_desc,
        out_desc,
        ln_weight,
        ln_bias,
        mean,
        rstd,
        eps,
        M_val,
        N_val,
        K_val,
    )

    return out, mean, rstd


def _matmul_only_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    """Matmul-only using the same kernel with SKIP_NORM=True (for benchmarking)."""
    assert x.is_contiguous() and w.is_contiguous()
    M_val, K_val = x.shape
    K2, N_val = w.shape
    assert K_val == K2

    out = torch.empty((M_val, N_val), device=x.device, dtype=x.dtype)
    ln_weight = torch.empty((N_val,), device=x.device, dtype=x.dtype)
    ln_bias = torch.empty((N_val,), device=x.device, dtype=x.dtype)
    mean = torch.empty((M_val,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((M_val,), device=x.device, dtype=torch.float32)

    dummy_block = [1, 1]
    a_desc = TensorDescriptor(x, x.shape, x.stride(), dummy_block)
    b_desc = TensorDescriptor(w, w.shape, w.stride(), dummy_block)
    out_desc = TensorDescriptor(out, out.shape, out.stride(), dummy_block)

    def alloc_fn(size: int, alignment: int, _):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    def grid(META):
        num_reduction_ctas = N_val // META["BLOCK_N"]
        num_m_tiles = triton.cdiv(M_val, META["BLOCK_M"])
        return (num_m_tiles * num_reduction_ctas,)

    get_matmul_layernorm_kernel(N_val)[grid](
        a_desc,
        b_desc,
        out_desc,
        ln_weight,
        ln_bias,
        mean,
        rstd,
        1e-5,
        M_val,
        N_val,
        K_val,
        SKIP_NORM=True,
    )

    return out


# ============================================================================
# Custom Op Registration
# ============================================================================


@torch.library.custom_op("ads_mkl::tlx_matmul_layernorm_fwd", mutates_args=())
def tlx_matmul_layernorm_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused matmul + layernorm forward: out = layernorm(x @ w, ln_weight, ln_bias).

    Args:
        x: Input tensor of shape [M, K], bf16
        w: Weight matrix of shape [K, N], bf16 (N must be divisible by BLOCK_N)
        ln_weight: LayerNorm weight of shape [N]
        ln_bias: LayerNorm bias of shape [N]
        eps: Epsilon for numerical stability

    Returns:
        out: Normalized output of shape [M, N], bf16
        mean: Mean tensor of shape [M], f32
        rstd: Reciprocal standard deviation of shape [M], f32
    """
    return _matmul_layernorm_fwd(x, w, ln_weight, ln_bias, eps)


@tlx_matmul_layernorm_fwd.register_fake
def _(
    x: torch.Tensor,
    w: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, K = x.shape
    K2, N = w.shape
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    mean = torch.empty((M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    return out, mean, rstd


# ============================================================================
# Autograd: forward-only fused kernel, backward uses unfused PyTorch
# ============================================================================


def _setup_context(ctx, inputs, output):
    x, w, ln_weight, ln_bias, eps = inputs
    _, mean, rstd = output
    ctx.save_for_backward(x, w, ln_weight, ln_bias, mean, rstd)


def _backward(ctx, grad_out, grad_mean, grad_rstd):
    x, w, ln_weight, ln_bias, mean, rstd = ctx.saved_tensors

    z = x @ w

    normalized_shape = [z.shape[-1]]
    grad_z, grad_ln_weight, grad_ln_bias = torch.ops.aten.native_layer_norm_backward(
        grad_out,
        z,
        normalized_shape,
        mean,
        rstd,
        ln_weight,
        ln_bias,
        [True, True, True],
    )

    grad_x = grad_z @ w.t()
    grad_w = x.t() @ grad_z

    return grad_x, grad_w, grad_ln_weight, grad_ln_bias, None


tlx_matmul_layernorm_fwd.register_autograd(_backward, setup_context=_setup_context)


@register_flop_formula(torch.ops.ads_mkl.tlx_matmul_layernorm_fwd)
def _matmul_layernorm_flop(
    x_shape,
    w_shape,
    *args,
    **kwargs,
) -> int:
    m, k = x_shape
    k2, n = w_shape
    return m * k * n * 2
