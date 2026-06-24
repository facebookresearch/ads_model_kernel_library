# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors
import math
import os

import torch
import torch.nn

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl
import triton.language.extra.tlx as tlx  # @manual=//triton:triton
from triton.tools.tensor_descriptor import TensorDescriptor


DISABLE_AUTOTUNE = os.environ.get("ADS_MKL_DISABLE_AUTOTUNE") == "1"


def selected_autotune_configs(configs):
    config_index = os.environ.get("TLX_CONFIG_INDEX")
    if config_index is None:
        return configs

    index = int(config_index)
    if index < 0 or index >= len(configs):
        raise ValueError(
            f"TLX_CONFIG_INDEX={index} out of range for {len(configs)} configs"
        )
    print(f"Selecting autotune config index {index} of {len(configs)}")
    return [configs[index]]


def get_reduced_autotune_config(N, fused):
    if fused:
        BN = 128
        num_ctas = N // BN
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": BN,
                    "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": 2,
                    "NUM_SMEM_BUFFERS": 3,
                    "NUM_TMEM_BUFFERS": 2,
                    "NUM_MMA_GROUPS": 1,
                    "EPILOGUE_SUBTILE": 4,
                    "PAIR_CTA": False,
                    "NUM_REDUCTION_CTAS": num_ctas,
                    "CLUSTER_LAUNCH_CONTROL": True,
                    "DIRECT_HN_LOAD": False,
                    "NUM_H_BUFFERS": 2,
                    "STAGE_DH": False,
                    "NUM_C_BUFFERS": 2,
                    "REDUCTION_SUBTILE": 1,
                },
                num_warps=4,
                num_stages=1,
                pre_hook=matmul_tma_set_block_size_hook,
                ctas_per_cga=(num_ctas, 1, 1),
            )
        ]

    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "NUM_SMEM_BUFFERS": 4,
                "NUM_TMEM_BUFFERS": 2,
                "NUM_MMA_GROUPS": 1,
                "EPILOGUE_SUBTILE": 1,
                "PAIR_CTA": False,
                "NUM_REDUCTION_CTAS": 1,
                "CLUSTER_LAUNCH_CONTROL": True,
                "DIRECT_HN_LOAD": False,
                "NUM_H_BUFFERS": 2,
                "STAGE_DH": False,
                "NUM_C_BUFFERS": 2,
                "REDUCTION_SUBTILE": 4,
            },
            num_warps=4,
            num_stages=1,
            pre_hook=matmul_tma_set_block_size_hook,
            ctas_per_cga=(1, 1, 1),
        )
    ]


def get_cuda_autotune_config(N, fused):
    if fused:
        return selected_autotune_configs(
            [
                triton.Config(
                    {
                        "BLOCK_SIZE_M": BM,
                        "BLOCK_SIZE_N": BN,
                        "BLOCK_SIZE_K": BK,
                        "GROUP_SIZE_M": 2,
                        "NUM_SMEM_BUFFERS": s,
                        "NUM_TMEM_BUFFERS": t,
                        "NUM_MMA_GROUPS": m,
                        "EPILOGUE_SUBTILE": subtile,
                        "PAIR_CTA": False,
                        "NUM_REDUCTION_CTAS": N // BN,
                        "CLUSTER_LAUNCH_CONTROL": clc,
                        "DIRECT_HN_LOAD": False,
                        "NUM_H_BUFFERS": hb,
                        "STAGE_DH": False,
                        "NUM_C_BUFFERS": 2,
                        "REDUCTION_SUBTILE": reduction_subtile,
                    },
                    num_warps=4,
                    num_stages=1,
                    pre_hook=matmul_tma_set_block_size_hook,
                    ctas_per_cga=(N // BN, 1, 1),
                )
                for BM in [64, 128]
                for BN in [128, 256]
                if N % BN == 0
                for BK in [64, 128]
                for s in [2, 3, 4]
                for t in [1, 2]
                for subtile in [4, 8]
                for m in [1]
                for clc in [True]
                for hb in [1, 2, 3]
                for reduction_subtile in [1, 2, 4]
                if N // BN <= 8 and 64 <= BM // m <= 128
            ]
        )
    else:
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": BM,
                    "BLOCK_SIZE_N": BN,
                    "BLOCK_SIZE_K": BK,
                    "GROUP_SIZE_M": 8,
                    "NUM_SMEM_BUFFERS": s,
                    "NUM_TMEM_BUFFERS": t,
                    "NUM_MMA_GROUPS": m,
                    "EPILOGUE_SUBTILE": subtile,
                    "PAIR_CTA": False,
                    "NUM_REDUCTION_CTAS": 1,
                    "CLUSTER_LAUNCH_CONTROL": True,
                    "DIRECT_HN_LOAD": False,
                    "NUM_H_BUFFERS": 2,
                    "STAGE_DH": False,
                    "NUM_C_BUFFERS": 2,
                    "REDUCTION_SUBTILE": 4,
                },
                num_warps=4,
                num_stages=1,
                pre_hook=matmul_tma_set_block_size_hook,
                ctas_per_cga=(1, 1, 1),
            )
            for BM in [128]
            for BN in [128]
            for BK in [64, 128]
            for s in [4, 5, 6]
            for t in [2, 3]
            for m in [1]
            for subtile in [1, 2, 4]
        ]


def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    NUM_MMA_GROUPS = nargs.get("NUM_MMA_GROUPS", 1)
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    nargs["a_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_K]
    if nargs.get("PAIR_CTA", False):
        nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N // 2]
    else:
        nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N]
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", 1)
    nargs["c_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_N // EPILOGUE_SUBTILE]
    nargs["hn_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_N]


@triton.jit
def _compute_pid(tile_id, num_pid_m, num_pid_n):
    pid_n = tile_id % num_pid_n
    pid_m = tile_id // num_pid_n
    return pid_m, pid_n


def preprocess_configs(configs, named_args, **kwargs):
    MAX_SHARED_MEMORY = 232448
    MAX_TENSOR_MEMORY = 256 * 1024
    MBARRIER_SIZE = 8
    CLC_RESPONSE_SIZE = 16

    pruned_configs = []
    for conf in configs:
        M = named_args["M"]
        N = named_args["N"]
        BLOCK_M = conf.kwargs["BLOCK_SIZE_M"]
        BLOCK_N = conf.kwargs["BLOCK_SIZE_N"]
        BLOCK_K = conf.kwargs["BLOCK_SIZE_K"]
        NUM_SMEM_BUFFERS = conf.kwargs["NUM_SMEM_BUFFERS"]
        NUM_TMEM_BUFFERS = conf.kwargs["NUM_TMEM_BUFFERS"]
        PAIR_CTA = conf.kwargs["PAIR_CTA"]
        CLUSTER_LAUNCH_CONTROL = conf.kwargs["CLUSTER_LAUNCH_CONTROL"]
        NUM_MMA_GROUPS = conf.kwargs["NUM_MMA_GROUPS"]
        NUM_REDUCTION_CTAS = conf.kwargs["NUM_REDUCTION_CTAS"]
        DIRECT_HN_LOAD = conf.kwargs["DIRECT_HN_LOAD"]
        STAGE_DH = conf.kwargs["STAGE_DH"]
        NUM_C_BUFFERS = conf.kwargs["NUM_C_BUFFERS"]

        if BLOCK_M // NUM_MMA_GROUPS > 128:
            continue

        num_tiles_m = math.ceil(M / BLOCK_M)
        num_tiles_n = math.ceil(N / BLOCK_N)
        pair_cta_compatible = (
            num_tiles_m % 2 == 0 and (num_tiles_m * num_tiles_n) % 2 == 0
        )
        if not pair_cta_compatible:
            conf.kwargs["PAIR_CTA"] = False
            PAIR_CTA = False

        BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
        NUM_H_BUFFERS = conf.kwargs["NUM_H_BUFFERS"]
        smem_a = BLOCK_M * BLOCK_K * 2 * NUM_SMEM_BUFFERS
        smem_b_size = (BLOCK_N // 2) if PAIR_CTA else BLOCK_N
        smem_b = BLOCK_K * smem_b_size * 2 * NUM_SMEM_BUFFERS
        EPILOGUE_SUBTILE = conf.kwargs["EPILOGUE_SUBTILE"]
        smem_epilog = (
            (BLOCK_M // NUM_MMA_GROUPS)
            * (BLOCK_N // EPILOGUE_SUBTILE)
            * 2
            * NUM_C_BUFFERS
        )
        smem_barriers = NUM_SMEM_BUFFERS * MBARRIER_SIZE
        if PAIR_CTA:
            smem_barriers += NUM_SMEM_BUFFERS * NUM_MMA_GROUPS * MBARRIER_SIZE
        smem_barriers += NUM_TMEM_BUFFERS
        smem_clc = (
            (CLC_RESPONSE_SIZE + MBARRIER_SIZE * 2) if CLUSTER_LAUNCH_CONTROL else 0
        )
        uses_hn_smem = not DIRECT_HN_LOAD
        smem_hn = BLOCK_M_SPLIT * BLOCK_N * 2 * NUM_H_BUFFERS if uses_hn_smem else 0
        reduction_buf_ctas = 1 if NUM_REDUCTION_CTAS == 2 else NUM_REDUCTION_CTAS
        smem_reduction = BLOCK_M_SPLIT * 4 * reduction_buf_ctas
        total_smem = (
            smem_a
            + smem_b
            + smem_epilog
            + smem_barriers
            + smem_clc
            + smem_hn
            + smem_reduction
            + (BLOCK_M_SPLIT * BLOCK_N * 2 if STAGE_DH else 0)
        )
        if total_smem > MAX_SHARED_MEMORY:
            continue

        # h_norm TMEM staging buffer
        total_tmem = BLOCK_M * BLOCK_N * 4 * NUM_TMEM_BUFFERS
        if uses_hn_smem:
            total_tmem += BLOCK_M_SPLIT * BLOCK_N * 2
        if total_tmem > MAX_TENSOR_MEMORY:
            continue

        pruned_configs.append(conf)

    return pruned_configs


@triton.jit
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS):
    bufIdx = accum_cnt % NUM_BUFFERS
    phase = (accum_cnt // NUM_BUFFERS) & 1
    return bufIdx, phase


@triton.jit
def _compute_grid_info(
    M,
    N,
    K,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    return start_pid, num_pid_m, num_pid_n, num_tiles, k_tiles


# ── MMA consumer ────────────────────────────────────────────────────────────


@triton.jit
def _process_tile_mma_inner(
    k_tiles,
    NUM_SMEM_BUFFERS,
    NUM_MMA_GROUPS,
    NUM_TMEM_BUFFERS,
    buffers_A,
    buffers_B,
    tmem_buffers,
    A_smem_full_bars,
    B_smem_full_bars,
    A_smem_empty_bars,
    tmem_full_bars,
    cur_tmem_buf,
    tmem_empty_bars,
    tmem_write_phase,
    smem_accum_cnt,
    PAIR_CTA,
    cta_bars,
    pred_leader_cta,
    cluster_cta_rank,
):
    buf, phase = _get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)
    tlx.barrier_wait(B_smem_full_bars[buf], phase)

    for group_id in tl.static_range(NUM_MMA_GROUPS):
        a_buf = group_id * NUM_SMEM_BUFFERS + buf
        acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
        tlx.barrier_wait(A_smem_full_bars[a_buf], phase)
        tlx.barrier_wait(
            tmem_empty_bars[group_id * NUM_TMEM_BUFFERS + cur_tmem_buf],
            tmem_write_phase ^ 1,
        )
        if PAIR_CTA:
            tlx.barrier_arrive(
                cta_bars[a_buf],
                arrive_count=1,
                remote_cta_rank=cluster_cta_rank & ~1,
            )
            tlx.barrier_wait(cta_bars[a_buf], phase=phase, pred=pred_leader_cta)
        tlx.async_dot(
            buffers_A[a_buf],
            buffers_B[buf],
            tmem_buffers[acc_buf],
            use_acc=False,
            mBarriers=[A_smem_empty_bars[a_buf]],
            two_ctas=PAIR_CTA,
            out_dtype=tl.float32,
        )

    smem_accum_cnt += 1

    for _ in range(1, k_tiles):
        buf, phase = _get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)
        tlx.barrier_wait(B_smem_full_bars[buf], phase)
        for group_id in tl.static_range(NUM_MMA_GROUPS):
            a_buf = group_id * NUM_SMEM_BUFFERS + buf
            acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
            tlx.barrier_wait(A_smem_full_bars[a_buf], phase)
            if PAIR_CTA:
                tlx.barrier_arrive(
                    cta_bars[a_buf],
                    arrive_count=1,
                    remote_cta_rank=cluster_cta_rank & ~1,
                )
                tlx.barrier_wait(cta_bars[a_buf], phase=phase, pred=pred_leader_cta)
            tlx.async_dot(
                buffers_A[a_buf],
                buffers_B[buf],
                tmem_buffers[acc_buf],
                use_acc=True,
                mBarriers=[A_smem_empty_bars[a_buf]],
                two_ctas=PAIR_CTA,
                out_dtype=tl.float32,
            )
        smem_accum_cnt += 1

    last_buf, last_phase = _get_bufidx_phase(smem_accum_cnt - 1, NUM_SMEM_BUFFERS)
    for group_id in tl.static_range(NUM_MMA_GROUPS):
        a_buf = group_id * NUM_SMEM_BUFFERS + last_buf
        tlx.barrier_wait(A_smem_empty_bars[a_buf], last_phase)
        acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
        tlx.barrier_arrive(tmem_full_bars[acc_buf], 1)

    return smem_accum_cnt


# ── TMA producer ────────────────────────────────────────────────────────────


@triton.jit
def _process_tile_producer_inner(
    tile_id,
    num_pid_m,
    num_pid_n,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    NUM_MMA_GROUPS,
    k_tiles,
    NUM_SMEM_BUFFERS,
    a_desc,
    b_desc,
    buffers_A,
    buffers_B,
    A_smem_full_bars,
    B_smem_full_bars,
    A_smem_empty_bars,
    smem_accum_cnt,
    PAIR_CTA,
    cluster_cta_rank,
):
    pid_m, pid_n = _compute_pid(tile_id, num_pid_m, num_pid_n)
    dsize: tl.constexpr = tlx.size_of(tlx.dtype_of(b_desc))
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_SIZE_M // NUM_MMA_GROUPS
    if PAIR_CTA:
        offs_bn = pid_n * BLOCK_SIZE_N + cluster_cta_rank * (BLOCK_SIZE_N // 2)
        expected_bytes: tl.constexpr = dsize * BLOCK_SIZE_N * BLOCK_SIZE_K // 2
    else:
        offs_bn = pid_n * BLOCK_SIZE_N
        expected_bytes: tl.constexpr = dsize * BLOCK_SIZE_N * BLOCK_SIZE_K

    for k in range(0, k_tiles):
        buf, phase = _get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)
        offs_k = k * BLOCK_SIZE_K

        a_buf = buf
        tlx.barrier_wait(A_smem_empty_bars[a_buf], phase ^ 1)
        offs_am = pid_m * BLOCK_SIZE_M
        tlx.barrier_expect_bytes(
            A_smem_full_bars[a_buf], dsize * BLOCK_M_SPLIT * BLOCK_SIZE_K
        )
        tlx.async_descriptor_load(
            a_desc,
            buffers_A[a_buf],
            [offs_am, offs_k],
            A_smem_full_bars[a_buf],
            eviction_policy="evict_first",
        )

        last_a_buf = (NUM_MMA_GROUPS - 1) * NUM_SMEM_BUFFERS + buf
        tlx.barrier_wait(A_smem_empty_bars[last_a_buf], phase ^ 1)
        tlx.barrier_expect_bytes(B_smem_full_bars[buf], expected_bytes)
        tlx.async_descriptor_load(
            b_desc,
            buffers_B[buf],
            [offs_k, offs_bn],
            B_smem_full_bars[buf],
            eviction_policy="evict_last",
        )

        for group_id in tl.static_range(1, NUM_MMA_GROUPS):
            a_buf = group_id * NUM_SMEM_BUFFERS + buf
            tlx.barrier_wait(A_smem_empty_bars[a_buf], phase ^ 1)
            offs_am2 = offs_am + group_id * BLOCK_M_SPLIT
            tlx.barrier_expect_bytes(
                A_smem_full_bars[a_buf], dsize * BLOCK_M_SPLIT * BLOCK_SIZE_K
            )
            tlx.async_descriptor_load(
                a_desc,
                buffers_A[a_buf],
                [offs_am2, offs_k],
                A_smem_full_bars[a_buf],
                eviction_policy="evict_first",
            )

        smem_accum_cnt += 1

    return smem_accum_cnt


# ── main kernel ─────────────────────────────────────────────────────────────


@triton.jit
def matmul_rmsnorm_bwd_kernel(  # noqa: C901
    a_desc,
    b_desc,
    c_desc,
    hn_desc,
    h_norm_ptr,
    rrms_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    PAIR_CTA: tl.constexpr,
    CLUSTER_LAUNCH_CONTROL: tl.constexpr,
    DIRECT_HN_LOAD: tl.constexpr,
    NUM_H_BUFFERS: tl.constexpr,
    STAGE_DH: tl.constexpr,
    NUM_C_BUFFERS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    FUSED_RMS_NORM_BWD: tl.constexpr,
    NUM_REDUCTION_CTAS: tl.constexpr,
    REDUCTION_SUBTILE: tl.constexpr = 4,
):
    NUM_CTAS_IN_CLUSTER: tl.constexpr = NUM_REDUCTION_CTAS
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_SIZE_M // NUM_MMA_GROUPS

    if FUSED_RMS_NORM_BWD and not DIRECT_HN_LOAD:
        # h_norm SMEM: loaded by producer via TMA, consumed by epilogue
        hn_smem = tlx.local_alloc(
            (BLOCK_M_SPLIT, BLOCK_SIZE_N),
            tl.bfloat16,
            NUM_H_BUFFERS,
        )
        hn_full_bars = tlx.alloc_barriers(num_barriers=NUM_H_BUFFERS, arrive_count=1)
        hn_empty_bars = tlx.alloc_barriers(num_barriers=NUM_H_BUFFERS, arrive_count=1)

        # Triton compiler limitation: tl.sum(axis=1) on data loaded from SMEM via
        # local_load causes TritonGPURemoveLayoutConversions to fail because SMEM
        # tensors have #linear layout incompatible with the reduction pass.
        # Workaround: route h_norm through a TMEM staging buffer. The TMEM
        # local_load produces a layout that tl.sum accepts.
        hn_tmem = tlx.local_alloc(
            (BLOCK_M_SPLIT, BLOCK_SIZE_N),
            tl.bfloat16,
            1,
            tlx.storage_kind.tmem,
        )

    if FUSED_RMS_NORM_BWD:
        reduction_barriers = tlx.alloc_barriers(num_barriers=1)
        REDUCTION_BUF_CTAS: tl.constexpr = (
            1 if NUM_REDUCTION_CTAS == 2 else NUM_REDUCTION_CTAS
        )
        reduction_buf = tlx.local_alloc(
            (BLOCK_M_SPLIT, 1),
            tl.float32,
            REDUCTION_BUF_CTAS,
        )
        cross_cta_expected_bytes: tl.constexpr = (
            BLOCK_M_SPLIT * tlx.size_of(tl.float32) * (NUM_REDUCTION_CTAS - 1)
        )
        reduction_phase = 0

    # Matmul SMEM/TMEM buffers
    buffers_A = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_SIZE_K),
        tlx.dtype_of(a_desc),
        NUM_SMEM_BUFFERS * NUM_MMA_GROUPS,
    )
    if PAIR_CTA:
        buffers_B = tlx.local_alloc(
            (BLOCK_SIZE_K, BLOCK_SIZE_N // 2),
            tl.bfloat16,
            NUM_SMEM_BUFFERS,
        )
    else:
        buffers_B = tlx.local_alloc(
            (BLOCK_SIZE_K, BLOCK_SIZE_N),
            tl.bfloat16,
            NUM_SMEM_BUFFERS,
        )
    tmem_buffers = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_SIZE_N),
        tl.float32,
        NUM_TMEM_BUFFERS * NUM_MMA_GROUPS,
        tlx.storage_kind.tmem,
    )

    epilog_slice: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
    c_smem_buffers = tlx.local_alloc(
        (BLOCK_M_SPLIT, epilog_slice),
        tlx.dtype_of(c_desc),
        NUM_C_BUFFERS,
    )
    if FUSED_RMS_NORM_BWD and STAGE_DH:
        dh_smem_buffers = tlx.local_alloc(
            (BLOCK_M_SPLIT, epilog_slice),
            tl.bfloat16,
            EPILOGUE_SUBTILE,
        )

    if PAIR_CTA:
        cluster_cta_rank = tlx.cluster_cta_rank()
        pred_leader_cta = cluster_cta_rank % 2 == 0
        cta_bars = tlx.alloc_barriers(
            num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS,
            arrive_count=2,
        )
    else:
        cluster_cta_rank = 0
        pred_leader_cta = False
        cta_bars = None

    A_smem_full_bars = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS,
        arrive_count=1,
    )
    A_smem_empty_bars = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS,
        arrive_count=1,
    )
    B_smem_full_bars = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS,
        arrive_count=1,
    )
    tmem_full_bars = tlx.alloc_barriers(
        num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS,
        arrive_count=1,
    )
    tmem_empty_bars = tlx.alloc_barriers(
        num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS,
        arrive_count=1,
    )

    if CLUSTER_LAUNCH_CONTROL:
        clc_context = tlx.clc_create_context(3 * NUM_CTAS_IN_CLUSTER)

    hn_expected_bytes: tl.constexpr = (
        BLOCK_M_SPLIT * BLOCK_SIZE_N * tlx.size_of(tl.bfloat16)
    )

    with tlx.async_tasks():
        # ── epilogue ────────────────────────────────────────────────────
        with tlx.async_task("default"):
            start_pid, num_pid_m, num_pid_n, num_tiles, k_tiles = _compute_grid_info(
                M,
                N,
                K,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
            )

            tmem_accum_cnt = 0
            tile_id = start_pid
            clc_phase_producer = 1
            clc_phase_consumer = 0
            hn_consume_cnt = 0

            while tile_id != -1:
                if CLUSTER_LAUNCH_CONTROL:
                    tlx.clc_producer(
                        clc_context,
                        clc_phase_producer,
                        multi_ctas=NUM_CTAS_IN_CLUSTER != 1,
                    )
                    clc_phase_producer ^= 1

                pid_m, pid_n = _compute_pid(tile_id, num_pid_m, num_pid_n)
                offs_bn = pid_n * BLOCK_SIZE_N
                slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
                reduction_subtile: tl.constexpr = REDUCTION_SUBTILE
                reduction_slice_size: tl.constexpr = BLOCK_SIZE_N // reduction_subtile
                inv_n: tl.constexpr = 1.0 / (BLOCK_SIZE_N * NUM_REDUCTION_CTAS)
                cur_tmem_buf, tmem_read_phase = _get_bufidx_phase(
                    tmem_accum_cnt,
                    NUM_TMEM_BUFFERS,
                )

                for group_id in tl.static_range(NUM_MMA_GROUPS):
                    buf_idx = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
                    offs_am = pid_m * BLOCK_SIZE_M + group_id * BLOCK_M_SPLIT

                    if FUSED_RMS_NORM_BWD and not DIRECT_HN_LOAD:
                        offs_m = offs_am + tl.arange(0, BLOCK_M_SPLIT)
                        rrms_vals = tl.load(
                            rrms_ptr + offs_m,
                            eviction_policy="evict_last",
                        )[:, None]

                        # Load h_norm ONCE per group (full BLOCK_N). Producer
                        # starts this TMA before the K-loop, so stage it while
                        # the matmul result may still be finishing.
                        hn_buf, hn_phase = _get_bufidx_phase(
                            hn_consume_cnt,
                            NUM_H_BUFFERS,
                        )
                        tlx.barrier_wait(hn_full_bars[hn_buf], hn_phase)
                        hn_raw = tlx.local_load(hn_smem[hn_buf])
                        tlx.fence_async_shared()
                        tlx.barrier_arrive(hn_empty_bars[hn_buf], 1)
                        hn_consume_cnt += 1

                        # TMEM staging for tl.sum layout compatibility
                        tlx.local_store(hn_tmem[0], hn_raw)
                    # Wait for matmul result
                    tlx.barrier_wait(tmem_full_bars[buf_idx], tmem_read_phase)
                    acc_tmem = tmem_buffers[buf_idx]

                    if FUSED_RMS_NORM_BWD:
                        cta_rank = tlx.cluster_cta_rank()

                        if DIRECT_HN_LOAD:
                            offs_m = offs_am + tl.arange(0, BLOCK_M_SPLIT)
                            rrms_vals = tl.load(
                                rrms_ptr + offs_m,
                                eviction_policy="evict_last",
                            )[:, None]

                        # Pass 1: accumulate local_dot across subtiles
                        # (don't signal tmem_empty yet, we re-read in pass 2)
                        local_dot = tl.zeros(
                            (BLOCK_M_SPLIT, 1),
                            dtype=tl.float32,
                        )
                        for slice_id in tl.static_range(reduction_subtile):
                            if DIRECT_HN_LOAD:
                                offs_n = (
                                    offs_bn
                                    + slice_id * reduction_slice_size
                                    + tl.arange(0, reduction_slice_size)
                                )
                                hn_slice = tl.load(
                                    h_norm_ptr + offs_m[:, None] * N + offs_n[None, :],
                                    mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
                                    other=0.0,
                                ).to(tl.float32)
                            else:
                                hn_sub = tlx.local_slice(
                                    hn_tmem[0],
                                    [0, slice_id * reduction_slice_size],
                                    [BLOCK_M_SPLIT, reduction_slice_size],
                                )
                                hn_slice = tlx.local_load(hn_sub)
                            dh_sub = tlx.local_slice(
                                acc_tmem,
                                [0, slice_id * reduction_slice_size],
                                [BLOCK_M_SPLIT, reduction_slice_size],
                            )
                            dh_slice = tlx.local_load(dh_sub)
                            if STAGE_DH:
                                tlx.local_store(
                                    dh_smem_buffers[slice_id],
                                    dh_slice.to(tl.bfloat16),
                                )
                            local_dot += tl.sum(
                                dh_slice.to(tl.bfloat16) * hn_slice,
                                axis=1,
                                keep_dims=True,
                            )

                        if STAGE_DH:
                            tlx.fence_async_shared()
                            tlx.barrier_arrive(tmem_empty_bars[buf_idx], 1)

                        if NUM_REDUCTION_CTAS == 1:
                            c_val = local_dot * inv_n
                        else:
                            # DSMEM cross-CTA reduction. The barrier must know
                            # expected remote bytes before async stores arrive.
                            tlx.barrier_expect_bytes(
                                reduction_barriers[0],
                                cross_cta_expected_bytes,
                            )
                            if NUM_REDUCTION_CTAS == 2:
                                for i in tl.static_range(NUM_REDUCTION_CTAS):
                                    if cta_rank != i:
                                        tlx.async_remote_shmem_store(
                                            dst=reduction_buf[0],
                                            src=local_dot,
                                            remote_cta_rank=i,
                                            barrier=reduction_barriers[0],
                                        )
                            else:
                                for i in tl.static_range(NUM_REDUCTION_CTAS):
                                    if cta_rank != i:
                                        tlx.async_remote_shmem_store(
                                            dst=reduction_buf[cta_rank],
                                            src=local_dot,
                                            remote_cta_rank=i,
                                            barrier=reduction_barriers[0],
                                        )
                                    else:
                                        tlx.local_store(
                                            reduction_buf[cta_rank],
                                            local_dot,
                                        )

                            tlx.barrier_wait(
                                reduction_barriers[0],
                                phase=reduction_phase,
                            )
                            if NUM_REDUCTION_CTAS == 2:
                                global_dot = local_dot + tlx.local_load(
                                    reduction_buf[0]
                                ).to(tl.float32)
                            else:
                                global_dot = tl.zeros(
                                    (BLOCK_M_SPLIT, 1),
                                    dtype=tl.float32,
                                )
                                for i in tl.static_range(NUM_REDUCTION_CTAS):
                                    global_dot += tlx.local_load(reduction_buf[i]).to(
                                        tl.float32
                                    )
                            c_val = global_dot * inv_n
                            reduction_phase ^= 1
                        # Pass 2: re-read TMEM, apply correction, store,
                        # signal tmem_empty
                        for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                            if DIRECT_HN_LOAD:
                                offs_n = (
                                    offs_bn
                                    + slice_id * slice_size
                                    + tl.arange(0, slice_size)
                                )
                                hn_slice = tl.load(
                                    h_norm_ptr + offs_m[:, None] * N + offs_n[None, :],
                                    mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
                                    other=0.0,
                                )
                            else:
                                hn_sub = tlx.local_slice(
                                    hn_tmem[0],
                                    [0, slice_id * slice_size],
                                    [BLOCK_M_SPLIT, slice_size],
                                )
                                hn_slice = tlx.local_load(hn_sub)
                            if STAGE_DH:
                                dh_slice = tlx.local_load(
                                    dh_smem_buffers[slice_id],
                                ).to(tl.float32)
                            else:
                                dh_sub = tlx.local_slice(
                                    acc_tmem,
                                    [0, slice_id * slice_size],
                                    [BLOCK_M_SPLIT, slice_size],
                                )
                                dh_slice = tlx.local_load(dh_sub)
                            if not STAGE_DH and slice_id == EPILOGUE_SUBTILE - 1:
                                tlx.barrier_arrive(tmem_empty_bars[buf_idx], 1)
                            result_out = (dh_slice - hn_slice * c_val) * rrms_vals
                            c_smem = c_smem_buffers[slice_id % NUM_C_BUFFERS]
                            tlx.async_descriptor_store_wait(NUM_C_BUFFERS - 1)
                            tlx.local_store(
                                c_smem,
                                result_out.to(tlx.dtype_of(c_desc)),
                            )
                            tlx.async_descriptor_store(
                                c_desc,
                                c_smem,
                                [offs_am, offs_bn + slice_id * slice_size],
                            )
                    else:
                        # Unfused: standard subtile loop
                        for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                            acc_subslice = tlx.local_slice(
                                acc_tmem,
                                [0, slice_id * slice_size],
                                [BLOCK_M_SPLIT, slice_size],
                            )
                            result = tlx.local_load(acc_subslice)
                            if slice_id == EPILOGUE_SUBTILE - 1:
                                tlx.barrier_arrive(tmem_empty_bars[buf_idx], 1)
                            c_smem = c_smem_buffers[slice_id % NUM_C_BUFFERS]
                            tlx.async_descriptor_store_wait(NUM_C_BUFFERS - 1)
                            tlx.local_store(
                                c_smem,
                                result.to(tlx.dtype_of(c_desc)),
                            )
                            tlx.fence_async_shared()
                            tlx.async_descriptor_store(
                                c_desc,
                                c_smem,
                                [offs_am, offs_bn + slice_id * slice_size],
                            )

                tmem_accum_cnt += 1
                if CLUSTER_LAUNCH_CONTROL:
                    tile_id = tlx.clc_consumer(
                        clc_context,
                        clc_phase_consumer,
                        multi_ctas=NUM_CTAS_IN_CLUSTER != 1,
                    )
                    clc_phase_consumer ^= 1
                else:
                    tile_id += NUM_SMS
                tlx.async_descriptor_store_wait(0)

        # ── MMA consumer ────────────────────────────────────────────────
        with tlx.async_task(num_warps=1, num_regs=24):
            start_pid, num_pid_m, num_pid_n, num_tiles, k_tiles = _compute_grid_info(
                M,
                N,
                K,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
            )

            tmem_accum_cnt = 0
            smem_accum_cnt = 0
            tile_id = start_pid
            clc_phase_consumer = 0

            while tile_id != -1:
                cur_tmem_buf, tmem_write_phase = _get_bufidx_phase(
                    tmem_accum_cnt,
                    NUM_TMEM_BUFFERS,
                )
                smem_accum_cnt = _process_tile_mma_inner(
                    k_tiles=k_tiles,
                    NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                    NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                    NUM_TMEM_BUFFERS=NUM_TMEM_BUFFERS,
                    buffers_A=buffers_A,
                    buffers_B=buffers_B,
                    tmem_buffers=tmem_buffers,
                    A_smem_full_bars=A_smem_full_bars,
                    B_smem_full_bars=B_smem_full_bars,
                    A_smem_empty_bars=A_smem_empty_bars,
                    tmem_full_bars=tmem_full_bars,
                    cur_tmem_buf=cur_tmem_buf,
                    tmem_empty_bars=tmem_empty_bars,
                    tmem_write_phase=tmem_write_phase,
                    smem_accum_cnt=smem_accum_cnt,
                    PAIR_CTA=PAIR_CTA,
                    cta_bars=cta_bars,
                    pred_leader_cta=pred_leader_cta,
                    cluster_cta_rank=cluster_cta_rank,
                )
                tmem_accum_cnt += 1
                if CLUSTER_LAUNCH_CONTROL:
                    tile_id = tlx.clc_consumer(
                        clc_context,
                        clc_phase_consumer,
                        multi_ctas=NUM_CTAS_IN_CLUSTER != 1,
                    )
                    clc_phase_consumer ^= 1
                else:
                    tile_id += NUM_SMS

        # ── producer (TMA load) ─────────────────────────────────────────
        with tlx.async_task(num_warps=1, num_regs=32):
            start_pid, num_pid_m, num_pid_n, num_tiles, k_tiles = _compute_grid_info(
                M,
                N,
                K,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
            )

            smem_accum_cnt = 0
            tile_id = start_pid
            clc_phase_consumer = 0
            hn_prod_cnt = 0

            while tile_id != -1:
                if FUSED_RMS_NORM_BWD and not DIRECT_HN_LOAD:
                    pid_m_p, pid_n_p = _compute_pid(
                        tile_id,
                        num_pid_m,
                        num_pid_n,
                    )
                    offs_bn_hn = pid_n_p * BLOCK_SIZE_N
                    for group_id in tl.static_range(NUM_MMA_GROUPS):
                        offs_am_hn = pid_m_p * BLOCK_SIZE_M + group_id * BLOCK_M_SPLIT
                        hn_buf, hn_phase = _get_bufidx_phase(
                            hn_prod_cnt,
                            NUM_H_BUFFERS,
                        )
                        tlx.barrier_wait(
                            hn_empty_bars[hn_buf],
                            hn_phase ^ 1,
                        )
                        tlx.barrier_expect_bytes(
                            hn_full_bars[hn_buf],
                            hn_expected_bytes,
                        )
                        tlx.async_descriptor_load(
                            hn_desc,
                            hn_smem[hn_buf],
                            [offs_am_hn, offs_bn_hn],
                            hn_full_bars[hn_buf],
                            eviction_policy="evict_first",
                        )
                        hn_prod_cnt += 1

                # K-loop: load A and B for the matmul
                smem_accum_cnt = _process_tile_producer_inner(
                    tile_id=tile_id,
                    num_pid_m=num_pid_m,
                    num_pid_n=num_pid_n,
                    BLOCK_SIZE_M=BLOCK_SIZE_M,
                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                    BLOCK_SIZE_K=BLOCK_SIZE_K,
                    NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                    k_tiles=k_tiles,
                    NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                    a_desc=a_desc,
                    b_desc=b_desc,
                    buffers_A=buffers_A,
                    buffers_B=buffers_B,
                    A_smem_full_bars=A_smem_full_bars,
                    B_smem_full_bars=B_smem_full_bars,
                    A_smem_empty_bars=A_smem_empty_bars,
                    smem_accum_cnt=smem_accum_cnt,
                    PAIR_CTA=PAIR_CTA,
                    cluster_cta_rank=cluster_cta_rank,
                )

                if CLUSTER_LAUNCH_CONTROL:
                    tile_id = tlx.clc_consumer(
                        clc_context,
                        clc_phase_consumer,
                        multi_ctas=NUM_CTAS_IN_CLUSTER != 1,
                    )
                    clc_phase_consumer ^= 1
                else:
                    tile_id += NUM_SMS

        # ── dummy pad (warp count must be multiple of 4) ────────────────
        with tlx.async_task(num_warps=1, num_regs=24):
            pass


# ── host-side launch ────────────────────────────────────────────────────────

_kernel_cache = {}
_num_sms_cache = {}


def _num_sms(device):
    if device not in _num_sms_cache:
        _num_sms_cache[device] = torch.cuda.get_device_properties(
            device,
        ).multi_processor_count
    return _num_sms_cache[device]


def tlx_matmul_rmsnorm_bwd(
    dy,
    w2_t,
    h_norm=None,
    rrms=None,
    fused_rms_norm_bwd=False,
    out=None,
):
    """
    Fused: dh = (dh_norm - h_norm * mean(dh_norm * h_norm)) * rrms
    where dh_norm = dy @ W2^T (matmul), and h_norm/rrms are saved from forward.
    """
    assert dy.shape[1] == w2_t.shape[0]
    assert dy.is_contiguous()
    M, K = dy.shape
    _, N = w2_t.shape

    dh = out
    if dh is None:
        dh = torch.empty((M, N), device=dy.device, dtype=torch.bfloat16)

    # num_ctas is derived per-config from N // BN in get_cuda_autotune_config
    if h_norm is None:
        h_norm = torch.empty((128, N), device=dy.device, dtype=torch.bfloat16)
    if rrms is None:
        rrms = torch.empty((128,), device=dy.device, dtype=torch.float32)

    dummy_block = [1, 1]
    a_desc = TensorDescriptor(dy, dy.shape, dy.stride(), dummy_block)
    b_desc = TensorDescriptor(w2_t, w2_t.shape, w2_t.stride(), dummy_block)
    c_desc = TensorDescriptor(dh, dh.shape, dh.stride(), dummy_block)
    hn_desc = TensorDescriptor(h_norm, h_norm.shape, h_norm.stride(), dummy_block)

    NUM_SMS = _num_sms(dy.device)

    def grid(META):
        num_tiles = triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(
            N,
            META["BLOCK_SIZE_N"],
        )
        if META["CLUSTER_LAUNCH_CONTROL"]:
            return (num_tiles,)
        return (min(num_tiles, NUM_SMS),)

    cache_key = (N, fused_rms_norm_bwd)
    if cache_key not in _kernel_cache:
        _kernel_cache[cache_key] = triton.autotune(
            configs=get_reduced_autotune_config(N, fused_rms_norm_bwd)
            if DISABLE_AUTOTUNE
            else get_cuda_autotune_config(N, fused_rms_norm_bwd),
            key=["M", "N", "K"],
            prune_configs_by={"early_config_prune": preprocess_configs},
        )(matmul_rmsnorm_bwd_kernel)

    _kernel_cache[cache_key][grid](
        a_desc,
        b_desc,
        c_desc,
        hn_desc,
        h_norm,
        rrms,
        M,
        N,
        K,
        FUSED_RMS_NORM_BWD=fused_rms_norm_bwd,
        NUM_SMS=NUM_SMS,
    )
    return dh
