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


def selected_autotune_configs(configs, config_index_env="TLX_CONFIG_INDEX"):
    config_index = os.environ.get(config_index_env, os.environ.get("TLX_CONFIG_INDEX"))
    if config_index is None:
        return configs

    index = int(config_index)
    if index < 0 or index >= len(configs):
        raise ValueError(
            f"{config_index_env}={index} out of range for {len(configs)} configs"
        )
    print(f"Selecting autotune config index {index} of {len(configs)}")
    return [configs[index]]


def _is_valid_cta_config(N, BN):
    """Check BLOCK_N produces valid CTA clustering for fused mode."""
    if N % BN != 0:
        return False
    num_ctas = N // BN
    if num_ctas < 2:
        return False
    return num_ctas <= 8


def get_reduced_autotune_config(N, fused):
    """Single config for ADS_MKL_DISABLE_AUTOTUNE=1."""
    if fused:
        for BN in [128]:
            if not _is_valid_cta_config(N, BN):
                continue
            num_ctas = N // BN
            return [
                triton.Config(
                    {
                        "BLOCK_SIZE_M": 128,
                        "BLOCK_SIZE_N": BN,
                        "BLOCK_SIZE_K": 64,
                        "GROUP_SIZE_M": 8,
                        "NUM_SMEM_BUFFERS": 3,
                        "NUM_TMEM_BUFFERS": 2,
                        "NUM_MMA_GROUPS": 1,
                        "EPILOGUE_SUBTILE": 1,
                        "PAIR_CTA": False,
                        "NUM_REDUCTION_CTAS": num_ctas,
                        "CLUSTER_LAUNCH_CONTROL": True,
                        "DIRECT_HN_LOAD": False,
                        "NUM_H_BUFFERS": 1,
                        "STAGE_DH": False,
                        "NUM_C_BUFFERS": 2,
                    },
                    num_warps=4,
                    num_stages=1,
                    pre_hook=matmul_tma_set_block_size_hook,
                    ctas_per_cga=(num_ctas, 1, 1),
                )
            ]
        return []

    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "NUM_SMEM_BUFFERS": 3,
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
            },
            num_warps=4,
            num_stages=1,
            pre_hook=matmul_tma_set_block_size_hook,
            ctas_per_cga=(1, 1, 1),
        )
    ]


def get_cuda_autotune_config(N, fused):
    if fused:
        fused_candidates = [
            # BM, BN, BK, smem, tmem, subtile, num_warps, c_buffers, mma_groups, direct_hn_load, epilogue_warps, approx_mean_dyhat, pipeline_stores
            (128, 128, 64, 5, 2, 4, 4, 2, 1, False, 2, False),
            (128, 128, 64, 5, 2, 4, 8, 2, 1, False, 2, False),
            (128, 128, 64, 4, 2, 2, 4, 2, 1, False, 2, False),
            (128, 128, 64, 4, 2, 2, 4, 2, 1, False, 2, False, True),
            (128, 128, 64, 4, 2, 8, 4, 2, 1, False, 2, False),
            (64, 128, 64, 5, 2, 4, 4, 2, 1, False, 2, False),
            (64, 128, 64, 5, 2, 4, 8, 2, 1, False, 2, False),
            (64, 256, 64, 3, 2, 4, 4, 2, 1, False, 2, False),
            (64, 256, 64, 3, 2, 8, 4, 2, 1, False, 2, False),
            (128, 256, 64, 2, 1, 4, 4, 2, 1, False, 2, False),
            (128, 256, 64, 2, 1, 2, 4, 1, 1, False, 2, False),
            (128, 256, 64, 2, 2, 4, 4, 2, 1, False, 2, False),
            (128, 256, 64, 3, 2, 8, 4, 2, 1, False, 2, False),
            (128, 256, 64, 2, 1, 2, 4, 1, 1, False, 1, False),
            (128, 256, 64, 2, 1, 2, 4, 1, 1, False, 4, False),
            (128, 256, 64, 2, 1, 2, 4, 1, 1, False, 8, False),
        ]
        if N == 512:
            fused_candidates = [
                candidate
                for candidate in fused_candidates
                if candidate[:6] != (128, 128, 64, 5, 2, 4) or candidate[7] != 2
            ]
        configs = []
        for candidate in fused_candidates:
            (
                BM,
                BN,
                BK,
                s,
                t,
                subtile,
                nw,
                c_buffers,
                m,
                direct_hn_load,
                epilogue_warps,
                approx_mean_dyhat,
            ) = candidate[:12]
            pipeline_stores = len(candidate) > 12 and candidate[12]
            if not _is_valid_cta_config(N, BN):
                continue
            if N >= 2048 and BM == 64:
                continue
            if not (N // BN <= 8 and 64 <= BM // m <= 128):
                continue
            if BN % subtile != 0:
                continue
            configs.append(
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
                        "NUM_REDUCTION_CTAS": N // BN,
                        "CLUSTER_LAUNCH_CONTROL": True,
                        "DIRECT_HN_LOAD": direct_hn_load,
                        "NUM_H_BUFFERS": 1,
                        "STAGE_DH": False,
                        "NUM_C_BUFFERS": c_buffers,
                        "EPILOGUE_NUM_WARPS": epilogue_warps,
                        "APPROX_MEAN_DYHAT_ZERO": approx_mean_dyhat,
                        "PIPELINE_STORES": pipeline_stores,
                    },
                    num_warps=nw,
                    num_stages=1,
                    pre_hook=matmul_tma_set_block_size_hook,
                    ctas_per_cga=(N // BN, 1, 1),
                )
            )
        return selected_autotune_configs(configs, "TLX_FUSED_CONFIG_INDEX")
    else:
        configs = [
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
                },
                num_warps=4,
                num_stages=1,
                pre_hook=matmul_tma_set_block_size_hook,
                ctas_per_cga=(1, 1, 1),
            )
            for BM in [128]
            for BN in [128, 256]
            for BK in [64, 128]
            for s in [4, 5, 6]
            for t in [2, 3]
            for m in [1]
            for subtile in [1, 2, 4]
        ]
        for BM, BN, s, t, c_buffers, subtile in [
            (256, 128, 2, 1, 1, 2),
            (256, 128, 2, 1, 1, 4),
            (256, 128, 2, 1, 2, 4),
            (256, 128, 3, 1, 1, 4),
            (256, 128, 3, 1, 2, 4),
            (256, 256, 1, 1, 1, 4),
            (256, 256, 1, 1, 1, 8),
            (256, 256, 2, 1, 1, 4),
            (256, 256, 2, 1, 1, 8),
        ]:
            configs.append(
                triton.Config(
                    {
                        "BLOCK_SIZE_M": BM,
                        "BLOCK_SIZE_N": BN,
                        "BLOCK_SIZE_K": 64,
                        "GROUP_SIZE_M": 8,
                        "NUM_SMEM_BUFFERS": s,
                        "NUM_TMEM_BUFFERS": t,
                        "NUM_MMA_GROUPS": 2,
                        "EPILOGUE_SUBTILE": subtile,
                        "PAIR_CTA": False,
                        "NUM_REDUCTION_CTAS": 1,
                        "CLUSTER_LAUNCH_CONTROL": True,
                        "DIRECT_HN_LOAD": False,
                        "NUM_H_BUFFERS": 2,
                        "STAGE_DH": False,
                        "NUM_C_BUFFERS": c_buffers,
                    },
                    num_warps=4,
                    num_stages=1,
                    pre_hook=matmul_tma_set_block_size_hook,
                    ctas_per_cga=(1, 1, 1),
                )
            )
        return selected_autotune_configs(configs, "TLX_UNFUSED_CONFIG_INDEX")


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


def preprocess_configs(configs, named_args, **kwargs):  # noqa: C901
    MAX_SHARED_MEMORY = 232 * 1024
    MAX_TENSOR_MEMORY = 256 * 1024
    MBARRIER_SIZE = 8
    CLC_RESPONSE_SIZE = 16

    pruned_configs = []
    for conf in configs:
        M = named_args["M"]
        N = named_args["N"]
        K = named_args["K"]
        BLOCK_M = conf.kwargs["BLOCK_SIZE_M"]
        BLOCK_N = conf.kwargs["BLOCK_SIZE_N"]
        BLOCK_K = conf.kwargs["BLOCK_SIZE_K"]
        NUM_SMEM_BUFFERS = conf.kwargs["NUM_SMEM_BUFFERS"]
        NUM_TMEM_BUFFERS = conf.kwargs["NUM_TMEM_BUFFERS"]
        PAIR_CTA = conf.kwargs["PAIR_CTA"]
        CLUSTER_LAUNCH_CONTROL = conf.kwargs["CLUSTER_LAUNCH_CONTROL"]
        NUM_MMA_GROUPS = conf.kwargs["NUM_MMA_GROUPS"]
        NUM_REDUCTION_CTAS = conf.kwargs["NUM_REDUCTION_CTAS"]
        EPILOGUE_SUBTILE = conf.kwargs["EPILOGUE_SUBTILE"]
        NUM_C_BUFFERS = conf.kwargs.get("NUM_C_BUFFERS", 2)
        APPROX_MEAN_DYHAT_ZERO = conf.kwargs.get("APPROX_MEAN_DYHAT_ZERO", False)
        PIPELINE_STORES = conf.kwargs.get("PIPELINE_STORES", False)
        DIRECT_HN_LOAD = conf.kwargs.get("DIRECT_HN_LOAD", False)
        NUM_H_BUFFERS = conf.kwargs.get("NUM_H_BUFFERS", 2)
        STAGE_DH = conf.kwargs.get("STAGE_DH", False)
        uses_fused_reduction = NUM_REDUCTION_CTAS != 1

        if BLOCK_M // NUM_MMA_GROUPS > 128:
            continue
        if APPROX_MEAN_DYHAT_ZERO and K != 512:
            continue
        if BLOCK_M > M:
            continue
        if N % BLOCK_N != 0:
            continue
        expected_ctas = N // BLOCK_N
        if NUM_REDUCTION_CTAS == 1:
            if conf.ctas_per_cga[0] != 1:
                continue
        else:
            if conf.ctas_per_cga[0] != expected_ctas:
                continue
            if NUM_REDUCTION_CTAS != expected_ctas:
                continue

        num_tiles_m = math.ceil(M / BLOCK_M)
        num_tiles_n = math.ceil(N / BLOCK_N)
        store_pipeline_compatible = (
            uses_fused_reduction
            and NUM_REDUCTION_CTAS == 4
            and BLOCK_M == 128
            and BLOCK_N == 128
            and BLOCK_K == 64
            and NUM_SMEM_BUFFERS == 4
            and NUM_TMEM_BUFFERS == 2
            and NUM_MMA_GROUPS == 1
            and EPILOGUE_SUBTILE == 2
            and NUM_C_BUFFERS == 2
            and num_tiles_m <= 2048
        )
        if PIPELINE_STORES != store_pipeline_compatible:
            continue

        pair_cta_compatible = (
            num_tiles_m % 2 == 0 and (num_tiles_m * num_tiles_n) % 2 == 0
        )
        if not pair_cta_compatible:
            conf.kwargs["PAIR_CTA"] = False
            PAIR_CTA = False

        BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
        smem_a = BLOCK_M * BLOCK_K * 2 * NUM_SMEM_BUFFERS
        smem_b_size = (BLOCK_N // 2) if PAIR_CTA else BLOCK_N
        smem_b = BLOCK_K * smem_b_size * 2 * NUM_SMEM_BUFFERS
        smem_epilog = BLOCK_M_SPLIT * (BLOCK_N // EPILOGUE_SUBTILE) * 2 * NUM_C_BUFFERS
        smem_barriers = NUM_SMEM_BUFFERS * MBARRIER_SIZE
        if PAIR_CTA:
            smem_barriers += NUM_SMEM_BUFFERS * NUM_MMA_GROUPS * MBARRIER_SIZE
        smem_barriers += NUM_TMEM_BUFFERS
        smem_clc = (
            (CLC_RESPONSE_SIZE + MBARRIER_SIZE * 2) if CLUSTER_LAUNCH_CONTROL else 0
        )
        smem_hn = (
            0
            if not uses_fused_reduction or DIRECT_HN_LOAD
            else BLOCK_M_SPLIT * BLOCK_N * 2 * NUM_H_BUFFERS
        )
        smem_dh = (
            BLOCK_M_SPLIT * BLOCK_N * 2 if uses_fused_reduction and STAGE_DH else 0
        )
        if not uses_fused_reduction:
            smem_reduction = 0
        elif NUM_REDUCTION_CTAS == 2:
            smem_reduction = BLOCK_M_SPLIT * 4 * 2
        else:
            NUM_DSMEM_BUFFERS = 3
            smem_reduction = (
                BLOCK_M_SPLIT * 4 * (NUM_REDUCTION_CTAS - 1) * 2 * NUM_DSMEM_BUFFERS
            )
        smem_grad_accum = BLOCK_N * 4 * 2
        total_smem = (
            smem_a
            + smem_b
            + smem_epilog
            + smem_barriers
            + smem_clc
            + smem_hn
            + smem_dh
            + smem_reduction
            + (smem_grad_accum if uses_fused_reduction else 0)
        )
        if total_smem > MAX_SHARED_MEMORY:
            continue

        # h_norm TMEM staging buffer
        total_tmem = BLOCK_M * BLOCK_N * 4 * NUM_TMEM_BUFFERS
        if uses_fused_reduction and not DIRECT_HN_LOAD:
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


# -- MMA consumer ---------------------------------------------------------------


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


# -- TMA producer ---------------------------------------------------------------


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
            a_desc, buffers_A[a_buf], [offs_am, offs_k], A_smem_full_bars[a_buf]
        )

        last_a_buf = (NUM_MMA_GROUPS - 1) * NUM_SMEM_BUFFERS + buf
        tlx.barrier_wait(A_smem_empty_bars[last_a_buf], phase ^ 1)
        tlx.barrier_expect_bytes(B_smem_full_bars[buf], expected_bytes)
        tlx.async_descriptor_load(
            b_desc, buffers_B[buf], [offs_k, offs_bn], B_smem_full_bars[buf]
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
            )

        smem_accum_cnt += 1

    return smem_accum_cnt


# -- main kernel -----------------------------------------------------------------


@triton.jit
def matmul_layernorm_bwd_kernel(  # noqa: C901
    a_desc,
    b_desc,
    c_desc,
    hn_desc,
    h_norm_ptr,
    rstd_ptr,
    ln_weight_ptr,
    dweight_ptr,
    dbias_ptr,
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
    NUM_SMS: tl.constexpr,
    FUSED_LAYER_NORM_BWD: tl.constexpr,
    NUM_REDUCTION_CTAS: tl.constexpr,
    NUM_C_BUFFERS: tl.constexpr,
    HAS_WEIGHT_BIAS: tl.constexpr = False,
    EPILOGUE_NUM_WARPS: tl.constexpr = 2,
    APPROX_MEAN_DYHAT_ZERO: tl.constexpr = False,
    PIPELINE_STORES: tl.constexpr = False,
):
    NUM_CTAS_IN_CLUSTER: tl.constexpr = NUM_REDUCTION_CTAS
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_SIZE_M // NUM_MMA_GROUPS
    NUM_DSMEM_BUFFERS: tl.constexpr = 3
    NUM_DSMEM_PEER_SLOTS: tl.constexpr = NUM_REDUCTION_CTAS - 1

    if FUSED_LAYER_NORM_BWD and not DIRECT_HN_LOAD:
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

    if HAS_WEIGHT_BIAS:
        dw_accum = tlx.local_alloc((1, BLOCK_SIZE_N), tl.float32, 1)
        db_accum = tlx.local_alloc((1, BLOCK_SIZE_N), tl.float32, 1)

    if FUSED_LAYER_NORM_BWD:
        if NUM_REDUCTION_CTAS == 2:
            reduction_barriers = tlx.alloc_barriers(num_barriers=1)
            if not APPROX_MEAN_DYHAT_ZERO:
                reduction_buf_sum = tlx.local_alloc((BLOCK_M_SPLIT, 1), tl.float32, 1)
            reduction_buf_dot = tlx.local_alloc((BLOCK_M_SPLIT, 1), tl.float32, 1)
            reduction_phase = 0
        else:
            reduction_barriers = tlx.alloc_barriers(num_barriers=NUM_DSMEM_BUFFERS)
            if not APPROX_MEAN_DYHAT_ZERO:
                reduction_buf_sum = tlx.local_alloc(
                    (BLOCK_M_SPLIT, 1),
                    tl.float32,
                    NUM_DSMEM_PEER_SLOTS * NUM_DSMEM_BUFFERS,
                )
            reduction_buf_dot = tlx.local_alloc(
                (BLOCK_M_SPLIT, 1),
                tl.float32,
                NUM_DSMEM_PEER_SLOTS * NUM_DSMEM_BUFFERS,
            )
        NUM_REDUCED_VALUES: tl.constexpr = 1 if APPROX_MEAN_DYHAT_ZERO else 2
        cross_cta_expected_bytes: tl.constexpr = (
            BLOCK_M_SPLIT
            * tlx.size_of(tl.float32)
            * (NUM_REDUCTION_CTAS - 1)
            * NUM_REDUCED_VALUES
        )

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
    if FUSED_LAYER_NORM_BWD and STAGE_DH:
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

    if FUSED_LAYER_NORM_BWD and not DIRECT_HN_LOAD:
        hn_expected_bytes: tl.constexpr = (
            BLOCK_M_SPLIT * BLOCK_SIZE_N * tlx.size_of(tl.bfloat16)
        )

    with tlx.async_tasks():
        # -- epilogue --------------------------------------------------------
        with tlx.async_task("default", num_warps=EPILOGUE_NUM_WARPS):
            start_pid, num_pid_m, num_pid_n, num_tiles, k_tiles = _compute_grid_info(
                M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
            )

            tmem_accum_cnt = 0
            tile_id = start_pid
            clc_phase_producer = 1
            clc_phase_consumer = 0
            hn_consume_cnt = 0

            if HAS_WEIGHT_BIAS:
                tlx.local_store(
                    dw_accum[0],
                    tl.zeros((1, BLOCK_SIZE_N), dtype=tl.float32),
                )
                tlx.local_store(
                    db_accum[0],
                    tl.zeros((1, BLOCK_SIZE_N), dtype=tl.float32),
                )

            if FUSED_LAYER_NORM_BWD:
                if NUM_REDUCTION_CTAS != 2:
                    dsmem_buffer_id = 0
                    dsmem_producer_phase = 0

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
                cur_tmem_buf, tmem_read_phase = _get_bufidx_phase(
                    tmem_accum_cnt,
                    NUM_TMEM_BUFFERS,
                )

                for group_id in tl.static_range(NUM_MMA_GROUPS):
                    buf_idx = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
                    offs_am = pid_m * BLOCK_SIZE_M + group_id * BLOCK_M_SPLIT

                    if FUSED_LAYER_NORM_BWD:
                        cta_rank = tlx.cluster_cta_rank()
                        offs_m = offs_am + tl.arange(0, BLOCK_M_SPLIT)

                        # Producer starts this TMA before the K-loop, so stage
                        # h_norm while the matmul result may still be finishing.
                        if not DIRECT_HN_LOAD:
                            hn_buf, hn_phase = _get_bufidx_phase(
                                hn_consume_cnt,
                                NUM_H_BUFFERS,
                            )
                            tlx.barrier_wait(hn_full_bars[hn_buf], hn_phase)
                            hn_raw = tlx.local_load(hn_smem[hn_buf])
                            tlx.barrier_arrive(hn_empty_bars[hn_buf], 1)
                            hn_consume_cnt += 1

                            # TMEM staging for tl.sum layout compatibility
                            tlx.local_store(hn_tmem[0], hn_raw)

                        rstd_vals = tl.load(
                            rstd_ptr + offs_m,
                            mask=offs_m < M,
                        )[:, None]

                        # Wait for matmul result
                        tlx.barrier_wait(tmem_full_bars[buf_idx], tmem_read_phase)
                        acc_tmem = tmem_buffers[buf_idx]

                        # Pass 1: accumulate local_sum_dyhat and local_sum_dyhat_xhat
                        # across subtiles (don't signal tmem_empty yet -- we
                        # re-read in pass 2)
                        if not APPROX_MEAN_DYHAT_ZERO:
                            local_sum_dyhat = tl.zeros(
                                (BLOCK_M_SPLIT, 1),
                                dtype=tl.float32,
                            )
                        local_sum_dyhat_xhat = tl.zeros(
                            (BLOCK_M_SPLIT, 1),
                            dtype=tl.float32,
                        )
                        for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                            dh_sub = tlx.local_slice(
                                acc_tmem,
                                [0, slice_id * slice_size],
                                [BLOCK_M_SPLIT, slice_size],
                            )
                            dh_slice = tlx.local_load(dh_sub).to(tl.float32)
                            if STAGE_DH:
                                tlx.local_store(
                                    dh_smem_buffers[slice_id],
                                    dh_slice.to(tl.bfloat16),
                                )

                            if HAS_WEIGHT_BIAS:
                                ln_w_slice = tl.load(
                                    ln_weight_ptr
                                    + offs_bn
                                    + slice_id * slice_size
                                    + tl.arange(0, slice_size)
                                ).to(tl.float32)
                                dy_hat_slice = dh_slice * ln_w_slice[None, :]
                            else:
                                dy_hat_slice = dh_slice

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
                                ).to(tl.float32)
                            else:
                                hn_sub = tlx.local_slice(
                                    hn_tmem[0],
                                    [0, slice_id * slice_size],
                                    [BLOCK_M_SPLIT, slice_size],
                                )
                                hn_slice = tlx.local_load(hn_sub).to(tl.float32)

                            if not APPROX_MEAN_DYHAT_ZERO:
                                local_sum_dyhat += tl.sum(
                                    dy_hat_slice,
                                    axis=1,
                                    keep_dims=True,
                                )
                            local_sum_dyhat_xhat += tl.sum(
                                dy_hat_slice * hn_slice,
                                axis=1,
                                keep_dims=True,
                            )

                        if STAGE_DH:
                            tlx.fence_async_shared()
                            tlx.barrier_arrive(tmem_empty_bars[buf_idx], 1)

                        # DSMEM cross-CTA reduction.
                        if NUM_REDUCTION_CTAS == 2:
                            tlx.barrier_expect_bytes(
                                reduction_barriers[0],
                                cross_cta_expected_bytes,
                            )
                            for i in tl.static_range(NUM_REDUCTION_CTAS):
                                if cta_rank != i:
                                    if not APPROX_MEAN_DYHAT_ZERO:
                                        tlx.async_remote_shmem_store(
                                            dst=reduction_buf_sum[0],
                                            src=local_sum_dyhat,
                                            remote_cta_rank=i,
                                            barrier=reduction_barriers[0],
                                        )
                                    tlx.async_remote_shmem_store(
                                        dst=reduction_buf_dot[0],
                                        src=local_sum_dyhat_xhat,
                                        remote_cta_rank=i,
                                        barrier=reduction_barriers[0],
                                    )

                            tlx.barrier_wait(
                                reduction_barriers[0],
                                phase=reduction_phase,
                            )
                            if not APPROX_MEAN_DYHAT_ZERO:
                                global_sum_dyhat = local_sum_dyhat + tlx.local_load(
                                    reduction_buf_sum[0]
                                ).to(tl.float32)
                            global_sum_dyhat_xhat = (
                                local_sum_dyhat_xhat
                                + tlx.local_load(reduction_buf_dot[0]).to(tl.float32)
                            )
                            reduction_phase ^= 1
                        else:
                            tlx.barrier_expect_bytes(
                                reduction_barriers[dsmem_buffer_id],
                                cross_cta_expected_bytes,
                            )
                            buf_offset = dsmem_buffer_id * NUM_DSMEM_PEER_SLOTS
                            for i in tl.static_range(NUM_REDUCTION_CTAS):
                                if cta_rank != i:
                                    dst_idx = cta_rank if cta_rank < i else cta_rank - 1
                                    if not APPROX_MEAN_DYHAT_ZERO:
                                        tlx.async_remote_shmem_store(
                                            dst=reduction_buf_sum[buf_offset + dst_idx],
                                            src=local_sum_dyhat,
                                            remote_cta_rank=i,
                                            barrier=reduction_barriers[dsmem_buffer_id],
                                        )
                                    tlx.async_remote_shmem_store(
                                        dst=reduction_buf_dot[buf_offset + dst_idx],
                                        src=local_sum_dyhat_xhat,
                                        remote_cta_rank=i,
                                        barrier=reduction_barriers[dsmem_buffer_id],
                                    )

                            tlx.barrier_wait(
                                reduction_barriers[dsmem_buffer_id],
                                phase=dsmem_producer_phase,
                            )
                            if not APPROX_MEAN_DYHAT_ZERO:
                                global_sum_dyhat = local_sum_dyhat
                            global_sum_dyhat_xhat = local_sum_dyhat_xhat
                            for i in tl.static_range(NUM_DSMEM_PEER_SLOTS):
                                if not APPROX_MEAN_DYHAT_ZERO:
                                    global_sum_dyhat += tlx.local_load(
                                        tlx.local_view(
                                            reduction_buf_sum,
                                            buf_offset + i,
                                        )
                                    ).to(tl.float32)
                                global_sum_dyhat_xhat += tlx.local_load(
                                    tlx.local_view(
                                        reduction_buf_dot,
                                        buf_offset + i,
                                    )
                                ).to(tl.float32)

                        inv_N = 1.0 / N
                        if APPROX_MEAN_DYHAT_ZERO:
                            mean_dyhat = 0.0
                        else:
                            mean_dyhat = global_sum_dyhat * inv_N
                        mean_dyhat_xhat = global_sum_dyhat_xhat * inv_N

                        # Pass 2: re-read TMEM, apply layernorm backward
                        # correction, store, signal tmem_empty
                        # dz = rstd * (dy_hat - mean_dyhat - z_norm * mean_dyhat_xhat)
                        for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                            if STAGE_DH:
                                dh_slice = tlx.local_load(dh_smem_buffers[slice_id]).to(
                                    tl.float32
                                )
                            else:
                                dh_sub = tlx.local_slice(
                                    acc_tmem,
                                    [0, slice_id * slice_size],
                                    [BLOCK_M_SPLIT, slice_size],
                                )
                                dh_slice = tlx.local_load(dh_sub).to(tl.float32)

                            if HAS_WEIGHT_BIAS:
                                ln_w_slice = tl.load(
                                    ln_weight_ptr
                                    + offs_bn
                                    + slice_id * slice_size
                                    + tl.arange(0, slice_size)
                                ).to(tl.float32)
                                dy_hat_slice = dh_slice * ln_w_slice[None, :]
                            else:
                                dy_hat_slice = dh_slice

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
                                ).to(tl.float32)
                            else:
                                hn_sub = tlx.local_slice(
                                    hn_tmem[0],
                                    [0, slice_id * slice_size],
                                    [BLOCK_M_SPLIT, slice_size],
                                )
                                hn_slice = tlx.local_load(hn_sub).to(tl.float32)

                            result_out = rstd_vals * (
                                dy_hat_slice - mean_dyhat - hn_slice * mean_dyhat_xhat
                            )
                            if not STAGE_DH and slice_id == EPILOGUE_SUBTILE - 1:
                                tlx.barrier_arrive(tmem_empty_bars[buf_idx], 1)
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

                            if HAS_WEIGHT_BIAS:
                                dw_sub = tlx.local_slice(
                                    dw_accum[0],
                                    [0, slice_id * slice_size],
                                    [1, slice_size],
                                )
                                db_sub = tlx.local_slice(
                                    db_accum[0],
                                    [0, slice_id * slice_size],
                                    [1, slice_size],
                                )
                                cur_dw = tlx.local_load(dw_sub)
                                cur_db = tlx.local_load(db_sub)
                                dh_bf16 = dh_slice.to(tl.bfloat16)
                                hn_bf16 = hn_slice.to(tl.bfloat16)
                                cur_dw += tl.sum(
                                    dh_bf16 * hn_bf16, axis=0, keep_dims=True
                                ).to(tl.float32)
                                cur_db += tl.sum(dh_bf16, axis=0, keep_dims=True).to(
                                    tl.float32
                                )
                                tlx.local_store(dw_sub, cur_dw)
                                tlx.local_store(db_sub, cur_db)

                        if NUM_REDUCTION_CTAS != 2:
                            dsmem_buffer_id += 1
                            if dsmem_buffer_id == NUM_DSMEM_BUFFERS:
                                dsmem_buffer_id = 0
                                dsmem_producer_phase = dsmem_producer_phase ^ 1

                    else:
                        tlx.barrier_wait(tmem_full_bars[buf_idx], tmem_read_phase)
                        acc_tmem = tmem_buffers[buf_idx]
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
                            tlx.local_store(c_smem, result.to(tlx.dtype_of(c_desc)))
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
                if PIPELINE_STORES:
                    tlx.async_descriptor_store_wait(NUM_C_BUFFERS - 1)
                else:
                    tlx.async_descriptor_store_wait(0)

            if PIPELINE_STORES:
                tlx.async_descriptor_store_wait(0)
            # After persistent loop: one atomic_add per CTA
            if HAS_WEIGHT_BIAS:
                cta_rank_final = tlx.cluster_cta_rank()
                final_dw = tlx.local_load(dw_accum[0])
                final_db = tlx.local_load(db_accum[0])
                offs_n = cta_rank_final * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                tl.atomic_add(
                    dweight_ptr + offs_n,
                    tl.reshape(final_dw, (BLOCK_SIZE_N,)),
                )
                tl.atomic_add(
                    dbias_ptr + offs_n,
                    tl.reshape(final_db, (BLOCK_SIZE_N,)),
                )

        # -- MMA consumer ----------------------------------------------------
        with tlx.async_task(num_warps=1, num_regs=24):
            start_pid, num_pid_m, num_pid_n, num_tiles, k_tiles = _compute_grid_info(
                M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
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

        # -- producer (TMA load) ---------------------------------------------
        with tlx.async_task(num_warps=1, num_regs=24):
            start_pid, num_pid_m, num_pid_n, num_tiles, k_tiles = _compute_grid_info(
                M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
            )

            smem_accum_cnt = 0
            tile_id = start_pid
            clc_phase_consumer = 0
            hn_prod_cnt = 0

            while tile_id != -1:
                if FUSED_LAYER_NORM_BWD and not DIRECT_HN_LOAD:
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


# -- host-side launch ------------------------------------------------------------

_kernel_cache = {}
_dummy_float32_cache = {}
_num_sms_cache = {}


def _dummy_float32(device):
    if device not in _dummy_float32_cache:
        _dummy_float32_cache[device] = torch.empty(
            (1,),
            device=device,
            dtype=torch.float32,
        )
    return _dummy_float32_cache[device]


def _num_sms(device):
    if device not in _num_sms_cache:
        _num_sms_cache[device] = torch.cuda.get_device_properties(
            device,
        ).multi_processor_count
    return _num_sms_cache[device]


def tlx_matmul_layernorm_bwd(
    dy,
    w2_t,
    h_norm=None,
    rstd=None,
    ln_weight=None,
    fused_layer_norm_bwd=False,
    has_weight_bias=False,
    out=None,
):
    """
    Fused: dz = rstd * (dy_hat - (1/N)*sum_dyhat - (1/N)*z_norm*sum_dyhat_xhat)
    where dy_hat = dout * ln_weight, dout = dy @ W2^T (matmul),
    z_norm/rstd are saved from forward, and ln_weight is the layernorm weight.

    When has_weight_bias=True, also computes dweight and dbias via SMEM
    accumulation + atomic_add. Returns (dz, dweight, dbias).
    """
    assert dy.shape[1] == w2_t.shape[0]
    assert dy.is_contiguous()
    M, K = dy.shape
    _, N = w2_t.shape

    if out is None:
        dz = torch.empty((M, N), device=dy.device, dtype=torch.bfloat16)
    else:
        assert out.shape == (M, N)
        assert out.is_contiguous()
        dz = out
    if has_weight_bias:
        assert out is None
        dweight = torch.zeros((N,), device=dy.device, dtype=torch.float32)
        dbias = torch.zeros((N,), device=dy.device, dtype=torch.float32)
    else:
        dweight = _dummy_float32(dy.device)
        dbias = dweight

    if h_norm is None:
        h_norm = torch.empty((128, N), device=dy.device, dtype=torch.bfloat16)
    if rstd is None:
        rstd = torch.empty((128,), device=dy.device, dtype=torch.float32)
    if ln_weight is None:
        if has_weight_bias:
            ln_weight = torch.empty((N,), device=dy.device, dtype=torch.bfloat16)
        else:
            ln_weight = dy

    if fused_layer_norm_bwd and not has_weight_bias and K >= 2048 and N >= 2048:
        dh_norm = tlx_matmul_layernorm_bwd(dy, w2_t, out=dz)
        return layernorm_bwd_post(dh_norm, h_norm, rstd, out=dh_norm)

    dummy_block = [1, 1]
    a_desc = TensorDescriptor(dy, dy.shape, dy.stride(), dummy_block)
    b_desc = TensorDescriptor(w2_t, w2_t.shape, w2_t.stride(), dummy_block)
    c_desc = TensorDescriptor(dz, dz.shape, dz.stride(), dummy_block)
    hn_desc = TensorDescriptor(h_norm, h_norm.shape, h_norm.stride(), dummy_block)

    NUM_SMS = _num_sms(dy.device)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    cache_key = (N, fused_layer_norm_bwd, has_weight_bias)
    if cache_key not in _kernel_cache:
        _kernel_cache[cache_key] = triton.autotune(
            configs=get_reduced_autotune_config(N, fused_layer_norm_bwd)
            if DISABLE_AUTOTUNE
            else get_cuda_autotune_config(N, fused_layer_norm_bwd),
            key=["M", "N", "K"],
            prune_configs_by={"early_config_prune": preprocess_configs},
            restore_value=["dweight_ptr", "dbias_ptr"],
        )(matmul_layernorm_bwd_kernel)

    _kernel_cache[cache_key][grid](
        a_desc,
        b_desc,
        c_desc,
        hn_desc,
        h_norm,
        rstd,
        ln_weight,
        dweight,
        dbias,
        M,
        N,
        K,
        FUSED_LAYER_NORM_BWD=fused_layer_norm_bwd,
        HAS_WEIGHT_BIAS=has_weight_bias,
        NUM_SMS=NUM_SMS,
    )
    if has_weight_bias:
        return dz, dweight.to(dy.dtype), dbias.to(dy.dtype)
    return dz


@triton.jit
def layernorm_bwd_post_kernel(
    dh_norm_ptr,
    h_norm_ptr,
    rstd_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    offsets = offs_m[:, None] * N + offs_n[None, :]

    dh_norm = tl.load(dh_norm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    h_norm = tl.load(h_norm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + offs_m, mask=offs_m < M, other=0.0)[:, None]

    mean_dh = tl.sum(dh_norm, axis=1, keep_dims=True) / N
    mean_dh_xhat = tl.sum(dh_norm * h_norm, axis=1, keep_dims=True) / N
    out = (dh_norm - mean_dh - h_norm * mean_dh_xhat) * rstd
    tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=mask)


@triton.jit
def layernorm_bwd_post_fast_kernel(
    dh_norm_ptr,
    h_norm_ptr,
    rstd_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offsets = offs_m[:, None] * N + offs_n[None, :]

    dh_norm = tl.load(dh_norm_ptr + offsets).to(tl.float32)
    h_norm = tl.load(h_norm_ptr + offsets).to(tl.float32)
    rstd = tl.load(rstd_ptr + offs_m)[:, None]

    mean_dh = tl.sum(dh_norm, axis=1, keep_dims=True) / N
    mean_dh_xhat = tl.sum(dh_norm * h_norm, axis=1, keep_dims=True) / N
    out = (dh_norm - mean_dh - h_norm * mean_dh_xhat) * rstd
    tl.store(out_ptr + offsets, out.to(tl.bfloat16))


def layernorm_bwd_post(dh_norm, h_norm, rstd, out=None):
    M, N = dh_norm.shape
    if out is None:
        out = torch.empty_like(dh_norm)
    block_n = triton.next_power_of_2(N)
    block_m = int(os.environ.get("TLX_POST_BLOCK_M", "4"))
    num_warps = int(os.environ.get("TLX_POST_NUM_WARPS", "4"))

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]),)

    if os.environ.get("TLX_POST_FAST") == "1":
        layernorm_bwd_post_fast_kernel[grid](
            dh_norm,
            h_norm,
            rstd,
            out,
            N,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            num_warps=num_warps,
        )
    else:
        layernorm_bwd_post_kernel[grid](
            dh_norm,
            h_norm,
            rstd,
            out,
            M,
            N,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            num_warps=num_warps,
        )
    return out
