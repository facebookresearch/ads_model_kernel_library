# TLX GDPA kernel optimized for Blackwell Warp Specialization

# pyre-ignore-all-errors
import math
import os
from functools import lru_cache
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
import triton  # @manual=//triton:triton
import triton.language as tl  # @manual=//triton:triton
import triton.language.extra.tlx as tlx  # @manual=//triton:triton
import triton.profiler as proton  # @manual=//triton:triton
import triton.profiler.language as pl  # @manual=//triton:triton
from register_helpers import custom_register_kernel
from tlx_math import (
    _add_f32x2,
    _mul_f32x2,
    _sub_f32x2,
    activation_string_to_int,
    apply_activation,
    apply_activation_grad,
    fast_dividef,
)
from torch._inductor.runtime.triton_helpers import libdevice
from torch.utils.flop_counter import (
    _unpack_flash_attention_nested_shapes,
    register_flop_formula,
    sdpa_backward_flop_count,
    sdpa_flop_count,
)
from triton.runtime.jit import JITFunction
from triton.tools.tensor_descriptor import TensorDescriptor  # @manual=//triton:triton
from utils import should_use_i64_idx
from vararg_kernel import unroll_varargs


# to make linter happy
VAR_ARGS_ARRAY = List[Any]

AUTOTUNE_CONFIG_SET = os.environ.get("ADS_MKL_AUTOTUNE_CONFIG_SET", "default")
RESERVED_SMS_FOR_COMMS = int(os.environ.get("ADS_MKL_RESERVED_SMS_FOR_COMMS", "0"))


@lru_cache
def get_num_sms() -> Optional[int]:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties("cuda").multi_processor_count


def is_cuda() -> bool:
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    BLOCK_D = nargs["BLOCK_D"]
    if not isinstance(nargs["Q"], TensorDescriptor):
        # early return for on-device TMA
        return
    NUM_MMA_GROUPS = nargs["NUM_MMA_GROUPS"]
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    nargs["Q"].block_shape = [BLOCK_M_SPLIT, BLOCK_D]
    nargs["V"].block_shape = [BLOCK_N, BLOCK_D]
    nargs["K"].block_shape = [BLOCK_N, BLOCK_D]
    nargs["Out"].block_shape = [BLOCK_M_SPLIT, BLOCK_D]
    if "Residual" in nargs and isinstance(nargs["Residual"], TensorDescriptor):
        nargs["Residual"].block_shape = [BLOCK_M_SPLIT, BLOCK_D]


def _create_autotune_configs(
    pingpong_opts: List[bool],
    ract_opts: List[int],
    rmma_opts: List[int],
    rload_opts: List[int],
    ns_opts: List[int],
    actsplit_opts: List[int],
    use_outer_swp_opts: List[bool] | None = None,
    use_clc: List[bool] | None = None,
    block_m: Tuple[int] = (256,),
    block_n: Tuple[int] = (128,),
    num_bufs_q: Tuple[int] = (1,),
    num_bufs_kv: Tuple[int] = (3,),
    num_bufs_qk: Tuple[int] = (1,),
    num_bufs_o: Tuple[int] = (1,),
    ctas_per_cga: Tuple[int, int, int] = (1, 1, 1),
) -> List[triton.Config]:
    """Helper to generate autotune configs with common parameters."""
    if use_outer_swp_opts is None:
        use_outer_swp_opts = [False]
    if use_clc is None:
        use_clc = [False]
    return [
        triton.Config(
            {
                "BLOCK_M": bm,
                "BLOCK_N": bn,
                "NUM_BUFFERS_Q": buf_q,
                "NUM_BUFFERS_KV": buf_kv,
                "NUM_BUFFERS_QK": buf_qk,
                "NUM_BUFFERS_O": buf_o,
                "SUBTILING": True,
                "PINGPONG": pp,
                "NUM_REGS_ACT": ract,
                "NUM_REGS_MMA": rmma,
                "NUM_REGS_LOAD": rload,
                "NUM_SUBSLICES": ns,
                "NUM_ACT_SPLITS": actsplit,
                "USE_OUTER_SWP": use_outer_swp,
                "USE_CLC": clc,
            },
            num_warps=8,
            num_stages=1,
            pre_hook=_host_descriptor_pre_hook,
            ctas_per_cga=ctas_per_cga,
        )
        for pp in pingpong_opts
        for ract in ract_opts
        for rmma in rmma_opts
        for rload in rload_opts
        for ns in ns_opts
        for actsplit in actsplit_opts
        for use_outer_swp in use_outer_swp_opts
        for clc in use_clc
        for bm in block_m
        for bn in block_n
        for buf_q in num_bufs_q
        for buf_kv in num_bufs_kv
        for buf_qk in num_bufs_qk
        for buf_o in num_bufs_o
    ]


@lru_cache
def get_cuda_autotune_config(
    enable_load_balancing: bool,
    ctas_per_cga: Tuple[int, int, int] = (1, 1, 1),
    num_buffers_kv: int = 3,
):
    # If no load balancing, outer SWP + CLC would create deadlock. Explicitly avoiding that combination here
    use_outer_swp_opts_override = [False] if not enable_load_balancing else None
    if AUTOTUNE_CONFIG_SET == "omnifm_v2":
        return _create_autotune_configs(
            pingpong_opts=[True],
            ract_opts=[256],
            rmma_opts=[64],
            rload_opts=[64],
            ns_opts=[2],
            actsplit_opts=[4],
            use_outer_swp_opts=use_outer_swp_opts_override or [True],
            ctas_per_cga=ctas_per_cga,
            num_bufs_kv=(num_buffers_kv,),
        )
    if AUTOTUNE_CONFIG_SET == "omnifm_v4_disable_autotune":
        return _create_autotune_configs(
            pingpong_opts=[True],
            ract_opts=[192],
            rmma_opts=[24],
            rload_opts=[24],
            ns_opts=[2],
            actsplit_opts=[4],
            use_outer_swp_opts=[True],
            ctas_per_cga=ctas_per_cga,
            num_bufs_kv=(num_buffers_kv,),
        )
    if os.getenv("ENABLE_PROTON") == "1":
        return _create_autotune_configs(
            pingpong_opts=[False],
            ract_opts=[232],
            rmma_opts=[24],
            rload_opts=[24],
            ns_opts=[2],
            actsplit_opts=[4],
            use_outer_swp_opts=use_outer_swp_opts_override or [True],
            use_clc=[True],
            ctas_per_cga=ctas_per_cga,
            num_bufs_kv=(num_buffers_kv,),
        )
    if os.environ.get("ADS_MKL_DISABLE_AUTOTUNE") == "1":
        return _create_autotune_configs(
            pingpong_opts=[True],
            ract_opts=[232],
            rmma_opts=[24],
            rload_opts=[24],
            ns_opts=[4],
            actsplit_opts=[4],
            use_outer_swp_opts=use_outer_swp_opts_override or [False],
            use_clc=[True],
            ctas_per_cga=ctas_per_cga,
            num_bufs_kv=(num_buffers_kv,),
        )
    return _create_autotune_configs(
        pingpong_opts=[True, False],  # Ping pong is useful for 256*128 for pffn case
        ract_opts=[192, 256],
        rmma_opts=[24, 64, 96],
        rload_opts=[24, 64, 96],
        ns_opts=[2, 4],  # 8 is slow
        actsplit_opts=[4, 8, 16],
        use_outer_swp_opts=use_outer_swp_opts_override or [True, False],
        use_clc=[True],
        ctas_per_cga=ctas_per_cga,
        num_bufs_kv=(num_buffers_kv,),
    )


def get_cuda_autotune_config_short_kv(
    block_n: int,
    ctas_per_cga: Tuple[int, int, int] = (1, 1, 1),
    num_buffers_kv: int = 3,
):
    if AUTOTUNE_CONFIG_SET == "omnifm_v2":
        return _create_autotune_configs(
            pingpong_opts=[False],
            ract_opts=[256],
            rmma_opts=[96],
            rload_opts=[96],
            ns_opts=[2],
            actsplit_opts=[4],
            use_outer_swp_opts=[False],
            use_clc=[False],
            block_m=[128],
            block_n=[block_n],
            num_bufs_q=[2],
            num_bufs_kv=[num_buffers_kv],
            num_bufs_qk=[1],
            num_bufs_o=[2],
            ctas_per_cga=ctas_per_cga,
        )
    if (
        os.getenv("ENABLE_PROTON") == "1"
        or os.environ.get("ADS_MKL_DISABLE_AUTOTUNE") == "1"
    ):
        return _create_autotune_configs(
            pingpong_opts=[False],
            ract_opts=[256],
            rmma_opts=[32],
            rload_opts=[32],
            ns_opts=[2],
            actsplit_opts=[4],
            use_outer_swp_opts=[False],
            use_clc=[True],
            block_m=[64],
            block_n=[block_n],
            num_bufs_q=[3],
            num_bufs_kv=[num_buffers_kv],
            num_bufs_qk=[1],
            num_bufs_o=[2],
            ctas_per_cga=ctas_per_cga,
        )
    return _create_autotune_configs(
        pingpong_opts=[False],
        ract_opts=[192, 256],
        rmma_opts=[24, 64, 96],
        rload_opts=[24, 64, 96],
        ns_opts=[1, 2],  # 8 is slow
        actsplit_opts=[4],
        use_outer_swp_opts=[False],
        use_clc=[True],
        block_m=[128],
        block_n=[block_n],
        num_bufs_q=[2, 3],
        num_bufs_kv=[2, 3, 4],
        num_bufs_qk=[1, 2],
        num_bufs_o=[2, 3],
        ctas_per_cga=ctas_per_cga,
    )


@lru_cache
def _get_autotune_kernel(
    kernel: JITFunction,
    enable_load_balancing: bool,
    short_kv: bool = False,
    short_kv_block_n: int = 128,
    ctas_per_cga: Tuple[int, int, int] = (1, 1, 1),
    num_buffers_kv: int = 3,
) -> JITFunction:
    return triton.autotune(
        configs=get_cuda_autotune_config(
            enable_load_balancing, ctas_per_cga, num_buffers_kv
        )
        if not short_kv
        else get_cuda_autotune_config_short_kv(
            short_kv_block_n, ctas_per_cga, num_buffers_kv
        ),
        key=["N_CTX", "HEAD_DIM", "H", "G", "FUSED_QKV", "FUSED_KV"],
    )(kernel)


## Iterative tuning with intra-kernel profiler
## 1. identify critical resource
## 2. assuming it is gemm, make sure there is no bubble in gemm partition

## Potential issues
## -- bubbles in gemm partition due to _compute_qlen
## ---- if that is the case via intra-kernel profiler, try pre-compute _compute_qlen
## -- load imbalance
## ---- use dynamic scheduler
## ---- grab the next tile one iteration ahead (i.e SWP of the outer loop)
## -- if descriptor setup is an issue, try SWP the setup for inner loop (i.e desc_k,v)


## Overall warpspec configuration
## default + 3 partitions:
##   default is activation0 with 4 warps, partition0 is activatation1 with 4 warps
##   partition1 is gemm, partition 2 is load
@triton.jit  # pragma: no cover
def _compute_qlen(
    tile_idx,
    n_tile_num,
    Q_offsets,
    K_offsets,
    seq_index,
    SORT_BY_SEQ_LENGTH: tl.constexpr,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    ENABLE_LOAD_BALANCING: tl.constexpr,
    valid_tiles_b,
    valid_tiles_m,
    NUM_VALID_TILES_PER_HEAD,
    BROADCAST_Q: tl.constexpr,
):
    if ENABLE_LOAD_BALANCING:
        off_tile = tile_idx // H
        off_z = tl.load(valid_tiles_b + off_tile).to(tl.int64)
        begin_q = tl.load(valid_tiles_m + off_tile).to(tl.int64)
        end_q = tl.load(valid_tiles_m + off_tile + 1).to(tl.int64)
    else:
        off_hz = tile_idx // n_tile_num
        off_z = off_hz // H
        if SORT_BY_SEQ_LENGTH:
            off_z = tl.load(seq_index + off_z)
        if not BROADCAST_Q:
            off_q_z = off_z
        else:
            off_q_z = 0
        begin_q = tl.load(Q_offsets + off_q_z)
        end_q = tl.load(Q_offsets + off_q_z + 1)

    qlen = end_q - begin_q
    qlen = tl.minimum(qlen, N_CTX)

    begin_k = tl.load(K_offsets + off_z)
    end_k = tl.load(K_offsets + off_z + 1)
    klen = end_k - begin_k

    return begin_q, end_q, begin_k, end_k, qlen, klen


@triton.jit  # pragma: no cover
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS):
    bufIdx = accum_cnt % NUM_BUFFERS
    phase = (accum_cnt // NUM_BUFFERS) & 1
    return bufIdx, phase


@triton.jit  # pragma: no cover
def _compute_rms_norm(
    x,
    cluster_cta_rank,
    reduction_buf,
    barrier,
    phase,
    buf_offset,
    BLOCK_SIZE_M: tl.constexpr,
    N: tl.constexpr,
    NUM_REDUCTION_CTAS: tl.constexpr,
    expected_bytes: tl.constexpr,
    rms_norm_weight,
    HAS_RMS_NORM_WEIGHT: tl.constexpr,
):
    if NUM_REDUCTION_CTAS:
        tlx.barrier_expect_bytes(barrier, expected_bytes)

    local_partial_sum = tl.sum(x * x, axis=1, keep_dims=True).to(tl.float32)
    for i in tl.static_range(NUM_REDUCTION_CTAS):
        if cluster_cta_rank != i:
            tlx.async_remote_shmem_store(
                dst=reduction_buf[buf_offset + cluster_cta_rank],
                src=local_partial_sum,
                remote_cta_rank=i,
                barrier=barrier,
            )
        else:
            tlx.local_store(
                reduction_buf[buf_offset + cluster_cta_rank], local_partial_sum
            )

    if NUM_REDUCTION_CTAS:
        tlx.barrier_wait(barrier, phase=phase)

    final_square_sum = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    for i in tl.static_range(NUM_REDUCTION_CTAS):
        final_square_sum += tlx.local_load(reduction_buf[buf_offset + i])

    rrms = libdevice.rsqrt(
        fast_dividef(
            final_square_sum,
            tl.full(final_square_sum.shape, N, dtype=tl.float32),
        )
        + 1e-5
    )
    result = x * rrms
    if HAS_RMS_NORM_WEIGHT:
        result = result * rms_norm_weight[None, :]
    return result, rrms


@triton.jit  # pragma: no cover
def _compute_layer_norm(
    x,
    cluster_cta_rank,
    reduction_buf,
    barrier,
    phase,
    buf_offset,
    BLOCK_SIZE_M: tl.constexpr,
    N: tl.constexpr,
    NUM_REDUCTION_CTAS: tl.constexpr,
    expected_bytes: tl.constexpr,
    ln_weight,
    ln_bias,
    HAS_LN_BIAS: tl.constexpr,
):
    """Compute LayerNorm via 1 DSMEM round-trip: send sum(x) and sum(x^2) on the same barrier.

    Uses Var(X) = E[X^2] - E[X]^2 to avoid a second reduction pass.
    Two separate [BLOCK_M, 1] stores share a single barrier with doubled expected_bytes.
    """
    if NUM_REDUCTION_CTAS:
        tlx.barrier_expect_bytes(barrier, expected_bytes)

    # Compute both sum(x) and sum(x^2) locally
    local_sum = tl.sum(x, axis=1, keep_dims=True).to(tl.float32)
    local_sum_sq = tl.sum(_mul_f32x2(x, x), axis=1, keep_dims=True).to(tl.float32)

    # Send both partials via 2 stores to the same barrier
    HALF: tl.constexpr = NUM_REDUCTION_CTAS  # offset between sum and sum_sq slots
    for i in tl.static_range(NUM_REDUCTION_CTAS):
        if cluster_cta_rank != i:
            tlx.async_remote_shmem_store(
                dst=reduction_buf[buf_offset + cluster_cta_rank],
                src=local_sum,
                remote_cta_rank=i,
                barrier=barrier,
            )
            tlx.async_remote_shmem_store(
                dst=reduction_buf[buf_offset + HALF + cluster_cta_rank],
                src=local_sum_sq,
                remote_cta_rank=i,
                barrier=barrier,
            )
        else:
            tlx.local_store(reduction_buf[buf_offset + cluster_cta_rank], local_sum)
            tlx.local_store(
                reduction_buf[buf_offset + HALF + cluster_cta_rank], local_sum_sq
            )

    if NUM_REDUCTION_CTAS:
        tlx.barrier_wait(barrier, phase=phase)

    total_sum = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    total_sum_sq = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    for i in tl.static_range(NUM_REDUCTION_CTAS):
        total_sum += tlx.local_load(reduction_buf[buf_offset + i])
        total_sum_sq += tlx.local_load(reduction_buf[buf_offset + HALF + i])

    mean = fast_dividef(
        total_sum,
        tl.full(total_sum.shape, N, dtype=tl.float32),
    )
    # Var(X) = E[X^2] - E[X]^2
    variance = _sub_f32x2(
        fast_dividef(
            total_sum_sq,
            tl.full(total_sum_sq.shape, N, dtype=tl.float32),
        ),
        _mul_f32x2(mean, mean),
    )
    rstd = libdevice.rsqrt(variance + 1e-5)

    # Normalize using f32x2 for [M,D] ops with [M,1] broadcast
    result = _mul_f32x2(_sub_f32x2(x, mean), rstd)
    # Weight/bias are [1,D] broadcast — f32x2 doesn't support this, use standard ops
    if ln_weight is not None:
        result = result * ln_weight[None, :]
    if HAS_LN_BIAS:
        result = result + ln_bias[None, :]

    return result, mean, rstd


@triton.jit  # pragma: no cover
def _compute_rms_norm_backward(
    dy,
    y,
    rrms_vals,
    cluster_cta_rank,
    reduction_buf,
    barrier,
    phase,
    buf_offset,
    BLOCK_SIZE_M: tl.constexpr,
    N: tl.constexpr,
    NUM_REDUCTION_CTAS: tl.constexpr,
    expected_bytes: tl.constexpr,
    rms_norm_weight,
    HAS_RMS_NORM_WEIGHT: tl.constexpr,
    idx,
    PROTON_TILE: tl.constexpr,
    ENABLE_PROTON: tl.constexpr,
):
    # Compute local partial sum: sum(dy * y, axis=1) per CTA
    if ENABLE_PROTON and idx == PROTON_TILE:
        pl.enter_scope("act_compute_rms_bwd_sum")
    dy_y_local = tl.sum(_mul_f32x2(dy, y), axis=1, keep_dims=True).to(tl.float32)
    if ENABLE_PROTON and idx == PROTON_TILE:
        pl.exit_scope("act_compute_rms_bwd_sum")

    # Cross-CTA DSMEM reduce (same pattern as _compute_rms_norm)
    # Only allocate SMEM slots for remote CTAs; local partial stays in register.
    if ENABLE_PROTON and idx == PROTON_TILE:
        pl.enter_scope("act_compute_rms_bwd_shmem")
    for i in tl.static_range(NUM_REDUCTION_CTAS):
        if cluster_cta_rank != i:
            # Remap: skip local CTA's slot to pack into NUM_REDUCTION_CTAS-1 slots
            dst_idx = cluster_cta_rank if cluster_cta_rank < i else cluster_cta_rank - 1
            tlx.async_remote_shmem_store(
                dst=reduction_buf[buf_offset + dst_idx],
                src=dy_y_local,
                remote_cta_rank=i,
                barrier=barrier,
            )
    if ENABLE_PROTON and idx == PROTON_TILE:
        pl.exit_scope("act_compute_rms_bwd_shmem")

    if ENABLE_PROTON and idx == PROTON_TILE:
        pl.enter_scope("act_compute_rms_bwd_wait_shmem")
    if NUM_REDUCTION_CTAS:
        tlx.barrier_expect_bytes(barrier, expected_bytes)
        tlx.barrier_wait(barrier, phase=phase)
    if ENABLE_PROTON and idx == PROTON_TILE:
        pl.exit_scope("act_compute_rms_bwd_wait_shmem")

    dy_y_total = dy_y_local  # start with local CTA's partial (register)
    for i in tl.static_range(NUM_REDUCTION_CTAS - 1):
        dy_y_total += tlx.local_load(reduction_buf[buf_offset + i])

    dy_y_mean = fast_dividef(
        dy_y_total,
        tl.full(dy_y_total.shape, N, dtype=tl.float32),
    )

    if HAS_RMS_NORM_WEIGHT:
        w = rms_norm_weight[None, :].to(tl.float32)
        # do_gdpa = rrms_vals * (dy * w - (y / w) * dy_y_mean)
        # f32x2 for all [M,D] elementwise ops; broadcast via standard mul first
        dy_w = dy * w  # broadcast [1,D] → [M,D]
        y_over_w = fast_dividef(y, w)  # broadcast [1,D] → [M,D]
        y_over_w_mean = _mul_f32x2(y_over_w, dy_y_mean)
        do_gdpa = _mul_f32x2(rrms_vals, _sub_f32x2(dy_w, y_over_w_mean))
    else:
        # do_gdpa = rrms_vals * (dy - y * dy_y_mean)
        y_mean = _mul_f32x2(y, dy_y_mean)
        do_gdpa = _mul_f32x2(rrms_vals, _sub_f32x2(dy, y_mean))

    return do_gdpa


@triton.jit  # pragma: no cover
def _compute_layer_norm_backward(
    dy,
    x_hat,
    rstd_vals,
    cluster_cta_rank,
    reduction_buf,
    barrier,
    phase,
    buf_offset,
    BLOCK_SIZE_M: tl.constexpr,
    N: tl.constexpr,
    NUM_REDUCTION_CTAS: tl.constexpr,
    expected_bytes: tl.constexpr,
    ln_weight,
    HAS_LN_WEIGHT: tl.constexpr,
):
    """Compute LayerNorm backward via DSMEM cross-CTA reduction.

    dx = rstd * (dy*w - (c1 + x_hat * c2) / N)
    where c1 = sum(dy*w, axis=1), c2 = sum(dy*w * x_hat, axis=1)
    """
    if HAS_LN_WEIGHT:
        # [1,D] broadcast — use standard mul (f32x2 doesn't support [1,D] broadcast)
        dy_w = dy * ln_weight[None, :].to(tl.float32)
    else:
        dy_w = dy

    c1_local = tl.sum(dy_w, axis=1, keep_dims=True).to(tl.float32)
    c2_local = tl.sum(_mul_f32x2(dy_w, x_hat), axis=1, keep_dims=True).to(tl.float32)

    # DSMEM exchange: send c1 and c2 on the same barrier (doubled expected_bytes)
    HALF: tl.constexpr = NUM_REDUCTION_CTAS
    for i in tl.static_range(NUM_REDUCTION_CTAS):
        if cluster_cta_rank != i:
            tlx.async_remote_shmem_store(
                dst=reduction_buf[buf_offset + cluster_cta_rank],
                src=c1_local,
                remote_cta_rank=i,
                barrier=barrier,
            )
            tlx.async_remote_shmem_store(
                dst=reduction_buf[buf_offset + HALF + cluster_cta_rank],
                src=c2_local,
                remote_cta_rank=i,
                barrier=barrier,
            )
        else:
            tlx.local_store(reduction_buf[buf_offset + cluster_cta_rank], c1_local)
            tlx.local_store(
                reduction_buf[buf_offset + HALF + cluster_cta_rank], c2_local
            )

    if NUM_REDUCTION_CTAS:
        tlx.barrier_expect_bytes(barrier, expected_bytes)
        tlx.barrier_wait(barrier, phase=phase)

    c1 = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    c2 = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    for i in tl.static_range(NUM_REDUCTION_CTAS):
        c1 += tlx.local_load(reduction_buf[buf_offset + i])
        c2 += tlx.local_load(reduction_buf[buf_offset + HALF + i])

    # dx = rstd * (dy*w - (c1 + x_hat * c2) / N)
    correction = _add_f32x2(c1, _mul_f32x2(x_hat, c2))
    correction = correction * (1.0 / N)
    dx = _mul_f32x2(rstd_vals, _sub_f32x2(dy_w, correction))
    return dx


# Block sizes: 128 x 128
# Barriers:
#   producer_acquire uses the same barrier as consumer_release
#   producer_commit uses the same barriers as consumer_wait
# Channels:
#   If consumer of the channel, will have two barriers consumer_x and consumer_release_x
#   If producer of the channel, will have two barriers producer_x and producer_commit_x
#   q0, q1, k, v: consumers of the channels
#   qk0, qk1: producers
#   p0, p1: sharing tmem spaces, and barriers with qk0, qk1 (consumers)
#   o0, o1


@lru_cache
def compute_valid_tiles(
    cpu_query_offsets_list: tuple,
    bs: int,
    BLOCK_M: int,
):
    cpu_query_offsets = torch.tensor(cpu_query_offsets_list, dtype=torch.int32)
    assert cpu_query_offsets.is_cpu
    lens = cpu_query_offsets[1:] - cpu_query_offsets[:-1]
    tiles_per_example = (lens + BLOCK_M - 1) // BLOCK_M  # cdiv
    Bs = torch.repeat_interleave(torch.arange(0, bs), tiles_per_example, dim=0)

    offsets = (
        torch.arange(tiles_per_example.sum())
        - torch.repeat_interleave(
            tiles_per_example.cumsum(0) - tiles_per_example, tiles_per_example
        )
    )  # [0, 1, 2, ..., num_tiles_for_example_1 - 1, 0, 1, 2, ..., num_tiles_for_example_2 - 1, 0, 1 ...]
    offsets *= BLOCK_M  # [0, BLOCK_M, 2 * BLOCK_M, ..., (num_tiles_for_example_1 - 1) * BLOCK_M, ...]
    Ms = torch.repeat_interleave(
        cpu_query_offsets[:-1], tiles_per_example, dim=0
    )  # [seq_start_0] * num of tiles for example 0 + [seq_start_1] * num of tiles for example 1 + ...
    Ms += offsets  # [seq_start_0, seq_start_0 + BLOCK_M, seq_start_0 + 2 * BLOCK_M, ..., seq_start_0 + (num_tiles_for_example_0 - 1) * BLOCK_M, seq_start_1, ...]
    Ms = torch.cat([Ms, cpu_query_offsets[-1:]])
    return Ms, Bs
    # return (t.pin_memory().to("cuda", non_blocking=True) for t in (Ms, Bs))


@triton.jit  # pragma: no cover
def gdpa_kernel_tma_ws_blackwell(
    ensemble_activation_list: "VAR_ARGS_ARRAY",
    Q,
    Q_offsets,
    K,
    K_offsets,
    V,
    Out,  #
    Out_offsets,
    ad_to_request_offset_ptr,
    seq_index,
    stride_qm,
    stride_qh,
    stride_qk,  #
    stride_kn,
    stride_kh,
    stride_kk,  #
    stride_vn,
    stride_vh,
    stride_vk,  #
    stride_om,
    stride_oh,
    stride_ok,  #
    Z,
    H,  # number of q heads.
    G,  # number of q head in each group. number of k v head will be H//G
    N_CTX,
    N_CTX_KV,  #
    total_len_q,
    total_len_kv,  #
    qk_scale,  #
    is_predict: tl.constexpr,  #
    Q_SHAPE_0,
    FUSED_QKV: tl.constexpr,  #
    FUSED_KV: tl.constexpr,  #
    SORT_BY_SEQ_LENGTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    BLOCK_D: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    USE_START_END_OFFSETS: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    BROADCAST_Q: tl.constexpr,
    IS_DENSE_KV: tl.constexpr,
    IS_ALIGNED_KV: tl.constexpr,
    activation_enum_int: tl.constexpr,
    USE_ON_DEVICE_TMA: tl.constexpr,
    NUM_INNER_ITER: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_BUFFERS_QK: tl.constexpr,
    NUM_BUFFERS_O: tl.constexpr,
    SUBTILING: tl.constexpr,
    PINGPONG: tl.constexpr,
    NUM_REGS_ACT: tl.constexpr,
    NUM_REGS_MMA: tl.constexpr,
    NUM_REGS_LOAD: tl.constexpr,
    NUM_SUBSLICES: tl.constexpr,
    NUM_ACT_SPLITS: tl.constexpr,
    MERGE_EPI: tl.constexpr,
    ENABLE_PROTON: tl.constexpr,
    PROTON_TILE: tl.constexpr,
    ENABLE_LOAD_BALANCING: tl.constexpr,
    valid_tiles_b,
    valid_tiles_m,
    NUM_VALID_TILES_PER_HEAD,
    USE_OUTER_SWP: tl.constexpr,  # Use outer loop software pipelining
    USE_CLC: tl.constexpr,
    USE_I64_IDX: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    FUSED_RMS_NORM: tl.constexpr,
    NUM_REDUCTION_CTAS: tl.constexpr,
    rms_norm_weight,
    HAS_RMS_NORM_WEIGHT: tl.constexpr,
    rrms_out,  # [total_q, 1] reciprocal RMS for backward
    FUSED_RESIDUAL_ADD: tl.constexpr,  # add arbitrary residual to output AFTER norm
    Residual,  # TMA descriptor for the residual tensor (used when FUSED_RESIDUAL_ADD)
    FUSED_Q_RESIDUAL_ADD: tl.constexpr,  # optimized: reuse Q from q_buf as residual
    RmsNormOut=None,  # raw pointer for storing pre-residual RMSNorm output
    STORE_RMS_NORM_OUT: tl.constexpr = False,  # store y = RMSNorm(x) before residual add
):
    n_tile_num = tl.cdiv(N_CTX, BLOCK_M)
    prog_id = tl.program_id(0)
    if USE_I64_IDX:
        prog_id = prog_id.to(tl.int64)
    num_progs = tl.num_programs(0)

    if ENABLE_LOAD_BALANCING:
        total_tiles = NUM_VALID_TILES_PER_HEAD * H
    else:
        total_tiles = n_tile_num * Z * H

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1

    tile_idx = prog_id
    if not USE_ON_DEVICE_TMA:
        q_desc = Q
        k_desc = K
        v_desc = V
        # XXX: leaving output as on-device TMA due to numerics issues
        # o_desc = Out

    # start with on-device TMA where descriptors for k, v are set up outside of the persistent
    # loop and descriptor for q is set up inside the persistent loop.
    if USE_ON_DEVICE_TMA:
        q_desc = tl.make_tensor_descriptor(
            Q,
            shape=[total_len_q, HEAD_DIM * H],
            strides=[HEAD_DIM * H, 1],
            block_shape=[BLOCK_M // 2, BLOCK_D],
        )
        k_desc = tl.make_tensor_descriptor(
            K,
            shape=[total_len_kv, HEAD_DIM * H // G],
            strides=[HEAD_DIM * H // G, 1],
            block_shape=[BLOCK_N, BLOCK_D],
        )
        v_desc = tl.make_tensor_descriptor(
            V,
            shape=[total_len_kv, HEAD_DIM * H // G],
            strides=[HEAD_DIM * H // G, 1],
            block_shape=[BLOCK_N, BLOCK_D],
        )

    if USE_ON_DEVICE_TMA:
        dtype = V.dtype.element_ty
    else:
        dtype = tlx.dtype_of(v_desc)

    # allocate buffers for q0, q1
    q0_buf = tlx.local_alloc((BLOCK_M // 2, BLOCK_D), dtype, NUM_BUFFERS_Q)
    q1_buf = tlx.local_alloc((BLOCK_M // 2, BLOCK_D), dtype, NUM_BUFFERS_Q)

    # allocate buffers for k, v
    kv_buf = tlx.local_alloc((BLOCK_N, BLOCK_D), dtype, NUM_BUFFERS_KV)  # k
    if not MERGE_EPI:
        o0_smem = tlx.local_alloc((BLOCK_M // 2, HEAD_DIM), dtype, 1)
        o1_smem = tlx.local_alloc((BLOCK_M // 2, HEAD_DIM), dtype, 1)
        o0_smem_fulls = tlx.alloc_barriers(num_barriers=1)
        o1_smem_fulls = tlx.alloc_barriers(num_barriers=1)
        o0_smem_empties = tlx.alloc_barriers(num_barriers=1)
        o1_smem_empties = tlx.alloc_barriers(num_barriers=1)
    # Residual buffers reuse o_smem: Load warp -> Act warp, then Act stores
    # output back to the same physical buffer for Epilogue warp.
    if FUSED_RESIDUAL_ADD:
        res0_buf = tlx.local_alloc((BLOCK_M // 2, BLOCK_D), dtype, 1, reuse=o0_smem)
        res1_buf = tlx.local_alloc((BLOCK_M // 2, BLOCK_D), dtype, 1, reuse=o1_smem)
        res0_fulls = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
        res0_empties = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
        res1_fulls = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
        res1_empties = tlx.alloc_barriers(num_barriers=1, arrive_count=1)

    if STORE_RMS_NORM_OUT:
        y_smem0 = tlx.local_alloc((BLOCK_M // 2, BLOCK_D), dtype, 1, reuse=o0_smem)
        y_smem1 = tlx.local_alloc((BLOCK_M // 2, BLOCK_D), dtype, 1, reuse=o1_smem)
        y_smem0_fulls = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
        y_smem0_empties = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
        y_smem1_fulls = tlx.alloc_barriers(num_barriers=1, arrive_count=1)
        y_smem1_empties = tlx.alloc_barriers(num_barriers=1, arrive_count=1)

    # allocate tmem for outputs of 4 dots (after partitioning)
    # qk0 = q0 @ k, qk1 = q1 @ k, p0 = act (qk0), p1 = act (qk1)
    # acc0 = p0 @ v, acc1 = p1 @ v
    qk0_buf = tlx.local_alloc(
        (BLOCK_M // 2, BLOCK_N),
        tl.float32,
        1,
        tlx.storage_kind.tmem,
    )
    qk1_buf = tlx.local_alloc(
        (BLOCK_M // 2, BLOCK_N),
        tl.float32,
        1,
        tlx.storage_kind.tmem,
    )
    p0_buf = tlx.local_alloc(
        (BLOCK_M // 2, BLOCK_N // NUM_SUBSLICES),
        dtype,
        NUM_SUBSLICES,
        tlx.storage_kind.tmem,
        reuse=qk0_buf,
    )
    p1_buf = tlx.local_alloc(
        (BLOCK_M // 2, BLOCK_N // NUM_SUBSLICES),
        dtype,
        NUM_SUBSLICES,
        tlx.storage_kind.tmem,
        reuse=qk1_buf,
    )
    o0_buf = tlx.local_alloc(
        (BLOCK_M // 2, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem
    )
    o1_buf = tlx.local_alloc(
        (BLOCK_M // 2, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem
    )

    # allocate barriers
    q0_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q, arrive_count=1)
    q1_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q, arrive_count=1)
    q0_empties = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS_Q, arrive_count=2 if FUSED_Q_RESIDUAL_ADD else 1
    )
    q1_empties = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS_Q, arrive_count=2 if FUSED_Q_RESIDUAL_ADD else 1
    )
    kv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV, arrive_count=1)
    kv_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV, arrive_count=1)
    # initialize kv empty barriers for all buffers
    for _kv_i in tl.static_range(NUM_BUFFERS_KV):
        tlx.barrier_arrive(kv_empties[_kv_i], 1)

    # We only slice pv not qk. So we only need 1 barrier for qk_fulls
    # Since we use same buffer for qk and p, we need NUM_SUBSLICES barriers for p_fulls
    p0_fulls = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS_QK * NUM_SUBSLICES, arrive_count=1
    )
    qk0_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_QK, arrive_count=1)
    p1_fulls = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS_QK * NUM_SUBSLICES, arrive_count=1
    )
    qk1_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_QK, arrive_count=1)

    o0_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O, arrive_count=1)
    o0_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O, arrive_count=1)
    o1_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O, arrive_count=1)
    o1_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O, arrive_count=1)

    if FUSED_RMS_NORM:
        NUM_REDUCTION_BUFS: tl.constexpr = 2
        reduction_barriers_o0 = tlx.alloc_barriers(num_barriers=NUM_REDUCTION_BUFS)
        reduction_barriers_o1 = tlx.alloc_barriers(num_barriers=NUM_REDUCTION_BUFS)
        reduction_buf_o0 = tlx.local_alloc(
            (BLOCK_M // 2, 1),
            tl.float32,
            NUM_REDUCTION_CTAS * NUM_REDUCTION_BUFS,
        )
        reduction_buf_o1 = tlx.local_alloc(
            (BLOCK_M // 2, 1),
            tl.float32,
            NUM_REDUCTION_CTAS * NUM_REDUCTION_BUFS,
        )
        cross_cta_reduction_expected_bytes: tl.constexpr = (
            (BLOCK_M // 2) * tlx.size_of(tl.float32) * (NUM_REDUCTION_CTAS - 1)
        )

    if USE_CLC:
        if FUSED_RMS_NORM:
            clc_context = tlx.clc_create_context(5 * NUM_REDUCTION_CTAS)
        else:
            clc_context = tlx.clc_create_context(5)

    with tlx.async_tasks():
        # Act0 warp
        with tlx.async_task("default"):
            if FUSED_RMS_NORM:
                tlx.cluster_barrier()
                cluster_cta_rank = tlx.cluster_cta_rank()
                reduction_iter_o0 = 0
            accum_cnt = 0
            accum_cnt_outer = 0
            clc_phase_consumer = 0
            idx = 0
            has_more_tile = True

            while has_more_tile:
                begin_q, end_q, begin_k, end_k, qlen, klen = _compute_qlen(
                    tile_idx,
                    n_tile_num,
                    Q_offsets,
                    K_offsets,
                    seq_index,
                    SORT_BY_SEQ_LENGTH,
                    H,
                    N_CTX,
                    ENABLE_LOAD_BALANCING,
                    valid_tiles_b,
                    valid_tiles_m,
                    NUM_VALID_TILES_PER_HEAD,
                    BROADCAST_Q,
                )

                if ENABLE_LOAD_BALANCING:
                    start_m = 0
                    off_h = tile_idx % H
                else:
                    if FUSED_RMS_NORM:
                        off_h = tile_idx % H
                        pid = (tile_idx // H) % n_tile_num
                        off_z = (tile_idx // H) // n_tile_num
                    else:
                        pid = tile_idx % n_tile_num
                        off_hz = tile_idx // n_tile_num
                        off_h = off_hz % H
                        off_z = off_hz // H
                    start_m = pid
                out_offset = off_h.to(tl.int64) * stride_oh

                cur_act_enum = activation_enum_int
                for i in range(len(ensemble_activation_list)):
                    if off_h == i:
                        cur_act_enum = ensemble_activation_list[i]

                if start_m * BLOCK_M < qlen:
                    lo, hi = 0, klen
                    if WINDOW_SIZE is not None:
                        lo = max(
                            lo, ((start_m * BLOCK_M - WINDOW_SIZE) // BLOCK_N) * BLOCK_N
                        )
                        hi = min(hi, (start_m + 1) * BLOCK_M + WINDOW_SIZE)

                    # tl.device_print("default", hi)
                    for start_n in range(lo, hi, BLOCK_N):
                        start_n = tl.multiple_of(start_n, BLOCK_N)
                        # tl.device_print("default start_n", start_n)
                        bufIdx = accum_cnt % NUM_BUFFERS_QK
                        phase = (accum_cnt // NUM_BUFFERS_QK) & 1
                        # tl.device_print("default qk0_fulls", accum_cnt)
                        # tl.device_print("default qk0_fulls_phase", phase)

                        tlx.barrier_wait(qk0_fulls[bufIdx], phase)
                        if PINGPONG:
                            tlx.named_barrier_wait(9, 256)
                        SUBSLICE_SIZE_N: tl.constexpr = BLOCK_N // NUM_SUBSLICES
                        for slice_id in tl.static_range(0, NUM_SUBSLICES):
                            qk_view_i = tlx.subslice(
                                qk0_buf[bufIdx],
                                SUBSLICE_SIZE_N * slice_id,
                                SUBSLICE_SIZE_N,
                            )
                            qk_i = tlx.local_load(qk_view_i)
                            p0_i = apply_activation(
                                qk_i, dtype, cur_act_enum, NUM_ACT_SPLITS
                            )
                            p0_i = p0_i.to(dtype)
                            if not IS_ALIGNED_KV:
                                offs_n = tl.arange(0, SUBSLICE_SIZE_N)
                                p0_i = tl.where(
                                    (
                                        offs_n[None, :]
                                        < end_k
                                        - begin_k
                                        - start_n
                                        - SUBSLICE_SIZE_N * slice_id
                                    ),
                                    p0_i,
                                    0.0,
                                )
                            if WINDOW_SIZE is not None:
                                offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M // 2)
                                offs_n = (
                                    start_n
                                    + SUBSLICE_SIZE_N * slice_id
                                    + tl.arange(0, SUBSLICE_SIZE_N)
                                )
                                window_mask = (
                                    tl.abs(offs_m[:, None] - offs_n[None, :])
                                    <= WINDOW_SIZE
                                )
                                p0_i = tl.where(window_mask, p0_i, 0.0)
                            tlx.local_store(
                                p0_buf[bufIdx * NUM_SUBSLICES + slice_id], p0_i
                            )
                            # p and qk reuse tmem space, single producer commit for p via consumer_release_qk
                            tlx.barrier_arrive(
                                p0_fulls[bufIdx * NUM_SUBSLICES + slice_id], 1
                            )

                        if PINGPONG:
                            tlx.named_barrier_arrive(10, 256)

                        # wait for o0, o1 per iteration
                        bufIdx = accum_cnt % NUM_BUFFERS_O
                        phase = (accum_cnt // NUM_BUFFERS_O) & 1
                        # consumer wait of o0: producer_commit
                        # tl.device_print("default o0_fulls", accum_cnt)
                        # tl.device_print("default o0_fulls_phase", phase)
                        # there is no need to wait for o0 at each iteration
                        # tlx.barrier_wait(o0_fulls[bufIdx], phase)
                        accum_cnt += 1

                    # epilogue here, load from tmem
                    bufIdx_o_outer, phase_o_outer = _get_bufidx_phase(
                        accum_cnt_outer, NUM_BUFFERS_O
                    )
                    tlx.barrier_wait(o0_fulls[bufIdx_o_outer], phase_o_outer)
                    o0 = tlx.local_load(o0_buf[bufIdx_o_outer])
                    # release o0 here
                    # tl.device_print("default o0_empties", accum_cnt_outer)
                    tlx.barrier_arrive(o0_empties[bufIdx_o_outer], 1)
                    if FUSED_Q_RESIDUAL_ADD:
                        # Optimized: reuse Q from q_buf (already loaded by Load warp)
                        q_bufIdx_act = accum_cnt_outer % NUM_BUFFERS_Q
                        q_res = tlx.local_load(q0_buf[q_bufIdx_act])
                        # Signal our arrival on q0_empties (second of two)
                        tlx.barrier_arrive(q0_empties[q_bufIdx_act])
                        o0 = o0 + q_res.to(tl.float32)
                    if FUSED_RMS_NORM:
                        if HAS_RMS_NORM_WEIGHT:
                            w_offs = off_h * HEAD_DIM + tl.arange(0, HEAD_DIM)
                            w_slice = tl.load(
                                rms_norm_weight + w_offs, eviction_policy="evict_last"
                            )
                        else:
                            w_slice = None
                        reduction_buf_idx_o0 = reduction_iter_o0 % NUM_REDUCTION_BUFS
                        reduction_phase_o0 = (
                            reduction_iter_o0 // NUM_REDUCTION_BUFS
                        ) & 1
                        o0, rrms0 = _compute_rms_norm(
                            x=o0,
                            cluster_cta_rank=cluster_cta_rank,
                            reduction_buf=reduction_buf_o0,
                            barrier=reduction_barriers_o0[reduction_buf_idx_o0],
                            phase=reduction_phase_o0,
                            buf_offset=reduction_buf_idx_o0 * NUM_REDUCTION_CTAS,
                            BLOCK_SIZE_M=BLOCK_M // 2,
                            N=HEAD_DIM * H,
                            NUM_REDUCTION_CTAS=NUM_REDUCTION_CTAS,
                            expected_bytes=cross_cta_reduction_expected_bytes,
                            rms_norm_weight=w_slice,
                            HAS_RMS_NORM_WEIGHT=HAS_RMS_NORM_WEIGHT,
                        )
                        reduction_iter_o0 += 1
                        # Store rrms (only one CTA needs to write per row)
                        if cluster_cta_rank == 0:
                            rrms_offs = (begin_q + start_m * BLOCK_M) + tl.arange(
                                0, BLOCK_M // 2
                            )
                            rrms_mask = rrms_offs < end_q
                            tl.store(
                                rrms_out + rrms_offs,
                                rrms0.reshape(BLOCK_M // 2),
                                mask=rrms_mask,
                            )
                    if STORE_RMS_NORM_OUT and FUSED_RESIDUAL_ADD:
                        # Reordered: read residual FIRST, then write y, then add residual
                        res_phase = accum_cnt_outer & 1
                        tlx.barrier_wait(res0_fulls[0], res_phase)
                        res_val = tlx.local_load(res0_buf[0])
                        tlx.fence_async_shared()
                        # DON'T arrive res0_empties yet — y_smem0 store targets same buffer
                        tlx.local_store(y_smem0[0], o0.to(dtype))
                        tlx.fence_async_shared()
                        tlx.barrier_arrive(y_smem0_fulls[0])
                        # NOW release res0_buf
                        tlx.barrier_arrive(res0_empties[0])
                        o0 = o0 + res_val.to(tl.float32)
                    elif STORE_RMS_NORM_OUT:
                        tlx.local_store(y_smem0[0], o0.to(dtype))
                        tlx.fence_async_shared()
                        tlx.barrier_arrive(y_smem0_fulls[0])
                    elif FUSED_RESIDUAL_ADD:
                        res_phase = accum_cnt_outer & 1
                        tlx.barrier_wait(res0_fulls[0], res_phase)
                        res_val = tlx.local_load(res0_buf[0])
                        tlx.barrier_arrive(res0_empties[0])
                        o0 = o0 + res_val.to(tl.float32)
                    if not BROADCAST_Q:
                        begin_o = begin_q
                        end_o = end_q
                    else:
                        begin_o = qlen * off_z
                        end_o = qlen * (off_z + 1)
                    if USE_ON_DEVICE_TMA and MERGE_EPI:
                        o_desc = tl.make_tensor_descriptor(
                            Out,
                            shape=[end_o.to(tl.int32), HEAD_DIM * H],
                            strides=[HEAD_DIM * H, 1],
                            block_shape=[BLOCK_M // 2, BLOCK_D],
                        )
                    if USE_ON_DEVICE_TMA:
                        o0 = o0.to(Out.type.element_ty)
                    else:
                        o_desc = tl.make_tensor_descriptor(
                            Out,
                            shape=[end_o.to(tl.int32), HEAD_DIM * H],
                            strides=[HEAD_DIM * H, 1],
                            block_shape=[BLOCK_M // 2, BLOCK_D],
                        )
                        o0 = o0.to(tlx.dtype_of(o_desc))
                    if MERGE_EPI:
                        o_desc.store(
                            [
                                (begin_o + start_m * BLOCK_M).to(tl.int32),
                                (out_offset).to(tl.int32),
                            ],
                            o0,
                        )
                    else:
                        _, phase_o_outer = _get_bufidx_phase(accum_cnt_outer, 1)
                        if STORE_RMS_NORM_OUT:
                            # y_smem0 reuses o0_smem physical buffer. Wait for
                            # epilogue to finish TMA-storing y before overwriting
                            # with o (current iteration, no phase flip).
                            tlx.barrier_wait(y_smem0_empties[0], phase_o_outer)
                        elif not FUSED_RESIDUAL_ADD:
                            # When FUSED_RESIDUAL_ADD, Load warp already waited
                            # on o0_smem_empties before writing residual, so Epi
                            # from the previous tile is guaranteed done.
                            tlx.barrier_wait(o0_smem_empties[0], phase_o_outer ^ 1)
                        tlx.local_store(o0_smem[0], o0)
                        tlx.fence_async_shared()
                        tlx.barrier_arrive(o0_smem_fulls[0])
                    accum_cnt_outer += 1
                if USE_CLC:
                    tile_idx = tlx.clc_consumer(
                        clc_context,
                        clc_phase_consumer,
                        multi_ctas=FUSED_RMS_NORM,
                    )
                    if USE_I64_IDX:
                        tile_idx = tile_idx.to(tl.int64)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx != -1
                else:
                    tile_idx += num_progs
                    idx += 1
                    has_more_tile = idx < tiles_per_sm

        # Act1 warp
        with tlx.async_task(num_warps=4, registers=NUM_REGS_ACT):
            if FUSED_RMS_NORM:
                tlx.cluster_barrier()
                cluster_cta_rank = tlx.cluster_cta_rank()
                reduction_iter_o1 = 0
            accum_cnt = 0
            accum_cnt_outer = 0
            if PINGPONG:
                tlx.named_barrier_arrive(9, 256)
            clc_phase_consumer = 0
            idx = 0
            has_more_tile = True
            while has_more_tile:
                begin_q, end_q, begin_k, end_k, qlen, klen = _compute_qlen(
                    tile_idx,
                    n_tile_num,
                    Q_offsets,
                    K_offsets,
                    seq_index,
                    SORT_BY_SEQ_LENGTH,
                    H,
                    N_CTX,
                    ENABLE_LOAD_BALANCING,
                    valid_tiles_b,
                    valid_tiles_m,
                    NUM_VALID_TILES_PER_HEAD,
                    BROADCAST_Q,
                )

                if ENABLE_LOAD_BALANCING:
                    start_m = 0
                    off_h = tile_idx % H
                else:
                    if FUSED_RMS_NORM:
                        off_h = tile_idx % H
                        pid = (tile_idx // H) % n_tile_num
                        off_z = (tile_idx // H) // n_tile_num
                    else:
                        pid = tile_idx % n_tile_num
                        off_hz = tile_idx // n_tile_num
                        off_h = off_hz % H
                    start_m = pid
                out_offset = off_h.to(tl.int64) * stride_oh

                cur_act_enum = activation_enum_int
                for i in range(len(ensemble_activation_list)):
                    if off_h == i:
                        cur_act_enum = ensemble_activation_list[i]

                if start_m * BLOCK_M < qlen:
                    lo, hi = 0, klen
                    if WINDOW_SIZE is not None:
                        lo = max(
                            lo, ((start_m * BLOCK_M - WINDOW_SIZE) // BLOCK_N) * BLOCK_N
                        )
                        hi = min(hi, (start_m + 1) * BLOCK_M + WINDOW_SIZE)

                    for start_n in range(lo, hi, BLOCK_N):
                        start_n = tl.multiple_of(start_n, BLOCK_N)
                        ## communication channel for qk1, p1
                        bufIdx = accum_cnt % NUM_BUFFERS_QK
                        phase = (accum_cnt // NUM_BUFFERS_QK) & 1
                        # if ENABLE_PROTON and idx == PROTON_TILE:
                        #    pl.enter_scope("consumer_qk0_view")
                        tlx.barrier_wait(qk1_fulls[bufIdx], phase)
                        # if ENABLE_PROTON and idx == PROTON_TILE:
                        #    pl.exit_scope("consumer_qk0_view")

                        if PINGPONG:
                            tlx.named_barrier_wait(10, 256)
                        SUBSLICE_SIZE_N: tl.constexpr = BLOCK_N // NUM_SUBSLICES
                        for slice_id in tl.static_range(0, NUM_SUBSLICES):
                            qk_view_i = tlx.subslice(
                                qk1_buf[bufIdx],
                                SUBSLICE_SIZE_N * slice_id,
                                SUBSLICE_SIZE_N,
                            )
                            qk_i = tlx.local_load(qk_view_i)
                            p1_i = apply_activation(
                                qk_i, dtype, cur_act_enum, NUM_ACT_SPLITS
                            )
                            p1_i = p1_i.to(dtype)
                            if not IS_ALIGNED_KV:
                                offs_n = tl.arange(0, SUBSLICE_SIZE_N)
                                p1_i = tl.where(
                                    (
                                        offs_n[None, :]
                                        < end_k
                                        - begin_k
                                        - start_n
                                        - SUBSLICE_SIZE_N * slice_id
                                    ),
                                    p1_i,
                                    0.0,
                                )
                            if WINDOW_SIZE is not None:
                                offs_m = (
                                    start_m * BLOCK_M
                                    + BLOCK_M // 2
                                    + tl.arange(0, BLOCK_M // 2)
                                )
                                offs_n = (
                                    start_n
                                    + SUBSLICE_SIZE_N * slice_id
                                    + tl.arange(0, SUBSLICE_SIZE_N)
                                )
                                window_mask = (
                                    tl.abs(offs_m[:, None] - offs_n[None, :])
                                    <= WINDOW_SIZE
                                )
                                p1_i = tl.where(window_mask, p1_i, 0.0)
                            tlx.local_store(
                                p1_buf[bufIdx * NUM_SUBSLICES + slice_id], p1_i
                            )
                            # p and qk reuse tmem space, single producer commit for p via consumer_release_qk
                            tlx.barrier_arrive(
                                p1_fulls[bufIdx * NUM_SUBSLICES + slice_id], 1
                            )

                        if PINGPONG:
                            tlx.named_barrier_arrive(9, 256)

                        # wait for o0, o1 per iteration
                        bufIdx = accum_cnt % NUM_BUFFERS_O
                        phase = (accum_cnt // NUM_BUFFERS_O) & 1
                        # consumer wait of o1
                        # there is no need to wait for o1 at each iteration
                        # tlx.barrier_wait(o1_fulls[bufIdx], phase)
                        accum_cnt += 1
                    # epilogue here, load from tmem
                    bufIdx_o_outer, phase_o_outer = _get_bufidx_phase(
                        accum_cnt_outer, NUM_BUFFERS_O
                    )
                    if USE_ON_DEVICE_TMA and MERGE_EPI:
                        o_desc = tl.make_tensor_descriptor(
                            Out,
                            shape=[end_q.to(tl.int32), HEAD_DIM * H],
                            strides=[HEAD_DIM * H, 1],
                            block_shape=[BLOCK_M // 2, BLOCK_D],
                        )
                    tlx.barrier_wait(o1_fulls[bufIdx_o_outer], phase_o_outer)
                    o1 = tlx.local_load(o1_buf[bufIdx_o_outer])
                    # release o1 here
                    tlx.barrier_arrive(o1_empties[bufIdx_o_outer], 1)
                    if FUSED_Q_RESIDUAL_ADD:
                        # Optimized: reuse Q from q_buf (already loaded by Load warp)
                        q_bufIdx_act = accum_cnt_outer % NUM_BUFFERS_Q
                        q_res = tlx.local_load(q1_buf[q_bufIdx_act])
                        # Signal our arrival on q1_empties (second of two)
                        tlx.barrier_arrive(q1_empties[q_bufIdx_act])
                        o1 = o1 + q_res.to(tl.float32)
                    if FUSED_RMS_NORM:
                        if HAS_RMS_NORM_WEIGHT:
                            w_offs = off_h * HEAD_DIM + tl.arange(0, HEAD_DIM)
                            w_slice = tl.load(
                                rms_norm_weight + w_offs, eviction_policy="evict_last"
                            )
                        else:
                            w_slice = None
                        reduction_buf_idx_o1 = reduction_iter_o1 % NUM_REDUCTION_BUFS
                        reduction_phase_o1 = (
                            reduction_iter_o1 // NUM_REDUCTION_BUFS
                        ) & 1
                        o1, rrms1 = _compute_rms_norm(
                            x=o1,
                            cluster_cta_rank=cluster_cta_rank,
                            reduction_buf=reduction_buf_o1,
                            barrier=reduction_barriers_o1[reduction_buf_idx_o1],
                            phase=reduction_phase_o1,
                            buf_offset=reduction_buf_idx_o1 * NUM_REDUCTION_CTAS,
                            BLOCK_SIZE_M=BLOCK_M // 2,
                            N=HEAD_DIM * H,
                            NUM_REDUCTION_CTAS=NUM_REDUCTION_CTAS,
                            expected_bytes=cross_cta_reduction_expected_bytes,
                            rms_norm_weight=w_slice,
                            HAS_RMS_NORM_WEIGHT=HAS_RMS_NORM_WEIGHT,
                        )
                        reduction_iter_o1 += 1
                        # Store rrms (only one CTA needs to write per row)
                        if cluster_cta_rank == 0:
                            rrms_offs = (
                                begin_q + start_m * BLOCK_M + BLOCK_M // 2
                            ) + tl.arange(0, BLOCK_M // 2)
                            rrms_mask = rrms_offs < end_q
                            tl.store(
                                rrms_out + rrms_offs,
                                rrms1.reshape(BLOCK_M // 2),
                                mask=rrms_mask,
                            )
                    if STORE_RMS_NORM_OUT and FUSED_RESIDUAL_ADD:
                        # Reordered: read residual FIRST, then write y, then add residual
                        res_phase = accum_cnt_outer & 1
                        tlx.barrier_wait(res1_fulls[0], res_phase)
                        res_val = tlx.local_load(res1_buf[0])
                        tlx.fence_async_shared()
                        # DON'T arrive res1_empties yet — y_smem1 store targets same buffer
                        tlx.local_store(y_smem1[0], o1.to(dtype))
                        tlx.fence_async_shared()
                        tlx.barrier_arrive(y_smem1_fulls[0])
                        # NOW release res1_buf
                        tlx.barrier_arrive(res1_empties[0])
                        o1 = o1 + res_val.to(tl.float32)
                    elif STORE_RMS_NORM_OUT:
                        tlx.local_store(y_smem1[0], o1.to(dtype))
                        tlx.fence_async_shared()
                        tlx.barrier_arrive(y_smem1_fulls[0])
                    elif FUSED_RESIDUAL_ADD:
                        res_phase = accum_cnt_outer & 1
                        tlx.barrier_wait(res1_fulls[0], res_phase)
                        res_val = tlx.local_load(res1_buf[0])
                        tlx.barrier_arrive(res1_empties[0])
                        o1 = o1 + res_val.to(tl.float32)
                    if MERGE_EPI:
                        if not BROADCAST_Q:
                            begin_o = begin_q
                            end_o = end_q
                        else:
                            begin_o = qlen * off_z
                            end_o = qlen * (off_z + 1)
                        if USE_ON_DEVICE_TMA:
                            o_desc = tl.make_tensor_descriptor(
                                Out,
                                shape=[end_o.to(tl.int32), HEAD_DIM * H],
                                strides=[HEAD_DIM * H, 1],
                                block_shape=[BLOCK_M // 2, BLOCK_D],
                            )
                    if USE_ON_DEVICE_TMA:
                        o1 = o1.to(Out.type.element_ty)
                    else:
                        o_desc = tl.make_tensor_descriptor(
                            Out,
                            shape=[end_q.to(tl.int32), HEAD_DIM * H],
                            strides=[HEAD_DIM * H, 1],
                            block_shape=[BLOCK_M // 2, BLOCK_D],
                        )
                        o1 = o1.to(tlx.dtype_of(o_desc))
                    if MERGE_EPI:
                        o_desc.store(
                            [
                                (begin_q + start_m * BLOCK_M + BLOCK_M // 2).to(
                                    tl.int32
                                ),
                                (out_offset).to(tl.int32),
                            ],
                            o1,
                        )
                    else:
                        _, phase_o_outer = _get_bufidx_phase(accum_cnt_outer, 1)
                        if STORE_RMS_NORM_OUT:
                            # y_smem1 reuses o1_smem physical buffer. Wait for
                            # epilogue to finish TMA-storing y before overwriting
                            # with o (current iteration, no phase flip).
                            tlx.barrier_wait(y_smem1_empties[0], phase_o_outer)
                        elif not FUSED_RESIDUAL_ADD:
                            # When FUSED_RESIDUAL_ADD, Load warp already waited
                            # on o1_smem_empties before writing residual, so Epi
                            # from the previous tile is guaranteed done.
                            tlx.barrier_wait(o1_smem_empties[0], phase_o_outer ^ 1)
                        tlx.local_store(o1_smem[0], o1)
                        tlx.fence_async_shared()
                        tlx.barrier_arrive(o1_smem_fulls[0])
                    accum_cnt_outer += 1
                if USE_CLC:
                    tile_idx = tlx.clc_consumer(
                        clc_context,
                        clc_phase_consumer,
                        multi_ctas=FUSED_RMS_NORM,
                    )
                    if USE_I64_IDX:
                        tile_idx = tile_idx.to(tl.int64)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx != -1
                else:
                    tile_idx += num_progs
                    idx += 1
                    has_more_tile = idx < tiles_per_sm

        # MMA warp
        with tlx.async_task(num_warps=1, registers=NUM_REGS_MMA):
            if FUSED_RMS_NORM:
                tlx.cluster_barrier()
            accum_cnt_q = 0
            accum_cnt_kv = 0
            accum_cnt_o = 0
            accum_cnt_qk = 0
            accum_cnt_outer = 0

            if USE_OUTER_SWP:
                clc_phase_consumer = 0
                # Outer loop software pipelining implementation
                # Uses flattened loop with pipeline state tracking
                # 1st q0k, q1k, p0v. Pipeline p1v and epi in loop
                begin_q, end_q, begin_k, end_k, qlen, klen = _compute_qlen(
                    tile_idx,
                    n_tile_num,
                    Q_offsets,
                    K_offsets,
                    seq_index,
                    SORT_BY_SEQ_LENGTH,
                    H,
                    N_CTX,
                    ENABLE_LOAD_BALANCING,
                    valid_tiles_b,
                    valid_tiles_m,
                    NUM_VALID_TILES_PER_HEAD,
                    BROADCAST_Q,
                )
                if ENABLE_LOAD_BALANCING:
                    start_m = 0
                else:
                    if FUSED_RMS_NORM:
                        pid = (tile_idx // H) % n_tile_num
                    else:
                        pid = tile_idx % n_tile_num
                    start_m = pid

                # if the first tile is invalid, retry until we get a valid one
                # FIXME this assumes each SM will encounter at least one valid tile in its lifetime
                idx = 1
                has_more_tile = True
                while has_more_tile and start_m * BLOCK_M >= qlen:
                    if USE_CLC:
                        tile_idx = tlx.clc_consumer(
                            clc_context,
                            clc_phase_consumer,
                            multi_ctas=FUSED_RMS_NORM,
                        )
                        clc_phase_consumer ^= 1
                        has_more_tile = tile_idx != -1
                    else:
                        tile_idx += num_progs
                    idx += NUM_INNER_ITER

                    begin_q, end_q, begin_k, end_k, qlen, klen = _compute_qlen(
                        tile_idx,
                        n_tile_num,
                        Q_offsets,
                        K_offsets,
                        seq_index,
                        SORT_BY_SEQ_LENGTH,
                        H,
                        N_CTX,
                        ENABLE_LOAD_BALANCING,
                        valid_tiles_b,
                        valid_tiles_m,
                        NUM_VALID_TILES_PER_HEAD,
                        BROADCAST_Q,
                    )

                    if ENABLE_LOAD_BALANCING:
                        start_m = 0
                    else:
                        if FUSED_RMS_NORM:
                            pid = (tile_idx // H) % n_tile_num
                        else:
                            pid = tile_idx % n_tile_num
                        start_m = pid

                # prologue
                bufIdx_q, phase_q = _get_bufidx_phase(accum_cnt_q, NUM_BUFFERS_Q)
                bufIdx_k, phase_k = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                bufIdx_qk, phase_qk = _get_bufidx_phase(accum_cnt_qk, NUM_BUFFERS_QK)

                lo, hi = 0, klen
                if WINDOW_SIZE is not None:
                    lo = max(
                        lo, ((start_m * BLOCK_M - WINDOW_SIZE) // BLOCK_N) * BLOCK_N
                    )
                    hi = min(hi, (start_m + 1) * BLOCK_M + WINDOW_SIZE)

                tlx.barrier_wait(q0_fulls[bufIdx_q], phase_q)  # consumer wait for q0
                tlx.barrier_wait(kv_fulls[bufIdx_k], phase_k)  # consumer wait for k
                tlx.async_dot(
                    q0_buf[bufIdx_q],
                    tlx.local_trans(kv_buf[bufIdx_k]),
                    qk0_buf[bufIdx_qk],
                    use_acc=False,
                    mBarriers=[qk0_fulls[bufIdx_qk]],
                )
                if NUM_INNER_ITER == 1:  # done with first tile
                    tlx.tcgen05_commit(q0_empties[bufIdx_q])

                tlx.barrier_wait(q1_fulls[bufIdx_q], phase_q)  # consumer wait for q1
                tlx.async_dot(
                    q1_buf[bufIdx_q],
                    tlx.local_trans(kv_buf[bufIdx_k]),
                    qk1_buf[bufIdx_qk],
                    use_acc=False,
                    mBarriers=[kv_empties[bufIdx_k], qk1_fulls[bufIdx_qk]],
                )
                if NUM_INNER_ITER == 1:  # done with first tile
                    tlx.tcgen05_commit(q1_empties[bufIdx_q])

                bufIdx_v, phase_v = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
                tlx.barrier_wait(kv_fulls[bufIdx_v], phase_v)  # consumer wait for v
                bufIdx_o_outer, phase_o_outer = _get_bufidx_phase(
                    accum_cnt_outer, NUM_BUFFERS_O
                )
                tlx.barrier_wait(
                    o0_empties[bufIdx_o_outer], phase_o_outer ^ 1
                )  # producer acquire for o0
                bufIdx_p, phase_p = _get_bufidx_phase(accum_cnt_qk, NUM_BUFFERS_QK)
                bufIdx_o, phase_o = _get_bufidx_phase(accum_cnt_o, NUM_BUFFERS_O)
                for slice_id in tl.static_range(0, NUM_SUBSLICES):
                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.enter_scope("dot_wait_p0")
                    tlx.barrier_wait(
                        p0_fulls[bufIdx_p * NUM_SUBSLICES + slice_id], phase_p
                    )  # consumer wait for p0
                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.exit_scope("dot_wait_p0")
                    v_slice = tlx.local_slice(
                        kv_buf[bufIdx_v],
                        [BLOCK_N * slice_id // NUM_SUBSLICES, 0],
                        [BLOCK_N // NUM_SUBSLICES, HEAD_DIM],
                    )
                    tlx.async_dot(  # p0 . v -> o0
                        p0_buf[bufIdx_p * NUM_SUBSLICES + slice_id],
                        v_slice,
                        o0_buf[bufIdx_o],
                        use_acc=slice_id > 0,
                    )
                if NUM_INNER_ITER == 1:  # done with first tile
                    tlx.tcgen05_commit(o0_fulls[bufIdx_o])

                iter = 0  # counter for 2nd stage P1V dot
                first = True  # flag for 0th iter of 2nd stage
                iter_next = 1 % NUM_INNER_ITER  # counter for 1st stage dot
                accum_cnt_qk_prev = accum_cnt_qk
                accum_cnt_o_prev = accum_cnt_o
                accum_cnt_kv += 2
                accum_cnt_qk += 1
                accum_cnt_o += 1
                accum_cnt_outer_prev = accum_cnt_outer
                # handle NUM_INNER_ITER ==1 case
                if iter_next == 0:
                    first_next = True
                    accum_cnt_q += 1
                    accum_cnt_outer += 1
                else:
                    first_next = False  # flag for 0th iter of 1st stage

                while has_more_tile:
                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.enter_scope("dot_tile")
                    if iter_next == 0:
                        if USE_CLC:
                            tile_idx = tlx.clc_consumer(
                                clc_context,
                                clc_phase_consumer,
                                multi_ctas=FUSED_RMS_NORM,
                            )
                            if USE_I64_IDX:
                                tile_idx = tile_idx.to(tl.int64)
                            clc_phase_consumer = clc_phase_consumer ^ 1
                            has_more_tile = tile_idx != -1
                        else:
                            tile_idx = tile_idx + num_progs
                        begin_q, end_q, begin_k, end_k, qlen, klen = _compute_qlen(
                            tile_idx,
                            n_tile_num,
                            Q_offsets,
                            K_offsets,
                            seq_index,
                            SORT_BY_SEQ_LENGTH,
                            H,
                            N_CTX,
                            ENABLE_LOAD_BALANCING,
                            valid_tiles_b,
                            valid_tiles_m,
                            NUM_VALID_TILES_PER_HEAD,
                            BROADCAST_Q,
                        )
                        if ENABLE_LOAD_BALANCING:
                            start_m = 0
                        else:
                            if FUSED_RMS_NORM:
                                pid = (tile_idx // H) % n_tile_num
                            else:
                                pid = tile_idx % n_tile_num
                            start_m = pid

                    if tile_idx != -1 and start_m * BLOCK_M < qlen:
                        if iter == 0:
                            first = True
                        else:
                            first = False
                        if iter_next == 0:
                            first_next = True
                        else:
                            first_next = False

                        bufIdx_q, phase_q = _get_bufidx_phase(
                            accum_cnt_q, NUM_BUFFERS_Q
                        )
                        bufIdx_k, phase_k = _get_bufidx_phase(
                            accum_cnt_kv, NUM_BUFFERS_KV
                        )
                        bufIdx_qk, phase_qk = _get_bufidx_phase(
                            accum_cnt_qk, NUM_BUFFERS_QK
                        )
                        # q0 dot k
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope(
                                "dot_wait_k",
                            )
                        tlx.barrier_wait(q0_fulls[bufIdx_q], phase_q, first_next)
                        tlx.barrier_wait(kv_fulls[bufIdx_k], phase_k)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("dot_wait_k")
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("dot_async_dot_q0k")
                        tlx.async_dot(
                            q0_buf[bufIdx_q],
                            tlx.local_trans(kv_buf[bufIdx_k]),
                            qk0_buf[bufIdx_qk],
                            use_acc=False,
                            mBarriers=[qk0_fulls[bufIdx_qk]],
                        )
                        if iter_next == NUM_INNER_ITER - 1:
                            tlx.tcgen05_commit(q0_empties[bufIdx_q])
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("dot_async_dot_q0k")

                        # p1 dot v for previous iteration
                        bufIdx_qk1, phase_qk1 = _get_bufidx_phase(
                            accum_cnt_qk_prev, NUM_BUFFERS_QK
                        )
                        bufIdx_o_outer_prev, phase_o_outer_prev = _get_bufidx_phase(
                            accum_cnt_outer_prev, NUM_BUFFERS_O
                        )
                        tlx.barrier_wait(
                            o1_empties[bufIdx_o_outer_prev],
                            phase_o_outer_prev ^ 1,
                            first,
                        )
                        bufIdx_o1, phase_o1 = _get_bufidx_phase(
                            accum_cnt_o_prev,
                            NUM_BUFFERS_O,
                        )
                        bufIdx_v, phase_v = _get_bufidx_phase(
                            accum_cnt_kv - 1, NUM_BUFFERS_KV
                        )

                        for slice_id in tl.static_range(0, NUM_SUBSLICES):
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("dot_wait_p1")
                            tlx.barrier_wait(
                                p1_fulls[bufIdx_qk1 * NUM_SUBSLICES + slice_id],
                                phase_qk1,
                            )
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("dot_wait_p1")
                            v_slice = tlx.local_slice(
                                kv_buf[bufIdx_v],
                                [BLOCK_N * slice_id // NUM_SUBSLICES, 0],
                                [BLOCK_N // NUM_SUBSLICES, HEAD_DIM],
                            )
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("dot_async_dot_p1v")
                            tlx.async_dot(
                                p1_buf[bufIdx_qk1 * NUM_SUBSLICES + slice_id],
                                v_slice,
                                o1_buf[bufIdx_o1],
                                use_acc=not (first and slice_id == 0),
                                mBarriers=[kv_empties[bufIdx_v]]
                                if slice_id == NUM_SUBSLICES - 1
                                else [],
                            )
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("dot_async_dot_p1v")
                        if iter == NUM_INNER_ITER - 1:
                            tlx.tcgen05_commit(o1_fulls[bufIdx_o_outer])

                        # q1 dot k
                        tlx.barrier_wait(q1_fulls[bufIdx_q], phase_q, first_next)
                        bufIdx_qk1_next, phase_qk1_next = _get_bufidx_phase(
                            accum_cnt_qk, NUM_BUFFERS_QK
                        )
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("dot_async_dot_q1k")
                        tlx.async_dot(
                            q1_buf[bufIdx_q],
                            tlx.local_trans(kv_buf[bufIdx_k]),
                            qk1_buf[bufIdx_qk1_next],
                            use_acc=False,
                            mBarriers=[
                                kv_empties[bufIdx_k],
                                qk1_fulls[bufIdx_qk1_next],
                            ],
                        )
                        if iter_next == NUM_INNER_ITER - 1:
                            tlx.tcgen05_commit(q1_empties[bufIdx_q])
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("dot_async_dot_q1k")

                        # next iteration p0 dot v
                        bufIdx_v, phase_v = _get_bufidx_phase(
                            accum_cnt_kv + 1, NUM_BUFFERS_KV
                        )
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("dot_wait_v")
                        tlx.barrier_wait(kv_fulls[bufIdx_v], phase_v)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("dot_wait_v")

                        bufIdx_o_outer, phase_o_outer = _get_bufidx_phase(
                            accum_cnt_outer, NUM_BUFFERS_O
                        )
                        tlx.barrier_wait(
                            o0_empties[bufIdx_o_outer], phase_o_outer ^ 1, first_next
                        )

                        bufIdx_qk1_next, phase_qk1_next = _get_bufidx_phase(
                            accum_cnt_qk, NUM_BUFFERS_QK
                        )
                        bufIdx_o_next, phase_o_next = _get_bufidx_phase(
                            accum_cnt_o, NUM_BUFFERS_O
                        )
                        for slice_id in tl.static_range(0, NUM_SUBSLICES):
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("dot_wait_p0")
                            tlx.barrier_wait(
                                p0_fulls[bufIdx_qk1_next * NUM_SUBSLICES + slice_id],
                                phase_qk,
                            )
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("dot_wait_p0")
                            v_slice = tlx.local_slice(
                                kv_buf[bufIdx_v],
                                [BLOCK_N * slice_id // NUM_SUBSLICES, 0],
                                [BLOCK_N // NUM_SUBSLICES, HEAD_DIM],
                            )
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("dot_async_dot_p0v")
                            tlx.async_dot(
                                p0_buf[bufIdx_qk * NUM_SUBSLICES + slice_id],
                                v_slice,
                                o0_buf[bufIdx_o_next],
                                use_acc=not (first_next and slice_id == 0),
                            )
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("dot_async_dot_p0v")

                        accum_cnt_kv += 2
                        accum_cnt_qk += 1
                        accum_cnt_qk_prev += 1
                        accum_cnt_o += 1
                        accum_cnt_o_prev += 1
                        accum_cnt_outer_prev = accum_cnt_outer

                        if iter_next == NUM_INNER_ITER - 1:
                            tlx.tcgen05_commit(o0_fulls[bufIdx_o])
                            accum_cnt_q += 1
                            accum_cnt_outer += 1

                        iter = (iter + 1) % NUM_INNER_ITER
                        iter_next = (iter_next + 1) % NUM_INNER_ITER
                        first = False
                        first_next = False
                        idx += 1
                    else:
                        # skip the entire "inner loop" for this Q tile
                        idx += NUM_INNER_ITER

                    if not USE_CLC:
                        has_more_tile = idx < tiles_per_sm * NUM_INNER_ITER

                # Epilogue for outer SWP
                # Always wait when inner loop iter is 1
                if NUM_INNER_ITER == 1:
                    first_next = True
                bufIdx_o_outer_prev, phase_o_outer_prev = _get_bufidx_phase(
                    accum_cnt_outer_prev, NUM_BUFFERS_O
                )

                tlx.barrier_wait(
                    o1_empties[bufIdx_o_outer_prev], phase_o_outer_prev ^ 1, first_next
                )
                bufIdx_qk1, phase_qk1 = _get_bufidx_phase(
                    accum_cnt_qk_prev, NUM_BUFFERS_QK
                )
                bufIdx_o, phase_o = _get_bufidx_phase(accum_cnt_o_prev, NUM_BUFFERS_O)
                bufIdx_v, phase_v = _get_bufidx_phase(accum_cnt_kv - 1, NUM_BUFFERS_KV)
                for slice_id in tl.static_range(0, NUM_SUBSLICES):
                    tlx.barrier_wait(
                        p1_fulls[bufIdx_qk1 * NUM_SUBSLICES + slice_id], phase_qk1
                    )
                    m_barriers = (
                        [o1_fulls[bufIdx_o], kv_empties[bufIdx_v]]
                        if slice_id == NUM_SUBSLICES - 1
                        else []
                    )
                    v_slice = tlx.local_slice(
                        kv_buf[bufIdx_v],
                        [BLOCK_N * slice_id // NUM_SUBSLICES, 0],
                        [BLOCK_N // NUM_SUBSLICES, HEAD_DIM],
                    )
                    tlx.async_dot(
                        p1_buf[bufIdx_qk1 * NUM_SUBSLICES + slice_id],
                        v_slice,
                        o1_buf[bufIdx_o],
                        use_acc=not (first_next and slice_id == 0),
                        mBarriers=m_barriers,
                    )
                if ENABLE_PROTON and idx == PROTON_TILE:
                    pl.exit_scope("dot_tile")
            else:
                # Standard nested loop implementation
                clc_phase_consumer = 0
                idx = 0
                has_more_tile = True
                while has_more_tile:
                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.enter_scope("dot_tile")
                    begin_q, end_q, begin_k, end_k, qlen, klen = _compute_qlen(
                        tile_idx,
                        n_tile_num,
                        Q_offsets,
                        K_offsets,
                        seq_index,
                        SORT_BY_SEQ_LENGTH,
                        H,
                        N_CTX,
                        ENABLE_LOAD_BALANCING,
                        valid_tiles_b,
                        valid_tiles_m,
                        NUM_VALID_TILES_PER_HEAD,
                        BROADCAST_Q,
                    )

                    if ENABLE_LOAD_BALANCING:
                        start_m = 0
                    else:
                        if FUSED_RMS_NORM:
                            pid = (tile_idx // H) % n_tile_num
                        else:
                            pid = tile_idx % n_tile_num
                        start_m = pid

                    if start_m * BLOCK_M < qlen:
                        # prologue
                        bufIdx_q, phase_q = _get_bufidx_phase(
                            accum_cnt_q, NUM_BUFFERS_Q
                        )
                        bufIdx_k, phase_k = _get_bufidx_phase(
                            accum_cnt_kv, NUM_BUFFERS_KV
                        )
                        bufIdx_qk, phase_qk = _get_bufidx_phase(
                            accum_cnt_qk, NUM_BUFFERS_QK
                        )
                        accum_cnt_qk1 = accum_cnt_qk

                        lo, hi = 0, klen
                        if WINDOW_SIZE is not None:
                            lo = max(
                                lo,
                                ((start_m * BLOCK_M - WINDOW_SIZE) // BLOCK_N)
                                * BLOCK_N,
                            )
                            hi = min(hi, (start_m + 1) * BLOCK_M + WINDOW_SIZE)

                        # tl.device_print("gemm q0_fulls_prologue", accum_cnt_q)
                        # tl.device_print("gemm q0_fulls_phase", phase_q)
                        tlx.barrier_wait(
                            q0_fulls[bufIdx_q], phase_q
                        )  # consumer wait for q0
                        # tl.device_print("gemm consumer_k", accum_cnt_kv)
                        # tl.device_print("gemm consumer_k_buf", bufIdx_k)
                        # tl.device_print("gemm consumer_k_phase", phase_k)
                        tlx.barrier_wait(
                            kv_fulls[bufIdx_k], phase_k
                        )  # consumer wait for k
                        # Do we need the initial acquire here?
                        # dot partition has producer commit for qk0, activation partition consumer wait for qk0
                        # activation partition producer commit for p0, dot partition has consumer wait for p0
                        # tlx.barrier_wait(p0_fulls_view, phase_qk)  # producer acquire for qk0
                        # producer commit for qk0
                        tlx.async_dot(
                            q0_buf[bufIdx_q],
                            tlx.local_trans(kv_buf[bufIdx_k]),
                            qk0_buf[bufIdx_qk],
                            use_acc=False,
                            mBarriers=[qk0_fulls[bufIdx_qk]],
                        )
                        # accum_cnt_qk += 1

                        # tl.device_print("gemm q1_fulls", accum_cnt_q)
                        # tl.device_print("gemm q1_fulls_phase", phase_q)
                        tlx.barrier_wait(
                            q1_fulls[bufIdx_q], phase_q
                        )  # consumer wait for q1
                        # tlx.barrier_wait(p1_fulls[bufIdx_qk], phase_qk)  # producer acquire for qk1
                        # consumer release for k, producer commit for qk1
                        tlx.async_dot(
                            q1_buf[bufIdx_q],
                            tlx.local_trans(kv_buf[bufIdx_k]),
                            qk1_buf[bufIdx_qk],
                            use_acc=False,
                            mBarriers=[kv_empties[bufIdx_k], qk1_fulls[bufIdx_qk]],
                        )
                        # tl.device_print("gemm consumer_release_k", accum_cnt_kv)
                        # tl.device_print("gemm consumer_release_k_buf", bufIdx_k)
                        # accum_cnt_qk1 += 1

                        bufIdx_v, phase_v = _get_bufidx_phase(
                            accum_cnt_kv + 1, NUM_BUFFERS_KV
                        )
                        # tl.device_print("gemm consumer_v", accum_cnt_kv + 1)
                        # tl.device_print("gemm consumer_v_buf", bufIdx_v)
                        # tl.device_print("gemm consumer_v_phase", phase_v)
                        tlx.barrier_wait(
                            kv_fulls[bufIdx_v], phase_v
                        )  # consumer wait for v
                        # need to acquire o0 to make sure epilogue is done, this is needed for each outer loop
                        bufIdx_o_outer, phase_o_outer = _get_bufidx_phase(
                            accum_cnt_outer, NUM_BUFFERS_O
                        )
                        # tl.device_print("gemm o0_empties", accum_cnt_outer)
                        # tl.device_print("gemm o0_empties_phase", phase_o_outer)
                        # DEBUG_PERF
                        tlx.barrier_wait(
                            o0_empties[bufIdx_o_outer], phase_o_outer ^ 1
                        )  # producer acquire for o0
                        # For reuse of qk0 and p0, we can simplify the barriers
                        #   activation partition: consumer wait for qk0, ... update p, producer commit of p0
                        #   dot partition: producer commit of qk0, ..., consumer wait for p0 (use the same barrier as p0_fulls)
                        bufIdx_p, phase_p = _get_bufidx_phase(
                            accum_cnt_qk, NUM_BUFFERS_QK
                        )
                        bufIdx_o, phase_o = _get_bufidx_phase(
                            accum_cnt_o, NUM_BUFFERS_O
                        )
                        for slice_id in tl.static_range(0, NUM_SUBSLICES):
                            # tl.device_print("gemm p0_fulls", accum_cnt_qk)
                            # tl.device_print("gemm p0_fulls_phase", phase_p)
                            # DEBUG_PERF_P
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("dot_wait_p0")
                            tlx.barrier_wait(
                                p0_fulls[bufIdx_p * NUM_SUBSLICES + slice_id], phase_p
                            )  # consumer wait for p0 due to reuse of p0 and qk0
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("dot_wait_p0")
                            # reinterpret qk0 as p0
                            v_slice = tlx.local_slice(
                                kv_buf[bufIdx_v],
                                [BLOCK_N * slice_id // NUM_SUBSLICES, 0],
                                [BLOCK_N // NUM_SUBSLICES, HEAD_DIM],
                            )

                            tlx.async_dot(  # p0 . v -> o0
                                p0_buf[bufIdx_p * NUM_SUBSLICES + slice_id],
                                v_slice,
                                o0_buf[bufIdx_o],
                                use_acc=slice_id > 0,
                            )
                        accum_cnt_o1 = accum_cnt_o

                        first = True
                        # mma_iters = (hi - lo) // BLOCK_N
                        accum_cnt_kv += 2
                        accum_cnt_qk += 1
                        accum_cnt_o += 1
                        # tl.device_print("gemm for ", hi)
                        # tl.device_print("gemm mma_iters ", mma_iters)

                        for _it in range(lo + BLOCK_N, hi, BLOCK_N):
                            # for _it in range(mma_iters - 1):
                            # tl.device_print("gemm iter ", _it)
                            bufIdx_k, phase_k = _get_bufidx_phase(
                                accum_cnt_kv, NUM_BUFFERS_KV
                            )
                            bufIdx_qk, phase_qk = _get_bufidx_phase(
                                accum_cnt_qk, NUM_BUFFERS_QK
                            )

                            # q0 dot k
                            # tl.device_print("gemm consumer_k", accum_cnt_kv)
                            # tl.device_print("gemm consumer_k_buf", bufIdx_k)
                            # tl.device_print("gemm consumer_k_phase", phase_k)
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("dot_wait_k")
                            tlx.barrier_wait(
                                kv_fulls[bufIdx_k], phase_k
                            )  # consumer wait for k
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("dot_wait_k")
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("dot_async_dot_q0k")
                            tlx.async_dot(
                                q0_buf[bufIdx_q],
                                tlx.local_trans(kv_buf[bufIdx_k]),
                                qk0_buf[bufIdx_qk],
                                use_acc=False,
                                mBarriers=[qk0_fulls[bufIdx_qk]],
                            )
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("dot_async_dot_q0k")

                            # p1 dot v for previous iteration
                            bufIdx_qk1, phase_qk1 = _get_bufidx_phase(
                                accum_cnt_qk1, NUM_BUFFERS_QK
                            )

                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("dot_wait_o1")
                            tlx.barrier_wait(
                                o1_empties[bufIdx_o_outer], phase_o_outer ^ 1, first
                            )  # producer acquire for o1, only needed for first iteration
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("dot_wait_o1")
                            bufIdx_o1, phase_o1 = _get_bufidx_phase(
                                accum_cnt_o1,
                                NUM_BUFFERS_O,  # previous iteration
                            )
                            bufIdx_v, phase_v = _get_bufidx_phase(
                                accum_cnt_kv - 1, NUM_BUFFERS_KV
                            )
                            # release v for previous iteartion, accum_cnt_kv already advanced

                            for slice_id in tl.static_range(0, NUM_SUBSLICES):
                                # tl.device_print("gemm o1_empties", accum_cnt_outer)
                                # tl.device_print("gemm o1_empties_phase", phase_o_outer)
                                # DEBUG_PERF

                                # tl.device_print("gemm p1_fulls", accum_cnt_qk1)
                                # tl.device_print("gemm p1_fulls_phase", phase_qk1)
                                # DEBUG_PERF_P
                                if ENABLE_PROTON and idx == PROTON_TILE:
                                    pl.enter_scope("dot_wait_p1")
                                tlx.barrier_wait(
                                    p1_fulls[bufIdx_qk1 * NUM_SUBSLICES + slice_id],
                                    phase_qk1,
                                )  # consumer wait for p1 use p1_fulls due to reuse
                                if ENABLE_PROTON and idx == PROTON_TILE:
                                    pl.exit_scope("dot_wait_p1")
                                # done using v from previous iteration

                                # reinterpret as p1
                                v_slice = tlx.local_slice(
                                    kv_buf[bufIdx_v],
                                    [BLOCK_N * slice_id // NUM_SUBSLICES, 0],
                                    [BLOCK_N // NUM_SUBSLICES, HEAD_DIM],
                                )
                                if ENABLE_PROTON and idx == PROTON_TILE:
                                    pl.enter_scope("dot_async_dot_p1v")
                                tlx.async_dot(  # p1 . v from previous iteration
                                    p1_buf[bufIdx_qk1 * NUM_SUBSLICES + slice_id],
                                    v_slice,
                                    o1_buf[bufIdx_o1],
                                    use_acc=not (first and slice_id == 0),
                                    mBarriers=[
                                        kv_empties[bufIdx_v],
                                    ]
                                    if slice_id == NUM_SUBSLICES - 1
                                    else [],
                                )
                                if ENABLE_PROTON and idx == PROTON_TILE:
                                    pl.exit_scope("dot_async_dot_p1v")
                                # tl.device_print("gemm consumer_release_v", accum_cnt_kv - 1)
                                # tl.device_print("gemm consumer_release_v_buf", bufIdx_v)

                            # q1 dot k, done using k for this iteration
                            bufIdx_qk1_next, phase_qk1_next = _get_bufidx_phase(
                                accum_cnt_qk1 + 1, NUM_BUFFERS_QK
                            )
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("dot_async_dot_q1k")
                            tlx.async_dot(
                                q1_buf[bufIdx_q],
                                tlx.local_trans(kv_buf[bufIdx_k]),
                                qk1_buf[bufIdx_qk1_next],
                                use_acc=False,
                                mBarriers=[
                                    kv_empties[bufIdx_k],
                                    qk1_fulls[bufIdx_qk1_next],
                                ],
                            )
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("dot_async_dot_q1k")
                            # tl.device_print("gemm consumer_release_k", accum_cnt_kv)
                            # tl.device_print("gemm consumer_release_k_buf", bufIdx_k)

                            # p0 dot v
                            bufIdx_v, phase_v = _get_bufidx_phase(
                                accum_cnt_kv + 1, NUM_BUFFERS_KV
                            )
                            # tl.device_print("gemm consumer_v", accum_cnt_kv + 1)
                            # tl.device_print("gemm consumer_v_buf", bufIdx_v)
                            # tl.device_print("gemm consumer_v_phase", phase_v)
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("dot_wait_v")
                            tlx.barrier_wait(
                                kv_fulls[bufIdx_v], phase_v
                            )  # consumer wait for v
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("dot_wait_v")
                            # no need to acquire o0 as this is the only partition updating it
                            # tlx.barrier_wait(o0_empties)  # producer acquire for o0
                            bufIdx_o, phase_o = _get_bufidx_phase(
                                accum_cnt_o, NUM_BUFFERS_O
                            )
                            for slice_id in tl.static_range(0, NUM_SUBSLICES):
                                # tl.device_print("gemm p0_fulls", accum_cnt_qk)
                                # tl.device_print("gemm p0_fulls_phase", phase_qk)
                                # DEBUG_PERF_P

                                if ENABLE_PROTON and idx == PROTON_TILE:
                                    pl.enter_scope("dot_wait_p0")
                                tlx.barrier_wait(
                                    p0_fulls[bufIdx_qk * NUM_SUBSLICES + slice_id],
                                    phase_qk,
                                )  # consumer wait for p0 use p0_fulls due to reuse
                                if ENABLE_PROTON and idx == PROTON_TILE:
                                    pl.exit_scope("dot_wait_p0")
                                v_slice = tlx.local_slice(
                                    kv_buf[bufIdx_v],
                                    [BLOCK_N * slice_id // NUM_SUBSLICES, 0],
                                    [BLOCK_N // NUM_SUBSLICES, HEAD_DIM],
                                )
                                if ENABLE_PROTON and idx == PROTON_TILE:
                                    pl.enter_scope("dot_async_dot_p0v")
                                tlx.async_dot(
                                    p0_buf[bufIdx_qk * NUM_SUBSLICES + slice_id],
                                    v_slice,
                                    o0_buf[bufIdx_o],
                                    use_acc=True,
                                )
                                if ENABLE_PROTON and idx == PROTON_TILE:
                                    pl.exit_scope("dot_async_dot_p0v")

                            first = False
                            accum_cnt_kv += 2
                            accum_cnt_qk += 1
                            accum_cnt_qk1 += 1
                            accum_cnt_o += 1
                            accum_cnt_o1 += 1

                        # epilogue
                        # commit to release q0, q1
                        tlx.tcgen05_commit(q0_empties[bufIdx_q])
                        tlx.tcgen05_commit(q1_empties[bufIdx_q])
                        tlx.tcgen05_commit(o0_fulls[bufIdx_o])
                        # tl.device_print("gemm o1_empties_epilogue", accum_cnt_outer)
                        # tl.device_print("gemm o1_empties_phase", phase_o_outer)
                        # DEBUG_PERF
                        tlx.barrier_wait(
                            o1_empties[bufIdx_o_outer], phase_o_outer ^ 1, first
                        )  # producer acquire for o1 at the first iteration
                        bufIdx_qk1, phase_qk1 = _get_bufidx_phase(
                            accum_cnt_qk1, NUM_BUFFERS_QK
                        )
                        bufIdx_o, phase_o = _get_bufidx_phase(
                            accum_cnt_o1, NUM_BUFFERS_O
                        )
                        # we already advanced the counter
                        bufIdx_v, phase_v = _get_bufidx_phase(
                            accum_cnt_kv - 1, NUM_BUFFERS_KV
                        )
                        for slice_id in tl.static_range(0, NUM_SUBSLICES):
                            # tl.device_print("gemm p1_fulls_epilogue", accum_cnt_qk1)
                            # tl.device_print("gemm p1_fulls_phase", phase_qk1)
                            # DEBUG_PERF_P
                            tlx.barrier_wait(
                                p1_fulls[bufIdx_qk1 * NUM_SUBSLICES + slice_id],
                                phase_qk1,
                            )  # consumer wait for p1 due to reuse of p1 and qk1

                            # release p0, p1 via qk0_fulls, qk1 barriers
                            m_barriers = (
                                [
                                    o1_fulls[bufIdx_o],
                                    kv_empties[bufIdx_v],
                                ]
                                if slice_id == NUM_SUBSLICES - 1
                                else []
                            )
                            v_slice = tlx.local_slice(
                                kv_buf[bufIdx_v],
                                [BLOCK_N * slice_id // NUM_SUBSLICES, 0],
                                [BLOCK_N // NUM_SUBSLICES, HEAD_DIM],
                            )
                            tlx.async_dot(  # p1 . v in last iteration
                                p1_buf[bufIdx_qk1 * NUM_SUBSLICES + slice_id],
                                v_slice,
                                o1_buf[bufIdx_o],
                                use_acc=not (first and slice_id == 0),
                                mBarriers=m_barriers,
                            )
                        # tl.device_print("gemm consumer_release_v", accum_cnt_kv - 1)
                        # tl.device_print("gemm consumer_release_v_buf", bufIdx_v)
                        accum_cnt_qk1 += 1
                        accum_cnt_q += 1
                        accum_cnt_outer += 1
                        # signal producer commit of epi0 and epi1, we don't want to block the gemm partition
                        # to wait for the completion
                    if USE_CLC:
                        tile_idx = tlx.clc_consumer(
                            clc_context,
                            clc_phase_consumer,
                            multi_ctas=FUSED_RMS_NORM,
                        )
                        if USE_I64_IDX:
                            tile_idx = tile_idx.to(tl.int64)
                        clc_phase_consumer = clc_phase_consumer ^ 1
                        has_more_tile = tile_idx != -1
                    else:
                        tile_idx += num_progs
                        idx += 1
                        has_more_tile = idx < tiles_per_sm
                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.exit_scope("dot_tile")

        # load warp
        with tlx.async_task(num_warps=1, registers=NUM_REGS_LOAD):
            if FUSED_RMS_NORM:
                tlx.cluster_barrier()
            accum_count_q = 0
            accum_cnt_kv = 0
            clc_phase_consumer = 0
            idx = 0
            has_more_tile = True
            while has_more_tile:
                begin_q, end_q, begin_k, end_k, qlen, klen = _compute_qlen(
                    tile_idx,
                    n_tile_num,
                    Q_offsets,
                    K_offsets,
                    seq_index,
                    SORT_BY_SEQ_LENGTH,
                    H,
                    N_CTX,
                    ENABLE_LOAD_BALANCING,
                    valid_tiles_b,
                    valid_tiles_m,
                    NUM_VALID_TILES_PER_HEAD,
                    BROADCAST_Q,
                )

                if ENABLE_LOAD_BALANCING:
                    start_m = 0
                    off_h = tile_idx % H
                else:
                    if FUSED_RMS_NORM:
                        off_h = tile_idx % H
                        pid = (tile_idx // H) % n_tile_num
                        off_z = (tile_idx // H) // n_tile_num
                        if SORT_BY_SEQ_LENGTH:
                            off_z = tl.load(seq_index + off_z)
                    else:
                        pid = tile_idx % n_tile_num
                        off_hz = tile_idx // n_tile_num
                        off_z = off_hz // H
                        if SORT_BY_SEQ_LENGTH:
                            off_z = tl.load(seq_index + off_z)
                        off_h = off_hz % H
                    start_m = pid
                off_h_kv = off_h // G
                q_offset = off_h.to(tl.int64) * stride_qh
                kv_offset = off_h_kv.to(tl.int64) * stride_kh

                if start_m * BLOCK_M < qlen:
                    # begin_o = tl.load(Out_offsets + off_z) # confirm if tma store should use begin_q
                    lo, hi = 0, klen
                    if WINDOW_SIZE is not None:
                        lo = max(
                            lo, ((start_m * BLOCK_M - WINDOW_SIZE) // BLOCK_N) * BLOCK_N
                        )
                        hi = min(hi, (start_m + 1) * BLOCK_M + WINDOW_SIZE)
                    # calculate bufIdx and phase from accum_count_q
                    q_bufIdx = accum_count_q % NUM_BUFFERS_Q
                    q_phase = (accum_count_q // NUM_BUFFERS_Q) & 1
                    # producer acquire: q0_empties
                    tlx.barrier_wait(q0_empties[q_bufIdx], q_phase ^ 1)
                    # barrier for producer commit
                    tlx.barrier_expect_bytes(
                        q0_fulls[q_bufIdx], BLOCK_M // 2 * BLOCK_D * 2
                    )  # num_bytes)
                    tlx.async_descriptor_load(
                        q_desc,
                        q0_buf[q_bufIdx],
                        [
                            (begin_q + start_m * BLOCK_M).to(tl.int32),
                            (q_offset).to(tl.int32),
                        ],
                        q0_fulls[q_bufIdx],
                    )

                    k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                    # producer acquire
                    tlx.barrier_wait(kv_empties[k_bufIdx], k_phase)  # ^ 1)
                    # barrier for producer commit
                    tlx.barrier_expect_bytes(
                        kv_fulls[k_bufIdx], BLOCK_N * BLOCK_D * 2
                    )  # num_bytes)
                    start_n = lo
                    tlx.async_descriptor_load(
                        k_desc,
                        kv_buf[k_bufIdx],
                        [
                            (begin_k + start_n).to(tl.int32),
                            (kv_offset).to(tl.int32),
                        ],
                        kv_fulls[k_bufIdx],
                    )

                    # producer acquire
                    tlx.barrier_wait(q1_empties[q_bufIdx], q_phase ^ 1)
                    # barrier for producer commit
                    tlx.barrier_expect_bytes(
                        q1_fulls[q_bufIdx], BLOCK_M // 2 * BLOCK_D * 2
                    )  # num_bytes)
                    tlx.async_descriptor_load(
                        q_desc,
                        q1_buf[q_bufIdx],
                        [
                            (begin_q + start_m * BLOCK_M + BLOCK_M // 2).to(tl.int32),
                            (q_offset).to(tl.int32),
                        ],
                        q1_fulls[q_bufIdx],
                    )

                    v_bufIdx, v_phase = _get_bufidx_phase(
                        accum_cnt_kv + 1, NUM_BUFFERS_KV
                    )
                    tlx.barrier_wait(kv_empties[v_bufIdx], v_phase)  # ^ 1)
                    # barrier for producer commit
                    tlx.barrier_expect_bytes(kv_fulls[v_bufIdx], BLOCK_N * BLOCK_D * 2)
                    tlx.async_descriptor_load(
                        v_desc,
                        kv_buf[v_bufIdx],
                        [
                            (begin_k + start_n).to(tl.int32),
                            (kv_offset).to(tl.int32),
                        ],
                        kv_fulls[v_bufIdx],
                    )
                    accum_cnt_kv += 2

                    for start_n in range(lo + BLOCK_N, hi, BLOCK_N):
                        start_n = tl.multiple_of(start_n, BLOCK_N)
                        k_bufIdx, k_phase = _get_bufidx_phase(
                            accum_cnt_kv, NUM_BUFFERS_KV
                        )
                        # producer acquire
                        # tl.device_print("load consumer_release_k", accum_cnt_kv)
                        # tl.device_print("load consumer_release_k_buf", k_bufIdx)
                        # tl.device_print("load consumer_release_k_phase", k_phase)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("load_wait_k_empty")
                        tlx.barrier_wait(kv_empties[k_bufIdx], k_phase)  # ^ 1)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("load_wait_k_empty")
                        # barrier for producer commit
                        tlx.barrier_expect_bytes(
                            kv_fulls[k_bufIdx], BLOCK_N * BLOCK_D * 2
                        )  # num_bytes)
                        tlx.async_descriptor_load(
                            k_desc,
                            kv_buf[k_bufIdx],
                            [
                                (begin_k + start_n).to(tl.int32),
                                (kv_offset).to(tl.int32),
                            ],
                            kv_fulls[k_bufIdx],
                        )
                        # tl.device_print("load accum_cnt_kv", accum_cnt_kv)
                        # tl.device_print("load consumer_k_buf", k_bufIdx)
                        # k_view = tlx.local_trans(k_view)

                        # producer acquire
                        v_bufIdx, v_phase = _get_bufidx_phase(
                            accum_cnt_kv + 1, NUM_BUFFERS_KV
                        )
                        # tl.device_print("load accum_cnt_kv", accum_cnt_kv + 1)
                        # tl.device_print("load consumer_release_v_buf", v_bufIdx)
                        # tl.device_print("load consumer_release_v_phase", v_phase)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("load_wait_v_empty")
                        tlx.barrier_wait(kv_empties[v_bufIdx], v_phase)  # ^ 1)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("load_wait_v_empty")
                        # barrier for producer commit
                        tlx.barrier_expect_bytes(
                            kv_fulls[v_bufIdx], BLOCK_N * BLOCK_D * 2
                        )
                        tlx.async_descriptor_load(
                            v_desc,
                            kv_buf[v_bufIdx],
                            [
                                (begin_k + start_n).to(tl.int32),
                                (kv_offset).to(tl.int32),
                            ],
                            kv_fulls[v_bufIdx],
                        )
                        # tl.device_print("load consumer_v_buf", v_bufIdx)
                        accum_cnt_kv += 2
                    # outside of inner for
                    accum_count_q += 1
                    # Load residual into q_res buffers (reuse o_smem physical storage)
                    # Only for generic FUSED_RESIDUAL_ADD (not Q-reuse optimization)
                    if FUSED_RESIDUAL_ADD:
                        res_phase = (accum_count_q - 1) & 1
                        # Load residual0 into res0_buf (reuses o0_smem).
                        # Must wait for both: Act0 consumed previous residual,
                        # AND Epi consumed previous output (same physical buffer).
                        tlx.barrier_wait(res0_empties[0], res_phase ^ 1)
                        tlx.barrier_wait(o0_smem_empties[0], res_phase ^ 1)
                        tlx.barrier_expect_bytes(
                            res0_fulls[0], BLOCK_M // 2 * BLOCK_D * 2
                        )
                        tlx.async_descriptor_load(
                            Residual,
                            res0_buf[0],
                            [
                                (begin_q + start_m * BLOCK_M).to(tl.int32),
                                (q_offset).to(tl.int32),
                            ],
                            res0_fulls[0],
                        )
                        # Load residual1 into res1_buf (reuses o1_smem, separate buffer)
                        tlx.barrier_wait(res1_empties[0], res_phase ^ 1)
                        tlx.barrier_wait(o1_smem_empties[0], res_phase ^ 1)
                        tlx.barrier_expect_bytes(
                            res1_fulls[0], BLOCK_M // 2 * BLOCK_D * 2
                        )
                        tlx.async_descriptor_load(
                            Residual,
                            res1_buf[0],
                            [
                                (begin_q + start_m * BLOCK_M + BLOCK_M // 2).to(
                                    tl.int32
                                ),
                                (q_offset).to(tl.int32),
                            ],
                            res1_fulls[0],
                        )
                if USE_CLC:
                    tile_idx = tlx.clc_consumer(
                        clc_context,
                        clc_phase_consumer,
                        multi_ctas=FUSED_RMS_NORM,
                    )
                    if USE_I64_IDX:
                        tile_idx = tile_idx.to(tl.int64)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx != -1
                else:
                    tile_idx += num_progs
                    idx += 1
                    has_more_tile = idx < tiles_per_sm
        with tlx.async_task(num_warps=1, registers=24):  # epilogue
            if FUSED_RMS_NORM:
                tlx.cluster_barrier()
            # Can we guard this with not MERGE_EPI?
            accum_cnt_outer = 0
            clc_phase_consumer = 0
            clc_phase_producer = 1
            idx = 0
            has_more_tile = True
            while has_more_tile:
                if USE_CLC:
                    tlx.clc_producer(
                        clc_context,
                        clc_phase_producer,
                        multi_ctas=FUSED_RMS_NORM,
                    )
                    clc_phase_producer = clc_phase_producer ^ 1

                begin_q, end_q, begin_k, end_k, qlen, klen = _compute_qlen(
                    tile_idx,
                    n_tile_num,
                    Q_offsets,
                    K_offsets,
                    seq_index,
                    SORT_BY_SEQ_LENGTH,
                    H,
                    N_CTX,
                    ENABLE_LOAD_BALANCING,
                    valid_tiles_b,
                    valid_tiles_m,
                    NUM_VALID_TILES_PER_HEAD,
                    BROADCAST_Q,
                )

                if ENABLE_LOAD_BALANCING:
                    start_m = 0
                    off_h = tile_idx % H
                else:
                    if FUSED_RMS_NORM:
                        off_h = tile_idx % H
                        pid = (tile_idx // H) % n_tile_num
                        off_z = (tile_idx // H) // n_tile_num
                    else:
                        pid = tile_idx % n_tile_num
                        off_hz = tile_idx // n_tile_num
                        off_h = off_hz % H
                        off_z = off_hz // H
                    start_m = pid
                out_offset = off_h.to(tl.int64) * stride_oh
                if not BROADCAST_Q:
                    begin_o = begin_q
                    end_o = end_q
                else:
                    begin_o = qlen * off_z
                    end_o = qlen * (off_z + 1)

                if not MERGE_EPI and start_m * BLOCK_M < qlen:
                    out_offset = off_h.to(tl.int64) * stride_oh
                    # keeping output as device TMA even for host TMA enabled
                    o_desc = tl.make_tensor_descriptor(
                        Out,
                        shape=[end_o.to(tl.int32), HEAD_DIM * H],
                        strides=[HEAD_DIM * H, 1],
                        block_shape=[BLOCK_M // 2, BLOCK_D],
                    )
                    _, phase_o_outer = _get_bufidx_phase(accum_cnt_outer, 1)

                    if STORE_RMS_NORM_OUT:
                        # TMA store pre-residual y0 and y1 to RmsNormOut
                        rms_norm_out_desc = tl.make_tensor_descriptor(
                            RmsNormOut,
                            shape=[end_o.to(tl.int32), HEAD_DIM * H],
                            strides=[HEAD_DIM * H, 1],
                            block_shape=[BLOCK_M // 2, BLOCK_D],
                        )
                        tlx.barrier_wait(y_smem0_fulls[0], phase_o_outer)
                        tlx.async_descriptor_store(
                            rms_norm_out_desc,
                            y_smem0[0],
                            [
                                (begin_o + start_m * BLOCK_M).to(tl.int32),
                                (out_offset).to(tl.int32),
                            ],
                        )
                        tlx.async_descriptor_store_wait(0)
                        tlx.barrier_arrive(y_smem0_empties[0])
                        tlx.barrier_wait(y_smem1_fulls[0], phase_o_outer)
                        tlx.async_descriptor_store(
                            rms_norm_out_desc,
                            y_smem1[0],
                            [
                                (begin_o + start_m * BLOCK_M + BLOCK_M // 2).to(
                                    tl.int32
                                ),
                                (out_offset).to(tl.int32),
                            ],
                        )
                        tlx.async_descriptor_store_wait(0)
                        tlx.barrier_arrive(y_smem1_empties[0])

                    # wait for o0
                    tlx.barrier_wait(o0_smem_fulls[0], phase_o_outer)
                    tlx.async_descriptor_store(
                        o_desc,
                        o0_smem[0],
                        [
                            (begin_o + start_m * BLOCK_M).to(tl.int32),
                            (out_offset).to(tl.int32),
                        ],
                    )
                    tlx.async_descriptor_store_wait(0)
                    tlx.barrier_arrive(o0_smem_empties[0])
                    tlx.barrier_wait(o1_smem_fulls[0], phase_o_outer)
                    tlx.async_descriptor_store(
                        o_desc,
                        o1_smem[0],
                        [
                            (begin_o + start_m * BLOCK_M + BLOCK_M // 2).to(tl.int32),
                            (out_offset).to(tl.int32),
                        ],
                    )
                    tlx.async_descriptor_store_wait(0)
                    tlx.barrier_arrive(o1_smem_empties[0])
                    accum_cnt_outer += 1
                if USE_CLC:
                    tile_idx = tlx.clc_consumer(
                        clc_context,
                        clc_phase_consumer,
                        multi_ctas=FUSED_RMS_NORM,
                    )
                    if USE_I64_IDX:
                        tile_idx = tile_idx.to(tl.int64)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx != -1
                else:
                    tile_idx += num_progs
                    idx += 1
                    has_more_tile = idx < tiles_per_sm
        # Empty warp group for cluster_barrier sync (like matmul kernel's 4th group).
        # Brings total warps to 12 (multiple of 4) which is required for cluster_barrier.
        if FUSED_RMS_NORM:
            with tlx.async_task(num_warps=1, registers=24):
                tlx.cluster_barrier()


@triton.jit  # pragma: no cover
def gdpa_kernel_tma_ws_blackwell_short_kv(
    ensemble_activation_list: "VAR_ARGS_ARRAY",
    Q,
    Q_offsets,
    K,
    K_offsets,
    V,
    Out,  #
    Out_offsets,
    ad_to_request_offset_ptr,
    seq_index,
    stride_qm,
    stride_qh,
    stride_qk,  #
    stride_kn,
    stride_kh,
    stride_kk,  #
    stride_vn,
    stride_vh,
    stride_vk,  #
    stride_om,
    stride_oh,
    stride_ok,  #
    Z,
    H,  # number of q heads.
    G,  # number of q head in each group. number of k v head will be H//G
    N_CTX,
    N_CTX_KV,  #
    total_len_q,
    total_len_kv,  #
    qk_scale,  #
    is_predict: tl.constexpr,  #
    Q_SHAPE_0,
    FUSED_QKV: tl.constexpr,  #
    FUSED_KV: tl.constexpr,  #
    SORT_BY_SEQ_LENGTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    BLOCK_D: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    USE_START_END_OFFSETS: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    BROADCAST_Q: tl.constexpr,
    IS_DENSE_KV: tl.constexpr,
    IS_ALIGNED_KV: tl.constexpr,
    activation_enum_int: tl.constexpr,
    USE_ON_DEVICE_TMA: tl.constexpr,
    NUM_INNER_ITER: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_BUFFERS_QK: tl.constexpr,
    NUM_BUFFERS_O: tl.constexpr,
    SUBTILING: tl.constexpr,
    PINGPONG: tl.constexpr,
    NUM_REGS_ACT: tl.constexpr,
    NUM_REGS_MMA: tl.constexpr,
    NUM_REGS_LOAD: tl.constexpr,
    NUM_SUBSLICES: tl.constexpr,
    NUM_ACT_SPLITS: tl.constexpr,
    MERGE_EPI: tl.constexpr,
    ENABLE_PROTON: tl.constexpr,
    PROTON_TILE: tl.constexpr,
    ENABLE_LOAD_BALANCING: tl.constexpr,
    valid_tiles_b,
    valid_tiles_m,
    NUM_VALID_TILES_PER_HEAD,
    USE_OUTER_SWP: tl.constexpr,  # Use outer loop software pipelining
    USE_CLC: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    USE_I64_IDX: tl.constexpr,
    FUSED_RMS_NORM: tl.constexpr,
    NUM_REDUCTION_CTAS: tl.constexpr,
    rms_norm_weight,
    HAS_RMS_NORM_WEIGHT: tl.constexpr,
    rrms_out,  # [total_q, 1] reciprocal RMS for backward
    FUSED_RESIDUAL_ADD: tl.constexpr,  # add arbitrary residual to output AFTER norm
    Residual,  # TMA descriptor for the residual tensor (used when FUSED_RESIDUAL_ADD)
    FUSED_Q_RESIDUAL_ADD: tl.constexpr,  # optimized: reuse Q from q_buf as residual
    RmsNormOut=None,  # raw pointer for storing pre-residual RMSNorm output
    STORE_RMS_NORM_OUT: tl.constexpr = False,  # store y = RMSNorm(x) before residual add
    FUSED_LAYERNORM: tl.constexpr = False,  # enable layernorm prologue on Q
    layernorm_weight=None,  # [H * HEAD_DIM] layernorm weight
    layernorm_bias=None,  # [H * HEAD_DIM] layernorm bias
    HAS_LAYERNORM_WEIGHT: tl.constexpr = False,
    HAS_LAYERNORM_BIAS: tl.constexpr = False,
    ln_mean_out=None,  # [total_q] float32 output for backward
    ln_rstd_out=None,  # [total_q] float32 output for backward
):
    n_tile_num = tl.cdiv(N_CTX_KV, BLOCK_N)
    prog_id = tl.program_id(0)
    if USE_I64_IDX:
        prog_id = prog_id.to(tl.int64)
    num_progs = tl.num_programs(0)

    # if ENABLE_LOAD_BALANCING:
    #     total_tiles = NUM_VALID_TILES_PER_HEAD * H
    total_tiles = n_tile_num * Z * H

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1

    tile_idx = prog_id
    if not USE_ON_DEVICE_TMA:
        q_desc = Q
        k_desc = K
        v_desc = V
        # XXX: leaving output as on-device TMA due to numerics issues
        # o_desc = Out

    # start with on-device TMA where descriptors for k, v are set up outside of the persistent
    # loop and descriptor for q is set up inside the persistent loop.
    if USE_ON_DEVICE_TMA:
        q_desc = tl.make_tensor_descriptor(
            Q,
            shape=[total_len_q, HEAD_DIM * H],
            strides=[HEAD_DIM * H, 1],
            block_shape=[BLOCK_M, BLOCK_D],
        )
        k_desc = tl.make_tensor_descriptor(
            K,
            shape=[total_len_kv, HEAD_DIM * H // G],
            strides=[HEAD_DIM * H // G, 1],
            block_shape=[BLOCK_N, BLOCK_D],
        )
        v_desc = tl.make_tensor_descriptor(
            V,
            shape=[total_len_kv, HEAD_DIM * H // G],
            strides=[HEAD_DIM * H // G, 1],
            block_shape=[BLOCK_N, BLOCK_D],
        )

    if USE_ON_DEVICE_TMA:
        dtype = V.dtype.element_ty
    else:
        dtype = tlx.dtype_of(v_desc)

    # allocate buffers for q0, q1
    q_buf = tlx.local_alloc((BLOCK_M, BLOCK_D), dtype, NUM_BUFFERS_Q)
    # q1_buf = tlx.local_alloc((BLOCK_M // 2, BLOCK_D), dtype, NUM_BUFFERS_Q)

    # allocate buffers for k, v
    kv_buf = tlx.local_alloc((BLOCK_N, BLOCK_D), dtype, NUM_BUFFERS_KV)  # k
    if not MERGE_EPI:
        o_smem = tlx.local_alloc((BLOCK_M, HEAD_DIM), dtype, NUM_BUFFERS_O)
        # o1_smem = tlx.local_alloc((BLOCK_M // 2, HEAD_DIM), dtype, 1)
        o_smem_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O)
        # o1_smem_fulls = tlx.alloc_barriers(num_barriers=1)
        o_smem_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O)
        # o1_smem_empties = tlx.alloc_barriers(num_barriers=1)

    # allocate tmem for outputs of 4 dots (after partitioning)
    # qk0 = q0 @ k, qk1 = q1 @ k, p0 = act (qk0), p1 = act (qk1)
    # acc0 = p0 @ v, acc1 = p1 @ v
    qk_buf = tlx.local_alloc(
        (BLOCK_M, BLOCK_N),
        tl.float32,
        1,
        tlx.storage_kind.tmem,
    )
    # qk1_buf = tlx.local_alloc(
    #     (BLOCK_M // 2, BLOCK_N),
    #     tl.float32,
    #     1,
    #     tlx.storage_kind.tmem,
    # )
    p_buf = tlx.local_alloc(
        (BLOCK_M, BLOCK_N // NUM_SUBSLICES),
        dtype,
        NUM_SUBSLICES,
        tlx.storage_kind.tmem,
        reuse=qk_buf,
    )
    # p1_buf = tlx.local_alloc(
    #     (BLOCK_M // 2, BLOCK_N // NUM_SUBSLICES),
    #     dtype,
    #     NUM_SUBSLICES,
    #     tlx.storage_kind.tmem,
    #     reuse=qk1_buf,
    # )
    o_buf = tlx.local_alloc(
        (BLOCK_M, HEAD_DIM), tl.float32, NUM_BUFFERS_O, tlx.storage_kind.tmem
    )
    # o1_buf = tlx.local_alloc(
    #     (BLOCK_M // 2, HEAD_DIM), tl.float32, 1, tlx.storage_kind.tmem
    # )

    # When both FUSED_LAYERNORM and FUSED_Q_RESIDUAL_ADD, the LN warp stores
    # normalized Q directly into o_buf (TMEM) so MMA can accumulate p@v onto it
    # with use_acc=True, eliminating Act0's separate Q residual add.
    LN_Q_RESIDUAL_TO_O: tl.constexpr = FUSED_LAYERNORM and FUSED_Q_RESIDUAL_ADD

    # allocate barriers
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q, arrive_count=1)
    # q1_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q, arrive_count=1)
    q_empties = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS_Q,
        arrive_count=2 if (FUSED_Q_RESIDUAL_ADD and not LN_Q_RESIDUAL_TO_O) else 1,
    )
    kv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV, arrive_count=1)
    kv_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV, arrive_count=1)

    # We only slice pv not qk. So we only need 1 barrier for qk_fulls
    # Since we use same buffer for qk and p, we need NUM_SUBSLICES barriers for p_fulls
    p_fulls = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS_QK * NUM_SUBSLICES, arrive_count=1
    )
    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_QK, arrive_count=1)
    # p1_fulls = tlx.alloc_barriers(
    #     num_barriers=NUM_BUFFERS_QK * NUM_SUBSLICES, arrive_count=1
    # )
    # qk1_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_QK, arrive_count=1)

    o_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O, arrive_count=1)
    o_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O, arrive_count=1)
    o_smem_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O, arrive_count=1)
    o_smem_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O, arrive_count=1)
    # o1_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O, arrive_count=1)
    # o1_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O, arrive_count=1)

    # Residual buffers: separate SMEM (no longer aliased with o_smem)
    if FUSED_RESIDUAL_ADD:
        res_buf = tlx.local_alloc((BLOCK_M, BLOCK_D), dtype, NUM_BUFFERS_O)
        res_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O, arrive_count=1)
        res_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O, arrive_count=1)

    if STORE_RMS_NORM_OUT:
        y_smem = tlx.local_alloc((BLOCK_M, BLOCK_D), dtype, NUM_BUFFERS_O)
        y_smem_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O, arrive_count=1)
        y_smem_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O, arrive_count=1)

    if FUSED_RMS_NORM:
        NUM_REDUCTION_BUFS: tl.constexpr = 2
        reduction_barriers = tlx.alloc_barriers(num_barriers=NUM_REDUCTION_BUFS)
        reduction_buf = tlx.local_alloc(
            (BLOCK_M, 1),
            tl.float32,
            NUM_REDUCTION_CTAS * NUM_REDUCTION_BUFS,
        )
        cross_cta_reduction_expected_bytes: tl.constexpr = (
            BLOCK_M * tlx.size_of(tl.float32) * (NUM_REDUCTION_CTAS - 1)
        )

    if FUSED_LAYERNORM:
        # LayerNorm prologue: 1 merged DSMEM reduction per M-tile (sum + sum_sq
        # on same barrier), double-buffered to overlap with next tile.
        # Each buffer has 2*NUM_REDUCTION_CTAS slots: first half for sum, second for sum_sq.
        NUM_LN_REDUCTION_BUFS: tl.constexpr = 2
        ln_reduction_barriers = tlx.alloc_barriers(num_barriers=NUM_LN_REDUCTION_BUFS)
        ln_reduction_buf = tlx.local_alloc(
            (BLOCK_M, 1),
            tl.float32,
            2 * NUM_REDUCTION_CTAS * NUM_LN_REDUCTION_BUFS,
        )
        # 2x expected_bytes: each remote CTA sends 2 stores (sum + sum_sq)
        ln_cross_cta_expected_bytes: tl.constexpr = (
            2 * BLOCK_M * tlx.size_of(tl.float32) * (NUM_REDUCTION_CTAS - 1)
        )
        # Act signals MMA that normalized Q is ready in q_buf
        q_norm_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q, arrive_count=1)

    # LN warp signals MMA that Q residual has been stored into o_buf
    if LN_Q_RESIDUAL_TO_O:
        o_q_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O, arrive_count=1)

    if USE_CLC:
        if FUSED_LAYERNORM:
            clc_context = tlx.clc_create_context(5 * NUM_REDUCTION_CTAS)
        elif FUSED_RMS_NORM:
            clc_context = tlx.clc_create_context(4 * NUM_REDUCTION_CTAS)
        else:
            clc_context = tlx.clc_create_context(4)

    with tlx.async_tasks():
        # Act0 warp
        with tlx.async_task("default"):
            if FUSED_RMS_NORM:
                tlx.cluster_barrier()
                cluster_cta_rank = tlx.cluster_cta_rank()
                reduction_iter = 0
            accum_cnt = 0
            clc_phase_consumer = 0
            idx = 0
            has_more_tile = True

            while has_more_tile:
                begin_q, end_q, begin_k, end_k, qlen, klen = _compute_qlen(
                    tile_idx,
                    n_tile_num,
                    Q_offsets,
                    K_offsets,
                    seq_index,
                    SORT_BY_SEQ_LENGTH,
                    H,
                    N_CTX,
                    ENABLE_LOAD_BALANCING,
                    valid_tiles_b,
                    valid_tiles_m,
                    NUM_VALID_TILES_PER_HEAD,
                    BROADCAST_Q,
                )

                if FUSED_RMS_NORM:
                    off_h = tile_idx % H
                    pid = (tile_idx // H) % n_tile_num
                    off_z = (tile_idx // H) // n_tile_num
                else:
                    pid = tile_idx % n_tile_num
                    off_hz = tile_idx // n_tile_num
                    off_h = off_hz % H
                    off_z = off_hz // H
                start_n = pid
                out_offset = off_h.to(tl.int64) * stride_oh

                cur_act_enum = activation_enum_int
                for i in range(len(ensemble_activation_list)):
                    if off_h == i:
                        cur_act_enum = ensemble_activation_list[i]

                if start_n * BLOCK_N < klen:
                    lo, hi = 0, qlen
                    if WINDOW_SIZE is not None:
                        lo = max(
                            lo, ((start_n * BLOCK_N - WINDOW_SIZE) // BLOCK_M) * BLOCK_M
                        )
                        hi = min(hi, (start_n + 1) * BLOCK_N + WINDOW_SIZE)

                    for start_m in range(lo, hi, BLOCK_M):
                        start_m = tl.multiple_of(start_m, BLOCK_M)

                        bufIdx = accum_cnt % NUM_BUFFERS_QK
                        phase = (accum_cnt // NUM_BUFFERS_QK) & 1

                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("act_wait_qk")
                        tlx.barrier_wait(qk_fulls[bufIdx], phase)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("act_wait_qk")
                        # if PINGPONG:
                        #     tlx.named_barrier_wait(9, 256)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("act_compute_activation")
                        SUBSLICE_SIZE_N: tl.constexpr = BLOCK_N // NUM_SUBSLICES
                        for slice_id in tl.static_range(0, NUM_SUBSLICES):
                            qk_view_i = tlx.subslice(
                                qk_buf[bufIdx],
                                SUBSLICE_SIZE_N * slice_id,
                                SUBSLICE_SIZE_N,
                            )
                            qk_i = tlx.local_load(qk_view_i)
                            p_i = apply_activation(
                                qk_i, dtype, cur_act_enum, NUM_ACT_SPLITS
                            )
                            p_i = p_i.to(dtype)
                            if not IS_ALIGNED_KV:
                                offs_n = tl.arange(0, SUBSLICE_SIZE_N)
                                p_i = tl.where(
                                    (
                                        offs_n[None, :]
                                        < end_k
                                        - begin_k
                                        - start_n * BLOCK_N
                                        - SUBSLICE_SIZE_N * slice_id
                                    ),
                                    p_i,
                                    0.0,
                                )
                            if WINDOW_SIZE is not None:
                                offs_m = start_m + tl.arange(0, BLOCK_M)
                                offs_n = (
                                    start_n * BLOCK_N
                                    + SUBSLICE_SIZE_N * slice_id
                                    + tl.arange(0, SUBSLICE_SIZE_N)
                                )
                                window_mask = (
                                    tl.abs(offs_m[:, None] - offs_n[None, :])
                                    <= WINDOW_SIZE
                                )
                                p_i = tl.where(window_mask, p_i, 0.0)
                            tlx.local_store(
                                p_buf[bufIdx * NUM_SUBSLICES + slice_id], p_i
                            )
                            # p and qk reuse tmem space, single producer commit for p via consumer_release_qk
                            tlx.fence_async_shared()
                            tlx.barrier_arrive(
                                p_fulls[bufIdx * NUM_SUBSLICES + slice_id], 1
                            )

                        # if PINGPONG:
                        #     tlx.named_barrier_arrive(10, 256)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("act_compute_activation")

                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("act_wait_o")
                        bufIdx_o, phase_o = _get_bufidx_phase(accum_cnt, NUM_BUFFERS_O)
                        tlx.barrier_wait(o_fulls[bufIdx_o], phase_o)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("act_wait_o")
                            pl.enter_scope("act_load_o")
                        o = tlx.local_load(o_buf[bufIdx_o])
                        # release o0 here
                        tlx.barrier_arrive(o_empties[bufIdx_o], 1)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("act_load_o")

                        # Load residual early and release res_buf so
                        # the LN/Load warp can pipeline the next tile's
                        # residual during Q residual add + RMS norm.
                        if FUSED_RESIDUAL_ADD:
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("act_wait_res")
                            bufIdx_res, phase_res = bufIdx_o, phase_o
                            tlx.barrier_wait(res_fulls[bufIdx_res], phase_res)
                            res_val = tlx.local_load(res_buf[bufIdx_res])
                            tlx.barrier_arrive(res_empties[bufIdx_res])
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("act_wait_res")

                        if FUSED_Q_RESIDUAL_ADD and not LN_Q_RESIDUAL_TO_O:
                            # When LN_Q_RESIDUAL_TO_O, the LN warp already stored
                            # Q into o_buf and MMA accumulated p@v onto it.
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("act_q_residual_add")
                            bufIdx_q, _ = _get_bufidx_phase(accum_cnt, NUM_BUFFERS_Q)
                            # Reuse Q from q_buf (already loaded by Load warp)
                            q_res = tlx.local_load(q_buf[bufIdx_q])
                            # Signal our arrival on q_empties (second of two)
                            tlx.barrier_arrive(q_empties[bufIdx_q])
                            o = o + q_res.to(tl.float32)
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("act_q_residual_add")
                        if FUSED_RMS_NORM:
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("act_compute_rmsnorm")
                            if HAS_RMS_NORM_WEIGHT:
                                w_offs = off_h * HEAD_DIM + tl.arange(0, HEAD_DIM)
                                w_slice = tl.load(
                                    rms_norm_weight + w_offs,
                                    eviction_policy="evict_last",
                                )
                            else:
                                w_slice = None
                            reduction_buf_idx = reduction_iter % NUM_REDUCTION_BUFS
                            reduction_phase = (reduction_iter // NUM_REDUCTION_BUFS) & 1
                            o, rrms_val = _compute_rms_norm(
                                x=o,
                                cluster_cta_rank=cluster_cta_rank,
                                reduction_buf=reduction_buf,
                                barrier=reduction_barriers[reduction_buf_idx],
                                phase=reduction_phase,
                                buf_offset=reduction_buf_idx * NUM_REDUCTION_CTAS,
                                BLOCK_SIZE_M=BLOCK_M,
                                N=HEAD_DIM * H,
                                NUM_REDUCTION_CTAS=NUM_REDUCTION_CTAS,
                                expected_bytes=cross_cta_reduction_expected_bytes,
                                rms_norm_weight=w_slice,
                                HAS_RMS_NORM_WEIGHT=HAS_RMS_NORM_WEIGHT,
                            )
                            reduction_iter += 1
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("act_compute_rmsnorm")
                                pl.enter_scope("act_store_rrms")
                            # Store rrms (only one CTA needs to write per row)
                            if cluster_cta_rank == 0:
                                rrms_offs = (begin_q + start_m) + tl.arange(0, BLOCK_M)
                                rrms_mask = rrms_offs < end_q
                                tl.store(
                                    rrms_out + rrms_offs,
                                    rrms_val.reshape(BLOCK_M),
                                    mask=rrms_mask,
                                )
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("act_store_rrms")

                        if not MERGE_EPI:
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("act_wait_epi")
                            # Wait for Epi to finish consuming previous o/y
                            # at this slot.  Epi processes y before o, so
                            # o_smem_empties guarantees y_smem is also free.
                            tlx.barrier_wait(
                                o_smem_empties[bufIdx_o],
                                phase_o ^ 1,
                            )
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("act_wait_epi")
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("act_store_o")
                        if STORE_RMS_NORM_OUT:
                            # Write pre-residual y to y_smem
                            tlx.local_store(y_smem[bufIdx_o], o.to(dtype))
                            tlx.fence_async_shared()
                            tlx.barrier_arrive(y_smem_fulls[bufIdx_o])
                        if FUSED_RESIDUAL_ADD:
                            # res_val already loaded and res_buf released above
                            o = o + res_val.to(tl.float32)

                        if not BROADCAST_Q:
                            begin_o = begin_q
                            end_o = end_q
                        else:
                            begin_o = qlen * off_z
                            end_o = qlen * (off_z + 1)
                        if USE_ON_DEVICE_TMA and MERGE_EPI:
                            o_desc = tl.make_tensor_descriptor(
                                Out,
                                shape=[end_o.to(tl.int32), HEAD_DIM * H],
                                strides=[HEAD_DIM * H, 1],
                                block_shape=[BLOCK_M, BLOCK_D],
                            )
                        if USE_ON_DEVICE_TMA:
                            o = o.to(Out.type.element_ty)
                        else:
                            o_desc = tl.make_tensor_descriptor(
                                Out,
                                shape=[end_o.to(tl.int32), HEAD_DIM * H],
                                strides=[HEAD_DIM * H, 1],
                                block_shape=[BLOCK_M, BLOCK_D],
                            )
                            o = o.to(tlx.dtype_of(o_desc))
                        if MERGE_EPI:
                            o_desc.store(
                                [
                                    (begin_o + start_m).to(tl.int32),
                                    (out_offset).to(tl.int32),
                                ],
                                o,
                            )
                        else:
                            # o_smem_empties already waited above
                            tlx.local_store(o_smem[bufIdx_o], o)
                            tlx.fence_async_shared()
                            tlx.barrier_arrive(o_smem_fulls[bufIdx_o])
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("act_store_o")
                        accum_cnt += 1

                if USE_CLC:
                    tile_idx = tlx.clc_consumer(
                        clc_context,
                        clc_phase_consumer,
                        multi_ctas=FUSED_RMS_NORM,
                    )
                    if USE_I64_IDX:
                        tile_idx = tile_idx.to(tl.int64)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx != -1
                    if ENABLE_PROTON:
                        idx += 1
                else:
                    tile_idx += num_progs
                    idx += 1
                    has_more_tile = idx < tiles_per_sm

        # # Act1 warp
        # with tlx.async_task(num_warps=4, registers=NUM_REGS_ACT):
        #     accum_cnt = 0
        #     accum_cnt_outer = 0
        #     if PINGPONG:
        #         tlx.named_barrier_arrive(9, 256)
        #     for idx in range(0, tiles_per_sm):
        #         begin_q, end_q, begin_k, end_k, qlen, klen = _compute_qlen(
        #             tile_idx,
        #             n_tile_num,
        #             Q_offsets,
        #             K_offsets,
        #             seq_index,
        #             SORT_BY_SEQ_LENGTH,
        #             H,
        #             N_CTX,
        #             ENABLE_LOAD_BALANCING,
        #             valid_tiles_b,
        #             valid_tiles_m,
        #             NUM_VALID_TILES_PER_HEAD,
        #             BROADCAST_Q,
        #         )

        #         if ENABLE_LOAD_BALANCING:
        #             start_m = 0
        #             off_h = tile_idx % H
        #         else:
        #             pid = tile_idx % n_tile_num
        #             start_m = pid
        #             off_hz = tile_idx // n_tile_num
        #             off_h = off_hz % H

        #         cur_act_enum = activation_enum_int
        #         for i in range(len(ensemble_activation_list)):
        #             if off_h == i:
        #                 cur_act_enum = ensemble_activation_list[i]

        #         if start_m * BLOCK_M < qlen:
        #             lo, hi = 0, klen
        #             if WINDOW_SIZE is not None:
        #                 lo = max(
        #                     lo, ((start_m * BLOCK_M - WINDOW_SIZE) // BLOCK_N) * BLOCK_N
        #                 )
        #                 hi = min(hi, (start_m + 1) * BLOCK_M + WINDOW_SIZE)

        #             for start_n in range(lo, hi, BLOCK_N):
        #                 start_n = tl.multiple_of(start_n, BLOCK_N)
        #                 ## communication channel for qk1, p1
        #                 bufIdx = accum_cnt % NUM_BUFFERS_QK
        #                 phase = (accum_cnt // NUM_BUFFERS_QK) & 1
        #                 # if ENABLE_PROTON and idx == PROTON_TILE:
        #                 #    pl.enter_scope("consumer_qk0_view")
        #                 tlx.barrier_wait(qk1_fulls[bufIdx], phase)
        #                 # if ENABLE_PROTON and idx == PROTON_TILE:
        #                 #    pl.exit_scope("consumer_qk0_view")

        #                 if PINGPONG:
        #                     tlx.named_barrier_wait(10, 256)
        #                 SUBSLICE_SIZE_N: tl.constexpr = BLOCK_N // NUM_SUBSLICES
        #                 for slice_id in tl.static_range(0, NUM_SUBSLICES):
        #                     qk_view_i = tlx.subslice(
        #                         qk1_buf[bufIdx], SUBSLICE_SIZE_N * slice_id, SUBSLICE_SIZE_N
        #                     )
        #                     qk_i = tlx.local_load(qk_view_i)
        #                     p1_i = apply_activation(
        #                         qk_i, dtype, cur_act_enum, NUM_ACT_SPLITS
        #                     )
        #                     p1_i = p1_i.to(dtype)
        #                     if not IS_ALIGNED_KV:
        #                         offs_n = tl.arange(0, SUBSLICE_SIZE_N)
        #                         p1_i = tl.where(
        #                             (
        #                                 offs_n[None, :]
        #                                 < end_k
        #                                 - begin_k
        #                                 - start_n
        #                                 - SUBSLICE_SIZE_N * slice_id
        #                             ),
        #                             p1_i,
        #                             0.0,
        #                         )
        #                     if WINDOW_SIZE is not None:
        #                         offs_m = (
        #                             start_m * BLOCK_M
        #                             + BLOCK_M // 2
        #                             + tl.arange(0, BLOCK_M // 2)
        #                         )
        #                         offs_n = (
        #                             start_n
        #                             + SUBSLICE_SIZE_N * slice_id
        #                             + tl.arange(0, SUBSLICE_SIZE_N)
        #                         )
        #                         window_mask = (
        #                             tl.abs(offs_m[:, None] - offs_n[None, :])
        #                             <= WINDOW_SIZE
        #                         )
        #                         p1_i = tl.where(window_mask, p1_i, 0.0)
        #                     p1_view_i = tlx.subslice(
        #                         p1_buf[bufIdx], SUBSLICE_SIZE_N * slice_id, SUBSLICE_SIZE_N
        #                     )
        #                     tlx.local_store(p1_view_i, p1_i)
        #                     tlx.fence_async_shared()
        #                     # p and qk reuse tmem space, single producer commit for p via consumer_release_qk
        #                     tlx.barrier_arrive(p1_fulls[bufIdx * NUM_SUBSLICES + slice_id], 1)

        #                 if PINGPONG:
        #                     tlx.named_barrier_arrive(9, 256)

        #                 # wait for o0, o1 per iteration
        #                 bufIdx = accum_cnt % NUM_BUFFERS_O
        #                 phase = (accum_cnt // NUM_BUFFERS_O) & 1
        #                 # consumer wait of o1
        #                 # consumer_o1_view = o1_fulls[bufIdx]
        #                 # there is no need to wait for o1 at each iteration
        #                 # tlx.barrier_wait(consumer_o1_view, phase)
        #                 accum_cnt += 1
        #             # epilogue here, load from tmem
        #             bufIdx_o_outer, phase_o_outer = _get_bufidx_phase(
        #                 accum_cnt_outer, NUM_BUFFERS_O
        #             )
        #             if USE_ON_DEVICE_TMA and MERGE_EPI:
        #                 o_desc = tl.make_tensor_descriptor(
        #                     Out,
        #                     shape=[end_q.to(tl.int32), HEAD_DIM * H],
        #                     strides=[HEAD_DIM * H, 1],
        #                     block_shape=[BLOCK_M // 2, BLOCK_D],
        #                 )
        #             tlx.barrier_wait(o1_fulls[bufIdx_o_outer], phase_o_outer)
        #             o1 = tlx.local_load(o1_buf[bufIdx_o_outer])
        #             # release o1 here
        #             tlx.barrier_arrive(o1_empties[bufIdx_o_outer], 1)
        #             if MERGE_EPI:
        #                 if not BROADCAST_Q:
        #                     begin_o = begin_q
        #                     end_o = end_q
        #                 else:
        #                     begin_o = qlen * off_z
        #                     end_o = qlen * (off_z + 1)
        #                 if USE_ON_DEVICE_TMA:
        #                     o_desc = tl.make_tensor_descriptor(
        #                         Out,
        #                         shape=[end_o.to(tl.int32), HEAD_DIM * H],
        #                         strides=[HEAD_DIM * H, 1],
        #                         block_shape=[BLOCK_M // 2, BLOCK_D],
        #                     )
        #             if USE_ON_DEVICE_TMA:
        #                 o1 = o1.to(Out.type.element_ty)
        #             else:
        #                 o_desc = tl.make_tensor_descriptor(
        #                     Out,
        #                     shape=[end_q.to(tl.int32), HEAD_DIM * H],
        #                     strides=[HEAD_DIM * H, 1],
        #                     block_shape=[BLOCK_M // 2, BLOCK_D],
        #                 )
        #                 o1 = o1.to(tlx.dtype_of(o_desc))
        #             if MERGE_EPI:
        #                 o_desc.store(
        #                     [
        #                         (begin_q + start_m * BLOCK_M + BLOCK_M // 2).to(
        #                             tl.int32
        #                         ),
        #                         (out_offset).to(tl.int32),
        #                     ],
        #                     o1,
        #                 )
        #             else:
        #                 _, phase_o_outer = _get_bufidx_phase(accum_cnt_outer, 1)
        #                 tlx.barrier_wait(o1_smem_empties[0], phase_o_outer ^ 1)
        #                 tlx.local_store(o1_smem[0], o1)
        #                 tlx.fence_async_shared()
        #                 tlx.barrier_arrive(o1_smem_fulls[0])
        #             accum_cnt_outer += 1
        #         tile_idx += num_progs

        # LayerNorm prologue warp: normalize raw Q before MMA consumes it.
        # Offloads the cross-CTA DSMEM reduction from the Act warp.
        if FUSED_LAYERNORM:
            with tlx.async_task(num_warps=4, registers=88):
                tlx.cluster_barrier()
                cluster_cta_rank_ln = tlx.cluster_cta_rank()
                ln_reduction_iter = 0
                accum_cnt_ln = 0
                clc_phase_consumer = 0
                idx = 0
                has_more_tile = True
                while has_more_tile:
                    begin_q, end_q, begin_k, end_k, qlen, klen = _compute_qlen(
                        tile_idx,
                        n_tile_num,
                        Q_offsets,
                        K_offsets,
                        seq_index,
                        SORT_BY_SEQ_LENGTH,
                        H,
                        N_CTX,
                        ENABLE_LOAD_BALANCING,
                        valid_tiles_b,
                        valid_tiles_m,
                        NUM_VALID_TILES_PER_HEAD,
                        BROADCAST_Q,
                    )

                    # FUSED_LAYERNORM requires FUSED_RMS_NORM
                    off_h = tile_idx % H
                    pid = (tile_idx // H) % n_tile_num
                    off_z = (tile_idx // H) // n_tile_num
                    if SORT_BY_SEQ_LENGTH:
                        off_z = tl.load(seq_index + off_z)
                    start_n = pid

                    if start_n * BLOCK_N < klen:
                        lo, hi = 0, qlen
                        if WINDOW_SIZE is not None:
                            lo = max(
                                lo,
                                ((start_n * BLOCK_N - WINDOW_SIZE) // BLOCK_M)
                                * BLOCK_M,
                            )
                            hi = min(hi, (start_n + 1) * BLOCK_N + WINDOW_SIZE)

                        for start_m in range(lo, hi, BLOCK_M):
                            start_m = tl.multiple_of(start_m, BLOCK_M)

                            bufIdx_q_ln, phase_q_ln = _get_bufidx_phase(
                                accum_cnt_ln, NUM_BUFFERS_Q
                            )
                            # Wait for raw Q from Load warp
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("ln_wait_q")
                            tlx.barrier_wait(q_fulls[bufIdx_q_ln], phase_q_ln)
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("ln_wait_q")
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("ln_compute")

                            # --- 2-pass subtiled LayerNorm ---
                            # Avoids holding full [M, D] in registers.
                            LN_FWD_SUBTILE: tl.constexpr = 8
                            LN_SLICE: tl.constexpr = HEAD_DIM // LN_FWD_SUBTILE
                            N_LN: tl.constexpr = HEAD_DIM * H
                            q_tile_ln = q_buf[bufIdx_q_ln]

                            # Pass 1: accumulate sum(x) and sum(x^2) across subtiles
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("ln_pass1_stats")
                            ln_sum_accum = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
                            ln_sum_sq_accum = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
                            for _ln_s in tl.static_range(LN_FWD_SUBTILE):
                                raw_sub = tlx.local_load(
                                    tlx.local_slice(
                                        q_tile_ln,
                                        [0, LN_SLICE * _ln_s],
                                        [BLOCK_M, LN_SLICE],
                                    )
                                ).to(tl.float32)
                                ln_sum_accum += tl.sum(
                                    raw_sub, axis=1, keep_dims=True
                                ).to(tl.float32)
                                ln_sum_sq_accum += tl.sum(
                                    _mul_f32x2(raw_sub, raw_sub),
                                    axis=1,
                                    keep_dims=True,
                                ).to(tl.float32)
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("ln_pass1_stats")

                            # DSMEM exchange: send sum and sum_sq on same barrier
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("ln_dsmem_store")
                            ln_buf_idx = ln_reduction_iter % NUM_LN_REDUCTION_BUFS
                            ln_phase = (ln_reduction_iter // NUM_LN_REDUCTION_BUFS) & 1
                            ln_barrier = ln_reduction_barriers[ln_buf_idx]
                            ln_buf_off = ln_buf_idx * 2 * NUM_REDUCTION_CTAS
                            LN_HALF: tl.constexpr = NUM_REDUCTION_CTAS

                            if NUM_REDUCTION_CTAS:
                                tlx.barrier_expect_bytes(
                                    ln_barrier, ln_cross_cta_expected_bytes
                                )
                            for _cta_i in tl.static_range(NUM_REDUCTION_CTAS):
                                if cluster_cta_rank_ln != _cta_i:
                                    tlx.async_remote_shmem_store(
                                        dst=ln_reduction_buf[
                                            ln_buf_off + cluster_cta_rank_ln
                                        ],
                                        src=ln_sum_accum,
                                        remote_cta_rank=_cta_i,
                                        barrier=ln_barrier,
                                    )
                                    tlx.async_remote_shmem_store(
                                        dst=ln_reduction_buf[
                                            ln_buf_off + LN_HALF + cluster_cta_rank_ln
                                        ],
                                        src=ln_sum_sq_accum,
                                        remote_cta_rank=_cta_i,
                                        barrier=ln_barrier,
                                    )
                                else:
                                    tlx.local_store(
                                        ln_reduction_buf[
                                            ln_buf_off + cluster_cta_rank_ln
                                        ],
                                        ln_sum_accum,
                                    )
                                    tlx.local_store(
                                        ln_reduction_buf[
                                            ln_buf_off + LN_HALF + cluster_cta_rank_ln
                                        ],
                                        ln_sum_sq_accum,
                                    )

                            if NUM_REDUCTION_CTAS:
                                if ENABLE_PROTON and idx == PROTON_TILE:
                                    pl.exit_scope("ln_dsmem_store")
                                    pl.enter_scope("ln_dsmem_reduce")
                                tlx.barrier_wait(ln_barrier, phase=ln_phase)

                            # Reduce across CTAs
                            total_sum = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
                            total_sum_sq = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
                            for _cta_i in tl.static_range(NUM_REDUCTION_CTAS):
                                total_sum += tlx.local_load(
                                    ln_reduction_buf[ln_buf_off + _cta_i]
                                )
                                total_sum_sq += tlx.local_load(
                                    ln_reduction_buf[ln_buf_off + LN_HALF + _cta_i]
                                )

                            ln_mean_val = total_sum / N_LN
                            ln_variance = _sub_f32x2(
                                total_sum_sq / N_LN,
                                _mul_f32x2(ln_mean_val, ln_mean_val),
                            )
                            ln_rstd_val = libdevice.rsqrt(ln_variance + 1e-5)
                            ln_reduction_iter += 1
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("ln_dsmem_reduce")

                            # Wait for res_buf before Pass 2 (which also
                            # stores raw_q to res_buf before overwriting q_buf)
                            if FUSED_RESIDUAL_ADD:
                                bufIdx_res_ln, phase_res_ln = _get_bufidx_phase(
                                    accum_cnt_ln, NUM_BUFFERS_O
                                )
                                if ENABLE_PROTON and idx == PROTON_TILE:
                                    pl.enter_scope("ln_wait_res_empty")
                                tlx.barrier_wait(
                                    res_empties[bufIdx_res_ln],
                                    phase_res_ln ^ 1,
                                )
                                if ENABLE_PROTON and idx == PROTON_TILE:
                                    pl.exit_scope("ln_wait_res_empty")

                            # Wait for o_buf before storing Q residual
                            if LN_Q_RESIDUAL_TO_O:
                                bufIdx_o_ln, phase_o_ln = _get_bufidx_phase(
                                    accum_cnt_ln, NUM_BUFFERS_O
                                )
                                if ENABLE_PROTON and idx == PROTON_TILE:
                                    pl.enter_scope("ln_wait_o_empty")
                                tlx.barrier_wait(
                                    o_empties[bufIdx_o_ln],
                                    phase_o_ln ^ 1,
                                )  # producer acquire for o_buf
                                if ENABLE_PROTON and idx == PROTON_TILE:
                                    pl.exit_scope("ln_wait_o_empty")

                            # Pass 2: re-read raw subtiles, store residual,
                            # normalize and write back to q_buf
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("ln_pass2_norm")
                            for _ln_s in tl.static_range(LN_FWD_SUBTILE):
                                q_sub_view = tlx.local_slice(
                                    q_tile_ln,
                                    [0, LN_SLICE * _ln_s],
                                    [BLOCK_M, LN_SLICE],
                                )
                                raw_sub = tlx.local_load(q_sub_view).to(tl.float32)

                                # Store raw_q subtile to res_buf before
                                # overwriting with normalized value
                                if FUSED_RESIDUAL_ADD:
                                    res_sub_view = tlx.local_slice(
                                        res_buf[bufIdx_res_ln],
                                        [0, LN_SLICE * _ln_s],
                                        [BLOCK_M, LN_SLICE],
                                    )
                                    tlx.local_store(res_sub_view, raw_sub.to(dtype))

                                # Normalize subtile
                                ln_sub = _mul_f32x2(
                                    _sub_f32x2(raw_sub, ln_mean_val),
                                    ln_rstd_val,
                                )
                                # Load weight/bias for this subtile slice
                                if HAS_LAYERNORM_WEIGHT:
                                    w_sub_offs = (
                                        off_h * HEAD_DIM
                                        + LN_SLICE * _ln_s
                                        + tl.arange(0, LN_SLICE)
                                    )
                                    w_sub = tl.load(
                                        layernorm_weight + w_sub_offs,
                                        eviction_policy="evict_last",
                                    )
                                    ln_sub = ln_sub * w_sub[None, :]
                                    if HAS_LAYERNORM_BIAS:
                                        b_sub = tl.load(
                                            layernorm_bias + w_sub_offs,
                                            eviction_policy="evict_last",
                                        )
                                        ln_sub = ln_sub + b_sub[None, :]

                                # Write normalized subtile back to q_buf
                                tlx.local_store(q_sub_view, ln_sub.to(dtype))

                                # Also store normalized Q to o_buf for Q residual
                                if LN_Q_RESIDUAL_TO_O:
                                    o_sub_view = tlx.subslice(
                                        o_buf[bufIdx_o_ln], LN_SLICE * _ln_s, LN_SLICE
                                    )
                                    tlx.local_store(o_sub_view, ln_sub.to(tl.float32))

                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("ln_pass2_norm")
                            tlx.fence_async_shared()
                            # Signal MMA that normalized Q is ready
                            tlx.barrier_arrive(q_norm_fulls[bufIdx_q_ln])
                            if FUSED_RESIDUAL_ADD:
                                tlx.barrier_arrive(res_fulls[bufIdx_res_ln])
                            # Signal MMA that Q residual is in o_buf
                            if LN_Q_RESIDUAL_TO_O:
                                tlx.barrier_arrive(o_q_fulls[bufIdx_o_ln])
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("ln_compute")

                            # Save mean/rstd for backward (only CTA rank 0)
                            if cluster_cta_rank_ln == 0:
                                ln_offs = (begin_q + start_m) + tl.arange(0, BLOCK_M)
                                ln_mask = ln_offs < end_q
                                tl.store(
                                    ln_mean_out + ln_offs,
                                    ln_mean_val.reshape(BLOCK_M),
                                    mask=ln_mask,
                                )
                                tl.store(
                                    ln_rstd_out + ln_offs,
                                    ln_rstd_val.reshape(BLOCK_M),
                                    mask=ln_mask,
                                )

                            accum_cnt_ln += 1

                    if USE_CLC:
                        tile_idx = tlx.clc_consumer(
                            clc_context,
                            clc_phase_consumer,
                            multi_ctas=FUSED_RMS_NORM,
                        )
                        if USE_I64_IDX:
                            tile_idx = tile_idx.to(tl.int64)
                        clc_phase_consumer = clc_phase_consumer ^ 1
                        has_more_tile = tile_idx != -1
                        if ENABLE_PROTON:
                            idx += 1
                    else:
                        tile_idx += num_progs
                        idx += 1
                        has_more_tile = idx < tiles_per_sm

        # MMA warp (reduced regs when LN warp present to free register budget)
        with tlx.async_task(num_warps=1, registers=NUM_REGS_MMA):
            if FUSED_RMS_NORM:
                tlx.cluster_barrier()
            accum_cnt_q = 0
            accum_cnt_kv = 0
            accum_cnt_o = 0
            accum_cnt_qk = 0

            clc_phase_consumer = 0
            idx = 0
            has_more_tile = True
            while has_more_tile:
                if ENABLE_PROTON and idx == PROTON_TILE:
                    pl.enter_scope("dot_tile")
                begin_q, end_q, begin_k, end_k, qlen, klen = _compute_qlen(
                    tile_idx,
                    n_tile_num,
                    Q_offsets,
                    K_offsets,
                    seq_index,
                    SORT_BY_SEQ_LENGTH,
                    H,
                    N_CTX,
                    ENABLE_LOAD_BALANCING,
                    valid_tiles_b,
                    valid_tiles_m,
                    NUM_VALID_TILES_PER_HEAD,
                    BROADCAST_Q,
                )

                if FUSED_RMS_NORM:
                    pid = (tile_idx // H) % n_tile_num
                else:
                    pid = tile_idx % n_tile_num
                start_n = pid

                if start_n * BLOCK_N < klen:
                    # prologue
                    bufIdx_q, phase_q = _get_bufidx_phase(accum_cnt_q, NUM_BUFFERS_Q)
                    bufIdx_k, phase_k = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                    bufIdx_qk, phase_qk = _get_bufidx_phase(
                        accum_cnt_qk, NUM_BUFFERS_QK
                    )

                    lo, hi = 0, qlen
                    if WINDOW_SIZE is not None:
                        lo = max(
                            lo, ((start_n * BLOCK_N - WINDOW_SIZE) // BLOCK_M) * BLOCK_M
                        )
                        hi = min(hi, (start_n + 1) * BLOCK_N + WINDOW_SIZE)

                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.enter_scope("mma_wait_k_q_pro")
                    tlx.barrier_wait(kv_fulls[bufIdx_k], phase_k)  # consumer wait for k
                    if FUSED_LAYERNORM:
                        tlx.barrier_wait(
                            q_norm_fulls[bufIdx_q], phase_q
                        )  # wait for normalized Q from Act
                    else:
                        tlx.barrier_wait(
                            q_fulls[bufIdx_q], phase_q
                        )  # consumer wait for q0
                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.exit_scope("mma_wait_k_q_pro")

                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.enter_scope("mma_qk_dot_pro")
                    tlx.async_dot(
                        q_buf[bufIdx_q],
                        tlx.local_trans(kv_buf[bufIdx_k]),
                        qk_buf[bufIdx_qk],
                        use_acc=False,
                        mBarriers=[qk_fulls[bufIdx_qk], q_empties[bufIdx_q]],
                    )
                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.exit_scope("mma_qk_dot_pro")

                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.enter_scope("mma_wait_v_pro")
                    bufIdx_v, phase_v = _get_bufidx_phase(
                        accum_cnt_kv + 1, NUM_BUFFERS_KV
                    )
                    tlx.barrier_wait(kv_fulls[bufIdx_v], phase_v)  # consumer wait for v
                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.exit_scope("mma_wait_v_pro")

                    bufIdx_o, phase_o = _get_bufidx_phase(accum_cnt_o, NUM_BUFFERS_O)
                    if LN_Q_RESIDUAL_TO_O:
                        # Wait for LN warp to store Q residual into o_buf
                        tlx.barrier_wait(o_q_fulls[bufIdx_o], phase_o)
                    else:
                        tlx.barrier_wait(
                            o_empties[bufIdx_o], phase_o ^ 1
                        )  # producer acquire for o0

                    # For reuse of qk0 and p0, we can simplify the barriers
                    #   activation partition: consumer wait for qk0, ... update p, producer commit of p0
                    #   dot partition: producer commit of qk0, ..., consumer wait for p0 (use the same barrier as p0_fulls)
                    bufIdx_p, phase_p = _get_bufidx_phase(accum_cnt_qk, NUM_BUFFERS_QK)
                    bufIdx_o, phase_o = _get_bufidx_phase(accum_cnt_o, NUM_BUFFERS_O)
                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.enter_scope("mma_pv_dot_pro")
                    for slice_id in tl.static_range(0, NUM_SUBSLICES):
                        tlx.barrier_wait(
                            p_fulls[bufIdx_p * NUM_SUBSLICES + slice_id], phase_p
                        )  # consumer wait for p0 due to reuse of p0 and qk0
                        # reinterpret qk0 as p0
                        v_slice = tlx.local_slice(
                            kv_buf[bufIdx_v],
                            [BLOCK_N * slice_id // NUM_SUBSLICES, 0],
                            [BLOCK_N // NUM_SUBSLICES, HEAD_DIM],
                        )

                        tlx.async_dot(  # p0 . v -> o0
                            p_buf[bufIdx_p * NUM_SUBSLICES + slice_id],
                            v_slice,
                            o_buf[bufIdx_o],
                            mBarriers=[o_fulls[bufIdx_o]]
                            if slice_id == NUM_SUBSLICES - 1
                            else [],
                            use_acc=LN_Q_RESIDUAL_TO_O or slice_id > 0,
                        )
                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.exit_scope("mma_pv_dot_pro")

                    accum_cnt_q += 1
                    accum_cnt_qk += 1
                    accum_cnt_o += 1

                    for _it in range(lo + BLOCK_M, hi, BLOCK_M):
                        bufIdx_q, phase_q = _get_bufidx_phase(
                            accum_cnt_q, NUM_BUFFERS_Q
                        )
                        bufIdx_qk, phase_qk = _get_bufidx_phase(
                            accum_cnt_qk, NUM_BUFFERS_QK
                        )
                        bufIdx_o, phase_o = _get_bufidx_phase(
                            accum_cnt_o, NUM_BUFFERS_O
                        )

                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("mma_wait_q_loop")
                        if FUSED_LAYERNORM:
                            tlx.barrier_wait(
                                q_norm_fulls[bufIdx_q], phase_q
                            )  # wait for normalized Q from Act
                        else:
                            tlx.barrier_wait(
                                q_fulls[bufIdx_q], phase_q
                            )  # consumer wait for k
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("mma_wait_q_loop")

                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("mma_qk_dot_loop")
                        tlx.async_dot(
                            q_buf[bufIdx_q],
                            tlx.local_trans(kv_buf[bufIdx_k]),
                            qk_buf[bufIdx_qk],
                            use_acc=False,
                            mBarriers=[qk_fulls[bufIdx_qk], q_empties[bufIdx_q]],
                        )
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("mma_qk_dot_loop")

                        bufIdx_o, phase_o = _get_bufidx_phase(
                            accum_cnt_o, NUM_BUFFERS_O
                        )
                        if LN_Q_RESIDUAL_TO_O:
                            # Wait for LN warp to store Q residual into o_buf
                            tlx.barrier_wait(o_q_fulls[bufIdx_o], phase_o)
                        else:
                            tlx.barrier_wait(
                                o_empties[bufIdx_o], phase_o ^ 1
                            )  # producer acquire for o0
                        for slice_id in tl.static_range(0, NUM_SUBSLICES):
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("mma_pv_wait")

                            tlx.barrier_wait(
                                p_fulls[bufIdx_qk * NUM_SUBSLICES + slice_id], phase_qk
                            )  # consumer wait for p1 use p1_fulls due to reuse
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("mma_pv_wait")
                                pl.enter_scope("mma_pv_dot")

                            # reinterpret as p1
                            v_slice = tlx.local_slice(
                                kv_buf[bufIdx_v],
                                [BLOCK_N * slice_id // NUM_SUBSLICES, 0],
                                [BLOCK_N // NUM_SUBSLICES, HEAD_DIM],
                            )

                            tlx.async_dot(
                                p_buf[bufIdx_qk * NUM_SUBSLICES + slice_id],
                                v_slice,
                                o_buf[bufIdx_o],
                                use_acc=LN_Q_RESIDUAL_TO_O or slice_id > 0,
                                mBarriers=[o_fulls[bufIdx_o]]  # TODO p_empty?
                                if slice_id == NUM_SUBSLICES - 1
                                else [],
                            )
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("mma_pv_dot")

                        accum_cnt_q += 1
                        accum_cnt_qk += 1
                        accum_cnt_o += 1

                    # epilogue
                    # commit to release k, v
                    tlx.tcgen05_commit(kv_empties[bufIdx_k])
                    tlx.tcgen05_commit(kv_empties[bufIdx_v])

                    accum_cnt_kv += 2
                if USE_CLC:
                    tile_idx = tlx.clc_consumer(
                        clc_context,
                        clc_phase_consumer,
                        multi_ctas=FUSED_RMS_NORM,
                    )
                    if USE_I64_IDX:
                        tile_idx = tile_idx.to(tl.int64)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx != -1
                    if ENABLE_PROTON:
                        idx += 1
                else:
                    tile_idx += num_progs
                    idx += 1
                    has_more_tile = idx < tiles_per_sm

        # load warp (reduced regs when LN warp present to free register budget)
        with tlx.async_task(num_warps=1, registers=NUM_REGS_LOAD):
            if FUSED_RMS_NORM:
                tlx.cluster_barrier()
            accum_cnt_q = 0
            accum_cnt_kv = 0
            clc_phase_consumer = 0
            idx = 0
            has_more_tile = True
            while has_more_tile:
                begin_q, end_q, begin_k, end_k, qlen, klen = _compute_qlen(
                    tile_idx,
                    n_tile_num,
                    Q_offsets,
                    K_offsets,
                    seq_index,
                    SORT_BY_SEQ_LENGTH,
                    H,
                    N_CTX,
                    ENABLE_LOAD_BALANCING,
                    valid_tiles_b,
                    valid_tiles_m,
                    NUM_VALID_TILES_PER_HEAD,
                    BROADCAST_Q,
                )

                if FUSED_RMS_NORM:
                    off_h = tile_idx % H
                    pid = (tile_idx // H) % n_tile_num
                    off_z = (tile_idx // H) // n_tile_num
                    if SORT_BY_SEQ_LENGTH:
                        off_z = tl.load(seq_index + off_z)
                else:
                    pid = tile_idx % n_tile_num
                    off_hz = tile_idx // n_tile_num
                    off_z = off_hz // H
                    if SORT_BY_SEQ_LENGTH:
                        off_z = tl.load(seq_index + off_z)
                    off_h = off_hz % H
                start_n = pid

                off_h_kv = off_h // G
                q_offset = off_h.to(tl.int64) * stride_qh
                kv_offset = off_h_kv.to(tl.int64) * stride_kh

                if start_n * BLOCK_N < klen:
                    lo, hi = 0, qlen
                    if WINDOW_SIZE is not None:
                        lo = max(
                            lo, ((start_n * BLOCK_N - WINDOW_SIZE) // BLOCK_M) * BLOCK_M
                        )
                        hi = min(hi, (start_n + 1) * BLOCK_N + WINDOW_SIZE)

                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.enter_scope("load_k")
                    k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                    tlx.barrier_wait(kv_empties[k_bufIdx], k_phase ^ 1)
                    tlx.barrier_expect_bytes(kv_fulls[k_bufIdx], BLOCK_N * BLOCK_D * 2)
                    tlx.async_descriptor_load(
                        k_desc,
                        kv_buf[k_bufIdx],
                        [
                            (begin_k + start_n * BLOCK_N).to(tl.int32),
                            (kv_offset).to(tl.int32),
                        ],
                        kv_fulls[k_bufIdx],
                    )
                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.exit_scope("load_k")

                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.enter_scope("load_q_pro")
                    start_m = lo
                    q_bufIdx, q_phase = _get_bufidx_phase(accum_cnt_q, NUM_BUFFERS_Q)
                    tlx.barrier_wait(q_empties[q_bufIdx], q_phase ^ 1)
                    tlx.barrier_expect_bytes(
                        q_fulls[q_bufIdx], BLOCK_M * BLOCK_D * 2
                    )  # num_bytes)
                    tlx.async_descriptor_load(
                        q_desc,
                        q_buf[q_bufIdx],
                        [
                            (begin_q + start_m).to(tl.int32),
                            (q_offset).to(tl.int32),
                        ],
                        q_fulls[q_bufIdx],
                    )
                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.exit_scope("load_q_pro")

                    # Load residual into res_buf (separate SMEM)
                    if FUSED_RESIDUAL_ADD and not FUSED_LAYERNORM:
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("load_res_pro")
                        res_bufIdx, res_phase = _get_bufidx_phase(
                            accum_cnt_q, NUM_BUFFERS_O
                        )
                        # Wait for Act to finish consuming previous residual
                        tlx.barrier_wait(res_empties[res_bufIdx], res_phase ^ 1)
                        tlx.barrier_expect_bytes(
                            res_fulls[res_bufIdx], BLOCK_M * BLOCK_D * 2
                        )
                        tlx.async_descriptor_load(
                            Residual,
                            res_buf[res_bufIdx],
                            [
                                (begin_q + start_m).to(tl.int32),
                                (q_offset).to(tl.int32),
                            ],
                            res_fulls[res_bufIdx],
                        )
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("load_res_pro")

                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.enter_scope("load_v")
                    v_bufIdx, v_phase = _get_bufidx_phase(
                        accum_cnt_kv + 1, NUM_BUFFERS_KV
                    )
                    tlx.barrier_wait(kv_empties[v_bufIdx], v_phase ^ 1)
                    # barrier for producer commit
                    tlx.barrier_expect_bytes(kv_fulls[v_bufIdx], BLOCK_N * BLOCK_D * 2)
                    tlx.async_descriptor_load(
                        v_desc,
                        kv_buf[v_bufIdx],
                        [
                            (begin_k + start_n * BLOCK_N).to(tl.int32),
                            (kv_offset).to(tl.int32),
                        ],
                        kv_fulls[v_bufIdx],
                    )
                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.exit_scope("load_v")
                    accum_cnt_q += 1

                    for start_m in range(lo + BLOCK_M, hi, BLOCK_M):
                        start_m = tl.multiple_of(start_m, BLOCK_M)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("load_q_loop_wait")
                        q_bufIdx, q_phase = _get_bufidx_phase(
                            accum_cnt_q, NUM_BUFFERS_Q
                        )
                        # producer acquire
                        tlx.barrier_wait(q_empties[q_bufIdx], q_phase ^ 1)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("load_q_loop_wait")
                            pl.enter_scope("load_q_loop_tma")
                        # barrier for producer commit
                        tlx.barrier_expect_bytes(
                            q_fulls[q_bufIdx], BLOCK_M * BLOCK_D * 2
                        )  # num_bytes)
                        tlx.async_descriptor_load(
                            q_desc,
                            q_buf[q_bufIdx],
                            [
                                (begin_q + start_m).to(tl.int32),
                                (q_offset).to(tl.int32),
                            ],
                            q_fulls[q_bufIdx],
                        )
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("load_q_loop_tma")

                        # Load residual into res_buf (separate SMEM)
                        if FUSED_RESIDUAL_ADD and not FUSED_LAYERNORM:
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("load_res_loop")
                            res_bufIdx, res_phase = _get_bufidx_phase(
                                accum_cnt_q, NUM_BUFFERS_O
                            )
                            # Wait for Act to finish consuming previous residual
                            tlx.barrier_wait(res_empties[res_bufIdx], res_phase ^ 1)
                            tlx.barrier_expect_bytes(
                                res_fulls[res_bufIdx],
                                BLOCK_M * BLOCK_D * 2,
                            )
                            tlx.async_descriptor_load(
                                Residual,
                                res_buf[res_bufIdx],
                                [
                                    (begin_q + start_m).to(tl.int32),
                                    (q_offset).to(tl.int32),
                                ],
                                res_fulls[res_bufIdx],
                            )
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("load_res_loop")

                        accum_cnt_q += 1
                    # outside of inner for
                    accum_cnt_kv += 2
                if USE_CLC:
                    tile_idx = tlx.clc_consumer(
                        clc_context,
                        clc_phase_consumer,
                        multi_ctas=FUSED_RMS_NORM,
                    )
                    if USE_I64_IDX:
                        tile_idx = tile_idx.to(tl.int64)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx != -1
                    if ENABLE_PROTON:
                        idx += 1
                else:
                    tile_idx += num_progs
                    idx += 1
                    has_more_tile = idx < tiles_per_sm
        with tlx.async_task(num_warps=1, registers=24):  # epilogue
            if FUSED_RMS_NORM:
                tlx.cluster_barrier()
            # Can we guard this with not MERGE_EPI?
            accum_cnt_o = 0
            clc_phase_consumer = 0
            clc_phase_producer = 1
            idx = 0
            has_more_tile = True
            while has_more_tile:
                if USE_CLC:
                    tlx.clc_producer(
                        clc_context,
                        clc_phase_producer,
                        multi_ctas=FUSED_RMS_NORM,
                    )
                    clc_phase_producer = clc_phase_producer ^ 1

                begin_q, end_q, begin_k, end_k, qlen, klen = _compute_qlen(
                    tile_idx,
                    n_tile_num,
                    Q_offsets,
                    K_offsets,
                    seq_index,
                    SORT_BY_SEQ_LENGTH,
                    H,
                    N_CTX,
                    ENABLE_LOAD_BALANCING,
                    valid_tiles_b,
                    valid_tiles_m,
                    NUM_VALID_TILES_PER_HEAD,
                    BROADCAST_Q,
                )

                if FUSED_RMS_NORM:
                    off_h = tile_idx % H
                    pid = (tile_idx // H) % n_tile_num
                    off_z = (tile_idx // H) // n_tile_num
                else:
                    pid = tile_idx % n_tile_num
                    off_hz = tile_idx // n_tile_num
                    off_h = off_hz % H
                    off_z = off_hz // H
                start_n = pid

                out_offset = off_h.to(tl.int64) * stride_oh
                if not BROADCAST_Q:
                    begin_o = begin_q
                    end_o = end_q
                else:
                    begin_o = qlen * off_z
                    end_o = qlen * (off_z + 1)

                if not MERGE_EPI and start_n * BLOCK_N < klen:
                    lo, hi = 0, qlen
                    if WINDOW_SIZE is not None:
                        lo = max(
                            lo, ((start_n * BLOCK_N - WINDOW_SIZE) // BLOCK_M) * BLOCK_M
                        )
                        hi = min(hi, (start_n + 1) * BLOCK_N + WINDOW_SIZE)
                    out_offset = off_h.to(tl.int64) * stride_oh
                    # keeping output as device TMA even for host TMA enabled
                    o_desc = tl.make_tensor_descriptor(
                        Out,
                        shape=[end_o.to(tl.int32), HEAD_DIM * H],
                        strides=[HEAD_DIM * H, 1],
                        block_shape=[BLOCK_M, BLOCK_D],
                    )

                    for start_m in range(lo, hi, BLOCK_M):
                        bufIdx_o, phase_o = _get_bufidx_phase(
                            accum_cnt_o, NUM_BUFFERS_O
                        )
                        bufIdx_y = bufIdx_o
                        phase_y = phase_o

                        if STORE_RMS_NORM_OUT:
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("epi_store_y")
                            # TMA store pre-residual y from y_smem[bufIdx_y]
                            rms_norm_out_desc = tl.make_tensor_descriptor(
                                RmsNormOut,
                                shape=[end_o.to(tl.int32), HEAD_DIM * H],
                                strides=[HEAD_DIM * H, 1],
                                block_shape=[BLOCK_M, BLOCK_D],
                            )
                            tlx.barrier_wait(y_smem_fulls[bufIdx_y], phase_y)
                            tlx.async_descriptor_store(
                                rms_norm_out_desc,
                                y_smem[bufIdx_y],
                                [
                                    (begin_o + start_m).to(tl.int32),
                                    (out_offset).to(tl.int32),
                                ],
                            )
                            tlx.async_descriptor_store_wait(0)
                            tlx.barrier_arrive(y_smem_empties[bufIdx_y])
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("epi_store_y")

                        # TMA store o
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("epi_store_o")
                        tlx.barrier_wait(o_smem_fulls[bufIdx_o], phase_o)
                        tlx.async_descriptor_store(
                            o_desc,
                            o_smem[bufIdx_o],
                            [
                                (begin_o + start_m).to(tl.int32),
                                (out_offset).to(tl.int32),
                            ],
                        )
                        tlx.async_descriptor_store_wait(0)
                        tlx.barrier_arrive(o_smem_empties[bufIdx_o])
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("epi_store_o")

                        accum_cnt_o += 1
                if USE_CLC:
                    tile_idx = tlx.clc_consumer(
                        clc_context,
                        clc_phase_consumer,
                        multi_ctas=FUSED_RMS_NORM,
                    )
                    if USE_I64_IDX:
                        tile_idx = tile_idx.to(tl.int64)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx != -1
                    if ENABLE_PROTON:
                        idx += 1
                else:
                    tile_idx += num_progs
                    idx += 1
                    has_more_tile = idx < tiles_per_sm

        # Empty warp group
        # Brings total warps to a multiple of 4 which is required for cluster_barrier.
        if FUSED_RMS_NORM:
            with tlx.async_task(num_warps=1, registers=24):
                tlx.cluster_barrier()


def next_power_of_2(x):
    return 2 ** (math.ceil(math.log(x, 2)))


def expect_contiguous(x: torch.Tensor) -> torch.Tensor:
    if x is not None and x.stride(-1) != 1:
        return x.contiguous()
    return x


# assume is_predict: tl.constexpr,  #  false
#    FUSED_QKV: tl.constexpr,  # false
#    FUSED_KV: tl.constexpr,  # false
#    SORT_BY_SEQ_LENGTH: tl.constexpr,  false
#    STAGE: tl.constexpr,  #
#    USE_START_END_OFFSETS: tl.constexpr,  false
#    WINDOW_SIZE: tl.constexpr,
#    BROADCAST_Q: tl.constexpr, false
#    IS_DENSE_KV: tl.constexpr,  (true)
@torch.library.custom_op("ads_mkl::tlx_gdpa_megakernel", mutates_args=())
def _tlx_generalized_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_offset: torch.Tensor,
    key_offset: torch.Tensor,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    ad_to_request_offset: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    attn_offset: torch.Tensor | None = None,
    is_causal: bool = False,
    qk_scale: float | None = None,
    seq_index: torch.Tensor | None = None,
    allow_tf32: bool = True,
    output_offset: torch.Tensor | None = None,
    use_start_end_offsets: bool = False,
    window_size: int | None = None,
    broadcast_q: bool = False,
    activation: str = "fast_gelu",
    enable_persistent: bool = False,
    enable_tma: bool = False,
    enable_ws: bool = False,
    use_dq_atomic_add: bool = False,
    total_num_objects: int | None = None,
    bwd_opt_tech: str = "base",
    cpu_query_offset: torch.Tensor | None = None,
    use_on_device_tma: bool = False,
    ensemble_activation_list: List[int] | None = None,
    fused_rms_norm: bool = False,
    rms_norm_weight: torch.Tensor | None = None,
    fused_residual_add: bool = False,
    residual: torch.Tensor | None = None,
    fused_q_residual_add: bool = False,
    fused_layernorm: bool = False,
    layernorm_weight: torch.Tensor | None = None,
    layernorm_bias: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if qk_scale is None:
        qk_scale = 1.0

    HEAD_DIM_Q = query.shape[-1]
    HEAD_DIM_K = key.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    # HEAD_DIM_V = value.shape[-1]
    sort_by_seq_length = seq_index is not None

    if output_offset is None:
        output_offset = query_offset

    # check whether kv is dense tensor
    bs = key_offset.size(0) - 1
    L, _, _ = key.shape
    is_dense_kv = bs * max_seq_len_kv == L

    # assume BN = 128
    num_inner_iter = triton.cdiv(max_seq_len_kv, 128)

    BLOCK_D = max(next_power_of_2(HEAD_DIM_Q), 16)
    if broadcast_q:
        BATCH = key_offset.size(0) - 1
    else:
        BATCH = (
            query_offset.size(0) // 2
            if use_start_end_offsets
            else query_offset.size(0) - 1
        )

    if use_start_end_offsets:
        o = torch.empty(
            (
                total_num_objects,
                query.shape[1],
                HEAD_DIM_Q,
            ),
            device=query.device,
            dtype=query.dtype,
        )
    else:
        o = torch.empty(
            (
                BATCH * query.shape[0] if broadcast_q else query.shape[0],
                query.shape[1],
                HEAD_DIM_Q,
            ),
            device=query.device,
            dtype=query.dtype,
        )
    # Allocate rrms output for backward when fused_rms_norm
    # Always allocate (even if unused) since custom_op return type cannot be Optional
    rrms_out = torch.empty(
        o.shape[0] if fused_rms_norm else 0,
        device=query.device,
        dtype=torch.float32,
    )

    # Allocate rms_norm_out to store pre-residual RMSNorm output for backward.
    # When fused_q_residual_add + fused_layernorm, backward reconstructs
    # y = output - raw_Q instead, so skip the store.
    store_rms_norm_out = (
        fused_rms_norm
        and fused_residual_add
        and not (fused_q_residual_add and fused_layernorm)
    )
    if store_rms_norm_out:
        rms_norm_out = torch.empty_like(o)
    else:
        rms_norm_out = torch.empty(0, device=query.device, dtype=query.dtype)

    # Allocate layernorm mean/rstd outputs for backward
    if fused_layernorm:
        assert fused_rms_norm, (
            "FUSED_LAYERNORM requires FUSED_RMS_NORM (same clustering infrastructure)"
        )
        ln_mean_out = torch.empty(o.shape[0], device=query.device, dtype=torch.float32)
        ln_rstd_out = torch.empty(o.shape[0], device=query.device, dtype=torch.float32)
    else:
        ln_mean_out = torch.empty(0, device=query.device, dtype=torch.float32)
        ln_rstd_out = torch.empty(0, device=query.device, dtype=torch.float32)

    stage = 1  # When supporting causal, change to 3
    extra_kern_args = {}
    # extra_kern_args["maxnreg"] = 168
    nheads = query.shape[1]
    G = query.shape[1] // key.shape[1]
    assert query.shape[1] % key.shape[1] == 0
    # batch_size = BATCH * nheads
    NUM_SMS = (
        get_num_sms() or 1000000
    )  # * 8  # if num sms is None, use a large number so that it is a no-op
    if (
        AUTOTUNE_CONFIG_SET == "omnifm_v2"
        or AUTOTUNE_CONFIG_SET == "omnifm_v4_disable_autotune"
    ):
        NUM_SMS = NUM_SMS - RESERVED_SMS_FOR_COMMS
    # print("NUM_SMS", NUM_SMS)
    # print(triton.cdiv(max_seq_len_q, 256) * BATCH * nheads)

    q = expect_contiguous(query)
    k = expect_contiguous(key)
    v = expect_contiguous(value)
    kstrides = k.stride()
    vstrides = v.stride()

    dummy_block = [1, 1]
    _N_CTX_KV = max_seq_len_kv  # noqa: F841
    _N_CTX = max_seq_len_q  # noqa: F841
    HEAD_DIM = HEAD_DIM_K
    _Z = BATCH  # noqa: F841
    H = nheads
    y_dim_q = query.shape[0]  # for Q and O descriptors
    y_dim_kv = key.shape[0]  # for K and V descriptors
    x_dim = HEAD_DIM * H // G
    if fused_rms_norm:
        # TODO supports the trivial case with H = 1
        assert H > 1, "RMSNorm fusion only supports multiple heads at the moment"

    if not use_on_device_tma:
        desc_q = TensorDescriptor(
            q,
            shape=[y_dim_q, HEAD_DIM * H],
            strides=[q.stride(0), q.stride(2)],
            block_shape=dummy_block,
        )
        desc_v = TensorDescriptor(
            v,
            shape=[y_dim_kv, x_dim],
            strides=[v.stride(0), v.stride(2)],
            block_shape=dummy_block,
        )
        desc_k = TensorDescriptor(
            k,
            shape=[y_dim_kv, x_dim],
            strides=[k.stride(0), k.stride(2)],
            block_shape=dummy_block,
        )
        # XXX: leaving output as on-device TMA due to numerics issues
        # desc_o = TensorDescriptor(
        #     o,
        #     shape=[y_dim_q, HEAD_DIM * H],
        #     strides=[o.stride(0), o.stride(2)],
        #     block_shape=dummy_block,
        # )

    # Create Residual TMA descriptor for generic fused_residual_add
    if fused_residual_add and residual is not None:
        res = expect_contiguous(residual)
        if not use_on_device_tma:
            desc_residual = TensorDescriptor(
                res,
                shape=[y_dim_q, HEAD_DIM * H],
                strides=[res.stride(0), res.stride(2)],
                block_shape=dummy_block,
            )
        else:
            desc_residual = res
    else:
        desc_residual = q if use_on_device_tma else desc_q  # dummy, won't be used

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, _):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    use_short_kv_kernel = (
        max_seq_len_kv in (64, 128)
        and is_dense_kv
        and not broadcast_q
        and window_size is None
    )

    if (
        cpu_query_offset is None or broadcast_q or use_short_kv_kernel
    ):  # broadcast_q / short_kv case does not need load balancing
        enable_load_balancing = False
        valid_tiles_m, valid_tiles_b = None, None
        num_valid_tiles_per_head = None
    else:
        enable_load_balancing = True
        valid_tiles_m, valid_tiles_b = compute_valid_tiles(
            tuple(cpu_query_offset.tolist()), bs, 256
        )  # TODO this hardcodes BLOCK_M to 256
        num_valid_tiles_per_head = len(valid_tiles_b)
        valid_tiles_m = valid_tiles_m.pin_memory().to("cuda", non_blocking=True)
        valid_tiles_b = valid_tiles_b.pin_memory().to("cuda", non_blocking=True)

    activation_enum_int = activation_string_to_int(activation)
    ensemble_activation_list = ensemble_activation_list or []

    enable_proton = True if os.getenv("ENABLE_PROTON") == "1" else False

    # LayerNorm prologue params — only supported by short-KV kernel
    if fused_layernorm:
        assert use_short_kv_kernel, (
            "FUSED_LAYERNORM only supported for short-KV kernel (max_seq_len_kv in {64, 128})"
        )

    # Always pass layernorm params to short-KV kernel (with dummies when disabled)
    # to avoid Triton compiler NoneType errors on default param values
    if use_short_kv_kernel:
        _dummy_ptr = rrms_out  # non-None pointer for type resolution
        extra_kern_args["FUSED_LAYERNORM"] = fused_layernorm
        extra_kern_args["layernorm_weight"] = (
            layernorm_weight if layernorm_weight is not None else _dummy_ptr
        )
        extra_kern_args["layernorm_bias"] = (
            layernorm_bias if layernorm_bias is not None else _dummy_ptr
        )
        extra_kern_args["HAS_LAYERNORM_WEIGHT"] = layernorm_weight is not None
        extra_kern_args["HAS_LAYERNORM_BIAS"] = layernorm_bias is not None
        extra_kern_args["ln_mean_out"] = ln_mean_out
        extra_kern_args["ln_rstd_out"] = ln_rstd_out

    if use_short_kv_kernel:
        # short kv kernel
        def grid_tma_persistent(META):
            total_ctas = triton.cdiv(max_seq_len_kv, META["BLOCK_N"]) * BATCH * nheads

            if not META["USE_CLC"]:
                num_sms = NUM_SMS
                if fused_rms_norm:
                    num_sms = (
                        num_sms // nheads * nheads
                    )  # ensure grid size is a multiple of CTA cluster size
                total_ctas = min(num_sms, total_ctas)

            return (total_ctas, 1, 1)

        unrolled_kernel = unroll_varargs(
            gdpa_kernel_tma_ws_blackwell_short_kv,
            N=len(ensemble_activation_list),
            unroll_as_const=True,
        )

        ctas_per_cga = (nheads, 1, 1) if fused_rms_norm else (1, 1, 1)
        num_buffers_kv = 2
        autotuned_kernel_fn = _get_autotune_kernel(
            unrolled_kernel,
            enable_load_balancing=enable_load_balancing,
            short_kv=True,
            short_kv_block_n=max_seq_len_kv,  # can only be 64 or 128
            ctas_per_cga=ctas_per_cga,
            num_buffers_kv=num_buffers_kv,
        )

        # short kv kernel only supports dense kv with length 64, 128. So kv must be aligned
        is_aligned_kv = True
    else:

        def grid_tma_persistent(META):
            if META["ENABLE_LOAD_BALANCING"]:
                total_ctas = num_valid_tiles_per_head * nheads
            else:
                total_ctas = (
                    triton.cdiv(max_seq_len_q, META["BLOCK_M"]) * BATCH * nheads
                )

            if not META["USE_CLC"]:
                num_sms = NUM_SMS
                if META["FUSED_RMS_NORM"]:
                    num_sms = (
                        num_sms // nheads * nheads
                    )  # ensure grid size is a multiple of CTA cluster size
                total_ctas = min(num_sms, total_ctas)

            return (total_ctas, 1, 1)

        unrolled_kernel = unroll_varargs(
            gdpa_kernel_tma_ws_blackwell,
            N=len(ensemble_activation_list),
            unroll_as_const=True,
        )
        ctas_per_cga = (nheads, 1, 1) if fused_rms_norm else (1, 1, 1)
        num_buffers_kv = 2 if fused_rms_norm else 3
        autotuned_kernel_fn = _get_autotune_kernel(
            unrolled_kernel,
            enable_load_balancing=enable_load_balancing,
            ctas_per_cga=ctas_per_cga,
            num_buffers_kv=num_buffers_kv,
        )

        is_aligned_kv = is_dense_kv and max_seq_len_kv % 128 == 0

    if enable_proton:
        proton_mode = proton.mode.Default(metric_type="cycle", optimizations="clock32")
        proton.start(
            "proton_fwd", data="trace", backend="instrumentation", mode=proton_mode
        )

    autotuned_kernel_fn[grid_tma_persistent](
        *ensemble_activation_list,
        q if use_on_device_tma else desc_q,
        query_offset,
        k if use_on_device_tma else desc_k,
        key_offset,
        v if use_on_device_tma else desc_v,
        o,  # if use_on_device_tma else desc_o,
        output_offset,
        ad_to_request_offset,
        seq_index,
        q.stride(0),
        q.stride(1),
        q.stride(2),  #
        kstrides[0],
        kstrides[1],
        kstrides[2],  #
        vstrides[0],
        vstrides[1],
        vstrides[2],  #
        o.stride(0),
        o.stride(1),
        o.stride(2),  #
        BATCH,
        nheads,  #
        G,
        N_CTX=max_seq_len_q,
        N_CTX_KV=max_seq_len_kv,  #
        total_len_q=query.shape[0],
        total_len_kv=key.shape[0],  #
        qk_scale=qk_scale,
        is_predict=False,
        Q_SHAPE_0=query.shape[0],
        FUSED_QKV=False,  # fused_qkv,
        FUSED_KV=False,  # fused_kv,
        SORT_BY_SEQ_LENGTH=sort_by_seq_length,
        HEAD_DIM=HEAD_DIM_K,  #
        BLOCK_D=BLOCK_D,
        STAGE=stage,  #
        USE_START_END_OFFSETS=use_start_end_offsets,
        WINDOW_SIZE=window_size,
        BROADCAST_Q=broadcast_q,
        IS_DENSE_KV=is_dense_kv,
        IS_ALIGNED_KV=is_aligned_kv,
        activation_enum_int=activation_enum_int,
        USE_ON_DEVICE_TMA=use_on_device_tma,
        NUM_INNER_ITER=num_inner_iter,
        MERGE_EPI=False,
        ENABLE_PROTON=enable_proton,
        PROTON_TILE=10,
        ENABLE_LOAD_BALANCING=enable_load_balancing,
        valid_tiles_b=valid_tiles_b,
        valid_tiles_m=valid_tiles_m,
        NUM_VALID_TILES_PER_HEAD=num_valid_tiles_per_head,
        USE_I64_IDX=should_use_i64_idx(q, k, v, o),
        NUM_MMA_GROUPS=1 if use_short_kv_kernel else 2,
        FUSED_RMS_NORM=fused_rms_norm,
        NUM_REDUCTION_CTAS=nheads if fused_rms_norm else 1,
        rms_norm_weight=rms_norm_weight,
        HAS_RMS_NORM_WEIGHT=rms_norm_weight is not None,
        rrms_out=rrms_out,
        FUSED_RESIDUAL_ADD=fused_residual_add,
        Residual=desc_residual,
        FUSED_Q_RESIDUAL_ADD=fused_q_residual_add,
        RmsNormOut=rms_norm_out if store_rms_norm_out else o,  # dummy when not storing
        STORE_RMS_NORM_OUT=store_rms_norm_out,
        **extra_kern_args,
    )

    if enable_proton:
        torch.cuda.synchronize()
        proton.finalize()

    return o, rrms_out, rms_norm_out, ln_mean_out, ln_rstd_out


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


@register_flop_formula(torch.ops.ads_mkl.tlx_gdpa_megakernel, get_raw=True)
def generalized_dot_product_attention_forward_flop(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_offset: torch.Tensor,
    key_offset: torch.Tensor,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    ad_to_request_offset: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    attn_offset: torch.Tensor | None = None,
    is_causal: bool = True,
    qk_scale: float | None = None,
    seq_index: torch.Tensor | None = None,
    allow_tf32: bool = True,
    output_offset: torch.Tensor | None = None,
    use_start_end_offsets: bool = False,
    window_size: int | None = None,
    broadcast_q: bool = False,
    *args,
    **kwargs,
) -> int:
    """Count flops for self-attention."""
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
    if bs_q != bs_k:
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


def _generalized_dot_product_attention_setup_context(ctx, inputs, output):
    (
        query,
        key,
        value,
        query_offset,
        key_offset,
        max_seq_len_q,  #
        max_seq_len_kv,  #
        ad_to_request_offset,
        attn_mask,
        attn_offset,
        is_causal,
        qk_scale,
        seq_index,
        allow_tf32,
        output_offset,
        use_start_end_offsets,
        window_size,
        broadcast_q,
        activation,
        enable_persistent,
        enable_tma,
        enable_ws,
        use_dq_atomic_add,
        total_num_objects,
        bwd_opt_tech,
        cpu_query_offset,
        use_on_device_tma,
        ensemble_activation_list,
        fused_rms_norm,
        rms_norm_weight,
        fused_residual_add,
        residual,
        fused_q_residual_add,
        fused_layernorm,
        layernorm_weight,
        layernorm_bias,
    ) = inputs
    o, rrms_out, rms_norm_out, ln_mean_out, ln_rstd_out = output
    fused_qkv = key is None and value is None
    fused_kv = key is not None and value is None
    if fused_qkv:
        HEAD_DIM_K = query.shape[-1] // 3
    elif fused_kv:
        HEAD_DIM_K = key.shape[-1] // 2
    else:
        HEAD_DIM_K = key.shape[-1]
    BLOCK_D = max(next_power_of_2(HEAD_DIM_K), 32)

    if broadcast_q:
        BATCH = key_offset.size(0) - 1
    else:
        BATCH = (
            query_offset.size(0) // 2
            if use_start_end_offsets
            else query_offset.size(0) - 1
        )
    nheads = query.shape[1]
    sort_by_seq_length = seq_index is not None
    if fused_qkv:
        # fused qkv only support q/k/v have the same shape
        nheads_k = nheads
    else:
        nheads_k = key.shape[1]
    assert (nheads % nheads_k) == 0

    if output_offset is None:
        output_offset = query_offset

    # Save rrms_out and rms_norm_weight for RMS norm backward
    tensors_to_save = [
        query,
        key,
        value,
        o,
        query_offset,
        key_offset,
        output_offset,
        seq_index,
    ]
    # When both fused_rms_norm and fused_residual_add, the backward needs
    # pre-residual y = RMSNorm(x) rather than post-residual o = y + residual.
    # When fused_q_residual_add + fused_layernorm, backward reconstructs y from
    # output - raw_Q instead, so we keep the actual output in tensors_to_save[3].
    if (
        fused_rms_norm
        and fused_residual_add
        and not (fused_q_residual_add and fused_layernorm)
        and rms_norm_out.numel() > 0
    ):
        tensors_to_save[3] = rms_norm_out
    if fused_rms_norm:
        tensors_to_save.append(rrms_out)
    if rms_norm_weight is not None:
        tensors_to_save.append(rms_norm_weight)
    # Save LayerNorm tensors for backward
    if fused_layernorm:
        # Save mean/rstd from forward kernel for exact LN(Q) reconstruction
        tensors_to_save.append(ln_mean_out)
        tensors_to_save.append(ln_rstd_out)
        if layernorm_weight is not None:
            tensors_to_save.append(layernorm_weight)
        if layernorm_bias is not None:
            tensors_to_save.append(layernorm_bias)
    ctx.save_for_backward(*tensors_to_save)
    ctx.fused_rms_norm = fused_rms_norm
    ctx.has_rms_norm_weight = rms_norm_weight is not None
    ctx.fused_layernorm = fused_layernorm
    ctx.has_layernorm_weight = layernorm_weight is not None
    ctx.has_layernorm_bias = layernorm_bias is not None
    ctx.HEAD_DIM = HEAD_DIM_K
    ctx.BLOCK_D = BLOCK_D
    ctx.causal = is_causal
    ctx.N_CTX = max_seq_len_q
    ctx.N_CTX_KV = max_seq_len_kv
    ctx.BATCH = BATCH
    ctx.N_HEAD = nheads
    ctx.G = nheads // nheads_k
    ctx.fused_qkv = fused_qkv
    ctx.fused_kv = fused_kv
    ctx.use_start_end_offsets = use_start_end_offsets
    ctx.window_size = window_size
    ctx.broadcast_q = broadcast_q
    # ctx only support int type
    ctx.activation_enum_int = activation_string_to_int(activation)
    ctx.ensemble_activation_list = ensemble_activation_list
    ctx.enable_persistent = enable_persistent and "persistent" in bwd_opt_tech
    ctx.enable_tma = enable_tma and "tma" in bwd_opt_tech
    ctx.enable_ws = enable_ws and "ws" in bwd_opt_tech
    ctx.use_dq_atomic_add = use_dq_atomic_add and "dq_atomic_add" in bwd_opt_tech
    ctx.qk_scale = qk_scale
    ctx.sort_by_seq_length = sort_by_seq_length
    ctx.fused_residual_add = fused_residual_add
    ctx.fused_q_residual_add = fused_q_residual_add
    ctx.has_residual = residual is not None


@triton.jit
def _gdpa_bwd_tlx_compute_num_steps(
    qlen,
    start_n,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
):
    start_m_inner = 0
    num_steps = tl.cdiv((qlen - start_m_inner), BLOCK_M1)
    if WINDOW_SIZE is not None:
        start_m_inner = (
            max(start_m_inner, start_n - WINDOW_SIZE) // BLOCK_M1
        ) * BLOCK_M1
        end_m_inner = (
            tl.cdiv(min(qlen, start_n + BLOCK_N1 + WINDOW_SIZE), BLOCK_M1) * BLOCK_M1
        )
        num_steps = (end_m_inner - start_m_inner) // BLOCK_M1
    return num_steps.to(tl.int32), start_m_inner


@triton.jit
def _gdpa_bwd_tlx_compute_activation(
    desc_q,
    desc_dk,
    DK,
    desc_dv,
    DV,
    k_tiles,
    v_tiles,
    qk_tiles,
    dpT_tiles,
    ppT_tiles,
    dsT_tiles,
    dv_tiles,
    dk_tiles,
    qk_fulls,
    dpT_fulls,
    ppT_fulls,
    dsT_fulls,
    dv_fulls,
    dk_fulls,
    dv_empties,
    dk_empties,
    off_h_kv,
    begin_k,
    accum_cnt_outer,
    accum_cnt_inner,
    stride_km,
    stride_kh,
    stride_d,
    start_n,
    start_m,
    qlen,
    klen,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    MASK: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    NUM_BUFFERS_TMEM: tl.constexpr,
    NUM_ACT_SPLITS: tl.constexpr,
    activation_enum_int: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    IS_ALIGNED_KV: tl.constexpr,
    # Fused RMS norm backward params
    do_tiles=None,
    do_fulls=None,
    do_raw_fulls=None,
    do_empties=None,
    do_rms_tiles=None,
    do_rms_empties=None,
    o_tiles=None,
    o_fulls=None,
    o_empties=None,
    rrms_out=None,
    rms_norm_weight=None,
    reduction_buf=None,
    reduction_barriers=None,
    begin_q=0,
    off_h=0,
    stride_oh=0,
    NUM_BUFFERS_DO: tl.constexpr = 1,
    NUM_BUFFERS_DO_RMS: tl.constexpr = 1,
    NUM_BUFFERS_O: tl.constexpr = 1,
    FUSED_RMS_NORM: tl.constexpr = False,
    HAS_RMS_NORM_WEIGHT: tl.constexpr = False,
    NUM_REDUCTION_CTAS: tl.constexpr = 0,
    NUM_REDUCTION_BUFS: tl.constexpr = 2,
    N_RMS: tl.constexpr = 0,
    CROSS_CTA_REDUCTION_EXPECTED_BYTES: tl.constexpr = 0,
    dw_accum_buf=None,
    w_slice_buf=None,
    dw_scratch_buf=None,
    dw_scratch_fulls=None,
    dw_scratch_empties=None,
    NUM_BUFFERS_DW_SCRATCH: tl.constexpr = 1,
    FUSED_Q_RESIDUAL_ADD: tl.constexpr = False,
    # Fused LayerNorm params
    FUSED_LAYERNORM: tl.constexpr = False,
    q_tiles=None,
    q_fulls=None,
    q_norm_fulls=None,
    NUM_BUFFERS_Q: tl.constexpr = 1,
    layernorm_weight_ptr=None,
    layernorm_bias_ptr=None,
    ln_w_buf=None,
    ln_b_buf=None,
    HAS_LAYERNORM_WEIGHT: tl.constexpr = False,
    HAS_LAYERNORM_BIAS: tl.constexpr = False,
    ln_mean_out=None,
    ln_rstd_out=None,
    dpT_empties=None,
    ENABLE_PROTON: tl.constexpr = False,
    PROTON_TILE: tl.constexpr = 0,
    idx=0,
    # dKV SMEM staging params
    sdv_store_buf=None,
    sdk_store_buf=None,
    k_mma_done=None,
    v_empties=None,
    k_empties=None,
    NUM_BUFFERS_KV: tl.constexpr = 1,
):
    dkv_buf_id, dkv_phase = _get_bufidx_phase(accum_cnt_outer, NUM_BUFFERS_TMEM)
    dk_tile = dk_tiles[dkv_buf_id]
    dv_tile = dv_tiles[dkv_buf_id]
    dk_full = dk_fulls[dkv_buf_id]
    dv_full = dv_fulls[dkv_buf_id]
    dk_empty = dk_empties[dkv_buf_id]
    dv_empty = dv_empties[dkv_buf_id]

    num_steps, start_m = _gdpa_bwd_tlx_compute_num_steps(
        qlen, start_n, BLOCK_M1, BLOCK_N1, WINDOW_SIZE
    )

    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    for i in tl.range(0, num_steps, 1, loop_unroll_factor=1):
        tmem_buf_id, tmem_phase = _get_bufidx_phase(
            accum_cnt_inner + i, NUM_BUFFERS_TMEM
        )
        qk_tile = qk_tiles[tmem_buf_id]
        ppT_tile = ppT_tiles[tmem_buf_id]
        dpT_tile = dpT_tiles[tmem_buf_id]
        dsT_tile = dsT_tiles[tmem_buf_id]
        qk_full = qk_fulls[tmem_buf_id]
        ppT_full = ppT_fulls[tmem_buf_id]
        dpT_full = dpT_fulls[tmem_buf_id]
        dsT_full = dsT_fulls[tmem_buf_id]

        # Fused LayerNorm: compute LN(Q) using precomputed mean/rstd from forward.
        # No cross-CTA DSMEM reduction needed — just load scalars and normalize.
        if FUSED_LAYERNORM:
            q_buf_id, q_phase = _get_bufidx_phase(accum_cnt_inner + i, NUM_BUFFERS_Q)
            q_tile_ln = q_tiles[q_buf_id]
            q_full_ln = q_fulls[q_buf_id]
            q_norm_full = q_norm_fulls[q_buf_id]

            # Wait for raw Q from load warp
            if ENABLE_PROTON and idx == PROTON_TILE:
                pl.enter_scope("act_wait_q_ln")
            tlx.barrier_wait(q_full_ln, q_phase)
            if ENABLE_PROTON and idx == PROTON_TILE:
                pl.exit_scope("act_wait_q_ln")

            if ENABLE_PROTON and idx == PROTON_TILE:
                pl.enter_scope("act_compute_ln_q")
            raw_q = tlx.local_load(q_tile_ln).to(tl.float32)

            # Load precomputed mean/rstd from forward pass (per-row scalars)
            ln_offs = begin_q + start_m + i * BLOCK_M1 + tl.arange(0, BLOCK_M1)
            ln_mask = ln_offs < (begin_q + qlen)
            mean_vals = tl.load(ln_mean_out + ln_offs, mask=ln_mask, other=0.0)[:, None]
            rstd_vals = tl.load(ln_rstd_out + ln_offs, mask=ln_mask, other=1.0)[:, None]

            # Compute LN(Q) = (raw_q - mean) * rstd * weight [+ bias]
            ln_q = _mul_f32x2(_sub_f32x2(raw_q, mean_vals), rstd_vals)

            # Read LN weight/bias from SMEM cache (loaded once before persistent loop)
            if HAS_LAYERNORM_WEIGHT:
                ln_w = tlx.local_load(ln_w_buf[0])  # (1, BLOCK_D) f32
                ln_q = ln_w * ln_q
            if HAS_LAYERNORM_BIAS:
                ln_b = tlx.local_load(ln_b_buf[0])  # (1, BLOCK_D) f32
                ln_q = ln_q + ln_b

            # Write LN(Q) back to q_tile for MMA
            dtype = tlx.dtype_of(desc_q)
            tlx.local_store(q_tile_ln, ln_q.to(dtype))
            tlx.fence_async_shared()
            tlx.barrier_arrive(q_norm_full)
            if ENABLE_PROTON and idx == PROTON_TILE:
                pl.exit_scope("act_compute_ln_q")

        # Fused RMS norm backward: transform do[i] before MMA uses it
        if FUSED_RMS_NORM:
            do_buf_id, do_phase = _get_bufidx_phase(accum_cnt_inner + i, NUM_BUFFERS_DO)
            do_tile = do_tiles[do_buf_id]
            do_full = do_fulls[do_buf_id]
            do_raw_full = do_raw_fulls[do_buf_id]
            do_empty = do_empties[do_buf_id]
            do_rms_buf_id, do_rms_phase = _get_bufidx_phase(
                accum_cnt_inner + i, NUM_BUFFERS_DO_RMS
            )
            do_rms_tile = do_rms_tiles[do_rms_buf_id]
            do_rms_empty = do_rms_empties[do_rms_buf_id]
            o_buf_id, o_phase = _get_bufidx_phase(accum_cnt_inner + i, NUM_BUFFERS_O)
            o_tile = o_tiles[o_buf_id]
            o_full = o_fulls[o_buf_id]
            o_empty = o_empties[o_buf_id]

            # Wait for raw do and o from load warp
            if ENABLE_PROTON and idx == PROTON_TILE:
                pl.enter_scope("act_wait_do_raw")
            tlx.barrier_wait(do_raw_full, do_phase)
            if ENABLE_PROTON and idx == PROTON_TILE:
                pl.exit_scope("act_wait_do_raw")
            if ENABLE_PROTON and idx == PROTON_TILE:
                pl.enter_scope("act_wait_o")
            tlx.barrier_wait(o_full, o_phase)
            if ENABLE_PROTON and idx == PROTON_TILE:
                pl.exit_scope("act_wait_o")

            if ENABLE_PROTON and idx == PROTON_TILE:
                pl.enter_scope("act_load_dy")
            dy = tlx.local_load(do_tile).to(tl.float32)
            if ENABLE_PROTON and idx == PROTON_TILE:
                pl.exit_scope("act_load_dy")
            # Raw do read complete — release do_tile for load warp.
            # (do_empties arrive_count accounts for RED warp if FUSED_RESIDUAL_ADD.)
            tlx.fence_async_shared()
            tlx.barrier_arrive(do_empty)
            if ENABLE_PROTON and idx == PROTON_TILE:
                pl.enter_scope("act_load_o")
            if FUSED_Q_RESIDUAL_ADD and FUSED_LAYERNORM:
                # Reconstruct y = output - raw_Q. The post-norm residual R
                # is the raw Q tensor, so output = y + R = y + raw_Q.
                # raw_q is already in registers from the LN block above.
                y = tlx.local_load(o_tile).to(tl.float32) - raw_q
            else:
                y = tlx.local_load(o_tile).to(tl.float32)
            if ENABLE_PROTON and idx == PROTON_TILE:
                pl.exit_scope("act_load_o")

            if ENABLE_PROTON and idx == PROTON_TILE:
                pl.enter_scope("act_load_rrms")
            # Load rrms for this tile (masked for padded rows)
            rrms_offs = begin_q + start_m + i * BLOCK_M1 + tl.arange(0, BLOCK_M1)
            rrms_mask = rrms_offs < (begin_q + qlen)
            rrms_vals = tl.load(rrms_out + rrms_offs, mask=rrms_mask, other=1.0)[
                :, None
            ]
            if ENABLE_PROTON and idx == PROTON_TILE:
                pl.exit_scope("act_load_rrms")

            # Load rms_norm_weight slice from SMEM cache (loaded once before persistent loop)
            if HAS_RMS_NORM_WEIGHT:
                w_slice = tl.sum(
                    tlx.local_load(w_slice_buf[0]).to(tl.float32), axis=0
                )  # [1,D] -> [D]
            else:
                w_slice = None

            # Double-buffered DSMEM reduction
            red_buf_id = (accum_cnt_inner + i) % NUM_REDUCTION_BUFS
            red_phase = ((accum_cnt_inner + i) // NUM_REDUCTION_BUFS) & 1
            red_barrier = reduction_barriers[red_buf_id]
            red_buf_offset = red_buf_id * (NUM_REDUCTION_CTAS - 1)
            cluster_cta_rank = tlx.cluster_cta_rank()

            if ENABLE_PROTON and idx == PROTON_TILE:
                pl.enter_scope("act_compute_rms_bwd")
            do_gdpa = _compute_rms_norm_backward(
                dy,
                y,
                rrms_vals,
                cluster_cta_rank,
                reduction_buf,
                red_barrier,
                red_phase,
                red_buf_offset,
                BLOCK_M1,
                N_RMS,
                NUM_REDUCTION_CTAS,
                CROSS_CTA_REDUCTION_EXPECTED_BYTES,
                w_slice,
                HAS_RMS_NORM_WEIGHT,
                idx,
                PROTON_TILE,
                ENABLE_PROTON,
            )

            # Write do_gdpa to separate do_rms buffer (not overwriting do_tile).
            dtype = tlx.dtype_of(desc_q)
            if ENABLE_PROTON and idx == PROTON_TILE:
                pl.enter_scope("act_compute_rms_bwd_wait_empty")
            tlx.barrier_wait(do_rms_empty, do_rms_phase ^ 1)
            if ENABLE_PROTON and idx == PROTON_TILE:
                pl.exit_scope("act_compute_rms_bwd_wait_empty")
            tlx.local_store(do_rms_tile, do_gdpa.to(dtype))
            tlx.fence_async_shared()
            tlx.barrier_arrive(do_full)  # Tell MMA: do_gdpa ready in do_rms_tile
            tlx.barrier_arrive(o_empty)  # Release o buffer for load warp
            if ENABLE_PROTON and idx == PROTON_TILE:
                pl.exit_scope("act_compute_rms_bwd")

            # Accumulate drms_norm_weight: store intermediate to SMEM scratch,
            # defer tl.sum(axis=0) to the reduction warp (off activation critical path)
            if HAS_RMS_NORM_WEIGHT and start_n == 0:
                x_hat = fast_dividef(y, w_slice[None, :].to(tl.float32))
                m_offs = start_m + i * BLOCK_M1 + tl.arange(0, BLOCK_M1)
                m_mask = m_offs[:, None] < qlen
                dw_2d = tl.where(m_mask, dy * x_hat, 0.0)  # [M,D]
                # Wait for reduction warp to finish reading previous scratch
                dw_scratch_buf_id, dw_scratch_phase = _get_bufidx_phase(
                    accum_cnt_inner + i, NUM_BUFFERS_DW_SCRATCH
                )
                dw_scratch_tile = dw_scratch_buf[dw_scratch_buf_id]
                dw_scratch_empty = dw_scratch_empties[dw_scratch_buf_id]
                dw_scratch_full = dw_scratch_fulls[dw_scratch_buf_id]
                tlx.barrier_wait(dw_scratch_empty, dw_scratch_phase ^ 1)
                tlx.local_store(dw_scratch_tile, dw_2d.to(tlx.dtype_of(desc_q)))
                tlx.fence_async_shared()
                tlx.barrier_arrive(dw_scratch_full)

        # wait for qkT = tl.dot(k, qT)
        if ENABLE_PROTON and idx == PROTON_TILE:
            pl.enter_scope("act_wait_qk")
        tlx.barrier_wait(qk_full, tmem_phase)
        if ENABLE_PROTON and idx == PROTON_TILE:
            pl.exit_scope("act_wait_qk")

        if ENABLE_PROTON and idx == PROTON_TILE:
            pl.enter_scope("act_compute_ppT")
        pT = tlx.local_load(qk_tile)

        # only M masking is needed. N masking is done in dV's store
        pT = tl.where(offs_m[None, :] < qlen, pT, 0.0)
        # Autoregressive masking.
        if MASK:
            mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(mask, pT, 0.0)
        # Sliding window masking.
        if WINDOW_SIZE is not None:
            window_mask = tl.abs(offs_m[None, :] - offs_n[:, None]) <= WINDOW_SIZE
            pT = tl.where(window_mask, pT, 0.0)
        # Compute dV.
        dtype = tlx.dtype_of(desc_q)
        ppT = apply_activation(pT, dtype, activation_enum_int, NUM_ACT_SPLITS)
        # ppT *= qk_scale
        ppT = ppT.to(dtype)
        tlx.local_store(ppT_tile, ppT)
        tlx.barrier_arrive(ppT_full)
        if ENABLE_PROTON and idx == PROTON_TILE:
            pl.exit_scope("act_compute_ppT")

        ##
        pT = apply_activation_grad(pT, dtype, activation_enum_int)
        # pT *= qk_scale

        if IS_ALIGNED_KV:
            pT = tl.where(offs_m[None, :] < qlen, pT, 0.0)
        else:
            pT = tl.where((offs_m[None, :] < qlen) & (offs_n[:, None] < klen), pT, 0.0)
        # Autoregressive masking.
        if MASK:
            mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(mask, pT, 0.0)
        # Sliding window masking.
        if WINDOW_SIZE is not None:
            window_mask = tl.abs(offs_m[None, :] - offs_n[:, None]) <= WINDOW_SIZE
            pT = tl.where(window_mask, pT, 0.0)

        # Wait for dpT = tl.dot(v, tl.trans(do))
        if ENABLE_PROTON and idx == PROTON_TILE:
            pl.enter_scope("act_wait_dpT")
        tlx.barrier_wait(dpT_full, tmem_phase)
        if ENABLE_PROTON and idx == PROTON_TILE:
            pl.exit_scope("act_wait_dpT")

        if ENABLE_PROTON and idx == PROTON_TILE:
            pl.enter_scope("act_compute_dsT")
        dpT = tlx.local_load(dpT_tile)
        if FUSED_LAYERNORM:
            dpT_empty = dpT_empties[tmem_buf_id]
            tlx.fence_async_shared()
            tlx.barrier_arrive(dpT_empty)
        dsT = pT * dpT

        dsT = dsT.to(tlx.dtype_of(desc_q))
        tlx.local_store(dsT_tile, dsT)
        tlx.fence_async_shared()
        tlx.barrier_arrive(dsT_full)
        if ENABLE_PROTON and idx == PROTON_TILE:
            pl.exit_scope("act_compute_dsT")

        offs_m += BLOCK_M1

    slice_size: tl.constexpr = BLOCK_D // EPILOGUE_SUBTILE
    DKV_STORE_NCOL: tl.constexpr = BLOCK_D // 2
    DKV_STORE_ITERS: tl.constexpr = BLOCK_D // DKV_STORE_NCOL
    kv_buf_id, kv_phase = _get_bufidx_phase(accum_cnt_outer, NUM_BUFFERS_KV)

    if IS_ALIGNED_KV:
        # dV epilogue: TMEM → regs → SMEM staging (sdv_store_buf) → async TMA store.
        # sdv_store_buf reuses v_tiles SMEM (free after dv_fulls; MMA's last v_tiles
        # read — the dpT dot — is committed before dv_fulls).
        if ENABLE_PROTON and idx == PROTON_TILE:
            pl.enter_scope("act_wait_dv")
        tlx.barrier_wait(dv_full, dkv_phase)
        if ENABLE_PROTON and idx == PROTON_TILE:
            pl.exit_scope("act_wait_dv")
        if ENABLE_PROTON and idx == PROTON_TILE:
            pl.enter_scope("act_store_dv")
        for slice_id in tl.static_range(DKV_STORE_ITERS):
            dv_slice = tlx.subslice(dv_tile, DKV_STORE_NCOL * slice_id, DKV_STORE_NCOL)
            dv_r = tlx.local_load(dv_slice)
            tlx.async_descriptor_store_wait(0)
            tlx.local_store(sdv_store_buf[kv_buf_id], dv_r.to(DV.dtype.element_ty))
            tlx.fence("async_shared")
            tlx.async_descriptor_store(
                desc_dv,
                sdv_store_buf[kv_buf_id],
                [
                    (begin_k + start_n).to(tl.int32),
                    (off_h_kv * stride_kh).to(tl.int32) + slice_id * DKV_STORE_NCOL,
                ],
            )
        tlx.barrier_arrive(dv_empty)
        # Signal v_empties so load warp can refill v_tiles (aliased by sdv_store_buf).
        # async_descriptor_store_wait(0) ensures all staging stores complete first.
        tlx.async_descriptor_store_wait(0)
        v_empty_bar = v_empties[kv_buf_id]
        tlx.barrier_arrive(v_empty_bar)
        if ENABLE_PROTON and idx == PROTON_TILE:
            pl.exit_scope("act_store_dv")

        # dK epilogue: TMEM → regs → SMEM staging (sdk_store_buf) → async TMA store.
        # sdk_store_buf reuses k_tiles SMEM. MMA's dq dot still reads k_tiles after
        # dk_fulls, so we must wait on k_mma_done before writing sdk_store_buf.
        if ENABLE_PROTON and idx == PROTON_TILE:
            pl.enter_scope("act_wait_dk")
        tlx.barrier_wait(dk_full, dkv_phase)
        if ENABLE_PROTON and idx == PROTON_TILE:
            pl.exit_scope("act_wait_dk")
        # Wait for MMA's dq dot (last k_tiles read) before writing sdk_store_buf.
        k_mma_done_bar = k_mma_done[kv_buf_id]
        tlx.barrier_wait(k_mma_done_bar, kv_phase)
        if ENABLE_PROTON and idx == PROTON_TILE:
            pl.enter_scope("act_store_dk")
        for slice_id in tl.static_range(DKV_STORE_ITERS):
            dk_slice = tlx.subslice(dk_tile, DKV_STORE_NCOL * slice_id, DKV_STORE_NCOL)
            dk_r = tlx.local_load(dk_slice)
            tlx.async_descriptor_store_wait(0)
            tlx.local_store(sdk_store_buf[kv_buf_id], dk_r.to(DK.dtype.element_ty))
            tlx.fence("async_shared")
            tlx.async_descriptor_store(
                desc_dk,
                sdk_store_buf[kv_buf_id],
                [
                    (begin_k + start_n).to(tl.int32),
                    (off_h_kv * stride_kh).to(tl.int32) + slice_id * DKV_STORE_NCOL,
                ],
            )
        tlx.async_descriptor_store_wait(0)
        # All staging stores done + MMA done reading k_tiles →
        # safe for load warp to refill k_tiles (aliased by sdk_store_buf).
        k_empty_bar = k_empties[kv_buf_id]
        tlx.barrier_arrive(k_empty_bar)
        tlx.barrier_arrive(dk_empty)
        if ENABLE_PROTON and idx == PROTON_TILE:
            pl.exit_scope("act_store_dk")
    else:
        # Signal v_empties and k_empties so load warp can refill k/v SMEM tiles.
        # MMA no longer signals these (it signals k_mma_done instead of k_empty,
        # and v_empty commit was removed).  In the IS_ALIGNED_KV path above,
        # ACT signals them after SMEM staging; here we must do the same.
        v_empty_bar = v_empties[kv_buf_id]
        tlx.barrier_arrive(v_empty_bar)
        # Wait for MMA's dq dot (last k_tiles read) before releasing k_empties.
        k_mma_done_bar = k_mma_done[kv_buf_id]
        tlx.barrier_wait(k_mma_done_bar, kv_phase)
        k_empty_bar = k_empties[kv_buf_id]
        tlx.barrier_arrive(k_empty_bar)
        offs_k = tl.arange(0, slice_size)
        kmask = offs_n[:, None] < klen

        # Write back dV.
        dv_ptrs = DV + offs_n[:, None] * stride_km + offs_k[None, :] * stride_d
        tlx.barrier_wait(dv_full, dkv_phase)
        for slice_id in tl.static_range(EPILOGUE_SUBTILE):
            dv_slice = tlx.subslice(dv_tile, slice_size * slice_id, slice_size)
            dv_r = tlx.local_load(dv_slice)
            tl.store(dv_ptrs, dv_r.to(DV.dtype.element_ty), mask=kmask)
            dv_ptrs += slice_size * stride_d
        tlx.barrier_arrive(dv_empty)

        # Write back dK.
        dk_ptrs = DK + offs_n[:, None] * stride_km + offs_k[None, :] * stride_d
        tlx.barrier_wait(dk_full, dkv_phase)
        for slice_id in tl.static_range(EPILOGUE_SUBTILE):
            dk_slice = tlx.subslice(dk_tile, slice_size * slice_id, slice_size)
            dk_r = tlx.local_load(dk_slice)
            tl.store(dk_ptrs, dk_r.to(DK.dtype.element_ty), mask=kmask)
            dk_ptrs += slice_size * stride_d
        tlx.barrier_arrive(dk_empty)

    return num_steps


@triton.jit
def bwd_calculate_offsets(
    tile_idx,
    seq_index,
    H,
    G,
    n_tile_num,
    SORT_BY_SEQ_LENGTH: tl.constexpr,
    BROADCAST_Q: tl.constexpr,
    USE_I64_IDX: tl.constexpr,
    FUSED_RMS_NORM: tl.constexpr = False,
):
    if USE_I64_IDX:
        tile_idx = tile_idx.to(tl.int64)
    if FUSED_RMS_NORM:
        # H is fastest-varying: adjacent CTAs in cluster share (batch, kv_tile)
        off_h = tile_idx % H
        pid = (tile_idx // H) % n_tile_num
        off_z = (tile_idx // H) // n_tile_num
    else:
        off_hz = tile_idx // n_tile_num
        off_z = off_hz // H
        off_h = off_hz % H
        pid = tile_idx % n_tile_num

    if SORT_BY_SEQ_LENGTH:
        off_z = tl.load(seq_index + off_z)
    if BROADCAST_Q:
        off_q_z = 0
    else:
        off_q_z = off_z

    off_h_kv = off_h // G

    return off_z, off_h, off_h_kv, off_q_z, pid


def _bwd_host_descriptor_pre_hook(nargs: Dict[str, Any]) -> None:
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    BLOCK_M1 = nargs["BLOCK_M1"]
    BLOCK_N1 = nargs["BLOCK_N1"]
    BLOCK_D = nargs["BLOCK_D"]
    EPILOGUE_SUBTILE = nargs["EPILOGUE_SUBTILE"]
    LN_BWD_SUBTILE = nargs.get("LN_BWD_SUBTILE", EPILOGUE_SUBTILE)
    # When FUSED_LAYERNORM, desc_dq uses LN_BWD_SUBTILE for subtiling
    # (reduction warp uses LN_BWD_SUBTILE for both LN backward and non-LN dQ writeback)
    dq_subtile = LN_BWD_SUBTILE if nargs.get("FUSED_LAYERNORM") else EPILOGUE_SUBTILE
    nargs["desc_q"].block_shape = [BLOCK_M1, BLOCK_D]
    nargs["desc_v"].block_shape = [BLOCK_N1, BLOCK_D]
    nargs["desc_k"].block_shape = [BLOCK_N1, BLOCK_D]
    nargs["desc_do"].block_shape = [BLOCK_M1, BLOCK_D]
    nargs["desc_dq"].block_shape = [BLOCK_M1, BLOCK_D // dq_subtile]
    if (
        nargs.get("FUSED_RMS_NORM")
        and nargs.get("FUSED_RESIDUAL_ADD")
        and nargs.get("desc_do_raw") is not None
    ):
        nargs["desc_do_raw"].block_shape = [BLOCK_M1, BLOCK_D]
    if nargs["IS_ALIGNED_KV"]:
        nargs["desc_dk"].block_shape = [BLOCK_N1, BLOCK_D // 2]
        nargs["desc_dv"].block_shape = [BLOCK_N1, BLOCK_D // 2]
    if nargs.get("FUSED_RMS_NORM") and nargs.get("desc_o") is not None:
        nargs["desc_o"].block_shape = [BLOCK_M1, BLOCK_D]


def get_tlx_bwd_autotune_config(
    ctas_per_cga: Tuple[int, int, int] = (1, 1, 1),
    fused_rms_norm: bool = False,
) -> List[triton.Config]:
    if (
        AUTOTUNE_CONFIG_SET == "omnifm_v2"
        or os.environ.get("ADS_MKL_DISABLE_AUTOTUNE") == "1"
    ):
        if fused_rms_norm:
            return [
                triton.Config(
                    {
                        "BLOCK_M1": BM,
                        "BLOCK_N1": BN,
                        "NUM_BUFFERS_KV": buf_kv,
                        "NUM_BUFFERS_Q": buf_q,  # NUM_BUFFERS_Q has to be at least 2
                        "NUM_BUFFERS_DO": buf_do,
                        "NUM_BUFFERS_DS": buf_ds,
                        "NUM_BUFFERS_TMEM": buf_tem,
                        "NUM_ACT_SPLITS": 1,
                        "EPILOGUE_SUBTILE": epi,
                        "USE_CLC": True,
                        "NUM_REGS_ACT": ract,
                        "NUM_REGS_RED": rred,
                        "NUM_REGS_MMA": rmma,
                        "NUM_REGS_LOAD": rload,
                        "LN_BWD_SUBTILE": ln_subtile,
                    },
                    num_warps=w,
                    num_stages=1,
                    pre_hook=_bwd_host_descriptor_pre_hook,
                    ctas_per_cga=ctas_per_cga,
                )
                for BM in [64]
                for BN in [128]
                for epi in [4]
                for buf_kv in [1]
                for buf_q in [3]
                for buf_do in [2]
                for buf_ds in [1]
                for buf_tem in [1]
                for w in [8]
                for ract in [168]
                for rred in [80]
                for rmma in [48]
                for rload in [24]
                for ln_subtile in [4]
            ]
        else:
            return [
                triton.Config(
                    {
                        "BLOCK_M1": BM,
                        "BLOCK_N1": BN,
                        "NUM_BUFFERS_KV": buf_kv,
                        "NUM_BUFFERS_Q": buf_q,  # NUM_BUFFERS_Q has to be at least 2
                        "NUM_BUFFERS_DO": buf_do,
                        "NUM_BUFFERS_DS": buf_ds,
                        "NUM_BUFFERS_TMEM": buf_tem,
                        "NUM_ACT_SPLITS": 1,
                        "EPILOGUE_SUBTILE": epi,
                        "USE_CLC": True,
                        "NUM_REGS_ACT": ract,
                        "NUM_REGS_RED": rred,
                        "NUM_REGS_MMA": rmma,
                        "NUM_REGS_LOAD": rload,
                    },
                    num_warps=w,
                    num_stages=1,
                    pre_hook=_bwd_host_descriptor_pre_hook,
                    ctas_per_cga=ctas_per_cga,
                )
                for BM in [64]
                for BN in [128]
                for epi in [1]
                for buf_kv in [1]
                for buf_q in [3]
                for buf_do in [2]
                for buf_ds in [1]
                for buf_tem in [1]
                for w in [8]
                for ract in [256]
                for rred in [64]
                for rmma in [24]
                for rload in [48]
            ]
    return [
        triton.Config(
            {
                "BLOCK_M1": BM,
                "BLOCK_N1": BN,
                "NUM_BUFFERS_KV": buf_kv,
                "NUM_BUFFERS_Q": buf_q,  # NUM_BUFFERS_Q has to be at least 2
                "NUM_BUFFERS_DO": buf_do,
                "NUM_BUFFERS_DS": buf_ds,
                "NUM_BUFFERS_TMEM": buf_tem,
                "NUM_ACT_SPLITS": 1,
                "EPILOGUE_SUBTILE": epi,
                "USE_CLC": True,
                "NUM_REGS_ACT": ract,
                "NUM_REGS_RED": rred,
                "NUM_REGS_MMA": rmma,
                "NUM_REGS_LOAD": rload,
                "LN_BWD_SUBTILE": ln_subtile,
            },
            num_warps=w,
            num_stages=1,
            pre_hook=_bwd_host_descriptor_pre_hook,
            ctas_per_cga=ctas_per_cga,
        )
        for BM in [64]
        for BN in [128]
        for epi in [1, 2, 4]
        for buf_kv in [1]
        for buf_q in [3]
        for buf_do in [2]
        for buf_ds in [1]
        for buf_tem in [1]
        for w in [8]
        for ract in [168, 192, 256]
        for rred in [80, 108, 128, 144]
        for rmma in [24, 48]
        for rload in [24]
        for ln_subtile in [1, 2, 4]
    ]


@lru_cache
def _get_autotune_kernel_backward(
    kernel: JITFunction,
    ctas_per_cga: Tuple[int, int, int] = (1, 1, 1),
    has_drms_norm_weight: bool = False,
    fused_rms_norm: bool = False,
    has_dlayernorm_weight: bool = False,
    has_dlayernorm_bias: bool = False,
) -> JITFunction:
    restore = ["DQ"]
    if has_drms_norm_weight:
        restore.append("drms_norm_weight_out")
    if has_dlayernorm_weight:
        restore.append("dlayernorm_weight_out")
    if has_dlayernorm_bias:
        restore.append("dlayernorm_bias_out")
    return triton.autotune(
        configs=get_tlx_bwd_autotune_config(
            ctas_per_cga=ctas_per_cga, fused_rms_norm=fused_rms_norm
        ),
        key=[
            "N_CTX",
            "N_CTX_KV",
            "HEAD_DIM",
            "H",
            "G",
            "FUSED_QKV",
            "FUSED_KV",
            "FUSED_RMS_NORM",
            "FUSED_Q_RESIDUAL_ADD",
            "FUSED_RESIDUAL_ADD",
            "FUSED_LAYERNORM",
        ],
        restore_value=restore,
    )(kernel)


@triton.jit
def gdpa_backward_tlx(
    ensemble_activation_list: "VAR_ARGS_ARRAY",
    desc_q: Any,
    Q_offsets: Any,
    desc_k: Any,
    K_offsets: Any,
    desc_v: Any,
    seq_index: Any,  #
    desc_do: Any,  #
    Out_offsets: Any,
    desc_dq: Any,
    desc_do_raw: Any,
    DQ: Any,
    desc_dk,
    DK: Any,
    desc_dv,
    DV: Any,  #
    stride_qm: Any,
    stride_km: Any,
    stride_qh: Any,
    stride_kh: Any,
    stride_d: Any,
    stride_dom: Any,
    stride_doh: Any,
    Z: Any,
    H: Any,
    G: Any,
    N_CTX: Any,  #
    N_CTX_KV: Any,  #
    qk_scale: Any,  #
    FUSED_QKV: tl.constexpr,  #
    FUSED_KV: tl.constexpr,  #
    SORT_BY_SEQ_LENGTH: tl.constexpr,  #
    BLOCK_D: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    USE_START_END_OFFSETS: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    BROADCAST_Q: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_DO: tl.constexpr,
    NUM_BUFFERS_DS: tl.constexpr,
    NUM_BUFFERS_TMEM: tl.constexpr,
    NUM_ACT_SPLITS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    IS_DENSE_KV: tl.constexpr,
    IS_ALIGNED_KV: tl.constexpr,
    activation_enum_int: tl.constexpr,
    USE_I64_IDX: tl.constexpr,
    USE_CLC: tl.constexpr,
    desc_o: Any = None,
    rrms_out: Any = None,
    rms_norm_weight: Any = None,
    drms_norm_weight_out: Any = None,
    stride_oh: Any = 0,
    FUSED_RMS_NORM: tl.constexpr = False,
    HAS_RMS_NORM_WEIGHT: tl.constexpr = False,
    NUM_REDUCTION_CTAS: tl.constexpr = 0,
    NUM_REDUCTION_BUFS: tl.constexpr = 2,
    FUSED_Q_RESIDUAL_ADD: tl.constexpr = False,
    FUSED_RESIDUAL_ADD: tl.constexpr = False,
    NUM_BUFFERS_O: tl.constexpr = 1,
    NUM_REGS_ACT: tl.constexpr = 256,
    NUM_REGS_RED: tl.constexpr = 64,
    NUM_REGS_MMA: tl.constexpr = 24,
    NUM_REGS_LOAD: tl.constexpr = 48,
    FUSED_LAYERNORM: tl.constexpr = False,
    layernorm_weight: Any = None,
    layernorm_bias: Any = None,
    HAS_LAYERNORM_WEIGHT: tl.constexpr = False,
    HAS_LAYERNORM_BIAS: tl.constexpr = False,
    # LN backward kernel params
    Q_raw: Any = None,
    ln_mean_out: Any = None,
    ln_rstd_out: Any = None,
    dlayernorm_weight_out: Any = None,
    dlayernorm_bias_out: Any = None,
    LN_BWD_SUBTILE: tl.constexpr = 1,
    ENABLE_PROTON: tl.constexpr = False,
    PROTON_TILE: tl.constexpr = 0,
) -> None:
    n_tile_num = tl.cdiv(N_CTX_KV, BLOCK_N1)
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)

    total_tiles = n_tile_num * Z * H

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1

    tile_idx = prog_id

    # allocate smem buffers
    k_tiles = tlx.local_alloc((BLOCK_N1, BLOCK_D), tlx.dtype_of(desc_k), NUM_BUFFERS_KV)
    v_tiles = tlx.local_alloc((BLOCK_N1, BLOCK_D), tlx.dtype_of(desc_v), NUM_BUFFERS_KV)
    q_tiles = tlx.local_alloc((BLOCK_M1, BLOCK_D), tlx.dtype_of(desc_q), NUM_BUFFERS_Q)
    do_tiles = tlx.local_alloc(
        (BLOCK_M1, BLOCK_D), tlx.dtype_of(desc_do), NUM_BUFFERS_DO
    )
    dsT_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1), tlx.dtype_of(desc_q), NUM_BUFFERS_DS
    )

    # SMEM staging buffers for dKV epilogue TMA stores.
    # Use BLOCK_D // 2 columns (= 128 bytes for bf16), matching swizzle-128.
    # sdv reuses v_tiles (free after dv_fulls; MMA's last v_tiles read —
    # the dpT dot — precedes dv_fulls).
    # sdk reuses k_tiles (MMA's dq dot still reads k_tiles after dk_fulls,
    # so the ACT task must wait on k_mma_done before writing sdk).
    DKV_STORE_NCOL: tl.constexpr = BLOCK_D // 2
    if IS_ALIGNED_KV:
        sdv_store_buf = tlx.local_alloc(
            (BLOCK_N1, DKV_STORE_NCOL),
            tlx.dtype_of(desc_dv),
            NUM_BUFFERS_KV,
            reuse=v_tiles if HEAD_DIM == 128 else None,
        )
        sdk_store_buf = tlx.local_alloc(
            (BLOCK_N1, DKV_STORE_NCOL),
            tlx.dtype_of(desc_dk),
            NUM_BUFFERS_KV,
            reuse=k_tiles if HEAD_DIM == 128 else None,
        )
    else:
        sdv_store_buf = None
        sdk_store_buf = None

    # allocate barriers for smem buffers
    k_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    v_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    k_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    v_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    # k_mma_done: signaled by MMA after dq dot (last k_tiles read).
    # ACT waits on this before writing sdk_store_buf which aliases k_tiles.
    k_mma_done = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
    do_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
    dsT_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DS)
    q_empties = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS_Q,
        arrive_count=2 if FUSED_LAYERNORM else 1,
    )
    if FUSED_RMS_NORM:
        # When FUSED_RMS_NORM, do_tiles keeps raw do (never overwritten).
        # Arrivers: ACT (after reading raw do) + RED (if FUSED_RESIDUAL_ADD).
        do_empties = tlx.alloc_barriers(
            num_barriers=NUM_BUFFERS_DO,
            arrive_count=2 if FUSED_RESIDUAL_ADD else 1,
        )
    else:
        # Without FUSED_RMS_NORM, do_tiles = raw do, consumed by MMA + RED.
        do_empties = tlx.alloc_barriers(
            num_barriers=NUM_BUFFERS_DO,
            arrive_count=2 if FUSED_Q_RESIDUAL_ADD else 1,
        )

    # Fused RMS norm backward: allocate o_tiles, barriers, and DSMEM reduction buffers
    if FUSED_RMS_NORM:
        # Separate buffer for post-RMSNorm-backward do (do_gdpa).
        # ACT writes here instead of overwriting do_tiles, so reduction warp
        # can read raw do from do_tiles for FUSED_RESIDUAL_ADD.
        NUM_BUFFERS_DO_RMS: tl.constexpr = 2
        do_rms_tiles = tlx.local_alloc(
            (BLOCK_M1, BLOCK_D), tlx.dtype_of(desc_do), NUM_BUFFERS_DO_RMS
        )
        # do_rms_empties: MMA (dv dot) + RED (if Q_RESIDUAL_ADD) arrive.
        do_rms_empties = tlx.alloc_barriers(
            num_barriers=NUM_BUFFERS_DO_RMS,
            arrive_count=2 if FUSED_Q_RESIDUAL_ADD else 1,
        )
        o_tiles = tlx.local_alloc(
            (BLOCK_M1, BLOCK_D), tlx.dtype_of(desc_o), NUM_BUFFERS_O
        )
        o_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O)
        o_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_O)
        do_raw_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DO)
        reduction_barriers = tlx.alloc_barriers(num_barriers=NUM_REDUCTION_BUFS)
        cross_cta_reduction_expected_bytes: tl.constexpr = (
            BLOCK_M1 * 4 * (NUM_REDUCTION_CTAS - 1)  # float32 from each remote CTA
        )
        reduction_buf = tlx.local_alloc(
            (BLOCK_M1, 1), tl.float32, (NUM_REDUCTION_CTAS - 1) * NUM_REDUCTION_BUFS
        )

    # Fused LayerNorm backward: ACT normalizes raw Q using precomputed mean/rstd
    if FUSED_LAYERNORM:
        q_norm_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_Q)
        # SMEM for LN backward DSMEM reduction (c1 + c2 exchange)
        ln_reduction_buf = tlx.local_alloc(
            (BLOCK_M1, 1), tl.float32, 2 * (NUM_REDUCTION_CTAS - 1) * NUM_REDUCTION_BUFS
        )
        ln_bwd_barriers = tlx.alloc_barriers(num_barriers=NUM_REDUCTION_BUFS)
        ln_bwd_buf = ln_reduction_buf
        ln_bwd_expected_bytes: tl.constexpr = (
            BLOCK_M1 * 4 * (NUM_REDUCTION_CTAS - 1) * 2  # c1 + c2
        )
    else:
        q_norm_fulls = q_fulls

    # SMEM accumulator for drms_norm_weight across persistent loop (always allocate)
    dw_accum_buf = tlx.local_alloc((1, BLOCK_D), tl.float32, 1)
    # SMEM accumulators for dlayernorm_weight/bias across persistent loop
    # Use LN_BWD_SUBTILE separate buffers (one per subtile) to avoid local_slice
    LN_BWD_SLICE: tl.constexpr = BLOCK_D // LN_BWD_SUBTILE
    if FUSED_LAYERNORM and HAS_LAYERNORM_WEIGHT:
        dlw_accum_buf = tlx.local_alloc((1, LN_BWD_SLICE), tl.float32, LN_BWD_SUBTILE)
    if FUSED_LAYERNORM and HAS_LAYERNORM_BIAS:
        dlb_accum_buf = tlx.local_alloc((1, LN_BWD_SLICE), tl.float32, LN_BWD_SUBTILE)
    # SMEM cache for LN weight/bias (layernorm per head, constant per CTA with clustering)
    if FUSED_LAYERNORM and HAS_LAYERNORM_WEIGHT:
        ln_w_buf = tlx.local_alloc((1, BLOCK_D), tl.float32, 1)
    if FUSED_LAYERNORM and HAS_LAYERNORM_BIAS:
        ln_b_buf = tlx.local_alloc((1, BLOCK_D), tl.float32, 1)
    # SMEM cache for w_slice (rms_norm_weight per head, constant per CTA)
    if HAS_RMS_NORM_WEIGHT:
        w_slice_buf = tlx.local_alloc((1, BLOCK_D), tlx.dtype_of(rms_norm_weight), 1)
        # SMEM scratch for dw intermediate — double-buffered, with barriers for
        # activation→reduction warp handoff (tl.sum deferred to reduction warp)
        NUM_BUFFERS_DW_SCRATCH: tl.constexpr = 2
        dw_scratch_buf = tlx.local_alloc(
            (BLOCK_M1, BLOCK_D), tlx.dtype_of(desc_do), NUM_BUFFERS_DW_SCRATCH
        )
        dw_scratch_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DW_SCRATCH)
        dw_scratch_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_DW_SCRATCH)

    # allocate tmem buffers
    qk_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1), tl.float32, NUM_BUFFERS_TMEM, tlx.storage_kind.tmem
    )
    ppT_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1),
        tlx.dtype_of(desc_do),
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )
    dv_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_D), tl.float32, NUM_BUFFERS_TMEM, tlx.storage_kind.tmem
    )
    dk_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_D), tl.float32, NUM_BUFFERS_TMEM, tlx.storage_kind.tmem
    )
    dq_tiles = tlx.local_alloc(
        (BLOCK_M1, BLOCK_D), tl.float32, NUM_BUFFERS_TMEM, tlx.storage_kind.tmem
    )
    dpT_tiles = tlx.local_alloc(
        (BLOCK_N1, BLOCK_M1),
        tl.float32,
        NUM_BUFFERS_TMEM,
        tlx.storage_kind.tmem,
        reuse=dq_tiles if not FUSED_LAYERNORM else None,
    )

    # allocate barriers for tmem buffers
    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    ppT_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dpT_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dk_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dv_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dk_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dq_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    ppT_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    dq_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)
    if FUSED_LAYERNORM:
        dpT_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_TMEM)

    # SMEM staging buffer for async TMA reduce-add of dQ (double-buffered).
    # Replaces desc_dq.atomic_add() which hides internal SMEM scratch and
    # does not support pipelining.  With explicit staging, slice N's TMA
    # overlaps with slice N+1's TMEM->SMEM copy.
    if FUSED_LAYERNORM:
        DQ_REDUCE_NCOL: tl.constexpr = BLOCK_D // LN_BWD_SUBTILE
    else:
        DQ_REDUCE_NCOL: tl.constexpr = BLOCK_D // EPILOGUE_SUBTILE
    DQ_REDUCE_STAGES: tl.constexpr = 2 if LN_BWD_SUBTILE >= 4 else 1
    dq_store_buf = tlx.local_alloc(
        (BLOCK_M1, DQ_REDUCE_NCOL), tlx.dtype_of(desc_dq), DQ_REDUCE_STAGES
    )

    if USE_CLC:
        clc_context = tlx.clc_create_context(4)

    with tlx.async_tasks():
        # activation
        with tlx.async_task("default"):
            accum_cnt_inner = 0
            accum_cnt_outer = 0
            clc_phase_consumer = 0
            clc_phase_producer = 1
            _clc_buf = 0  # noqa: F841
            idx = 0
            has_more_tile = True

            # Initialize SMEM accumulator for drms_norm_weight to zeros
            if FUSED_RMS_NORM and HAS_RMS_NORM_WEIGHT:
                dw_zeros = tl.zeros([1, BLOCK_D], dtype=tl.float32)
                tlx.local_store(dw_accum_buf[0], dw_zeros)
                # Cache w_slice in SMEM (constant per CTA, off_h tied to cluster rank)
                w_off_h = tlx.cluster_cta_rank()
                w_offs_init = w_off_h * HEAD_DIM + tl.arange(0, BLOCK_D)
                w_data = tl.load(rms_norm_weight + w_offs_init)
                tlx.local_store(w_slice_buf[0], w_data[None, :])

            # Cache LN weight/bias in SMEM (constant per CTA with clustering)
            if FUSED_LAYERNORM and HAS_LAYERNORM_WEIGHT:
                ln_w_off_h = tlx.cluster_cta_rank()
                ln_w_offs_init = ln_w_off_h * HEAD_DIM + tl.arange(0, BLOCK_D)
                ln_w_data = tl.load(layernorm_weight + ln_w_offs_init).to(tl.float32)
                tlx.local_store(ln_w_buf[0], ln_w_data[None, :])
                if HAS_LAYERNORM_BIAS:
                    ln_b_data = tl.load(layernorm_bias + ln_w_offs_init).to(tl.float32)
                    tlx.local_store(ln_b_buf[0], ln_b_data[None, :])

            while has_more_tile:
                off_z, off_h, off_h_kv, off_q_z, pid = bwd_calculate_offsets(
                    tile_idx,
                    seq_index,
                    H,
                    G,
                    n_tile_num,
                    SORT_BY_SEQ_LENGTH,
                    BROADCAST_Q,
                    USE_I64_IDX,
                    FUSED_RMS_NORM,
                )

                begin_q = tl.load(Q_offsets + off_q_z)
                end_q = tl.load(Q_offsets + off_q_z + 1)
                start_n = pid * BLOCK_N1
                start_m = 0
                off_h2 = off_h.to(tl.int64)
                qlen = end_q - begin_q
                if FUSED_QKV:
                    begin_k = begin_q
                    end_k = end_q
                    klen = qlen
                else:
                    begin_k = tl.load(K_offsets + off_z)
                    end_k = tl.load(K_offsets + off_z + 1)
                    klen = end_k - begin_k

                if start_n < klen:
                    cur_act_enum = activation_enum_int
                    for i in range(len(ensemble_activation_list)):
                        if off_h == i:
                            cur_act_enum = ensemble_activation_list[i]

                    kv_offset = off_h_kv.to(tl.int64) * stride_kh + begin_k * stride_km
                    num_steps = _gdpa_bwd_tlx_compute_activation(
                        desc_q=desc_q,
                        desc_dk=desc_dk,
                        DK=DK + kv_offset,
                        desc_dv=desc_dv,
                        DV=DV + kv_offset,
                        k_tiles=k_tiles,
                        v_tiles=v_tiles,
                        qk_tiles=qk_tiles,
                        dpT_tiles=dpT_tiles,
                        ppT_tiles=ppT_tiles,
                        dsT_tiles=dsT_tiles,
                        dv_tiles=dv_tiles,
                        dk_tiles=dk_tiles,
                        qk_fulls=qk_fulls,
                        dpT_fulls=dpT_fulls,
                        ppT_fulls=ppT_fulls,
                        dsT_fulls=dsT_fulls,
                        dv_fulls=dv_fulls,
                        dk_fulls=dk_fulls,
                        dv_empties=dv_empties,
                        dk_empties=dk_empties,
                        off_h_kv=off_h_kv,
                        begin_k=begin_k,
                        accum_cnt_outer=accum_cnt_outer,
                        accum_cnt_inner=accum_cnt_inner,
                        stride_km=stride_km,
                        stride_kh=stride_kh,
                        stride_d=stride_d,
                        start_n=start_n,
                        start_m=start_m,
                        qlen=qlen,
                        klen=klen,
                        HEAD_DIM=HEAD_DIM,
                        BLOCK_D=BLOCK_D,
                        BLOCK_M1=BLOCK_M1,
                        BLOCK_N1=BLOCK_N1,
                        MASK=False,
                        WINDOW_SIZE=WINDOW_SIZE,
                        NUM_BUFFERS_TMEM=NUM_BUFFERS_TMEM,
                        NUM_ACT_SPLITS=NUM_ACT_SPLITS,
                        activation_enum_int=cur_act_enum,
                        EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
                        IS_ALIGNED_KV=IS_ALIGNED_KV,
                        # Fused RMS norm backward params
                        do_tiles=do_tiles,
                        do_fulls=do_fulls,
                        do_raw_fulls=do_raw_fulls if FUSED_RMS_NORM else None,
                        do_empties=do_empties,
                        do_rms_tiles=do_rms_tiles if FUSED_RMS_NORM else None,
                        do_rms_empties=do_rms_empties if FUSED_RMS_NORM else None,
                        o_tiles=o_tiles if FUSED_RMS_NORM else None,
                        o_fulls=o_fulls if FUSED_RMS_NORM else None,
                        o_empties=o_empties if FUSED_RMS_NORM else None,
                        rrms_out=rrms_out,
                        rms_norm_weight=rms_norm_weight,
                        reduction_buf=reduction_buf if FUSED_RMS_NORM else None,
                        reduction_barriers=reduction_barriers
                        if FUSED_RMS_NORM
                        else None,
                        begin_q=begin_q,
                        off_h=off_h,
                        stride_oh=stride_oh,
                        NUM_BUFFERS_DO=NUM_BUFFERS_DO,
                        NUM_BUFFERS_DO_RMS=NUM_BUFFERS_DO_RMS if FUSED_RMS_NORM else 1,
                        NUM_BUFFERS_O=NUM_BUFFERS_O,
                        FUSED_RMS_NORM=FUSED_RMS_NORM,
                        HAS_RMS_NORM_WEIGHT=HAS_RMS_NORM_WEIGHT,
                        NUM_REDUCTION_CTAS=NUM_REDUCTION_CTAS,
                        NUM_REDUCTION_BUFS=NUM_REDUCTION_BUFS,
                        N_RMS=H * HEAD_DIM if FUSED_RMS_NORM else 0,
                        CROSS_CTA_REDUCTION_EXPECTED_BYTES=cross_cta_reduction_expected_bytes
                        if FUSED_RMS_NORM
                        else 0,
                        dw_accum_buf=dw_accum_buf,
                        w_slice_buf=w_slice_buf if HAS_RMS_NORM_WEIGHT else None,
                        dw_scratch_buf=dw_scratch_buf if HAS_RMS_NORM_WEIGHT else None,
                        dw_scratch_fulls=dw_scratch_fulls
                        if HAS_RMS_NORM_WEIGHT
                        else None,
                        dw_scratch_empties=dw_scratch_empties
                        if HAS_RMS_NORM_WEIGHT
                        else None,
                        NUM_BUFFERS_DW_SCRATCH=NUM_BUFFERS_DW_SCRATCH
                        if HAS_RMS_NORM_WEIGHT
                        else 1,
                        FUSED_Q_RESIDUAL_ADD=FUSED_Q_RESIDUAL_ADD,
                        # Fused LayerNorm params
                        FUSED_LAYERNORM=FUSED_LAYERNORM,
                        q_tiles=q_tiles if FUSED_LAYERNORM else None,
                        q_fulls=q_fulls if FUSED_LAYERNORM else None,
                        q_norm_fulls=q_norm_fulls if FUSED_LAYERNORM else None,
                        NUM_BUFFERS_Q=NUM_BUFFERS_Q,
                        layernorm_weight_ptr=layernorm_weight
                        if FUSED_LAYERNORM and HAS_LAYERNORM_WEIGHT
                        else None,
                        layernorm_bias_ptr=layernorm_bias
                        if FUSED_LAYERNORM and HAS_LAYERNORM_BIAS
                        else None,
                        ln_w_buf=ln_w_buf
                        if FUSED_LAYERNORM and HAS_LAYERNORM_WEIGHT
                        else None,
                        ln_b_buf=ln_b_buf
                        if FUSED_LAYERNORM and HAS_LAYERNORM_BIAS
                        else None,
                        HAS_LAYERNORM_WEIGHT=HAS_LAYERNORM_WEIGHT,
                        HAS_LAYERNORM_BIAS=HAS_LAYERNORM_BIAS,
                        ln_mean_out=ln_mean_out if FUSED_LAYERNORM else None,
                        ln_rstd_out=ln_rstd_out if FUSED_LAYERNORM else None,
                        dpT_empties=dpT_empties if FUSED_LAYERNORM else None,
                        ENABLE_PROTON=ENABLE_PROTON,
                        PROTON_TILE=PROTON_TILE,
                        idx=idx,
                        # dKV SMEM staging params
                        sdv_store_buf=sdv_store_buf,
                        sdk_store_buf=sdk_store_buf,
                        k_mma_done=k_mma_done,
                        v_empties=v_empties,
                        k_empties=k_empties,
                        NUM_BUFFERS_KV=NUM_BUFFERS_KV,
                    )
                    accum_cnt_inner += num_steps
                    accum_cnt_outer += 1

                if USE_CLC:
                    tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx != -1
                    idx += 1
                else:
                    tile_idx += num_progs
                    idx += 1
                    has_more_tile = idx < tiles_per_sm

            # dw flush moved to reduction warp (which now owns dw_accum_buf)

        # reduction — increase registers when doing LN backward in this warp
        with tlx.async_task(num_warps=4, registers=NUM_REGS_RED):
            accum_cnt_inner = 0
            accum_cnt_outer = 0
            clc_phase_consumer = 0
            clc_phase_producer = 1
            _clc_buf = 0  # noqa: F841
            idx = 0
            has_more_tile = True

            # Initialize SMEM accumulators for dlayernorm_weight/bias to zeros
            if FUSED_LAYERNORM and HAS_LAYERNORM_WEIGHT:
                dlw_zeros = tl.zeros([1, LN_BWD_SLICE], dtype=tl.float32)
                for _init_i in tl.static_range(LN_BWD_SUBTILE):
                    tlx.local_store(dlw_accum_buf[_init_i], dlw_zeros)
            if FUSED_LAYERNORM and HAS_LAYERNORM_BIAS:
                dlb_zeros = tl.zeros([1, LN_BWD_SLICE], dtype=tl.float32)
                for _init_i in tl.static_range(LN_BWD_SUBTILE):
                    tlx.local_store(dlb_accum_buf[_init_i], dlb_zeros)

            while has_more_tile:
                off_z, off_h, off_h_kv, off_q_z, pid = bwd_calculate_offsets(
                    tile_idx,
                    seq_index,
                    H,
                    G,
                    n_tile_num,
                    SORT_BY_SEQ_LENGTH,
                    BROADCAST_Q,
                    USE_I64_IDX,
                    FUSED_RMS_NORM,
                )

                begin_q = tl.load(Q_offsets + off_q_z)
                end_q = tl.load(Q_offsets + off_q_z + 1)
                start_n = pid * BLOCK_N1
                start_m = 0
                off_h2 = off_h.to(tl.int64)
                qlen = end_q - begin_q
                if FUSED_RMS_NORM and FUSED_RESIDUAL_ADD:
                    do_raw_desc = tl.make_tensor_descriptor(
                        DQ,
                        shape=[end_q.to(tl.int32), (HEAD_DIM * H).to(tl.int32)],
                        strides=[HEAD_DIM * H, 1],
                        block_shape=[BLOCK_M1, BLOCK_D],
                    )
                if FUSED_QKV:
                    begin_k = begin_q
                    end_k = end_q
                    klen = qlen
                else:
                    begin_k = tl.load(K_offsets + off_z)
                    end_k = tl.load(K_offsets + off_z + 1)
                    klen = end_k - begin_k

                if start_n < klen:
                    num_steps, start_m = _gdpa_bwd_tlx_compute_num_steps(
                        qlen, start_n, BLOCK_M1, BLOCK_N1, WINDOW_SIZE
                    )
                    curr_m = start_m
                    step_m = BLOCK_M1

                    for i in tl.range(0, num_steps, 1, loop_unroll_factor=1):
                        tmem_buf_id, tmem_phase = _get_bufidx_phase(
                            accum_cnt_inner + i, NUM_BUFFERS_TMEM
                        )
                        dq_tile = dq_tiles[tmem_buf_id]
                        dq_full = dq_fulls[tmem_buf_id]
                        dq_empty = dq_empties[tmem_buf_id]

                        # Early wait for raw do: store to dq via TMA and
                        # release do_tiles BEFORE waiting on dq_full, so
                        # the TMA store overlaps with MMA's dq compute.
                        if FUSED_RMS_NORM and FUSED_RESIDUAL_ADD:
                            do_buf_id_raw, do_phase_raw = _get_bufidx_phase(
                                accum_cnt_inner + i, NUM_BUFFERS_DO
                            )
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("red_early_do_raw_wait")
                            tlx.barrier_wait(do_raw_fulls[do_buf_id_raw], do_phase_raw)
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("red_early_do_raw_wait")
                            if start_n == 0:
                                if ENABLE_PROTON and idx == PROTON_TILE:
                                    pl.enter_scope("red_do_raw_store")
                                tlx.async_descriptor_store(
                                    do_raw_desc,
                                    do_tiles[do_buf_id_raw],
                                    [
                                        (begin_q + curr_m).to(tl.int32),
                                        (off_h2 * stride_qh).to(tl.int32),
                                    ],
                                    store_reduce="add",
                                )
                                if ENABLE_PROTON and idx == PROTON_TILE:
                                    pl.exit_scope("red_do_raw_store")

                        # wait for dq = tl.dot(tl.trans(dsT), k)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("red_wait_dq")
                        tlx.barrier_wait(dq_full, tmem_phase)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("red_wait_dq")

                        # Set up do buffers for reduction warp:
                        # - Q_RESIDUAL_ADD needs do_gdpa (post-RMS-bwd) from do_rms_tiles
                        # - RESIDUAL_ADD needs raw do from do_tiles
                        if FUSED_Q_RESIDUAL_ADD or FUSED_RESIDUAL_ADD:
                            do_buf_id, do_phase = _get_bufidx_phase(
                                accum_cnt_inner + i, NUM_BUFFERS_DO
                            )
                        if FUSED_Q_RESIDUAL_ADD:
                            # do_gdpa: in do_rms_tiles when RMS_NORM, else do_tiles
                            if FUSED_RMS_NORM:
                                do_rms_buf_id, _ = _get_bufidx_phase(
                                    accum_cnt_inner + i, NUM_BUFFERS_DO_RMS
                                )
                                do_tile = do_rms_tiles[do_rms_buf_id]
                                do_empty = do_rms_empties[do_rms_buf_id]
                            else:
                                do_tile = do_tiles[do_buf_id]
                                do_empty = do_empties[do_buf_id]
                            do_full = do_fulls[do_buf_id]
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("red_wait_do")
                            tlx.barrier_wait(do_full, do_phase)
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("red_wait_do")
                        if FUSED_RESIDUAL_ADD and not FUSED_RMS_NORM:
                            # Fallback for non-RMS_NORM: read raw do here.
                            # When FUSED_RMS_NORM, handled by early do_raw
                            # store above.
                            do_raw_tile = do_tiles[do_buf_id]
                            do_raw_empty = do_empties[do_buf_id]
                            if not FUSED_Q_RESIDUAL_ADD:
                                do_raw_full_bar = do_fulls[do_buf_id]
                                tlx.barrier_wait(do_raw_full_bar, do_phase)

                        slice_size: tl.constexpr = BLOCK_D // EPILOGUE_SUBTILE
                        ln_bwd_slice: tl.constexpr = BLOCK_D // LN_BWD_SUBTILE

                        # Get q_tile buffer for reading LN(Q) (FUSED_LAYERNORM only)
                        if FUSED_LAYERNORM:
                            q_buf_id_red, q_phase_red = _get_bufidx_phase(
                                accum_cnt_inner + i, NUM_BUFFERS_Q
                            )
                            q_tile_red = q_tiles[q_buf_id_red]
                            q_empty_red = q_empties[q_buf_id_red]

                        if FUSED_LAYERNORM and start_n == 0:
                            # Two-pass LN backward in Reduction warp
                            # Load mean, rstd for this M-tile from global memory
                            q_offs_m = begin_q + curr_m + tl.arange(0, BLOCK_M1)
                            q_mask = q_offs_m < (begin_q + qlen)
                            _mean_vals = tl.load(  # noqa: F841
                                ln_mean_out + q_offs_m, mask=q_mask, other=0.0
                            )[:, None]
                            rstd_vals = tl.load(
                                ln_rstd_out + q_offs_m, mask=q_mask, other=1.0
                            )[:, None]

                            # Pass 1: accumulate c1, c2 across subtiles
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("red_ln_pass1")
                            c1_accum = tl.zeros((BLOCK_M1, 1), dtype=tl.float32)
                            c2_accum = tl.zeros((BLOCK_M1, 1), dtype=tl.float32)
                            for slice_id in tl.static_range(LN_BWD_SUBTILE):
                                dq_slice = tlx.subslice(
                                    dq_tile, ln_bwd_slice * slice_id, ln_bwd_slice
                                )
                                dq_r = tlx.local_load(dq_slice)
                                if FUSED_Q_RESIDUAL_ADD:
                                    do_slice = tlx.local_slice(
                                        do_tile,
                                        [0, ln_bwd_slice * slice_id],
                                        [BLOCK_M1, ln_bwd_slice],
                                    )
                                    do_r = tlx.local_load(do_slice).to(tl.float32)
                                    m_offs = curr_m + tl.arange(0, BLOCK_M1)
                                    do_r = tl.where(m_offs[:, None] < qlen, do_r, 0.0)
                                    dq_r = _add_f32x2(dq_r, do_r)
                                # Read LN(Q) subtile from q_tile, derive x_hat
                                ln_q_sub = tlx.local_load(
                                    tlx.local_slice(
                                        q_tile_red,
                                        [0, ln_bwd_slice * slice_id],
                                        [BLOCK_M1, ln_bwd_slice],
                                    )
                                ).to(tl.float32)
                                if layernorm_weight is not None:
                                    ln_w_sub = tlx.local_load(
                                        tlx.local_slice(
                                            ln_w_buf[0],
                                            [0, ln_bwd_slice * slice_id],
                                            [1, ln_bwd_slice],
                                        )
                                    )  # (1, ln_bwd_slice) f32
                                    if HAS_LAYERNORM_BIAS:
                                        ln_b_sub = tlx.local_load(
                                            tlx.local_slice(
                                                ln_b_buf[0],
                                                [0, ln_bwd_slice * slice_id],
                                                [1, ln_bwd_slice],
                                            )
                                        )  # (1, ln_bwd_slice) f32
                                        x_hat_sub = fast_dividef(
                                            ln_q_sub - ln_b_sub,
                                            ln_w_sub.to(tl.float32),
                                        )
                                    else:
                                        x_hat_sub = fast_dividef(
                                            ln_q_sub, ln_w_sub.to(tl.float32)
                                        )
                                    dq_w_sub = dq_r * ln_w_sub
                                else:
                                    x_hat_sub = ln_q_sub
                                    dq_w_sub = dq_r
                                c1_accum += tl.sum(dq_w_sub, axis=1, keep_dims=True)
                                c2_accum += tl.sum(
                                    _mul_f32x2(dq_w_sub, x_hat_sub),
                                    axis=1,
                                    keep_dims=True,
                                )
                                # Accumulate dlw/dlb per subtile into SMEM accumulator
                                if layernorm_weight is not None:
                                    m_mask_dlw = (curr_m + tl.arange(0, BLOCK_M1))[
                                        :, None
                                    ] < qlen
                                    dlw_2d = tl.where(
                                        m_mask_dlw, _mul_f32x2(dq_r, x_hat_sub), 0.0
                                    )
                                    dlw_partial = tl.sum(
                                        dlw_2d, axis=0
                                    )  # [ln_bwd_slice]
                                    dlw_prev = tlx.local_load(dlw_accum_buf[slice_id])
                                    tlx.local_store(
                                        dlw_accum_buf[slice_id],
                                        dlw_prev + dlw_partial[None, :],
                                    )
                                if HAS_LAYERNORM_BIAS:
                                    m_mask_dlb = (curr_m + tl.arange(0, BLOCK_M1))[
                                        :, None
                                    ] < qlen
                                    dlb_2d = tl.where(m_mask_dlb, dq_r, 0.0)
                                    dlb_partial = tl.sum(
                                        dlb_2d, axis=0
                                    )  # [ln_bwd_slice]
                                    dlb_prev = tlx.local_load(dlb_accum_buf[slice_id])
                                    tlx.local_store(
                                        dlb_accum_buf[slice_id],
                                        dlb_prev + dlb_partial[None, :],
                                    )
                                # Match type for Triton compiler
                                dq_r = dq_r.to(tlx.dtype_of(desc_dq))

                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("red_ln_pass1")

                            # Deferred do_raw TMA store wait + buffer release:
                            # issued before red_ln_pass1 to overlap TMA with
                            # LN backward computation.
                            if FUSED_RMS_NORM and FUSED_RESIDUAL_ADD:
                                if ENABLE_PROTON and idx == PROTON_TILE:
                                    pl.enter_scope("red_do_raw_store_wait")
                                tlx.async_descriptor_store_wait(0)
                                if ENABLE_PROTON and idx == PROTON_TILE:
                                    pl.exit_scope("red_do_raw_store_wait")
                                tlx.barrier_arrive(do_empties[do_buf_id_raw])

                            # DSMEM exchange c1 and c2
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("red_ln_dsmem")
                            ln_bwd_buf_id = (accum_cnt_inner + i) % NUM_REDUCTION_BUFS
                            ln_bwd_phase = (
                                (accum_cnt_inner + i) // NUM_REDUCTION_BUFS
                            ) & 1
                            ln_bwd_barrier = ln_bwd_barriers[ln_bwd_buf_id]
                            HALF_RED: tl.constexpr = NUM_REDUCTION_CTAS - 1
                            ln_bwd_buf_offset = ln_bwd_buf_id * 2 * HALF_RED
                            cluster_cta_rank_red = tlx.cluster_cta_rank()
                            c1_accum_f32 = c1_accum.to(tl.float32)
                            c2_accum_f32 = c2_accum.to(tl.float32)
                            for cta_i in tl.static_range(NUM_REDUCTION_CTAS):
                                if cluster_cta_rank_red != cta_i:
                                    # Remap: skip local CTA's slot
                                    dst_idx = (
                                        cluster_cta_rank_red
                                        if cluster_cta_rank_red < cta_i
                                        else cluster_cta_rank_red - 1
                                    )
                                    tlx.async_remote_shmem_store(
                                        dst=ln_bwd_buf[ln_bwd_buf_offset + dst_idx],
                                        src=c1_accum_f32,
                                        remote_cta_rank=cta_i,
                                        barrier=ln_bwd_barrier,
                                    )
                                    tlx.async_remote_shmem_store(
                                        dst=ln_bwd_buf[
                                            ln_bwd_buf_offset + HALF_RED + dst_idx
                                        ],
                                        src=c2_accum_f32,
                                        remote_cta_rank=cta_i,
                                        barrier=ln_bwd_barrier,
                                    )
                            if NUM_REDUCTION_CTAS:
                                tlx.barrier_expect_bytes(
                                    ln_bwd_barrier, ln_bwd_expected_bytes
                                )
                                tlx.barrier_wait(ln_bwd_barrier, phase=ln_bwd_phase)
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("red_ln_dsmem")
                            c1_total = c1_accum_f32  # start with local partial
                            c2_total = c2_accum_f32
                            for cta_i in tl.static_range(HALF_RED):
                                c1_total += tlx.local_load(
                                    ln_bwd_buf[ln_bwd_buf_offset + cta_i]
                                )
                                c2_total += tlx.local_load(
                                    ln_bwd_buf[ln_bwd_buf_offset + HALF_RED + cta_i]
                                )
                            N_LN: tl.constexpr = H * HEAD_DIM

                            # Pass 2: re-read dq from TMEM, apply LN backward correction, atomic_add
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("red_ln_pass2")
                            for slice_id in tl.static_range(LN_BWD_SUBTILE):
                                if ENABLE_PROTON and idx == PROTON_TILE:
                                    pl.enter_scope("red_p2_load")
                                dq_slice = tlx.subslice(
                                    dq_tile, ln_bwd_slice * slice_id, ln_bwd_slice
                                )
                                dq_r = tlx.local_load(dq_slice)
                                if slice_id == LN_BWD_SUBTILE - 1:
                                    tlx.fence_async_shared()
                                    tlx.barrier_arrive(dq_empty)
                                if FUSED_Q_RESIDUAL_ADD:
                                    do_slice = tlx.local_slice(
                                        do_tile,
                                        [0, ln_bwd_slice * slice_id],
                                        [BLOCK_M1, ln_bwd_slice],
                                    )
                                    do_r = tlx.local_load(do_slice).to(tl.float32)
                                    if slice_id == LN_BWD_SUBTILE - 1:
                                        tlx.fence_async_shared()
                                        tlx.barrier_arrive(do_empty)
                                    m_offs = curr_m + tl.arange(0, BLOCK_M1)
                                    do_r = tl.where(m_offs[:, None] < qlen, do_r, 0.0)
                                    dq_r = _add_f32x2(dq_r, do_r)
                                # Re-read LN(Q) from q_tile for x_hat
                                ln_q_sub2 = tlx.local_load(
                                    tlx.local_slice(
                                        q_tile_red,
                                        [0, ln_bwd_slice * slice_id],
                                        [BLOCK_M1, ln_bwd_slice],
                                    )
                                ).to(tl.float32)
                                if layernorm_weight is not None:
                                    ln_w_sub2 = tlx.local_load(
                                        tlx.local_slice(
                                            ln_w_buf[0],
                                            [0, ln_bwd_slice * slice_id],
                                            [1, ln_bwd_slice],
                                        )
                                    )  # (1, ln_bwd_slice) f32
                                    if HAS_LAYERNORM_BIAS:
                                        ln_b_sub2 = tlx.local_load(
                                            tlx.local_slice(
                                                ln_b_buf[0],
                                                [0, ln_bwd_slice * slice_id],
                                                [1, ln_bwd_slice],
                                            )
                                        )  # (1, ln_bwd_slice) f32
                                        x_hat_sub = fast_dividef(
                                            ln_q_sub2 - ln_b_sub2,
                                            ln_w_sub2.to(tl.float32),
                                        )
                                    else:
                                        x_hat_sub = fast_dividef(
                                            ln_q_sub2, ln_w_sub2.to(tl.float32)
                                        )
                                    dq_w_sub = dq_r * ln_w_sub2
                                else:
                                    x_hat_sub = ln_q_sub2
                                    dq_w_sub = dq_r
                                if ENABLE_PROTON and idx == PROTON_TILE:
                                    pl.exit_scope("red_p2_load")
                                # dx = rstd * (dq_w - (c1 + x_hat * c2) / N)
                                if ENABLE_PROTON and idx == PROTON_TILE:
                                    pl.enter_scope("red_p2_compute")
                                correction_sub = _add_f32x2(
                                    c1_total, _mul_f32x2(x_hat_sub, c2_total)
                                )
                                correction_sub = fast_dividef(
                                    correction_sub,
                                    tl.full(
                                        correction_sub.shape, N_LN, dtype=tl.float32
                                    ),
                                )
                                dq_raw_sub = _mul_f32x2(
                                    rstd_vals, _sub_f32x2(dq_w_sub, correction_sub)
                                )
                                # Fused residual add: dq += raw do (bypasses LN+RMS backward)
                                if FUSED_RESIDUAL_ADD and not FUSED_RMS_NORM:
                                    if ENABLE_PROTON and idx == PROTON_TILE:
                                        pl.enter_scope("red_p2_d_residual_load")
                                    do_raw_slice = tlx.local_slice(
                                        do_raw_tile,
                                        [0, ln_bwd_slice * slice_id],
                                        [BLOCK_M1, ln_bwd_slice],
                                    )
                                    do_raw_r = tlx.local_load(do_raw_slice).to(
                                        tl.float32
                                    )
                                    if ENABLE_PROTON and idx == PROTON_TILE:
                                        pl.exit_scope("red_p2_d_residual_load")
                                    if ENABLE_PROTON and idx == PROTON_TILE:
                                        pl.enter_scope("red_p2_d_residual_compute")
                                    if slice_id == LN_BWD_SUBTILE - 1:
                                        tlx.fence_async_shared()
                                        tlx.barrier_arrive(do_raw_empty)
                                    m_offs_raw = curr_m + tl.arange(0, BLOCK_M1)
                                    do_raw_r = tl.where(
                                        m_offs_raw[:, None] < qlen, do_raw_r, 0.0
                                    )
                                    dq_raw_sub = _add_f32x2(dq_raw_sub, do_raw_r)
                                    if ENABLE_PROTON and idx == PROTON_TILE:
                                        pl.exit_scope("red_p2_d_residual_compute")
                                dq_raw_sub = dq_raw_sub.to(tlx.dtype_of(desc_dq))
                                if ENABLE_PROTON and idx == PROTON_TILE:
                                    pl.exit_scope("red_p2_compute")
                                if ENABLE_PROTON and idx == PROTON_TILE:
                                    pl.enter_scope("red_p2_dq_store")
                                dq_smem_idx = slice_id % DQ_REDUCE_STAGES
                                tlx.async_descriptor_store_wait(DQ_REDUCE_STAGES - 1)
                                tlx.local_store(dq_store_buf[dq_smem_idx], dq_raw_sub)
                                tlx.fence_async_shared()
                                tlx.async_descriptor_store(
                                    desc_dq,
                                    dq_store_buf[dq_smem_idx],
                                    [
                                        (begin_q + curr_m).to(tl.int32),
                                        (off_h2 * stride_qh).to(tl.int32)
                                        + slice_id * ln_bwd_slice,
                                    ],
                                    store_reduce="add",
                                )
                                if ENABLE_PROTON and idx == PROTON_TILE:
                                    pl.exit_scope("red_p2_dq_store")
                                # Match type
                                dq_r = dq_r.to(tlx.dtype_of(desc_dq))

                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("red_ln_pass2")

                            # Fence SMEM reads of q_tile before releasing to load warp
                            tlx.fence_async_shared()
                            tlx.barrier_arrive(q_empty_red)
                        elif FUSED_LAYERNORM:
                            # Non-LN iteration but FUSED_LAYERNORM — use LN_BWD_SUBTILE
                            # for dq (desc_dq block_shape matches LN_BWD_SUBTILE)
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("red_dq_store")
                            for slice_id in tl.static_range(LN_BWD_SUBTILE):
                                dq_slice = tlx.subslice(
                                    dq_tile, ln_bwd_slice * slice_id, ln_bwd_slice
                                )
                                dq_r = tlx.local_load(dq_slice)
                                if FUSED_Q_RESIDUAL_ADD and start_n == 0:
                                    do_slice = tlx.local_slice(
                                        do_tile,
                                        [0, ln_bwd_slice * slice_id],
                                        [BLOCK_M1, ln_bwd_slice],
                                    )
                                    do_r = tlx.local_load(do_slice).to(tl.float32)
                                    m_offs = curr_m + tl.arange(0, BLOCK_M1)
                                    do_r = tl.where(m_offs[:, None] < qlen, do_r, 0.0)
                                    dq_r = _add_f32x2(dq_r, do_r)
                                dq_r = dq_r.to(tlx.dtype_of(desc_dq))
                                dq_smem_idx = slice_id % DQ_REDUCE_STAGES
                                tlx.async_descriptor_store_wait(DQ_REDUCE_STAGES - 1)
                                tlx.local_store(dq_store_buf[dq_smem_idx], dq_r)
                                tlx.fence_async_shared()
                                tlx.async_descriptor_store(
                                    desc_dq,
                                    dq_store_buf[dq_smem_idx],
                                    [
                                        (begin_q + curr_m).to(tl.int32),
                                        (off_h2 * stride_qh).to(tl.int32)
                                        + slice_id * ln_bwd_slice,
                                    ],
                                    store_reduce="add",
                                )

                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("red_dq_store")

                            # release dq
                            tlx.barrier_arrive(dq_empty)

                            if FUSED_Q_RESIDUAL_ADD:
                                tlx.barrier_arrive(do_empty)
                            if FUSED_RESIDUAL_ADD and not FUSED_RMS_NORM:
                                tlx.barrier_arrive(do_raw_empty)
                            if FUSED_RMS_NORM and FUSED_RESIDUAL_ADD:
                                tlx.barrier_arrive(do_empties[do_buf_id_raw])

                            # Release q_tile (dummy arrive — didn't read it)
                            tlx.barrier_arrive(q_empty_red)
                        else:
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("red_dq_store")
                            for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                                dq_slice = tlx.subslice(
                                    dq_tile, slice_size * slice_id, slice_size
                                )
                                dq_r = tlx.local_load(dq_slice)
                                if FUSED_Q_RESIDUAL_ADD and start_n == 0:
                                    do_slice = tlx.local_slice(
                                        do_tile,
                                        [0, slice_size * slice_id],
                                        [BLOCK_M1, slice_size],
                                    )
                                    do_r = tlx.local_load(do_slice).to(tl.float32)
                                    m_offs = curr_m + tl.arange(0, BLOCK_M1)
                                    do_r = tl.where(m_offs[:, None] < qlen, do_r, 0.0)
                                    dq_r = _add_f32x2(dq_r, do_r)
                                if (
                                    FUSED_RESIDUAL_ADD
                                    and not FUSED_RMS_NORM
                                    and start_n == 0
                                ):
                                    if ENABLE_PROTON and idx == PROTON_TILE:
                                        pl.enter_scope("red_d_residual")
                                    do_raw_slice = tlx.local_slice(
                                        do_raw_tile,
                                        [0, slice_size * slice_id],
                                        [BLOCK_M1, slice_size],
                                    )
                                    do_raw_r = tlx.local_load(do_raw_slice).to(
                                        tl.float32
                                    )
                                    if slice_id == EPILOGUE_SUBTILE - 1:
                                        tlx.fence_async_shared()
                                        tlx.barrier_arrive(do_raw_empty)
                                    m_offs_raw = curr_m + tl.arange(0, BLOCK_M1)
                                    do_raw_r = tl.where(
                                        m_offs_raw[:, None] < qlen, do_raw_r, 0.0
                                    )
                                    dq_r = _add_f32x2(dq_r, do_raw_r)
                                    if ENABLE_PROTON and idx == PROTON_TILE:
                                        pl.exit_scope("red_d_residual")
                                dq_r = dq_r.to(tlx.dtype_of(desc_dq))
                                dq_smem_idx = slice_id % DQ_REDUCE_STAGES
                                tlx.async_descriptor_store_wait(DQ_REDUCE_STAGES - 1)
                                tlx.local_store(dq_store_buf[dq_smem_idx], dq_r)
                                tlx.fence_async_shared()
                                tlx.async_descriptor_store(
                                    desc_dq,
                                    dq_store_buf[dq_smem_idx],
                                    [
                                        (begin_q + curr_m).to(tl.int32),
                                        (off_h2 * stride_qh).to(tl.int32)
                                        + slice_id * slice_size,
                                    ],
                                    store_reduce="add",
                                )

                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("red_dq_store")

                            # release dq
                            tlx.barrier_arrive(dq_empty)

                            if FUSED_Q_RESIDUAL_ADD:
                                tlx.barrier_arrive(do_empty)
                            if FUSED_RESIDUAL_ADD and not FUSED_RMS_NORM:
                                if start_n != 0:
                                    # Still need to arrive for pipeline flow even
                                    # though we didn't read do_raw_tile this iteration.
                                    tlx.barrier_arrive(do_raw_empty)
                            if FUSED_RMS_NORM and FUSED_RESIDUAL_ADD:
                                if start_n == 0:
                                    if ENABLE_PROTON and idx == PROTON_TILE:
                                        pl.enter_scope("red_do_raw_store_wait")
                                    tlx.async_descriptor_store_wait(0)
                                    if ENABLE_PROTON and idx == PROTON_TILE:
                                        pl.exit_scope("red_do_raw_store_wait")
                                tlx.barrier_arrive(do_empties[do_buf_id_raw])

                        # Deferred dw accumulation: tl.sum moved from activation warp
                        if HAS_RMS_NORM_WEIGHT and start_n == 0:
                            dw_scratch_buf_id, dw_scratch_phase = _get_bufidx_phase(
                                accum_cnt_inner + i, NUM_BUFFERS_DW_SCRATCH
                            )
                            dw_scratch_tile = dw_scratch_buf[dw_scratch_buf_id]
                            dw_scratch_full = dw_scratch_fulls[dw_scratch_buf_id]
                            dw_scratch_empty = dw_scratch_empties[dw_scratch_buf_id]
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("red_wait_dw_scratch")
                            tlx.barrier_wait(dw_scratch_full, dw_scratch_phase)
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("red_wait_dw_scratch")
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("red_dw_accum")
                            dw_2d = tlx.local_load(dw_scratch_tile).to(tl.float32)
                            dw_partial = tl.sum(dw_2d, axis=0)  # [M,D] -> [D]
                            dw_prev = tlx.local_load(dw_accum_buf[0])
                            dw_new = dw_prev + dw_partial
                            tlx.local_store(dw_accum_buf[0], dw_new)
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("red_dw_accum")
                            tlx.barrier_arrive(dw_scratch_empty)

                        # Increment pointers.
                        curr_m += step_m

                    accum_cnt_inner += num_steps
                    accum_cnt_outer += 1

                if USE_CLC:
                    tlx.clc_producer(clc_context, clc_phase_producer)
                    clc_phase_producer = clc_phase_producer ^ 1

                    tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx != -1
                    idx += 1
                else:
                    tile_idx += num_progs
                    idx += 1
                    has_more_tile = idx < tiles_per_sm

            # Flush SMEM accumulator to global memory (single atomic_add)
            if FUSED_RMS_NORM and HAS_RMS_NORM_WEIGHT:
                dw_final = tl.sum(tlx.local_load(dw_accum_buf[0]), axis=0)
                dw_off_h = tlx.cluster_cta_rank()
                dw_offs = dw_off_h * HEAD_DIM + tl.arange(0, BLOCK_D)
                tl.atomic_add(drms_norm_weight_out + dw_offs, dw_final, sem="relaxed")

            # Flush dlayernorm_weight/bias SMEM accumulators to global memory
            if FUSED_LAYERNORM:
                dlw_off_h = tlx.cluster_cta_rank()
                if HAS_LAYERNORM_WEIGHT:
                    for _flush_i in tl.static_range(LN_BWD_SUBTILE):
                        dlw_final = tlx.local_load(dlw_accum_buf[_flush_i])
                        tl.atomic_add(
                            dlayernorm_weight_out
                            + dlw_off_h * HEAD_DIM
                            + LN_BWD_SLICE * _flush_i
                            + tl.arange(0, LN_BWD_SLICE),
                            tl.sum(dlw_final, axis=0),
                            sem="relaxed",
                        )
                if HAS_LAYERNORM_BIAS:
                    for _flush_i in tl.static_range(LN_BWD_SUBTILE):
                        dlb_final = tlx.local_load(dlb_accum_buf[_flush_i])
                        tl.atomic_add(
                            dlayernorm_bias_out
                            + dlw_off_h * HEAD_DIM
                            + LN_BWD_SLICE * _flush_i
                            + tl.arange(0, LN_BWD_SLICE),
                            tl.sum(dlb_final, axis=0),
                            sem="relaxed",
                        )

        # mma
        with tlx.async_task(num_warps=1, registers=NUM_REGS_MMA):
            accum_cnt_inner = 0
            accum_cnt_outer = 0
            clc_phase_consumer = 0
            _clc_buf = 0  # noqa: F841
            idx = 0
            has_more_tile = True

            while has_more_tile:
                off_z, off_h, off_h_kv, off_q_z, pid = bwd_calculate_offsets(
                    tile_idx,
                    seq_index,
                    H,
                    G,
                    n_tile_num,
                    SORT_BY_SEQ_LENGTH,
                    BROADCAST_Q,
                    USE_I64_IDX,
                    FUSED_RMS_NORM,
                )

                begin_q = tl.load(Q_offsets + off_q_z)
                end_q = tl.load(Q_offsets + off_q_z + 1)
                start_n = pid * BLOCK_N1
                off_h2 = off_h.to(tl.int64)
                qlen = end_q - begin_q
                if FUSED_QKV:
                    begin_k = begin_q
                    end_k = end_q
                    klen = qlen
                else:
                    begin_k = tl.load(K_offsets + off_z)
                    end_k = tl.load(K_offsets + off_z + 1)
                    klen = end_k - begin_k

                if start_n < klen:
                    # Wait for K, V ready
                    kv_buf_id, kv_phase = _get_bufidx_phase(
                        accum_cnt_outer, NUM_BUFFERS_KV
                    )
                    k_tile = k_tiles[kv_buf_id]
                    v_tile = v_tiles[kv_buf_id]
                    k_full = k_fulls[kv_buf_id]
                    v_full = v_fulls[kv_buf_id]
                    k_mma_done_bar = k_mma_done[kv_buf_id]
                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.enter_scope("mma_wait_kv")
                    tlx.barrier_wait(k_full, kv_phase)
                    tlx.barrier_wait(v_full, kv_phase)
                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.exit_scope("mma_wait_kv")

                    # Wait for dK, dV to be released
                    dkv_buf_id, dkv_phase = _get_bufidx_phase(
                        accum_cnt_outer, NUM_BUFFERS_TMEM
                    )
                    dk_tile = dk_tiles[dkv_buf_id]
                    dv_tile = dv_tiles[dkv_buf_id]
                    dk_full = dk_fulls[dkv_buf_id]
                    dv_full = dv_fulls[dkv_buf_id]
                    dk_empty = dk_empties[dkv_buf_id]
                    dv_empty = dv_empties[dkv_buf_id]

                    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
                    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)

                    num_steps, start_m = _gdpa_bwd_tlx_compute_num_steps(
                        qlen, start_n, BLOCK_M1, BLOCK_N1, WINDOW_SIZE
                    )

                    for i in tl.range(0, num_steps, 1, num_stages=0):
                        q_buf_id, q_phase = _get_bufidx_phase(
                            accum_cnt_inner + i, NUM_BUFFERS_Q
                        )
                        tmem_buf_id, tmem_phase = _get_bufidx_phase(
                            accum_cnt_inner + i, NUM_BUFFERS_TMEM
                        )
                        ds_buf_id, ds_phase = _get_bufidx_phase(
                            accum_cnt_inner + i, NUM_BUFFERS_DS
                        )

                        q_tile = q_tiles[q_buf_id]
                        q_full = q_norm_fulls[q_buf_id]
                        q_empty = q_empties[q_buf_id]

                        do_buf_id, do_phase = _get_bufidx_phase(
                            accum_cnt_inner + i, NUM_BUFFERS_DO
                        )
                        if FUSED_RMS_NORM:
                            do_rms_buf_id, _ = _get_bufidx_phase(
                                accum_cnt_inner + i, NUM_BUFFERS_DO_RMS
                            )
                            do_tile = do_rms_tiles[do_rms_buf_id]
                            do_empty = do_rms_empties[do_rms_buf_id]
                        else:
                            do_tile = do_tiles[do_buf_id]
                            do_empty = do_empties[do_buf_id]
                        do_full = do_fulls[do_buf_id]

                        qk_tile = qk_tiles[tmem_buf_id]
                        ppT_tile = ppT_tiles[tmem_buf_id]
                        dpT_tile = dpT_tiles[tmem_buf_id]
                        dsT_tile = dsT_tiles[ds_buf_id]
                        dq_tile = dq_tiles[tmem_buf_id]

                        ppT_empty = ppT_empties[tmem_buf_id]
                        dq_empty = dq_empties[tmem_buf_id]
                        if FUSED_LAYERNORM:
                            dpT_empty = dpT_empties[tmem_buf_id]
                        qk_full = qk_fulls[tmem_buf_id]
                        ppT_full = ppT_fulls[tmem_buf_id]
                        dpT_full = dpT_fulls[tmem_buf_id]
                        dsT_full = dsT_fulls[ds_buf_id]
                        dq_full = dq_fulls[tmem_buf_id]

                        # 1. qkT = dot(k, qT)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("mma_wait_q")
                        tlx.barrier_wait(q_full, q_phase)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("mma_wait_q")
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("mma_wait_ppT_empty")
                        tlx.barrier_wait(ppT_empty, tmem_phase ^ 1)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("mma_wait_ppT_empty")
                        qT = tlx.local_trans(q_tile)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("mma_dot_qkT")
                        tlx.async_dot(
                            k_tile,
                            qT,
                            qk_tile,
                            use_acc=False,
                            mBarriers=[qk_full],
                        )
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("mma_dot_qkT")

                        # 2. dpT = dot(v, doT)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("mma_wait_do")
                        tlx.barrier_wait(do_full, do_phase)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("mma_wait_do")
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("mma_wait_dpT_empty")
                        if FUSED_LAYERNORM:
                            tlx.barrier_wait(dpT_empty, tmem_phase ^ 1)
                        else:
                            tlx.barrier_wait(dq_empty, tmem_phase ^ 1)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("mma_wait_dpT_empty")
                        doT = tlx.local_trans(do_tile)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("mma_dot_dpT")
                        tlx.async_dot(
                            v_tile,
                            doT,
                            dpT_tile,
                            use_acc=False,
                            mBarriers=[dpT_full],
                        )
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("mma_dot_dpT")

                        # 3. dv += dot(ppT, do)
                        if i == 0:
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("mma_wait_dv_empty")
                            tlx.barrier_wait(dv_empty, dkv_phase ^ 1)
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("mma_wait_dv_empty")
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("mma_wait_ppT")
                        tlx.barrier_wait(ppT_full, tmem_phase)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("mma_wait_ppT")
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("mma_dot_dv")
                        tlx.async_dot(
                            ppT_tile,
                            do_tile,
                            dv_tile,
                            use_acc=i > 0,
                            mBarriers=[do_empty, ppT_empty],
                        )
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("mma_dot_dv")

                        # 4. Wait for dsT (ACT computes activation grad)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("mma_wait_dsT")
                        tlx.barrier_wait(dsT_full, ds_phase)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("mma_wait_dsT")

                        # 5. dk += dot(dsT, qT)
                        if i == 0:
                            tlx.barrier_wait(dk_empty, dkv_phase ^ 1)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("mma_dot_dk")
                        tlx.async_dot(
                            dsT_tile,
                            q_tile,
                            dk_tile,
                            use_acc=i > 0,
                            mBarriers=[q_empty],
                        )
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("mma_dot_dk")

                        # 6. dq = dot(dsT^T, k)
                        if FUSED_LAYERNORM:
                            tlx.barrier_wait(dq_empty, tmem_phase ^ 1)
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("mma_dot_dq")
                        dsT_view = tlx.local_trans(dsT_tile)
                        tlx.async_dot(
                            dsT_view,
                            k_tile,
                            dq_tile,
                            use_acc=False,
                            mBarriers=[dq_full],
                        )
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("mma_dot_dq")

                    # Commit all pending TMEM writes (dv, dk, dq dots) and signal dv_full.
                    # Then arrive dk_full and k_mma_done_bar — data already committed.
                    tlx.tcgen05_commit(dv_full)
                    tlx.tcgen05_commit(dk_full)
                    tlx.tcgen05_commit(k_mma_done_bar)

                    accum_cnt_inner += num_steps
                    accum_cnt_outer += 1

                if USE_CLC:
                    tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx != -1
                    idx += 1
                else:
                    tile_idx += num_progs
                    idx += 1
                    has_more_tile = idx < tiles_per_sm

        # load
        with tlx.async_task(num_warps=1, registers=NUM_REGS_LOAD):
            accum_cnt_inner = 0
            accum_cnt_outer = 0
            clc_phase_consumer = 0
            _clc_buf = 0  # noqa: F841
            idx = 0
            has_more_tile = True

            while has_more_tile:
                off_z, off_h, off_h_kv, off_q_z, pid = bwd_calculate_offsets(
                    tile_idx,
                    seq_index,
                    H,
                    G,
                    n_tile_num,
                    SORT_BY_SEQ_LENGTH,
                    BROADCAST_Q,
                    USE_I64_IDX,
                    FUSED_RMS_NORM,
                )

                begin_q = tl.load(Q_offsets + off_q_z)
                end_q = tl.load(Q_offsets + off_q_z + 1)
                start_n = pid * BLOCK_N1
                start_m = 0
                off_h2 = off_h.to(tl.int64)
                qlen = end_q - begin_q
                if FUSED_QKV:
                    begin_k = begin_q
                    end_k = end_q
                    klen = qlen
                else:
                    begin_k = tl.load(K_offsets + off_z)
                    end_k = tl.load(K_offsets + off_z + 1)
                    klen = end_k - begin_k

                # Some of the ops are used for both producer and consumer, some are used by consumer
                # Try to correctly specialize the IfOp by marking all ops.
                # invert of start_n > klen and start_m > qlen
                if start_n < klen:
                    # load K and V: they stay in SRAM throughout the inner loop.
                    kv_buf_id, kv_phase = _get_bufidx_phase(
                        accum_cnt_outer, NUM_BUFFERS_KV
                    )

                    k_tile = k_tiles[kv_buf_id]
                    v_tile = v_tiles[kv_buf_id]
                    k_full = k_fulls[kv_buf_id]
                    v_full = v_fulls[kv_buf_id]
                    k_empty = k_empties[kv_buf_id]
                    v_empty = v_empties[kv_buf_id]

                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.enter_scope("load_k")
                    tlx.barrier_wait(k_empty, kv_phase ^ 1)
                    tlx.barrier_expect_bytes(k_full, 2 * BLOCK_N1 * BLOCK_D)  # float16
                    tlx.async_descriptor_load(
                        desc_k,
                        k_tile,
                        [
                            (begin_k + start_n).to(tl.int32),
                            (off_h_kv * stride_kh).to(tl.int32),
                        ],
                        k_full,
                    )
                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.exit_scope("load_k")

                    num_steps, start_m = _gdpa_bwd_tlx_compute_num_steps(
                        qlen, start_n, BLOCK_M1, BLOCK_N1, WINDOW_SIZE
                    )
                    curr_m = start_m
                    step_m = BLOCK_M1

                    # load first Q
                    q_buf_id, q_phase = _get_bufidx_phase(
                        accum_cnt_inner, NUM_BUFFERS_Q
                    )
                    q_tile = q_tiles[q_buf_id]
                    q_full = q_fulls[q_buf_id]
                    q_empty = q_empties[q_buf_id]

                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.enter_scope("load_q")
                    tlx.barrier_wait(q_empty, q_phase ^ 1)
                    tlx.barrier_expect_bytes(q_full, 2 * BLOCK_M1 * BLOCK_D)  # float16
                    tlx.async_descriptor_load(
                        desc_q,
                        q_tile,
                        [
                            (begin_q + curr_m).to(tl.int32),
                            (off_h2 * stride_qh).to(tl.int32),
                        ],
                        q_full,
                    )
                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.exit_scope("load_q")

                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.enter_scope("load_v")
                    tlx.barrier_wait(v_empty, kv_phase ^ 1)
                    tlx.barrier_expect_bytes(v_full, 2 * BLOCK_N1 * BLOCK_D)  # float16
                    tlx.async_descriptor_load(
                        desc_v,
                        v_tile,
                        [
                            (begin_k + start_n).to(tl.int32),
                            (off_h_kv * stride_kh).to(tl.int32),
                        ],
                        v_full,
                    )
                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.exit_scope("load_v")

                    if not BROADCAST_Q:
                        begin_o = begin_q
                    else:
                        begin_o = qlen * off_z

                    # load first dO
                    do_buf_id, do_phase = _get_bufidx_phase(
                        accum_cnt_inner, NUM_BUFFERS_DO
                    )
                    do_tile = do_tiles[do_buf_id]
                    do_full = do_fulls[do_buf_id]
                    do_empty = do_empties[do_buf_id]

                    if ENABLE_PROTON and idx == PROTON_TILE:
                        pl.enter_scope("load_do")
                    tlx.barrier_wait(do_empty, do_phase ^ 1)
                    if FUSED_RMS_NORM:
                        do_raw_full = do_raw_fulls[do_buf_id]
                        tlx.barrier_expect_bytes(do_raw_full, 2 * BLOCK_M1 * BLOCK_D)
                        tlx.async_descriptor_load(
                            desc_do,
                            do_tile,
                            [
                                (begin_o + curr_m).to(tl.int32),
                                (off_h2 * stride_qh).to(tl.int32),
                            ],
                            do_raw_full,
                        )
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("load_do")
                        # Also load o (post-norm output)
                        o_buf_id, o_phase = _get_bufidx_phase(
                            accum_cnt_inner, NUM_BUFFERS_O
                        )
                        o_tile = o_tiles[o_buf_id]
                        o_full = o_fulls[o_buf_id]
                        o_empty = o_empties[o_buf_id]
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("load_o")
                        tlx.barrier_wait(o_empty, o_phase ^ 1)
                        tlx.barrier_expect_bytes(o_full, 2 * BLOCK_M1 * BLOCK_D)
                        tlx.async_descriptor_load(
                            desc_o,
                            o_tile,
                            [
                                (begin_o + curr_m).to(tl.int32),
                                (off_h2 * stride_oh).to(tl.int32),
                            ],
                            o_full,
                        )
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("load_o")
                    else:
                        tlx.barrier_expect_bytes(
                            do_full, 2 * BLOCK_M1 * BLOCK_D
                        )  # float16
                        tlx.async_descriptor_load(
                            desc_do,
                            do_tile,
                            [
                                (begin_o + curr_m).to(tl.int32),
                                (off_h2 * stride_qh).to(tl.int32),
                            ],
                            do_full,
                        )
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("load_do")

                    for i in tl.range(1, num_steps, 1, loop_unroll_factor=1):
                        # Increment pointers first due to the prologue loading
                        curr_m += step_m

                        q_buf_id, q_phase = _get_bufidx_phase(
                            accum_cnt_inner + i, NUM_BUFFERS_Q
                        )
                        q_tile = q_tiles[q_buf_id]
                        q_full = q_fulls[q_buf_id]
                        q_empty = q_empties[q_buf_id]

                        do_buf_id, do_phase = _get_bufidx_phase(
                            accum_cnt_inner + i, NUM_BUFFERS_DO
                        )
                        do_tile = do_tiles[do_buf_id]
                        do_full = do_fulls[do_buf_id]
                        do_empty = do_empties[do_buf_id]

                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("load_loop_q")
                        tlx.barrier_wait(q_empty, q_phase ^ 1)
                        tlx.barrier_expect_bytes(
                            q_full, 2 * BLOCK_M1 * BLOCK_D
                        )  # float16
                        tlx.async_descriptor_load(
                            desc_q,
                            q_tile,
                            [
                                (begin_q + curr_m).to(tl.int32),
                                (off_h2 * stride_qh).to(tl.int32),
                            ],
                            q_full,
                        )
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.exit_scope("load_loop_q")

                        # we may need this line for correctness
                        # offs_m = curr_m + tl.arange(0, BLOCK_M1)
                        # qmask = (offs_k[:, None] < HEAD_DIM) & (offs_m[None, :] < qlen)
                        # qT = tl.where(qmask, qT, 0.0)

                        # move dot ahead hoping to overlap with other computations
                        if ENABLE_PROTON and idx == PROTON_TILE:
                            pl.enter_scope("load_loop_do")
                        tlx.barrier_wait(do_empty, do_phase ^ 1)
                        if FUSED_RMS_NORM:
                            do_raw_full = do_raw_fulls[do_buf_id]
                            tlx.barrier_expect_bytes(
                                do_raw_full, 2 * BLOCK_M1 * BLOCK_D
                            )
                            tlx.async_descriptor_load(
                                desc_do,
                                do_tile,
                                [
                                    (begin_o + curr_m).to(tl.int32),
                                    (off_h2 * stride_qh).to(tl.int32),
                                ],
                                do_raw_full,
                            )
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("load_loop_do")
                            # Also load o for this iteration
                            o_buf_id, o_phase = _get_bufidx_phase(
                                accum_cnt_inner + i, NUM_BUFFERS_O
                            )
                            o_tile = o_tiles[o_buf_id]
                            o_full = o_fulls[o_buf_id]
                            o_empty = o_empties[o_buf_id]
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.enter_scope("load_loop_o")
                            tlx.barrier_wait(o_empty, o_phase ^ 1)
                            tlx.barrier_expect_bytes(o_full, 2 * BLOCK_M1 * BLOCK_D)
                            tlx.async_descriptor_load(
                                desc_o,
                                o_tile,
                                [
                                    (begin_o + curr_m).to(tl.int32),
                                    (off_h2 * stride_oh).to(tl.int32),
                                ],
                                o_full,
                            )
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("load_loop_o")
                        else:
                            tlx.barrier_expect_bytes(
                                do_full, 2 * BLOCK_M1 * BLOCK_D
                            )  # float16
                            tlx.async_descriptor_load(
                                desc_do,
                                do_tile,
                                [
                                    (begin_o + curr_m).to(tl.int32),
                                    (off_h2 * stride_qh).to(tl.int32),
                                ],
                                do_full,
                            )
                            if ENABLE_PROTON and idx == PROTON_TILE:
                                pl.exit_scope("load_loop_do")

                        # omask = (offs_m[:, None] < qlen) & (offs_k[None, :] < HEAD_DIM)
                        # do = tl.where(omask, do, 0.0)
                    accum_cnt_inner += num_steps
                    accum_cnt_outer += 1

                if USE_CLC:
                    tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx != -1
                    idx += 1
                else:
                    tile_idx += num_progs
                    idx += 1
                    has_more_tile = idx < tiles_per_sm


@torch.library.custom_op("ads_mkl::tlx_gdpa_megakernel_backward", mutates_args=())
def tlx_generalized_dot_product_attention_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    BATCH: int,
    N_HEAD: int,
    Q_GROUP: int,
    N_CTX: int,
    N_CTX_KV: int,
    BLOCK_D: int,
    HEAD_DIM: int,
    fused_qkv: bool = False,
    fused_kv: bool = False,
    qk_scale: float | None = None,
    sort_by_seq_length: bool = False,
    seq_index: torch.Tensor | None = None,
    output_offset: torch.Tensor | None = None,
    use_start_end_offsets: bool = False,
    window_size: int | None = None,
    broadcast_q: bool = False,
    activation_enum_int: int = 0,
    ensemble_activation_list: List[int] | None = None,
    fused_rms_norm: bool = False,
    rrms_out: torch.Tensor | None = None,
    rms_norm_weight: torch.Tensor | None = None,
    fused_residual_add: bool = False,
    fused_q_residual_add: bool = False,
    fused_layernorm: bool = False,
    layernorm_weight: torch.Tensor | None = None,
    layernorm_bias: torch.Tensor | None = None,
    ln_mean_out: torch.Tensor | None = None,
    ln_rstd_out: torch.Tensor | None = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if qk_scale is None:
        qk_scale = 1.0
    dk_dv_init = torch.zeros if use_start_end_offsets or Q_GROUP > 1 else torch.empty
    if broadcast_q:
        dq = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
    else:
        dq = torch.zeros(q.shape, device=q.device, dtype=q.dtype)

    dk_dv_dtype = None
    if not fused_qkv:  # k is None with fused_qkv
        dk_dv_dtype = k.dtype

    if fused_qkv:
        dk, dv = None, None
    elif fused_kv:
        dk = dk_dv_init(k.shape, device=k.device, dtype=dk_dv_dtype)
        dv = None
    else:
        dk = dk_dv_init(k.shape, device=k.device, dtype=dk_dv_dtype)
        dv = dk_dv_init(v.shape, device=v.device, dtype=dk_dv_dtype)

    do = expect_contiguous(do)

    if output_offset is None:
        output_offset = q_offsets

    bs = k_offsets.size(0) - 1
    L, _, _ = k.shape
    is_dense_kv = bs * N_CTX_KV == L
    is_aligned_kv = is_dense_kv and N_CTX_KV % 128 == 0

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
    if fused_rms_norm:
        desc_o = TensorDescriptor(
            o,
            shape=[o.shape[0], HEAD_DIM * N_HEAD],
            strides=[HEAD_DIM * N_HEAD, 1],
            block_shape=dummy_block,
        )
    else:
        desc_o = None
    desc_dq = TensorDescriptor(
        dq,
        shape=[dq.shape[0], HEAD_DIM * N_HEAD],
        strides=[HEAD_DIM * N_HEAD, 1],
        block_shape=dummy_block,
    )
    if fused_rms_norm and fused_residual_add:
        desc_do_raw = TensorDescriptor(
            dq,
            shape=[dq.shape[0], HEAD_DIM * N_HEAD],
            strides=[HEAD_DIM * N_HEAD, 1],
            block_shape=dummy_block,
        )
    else:
        desc_do_raw = None
    if is_aligned_kv:
        desc_dk = TensorDescriptor(
            dk,
            shape=[dk.shape[0], HEAD_DIM * N_HEAD],
            strides=[HEAD_DIM * N_HEAD, 1],
            block_shape=dummy_block,
        )
        desc_dv = TensorDescriptor(
            dv,
            shape=[dv.shape[0], HEAD_DIM * N_HEAD],
            strides=[HEAD_DIM * N_HEAD, 1],
            block_shape=dummy_block,
        )
    else:
        desc_dk, desc_dv = None, None

    def alloc_fn(size: int, alignment: int, _):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    NUM_SMS = (
        get_num_sms() or 1000000
    )  # if num sms is None, use a large number so that it is a no-op
    if fused_rms_norm or fused_layernorm:
        NUM_SMS = NUM_SMS // N_HEAD * N_HEAD

    def grid(args):
        return (
            min(NUM_SMS, triton.cdiv(N_CTX_KV, args["BLOCK_N1"]) * BATCH * N_HEAD)
            if not args["USE_CLC"]
            else triton.cdiv(N_CTX_KV, args["BLOCK_N1"]) * BATCH * N_HEAD,
            1,
            1,
        )

    ensemble_activation_list = ensemble_activation_list or []

    unrolled_kernel = unroll_varargs(
        gdpa_backward_tlx,
        N=len(ensemble_activation_list),
        unroll_as_const=True,
    )

    drms_norm_weight_out = None
    if fused_rms_norm and rms_norm_weight is not None:
        drms_norm_weight_out = torch.zeros(
            N_HEAD * HEAD_DIM, device=q.device, dtype=torch.float32
        )

    dlayernorm_weight_out = None
    dlayernorm_bias_out = None
    if fused_layernorm and layernorm_weight is not None:
        dlayernorm_weight_out = torch.zeros(
            N_HEAD * HEAD_DIM, device=q.device, dtype=torch.float32
        )
    if fused_layernorm and layernorm_bias is not None:
        dlayernorm_bias_out = torch.zeros(
            N_HEAD * HEAD_DIM, device=q.device, dtype=torch.float32
        )

    ctas_per_cga = (N_HEAD, 1, 1) if (fused_rms_norm or fused_layernorm) else (1, 1, 1)

    autotuned_kernel_fn = _get_autotune_kernel_backward(
        unrolled_kernel,
        ctas_per_cga=ctas_per_cga,
        has_drms_norm_weight=rms_norm_weight is not None,
        fused_rms_norm=fused_rms_norm,
        has_dlayernorm_weight=fused_layernorm and layernorm_weight is not None,
        has_dlayernorm_bias=fused_layernorm and layernorm_bias is not None,
    )

    enable_proton = os.getenv("ENABLE_PROTON") == "1"
    if enable_proton:
        proton_mode = proton.mode.Default(
            metric_type="cycle", optimizations="clock32", buffer_type="global"
        )
        proton.start(
            "proton_bwd", data="trace", backend="instrumentation", mode=proton_mode
        )

    autotuned_kernel_fn[grid](
        *ensemble_activation_list,
        desc_q,
        q_offsets,
        desc_k,
        k_offsets,
        desc_v,
        seq_index,  #
        desc_do,
        output_offset,
        desc_dq,  #
        desc_do_raw,
        dq,
        desc_dk,
        dk,
        desc_dv,
        dv,  #
        q.stride(0),
        q.stride(0) if fused_qkv else k.stride(0),
        q.stride(1),
        q.stride(1) if fused_qkv else k.stride(1),
        q.stride(2),
        do.stride(0),
        do.stride(1),
        BATCH,
        N_HEAD,
        Q_GROUP,  #
        N_CTX,
        N_CTX_KV,  #
        qk_scale,  #
        FUSED_QKV=fused_qkv,
        FUSED_KV=fused_kv,
        SORT_BY_SEQ_LENGTH=sort_by_seq_length,
        BLOCK_D=BLOCK_D,
        HEAD_DIM=HEAD_DIM,
        USE_START_END_OFFSETS=use_start_end_offsets,
        WINDOW_SIZE=window_size,
        BROADCAST_Q=broadcast_q,
        IS_DENSE_KV=is_dense_kv,
        IS_ALIGNED_KV=is_aligned_kv,
        activation_enum_int=activation_enum_int,
        USE_I64_IDX=should_use_i64_idx(q, k, v, o),
        desc_o=desc_o,
        rrms_out=rrms_out,
        rms_norm_weight=rms_norm_weight,
        drms_norm_weight_out=drms_norm_weight_out,
        stride_oh=o.stride(1) if fused_rms_norm else 0,
        FUSED_RMS_NORM=fused_rms_norm,
        HAS_RMS_NORM_WEIGHT=rms_norm_weight is not None,
        NUM_REDUCTION_CTAS=N_HEAD if (fused_rms_norm or fused_layernorm) else 0,
        NUM_REDUCTION_BUFS=2,
        FUSED_Q_RESIDUAL_ADD=fused_q_residual_add,
        FUSED_RESIDUAL_ADD=fused_residual_add,
        NUM_BUFFERS_O=1,
        FUSED_LAYERNORM=fused_layernorm,
        layernorm_weight=layernorm_weight,
        layernorm_bias=layernorm_bias,
        HAS_LAYERNORM_WEIGHT=layernorm_weight is not None,
        HAS_LAYERNORM_BIAS=layernorm_bias is not None,
        Q_raw=q,
        ln_mean_out=ln_mean_out,
        ln_rstd_out=ln_rstd_out,
        dlayernorm_weight_out=dlayernorm_weight_out,
        dlayernorm_bias_out=dlayernorm_bias_out,
        ENABLE_PROTON=True if os.getenv("ENABLE_PROTON") == "1" else False,
        PROTON_TILE=10,
    )

    if enable_proton:
        torch.cuda.synchronize()
        proton.finalize()

    if broadcast_q:
        dq = dq.to(q.dtype)

    return dq, dk, dv, drms_norm_weight_out, dlayernorm_weight_out, dlayernorm_bias_out


def _tlx_generalized_dot_product_attention_backward(
    ctx, do, d_rrms_out, d_rms_norm_out, d_ln_mean_out, d_ln_rstd_out
):
    # Unpack saved tensors
    saved = list(ctx.saved_tensors)
    q, k, v, o, q_offsets, k_offsets, output_offsets, seq_index = saved[:8]
    idx = 8
    rrms = None
    rms_norm_weight = None
    if ctx.fused_rms_norm:
        rrms = saved[idx]
        idx += 1
    if ctx.has_rms_norm_weight:
        rms_norm_weight = saved[idx]
        idx += 1
    # Unpack LayerNorm saved tensors
    layernorm_weight = None
    layernorm_bias = None
    ln_mean = None
    ln_rstd = None
    if ctx.fused_layernorm:
        ln_mean = saved[idx]
        idx += 1
        ln_rstd = saved[idx]
        idx += 1
        if ctx.has_layernorm_weight:
            layernorm_weight = saved[idx]
            idx += 1
        if ctx.has_layernorm_bias:
            layernorm_bias = saved[idx]
            idx += 1

    drms_norm_weight = None
    dq, dk, dv, drms_norm_weight_out, dlayernorm_weight_out, dlayernorm_bias_out = (
        tlx_generalized_dot_product_attention_backward(
            do,
            q,
            k,
            v,
            o,
            q_offsets,
            k_offsets,
            ctx.BATCH,
            ctx.N_HEAD,
            ctx.G,
            ctx.N_CTX,
            ctx.N_CTX_KV,
            ctx.BLOCK_D,
            ctx.HEAD_DIM,
            ctx.fused_qkv,
            ctx.fused_kv,
            ctx.qk_scale,
            ctx.sort_by_seq_length,
            seq_index,
            output_offsets,
            ctx.use_start_end_offsets,
            ctx.window_size,
            ctx.broadcast_q,
            ctx.activation_enum_int,
            ctx.ensemble_activation_list,
            fused_rms_norm=ctx.fused_rms_norm,
            rrms_out=rrms,
            rms_norm_weight=rms_norm_weight,
            fused_residual_add=ctx.fused_residual_add,
            fused_q_residual_add=ctx.fused_q_residual_add,
            fused_layernorm=ctx.fused_layernorm,
            layernorm_weight=layernorm_weight,
            layernorm_bias=layernorm_bias,
            ln_mean_out=ln_mean,
            ln_rstd_out=ln_rstd,
        )
    )

    if drms_norm_weight_out is not None:
        drms_norm_weight = drms_norm_weight_out.to(rms_norm_weight.dtype)

    dlayernorm_weight = None
    dlayernorm_bias = None
    if dlayernorm_weight_out is not None:
        dlayernorm_weight = dlayernorm_weight_out.to(layernorm_weight.dtype)
    if dlayernorm_bias_out is not None:
        dlayernorm_bias = dlayernorm_bias_out.to(layernorm_bias.dtype)

    # d_residual is now fused into dq inside the kernel (dq += do when
    # fused_residual_add, since residual is always the pre-layernorm Q)

    # Return grads for all inputs (36 total):
    # query(dq), key(dk), value(dv), 25x None, fused_rms_norm(None),
    # rms_norm_weight(drms_norm_weight), fused_residual_add(None),
    # residual(None), fused_q_residual_add(None),
    # fused_layernorm(None), layernorm_weight(dlayernorm_weight),
    # layernorm_bias(dlayernorm_bias)
    return (
        dq,
        dk,
        dv,
        *((None,) * 25),
        None,
        drms_norm_weight,
        None,
        None,
        None,
        None,
        dlayernorm_weight,
        dlayernorm_bias,
    )


@register_flop_formula(torch.ops.ads_mkl.tlx_gdpa_megakernel_backward, get_raw=True)
def generalized_dot_product_attention_backward_flop(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    BATCH: int,
    N_HEAD: int,
    Q_GROUP: int,
    N_CTX: int,
    N_CTX_KV: int,
    BLOCK_D: int,
    HEAD_DIM: int,
    fused_qkv: bool = False,
    fused_kv: bool = False,
    qk_scale: float | None = None,
    sort_by_seq_length: bool = False,
    seq_index: torch.Tensor | None = None,
    output_offset: torch.Tensor | None = None,
    use_start_end_offsets: bool = False,
    window_size: int | None = None,
    broadcast_q: bool = False,
    *args,
    **kwargs,
) -> int:
    """Count flops for self-attention."""
    if fused_qkv:
        HEAD_DIM = q.shape[-1] // 3
        q, k, v = q.split(HEAD_DIM, dim=-1)
    elif fused_kv:
        HEAD_DIM = k.shape[-1] // 2
        k, v = k.split(HEAD_DIM, dim=-1)
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


_tlx_generalized_dot_product_attention.register_autograd(
    _tlx_generalized_dot_product_attention_backward,
    setup_context=_generalized_dot_product_attention_setup_context,
)


@torch.jit.script_if_tracing
@custom_register_kernel("ads_mkl::tlx_gdpa_megakernel", "cpu")
def cpu_generalized_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_offset: torch.Tensor,
    key_offset: torch.Tensor,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    ad_to_request_offset: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    attn_offset: torch.Tensor | None = None,
    is_causal: bool = False,
    qk_scale: float | None = None,
    seq_index: torch.Tensor | None = None,
    allow_tf32: bool = True,
    output_offset: torch.Tensor | None = None,
    use_start_end_offsets: bool = False,
    window_size: int | None = None,
    broadcast_q: bool = False,
    activation: str = "raw",
    enable_persistent: bool = False,
    enable_tma: bool = False,
    enable_ws: bool = False,
    use_dq_atomic_add: bool = False,
    total_num_objects: int | None = None,
    ts_encoding_params: torch.Tensor | None = None,
    ts_event_request_even_params: torch.Tensor | None = None,
    ts_event_request_log_params: torch.Tensor | None = None,
    ts_adjacent_event_log_params: torch.Tensor | None = None,
    ts_encoding_bucket_values: torch.Tensor | None = None,
    ts_event_request_even_bucket_values: torch.Tensor | None = None,
    ts_event_request_log_bucket_values: torch.Tensor | None = None,
    ts_adjacent_event_log_bucket_values: torch.Tensor | None = None,
    pos_emb_params: torch.Tensor | None = None,
    pos_emb_bucket_values: torch.Tensor | None = None,
    bwd_opt_tech: str = "base",
) -> torch.Tensor:
    assert attn_mask is None, "attn_mask/bias unsupported"
    assert not is_causal, "Causal unsupported"
    fused_qkv = key is None and value is None
    fused_kv = key is not None and value is None
    query = query.to(torch.float32) if query.dtype != torch.float32 else query

    # shape constraints
    if fused_qkv:
        _HEAD_DIM_Q = query.shape[-1] // 3  # noqa: F841
    elif fused_kv:
        _HEAD_DIM_Q = query.shape[-1]  # noqa: F841
    else:
        _HEAD_DIM_Q = query.shape[-1]  # noqa: F841

    bs = query_offset.size(0) - 1
    o = torch.zeros_like(query)

    is_predict = True
    if ad_to_request_offset is None:
        is_predict = False

    for i in range(bs):
        q_start = query_offset[i]
        q_end = query_offset[i + 1]
        q_i = query[q_start:q_end, :, :].transpose(0, 1)

        if is_predict:
            assert isinstance(ad_to_request_offset, torch.Tensor)
            b = ad_to_request_offset[i]
            kv_start = key_offset[b]
            kv_end = key_offset[b + 1]
        else:
            kv_start = key_offset[i]
            kv_end = key_offset[i + 1]
        k_i = key[kv_start:kv_end, :, :].transpose(0, 1)
        v_i = value[kv_start:kv_end, :, :].transpose(0, 1)

        s = torch.matmul(q_i, k_i.transpose(-2, -1))
        p = torch.nn.functional.gelu(s)
        out = torch.matmul(p, v_i)
        out = out.transpose(0, 1).contiguous()
        o[q_start:q_end, :, :] = out

    return o


@torch.library.register_fake("ads_mkl::tlx_gdpa_megakernel")
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_offset: torch.Tensor,
    key_offset: torch.Tensor,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    ad_to_request_offset: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    attn_offset: torch.Tensor | None = None,
    is_causal: bool = False,
    qk_scale: float | None = None,
    seq_index: torch.Tensor | None = None,
    allow_tf32: bool = True,
    output_offset: torch.Tensor | None = None,
    use_start_end_offsets: bool = False,
    window_size: int | None = None,
    broadcast_q: bool = False,
    activation: str = "fast_gelu",
    enable_persistent: bool = False,
    enable_tma: bool = False,
    enable_ws: bool = False,
    use_dq_atomic_add: bool = False,
    total_num_objects: int | None = None,
    bwd_opt_tech: str = "base",
    cpu_query_offset: torch.Tensor | None = None,
    use_on_device_tma: bool = False,
    ensemble_activation_list: List[int] | None = None,
    fused_rms_norm: bool = False,
    rms_norm_weight: torch.Tensor | None = None,
    fused_residual_add: bool = False,
    residual: torch.Tensor | None = None,
    fused_q_residual_add: bool = False,
    fused_layernorm: bool = False,
    layernorm_weight: torch.Tensor | None = None,
    layernorm_bias: torch.Tensor | None = None,
):
    if not broadcast_q:
        o = torch.zeros_like(query)
    else:
        BATCH = key_offset.size(0) - 1
        o = torch.zeros(
            (BATCH * query.shape[0], query.shape[1], query.shape[2]),
            dtype=query.dtype,
            device=query.device,
        )
    rrms_out = torch.empty(
        o.shape[0] if fused_rms_norm else 0,
        dtype=torch.float32,
        device=query.device,
    )
    store_rms_norm_out = (
        fused_rms_norm
        and fused_residual_add
        and not (fused_q_residual_add and fused_layernorm)
    )
    if store_rms_norm_out:
        rms_norm_out = torch.empty_like(o)
    else:
        rms_norm_out = torch.empty(0, dtype=query.dtype, device=query.device)
    ln_mean_out = torch.empty(
        o.shape[0] if fused_layernorm else 0, dtype=torch.float32, device=query.device
    )
    ln_rstd_out = torch.empty(
        o.shape[0] if fused_layernorm else 0, dtype=torch.float32, device=query.device
    )
    return o, rrms_out, rms_norm_out, ln_mean_out, ln_rstd_out


@torch.library.register_fake("ads_mkl::tlx_gdpa_megakernel_backward")
def __(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    BATCH: int,
    N_HEAD: int,
    Q_GROUP: int,
    N_CTX: int,
    N_CTX_KV: int,
    BLOCK_D: int,
    HEAD_DIM: int,
    fused_qkv: bool = False,
    fused_kv: bool = False,
    qk_scale: float | None = None,
    sort_by_seq_length: bool = False,
    seq_index: torch.Tensor | None = None,
    output_offset: torch.Tensor | None = None,
    use_start_end_offsets: bool = False,
    window_size: int | None = None,
    broadcast_q: bool = False,
    activation_enum_int: int = 0,
    ensemble_activation_list: List[int] | None = None,
    fused_rms_norm: bool = False,
    rrms_out: torch.Tensor | None = None,
    rms_norm_weight: torch.Tensor | None = None,
    fused_residual_add: bool = False,
    fused_q_residual_add: bool = False,
    fused_layernorm: bool = False,
    layernorm_weight: torch.Tensor | None = None,
    layernorm_bias: torch.Tensor | None = None,
    ln_mean_out: torch.Tensor | None = None,
    ln_rstd_out: torch.Tensor | None = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    drms_norm_weight_out = None
    if fused_rms_norm and rms_norm_weight is not None:
        drms_norm_weight_out = torch.zeros(
            N_HEAD * HEAD_DIM, device=q.device, dtype=torch.float32
        )
    dlayernorm_weight_out = None
    dlayernorm_bias_out = None
    if fused_layernorm and layernorm_weight is not None:
        dlayernorm_weight_out = torch.zeros(
            N_HEAD * HEAD_DIM, device=q.device, dtype=torch.float32
        )
    if fused_layernorm and layernorm_bias is not None:
        dlayernorm_bias_out = torch.zeros(
            N_HEAD * HEAD_DIM, device=q.device, dtype=torch.float32
        )
    return (
        torch.zeros_like(q),
        torch.zeros_like(k),
        torch.zeros_like(v),
        drms_norm_weight_out,
        dlayernorm_weight_out,
        dlayernorm_bias_out,
    )


@torch.fx.wrap
def tlx_gdpa_megakernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_offset: torch.Tensor,
    key_offset: torch.Tensor,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    ad_to_request_offset: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    attn_offset: torch.Tensor | None = None,
    is_causal: bool = False,
    qk_scale: float | None = None,
    seq_index: torch.Tensor | None = None,
    allow_tf32: bool = True,
    output_offset: torch.Tensor | None = None,
    use_start_end_offsets: bool = False,
    window_size: int | None = None,
    broadcast_q: bool = False,
    activation: str = "fast_gelu",
    enable_persistent: bool = False,
    enable_tma: bool = False,
    enable_ws: bool = False,
    use_dq_atomic_add: bool = False,
    total_num_objects: int | None = None,
    bwd_opt_tech: str = "base",
    cpu_query_offset: torch.Tensor | None = None,
    use_on_device_tma: bool = False,
    ensemble_activation_list: List[int] | None = None,
    fused_rms_norm: bool = False,
    rms_norm_weight: torch.Tensor | None = None,
    fused_residual_add: bool = False,
    residual: torch.Tensor | None = None,
    fused_q_residual_add: bool = False,
    fused_layernorm: bool = False,
    layernorm_weight: torch.Tensor | None = None,
    layernorm_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Perform Flash attention for jagged tensors. Typically offsets are of the
    format [0, start1, start2, start3, ...] where start(i+1)=end(i). To support paradigms like
    shared sequence, we support `use_start_end_offsets` which would expect offsets array like [start0, end0, start1, end1, ...],
    thus supporting overlapping offsets. Currently query offsets need to be non-overlapping as output
    shape is same as query shape.
    """
    # [T199203647] The jit.trace() and .script() dispatching logic is used to handle the case of JIT C++ TorchScript runtime,
    # where `torch._library.triton_op` does not yet support it. #This manual dispatch will no longer be necessary
    # once `triton_op` supports TorchScript fallback.
    if torch.jit.is_tracing() or torch.jit.is_scripting():
        return cpu_generalized_dot_product_attention(
            query=query,
            key=key,
            value=value,
            query_offset=query_offset,
            key_offset=key_offset,
            max_seq_len_q=max_seq_len_q,
            max_seq_len_kv=max_seq_len_kv,
            ad_to_request_offset=ad_to_request_offset,
            attn_mask=attn_mask,
            attn_offset=attn_offset,
            is_causal=is_causal,
            qk_scale=qk_scale,
            seq_index=seq_index,
            allow_tf32=allow_tf32,
            activation=activation,
        )
    query = expect_contiguous(query)
    key = expect_contiguous(key)
    value = expect_contiguous(value)
    return _tlx_generalized_dot_product_attention(
        query,
        key,
        value,
        query_offset,
        key_offset,
        max_seq_len_q,
        max_seq_len_kv,
        ad_to_request_offset,
        attn_mask,
        attn_offset,
        is_causal,
        qk_scale,
        seq_index,
        allow_tf32,
        output_offset,
        use_start_end_offsets,
        window_size,
        broadcast_q,
        activation,
        enable_persistent,
        enable_tma,
        enable_ws,
        use_dq_atomic_add,
        total_num_objects,
        bwd_opt_tech,
        cpu_query_offset,
        use_on_device_tma,
        ensemble_activation_list,
        fused_rms_norm,
        rms_norm_weight,
        fused_residual_add,
        residual,
        fused_q_residual_add,
        fused_layernorm,
        layernorm_weight,
        layernorm_bias,
    )[0]  # Return only o, rrms_out is saved internally for backward
