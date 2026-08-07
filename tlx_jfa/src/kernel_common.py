# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors

"""JIT helpers shared by the forward kernel and both backward kernels."""

import triton  # @manual=//triton:triton
import triton.language as tl  # @manual=//triton:triton


@triton.jit  # pragma: no cover
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS_KV):
    bufIdx = accum_cnt % NUM_BUFFERS_KV
    phase = (accum_cnt // NUM_BUFFERS_KV) & 1
    return bufIdx, phase


@triton.jit  # pragma: no cover
def bwd_lookup_offsets(
    tile_to_batch,
    tile_to_head,
    tile_to_block,
    G,
    tile_idx,
    BROADCAST_Q: tl.constexpr,
):
    off_z = tl.load(tile_to_batch + tile_idx)
    off_h = tl.load(tile_to_head + tile_idx)
    pid = tl.load(tile_to_block + tile_idx)
    off_h_kv = off_h // G
    if BROADCAST_Q:
        off_q_z = 0
    else:
        off_q_z = off_z
    return off_z, off_h, off_h_kv, off_q_z, pid


@triton.jit  # pragma: no cover
def bwd_calculate_num_steps(
    qlen,
    start_n,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
):
    start_m = 0
    num_steps = tl.cdiv((qlen - start_m), BLOCK_M1)
    if WINDOW_SIZE is not None:
        start_m = (max(start_m, start_n - WINDOW_SIZE) // BLOCK_M1) * BLOCK_M1
        end_m_inner = (
            tl.cdiv(min(qlen, start_n + BLOCK_N1 + WINDOW_SIZE), BLOCK_M1) * BLOCK_M1
        )
        num_steps = (end_m_inner - start_m) // BLOCK_M1
    return num_steps.to(tl.int32), start_m


@triton.jit
def _reduce_or(x, y):
    return x | y


@triton.jit  # pragma: no cover
def _split_n(x, SPLIT_FACTOR: tl.constexpr):
    if SPLIT_FACTOR == 1:
        return (x,)
    else:
        x0, x1 = x.reshape([x.shape[0], 2, x.shape[1] // 2]).permute(0, 2, 1).split()
        return _split_n(x0, SPLIT_FACTOR // 2) + _split_n(x1, SPLIT_FACTOR // 2)


@triton.jit
def _join_n(xs):
    if len(xs) == 1:
        return xs[0]
    else:
        x0 = _join_n(xs[: len(xs) // 2])
        x1 = _join_n(xs[len(xs) // 2 :])
        x = tl.join(x0, x1).permute(0, 2, 1).reshape([x0.shape[0], x0.shape[1] * 2])
        return x
