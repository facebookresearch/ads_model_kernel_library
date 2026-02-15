# @nolint
# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# [2025-07-04] Version in Cute-DSL, for Hopper and Blackwell. You'd need to install nvidia-cutlass-dsl==4.1.0.

# Supported features:
# - BF16 & FP16 dtype
# - noncausal & causal attention
# - MHA, GQA, MQA
# - hdim 64, 96, 128.
# - (hdim_qk, hdim_v) = (192, 128) for Blackwell (i.e. DeepSeek shape)
# - varlen
# - sliding window
# - bwd pass for Ampere (will also run on Hopper/Blackwell, but will be slow)
# - FP8 forward

# Features not supported yet:
# - split (i.e. FlashDecoding)
# - tuned block sizes
# - paged KV
# - append KV to existing KV cache
# - bwd pass optimized for Hopper/Blackwell
# - FP8 bwd pass
# pyre-ignore-all-errors
import math
from typing import Generator, List, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from ads_mkl.ops.cute_dsl.gdpa.src import utils
from ads_mkl.ops.cute_dsl.gdpa.src.flash_bwd_postprocess import (
    FlashAttentionBackwardPostprocess,
)
from ads_mkl.ops.cute_dsl.gdpa.src.flash_bwd_preprocess import (
    FlashAttentionBackwardPreprocess,
)
from ads_mkl.ops.cute_dsl.gdpa.src.flash_bwd_sm100 import FlashAttentionBackwardSm100
from ads_mkl.ops.cute_dsl.gdpa.src.flash_fwd_combine import FlashAttentionForwardCombine
from ads_mkl.ops.cute_dsl.gdpa.src.flash_fwd_sm100 import FlashAttentionForwardSm100
from cutlass.cute.runtime import from_dlpack
from torch.utils.flop_counter import (
    _unpack_flash_attention_nested_shapes,
    register_flop_formula,
    sdpa_backward_flop_count,
    sdpa_flop_count,
)


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


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


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.uint8: cutlass.Float8E4M3FN,  # FP8 E4M3 (legacy)
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,  # PyTorch native FP8 E4M3
}


def get_fp8_dtype(tensor: torch.Tensor, fp8_format: str = "e4m3"):
    """Get FP8 CUTE dtype based on format hint."""
    if tensor.dtype != torch.uint8:
        return None
    if fp8_format.lower() == "e4m3":
        return cutlass.Float8E4M3FN
    elif fp8_format.lower() == "e5m2":
        return cutlass.Float8E5M2
    else:
        return cutlass.Float8E4M3FN  # default


# pyre-ingore
def precompute_for_varline(jagged_tensor, cu_seqlens, m_block_size=128, is_bwd=False):
    num_head = jagged_tensor.shape[-2]
    batch_size = cu_seqlens.shape[0] - 1
    total_len = jagged_tensor.shape[0]

    q_stage = FlashAttentionForwardSm100.q_stage if not is_bwd else 1
    tile_size = m_block_size * q_stage * 1.0
    total_tiles = int(
        ((total_len + batch_size * (tile_size - 1)) // tile_size) * num_head
    )
    seq_q = torch.diff(cu_seqlens)
    block_q = ((seq_q + tile_size - 1) // tile_size) * num_head
    cu_block_q = block_q.cumsum(0, dtype=torch.int32)
    # Making `arange(batch_size + 1)`  and `to_fill_block_q` is to avoid the stream sync introduced by `repeat_interleave`.
    to_fill_block_q = torch.cat(
        [
            cu_block_q,
            torch.full((1,), total_tiles, dtype=torch.int32, device=cu_block_q.device),
        ]
    ) - torch.cat(
        [torch.full((1,), 0, dtype=torch.int32, device=cu_block_q.device), cu_block_q]
    )
    tile_to_batch = torch.arange(
        batch_size + 1, device=seq_q.device, dtype=torch.int32
    ).repeat_interleave(to_fill_block_q, output_size=total_tiles)
    return cu_block_q, tile_to_batch


# pyre-ingore
def precompute_for_varline_load_balancing(
    q_tensor,
    q_cu_seqlens,
    k_cu_seqlens,
    m_block_size=128,
    n_block_size=128,
    is_bwd=False,
    sm_count=144,
):
    num_head = q_tensor.shape[-2]
    batch_size = q_cu_seqlens.shape[0] - 1
    q_tile_size = m_block_size * FlashAttentionForwardSm100.q_stage * 1
    k_tile_size = n_block_size * 1

    main_cu_seqlens = k_cu_seqlens if is_bwd else q_cu_seqlens
    workload_cu_seqlens = q_cu_seqlens if is_bwd else k_cu_seqlens
    main_tile_size = k_tile_size if is_bwd else q_tile_size
    workload_tile_size = q_tile_size if is_bwd else k_tile_size

    main_seqlens = torch.diff(main_cu_seqlens)
    tiles_per_example = (main_seqlens + main_tile_size - 1) // main_tile_size

    workload_seqlens = torch.diff(workload_cu_seqlens)
    workloads_per_example = (
        workload_seqlens + workload_tile_size - 1
    ) // workload_tile_size

    batch_indices = torch.arange(
        batch_size, dtype=torch.int32, device=main_cu_seqlens.device
    )
    head_indices = torch.arange(
        num_head, dtype=torch.int32, device=main_cu_seqlens.device
    )

    # Generate all (batch, head, tile_idx) combinations
    batch_expanded = torch.repeat_interleave(
        batch_indices, num_head * tiles_per_example
    )
    head_expanded = head_indices.repeat(tiles_per_example.sum())
    tile_offsets = torch.arange(
        tiles_per_example.sum(), dtype=torch.int32, device=main_cu_seqlens.device
    )
    cumsum_tiles = tiles_per_example.cumsum(0, dtype=torch.int32)
    tile_start_indices = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=main_cu_seqlens.device),
            cumsum_tiles[:-1],
        ]
    )
    tile_local_indices = tile_offsets - torch.repeat_interleave(
        tile_start_indices, tiles_per_example
    )
    tile_local_indices = tile_local_indices.repeat_interleave(num_head)
    # Workload for each tile
    workload_expanded = torch.repeat_interleave(
        workloads_per_example, num_head * tiles_per_example
    )
    # Sort by workload in descending order
    sorted_indices = torch.argsort(workload_expanded, descending=True)
    # Dispatch to SMs using zigzag pattern
    num_sm = sm_count
    total_tiles = len(sorted_indices)
    tile_indices = torch.arange(total_tiles)
    cycle = tile_indices // num_sm
    pos_in_cycle = tile_indices % num_sm
    zigzag_indices = (
        torch.where(cycle % 2 == 0, pos_in_cycle, num_sm - 1 - pos_in_cycle)
        + cycle * num_sm
    )
    if total_tiles % num_sm != 0:
        zigzag_indices[-(total_tiles % num_sm) :] = torch.arange(
            total_tiles % num_sm + total_tiles // num_sm * num_sm - 1,
            total_tiles // num_sm * num_sm - 1,
            -1,
            device=main_cu_seqlens.device,
            dtype=torch.int32,
        )
    sorted_indices = sorted_indices[zigzag_indices]
    tile_to_batch = batch_expanded[sorted_indices]
    tile_to_head = head_expanded[sorted_indices]
    tile_to_block = tile_local_indices[sorted_indices]

    return tile_to_batch, tile_to_head, tile_to_block


def _extract_sequence_dimensions(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    page_table: Optional[torch.Tensor],
):
    num_head, head_dim = q.shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q.shape[0]
    if page_table is not None:
        assert cu_seqlens_k is None, "page_table is not supported with cu_seqlens_k"
        assert page_table.dtype == torch.int32, "page_table must be int32"
        assert page_table.stride(-1) == 1, (
            "page_table must be contiguous in the last dimension"
        )
        max_num_pages_per_seq = page_table.shape[1]
        assert page_table.shape == (batch_size, max_num_pages_per_seq)
        num_pages, page_size = k.shape[:2]
        seqlen_k = num_pages * page_size
    else:
        num_pages, page_size = None, None
        seqlen_k = k.shape[-3]
    return batch_size, seqlen_q, total_q, seqlen_k, num_pages, page_size


def _validate_flash_attn_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    seqlen_k: int,
    num_head: int,
    num_head_kv: int,
    head_dim: int,
    head_dim_v: int,
    num_pages: Optional[int],
    page_size: Optional[int],
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    page_table: Optional[torch.Tensor],
    learnable_sink: Optional[torch.Tensor],
):
    if cu_seqlens_k is None:
        if page_table is None:
            assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
            assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
        else:
            assert k.shape == (num_pages, page_size, num_head_kv, head_dim)
            assert v.shape == (num_pages, page_size, num_head_kv, head_dim_v)
    else:
        assert k.shape == (seqlen_k, num_head_kv, head_dim)
        assert v.shape == (seqlen_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (batch_size + 1,), (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )
    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (batch_size + 1,), (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )
    assert seqused_q is None or seqused_q.shape == (batch_size,), (
        "seqused_q must have shape (batch_size,)"
    )
    assert seqused_k is None or seqused_k.shape == (batch_size,), (
        "seqused_k must have shape (batch_size,)"
    )
    assert q.dtype in [
        torch.float16,
        torch.bfloat16,
        torch.uint8,  # FP8 E4M3 (legacy)
        torch.float8_e4m3fn,  # PyTorch native FP8 E4M3
    ], "inputs must be float16, bfloat16, uint8 (fp8), or float8_e4m3fn"
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
    for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k]:
        if t is not None:
            assert t.dtype == torch.int32, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be int32"
            )
            assert t.stride(0) == 1, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be contiguous"
            )
    if learnable_sink is not None:
        assert learnable_sink.shape == (num_head,)
        assert learnable_sink.dtype == torch.bfloat16, "learnable_sink must be bfloat16"
    assert all(
        t is None or t.is_cuda
        for t in (
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            page_table,
            learnable_sink,
        )
    ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    assert head_dim <= 256, "head_dim must be less than or equal to 256"
    alignment = 16 // q.element_size()
    assert head_dim % alignment == 0, f"head_dim must be divisible by {alignment}"
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"


def _process_window_and_causal(
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
):
    if causal:
        window_size_right = 0
    local = window_size_left is not None or window_size_right is not None
    if window_size_left is not None or window_size_right is not None:
        if window_size_left is None and window_size_right == 0:
            causal, local = True, False
        else:
            causal, local = False, True
    return causal, local, window_size_right


def _adjust_block_sizes_for_compute_capability(
    compute_capability: int,
    head_dim: int,
    head_dim_v: int,
    causal: bool,
    local: bool,
    n_block_size: int,
    pack_gqa: bool,
    qhead_per_kvhead: int,
    cu_seqlens_q: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
):
    if compute_capability == 9:  # TODO: tune block size according to hdim
        if head_dim == head_dim_v == 128 and not causal and not local:
            n_block_size = 192
    if compute_capability == 10:
        # TODO: fix the varlen case
        if (
            pack_gqa
            and (128 % qhead_per_kvhead != 0)
            or (cu_seqlens_q is not None or seqused_q is not None)
        ):
            pack_gqa = False
    return n_block_size, pack_gqa


def _prepare_cute_tensors_and_blockscaling(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor],
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    learnable_sink: Optional[torch.Tensor],
    page_table: Optional[torch.Tensor],
    tile_to_batch_q: Optional[torch.Tensor],
    tile_to_head_q: Optional[torch.Tensor],
    tile_to_block_q: Optional[torch.Tensor],
    sfq: Optional[torch.Tensor],
    sfk: Optional[torch.Tensor],
    sfv: Optional[torch.Tensor],
    cu_seqlens_sf_q: Optional[torch.Tensor],
    cu_seqlens_sf_k: Optional[torch.Tensor],
):
    dtype = torch2cute_dtype_map[q.dtype]
    q_tensor, k_tensor, v_tensor, o_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=t.ndim - 1
        )
        for t in (q, k, v, out)
    ]
    lse_tensor = (
        from_dlpack(lse.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=lse.ndim - 1
        )
        if lse is not None
        else None
    )
    (
        cu_seqlens_q_tensor,
        cu_seqlens_k_tensor,
        seqused_q_tensor,
        seqused_k_tensor,
        learnable_sink_tensor,
    ) = [
        from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)
        if t is not None
        else None
        for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
    ]
    page_table_tensor = (
        from_dlpack(page_table.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=1
        )
        if page_table is not None
        else None
    )

    tile_to_batch_tensor, tile_to_head_tensor, tile_to_block_tensor = [
        from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)
        if t is not None
        else None
        for t in (tile_to_batch_q, tile_to_head_q, tile_to_block_q)
    ]

    # Handle blockscaling: if scale factor tensors are provided, enable blockscaling
    blockscaled = sfq is not None and sfk is not None
    sfq_tensor = (
        from_dlpack(sfq.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=sfq.ndim - 1
        )
        if sfq is not None
        else None
    )
    sfk_tensor = (
        from_dlpack(sfk.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=sfk.ndim - 1
        )
        if sfk is not None
        else None
    )
    sfv_tensor = (
        from_dlpack(sfv.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=sfv.ndim - 1
        )
        if sfv is not None
        else None
    )

    # For blockscaled varlen: require pre-computed 128-aligned cu_seqlens for SF offsets
    # Use ads_mkl.ops.utils.mxfp8_utils.quantize_varlen_mxfp8_sf_only to generate these.
    if blockscaled and cu_seqlens_q is not None:
        assert cu_seqlens_sf_q is not None, (
            "For blockscaled varlen attention, cu_seqlens_sf_q must be provided. "
            "Use ads_mkl.ops.utils.mxfp8_utils.quantize_varlen_mxfp8_sf_only() to generate 128-aligned SF offsets."
        )
    if blockscaled and cu_seqlens_k is not None:
        assert cu_seqlens_sf_k is not None, (
            "For blockscaled varlen attention, cu_seqlens_sf_k must be provided. "
            "Use ads_mkl.ops.utils.mxfp8_utils.quantize_varlen_mxfp8_sf_only() to generate 128-aligned SF offsets."
        )

    # Convert SF cu_seqlens to cute tensors
    cu_seqlens_sf_q_tensor = (
        from_dlpack(cu_seqlens_sf_q.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=0
        )
        if cu_seqlens_sf_q is not None
        else None
    )
    cu_seqlens_sf_k_tensor = (
        from_dlpack(cu_seqlens_sf_k.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=0
        )
        if cu_seqlens_sf_k is not None
        else None
    )

    # Compute total_sf_q/k for SF-only approach (total padded tokens for SF layout)
    # These are used by the kernel to create SF layouts with correct dimensions
    total_sf_q = (
        int(cu_seqlens_sf_q[-1].item()) if cu_seqlens_sf_q is not None else None
    )
    total_sf_k = (
        int(cu_seqlens_sf_k[-1].item()) if cu_seqlens_sf_k is not None else None
    )

    return (
        dtype,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        cu_seqlens_q_tensor,
        cu_seqlens_k_tensor,
        seqused_q_tensor,
        seqused_k_tensor,
        learnable_sink_tensor,
        page_table_tensor,
        tile_to_batch_tensor,
        tile_to_head_tensor,
        tile_to_block_tensor,
        blockscaled,
        sfq_tensor,
        sfk_tensor,
        sfv_tensor,
        cu_seqlens_sf_q_tensor,
        cu_seqlens_sf_k_tensor,
        total_sf_q,
        total_sf_k,
    )


def _compile_flash_attn_kernel(
    compile_key,
    compile_cache,
    compute_capability: int,
    page_size: Optional[int],
    head_dim: int,
    head_dim_v: int,
    qhead_per_kvhead: int,
    causal: bool,
    local: bool,
    pack_gqa: bool,
    cu_seqlens_q: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    prefer_persistent: bool,
    activation: str,
    blockscaled: bool,
    sf_vec_size: int,
    q_tensor,
    k_tensor,
    v_tensor,
    o_tensor,
    lse_tensor,
    softmax_scale: float,
    current_stream,
    cu_seqlens_q_tensor,
    cu_seqlens_k_tensor,
    seqused_q_tensor,
    seqused_k_tensor,
    max_seqlen_q: Optional[int],
    page_table_tensor,
    softcap: Optional[float],
    window_size_left: Optional[int],
    window_size_right: Optional[int],
    learnable_sink_tensor,
    tile_to_batch_tensor,
    tile_to_head_tensor,
    tile_to_block_tensor,
    sfq_tensor,
    sfk_tensor,
    sfv_tensor,
    cu_seqlens_sf_q_tensor,
    cu_seqlens_sf_k_tensor,
    total_sf_q: Optional[int],
    total_sf_k: Optional[int],
):
    if compile_key not in compile_cache:
        if compute_capability == 10:
            assert page_size in [None, 128], (
                "Only page_size=128 is supported for paged KV on SM 10.0"
            )
            fa_fwd = FlashAttentionForwardSm100(
                head_dim,
                head_dim_v,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                is_persistent=not causal
                and not local
                and ((cu_seqlens_q is None and seqused_q is None) or prefer_persistent),
                activation=activation,
                blockscaled=blockscaled,
                sf_vec_size=sf_vec_size,
            )
        else:
            raise ValueError(
                f"Unsupported compute capability: {compute_capability}. Supported: 9.x, 10.x"
            )
        # TODO: check @can_implement
        compile_cache[compile_key] = cute.compile(
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            current_stream,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            max_seqlen_q,
            page_table_tensor,
            softcap,
            window_size_left,
            window_size_right,
            learnable_sink_tensor,
            tile_to_batch_tensor,
            tile_to_head_tensor,
            tile_to_block_tensor,
            sfq_tensor,
            sfk_tensor,
            sfv_tensor,
            cu_seqlens_sf_q_tensor,
            cu_seqlens_sf_k_tensor,
            total_sf_q,
            total_sf_k,
        )


def _flash_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    # m_block_size: int = 128,
    # n_block_size: int = 64,
    # num_threads: int = 128,
    m_block_size: int = 128,
    n_block_size: int = 128,
    num_threads: int = 384,
    pack_gqa: Optional[bool] = None,
    _compute_capability: Optional[int] = None,
    prefer_persistent: Optional[bool] = None,
    tile_to_batch_q: Optional[torch.Tensor] = None,
    tile_to_head_q: Optional[torch.Tensor] = None,
    tile_to_block_q: Optional[torch.Tensor] = None,
    activation: str = "fast_gelu",
    sfq: Optional[torch.Tensor] = None,
    sfk: Optional[torch.Tensor] = None,
    sfv: Optional[torch.Tensor] = None,
    sf_vec_size: int = 32,
    cu_seqlens_sf_q: Optional[torch.Tensor] = None,
    cu_seqlens_sf_k: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    batch_size, seqlen_q, total_q, seqlen_k, num_pages, page_size = (
        _extract_sequence_dimensions(q, k, cu_seqlens_q, cu_seqlens_k, page_table)
    )
    num_head, head_dim = q.shape[-2:]
    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]
    _validate_flash_attn_inputs(
        q,
        k,
        v,
        batch_size,
        seqlen_k,
        num_head,
        num_head_kv,
        head_dim,
        head_dim_v,
        num_pages,
        page_size,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        page_table,
        learnable_sink,
    )
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if softcap == 0.0:
        softcap = None
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

    out_torch_dtype = q.dtype if q.dtype != torch.float8_e4m3fn else torch.bfloat16
    device = q.device
    q_batch_seqlen_shape = (
        (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)
    )
    out = torch.empty(
        *q_batch_seqlen_shape,
        num_head,
        head_dim_v,
        dtype=out_torch_dtype,
        device=device,
    )
    lse_shape = (
        (batch_size, num_head, seqlen_q)
        if cu_seqlens_q is None
        else (num_head, total_q)
    )
    requires_grad = q.requires_grad or k.requires_grad or v.requires_grad
    lse = (
        torch.empty(lse_shape, dtype=torch.float32, device=device)
        if requires_grad
        else None
    )

    causal, local, window_size_right = _process_window_and_causal(
        causal, window_size_left, window_size_right
    )
    compute_capability = (
        torch.cuda.get_device_capability()[0]
        if _compute_capability is None
        else _compute_capability
    )
    assert compute_capability in [9, 10], (
        "Unsupported compute capability. Supported: 9.x, 10.x"
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    n_block_size, pack_gqa = _adjust_block_sizes_for_compute_capability(
        compute_capability,
        head_dim,
        head_dim_v,
        causal,
        local,
        n_block_size,
        pack_gqa,
        qhead_per_kvhead,
        cu_seqlens_q,
        seqused_q,
    )

    (
        dtype,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        cu_seqlens_q_tensor,
        cu_seqlens_k_tensor,
        seqused_q_tensor,
        seqused_k_tensor,
        learnable_sink_tensor,
        page_table_tensor,
        tile_to_batch_tensor,
        tile_to_head_tensor,
        tile_to_block_tensor,
        blockscaled,
        sfq_tensor,
        sfk_tensor,
        sfv_tensor,
        cu_seqlens_sf_q_tensor,
        cu_seqlens_sf_k_tensor,
        total_sf_q,
        total_sf_k,
    ) = _prepare_cute_tensors_and_blockscaling(
        q,
        k,
        v,
        out,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        learnable_sink,
        page_table,
        tile_to_batch_q,
        tile_to_head_q,
        tile_to_block_q,
        sfq,
        sfk,
        sfv,
        cu_seqlens_sf_q,
        cu_seqlens_sf_k,
    )

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        softcap is not None,
        lse is None,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        seqused_q is None,
        seqused_k is None,
        page_table is not None,
        window_size_left is not None,
        window_size_right is not None,
        learnable_sink is not None,
        m_block_size,
        n_block_size,
        num_threads,
        pack_gqa,
        compute_capability,
        activation,
        blockscaled,
        sf_vec_size,
        cu_seqlens_sf_q is not None,
        cu_seqlens_sf_k is not None,
    )
    _compile_flash_attn_kernel(
        compile_key,
        _flash_attn_fwd.compile_cache,
        compute_capability,
        page_size,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        local,
        pack_gqa,
        cu_seqlens_q,
        seqused_q,
        prefer_persistent,
        activation,
        blockscaled,
        sf_vec_size,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        softmax_scale,
        current_stream,
        cu_seqlens_q_tensor,
        cu_seqlens_k_tensor,
        seqused_q_tensor,
        seqused_k_tensor,
        max_seqlen_q,
        page_table_tensor,
        softcap,
        window_size_left,
        window_size_right,
        learnable_sink_tensor,
        tile_to_batch_tensor,
        tile_to_head_tensor,
        tile_to_block_tensor,
        sfq_tensor,
        sfk_tensor,
        sfv_tensor,
        cu_seqlens_sf_q_tensor,
        cu_seqlens_sf_k_tensor,
        total_sf_q,
        total_sf_k,
    )
    _flash_attn_fwd.compile_cache[compile_key](
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        softmax_scale,
        current_stream,
        cu_seqlens_q_tensor,
        cu_seqlens_k_tensor,
        seqused_q_tensor,
        seqused_k_tensor,
        max_seqlen_q,
        page_table_tensor,
        softcap,
        window_size_left,
        window_size_right,
        learnable_sink_tensor,
        tile_to_batch_tensor,
        tile_to_head_tensor,
        tile_to_block_tensor,
        sfq_tensor,
        sfk_tensor,
        sfv_tensor,
        cu_seqlens_sf_q_tensor,
        cu_seqlens_sf_k_tensor,
        total_sf_q,
        total_sf_k,
    )
    return out, lse


_flash_attn_fwd.compile_cache = {}


def _flash_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: float = 0.0,
    m_block_size: int = 64,
    n_block_size: int = 128,
    num_threads: int = 256,
    pack_gqa: bool = False,
    num_stages_Q: int = 2,
    num_stages_dO: int = 2,
    SdP_swapAB: bool = False,
    dKV_swapAB: bool = False,
    dQ_swapAB: bool = False,
    AtomLayoutMSdP: int = 2,
    AtomLayoutNdKV: int = 2,
    AtomLayoutMdQ: int = 2,
    V_in_regs: bool = False,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_k: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    tile_to_batch_k: Optional[torch.Tensor] = None,
    tile_to_head_k: Optional[torch.Tensor] = None,
    tile_to_block_k: Optional[torch.Tensor] = None,
    prefer_persistent: bool = False,
    activation: str = "fast_gelu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    compute_capability = torch.cuda.get_device_capability()[0]
    assert compute_capability in [9, 10], (
        "Unsupported compute capability. Supported: 9.x, 10.x"
    )

    if compute_capability == 9:
        m_block_size = 80 if not causal else 64
        n_block_size = 128
        num_stages_Q = 2
        num_stages_dO = 2
        num_stages_PdS = 2
        SdP_swapAB = True
        dKV_swapAB = False
        dQ_swapAB = not causal
        AtomLayoutMSdP = 1
        AtomLayoutNdKV = 2
        AtomLayoutMdQ = 1
    else:
        m_block_size = 128
        n_block_size = 128
        dQ_swapAB = False
        AtomLayoutMdQ = 1
    q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = [
        maybe_contiguous(t)
        for t in (
            q,
            k,
            v,
            out,
            dout,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
        )
    ]
    num_head, head_dim = q.shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q.shape[0]

    if cu_seqlens_k is None:
        batch_size, seqlen_k = k.shape[:2]
        total_k = batch_size * seqlen_k
    else:
        batch_size = cu_seqlens_k.shape[0] - 1
        seqlen_k = None
        total_k = k.shape[0]

    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]

    if cu_seqlens_k is None:
        assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
        assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
    else:
        assert k.shape == (total_k, num_head_kv, head_dim)
        assert v.shape == (total_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (batch_size + 1,), (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (batch_size + 1,), (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )

        assert out.shape == (total_q, num_head, head_dim_v)
        assert dout.shape == (total_q, num_head, head_dim_v)
        assert lse.shape == (num_head, total_q), (
            "lse must have shape (num_head, total_q)"
        )
    else:
        assert out.shape == (batch_size, seqlen_q, num_head, head_dim_v)
        assert dout.shape == (batch_size, seqlen_q, num_head, head_dim_v)
        assert lse.shape == (batch_size, num_head, seqlen_q), (
            "lse must have shape (batch_size, num_head, seqlen_q)"
        )

    assert q.dtype in [
        torch.float16,
        torch.bfloat16,
        torch.uint8,  # FP8 E4M3 (legacy)
        torch.float8_e4m3fn,  # PyTorch native FP8 E4M3
    ], "inputs must be float16, bfloat16, uint8 (fp8), or float8_e4m3fn"
    assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype, (
        "inputs must have the same dtype"
    )
    for t in [cu_seqlens_q, cu_seqlens_k]:
        if t is not None:
            assert t.dtype == torch.int32, "cu_seqlens_q, cu_seqlens_k must be int32"
    assert lse.dtype == torch.float32, "lse must be float32"
    assert all(
        t is None or t.is_cuda
        for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k)
    ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    assert head_dim <= 256, "head_dim must be less than or equal to 256"
    alignment = 16 // q.element_size()
    assert head_dim % alignment == 0, f"head_dim must be divisible by {alignment}"
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

    device = q.device
    # TODO: check if this is the right rounding
    # For FP8, use BF16 for dQ accumulation since TMA reduce-add doesn't support FP8 atomics
    is_fp8 = q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    dq_accum_dtype = torch.bfloat16 if is_fp8 else q.dtype
    dq = torch.zeros(q.shape, dtype=dq_accum_dtype, device=device)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    head_dim_rounded = (head_dim + 32 - 1) // 32 * 32

    if cu_seqlens_q is None:
        seqlen_q_rounded = (seqlen_q + m_block_size - 1) // m_block_size * m_block_size
        dq_accum = torch.zeros(
            batch_size,
            num_head,
            seqlen_q_rounded * head_dim_rounded,
            dtype=torch.float32,
            device=device,
        )
        dpsum = torch.empty(
            batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device
        )
        lse_log2 = torch.empty(
            batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device
        )
    else:
        total_q_rounded_padded = (
            (total_q + cu_seqlens_q.shape[0] * m_block_size - 1)
            // m_block_size
            * m_block_size
        )
        dq_accum = torch.zeros(
            num_head,
            total_q_rounded_padded * head_dim_rounded,
            dtype=torch.float32,
            device=device,
        )
        dpsum = torch.empty(
            num_head, total_q_rounded_padded, dtype=torch.float32, device=device
        )
        lse_log2 = torch.empty(
            num_head, total_q_rounded_padded, dtype=torch.float32, device=device
        )

    if qhead_per_kvhead > 1:
        head_dim_v_rounded = (head_dim_v + 32 - 1) // 32 * 32
        if cu_seqlens_k is None:
            seqlen_k_rounded = (
                (seqlen_k + n_block_size - 1) // n_block_size * n_block_size
            )
            dk_accum = torch.zeros(
                batch_size,
                num_head_kv,
                seqlen_k_rounded * head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
            dv_accum = torch.zeros(
                batch_size,
                num_head_kv,
                seqlen_k_rounded * head_dim_v_rounded,
                dtype=torch.float32,
                device=device,
            )
        else:
            total_k_rounded_padded = (
                (total_k + cu_seqlens_k.shape[0] * n_block_size - 1)
                // n_block_size
                * n_block_size
            )
            dk_accum = torch.zeros(
                num_head_kv,
                total_k_rounded_padded * head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
            dv_accum = torch.zeros(
                num_head_kv,
                total_k_rounded_padded * head_dim_v_rounded,
                dtype=torch.float32,
                device=device,
            )

    dtype = torch2cute_dtype_map[q.dtype]
    (
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        do_tensor,
        dq_tensor,
        dk_tensor,
        dv_tensor,
    ) = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=t.ndim - 1
        )
        for t in (q, k, v, out, dout, dq, dk, dv)
    ]
    lse_tensor = from_dlpack(lse.detach(), assumed_align=4).mark_layout_dynamic(
        leading_dim=lse.ndim - 1
    )
    dq_accum_tensor, dpsum_tensor, lse_log2_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=t.ndim - 1
        )
        for t in (dq_accum, dpsum, lse_log2)
    ]
    if qhead_per_kvhead > 1:
        dk_accum_tensor, dv_accum_tensor = [
            from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(
                leading_dim=t.ndim - 1
            )
            for t in (dk_accum, dv_accum)
        ]
    cu_seqlens_q_tensor, cu_seqlens_k_tensor, seqused_q_tensor, seqused_k_tensor = [
        from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=t.ndim - 1
        )
        if t is not None
        else None
        for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
    ]
    tile_to_batch_k_tensor, tile_to_head_k_tensor, tile_to_block_k_tensor = [
        from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)
        if t is not None
        else None
        for t in (tile_to_batch_k, tile_to_head_k, tile_to_block_k)
    ]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Preprocess kernel: compute (o * dout).sum(dim=-1), lse * log2_e, and zero out dq_accum.
    # compile_key_pre = (dtype, head_dim_v, m_block_size, num_threads)
    # if compile_key_pre not in _flash_attn_bwd.compile_cache_pre:
    #     fa_bwd_pre = FlashAttentionBackwardPreprocess(
    #         dtype, head_dim_v, m_block_size, num_threads=num_threads,
    #     )
    #     # TODO: check @can_implement
    #     _flash_attn_bwd.compile_cache_pre[compile_key_pre] = cute.compile(
    #         fa_bwd_pre, o_tensor, do_tensor, dpsum_tensor, lse_tensor, lse_log2_tensor,
    #         dq_accum_tensor, cu_seqlens_q_tensor, seqused_q_tensor, current_stream
    #     )
    # _flash_attn_bwd.compile_cache_pre[compile_key_pre](
    #     o_tensor, do_tensor, dpsum_tensor, lse_tensor, lse_log2_tensor, dq_accum_tensor,
    #     cu_seqlens_q_tensor, seqused_q_tensor, current_stream
    # )

    # Backward kernel: compute dk, dv, dq_accum.
    compile_key = (
        compute_capability,
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        softcap != 0.0,
        m_block_size,
        n_block_size,
        num_threads,
        pack_gqa,
        activation,
    )
    num_threads = 384
    if compile_key not in _flash_attn_bwd.compile_cache:
        fa_bwd_obj = FlashAttentionBackwardSm100(
            # dtype,
            head_dim,
            head_dim_v,
            is_causal=causal,
            qhead_per_kvhead=qhead_per_kvhead,
            is_persistent=prefer_persistent,
            activation=activation,
            # tile_m=m_block_size,
            # tile_n=n_block_size,
        )
        # TODO: check @can_implement
        _flash_attn_bwd.compile_cache[compile_key] = cute.compile(
            fa_bwd_obj,
            q_tensor,
            k_tensor,
            v_tensor,
            do_tensor,
            lse_log2_tensor,
            dpsum_tensor,
            dq_tensor,
            dk_tensor if qhead_per_kvhead == 1 else dk_accum_tensor,
            dv_tensor if qhead_per_kvhead == 1 else dv_accum_tensor,
            softmax_scale,
            current_stream,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            tile_to_batch_k_tensor,
            tile_to_head_k_tensor,
            tile_to_block_k_tensor,
            max_seqlen_k,
        )

    _flash_attn_bwd.compile_cache[compile_key](
        q_tensor,
        k_tensor,
        v_tensor,
        do_tensor,
        lse_log2_tensor,
        dpsum_tensor,
        dq_tensor,
        dk_tensor if qhead_per_kvhead == 1 else dk_accum_tensor,
        dv_tensor if qhead_per_kvhead == 1 else dv_accum_tensor,
        softmax_scale,
        current_stream,
        cu_seqlens_q_tensor,
        cu_seqlens_k_tensor,
        seqused_q_tensor,
        seqused_k_tensor,
        tile_to_batch_k_tensor,
        tile_to_head_k_tensor,
        tile_to_block_k_tensor,
        max_seqlen_k,
    )

    return dq, dk, dv


_flash_attn_bwd.compile_cache_pre = {}
_flash_attn_bwd.compile_cache = {}
_flash_attn_bwd.compile_cache_post = {}


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
        pack_gqa: Optional[bool] = None,
        prefer_persistent: bool = False,
    ):
        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=learnable_sink,
            softcap=softcap,
            pack_gqa=pack_gqa,
            prefer_persistent=prefer_persistent,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.prefer_persistent = prefer_persistent
        return out, lse

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            ctx.softmax_scale,
            ctx.causal,
            ctx.softcap,
            prefer_persistent=ctx.prefer_persistent,
        )
        return dq, dk, dv, *((None,) * 20)


@torch.library.custom_op(
    "ads_mkl::cutedsl_generalized_dot_product_attention", mutates_args=()
)
def cutedsl_generalized_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    pack_gqa: Optional[bool] = None,
    prefer_persistent: Optional[bool] = None,
    tile_to_batch_q: Optional[torch.Tensor] = None,
    tile_to_head_q: Optional[torch.Tensor] = None,
    tile_to_block_q: Optional[torch.Tensor] = None,
    tile_to_batch_k: Optional[torch.Tensor] = None,
    tile_to_head_k: Optional[torch.Tensor] = None,
    tile_to_block_k: Optional[torch.Tensor] = None,
    activation: str = "fast_gelu",
    sfq: Optional[torch.Tensor] = None,
    sfk: Optional[torch.Tensor] = None,
    sfv: Optional[torch.Tensor] = None,
    sf_vec_size: int = 32,
    cu_seqlens_sf_q: Optional[torch.Tensor] = None,
    cu_seqlens_sf_k: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CuteDSL Generalized Dot Product Attention forward pass.

    This function supports blockscaled (MXFP8) attention when scale factor tensors
    are provided. If sfq and sfk are both provided, blockscaling is automatically enabled.

    Args:
        q: Query tensor (total_q, num_heads, head_dim) for varlen or (batch, seqlen, num_heads, head_dim)
        k: Key tensor (total_k, num_heads_kv, head_dim) for varlen or (batch, seqlen, num_heads_kv, head_dim)
        v: Value tensor (total_k, num_heads_kv, head_dim_v) for varlen or (batch, seqlen, num_heads_kv, head_dim_v)
        cu_seqlens_q: Cumulative sequence lengths for queries (batch_size + 1,)
        cu_seqlens_k: Cumulative sequence lengths for keys (batch_size + 1,)
        seqused_q: Used sequence lengths for queries
        seqused_k: Used sequence lengths for keys
        max_seqlen_q: Maximum sequence length for queries
        max_seqlen_k: Maximum sequence length for keys
        page_table: Page table for paged KV cache
        softmax_scale: Softmax scale factor (default: 1/sqrt(head_dim))
        causal: Whether to use causal attention
        window_size_left: Left window size for sliding window attention
        window_size_right: Right window size for sliding window attention
        learnable_sink: Learnable sink tokens
        softcap: Softmax cap value
        pack_gqa: Whether to pack GQA
        prefer_persistent: Whether to prefer persistent kernel
        tile_to_batch_q: Tile to batch mapping for queries
        tile_to_head_q: Tile to head mapping for queries
        tile_to_block_q: Tile to block mapping for queries
        tile_to_batch_k: Tile to batch mapping for keys
        tile_to_head_k: Tile to head mapping for keys
        tile_to_block_k: Tile to block mapping for keys
        activation: Activation function name (e.g., "fast_gelu", "relu", "identity")
        sfq: Scale factor tensor for Q (MXFP8 blockscaling). If provided with sfk, enables blockscaling.
        sfk: Scale factor tensor for K (MXFP8 blockscaling). If provided with sfq, enables blockscaling.
        sfv: Scale factor tensor for V (MXFP8 blockscaling). For blockscaled PV GEMM.
        sf_vec_size: Scale factor vector size (default: 32 for MXFP8)
        cu_seqlens_sf_q: 128-aligned cumulative sequence lengths for Q scale factor offsets (varlen MXFP8 only)
        cu_seqlens_sf_k: 128-aligned cumulative sequence lengths for K/V scale factor offsets (varlen MXFP8 only)

    Returns:
        Tuple of (output, lse) where output is the attention output and lse is the log-sum-exp.
    """
    out, lse = _flash_attn_fwd(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q=max_seqlen_q,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        learnable_sink=learnable_sink,
        softcap=softcap,
        pack_gqa=pack_gqa,
        prefer_persistent=prefer_persistent,
        tile_to_batch_q=tile_to_batch_q,
        tile_to_head_q=tile_to_head_q,
        tile_to_block_q=tile_to_block_q,
        activation=activation,
        sfq=sfq,
        sfk=sfk,
        sfv=sfv,
        sf_vec_size=sf_vec_size,
        cu_seqlens_sf_q=cu_seqlens_sf_q,
        cu_seqlens_sf_k=cu_seqlens_sf_k,
    )
    return out, lse


def _cutedsl_generalized_dot_product_attention_setup_context(ctx, inputs, output):
    (
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        page_table,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        learnable_sink,
        softcap,
        pack_gqa,
        prefer_persistent,
        _,
        _,
        _,
        tile_to_batch_k,
        tile_to_head_k,
        tile_to_block_k,
        activation,
        sfq,
        sfk,
        sfv,
        sf_vec_size,
        cu_seqlens_sf_q,
        cu_seqlens_sf_k,
    ) = inputs
    out, lse = output
    ctx.save_for_backward(
        q,
        k,
        v,
        out,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        tile_to_batch_k,
        tile_to_head_k,
        tile_to_block_k,
    )
    ctx.softmax_scale = softmax_scale
    ctx.causal = causal
    ctx.softcap = softcap
    ctx.max_seqlen_k = max_seqlen_k
    ctx.max_seqlen_q = max_seqlen_q
    ctx.prefer_persistent = prefer_persistent
    ctx.activation = activation


@torch.library.custom_op(
    "ads_mkl::cutedsl_generalized_dot_product_attention_backward", mutates_args=()
)
def cutedsl_generalized_dot_product_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: float = 0.0,
    m_block_size: int = 64,
    n_block_size: int = 128,
    num_threads: int = 256,
    pack_gqa: bool = False,
    num_stages_Q: int = 2,
    num_stages_dO: int = 2,
    SdP_swapAB: bool = False,
    dKV_swapAB: bool = False,
    dQ_swapAB: bool = False,
    AtomLayoutMSdP: int = 2,
    AtomLayoutNdKV: int = 2,
    AtomLayoutMdQ: int = 2,
    V_in_regs: bool = False,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_k: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    tile_to_batch_k: Optional[torch.Tensor] = None,
    tile_to_head_k: Optional[torch.Tensor] = None,
    tile_to_block_k: Optional[torch.Tensor] = None,
    prefer_persistent: bool = False,
    activation: str = "fast_gelu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output = _flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        lse,
        softmax_scale,
        causal,
        softcap,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        max_seqlen_k=max_seqlen_k,
        max_seqlen_q=max_seqlen_q,
        tile_to_batch_k=tile_to_batch_k,
        tile_to_head_k=tile_to_head_k,
        tile_to_block_k=tile_to_block_k,
        prefer_persistent=prefer_persistent,
        activation=activation,
    )
    return output


def _cutedsl_generalized_dot_product_attention_backward(ctx, dout, *args):
    (
        q,
        k,
        v,
        out,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        tile_to_batch_k,
        tile_to_head_k,
        tile_to_block_k,
    ) = ctx.saved_tensors
    # q, k, v, out, lse = ctx.saved_tensors
    dq, dk, dv = torch.ops.ads_mkl.cutedsl_generalized_dot_product_attention_backward(
        q,
        k,
        v,
        out,
        dout,
        lse,
        ctx.softmax_scale,
        ctx.causal,
        ctx.softcap,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        max_seqlen_k=ctx.max_seqlen_k,
        max_seqlen_q=ctx.max_seqlen_q,
        tile_to_batch_k=tile_to_batch_k,
        tile_to_head_k=tile_to_head_k,
        tile_to_block_k=tile_to_block_k,
        prefer_persistent=ctx.prefer_persistent,
        activation=ctx.activation,
    )
    return dq, dk, dv, *((None,) * 28)


cutedsl_generalized_dot_product_attention.register_autograd(
    _cutedsl_generalized_dot_product_attention_backward,
    setup_context=_cutedsl_generalized_dot_product_attention_setup_context,
)


@register_flop_formula(
    torch.ops.ads_mkl.cutedsl_generalized_dot_product_attention, get_raw=True
)
def generalized_dot_product_attention_forward_flop(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    pack_gqa: Optional[bool] = None,
    prefer_persistent: Optional[bool] = None,
    cu_block_q: Optional[torch.Tensor] = None,
    tile_to_batch: Optional[torch.Tensor] = None,
    *args,
    **kwargs,
) -> int:
    """Count flops for self-attention."""
    fused_qkv = k is None and v is None
    fused_kv = k is not None and v is None
    if fused_qkv:
        HEAD_DIM = q.shape[-1] // 3
        query, k, value = q.split(HEAD_DIM, dim=-1)
    elif fused_kv:
        HEAD_DIM = k.shape[-1] // 2
        k, value = k.split(HEAD_DIM, dim=-1)
    # bs_q = cu_seqlens_q.size(0) - 1
    # bs_k = cu_seqlens_k.size(0) - 1
    # if bs_q != bs_k:
    #     # broadcast q bs to k bs
    #     assert bs_k % bs_q == 0
    #     assert broadcast_q
    #     query_length = query_offset[1]
    #     query_offset = torch.arange(bs_k + 1, device=query.device) * query_length
    #     query = query.repeat_interleave(bs_k // bs_q, dim=0)
    if q.is_meta:
        sizes = _unpack_nested_shapes_meta(
            query=q,
            key=k,
            value=v,
            cum_seq_q=cu_seqlens_q,
            cum_seq_k=cu_seqlens_k,
            max_q=max_seqlen_q,
            max_k=max_seqlen_q,
        )
    else:
        sizes = _unpack_flash_attention_nested_shapes(
            query=q,
            key=k,
            value=v,
            cum_seq_q=cu_seqlens_q,
            cum_seq_k=cu_seqlens_k,
            max_q=max_seqlen_q,
            max_k=max_seqlen_q,
        )
    # if window_size is not None:
    #     # replace number of keys and values
    #     sizes = (
    #         (
    #             query_shape,
    #             (_b2, _h2, 2 * window_size + 1, _d2),
    #             (_b3, _h3, 2 * window_size + 1, d_v),
    #             grad_out_shape,
    #         )
    #         for (
    #             query_shape,
    #             (_b2, _h2, s_k, _d2),
    #             (_b3, _h3, _s3, d_v),
    #             grad_out_shape,
    #         ) in sizes
    #     )
    return sum(
        sdpa_flop_count(query_shape, key_shape, value_shape)
        for query_shape, key_shape, value_shape, _ in sizes
    )


@register_flop_formula(
    torch.ops.ads_mkl.cutedsl_generalized_dot_product_attention_backward, get_raw=True
)
def generalized_dot_product_attention_backward_flop(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: float = 0.0,
    m_block_size: int = 64,
    n_block_size: int = 128,
    num_threads: int = 256,
    pack_gqa: bool = False,
    num_stages_Q: int = 2,
    num_stages_dO: int = 2,
    SdP_swapAB: bool = False,
    dKV_swapAB: bool = False,
    dQ_swapAB: bool = False,
    AtomLayoutMSdP: int = 2,
    AtomLayoutNdKV: int = 2,
    AtomLayoutMdQ: int = 2,
    V_in_regs: bool = False,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_k: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    *args,
    **kwargs,
):
    shapes = _unpack_flash_attention_nested_shapes(
        query=q,
        key=k,
        value=v,
        grad_out=dout,
        cum_seq_q=cu_seqlens_q,
        cum_seq_k=cu_seqlens_k,
        max_q=max_seqlen_q,
        max_k=max_seqlen_k,
    )

    return sum(
        sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape)
        for query_shape, key_shape, value_shape, grad_out_shape in shapes
    )


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    pack_gqa: Optional[bool] = None,
    prefer_persistent: bool = False,
):
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        window_size,
        learnable_sink,
        softcap,
        pack_gqa,
        prefer_persistent,
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    # TODO: also need max_seqlen_k for bwd tile scheduler
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    pack_gqa: Optional[bool] = None,
    prefer_persistent: Optional[bool] = None,
    tile_to_batch_q: Optional[torch.Tensor] = None,
    tile_to_head_q: Optional[torch.Tensor] = None,
    tile_to_block_q: Optional[torch.Tensor] = None,
    tile_to_batch_k: Optional[torch.Tensor] = None,
    tile_to_head_k: Optional[torch.Tensor] = None,
    tile_to_block_k: Optional[torch.Tensor] = None,
    activation: str = "fast_gelu",
    sfq: Optional[torch.Tensor] = None,
    sfk: Optional[torch.Tensor] = None,
    sfv: Optional[torch.Tensor] = None,
    sf_vec_size: int = 32,
    cu_seqlens_sf_q: Optional[torch.Tensor] = None,
    cu_seqlens_sf_k: Optional[torch.Tensor] = None,
):
    return cutedsl_generalized_dot_product_attention(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        page_table,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        learnable_sink,
        softcap,
        pack_gqa,
        prefer_persistent,
        tile_to_batch_q=tile_to_batch_q,
        tile_to_head_q=tile_to_head_q,
        tile_to_block_q=tile_to_block_q,
        tile_to_batch_k=tile_to_batch_k,
        tile_to_head_k=tile_to_head_k,
        tile_to_block_k=tile_to_block_k,
        activation=activation,
        sfq=sfq,
        sfk=sfk,
        sfv=sfv,
        sf_vec_size=sf_vec_size,
        cu_seqlens_sf_q=cu_seqlens_sf_q,
        cu_seqlens_sf_k=cu_seqlens_sf_k,
    )


def _flash_attn_fwd_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    num_splits_dynamic_ptr: Optional[torch.Tensor] = None,
    semaphore_to_reset: Optional[torch.Tensor] = None,
) -> None:
    """Forward combine kernel for split attention computation.

    Combines partial outputs and log-sum-exp values from multiple splits
    of attention computation into final outputs.

    Args:
        out_partial: Partial outputs tensor (num_splits, batch, seqlen, nheads, headdim) or
                                            (num_splits, total_q, nheads, headdim) if there's cu_seqlens
        lse_partial: Partial LSE tensor (num_splits, batch, seqlen, nheads) or
                                       (num_splits, total_q, nheads) if there's cu_seqlens
        out: Output tensor (batch, seqlen, nheads, headdim) or (total_q, nheads, headdim) if there's cu_seqlens
        lse: Output LSE tensor (batch, seqlen, nheads) or (total_q, nheads) if there's cu_seqlens.
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        seqused: Used sequence lengths for each batch
        num_splits_dynamic_ptr: Dynamic number of splits per batch
        semaphore_to_reset: Semaphore for synchronization
        k_block_size: Block size for head dimension

    Returns:
        None
    """
    # Input validation
    assert out_partial.dim() in [4, 5], "out_partial must have 4 or 5 dimensions"
    assert lse_partial.dim() in [3, 4], "lse_partial must have 3 or 4 dimensions"
    assert out_partial.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        "out_partial must be fp16, bf16, or fp32"
    )
    assert lse_partial.dtype == torch.float32, "lse_partial must be fp32"
    assert out_partial.is_cuda and lse_partial.is_cuda, "tensors must be on CUDA device"
    assert out_partial.stride(-1) == 1, (
        "out_partial must be contiguous in the last dimension"
    )
    assert lse_partial.stride(-2) == 1, (
        "lse_partial must be contiguous in the seqlen dimension"
    )
    assert lse_partial.shape == out_partial.shape[:-1]

    # Determine if this is variable length based on dimensions
    is_varlen = out_partial.dim() == 4

    # Validate output tensor shapes and types
    assert out.shape == out_partial.shape[1:], "out shape mismatch"
    if lse is not None:
        assert lse.shape == lse_partial.shape[1:], "lse shape mismatch"
        assert lse.dtype == torch.float32, "lse must be fp32"

    # Validate optional tensors
    for t, name in [
        (cu_seqlens, "cu_seqlens"),
        (seqused, "seqused"),
        (num_splits_dynamic_ptr, "num_splits_dynamic_ptr"),
    ]:
        if t is not None:
            assert t.dtype == torch.int32, f"{name} must be int32"
            assert t.is_cuda, f"{name} must be on CUDA device"
            assert t.is_contiguous(), f"{name} must be contiguous"

    head_dim = out_partial.shape[-1]
    num_splits = out_partial.shape[0]
    assert num_splits <= 256
    # If hdim is 96 or 192, it's faster to round them to 128 or 256 respectively
    # so that kBlockM is smaller and we have more parallelism.
    k_block_size = 64 if head_dim <= 64 else 128
    # We want kBlockM to be as small as possible to maximize parallelism.
    # E.g., if hdim is 64, we want kBlockM to be 16 so that we can use 256 threads, each reading 4 elements (floats).
    m_block_size = (
        8 if k_block_size % 128 == 0 else (16 if k_block_size % 64 == 0 else 32)
    )
    log_max_splits = max(math.ceil(math.log2(num_splits)), 4)
    if m_block_size == 8:
        # If kBlockM == 8 then the minimum number of splits is 32.
        # TODO: we can deal w this by using 128 threads instead
        log_max_splits = max(log_max_splits, 5)

    # Convert to cute tensors (using kernel-formatted tensors)
    out_partial_tensor = from_dlpack(
        out_partial.detach(), assumed_align=16
    ).mark_layout_dynamic(leading_dim=4)
    lse_partial_tensor = from_dlpack(
        lse_partial.detach(), assumed_align=4
    ).mark_layout_dynamic(leading_dim=lse_partial.ndim - 2)
    out_tensor = from_dlpack(out.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=3
    )
    lse_tensor = (
        from_dlpack(lse.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=lse.ndim - 2
        )
        if lse is not None
        else None
    )

    optional_tensors = [
        from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)
        if t is not None
        else None
        for t in (cu_seqlens, seqused, num_splits_dynamic_ptr, semaphore_to_reset)
    ]
    cu_seqlens_tensor, seqused_tensor, num_splits_dynamic_tensor, semaphore_tensor = (
        optional_tensors
    )

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Create combine kernel configuration
    dtype = torch2cute_dtype_map[out.dtype]
    dtype_partial = torch2cute_dtype_map[out_partial.dtype]

    compile_key = (
        dtype,
        dtype_partial,
        head_dim,
        m_block_size,
        k_block_size,
        log_max_splits,
        cu_seqlens is not None,
        seqused is not None,
        lse is not None,
    )

    if compile_key not in _flash_attn_fwd_combine.compile_cache:
        fa_combine = FlashAttentionForwardCombine(
            dtype=dtype,
            dtype_partial=dtype_partial,
            head_dim=head_dim,
            m_block_size=m_block_size,
            k_block_size=k_block_size,
            log_max_splits=log_max_splits,
        )

        # Check if implementation is supported
        if not fa_combine.can_implement(
            dtype,
            dtype_partial,
            head_dim,
            m_block_size,
            k_block_size,
            log_max_splits,
            num_threads=256,
        ):
            raise RuntimeError(
                f"FlashAttention combine kernel cannot be implemented with given parameters"
            )

        _flash_attn_fwd_combine.compile_cache[compile_key] = cute.compile(
            fa_combine,
            out_partial_tensor,
            lse_partial_tensor,
            out_tensor,
            lse_tensor,
            cu_seqlens_tensor,
            seqused_tensor,
            num_splits_dynamic_tensor,
            semaphore_tensor,
            current_stream,
        )

    _flash_attn_fwd_combine.compile_cache[compile_key](
        out_partial_tensor,
        lse_partial_tensor,
        out_tensor,
        lse_tensor,
        cu_seqlens_tensor,
        seqused_tensor,
        num_splits_dynamic_tensor,
        semaphore_tensor,
        current_stream,
    )


_flash_attn_fwd_combine.compile_cache = {}


def flash_attn_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    return_lse: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Flash Attention combine function for split attention computation.

    Combines partial outputs and log-sum-exp values from multiple splits
    of attention computation into final outputs. This is the main user-facing
    interface for the combine kernel.

    Args:
        out_partial: Partial outputs tensor with shape:
            - (num_splits, batch_size, seqlen, num_heads, head_size) for regular batched input
            - (num_splits, total_q, num_heads, head_size) for variable length input
        lse_partial: Partial LSE tensor with shape:
            - (num_splits, batch_size, seqlen, num_heads) for regular batched input
            - (num_splits, total_q, num_heads) for variable length input
        out: Optional output tensor. If None, will be created automatically.
        out_dtype: Optional output dtype. If None, will use fp16/bf16 based on input.
        return_lse: Whether to return the combined LSE tensor. Default is True.

    Returns:
        Tuple of (out, lse) where:
        - out: Combined output tensor with shape (batch_size, seqlen, num_heads, head_size)
              or (total_q, num_heads, head_size) for varlen
        - lse: Combined log-sum-exp tensor with shape (batch_size, seqlen, num_heads)
              or (total_q, num_heads) for varlen. None if return_lse=False

    Note:
        This function expects the input tensors to be in the format produced by
        split attention computation, where the first dimension is num_splits.
        The permuting from user format to kernel format is now done inside the kernel.
    """
    # Input validation
    assert out_partial.dim() in [4, 5], "out_partial must have 4 or 5 dimensions"
    assert lse_partial.dim() in [3, 4], "lse_partial must have 3 or 4 dimensions"
    assert out_partial.dtype == torch.float32, (
        "out_partial must be fp32 (from accumulation)"
    )
    assert lse_partial.dtype == torch.float32, "lse_partial must be fp32"

    # Determine if this is variable length based on dimensions
    is_varlen = out_partial.dim() == 4

    if is_varlen:
        # Variable length: (num_splits, total_q, num_heads, head_size)
        num_splits, total_q, num_heads, head_size = out_partial.shape
        assert lse_partial.shape == (num_splits, total_q, num_heads), (
            "lse_partial shape mismatch for varlen"
        )
        batch_size = 1  # Treat as single batch for varlen
        seqlen = total_q
    else:
        # Regular batched: (num_splits, batch_size, seqlen, num_heads, head_size)
        num_splits, batch_size, seqlen, num_heads, head_size = out_partial.shape
        assert lse_partial.shape == (num_splits, batch_size, seqlen, num_heads), (
            "lse_partial shape mismatch"
        )

    # Determine output dtype
    if out_dtype is None:
        out_dtype = out_partial.dtype

    # Create output if not provided
    device = out_partial.device
    if out is None:
        if is_varlen:
            out = torch.empty(
                total_q, num_heads, head_size, dtype=out_dtype, device=device
            )
        else:
            out = torch.empty(
                batch_size, seqlen, num_heads, head_size, dtype=out_dtype, device=device
            )

    # Create lse output only if requested
    if return_lse:
        if is_varlen:
            lse = torch.empty(
                num_heads, total_q, dtype=torch.float32, device=device
            ).transpose(0, 1)
        else:
            lse = torch.empty(
                batch_size, num_heads, seqlen, dtype=torch.float32, device=device
            ).transpose(1, 2)
    else:
        lse = None

    _flash_attn_fwd_combine(out_partial, lse_partial, out, lse)
    return out, lse
