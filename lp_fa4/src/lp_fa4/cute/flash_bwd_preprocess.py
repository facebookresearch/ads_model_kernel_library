# @nolint
# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# A reimplementation of https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_bwd_preprocess_kernel.h
# from Cutlass C++ to Cute-DSL.
#
# Computes D_i = (dO_i * O_i).sum(dim=-1), optionally adjusted for LSE gradient:
#   D'_i = D_i - dLSE_i
# This works because in the backward pass:
#   dS_ij = P_ij * (dP_ij - D_i)                     [standard]
# When LSE is differentiable, d(loss)/d(S_ij) gets an extra term dLSE_i * P_ij
# (since d(LSE_i)/d(S_ij) = P_ij), giving:
#   dS_ij = P_ij * (dP_ij - D_i) + dLSE_i * P_ij
#         = P_ij * (dP_ij - (D_i - dLSE_i))
# So the main backward kernel is unchanged; we just replace D with D' = D - dLSE here.
import math
import operator
from functools import partial
from typing import Callable, Type, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cutlass_dsl import Arch, BaseDSL

from quack import copy_utils, layout_utils

from lp_fa4.cute import utils
from lp_fa4.cute.seqlen_info import SeqlenInfo
from quack.cute_dsl_utils import ParamsBase
from lp_fa4.cute.tile_scheduler import (
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)
from lp_fa4.cute.pack_gqa import pack_gqa_layout
from lp_fa4.cute.softmax import (
    E4M3_MAX_NORM_RCP,
    fused_abs_max_f32,
    fused_amax_to_e8m0_scale_f32_hw,
    max_f32,
)


class FlashAttentionBackwardPreprocess:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: int,
        tile_m: int = 128,
        num_threads: int = 256,
        use_padded_offsets: bool = True,
        nheads_major: bool = False,
        pack_gqa: bool = False,
        qhead_per_kvhead: int = 1,
        nheads_kv: int = 1,
        quantize_do: bool = False,
        broadcast_q: bool = False,
    ):
        """
        All contiguous dimensions must be at least 16 bytes aligned which indicates the head dimension
        should be a multiple of 8.

        :param head_dim: head dimension
        :type head_dim: int
        :param tile_m: m block size
        :type tile_m: int
        :param num_threads: number of threads
        :type num_threads: int
        """
        self.use_pdl = BaseDSL._get_dsl().get_arch_enum() >= Arch.sm_90a
        self.dtype = dtype
        self.tile_m = tile_m
        # padding head_dim to a multiple of 32 as k_block_size
        hdim_multiple_of = 32
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        self.head_dim_v_padded = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.num_threads = num_threads
        self.use_padded_offsets = use_padded_offsets
        self.nheads_major = nheads_major
        self.pack_gqa = pack_gqa
        self.qhead_per_kvhead = qhead_per_kvhead
        self.nheads_kv = nheads_kv
        self.dqaccum_dtype = Float32
        self.quantize_do = quantize_do
        self.broadcast_q = broadcast_q
        if self.quantize_do:
            assert self.tile_m == 128 and self.head_dim_v_padded == 128
            num_kblocks = self.head_dim_v_padded // 32
            num_groups = self.tile_m // 32
            sy_bytes = self.tile_m * self.head_dim_v_padded * self.dtype.width // 8
            samax_bytes = self.tile_m * num_kblocks * Float32.width // 8
            samax_reduced_bytes = num_groups * num_kblocks * Float32.width // 8
            sinv_scale_bytes = num_groups * num_kblocks * Float32.width // 8
            sscale_e8_bytes = num_groups * num_kblocks * cutlass.Uint8.width // 8
            self.smem_size = (
                cute.round_up(sy_bytes, 16)
                + cute.round_up(samax_bytes, 4)
                + cute.round_up(samax_reduced_bytes, 4)
                + cute.round_up(sinv_scale_bytes, 4)
                + cute.round_up(sscale_e8_bytes, 4)
            )
        else:
            self.smem_size = 0

    @staticmethod
    def can_implement(dtype, head_dim, tile_m, num_threads) -> bool:
        """Check if the kernel can be implemented with the given parameters.

        :param dtype: data type
        :type dtype: cutlass.Numeric
        :param head_dim: head dimension
        :type head_dim: int
        :param tile_m: m block size
        :type tile_m: int
        :param num_threads: number of threads
        :type num_threads: int

        :return: True if the kernel can be implemented, False otherwise
        :rtype: bool
        """
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if num_threads < tile_m:  # For multiplying lse with log2
            return False
        return True

    def _setup_attributes(self):
        # ///////////////////////////////////////////////////////////////////////////////
        # GMEM Tiled copy:
        # ///////////////////////////////////////////////////////////////////////////////
        # Thread layouts for copies
        # We want kBlockKGmem to be a power of 2 so that when we do the summing,
        # it's just between threads in the same warp
        gmem_k_block_size = (
            128
            if self.head_dim_v_padded % 128 == 0
            else (
                64
                if self.head_dim_v_padded % 64 == 0
                else (32 if self.head_dim_v_padded % 32 == 0 else 16)
            )
        )
        num_copy_elems = 128 // self.dtype.width
        threads_per_row = gmem_k_block_size // num_copy_elems
        self.gmem_tiled_copy_O = copy_utils.tiled_copy_2d(
            self.dtype, threads_per_row, self.num_threads, num_copy_elems
        )
        universal_copy_bits = 128
        num_copy_elems_dQaccum = universal_copy_bits // self.dqaccum_dtype.width
        assert (
            self.tile_m * self.head_dim_padded // num_copy_elems_dQaccum
        ) % self.num_threads == 0
        self.gmem_tiled_copy_dQaccum = copy_utils.tiled_copy_1d(
            self.dqaccum_dtype, self.num_threads, num_copy_elems_dQaccum
        )
        if self.quantize_do:
            fp8_copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cutlass.Float8E4M3FN,
                num_bits_per_copy=num_copy_elems * cutlass.Float8E4M3FN.width,
            )
            fp8_thr_layout = cute.make_ordered_layout(
                (self.num_threads // threads_per_row, threads_per_row),
                order=(1, 0),
            )
            self.gmem_tiled_copy_dO_fp8 = cute.make_tiled_copy_tv(
                fp8_copy_atom,
                fp8_thr_layout,
                cute.make_layout((1, num_copy_elems)),
            )

    @cute.jit
    def __call__(
        self,
        mO: cute.Tensor,  # (batch, seqlen, nheads, head_dim_v) or (total_q, nheads, head_dim_v)
        mdO: cute.Tensor,  # same shape as mO
        mPdPsum: cute.Tensor,  # (batch, nheads, seqlen_padded) or (nheads, total_q_padded)
        mLSE: Optional[cute.Tensor],  # (batch, nheads, seqlen) or (nheads, total_q)
        mLSElog2: Optional[cute.Tensor],  # same shape as mPdPsum
        # (batch, nheads, seqlen_padded * head_dim_v) or (nheads, total_q_padded * head_dim_v)
        mdQaccum: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],  # (batch + 1,)
        mSeqUsedQ: Optional[cute.Tensor],  # (batch,)
        mdLSE: Optional[cute.Tensor],  # (batch, nheads, seqlen) or (nheads, total_q)
        mRowMax: Optional[cute.Tensor],  # (b, s, n, h) or (t, n, h)
        mScaleP: Optional[cute.Tensor],  # == mRowMax
        softmax_scale: Float32,
        mdO_fp8: Optional[cute.Tensor] = None,
        mSFdO: Optional[cute.Tensor] = None,
        mSFdO_dV: Optional[cute.Tensor] = None,
        mCuSeqlensSFdO: Optional[cute.Tensor] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # Get the data type and check if it is fp16 or bf16
        if const_expr(not (mO.element_type == mdO.element_type)):
            raise TypeError("All tensors must have the same data type")
        if const_expr(mO.element_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mPdPsum.element_type not in [Float32]):
            raise TypeError("PdPsum tensor must be Float32")
        if const_expr(mdQaccum is not None):
            assert self.nheads_major is False
            assert self.pack_gqa is False
            assert self.use_padded_offsets is True
            if const_expr(mdQaccum.element_type not in [Float32, cutlass.Float16]):
                raise TypeError("dQaccum tensor must be Float32 or Float16")
            self.dqaccum_dtype = mdQaccum.element_type
        if const_expr(mLSE is not None):
            if const_expr(mLSE.element_type not in [Float32]):
                raise TypeError("LSE tensor must be Float32")
        if const_expr(mLSElog2 is not None):
            if const_expr(mLSElog2.element_type not in [Float32]):
                raise TypeError("LSElog2 tensor must be Float32")
        if const_expr(mdLSE is not None):
            if const_expr(mdLSE.element_type not in [Float32]):
                raise TypeError("dLSE tensor must be Float32")
        if const_expr(mScaleP is not None):
            assert self.nheads_major is True
            assert self.pack_gqa is True
            assert mRowMax is not None
            if const_expr(mScaleP.element_type not in [Float32]):
                raise TypeError("ScaleP tensor must be Float32")
            if const_expr(mRowMax.element_type not in [Float32]):
                raise TypeError("RowMax tensor must be Float32")
        if const_expr(self.quantize_do):
            assert (
                mdO_fp8 is not None
                and mSFdO is not None
                and mSFdO_dV is not None
                and mCuSeqlensSFdO is not None
            )
            if const_expr(mdO_fp8.element_type != cutlass.Float8E4M3FN):
                raise TypeError("Quantized dO output must use E4M3 elements")
            if const_expr(
                mSFdO.element_type != cutlass.Uint8
                or mSFdO_dV.element_type != cutlass.Uint8
            ):
                raise TypeError("dO scale outputs must use uint8 elements")
        else:
            assert (
                mdO_fp8 is None
                and mSFdO is None
                and mSFdO_dV is None
                and mCuSeqlensSFdO is None
            )

        if const_expr(self.quantize_do):
            mSFdO, mSFdO_dV = [
                cute.make_tensor(
                    cute.recast_ptr(mSF.iterator, dtype=cutlass.Uint8),
                    cute.make_layout((cute.size(mSF.shape),)),
                )
                for mSF in (mSFdO, mSFdO_dV)
            ]

        self._setup_attributes()

        # (b, s, h, d)  -> (s, d, h, b)  or
        # (total, h, d) -> (total, d, h)
        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mO, mdO, mdO_fp8 = [
            cute.make_tensor(mX.iterator, cute.select(mX.layout, mode=QO_layout_transpose))
            if mX is not None
            else None
            for mX in (mO, mdO, mdO_fp8)
        ]

        if const_expr(not self.nheads_major):
            # (batch, nheads, seqlen) -> (seqlen, nheads, batch) or
            # (nheads, total_q) -> (total_q, nheads)
            transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        else:
            # (batch, seqlen, nheads) -> (seqlen, nheads, batch) or
            # (total_q, nheads) -> (total_q, nheads)
            transpose = [1, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 1]
        mPdPsum, mLSE, mLSElog2, mdLSE, mdQaccum = [
            layout_utils.select(mX, transpose) if mX is not None else None
            for mX in (mPdPsum, mLSE, mLSElog2, mdLSE, mdQaccum)
        ]

        # (b, s, n, h) => (s, n, h, b) or
        # (total, n, h) == (total, n, h)
        rowmax_layout_transpose = [1, 2, 3, 0] if const_expr(mCuSeqlensQ is None) else [0, 1, 2]
        if const_expr(mRowMax is not None):
            mRowMax = layout_utils.select(mRowMax, rowmax_layout_transpose)
        if const_expr(mScaleP is not None):
            mScaleP = layout_utils.select(mScaleP, rowmax_layout_transpose)

        # pack gqa
        if const_expr(self.pack_gqa):
            mO, mdO, mRowMax, mScaleP = [
                pack_gqa_layout(mX, self.qhead_per_kvhead, self.nheads_kv, head_idx=2)
                if mX is not None
                else None
                for mX in (mO, mdO, mRowMax, mScaleP)
            ]
            mPdPsum, mLSE, mLSElog2, mdLSE = [
                pack_gqa_layout(mX, self.qhead_per_kvhead, self.nheads_kv, head_idx=1)
                if mX is not None
                else None
                for mX in (mPdPsum, mLSE, mLSElog2, mdLSE)
            ]

        # mO: (s, d, h, b) or (total, d, h)
        if const_expr(mCuSeqlensQ is not None):
            num_head = mO.shape[2]
            num_batch = mCuSeqlensQ.shape[0] - 1
            if const_expr(self.broadcast_q):
                assert not self.pack_gqa
                TileScheduler = SingleTileScheduler
                num_block = cute.ceil_div(
                    mO.shape[0] // num_batch, self.tile_m
                )
            else:
                TileScheduler = SingleTileVarlenScheduler
                num_block = cute.ceil_div(mO.shape[0], self.tile_m)
        else:
            TileScheduler = SingleTileScheduler
            num_head = mO.shape[2]
            num_batch = mO.shape[3]
            num_block = cute.ceil_div(mO.shape[0], self.tile_m)

        tile_sched_args = TileSchedulerArguments(
            num_block=num_block,
            num_head=num_head,
            num_batch=num_batch,
            num_splits=1,
            seqlen_k=0,
            headdim=0,
            headdim_v=mO.shape[1],
            total_q=cute.size(mO.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mO.shape[0]) * cute.size(mO.shape[3]),
            tile_shape_mn=(self.tile_m, 1),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = softmax_scale * LOG2_E

        self.kernel(
            mO,
            mdO,
            mPdPsum,
            mLSE,
            mLSElog2,
            mdQaccum,
            mCuSeqlensQ,
            mSeqUsedQ,
            mdLSE,
            mRowMax,
            mScaleP,
            softmax_scale_log2,
            self.gmem_tiled_copy_O,
            self.gmem_tiled_copy_dQaccum,
            tile_sched_params,
            TileScheduler,
            mdO_fp8,
            mSFdO,
            mSFdO_dV,
            mCuSeqlensSFdO,
            self.gmem_tiled_copy_dO_fp8 if self.quantize_do else None,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
            smem=self.smem_size,
        )

    @cute.kernel
    def kernel(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mPdPsum: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mLSElog2: Optional[cute.Tensor],
        mdQaccum: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mdLSE: Optional[cute.Tensor],
        mRowMax: Optional[cute.Tensor],
        mScaleP: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_dQaccum: cute.TiledCopy,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        mdO_fp8: Optional[cute.Tensor],
        mSFdO: Optional[cute.Tensor],
        mSFdO_dV: Optional[cute.Tensor],
        mCuSeqlensSFdO: Optional[cute.Tensor],
        gmem_tiled_copy_dO_fp8: Optional[cute.TiledCopy],
    ):
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()

        # PDL can start this kernel before upstream tensors and varlen metadata are ready.
        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, head_idx, batch_idx, _ = work_tile.tile_idx

        if work_tile.is_valid_tile:
            # ///////////////////////////////////////////////////////////////////////////////
            # Get the appropriate tiles for this thread block.
            # ///////////////////////////////////////////////////////////////////////////////
            seqlen_static = mO.shape[0] if const_expr(not self.pack_gqa) else mO.shape[0][1]
            seqlen = SeqlenInfo.create(
                batch_idx, seqlen_static, mCuSeqlensQ, mSeqUsedQ, tile=self.tile_m
            )
            # (seqlen, dv)
            mO_cur, mdO_cur = [
                seqlen.offset_batch(mX, batch_idx, dim=3)[None, None, head_idx] for mX in (mO, mdO)
            ]
            mPdPsum_cur = seqlen.offset_batch(
                mPdPsum, batch_idx, dim=2, padded=self.use_padded_offsets
            )[None, head_idx]
            headdim_v = mO_cur.shape[1]
            seqlen_q = (
                seqlen.seqlen
                if const_expr(not self.pack_gqa)
                else seqlen.seqlen * self.qhead_per_kvhead
            )
            seqlen_q_rounded = cute.round_up(seqlen_q, self.tile_m)
            seqlen_limit = seqlen_q - m_block * self.tile_m

            lse = None
            if const_expr(mLSE is not None):
                mLSE_cur = seqlen.offset_batch(mLSE, batch_idx, dim=2)[None, head_idx]
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
                lse = Float32.inf
                if tidx < seqlen_limit:
                    lse = gLSE[tidx]

            blk_shape = (self.tile_m, self.head_dim_v_padded)
            gO = cute.local_tile(mO_cur, blk_shape, (m_block, 0))
            gdO = cute.local_tile(mdO_cur, blk_shape, (m_block, 0))
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            # (CPY_Atom, CPY_M, CPY_K)
            tOgO = gmem_thr_copy_O.partition_S(gO)
            tOgdO = gmem_thr_copy_O.partition_S(gdO)
            if const_expr(self.quantize_do):
                mdO_fp8_cur = seqlen.offset_batch(mdO_fp8, batch_idx, dim=3)[
                    None, None, head_idx
                ]
                gdO_fp8 = cute.local_tile(mdO_fp8_cur, blk_shape, (m_block, 0))
                gmem_thr_copy_fp8 = gmem_tiled_copy_dO_fp8.get_slice(tidx)
                tOgdO_fp8 = gmem_thr_copy_fp8.partition_S(gdO_fp8)
                smem = cutlass.utils.SmemAllocator()
                sY = smem.allocate_tensor(
                    self.dtype,
                    cute.make_layout((self.tile_m * self.head_dim_v_padded,)),
                    byte_alignment=16,
                )
                num_kblocks = self.head_dim_v_padded // 32
                num_groups = self.tile_m // 32
                sAmax = smem.allocate_tensor(
                    Float32,
                    cute.make_layout((self.tile_m * num_kblocks,)),
                    byte_alignment=4,
                )
                sAmax_reduced = smem.allocate_tensor(
                    Float32,
                    cute.make_layout((num_groups * num_kblocks,)),
                    byte_alignment=4,
                )
                sInvScale = smem.allocate_tensor(
                    Float32,
                    cute.make_layout((num_groups * num_kblocks,)),
                    byte_alignment=4,
                )
                sScaleE8 = smem.allocate_tensor(
                    cutlass.Uint8,
                    cute.make_layout((num_groups * num_kblocks,)),
                    byte_alignment=4,
                )
                for idx in cutlass.range(
                    cute.ceil_div(self.tile_m * num_kblocks, self.num_threads),
                    unroll_full=True,
                ):
                    amax_idx = tidx + idx * self.num_threads
                    if amax_idx < self.tile_m * num_kblocks:
                        sAmax[amax_idx] = Float32(0.0)
                cute.arch.sync_threads()
            else:
                tOgdO_fp8 = None
                gmem_thr_copy_fp8 = None
                sY = sAmax = sAmax_reduced = sInvScale = sScaleE8 = None
            cO = cute.make_identity_tensor(blk_shape)
            tOcO = gmem_thr_copy_O.partition_S(cO)
            t0OcO = gmem_thr_copy_O.get_slice(0).partition_S(cO)
            tOpO = None
            if const_expr(self.check_hdim_v_oob):
                tOpO = copy_utils.predicate_k(tOcO, limit=headdim_v)
            # Each copy will use the same predicate
            copy = partial(copy_utils.copy, pred=tOpO)

            num_m_slices = cute.size(tOgO.shape[1])
            threads_per_row = gmem_tiled_copy_O.layout_src_tv_tiled[0].shape[0]
            assert cute.arch.WARP_SIZE % threads_per_row == 0
            if const_expr(self.quantize_do):
                tOrO = cute.make_rmem_tensor_like(tOgO[None, 0, None])
                tOrdO = cute.make_rmem_tensor_like(tOgdO[None, 0, None])
                tOrO_flat = cute.make_tensor(
                    tOrO.iterator, cute.make_layout((cute.size(tOrO),))
                )
                tOrdO_flat = cute.make_tensor(
                    tOrdO.iterator, cute.make_layout((cute.size(tOrdO),))
                )
                PdP_sum = cute.make_rmem_tensor(num_m_slices, Float32)
            else:
                tOrO = cute.make_rmem_tensor_like(tOgO)
                tOrdO = cute.make_rmem_tensor_like(tOgdO)
                if const_expr(self.check_hdim_v_oob):
                    tOrO.fill(0.0)
                    tOrdO.fill(0.0)
            assert tOgO.shape == tOgdO.shape
            elems_per_thread = cute.size(tOgdO.shape[0])
            for m in cutlass.range(num_m_slices, unroll_full=True):
                # Instead of using tOcO, we using t0OcO and subtract the offset from the limit.
                # This is bc the entries of t0OcO are known at compile time.
                if const_expr(self.quantize_do):
                    tOrdO.fill(0.0)
                    if const_expr(self.check_hdim_v_oob):
                        tOrO.fill(0.0)
                if t0OcO[0, m, 0][0] < seqlen_limit - tOcO[0][0]:
                    if const_expr(self.quantize_do):
                        copy(tOgO[None, m, None], tOrO)
                        copy(tOgdO[None, m, None], tOrdO)
                    else:
                        copy(tOgO[None, m, None], tOrO[None, m, None])
                        copy(tOgdO[None, m, None], tOrdO[None, m, None])
                elif const_expr(self.quantize_do):
                    tOrO.fill(0.0)
                if const_expr(self.quantize_do):
                    pdpsum = Float32(0.0)
                    for i in cutlass.range_constexpr(elems_per_thread):
                        pdpsum += Float32(tOrO_flat[i]) * Float32(tOrdO_flat[i])
                    PdP_sum[m] = utils.warp_reduce(
                        pdpsum, operator.add, width=threads_per_row
                    )
                    row = tOcO[0, m, 0][0]
                    k_col = tOcO[0, m, 0][1]
                    for i in cutlass.range_constexpr(elems_per_thread):
                        sY[row * self.head_dim_v_padded + k_col + i] = tOrdO_flat[i]
                    block_amax = Float32(0.0)
                    for i in cutlass.range_constexpr(elems_per_thread):
                        block_amax = fused_abs_max_f32(
                            block_amax, Float32(tOrdO_flat[i])
                        )
                    threads_per_kblock = 32 // elems_per_thread
                    # The shuffle uses a full mask, so invalid rows contribute zeros.
                    for shift in cutlass.range_constexpr(
                        int(math.log2(threads_per_kblock))
                    ):
                        block_amax = max_f32(
                            block_amax,
                            cute.arch.shuffle_sync_bfly(
                                block_amax, offset=1 << shift
                            ),
                        )
                    if k_col % 32 == 0:
                        sAmax[row * num_kblocks + k_col // 32] = block_amax
            if const_expr(self.quantize_do):
                cute.arch.sync_threads()
                num_tiles = num_groups * num_kblocks
                if tidx < num_tiles:
                    group = tidx // num_kblocks
                    k_block = tidx % num_kblocks
                    group_amax = Float32(0.0)
                    for row_in_group in cutlass.range(32, unroll_full=True):
                        group_amax = max_f32(
                            group_amax,
                            sAmax[
                                (group * 32 + row_in_group) * num_kblocks
                                + k_block
                            ],
                        )
                    sAmax_reduced[group * num_kblocks + k_block] = group_amax
                cute.arch.sync_threads()

                max_norm_rcp = Float32(E4M3_MAX_NORM_RCP)
                if tidx < num_tiles:
                    group_amax = sAmax_reduced[tidx]
                    inv_scale, scale_e8 = fused_amax_to_e8m0_scale_f32_hw(
                        group_amax, max_norm_rcp
                    )
                    if group_amax == Float32(0.0):
                        inv_scale = Float32(1.0)
                        scale_e8 = cutlass.Uint32(127)
                    sInvScale[tidx] = inv_scale
                    sScaleE8[tidx] = cutlass.Uint8(
                        scale_e8 & cutlass.Uint32(0xFF)
                    )
                cute.arch.sync_threads()

                num_heads = Int32(mO.shape[2])
                total_scale_tokens = Int32(
                    cute.size(mSFdO.shape) // (num_heads * num_kblocks)
                )
                num_token_atoms = total_scale_tokens // Int32(128)
                tOrdO_fp8 = cute.make_rmem_tensor_like(tOgdO_fp8[None, 0, None])
                for m in cutlass.range(num_m_slices, unroll_full=True):
                    row = tOcO[0, m, 0][0]
                    k_col = tOcO[0, m, 0][1]
                    scale_idx = (row // 32) * num_kblocks + k_col // 32
                    if t0OcO[0, m, 0][0] < seqlen_limit - tOcO[0][0]:
                        inv_scale = sInvScale[scale_idx]
                        dO_scaled = cute.make_rmem_tensor(
                            elems_per_thread, Float32
                        )
                        for i in cutlass.range_constexpr(elems_per_thread):
                            dO_scaled[i] = (
                                Float32(
                                    sY[
                                        row * self.head_dim_v_padded
                                        + k_col
                                        + i
                                    ]
                                )
                                * inv_scale
                            )
                        utils.cvt_fp8(dO_scaled, tOrdO_fp8)
                        cute.copy(
                            gmem_thr_copy_fp8,
                            tOrdO_fp8,
                            tOgdO_fp8[None, m, None],
                            pred=tOpO,
                        )

                    global_scale_token = (
                        Int32(mCuSeqlensSFdO[batch_idx])
                        + Int32(m_block * self.tile_m)
                        + Int32(row)
                    )
                    scale_e8 = sScaleE8[scale_idx]
                    if k_col % 32 == 0:
                        global_scale_row = (
                            Int32(head_idx) * total_scale_tokens
                            + global_scale_token
                        )
                        atom_row = global_scale_row % Int32(128)
                        sf_offset = (
                            (global_scale_row // Int32(128)) * Int32(512)
                            + (atom_row % Int32(32)) * Int32(16)
                            + (atom_row // Int32(32)) * Int32(4)
                            + Int32(k_col // 32)
                        )
                        mSFdO[sf_offset] = scale_e8

                    if row % 32 == 0:
                        token_group = global_scale_token // Int32(32)
                        for i in cutlass.range_constexpr(elems_per_thread):
                            hdim_row = Int32(k_col + i)
                            sf_dv_offset = (
                                (
                                    Int32(head_idx) * num_token_atoms
                                    + token_group // Int32(4)
                                )
                                * Int32(512)
                                + (hdim_row % Int32(32)) * Int32(16)
                                + (hdim_row // Int32(32)) * Int32(4)
                                + token_group % Int32(4)
                            )
                            mSFdO_dV[sf_dv_offset] = scale_e8
            # PDL consumers wait before reading these outputs; ordinary consumers
            # remain ordered by the CUDA stream.
            if const_expr(self.use_pdl):
                if const_expr(self.quantize_do):
                    cute.arch.sync_threads()
                cute.arch.griddepcontrol_launch_dependents()
            if const_expr(not self.quantize_do):
                # Sum across the "k" dimension
                pdpsum = (tOrO.load().to(Float32) * tOrdO.load().to(Float32)).reduce(
                    cute.ReductionOp.ADD,
                    init_val=0.0,
                    reduction_profile=(0, None, 1),
                )
                pdpsum = utils.warp_reduce(
                    pdpsum, operator.add, width=threads_per_row
                )
                PdP_sum = cute.make_rmem_tensor(num_m_slices, Float32)
                PdP_sum.store(pdpsum)

            # If dLSE is provided, compute D' = D - dLSE (see module docstring for derivation).
            gdLSE = None
            if const_expr(mdLSE is not None):
                mdLSE_cur = seqlen.offset_batch(mdLSE, batch_idx, dim=2)[None, head_idx]
                gdLSE = cute.local_tile(mdLSE_cur, (self.tile_m,), (m_block,))

            # Write PdPsum from rmem -> gmem
            gPdPsum = cute.local_tile(mPdPsum_cur, (self.tile_m,), (m_block,))
            # Only the thread corresponding to column 0 writes out the PdPsum to gmem
            if tOcO[0, 0, 0][1] == 0:
                for m in cutlass.range(cute.size(PdP_sum), unroll_full=True):
                    row = tOcO[0, m, 0][0]
                    PdPsum_val = 0.0
                    if row < seqlen_limit:
                        PdPsum_val = PdP_sum[m]
                        if const_expr(mdLSE is not None):
                            PdPsum_val -= gdLSE[row]
                    gPdPsum[row] = PdPsum_val

            # Clear dQaccum
            if const_expr(mdQaccum is not None):
                mdQaccum_cur = seqlen.offset_batch(
                    mdQaccum,
                    batch_idx,
                    dim=2,
                    padded=self.use_padded_offsets,
                    multiple=self.head_dim_padded,
                )[None, head_idx]
                blkdQaccum_shape = (self.tile_m * self.head_dim_padded,)
                gdQaccum = cute.local_tile(mdQaccum_cur, blkdQaccum_shape, (m_block,))
                gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_slice(tidx)
                tdQgdQaccum = gmem_thr_copy_dQaccum.partition_S(gdQaccum)
                zero = cute.make_rmem_tensor_like(tdQgdQaccum)
                zero.fill(0.0)
                cute.copy(gmem_tiled_copy_dQaccum, zero, tdQgdQaccum)

            LOG2_E = math.log2(math.e)
            lse_log2 = lse * LOG2_E if lse != -Float32.inf else 0.0
            if const_expr(mLSElog2 is not None):
                mLSElog2_cur = seqlen.offset_batch(
                    mLSElog2, batch_idx, dim=2, padded=self.use_padded_offsets
                )[None, head_idx]
                gLSElog2 = cute.local_tile(mLSElog2_cur, (self.tile_m,), (m_block,))
                LOG2_E = math.log2(math.e)
                if tidx < seqlen_q_rounded - m_block * self.tile_m:
                    gLSElog2[tidx] = lse_log2

            if const_expr(mRowMax is not None):
                assert mLSE is not None
                # (s, n)
                mRowMax_cur, mScaleP_cur = [
                    seqlen.offset_batch(mX, batch_idx, dim=3)[None, None, head_idx]
                    for mX in (mRowMax, mScaleP)
                ]
                # (tile_m, n)
                gRowMax, gScaleP = [
                    cute.local_tile(mX, (self.tile_m,), (m_block, None))
                    for mX in (mRowMax_cur, mScaleP_cur)
                ]

                assert self.tile_m <= self.num_threads
                if const_expr(self.tile_m == self.num_threads) or tidx < self.tile_m:
                    for n in cutlass.range(gRowMax.shape[1], unroll=4):
                        row_max = gRowMax[tidx, n]
                        scale = 0.0
                        if row_max != -Float32.inf and lse != -Float32.inf:
                            scale = softmax_scale_log2 * row_max - lse_log2
                            scale = cute.math.exp2(scale, fastmath=True)
                        gScaleP[tidx, n] = scale
