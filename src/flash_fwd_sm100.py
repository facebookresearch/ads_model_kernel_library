# @nolint
# Supported features:
# - BF16 & FP16 dtype
# - noncausal & causal attention
# - MHA, GQA, MQA
# - hdim 64, 96, 128, (192, 128).
# - varlen
# - sliding window
# Unsupported features that will be added later:
# - split-kv (optimizing for inference)
# - more hdim (192, 256)
# Based on the cutlass example and cute-dsl example:
# https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/fmha.py
# pyre-ignore-all-errors
import enum
import math
from typing import Type, Tuple, Callable, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
import cutlass.utils.blockscaled_layout as blockscaled_utils

import ads_mkl.ops.cute_dsl.gdpa.src.utils as utils
# import ads_mkl.ops.cute_dsl.gdpa.src.pipeline as pipeline
from ads_mkl.ops.cute_dsl.gdpa.src.mask import AttentionMask
from ads_mkl.ops.cute_dsl.gdpa.src.seqlen_info import SeqlenInfoQK
from ads_mkl.ops.cute_dsl.gdpa.src.block_info import BlockInfo
from ads_mkl.ops.cute_dsl.gdpa.src.pack_gqa import PackGQA
from ads_mkl.ops.cute_dsl.gdpa.src import blackwell_helpers as sm100_utils
from ads_mkl.ops.cute_dsl.gdpa.src.tile_scheduler import TileSchedulerArguments, SingleTileScheduler, StaticPersistentTileScheduler, SingleTileLPTScheduler, SingleTileVarlenScheduler, SingleTileVarlenSchedulerSimple, ParamsBase, SingleTileVarlenSchedulerPersistent
from ads_mkl.ops.cute_dsl.gdpa.src.activation_utils import Gelu, Relu, pack_4xu8_to_u32, store_u32_shared


# class NamedBarrierFwd(enum.IntEnum):
#     Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
#     WarpSchedulerWG1 = enum.auto()
#     WarpSchedulerWG2 = enum.auto()
#     WarpSchedulerWG3 = enum.auto()
#     PFull = enum.auto()
#     PEmpty = enum.auto()


class NamedBarrierIds(enum.IntEnum):
    """Named barrier IDs for GDPA kernel synchronization.

    Barrier 0 is reserved for sync_threads(), so we start from 1.
    """
    TMEM_ALLOC_SYNC = 1  # Barrier for TMEM allocation sync
    MMA_P0 = 2           # Softmax signals done reading S0, MMA waits before writing SF1
    MMA_P1 = 3           # Softmax signals done reading S1, MMA waits before writing SF0


class TmemLayout:
    """TMEM allocation layout for GDPA kernel.

    TMEM Layout (512 columns total):
        0        64       128      192      256      384      512
        |--------|--------|--------|--------|--------|--------|
        |   S0 (0-128)    |   S1 (128-256)  |   O0   |   O1   |
        |   P0   |        |   P1   |        |(256-384|(384-512|
        |  (64)  |        | (192)  |        |        |        |

    For blockscaled MXFP8, scale factors are placed at P + P_size:
        - SF0 at P1 + P_size = ~224 (in S1 region) - for S0 GEMM
        - SF1 at P0 + P_size = ~96 (in S0 region) - for S1 GEMM
    """

    def __init__(self, n_block_size: int, head_dim_v_padded: int, q_stage: int = 2):
        self.n_block_size = n_block_size
        self.head_dim_v_padded = head_dim_v_padded
        self.q_stage = q_stage

        # Compute offsets
        self._s_offset = [0, n_block_size]  # S0 at 0, S1 at n_block_size
        self._s_to_p_offset = n_block_size // 4  # P offset within S region
        self._p_offset = [self._s_offset[i] + self._s_to_p_offset for i in range(2)]
        base_o = self._s_offset[-1] + n_block_size  # After S1
        self._o_offset = [base_o + i * head_dim_v_padded for i in range(q_stage)]
        self._total = self._o_offset[-1] + head_dim_v_padded

        # Prologue SF offsets (overlap O region, safe for first tile before O is used)
        # For the first tile, we place scale factors in the O region to avoid race conditions
        # with the S region where GEMM results are being written
        sf_size = 8  # kSizeSF in C++ - size of scale factor block
        sfk_offset = sf_size  # SFK offset relative to SFQ
        self._sf_prologue_offsets = [
            [self._o_offset[1], self._o_offset[1] + sfk_offset],  # SF0: SFQ0 in O1, SFK0 after
            [self._o_offset[0], self._o_offset[0] + sfk_offset],  # SF1: SFQ1 in O0, SFK1 after
        ]

    @property
    def s_offset(self) -> list:
        """S accumulator offsets: [S0, S1]"""
        return self._s_offset

    @property
    def o_offset(self) -> list:
        """O accumulator offsets: [O0, O1]"""
        return self._o_offset

    @property
    def p_offset(self) -> list:
        """P (intermediate softmax) offsets: [P0, P1]"""
        return self._p_offset

    @property
    def s_to_p_offset(self) -> int:
        """Offset from S region start to P region"""
        return self._s_to_p_offset

    @property
    def sf_prologue_offsets(self) -> list:
        """Prologue SF offsets [stage][sfq/sfk] - overlap O region for first tile"""
        return self._sf_prologue_offsets

    @property
    def total_columns(self) -> int:
        """Total TMEM columns used"""
        return self._total

    def validate(self, max_columns: int = 512) -> None:
        """Validate layout fits in SM100 TMEM."""
        assert self._total <= max_columns, f"TMEM overflow: {self._total} > {max_columns}"


# SM100 TMEM capacity
SM100_TMEM_CAPACITY_COLUMNS = 512


class FlashAttentionForwardSm100:

    arch = 100
    q_stage = 2

    def __init__(
        self,
        # dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        is_causal: bool = False,
        is_local: bool = False,
        pack_gqa: bool = False,
        m_block_size: int = 128,
        n_block_size: int = 128,
        is_persistent: bool = True,
        activation: str = "fast_gelu",
        unroll_kv: bool = True,
        blockscaled: bool = False,
        sf_vec_size: int = 32,
    ):
        # self.dtype = dtype
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.head_dim_v_padded = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.same_hdim_kv_padded = self.head_dim_padded == self.head_dim_v_padded
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.q_stage = FlashAttentionForwardSm100.q_stage
        assert self.q_stage in [1, 2]
        self.activation = activation

        # 2 Q tile per CTA
        self.cta_tiler = (self.q_stage * m_block_size, n_block_size, self.head_dim_padded)
        self.mma_tiler_qk = (m_block_size, n_block_size, self.head_dim_padded)
        self.mma_tiler_pv = (m_block_size, self.head_dim_v_padded, n_block_size)
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.cluster_shape_mn = (1, 1)
        self.is_persistent = is_persistent
        self.unroll_kv = unroll_kv

        self.is_causal = is_causal
        self.is_local = is_local
        self.qhead_per_kvhead = qhead_per_kvhead
        self.pack_gqa = pack_gqa
        if pack_gqa:
            assert m_block_size % self.qhead_per_kvhead == 0, "For PackGQA, m_block_size must be divisible by qhead_per_kvhead"
        # Does S1 need to wait for S0 to finish
        # self.s0_s1_barrier = self.head_dim_padded in [64, 96] and (not self.is_causal and not self.is_local)
        self.s0_s1_barrier = False
        self.overlap_sO_sQ = self.head_dim_padded == 192 and self.head_dim_v_padded >= 64
        if self.overlap_sO_sQ:
            assert self.head_dim_padded >= self.head_dim_v_padded  # We assume sQ is larger than sO
            self.is_persistent = False

        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.load_warp_id = 9
        self.epilogue_warp_ids = (10,)
        self.empty_warp_ids = (11,)
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                *self.epilogue_warp_ids,
                *self.empty_warp_ids,
            )
        )

        self.tmem_alloc_sync_bar_id = NamedBarrierIds.TMEM_ALLOC_SYNC

        # Named barrier IDs for MXFwdNamedBarriers (TMEM overlap synchronization)
        # These barriers coordinate between MMA warp and softmax warp for blockscaled GEMM:
        # - MMA_P0: softmax signals when done reading S0, MMA waits before writing SFQ1/SFK1 (overlaps S0)
        # - MMA_P1: softmax signals when done reading S1, MMA waits before writing SFQ0/SFK0 (overlaps S1)
        self.mbar_mma_p0_id = NamedBarrierIds.MMA_P0
        self.mbar_mma_p1_id = NamedBarrierIds.MMA_P1
        # Thread count for MXFwdNamedBarriers: softmax warps + MMA warp
        self.mbar_mma_threads = cute.arch.WARP_SIZE * (len(self.softmax0_warp_ids) + 1)  # 160 threads

        # TMEM layout abstraction
        self.tmem_layout = TmemLayout(
            n_block_size=self.n_block_size,
            head_dim_v_padded=self.head_dim_v_padded,
            q_stage=self.q_stage,
        )
        self.tmem_layout.validate(SM100_TMEM_CAPACITY_COLUMNS)

        # Aliases for backward compatibility (use self.tmem_layout properties instead)
        self.tmem_s_offset = self.tmem_layout.s_offset
        self.tmem_o_offset = self.tmem_layout.o_offset
        self.tmem_total = self.tmem_layout.total_columns
        self.tmem_s_to_p_offset = self.tmem_layout.s_to_p_offset
        self.tmem_p_offset = self.tmem_layout.p_offset

        # vec buffer for row_max & row_sum
        self.tmem_vec_offset = self.tmem_s_offset

        if self.head_dim_padded < 96:
            self.num_regs_softmax = 200
            self.num_regs_correction = 64  # TODO: re-allocate the unused "correction" regs to other warps
            self.num_regs_other = 48
        else:
            self.num_regs_softmax = 192 if self.is_causal or self.is_local else 184
            # self.num_regs_softmax = 176
            # self.num_regs_correction = 96
            # self.num_regs_correction = 80
            # self.num_regs_correction = 64 if self.is_causal or self.is_local else 80
            self.num_regs_correction = 64
            # self.num_regs_other = 32
            # self.num_regs_other = 64
            # self.num_regs_other = 80
            # self.num_regs_other = 48
            # self.num_regs_other = 96 if self.is_causal or self.is_local else 80
            self.num_regs_other = 64 if self.is_causal or self.is_local else 80
        self.num_regs_empty = 24

        self.buffer_align_bytes = 1024

        # Divisor for splitting TMEM store into two phases for P values
        # This needs to be <= min(tStP_r2t.shape[2]) to avoid writing 0 elements in first phase
        # FP16 has shape[2]=4, FP8 has shape[2]=2, so we use dtype-dependent calculation
        # Note: This will be set properly in __call__ when q_dtype is known
        self.tmem_store_split_divisor = None

        # Block scaling specific
        self.blockscaled = blockscaled
        self.sf_vec_size = sf_vec_size
        # Scale factor dtype for MXFP8: Float8E8M0FNU (exponent-only format)
        self.sf_dtype = cutlass.Float8E8M0FNU if blockscaled else None

    def _get_tmem_store_split_divisor(self) -> int:
        """Calculate divisor for splitting TMEM store into two phases for P values.

        This needs to be <= min(tStP_r2t.shape[2]) to avoid writing 0 elements in first phase.
        FP16/BF16 (16-bit) has shape[2]=4, FP8 (8-bit) has shape[2]=2.

        Returns:
            4 for FP16/BF16 (16-bit) dtypes, 2 for FP8 (8-bit) dtypes.
        """
        return 4 if self.q_dtype.width >= 16 else 2

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        self.kv_stage = 4 if self.q_dtype.width == 8 else 3
        self.acc_stage = 1
        self.epi_stage = 2
        # For hdim 192,128, we don't have enough smem to store all 3 stages of KV:
        # 128 x 192 x 2 bytes x 3 stages = 144KB, and we need 96KB for Q.
        # Instead we store smem as [smem_large, smem_small, smem_large], where smem_large is
        # 128 x 192 and smem_small is 128 x 128. We set the stride between the stages to be
        # 128 * 160, so that indexing the 0th and 2nd stages will get the right address,
        # but for the 1st stage we need to add or subtract (depending on phase) 128 x 64.
        self.uneven_kv_smem = self.head_dim_padded == 192 and self.head_dim_v_padded == 128 and self.kv_stage == 3
        self.uneven_kv_smem_offset = self.m_block_size * (self.head_dim_padded - self.head_dim_v_padded) // 2 if self.uneven_kv_smem else 0
        assert self.uneven_kv_smem_offset % 1024 == 0

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        mK: cute.Tensor,  # (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table
        mV: cute.Tensor,  # (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table
        mO: cute.Tensor,  # (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        stream: cuda.CUstream,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mMaxSeqlenQ: Int32 | int | None = None,
        mPageTable: Optional[cute.Tensor] = None,  # (b_k, max_num_pages_per_seq)
        softcap: Float32 | float | None = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        mTileToBatch: Optional[cute.Tensor] = None,
        mTileToHead: Optional[cute.Tensor] = None,
        mTileToBlock: Optional[cute.Tensor] = None,
        # Scale factor tensors for blockscaled (MXFP8)
        mSFQ: Optional[cute.Tensor] = None,  # Scale factors for Q
        mSFK: Optional[cute.Tensor] = None,  # Scale factors for K
        mSFV: Optional[cute.Tensor] = None,  # Scale factors for V
        # 128-aligned cu_seqlens for scale factor offsets (varlen MXFP8)
        mCuSeqlensSFQ: Optional[cute.Tensor] = None,
        mCuSeqlensSFK: Optional[cute.Tensor] = None,
        # Total padded tokens for SF layout (SF-only approach)
        # When provided, SF layout uses this instead of deriving from mQ/mK.shape
        total_sf_q: Int32 | int | None = None,
        total_sf_k: Int32 | int | None = None,
    ):
        """Execute the Fused Multi-Head Attention operation on the provided tensors.

        This method prepares the input tensors for processing, validates their shapes and types,
        configures the computation parameters, and launches the CUDA kernel.

        The method handles:
        1. Tensor layout transformations for specific memory access patterns
        2. Validation of tensor shapes and data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch with appropriate parameters
        """

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type

        self.tmem_store_split_divisor = self._get_tmem_store_split_divisor()

        # Assume all strides are divisible by 128 bits except the last stride
        new_stride = lambda t: (*(cute.assume(s, divby=128 // t.element_type.width) for s in t.stride[:-1]), t.stride[-1])
        mQ, mK, mV, mO = [cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t))) for t in (mQ, mK, mV, mO)]
        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ, mO = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=QO_layout_transpose))
            for t in (mQ, mO)
        ]
        # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there's cu_seqlens_k or (page_size, d, h_k, num_pages) if there's page_table
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK, mV = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=KV_layout_transpose))
            for t in (mK, mV)
        ]
        LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose)) if const_expr(mLSE is not None) else None
        # (s, d, h, b) -> (d, s, h, b)
        V_layout_transpose = [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=V_layout_transpose))

        self.q_major_mode = cutlass.utils.LayoutEnum.from_tensor(mQ).mma_major_mode()
        self.k_major_mode = cutlass.utils.LayoutEnum.from_tensor(mK).mma_major_mode()
        self.v_major_mode = cutlass.utils.LayoutEnum.from_tensor(mV).mma_major_mode()
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)

        if const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mQ is not supported")
        if const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of mK is not supported")
        if const_expr(self.v_major_mode != tcgen05.OperandMajorMode.MN):
            raise RuntimeError("The layout of mV is not supported")

        if const_expr(mMaxSeqlenQ is not None):
            mMaxSeqlenQ = Int32(mMaxSeqlenQ)

        # check type consistency
        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        self._setup_attributes()
        self.use_tma_O = self.arch >= 90 and mCuSeqlensQ is None and mSeqUsedQ is None
        # This can be tuned
        self.e2e_freq = 4
        if const_expr(self.head_dim_padded > 64 and not self.is_causal and not self.is_local and self.pack_gqa):
            self.e2e_freq = 32 if mCuSeqlensQ is not None or mSeqUsedQ is not None else 10

        cta_group = tcgen05.CtaGroup.ONE
        # the intermediate tensor p is from tmem & mK-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        # When blockscaled, use blockscaled tiled_mma for QK which supports SFA/SFB fields
        # Otherwise, use the standard trivial tiled_mma
        if const_expr(self.blockscaled):
            tiled_mma_qk = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                cutlass.Float8E4M3FN,  # FP8 dtype for blockscaled
                self.q_major_mode,
                self.k_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                cta_group,
                self.mma_tiler_qk[:2],
            )
        else:
            tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
                self.q_dtype,
                self.q_major_mode,
                self.k_major_mode,
                self.qk_acc_dtype,
                cta_group,
                self.mma_tiler_qk[:2],
            )
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.mma_tiler_pv[:2],
            p_source,
        )

        # Create blockscaled tiled_mma_pv for blockscaled P*V GEMM
        # This needs a_source=TMEM since P comes from TMEM (TS mode)
        tiled_mma_pv_blockscaled = None
        if const_expr(self.blockscaled):
            tiled_mma_pv_blockscaled = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                cutlass.Float8E4M3FN,  # FP8 dtype for blockscaled
                p_major_mode,
                self.v_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                cta_group,
                self.mma_tiler_pv[:2],
                p_source,  # A operand comes from TMEM (TS mode for P*V)
            )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )

        self.epi_tile = self.mma_tiler_pv[:2]

        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.q_stage,
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage,
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler_pv, self.q_dtype, self.acc_stage,
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv, self.mma_tiler_pv, self.v_dtype, self.kv_stage,
        )
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype, self.o_layout, self.epi_tile, self.epi_stage,
        )

        sSFQ_layout = None
        sSFK_layout = None
        sSFV_layout = None
        sSFP_layout = None
        if const_expr(self.blockscaled):
            # SFQ uses the blockscaled tiled_mma_qk (same shape as Q/K but blockscaled)
            sSFQ_layout = blockscaled_utils.make_smem_layout_sfa(
                tiled_mma_qk, self.mma_tiler_qk, self.sf_vec_size, self.q_stage,
            )
            # SFK uses the blockscaled tiled_mma_qk for operand B (K's scale factors)
            sSFK_layout = blockscaled_utils.make_smem_layout_sfb(
                tiled_mma_qk, self.mma_tiler_qk, self.sf_vec_size, self.kv_stage,
            )
            # SFV: V is operand B for PV GEMM, use mma_tiler_pv for correct dimensions
            # mma_tiler_pv = (M, D, N) where D=head_dim_v is the "N" mode for operand B
            # This matches V's memory layout (d, s_k, h_k, b_k)
            sSFV_layout = blockscaled_utils.make_smem_layout_sfb(
                tiled_mma_pv_blockscaled, self.mma_tiler_pv, self.sf_vec_size, self.kv_stage,
            )
            # SFP uses the blockscaled tiled_mma_pv for operand A (P's scale factors)
            # SFP needs 2 stages for P0/P1 double-buffering (unlike tP which uses acc_stage=1)
            sfp_num_stages = 2  # Always 2 for blockscaled P0/P1 double-buffering
            sSFP_layout = blockscaled_utils.make_smem_layout_sfa(
                tiled_mma_pv_blockscaled, self.mma_tiler_pv, self.sf_vec_size, sfp_num_stages,
            )
        if const_expr(not self.same_hdim_kv_padded):
            # sK and sV are using the same physical smem so we need to adjust the stride so that they line up
            stride_sK = const_expr(max(sK_layout.outer.stride[-1], 0))  # take max to turn tuple to Int32
            stride_sV = const_expr(max(sV_layout.outer.stride[-1], 0))
            stage_stride = const_expr(max(stride_sK, stride_sV) if not self.uneven_kv_smem else (stride_sK + stride_sV) // 2)
            sK_layout = cute.make_composed_layout(sK_layout.inner, 0, cute.make_layout((*sK_layout.outer.shape[:-1], self.kv_stage), stride=(*sK_layout.outer.stride[:-1], stage_stride)))
            sV_layout = cute.make_composed_layout(sV_layout.inner, 0, cute.make_layout((*sV_layout.outer.shape[:-1], self.kv_stage), stride=(*sV_layout.outer.stride[:-1], stage_stride)))

        if const_expr(self.pack_gqa):
            shape_Q_packed = ((self.qhead_per_kvhead, mQ.shape[0]), mQ.shape[1], mK.shape[2], *mQ.shape[3:])
            stride_Q_packed = ((mQ.stride[2], mQ.stride[0]), mQ.stride[1], mQ.stride[2] * self.qhead_per_kvhead, *mQ.stride[3:])
            mQ = cute.make_tensor(mQ.iterator, cute.make_layout(shape_Q_packed, stride=stride_Q_packed))
            shape_O_packed = ((self.qhead_per_kvhead, mO.shape[0]), mK.shape[1], mK.shape[2], *mO.shape[3:])
            stride_O_packed = ((mO.stride[2], mO.stride[0]), mO.stride[1], mO.stride[2] * self.qhead_per_kvhead, *mO.stride[3:])
            mO = cute.make_tensor(mO.iterator, cute.make_layout(shape_O_packed, stride=stride_O_packed))
            if const_expr(mLSE is not None):
                shape_LSE_packed = ((self.qhead_per_kvhead, mLSE.shape[0]), mK.shape[2], *mLSE.shape[2:])
                stride_LSE_packed = ((mLSE.stride[1], mLSE.stride[0]), mLSE.stride[1] * self.qhead_per_kvhead, *mLSE.stride[2:])
                mLSE = cute.make_tensor(mLSE.iterator, cute.make_layout(shape_LSE_packed, stride=stride_LSE_packed))

        # TMA load for Q
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
        )

        # TMA load for K
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mK,
            cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for V
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mV,
            cute.select(sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_pv,
            tiled_mma_pv,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_SFQ, tma_tensor_SFQ = None, None
        tma_atom_SFK, tma_tensor_SFK = None, None
        tma_atom_SFV, tma_tensor_SFV = None, None
        if const_expr(self.blockscaled):
            # For SF-only approach: use total_sf_q/k if provided (128-aligned total)
            # Otherwise, derive from mQ/mK shape (OLD approach where data is also padded)
            if const_expr(total_sf_q is not None):
                # Replace first dimension with padded total for SF layout
                sfq_shape = (total_sf_q, mQ.shape[1], mQ.shape[2])
            else:
                sfq_shape = mQ.shape
            sfq_layout = blockscaled_utils.tile_atom_to_shape_SF(sfq_shape, self.sf_vec_size)
            mSFQ = cute.make_tensor(mSFQ.iterator, sfq_layout)

            # Slice out the stage dimension (last mode) from sSFQ_layout
            # sSFQ_layout has shape: (((Atom_Inst_M, Rest_M),(Atom_Inst_K, Rest_K)), MMA_M, MMA_K, STAGE)
            # We need to pass only the first 3 modes (without stage) to make_tiled_tma_atom_A
            sSFQ_layout_per_stage = cute.select(sSFQ_layout, mode=[0, 1, 2])
            tma_atom_SFQ, tma_tensor_SFQ = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                mSFQ,
                sSFQ_layout_per_stage,
                self.mma_tiler_qk,
                tiled_mma_qk,  # Use blockscaled tiled_mma_qk for SFQ TMA
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )

            # SFK TMA atom creation (K's scale factors, operand B)
            # mK has shape (s_k, d, h_k, b_k) after transpose, or (total_k, d, h_k) for varlen
            # For SF-only approach: use total_sf_k if provided (128-aligned total)
            if const_expr(total_sf_k is not None):
                # Replace first dimension with padded total for SF layout
                sfk_shape = (total_sf_k, mK.shape[1], mK.shape[2]) if len(mK.shape) == 3 else (total_sf_k, mK.shape[1], mK.shape[2], mK.shape[3])
            else:
                sfk_shape = mK.shape
            sfk_layout = blockscaled_utils.tile_atom_to_shape_SF(sfk_shape, self.sf_vec_size)
            mSFK = cute.make_tensor(mSFK.iterator, sfk_layout)

            sSFK_layout_per_stage = cute.select(sSFK_layout, mode=[0, 1, 2])
            tma_atom_SFK, tma_tensor_SFK = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mSFK,
                sSFK_layout_per_stage,
                self.mma_tiler_qk,
                tiled_mma_qk,  # Use blockscaled tiled_mma_qk for SFK TMA
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )

            # SFV TMA atom creation (V's scale factors, operand B for PV GEMM)
            # Only create if mSFV is provided (Phase 1: optional SFV loading)
            if const_expr(mSFV is not None):
                # mV has shape (d, s_k, h_k, b_k) after transpose, or (d, total_k, h_k) for varlen
                # For PV GEMM: V is operand B with shape (D, N) = (head_dim_v, s_k)
                # For SF-only approach: use total_sf_k if provided (128-aligned total)
                if const_expr(total_sf_k is not None):
                    # Replace second dimension (s_k) with padded total for SF layout
                    sfv_shape = (mV.shape[0], total_sf_k, mV.shape[2]) if len(mV.shape) == 3 else (mV.shape[0], total_sf_k, mV.shape[2], mV.shape[3])
                else:
                    sfv_shape = mV.shape
                sfv_layout = blockscaled_utils.tile_atom_to_shape_SF(sfv_shape, self.sf_vec_size)

                mSFV_tensor = cute.make_tensor(mSFV.iterator, sfv_layout)

                sSFV_layout_per_stage = cute.select(sSFV_layout, mode=[0, 1, 2])

                # SFV is operand B for PV GEMM, use mma_tiler_pv and tiled_mma_pv_blockscaled
                # This matches V's memory layout (d, s_k, h_k, b_k) and sSFV_layout
                tma_atom_SFV, tma_tensor_SFV = cute.nvgpu.make_tiled_tma_atom_B(
                    tma_load_op,
                    mSFV_tensor,
                    sSFV_layout_per_stage,
                    self.mma_tiler_pv,  # PV tiler for V operand
                    tiled_mma_pv_blockscaled,  # PV blockscaled MMA
                    self.cluster_layout_vmnk.shape,
                    internal_type=cutlass.Int16,
                )

        o_cta_v_layout = cute.composition(cute.make_identity_layout(mO.shape), self.epi_tile)

        if const_expr(not self.use_tma_O):
            self.epilogue_warp_ids = (10, 11)
            self.empty_warp_ids = ()
        self.num_epilogue_threads = cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        if const_expr(self.use_tma_O):
            tma_atom_O, mO = cpasync.make_tiled_tma_atom(
                tma_store_op,
                mO,
                cute.select(sO_layout, mode=[0, 1]),
                o_cta_v_layout,
            )
            gmem_tiled_copy_O = None
        else:
            tma_atom_O = None
            universal_copy_bits = 128
            async_copy_elems = universal_copy_bits // self.o_dtype.width
            atom_universal_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.o_dtype, num_bits_per_copy=universal_copy_bits,
            )
            tO_shape_dim_1 = sO_layout.outer.shape[1][0] // async_copy_elems
            tO_layout = cute.make_ordered_layout(
                (self.num_epilogue_threads // tO_shape_dim_1, tO_shape_dim_1), order=(1, 0),
            )
            # So that we don't have to check if we overshoot kBlockM when we store O
            assert self.m_block_size % tO_layout.shape[0] == 0
            vO_layout = cute.make_layout((1, async_copy_elems))
            gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, tO_layout, vO_layout)

        self.tma_copy_q_bytes = int(cute.size_in_bytes(self.q_dtype, cute.select(sQ_layout, mode=[0, 1, 2])))
        self.tma_copy_k_bytes = int(cute.size_in_bytes(self.k_dtype, cute.select(sK_layout, mode=[0, 1, 2])))
        self.tma_copy_v_bytes = int(cute.size_in_bytes(self.v_dtype, cute.select(sV_layout, mode=[0, 1, 2])))

        # Add scale factor sizes to transaction bytes when blockscaled is enabled
        if const_expr(self.blockscaled):
            self.tma_copy_sfq_bytes = int(cute.size_in_bytes(self.sf_dtype, cute.select(sSFQ_layout, mode=[0, 1, 2])))
            self.tma_copy_sfk_bytes = int(cute.size_in_bytes(self.sf_dtype, cute.select(sSFK_layout, mode=[0, 1, 2])))
            self.tma_copy_q_bytes += self.tma_copy_sfq_bytes
            self.tma_copy_k_bytes += self.tma_copy_sfk_bytes
            # Only add SFV bytes if mSFV is provided (Phase 1: optional SFV loading)
            if const_expr(mSFV is not None):
                self.tma_copy_sfv_bytes = int(cute.size_in_bytes(self.sf_dtype, cute.select(sSFV_layout, mode=[0, 1, 2])))
                self.tma_copy_v_bytes += self.tma_copy_sfv_bytes
            else:
                self.tma_copy_sfv_bytes = 0
        else:
            self.tma_copy_sfq_bytes = 0
            self.tma_copy_sfk_bytes = 0
            self.tma_copy_sfv_bytes = 0

        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            if const_expr(self.is_persistent):
                TileScheduler = SingleTileVarlenSchedulerPersistent
            else:
                TileScheduler = SingleTileVarlenSchedulerSimple if const_expr(mMaxSeqlenQ is not None) else SingleTileVarlenScheduler
        else:
            if const_expr(self.is_causal or self.is_local):
                TileScheduler = SingleTileLPTScheduler
            else:
                TileScheduler = SingleTileScheduler if const_expr(not self.is_persistent) else StaticPersistentTileScheduler
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.cta_tiler[0]),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]) if const_expr(mCuSeqlensQ is None) else cute.size(mCuSeqlensQ.shape[0] - 1),
            cute.size(mK.shape[0]) if const_expr(mPageTable is None) else mK.shape[0] * mPageTable.shape[1],
            mQ.shape[1],
            mV.shape[0],  # Note that this is different from Sm90 since we transpose mV in Sm100
            total_len=cute.size(mQ.shape[0]) if const_expr(mCuSeqlensQ is not None) else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            mMaxSeqlenQ=mMaxSeqlenQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,
            lpt=self.is_causal or self.is_local,
            mTileToBatch=mTileToBatch,
            mTileToHead=mTileToHead,
            mTileToBlock=mTileToBlock
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        self.mbar_load_q_full_offset = 0
        self.mbar_load_q_empty_offset = self.mbar_load_q_full_offset + self.q_stage
        self.mbar_load_kv_full_offset = self.mbar_load_q_empty_offset + self.q_stage
        self.mbar_load_kv_empty_offset = self.mbar_load_kv_full_offset + self.kv_stage
        self.mbar_P_full_1_offset = self.mbar_load_kv_empty_offset + self.kv_stage
        self.mbar_S_full_offset = self.mbar_P_full_1_offset + 2
        self.mbar_O_full_offset = self.mbar_S_full_offset + 2
        # self.mbar_softmax_corr_full_offset = self.mbar_O_full_offset + 2
        # self.mbar_softmax_corr_empty_offset = self.mbar_softmax_corr_full_offset + 2
        self.mbar_corr_epi_full_offset = self.mbar_O_full_offset + self.epi_stage
        self.mbar_corr_epi_empty_offset = self.mbar_corr_epi_full_offset + self.epi_stage
        self.mbar_s0_s1_sequence_offset = self.mbar_corr_epi_empty_offset + 2
        self.mbar_tmem_dealloc_offset = self.mbar_s0_s1_sequence_offset + 8
        self.mbar_P_full_2_offset = self.mbar_tmem_dealloc_offset + 1
        self.mbar_total = self.mbar_P_full_2_offset + 2

        sO_size = cute.cosize(sO_layout) if const_expr(not self.overlap_sO_sQ) else 0

        @cute.struct
        class SharedStorage:
            # m_barriers for pipelines
            mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mbar_total]
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            # store row max and row sum
            sScale: cute.struct.MemRange[Float32, self.q_stage * self.m_block_size * 2]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(sQ_layout)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                # cute.cosize(sK_layout) is correct even in the case of self.uneven_kv_smem
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]
            # Scale factor SMEM storage for blockscaled (MXFP8)
            sSFQ: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype if const_expr(self.blockscaled) else cutlass.Float8E8M0FNU,
                    cute.cosize(sSFQ_layout) if const_expr(self.blockscaled) else 0,
                ],
                self.buffer_align_bytes,
            ]
            sSFK: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype if const_expr(self.blockscaled) else cutlass.Float8E8M0FNU,
                    cute.cosize(sSFK_layout) if const_expr(self.blockscaled) else 0,
                ],
                self.buffer_align_bytes,
            ]
            sSFV: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype if const_expr(self.blockscaled) else cutlass.Float8E8M0FNU,
                    cute.cosize(sSFV_layout) if const_expr(self.blockscaled) else 0,
                ],
                self.buffer_align_bytes,
            ]
            # SFP: P's scale factors (for blockscaled PV GEMM)
            sSFP: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype if const_expr(self.blockscaled) else cutlass.Float8E8M0FNU,
                    cute.cosize(sSFP_layout) if const_expr(self.blockscaled) else 0,
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # If there's tanh softcapping, we do tanh(scores * softmax_scale / softcap_val) * softcap_val.
        # Right after this, we multiply by log2(e) before applying exp2.
        # To reduce the number of instructions, we instead pre-multiply softmax_scale / softcap_val
        # (assigning it to softcap_val) and pre-multiply softcap_val * log2(e)
        # (assigning it to softmax_scale_log2).
        LOG2_E = math.log2(math.e)
        if const_expr(softcap is None):
            softmax_scale_log2 = softmax_scale * LOG2_E
            softcap_val = None
        else:
            softmax_scale_log2 = softcap * LOG2_E
            softcap_val = Float32(softmax_scale / softcap)
        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)
        # Launch the kernel synchronously
        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            # Scale factor TMA parameters (for blockscaled MXFP8)
            tma_tensor_SFQ,
            tma_atom_SFQ,
            sSFQ_layout,
            tma_tensor_SFK,
            tma_atom_SFK,
            sSFK_layout,
            tma_tensor_SFV,
            tma_atom_SFV,
            sSFV_layout,
            sSFP_layout,
            softmax_scale_log2,
            softcap_val,
            window_size_left,
            window_size_right,
            learnable_sink,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sO_layout,
            gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tiled_mma_pv_blockscaled,
            tile_sched_params,
            # 128-aligned cu_seqlens for scale factor offsets (varlen MXFP8)
            mCuSeqlensSFQ,
            mCuSeqlensSFK,
        )(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # (s_q, d, h, b) or (total_q, d, h) if there is cu_seqlens_q
        mK: cute.Tensor,  # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there is cu_seqlens_k or (page_size, d, h_k, num_pages) if there is page_table
        mV: cute.Tensor,  # (d, s_k, h_k, b_k) or (d, total_k, h_k) if there is cu_seqlens_k or (d, page_size, h_k, num_pages) if there is page_table
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: Optional[cute.CopyAtom],
        mSFQ: Optional[cute.Tensor],   # Note mSFQ is TMA tensor (coord tensor)
        tma_atom_SFQ: Optional[cute.CopyAtom],
        sSFQ_layout: Optional[cute.Layout],
        mSFK: Optional[cute.Tensor],   # SFK TMA tensor (coord tensor)
        tma_atom_SFK: Optional[cute.CopyAtom],
        sSFK_layout: Optional[cute.Layout],
        mSFV: Optional[cute.Tensor],   # SFV TMA tensor (coord tensor)
        tma_atom_SFV: Optional[cute.CopyAtom],
        sSFV_layout: Optional[cute.Layout],
        sSFP_layout: Optional[cute.Layout],  # SFP SMEM layout (P's scale factors)
        softmax_scale_log2: Float32,
        softcap_val: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_O: Optional[cute.TiledCopy],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_pv_blockscaled: Optional[cute.TiledMma],
        tile_sched_params: ParamsBase,
        # 128-aligned cu_seqlens for scale factor offsets (varlen MXFP8)
        mCuSeqlensSFQ: Optional[cute.Tensor] = None,
        mCuSeqlensSFK: Optional[cute.Tensor] = None,
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Softmax warps: Compute softmax normalization on attention scores
        4. Correction warps: Apply adjustments to intermediate results
        5. Epilogue warp: Handles final output transformation and storage

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking.
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)
            if const_expr(tma_atom_O is not None):
                cpasync.prefetch_descriptor(tma_atom_O)

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mbar_ptr = storage.mbar_ptr.data_ptr()
        if warp_idx == 1:
            # Init "full" barrier with number of producers, "empty" barrier with number of consumers
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_load_q_full_offset + i, len([self.load_warp_id]))
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_load_q_empty_offset + i, len([self.mma_warp_id]))
        # if warp_idx == 2:
        #     for i in cutlass.range_constexpr(2):
        #         cute.arch.mbarrier_init(mbar_ptr + self.mbar_softmax_corr_empty_offset + i, cute.arch.WARP_SIZE * 4)
        #         cute.arch.mbarrier_init(mbar_ptr + self.mbar_softmax_corr_full_offset + i, cute.arch.WARP_SIZE * 4)
        if warp_idx == 3:
            if const_expr(self.s0_s1_barrier):
                for i in cutlass.range_constexpr(8):
                    cute.arch.mbarrier_init(mbar_ptr + self.mbar_s0_s1_sequence_offset + i, cute.arch.WARP_SIZE)
        if warp_idx == 4:
            for i in cutlass.range_constexpr(self.q_stage):
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_corr_epi_full_offset + i, cute.arch.WARP_SIZE * len(self.softmax0_warp_ids))
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_corr_epi_empty_offset + i, cute.arch.WARP_SIZE * len(self.epilogue_warp_ids))
        if warp_idx == 5:
            for i in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_P_full_1_offset + i, cute.arch.WARP_SIZE * len(self.softmax0_warp_ids))
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_S_full_offset + i, len([self.mma_warp_id]))
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_O_full_offset + i, len([self.mma_warp_id]))
        if warp_idx == 6:
            for i in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(mbar_ptr + self.mbar_P_full_2_offset + i, cute.arch.WARP_SIZE * len(self.softmax0_warp_ids))
        if warp_idx == 7:
            cute.arch.mbarrier_init(
                mbar_ptr + self.mbar_tmem_dealloc_offset,
                cute.arch.WARP_SIZE
                * len(
                    (
                        *self.softmax0_warp_ids,
                        *self.softmax1_warp_ids,
                    )
                ),
            )
        # Relying on pipeline_kv constructor to call mbarrier_init_fence and sync
        pipeline_kv = self.make_and_init_load_kv_pipeline(mbar_ptr + self.mbar_load_kv_full_offset)

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # sQ_pi = storage.sQ.get_tensor(sQ_layout)
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # sK_pi = storage.sK.get_tensor(sK_layout)
        # (MMA, MMA_K, MMA_D, PIPE)
        # Strip swizzle info to reuse smem
        sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)
        # Scale factor SMEM tensors (for blockscaled MXFP8)
        sSFQ = None
        sSFK = None
        sSFV = None
        sSFP = None
        if const_expr(self.blockscaled):
            sSFQ = storage.sSFQ.get_tensor(sSFQ_layout)
            sSFK = storage.sSFK.get_tensor(sSFK_layout)
            sSFV = storage.sSFV.get_tensor(sSFV_layout)
            sSFP = storage.sSFP.get_tensor(sSFP_layout)
        if const_expr(not self.overlap_sO_sQ):
            sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        else:
            sO = cute.make_tensor(cute.recast_ptr(sQ.iterator, sO_layout.inner), sO_layout.outer)

        sScale = storage.sScale.get_tensor(cute.make_layout(self.q_stage * self.m_block_size * 2))

        cta_mma_qk = tiled_mma_qk.get_slice(0)  # default 1SM
        cta_mma_pv = tiled_mma_pv.get_slice(0)  # default 1SM
        # For blockscaled, use the blockscaled tiled_mma for partitions
        cta_mma_pv_blockscaled = None
        if const_expr(self.blockscaled):
            cta_mma_pv_blockscaled = tiled_mma_pv_blockscaled.get_slice(0)

        qk_acc_shape = cta_mma_qk.partition_shape_C((self.mma_tiler_qk[0], self.mma_tiler_qk[1]))
        tStS_fake = cta_mma_qk.make_fragment_C(qk_acc_shape)
        # This is a fake tensor, by right need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem,
                                 assumed_align=16)
        tStS = cute.make_tensor(tmem_ptr, tStS_fake.layout)

        pv_acc_shape = cta_mma_pv.partition_shape_C((self.mma_tiler_pv[0], self.mma_tiler_pv[1]))
        tOtO = cta_mma_pv.make_fragment_C(pv_acc_shape)

        tStSs = tuple(cute.make_tensor(tStS.iterator + self.tmem_s_offset[stage], tStS.layout)
                      for stage in range(2))
        tOtOs = tuple(cute.make_tensor(tOtO.iterator + self.tmem_o_offset[stage], tOtO.layout)
                      for stage in range(self.q_stage))

        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = cta_mma_pv.make_fragment_A(tP)[None, None, None, 0]

        tOrPs = [cute.make_tensor(
            tOrP.iterator
            + self.qk_acc_dtype.width // self.q_dtype.width * self.tmem_p_offset[stage],
            tOrP.layout,
        ) for stage in range(2)]

        # For blockscaled, create partitions using the blockscaled tiled_mma
        tOtOs_blockscaled = None
        tOrPs_blockscaled = None
        if const_expr(self.blockscaled):
            pv_acc_shape_bs = cta_mma_pv_blockscaled.partition_shape_C((self.mma_tiler_pv[0], self.mma_tiler_pv[1]))
            tOtO_bs = cta_mma_pv_blockscaled.make_fragment_C(pv_acc_shape_bs)
            tOtOs_blockscaled = tuple(cute.make_tensor(tOtO_bs.iterator + self.tmem_o_offset[stage], tOtO_bs.layout)
                          for stage in range(self.q_stage))

            tP_bs = cute.make_tensor(tStS.iterator, tP_layout.outer)
            tOrP_bs = cta_mma_pv_blockscaled.make_fragment_A(tP_bs)[None, None, None, 0]
            tOrPs_blockscaled = [cute.make_tensor(
                tOrP_bs.iterator
                + self.qk_acc_dtype.width // self.q_dtype.width * self.tmem_p_offset[stage],
                tOrP_bs.layout,
            ) for stage in range(2)]

        block_info = BlockInfo(
            # This is cta_tiler, not mma_tiler_qk, since we move by block by (2 * mma_tiler[0], mma_tiler[1])
            self.cta_tiler[0], self.cta_tiler[1], self.is_causal, self.is_local,
            window_size_left, window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK,
            seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0] if const_expr(mPageTable is None) else mK.shape[0] * mPageTable.shape[1],
            mCuSeqlensQ=mCuSeqlensQ, mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ, mSeqUsedK=mSeqUsedK,
            mCuSeqlensSFQ=mCuSeqlensSFQ, mCuSeqlensSFK=mCuSeqlensSFK,
        )
        AttentionMaskCls = partial(
            AttentionMask, self.m_block_size, self.n_block_size,
            window_size_left=window_size_left, window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(len(self.empty_warp_ids) > 0):
            if warp_idx == self.empty_warp_ids[0]:
                cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            self.load(
                cta_mma_qk,
                cta_mma_pv,
                mQ,
                mSFQ,
                mSFK,
                mSFV,
                mK,
                mV,
                sQ,
                sSFQ,
                sSFK,
                sSFV,
                sK,
                sV,
                mPageTable,
                tma_atom_Q,
                tma_atom_SFQ,
                tma_atom_K,
                tma_atom_SFK,
                tma_atom_V,
                tma_atom_SFV,
                pipeline_kv,
                mbar_ptr,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
        # if warp_idx == self.mma_warp_id or warp_idx == self.empty_warp_ids:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            # Alloc tmem buffer
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            if warp_idx == self.mma_warp_id:
                cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
                cute.arch.sync_warp()

            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                tiled_mma_pv_blockscaled,
                sQ,
                sK,
                sV,
                sSFQ,
                sSFK,
                sSFV,
                sSFP,
                sQ_layout.inner,
                sK_layout.inner,
                sV_layout.inner,
                tStSs,
                tOtOs,
                tOrPs,
                tOtOs_blockscaled,
                tOrPs_blockscaled,
                pipeline_kv,
                mbar_ptr,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
            )

            # if warp_idx == self.mma_warp_id:
            # dealloc tmem buffer
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_tmem_dealloc_offset, 0)
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            #  Retrieving tmem ptr and make acc
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                Float32,
                alignment=16,
                ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
            )
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.epilogue_warp_ids[0] and warp_idx <= self.epilogue_warp_ids[-1]:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            self.epilogue_s2g(mO, sO, gmem_tiled_copy_O, tma_atom_O, mbar_ptr, SeqlenInfoCls, TileSchedulerCls)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx < self.mma_warp_id:
            # increase register after decreasing
            cute.arch.warpgroup_reg_alloc(self.num_regs_softmax)
            softmax_loop = partial(
                self.softmax_loop,
                softmax_scale_log2=softmax_scale_log2,
                cta_mma_qk=cta_mma_qk,
                cta_mma_pv=cta_mma_pv,
                sScale=sScale,
                mLSE=mLSE,
                sO=sO,
                learnable_sink=learnable_sink,
                mbar_ptr=mbar_ptr,
                block_info=block_info,
                SeqlenInfoCls=SeqlenInfoCls,
                AttentionMaskCls=AttentionMaskCls,
                TileSchedulerCls=TileSchedulerCls,
                sSFP=sSFP if const_expr(self.blockscaled) else None,
            )

            if const_expr(not self.s0_s1_barrier):
                stage = Int32(0 if warp_idx < self.softmax1_warp_ids[0] else 1)
                softmax_loop(
                    stage=stage,
                    tStSi=cute.make_tensor(tStS.iterator + (self.tmem_s_offset[0] if stage == 0 else self.tmem_s_offset[1]), tStS.layout),
                    tOtOi=tOtOs[0] if stage == 0 else tOtOs[1])
                cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)
            else:
                # If there's s0_s1_barrier, it's faster to have 2 WGs having different code
                if warp_idx < self.softmax1_warp_ids[0]:
                    tStSi = cute.make_tensor(tStS.iterator + self.tmem_s_offset[0], tStS.layout)
                    softmax_loop(stage=0, tStSi=tStSi, tOtOi=tOtOs[0])
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)
                if warp_idx < self.mma_warp_id and warp_idx >= self.softmax1_warp_ids[0]:
                    tStSi = cute.make_tensor(tStS.iterator + self.tmem_s_offset[1], tStS.layout)
                    softmax_loop(stage=1, tStSi=tStSi, tOtOi=tOtOs[1])
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_tmem_dealloc_offset)


        return

    @cute.jit
    def load(
        self,
        cta_mma_qk: cute.core.ThrMma,
        cta_mma_pv: cute.core.ThrMma,
        mQ: cute.Tensor,
        mSFQ: cute.Tensor,
        mSFK: cute.Tensor,  # SFK TMA tensor (coord tensor)
        mSFV: cute.Tensor,  # SFV TMA tensor (coord tensor)
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sSFQ: cute.Tensor,
        sSFK: cute.Tensor,
        sSFV: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_SFQ: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_SFK: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_SFV: cute.CopyAtom,
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):

        q_producer_phase = Int32(1)
        kv_producer_state = cutlass.pipeline.make_pipeline_state(cutlass.pipeline.PipelineUserType.Producer, self.kv_stage)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(not seqlen.has_cu_seqlens_q):
                mQ_cur = mQ[None, None, head_idx, batch_idx]
            else:
                offset = seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
                mQ_cur = cute.domain_offset((offset, 0), mQ[None, None, head_idx])
            gQ = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_qk, mode=[0, 2]), (None, 0))

            head_idx_kv = head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
            if const_expr(mPageTable is None):
                if const_expr(not seqlen.has_cu_seqlens_k):
                    mK_cur, mV_cur = [t[None, None, head_idx_kv, batch_idx] for t in (mK, mV)]
                else:
                    mK_cur = cute.domain_offset((seqlen.offset_k, 0), mK[None, None, head_idx_kv])
                    mV_cur = cute.domain_offset((0, seqlen.offset_k), mV[None, None, head_idx_kv])
                gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0))
                gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None))
            else:
                # Need to keep batch coord None since we'll index into it with page idx
                mK_cur, mV_cur = [t[None, None, head_idx_kv, None] for t in (mK, mV)]
                gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0, None))
                gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None, None))
            tSgQ = cta_mma_qk.partition_A(gQ)
            tSgK = cta_mma_qk.partition_B(gK)
            tOgV = cta_mma_pv.partition_B(gV)
            tQsQ, tQgQ = cpasync.tma_partition(
                tma_atom_Q,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sQ, 0, 3),
                cute.group_modes(tSgQ, 0, 3),
            )
            tKsK, tKgK = cpasync.tma_partition(
                tma_atom_K,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sK, 0, 3),
                cute.group_modes(tSgK, 0, 3),
            )
            tVsV, tVgV = cpasync.tma_partition(
                tma_atom_V,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sV, 0, 3),
                cute.group_modes(tOgV, 0, 3),
            )

            tSFQsSFQ, tSFQgSFQ = None, None
            tSFKsSFK, tSFKgSFK = None, None
            tSFVsSFV, tSFVgSFV = None, None
            if const_expr(self.blockscaled):
                if const_expr(not seqlen.has_cu_seqlens_q):  # Non-varlen (untested)
                    mSFQ_cur = mSFQ[None, None, head_idx, batch_idx]
                else:
                    # Use offset_sf_q for 128-aligned scale factor offsets
                    offset = seqlen.offset_sf_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_sf_q)
                    mSFQ_cur = cute.domain_offset((offset, 0), mSFQ[None, None, head_idx])

                gSFQ = cute.local_tile(mSFQ_cur, cute.select(self.mma_tiler_qk, mode=[0, 2]), (None, 0))
                tSgSFQ = cta_mma_qk.partition_A(gSFQ)
                tSFQsSFQ, tSFQgSFQ = cpasync.tma_partition(
                    tma_atom_SFQ,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sSFQ, 0, 3),
                    cute.group_modes(tSgSFQ, 0, 3),
                )
                # Filter zeros like QUACK GEMM does for scale factors
                tSFQsSFQ = cute.filter_zeros(tSFQsSFQ)
                tSFQgSFQ = cute.filter_zeros(tSFQgSFQ)

                # SFK TMA partition (K's scale factors, operand B)
                head_idx_kv = head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
                if const_expr(mPageTable is None):
                    if const_expr(not seqlen.has_cu_seqlens_k):
                        mSFK_cur = mSFK[None, None, head_idx_kv, batch_idx]
                    else:
                        # Use offset_sf_k for 128-aligned scale factor offsets
                        mSFK_cur = cute.domain_offset((seqlen.offset_sf_k, 0), mSFK[None, None, head_idx_kv])
                else:
                    # Paged attention - need to keep batch coord None
                    mSFK_cur = mSFK[None, None, head_idx_kv, None]

                gSFK = cute.local_tile(mSFK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0)) if const_expr(mPageTable is None) else cute.local_tile(mSFK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0, None))
                tSgSFK = cta_mma_qk.partition_B(gSFK)
                tSFKsSFK, tSFKgSFK = cpasync.tma_partition(
                    tma_atom_SFK,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sSFK, 0, 3),
                    cute.group_modes(tSgSFK, 0, 3),
                )
                # Filter zeros like QUACK GEMM does for scale factors
                tSFKsSFK = cute.filter_zeros(tSFKsSFK)
                tSFKgSFK = cute.filter_zeros(tSFKgSFK)

                # SFV TMA partition (V's scale factors, operand B for PV GEMM)
                # Only partition if mSFV is provided (Phase 1: optional SFV loading)
                if const_expr(mSFV is not None):
                    # Use same head_idx_kv as V
                    if const_expr(mPageTable is None):
                        if const_expr(not seqlen.has_cu_seqlens_k):
                            mSFV_cur = mSFV[None, None, head_idx_kv, batch_idx]
                        else:
                            # Use offset_sf_k for 128-aligned scale factor offsets
                            mSFV_cur = cute.domain_offset((0, seqlen.offset_sf_k), mSFV[None, None, head_idx_kv])
                    else:
                        # Paged attention - need to keep batch coord None
                        mSFV_cur = mSFV[None, None, head_idx_kv, None]

                    gSFV = cute.local_tile(mSFV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None)) if const_expr(mPageTable is None) else cute.local_tile(mSFV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None, None))
                    tOgSFV = cta_mma_pv.partition_B(gSFV)
                    tSFVsSFV, tSFVgSFV = cpasync.tma_partition(
                        tma_atom_SFV,
                        0,  # no multicast
                        cute.make_layout(1),
                        cute.group_modes(sSFV, 0, 3),
                        cute.group_modes(tOgSFV, 0, 3),
                    )
                    # Filter zeros like QUACK GEMM does for scale factors
                    tSFVsSFV = cute.filter_zeros(tSFVsSFV)
                    tSFVgSFV = cute.filter_zeros(tSFVgSFV)

            load_Q = partial(
                self.load_Q, tma_atom_Q, tQgQ, tQsQ,
                mbar_ptr + self.mbar_load_q_full_offset, mbar_ptr + self.mbar_load_q_empty_offset,
                phase=q_producer_phase,
                 # Scale factor parameters (for blockscaled MXFP8)
                tma_atom_SFQ=tma_atom_SFQ if const_expr(self.blockscaled) else None,
                tSFQgSFQ=tSFQgSFQ,
                tSFQsSFQ=tSFQsSFQ,
            )
            # We have to use mbarrier directly in the load for KV instead of replying on
            # pipeline_kv, because we could have different number of TMA bytes for K and V
            load_K = partial(
                self.load_KV, tma_atom_K, tKgK, tKsK,
                mbar_ptr + self.mbar_load_kv_full_offset, mbar_ptr + self.mbar_load_kv_empty_offset,
                K_or_V="K",
                # Scale factor parameters for SFK (only when blockscaled)
                tma_atom_SFK=tma_atom_SFK if const_expr(self.blockscaled) else None,
                tSFKgSFK=tSFKgSFK,
                tSFKsSFK=tSFKsSFK,
            )
            load_V = partial(
                self.load_KV, tma_atom_V, tVgV, tVsV,
                mbar_ptr + self.mbar_load_kv_full_offset, mbar_ptr + self.mbar_load_kv_empty_offset,
                K_or_V="V",
                # Scale factor parameters for SFV (only when blockscaled)
                tma_atom_SFV=tma_atom_SFV if const_expr(self.blockscaled) else None,
                tSFVgSFV=tSFVgSFV,
                tSFVsSFV=tSFVsSFV,
            )

            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
            load_Q(block=self.q_stage * m_block + 0, stage=0)  # Q0
            page_idx = mPageTable[batch_idx, n_block_max - 1] if const_expr(mPageTable is not None) else None
            load_K(block=n_block_max - 1, producer_state=kv_producer_state, page_idx=page_idx)  # K0
            kv_producer_state.advance()
            if const_expr(self.q_stage == 2):
                load_Q(block=self.q_stage * m_block + 1, stage=1)  # Q1
            q_producer_phase ^= 1
            load_V(block=n_block_max - 1, producer_state=kv_producer_state, page_idx=page_idx)  # V0
            kv_producer_state.advance()
            for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                n_block = n_block_max - 2 - i
                page_idx = mPageTable[batch_idx, n_block] if const_expr(mPageTable is not None) else None
                load_K(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)  # Ki
                kv_producer_state.advance()
                load_V(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)  # Vi
                kv_producer_state.advance()
            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
            # End of persistent scheduler loop

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_pv_blockscaled: Optional[cute.TiledMma],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sSFQ: Optional[cute.Tensor],
        sSFK: Optional[cute.Tensor],
        sSFV: Optional[cute.Tensor],
        sSFP: Optional[cute.Tensor],
        sQ_swizzle: cute.Swizzle,
        sK_swizzle: cute.Swizzle,
        sV_swizzle: cute.Swizzle,
        tStSs: Tuple[cute.Tensor, cute.Tensor],
        tOtOs: tuple[cute.Tensor],
        tOrPs: Tuple[cute.Tensor, cute.Tensor],
        tOtOs_blockscaled: Optional[tuple[cute.Tensor]],
        tOrPs_blockscaled: Optional[Tuple[cute.Tensor, cute.Tensor]],
        pipeline_kv: cutlass.pipeline.PipelineAsync,
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        cta_mma_qk = tiled_mma_qk.get_slice(0)  # default 1SM
        cta_mma_pv = tiled_mma_pv.get_slice(0)  # default 1SM
        tSrQ = cta_mma_qk.make_fragment_A(sQ)
        tSrK = cta_mma_qk.make_fragment_B(sK)
        tOrV = cta_mma_pv.make_fragment_B(sV)
        if const_expr(self.q_stage == 2):
            tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 1])
        else:
            tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 0])

        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op

        # Setup blockscaled GEMM components (TMEM layouts and S2T copies)
        # For N>128 (multiple tiles), SF must be in S region, NOT O region.
        # Original placement at 256 conflicts with O0 writes.
        # Following C++ pattern: place SF after P within S regions, with barrier sync.
        # - SF0 for S0 GEMM: at P1 + P_size (in S1 region, ~200)
        # - SF1 for S1 GEMM: at P0 + P_size (in S0 region, ~72)
        # Main loop SF tensors (at P+P_size, overlap with opposite S region)
        tCtSFQs, tCtSFKs = [None, None], [None, None]
        tiled_copy_s2t_sfq, tCsSFQ_compact_s2t, tCtSFQ_compact_s2ts = None, None, [None, None]
        tiled_copy_s2t_sfk, tCsSFK_compact_s2t, tCtSFK_compact_s2ts = None, None, [None, None]

        if const_expr(self.blockscaled):
            # Create base TMEM pointer for scale factor offset calculations
            tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)

            # Compute P size (columns used by P accumulators)
            # P uses tilePlikeFP32 columns: mma_tiler_qk[1] // 32 * v_dtype.width
            tilePlikeFP32 = self.mma_tiler_qk[1] // 32 * self.v_dtype.width

            # SF offsets: place SF after P within S regions
            # - SF0 at P1 + tilePlikeFP32 (in S1 region): for S0 GEMM
            # - SF1 at P0 + tilePlikeFP32 (in S0 region): for S1 GEMM
            # This creates overlap: SF0 overlaps S1, SF1 overlaps S0
            # Barriers synchronize: wait for softmax to finish S before writing SF
            sf_offsets = [
                self.tmem_p_offset[1] + tilePlikeFP32 * 2,  # e.g., 192 + 32 = 224 (in S1 region)
                self.tmem_p_offset[0] + tilePlikeFP32 * 2,  # e.g., 64 + 32 = 96 (in S0 region)
            ]

            # Create TMEM layouts (shared between both stages)
            sSFQ_layout_per_stage = cute.slice_(sSFQ.layout, (None, None, None, 0))
            tCtSFQ_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_qk,
                self.mma_tiler_qk,
                self.sf_vec_size,
                sSFQ_layout_per_stage,
            )
            sSFK_layout_per_stage = cute.slice_(sSFK.layout, (None, None, None, 0))
            tCtSFK_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_qk,
                self.mma_tiler_qk,
                self.sf_vec_size,
                sSFK_layout_per_stage,
            )

            # Get SFK offset relative to SFQ
            temp_sfq_ptr = cute.recast_ptr(tmem_ptr, dtype=self.sf_dtype)
            temp_tCtSFQ = cute.make_tensor(temp_sfq_ptr, tCtSFQ_layout)
            sfk_relative_offset = tcgen05.find_tmem_tensor_col_offset(temp_tCtSFQ)

            # Create stage-specific SF TMEM tensors
            # Stage 0: SF0 at 224 (in S1 region)
            sfq_tmem_ptr_0 = cute.recast_ptr(tmem_ptr + sf_offsets[0], dtype=self.sf_dtype)
            tCtSFQs[0] = cute.make_tensor(sfq_tmem_ptr_0, tCtSFQ_layout)
            sfk_tmem_ptr_0 = cute.recast_ptr(tmem_ptr + sf_offsets[0] + sfk_relative_offset, dtype=self.sf_dtype)
            tCtSFKs[0] = cute.make_tensor(sfk_tmem_ptr_0, tCtSFK_layout)

            # Stage 1: SF1 at 96 (in S0 region)
            sfq_tmem_ptr_1 = cute.recast_ptr(tmem_ptr + sf_offsets[1], dtype=self.sf_dtype)
            tCtSFQs[1] = cute.make_tensor(sfq_tmem_ptr_1, tCtSFQ_layout)
            sfk_tmem_ptr_1 = cute.recast_ptr(tmem_ptr + sf_offsets[1] + sfk_relative_offset, dtype=self.sf_dtype)
            tCtSFKs[1] = cute.make_tensor(sfk_tmem_ptr_1, tCtSFK_layout)

            # Create S2T copy partitions (shared tiled_copy, stage-specific TMEM destinations)
            tiled_copy_s2t_sfq, tCsSFQ_compact_s2t, tCtSFQ_compact_s2ts[0] = sm100_utils.make_s2t_copy_partitions(
                sSFQ, tCtSFQs[0], self.sf_dtype
            )
            _, _, tCtSFQ_compact_s2ts[1] = sm100_utils.make_s2t_copy_partitions(
                sSFQ, tCtSFQs[1], self.sf_dtype
            )
            tiled_copy_s2t_sfk, tCsSFK_compact_s2t, tCtSFK_compact_s2ts[0] = sm100_utils.make_s2t_copy_partitions(
                sSFK, tCtSFKs[0], self.sf_dtype
            )
            _, _, tCtSFK_compact_s2ts[1] = sm100_utils.make_s2t_copy_partitions(
                sSFK, tCtSFKs[1], self.sf_dtype
            )

            # Prologue SF TMEM tensors (use O region, safe for first tile before O is used)
            # For the first tile, we avoid the S region to prevent race conditions
            sf_prologue_offsets = self.tmem_layout.sf_prologue_offsets
            tCtSFQs_prologue = [None, None]
            tCtSFKs_prologue = [None, None]
            tCtSFQ_compact_s2ts_prologue = [None, None]
            tCtSFK_compact_s2ts_prologue = [None, None]

            # Stage 0 prologue tensors
            sfq_prologue_ptr_0 = cute.recast_ptr(tmem_ptr + sf_prologue_offsets[0][0], dtype=self.sf_dtype)
            tCtSFQs_prologue[0] = cute.make_tensor(sfq_prologue_ptr_0, tCtSFQ_layout)
            sfk_prologue_ptr_0 = cute.recast_ptr(tmem_ptr + sf_prologue_offsets[0][1], dtype=self.sf_dtype)
            tCtSFKs_prologue[0] = cute.make_tensor(sfk_prologue_ptr_0, tCtSFK_layout)
            _, _, tCtSFQ_compact_s2ts_prologue[0] = sm100_utils.make_s2t_copy_partitions(
                sSFQ, tCtSFQs_prologue[0], self.sf_dtype
            )
            _, _, tCtSFK_compact_s2ts_prologue[0] = sm100_utils.make_s2t_copy_partitions(
                sSFK, tCtSFKs_prologue[0], self.sf_dtype
            )

            # Stage 1 prologue tensors
            sfq_prologue_ptr_1 = cute.recast_ptr(tmem_ptr + sf_prologue_offsets[1][0], dtype=self.sf_dtype)
            tCtSFQs_prologue[1] = cute.make_tensor(sfq_prologue_ptr_1, tCtSFQ_layout)
            sfk_prologue_ptr_1 = cute.recast_ptr(tmem_ptr + sf_prologue_offsets[1][1], dtype=self.sf_dtype)
            tCtSFKs_prologue[1] = cute.make_tensor(sfk_prologue_ptr_1, tCtSFK_layout)
            _, _, tCtSFQ_compact_s2ts_prologue[1] = sm100_utils.make_s2t_copy_partitions(
                sSFQ, tCtSFQs_prologue[1], self.sf_dtype
            )
            _, _, tCtSFK_compact_s2ts_prologue[1] = sm100_utils.make_s2t_copy_partitions(
                sSFK, tCtSFKs_prologue[1], self.sf_dtype
            )

            # =========================================================================
            # PV Block-Scaled: TMEM tensors and S2T copies for SFP and SFV
            # Following C++ pattern: SFP at P + kSizeP, SFV at SFP + kSizeSF
            # =========================================================================
            # Create TMEM layouts for PV scale factors
            # Use tiled_mma_pv_blockscaled for proper scale factor layouts
            sSFP_layout_per_stage = cute.slice_(sSFP.layout, (None, None, None, 0))
            tCtSFP_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_pv_blockscaled,
                self.mma_tiler_pv,
                self.sf_vec_size,
                sSFP_layout_per_stage,
            )
            sSFV_layout_per_stage = cute.slice_(sSFV.layout, (None, None, None, 0))
            tCtSFV_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_pv_blockscaled,
                self.mma_tiler_pv,
                self.sf_vec_size,
                sSFV_layout_per_stage,
            )

            # Get SFV offset relative to SFP
            temp_sfp_ptr = cute.recast_ptr(tmem_ptr, dtype=self.sf_dtype)
            temp_tCtSFP = cute.make_tensor(temp_sfp_ptr, tCtSFP_layout)
            sfv_relative_offset = tcgen05.find_tmem_tensor_col_offset(temp_tCtSFP)

            # PV SF offsets: place SF after P for each stage
            # SFP0 = P0 + tilePlikeFP32, SFP1 = P1 + tilePlikeFP32
            sfp_offsets = [
                self.tmem_p_offset[0] + tilePlikeFP32,  # e.g., 64 + 32 = 96
                self.tmem_p_offset[1] + tilePlikeFP32,  # e.g., 192 + 32 = 224
            ]

            # Create stage-specific SF TMEM tensors for PV
            tCtSFPs = [None, None]
            tCtSFVs = [None, None]

            # Stage 0: SFP0 at P0 + tilePlikeFP32, SFV0 after SFP0
            sfp_tmem_ptr_0 = cute.recast_ptr(tmem_ptr + sfp_offsets[0], dtype=self.sf_dtype)
            tCtSFPs[0] = cute.make_tensor(sfp_tmem_ptr_0, tCtSFP_layout)
            sfv_tmem_ptr_0 = cute.recast_ptr(tmem_ptr + sfp_offsets[0] + sfv_relative_offset, dtype=self.sf_dtype)
            tCtSFVs[0] = cute.make_tensor(sfv_tmem_ptr_0, tCtSFV_layout)

            # Stage 1: SFP1 at P1 + tilePlikeFP32, SFV1 after SFP1
            sfp_tmem_ptr_1 = cute.recast_ptr(tmem_ptr + sfp_offsets[1], dtype=self.sf_dtype)
            tCtSFPs[1] = cute.make_tensor(sfp_tmem_ptr_1, tCtSFP_layout)
            sfv_tmem_ptr_1 = cute.recast_ptr(tmem_ptr + sfp_offsets[1] + sfv_relative_offset, dtype=self.sf_dtype)
            tCtSFVs[1] = cute.make_tensor(sfv_tmem_ptr_1, tCtSFV_layout)

            # Create S2T copy partitions for SFP and SFV
            tiled_copy_s2t_sfp, tCsSFP_compact_s2t, tCtSFP_compact_s2ts = None, None, [None, None]
            tiled_copy_s2t_sfv, tCsSFV_compact_s2t, tCtSFV_compact_s2ts = None, None, [None, None]

            tiled_copy_s2t_sfp, tCsSFP_compact_s2t, tCtSFP_compact_s2ts[0] = sm100_utils.make_s2t_copy_partitions(
                sSFP, tCtSFPs[0], self.sf_dtype
            )
            _, _, tCtSFP_compact_s2ts[1] = sm100_utils.make_s2t_copy_partitions(
                sSFP, tCtSFPs[1], self.sf_dtype
            )
            tiled_copy_s2t_sfv, tCsSFV_compact_s2t, tCtSFV_compact_s2ts[0] = sm100_utils.make_s2t_copy_partitions(
                sSFV, tCtSFVs[0], self.sf_dtype
            )
            _, _, tCtSFV_compact_s2ts[1] = sm100_utils.make_s2t_copy_partitions(
                sSFV, tCtSFVs[1], self.sf_dtype
            )

        # Create GEMM functions - use blockscaled version when enabled
        if const_expr(self.blockscaled):
            # For blockscaled GEMM, we use the CUTE API approach which properly handles
            # scale factor TMEM addresses per K iteration (unlike the raw PTX approach)
            # We pass the tiled_mma and TMEM scale factor tensors directly
            #
            # NOTE: The gemm_blockscaled function will:
            # 1. Loop over K phases
            # 2. For each K phase, set tiled_mma.set(tcgen05.Field.SFA, tCtSFA[kphase_idx].iterator)
            # 3. Call cute.gemm(tiled_mma, acc, ...)
            # This ensures the scale factor addresses are properly incremented per K iteration
            #
            # Function signature: gemm_blockscaled(tiled_mma, acc, tCrA, tCrB, tCtSFA, tCtSFB, zero_init)
            # tCrB is passed dynamically, so we use a lambda to adapt the interface
            # Each stage uses its own SF tensors (SF0 in S1 region for S0 GEMM, SF1 in S0 region for S1 GEMM)
            gemm_Si = [
                lambda tCrB, sB, stage=stage: sm100_utils.gemm_blockscaled(
                    tiled_mma_qk, tStSs[stage], tSrQs[stage], tCrB,
                    tCtSFQs[stage], tCtSFKs[stage],
                    zero_init=True
                )
                for stage in range(2)
            ]
        else:
            gemm_Si = [
                partial(
                    sm100_utils.gemm_ptx_partial,
                    qk_mma_op, self.tmem_s_offset[stage], tSrQs[stage], sA=sQ[None, None, None, stage],
                    zero_init=True
                )
                for stage in range(2)
            ]
            # Initialize these to None when not blockscaled
            tCtSFPs = [None, None]
            tCtSFVs = [None, None]
            tiled_copy_s2t_sfp, tCsSFP_compact_s2t, tCtSFP_compact_s2ts = None, None, [None, None]
            tiled_copy_s2t_sfv, tCsSFV_compact_s2t, tCtSFV_compact_s2ts = None, None, [None, None]

        # Create GEMM functions for PV
        if const_expr(self.blockscaled):
            # For blockscaled PV GEMM, we use gemm_blockscaled with SFP and SFV scale factors
            # Use blockscaled partitions and tiled_mma_pv_blockscaled which supports SFA/SFB fields
            gemm_Pi = [
                lambda tCrB, sB, zero_init, stage=stage, **kwargs: sm100_utils.gemm_blockscaled(
                    tiled_mma_pv_blockscaled, tOtOs_blockscaled[stage], tOrPs_blockscaled[stage], tCrB,
                    tCtSFPs[stage], tCtSFVs[stage],
                    zero_init=zero_init
                )
                for stage in range(2)
            ]
        else:
            gemm_Pi = [
                partial(
                    sm100_utils.gemm_ptx_partial,
                    pv_mma_op, self.tmem_o_offset[stage if self.q_stage == 2 else 0], tOrPs[stage],
                    sA=None
                )
                for stage in range(2)
            ]

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        P_full_1_phase = Int32(0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        is_first_tile = True
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
            if is_first_tile or const_expr(not self.unroll_kv):
                for stage in cutlass.range_constexpr(self.q_stage):
                    # GEMM_QK00 (Q0 * K0 -> S0) or GEMM_QK01 (Q1 * K0 -> S1)
                    # 1. wait for Q0 / Q1
                    cute.arch.mbarrier_wait(mbar_ptr + self.mbar_load_q_full_offset + stage, mma_q_consumer_phase)
                    # 2. wait for K0
                    if const_expr(stage == 0):
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    tSrKi = tSrK[None, None, None, mma_kv_consumer_state.index]
                    # We don't need to acquire empty S0 / S1.
                    # For the first iteration, we don't need to wait as we're guaranteed S0 / S1
                    # are empty. For subsequent iterations, the wait happened at the end
                    # of the while loop.
                    # 3. gemm
                    sK_cur = sK[None, None, None, mma_kv_consumer_state.index]
                    if const_expr(self.uneven_kv_smem):
                        sK_cur = self.offset_kv_smem(sK_cur, mma_kv_consumer_state.index, mma_kv_consumer_state.phase)
                    # S2T copy for scale factors (SMEM -> TMEM) before blockscaled GEMM
                    if const_expr(self.blockscaled):
                        # Prologue (first tile): copy SF to O region (safe, O not used yet)
                        # This avoids race condition where SF overlapping S region could
                        # overwrite GEMM results before they're consumed by softmax
                        s2t_sfq_stage_coord = (None, None, None, None, stage)
                        cute.copy(
                            tiled_copy_s2t_sfq,
                            tCsSFQ_compact_s2t[s2t_sfq_stage_coord],
                            tCtSFQ_compact_s2ts_prologue[stage],
                        )
                        s2t_sfk_stage_coord = (None, None, None, None, mma_kv_consumer_state.index)
                        cute.copy(
                            tiled_copy_s2t_sfk,
                            tCsSFK_compact_s2t[s2t_sfk_stage_coord],
                            tCtSFK_compact_s2ts_prologue[stage],
                        )
                        cute.arch.fence_view_async_tmem_store()
                        # Call gemm_blockscaled directly with prologue SF tensors
                        sm100_utils.gemm_blockscaled(
                            tiled_mma_qk, tStSs[stage], tSrQs[stage], tSrKi,
                            tCtSFQs_prologue[stage], tCtSFKs_prologue[stage],
                            zero_init=True
                        )
                    else:
                        gemm_Si[stage](tCrB=tSrKi, sB=sK_cur)
                    # 4. release S0 / S1
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + stage)
                mma_q_consumer_phase ^= 1
                # 5. release K0
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                is_first_tile = False
            # End of GEMM (Q1 * K0 -> S1)
            # Note: Q0 & Q1 are still needed in the seqlen_kv loop
            # so we need to release them after the seqlen_kv loop
            # O hasn't been accumulated yet, its first MMA calculation doesn't need to accumulate
            O_should_accumulate = False
            for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                # 1. wait for V0
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                mma_kv_release_state = mma_kv_consumer_state.clone()
                Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                tOrVi = tOrV[None, None, None, Vi_index]
                for stage in cutlass.range_constexpr(2):
                    # 2. acquire corrected O0/O1_partial and P0 / P1
                    # For the first iteration in this work tile, waiting for O0/O1_partial
                    # means that the correction warps has finished reading tO during
                    # the last iteration of the previous work tile has finished.
                    cute.arch.mbarrier_wait(mbar_ptr + self.mbar_P_full_1_offset + stage, P_full_1_phase)
                    # 3. gemm
                    # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                    # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)
                    sV_cur = sV[None, None, None, Vi_index]
                    if const_expr(self.uneven_kv_smem):
                        sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                    # S2T copy for SFP and SFV before blockscaled PV GEMM
                    if const_expr(self.blockscaled):
                        s2t_sfp_stage_coord = (None, None, None, None, stage)
                        cute.copy(tiled_copy_s2t_sfp, tCsSFP_compact_s2t[s2t_sfp_stage_coord], tCtSFP_compact_s2ts[stage])
                        s2t_sfv_stage_coord = (None, None, None, None, Vi_index)
                        cute.copy(tiled_copy_s2t_sfv, tCsSFV_compact_s2t[s2t_sfv_stage_coord], tCtSFV_compact_s2ts[stage])
                        cute.arch.fence_view_async_tmem_store()
                    gemm_Pi[stage](tCrB=tOrVi, sB=sV_cur, zero_init=not O_should_accumulate, mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + stage, mbar_phase= P_full_1_phase)
                    # 4. release accumulated O0_partial / O1_partial
                    # Don't need to signal O_full to the correction warps anymore since the
                    # correction warps wait for the softmax warps anyway. By the time the softmax
                    # warps finished, S_i for the next iteration must have been done, so O_i-1
                    # must have been done as well.
                    # with cute.arch.elect_one():
                    #     tcgen05.commit(mbar_ptr + self.mbar_O_full_offset + stage)
                    # 5. release V(i-1)
                    if const_expr(stage == 1):
                        pipeline_kv.consumer_release(mma_kv_release_state)
                        mma_kv_release_state.advance()
                    # End of GEMM_PV00 (P0 * V0 -> O0_partial)

                    # GEMM_QK0i (Q0 * Ki -> S0)
                    # 1. wait for Ki
                    if const_expr(stage == 0):
                        mma_kv_consumer_state.advance()
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    Ki_index, Ki_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                    # 2. gemm
                    # Don't need to wait for the softmax warp to have finished reading the previous
                    # Si, since this gemm is scheduled after the PV gemm, which guaranteed that Si
                    # has been read and Pi has been written.
                    # sm100_utils.gemm(tiled_mma_qk, tStS0, tSrQs[0], tSrK[None, None, None, Ki_index], zero_init=True)
                    sK_cur = sK[None, None, None, Ki_index]
                    if const_expr(self.uneven_kv_smem):
                        sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                    # S2T copy for SFQ and SFK before blockscaled GEMM
                    if const_expr(self.blockscaled):
                        # TMEM synchronization: scale factors for Si overlap with S(1-i) region
                        # Before S0 GEMM (using SF0 which overlaps S1): wait for softmax to finish reading S1
                        # Before S1 GEMM (using SF1 which overlaps S0): wait for softmax to finish reading S0
                        if stage == 0:
                            cute.arch.barrier(barrier_id=self.mbar_mma_p1_id, number_of_threads=self.mbar_mma_threads)
                        else:
                            cute.arch.barrier(barrier_id=self.mbar_mma_p0_id, number_of_threads=self.mbar_mma_threads)
                        # Copy SFQ from SMEM to TMEM (indexed by Q stage)
                        # Use stage-specific TMEM destination
                        s2t_sfq_stage_coord = (None, None, None, None, stage)
                        cute.copy(
                            tiled_copy_s2t_sfq,
                            tCsSFQ_compact_s2t[s2t_sfq_stage_coord],
                            tCtSFQ_compact_s2ts[stage],
                        )
                        # Copy SFK from SMEM to TMEM (indexed by KV pipeline stage)
                        s2t_sfk_stage_coord = (None, None, None, None, Ki_index)
                        cute.copy(
                            tiled_copy_s2t_sfk,
                            tCsSFK_compact_s2t[s2t_sfk_stage_coord],
                            tCtSFK_compact_s2ts[stage],
                        )
                        # Fence to ensure S2T copies complete before GEMM reads from TMEM
                        cute.arch.fence_view_async_tmem_store()
                    gemm_Si[stage](tCrB=tSrK[None, None, None, Ki_index], sB=sK_cur)
                    # 3. release S0
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + stage)
                    # End of GEMM_QK0i (Q0 * Ki -> S0)
                    if i == n_block_max - 1 - n_block_min - 1:
                        with cute.arch.elect_one():
                            tcgen05.commit(mbar_ptr + self.mbar_load_q_empty_offset + stage)
                # 4. release Ki
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                P_full_1_phase ^= 1
                O_should_accumulate = True
            # End of seqlen_kv loop

            # release Q0 & Q1 if for-loop above doesn't execute
            if not O_should_accumulate:
                with cute.arch.elect_one():
                    for stage in cutlass.range_constexpr(self.q_stage):
                        tcgen05.commit(mbar_ptr + self.mbar_load_q_empty_offset + stage)

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
            is_next_tile_valid = work_tile.is_valid_tile

            if not is_next_tile_valid or const_expr(not self.unroll_kv):
                # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                # 1. wait for V0
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                tOrVi = tOrV[None, None, None, Vi_index]
                for stage in cutlass.range_constexpr(2):
                    # 2. acquire corrected Oi_partial and Pi
                    cute.arch.mbarrier_wait(mbar_ptr + self.mbar_P_full_1_offset + stage, P_full_1_phase)
                    # 3. gemm
                    # sm100_utils.gemm(tiled_mma_pv, tOtO0, tOrP0, tOrVi, zero_init=True)
                    # gemm_Pi[stage](tCrB=tOrVi, sB=sV[None, None, None, Vi_index], zero_init=not O_should_accumulate)
                    sV_cur = sV[None, None, None, Vi_index]
                    if const_expr(self.uneven_kv_smem):
                        sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                    # S2T copy for SFP and SFV before blockscaled PV GEMM
                    if const_expr(self.blockscaled):
                        s2t_sfp_stage_coord = (None, None, None, None, stage)
                        cute.copy(tiled_copy_s2t_sfp, tCsSFP_compact_s2t[s2t_sfp_stage_coord], tCtSFP_compact_s2ts[stage])
                        s2t_sfv_stage_coord = (None, None, None, None, Vi_index)
                        cute.copy(tiled_copy_s2t_sfv, tCsSFV_compact_s2t[s2t_sfv_stage_coord], tCtSFV_compact_s2ts[stage])
                        cute.arch.fence_view_async_tmem_store()
                    gemm_Pi[stage](tCrB=tOrVi, sB=sV_cur, zero_init=not O_should_accumulate, mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + stage, mbar_phase=P_full_1_phase)
                    # 4. release accumulated O0_partial
                    # We do need O_full here since for the last tile, by the time the softmax warp
                    # has signaled to the correction warp, the softmax warp has just finished compute
                    # the row sum of the current tile. It does not guarantee that the 1st tile
                    # of the next work tile has been computed yet.
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_O_full_offset + stage)
                    # End of GEMM_PV00 (P0 * V0 -> O0_partial)
                P_full_1_phase ^= 1
                # 5. release Vi_end
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                # End of GEMM_PV1(i_end) (P1 * Vi_end -> O1)
            else:
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                mma_kv_release_state = mma_kv_consumer_state.clone()
                Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                tOrVi = tOrV[None, None, None, Vi_index]
                for stage in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_wait(mbar_ptr + self.mbar_P_full_1_offset + stage, P_full_1_phase)
                    sV_cur = sV[None, None, None, Vi_index]
                    if const_expr(self.uneven_kv_smem):
                        sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                    # S2T copy for SFP and SFV before blockscaled PV GEMM
                    if const_expr(self.blockscaled):
                        s2t_sfp_stage_coord = (None, None, None, None, stage)
                        cute.copy(tiled_copy_s2t_sfp, tCsSFP_compact_s2t[s2t_sfp_stage_coord], tCtSFP_compact_s2ts[stage])
                        s2t_sfv_stage_coord = (None, None, None, None, Vi_index)
                        cute.copy(tiled_copy_s2t_sfv, tCsSFV_compact_s2t[s2t_sfv_stage_coord], tCtSFV_compact_s2ts[stage])
                        cute.arch.fence_view_async_tmem_store()
                    gemm_Pi[stage](tCrB=tOrVi, sB=sV_cur, zero_init=not O_should_accumulate, mbar_ptr=mbar_ptr + self.mbar_P_full_2_offset + stage, mbar_phase= P_full_1_phase)
                    # 4. release accumulated O0_partial / O1_partial
                    # Don't need to signal O_full to the correction warps anymore since the
                    # correction warps wait for the softmax warps anyway. By the time the softmax
                    # warps finished, S_i for the next iteration must have been done, so O_i-1
                    # must have been done as well.
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_O_full_offset + stage)
                    # End of GEMM_PV00 (P0 * V0 -> O0_partial)

                    if const_expr(stage == 1):
                        pipeline_kv.consumer_release(mma_kv_release_state)
                        mma_kv_release_state.advance()
                    # End of GEMM_PV00 (P0 * V0 -> O0_partial)

                    # GEMM_QK0i (Q0 * Ki -> S0)
                    # GEMM_QK00 (Q0 * K0 -> S0) or GEMM_QK01 (Q1 * K0 -> S1)
                    # 1. wait for Q0 / Q1
                    cute.arch.mbarrier_wait(mbar_ptr + self.mbar_load_q_full_offset + stage, mma_q_consumer_phase)
                    # 2. wait for K0
                    if const_expr(stage == 0):
                        mma_kv_consumer_state.advance()
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    tSrKi = tSrK[None, None, None, mma_kv_consumer_state.index]
                    # We don't need to acquire empty S0 / S1.
                    # For the first iteration, we don't need to wait as we're guaranteed S0 / S1
                    # are empty. For subsequent iterations, the wait happened at the end
                    # of the while loop.
                    # 3. gemm
                    # sm100_utils.gemm(tiled_mma_qk, tStSs[stage], tSrQs[stage], tSrKi, zero_init=True)
                    sK_cur = sK[None, None, None, mma_kv_consumer_state.index]
                    if const_expr(self.uneven_kv_smem):
                        sK_cur = self.offset_kv_smem(sK_cur, mma_kv_consumer_state.index, mma_kv_consumer_state.phase)
                    # S2T copy for SFQ and SFK before blockscaled GEMM (new K block from next tile)
                    # BUG FIX: Must copy SFQ too! Each stage needs its own Q scale factors.
                    if const_expr(self.blockscaled):
                        # Add barrier wait like in main N loop (lines 1562-1565)
                        if stage == 0:
                            cute.arch.barrier(barrier_id=self.mbar_mma_p1_id, number_of_threads=self.mbar_mma_threads)
                        else:
                            cute.arch.barrier(barrier_id=self.mbar_mma_p0_id, number_of_threads=self.mbar_mma_threads)

                        # Copy SFQ from SMEM to TMEM (indexed by Q stage)
                        # Use stage-specific TMEM destination
                        s2t_sfq_stage_coord = (None, None, None, None, stage)
                        cute.copy(
                            tiled_copy_s2t_sfq,
                            tCsSFQ_compact_s2t[s2t_sfq_stage_coord],
                            tCtSFQ_compact_s2ts[stage],
                        )
                        # Copy SFK from SMEM to TMEM (indexed by KV pipeline stage)
                        s2t_sfk_stage_coord = (None, None, None, None, mma_kv_consumer_state.index)
                        cute.copy(
                            tiled_copy_s2t_sfk,
                            tCsSFK_compact_s2t[s2t_sfk_stage_coord],
                            tCtSFK_compact_s2ts[stage],
                        )
                        cute.arch.fence_view_async_tmem_store()
                        # Use blockscaled GEMM with proper SF tensors
                        sm100_utils.gemm_blockscaled(
                            tiled_mma_qk, tStSs[stage], tSrQs[stage], tSrKi,
                            tCtSFQs[stage], tCtSFKs[stage],
                            zero_init=True
                        )
                    else:
                        gemm_Si[stage](tCrB=tSrKi, sB=sK_cur)
                    # 4. release S0 / S1
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_ptr + self.mbar_S_full_offset + stage)
                mma_q_consumer_phase ^= 1
                P_full_1_phase ^= 1
                # 5. release K0
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
        # End of persistent scheduler loop

    # for both softmax0 and softmax1 warp group
    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        cta_mma_qk: cute.core.ThrMma,
        cta_mma_pv: cute.core.ThrMma,
        tStSi: cute.Tensor,
        tOtOi: tuple[cute.Tensor],
        sScale: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sO: cute.Tensor,
        learnable_sink: Optional[cute.Tensor],
        mbar_ptr: cute.Pointer,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        sSFP: Optional[cute.Tensor] = None,  # P's scale factors for blockscaled MXFP8
    ):
        """Compute softmax on attention scores from QK matrix multiplication.

        This method handles the softmax computation for either the first or second half of the
        attention matrix, depending on the 'stage' parameter. It calculates row-wise maximum
        and sum values needed for stable softmax computation, applies optional masking, and
        transforms raw attention scores into probability distributions.

        The implementation uses specialized memory access patterns and efficient math operations
        for computing exp(x) using exp2 functions. It also coordinates pipeline
        synchronization between MMA, correction, and sequence processing stages.
        """
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE
            # * (len(self.softmax0_warp_ids) if stage == 0 else len(self.softmax1_warp_ids)
            * (len(self.softmax0_warp_ids)
            )
        )

        tilePlikeFP32 = self.mma_tiler_qk[1] // 32 * self.v_dtype.width
        tStP_layout = cute.composition(tStSi.layout, cute.make_layout((self.m_block_size, tilePlikeFP32)))
        tStP = cute.make_tensor(tStSi.iterator + self.tmem_s_to_p_offset, tStP_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32,
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStSi).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tStSi)

        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32,
        )
        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP)
        thr_tmem_store = tiled_tmem_store.get_slice(tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)

        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)

        mma_si_consumer_phase = Int32(0)
        s0_s1_sequence_phase = Int32(1 if stage == 0 else 0)

        # self.warp_scheduler_barrier_init()

        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        mbar_s0_s1_sequence_offset = self.mbar_s0_s1_sequence_offset + warp_idx_in_wg

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        is_first = True
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
            mask = AttentionMaskCls(seqlen.seqlen_q, seqlen.seqlen_k)
            mask_fn = partial(
                mask.apply_mask_sm100, m_block=m_block * 2 + stage, thr_mma=cta_mma_qk, thr_tmem_load=thr_tmem_load, mask_causal=self.is_causal, mask_local=self.is_local
            )
            gelu = Gelu(softmax_scale_log2)
            relu = Relu(softmax_scale_log2)

            softmax_step = partial(
                self.softmax_step,
                gelu=gelu,
                relu=relu,
                mbar_ptr=mbar_ptr,
                mbar_s0_s1_sequence_offset=mbar_s0_s1_sequence_offset,
                cta_mma_qk=cta_mma_qk,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                tStS_t2r=tStS_t2r,
                tStP_r2t=tStP_r2t,
                stage=stage,
                sSFP=sSFP,
            )


            # 1 masking iter
            mma_si_consumer_phase, s0_s1_sequence_phase = softmax_step(mma_si_consumer_phase, s0_s1_sequence_phase, n_block_max - 1, is_first=is_first, mask_fn=partial(mask_fn, mask_seqlen=True))
            is_first = False
            n_block_max -= 1
            # Next couple of iterations with causal masking
            if const_expr(self.is_causal or self.is_local):
                n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                    seqlen, m_block, n_block_min
                )
                for n_tile in cutlass.range(n_block_max - n_block_min_causal_local_mask, unroll=1):
                    n_block = n_block_max - 1 - n_tile
                    mma_si_consumer_phase, s0_s1_sequence_phase = softmax_step(mma_si_consumer_phase, s0_s1_sequence_phase, n_block, mask_fn=partial(mask_fn, mask_seqlen=False))
                n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
            # The remaining iterations have no masking
            n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
                seqlen, m_block, n_block_min
            )
            for n_tile in cutlass.range(n_block_max - n_block_min_before_local_mask, unroll=1):
                n_block = n_block_max - n_tile - 1
                mma_si_consumer_phase, s0_s1_sequence_phase = softmax_step(mma_si_consumer_phase, s0_s1_sequence_phase, n_block)
            # Separate iterations with local masking on the left
            if const_expr(self.is_local and block_info.window_size_left is not None):
                n_block_max = cutlass.min(n_block_max, n_block_min_before_local_mask)
                for n_tile in cutlass.range(0, n_block_max - n_block_min, unroll=1):
                    n_block = n_block_max - 1 - n_tile
                    mma_si_consumer_phase, s0_s1_sequence_phase = softmax_step(mma_si_consumer_phase, s0_s1_sequence_phase, n_block, mask_fn=partial(mask_fn, mask_seqlen=False))
                    # Now that we no longer already have the 1st iteration, need mask_seqlen=True here

            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_O_full_offset + stage, o_corr_consumer_phase)
            cute.arch.mbarrier_wait(mbar_ptr + self.mbar_corr_epi_empty_offset + stage, corr_epi_producer_phase)
            self.correction_epilogue(
                cta_mma_pv, tOtOi, tidx, sO[None, None, stage],
            )
            cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_corr_epi_full_offset + stage)

            o_corr_consumer_phase ^= 1
            corr_epi_producer_phase ^= 1

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        gelu: Gelu,
        relu: Relu,
        mbar_ptr: cute.Pointer,
        mbar_s0_s1_sequence_offset: Int32,
        cta_mma_qk: cute.core.ThrMma,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStP_r2t: cute.Tensor,
        stage: int | Int32,
        sSFP: Optional[cute.Tensor] = None,  # P's scale factors for blockscaled MXFP8
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        """Perform a single step of the softmax computation on a block of attention scores.

        This method processes one block of the attention matrix, computing numerically stable
        softmax by first finding the row maximum, subtracting it from all elements, applying
        exponential function, and then normalizing by the sum of exponentials. It also handles
        optional masking of attention scores.

        The method involves several key operations:
        1. Loading attention scores from tensor memory
        2. Applying optional masking based on position
        3. Computing row-wise maximum values for numerical stability
        4. Transforming scores using exp2(x*scale - max*scale)
        5. Computing row sums for normalization
        6. Coordinating pipeline synchronization between different processing stages
        """
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScS = cta_mma_qk.partition_C(cute.make_identity_tensor((self.mma_tiler_qk[0], self.mma_tiler_qk[1])))

        tScP_layout = cute.composition(tScS.layout, cute.make_layout((self.m_block_size, tilePlikeFP32)))
        tScP = cute.make_tensor(tScS.iterator, tScP_layout)
        tScS_t2r_shape = thr_tmem_load.partition_D(tScS).shape

        # Wait for Si
        cute.arch.mbarrier_wait(mbar_ptr + self.mbar_S_full_offset + stage, mma_si_consumer_phase)
        tSrS_t2r = cute.make_fragment(tScS_t2r_shape, self.qk_acc_dtype)
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)

        # Signal MXFwdNamedBarrier after reading S from TMEM
        # This tells MMA warp it's safe to overwrite scale factor region that overlaps with this S
        # - Stage 0 (S0): signal MMA_P0 (allows MMA to write SFQ1/SFK1 which overlap S0)
        # - Stage 1 (S1): signal MMA_P1 (allows MMA to write SFQ0/SFK0 which overlap S1)
        # Per C++ reference:
        # - Stage 0: Skip signaling on first_call (MMA hasn't entered N loop yet)
        # - Stage 1: Always signal (skip only on final_call, but we always signal for simplicity)
        if const_expr(self.blockscaled):
            if stage == 0:
                if not is_first:
                    cute.arch.fence_view_async_tmem_load()
                    cute.arch.barrier_arrive(barrier_id=self.mbar_mma_p0_id, number_of_threads=self.mbar_mma_threads)
            else:
                # Stage 1: Always signal MMA_P1 - MMA waits on this before writing SF for S0
                cute.arch.fence_view_async_tmem_load()
                cute.arch.barrier_arrive(barrier_id=self.mbar_mma_p1_id, number_of_threads=self.mbar_mma_threads)

        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block)

        if const_expr(self.s0_s1_barrier):
            cute.arch.mbarrier_wait(mbar_ptr + mbar_s0_s1_sequence_offset + stage * 4, s0_s1_sequence_phase)
        tSrP_r2t_f32 = cute.make_fragment(thr_tmem_store.partition_S(tScP).shape, Float32)
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype), tSrS_t2r.layout,
        )

        if const_expr(self.activation == "relu"):
            if const_expr(self.blockscaled):
                # Blockscaled mode: compute scales and quantize
                sf0, sf1, sf2, sf3 = relu.relu_and_convert_blockscaled(tSrS_t2r, tSrP_r2t)
            else:
                relu.relu_and_convert(tSrS_t2r, tSrP_r2t, e2e_freq=self.e2e_freq, e2e_frg_limit=0)
        else:
            if const_expr(self.blockscaled):
                # Blockscaled mode: compute scales and quantize
                sf0, sf1, sf2, sf3 = gelu.gelu_and_convert_blockscaled(tSrS_t2r, tSrP_r2t)
            else:
                gelu.gelu_and_convert(tSrS_t2r, tSrP_r2t, e2e_freq=self.e2e_freq, e2e_frg_limit=0)

        # Write scale factors to sSFP for blockscaled mode
        # Pack 4 scale bytes into uint32 and write to SMEM
        # Following C++ pattern: tCsSFP_compact(thread_idx, sfp_stage) = arr_SF_P packed as uint32
        if const_expr(self.blockscaled):
            # softmax0_warp_ids has 4 warps (indices 0-3), so 128 threads per stage
            tidx = cute.arch.thread_idx()[0] % 128
            # Pack 4 uint8 scales into uint32 using PTX (little-endian: sf0 is lowest byte)
            scale_packed = pack_4xu8_to_u32(sf0, sf1, sf2, sf3)
            # Match C++ pattern: group_modes<0,3>(recast<uint32>(filter_zeros(SmemLayoutSFP{})))
            # 1. filter_zeros: removes zero-stride modes
            # 2. recast to uint32: 4 bytes per element
            # 3. group_modes<0,3>: flattens first 3 modes into thread index, keeps stage as mode 1
            #
            # Use filter_zeros + group_modes on the single-stage layout (per C++ SmemLayoutSFP)
            sSFP_filtered = cute.filter_zeros(sSFP)

            # Group modes 0 through rank-2 (all except last stage mode) into thread index
            # This matches C++ group_modes<0,3> which flattens first 3 modes
            sSFP_grouped = cute.group_modes(sSFP_filtered, 0, cute.rank(sSFP_filtered.layout) - 1)

            # Recast to uint32 pointer - this is the key step that makes strides work in uint32 units
            # The grouped layout's strides are in bytes, but after recast the effective stride
            # in uint32 units is stride/4. For stride (4,512), this becomes (1,128) in uint32 units.
            sSFP_u32_ptr = cute.recast_ptr(sSFP.iterator, dtype=cutlass.Uint32)

            # Create the compact tensor using recast_layout which properly adjusts strides for uint32
            # This matches C++ pattern: recast<uint32_t>(make_tensor(ptr, filter_zeros(layout)))
            # followed by group_modes<0,3>
            # recast_layout takes (new_type_bits, old_type_bits, src_layout)
            # sSFP has Uint8 elements (8 bits), we want Uint32 (32 bits)
            sSFP_u32_layout = cute.recast_layout(32, 8, sSFP_grouped.layout)
            sSFP_u32 = cute.make_tensor(sSFP_u32_ptr, sSFP_u32_layout)
            sfp_stage = Int32(0) if stage == 0 else Int32(1)

            # Write using layout-aware indexing: (thread_idx, stage)
            sSFP_u32[tidx, sfp_stage] = scale_packed
            # Fence SMEM write so MMA warp sees SFP when it does S2T copy
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta,
            )
        # Sequence barrier arrive
        if const_expr(self.s0_s1_barrier):
            cute.arch.mbarrier_arrive(mbar_ptr + mbar_s0_s1_sequence_offset + (1 - stage) * 4)

        for i in cutlass.range_constexpr(
            cute.size(tStP_r2t.shape[2]) // self.tmem_store_split_divisor * 3
        ):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()
        # Notify mma warp that P is ready
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_1_offset + stage)
        for i in cutlass.range_constexpr(
            cute.size(tStP_r2t.shape[2]) // self.tmem_store_split_divisor * 3,
            cute.size(tStP_r2t.shape[2]),
        ):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()
        # Notify mma warp that the 2nd half of P is ready
        cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_2_offset + stage)

        return mma_si_consumer_phase ^ 1, s0_s1_sequence_phase ^ 1


    @cute.jit
    def correction_epilogue(
        self,
        thr_mma: cute.core.ThrMma,
        tOtO: cute.Tensor,
        thread_idx: Int32,
        # scale: Float32,
        sO: cute.Tensor,
    ):
        """Apply final scaling and transformation to attention output before writing to global memory.

        This correction_epilogue function handles the final processing step for attention output values.
        It applies a scaling factor to the accumulated attention results and prepares the
        data for efficient transfer back to global memory.

        The method performs:
        1. Loading of accumulated attention results from tensor memory
        2. Application of the final output scaling factor
        3. Type conversion if necessary (typically from higher precision accumulator to output precision)
        4. Reorganization of data for optimal memory access patterns
        5. Preparation for efficient TMA store operations

        :param thr_mma: Thread MMA operation for the computation
        :type thr_mma: cute.core.ThrMma
        :param tOtO: Tensor containing accumulated attention output
        :type tOtO: cute.Tensor
        :param scale: Final scaling factor to apply to the output
        :type scale: Float32
        :param sO: Shared memory tensor for the final output
        :type sO: cute.Tensor
        """

        cO = cute.make_identity_tensor((self.mma_tiler_pv[0], self.mma_tiler_pv[1]))
        corr_tile_size = 32 * 8 // self.o_dtype.width
        tOsO = thr_mma.partition_C(sO)
        tOcO = thr_mma.partition_C(cO)

        tOtO_i = cute.logical_divide(tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.logical_divide(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOsO_i = cute.logical_divide(tOsO, cute.make_layout((self.m_block_size, corr_tile_size)))

        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
            self.mma_tiler_pv,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=False,
        )

        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_i[(None, None), 0])

        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy(
            smem_copy_atom,
            layout_tv=tiled_tmem_load.layout_dst_tv_tiled,
            tiler_mn=tiled_tmem_load.tiler_mn,
        )

        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tOsO_s2r = thr_tmem_load.partition_D(tOsO_i[(None, None), None])
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])

        for i in cutlass.range_constexpr(self.head_dim_v_padded // corr_tile_size):
            tOtO_t2r_i = tOtO_t2r[None, 0, 0, i]
            tOsO_r2s_i = tOsO_s2r[None, 0, 0, i]
            tOrO_frg = cute.make_fragment(tOcO_t2r[None, 0, 0, i].shape, self.pv_acc_dtype)
            cute.copy(tiled_tmem_load, tOtO_t2r_i, tOrO_frg)
            # for j in cutlass.range_constexpr(0, cute.size(tOrO_frg), 2):
            #     tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
            #         (tOrO_frg[j], tOrO_frg[j + 1]), (scale, scale),
            #     )
            tSMrO = cute.make_fragment(tOrO_frg.shape, self.o_dtype)
            o_vec = tOrO_frg.load()
            tSMrO.store(o_vec.to(self.o_dtype))
            cute.copy(tiled_smem_store, tSMrO, tOsO_r2s_i)

        # fence view async shared
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta,
        )

    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        mbar_ptr: cute.Pointer,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        epi_consumer_phase = Int32(0)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(not seqlen.has_cu_seqlens_q):
                mO_cur = mO[None, None, head_idx, batch_idx]
            else:
                offset = seqlen.offset_q if const_expr(not self.pack_gqa) else (0, seqlen.offset_q)
                mO_cur = cute.domain_offset((offset, 0), mO[None, None, head_idx])
            gO = cute.local_tile(mO_cur, (self.m_block_size, self.head_dim_v_padded), (None, 0))
            if const_expr(self.use_tma_O):
                tOsO, tOgO = cpasync.tma_partition(
                    tma_atom_O,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sO, 0, 2),
                    cute.group_modes(gO, 0, 2),
                )
                for stage in cutlass.range_constexpr(self.q_stage):
                    # wait from corr, issue tma store on smem
                    # 1. wait for O0 / O1 final
                    cute.arch.mbarrier_wait(mbar_ptr + self.mbar_corr_epi_full_offset + stage, epi_consumer_phase)
                    # 2. copy O0 / O1 to gmem
                    cute.copy(tma_atom_O, tOsO[None, stage], tOgO[None, self.q_stage * m_block + stage])
                    cute.arch.cp_async_bulk_commit_group()
                for stage in cutlass.range_constexpr(self.q_stage):
                    # Ensure O0 / O1 buffer is ready to be released
                    cute.arch.cp_async_bulk_wait_group(1 - stage, read=True)
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_corr_epi_empty_offset + stage)
            else:
                tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.epilogue_warp_ids))
                gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
                tOsO = gmem_thr_copy_O.partition_S(sO)
                cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
                tOgO = gmem_thr_copy_O.partition_D(gO)
                tOcO = gmem_thr_copy_O.partition_S(cO)
                t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
                tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
                # TODO: the packgqa case isn't correct rn (sometimes IMA), disabling it
                assert not self.pack_gqa
                pack_gqa = PackGQA(self.m_block_size, self.head_dim_v_padded, self.check_hdim_v_oob, self.qhead_per_kvhead)
                for stage in cutlass.range_constexpr(self.q_stage):
                    # wait from corr, issue tma store on smem
                    # 1. wait for O0 / O1 final
                    cute.arch.mbarrier_wait(mbar_ptr + self.mbar_corr_epi_full_offset + stage, epi_consumer_phase)
                    # 2. copy O0 / O1 to gmem
                    # load acc O from smem to rmem for wider vectorization
                    tOrO = cute.make_fragment_like(tOsO[None, None, None, 0], self.o_dtype)
                    cute.autovec_copy(tOsO[None, None, None, stage], tOrO)
                    # copy acc O from rmem to gmem
                    if const_expr(not self.pack_gqa):
                        for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                            if t0OcO[0, rest_m, 0][0] < seqlen.seqlen_q - (self.q_stage * m_block + stage) * self.m_block_size - tOcO[0][0]:
                                cute.copy(
                                    gmem_tiled_copy_O,
                                    tOrO[None, rest_m, None],
                                    tOgO[None, rest_m, None, self.q_stage * m_block + stage],
                                    pred=tOpO[None, rest_m, None] if self.check_hdim_v_oob else None,
                                )
                    else:
                        pack_gqa.store_O(mO_cur, tOrO, gmem_tiled_copy_O, tidx, self.q_stage * m_block + stage, seqlen.seqlen_q)
                    cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_corr_epi_empty_offset + stage)

            # Advance to next tile
            epi_consumer_phase ^= 1
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    def load_Q(
        self,
        tma_atom: cute.CopyAtom,
        tQgQ: cute.Tensor,
        tQsQ: cute.Tensor,
        mbar_full_ptr: cute.Pointer,
        mbar_empty_ptr: cute.Pointer,
        block: Int32,
        stage: int,
        phase: Int32,
        # Scale factor parameters (for blockscaled MXFP8)
        tma_atom_SFQ: Optional[cute.CopyAtom] = None,
        tSFQgSFQ: Optional[cute.Tensor] = None,
        tSFQsSFQ: Optional[cute.Tensor] = None,
    ):
        cute.arch.mbarrier_wait(mbar_empty_ptr + stage, phase)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_full_ptr + stage, self.tma_copy_q_bytes)
        cute.copy(
            tma_atom, tQgQ[None, block], tQsQ[None, stage], tma_bar_ptr=mbar_full_ptr + stage
        )
        # Load scale factor SFQ together with Q (using same barrier)
        if const_expr(self.blockscaled):
            cute.copy(
                tma_atom_SFQ, tSFQgSFQ[None, block], tSFQsSFQ[None, stage], tma_bar_ptr=mbar_full_ptr + stage
            )

    @cute.jit
    def load_KV(
        self,
        tma_atom: cute.CopyAtom,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        mbar_full_ptr: cute.Pointer,
        mbar_empty_ptr: cute.Pointer,
        block: Int32,
        producer_state: cutlass.pipeline.PipelineState,
        K_or_V: str,
        page_idx: Optional[Int32] = None,
        # Scale factor parameters for SFK (only used when K_or_V == "K")
        tma_atom_SFK: Optional[cute.CopyAtom] = None,
        tSFKgSFK: Optional[cute.Tensor] = None,
        tSFKsSFK: Optional[cute.Tensor] = None,
        # Scale factor parameters for SFV (only used when K_or_V == "V")
        tma_atom_SFV: Optional[cute.CopyAtom] = None,
        tSFVgSFV: Optional[cute.Tensor] = None,
        tSFVsSFV: Optional[cute.Tensor] = None,
    ):
        assert K_or_V in ("K", "V")
        tma_copy_bytes = self.tma_copy_k_bytes if const_expr(K_or_V == "K") else self.tma_copy_v_bytes
        stage, phase = producer_state.index, producer_state.phase
        cute.arch.mbarrier_wait(mbar_empty_ptr + stage, phase)
        if const_expr(K_or_V == "K" and self.uneven_kv_smem):
            # Before this round, the smem location was occupied by V, which is smaller than
            # K. So we need to wait for the stage after that (stage 1) to be empty as well.
            if stage == 0:
                cute.arch.mbarrier_wait(mbar_empty_ptr + 1, phase)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_full_ptr + stage, tma_copy_bytes)
        tXsX_cur = tXsX[None, stage]
        if const_expr(self.uneven_kv_smem):
            tXsX_cur = self.offset_kv_smem(tXsX_cur, stage, phase ^ 1)
        tXgX_cur = tXgX[None, block] if const_expr(page_idx is None) else tXgX[None, 0, page_idx]
        cute.copy(tma_atom, tXgX_cur, tXsX_cur, tma_bar_ptr=mbar_full_ptr + stage)

        # Load scale factor SFK together with K (using same barrier)
        if const_expr(K_or_V == "K" and self.blockscaled):
            tSFKsSFK_cur = tSFKsSFK[None, stage]
            tSFKgSFK_cur = tSFKgSFK[None, block] if const_expr(page_idx is None) else tSFKgSFK[None, 0, page_idx]
            cute.copy(tma_atom_SFK, tSFKgSFK_cur, tSFKsSFK_cur, tma_bar_ptr=mbar_full_ptr + stage)

        # Load scale factor SFV together with V (using same barrier)
        # Only load if tma_atom_SFV is provided (Phase 1: optional SFV loading)
        if const_expr(K_or_V == "V" and self.blockscaled and tma_atom_SFV is not None):
            tSFVsSFV_cur = tSFVsSFV[None, stage]
            tSFVgSFV_cur = tSFVgSFV[None, block] if const_expr(page_idx is None) else tSFVgSFV[None, 0, page_idx]
            # with cute.arch.elect_one():
            #     cute.printf("tSFVgSFV_cur.layout: {}", tSFVgSFV_cur.layout)
            #     cute.printf("tSFVsSFV_cur.layout: {}", tSFVsSFV_cur.layout)
            cute.copy(tma_atom_SFV, tSFVgSFV_cur, tSFVsSFV_cur, tma_bar_ptr=mbar_full_ptr + stage)

    @cute.jit
    def offset_kv_smem(self, sX: cute.Tensor, stage: Int32, phase: Int32):
        if const_expr(self.uneven_kv_smem):
            # smem layout is [smem_large, smem_small, smem_large], and the current stride is
            # (smem_large + smem_small) // 2. So for stage == 1, move right by offset if
            # phase == 0, or left by offset if phase == 1.
            offset = 0 if stage != 1 else self.uneven_kv_smem_offset * (1 - 2 * phase)
            return cute.make_tensor(sX.iterator + offset, sX.layout)
        else:
            return sX

    def make_and_init_load_kv_pipeline(self, load_kv_mbar_ptr):
        load_kv_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, len([self.load_warp_id])
        )
        load_kv_consumer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread, len([self.mma_warp_id]))
        return cutlass.pipeline.PipelineTmaUmma.create(
            barrier_storage=load_kv_mbar_ptr,
            num_stages=self.kv_stage,
            producer_group=load_kv_producer_group,
            consumer_group=load_kv_consumer_group,
            tx_count=self.tma_copy_k_bytes,
        )

    # @cute.jit
    # def warp_scheduler_barrier_init(self):
    #     warp_group_idx = utils.canonical_warp_group_idx(sync=False)
    #     if warp_group_idx == 0:
    #         cute.arch.barrier_arrive(
    #             barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1), number_of_threads=2 * 128,
    #         )

    # def warp_scheduler_barrier_sync(self):
    #     cute.arch.barrier(
    #         barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + utils.canonical_warp_group_idx(sync=False),
    #         number_of_threads=2 * 128
    #     )

    # def warp_scheduler_barrier_arrive(self):
    #     cur_wg = utils.canonical_warp_group_idx(sync=False)
    #     next_wg = 1 - cur_wg
    #     cute.arch.barrier_arrive(
    #         barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg, number_of_threads=2 * 128,
    #     )
