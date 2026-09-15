# @nolint
# Copyright (c) 2025, Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao.
import math
from functools import partial
from typing import Callable, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils_basic
import cutlass.utils.blockscaled_layout as blockscaled_utils
import quack.activation
from cutlass import const_expr, Float32, Int32, Int64, Uint32, Uint8
from cutlass._mlir.dialects import llvm
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import PipelineAsync
from cutlass.utils import LayoutEnum
from lp_fa4.cute import barrier, copy_utils, pipeline, utils
from lp_fa4.cute.blackwell_helpers import (  # noqa
    gemm_blockscaled,
    gemm_ptx_w_idx,
    gemm_w_idx,
    make_s2t_copy_partitions,
)
from lp_fa4.cute.block_info import BlockInfo
from lp_fa4.cute.block_sparse_utils import (
    get_block_sparse_iteration_info_bwd,
    get_m_block_from_iter_bwd,
    get_total_q_block_count_bwd,
    produce_block_sparse_q_loads_bwd_sm100,
)
from lp_fa4.cute.block_sparsity import BlockSparseTensors
from lp_fa4.cute.cute_dsl_utils import assume_tensor_aligned
from lp_fa4.cute.mask import AttentionMask
from lp_fa4.cute.named_barrier import NamedBarrierBwdSm100
from lp_fa4.cute.seqlen_info import SeqlenInfoQK
from lp_fa4.cute.softmax import (
    apply_score_mod_bwd_inner,
    apply_score_mod_inner,
    E4M3_MAX_NORM_RCP,
    fused_abs_max_f32,
    fused_amax_to_e8m0_scale_f32_hw as fused_amax_to_e8m0_scale_f32,
    max_f32,
    min_f32,
    redux_sync_max_abs_f32,
)
from lp_fa4.cute.tile_scheduler import (
    PersistentVarlenLookupScheduler,
    PersistentVarlenTileScheduler,
    SingleTileLPTBwdScheduler,  # noqa
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)
from lp_fa4.cute.utils import AuxData
from quack import layout_utils
from quack.cute_dsl_utils import ParamsBase


@dsl_user_op
def _st_shared_b8(addr: Int32, val: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.b8 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _st_shared_b32(addr: Int32, val: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.b32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class FlashAttentionBackwardSm100:
    arch = 100
    sf_vec_size = 32

    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        is_causal: bool = False,
        is_local: bool = False,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        tile_m: int = 128,
        tile_n: int = 128,
        is_persistent: bool = False,
        deterministic: bool = False,
        spt: Optional[bool] = None,
        cluster_size: int = 1,
        use_2cta_instrs: bool = False,
        score_mod: cutlass.Constexpr | None = None,
        score_mod_bwd: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        q_subtile_factor: cutlass.Constexpr[int] = 1,
        blockscaled: cutlass.Constexpr[bool] = False,
        output_mxfp8_dkv: cutlass.Constexpr[bool] = False,
        const_p_scale: cutlass.Constexpr[bool] = False,
        broadcast_q: cutlass.Constexpr[bool] = False,
        single_do_payload: cutlass.Constexpr[bool] = False,
        elide_full_tile_mask: cutlass.Constexpr[bool] = False,
        full_k_tiles_only: cutlass.Constexpr[bool] = False,
        broadcast_q_full_q_tiles: cutlass.Constexpr[bool] = False,
    ):
        if blockscaled:
            assert head_dim == 128
            assert head_dim_v is None or head_dim_v == 128
            assert tile_m in (64, 128) and tile_n == 128
            assert qhead_per_kvhead == 1
            assert cluster_size in (1, 2) and not use_2cta_instrs
            assert not is_persistent or cluster_size == 1
            assert not is_causal and not is_local and not deterministic
            assert score_mod is None and score_mod_bwd is None and mask_mod is None
            assert not has_aux_tensors
        assert not output_mxfp8_dkv or blockscaled
        assert not const_p_scale or blockscaled
        assert not single_do_payload or (blockscaled and not use_2cta_instrs)
        assert not full_k_tiles_only or (blockscaled and broadcast_q)
        assert not elide_full_tile_mask or (
            blockscaled and (not broadcast_q or full_k_tiles_only)
        )
        assert not broadcast_q_full_q_tiles or broadcast_q

        # MXFP8 scale-factor tiles require a 128-element padded head dimension.
        hdim_multiple_of = self.sf_vec_size * 4 if blockscaled else 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv

        self.tile_m = tile_m
        self.tile_n = tile_n

        assert self.tile_hdim <= 128 or (self.tile_hdim == 192 and self.tile_hdimv == 128)
        assert self.tile_hdimv <= 128

        self.use_2cta_instrs = bool(use_2cta_instrs and cluster_size == 2)
        self.cta_group_size = 2 if self.use_2cta_instrs else 1
        self.use_cluster2_group1 = bool(
            blockscaled and cluster_size == 2 and not self.use_2cta_instrs
        )
        assert self.tile_hdim != 192 or self.use_2cta_instrs, "Must use 2CTA for hdim 192"

        # CTA tiler
        self.cta_tiler = (tile_n, tile_m, self.tile_hdim)
        # S = K @ Q.T
        self.mma_tiler_kq = (self.cta_group_size * tile_n, tile_m, self.tile_hdim)
        # dP = V @ dO.T
        self.mma_tiler_vdo = (self.cta_group_size * tile_n, tile_m, self.tile_hdimv)
        # dV = P.T @ dO
        self.mma_tiler_pdo = (self.cta_group_size * tile_n, self.tile_hdimv, tile_m)
        # dK = dS.T @ Q
        self.mma_tiler_dsq = (self.cta_group_size * tile_n, self.tile_hdim, tile_m)
        # dQ = dS @ K
        # 2-CTA: reduction dim is cluster-wide (tile_n * cta_group_size).
        self.mma_tiler_dsk = (tile_m, self.tile_hdim, tile_n * self.cta_group_size)

        self.acc_dtype = Float32

        assert cluster_size in (1, 2), "Only cluster_size=1 or 2 is supported"
        self.cluster_shape_mn = (cluster_size, 1)
        self.is_persistent = is_persistent
        self.is_causal = is_causal
        self.is_local = is_local
        self.qhead_per_kvhead = qhead_per_kvhead
        self.pack_gqa = False
        self.deterministic = deterministic
        self.spt_override = spt
        self.blockscaled = blockscaled
        self.output_mxfp8_dkv = output_mxfp8_dkv
        self.sf_dtype = cutlass.Float8E8M0FNU if blockscaled else None
        self.const_p_scale = const_p_scale
        self.broadcast_q = broadcast_q
        self.single_do_payload = single_do_payload
        self.bf16_broadcast_q = broadcast_q and not blockscaled
        self.elide_full_tile_mask = elide_full_tile_mask
        self.full_k_tiles_only = full_k_tiles_only
        self.broadcast_q_full_q_tiles = broadcast_q_full_q_tiles
        self.use_fused_mxfp8_full_tile_schedule = bool(
            self.blockscaled
            and self.output_mxfp8_dkv
            and self.broadcast_q
            and self.is_persistent
            and not self.use_2cta_instrs
            and self.cluster_shape_mn == (1, 1)
            and self.single_do_payload
            and self.full_k_tiles_only
        )
        assert not self.bf16_broadcast_q or (
            self.is_persistent
            and self.broadcast_q
            and not self.use_2cta_instrs
            and self.cluster_shape_mn == (1, 1)
            and not self.deterministic
            and self.qhead_per_kvhead == 1
            and not self.blockscaled
            and not self.is_causal
            and not self.is_local
            and self.tile_hdim == 128
            and self.tile_hdimv == 128
            and self.tile_m == 128
            and self.tile_n == 128
        )

        # Score mod and mask mod support
        self.score_mod = score_mod
        self.score_mod_bwd = score_mod_bwd
        self.mask_mod = mask_mod
        self.has_aux_tensors = has_aux_tensors
        self.q_subtile_factor = q_subtile_factor
        # For score_mod, use vec_size=1 (like forward) to handle per-element indices
        if cutlass.const_expr(has_aux_tensors):
            self.vec_size: cutlass.Constexpr = 1
        else:
            self.vec_size: cutlass.Constexpr = 4
        self.qk_acc_dtype = Float32

        # Speed optimizations, does not affect correctness
        self.shuffle_LSE = False
        self.shuffle_dPsum = False
        self.reduce_warp_ids = (0, 1, 2, 3)
        self.compute_warp_ids = (4, 5, 6, 7, 8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.relay_warp_id = 14
        self.empty_warp_id = 15

        # 16 warps -> 512 threads
        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.reduce_warp_ids,
                *self.compute_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.relay_warp_id,
                self.empty_warp_id,
            )
        )
        # NamedBarrier
        self.compute_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.Compute),
            num_threads=len(self.compute_warp_ids) * cute.arch.WARP_SIZE,
        )
        # self.epilogue_sync_barrier = pipeline.NamedBarrier(
        #     barrier_id=2,
        #     num_threads=self.num_compute_warps * self.threads_per_warp,
        # )
        self.reduce_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.dQaccReduce),
            num_threads=len(self.reduce_warp_ids) * cute.arch.WARP_SIZE,
        )
        # TMEM setup
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        # self.tmem_dK_offset = 0
        # self.tmem_dV_offset = self.tmem_dK_offset + self.tile_hdim
        # self.tmem_dQ_offset = self.tmem_dV_offset + self.tile_hdimv
        # self.tmem_dP_offset = self.tmem_dQ_offset  # overlap with dQ
        # self.tmem_S_offset = self.tmem_dQ_offset + max(self.tile_m, self.tile_hdim)
        # self.tmem_P_offset = self.tmem_S_offset  # overlap with S
        # self.tmem_total = self.tmem_S_offset + self.tile_n
        # assert self.tmem_total <= self.tmem_alloc_cols

        if self.use_2cta_instrs and self.tile_hdim == 192 and self.tile_hdimv == 128:
            assert self.tile_m == 128
            assert self.tile_n == 128
            self.tmem_dV_offset = 0
            self.tmem_dK_offset = self.tmem_dV_offset + self.tile_hdimv
            self.tmem_S_offset = self.tmem_dK_offset + self.tile_hdim
            self.tmem_P_offset = self.tmem_S_offset  # overlap with S
            self.tmem_dP_offset = 512 - self.tile_m
            self.tmem_dS_offset = self.tmem_dP_offset  # overlaps with dP
            self.tmem_dQ_offset = 512 - self.tile_hdim // 2
        else:
            self.tmem_S_offset = 0
            self.tmem_P_offset = 0  # overlap with S
            self.tmem_dV_offset = self.tmem_S_offset + self.tile_n
            self.tmem_dP_offset = self.tmem_dV_offset + self.tile_hdimv
            self.tmem_dQ_offset = (
                (self.tmem_S_offset + (self.tile_hdim // 2))
                if self.use_2cta_instrs
                else self.tmem_dP_offset
            )
            self.tmem_dK_offset = self.tmem_dP_offset + self.tile_m
            self.tmem_dS_offset = self.tmem_dP_offset  # overlap with dP

        self.tmem_SF_prologue_offset = self.tmem_dK_offset
        self.tmem_SF_offset = self.tmem_dP_offset + (32 if self.use_2cta_instrs else 48)
        self.tmem_SF_offset_dP = self.tmem_S_offset + 48

        if (not is_causal and not is_local) or deterministic:
            self.num_regs_reduce = 136 if self.use_2cta_instrs else 152
            self.num_regs_compute = 136
            self.num_regs_load = 104 if self.use_2cta_instrs else 96 - 8
            self.num_regs_mma = 104 if self.use_2cta_instrs else self.num_regs_load
        else:
            self.num_regs_reduce = 136 if self.use_2cta_instrs else 136
            self.num_regs_compute = 136 if self.use_2cta_instrs else 144
            self.num_regs_load = 104 if self.use_2cta_instrs else 96 - 8
            self.num_regs_mma = 104 if self.use_2cta_instrs else self.num_regs_load
        if (
            self.blockscaled
            and self.output_mxfp8_dkv
            and self.broadcast_q
            and self.is_persistent
            and not self.use_2cta_instrs
            and self.cluster_shape_mn == (1, 1)
            and self.single_do_payload
        ):
            self.num_regs_reduce = 136
            self.num_regs_compute = 144
        self.num_regs_empty = 24

        if const_expr(self.tile_hdim == 192):
            if not is_causal and not is_local:
                self.num_regs_reduce = 128 + 8
                self.num_regs_compute = 128 + 8
                self.num_regs_load = 128 - 24
                self.num_regs_mma = self.num_regs_load
            else:
                self.num_regs_reduce = 128 + 8
                self.num_regs_compute = 128 + 8
                self.num_regs_load = 128 - 24
                self.num_regs_mma = self.num_regs_load

        assert (
            self.num_regs_reduce
            + self.num_regs_compute * 2
            + max(self.num_regs_load, self.num_regs_mma)
            <= 512
        )
        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.Q_stage = 1 if self.use_2cta_instrs else 2
        self.dO_stage = 1
        self.single_stage = 1
        # LSE_stage = Q_stage and dPsum_stage = dO_stage
        self.sdKVaccum_stage = 2
        # number of tma reduce adds per dQacc mma
        # todo: try 32/1 or 48/2 for 2cta d=192 dv=128
        if self.use_2cta_instrs and self.tile_hdim == 192:
            self.dQ_reduce_ncol_t2r = 32
            self.dQ_reduce_ncol = 24 if not self.is_causal else 32
            self.sdQaccum_stage = 2 if not self.is_causal else 1
        else:
            if self.use_2cta_instrs:
                self.dQ_reduce_ncol = 16 if self.deterministic else 8
                self.sdQaccum_stage = 2 if self.deterministic else 4
                self.dQ_reduce_ncol_t2r = 32
            else:
                self.dQ_reduce_ncol = 32
                self.sdQaccum_stage = (
                    5
                    if self.dqaccum_dtype == cutlass.Float16
                    and self.use_dedicated_mxfp8_dkv_tma
                    else max(1, 64 // self.dQ_reduce_ncol)
                )
                self.dQ_reduce_ncol_t2r = 32
        assert (self.tile_hdim // self.cta_group_size) % self.dQ_reduce_ncol == 0
        self.dQaccum_reduce_stage = self.tile_hdim // self.dQ_reduce_ncol
        self.dQaccum_reduce_stage_t2r = self.tile_hdim // self.dQ_reduce_ncol_t2r
        self.cluster_reduce_dQ = False and cute.size(self.cluster_shape_mn) > 1
        # number of tma reduce adds for dKacc and dVacc epilogue (must divide hdim_per_wg)
        self.dK_reduce_ncol = math.gcd(32, self.tile_hdim // 2)
        # CTA group for MMA operations
        self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE

    def _get_tiled_mma(self):
        # S.T = K @ Q.T
        tiled_mma_S = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_kq[:2],
        )
        # dP.T = V @ dO.T
        tiled_mma_dP = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_vdo[:2],
        )
        # dV += P.T @ dO --> (K, MN) major
        tiled_mma_dV = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,  # P_major_mode
            tcgen05.OperandMajorMode.MN,  # dO_major_mode
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_pdo[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        # dK += dS.T @ Q
        tiled_mma_dK = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,  # dS_major_mode
            tcgen05.OperandMajorMode.MN,  # Q_major_mode
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_dsq[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        # dQ = dS @ K
        tiled_mma_dQ = sm100_utils_basic.make_trivial_tiled_mma(
            self.k_dtype,
            tcgen05.OperandMajorMode.MN,  # dS_major_mode
            tcgen05.OperandMajorMode.MN,  # Kt_major_mode
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_dsk[:2],
        )
        return tiled_mma_S, tiled_mma_dP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ

    def _get_tiled_mma_blockscaled(self):
        tiled_mma_S = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_tiler_kq[:2],
        )
        tiled_mma_dP = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_tiler_vdo[:2],
        )
        tiled_mma_dV = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_tiler_pdo[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        tiled_mma_dK = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_tiler_dsq[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        tiled_mma_dQ = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
            self.k_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_dsk[:2],
        )
        return (
            tiled_mma_S,
            tiled_mma_dP,
            tiled_mma_dV,
            tiled_mma_dK,
            tiled_mma_dQ,
        )

    def _get_tiled_mma_sfb(self):
        self.mma_tiler_kq_sfb = (
            self.mma_tiler_kq[0] // self.cta_group_size,
            cute.round_up(self.mma_tiler_kq[1], 128),
            self.mma_tiler_kq[2],
        )
        self.mma_tiler_vdo_sfb = (
            self.mma_tiler_vdo[0] // self.cta_group_size,
            cute.round_up(self.mma_tiler_vdo[1], 128),
            self.mma_tiler_vdo[2],
        )
        self.mma_tiler_pdo_sfb = (
            self.mma_tiler_pdo[0] // self.cta_group_size,
            cute.round_up(self.mma_tiler_pdo[1], 128),
            self.mma_tiler_pdo[2],
        )
        self.mma_tiler_dsq_sfb = (
            self.mma_tiler_dsq[0] // self.cta_group_size,
            cute.round_up(self.mma_tiler_dsq[1], 128),
            self.mma_tiler_dsq[2],
        )
        self.mma_tiler_dsk_sfb = (
            self.mma_tiler_dsk[0] // self.cta_group_size,
            cute.round_up(self.mma_tiler_dsk[1], 128),
            self.mma_tiler_dsk[2],
        )
        tiled_mma_S_sfb = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_kq_sfb[:2],
        )
        tiled_mma_dP_sfb = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_vdo_sfb[:2],
        )
        tiled_mma_dV_sfb = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_pdo_sfb[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        tiled_mma_dK_sfb = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_dsq_sfb[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        tiled_mma_dQ_sfb = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
            self.k_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_dsk_sfb[:2],
        )
        return (
            tiled_mma_S_sfb,
            tiled_mma_dP_sfb,
            tiled_mma_dV_sfb,
            tiled_mma_dK_sfb,
            tiled_mma_dQ_sfb,
        )

    @staticmethod
    def _make_smem_u32_view(tensor):
        ptr = cute.recast_ptr(tensor.iterator, dtype=cutlass.Uint32)
        filtered = cute.filter_zeros(tensor)
        grouped = cute.group_modes(filtered, 0, cute.rank(filtered.layout) - 1)
        layout = cute.recast_layout(32, 8, grouped.layout)
        return cute.make_tensor(ptr, layout)

    @cute.jit
    def _fill_sf(self, tensor: cute.Tensor, packed_word: cutlass.Int32):
        assert cute.cosize(tensor.layout) % 4 == 0
        num_words = cute.cosize(tensor.layout) // 4
        tensor_u32 = cute.make_tensor(
            cute.recast_ptr(tensor.iterator, dtype=cutlass.Uint32),
            cute.make_layout((num_words,)),
        )
        lane_idx = cute.arch.thread_idx()[0] % cute.arch.WARP_SIZE
        for row_group in cutlass.range_constexpr(
            cute.ceil_div(num_words, cute.arch.WARP_SIZE)
        ):
            row = lane_idx + row_group * cute.arch.WARP_SIZE
            if row < num_words:
                addr = Int32(utils.elem_pointer(tensor_u32, (row,)).toint())
                _st_shared_b32(addr, packed_word)

    def _setup_smem_layout(self):
        # S.T = K @ Q.T
        sK_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_S,
            self.mma_tiler_kq,
            self.k_dtype,
            1,
        )
        self.sK_layout = cute.slice_(sK_layout, (None, None, None, 0))
        self.sQ_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_S,
            self.mma_tiler_kq,
            self.q_dtype,
            self.Q_stage,
        )
        # dP.T = V @ dO.T
        sV_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            self.v_dtype,
            1,
        )
        self.sV_layout = cute.slice_(sV_layout, (None, None, None, 0))
        self.sdOt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            self.do_dtype,
            self.dO_stage,
        )
        # dV += P.T @ dO
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.do_dtype,
            1,
        )
        self.tP_layout = cute.slice_(tP_layout, (None, None, None, 0))
        self.sdO_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.do_dtype,
            self.dO_stage,
        )
        # dK += dS.T @ Q
        sdSt_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.ds_dtype,
            1,
        )
        self.sdSt_layout = cute.slice_(sdSt_layout, (None, None, None, 0))
        tdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.ds_dtype,
            1,
        )
        self.tdS_layout = cute.slice_(tdS_layout, (None, None, None, 0))
        self.sQt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.q_dtype,
            self.Q_stage,
        )
        # dQ = dS @ K
        sdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.ds_dtype,
            1,
        )
        self.sdS_layout = cute.slice_(sdS_layout, (None, None, None, 0))
        sKt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.k_dtype,
            1,
        )
        self.sKt_layout = cute.slice_(sKt_layout, (None, None, None, 0))
        if const_expr(self.blockscaled):
            sdS_dQ_layout = sm100_utils_basic.make_smem_layout_a(
                self.tiled_mma_dQ_bs,
                self.mma_tiler_dsk,
                self.ds_dtype,
                1,
            )
            self.sdS_dQ_data_layout = cute.slice_(
                sdS_dQ_layout, (None, None, None, 0)
            )
            sKt_dQ_layout = sm100_utils_basic.make_smem_layout_b(
                self.tiled_mma_dQ_bs,
                self.mma_tiler_dsk,
                self.k_dtype,
                1,
            )
            self.sKt_dQ_data_layout = cute.slice_(
                sKt_dQ_layout, (None, None, None, 0)
            )
        else:
            self.sdS_dQ_data_layout = None
            self.sKt_dQ_data_layout = None
        self.sdS_xchg_layout = cute.make_layout(shape=(self.tile_n, self.tile_m // 2))

        self.sdQaccum_layout = cute.make_layout(
            (self.tile_m * self.dQ_reduce_ncol, self.sdQaccum_stage)
        )
        self.dQ_reduce_mrow = self.tile_m // self.dQaccum_reduce_stage
        assert self.tile_m % self.dQaccum_reduce_stage == 0
        self.sdQaccum_tma_tile = (self.dQ_reduce_mrow, self.tile_hdim)
        self.sdQaccum_tma_layout = cute.make_layout(
            (self.dQ_reduce_mrow, self.tile_hdim, self.sdQaccum_stage),
            stride=(
                self.tile_hdim,
                1,
                self.dQ_reduce_mrow * self.tile_hdim,
            ),
        )
        self.sLSE_layout = cute.make_layout(
            shape=(self.tile_m, self.Q_stage), stride=(1, cute.round_up(self.tile_m, 64))
        )
        self.sdPsum_layout = cute.make_layout(
            shape=(self.tile_m, self.dO_stage),
            stride=(1, cute.round_up(self.tile_m, 64)),
        )
        self.sdK_epi_tile = (
            self.tile_n,
            math.gcd(128 // (self.dk_dtype.width // 8), self.tile_hdim // 2),  # 64 or 32
        )  # subtiles mma_tiler_dsq[:2] = mma_tiler_pdo[:2]
        self.sdV_epi_tile = (
            self.tile_n,
            math.gcd(128 // (self.dk_dtype.width // 8), self.tile_hdimv // 2),  # 64 or 32
        )  # subtiles mma_tiler_dsq[:2] = mma_tiler_pdo[:2]
        # headdim_64 gets 1 stage
        self.num_epi_stages = max(1, (self.tile_hdim // 2) // self.sdK_epi_tile[1])
        self.num_epi_stages_v = max(1, (self.tile_hdimv // 2) // self.sdV_epi_tile[1])
        self.sdK_flat_epi_tile = self.tile_n * (self.tile_hdim // 2) // self.num_epi_stages
        self.sdV_flat_epi_tile = self.tile_n * (self.tile_hdimv // 2) // self.num_epi_stages_v
        if const_expr(not self.dKV_postprocess):
            self.sdK_layout = sm100_utils_basic.make_smem_layout_epi(
                self.dk_dtype,
                LayoutEnum.ROW_MAJOR,
                self.sdK_epi_tile,
                2,  # num compute wgs
            )
            self.sdV_layout = sm100_utils_basic.make_smem_layout_epi(
                self.dv_dtype,
                LayoutEnum.ROW_MAJOR,
                self.sdV_epi_tile,
                2,  # num compute wgs
            )
        else:
            self.sdK_layout = cute.make_layout((self.tile_n * self.dK_reduce_ncol, 2))
            # self.dK_reduce_ncol same for dV
            self.sdV_layout = cute.make_layout((self.tile_n * self.dK_reduce_ncol, 2))

        self.sSFQ_layout = None
        self.sSFK_layout = None
        self.sSFV_layout = None
        self.sSFDO_layout = None

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdQaccumTma: Optional[cute.Tensor],
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mdSFK_out: Optional[cute.Tensor],
        mdSFV_out: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        dq_accum_scale: Float32 = 1.0,
        mSFQ: Optional[cute.Tensor] = None,
        mSFK: Optional[cute.Tensor] = None,
        mSFV: Optional[cute.Tensor] = None,
        mSFDO: Optional[cute.Tensor] = None,
        mQ_dK: Optional[cute.Tensor] = None,
        mSFQ_dK: Optional[cute.Tensor] = None,
        mdO_dV: Optional[cute.Tensor] = None,
        mSFDO_dV: Optional[cute.Tensor] = None,
        mK_dQ: Optional[cute.Tensor] = None,
        mSFK_dQ: Optional[cute.Tensor] = None,
        mCuSeqlensSFQ: Optional[cute.Tensor] = None,
        mCuSeqlensSFK: Optional[cute.Tensor] = None,
        total_sf_q: Int32 | int | None = None,
        total_sf_k: Int32 | int | None = None,
        mTileToBatch: Optional[cute.Tensor] = None,
        mTileToHead: Optional[cute.Tensor] = None,
        mTileToBlock: Optional[cute.Tensor] = None,
        aux_data: Optional[AuxData] = None,
        # Block-sparse tensors (Q direction - for iterating m_blocks per n_block):
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.do_dtype = mdO.element_type
        self.lse_dtype = mLSE.element_type
        self.dpsum_dtype = mdPsum.element_type
        self.dqaccum_dtype = mdQaccum.element_type
        self.dk_dtype = mdK.element_type
        self.dv_dtype = mdV.element_type
        self.ds_dtype = self.q_dtype

        if const_expr(self.dqaccum_dtype not in [Float32, cutlass.Float16]):
            raise TypeError("dQaccum must use Float32 or Float16 elements")
        if const_expr(
            self.dqaccum_dtype == cutlass.Float16 and not self.blockscaled
        ):
            raise TypeError("Float16 dQ accumulation requires MXFP8 backward")

        if const_expr(self.blockscaled):
            if const_expr(
                any(
                    tensor.element_type != cutlass.Float8E4M3FN
                    for tensor in (mQ, mK, mV, mdO)
                )
            ):
                raise TypeError("MXFP8 backward requires E4M3 data operands")
            if const_expr(aux_data is not None and aux_data.tensors is not None):
                raise TypeError("MXFP8 backward does not support aux tensors")
            if const_expr(self.output_mxfp8_dkv):
                if const_expr(
                    self.dk_dtype != cutlass.Float8E4M3FN
                    or self.dv_dtype != cutlass.Float8E4M3FN
                ):
                    raise TypeError("MXFP8 dK/dV output requires E4M3 output tensors")
                if const_expr(mdSFK_out is None or mdSFV_out is None):
                    raise TypeError("MXFP8 dK/dV output requires scale tensors")
                if const_expr(
                    mdSFK_out.element_type != Uint8
                    or mdSFV_out.element_type != Uint8
                ):
                    raise TypeError("MXFP8 dK/dV output scales must use uint8 elements")
            elif const_expr(mdSFK_out is not None or mdSFV_out is not None):
                raise TypeError("dK/dV output scales require MXFP8 output mode")
            if const_expr(mCuSeqlensQ is None or mCuSeqlensK is None):
                raise TypeError(
                    "MXFP8 backward currently requires packed varlen inputs"
                )
            external_sfs = (
                mSFQ,
                mSFK,
                mSFV,
                mSFDO,
                mSFQ_dK,
                mSFDO_dV,
                mSFK_dQ,
            )
            any_external_sfs = any(tensor is not None for tensor in external_sfs)
            all_external_sfs = all(tensor is not None for tensor in external_sfs)
            if const_expr(any_external_sfs and not all_external_sfs):
                raise TypeError("MXFP8 external scale tensors must be provided together")
            self.use_external_mxfp8_scales = all_external_sfs
            if const_expr(self.use_external_mxfp8_scales):
                if const_expr(
                    any(
                        tensor.element_type
                        not in (cutlass.Float8E8M0FNU, cutlass.Uint8)
                        for tensor in external_sfs
                    )
                ):
                    raise TypeError("MXFP8 scale tensors must use E8M0 or uint8 elements")
                if const_expr(total_sf_q is None or total_sf_k is None):
                    raise TypeError(
                        "MXFP8 external scales require padded total_sf_q and total_sf_k"
                    )
                if const_expr(
                    mCuSeqlensSFQ is None or mCuSeqlensSFK is None
                ):
                    raise TypeError(
                        "MXFP8 external scales require split Q and K scale offsets"
                    )
                if const_expr(
                    mQ_dK is None
                    or mdO_dV is None
                    or mK_dQ is None
                    or mSFK_dQ is None
                ):
                    raise TypeError(
                        "MXFP8 external scales require Q_dK, dO_dV, K_dQ, and SFK_dQ"
                    )
            external_payloads = (mQ_dK, mdO_dV, mK_dQ)
            if const_expr(
                any(
                    tensor is not None
                    and tensor.element_type != cutlass.Float8E4M3FN
                    for tensor in external_payloads
                )
            ):
                raise TypeError("MXFP8 alternate data operands must use E4M3 elements")
        else:
            self.use_external_mxfp8_scales = False
            if const_expr(
                any(
                    tensor is not None
                    for tensor in (
                        mSFQ,
                        mSFK,
                        mSFV,
                        mSFDO,
                        mQ_dK,
                        mSFQ_dK,
                        mdO_dV,
                        mSFDO_dV,
                        mK_dQ,
                        mSFK_dQ,
                        mCuSeqlensSFQ,
                        mCuSeqlensSFK,
                    )
                )
            ):
                raise TypeError("MXFP8 external operands require blockscaled mode")
        self.use_dedicated_mxfp8_dkv_tma = bool(
            self.output_mxfp8_dkv
            and self.full_k_tiles_only
            and self.use_external_mxfp8_scales
            and self.is_persistent
            and self.cluster_shape_mn == (1, 1)
            and self.tile_hdim == 128
            and self.tile_hdimv == 128
            and self.single_do_payload
        )
        self.use_dedicated_k_pipeline = self.bf16_broadcast_q or (
            self.blockscaled
            and self.broadcast_q
            and self.use_external_mxfp8_scales
            and self.cluster_shape_mn == (1, 1)
        )
        self.unified_ds_scale = self.use_cluster2_group1 or (
            self.blockscaled
            and self.broadcast_q
            and self.use_external_mxfp8_scales
            and self.cluster_shape_mn == (1, 1)
        )
        if const_expr(mCuSeqlensSFQ is not None):
            mCuSeqlensSFQ = assume_tensor_aligned(mCuSeqlensSFQ)
        if const_expr(mCuSeqlensSFK is not None):
            mCuSeqlensSFK = assume_tensor_aligned(mCuSeqlensSFK)
        mQ_dK = assume_tensor_aligned(mQ_dK) if const_expr(mQ_dK is not None) else None
        mdO_dV = assume_tensor_aligned(mdO_dV) if const_expr(mdO_dV is not None) else None
        mK_dQ = assume_tensor_aligned(mK_dQ) if const_expr(mK_dQ is not None) else None
        mSFK_dQ = (
            assume_tensor_aligned(mSFK_dQ)
            if const_expr(mSFK_dQ is not None)
            else None
        )
        self.is_varlen_k = mCuSeqlensK is not None or mSeqUsedK is not None
        self.is_varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None
        self.use_tma_store = not self.output_mxfp8_dkv and (
            self.bf16_broadcast_q
            or not (self.qhead_per_kvhead == 1 and mCuSeqlensK is not None)
        )
        # self.use_tma_store = not self.qhead_per_kvhead == 1
        self.dKV_postprocess = self.qhead_per_kvhead > 1

        if const_expr(self.dKV_postprocess):
            assert self.dk_dtype.width == 32, "Must accumulate dK in float precision for GQA"
            assert self.dv_dtype.width == 32, "Must accumulate dV in float precision for GQA"

        mdQaccum, mdK, mdV = [assume_tensor_aligned(t) for t in (mdQaccum, mdK, mdV)]
        mdSFK_out = (
            assume_tensor_aligned(mdSFK_out)
            if const_expr(mdSFK_out is not None)
            else None
        )
        mdSFV_out = (
            assume_tensor_aligned(mdSFV_out)
            if const_expr(mdSFV_out is not None)
            else None
        )

        # (b, s, n, h) --> (s, h, n, b) or (t, n, h) -> (t, h, n)
        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ, mdO = [layout_utils.select(t, mode=QO_layout_transpose) for t in (mQ, mdO)]
        if const_expr(mQ_dK is not None):
            mQ_dK = layout_utils.select(mQ_dK, mode=QO_layout_transpose)
        if const_expr(mdO_dV is not None):
            mdO_dV = layout_utils.select(mdO_dV, mode=QO_layout_transpose)

        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK, mV = [layout_utils.select(t, mode=KV_layout_transpose) for t in (mK, mV)]
        if const_expr(mK_dQ is not None):
            mK_dQ = layout_utils.select(mK_dQ, mode=KV_layout_transpose)

        # (b, n, s) --> (s, n, b) or (n, t) --> (t, n)
        LSE_dPsum_dQaccum_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mLSE, mdPsum, mdQaccum = [
            layout_utils.select(t, mode=LSE_dPsum_dQaccum_transpose)
            for t in (mLSE, mdPsum, mdQaccum)
        ]

        if const_expr(not self.dKV_postprocess):
            layout_dKV_transpose = KV_layout_transpose
        else:
            layout_dKV_transpose = (
                [2, 1, 0] if const_expr(mCuSeqlensK is None) else [1, 0]
            )
        mdK, mdV = [
            layout_utils.select(t, mode=layout_dKV_transpose) for t in (mdK, mdV)
        ]
        mdSFK_out = (
            layout_utils.select(mdSFK_out, mode=layout_dKV_transpose)
            if const_expr(mdSFK_out is not None)
            else None
        )
        mdSFV_out = (
            layout_utils.select(mdSFV_out, mode=layout_dKV_transpose)
            if const_expr(mdSFV_out is not None)
            else None
        )
        # (s, h, n, b) --> (h, s, n, b) or (t, h, n) -> (h, t, b)
        dO_transpose = [1, 0, 2, 3] if const_expr(mCuSeqlensQ is None) else [1, 0, 2]
        mdO = layout_utils.select(mdO, mode=dO_transpose)
        if const_expr(mdO_dV is not None):
            mdO_dV = layout_utils.select(mdO_dV, mode=dO_transpose)
        # Transposes for 2-CTA K/Q paths (Q follows Q seqlens, K follows K seqlens)
        transpose_sh_q = dO_transpose
        transpose_sh_k = [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]

        # (b, n, block, stage) -> (block, stage, n, b)
        semaphore_transpose = [2, 3, 1, 0]
        if const_expr(self.deterministic):
            assert mdQ_semaphore is not None
            mdQ_semaphore = layout_utils.select(mdQ_semaphore, mode=semaphore_transpose)

        if const_expr(self.deterministic and self.qhead_per_kvhead > 1):
            assert mdK_semaphore is not None
            assert mdV_semaphore is not None
            mdK_semaphore, mdV_semaphore = [
                layout_utils.select(t, mode=semaphore_transpose)
                for t in (mdK_semaphore, mdV_semaphore)
            ]
        else:
            mdK_semaphore = None
            mdV_semaphore = None

        self._setup_attributes()
        (
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dK,
            self.tiled_mma_dV,
            self.tiled_mma_dQ,
        ) = self._get_tiled_mma()
        if const_expr(self.blockscaled):
            (
                self.tiled_mma_S_bs,
                self.tiled_mma_dP_bs,
                self.tiled_mma_dV_bs,
                self.tiled_mma_dK_bs,
                self.tiled_mma_dQ_bs,
            ) = self._get_tiled_mma_blockscaled()
            (
                self.tiled_mma_S_sfb,
                self.tiled_mma_dP_sfb,
                self.tiled_mma_dV_sfb,
                self.tiled_mma_dK_sfb,
                self.tiled_mma_dQ_sfb,
            ) = self._get_tiled_mma_sfb()
            self.tiled_mma_S = self.tiled_mma_S_bs
            self.tiled_mma_dP = self.tiled_mma_dP_bs
            self.tiled_mma_dV = self.tiled_mma_dV_bs
            self.tiled_mma_dK = self.tiled_mma_dK_bs
        else:
            self.tiled_mma_S_bs = None
            self.tiled_mma_dP_bs = None
            self.tiled_mma_dV_bs = None
            self.tiled_mma_dK_bs = None
            self.tiled_mma_dQ_bs = None
            self.tiled_mma_S_sfb = None
            self.tiled_mma_dP_sfb = None
            self.tiled_mma_dV_sfb = None
            self.tiled_mma_dK_sfb = None
            self.tiled_mma_dQ_sfb = None
        self._setup_smem_layout()

        if const_expr(self.blockscaled):
            self.sSFQ_layout = blockscaled_utils.make_smem_layout_sfb(
                self.tiled_mma_S_bs,
                self.mma_tiler_kq,
                self.sf_vec_size,
                self.Q_stage,
            )
            self.sSFK_layout = blockscaled_utils.make_smem_layout_sfa(
                self.tiled_mma_S_bs, self.mma_tiler_kq, self.sf_vec_size, 1
            )
            self.sSFV_layout = blockscaled_utils.make_smem_layout_sfa(
                self.tiled_mma_dP_bs, self.mma_tiler_vdo, self.sf_vec_size, 1
            )
            self.sSFDO_layout = blockscaled_utils.make_smem_layout_sfb(
                self.tiled_mma_dP_bs,
                self.mma_tiler_vdo,
                self.sf_vec_size,
                self.dO_stage,
            )
            self.sSFP_layout = blockscaled_utils.make_smem_layout_sfa(
                self.tiled_mma_dV_bs, self.mma_tiler_pdo, self.sf_vec_size, 2
            )
            self.sSFDS_layout = blockscaled_utils.make_smem_layout_sfa(
                self.tiled_mma_dK_bs, self.mma_tiler_dsq, self.sf_vec_size, 2
            )
            self.sSFQ_dK_layout = blockscaled_utils.make_smem_layout_sfb(
                self.tiled_mma_dK_bs,
                self.mma_tiler_dsq,
                self.sf_vec_size,
                self.Q_stage,
            )
            self.sSFDO_dV_layout = blockscaled_utils.make_smem_layout_sfb(
                self.tiled_mma_dV_bs,
                self.mma_tiler_pdo,
                self.sf_vec_size,
                self.dO_stage,
            )
            self.sSFDS_dQ_layout = blockscaled_utils.make_smem_layout_sfa(
                self.tiled_mma_dQ_bs,
                self.mma_tiler_dsk,
                self.sf_vec_size,
                2,
            )
            self.sSFK_dQ_layout = blockscaled_utils.make_smem_layout_sfb(
                self.tiled_mma_dQ_bs,
                self.mma_tiler_dsk,
                self.sf_vec_size,
                1,
            )
        else:
            self.sSFP_layout = None
            self.sSFDS_layout = None
            self.sSFQ_dK_layout = None
            self.sSFDO_dV_layout = None
            self.sSFDS_dQ_layout = None
            self.sSFK_dQ_layout = None

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (self.tiled_mma_S.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = (
            cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk),
                (self.tiled_mma_S_sfb.thr_id.shape,),
            )
            if const_expr(self.blockscaled)
            else None
        )
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_q_do_mcast = self.num_mcast_ctas_b > 1

        if const_expr(not self.dKV_postprocess):
            self.mdK_layout_enum = LayoutEnum.from_tensor(mdK)
            self.mdV_layout_enum = LayoutEnum.from_tensor(mdV)
            dK_major_mode = self.mdK_layout_enum.mma_major_mode()
            dV_major_mode = self.mdV_layout_enum.mma_major_mode()
            if const_expr(dK_major_mode != tcgen05.OperandMajorMode.K):
                raise RuntimeError("The layout of mdK is wrong")
            if const_expr(dV_major_mode != tcgen05.OperandMajorMode.K):
                raise RuntimeError("The layout of mdV is wrong")

        if const_expr(
            (self.use_tma_store or self.use_dedicated_mxfp8_dkv_tma)
            and not self.dKV_postprocess
        ):
            tma_copy_op_dKV = cpasync.CopyBulkTensorTileS2GOp()
            tma_atom_dK, mdK_tma_tensor = cpasync.make_tiled_tma_atom(
                tma_copy_op_dKV,
                mdK,
                cute.select(self.sdK_layout, mode=[0, 1]),
                self.sdK_epi_tile,
                1,  # no mcast
            )
            tma_atom_dV, mdV_tma_tensor = cpasync.make_tiled_tma_atom(
                tma_copy_op_dKV,
                mdV,
                cute.select(self.sdV_layout, mode=[0, 1]),
                self.sdV_epi_tile,
                1,  # no mcast
            )
        else:
            mdV_tma_tensor = mdV
            mdK_tma_tensor = mdK
            tma_atom_dV = None
            tma_atom_dK = None

        if const_expr(
            self.use_external_mxfp8_scales
            and not self.use_2cta_instrs
            and not self.broadcast_q
        ):
            assert mdQaccumTma is not None
            tma_atom_dQ, mdQaccum_tma_tensor = cpasync.make_tiled_tma_atom(
                cpasync.CopyReduceBulkTensorTileS2GOp(),
                mdQaccumTma,
                cute.select(self.sdQaccum_tma_layout, mode=[0, 1]),
                self.sdQaccum_tma_tile,
                1,
            )
        else:
            tma_atom_dQ = None
            mdQaccum_tma_tensor = None


        if const_expr(not self.dKV_postprocess):
            thr_layout_r2s_dKV = cute.make_ordered_layout((128, 1), order=(1, 0))  # 128 threads
            val_layout_r2s_dKV = cute.make_ordered_layout(
                (1, 128 // self.dk_dtype.width), order=(1, 0)
            )  # 4 or 8 vals for 16 byte store
            copy_atom_r2s_dKV = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dk_dtype,
                num_bits_per_copy=128,
            )
            tiled_copy_r2s_dKV = cute.make_tiled_copy_tv(
                copy_atom_r2s_dKV, thr_layout_r2s_dKV, val_layout_r2s_dKV
            )
        else:
            tiled_copy_r2s_dKV = copy_utils.tiled_copy_1d(
                Float32, 128, num_copy_elems=128 // Float32.width
            )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        # S.T = K @ Q.T
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mK,
            cute.select(self.sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_S,
            self.cluster_layout_vmnk.shape,
        )
        Q_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma_S.thr_id
        )
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            Q_tma_op,
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_S,
            self.cluster_layout_vmnk.shape,
        )
        # dP.T = V @ dO.T
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mV,
            cute.select(self.sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_vdo,
            self.tiled_mma_dP,
            self.cluster_layout_vmnk.shape,
        )
        # dV = P.T @ dO
        dO_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma_dV.thr_id
        )
        mdO_dV_operand = mdO_dV if const_expr(mdO_dV is not None) else mdO
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
            dO_tma_op,
            mdO_dV_operand,
            cute.select(self.sdO_layout, mode=[0, 1, 2]),
            self.mma_tiler_pdo,
            self.tiled_mma_dV,
            self.cluster_layout_vmnk.shape,
        )
        # ------------------------------------------------------------
        # 2-CTA
        # ------------------------------------------------------------
        tma_atom_dOt = tma_tensor_dOt = None
        if const_expr(
            (self.use_2cta_instrs or self.use_external_mxfp8_scales)
            and not self.single_do_payload
        ):
            tma_atom_dOt, tma_tensor_dOt = cute.nvgpu.make_tiled_tma_atom_B(
                dO_tma_op,
                layout_utils.select(mdO, mode=transpose_sh_q),
                cute.select(self.sdOt_layout, mode=[0, 1, 2]),
                self.mma_tiler_vdo,
                self.tiled_mma_dP,
                self.cluster_layout_vmnk.shape,
            )
        tma_atom_Qt = tma_tensor_Qt = None
        if const_expr(self.use_2cta_instrs or self.use_external_mxfp8_scales):
            mQ_dK_operand = mQ_dK if const_expr(mQ_dK is not None) else mQ
            tma_atom_Qt, tma_tensor_Qt = cute.nvgpu.make_tiled_tma_atom_B(
                Q_tma_op,
                layout_utils.select(mQ_dK_operand, mode=transpose_sh_q),
                cute.select(self.sQt_layout, mode=[0, 1, 2]),
                self.mma_tiler_dsq,
                self.tiled_mma_dK,
                self.cluster_layout_vmnk.shape,
            )
        tma_atom_Kt = tma_tensor_Kt = None
        if const_expr(self.use_2cta_instrs or self.use_external_mxfp8_scales):
            mK_dQ_operand = mK_dQ if const_expr(mK_dQ is not None) else mK
            tiled_mma_Kt = (
                self.tiled_mma_dQ_bs
                if const_expr(self.use_external_mxfp8_scales)
                else self.tiled_mma_dQ
            )
            sKt_tma_layout = (
                self.sKt_dQ_data_layout
                if const_expr(self.use_external_mxfp8_scales)
                else self.sKt_layout
            )
            if const_expr(self.use_external_mxfp8_scales):
                tma_atom_Kt, tma_tensor_Kt = cute.nvgpu.make_tiled_tma_atom_B(
                    tma_load_op,
                    layout_utils.select(mK_dQ_operand, mode=transpose_sh_k),
                    cute.select(sKt_tma_layout, mode=[0, 1, 2]),
                    self.mma_tiler_dsk,
                    tiled_mma_Kt,
                )
            else:
                Kt_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
                    self.cluster_shape_mnk, tiled_mma_Kt.thr_id
                )
                tma_atom_Kt, tma_tensor_Kt = cute.nvgpu.make_tiled_tma_atom_B(
                    Kt_tma_op,
                    layout_utils.select(mK_dQ_operand, mode=transpose_sh_k),
                    cute.select(sKt_tma_layout, mode=[0, 1, 2]),
                    self.mma_tiler_dsk,
                    tiled_mma_Kt,
                    self.cluster_layout_vmnk.shape,
                )

        tma_atom_SFQ = tma_tensor_SFQ = None
        tma_atom_SFK = tma_tensor_SFK = None
        tma_atom_SFV = tma_tensor_SFV = None
        tma_atom_SFDO = tma_tensor_SFDO = None
        tma_atom_SFQ_dK = tma_tensor_SFQ_dK = None
        tma_atom_SFDO_dV = tma_tensor_SFDO_dV = None
        tma_atom_SFK_dQ = tma_tensor_SFK_dQ = None
        if const_expr(self.blockscaled and self.use_external_mxfp8_scales):
            sfq_layout = blockscaled_utils.tile_atom_to_shape_SF(
                (total_sf_q, self.tile_hdim, mQ.shape[2]), self.sf_vec_size
            )
            mSFQ_tma = cute.make_tensor(mSFQ.iterator, sfq_layout)
            sfq_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_SFB(
                self.cluster_shape_mnk, self.tiled_mma_S_bs.thr_id
            )
            tma_atom_SFQ, tma_tensor_SFQ = cute.nvgpu.make_tiled_tma_atom_B(
                sfq_tma_op,
                mSFQ_tma,
                cute.select(self.sSFQ_layout, mode=[0, 1, 2]),
                self.mma_tiler_kq_sfb,
                self.tiled_mma_S_sfb,
                self.cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Int16,
            )

            sfk_layout = blockscaled_utils.tile_atom_to_shape_SF(
                (total_sf_k, self.tile_hdim, mK.shape[2]), self.sf_vec_size
            )
            mSFK_tma = cute.make_tensor(mSFK.iterator, sfk_layout)
            tma_atom_SFK, tma_tensor_SFK = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                mSFK_tma,
                cute.select(self.sSFK_layout, mode=[0, 1, 2]),
                self.mma_tiler_kq,
                self.tiled_mma_S_bs,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )

            sfv_layout = blockscaled_utils.tile_atom_to_shape_SF(
                (total_sf_k, self.tile_hdimv, mV.shape[2]), self.sf_vec_size
            )
            mSFV_tma = cute.make_tensor(mSFV.iterator, sfv_layout)
            tma_atom_SFV, tma_tensor_SFV = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                mSFV_tma,
                cute.select(self.sSFV_layout, mode=[0, 1, 2]),
                self.mma_tiler_vdo,
                self.tiled_mma_dP_bs,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )

            total_sf_do = (
                mdO.shape[1] if const_expr(self.broadcast_q) else total_sf_q
            )
            sfdo_layout = blockscaled_utils.tile_atom_to_shape_SF(
                (self.tile_hdimv, total_sf_do, mdO.shape[2]), self.sf_vec_size
            )
            mSFDO_tma = cute.make_tensor(mSFDO.iterator, sfdo_layout)
            sfdo_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_SFB(
                self.cluster_shape_mnk, self.tiled_mma_dP_bs.thr_id
            )
            tma_atom_SFDO, tma_tensor_SFDO = cute.nvgpu.make_tiled_tma_atom_B(
                sfdo_tma_op,
                mSFDO_tma,
                cute.select(self.sSFDO_layout, mode=[0, 1, 2]),
                self.mma_tiler_vdo_sfb,
                self.tiled_mma_dP_sfb,
                self.cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Int16,
            )

            sfq_dk_layout = blockscaled_utils.tile_atom_to_shape_SF(
                (mQ.shape[2], total_sf_q, self.tile_hdim), self.sf_vec_size
            )
            mSFQ_dK_tma = cute.make_tensor(mSFQ_dK.iterator, sfq_dk_layout)
            sfq_dk_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_SFB(
                self.cluster_shape_mnk, self.tiled_mma_dK_bs.thr_id
            )
            tma_atom_SFQ_dK, tma_tensor_SFQ_dK = cute.nvgpu.make_tiled_tma_atom_B(
                sfq_dk_tma_op,
                mSFQ_dK_tma,
                cute.select(self.sSFQ_dK_layout, mode=[0, 1, 2]),
                self.mma_tiler_dsq_sfb,
                self.tiled_mma_dK_sfb,
                self.cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Int16,
            )

            sfdo_dv_layout = blockscaled_utils.tile_atom_to_shape_SF(
                (mdO.shape[2], total_sf_do, self.tile_hdimv), self.sf_vec_size
            )
            mSFDO_dV_tma = cute.make_tensor(mSFDO_dV.iterator, sfdo_dv_layout)
            sfdo_dv_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_SFB(
                self.cluster_shape_mnk, self.tiled_mma_dV_bs.thr_id
            )
            tma_atom_SFDO_dV, tma_tensor_SFDO_dV = (
                cute.nvgpu.make_tiled_tma_atom_B(
                    sfdo_dv_tma_op,
                    mSFDO_dV_tma,
                    cute.select(self.sSFDO_dV_layout, mode=[0, 1, 2]),
                    self.mma_tiler_pdo_sfb,
                    self.tiled_mma_dV_sfb,
                    self.cluster_layout_sfb_vmnk.shape,
                    internal_type=cutlass.Int16,
                )
            )

            sfk_dq_layout = blockscaled_utils.tile_atom_to_shape_SF(
                (mK.shape[2], total_sf_k, self.tile_hdim), self.sf_vec_size
            )
            mSFK_dQ_tma = cute.make_tensor(mSFK_dQ.iterator, sfk_dq_layout)
            tma_atom_SFK_dQ, tma_tensor_SFK_dQ = (
                cute.nvgpu.make_tiled_tma_atom_B(
                    tma_load_op,
                    mSFK_dQ_tma,
                    cute.select(self.sSFK_dQ_layout, mode=[0, 1, 2]),
                    self.mma_tiler_dsk_sfb,
                    self.tiled_mma_dQ_sfb,
                    internal_type=cutlass.Int16,
                )
            )

        self.tma_copy_bytes = {
            name: self.cta_group_size
            * cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
                ("dO", mdO, self.sdO_layout),
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dQ"] = (
            self.tile_m * self.dQ_reduce_ncol * self.dqaccum_dtype.width // 8
        )
        self.tma_copy_bytes["dKacc"] = (
            self.tile_n * self.dK_reduce_ncol * Float32.width // 8
        )
        self.tma_copy_bytes["dS"] = cute.size_in_bytes(self.ds_dtype, self.sdS_layout)
        self.tma_copy_bytes["sdS_xchg"] = (
            self.tma_copy_bytes["dS"] // 2
        )  # Half of dS for exchange
        self.tma_copy_bytes["Qt"] = self.tma_copy_bytes["Q"]
        self.tma_copy_bytes["Kt"] = self.tma_copy_bytes["K"]
        self.tma_copy_bytes["dOt"] = (
            self.tma_copy_bytes["dO"] if const_expr(tma_atom_dOt is not None) else 0
        )
        if const_expr(self.blockscaled and self.use_external_mxfp8_scales):
            sf_copy_bytes = {
                name: cute.size_in_bytes(
                    self.sf_dtype, cute.select(layout, mode=[0, 1, 2])
                )
                for name, layout in [
                    ("Q", self.sSFQ_layout),
                    ("K", self.sSFK_layout),
                    ("V", self.sSFV_layout),
                    ("dOt", self.sSFDO_layout),
                    ("Qt", self.sSFQ_dK_layout),
                    ("dO", self.sSFDO_dV_layout),
                    ("Kt", self.sSFK_dQ_layout),
                ]
            }
            for name, num_bytes in sf_copy_bytes.items():
                self.tma_copy_bytes[name] += self.cta_group_size * num_bytes

        if const_expr(mTileToBatch is not None):
            assert mTileToHead is not None and mTileToBlock is not None
            TileScheduler = PersistentVarlenLookupScheduler
        elif const_expr(self.is_persistent and self.is_varlen_k):
            TileScheduler = PersistentVarlenTileScheduler
        elif const_expr(self.is_varlen_k):
            TileScheduler = SingleTileVarlenScheduler
        elif const_expr(self.deterministic):
            TileScheduler = SingleTileLPTBwdScheduler
        else:
            TileScheduler = SingleTileScheduler
        if const_expr(self.spt_override is None):
            self.spt = (self.is_causal or self.is_local) and self.deterministic
        else:
            assert self.spt_override is not None
            self.spt = self.spt_override and self.deterministic
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mK.shape[0]), self.cta_tiler[0]),  # num_blocks
            cute.size(mQ.shape[2]),  # num_heads = num_query_heads
            cute.size(mK.shape[3])
            if const_expr(mCuSeqlensK is None)
            else cute.size(mCuSeqlensK) - 1,  # num_batches
            1,  # num_splits
            cute.size(mK.shape[0]),  # seqlen_k or total K tokens
            mQ.shape[1],  # headdim
            mV.shape[1],  # headdim_v
            total_q=cute.size(mK.shape[0])  # pass total_k for total_q
            if const_expr(mCuSeqlensK is not None)
            else cute.size(mK.shape[0]) * cute.size(mK.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],  # (tile_n, tile_m)
            cluster_shape_mn=self.cluster_shape_mnk[:2],
            mCuSeqlensQ=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedK,
            qhead_per_kvhead_packgqa=1,  # pack_gqa disabled for bwd
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,
            lpt=self.spt,
            head_swizzle=self.deterministic,
            mTileToBatch=mTileToBatch,
            mTileToHead=mTileToHead,
            mTileToBlock=mTileToBlock,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        # Compute allocation sizes for shared buffers that are reused
        # sQ is reused for sdK, sdO is reused for sdV
        sQ_alloc_bytes = max(
            cute.size_in_bytes(self.q_dtype, self.sQ_layout),
            cute.size_in_bytes(self.dk_dtype, self.sdK_layout),
        )
        sdO_alloc_bytes = max(
            cute.size_in_bytes(self.dv_dtype, self.sdV_layout),
            cute.size_in_bytes(self.do_dtype, self.sdO_layout),
            cute.size_in_bytes(self.do_dtype, self.sdOt_layout)
            if const_expr(self.single_do_payload)
            else 0,
        )
        sSFQ_alloc_size = (
            cute.cosize(self.sSFQ_layout) if const_expr(self.blockscaled) else 0
        )
        sSFK_alloc_size = (
            cute.cosize(self.sSFK_layout) if const_expr(self.blockscaled) else 0
        )
        sSFV_alloc_size = (
            cute.cosize(self.sSFV_layout) if const_expr(self.blockscaled) else 0
        )
        sSFDO_alloc_size = (
            cute.cosize(self.sSFDO_layout) if const_expr(self.blockscaled) else 0
        )
        sSFP_alloc_size = (
            cute.cosize(self.sSFP_layout) if const_expr(self.blockscaled) else 0
        )
        sSFDS_alloc_size = (
            cute.cosize(self.sSFDS_layout) if const_expr(self.blockscaled) else 0
        )
        sSFQ_dK_alloc_size = (
            cute.cosize(self.sSFQ_dK_layout) if const_expr(self.blockscaled) else 0
        )
        sSFDO_dV_alloc_size = (
            cute.cosize(self.sSFDO_dV_layout) if const_expr(self.blockscaled) else 0
        )
        sSFDS_dQ_alloc_size = (
            cute.cosize(self.sSFDS_dQ_layout)
            if const_expr(self.blockscaled and not self.use_2cta_instrs)
            else 0
        )
        sSFK_dQ_alloc_size = (
            cute.cosize(self.sSFK_dQ_layout)
            if const_expr(self.blockscaled and not self.use_2cta_instrs)
            else 0
        )
        sf_dtype_alloc = self.sf_dtype if const_expr(self.blockscaled) else cute.Uint8

        sdK_bytes = cute.size_in_bytes(self.dk_dtype, self.sdK_layout)
        sdV_bytes = cute.size_in_bytes(self.dv_dtype, self.sdV_layout)
        sdK_mxfp8_tma_bytes = (
            sdK_bytes if const_expr(self.use_dedicated_mxfp8_dkv_tma) else 0
        )
        sdV_mxfp8_tma_bytes = (
            sdV_bytes if const_expr(self.use_dedicated_mxfp8_dkv_tma) else 0
        )
        if const_expr(self.use_dedicated_mxfp8_dkv_tma):
            assert sdK_mxfp8_tma_bytes == 16 * 1024
            assert sdV_mxfp8_tma_bytes == 16 * 1024
        assert sdV_bytes <= sdO_alloc_bytes, "sdV doesn't fit in sdO storage allocation"
        assert sdK_bytes <= sQ_alloc_bytes, "sdK doesn't fit in sQ storage allocation"
        # 2-CTA: sdV reuses sV, sdK reuses sK
        sV_bytes = cute.size_in_bytes(self.v_dtype, self.sV_layout)
        sK_bytes = cute.size_in_bytes(self.k_dtype, self.sK_layout)
        sV_2cta_alloc_bytes = max(sV_bytes, sdV_bytes)
        sK_2cta_alloc_bytes = max(sK_bytes, sdK_bytes)
        if const_expr(self.use_2cta_instrs or self.use_external_mxfp8_scales):
            assert sdV_bytes <= sV_2cta_alloc_bytes, (
                "sdV doesn't fit in sV storage allocation (2-CTA)"
            )
            assert sdK_bytes <= sK_2cta_alloc_bytes, (
                "sdK doesn't fit in sK storage allocation (2-CTA)"
            )

        if const_expr(self.use_2cta_instrs):
            sQt_size = cute.cosize(self.sQt_layout) if const_expr(self.tile_hdim <= 128) else 0
            sdOt_size = cute.cosize(self.sdOt_layout) if const_expr(self.tile_hdim <= 128) else 0
            sdS_xchg_size = (
                cute.cosize(self.sdS_xchg_layout) if const_expr(self.tile_hdim <= 128) else 0
            )
        if const_expr(self.use_2cta_instrs):

            @cute.struct
            class SharedStorage:
                Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
                LSE_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                dPsum_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
                S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                S_drain_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, 2 * self.single_stage
                ]
                dP_drain_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, 2 * self.single_stage
                ]
                dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dKV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.sdKVaccum_stage]
                dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
                dQ_cluster_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.dQaccum_reduce_stage // 2
                ]
                dQ_cluster_empty_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.dQaccum_reduce_stage // 2
                ]
                tmem_holding_buf: Int32
                tmem_dealloc_mbar_ptr: cutlass.Int64

                # 2-CTA
                Qt_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                Kt_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dS_cluster_empty_mbar_ptr: cutlass.Int64
                dS_cluster_full_mbar_ptr: cutlass.Int64
                dS_cluster_leader_mbar_ptr: cutlass.Int64
                dQaccum_empty_mbar_ptr: cutlass.Int64

                sQ: cute.struct.Align[
                    cute.struct.MemRange[self.q_dtype, cute.cosize(self.sQ_layout)],
                    self.buffer_align_bytes,
                ]
                sK: cute.struct.Align[
                    cute.struct.MemRange[cute.Uint8, sK_2cta_alloc_bytes],
                    self.buffer_align_bytes,
                ]
                sV: cute.struct.Align[
                    cute.struct.MemRange[cute.Uint8, sV_2cta_alloc_bytes],
                    self.buffer_align_bytes,
                ]
                sdO: cute.struct.Align[
                    cute.struct.MemRange[self.do_dtype, cute.cosize(self.sdO_layout)],
                    self.buffer_align_bytes,
                ]
                sQt: cute.struct.Align[
                    cute.struct.MemRange[self.q_dtype, sQt_size],
                    self.buffer_align_bytes,
                ]
                sdOt: cute.struct.Align[
                    cute.struct.MemRange[self.do_dtype, sdOt_size],
                    self.buffer_align_bytes,
                ]
                sdS_xchg: cute.struct.Align[
                    cute.struct.MemRange[self.ds_dtype, sdS_xchg_size],
                    self.buffer_align_bytes,
                ]
                sKt: cute.struct.Align[
                    cute.struct.MemRange[self.k_dtype, cute.cosize(self.sKt_layout)],
                    self.buffer_align_bytes,
                ]
                sdS: cute.struct.Align[
                    cute.struct.MemRange[self.ds_dtype, cute.cosize(self.sdSt_layout)],
                    self.buffer_align_bytes,
                ]
                sLSE: cute.struct.Align[
                    cute.struct.MemRange[self.lse_dtype, cute.cosize(self.sLSE_layout)],
                    128,
                ]
                sdPsum: cute.struct.Align[
                    cute.struct.MemRange[self.dpsum_dtype, cute.cosize(self.sdPsum_layout)],
                    128,
                ]
                sdQaccum: cute.struct.Align[
                    cute.struct.MemRange[self.dqaccum_dtype, cute.cosize(self.sdQaccum_layout)],
                    self.buffer_align_bytes if sdS_xchg_size == 0 else 128,
                ]
                sSFQ: cute.struct.Align[
                    cute.struct.MemRange[sf_dtype_alloc, sSFQ_alloc_size], 128
                ]
                sSFK: cute.struct.Align[
                    cute.struct.MemRange[sf_dtype_alloc, sSFK_alloc_size], 128
                ]
                sSFV: cute.struct.Align[
                    cute.struct.MemRange[sf_dtype_alloc, sSFV_alloc_size], 128
                ]
                sSFDO: cute.struct.Align[
                    cute.struct.MemRange[sf_dtype_alloc, sSFDO_alloc_size], 128
                ]
                sSFP: cute.struct.Align[
                    cute.struct.MemRange[sf_dtype_alloc, sSFP_alloc_size], 128
                ]
                sSFDS: cute.struct.Align[
                    cute.struct.MemRange[sf_dtype_alloc, sSFDS_alloc_size], 128
                ]
                sSFQ_dK: cute.struct.Align[
                    cute.struct.MemRange[sf_dtype_alloc, sSFQ_dK_alloc_size], 128
                ]
                sSFDO_dV: cute.struct.Align[
                    cute.struct.MemRange[sf_dtype_alloc, sSFDO_dV_alloc_size], 128
                ]
                sSFDS_dQ: cute.struct.Align[
                    cute.struct.MemRange[sf_dtype_alloc, sSFDS_dQ_alloc_size], 128
                ]
                sSFK_dQ: cute.struct.Align[
                    cute.struct.MemRange[sf_dtype_alloc, sSFK_dQ_alloc_size], 128
                ]
        else:

            @cute.struct
            class SharedStorage:
                Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                K_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64,
                    2
                    * self.single_stage
                    if const_expr(self.use_dedicated_k_pipeline)
                    else 0,
                ]
                dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
                LSE_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                dPsum_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
                S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                S_drain_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, 2 * self.single_stage
                ]
                dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]
                dKV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.sdKVaccum_stage]
                dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
                dQ_cluster_full_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.dQaccum_reduce_stage // 2
                ]
                dQ_cluster_empty_mbar_ptr: cute.struct.MemRange[
                    cutlass.Int64, self.dQaccum_reduce_stage // 2
                ]
                tmem_holding_buf: Int32
                tmem_dealloc_mbar_ptr: Int64
                Qt_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
                Kt_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.single_stage]

                sQ: cute.struct.Align[
                    cute.struct.MemRange[cute.Uint8, sQ_alloc_bytes],
                    self.buffer_align_bytes,
                ]
                sK: cute.struct.Align[
                    cute.struct.MemRange[self.k_dtype, cute.cosize(self.sK_layout)],
                    self.buffer_align_bytes,
                ]
                sQt: cute.struct.Align[
                    cute.struct.MemRange[
                        self.q_dtype,
                        0
                        if const_expr(not self.use_external_mxfp8_scales)
                        else cute.cosize(self.sQt_layout),
                    ],
                    self.buffer_align_bytes,
                ]
                sKt: cute.struct.Align[
                    cute.struct.MemRange[
                        self.k_dtype,
                        0
                        if const_expr(not self.use_external_mxfp8_scales)
                        else cute.cosize(self.sKt_layout),
                    ],
                    self.buffer_align_bytes,
                ]
                sV: cute.struct.Align[
                    cute.struct.MemRange[self.v_dtype, cute.cosize(self.sV_layout)],
                    self.buffer_align_bytes,
                ]
                sdO: cute.struct.Align[
                    cute.struct.MemRange[cute.Uint8, sdO_alloc_bytes],
                    self.buffer_align_bytes,
                ]
                sdOt: cute.struct.Align[
                    cute.struct.MemRange[
                        self.do_dtype,
                        0
                        if const_expr(
                            not self.use_external_mxfp8_scales
                            or self.single_do_payload
                        )
                        else cute.cosize(self.sdOt_layout),
                    ],
                    self.buffer_align_bytes,
                ]
                sdS: cute.struct.Align[
                    cute.struct.MemRange[self.ds_dtype, cute.cosize(self.sdSt_layout)],
                    128,
                ]
                sLSE: cute.struct.Align[
                    cute.struct.MemRange[self.lse_dtype, cute.cosize(self.sLSE_layout)],
                    128,
                ]
                sdPsum: cute.struct.Align[
                    cute.struct.MemRange[self.dpsum_dtype, cute.cosize(self.sdPsum_layout)],
                    128,
                ]
                sdQaccum: cute.struct.Align[
                    cute.struct.MemRange[self.dqaccum_dtype, cute.cosize(self.sdQaccum_layout)],
                    self.buffer_align_bytes,
                ]
                sSFQ: cute.struct.Align[
                    cute.struct.MemRange[sf_dtype_alloc, sSFQ_alloc_size], 128
                ]
                sSFK: cute.struct.Align[
                    cute.struct.MemRange[sf_dtype_alloc, sSFK_alloc_size], 128
                ]
                sSFV: cute.struct.Align[
                    cute.struct.MemRange[sf_dtype_alloc, sSFV_alloc_size], 128
                ]
                sSFDO: cute.struct.Align[
                    cute.struct.MemRange[sf_dtype_alloc, sSFDO_alloc_size], 128
                ]
                sSFP: cute.struct.Align[
                    cute.struct.MemRange[sf_dtype_alloc, sSFP_alloc_size], 128
                ]
                sSFDS: cute.struct.Align[
                    cute.struct.MemRange[sf_dtype_alloc, sSFDS_alloc_size], 128
                ]
                sSFQ_dK: cute.struct.Align[
                    cute.struct.MemRange[sf_dtype_alloc, sSFQ_dK_alloc_size], 128
                ]
                sSFDO_dV: cute.struct.Align[
                    cute.struct.MemRange[sf_dtype_alloc, sSFDO_dV_alloc_size], 128
                ]
                sSFDS_dQ: cute.struct.Align[
                    cute.struct.MemRange[sf_dtype_alloc, sSFDS_dQ_alloc_size], 128
                ]
                sSFK_dQ: cute.struct.Align[
                    cute.struct.MemRange[sf_dtype_alloc, sSFK_dQ_alloc_size], 128
                ]
                sdV_mxfp8_tma: cute.struct.Align[
                    cute.struct.MemRange[cute.Uint8, sdV_mxfp8_tma_bytes],
                    self.buffer_align_bytes,
                ]
                sdK_mxfp8_tma: cute.struct.Align[
                    cute.struct.MemRange[cute.Uint8, sdK_mxfp8_tma_bytes],
                    self.buffer_align_bytes,
                ]
        self.shared_storage = SharedStorage

        LOG2_E = math.log2(math.e)
        if const_expr(self.score_mod is None):
            # Without score_mod: bake scale into log2
            softmax_scale_log2 = softmax_scale * LOG2_E
        else:
            # With score_mod: score_mod applied to S * softmax_scale, then use LOG2_E only
            softmax_scale_log2 = LOG2_E

        if const_expr(window_size_left is not None):
            window_size_left = Int32(window_size_left)
        if const_expr(window_size_right is not None):
            window_size_right = Int32(window_size_right)

        fastdiv_mods = None
        if const_expr(aux_data is not None and aux_data.tensors is not None):
            seqlen_q = cute.size(mQ.shape[0]) // (
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            )
            seqlen_k = cute.size(mK.shape[0])
            seqlen_q_divmod = FastDivmodDivisor(seqlen_q)
            seqlen_k_divmod = FastDivmodDivisor(seqlen_k)
            fastdiv_mods = (seqlen_q_divmod, seqlen_k_divmod)
        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)

        if const_expr(self.use_2cta_instrs):
            assert blocksparse_tensors is None, (
                "2-CTA mode does not support block sparsity. "
                "Please create kernel with use_2cta_instrs=False for block sparse attention."
            )
        # 2-CTA: 231424 and 1-CTA: 232448
        # print("SMEM: ", self.shared_storage.size_in_bytes())
        if const_expr(
            self.use_block_sparsity
            or (aux_data is not None and aux_data.tensors is not None)
        ):
            assert all(
                x is None for x in (mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK)
            ), "Variable sequence length is not supported yet for aux tensors in bwd"

        self.kernel(
            tma_tensor_Q,
            tma_tensor_Qt,
            tma_tensor_K,
            tma_tensor_Kt,
            tma_tensor_V,
            mLSE,
            mdPsum,
            tma_tensor_dO,
            tma_tensor_dOt,
            mdV,
            mdK,
            mdSFV_out,
            mdSFK_out,
            mdQaccum,
            mdQaccum_tma_tensor,
            mdV_tma_tensor,
            mdK_tma_tensor,
            mdQ_semaphore,
            mdK_semaphore,
            mdV_semaphore,
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuSeqlensSFQ,
            mCuSeqlensSFK,
            mSeqUsedQ,
            mSeqUsedK,
            tma_atom_Q,
            tma_atom_Qt,
            tma_atom_K,
            tma_atom_Kt,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dOt,
            tma_atom_dV,
            tma_atom_dK,
            tma_atom_dQ,
            tma_atom_SFQ,
            tma_tensor_SFQ,
            tma_atom_SFK,
            tma_tensor_SFK,
            tma_atom_SFV,
            tma_tensor_SFV,
            tma_atom_SFDO,
            tma_tensor_SFDO,
            tma_atom_SFQ_dK,
            tma_tensor_SFQ_dK,
            tma_atom_SFDO_dV,
            tma_tensor_SFDO_dV,
            tma_atom_SFK_dQ,
            tma_tensor_SFK_dQ,
            self.sQ_layout,
            self.sQt_layout,
            self.sK_layout,
            self.sKt_layout,
            self.sV_layout,
            self.sLSE_layout,
            self.sdPsum_layout,
            self.sdO_layout,
            self.sdOt_layout,
            self.sdSt_layout,
            self.sdS_layout,
            self.sdS_xchg_layout,
            self.sdQaccum_layout,
            self.sdQaccum_tma_layout,
            self.sdK_layout,
            self.sdV_layout,
            self.tP_layout,
            self.tdS_layout,
            self.sdS_dQ_data_layout if const_expr(self.blockscaled) else None,
            self.sKt_dQ_data_layout if const_expr(self.blockscaled) else None,
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dV,
            self.tiled_mma_dK,
            self.tiled_mma_dQ,
            tiled_copy_r2s_dKV,
            softmax_scale,
            dq_accum_scale,
            softmax_scale_log2,
            window_size_left,
            window_size_right,
            tile_sched_params,
            self.sSFQ_layout if const_expr(self.blockscaled) else None,
            self.sSFK_layout if const_expr(self.blockscaled) else None,
            self.sSFV_layout if const_expr(self.blockscaled) else None,
            self.sSFDO_layout if const_expr(self.blockscaled) else None,
            self.sSFP_layout if const_expr(self.blockscaled) else None,
            self.sSFDS_layout if const_expr(self.blockscaled) else None,
            self.sSFQ_dK_layout if const_expr(self.blockscaled) else None,
            self.sSFDO_dV_layout if const_expr(self.blockscaled) else None,
            self.sSFDS_dQ_layout if const_expr(self.blockscaled) else None,
            self.sSFK_dQ_layout if const_expr(self.blockscaled) else None,
            self.tiled_mma_S_bs if const_expr(self.blockscaled) else None,
            self.tiled_mma_dP_bs if const_expr(self.blockscaled) else None,
            self.tiled_mma_dV_bs if const_expr(self.blockscaled) else None,
            self.tiled_mma_dK_bs if const_expr(self.blockscaled) else None,
            self.tiled_mma_dQ_bs if const_expr(self.blockscaled) else None,
            self.tiled_mma_S_sfb if const_expr(self.blockscaled) else None,
            self.tiled_mma_dP_sfb if const_expr(self.blockscaled) else None,
            self.tiled_mma_dV_sfb if const_expr(self.blockscaled) else None,
            self.tiled_mma_dK_sfb if const_expr(self.blockscaled) else None,
            self.tiled_mma_dQ_sfb if const_expr(self.blockscaled) else None,
            aux_data,
            fastdiv_mods,
            blocksparse_tensors,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk if cute.size(self.cluster_shape_mnk) > 1 else None,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    def _generate_attention_mask_cls(self, window_size_left, window_size_right):
        return partial(
            AttentionMask,
            self.tile_m,
            self.tile_n * self.cta_group_size,
            swap_AB=True,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mQt: Optional[cute.Tensor],
        mK: cute.Tensor,
        mKt: Optional[cute.Tensor],
        mV: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdO: cute.Tensor,
        mdOt: Optional[cute.Tensor],
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        mdSFV_out: Optional[cute.Tensor],
        mdSFK_out: Optional[cute.Tensor],
        mdQaccum: cute.Tensor,
        mdQaccum_tma_tensor: Optional[cute.Tensor],
        mdV_tma_tensor: Optional[cute.Tensor],
        mdK_tma_tensor: Optional[cute.Tensor],
        mdQ_semaphore: Optional[cute.Tensor],
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mCuSeqlensSFQ: Optional[cute.Tensor],
        mCuSeqlensSFK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_Qt: Optional[cute.CopyAtom],
        tma_atom_K: cute.CopyAtom,
        tma_atom_Kt: Optional[cute.CopyAtom],
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dOt: Optional[cute.CopyAtom],
        tma_atom_dV: Optional[cute.CopyAtom],
        tma_atom_dK: Optional[cute.CopyAtom],
        tma_atom_dQ: Optional[cute.CopyAtom],
        tma_atom_SFQ: Optional[cute.CopyAtom],
        mSFQ: Optional[cute.Tensor],
        tma_atom_SFK: Optional[cute.CopyAtom],
        mSFK: Optional[cute.Tensor],
        tma_atom_SFV: Optional[cute.CopyAtom],
        mSFV: Optional[cute.Tensor],
        tma_atom_SFDO: Optional[cute.CopyAtom],
        mSFDO: Optional[cute.Tensor],
        tma_atom_SFQ_dK: Optional[cute.CopyAtom],
        mSFQ_dK: Optional[cute.Tensor],
        tma_atom_SFDO_dV: Optional[cute.CopyAtom],
        mSFDO_dV: Optional[cute.Tensor],
        tma_atom_SFK_dQ: Optional[cute.CopyAtom],
        mSFK_dQ: Optional[cute.Tensor],
        sQ_layout: cute.ComposedLayout,
        sQt_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sKt_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        sdPsum_layout: cute.Layout,
        sdO_layout: cute.ComposedLayout,
        sdOt_layout: cute.ComposedLayout,
        sdSt_layout: cute.ComposedLayout,
        sdS_layout: cute.ComposedLayout,
        sdS_xchg_layout: cute.Layout,
        sdQaccum_layout: cute.Layout,
        sdQaccum_tma_layout: cute.Layout,
        sdK_layout: cute.ComposedLayout | cute.Layout,
        sdV_layout: cute.ComposedLayout | cute.Layout,
        tP_layout: cute.ComposedLayout,
        tdS_layout: cute.ComposedLayout,
        sdS_dQ_data_layout: Optional[cute.ComposedLayout],
        sKt_dQ_data_layout: Optional[cute.ComposedLayout],
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        tiled_copy_r2s_dKV: cute.TiledCopy,
        softmax_scale: cutlass.Float32,
        dq_accum_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        tile_sched_params: ParamsBase,
        sSFQ_layout: Optional[cute.Layout],
        sSFK_layout: Optional[cute.Layout],
        sSFV_layout: Optional[cute.Layout],
        sSFDO_layout: Optional[cute.Layout],
        sSFP_layout: Optional[cute.Layout],
        sSFDS_layout: Optional[cute.Layout],
        sSFQ_dK_layout: Optional[cute.Layout],
        sSFDO_dV_layout: Optional[cute.Layout],
        sSFDS_dQ_layout: Optional[cute.Layout],
        sSFK_dQ_layout: Optional[cute.Layout],
        tiled_mma_S_bs: Optional[cute.TiledMma],
        tiled_mma_dP_bs: Optional[cute.TiledMma],
        tiled_mma_dV_bs: Optional[cute.TiledMma],
        tiled_mma_dK_bs: Optional[cute.TiledMma],
        tiled_mma_dQ_bs: Optional[cute.TiledMma],
        tiled_mma_S_sfb: Optional[cute.TiledMma],
        tiled_mma_dP_sfb: Optional[cute.TiledMma],
        tiled_mma_dV_sfb: Optional[cute.TiledMma],
        tiled_mma_dK_sfb: Optional[cute.TiledMma],
        tiled_mma_dQ_sfb: Optional[cute.TiledMma],
        aux_data: Optional[AuxData] = None,
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % self.cta_group_size
        is_leader_cta = mma_tile_coord_v == 0

        # Prefetch tma descriptor
        if warp_idx == self.load_warp_id:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_Q)
                if const_expr(tma_atom_Qt is not None):
                    cpasync.prefetch_descriptor(tma_atom_Qt)
                cpasync.prefetch_descriptor(tma_atom_K)
                if const_expr(tma_atom_Kt is not None):
                    cpasync.prefetch_descriptor(tma_atom_Kt)
                cpasync.prefetch_descriptor(tma_atom_V)
                if const_expr(tma_atom_dOt is not None):
                    cpasync.prefetch_descriptor(tma_atom_dOt)
                cpasync.prefetch_descriptor(tma_atom_dO)
                if const_expr(tma_atom_dV is not None):
                    cpasync.prefetch_descriptor(tma_atom_dV)
                if const_expr(tma_atom_dK is not None):
                    cpasync.prefetch_descriptor(tma_atom_dK)
                if const_expr(tma_atom_dQ is not None):
                    cpasync.prefetch_descriptor(tma_atom_dQ)
                if const_expr(tma_atom_SFQ is not None):
                    cpasync.prefetch_descriptor(tma_atom_SFQ)
                    cpasync.prefetch_descriptor(tma_atom_SFK)
                    cpasync.prefetch_descriptor(tma_atom_SFV)
                    cpasync.prefetch_descriptor(tma_atom_SFDO)
                    cpasync.prefetch_descriptor(tma_atom_SFQ_dK)
                    cpasync.prefetch_descriptor(tma_atom_SFDO_dV)
                    cpasync.prefetch_descriptor(tma_atom_SFK_dQ)

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_S.thr_id.shape,),
        )
        cluster_layout_sfb_vmnk = (
            cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk),
                (tiled_mma_S_sfb.thr_id.shape,),
            )
            if const_expr(self.blockscaled)
            else None
        )

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        dQ_cluster_full_mbar_ptr = storage.dQ_cluster_full_mbar_ptr.data_ptr()
        dQ_cluster_empty_mbar_ptr = storage.dQ_cluster_empty_mbar_ptr.data_ptr()

        if const_expr(self.use_2cta_instrs):
            dS_cluster_full_mbar_ptr = storage.dS_cluster_full_mbar_ptr
            dS_cluster_empty_mbar_ptr = storage.dS_cluster_empty_mbar_ptr
            dS_cluster_leader_mbar_ptr = storage.dS_cluster_leader_mbar_ptr
            dQaccum_empty_mbar_ptr = storage.dQaccum_empty_mbar_ptr
        else:
            dS_cluster_full_mbar_ptr = None
            dS_cluster_empty_mbar_ptr = None
            dS_cluster_leader_mbar_ptr = None
            dQaccum_empty_mbar_ptr = None

        # Barrier initialization
        if const_expr(self.use_2cta_instrs):
            if const_expr(self.tile_hdim == 192):
                if warp_idx == 2:
                    cute.arch.mbarrier_init(
                        dQaccum_empty_mbar_ptr,
                        len(self.reduce_warp_ids),
                    )
            if warp_idx == 4:
                cute.arch.mbarrier_init(dS_cluster_full_mbar_ptr, 1)
                cute.arch.mbarrier_init(dS_cluster_empty_mbar_ptr, 1)
                cute.arch.mbarrier_init(dS_cluster_leader_mbar_ptr, 2)

        if const_expr(self.cluster_reduce_dQ):
            if warp_idx == 4:
                for i in range(self.dQaccum_reduce_stage // 2):
                    cute.arch.mbarrier_init(dQ_cluster_full_mbar_ptr + i, 1)
                    cute.arch.mbarrier_init(dQ_cluster_empty_mbar_ptr + i, 1)

        tmem_alloc_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE
            * len((self.mma_warp_id, *self.compute_warp_ids, *self.reduce_warp_ids)),
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        # UMMA producers and AsyncThread consumers
        pipeline_producer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        pipeline_consumer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids) * self.cta_group_size
        )
        pipeline_consumer_group_MMA_AsyncThread_dQ = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.reduce_warp_ids) * self.cta_group_size,
        )  # Compute

        # AsyncThread producers and UMMA consumers
        # Only 1 thread per warp will signal
        pipeline_PdS_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.compute_warp_ids) * self.cta_group_size,
        )  # Compute
        pipeline_PdS_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )  # MMA
        if const_expr(self.use_2cta_instrs):
            pipeline_S_P = cutlass.pipeline.PipelineUmmaAsync.create(
                num_stages=1,
                producer_group=pipeline_producer_group_MMA_AsyncThread,
                consumer_group=pipeline_consumer_group_MMA_AsyncThread,
                barrier_storage=storage.S_mbar_ptr.data_ptr(),
                cta_layout_vmnk=cluster_layout_vmnk,
            )
            pipeline_S_drain = cutlass.pipeline.PipelineUmmaAsync.create(
                num_stages=1,
                producer_group=pipeline_producer_group_MMA_AsyncThread,
                consumer_group=pipeline_consumer_group_MMA_AsyncThread,
                barrier_storage=storage.S_drain_mbar_ptr.data_ptr(),
                cta_layout_vmnk=cluster_layout_vmnk,
            )
            pipeline_dP = cutlass.pipeline.PipelineUmmaAsync.create(
                num_stages=1,
                producer_group=pipeline_producer_group_MMA_AsyncThread,
                consumer_group=pipeline_consumer_group_MMA_AsyncThread,
                barrier_storage=storage.dP_mbar_ptr.data_ptr(),
                cta_layout_vmnk=cluster_layout_vmnk,
            )
            if const_expr(self.blockscaled and self.tile_hdim == 128):
                pipeline_dP_drain = cutlass.pipeline.PipelineUmmaAsync.create(
                    num_stages=1,
                    producer_group=pipeline_producer_group_MMA_AsyncThread,
                    consumer_group=pipeline_consumer_group_MMA_AsyncThread,
                    barrier_storage=storage.dP_drain_mbar_ptr.data_ptr(),
                    cta_layout_vmnk=cluster_layout_vmnk,
                )
            else:
                pipeline_dP_drain = pipeline_dP
            pipeline_dKV = cutlass.pipeline.PipelineUmmaAsync.create(
                num_stages=2,
                producer_group=pipeline_producer_group_MMA_AsyncThread,
                consumer_group=pipeline_consumer_group_MMA_AsyncThread,
                barrier_storage=storage.dKV_mbar_ptr.data_ptr(),
                cta_layout_vmnk=cluster_layout_vmnk,
            )
            pipeline_dQ = cutlass.pipeline.PipelineUmmaAsync.create(
                num_stages=1,
                producer_group=pipeline_producer_group_MMA_AsyncThread,
                consumer_group=pipeline_consumer_group_MMA_AsyncThread_dQ,
                barrier_storage=storage.dQ_mbar_ptr.data_ptr(),
                cta_layout_vmnk=cluster_layout_vmnk,
            )
            pipeline_dS = cutlass.pipeline.PipelineAsyncUmma.create(
                num_stages=1,
                producer_group=pipeline_PdS_producer_group,
                consumer_group=pipeline_PdS_consumer_group,
                barrier_storage=storage.dS_mbar_ptr.data_ptr(),
                cta_layout_vmnk=cluster_layout_vmnk,
            )
        else:
            pipeline_S_P = cutlass.pipeline.PipelineUmmaAsync.create(
                num_stages=1,
                producer_group=pipeline_producer_group_MMA_AsyncThread,
                consumer_group=pipeline_consumer_group_MMA_AsyncThread,
                barrier_storage=storage.S_mbar_ptr.data_ptr(),
            )
            pipeline_S_drain = cutlass.pipeline.PipelineUmmaAsync.create(
                num_stages=1,
                producer_group=pipeline_producer_group_MMA_AsyncThread,
                consumer_group=pipeline_consumer_group_MMA_AsyncThread,
                barrier_storage=storage.S_drain_mbar_ptr.data_ptr(),
            )
            pipeline_dP = cutlass.pipeline.PipelineUmmaAsync.create(
                num_stages=1,
                producer_group=pipeline_producer_group_MMA_AsyncThread,
                consumer_group=pipeline_consumer_group_MMA_AsyncThread,
                barrier_storage=storage.dP_mbar_ptr.data_ptr(),
            )
            pipeline_dP_drain = pipeline_dP
            pipeline_dKV = cutlass.pipeline.PipelineUmmaAsync.create(
                num_stages=2,
                producer_group=pipeline_producer_group_MMA_AsyncThread,
                consumer_group=pipeline_consumer_group_MMA_AsyncThread,
                barrier_storage=storage.dKV_mbar_ptr.data_ptr(),
            )
            pipeline_dQ = cutlass.pipeline.PipelineUmmaAsync.create(
                num_stages=1,
                producer_group=pipeline_producer_group_MMA_AsyncThread,
                consumer_group=pipeline_consumer_group_MMA_AsyncThread_dQ,
                barrier_storage=storage.dQ_mbar_ptr.data_ptr(),
            )
            pipeline_dS = cutlass.pipeline.PipelineAsyncUmma.create(
                num_stages=1,
                producer_group=pipeline_PdS_producer_group,
                consumer_group=pipeline_PdS_consumer_group,
                barrier_storage=storage.dS_mbar_ptr.data_ptr(),
            )

        # TMA producer and UMMA consumers
        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.load_warp_id])
        )
        # The arrive count is the number of mcast size
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id]) * self.num_mcast_ctas_b
        )
        pipeline_consumer_group_compute = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.compute_warp_ids) * 1,
        )
        pipeline_LSE = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.LSE_mbar_ptr.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group_compute,
            tx_count=self.tma_copy_bytes["LSE"],
            # cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_dPsum = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.dPsum_mbar_ptr.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group_compute,
            tx_count=self.tma_copy_bytes["dPsum"],
            # cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_Q = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.Q_mbar_ptr.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        if const_expr(self.use_dedicated_k_pipeline):
            pipeline_K = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.K_mbar_ptr.data_ptr(),
                num_stages=self.single_stage,
                producer_group=pipeline_producer_group,
                consumer_group=pipeline_consumer_group,
                tx_count=self.tma_copy_bytes["K"],
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
        else:
            pipeline_K = pipeline_Q

        if const_expr(self.use_2cta_instrs or self.use_external_mxfp8_scales):
            if const_expr(self.tile_hdim == 192):
                pipeline_Qt = pipeline_Q
            else:
                pipeline_Qt = pipeline.PipelineTmaUmma.create(
                    barrier_storage=storage.Qt_mbar_ptr.data_ptr(),
                    num_stages=self.Q_stage,
                    producer_group=pipeline_producer_group,
                    consumer_group=pipeline_consumer_group,
                    tx_count=self.tma_copy_bytes["Qt"],
                    cta_layout_vmnk=cluster_layout_vmnk,
                    defer_sync=True,
                )
            pipeline_Kt = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.Kt_mbar_ptr.data_ptr(),
                num_stages=self.single_stage,
                producer_group=pipeline_producer_group,
                consumer_group=pipeline_consumer_group,
                tx_count=self.tma_copy_bytes["Kt"],
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
        else:
            pipeline_Qt = pipeline_Kt = pipeline_Q

        pipeline_dO = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.dO_mbar_ptr.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["dO"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=False,
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner, dtype=self.q_dtype)
        if const_expr(
            (self.use_2cta_instrs and self.tile_hdim <= 128)
            or self.use_external_mxfp8_scales
        ):
            sQt = storage.sQt.get_tensor(
                sQt_layout.outer, swizzle=sQt_layout.inner, dtype=self.q_dtype
            )
        else:
            sQt = cute.make_tensor(
                cute.recast_ptr(sQ.iterator, sQt_layout.inner, dtype=self.q_dtype), sQt_layout.outer
            )
        sK = storage.sK.get_tensor(
            sK_layout.outer, swizzle=sK_layout.inner, dtype=self.k_dtype
        )
        if const_expr(self.use_2cta_instrs or self.use_external_mxfp8_scales):
            sKt_active_layout = (
                sKt_dQ_data_layout
                if const_expr(self.use_external_mxfp8_scales)
                else sKt_layout
            )
            sKt = storage.sKt.get_tensor(
                sKt_active_layout.outer, swizzle=sKt_active_layout.inner
            )
        else:
            sKt = cute.make_tensor(
                cute.recast_ptr(sK.iterator, sKt_layout.inner), sKt_layout.outer
            )
        sV = storage.sV.get_tensor(
            sV_layout.outer, swizzle=sV_layout.inner, dtype=self.v_dtype
        )
        sdSt = storage.sdS.get_tensor(sdSt_layout.outer, swizzle=sdSt_layout.inner)
        sdS_active_layout = (
            sdS_dQ_data_layout
            if const_expr(self.use_external_mxfp8_scales)
            else sdS_layout
        )
        sdS = cute.make_tensor(
            cute.recast_ptr(sdSt.iterator, sdS_active_layout.inner),
            sdS_active_layout.outer,
        )
        if const_expr(self.use_2cta_instrs):
            if const_expr(self.tile_hdim <= 128):
                sdS_xchg = storage.sdS_xchg.get_tensor(sdS_xchg_layout)
            else:
                sdS_xchg = storage.sdQaccum.get_tensor(sdS_xchg_layout, dtype=self.ds_dtype)
        else:
            sdS_xchg = None

        sdO = storage.sdO.get_tensor(
            sdO_layout.outer, swizzle=sdO_layout.inner, dtype=self.do_dtype
        )
        if const_expr(
            (self.use_2cta_instrs and self.tile_hdim <= 128)
            or (
                self.use_external_mxfp8_scales and not self.single_do_payload
            )
        ):
            sdOt = storage.sdOt.get_tensor(
                sdOt_layout.outer, swizzle=sdOt_layout.inner, dtype=self.do_dtype
            )
        else:
            sdOt = cute.make_tensor(
                cute.recast_ptr(sdO.iterator, sdOt_layout.inner, dtype=self.do_dtype),
                sdOt_layout.outer,
            )

        sLSE = storage.sLSE.get_tensor(sLSE_layout)
        sdPsum = storage.sdPsum.get_tensor(sdPsum_layout)
        if const_expr(self.use_dedicated_mxfp8_dkv_tma):
            sdV = storage.sdV_mxfp8_tma.get_tensor(
                sdV_layout.outer, swizzle=sdV_layout.inner, dtype=self.dv_dtype
            )
            sdK = storage.sdK_mxfp8_tma.get_tensor(
                sdK_layout.outer, swizzle=sdK_layout.inner, dtype=self.dk_dtype
            )
        elif const_expr(self.use_2cta_instrs):
            if const_expr(not self.dKV_postprocess):
                sdV = storage.sV.get_tensor(
                    sdV_layout.outer, swizzle=sdV_layout.inner, dtype=self.dv_dtype
                )
                sdK = storage.sK.get_tensor(
                    sdK_layout.outer, swizzle=sdK_layout.inner, dtype=self.dk_dtype
                )
            else:
                sdV = storage.sV.get_tensor(sdV_layout, dtype=self.dv_dtype)
                sdK = storage.sK.get_tensor(sdK_layout, dtype=self.dk_dtype)
        elif const_expr(not self.dKV_postprocess):
            sdV = storage.sdO.get_tensor(
                sdV_layout.outer, swizzle=sdV_layout.inner, dtype=self.dv_dtype
            )
            sdK = storage.sQ.get_tensor(
                sdK_layout.outer, swizzle=sdK_layout.inner, dtype=self.dk_dtype
            )
        else:
            sdV = storage.sdO.get_tensor(sdV_layout, dtype=self.dv_dtype)
            sdK = storage.sQ.get_tensor(sdK_layout, dtype=self.dk_dtype)

        # Buffer sizing is guaranteed by max(...) in SharedStorage declarations
        # for both sQ (reused as sdK) and sdO (reused as sdV)
        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)

        sSFQ = sSFK = sSFV = sSFDO = sSFP = sSFDS = None
        sSFQ_dK = sSFDO_dV = None
        sSFDS_dQ = sSFK_dQ = None
        sSFP_u32 = sSFDS_u32 = sSFDS_dQ_u32 = None
        if const_expr(self.blockscaled):
            sSFQ = storage.sSFQ.get_tensor(sSFQ_layout)
            sSFK = storage.sSFK.get_tensor(sSFK_layout)
            sSFV = storage.sSFV.get_tensor(sSFV_layout)
            sSFDO = storage.sSFDO.get_tensor(sSFDO_layout)
            sSFP = storage.sSFP.get_tensor(sSFP_layout)
            sSFDS = storage.sSFDS.get_tensor(sSFDS_layout)
            sSFQ_dK = storage.sSFQ_dK.get_tensor(sSFQ_dK_layout)
            sSFDO_dV = storage.sSFDO_dV.get_tensor(sSFDO_dV_layout)
            if const_expr(not self.use_2cta_instrs):
                sSFDS_dQ = storage.sSFDS_dQ.get_tensor(sSFDS_dQ_layout)
                sSFK_dQ = storage.sSFK_dQ.get_tensor(sSFK_dQ_layout)
        if const_expr(self.blockscaled):
            sSFP_u32 = FlashAttentionBackwardSm100._make_smem_u32_view(sSFP)
            sSFDS_u32 = FlashAttentionBackwardSm100._make_smem_u32_view(sSFDS)
            if const_expr(sSFDS_dQ is not None):
                sSFDS_dQ_u32 = FlashAttentionBackwardSm100._make_smem_u32_view(
                    sSFDS_dQ
                )
        # TMEM
        # This is a fake tensor, by right need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
        # S
        thr_mma_S = tiled_mma_S.get_slice(mma_tile_coord_v)
        thr_mma_S_sfb = (
            tiled_mma_S_sfb.get_slice(0)
            if const_expr(self.blockscaled)
            else None
        )
        Sacc_shape = thr_mma_S.partition_shape_C(self.mma_tiler_kq[:2])  # (M, N)
        tStS = thr_mma_S.make_fragment_C(Sacc_shape)
        # (MMA, MMA_M, MMA_N)
        tStS = cute.make_tensor(tmem_ptr + self.tmem_S_offset, tStS.layout)
        # dP
        thr_mma_dP = tiled_mma_dP.get_slice(mma_tile_coord_v)
        thr_mma_dP_sfb = (
            tiled_mma_dP_sfb.get_slice(0)
            if const_expr(self.blockscaled)
            else None
        )
        dPacc_shape = thr_mma_dP.partition_shape_C(self.mma_tiler_vdo[:2])
        tdPtdP = thr_mma_dP.make_fragment_C(dPacc_shape)
        tdPtdP = cute.make_tensor(tmem_ptr + self.tmem_dP_offset, tdPtdP.layout)
        # dV
        thr_mma_dV = tiled_mma_dV.get_slice(mma_tile_coord_v)
        thr_mma_dV_sfb = (
            tiled_mma_dV_sfb.get_slice(0)
            if const_expr(self.blockscaled)
            else None
        )
        dvacc_shape = thr_mma_dV.partition_shape_C(self.mma_tiler_pdo[:2])
        tdVtdV = thr_mma_dV.make_fragment_C(dvacc_shape)
        tdVtdV = cute.make_tensor(tmem_ptr + self.tmem_dV_offset, tdVtdV.layout)
        tP = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_P_offset, dtype=self.do_dtype), tP_layout.outer
        )
        # dK
        thr_mma_dK = tiled_mma_dK.get_slice(mma_tile_coord_v)
        thr_mma_dK_sfb = (
            tiled_mma_dK_sfb.get_slice(0)
            if const_expr(self.blockscaled)
            else None
        )
        dkacc_shape = thr_mma_dK.partition_shape_C(self.mma_tiler_dsq[:2])
        tdKtdK = thr_mma_dK.make_fragment_C(dkacc_shape)
        tdKtdK = cute.make_tensor(tmem_ptr + self.tmem_dK_offset, tdKtdK.layout)
        tdS = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_dS_offset, dtype=self.ds_dtype), tdS_layout.outer
        )
        # dQ
        thr_mma_dQ = (
            tiled_mma_dQ_bs.get_slice(0)
            if const_expr(self.use_external_mxfp8_scales)
            else tiled_mma_dQ.get_slice(mma_tile_coord_v)
        )
        thr_mma_dQ_sfb = (
            tiled_mma_dQ_sfb.get_slice(0)
            if const_expr(self.use_external_mxfp8_scales)
            else None
        )
        dQacc_shape = thr_mma_dQ.partition_shape_C(self.mma_tiler_dsk[:2])
        tdQtdQ = thr_mma_dQ.make_fragment_C(dQacc_shape)
        tdQtdQ = cute.make_tensor(tmem_ptr + self.tmem_dQ_offset, tdQtdQ.layout)

        tCtSFK = tCtSFQ = tCtSFV = tCtSFDO = None
        tCtSFK_prologue = tCtSFQ_prologue = None
        tCtSFV_prologue = tCtSFDO_prologue = None
        tCtSFP = tCtSFDO_dV = tCtSFDS = tCtSFQ_dK = None
        tCtSFDS_dQ = tCtSFK_dQ = None
        if const_expr(self.blockscaled):
            sSFK_stage_layout = cute.slice_(sSFK.layout, (None, None, None, 0))
            sSFQ_stage_layout = cute.slice_(sSFQ.layout, (None, None, None, 0))
            tCtSFK_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_S_bs,
                self.mma_tiler_kq,
                self.sf_vec_size,
                sSFK_stage_layout,
            )
            tCtSFQ_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_S_bs,
                self.mma_tiler_kq,
                self.sf_vec_size,
                sSFQ_stage_layout,
            )
            sfq_relative_offset = tcgen05.find_tmem_tensor_col_offset(
                cute.make_tensor(
                    cute.recast_ptr(tmem_ptr, dtype=self.sf_dtype), tCtSFK_layout
                )
            )
            tCtSFK = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + self.tmem_SF_offset, dtype=self.sf_dtype),
                tCtSFK_layout,
            )
            tCtSFQ = cute.make_tensor(
                cute.recast_ptr(
                    tmem_ptr + self.tmem_SF_offset + sfq_relative_offset,
                    dtype=self.sf_dtype,
                ),
                tCtSFQ_layout,
            )

            sSFV_stage_layout = cute.slice_(sSFV.layout, (None, None, None, 0))
            sSFDO_stage_layout = cute.slice_(sSFDO.layout, (None, None, None, 0))
            tCtSFV_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_dP_bs,
                self.mma_tiler_vdo,
                self.sf_vec_size,
                sSFV_stage_layout,
            )
            tCtSFDO_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_dP_bs,
                self.mma_tiler_vdo,
                self.sf_vec_size,
                sSFDO_stage_layout,
            )
            sfdo_relative_offset = tcgen05.find_tmem_tensor_col_offset(
                cute.make_tensor(
                    cute.recast_ptr(tmem_ptr, dtype=self.sf_dtype), tCtSFV_layout
                )
            )
            tCtSFV = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + self.tmem_SF_offset_dP, dtype=self.sf_dtype),
                tCtSFV_layout,
            )
            tCtSFDO = cute.make_tensor(
                cute.recast_ptr(
                    tmem_ptr + self.tmem_SF_offset_dP + sfdo_relative_offset,
                    dtype=self.sf_dtype,
                ),
                tCtSFDO_layout,
            )

            sf_prologue_ptr = cute.recast_ptr(
                tmem_ptr + self.tmem_SF_prologue_offset, dtype=self.sf_dtype
            )
            tCtSFK_prologue = cute.make_tensor(sf_prologue_ptr, tCtSFK_layout)
            tCtSFQ_prologue = cute.make_tensor(
                cute.recast_ptr(
                    tmem_ptr + self.tmem_SF_prologue_offset + sfq_relative_offset,
                    dtype=self.sf_dtype,
                ),
                tCtSFQ_layout,
            )
            tCtSFV_prologue = cute.make_tensor(sf_prologue_ptr, tCtSFV_layout)
            tCtSFDO_prologue = cute.make_tensor(
                cute.recast_ptr(
                    tmem_ptr + self.tmem_SF_prologue_offset + sfdo_relative_offset,
                    dtype=self.sf_dtype,
                ),
                tCtSFDO_layout,
            )

            sSFP_stage_layout = cute.slice_(sSFP.layout, (None, None, None, 0))
            sSFDO_dV_stage_layout = cute.slice_(sSFDO_dV.layout, (None, None, None, 0))
            tCtSFP_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_dV_bs,
                self.mma_tiler_pdo,
                self.sf_vec_size,
                sSFP_stage_layout,
            )
            tCtSFDO_dV_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_dV_bs,
                self.mma_tiler_pdo,
                self.sf_vec_size,
                sSFDO_dV_stage_layout,
            )
            sfdo_dv_relative_offset = tcgen05.find_tmem_tensor_col_offset(
                cute.make_tensor(
                    cute.recast_ptr(tmem_ptr, dtype=self.sf_dtype), tCtSFP_layout
                )
            )
            sfp_offset = self.tmem_S_offset + 56
            tCtSFP = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + sfp_offset, dtype=self.sf_dtype),
                tCtSFP_layout,
            )
            tCtSFDO_dV = cute.make_tensor(
                cute.recast_ptr(
                    tmem_ptr + sfp_offset + sfdo_dv_relative_offset,
                    dtype=self.sf_dtype,
                ),
                tCtSFDO_dV_layout,
            )

            sSFDS_stage_layout = cute.slice_(sSFDS.layout, (None, None, None, 0))
            sSFQ_dK_stage_layout = cute.slice_(sSFQ_dK.layout, (None, None, None, 0))
            tCtSFDS_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma_dK_bs,
                self.mma_tiler_dsq,
                self.sf_vec_size,
                sSFDS_stage_layout,
            )
            tCtSFQ_dK_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_dK_bs,
                self.mma_tiler_dsq,
                self.sf_vec_size,
                sSFQ_dK_stage_layout,
            )
            sfq_dk_relative_offset = tcgen05.find_tmem_tensor_col_offset(
                cute.make_tensor(
                    cute.recast_ptr(tmem_ptr, dtype=self.sf_dtype), tCtSFDS_layout
                )
            )
            sfds_offset = self.tmem_dP_offset + (64 if self.use_2cta_instrs else 80)
            tCtSFDS = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + sfds_offset, dtype=self.sf_dtype),
                tCtSFDS_layout,
            )
            tCtSFQ_dK = cute.make_tensor(
                cute.recast_ptr(
                    tmem_ptr + sfds_offset + sfq_dk_relative_offset,
                    dtype=self.sf_dtype,
                ),
                tCtSFQ_dK_layout,
            )

            if const_expr(not self.use_2cta_instrs):
                sSFDS_dQ_stage_layout = cute.slice_(
                    sSFDS_dQ.layout, (None, None, None, 0)
                )
                sSFK_dQ_stage_layout = cute.slice_(
                    sSFK_dQ.layout, (None, None, None, 0)
                )
                tCtSFDS_dQ_layout = blockscaled_utils.make_tmem_layout_sfa(
                    tiled_mma_dQ_bs,
                    self.mma_tiler_dsk,
                    self.sf_vec_size,
                    sSFDS_dQ_stage_layout,
                )
                tCtSFK_dQ_layout = blockscaled_utils.make_tmem_layout_sfb(
                    tiled_mma_dQ_bs,
                    self.mma_tiler_dsk,
                    self.sf_vec_size,
                    sSFK_dQ_stage_layout,
                )
                sfk_dq_relative_offset = tcgen05.find_tmem_tensor_col_offset(
                    cute.make_tensor(
                        cute.recast_ptr(tmem_ptr, dtype=self.sf_dtype),
                        tCtSFDS_dQ_layout,
                    )
                )
                sfds_dq_offset = self.tmem_S_offset + 80
                tCtSFDS_dQ = cute.make_tensor(
                    cute.recast_ptr(
                        tmem_ptr + sfds_dq_offset, dtype=self.sf_dtype
                    ),
                    tCtSFDS_dQ_layout,
                )
                tCtSFK_dQ = cute.make_tensor(
                    cute.recast_ptr(
                        tmem_ptr + sfds_dq_offset + sfk_dq_relative_offset,
                        dtype=self.sf_dtype,
                    ),
                    tCtSFK_dQ_layout,
                )

        block_info = BlockInfo(
            self.tile_m,
            # self.tile_n,
            self.tile_n * self.cluster_shape_mnk[0],  # careful, this case is not very well-tested
            self.is_causal,
            self.is_local,
            False,  # is_split_kv
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mCuSeqlensSFQ=mCuSeqlensSFQ,
            mCuSeqlensSFK=mCuSeqlensSFK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            tile_m=self.tile_m,
            tile_n=self.tile_n * self.cluster_shape_mnk[0],
            broadcast_q=self.broadcast_q,
        )
        TileSchedulerCls = partial(
            self.tile_scheduler_cls.create, tile_sched_params
        )

        AttentionMaskCls = self._generate_attention_mask_cls(
            window_size_left, window_size_right
        )
        if const_expr(self.blockscaled and self.const_p_scale):
            if warp_idx == self.load_warp_id:
                self._fill_sf(sSFP, Int32(0x77777777))
                cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()
        if const_expr(self.use_cluster2_group1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        #  EMPTY
        # (15)
        if warp_idx == self.empty_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_empty)

        #  RELAY
        # (14)
        if warp_idx == self.relay_warp_id:
            cute.arch.setmaxregister_decrease(
                self.num_regs_mma if self.use_2cta_instrs else self.num_regs_empty
            )
            if const_expr(self.use_2cta_instrs):
                self.relay(
                    dS_cluster_full_mbar_ptr,
                    dS_cluster_empty_mbar_ptr,
                    dS_cluster_leader_mbar_ptr,
                    pipeline_dS,
                    is_leader_cta,
                    cluster_layout_vmnk,
                    block_info,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                )

        #  LOAD
        # (13)
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            self.load(
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                thr_mma_dK,
                thr_mma_dQ,
                thr_mma_S_sfb,
                thr_mma_dP_sfb,
                thr_mma_dV_sfb,
                thr_mma_dK_sfb,
                thr_mma_dQ_sfb,
                mQ,
                mK,
                mKt,
                mV,
                mdO,
                mQt,
                mdOt,
                mLSE,
                mdPsum,
                sQ,
                sK,
                sKt,
                sV,
                sdO,
                sQt,
                sdOt,
                sLSE,
                sdPsum,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_Kt,
                tma_atom_V,
                tma_atom_dO,
                tma_atom_Qt,
                tma_atom_dOt,
                tma_atom_SFQ,
                tma_atom_SFK,
                tma_atom_SFV,
                tma_atom_SFDO,
                tma_atom_SFQ_dK,
                tma_atom_SFDO_dV,
                tma_atom_SFK_dQ,
                mSFQ,
                mSFK,
                mSFV,
                mSFDO,
                mSFQ_dK,
                mSFDO_dV,
                mSFK_dQ,
                sSFQ,
                sSFK,
                sSFV,
                sSFDO,
                sSFQ_dK,
                sSFDO_dV,
                sSFK_dQ,
                pipeline_Q,
                pipeline_K,
                pipeline_Qt,
                pipeline_Kt,
                pipeline_dO,
                pipeline_LSE,
                pipeline_dPsum,
                pipeline_dKV,
                cluster_layout_vmnk,
                cluster_layout_sfb_vmnk,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
                should_load_Q=True,
                should_load_dO=True,
            )

        #  MMA
        # (12)
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)

            # Alloc tmem buffer
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)

            self.mma(
                tiled_mma_S,
                tiled_mma_dP,
                tiled_mma_dV,
                tiled_mma_dK,
                tiled_mma_dQ,
                sQ,
                sQt,
                sK,
                sKt,
                sV,
                sdO,
                sdOt,
                tP,
                sdSt,
                sdS,
                tdS,
                tStS,
                tdPtdP,
                tdVtdV,
                tdKtdK,
                tdQtdQ,
                dS_cluster_full_mbar_ptr,
                dS_cluster_empty_mbar_ptr,
                dS_cluster_leader_mbar_ptr,
                pipeline_Q,
                pipeline_K,
                pipeline_Qt,
                pipeline_Kt,
                pipeline_dO,
                pipeline_S_P,
                pipeline_S_drain,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                pipeline_dP_drain,
                pipeline_dQ,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                is_leader_cta,
                blocksparse_tensors,
                sSFQ=sSFQ,
                sSFK=sSFK,
                sSFV=sSFV,
                sSFDO=sSFDO,
                sSFP=sSFP,
                sSFDS=sSFDS,
                sSFQ_dK=sSFQ_dK,
                sSFDO_dV=sSFDO_dV,
                sSFDS_dQ=sSFDS_dQ,
                sSFK_dQ=sSFK_dQ,
                tCtSFK=tCtSFK,
                tCtSFQ=tCtSFQ,
                tCtSFV=tCtSFV,
                tCtSFDO=tCtSFDO,
                tCtSFK_prologue=tCtSFK_prologue,
                tCtSFQ_prologue=tCtSFQ_prologue,
                tCtSFV_prologue=tCtSFV_prologue,
                tCtSFDO_prologue=tCtSFDO_prologue,
                tCtSFP=tCtSFP,
                tCtSFDO_dV=tCtSFDO_dV,
                tCtSFDS=tCtSFDS,
                tCtSFQ_dK=tCtSFQ_dK,
                tCtSFDS_dQ=tCtSFDS_dQ,
                tCtSFK_dQ=tCtSFK_dQ,
                tiled_mma_S_bs=tiled_mma_S_bs,
                tiled_mma_dP_bs=tiled_mma_dP_bs,
                tiled_mma_dV_bs=tiled_mma_dV_bs,
                tiled_mma_dK_bs=tiled_mma_dK_bs,
                tiled_mma_dQ_bs=tiled_mma_dQ_bs,
            )
            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        # Compute
        # (4, 5, 6, 7, 8, 9, 10, 11) --> 8 warps
        if warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_compute)  # 8 warps
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            self.compute_loop(
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                thr_mma_dK,
                tStS,
                tdPtdP,
                tdVtdV,
                tdKtdK,
                sLSE,
                sdPsum,
                mdV,
                mdK,
                mdSFV_out,
                mdSFK_out,
                sdS,
                sdS_active_layout,
                sdS_xchg,
                pipeline_LSE,
                pipeline_dPsum,
                pipeline_S_P,
                pipeline_S_drain,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                pipeline_dP_drain,
                dS_cluster_empty_mbar_ptr,
                dS_cluster_full_mbar_ptr,
                dQaccum_empty_mbar_ptr,
                softmax_scale,
                softmax_scale_log2,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                sdV,
                sdK,
                mdV_tma_tensor,
                mdK_tma_tensor,
                tma_atom_dV,
                tma_atom_dK,
                tiled_copy_r2s_dKV,
                mdK_semaphore,
                mdV_semaphore,
                aux_data,
                fastdiv_mods,
                blocksparse_tensors,
                sSFP=sSFP,
                sSFDS=sSFDS,
                sSFP_u32=sSFP_u32,
                sSFDS_u32=sSFDS_u32,
                sSFDS_dQ_u32=sSFDS_dQ_u32,
            )
            tmem_alloc_barrier.arrive()

        # Reduce
        # (0, 1, 2, 3) - dQ
        if warp_idx >= self.reduce_warp_ids[0] and warp_idx <= self.reduce_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_reduce)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(Float32)
            self.dQacc_reduce(
                tma_atom_dQ,
                mdQaccum,
                mdQaccum_tma_tensor,
                sdQaccum,
                sdQaccum_tma_layout,
                thr_mma_dQ,
                tdQtdQ,
                pipeline_dQ,
                dQaccum_empty_mbar_ptr,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                mdQ_semaphore,
                dq_accum_scale,
                blocksparse_tensors,
            )
            tmem_alloc_barrier.arrive()

        return

    @cute.jit
    def relay(
        self,
        dS_cluster_full_mbar_ptr: cute.Pointer,
        dS_cluster_empty_mbar_ptr: cute.Pointer,
        dS_cluster_leader_mbar_ptr: cute.Pointer,
        pipeline_dS: PipelineAsync,
        is_leader_cta: cutlass.Boolean,
        cluster_layout_vmnk: cute.Layout,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        dS_cluster_phase = Int32(0)
        if const_expr(self.blockscaled):
            consumer_state_dS = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Consumer, 1
            )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen, n_block // self.cluster_shape_mnk[0]
            )
            process_tile = (
                const_expr(
                    not self.is_local and not self.is_varlen_q
                )
                or m_block_min < m_block_max
            )

            if process_tile:
                num_iters = m_block_max - m_block_min
                for _ in cutlass.range(num_iters, unroll=1):
                    if const_expr(self.blockscaled):
                        if is_leader_cta:
                            pipeline_dS.consumer_wait(consumer_state_dS)

                    # Wait for dS_xchg from peer CTA
                    cute.arch.mbarrier_wait(dS_cluster_full_mbar_ptr, phase=dS_cluster_phase)

                    # Arrive on MMA leader warp
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(dS_cluster_leader_mbar_ptr, Int32(0))

                    if const_expr(self.blockscaled):
                        if is_leader_cta:
                            consumer_state_dS.advance()
                    dS_cluster_phase ^= 1

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def load(
        self,
        thr_mma_S: cute.ThrMma,
        thr_mma_dP: cute.ThrMma,
        thr_mma_dV: cute.ThrMma,
        thr_mma_dK: cute.ThrMma,
        thr_mma_dQ: cute.ThrMma,
        thr_mma_S_sfb: Optional[cute.ThrMma],
        thr_mma_dP_sfb: Optional[cute.ThrMma],
        thr_mma_dV_sfb: Optional[cute.ThrMma],
        thr_mma_dK_sfb: Optional[cute.ThrMma],
        thr_mma_dQ_sfb: Optional[cute.ThrMma],
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mKt: Optional[cute.Tensor],
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mQt: Optional[cute.Tensor],
        mdOt: Optional[cute.Tensor],
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sKt: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sQt: cute.Tensor,
        sdOt: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_Kt: Optional[cute.CopyAtom],
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_Qt: Optional[cute.CopyAtom],
        tma_atom_dOt: Optional[cute.CopyAtom],  # 2-CTA only
        tma_atom_SFQ: Optional[cute.CopyAtom],
        tma_atom_SFK: Optional[cute.CopyAtom],
        tma_atom_SFV: Optional[cute.CopyAtom],
        tma_atom_SFDO: Optional[cute.CopyAtom],
        tma_atom_SFQ_dK: Optional[cute.CopyAtom],
        tma_atom_SFDO_dV: Optional[cute.CopyAtom],
        tma_atom_SFK_dQ: Optional[cute.CopyAtom],
        mSFQ: Optional[cute.Tensor],
        mSFK: Optional[cute.Tensor],
        mSFV: Optional[cute.Tensor],
        mSFDO: Optional[cute.Tensor],
        mSFQ_dK: Optional[cute.Tensor],
        mSFDO_dV: Optional[cute.Tensor],
        mSFK_dQ: Optional[cute.Tensor],
        sSFQ: Optional[cute.Tensor],
        sSFK: Optional[cute.Tensor],
        sSFV: Optional[cute.Tensor],
        sSFDO: Optional[cute.Tensor],
        sSFQ_dK: Optional[cute.Tensor],
        sSFDO_dV: Optional[cute.Tensor],
        sSFK_dQ: Optional[cute.Tensor],
        pipeline_Q: PipelineAsync,
        pipeline_K: PipelineAsync,
        pipeline_Qt: PipelineAsync,
        pipeline_Kt: PipelineAsync,
        pipeline_dO: PipelineAsync,
        pipeline_LSE: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        pipeline_dKV: PipelineAsync,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: Optional[cute.Layout],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        should_load_Q: bool = True,
        should_load_dO: bool = True,
    ):
        producer_state_Q_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_K = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.single_stage
        )
        producer_state_Qt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_Kt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.single_stage
        )
        producer_state_dO_dPsum = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
        )
        producer_state_Q_Qt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_O_Ot = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
        )
        producer_state_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_dPsum = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
        )
        dKV_alias_empty_phase = Int32(0)

        # Compute multicast mask for Q & dO buffer full
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        q_do_mcast_mask = None
        if const_expr(self.is_q_do_mcast):
            q_do_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
        if const_expr(self.use_external_mxfp8_scales):
            block_in_cluster_coord_sfb_vmnk = (
                cluster_layout_sfb_vmnk.get_flat_coord(cta_rank_in_cluster)
            )
            sfb_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
            )
            sfb_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk,
                block_in_cluster_coord_sfb_vmnk,
                mcast_mode=1,
            )
        else:
            block_in_cluster_coord_sfb_vmnk = None
            sfb_cta_layout = None
            sfb_mcast_mask = None

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen, n_block // self.cluster_shape_mnk[0]
            )
            head_idx_kv = head_idx // self.qhead_per_kvhead
            n_block_cta_group = n_block // self.cta_group_size

            # GMEM tensors (varlen-aware)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[
                None, None, head_idx_kv
            ]
            mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[
                None, None, head_idx_kv
            ]
            offset_o = seqlen.offset_o
            if const_expr(not seqlen.has_cu_seqlens_q):
                mdO_cur = mdO[None, None, head_idx, batch_idx]
            else:
                mdO_cur = cute.domain_offset((0, offset_o), mdO[None, None, head_idx])
            mLSE_cur = seqlen.offset_batch_O(mLSE, batch_idx, dim=2, padded=True)[
                None, head_idx
            ]
            mdPsum_cur = seqlen.offset_batch_O(mdPsum, batch_idx, dim=2, padded=True)[
                None, head_idx
            ]

            if const_expr(tma_atom_Qt is not None):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mQt_cur = mQt[None, None, head_idx, batch_idx]
                else:
                    mQt_cur = cute.domain_offset((0, seqlen.offset_q, 0), mQt)[
                        None, None, head_idx
                    ]
            if const_expr(tma_atom_dOt is not None):
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mdOt_cur = mdOt[None, None, head_idx, batch_idx]
                else:
                    mdOt_cur = cute.domain_offset((offset_o, 0, 0), mdOt)[
                        None, None, head_idx
                    ]
            if const_expr(tma_atom_Kt is not None):
                if const_expr(not seqlen.has_cu_seqlens_k):
                    mKt_cur = mKt[None, None, head_idx_kv, batch_idx]
                else:
                    mKt_cur = cute.domain_offset((0, seqlen.offset_k, 0), mKt)[
                        None, None, head_idx_kv
                    ]

            # (1) S.T = K @ Q.T
            gK = cute.local_tile(
                mK_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (n_block_cta_group, 0)
            )
            tSgK = thr_mma_S.partition_A(gK)

            gQ = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_kq, mode=[1, 2]), (None, 0))
            tSgQ = thr_mma_S.partition_B(gQ)
            gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
            gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))
            gdO = cute.local_tile(mdO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None))
            tdPgdO = thr_mma_dV.partition_B(gdO)

            a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            b_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
            )

            tSFQsSFQ = tSFQgSFQ = None
            tSFKsSFK = tSFKgSFK = None
            tSFVsSFV = tSFVgSFV = None
            tSFDOsSFDO = tSFDOgSFDO = None
            tSFQ_dKsSFQ_dK = tSFQ_dKgSFQ_dK = None
            tSFDO_dVsSFDO_dV = tSFDO_dVgSFDO_dV = None
            tSFK_dQsSFK_dQ = tSFK_dQgSFK_dQ = None
            if const_expr(self.blockscaled and self.use_external_mxfp8_scales):
                mSFQ_cur = cute.domain_offset(
                    (seqlen.offset_sf_q, 0), mSFQ[None, None, head_idx]
                )
                gSFQ = cute.local_tile(
                    mSFQ_cur,
                    cute.select(self.mma_tiler_kq, mode=[1, 2]),
                    (None, 0),
                )
                tSgSFQ = thr_mma_S_sfb.partition_B(gSFQ)
                tSFQsSFQ, tSFQgSFQ = cpasync.tma_partition(
                    tma_atom_SFQ,
                    block_in_cluster_coord_sfb_vmnk[1],
                    sfb_cta_layout,
                    cute.group_modes(sSFQ, 0, 3),
                    cute.group_modes(tSgSFQ, 0, 3),
                )
                tSFQsSFQ = cute.filter_zeros(tSFQsSFQ)
                tSFQgSFQ = cute.filter_zeros(tSFQgSFQ)

                mSFK_cur = cute.domain_offset(
                    (seqlen.offset_sf_k, 0), mSFK[None, None, head_idx_kv]
                )
                gSFK = cute.local_tile(
                    mSFK_cur,
                    cute.select(self.mma_tiler_kq, mode=[0, 2]),
                    (None, 0),
                )
                tSgSFK = thr_mma_S.partition_A(gSFK)
                tSFKsSFK, tSFKgSFK = cpasync.tma_partition(
                    tma_atom_SFK,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sSFK, 0, 3),
                    cute.group_modes(tSgSFK, 0, 3),
                )
                tSFKsSFK = cute.filter_zeros(tSFKsSFK)
                tSFKgSFK = cute.filter_zeros(tSFKgSFK)

                mSFV_cur = cute.domain_offset(
                    (seqlen.offset_sf_k, 0), mSFV[None, None, head_idx_kv]
                )
                gSFV = cute.local_tile(
                    mSFV_cur,
                    cute.select(self.mma_tiler_vdo, mode=[0, 2]),
                    (None, 0),
                )
                tSgSFV = thr_mma_dP.partition_A(gSFV)
                tSFVsSFV, tSFVgSFV = cpasync.tma_partition(
                    tma_atom_SFV,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sSFV, 0, 3),
                    cute.group_modes(tSgSFV, 0, 3),
                )
                tSFVsSFV = cute.filter_zeros(tSFVsSFV)
                tSFVgSFV = cute.filter_zeros(tSFVgSFV)

                sfdo_offset = (
                    seqlen.offset_o
                    if const_expr(self.broadcast_q)
                    else seqlen.offset_sf_q
                )
                mSFDO_cur = cute.domain_offset(
                    (0, sfdo_offset), mSFDO[None, None, head_idx]
                )
                gSFDO = cute.local_tile(
                    mSFDO_cur,
                    cute.select(self.mma_tiler_vdo, mode=[1, 2]),
                    (0, None),
                )
                tSgSFDO = thr_mma_dP_sfb.partition_B(gSFDO)
                tSFDOsSFDO, tSFDOgSFDO = cpasync.tma_partition(
                    tma_atom_SFDO,
                    block_in_cluster_coord_sfb_vmnk[1],
                    sfb_cta_layout,
                    cute.group_modes(sSFDO, 0, 3),
                    cute.group_modes(tSgSFDO, 0, 3),
                )
                tSFDOsSFDO = cute.filter_zeros(tSFDOsSFDO)
                tSFDOgSFDO = cute.filter_zeros(tSFDOgSFDO)

                mSFQ_dK_cur = cute.domain_offset(
                    (0, seqlen.offset_sf_q), mSFQ_dK[None, None, head_idx]
                )
                gSFQ_dK = cute.local_tile(
                    mSFQ_dK_cur,
                    cute.select(self.mma_tiler_dsq, mode=[1, 2]),
                    (0, None),
                )
                tSgSFQ_dK = thr_mma_dK_sfb.partition_B(gSFQ_dK)
                tSFQ_dKsSFQ_dK, tSFQ_dKgSFQ_dK = cpasync.tma_partition(
                    tma_atom_SFQ_dK,
                    block_in_cluster_coord_sfb_vmnk[1],
                    sfb_cta_layout,
                    cute.group_modes(sSFQ_dK, 0, 3),
                    cute.group_modes(tSgSFQ_dK, 0, 3),
                )
                tSFQ_dKsSFQ_dK = cute.filter_zeros(tSFQ_dKsSFQ_dK)
                tSFQ_dKgSFQ_dK = cute.filter_zeros(tSFQ_dKgSFQ_dK)

                mSFDO_dV_cur = cute.domain_offset(
                    (0, sfdo_offset), mSFDO_dV[None, None, head_idx]
                )
                gSFDO_dV = cute.local_tile(
                    mSFDO_dV_cur,
                    cute.select(self.mma_tiler_pdo, mode=[1, 2]),
                    (0, None),
                )
                tSgSFDO_dV = thr_mma_dV_sfb.partition_B(gSFDO_dV)
                tSFDO_dVsSFDO_dV, tSFDO_dVgSFDO_dV = cpasync.tma_partition(
                    tma_atom_SFDO_dV,
                    block_in_cluster_coord_sfb_vmnk[1],
                    sfb_cta_layout,
                    cute.group_modes(sSFDO_dV, 0, 3),
                    cute.group_modes(tSgSFDO_dV, 0, 3),
                )
                tSFDO_dVsSFDO_dV = cute.filter_zeros(tSFDO_dVsSFDO_dV)
                tSFDO_dVgSFDO_dV = cute.filter_zeros(tSFDO_dVgSFDO_dV)

                mSFK_dQ_cur = cute.domain_offset(
                    (0, seqlen.offset_sf_k), mSFK_dQ[None, None, head_idx_kv]
                )
                gSFK_dQ = cute.local_tile(
                    mSFK_dQ_cur,
                    cute.select(self.mma_tiler_dsk, mode=[1, 2]),
                    (0, None),
                )
                tSgSFK_dQ = thr_mma_dQ_sfb.partition_B(gSFK_dQ)
                tSFK_dQsSFK_dQ, tSFK_dQgSFK_dQ = cpasync.tma_partition(
                    tma_atom_SFK_dQ,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sSFK_dQ, 0, 3),
                    cute.group_modes(tSgSFK_dQ, 0, 3),
                )
                tSFK_dQsSFK_dQ = cute.filter_zeros(tSFK_dQsSFK_dQ)
                tSFK_dQgSFK_dQ = cute.filter_zeros(tSFK_dQgSFK_dQ)

            load_K, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_K,
                block_in_cluster_coord_vmnk[2],
                a_cta_layout,
                tSgK,
                sK,
                single_stage=True,
            )

            load_Q, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q,
                cta_coord=block_in_cluster_coord_vmnk[1],
                cta_layout=b_cta_layout,
                src_tensor=tSgQ,
                dst_tensor=sQ,
                mcast_mask=q_do_mcast_mask,
            )
            load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)

            # (2) dP = V @ dO.T
            gV = cute.local_tile(
                mV_cur, cute.select(self.mma_tiler_vdo, mode=[0, 2]), (n_block_cta_group, 0)
            )
            tdPgV = thr_mma_dP.partition_A(gV)

            load_V, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_V,
                0,
                cute.make_layout(1),
                tdPgV,
                sV,
                single_stage=True,
            )

            if const_expr(tma_atom_dOt is not None):
                gdOt = cute.local_tile(
                    mdOt_cur, cute.select(self.mma_tiler_vdo, mode=[1, 2]), (None, 0)
                )
                tdPgdO = thr_mma_dP.partition_B(gdOt)
                load_dOt, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_dOt,
                    cta_coord=block_in_cluster_coord_vmnk[1],
                    cta_layout=b_cta_layout,
                    src_tensor=tdPgdO,
                    dst_tensor=sdOt,
                    mcast_mask=q_do_mcast_mask,
                )
                load_dOt = copy_utils.tma_producer_copy_fn(load_dOt, pipeline_dO)

            # (3) dV += P.T @ dO
            gdO = cute.local_tile(mdO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None))
            tdVgdO = thr_mma_dV.partition_B(gdO)
            load_dO, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dO,
                cta_coord=block_in_cluster_coord_vmnk[1],
                cta_layout=b_cta_layout,
                src_tensor=tdVgdO,
                dst_tensor=sdO,
                mcast_mask=q_do_mcast_mask,
            )
            load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)

            # (4) dK += dS.T @ Q (2-CTA: needs separate Qt load)
            if const_expr(tma_atom_Qt is not None):
                gQt = cute.local_tile(
                    mQt_cur, cute.select(self.mma_tiler_dsq, mode=[1, 2]), (0, None)
                )
                tdKgQt = thr_mma_dK.partition_B(gQt)
                load_Qt, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Qt,
                    cta_coord=block_in_cluster_coord_vmnk[1],
                    cta_layout=b_cta_layout,
                    src_tensor=tdKgQt,
                    dst_tensor=sQt,
                    mcast_mask=q_do_mcast_mask,
                )
                load_Qt = copy_utils.tma_producer_copy_fn(load_Qt, pipeline_Qt)

            # (5) dQ = dS @ K
            if const_expr(tma_atom_Kt is not None):
                gKt = cute.local_tile(
                    mKt_cur, cute.select(self.mma_tiler_dsk, mode=[1, 2]), (0, n_block_cta_group)
                )
                tdQgK = thr_mma_dQ.partition_B(gKt)

                if const_expr(self.use_external_mxfp8_scales):
                    load_Kt, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_Kt,
                        0,
                        cute.make_layout(1),
                        tdQgK,
                        sKt,
                        single_stage=True,
                    )
                else:
                    load_Kt, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_Kt,
                        block_in_cluster_coord_vmnk[1],
                        b_cta_layout,
                        tdQgK,
                        sKt,
                        single_stage=True,
                    )

            copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
            copy_stats = partial(utils.copy_bulk_g2s_maybe_elect, copy_atom_stats)
            # copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SMulticastOp(), Float32)
            # sLSE = cute.logical_divide(sLSE, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # gLSE = cute.logical_divide(gLSE, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # sdPsum = cute.logical_divide(sdPsum, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # gdPsum = cute.logical_divide(gdPsum, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # copy_stats = partial(cute.copy, copy_atom_stats, mcast_mask=q_do_mcast_mask)

            # some tiles might be empty due to block sparsity
            if const_expr(self.use_block_sparsity):
                (
                    curr_q_cnt,
                    curr_q_idx,
                    curr_full_cnt,
                    curr_full_idx,
                    loop_count,
                ) = get_block_sparse_iteration_info_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    n_block,
                    q_subtile_factor=self.q_subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = loop_count > Int32(0)
            else:
                loop_count = m_block_max - m_block_min
                process_tile = (
                    const_expr(
                        not self.is_local and not self.is_varlen_q
                    )
                    or m_block_min < m_block_max
                )

            if process_tile:
                if const_expr(self.use_block_sparsity):
                    producer_state_Q_LSE, producer_state_dO_dPsum = (
                        produce_block_sparse_q_loads_bwd_sm100(
                            blocksparse_tensors,
                            batch_idx,
                            head_idx,
                            n_block,
                            producer_state_Q_LSE,
                            producer_state_dO_dPsum,
                            pipeline_Q,
                            pipeline_LSE,
                            pipeline_dO,
                            pipeline_dPsum,
                            load_K,
                            load_V,
                            load_Q,
                            load_dO,
                            copy_stats,
                            gLSE,
                            sLSE,
                            gdPsum,
                            sdPsum,
                            self.tma_copy_bytes["K"],
                            self.tma_copy_bytes["V"],
                            should_load_Q=should_load_Q,
                            should_load_dO=should_load_dO,
                            q_subtile_factor=self.q_subtile_factor,
                            m_block_max=m_block_max,
                        )
                    )
                else:
                    first_m_block = m_block_min
                    if const_expr(self.use_2cta_instrs and self.tile_hdim == 192):
                        #### Prologue ####
                        assert should_load_Q and should_load_dO
                        # K & Q (for S)
                        pipeline_Q.producer_acquire(
                            producer_state_Q_Qt,
                            extra_tx_count=self.tma_copy_bytes["K"],
                        )
                        load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_Qt))
                        load_Q(first_m_block, producer_state=producer_state_Q_Qt)
                        pipeline_Q.producer_commit(producer_state_Q_Qt)
                        producer_state_Q_Qt.advance()
                        # LSE
                        pipeline_LSE.producer_acquire(producer_state_LSE)
                        copy_stats(
                            gLSE[None, first_m_block],
                            sLSE[None, producer_state_LSE.index],
                            mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_LSE),
                        )
                        producer_state_LSE.advance()

                        # dOt + V, for dP.T = V @ dO.T
                        pipeline_dO.producer_acquire(
                            producer_state_O_Ot,
                            extra_tx_count=self.tma_copy_bytes["V"],
                        )
                        load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_O_Ot))
                        load_dOt(first_m_block, producer_state=producer_state_O_Ot)
                        pipeline_dO.producer_commit(producer_state_O_Ot)
                        producer_state_O_Ot.advance()
                        # dPsum
                        pipeline_dPsum.producer_acquire(producer_state_dPsum)
                        copy_stats(
                            gdPsum[None, first_m_block],
                            sdPsum[None, producer_state_dPsum.index],
                            mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dPsum),
                        )
                        producer_state_dPsum.advance()

                        # Qt, for dK = dS.T @ Q
                        pipeline_Qt.producer_acquire(
                            producer_state_Q_Qt,
                            extra_tx_count=self.tma_copy_bytes["K"],
                        )
                        load_Qt(first_m_block, producer_state=producer_state_Q_Qt)
                        load_Kt(tma_bar_ptr=pipeline_Qt.producer_get_barrier(producer_state_Q_Qt))
                        pipeline_Qt.producer_commit(producer_state_Q_Qt)
                        producer_state_Q_Qt.advance()

                        # dO, for dV = P.T @ dO
                        pipeline_dO.producer_acquire(producer_state_O_Ot)
                        load_dO(first_m_block, producer_state=producer_state_O_Ot)
                        pipeline_dO.producer_commit(producer_state_O_Ot)
                        producer_state_O_Ot.advance()

                        #### Mainloop ####
                        # 2CTA: [lse | Q | dOt | dPsum | Qt | dO]
                        for m_block in cutlass.range(m_block_min + 1, m_block_max, unroll=1):
                            # LSE
                            pipeline_LSE.producer_acquire(producer_state_LSE)
                            copy_stats(
                                gLSE[None, m_block],
                                sLSE[None, producer_state_LSE.index],
                                mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_LSE),
                            )
                            producer_state_LSE.advance()

                            # Q
                            pipeline_Q.producer_acquire(producer_state_Q_Qt)
                            load_Q(m_block, producer_state=producer_state_Q_Qt)
                            pipeline_Q.producer_commit(producer_state_Q_Qt)
                            producer_state_Q_Qt.advance()

                            # dPsum
                            pipeline_dPsum.producer_acquire(producer_state_dPsum)
                            copy_stats(
                                gdPsum[None, m_block],
                                sdPsum[None, producer_state_dPsum.index],
                                mbar_ptr=pipeline_dPsum.producer_get_barrier(
                                    producer_state_dPsum
                                ),
                            )
                            producer_state_dPsum.advance()

                            # dOt, for dP.T = V @ dO.T
                            pipeline_dO.producer_acquire(producer_state_O_Ot)
                            load_dOt(m_block, producer_state=producer_state_O_Ot)
                            pipeline_dO.producer_commit(producer_state_O_Ot)
                            producer_state_O_Ot.advance()

                            # Qt, for dK = dS.T @ Q
                            pipeline_Qt.producer_acquire(producer_state_Q_Qt)
                            load_Qt(m_block, producer_state=producer_state_Q_Qt)
                            pipeline_Qt.producer_commit(producer_state_Q_Qt)
                            producer_state_Q_Qt.advance()

                            # dO, for dV = P.T @ dO
                            pipeline_dO.producer_acquire(producer_state_O_Ot)
                            load_dO(m_block, producer_state=producer_state_O_Ot)
                            pipeline_dO.producer_commit(producer_state_O_Ot)
                            producer_state_O_Ot.advance()

                    else:
                        #### Prologue ####
                        if const_expr(should_load_Q):
                            if const_expr(self.use_dedicated_k_pipeline):
                                pipeline_K.producer_acquire(producer_state_K)
                                load_K(
                                    tma_bar_ptr=pipeline_K.producer_get_barrier(
                                        producer_state_K
                                    )
                                )
                                if const_expr(self.use_external_mxfp8_scales):
                                    cute.copy(
                                        tma_atom_SFK,
                                        tSFKgSFK[None, n_block_cta_group],
                                        tSFKsSFK[None, 0],
                                        tma_bar_ptr=pipeline_K.producer_get_barrier(
                                            producer_state_K
                                        ),
                                    )
                                pipeline_K.producer_commit(producer_state_K)
                                producer_state_K.advance()
                                pipeline_Q.producer_acquire(producer_state_Q_LSE)
                            else:
                                pipeline_Q.producer_acquire(
                                    producer_state_Q_LSE,
                                    extra_tx_count=self.tma_copy_bytes["K"],
                                )
                                load_K(
                                    tma_bar_ptr=pipeline_Q.producer_get_barrier(
                                        producer_state_Q_LSE
                                    )
                                )
                                if const_expr(self.use_external_mxfp8_scales):
                                    cute.copy(
                                        tma_atom_SFK,
                                        tSFKgSFK[None, n_block_cta_group],
                                        tSFKsSFK[None, 0],
                                        tma_bar_ptr=pipeline_Q.producer_get_barrier(
                                            producer_state_Q_LSE
                                        ),
                                    )
                            load_Q(first_m_block, producer_state=producer_state_Q_LSE)
                            if const_expr(self.use_external_mxfp8_scales):
                                cute.copy(
                                    tma_atom_SFQ,
                                    tSFQgSFQ[None, first_m_block],
                                    tSFQsSFQ[None, producer_state_Q_LSE.index],
                                    tma_bar_ptr=pipeline_Q.producer_get_barrier(
                                        producer_state_Q_LSE
                                    ),
                                    mcast_mask=sfb_mcast_mask,
                                )
                            pipeline_Q.producer_commit(producer_state_Q_LSE)

                            # LSE
                            pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                            copy_stats(
                                gLSE[None, first_m_block],
                                sLSE[None, producer_state_Q_LSE.index],
                                mbar_ptr=pipeline_LSE.producer_get_barrier(
                                    producer_state_Q_LSE
                                ),
                            )
                            producer_state_Q_LSE.advance()

                        if const_expr(should_load_dO):
                            pipeline_dO.producer_acquire(
                                producer_state_dO_dPsum,
                                extra_tx_count=self.tma_copy_bytes["V"]
                                + self.tma_copy_bytes["dOt"],
                            )
                            load_V(
                                tma_bar_ptr=pipeline_dO.producer_get_barrier(
                                    producer_state_dO_dPsum
                                )
                            )
                            if const_expr(self.use_external_mxfp8_scales):
                                cute.copy(
                                    tma_atom_SFV,
                                    tSFVgSFV[None, n_block_cta_group],
                                    tSFVsSFV[None, 0],
                                    tma_bar_ptr=pipeline_dO.producer_get_barrier(
                                        producer_state_dO_dPsum
                                    ),
                                )
                            load_dO(
                                first_m_block, producer_state=producer_state_dO_dPsum
                            )
                            if const_expr(self.use_external_mxfp8_scales):
                                cute.copy(
                                    tma_atom_SFDO_dV,
                                    tSFDO_dVgSFDO_dV[None, first_m_block],
                                    tSFDO_dVsSFDO_dV[
                                        None, producer_state_dO_dPsum.index
                                    ],
                                    tma_bar_ptr=pipeline_dO.producer_get_barrier(
                                        producer_state_dO_dPsum
                                    ),
                                    mcast_mask=sfb_mcast_mask,
                                )
                            if const_expr(tma_atom_dOt is not None):
                                load_dOt(
                                    first_m_block,
                                    producer_state=producer_state_dO_dPsum,
                                )
                            if const_expr(self.use_external_mxfp8_scales):
                                cute.copy(
                                    tma_atom_SFDO,
                                    tSFDOgSFDO[None, first_m_block],
                                    tSFDOsSFDO[
                                        None, producer_state_dO_dPsum.index
                                    ],
                                    tma_bar_ptr=pipeline_dO.producer_get_barrier(
                                        producer_state_dO_dPsum
                                    ),
                                    mcast_mask=sfb_mcast_mask,
                                )
                            pipeline_dO.producer_commit(producer_state_dO_dPsum)

                            # dPsum
                            pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                            copy_stats(
                                gdPsum[None, first_m_block],
                                sdPsum[None, producer_state_dO_dPsum.index],
                                mbar_ptr=pipeline_dPsum.producer_get_barrier(
                                    producer_state_dO_dPsum
                                ),
                            )
                            producer_state_dO_dPsum.advance()

                        if const_expr(tma_atom_Kt is not None):
                            pipeline_Kt.producer_acquire(producer_state_Kt)
                            load_Kt(
                                tma_bar_ptr=pipeline_Kt.producer_get_barrier(
                                    producer_state_Kt
                                )
                            )
                            if const_expr(self.use_external_mxfp8_scales):
                                cute.copy(
                                    tma_atom_SFK_dQ,
                                    tSFK_dQgSFK_dQ[None, n_block_cta_group],
                                    tSFK_dQsSFK_dQ[None, 0],
                                    tma_bar_ptr=pipeline_Kt.producer_get_barrier(
                                        producer_state_Kt
                                    ),
                                )
                            pipeline_Kt.producer_commit(producer_state_Kt)
                            producer_state_Kt.advance()
                        #### Main Loop ####
                        for iter_idx in cutlass.range(1, loop_count, unroll=1):
                            m_block = m_block_min + iter_idx
                            if const_expr(should_load_Q):
                                if const_expr(tma_atom_Qt is not None):
                                    pipeline_Qt.producer_acquire(producer_state_Qt)
                                    load_Qt(
                                        m_block - 1, producer_state=producer_state_Qt
                                    )
                                    if const_expr(self.use_external_mxfp8_scales):
                                        cute.copy(
                                            tma_atom_SFQ_dK,
                                            tSFQ_dKgSFQ_dK[None, m_block - 1],
                                            tSFQ_dKsSFQ_dK[
                                                None, producer_state_Qt.index
                                            ],
                                            tma_bar_ptr=pipeline_Qt.producer_get_barrier(
                                                producer_state_Qt
                                            ),
                                            mcast_mask=sfb_mcast_mask,
                                        )
                                    pipeline_Qt.producer_commit(producer_state_Qt)
                                    producer_state_Qt.advance()

                                # Q (for S)
                                pipeline_Q.producer_acquire(producer_state_Q_LSE)
                                load_Q(m_block, producer_state=producer_state_Q_LSE)
                                if const_expr(self.use_external_mxfp8_scales):
                                    cute.copy(
                                        tma_atom_SFQ,
                                        tSFQgSFQ[None, m_block],
                                        tSFQsSFQ[None, producer_state_Q_LSE.index],
                                        tma_bar_ptr=pipeline_Q.producer_get_barrier(
                                            producer_state_Q_LSE
                                        ),
                                        mcast_mask=sfb_mcast_mask,
                                    )
                                pipeline_Q.producer_commit(producer_state_Q_LSE)

                                # LSE
                                pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                                copy_stats(
                                    gLSE[None, m_block],
                                    sLSE[None, producer_state_Q_LSE.index],
                                    mbar_ptr=pipeline_LSE.producer_get_barrier(
                                        producer_state_Q_LSE
                                    ),
                                )
                                producer_state_Q_LSE.advance()

                            if const_expr(should_load_dO):
                                pipeline_dO.producer_acquire(
                                    producer_state_dO_dPsum,
                                    extra_tx_count=self.tma_copy_bytes["dOt"],
                                )
                                load_dO(
                                    m_block, producer_state=producer_state_dO_dPsum
                                )
                                if const_expr(self.use_external_mxfp8_scales):
                                    cute.copy(
                                        tma_atom_SFDO_dV,
                                        tSFDO_dVgSFDO_dV[None, m_block],
                                        tSFDO_dVsSFDO_dV[
                                            None, producer_state_dO_dPsum.index
                                        ],
                                        tma_bar_ptr=pipeline_dO.producer_get_barrier(
                                            producer_state_dO_dPsum
                                        ),
                                        mcast_mask=sfb_mcast_mask,
                                    )
                                if const_expr(tma_atom_dOt is not None):
                                    load_dOt(
                                        m_block,
                                        producer_state=producer_state_dO_dPsum,
                                    )
                                if const_expr(self.use_external_mxfp8_scales):
                                    cute.copy(
                                        tma_atom_SFDO,
                                        tSFDOgSFDO[None, m_block],
                                        tSFDOsSFDO[
                                            None, producer_state_dO_dPsum.index
                                        ],
                                        tma_bar_ptr=pipeline_dO.producer_get_barrier(
                                            producer_state_dO_dPsum
                                        ),
                                        mcast_mask=sfb_mcast_mask,
                                    )
                                pipeline_dO.producer_commit(producer_state_dO_dPsum)

                                # dPsum
                                pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                                copy_stats(
                                    gdPsum[None, m_block],
                                    sdPsum[None, producer_state_dO_dPsum.index],
                                    mbar_ptr=pipeline_dPsum.producer_get_barrier(
                                        producer_state_dO_dPsum
                                    ),
                                )
                                producer_state_dO_dPsum.advance()

                        #### Tail ####
                        if const_expr(should_load_Q):
                            if const_expr(tma_atom_Qt is not None):
                                pipeline_Qt.producer_acquire(producer_state_Qt)
                                load_Qt(
                                    m_block_max - 1, producer_state=producer_state_Qt
                                )
                                if const_expr(self.use_external_mxfp8_scales):
                                    cute.copy(
                                        tma_atom_SFQ_dK,
                                        tSFQ_dKgSFQ_dK[None, m_block_max - 1],
                                        tSFQ_dKsSFQ_dK[
                                            None, producer_state_Qt.index
                                        ],
                                        tma_bar_ptr=pipeline_Qt.producer_get_barrier(
                                            producer_state_Qt
                                        ),
                                        mcast_mask=sfb_mcast_mask,
                                    )
                                pipeline_Qt.producer_commit(producer_state_Qt)
                                producer_state_Qt.advance()

                if const_expr(not self.is_persistent):
                    if const_expr(self.use_2cta_instrs and self.tile_hdim == 192):
                        pipeline_Q.producer_tail(producer_state_Q_Qt)
                        pipeline_LSE.producer_tail(producer_state_LSE)
                        pipeline_dO.producer_tail(producer_state_O_Ot)
                        pipeline_dPsum.producer_tail(producer_state_dPsum)
                    else:
                        if const_expr(should_load_Q):
                            if const_expr(self.use_dedicated_k_pipeline):
                                pipeline_K.producer_tail(producer_state_K)
                            pipeline_Q.producer_tail(producer_state_Q_LSE.clone())
                            pipeline_LSE.producer_tail(producer_state_Q_LSE)
                            if const_expr(tma_atom_Qt is not None):
                                pipeline_Qt.producer_tail(producer_state_Qt)
                        if const_expr(should_load_dO):
                            pipeline_dO.producer_tail(producer_state_dO_dPsum.clone())
                            pipeline_dPsum.producer_tail(producer_state_dO_dPsum)
                        if const_expr(tma_atom_Kt is not None):
                            pipeline_Kt.producer_tail(producer_state_Kt)

            if const_expr(self.bf16_broadcast_q):
                if process_tile:
                    # sdK and sdV alias the next tile's sQ and sdO load destinations.
                    pipeline_dKV.sync_object_empty.wait(
                        Int32(0), dKV_alias_empty_phase
                    )
                    pipeline_dKV.sync_object_empty.wait(
                        Int32(1), dKV_alias_empty_phase
                    )
                    dKV_alias_empty_phase ^= 1

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()


        if const_expr(self.is_persistent):
            if const_expr(self.use_2cta_instrs and self.tile_hdim == 192):
                pipeline_Q.producer_tail(producer_state_Q_Qt)
                pipeline_LSE.producer_tail(producer_state_LSE)
                pipeline_dO.producer_tail(producer_state_O_Ot)
                pipeline_dPsum.producer_tail(producer_state_dPsum)
            else:
                if const_expr(should_load_Q):
                    if const_expr(self.use_dedicated_k_pipeline):
                        pipeline_K.producer_tail(producer_state_K)
                    pipeline_Q.producer_tail(producer_state_Q_LSE.clone())
                    pipeline_LSE.producer_tail(producer_state_Q_LSE)
                    if const_expr(tma_atom_Qt is not None):
                        pipeline_Qt.producer_tail(producer_state_Qt)
                if const_expr(should_load_dO):
                    pipeline_dO.producer_tail(producer_state_dO_dPsum.clone())
                    pipeline_dPsum.producer_tail(producer_state_dO_dPsum)
                if const_expr(tma_atom_Kt is not None):
                    pipeline_Kt.producer_tail(producer_state_Kt)

    @cute.jit
    def mma(
        self,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        sQ: cute.Tensor,
        sQt: cute.Tensor,
        sK: cute.Tensor,
        sKt: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sdOt: cute.Tensor,
        tP: cute.Tensor,
        sdSt: cute.Tensor,
        sdS: cute.Tensor,
        tdS: cute.Tensor,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdQtdQ: cute.Tensor,
        dS_cluster_full_mbar_ptr: cute.Pointer,
        dS_cluster_empty_mbar_ptr: cute.Pointer,
        dS_cluster_leader_mbar_ptr: cute.Pointer,
        pipeline_Q: PipelineAsync,
        pipeline_K: PipelineAsync,
        pipeline_Qt: PipelineAsync,
        pipeline_Kt: PipelineAsync,
        pipeline_dO: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_S_drain: PipelineAsync,
        pipeline_dS: PipelineAsync,
        pipeline_dKV: PipelineAsync,
        pipeline_dP: PipelineAsync,
        pipeline_dP_drain: PipelineAsync,
        pipeline_dQ: PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        is_leader_cta: cutlass.Boolean,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        sSFQ: Optional[cute.Tensor] = None,
        sSFK: Optional[cute.Tensor] = None,
        sSFV: Optional[cute.Tensor] = None,
        sSFDO: Optional[cute.Tensor] = None,
        sSFP: Optional[cute.Tensor] = None,
        sSFDS: Optional[cute.Tensor] = None,
        sSFQ_dK: Optional[cute.Tensor] = None,
        sSFDO_dV: Optional[cute.Tensor] = None,
        sSFDS_dQ: Optional[cute.Tensor] = None,
        sSFK_dQ: Optional[cute.Tensor] = None,
        tCtSFK: Optional[cute.Tensor] = None,
        tCtSFQ: Optional[cute.Tensor] = None,
        tCtSFV: Optional[cute.Tensor] = None,
        tCtSFDO: Optional[cute.Tensor] = None,
        tCtSFK_prologue: Optional[cute.Tensor] = None,
        tCtSFQ_prologue: Optional[cute.Tensor] = None,
        tCtSFV_prologue: Optional[cute.Tensor] = None,
        tCtSFDO_prologue: Optional[cute.Tensor] = None,
        tCtSFP: Optional[cute.Tensor] = None,
        tCtSFDO_dV: Optional[cute.Tensor] = None,
        tCtSFDS: Optional[cute.Tensor] = None,
        tCtSFQ_dK: Optional[cute.Tensor] = None,
        tCtSFDS_dQ: Optional[cute.Tensor] = None,
        tCtSFK_dQ: Optional[cute.Tensor] = None,
        tiled_mma_S_bs: Optional[cute.TiledMma] = None,
        tiled_mma_dP_bs: Optional[cute.TiledMma] = None,
        tiled_mma_dV_bs: Optional[cute.TiledMma] = None,
        tiled_mma_dK_bs: Optional[cute.TiledMma] = None,
        tiled_mma_dQ_bs: Optional[cute.TiledMma] = None,
    ):
        # [2025-10-21] For reasons I don't understand, putting these partitioning in the main
        # kernel (before warp specialization) is a lot slower tha putting them here.
        # Partition smem / tmem tensors
        # S = K @ Q.T
        tSrK = tiled_mma_S.make_fragment_A(sK)
        tSrQ = tiled_mma_S.make_fragment_B(sQ)
        # dP = V @ dOt.T
        tdPrV = tiled_mma_dP.make_fragment_A(sV)
        tdPrdOt = tiled_mma_dP.make_fragment_B(sdOt)
        # dK = dS.T @ Q
        # For 2-CTA, dS (dK mma) MUST come from TMEM (cannot use SMEM)
        tdKrQ = tiled_mma_dK.make_fragment_B(sQt)
        if const_expr(self.blockscaled):
            tdKrdS = tiled_mma_dK_bs.make_fragment_A(tdS)
            tdKrdS = cute.make_tensor(
                cute.recast_ptr(tdS.iterator, dtype=self.ds_dtype),
                tdKrdS.layout,
            )
            tdKrQ_dK = tiled_mma_dK_bs.make_fragment_B(sQt)
        else:
            tdKrdS = tiled_mma_dK.make_fragment_A(tdS)  # From TMEM

        # dQ = dS @ K
        if const_expr(self.use_external_mxfp8_scales):
            tdQrdS = tiled_mma_dQ_bs.make_fragment_A(sdS)
            tdQrK = tiled_mma_dQ_bs.make_fragment_B(sKt)
        else:
            tdQrdS = tiled_mma_dQ.make_fragment_A(sdS)
            tdQrK = tiled_mma_dQ.make_fragment_B(sKt)
        # dV = P @ dO.T
        tdVrdO = tiled_mma_dV.make_fragment_B(sdO)
        tdVrP = tiled_mma_dV.make_fragment_A(tP)
        if const_expr(self.blockscaled):
            tiled_copy_s2t_sfk, tCsSFK_s2t, tCtSFK_s2t = make_s2t_copy_partitions(
                sSFK, tCtSFK, self.sf_dtype, self.cta_group
            )
            tiled_copy_s2t_sfq, tCsSFQ_s2t, tCtSFQ_s2t = make_s2t_copy_partitions(
                sSFQ, tCtSFQ, self.sf_dtype, self.cta_group
            )
            tiled_copy_s2t_sfv, tCsSFV_s2t, tCtSFV_s2t = make_s2t_copy_partitions(
                sSFV, tCtSFV, self.sf_dtype, self.cta_group
            )
            tiled_copy_s2t_sfdo, tCsSFDO_s2t, tCtSFDO_s2t = make_s2t_copy_partitions(
                sSFDO, tCtSFDO, self.sf_dtype, self.cta_group
            )
            _, _, tCtSFK_s2t_prologue = make_s2t_copy_partitions(
                sSFK, tCtSFK_prologue, self.sf_dtype, self.cta_group
            )
            _, _, tCtSFQ_s2t_prologue = make_s2t_copy_partitions(
                sSFQ, tCtSFQ_prologue, self.sf_dtype, self.cta_group
            )
            _, _, tCtSFV_s2t_prologue = make_s2t_copy_partitions(
                sSFV, tCtSFV_prologue, self.sf_dtype, self.cta_group
            )
            _, _, tCtSFDO_s2t_prologue = make_s2t_copy_partitions(
                sSFDO, tCtSFDO_prologue, self.sf_dtype, self.cta_group
            )
            tiled_copy_s2t_sfp, tCsSFP_s2t, tCtSFP_s2t = make_s2t_copy_partitions(
                sSFP, tCtSFP, self.sf_dtype, self.cta_group
            )
            tiled_copy_s2t_sfdo_dv, tCsSFDO_dV_s2t, tCtSFDO_dV_s2t = (
                make_s2t_copy_partitions(
                    sSFDO_dV, tCtSFDO_dV, self.sf_dtype, self.cta_group
                )
            )
            tiled_copy_s2t_sfds, tCsSFDS_s2t, tCtSFDS_s2t = make_s2t_copy_partitions(
                sSFDS, tCtSFDS, self.sf_dtype, self.cta_group
            )
            tiled_copy_s2t_sfq_dk, tCsSFQ_dK_s2t, tCtSFQ_dK_s2t = (
                make_s2t_copy_partitions(
                    sSFQ_dK, tCtSFQ_dK, self.sf_dtype, self.cta_group
                )
            )
            if const_expr(self.use_external_mxfp8_scales):
                (
                    tiled_copy_s2t_sfds_dq,
                    tCsSFDS_dQ_s2t,
                    tCtSFDS_dQ_s2t,
                ) = make_s2t_copy_partitions(
                    sSFDS_dQ,
                    tCtSFDS_dQ,
                    self.sf_dtype,
                    tcgen05.CtaGroup.ONE,
                )
                (
                    tiled_copy_s2t_sfk_dq,
                    tCsSFK_dQ_s2t,
                    tCtSFK_dQ_s2t,
                ) = make_s2t_copy_partitions(
                    sSFK_dQ,
                    tCtSFK_dQ,
                    self.sf_dtype,
                    tcgen05.CtaGroup.ONE,
                )
        # mma_qk_fn = partial(gemm_w_idx, tiled_mma_S, tStS, tSrK, tSrQ, zero_init=True)
        mma_qk_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_S,
            tStS,
            tSrK,
            tSrQ,
            sA=sK,
            sB=sQ,
            zero_init=True,
            cta_group=self.cta_group_size,
        )
        # mma_dov_fn = partial(gemm_w_idx, tiled_mma_dP, tdPtdP, tdPrV, tdPrdOt, zero_init=True)
        mma_dov_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_dP,
            tdPtdP,
            tdPrV,
            tdPrdOt,
            sA=sV,
            sB=sdOt,
            zero_init=True,
            cta_group=self.cta_group_size,
        )
        # mma_pdo_fn = partial(gemm_w_idx, tiled_mma_dV, tdVtdV, tdVrP, tdVrdO)
        mma_pdo_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_dV,
            tdVtdV,
            tdVrP,
            tdVrdO,
            sA=None,
            sB=sdO,
            tA_addr=self.tmem_P_offset,
            cta_group=self.cta_group_size,
        )
        num_unroll_groups = 2 if const_expr(self.use_2cta_instrs) else 1
        mma_dsk_fn = partial(
            gemm_w_idx,
            tiled_mma_dQ,
            tdQtdQ,
            tdQrdS,
            tdQrK,
            zero_init=True,
            num_unroll_groups=num_unroll_groups,
        )
        # mma_dsk_fn = partial(
        #     gemm_ptx_w_idx, tiled_mma_dQ, tdQtdQ, tdQrdS, tdQrK, sA=sdS, sB=sKt, zero_init=True
        # )
        # Need to explicitly pass in tA_addr for correctness
        mma_dsq_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_dK,
            tdKtdK,
            tdKrdS,
            tdKrQ,
            sA=None,
            sB=sQt,
            tA_addr=self.tmem_dS_offset,
            cta_group=self.cta_group_size,
        )

        pipeline_Q_consumer = pipeline_Q.make_consumer()
        pipeline_K_consumer = pipeline_K.make_consumer()

        consumer_state_Qt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_Q = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_Kt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.single_stage
        )
        consumer_state_dO = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )
        producer_phase_acc = Int32(1)  # For S & P, dP, dQ
        producer_phase_dQ = Int32(1)  # 2-CTA: separate phase for dQ pipeline
        producer_state_dQ = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, 1
        )
        consumer_state_dS = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        producer_phase_dKV = Int32(1)
        cta_group = pipeline_S_P.cta_group

        dS_cluster_phase = Int32(0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            sfds_consumer_stage = Int32(0)
            seqlen = SeqlenInfoCls(batch_idx)  # must be seqlen_k
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen, n_block // self.cluster_shape_mnk[0]
            )

            if const_expr(self.use_block_sparsity):
                block_iter_count = get_total_q_block_count_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    n_block,
                    q_subtile_factor=self.q_subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = block_iter_count > Int32(0)
            else:
                block_iter_count = m_block_max - m_block_min
                process_tile = (
                    const_expr(
                        not self.is_local and not self.is_varlen_q
                    )
                    or m_block_min < m_block_max
                )

            if const_expr(self.use_2cta_instrs and self.tile_hdim == 192):
                if is_leader_cta and process_tile:
                    accumulate_dK = False
                    accumulate_dV = False

                    # -----------------------------------------------------------
                    ###### MAIN LOOP
                    # -----------------------------------------------------------
                    # 1. S.T  = K    @ Q.T
                    # 2. dP.T = V    @ dO.T
                    # 3. dK   = dS.T @ Q
                    # 4. dV   = P.T  @ dO
                    # 5. dQ   = dS   @ K

                    main_loop_iters = m_block_max - m_block_min

                    # empty waits
                    # pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    # pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)

                    for _ in cutlass.range(main_loop_iters, unroll=1):
                        # 1) S.T = K @ Q.T
                        pipeline_Q.consumer_wait(consumer_state_Q)
                        pipeline_dQ.sync_object_empty.wait(
                            0, producer_phase_acc
                        )  # dQ tmem overlaps with S
                        mma_qk_fn(B_idx=consumer_state_Q.index)
                        pipeline_S_P.sync_object_full.arrive(
                            0, pipeline_S_P.producer_mask, cta_group
                        )
                        pipeline_Q.consumer_release(consumer_state_Q)
                        consumer_state_Q.advance()

                        producer_phase_acc ^= 1

                        # 2) dP.T = V @ dO.T
                        pipeline_dO.consumer_wait(consumer_state_dO)
                        pipeline_S_P.sync_object_empty.wait(
                            0, producer_phase_acc
                        )  # dP tmem overlaps with S
                        mma_dov_fn(B_idx=consumer_state_dO.index)
                        pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)
                        pipeline_dO.consumer_release(consumer_state_dO)
                        consumer_state_dO.advance()

                        # 3) dK = dS.T @ Q
                        pipeline_Q.consumer_wait(consumer_state_Q)
                        pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)  # dP -> dS
                        mma_dsq_fn(B_idx=consumer_state_Q.index, zero_init=not accumulate_dK)
                        pipeline_Q.consumer_release(consumer_state_Q)
                        consumer_state_Q.advance()
                        accumulate_dK = True

                        # 4) dV = P.T @ dO
                        # Note: if dS is written to tmem, P must be written to tmem
                        pipeline_dO.consumer_wait(consumer_state_dO)
                        mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=not accumulate_dV)
                        pipeline_dO.consumer_release(consumer_state_dO)
                        consumer_state_dO.advance()
                        accumulate_dV = True

                        # 5) dQ = dS @ K
                        pipeline_dS.consumer_wait(consumer_state_dS)
                        cute.arch.mbarrier_wait(dS_cluster_leader_mbar_ptr, phase=dS_cluster_phase)
                        mma_dsk_fn()
                        pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                        pipeline_dS.consumer_release(consumer_state_dS)
                        consumer_state_dS.advance()
                        dS_cluster_phase ^= 1

                    # signal to the epilogue that dV is ready
                    pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
                    pipeline_dKV.sync_object_full.arrive(0, pipeline_dKV.producer_mask, cta_group)
                    # signal to the epilogue that dK is ready
                    pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)
                    pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
                    producer_phase_dKV ^= 1
            elif const_expr(self.use_2cta_instrs):
                if is_leader_cta and process_tile:
                    accumulate_dK = False
                    # -----------------------------------------------------------
                    ###### Prologue
                    # -----------------------------------------------------------
                    # 1. S  = Q0 @ K.T
                    # 2. dP = V @ dOt.T
                    # 3. dV = P @ dO

                    # 1) S = K @ Q
                    pipeline_Q.consumer_wait(consumer_state_Q)
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    if const_expr(self.blockscaled):
                        cute.copy(
                            tiled_copy_s2t_sfk,
                            tCsSFK_s2t[None, None, None, None, 0],
                            tCtSFK_s2t_prologue,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfq,
                            tCsSFQ_s2t[None, None, None, None, consumer_state_Q.index],
                            tCtSFQ_s2t_prologue,
                        )
                        cute.arch.fence_view_async_tmem_store()
                        gemm_blockscaled(
                            tiled_mma_S_bs,
                            tStS,
                            tSrK,
                            tSrQ[None, None, None, consumer_state_Q.index],
                            tCtSFK_prologue,
                            tCtSFQ_prologue,
                            zero_init=True,
                        )
                    else:
                        mma_qk_fn(B_idx=consumer_state_Q.index)
                    pipeline_S_P.sync_object_full.arrive(
                        0, pipeline_S_P.producer_mask, cta_group
                    )
                    pipeline_Q.consumer_release(consumer_state_Q)
                    consumer_state_Q.advance()

                    # 2) dP = V @ dOt.T
                    pipeline_dO.consumer_wait(consumer_state_dO)
                    pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)
                    if const_expr(self.blockscaled):
                        cute.copy(
                            tiled_copy_s2t_sfv,
                            tCsSFV_s2t[None, None, None, None, 0],
                            tCtSFV_s2t_prologue,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfdo,
                            tCsSFDO_s2t[
                                None, None, None, None, consumer_state_dO.index
                            ],
                            tCtSFDO_s2t_prologue,
                        )
                        cute.arch.fence_view_async_tmem_store()
                        gemm_blockscaled(
                            tiled_mma_dP_bs,
                            tdPtdP,
                            tdPrV,
                            tdPrdOt[None, None, None, consumer_state_dO.index],
                            tCtSFV_prologue,
                            tCtSFDO_prologue,
                            zero_init=True,
                        )
                    else:
                        mma_dov_fn(B_idx=consumer_state_dO.index)
                    pipeline_dP.sync_object_full.arrive(
                        0, pipeline_dP.producer_mask, cta_group
                    )

                    # 3) dV = P.T @ dO
                    producer_phase_acc ^= 1
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    if const_expr(self.blockscaled):
                        cute.copy(
                            tiled_copy_s2t_sfp,
                            tCsSFP_s2t[None, None, None, None, 0],
                            tCtSFP_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfdo_dv,
                            tCsSFDO_dV_s2t[
                                None, None, None, None, consumer_state_dO.index
                            ],
                            tCtSFDO_dV_s2t,
                        )
                        cute.arch.fence_view_async_tmem_store()
                        gemm_blockscaled(
                            tiled_mma_dV_bs,
                            tdVtdV,
                            tdVrP,
                            tdVrdO[None, None, None, consumer_state_dO.index],
                            tCtSFP,
                            tCtSFDO_dV,
                            zero_init=True,
                        )
                    else:
                        mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=True)
                    pipeline_dO.consumer_release(consumer_state_dO)
                    consumer_state_dO.advance()

                    pipeline_Kt.consumer_wait(consumer_state_Kt)
                    # -----------------------------------------------------------
                    ###### MAIN LOOP
                    # -----------------------------------------------------------
                    # 1. S.T  = K    @ Q.T
                    # 2. dK   = dS.T @ Q
                    # 3. dP.T = V    @ dO.T
                    # 4. dQ   = dS   @ K
                    # 5. dV   = P.T  @ dO

                    main_loop_iters = (
                        block_iter_count - 1
                        if const_expr(self.use_block_sparsity)
                        else m_block_max - m_block_min - 1
                    )

                    for _ in cutlass.range(main_loop_iters, unroll=1):
                        # (1) S.T = K @ Q.T (next)
                        pipeline_Q.consumer_wait(consumer_state_Q)
                        pipeline_dQ.sync_object_empty.wait(0, producer_phase_dQ)
                        if const_expr(self.blockscaled):
                            pipeline_dP_drain.sync_object_empty.wait(
                                0, producer_phase_acc
                            )
                        if const_expr(self.blockscaled):
                            cute.copy(
                                tiled_copy_s2t_sfk,
                                tCsSFK_s2t[None, None, None, None, 0],
                                tCtSFK_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfq,
                                tCsSFQ_s2t[
                                    None,
                                    None,
                                    None,
                                    None,
                                    consumer_state_Q.index,
                                ],
                                tCtSFQ_s2t,
                            )
                            cute.arch.fence_view_async_tmem_store()
                            gemm_blockscaled(
                                tiled_mma_S_bs,
                                tStS,
                                tSrK,
                                tSrQ[None, None, None, consumer_state_Q.index],
                                tCtSFK,
                                tCtSFQ,
                                zero_init=True,
                            )
                        else:
                            mma_qk_fn(B_idx=consumer_state_Q.index)
                        pipeline_S_P.sync_object_full.arrive(
                            0, pipeline_S_P.producer_mask, cta_group
                        )
                        pipeline_Q.consumer_release(consumer_state_Q)
                        consumer_state_Q.advance()

                        # pipeline_dS.consumer_wait(consumer_state_dS)
                        # (2) dK += dS.T @ Q (cur)
                        pipeline_Qt.consumer_wait(consumer_state_Qt)
                        pipeline_dP.sync_object_empty.wait(
                            0, producer_phase_acc
                        )  # dP -> dS
                        if const_expr(self.blockscaled):
                            cute.copy(
                                tiled_copy_s2t_sfds,
                                tCsSFDS_s2t[
                                    None,
                                    None,
                                    None,
                                    None,
                                    sfds_consumer_stage,
                                ],
                                tCtSFDS_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfq_dk,
                                tCsSFQ_dK_s2t[
                                    None,
                                    None,
                                    None,
                                    None,
                                    consumer_state_Qt.index,
                                ],
                                tCtSFQ_dK_s2t,
                            )
                            cute.arch.fence_view_async_tmem_store()
                            gemm_blockscaled(
                                tiled_mma_dK_bs,
                                tdKtdK,
                                tdKrdS,
                                tdKrQ_dK[None, None, None, consumer_state_Qt.index],
                                tCtSFDS,
                                tCtSFQ_dK,
                                zero_init=not accumulate_dK,
                            )
                        else:
                            mma_dsq_fn(
                                B_idx=consumer_state_Qt.index,
                                zero_init=not accumulate_dK,
                            )
                        accumulate_dK = True
                        pipeline_Qt.consumer_release(consumer_state_Qt)
                        consumer_state_Qt.advance()

                        if const_expr(self.blockscaled):
                            pipeline_dO.consumer_wait(consumer_state_dO)
                            pipeline_S_drain.sync_object_empty.wait(
                                0, producer_phase_acc ^ 1
                            )

                        # (3) dP.T = V @ dO.T (next)
                        if const_expr(not self.blockscaled):
                            pipeline_dO.consumer_wait(consumer_state_dO)
                        if const_expr(self.blockscaled):
                            cute.copy(
                                tiled_copy_s2t_sfv,
                                tCsSFV_s2t[None, None, None, None, 0],
                                tCtSFV_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfdo,
                                tCsSFDO_s2t[
                                    None,
                                    None,
                                    None,
                                    None,
                                    consumer_state_dO.index,
                                ],
                                tCtSFDO_s2t,
                            )
                            cute.arch.fence_view_async_tmem_store()
                            gemm_blockscaled(
                                tiled_mma_dP_bs,
                                tdPtdP,
                                tdPrV,
                                tdPrdOt[None, None, None, consumer_state_dO.index],
                                tCtSFV,
                                tCtSFDO,
                                zero_init=True,
                            )
                        else:
                            mma_dov_fn(B_idx=consumer_state_dO.index)
                        pipeline_dP.sync_object_full.arrive(
                            0, pipeline_dP.producer_mask, cta_group
                        )

                        # (4) dQ = dS @ K (cur)
                        pipeline_dS.consumer_wait(consumer_state_dS)
                        cute.arch.mbarrier_wait(
                            dS_cluster_leader_mbar_ptr, phase=dS_cluster_phase
                        )
                        # M128 group-2 MX scales are unsupported. The 2CTA
                        # path brackets this plain E4M3 MMA with a fixed scale.
                        mma_dsk_fn()
                        pipeline_dQ.sync_object_full.arrive(
                            0, pipeline_dQ.producer_mask, cta_group
                        )
                        pipeline_dS.consumer_release(consumer_state_dS)
                        consumer_state_dS.advance()
                        dS_cluster_phase ^= 1
                        producer_phase_dQ ^= 1

                        # (5) dV += P.T @ dO (next)
                        producer_phase_acc ^= 1
                        pipeline_S_P.sync_object_empty.wait(
                            0, producer_phase_acc
                        )  # S -> P
                        if const_expr(self.blockscaled):
                            cute.copy(
                                tiled_copy_s2t_sfp,
                                tCsSFP_s2t[None, None, None, None, 0],
                                tCtSFP_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfdo_dv,
                                tCsSFDO_dV_s2t[
                                    None,
                                    None,
                                    None,
                                    None,
                                    consumer_state_dO.index,
                                ],
                                tCtSFDO_dV_s2t,
                            )
                            cute.arch.fence_view_async_tmem_store()
                            gemm_blockscaled(
                                tiled_mma_dV_bs,
                                tdVtdV,
                                tdVrP,
                                tdVrdO[None, None, None, consumer_state_dO.index],
                                tCtSFP,
                                tCtSFDO_dV,
                                zero_init=False,
                            )
                        else:
                            mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=False)
                        pipeline_dO.consumer_release(consumer_state_dO)
                        consumer_state_dO.advance()

                    pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)

                    # signal to the epilogue that dV is ready
                    pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
                    pipeline_dKV.sync_object_full.arrive(0, pipeline_dKV.producer_mask, cta_group)
                    pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)

                    # -----------------------------------------------------------
                    # Tail: Remaining dK and dQ
                    # -----------------------------------------------------------
                    # pipeline_dS.consumer_wait(consumer_state_dS)
                    # dK += dS.T @ Q
                    pipeline_Qt.consumer_wait(consumer_state_Qt)
                    pipeline_dP.sync_object_empty.wait(
                        0, producer_phase_acc
                    )  # dP -> dS
                    if const_expr(self.blockscaled):
                        cute.copy(
                            tiled_copy_s2t_sfds,
                            tCsSFDS_s2t[None, None, None, None, 0],
                            tCtSFDS_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfq_dk,
                            tCsSFQ_dK_s2t[
                                None, None, None, None, consumer_state_Qt.index
                            ],
                            tCtSFQ_dK_s2t,
                        )
                        cute.arch.fence_view_async_tmem_store()
                        gemm_blockscaled(
                            tiled_mma_dK_bs,
                            tdKtdK,
                            tdKrdS,
                            tdKrQ_dK[None, None, None, consumer_state_Qt.index],
                            tCtSFDS,
                            tCtSFQ_dK,
                            zero_init=not accumulate_dK,
                        )
                    else:
                        mma_dsq_fn(
                            B_idx=consumer_state_Qt.index,
                            zero_init=not accumulate_dK,
                        )
                    pipeline_Qt.consumer_release(consumer_state_Qt)
                    consumer_state_Qt.advance()
                    # signal to the epilogue that dK is ready
                    pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
                    producer_phase_dKV ^= 1

                    # dQ = dS @ K
                    pipeline_dS.consumer_wait(consumer_state_dS)
                    cute.arch.mbarrier_wait(dS_cluster_leader_mbar_ptr, phase=dS_cluster_phase)
                    pipeline_dQ.sync_object_empty.wait(0, producer_phase_dQ)
                    mma_dsk_fn()
                    pipeline_dQ.sync_object_full.arrive(
                        0, pipeline_dQ.producer_mask, cta_group
                    )
                    pipeline_dS.consumer_release(consumer_state_dS)
                    pipeline_Kt.consumer_release(consumer_state_Kt)
                    consumer_state_dS.advance()
                    consumer_state_Kt.advance()
                    dS_cluster_phase ^= 1
                    producer_phase_dQ ^= 1

                    producer_phase_acc ^= 1
            else:
                if is_leader_cta and process_tile:
                    accumulate_dK = False
                    # -----------------------------------------------------------
                    ###### Prologue
                    # -----------------------------------------------------------
                    # 1. S  = Q0 @ K.T
                    # 2. dP = V @ dOt.T
                    # 3. dV = P @ dO

                    # 1) S = K @ Q
                    if const_expr(self.use_dedicated_k_pipeline):
                        handle_K = pipeline_K_consumer.wait_and_advance()
                    handle_Q = pipeline_Q_consumer.wait_and_advance()
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    if const_expr(self.blockscaled):
                        cute.copy(
                            tiled_copy_s2t_sfk,
                            tCsSFK_s2t[None, None, None, None, 0],
                            tCtSFK_s2t_prologue,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfq,
                            tCsSFQ_s2t[None, None, None, None, handle_Q.index],
                            tCtSFQ_s2t_prologue,
                        )
                        cute.arch.fence_view_async_tmem_store()
                        gemm_blockscaled(
                            tiled_mma_S_bs,
                            tStS,
                            tSrK,
                            tSrQ[None, None, None, handle_Q.index],
                            tCtSFK_prologue,
                            tCtSFQ_prologue,
                            zero_init=True,
                        )
                        handle_Q.release()
                    else:
                        mma_qk_fn(B_idx=handle_Q.index)
                    pipeline_S_P.sync_object_full.arrive(
                        0, pipeline_S_P.producer_mask, cta_group
                    )

                    # 2) dP = V @ dOt.T
                    pipeline_dO.consumer_wait(consumer_state_dO)
                    pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)
                    pipeline_dQ.producer_acquire(producer_state_dQ)
                    if const_expr(self.blockscaled):
                        cute.copy(
                            tiled_copy_s2t_sfv,
                            tCsSFV_s2t[None, None, None, None, 0],
                            tCtSFV_s2t_prologue,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfdo,
                            tCsSFDO_s2t[
                                None, None, None, None, consumer_state_dO.index
                            ],
                            tCtSFDO_s2t_prologue,
                        )
                        cute.arch.fence_view_async_tmem_store()
                        gemm_blockscaled(
                            tiled_mma_dP_bs,
                            tdPtdP,
                            tdPrV,
                            tdPrdOt[None, None, None, consumer_state_dO.index],
                            tCtSFV_prologue,
                            tCtSFDO_prologue,
                            zero_init=True,
                        )
                    else:
                        mma_dov_fn(B_idx=consumer_state_dO.index)
                    pipeline_dP.sync_object_full.arrive(
                        0, pipeline_dP.producer_mask, cta_group
                    )

                    producer_phase_acc ^= 1
                    # 3) dV = P.T @ dO
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    if const_expr(self.blockscaled):
                        cute.copy(
                            tiled_copy_s2t_sfp,
                            tCsSFP_s2t[None, None, None, None, 0],
                            tCtSFP_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfdo_dv,
                            tCsSFDO_dV_s2t[
                                None, None, None, None, consumer_state_dO.index
                            ],
                            tCtSFDO_dV_s2t,
                        )
                        cute.arch.fence_view_async_tmem_store()
                        gemm_blockscaled(
                            tiled_mma_dV_bs,
                            tdVtdV,
                            tdVrP,
                            tdVrdO[None, None, None, consumer_state_dO.index],
                            tCtSFP,
                            tCtSFDO_dV,
                            zero_init=True,
                        )
                    else:
                        mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=True)
                    pipeline_dO.consumer_release(consumer_state_dO)
                    consumer_state_dO.advance()
                    if const_expr(self.blockscaled):
                        pipeline_Kt.consumer_wait(consumer_state_Kt)

                    # -----------------------------------------------------------
                    ###### MAIN LOOP
                    # -----------------------------------------------------------
                    # 1. S  = K    @ Q.T
                    # 2. dQ = dS   @ K
                    # 3. dK = dS.T @ Q
                    # 4. dP = V    @ dOt.T
                    # 5. dV = P.T  @ dO

                    # For block sparsity, we use block_iter_count; for dense, use m_block range
                    # MMA doesn't need actual m_block indices, just the iteration count
                    main_loop_iters = (
                        block_iter_count - 1
                        if const_expr(self.use_block_sparsity)
                        else m_block_max - m_block_min - 1
                    )

                    handle_Q_next = handle_Q
                    for _ in cutlass.range(main_loop_iters, unroll=1):
                        # (1) S.T = K @ Q.T
                        handle_Q_next = pipeline_Q_consumer.wait_and_advance()
                        if const_expr(self.blockscaled):
                            # Wait until the overlapping dP scale band is free,
                            # then rewrite it to clear stale F32 bytes.
                            pipeline_dP_drain.sync_object_empty.wait(
                                0, producer_phase_acc
                            )
                            cute.copy(
                                tiled_copy_s2t_sfk,
                                tCsSFK_s2t[None, None, None, None, 0],
                                tCtSFK_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfq,
                                tCsSFQ_s2t[
                                    None,
                                    None,
                                    None,
                                    None,
                                    handle_Q_next.index,
                                ],
                                tCtSFQ_s2t,
                            )
                            cute.arch.fence_view_async_tmem_store()
                            gemm_blockscaled(
                                tiled_mma_S_bs,
                                tStS,
                                tSrK,
                                tSrQ[None, None, None, handle_Q_next.index],
                                tCtSFK,
                                tCtSFQ,
                                zero_init=True,
                            )
                            handle_Q_next.release()
                        else:
                            mma_qk_fn(B_idx=handle_Q_next.index)
                        pipeline_S_P.sync_object_full.arrive(
                            0, pipeline_S_P.producer_mask, cta_group
                        )

                        # (2) dK += dS.T @ Q
                        pipeline_dS.consumer_wait(consumer_state_dS)
                        if const_expr(self.blockscaled):
                            pipeline_Qt.consumer_wait(consumer_state_Qt)
                            cute.copy(
                                tiled_copy_s2t_sfds,
                                tCsSFDS_s2t[
                                    None,
                                    None,
                                    None,
                                    None,
                                    sfds_consumer_stage,
                                ],
                                tCtSFDS_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfq_dk,
                                tCsSFQ_dK_s2t[
                                    None,
                                    None,
                                    None,
                                    None,
                                    consumer_state_Qt.index,
                                ],
                                tCtSFQ_dK_s2t,
                            )
                            cute.arch.fence_view_async_tmem_store()
                            gemm_blockscaled(
                                tiled_mma_dK_bs,
                                tdKtdK,
                                tdKrdS,
                                tdKrQ_dK[
                                    None, None, None, consumer_state_Qt.index
                                ],
                                tCtSFDS,
                                tCtSFQ_dK,
                                zero_init=not accumulate_dK,
                            )
                            pipeline_Qt.consumer_release(consumer_state_Qt)
                            consumer_state_Qt.advance()
                        else:
                            mma_dsq_fn(
                                B_idx=handle_Q.index,
                                zero_init=not accumulate_dK,
                            )
                            handle_Q.release()
                        accumulate_dK = True

                        # (3) dQ = dS @ K
                        if const_expr(self.blockscaled):
                            pipeline_S_drain.sync_object_empty.wait(
                                0, producer_phase_acc ^ 1
                            )
                            cute.copy(
                                tiled_copy_s2t_sfds_dq,
                                tCsSFDS_dQ_s2t[
                                    None,
                                    None,
                                    None,
                                    None,
                                    sfds_consumer_stage,
                                ],
                                tCtSFDS_dQ_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfk_dq,
                                tCsSFK_dQ_s2t[None, None, None, None, 0],
                                tCtSFK_dQ_s2t,
                            )
                            cute.arch.fence_view_async_tmem_store()
                            gemm_blockscaled(
                                tiled_mma_dQ_bs,
                                tdQtdQ,
                                tdQrdS,
                                tdQrK,
                                tCtSFDS_dQ,
                                tCtSFK_dQ,
                                zero_init=True,
                            )
                        else:
                            mma_dsk_fn()
                        pipeline_dQ.producer_commit(producer_state_dQ)
                        producer_state_dQ.advance()
                        pipeline_dS.consumer_release(consumer_state_dS)
                        consumer_state_dS.advance()
                        if const_expr(not self.use_cluster2_group1):
                            sfds_consumer_stage ^= 1

                        # (4) dP = V @ dO.T
                        pipeline_dO.consumer_wait(consumer_state_dO)
                        pipeline_dQ.producer_acquire(producer_state_dQ)
                        if const_expr(self.blockscaled):
                            cute.copy(
                                tiled_copy_s2t_sfv,
                                tCsSFV_s2t[None, None, None, None, 0],
                                tCtSFV_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfdo,
                                tCsSFDO_s2t[
                                    None,
                                    None,
                                    None,
                                    None,
                                    consumer_state_dO.index,
                                ],
                                tCtSFDO_s2t,
                            )
                            cute.arch.fence_view_async_tmem_store()
                            gemm_blockscaled(
                                tiled_mma_dP_bs,
                                tdPtdP,
                                tdPrV,
                                tdPrdOt[
                                    None, None, None, consumer_state_dO.index
                                ],
                                tCtSFV,
                                tCtSFDO,
                                zero_init=True,
                            )
                        else:
                            mma_dov_fn(B_idx=consumer_state_dO.index)
                        pipeline_dP.sync_object_full.arrive(
                            0, pipeline_dP.producer_mask, cta_group
                        )

                        # (5) dV += P.T @ dO
                        producer_phase_acc ^= 1
                        pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                        if const_expr(self.blockscaled):
                            cute.copy(
                                tiled_copy_s2t_sfp,
                                tCsSFP_s2t[None, None, None, None, 0],
                                tCtSFP_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfdo_dv,
                                tCsSFDO_dV_s2t[
                                    None,
                                    None,
                                    None,
                                    None,
                                    consumer_state_dO.index,
                                ],
                                tCtSFDO_dV_s2t,
                            )
                            cute.arch.fence_view_async_tmem_store()
                            gemm_blockscaled(
                                tiled_mma_dV_bs,
                                tdVtdV,
                                tdVrP,
                                tdVrdO[
                                    None, None, None, consumer_state_dO.index
                                ],
                                tCtSFP,
                                tCtSFDO_dV,
                                zero_init=False,
                            )
                        else:
                            mma_pdo_fn(
                                B_idx=consumer_state_dO.index,
                                zero_init=False,
                            )
                        pipeline_dO.consumer_release(consumer_state_dO)
                        consumer_state_dO.advance()

                        handle_Q = handle_Q_next

                    if const_expr(
                        self.use_dedicated_k_pipeline
                        and self.use_fused_mxfp8_full_tile_schedule
                    ):
                        handle_K.release()

                    if const_expr(not self.is_persistent):
                        pipeline_S_P.sync_object_full.arrive(
                            0, pipeline_S_P.producer_mask, cta_group
                        )

                    # signal to the epilogue that dV is ready
                    # pipeline_dKV.producer_acquire(producer_state_dKV)
                    pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
                    # pipeline_dKV.producer_commit(producer_state_dKV)
                    pipeline_dKV.sync_object_full.arrive(0, pipeline_dKV.producer_mask, cta_group)
                    # producer_state_dKV.advance()
                    # pipeline_dKV.producer_acquire(producer_state_dKV)
                    pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)

                    # -----------------------------------------------------------
                    # Tail: Remaining dK and dQ
                    # -----------------------------------------------------------
                    # 1) dK += dS.T @ Q
                    pipeline_dS.consumer_wait(consumer_state_dS)
                    if const_expr(self.blockscaled):
                        pipeline_Qt.consumer_wait(consumer_state_Qt)
                        cute.copy(
                            tiled_copy_s2t_sfds,
                            tCsSFDS_s2t[
                                None,
                                None,
                                None,
                                None,
                                sfds_consumer_stage,
                            ],
                            tCtSFDS_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfq_dk,
                            tCsSFQ_dK_s2t[
                                None,
                                None,
                                None,
                                None,
                                consumer_state_Qt.index,
                            ],
                            tCtSFQ_dK_s2t,
                        )
                        cute.arch.fence_view_async_tmem_store()
                        gemm_blockscaled(
                            tiled_mma_dK_bs,
                            tdKtdK,
                            tdKrdS,
                            tdKrQ_dK[None, None, None, consumer_state_Qt.index],
                            tCtSFDS,
                            tCtSFQ_dK,
                            zero_init=not accumulate_dK,
                        )
                        pipeline_Qt.consumer_release(consumer_state_Qt)
                        consumer_state_Qt.advance()
                    else:
                        mma_dsq_fn(
                            B_idx=handle_Q.index,
                            zero_init=not accumulate_dK,
                        )
                    # signal to the epilogue that dK is ready
                    pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
                    producer_phase_dKV ^= 1

                    # 2) dQ = dS @ K
                    if const_expr(self.blockscaled):
                        pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                        cute.copy(
                            tiled_copy_s2t_sfds_dq,
                            tCsSFDS_dQ_s2t[
                                None,
                                None,
                                None,
                                None,
                                sfds_consumer_stage,
                            ],
                            tCtSFDS_dQ_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfk_dq,
                            tCsSFK_dQ_s2t[None, None, None, None, 0],
                            tCtSFK_dQ_s2t,
                        )
                        cute.arch.fence_view_async_tmem_store()
                        gemm_blockscaled(
                            tiled_mma_dQ_bs,
                            tdQtdQ,
                            tdQrdS,
                            tdQrK,
                            tCtSFDS_dQ,
                            tCtSFK_dQ,
                            zero_init=True,
                        )
                    else:
                        mma_dsk_fn()
                    pipeline_dQ.producer_commit(producer_state_dQ)
                    producer_state_dQ.advance()
                    if const_expr(not self.blockscaled):
                        handle_Q.release()
                    if const_expr(
                        self.use_dedicated_k_pipeline
                        and not self.use_fused_mxfp8_full_tile_schedule
                    ):
                        handle_K.release()
                    pipeline_dS.consumer_release(consumer_state_dS)
                    consumer_state_dS.advance()
                    if const_expr(self.blockscaled):
                        pipeline_Kt.consumer_release(consumer_state_Kt)
                        consumer_state_Kt.advance()

                    producer_phase_acc ^= 1
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # Currently it hangs if we have this S_P.producer_tail, will need to understand why
        # pipeline_S_P.producer_tail(producer_state_S_P)
        # pipeline_dP.producer_tail(producer_state_dP)
        # pipeline_dKV.producer_tail(producer_state_dKV)
        # pipeline_dQ.producer_tail(producer_state_dQ)

    @cute.jit
    def split_wg(
        self,
        t: cute.Tensor,
        wg_idx: cutlass.Int32,
        num_wg: cutlass.Constexpr[int],
    ):
        reduced_shape = cute.product_each(t.shape)
        rank = len(reduced_shape)
        if const_expr(reduced_shape[1] > 1):
            assert rank >= 2, "Need rank >= 2 for t in split_wg"
            t = cute.logical_divide(t, (reduced_shape[0], reduced_shape[1] // num_wg))
            coord = (None, (None, wg_idx)) + (None,) * (rank - 2)
        else:
            assert rank >= 3, "Need rank >= 3 for t in split_wg"
            if const_expr(rank == 3):
                t = cute.logical_divide(
                    t, (reduced_shape[0], reduced_shape[1], reduced_shape[2] // num_wg)
                )
                coord = (
                    None,
                    None,
                    (None, wg_idx),
                ) + (None,) * (rank - 3)
            else:
                t = cute.logical_divide(
                    t,
                    (
                        reduced_shape[0],
                        reduced_shape[1],
                        reduced_shape[2],
                        reduced_shape[3] // num_wg,
                    ),
                )
                coord = (
                    None,
                    None,
                    None,
                    (None, wg_idx),
                ) + (None,) * (rank - 4)
        return t[coord]

    @cute.jit
    def apply_score_mod(
        self,
        tSrS_t2r,
        thr_copy_t2r,
        thr_mma_S,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax_scale,
        seqlen_info,
        aux_data: Optional[AuxData] = None,
        fastdiv_mods=(None, None),
    ):
        """Apply forward score modification for SM100 backward pass."""
        # In bwd, S is computed as K @ Q.T so dimensions are (tile_n, tile_m).
        # With 2CTA, partition_C must see the full cluster tile so each CTA
        # gets its own half of the tile.
        cluster_tile_n = self.tile_n * self.cta_group_size
        cluster_n_block = n_block // self.cta_group_size
        cS = cute.make_identity_tensor((cluster_tile_n, self.tile_m))
        cS = cute.domain_offset((cluster_n_block * cluster_tile_n, m_block * self.tile_m), cS)
        tScS = thr_mma_S.partition_C(cS)
        tScS_idx = thr_copy_t2r.partition_D(tScS)

        apply_score_mod_inner(
            tSrS_t2r,
            tScS_idx,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_data,
            fastdiv_mods,
            seqlen_info,
            constant_q_idx=None,
            qhead_per_kvhead=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            transpose_indices=True,
        )

    @cute.jit
    def apply_score_mod_bwd(
        self,
        grad_tensor,
        score_tensor,
        index_tensor,
        batch_idx,
        head_idx,
        softmax_scale,
        seqlen_info,
        aux_data: Optional[AuxData] = None,
        fastdiv_mods=(None, None),
    ):
        """Apply backward score modification (joint graph) for SM100."""
        apply_score_mod_bwd_inner(
            grad_tensor,
            score_tensor,
            index_tensor,
            self.score_mod_bwd,
            batch_idx,
            head_idx,
            softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_data,
            fastdiv_mods,
            seqlen_info,
            constant_q_idx=None,
            qhead_per_kvhead=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            transpose_indices=True,
        )

    @cute.jit
    def compute_loop(
        self,
        thr_mma_S: cute.ThrMma,
        thr_mma_dP: cute.ThrMma,
        thr_mma_dV: cute.ThrMma,
        thr_mma_dK: cute.ThrMma,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        mdSFV_out: Optional[cute.Tensor],
        mdSFK_out: Optional[cute.Tensor],
        sdS: cute.Tensor,
        sdS_mma_layout: cute.ComposedLayout,
        sdS_xchg: cute.Tensor,
        pipeline_LSE: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_S_drain: PipelineAsync,
        pipeline_dS: PipelineAsync,
        pipeline_dKV: PipelineAsync,
        pipeline_dP: PipelineAsync,
        pipeline_dP_drain: PipelineAsync,
        dS_cluster_empty_mbar_ptr: cute.Pointer,
        dS_cluster_full_mbar_ptr: cute.Pointer,
        dQaccum_empty_mbar_ptr: cute.Pointer,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        sdV: Optional[cute.Tensor],
        sdK: Optional[cute.Tensor],
        mdV_tma_tensor: Optional[cute.Tensor],
        mdK_tma_tensor: Optional[cute.Tensor],
        tma_atom_dV: Optional[cute.CopyAtom],
        tma_atom_dK: Optional[cute.CopyAtom],
        tiled_copy_r2s_dKV: Optional[cute.TiledCopy],
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        aux_data: Optional[AuxData] = None,
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        sSFP: Optional[cute.Tensor] = None,
        sSFDS: Optional[cute.Tensor] = None,
        sSFP_u32: Optional[cute.Tensor] = None,
        sSFDS_u32: Optional[cute.Tensor] = None,
        sSFDS_dQ_u32: Optional[cute.Tensor] = None,
    ):
        sLSE_2D = cute.make_tensor(
            sLSE.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.Q_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        sdPsum_2D = cute.make_tensor(
            sdPsum.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.dO_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        # if const_expr(self.SdP_swapAB):
        sLSE_2D = layout_utils.transpose_view(sLSE_2D)
        sdPsum_2D = layout_utils.transpose_view(sdPsum_2D)

        # tix: [128...384]  8 warps
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())  # 4-11
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))
        # tidx = cute.arch.thread_idx()[0] - (cute.arch.WARP_SIZE * self.compute_warp_ids[0])
        dp_idx = tidx % 128
        num_wg = len(self.compute_warp_ids) // 4  # 2
        # wg_idx:
        # 0: [256...384]
        # 1: [128...256]

        tileP_f32_like = self.cta_tiler[1] // 32 * self.v_dtype.width
        # tStS has shape ((128, 128), 1, 1), tStP has shape ((128, 64), 1, 1)
        # tP overlap with tS
        tStP = cute.composition(tStS, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        tStP = cute.make_tensor(tStS.iterator, tStP.layout)  # Otherwise the tmem address is wrong
        tScS = thr_mma_S.partition_C(cute.make_identity_tensor(self.mma_tiler_kq[:2]))
        tScP = cute.composition(tScS, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        # tdS overlap with tdP
        tdPtdS = cute.composition(tdPtdP, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))

        # 2-CTA assumes: repetiton should always be 32 & 16
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32
        )
        tmem_store_repetition = 16 if self.v_dtype.width == 16 else 8
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(tmem_store_repetition)),
            Float32,
        )

        # tmem -> rmem
        thr_copy_t2r = copy_utils.make_tmem_copy(tmem_load_atom, num_wg).get_slice(tidx)
        tStS_t2r = thr_copy_t2r.partition_S(tStS)  # (((32, 32), 1), 2, 1, 1)
        tdPtdP_t2r = thr_copy_t2r.partition_S(tdPtdP)
        tScS_t2r = thr_copy_t2r.partition_D(tScS)  # ((32, 1), 2, 1, 1)
        t0ScS_t2r = thr_copy_t2r.get_slice(0).partition_D(tScS)  # ((32, 1), 2, 1, 1)
        # ((32, 1), 2, 1, 1, STAGE)
        tSsLSE = thr_copy_t2r.partition_D(thr_mma_S.partition_C(sLSE_2D))
        tSsdPsum = thr_copy_t2r.partition_D(thr_mma_dP.partition_C(sdPsum_2D))
        # rmem -> tmem
        thr_copy_r2t = copy_utils.make_tmem_copy(tmem_store_atom, num_wg).get_slice(tidx)
        tScP_r2t = thr_copy_r2t.partition_S(tScP)
        tStP_r2t = thr_copy_r2t.partition_D(tStP)
        tdPtdS_r2t = thr_copy_r2t.partition_D(tdPtdS)
        # rmem -> smem
        # This part is a bit iffy, we might be making a lot of assumptions here
        copy_atom_r2s = sm100_utils_basic.get_smem_store_op(
            LayoutEnum.ROW_MAJOR, self.ds_dtype, Float32, thr_copy_t2r
        )
        thr_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, thr_copy_t2r).get_slice(tidx)

        # We assume the swizzle (i.e. layout.inner) stays the same
        sdS_epi_layout = sm100_utils_basic.make_smem_layout_epi(
            self.ds_dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_m), 1
        )
        sdS_data_layout = cute.slice_(
            sdS_epi_layout.outer, (None, None, 0)
        )  # ((8,16), (64,2))
        # Need to group into 1 mode to be compatible w thr_copy_r2s
        sdS_layout = cute.make_layout(
            (sdS_data_layout.shape,), stride=(sdS_data_layout.stride,)
        )
        sdS_epi = cute.make_tensor(sdS.iterator, sdS_layout)
        tRS_sdS = thr_copy_r2s.partition_D(sdS_epi)

        if const_expr(self.use_2cta_instrs):
            sdS_xchg_epi = cute.make_tensor(
                cute.recast_ptr(sdS_xchg.iterator, sdS_mma_layout.inner),
                sdS_layout,
            )
            tRS_sdS_xchg = thr_copy_r2s.partition_D(sdS_xchg_epi)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        # 2-CTA: CTA 0 exchanges stage 1 (bottom half), CTA 1 exchanges stage 0 (top half)
        exchange_stage = cta_rank_in_cluster ^ 1 if const_expr(self.use_2cta_instrs) else Int32(0)

        consumer_state_S_P_dP = pipeline.make_pipeline_state(  # Our impl has shortcut for stage==1
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        # consumer_phase_S_P_dP = Int32(0)
        producer_state_dS = pipeline.make_pipeline_state(  # Our impl has shortcut for stage==1
            cutlass.pipeline.PipelineUserType.Producer, 1
        )
        consumer_state_dKV = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 2
        )
        consumer_state_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_dPsum = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen, n_block // self.cluster_shape_mnk[0]
            )
            cluster_n_block = n_block // self.cta_group_size
            if const_expr(not self.elide_full_tile_mask):
                mask = AttentionMaskCls(seqlen)
                check_n_boundary = (
                    (cluster_n_block + 1) * self.tile_n * self.cta_group_size
                    > seqlen.seqlen_k
                )
                mask_fn = partial(
                    mask.apply_mask_sm100_transposed,
                    tScS_t2r=tScS_t2r,
                    t0ScS_t2r=t0ScS_t2r,
                    n_block=cluster_n_block,
                    mask_seqlen=True,
                    mask_causal=self.is_causal,
                    mask_local=self.is_local,
                    mask_mod=self.mask_mod,
                    batch_idx=batch_idx,
                    head_idx=head_idx,
                    aux_data=aux_data,
                    fastdiv_mods=fastdiv_mods,
                )

            # prefetch_LSE = not self.is_causal
            prefetch_LSE = False

            curr_q_cnt = Int32(0)
            curr_q_idx = None
            curr_full_cnt = Int32(0)
            curr_full_idx = None
            loop_count = m_block_max - m_block_min
            if const_expr(self.use_block_sparsity):
                assert blocksparse_tensors is not None
                (
                    curr_q_cnt,
                    curr_q_idx,
                    curr_full_cnt,
                    curr_full_idx,
                    loop_count,
                ) = get_block_sparse_iteration_info_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    n_block,
                    q_subtile_factor=self.q_subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = loop_count > Int32(0)
            else:
                process_tile = (
                    const_expr(
                        not self.is_local and not self.is_varlen_q
                    )
                    or m_block_min < m_block_max
                )
                loop_count = m_block_max - m_block_min

            # Mainloop
            # Block sparsity: iterate over sparse m_block count and derive actual m_block
            # from Q_IDX/FULL_Q_IDX tensors. Dense: iterate m_block_min..m_block_max directly.
            for iter_idx in cutlass.range(loop_count, unroll=1):
                m_block = m_block_min + iter_idx
                is_full_block = False
                if const_expr(self.use_block_sparsity):
                    m_block, is_full_block = get_m_block_from_iter_bwd(
                        iter_idx,
                        curr_q_cnt,
                        curr_q_idx,
                        curr_full_cnt,
                        curr_full_idx,
                        q_subtile_factor=self.q_subtile_factor,
                        m_block_max=m_block_max,
                    )
                # Prefetch 1 stage of LSE
                pipeline_LSE.consumer_wait(consumer_state_LSE)
                tSrLSE_s2r = cute.make_rmem_tensor(
                    tScS_t2r[None, 0, 0, 0].shape, Float32
                )
                if const_expr(prefetch_LSE and not self.shuffle_LSE):
                    cute.autovec_copy(tSsLSE[None, 0, 0, 0, consumer_state_LSE.index], tSrLSE_s2r)

                pipeline_S_P.consumer_wait(consumer_state_S_P_dP)
                # pipeline_S_P.sync_object_full.wait(0, consumer_phase_S_P_dP)
                #### TMEM->RMEM (Load S from TMEM)
                tSrS_t2r = cute.make_rmem_tensor(tScS_t2r.shape, Float32)
                cute.copy(thr_copy_t2r, tStS_t2r, tSrS_t2r)

                if const_expr(self.use_cluster2_group1):
                    cute.arch.fence_view_async_tmem_load()
                    self.compute_sync_barrier.arrive_and_wait()
                    with cute.arch.elect_one():
                        pipeline_S_drain.sync_object_empty.arrive(
                            0, pipeline_S_drain.consumer_mask
                        )

                if const_expr(self.tile_hdim == 192):
                    # Signal S tmem load completion using pipeline_S_P when hdim 192
                    # dP is overlapped with S
                    cute.arch.fence_view_async_tmem_load()
                    with cute.arch.elect_one():
                        pipeline_S_P.consumer_release(consumer_state_S_P_dP)
                elif const_expr(self.use_2cta_instrs and self.tile_hdim <= 128):
                    # Signal S tmem load completion using pipeline_dS when 2cta hdim 128
                    # dQ is overlapped with S
                    if iter_idx > 0:
                        cute.arch.fence_view_async_tmem_load()
                        with cute.arch.elect_one():
                            pipeline_dS.producer_commit(producer_state_dS)
                        producer_state_dS.advance()

                if const_expr(self.score_mod_bwd is not None):
                    tSrS_pre = cute.make_fragment_like(tSrS_t2r)
                    cute.autovec_copy(tSrS_t2r, tSrS_pre)

                if const_expr(self.score_mod is not None):
                    # Apply score_mod FIRST -> matches forward
                    self.apply_score_mod(
                        tSrS_t2r,
                        thr_copy_t2r,
                        thr_mma_S,
                        batch_idx,
                        head_idx,
                        m_block,
                        n_block,
                        softmax_scale,
                        seqlen,
                        aux_data,
                        fastdiv_mods,
                    )

                if const_expr(not self.elide_full_tile_mask):
                    #### APPLY MASK (after score_mod, matching forward pass order)
                    check_m_boundary = False
                    if const_expr(not self.broadcast_q_full_q_tiles):
                        check_m_boundary = (
                            (m_block + 1) * self.tile_m > seqlen.seqlen_q
                        )
                    if (
                        const_expr(
                            not self.blockscaled
                            and not self.broadcast_q_full_q_tiles
                        )
                        or check_n_boundary
                        or check_m_boundary
                    ):
                        mask_fn(
                            tSrS_t2r,
                            m_block=m_block,
                            is_full_block=is_full_block,
                            check_m_boundary=check_m_boundary,
                        )
                num_stages = cute.size(tScS_t2r, mode=[1])
                # ---------------------------------------------
                #### P = exp(S - LSE)
                # ---------------------------------------------
                lane_idx = cute.arch.lane_idx()
                tSrP_r2t_f32 = cute.make_rmem_tensor(tScP_r2t.shape, Float32)  # 64
                tSrP_r2t = cute.recast_tensor(tSrP_r2t_f32, self.q_dtype)
                for stage in cutlass.range_constexpr(num_stages):
                    if const_expr(
                        self.blockscaled
                        and not self.const_p_scale
                    ):
                        p_amax = Float32(0.0)
                    tSrS_cur = tSrS_t2r[None, stage, 0, 0]
                    tSsLSE_cur = tSsLSE[None, stage, 0, 0, consumer_state_LSE.index]
                    if const_expr(not self.shuffle_LSE):
                        if const_expr(stage > 0 or not prefetch_LSE):
                            cute.autovec_copy(tSsLSE_cur, tSrLSE_s2r)
                        tSrLSE = tSrLSE_s2r
                    else:
                        tSrLSE = tSsLSE_cur[lane_idx]
                    for v in cutlass.range_constexpr(cute.size(tSrS_t2r, mode=[0]) // 2):
                        if const_expr(not self.shuffle_LSE):
                            lse_pair = (tSrLSE[2 * v], tSrLSE[2 * v + 1])
                        else:
                            lse_pair = (
                                utils.shuffle_sync(tSrLSE, offset=2 * v),
                                utils.shuffle_sync(tSrLSE, offset=2 * v + 1),
                            )
                        tSrS_cur[2 * v], tSrS_cur[2 * v + 1] = (
                            cute.arch.fma_packed_f32x2(
                                ((tSrS_cur[2 * v], tSrS_cur[2 * v + 1])),
                                (softmax_scale_log2, softmax_scale_log2),
                                (-lse_pair[0], -lse_pair[1]),
                            )
                        )
                        if const_expr(self.blockscaled):
                            p0 = cute.math.exp2(
                                min_f32(tSrS_cur[2 * v], Float32(20.0)),
                                fastmath=True,
                            )
                            p1 = cute.math.exp2(
                                min_f32(tSrS_cur[2 * v + 1], Float32(20.0)),
                                fastmath=True,
                            )
                            tSrS_cur[2 * v] = p0
                            tSrS_cur[2 * v + 1] = p1
                            if const_expr(
                                not self.const_p_scale
                            ):
                                p_amax = max_f32(p_amax, max_f32(p0, p1))
                        else:
                            tSrS_cur[2 * v] = cute.math.exp2(
                                tSrS_cur[2 * v], fastmath=True
                            )
                            tSrS_cur[2 * v + 1] = cute.math.exp2(
                                tSrS_cur[2 * v + 1], fastmath=True
                            )
                    if const_expr(self.blockscaled):
                        if const_expr(self.const_p_scale):
                            inv_scale_p = Float32(256.0)
                        else:
                            inv_scale_p, sfp_u32 = fused_amax_to_e8m0_scale_f32(
                                p_amax, Float32(E4M3_MAX_NORM_RCP)
                            )
                            if p_amax == Float32(0.0):
                                inv_scale_p = Float32(1.0)
                                sfp_u32 = Uint32(127)
                            wg_idx = tidx // 128
                            if dp_idx < cute.size(sSFP_u32.shape[0]):
                                sfp_byte_addr = Int32(
                                    utils.elem_pointer(
                                        sSFP_u32, (dp_idx, Int32(0))
                                    ).toint()
                                )
                                sfp_byte_offset = Int32(
                                    const_expr(stage * 2)
                                ) + Int32(wg_idx)
                                _st_shared_b8(
                                    sfp_byte_addr + sfp_byte_offset,
                                    Int32(sfp_u32 & Uint32(0xFF)),
                                )
                        p_dst = tSrP_r2t[None, stage, 0, 0]
                        p_dst_i32 = cute.recast_tensor(p_dst, Int32)
                        for v in cutlass.range_constexpr(cute.size(p_dst_i32)):
                            p01 = cute.arch.mul_packed_f32x2(
                                (tSrS_cur[4 * v], tSrS_cur[4 * v + 1]),
                                (inv_scale_p, inv_scale_p),
                            )
                            p23 = cute.arch.mul_packed_f32x2(
                                (tSrS_cur[4 * v + 2], tSrS_cur[4 * v + 3]),
                                (inv_scale_p, inv_scale_p),
                            )
                            p_dst_i32[v] = utils.cvt_fp8x4_f32(
                                p01[0], p01[1], p23[0], p23[1], self.q_dtype
                            )
                    else:
                        utils.cvt_f16(tSrS_cur, tSrP_r2t[None, stage, 0, 0])
                    if const_expr(stage == 0 and not self.use_cluster2_group1):
                        cute.arch.fence_view_async_tmem_load()
                        # Without this barrier, we could have 1 warp writing to P in tmem while
                        # another warp is still reading S from tmem.
                        self.compute_sync_barrier.arrive_and_wait()
                        if const_expr(self.blockscaled):
                            with cute.arch.elect_one():
                                pipeline_S_drain.sync_object_empty.arrive(
                                    0, pipeline_S_drain.consumer_mask
                                )
                    cute.copy(
                        thr_copy_r2t,
                        tSrP_r2t_f32[None, stage, None, None],
                        tStP_r2t[None, stage, None, None],
                    )

                cute.arch.fence_view_async_tmem_store()
                cute.arch.fence_view_async_shared()
                if const_expr(self.use_fused_mxfp8_full_tile_schedule):
                    cute.arch.sync_warp()
                else:
                    self.compute_sync_barrier.arrive_and_wait()
                if const_expr(not self.tile_hdim == 192):
                    # Signal tmem store P completion with pipeline_S_P
                    with cute.arch.elect_one():
                        pipeline_S_P.consumer_release(consumer_state_S_P_dP)
                        # pipeline_S_P.sync_object_empty.arrive(0, pipeline_S_P.consumer_mask)
                pipeline_LSE.consumer_release(consumer_state_LSE)
                consumer_state_LSE.advance()
                # ---------------------------------------------
                # dS.T = P.T * (dP.T - D)
                # ---------------------------------------------
                pipeline_dPsum.consumer_wait(consumer_state_dPsum)
                pipeline_dP.consumer_wait(consumer_state_S_P_dP)
                # pipeline_dP.sync_object_full.wait(0, consumer_phase_S_P_dP)
                ### Now delayed to after loop
                # consumer_state_S_P_dP.advance()
                # consumer_phase_S_P_dP ^= 1

                ##### dS.T = P.T * (dP.T - Psum)
                cluster_tile_n = self.tile_n * self.cta_group_size
                cS_bwd = cute.make_identity_tensor((cluster_tile_n, self.tile_m))
                cS_bwd = cute.domain_offset(
                    (cluster_n_block * cluster_tile_n, m_block * self.tile_m),
                    cS_bwd,
                )
                tScS_bwd = thr_mma_S.partition_C(cS_bwd)
                tScS_idx_bwd = thr_copy_t2r.partition_D(tScS_bwd)
                for stage in cutlass.range_constexpr(num_stages):
                    tdPrdP_t2r = cute.make_rmem_tensor(
                        tScS_t2r[None, 0, None, None].shape, Float32
                    )
                    cute.copy(
                        thr_copy_t2r, tdPtdP_t2r[None, stage, None, None], tdPrdP_t2r
                    )
                    cute.arch.fence_view_async_tmem_load()
                    self.compute_sync_barrier.arrive_and_wait()
                    if const_expr(
                        self.use_2cta_instrs
                        and self.blockscaled
                        and self.tile_hdim == 128
                        and stage == 0
                    ):
                        with cute.arch.elect_one():
                            pipeline_dP_drain.sync_object_empty.arrive(
                                0, pipeline_dP_drain.consumer_mask
                            )
                    if const_expr(
                        self.blockscaled
                        and not self.use_2cta_instrs
                        and stage == num_stages - 1
                    ):
                        with cute.arch.elect_one():
                            pipeline_dP.consumer_release(consumer_state_S_P_dP)
                    tdPrdP_cur = tdPrdP_t2r[None, 0, 0]
                    tSrS_cur = tSrS_t2r[None, stage, 0, 0]
                    tSsdPsum_cur = tSsdPsum[None, stage, 0, 0, consumer_state_dPsum.index]
                    if const_expr(not self.shuffle_dPsum):
                        tSrdPsum = cute.make_fragment_like(tSsdPsum_cur, Float32)
                        cute.autovec_copy(tSsdPsum_cur, tSrdPsum)
                    else:
                        tSrdPsum = tSsdPsum_cur[lane_idx]
                    for v in cutlass.range_constexpr(cute.size(tdPrdP_t2r, mode=[0]) // 2):
                        if const_expr(not self.shuffle_dPsum):
                            dPsum_pair = (tSrdPsum[2 * v], tSrdPsum[2 * v + 1])
                        else:
                            dPsum_pair = (
                                utils.shuffle_sync(tSrdPsum, offset=2 * v),
                                utils.shuffle_sync(tSrdPsum, offset=2 * v + 1),
                            )
                        tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = (
                            quack.activation.sub_packed_f32x2(
                                (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                                dPsum_pair,
                            )
                        )
                        tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = (
                            cute.arch.mul_packed_f32x2(
                                (tSrS_cur[2 * v], tSrS_cur[2 * v + 1]),
                                (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                            )
                        )
                    tScS_idx_cur = tScS_idx_bwd[None, stage, 0, 0]
                    if const_expr(self.score_mod_bwd is not None):
                        tSrS_pre_cur = tSrS_pre[None, stage, 0, 0]
                        self.apply_score_mod_bwd(
                            tdPrdP_cur,
                            tSrS_pre_cur,
                            tScS_idx_cur,
                            batch_idx,
                            head_idx,
                            softmax_scale,
                            seqlen,
                            aux_data,
                            fastdiv_mods,
                        )

                    if const_expr(
                        not self.elide_full_tile_mask
                        and not (self.blockscaled and self.broadcast_q)
                    ):
                        if check_n_boundary or check_m_boundary:
                            for i in cutlass.range(
                                cute.size(tdPrdP_cur), unroll_full=True
                            ):
                                kv_idx, q_idx = tScS_idx_cur[i]
                                if (
                                    kv_idx >= seqlen.seqlen_k
                                    or q_idx >= seqlen.seqlen_q
                                ):
                                    tdPrdP_cur[i] = 0.0

                    if const_expr(
                        stage == 0 and self.blockscaled and self.use_2cta_instrs
                    ):
                        pipeline_dS.producer_acquire(producer_state_dS)

                    tdPrdS_cvt = cute.make_fragment_like(tdPrdP_cur, self.ds_dtype)
                    tdPrdS_smem_cvt = tdPrdS_cvt
                    if const_expr(self.blockscaled):
                        local_ds_amax = Float32(0.0)
                        for i in cutlass.range_constexpr(cute.size(tdPrdP_cur)):
                            local_ds_amax = fused_abs_max_f32(
                                local_ds_amax, tdPrdP_cur[i]
                            )
                        # TODO: Compare unified 32x32 dS scaling with 1x32 dK
                        # scaling on production dumps and quantify dK degradation.
                        dk_ds_amax = (
                            redux_sync_max_abs_f32(local_ds_amax)
                            if const_expr(self.unified_ds_scale)
                            else local_ds_amax
                        )
                        inv_scale_dk, sfds_dK_u32 = fused_amax_to_e8m0_scale_f32(
                            dk_ds_amax, Float32(E4M3_MAX_NORM_RCP)
                        )
                        if (
                            dk_ds_amax == Float32(0.0)
                            or seqlen.seqlen_k == Int32(1)
                        ):
                            inv_scale_dk = Float32(1.0)
                            sfds_dK_u32 = Uint32(127)
                        if seqlen.seqlen_k == Int32(1):
                            inv_scale_dk = Float32(0.0)
                        scaled_ds_dk = cute.make_fragment_like(tdPrdP_cur, Float32)
                        for v in cutlass.range_constexpr(cute.size(tdPrdP_cur) // 2):
                            scaled_ds_dk[2 * v], scaled_ds_dk[2 * v + 1] = (
                                cute.arch.mul_packed_f32x2(
                                    (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                                    (inv_scale_dk, inv_scale_dk),
                                )
                            )
                        utils.cvt_fp8(scaled_ds_dk, tdPrdS_cvt)

                        sfds_dQ_u32 = sfds_dK_u32
                        if const_expr(not self.unified_ds_scale):
                            dq_ds_amax = redux_sync_max_abs_f32(local_ds_amax)
                            inv_scale_dq, sfds_dQ_u32 = fused_amax_to_e8m0_scale_f32(
                                dq_ds_amax, Float32(E4M3_MAX_NORM_RCP)
                            )
                            if (
                                dq_ds_amax == Float32(0.0)
                                or seqlen.seqlen_k == Int32(1)
                            ):
                                inv_scale_dq = Float32(1.0)
                                sfds_dQ_u32 = Uint32(127)
                            if seqlen.seqlen_k == Int32(1):
                                inv_scale_dq = Float32(0.0)
                            scaled_ds_dq = cute.make_fragment_like(
                                tdPrdP_cur, Float32
                            )
                            for v in cutlass.range_constexpr(
                                cute.size(tdPrdP_cur) // 2
                            ):
                                scaled_ds_dq[2 * v], scaled_ds_dq[2 * v + 1] = (
                                    cute.arch.mul_packed_f32x2(
                                        (
                                            tdPrdP_cur[2 * v],
                                            tdPrdP_cur[2 * v + 1],
                                        ),
                                        (inv_scale_dq, inv_scale_dq),
                                    )
                                )
                            tdPrdS_smem_cvt = cute.make_fragment_like(
                                tdPrdP_cur, self.ds_dtype
                            )
                            utils.cvt_fp8(scaled_ds_dq, tdPrdS_smem_cvt)

                        wg_idx_ds = tidx // 128
                        sfds_buffer_stage = (
                            Int32(0)
                            if const_expr(self.use_cluster2_group1)
                            else iter_idx & Int32(1)
                        )
                        if dp_idx < cute.size(sSFDS_u32.shape[0]):
                            sfds_byte_addr = Int32(
                                utils.elem_pointer(
                                    sSFDS_u32, (dp_idx, sfds_buffer_stage)
                                ).toint()
                            )
                            sfds_byte_offset = Int32(const_expr(stage * 2)) + Int32(
                                wg_idx_ds
                            )
                            _st_shared_b8(
                                sfds_byte_addr + sfds_byte_offset,
                                Int32(sfds_dK_u32 & Uint32(0xFF)),
                            )
                        warp_within_wg = (tidx % 128) // 32
                        target_dp_idx = (
                            Int32(const_expr(stage * 2)) + Int32(wg_idx_ds)
                        ) * Int32(32) + lane_idx
                        if target_dp_idx < cute.size(sSFDS_dQ_u32.shape[0]):
                            sfds_dq_byte_addr = Int32(
                                utils.elem_pointer(
                                    sSFDS_dQ_u32,
                                    (target_dp_idx, sfds_buffer_stage),
                                ).toint()
                            )
                            _st_shared_b8(
                                sfds_dq_byte_addr + Int32(warp_within_wg),
                                Int32(sfds_dQ_u32 & Uint32(0xFF)),
                            )
                    else:
                        utils.cvt_f16(tdPrdP_cur, tdPrdS_cvt)
                    if const_expr(stage == 0):
                        pipeline_dS.producer_acquire(producer_state_dS)
                        if const_expr(self.use_2cta_instrs):
                            tdPrdS_xchg = cute.make_fragment_like(
                                tdPrdS_smem_cvt, self.ds_dtype
                            )

                    # RMEM->TMEM: always write to TMEM for MMA
                    tdPrdS_r2t_f32 = cute.recast_tensor(tdPrdS_cvt, Float32)
                    cute.copy(thr_copy_r2t, tdPrdS_r2t_f32, tdPtdS_r2t[None, stage, 0, 0])

                    # RMEM->SMEM: For 2-CTA, keep exchange stage in registers, write non-exchange to sdS
                    if const_expr(self.use_2cta_instrs):
                        if exchange_stage == stage:
                            cute.autovec_copy(tdPrdS_smem_cvt, tdPrdS_xchg)
                        else:
                            cute.autovec_copy(
                                tdPrdS_smem_cvt, tRS_sdS[None, stage]
                            )
                    else:
                        cute.autovec_copy(tdPrdS_smem_cvt, tRS_sdS[None, stage])

                cute.arch.fence_view_async_tmem_store()
                if const_expr(self.blockscaled):
                    cute.arch.fence_proxy("async.shared", space="cta")

                if const_expr(self.use_2cta_instrs):
                    cute.arch.sync_warp()
                    # use pipeline_dP to signal tmem store of dS
                    with cute.arch.elect_one():
                        pipeline_dP.consumer_release(consumer_state_S_P_dP)
                consumer_state_S_P_dP.advance()

                # After the loop: copy exchange registers to sdS_xchg buffer
                if const_expr(self.use_2cta_instrs):
                    # when hdim 192, sdQaccum overlapped with sdS_xchg
                    if const_expr(self.tile_hdim == 192):
                        cute.arch.mbarrier_wait(
                            dQaccum_empty_mbar_ptr, phase=producer_state_dS.phase
                        )
                    cute.autovec_copy(tdPrdS_xchg, tRS_sdS_xchg[None, 0])

                cute.arch.fence_view_async_shared()
                if const_expr(self.use_fused_mxfp8_full_tile_schedule):
                    cute.arch.sync_warp()
                else:
                    self.compute_sync_barrier.arrive_and_wait()
                if const_expr(not self.blockscaled and not self.use_2cta_instrs):
                    with cute.arch.elect_one():
                        pipeline_dP.consumer_release(consumer_state_S_P_dP)
                pipeline_dPsum.consumer_release(consumer_state_dPsum)
                consumer_state_dPsum.advance()
                # when 2cta hdim 128, pipeline_dS also signals S tmem load completion so is deferred
                if const_expr(not (self.use_2cta_instrs and self.tile_hdim == 128)):
                    with cute.arch.elect_one():
                        pipeline_dS.producer_commit(producer_state_dS)
                    producer_state_dS.advance()

                # 2-CTA: DSMEM copy from sdS_xchg to peer's sdS buffer
                if const_expr(self.use_2cta_instrs):
                    stage_copy_bytes = const_expr(self.tma_copy_bytes["dS"] // 2)
                    stage_copy_elems = const_expr(stage_copy_bytes // (self.ds_dtype.width // 8))
                    if tidx == 0:
                        peer_cta_rank_in_cluster = cta_rank_in_cluster ^ 1
                        smem_src_ptr = sdS_xchg.iterator
                        # Destination is peer's sdS at our CTA's offset (exchange_stage position)
                        smem_dst_ptr = sdS.iterator + cta_rank_in_cluster * stage_copy_elems
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            dS_cluster_full_mbar_ptr,
                            stage_copy_bytes,
                            peer_cta_rank_in_cluster=peer_cta_rank_in_cluster,
                        )
                        copy_utils.cpasync_bulk_s2cluster(
                            smem_src_ptr,
                            smem_dst_ptr,
                            dS_cluster_full_mbar_ptr,
                            stage_copy_bytes,
                            peer_cta_rank_in_cluster=peer_cta_rank_in_cluster,
                        )

            # Final signal for dS smem store completion
            if const_expr(self.use_2cta_instrs and self.tile_hdim == 128):
                if process_tile:
                    with cute.arch.elect_one():
                        pipeline_dS.producer_commit(producer_state_dS)
                    producer_state_dS.advance()

            # Epilogue
            # Run epilogue if we processed any m_blocks for this n_block
            if process_tile:
                if const_expr(self.use_dedicated_mxfp8_dkv_tma):
                    consumer_state_dKV = self.epilogue_dKV(
                        dp_idx,
                        warp_idx,
                        batch_idx,
                        head_idx,
                        n_block,
                        seqlen,
                        thr_mma_dV,
                        thr_mma_dK,
                        tdVtdV,
                        tdKtdK,
                        mdV,
                        mdK,
                        pipeline_dKV,
                        consumer_state_dKV,
                        softmax_scale,
                        mdSFK_out,
                        mdSFV_out,
                        mdV_tma_tensor,
                        mdK_tma_tensor,
                        sdV,
                        sdK,
                        tma_atom_dV,
                        tma_atom_dK,
                        tiled_copy_r2s_dKV,
                    )
                elif const_expr(not self.use_tma_store):
                    consumer_state_dKV = self.epilogue_dKV(
                        dp_idx,
                        warp_idx,
                        batch_idx,
                        head_idx,
                        n_block,
                        seqlen,
                        thr_mma_dV,
                        thr_mma_dK,
                        tdVtdV,
                        tdKtdK,
                        mdV,
                        mdK,
                        pipeline_dKV,
                        consumer_state_dKV,
                        softmax_scale,
                        mdSFK_out,
                        mdSFV_out,
                    )
                elif const_expr(self.bf16_broadcast_q) and (
                    n_block + 1
                ) * self.tile_n > seqlen.seqlen_k:
                    consumer_state_dKV = self.epilogue_dKV(
                        dp_idx,
                        warp_idx,
                        batch_idx,
                        head_idx,
                        n_block,
                        seqlen,
                        thr_mma_dV,
                        thr_mma_dK,
                        tdVtdV,
                        tdKtdK,
                        mdV,
                        mdK,
                        pipeline_dKV,
                        consumer_state_dKV,
                        softmax_scale,
                        mdSFK_out,
                        mdSFV_out,
                    )
                else:
                    consumer_state_dKV = self.epilogue_dKV_tma_pair(
                        dp_idx,
                        batch_idx,
                        head_idx,
                        n_block,
                        seqlen,
                        thr_mma_dV,
                        thr_mma_dK,
                        tdVtdV,
                        tdKtdK,
                        mdV_tma_tensor,
                        mdK_tma_tensor,
                        sdV,
                        sdK,
                        tma_atom_dV,
                        tma_atom_dK,
                        tiled_copy_r2s_dKV,
                        pipeline_dKV,
                        consumer_state_dKV,
                        softmax_scale,
                        mdV_semaphore,
                        mdK_semaphore,
                    )
            # Zero dK/dV for empty tiles (local attention or block sparsity)
            # When total_m_block_cnt == 0 for block sparsity, no Q tiles contribute to this KV tile
            if const_expr(not self.dKV_postprocess):
                should_zero_dKV = False
                if const_expr(self.is_local or self.is_varlen_q):
                    should_zero_dKV = m_block_min >= m_block_max
                if const_expr(self.use_block_sparsity):
                    # For block sparsity, zero when no m_blocks contribute to this n_block
                    if not process_tile:
                        should_zero_dKV = True

                if should_zero_dKV:
                    # For 2-CTA: use cluster-wide tile size (cta_group_size * tile_n)
                    cluster_tile_n = self.tile_n * self.cta_group_size
                    n_block_for_tile = n_block // self.cta_group_size
                    gmem_tiled_copy_zero_dK = copy_utils.tiled_copy_2d(
                        self.dk_dtype,
                        math.gcd(64, self.tile_hdim),
                        128,  # num_threads
                    )
                    gmem_tiled_copy_zero_dV = copy_utils.tiled_copy_2d(
                        self.dv_dtype,
                        math.gcd(64, self.tile_hdimv),
                        128,  # num_threads
                    )
                    gmem_thr_copy_zero_dK = gmem_tiled_copy_zero_dK.get_slice(dp_idx)
                    gmem_thr_copy_zero_dV = gmem_tiled_copy_zero_dV.get_slice(dp_idx)
                    mdV_cur = seqlen.offset_batch_K(mdV, batch_idx, dim=3)[None, None, head_idx]
                    mdK_cur = seqlen.offset_batch_K(mdK, batch_idx, dim=3)[None, None, head_idx]
                    gdK = cute.local_tile(
                        mdK_cur, (cluster_tile_n, self.tile_hdim), (n_block_for_tile, 0)
                    )
                    gdV = cute.local_tile(
                        mdV_cur, (cluster_tile_n, self.tile_hdimv), (n_block_for_tile, 0)
                    )
                    tdKgdK = gmem_thr_copy_zero_dK.partition_D(gdK)
                    tdVgdV = gmem_thr_copy_zero_dV.partition_D(gdV)
                    cdK = cute.make_identity_tensor((cluster_tile_n, self.tile_hdim))
                    cdV = cute.make_identity_tensor((cluster_tile_n, self.tile_hdimv))
                    tdKcdK = gmem_thr_copy_zero_dK.partition_D(cdK)
                    tdVcdV = gmem_thr_copy_zero_dV.partition_D(cdV)
                    assert cute.size(tdKgdK[None, 0, 0]) == cute.size(tdVgdV[None, 0, 0])
                    zero = cute.make_fragment_like(tdKgdK[None, 0, 0])
                    zero.fill(0.0)
                    if tidx < 128:
                        for i in cutlass.range_constexpr(tdKgdK.shape[1]):
                            row_idx = tdKcdK[0, i, 0][0]
                            if row_idx < seqlen.seqlen_k - cluster_tile_n * n_block_for_tile:
                                for j in cutlass.range_constexpr(tdKgdK.shape[2]):
                                    cute.copy(gmem_tiled_copy_zero_dK, zero, tdKgdK[None, i, j])
                    else:
                        for i in cutlass.range_constexpr(tdVgdV.shape[1]):
                            row_idx = tdVcdV[0, i, 0][0]
                            if row_idx < seqlen.seqlen_k - cluster_tile_n * n_block_for_tile:
                                for j in cutlass.range_constexpr(tdVgdV.shape[2]):
                                    cute.copy(gmem_tiled_copy_zero_dV, zero, tdVgdV[None, i, j])
                    if const_expr(self.output_mxfp8_dkv):
                        if tidx < cluster_tile_n:
                            global_row = cluster_tile_n * n_block_for_tile + tidx
                            if global_row < seqlen.seqlen_k:
                                mdSFK_cur = seqlen.offset_batch_K(
                                    mdSFK_out, batch_idx, dim=3
                                )[None, None, head_idx]
                                mdSFV_cur = seqlen.offset_batch_K(
                                    mdSFV_out, batch_idx, dim=3
                                )[None, None, head_idx]
                                for col in cutlass.range_constexpr(
                                    self.tile_hdim // self.sf_vec_size
                                ):
                                    mdSFK_cur[(global_row, col)] = Uint8(0x7F)
                                for col in cutlass.range_constexpr(
                                    self.tile_hdimv // self.sf_vec_size
                                ):
                                    mdSFV_cur[(global_row, col)] = Uint8(0x7F)

            if const_expr(self.is_persistent) and process_tile:
                with cute.arch.elect_one():
                    pipeline_S_P.sync_object_empty.arrive(0, pipeline_S_P.consumer_mask)
                    pipeline_S_drain.sync_object_empty.arrive(
                        0, pipeline_S_drain.consumer_mask
                    )
                    if const_expr(
                        self.use_2cta_instrs
                        and self.blockscaled
                        and self.tile_hdim == 128
                    ):
                        pipeline_dP_drain.sync_object_empty.arrive(
                            0, pipeline_dP_drain.consumer_mask
                        )
                    pipeline_dP.sync_object_empty.arrive(0, pipeline_dP.consumer_mask)

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        if const_expr(self.use_dedicated_mxfp8_dkv_tma):
            leader_warp = (
                cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
            ) == 0
            if leader_warp:
                cute.arch.cp_async_bulk_wait_group(0, read=False)

    @cute.jit
    def _dq_semaphore_lock_value(
        self,
        iter_idx: Int32,
        curr_q_cnt: Int32,
        curr_dq_write_order: Optional[cute.Tensor],
        curr_dq_write_order_full: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        block_info: BlockInfo,
        seqlen,
        m_block: Int32,
        n_block: Int32,
    ) -> Int32:
        lock_value = n_block
        if const_expr(self.spt):
            n_block_max_for_m_block = block_info.get_n_block_max_for_m_block(seqlen, m_block)
            lock_value = n_block_max_for_m_block - 1 - n_block
        if const_expr(self.use_block_sparsity):
            assert blocksparse_tensors is not None
            if const_expr(blocksparse_tensors.dq_write_order is not None):
                sparse_iter = iter_idx // self.q_subtile_factor
                if sparse_iter < curr_q_cnt:
                    assert curr_dq_write_order is not None
                    lock_value = curr_dq_write_order[sparse_iter]
                else:
                    assert curr_dq_write_order_full is not None
                    lock_value = curr_dq_write_order_full[sparse_iter - curr_q_cnt]
        return lock_value

    @cute.jit
    def dQacc_reduce(
        self,
        tma_atom_dQ: Optional[cute.CopyAtom],
        mdQaccum: cute.Tensor,
        mdQaccum_tma_tensor: Optional[cute.Tensor],
        sdQaccum: cute.Tensor,
        sdQaccum_tma_layout: cute.Layout,
        thr_mma_dQ: cute.ThrMma,
        tdQtdQ: cute.Tensor,
        pipeline_dQ: PipelineAsync,
        dQaccum_empty_mbar_ptr: Optional[cute.Pointer],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        mdQ_semaphore: Optional[cute.Tensor],
        dq_accum_scale: Float32,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
    ):
        num_reduce_threads = cute.arch.WARP_SIZE * len(self.reduce_warp_ids)
        tidx = cute.arch.thread_idx()[0] % num_reduce_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx() % len(self.reduce_warp_ids))
        is_tma_warp = warp_idx == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        # TMEM -> RMEM
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dQ_reduce_ncol_t2r)), Float32
        )
        thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ).get_slice(tidx)
        tdQtdQ_t2r = thr_copy_t2r.partition_S(tdQtdQ)
        tdQcdQ = thr_mma_dQ.partition_C(cute.make_identity_tensor(self.mma_tiler_dsk[:2]))
        tdQrdQ_t2r_shape = thr_copy_t2r.partition_D(tdQcdQ).shape
        # For 2-CTA: reduce_stage = dQaccum_reduce_stage_t2r / cta_group_size
        expected_reduce_stages_t2r = self.dQaccum_reduce_stage_t2r // self.cta_group_size
        assert cute.size(tdQrdQ_t2r_shape, mode=[1]) == expected_reduce_stages_t2r, (
            "dQaccum t2r reduce stage mismatch"
        )
        expected_reduce_stages = self.dQaccum_reduce_stage // self.cta_group_size
        # 2-CTA: CTA 0 -> (M/2, D) (stage 0, 1) & CTA 1 -> (M/2, D) (stage 2, 3)
        stage_offset = (
            expected_reduce_stages * cta_rank_in_cluster if const_expr(self.use_2cta_instrs) else 0
        )

        thr_copy_dQaccum_r2s = copy_utils.tiled_copy_1d(
            self.dqaccum_dtype,
            num_reduce_threads,
            num_copy_elems=128 // Float32.width,
        ).get_slice(tidx)
        sdQaccum_flat = cute.make_tensor(
            sdQaccum.iterator,
            cute.make_layout((self.tile_m * self.dQ_reduce_ncol, self.sdQaccum_stage)),
        )
        tdQsdQ = thr_copy_dQaccum_r2s.partition_D(sdQaccum_flat)
        sdQaccum_tma = cute.make_tensor(sdQaccum.iterator, sdQaccum_tma_layout)

        read_flag = const_expr(not self.deterministic)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        dQ_consumer_state = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        dQ_tma_store_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.sdQaccum_stage
        )
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            n_block_cta_group = n_block // self.cta_group_size  # for 2cta
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen, n_block_cta_group
            )
            if const_expr(tma_atom_dQ is not None):
                assert mdQaccum_tma_tensor is not None
                rows_per_head = mdQaccum.shape[0] // self.tile_hdim
                mdQaccum_tma_cur = cute.domain_offset(
                    (
                        head_idx * rows_per_head + seqlen.padded_offset_q,
                        0,
                    ),
                    mdQaccum_tma_tensor,
                )
                gdQaccum_tma = cute.local_tile(
                    mdQaccum_tma_cur,
                    self.sdQaccum_tma_tile,
                    (None, 0),
                )
                gdQaccum_tma_flat = cute.group_modes(gdQaccum_tma, 0, 2)
                tdQsdQ_tma, tdQgdQ_tma = cpasync.tma_partition(
                    tma_atom_dQ,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sdQaccum_tma, 0, 2),
                    gdQaccum_tma_flat,
                )
            else:
                if const_expr(not seqlen.has_cu_seqlens_q):
                    mdQaccum_cur = mdQaccum[None, head_idx, batch_idx]
                else:
                    dq_padded_offset = (
                        seqlen.padded_offset_q
                        if const_expr(self.broadcast_q)
                        else seqlen.padded_offset_o
                    )
                    mdQaccum_cur = cute.domain_offset(
                        (dq_padded_offset * self.tile_hdim,),
                        mdQaccum[None, head_idx],
                    )
                gdQaccum_ = cute.local_tile(
                    mdQaccum_cur,
                    (self.tile_m * self.tile_hdim,),
                    (None,),
                )
                gdQaccum = cute.flat_divide(
                    gdQaccum_,
                    (
                        self.tile_m
                        * self.tile_hdim
                        // self.dQaccum_reduce_stage,
                    ),
                )

            if const_expr(self.deterministic):
                assert mdQ_semaphore is not None
                mdQ_semaphore_cur = mdQ_semaphore[None, None, head_idx, batch_idx]

            delay_semaphore_release = not self.tile_hdim == 192 and not self.use_block_sparsity

            curr_q_cnt = Int32(0)
            curr_q_idx = None
            curr_full_cnt = Int32(0)
            curr_full_idx = None
            curr_dq_write_order = None
            curr_dq_write_order_full = None
            loop_count = m_block_max - m_block_min
            if const_expr(self.use_block_sparsity):
                assert blocksparse_tensors is not None
                (
                    curr_q_cnt,
                    curr_q_idx,
                    curr_full_cnt,
                    curr_full_idx,
                    loop_count,
                ) = get_block_sparse_iteration_info_bwd(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    n_block,
                    q_subtile_factor=self.q_subtile_factor,
                    m_block_max=m_block_max,
                )
                process_tile = loop_count > Int32(0)
            else:
                process_tile = (
                    const_expr(
                        not self.is_local and not self.is_varlen_q
                    )
                    or m_block_min < m_block_max
                )
                loop_count = m_block_max - m_block_min

            if const_expr(self.deterministic and self.use_block_sparsity):
                assert blocksparse_tensors is not None
                if const_expr(blocksparse_tensors.dq_write_order is not None):
                    assert blocksparse_tensors.dq_write_order is not None
                    curr_dq_write_order = blocksparse_tensors.dq_write_order[
                        batch_idx, head_idx, n_block, None
                    ]
                    if const_expr(
                        blocksparse_tensors.dq_write_order_full is not None
                    ):
                        assert blocksparse_tensors.dq_write_order_full is not None
                        curr_dq_write_order_full = (
                            blocksparse_tensors.dq_write_order_full[
                                batch_idx, head_idx, n_block, None
                            ]
                        )

            # dQacc_reduce mainloop
            # Block sparsity: iterate over sparse m_block count and derive actual m_block
            # from Q_IDX/FULL_Q_IDX tensors. Dense: iterate m_block_min..m_block_max directly.
            for iter_idx in cutlass.range(loop_count, unroll=1):
                m_block = m_block_min + iter_idx
                m_block_oob_upper = False
                if const_expr(self.use_block_sparsity):
                    m_block, _ = get_m_block_from_iter_bwd(
                        iter_idx,
                        curr_q_cnt,
                        curr_q_idx,
                        curr_full_cnt,
                        curr_full_idx,
                        q_subtile_factor=self.q_subtile_factor,
                        m_block_max=m_block_max,
                    )
                    m_block_oob_upper = m_block >= m_block_max
                pipeline_dQ.consumer_wait(dQ_consumer_state)
                # TMEM -> RMEM
                tdQrdQ_t2r = cute.make_rmem_tensor(tdQrdQ_t2r_shape, Float32)
                cute.copy(thr_copy_t2r, tdQtdQ_t2r, tdQrdQ_t2r)
                cute.arch.fence_view_async_tmem_load()
                cute.arch.sync_warp()
                if const_expr(self.dqaccum_dtype == cutlass.Float16):
                    # Keep dQ partials above FP16 underflow; postprocess removes this scale.
                    fp16_dq_scale = Float32(dq_accum_scale)
                    for v in cutlass.range_constexpr(cute.size(tdQrdQ_t2r) // 2):
                        tdQrdQ_t2r[2 * v], tdQrdQ_t2r[2 * v + 1] = (
                            cute.arch.mul_packed_f32x2(
                                (tdQrdQ_t2r[2 * v], tdQrdQ_t2r[2 * v + 1]),
                                (fp16_dq_scale, fp16_dq_scale),
                            )
                        )
                with cute.arch.elect_one():
                    pipeline_dQ.consumer_release(dQ_consumer_state)
                dQ_consumer_state.advance()

                if m_block_max > 0:
                    m_block = cutlass.min(m_block, m_block_max - 1)
                if const_expr(self.use_2cta_instrs):
                    tdQrdQ_shape = (
                        self.dQ_reduce_ncol,
                        self.tile_hdim
                        // self.cta_group_size
                        // self.dQ_reduce_ncol,
                    )
                    tdQrdQ = cute.make_tensor(
                        tdQrdQ_t2r.iterator, tdQrdQ_shape
                    )
                    num_reduce_stages = cute.size(tdQrdQ, mode=[1])
                else:
                    num_reduce_stages = cute.size(tdQrdQ_t2r, mode=[1])
                for stage in cutlass.range_constexpr(num_reduce_stages):
                    smem_idx = dQ_tma_store_producer_state.index
                    tdQsdQ_r2s = tdQsdQ[None, None, smem_idx]
                    if const_expr(self.use_2cta_instrs):
                        tdQrdQ_r2s = cute.make_tensor(
                            tdQrdQ[None, stage].iterator,
                            tdQsdQ_r2s.shape,
                        )
                    else:
                        tdQrdQ_r2s = cute.make_tensor(
                            tdQrdQ_t2r[None, stage, None, None].iterator,
                            tdQsdQ_r2s.shape,
                        )
                    if const_expr(self.deterministic and stage == 0):
                        if not m_block_oob_upper:
                            lock_value = self._dq_semaphore_lock_value(
                                iter_idx,
                                curr_q_cnt,
                                curr_dq_write_order,
                                curr_dq_write_order_full,
                                blocksparse_tensors,
                                block_info,
                                seqlen,
                                m_block,
                                n_block_cta_group,
                            )
                            barrier.wait_eq(
                                mdQ_semaphore_cur[(m_block, None)].iterator,
                                tidx,
                                cta_rank_in_cluster,
                                lock_value,
                            )
                    if const_expr(
                        tma_atom_dQ is not None
                        and self.dqaccum_dtype == cutlass.Float32
                    ):
                        cute.copy(
                            thr_copy_dQaccum_r2s,
                            tdQrdQ_r2s,
                            tdQsdQ_r2s,
                        )
                    else:
                        copy_utils.cvt_copy(
                            thr_copy_dQaccum_r2s,
                            tdQrdQ_r2s,
                            tdQsdQ_r2s,
                        )
                        cute.arch.fence_view_async_shared()
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.reduce_sync_barrier.arrive_and_wait()
                    if is_tma_warp and not m_block_oob_upper:
                        cute.arch.fence_proxy("async.shared", space="cta")
                        if const_expr(tma_atom_dQ is not None):
                            cute.copy(
                                tma_atom_dQ,
                                tdQsdQ_tma[None, smem_idx],
                                tdQgdQ_tma[
                                    None,
                                    m_block * self.dQaccum_reduce_stage + stage,
                                ],
                            )
                        else:
                            gdQaccum_cur = gdQaccum[None, None, m_block]
                            with cute.arch.elect_one():
                                if const_expr(
                                    self.dqaccum_dtype == cutlass.Float16
                                ):
                                    copy_utils.cpasync_reduce_bulk_add_f16(
                                        sdQaccum_flat[None, smem_idx].iterator,
                                        gdQaccum_cur[
                                            None, stage + stage_offset
                                        ].iterator,
                                        self.tma_copy_bytes["dQ"],
                                    )
                                else:
                                    copy_utils.cpasync_reduce_bulk_add_f32(
                                        sdQaccum_flat[None, smem_idx].iterator,
                                        gdQaccum_cur[
                                            None, stage + stage_offset
                                        ].iterator,
                                        self.tma_copy_bytes["dQ"],
                                    )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(
                            self.sdQaccum_stage - 1, read=read_flag
                        )
                    elif is_tma_warp:
                        # Drain pending stores before reusing the shared buffer.
                        cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                    dQ_tma_store_producer_state.advance()
                    self.reduce_sync_barrier.arrive_and_wait()
                    # semaphore release for prior m_block
                    if const_expr(
                        self.deterministic and stage == 0 and delay_semaphore_release
                    ):
                        if m_block > m_block_min:
                            barrier.arrive_inc(
                                mdQ_semaphore_cur[(m_block - 1, None)].iterator,
                                tidx,
                                cta_rank_in_cluster,
                                1,
                            )


                if const_expr(self.tile_hdim == 192):
                    if const_expr(self.sdQaccum_stage > 1):
                        if is_tma_warp:
                            cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                        self.reduce_sync_barrier.arrive_and_wait()
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(dQaccum_empty_mbar_ptr)

                # semaphore release
                # NOTE: arrive_inc calls red_release which issues membar
                if const_expr(self.deterministic and not delay_semaphore_release):
                    if const_expr(self.sdQaccum_stage > 1 and not self.tile_hdim == 192):
                        if is_tma_warp and not m_block_oob_upper:
                            cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                        self.reduce_sync_barrier.arrive_and_wait()
                    if not m_block_oob_upper:
                        barrier.arrive_inc(
                            mdQ_semaphore_cur[m_block, None].iterator, tidx, cta_rank_in_cluster, 1
                        )

            if process_tile:
                if is_tma_warp:
                    # The staged loop waits only for source reads to preserve
                    # overlap. Fully drain global reductions before this CTA
                    # advances to another tile or exits.
                    cute.arch.cp_async_bulk_wait_group(0, read=False)
                self.reduce_sync_barrier.arrive_and_wait()
                # final semaphore release
                if const_expr(self.deterministic and delay_semaphore_release):
                    barrier.arrive_inc(
                        mdQ_semaphore_cur[(m_block_max - 1, None)].iterator,
                        tidx,
                        cta_rank_in_cluster,
                        1,
                    )

            if const_expr(
                self.deterministic
                and not self.spt
                and not self.use_block_sparsity
                and block_info.window_size_left is not None
            ):
                m_block_global_max = cute.ceil_div(seqlen.seqlen_q, self.tile_m)
                for m_block in cutlass.range(m_block_max, m_block_global_max, unroll=1):
                    barrier.arrive_inc(
                        mdQ_semaphore_cur[(m_block, None)].iterator, tidx, cta_rank_in_cluster, 1
                    )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def epilogue_dKV(
        self,
        tidx: Int32,
        warp_idx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        seqlen,
        thr_mma_dV: cute.ThrMma,
        thr_mma_dK: cute.ThrMma,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        pipeline_dKV: PipelineAsync,
        consumer_state_dKV: cutlass.pipeline.PipelineState,
        softmax_scale: Float32,
        mdSFK_out: Optional[cute.Tensor],
        mdSFV_out: Optional[cute.Tensor],
        mdV_tma_tensor: Optional[cute.Tensor] = None,
        mdK_tma_tensor: Optional[cute.Tensor] = None,
        sdV_mxfp8_tma: Optional[cute.Tensor] = None,
        sdK_mxfp8_tma: Optional[cute.Tensor] = None,
        tma_atom_dV: Optional[cute.CopyAtom] = None,
        tma_atom_dK: Optional[cute.CopyAtom] = None,
        tiled_copy_r2s_dKV: Optional[cute.TiledCopy] = None,
    ):
        wg_idx = (
            cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))
        ) // 128
        num_wg = cute.arch.WARP_SIZE * len(self.compute_warp_ids) // 128

        if const_expr(self.use_dedicated_mxfp8_dkv_tma):
            assert mdV_tma_tensor is not None
            assert mdK_tma_tensor is not None
            assert sdV_mxfp8_tma is not None
            assert sdK_mxfp8_tma is not None
            assert tma_atom_dV is not None
            assert tma_atom_dK is not None
            assert tiled_copy_r2s_dKV is not None
            leader_warp = (
                cute.arch.make_warp_uniform(warp_idx) % 4
            ) == 0
            barrier_id = int(NamedBarrierBwdSm100.EpilogueWG1) + wg_idx
            if leader_warp:
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.barrier_arrive(
                    barrier_id=barrier_id,
                    number_of_threads=128 + cute.arch.WARP_SIZE,
                )
            cute.arch.barrier(
                barrier_id=barrier_id,
                number_of_threads=128 + cute.arch.WARP_SIZE,
            )

        assert self.qhead_per_kvhead == 1, "This epilogue path is only for MHA"
        mdV_cur = mdK_cur = None
        if const_expr(not self.use_dedicated_mxfp8_dkv_tma):
            mdV_cur = seqlen.offset_batch_K(mdV, batch_idx, dim=3)[
                None, None, head_idx
            ]
            mdK_cur = seqlen.offset_batch_K(mdK, batch_idx, dim=3)[
                None, None, head_idx
            ]

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)), Float32
        )
        # dV
        pipeline_dKV.consumer_wait(consumer_state_dKV)

        tiled_tmem_ld_dV = tcgen05.make_tmem_copy(tmem_load_atom, tdVtdV)
        thr_tmem_ld_dV = tiled_tmem_ld_dV.get_slice(tidx)

        tdVtdV_t2r_p = thr_tmem_ld_dV.partition_S(tdVtdV)
        tdVtdV_t2r = self.split_wg(tdVtdV_t2r_p, wg_idx, num_wg)

        cdV = cute.make_identity_tensor((self.mma_tiler_pdo[0], self.mma_tiler_pdo[1]))
        tdVcdV = thr_mma_dV.partition_C(cdV)
        tdVcdV_tensor = cute.make_tensor(tdVcdV.iterator, tdVcdV.layout)

        tdVcdV_t2r_p = thr_tmem_ld_dV.partition_D(tdVcdV_tensor)
        tdVcdV_t2r = self.split_wg(tdVcdV_t2r_p, wg_idx, num_wg)
        tdVrdV_t2r = cute.make_rmem_tensor(tdVcdV_t2r.shape, Float32)

        cute.copy(thr_tmem_ld_dV, tdVtdV_t2r, tdVrdV_t2r)
        cute.arch.fence_view_async_tmem_load()

        universal_copy_bits = 128
        tiled_gmem_store_dV = None
        if const_expr(not self.use_dedicated_mxfp8_dkv_tma):
            atom_universal_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dv_dtype,
                num_bits_per_copy=universal_copy_bits,
            )
            layout_tv_dV = tiled_tmem_ld_dV.layout_dst_tv_tiled
            if const_expr(self.output_mxfp8_dkv):
                num_rep_dV = 16
                elems_per_copy_dV = universal_copy_bits // self.dv_dtype.width
                layout_tv_dV = cute.make_layout(
                    ((32, elems_per_copy_dV, num_wg), (num_rep_dV, 32)),
                    stride=(
                        (0, 1, elems_per_copy_dV * num_rep_dV),
                        (
                            elems_per_copy_dV,
                            elems_per_copy_dV * num_rep_dV * num_wg,
                        ),
                    ),
                )
            tiled_gmem_store_dV = cute.make_tiled_copy(
                atom_universal_copy,
                layout_tv=layout_tv_dV,
                tiler_mn=tiled_tmem_ld_dV.tiler_mn,
            )

        tdVrdV_r2s = cute.make_rmem_tensor(tdVrdV_t2r.shape, self.dv_dtype)
        if const_expr(self.output_mxfp8_dkv):
            mdSFV_cur = seqlen.offset_batch_K(mdSFV_out, batch_idx, dim=3)[
                None, None, head_idx
            ]
            global_row_dV = n_block * Int32(self.tile_n) + tidx
            sf_row_valid_dV = global_row_dV < seqlen.seqlen_k
            flat_dV = cute.make_rmem_tensor(tdVrdV_t2r.shape, Float32)
            flat_dV.store(tdVrdV_t2r.load())
            total_elems_dV = cute.size(flat_dV.shape)
            for sf_blk in cutlass.range_constexpr(total_elems_dV // self.sf_vec_size):
                base = sf_blk * self.sf_vec_size
                block_amax_dV = Float32(0.0)
                for k in cutlass.range_constexpr(self.sf_vec_size):
                    block_amax_dV = fused_abs_max_f32(
                        block_amax_dV, flat_dV[base + k]
                    )
                block_amax_dV_for_reduce = Float32(0.0)
                if sf_row_valid_dV:
                    block_amax_dV_for_reduce = block_amax_dV
                square_amax_dV = redux_sync_max_abs_f32(
                    block_amax_dV_for_reduce
                )
                inv_scale_dV, sf_u32_dV = fused_amax_to_e8m0_scale_f32(
                    square_amax_dV, Float32(E4M3_MAX_NORM_RCP)
                )
                for k in cutlass.range_constexpr(self.sf_vec_size // 2):
                    flat_dV[base + 2 * k], flat_dV[base + 2 * k + 1] = (
                        cute.arch.mul_packed_f32x2(
                            (flat_dV[base + 2 * k], flat_dV[base + 2 * k + 1]),
                            (inv_scale_dV, inv_scale_dV),
                        )
                    )
                col_base_dV = Int32(
                    tdVcdV_t2r[((0, 0), sf_blk * 2, 0, 0)][1]
                )
                if sf_row_valid_dV:
                    mdSFV_cur[(global_row_dV, col_base_dV >> Int32(5))] = Uint8(
                        sf_u32_dV & Uint32(0xFF)
                    )
            utils.cvt_fp8(flat_dV, tdVrdV_r2s)
        else:
            for i in cutlass.range_constexpr(cute.size(tdVrdV_t2r, mode=[1])):
                dV_vec = tdVrdV_t2r[(None, i, 0, 0)].load()
                tdVrdV_r2s[(None, i, 0, 0)].store(dV_vec.to(self.dv_dtype))

        tdVgdV_r2g = None
        if const_expr(not self.use_dedicated_mxfp8_dkv_tma):
            assert mdV_cur is not None
            gdV = cute.local_tile(
                mdV_cur,
                (self.mma_tiler_pdo[0], self.tile_hdimv),
                (None, 0),
            )
            gdV_tile = gdV[None, None, n_block // self.cta_group_size]
            tdVgdV = thr_mma_dV.partition_C(gdV_tile)
            tdVgdV_r2g_p = thr_tmem_ld_dV.partition_D(tdVgdV)
            tdVgdV_r2g = self.split_wg(tdVgdV_r2g_p, wg_idx, num_wg)

        if const_expr(self.use_dedicated_mxfp8_dkv_tma):
            self.store_mxfp8_dKV_tma(
                tidx,
                warp_idx,
                batch_idx,
                head_idx,
                n_block,
                seqlen,
                mdV_tma_tensor,
                sdV_mxfp8_tma,
                tdVrdV_r2s,
                tma_atom_dV,
                tiled_copy_r2s_dKV,
                "V",
            )
        elif tidx < seqlen.seqlen_k - self.tile_n * n_block:
            assert tiled_gmem_store_dV is not None
            assert tdVgdV_r2g is not None
            cute.copy(tiled_gmem_store_dV, tdVrdV_r2s, tdVgdV_r2g)

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()

        # dK
        pipeline_dKV.consumer_wait(consumer_state_dKV)

        tiled_tmem_ld_dK = tcgen05.make_tmem_copy(tmem_load_atom, tdKtdK)
        thr_tmem_ld_dK = tiled_tmem_ld_dK.get_slice(tidx)

        tdKtdK_t2r_p = thr_tmem_ld_dK.partition_S(tdKtdK)
        tdKtdK_t2r = self.split_wg(tdKtdK_t2r_p, wg_idx, num_wg)

        cdK = cute.make_identity_tensor((self.mma_tiler_dsq[0], self.mma_tiler_dsq[1]))
        tdKcdK = thr_mma_dK.partition_C(cdK)
        tdKcdK_tensor = cute.make_tensor(tdKcdK.iterator, tdKcdK.layout)

        tdKcdK_t2r_p = thr_tmem_ld_dK.partition_D(tdKcdK_tensor)
        tdKcdK_t2r = self.split_wg(tdKcdK_t2r_p, wg_idx, num_wg)
        tdKrdK_t2r = cute.make_rmem_tensor(tdKcdK_t2r.shape, Float32)

        cute.copy(tiled_tmem_ld_dK, tdKtdK_t2r, tdKrdK_t2r)
        cute.arch.fence_view_async_tmem_load()

        universal_copy_bits = 128
        tiled_gmem_store_dK = None
        if const_expr(not self.use_dedicated_mxfp8_dkv_tma):
            atom_universal_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dk_dtype,
                num_bits_per_copy=universal_copy_bits,
            )
            layout_tv_dK = tiled_tmem_ld_dK.layout_dst_tv_tiled
            if const_expr(self.output_mxfp8_dkv):
                num_rep_dK = 16
                elems_per_copy_dK = universal_copy_bits // self.dk_dtype.width
                layout_tv_dK = cute.make_layout(
                    ((32, elems_per_copy_dK, num_wg), (num_rep_dK, 32)),
                    stride=(
                        (0, 1, elems_per_copy_dK * num_rep_dK),
                        (
                            elems_per_copy_dK,
                            elems_per_copy_dK * num_rep_dK * num_wg,
                        ),
                    ),
                )
            tiled_gmem_store_dK = cute.make_tiled_copy(
                atom_universal_copy,
                layout_tv=layout_tv_dK,
                tiler_mn=tiled_tmem_ld_dK.tiler_mn,
            )

        tdKrdK_r2s = cute.make_rmem_tensor(tdKrdK_t2r.shape, self.dk_dtype)
        if const_expr(self.output_mxfp8_dkv):
            mdSFK_cur = seqlen.offset_batch_K(mdSFK_out, batch_idx, dim=3)[
                None, None, head_idx
            ]
            global_row_dK = n_block * Int32(self.tile_n) + tidx
            sf_row_valid_dK = global_row_dK < seqlen.seqlen_k
            flat_dK = cute.make_rmem_tensor(tdKrdK_t2r.shape, Float32)
            flat_dK.store(tdKrdK_t2r.load())
            total_elems_dK = cute.size(flat_dK.shape)
            for k in cutlass.range_constexpr(total_elems_dK // 2):
                flat_dK[2 * k], flat_dK[2 * k + 1] = cute.arch.mul_packed_f32x2(
                    (flat_dK[2 * k], flat_dK[2 * k + 1]),
                    (softmax_scale, softmax_scale),
                )
            for sf_blk in cutlass.range_constexpr(total_elems_dK // self.sf_vec_size):
                base = sf_blk * self.sf_vec_size
                block_amax_dK = Float32(0.0)
                for k in cutlass.range_constexpr(self.sf_vec_size):
                    block_amax_dK = fused_abs_max_f32(
                        block_amax_dK, flat_dK[base + k]
                    )
                block_amax_dK_for_reduce = Float32(0.0)
                if sf_row_valid_dK:
                    block_amax_dK_for_reduce = block_amax_dK
                square_amax_dK = redux_sync_max_abs_f32(
                    block_amax_dK_for_reduce
                )
                inv_scale_dK, sf_u32_dK = fused_amax_to_e8m0_scale_f32(
                    square_amax_dK, Float32(E4M3_MAX_NORM_RCP)
                )
                for k in cutlass.range_constexpr(self.sf_vec_size // 2):
                    flat_dK[base + 2 * k], flat_dK[base + 2 * k + 1] = (
                        cute.arch.mul_packed_f32x2(
                            (flat_dK[base + 2 * k], flat_dK[base + 2 * k + 1]),
                            (inv_scale_dK, inv_scale_dK),
                        )
                    )
                col_base_dK = Int32(
                    tdKcdK_t2r[((0, 0), sf_blk * 2, 0, 0)][1]
                )
                if sf_row_valid_dK:
                    mdSFK_cur[(global_row_dK, col_base_dK >> Int32(5))] = Uint8(
                        sf_u32_dK & Uint32(0xFF)
                    )
            utils.cvt_fp8(flat_dK, tdKrdK_r2s)
        else:
            for i in cutlass.range_constexpr(cute.size(tdKrdK_t2r, mode=[1])):
                dK_vec = tdKrdK_t2r[(None, i, 0, 0)].load() * softmax_scale
                tdKrdK_r2s[(None, i, 0, 0)].store(dK_vec.to(self.dk_dtype))

        tdKgdK_r2g = None
        if const_expr(not self.use_dedicated_mxfp8_dkv_tma):
            assert mdK_cur is not None
            gdK = cute.local_tile(
                mdK_cur,
                (self.mma_tiler_dsq[0], self.tile_hdim),
                (None, 0),
            )
            gdK_tile = gdK[None, None, n_block // self.cta_group_size]
            tdKgdK = thr_mma_dK.partition_C(gdK_tile)
            tdKgdK_r2g_p = thr_tmem_ld_dK.partition_D(tdKgdK)
            tdKgdK_r2g = self.split_wg(tdKgdK_r2g_p, wg_idx, num_wg)

        if const_expr(self.use_dedicated_mxfp8_dkv_tma):
            self.store_mxfp8_dKV_tma(
                tidx,
                warp_idx,
                batch_idx,
                head_idx,
                n_block,
                seqlen,
                mdK_tma_tensor,
                sdK_mxfp8_tma,
                tdKrdK_r2s,
                tma_atom_dK,
                tiled_copy_r2s_dKV,
                "K",
            )
        elif tidx < seqlen.seqlen_k - self.tile_n * n_block:
            assert tiled_gmem_store_dK is not None
            assert tdKgdK_r2g is not None
            cute.copy(tiled_gmem_store_dK, tdKrdK_r2s, tdKgdK_r2g)

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()
        return consumer_state_dKV

    @cute.jit
    def store_mxfp8_dKV_tma(
        self,
        tidx: Int32,
        warp_idx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        seqlen,
        mdKV_tma_tensor: cute.Tensor,
        sdKV_mxfp8_tma: cute.Tensor,
        tdKVrdKV: cute.Tensor,
        tma_atom_dKV: cute.CopyAtom,
        tiled_copy_r2s_dKV: cute.TiledCopy,
        K_or_V: cutlass.Constexpr[str],
    ):
        assert K_or_V in ("K", "V")
        tile_hdim = self.tile_hdim if const_expr(K_or_V == "K") else self.tile_hdimv
        epi_tile = self.sdK_epi_tile if const_expr(K_or_V == "K") else self.sdV_epi_tile
        num_compute_threads = cute.arch.WARP_SIZE * len(self.compute_warp_ids)
        wg_idx = (cute.arch.thread_idx()[0] % num_compute_threads) // 128
        num_wg = num_compute_threads // 128
        leader_warp = (cute.arch.make_warp_uniform(warp_idx) % 4) == 0
        barrier_id = int(NamedBarrierBwdSm100.EpilogueWG1) + wg_idx

        sdKV_wg = sdKV_mxfp8_tma[None, None, wg_idx]
        thr_copy_r2s_dKV = tiled_copy_r2s_dKV.get_slice(tidx)
        tdKVsdKV_r2s = thr_copy_r2s_dKV.partition_D(sdKV_wg)
        assert cute.size(tdKVrdKV) == cute.size(tdKVsdKV_r2s)
        tdKVrdKV_r2s = cute.make_tensor(tdKVrdKV.iterator, tdKVsdKV_r2s.shape)
        cute.copy(thr_copy_r2s_dKV, tdKVrdKV_r2s, tdKVsdKV_r2s)
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=128)

        head_idx_kv = head_idx // self.qhead_per_kvhead
        mdKV_cur = seqlen.offset_batch_K(mdKV_tma_tensor, batch_idx, dim=3)[
            None, None, head_idx_kv
        ]
        gdKV = cute.local_tile(
            mdKV_cur,
            (self.tile_n, tile_hdim),
            (n_block, 0),
        )
        gdKV_wg = self.split_wg(gdKV, wg_idx, num_wg)
        gdKV_epi = cute.local_tile(gdKV_wg, epi_tile, (0, None))
        tdKVsdKV, tdKVgdKV = cpasync.tma_partition(
            tma_atom_dKV,
            0,
            cute.make_layout(1),
            cute.group_modes(sdKV_wg, 0, 2),
            cute.group_modes(gdKV_epi, 0, 2),
        )
        assert len(tdKVsdKV.shape) == 1
        assert len(tdKVgdKV.shape) == 2
        assert cute.size(tdKVgdKV.shape[1]) == 1

        if leader_warp:
            cute.copy(tma_atom_dKV, tdKVsdKV, tdKVgdKV[None, 0])
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.barrier_arrive(
                barrier_id=barrier_id,
                number_of_threads=128 + cute.arch.WARP_SIZE,
            )
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(
            barrier_id=barrier_id,
            number_of_threads=128 + cute.arch.WARP_SIZE,
        )

    @cute.jit
    def epilogue_dKV_tma_pair(
        self,
        tidx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        seqlen,
        thr_mma_dV: cute.ThrMma,
        thr_mma_dK: cute.ThrMma,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        sdV: cute.Tensor,
        sdK: cute.Tensor,
        tma_atom_dV: cute.CopyAtom,
        tma_atom_dK: cute.CopyAtom,
        tiled_copy_r2s_dKV: cute.TiledCopy,
        pipeline_dKV: PipelineAsync,
        consumer_state_dKV: cutlass.pipeline.PipelineState,
        softmax_scale: Float32,
        mdV_semaphore: Optional[cute.Tensor],
        mdK_semaphore: Optional[cute.Tensor],
    ) -> cutlass.pipeline.PipelineState:
        thr_copy_r2s_dKV = tiled_copy_r2s_dKV.get_slice(tidx)
        consumer_state_dKV = self.epilogue_dK_or_dV_tma(
            tidx,
            batch_idx,
            head_idx,
            n_block,
            seqlen,
            thr_mma_dV,
            tdVtdV,
            mdV,
            sdV,
            tma_atom_dV,
            thr_copy_r2s_dKV,
            pipeline_dKV,
            consumer_state_dKV,
            None,
            int(NamedBarrierBwdSm100.EpilogueWG1),
            mdV_semaphore,
            "V",
        )
        return self.epilogue_dK_or_dV_tma(
            tidx,
            batch_idx,
            head_idx,
            n_block,
            seqlen,
            thr_mma_dK,
            tdKtdK,
            mdK,
            sdK,
            tma_atom_dK,
            thr_copy_r2s_dKV,
            pipeline_dKV,
            consumer_state_dKV,
            softmax_scale if const_expr(not self.dKV_postprocess) else None,
            int(NamedBarrierBwdSm100.EpilogueWG1),
            mdK_semaphore,
            "K",
        )

    @cute.jit
    def epilogue_dK_or_dV_tma(
        self,
        tidx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        seqlen,
        thr_mma: cute.ThrMma,
        tdKVtdKV: cute.Tensor,
        mdKV: cute.Tensor,
        sdKV: cute.Tensor,
        tma_atom_dKV: cute.CopyAtom,
        thr_copy_r2s_dKV: cute.TiledCopy,
        pipeline_dKV: PipelineAsync,
        consumer_state_dKV: cutlass.pipeline.PipelineState,
        scale: Optional[Float32],
        barrier_id: Int32,
        mdKV_semaphore: Optional[cute.Tensor],
        K_or_V: cutlass.Constexpr[str],
    ) -> cutlass.pipeline.PipelineState:
        assert K_or_V in ("K", "V")
        tile_hdim = self.tile_hdim if const_expr(K_or_V == "K") else self.tile_hdimv
        dtype = self.dk_dtype if const_expr(K_or_V == "K") else self.dv_dtype
        epi_tile = self.sdK_epi_tile if const_expr(K_or_V == "K") else self.sdV_epi_tile
        flat_epi_tile = (
            self.sdK_flat_epi_tile if const_expr(K_or_V == "K") else self.sdV_flat_epi_tile
        )
        num_compute_threads = cute.arch.WARP_SIZE * len(self.compute_warp_ids)
        wg_idx = (cute.arch.thread_idx()[0] % num_compute_threads) // 128
        num_wg = num_compute_threads // 128
        leader_warp = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) == 0

        cta_group_tile_n = const_expr(self.tile_n * self.cta_group_size)

        if const_expr(not self.dKV_postprocess):
            sdKV = sdKV[None, None, wg_idx]  # (tile_n, 64) for bf16
        else:
            sdKV = sdKV[None, wg_idx]  # (tile_n * 32) for fp32

        # (8, tile_n / 128, 64 / 8) = (8, 1, 8) or (4, tile_n * 32 / (128 * 4)) = (4, 8)
        tdKVsdKV_r2s = thr_copy_r2s_dKV.partition_D(sdKV)

        head_idx_kv = head_idx // self.qhead_per_kvhead
        if const_expr(not self.dKV_postprocess):
            if const_expr(self.bf16_broadcast_q):
                assert seqlen.has_cu_seqlens_k
                mdKV_cur = seqlen.offset_batch_K(mdKV, batch_idx, dim=3)[
                    None, None, head_idx_kv
                ]
            else:
                assert not seqlen.has_cu_seqlens_k, "varlen uses non tma store path"
                mdKV_cur = mdKV[None, None, head_idx_kv, batch_idx]
            gdKV_p = cute.local_tile(
                mdKV_cur, (self.tile_n, tile_hdim), (n_block, 0)
            )  # (tile_n, hdim) - per CTA
            gdKV = self.split_wg(gdKV_p, wg_idx, num_wg)  # (tile_n, hdim / 2)
            gdKV_epi = cute.local_tile(
                gdKV, epi_tile, (0, None)
            )  # (tile_n, 64, epi_stage = (hdim / 2) / 64)
        else:
            # n_block_group = n_block // self.cta_group_size
            if const_expr(not seqlen.has_cu_seqlens_k):
                mdKV_cur = mdKV[None, head_idx_kv, batch_idx]  # (seqlen * hdim)
            else:
                mdKV_cur = cute.domain_offset(
                    (seqlen.padded_offset_k * tile_hdim,), mdKV[None, head_idx_kv]
                )
            gdKV_p = cute.local_tile(
                mdKV_cur, (self.tile_n * tile_hdim,), (n_block,)
            )  # (tile_n * hdim)
            gdKV = cute.logical_divide(gdKV_p, (self.tile_n * tile_hdim // num_wg,))[
                ((None, wg_idx),)
            ]  # (tile_n * hdim / 2)
            gdKV_epi = cute.flat_divide(
                gdKV, (flat_epi_tile,)
            )  # (tile_n * hdim / 2 / epi_stage, epi_stage)

        deterministic_KV = self.deterministic and self.qhead_per_kvhead > 1
        if const_expr(deterministic_KV):
            assert mdKV_semaphore is not None
            mdKV_semaphore_cur = mdKV_semaphore[n_block, None, head_idx_kv, batch_idx]

        if const_expr(not self.dKV_postprocess):
            tdKVsdKV, tdKVgdKV = cpasync.tma_partition(
                tma_atom_dKV,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sdKV, 0, 2),
                cute.group_modes(gdKV_epi, 0, 2),
            )  # (TMA) and (TMA, EPI_STAGE)
            assert len(tdKVsdKV.shape) == 1, "Wrong rank for SMEM fragment tdKVsdKV"
            assert len(tdKVgdKV.shape) == 2, "Wrong rank for GMEM fragment tdKVgdKV"
            num_epi_stages = cute.size(tdKVgdKV.shape[1])
            if const_expr(K_or_V == "K"):
                assert num_epi_stages == self.num_epi_stages, "Epi stage calculation is wrong (K)"
            else:
                assert num_epi_stages == self.num_epi_stages_v, "Epi stage calculation is wrong (V)"
        else:
            num_epi_stages = (
                self.num_epi_stages if const_expr(K_or_V == "K") else self.num_epi_stages_v
            )

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dK_reduce_ncol)), Float32
        )

        read_flag = const_expr(not deterministic_KV)

        pipeline_dKV.consumer_wait(consumer_state_dKV)

        # semaphore acquire
        if const_expr(deterministic_KV):
            barrier.wait_eq(
                mdKV_semaphore_cur.iterator, tidx, wg_idx, head_idx % self.qhead_per_kvhead
            )
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)

        for epi_stage in cutlass.range_constexpr(num_epi_stages):
            # TMEM -> RMEM -- setup
            thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV).get_slice(tidx)
            tdKVtdKV_t2r_p = thr_copy_t2r.partition_S(tdKVtdKV)
            tdKVtdKV_t2r = self.split_wg(tdKVtdKV_t2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVtdKV_t2r = tdKVtdKV_t2r[None, epi_stage]

            cdKV = cute.make_identity_tensor((cta_group_tile_n, tile_hdim))
            tdKVcdKV = thr_mma.partition_C(cdKV)
            tdKVcdKV_t2r_p = thr_copy_t2r.partition_D(tdKVcdKV)
            tdKVcdKV_t2r = self.split_wg(tdKVcdKV_t2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVcdKV_t2r = tdKVcdKV_t2r[None, epi_stage]

            tdKVrdKV_t2r = cute.make_rmem_tensor(tdKVcdKV_t2r.shape, Float32)

            assert cute.size(tdKVrdKV_t2r) == cute.size(tdKVtdKV_t2r) // cute.arch.WARP_SIZE, (
                "RMEM<->TMEM fragment size mismatch"
            )

            # TMEM -> RMEM -- copy and fence
            cute.copy(thr_copy_t2r, tdKVtdKV_t2r, tdKVrdKV_t2r)
            cute.arch.fence_view_async_tmem_load()

            # RMEM -- scale and convert
            if const_expr(scale is not None):
                for i in cutlass.range(cute.size(tdKVrdKV_t2r.shape) // 2, unroll_full=True):
                    tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1] = cute.arch.mul_packed_f32x2(
                        (tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1]), (scale, scale)
                    )
            tdKVrdKV = cute.make_rmem_tensor(tdKVrdKV_t2r.shape, dtype)  # (32 columns)
            tdKVrdKV.store(tdKVrdKV_t2r.load().to(dtype))

            # RMEM -> SMEM -- copy, fence and barrier
            tdKVrdKV_r2s = cute.make_tensor(tdKVrdKV.iterator, tdKVsdKV_r2s.shape)
            cute.copy(thr_copy_r2s_dKV, tdKVrdKV_r2s, tdKVsdKV_r2s)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)

            # SMEM -> GMEM
            if leader_warp:
                if const_expr(not self.dKV_postprocess):
                    cute.copy(tma_atom_dKV, tdKVsdKV, tdKVgdKV[None, epi_stage])
                else:
                    with cute.arch.elect_one():
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sdKV.iterator,
                            gdKV_epi[None, epi_stage].iterator,
                            self.tma_copy_bytes["dKacc"],
                        )
                if const_expr(
                    epi_stage < num_epi_stages - 1 or self.bf16_broadcast_q
                ):
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                cute.arch.barrier_arrive(
                    barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE
                )

            # Barrier since all warps need to wait for SMEM to be freed
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE
            )

        # semaphore release
        # NOTE: arrive_inc calls red_release which issues membar
        if const_expr(deterministic_KV):
            if leader_warp:
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)
            barrier.arrive_inc(mdKV_semaphore_cur.iterator, tidx, wg_idx, 1)

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()
        return consumer_state_dKV
