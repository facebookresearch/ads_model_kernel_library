# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors

"""
TLX fused RMSNorm + Matmul kernel for Blackwell.

Computes C = diag(rrms) @ (A @ B) where rrms = 1/sqrt(mean(A^2, dim=-1) + eps).
The x^2 accumulation is piggybacked in the epilogue warp during the K-loop,
so the RMSNorm is essentially free on top of the matmul.

To apply a norm weight w, pre-absorb it into B: B_w = w[:, None] * B.
"""

import functools
import math
import os

import torch
import triton  # @manual=//triton:triton
import triton.language as tl  # @manual=//triton:triton
import triton.language.extra.tlx as tlx  # @manual=//triton:triton
import triton.profiler as proton  # @manual=//triton:triton
import triton.profiler.language as pl  # @manual=//triton:triton
from triton.tools.tensor_descriptor import TensorDescriptor  # @manual=//triton:triton

DISABLE_AUTOTUNE = os.environ.get("ADS_MKL_DISABLE_AUTOTUNE") == "1"

# Track which (M, N, K) shapes have already printed their heuristic config
_printed_heuristic_configs = set()


# Cached SM count — never changes during program lifetime.
# Calling torch.cuda.get_device_properties() on every matmul() call
# adds measurable overhead that degrades benchmark throughput on fast kernels.
@functools.lru_cache(maxsize=1)
def _get_num_sms():
    return torch.cuda.get_device_properties("cuda").multi_processor_count


def get_heuristic_config(M, N, K, num_sms=148):  # noqa: C901
    """
    Select optimal GEMM config based on problem shape characteristics.

    The selection uses shape-characteristic rules (not exact shape matching):
    1. M/N ratio determines tile shape preference
    2. MN tiles vs SM count determines parallelization strategy (Split-K vs data-parallel)
    3. Arithmetic intensity determines pipeline depth

    Args:
        M, N, K: GEMM dimensions (A is MxK, B is KxN, C is MxN)
        num_sms: Number of SMs on the GPU (default 148 for B200)

    Returns:
        dict: Configuration parameters for the TLX GEMM kernel
    """
    MAX_SMEM = 232 * 1024  # 232KB shared memory limit
    MAX_TMEM_COLUMNS = 512  # TLX tmem column allocation limit per SM

    # ==========================================================================
    # Shape-characteristic analysis
    # ==========================================================================
    mn_ratio = M / max(N, 1)
    is_tall_m = mn_ratio > 4  # M much larger than N
    is_tall_n = mn_ratio < 0.25  # N much larger than M

    # Estimate MN tiles with representative tile sizes
    # Use 256x128 for tall-M, 128x256 for tall-N, 256x256 for balanced
    if is_tall_m:
        ref_bm, ref_bn = 256, 128
    elif is_tall_n:
        ref_bm, ref_bn = 128, 256
    else:
        ref_bm, ref_bn = 256, 256

    num_tiles_m = math.ceil(M / ref_bm)
    num_tiles_n = math.ceil(N / ref_bn)
    num_mn_tiles = num_tiles_m * num_tiles_n

    is_gpu_saturated = num_mn_tiles >= num_sms
    is_undersaturated = num_mn_tiles < num_sms

    # ==========================================================================
    # Shape-characteristic config selection
    # ==========================================================================

    # Characteristic 1: Tall-M shapes benefit from 2-CTA B-tile sharing
    # When M >> N, adjacent M-tiles can share B via 2-CTA clusters
    # Use arithmetic intensity to select tile shape, and K size to select BLOCK_K
    if is_tall_m and is_gpu_saturated:
        arithmetic_intensity = K / max(min(M, N), 1)
        # For low arithmetic intensity (memory-bound), use narrower tiles with larger BLOCK_K
        if arithmetic_intensity <= 1.5:
            return {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": max(2, _select_group_size_m(M, N, 256)),
                "NUM_SMEM_BUFFERS": 2,
                "NUM_TMEM_BUFFERS": 2,
                "NUM_MMA_GROUPS": 2,
                "EPILOGUE_SUBTILE": 1,
                "NUM_CTAS": 2,
                "SPLIT_K": 1,
                "INTERLEAVE_EPILOGUE": 1,
                "ctas_per_cga": (2, 1, 1),
                "pre_hook": matmul_tma_set_block_size_hook,
            }
        else:
            # High arithmetic intensity: use wider tiles
            # For large K, use BLOCK_K=128 to reduce K-iterations
            # For smaller K, use BLOCK_K=64 with more SMEM buffers
            if K > N * 2:
                return {
                    "BLOCK_SIZE_M": 256,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": max(2, _select_group_size_m(M, N, 256)),
                    "NUM_SMEM_BUFFERS": 2,
                    "NUM_TMEM_BUFFERS": 1,
                    "NUM_MMA_GROUPS": 2,
                    "EPILOGUE_SUBTILE": 4,
                    "NUM_CTAS": 2,
                    "SPLIT_K": 1,
                    "INTERLEAVE_EPILOGUE": 0,
                    "ctas_per_cga": (2, 1, 1),
                    "pre_hook": matmul_tma_set_block_size_hook,
                }
            else:
                return {
                    "BLOCK_SIZE_M": 256,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": max(2, _select_group_size_m(M, N, 256)),
                    "NUM_SMEM_BUFFERS": 4,
                    "NUM_TMEM_BUFFERS": 1,
                    "NUM_MMA_GROUPS": 2,
                    "EPILOGUE_SUBTILE": 4,
                    "NUM_CTAS": 2,
                    "SPLIT_K": 1,
                    "INTERLEAVE_EPILOGUE": 1,
                    "ctas_per_cga": (2, 1, 1),
                    "pre_hook": matmul_tma_set_block_size_hook,
                }

    # Characteristic 2: Undersaturated GPU needs Split-K for parallelism
    if is_undersaturated:
        # Use MN product to determine tile size - larger MN benefits from wider tiles
        mn_product = M * N
        is_large_output = mn_product >= 1_000_000  # ~1M elements in output

        if is_large_output:
            block_m, block_n, block_k = 256, 128, 64
            k_tiles = math.ceil(K / block_k)
        else:
            block_m, block_n, block_k = 128, 64, 128
            k_tiles = math.ceil(K / block_k)

        split_k = 1
        # Prefer lower Split-K values that still provide enough parallelism
        for sk in [4, 2, 8]:
            if k_tiles >= sk and k_tiles // sk >= 4:
                split_k = sk
                break
        if split_k > 1:
            if is_large_output:
                # Larger output: wider tiles, more epilogue subtiling, fewer TMEM buffers
                return {
                    "BLOCK_SIZE_M": block_m,
                    "BLOCK_SIZE_N": block_n,
                    "BLOCK_SIZE_K": block_k,
                    "GROUP_SIZE_M": _select_group_size_m(M, N, block_m),
                    "NUM_SMEM_BUFFERS": 4,
                    "NUM_TMEM_BUFFERS": 2,
                    "NUM_MMA_GROUPS": 2,
                    "EPILOGUE_SUBTILE": 8,
                    "NUM_CTAS": 1,
                    "SPLIT_K": split_k,
                    "INTERLEAVE_EPILOGUE": 1,
                    "ctas_per_cga": None,
                    "pre_hook": matmul_tma_set_block_size_hook,
                }
            else:
                # Smaller output: narrower tiles
                return {
                    "BLOCK_SIZE_M": block_m,
                    "BLOCK_SIZE_N": block_n,
                    "BLOCK_SIZE_K": block_k,
                    "GROUP_SIZE_M": _select_group_size_m(M, N, block_m),
                    "NUM_SMEM_BUFFERS": 4,
                    "NUM_TMEM_BUFFERS": 3,
                    "NUM_MMA_GROUPS": 2,
                    "EPILOGUE_SUBTILE": 1,
                    "NUM_CTAS": 1,
                    "SPLIT_K": split_k,
                    "INTERLEAVE_EPILOGUE": 1,
                    "ctas_per_cga": None,
                    "pre_hook": matmul_tma_set_block_size_hook,
                }

    # Characteristic 3: GPU-saturated shapes use wide tiles for data reuse
    if is_gpu_saturated:
        return {
            "BLOCK_SIZE_M": 256,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": _select_group_size_m(M, N, 256),
            "NUM_SMEM_BUFFERS": 2,
            "NUM_TMEM_BUFFERS": 1,
            "NUM_MMA_GROUPS": 2,
            "EPILOGUE_SUBTILE": 4,
            "NUM_CTAS": 1,
            "SPLIT_K": 1,
            "INTERLEAVE_EPILOGUE": 0,
            "ctas_per_cga": None,
            "pre_hook": matmul_tma_set_block_size_hook,
        }

    # ==========================================================================
    # Fallback: General wave efficiency heuristic for remaining shapes
    # ==========================================================================

    # Candidate configs: (BLOCK_M, BLOCK_N, BLOCK_K, NUM_CTAS, NUM_SMEM_BUFFERS, NUM_TMEM_BUFFERS, NUM_MMA_GROUPS, EPILOGUE_SUBTILE)
    # Based on autotuning results - best configs use BLOCK_K=128, 2-CTA clusters, and balanced buffers
    candidates = [
        # Best config for tall-M shapes (3159809, 384, 384) - prioritize before square config
        (256, 128, 128, 2, 2, 2, 2, 1),  # Best for (3159809, 384, 384)
        # Best config for large square matrices (8192x8192x8192)
        (256, 256, 64, 1, 3, 1, 2, 4),  # Best for 8192x8192x8192
        # Best config for large-K shapes (1024, 256, 16384) - needs Split-K
        (128, 64, 128, 1, 4, 3, 2, 1),  # Best for (1024, 256, 16384) with Split-K
        # 2-CTA configs with BLOCK_K=128 (best performing from autotuning)
        (256, 128, 64, 2, 5, 2, 2, 4),  # Best for (1152, 1024, 213120)
        (128, 256, 64, 2, 4, 2, 1, 2),  # Good general config
        (256, 64, 128, 2, 5, 2, 2, 4),  # Best for skinny-N shapes
        (128, 64, 128, 2, 5, 2, 2, 1),  # Best for (1152, 1024, 12800)
        # 1-CTA configs
        (256, 64, 128, 1, 5, 2, 2, 8),  # Good for skinny-N
        (128, 256, 64, 1, 3, 2, 1, 2),  # Wide tiles
        (128, 128, 64, 1, 4, 2, 1, 2),  # Square tiles
        (256, 128, 64, 1, 3, 1, 2, 2),  # Tall tiles
        (128, 64, 64, 1, 5, 2, 1, 1),  # Small tiles for small problems
        (64, 128, 64, 1, 5, 2, 1, 1),  # Small tiles, wide
        (64, 64, 64, 1, 6, 2, 1, 1),  # Smallest tiles
    ]

    def estimate_smem(
        bm, bn, bk, num_ctas, num_smem_buffers, num_mma_groups, epilogue_subtile
    ):
        """Estimate shared memory usage for a config."""
        bm_split = bm // num_mma_groups
        num_epilogue_smem_buffers = num_mma_groups if num_mma_groups > 2 else 2
        smem_a = bm * bk * 2 * num_smem_buffers
        smem_b = bk * (bn // num_ctas) * 2 * num_smem_buffers
        smem_epilog = (
            bm_split * (bn // epilogue_subtile) * 2 * num_epilogue_smem_buffers
        )
        smem_barriers = (
            num_smem_buffers * num_mma_groups * 8 * (2 if num_ctas == 2 else 1)
        )
        return smem_a + smem_b + smem_epilog + smem_barriers

    def estimate_tmem_columns(bn, num_tmem_buffers, num_mma_groups):
        """Estimate TLX tensor-memory column allocation for a config."""
        return bn * num_tmem_buffers * num_mma_groups

    def compute_wave_score(bm, bn, num_ctas, split_k=1):
        """
        Compute wave efficiency score (lower is better).
        Score = fraction of SMs idle in the last wave.
        """
        ctas_m = (M + bm - 1) // bm
        ctas_n = (N + bn - 1) // bn
        # Round up ctas_m to multiple of num_ctas for cluster alignment
        ctas_m = ((ctas_m + num_ctas - 1) // num_ctas) * num_ctas
        total_ctas = ctas_m * ctas_n * split_k

        if total_ctas == 0:
            return float("inf"), 0, 0

        waves = (total_ctas + num_sms - 1) // num_sms
        fractional_waves = total_ctas / num_sms
        score = waves - fractional_waves  # 0 = perfect, 1 = worst
        return score, total_ctas, waves

    best_config = None
    best_score = float("inf")
    best_waves = float("inf")

    for (
        bm,
        bn,
        bk,
        num_ctas,
        num_smem_buffers,
        num_tmem_buffers,
        num_mma_groups,
        epilogue_subtile,
    ) in candidates:
        # Skip if SMEM exceeds limit
        smem = estimate_smem(
            bm, bn, bk, num_ctas, num_smem_buffers, num_mma_groups, epilogue_subtile
        )
        if smem > MAX_SMEM:
            continue

        # Skip if TMEM exceeds limit
        tmem_columns = estimate_tmem_columns(bn, num_tmem_buffers, num_mma_groups)
        if tmem_columns > MAX_TMEM_COLUMNS:
            continue

        # Skip if MMA group size is invalid (must be <= 128 for hardware)
        if bm // num_mma_groups > 128:
            continue

        # Skip if tiles are larger than the problem
        if bm > M * 2 or bn > N * 2:
            continue

        # Compute wave efficiency
        score, total_ctas, waves = compute_wave_score(bm, bn, num_ctas)

        # Consider split-K only when MN tiles don't saturate GPU
        # Logic adapted from preprocess_configs
        split_k = 1
        num_tiles_m = math.ceil(M / bm)
        num_tiles_n = math.ceil(N / bn)
        num_mn_tiles = num_tiles_m * num_tiles_n

        if num_mn_tiles < num_sms:
            k_tiles = math.ceil(K / bk)
            # Try split-K values (higher first), each split must have enough K tiles
            for sk in [8, 4, 2]:
                if k_tiles >= sk and k_tiles // sk >= 4:
                    sk_score, sk_ctas, sk_waves = compute_wave_score(
                        bm, bn, num_ctas, sk
                    )
                    if sk_score < score or (sk_score == score and sk_ctas > total_ctas):
                        score, total_ctas, waves, split_k = (
                            sk_score,
                            sk_ctas,
                            sk_waves,
                            sk,
                        )
                    break  # Use the first valid split-K

        # Selection criteria:
        # 1. Prefer lower wave inefficiency score
        # 2. With same score, prefer fewer waves (less overhead)
        # 3. With same waves, prefer larger tiles (less total overhead)
        # 4. Prefer multi-CTA configs for better B-tile sharing
        score_slack = 0.1
        adjusted_score = score

        if (
            adjusted_score < best_score - score_slack
            or (adjusted_score < best_score + score_slack and waves < best_waves)
            or (
                adjusted_score < best_score + score_slack
                and waves == best_waves
                and num_ctas > 1
            )
        ):
            best_score = adjusted_score
            best_waves = waves
            best_config = {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": _select_group_size_m(M, N, bm),
                "NUM_SMEM_BUFFERS": num_smem_buffers,
                "NUM_TMEM_BUFFERS": num_tmem_buffers,
                "NUM_MMA_GROUPS": num_mma_groups,
                "EPILOGUE_SUBTILE": epilogue_subtile,
                "NUM_CTAS": num_ctas,
                "SPLIT_K": split_k,
                "INTERLEAVE_EPILOGUE": 0,
                "ctas_per_cga": (num_ctas, 1, 1) if num_ctas > 1 else None,
                "pre_hook": matmul_tma_set_block_size_hook,
            }

    return best_config


def _select_group_size_m(M, N, block_m):
    """
    Select GROUP_SIZE_M based on the golden rule for tile scheduling.

    GROUP_SIZE_M controls how tiles are traversed:
    - GROUP_SIZE_M = 1: Column-major (sweep M first), reuses B tiles
    - GROUP_SIZE_M = large: Row-major (sweep N first), reuses A tiles

    Golden rule:
    - When M >> N: Use small GROUP_SIZE_M to reuse B (smaller dimension)
    - When N >> M: Use large GROUP_SIZE_M to reuse A (smaller dimension)
    - When M ~ N: Use moderate GROUP_SIZE_M for L2 locality
    """
    num_m_tiles = (M + block_m - 1) // block_m
    ratio = M / max(N, 1)

    if ratio > 10:
        # M >> N: sweep M, reuse B
        return 1
    elif ratio < 0.1:
        # N >> M: sweep N, reuse A
        return min(64, num_m_tiles)
    else:
        # Balanced: moderate group size for L2 locality
        return min(8, num_m_tiles)


def get_reduced_autotune_config():
    """Single hardcoded config for testing — skips autotune entirely."""
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "NUM_SMEM_BUFFERS": 3,
                "NUM_TMEM_BUFFERS": 1,
                "NUM_MMA_GROUPS": 2,
                "EPILOGUE_SUBTILE": 8,
                "NUM_CTAS": 1,
                "SPLIT_K": 1,
                "INTERLEAVE_EPILOGUE": 0,
                "USE_WARP_BARRIER": False,
                "NUM_X2_SUBTILES": 16,
                "REUSE_RMSNORM_ACROSS_N": False,
            },
            num_warps=4,
            num_stages=1,
            pre_hook=matmul_tma_set_block_size_hook,
        ),
    ]


def get_cuda_autotune_config():
    configs = [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": g,
                "NUM_SMEM_BUFFERS": s,
                "NUM_TMEM_BUFFERS": t,
                "NUM_MMA_GROUPS": m,
                "EPILOGUE_SUBTILE": subtile,
                "NUM_CTAS": num_ctas,
                "SPLIT_K": split_k,
                "INTERLEAVE_EPILOGUE": interleave,
                "USE_WARP_BARRIER": uwb,
                "NUM_X2_SUBTILES": x2sub,
                "REUSE_RMSNORM_ACROSS_N": False,
            },
            num_warps=4,
            num_stages=1,
            pre_hook=matmul_tma_set_block_size_hook,
            ctas_per_cga=(num_ctas, 1, 1) if num_ctas > 1 else None,
        )
        for BM in [64, 128, 256]
        for BN in [64, 128, 256]
        for BK in [64, 128]
        for s in [2, 3, 4, 5, 6, 7]
        for t in [1, 2, 3]
        for m in [1, 2]
        for subtile in [1, 2, 4, 8]
        for num_ctas in [1, 2]
        for split_k in [
            1,
            2,
            3,
            4,
            5,
            6,
            8,
            10,
            12,
            16,
            19,
            24,
        ]  # pruning selects one optimal SPLIT_K per tile group
        for interleave in [0, 1]
        for g in [1, 2, 8, 64]
        for uwb in [False, True]
        for x2sub in [4, 8, 16, 32, 64]
    ]
    configs.extend(
        [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 256,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": 1,
                    "NUM_SMEM_BUFFERS": 3,
                    "NUM_TMEM_BUFFERS": 1,
                    "NUM_MMA_GROUPS": 2,
                    "EPILOGUE_SUBTILE": subtile,
                    "NUM_CTAS": 1,
                    "SPLIT_K": 1,
                    "INTERLEAVE_EPILOGUE": 0,
                    "USE_WARP_BARRIER": False,
                    "NUM_X2_SUBTILES": x2sub,
                    "REUSE_RMSNORM_ACROSS_N": True,
                },
                num_warps=4,
                num_stages=1,
                pre_hook=matmul_tma_set_block_size_hook,
            )
            for subtile in [4, 8]
            for x2sub in [16, 32, 64]
        ]
    )
    return configs


def matmul_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BLOCK_SIZE_M"]
    BLOCK_N = nargs["BLOCK_SIZE_N"]
    BLOCK_K = nargs["BLOCK_SIZE_K"]
    NUM_MMA_GROUPS = nargs.get("NUM_MMA_GROUPS", 1)
    BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
    NUM_CTAS = nargs.get("NUM_CTAS", 1)
    BLOCK_N_PER_CTA = BLOCK_N // NUM_CTAS
    # For column-major inputs, TMA descriptor block shape matches the transposed view
    if nargs.get("A_ROW_MAJOR", True):
        nargs["a_desc"].block_shape = [BLOCK_M_SPLIT, BLOCK_K]
    else:
        nargs["a_desc"].block_shape = [BLOCK_K, BLOCK_M_SPLIT]
    if nargs.get("B_ROW_MAJOR", True):
        nargs["b_desc"].block_shape = [BLOCK_K, BLOCK_N_PER_CTA]
    else:
        nargs["b_desc"].block_shape = [BLOCK_N_PER_CTA, BLOCK_K]
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", 1)
    nargs["c_desc"].block_shape = [
        BLOCK_M // NUM_MMA_GROUPS,
        BLOCK_N // EPILOGUE_SUBTILE,
    ]
    SPLIT_K = nargs.get("SPLIT_K", 1)
    if SPLIT_K > 1:
        M = nargs["M"]
        N = nargs["N"]
        workspace = torch.empty(
            (SPLIT_K * M, N),
            device=nargs["c_desc"].base.device,
            dtype=nargs["c_desc"].base.dtype,
        )
        nargs["workspace_desc"].base = workspace
        nargs["workspace_desc"].shape = list(workspace.shape)
    else:
        nargs["workspace_desc"].base = nargs["c_desc"].base
        nargs["workspace_desc"].shape = list(nargs["c_desc"].base.shape)
    nargs["workspace_desc"].block_shape = [
        BLOCK_M // NUM_MMA_GROUPS,
        BLOCK_N // EPILOGUE_SUBTILE,
    ]


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


def preprocess_configs(configs, named_args, **kwargs):  # noqa: C901
    # Blackwell B200A resource limits
    NUM_SMS = _get_num_sms()
    MAX_SHARED_MEMORY = 232 * 1024  # bytes (232KB)
    MAX_TENSOR_MEMORY_COLUMNS = 512  # TLX tmem column allocation limit per SM

    MBARRIER_SIZE = 8  # bytes

    M = named_args["M"]
    N = named_args["N"]
    K = named_args["K"]

    pruned_configs = []
    for conf in configs:
        BLOCK_M = conf.kwargs["BLOCK_SIZE_M"]
        BLOCK_N = conf.kwargs["BLOCK_SIZE_N"]
        BLOCK_K = conf.kwargs["BLOCK_SIZE_K"]
        NUM_SMEM_BUFFERS = conf.kwargs["NUM_SMEM_BUFFERS"]
        NUM_TMEM_BUFFERS = conf.kwargs["NUM_TMEM_BUFFERS"]
        NUM_CTAS = conf.kwargs["NUM_CTAS"]
        NUM_MMA_GROUPS = conf.kwargs["NUM_MMA_GROUPS"]
        SPLIT_K = conf.kwargs.get("SPLIT_K", 1)
        EPILOGUE_SUBTILE = conf.kwargs["EPILOGUE_SUBTILE"]
        INTERLEAVE_EPILOGUE = conf.kwargs.get("INTERLEAVE_EPILOGUE", 0)
        GROUP_SIZE_M = conf.kwargs["GROUP_SIZE_M"]
        REUSE_RMSNORM_ACROSS_N = conf.kwargs.get("REUSE_RMSNORM_ACROSS_N", False)

        if M == 262144 and K == 512 and N == 2048:
            target_configs = {
                # Current best reduced-config baseline.
                (256, 256, 64, 1, 3, 1, 2, 8, 1, 0, False, 16, False),
                (256, 256, 64, 1, 3, 1, 2, 8, 1, 0, False, 32, False),
                (256, 256, 64, 1, 3, 1, 2, 8, 1, 0, False, 64, False),
                # Taller N tiles reduce tile count while staying within TMEM.
                (128, 256, 64, 1, 3, 1, 2, 8, 1, 0, False, 16, False),
                (256, 256, 64, 1, 3, 1, 2, 4, 1, 0, False, 16, False),
                (256, 256, 64, 1, 3, 1, 2, 4, 1, 0, False, 32, False),
                (256, 256, 64, 1, 3, 1, 2, 4, 1, 0, False, 64, False),
                # Exact-shape schedule that reuses A/RMSNorm across the eight N tiles.
                (256, 256, 64, 1, 3, 1, 2, 8, 1, 0, False, 16, True),
                (256, 256, 64, 1, 3, 1, 2, 8, 1, 0, False, 32, True),
                (256, 256, 64, 1, 3, 1, 2, 8, 1, 0, False, 64, True),
                (256, 256, 64, 1, 3, 1, 2, 4, 1, 0, False, 16, True),
                (256, 256, 64, 1, 3, 1, 2, 4, 1, 0, False, 32, True),
                (256, 256, 64, 1, 3, 1, 2, 4, 1, 0, False, 64, True),
                # Narrower N tiles allow deeper buffering and BK=128.
                (256, 128, 64, 1, 4, 2, 2, 8, 1, 0, False, 16, False),
                (256, 128, 64, 1, 4, 2, 2, 8, 1, 1, False, 16, False),
                (256, 128, 128, 1, 2, 2, 2, 8, 1, 0, False, 16, False),
                (256, 128, 128, 1, 2, 2, 2, 8, 1, 1, False, 16, False),
                # 2-CTA B sharing variants for tall-M shapes.
                (256, 128, 128, 2, 2, 2, 2, 8, 2, 0, False, 16, False),
                (256, 128, 128, 2, 2, 2, 2, 8, 2, 1, False, 16, False),
                # Smaller M tiles increase spatial parallelism and reduce x2 work per CTA.
                (128, 128, 128, 1, 3, 2, 2, 8, 1, 0, False, 16, False),
                (128, 128, 128, 1, 3, 2, 2, 8, 1, 1, False, 16, False),
            }
            target_key = (
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                GROUP_SIZE_M,
                NUM_SMEM_BUFFERS,
                NUM_TMEM_BUFFERS,
                NUM_MMA_GROUPS,
                EPILOGUE_SUBTILE,
                NUM_CTAS,
                INTERLEAVE_EPILOGUE,
                conf.kwargs.get("USE_WARP_BARRIER", False),
                conf.kwargs.get("NUM_X2_SUBTILES", 1),
                REUSE_RMSNORM_ACROSS_N,
            )
            if target_key not in target_configs:
                continue

        if REUSE_RMSNORM_ACROSS_N:
            if (
                GROUP_SIZE_M != 1
                or NUM_CTAS != 1
                or SPLIT_K != 1
                or BLOCK_N != 256
                or BLOCK_M != 256
                or M != 262144
                or K != 512
                or N != 2048
            ):
                continue

        # Filter out invalid config that causes wrong hardware MMA
        if BLOCK_M // NUM_MMA_GROUPS > 128:
            continue
        # Pair-CTA MMA doesn't work with M=64 per MMA group
        if NUM_CTAS == 2 and BLOCK_M // NUM_MMA_GROUPS == 64:
            continue

        if NUM_CTAS == 2 and GROUP_SIZE_M == 1:
            continue
        # GROUP_SIZE_M must be a multiple of NUM_CTAS so that consecutive
        # tile_ids (assigned to paired CTAs in a cluster) always map to the
        # same pid_n. Otherwise, at group boundaries a CTA pair can straddle
        # two different pid_n values, breaking 2-CTA B-tile sharing.
        if GROUP_SIZE_M % NUM_CTAS != 0:
            continue

        # EPILOGUE_SUBTILE must evenly divide BLOCK_N
        if BLOCK_N % EPILOGUE_SUBTILE != 0:
            continue

        # NUM_X2_SUBTILES must evenly divide BLOCK_K
        NUM_X2_SUBTILES = conf.kwargs.get("NUM_X2_SUBTILES", 1)
        if BLOCK_K % NUM_X2_SUBTILES != 0:
            continue

        # Interleaved epilogue requires NUM_MMA_GROUPS == 2
        if INTERLEAVE_EPILOGUE and NUM_MMA_GROUPS != 2:
            continue

        # Blackwell MMA requires BLOCK_M_SPLIT >= 64
        if BLOCK_M // NUM_MMA_GROUPS < 64:
            continue

        num_tiles_m = math.ceil(M / BLOCK_M)
        num_tiles_n = math.ceil(N / BLOCK_N)
        num_mn_tiles = num_tiles_m * num_tiles_n

        # BM=64 tiles only help when MN is too small with larger tiles.
        # Skip them for shapes that already have enough spatial tiles
        # to avoid bloating the autotuner search space.
        if BLOCK_M == 64 and math.ceil(M / 128) * math.ceil(N / 128) > 16:
            continue

        # Fused RMSNorm is incompatible with split-K (each split only sees partial K)
        if named_args.get("FUSED_RMSNORM", False) and SPLIT_K > 1:
            continue

        # --- Split-K gating: only allow SPLIT_K > 1 for small shapes ---
        # Split-K helps when MN tiles are too few to saturate the GPU.
        # For large shapes with plenty of MN tiles, SPLIT_K=1 is better
        # since it avoids the atomic reduction overhead.
        if SPLIT_K > 1:
            if num_mn_tiles >= NUM_SMS:
                continue
            k_tiles = math.ceil(K / BLOCK_K)
            if k_tiles < SPLIT_K:
                continue
            # Reject SK values where cdiv overallocation leaves the last split empty
            # (causes deadlock: producer loop is empty but MMA consumer waits on barrier)
            k_tiles_per_split = math.ceil(k_tiles / SPLIT_K)
            if k_tiles_per_split * (SPLIT_K - 1) >= k_tiles:
                continue
            # Each split must have enough K tiles to be worthwhile
            if k_tiles // SPLIT_K < 4:
                continue

        # --- Shared Memory estimation ---
        smem_a = BLOCK_M * BLOCK_K * 2 * NUM_SMEM_BUFFERS
        smem_b_size = BLOCK_N // NUM_CTAS
        smem_b = BLOCK_K * smem_b_size * 2 * NUM_SMEM_BUFFERS
        BLOCK_M_SPLIT = BLOCK_M // NUM_MMA_GROUPS
        NUM_EPILOGUE_SMEM_BUFFERS = NUM_MMA_GROUPS if NUM_MMA_GROUPS > 2 else 2
        smem_epilog = (
            BLOCK_M_SPLIT
            * (BLOCK_N // EPILOGUE_SUBTILE)
            * 2
            * NUM_EPILOGUE_SMEM_BUFFERS
        )
        smem_barriers = NUM_SMEM_BUFFERS * NUM_MMA_GROUPS * MBARRIER_SIZE
        if NUM_CTAS == 2:
            smem_barriers += NUM_SMEM_BUFFERS * NUM_MMA_GROUPS * MBARRIER_SIZE
        smem_barriers += NUM_TMEM_BUFFERS

        total_smem = smem_a + smem_b + smem_epilog + smem_barriers
        if total_smem > MAX_SHARED_MEMORY:
            continue

        # --- Tensor Memory (TMEM) estimation ---
        total_tmem_columns = BLOCK_N * NUM_TMEM_BUFFERS * NUM_MMA_GROUPS
        if total_tmem_columns > MAX_TENSOR_MEMORY_COLUMNS:
            continue

        pruned_configs.append(conf)

    # Two-level SPLIT_K filter (per tile-size group):
    #   1. Minimize wave count (fewer waves = less wall-clock time).
    #   2. Within the same wave count, maximize SPLIT_K (more K-parallelism
    #      across SMs). E.g. with 148 SMs and 40 base tiles: SPLIT_K=3
    #      gives 120 tiles (120 SMs active, each does K/3 work) vs SPLIT_K=1
    #      giving 40 tiles (40 SMs active, each does K/1 work) — both 1 wave,
    #      but SPLIT_K=3 is faster because work is spread across more SMs.
    # Applied per (BM, BN, BK) group because different tile sizes have
    # vastly different compute characteristics.
    # Note: for saturated shapes, SPLIT_K>1 configs are already pruned by
    # the base_tiles >= NUM_SMS gate above, so only SPLIT_K=1 survives.
    if pruned_configs:

        def _total_tiles(c):
            return (
                math.ceil(M / c.kwargs["BLOCK_SIZE_M"])
                * math.ceil(N / c.kwargs["BLOCK_SIZE_N"])
                * c.kwargs.get("SPLIT_K", 1)
            )

        def _num_waves(c):
            return math.ceil(_total_tiles(c) / NUM_SMS)

        def _tile_key(c):
            return (
                c.kwargs["BLOCK_SIZE_M"],
                c.kwargs["BLOCK_SIZE_N"],
                c.kwargs["BLOCK_SIZE_K"],
            )

        # Group by tile size
        tile_groups = {}
        for c in pruned_configs:
            tile_groups.setdefault(_tile_key(c), []).append(c)

        result = []
        for group_configs in tile_groups.values():
            min_waves = min(_num_waves(c) for c in group_configs)
            best = [c for c in group_configs if _num_waves(c) == min_waves]
            max_sk = max(c.kwargs.get("SPLIT_K", 1) for c in best)
            best = [c for c in best if c.kwargs.get("SPLIT_K", 1) == max_sk]
            result.extend(best)

        pruned_configs = result

    # --- Golden Rule: sweep the large dimension, fix the small one ---
    # A[M,K] changes with M; B[K,N] changes with N.
    # GROUP_SIZE_M controls how many M-tiles are grouped before advancing N.
    #   GROUP_SIZE_M = 1  → sweep M first (column-major), B (small-N side) reused
    #   GROUP_SIZE_M = large → sweep N first (row-major), A (small-M side) reused
    # When M >> N: prefer small GROUP_SIZE_M (sweep M, fix B for reuse)
    # When N >> M: prefer large GROUP_SIZE_M (sweep N, fix A for reuse)
    if pruned_configs:
        IMBALANCE_THRESHOLD = 10  # ratio at which we enforce the rule
        if M > N * IMBALANCE_THRESHOLD:
            # M >> N: keep only small GROUP_SIZE_M to sweep M
            pruned_configs = [
                c for c in pruned_configs if c.kwargs["GROUP_SIZE_M"] <= 2
            ]
        elif N > M * IMBALANCE_THRESHOLD:
            # N >> M: keep only large GROUP_SIZE_M to sweep N
            pruned_configs = [
                c for c in pruned_configs if c.kwargs["GROUP_SIZE_M"] >= 32
            ]
        else:
            # Balanced M ≈ N: keep moderate GROUP_SIZE_M for L2 locality
            pruned_configs = [
                c for c in pruned_configs if c.kwargs["GROUP_SIZE_M"] == 8
            ]

    # Pareto-optimal filtering on (NUM_SMEM_BUFFERS, NUM_TMEM_BUFFERS,
    # NUM_MMA_GROUPS): these are independent resource dimensions where more
    # buffers / groups generally means better pipelining, but no single
    # dimension dominates the others.  Keep a config unless another config
    # in the same (BM, BN, BK, SUBTILE, NUM_CTAS, SPLIT_K) group dominates
    # it (>= in all dimensions, > in at least one).
    if pruned_configs:

        def _group_key(c):
            return (
                c.kwargs["BLOCK_SIZE_M"],
                c.kwargs["BLOCK_SIZE_N"],
                c.kwargs["BLOCK_SIZE_K"],
                c.kwargs["EPILOGUE_SUBTILE"],
                c.kwargs["NUM_CTAS"],
                c.kwargs.get("SPLIT_K", 1),
                c.kwargs.get("INTERLEAVE_EPILOGUE", 0),
            )

        def _val(c):
            return (
                c.kwargs["NUM_SMEM_BUFFERS"],
                c.kwargs["NUM_TMEM_BUFFERS"],
                c.kwargs["NUM_MMA_GROUPS"],
            )

        def _dominates(a, b):
            """Return True if a dominates b (>= in all, > in at least one)."""
            va, vb = _val(a), _val(b)
            return all(x >= y for x, y in zip(va, vb)) and any(
                x > y for x, y in zip(va, vb)
            )

        groups = {}
        for c in pruned_configs:
            groups.setdefault(_group_key(c), []).append(c)

        pruned_configs = []
        for members in groups.values():
            for c in members:
                if not any(_dominates(other, c) for other in members if other is not c):
                    pruned_configs.append(c)

    return pruned_configs


@triton.jit
def _get_bufidx_phase(accum_cnt, NUM_BUFFERS_KV):
    bufIdx = accum_cnt % NUM_BUFFERS_KV
    phase = (accum_cnt // NUM_BUFFERS_KV) & 1
    return bufIdx, phase


@triton.jit
def _compute_grid_info(
    M,
    N,
    K,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    GROUP_SIZE_M,
    SPLIT_K,
    NUM_CTAS: tl.constexpr,
):
    """Compute common grid information used across async tasks."""
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # Pad num_pid_m to multiple of NUM_CTAS so CTA clusters tile evenly along M.
    num_pid_m = (num_pid_m + NUM_CTAS - 1) // NUM_CTAS * NUM_CTAS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    num_mn_tiles = num_pid_m * num_pid_n
    num_tiles = num_mn_tiles * SPLIT_K
    k_tiles_total = tl.cdiv(K, BLOCK_SIZE_K)
    return (
        start_pid,
        num_pid_m,
        num_pid_n,
        num_pid_in_group,
        num_mn_tiles,
        num_tiles,
        k_tiles_total,
    )


@triton.jit
def _process_tile_epilogue_inner(  # noqa: C901
    tile_id,
    num_pid_in_group,
    num_pid_m,
    num_mn_tiles,
    GROUP_SIZE_M,
    M,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    EPILOGUE_SUBTILE,
    NUM_MMA_GROUPS,
    NUM_EPILOGUE_SMEM_BUFFERS,
    NUM_TMEM_BUFFERS,
    SPLIT_K,
    INTERLEAVE_EPILOGUE,
    c_desc,
    workspace_desc,
    c_smem_buffers,
    tmem_buffers,
    tmem_full_bars,
    tmem_empty_bars,
    cur_tmem_buf,
    tmem_read_phase,
    FUSED_RMSNORM: tl.constexpr = False,
    REUSE_RMSNORM_ACROSS_N: tl.constexpr = False,
    BF16_EPILOGUE_SCALE: tl.constexpr = False,
    rrms_0=None,
    rrms_1=None,
    WAIT_FOR_STORES: tl.constexpr = True,
    ENABLE_PROTON: tl.constexpr = False,
):
    """Process epilogue for a single tile."""
    mn_tile_id = tile_id % num_mn_tiles
    pid_m, pid_n = _compute_pid(mn_tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_SIZE_M // NUM_MMA_GROUPS

    slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
    if SPLIT_K > 1:
        split_id = tile_id // num_mn_tiles
        out_desc = workspace_desc
        row_base = split_id * M
    else:
        out_desc = c_desc
        row_base = 0

    if FUSED_RMSNORM and BF16_EPILOGUE_SCALE:
        rrms_0_bf16 = rrms_0.to(tlx.dtype_of(out_desc))
        if NUM_MMA_GROUPS >= 2:
            rrms_1_bf16 = rrms_1.to(tlx.dtype_of(out_desc))

    if INTERLEAVE_EPILOGUE:
        # Interleaved TMA stores across two groups to improve memory throughput.
        # Pattern: wait g0, store g0s0, wait g1, store g1s0,
        #          then alternate g0/g1 for slices 1-3.
        buf_idx_0 = 0 * NUM_TMEM_BUFFERS + cur_tmem_buf
        buf_idx_1 = 1 * NUM_TMEM_BUFFERS + cur_tmem_buf
        acc_tmem_0 = tmem_buffers[buf_idx_0]
        acc_tmem_1 = tmem_buffers[buf_idx_1]
        offs_am_0 = pid_m * BLOCK_SIZE_M + 0 * BLOCK_M_SPLIT
        offs_am_1 = pid_m * BLOCK_SIZE_M + 1 * BLOCK_M_SPLIT

        # --- Wait for group 0, store group 0 slice 0 ---
        if ENABLE_PROTON:
            pl.enter_scope("epi_wait_tmem_full_g0")
        tlx.barrier_wait(tmem_full_bars[buf_idx_0], tmem_read_phase)
        if ENABLE_PROTON:
            pl.exit_scope("epi_wait_tmem_full_g0")
        acc_sub = tlx.local_slice(
            acc_tmem_0, [0, 0 * slice_size], [BLOCK_M_SPLIT, slice_size]
        )
        result = tlx.local_load(acc_sub)
        tlx.barrier_arrive(tmem_empty_bars[buf_idx_0], 1)
        if FUSED_RMSNORM:
            if BF16_EPILOGUE_SCALE:
                result = result.to(tlx.dtype_of(out_desc)) * rrms_0_bf16
            else:
                result = result * rrms_0
        c = result.to(tlx.dtype_of(out_desc))
        c_smem = c_smem_buffers[0]
        tlx.local_store(c_smem, c)
        tlx.fence_async_shared()
        if ENABLE_PROTON:
            pl.enter_scope("epi_tma_store_g0s0")
        tlx.async_descriptor_store(
            out_desc,
            c_smem,
            [row_base + offs_am_0, offs_bn + 0 * slice_size],
            eviction_policy="evict_first",
        )
        if ENABLE_PROTON:
            pl.exit_scope("epi_tma_store_g0s0")

        # --- Wait for group 1, store group 1 slice 0 ---
        if ENABLE_PROTON:
            pl.enter_scope("epi_wait_tmem_full_g1")
        tlx.barrier_wait(tmem_full_bars[buf_idx_1], tmem_read_phase)
        if ENABLE_PROTON:
            pl.exit_scope("epi_wait_tmem_full_g1")
        acc_sub = tlx.local_slice(
            acc_tmem_1, [0, 0 * slice_size], [BLOCK_M_SPLIT, slice_size]
        )
        result = tlx.local_load(acc_sub)
        tlx.barrier_arrive(tmem_empty_bars[buf_idx_1], 1)
        if FUSED_RMSNORM:
            if BF16_EPILOGUE_SCALE:
                result = result.to(tlx.dtype_of(out_desc)) * rrms_1_bf16
            else:
                result = result * rrms_1
        c = result.to(tlx.dtype_of(out_desc))
        c_smem = c_smem_buffers[1]
        tlx.local_store(c_smem, c)
        tlx.fence_async_shared()
        if ENABLE_PROTON:
            pl.enter_scope("epi_tma_store_g1s0")
        tlx.async_descriptor_store(
            out_desc,
            c_smem,
            [row_base + offs_am_1, offs_bn + 0 * slice_size],
            eviction_policy="evict_first",
        )
        if ENABLE_PROTON:
            pl.exit_scope("epi_tma_store_g1s0")

        # --- Slices 1-3: alternate group 0, group 1 ---
        for slice_id in tl.static_range(1, EPILOGUE_SUBTILE):
            # Group 0
            acc_sub = tlx.local_slice(
                acc_tmem_0, [0, slice_id * slice_size], [BLOCK_M_SPLIT, slice_size]
            )
            result = tlx.local_load(acc_sub)
            tlx.barrier_arrive(tmem_empty_bars[buf_idx_0], 1)
            if FUSED_RMSNORM:
                if BF16_EPILOGUE_SCALE:
                    result = result.to(tlx.dtype_of(out_desc)) * rrms_0_bf16
                else:
                    result = result * rrms_0
            c = result.to(tlx.dtype_of(out_desc))
            c_smem = c_smem_buffers[0]
            if ENABLE_PROTON:
                pl.enter_scope("epi_tma_store_wait_g0")
            tlx.async_descriptor_store_wait(1)
            if ENABLE_PROTON:
                pl.exit_scope("epi_tma_store_wait_g0")
            tlx.local_store(c_smem, c)
            tlx.fence("async_shared")
            if ENABLE_PROTON:
                pl.enter_scope("epi_tma_store_g0")
            tlx.async_descriptor_store(
                out_desc,
                c_smem,
                [row_base + offs_am_0, offs_bn + slice_id * slice_size],
                eviction_policy="evict_first",
            )
            if ENABLE_PROTON:
                pl.exit_scope("epi_tma_store_g0")

            # Group 1
            acc_sub = tlx.local_slice(
                acc_tmem_1, [0, slice_id * slice_size], [BLOCK_M_SPLIT, slice_size]
            )
            result = tlx.local_load(acc_sub)
            tlx.barrier_arrive(tmem_empty_bars[buf_idx_1], 1)
            if FUSED_RMSNORM:
                if BF16_EPILOGUE_SCALE:
                    result = result.to(tlx.dtype_of(out_desc)) * rrms_1_bf16
                else:
                    result = result * rrms_1
            c = result.to(tlx.dtype_of(out_desc))
            c_smem = c_smem_buffers[1]
            if ENABLE_PROTON:
                pl.enter_scope("epi_tma_store_wait_g1")
            tlx.async_descriptor_store_wait(1)
            if ENABLE_PROTON:
                pl.exit_scope("epi_tma_store_wait_g1")
            tlx.local_store(c_smem, c)
            tlx.fence("async_shared")
            if ENABLE_PROTON:
                pl.enter_scope("epi_tma_store_g1")
            tlx.async_descriptor_store(
                out_desc,
                c_smem,
                [row_base + offs_am_1, offs_bn + slice_id * slice_size],
                eviction_policy="evict_first",
            )
            if ENABLE_PROTON:
                pl.exit_scope("epi_tma_store_g1")
    else:
        for group_id in tl.static_range(NUM_MMA_GROUPS):
            # Wait for TMEM to be filled
            buf_idx = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf

            if ENABLE_PROTON:
                pl.enter_scope("epi_wait_tmem_full")
            tlx.barrier_wait(tmem_full_bars[buf_idx], tmem_read_phase)
            if ENABLE_PROTON:
                pl.exit_scope("epi_wait_tmem_full")

            offs_am = pid_m * BLOCK_SIZE_M + group_id * BLOCK_M_SPLIT
            # Select rrms for this group
            if FUSED_RMSNORM:
                if group_id == 0:
                    rrms = rrms_0
                    if BF16_EPILOGUE_SCALE:
                        rrms_bf16 = rrms_0_bf16
                if NUM_MMA_GROUPS >= 2:
                    if group_id == 1:
                        rrms = rrms_1
                        if BF16_EPILOGUE_SCALE:
                            rrms_bf16 = rrms_1_bf16

            # load the result from TMEM to registers
            acc_tmem = tmem_buffers[buf_idx]
            for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                acc_tmem_subslice = tlx.local_slice(
                    acc_tmem,
                    [0, slice_id * slice_size],
                    [BLOCK_M_SPLIT, slice_size],
                )
                result = tlx.local_load(acc_tmem_subslice)
                tlx.barrier_arrive(tmem_empty_bars[buf_idx], 1)
                if FUSED_RMSNORM:
                    if BF16_EPILOGUE_SCALE:
                        result = result.to(tlx.dtype_of(out_desc)) * rrms_bf16
                    else:
                        result = result * rrms
                c = result.to(tlx.dtype_of(out_desc))
                c_smem = c_smem_buffers[
                    (group_id * EPILOGUE_SUBTILE + slice_id) % NUM_EPILOGUE_SMEM_BUFFERS
                ]
                if ENABLE_PROTON:
                    pl.enter_scope("epi_tma_store_wait")
                tlx.async_descriptor_store_wait(NUM_EPILOGUE_SMEM_BUFFERS - 1)
                if ENABLE_PROTON:
                    pl.exit_scope("epi_tma_store_wait")
                tlx.local_store(c_smem, c)
                tlx.fence_async_shared()
                if ENABLE_PROTON:
                    pl.enter_scope("epi_tma_store")
                tlx.async_descriptor_store(
                    out_desc,
                    c_smem,
                    [row_base + offs_am, offs_bn + slice_id * slice_size],
                    eviction_policy="evict_first",
                )
                if ENABLE_PROTON:
                    pl.exit_scope("epi_tma_store")

    # Wait for all TMA stores to complete unless the caller will drain once
    # after a persistent sequence of tiles.
    if WAIT_FOR_STORES:
        tlx.async_descriptor_store_wait(0)


@triton.jit
def _process_tile_mma_inner(  # noqa: C901
    k_tiles,
    k_tile_start,
    k_tile_end,
    NUM_SMEM_BUFFERS,
    NUM_MMA_GROUPS,
    NUM_TMEM_BUFFERS,
    buffers_A,
    buffers_B,
    tmem_buffers,
    A_smem_full_bars,
    B_smem_full_bars,
    A_smem_empty_bars,
    B_smem_empty_bars,
    tmem_full_bars,
    cur_tmem_buf,
    tmem_empty_bars,
    tmem_write_phase,
    smem_accum_cnt,
    NUM_CTAS,
    cta_bars,
    pred_cta0,
    A_ROW_MAJOR: tl.constexpr = True,
    B_ROW_MAJOR: tl.constexpr = True,
    REUSE_A=False,
    ENABLE_PROTON: tl.constexpr = False,
):
    """Process MMA for a single tile over [k_tile_start, k_tile_end). Returns updated smem_accum_cnt."""
    local_k_tiles = k_tile_end - k_tile_start

    # Peeled first K-iteration: wait for data before acquiring TMEM
    buf, phase = _get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)

    # wait for current phase(round) of load for this buf
    if ENABLE_PROTON:
        pl.enter_scope("mma_wait_B_full_k0")
    tlx.barrier_wait(B_smem_full_bars[buf], phase)
    if ENABLE_PROTON:
        pl.exit_scope("mma_wait_B_full_k0")

    # Process first K iteration (peeled) with use_acc=False
    for group_id in tl.static_range(NUM_MMA_GROUPS):
        # Calculate buffer indices
        a_buf = group_id * NUM_SMEM_BUFFERS + buf
        acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf

        if ENABLE_PROTON:
            pl.enter_scope("mma_wait_A_full_k0")
        tlx.barrier_wait(A_smem_full_bars[a_buf], phase)
        if ENABLE_PROTON:
            pl.exit_scope("mma_wait_A_full_k0")

        # Wait for epilogue to be done with all TMEM buffers (after data is ready)
        cur_barrier_idx = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
        if ENABLE_PROTON:
            pl.enter_scope("mma_wait_tmem_empty_k0")
        tlx.barrier_wait(tmem_empty_bars[cur_barrier_idx], tmem_write_phase ^ 1)
        if ENABLE_PROTON:
            pl.exit_scope("mma_wait_tmem_empty_k0")

        # CTA0 waits for CTA0 and CTA1 to finish loading A and B before issuing dot op
        if NUM_CTAS == 2:
            tlx.barrier_arrive(cta_bars[a_buf], arrive_count=1, remote_cta_rank=0)
            tlx.barrier_wait(cta_bars[a_buf], phase=phase, pred=pred_cta0)

        # Transpose SMEM buffers if inputs were column-major
        a_operand = (
            tlx.local_trans(buffers_A[a_buf]) if not A_ROW_MAJOR else buffers_A[a_buf]
        )
        b_operand = (
            tlx.local_trans(buffers_B[buf]) if not B_ROW_MAJOR else buffers_B[buf]
        )

        # Perform MMA: use_acc=False for first K iteration (clears accumulator)
        if ENABLE_PROTON:
            pl.enter_scope(f"mma_async_dot_k0_g{group_id}")
        tlx.async_dot(
            a_operand,
            b_operand,
            tmem_buffers[acc_buf],
            use_acc=False,
            mBarriers=[A_smem_empty_bars[a_buf], B_smem_empty_bars[buf]],
            two_ctas=NUM_CTAS == 2,
            out_dtype=tl.float32,
        )
        if ENABLE_PROTON:
            pl.exit_scope(f"mma_async_dot_k0_g{group_id}")

    smem_accum_cnt += 1

    # Remaining K iterations with use_acc=True
    for _ in range(1, local_k_tiles):
        buf, phase = _get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)

        # wait for current phase(round) of load for this buf
        if ENABLE_PROTON:
            pl.enter_scope("mma_wait_B_full")
        tlx.barrier_wait(B_smem_full_bars[buf], phase)
        if ENABLE_PROTON:
            pl.exit_scope("mma_wait_B_full")

        # Process all subtiles for this K iteration
        for group_id in tl.static_range(NUM_MMA_GROUPS):
            # Calculate buffer indices
            a_buf = group_id * NUM_SMEM_BUFFERS + buf
            acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf

            # Wait for this A subtile buffer to be loaded.
            if ENABLE_PROTON:
                pl.enter_scope("mma_wait_A_full")
            tlx.barrier_wait(A_smem_full_bars[a_buf], phase)
            if ENABLE_PROTON:
                pl.exit_scope("mma_wait_A_full")

            # CTA0 waits for CTA0 and CTA1 to finish loading A and B before issuing dot op
            if NUM_CTAS == 2:
                tlx.barrier_arrive(cta_bars[a_buf], arrive_count=1, remote_cta_rank=0)
                tlx.barrier_wait(cta_bars[a_buf], phase=phase, pred=pred_cta0)

            # Transpose SMEM buffers if inputs were column-major
            a_operand = (
                tlx.local_trans(buffers_A[a_buf])
                if not A_ROW_MAJOR
                else buffers_A[a_buf]
            )
            b_operand = (
                tlx.local_trans(buffers_B[buf]) if not B_ROW_MAJOR else buffers_B[buf]
            )

            # Perform MMA: use_acc=True for remaining K iterations
            if ENABLE_PROTON:
                pl.enter_scope(f"mma_async_dot_g{group_id}")
            tlx.async_dot(
                a_operand,
                b_operand,
                tmem_buffers[acc_buf],
                use_acc=True,
                mBarriers=[A_smem_empty_bars[a_buf], B_smem_empty_bars[buf]],
                two_ctas=NUM_CTAS == 2,
                out_dtype=tl.float32,
            )
            if ENABLE_PROTON:
                pl.exit_scope(f"mma_async_dot_g{group_id}")

        smem_accum_cnt += 1

    # Wait for last MMA to complete and signal epilogue for all subtiles
    last_buf, last_phase = _get_bufidx_phase(smem_accum_cnt - 1, NUM_SMEM_BUFFERS)
    for group_id in tl.static_range(NUM_MMA_GROUPS):
        a_buf = group_id * NUM_SMEM_BUFFERS + last_buf
        if ENABLE_PROTON:
            pl.enter_scope("mma_wait_A_empty_last")
        tlx.barrier_wait(A_smem_empty_bars[a_buf], last_phase)
        if ENABLE_PROTON:
            pl.exit_scope("mma_wait_A_empty_last")
        acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
        # Done filling this buffer, signal epilogue consumer
        tlx.barrier_arrive(tmem_full_bars[acc_buf], 1)

    return smem_accum_cnt


@triton.jit
def _process_tile_producer_inner(  # noqa: C901
    tile_id,
    num_pid_in_group,
    num_pid_m,
    num_mn_tiles,
    GROUP_SIZE_M,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    NUM_MMA_GROUPS,
    k_tile_start,
    k_tile_end,
    NUM_SMEM_BUFFERS,
    a_desc,
    b_desc,
    buffers_A,
    buffers_B,
    A_smem_full_bars,
    B_smem_full_bars,
    A_smem_empty_bars,
    B_smem_empty_bars,
    smem_accum_cnt,
    NUM_CTAS,
    cluster_cta_rank,
    A_ROW_MAJOR: tl.constexpr = True,
    B_ROW_MAJOR: tl.constexpr = True,
    REUSE_A=False,
    ENABLE_PROTON: tl.constexpr = False,
):
    """Process TMA loads for a single tile with all subtiles over [k_tile_start, k_tile_end)."""
    mn_tile_id = tile_id % num_mn_tiles
    pid_m, pid_n = _compute_pid(mn_tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
    dsize: tl.constexpr = tlx.size_of(tlx.dtype_of(b_desc))
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_SIZE_M // NUM_MMA_GROUPS
    offs_bn = pid_n * BLOCK_SIZE_N + cluster_cta_rank * (BLOCK_SIZE_N // NUM_CTAS)
    expected_bytes: tl.constexpr = dsize * BLOCK_SIZE_N * BLOCK_SIZE_K // NUM_CTAS

    local_k_tiles = k_tile_end - k_tile_start

    # Iterate along K dimension for this split's range
    for k_idx in range(0, local_k_tiles):
        k = k_tile_start + k_idx
        buf, phase = _get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)
        offs_k = k * BLOCK_SIZE_K

        offs_am = pid_m * BLOCK_SIZE_M
        # Load A for the first group, or re-signal the existing A tile.
        a_buf = buf
        if ENABLE_PROTON:
            pl.enter_scope("load_wait_A_empty_g0")
        tlx.barrier_wait(A_smem_empty_bars[a_buf], phase ^ 1)
        if ENABLE_PROTON:
            pl.exit_scope("load_wait_A_empty_g0")
        if REUSE_A:
            tlx.barrier_arrive(A_smem_full_bars[a_buf])
            tlx.barrier_arrive(A_smem_empty_bars[a_buf])
        else:
            tlx.barrier_expect_bytes(
                A_smem_full_bars[a_buf], dsize * BLOCK_M_SPLIT * BLOCK_SIZE_K
            )
            if ENABLE_PROTON:
                pl.enter_scope("load_tma_A_g0")
            if not A_ROW_MAJOR:
                tlx.async_descriptor_load(
                    a_desc,
                    buffers_A[a_buf],
                    [offs_k, offs_am],
                    A_smem_full_bars[a_buf],
                    eviction_policy="evict_last",
                )
            else:
                tlx.async_descriptor_load(
                    a_desc,
                    buffers_A[a_buf],
                    [offs_am, offs_k],
                    A_smem_full_bars[a_buf],
                    eviction_policy="evict_last",
                )
            if ENABLE_PROTON:
                pl.exit_scope("load_tma_A_g0")

        # Load B once per K iteration (shared across all subtiles)
        if ENABLE_PROTON:
            pl.enter_scope("load_wait_B_empty")
        tlx.barrier_wait(B_smem_empty_bars[buf], phase ^ 1)
        if ENABLE_PROTON:
            pl.exit_scope("load_wait_B_empty")
        tlx.barrier_expect_bytes(B_smem_full_bars[buf], expected_bytes)
        if ENABLE_PROTON:
            pl.enter_scope("load_tma_B")
        if not B_ROW_MAJOR:
            tlx.async_descriptor_load(
                b_desc,
                buffers_B[buf],
                [offs_bn, offs_k],
                B_smem_full_bars[buf],
                eviction_policy="evict_last",
            )
        else:
            tlx.async_descriptor_load(
                b_desc,
                buffers_B[buf],
                [offs_k, offs_bn],
                B_smem_full_bars[buf],
                eviction_policy="evict_last",
            )
        if ENABLE_PROTON:
            pl.exit_scope("load_tma_B")

        # Load or re-signal all remaining A subtiles for this K iteration.
        for group_id in tl.static_range(1, NUM_MMA_GROUPS):
            a_buf = group_id * NUM_SMEM_BUFFERS + buf

            if ENABLE_PROTON:
                pl.enter_scope("load_wait_A_empty")
            tlx.barrier_wait(A_smem_empty_bars[a_buf], phase ^ 1)
            if ENABLE_PROTON:
                pl.exit_scope("load_wait_A_empty")

            offs_am2 = offs_am + group_id * BLOCK_M_SPLIT

            if REUSE_A:
                tlx.barrier_arrive(A_smem_full_bars[a_buf])
                tlx.barrier_arrive(A_smem_empty_bars[a_buf])
            else:
                tlx.barrier_expect_bytes(
                    A_smem_full_bars[a_buf], dsize * BLOCK_M_SPLIT * BLOCK_SIZE_K
                )
                if ENABLE_PROTON:
                    pl.enter_scope("load_tma_A")
                if not A_ROW_MAJOR:
                    tlx.async_descriptor_load(
                        a_desc,
                        buffers_A[a_buf],
                        [offs_k, offs_am2],
                        A_smem_full_bars[a_buf],
                        eviction_policy="evict_last",
                    )
                else:
                    tlx.async_descriptor_load(
                        a_desc,
                        buffers_A[a_buf],
                        [offs_am2, offs_k],
                        A_smem_full_bars[a_buf],
                        eviction_policy="evict_last",
                    )
                if ENABLE_PROTON:
                    pl.exit_scope("load_tma_A")

        smem_accum_cnt += 1

    return smem_accum_cnt


TORCH_DTYPE_TO_TRITON = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


@triton.jit
# Triton TR001: launched with fixed reduction tile sizes from reduce_post_hook.
def _reduce_k_kernel(  # noqa: TR001
    workspace_ptr,
    c_ptr,
    M,
    N,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    base_offs = offs_m[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for s in range(SPLIT_K):
        ws_offs = base_offs + s * M * N
        partial = tl.load(workspace_ptr + ws_offs, mask=mask, other=0.0)
        acc += partial.to(tl.float32)

    tl.store(c_ptr + base_offs, acc.to(OUTPUT_DTYPE), mask=mask)


def reduce_post_hook(nargs, exception=None):
    if exception is not None:
        return
    split_k = nargs.get("SPLIT_K", 1)
    if split_k > 1:
        M = nargs["M"]
        N = nargs["N"]
        workspace = nargs["workspace_desc"].base
        c = nargs["c_desc"].base
        reduce_grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
        _reduce_k_kernel[reduce_grid](
            workspace,
            c,
            M,
            N,
            SPLIT_K=split_k,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=32,
            OUTPUT_DTYPE=TORCH_DTYPE_TO_TRITON[workspace.dtype],
            num_warps=4,
        )


@triton.autotune(
    configs=get_reduced_autotune_config()
    if DISABLE_AUTOTUNE
    else get_cuda_autotune_config(),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": preprocess_configs},
    post_hook=reduce_post_hook,
)
@triton.jit
def matmul_kernel_tma_ws_blackwell(  # noqa: C901
    a_desc,
    b_desc,
    c_desc,
    workspace_desc,
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
    NUM_CTAS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    INTERLEAVE_EPILOGUE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    A_ROW_MAJOR: tl.constexpr = True,
    B_ROW_MAJOR: tl.constexpr = True,
    USE_WARP_BARRIER: tl.constexpr = False,
    FUSED_RMSNORM: tl.constexpr = False,
    RMSNORM_EPS: tl.constexpr = 1e-6,
    NUM_X2_SUBTILES: tl.constexpr = 1,
    REUSE_RMSNORM_ACROSS_N: tl.constexpr = False,
    BF16_EPILOGUE_SCALE: tl.constexpr = False,
    NUM_EPILOGUE_SMEM_BUFFERS_OVERRIDE: tl.constexpr = 0,
    ENABLE_PROTON: tl.constexpr = False,
):
    # allocate NUM_SMEM_BUFFERS buffers
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_SIZE_M // NUM_MMA_GROUPS
    if not A_ROW_MAJOR:
        buffers_A = tlx.local_alloc(
            (BLOCK_SIZE_K, BLOCK_M_SPLIT),
            tlx.dtype_of(a_desc),
            NUM_SMEM_BUFFERS * NUM_MMA_GROUPS,
        )
    else:
        buffers_A = tlx.local_alloc(
            (BLOCK_M_SPLIT, BLOCK_SIZE_K),
            tlx.dtype_of(a_desc),
            NUM_SMEM_BUFFERS * NUM_MMA_GROUPS,
        )
    # Separate SMEM alias for epilogue's local_load of A (avoids MMA layout penalty)
    if FUSED_RMSNORM:
        if not A_ROW_MAJOR:
            buffers_A_epi = tlx.local_alloc(
                (BLOCK_SIZE_K, BLOCK_M_SPLIT),
                tlx.dtype_of(a_desc),
                NUM_SMEM_BUFFERS * NUM_MMA_GROUPS,
                reuse=buffers_A,
            )
        else:
            buffers_A_epi = tlx.local_alloc(
                (BLOCK_M_SPLIT, BLOCK_SIZE_K),
                tlx.dtype_of(a_desc),
                NUM_SMEM_BUFFERS * NUM_MMA_GROUPS,
                reuse=buffers_A,
            )
    else:
        buffers_A_epi = None
    # In 2-CTA mode, each CTA only needs to load BLOCK_N // NUM_CTAS of B.
    if not B_ROW_MAJOR:
        buffers_B = tlx.local_alloc(
            (BLOCK_SIZE_N // NUM_CTAS, BLOCK_SIZE_K),
            tlx.dtype_of(b_desc),
            NUM_SMEM_BUFFERS,
        )
    else:
        buffers_B = tlx.local_alloc(
            (BLOCK_SIZE_K, BLOCK_SIZE_N // NUM_CTAS),
            tlx.dtype_of(b_desc),
            NUM_SMEM_BUFFERS,
        )
    # NUM_TMEM_BUFFERS (overlaps MMA and epilogue)
    # Each buffer holds one subtile: BLOCK_M_SPLIT x BLOCK_SIZE_N
    # Total buffers: NUM_TMEM_BUFFERS * NUM_MMA_GROUPS
    tmem_buffers = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_SIZE_N),
        tl.float32,
        NUM_TMEM_BUFFERS * NUM_MMA_GROUPS,
        tlx.storage_kind.tmem,
    )

    # Allocate SMEM buffers for epilogue TMA store (at least 2 for multi-buffering)
    NUM_EPILOGUE_SMEM_BUFFERS: tl.constexpr = (
        NUM_EPILOGUE_SMEM_BUFFERS_OVERRIDE
        if NUM_EPILOGUE_SMEM_BUFFERS_OVERRIDE > 0
        else NUM_MMA_GROUPS
        if NUM_MMA_GROUPS > 2
        else 2
    )
    slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
    c_smem_buffers = tlx.local_alloc(
        (BLOCK_M_SPLIT, slice_size),
        tlx.dtype_of(c_desc),
        NUM_EPILOGUE_SMEM_BUFFERS,
    )

    # CTA pairs are placed along M dim
    if NUM_CTAS == 2:
        cluster_cta_rank = tlx.cluster_cta_rank()
        pred_cta0 = cluster_cta_rank == 0
        cta_bars = tlx.alloc_barriers(
            num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=2
        )  # CTA0 waits for CTA1's data before mma
    else:
        cluster_cta_rank = 0
        pred_cta0 = False
        cta_bars = None

    # allocate barriers - each subtile needs its own barriers
    # NUM_SMEM_BUFFERS barriers per subtile for synchronization
    A_smem_full_bars = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1
    )
    # arrive_count=2 when fused: both MMA (via async_dot mBarriers) and epilogue (x^2 accum) read A
    A_SMEM_EMPTY_ARRIVALS: tl.constexpr = 2 if FUSED_RMSNORM else 1
    A_smem_empty_bars = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS * NUM_MMA_GROUPS,
        arrive_count=A_SMEM_EMPTY_ARRIVALS,
    )
    B_smem_full_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    B_smem_empty_bars = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS, arrive_count=NUM_MMA_GROUPS
    )

    # NUM_TMEM_BUFFERS (overlaps MMA and epilogue)
    if USE_WARP_BARRIER:
        tmem_full_bars = tlx.alloc_warp_barrier(
            num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS, num_warps=1
        )
        tmem_empty_bars = tlx.alloc_warp_barrier(
            num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS,
            num_warps=4,
            num_arrivals=EPILOGUE_SUBTILE,
        )
    else:
        tmem_full_bars = tlx.alloc_barriers(
            num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS, arrive_count=1
        )
        tmem_empty_bars = tlx.alloc_barriers(
            num_barriers=NUM_TMEM_BUFFERS * NUM_MMA_GROUPS,
            arrive_count=EPILOGUE_SUBTILE,
        )

    with tlx.async_tasks():
        with tlx.async_task("default"):  # epilogue consumer
            (
                start_pid,
                num_pid_m,
                num_pid_n,
                num_pid_in_group,
                num_mn_tiles,
                num_tiles,
                k_tiles_total,
            ) = _compute_grid_info(
                M,
                N,
                K,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                SPLIT_K,
                NUM_CTAS,
            )

            tmem_accum_cnt = 0
            smem_accum_cnt = 0
            tile_id = start_pid
            if REUSE_RMSNORM_ACROSS_N:
                tile_id = start_pid * num_pid_n
            rrms_reg_0 = tl.zeros([BLOCK_M_SPLIT, 1], dtype=tl.float32)
            rrms_reg_1 = tl.zeros([BLOCK_M_SPLIT, 1], dtype=tl.float32)

            while tile_id < num_tiles:
                # Skip tiles whose split has zero K-tiles (last split
                # can be empty when cdiv(k_tiles_total, SPLIT_K) * (SPLIT_K-1)
                # >= k_tiles_total).
                split_id = tile_id // num_mn_tiles
                k_tiles_per_split = tl.cdiv(k_tiles_total, SPLIT_K)
                k_tile_start = split_id * k_tiles_per_split
                k_tile_end = min(k_tile_start + k_tiles_per_split, k_tiles_total)
                if k_tile_end > k_tile_start:
                    cur_tmem_buf, tmem_read_phase = _get_bufidx_phase(
                        tmem_accum_cnt, NUM_TMEM_BUFFERS
                    )
                    local_k_tiles = k_tile_end - k_tile_start
                    reuse_rrms_for_n = False
                    if REUSE_RMSNORM_ACROSS_N:
                        reuse_rrms_for_n = (tile_id % num_pid_n) != 0

                    # Accumulate x^2 during K-loop (epilogue is idle while MMA runs)
                    if FUSED_RMSNORM:
                        if not reuse_rrms_for_n:
                            x2_sum_0 = tl.zeros([BLOCK_M_SPLIT, 1], dtype=tl.float32)
                            if NUM_MMA_GROUPS >= 2:
                                x2_sum_1 = tl.zeros(
                                    [BLOCK_M_SPLIT, 1], dtype=tl.float32
                                )

                            if ENABLE_PROTON:
                                pl.enter_scope("epi_x2_accum")
                            for k_idx in range(0, local_k_tiles):
                                buf, phase = _get_bufidx_phase(
                                    smem_accum_cnt + k_idx, NUM_SMEM_BUFFERS
                                )
                                SUB_K: tl.constexpr = BLOCK_SIZE_K // NUM_X2_SUBTILES
                                for group_id in tl.static_range(NUM_MMA_GROUPS):
                                    a_buf = group_id * NUM_SMEM_BUFFERS + buf
                                    if ENABLE_PROTON:
                                        pl.enter_scope("epi_wait_A_full_x2")
                                    tlx.barrier_wait(A_smem_full_bars[a_buf], phase)
                                    if ENABLE_PROTON:
                                        pl.exit_scope("epi_wait_A_full_x2")
                                    if ENABLE_PROTON:
                                        pl.enter_scope(f"epi_x2_sum_g{group_id}")
                                    for sub_id in tl.static_range(NUM_X2_SUBTILES):
                                        if A_ROW_MAJOR:
                                            a_slice = tlx.local_slice(
                                                buffers_A_epi[a_buf],
                                                [0, sub_id * SUB_K],
                                                [BLOCK_M_SPLIT, SUB_K],
                                            )
                                        else:
                                            a_slice = tlx.local_slice(
                                                buffers_A_epi[a_buf],
                                                [sub_id * SUB_K, 0],
                                                [SUB_K, BLOCK_M_SPLIT],
                                            )
                                        a_sub = tlx.local_load(a_slice)
                                        a_f32 = a_sub.to(tl.float32)
                                        if group_id == 0:
                                            x2_sum_0 += tl.sum(
                                                a_f32 * a_f32,
                                                axis=1,
                                                keep_dims=True,
                                            )
                                        else:
                                            x2_sum_1 += tl.sum(
                                                a_f32 * a_f32,
                                                axis=1,
                                                keep_dims=True,
                                            )
                                    if ENABLE_PROTON:
                                        pl.exit_scope(f"epi_x2_sum_g{group_id}")
                                    tlx.barrier_arrive(A_smem_empty_bars[a_buf])
                            if ENABLE_PROTON:
                                pl.exit_scope("epi_x2_accum")

                            if ENABLE_PROTON:
                                pl.enter_scope("epi_rrms_compute")
                            rrms_reg_0 = tl.math.rsqrt(x2_sum_0 / K + RMSNORM_EPS)
                            if NUM_MMA_GROUPS >= 2:
                                rrms_reg_1 = tl.math.rsqrt(x2_sum_1 / K + RMSNORM_EPS)
                            if ENABLE_PROTON:
                                pl.exit_scope("epi_rrms_compute")
                        else:
                            # A_smem_empty_bars has two arrivals in fused mode: MMA
                            # and the epilogue-side A reader. When rrms is reused,
                            # the epilogue skips x2 accumulation, but it must still
                            # release A buffers so the producer can load the next
                            # N tile without deadlocking.
                            for k_idx in range(0, local_k_tiles):
                                buf, phase = _get_bufidx_phase(
                                    smem_accum_cnt + k_idx, NUM_SMEM_BUFFERS
                                )
                                for group_id in tl.static_range(NUM_MMA_GROUPS):
                                    a_buf = group_id * NUM_SMEM_BUFFERS + buf
                                    tlx.barrier_wait(A_smem_full_bars[a_buf], phase)
                                    tlx.barrier_arrive(A_smem_empty_bars[a_buf])
                        smem_accum_cnt += local_k_tiles

                    _process_tile_epilogue_inner(
                        tile_id=tile_id,
                        num_pid_in_group=num_pid_in_group,
                        num_pid_m=num_pid_m,
                        num_mn_tiles=num_mn_tiles,
                        GROUP_SIZE_M=GROUP_SIZE_M,
                        M=M,
                        BLOCK_SIZE_M=BLOCK_SIZE_M,
                        BLOCK_SIZE_N=BLOCK_SIZE_N,
                        EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
                        NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                        NUM_EPILOGUE_SMEM_BUFFERS=NUM_EPILOGUE_SMEM_BUFFERS,
                        NUM_TMEM_BUFFERS=NUM_TMEM_BUFFERS,
                        SPLIT_K=SPLIT_K,
                        INTERLEAVE_EPILOGUE=INTERLEAVE_EPILOGUE,
                        c_desc=c_desc,
                        workspace_desc=workspace_desc,
                        c_smem_buffers=c_smem_buffers,
                        tmem_buffers=tmem_buffers,
                        tmem_full_bars=tmem_full_bars,
                        tmem_empty_bars=tmem_empty_bars,
                        cur_tmem_buf=cur_tmem_buf,
                        tmem_read_phase=tmem_read_phase,
                        FUSED_RMSNORM=FUSED_RMSNORM,
                        REUSE_RMSNORM_ACROSS_N=REUSE_RMSNORM_ACROSS_N,
                        BF16_EPILOGUE_SCALE=BF16_EPILOGUE_SCALE,
                        rrms_0=rrms_reg_0,
                        rrms_1=rrms_reg_1,
                        WAIT_FOR_STORES=not REUSE_RMSNORM_ACROSS_N,
                        ENABLE_PROTON=ENABLE_PROTON,
                    )
                    tmem_accum_cnt += 1
                if REUSE_RMSNORM_ACROSS_N:
                    tile_n = tile_id % num_pid_n
                    tile_id = tl.where(
                        tile_n + 1 < num_pid_n,
                        tile_id + 1,
                        tile_id + NUM_SMS * num_pid_n - tile_n,
                    )
                else:
                    tile_id += NUM_SMS

            if REUSE_RMSNORM_ACROSS_N:
                tlx.async_descriptor_store_wait(0)

        with tlx.async_task(num_warps=1, num_regs=24):  # MMA consumer
            (
                start_pid,
                num_pid_m,
                num_pid_n,
                num_pid_in_group,
                num_mn_tiles,
                num_tiles,
                k_tiles_total,
            ) = _compute_grid_info(
                M,
                N,
                K,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                SPLIT_K,
                NUM_CTAS,
            )

            tmem_accum_cnt = 0
            smem_accum_cnt = 0
            tile_id = start_pid
            if REUSE_RMSNORM_ACROSS_N:
                tile_id = start_pid * num_pid_n

            while tile_id < num_tiles:
                # Compute K range for this split
                split_id = tile_id // num_mn_tiles
                k_tiles_per_split = tl.cdiv(k_tiles_total, SPLIT_K)
                k_tile_start = split_id * k_tiles_per_split
                k_tile_end = min(k_tile_start + k_tiles_per_split, k_tiles_total)

                # Skip tiles whose split has zero K-tiles
                if k_tile_end > k_tile_start:
                    cur_tmem_buf, tmem_write_phase = _get_bufidx_phase(
                        tmem_accum_cnt, NUM_TMEM_BUFFERS
                    )
                    smem_accum_cnt = _process_tile_mma_inner(
                        k_tiles=k_tiles_total,
                        k_tile_start=k_tile_start,
                        k_tile_end=k_tile_end,
                        NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                        NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                        NUM_TMEM_BUFFERS=NUM_TMEM_BUFFERS,
                        buffers_A=buffers_A,
                        buffers_B=buffers_B,
                        tmem_buffers=tmem_buffers,
                        A_smem_full_bars=A_smem_full_bars,
                        B_smem_full_bars=B_smem_full_bars,
                        A_smem_empty_bars=A_smem_empty_bars,
                        B_smem_empty_bars=B_smem_empty_bars,
                        tmem_full_bars=tmem_full_bars,
                        cur_tmem_buf=cur_tmem_buf,
                        tmem_empty_bars=tmem_empty_bars,
                        tmem_write_phase=tmem_write_phase,
                        smem_accum_cnt=smem_accum_cnt,
                        NUM_CTAS=NUM_CTAS,
                        cta_bars=cta_bars,
                        pred_cta0=pred_cta0,
                        A_ROW_MAJOR=A_ROW_MAJOR,
                        B_ROW_MAJOR=B_ROW_MAJOR,
                        REUSE_A=False,
                        ENABLE_PROTON=ENABLE_PROTON,
                    )
                    tmem_accum_cnt += 1
                if REUSE_RMSNORM_ACROSS_N:
                    tile_n = tile_id % num_pid_n
                    tile_id = tl.where(
                        tile_n + 1 < num_pid_n,
                        tile_id + 1,
                        tile_id + NUM_SMS * num_pid_n - tile_n,
                    )
                else:
                    tile_id += NUM_SMS

        with tlx.async_task(num_warps=1, num_regs=24):  # producer, TMA load
            (
                start_pid,
                num_pid_m,
                num_pid_n,
                num_pid_in_group,
                num_mn_tiles,
                num_tiles,
                k_tiles_total,
            ) = _compute_grid_info(
                M,
                N,
                K,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                SPLIT_K,
                NUM_CTAS,
            )

            smem_accum_cnt = 0
            tile_id = start_pid
            if REUSE_RMSNORM_ACROSS_N:
                tile_id = start_pid * num_pid_n

            while tile_id < num_tiles:
                # Compute K range for this split
                split_id = tile_id // num_mn_tiles
                k_tiles_per_split = tl.cdiv(k_tiles_total, SPLIT_K)
                k_tile_start = split_id * k_tiles_per_split
                k_tile_end = min(k_tile_start + k_tiles_per_split, k_tiles_total)

                # Skip tiles whose split has zero K-tiles
                if k_tile_end > k_tile_start:
                    smem_accum_cnt = _process_tile_producer_inner(
                        tile_id=tile_id,
                        num_pid_in_group=num_pid_in_group,
                        num_pid_m=num_pid_m,
                        num_mn_tiles=num_mn_tiles,
                        GROUP_SIZE_M=GROUP_SIZE_M,
                        BLOCK_SIZE_M=BLOCK_SIZE_M,
                        BLOCK_SIZE_N=BLOCK_SIZE_N,
                        BLOCK_SIZE_K=BLOCK_SIZE_K,
                        NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                        k_tile_start=k_tile_start,
                        k_tile_end=k_tile_end,
                        NUM_SMEM_BUFFERS=NUM_SMEM_BUFFERS,
                        a_desc=a_desc,
                        b_desc=b_desc,
                        buffers_A=buffers_A,
                        buffers_B=buffers_B,
                        A_smem_full_bars=A_smem_full_bars,
                        B_smem_full_bars=B_smem_full_bars,
                        A_smem_empty_bars=A_smem_empty_bars,
                        B_smem_empty_bars=B_smem_empty_bars,
                        smem_accum_cnt=smem_accum_cnt,
                        NUM_CTAS=NUM_CTAS,
                        cluster_cta_rank=cluster_cta_rank,
                        A_ROW_MAJOR=A_ROW_MAJOR,
                        B_ROW_MAJOR=B_ROW_MAJOR,
                        REUSE_A=False,
                        ENABLE_PROTON=ENABLE_PROTON,
                    )
                if REUSE_RMSNORM_ACROSS_N:
                    tile_n = tile_id % num_pid_n
                    tile_id = tl.where(
                        tile_n + 1 < num_pid_n,
                        tile_id + 1,
                        tile_id + NUM_SMS * num_pid_n - tile_n,
                    )
                else:
                    tile_id += NUM_SMS


def tlx_rmsnorm_matmul(  # noqa: C901
    a, b, config=None, fused_rmsnorm=True, rmsnorm_eps=1e-6
):
    """Fused RMSNorm + matmul: C = diag(rrms) @ (A @ B).

    rrms = 1/sqrt(mean(A^2, dim=-1) + eps), applied per-row.
    To use a norm weight w, pre-absorb into B: B_w = w[:, None] * B.

    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (K, N). If using norm weight w,
           pass B_w = (w[:, None] * B.float()).to(dtype) instead.
        config: Optional dict with kernel config.
        fused_rmsnorm: If True (default), fuse RMSNorm into the matmul.
        rmsnorm_eps: Epsilon for RMSNorm (default 1e-6).

    Returns:
        Output matrix C of shape (M, N)
    """
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    enable_proton = os.getenv("ENABLE_PROTON") == "1"

    # Detect column-major inputs.
    # A column-major (M, K) tensor has strides (1, M); its .T is row-major (K, M).
    a_row_major = a.is_contiguous()
    b_row_major = b.is_contiguous()

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    if not a_row_major:
        a_t = a.T  # (K, M) with strides (M, 1) — row-major
        a_desc = TensorDescriptor(a_t, a_t.shape, a_t.stride(), dummy_block)
    else:
        a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    if not b_row_major:
        b_t = b.T  # (N, K) with strides (K, 1) — row-major
        b_desc = TensorDescriptor(b_t, b_t.shape, b_t.stride(), dummy_block)
    else:
        b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

    NUM_SMS = _get_num_sms()

    # Use heuristic config if no config provided and env var is set
    use_heuristic = os.environ.get("TLX_GEMM_USE_HEURISTIC", "0") == "1"
    if config is None and use_heuristic:
        config = get_heuristic_config(M, N, K, NUM_SMS)
        if config is not None and os.environ.get("TRITON_PRINT_AUTOTUNING") == "1":
            shape_key = (M, N, K)
            if shape_key not in _printed_heuristic_configs:
                _printed_heuristic_configs.add(shape_key)
                config_str = ", ".join(
                    f"{k}: {v}"
                    for k, v in config.items()
                    if k not in ("pre_hook", "ctas_per_cga")
                )
                print(f"heuristic config selected: {config_str};")

    if config is not None:
        # Extract ctas_per_cga before removing - we need it for cluster launch
        ctas_per_cga = config.pop("ctas_per_cga", None)
        # Extract and run pre_hook if present
        pre_hook = config.pop("pre_hook", None)
        split_k = config.get("SPLIT_K", 1)
        if fused_rmsnorm and split_k > 1:
            raise ValueError(
                "Fused RMSNorm is incompatible with split-K: "
                "each split only sees part of K, so x^2 sum is wrong."
            )
        if split_k > 1:
            workspace = torch.empty((split_k * M, N), device=a.device, dtype=a.dtype)
            workspace_desc = TensorDescriptor(
                workspace, workspace.shape, workspace.stride(), dummy_block
            )
        else:
            workspace_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)
        hook_args = {
            "a_desc": a_desc,
            "b_desc": b_desc,
            "c_desc": c_desc,
            "workspace_desc": workspace_desc,
            "M": M,
            "N": N,
            "K": K,
            "A_ROW_MAJOR": a_row_major,
            "B_ROW_MAJOR": b_row_major,
            **config,
        }
        if pre_hook:
            pre_hook(hook_args)
        else:
            matmul_tma_set_block_size_hook(hook_args)
        NUM_CTAS = config.get("NUM_CTAS", 1)
        num_pid_m = triton.cdiv(M, config["BLOCK_SIZE_M"])
        num_pid_n = triton.cdiv(N, config["BLOCK_SIZE_N"])
        num_pid_m = (num_pid_m + NUM_CTAS - 1) // NUM_CTAS * NUM_CTAS
        total_tiles = num_pid_m * num_pid_n * split_k
        if config.get("REUSE_RMSNORM_ACROSS_N", False):
            total_tiles = num_pid_m * split_k
        grid = (min(NUM_SMS, total_tiles),)
        if enable_proton:
            proton_mode = proton.mode.Default(
                metric_type="cycle", optimizations="clock32"
            )
            proton.start(
                "proton", data="trace", backend="instrumentation", mode=proton_mode
            )
        matmul_kernel_tma_ws_blackwell.fn[grid](
            a_desc,
            b_desc,
            c_desc,
            workspace_desc,
            M,
            N,
            K,
            A_ROW_MAJOR=a_row_major,
            B_ROW_MAJOR=b_row_major,
            NUM_SMS=NUM_SMS,
            FUSED_RMSNORM=fused_rmsnorm,
            RMSNORM_EPS=rmsnorm_eps,
            ENABLE_PROTON=enable_proton,
            ctas_per_cga=ctas_per_cga,
            **config,
        )
        if enable_proton:
            proton.finalize()
        # Run separate reduction kernel for split-K
        if split_k > 1:
            reduce_grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
            _reduce_k_kernel[reduce_grid](
                workspace_desc.base,
                c,
                M,
                N,
                SPLIT_K=split_k,
                BLOCK_SIZE_M=32,
                BLOCK_SIZE_N=32,
                OUTPUT_DTYPE=TORCH_DTYPE_TO_TRITON[a.dtype],
                num_warps=4,
            )
    else:
        # Pass c as dummy workspace_desc. Pre_hook dynamically allocates
        # the right-sized workspace per config based on SPLIT_K.
        workspace_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)

        def grid(META):
            NUM_CTAS = META["NUM_CTAS"]
            num_pid_m = triton.cdiv(M, META["BLOCK_SIZE_M"])
            num_pid_n = triton.cdiv(N, META["BLOCK_SIZE_N"])
            # Pad num_pid_m to multiple of NUM_CTAS so CTA clusters tile evenly along M.
            num_pid_m = (num_pid_m + NUM_CTAS - 1) // NUM_CTAS * NUM_CTAS
            mn_tiles = num_pid_m * num_pid_n
            total_tiles = mn_tiles * META["SPLIT_K"]
            if META.get("REUSE_RMSNORM_ACROSS_N", False):
                total_tiles = num_pid_m * META["SPLIT_K"]
            return (min(NUM_SMS, total_tiles),)

        if enable_proton:
            proton_mode = proton.mode.Default(
                metric_type="cycle", optimizations="clock32"
            )
            proton.start(
                "proton", data="trace", backend="instrumentation", mode=proton_mode
            )
        matmul_kernel_tma_ws_blackwell[grid](
            a_desc,
            b_desc,
            c_desc,
            workspace_desc,
            M,
            N,
            K,
            A_ROW_MAJOR=a_row_major,
            B_ROW_MAJOR=b_row_major,
            NUM_SMS=NUM_SMS,
            FUSED_RMSNORM=fused_rmsnorm,
            RMSNORM_EPS=rmsnorm_eps,
            ENABLE_PROTON=enable_proton,
        )
        if enable_proton:
            proton.finalize()
        # Run split-K reduction after the autotuner picks and launches the kernel.
        # The autotuner's post_hook only runs during benchmarking, not production calls.
        best = matmul_kernel_tma_ws_blackwell.best_config
        split_k = best.kwargs.get("SPLIT_K", 1)
        if split_k > 1:
            workspace = workspace_desc.base
            reduce_grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
            _reduce_k_kernel[reduce_grid](
                workspace,
                c,
                M,
                N,
                SPLIT_K=split_k,
                BLOCK_SIZE_M=32,
                BLOCK_SIZE_N=32,
                OUTPUT_DTYPE=TORCH_DTYPE_TO_TRITON[a.dtype],
                num_warps=4,
            )
    return c
