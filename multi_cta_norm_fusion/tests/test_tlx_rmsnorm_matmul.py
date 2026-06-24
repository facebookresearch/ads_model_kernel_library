# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors

"""Accuracy tests for TLX fused RMSNorm + matmul kernel on B200."""

import unittest

from oss_test_utils import assert_close, skip_unless_blackwell, torch
from tlx_rmsnorm_matmul import matmul_tma_set_block_size_hook, tlx_rmsnorm_matmul

HAS_TLX_KERNEL = True


def rmsnorm_matmul_reference(x, b, weight=None, eps=1e-6):
    """Reference: RMSNorm(X, weight) @ B in fp32."""
    xf = x.float()
    bf = b.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    rrms = torch.rsqrt(variance + eps)
    x_norm = xf * rrms
    if weight is not None:
        x_norm = x_norm * weight.float()[None, :]
    return (x_norm @ bf).to(x.dtype)


class TLXRMSNormMatmulTest(unittest.TestCase):
    @skip_unless_blackwell(HAS_TLX_KERNEL)
    def test_fused_rmsnorm_matmul(self):
        """Fused RMSNorm matmul: C = diag(rrms) @ (X @ B_w)."""
        device = "cuda"
        dtype = torch.bfloat16
        for M in [1024 * 64, 1024 * 128, 1024 * 512]:
            for K in [256, 512, 1024]:
                for N in [256, 512, 1024]:
                    with self.subTest(M=M, K=K, N=N):
                        torch.manual_seed(42)
                        X = torch.randn(M, K, dtype=dtype, device=device)
                        B = torch.randn(K, N, dtype=dtype, device=device)

                        ref = rmsnorm_matmul_reference(X, B)
                        out = tlx_rmsnorm_matmul(X, B, fused_rmsnorm=True)

                        assert_close(out, ref, atol=3e-2, rtol=3e-2)

    @skip_unless_blackwell(HAS_TLX_KERNEL)
    def test_plain_matmul(self):
        """Unfused matmul (fused_rmsnorm=False) matches torch.matmul."""
        device = "cuda"
        dtype = torch.bfloat16
        for M in [1024 * 64, 1024 * 128, 1024 * 512]:
            for K in [256, 512, 1024]:
                for N in [256, 512, 1024]:
                    with self.subTest(M=M, K=K, N=N):
                        torch.manual_seed(42)
                        X = torch.randn(M, K, dtype=dtype, device=device)
                        W = torch.randn(K, N, dtype=dtype, device=device)

                        ref = (X.float() @ W.float()).to(dtype)
                        out = tlx_rmsnorm_matmul(X, W, fused_rmsnorm=False)

                        assert_close(out, ref, atol=1e-2, rtol=1e-2)

    @skip_unless_blackwell(HAS_TLX_KERNEL)
    def test_bf16_epilogue_scale(self):
        """BF16 epilogue scale mode stays within fused RMSNorm matmul tolerance."""
        device = "cuda"
        torch.manual_seed(42)

        M, K, N = 8192, 512, 512
        dtype = torch.bfloat16
        X = torch.randn(M, K, dtype=dtype, device=device)
        B = torch.randn(K, N, dtype=dtype, device=device)

        ref = rmsnorm_matmul_reference(X, B)
        config = {
            "BLOCK_SIZE_M": 256,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "NUM_SMEM_BUFFERS": 2,
            "NUM_TMEM_BUFFERS": 2,
            "NUM_MMA_GROUPS": 2,
            "EPILOGUE_SUBTILE": 4,
            "NUM_CTAS": 1,
            "SPLIT_K": 1,
            "INTERLEAVE_EPILOGUE": 0,
            "USE_WARP_BARRIER": False,
            "NUM_X2_SUBTILES": 16,
            "REUSE_RMSNORM_ACROSS_N": False,
            "BF16_EPILOGUE_SCALE": True,
            "ctas_per_cga": None,
            "pre_hook": matmul_tma_set_block_size_hook,
        }
        out = tlx_rmsnorm_matmul(X, B, config=config, fused_rmsnorm=True)

        assert_close(out, ref, atol=3e-2, rtol=3e-2)


if __name__ == "__main__":
    unittest.main()
