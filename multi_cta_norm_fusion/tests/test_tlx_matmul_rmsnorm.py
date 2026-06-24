# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors

"""Accuracy tests for TLX fused matmul + RMSNorm forward kernels on B200."""

import unittest

from oss_test_utils import assert_close, skip_unless_blackwell, torch
from tlx_matmul_rmsnorm import tlx_matmul

HAS_TLX_KERNEL = True


class TLXMatmulRMSNormTest(unittest.TestCase):
    @skip_unless_blackwell(HAS_TLX_KERNEL)
    def test_fused_rmsnorm_forward(self):
        device = "cuda"
        dtype = torch.bfloat16
        torch.manual_seed(42)

        M, K, N = 8192, 512, 512
        x = torch.randn(M, K, device=device, dtype=dtype)
        w = torch.randn(K, N, device=device, dtype=dtype)

        ref = torch.nn.functional.rms_norm(x @ w, (N,))
        out = tlx_matmul(x, w, fused_rms_norm=True)

        assert_close(out, ref, atol=3e-2, rtol=3e-2)


if __name__ == "__main__":
    unittest.main()
