# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors

"""Accuracy tests for TLX fused matmul + LayerNorm forward kernels on B200."""

import unittest

from oss_test_utils import assert_close, skip_unless_blackwell, torch
from tlx_matmul_layernorm import tlx_matmul_layernorm_fwd

HAS_TLX_KERNEL = True


class TLXMatmulLayerNormTest(unittest.TestCase):
    @skip_unless_blackwell(HAS_TLX_KERNEL)
    def test_fused_layernorm_forward(self):
        device = "cuda"
        dtype = torch.bfloat16
        eps = 1e-5
        torch.manual_seed(42)

        M, K, N = 8192, 512, 512
        x = torch.randn(M, K, device=device, dtype=dtype)
        w = torch.randn(K, N, device=device, dtype=dtype)
        ln_weight = torch.randn(N, device=device, dtype=dtype)
        ln_bias = torch.randn(N, device=device, dtype=dtype)

        ref = torch.nn.functional.layer_norm(x @ w, (N,), ln_weight, ln_bias, eps)
        out, mean, rstd = tlx_matmul_layernorm_fwd(
            x,
            w,
            ln_weight,
            ln_bias,
            eps,
        )

        assert_close(out, ref, atol=5e-2, rtol=5e-2)
        self.assertEqual(mean.shape, (M,))
        self.assertEqual(rstd.shape, (M,))


if __name__ == "__main__":
    unittest.main()
