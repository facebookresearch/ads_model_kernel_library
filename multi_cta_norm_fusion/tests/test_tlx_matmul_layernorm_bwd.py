# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors

"""Accuracy tests for TLX fused matmul + LayerNorm backward kernels on B200."""

import unittest

from oss_test_utils import assert_close, skip_unless_blackwell, torch
from tlx_matmul_layernorm_bwd import tlx_matmul_layernorm_bwd

HAS_TLX_KERNEL = True


class TLXMatmulLayerNormBackwardTest(unittest.TestCase):
    @skip_unless_blackwell(HAS_TLX_KERNEL)
    def test_fused_layernorm_backward(self):
        device = "cuda"
        dtype = torch.bfloat16
        eps = 1e-5
        torch.manual_seed(42)

        M, K, N = 8192, 512, 512
        h = torch.randn(M, N, device=device, dtype=dtype)
        w2 = torch.randn(N, K, device=device, dtype=dtype)
        dy = torch.randn(M, K, device=device, dtype=dtype) * 0.01

        mean = torch.mean(h.float(), dim=-1, keepdim=True)
        h_centered = h.float() - mean
        rstd_full = torch.rsqrt(
            torch.mean(h_centered * h_centered, dim=-1, keepdim=True) + eps
        )
        rstd = rstd_full.squeeze(-1).to(torch.float32)
        h_norm = (h_centered * rstd_full).to(dtype)

        dh_norm_ref = dy @ w2.T
        h_norm_f = h_norm.float()
        mean_dh = torch.mean(dh_norm_ref.float(), dim=-1, keepdim=True)
        mean_dh_xhat = torch.mean(
            dh_norm_ref.float() * h_norm_f,
            dim=-1,
            keepdim=True,
        )
        ref = (
            (dh_norm_ref.float() - mean_dh - h_norm_f * mean_dh_xhat)
            * rstd[:, None].float()
        ).to(dtype)

        out = tlx_matmul_layernorm_bwd(
            dy,
            w2.T.contiguous(),
            h_norm=h_norm,
            rstd=rstd,
            fused_layer_norm_bwd=True,
        )

        assert_close(out, ref, atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    unittest.main()
