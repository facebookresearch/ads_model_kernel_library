# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors

"""Accuracy tests for TLX fused matmul + RMSNorm backward kernels on B200."""

import unittest

from oss_test_utils import assert_close, skip_unless_blackwell, torch
from tlx_matmul_rmsnorm_bwd import tlx_matmul_rmsnorm_bwd

HAS_TLX_KERNEL = True


class TLXMatmulRMSNormBackwardTest(unittest.TestCase):
    @skip_unless_blackwell(HAS_TLX_KERNEL)
    def test_fused_rmsnorm_backward(self):
        device = "cuda"
        dtype = torch.bfloat16
        eps = 1e-5
        torch.manual_seed(42)

        M, K, N = 8192, 512, 512
        h = torch.randn(M, N, device=device, dtype=dtype)
        w2 = torch.randn(N, K, device=device, dtype=dtype)
        dy = torch.randn(M, K, device=device, dtype=dtype) * 0.01

        rms = torch.sqrt(torch.mean(h.float() ** 2, dim=-1, keepdim=True) + eps)
        rrms = (1.0 / rms).squeeze(-1).to(torch.float32)
        h_norm = (h.float() * (1.0 / rms)).to(dtype)

        dh_norm_ref = dy @ w2.T
        h_norm_f = h_norm.float()
        c = torch.mean(dh_norm_ref.float() * h_norm_f, dim=-1, keepdim=True)
        ref = ((dh_norm_ref.float() - h_norm_f * c) * rrms[:, None].float()).to(dtype)

        out = tlx_matmul_rmsnorm_bwd(
            dy,
            w2.T.contiguous(),
            h_norm=h_norm,
            rrms=rrms,
            fused_rms_norm_bwd=True,
        )

        assert_close(out, ref, atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    unittest.main()
