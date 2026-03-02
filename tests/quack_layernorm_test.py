# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

# pyre-ignore-all-errors
import unittest

import torch
from ads_mkl.ops.cute_dsl.quack.quack_layernorm import (
    layernorm,
    layernorm_ref,
    mean_ref,
    rstd_ref,
)
from hypothesis import given, settings, strategies as st, Verbosity

# test modified from https://github.com/Dao-AILab/quack/blob/main/tests/test_layernorm.py


class QuackLayerNormTest(unittest.TestCase):
    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        "need at least B200 GPU or no GPU available",
    )
    # pyre-ignore
    @given(
        M=st.sampled_from([1, 37, 199]),
        N=st.sampled_from(
            [
                256,
                512,
                760,
                1024,
                1128,
                2048,
                4096,
                8192,
                16384,
                32768,
                65536,
                131072,
                262144,
            ]
        ),
        input_dtype=st.sampled_from([torch.float16, torch.bfloat16, torch.float32]),
        weight_dtype=st.sampled_from([torch.float16, torch.bfloat16, torch.float32]),
        eps=st.sampled_from([1e-5, 1e-6]),
    )
    @settings(
        deadline=None,
        verbosity=Verbosity.verbose,
        max_examples=5,
    )
    def test_layernorm_forward_backward(
        self, M, N, input_dtype, weight_dtype: torch.dtype, eps
    ):
        """Test LayerNorm forward+backward pass against reference implementation."""
        device = "cuda"

        # tolerance depends on precision
        if input_dtype == torch.bfloat16:
            atol = 1e-2
            rtol = 1e-2
        elif input_dtype == torch.float16:
            atol = 1e-3
            rtol = 1e-3
        else:
            atol = 1e-4
            rtol = 1e-4

        torch.random.manual_seed(0)
        x = torch.randn(M, N, device=device, dtype=input_dtype, requires_grad=True)
        weight = torch.randn(N, device=device, dtype=torch.float32, requires_grad=True)

        # pure‐PyTorch refs
        x_ref = x.detach().clone().requires_grad_()
        weight_ref = weight.detach().clone().requires_grad_()

        # case 1: without bias
        out, rstd, mean = layernorm(x, weight, eps=eps)
        out_ref = layernorm_ref(x_ref, weight_ref, eps=eps)
        rstd_ref_val = rstd_ref(x_ref, eps=eps)
        mean_ref_val = mean_ref(x_ref)

        # shapes & dtypes
        assert out.shape == x.shape
        assert out.dtype == input_dtype
        assert rstd.shape == (M,) and rstd.dtype == torch.float32
        assert mean.shape == (M,) and mean.dtype == torch.float32

        # numeric check
        torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(rstd, rstd_ref_val, atol=6e-4, rtol=6e-4)
        torch.testing.assert_close(mean, mean_ref_val, atol=6e-4, rtol=6e-4)

        # case 2: with bias
        x = torch.randn(M, N, device=device, dtype=input_dtype, requires_grad=True)
        weight = torch.randn(N, device=device, dtype=torch.float32, requires_grad=True)
        bias = torch.randn(N, device=device, dtype=torch.float32, requires_grad=True)

        # pure‐PyTorch refs
        x_ref = x.detach().clone().requires_grad_()
        weight_ref = weight.detach().clone().requires_grad_()
        bias_ref = bias.detach().clone().requires_grad_()

        out, rstd, mean = layernorm(x, weight, eps=eps, bias=bias)
        out_ref = layernorm_ref(x_ref, weight_ref, eps=eps, bias=bias_ref)
        rstd_ref_val = rstd_ref(x_ref, eps=eps)
        mean_ref_val = mean_ref(x_ref)

        # shapes & dtypes
        assert out.shape == x.shape
        assert out.dtype == input_dtype
        assert rstd.shape == (M,) and rstd.dtype == torch.float32
        assert mean.shape == (M,) and mean.dtype == torch.float32

        # numeric check
        torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(rstd, rstd_ref_val, atol=6e-4, rtol=6e-4)
        torch.testing.assert_close(mean, mean_ref_val, atol=6e-4, rtol=6e-4)

        # Backward pass
        if N > 128 * 1024 and input_dtype == torch.float32:
            # Skip backward pass for due to not enough smem
            return
        grad_out = torch.randn_like(out)
        torch.cuda.synchronize()
        out_ref.backward(grad_out)
        out.backward(grad_out)
        torch.testing.assert_close(x.grad, x_ref.grad, atol=atol, rtol=1e-3)
        if weight_dtype == torch.float32:
            weight_atol = 1e-4
        else:
            weight_atol = (
                # pyre-fixme[16]: `float` has no attribute `abs`
                2 * (weight_ref.grad + 0.3 - 0.3 - weight_ref.grad).abs().max()
            )
        torch.testing.assert_close(
            weight.grad, weight_ref.grad, atol=weight_atol, rtol=1e-3
        )
