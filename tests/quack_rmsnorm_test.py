# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.

# pyre-ignore-all-errors
import unittest

import torch
from hypothesis import given, settings, strategies as st, Verbosity

try:
    from ads_mkl.ops.cute_dsl.quack.quack_rmsnorm import (
        quack_rmsnorm,
        rmsnorm,
        rmsnorm_ref,
    )

    QUACK_AVAILABLE = True
except ImportError as e:
    # Handle cases where CUDA dependencies are not available
    QUACK_AVAILABLE = False
    import warnings

    warnings.warn(
        f"Quack import failed: {e}. Some tests will be skipped.", stacklevel=2
    )

# test modified from https://github.com/Dao-AILab/quack/blob/main/tests/test_rmsnorm.py


class QuackRMSNormTest(unittest.TestCase):
    @unittest.skipIf(not QUACK_AVAILABLE, "Quack dependencies not available")
    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        "need at least B200 GPU or no GPU available",
    )
    # pyre-ignore
    @given(
        M=st.sampled_from([1, 37, 199, 8 * 1024]),
        N=st.sampled_from(
            [
                192,
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
            ]
        ),
        input_dtype=st.sampled_from([torch.bfloat16, torch.float32]),
        weight_dtype=st.sampled_from([torch.bfloat16, torch.float32]),
        eps=st.sampled_from([1e-5]),
        use_compile=st.sampled_from([False]),
    )
    @settings(
        deadline=None,
        verbosity=Verbosity.verbose,
        max_examples=10,
    )
    def test_rmsnorm_forward_backward(
        self, M, N, input_dtype, weight_dtype, eps, use_compile
    ):
        """Test RMSNorm forward pass against reference implementation."""
        if N >= 256 * 1024 and input_dtype == torch.float32 and M >= 8 * 1024:
            # Skipping large tensor test for float32 to avoid OOM
            return
        device = "cuda"
        # Set tolerance based on dtype
        if input_dtype == torch.bfloat16:
            atol = 1e-1
        elif input_dtype == torch.float16:
            atol = 1e-2
        else:
            atol = 1e-4
        torch.random.manual_seed(0)
        x = torch.randn(M, N, device=device, dtype=input_dtype, requires_grad=True)
        weight = torch.randn(N, device=device, dtype=weight_dtype, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_()
        weight_ref = weight.detach().clone().requires_grad_()
        function = torch.compile(rmsnorm, fullgraph=True) if use_compile else rmsnorm
        out = function(x, weight, eps=eps)[0]
        out_ref = rmsnorm_ref(x_ref, weight_ref, eps=eps)

        assert out.shape == x.shape
        assert out.dtype == input_dtype
        torch.testing.assert_close(out, out_ref, atol=atol, rtol=1e-3)
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
                2 * (weight_ref.grad + 0.3 - 0.3 - weight_ref.grad).abs().max()
            )
        torch.testing.assert_close(
            weight.grad, weight_ref.grad, atol=weight_atol, rtol=1e-3
        )

    @unittest.skipIf(not QUACK_AVAILABLE, "Quack dependencies not available")
    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        "need at least B200 GPU or no GPU available",
    )
    # pyre-ignore
    @given(
        use_compile=st.sampled_from([False]),
    )
    @settings(
        deadline=None,
        verbosity=Verbosity.verbose,
        max_examples=5,
    )
    def test_rmsnorm_with_residual(self, use_compile):
        """Test RMSNorm with residual connection - both forward and backward."""
        device = "cuda"
        M, N = 32, 1024
        eps = 1e-6
        input_dtype = torch.float16
        weight_dtype = torch.float32

        torch.random.manual_seed(0)
        x = torch.randn(M, N, device=device, dtype=input_dtype, requires_grad=True)
        weight = torch.randn(N, device=device, dtype=weight_dtype, requires_grad=True)
        residual = torch.randn(
            M, N, device=device, dtype=input_dtype, requires_grad=True
        )

        x_ref = x.detach().clone().requires_grad_()
        weight_ref = weight.detach().clone().requires_grad_()
        residual_ref = residual.detach().clone().requires_grad_()

        function = torch.compile(rmsnorm, fullgraph=True) if use_compile else rmsnorm
        out, residual_out, _ = function(x, weight, residual=residual, eps=eps)
        out_ref, residual_out_ref = rmsnorm_ref(
            x_ref, weight_ref, residual=residual_ref, eps=eps
        )

        assert out.shape == x.shape
        assert out.dtype == input_dtype
        torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-3)
        torch.testing.assert_close(residual_out, residual_out_ref, atol=1e-2, rtol=1e-3)

        grad_out = torch.randn_like(out)
        torch.cuda.synchronize()
        out_ref.backward(grad_out)
        out.backward(grad_out)
        torch.testing.assert_close(x.grad, x_ref.grad, atol=1e-2, rtol=1e-3)
        torch.testing.assert_close(weight.grad, weight_ref.grad, atol=1e-2, rtol=1e-3)
        torch.testing.assert_close(
            residual.grad, residual_ref.grad, atol=1e-2, rtol=1e-3
        )
        print("residual grad", residual.grad)
        print("residual ref grad", residual_ref.grad)

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        "need at least B200 GPU or no GPU available",
    )
    # pyre-ignore
    @given(
        M=st.sampled_from([32 * 1024]),
        N=st.sampled_from([131072, 262144]),
        input_dtype=st.sampled_from([torch.bfloat16]),
        eps=st.sampled_from([1e-5]),
    )
    @settings(
        deadline=None,
        verbosity=Verbosity.verbose,
        max_examples=5,
    )
    def test_rmsnorm_large_tensor(
        self, M: int, N: int, input_dtype: torch.dtype, eps: float
    ):
        """Test RMSNorm forward pass against reference implementation."""
        device = "cuda"
        # Set tolerance based on dtype
        if input_dtype == torch.bfloat16:
            atol = 1e-1
        elif input_dtype == torch.float16:
            atol = 1e-2
        else:
            atol = 1e-4
        torch.random.manual_seed(0)
        torch.cuda.empty_cache()
        x = torch.randn(M, N, device=device, dtype=input_dtype, requires_grad=False)
        weight = torch.randn(N, device=device, dtype=torch.float32, requires_grad=False)
        out = rmsnorm(x, weight, eps=eps)[0]
        # Need to compile, otherwise it OOMs
        rmsnorm_compiled = torch.compile(rmsnorm_ref)
        # Run once with smaller input to avoid OOMs
        rmsnorm_compiled(x[:32], weight, eps=eps)
        out_ref = rmsnorm_compiled(x, weight, eps=eps)
        # Need to chunk, otherwise it OOMs
        assert all(
            (out_c - out_ref_c).abs().max() < atol
            for out_c, out_ref_c in zip(out.chunk(16), out_ref.chunk(16))
        )

    @unittest.skipIf(not QUACK_AVAILABLE, "Quack dependencies not available")
    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        "need at least B200 GPU or no GPU available",
    )
    def test_quack_rmsnorm_cuda(self):
        """Test quack_rmsnorm wrapper with CUDA."""
        device = "cuda"
        M, N = 32, 128
        eps = 1e-5

        x = torch.randn(M, N, device=device, dtype=torch.float16)
        weight = torch.randn(N, device=device, dtype=torch.float32)

        # Test CUDA execution (should use rmsnorm)
        out, _, rstd = quack_rmsnorm(x, weight, eps=eps)

        # Verify output properties
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        assert rstd.shape == (M,)

        # Test against reference
        x_ref = x.detach().clone().requires_grad_()
        weight_ref = weight.detach().clone().requires_grad_()
        out_ref = rmsnorm_ref(x_ref, weight_ref, eps=eps)

        torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-3)
