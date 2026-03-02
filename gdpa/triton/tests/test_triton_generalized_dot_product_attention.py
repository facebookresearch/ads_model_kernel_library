#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for triton_generalized_dot_product_attention module.

This test suite focuses on testing utility functions, configuration functions,
and the main API entry points with proper mocking to avoid side effects.
"""

import unittest
from unittest.mock import Mock, patch

import torch

from ..triton_generalized_dot_product_attention import (
    cpu_generalized_dot_product_attention,
    create_dummy_tensor,
    expect_contiguous,
    generalized_dot_product_attention,
    is_hip,
    next_power_of_2,
)


class UtilityFunctionsTestCase(unittest.TestCase):
    """Test utility functions in the module."""

    def test_next_power_of_2_with_power_of_2(self) -> None:
        """Test next_power_of_2 returns the same value for powers of 2."""
        # Arrange & Act & Assert
        self.assertEqual(next_power_of_2(1), 1)
        self.assertEqual(next_power_of_2(2), 2)
        self.assertEqual(next_power_of_2(4), 4)
        self.assertEqual(next_power_of_2(8), 8)
        self.assertEqual(next_power_of_2(16), 16)
        self.assertEqual(next_power_of_2(32), 32)
        self.assertEqual(next_power_of_2(64), 64)
        self.assertEqual(next_power_of_2(128), 128)

    def test_next_power_of_2_with_non_power_of_2(self) -> None:
        """Test next_power_of_2 rounds up to next power of 2."""
        # Arrange & Act & Assert
        self.assertEqual(next_power_of_2(3), 4)
        self.assertEqual(next_power_of_2(5), 8)
        self.assertEqual(next_power_of_2(7), 8)
        self.assertEqual(next_power_of_2(9), 16)
        self.assertEqual(next_power_of_2(15), 16)
        self.assertEqual(next_power_of_2(17), 32)
        self.assertEqual(next_power_of_2(100), 128)

    def test_next_power_of_2_with_large_numbers(self) -> None:
        """Test next_power_of_2 with larger values."""
        # Arrange & Act & Assert
        self.assertEqual(next_power_of_2(1000), 1024)
        self.assertEqual(next_power_of_2(2000), 2048)
        self.assertEqual(next_power_of_2(5000), 8192)

    def test_expect_contiguous_with_contiguous_tensor(self) -> None:
        """Test expect_contiguous returns same tensor when already contiguous."""
        # Arrange
        tensor = torch.randn(4, 8, 16)

        # Act
        result = expect_contiguous(tensor)

        # Assert
        self.assertIs(result, tensor)
        self.assertTrue(result.is_contiguous())

    def test_expect_contiguous_with_non_contiguous_tensor(self) -> None:
        """Test expect_contiguous returns contiguous copy for non-contiguous tensor."""
        # Arrange
        tensor = torch.randn(4, 8, 16).transpose(0, 1)
        self.assertFalse(tensor.is_contiguous())

        # Act
        result = expect_contiguous(tensor)

        # Assert
        self.assertIsNot(result, tensor)
        self.assertTrue(result.is_contiguous())
        self.assertTrue(torch.equal(result, tensor))

    def test_create_dummy_tensor_creates_int32_tensor(self) -> None:
        """Test create_dummy_tensor creates a tensor with correct dtype and device."""
        # Arrange
        input_tensor = torch.randn(4, 8, device="cpu")

        # Act
        result = create_dummy_tensor(input_tensor)

        # Assert
        self.assertEqual(result.dtype, torch.int32)
        self.assertEqual(result.device, input_tensor.device)
        self.assertEqual(result.shape, (1,))
        self.assertEqual(result.item(), 1)

    @patch(
        "ads_mkl.ops.cute_dsl.gdpa.triton.triton_generalized_dot_product_attention.triton"
    )
    def test_is_hip_returns_true_when_backend_is_hip(self, mock_triton: Mock) -> None:
        """Test is_hip returns True when triton backend is HIP."""
        # Arrange
        mock_target = Mock()
        mock_target.backend = "hip"
        mock_triton.runtime.driver.active.get_current_target.return_value = mock_target

        # Act
        result = is_hip()

        # Assert
        self.assertTrue(result)

    @patch(
        "ads_mkl.ops.cute_dsl.gdpa.triton.triton_generalized_dot_product_attention.triton"
    )
    def test_is_hip_returns_false_when_backend_is_not_hip(
        self, mock_triton: Mock
    ) -> None:
        """Test is_hip returns False when triton backend is not HIP."""
        # Arrange
        mock_target = Mock()
        mock_target.backend = "cuda"
        mock_triton.runtime.driver.active.get_current_target.return_value = mock_target

        # Act
        result = is_hip()

        # Assert
        self.assertFalse(result)

    @patch(
        "ads_mkl.ops.cute_dsl.gdpa.triton.triton_generalized_dot_product_attention.triton"
    )
    def test_is_hip_handles_runtime_error(self, mock_triton: Mock) -> None:
        """Test is_hip returns False when RuntimeError is raised."""
        # Arrange
        mock_triton.runtime.driver.active.get_current_target.side_effect = RuntimeError(
            "No GPU"
        )

        # Act
        result = is_hip()

        # Assert
        self.assertFalse(result)


class CPUGeneralizedDotProductAttentionTestCase(unittest.TestCase):
    """Test CPU implementation of generalized dot product attention."""

    def test_cpu_gdpa_with_basic_inputs(self) -> None:
        """Test CPU GDPA with simple tensor inputs."""
        # Arrange
        batch_size = 2
        num_heads = 4
        head_dim = 16
        seq_len_q = 8
        seq_len_kv = 8

        query = torch.randn(seq_len_q * batch_size, num_heads, head_dim)
        key = torch.randn(seq_len_kv * batch_size, num_heads, head_dim)
        value = torch.randn(seq_len_kv * batch_size, num_heads, head_dim)

        query_offset = torch.tensor([0, seq_len_q, seq_len_q * 2], dtype=torch.int32)
        key_offset = torch.tensor([0, seq_len_kv, seq_len_kv * 2], dtype=torch.int32)

        # Act
        output = cpu_generalized_dot_product_attention(
            query=query,
            key=key,
            value=value,
            query_offset=query_offset,
            key_offset=key_offset,
            max_seq_len_q=seq_len_q,
            max_seq_len_kv=seq_len_kv,
        )

        # Assert
        self.assertEqual(output.shape, query.shape)
        self.assertEqual(output.dtype, torch.float32)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_cpu_gdpa_with_different_seq_lengths(self) -> None:
        """Test CPU GDPA with different query and key sequence lengths."""
        # Arrange
        num_heads = 2
        head_dim = 8
        seq_len_q = 4
        seq_len_kv = 6

        query = torch.randn(seq_len_q, num_heads, head_dim)
        key = torch.randn(seq_len_kv, num_heads, head_dim)
        value = torch.randn(seq_len_kv, num_heads, head_dim)

        query_offset = torch.tensor([0, seq_len_q], dtype=torch.int32)
        key_offset = torch.tensor([0, seq_len_kv], dtype=torch.int32)

        # Act
        output = cpu_generalized_dot_product_attention(
            query=query,
            key=key,
            value=value,
            query_offset=query_offset,
            key_offset=key_offset,
            max_seq_len_q=seq_len_q,
            max_seq_len_kv=seq_len_kv,
        )

        # Assert
        self.assertEqual(output.shape, query.shape)
        self.assertFalse(torch.isnan(output).any())

    def test_cpu_gdpa_with_ad_to_request_offset(self) -> None:
        """Test CPU GDPA with ad_to_request_offset for prediction mode."""
        # Arrange
        batch_size = 2
        num_heads = 2
        head_dim = 8
        seq_len_q = 4
        seq_len_kv = 4

        query = torch.randn(seq_len_q * batch_size, num_heads, head_dim)
        key = torch.randn(seq_len_kv * batch_size, num_heads, head_dim)
        value = torch.randn(seq_len_kv * batch_size, num_heads, head_dim)

        query_offset = torch.tensor([0, seq_len_q, seq_len_q * 2], dtype=torch.int32)
        key_offset = torch.tensor([0, seq_len_kv, seq_len_kv * 2], dtype=torch.int32)
        ad_to_request_offset = torch.tensor([0, 1], dtype=torch.int32)

        # Act
        output = cpu_generalized_dot_product_attention(
            query=query,
            key=key,
            value=value,
            query_offset=query_offset,
            key_offset=key_offset,
            max_seq_len_q=seq_len_q,
            max_seq_len_kv=seq_len_kv,
            ad_to_request_offset=ad_to_request_offset,
        )

        # Assert
        self.assertEqual(output.shape, query.shape)
        self.assertFalse(torch.isnan(output).any())

    def test_cpu_gdpa_raises_on_causal(self) -> None:
        """Test CPU GDPA raises assertion error when is_causal=True."""
        # Arrange
        query = torch.randn(4, 2, 8)
        key = torch.randn(4, 2, 8)
        value = torch.randn(4, 2, 8)
        query_offset = torch.tensor([0, 4], dtype=torch.int32)
        key_offset = torch.tensor([0, 4], dtype=torch.int32)

        # Act & Assert
        with self.assertRaises(AssertionError):
            cpu_generalized_dot_product_attention(
                query=query,
                key=key,
                value=value,
                query_offset=query_offset,
                key_offset=key_offset,
                max_seq_len_q=4,
                max_seq_len_kv=4,
                is_causal=True,
            )

    def test_cpu_gdpa_raises_on_attn_mask(self) -> None:
        """Test CPU GDPA raises assertion error when attn_mask is provided."""
        # Arrange
        query = torch.randn(4, 2, 8)
        key = torch.randn(4, 2, 8)
        value = torch.randn(4, 2, 8)
        query_offset = torch.tensor([0, 4], dtype=torch.int32)
        key_offset = torch.tensor([0, 4], dtype=torch.int32)
        attn_mask = torch.ones(4, 4)

        # Act & Assert
        with self.assertRaises(AssertionError):
            cpu_generalized_dot_product_attention(
                query=query,
                key=key,
                value=value,
                query_offset=query_offset,
                key_offset=key_offset,
                max_seq_len_q=4,
                max_seq_len_kv=4,
                attn_mask=attn_mask,
            )

    def test_cpu_gdpa_output_shape_matches_query(self) -> None:
        """Test CPU GDPA output has same shape as query input."""
        # Arrange
        batch_size = 3
        num_heads = 4
        head_dim = 16
        seq_len_q = 6
        seq_len_kv = 8

        query = torch.randn(seq_len_q * batch_size, num_heads, head_dim)
        key = torch.randn(seq_len_kv * batch_size, num_heads, head_dim)
        value = torch.randn(seq_len_kv * batch_size, num_heads, head_dim)

        query_offset = torch.tensor(
            [0, seq_len_q, seq_len_q * 2, seq_len_q * 3], dtype=torch.int32
        )
        key_offset = torch.tensor(
            [0, seq_len_kv, seq_len_kv * 2, seq_len_kv * 3], dtype=torch.int32
        )

        # Act
        output = cpu_generalized_dot_product_attention(
            query=query,
            key=key,
            value=value,
            query_offset=query_offset,
            key_offset=key_offset,
            max_seq_len_q=seq_len_q,
            max_seq_len_kv=seq_len_kv,
        )

        # Assert
        self.assertEqual(output.shape, query.shape)
        self.assertEqual(output.dtype, torch.float32)


class GeneralizedDotProductAttentionTestCase(unittest.TestCase):
    """Test main API for generalized dot product attention."""

    @patch(
        "ads_mkl.ops.cute_dsl.gdpa.triton.triton_generalized_dot_product_attention.torch.jit.is_tracing"
    )
    @patch(
        "ads_mkl.ops.cute_dsl.gdpa.triton.triton_generalized_dot_product_attention.cpu_generalized_dot_product_attention"
    )
    def test_gdpa_calls_cpu_version_when_tracing(
        self, mock_cpu_gdpa: Mock, mock_is_tracing: Mock
    ) -> None:
        """Test GDPA calls CPU implementation during torch.jit tracing."""
        # Arrange
        mock_is_tracing.return_value = True
        query = torch.randn(4, 2, 8)
        key = torch.randn(4, 2, 8)
        value = torch.randn(4, 2, 8)
        query_offset = torch.tensor([0, 4], dtype=torch.int32)
        key_offset = torch.tensor([0, 4], dtype=torch.int32)
        mock_cpu_gdpa.return_value = torch.zeros_like(query)

        # Act
        result = generalized_dot_product_attention(
            query=query,
            key=key,
            value=value,
            query_offset=query_offset,
            key_offset=key_offset,
            max_seq_len_q=4,
            max_seq_len_kv=4,
        )

        # Assert
        mock_cpu_gdpa.assert_called_once()
        self.assertIsNotNone(result)

    @patch(
        "ads_mkl.ops.cute_dsl.gdpa.triton.triton_generalized_dot_product_attention.torch.jit.is_scripting"
    )
    @patch(
        "ads_mkl.ops.cute_dsl.gdpa.triton.triton_generalized_dot_product_attention.torch.jit.is_tracing"
    )
    @patch(
        "ads_mkl.ops.cute_dsl.gdpa.triton.triton_generalized_dot_product_attention.cpu_generalized_dot_product_attention"
    )
    def test_gdpa_calls_cpu_version_when_scripting(
        self, mock_cpu_gdpa: Mock, mock_is_tracing: Mock, mock_is_scripting: Mock
    ) -> None:
        """Test GDPA calls CPU implementation during torch.jit scripting."""
        # Arrange
        mock_is_tracing.return_value = False
        mock_is_scripting.return_value = True
        query = torch.randn(4, 2, 8)
        key = torch.randn(4, 2, 8)
        value = torch.randn(4, 2, 8)
        query_offset = torch.tensor([0, 4], dtype=torch.int32)
        key_offset = torch.tensor([0, 4], dtype=torch.int32)
        mock_cpu_gdpa.return_value = torch.zeros_like(query)

        # Act
        result = generalized_dot_product_attention(
            query=query,
            key=key,
            value=value,
            query_offset=query_offset,
            key_offset=key_offset,
            max_seq_len_q=4,
            max_seq_len_kv=4,
        )

        # Assert
        mock_cpu_gdpa.assert_called_once()
        self.assertIsNotNone(result)

    @patch(
        "ads_mkl.ops.cute_dsl.gdpa.triton.triton_generalized_dot_product_attention.torch.ops.gdpa.generalized_dot_product_attention"
    )
    @patch(
        "ads_mkl.ops.cute_dsl.gdpa.triton.triton_generalized_dot_product_attention.torch.jit.is_scripting"
    )
    @patch(
        "ads_mkl.ops.cute_dsl.gdpa.triton.triton_generalized_dot_product_attention.torch.jit.is_tracing"
    )
    def test_gdpa_calls_triton_op_when_not_jit(
        self, mock_is_tracing: Mock, mock_is_scripting: Mock, mock_triton_op: Mock
    ) -> None:
        """Test GDPA calls torch.ops implementation when not in JIT mode."""
        # Arrange
        mock_is_tracing.return_value = False
        mock_is_scripting.return_value = False
        query = torch.randn(4, 2, 8)
        key = torch.randn(4, 2, 8)
        value = torch.randn(4, 2, 8)
        query_offset = torch.tensor([0, 4], dtype=torch.int32)
        key_offset = torch.tensor([0, 4], dtype=torch.int32)
        mock_triton_op.return_value = torch.zeros_like(query)

        # Act
        result = generalized_dot_product_attention(
            query=query,
            key=key,
            value=value,
            query_offset=query_offset,
            key_offset=key_offset,
            max_seq_len_q=4,
            max_seq_len_kv=4,
        )

        # Assert
        mock_triton_op.assert_called_once()
        self.assertIsNotNone(result)

    @patch(
        "ads_mkl.ops.cute_dsl.gdpa.triton.triton_generalized_dot_product_attention.torch.ops.gdpa.generalized_dot_product_attention"
    )
    @patch(
        "ads_mkl.ops.cute_dsl.gdpa.triton.triton_generalized_dot_product_attention.torch.jit.is_scripting"
    )
    @patch(
        "ads_mkl.ops.cute_dsl.gdpa.triton.triton_generalized_dot_product_attention.torch.jit.is_tracing"
    )
    def test_gdpa_makes_tensors_contiguous(
        self, mock_is_tracing: Mock, mock_is_scripting: Mock, mock_triton_op: Mock
    ) -> None:
        """Test GDPA ensures input tensors are contiguous before calling triton op."""
        # Arrange
        mock_is_tracing.return_value = False
        mock_is_scripting.return_value = False
        query = torch.randn(8, 4, 2).transpose(0, 2)  # Non-contiguous
        key = torch.randn(8, 4, 2).transpose(0, 2)  # Non-contiguous
        value = torch.randn(8, 4, 2).transpose(0, 2)  # Non-contiguous
        query_offset = torch.tensor([0, 4], dtype=torch.int32)
        key_offset = torch.tensor([0, 4], dtype=torch.int32)
        mock_triton_op.return_value = torch.zeros(4, 2, 8)

        # Act
        generalized_dot_product_attention(
            query=query,
            key=key,
            value=value,
            query_offset=query_offset,
            key_offset=key_offset,
            max_seq_len_q=4,
            max_seq_len_kv=4,
        )

        # Assert
        mock_triton_op.assert_called_once()
        # Verify that contiguous tensors were passed
        call_args = mock_triton_op.call_args
        self.assertTrue(call_args.kwargs["query"].is_contiguous())
        self.assertTrue(call_args.kwargs["key"].is_contiguous())
        self.assertTrue(call_args.kwargs["value"].is_contiguous())


if __name__ == "__main__":
    unittest.main()
