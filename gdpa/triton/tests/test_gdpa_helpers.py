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
Unit tests for helper functions in triton_generalized_dot_product_attention.py
"""

import unittest
from unittest.mock import MagicMock, patch

import torch
from ads_mkl.ops.oss.gdpa.triton.triton_generalized_dot_product_attention import (
    _compute_batch_size,
    _compute_group_size,
    _compute_head_dimensions,
    _compute_kernel_strides,
    _create_output_tensor,
    _get_extra_kernel_args,
    bwd_keep,
    create_dummy_tensor,
    expect_contiguous,
    next_power_of_2,
)


class NextPowerOf2TestCase(unittest.TestCase):
    """Test next_power_of_2 function."""

    def test_powers_of_2(self) -> None:
        """Powers of 2 should return themselves."""
        for p in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            self.assertEqual(next_power_of_2(p), p)

    def test_non_powers_of_2(self) -> None:
        """Non-powers of 2 should round up."""
        self.assertEqual(next_power_of_2(3), 4)
        self.assertEqual(next_power_of_2(5), 8)
        self.assertEqual(next_power_of_2(6), 8)
        self.assertEqual(next_power_of_2(7), 8)
        self.assertEqual(next_power_of_2(9), 16)
        self.assertEqual(next_power_of_2(15), 16)
        self.assertEqual(next_power_of_2(17), 32)
        self.assertEqual(next_power_of_2(33), 64)
        self.assertEqual(next_power_of_2(65), 128)
        self.assertEqual(next_power_of_2(100), 128)
        self.assertEqual(next_power_of_2(1000), 1024)
        self.assertEqual(next_power_of_2(2000), 2048)


class ExpectContiguousTestCase(unittest.TestCase):
    """Test expect_contiguous function."""

    def test_contiguous_tensor_unchanged(self) -> None:
        """Contiguous tensor should be returned as-is."""
        t = torch.randn(4, 8, 16)
        result = expect_contiguous(t)
        self.assertIs(result, t)

    def test_non_contiguous_tensor_made_contiguous(self) -> None:
        """Non-contiguous tensor should be made contiguous."""
        t = torch.randn(16, 8, 4).transpose(0, 2)
        self.assertFalse(t.is_contiguous())
        result = expect_contiguous(t)
        self.assertTrue(result.is_contiguous())
        self.assertEqual(result.shape, t.shape)

    def test_preserves_data(self) -> None:
        """Data values should be preserved after making contiguous."""
        t = torch.randn(8, 4).transpose(0, 1)
        result = expect_contiguous(t)
        torch.testing.assert_close(result, t)


class CreateDummyTensorTestCase(unittest.TestCase):
    """Test create_dummy_tensor function."""

    def test_creates_tensor_same_device(self) -> None:
        """Dummy tensor should be on the same device as input."""
        t = torch.randn(4, 8)
        dummy = create_dummy_tensor(t)
        self.assertEqual(dummy.device, t.device)

    def test_creates_int32_tensor(self) -> None:
        """Dummy tensor should be int32 regardless of input dtype."""
        t = torch.randn(4, 8, dtype=torch.float32)
        dummy = create_dummy_tensor(t)
        self.assertEqual(dummy.dtype, torch.int32)

    def test_dummy_tensor_has_one_element(self) -> None:
        """Dummy tensor should have exactly one element."""
        t = torch.randn(4, 8)
        dummy = create_dummy_tensor(t)
        self.assertEqual(dummy.numel(), 1)

    def test_dummy_tensor_value_is_one(self) -> None:
        """Dummy tensor should contain the value 1."""
        t = torch.randn(4, 8)
        dummy = create_dummy_tensor(t)
        self.assertEqual(dummy.item(), 1)


class ComputeHeadDimensionsTestCase(unittest.TestCase):
    """Test _compute_head_dimensions function."""

    def test_standard_qkv_separate(self) -> None:
        """Test standard case with separate Q, K, V tensors."""
        query = torch.randn(10, 8, 64)
        key = torch.randn(10, 8, 64)
        value = torch.randn(10, 8, 64)
        head_q, head_k, head_v = _compute_head_dimensions(
            query, key, value, fused_qkv=False, fused_kv=False
        )
        self.assertEqual(head_q, 64)
        self.assertEqual(head_k, 64)
        self.assertEqual(head_v, 64)

    def test_fused_qkv(self) -> None:
        """Test fused QKV case (head_dim = last_dim / 3)."""
        query = torch.randn(10, 8, 192)  # 192 / 3 = 64
        key = torch.randn(10, 8, 192)
        value = torch.randn(10, 8, 192)
        head_q, head_k, head_v = _compute_head_dimensions(
            query, key, value, fused_qkv=True, fused_kv=False
        )
        self.assertEqual(head_q, 64)
        self.assertEqual(head_k, 64)
        self.assertEqual(head_v, 64)

    def test_fused_kv(self) -> None:
        """Test fused KV case (key head_dim = key.last_dim / 2)."""
        query = torch.randn(10, 8, 64)
        key = torch.randn(10, 8, 128)  # 128 / 2 = 64
        value = torch.randn(10, 8, 128)
        head_q, head_k, head_v = _compute_head_dimensions(
            query, key, value, fused_qkv=False, fused_kv=True
        )
        self.assertEqual(head_q, 64)
        self.assertEqual(head_k, 64)
        self.assertEqual(head_v, 64)

    def test_assertion_fails_when_dims_mismatch(self) -> None:
        """Test assertion when head dimensions don't match."""
        query = torch.randn(10, 8, 64)
        key = torch.randn(10, 8, 32)  # Different head dim
        value = torch.randn(10, 8, 64)
        with self.assertRaises(AssertionError):
            _compute_head_dimensions(query, key, value, fused_qkv=False, fused_kv=False)


class CreateOutputTensorTestCase(unittest.TestCase):
    """Test _create_output_tensor function."""

    def test_standard_output_shape(self) -> None:
        """Test output shape for standard (non-offset) case."""
        query = torch.randn(10, 8, 64)
        out = _create_output_tensor(
            query,
            HEAD_DIM_Q=64,
            BATCH=1,
            use_start_end_offsets=False,
            total_num_objects=0,
            broadcast_q=False,
        )
        self.assertEqual(out.shape, (10, 8, 64))
        self.assertEqual(out.dtype, query.dtype)

    def test_start_end_offsets_output_shape(self) -> None:
        """Test output shape with start/end offsets."""
        query = torch.randn(10, 8, 64)
        out = _create_output_tensor(
            query,
            HEAD_DIM_Q=64,
            BATCH=2,
            use_start_end_offsets=True,
            total_num_objects=20,
            broadcast_q=False,
        )
        self.assertEqual(out.shape, (20, 8, 64))

    def test_broadcast_q_output_shape(self) -> None:
        """Test output shape with broadcast_q=True."""
        query = torch.randn(10, 8, 64)
        out = _create_output_tensor(
            query,
            HEAD_DIM_Q=64,
            BATCH=3,
            use_start_end_offsets=False,
            total_num_objects=0,
            broadcast_q=True,
        )
        self.assertEqual(out.shape, (30, 8, 64))


class ComputeKernelStridesTestCase(unittest.TestCase):
    """Test _compute_kernel_strides function."""

    def test_separate_qkv(self) -> None:
        """Test strides for separate Q, K, V tensors."""
        query = torch.randn(10, 8, 64)
        key = torch.randn(12, 8, 64)
        value = torch.randn(12, 8, 64)
        kstrides, vstrides = _compute_kernel_strides(
            query, key, value, fused_qkv=False, fused_kv=False
        )
        self.assertEqual(kstrides, key.stride())
        self.assertEqual(vstrides, value.stride())

    def test_fused_qkv_uses_query_strides(self) -> None:
        """Test fused QKV uses query tensor strides."""
        query = torch.randn(10, 8, 192)
        key = torch.randn(10, 8, 64)
        value = torch.randn(10, 8, 64)
        kstrides, vstrides = _compute_kernel_strides(
            query, key, value, fused_qkv=True, fused_kv=False
        )
        expected = (query.stride(0), query.stride(1), query.stride(2))
        self.assertEqual(kstrides, expected)
        self.assertEqual(vstrides, expected)

    def test_fused_kv_uses_key_strides(self) -> None:
        """Test fused KV uses key tensor strides."""
        query = torch.randn(10, 8, 64)
        key = torch.randn(12, 8, 128)
        value = torch.randn(12, 8, 128)
        kstrides, vstrides = _compute_kernel_strides(
            query, key, value, fused_qkv=False, fused_kv=True
        )
        expected = (key.stride(0), key.stride(1), key.stride(2))
        self.assertEqual(kstrides, expected)
        self.assertEqual(vstrides, expected)


class GetExtraKernelArgsTestCase(unittest.TestCase):
    """Test _get_extra_kernel_args function."""

    @patch(
        "ads_mkl.ops.oss.gdpa.triton.triton_generalized_dot_product_attention.is_amd"
    )
    @patch(
        "ads_mkl.ops.oss.gdpa.triton.triton_generalized_dot_product_attention.is_mtia"
    )
    def test_nvidia_returns_empty(
        self, mock_mtia: MagicMock, mock_amd: MagicMock
    ) -> None:
        """Test NVIDIA returns empty dict."""
        mock_amd.return_value = False
        mock_mtia.return_value = False
        result = _get_extra_kernel_args(64)
        self.assertEqual(result, {})

    @patch(
        "ads_mkl.ops.oss.gdpa.triton.triton_generalized_dot_product_attention.is_amd"
    )
    @patch(
        "ads_mkl.ops.oss.gdpa.triton.triton_generalized_dot_product_attention.is_mtia"
    )
    def test_amd_small_head_dim(
        self, mock_mtia: MagicMock, mock_amd: MagicMock
    ) -> None:
        """Test AMD with small head dim returns waves_per_eu=3."""
        mock_amd.return_value = True
        mock_mtia.return_value = False
        result = _get_extra_kernel_args(64)
        self.assertEqual(result["waves_per_eu"], 3)
        self.assertTrue(result["allow_flush_denorm"])

    @patch(
        "ads_mkl.ops.oss.gdpa.triton.triton_generalized_dot_product_attention.is_amd"
    )
    @patch(
        "ads_mkl.ops.oss.gdpa.triton.triton_generalized_dot_product_attention.is_mtia"
    )
    def test_amd_large_head_dim(
        self, mock_mtia: MagicMock, mock_amd: MagicMock
    ) -> None:
        """Test AMD with large head dim returns waves_per_eu=2."""
        mock_amd.return_value = True
        mock_mtia.return_value = False
        result = _get_extra_kernel_args(128)
        self.assertEqual(result["waves_per_eu"], 2)

    @patch(
        "ads_mkl.ops.oss.gdpa.triton.triton_generalized_dot_product_attention.is_amd"
    )
    @patch(
        "ads_mkl.ops.oss.gdpa.triton.triton_generalized_dot_product_attention.is_mtia"
    )
    def test_mtia_returns_mtia_args(
        self, mock_mtia: MagicMock, mock_amd: MagicMock
    ) -> None:
        """Test MTIA returns MTIA-specific args."""
        mock_amd.return_value = False
        mock_mtia.return_value = True
        result = _get_extra_kernel_args(64)
        self.assertIn("dual_core_strategy", result)
        self.assertIn("enable_dot_boundary_prop", result)


class ComputeGroupSizeTestCase(unittest.TestCase):
    """Test _compute_group_size function."""

    def test_fused_qkv_returns_1(self) -> None:
        """Test fused QKV always returns group size 1."""
        query = torch.randn(10, 8, 192)
        key = torch.randn(10, 8, 192)
        G = _compute_group_size(query, key, fused_qkv=True)
        self.assertEqual(G, 1)

    def test_same_num_heads_returns_1(self) -> None:
        """Test same Q and K heads returns group size 1."""
        query = torch.randn(10, 8, 64)
        key = torch.randn(10, 8, 64)
        G = _compute_group_size(query, key, fused_qkv=False)
        self.assertEqual(G, 1)

    def test_gqa_group_size(self) -> None:
        """Test GQA with 8 Q heads and 2 K heads returns group 4."""
        query = torch.randn(10, 8, 64)
        key = torch.randn(10, 2, 64)
        G = _compute_group_size(query, key, fused_qkv=False)
        self.assertEqual(G, 4)

    def test_mqa_group_size(self) -> None:
        """Test MQA with 8 Q heads and 1 K head returns group 8."""
        query = torch.randn(10, 8, 64)
        key = torch.randn(10, 1, 64)
        G = _compute_group_size(query, key, fused_qkv=False)
        self.assertEqual(G, 8)

    def test_assertion_on_non_divisible(self) -> None:
        """Test assertion when Q heads not divisible by K heads."""
        query = torch.randn(10, 7, 64)
        key = torch.randn(10, 3, 64)
        with self.assertRaises(AssertionError):
            _compute_group_size(query, key, fused_qkv=False)


class ComputeBatchSizeTestCase(unittest.TestCase):
    """Test _compute_batch_size function."""

    def test_broadcast_q_uses_key_offset(self) -> None:
        """Test broadcast_q uses key_offset size."""
        query_offset = torch.tensor([0, 4, 8], dtype=torch.int32)
        key_offset = torch.tensor([0, 6, 12, 18, 24], dtype=torch.int32)
        BATCH = _compute_batch_size(
            query_offset, key_offset, broadcast_q=True, use_start_end_offsets=False
        )
        self.assertEqual(BATCH, 4)  # key_offset.size(0) - 1

    def test_standard_uses_query_offset(self) -> None:
        """Test standard case uses query_offset size."""
        query_offset = torch.tensor([0, 4, 8, 12], dtype=torch.int32)
        key_offset = torch.tensor([0, 6, 12, 18], dtype=torch.int32)
        BATCH = _compute_batch_size(
            query_offset, key_offset, broadcast_q=False, use_start_end_offsets=False
        )
        self.assertEqual(BATCH, 3)  # query_offset.size(0) - 1

    def test_start_end_offsets_divides_by_2(self) -> None:
        """Test start/end offsets divides query_offset size by 2."""
        query_offset = torch.tensor([0, 4, 8, 12, 0, 4], dtype=torch.int32)
        key_offset = torch.tensor([0, 6, 12], dtype=torch.int32)
        BATCH = _compute_batch_size(
            query_offset, key_offset, broadcast_q=False, use_start_end_offsets=True
        )
        self.assertEqual(BATCH, 3)  # query_offset.size(0) // 2


class BwdKeepTestCase(unittest.TestCase):
    """Test bwd_keep function for config filtering."""

    def test_keeps_config_when_block_n1_divisible_by_block_m1(self) -> None:
        """Test keeps configs where BLOCK_N1 % BLOCK_M1 == 0."""
        conf = MagicMock()
        conf.kwargs = {"BLOCK_N1": 128, "BLOCK_M1": 64}
        self.assertTrue(bwd_keep(conf))

    def test_rejects_config_when_block_n1_not_divisible_by_block_m1(self) -> None:
        """Test rejects configs where BLOCK_N1 % BLOCK_M1 != 0."""
        conf = MagicMock()
        conf.kwargs = {"BLOCK_N1": 128, "BLOCK_M1": 96}
        self.assertFalse(bwd_keep(conf))

    def test_keeps_config_equal_blocks(self) -> None:
        """Test keeps configs where BLOCK_N1 == BLOCK_M1."""
        conf = MagicMock()
        conf.kwargs = {"BLOCK_N1": 64, "BLOCK_M1": 64}
        self.assertTrue(bwd_keep(conf))

    def test_rejects_config_odd_ratio(self) -> None:
        """Test rejects configs with non-zero remainder."""
        conf = MagicMock()
        conf.kwargs = {"BLOCK_N1": 100, "BLOCK_M1": 64}
        self.assertFalse(bwd_keep(conf))


if __name__ == "__main__":
    unittest.main()
