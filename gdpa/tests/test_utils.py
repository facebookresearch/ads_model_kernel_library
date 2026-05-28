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
Unit tests for ads_mkl/ops/oss/gdpa/utils/utils.py
"""

import logging
import os
import unittest
from unittest.mock import MagicMock, patch

import torch
from ads_mkl.ops.oss.gdpa.utils.utils import (
    dump_kernel_info,
    DUMP_KERNEL_INFO_ENV_VAR,
    get_num_sms,
    get_num_warps,
    should_use_i64_idx,
)


class GetNumWarpsTestCase(unittest.TestCase):
    """Test get_num_warps function."""

    def test_get_num_warps_returns_int(self) -> None:
        """Test that get_num_warps returns an integer."""
        result = get_num_warps()
        self.assertIsInstance(result, int)
        self.assertIn(result, [16, 32])


class ShouldUseI64IdxTestCase(unittest.TestCase):
    """Test should_use_i64_idx function."""

    def test_small_tensors_return_false(self) -> None:
        """Test small tensors don't need 64-bit indexing."""
        t1 = torch.randn(10, 10)
        t2 = torch.randn(100, 100)
        self.assertFalse(should_use_i64_idx(t1, t2))

    def test_large_tensor_returns_true(self) -> None:
        """Test tensor with >= 2^31 elements needs 64-bit indexing."""
        # Create a mock tensor that pretends to have >= 2^31 elements
        large_tensor = MagicMock(spec=torch.Tensor)
        large_tensor.numel.return_value = 2**31
        self.assertTrue(should_use_i64_idx(large_tensor))

    def test_empty_args_return_false(self) -> None:
        """Test no arguments returns False."""
        self.assertFalse(should_use_i64_idx())

    def test_single_small_tensor(self) -> None:
        """Test single small tensor returns False."""
        t = torch.randn(10)
        self.assertFalse(should_use_i64_idx(t))

    def test_one_large_among_many_returns_true(self) -> None:
        """Test that if any tensor is large, returns True."""
        t1 = torch.randn(10)
        large_tensor = MagicMock(spec=torch.Tensor)
        large_tensor.numel.return_value = 2**31
        t3 = torch.randn(10)
        self.assertTrue(should_use_i64_idx(t1, large_tensor, t3))

    def test_just_below_threshold_returns_false(self) -> None:
        """Test tensor just below 2^31 elements returns False."""
        tensor = MagicMock(spec=torch.Tensor)
        tensor.numel.return_value = 2**31 - 1
        self.assertFalse(should_use_i64_idx(tensor))


class DumpKernelInfoTestCase(unittest.TestCase):
    """Test dump_kernel_info function."""

    def test_dump_disabled_by_default(self) -> None:
        """Test that dump is disabled when env var is not set."""
        kernel_info = MagicMock()
        with patch.dict(os.environ, {}, clear=True):
            # Should not raise and not write any files
            dump_kernel_info(kernel_info)

    def test_dump_disabled_when_env_var_is_zero(self) -> None:
        """Test that dump is disabled when env var is '0'."""
        kernel_info = MagicMock()
        with patch.dict(os.environ, {DUMP_KERNEL_INFO_ENV_VAR: "0"}):
            dump_kernel_info(kernel_info)

    @patch("builtins.open", create=True)
    def test_dump_enabled_when_env_var_is_one(self, mock_open: MagicMock) -> None:
        """Test that dump writes files when env var is '1'."""
        kernel_info = MagicMock()
        kernel_info.metadata.name = "test_kernel"
        kernel_info.n_spills = 0
        kernel_info.n_regs = 32
        kernel_info.asm = {
            "ttir": "ttir_content",
            "ttgir": "ttgir_content",
            "llir": "llir_content",
            "ptx": "ptx_content",
        }
        with patch.dict(os.environ, {DUMP_KERNEL_INFO_ENV_VAR: "1"}):
            with self.assertLogs(level=logging.INFO):
                dump_kernel_info(kernel_info)


class GetNumSmsTestCase(unittest.TestCase):
    """Test get_num_sms function."""

    @patch("ads_mkl.ops.oss.gdpa.utils.utils.torch.cuda.is_available")
    def test_returns_none_when_cuda_not_available(
        self, mock_cuda_available: MagicMock
    ) -> None:
        """Test returns None when CUDA is not available."""
        mock_cuda_available.return_value = False
        # Clear lru_cache
        get_num_sms.cache_clear()
        result = get_num_sms()
        self.assertIsNone(result)

    @patch("ads_mkl.ops.oss.gdpa.utils.utils.torch.cuda.get_device_properties")
    @patch("ads_mkl.ops.oss.gdpa.utils.utils.torch.cuda.is_available")
    def test_returns_sm_count_when_cuda_available(
        self, mock_cuda_available: MagicMock, mock_get_props: MagicMock
    ) -> None:
        """Test returns SM count when CUDA is available."""
        mock_cuda_available.return_value = True
        mock_props = MagicMock()
        mock_props.multi_processor_count = 108
        mock_get_props.return_value = mock_props
        # Clear lru_cache
        get_num_sms.cache_clear()
        result = get_num_sms()
        self.assertEqual(result, 108)


if __name__ == "__main__":
    unittest.main()
