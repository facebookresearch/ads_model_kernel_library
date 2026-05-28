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
Unit tests for ads_mkl/ops/oss/gdpa/utils/tma_utils.py
"""

import unittest
from unittest.mock import MagicMock, patch

import torch
from ads_mkl.ops.oss.gdpa.utils.tma_utils import is_tma_supported, TmaAutoTuneHelper


class IsTmaSupportedTestCase(unittest.TestCase):
    """Test is_tma_supported function."""

    @patch("ads_mkl.ops.oss.gdpa.utils.tma_utils.torch.cuda.is_available")
    def test_returns_false_when_cuda_not_available(
        self, mock_available: MagicMock
    ) -> None:
        """Test returns False when CUDA is not available."""
        mock_available.return_value = False
        self.assertFalse(is_tma_supported())

    @patch("ads_mkl.ops.oss.gdpa.utils.tma_utils.torch.version")
    @patch("ads_mkl.ops.oss.gdpa.utils.tma_utils.torch.cuda.get_device_capability")
    @patch("ads_mkl.ops.oss.gdpa.utils.tma_utils.torch.cuda.is_available")
    def test_returns_true_for_sm90_with_cuda_12_4(
        self,
        mock_available: MagicMock,
        mock_capability: MagicMock,
        mock_version: MagicMock,
    ) -> None:
        """Test returns True for SM90+ with CUDA >= 12.4."""
        mock_available.return_value = True
        mock_capability.return_value = (9, 0)
        mock_version.cuda = "12.4"
        self.assertTrue(is_tma_supported())

    @patch("ads_mkl.ops.oss.gdpa.utils.tma_utils.torch.version")
    @patch("ads_mkl.ops.oss.gdpa.utils.tma_utils.torch.cuda.get_device_capability")
    @patch("ads_mkl.ops.oss.gdpa.utils.tma_utils.torch.cuda.is_available")
    def test_returns_false_for_sm80(
        self,
        mock_available: MagicMock,
        mock_capability: MagicMock,
        mock_version: MagicMock,
    ) -> None:
        """Test returns False for SM80 (below SM90)."""
        mock_available.return_value = True
        mock_capability.return_value = (8, 0)
        mock_version.cuda = "12.4"
        self.assertFalse(is_tma_supported())

    @patch("ads_mkl.ops.oss.gdpa.utils.tma_utils.torch.version")
    @patch("ads_mkl.ops.oss.gdpa.utils.tma_utils.torch.cuda.get_device_capability")
    @patch("ads_mkl.ops.oss.gdpa.utils.tma_utils.torch.cuda.is_available")
    def test_returns_false_for_old_cuda(
        self,
        mock_available: MagicMock,
        mock_capability: MagicMock,
        mock_version: MagicMock,
    ) -> None:
        """Test returns False for CUDA < 12.4."""
        mock_available.return_value = True
        mock_capability.return_value = (9, 0)
        mock_version.cuda = "12.3"
        self.assertFalse(is_tma_supported())

    @patch("ads_mkl.ops.oss.gdpa.utils.tma_utils.torch.cuda.is_available")
    def test_returns_false_on_exception(self, mock_available: MagicMock) -> None:
        """Test returns False on any exception."""
        mock_available.side_effect = RuntimeError("No CUDA")
        self.assertFalse(is_tma_supported())


class TmaAutoTuneHelperTestCase(unittest.TestCase):
    """Test TmaAutoTuneHelper constants and KernelParamWrapper."""

    def test_tma_size_constant(self) -> None:
        """Test TMA_SIZE is 128."""
        self.assertEqual(TmaAutoTuneHelper.TMA_SIZE, 128)

    def test_kernel_param_wrapper_returns_pointer(self) -> None:
        """Test KernelParamWrapper returns correct pointer."""
        desc = torch.empty(128, device="cpu", dtype=torch.int8)
        wrapper = TmaAutoTuneHelper.KernelParamWrapper(desc)
        self.assertEqual(wrapper.tma_desc_cpu_ptr(), desc.data_ptr())

    def test_kernel_param_wrapper_stores_desc(self) -> None:
        """Test KernelParamWrapper stores the descriptor reference."""
        desc = torch.empty(128, device="cpu", dtype=torch.int8)
        wrapper = TmaAutoTuneHelper.KernelParamWrapper(desc)
        self.assertIs(wrapper.desc, desc)

    def test_kernel_param_wrapper_alignment(self) -> None:
        """Test KernelParamWrapper data_ptr is accessible."""
        desc = torch.empty(128, device="cpu", dtype=torch.int8)
        wrapper = TmaAutoTuneHelper.KernelParamWrapper(desc)
        ptr = wrapper.tma_desc_cpu_ptr()
        self.assertIsInstance(ptr, int)
        self.assertGreater(ptr, 0)


if __name__ == "__main__":
    unittest.main()
