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
Unit tests for ads_mkl/ops/oss/gdpa/triton/hardware.py
"""

import unittest
from unittest.mock import MagicMock, patch

from ads_mkl.ops.oss.gdpa.triton.hardware import (
    _cap,
    _cap_list,
    _max_block_k_amd,
    _max_block_m_amd,
    _max_block_n_amd,
    _max_group_m_amd,
    _max_stages_amd,
    _max_warps_amd,
    block_dot_hw_supported,
    block_k_hw_supported,
    block_m_hw_supported,
    block_n_hw_supported,
    group_m_hw_supported,
    is_amd,
    is_mtia,
    stages_hw_supported,
    warps_hw_supported,
)


class IsAmdTestCase(unittest.TestCase):
    """Test is_amd function."""

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.torch.version")
    def test_returns_true_when_hip_is_set(self, mock_version: MagicMock) -> None:
        mock_version.hip = "5.7"
        self.assertTrue(is_amd())

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.torch.version")
    def test_returns_false_when_hip_is_none(self, mock_version: MagicMock) -> None:
        mock_version.hip = None
        self.assertFalse(is_amd())


class IsMtiaTestCase(unittest.TestCase):
    """Test is_mtia function."""

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.driver")
    def test_returns_true_when_backend_is_mtia(self, mock_driver: MagicMock) -> None:
        mock_driver.active.get_current_target.return_value.backend = "mtia"
        self.assertTrue(is_mtia())

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.driver")
    def test_returns_false_when_backend_is_not_mtia(
        self, mock_driver: MagicMock
    ) -> None:
        mock_driver.active.get_current_target.return_value.backend = "cuda"
        self.assertFalse(is_mtia())

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.driver")
    def test_returns_false_on_runtime_error(self, mock_driver: MagicMock) -> None:
        mock_driver.active.get_current_target.side_effect = RuntimeError("No device")
        self.assertFalse(is_mtia())

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.driver")
    def test_returns_false_on_system_error(self, mock_driver: MagicMock) -> None:
        mock_driver.active.get_current_target.side_effect = SystemError("System error")
        self.assertFalse(is_mtia())


class AmdConstantsTestCase(unittest.TestCase):
    """Test AMD hardware constant functions."""

    def test_max_block_m_amd(self) -> None:
        self.assertEqual(_max_block_m_amd(), 64)

    def test_max_block_n_amd(self) -> None:
        self.assertEqual(_max_block_n_amd(), 64)

    def test_max_block_k_amd(self) -> None:
        self.assertEqual(_max_block_k_amd(), 32)

    def test_max_group_m_amd(self) -> None:
        self.assertEqual(_max_group_m_amd(), 8)

    def test_max_stages_amd(self) -> None:
        self.assertEqual(_max_stages_amd(), 1)

    def test_max_warps_amd(self) -> None:
        self.assertEqual(_max_warps_amd(), 4)


class CapListTestCase(unittest.TestCase):
    """Test _cap_list function."""

    def test_caps_values_above_limit(self) -> None:
        result = _cap_list([16, 32, 64, 128], 64)
        self.assertEqual(result, [16, 32, 64])

    def test_empty_result_returns_cap_value(self) -> None:
        result = _cap_list([128, 256], 64)
        self.assertEqual(result, [64])

    def test_filters_not_allowed_values(self) -> None:
        result = _cap_list([1, 2, 4, 8], 8, not_allowed_values={2, 4})
        self.assertEqual(result, [1, 8])

    def test_all_values_below_cap(self) -> None:
        result = _cap_list([1, 2, 4], 8)
        self.assertEqual(result, [1, 2, 4])

    def test_empty_input_returns_cap_value(self) -> None:
        result = _cap_list([], 64)
        self.assertEqual(result, [64])

    def test_all_filtered_returns_cap_value(self) -> None:
        result = _cap_list([1, 2, 3], 8, not_allowed_values={1, 2, 3})
        self.assertEqual(result, [8])


class CapTestCase(unittest.TestCase):
    """Test _cap function."""

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.is_amd")
    def test_returns_unchanged_when_not_amd(self, mock_is_amd: MagicMock) -> None:
        mock_is_amd.return_value = False
        self.assertEqual(_cap([128, 256], 64), [128, 256])
        self.assertEqual(_cap((128, 256), 64), (128, 256))
        self.assertEqual(_cap(128, 64), 128)

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.is_amd")
    def test_caps_list_when_amd(self, mock_is_amd: MagicMock) -> None:
        mock_is_amd.return_value = True
        result = _cap([16, 32, 64, 128], 64)
        self.assertEqual(result, [16, 32, 64])

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.is_amd")
    def test_caps_tuple_when_amd(self, mock_is_amd: MagicMock) -> None:
        mock_is_amd.return_value = True
        result = _cap((16, 32, 64, 128), 64)
        self.assertEqual(result, (16, 32, 64))

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.is_amd")
    def test_caps_int_when_amd(self, mock_is_amd: MagicMock) -> None:
        mock_is_amd.return_value = True
        result = _cap(128, 64)
        self.assertEqual(result, 64)

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.is_amd")
    def test_int_not_allowed_value_returns_cap(self, mock_is_amd: MagicMock) -> None:
        mock_is_amd.return_value = True
        result = _cap(32, 64, not_allowed_values={32})
        self.assertEqual(result, 64)

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.is_amd")
    def test_int_allowed_value_returns_min(self, mock_is_amd: MagicMock) -> None:
        mock_is_amd.return_value = True
        result = _cap(32, 64, not_allowed_values={16})
        self.assertEqual(result, 32)


class HwSupportedFunctionsTestCase(unittest.TestCase):
    """Test hardware supported wrapper functions."""

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.is_amd")
    def test_block_m_hw_supported_nvidia(self, mock_is_amd: MagicMock) -> None:
        mock_is_amd.return_value = False
        self.assertEqual(block_m_hw_supported(128), 128)

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.is_amd")
    def test_block_m_hw_supported_amd(self, mock_is_amd: MagicMock) -> None:
        mock_is_amd.return_value = True
        self.assertEqual(block_m_hw_supported(128), 64)

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.is_amd")
    def test_block_n_hw_supported_nvidia(self, mock_is_amd: MagicMock) -> None:
        mock_is_amd.return_value = False
        self.assertEqual(block_n_hw_supported(128), 128)

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.is_amd")
    def test_block_n_hw_supported_amd(self, mock_is_amd: MagicMock) -> None:
        mock_is_amd.return_value = True
        self.assertEqual(block_n_hw_supported(128), 64)

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.is_amd")
    def test_block_k_hw_supported_nvidia(self, mock_is_amd: MagicMock) -> None:
        mock_is_amd.return_value = False
        self.assertEqual(block_k_hw_supported(64), 64)

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.is_amd")
    def test_block_k_hw_supported_amd(self, mock_is_amd: MagicMock) -> None:
        mock_is_amd.return_value = True
        self.assertEqual(block_k_hw_supported(64), 32)

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.is_amd")
    def test_group_m_hw_supported_nvidia(self, mock_is_amd: MagicMock) -> None:
        mock_is_amd.return_value = False
        self.assertEqual(group_m_hw_supported(32), 32)

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.is_amd")
    def test_group_m_hw_supported_amd(self, mock_is_amd: MagicMock) -> None:
        mock_is_amd.return_value = True
        self.assertEqual(group_m_hw_supported(32), 8)

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.is_amd")
    def test_stages_hw_supported_nvidia(self, mock_is_amd: MagicMock) -> None:
        mock_is_amd.return_value = False
        self.assertEqual(stages_hw_supported(4), 4)

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.is_amd")
    def test_stages_hw_supported_amd(self, mock_is_amd: MagicMock) -> None:
        mock_is_amd.return_value = True
        self.assertEqual(stages_hw_supported(4), 1)

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.is_amd")
    def test_warps_hw_supported_nvidia(self, mock_is_amd: MagicMock) -> None:
        mock_is_amd.return_value = False
        self.assertEqual(warps_hw_supported(8), 8)

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.is_amd")
    def test_warps_hw_supported_amd(self, mock_is_amd: MagicMock) -> None:
        mock_is_amd.return_value = True
        self.assertEqual(warps_hw_supported(8), 4)

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.is_mtia")
    def test_block_dot_hw_supported_not_mtia(self, mock_is_mtia: MagicMock) -> None:
        mock_is_mtia.return_value = False
        self.assertEqual(block_dot_hw_supported(32), 32)

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.is_mtia")
    def test_block_dot_hw_supported_mtia_small(self, mock_is_mtia: MagicMock) -> None:
        mock_is_mtia.return_value = True
        self.assertEqual(block_dot_hw_supported(32), 64)

    @patch("ads_mkl.ops.oss.gdpa.triton.hardware.is_mtia")
    def test_block_dot_hw_supported_mtia_large(self, mock_is_mtia: MagicMock) -> None:
        mock_is_mtia.return_value = True
        self.assertEqual(block_dot_hw_supported(128), 128)


if __name__ == "__main__":
    unittest.main()
