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
Unit tests for ads_mkl/ops/oss/gdpa/triton/register_helpers.py
"""

import unittest
from unittest.mock import MagicMock, patch

from ads_mkl.ops.oss.gdpa.triton.register_helpers import (
    custom_register_kernel,
    custom_triton_op,
)


class CustomTritonOpTestCase(unittest.TestCase):
    """Test custom_triton_op decorator."""

    @patch(
        "ads_mkl.ops.oss.gdpa.triton.register_helpers.torch._C._dispatch_has_kernel_for_dispatch_key"
    )
    @patch("ads_mkl.ops.oss.gdpa.triton.register_helpers.torch._library.triton_op")
    def test_registers_new_op_when_not_exists(
        self, mock_triton_op: MagicMock, mock_has_kernel: MagicMock
    ) -> None:
        """Test registers a new op when it doesn't exist."""
        mock_has_kernel.return_value = False
        mock_triton_op.return_value = lambda x: x

        @custom_triton_op("test::my_op", mutates_args=())
        def my_op():
            pass

        mock_triton_op.assert_called_once()

    @patch(
        "ads_mkl.ops.oss.gdpa.triton.register_helpers.torch._C._dispatch_has_kernel_for_dispatch_key"
    )
    def test_returns_func_unchanged_when_op_exists(
        self, mock_has_kernel: MagicMock
    ) -> None:
        """Test returns function unchanged when op already exists."""
        mock_has_kernel.return_value = True

        def my_op():
            return 42

        result = custom_triton_op("test::my_op", mutates_args=())(my_op)
        self.assertIs(result, my_op)

    @patch(
        "ads_mkl.ops.oss.gdpa.triton.register_helpers.torch._C._dispatch_has_kernel_for_dispatch_key"
    )
    @patch("ads_mkl.ops.oss.gdpa.triton.register_helpers.torch._library.triton_op")
    def test_registers_on_exception(
        self, mock_triton_op: MagicMock, mock_has_kernel: MagicMock
    ) -> None:
        """Test registers op when dispatch check raises exception."""
        mock_has_kernel.side_effect = Exception("dispatch error")
        mock_triton_op.return_value = lambda x: x

        @custom_triton_op("test::my_op2", mutates_args=())
        def my_op():
            pass

        mock_triton_op.assert_called_once()


class CustomRegisterKernelTestCase(unittest.TestCase):
    """Test custom_register_kernel decorator."""

    @patch(
        "ads_mkl.ops.oss.gdpa.triton.register_helpers.torch._C._dispatch_has_kernel_for_dispatch_key"
    )
    @patch("ads_mkl.ops.oss.gdpa.triton.register_helpers.torch.library.register_kernel")
    def test_registers_new_kernel_when_not_exists(
        self, mock_register: MagicMock, mock_has_kernel: MagicMock
    ) -> None:
        """Test registers a new kernel when it doesn't exist."""
        mock_has_kernel.return_value = False
        mock_register.return_value = lambda x: x

        @custom_register_kernel("test::my_op", "cpu")
        def my_kernel():
            pass

        mock_register.assert_called_once()

    @patch(
        "ads_mkl.ops.oss.gdpa.triton.register_helpers.torch._C._dispatch_has_kernel_for_dispatch_key"
    )
    def test_returns_func_unchanged_when_kernel_exists(
        self, mock_has_kernel: MagicMock
    ) -> None:
        """Test returns function unchanged when kernel already exists."""
        mock_has_kernel.return_value = True

        def my_kernel():
            return 42

        result = custom_register_kernel("test::my_op", "cpu")(my_kernel)
        self.assertIs(result, my_kernel)

    @patch(
        "ads_mkl.ops.oss.gdpa.triton.register_helpers.torch._C._dispatch_has_kernel_for_dispatch_key"
    )
    @patch("ads_mkl.ops.oss.gdpa.triton.register_helpers.torch.library.register_kernel")
    def test_registers_on_exception(
        self, mock_register: MagicMock, mock_has_kernel: MagicMock
    ) -> None:
        """Test registers kernel when dispatch check raises exception."""
        mock_has_kernel.side_effect = Exception("dispatch error")
        mock_register.return_value = lambda x: x

        @custom_register_kernel("test::my_op2", "cpu")
        def my_kernel():
            pass

        mock_register.assert_called_once()


if __name__ == "__main__":
    unittest.main()
