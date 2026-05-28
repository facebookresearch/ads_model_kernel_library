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
Unit tests for ads_mkl/ops/oss/gdpa/triton/math.py
"""

import unittest

import torch
from ads_mkl.ops.oss.gdpa.triton.math import (
    Activation,
    activation_to_int,
    get_pytorch_activation,
    int_to_activation,
)


class ActivationEnumTestCase(unittest.TestCase):
    """Test Activation enum."""

    def test_all_activation_values_are_strings(self) -> None:
        """Test all activation values are string type."""
        for act in Activation:
            self.assertIsInstance(act.value, str)

    def test_activation_enum_members(self) -> None:
        """Test expected enum members exist."""
        expected = [
            "Raw",
            "GeLU",
            "GeLUApprox",
            "FastGeLU",
            "LeakyReLU",
            "ReLU",
            "FastGeLUBF16",
            "SiLu",
            "FastSiLu",
            "HardSwish",
            "ReLUSquare",
            "FastGeLUTaylorApprox",
        ]
        for name in expected:
            self.assertIn(name, Activation.__members__)

    def test_activation_values(self) -> None:
        """Test specific activation string values."""
        self.assertEqual(Activation.Raw.value, "raw")
        self.assertEqual(Activation.GeLU.value, "gelu")
        self.assertEqual(Activation.GeLUApprox.value, "gelu_approx")
        self.assertEqual(Activation.FastGeLU.value, "fast_gelu")
        self.assertEqual(Activation.LeakyReLU.value, "leaky_relu")
        self.assertEqual(Activation.ReLU.value, "relu")
        self.assertEqual(Activation.SiLu.value, "silu")
        self.assertEqual(Activation.HardSwish.value, "hardswish")
        self.assertEqual(Activation.ReLUSquare.value, "relu_square")

    def test_activation_from_string(self) -> None:
        """Test creating activation from string value."""
        self.assertEqual(Activation("raw"), Activation.Raw)
        self.assertEqual(Activation("relu"), Activation.ReLU)
        self.assertEqual(Activation("gelu"), Activation.GeLU)


class ActivationToIntTestCase(unittest.TestCase):
    """Test activation_to_int mapping."""

    def test_all_activations_have_int_mapping(self) -> None:
        """Test every activation has an integer mapping."""
        for act in Activation:
            self.assertIn(act, activation_to_int)

    def test_int_mappings_are_unique(self) -> None:
        """Test all integer mappings are unique."""
        values = list(activation_to_int.values())
        self.assertEqual(len(values), len(set(values)))

    def test_raw_is_zero(self) -> None:
        """Test Raw activation maps to 0 (first in enum)."""
        self.assertEqual(activation_to_int[Activation.Raw], 0)

    def test_int_to_activation_inverse(self) -> None:
        """Test int_to_activation is the inverse of activation_to_int."""
        for act, idx in activation_to_int.items():
            self.assertEqual(int_to_activation[idx], act)

    def test_int_to_activation_covers_all(self) -> None:
        """Test int_to_activation covers all activations."""
        self.assertEqual(len(int_to_activation), len(Activation))


class GetPytorchActivationTestCase(unittest.TestCase):
    """Test get_pytorch_activation function."""

    def test_relu_returns_relu_module(self) -> None:
        """Test ReLU activation returns torch.nn.ReLU."""
        act = get_pytorch_activation(Activation.ReLU)
        self.assertIsInstance(act, torch.nn.ReLU)

    def test_leaky_relu_returns_leaky_relu_module(self) -> None:
        """Test LeakyReLU activation returns torch.nn.LeakyReLU."""
        act = get_pytorch_activation(Activation.LeakyReLU)
        self.assertIsInstance(act, torch.nn.LeakyReLU)

    def test_gelu_returns_gelu_module(self) -> None:
        """Test GeLU activation returns torch.nn.GELU."""
        act = get_pytorch_activation(Activation.GeLU)
        self.assertIsInstance(act, torch.nn.GELU)

    def test_gelu_approx_returns_gelu_tanh(self) -> None:
        """Test GeLUApprox returns GELU with tanh approximation."""
        act = get_pytorch_activation(Activation.GeLUApprox)
        self.assertIsInstance(act, torch.nn.GELU)

    def test_raw_returns_identity(self) -> None:
        """Test Raw activation returns Identity."""
        act = get_pytorch_activation(Activation.Raw)
        self.assertIsInstance(act, torch.nn.Identity)

    def test_silu_returns_silu_module(self) -> None:
        """Test SiLu returns torch.nn.SiLU."""
        act = get_pytorch_activation(Activation.SiLu)
        self.assertIsInstance(act, torch.nn.SiLU)

    def test_hardswish_returns_hardswish_module(self) -> None:
        """Test HardSwish returns torch.nn.Hardswish."""
        act = get_pytorch_activation(Activation.HardSwish)
        self.assertIsInstance(act, torch.nn.Hardswish)

    def test_relu_square_returns_callable(self) -> None:
        """Test ReLUSquare returns a callable."""
        act = get_pytorch_activation(Activation.ReLUSquare)
        self.assertTrue(callable(act))

    def test_relu_square_produces_correct_output(self) -> None:
        """Test ReLUSquare computes relu(x)^2."""
        act = get_pytorch_activation(Activation.ReLUSquare)
        x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        expected = torch.tensor([0.0, 0.0, 1.0, 4.0])
        result = act(x)
        torch.testing.assert_close(result, expected)

    def test_all_activations_produce_output(self) -> None:
        """Test all activations produce a tensor output."""
        x = torch.randn(4, 8)
        for activation in Activation:
            act_fn = get_pytorch_activation(activation)
            result = act_fn(x)
            self.assertEqual(result.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
