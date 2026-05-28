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
Unit tests for ads_mkl/ops/oss/gdpa/triton/vararg_kernel.py
"""

import ast
import unittest

from ads_mkl.ops.oss.gdpa.triton.vararg_kernel import (
    _VisitorConditionalKernel,
    _VisitorUnrollKernel,
    _VisitorVarargKernel,
    VarargModes,
)


class VarargModesTestCase(unittest.TestCase):
    """Test VarargModes enum."""

    def test_unroll_mode_value(self) -> None:
        """Test UNROLL mode has correct value."""
        self.assertEqual(VarargModes.UNROLL.value, "unroll")

    def test_conditional_mode_value(self) -> None:
        """Test CONDITIONAL mode has correct value."""
        self.assertEqual(VarargModes.CONDITIONAL.value, "conditional")

    def test_enum_has_two_members(self) -> None:
        """Test enum has exactly two members."""
        self.assertEqual(len(VarargModes), 2)


class VisitorVarargKernelTestCase(unittest.TestCase):
    """Test _VisitorVarargKernel AST node transformer."""

    def test_initialization(self) -> None:
        """Test visitor initializes with correct attributes."""
        visitor = _VisitorVarargKernel(N=3, unroll_as_const=False)
        self.assertEqual(visitor.N, 3)
        self.assertFalse(visitor.unroll_as_const)
        self.assertEqual(visitor.inline_variables, set())

    def test_visit_ann_assign_detects_var_args(self) -> None:
        """Test that annotated assignments with VAR_ARGS_ARRAY are detected."""
        visitor = _VisitorVarargKernel(N=3, unroll_as_const=False)
        # Create AST node: `my_var: "VAR_ARGS_ARRAY"`
        node = ast.AnnAssign(
            target=ast.Name(id="my_var"),
            annotation=ast.Constant(value="VAR_ARGS_ARRAY"),
            value=None,
            simple=1,
        )
        result = visitor.visit_AnnAssign(node)
        self.assertEqual(result, [])
        self.assertIn("my_var", visitor.inline_variables)

    def test_visit_ann_assign_ignores_non_varargs(self) -> None:
        """Test that non-VAR_ARGS_ARRAY annotations are preserved."""
        visitor = _VisitorVarargKernel(N=3, unroll_as_const=False)
        # Create AST node: `my_var: int = 5`
        node = ast.AnnAssign(
            target=ast.Name(id="my_var"),
            annotation=ast.Name(id="int"),
            value=ast.Constant(value=5),
            simple=1,
        )
        result = visitor.visit_AnnAssign(node)
        self.assertIsNotNone(result)
        self.assertNotIn("my_var", visitor.inline_variables)

    def test_visit_arguments_expands_varargs(self) -> None:
        """Test function arguments with VAR_ARGS_ARRAY are expanded."""
        visitor = _VisitorVarargKernel(N=3, unroll_as_const=False)
        # Create argument: `buffers: "VAR_ARGS_ARRAY"`
        args_node = ast.arguments(
            args=[
                ast.arg(arg="x"),
                ast.arg(
                    arg="buffers",
                    annotation=ast.Constant(value="VAR_ARGS_ARRAY"),
                ),
            ],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
            posonlyargs=[],
        )
        result = visitor.visit_arguments(args_node)
        # Should have x + buffers0, buffers1, buffers2 = 4 args
        self.assertEqual(len(result.args), 4)
        self.assertEqual(result.args[0].arg, "x")
        self.assertEqual(result.args[1].arg, "buffers0")
        self.assertEqual(result.args[2].arg, "buffers1")
        self.assertEqual(result.args[3].arg, "buffers2")
        self.assertIn("buffers", visitor.inline_variables)

    def test_visit_arguments_with_unroll_as_const(self) -> None:
        """Test arguments are annotated as tl.constexpr when unroll_as_const=True."""
        visitor = _VisitorVarargKernel(N=2, unroll_as_const=True)
        args_node = ast.arguments(
            args=[
                ast.arg(
                    arg="params",
                    annotation=ast.Constant(value="VAR_ARGS_ARRAY"),
                ),
            ],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
            posonlyargs=[],
        )
        result = visitor.visit_arguments(args_node)
        self.assertEqual(len(result.args), 2)
        # Check annotations exist (tl.constexpr)
        for arg in result.args:
            self.assertIsNotNone(arg.annotation)


class VisitorUnrollKernelTestCase(unittest.TestCase):
    """Test _VisitorUnrollKernel AST node transformer."""

    def test_unrolls_for_loop(self) -> None:
        """Test that for loops over vararg range are unrolled."""
        visitor = _VisitorUnrollKernel(N=3, unroll_as_const=False)
        visitor.inline_variables = {"buffers"}
        # Create: `for i in range(len(buffers)): x = buffers[i]`
        code = "for i in range(len(buffers)):\n    x = buffers[i]"
        tree = ast.parse(code)
        result = visitor.visit(tree)
        ast.fix_missing_locations(result)
        # The for loop should be unrolled into 3 assignments
        new_src = ast.unparse(result)
        self.assertIn("buffers0", new_src)
        self.assertIn("buffers1", new_src)
        self.assertIn("buffers2", new_src)
        self.assertNotIn("for i in range", new_src)


class VisitorConditionalKernelTestCase(unittest.TestCase):
    """Test _VisitorConditionalKernel AST node transformer."""

    def test_visit_subscript_creates_if_chain(self) -> None:
        """Test that array subscript is replaced with if-else chain."""
        visitor = _VisitorConditionalKernel(N=3, unroll_as_const=False)
        visitor.inline_variables = {"buffers"}
        # Create AST for: buffers[i]
        node = ast.Subscript(
            value=ast.Name(id="buffers"),
            slice=ast.Name(id="i"),
        )
        result = visitor.visit_Subscript(node)
        # Should be an IfExp chain: buffers0 if i == 0 else buffers1 if i == 1 else buffers2
        self.assertIsInstance(result, ast.IfExp)

    def test_visit_call_replaces_len(self) -> None:
        """Test that len(vararg) is replaced with constant N."""
        visitor = _VisitorConditionalKernel(N=5, unroll_as_const=False)
        visitor.inline_variables = {"buffers"}
        # Create AST for: len(buffers)
        node = ast.Call(
            func=ast.Name(id="len"),
            args=[ast.Name(id="buffers")],
            keywords=[],
        )
        result = visitor.visit_Call(node)
        self.assertIsInstance(result, ast.Constant)
        self.assertEqual(result.value, 5)

    def test_visit_call_preserves_non_vararg_len(self) -> None:
        """Test that len(non_vararg) is preserved."""
        visitor = _VisitorConditionalKernel(N=3, unroll_as_const=False)
        visitor.inline_variables = {"buffers"}
        # Create AST for: len(other_var)
        node = ast.Call(
            func=ast.Name(id="len"),
            args=[ast.Name(id="other_var")],
            keywords=[],
        )
        result = visitor.visit_Call(node)
        # Should remain a Call node, not replaced with constant
        self.assertIsInstance(result, ast.Call)


if __name__ == "__main__":
    unittest.main()
