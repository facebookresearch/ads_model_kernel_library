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

# Copyright (c) 2025, Tri Dao.
"""Fast math utilities for GPU kernels.

Provides count-leading-zeros (clz), fast integer divmod via multiply-high,
and log2 computation.
"""

# pyre-strict

import cutlass  # pyre-ignore[21]
import cutlass.cute as cute  # pyre-ignore[21]
from cutlass import Int32, Uint32  # pyre-ignore[21]
from cutlass._mlir.dialects import llvm  # pyre-ignore[21]
from cutlass.cutlass_dsl import dsl_user_op, T  # pyre-ignore[21]


@cute.jit  # pyre-ignore[56]
# pyre-ignore[11]
def clz(x: Int32) -> Int32:
    """Count leading zeros in a 32-bit integer.

    Returns the number of zero bits before the first set bit (32 if input is 0).
    """
    # for i in cutlass.range_constexpr(32):
    #     if (1 << (31 - i)) & x:
    #         return Int32(i)
    # return Int32(32)
    # Early exit is not supported yet
    res = Int32(32)
    done = False
    for i in cutlass.range(32):
        if ((1 << (31 - i)) & x) and not done:
            res = Int32(i)
            done = True
    return res


def find_log2(x: Int32) -> Int32:
    """Compute ceil(log2(x)).

    Returns the smallest n such that 2^n >= x.
    """
    a: Int32 = Int32(31 - clz(x))
    return a + ((x & (x - 1)) != 0)  # Round up, add 1 if not a power of 2.


@dsl_user_op  # pyre-ignore[56]
# pyre-ignore[11]
def umulhi(
    a: Int32,
    b: Int32,
    *,
    loc: object | None = None,
    ip: object | None = None,
) -> Uint32:  # pyre-ignore[11]
    """Unsigned multiply-high: returns the upper 32 bits of the 64-bit product a*b.

    Uses PTX mul.hi.u32 instruction.
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "mul.hi.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class FastDivmod:
    """Fast integer division and modulo using multiply-high.

    Precomputes a multiplier and shift on the host to replace expensive division
    with a multiply+shift on the GPU.
    """

    # pyre-ignore[11]
    divisor: Int32
    # pyre-ignore[11]
    multiplier: Uint32
    # pyre-ignore[11]
    shift_right: Uint32
    _loc: object | None
    _values_pos: list[int]  # pyre-ignore[13]

    def __init__(
        self,
        divisor: Int32,
        multipler: Uint32,
        shift_right: Uint32,
        *,
        loc: object | None = None,
        ip: object | None = None,
    ) -> None:
        self.divisor = divisor  # pyre-ignore[4]
        self.multiplier = multipler  # pyre-ignore[4]
        self.shift_right = shift_right  # pyre-ignore[4]
        self._loc = loc

    # called by host
    @staticmethod
    def create(
        divisor: Int32,
        *,
        loc: object | None = None,
        ip: object | None = None,
    ) -> "FastDivmod":
        """Construct the FastDivmod object, in host code.
        This precomputes some values based on the divisor and is computationally expensive.
        """
        p = Uint32(31 + find_log2(divisor))
        divisor_u32 = Uint32(divisor)
        multiplier = Uint32(((cutlass.Uint64(1) << p) + divisor_u32 - 1) // divisor_u32)
        shift_right = Uint32(p - 32)
        return FastDivmod(divisor, multiplier, shift_right, loc=loc, ip=ip)

    @cute.jit  # pyre-ignore[56]
    def div(self, dividend: Int32) -> Int32:
        return (
            Int32(umulhi(dividend, self.multiplier) >> self.shift_right)
            if self.divisor != 1
            else dividend
        )

    # pyre-ignore[11]
    def divmod(self, dividend: Int32) -> tuple[Int32, Int32]:
        quotient = self.div(dividend)
        remainder = dividend - quotient * self.divisor
        return quotient, remainder

    # pyre-ignore[3]
    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.divisor, self.multiplier, self.shift_right]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    # pyre-ignore[2, 3]
    def __new_from_mlir_values__(self, values) -> "FastDivmod":
        obj_list = []
        for obj, n_items in zip(
            [self.divisor, self.multiplier, self.shift_right], self._values_pos
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return FastDivmod(*(tuple(obj_list)), loc=self._loc)
