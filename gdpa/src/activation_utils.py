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

# pyre-strict

import cutlass  # pyre-ignore[21]
import cutlass.cute as cute  # pyre-ignore[21]
from ads_mkl.ops.cute_dsl.gdpa.src.utils import mul_packed_f32x2, sub_packed_f32x2
from cutlass import Float32, Int64, Uint32, Uint8  # pyre-ignore[21]
from cutlass._mlir.dialects import llvm, vector  # pyre-ignore[21]
from cutlass.cutlass_dsl import dsl_user_op, T  # pyre-ignore[21]

# MXFP8 block scaling constants
E4M3_MAX_NORM_RCP: float = 1.0 / 448.0
E8M0_NEUTRAL_SCALE: int = 127


# pyre-ignore[56]
@dsl_user_op
def pack_4xu8_to_u32(
    # pyre-ignore[11]
    b0: Uint8,
    b1: Uint8,
    b2: Uint8,
    b3: Uint8,
    *,
    loc: object | None = None,
    ip: object | None = None,
    # pyre-ignore[11]
) -> Uint32:
    """Pack 4 Uint8 values into a Uint32 (little-endian: b0 is lowest byte)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint8(b0).ir_value(loc=loc, ip=ip),
                Uint8(b1).ir_value(loc=loc, ip=ip),
                Uint8(b2).ir_value(loc=loc, ip=ip),
                Uint8(b3).ir_value(loc=loc, ip=ip),
            ],
            "{\n"
            ".reg .b32 tmp0, tmp1, tmp2, tmp3;\n"
            "cvt.u32.u8 tmp0, $1;\n"
            "cvt.u32.u8 tmp1, $2;\n"
            "cvt.u32.u8 tmp2, $3;\n"
            "cvt.u32.u8 tmp3, $4;\n"
            "shl.b32 tmp1, tmp1, 8;\n"
            "shl.b32 tmp2, tmp2, 16;\n"
            "shl.b32 tmp3, tmp3, 24;\n"
            "or.b32 $0, tmp0, tmp1;\n"
            "or.b32 $0, $0, tmp2;\n"
            "or.b32 $0, $0, tmp3;\n"
            "}\n",
            "=r,c,c,c,c",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# pyre-ignore[56]
@dsl_user_op
def store_u32_shared(
    # pyre-ignore[11]
    ptr: Int64,
    val: Uint32,
    *,
    loc: object | None = None,
    ip: object | None = None,
) -> None:
    """Store a Uint32 to shared memory."""
    llvm.inline_asm(
        None,
        [
            Int64(ptr).ir_value(loc=loc, ip=ip),
            Uint32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# pyre-ignore[56]
@dsl_user_op
def tanh(
    # pyre-ignore[11]
    a: float | Float32,
    *,
    loc: object | None = None,
    ip: object | None = None,
) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# pyre-ignore[56]
@dsl_user_op
def max_f32(
    a: float | Float32,
    b: float | Float32,
    *,
    loc: object | None = None,
    ip: object | None = None,
) -> Float32:
    """Compute max(a, b) using PTX max.f32 instruction."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# pyre-ignore[56]
@dsl_user_op
def step_f32(
    x: float | Float32, *, loc: object | None = None, ip: object | None = None
) -> Float32:
    """Step function: returns 1.0 if x >= 0, else 0.0 (used for ReLU gradient)."""
    # Uses setp.ge.f32 to set predicate, then selp to select 1.0 or 0.0
    # 0f3F800000 is 1.0f in hex, 0f00000000 is 0.0f in hex
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "{ .reg .pred p; setp.ge.f32 p, $1, 0f00000000; selp.f32 $0, 0f3F800000, 0f00000000, p; }",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# pyre-ignore[56]
@dsl_user_op
def abs_f32(
    val: Float32, *, loc: object | None = None, ip: object | None = None
) -> Float32:
    """Compute |val| using PTX abs.f32 instruction."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(val).ir_value(loc=loc, ip=ip)],
            "abs.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# pyre-ignore[56]
@dsl_user_op
def max3_f32(
    a: float | Float32,
    b: float | Float32,
    c: float | Float32,
    *,
    loc: object | None = None,
    ip: object | None = None,
) -> Float32:
    """Compute max(a, max(b, c)) - materializes as FMNMX3 on Blackwell.

    This pattern is recognized by the Blackwell compiler and materialized
    as a single 3-input max instruction (FMNMX3), reducing instruction count.
    """
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Float32(c).ir_value(loc=loc, ip=ip),
            ],
            "{\n.reg .f32 tmp;\nmax.f32 tmp, $2, $3;\nmax.f32 $0, $1, tmp;\n}\n",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# pyre-ignore[56]
@dsl_user_op
def mul_cvt_relu_8x_e4m3(
    in_0: Float32,
    in_1: Float32,
    in_2: Float32,
    in_3: Float32,
    in_4: Float32,
    in_5: Float32,
    in_6: Float32,
    in_7: Float32,
    inv_scale: Float32,
    *,
    loc: object | None = None,
    ip: object | None = None,
) -> Int64:
    """
    Fused multiply, ReLU, and convert 8x f32 values to e4m3x8.

    Uses cvt.rn.satfinite.relu.e4m3x2.f32 to fuse ReLU into the FP8 conversion.
    Key insight: relu(x)*inv_scale == relu(x*inv_scale) when inv_scale > 0.

    This matches the C++ mul_cvt_relu_8x_e4m3 function from gdpa_common.hpp.

    Args:
        in_0 to in_7: Eight input float32 values
        inv_scale: Inverse scale factor (applied to all values before ReLU+convert)

    Returns:
        Packed e4m3x8 (eight FP8 E4M3 values in int64)
        Layout: [in_0, in_1, ..., in_7] -> [byte0, byte1, ..., byte7]
    """
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [
                Float32(in_0).ir_value(loc=loc, ip=ip),
                Float32(in_1).ir_value(loc=loc, ip=ip),
                Float32(in_2).ir_value(loc=loc, ip=ip),
                Float32(in_3).ir_value(loc=loc, ip=ip),
                Float32(in_4).ir_value(loc=loc, ip=ip),
                Float32(in_5).ir_value(loc=loc, ip=ip),
                Float32(in_6).ir_value(loc=loc, ip=ip),
                Float32(in_7).ir_value(loc=loc, ip=ip),
                Float32(inv_scale).ir_value(loc=loc, ip=ip),
            ],
            "{\n"
            ".reg .f32 s0, s1, s2, s3, s4, s5, s6, s7;\n"
            ".reg .b16 fp8_01, fp8_23, fp8_45, fp8_67;\n"
            ".reg .b32 lo32, hi32;\n"
            # Scale all inputs
            "mul.rn.f32 s0, $1, $9;\n"
            "mul.rn.f32 s1, $2, $9;\n"
            "mul.rn.f32 s2, $3, $9;\n"
            "mul.rn.f32 s3, $4, $9;\n"
            "mul.rn.f32 s4, $5, $9;\n"
            "mul.rn.f32 s5, $6, $9;\n"
            "mul.rn.f32 s6, $7, $9;\n"
            "mul.rn.f32 s7, $8, $9;\n"
            # Fused ReLU + FP8 conversion using .relu modifier
            "cvt.rn.satfinite.relu.e4m3x2.f32 fp8_01, s1, s0;\n"
            "cvt.rn.satfinite.relu.e4m3x2.f32 fp8_23, s3, s2;\n"
            "cvt.rn.satfinite.relu.e4m3x2.f32 fp8_45, s5, s4;\n"
            "cvt.rn.satfinite.relu.e4m3x2.f32 fp8_67, s7, s6;\n"
            # Pack into uint64
            "mov.b32 lo32, {fp8_01, fp8_23};\n"
            "mov.b32 hi32, {fp8_45, fp8_67};\n"
            "mov.b64 $0, {lo32, hi32};\n"
            "}\n",
            "=l,f,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# pyre-ignore[56]
@dsl_user_op
def unpack_i64_to_u32_pair(
    packed: Int64, *, loc: object | None = None, ip: object | None = None
) -> tuple[Uint32, Uint32]:
    """Unpack int64 to two uint32 values (low and high halves).

    Avoids PRMT by directly accessing uint32 words using mov.b64.
    This matches the C++ pattern of casting uint64_t* to uint32_t*.
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int64(packed).ir_value(loc=loc, ip=ip)],
        "mov.b64 {$0, $1}, $2;\n",
        "=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    lo = Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip))
    hi = Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip))
    return lo, hi


# pyre-ignore[56]
@dsl_user_op
def cvt_relu_8x_e4m3(
    s0: Float32,
    s1: Float32,
    s2: Float32,
    s3: Float32,
    s4: Float32,
    s5: Float32,
    s6: Float32,
    s7: Float32,
    *,
    loc: object | None = None,
    ip: object | None = None,
) -> Int64:
    """
    Fused ReLU and convert 8x pre-scaled f32 values to e4m3x8.

    Uses cvt.rn.satfinite.relu.e4m3x2.f32 to fuse ReLU into the FP8 conversion.
    This function expects values that have already been scaled by inv_scale
    using SIMD mul_packed_f32x2 operations.

    Args:
        s0 to s7: Eight pre-scaled float32 values (already multiplied by inv_scale)

    Returns:
        Packed e4m3x8 (eight FP8 E4M3 values in int64)
        Layout: [s0, s1, ..., s7] -> [byte0, byte1, ..., byte7]
    """
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [
                Float32(s0).ir_value(loc=loc, ip=ip),
                Float32(s1).ir_value(loc=loc, ip=ip),
                Float32(s2).ir_value(loc=loc, ip=ip),
                Float32(s3).ir_value(loc=loc, ip=ip),
                Float32(s4).ir_value(loc=loc, ip=ip),
                Float32(s5).ir_value(loc=loc, ip=ip),
                Float32(s6).ir_value(loc=loc, ip=ip),
                Float32(s7).ir_value(loc=loc, ip=ip),
            ],
            "{\n"
            ".reg .b16 fp8_01, fp8_23, fp8_45, fp8_67;\n"
            ".reg .b32 lo32, hi32;\n"
            # Fused ReLU + FP8 conversion using .relu modifier
            "cvt.rn.satfinite.relu.e4m3x2.f32 fp8_01, $2, $1;\n"
            "cvt.rn.satfinite.relu.e4m3x2.f32 fp8_23, $4, $3;\n"
            "cvt.rn.satfinite.relu.e4m3x2.f32 fp8_45, $6, $5;\n"
            "cvt.rn.satfinite.relu.e4m3x2.f32 fp8_67, $8, $7;\n"
            # Pack into uint64
            "mov.b32 lo32, {fp8_01, fp8_23};\n"
            "mov.b32 hi32, {fp8_45, fp8_67};\n"
            "mov.b64 $0, {lo32, hi32};\n"
            "}\n",
            "=l,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# pyre-ignore[56]
@dsl_user_op
def cvt_8x_e4m3(
    s0: Float32,
    s1: Float32,
    s2: Float32,
    s3: Float32,
    s4: Float32,
    s5: Float32,
    s6: Float32,
    s7: Float32,
    *,
    loc: object | None = None,
    ip: object | None = None,
) -> Int64:
    """
    Convert 8x pre-scaled f32 values to e4m3x8 (no activation fused).

    Uses cvt.rn.satfinite.e4m3x2.f32 for FP8 conversion.
    This function expects values that have already been scaled by inv_scale
    and had the activation (e.g., GELU) applied.

    Args:
        s0 to s7: Eight pre-scaled float32 values (already multiplied by inv_scale)

    Returns:
        Packed e4m3x8 (eight FP8 E4M3 values in int64)
        Layout: [s0, s1, ..., s7] -> [byte0, byte1, ..., byte7]
    """
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [
                Float32(s0).ir_value(loc=loc, ip=ip),
                Float32(s1).ir_value(loc=loc, ip=ip),
                Float32(s2).ir_value(loc=loc, ip=ip),
                Float32(s3).ir_value(loc=loc, ip=ip),
                Float32(s4).ir_value(loc=loc, ip=ip),
                Float32(s5).ir_value(loc=loc, ip=ip),
                Float32(s6).ir_value(loc=loc, ip=ip),
                Float32(s7).ir_value(loc=loc, ip=ip),
            ],
            "{\n"
            ".reg .b16 fp8_01, fp8_23, fp8_45, fp8_67;\n"
            ".reg .b32 lo32, hi32;\n"
            # FP8 conversion without .relu modifier
            "cvt.rn.satfinite.e4m3x2.f32 fp8_01, $2, $1;\n"
            "cvt.rn.satfinite.e4m3x2.f32 fp8_23, $4, $3;\n"
            "cvt.rn.satfinite.e4m3x2.f32 fp8_45, $6, $5;\n"
            "cvt.rn.satfinite.e4m3x2.f32 fp8_67, $8, $7;\n"
            # Pack into uint64
            "mov.b32 lo32, {fp8_01, fp8_23};\n"
            "mov.b32 hi32, {fp8_45, fp8_67};\n"
            "mov.b64 $0, {lo32, hi32};\n"
            "}\n",
            "=l,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# pyre-ignore[56]
@dsl_user_op
def fused_amax_to_e8m0_scale_f32(
    amax: Float32,
    max_norm_rcp: Float32,
    *,
    loc: object | None = None,
    ip: object | None = None,
) -> tuple[Uint8, Float32]:
    """
    Convert F32 AMAX to E8M0 scale and inverse scale.

    This matches the C++ fused_amax_to_e8m0_rceil function:
    1. scale_f32 = amax * max_norm_rcp
    2. Extract exponent, round up (RCEIL) if mantissa != 0
    3. Compute inverse scale = 2^(127 - e8m0_exp)

    Args:
        amax: Maximum absolute value of the block (F32)
        max_norm_rcp: Reciprocal of max normal value (1/448 for E4M3)

    Returns:
        Tuple of (e8m0_scale, inv_scale):
        - e8m0_scale: E8M0 biased exponent (uint8)
        - inv_scale: Inverse scale factor = 2^(127 - e8m0_exp) (float32)
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.f32()]),
        [
            Float32(amax).ir_value(loc=loc, ip=ip),
            Float32(max_norm_rcp).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        # Step 1: Multiply by max_norm_rcp
        ".reg .f32 fae_scaled;\n"
        "mul.f32 fae_scaled, $2, $3;\n"
        # Step 2: Extract exponent and mantissa, round up if mantissa != 0 (RCEIL)
        ".reg .b32 fae_bits, fae_exp, fae_mantissa;\n"
        ".reg .pred fae_has_mantissa;\n"
        "mov.b32 fae_bits, fae_scaled;\n"
        "bfe.u32 fae_exp, fae_bits, 23, 8;\n"
        "and.b32 fae_mantissa, fae_bits, 0x7FFFFF;\n"
        "setp.ne.u32 fae_has_mantissa, fae_mantissa, 0;\n"
        "@fae_has_mantissa add.u32 fae_exp, fae_exp, 1;\n"
        "mov.b32 $0, fae_exp;\n"
        # Step 3: Compute inverse scale = 2^(127 - e8m0_exp)
        # inv_scale = 2^(-(e8m0 - 127)) = 2^(127 - e8m0)
        # In IEEE754, this has exponent = 254 - e8m0
        ".reg .u32 fae_inv_exp, fae_inv_bits;\n"
        "sub.u32 fae_inv_exp, 254, fae_exp;\n"
        "shl.b32 fae_inv_bits, fae_inv_exp, 23;\n"
        "mov.b32 $1, fae_inv_bits;\n"
        "}\n",
        "=r,=f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    e8m0_u32 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    inv_scale_val = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)

    # Truncate e8m0 to uint8
    e8m0_scale = Uint8(
        llvm.inline_asm(
            T.i8(),
            [e8m0_u32],
            "cvt.u8.u32 $0, $1;\n",
            "=c,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )

    return e8m0_scale, Float32(inv_scale_val)


class Relu:
    """ReLU activation class for CuteDSL GDPA.

    ReLU: max(0, x)
    ReLU gradient: 1 if x >= 0, else 0
    """

    # pyre-ignore[11]
    def __init__(
        self,
        scale_qk: Float32,
    ) -> None:
        self.scale_qk: object = scale_qk
        self.c_zero: object = (Float32(0.0), Float32(0.0))
        self.c_one: object = (Float32(1.0), Float32(1.0))

    # pyre-ignore[56]
    @cute.jit
    def relu(self, x: tuple[Float32, Float32]) -> tuple[Float32, Float32]:
        """Apply ReLU activation: max(0, x) for packed f32x2."""
        return (max_f32(x[0], Float32(0.0)), max_f32(x[1], Float32(0.0)))

    # pyre-ignore[56]
    @cute.jit
    def grad_relu(self, x: tuple[Float32, Float32]) -> tuple[Float32, Float32]:
        """Compute ReLU gradient: 1 if x >= 0, else 0 for packed f32x2."""
        return (step_f32(x[0]), step_f32(x[1]))

    # pyre-ignore[56]
    @cute.jit
    def activation_and_gradient_relu(
        self, x: tuple[Float32, Float32]
    ) -> tuple[Float32, Float32, Float32, Float32]:
        """Compute both ReLU activation and gradient simultaneously.

        Returns (act_x0, act_x1, grad_x0, grad_x1).
        """
        act = self.relu(x)
        grad = self.grad_relu(x)
        # pyre-ignore[60]
        return *act, *grad

    # pyre-ignore[56]
    @cute.jit
    def relu_and_convert(
        self,
        # pyre-ignore[11]
        acc_S_row: cute.Tensor,
        acc_S_row_converted: cute.Tensor,
        # pyre-ignore[11]
        e2e_freq: cutlass.Constexpr[int] = 16,
        e2e_res: cutlass.Constexpr[int] = 4,
        e2e_frg_limit: cutlass.Constexpr[int] = 1,
    ) -> None:
        """Apply ReLU activation and convert to target dtype (for forward pass)."""
        assert cute.size(acc_S_row.shape) % 2 == 0, (
            "acc_S_row must have an even number of elements"
        )
        frg_tile = 32
        assert frg_tile % 2 == 0
        frg_cnt = cute.size(acc_S_row) // frg_tile
        assert cute.size(acc_S_row) % frg_tile == 0
        acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))
        acc_S_row_converted_frg = cute.logical_divide(
            acc_S_row_converted, cute.make_layout(frg_tile)
        )
        for j in cutlass.range_constexpr(frg_cnt):
            for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):
                s_f2 = (acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j])
                acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = self.relu(s_f2)
            acc_S_row_converted_frg[None, j].store(
                acc_S_row_frg[None, j].load().to(acc_S_row_converted.element_type)
            )

    # pyre-ignore[56]
    @cute.jit
    def grad_relu_and_convert(
        self,
        acc_S_row: cute.Tensor,
        acc_dS_row: cute.Tensor,
        acc_P_row: cute.Tensor,
        acc_P_row_f16: cute.Tensor,
        e2e_freq: cutlass.Constexpr[int] = 16,
        e2e_res: cutlass.Constexpr[int] = 4,
        e2e_frg_limit: cutlass.Constexpr[int] = 1,
    ) -> None:
        """Apply ReLU activation and compute gradient, convert to target dtype (for backward pass)."""
        assert cute.size(acc_S_row.shape) % 2 == 0, (
            "acc_S_row must have an even number of elements"
        )
        frg_tile = 32
        assert frg_tile % 2 == 0
        frg_cnt = cute.size(acc_S_row) // frg_tile
        assert cute.size(acc_S_row) % frg_tile == 0
        acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))
        acc_P_row_frg = cute.logical_divide(acc_P_row, cute.make_layout(frg_tile))
        acc_dS_row_frg = cute.logical_divide(acc_dS_row, cute.make_layout(frg_tile))
        acc_P_row_f16_frg = cute.logical_divide(
            acc_P_row_f16, cute.make_layout(frg_tile)
        )

        for j in cutlass.range_constexpr(frg_cnt):
            for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):
                s_f2 = (acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j])
                (
                    acc_P_row_frg[k, j],
                    acc_P_row_frg[k + 1, j],
                    acc_dS_row_frg[k, j],
                    acc_dS_row_frg[k + 1, j],
                ) = self.activation_and_gradient_relu(s_f2)
            acc_P_row_f16_frg[None, j].store(
                acc_P_row_frg[None, j].load().to(acc_P_row_f16_frg.element_type)
            )

    # pyre-ignore[56]
    @cute.jit
    def relu_and_convert_blockscaled(
        self,
        acc_S_row: cute.Tensor,
        acc_S_row_converted: cute.Tensor,
    ) -> tuple[Uint8, Uint8, Uint8, Uint8]:
        """Apply ReLU, compute per-32-block scales, and quantize for MXFP8.

        Optimized version matching C++ performance (D89909207):
        - Uses 3-input max pattern for AMAX (FMNMX3 on Blackwell)
        - Uses cvt.rn.satfinite.relu.e4m3x2.f32 for fused ReLU+FP8 conversion
        - No abs() needed for AMAX since ReLU zeroes negatives
        - Direct uint32 stores to avoid PRMT instructions

        Key insight: relu(x)*inv_scale == relu(x*inv_scale) when inv_scale > 0,
        so we compute AMAX on raw values, then fuse scale+relu+convert.

        Args:
            acc_S_row: Input tensor with 128 F32 values from TMEM (S accumulator)
            acc_S_row_converted: Output tensor for 128 FP8 values (P for TMEM)

        Returns:
            Tuple of 4 E8M0 scale factors (one per 32-element block)
        """
        BLOCK_SIZE = 32
        max_norm_rcp = Float32(E4M3_MAX_NORM_RCP)

        acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(BLOCK_SIZE))
        # Recast converted tensor to Uint32 for direct 4-byte stores
        acc_S_row_converted_u32_ptr = cute.recast_ptr(
            acc_S_row_converted.iterator, dtype=Uint32
        )
        acc_S_row_converted_u32 = cute.make_tensor(
            acc_S_row_converted_u32_ptr,
            cute.recast_layout(32, 8, acc_S_row_converted.layout),
        )
        acc_S_row_converted_u32_frg = cute.logical_divide(
            acc_S_row_converted_u32, cute.make_layout(BLOCK_SIZE // 4)
        )

        # ===== Process block 0 =====
        # Step 1: AMAX using 3-input max (no abs for ReLU - negatives zeroed)
        block_amax_0 = Float32(0.0)
        # Process 30 elements in groups of 3 for FMNMX3
        for k in cutlass.range_constexpr(0, 30, 3):
            block_amax_0 = max3_f32(
                block_amax_0,
                acc_S_row_frg[k, 0],
                max_f32(acc_S_row_frg[k + 1, 0], acc_S_row_frg[k + 2, 0]),
            )
        # Remainder (2 elements)
        block_amax_0 = max_f32(block_amax_0, acc_S_row_frg[30, 0])
        block_amax_0 = max_f32(block_amax_0, acc_S_row_frg[31, 0])

        # Step 2: Compute E8M0 scale
        scale0, inv_scale_0 = fused_amax_to_e8m0_scale_f32(block_amax_0, max_norm_rcp)
        inv_scale_0_pair = (inv_scale_0, inv_scale_0)

        # Step 3: SIMD scale (FMUL2) + fused ReLU+convert (8 elements at a time)
        for k in cutlass.range_constexpr(0, BLOCK_SIZE, 8):
            # SIMD float2 multiplication using mul_packed_f32x2 (generates FMUL2)
            s01 = mul_packed_f32x2(
                (acc_S_row_frg[k + 0, 0], acc_S_row_frg[k + 1, 0]), inv_scale_0_pair
            )
            s23 = mul_packed_f32x2(
                (acc_S_row_frg[k + 2, 0], acc_S_row_frg[k + 3, 0]), inv_scale_0_pair
            )
            s45 = mul_packed_f32x2(
                (acc_S_row_frg[k + 4, 0], acc_S_row_frg[k + 5, 0]), inv_scale_0_pair
            )
            s67 = mul_packed_f32x2(
                (acc_S_row_frg[k + 6, 0], acc_S_row_frg[k + 7, 0]), inv_scale_0_pair
            )
            # Fused ReLU + convert (cvt.rn.satfinite.relu.e4m3x2.f32)
            fp8_packed = cvt_relu_8x_e4m3(
                s01[0], s01[1], s23[0], s23[1], s45[0], s45[1], s67[0], s67[1]
            )
            # Direct uint32 store (avoids PRMT)
            lo32, hi32 = unpack_i64_to_u32_pair(fp8_packed)
            base_idx = k // 4
            acc_S_row_converted_u32_frg[base_idx, 0] = lo32
            acc_S_row_converted_u32_frg[base_idx + 1, 0] = hi32

        # ===== Process block 1 =====
        block_amax_1 = Float32(0.0)
        for k in cutlass.range_constexpr(0, 30, 3):
            block_amax_1 = max3_f32(
                block_amax_1,
                acc_S_row_frg[k, 1],
                max_f32(acc_S_row_frg[k + 1, 1], acc_S_row_frg[k + 2, 1]),
            )
        block_amax_1 = max_f32(block_amax_1, acc_S_row_frg[30, 1])
        block_amax_1 = max_f32(block_amax_1, acc_S_row_frg[31, 1])
        scale1, inv_scale_1 = fused_amax_to_e8m0_scale_f32(block_amax_1, max_norm_rcp)
        inv_scale_1_pair = (inv_scale_1, inv_scale_1)

        for k in cutlass.range_constexpr(0, BLOCK_SIZE, 8):
            s01 = mul_packed_f32x2(
                (acc_S_row_frg[k + 0, 1], acc_S_row_frg[k + 1, 1]), inv_scale_1_pair
            )
            s23 = mul_packed_f32x2(
                (acc_S_row_frg[k + 2, 1], acc_S_row_frg[k + 3, 1]), inv_scale_1_pair
            )
            s45 = mul_packed_f32x2(
                (acc_S_row_frg[k + 4, 1], acc_S_row_frg[k + 5, 1]), inv_scale_1_pair
            )
            s67 = mul_packed_f32x2(
                (acc_S_row_frg[k + 6, 1], acc_S_row_frg[k + 7, 1]), inv_scale_1_pair
            )
            fp8_packed = cvt_relu_8x_e4m3(
                s01[0], s01[1], s23[0], s23[1], s45[0], s45[1], s67[0], s67[1]
            )
            lo32, hi32 = unpack_i64_to_u32_pair(fp8_packed)
            base_idx = k // 4
            acc_S_row_converted_u32_frg[base_idx, 1] = lo32
            acc_S_row_converted_u32_frg[base_idx + 1, 1] = hi32

        # ===== Process block 2 =====
        block_amax_2 = Float32(0.0)
        for k in cutlass.range_constexpr(0, 30, 3):
            block_amax_2 = max3_f32(
                block_amax_2,
                acc_S_row_frg[k, 2],
                max_f32(acc_S_row_frg[k + 1, 2], acc_S_row_frg[k + 2, 2]),
            )
        block_amax_2 = max_f32(block_amax_2, acc_S_row_frg[30, 2])
        block_amax_2 = max_f32(block_amax_2, acc_S_row_frg[31, 2])
        scale2, inv_scale_2 = fused_amax_to_e8m0_scale_f32(block_amax_2, max_norm_rcp)
        inv_scale_2_pair = (inv_scale_2, inv_scale_2)

        for k in cutlass.range_constexpr(0, BLOCK_SIZE, 8):
            s01 = mul_packed_f32x2(
                (acc_S_row_frg[k + 0, 2], acc_S_row_frg[k + 1, 2]), inv_scale_2_pair
            )
            s23 = mul_packed_f32x2(
                (acc_S_row_frg[k + 2, 2], acc_S_row_frg[k + 3, 2]), inv_scale_2_pair
            )
            s45 = mul_packed_f32x2(
                (acc_S_row_frg[k + 4, 2], acc_S_row_frg[k + 5, 2]), inv_scale_2_pair
            )
            s67 = mul_packed_f32x2(
                (acc_S_row_frg[k + 6, 2], acc_S_row_frg[k + 7, 2]), inv_scale_2_pair
            )
            fp8_packed = cvt_relu_8x_e4m3(
                s01[0], s01[1], s23[0], s23[1], s45[0], s45[1], s67[0], s67[1]
            )
            lo32, hi32 = unpack_i64_to_u32_pair(fp8_packed)
            base_idx = k // 4
            acc_S_row_converted_u32_frg[base_idx, 2] = lo32
            acc_S_row_converted_u32_frg[base_idx + 1, 2] = hi32

        # ===== Process block 3 =====
        block_amax_3 = Float32(0.0)
        for k in cutlass.range_constexpr(0, 30, 3):
            block_amax_3 = max3_f32(
                block_amax_3,
                acc_S_row_frg[k, 3],
                max_f32(acc_S_row_frg[k + 1, 3], acc_S_row_frg[k + 2, 3]),
            )
        block_amax_3 = max_f32(block_amax_3, acc_S_row_frg[30, 3])
        block_amax_3 = max_f32(block_amax_3, acc_S_row_frg[31, 3])
        scale3, inv_scale_3 = fused_amax_to_e8m0_scale_f32(block_amax_3, max_norm_rcp)
        inv_scale_3_pair = (inv_scale_3, inv_scale_3)

        for k in cutlass.range_constexpr(0, BLOCK_SIZE, 8):
            s01 = mul_packed_f32x2(
                (acc_S_row_frg[k + 0, 3], acc_S_row_frg[k + 1, 3]), inv_scale_3_pair
            )
            s23 = mul_packed_f32x2(
                (acc_S_row_frg[k + 2, 3], acc_S_row_frg[k + 3, 3]), inv_scale_3_pair
            )
            s45 = mul_packed_f32x2(
                (acc_S_row_frg[k + 4, 3], acc_S_row_frg[k + 5, 3]), inv_scale_3_pair
            )
            s67 = mul_packed_f32x2(
                (acc_S_row_frg[k + 6, 3], acc_S_row_frg[k + 7, 3]), inv_scale_3_pair
            )
            fp8_packed = cvt_relu_8x_e4m3(
                s01[0], s01[1], s23[0], s23[1], s45[0], s45[1], s67[0], s67[1]
            )
            lo32, hi32 = unpack_i64_to_u32_pair(fp8_packed)
            base_idx = k // 4
            acc_S_row_converted_u32_frg[base_idx, 3] = lo32
            acc_S_row_converted_u32_frg[base_idx + 1, 3] = hi32

        return scale0, scale1, scale2, scale3


class Gelu:
    # pyre-ignore[11]
    def __init__(
        self,
        scale_qk: Float32,
    ) -> None:
        self.scale_qk: object = scale_qk
        self.c_half: object = (Float32(0.5), Float32(0.5))
        self.c_one: object = (Float32(1.0), Float32(1.0))
        self.c_two: object = (Float32(2.0), Float32(2.0))
        self.c_three: object = (Float32(3.0), Float32(3.0))
        self.c_four: object = (Float32(4.0), Float32(4.0))
        self.c_six: object = (Float32(6.0), Float32(6.0))
        self.c_alpha: object = (Float32(0.044715), Float32(0.044715))
        self.c_beta: object = (Float32(0.7978845608), Float32(0.7978845608))
        self.c_gamma: object = (Float32(0.1070322243), Float32(0.1070322243))
        self.c_sqrt_2: object = (
            Float32(0.7071067811865475),
            Float32(0.7071067811865475),
        )
        self.c_sqrt_2pi: object = (
            Float32(1.1283791670955126),
            Float32(1.1283791670955126),
        )
        self.c_minus_1_div_3: object = (
            Float32(-0.3333333333333333),
            Float32(-0.3333333333333333),
        )
        self.c_1_div_10: object = (Float32(0.1), Float32(0.1))
        self.c_taylor_c2: object = (
            Float32(0.3989422804014327),
            Float32(0.3989422804014327),
        )
        self.c_taylor_c4: object = (
            Float32(-0.06649038006690543),
            Float32(-0.06649038006690543),
        )
        self.c_taylor_c6: object = (
            Float32(0.00997355701003582),
            Float32(0.00997355701003582),
        )
        self.c_taylor_c8: object = (
            Float32(-0.001187328215480045),
            Float32(-0.001187328215480045),
        )
        self.c_taylor_c10: object = (
            Float32(0.0001171766272366646),
            Float32(0.0001171766272366646),
        )
        self.c_talyor_2_mul_c2: object = (
            # pyre-ignore[16]
            Float32(self.c_taylor_c2[0] * self.c_two[0]),
            # pyre-ignore[16]
            Float32(self.c_taylor_c2[1] * self.c_two[1]),
        )
        self.c_taylor_4_mul_c4: object = (
            # pyre-ignore[16]
            Float32(self.c_taylor_c4[0] * self.c_four[0]),
            # pyre-ignore[16]
            Float32(self.c_taylor_c4[1] * self.c_four[1]),
        )
        self.c_taylor_6_mul_c6: object = (
            # pyre-ignore[16]
            Float32(self.c_taylor_c6[0] * self.c_six[0]),
            # pyre-ignore[16]
            Float32(self.c_taylor_c6[1] * self.c_six[1]),
        )

    # pyre-ignore[56]
    @cute.jit
    def gelu_tanh(self, x: tuple[Float32, Float32]) -> tuple[Float32, Float32]:
        # x^2
        x2 = cute.arch.mul_packed_f32x2(x, x)

        # 1 + alpha * x^2
        inner = cute.arch.fma_packed_f32x2(self.c_alpha, x2, self.c_one)

        # beta * inner * x
        tanh_arg = cute.arch.mul_packed_f32x2(x, inner)
        tanh_arg_x, tanh_arg_y = cute.arch.mul_packed_f32x2(tanh_arg, self.c_beta)

        t_x = tanh(tanh_arg_x)
        t_y = tanh(tanh_arg_y)

        return (t_x, t_y)

    # pyre-ignore[56]
    @cute.jit
    def activation_and_gradient_fast_gelu(
        self, x: tuple[Float32, Float32]
    ) -> tuple[Float32, Float32, Float32, Float32]:
        tanh = self.gelu_tanh(x)
        tanh_1 = cute.arch.add_packed_f32x2(tanh, self.c_one)
        half_x = cute.arch.mul_packed_f32x2(x, self.c_half)

        x2 = cute.arch.mul_packed_f32x2(x, x)
        x2_gemma_beta = cute.arch.fma_packed_f32x2(x2, self.c_gamma, self.c_beta)
        tanh2 = cute.arch.mul_packed_f32x2(tanh, tanh)
        one_minus_tanh2 = sub_packed_f32x2(self.c_one, tanh2)
        term1_tmp = cute.arch.mul_packed_f32x2(one_minus_tanh2, x2_gemma_beta)
        term1 = cute.arch.mul_packed_f32x2(half_x, term1_tmp)

        grad = cute.arch.fma_packed_f32x2(tanh_1, self.c_half, term1)
        act = cute.arch.mul_packed_f32x2(half_x, tanh_1)

        # pyre-ignore[60]
        return *act, *grad

    # pyre-ignore[56]
    @cute.jit
    def fast_gelu(self, x: tuple[Float32, Float32]) -> tuple[Float32, Float32]:
        # TODO: Seems like the complier may be able to handle this, without explicitly calling packed_f32x2:

        # x * 0.5 * (1+tanh)
        tanh = self.gelu_tanh(x)
        tanh_1 = cute.arch.add_packed_f32x2(tanh, self.c_one)
        out = cute.arch.mul_packed_f32x2(x, self.c_half)
        out = cute.arch.mul_packed_f32x2(out, tanh_1)
        return out

    # pyre-ignore[56]
    @cute.jit
    def grad_fast_gelu(self, x: tuple[Float32, Float32]) -> tuple[Float32, Float32]:
        # TODO: Seems like the complier may be able to handle this, without explicitly calling packed_f32x2:

        # 0.5 * x * ((1 - tanh²) * (0.7978845608 + 0.1070322243 × x²)) + 0.5 * (1+ tanh)
        tanh = self.gelu_tanh(x)
        x2 = cute.arch.mul_packed_f32x2(x, x)
        x2_gemma_beta = cute.arch.fma_packed_f32x2(x2, self.c_gamma, self.c_beta)
        tanh2 = cute.arch.mul_packed_f32x2(tanh, tanh)
        one_minus_tanh2 = sub_packed_f32x2(self.c_one, tanh2)
        term1_tmp = cute.arch.mul_packed_f32x2(one_minus_tanh2, x2_gemma_beta)
        half_x = cute.arch.mul_packed_f32x2(x, self.c_half)
        term1 = cute.arch.mul_packed_f32x2(half_x, term1_tmp)

        one_plus_tanh = cute.arch.add_packed_f32x2(self.c_one, tanh)

        out = cute.arch.fma_packed_f32x2(one_plus_tanh, self.c_half, term1)
        return out

    # pyre-ignore[56]
    @cute.jit
    def activation_and_gradient_gelu_taylor_deg6(
        self, x: tuple[Float32, Float32]
    ) -> tuple[Float32, Float32, Float32, Float32]:
        # act: 0.5*x + x2*(c2 + x2*(c4 + x2*c6))
        #     = 0.5 * x + x^2 * c2 + x^4 * c4 + x^6 * c6
        x2 = cute.arch.mul_packed_f32x2(x, x)
        x_half = cute.arch.mul_packed_f32x2(x, self.c_half)
        tmp = cute.arch.fma_packed_f32x2(
            x2, self.c_taylor_c6, self.c_taylor_c4
        )  # c4 + x2*c6
        tmp = cute.arch.fma_packed_f32x2(x2, tmp, self.c_taylor_c2)  # c2 + x2*tmp

        # grad: 0.5 + (2x * c2) + (4x * x^2 * c4) + (6x * x^4 * c6)
        #    = 0.5 + (2*c2 * x) + (4*c4 * x * x^2) + (6*c6 * x * x^4)
        #     = 0.5 + (2*c2 * x) + (4*c4 * x^3) + (6*c6 * x^2 * x^3)
        x3 = cute.arch.mul_packed_f32x2(x2, x)
        x2_c6 = cute.arch.mul_packed_f32x2(self.c_taylor_6_mul_c6, x2)
        part3 = cute.arch.mul_packed_f32x2(x3, x2_c6)  # (6*c6 * x^2 * x^3)
        part2 = cute.arch.fma_packed_f32x2(
            x3, self.c_taylor_4_mul_c4, part3
        )  # (4*c4 * x^3) + (6*c6 * x^2 * x^3)
        part1 = cute.arch.fma_packed_f32x2(
            x, self.c_talyor_2_mul_c2, self.c_half
        )  # 0.5 + (2*c2 * x)

        grad = cute.arch.add_packed_f32x2(part1, part2)
        act = cute.arch.fma_packed_f32x2(x2, tmp, x_half)  # 0.5*x + x2*tmp
        # pyre-ignore[60]
        return *act, *grad

    # pyre-ignore[56]
    @cute.jit
    def grad_gelu_taylor_deg6(
        self, x: tuple[Float32, Float32]
    ) -> tuple[Float32, Float32]:
        # 0.5*x + x2*(c2 + x2*(c4 + x2*c6))
        # grad: 0.5 + 2x * (c2 + x^2 * (c4 + x^2 * c6)) + x^2 * (2x * (c4 + x^2 * c6) + x^2 * (2x*c6))
        #     = 0.5 + 2x * c2 + 2 * x^2 * 2x * (c4 + x^2 * c6) + x^2 * x^2 * 2x * c6
        #     = 0.5 + 2x * c2 + 2 * x^2 * 2x * c4 + 2 * x^4 * 2x * c6 + x^4 * 2x * c6
        #     = 0.5 + (2x * c2) + (2 * x^2 * 2x * c4) + (3 * x^4 * 2x * c6)
        #     = 0.5 + (2x * c2) + (4x * x^2 * c4) + (6x * x^4 * c6)
        #     = 0.5 + (2*c2 * x) + (4*c4 * x * x^2) + (6*c6 * x * x^4)
        #     = 0.5 + (2*c2 * x) + (4*c4 * x^3) + (6*c6 * x^2 * x^3)
        x2 = cute.arch.mul_packed_f32x2(x, x)
        x3 = cute.arch.mul_packed_f32x2(x2, x)
        x2_c6 = cute.arch.mul_packed_f32x2(self.c_taylor_6_mul_c6, x2)
        part3 = cute.arch.mul_packed_f32x2(x3, x2_c6)  # (6*c6 * x^2 * x^3)
        part2 = cute.arch.fma_packed_f32x2(
            x3, self.c_taylor_4_mul_c4, part3
        )  # (4*c4 * x^3) + (6*c6 * x^2 * x^3)
        part1 = cute.arch.fma_packed_f32x2(
            x, self.c_talyor_2_mul_c2, self.c_half
        )  # 0.5 + (2*c2 * x)

        grad = cute.arch.add_packed_f32x2(part1, part2)

        return grad

    # pyre-ignore[56]
    @cute.jit
    def gelu_taylor_deg6(self, x: tuple[Float32, Float32]) -> tuple[Float32, Float32]:
        # 0.5*x + x2*(c2 + x2*(c4 + x2*c6))
        x2 = cute.arch.mul_packed_f32x2(x, x)
        x_half = cute.arch.mul_packed_f32x2(x, self.c_half)
        tmp = cute.arch.fma_packed_f32x2(
            x2, self.c_taylor_c6, self.c_taylor_c4
        )  # c4 + x2*c6
        tmp = cute.arch.fma_packed_f32x2(x2, tmp, self.c_taylor_c2)  # c2 + x2*tmp
        out = cute.arch.fma_packed_f32x2(x2, tmp, x_half)  # 0.5*x + x2*tmp
        return out

    # pyre-ignore[56]
    @cute.jit
    def gelu_taylor_deg10(self, x: tuple[Float32, Float32]) -> tuple[Float32, Float32]:
        # 0.5*x + x^2*(c2 + x^2*(c4 + x^2*(c6 + x^2*(c8 + x^2*c10))))
        x2 = cute.arch.mul_packed_f32x2(x, x)
        x_half = cute.arch.mul_packed_f32x2(x, self.c_half)
        tmp = cute.arch.fma_packed_f32x2(
            x2, self.c_taylor_c10, self.c_taylor_c8
        )  # c8 + x^2*c10
        tmp = cute.arch.fma_packed_f32x2(x2, tmp, self.c_taylor_c6)  # c6 + x^2*tmp
        tmp = cute.arch.fma_packed_f32x2(x2, tmp, self.c_taylor_c4)  # c4 + x^2*tmp
        tmp = cute.arch.fma_packed_f32x2(x2, tmp, self.c_taylor_c2)  # c2 + x^2*tmp
        out = cute.arch.fma_packed_f32x2(x2, tmp, x_half)  # 0.5*x + x^2*tmp
        return out

    # pyre-ignore[56]
    @cute.jit
    def gelu_and_convert(
        self,
        acc_S_row: cute.Tensor,
        acc_S_row_converted: cute.Tensor,
        e2e_freq: cutlass.Constexpr[int] = 16,
        e2e_res: cutlass.Constexpr[int] = 4,
        e2e_frg_limit: cutlass.Constexpr[int] = 1,
    ) -> None:
        assert cute.size(acc_S_row.shape) % 2 == 0, (
            "acc_S_row must have an even number of elements"
        )
        frg_tile = 32
        assert frg_tile % 2 == 0
        frg_cnt = cute.size(acc_S_row) // frg_tile
        assert cute.size(acc_S_row) % frg_tile == 0
        acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))
        acc_S_row_converted_frg = cute.logical_divide(
            acc_S_row_converted, cute.make_layout(frg_tile)
        )
        for j in cutlass.range_constexpr(frg_cnt):
            for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):
                if cutlass.const_expr(
                    k % e2e_freq < e2e_freq - e2e_res or j >= frg_cnt - e2e_frg_limit
                ):
                    s_f2 = (acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j])
                    acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = self.fast_gelu(s_f2)
                else:
                    s_f2 = (acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j])
                    acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = (
                        self.gelu_taylor_deg6(s_f2)
                    )
            acc_S_row_converted_frg[None, j].store(
                acc_S_row_frg[None, j].load().to(acc_S_row_converted.element_type)
            )

    # pyre-ignore[56]
    @cute.jit
    def grad_gelu_and_convert(
        self,
        acc_S_row: cute.Tensor,
        acc_dS_row: cute.Tensor,
        acc_P_row: cute.Tensor,
        acc_P_row_f16: cute.Tensor,
        e2e_freq: cutlass.Constexpr[int] = 16,
        e2e_res: cutlass.Constexpr[int] = 4,
        e2e_frg_limit: cutlass.Constexpr[int] = 1,
    ) -> None:
        assert cute.size(acc_S_row.shape) % 2 == 0, (
            "acc_S_row must have an even number of elements"
        )
        frg_tile = 32
        assert frg_tile % 2 == 0
        frg_cnt = cute.size(acc_S_row) // frg_tile
        assert cute.size(acc_S_row) % frg_tile == 0
        acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))
        acc_P_row_frg = cute.logical_divide(acc_P_row, cute.make_layout(frg_tile))
        acc_dS_row_frg = cute.logical_divide(acc_dS_row, cute.make_layout(frg_tile))
        acc_P_row_f16_frg = cute.logical_divide(
            acc_P_row_f16, cute.make_layout(frg_tile)
        )

        for j in cutlass.range_constexpr(frg_cnt):
            for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):
                if cutlass.const_expr(
                    k % e2e_freq < e2e_freq - e2e_res or j >= frg_cnt - e2e_frg_limit
                ):
                    s_f2 = (acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j])
                    (
                        acc_P_row_frg[k, j],
                        acc_P_row_frg[k + 1, j],
                        acc_dS_row_frg[k, j],
                        acc_dS_row_frg[k + 1, j],
                    ) = self.activation_and_gradient_fast_gelu(s_f2)
                else:
                    s_f2 = (acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j])
                    (
                        acc_P_row_frg[k, j],
                        acc_P_row_frg[k + 1, j],
                        acc_dS_row_frg[k, j],
                        acc_dS_row_frg[k + 1, j],
                    ) = self.activation_and_gradient_gelu_taylor_deg6(s_f2)
            acc_P_row_f16_frg[None, j].store(
                acc_P_row_frg[None, j].load().to(acc_P_row_f16_frg.element_type)
            )

    # pyre-ignore[56]
    @cute.jit
    def gelu_and_convert_blockscaled(
        self,
        acc_S_row: cute.Tensor,
        acc_S_row_converted: cute.Tensor,
    ) -> tuple[Uint8, Uint8, Uint8, Uint8]:
        """Apply GELU, compute per-32-block scales, and quantize for MXFP8.

        Optimized version matching C++ performance (D89908999):
        - Uses 3-input max pattern for AMAX (FMNMX3 on Blackwell) with abs
        - Uses mul_packed_f32x2 for SIMD scaling (FMUL2)
        - Uses cvt_8x_e4m3 for efficient FP8 conversion
        - Direct uint32 stores to avoid PRMT instructions

        Unlike ReLU, GELU produces negative values so we:
        1. Apply GELU first (can't fuse into convert)
        2. Compute AMAX using abs() before max
        3. Scale + convert + store

        Args:
            acc_S_row: Input tensor with 128 F32 values from TMEM (S accumulator)
            acc_S_row_converted: Output tensor for 128 FP8 values (P for TMEM)

        Returns:
            Tuple of 4 E8M0 scale factors (one per 32-element block)
        """
        BLOCK_SIZE = 32
        max_norm_rcp = Float32(E4M3_MAX_NORM_RCP)

        acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(BLOCK_SIZE))
        # Recast converted tensor to Uint32 for direct 4-byte stores
        acc_S_row_converted_u32_ptr = cute.recast_ptr(
            acc_S_row_converted.iterator, dtype=Uint32
        )
        acc_S_row_converted_u32 = cute.make_tensor(
            acc_S_row_converted_u32_ptr,
            cute.recast_layout(32, 8, acc_S_row_converted.layout),
        )
        acc_S_row_converted_u32_frg = cute.logical_divide(
            acc_S_row_converted_u32, cute.make_layout(BLOCK_SIZE // 4)
        )

        # ===== Process block 0 =====
        # Step 1a: Apply GELU to all 32 elements
        for k in cutlass.range_constexpr(0, BLOCK_SIZE, 2):
            s_f2 = (acc_S_row_frg[k, 0], acc_S_row_frg[k + 1, 0])
            gelu_f2 = self.fast_gelu(s_f2)
            acc_S_row_frg[k, 0], acc_S_row_frg[k + 1, 0] = gelu_f2

        # Step 1b: AMAX using 3-input max with abs (GELU can produce negatives)
        block_amax_0 = Float32(0.0)
        # Process 30 elements in groups of 3 for FMNMX3
        for k in cutlass.range_constexpr(0, 30, 3):
            block_amax_0 = max3_f32(
                block_amax_0,
                abs_f32(acc_S_row_frg[k, 0]),
                max_f32(
                    abs_f32(acc_S_row_frg[k + 1, 0]), abs_f32(acc_S_row_frg[k + 2, 0])
                ),
            )
        # Remainder (2 elements)
        block_amax_0 = max_f32(block_amax_0, abs_f32(acc_S_row_frg[30, 0]))
        block_amax_0 = max_f32(block_amax_0, abs_f32(acc_S_row_frg[31, 0]))

        # Step 2: Compute E8M0 scale
        scale0, inv_scale_0 = fused_amax_to_e8m0_scale_f32(block_amax_0, max_norm_rcp)
        inv_scale_0_pair = (inv_scale_0, inv_scale_0)

        # Step 3: SIMD scale (FMUL2) + convert + direct u32 store (8 elements at a time)
        for k in cutlass.range_constexpr(0, BLOCK_SIZE, 8):
            # SIMD float2 multiplication using mul_packed_f32x2 (generates FMUL2)
            s01 = mul_packed_f32x2(
                (acc_S_row_frg[k + 0, 0], acc_S_row_frg[k + 1, 0]), inv_scale_0_pair
            )
            s23 = mul_packed_f32x2(
                (acc_S_row_frg[k + 2, 0], acc_S_row_frg[k + 3, 0]), inv_scale_0_pair
            )
            s45 = mul_packed_f32x2(
                (acc_S_row_frg[k + 4, 0], acc_S_row_frg[k + 5, 0]), inv_scale_0_pair
            )
            s67 = mul_packed_f32x2(
                (acc_S_row_frg[k + 6, 0], acc_S_row_frg[k + 7, 0]), inv_scale_0_pair
            )
            # Convert to FP8 (no .relu - GELU already applied)
            fp8_packed = cvt_8x_e4m3(
                s01[0], s01[1], s23[0], s23[1], s45[0], s45[1], s67[0], s67[1]
            )
            # Direct uint32 store (avoids PRMT)
            lo32, hi32 = unpack_i64_to_u32_pair(fp8_packed)
            base_idx = k // 4
            acc_S_row_converted_u32_frg[base_idx, 0] = lo32
            acc_S_row_converted_u32_frg[base_idx + 1, 0] = hi32

        # ===== Process block 1 =====
        for k in cutlass.range_constexpr(0, BLOCK_SIZE, 2):
            s_f2 = (acc_S_row_frg[k, 1], acc_S_row_frg[k + 1, 1])
            gelu_f2 = self.fast_gelu(s_f2)
            acc_S_row_frg[k, 1], acc_S_row_frg[k + 1, 1] = gelu_f2

        block_amax_1 = Float32(0.0)
        for k in cutlass.range_constexpr(0, 30, 3):
            block_amax_1 = max3_f32(
                block_amax_1,
                abs_f32(acc_S_row_frg[k, 1]),
                max_f32(
                    abs_f32(acc_S_row_frg[k + 1, 1]), abs_f32(acc_S_row_frg[k + 2, 1])
                ),
            )
        block_amax_1 = max_f32(block_amax_1, abs_f32(acc_S_row_frg[30, 1]))
        block_amax_1 = max_f32(block_amax_1, abs_f32(acc_S_row_frg[31, 1]))
        scale1, inv_scale_1 = fused_amax_to_e8m0_scale_f32(block_amax_1, max_norm_rcp)
        inv_scale_1_pair = (inv_scale_1, inv_scale_1)

        for k in cutlass.range_constexpr(0, BLOCK_SIZE, 8):
            s01 = mul_packed_f32x2(
                (acc_S_row_frg[k + 0, 1], acc_S_row_frg[k + 1, 1]), inv_scale_1_pair
            )
            s23 = mul_packed_f32x2(
                (acc_S_row_frg[k + 2, 1], acc_S_row_frg[k + 3, 1]), inv_scale_1_pair
            )
            s45 = mul_packed_f32x2(
                (acc_S_row_frg[k + 4, 1], acc_S_row_frg[k + 5, 1]), inv_scale_1_pair
            )
            s67 = mul_packed_f32x2(
                (acc_S_row_frg[k + 6, 1], acc_S_row_frg[k + 7, 1]), inv_scale_1_pair
            )
            fp8_packed = cvt_8x_e4m3(
                s01[0], s01[1], s23[0], s23[1], s45[0], s45[1], s67[0], s67[1]
            )
            lo32, hi32 = unpack_i64_to_u32_pair(fp8_packed)
            base_idx = k // 4
            acc_S_row_converted_u32_frg[base_idx, 1] = lo32
            acc_S_row_converted_u32_frg[base_idx + 1, 1] = hi32

        # ===== Process block 2 =====
        for k in cutlass.range_constexpr(0, BLOCK_SIZE, 2):
            s_f2 = (acc_S_row_frg[k, 2], acc_S_row_frg[k + 1, 2])
            gelu_f2 = self.fast_gelu(s_f2)
            acc_S_row_frg[k, 2], acc_S_row_frg[k + 1, 2] = gelu_f2

        block_amax_2 = Float32(0.0)
        for k in cutlass.range_constexpr(0, 30, 3):
            block_amax_2 = max3_f32(
                block_amax_2,
                abs_f32(acc_S_row_frg[k, 2]),
                max_f32(
                    abs_f32(acc_S_row_frg[k + 1, 2]), abs_f32(acc_S_row_frg[k + 2, 2])
                ),
            )
        block_amax_2 = max_f32(block_amax_2, abs_f32(acc_S_row_frg[30, 2]))
        block_amax_2 = max_f32(block_amax_2, abs_f32(acc_S_row_frg[31, 2]))
        scale2, inv_scale_2 = fused_amax_to_e8m0_scale_f32(block_amax_2, max_norm_rcp)
        inv_scale_2_pair = (inv_scale_2, inv_scale_2)

        for k in cutlass.range_constexpr(0, BLOCK_SIZE, 8):
            s01 = mul_packed_f32x2(
                (acc_S_row_frg[k + 0, 2], acc_S_row_frg[k + 1, 2]), inv_scale_2_pair
            )
            s23 = mul_packed_f32x2(
                (acc_S_row_frg[k + 2, 2], acc_S_row_frg[k + 3, 2]), inv_scale_2_pair
            )
            s45 = mul_packed_f32x2(
                (acc_S_row_frg[k + 4, 2], acc_S_row_frg[k + 5, 2]), inv_scale_2_pair
            )
            s67 = mul_packed_f32x2(
                (acc_S_row_frg[k + 6, 2], acc_S_row_frg[k + 7, 2]), inv_scale_2_pair
            )
            fp8_packed = cvt_8x_e4m3(
                s01[0], s01[1], s23[0], s23[1], s45[0], s45[1], s67[0], s67[1]
            )
            lo32, hi32 = unpack_i64_to_u32_pair(fp8_packed)
            base_idx = k // 4
            acc_S_row_converted_u32_frg[base_idx, 2] = lo32
            acc_S_row_converted_u32_frg[base_idx + 1, 2] = hi32

        # ===== Process block 3 =====
        for k in cutlass.range_constexpr(0, BLOCK_SIZE, 2):
            s_f2 = (acc_S_row_frg[k, 3], acc_S_row_frg[k + 1, 3])
            gelu_f2 = self.fast_gelu(s_f2)
            acc_S_row_frg[k, 3], acc_S_row_frg[k + 1, 3] = gelu_f2

        block_amax_3 = Float32(0.0)
        for k in cutlass.range_constexpr(0, 30, 3):
            block_amax_3 = max3_f32(
                block_amax_3,
                abs_f32(acc_S_row_frg[k, 3]),
                max_f32(
                    abs_f32(acc_S_row_frg[k + 1, 3]), abs_f32(acc_S_row_frg[k + 2, 3])
                ),
            )
        block_amax_3 = max_f32(block_amax_3, abs_f32(acc_S_row_frg[30, 3]))
        block_amax_3 = max_f32(block_amax_3, abs_f32(acc_S_row_frg[31, 3]))
        scale3, inv_scale_3 = fused_amax_to_e8m0_scale_f32(block_amax_3, max_norm_rcp)
        inv_scale_3_pair = (inv_scale_3, inv_scale_3)

        for k in cutlass.range_constexpr(0, BLOCK_SIZE, 8):
            s01 = mul_packed_f32x2(
                (acc_S_row_frg[k + 0, 3], acc_S_row_frg[k + 1, 3]), inv_scale_3_pair
            )
            s23 = mul_packed_f32x2(
                (acc_S_row_frg[k + 2, 3], acc_S_row_frg[k + 3, 3]), inv_scale_3_pair
            )
            s45 = mul_packed_f32x2(
                (acc_S_row_frg[k + 4, 3], acc_S_row_frg[k + 5, 3]), inv_scale_3_pair
            )
            s67 = mul_packed_f32x2(
                (acc_S_row_frg[k + 6, 3], acc_S_row_frg[k + 7, 3]), inv_scale_3_pair
            )
            fp8_packed = cvt_8x_e4m3(
                s01[0], s01[1], s23[0], s23[1], s45[0], s45[1], s67[0], s67[1]
            )
            lo32, hi32 = unpack_i64_to_u32_pair(fp8_packed)
            base_idx = k // 4
            acc_S_row_converted_u32_frg[base_idx, 3] = lo32
            acc_S_row_converted_u32_frg[base_idx + 1, 3] = hi32

        return scale0, scale1, scale2, scale3
