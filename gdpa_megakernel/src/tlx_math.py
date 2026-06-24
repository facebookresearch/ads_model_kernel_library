# pyre-ignore-all-errors

from enum import Enum
from typing import Any

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

try:
    from triton.language.extra.libdevice import fast_dividef, fast_expf
except ImportError:
    try:
        from triton.language.extra.cuda.libdevice import fast_dividef, fast_expf
    except ImportError:
        from triton.language.math import fast_dividef, fast_expf


class Activation(str, Enum):
    Raw = "raw"
    GeLU = "gelu"
    GeLUApprox = "gelu_approx"
    FastGeLU = "fast_gelu"
    LeakyReLU = "leaky_relu"
    ReLU = "relu"
    FastGeLUBF16 = "fast_gelu_bf16"
    SiLu = "silu"
    FastSiLu = "fast_silu"
    HardSwish = "hardswish"
    ReLUSquare = "relu_square"
    FastGeLUTaylorApprox = "fast_gelu_taylor_approx"


activation_to_int = {act: i for i, act in enumerate(Activation)}


def activation_string_to_int(s: str):
    if s not in Activation._value2member_map_:
        raise ValueError(f"Unsupported activation function: {s}")
    return activation_to_int.get(Activation(s))


def get_pytorch_activation(activation: str):
    return {
        Activation.Raw.value: torch.nn.Identity(),
        Activation.GeLU.value: torch.nn.GELU(),
        Activation.GeLUApprox.value: torch.nn.GELU(approximate="tanh"),
        Activation.FastGeLU.value: torch.nn.GELU(approximate="tanh"),
        Activation.LeakyReLU.value: torch.nn.LeakyReLU(0.01),
        Activation.ReLU.value: torch.nn.ReLU(),
        Activation.FastGeLUBF16.value: torch.nn.GELU(approximate="tanh"),
        Activation.SiLu.value: torch.nn.SiLU(),
        Activation.FastSiLu.value: torch.nn.SiLU(),
        Activation.HardSwish.value: torch.nn.Hardswish(),
        Activation.ReLUSquare.value: lambda x: torch.relu(x) * torch.relu(x),
        Activation.FastGeLUTaylorApprox.value: torch.nn.GELU(approximate="tanh"),
    }[activation]


@triton.jit
def raw(x):
    return x


@triton.jit
def raw_grad(x):
    return tl.full(x.shape, 1.0, x.dtype)


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def relu(x):
    zero = 0.0
    return tl.where(x >= 0, x, zero.to(x.dtype))


@triton.jit
def relu_grad(x):
    zero = 0.0
    one = 1.0
    return tl.where(x >= 0, one.to(x.dtype), zero.to(x.dtype))


@triton.jit
def relu_square(x):
    zero = 0.0
    y = tl.where(x >= 0, x, zero.to(x.dtype))
    return y * y


@triton.jit
def relu_square_grad(x):
    zero = 0.0
    two = 2.0
    return tl.where(x >= 0, two.to(x.dtype) * x, zero.to(x.dtype))


@triton.jit
def leaky_relu(x):
    scale = 0.01 + 0.0
    return tl.where(x >= 0, x, scale.to(x.dtype) * x)


@triton.jit
def leaky_relu_grad(x):
    min_grad = 0.01
    max_grad = 1.0
    return tl.where(x >= 0, max_grad.to(x.dtype), min_grad.to(x.dtype))


@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + libdevice.erf(x * 0.7071067811865476))


@triton.jit
def gelu_grad(x):
    cdf = 0.5 * (1.0 + libdevice.erf(x * 0.7071067811865476))
    pdf = tl.exp(-0.5 * x * x) * 0.3989422804014327
    return cdf + x * pdf


@triton.jit
def gelu_approx(x):
    return 0.5 * x * (1.0 + tanh(0.7978845608 * x * (1.0 + 0.044715 * x * x)))


@triton.jit
def gelu_approx_grad(x):
    tanh_out = tanh(0.7978845608 * x * (1 + 0.044715 * x * x))
    return 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.7978845608 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)


@triton.jit
def _add_f32x2(a: Any, b: Any) -> Any:
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            add.f32x2 rc, ra, rb;
            mov.b64 { $0, $1 }, rc;
        }
        """,
        "=r,=r,r,r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _sub_f32x2(a: Any, b: Any) -> Any:
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            sub.f32x2 rc, ra, rb;
            mov.b64 { $0, $1 }, rc;
        }
        """,
        "=r,=r,r,r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _mul_f32x2(a: Any, b: Any) -> Any:
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            mul.f32x2 rc, ra, rb;
            mov.b64 { $0, $1 }, rc;
        }
        """,
        "=r,=r,r,r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _fma_f32x2(a: Any, b: Any, c: Any) -> Any:
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b64 ra, rb, rc, rd;
            mov.b64 ra, { $2, $3 };
            mov.b64 rb, { $4, $5 };
            mov.b64 rc, { $6, $7 };
            fma.rn.f32x2 rd, ra, rb, rc;
            mov.b64 { $0, $1 }, rd;
        }
        """,
        "=r,=r,r,r,r,r,r,r",
        [a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )


@triton.jit
def tanh_approx_fp32(x: Any) -> Any:
    return tl.inline_asm_elementwise(
        asm="""
            tanh.approx.f32 $0, $1;
            """,
        constraints="=r,r",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def tanh_approx_bf16(x: Any) -> Any:
    return tl.inline_asm_elementwise(
        asm="""
        tanh.approx.bf16 $0, $1;
        """,
        constraints="=h,h",
        args=[x],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=1,
    )


@triton.jit
def fast_gelu(x: Any) -> Any:
    c1 = 0.0356774081
    c0 = 0.7978845608
    c2 = 0.5
    square = _mul_f32x2(x, x)
    inner = _fma_f32x2(c1, square, c0)
    inner = _mul_f32x2(inner, x)
    out = _fma_f32x2(x, tanh_approx_fp32(inner), x)
    return _mul_f32x2(out, c2)


@triton.jit
def fast_gelu_grad(x: Any) -> Any:
    c1 = 0.0356774081
    c0 = 0.7978845608
    c2 = 0.5
    c3 = 0.1070322243
    square = _mul_f32x2(x, x)
    inner = _mul_f32x2(_fma_f32x2(c1, square, c0), x)
    tanh_out = tanh_approx_fp32(inner)
    x2_term = _fma_f32x2(c3, square, c0)
    tanh2_term = _sub_f32x2(1.0, _mul_f32x2(tanh_out, tanh_out))
    out = _add_f32x2(_mul_f32x2(_mul_f32x2(tanh2_term, x2_term), x), tanh_out)
    return _fma_f32x2(c2, out, c2)


@triton.jit
def fast_gelu_bf16(x):
    return x * 0.5 * (1 + tanh_approx_bf16(0.796875 * x * (1.0 + 0.044715 * x * x)))


@triton.jit
def fast_gelu_bf16_grad(x):
    tanh_out = tanh_approx_bf16(0.7978845608 * x * (1.0 + 0.044715 * x * x))
    return 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.7978845608 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def silu_grad(x):
    sig = tl.sigmoid(x)
    return sig * (1 + x * (1 - sig))


@triton.jit
def fast_silu(x):
    return fast_dividef(x, 1.0 + fast_expf(-x))


@triton.jit
def fast_silu_grad(x):
    sig = fast_dividef(1.0, 1.0 + fast_expf(-x))
    return sig * (1 + x * (1 - sig))


@triton.jit
def hardswish(x):
    zero = 0.0
    six = 6.0
    three = 3.0
    inv_six = 1.0 / 6.0
    t = tl.clamp(x + three.to(x.dtype), zero.to(x.dtype), six.to(x.dtype))
    return x * t * inv_six.to(x.dtype)


@triton.jit
def hardswish_grad(x):
    zero = 0.0
    one = 1.0
    three = 3.0
    half = 0.5
    third = 1.0 / 3.0
    cond_neg = x <= (-three.to(x.dtype))
    cond_pos = x >= three.to(x.dtype)
    mid = third.to(x.dtype) * x + half.to(x.dtype)
    return tl.where(
        cond_pos, one.to(x.dtype), tl.where(cond_neg, zero.to(x.dtype), mid)
    )


@triton.jit
def fast_gelu_taylor_deg6(x: Any) -> Any:
    c_half = 0.5
    c_taylor_c2 = 0.3989422804014327
    c_taylor_c4 = -0.06649038006690543
    c_taylor_c6 = 0.00997355701003582
    x2 = _mul_f32x2(x, x)
    x_half = _mul_f32x2(x, c_half)
    tmp = _fma_f32x2(x2, c_taylor_c6, c_taylor_c4)
    tmp = _fma_f32x2(x2, tmp, c_taylor_c2)
    return _fma_f32x2(x2, tmp, x_half)


@triton.jit
def _split_n(x: Any, SPLIT_FACTOR: tl.constexpr) -> Any:
    if SPLIT_FACTOR == 1:
        return (x,)
    x0, x1 = x.reshape([x.shape[0], 2, x.shape[1] // 2]).permute(0, 2, 1).split()
    return _split_n(x0, SPLIT_FACTOR // 2) + _split_n(x1, SPLIT_FACTOR // 2)


@triton.jit
def _join_n(xs: Any) -> Any:
    if len(xs) == 1:
        return xs[0]
    x0 = _join_n(xs[: len(xs) // 2])
    x1 = _join_n(xs[len(xs) // 2 :])
    return tl.join(x0, x1).permute(0, 2, 1).reshape([x0.shape[0], x0.shape[1] * 2])


@triton.jit
def column_wise_split_fast_gelu(x: Any, SPLIT_FACTOR: tl.constexpr) -> Any:
    xs = _split_n(x, SPLIT_FACTOR)
    res = ()
    for sid in tl.static_range(0, SPLIT_FACTOR):
        if sid == SPLIT_FACTOR - 1:
            x_i = fast_gelu_taylor_deg6(xs[sid])
        else:
            x_i = fast_gelu(xs[sid])
        res = res + (x_i,)
    return _join_n(res)


@triton.jit
def apply_activation(
    qk: Any,
    dtype: Any,
    activation_enum_int: Any,
    SPLIT_FACTOR: tl.constexpr,
) -> Any:
    if activation_enum_int == 0:
        p = raw(qk)
    elif activation_enum_int == 1:
        p = gelu(qk)
    elif activation_enum_int == 2:
        p = gelu_approx(qk)
    elif activation_enum_int == 3:
        p = fast_gelu(qk)
    elif activation_enum_int == 4:
        p = leaky_relu(qk)
    elif activation_enum_int == 5:
        p = relu(qk)
    elif activation_enum_int == 6:
        qk_bf16 = qk.to(dtype)
        p = fast_gelu_bf16(qk_bf16).to(tl.float32)
    elif activation_enum_int == 7:
        p = silu(qk)
    elif activation_enum_int == 8:
        p = fast_silu(qk)
    elif activation_enum_int == 9:
        p = hardswish(qk)
    elif activation_enum_int == 10:
        p = relu_square(qk)
    else:
        p = qk
    return p


@triton.jit
def apply_activation_grad(
    qk: Any,
    dtype: Any,
    activation_enum_int: Any,
) -> Any:
    if activation_enum_int == 0:
        p = raw_grad(qk)
    elif activation_enum_int == 1:
        p = gelu_grad(qk)
    elif activation_enum_int == 2:
        p = gelu_approx_grad(qk)
    elif activation_enum_int == 3:
        p = fast_gelu_grad(qk)
    elif activation_enum_int == 4:
        p = leaky_relu_grad(qk)
    elif activation_enum_int == 5:
        p = relu_grad(qk)
    elif activation_enum_int == 6:
        qk_bf16 = qk.to(dtype)
        p = fast_gelu_bf16_grad(qk_bf16).to(tl.float32)
    elif activation_enum_int == 7:
        p = silu_grad(qk)
    elif activation_enum_int == 8:
        p = fast_silu_grad(qk)
    elif activation_enum_int == 9:
        p = hardswish_grad(qk)
    elif activation_enum_int == 10:
        p = relu_square_grad(qk)
    else:
        p = qk
    return p
