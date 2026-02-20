# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""
This file defines common math functions, sometimes relying on optimized PTX for performance. Note that the functions relying on PTX
will only be supported on NVIDIA GPUs
"""

from enum import Enum
from typing import Optional

import torch
import triton
import triton.language as tl
from ads_mkl.ops.cute_dsl.gdpa.triton.hardware import is_amd
from torch._inductor.runtime.triton_helpers import libdevice

try:
    from triton.language.extra.libdevice import fast_dividef, fast_expf
except ImportError:
    try:
        from triton.language.extra.cuda.libdevice import fast_dividef, fast_expf
    except ImportError:
        from triton.language.math import fast_dividef, fast_expf  # pyre-fixme[21]


# Don't change the order of the enum values, as they are used to index
# Only add new activation functions at the end of the enum
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


def get_pytorch_activation(activation: Activation):
    return {
        Activation.ReLU: torch.nn.ReLU(),
        Activation.LeakyReLU: torch.nn.LeakyReLU(),
        Activation.GeLU: torch.nn.GELU(),
        Activation.GeLUApprox: torch.nn.GELU(approximate="tanh"),
        Activation.FastGeLU: torch.nn.GELU(
            approximate="tanh"
        ),  # fast gelu use tanh approximation similar to approximate gelu
        Activation.FastGeLUBF16: torch.nn.GELU(approximate="tanh"),
        Activation.Raw: torch.nn.Identity(),
        Activation.SiLu: torch.nn.SiLU(),
        Activation.FastSiLu: torch.nn.SiLU(),
        Activation.HardSwish: torch.nn.Hardswish(),
        Activation.ReLUSquare: (lambda x: torch.relu(x) * torch.relu(x)),
    }[activation]


def get_triton_activation_kernel(activation: Optional[Activation]):
    return (
        {
            Activation.Raw: raw,
            Activation.ReLU: relu,
            Activation.LeakyReLU: leaky_relu,
            Activation.GeLU: gelu,
            Activation.GeLUApprox: gelu_approx,
            Activation.FastGeLU: fast_gelu,
            Activation.HardSwish: hardswish,
            Activation.ReLUSquare: relu_square,
        }[activation]
        if activation
        else None
    )


def get_triton_activation_bwd_kernel(activation: Optional[Activation]):
    return (
        {
            Activation.Raw: raw_grad,
            Activation.ReLU: relu_grad,
            Activation.LeakyReLU: leaky_relu_grad,
            Activation.GeLU: gelu_grad,
            Activation.GeLUApprox: gelu_approx_grad,
            Activation.FastGeLU: fast_gelu_grad,
            Activation.HardSwish: hardswish_grad,
            Activation.ReLUSquare: relu_square_grad,
        }[activation]
        if activation
        else None
    )


# pyre-fixme[6]: For 1st argument expected `Iterable[_T]` but got `Type[Activation]`.
activation_to_int = {act: i for i, act in enumerate(Activation)}
int_to_activation = {i: act for act, i in activation_to_int.items()}


def activation_string_to_int(s: str):
    if s not in Activation._value2member_map_:
        raise ValueError(f"Unsupported activation function: {s}")
    enum_val = Activation(s)
    return activation_to_int.get(enum_val)


def is_hip_mtia_or_a100():
    try:
        if triton.runtime.driver.active.get_current_target().backend in ["hip", "mtia"]:
            return True
        elif torch.cuda.get_device_capability()[0] < 9:  # A100
            return True
        return False
    except Exception:
        return False


@triton.jit  # pragma: no cover
def raw(x):
    return x


@triton.jit  # pragma: no cover
def raw_grad(x: tl.tensor) -> float:
    return 1.0


@triton.jit  # pragma: no cover
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit  # pragma: no cover
def cosh(x):
    exp_x = tl.exp(x)
    return (exp_x + 1.0 / exp_x) * 0.5


@triton.jit  # pragma: no cover
def relu(x: tl.tensor) -> tl.tensor:
    zero = 0.0
    return tl.where(x >= 0, x, zero.to(x.dtype))  # pyre-fixme[16]


@triton.jit  # pragma: no cover
def relu_grad(x):
    zero = 0.0
    one = 1.0
    return tl.where(x >= 0, one.to(x.dtype), zero.to(x.dtype))


@triton.jit  # pragma: no cover
def relu_square(x):
    zero = 0.0
    y = tl.where(x >= 0, x, zero.to(x.dtype))
    return y * y


@triton.jit  # pragma: no cover
def relu_square_grad(x):
    zero = 0.0
    two = 2.0
    return tl.where(x >= 0, two.to(x.dtype) * x, zero.to(x.dtype))


@triton.jit  # pragma: no cover
def leaky_relu(x):
    scale = 0.01 + 0.0
    scale = scale.to(x.dtype)
    return tl.where(x >= 0, x, scale * x)


@triton.jit  # pragma: no cover
def leaky_relu_grad(x):
    min_grad = 0.01
    max_grad = 1

    min_grad = min_grad.to(x.dtype)
    max_grad = max_grad.to(x.dtype)

    return tl.where(x >= 0, max_grad, min_grad)


@triton.jit  # pragma: no cover
def gelu(x):
    return x * 0.5 * (1.0 + libdevice.erf(x * 0.7071067811865476))


@triton.jit  # pragma: no cover
def gelu_grad(x):
    cdf = 0.5 * (1.0 + libdevice.erf(x * 0.7071067811865476))
    pdf = tl.exp(-0.5 * x * x) * 0.3989422804014327
    return cdf + x * pdf


@triton.jit  # pragma: no cover
def gelu_approx(x):
    return 0.5 * x * (1.0 + tanh(0.7978845608 * x * (1.0 + 0.044715 * x * x)))


@triton.jit  # pragma: no cover
def gelu_approx_grad(x):
    tanh_out = tanh(0.7978845608 * x * (1 + 0.044715 * x * x))
    return 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.7978845608 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)


if is_hip_mtia_or_a100():
    # For AMD or A100, use tanh as a fallback
    @triton.jit  # pragma: no cover
    def tanh_approx_fp32(x):
        return tanh(x)
else:

    @triton.jit  # pragma: no cover
    def tanh_approx_fp32(x):
        output = tl.inline_asm_elementwise(
            asm="""
            tanh.approx.f32 $0, $1;
            """,
            constraints="=r,r",
            args=[x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        return output


if is_hip_mtia_or_a100():
    # For AMD or A100, use tanh as a fallback
    if is_amd():

        @triton.jit  # pragma: no cover
        def tanh_approx_bf16(x):
            # 2 * tl.sigmoid(2 * x) - 1
            return 2 * sigmoid_approx_bf16(2 * x) - 1
    else:

        @triton.jit  # pragma: no cover
        def tanh_approx_bf16(x):
            return tanh(x)
else:

    @triton.jit  # pragma: no cover
    def tanh_approx_bf16(x):
        output = tl.inline_asm_elementwise(
            asm="""
            tanh.approx.bf16 $0, $1;
            """,
            constraints="=h,h",
            args=[x],
            dtype=tl.bfloat16,
            is_pure=True,
            pack=1,
        )
        return output


if is_hip_mtia_or_a100():
    # For AMD or A100, use tl.sigmoid as a fallback
    @triton.jit  # pragma: no cover
    def sigmoid_approx_fp32(x):
        return tl.sigmoid(x)
else:

    @triton.jit  # pragma: no cover
    def sigmoid_approx_fp32(x):
        output = 0.5 * tanh_approx_fp32(0.5 * x) + 0.5
        return output


if is_hip_mtia_or_a100():
    # For AMD or A100, use tl.sigmoid as a fallback
    if is_amd():

        @triton.jit  # pragma: no cover
        def sigmoid_approx_bf16(x: tl.tensor) -> tl.tensor:
            x32 = x.to(tl.float32)
            y = tl.sigmoid(x32)
            return y.to(tl.bfloat16)
    else:

        @triton.jit  # pragma: no cover
        def sigmoid_approx_bf16(x):
            return tl.sigmoid(x)
else:

    @triton.jit  # pragma: no cover
    def sigmoid_approx_bf16(x):
        output = 0.5 * tanh_approx_bf16(0.5 * x) + 0.5
        # output = fast_dividef(1.0, 1.0 + fast_expf(-x))
        return output


@triton.jit  # pragma: no cover
def fast_gelu(x):
    return x * 0.5 * (1 + tanh_approx_fp32(0.7978845608 * x * (1.0 + 0.044715 * x * x)))


@triton.jit  # pragma: no cover
def fast_gelu_grad(x):
    tanh_out = tanh_approx_fp32(0.7978845608 * x * (1.0 + 0.044715 * x * x))
    return 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.7978845608 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)


@triton.jit  # pragma: no cover
def fast_gelu_bf16(x):
    return x * 0.5 * (1 + tanh_approx_bf16(0.796875 * x * (1.0 + 0.044715 * x * x)))


@triton.jit  # pragma: no cover
def fast_gelu_bf16_grad(x):
    tanh_out = tanh_approx_bf16(0.7978845608 * x * (1.0 + 0.044715 * x * x))
    return 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.7978845608 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)


@triton.jit  # pragma: no cover
def silu(x: tl.tensor) -> tl.tensor:
    return x * tl.sigmoid(x)


@triton.jit  # pragma: no cover
def silu_grad(x):
    sig = tl.sigmoid(x)
    return sig * (1 + x * (1 - sig))


@triton.jit  # pragma: no cover
def fast_silu(x):
    return fast_dividef(x, 1.0 + fast_expf(-x))


@triton.jit  # pragma: no cover
def fast_silu_grad(x):
    sig = fast_dividef(1.0, 1.0 + fast_expf(-x))
    return sig * (1 + x * (1 - sig))


@triton.jit  # pragma: no cover
def hardswish(x):
    zero = 0.0
    six = 6.0
    three = 3.0
    inv_six = 1.0 / 6.0
    t = tl.minimum(tl.maximum(x + three.to(x.dtype), zero.to(x.dtype)), six.to(x.dtype))
    return x * t * inv_six.to(x.dtype)


@triton.jit  # pragma: no cover
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
