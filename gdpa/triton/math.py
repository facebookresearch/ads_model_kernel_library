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
This file defines common math functions, sometimes relying on optimized PTX for performance. Note that the functions relying on PTX
will only be supported on NVIDIA GPUs
"""

from enum import Enum
from typing import Optional

import torch
import triton
import triton.language as tl
from ads_mkl.ops.cute_dsl.gdpa.triton.hardware import is_mtia
from torch._inductor.runtime.triton_helpers import libdevice

from .hardware import is_amd

try:
    from triton.language.extra.libdevice import fast_dividef, fast_expf
except ImportError:
    try:
        from triton.language.extra.cuda.libdevice import fast_dividef, fast_expf
    except ImportError:
        from triton.language.math import fast_dividef, fast_expf  # type: ignore


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
    FastGeLUTaylorApprox = "fast_gelu_taylor_approx"


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
        Activation.FastGeLUTaylorApprox: torch.nn.GELU(approximate="tanh"),
    }[activation]


activation_to_int = {act: i for i, act in enumerate(Activation)}  # type: ignore
int_to_activation = {i: act for act, i in activation_to_int.items()}


def activation_string_to_int(s: str) -> Optional[int]:
    if s not in Activation._value2member_map_:
        raise ValueError(f"Unsupported activation function: {s}")
    enum_val = Activation(s)
    return activation_to_int.get(enum_val)


def is_mtia_or_a100() -> bool:
    try:
        if triton.runtime.driver.active.get_current_target().backend in ["mtia"]:
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
def raw_grad(x):
    return tl.full(x.shape, 1.0, x.dtype)


@triton.jit  # pragma: no cover
def tanh(x: tl.tensor) -> tl.tensor:
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1  # pyre-ignore[7]


@triton.jit  # pragma: no cover
def relu(x):
    zero = 0.0
    return tl.where(x >= 0, x, zero.to(x.dtype))  # type: ignore


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


if is_mtia():

    @triton.jit  # pragma: no cover
    def gelu(x):
        return libdevice.gelu(x)
else:

    @triton.jit  # pragma: no cover
    def gelu(x):
        return x * 0.5 * (1.0 + libdevice.erf(x * 0.7071067811865476))


if is_mtia():

    @triton.jit  # pragma: no cover
    def gelu_grad(x):
        return libdevice.dgelu(x)
else:

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


if is_mtia_or_a100():
    # For MTIA or A100, use tanh as a fallback
    @triton.jit  # pragma: no cover
    def tanh_approx_fp32(x):
        return tanh(x)

    @triton.jit  # pragma: no cover
    def tanh_approx_bf16(x):
        x32 = x.to(tl.float32)
        y = tanh(x32)
        return y.to(tl.bfloat16)

    @triton.jit  # pragma: no cover
    def sigmoid_approx_fp32(x):
        return tl.sigmoid(x)

    @triton.jit  # pragma: no cover
    def sigmoid_approx_bf16(x):
        x32 = x.to(tl.float32)
        y = tl.sigmoid(x32)
        return y.to(tl.bfloat16)

elif is_amd():

    @triton.jit
    def sigmoid_approx_fp32(x):
        return fast_dividef(1.0, 1.0 + fast_expf(-x))

    @triton.jit
    def sigmoid_approx_bf16(x):
        x32 = x.to(tl.float32)
        y = sigmoid_approx_fp32(x32)
        return y.to(tl.bfloat16)

    @triton.jit
    def tanh_approx_fp32(x):
        return 2 * sigmoid_approx_fp32(2 * x) - 1

    @triton.jit
    def tanh_approx_bf16(x):
        return 2 * sigmoid_approx_bf16(2 * x) - 1

else:
    # H100 and above, use inline asm
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

    @triton.jit  # pragma: no cover
    def sigmoid_approx_fp32(x):
        output = 0.5 * tanh_approx_fp32(0.5 * x) + 0.5
        return output

    @triton.jit  # pragma: no cover
    def sigmoid_approx_bf16(x):
        output = 0.5 * tanh_approx_bf16(0.5 * x) + 0.5
        # output = fast_dividef(1.0, 1.0 + fast_expf(-x))
        return output


if is_mtia():

    @triton.jit  # pragma: no cover
    def fast_gelu_grad(x: tl.tensor) -> tl.tensor:
        return libdevice.dgelu(x)

    @triton.jit
    def fast_gelu_joint(x):
        k = 0.7978845608
        tanh_out = tanh_approx_fp32(x * (k + k * 0.044715 * x * x))
        return x * 0.5 * (1 + tanh_out), tanh_out

elif is_amd():

    @triton.jit
    def fast_gelu_joint(x):
        k = 2.0 * 0.7978845608
        sigmoid_out = sigmoid_approx_fp32(x * (k + k * 0.044715 * x * x))
        tanh_out = 2 * sigmoid_out - 1  # Convert sigmoid back to tanh
        return x * sigmoid_out, tanh_out

    @triton.jit
    def fast_gelu_grad(x):
        # sig = sigmoid(2u), where u = 0.7978845608 * x * (1.0 + 0.044715 * x * x)
        k = 2.0 * 0.7978845608
        sig_out = sigmoid_approx_fp32(k * x * (1.0 + 0.044715 * x * x))
        return (
            2 * x * sig_out * (1 - sig_out) * (0.7978845608 + 0.1070322243 * x * x)
            + sig_out
        )

else:

    @triton.jit  # pragma: no cover
    def fast_gelu_grad(x):
        tanh_out = tanh_approx_fp32(0.7978845608 * x * (1.0 + 0.044715 * x * x))
        return 0.5 * x * (
            (1 - tanh_out * tanh_out) * (0.7978845608 + 0.1070322243 * x * x)
        ) + 0.5 * (1 + tanh_out)

    @triton.jit
    def fast_gelu_joint(x):
        k = 0.7978845608
        tanh_out = tanh_approx_fp32(x * (k + k * 0.044715 * x * x))
        return x * 0.5 * (1 + tanh_out), tanh_out


@triton.jit  # pragma: no cover
def fast_gelu(x):
    return fast_gelu_joint(x)[0]


@triton.jit  # pragma: no cover
def fast_gelu_bf16(x):
    return x * 0.5 * (1 + tanh_approx_bf16(0.796875 * x * (1.0 + 0.044715 * x * x)))


@triton.jit  # pragma: no cover
def fast_gelu_bf16_grad(x: tl.tensor) -> tl.tensor:
    tanh_out = tanh_approx_bf16(0.7978845608 * x * (1.0 + 0.044715 * x * x))
    return 0.5 * x * (  # pyre-ignore[7]
        (1 - tanh_out * tanh_out) * (0.7978845608 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)


@triton.jit  # pragma: no cover
def silu(x):
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
