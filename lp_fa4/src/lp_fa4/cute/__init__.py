# @nolint
"""Flash Attention CUTE (CUDA Template Engine) implementation."""

from .runtime import activate_cutlass_wheel_path, activate_tvm_ffi_compat


activate_cutlass_wheel_path()
activate_tvm_ffi_compat()

from .interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)
from .mxfp8 import (
    Mxfp8VarlenMeta,
    Mxfp8VarlenTensor,
    mxfp8_flash_attn_varlen_backward,
    mxfp8_flash_attn_varlen_forward,
    quantize_mxfp8_varlen,
)

__all__ = [
    "Mxfp8VarlenMeta",
    "Mxfp8VarlenTensor",
    "flash_attn_func",
    "flash_attn_varlen_func",
    "mxfp8_flash_attn_varlen_backward",
    "mxfp8_flash_attn_varlen_forward",
    "quantize_mxfp8_varlen",
]
