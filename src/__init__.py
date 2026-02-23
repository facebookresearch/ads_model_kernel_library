# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# @nolint

# pyre-unsafe
"""Flash Attention CUTE (CUDA Template Engine) implementation."""

from .interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)

__version__ = "0.1.0"

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]
