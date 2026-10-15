# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
import sys
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("PyTorch is not installed") from error


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("ADS_MKL_DISABLE_AUTOTUNE", "1")

_FT = TypeVar("_FT", bound=Callable[..., object])


def assert_close(
    actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float
) -> None:
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def has_blackwell_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 10


def skip_unless_blackwell(has_kernel: bool) -> Callable[[_FT], _FT]:
    return unittest.skipUnless(
        has_kernel and has_blackwell_gpu(),
        "Blackwell GPU or TLX kernel support not available",
    )
