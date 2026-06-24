# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors

import os
import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("PyTorch is not installed") from error


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("ADS_MKL_DISABLE_AUTOTUNE", "1")


def assert_close(actual, expected, *, atol, rtol) -> None:
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def has_blackwell_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 10


def skip_unless_blackwell(has_kernel: bool):
    return unittest.skipUnless(
        has_kernel and has_blackwell_gpu(),
        "Blackwell GPU or TLX kernel support not available",
    )
