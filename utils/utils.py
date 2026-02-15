# pyre-unsafe
import logging
import os
from functools import lru_cache

import torch
import triton

DUMP_KERNEL_INFO_ENV_VAR = "ADS_MKL_DUMP_KERNEL_INFO"


def dump_kernel_info(kernel_info):
    """
    Dump ttir, ttgir, llir, ptx to text files for analysis
    """
    if os.environ.get(DUMP_KERNEL_INFO_ENV_VAR, "0") == "1":
        logging.info(f"{kernel_info.metadata=}")
        logging.info(f"{kernel_info.n_spills=} {kernel_info.n_regs=}")

        # Dump all IR for NVIDIA GPUs.
        for ir in ["ttir", "ttgir", "llir", "ptx"]:
            kname = kernel_info.metadata.name
            logging.info(f"Dumping kname: {kname}")
            with open(f"{kname}_{ir}.txt", "w") as f:
                f.write(kernel_info.asm[ir])


def get_autotune_kernel(kernel, autotune_configs, **kwargs):
    return triton.autotune(configs=autotune_configs, **kwargs)(kernel)


def get_num_warps() -> int:
    if torch.version.hip:
        # AMD has 64 threads per warp vs NVDIA has 32 threads per warp.
        # And the max threads per block is 1024 for both hardware. So for AMD num_warps = 1024/64 = 16.
        return 16
    else:
        return 32


@lru_cache
def get_num_sms() -> int | None:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties("cuda").multi_processor_count


def should_use_i64_idx(*tensors: torch.Tensor) -> bool:
    return any(isinstance(t, torch.Tensor) and t.numel() >= 2**31 for t in tensors)
