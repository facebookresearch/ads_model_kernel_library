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

import logging
import os
from functools import lru_cache
from typing import Any

import torch
import triton

DUMP_KERNEL_INFO_ENV_VAR = "GDPA_DUMP_KERNEL_INFO"


def dump_kernel_info(kernel_info: Any) -> None:
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


def get_autotune_kernel(kernel: Any, autotune_configs: list[Any], **kwargs: Any) -> Any:
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


def generate_jagged_data(
    max_len: int = 200,
    B: int = 1536,
    H: int = 2,
    dim: int = 256,
    dff: int = 256,
    sparsity: float = 0.0,
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    torch.set_default_device("cuda")
    # q is jagged tensor with shape (sum(seq), H, dim)
    # K and V are dense tensor with shape (B*dff, H, dim)
    # generate the lengths for each example
    low = 0
    high = 1
    if sparsity >= 0.5:
        low = int((sparsity * 2 - 1) * max_len)
        high = max_len
    else:
        low = 1
        high = int(sparsity * 2 * max_len)
    # Avoid case of q length to be 0 as we assume this won't be a valid case
    low = max(low, 1)

    lengths = torch.randint(low=low, high=high + 1, size=(B,))
    q_offsets = torch.cat([torch.zeros((1,)), torch.cumsum(lengths, dim=-1)]).to(
        torch.int
    )

    # generate qkv
    Q = torch.randn(
        int(q_offsets[-1].item()),
        H,
        dim,
        dtype=dtype,
        requires_grad=requires_grad,
    ).contiguous()

    K = torch.randn(
        B * dff,
        H,
        dim,
        dtype=dtype,
        requires_grad=requires_grad,
    ).contiguous()
    V = torch.randn(
        B * dff,
        H,
        dim,
        dtype=dtype,
        requires_grad=requires_grad,
    ).contiguous()

    kv_offsets = (
        torch.arange(
            B + 1,
            dtype=torch.int,
        )
        * dff
    )

    return Q, K, V, q_offsets, kv_offsets, lengths, max_len
