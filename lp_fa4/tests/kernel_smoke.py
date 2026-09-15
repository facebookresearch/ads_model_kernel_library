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

from __future__ import annotations

import torch
import torch.nn.functional as F
from lp_fa4.cute import flash_attn_func


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major < 10:
        raise RuntimeError(
            f"LP-FA4 requires a Blackwell GPU; found compute capability {major}.{minor}"
        )

    torch.manual_seed(0)
    shape = (1, 256, 16, 128)
    q, k, v = (
        torch.randn(
            shape,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        for _ in range(3)
    )

    out, _lse = flash_attn_func(q, k, v)
    reference = F.scaled_dot_product_attention(
        q.detach().float().transpose(1, 2),
        k.detach().float().transpose(1, 2),
        v.detach().float().transpose(1, 2),
    ).transpose(1, 2)
    torch.testing.assert_close(out.float(), reference, atol=3e-2, rtol=3e-2)
    out.backward(torch.randn_like(out))
    torch.cuda.synchronize()

    tensors = {"out": out, "dq": q.grad, "dk": k.grad, "dv": v.grad}
    for name, tensor in tensors.items():
        if tensor is None or not torch.isfinite(tensor).all():
            raise RuntimeError(f"{name} is missing or contains a non-finite value")

    device_name = torch.cuda.get_device_name()
    print(f"LP-FA4 BF16 forward/backward smoke passed on {device_name}: {shape=}")


if __name__ == "__main__":
    main()
