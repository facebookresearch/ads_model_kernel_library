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


import torch


def rotate_every_two_jagged(x: torch.Tensor) -> torch.Tensor:
    b_n, h, d = x.shape
    x = x.view(b_n, h, d // 2, 2)
    x1, x2 = x.chunk(2, dim=-1)
    x = torch.cat((-x2, x1), dim=-1)
    return x.view(b_n, h, d)


def apply_rotary_pos_emb_jagged(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sinu_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b_n, d = sinu_pos.shape
    pos = sinu_pos.view(-1, 2, d // 2)
    sin, cos = pos.chunk(2, dim=-2)

    # Use interleaved repeat for better Triton efficiency.
    sin = sin.view(-1, 1, d // 2, 1).repeat(1, 1, 1, 2).view(-1, 1, d)
    cos = cos.view(-1, 1, d // 2, 1).repeat(1, 1, 1, 2).view(-1, 1, d)

    q = (q * cos) + (rotate_every_two_jagged(q) * sin)
    k = (k * cos) + (rotate_every_two_jagged(k) * sin)
    v = (v * cos) + (rotate_every_two_jagged(v) * sin)
    return q, k, v
