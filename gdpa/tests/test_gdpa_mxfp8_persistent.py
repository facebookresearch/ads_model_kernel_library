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

"""MXFP8 (blockscaled) GDPA forward: the persistent schedule, with several tiles per
CTA, must be bit-identical to one tile per CTA.

Regression for softmax_step signalling MMA (bar.arrive) before its S tcgen05.ld had
completed: ptxas dropped the branch-local tcgen05.wait::ld, and MMA's scale-factor
writes corrupted S/P (inf rows), mostly under persistent timing. The race is timing
dependent, so a broken kernel fails some (not necessarily all) cases.
"""

import pytest
import torch

from ..src.interface import (
    cutedsl_generalized_dot_product_attention as gdpa,
    precompute_for_varline_load_balancing,
)

D, KV = 128, 256
# ~500 256-row tiles: several tiles per CTA in the persistent schedule
LENS = torch.randint(
    1, 1000, (200,), generator=torch.Generator().manual_seed(1)
).tolist()


@pytest.mark.parametrize("H", [2, 3])
@pytest.mark.parametrize("act", ["relu", "fast_gelu"])
def test_gdpa_mxfp8_persistent_matches_one_tile(H, act):
    from ads_mkl.ops.utils.mxfp8_utils import (
        quantize_for_varlen_attention_mxfp8 as quantize,
        quantize_for_varlen_attention_mxfp8_v as quantize_v,
    )

    torch.manual_seed(0)
    B = len(LENS)
    lq = torch.tensor(LENS)
    oq = torch.cat([torch.zeros(1, dtype=torch.int32), lq.cumsum(0).int()]).cuda()
    ok = torch.arange(0, (B + 1) * KV, KV, dtype=torch.int32, device="cuda")
    T = int(lq.sum())
    q_scale = 1.0 if act == "relu" else 0.05  # fast_gelu's Taylor terms need |s| <~ 2.5
    q = (torch.randn(T, H, D, device="cuda") * q_scale).bfloat16()
    k = (torch.randn(B * KV, H, D, device="cuda") * 0.3).bfloat16()
    v = torch.randn(B * KV, H, D, device="cuda").bfloat16()
    q8, sfq, cuq = quantize(q, oq, H)
    k8, sfk, cuk = quantize(k, ok, H)
    v8, sfv, _ = quantize_v(v, ok, H)
    kw = {
        "max_seqlen_q": max(LENS),
        "max_seqlen_k": KV,
        "activation": act,
        "sfq": sfq,
        "sfk": sfk,
        "sfv": sfv,
        "cu_seqlens_sf_q": cuq,
        "cu_seqlens_sf_k": cuk,
    }
    ref, _ = gdpa(q8, k8, v8, oq, ok, **kw)
    assert torch.isfinite(ref).all()
    tb, th, tk = precompute_for_varline_load_balancing(q8, oq, ok)
    assert tb.numel() > 2 * torch.cuda.get_device_properties(0).multi_processor_count
    for _ in range(3):
        out, _ = gdpa(
            q8,
            k8,
            v8,
            oq,
            ok,
            prefer_persistent=True,
            tile_to_batch_q=tb,
            tile_to_head_q=th,
            tile_to_block_q=tk,
            **kw,
        )
        assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
