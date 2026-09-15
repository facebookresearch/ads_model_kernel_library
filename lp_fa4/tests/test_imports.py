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

import unittest

from lp_fa4.cute import (
    flash_attn_func,
    flash_attn_varlen_func,
    mxfp8_flash_attn_varlen_backward,
    mxfp8_flash_attn_varlen_forward,
    Mxfp8VarlenMeta,
    Mxfp8VarlenTensor,
    quantize_mxfp8_varlen,
)


class LpFa4ImportTest(unittest.TestCase):
    def test_public_attention_entry_points_are_importable(self) -> None:
        self.assertTrue(callable(flash_attn_func))
        self.assertTrue(callable(flash_attn_varlen_func))
        self.assertTrue(callable(Mxfp8VarlenMeta))
        self.assertTrue(callable(Mxfp8VarlenTensor))
        self.assertTrue(callable(quantize_mxfp8_varlen))
        self.assertTrue(callable(mxfp8_flash_attn_varlen_forward))
        self.assertTrue(callable(mxfp8_flash_attn_varlen_backward))

    def test_mxfp8_metadata_pads_each_sequence_independently(self) -> None:
        metadata = Mxfp8VarlenMeta.from_lengths(
            [17, 0, 129],
            device="cpu",
        )

        self.assertEqual([0, 17, 17, 146], metadata.cu_seqlens.tolist())
        self.assertEqual([0, 128, 128, 384], metadata.cu_seqlens_sf.tolist())
        self.assertEqual(129, metadata.max_seqlen)
        self.assertEqual(3, metadata.batch_size)
