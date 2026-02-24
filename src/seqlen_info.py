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

# pyre-ignore-all-errors
from typing import Optional

import cutlass
import cutlass.cute as cute

"""
This consolidates all the info related to sequence length. This is so that we can do all
the gmem reads once at the beginning of each tile, rather than having to repeat these reads
to compute various things like n_block_min, n_block_max, etc.
"""


class SeqlenInfo:
    """Resolves sequence offset and length for a single batch element.

    Handles three cases: fixed-length (no cu_seqlens), variable-length with cumulative
    sequence lengths, and variable-length with explicit seqused.
    """

    def __init__(
        self,
        batch_idx: cutlass.Int32,
        seqlen_static: cutlass.Int32,
        cu_seqlens: Optional[cute.Tensor] = None,
        seqused: Optional[cute.Tensor] = None,
    ):
        self.offset = (
            0 if cutlass.const_expr(cu_seqlens is None) else cu_seqlens[batch_idx]
        )
        if cutlass.const_expr(seqused is not None):
            self.seqlen = seqused[batch_idx]
        elif cutlass.const_expr(cu_seqlens is not None):
            self.seqlen = cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx]
        else:
            self.seqlen = seqlen_static


class SeqlenInfoQK:
    """Resolves sequence offsets and lengths for both Q and K sequences.

    Also resolves scale factor (SF) offsets for MXFP8 blockscaled attention, which
    require separate 128-aligned cumulative sequence lengths.
    """

    def __init__(
        self,
        batch_idx: cutlass.Int32,
        seqlen_q_static: cutlass.Int32,
        seqlen_k_static: cutlass.Int32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        # 128-aligned cu_seqlens for scale factor offsets (varlen MXFP8 only)
        mCuSeqlensSFQ: Optional[cute.Tensor] = None,
        mCuSeqlensSFK: Optional[cute.Tensor] = None,
    ):
        self.offset_q = (
            0 if cutlass.const_expr(mCuSeqlensQ is None) else mCuSeqlensQ[batch_idx]
        )
        self.offset_k = (
            0 if cutlass.const_expr(mCuSeqlensK is None) else mCuSeqlensK[batch_idx]
        )
        # SF offsets: use padded cu_seqlens if provided, otherwise use data offsets
        self.offset_sf_q = (
            self.offset_q
            if cutlass.const_expr(mCuSeqlensSFQ is None)
            else mCuSeqlensSFQ[batch_idx]
        )
        self.offset_sf_k = (
            self.offset_k
            if cutlass.const_expr(mCuSeqlensSFK is None)
            else mCuSeqlensSFK[batch_idx]
        )
        if cutlass.const_expr(mSeqUsedQ is not None):
            self.seqlen_q = mSeqUsedQ[batch_idx]
        else:
            self.seqlen_q = (
                seqlen_q_static
                if cutlass.const_expr(mCuSeqlensQ is None)
                else mCuSeqlensQ[batch_idx + 1] - self.offset_q
            )
        if cutlass.const_expr(mSeqUsedK is not None):
            self.seqlen_k = mSeqUsedK[batch_idx]
        else:
            self.seqlen_k = (
                seqlen_k_static
                if cutlass.const_expr(mCuSeqlensK is None)
                else mCuSeqlensK[batch_idx + 1] - self.offset_k
            )
        self.has_cu_seqlens_q: int = mCuSeqlensQ is not None
        self.has_cu_seqlens_k: int = mCuSeqlensK is not None
