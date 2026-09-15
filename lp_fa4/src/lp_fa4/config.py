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

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass


FP16_DQ_ACCUM_SCALE_LOG2_ENV = "LP_FA4_BWD_FP16_DQ_ACCUM_SCALE_LOG2"
DEFAULT_FP16_DQ_ACCUM_SCALE_LOG2 = 12
MIN_FP16_DQ_ACCUM_SCALE_LOG2 = 0
MAX_FP16_DQ_ACCUM_SCALE_LOG2 = 15


def _validate_fp16_dq_accum_scale_log2(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("fp16_dq_accum_scale_log2 must be an integer")
    if not MIN_FP16_DQ_ACCUM_SCALE_LOG2 <= value <= MAX_FP16_DQ_ACCUM_SCALE_LOG2:
        raise ValueError(
            "fp16_dq_accum_scale_log2 must be in "
            f"[{MIN_FP16_DQ_ACCUM_SCALE_LOG2}, "
            f"{MAX_FP16_DQ_ACCUM_SCALE_LOG2}], got {value}"
        )
    return value


def _parse_fp16_dq_accum_scale_log2(raw_value: str) -> int:
    try:
        value = int(raw_value, 10)
    except ValueError as error:
        raise ValueError(
            f"{FP16_DQ_ACCUM_SCALE_LOG2_ENV} must be an integer, got {raw_value!r}"
        ) from error
    return _validate_fp16_dq_accum_scale_log2(value)


@dataclass(frozen=True)
class LpFa4Config:
    fp16_dq_accum_scale_log2: int = DEFAULT_FP16_DQ_ACCUM_SCALE_LOG2

    def __post_init__(self) -> None:
        _validate_fp16_dq_accum_scale_log2(self.fp16_dq_accum_scale_log2)

    @classmethod
    def from_env(
        cls,
        *,
        fp16_dq_accum_scale_log2: int | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> LpFa4Config:
        if fp16_dq_accum_scale_log2 is not None:
            return cls(fp16_dq_accum_scale_log2=fp16_dq_accum_scale_log2)
        source = os.environ if environ is None else environ
        raw_value = source.get(FP16_DQ_ACCUM_SCALE_LOG2_ENV)
        if raw_value is None:
            return cls()
        return cls(fp16_dq_accum_scale_log2=_parse_fp16_dq_accum_scale_log2(raw_value))

    @property
    def fp16_dq_accum_scale(self) -> float:
        return math.ldexp(1.0, self.fp16_dq_accum_scale_log2)
