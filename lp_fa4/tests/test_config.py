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

import os
import unittest
from unittest import mock

from lp_fa4 import (
    __version__,
    DEFAULT_FP16_DQ_ACCUM_SCALE_LOG2,
    FP16_DQ_ACCUM_SCALE_LOG2_ENV,
    LpFa4Config,
)


class LpFa4ConfigTest(unittest.TestCase):
    def test_source_version_matches_distribution_metadata(self) -> None:
        self.assertEqual("0.1.0.dev0", __version__)

    def test_public_default_is_two_to_the_twelfth(self) -> None:
        config = LpFa4Config()

        self.assertEqual(12, DEFAULT_FP16_DQ_ACCUM_SCALE_LOG2)
        self.assertEqual(12, config.fp16_dq_accum_scale_log2)
        self.assertEqual(4096.0, config.fp16_dq_accum_scale)
        self.assertEqual(config, LpFa4Config.from_env(environ={}))

    def test_environment_override_is_resolved_once(self) -> None:
        with mock.patch.dict(
            os.environ,
            {FP16_DQ_ACCUM_SCALE_LOG2_ENV: "11"},
            clear=True,
        ):
            config = LpFa4Config.from_env()

        self.assertEqual(11, config.fp16_dq_accum_scale_log2)
        self.assertEqual(2048.0, config.fp16_dq_accum_scale)

    def test_explicit_value_takes_precedence_over_environment(self) -> None:
        config = LpFa4Config.from_env(
            fp16_dq_accum_scale_log2=13,
            environ={FP16_DQ_ACCUM_SCALE_LOG2_ENV: "7"},
        )

        self.assertEqual(13, config.fp16_dq_accum_scale_log2)
        self.assertEqual(8192.0, config.fp16_dq_accum_scale)

    def test_invalid_explicit_values_are_rejected(self) -> None:
        for value in (-1, 16):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    LpFa4Config(fp16_dq_accum_scale_log2=value)
        with self.assertRaises(TypeError):
            LpFa4Config(fp16_dq_accum_scale_log2=True)

    def test_invalid_environment_values_are_rejected(self) -> None:
        for raw_value in ("", "12.5", "-1", "16"):
            with self.subTest(raw_value=raw_value):
                with self.assertRaises(ValueError):
                    LpFa4Config.from_env(
                        environ={FP16_DQ_ACCUM_SCALE_LOG2_ENV: raw_value}
                    )
