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

import importlib
import importlib.util
import inspect
import site
from functools import cache
from pathlib import Path
from typing import Any, Callable, cast

from torch._guards import active_fake_mode


@cache
def activate_cutlass_wheel_path() -> None:
    if importlib.util.find_spec("cutlass") is not None:
        return
    spec = importlib.util.find_spec("nvidia_cutlass_dsl")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("nvidia-cutlass-dsl is not installed")
    package_root = Path(next(iter(spec.submodule_search_locations))).parent
    # Buck PARs do not process the wheel's path configuration automatically.
    site.addsitedir(str(package_root))


@cache
def activate_tvm_ffi_compat() -> None:
    kwargs_wrapper = importlib.import_module("tvm_ffi.utils.kwargs_wrapper")
    original = cast(
        Callable[..., Any],
        kwargs_wrapper.__dict__["make_kwargs_wrapper"],
    )
    if "map_dataclass_to_tuple" in inspect.signature(original).parameters:
        return

    def make_kwargs_wrapper(
        *args: Any,
        **kwargs: Any,
    ) -> Callable[..., Any]:
        dataclass_args = kwargs.pop("map_dataclass_to_tuple", ())
        if dataclass_args:
            raise RuntimeError("LP-FA4 requires TVM-FFI dataclass argument support")
        return original(*args, **kwargs)

    kwargs_wrapper.__dict__["make_kwargs_wrapper"] = make_kwargs_wrapper


def is_fake_mode() -> bool:
    return active_fake_mode() is not None
