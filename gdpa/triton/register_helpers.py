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


# workaround for duplicate operator implementation issue due to torch package
def custom_triton_op(qualname, mutates_args):
    def wrapper(func):
        try:
            op_exists = torch._C._dispatch_has_kernel_for_dispatch_key(qualname, "CUDA")
        except Exception:
            op_exists = False

        if op_exists is False:
            return torch._library.triton_op(qualname, func, mutates_args=mutates_args)
        else:
            return func

    return wrapper


# workaround for duplicate operator implementation issue due to torch package
def custom_register_kernel(qualname, device_types):
    def wrapper(func):
        try:
            op_exists = torch._C._dispatch_has_kernel_for_dispatch_key(qualname, "CPU")
        except Exception:
            op_exists = False

        if op_exists is False:
            return torch.library.register_kernel(qualname, device_types, func)
        else:
            return func

    return wrapper
