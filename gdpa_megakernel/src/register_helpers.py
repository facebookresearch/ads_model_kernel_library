# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-ignore-all-errors
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
