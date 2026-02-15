# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""
This file contains hardware-specific config for Triton kernels
"""

from typing import List, Optional, Set, Tuple, Union

import torch
import triton


def is_amd() -> bool:
    try:
        result = triton.runtime.driver.active.get_current_target().backend == "hip"
        return result
    except RuntimeError:
        return False


def is_mtia() -> bool:
    try:
        result = triton.runtime.driver.active.get_current_target().backend == "mtia"
        return result
    except RuntimeError:
        return False


def _max_block_m_amd() -> int:
    return 64


def _max_block_n_amd() -> int:
    return 64


def _max_block_k_amd() -> int:
    return 32


def _max_group_m_amd() -> int:
    return 8  # another option is 32, 4


def _max_stages_amd() -> int:
    return 1


def _max_warps_amd() -> int:
    return 4  # another option is 2


def _cap_list(
    x: List[int], cap_value: int, not_allowed_values: Optional[Set[int]] = None
) -> List[int]:
    result = []
    for element in x:
        if element > cap_value:
            continue
        if not_allowed_values is not None and element in not_allowed_values:
            continue
        result.append(element)
    if len(result) == 0:
        # Guarantee at least one element
        result.append(cap_value)
    return result


def _cap(
    x: Union[List[int], Tuple[int, ...], int],
    cap_value: int,
    not_allowed_values: Optional[Set[int]] = None,
) -> Union[List[int], Tuple[int, ...], int]:
    if not is_amd():
        return x

    if isinstance(x, list):
        return _cap_list(x, cap_value)

    if isinstance(x, tuple):
        result = _cap_list(list(x), cap_value)
        return tuple(result)

    assert isinstance(x, int)
    if not_allowed_values is not None and x in not_allowed_values:
        return cap_value
    else:
        return min(x, cap_value)


@torch.fx.wrap
def block_m_hw_supported(
    x: Union[List[int], Tuple[int, ...], int],
) -> Union[List[int], Tuple[int, ...], int]:
    return _cap(x, _max_block_m_amd())


@torch.fx.wrap
def block_n_hw_supported(
    x: Union[List[int], Tuple[int, ...], int],
) -> Union[List[int], Tuple[int, ...], int]:
    return _cap(x, _max_block_n_amd())


@torch.fx.wrap
def block_k_hw_supported(
    x: Union[List[int], Tuple[int, ...], int],
) -> Union[List[int], Tuple[int, ...], int]:
    return _cap(x, _max_block_k_amd())


@torch.fx.wrap
def group_m_hw_supported(
    x: Union[List[int], Tuple[int, ...], int],
) -> Union[List[int], Tuple[int, ...], int]:
    return _cap(x, _max_group_m_amd())


@torch.fx.wrap
def stages_hw_supported(
    x: Union[List[int], Tuple[int, ...], int],
) -> Union[List[int], Tuple[int, ...], int]:
    not_allowed_values = {0}  # not sure if num_stages=0 is allowed on AMD.
    return _cap(x, _max_stages_amd(), not_allowed_values)


@torch.fx.wrap
def warps_hw_supported(
    x: Union[List[int], Tuple[int, ...], int],
) -> Union[List[int], Tuple[int, ...], int]:
    return _cap(x, _max_warps_amd())


@torch.fx.wrap
def block_dot_hw_supported(x: int) -> int:
    return max(64, x) if is_mtia() else x
