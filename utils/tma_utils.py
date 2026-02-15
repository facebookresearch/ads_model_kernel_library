# pyre-unsafe
import torch
import triton


class TmaAutoTuneHelper:
    # duck typing wrapper to implement the same interface as TmaDescKernelParam in Triton PR #4498
    class KernelParamWrapper:
        def __init__(self, desc):
            self.desc = desc

        def tma_desc_cpu_ptr(self):
            return self.desc.data_ptr()

    TMA_SIZE = 128

    def __init__(self):
        self.fill_1d_tma_descriptor_inner = (
            triton.runtime.driver.active.utils.fill_1d_tma_descriptor
        )
        self.fill_2d_tma_descriptor_inner = (
            triton.runtime.driver.active.utils.fill_2d_tma_descriptor
        )
        self.descriptors = {}

    # Call this method outside of the lambda function for grid size
    def init_tma_descriptor(self, name):
        self.descriptors[name] = torch.empty(
            TmaAutoTuneHelper.TMA_SIZE, device="cpu", dtype=torch.int8
        )

    # Call this method inside the lambda function for grid size
    def fill_1d_tma_descriptor(self, name, ptr, dim, block_dim, element_size):
        desc_x = self.descriptors[name]
        assert desc_x.data_ptr() % 64 == 0
        self.fill_1d_tma_descriptor_inner(
            ptr, dim, block_dim, element_size, desc_x.data_ptr()
        )

    # Call this method inside the lambda function for grid size
    def fill_2d_tma_descriptor(
        self, name, ptr, dim1, dim0, block_dim1, block_dim0, element_size
    ):
        desc_x = self.descriptors[name]
        assert desc_x.data_ptr() % 64 == 0
        self.fill_2d_tma_descriptor_inner(
            ptr, dim1, dim0, block_dim1, block_dim0, element_size, desc_x.data_ptr()
        )

    def get_tma_descriptor_kernel_param(self, name):
        assert self.descriptors[name] is not None
        return self.KernelParamWrapper(self.descriptors[name])


def is_tma_supported():
    try:
        return (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 9
            and torch.version.cuda >= "12.4"
        )
    except:
        return False
