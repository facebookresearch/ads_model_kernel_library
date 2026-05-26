# TLX Block Attention

Triton TLX-based block attention kernels for variable-length attention workloads.

## Layout

- `triton/` - Triton TLX kernel implementation
- `tests/` - project tests

## Quick Start

```python
from ads_mkl.ops.oss.block_attention.triton.tlx_block_attention import (
    block_attention_api,
)
```

## Run

```bash
buck2 test fbcode//ads_mkl/ops/oss/block_attention:test_tlx_block_attention
```

This test target includes:
- `fbcode/ads_mkl/ops/oss/block_attention/triton/tests/test_tlx_block_attention.py`
- `fbcode/ads_mkl/ops/oss/block_attention/tests/block_attention_test.py`
