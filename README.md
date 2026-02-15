# GDPA: Generalized Dot Product Attention

A high-performance Flash Attention implementation using NVIDIA's CUTE-DSL (CUDA Template Engine) for Hopper and Blackwell GPU architectures.

## Overview

GDPA provides optimized GPU kernels for scaled dot-product attention, featuring:

- **Supported Datatypes**: BF16, FP16, and FP8
- **Head Configurations**: Multi-Head Attention (MHA)
- **Head Dimensions**: 64, 96, 128, 192, 256
- **Variable Length Sequences**: Support for batched sequences with different lengths
- **Paged KV Cache**: Memory-efficient inference with paged key-value caching

## Requirements

- Python >= 3.12
- PyTorch
- NVIDIA CUTLASS-DSL >= 4.1.0
- NVIDIA GPU with Hopper (SM90) or Blackwell (SM100) architecture
- CUDA 12.x

## Installation

```bash
pip install nvidia-cutlass-dsl>=4.2.1 torch einops
```

## Quick Start

```python
import torch
from gdpa.src.interface import flash_attn_func

# Create input tensors
batch_size, seq_len, num_heads, head_dim = 2, 1024, 32, 128
q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")

# Run flash attention
output = flash_attn_func(q, k, v)
```

## Features

### Optimized for recommendation models

- Optimize Gelu activation function with fast approximation
- Flatten inner loop to increase pipeline efficiency

### FP8 Support

- Native FP8 (E4M3) computation for maximum throughput
- Automatic scaling and descaling
- Compatible with transformer engine workflows

## Architecture Support

| Feature | Blackwell (SM100) |
|---------|-------------------|
| BF16/FP16 Forward |✓ |
| BF16/FP16 Backward | ✓ |

## Benchmarks

Run the benchmark suite:

```bash
python tests/benchmark_attn.py
python tests/benchmark_attn_fp8.py
```

## Testing

```bash
pytest tests/test_flash_attn.py
```

## License

Apache License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This implementation builds upon the Flash Attention algorithm by Tri Dao et al. and leverages NVIDIA's CUTLASS library for high-performance GPU kernels.

## References

- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)
