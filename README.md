# Ads Model Kernel Library

High-performance GPU kernels for Meta Ads Recommendation Systems, developed by Meta Ads AI. This library provides optimized GPU kernel implementations that have been published.

## Projects

More coming soon!

| Project | Description | Architecture | Path |
|---------|-------------|--------------|------|
| [GDPA](gdpa/) | Generalized Dot Product Attention kernels | Blackwell (SM100) | `gdpa/` |

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- NVIDIA GPU with Hopper (SM90) or Blackwell (SM100) architecture
- CUDA >= 12.0
- [nvidia-cutlass-dsl](https://github.com/NVIDIA/cutlass) >= 4.1.0

## Installation

```bash
pip install nvidia-cutlass-dsl>=4.1.0 torch einops
```

## Quick Start

See individual project READMEs for detailed usage:

- [GDPA Quick Start](gdpa/README.md#quick-start)

## Contributors

**Meta Ads AI:** Jiaqi Xu, Chao Chen, Hongtao Yu, Dev Shanker, Jacky Zhou, Han Xu, Jake Siso, Xiaoyi Liu, Huayu Li, Markus Hoehnerbach, Manman Ren

More contributors will be added as we publish more kernels.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
