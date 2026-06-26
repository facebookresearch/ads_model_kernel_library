# Ads Model Kernel Library

High-performance GPU kernels for Meta Ads Recommendation Systems, developed by Meta Ads AI. This library provides optimized GPU kernel implementations that have been published.

## Projects

| Project | Description | Architecture | Path |
|---------|-------------|--------------|------|
| [GDPA](gdpa/) | Generalized Dot Product Attention kernels | Blackwell (SM100) | `gdpa/` |
| [TLX Block Attention](block_attention/) | Triton TLX block attention kernels | Blackwell (SM100) | `block_attention/` |
| [TLX Multi-CTA Norm Fusion](multi_cta_norm_fusion/) | Triton TLX fused matmul with RMSNorm and LayerNorm kernels | Blackwell (SM100) | `multi_cta_norm_fusion/` |
| [TLX GDPA Megakernel](gdpa_megakernel/) | Triton TLX generalized dot product attention megakernel | Blackwell (SM100) | `gdpa_megakernel/` |

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- NVIDIA GPU with Hopper (SM90) or Blackwell (SM100) architecture
- CUDA >= 12.0
- [nvidia-cutlass-dsl](https://github.com/NVIDIA/cutlass) >= 4.1.0
- `fbtriton==3.6.1` for TLX packages

## Installation

For CuteDSL GDPA:

```bash
pip install nvidia-cutlass-dsl>=4.1.0 torch einops
```

For TLX packages, see the project-specific `environment.yml` files or install the matching Triton build:

```bash
pip install fbtriton==3.6.1
```

## Quick Start

See individual project READMEs for detailed usage:

- [GDPA Quick Start](gdpa/README.md#quick-start)
- [TLX Block Attention Quick Start](block_attention/README.md#quick-start)
- [TLX Multi-CTA Norm Fusion](multi_cta_norm_fusion/README.md)
- [TLX GDPA Megakernel](gdpa_megakernel/README.md)

## Contributors

**Meta Ads AI:** Jiaqi Xu, Chao Chen, Hongtao Yu, Dev Shanker, Junqing (Jacky) Zhou, Han Xu, Jake Siso, Xiaoyi Liu, Huayu Li, Markus Hoehnerbach, Manman Ren

More contributors will be added as we publish more kernels.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
