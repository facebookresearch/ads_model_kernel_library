# LP-FA4

LP-FA4 provides low-precision FlashAttention-4 kernels for NVIDIA Blackwell
GPUs. The initial MXFP8 API supports noncausal, head-dimension-128 MHA with
dense, packed variable-length, and shared-query workloads. Dense attention is
represented by uniform packed sequence offsets.

> **Development status:** this is an unpublished release candidate. Kernel and
> benchmark interfaces may change before the first public release.

## Requirements

- Linux and Python 3.12
- An NVIDIA Blackwell GPU (SM10x for the MXFP8 path)
- CUDA 13
- PyTorch 2.13.0, CUTLASS DSL 4.6.1, and Quack 0.6.2; the complete tested
  direct-dependency set is pinned in `constraints-cu13.txt`

Create the complete development environment with Conda:

```bash
conda env create -f environment.yml
conda activate lp-fa4
python -m unittest discover -s tests -p "test_*.py"
```

Alternatively, install into an existing Python 3.12 environment:

```bash
python -m pip install -c constraints-cu13.txt ".[dev]"
python -m unittest discover -s tests -p "test_*.py"
```

## BF16 smoke test

The high-level BF16 API uses tensors in `(batch, sequence, heads, dimension)`
order:

```python
import torch

from lp_fa4.cute import flash_attn_func

q, k, v = (
    torch.randn(
        1,
        256,
        16,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    for _ in range(3)
)
out, _ = flash_attn_func(q, k, v)
out.backward(torch.randn_like(out))
```

Run the BF16 forward/backward smoke test directly from the source checkout:

```bash
CUDA_VISIBLE_DEVICES=0 python tests/kernel_smoke.py
```

## MXFP8 forward and backward

The MXFP8 API is explicit: inputs are packed `(tokens, heads, 128)` BF16 or
FP32 tensors, each sequence receives independent 128-token-padded scale
storage, and quantization happens before kernel timing. Q, K, and V retain both
hardware orientations needed across forward and backward.

```python
import torch

from lp_fa4.cute import (
    Mxfp8VarlenMeta,
    mxfp8_flash_attn_varlen_backward,
    mxfp8_flash_attn_varlen_forward,
    quantize_mxfp8_varlen,
)

seqlen, heads, dim = 2048, 16, 128
metadata = Mxfp8VarlenMeta.from_lengths([seqlen], device="cuda")
q, k, v = (
    torch.randn(seqlen, heads, dim, device="cuda", dtype=torch.bfloat16)
    for _ in range(3)
)
q_mx, k_mx, v_mx = (
    quantize_mxfp8_varlen(x, metadata) for x in (q, k, v)
)
out, lse = mxfp8_flash_attn_varlen_forward(
    q_mx,
    k_mx,
    v_mx,
    metadata,
    metadata,
)
dq, dk_mx, dv_mx, sf_dk, sf_dv = mxfp8_flash_attn_varlen_backward(
    q_mx,
    k_mx,
    v_mx,
    out,
    torch.randn_like(out),
    lse,
    metadata,
    metadata,
    output_mxfp8_dkv=True,
)
```

With `output_mxfp8_dkv=True`, backward emits E4M3 `dK` and `dV` payloads plus
E8M0 scales with one scale per 32-element block. `dQ` remains BF16. Omitting
the option retains the three-BF16-gradient compatibility result.

Quack 0.6.2's RCEIL scaling is the default. Use `scaling_mode="floor"` only
when intentionally reproducing a FLOOR-scaled input pipeline.

Run the packed variable-length MXFP8 forward/backward accuracy smoke test with:

```bash
CUDA_VISIBLE_DEVICES=0 python tests/mxfp8_smoke.py
```

## Benchmark reproduction

The benchmark quantizes once, compiles before warmup, times with CUDA events,
and emits machine-readable JSON including raw samples, padded scale offsets,
the workload, and environment metadata. Backward timing uses the fused MXFP8
`dK`/`dV` output path. The default timing protocol uses a 1-second warmup and a
5-second measurement window. Workload-defining shape arguments have no
defaults; the command below is the only bundled reference shape.

```bash
CUDA_VISIBLE_DEVICES=0 python -m lp_fa4.benchmarks.mxfp8 \
  --mode both \
  --batch-size 8 \
  --seqlen-q 2048 \
  --seqlen-k 2048 \
  --num-heads 16 \
  --seed 0 \
  --warmup-ms 1000 \
  --measure-ms 5000 \
  --scaling-mode rceil \
  --output results.json
```

## License

Meta-authored LP-FA4 code is licensed under the Apache License 2.0. The
vendored FlashAttention-4 CuTe runtime retains its BSD 3-Clause license and
attribution; see [`src/lp_fa4/cute/LICENSE`](src/lp_fa4/cute/LICENSE) and
[`src/lp_fa4/cute/AUTHORS`](src/lp_fa4/cute/AUTHORS).

## Acknowledgments

LP-FA4 builds on [FlashAttention-4](https://github.com/Dao-AILab/flash-attention)
by Tri Dao and contributors. The vendored runtime is based on upstream revision
`6a94f8b906cf5ab944385d64707f9387f3dd6be9`.
