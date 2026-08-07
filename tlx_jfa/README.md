# TLX Jagged Flash Attention

Triton TLX jagged flash attention kernel for Blackwell, with warp-specialized
forward and backward passes over variable-length (jagged) sequences.

## Supported variants

- **Jagged self- and cross-attention.** Query and key/value sequence lengths are
  independent, each given as `[batch_size + 1]` prefix-sum offsets.
- **PMA (pooling by multi-head attention).** A single shared query sequence
  attends to per-batch jagged key/value sequences, via `broadcast_q=True`. See
  the Kunlun paper (<https://arxiv.org/abs/2602.10016>) for the definition. This
  is the case routed through the 2-CTA collaborative-MMA backward.
- **Sliding window.** `window_size=W` restricts attention to a symmetric band of
  `+/- W` positions.
- **Grouped query attention** in the forward, where the query head count is a
  multiple of the key/value head count. The backward supports a single query
  group only.

Forward and backward are both supported; backward runs through autograd.

## Layout

- `src/tlx_jagged_flash_attention.py` - public API, forward kernel, and the 2-CTA
  (cluster) collaborative-MMA backward
- `src/bwd_1cta.py` - general 1-CTA backward, used for every configuration the
  2-CTA path does not cover
- `src/kernel_common.py` - JIT helpers shared by the forward and both backwards
- `src/tlx_math.py`, `src/register_helpers.py`, `src/utils.py` - local support helpers
- `tests/` - GPU correctness tests

The 2-CTA backward is scoped to the PMA case: `broadcast_q` with
`head_dim == 128`, a single query group, no sliding window, and load balancing
enabled. The launcher routes every other shape to the 1-CTA backward
automatically, so all supported variants work regardless of which kernel runs.

## Run

```bash
conda env create -f environment.yml
conda activate tlx-jfa
TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1 python -m unittest discover -s tests -p "test_*.py"
```

If conda channel access is restricted, create the environment with any available Python 3.12 conda channel and install the Python packages with pip:

```bash
conda create -n tlx-jfa python=3.12 pip
conda activate tlx-jfa
pip install --upgrade pip setuptools wheel
pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch
pip install fbtriton==3.6.1
TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1 python -m unittest discover -s tests -p "test_*.py"
```

To verify TLX is importable:

```bash
python -c 'import triton.language.extra.tlx as tlx; print(tlx)'
```

For interactive use outside the tests, add the kernel sources to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

## Usage

`query`, `key`, and `value` are jagged tensors of shape `[total_seq_len, num_heads, head_dim]`,
where `total_seq_len` is the sum of the per-batch sequence lengths. `query_offset` and
`key_offset` are `[batch_size + 1]` int32 prefix-sum offsets delimiting each sequence.

```python
from tlx_jagged_flash_attention import jagged_flash_attention

out = jagged_flash_attention(
    query=q,                      # [total_q, H, D]
    key=k,                        # [total_kv, H, D]
    value=v,                      # [total_kv, H, D]
    query_offset=q_offsets,       # [B + 1], int32
    key_offset=kv_offsets,        # [B + 1], int32
    max_seq_len_q=max_seq_len_q,
    max_seq_len_kv=max_seq_len_kv,
    sm_scale=head_dim**-0.5,
)
```

Optional arguments:

- `window_size` - restricts attention to a symmetric band of `+/- window_size` positions.
- `broadcast_q` - PMA: shares a single query sequence across all batches. `query`
  holds that one sequence, `query_offset` is `[0, q_len]`, and `output_offset`
  gives the per-batch output slots. The gradient of the shared query accumulates
  the contributions of every batch.
- `sm_scale` - softmax scale; defaults to `1 / sqrt(head_dim)`.

Backward is supported through autograd; call `.backward()` on the output.

## Limitations

- Requires a Blackwell (SM100+) GPU.
- `head_dim` must be `<= 128`. Larger head dims need the D-tiled IKBO forward, which
  is not part of this package.
- The backward requires a single query group (query and key/value head counts equal).

## Reference

- Kunlun: <https://arxiv.org/abs/2602.10016> (PMA definition)
- A PyTorch blog post covering this kernel is in preparation.
