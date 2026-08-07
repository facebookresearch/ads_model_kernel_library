# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors

import math
import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("PyTorch is not installed") from error


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def assert_close(actual, expected, *, atol, rtol) -> None:
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def has_blackwell_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 10


def skip_unless_blackwell(has_kernel: bool):
    return unittest.skipUnless(
        has_kernel and has_blackwell_gpu(),
        "Blackwell GPU or TLX kernel support not available",
    )


def make_jagged_qkv(
    *,
    seq_lens,
    kv_seq_lens=None,
    heads,
    dim,
    dtype,
    device,
    seed=0,
    requires_grad=True,
):
    """Build jagged q/k/v plus their offsets.

    Returns tensors shaped ``[total_len, heads, dim]`` and int32 offsets of
    length ``len(seq_lens) + 1``.
    """
    if kv_seq_lens is None:
        kv_seq_lens = seq_lens
    torch.manual_seed(seed)

    def _offsets(lens):
        return torch.tensor(
            [0] + list(torch.tensor(lens).cumsum(0)),
            device=device,
            dtype=torch.int32,
        )

    q_offsets = _offsets(seq_lens)
    kv_offsets = _offsets(kv_seq_lens)

    def _rand(total):
        t = torch.randn(total, heads, dim, device=device, dtype=dtype)
        t.requires_grad_(requires_grad)
        return t

    q = _rand(sum(seq_lens))
    k = _rand(sum(kv_seq_lens))
    v = _rand(sum(kv_seq_lens))
    return q, k, v, q_offsets, kv_offsets


def jagged_attention_reference(
    q,
    k,
    v,
    q_offsets,
    kv_offsets,
    *,
    sm_scale=None,
    window_size=None,
):
    """Per-sequence dense attention reference in float32.

    Loops over sequences so no padding is involved, which keeps the reference
    free of the masking subtleties the kernel handles internally.
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    q_off = q_offsets.tolist()
    kv_off = kv_offsets.tolist()
    outputs = []
    for i in range(len(q_off) - 1):
        q_i = q[q_off[i] : q_off[i + 1]].float().transpose(0, 1)
        k_i = k[kv_off[i] : kv_off[i + 1]].float().transpose(0, 1)
        v_i = v[kv_off[i] : kv_off[i + 1]].float().transpose(0, 1)

        scores = (q_i @ k_i.transpose(-2, -1)) * sm_scale
        if window_size is not None:
            m = q_i.shape[-2]
            n = k_i.shape[-2]
            rows = torch.arange(m, device=q.device)[:, None]
            cols = torch.arange(n, device=q.device)[None, :]
            scores = scores.masked_fill(
                (rows - cols).abs() > window_size, float("-inf")
            )
        probs = torch.softmax(scores, dim=-1)
        outputs.append((probs @ v_i).transpose(0, 1))

    return torch.cat(outputs, dim=0).to(q.dtype)
