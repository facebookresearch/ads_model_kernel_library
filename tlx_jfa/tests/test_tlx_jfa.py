# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors

"""Accuracy tests for the TLX jagged flash attention kernel on B200.

Three scenarios, each covering forward and backward against a pure-PyTorch
reference:

1. PMA -- broadcast_q with jagged key/value lengths. Routes to the 2-CTA
   collaborative-MMA backward.
2. Sliding window. Gated to the 1-CTA backward.
3. Dense non-broadcast -- uniform sequence lengths. Routes to the 2-CTA
   backward, since uniform length L gives 4-element-aligned stats offsets.
"""

import math
import os
import subprocess
import sys
import textwrap
import unittest

# Import as a plain directory of modules, both when discovered directly
# (`python -m unittest discover -s tests`) and when loaded as part of a
# package, where this directory is not otherwise on sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oss_test_utils import (  # noqa: E402
    assert_close,
    jagged_attention_reference,
    make_jagged_qkv,
    skip_unless_blackwell,
    SRC_DIR,
    torch,
)

try:
    from tlx_jagged_flash_attention import tlx_jagged_flash_attention

    HAS_TLX_KERNEL = True
except Exception:
    HAS_TLX_KERNEL = False


HEADS = 2
DIM = 128
DTYPE = torch.bfloat16
FWD_TOL = {"atol": 1e-2, "rtol": 1e-2}
BWD_TOL = {"atol": 2e-2, "rtol": 2e-2}
# Only needs to tell "ran" from "hung"; includes the child's Triton compilation.
CHILD_TIMEOUT_S = 900

# Forward over query tiles whose key range is empty: key sequences of length 0, and a
# sliding window that ends before the keys of its query tile do. Such tiles must not
# hang, and their output rows must be 0 (attention over no keys).
EMPTY_KEY_RANGE_SCRIPT = """
import torch
from oss_test_utils import assert_close, jagged_attention_reference, make_jagged_qkv
from tlx_jagged_flash_attention import tlx_jagged_flash_attention

cases = (
    # (query lens, key lens, window, query rows that see no key)
    ([200, 128, 64], [0, 150, 0], None, [(0, 200), (328, 392)]),
    # Rows 256..299 of sequence 0 lie past its 100 keys plus the window.
    ([300, 40], [100, 300], 64, [(256, 300)]),
)
for q_lens, kv_lens, window, empty in cases:
    q, k, v, q_offsets, kv_offsets = make_jagged_qkv(
        seq_lens=q_lens,
        kv_seq_lens=kv_lens,
        heads=2,
        dim=128,
        dtype=torch.bfloat16,
        device="cuda",
        seed=43,
        requires_grad=False,
    )
    ref = jagged_attention_reference(q, k, v, q_offsets, kv_offsets, window_size=window)
    rows = torch.cat([torch.arange(lo, hi) for lo, hi in empty]).cuda()
    # Rows that see no key of a non-empty range are NaN in the reference.
    seen = ~ref.float().isnan().any(dim=2).any(dim=1)
    for use_cpu_offsets in (False, True):
        out, lse = tlx_jagged_flash_attention(
            query=q,
            key=k,
            value=v,
            query_offset=q_offsets,
            key_offset=kv_offsets,
            max_seq_len_q=max(q_lens),
            max_seq_len_kv=max(kv_lens),
            window_size=window,
            cpu_query_offset=q_offsets.cpu() if use_cpu_offsets else None,
            cpu_key_offset=kv_offsets.cpu() if use_cpu_offsets else None,
        )
        torch.cuda.synchronize()
        assert torch.all(out[rows] == 0) and torch.all(lse[:, rows] == 0)
        assert_close(out[seen], ref[seen], atol=1e-2, rtol=1e-2)
print("OK")
"""


# Run first in every child: floating-point torch.empty/empty_like tensors come back
# NaN-filled, so rows that a kernel leaves unwritten fail the comparisons.
CHILD_PRELUDE = """
import torch


def _nan_filled(alloc):
    def wrapped(*args, **kwargs):
        t = alloc(*args, **kwargs)
        return t.fill_(float("nan")) if t.is_floating_point() else t

    return wrapped


torch.empty = _nan_filled(torch.empty)
torch.empty_like = _nan_filled(torch.empty_like)
"""

# Backward over key tiles that no query block reaches: empty query sequences, and a
# sliding window with cross attention (qlen <= start_n - window). Such tiles must not
# hang, and their dK/dV rows must be 0 (dK and dV come from torch.empty_like).
ZERO_STEP_KEY_TILES_SCRIPT = """
import torch
from oss_test_utils import assert_close, jagged_attention_reference, make_jagged_qkv
from tlx_jagged_flash_attention import tlx_jagged_flash_attention

# Window 64 with cross attention: sequence 0 has 100 queries and 400 keys, so its
# key tiles at 256 and 384 get no query block.
q_lens, kv_lens, window = [100, 172], [400, 130], 64
q, k, v, q_offsets, kv_offsets = make_jagged_qkv(
    seq_lens=q_lens,
    kv_seq_lens=kv_lens,
    heads=2,
    dim=128,
    dtype=torch.bfloat16,
    device="cuda",
    seed=29,
    requires_grad=False,
)
grad_out = torch.randn_like(q)
refs = [t.clone().requires_grad_(True) for t in (q, k, v)]
ref = jagged_attention_reference(*refs, q_offsets, kv_offsets, window_size=window)
ref.backward(grad_out)
# Keys that no query attends to have exactly zero reference dV rows.
unreached = (refs[2].grad == 0).all(dim=2).all(dim=1)
assert unreached.any()
for use_cpu_offsets in (False, True):
    leaves = [t.clone().requires_grad_(True) for t in (q, k, v)]
    out = tlx_jagged_flash_attention(
        query=leaves[0],
        key=leaves[1],
        value=leaves[2],
        query_offset=q_offsets,
        key_offset=kv_offsets,
        max_seq_len_q=max(q_lens),
        max_seq_len_kv=max(kv_lens),
        window_size=window,
        cpu_query_offset=q_offsets.cpu() if use_cpu_offsets else None,
        cpu_key_offset=kv_offsets.cpu() if use_cpu_offsets else None,
    )[0]
    out.backward(grad_out)
    torch.cuda.synchronize()
    for leaf, ref_leaf in zip(leaves, refs):
        assert_close(leaf.grad, ref_leaf.grad, atol=2e-2, rtol=2e-2)
    for leaf in leaves[1:]:
        assert torch.all(leaf.grad[unreached] == 0)
print("OK")
"""

# Forward with max_seq_len_q = {max_seq_len_q}; the longest query sequence is 300.
SHORT_MAX_SEQ_LEN_Q_SCRIPT = """
import torch
from tlx_jagged_flash_attention import tlx_jagged_flash_attention

q_offsets = torch.tensor([0, 300, 400], dtype=torch.int32, device="cuda")
k_offsets = torch.tensor([0, 200, 277], dtype=torch.int32, device="cuda")
q, k, v = (
    torch.randn(n, 2, 128, device="cuda", dtype=torch.bfloat16) for n in (400, 277, 277)
)
tlx_jagged_flash_attention(
    query=q,
    key=k,
    value=v,
    query_offset=q_offsets,
    key_offset=k_offsets,
    max_seq_len_q={max_seq_len_q},
    max_seq_len_kv=200,
)
torch.cuda.synchronize()
"""


def _run_in_child(script: str) -> subprocess.CompletedProcess:
    """Run a test script in a fresh Python process, so that a kernel hang or a
    device-side assert (which poisons the CUDA context) fails only the child. The
    child does not inherit this process's sys.path changes, so the kernel sources
    and the test helpers are put on its path."""
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    prelude = f"import sys\nsys.path[:0] = [{str(SRC_DIR)!r}, {tests_dir!r}]\n"
    prelude += textwrap.dedent(CHILD_PRELUDE)
    try:
        return subprocess.run(
            [sys.executable, "-c", prelude + textwrap.dedent(script)],
            check=False,
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as e:
        raise AssertionError(f"the kernel did not return within {e.timeout} s") from e


class TLXJaggedFlashAttentionBetaTest(unittest.TestCase):
    @skip_unless_blackwell(HAS_TLX_KERNEL)
    def test_pma_broadcast_q_jagged_kv(self) -> None:
        """One shared query attending to jagged key/value sequences.

        The production PMA shape. Load balancing is enabled and the shared
        query length is 128, so the backward routes to the 2-CTA kernel.
        """
        kv_lens = [128, 37, 200]
        batch = len(kv_lens)
        q_len = 128
        sm_scale = 1.0 / math.sqrt(DIM)

        _, k, v, _, kv_offsets = make_jagged_qkv(
            seq_lens=kv_lens,
            heads=HEADS,
            dim=DIM,
            dtype=DTYPE,
            device="cuda",
            seed=17,
            requires_grad=False,
        )
        shared_q = torch.randn(q_len, HEADS, DIM, device="cuda", dtype=DTYPE)
        grad_out = torch.randn(batch * q_len, HEADS, DIM, device="cuda", dtype=DTYPE)

        # Reference: materialize the per-batch query copies and let autograd
        # handle the broadcast, so this does not depend on the kernel at all.
        ref_q = shared_q.repeat(batch, 1, 1).detach().requires_grad_(True)
        ref_k = k.clone().detach().requires_grad_(True)
        ref_v = v.clone().detach().requires_grad_(True)
        repeated_q_offsets = torch.arange(
            0, batch * q_len + 1, q_len, device="cuda", dtype=torch.int32
        )
        ref = jagged_attention_reference(
            ref_q, ref_k, ref_v, repeated_q_offsets, kv_offsets, sm_scale=sm_scale
        )
        ref.backward(grad_out)
        # Each batch consumed the same shared query, so its grad is the sum over
        # the per-batch slices of the repeated query's grad.
        ref_dq = ref_q.grad.view(batch, q_len, HEADS, DIM).sum(dim=0)

        opt_q = shared_q.clone().detach().requires_grad_(True)
        opt_k = k.clone().detach().requires_grad_(True)
        opt_v = v.clone().detach().requires_grad_(True)
        broadcast_q_offsets = torch.tensor([0, q_len], device="cuda", dtype=torch.int32)
        out = tlx_jagged_flash_attention(
            query=opt_q,
            key=opt_k,
            value=opt_v,
            query_offset=broadcast_q_offsets,
            key_offset=kv_offsets,
            max_seq_len_q=q_len,
            max_seq_len_kv=max(kv_lens),
            output_offset=repeated_q_offsets,
            sm_scale=sm_scale,
            broadcast_q=True,
            cpu_query_offset=broadcast_q_offsets.to("cpu"),
            cpu_key_offset=kv_offsets.to("cpu"),
        )[0]
        out.backward(grad_out)

        assert_close(out, ref, **FWD_TOL)
        assert_close(opt_q.grad, ref_dq, **BWD_TOL)
        assert_close(opt_k.grad, ref_k.grad, **BWD_TOL)
        assert_close(opt_v.grad, ref_v.grad, **BWD_TOL)

    @skip_unless_blackwell(HAS_TLX_KERNEL)
    def test_sliding_window(self) -> None:
        """Banded attention. Windowed backwards are gated to the 1-CTA kernel."""
        self._run_vs_pytorch(seq_lens=[128, 37, 200], window_size=32)

    @skip_unless_blackwell(HAS_TLX_KERNEL)
    def test_dense_non_broadcast(self) -> None:
        """Uniform sequence lengths, no broadcast.

        Length 128 for every sequence gives stats offsets 0/128/256/384, all
        4-element aligned, so the backward routes to the 2-CTA kernel.
        """
        self._run_vs_pytorch(seq_lens=[128, 128, 128], window_size=None)

    @skip_unless_blackwell(HAS_TLX_KERNEL)
    def test_empty_key_range(self) -> None:
        """Query tiles with no key in range: empty key sequences, and a sliding
        window past the end of the keys (cross attention). The load and MMA groups
        used to issue the first key tile anyway while the softmax and correction
        groups ran no iteration, so the forward hung. Such tiles are now skipped by
        every group, and their O and M rows are written as zeros."""
        completed = _run_in_child(EMPTY_KEY_RANGE_SCRIPT)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    @skip_unless_blackwell(HAS_TLX_KERNEL)
    def test_zero_step_key_tiles(self) -> None:
        """Key tiles that no query block reaches (num_steps <= 0), from a sliding
        window with cross attention: a key sequence longer than its query sequence
        plus the window. The MMA and load tasks of the 1-CTA backward used to run the
        tile's prolog while the compute task ran no step, so the backward hung. Such
        tiles are now skipped by the MMA and load tasks, and the compute task writes
        their zero dK/dV rows. Runs with and without CPU offsets (load-balanced and
        dense tile schedules)."""
        completed = _run_in_child(ZERO_STEP_KEY_TILES_SCRIPT)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    @skip_unless_blackwell(HAS_TLX_KERNEL)
    def test_rejects_short_max_seq_len_q(self) -> None:
        """The forward grid covers max_seq_len_q rows of each query sequence and the
        kernel clamps every sequence to it, so a longer sequence used to get
        uninitialized O and M rows. The forward now fails a device-side assert; the
        control with the true maximum must succeed."""
        for max_seq_len_q, rejected in ((300, False), (172, True)):
            with self.subTest(max_seq_len_q=max_seq_len_q):
                completed = _run_in_child(
                    SHORT_MAX_SEQ_LEN_Q_SCRIPT.format(max_seq_len_q=max_seq_len_q)
                )
                output = completed.stdout + completed.stderr
                if rejected:
                    self.assertNotEqual(completed.returncode, 0, output)
                    self.assertIn("device-side assert", output.lower())
                else:
                    self.assertEqual(completed.returncode, 0, output)

    def _run_vs_pytorch(self, *, seq_lens: list[int], window_size: int | None) -> None:
        sm_scale = 1.0 / math.sqrt(DIM)
        q, k, v, q_offsets, kv_offsets = make_jagged_qkv(
            seq_lens=seq_lens,
            heads=HEADS,
            dim=DIM,
            dtype=DTYPE,
            device="cuda",
            seed=23,
            requires_grad=False,
        )
        grad_out = torch.randn_like(q)

        def _leaves() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return (
                q.clone().detach().requires_grad_(True),
                k.clone().detach().requires_grad_(True),
                v.clone().detach().requires_grad_(True),
            )

        ref_q, ref_k, ref_v = _leaves()
        ref = jagged_attention_reference(
            ref_q,
            ref_k,
            ref_v,
            q_offsets,
            kv_offsets,
            sm_scale=sm_scale,
            window_size=window_size,
        )
        ref.backward(grad_out)

        opt_q, opt_k, opt_v = _leaves()
        out = tlx_jagged_flash_attention(
            query=opt_q,
            key=opt_k,
            value=opt_v,
            query_offset=q_offsets,
            key_offset=kv_offsets,
            max_seq_len_q=max(seq_lens),
            max_seq_len_kv=max(seq_lens),
            sm_scale=sm_scale,
            window_size=window_size,
            cpu_query_offset=q_offsets.to("cpu"),
            cpu_key_offset=kv_offsets.to("cpu"),
        )[0]
        out.backward(grad_out)

        assert_close(out, ref, **FWD_TOL)
        assert_close(opt_q.grad, ref_q.grad, **BWD_TOL)
        assert_close(opt_k.grad, ref_k.grad, **BWD_TOL)
        assert_close(opt_v.grad, ref_v.grad, **BWD_TOL)


if __name__ == "__main__":
    unittest.main()
