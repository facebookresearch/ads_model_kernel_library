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
import sys
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


class TLXJaggedFlashAttentionBetaTest(unittest.TestCase):
    @skip_unless_blackwell(HAS_TLX_KERNEL)
    def test_pma_broadcast_q_jagged_kv(self):
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
    def test_sliding_window(self):
        """Banded attention. Windowed backwards are gated to the 1-CTA kernel."""
        self._run_vs_pytorch(seq_lens=[128, 37, 200], window_size=32)

    @skip_unless_blackwell(HAS_TLX_KERNEL)
    def test_dense_non_broadcast(self):
        """Uniform sequence lengths, no broadcast.

        Length 128 for every sequence gives stats offsets 0/128/256/384, all
        4-element aligned, so the backward routes to the 2-CTA kernel.
        """
        self._run_vs_pytorch(seq_lens=[128, 128, 128], window_size=None)

    def _run_vs_pytorch(self, *, seq_lens, window_size):
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

        def _leaves():
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
