# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors

"""Accuracy tests for the TLX GDPA megakernel on B200."""

import unittest

from oss_test_utils import assert_close, skip_unless_blackwell, torch

try:
    from tlx_gdpa_megakernel import tlx_gdpa_megakernel
    from tlx_math import get_pytorch_activation

    HAS_TLX_KERNEL = True
except Exception:
    HAS_TLX_KERNEL = False


def _make_dense_jagged_inputs(*, batch, seqlen, heads, dim, dtype, device):
    total = batch * seqlen
    q = torch.randn(total, heads, dim, device=device, dtype=dtype)
    k = torch.randn(total, heads, dim, device=device, dtype=dtype)
    v = torch.randn(total, heads, dim, device=device, dtype=dtype)
    offsets = torch.arange(
        0,
        total + 1,
        seqlen,
        device=device,
        dtype=torch.int32,
    )
    return q, k, v, offsets


def _gdpa_reference(q, k, v, offsets, activation):
    act = get_pytorch_activation(activation)
    out = torch.empty_like(q)
    for batch_id in range(offsets.numel() - 1):
        start = offsets[batch_id].item()
        end = offsets[batch_id + 1].item()
        q_i = q[start:end].float().transpose(0, 1)
        k_i = k[start:end].float().transpose(0, 1)
        v_i = v[start:end].float().transpose(0, 1)
        scores = q_i @ k_i.transpose(-2, -1)
        probs = act(scores)
        out[start:end] = (probs @ v_i).transpose(0, 1).to(q.dtype)
    return out


class TLXGDPAMegakernelTest(unittest.TestCase):
    @skip_unless_blackwell(HAS_TLX_KERNEL)
    def test_fused_rmsnorm_matches_reference(self):
        device = "cuda"
        dtype = torch.bfloat16
        activation = "fast_gelu"
        torch.manual_seed(42)

        q, k, v, offsets = _make_dense_jagged_inputs(
            batch=2,
            seqlen=128,
            heads=2,
            dim=128,
            dtype=dtype,
            device=device,
        )

        gdpa_ref = _gdpa_reference(q, k, v, offsets, activation)
        ref = (
            torch.nn.functional.rms_norm(
                gdpa_ref.view(gdpa_ref.shape[0], -1).float(),
                (q.shape[1] * q.shape[2],),
                eps=1e-5,
            )
            .to(dtype)
            .view_as(gdpa_ref)
        )
        out = tlx_gdpa_megakernel(
            query=q,
            key=k,
            value=v,
            query_offset=offsets,
            key_offset=offsets,
            max_seq_len_q=128,
            max_seq_len_kv=128,
            activation=activation,
            enable_tma=True,
            enable_ws=True,
            fused_rms_norm=True,
        )

        assert_close(out, ref, atol=3e-2, rtol=3e-2)

    @skip_unless_blackwell(HAS_TLX_KERNEL)
    def test_fused_layernorm_and_rmsnorm_matches_reference(self):
        device = "cuda"
        dtype = torch.bfloat16
        activation = "fast_gelu"
        torch.manual_seed(42)

        q, k, v, offsets = _make_dense_jagged_inputs(
            batch=2,
            seqlen=128,
            heads=2,
            dim=128,
            dtype=dtype,
            device=device,
        )
        normalized_shape = (q.shape[1] * q.shape[2],)

        ln_q = (
            torch.nn.functional.layer_norm(
                q.view(q.shape[0], -1).float(),
                normalized_shape,
                eps=1e-5,
            )
            .to(dtype)
            .view_as(q)
        )
        gdpa_ref = _gdpa_reference(ln_q, k, v, offsets, activation)
        ref = (
            torch.nn.functional.rms_norm(
                (gdpa_ref + ln_q).view(gdpa_ref.shape[0], -1).float(),
                normalized_shape,
                eps=1e-5,
            )
            .to(dtype)
            .view_as(gdpa_ref)
        )
        out = tlx_gdpa_megakernel(
            query=q,
            key=k,
            value=v,
            query_offset=offsets,
            key_offset=offsets,
            max_seq_len_q=128,
            max_seq_len_kv=128,
            activation=activation,
            enable_tma=True,
            enable_ws=True,
            fused_layernorm=True,
            fused_q_residual_add=True,
            fused_rms_norm=True,
        )

        assert_close(out, ref, atol=3e-2, rtol=3e-2)


if __name__ == "__main__":
    unittest.main()
