# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import unittest

import torch
from ads_mkl.ops.cute_dsl.block_attention.triton.tlx_block_attention import (
    apply_rotary_pos_emb_jagged,
    tlx_block_attention,
)


def dense_output_to_jagged(padded: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    chunks = []
    for b in range(offsets.numel() - 1):
        s = int(offsets[b].item())
        e = int(offsets[b + 1].item())
        chunks.append(padded[b, : e - s])
    return torch.cat(chunks, dim=0)


def jagged_to_padded(jagged: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    bsz = offsets.numel() - 1
    max_len = int((offsets[1:] - offsets[:-1]).max().item())
    out = torch.zeros(
        bsz,
        max_len,
        jagged.shape[1],
        jagged.shape[2],
        dtype=jagged.dtype,
        device=jagged.device,
    )
    for b in range(bsz):
        s = int(offsets[b].item())
        e = int(offsets[b + 1].item())
        out[b, : e - s] = jagged[s:e]
    return out


def generate_jagged_data(
    B: int,
    max_seq_len: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    lengths = torch.randint(1, max_seq_len + 1, (B,), device=device)
    offsets = torch.zeros(B + 1, dtype=torch.int32, device=device)
    offsets[1:] = torch.cumsum(lengths, dim=0).to(torch.int32)
    total = int(offsets[-1].item())

    q = torch.randn(total, H, D, device=device, dtype=dtype)
    k = torch.randn(total, H, D, device=device, dtype=dtype)
    v = torch.randn(total, H, D, device=device, dtype=dtype)
    do = torch.randn(total, H, D, device=device, dtype=dtype)
    return {
        "q_weights": q,
        "k_weights": k,
        "v_weights": v,
        "do": do,
        "q_offsets": offsets,
    }


def snr_db(signal: torch.Tensor, reference: torch.Tensor, eps: float = 1e-12) -> float:
    signal_f = signal.float()
    reference_f = reference.float()
    noise = signal_f - reference_f
    power_signal = torch.sum(reference_f * reference_f)
    power_noise = torch.sum(noise * noise)
    ratio = power_signal / (power_noise + eps)
    return float(10.0 * torch.log10(ratio).item())


def pytorch_block_attention_forward_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    offsets: torch.Tensor,
    block_size: int = 64,
    scale: float | None = None,
) -> torch.Tensor:
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    qf = q.float()
    kf = k.float()
    vf = v.float()
    out = torch.zeros_like(qf)

    h = q.shape[1]
    d = q.shape[2]
    for b in range(offsets.numel() - 1):
        s = int(offsets[b].item())
        e = int(offsets[b + 1].item())
        seq_len = e - s
        if seq_len == 0:
            continue

        nb = (seq_len + block_size - 1) // block_size
        padded = nb * block_size

        qb = qf[s:e]
        kb = kf[s:e]
        vb = vf[s:e]
        if padded != seq_len:
            pad = padded - seq_len
            qb = torch.nn.functional.pad(qb, (0, 0, 0, 0, 0, pad))
            kb = torch.nn.functional.pad(kb, (0, 0, 0, 0, 0, pad))
            vb = torch.nn.functional.pad(vb, (0, 0, 0, 0, 0, pad))

        qb = qb.reshape(nb, block_size, h, d).permute(2, 0, 1, 3)
        kb = kb.reshape(nb, block_size, h, d).permute(2, 0, 1, 3)
        vb = vb.reshape(nb, block_size, h, d).permute(2, 0, 1, 3)

        scores = torch.matmul(qb, kb.transpose(-2, -1)) * scale
        if padded != seq_len:
            valid = seq_len - (nb - 1) * block_size
            idx = torch.arange(block_size, device=q.device)
            mask = torch.ones(nb, block_size, device=q.device, dtype=torch.bool)
            mask[-1] = idx < valid
            m2d = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).unsqueeze(0)
            scores = torch.where(m2d, scores, torch.full_like(scores, -1e9))

        probs = torch.softmax(scores, dim=-1)
        ob = torch.matmul(probs, vb)
        ob = ob.permute(1, 2, 0, 3).reshape(padded, h, d)
        out[s:e] = ob[:seq_len]

    return out.to(q.dtype)


def pytorch_block_attention_backward_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    offsets: torch.Tensor,
    block_size: int = 64,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    qf = q.detach().float().requires_grad_(True)
    kf = k.detach().float().requires_grad_(True)
    vf = v.detach().float().requires_grad_(True)

    out = torch.zeros_like(qf)
    H = q.shape[1]
    D = q.shape[2]

    for b in range(offsets.numel() - 1):
        s = int(offsets[b].item())
        e = int(offsets[b + 1].item())
        L = e - s
        if L == 0:
            continue
        nb = (L + block_size - 1) // block_size
        padded = nb * block_size

        qb = qf[s:e]
        kb = kf[s:e]
        vb = vf[s:e]
        if padded != L:
            pad = padded - L
            qb = torch.nn.functional.pad(qb, (0, 0, 0, 0, 0, pad))
            kb = torch.nn.functional.pad(kb, (0, 0, 0, 0, 0, pad))
            vb = torch.nn.functional.pad(vb, (0, 0, 0, 0, 0, pad))

        qb = qb.reshape(nb, block_size, H, D).permute(2, 0, 1, 3)
        kb = kb.reshape(nb, block_size, H, D).permute(2, 0, 1, 3)
        vb = vb.reshape(nb, block_size, H, D).permute(2, 0, 1, 3)

        s_ = torch.matmul(qb, kb.transpose(-2, -1)) * scale
        if padded != L:
            valid = L - (nb - 1) * block_size
            idx = torch.arange(block_size, device=q.device)
            mask = torch.ones(nb, block_size, device=q.device, dtype=torch.bool)
            mask[-1] = idx < valid
            m2d = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).unsqueeze(0)
            s_ = torch.where(m2d, s_, torch.full_like(s_, -1e9))

        p = torch.softmax(s_, dim=-1)
        ob = torch.matmul(p, vb)
        ob = ob.permute(1, 2, 0, 3).reshape(padded, H, D)
        out[s:e] = ob[:L]

    out.backward(do.float())
    assert qf.grad is not None
    assert kf.grad is not None
    assert vf.grad is not None
    return qf.grad, kf.grad, vf.grad


class BlockAttentionTest(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_tlx_block_attention_forward_and_backward_vs_fp32_ref(self) -> None:
        torch.manual_seed(123)
        device = torch.device("cuda")
        H, D = 4, 64
        data = generate_jagged_data(
            B=64,
            max_seq_len=256,
            H=H,
            D=D,
            dtype=torch.bfloat16,
            device=device,
        )

        q = data["q_weights"].clone().detach().requires_grad_(True)
        k = data["k_weights"].clone().detach().requires_grad_(True)
        v = data["v_weights"].clone().detach().requires_grad_(True)
        do = data["do"]
        offsets = data["q_offsets"]
        offsets_cpu = offsets.cpu()

        out = tlx_block_attention(q, k, v, offsets, offsets_cpu)[0]
        print("[tlx_block_attention] smoke: direct-forward-called", flush=True)
        out.backward(do)
        torch.cuda.synchronize()

        ref_dq, ref_dk, ref_dv = pytorch_block_attention_backward_fp32(
            q.detach(),
            k.detach(),
            v.detach(),
            do.detach(),
            offsets_cpu,
            block_size=64,
            scale=1.0 / math.sqrt(D),
        )

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

        torch.testing.assert_close(
            q.grad.float(), ref_dq.float().to(device), rtol=0.2, atol=0.1
        )
        torch.testing.assert_close(
            k.grad.float(), ref_dk.float().to(device), rtol=0.2, atol=0.1
        )
        torch.testing.assert_close(
            v.grad.float(), ref_dv.float().to(device), rtol=0.2, atol=0.1
        )

        snr_q = snr_db(q.grad, ref_dq.to(device))
        snr_k = snr_db(k.grad, ref_dk.to(device))
        snr_v = snr_db(v.grad, ref_dv.to(device))

        self.assertGreater(
            snr_q,
            45.0,
            msg=f"SNR(q.grad) too low: {snr_q:.2f} dB (expected > 45.0 dB)",
        )
        self.assertGreater(
            snr_k,
            45.0,
            msg=f"SNR(k.grad) too low: {snr_k:.2f} dB (expected > 45.0 dB)",
        )
        self.assertGreater(
            snr_v,
            45.0,
            msg=f"SNR(v.grad) too low: {snr_v:.2f} dB (expected > 45.0 dB)",
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_fused_projection_rotary_blockattn_e2e_parity(self) -> None:
        torch.manual_seed(2024)
        device = torch.device("cuda")

        qhead = 3
        kvhead = 3
        head_dim = 64
        hidden = qhead * head_dim
        n_out = (qhead + 2 * kvhead) * head_dim

        b = 64
        max_len = 256
        lengths = torch.randint(1, max_len + 1, (b,), device=device)
        offsets = torch.zeros(b + 1, dtype=torch.int32, device=device)
        offsets[1:] = torch.cumsum(lengths, dim=0).to(torch.int32)
        m = int(offsets[-1].item())

        x = torch.randn(m, hidden, device=device, dtype=torch.bfloat16)
        w = torch.randn(hidden, n_out, device=device, dtype=torch.bfloat16) * 0.01
        sinu_pos = torch.randn(m, head_dim, device=device, dtype=torch.bfloat16)
        grad_out = torch.randn(m, qhead, head_dim, device=device, dtype=torch.bfloat16)

        x_new = x.clone().detach().requires_grad_(True)
        w_new = w.clone().detach().requires_grad_(True)
        dummy_q = torch.empty(1, qhead, head_dim, device=device, dtype=torch.bfloat16)
        dummy_k = torch.empty(1, kvhead, head_dim, device=device, dtype=torch.bfloat16)
        dummy_v = torch.empty(1, kvhead, head_dim, device=device, dtype=torch.bfloat16)

        out_new = tlx_block_attention(
            dummy_q,
            dummy_k,
            dummy_v,
            offsets,
            offsets.cpu(),
            include_projection_rotary=True,
            x=x_new,
            w=w_new,
            sinu_pos=sinu_pos,
            qhead=qhead,
            kvhead=kvhead,
            max_seq_len=max_len,
        )[0]
        out_new.backward(grad_out)
        self.assertIsNotNone(x_new.grad)
        self.assertIsNotNone(w_new.grad)

        x_ref = x.clone().detach().requires_grad_(True)
        w_ref = w.clone().detach().requires_grad_(True)
        qkv_ref = torch.matmul(x_ref, w_ref)
        d_model = qhead * head_dim
        kv_d_model = kvhead * head_dim
        q_ref = qkv_ref[:, :d_model].reshape(-1, qhead, head_dim).contiguous()
        k_ref = (
            qkv_ref[:, d_model : d_model + kv_d_model]
            .reshape(-1, kvhead, head_dim)
            .contiguous()
        )
        v_ref = (
            qkv_ref[:, d_model + kv_d_model :]
            .reshape(-1, kvhead, head_dim)
            .contiguous()
        )
        q_ref, k_ref, v_ref = apply_rotary_pos_emb_jagged(q_ref, k_ref, v_ref, sinu_pos)
        out_ref = pytorch_block_attention_forward_fp32(
            q_ref,
            k_ref,
            v_ref,
            offsets.cpu(),
            block_size=64,
            scale=1.0 / math.sqrt(head_dim),
        )
        out_ref.backward(grad_out)
        self.assertIsNotNone(x_ref.grad)
        self.assertIsNotNone(w_ref.grad)

        assert x_new.grad is not None
        assert w_new.grad is not None
        assert x_ref.grad is not None
        assert w_ref.grad is not None

        snr_fwd = snr_db(out_new, out_ref)
        snr_dx = snr_db(x_new.grad, x_ref.grad)
        snr_dw = snr_db(w_new.grad, w_ref.grad)

        self.assertGreater(
            snr_fwd,
            35.0,
            msg=f"SNR(fwd) too low: {snr_fwd:.2f} dB (expected > 35.0 dB)",
        )
        self.assertGreater(
            snr_dx,
            20.0,
            msg=f"SNR(dx) too low: {snr_dx:.2f} dB (expected > 20.0 dB)",
        )
        self.assertGreater(
            snr_dw,
            20.0,
            msg=f"SNR(dw) too low: {snr_dw:.2f} dB (expected > 20.0 dB)",
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_tlx_block_attention_executes_directly(self) -> None:
        torch.manual_seed(7)
        device = torch.device("cuda")

        H = 4
        D = 64
        data = generate_jagged_data(
            B=16,
            max_seq_len=128,
            H=H,
            D=D,
            dtype=torch.bfloat16,
            device=device,
        )

        q = data["q_weights"].clone().detach().requires_grad_(True)
        k = data["k_weights"].clone().detach().requires_grad_(True)
        v = data["v_weights"].clone().detach().requires_grad_(True)
        do = data["do"]
        offsets = data["q_offsets"]

        out = tlx_block_attention(q, k, v, offsets, offsets.cpu())[0]
        self.assertEqual(out.shape, q.shape)

        loss = (out.float() * do.float()).mean()
        loss.backward()
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertIsNotNone(v.grad)


if __name__ == "__main__":
    unittest.main()
