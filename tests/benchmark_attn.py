# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# @nolint

# pyre-strict

import time
from collections.abc import Callable
from typing import NamedTuple

import torch

Timing = NamedTuple("timing", [("mean", float)])


from ads_mkl.ops.cute_dsl.gdpa.src.interface import (
    flash_attn_func as flash_attn_func_python,
    flash_attn_varlen_func as flash_attn_varlen_func_python,
)
from einops import rearrange
from triton.testing import do_bench


def time_fwd(
    func: Callable[..., object],
    *args: object,
    warmup: int = 100,
    rep: int = 1000,
    verbose: bool = True,
    desc: str = "",
    **kwargs: object,
) -> Timing:
    return Timing(
        do_bench(lambda: func(*args, **kwargs), warmup=warmup, rep=rep) * 1e-3
    )


def flops(
    batch: int,
    nheads: int,
    seqlen_q: int,
    seqlen_k: int,
    headdim: int,
    headdim_v: int,
    causal: bool = False,
    window_size: tuple[int | None, int | None] = (None, None),
) -> float:
    if causal:
        avg_seqlen = (max(0, seqlen_k - seqlen_q) + seqlen_k) / 2
    else:
        if window_size == (None, None):
            avg_seqlen = seqlen_k
        else:
            row_idx = torch.arange(seqlen_q, device="cuda")
            col_left = (
                torch.maximum(
                    row_idx + seqlen_k - seqlen_q - window_size[0], torch.tensor(0)
                )
                if window_size[0] is not None
                else torch.zeros_like(row_idx)
            )
            col_right = (
                torch.minimum(
                    row_idx + seqlen_k - seqlen_q - window_size[1],
                    torch.tensor(seqlen_k - 1),
                )
                if window_size[1] is not None
                else torch.full_like(row_idx, seqlen_k - 1)
            )
            avg_seqlen = (col_right - col_left + 1).float().mean().item()
    return batch * nheads * 2 * seqlen_q * avg_seqlen * (headdim + headdim_v)


torch.manual_seed(0)
dropout_p = 0.0
causal = False
dtype = torch.bfloat16
# dtype = torch.float8_e4m3fn
dtype_gen = torch.bfloat16 if dtype == torch.float8_e4m3fn else dtype
device = "cuda"
verbose = True
varlen = True
has_backward = False
V_colmajor = False

headdim = 128
bs_seqlen_vals = [(1152, 1000, 512)]
# bs_seqlen_vals += [(100, 2025, 2025), (1000, 2025, 2025)]

nheads = 4
nheads_kv = nheads

headdim_v = headdim
has_qv = False
# sinks = torch.randn(nheads, dtype=torch.bfloat16, device=device)
sinks = None
softcap = 0.0
pack_gqa = None

for batch_size, seqlen_q, seqlen in bs_seqlen_vals:
    for varlen in [True, False]:  #
        num_splits = 0
        # window_size = (-1, -1)
        window_size = (None, None)
        window_size_fa = (-1, -1)
        # window_size = (seqlen // 2 - 1, 0)
        leftpad_k = None
        # leftpad_k = torch.full((batch_size,), 0, device=device, dtype=torch.int32)
        q = torch.randn(
            batch_size,
            seqlen_q,
            nheads,
            headdim,
            device=device,
            dtype=dtype_gen,
            requires_grad=has_backward,
        )
        k = torch.randn(
            batch_size,
            seqlen,
            nheads_kv,
            headdim,
            device=device,
            dtype=dtype_gen,
            requires_grad=has_backward,
        )
        v = torch.randn(
            batch_size,
            seqlen,
            nheads_kv,
            headdim_v,
            device=device,
            dtype=dtype_gen,
            requires_grad=has_backward,
        )
        q, k, v = [x.detach().to(dtype).requires_grad_(has_backward) for x in [q, k, v]]
        v_colmajor = (
            v.detach()
            .transpose(-1, -3)
            .contiguous()
            .transpose(-1, -3)
            .requires_grad_(has_backward)
        )
        v_fa3 = v if not V_colmajor else v_colmajor
        qv = (
            torch.randn(
                batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype_gen
            )
            if has_qv
            else None
        )
        # q = torch.randint(-2, 3, (batch_size, seqlen, nheads, headdim), device=device, dtype=torch.int32).to(dtype)
        # k = torch.randint(-2, 3, (batch_size, seqlen, nheads, headdim), device=device, dtype=torch.int32).to(dtype)
        # v = torch.randint(-2, 3, (batch_size, seqlen, nheads, headdim_v), device=device, dtype=torch.int32).to(dtype)
        g = torch.randn(
            batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype_gen
        )
        o = torch.randn(
            batch_size, seqlen_q, nheads, headdim_v, device=device, dtype=dtype_gen
        )
        stats = torch.randn(
            batch_size, seqlen_q, nheads, 1, device=device, dtype=torch.float32
        )

        if varlen:
            q_unpad, k_unpad, v_unpad = [
                rearrange(x.detach(), "b s h d -> (b s) h d").requires_grad_(
                    has_backward
                )
                for x in [q, k, v]
            ]
            cu_seqlens_q = (
                torch.arange(batch_size + 1, device=device, dtype=torch.int32)
                * seqlen_q
            )
            cu_seqlens_k = (
                torch.arange(batch_size + 1, device=device, dtype=torch.int32) * seqlen
            )
        # for causal in [False, True]:
        for causal in [False]:
            print(
                f"\n### {varlen= } {batch_size =}, {nheads = }, {headdim = }, {causal = }, {seqlen_q = }, {seqlen = } ###"
            )
            nFLOPS = flops(
                batch_size,
                nheads,
                seqlen_q,
                seqlen,
                headdim if not has_qv else headdim + headdim_v,
                headdim_v,
                causal=causal,
                window_size=window_size,
            )

            if flash_attn_func_python is not None:
                if not varlen:
                    m1_py = time_fwd(
                        flash_attn_func_python,
                        q,
                        k,
                        v_fa3,
                        causal=causal,
                        window_size=window_size,
                        learnable_sink=sinks,
                        softcap=softcap,
                        pack_gqa=pack_gqa,
                        verbose=verbose,
                        desc="Fav3 python",
                    )
                    print(f"### {q.shape = } {k.shape = } {v.shape = } ###")
                else:
                    m1_py = time_fwd(
                        flash_attn_varlen_func_python,
                        q_unpad,
                        k_unpad,
                        v_unpad,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        max_seqlen_q=seqlen_q,
                        page_table=None,
                        causal=causal,
                        window_size=window_size,
                        softcap=softcap,
                        pack_gqa=pack_gqa,
                        verbose=verbose,
                        desc="Fav3 python",
                    )
                    print(
                        f"### {q_unpad.shape = } {k_unpad.shape = } {v_unpad.shape = } ###"
                    )

            print(
                f"### FA Python fwd: {m1_py.mean * 1e3:.3f}ms, {(nFLOPS / m1_py.mean * 1e-12):.1f} TFLOPS"
            )

            time.sleep(1)
