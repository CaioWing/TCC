"""Attention implementations used by the benchmark suite.

The Triton kernel is intentionally scoped to the forward pass used in ViT-style
inference: dense non-causal attention without dropout.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_LOG2_E = 1.4426950408889634


def attention_eager(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Reference PyTorch attention that materializes the attention matrix."""
    _validate_qkv(q, k, v)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
    return torch.matmul(probs, v)


def attention_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """PyTorch optimized scaled dot-product attention baseline."""
    _validate_qkv(q, k, v)
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    )


@triton.jit
def _attention_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_on: tl.constexpr,
    stride_od: tl.constexpr,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    sm_scale_log2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_idx = pid_bh // num_heads
    head_idx = pid_bh - batch_idx * num_heads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_base = batch_idx * stride_qb + head_idx * stride_qh
    k_base = batch_idx * stride_kb + head_idx * stride_kh
    v_base = batch_idx * stride_vb + head_idx * stride_vh
    out_base = batch_idx * stride_ob + head_idx * stride_oh

    q = tl.load(
        q_ptr + q_base + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        k_offsets = start_n + offs_n
        k = tl.load(
            k_ptr + k_base + k_offsets[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=(k_offsets[:, None] < seq_len) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )
        v = tl.load(
            v_ptr + v_base + k_offsets[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=(k_offsets[:, None] < seq_len) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )

        qk = tl.dot(q, tl.trans(k), input_precision="tf32") * sm_scale_log2
        qk = tl.where(
            (offs_m[:, None] < seq_len) & (k_offsets[None, :] < seq_len),
            qk,
            -float("inf"),
        )

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="tf32")
        m_i = m_ij

    acc = acc / l_i[:, None]

    tl.store(
        out_ptr + out_base + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od,
        acc,
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < HEAD_DIM),
    )


def triton_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_m: int = 16,
    block_n: int = 64,
) -> torch.Tensor:
    """Run the custom dense attention forward kernel."""
    _validate_qkv(q, k, v)
    if not q.is_cuda:
        raise RuntimeError("triton_flash_attention requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("supported dtypes are float16, bfloat16 and float32")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    batch, heads, seq_len, head_dim = q.shape
    if head_dim > 128:
        raise ValueError("head_dim > 128 is outside the current kernel scope")

    block_d = triton.next_power_of_2(head_dim)
    out = torch.empty_like(q)
    grid = (triton.cdiv(seq_len, block_m), batch * heads)
    sm_scale_log2 = (1.0 / math.sqrt(head_dim)) * _LOG2_E
    num_warps = 4 if head_dim <= 64 else 8

    _attention_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        heads,
        seq_len,
        sm_scale_log2,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        HEAD_DIM=head_dim,
        num_warps=num_warps,
        num_stages=3,
    )
    return out


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("expected q, k and v with shape [batch, heads, seq_len, head_dim]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k and v must have the same shape in this benchmark")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k and v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k and v must have the same dtype")
