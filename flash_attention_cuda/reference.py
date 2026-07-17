"""Readable PyTorch attention implementations used as correctness oracles."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def _validate_reference_inputs(q: Tensor, k: Tensor, v: Tensor, causal: bool) -> None:
    if not all(isinstance(tensor, Tensor) for tensor in (q, k, v)):
        raise TypeError("q, k, and v must be torch.Tensor instances")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [batch, heads, sequence, head_dim]")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if not q.is_floating_point():
        raise TypeError("q, k, and v must have a floating-point dtype")
    if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:
        raise ValueError("q, k, and v must have the same batch and head dimensions")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must have the same sequence length")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError("this project requires q, k, and v to have the same head dimension")
    if min(q.shape[-2], k.shape[-2], q.shape[-1]) <= 0:
        raise ValueError("sequence lengths and head dimension must be non-zero")
    if causal and q.shape[-2] != k.shape[-2]:
        raise ValueError("causal attention currently requires equal query and key lengths")


def manual_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    causal: bool = False,
    softmax_scale: float | None = None,
) -> Tensor:
    """Compute exact attention with explicit score and probability matrices.

    This implementation is deliberately simple and materializes the quadratic
    attention matrix. Half and bfloat16 inputs are accumulated in float32 so it
    can serve as a stable correctness oracle for the custom kernel.
    """

    _validate_reference_inputs(q, k, v, causal)
    scale = float(softmax_scale) if softmax_scale is not None else q.shape[-1] ** -0.5
    if not math.isfinite(scale):
        raise ValueError("softmax_scale must be finite")

    accumulation_dtype = (
        torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    )
    q_acc = q.to(accumulation_dtype)
    k_acc = k.to(accumulation_dtype)
    v_acc = v.to(accumulation_dtype)

    scores = torch.matmul(q_acc, k_acc.transpose(-2, -1)) * scale
    if causal:
        query_length, key_length = q.shape[-2], k.shape[-2]
        future = torch.ones(
            (query_length, key_length), device=q.device, dtype=torch.bool
        ).triu(diagonal=1)
        scores = scores.masked_fill(future, -torch.inf)

    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v_acc).to(q.dtype)


def torch_sdpa_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    causal: bool = False,
    softmax_scale: float | None = None,
) -> Tensor:
    """Call PyTorch scaled-dot-product attention with the same project contract."""

    _validate_reference_inputs(q, k, v, causal)
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=causal,
        scale=softmax_scale,
    )
