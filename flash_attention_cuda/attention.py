"""Autograd-aware wrapper around the custom CUDA extension."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from .reference import _validate_reference_inputs, manual_attention, torch_sdpa_attention

try:
    from . import _C
except ImportError as exc:  # The reference implementations remain usable without nvcc.
    _C = None
    _EXTENSION_IMPORT_ERROR: ImportError | None = exc
else:
    _EXTENSION_IMPORT_ERROR = None


Implementation = Literal["auto", "cuda", "sdpa", "reference"]
_CUDA_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def extension_available() -> bool:
    """Return whether the compiled CUDA extension imported successfully."""

    return _C is not None


def extension_error() -> ImportError | None:
    """Return the captured extension import error, if any."""

    return _EXTENSION_IMPORT_ERROR


def _extension_failure_message() -> str:
    detail = f" Original import error: {_EXTENSION_IMPORT_ERROR}" if _EXTENSION_IMPORT_ERROR else ""
    return (
        "The custom CUDA extension is not available. Build it with "
        "`python -m pip install -v -e . --no-build-isolation` in an environment "
        f"that has a CUDA-enabled PyTorch installation and nvcc.{detail}"
    )


class _FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        causal: bool,
        softmax_scale: float,
    ) -> Tensor:
        output, logsumexp = _C.forward(q, k, v, causal, softmax_scale)
        ctx.save_for_backward(q, k, v, output, logsumexp)
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        return output

    @staticmethod
    @once_differentiable
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, None, None]:
        q, k, v, output, logsumexp = ctx.saved_tensors
        dq, dk, dv = _C.backward(
            grad_output.contiguous(),
            q,
            k,
            v,
            output,
            logsumexp,
            ctx.causal,
            ctx.softmax_scale,
        )
        return dq, dk, dv, None, None


def flash_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    implementation: Implementation = "auto",
) -> Tensor:
    """Compute scaled dot-product attention.

    ``implementation="cuda"`` requires the compiled extension and CUDA inputs.
    ``"auto"`` selects it when the input satisfies the native kernel contract,
    otherwise it uses PyTorch SDPA. ``"reference"`` selects the explicit,
    quadratic PyTorch oracle intended for tests and teaching.
    """

    _validate_reference_inputs(q, k, v, causal)
    if implementation not in ("auto", "cuda", "sdpa", "reference"):
        raise ValueError(
            "implementation must be one of: 'auto', 'cuda', 'sdpa', 'reference'"
        )

    scale = float(softmax_scale) if softmax_scale is not None else q.shape[-1] ** -0.5
    if not math.isfinite(scale):
        raise ValueError("softmax_scale must be finite")

    if implementation == "reference":
        return manual_attention(q, k, v, causal=causal, softmax_scale=scale)
    if implementation == "sdpa":
        return torch_sdpa_attention(q, k, v, causal=causal, softmax_scale=scale)

    native_compatible = (
        q.is_cuda
        and q.dtype in _CUDA_DTYPES
        and q.shape[-1] <= 256
        and extension_available()
    )
    if implementation == "cuda" and not native_compatible:
        if not q.is_cuda:
            raise ValueError("implementation='cuda' requires CUDA tensors")
        if q.dtype not in _CUDA_DTYPES:
            raise TypeError("the CUDA kernel supports float16, bfloat16, and float32")
        if q.shape[-1] > 256:
            raise ValueError("the CUDA kernel supports head dimensions up to 256")
        raise RuntimeError(_extension_failure_message())

    if native_compatible:
        return _FlashAttention.apply(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            causal,
            scale,
        )

    return torch_sdpa_attention(q, k, v, causal=causal, softmax_scale=scale)
