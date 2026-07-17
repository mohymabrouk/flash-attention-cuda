"""Always-on tests for dispatch, validation, and the public Python API."""

from __future__ import annotations

import math

import pytest
import torch

from flash_attention_cuda import flash_attention, manual_attention


def _inputs(
    *,
    batch: int = 2,
    heads: int = 3,
    query_length: int = 11,
    key_length: int | None = None,
    head_dim: int = 7,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key_length = query_length if key_length is None else key_length
    generator = torch.Generator().manual_seed(19)
    q = torch.randn(
        batch, heads, query_length, head_dim, dtype=dtype, generator=generator
    )
    k = torch.randn(batch, heads, key_length, head_dim, dtype=dtype, generator=generator)
    v = torch.randn(batch, heads, key_length, head_dim, dtype=dtype, generator=generator)
    return q, k, v


@pytest.mark.parametrize("implementation", ("reference", "sdpa", "auto"))
@pytest.mark.parametrize("causal", (False, True))
def test_public_cpu_implementations_match_reference(
    implementation: str, causal: bool
) -> None:
    q, k, v = _inputs()

    output = flash_attention(q, k, v, causal=causal, implementation=implementation)
    expected = manual_attention(q, k, v, causal=causal)

    assert output.shape == q.shape
    assert output.dtype == q.dtype
    assert output.device == q.device
    torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-5)


def test_public_api_supports_noncontiguous_inputs() -> None:
    generator = torch.Generator().manual_seed(12)
    bases = tuple(torch.randn(2, 2, 13, 18, generator=generator) for _ in range(3))
    q, k, v = (base[..., ::2] for base in bases)
    assert not q.is_contiguous() and not k.is_contiguous() and not v.is_contiguous()

    output = flash_attention(q, k, v, implementation="auto")
    expected = manual_attention(q, k, v)

    torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-5)


def test_public_auto_dispatch_preserves_first_order_gradients_on_cpu() -> None:
    tensors = tuple(tensor.requires_grad_() for tensor in _inputs(head_dim=5))
    grad_output = torch.randn_like(tensors[0])

    output = flash_attention(*tensors, causal=True, implementation="auto")
    actual_grads = torch.autograd.grad(output, tensors, grad_outputs=grad_output)

    oracle_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in tensors)
    oracle_output = manual_attention(*oracle_inputs, causal=True)
    expected_grads = torch.autograd.grad(
        oracle_output, oracle_inputs, grad_outputs=grad_output
    )

    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("scale", (0.25, 1.0, -0.125))
def test_public_api_honors_finite_scale(scale: float) -> None:
    q, k, v = _inputs(batch=1, heads=1)

    output = flash_attention(
        q, k, v, softmax_scale=scale, implementation="reference"
    )
    expected = manual_attention(q, k, v, softmax_scale=scale)

    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("scale", (math.inf, -math.inf, math.nan))
def test_public_api_rejects_nonfinite_scale(scale: float) -> None:
    q, k, v = _inputs()
    with pytest.raises(ValueError, match="finite"):
        flash_attention(q, k, v, softmax_scale=scale)


def test_public_api_rejects_unknown_implementation() -> None:
    q, k, v = _inputs()
    with pytest.raises(ValueError, match="implementation"):
        flash_attention(q, k, v, implementation="mystery")  # type: ignore[arg-type]


def test_cuda_implementation_rejects_cpu_tensors() -> None:
    q, k, v = _inputs()
    with pytest.raises(ValueError, match="requires CUDA"):
        flash_attention(q, k, v, implementation="cuda")


def test_validation_rejects_non_tensors() -> None:
    q, k, _ = _inputs()
    with pytest.raises(TypeError, match="torch.Tensor"):
        flash_attention(q, k, object())  # type: ignore[arg-type]


def test_validation_rejects_wrong_rank() -> None:
    q, k, v = _inputs()
    with pytest.raises(ValueError, match="shape"):
        flash_attention(q[0], k, v)


def test_validation_rejects_nonfloating_inputs() -> None:
    q = torch.ones(1, 1, 4, 8, dtype=torch.int64)
    with pytest.raises(TypeError, match="floating-point"):
        flash_attention(q, q, q)


def test_validation_rejects_mixed_dtypes() -> None:
    q, k, v = _inputs()
    with pytest.raises(ValueError, match="same dtype"):
        flash_attention(q, k.double(), v)


@pytest.mark.parametrize(
    "mutation,error",
    (
        ("batch", "batch and head"),
        ("heads", "batch and head"),
        ("key_value_length", "same sequence length"),
        ("head_dim", "same head dimension"),
    ),
)
def test_validation_rejects_incompatible_shapes(mutation: str, error: str) -> None:
    q, k, v = _inputs()
    if mutation == "batch":
        k = k[:1]
    elif mutation == "heads":
        k = k[:, :2]
    elif mutation == "key_value_length":
        v = v[..., :-1, :]
    else:
        k = k[..., :-1]

    with pytest.raises(ValueError, match=error):
        flash_attention(q, k, v)


@pytest.mark.parametrize(
    "shape",
    (
        (1, 1, 0, 8),
        (1, 1, 4, 0),
    ),
)
def test_validation_rejects_empty_dimensions(shape: tuple[int, ...]) -> None:
    q = torch.empty(shape)
    k = torch.empty(shape)
    v = torch.empty(shape)
    with pytest.raises(ValueError, match="non-zero"):
        flash_attention(q, k, v)


def test_validation_rejects_causal_cross_attention() -> None:
    q, k, v = _inputs(query_length=7, key_length=11)
    with pytest.raises(ValueError, match="causal.*equal"):
        flash_attention(q, k, v, causal=True)
