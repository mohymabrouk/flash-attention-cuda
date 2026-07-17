"""Always-on tests for the explicit PyTorch correctness oracle."""

from __future__ import annotations

import pytest
import torch

from flash_attention_cuda import manual_attention, torch_sdpa_attention


SELF_ATTENTION_SHAPES = (
    (1, 1, 1, 1),
    (1, 2, 17, 7),
    (2, 3, 33, 32),
    (1, 1, 65, 64),
)


@pytest.mark.parametrize("shape", SELF_ATTENTION_SHAPES)
@pytest.mark.parametrize("causal", (False, True))
def test_reference_matches_sdpa(shape: tuple[int, ...], causal: bool) -> None:
    generator = torch.Generator().manual_seed(1234)
    q = torch.randn(shape, generator=generator)
    k = torch.randn(shape, generator=generator)
    v = torch.randn(shape, generator=generator)

    actual = manual_attention(q, k, v, causal=causal)
    expected = torch_sdpa_attention(q, k, v, causal=causal)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("query_length,key_length", ((1, 7), (7, 1), (17, 31)))
def test_reference_supports_noncausal_cross_attention(
    query_length: int, key_length: int
) -> None:
    generator = torch.Generator().manual_seed(7)
    q = torch.randn(2, 2, query_length, 11, generator=generator)
    k = torch.randn(2, 2, key_length, 11, generator=generator)
    v = torch.randn(2, 2, key_length, 11, generator=generator)

    actual = manual_attention(q, k, v)
    expected = torch_sdpa_attention(q, k, v)

    assert actual.shape == q.shape
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_reference_honors_custom_softmax_scale() -> None:
    generator = torch.Generator().manual_seed(22)
    q = torch.randn(1, 2, 13, 9, generator=generator, dtype=torch.float64)
    k = torch.randn(1, 2, 13, 9, generator=generator, dtype=torch.float64)
    v = torch.randn(1, 2, 13, 9, generator=generator, dtype=torch.float64)

    actual = manual_attention(q, k, v, causal=True, softmax_scale=0.17)
    expected = torch_sdpa_attention(q, k, v, causal=True, softmax_scale=0.17)

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("causal", (False, True))
def test_reference_first_order_gradients_match_sdpa(causal: bool) -> None:
    generator = torch.Generator().manual_seed(101)
    inputs = tuple(
        torch.randn(1, 2, 9, 5, generator=generator, dtype=torch.float64)
        for _ in range(3)
    )
    grad_output = torch.randn(1, 2, 9, 5, generator=generator, dtype=torch.float64)

    reference_inputs = tuple(tensor.clone().requires_grad_() for tensor in inputs)
    reference_output = manual_attention(*reference_inputs, causal=causal)
    reference_grads = torch.autograd.grad(
        reference_output, reference_inputs, grad_outputs=grad_output
    )

    sdpa_inputs = tuple(tensor.clone().requires_grad_() for tensor in inputs)
    sdpa_output = torch_sdpa_attention(*sdpa_inputs, causal=causal)
    sdpa_grads = torch.autograd.grad(sdpa_output, sdpa_inputs, grad_outputs=grad_output)

    torch.testing.assert_close(reference_output, sdpa_output, rtol=1e-10, atol=1e-10)
    for actual, expected in zip(reference_grads, sdpa_grads, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-9)


def test_reference_is_stable_for_large_logits() -> None:
    q = torch.tensor([[[[1.0e4, -1.0e4], [-1.0e4, 1.0e4]]]])
    k = torch.tensor([[[[1.0e4, -1.0e4], [-1.0e4, 1.0e4]]]])
    v = torch.tensor([[[[3.0, -2.0], [-4.0, 5.0]]]])

    output = manual_attention(q, k, v)

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, v)


def test_reference_preserves_inputs() -> None:
    generator = torch.Generator().manual_seed(3)
    tensors = tuple(torch.randn(1, 1, 8, 4, generator=generator) for _ in range(3))
    originals = tuple(tensor.clone() for tensor in tensors)

    manual_attention(*tensors, causal=True)

    for actual, expected in zip(tensors, originals, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
