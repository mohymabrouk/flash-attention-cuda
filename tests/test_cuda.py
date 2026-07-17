"""CUDA output, autograd, stream, and wrapper-contract tests."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from flash_attention_cuda import extension_available, flash_attention, manual_attention


pytestmark = pytest.mark.cuda


OUTPUT_CASES = (
    # B, H, Nq, Nk, D, causal: boundaries and tails around common tile sizes.
    (1, 1, 1, 1, 1, False),
    (2, 3, 17, 17, 7, True),
    (1, 2, 31, 31, 31, False),
    (1, 2, 33, 33, 64, True),
    (1, 1, 63, 63, 96, False),
    (1, 1, 65, 65, 128, True),
    (1, 2, 31, 47, 55, False),
    (1, 1, 129, 97, 127, False),
    (1, 1, 73, 73, 256, True),
)

GRADIENT_CASES = (
    (1, 1, 17, 17, 7, False),
    (1, 2, 33, 33, 32, True),
    (1, 1, 65, 47, 63, False),
)


def _skip_unsupported_bfloat16(dtype: torch.dtype) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("this GPU/PyTorch build does not support CUDA bfloat16")


def _tolerances(dtype: torch.dtype, *, gradients: bool = False) -> tuple[float, float]:
    if dtype == torch.float32:
        return (8e-4, 8e-4) if gradients else (4e-4, 4e-4)
    if dtype == torch.float16:
        return (5e-2, 4e-2) if gradients else (3e-2, 2e-2)
    return (8e-2, 7e-2) if gradients else (6e-2, 5e-2)


def _random_tensors(
    shape: Sequence[int],
    *,
    key_length: int,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, query_length, head_dim = shape
    generator = torch.Generator(device=device).manual_seed(2025)
    q = torch.randn(
        batch,
        heads,
        query_length,
        head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    ).requires_grad_(requires_grad)
    k = torch.randn(
        batch,
        heads,
        key_length,
        head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    ).requires_grad_(requires_grad)
    v = torch.randn(
        batch,
        heads,
        key_length,
        head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    ).requires_grad_(requires_grad)
    return q, k, v


def test_extension_is_available(cuda_device: torch.device) -> None:
    assert cuda_device.type == "cuda"
    assert extension_available()


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("case", OUTPUT_CASES)
def test_cuda_output_matches_reference(
    cuda_device: torch.device,
    dtype: torch.dtype,
    case: tuple[int, int, int, int, int, bool],
) -> None:
    _skip_unsupported_bfloat16(dtype)
    batch, heads, query_length, key_length, head_dim, causal = case
    q, k, v = _random_tensors(
        (batch, heads, query_length, head_dim),
        key_length=key_length,
        dtype=dtype,
        device=cuda_device,
    )

    output = flash_attention(q, k, v, causal=causal, implementation="cuda")
    expected = manual_attention(q, k, v, causal=causal)

    rtol, atol = _tolerances(dtype)
    assert output.shape == q.shape
    assert output.dtype == dtype
    assert output.device == cuda_device
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("case", GRADIENT_CASES)
def test_cuda_first_order_gradients_match_reference(
    cuda_device: torch.device,
    dtype: torch.dtype,
    case: tuple[int, int, int, int, int, bool],
) -> None:
    _skip_unsupported_bfloat16(dtype)
    batch, heads, query_length, key_length, head_dim, causal = case
    raw_inputs = _random_tensors(
        (batch, heads, query_length, head_dim),
        key_length=key_length,
        dtype=dtype,
        device=cuda_device,
    )
    generator = torch.Generator(device=cuda_device).manual_seed(404)
    grad_output = torch.randn(
        batch,
        heads,
        query_length,
        head_dim,
        device=cuda_device,
        dtype=dtype,
        generator=generator,
    )

    cuda_inputs = tuple(tensor.detach().requires_grad_() for tensor in raw_inputs)
    cuda_output = flash_attention(
        *cuda_inputs, causal=causal, implementation="cuda"
    )
    cuda_grads = torch.autograd.grad(
        cuda_output, cuda_inputs, grad_outputs=grad_output
    )

    oracle_inputs = tuple(tensor.detach().requires_grad_() for tensor in raw_inputs)
    oracle_output = manual_attention(*oracle_inputs, causal=causal)
    oracle_grads = torch.autograd.grad(
        oracle_output, oracle_inputs, grad_outputs=grad_output
    )

    rtol, atol = _tolerances(dtype, gradients=True)
    torch.testing.assert_close(cuda_output, oracle_output, rtol=rtol, atol=atol)
    for actual, expected in zip(cuda_grads, oracle_grads, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def test_cuda_custom_scale_matches_reference(cuda_device: torch.device) -> None:
    q, k, v = _random_tensors(
        (1, 2, 37, 23), key_length=37, dtype=torch.float32, device=cuda_device
    )

    output = flash_attention(
        q,
        k,
        v,
        causal=True,
        softmax_scale=0.037,
        implementation="cuda",
    )
    expected = manual_attention(q, k, v, causal=True, softmax_scale=0.037)

    torch.testing.assert_close(output, expected, rtol=4e-4, atol=4e-4)


def test_cuda_is_stable_for_large_logits(cuda_device: torch.device) -> None:
    generator = torch.Generator(device=cuda_device).manual_seed(55)
    q = torch.randn(
        1, 2, 65, 32, device=cuda_device, generator=generator, dtype=torch.float32
    ) * 100.0
    k = torch.randn(
        1, 2, 65, 32, device=cuda_device, generator=generator, dtype=torch.float32
    ) * 100.0
    v = torch.randn(
        1, 2, 65, 32, device=cuda_device, generator=generator, dtype=torch.float32
    )

    output = flash_attention(q, k, v, implementation="cuda")
    expected = manual_attention(q, k, v)

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)


def test_cuda_wrapper_accepts_noncontiguous_inputs_and_backpropagates(
    cuda_device: torch.device,
) -> None:
    generator = torch.Generator(device=cuda_device).manual_seed(88)
    bases = tuple(
        torch.randn(
            1,
            2,
            33,
            34,
            device=cuda_device,
            generator=generator,
            dtype=torch.float32,
        )
        for _ in range(3)
    )
    inputs = tuple(base[..., ::2].detach().requires_grad_() for base in bases)
    assert all(not tensor.is_contiguous() for tensor in inputs)
    grad_output = torch.randn_like(inputs[0])

    output = flash_attention(*inputs, implementation="cuda")
    gradients = torch.autograd.grad(output, inputs, grad_outputs=grad_output)

    oracle_inputs = tuple(tensor.detach().requires_grad_() for tensor in inputs)
    oracle_output = manual_attention(*oracle_inputs)
    oracle_gradients = torch.autograd.grad(
        oracle_output, oracle_inputs, grad_outputs=grad_output
    )

    torch.testing.assert_close(output, oracle_output, rtol=4e-4, atol=4e-4)
    for actual, expected in zip(gradients, oracle_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=8e-4, atol=8e-4)


def test_cuda_runs_on_nondefault_stream(cuda_device: torch.device) -> None:
    q, k, v = _random_tensors(
        (1, 2, 65, 32), key_length=65, dtype=torch.float32, device=cuda_device
    )
    producer = torch.cuda.current_stream(cuda_device)
    stream = torch.cuda.Stream(device=cuda_device)
    stream.wait_stream(producer)

    with torch.cuda.stream(stream):
        output = flash_attention(q, k, v, causal=True, implementation="cuda")
        completion = torch.cuda.Event()
        completion.record(stream)
    completion.synchronize()

    expected = manual_attention(q, k, v, causal=True)
    torch.testing.assert_close(output, expected, rtol=4e-4, atol=4e-4)


def test_auto_selects_cuda_for_compatible_inputs(cuda_device: torch.device) -> None:
    q, k, v = _random_tensors(
        (1, 1, 17, 13), key_length=17, dtype=torch.float32, device=cuda_device
    )

    automatic = flash_attention(q, k, v, implementation="auto")
    explicit = flash_attention(q, k, v, implementation="cuda")

    torch.testing.assert_close(automatic, explicit, rtol=0.0, atol=0.0)


def test_cuda_rejects_unsupported_dtype(cuda_device: torch.device) -> None:
    q, k, v = _random_tensors(
        (1, 1, 8, 8), key_length=8, dtype=torch.float64, device=cuda_device
    )
    with pytest.raises(TypeError, match="float16.*bfloat16.*float32"):
        flash_attention(q, k, v, implementation="cuda")


def test_cuda_rejects_head_dimension_above_256(cuda_device: torch.device) -> None:
    q, k, v = _random_tensors(
        (1, 1, 4, 257), key_length=4, dtype=torch.float32, device=cuda_device
    )
    with pytest.raises(ValueError, match="up to 256"):
        flash_attention(q, k, v, implementation="cuda")


def test_validation_rejects_mixed_devices(cuda_device: torch.device) -> None:
    q, k, v = _random_tensors(
        (1, 1, 8, 8), key_length=8, dtype=torch.float32, device=cuda_device
    )
    with pytest.raises(ValueError, match="same device"):
        flash_attention(q, k.cpu(), v, implementation="cuda")
