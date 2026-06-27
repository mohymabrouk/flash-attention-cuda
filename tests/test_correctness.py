import pytest
import torch

from src.attention_naive import manual_attention, torch_sdpa_attention


@pytest.mark.parametrize("causal", [False, True])
def test_manual_attention_matches_torch_sdpa(causal):
    torch.manual_seed(0)

    batch = 2
    heads = 4
    seq_len = 128
    head_dim = 64

    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, heads, seq_len, head_dim)
    v = torch.randn(batch, heads, seq_len, head_dim)

    out_manual = manual_attention(q, k, v, causal=causal)
    out_torch = torch_sdpa_attention(q, k, v, causal=causal)

    torch.testing.assert_close(out_manual, out_torch, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("seq_len", [1, 2, 16, 128, 513])
@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("heads", [1, 4])
def test_manual_attention_shape(batch, heads, seq_len, head_dim):
    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, heads, seq_len, head_dim)
    v = torch.randn(batch, heads, seq_len, head_dim)

    out = manual_attention(q, k, v)

    assert out.shape == (batch, heads, seq_len, head_dim)


@pytest.mark.parametrize("seq_len", [1, 2, 16, 128, 513])
@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_manual_attention_matches_torch_sdpa_various_shapes(seq_len, head_dim):
    torch.manual_seed(0)

    batch = 2
    heads = 4

    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, heads, seq_len, head_dim)
    v = torch.randn(batch, heads, seq_len, head_dim)

    out_manual = manual_attention(q, k, v)
    out_torch = torch_sdpa_attention(q, k, v)

    torch.testing.assert_close(out_manual, out_torch, rtol=1e-4, atol=1e-4)