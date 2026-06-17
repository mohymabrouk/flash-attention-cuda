import torch

from src.attention_naive import manual_attention, torch_sdpa_attention


def test_manual_attention_matches_sdpa():
    torch.manual_seed(0)

    batch = 2
    heads = 4
    seq_len = 128
    head_dim = 64

    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, heads, seq_len, head_dim)
    v = torch.randn(batch, heads, seq_len, head_dim)

    out_manual = manual_attention(q, k, v, causal=False)
    out_torch = torch_sdpa_attention(q, k, v, causal=False)

    torch.testing.assert_close(out_manual, out_torch, rtol=1e-4, atol=1e-4)