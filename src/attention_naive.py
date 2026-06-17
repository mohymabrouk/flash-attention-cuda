import torch
import torch.nn.functional as F


def manual_attention(q, k, v, causal=False):
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1)
    scores = scores / (d ** 0.5)

    if causal:
        seq_len = q.shape[-2]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    return probs @ v


def torch_sdpa_attention(q, k, v, causal=False):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)