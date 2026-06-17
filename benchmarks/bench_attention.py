import time
import torch

from src.attention_naive import manual_attention


def benchmark(seq_len=512, head_dim=64):
    batch = 2
    heads = 8

    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, heads, seq_len, head_dim)
    v = torch.randn(batch, heads, seq_len, head_dim)

    for _ in range(10):
        manual_attention(q, k, v)

    start = time.perf_counter()

    for _ in range(100):
        manual_attention(q, k, v)

    end = time.perf_counter()

    print(f"seq_len={seq_len}")
    print(f"avg latency={(end-start)/100:.6f}s")


if __name__ == "__main__":
    benchmark()