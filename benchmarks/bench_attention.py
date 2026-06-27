import csv
import time
from pathlib import Path

import torch

from src.attention_naive import manual_attention, torch_sdpa_attention


def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def run_one(method_name, fn, q, k, v, ref, warmup=10, steps=100):
    device = q.device.type
    batch, heads, seq_len, head_dim = q.shape

    for _ in range(warmup):
        fn(q, k, v)
    sync(device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    for _ in range(steps):
        out = fn(q, k, v)

    sync(device)
    end = time.perf_counter()

    latency = (end - start) / steps

    peak_memory_mb = None
    if device == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2

    max_error = (out - ref).abs().max().item()
    tokens_per_second = batch * heads * seq_len / latency

    return {
        "method": method_name,
        "device": device,
        "dtype": str(q.dtype).replace("torch.", ""),
        "batch": batch,
        "heads": heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "causal": False,
        "latency_s": latency,
        "tokens_per_second": tokens_per_second,
        "peak_memory_mb": peak_memory_mb,
        "max_error": max_error,
        "output_mean": out.mean().item(),
    }


def benchmark(seq_len=512, head_dim=64, batch=2, heads=8):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    q = torch.randn(batch, heads, seq_len, head_dim, device=device)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device)

    ref = torch_sdpa_attention(q, k, v)

    methods = {
        "manual_pytorch": manual_attention,
        "torch_sdpa": torch_sdpa_attention,
    }

    rows = []

    for name, fn in methods.items():
        result = run_one(name, fn, q, k, v, ref)
        rows.append(result)

    return rows


if __name__ == "__main__":
    all_rows = []

    for seq_len in [128, 256, 512, 1024]:
        all_rows.extend(benchmark(seq_len=seq_len))

    Path("results").mkdir(exist_ok=True)

    output_path = "results/benchmark_baseline.csv"

    fieldnames = [
        "method",
        "device",
        "dtype",
        "batch",
        "heads",
        "seq_len",
        "head_dim",
        "causal",
        "latency_s",
        "tokens_per_second",
        "peak_memory_mb",
        "max_error",
        "output_mean",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    for row in all_rows:
        print(row)

    print(f"\nSaved results to {output_path}")