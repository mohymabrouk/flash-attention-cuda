import csv
import time
from pathlib import Path

import torch

from src.attention_naive import manual_attention, torch_sdpa_attention


def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def run_one(method_name, fn, q, k, v, warmup=10, steps=100):
    device = q.device.type

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

    return {
        "method": method_name,
        "latency_s": latency,
        "peak_memory_mb": peak_memory_mb,
        "output_mean": out.mean().item(),
    }


def benchmark(seq_len=512, head_dim=64, batch=2, heads=8):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    q = torch.randn(batch, heads, seq_len, head_dim, device=device)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device)

    methods = {
        "manual_pytorch": manual_attention,
        "torch_sdpa": torch_sdpa_attention,
    }

    rows = []

    for name, fn in methods.items():
        result = run_one(name, fn, q, k, v)
        result.update(
            {
                "device": device,
                "batch": batch,
                "heads": heads,
                "seq_len": seq_len,
                "head_dim": head_dim,
            }
        )
        rows.append(result)

    return rows


if __name__ == "__main__":
    all_rows = []

    for seq_len in [128, 256, 512, 1024]:
        all_rows.extend(benchmark(seq_len=seq_len))

    Path("results").mkdir(exist_ok=True)

    output_path = "results/benchmark_baseline.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    for row in all_rows:
        print(row)

    print(f"\nSaved results to {output_path}")