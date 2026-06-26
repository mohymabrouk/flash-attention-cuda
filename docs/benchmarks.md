# Benchmarks

This document reports baseline performance measurements for the reference PyTorch implementation before introducing custom CUDA kernels.

---

## Run

```bash
python -m benchmarks.bench_attention
```

---

## Baseline Configuration

| Parameter | Value |
|-----------|------:|
| Device | CPU |
| Batch Size | 2 |
| Heads | 8 |
| Head Dimension | 64 |
| Warmup Iterations | 10 |
| Timed Iterations | 100 |

---

## Results

| Sequence Length | Average Latency (s) |
|----------------:|--------------------:|
| 128 | 0.000666 |
| 256 | 0.002878 |
| 512 | 0.014200 |
| 1024 | 0.058212 |

---

## Scaling

Latency increases approximately quadratically with sequence length.

| Transition | Increase |
|------------|---------:|
| 128 → 256 | ≈ 4× |
| 256 → 512 | ≈ 5× |
| 512 → 1024 | ≈ 4× |

This is expected because standard attention has

```text
O(N²D)
```

time complexity and

```text
O(N²)
```

memory complexity.

---

## Current Implementation

Implemented methods:

- Manual PyTorch attention
- PyTorch Scaled Dot Product Attention (SDPA)

Planned implementations:

- ⏳ Naive CUDA attention
- ⏳ FlashAttention tiled CUDA kernel

---

## Metrics

The following metrics will be collected for every implementation.

| Metric | Description |
|---------|-------------|
| Latency | Average execution time |
| Peak GPU Memory | Maximum allocated memory |
| Throughput | Tokens per second |
| Maximum Error | Difference from PyTorch SDPA |

---

## Benchmark Matrix

### Sequence Lengths

- 128
- 256
- 512
- 1024
- 2048
- 4096

### Head Dimensions

- 64
- 128

### Batch Sizes

- 1
- 2
- 4
- 8

---

## Expected Outcome

Compared to the baseline implementation, the FlashAttention kernel should

- reduce GPU memory usage
- avoid materializing the full attention matrix
- scale better for long sequence lengths
- maintain numerical agreement with PyTorch SDPA

---

## Notes

The current baseline implementation explicitly computes

```text
QKᵀ
↓
Softmax
↓
PV
```

which requires storing the full attention matrix.

The FlashAttention implementation will instead compute attention block-by-block using shared-memory tiling and online softmax, avoiding the quadratic memory bottleneck.