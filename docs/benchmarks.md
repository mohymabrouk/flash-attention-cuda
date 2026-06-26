# Benchmarks

## Baseline Benchmark

Command:

```bash
python -m benchmarks.bench_attention
```

---

## Current Results

| Sequence Length | Latency |
|----------------|----------|
| 128 | 0.000666 s |
| 256 | 0.002878 s |
| 512 | 0.014200 s |
| 1024 | 0.058212 s |

---

## Observation

Latency grows approximately quadratically.

Example:

```text
128 → 256 ≈ 4x
256 → 512 ≈ 5x
512 → 1024 ≈ 4x
```

This reflects:

```text
O(N²)
```

attention complexity.

---

## Future Benchmark Matrix

### Sequence Lengths

```text
128
256
512
1024
2048
```

### Head Dimensions

```text
64
128
```

### Batch Sizes

```text
1
4
8
```

---

## Methods Compared

```text
Manual PyTorch Attention
PyTorch SDPA
Naive CUDA Attention
Flash-Style CUDA Attention
```

---

## Metrics

Measure:

```text
Latency
Peak Memory
Maximum Error
Throughput
```

---

## Goal

Demonstrate:

```text
Lower memory usage
Better scaling
Comparable numerical accuracy
```

relative to the baseline implementation.

The manual PyTorch implementation materializes the full N x N attention matrix.
PyTorch SDPA uses optimized kernels and is expected to be faster and more memory efficient on GPU.