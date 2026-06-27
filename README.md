# FlashAttention CUDA From Scratch

A from-scratch implementation of scaled dot-product attention, progressing from a PyTorch reference implementation to custom CUDA kernels and a FlashAttention-style tiled implementation.

The goal of this project is to understand and reproduce the core algorithmic ideas behind FlashAttention rather than wrapping existing implementations.

---

## Current Status

### Implemented

- Manual PyTorch scaled dot-product attention
- PyTorch SDPA reference implementation
- Correctness tests against PyTorch SDPA
- Benchmark framework (latency, throughput, memory, numerical error)

### Planned

- CUDA extension
- Naive CUDA forward kernel
- FlashAttention tiled kernel
- Performance profiling
- Technical report

---

## Quick Start

```bash
pip install -e .
pytest
python -m benchmarks.bench_attention
```

---

## Repository Structure

```text
src/
    attention_naive.py
    attention_cuda.cpp
    attention_kernel.cu

tests/
    test_correctness.py

benchmarks/
    bench_attention.py

docs/
    design.md
    math.md
    benchmarks.md

results/
    benchmark_baseline.csv
```

---

## Roadmap

- [x] PyTorch attention baseline
- [x] Correctness tests
- [x] Benchmark suite
- [ ] CUDA extension
- [ ] Naive CUDA attention
- [ ] FlashAttention-style tiled implementation
- [ ] Profiling and optimization
- [ ] Technical report

---

## References

- FlashAttention (Dao et al., 2022)
- FlashAttention-2 (Dao, 2023)
- PyTorch Scaled Dot Product Attention