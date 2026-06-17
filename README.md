# FlashAttention CUDA From Scratch

Status: Work in progress.

A from-scratch CUDA/PyTorch implementation of memory-efficient attention using tiled computation and online softmax.

## Roadmap

- [ ] Naive PyTorch attention baseline
- [ ] Correctness tests vs PyTorch SDPA
- [ ] Naive CUDA attention forward
- [ ] FlashAttention-style tiled kernel
- [ ] Benchmarks for latency, memory, and numerical error
- [ ] Technical report and profiling notes

## Project Structure

```text
src/          implementation
tests/        correctness tests
benchmarks/   latency and memory benchmarks
docs/         design, math, benchmark notes
notebooks/    validation experiments
