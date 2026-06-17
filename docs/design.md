# Design

## Goal

Implement FlashAttention-style CUDA kernels from scratch.

Project stages:

```text
PyTorch baseline
↓
Naive CUDA attention
↓
FlashAttention-style tiled attention
↓
Benchmarking
```

---

## Repository Structure

```text
src/
├── attention_naive.py
├── attention_cuda.cpp
├── attention_kernel.cu

tests/
benchmarks/
docs/
```

---

## Current Baseline

Implemented:

```text
QKᵀ
↓
softmax
↓
AV
```

using PyTorch operations.

Validated against:

```python
torch.nn.functional.scaled_dot_product_attention
```

---

## CUDA Roadmap

### Stage 1

Build extension successfully.

Files:

```text
attention_cuda.cpp
attention_kernel.cu
setup.py
```

Goal:

```text
import custom CUDA module from Python
```

### Stage 2

Implement separate kernels:

```text
QKᵀ
softmax
AV
```

Correctness first.

Optimization later.

### Stage 3

Implement FlashAttention concepts:

```text
shared memory tiling
online softmax
streaming accumulation
```

---

## Success Criteria

The final kernel should:

- Match PyTorch numerically
- Reduce memory usage
- Scale to longer sequence lengths
- Demonstrate FlashAttention principles