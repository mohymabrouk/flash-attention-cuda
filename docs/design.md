# CUDA Design

## 1. Purpose and scope

This project implements exact scaled dot-product attention as a PyTorch CUDA
extension to expose the core ideas behind FlashAttention: IO-aware tiling,
stable online softmax, fusion, and backward recomputation. It is an educational
kernel, not a drop-in replacement for the production FlashAttention library.

The native path supports:

| Property | Contract |
|---|---|
| Tensor layout | contiguous `[B, H, N, D]` inside the extension |
| Dtypes | FP32, FP16, BF16 |
| Head dimension | `1 <= D <= 256`, including tails |
| Sequence shape | rectangular non-causal; square causal |
| Scale | finite explicit value; Python default `1/sqrt(D)` |
| Training | forward and first-order backward |
| Accumulation | FP32 dot products, softmax state, shared tiles, gradients |

The Python wrapper accepts non-contiguous tensors by making contiguous copies.
It intentionally does not implement dropout, arbitrary masks or bias, GQA/MQA
head broadcasting, different Q/K and V dimensions, second-order gradients, or
CPU execution in the native extension.

## 2. Data flow

Materialized attention writes the complete score/probability matrix to device
memory between operators:

```text
Q,K ── GEMM ──> [Nq × Nk scores] ── softmax ──> [Nq × Nk P] ── GEMM with V ──> O
```

The custom forward kernel keeps only a small K/V tile on chip:

```text
Q row in registers
      │
      ├── K/V tile 0 in shared memory ── online update ──┐
      ├── K/V tile 1 in shared memory ── online update ──┤
      └── ...                                             │
                                                         v
                                      normalized O row + FP32 LSE
```

The quadratic arithmetic remains. What disappears is the quadratic global
intermediate allocation and its associated high-bandwidth-memory traffic.

## 3. Forward launch geometry

Compile-time constants define the basic mapping:

| Quantity | Value | Role |
|---|---:|---|
| Threads per block | 128 | Four complete warps |
| Query rows per block | 4 | One row owned by each warp |
| Key/value tile rows | 16 | Cooperatively loaded by the block |
| Maximum values per lane | 8 | Covers `D <= 8 × 32 = 256` |

Grid coordinates identify a query-row tile, attention head, and batch element.
All threads participate in cooperative shared-memory loads and block barriers,
including a warp whose query row falls beyond a sequence tail. An inactive warp
does no arithmetic, but it may not return before the barriers.

For each key/value tile, 128 threads load `16 × D` K elements and the same
number of V elements into two dynamic shared-memory arrays, converting each
element to FP32. Dynamic shared-memory demand is

```text
2 × 16 × D × sizeof(float)
```

or at most 32 KiB at `D=256`, below the common 48 KiB default per-block limit.
The block then synchronizes, allowing four query warps to reuse the tile.

Within a warp, lane `l` owns dimensions `l, l+32, ...`. Each lane multiplies its
query register values with the corresponding shared K values. Warp shuffles
reduce those partial sums to a scalar score. Every lane receives that score,
updates the same running maximum and denominator, and updates its own slice of
the output numerator using V from shared memory.

After a tile is consumed, a second block barrier prevents a fast warp from
overwriting shared memory while another warp still reads it. After the final
tile, each lane divides its numerator slice by the online denominator and stores
the output in the input dtype. Lane zero writes one FP32 log-sum-exp value.

## 4. Online state and causal work

Each active warp starts with

```text
running_max = -infinity
running_sum = 0
output_numerator[:] = 0
```

For a valid score `score`, it evaluates:

```text
new_max = max(running_max, score)
old_weight = 0                       if this is the first valid score
             exp(running_max-new_max) otherwise
new_weight = exp(score-new_max)

output_numerator = old_weight * output_numerator + new_weight * V[j]
running_sum       = old_weight * running_sum       + new_weight
running_max       = new_max
```

Causal attention skips keys with `key_index > query_index`. Skipping is both
cheaper and numerically safer than performing arithmetic on `-infinity` masks.
Every causal row has at least its diagonal key because causal mode requires
equal non-empty query and key lengths.

## 5. Backward organization

The forward pass saves output O and row LSE, not probabilities. Backward
recomputes each needed probability as

```text
p_ij = exp(scale * dot(q_i, k_j) - lse_i)
```

This follows FlashAttention's compute-for-memory trade: repeated dot products
are preferable to storing and reading a potentially enormous probability
matrix.

### 5.1 Delta prepass

One block owns each query row and reduces

```text
delta_i = dot(dout_i, out_i)
```

into an FP32 tensor `[B, H, Nq]`. This scalar is the contraction needed by the
softmax Jacobian.

### 5.2 Query-major dQ kernel

The dQ launch mirrors the forward mapping. Four warps own four query-gradient
rows while K and V tiles of 16 rows move through shared memory. For each valid
pair `(i,j)`, a warp computes

```text
dP_ij = dot(dout_i, v_j)
dS_ij = p_ij * (dP_ij - delta_i)
dq_i += scale * dS_ij * k_j
```

One warp is the sole owner of `dq_i`; no atomic addition is required.

### 5.3 Key-major dK/dV kernel

The second gradient launch transposes ownership: four warps own four key rows,
and tiles of 16 Q and dO rows are loaded into shared memory. Each key warp
streams all applicable queries:

```text
dk_j += scale * dS_ij * q_i
dv_j += p_ij * dout_i
```

Again, each output row has one owner. This removes global atomics and their
order-dependent accumulation, at the cost of reading/recomputing score data in
both gradient kernels.

## 6. Boundary handling

Correctness does not depend on tile-aligned shapes:

- grid division rounds query/key row counts upward;
- shared loads outside the final sequence tile write zeros;
- score loops stop at the valid row count;
- lane-strided dimension loops stop at `D`;
- inactive row-owning warps still reach every block barrier;
- 64-bit offsets are used for flattened tensor indexing.

Tests deliberately cover sequence lengths immediately below, on, and above the
16-row tile boundary and head dimensions immediately below, on, and above warp
and common model boundaries.

## 7. Host boundary and stream safety

The C++ binding rejects undefined, CPU, non-strided, non-contiguous, wrong-rank,
empty, mixed-dtype, mixed-device, or shape-incompatible tensors with a useful
`TORCH_CHECK` message. It also verifies saved LSE/output/gradient shapes during
backward and validates the scale.

The CUDA launcher installs a device guard for Q's device and launches on
PyTorch's current CUDA stream. It checks the launch immediately. Tests include
a non-default-stream case because silently using CUDA's default stream can
produce intermittent races even when single-stream tests pass.

## 8. Deliberate performance choices

The design prioritizes clarity and correctness:

- FP32 shared K/V tiles make accumulation semantics obvious, but double the
  shared bytes relative to FP16 storage.
- Scalar online updates use `expf`; fast-math is not enabled.
- Warp shuffles replace shared-memory dot-product reductions.
- Query tiling provides modest K/V reuse without tensor-core fragments.
- Separate key-major and query-major backward kernels avoid atomics.
- Fixed tile sizes keep the implementation inspectable but are not optimal for
  every D, sequence length, or GPU generation.

A production implementation would specialize tile shapes by dtype/head
dimension/architecture, vectorize loads, use tensor cores, pipeline asynchronous
copies, overlap softmax with matrix multiplication, support more masks and
layouts, and tune register pressure against occupancy. FlashAttention-2 and -3
develop many of those ideas; this repository cites them but does not claim to
reproduce their performance.

## 9. Validation gates

A kernel is considered correct only after all of these pass on a CUDA machine:

1. forward comparison against the forced PyTorch math SDPA backend;
2. dQ, dK, and dV comparison for a common upstream gradient;
3. causal and non-causal cases across supported dtypes;
4. tile and head-dimension tails;
5. large-magnitude inputs and finiteness checks;
6. non-contiguous Python inputs;
7. a non-default CUDA stream;
8. invalid native-input rejection;
9. strict test invocation that fails if the extension/GPU is unavailable;
10. Compute Sanitizer before publishing performance results.

This repository's authoring environment has no CUDA toolchain or GPU. The
Kaggle commands in the README make the GPU gate explicit; skipped CUDA tests
are never treated as evidence that the custom kernel passed.
