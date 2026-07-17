# Mathematical Notes

This note derives the equations implemented by the project. It distinguishes
arithmetic complexity from memory traffic: FlashAttention is exact attention
and still performs quadratic work in the sequence length. Its central gain is
that it avoids writing and rereading the quadratic score/probability matrices.

## 1. Notation and scaled dot-product attention

For one batch element and one attention head, let

\[
Q \in \mathbb{R}^{N_q \times d},\qquad
K,V \in \mathbb{R}^{N_k \times d}.
\]

The project uses the common scale \(s=1/\sqrt d\) by default, although the API
accepts any finite explicit scale. For query row \(i\) and key row \(j\),

\[
S_{ij}=sQ_iK_j^\mathsf{T},\qquad
P_{ij}=\frac{\exp(S_{ij})}{\sum_{t=1}^{N_k}\exp(S_{it})},\qquad
O_i=\sum_{j=1}^{N_k}P_{ij}V_j.
\]

For causal self-attention, scores with \(j>i\) are treated as \(-\infty\), so
their probabilities are zero. This implementation requires \(N_q=N_k\) in the
causal case; non-causal cross-attention may use different lengths.

Materialized attention performs \(\Theta(N_qN_kd)\) arithmetic and creates an
\(N_q\times N_k\) score or probability matrix per batch and head. The custom
kernel retains the same asymptotic arithmetic but uses \(O(N_qd+N_q)\) output
and saved-statistic storage beyond the inputs, with only fixed-size on-chip
tiles during a kernel launch.

## 2. Stable softmax

Directly evaluating \(\exp(x_j)\) can overflow. For a row \(x\), subtracting
its maximum leaves softmax unchanged:

\[
m=\max_j x_j,\qquad
\operatorname{softmax}(x)_j=
\frac{\exp(x_j-m)}{\sum_t\exp(x_t-m)}.
\]

Every exponent now has a non-positive argument. This prevents positive
overflow; very negative terms may underflow to zero, which is usually the
desired finite-precision behavior for negligible probabilities.

## 3. Online softmax recurrence

The difficulty is that a streaming kernel does not know the final row maximum
when it processes the first key tile. Maintain three quantities after seeing a
prefix \(J\) of keys:

\[
m_J=\max_{j\in J}S_{ij},\qquad
\ell_J=\sum_{j\in J}\exp(S_{ij}-m_J),
\]

and an unnormalized value accumulator

\[
A_J=\sum_{j\in J}\exp(S_{ij}-m_J)V_j.
\]

Suppose the next tile contains score set \(T\). Define

\[
m' = \max\!\left(m_J,\max_{j\in T}S_{ij}\right),
\]

\[
\ell' = \exp(m_J-m')\ell_J
       + \sum_{j\in T}\exp(S_{ij}-m'),
\]

\[
A' = \exp(m_J-m')A_J
    + \sum_{j\in T}\exp(S_{ij}-m')V_j.
\]

After all tiles, \(O_i=A'/\ell'\). The forward pass also stores

\[
\operatorname{LSE}_i=m'+\log\ell',
\]

which is the row log-sum-exp and is sufficient to reconstruct probabilities in
the backward pass.

### Why the recurrence is correct

For an old element \(j\in J\),

\[
\exp(S_{ij}-m')=
\exp(S_{ij}-m_J)\exp(m_J-m').
\]

Therefore multiplying both old accumulators by \(\exp(m_J-m')\) changes their
reference maximum without changing the represented sums. Adding the new terms
establishes the same invariant for \(J\cup T\). Induction from the empty prefix
proves that the final ratio is ordinary softmax attention in real arithmetic.
Floating-point evaluation is not bitwise identical because reductions and
rescaling are reassociated.

The CUDA kernel applies this recurrence one valid key row at a time inside a
tile. It initializes \(m=-\infty\), \(\ell=0\), and \(A=0\). The first valid
score explicitly uses a zero rescaling factor; this avoids evaluating the
undefined expression \(-\infty-(-\infty)\). Causally masked entries are skipped
rather than exponentiated.

## 4. Tile interpretation

The native forward launch uses a query tile of four rows (one row per warp) and
a key/value tile of 16 rows. A block repeatedly performs:

1. cooperatively load a \(16\times d\) K tile and V tile into FP32 shared memory;
2. compute warp-local query/key dot products and reduce them with warp shuffles;
3. update the warp-local online maximum, normalizer, and value numerator;
4. synchronize before the shared-memory tile is overwritten;
5. normalize and store output plus LSE after the last tile.

Tile sizes change the machine schedule, not the mathematical result. Sequence
tails are predicated, and head-dimension tails are handled by striding lanes in
steps of the 32-thread warp size.

## 5. Backward equations

Let \(G=\partial L/\partial O\) be the upstream gradient. For each query row,
define the scalar

\[
\delta_i = G_i O_i^\mathsf{T}
          = \sum_r G_{ir}O_{ir}.
\]

The probability can be recomputed without storing \(P\):

\[
P_{ij}=\exp(S_{ij}-\operatorname{LSE}_i).
\]

The intermediate derivatives are

\[
\frac{\partial L}{\partial P_{ij}}=G_iV_j^\mathsf{T},
\]

\[
\frac{\partial L}{\partial S_{ij}}=
P_{ij}\left(G_iV_j^\mathsf{T}-\delta_i\right).
\]

Writing this last quantity as \(D_{ij}\), the input gradients are

\[
\frac{\partial L}{\partial Q}=sDK,\qquad
\frac{\partial L}{\partial K}=sD^\mathsf{T}Q,\qquad
\frac{\partial L}{\partial V}=P^\mathsf{T}G.
\]

The implementation first computes all \(\delta_i\). A query-major streaming
kernel then owns each dQ row, while a key-major streaming kernel owns each dK
and dV row. Both recompute scores and probabilities from Q, K, V, and LSE. This
raises arithmetic relative to saving \(P\), but avoids quadratic saved state and
avoids atomics because every output-gradient row has exactly one owning warp.

## 6. Precision model

Input and output tensors may be FP32, FP16, or BF16. Dot products, online state,
shared-memory tiles, LSE, and backward accumulators use FP32. Results and input
gradients are cast to the input dtype only at their final stores.

FP32 accumulation reduces, but does not eliminate, differences from PyTorch:

- dot products and sums have a different reduction order;
- `expf` and fused instructions may differ between backends or GPU generations;
- FP16 and BF16 round at output stores;
- causal masking changes which operations execute;
- PyTorch SDPA may select a different fused backend unless the math backend is
  explicitly forced in a test.

Correctness tests therefore use dtype- and problem-size-aware tolerances and
check forward values, all three first-order gradients, and finiteness. Agreement
within a justified tolerance is evidence of numerical correctness; bitwise
identity is neither expected nor claimed.

## 7. Arithmetic and auxiliary-memory counts

Ignoring softmax scalar operations, the forward pass performs approximately
four floating-point operations per \((i,j,r)\): two for \(QK^\mathsf{T}\) and
two for \(PV\). A common non-causal estimate is therefore

\[
\operatorname{FLOPs}_{\text{fwd}}\approx4BHN_qN_kd.
\]

Causal self-attention processes roughly half the score pairs, so its leading
count is approximately half as large. These estimates are useful for effective
TFLOP/s, but they are not a hardware-instruction count and omit exponentials,
comparisons, casts, address calculations, and reductions.

The materialized reference allocates \(\Theta(BHN_qN_k)\) scores/probabilities.
The streaming native path saves only output and one FP32 LSE value per query,
plus one FP32 delta value per query during backward. Shared-memory and register
storage are fixed by tile size and \(d\), not by the full sequence length.

## Primary references

- Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with
  IO-Awareness](https://arxiv.org/abs/2205.14135), NeurIPS 2022.
- Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work
  Partitioning](https://arxiv.org/abs/2307.08691), ICLR 2024.
- Milakov and Gimelshein, [Online normalizer calculation for
  softmax](https://arxiv.org/abs/1805.02867), 2018.
- Vaswani et al., [Attention Is All You
  Need](https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html),
  NeurIPS 2017.
- Goldberg, [What Every Computer Scientist Should Know About Floating-Point
  Arithmetic](https://doi.org/10.1145/103162.103163), ACM Computing Surveys 1991.

The full report in `report/` gives the IO model, derivations, experimental
protocol, and complete bibliography in substantially greater detail.
