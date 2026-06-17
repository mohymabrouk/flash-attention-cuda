# Math Notes

## Standard Attention

Given:

```text
Q, K, V ∈ R^(B × H × N × D)
```

Where:

- B = batch size
- H = number of heads
- N = sequence length
- D = head dimension

Attention:

```text
Scores = QKᵀ / √D
P = softmax(Scores)
Output = PV
```

## Complexity

### Compute

```text
O(N²D)
```

### Memory

Attention matrix shape:

```text
N × N
```

Memory:

```text
O(N²)
```

This becomes the bottleneck for long-context transformers.

---

## Numerical Stability

Instead of:

```text
softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

Use:

```text
softmax(x_i) =
exp(x_i - max(x))
----------------
Σ exp(x_j - max(x))
```

to avoid overflow.

---

## FlashAttention Insight

Standard attention:

```text
QKᵀ
↓
store N×N matrix
↓
softmax
↓
PV
```

FlashAttention:

```text
load Q block
load K block
compute partial scores
update running softmax
accumulate output
discard temporary values
```

Avoids materializing the full attention matrix.

Memory complexity is dramatically reduced.