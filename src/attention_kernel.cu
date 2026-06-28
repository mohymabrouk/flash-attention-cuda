#include <torch/extension.h>

torch::Tensor cuda_attention_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
) {
    return torch::zeros_like(q);
}