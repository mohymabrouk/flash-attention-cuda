#include <torch/extension.h>

torch::Tensor cuda_attention_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
);

torch::Tensor cuda_attention(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
) {
    return cuda_attention_forward(q, k, v);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_attention", &cuda_attention, "Minimal CUDA attention forward");
}