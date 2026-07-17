#include <torch/extension.h>

#include <cmath>
#include <limits>
#include <string>
#include <tuple>

namespace py = pybind11;

std::tuple<torch::Tensor, torch::Tensor> flash_attention_cuda_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    bool causal,
    double scale);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
flash_attention_cuda_backward(
    const torch::Tensor& dout,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& out,
    const torch::Tensor& lse,
    bool causal,
    double scale);

namespace {

void check_supported_dtype(const torch::Tensor& tensor, const char* name) {
    const auto dtype = tensor.scalar_type();
    TORCH_CHECK(
        dtype == torch::kFloat32 || dtype == torch::kFloat16 ||
            dtype == torch::kBFloat16,
        name,
        " must have dtype float32, float16, or bfloat16, but got ",
        tensor.scalar_type());
}

void check_cuda_contiguous_4d(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.defined(), name, " must be a defined tensor");
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        tensor.layout() == torch::kStrided,
        name,
        " must use strided layout");
    TORCH_CHECK(tensor.dim() == 4, name, " must have shape [B, H, N, D]");
    TORCH_CHECK(
        tensor.is_contiguous(),
        name,
        " must be contiguous; call .contiguous() before invoking the native op");
    check_supported_dtype(tensor, name);
}

void check_scale(double scale) {
    TORCH_CHECK(std::isfinite(scale), "scale must be finite, but got ", scale);
    TORCH_CHECK(
        std::abs(scale) <=
            static_cast<double>(std::numeric_limits<float>::max()),
        "scale must be representable as float32, but got ",
        scale);
}

void check_qkv(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    bool causal,
    double scale) {
    check_cuda_contiguous_4d(q, "q");
    check_cuda_contiguous_4d(k, "k");
    check_cuda_contiguous_4d(v, "v");
    check_scale(scale);

    TORCH_CHECK(
        q.scalar_type() == k.scalar_type() &&
            q.scalar_type() == v.scalar_type(),
        "q, k, and v must have the same dtype");
    TORCH_CHECK(
        q.device() == k.device() && q.device() == v.device(),
        "q, k, and v must be on the same CUDA device");

    const auto batch = q.size(0);
    const auto heads = q.size(1);
    const auto query_length = q.size(2);
    const auto head_dim = q.size(3);
    const auto key_length = k.size(2);

    TORCH_CHECK(batch > 0, "batch size B must be positive");
    TORCH_CHECK(heads > 0, "head count H must be positive");
    TORCH_CHECK(query_length > 0, "query length Nq must be positive");
    TORCH_CHECK(key_length > 0, "key/value length Nk must be positive");
    TORCH_CHECK(
        head_dim >= 1 && head_dim <= 256,
        "head dimension D must satisfy 1 <= D <= 256, but got ",
        head_dim);

    TORCH_CHECK(
        k.size(0) == batch && v.size(0) == batch,
        "q, k, and v must have the same batch size");
    TORCH_CHECK(
        k.size(1) == heads && v.size(1) == heads,
        "q, k, and v must have the same number of heads");
    TORCH_CHECK(
        k.size(3) == head_dim && v.size(3) == head_dim,
        "q, k, and v must have the same head dimension D");
    TORCH_CHECK(
        v.size(2) == key_length,
        "k and v must have the same sequence length Nk");
    TORCH_CHECK(
        !causal || query_length == key_length,
        "causal attention requires Nq == Nk, but got Nq=",
        query_length,
        " and Nk=",
        key_length);
}

void check_like_q(
    const torch::Tensor& tensor,
    const torch::Tensor& q,
    const char* name) {
    check_cuda_contiguous_4d(tensor, name);
    TORCH_CHECK(
        tensor.scalar_type() == q.scalar_type(),
        name,
        " must have the same dtype as q");
    TORCH_CHECK(
        tensor.device() == q.device(),
        name,
        " must be on the same CUDA device as q");
    TORCH_CHECK(
        tensor.size(0) == q.size(0) && tensor.size(1) == q.size(1) &&
            tensor.size(2) == q.size(2) && tensor.size(3) == q.size(3),
        name,
        " must have the same shape as q");
}

void check_lse(const torch::Tensor& lse, const torch::Tensor& q) {
    TORCH_CHECK(lse.defined(), "lse must be a defined tensor");
    TORCH_CHECK(lse.is_cuda(), "lse must be a CUDA tensor");
    TORCH_CHECK(
        lse.layout() == torch::kStrided,
        "lse must use strided layout");
    TORCH_CHECK(lse.is_contiguous(), "lse must be contiguous");
    TORCH_CHECK(lse.scalar_type() == torch::kFloat32, "lse must be float32");
    TORCH_CHECK(
        lse.device() == q.device(),
        "lse must be on the same CUDA device as q");
    TORCH_CHECK(
        lse.dim() == 3 && lse.size(0) == q.size(0) &&
            lse.size(1) == q.size(1) && lse.size(2) == q.size(2),
        "lse must have shape [B, H, Nq]");
}

std::tuple<torch::Tensor, torch::Tensor> attention_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    bool causal,
    double scale) {
    check_qkv(q, k, v, causal, scale);
    return flash_attention_cuda_forward(q, k, v, causal, scale);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> attention_backward(
    const torch::Tensor& dout,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& out,
    const torch::Tensor& lse,
    bool causal,
    double scale) {
    check_qkv(q, k, v, causal, scale);
    check_like_q(dout, q, "dout");
    check_like_q(out, q, "out");
    check_lse(lse, q);
    return flash_attention_cuda_backward(
        dout, q, k, v, out, lse, causal, scale);
}

torch::Tensor cuda_attention_compat(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v) {
    TORCH_CHECK(
        q.dim() == 4 && q.size(3) > 0,
        "q must have shape [B, H, Nq, D] with D > 0");
    const double scale = 1.0 / std::sqrt(static_cast<double>(q.size(3)));
    return std::get<0>(attention_forward(q, k, v, false, scale));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &attention_forward,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("causal"),
        py::arg("scale"),
        "FlashAttention forward (CUDA)");
    m.def(
        "backward",
        &attention_backward,
        py::arg("dout"),
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("out"),
        py::arg("lse"),
        py::arg("causal"),
        py::arg("scale"),
        "FlashAttention backward (CUDA)");
    m.def(
        "cuda_attention",
        &cuda_attention_compat,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        "Compatibility wrapper for non-causal FlashAttention forward");
}
