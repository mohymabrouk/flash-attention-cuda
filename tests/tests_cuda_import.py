import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_extension_imports():
    from src.attention_cuda import cuda_attention

    q = torch.randn(2, 4, 16, 64, device="cuda")
    k = torch.randn(2, 4, 16, 64, device="cuda")
    v = torch.randn(2, 4, 16, 64, device="cuda")

    out = cuda_attention(q, k, v)

    assert out.shape == q.shape
    assert out.device.type == "cuda"