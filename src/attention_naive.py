"""Backward-compatible imports for the original repository module path."""

from flash_attention_cuda.reference import manual_attention, torch_sdpa_attention

__all__ = ["manual_attention", "torch_sdpa_attention"]
