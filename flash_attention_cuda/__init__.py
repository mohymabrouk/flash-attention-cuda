"""Public API for the educational FlashAttention CUDA extension."""

from .attention import (
    extension_available,
    extension_error,
    flash_attention,
)
from .reference import manual_attention, torch_sdpa_attention

__all__ = [
    "extension_available",
    "extension_error",
    "flash_attention",
    "manual_attention",
    "torch_sdpa_attention",
]

__version__ = "0.2.0"
