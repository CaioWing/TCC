"""Kernels and benchmark helpers for the TCC experiments."""

from .attention import (
    attention_eager,
    attention_sdpa,
    triton_flash_attention,
)

__all__ = [
    "attention_eager",
    "attention_sdpa",
    "triton_flash_attention",
]
