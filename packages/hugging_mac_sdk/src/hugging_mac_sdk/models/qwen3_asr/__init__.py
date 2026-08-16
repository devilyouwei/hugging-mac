"""Qwen3-ASR Core ML model package."""

from .definition import (
    QWEN3_ASR_DEFINITION,
    QWEN3_ASR_MANIFEST,
    register_qwen3_asr,
)

__all__ = ["QWEN3_ASR_DEFINITION", "QWEN3_ASR_MANIFEST", "register_qwen3_asr"]
