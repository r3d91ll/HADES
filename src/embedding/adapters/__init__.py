"""Embedding adapters for different model backends.

This module contains implementations of the EmbeddingAdapter interface
for different model engines including vLLM, CPU, and various transformer models.

Transformer models supported:
- ModernBERT: General purpose text embedding model
- CodeBERT: Specialized for code and technical documentation
"""

from __future__ import annotations

# Import all adapters so they register themselves
from .cpu_adapter import CPUEmbeddingAdapter
from .vllm_adapter import VLLMEmbeddingAdapter
from .encoder_adapter import EncoderEmbeddingAdapter

# For backward compatibility, make ModernBERTEmbeddingAdapter an alias for EncoderEmbeddingAdapter
ModernBERTEmbeddingAdapter = EncoderEmbeddingAdapter

# Re-export adapter classes
__all__: list[str] = [
    "CPUEmbeddingAdapter",
    "VLLMEmbeddingAdapter",
    "EncoderEmbeddingAdapter",
    "ModernBERTEmbeddingAdapter",  # For backward compatibility
]
