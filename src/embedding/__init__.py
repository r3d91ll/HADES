"""Embedding utilities for HADES-PathRAG.

This module provides interfaces and implementations for generating vector
embeddings from text and code content. It abstracts away the specific
model engine used for embeddings (vLLM, Ollama, etc.) and provides
a unified interface for all embedding operations.
"""

from __future__ import annotations

from .base import (
    EmbeddingAdapter,
    EmbeddingVector,
    get_adapter,
    register_adapter,
)
from .batch import batch_embed
from .registry import get_adapter_by_name

__all__: list[str] = [
    "EmbeddingAdapter",
    "EmbeddingVector",
    "get_adapter_by_name",
    "get_adapter",
    "register_adapter",
    "batch_embed",
]
