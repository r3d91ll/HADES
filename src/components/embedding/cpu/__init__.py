"""
CPU Embedding Component

This module provides CPU-based embedding functionality using
sentence transformers and other CPU-optimized models.
"""

from .processor import CPUEmbedder

__all__ = [
    "CPUEmbedder"
]