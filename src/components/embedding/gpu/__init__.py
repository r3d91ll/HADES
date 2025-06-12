"""
GPU Embedding Component

This module provides high-throughput GPU-based embedding functionality
using model engine abstraction for efficient batch processing.
"""

from .processor import GPUEmbedder

__all__ = [
    "GPUEmbedder"
]