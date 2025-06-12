"""
Chonky (GPU-accelerated) chunking implementation.

This module provides GPU-accelerated chunking using transformer models
and PyTorch for high-throughput text processing.
"""

from .processor import ChonkyChunker

__all__ = ['ChonkyChunker']