"""
CPU-optimized chunking implementation.

This module provides efficient CPU-based chunking algorithms
optimized for scenarios without GPU acceleration.
"""

from .processor import CPUChunker

__all__ = ['CPUChunker']