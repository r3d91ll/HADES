"""
Code-aware chunking implementation.

This module provides specialized chunking for source code
that respects programming language structure and syntax.
"""

from .processor import CodeChunker

__all__ = ['CodeChunker']