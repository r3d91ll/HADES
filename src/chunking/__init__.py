"""Chunking utilities for HADES-PathRAG.

This package provides the *public* API for splitting source documents into
semantic chunks.  For the moment we re-export the implementation that still
lives in ``src.ingest.processing.code_chunkers`` so existing functionality
continues to work while we incrementally migrate files.

Import pattern going forward::

    from src.ingest.chunking import chunk_code
"""

from __future__ import annotations

from .code_chunkers import chunk_code
from .code_chunkers.ast_chunker import chunk_python_code
# Text chunker (Chonky)
from .text_chunkers.chonky_chunker import chunk_text
from .text_chunkers.chonky_batch import chunk_text_batch

__all__: list[str] = [
    "chunk_code",
    "chunk_python_code",
    "chunk_text",
    "chunk_text_batch",
]
