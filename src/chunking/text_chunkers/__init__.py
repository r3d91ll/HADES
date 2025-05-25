from __future__ import annotations
"""Text chunkers package exposing the Chonky-based semantic splitter.

Public API:

    from src.ingest.chunking import chunk_text
    from src.ingest.chunking import chunk_text_batch
"""

from .chonky_chunker import chunk_text  # noqa: F401
from .chonky_batch import chunk_text_batch  # noqa: F401

__all__: list[str] = [
    "chunk_text",
    "chunk_text_batch",
]
