"""
Component type definitions for HADES.

This package contains type contracts used for communication between
different components in the HADES pipeline.
"""

from .contracts import (
    DocumentChunk,
    ChunkEmbedding,
    EmbeddingInput,
    GraphEnhancementInput
)

__all__ = [
    "DocumentChunk",
    "ChunkEmbedding",
    "EmbeddingInput",
    "GraphEnhancementInput"
]