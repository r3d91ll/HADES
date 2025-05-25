"""
Pipeline module for HADES-PathRAG.

This module provides a modular pipeline architecture for document processing,
embedding generation, ISNE enhancement, and storage operations. It's designed
to support both batch processing for new datastores and incremental updates
to existing datastores.
"""

from .stages import PipelineStage
from .schema import DocumentSchema, ChunkSchema, ValidationResult

__all__ = [
    "PipelineStage",
    "DocumentSchema",
    "ChunkSchema",
    "ValidationResult",
]
