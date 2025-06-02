"""
Type definitions for the ISNE module.

This package contains type definitions, data models, and related utilities
for the ISNE (Inductive Shallow Node Embedding) implementation.

DEPRECATED: Use the centralized types from src.types.isne instead.
"""

import warnings

warnings.warn(
    "The types in src.isne.types are deprecated. Use src.types.isne instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from the centralized location
from src.types.isne import (
    DocumentType,
    RelationType,
    IngestDocument,
    DocumentRelation,
    LoaderResult,
    EmbeddingVector
)

__all__ = [
    'DocumentType',
    'RelationType',
    'IngestDocument',
    'DocumentRelation',
    'LoaderResult',
    'EmbeddingVector'
]
