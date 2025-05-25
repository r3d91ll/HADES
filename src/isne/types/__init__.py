"""
Type definitions for the ISNE module.

This package contains type definitions, data models, and related utilities
for the ISNE (Inductive Shallow Node Embedding) implementation.
"""

from .models import (
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
