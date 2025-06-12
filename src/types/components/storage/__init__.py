"""
Storage type definitions for HADES.

This module contains all type definitions, protocols, and interfaces
related to storage operations including document, graph, and vector storage.
"""

from .interfaces import (
    DocumentRepository,
    GraphRepository,
    VectorRepository,
    UnifiedRepository,
    AbstractUnifiedRepository,
)

__all__ = [
    "DocumentRepository",
    "GraphRepository", 
    "VectorRepository",
    "UnifiedRepository",
    "AbstractUnifiedRepository",
]