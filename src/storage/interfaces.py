"""Repository interfaces for HADES-PathRAG.

This module re-exports the repository interfaces from the centralized type system
and defines additional composite interfaces needed by the HADES-PathRAG system.

All concrete repository implementations should conform to these interfaces.
"""

# Standard library imports
from typing import Dict, Any, Protocol, runtime_checkable

# Import interfaces from centralized type system
from src.types.storage import (
    # Repository interfaces
    DocumentRepository,
    GraphRepository,
    VectorRepository,
    
    # Query types
    QueryFilter,
    SortOption,
    QueryOptions,
    VectorQueryOptions,
    HybridQueryOptions,
    QueryResult,
    VectorSearchResult,
    HybridSearchResult,
    TraversalResult,
    PathResult,
)


@runtime_checkable
class UnifiedRepository(DocumentRepository, GraphRepository, VectorRepository, Protocol):
    """Unified repository interface combining document, graph, and vector operations.
    
    This interface represents the complete functionality required for the HADES-PathRAG system.
    It combines the three core repository interfaces (Document, Graph, Vector) and adds
    additional methods needed for collection management.
    """
    
    def setup_collections(self) -> None:
        """Set up the necessary collections and indexes in the repository."""
        ...
    
    def collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collections in the repository.
        
        Returns:
            Dictionary with collection statistics
        """
        ...


# Export all types for use by implementation modules
__all__ = [
    # Repository interfaces
    'DocumentRepository',
    'GraphRepository',
    'VectorRepository',
    'UnifiedRepository',
    
    # Query types
    'QueryFilter',
    'SortOption',
    'QueryOptions',
    'VectorQueryOptions',
    'HybridQueryOptions',
    'QueryResult',
    'VectorSearchResult',
    'HybridSearchResult',
    'TraversalResult',
    'PathResult',
]
