"""Embedding storage type definitions.

This module defines type annotations for embedding storage operations, including
embedding metadata, storage options, and vector search configurations.
"""

from typing import Dict, List, Any, Optional, TypedDict, Literal, Union
from src.types.common import EmbeddingVector, NodeID


class EmbeddingMetadata(TypedDict, total=False):
    """Metadata for embeddings in storage."""
    
    dimension: int
    """Dimensionality of the embedding vector."""
    
    model: str
    """Model used to generate the embedding."""
    
    version: str
    """Version of the embedding model."""
    
    created_at: str
    """Timestamp when the embedding was created."""
    
    document_id: str
    """ID of the document the embedding represents."""
    
    chunk_index: int
    """Index of the chunk if this is a chunk embedding."""
    
    embedding_type: str
    """Type of embedding (e.g., 'text', 'isne', 'image')."""


class EmbeddingIndexConfig(TypedDict, total=False):
    """Configuration for an embedding index."""
    
    name: str
    """Name of the index."""
    
    dimension: int
    """Dimensionality of vectors in this index."""
    
    metric: str
    """Distance metric to use ('cosine', 'euclidean', 'dot', etc.)."""
    
    dynamic: bool
    """Whether the index should be updated dynamically."""
    
    sparse: bool
    """Whether this is a sparse vector index."""


class EmbeddingStorageOptions(TypedDict, total=False):
    """Options for storing embeddings."""
    
    dimension: int
    """Dimensionality of the embedding vector."""
    
    distance_metric: str
    """Distance metric to use for similarity search."""
    
    return_values: bool
    """Whether to return embedding values in search results."""
    
    include_fields: List[str]
    """Fields to include in search results."""
    
    exclude_fields: List[str]
    """Fields to exclude from search results."""


class EmbeddingSearchOptions(TypedDict, total=False):
    """Options for searching embeddings."""
    
    k: int
    """Number of nearest neighbors to return."""
    
    threshold: float
    """Similarity threshold for including results."""
    
    include_metadata: bool
    """Whether to include metadata in results."""
    
    include_embeddings: bool
    """Whether to include embedding vectors in results."""
    
    filter: Dict[str, Any]
    """Filter to apply to search results."""
