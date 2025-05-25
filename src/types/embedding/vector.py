"""Embedding vector type definitions.

This module provides TypedDict definitions for embedding vectors,
adapter configurations, and embedding results.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict, Union
from datetime import datetime


# Type alias for embedding vectors - either a list of floats or a bytes object
EmbeddingVector = Union[List[float], bytes]


class EmbeddingConfig(TypedDict, total=False):
    """Configuration for embedding generation."""
    
    adapter_name: str
    """Name of the embedding adapter to use."""
    
    model_name: str
    """Name of the specific model within the adapter."""
    
    max_length: int
    """Maximum input length for the embedding model."""
    
    pooling_strategy: Literal["mean", "cls", "max"]
    """Strategy for pooling token embeddings."""
    
    batch_size: int
    """Batch size for embedding generation."""
    
    device: str
    """Device to run embeddings on ("cpu", "cuda:0", etc.)."""
    
    normalize_embeddings: bool
    """Whether to L2-normalize embeddings."""
    
    cache_embeddings: bool
    """Whether to cache embeddings to avoid recomputation."""
    
    cache_dir: Optional[str]
    """Directory to store embedding cache."""


class EmbeddingAdapterConfig(TypedDict, total=False):
    """Configuration for a specific embedding adapter."""
    
    name: str
    """Name of the embedding adapter."""
    
    description: str
    """Description of the embedding adapter."""
    
    default_model: str
    """Default model to use for this adapter."""
    
    supported_models: List[str]
    """List of models supported by this adapter."""
    
    max_input_length: int
    """Maximum input length supported by the adapter."""
    
    vector_dimension: int
    """Dimension of the output embedding vectors."""
    
    supports_batching: bool
    """Whether the adapter supports batched embedding generation."""
    
    default_pooling: str
    """Default pooling strategy."""
    
    recommended_settings: Dict[str, Any]
    """Recommended settings for different use cases."""


class EmbeddingResult(TypedDict, total=False):
    """Result of embedding generation."""
    
    text_id: str
    """Identifier for the embedded text."""
    
    text: Optional[str]
    """The text that was embedded (may be omitted to save space)."""
    
    vector: EmbeddingVector
    """The embedding vector."""
    
    model_name: str
    """Name of the model used to generate the embedding."""
    
    adapter_name: str
    """Name of the adapter used to generate the embedding."""
    
    vector_dim: int
    """Dimension of the embedding vector."""
    
    is_normalized: bool
    """Whether the vector is L2-normalized."""
    
    truncated: bool
    """Whether the input was truncated due to length."""
    
    created_at: Optional[Union[str, datetime]]
    """Timestamp when the embedding was generated."""
    
    metadata: Dict[str, Any]
    """Additional metadata about the embedding."""
