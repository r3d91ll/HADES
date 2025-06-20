"""Embedding vector type definitions.

This module provides core type definitions for embedding vectors and operations.
This is now a compatibility layer - main types have been moved to other modules
in src.types.embedding for better organization.

DEPRECATED: Import from specific modules instead:
- src.types.embedding.base for core types and protocols
- src.types.embedding.config for configuration types  
- src.types.embedding.results for result types
"""

from typing import Any, Dict, List, Optional, TypedDict, Union
from datetime import datetime, timezone

# Import consolidated types for backward compatibility
from src.types.embedding.base import EmbeddingVector, EmbeddingModelType, AdapterType, PoolingStrategy
from src.types.embedding.config import (
    EmbeddingConfig,
    EmbeddingAdapterConfig,
    PydanticEmbeddingConfig,
    PydanticEmbeddingAdapterConfig
)
from src.types.embedding.results import (
    EmbeddingResult,
    BatchEmbeddingResult,
    PydanticEmbeddingResult,
    PydanticBatchEmbeddingResult
)

# Legacy type aliases for backward compatibility
LegacyEmbeddingVector = Union[List[float], bytes]  # Original definition

__all__ = [
    # Core types (from base)
    "EmbeddingVector",
    "EmbeddingModelType", 
    "AdapterType",
    "PoolingStrategy",
    
    # Configuration types (from config)
    "EmbeddingConfig",
    "EmbeddingAdapterConfig", 
    "PydanticEmbeddingConfig",
    "PydanticEmbeddingAdapterConfig",
    
    # Result types (from results)
    "EmbeddingResult",
    "BatchEmbeddingResult",
    "PydanticEmbeddingResult", 
    "PydanticBatchEmbeddingResult",
    
    # Legacy compatibility
    "LegacyEmbeddingVector"
]
