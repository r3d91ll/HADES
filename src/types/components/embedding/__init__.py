"""Embedding type definitions.

This module provides centralized type definitions for embedding vectors, 
adapters, configurations, and results used throughout the HADES system.
"""

# Import base types and protocols
from src.types.embedding.base import (
    EmbeddingAdapter,
    EmbeddingVector,
    EmbeddingModelType,
    AdapterType, 
    PoolingStrategy,
    EmbeddingAdapterRegistry,
    ModernBERTEmbeddingAdapter
)

# Import configuration types
from src.types.embedding.config import (
    EmbeddingConfig,
    EmbeddingAdapterConfig,
    PydanticEmbeddingConfig,
    PydanticEmbeddingAdapterConfig,
    VLLMAdapterConfig,
    OllamaAdapterConfig,
    HuggingFaceAdapterConfig
)

# Import result types
from src.types.embedding.results import (
    EmbeddingResult,
    BatchEmbeddingResult,
    EmbeddingValidationResult,
    PydanticEmbeddingResult,
    PydanticBatchEmbeddingRequest,
    PydanticBatchEmbeddingResult,
    EmbeddingResults,
    BatchResults,
    ValidationResults,
    AnyEmbeddingResult,
    AnyBatchResult
)

# Import from vector.py for backward compatibility
from src.types.embedding.vector import LegacyEmbeddingVector

__all__ = [
    # Base types and protocols
    "EmbeddingAdapter",
    "EmbeddingVector", 
    "EmbeddingModelType",
    "AdapterType",
    "PoolingStrategy",
    "EmbeddingAdapterRegistry",
    "ModernBERTEmbeddingAdapter",
    
    # Configuration types
    "EmbeddingConfig",
    "EmbeddingAdapterConfig",
    "PydanticEmbeddingConfig", 
    "PydanticEmbeddingAdapterConfig",
    "VLLMAdapterConfig",
    "OllamaAdapterConfig",
    "HuggingFaceAdapterConfig",
    
    # Result types
    "EmbeddingResult",
    "BatchEmbeddingResult",
    "EmbeddingValidationResult",
    "PydanticEmbeddingResult",
    "PydanticBatchEmbeddingRequest",
    "PydanticBatchEmbeddingResult",
    "EmbeddingResults",
    "BatchResults", 
    "ValidationResults",
    "AnyEmbeddingResult",
    "AnyBatchResult",
    
    # Legacy compatibility
    "LegacyEmbeddingVector"
]
