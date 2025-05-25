"""
Embedding model schemas for HADES-PathRAG.

This module defines Pydantic models for embedding-related data structures,
including configuration models and result types for embedding operations.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import Field, field_validator

from ..common.base import BaseSchema
from ..common.types import EmbeddingVector, MetadataDict


class EmbeddingModelType(str, Enum):
    """Types of embedding models supported by the system."""
    TRANSFORMER = "transformer"
    SENTENCE_TRANSFORMER = "sentence_transformer"
    SIF = "sif"
    VLLM = "vllm"
    MODERNBERT = "modernbert"
    CUSTOM = "custom"


class EmbeddingConfig(BaseSchema):
    """Configuration for embedding generation."""
    
    model_name: str = Field(..., description="Name of the embedding model to use")
    model_type: EmbeddingModelType = Field(..., description="Type of the embedding model")
    batch_size: int = Field(default=32, description="Batch size for embedding generation")
    max_length: Optional[int] = Field(default=512, description="Maximum sequence length")
    normalize: bool = Field(default=True, description="Whether to normalize vectors")
    device: Optional[str] = Field(default=None, description="Device to use (e.g., 'cuda:0', 'cpu')")
    cache_dir: Optional[str] = Field(default=None, description="Directory for model caching")
    metadata: MetadataDict = Field(default_factory=dict, description="Additional configuration metadata")
    
    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size is positive."""
        if v <= 0:
            raise ValueError("Batch size must be positive")
        return v


class EmbeddingResult(BaseSchema):
    """Result of an embedding operation."""
    
    text: str = Field(..., description="Original text that was embedded")
    embedding: EmbeddingVector = Field(..., description="Generated embedding vector")
    model_name: str = Field(..., description="Name of the model used")
    metadata: MetadataDict = Field(default_factory=dict, description="Additional result metadata")
    
    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: Union[List[float], np.ndarray]) -> Union[List[float], np.ndarray]:
        """Validate embedding is not empty."""
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("Embedding vector cannot be empty")
        if isinstance(v, np.ndarray) and v.size == 0:
            raise ValueError("Embedding vector cannot be empty")
        return v


class BatchEmbeddingRequest(BaseSchema):
    """Request for batch embedding generation."""
    
    texts: List[str] = Field(..., description="List of texts to embed")
    model_name: Optional[str] = Field(default=None, description="Override model name")
    config: Optional[EmbeddingConfig] = Field(default=None, description="Override default configuration")
    metadata: MetadataDict = Field(default_factory=dict, description="Additional request metadata")
    
    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        """Validate texts list is not empty."""
        if len(v) == 0:
            raise ValueError("Texts list cannot be empty")
        return v


class BatchEmbeddingResult(BaseSchema):
    """Result of a batch embedding operation."""
    
    embeddings: List[EmbeddingResult] = Field(..., description="List of embedding results")
    model_name: str = Field(..., description="Name of the model used")
    metadata: MetadataDict = Field(default_factory=dict, description="Additional result metadata")
