"""
Result type definitions for embedding operations.

This module provides type definitions for embedding results, batch operations,
and validation results.
"""

from typing import Any, Dict, List, Optional, TypedDict, Union
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import numpy as np

from src.types.embedding.base import EmbeddingVector


class EmbeddingResult(TypedDict, total=False):
    """Result of embedding generation (TypedDict version)."""
    
    text_id: str  # Identifier for the embedded text
    text: Optional[str]  # The text that was embedded (may be omitted to save space)
    vector: EmbeddingVector  # The embedding vector
    model_name: str  # Name of the model used to generate the embedding
    adapter_name: str  # Name of the adapter used to generate the embedding
    vector_dim: int  # Dimension of the embedding vector
    is_normalized: bool  # Whether the vector is L2-normalized
    truncated: bool  # Whether the input was truncated due to length
    created_at: Optional[Union[str, datetime]]  # Timestamp when the embedding was generated
    processing_time: Optional[float]  # Time taken to generate the embedding (seconds)
    token_count: Optional[int]  # Number of tokens in the input
    metadata: Dict[str, Any]  # Additional metadata about the embedding


class BatchEmbeddingResult(TypedDict):
    """Result of batch embedding generation."""
    
    embeddings: List[EmbeddingResult]  # List of individual embedding results
    total_count: int  # Total number of embeddings generated
    successful_count: int  # Number of successfully generated embeddings
    failed_count: int  # Number of failed embedding generations
    total_processing_time: float  # Total time for the entire batch (seconds)
    batch_id: Optional[str]  # Unique identifier for the batch
    model_name: str  # Model used for the batch
    adapter_name: str  # Adapter used for the batch
    errors: List[Dict[str, Any]]  # List of errors that occurred


class EmbeddingValidationResult(TypedDict):
    """Result of embedding validation."""
    
    is_valid: bool  # Whether the embedding is valid
    vector_dimension: int  # Dimension of the embedding vector
    is_normalized: bool  # Whether the vector is normalized
    vector_norm: float  # L2 norm of the vector
    has_nan_values: bool  # Whether the vector contains NaN values
    has_inf_values: bool  # Whether the vector contains infinite values
    value_range: List[float]  # [min_value, max_value] in the vector
    errors: List[str]  # List of validation errors
    warnings: List[str]  # List of validation warnings


# Pydantic models for validation and serialization

class PydanticEmbeddingResult(BaseModel):
    """Result of an embedding operation (Pydantic version)."""
    
    text_id: str = Field(..., description="Identifier for the embedded text")
    text: str = Field(..., description="Original text that was embedded")
    vector: EmbeddingVector = Field(..., description="Generated embedding vector")
    model_name: str = Field(..., description="Name of the model used")
    adapter_name: str = Field(..., description="Name of the adapter used")
    vector_dim: int = Field(..., description="Dimension of the embedding vector")
    is_normalized: bool = Field(default=True, description="Whether the vector is L2-normalized")
    truncated: bool = Field(default=False, description="Whether the input was truncated")
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp when the embedding was generated")
    processing_time: Optional[float] = Field(default=None, description="Time taken to generate the embedding")
    token_count: Optional[int] = Field(default=None, description="Number of tokens in the input")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional result metadata")
    
    @field_validator("vector")
    @classmethod
    def validate_embedding(cls, v: Union[List[float], np.ndarray, bytes]) -> Union[List[float], np.ndarray, bytes]:
        """Validate embedding is not empty."""
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("Embedding vector cannot be empty")
        if isinstance(v, np.ndarray) and v.size == 0:
            raise ValueError("Embedding vector cannot be empty")
        if isinstance(v, bytes) and len(v) == 0:
            raise ValueError("Embedding vector cannot be empty")
        return v
    
    @field_validator("vector_dim")
    @classmethod
    def validate_vector_dim(cls, v: int) -> int:
        """Validate vector dimension is positive."""
        if v <= 0:
            raise ValueError("Vector dimension must be positive")
        return v
    
    class Config:
        extra = "allow"
        arbitrary_types_allowed = True


class PydanticBatchEmbeddingRequest(BaseModel):
    """Request for batch embedding generation (Pydantic version)."""
    
    texts: List[str] = Field(..., description="List of texts to embed")
    text_ids: Optional[List[str]] = Field(default=None, description="Optional list of IDs for the texts")
    model_name: Optional[str] = Field(default=None, description="Override model name")
    adapter_name: Optional[str] = Field(default=None, description="Override adapter name")
    batch_id: Optional[str] = Field(default=None, description="Unique identifier for the batch")
    config_override: Optional[Dict[str, Any]] = Field(default=None, description="Override default configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional request metadata")
    
    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        """Validate texts list is not empty."""
        if len(v) == 0:
            raise ValueError("Texts list cannot be empty")
        return v
    
    @field_validator("text_ids")
    @classmethod
    def validate_text_ids(cls, v: Optional[List[str]], info: Any) -> Optional[List[str]]:
        """Validate text_ids matches texts length if provided."""
        if v is not None and "texts" in info.data:
            if len(v) != len(info.data["texts"]):
                raise ValueError("text_ids length must match texts length")
        return v
    
    class Config:
        extra = "allow"
        arbitrary_types_allowed = True


class PydanticBatchEmbeddingResult(BaseModel):
    """Result of batch embedding generation (Pydantic version)."""
    
    embeddings: List[PydanticEmbeddingResult] = Field(..., description="List of individual embedding results")
    total_count: int = Field(..., description="Total number of embeddings requested")
    successful_count: int = Field(..., description="Number of successfully generated embeddings")
    failed_count: int = Field(default=0, description="Number of failed embedding generations")
    total_processing_time: float = Field(..., description="Total time for the entire batch")
    batch_id: Optional[str] = Field(default=None, description="Unique identifier for the batch")
    model_name: str = Field(..., description="Model used for the batch")
    adapter_name: str = Field(..., description="Adapter used for the batch")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="List of errors that occurred")
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp when the batch was processed")
    
    @field_validator("successful_count", "failed_count")
    @classmethod
    def validate_counts(cls, v: int) -> int:
        """Validate counts are non-negative."""
        if v < 0:
            raise ValueError("Count must be non-negative")
        return v
    
    class Config:
        extra = "allow"
        arbitrary_types_allowed = True


# Type aliases for commonly used result types
EmbeddingResults = List[EmbeddingResult]
BatchResults = List[BatchEmbeddingResult]
ValidationResults = List[EmbeddingValidationResult]

# Union types for flexibility
AnyEmbeddingResult = Union[EmbeddingResult, PydanticEmbeddingResult]
AnyBatchResult = Union[BatchEmbeddingResult, PydanticBatchEmbeddingResult]