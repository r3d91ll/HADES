"""
Configuration type definitions for embedding operations.

This module provides configuration classes and TypedDict definitions
for embedding adapter configurations and embedding generation settings.
"""

from typing import Any, Dict, List, Optional, TypedDict, Union
from pydantic import BaseModel, Field, field_validator

from src.types.embedding.base import EmbeddingModelType, AdapterType, PoolingStrategy


class EmbeddingConfig(TypedDict, total=False):
    """Configuration for embedding generation (TypedDict version)."""
    
    adapter_name: str  # Name of the embedding adapter to use
    model_name: str  # Name of the specific model within the adapter
    max_length: int  # Maximum input length for the embedding model
    pooling_strategy: str  # Strategy for pooling token embeddings
    batch_size: int  # Batch size for embedding generation
    device: str  # Device to run embeddings on ("cpu", "cuda:0", etc.)
    normalize_embeddings: bool  # Whether to L2-normalize embeddings
    cache_embeddings: bool  # Whether to cache embeddings to avoid recomputation
    cache_dir: Optional[str]  # Directory to store embedding cache
    temperature: Optional[float]  # Temperature for generation (if applicable)
    top_k: Optional[int]  # Top-k sampling (if applicable)
    top_p: Optional[float]  # Top-p sampling (if applicable)


class EmbeddingAdapterConfig(TypedDict, total=False):
    """Configuration for a specific embedding adapter (TypedDict version)."""
    
    name: str  # Name of the embedding adapter
    description: str  # Description of the embedding adapter
    adapter_type: str  # Type of adapter (huggingface, vllm, etc.)
    default_model: str  # Default model to use for this adapter
    supported_models: List[str]  # List of models supported by this adapter
    max_input_length: int  # Maximum input length supported by the adapter
    vector_dimension: int  # Dimension of the output embedding vectors
    supports_batching: bool  # Whether the adapter supports batched embedding generation
    default_pooling: str  # Default pooling strategy
    recommended_settings: Dict[str, Any]  # Recommended settings for different use cases
    api_endpoint: Optional[str]  # API endpoint (if applicable)
    api_key: Optional[str]  # API key (if applicable)
    timeout: Optional[int]  # Request timeout in seconds


# Pydantic models for validation and serialization

class PydanticEmbeddingConfig(BaseModel):
    """Configuration for embedding generation (Pydantic version)."""
    
    adapter_name: str = Field(..., description="Name of the embedding adapter to use")
    model_name: str = Field(..., description="Name of the specific model within the adapter")
    model_type: EmbeddingModelType = Field(default=EmbeddingModelType.TRANSFORMER, description="Type of the embedding model")
    max_length: int = Field(default=512, description="Maximum input length for the embedding model")
    pooling_strategy: PoolingStrategy = Field(default=PoolingStrategy.MEAN, description="Strategy for pooling token embeddings")
    batch_size: int = Field(default=32, description="Batch size for embedding generation")
    device: Optional[str] = Field(default=None, description="Device to run embeddings on")
    normalize_embeddings: bool = Field(default=True, description="Whether to L2-normalize embeddings")
    cache_embeddings: bool = Field(default=False, description="Whether to cache embeddings")
    cache_dir: Optional[str] = Field(default=None, description="Directory to store embedding cache")
    temperature: Optional[float] = Field(default=None, description="Temperature for generation")
    top_k: Optional[int] = Field(default=None, description="Top-k sampling")
    top_p: Optional[float] = Field(default=None, description="Top-p sampling")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration metadata")
    
    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size is positive."""
        if v <= 0:
            raise ValueError("Batch size must be positive")
        return v
    
    @field_validator("max_length")
    @classmethod
    def validate_max_length(cls, v: int) -> int:
        """Validate max length is positive."""
        if v <= 0:
            raise ValueError("Max length must be positive")
        return v
    
    class Config:
        extra = "allow"
        arbitrary_types_allowed = True


class PydanticEmbeddingAdapterConfig(BaseModel):
    """Configuration for a specific embedding adapter (Pydantic version)."""
    
    name: str = Field(..., description="Name of the embedding adapter")
    description: str = Field(default="", description="Description of the embedding adapter")
    adapter_type: AdapterType = Field(..., description="Type of adapter")
    default_model: str = Field(..., description="Default model to use for this adapter")
    supported_models: List[str] = Field(default_factory=list, description="List of models supported by this adapter")
    max_input_length: int = Field(default=512, description="Maximum input length supported by the adapter")
    vector_dimension: int = Field(..., description="Dimension of the output embedding vectors")
    supports_batching: bool = Field(default=True, description="Whether the adapter supports batched embedding generation")
    default_pooling: PoolingStrategy = Field(default=PoolingStrategy.MEAN, description="Default pooling strategy")
    recommended_settings: Dict[str, Any] = Field(default_factory=dict, description="Recommended settings for different use cases")
    api_endpoint: Optional[str] = Field(default=None, description="API endpoint (if applicable)")
    api_key: Optional[str] = Field(default=None, description="API key (if applicable)")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    
    @field_validator("vector_dimension")
    @classmethod
    def validate_vector_dimension(cls, v: int) -> int:
        """Validate vector dimension is positive."""
        if v <= 0:
            raise ValueError("Vector dimension must be positive")
        return v
    
    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout is positive."""
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v
    
    class Config:
        extra = "allow"
        arbitrary_types_allowed = True


# Configuration for specialized adapters

class VLLMAdapterConfig(TypedDict, total=False):
    """Configuration specific to VLLM adapters."""
    
    api_url: str  # VLLM API URL
    api_key: Optional[str]  # API key for authentication
    model_name: str  # Model name on VLLM server
    max_tokens: int  # Maximum tokens to generate
    temperature: float  # Sampling temperature
    top_p: float  # Top-p sampling
    timeout: int  # Request timeout


class OllamaAdapterConfig(TypedDict, total=False):
    """Configuration specific to Ollama adapters."""
    
    base_url: str  # Ollama base URL
    model_name: str  # Model name in Ollama
    timeout: int  # Request timeout
    keep_alive: str  # Keep alive setting
    options: Dict[str, Any]  # Additional Ollama options


class HuggingFaceAdapterConfig(TypedDict, total=False):
    """Configuration specific to Hugging Face adapters."""
    
    model_name: str  # Hugging Face model name
    revision: Optional[str]  # Model revision
    use_auth_token: Optional[str]  # Authentication token
    device_map: Optional[str]  # Device mapping strategy
    load_in_8bit: bool  # Whether to load in 8-bit mode
    load_in_4bit: bool  # Whether to load in 4-bit mode
    torch_dtype: Optional[str]  # PyTorch data type