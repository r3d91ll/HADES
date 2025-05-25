"""
Embedding adapter schemas for HADES-PathRAG.

This module defines Pydantic models for embedding adapter configurations
and interfaces, replacing the previous Protocol-based approach.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union, ClassVar

from pydantic import Field, field_validator

from ..common.base import BaseSchema
from ..common.types import EmbeddingVector, MetadataDict
from .models import EmbeddingConfig, EmbeddingModelType


class AdapterType(str, Enum):
    """Types of embedding adapters supported by the system."""
    HUGGINGFACE = "huggingface"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    MODERNBERT = "modernbert"
    OPENAI = "openai"
    COHERE = "cohere"
    VLLM = "vllm"
    CUSTOM = "custom"


class BaseAdapterConfig(BaseSchema):
    """Base configuration for embedding adapters."""
    
    adapter_type: AdapterType = Field(..., description="Type of adapter")
    model_name: str = Field(..., description="Name of the model to use")
    batch_size: int = Field(default=32, description="Batch size for embedding generation")
    metadata: MetadataDict = Field(default_factory=dict, description="Additional adapter metadata")
    
    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch size is positive."""
        if v <= 0:
            raise ValueError("Batch size must be positive")
        return v


class HuggingFaceAdapterConfig(BaseAdapterConfig):
    """Configuration for Hugging Face embedding adapter."""
    
    adapter_type: AdapterType = Field(default=AdapterType.HUGGINGFACE, 
                                      description="Type of adapter")
    cache_dir: Optional[str] = Field(default=None, 
                                    description="Directory for model caching")
    device: Optional[str] = Field(default=None, 
                                 description="Device to use (e.g., 'cuda:0', 'cpu')")
    model_kwargs: Dict[str, Any] = Field(default_factory=dict, 
                                        description="Additional model initialization kwargs")
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict, 
                                         description="Additional encoding kwargs")
    normalize: bool = Field(default=True, 
                           description="Whether to normalize vectors")


class SentenceTransformersAdapterConfig(BaseAdapterConfig):
    """Configuration for Sentence Transformers embedding adapter."""
    
    adapter_type: AdapterType = Field(default=AdapterType.SENTENCE_TRANSFORMERS, 
                                      description="Type of adapter")
    cache_dir: Optional[str] = Field(default=None, 
                                    description="Directory for model caching")
    device: Optional[str] = Field(default=None, 
                                 description="Device to use (e.g., 'cuda:0', 'cpu')")
    normalize: bool = Field(default=True, 
                           description="Whether to normalize vectors")
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict, 
                                         description="Additional encoding kwargs")


class ModernBERTAdapterConfig(BaseAdapterConfig):
    """Configuration for ModernBERT embedding adapter."""
    
    adapter_type: AdapterType = Field(default=AdapterType.MODERNBERT, 
                                      description="Type of adapter")
    api_key: Optional[str] = Field(default=None, 
                                  description="API key for ModernBERT service")
    api_base: str = Field(default="https://api.modernapi.com", 
                         description="Base URL for ModernBERT API")
    timeout: float = Field(default=60.0, 
                          description="Timeout for API calls in seconds")
    normalize: bool = Field(default=True, 
                           description="Whether to normalize vectors")


class AdapterResult(BaseSchema):
    """Result from an embedding adapter operation."""
    
    text_inputs: List[str] = Field(..., description="Original text inputs")
    embeddings: List[EmbeddingVector] = Field(..., description="Generated embedding vectors")
    model_name: str = Field(..., description="Name of the model used")
    adapter_type: AdapterType = Field(..., description="Type of adapter used")
    metadata: MetadataDict = Field(default_factory=dict, description="Additional result metadata")
    
    @field_validator("embeddings")
    @classmethod
    def validate_embeddings(cls, v: List[Union[List[float], np.ndarray]], values: Dict[str, Any]) -> List[Union[List[float], np.ndarray]]:
        """Validate embeddings length matches text inputs length."""
        text_inputs = values.data.get("text_inputs", [])
        if len(v) != len(text_inputs):
            raise ValueError(f"Number of embeddings ({len(v)}) must match number of text inputs ({len(text_inputs)})")
        return v
