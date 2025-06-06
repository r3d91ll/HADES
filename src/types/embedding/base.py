"""
Base type definitions for embedding adapters.

This module provides the core type definitions for embedding adapters,
protocols, and configuration structures.
"""

from typing import Any, Dict, List, Protocol, TypeVar, Union, runtime_checkable, Optional
from abc import abstractmethod
from enum import Enum
import numpy as np

# Type variable for adapter subclasses
T = TypeVar('T', bound='EmbeddingAdapter')

# Comprehensive embedding vector type that supports all formats
EmbeddingVector = Union[List[float], np.ndarray, bytes]


class EmbeddingModelType(str, Enum):
    """Types of embedding models supported by the system."""
    
    TRANSFORMER = "transformer"
    SENTENCE_TRANSFORMER = "sentence_transformer"
    SIF = "sif"
    VLLM = "vllm"
    MODERNBERT = "modernbert"
    OPENAI = "openai"
    COHERE = "cohere"
    CUSTOM = "custom"


class AdapterType(str, Enum):
    """Types of embedding adapters supported by the system."""
    
    HUGGINGFACE = "huggingface"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    MODERNBERT = "modernbert"
    OPENAI = "openai"
    COHERE = "cohere"
    VLLM = "vllm"
    OLLAMA = "ollama"
    CPU = "cpu"
    ENCODER = "encoder"
    CUSTOM = "custom"


class PoolingStrategy(str, Enum):
    """Pooling strategies for embedding generation."""
    
    MEAN = "mean"
    CLS = "cls"
    MAX = "max"
    WEIGHTED_MEAN = "weighted_mean"
    FIRST_LAST_AVG = "first_last_avg"


@runtime_checkable
class EmbeddingAdapter(Protocol):
    """Protocol defining the interface for embedding adapters."""
    
    @abstractmethod
    async def embed(self, texts: List[str], **kwargs: Any) -> List[EmbeddingVector]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            List of embedding vectors, one for each input text
            
        Raises:
            RuntimeError: If the embedding operation fails
        """
        ...
    
    @abstractmethod
    async def embed_single(self, text: str, **kwargs: Any) -> EmbeddingVector:
        """Generate an embedding for a single text.
        
        Args:
            text: Text string to embed
            **kwargs: Additional parameters to pass to the model
            
        Returns:
            Embedding vector for the input text
            
        Raises:
            RuntimeError: If the embedding operation fails
        """
        ...


# Registry type for embedding adapters
EmbeddingAdapterRegistry = Dict[str, type[EmbeddingAdapter]]

# Type aliases for backward compatibility
ModernBERTEmbeddingAdapter = EmbeddingAdapter  # For backward compatibility