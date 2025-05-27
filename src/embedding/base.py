"""Base interfaces and registry for embedding adapters.

This module defines the core interfaces for embedding generation and
provides a registry mechanism to manage different embedding adapters.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Protocol, TypeVar, Union, runtime_checkable, cast, Optional
import logging
import numpy as np

# Type definitions
# Note: List[float] is more JSON-serializable than np.ndarray
# Using a more flexible type definition to handle all embedding vector types
EmbeddingVector = Union[List[float], np.ndarray]  # Supports both list and ndarray return types

logger = logging.getLogger(__name__)

# Registry for adapters
_adapter_registry: Dict[str, type[EmbeddingAdapter]] = {}


@runtime_checkable
class EmbeddingAdapter(Protocol):
    """Protocol defining the interface for embedding adapters."""
    
    @abc.abstractmethod
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
        pass
    
    @abc.abstractmethod
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
        pass


def register_adapter(name: str, adapter_cls: type[EmbeddingAdapter]) -> None:
    """Register an embedding adapter implementation.
    
    Args:
        name: Name to register the adapter under
        adapter_cls: Adapter class to register
    """
    if name in _adapter_registry:
        logger.warning(f"Overwriting existing adapter registration for '{name}'")
    _adapter_registry[name] = adapter_cls
    logger.debug(f"Registered embedding adapter '{name}'")


def get_adapter(name: str, **init_kwargs: Any) -> EmbeddingAdapter:
    """Get an instance of a registered embedding adapter.
    
    Args:
        name: Name of the adapter to instantiate
        **init_kwargs: Keyword arguments to pass to the adapter constructor
        
    Returns:
        Instance of the requested adapter
        
    Raises:
        KeyError: If no adapter is registered with the given name
    """
    if name not in _adapter_registry:
        available = ", ".join(_adapter_registry.keys())
        raise KeyError(f"No embedding adapter registered as '{name}'. Available: {available}")
    
    adapter_cls = _adapter_registry[name]
    return adapter_cls(**init_kwargs)
