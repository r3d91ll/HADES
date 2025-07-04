"""
Embedding component factory.

This module provides factory functions for creating embedding components.
"""

import logging
from typing import Dict, Any, Optional, Protocol
from abc import abstractmethod

from src.types.common import EmbeddingVector
from src.components.registry import register_component

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Protocol for embedding components."""
    
    @abstractmethod
    def embed(self, text: str) -> EmbeddingVector:
        """Generate embedding for text."""
        ...
    
    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[EmbeddingVector]:
        """Generate embeddings for multiple texts."""
        ...
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get embedding dimension."""
        ...


class DummyEmbedder:
    """Dummy embedder for testing."""
    
    def __init__(self, dimension: int = 768):
        """Initialize dummy embedder."""
        self._dimension = dimension
    
    def embed(self, text: str) -> EmbeddingVector:
        """Generate dummy embedding."""
        import numpy as np
        return np.random.randn(self._dimension).tolist()
    
    def embed_batch(self, texts: list[str]) -> list[EmbeddingVector]:
        """Generate dummy embeddings."""
        return [self.embed(text) for text in texts]
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension


def create_embedder(embedder_type: str, config: Optional[Dict[str, Any]] = None) -> Embedder:
    """
    Create an embedder component.
    
    Args:
        embedder_type: Type of embedder (jina_v4, sentence_transformer, dummy)
        config: Configuration for the embedder
        
    Returns:
        Embedder instance
    """
    config = config or {}
    
    if embedder_type == "jina_v4":
        # Import here to avoid circular imports
        from src.jina_v4.jina_processor import JinaV4Processor
        return JinaV4Processor(config)
    
    elif embedder_type == "dummy":
        return DummyEmbedder(config.get("dimension", 768))
    
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")


# Register factory functions
register_component(
    "embedder",
    "factory",
    create_embedder,
    config={"embedder_type": "jina_v4"}
)


__all__ = [
    "Embedder",
    "DummyEmbedder",
    "create_embedder"
]