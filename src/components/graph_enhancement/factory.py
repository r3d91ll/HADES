"""
Graph enhancement component factory.

This module provides factory functions for creating graph enhancement components.
"""

import logging
from typing import Dict, Any, Optional, Protocol, List
from abc import abstractmethod

from src.types.common import EmbeddingVector, NodeID
from src.types.components.contracts import GraphEnhancementInput
from src.components.registry import register_component

logger = logging.getLogger(__name__)


class GraphEnhancer(Protocol):
    """Protocol for graph enhancement components."""
    
    @abstractmethod
    def enhance(self, input_data: GraphEnhancementInput) -> Dict[str, Any]:
        """Enhance embeddings with graph context."""
        ...
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the enhancer."""
        ...
    
    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if enhancer is initialized."""
        ...


class DummyGraphEnhancer:
    """Dummy graph enhancer for testing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize dummy enhancer."""
        self.config = config or {}
        self._initialized = False
    
    def enhance(self, input_data: GraphEnhancementInput) -> Dict[str, Any]:
        """Return input embeddings unchanged."""
        return {
            "enhanced_embeddings": input_data.embeddings,
            "graph_context": {},
            "metadata": {"enhancer": "dummy"}
        }
    
    def initialize(self) -> bool:
        """Initialize the enhancer."""
        self._initialized = True
        return True
    
    @property
    def is_initialized(self) -> bool:
        """Check if enhancer is initialized."""
        return self._initialized


def create_graph_enhancer(
    enhancer_type: str,
    config: Optional[Dict[str, Any]] = None
) -> GraphEnhancer:
    """
    Create a graph enhancement component.
    
    Args:
        enhancer_type: Type of enhancer (isne, dummy)
        config: Configuration for the enhancer
        
    Returns:
        GraphEnhancer instance
    """
    config = config or {}
    
    if enhancer_type == "isne":
        # Would import actual ISNE enhancer
        # For now, return dummy
        logger.warning("ISNE enhancer not implemented, using dummy")
        return DummyGraphEnhancer(config)
    
    elif enhancer_type == "dummy":
        return DummyGraphEnhancer(config)
    
    else:
        raise ValueError(f"Unknown enhancer type: {enhancer_type}")


# Register factory functions
register_component(
    "graph_enhancer",
    "factory",
    create_graph_enhancer,
    config={"enhancer_type": "isne"}
)


__all__ = [
    "GraphEnhancer",
    "DummyGraphEnhancer", 
    "create_graph_enhancer"
]