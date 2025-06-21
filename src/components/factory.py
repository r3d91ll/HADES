"""
HADES Component Factory System

Provides factory functions for creating components of all types.
Integrates with the component registry system.
"""

import logging
from typing import Dict, Any, Optional

from src.types.components.protocols import (
    Embedder, GraphEnhancer, RAGStrategy
)
from src.types.components.contracts import ComponentType

# Import specialized factories
from .embedding.factory import create_embedding_component
from .graph_enhancement.factory import create_graph_enhancement_component
from .registry import get_component

logger = logging.getLogger(__name__)


def create_rag_strategy(
    strategy_name: str, 
    config: Optional[Dict[str, Any]] = None
) -> RAGStrategy:
    """
    Create a RAG strategy component by name.
    
    Args:
        strategy_name: Name of the strategy to create (e.g., "pathrag", "core")
        config: Optional configuration dictionary
        
    Returns:
        RAGStrategy instance
        
    Raises:
        ValueError: If strategy_name is not recognized
        RuntimeError: If component creation fails
    """
    try:
        # Use the registry to get the component
        strategy: RAGStrategy = get_component("rag_strategy", strategy_name, config)
        logger.info(f"Created RAG strategy: {strategy_name}")
        return strategy
        
    except Exception as e:
        logger.error(f"Failed to create RAG strategy {strategy_name}: {e}")
        raise RuntimeError(f"RAG strategy creation failed: {e}") from e


def create_embedder(
    embedder_name: str,
    config: Optional[Dict[str, Any]] = None
) -> Embedder:
    """
    Create an embedder component by name.
    
    Args:
        embedder_name: Name of the embedder to create
        config: Optional configuration dictionary
        
    Returns:
        Embedder instance
    """
    return create_embedding_component(embedder_name, config)


def create_graph_enhancer(
    enhancer_name: str,
    config: Optional[Dict[str, Any]] = None
) -> GraphEnhancer:
    """
    Create a graph enhancer component by name.
    
    Args:
        enhancer_name: Name of the enhancer to create
        config: Optional configuration dictionary
        
    Returns:
        GraphEnhancer instance
    """
    return create_graph_enhancement_component(enhancer_name, config)


# Factory registry for dynamic creation
FACTORY_REGISTRY = {
    ComponentType.RAG_STRATEGY: create_rag_strategy,
    ComponentType.EMBEDDING: create_embedder, 
    ComponentType.GRAPH_ENHANCEMENT: create_graph_enhancer,
}


def create_component(
    component_type: ComponentType,
    component_name: str,
    config: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Create any component using the factory registry.
    
    Args:
        component_type: Type of component to create
        component_name: Name of the specific component
        config: Optional configuration dictionary
        
    Returns:
        Component instance
        
    Raises:
        ValueError: If component_type is not supported
        RuntimeError: If component creation fails
    """
    if component_type not in FACTORY_REGISTRY:
        raise ValueError(f"Unsupported component type: {component_type}")
    
    factory_func = FACTORY_REGISTRY[component_type]
    return factory_func(component_name, config)