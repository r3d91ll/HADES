"""
RAG Strategy Component Factory

This module provides a factory for creating RAG strategy components.
Integrated with the global component registry system.
"""

import logging
from typing import Dict, Any, Optional, Type, Callable, Tuple

from src.types.components.protocols import RAGStrategy
from src.types.components.contracts import ComponentType
from .core.processor import CoreRAGProcessor
from .pathrag.processor import PathRAGProcessor

logger = logging.getLogger(__name__)

# Registry of available RAG strategy implementations
RAG_STRATEGY_REGISTRY: Dict[str, Tuple[Type[RAGStrategy], Callable[[Optional[Dict[str, Any]]], RAGStrategy]]] = {
    "core": (CoreRAGProcessor, lambda config: CoreRAGProcessor()),
    "pathrag": (PathRAGProcessor, lambda config: PathRAGProcessor()),
}

def _register_components() -> None:
    """Register RAG strategy components with the global registry."""
    try:
        from src.components.registry import register_component
        
        for name, (component_class, factory_func) in RAG_STRATEGY_REGISTRY.items():
            register_component(
                component_type=ComponentType.RAG_STRATEGY,
                name=name,
                component_class=component_class,
                factory_func=factory_func,
                description=f"RAG strategy component: {name}"
            )
        
        logger.info(f"Registered {len(RAG_STRATEGY_REGISTRY)} RAG strategy components with global registry")
        
    except ImportError:
        logger.warning("Global registry not available, using local registry only")
    except Exception as e:
        logger.error(f"Failed to register RAG strategy components: {e}")

# Auto-register components when module is imported
_register_components()

def create_rag_strategy(
    strategy_name: str, 
    config: Optional[Dict[str, Any]] = None
) -> RAGStrategy:
    """
    Create a RAG strategy component by name.
    
    Args:
        strategy_name: Name of the strategy to create (e.g., "core", "pathrag")
        config: Optional configuration dictionary
        
    Returns:
        RAGStrategy instance
        
    Raises:
        ValueError: If strategy_name is not recognized
        RuntimeError: If component creation fails
    """
    if strategy_name not in RAG_STRATEGY_REGISTRY:
        available = list(RAG_STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown RAG strategy: {strategy_name}. Available: {available}")
    
    try:
        component_class, factory_func = RAG_STRATEGY_REGISTRY[strategy_name]
        component = factory_func(config)
        
        # Configure if needed
        if config and hasattr(component, 'configure'):
            component.configure(config)
        
        logger.info(f"Created RAG strategy component: {strategy_name}")
        return component
        
    except Exception as e:
        logger.error(f"Failed to create RAG strategy {strategy_name}: {e}")
        raise RuntimeError(f"RAG strategy creation failed: {e}") from e