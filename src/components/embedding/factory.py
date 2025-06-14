"""
Embedding Component Factory

This module provides a factory for creating embedding components.
Integrated with the global component registry system.
"""

import logging
from typing import Dict, Any, Optional, Type, Callable, Tuple

from src.types.components.protocols import Embedder
from src.types.components.contracts import ComponentType
from .core.processor import CoreEmbedder
from .cpu.processor import CPUEmbedder
from .encoder.processor import EncoderEmbedder
from .gpu.processor import GPUEmbedder


logger = logging.getLogger(__name__)


# Registry of available embedding implementations
EMBEDDING_REGISTRY: Dict[str, Tuple[Type[Embedder], Callable[[Optional[Dict[str, Any]]], Embedder]]] = {
    "core": (CoreEmbedder, lambda config: CoreEmbedder(config=config)),
    "cpu": (CPUEmbedder, lambda config: CPUEmbedder(config=config)),
    "encoder": (EncoderEmbedder, lambda config: EncoderEmbedder(config=config)),
    "gpu": (GPUEmbedder, lambda config: GPUEmbedder(config=config)),
    "vllm": (GPUEmbedder, lambda config: GPUEmbedder(config=config)),  # GPU embedder supports VLLM
}


def _register_components():
    """Register embedding components with the global registry."""
    try:
        from src.components.registry import register_component
        
        for name, (component_class, factory_func) in EMBEDDING_REGISTRY.items():
            register_component(
                component_type=ComponentType.EMBEDDING,
                name=name,
                component_class=component_class,
                factory_func=factory_func,
                description=f"Embedding component: {name}"
            )
        
        logger.info(f"Registered {len(EMBEDDING_REGISTRY)} embedding components with global registry")
        
    except ImportError:
        logger.warning("Global registry not available, using local registry only")
    except Exception as e:
        logger.error(f"Failed to register embedding components: {e}")


# Auto-register components when module is imported
_register_components()


def create_embedding_component(
    component_name: str, 
    config: Optional[Dict[str, Any]] = None
) -> Embedder:
    """
    Create an embedding component by name.
    
    Args:
        component_name: Name of the component to create (e.g., "core", "cpu", "gpu", "encoder", "vllm")
        config: Optional configuration dictionary
        
    Returns:
        Embedder instance
        
    Raises:
        ValueError: If component_name is not recognized
        RuntimeError: If component creation fails
    """
    if component_name not in EMBEDDING_REGISTRY:
        available = list(EMBEDDING_REGISTRY.keys())
        raise ValueError(f"Unknown embedding component: {component_name}. Available: {available}")
    
    try:
        component_class, factory_func = EMBEDDING_REGISTRY[component_name]
        component = factory_func(config)
        
        logger.info(f"Created embedding component: {component_name}")
        return component
        
    except Exception as e:
        logger.error(f"Failed to create embedding component {component_name}: {e}")
        raise RuntimeError(f"Component creation failed: {e}") from e


def get_available_embedding_components() -> Dict[str, Type[Embedder]]:
    """
    Get all available embedding components.
    
    Returns:
        Dictionary mapping component names to their classes
    """
    return {name: cls for name, (cls, _) in EMBEDDING_REGISTRY.items()}


def register_embedding_component(
    name: str, 
    component_class: Type[Embedder],
    factory_func: Optional[Callable[[Optional[Dict[str, Any]]], Embedder]] = None
) -> None:
    """
    Register a new embedding component.
    
    Args:
        name: Name to register the component under
        component_class: Class implementing Embedder protocol
        factory_func: Optional factory function, defaults to simple constructor
    """
    if factory_func is None:
        # Create a safe factory function that handles different constructor signatures
        def safe_factory(config: Optional[Dict[str, Any]]) -> Embedder:
            try:
                return component_class(config=config)  # type: ignore
            except TypeError:
                # Fallback for classes that don't accept config parameter
                return component_class()  # type: ignore
        factory_func = safe_factory
    
    EMBEDDING_REGISTRY[name] = (component_class, factory_func)
    logger.info(f"Registered embedding component: {name}")


def is_embedding_component_available(name: str) -> bool:
    """
    Check if an embedding component is available.
    
    Args:
        name: Component name to check
        
    Returns:
        True if component is available, False otherwise
    """
    return name in EMBEDDING_REGISTRY