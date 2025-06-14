"""
Chunking Component Factory

This module provides a factory for creating chunking components.
Integrated with the global component registry system.
"""

import logging
from typing import Dict, Any, Optional, Type, Callable, Tuple

from src.types.components.protocols import Chunker
from src.types.components.contracts import ComponentType
from .core.processor import CoreChunker
from .chunkers.cpu.processor import CPUChunker
from .chunkers.text.processor import TextChunker
from .chunkers.code.processor import CodeChunker
from .chunkers.chonky.processor import ChonkyChunker


logger = logging.getLogger(__name__)


# Registry of available chunking implementations
CHUNKING_REGISTRY: Dict[str, Tuple[Type[Chunker], Callable[[Optional[Dict[str, Any]]], Chunker]]] = {
    "core": (CoreChunker, lambda config: CoreChunker(config=config)),
    "cpu": (CPUChunker, lambda config: CPUChunker(config=config)),
    "text": (TextChunker, lambda config: TextChunker(config=config)),
    "code": (CodeChunker, lambda config: CodeChunker(config=config)),
    "chonky": (ChonkyChunker, lambda config: ChonkyChunker(config=config)),
}


def _register_components():
    """Register chunking components with the global registry."""
    try:
        from src.components.registry import register_component
        
        for name, (component_class, factory_func) in CHUNKING_REGISTRY.items():
            register_component(
                component_type=ComponentType.CHUNKING,
                name=name,
                component_class=component_class,
                factory_func=factory_func,
                description=f"Chunking component: {name}"
            )
        
        logger.info(f"Registered {len(CHUNKING_REGISTRY)} chunking components with global registry")
        
    except ImportError:
        logger.warning("Global registry not available, using local registry only")
    except Exception as e:
        logger.error(f"Failed to register chunking components: {e}")


# Auto-register components when module is imported
_register_components()


def create_chunking_component(
    component_name: str, 
    config: Optional[Dict[str, Any]] = None
) -> Chunker:
    """
    Create a chunking component by name.
    
    Args:
        component_name: Name of the component to create (e.g., "core", "cpu", "text", "code", "chonky")
        config: Optional configuration dictionary
        
    Returns:
        Chunker instance
        
    Raises:
        ValueError: If component_name is not recognized
        RuntimeError: If component creation fails
    """
    if component_name not in CHUNKING_REGISTRY:
        available = list(CHUNKING_REGISTRY.keys())
        raise ValueError(f"Unknown chunking component: {component_name}. Available: {available}")
    
    try:
        component_class, factory_func = CHUNKING_REGISTRY[component_name]
        component = factory_func(config)
        
        logger.info(f"Created chunking component: {component_name}")
        return component
        
    except Exception as e:
        logger.error(f"Failed to create chunking component {component_name}: {e}")
        raise RuntimeError(f"Component creation failed: {e}") from e


def get_available_chunking_components() -> Dict[str, Type[Chunker]]:
    """
    Get all available chunking components.
    
    Returns:
        Dictionary mapping component names to their classes
    """
    return {name: cls for name, (cls, _) in CHUNKING_REGISTRY.items()}


def register_chunking_component(
    name: str, 
    component_class: Type[Chunker],
    factory_func: Optional[Callable[[Optional[Dict[str, Any]]], Chunker]] = None
) -> None:
    """
    Register a new chunking component.
    
    Args:
        name: Name to register the component under
        component_class: Class implementing Chunker protocol
        factory_func: Optional factory function, defaults to simple constructor
    """
    if factory_func is None:
        # Create a safe factory function that handles different constructor signatures
        def safe_factory(config: Optional[Dict[str, Any]]) -> Chunker:
            try:
                return component_class(config=config)  # type: ignore
            except TypeError:
                # Fallback for classes that don't accept config parameter
                return component_class()  # type: ignore
        factory_func = safe_factory
    
    CHUNKING_REGISTRY[name] = (component_class, factory_func)
    logger.info(f"Registered chunking component: {name}")


def is_chunking_component_available(name: str) -> bool:
    """
    Check if a chunking component is available.
    
    Args:
        name: Component name to check
        
    Returns:
        True if component is available, False otherwise
    """
    return name in CHUNKING_REGISTRY