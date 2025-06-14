"""
Storage Component Factory

This module provides a factory for creating storage components.
Integrated with the global component registry system.
"""

import logging
from typing import Dict, Any, Optional, Type, Callable, Tuple

from src.types.components.protocols import Storage
from src.types.components.contracts import ComponentType
from .core.processor import CoreStorage
from .memory.storage import MemoryStorage
from .networkx.processor import NetworkXStorage
from .arangodb.storage import ArangoStorage


logger = logging.getLogger(__name__)


# Registry of available storage implementations
STORAGE_REGISTRY: Dict[str, Tuple[Type[Storage], Callable[[Optional[Dict[str, Any]]], Storage]]] = {
    "core": (CoreStorage, lambda config: CoreStorage(config=config)),
    "memory": (MemoryStorage, lambda config: MemoryStorage(config=config)),
    "networkx": (NetworkXStorage, lambda config: NetworkXStorage(config=config)),
    "arangodb": (ArangoStorage, lambda config: ArangoStorage(config=config)),
}


def _register_components():
    """Register storage components with the global registry."""
    try:
        from src.components.registry import register_component
        
        for name, (component_class, factory_func) in STORAGE_REGISTRY.items():
            register_component(
                component_type=ComponentType.STORAGE,
                name=name,
                component_class=component_class,
                factory_func=factory_func,
                description=f"Storage component: {name}"
            )
        
        logger.info(f"Registered {len(STORAGE_REGISTRY)} storage components with global registry")
        
    except ImportError:
        logger.warning("Global registry not available, using local registry only")
    except Exception as e:
        logger.error(f"Failed to register storage components: {e}")


# Auto-register components when module is imported
_register_components()


def create_storage_component(
    component_name: str, 
    config: Optional[Dict[str, Any]] = None
) -> Storage:
    """
    Create a storage component by name.
    
    Args:
        component_name: Name of the component to create (e.g., "core", "memory", "networkx", "arangodb")
        config: Optional configuration dictionary
        
    Returns:
        Storage instance
        
    Raises:
        ValueError: If component_name is not recognized
        RuntimeError: If component creation fails
    """
    if component_name not in STORAGE_REGISTRY:
        available = list(STORAGE_REGISTRY.keys())
        raise ValueError(f"Unknown storage component: {component_name}. Available: {available}")
    
    try:
        component_class, factory_func = STORAGE_REGISTRY[component_name]
        component = factory_func(config)
        
        logger.info(f"Created storage component: {component_name}")
        return component
        
    except Exception as e:
        logger.error(f"Failed to create storage component {component_name}: {e}")
        raise RuntimeError(f"Component creation failed: {e}") from e


def get_available_storage_components() -> Dict[str, Type[Storage]]:
    """
    Get all available storage components.
    
    Returns:
        Dictionary mapping component names to their classes
    """
    return {name: cls for name, (cls, _) in STORAGE_REGISTRY.items()}


def register_storage_component(
    name: str, 
    component_class: Type[Storage],
    factory_func: Optional[Callable[[Optional[Dict[str, Any]]], Storage]] = None
) -> None:
    """
    Register a new storage component.
    
    Args:
        name: Name to register the component under
        component_class: Class implementing Storage protocol
        factory_func: Optional factory function, defaults to simple constructor
    """
    if factory_func is None:
        # Create a safe factory function that handles different constructor signatures
        def safe_factory(config: Optional[Dict[str, Any]]) -> Storage:
            try:
                return component_class(config=config)  # type: ignore
            except TypeError:
                # Fallback for classes that don't accept config parameter
                return component_class()  # type: ignore
        factory_func = safe_factory
    
    STORAGE_REGISTRY[name] = (component_class, factory_func)
    logger.info(f"Registered storage component: {name}")


def is_storage_component_available(name: str) -> bool:
    """
    Check if a storage component is available.
    
    Args:
        name: Component name to check
        
    Returns:
        True if component is available, False otherwise
    """
    return name in STORAGE_REGISTRY