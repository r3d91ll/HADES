"""
Document Processing Component Factory

This module provides a factory for creating document processing components.
Integrated with the global component registry system.
"""

import logging
from typing import Dict, Any, Optional, Type, Callable, Tuple

from src.types.components.protocols import DocumentProcessor
from src.types.components.contracts import ComponentType
from .core.processor import CoreDocumentProcessor
from .docling.processor import DoclingDocumentProcessor


logger = logging.getLogger(__name__)


# Registry of available document processor implementations
DOCPROC_REGISTRY: Dict[str, Tuple[Type[DocumentProcessor], Callable[[Optional[Dict[str, Any]]], DocumentProcessor]]] = {
    "core": (CoreDocumentProcessor, lambda config: CoreDocumentProcessor(config=config)),
    "docling": (DoclingDocumentProcessor, lambda config: DoclingDocumentProcessor(config=config)),
}


def _register_components() -> None:
    """Register document processing components with the global registry."""
    try:
        from src.components.registry import register_component
        
        for name, (component_class, factory_func) in DOCPROC_REGISTRY.items():
            register_component(
                component_type=ComponentType.DOCPROC,
                name=name,
                component_class=component_class,
                factory_func=factory_func,
                description=f"Document processing component: {name}"
            )
        
        logger.info(f"Registered {len(DOCPROC_REGISTRY)} docproc components with global registry")
        
    except ImportError:
        logger.warning("Global registry not available, using local registry only")
    except Exception as e:
        logger.error(f"Failed to register docproc components: {e}")


# Auto-register components when module is imported
_register_components()


def create_docproc_component(
    component_name: str, 
    config: Optional[Dict[str, Any]] = None
) -> DocumentProcessor:
    """
    Create a document processing component by name.
    
    Args:
        component_name: Name of the component to create (e.g., "core", "docling")
        config: Optional configuration dictionary
        
    Returns:
        DocumentProcessor instance
        
    Raises:
        ValueError: If component_name is not recognized
        RuntimeError: If component creation fails
    """
    if component_name not in DOCPROC_REGISTRY:
        available = list(DOCPROC_REGISTRY.keys())
        raise ValueError(f"Unknown docproc component: {component_name}. Available: {available}")
    
    try:
        component_class, factory_func = DOCPROC_REGISTRY[component_name]
        component = factory_func(config)
        
        logger.info(f"Created docproc component: {component_name}")
        return component
        
    except Exception as e:
        logger.error(f"Failed to create docproc component {component_name}: {e}")
        raise RuntimeError(f"Component creation failed: {e}") from e


def get_available_docproc_components() -> Dict[str, Type[DocumentProcessor]]:
    """
    Get all available document processing components.
    
    Returns:
        Dictionary mapping component names to their classes
    """
    return {name: cls for name, (cls, _) in DOCPROC_REGISTRY.items()}


def register_docproc_component(
    name: str, 
    component_class: Type[DocumentProcessor],
    factory_func: Optional[Callable[[Optional[Dict[str, Any]]], DocumentProcessor]] = None
) -> None:
    """
    Register a new document processing component.
    
    Args:
        name: Name to register the component under
        component_class: Class implementing DocumentProcessor protocol
        factory_func: Optional factory function, defaults to simple constructor
    """
    if factory_func is None:
        # Create a safe factory function that handles different constructor signatures
        def safe_factory(config: Optional[Dict[str, Any]]) -> DocumentProcessor:
            try:
                return component_class(config=config)  # type: ignore
            except TypeError:
                # Fallback for classes that don't accept config parameter
                return component_class()  # type: ignore
        factory_func = safe_factory
    
    DOCPROC_REGISTRY[name] = (component_class, factory_func)
    logger.info(f"Registered docproc component: {name}")


def is_docproc_component_available(name: str) -> bool:
    """
    Check if a document processing component is available.
    
    Args:
        name: Component name to check
        
    Returns:
        True if component is available, False otherwise
    """
    return name in DOCPROC_REGISTRY