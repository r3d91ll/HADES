"""
Registry for document format adapters.

This module provides functionality to register and retrieve format-specific adapters.
"""

from typing import Dict, Type, Optional, TypeVar, cast, Any
from .base import BaseAdapter

# Type variable for concrete adapter classes that extend BaseAdapter
T = TypeVar('T', bound=BaseAdapter)

# Registry of format adapters
_ADAPTER_REGISTRY: Dict[str, Type[Any]] = {}


def clear_registry() -> None:
    """
    Clear all registered adapters from the registry.
    
    This is primarily useful for testing purposes.
    """
    global _ADAPTER_REGISTRY
    _ADAPTER_REGISTRY = {}


def register_adapter(format_type: str, adapter_class: Type[T]) -> None:
    """
    Register an adapter for a specific format.
    
    Args:
        format_type: Document format identifier (e.g., "pdf", "html")
        adapter_class: Adapter class to register (concrete implementation of BaseAdapter)
    
    Raises:
        ValueError: If format_type is empty or adapter_class is None
    """
    if not format_type:
        raise ValueError("Format type cannot be empty")
    if adapter_class is None:
        raise ValueError("Adapter class cannot be None")
    
    # Check if the adapter_class is a subclass of BaseAdapter
    # Use try/except for better error handling with non-class objects
    try:
        if not issubclass(adapter_class, BaseAdapter):
            raise ValueError(f"Adapter class must be a subclass of BaseAdapter")
    except TypeError:
        raise ValueError(f"Adapter class must be a class derived from BaseAdapter")
        
    global _ADAPTER_REGISTRY
    _ADAPTER_REGISTRY[format_type.lower()] = adapter_class


def get_adapter_class(format_type: str) -> Type[BaseAdapter]:
    """
    Get the adapter class for a specific format.
    
    Args:
        format_type: Document format identifier
        
    Returns:
        Adapter class for the specified format
        
    Raises:
        ValueError: If no adapter is registered for the format
    """
    format_type = format_type.lower()
    if format_type not in _ADAPTER_REGISTRY:
        raise ValueError(f"No adapter registered for format: {format_type}")
    
    # Cast the retrieved class to Type[BaseAdapter] since we know it implements the BaseAdapter interface
    return cast(Type[BaseAdapter], _ADAPTER_REGISTRY[format_type])


def get_adapter_for_format(format_type: str) -> BaseAdapter:
    """
    Get an instance of the adapter for a specific format.
    
    Args:
        format_type: Document format identifier
        
    Returns:
        Adapter instance for the specified format
    """
    adapter_class = get_adapter_class(format_type)
    return adapter_class()


def get_supported_formats() -> list:
    """
    Get a list of all supported document formats.
    
    Returns:
        List of format identifiers
    """
    return list(_ADAPTER_REGISTRY.keys())
