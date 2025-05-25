"""
Registry for embedding adapters.

This module provides a registry for embedding adapters, allowing the system
to dynamically select the appropriate adapter based on model type or content type.
"""

from typing import Dict, Type, Callable, Any, Optional
import logging

# Import the EmbeddingAdapter protocol from the base module
from src.embedding.base import EmbeddingAdapter

logger = logging.getLogger(__name__)

# Registry to store adapters by name
_ADAPTER_REGISTRY: Dict[str, Type[EmbeddingAdapter]] = {}


def register_adapter(name: str, adapter_cls: Type[EmbeddingAdapter]) -> None:
    """
    Register an embedding adapter.
    
    Args:
        name: Name of the adapter (e.g., "modernbert", "codebert")
        adapter_cls: Adapter class to register
    """
    global _ADAPTER_REGISTRY
    logger.info(f"Registering embedding adapter: {name} -> {adapter_cls.__name__}")
    _ADAPTER_REGISTRY[name] = adapter_cls


def get_adapter_by_name(name: str) -> Type[EmbeddingAdapter]:
    """
    Get an embedding adapter class by name.
    
    Args:
        name: Name of the adapter to retrieve
        
    Returns:
        Adapter class
        
    Raises:
        ValueError: If adapter is not found
    """
    if name not in _ADAPTER_REGISTRY:
        available = list(_ADAPTER_REGISTRY.keys())
        raise ValueError(f"Embedding adapter '{name}' not found. Available adapters: {available}")
    
    return _ADAPTER_REGISTRY[name]


def get_adapter_for_file_type(file_type: str) -> Type[EmbeddingAdapter]:
    """
    Get the appropriate embedding adapter for a file type.
    
    Args:
        file_type: Type of file (e.g., "python", "markdown", "text")
        
    Returns:
        Adapter class suitable for the file type
    """
    # Map file types to appropriate adapters
    file_type_map = {
        # Python files use the specialized ModernBERT variant for Python code
        "python": "python_code_bert",
        
        # Other code files still use CodeBERT
        "java": "codebert",
        "javascript": "codebert",
        "typescript": "codebert",
        "go": "codebert",
        "rust": "codebert",
        "c": "codebert",
        "cpp": "codebert",
        "csharp": "codebert",
        "php": "codebert",
        
        # Default to ModernBERT for all other file types (text documents)
        "markdown": "modernbert",
        "text": "modernbert",
        "pdf": "modernbert",
        "html": "modernbert",
        "default": "modernbert"
    }
    
    adapter_name = file_type_map.get(file_type, file_type_map["default"])
    logger.info(f"Selected adapter '{adapter_name}' for file type '{file_type}'")
    
    return get_adapter_by_name(adapter_name)
