"""
Registry for chunkers in HADES-PathRAG.

This module provides the registry for chunkers in the HADES-PathRAG system,
allowing chunkers to be registered and retrieved by name. It supports various
types of chunkers including those for text and different code formats.
"""

from typing import Dict, Type, Any, Optional

# Import chunkers
from src.chunking.code_chunkers.python_chunker import PythonCodeChunker
from src.chunking.code_chunkers.yaml_chunker import YAMLCodeChunker
from src.chunking.code_chunkers.json_chunker import JSONCodeChunker

# Registry for chunkers
_CHUNKER_REGISTRY: Dict[str, Type] = {}


def register_chunker(name: str, chunker_cls: Type) -> None:
    """Register a chunker class with a name.
    
    Args:
        name: Name to register the chunker with
        chunker_cls: Chunker class to register
    """
    _CHUNKER_REGISTRY[name] = chunker_cls


def get_chunker(name: str, **kwargs) -> Any:
    """Get a chunker instance by name.
    
    Args:
        name: Name of the chunker to get
        **kwargs: Arguments to pass to the chunker constructor
        
    Returns:
        Chunker instance
        
    Raises:
        ValueError: If no chunker is registered with the given name
    """
    if name not in _CHUNKER_REGISTRY:
        raise ValueError(f"No chunker registered with name '{name}'")
    
    chunker_cls = _CHUNKER_REGISTRY[name]
    return chunker_cls(**kwargs)


def list_chunkers() -> Dict[str, Type]:
    """List all registered chunkers.
    
    Returns:
        Dictionary mapping chunker names to chunker classes
    """
    return _CHUNKER_REGISTRY.copy()


# Register built-in chunkers
register_chunker("python_code", PythonCodeChunker)
register_chunker("yaml_code", YAMLCodeChunker)
register_chunker("json_code", JSONCodeChunker)
