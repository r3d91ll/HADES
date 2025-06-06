"""Base interfaces and registry for embedding adapters.

This module defines the core interfaces for embedding generation and
provides a registry mechanism to manage different embedding adapters.

NOTE: Type definitions have been migrated to src.types.embedding.
This module now imports types from the centralized location.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Union, cast, Optional
import logging
import numpy as np

# Import consolidated types
from src.types.embedding import EmbeddingAdapter, EmbeddingVector, EmbeddingAdapterRegistry

logger = logging.getLogger(__name__)

# Registry for adapters
_adapter_registry: EmbeddingAdapterRegistry = {}


# EmbeddingAdapter protocol is now imported from src.types.embedding


def register_adapter(name: str, adapter_cls: type[EmbeddingAdapter]) -> None:
    """Register an embedding adapter implementation.
    
    Args:
        name: Name to register the adapter under
        adapter_cls: Adapter class to register
    """
    if name in _adapter_registry:
        logger.warning(f"Overwriting existing adapter registration for '{name}'")
    _adapter_registry[name] = adapter_cls
    logger.debug(f"Registered embedding adapter '{name}'")


def get_adapter(name: str, **init_kwargs: Any) -> EmbeddingAdapter:
    """Get an instance of a registered embedding adapter.
    
    Args:
        name: Name of the adapter to instantiate
        **init_kwargs: Keyword arguments to pass to the adapter constructor
        
    Returns:
        Instance of the requested adapter
        
    Raises:
        KeyError: If no adapter is registered with the given name
    """
    if name not in _adapter_registry:
        available = ", ".join(_adapter_registry.keys())
        raise KeyError(f"No embedding adapter registered as '{name}'. Available: {available}")
    
    adapter_cls = _adapter_registry[name]
    return adapter_cls(**init_kwargs)
