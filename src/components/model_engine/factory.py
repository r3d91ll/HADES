"""
Model engine factory for HADES.

This module provides factory functions for creating model engines based on configuration.
It supports Haystack engines.
"""

from typing import Dict, Optional, Union, Type, Tuple, Callable, Any, cast

from src.types.components.protocols import ModelEngine
from src.components.model_engine.engines.haystack.processor import HaystackModelEngine


# Registry of available engine types with their specific constructors
# Each entry is a tuple of (engine_class, constructor_function)
ENGINE_REGISTRY: Dict[str, Tuple[Type[ModelEngine], Callable[[Optional[Dict[str, Any]]], ModelEngine]]] = {
    "haystack": (HaystackModelEngine, lambda config: HaystackModelEngine(config=config)),
}


def create_model_engine(engine_type: str, config: Optional[Dict[str, Any]] = None) -> ModelEngine:
    """
    Create a model engine of the specified type.
    
    Args:
        engine_type: Type of engine to create ("haystack")
        config: Optional configuration dictionary
        
    Returns:
        Initialized model engine instance
        
    Raises:
        ValueError: If the specified engine type is not supported
    """
    if engine_type not in ENGINE_REGISTRY:
        raise ValueError(
            f"Unsupported engine type: {engine_type}. "
            f"Supported types: {', '.join(ENGINE_REGISTRY.keys())}"
        )
    
    # Get the constructor function for this engine type
    _, constructor = ENGINE_REGISTRY[engine_type]
    
    # Create the engine using the appropriate constructor
    return constructor(config)


# Global engine instances
_haystack_engine: Optional[HaystackModelEngine] = None


def get_haystack_engine(config: Optional[Dict[str, Any]] = None) -> HaystackModelEngine:
    """
    Get the global Haystack model engine instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Global HaystackModelEngine instance
    """
    global _haystack_engine
    
    if _haystack_engine is None:
        _haystack_engine = HaystackModelEngine(config=config)
        
    return _haystack_engine
