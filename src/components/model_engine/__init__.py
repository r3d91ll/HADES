"""
HADES Model Engine Components

This module contains all model engine implementations for HADES.
Each implementation follows the ModelEngine protocol and provides
swappable model serving capabilities.

Available Implementations:
- vllm: vLLM-based model serving
- haystack: Haystack-based model serving  
- core: Core model engine functionality

Example:
    from src.components.model_engine.engines.vllm.processor import VLLMModelEngine
    from src.components.model_engine.engines.haystack.processor import HaystackModelEngine
"""

from typing import Dict, Type, Any
from src.types.components.protocols import ModelEngine

# Component registry for model engines
MODEL_ENGINE_REGISTRY: Dict[str, Type[ModelEngine]] = {}

def register_model_engine(name: str, engine_class: Type[ModelEngine]) -> None:
    """Register a model engine implementation."""
    MODEL_ENGINE_REGISTRY[name] = engine_class

def get_model_engine(name: str) -> Type[ModelEngine]:
    """Get a model engine implementation by name."""
    if name not in MODEL_ENGINE_REGISTRY:
        raise ValueError(f"Model engine '{name}' not found. Available: {list(MODEL_ENGINE_REGISTRY.keys())}")
    return MODEL_ENGINE_REGISTRY[name]

def list_model_engines() -> list[str]:
    """List all available model engine implementations."""
    return list(MODEL_ENGINE_REGISTRY.keys())

# Auto-register implementations when imported
def _auto_register() -> None:
    """Auto-register all model engine implementations."""
    try:
        from .engines.vllm.processor import VLLMModelEngine
        register_model_engine("vllm", VLLMModelEngine)
    except ImportError:
        pass
    
    try:
        from .engines.haystack.processor import HaystackModelEngine  
        register_model_engine("haystack", HaystackModelEngine)
    except ImportError:
        pass
        
    try:
        from .core.processor import CoreModelEngine
        register_model_engine("core", CoreModelEngine)
    except ImportError:
        pass

_auto_register()