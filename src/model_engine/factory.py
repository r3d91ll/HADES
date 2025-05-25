"""
Model engine factory for HADES-PathRAG.

This module provides factory functions for creating model engines based on configuration.
It supports both Haystack and vLLM engines.
"""

from typing import Dict, Optional, Union, Type

from src.model_engine.base import ModelEngine
from src.model_engine.engines.haystack import HaystackModelEngine
from src.model_engine.engines.vllm import VLLMModelEngine


# Registry of available engine types
ENGINE_REGISTRY: Dict[str, Type[ModelEngine]] = {
    "haystack": HaystackModelEngine,
    "vllm": VLLMModelEngine,
}


def create_model_engine(engine_type: str, config_path: Optional[str] = None) -> ModelEngine:
    """
    Create a model engine of the specified type.
    
    Args:
        engine_type: Type of engine to create ("haystack" or "vllm")
        config_path: Optional path to configuration file
        
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
    
    engine_class = ENGINE_REGISTRY[engine_type]
    
    if engine_type == "haystack":
        return engine_class(socket_path=config_path)
    else:
        return engine_class(config_path=config_path)


# Global engine instances
_haystack_engine: Optional[HaystackModelEngine] = None
_vllm_engine: Optional[VLLMModelEngine] = None


def get_haystack_engine(socket_path: Optional[str] = None) -> HaystackModelEngine:
    """
    Get the global Haystack model engine instance.
    
    Args:
        socket_path: Optional custom path to the Unix domain socket
        
    Returns:
        Global HaystackModelEngine instance
    """
    global _haystack_engine
    
    if _haystack_engine is None:
        _haystack_engine = HaystackModelEngine(socket_path=socket_path)
        
    return _haystack_engine


def get_vllm_engine(config_path: Optional[str] = None) -> VLLMModelEngine:
    """
    Get the global vLLM model engine instance.
    
    Args:
        config_path: Optional custom path to the vLLM configuration file
        
    Returns:
        Global VLLMModelEngine instance
    """
    global _vllm_engine
    
    if _vllm_engine is None:
        _vllm_engine = VLLMModelEngine(config_path=config_path)
        
    return _vllm_engine
