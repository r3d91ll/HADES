"""
HADES-PathRAG Model Engine.

This module provides a unified interface for working with language models,
including embedding generation, completion, and chat functionality.

Multiple engine implementations are supported, including:
- vLLM: High-performance inference with tensor parallelism
- Haystack: Integration with Haystack pipelines and components
"""

from src.model_engine.adapters.vllm_adapter import VLLMAdapter, start_vllm_server
from src.model_engine.base import ModelEngine
from src.model_engine.engines.haystack import HaystackModelEngine
from src.model_engine.server_manager import ServerManager, get_server_manager

__all__ = [
    # Base classes
    "ModelEngine",
    
    # Engines
    "HaystackModelEngine",
    "VLLMAdapter", 
    
    # Utilities
    "start_vllm_server",
    "ServerManager",
    "get_server_manager"
]
