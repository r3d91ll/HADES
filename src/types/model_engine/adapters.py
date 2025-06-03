"""
Model adapter types for HADES-PathRAG.

This module provides type definitions for model adapters that
provide a standardized interface to different model engine backends.
"""

from typing import Any, Dict, List, Optional, Protocol, TypedDict, Union, Callable
from .engine import ModelType


class ModelAdapterConfig(TypedDict, total=False):
    """Configuration for a model adapter."""
    model_id: str
    engine_type: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    context_window: int
    adapter_type: str
    adapter_options: Dict[str, Any]


class ModelAdapterResult(TypedDict, total=False):
    """Result from a model adapter inference."""
    model_id: str
    engine_type: str
    input: Union[str, List[str]]
    output: Union[str, List[str], List[float], List[List[float]]]
    metadata: Dict[str, Any]
    tokens_used: int
    tokens_generated: int
    latency_ms: float


class ModelAdapterProtocol(Protocol):
    """Protocol defining the interface for model adapters."""
    
    def __init__(self, config: Optional[ModelAdapterConfig] = None) -> None:
        """Initialize the adapter with optional configuration."""
        ...
    
    def infer(self, inputs: Union[str, List[str]], **kwargs: Any) -> ModelAdapterResult:
        """Perform inference with the model."""
        ...
    
    def get_type(self) -> ModelType:
        """Get the model type (embedding, completion, chat)."""
        ...
    
    def set_callback(self, callback: Callable[[str], None]) -> None:
        """Set a callback for streaming responses."""
        ...
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the adapter and underlying model."""
        ...


class ModelAdapterRegistry(TypedDict, total=False):
    """Registry mapping adapter type names to implementation classes."""
    embedding: Dict[str, Any]
    completion: Dict[str, Any] 
    chat: Dict[str, Any]
