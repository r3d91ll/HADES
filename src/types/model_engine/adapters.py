"""
Model adapter types for HADES-PathRAG.

This module provides type definitions for model adapters that
provide a standardized interface to different model engine backends.
Includes both concrete base classes and protocol definitions.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Protocol, TypedDict, TypeVar, Union
from .engine import ModelType

# Type for model responses
T = TypeVar('T')


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


# Concrete base classes for adapters
class ModelAdapter(ABC):
    """Base interface for all model adapters."""
    
    @abstractmethod
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        """Initialize the adapter with model configuration."""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model backend is available."""
        pass


class EmbeddingAdapter(ModelAdapter):
    """Interface for embedding generation."""
    
    @abstractmethod
    def get_embeddings(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """Generate embeddings for texts."""
        pass


class CompletionAdapter(ModelAdapter):
    """Interface for text completion."""
    
    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Complete a text prompt."""
        pass
    
    @abstractmethod
    async def complete_async(self, prompt: str, **kwargs: Any) -> str:
        """Complete a text prompt asynchronously."""
        pass


class ChatAdapter(ModelAdapter):
    """Interface for chat completion."""
    
    @abstractmethod
    def chat_complete(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        """Complete a chat conversation."""
        pass
    
    @abstractmethod
    async def chat_complete_async(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        """Complete a chat conversation asynchronously."""
        pass
