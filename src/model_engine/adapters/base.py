"""Base interfaces for model adapters."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, AsyncIterator, TypeVar

# Type for model responses
T = TypeVar('T')

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
