"""
Engine interface types for HADES-PathRAG model engines.

This module provides centralized type definitions for the model engine interfaces
used throughout the HADES-PathRAG system, including both abstract base classes
and protocol definitions for type checking.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, TypedDict, Union, Type


class EngineType(str, Enum):
    """Enum of supported model engine types."""
    HAYSTACK = "haystack"
    VLLM = "vllm"


class ModelType(str, Enum):
    """Enum of model types supported by engines."""
    EMBEDDING = "embedding"
    COMPLETION = "completion"
    CHAT = "chat"
    

class ModelStatus(str, Enum):
    """Status of a model within an engine."""
    LOADED = "loaded"
    LOADING = "loading"
    UNLOADING = "unloading"
    UNLOADED = "unloaded"
    ERROR = "error"


class EngineStatus(str, Enum):
    """Status of a model engine."""
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    STARTING = "starting"
    STOPPING = "stopping"


class ModelInfo(TypedDict, total=False):
    """Information about a loaded model."""
    model_id: str
    status: ModelStatus
    type: ModelType
    device: str
    load_time: float
    metadata: Dict[str, Any]


class HealthStatus(TypedDict, total=False):
    """Health status information for a model engine."""
    status: EngineStatus
    loaded_models: List[str]
    uptime_seconds: float
    memory_usage: Dict[str, float]
    error_message: Optional[str]


# Protocol definitions for engine interfaces
class ModelEngineProtocol(Protocol):
    """Protocol defining the interface for model engines."""
    
    @abstractmethod
    def load_model(self, model_id: str, **kwargs: Any) -> str:
        """Load a model into the engine."""
        ...
    
    @abstractmethod
    def unload_model(self, model_id: str) -> str:
        """Unload a model from the engine."""
        ...
    
    @abstractmethod
    def infer(self, model_id: str, inputs: Union[str, List[str]], **kwargs: Any) -> Dict[str, Any]:
        """Perform inference with a loaded model."""
        ...
    
    @abstractmethod
    def get_loaded_models(self) -> Dict[str, ModelInfo]:
        """Get information about currently loaded models."""
        ...
    
    @abstractmethod
    def health_check(self) -> HealthStatus:
        """Check the health status of the engine."""
        ...


# Concrete base class for model engines
class ModelEngine(ABC):
    """Base class for all model engines in HADES-PathRAG.
    
    A ModelEngine is responsible for loading, unloading, and running models
    for inference. It abstracts away the details of how models are loaded
    and where they are run (GPU, CPU, remote service, etc).
    
    Implementations should handle:
    - Model loading and unloading
    - Efficient memory management (shared GPU memory, etc)
    - Batched inference
    - Error handling and recovery
    """
    
    @abstractmethod
    def load_model(self, model_id: str, 
                  device: Optional[Union[str, List[str]]] = None) -> str:
        """Load a model into memory.
        
        Args:
            model_id: The ID of the model to load (HF model ID or local path)
            device: The device(s) to load the model onto (e.g., "cuda:0")
                    If None, uses the default device for the engine.
                    
        Returns:
            Status string ("loaded" or "already_loaded")
            
        Raises:
            RuntimeError: If model loading fails
        """
        pass
    
    @abstractmethod
    def unload_model(self, model_id: str) -> str:
        """Unload a model from memory.
        
        Args:
            model_id: The ID of the model to unload
            
        Returns:
            Status string ("unloaded")
        """
        pass
    
    @abstractmethod
    def infer(self, model_id: str, inputs: Union[str, List[str]], 
             task: str, **kwargs: Any) -> Any:
        """Run inference with a model.
        
        Args:
            model_id: The ID of the model to use
            inputs: Input text or list of texts
            task: The type of task to perform (e.g., "generate", "embed", "chunk")
            **kwargs: Task-specific parameters
            
        Returns:
            Model outputs in an appropriate format for the task
            
        Raises:
            ValueError: If the model doesn't support the task
            RuntimeError: If inference fails
        """
        pass
    
    @abstractmethod
    def get_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about currently loaded models.
        
        Returns:
            Dict mapping model_id to metadata about the model
        """
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the engine.
        
        Returns:
            Dict containing health information (status, memory usage, etc)
        """
        pass


# Type alias for engine registry
EngineRegistry = Dict[str, Type[ModelEngineProtocol]]
"""Dictionary mapping engine type strings to engine implementation classes."""
