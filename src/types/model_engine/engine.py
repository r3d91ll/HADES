"""
Engine interface types for HADES-PathRAG model engines.

This module provides centralized type definitions for the model engine interfaces
used throughout the HADES-PathRAG system.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, TypedDict, Union, Type, cast


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


# Type alias for engine registry
EngineRegistry = Dict[str, Type[ModelEngineProtocol]]
"""Dictionary mapping engine type strings to engine implementation classes."""
