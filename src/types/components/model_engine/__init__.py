"""
Model engine type definitions for HADES.

This module provides centralized type definitions for the model engine system,
including engines, adapters, configuration, inference requests/responses, and results.

The types are organized by component implementation:
- core/: Types for core model engine implementation
- engines/vllm/: Types for vLLM model engine implementation  
- engines/haystack/: Types for Haystack model engine implementation
"""

# Engine types
from .engine import (
    # Enums
    EngineType,
    ModelType,
    ModelStatus,
    EngineStatus,
    
    # TypedDicts
    ModelInfo,
    HealthStatus,
    
    # Protocols and base classes
    ModelEngineProtocol,
    ModelEngine,
    
    # Type aliases
    EngineRegistry,
)

# Adapter types
from .adapters import (
    # TypedDicts
    ModelAdapterConfig,
    ModelAdapterResult,
    ModelAdapterRegistry,
    
    # Protocols and base classes
    ModelAdapterProtocol,
    ModelAdapter,
    EmbeddingAdapter,
    CompletionAdapter,
    ChatAdapter,
    
    # Type variables
    T,
)

# Configuration types
from .config import (
    # TypedDicts
    HaystackEngineConfigType,
    VLLMServerConfigType,
    VLLMModelConfigType,
    VLLMConfigType,
    
    # Type aliases
    EngineConfigType,
)

# Inference types
from .inference import (
    # TypedDicts
    CompletionRequest,
    ChatCompletionRequest,
    EmbeddingRequest,
    ChatMessage,
    CompletionChoice,
    ChatCompletionChoice,
    CompletionResponse,
    ChatCompletionResponse,
    EmbeddingResponse,
    EmbeddingData,
    
    # Type aliases
    InferenceRequest,
    InferenceResponse,
)

# Result types
from .results import (
    # Enums
    ChatRole,
    
    # TypedDicts
    EmbeddingResult,
    CompletionResult,
    CompletionMetadata,
    ChatCompletionResult,
    ChatCompletionMetadata,
    StreamingChatCompletionResult,
    EmbeddingOptions,
    CompletionOptions,
    ChatCompletionOptions,
)

# Component-specific types from components_impl
from ...components_impl.model_engine.core import *
from ...components_impl.model_engine.engines.vllm import *
from ...components_impl.model_engine.engines.haystack import *

__all__ = [
    # Engine types
    "EngineType",
    "ModelType", 
    "ModelStatus",
    "EngineStatus",
    "ModelInfo",
    "HealthStatus",
    "ModelEngineProtocol",
    "ModelEngine",
    "EngineRegistry",
    
    # Adapter types
    "ModelAdapterConfig",
    "ModelAdapterResult",
    "ModelAdapterRegistry",
    "ModelAdapterProtocol",
    "ModelAdapter",
    "EmbeddingAdapter",
    "CompletionAdapter",
    "ChatAdapter",
    "T",
    
    # Configuration types
    "HaystackEngineConfigType",
    "VLLMServerConfigType",
    "VLLMModelConfigType",
    "VLLMConfigType",
    "EngineConfigType",
    
    # Inference types
    "CompletionRequest",
    "ChatCompletionRequest", 
    "EmbeddingRequest",
    "ChatMessage",
    "CompletionChoice",
    "ChatCompletionChoice",
    "CompletionResponse",
    "ChatCompletionResponse",
    "EmbeddingResponse",
    "EmbeddingData",
    "InferenceRequest",
    "InferenceResponse",
    
    # Result types
    "ChatRole",
    "EmbeddingResult",
    "CompletionResult",
    "CompletionMetadata",
    "ChatCompletionResult",
    "ChatCompletionMetadata",
    "StreamingChatCompletionResult",
    "EmbeddingOptions",
    "CompletionOptions",
    "ChatCompletionOptions",
]