"""
Configuration types for HADES model engines.

This module provides centralized type definitions for configuration objects
used by model engines throughout the HADES system.
"""

from typing import Any, Dict, List, Optional, TypedDict, Union


class HaystackEngineConfigType(TypedDict, total=False):
    """Configuration for Haystack model engines."""
    host: str
    port: int
    api_key: Optional[str]
    connection_timeout: int
    request_timeout: int
    retry_count: int
    models: Dict[str, Dict[str, Any]]
    cache_responses: bool
    cache_size: int
    use_local: bool
    local_path: Optional[str]


class VLLMServerConfigType(TypedDict, total=False):
    """Configuration for the vLLM server."""
    host: str
    port: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: Optional[int]
    dtype: str


class VLLMModelConfigType(TypedDict, total=False):
    """Configuration for a vLLM model."""
    model_id: str
    embedding_model_id: Optional[str]
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    context_window: int
    truncate_input: bool


class VLLMConfigType(TypedDict, total=False):
    """Configuration for the vLLM integration."""
    server: VLLMServerConfigType
    ingestion_models: Dict[str, VLLMModelConfigType]
    inference_models: Dict[str, VLLMModelConfigType]


# Common configuration type for all engines
EngineConfigType = Union[HaystackEngineConfigType, VLLMConfigType]


class ModelEngineConfig:
    """Base configuration for model engines."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.config.copy()
