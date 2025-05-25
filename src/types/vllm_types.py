"""
Type definitions for vLLM integration in HADES-PathRAG.

This module provides type annotations specific to the vLLM integration
to ensure consistency and improve type safety.
"""

import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union, TypedDict, Literal
from enum import Enum
from pydantic import BaseModel, Field


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


# Pydantic models for vLLM configuration
class VLLMServerConfig(BaseModel):
    """Pydantic model for vLLM server configuration."""
    host: str = Field(default="localhost", description="Host address for the vLLM server")
    port: int = Field(default=8000, description="Port for the vLLM server")
    tensor_parallel_size: int = Field(default=1, description="Number of GPUs to use for tensor parallelism")
    gpu_memory_utilization: float = Field(default=0.9, description="Fraction of GPU memory to use")
    max_model_len: Optional[int] = Field(default=None, description="Maximum sequence length")
    dtype: str = Field(default="auto", description="Data type for model weights")


class VLLMModelConfig(BaseModel):
    """Pydantic model for vLLM model configuration."""
    model_id: str = Field(..., description="Model identifier, e.g., 'Qwen2.5-7b-instruct'")
    embedding_model_id: Optional[str] = Field(default=None, description="Embedding model identifier")
    max_tokens: int = Field(default=1024, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Top-p sampling parameter")
    top_k: int = Field(default=40, description="Top-k sampling parameter")
    context_window: int = Field(default=8192, description="Context window size")
    truncate_input: bool = Field(default=True, description="Whether to truncate inputs that exceed context window")


class VLLMConfig(BaseModel):
    """Pydantic model for vLLM integration configuration."""
    server: VLLMServerConfig = Field(default_factory=VLLMServerConfig, description="Server configuration")
    ingestion_models: Dict[str, VLLMModelConfig] = Field(default_factory=dict, description="Models for ingestion")
    inference_models: Dict[str, VLLMModelConfig] = Field(default_factory=dict, description="Models for inference")


# Model mode type
class ModelMode(str, Enum):
    """Enum for model modes."""
    INFERENCE = "inference"
    INGESTION = "ingestion"


@dataclass
class VLLMProcessInfo:
    """Information about a running vLLM process."""
    process: subprocess.Popen  # The subprocess handle
    model_alias: str           # Model alias from configuration
    mode: ModelMode            # Inference or ingestion
    server_url: str            # Full server URL (http://host:port)
    start_time: float          # Unix timestamp when process was started

# Function type for checking server status
ServerStatusType = Dict[str, Any]
