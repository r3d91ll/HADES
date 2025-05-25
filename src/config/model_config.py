"""
Model configuration for HADES-PathRAG.

This module defines configuration structures and default settings for model
serving and inference in the HADES-PathRAG system, supporting the vLLM model
backend with extensibility for future backends.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, cast

from pydantic import BaseModel, Field
from src.types.model_types import ModelMode, ModelConfigType, ModelBackendConfigType, ServerConfigType


class ServerConfig(BaseModel):
    """Server configuration for model backends."""
    
    host: str = Field(default="localhost", description="Hostname for the model server")
    port: int = Field(default=8000, description="Port for the model server")
    tensor_parallel_size: int = Field(default=1, description="Number of GPUs to use for tensor parallelism")
    gpu_memory_utilization: float = Field(default=0.85, description="Fraction of GPU memory to use")
    max_model_len: Optional[int] = Field(default=None, description="Maximum sequence length")
    dtype: str = Field(
        default="auto", description="Data type for model weights and activations"
    )
    backend: str = Field(
        default="vllm", description="Model backend to use (vllm is currently supported)"
    )
    
    def to_dict(self) -> ServerConfigType:
        """Convert to dictionary representation."""
        return {
            "host": self.host,
            "port": self.port,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "dtype": self.dtype,
            "backend": self.backend
        }


class ModelBackendConfig(BaseModel):
    """Model configuration for specific model backends."""
    
    # Model identifiers
    model_id: str = Field(..., description="HuggingFace model ID or local path")
    embedding_model_id: Optional[str] = Field(
        default=None, description="Embedding model ID (defaults to model_id if not specified)"
    )
    
    # Model capabilities and parameters
    max_tokens: int = Field(default=2048, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Top-p sampling parameter")
    top_k: int = Field(default=40, description="Top-k sampling parameter")
    
    # Context handling
    context_window: int = Field(default=8192, description="Maximum context window size")
    truncate_input: bool = Field(default=True, description="Whether to truncate input to fit context window")
    
    # Backend-specific settings
    backend: str = Field(default="vllm", description="Model backend (vllm is currently supported)")
    batch_size: int = Field(default=32, description="Batch size for processing")
    
    def to_dict(self) -> ModelBackendConfigType:
        """Convert to dictionary representation."""
        return {
            "model_id": self.model_id,
            "embedding_model_id": self.embedding_model_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "context_window": self.context_window,
            "truncate_input": self.truncate_input,
            "backend": self.backend,
            "batch_size": self.batch_size
        }


class ModelConfig(BaseModel):
    """Configuration for model services."""
    
    server: ServerConfig = Field(default_factory=ServerConfig)
    ingestion_models: Dict[str, ModelBackendConfig] = Field(default_factory=dict)
    inference_models: Dict[str, ModelBackendConfig] = Field(default_factory=dict)
    
    def to_dict(self) -> ModelConfigType:
        """Convert configuration to dictionary format."""
        ingestion_models_dict: Dict[str, ModelBackendConfigType] = {}
        for name, model in self.ingestion_models.items():
            ingestion_models_dict[name] = model.to_dict()
        
        inference_models_dict: Dict[str, ModelBackendConfigType] = {}
        for name, model in self.inference_models.items():
            inference_models_dict[name] = model.to_dict()
        
        result: ModelConfigType = {
            "server": self.server.to_dict(),
            "ingestion_models": ingestion_models_dict,
            "inference_models": inference_models_dict
        }
        
        return result
    
    def get_model_config(self, model_alias: str, mode: str = "inference") -> ModelBackendConfig:
        """
        Get a specific model configuration by alias and mode.
        
        Args:
            model_alias: Name of the model config to retrieve
            mode: Either "inference" or "ingestion"
            
        Returns:
            The requested model configuration
            
        Raises:
            ValueError: If the mode is invalid or the model alias isn't found
        """
        if mode == "inference":
            if model_alias in self.inference_models:
                return self.inference_models[model_alias]
            else:
                available = list(self.inference_models.keys())
                raise ValueError(f"Model '{model_alias}' not found in inference models. Available: {available}")
        elif mode == "ingestion":
            if model_alias in self.ingestion_models:
                return self.ingestion_models[model_alias]
            else:
                available = list(self.ingestion_models.keys())
                raise ValueError(f"Model '{model_alias}' not found in ingestion models. Available: {available}")
        else:
            raise ValueError(f"Invalid mode '{mode}', must be 'inference' or 'ingestion'")
    
    def get_ingestion_models(self) -> Dict[str, ModelBackendConfig]:
        """Get dictionary of all ingestion models."""
        return self.ingestion_models
    
    def get_inference_models(self) -> Dict[str, ModelBackendConfig]:
        """Get dictionary of all inference models."""
        return self.inference_models
    
    @classmethod
    def load_from_yaml(cls, yaml_path: Optional[Union[str, Path]] = None) -> 'ModelConfig':
        """Load configuration from YAML file."""
        # Default to config directory in project root
        config_file_path: Path
        if yaml_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_file_path = project_root / "config" / "model_config.yaml"
        else:
            config_file_path = Path(yaml_path)
            
        if not config_file_path.exists():
            return cls()  # Return default config if file doesn't exist
            
        with open(config_file_path, 'r') as f:
            config_data = yaml.safe_load(f)
            
        # Create server config
        server_config = ServerConfig(**config_data.get('server', {}))
        
        # Create model configs
        ingestion_models = {}
        for model_name, model_data in config_data.get('ingestion', {}).items():
            ingestion_models[model_name] = ModelBackendConfig(**model_data)
        
        inference_models = {}
        for model_name, model_data in config_data.get('inference', {}).items():
            inference_models[model_name] = ModelBackendConfig(**model_data)
        
        return cls(
            server=server_config,
            ingestion_models=ingestion_models,
            inference_models=inference_models
        )
