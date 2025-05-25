"""
vLLM configuration for HADES-PathRAG.

This module defines configuration structures and default settings for vLLM
model serving and inference in the HADES-PathRAG system.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, cast

from pydantic import BaseModel, Field
from src.types.vllm_types import ModelMode, VLLMConfigType, VLLMModelConfigType, VLLMServerConfigType


class VLLMServerConfig(BaseModel):
    """Server configuration for vLLM."""
    
    host: str = Field(default="localhost", description="Hostname for the vLLM server")
    port: int = Field(default=8000, description="Port for the vLLM server")
    tensor_parallel_size: int = Field(default=1, description="Number of GPUs to use for tensor parallelism")
    gpu_memory_utilization: float = Field(default=0.85, description="Fraction of GPU memory to use")
    max_model_len: Optional[int] = Field(default=None, description="Maximum sequence length")
    dtype: str = Field(
        default="auto", description="Data type for model weights and activations"
    )
    
    def to_dict(self) -> VLLMServerConfigType:
        """Convert to dictionary representation."""
        return {
            "host": self.host,
            "port": self.port,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "dtype": self.dtype
        }


class VLLMModelConfig(BaseModel):
    """Model configuration for vLLM."""
    
    # Model identifiers
    model_id: str = Field(..., description="HuggingFace model ID or local path")
    embedding_model_id: Optional[str] = Field(
        default=None, description="Embedding model ID (defaults to model_id if not specified)"
    )
    
    # Server configuration
    port: Optional[int] = Field(default=None, description="Port for this specific model (overrides server port)")
    
    # Model capabilities and parameters
    max_tokens: int = Field(default=2048, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Top-p sampling parameter")
    top_k: int = Field(default=40, description="Top-k sampling parameter")
    
    # Context handling
    context_window: int = Field(default=8192, description="Maximum context window size")
    truncate_input: bool = Field(default=True, description="Whether to truncate input to fit context window")
    
    def to_dict(self) -> VLLMModelConfigType:
        """Convert to dictionary representation."""
        return {
            "model_id": self.model_id,
            "embedding_model_id": self.embedding_model_id,
            "port": self.port,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "context_window": self.context_window,
            "truncate_input": self.truncate_input
        }


class VLLMConfig(BaseModel):
    """Configuration for vLLM."""
    
    server: VLLMServerConfig = Field(default_factory=VLLMServerConfig)
    ingestion_models: Dict[str, VLLMModelConfig] = Field(default_factory=dict)
    inference_models: Dict[str, VLLMModelConfig] = Field(default_factory=dict)
    
    def to_dict(self) -> VLLMConfigType:
        """Convert configuration to dictionary format."""
        ingestion_models_dict: Dict[str, VLLMModelConfigType] = {}
        for name, model in self.ingestion_models.items():
            ingestion_models_dict[name] = model.to_dict()
        
        inference_models_dict: Dict[str, VLLMModelConfigType] = {}
        for name, model in self.inference_models.items():
            inference_models_dict[name] = model.to_dict()
        
        result: VLLMConfigType = {
            "server": self.server.to_dict(),
            "ingestion_models": ingestion_models_dict,
            "inference_models": inference_models_dict
        }
        
        return result
    
    @classmethod
    def load_from_yaml(cls, yaml_path: Optional[Union[str, Path]] = None) -> 'VLLMConfig':
        """Load configuration from YAML file."""
        # Default to config directory in project root
        config_file_path: Path
        if yaml_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_file_path = project_root / "config" / "vllm_config.yaml"
        else:
            config_file_path = Path(yaml_path)
            
        if not config_file_path.exists():
            return cls()  # Return default config if file doesn't exist
            
        with open(config_file_path, 'r') as f:
            config_data = yaml.safe_load(f)
            
        # Create server config
        server_config = VLLMServerConfig(**config_data.get('server', {}))
        
        # Create model configs
        ingestion_models = {}
        for model_name, model_data in config_data.get('ingestion', {}).items():
            ingestion_models[model_name] = VLLMModelConfig(**model_data)
            
        inference_models = {}
        for model_name, model_data in config_data.get('inference', {}).items():
            inference_models[model_name] = VLLMModelConfig(**model_data)
            
        # Create config instance
        config = cls(
            server=server_config,
            ingestion_models=ingestion_models,
            inference_models=inference_models
        )
        
        # Apply environment variable overrides
        config._apply_env_overrides()
        
        return config
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        # Server config overrides
        if host := os.getenv("VLLM_HOST"):
            self.server.host = host
        if port := os.getenv("VLLM_PORT"):
            self.server.port = int(port)
        if tensor_parallel := os.getenv("VLLM_TENSOR_PARALLEL_SIZE"):
            self.server.tensor_parallel_size = int(tensor_parallel)
        if gpu_memory := os.getenv("VLLM_GPU_MEMORY_UTILIZATION"):
            self.server.gpu_memory_utilization = float(gpu_memory)
        if dtype := os.getenv("VLLM_DTYPE"):
            self.server.dtype = dtype
        
        # Model overrides for ingestion
        if embedding_model := os.getenv("VLLM_EMBEDDING_MODEL"):
            if "embedding" in self.ingestion_models:
                self.ingestion_models["embedding"].model_id = embedding_model
        
        if chunking_model := os.getenv("VLLM_CHUNKING_MODEL"):
            if "chunking" in self.ingestion_models:
                self.ingestion_models["chunking"].model_id = chunking_model
        
        # Model overrides for inference
        if code_model := os.getenv("VLLM_CODE_MODEL"):
            if "code" in self.inference_models:
                self.inference_models["code"].model_id = code_model
        
        if general_model := os.getenv("VLLM_GENERAL_MODEL"):
            if "general" in self.inference_models:
                self.inference_models["general"].model_id = general_model
    
    def __init__(self, **data: Any):
        """Initialize vLLM configuration."""
        super().__init__(**data)
        
        # Set up default models if not provided
        if not self.ingestion_models:
            # Embedding model for vector representations
            self.ingestion_models["embedding"] = VLLMModelConfig(
                model_id=os.getenv("VLLM_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
                max_tokens=512,
                context_window=512
            )
            # Chunking model for Chonky
            self.ingestion_models["chunking"] = VLLMModelConfig(
                model_id=os.getenv("VLLM_CHUNKING_MODEL", "Qwen2.5-7b"),
                temperature=0.2,
                max_tokens=1024
            )
            
        if not self.inference_models:
            # Code model for programming-related tasks
            self.inference_models["code"] = VLLMModelConfig(
                model_id=os.getenv("VLLM_CODE_MODEL", "Qwen2.5-coder"),
                temperature=0.2,
                max_tokens=4096
            )
            
            # General model for chat and text generation
            self.inference_models["general"] = VLLMModelConfig(
                model_id=os.getenv("VLLM_GENERAL_MODEL", "Llama-3-8b"),
                temperature=0.7,
                max_tokens=2048
            )
    
    def get_model_config(self, model_alias: str, mode: ModelMode = "inference") -> VLLMModelConfig:
        """Get model configuration by alias and mode.
        
        Args:
            model_alias: The model alias to retrieve
            mode: Either 'inference' or 'ingestion'
        
        Returns:
            The model configuration
        """
        if mode == "inference":
            if model_alias == "default":
                model_alias = os.getenv("VLLM_DEFAULT_MODEL", "general")
            
            if model_alias not in self.inference_models:
                raise ValueError(f"Inference model alias '{model_alias}' not found in configuration")
            return self.inference_models[model_alias]
        elif mode == "ingestion":
            if model_alias == "default_embedding":
                model_alias = os.getenv("VLLM_DEFAULT_EMBEDDING_MODEL", "embedding")
            
            if model_alias not in self.ingestion_models:
                raise ValueError(f"Ingestion model alias '{model_alias}' not found in configuration")
            return self.ingestion_models[model_alias]
        else:
            raise ValueError(f"Invalid mode '{mode}', must be 'inference' or 'ingestion'")


# Function to construct vLLM server command
def make_vllm_command(
    server_config: VLLMServerConfig,
    model_id: Optional[str] = None # Allow specifying a model to load at startup
) -> List[str]:
    """
    Construct the command-line arguments for launching the vLLM OpenAI API server.

    Args:
        server_config: The server configuration settings.
        model_id: Optional model ID to load immediately.

    Returns:
        A list of strings representing the command-line arguments.
    """
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--host", str(server_config.host),
        "--port", str(server_config.port),
        "--tensor-parallel-size", str(server_config.tensor_parallel_size),
        "--gpu-memory-utilization", str(server_config.gpu_memory_utilization),
        "--dtype", server_config.dtype,
    ]

    if server_config.max_model_len is not None:
        cmd.extend(["--max-model-len", str(server_config.max_model_len)])

    # vLLM requires a model to be specified at startup
    if not model_id:
         # Attempt to get a default model if none provided
         # This logic might need refinement based on how defaults are handled
         default_model_candidates = list(VLLMConfig().inference_models.values()) + \
                                    list(VLLMConfig().ingestion_models.values())
         if default_model_candidates:
             model_id = default_model_candidates[0].model_id
         else:
             raise ValueError("Cannot determine a default model; a model_id must be provided to start the vLLM server.")

    cmd.extend(["--model", model_id])

    # Add other potential arguments based on VLLMServerConfig or future needs
    # e.g., --trust-remote-code, --swap-space, etc.

    return cmd
