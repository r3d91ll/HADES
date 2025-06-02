"""
Configuration for ISNE Pipeline.

This module provides configuration models for the ISNE pipeline.
"""

from typing import Optional, Dict, Any, TypedDict, NotRequired

from src.types.embedding import EmbeddingConfig


class PipelineConfig(TypedDict, total=False):
    """
    Configuration for ISNE pipeline.
    
    This TypedDict contains configuration options for the ISNE pipeline,
    including embedding, model, and hardware settings.
    """
    embedding_config: EmbeddingConfig
    use_gpu: NotRequired[bool]  # Default: True
    model_path: NotRequired[Optional[str]]  # Default: None
    validate: NotRequired[bool]  # Default: True
    batch_size: NotRequired[int]  # Default: 32
    additional_params: NotRequired[Dict[str, Any]]  # Default: {}
