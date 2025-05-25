"""
Central type definitions for HADES-PathRAG.

This module provides foundational type definitions used across
the HADES-PathRAG codebase to ensure type consistency.
"""

from .common import (
    NodeID, 
    EdgeID, 
    NodeData, 
    EdgeData, 
    EmbeddingVector,
    DocumentContent,
    StorageConfig,
    EmbeddingConfig,
    GraphConfig
)

from .vllm_types import (
    VLLMServerConfigType,
    VLLMModelConfigType,
    VLLMConfigType,
    ModelMode,
    ServerStatusType
)

__all__ = [
    # Common types
    "NodeID",
    "EdgeID", 
    "NodeData", 
    "EdgeData", 
    "EmbeddingVector",
    "DocumentContent",
    "StorageConfig",
    "EmbeddingConfig",
    "GraphConfig",
    
    # vLLM types
    "VLLMServerConfigType",
    "VLLMModelConfigType",
    "VLLMConfigType",
    "ModelMode",
    "ServerStatusType"
]
