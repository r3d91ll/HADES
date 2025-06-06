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

from .utils import (
    BatchStats,
)

from .validation import (
    PreValidationResult,
    PostValidationResult,
    ValidationSummary,
    ValidationStage,
    ValidationStatus,
    ValidationSeverity,
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
    "ServerStatusType",
    
    # Utils types
    "BatchStats",
    
    # Validation types
    "PreValidationResult",
    "PostValidationResult", 
    "ValidationSummary",
    "ValidationStage",
    "ValidationStatus",
    "ValidationSeverity",
]
