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

from .alerts import (
    AlertLevel,
    Alert,
)

from .api import (
    WriteRequest,
    QueryRequest,
    QueryResponse,
    WriteResponse,
    StatusResponse,
    QueryResult,
)

from .config import (
    EngineConfig,
    PipelineConfig,
    ArangoDBConfig,
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
    
    # Alert types
    "AlertLevel",
    "Alert",
    
    # API types
    "WriteRequest",
    "QueryRequest",
    "QueryResponse", 
    "WriteResponse",
    "StatusResponse",
    "QueryResult",
    
    # Config types
    "EngineConfig",
    "PipelineConfig",
    "ArangoDBConfig",
]
