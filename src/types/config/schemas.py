"""
Configuration schemas for all HADES modules.

This module provides comprehensive Pydantic schemas for validating
all configuration files in the src/config/ directory.
"""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, validator, root_validator
from pathlib import Path
from enum import Enum


class StorageBackend(str, Enum):
    """Supported storage backends."""
    ARANGODB = "arangodb"
    NANO = "nano"
    MILVUS = "milvus"
    CHROMA = "chroma"


class ArangoDBConfig(BaseModel):
    """ArangoDB storage configuration schema."""
    connection: Dict[str, Any] = Field(..., description="Connection settings")
    collections: Dict[str, str] = Field(..., description="Collection names")
    indexes: List[Dict[str, Any]] = Field(default_factory=list)
    performance: Dict[str, Any] = Field(default_factory=dict)
    graph: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('connection')
    def validate_connection(cls, v):
        required = ['url', 'database']
        for field in required:
            if field not in v:
                raise ValueError(f"Missing required connection field: {field}")
        return v


class ISNEConfig(BaseModel):
    """ISNE configuration schema."""
    model: Dict[str, Any] = Field(..., description="Model architecture settings")
    training: Dict[str, Any] = Field(..., description="Training parameters")
    bootstrap: Optional[Dict[str, Any]] = Field(None, description="Bootstrap settings")
    directory_aware: bool = Field(True, description="Enable directory awareness")
    
    @validator('model')
    def validate_model(cls, v):
        if 'embedding_dim' not in v:
            v['embedding_dim'] = 768  # Jina v4 default
        return v


class JinaV4Config(BaseModel):
    """Jina v4 processor configuration schema."""
    model_name: str = Field("jinaai/jina-embeddings-v4", description="Model identifier")
    device: str = Field("cuda", pattern="^(cuda|cpu)$")
    batch_size: int = Field(32, gt=0)
    max_length: int = Field(8192, gt=0)
    late_chunking: Dict[str, Any] = Field(default_factory=dict)
    keyword_extraction: Dict[str, Any] = Field(default_factory=dict)
    multimodal: Dict[str, Any] = Field(default_factory=dict)


class PathRAGConfig(BaseModel):
    """PathRAG configuration schema."""
    max_paths: int = Field(10, gt=0)
    path_length: int = Field(3, gt=0)
    weight_dimensions: List[str] = Field(default_factory=lambda: ["semantic", "structural", "temporal"])
    retrieval: Dict[str, Any] = Field(default_factory=dict)
    query_as_graph: bool = Field(True)


class ValidationConfig(BaseModel):
    """Validation configuration schema."""
    validation: Dict[str, Any] = Field(..., description="Validation settings")
    checks: Dict[str, Any] = Field(..., description="Validation checks")
    metrics: Dict[str, Any] = Field(default_factory=dict)
    performance: Dict[str, Any] = Field(default_factory=dict)