"""
Simplified contracts for testing model engine components.

This is a test-only version that avoids complex Pydantic v2 validation
to focus on testing the core component functionality.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ComponentType(str, Enum):
    """Component type enumeration."""
    MODEL_ENGINE = "model_engine"
    CHUNKING = "chunking"
    DOCPROC = "docproc"
    EMBEDDING = "embedding"
    STORAGE = "storage"


class ProcessingStatus(str, Enum):
    """Processing status enumeration."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    PENDING = "pending"


class ComponentMetadata(BaseModel):
    """Basic component metadata for testing."""
    component_type: ComponentType
    component_name: str
    component_version: str
    processing_time: Optional[float] = None
    processed_at: Optional[datetime] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    status: ProcessingStatus = ProcessingStatus.SUCCESS


class ModelEngineInput(BaseModel):
    """Simplified input contract for model engine operations."""
    requests: List[Dict[str, Any]]
    engine_config: Dict[str, Any] = Field(default_factory=dict)
    batch_config: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {"extra": "allow"}


class ModelInferenceResult(BaseModel):
    """Simplified model inference result."""
    request_id: str
    response_data: Dict[str, Any] = Field(default_factory=dict)
    processing_time: float = 0.0
    error: Optional[str] = None


class ModelEngineOutput(BaseModel):
    """Simplified output contract for model engine operations."""
    results: List[ModelInferenceResult]
    metadata: ComponentMetadata
    engine_stats: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)