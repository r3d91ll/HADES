"""
Base classes and types for the API module.

This module provides base classes for RAG engines and common types
used across the API layer.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from src.types.common import BaseSchema


class APIResponse(BaseModel):
    """Standard API response format."""
    success: bool = Field(..., description="Whether the request was successful")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message if failed")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    

class APIError(Exception):
    """Base exception for API errors."""
    pass


class EngineType(str, Enum):
    """Supported RAG engine types."""
    PATHRAG = "pathrag"
    GRAPHRAG = "graphrag"
    VECTORRAG = "vectorrag"
    HYBRID = "hybrid"


class RetrievalMode(str, Enum):
    """Retrieval modes for engines."""
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"
    PATHRAG = "pathrag"
    SEMANTIC = "semantic"
    STANDARD = "standard"
    KEYWORD = "keyword"


class EngineRetrievalRequest(BaseModel):
    """Request format for engine retrieval."""
    query: str = Field(..., description="Query text")
    mode: RetrievalMode = Field(RetrievalMode.HYBRID, description="Retrieval mode")
    top_k: int = Field(10, description="Number of results", ge=1, le=100)
    max_token_for_context: int = Field(1000, description="Max tokens for context")
    max_path_length: Optional[int] = Field(5, description="Max path length for PathRAG")
    filters: Optional[Dict[str, Any]] = Field(None, description="Query filters")
    engine_params: Optional[Dict[str, Any]] = Field(None, description="Engine-specific parameters")


class EngineRetrievalResult(BaseModel):
    """Result format from engine retrieval."""
    content: str = Field(..., description="Retrieved content")
    score: float = Field(..., description="Relevance score")
    source: str = Field(..., description="Source document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Result metadata")
    engine_metadata: Dict[str, Any] = Field(default_factory=dict, description="Engine-specific metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class BaseRAGEngine(ABC):
    """Abstract base class for RAG engines."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize engine with configuration."""
        self.config = config or {}
        self._engine_type = EngineType.PATHRAG
        self.is_initialized = False
        self._last_config_update = datetime.utcnow()
    
    @property
    @abstractmethod
    def engine_type(self) -> EngineType:
        """Get engine type."""
        pass
    
    @property
    @abstractmethod
    def supported_modes(self) -> List[RetrievalMode]:
        """Get supported retrieval modes."""
        pass
    
    @abstractmethod
    async def _execute_retrieval(self, request: EngineRetrievalRequest) -> List[EngineRetrievalResult]:
        """Execute the actual retrieval logic."""
        pass
    
    async def retrieve(self, request: EngineRetrievalRequest) -> List[EngineRetrievalResult]:
        """Perform retrieval based on request."""
        # Validate mode is supported
        if request.mode not in self.supported_modes:
            raise ValueError(f"Mode {request.mode} not supported by {self.engine_type} engine")
        
        # Execute retrieval
        return await self._execute_retrieval(request)
    
    @abstractmethod
    async def configure(self, config: Dict[str, Any]) -> bool:
        """Update engine configuration."""
        pass
    
    @abstractmethod
    async def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check engine health."""
        pass
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get engine metrics."""
        return {
            "engine_type": self.engine_type.value,
            "is_initialized": self.is_initialized,
            "supported_modes": [mode.value for mode in self.supported_modes],
            "last_config_update": self._last_config_update.isoformat()
        }
    
    @property
    def name(self) -> str:
        """Get engine name."""
        return self.engine_type.value
    
    @property
    def status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "name": self.name,
            "type": self.engine_type.value,
            "initialized": self.is_initialized,
            "config": self.config
        }


__all__ = [
    "APIResponse",
    "APIError",
    "EngineType",
    "RetrievalMode",
    "EngineRetrievalRequest",
    "EngineRetrievalResult",
    "BaseRAGEngine"
]