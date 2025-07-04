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


class EngineType(str, Enum):
    """Supported RAG engine types."""
    PATHRAG = "pathrag"
    GRAPHRAG = "graphrag"
    VECTORRAG = "vectorrag"
    HYBRID = "hybrid"


class RetrievalMode(str, Enum):
    """Retrieval modes for engines."""
    STANDARD = "standard"
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class EngineRetrievalRequest(BaseSchema):
    """Request format for engine retrieval."""
    query: str
    mode: RetrievalMode = RetrievalMode.STANDARD
    top_k: int = 10
    filters: Dict[str, Any] = Field(default_factory=dict)
    options: Dict[str, Any] = Field(default_factory=dict)


class EngineRetrievalResult(BaseSchema):
    """Result format from engine retrieval."""
    content: str
    score: float
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BaseRAGEngine(ABC):
    """Abstract base class for RAG engines."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize engine with configuration."""
        self.config = config or {}
        self.engine_type = EngineType.PATHRAG
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the engine."""
        pass
    
    @abstractmethod
    async def retrieve(self, request: EngineRetrievalRequest) -> List[EngineRetrievalResult]:
        """Perform retrieval based on request."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the engine."""
        pass
    
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
    "EngineType",
    "RetrievalMode",
    "EngineRetrievalRequest",
    "EngineRetrievalResult",
    "BaseRAGEngine"
]