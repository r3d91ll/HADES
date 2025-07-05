"""RAG strategy component types."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field

from ..common import BaseSchema, DocumentID, EmbeddingVector


class RAGMode(str, Enum):
    """RAG operation modes."""
    
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"
    PATH = "path"
    PATHRAG = "pathrag"  # PathRAG-specific mode
    ADAPTIVE = "adaptive"
    
    # Additional modes for different strategies
    DENSE = "dense"
    SPARSE = "sparse"
    RERANK = "rerank"


class PathInfo(BaseSchema):
    """Information about a retrieval path."""
    
    path_id: str = Field(..., description="Unique path identifier")
    nodes: List[str] = Field(..., description="Nodes in the path")
    edges: List[Dict[str, Any]] = Field(default_factory=list, description="Edges in the path")
    weight: float = Field(..., description="Path weight/score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Path metadata")


class RAGResult(BaseSchema):
    """Single RAG result."""
    
    id: str = Field(..., description="Result ID")
    document_id: DocumentID = Field(..., description="Document identifier")
    content: str = Field(..., description="Retrieved content")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Result metadata")
    paths: List[PathInfo] = Field(default_factory=list, description="Paths to this result")
    embedding: Optional[EmbeddingVector] = Field(None, description="Result embedding")


class RAGStrategyInput(BaseSchema):
    """Input for RAG strategy operations."""
    
    query: str = Field(..., description="Query text")
    mode: RAGMode = Field(RAGMode.HYBRID, description="RAG mode to use")
    embedding: Optional[EmbeddingVector] = Field(None, description="Query embedding")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Query filters")
    limit: int = Field(10, description="Maximum number of results", ge=1, le=100, alias="top_k")
    top_k: Optional[int] = Field(None, description="Maximum results (alias)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy-specific parameters")
    
    def __init__(self, **data: Any) -> None:
        """Initialize with field aliasing."""
        if 'top_k' in data and 'limit' not in data:
            data['limit'] = data['top_k']
        elif 'limit' in data and 'top_k' not in data:
            data['top_k'] = data['limit']
        super().__init__(**data)


class RAGStrategyOutput(BaseSchema):
    """Output from RAG strategy operations."""
    
    success: bool = Field(..., description="Whether the operation succeeded")
    results: List[RAGResult] = Field(default_factory=list, description="RAG results")
    total_count: int = Field(0, description="Total number of results")
    execution_time_ms: float = Field(0.0, description="Execution time in milliseconds")
    mode_used: RAGMode = Field(..., description="The RAG mode that was used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    error: Optional[str] = Field(None, description="Error message if failed")
    query: Optional[str] = Field(None, description="The query that was processed")