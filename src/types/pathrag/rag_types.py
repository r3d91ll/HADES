"""
RAG Strategy Types

This module defines all types related to the RAG (Retrieval-Augmented Generation)
strategy pattern, including PathRAG-specific types.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import Field

from ..common import BaseSchema, DocumentID, NodeID, EmbeddingVector


class RAGMode(str, Enum):
    """RAG operation modes."""
    STANDARD = "standard"
    PATHRAG = "pathrag"
    HYBRID = "hybrid"
    GRAPH = "graph"
    VECTOR = "vector"
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    NAIVE = "naive"
    GLOBAL = "global"
    LOCAL = "local"


class PathInfo(BaseSchema):
    """Information about a path in the knowledge graph."""
    path: List[NodeID]
    path_string: str  # Human-readable path representation
    score: float
    length: int
    edge_types: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGResult(BaseSchema):
    """Individual result from RAG retrieval."""
    content: str
    score: float
    source: str  # Source document or node ID
    document_id: Optional[DocumentID] = None
    node_id: Optional[NodeID] = None
    chunk_id: Optional[str] = None
    path: Optional[PathInfo] = None
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    # Additional fields for PathRAG
    item_id: Optional[str] = None
    diversity_score: Optional[float] = None


class RAGStrategyInput(BaseSchema):
    """Input for RAG strategy execution."""
    query: str
    mode: RAGMode = RAGMode.STANDARD
    top_k: int = 10
    similarity_threshold: float = 0.7
    max_path_length: int = 5
    include_metadata: bool = True
    filters: Dict[str, Any] = Field(default_factory=dict)
    query_embedding: Optional[EmbeddingVector] = None
    search_options: Dict[str, Any] = Field(default_factory=dict)
    context: Optional[str] = None
    user_id: Optional[str] = None
    max_token_for_context: Optional[int] = None  # For context length management


class RAGStrategyOutput(BaseSchema):
    """Output from RAG strategy execution."""
    results: List[RAGResult]
    total_results: int
    execution_time: float
    mode_used: RAGMode
    query: str
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    debug_info: Optional[Dict[str, Any]] = None
    # Additional fields for PathRAG
    generated_answer: Optional[str] = None
    answer_confidence: Optional[float] = None
    paths_explored: Optional[int] = None
    graph_stats: Optional[Dict[str, Any]] = None
    retrieval_stats: Optional[Dict[str, Any]] = None
    query_processing_time: Optional[float] = None
    mode: Optional[RAGMode] = None  # Alias for mode_used
    errors: Optional[List[str]] = None


class PathRAGConfig(BaseSchema):
    """Configuration specific to PathRAG strategy."""
    enable_path_expansion: bool = True
    path_scoring_method: str = "weighted"  # weighted, uniform, decay
    include_semantic_similarity: bool = True
    include_structural_similarity: bool = True
    max_paths_per_node: int = 5
    min_path_score: float = 0.1
    edge_weight_factor: float = 0.3
    node_weight_factor: float = 0.7


class RetrievalStats(BaseSchema):
    """Statistics from retrieval operations."""
    nodes_visited: int = 0
    edges_traversed: int = 0
    paths_evaluated: int = 0
    documents_retrieved: int = 0
    vectors_compared: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


__all__ = [
    "RAGMode",
    "PathInfo", 
    "RAGResult",
    "RAGStrategyInput",
    "RAGStrategyOutput",
    "PathRAGConfig",
    "RetrievalStats"
]