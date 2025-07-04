"""
Component contracts for HADES.

This module defines the contracts (data structures) used between components
in the HADES pipeline, particularly for communication between Jina v4 processor
and ISNE enhancement components.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import Field

from ..common import BaseSchema, EmbeddingVector, NodeID, DocumentID


class DocumentChunk(BaseSchema):
    """Represents a chunk of a document with its metadata."""
    id: str
    content: str
    document_id: DocumentID
    chunk_index: int
    chunk_size: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    

class ChunkEmbedding(BaseSchema):
    """Represents an embedding for a document chunk."""
    chunk_id: str
    embedding: EmbeddingVector
    embedding_dimension: int
    model_name: str
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    

class EmbeddingInput(BaseSchema):
    """Input format for embedding generation."""
    texts: List[str]
    model_name: str
    options: Dict[str, Any] = Field(default_factory=dict)
    

class GraphEnhancementInput(BaseSchema):
    """Input format for graph-based enhancement (ISNE)."""
    embeddings: List[EmbeddingVector]
    node_ids: List[NodeID]
    graph_structure: Dict[str, Any]  # Contains nodes and edges
    enhancement_type: str = "isne_default"
    options: Dict[str, Any] = Field(default_factory=dict)