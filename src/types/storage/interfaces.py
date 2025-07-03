"""
Storage interfaces and types for HADES.

This module defines the interfaces and data structures for storage components,
including document stores, vector databases, and graph databases.
"""

from typing import Protocol, Dict, Any, Optional, List, Tuple, runtime_checkable
from abc import abstractmethod
from datetime import datetime

from ..common import (
    BaseSchema,
    NodeID,
    DocumentID,
    EdgeID,
    EmbeddingVector,
    DocumentType,
    RelationType,
    ProcessingStatus
)


class DocumentData(BaseSchema):
    """Document data structure for storage."""
    document_id: DocumentID
    content: str
    document_type: DocumentType
    title: Optional[str] = None
    source_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())
    updated_at: Optional[datetime] = None
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    chunks: Optional[List[Dict[str, Any]]] = None
    embeddings: Optional[List[EmbeddingVector]] = None


class ChunkData(BaseSchema):
    """Chunk data structure for storage."""
    chunk_id: str
    document_id: DocumentID
    content: str
    chunk_index: int
    start_pos: int
    end_pos: int
    embedding: Optional[EmbeddingVector] = None
    keywords: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VectorSearchResult(BaseSchema):
    """Result from vector similarity search."""
    document_id: DocumentID
    chunk_id: Optional[str] = None
    score: float
    content: str
    embedding: Optional[EmbeddingVector] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphNodeData(BaseSchema):
    """Graph node data structure."""
    node_id: NodeID
    node_type: str
    properties: Dict[str, Any]
    embedding: Optional[EmbeddingVector] = None
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())
    updated_at: Optional[datetime] = None


class GraphEdgeData(BaseSchema):
    """Graph edge data structure."""
    edge_id: EdgeID
    source_id: NodeID
    target_id: NodeID
    edge_type: RelationType
    weight: float = 1.0
    properties: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())


class SupraWeight(BaseSchema):
    """Multi-dimensional edge weight structure."""
    edge_id: EdgeID
    dimensions: Dict[str, float]  # e.g., {"semantic": 0.8, "structural": 0.6, "temporal": 0.5}
    aggregated_weight: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class DocumentRepository(Protocol):
    """Interface for document storage repositories."""
    
    @abstractmethod
    async def create(self, document: DocumentData) -> DocumentID:
        """Create a new document."""
        ...
    
    @abstractmethod
    async def read(self, document_id: DocumentID) -> Optional[DocumentData]:
        """Read a document by ID."""
        ...
    
    @abstractmethod
    async def update(self, document_id: DocumentID, updates: Dict[str, Any]) -> bool:
        """Update a document."""
        ...
    
    @abstractmethod
    async def delete(self, document_id: DocumentID) -> bool:
        """Delete a document."""
        ...
    
    @abstractmethod
    async def list(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[DocumentData]:
        """List documents with optional filters."""
        ...
    
    @abstractmethod
    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[DocumentData]:
        """Search documents by text query."""
        ...


@runtime_checkable
class VectorRepository(Protocol):
    """Interface for vector storage repositories."""
    
    @abstractmethod
    async def add_embedding(
        self,
        embedding_id: str,
        embedding: EmbeddingVector,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add an embedding to the store."""
        ...
    
    @abstractmethod
    async def get_embedding(self, embedding_id: str) -> Optional[Tuple[EmbeddingVector, Dict[str, Any]]]:
        """Get an embedding by ID."""
        ...
    
    @abstractmethod
    async def search_similar(
        self,
        query_embedding: EmbeddingVector,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None
    ) -> List[VectorSearchResult]:
        """Search for similar embeddings."""
        ...
    
    @abstractmethod
    async def delete_embedding(self, embedding_id: str) -> bool:
        """Delete an embedding."""
        ...
    
    @abstractmethod
    async def create_index(self, index_type: str = "hnsw", **kwargs: Any) -> bool:
        """Create or update the vector index."""
        ...


@runtime_checkable
class GraphRepository(Protocol):
    """Interface for graph storage repositories."""
    
    @abstractmethod
    async def add_node(self, node: GraphNodeData) -> NodeID:
        """Add a node to the graph."""
        ...
    
    @abstractmethod
    async def get_node(self, node_id: NodeID) -> Optional[GraphNodeData]:
        """Get a node by ID."""
        ...
    
    @abstractmethod
    async def update_node(self, node_id: NodeID, updates: Dict[str, Any]) -> bool:
        """Update a node."""
        ...
    
    @abstractmethod
    async def delete_node(self, node_id: NodeID) -> bool:
        """Delete a node and its edges."""
        ...
    
    @abstractmethod
    async def add_edge(self, edge: GraphEdgeData) -> EdgeID:
        """Add an edge to the graph."""
        ...
    
    @abstractmethod
    async def get_edge(self, edge_id: EdgeID) -> Optional[GraphEdgeData]:
        """Get an edge by ID."""
        ...
    
    @abstractmethod
    async def get_neighbors(
        self,
        node_id: NodeID,
        edge_types: Optional[List[RelationType]] = None,
        direction: str = "both",  # "in", "out", "both"
        max_distance: int = 1
    ) -> List[Tuple[GraphNodeData, GraphEdgeData]]:
        """Get neighboring nodes and connecting edges."""
        ...
    
    @abstractmethod
    async def find_path(
        self,
        start_id: NodeID,
        end_id: NodeID,
        max_length: int = 5,
        edge_types: Optional[List[RelationType]] = None
    ) -> Optional[List[Tuple[GraphNodeData, Optional[GraphEdgeData]]]]:
        """Find a path between two nodes."""
        ...
    
    @abstractmethod
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a graph query (e.g., AQL for ArangoDB)."""
        ...


class StorageConfig(BaseSchema):
    """Configuration for storage systems."""
    storage_type: str  # "memory", "arangodb", "neo4j", "postgres", etc.
    connection_params: Dict[str, Any] = Field(default_factory=dict)
    collection_names: Dict[str, str] = Field(default_factory=dict)
    index_config: Dict[str, Any] = Field(default_factory=dict)
    cache_config: Optional[Dict[str, Any]] = None
    

class StorageMetrics(BaseSchema):
    """Metrics for storage operations."""
    total_documents: int = 0
    total_chunks: int = 0
    total_embeddings: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    storage_size_bytes: Optional[int] = None
    index_size_bytes: Optional[int] = None
    last_updated: datetime = Field(default_factory=lambda: datetime.utcnow())
    

# Import Field for Pydantic models
from pydantic import Field