"""Storage component types."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from ..common import BaseSchema, DocumentID, EmbeddingVector, NodeID
from ..storage.interfaces import DocumentData, VectorSearchResult


class StorageInput(BaseSchema):
    """Input for storage operations."""
    
    document_id: DocumentID = Field(..., description="Document identifier")
    content: str = Field(..., description="Document content")
    embedding: Optional[EmbeddingVector] = Field(None, description="Document embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    chunk_id: Optional[str] = Field(None, description="Chunk identifier if this is a chunk")
    parent_id: Optional[DocumentID] = Field(None, description="Parent document ID if this is a chunk")
    options: Dict[str, Any] = Field(default_factory=dict, description="Storage options")


class StoredItem(BaseSchema):
    """Stored item representation."""
    
    id: str = Field(..., description="Storage item ID", alias="item_id")
    item_id: Optional[str] = Field(None, description="Storage item ID (alias)")
    document_id: DocumentID = Field(..., description="Document identifier")
    content: str = Field(..., description="Stored content")
    embedding: Optional[EmbeddingVector] = Field(None, description="Stored embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Stored metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    storage_location: Optional[str] = Field(None, description="Storage location")
    storage_timestamp: Optional[datetime] = Field(None, description="Storage timestamp")
    index_status: Optional[str] = Field(None, description="Index status")
    retrieval_metadata: Optional[Dict[str, Any]] = Field(default_factory=lambda: {}, description="Retrieval metadata")
    
    def __init__(self, **data: Any) -> None:
        """Initialize with field aliasing."""
        if 'item_id' in data and 'id' not in data:
            data['id'] = data['item_id']
        elif 'id' in data and 'item_id' not in data:
            data['item_id'] = data['id']
        super().__init__(**data)


class StorageOutput(BaseSchema):
    """Output from storage operations."""
    
    success: bool = Field(..., description="Whether the operation succeeded")
    item: Optional[StoredItem] = Field(None, description="Stored item if successful")
    error: Optional[str] = Field(None, description="Error message if failed", alias="error_message")
    error_message: Optional[str] = Field(None, description="Error message (alias)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional output metadata")
    document_id: Optional[DocumentID] = Field(None, description="Document ID")
    documents: Optional[List[Dict[str, Any]]] = Field(None, description="Stored documents")
    
    def __init__(self, **data: Any) -> None:
        """Initialize with field aliasing."""
        if 'error_message' in data and 'error' not in data:
            data['error'] = data['error_message']
        elif 'error' in data and 'error_message' not in data:
            data['error_message'] = data['error']
        super().__init__(**data)


class QueryInput(BaseSchema):
    """Input for query operations."""
    
    query: str = Field(..., description="Query text")
    embedding: Optional[EmbeddingVector] = Field(None, description="Query embedding", alias="query_embedding")
    query_embedding: Optional[EmbeddingVector] = Field(None, description="Query embedding (alias)")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Query filters")
    limit: int = Field(10, description="Maximum number of results", ge=1, le=1000, alias="top_k")
    top_k: Optional[int] = Field(None, description="Maximum results (alias)")
    offset: int = Field(0, description="Result offset", ge=0)
    include_metadata: bool = Field(True, description="Whether to include metadata in results")
    search_options: Dict[str, Any] = Field(default_factory=dict, description="Search options")
    
    def __init__(self, **data: Any) -> None:
        """Initialize with field aliasing."""
        if 'query_embedding' in data and 'embedding' not in data:
            data['embedding'] = data['query_embedding']
        elif 'embedding' in data and 'query_embedding' not in data:
            data['query_embedding'] = data['embedding']
            
        if 'top_k' in data and 'limit' not in data:
            data['limit'] = data['top_k']
        elif 'limit' in data and 'top_k' not in data:
            data['top_k'] = data['limit']
        super().__init__(**data)


class RetrievalResult(BaseSchema):
    """Single retrieval result."""
    
    id: str = Field(..., description="Result ID", alias="item_id")
    item_id: Optional[str] = Field(None, description="Result ID (alias)")
    document_id: DocumentID = Field(..., description="Document identifier")
    content: str = Field(..., description="Retrieved content")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Result metadata")
    embedding: Optional[EmbeddingVector] = Field(None, description="Result embedding")
    path_info: Optional[Dict[str, Any]] = Field(None, description="Path information for PathRAG")
    
    def __init__(self, **data: Any) -> None:
        """Initialize with field aliasing."""
        if 'item_id' in data and 'id' not in data:
            data['id'] = data['item_id']
        elif 'id' in data and 'item_id' not in data:
            data['item_id'] = data['id']
        super().__init__(**data)


class QueryOutput(BaseSchema):
    """Output from query operations."""
    
    success: bool = Field(..., description="Whether the query succeeded")
    results: List[RetrievalResult] = Field(default_factory=list, description="Query results")
    total_count: int = Field(0, description="Total number of matching results")
    query_time_ms: float = Field(0.0, description="Query execution time in milliseconds", alias="search_time")
    search_time: Optional[float] = Field(None, description="Query time (alias)")
    error: Optional[str] = Field(None, description="Error message if failed", alias="errors")
    errors: Optional[Union[str, List[str]]] = Field(None, description="Error(s) (alias)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional query metadata")
    search_stats: Optional[Dict[str, Any]] = Field(None, description="Search statistics")
    query_info: Optional[Dict[str, Any]] = Field(None, description="Query information")
    
    def __init__(self, **data: Any) -> None:
        """Initialize with field aliasing."""
        if 'search_time' in data and 'query_time_ms' not in data:
            data['query_time_ms'] = data['search_time']
        elif 'query_time_ms' in data and 'search_time' not in data:
            data['search_time'] = data['query_time_ms']
            
        if 'errors' in data and 'error' not in data:
            if isinstance(data['errors'], list):
                data['error'] = '; '.join(data['errors'])
            else:
                data['error'] = data['errors']
        elif 'error' in data and 'errors' not in data:
            data['errors'] = data['error']
        super().__init__(**data)