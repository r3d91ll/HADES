"""Document storage type definitions.

This module defines type annotations for document storage operations, including
document metadata, storage operations, and retrieval responses.
"""

from typing import Dict, List, Any, Optional, Union, TypedDict, Literal
from datetime import datetime
from uuid import UUID

from src.types.common import NodeData, EdgeData, NodeID


# Document type literals
DocumentFormat = Literal["text", "pdf", "html", "code", "markdown", "json", "yaml", "xml"]

# Document status literals
DocumentStatus = Literal["pending", "processing", "processed", "error", "deleted"]

# Document content types
DocumentContentType = Literal["document", "chunk", "section", "page", "paragraph", "text"]


class DocumentMetadata(TypedDict, total=False):
    """Document metadata dictionary type."""
    
    title: str
    """Title of the document."""
    
    author: str
    """Author of the document."""
    
    created_at: Union[str, datetime]
    """Creation timestamp (ISO format string or datetime object)."""
    
    modified_at: Union[str, datetime]
    """Last modification timestamp (ISO format string or datetime object)."""
    
    source: str
    """Source of the document (e.g., filename, URL)."""
    
    content_type: DocumentFormat
    """Format of the document content."""
    
    language: str
    """Language of the document content."""
    
    word_count: int
    """Number of words in the document."""
    
    page_count: int
    """Number of pages in the document (for PDFs)."""
    
    version: str
    """Document version identifier."""
    
    tags: List[str]
    """List of tags associated with the document."""
    
    status: DocumentStatus
    """Processing status of the document."""
    
    error_message: Optional[str]
    """Error message if processing failed."""
    
    custom_metadata: Dict[str, Any]
    """Additional custom metadata fields."""


class DocumentStorageRequest(TypedDict, total=False):
    """Request type for document storage operations."""
    
    id: str
    """Document identifier (optional, will be generated if not provided)."""
    
    content: str
    """Document content."""
    
    metadata: DocumentMetadata
    """Document metadata."""
    
    chunks: List[Dict[str, Any]]
    """Document chunks (if already chunked)."""
    
    embeddings: Dict[str, Any]
    """Document embeddings (if already embedded)."""
    
    collection: str
    """Collection to store the document in."""
    
    update_if_exists: bool
    """Whether to update the document if it already exists."""


class DocumentQueryFilter(TypedDict, total=False):
    """Filter type for document queries."""
    
    metadata: Dict[str, Any]
    """Metadata field filters."""
    
    content_contains: str
    """Filter for documents containing specific text."""
    
    created_after: Union[str, datetime]
    """Filter for documents created after a timestamp."""
    
    created_before: Union[str, datetime]
    """Filter for documents created before a timestamp."""
    
    modified_after: Union[str, datetime]
    """Filter for documents modified after a timestamp."""
    
    modified_before: Union[str, datetime]
    """Filter for documents modified before a timestamp."""
    
    tags: List[str]
    """Filter for documents with specific tags (ANY match)."""
    
    all_tags: List[str]
    """Filter for documents with all specified tags (ALL match)."""
    
    status: DocumentStatus
    """Filter for documents with a specific status."""
    
    content_type: DocumentFormat
    """Filter for documents with a specific content type."""


class DocumentStorageResponse(TypedDict, total=False):
    """Response type for document storage operations."""
    
    success: bool
    """Whether the operation was successful."""
    
    document_id: str
    """ID of the stored document."""
    
    error: Optional[str]
    """Error message if the operation failed."""
    
    timestamp: Union[str, datetime]
    """Timestamp of the operation."""
    
    operation: str
    """Type of operation performed."""


class DocumentRetrievalResponse(TypedDict, total=False):
    """Response type for document retrieval operations."""
    
    document_id: str
    """ID of the retrieved document."""
    
    content: Optional[str]
    """Document content (may be None for metadata-only retrieval)."""
    
    metadata: DocumentMetadata
    """Document metadata."""
    
    chunks: Optional[List[Dict[str, Any]]]
    """Document chunks (if requested)."""
    
    embeddings: Optional[Dict[str, Any]]
    """Document embeddings (if requested)."""
    
    version: str
    """Document version identifier."""
    
    retrieved_at: Union[str, datetime]
    """Timestamp of retrieval."""


class DocumentIndexConfig(TypedDict, total=False):
    """Configuration for document indexing."""
    
    fields: List[str]
    """Fields to index for text search."""
    
    vector_dimensions: int
    """Dimensions of vector embeddings."""
    
    distance_metric: Literal["cosine", "euclidean", "dot_product"]
    """Distance metric for vector similarity."""
    
    hnsw_config: Dict[str, Any]
    """Configuration for HNSW index."""
    
    index_chunks: bool
    """Whether to index document chunks separately."""
    
    language_specific: bool
    """Whether to use language-specific indexing."""


class BulkOperationResult(TypedDict):
    """Result of a bulk operation."""
    
    success_count: int
    """Number of successful operations."""
    
    error_count: int
    """Number of failed operations."""
    
    errors: Dict[str, str]
    """Mapping of document IDs to error messages for failed operations."""
    
    operation: str
    """Type of operation performed."""
    
    timestamp: Union[str, datetime]
    """Timestamp of the operation."""


# Type aliases
DocumentID = str
"""Alias for document ID type."""


class ExtendedDocumentMetadata(DocumentMetadata, total=False):
    """Extended document metadata with additional fields used in the storage service."""
    
    # Extended fields beyond base DocumentMetadata
    symbol_type: str
    """Type of symbol if this is a code symbol document."""
    
    chunk_index: int
    """Index of the chunk if this is a document chunk."""
    
    token_count: int
    """Count of tokens in the document or chunk."""
    
    content_hash: str
    """Hash of the document content for deduplication."""
    
    isne_enhanced: bool
    """Whether this document has ISNE-enhanced embeddings."""
    
    parent_id: Optional[str]
    """ID of the parent document for chunks."""


class ExtendedNodeData(TypedDict, total=False):
    """Extended node data with additional fields for storage."""
    
    # Base NodeData fields
    id: str
    """Unique identifier for the node."""
    
    type: str
    """Type of the node (e.g., 'document', 'chunk')."""
    
    content: Optional[str]
    """Content of the node."""
    
    metadata: Dict[str, Any]
    """Metadata associated with the node."""
    
    # Extended fields
    _key: str
    """Optional database-specific key field."""
    
    system_metadata: Dict[str, Any]
    """Additional metadata for system use."""


class ExtendedEdgeData(TypedDict, total=False):
    """Extended edge data with additional fields used in the storage service."""
    
    # Base EdgeData fields
    from_id: NodeID
    """ID of the source node."""
    
    to_id: NodeID
    """ID of the destination node."""
    
    relation: str
    """Type of relationship between nodes."""
    
    # Extended fields
    weight: float
    """Weight or strength of the relationship."""
    
    similarity_score: float
    """Similarity score for similarity-based edges."""
    
    relation_type: str
    """More specific type of the relation."""
    
    metadata: Dict[str, Any]
    """Additional metadata for the edge."""


class ExtendedDocumentStorageResponse(DocumentStorageResponse, total=False):
    """Extended document storage response with additional fields."""
    
    status: str
    """Status of the storage operation."""
    
    message: str
    """Descriptive message about the operation result."""
    
    chunks_stored: int
    """Number of chunks stored during the operation."""
    
    embeddings_stored: int
    """Number of embeddings stored during the operation."""


# Export all document-related types
__all__ = [
    # Basic types
    'DocumentFormat',
    'DocumentStatus',
    'DocumentContentType',
    'DocumentID',
    
    # Core document types
    'DocumentMetadata',
    'DocumentStorageRequest',
    'DocumentQueryFilter',
    'DocumentStorageResponse',
    'DocumentRetrievalResponse',
    'DocumentIndexConfig',
    'BulkOperationResult',
    
    # Extended types
    'ExtendedDocumentMetadata',
    'ExtendedNodeData',
    'ExtendedEdgeData',
    'ExtendedDocumentStorageResponse',
]
