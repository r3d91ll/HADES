"""Extended document types for the storage module.

This module contains extended document types used by the storage module
that build upon the base document types.
"""

from typing import Dict, List, Any, Optional, Union, TypedDict
from src.types.storage.document import DocumentMetadata, DocumentStorageResponse
from src.types.common import NodeID


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
