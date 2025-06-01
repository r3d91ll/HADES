"""Document type definitions for document processing.

This module provides comprehensive type definitions for documents,
including structure, relationships, and validation requirements.
"""

from typing import Any, Dict, List, Optional, Union, TypedDict
from datetime import datetime
from pathlib import Path

from src.types.docproc.metadata import EntityDict, MetadataDict


class DocumentDict(TypedDict, total=False):
    """Dictionary representation of a document.
    
    This TypedDict defines the standardized structure for documents
    in the HADES-PathRAG ingestion pipeline. It serves as the
    contract between different pipeline stages.
    """
    
    document_id: str
    """Unique identifier for the document."""
    
    source_path: str
    """Path to the source document."""
    
    content: str
    """Document content as text."""
    
    content_type: str
    """Type of content (e.g., 'text', 'code')."""
    
    format: str
    """Format of the document (e.g., 'markdown', 'python')."""
    
    metadata: MetadataDict
    """Document metadata."""
    
    entities: List[EntityDict]
    """Entities extracted from the document."""
    
    chunks: Optional[List[Dict[str, Any]]]
    """Document chunks after chunking."""
    
    embedding: Optional[List[float]]
    """Document-level embedding if available."""
    
    embedding_model: Optional[str]
    """Model used to generate embedding."""
    
    error: Optional[str]
    """Error message if processing failed."""
    
    warnings: Optional[List[str]]
    """Warnings generated during processing."""
    
    processing_time: Optional[float]
    """Time taken to process the document (seconds)."""
    
    processed_at: Optional[Union[str, datetime]]
    """Timestamp when the document was processed."""
    
    raw_content: Optional[str]
    """Original unprocessed content (when available)."""


# Document relationship types for graph construction
class DocumentRelationDict(TypedDict, total=False):
    """Dictionary representation of a relationship between documents."""
    
    source_id: str
    """ID of the source document."""
    
    target_id: str
    """ID of the target document."""
    
    relation_type: str
    """Type of relationship (e.g., 'references', 'imports')."""
    
    confidence: float
    """Confidence score for the relationship (0.0-1.0)."""
    
    metadata: Dict[str, Any]
    """Additional metadata about the relationship."""


class DocumentCollectionDict(TypedDict, total=False):
    """Dictionary representation of a collection of documents."""
    
    collection_id: str
    """Unique identifier for the collection."""
    
    name: str
    """Name of the collection."""
    
    description: Optional[str]
    """Description of the collection."""
    
    documents: List[str]
    """List of document IDs in the collection."""
    
    metadata: Dict[str, Any]
    """Additional metadata about the collection."""
    
    created_at: Union[str, datetime]
    """Timestamp when the collection was created."""
    
    modified_at: Union[str, datetime]
    """Timestamp when the collection was last modified."""
