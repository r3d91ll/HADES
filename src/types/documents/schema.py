"""Document schema type definitions using TypedDict.

This module provides TypedDict definitions for document schemas,
allowing for structural typing and validation through type checkers.
"""

from typing import Any, Dict, List, Optional, TypedDict, Union
from datetime import datetime


class EntitySchema(TypedDict, total=False):
    """Schema for document entities using TypedDict."""
    
    entity_type: str
    """Type of the entity (e.g., "person", "organization")."""
    
    text: str
    """The text content of the entity."""
    
    start_pos: Optional[int]
    """Starting position in the document text."""
    
    end_pos: Optional[int]
    """Ending position in the document text."""
    
    confidence: Optional[float]
    """Confidence score for the entity extraction."""
    
    metadata: Dict[str, Any]
    """Additional metadata for the entity."""


class MetadataSchema(TypedDict, total=False):
    """Schema for document metadata using TypedDict."""
    
    source_path: Optional[str]
    """Path to the source document."""
    
    title: Optional[str]
    """Document title."""
    
    author: Optional[Union[str, List[str]]]
    """Document author(s)."""
    
    created_date: Optional[Union[str, datetime]]
    """Document creation date."""
    
    modified_date: Optional[Union[str, datetime]]
    """Document last modification date."""
    
    language: Optional[str]
    """Document language."""
    
    content_type: Optional[str]
    """Document content type."""
    
    tags: List[str]
    """List of tags for the document."""
    
    custom_metadata: Dict[str, Any]
    """Additional custom metadata."""


class ChunkSchema(TypedDict, total=False):
    """Schema for document chunks using TypedDict."""
    
    chunk_id: str
    """Unique identifier for the chunk."""
    
    document_id: str
    """ID of the parent document."""
    
    content: str
    """Text content of the chunk."""
    
    start_pos: Optional[int]
    """Starting position in the original document."""
    
    end_pos: Optional[int]
    """Ending position in the original document."""
    
    metadata: Dict[str, Any]
    """Metadata specific to this chunk."""
    
    embedding: Optional[List[float]]
    """Vector embedding for the chunk content."""
    
    embedding_model: Optional[str]
    """Name of the model used to generate the embedding."""
    
    isne_embedding: Optional[List[float]]
    """ISNE-enhanced vector embedding."""
    
    section: Optional[str]
    """Section or heading this chunk belongs to."""
    
    sequence: int
    """Sequence number of this chunk in the document."""


class DocumentSchema(TypedDict, total=False):
    """Schema for complete documents using TypedDict."""
    
    document_id: str
    """Unique identifier for the document."""
    
    content: str
    """Document content."""
    
    metadata: MetadataSchema
    """Document metadata."""
    
    entities: List[EntitySchema]
    """Entities extracted from the document."""
    
    chunks: List[ChunkSchema]
    """Document chunks after chunking."""
    
    errors: List[Dict[str, Any]]
    """Processing errors."""
    
    processing_time: Optional[float]
    """Time taken to process the document (seconds)."""
    
    processed_at: Optional[Union[str, datetime]]
    """Timestamp when the document was processed."""
