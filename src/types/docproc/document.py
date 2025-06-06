"""
Core document type definitions for document processing.

This module provides the central type definitions for standardized document
representations used throughout the document processing pipeline. It includes
both TypedDict definitions for runtime type safety and Pydantic models for
validation and serialization.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, TypedDict, Union, Literal

import pydantic
from pydantic import BaseModel, Field, validator

from src.types.common import DocumentID, NodeID
from src.types.docproc.enums import ContentCategory


class DocumentEntity(TypedDict, total=False):
    """Entity extracted from document content."""
    
    type: str  # Entity type (person, organization, location, etc.)
    text: str  # The entity text
    start: int  # Start character position
    end: int  # End character position
    confidence: float  # Confidence score (0.0-1.0)
    metadata: Dict[str, Any]  # Additional entity metadata


class DocumentMetadata(TypedDict, total=False):
    """Standard document metadata."""
    
    title: str  # Document title
    authors: List[str]  # List of authors
    source: str  # Source system or path
    created_at: str  # ISO format date string
    modified_at: str  # ISO format date string
    file_type: str  # Original file type/extension
    content_type: str  # MIME type
    content_category: ContentCategory  # Text, code, etc.
    language: str  # Document language
    size: int  # Size in bytes
    page_count: int  # Number of pages
    word_count: int  # Word count
    character_count: int  # Character count
    filename: str  # Original filename
    path: str  # Original file path
    keywords: List[str]  # Keywords or tags
    summary: str  # Document summary
    custom: Dict[str, Any]  # Custom metadata fields


class DocumentSection(TypedDict, total=False):
    """A section of a document."""
    
    id: str  # Section ID
    title: Optional[str]  # Section title
    level: int  # Heading level or depth
    content: str  # Section content
    start: int  # Start character position in full content
    end: int  # End character position in full content
    metadata: Dict[str, Any]  # Additional section metadata
    parent_id: Optional[str]  # Parent section ID


class ProcessedDocument(TypedDict):
    """Standard processed document structure."""
    
    id: str  # Document ID
    content: str  # Extracted text content
    content_type: str  # MIME type
    format: str  # Detected format
    content_category: ContentCategory  # Text, code, etc.
    raw_content: Optional[str]  # Original unprocessed content
    metadata: DocumentMetadata  # Document metadata
    entities: List[DocumentEntity]  # Extracted entities
    sections: Optional[List[DocumentSection]]  # Document sections
    error: Optional[str]  # Processing error, if any


class ChunkPreparationMarker(TypedDict):
    """Marker for chunk boundary or section."""
    
    type: str  # Marker type (section, paragraph, code, etc.)
    start: int  # Start character position
    end: int  # End character position
    metadata: Dict[str, Any]  # Additional marker metadata
    weight: float  # Importance weight (0.0-1.0)


class ChunkPreparedDocument(TypedDict):
    """Document prepared for chunking."""
    
    id: str  # Document ID
    content: str  # Content with markers
    markers: List[ChunkPreparationMarker]  # Chunk boundary markers
    document: ProcessedDocument  # Original processed document
    


class BatchProcessingStatistics(TypedDict):
    """Statistics from batch document processing."""
    
    total: int  # Total documents processed
    successful: int  # Successfully processed documents
    failed: int  # Failed documents
    formats: Dict[str, int]  # Count by format
    errors: Dict[str, int]  # Count by error type
    processing_time: Dict[str, float]  # Processing time statistics
    documents: List[str]  # List of processed document IDs


class DocumentProcessingResult(TypedDict):
    """Result of document processing operation."""
    
    success: bool  # Whether processing succeeded
    document: Optional[ProcessedDocument]  # Processed document if successful
    error: Optional[Dict[str, Any]]  # Error details if failed
    processing_time: float  # Processing time in seconds
    warnings: List[str]  # Processing warnings


# Pydantic models for validation

class PydanticDocumentEntity(BaseModel):
    """Pydantic model for document entity validation."""
    
    type: str = Field(..., description="Entity type")
    text: str = Field(..., description="Entity text")
    start: int = Field(..., description="Start position")
    end: int = Field(..., description="End position")
    confidence: Optional[float] = Field(None, description="Confidence score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        extra = "allow"


class PydanticDocumentMetadata(BaseModel):
    """Pydantic model for document metadata validation."""
    
    title: Optional[str] = Field(None, description="Document title")
    authors: List[str] = Field(default_factory=list, description="Author list")
    source: Optional[str] = Field(None, description="Source system or path")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    modified_at: Optional[str] = Field(None, description="Modification timestamp")
    file_type: Optional[str] = Field(None, description="File type/extension")
    content_type: Optional[str] = Field(None, description="MIME type")
    content_category: Optional[str] = Field(None, description="Content category")
    language: Optional[str] = Field(None, description="Document language")
    size: Optional[int] = Field(None, description="Size in bytes")
    page_count: Optional[int] = Field(None, description="Page count")
    word_count: Optional[int] = Field(None, description="Word count")
    character_count: Optional[int] = Field(None, description="Character count")
    filename: Optional[str] = Field(None, description="Original filename")
    path: Optional[str] = Field(None, description="Original file path")
    keywords: List[str] = Field(default_factory=list, description="Keywords/tags")
    summary: Optional[str] = Field(None, description="Document summary")
    custom: Dict[str, Any] = Field(default_factory=dict, description="Custom fields")
    
    class Config:
        extra = "allow"


# Factory function for metadata default
def _create_default_metadata() -> PydanticDocumentMetadata:
    """Create a default PydanticDocumentMetadata instance."""
    # Explicitly provide all fields to avoid mypy issues
    return PydanticDocumentMetadata(
        title=None,
        authors=[],
        source=None,
        created_at=None,
        modified_at=None,
        file_type=None,
        content_type=None,
        content_category=None,
        language=None,
        size=None,
        page_count=None,
        word_count=None,
        character_count=None,
        filename=None,
        path=None,
        keywords=[],
        summary=None,
        custom={}
    )


class PydanticProcessedDocument(BaseModel):
    """Pydantic model for processed document validation."""
    
    id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Extracted text content")
    content_type: str = Field(..., description="MIME type")
    format: str = Field(..., description="Detected format")
    content_category: str = Field(..., description="Content category")
    raw_content: Optional[str] = Field(None, description="Original content")
    metadata: PydanticDocumentMetadata = Field(default_factory=_create_default_metadata)
    entities: List[PydanticDocumentEntity] = Field(default_factory=list)
    error: Optional[str] = Field(None, description="Processing error")
    
    class Config:
        extra = "allow"


# Helper types for document processing
DocumentSource = Union[str, Path]
DocumentFilter = Dict[str, Any]
DocumentSortOptions = Dict[str, Union[str, int, bool]]
ProcessingMode = Literal["normal", "strict", "lenient", "draft"]
