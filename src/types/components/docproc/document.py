"""
Core document type definitions for document processing.

This module provides the central type definitions for standardized document
representations used throughout the document processing pipeline. It includes
both TypedDict definitions for runtime type safety and Pydantic models for
validation and serialization.

This module consolidates and replaces the previous src.types.documents module,
providing a unified type system for all document-related operations.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, TypedDict, Union, Literal

import pydantic
import uuid
from pydantic import BaseModel, Field, validator, field_validator, model_validator

from src.types.common import DocumentID, NodeID, BaseSchema, DocumentType, SchemaVersion, EmbeddingVector, MetadataDict, UUIDString
from src.types.components.docproc.enums import ContentCategory


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


# Chunk-related types consolidated from documents module

class ChunkSchema(TypedDict, total=False):
    """Schema for document chunks using TypedDict."""
    
    chunk_id: str  # Unique identifier for the chunk
    document_id: str  # ID of the parent document
    content: str  # Text content of the chunk
    start_pos: Optional[int]  # Starting position in the original document
    end_pos: Optional[int]  # Ending position in the original document
    metadata: Dict[str, Any]  # Metadata specific to this chunk
    embedding: Optional[List[float]]  # Vector embedding for the chunk content
    embedding_model: Optional[str]  # Name of the model used to generate the embedding
    isne_embedding: Optional[List[float]]  # ISNE-enhanced vector embedding
    section: Optional[str]  # Section or heading this chunk belongs to
    sequence: int  # Sequence number of this chunk in the document


class PydanticChunkSchema(BaseModel):
    """Pydantic model for chunk validation."""
    
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")
    start_pos: Optional[int] = Field(None, description="Start position in document")
    end_pos: Optional[int] = Field(None, description="End position in document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    embedding_model: Optional[str] = Field(None, description="Embedding model name")
    isne_embedding: Optional[List[float]] = Field(None, description="ISNE embedding")
    section: Optional[str] = Field(None, description="Section name")
    sequence: int = Field(..., description="Chunk sequence number")
    
    class Config:
        extra = "allow"


# Base document classes for compatibility

class BaseEntity:
    """Base class for entities extracted from documents.
    
    Provides compatibility with existing code that expects class-based entities.
    """
    
    def __init__(
        self, 
        entity_type: str,
        text: str,
        start_pos: Optional[int] = None,
        end_pos: Optional[int] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize a base entity.
        
        Args:
            entity_type: Type of the entity (e.g., "person", "organization")
            text: The text content of the entity
            start_pos: Starting position in the document text
            end_pos: Ending position in the document text
            confidence: Confidence score for the entity extraction
            metadata: Additional metadata for the entity
        """
        self.entity_type = entity_type
        self.text = text
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.confidence = confidence
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            "entity_type": self.entity_type,
            "text": self.text,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": self.confidence,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseEntity":
        """Create entity from dictionary representation."""
        return cls(
            entity_type=data["entity_type"],
            text=data["text"],
            start_pos=data.get("start_pos"),
            end_pos=data.get("end_pos"),
            confidence=data.get("confidence"),
            metadata=data.get("metadata", {})
        )


class BaseMetadata:
    """Base class for document metadata.
    
    Provides compatibility with existing code that expects class-based metadata.
    """
    
    def __init__(
        self,
        source_path: Optional[str] = None,
        title: Optional[str] = None,
        author: Optional[Union[str, List[str]]] = None,
        created_date: Optional[Union[str, datetime]] = None,
        modified_date: Optional[Union[str, datetime]] = None,
        language: Optional[str] = None,
        content_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize document metadata.
        
        Args:
            source_path: Path to the source document
            title: Document title
            author: Document author(s)
            created_date: Document creation date
            modified_date: Document last modification date
            language: Document language
            content_type: Document content type
            tags: List of tags for the document
            custom_metadata: Additional custom metadata
        """
        self.source_path = source_path
        self.title = title
        self.author = author
        self.created_date = created_date
        self.modified_date = modified_date
        self.language = language
        self.content_type = content_type
        self.tags = tags or []
        self.custom_metadata = custom_metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary representation."""
        return {
            "source_path": self.source_path,
            "title": self.title,
            "author": self.author,
            "created_date": self.created_date.isoformat() if isinstance(self.created_date, datetime) else self.created_date,
            "modified_date": self.modified_date.isoformat() if isinstance(self.modified_date, datetime) else self.modified_date,
            "language": self.language,
            "content_type": self.content_type,
            "tags": self.tags,
            "custom_metadata": self.custom_metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseMetadata":
        """Create metadata from dictionary representation."""
        return cls(
            source_path=data.get("source_path"),
            title=data.get("title"),
            author=data.get("author"),
            created_date=data.get("created_date"),
            modified_date=data.get("modified_date"),
            language=data.get("language"),
            content_type=data.get("content_type"),
            tags=data.get("tags"),
            custom_metadata=data.get("custom_metadata", {})
        )


class BaseDocument:
    """Base class for documents.
    
    Provides compatibility with existing code that expects class-based documents.
    """
    
    def __init__(
        self,
        document_id: str,
        content: str,
        metadata: Optional[Union[BaseMetadata, Dict[str, Any]]] = None,
        entities: Optional[List[Union[BaseEntity, Dict[str, Any]]]] = None,
        chunks: Optional[List[Dict[str, Any]]] = None,
        errors: Optional[List[Dict[str, Any]]] = None
    ):
        """Initialize a base document.
        
        Args:
            document_id: Unique identifier for the document
            content: Document content
            metadata: Document metadata
            entities: Entities extracted from the document
            chunks: Document chunks after chunking
            errors: Processing errors
        """
        self.document_id = document_id
        self.content = content
        
        # Convert metadata dictionary to BaseMetadata if needed
        if isinstance(metadata, dict):
            self.metadata = BaseMetadata.from_dict(metadata)
        else:
            self.metadata = metadata or BaseMetadata()
        
        # Process entities
        self.entities = []
        if entities:
            for entity in entities:
                if isinstance(entity, dict):
                    self.entities.append(BaseEntity.from_dict(entity))
                else:
                    self.entities.append(entity)
        
        self.chunks = chunks or []
        self.errors = errors or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary representation."""
        return {
            "document_id": self.document_id,
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "entities": [entity.to_dict() for entity in self.entities],
            "chunks": self.chunks,
            "errors": self.errors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseDocument":
        """Create document from dictionary representation."""
        return cls(
            document_id=data["document_id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            entities=data.get("entities", []),
            chunks=data.get("chunks", []),
            errors=data.get("errors", [])
        )


# Enhanced DocumentSchema that includes the best from both type systems

class DocumentSchema(TypedDict, total=False):
    """Complete document schema combining TypedDict and legacy schemas."""
    
    # Core document fields (from ProcessedDocument)
    id: str  # Document ID
    document_id: str  # Alternative document ID field for compatibility
    content: str  # Document content
    content_type: str  # MIME type
    format: str  # Detected format
    content_category: ContentCategory  # Text, code, etc.
    raw_content: Optional[str]  # Original unprocessed content
    
    # Metadata and entities
    metadata: DocumentMetadata  # Document metadata
    entities: List[DocumentEntity]  # Extracted entities
    sections: Optional[List[DocumentSection]]  # Document sections
    
    # Chunking support
    chunks: List[ChunkSchema]  # Document chunks after chunking
    
    # Processing information
    error: Optional[str]  # Processing error, if any
    errors: List[Dict[str, Any]]  # Processing errors
    processing_time: Optional[float]  # Time taken to process (seconds)
    processed_at: Optional[Union[str, datetime]]  # Processing timestamp


# TypedDict aliases for compatibility
EntitySchema = DocumentEntity  # Alias for backward compatibility
MetadataSchema = DocumentMetadata  # Alias for backward compatibility


# Helper types for document processing
DocumentSource = Union[str, Path]
DocumentFilter = Dict[str, Any]
DocumentSortOptions = Dict[str, Union[str, int, bool]]
ProcessingMode = Literal["normal", "strict", "lenient", "draft"]


# ================================
# CONSOLIDATED DOCUMENT TYPES FROM src/types/documents/
# ================================

class ConsolidatedChunkMetadata(BaseSchema):
    """Metadata for document chunks (consolidated from src.types.documents.base)."""
    
    start_offset: int = Field(..., description="Start position in the original document")
    end_offset: int = Field(..., description="End position in the original document")
    chunk_type: str = Field(default="text", description="Type of the chunk (text, code, etc.)")
    chunk_index: int = Field(..., description="Sequential index of the chunk")
    parent_id: str = Field(..., description="ID of the parent document")
    context_before: Optional[str] = Field(default=None, description="Text context before the chunk")
    context_after: Optional[str] = Field(default=None, description="Text context after the chunk")
    metadata: MetadataDict = Field(default_factory=lambda: {}, description="Additional chunk-specific metadata")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "start_offset": 0,
                    "end_offset": 100,
                    "chunk_type": "text",
                    "chunk_index": 0,
                    "parent_id": "doc123",
                    "metadata": {
                        "importance": "high"
                    }
                }
            ]
        }
    }


class ConsolidatedDocumentSchema(BaseSchema):
    """Pydantic schema for document validation (consolidated from src.types.documents.base).
    
    This model enforces structure and type safety for documents in the HADES system.
    """
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the document")
    content: str = Field(..., description="Document content text")
    source: str = Field(..., description="Origin of the document (filename, URL, etc.)")
    document_type: DocumentType = Field(..., description="Type of document")
    schema_version: SchemaVersion = Field(default=SchemaVersion.V2, description="Schema version for compatibility")
    title: Optional[str] = Field(default=None, description="Document title")
    author: Optional[str] = Field(default=None, description="Document author")
    created_at: Optional[datetime] = Field(default=None, description="Document creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Document last update timestamp")
    metadata: MetadataDict = Field(default_factory=lambda: {}, description="Additional document metadata")
    embedding: Optional[EmbeddingVector] = Field(default=None, description="Document embedding vector")
    embedding_model: Optional[str] = Field(default=None, description="Model used to generate the embedding")
    chunks: List[ConsolidatedChunkMetadata] = Field(default_factory=list, description="Document chunks metadata")
    tags: List[str] = Field(default_factory=list, description="Document tags for categorization")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "doc123",
                    "content": "Example document content",
                    "source": "example.txt",
                    "document_type": "text",
                    "title": "Example Document"
                }
            ]
        }
    }
    
    @field_validator("document_type")
    @classmethod
    def validate_document_type(cls, v: Any) -> DocumentType:
        """Validate document type."""
        if isinstance(v, str):
            try:
                return DocumentType(v)
            except ValueError:
                raise ValueError(f"Invalid document type: {v}")
        elif isinstance(v, DocumentType):
            return v
        else:
            raise ValueError(f"Invalid document type: {v}")
    
    @field_validator("id")
    @classmethod
    def validate_id(cls, v: Optional[str]) -> str:
        """Validate ID is not empty."""
        if not v:
            return str(uuid.uuid4())
        return v
    
    @model_validator(mode="after")
    def ensure_timestamps_and_title(self) -> "ConsolidatedDocumentSchema":
        """Ensure timestamps are present and derive title from source if not provided."""
        # Set creation time if not provided
        if self.created_at is None:
            self.created_at = datetime.now()
        
        # Set update time to creation time if not provided
        if self.updated_at is None:
            self.updated_at = self.created_at
            
        # Derive title from source if not provided
        if self.title is None:
            import os
            self.title = os.path.basename(self.source)
            
        return self
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the document
        """
        return self.model_dump_safe()


# Aliases for backward compatibility with src/types/documents/
ChunkMetadata = ConsolidatedChunkMetadata  # Main alias for system-wide use
DocumentSchema = ConsolidatedDocumentSchema  # Main alias for system-wide use
