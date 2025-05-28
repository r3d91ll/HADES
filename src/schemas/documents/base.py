"""
Base document schemas for HADES-PathRAG.

This module provides the core document models that form the foundation
of document processing in the system.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union, cast
import uuid
import os

from pydantic import Field, model_validator, field_validator, validator

from ..common.base import BaseSchema
from ..common.enums import DocumentType, SchemaVersion
from ..common.types import EmbeddingVector, MetadataDict, UUIDString
from ..common.validation import typed_field_validator, typed_model_validator


class ChunkMetadata(BaseSchema):
    """Metadata for document chunks."""
    
    start_offset: int = Field(..., description="Start position in the original document")
    end_offset: int = Field(..., description="End position in the original document")
    chunk_type: str = Field(default="text", description="Type of the chunk (text, code, etc.)")
    chunk_index: int = Field(..., description="Sequential index of the chunk")
    parent_id: str = Field(..., description="ID of the parent document")
    context_before: Optional[str] = Field(default=None, description="Text context before the chunk")
    context_after: Optional[str] = Field(default=None, description="Text context after the chunk")
    metadata: MetadataDict = Field(default_factory=dict, description="Additional chunk-specific metadata")
    
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


class DocumentSchema(BaseSchema):
    """Pydantic schema for document validation and standardization.
    
    This model enforces structure and type safety for documents in the HADES-PathRAG system.
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
    metadata: MetadataDict = Field(default_factory=dict, description="Additional document metadata")
    embedding: Optional[EmbeddingVector] = Field(default=None, description="Document embedding vector")
    embedding_model: Optional[str] = Field(default=None, description="Model used to generate the embedding")
    chunks: List[ChunkMetadata] = Field(default_factory=list, description="Document chunks metadata")
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
    
    @typed_field_validator("document_type")
    @classmethod
    def validate_document_type(cls, v: Union[str, DocumentType]) -> DocumentType:
        """Validate document type."""
        if isinstance(v, DocumentType):
            return v
        else:
            try:
                return DocumentType(v)
            except ValueError:
                raise ValueError(f"Invalid document type: {v}")
    
    @typed_field_validator("id")
    @classmethod
    def validate_id(cls, v: Optional[str]) -> str:
        """Validate ID is not empty."""
        if not v:
            return str(uuid.uuid4())
        return v
    
    @typed_model_validator(mode='before')
    @classmethod
    def ensure_timestamps_and_title(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure timestamps are present and derive title from source if not provided."""
        # Set creation time if not provided
        if values.get('created_at') is None:
            values['created_at'] = datetime.now()
        
        # Set updated time if not provided
        created_at = values.get('created_at')
        updated_at = values.get('updated_at')
        if updated_at is None or (created_at is not None and updated_at is not None and updated_at < created_at):
            values['updated_at'] = created_at
        
        # Generate title from source path if not set
        title = values.get('title')
        source = values.get('source')
        if title is None and source is not None and isinstance(source, str):
            values['title'] = os.path.basename(source)
        
        return values
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the document
        """
        return self.model_dump_safe()
        
    @classmethod
    def from_ingest_document(cls, doc: Any) -> DocumentSchema:
        """Convert an existing IngestDocument to a DocumentSchema.
        
        Args:
            doc: An IngestDocument instance
            
        Returns:
            DocumentSchema: A validated document schema instance
        """
        # Handle conversion from old types
        data = doc.to_dict() if hasattr(doc, 'to_dict') else doc.copy() if isinstance(doc, dict) else doc
        
        # Ensure document_type is present
        if 'document_type' not in data and 'type' in data:
            data['document_type'] = data['type']
            
        # Create and validate new schema instance
        return cls(**data)
