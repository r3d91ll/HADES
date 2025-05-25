"""
ISNE document schemas for HADES-PathRAG.

This module provides Pydantic models for ISNE document representation,
used for graph construction and embedding enhancement.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import uuid

from pydantic import Field, field_validator, model_validator

from ..common.base import BaseSchema
from ..common.types import EmbeddingVector, MetadataDict
from .models import ISNEDocumentType


class ISNEChunkSchema(BaseSchema):
    """Chunk representation within an ISNE document."""
    
    id: str = Field(..., description="Unique identifier for the chunk")
    parent_id: str = Field(..., description="ID of the parent document")
    start_index: int = Field(..., description="Start index in the document")
    end_index: int = Field(..., description="End index in the document")
    content: str = Field(..., description="Chunk content")
    embedding: Optional[EmbeddingVector] = Field(default=None, description="Chunk embedding vector")
    metadata: MetadataDict = Field(default_factory=dict, description="Additional chunk metadata")
    
    @field_validator("id")
    @classmethod
    def validate_id(cls, v: Optional[str]) -> str:
        """Validate ID is not empty."""
        if not v:
            return str(uuid.uuid4())
        return v


class ISNEDocumentSchema(BaseSchema):
    """Document representation for ISNE graph construction."""
    
    id: str = Field(..., description="Unique identifier for the document")
    type: ISNEDocumentType = Field(default=ISNEDocumentType.TEXT, description="Type of document")
    content: Optional[str] = Field(default=None, description="Document content")
    chunks: List[ISNEChunkSchema] = Field(default_factory=list, description="Document chunks")
    metadata: MetadataDict = Field(default_factory=dict, description="Additional document metadata")
    embeddings: Dict[str, List[float]] = Field(default_factory=dict, description="Document embeddings by type")
    embedding_model: str = Field(..., description="Model used to generate embeddings")
    source: str = Field(..., description="Source of the document (e.g., file path, URL)")
    created_at: Optional[str] = Field(default=None, description="Document creation timestamp")
    processed_at: Optional[str] = Field(default=None, description="Document processing timestamp")
    
    @field_validator("id")
    @classmethod
    def validate_id(cls, v: Optional[str]) -> str:
        """Validate ID is not empty."""
        if not v:
            return str(uuid.uuid4())
        return v
    
    @model_validator(mode="after")
    def validate_chunks(self) -> ISNEDocumentSchema:
        """Set parent_id for each chunk if not already set."""
        for chunk in self.chunks:
            if chunk.parent_id != self.id:
                chunk.parent_id = self.id
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary representation of the document
        """
        return self.model_dump_safe()
