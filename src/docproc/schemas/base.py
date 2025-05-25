"""
Base schemas for document processing validation.

This module provides base Pydantic models that define the common structure
for all documents processed in the pipeline. These models ensure consistent
data validation across different document types.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# Import from the centralized schema structure
from src.schemas.common.base import BaseSchema
from src.schemas.common.types import MetadataDict


class BaseEntity(BaseSchema):
    """Basic entity extracted from document content."""
    
    type: str = Field(..., description="Type of the entity")
    value: str = Field(..., description="Value or text of the entity")
    line: Optional[int] = Field(None, description="Line number where entity appears")
    confidence: float = Field(1.0, description="Confidence score (0.0-1.0) for this entity")
    
    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {v}")
        return v
    
    model_config = ConfigDict(extra="allow")  # Allow additional fields for adapter-specific entities


class BaseMetadata(BaseSchema):
    """Common document metadata across all document types."""
    
    language: str = Field(..., description="Language of the document (e.g., 'python', 'en')")
    format: str = Field(..., description="Format of the document (e.g., 'python', 'pdf')")
    content_type: str = Field(..., description="Type of the content (e.g., 'code', 'text')")
    file_size: Optional[int] = Field(None, description="Size of the file in bytes")
    line_count: Optional[int] = Field(None, description="Number of lines in the document")
    char_count: Optional[int] = Field(None, description="Number of characters in the document")
    has_errors: bool = Field(False, description="Whether document processing encountered errors")
    
    model_config = ConfigDict(extra="allow")  # Allow format-specific metadata fields


class BaseDocument(BaseSchema):
    """Base document model that all processed documents must conform to."""
    
    id: str = Field(..., min_length=4, description="Unique identifier for the document")
    source: str = Field(..., description="Path or identifier for document source")
    content: str = Field(..., description="Processed document content")
    content_type: str = Field("markdown", description="Format of the content (e.g., 'markdown', 'text')")
    format: str = Field(..., description="Format of the original document (e.g., 'python', 'pdf')")
    raw_content: Optional[str] = Field(None, description="Original unprocessed document content (deprecated)")
    metadata: BaseMetadata = Field(..., description="Document metadata")
    entities: List[BaseEntity] = Field(default_factory=list, description="Entities extracted from the document")
    error: Optional[str] = Field(None, description="Error message if processing failed")
    
    @model_validator(mode='after')
    def validate_error_consistency(self) -> 'BaseDocument':
        """Ensure error state is consistent with metadata."""
        if self.error and not self.metadata.has_errors:
            self.metadata.has_errors = True
        return self
    
    model_config = ConfigDict(extra="allow")  # Allow format-specific fields
