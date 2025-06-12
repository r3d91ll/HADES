"""
Document type definitions for chunking.

This module provides type definitions for documents that are processed
by chunkers, including base document models and document schemas.
"""

from typing import Dict, List, Any, Optional, Union, TypedDict
from pydantic import BaseModel, Field

# Type aliases from AST chunker
ChunkInfo = Dict[str, Any]
SymbolInfo = Dict[str, Any]


class BaseDocument(BaseModel):
    """Base document class for document processing.
    
    This is the consolidated document model that replaces the version
    previously defined in chonky_chunker.py.
    """
    
    content: str = Field(..., description="Document content")
    path: str = Field(..., description="Document path")
    type: str = Field(default="text", description="Document type")
    id: str = Field(default="", description="Document ID")
    chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Document chunks")
    
    class Config:
        extra = "allow"


class DocumentSchemaBase(BaseModel):
    """Base document schema for document processing.
    
    This provides the base schema that can be extended for specific
    document types and validation requirements.
    """
    
    content: str = Field(..., description="Document content")
    path: str = Field(..., description="Document path") 
    type: str = Field(default="text", description="Document type")
    id: str = Field(default="", description="Document ID")
    chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Document chunks")
    
    class Config:
        extra = "allow"


class DocumentTypeSchema(DocumentSchemaBase):
    """Complete document type schema for document processing.
    
    This is the main document type schema used throughout the chunking system
    for compile-time type checking. It extends the base schema with additional fields as needed.
    """
    
    # Additional fields can be added here as the schema evolves
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    format: Optional[str] = Field(None, description="Document format")
    language: Optional[str] = Field(None, description="Document language")
    
    class Config:
        extra = "allow"


# Type unions for backward compatibility
DocumentBaseType = Union[Dict[str, Any], BaseDocument]
DocumentSchemaType = Union[Dict[str, Any], BaseDocument, DocumentTypeSchema]


class DocumentChunkInfo(TypedDict, total=False):
    """Information about a document chunk."""
    
    chunk_id: str  # Unique identifier for the chunk
    document_id: str  # ID of the parent document
    content: str  # Chunk content
    start_position: int  # Start position in original document
    end_position: int  # End position in original document
    token_count: int  # Estimated token count
    chunk_type: str  # Type of chunk (text, code, etc.)
    metadata: Dict[str, Any]  # Additional chunk metadata


class ChunkingResult(TypedDict):
    """Result of a chunking operation."""
    
    document_id: str  # ID of the document that was chunked
    chunks: List[DocumentChunkInfo]  # List of chunks created
    total_chunks: int  # Total number of chunks
    total_tokens: int  # Total estimated tokens across all chunks
    chunking_time: float  # Time taken to chunk the document
    chunker_name: str  # Name of chunker used
    chunker_config: Dict[str, Any]  # Configuration used for chunking