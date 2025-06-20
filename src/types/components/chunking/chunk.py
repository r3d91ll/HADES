"""
Chunk type definitions for the chunking system.

This module provides type definitions for individual chunks and chunk metadata,
including specialized chunk types for different content types.
"""

from typing import Dict, List, Any, Optional, Union, TypedDict, Literal
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class ChunkType(str, Enum):
    """Types of chunks that can be created."""
    
    TEXT = "text"  # Plain text chunk
    CODE = "code"  # Code chunk
    HEADING = "heading"  # Document heading
    PARAGRAPH = "paragraph"  # Text paragraph
    LIST_ITEM = "list_item"  # List item
    TABLE = "table"  # Table data
    METADATA = "metadata"  # Metadata chunk


class ChunkingStrategy(str, Enum):
    """Strategies for chunking content."""
    
    FIXED_SIZE = "fixed_size"  # Fixed token/character size
    SEMANTIC = "semantic"  # Semantic boundary detection
    STRUCTURE = "structure"  # Document structure aware
    ADAPTIVE = "adaptive"  # Adaptive sizing based on content


class ChunkMetadata(BaseModel):
    """Metadata for document chunks.
    
    This consolidates the ChunkMetadata that was previously imported
    from src.schemas.documents.base.
    """
    
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    sequence: int = Field(..., description="Chunk sequence number in document")
    start_position: int = Field(..., description="Start position in original content")
    end_position: int = Field(..., description="End position in original content")
    token_count: Optional[int] = Field(None, description="Estimated token count")
    character_count: int = Field(..., description="Character count")
    chunk_type: ChunkType = Field(default=ChunkType.TEXT, description="Type of chunk")
    strategy: ChunkingStrategy = Field(default=ChunkingStrategy.FIXED_SIZE, description="Chunking strategy used")
    overlap_with_previous: Optional[int] = Field(None, description="Characters overlapping with previous chunk")
    overlap_with_next: Optional[int] = Field(None, description="Characters overlapping with next chunk")
    language: Optional[str] = Field(None, description="Detected language")
    quality_score: Optional[float] = Field(None, description="Quality score (0.0-1.0)")
    created_at: datetime = Field(default_factory=datetime.now, description="Chunk creation timestamp")
    custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata fields")
    
    class Config:
        extra = "allow"


class TextChunk(BaseModel):
    """A text chunk with content and metadata."""
    
    content: str = Field(..., description="Chunk text content")
    metadata: ChunkMetadata = Field(..., description="Chunk metadata")
    
    class Config:
        extra = "allow"


class CodeChunk(BaseModel):
    """A code chunk with additional code-specific metadata."""
    
    content: str = Field(..., description="Chunk code content")
    metadata: ChunkMetadata = Field(..., description="Chunk metadata")
    language: str = Field(..., description="Programming language")
    syntax_valid: bool = Field(default=True, description="Whether the chunk has valid syntax")
    function_names: List[str] = Field(default_factory=list, description="Function names in chunk")
    class_names: List[str] = Field(default_factory=list, description="Class names in chunk")
    imports: List[str] = Field(default_factory=list, description="Import statements in chunk")
    complexity_score: Optional[float] = Field(None, description="Code complexity score")
    
    class Config:
        extra = "allow"


# TypedDict versions for flexibility

class TextChunkDict(TypedDict, total=False):
    """TypedDict version of text chunk."""
    
    content: str  # Chunk content
    metadata: Dict[str, Any]  # Chunk metadata
    chunk_id: str  # Unique identifier
    document_id: str  # Parent document ID
    sequence: int  # Sequence number
    start_position: int  # Start position
    end_position: int  # End position
    token_count: Optional[int]  # Token count
    chunk_type: str  # Chunk type


class CodeChunkDict(TypedDict, total=False):
    """TypedDict version of code chunk."""
    
    content: str  # Chunk content
    metadata: Dict[str, Any]  # Chunk metadata
    chunk_id: str  # Unique identifier
    document_id: str  # Parent document ID
    sequence: int  # Sequence number
    start_position: int  # Start position
    end_position: int  # End position
    token_count: Optional[int]  # Token count
    chunk_type: str  # Chunk type
    language: str  # Programming language
    syntax_valid: bool  # Syntax validity
    function_names: List[str]  # Function names
    class_names: List[str]  # Class names
    imports: List[str]  # Import statements


# Union types for flexibility
AnyChunk = Union[TextChunk, CodeChunk]
AnyChunkDict = Union[TextChunkDict, CodeChunkDict, Dict[str, Any]]


class ChunkValidationResult(TypedDict):
    """Result of chunk validation."""
    
    is_valid: bool  # Whether the chunk is valid
    errors: List[str]  # Validation errors
    warnings: List[str]  # Validation warnings
    token_count: Optional[int]  # Validated token count
    quality_score: Optional[float]  # Quality assessment score