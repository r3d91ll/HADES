"""
Chunk type definitions for the HADES-PathRAG chunking system.

This module provides the centralized type definitions for chunks, including
the base chunk type, chunk metadata, and related type definitions.
"""

from typing import Dict, List, Any, Optional, Union, TypedDict, Literal
from pydantic import BaseModel, Field
import uuid
from datetime import datetime


class ChunkMetadata(TypedDict, total=False):
    """Metadata for a chunk."""
    # Identification
    id: str  # Unique identifier for the chunk
    parent_id: Optional[str]  # ID of the parent document
    
    # Source information
    path: str  # Path to the source document
    document_type: str  # Type of document (e.g., "python", "text")
    source_file: str  # Original source file
    
    # Position information
    line_start: int  # Starting line number in source document (1-indexed)
    line_end: int  # Ending line number in source document (1-indexed)
    char_start: Optional[int]  # Starting character position
    char_end: Optional[int]  # Ending character position
    
    # Content information
    token_count: int  # Number of tokens in the chunk
    chunk_type: str  # Type of chunk (e.g., "code", "text", "class", "function")
    language: Optional[str]  # Language of the code, if applicable
    
    # Symbol information for code
    symbol_type: Optional[str]  # Type of symbol (e.g., "function", "class", "method")
    name: Optional[str]  # Name of the symbol
    parent: Optional[str]  # Parent symbol (e.g., class name for a method)
    
    # Relationships
    next_chunk_id: Optional[str]  # ID of the next chunk in sequence
    prev_chunk_id: Optional[str]  # ID of the previous chunk in sequence
    related_chunks: Optional[List[str]]  # IDs of related chunks
    
    # Processing information
    created_at: str  # ISO format datetime when the chunk was created
    processing_time: Optional[float]  # Time taken to process the chunk
    
    # Custom metadata
    custom: Optional[Dict[str, Any]]  # Custom metadata fields


class Chunk(TypedDict):
    """A chunk of content with metadata."""
    id: str  # Unique identifier for the chunk
    content: str  # The content of the chunk
    metadata: ChunkMetadata  # Metadata for the chunk


class ChunkModel(BaseModel):
    """Pydantic model for a chunk."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Configuration for the ChunkModel."""
        extra = "allow"


class TextChunk(Chunk):
    """A chunk of text content."""
    content: str  # Text content
    metadata: ChunkMetadata  # Includes text-specific metadata


class CodeChunk(Chunk):
    """A chunk of code content."""
    content: str  # Code content
    metadata: ChunkMetadata  # Includes code-specific metadata
    

# Type aliases for clarity
ChunkDict = Dict[str, Any]
ChunkList = List[ChunkDict]
ChunkContent = str
ChunkId = str

# Output format types
OutputFormatType = Literal["document", "json", "dict", "schema"]

# Type for document that can be chunked
ChunkableDocument = Union[Dict[str, Any], BaseModel, str]


def create_chunk_metadata(
    chunk_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    path: str = "unknown",
    document_type: str = "text",
    source_file: str = "unknown",
    line_start: int = 0,
    line_end: int = 0,
    token_count: int = 0,
    chunk_type: str = "text",
    language: Optional[str] = None,
    symbol_type: Optional[str] = None,
    name: Optional[str] = None,
    parent: Optional[str] = None,
    **kwargs: Any
) -> ChunkMetadata:
    """Create a chunk metadata dictionary with default values.
    
    Args:
        chunk_id: Unique identifier for the chunk
        parent_id: ID of the parent document
        path: Path to the source document
        document_type: Type of document
        source_file: Original source file
        line_start: Starting line number
        line_end: Ending line number
        token_count: Number of tokens in the chunk
        chunk_type: Type of chunk
        language: Language of the code
        symbol_type: Type of symbol
        name: Name of the symbol
        parent: Parent symbol
        **kwargs: Additional metadata fields
    
    Returns:
        A ChunkMetadata dictionary
    """
    metadata: ChunkMetadata = {
        "id": chunk_id or str(uuid.uuid4()),
        "parent_id": parent_id,
        "path": path,
        "document_type": document_type,
        "source_file": source_file,
        "line_start": line_start,
        "line_end": line_end,
        "token_count": token_count,
        "chunk_type": chunk_type,
        "created_at": datetime.now().isoformat()
    }
    
    # Add optional fields if provided
    if language is not None:
        metadata["language"] = language
    if symbol_type is not None:
        metadata["symbol_type"] = symbol_type
    if name is not None:
        metadata["name"] = name
    if parent is not None:
        metadata["parent"] = parent
    
    # Add any additional metadata
    for key, value in kwargs.items():
        metadata[key] = value
    
    return metadata


def create_chunk(
    content: str, 
    metadata: Optional[ChunkMetadata] = None,
    **kwargs: Any
) -> Chunk:
    """Create a chunk with the given content and metadata.
    
    Args:
        content: The content of the chunk
        metadata: Metadata for the chunk
        **kwargs: Additional metadata fields
    
    Returns:
        A Chunk dictionary
    """
    chunk_id = str(uuid.uuid4())
    chunk_metadata = metadata or create_chunk_metadata(chunk_id=chunk_id, **kwargs)
    
    return {
        "id": chunk_id,
        "content": content,
        "metadata": chunk_metadata
    }
