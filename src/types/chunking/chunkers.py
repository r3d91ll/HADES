"""
Type definitions for specialized chunkers.

This module provides configuration and option types for different
chunker implementations (Python, JSON, YAML, text, etc.).
"""

from typing import Dict, List, Any, Optional, Union, TypedDict, Literal
from pydantic import BaseModel, Field
from enum import Enum

from src.types.chunking.base import ChunkingMode, ChunkBoundary


class PythonChunkerMode(str, Enum):
    """Modes for Python code chunking."""
    
    FUNCTION = "function"  # Chunk by function boundaries
    CLASS = "class"  # Chunk by class boundaries  
    MODULE = "module"  # Chunk by module sections
    MIXED = "mixed"  # Use multiple strategies


class JSONChunkerMode(str, Enum):
    """Modes for JSON chunking."""
    
    OBJECT = "object"  # Chunk by JSON objects
    ARRAY = "array"  # Chunk by array elements
    KEY_VALUE = "key_value"  # Chunk by key-value pairs
    NESTED = "nested"  # Respect nesting structure


class YAMLChunkerMode(str, Enum):
    """Modes for YAML chunking."""
    
    DOCUMENT = "document"  # Chunk by YAML documents
    SECTION = "section"  # Chunk by sections
    LIST = "list"  # Chunk by list items
    MAPPING = "mapping"  # Chunk by mapping entries


class PythonChunkerConfig(BaseModel):
    """Configuration for Python code chunkers."""
    
    mode: PythonChunkerMode = Field(default=PythonChunkerMode.FUNCTION, description="Chunking mode")
    max_tokens: int = Field(default=512, description="Maximum tokens per chunk")
    preserve_imports: bool = Field(default=True, description="Whether to preserve import statements")
    preserve_docstrings: bool = Field(default=True, description="Whether to preserve docstrings")
    include_decorators: bool = Field(default=True, description="Whether to include decorators")
    split_large_functions: bool = Field(default=True, description="Whether to split large functions")
    min_chunk_size: int = Field(default=50, description="Minimum chunk size in tokens")
    overlap_lines: int = Field(default=2, description="Number of overlapping lines")
    respect_indentation: bool = Field(default=True, description="Whether to respect Python indentation")
    
    class Config:
        extra = "allow"


class JSONChunkerConfig(BaseModel):
    """Configuration for JSON chunkers."""
    
    mode: JSONChunkerMode = Field(default=JSONChunkerMode.OBJECT, description="Chunking mode")
    max_tokens: int = Field(default=512, description="Maximum tokens per chunk")
    max_depth: Optional[int] = Field(None, description="Maximum nesting depth to preserve")
    preserve_structure: bool = Field(default=True, description="Whether to preserve JSON structure")
    pretty_print: bool = Field(default=True, description="Whether to pretty print JSON")
    validate_json: bool = Field(default=True, description="Whether to validate JSON syntax")
    
    class Config:
        extra = "allow"


class YAMLChunkerConfig(BaseModel):
    """Configuration for YAML chunkers."""
    
    mode: YAMLChunkerMode = Field(default=YAMLChunkerMode.DOCUMENT, description="Chunking mode")
    max_tokens: int = Field(default=512, description="Maximum tokens per chunk")
    preserve_comments: bool = Field(default=True, description="Whether to preserve comments")
    preserve_anchors: bool = Field(default=True, description="Whether to preserve YAML anchors")
    validate_yaml: bool = Field(default=True, description="Whether to validate YAML syntax")
    maintain_indentation: bool = Field(default=True, description="Whether to maintain indentation")
    
    class Config:
        extra = "allow"


class TextChunkerConfig(BaseModel):
    """Configuration for text chunkers."""
    
    mode: ChunkingMode = Field(default=ChunkingMode.SEMANTIC, description="Chunking mode")
    max_tokens: int = Field(default=512, description="Maximum tokens per chunk")
    overlap_tokens: int = Field(default=50, description="Number of overlapping tokens")
    boundary_strategy: ChunkBoundary = Field(default=ChunkBoundary.SENTENCE, description="Boundary strategy")
    preserve_paragraphs: bool = Field(default=True, description="Whether to preserve paragraph boundaries")
    preserve_sentences: bool = Field(default=True, description="Whether to preserve sentence boundaries")
    min_chunk_size: int = Field(default=100, description="Minimum chunk size in characters")
    language: Optional[str] = Field(None, description="Document language for language-specific processing")
    use_semantic_splitting: bool = Field(default=False, description="Whether to use semantic similarity for splitting")
    
    class Config:
        extra = "allow"


# TypedDict versions for backward compatibility

class PythonChunkerOptions(TypedDict, total=False):
    """Options for Python chunkers."""
    
    mode: str  # Chunking mode
    max_tokens: int  # Maximum tokens per chunk
    preserve_imports: bool  # Preserve imports
    preserve_docstrings: bool  # Preserve docstrings
    include_decorators: bool  # Include decorators
    split_large_functions: bool  # Split large functions
    min_chunk_size: int  # Minimum chunk size
    overlap_lines: int  # Overlapping lines
    respect_indentation: bool  # Respect indentation


class JSONChunkerOptions(TypedDict, total=False):
    """Options for JSON chunkers."""
    
    mode: str  # Chunking mode
    max_tokens: int  # Maximum tokens per chunk
    max_depth: Optional[int]  # Maximum nesting depth
    preserve_structure: bool  # Preserve structure
    pretty_print: bool  # Pretty print JSON
    validate_json: bool  # Validate JSON


class YAMLChunkerOptions(TypedDict, total=False):
    """Options for YAML chunkers."""
    
    mode: str  # Chunking mode
    max_tokens: int  # Maximum tokens per chunk
    preserve_comments: bool  # Preserve comments
    preserve_anchors: bool  # Preserve anchors
    validate_yaml: bool  # Validate YAML
    maintain_indentation: bool  # Maintain indentation


class TextChunkerOptions(TypedDict, total=False):
    """Options for text chunkers."""
    
    mode: str  # Chunking mode
    max_tokens: int  # Maximum tokens per chunk
    overlap_tokens: int  # Overlapping tokens
    boundary_strategy: str  # Boundary strategy
    preserve_paragraphs: bool  # Preserve paragraphs
    preserve_sentences: bool  # Preserve sentences
    min_chunk_size: int  # Minimum chunk size
    language: Optional[str]  # Document language
    use_semantic_splitting: bool  # Use semantic splitting


# Union type for all chunker options
ChunkerOptions = Union[
    PythonChunkerConfig,
    JSONChunkerConfig, 
    YAMLChunkerConfig,
    TextChunkerConfig,
    Dict[str, Any]
]


class ChunkerCapabilities(TypedDict):
    """Capabilities of a chunker."""
    
    supported_formats: List[str]  # Supported file formats
    supports_semantic_chunking: bool  # Supports semantic chunking
    supports_overlap: bool  # Supports chunk overlap
    supports_structure_preservation: bool  # Preserves document structure
    max_file_size: Optional[int]  # Maximum file size (bytes)
    preferred_token_range: List[int]  # [min_tokens, max_tokens]