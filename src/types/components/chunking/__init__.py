"""
Type definitions for the chunking module.

This package contains centralized type definitions for the document chunking
components of the HADES system, including chunker protocols, document models,
and chunk schemas.
"""

# Import base chunker types
from .base import (
    ChunkerProtocol,
    ChunkerRegistry,
    ChunkerConfig,
    ChunkingMode,
    ChunkBoundary
)

# Import document types  
from .document import (
    ChunkInfo,
    SymbolInfo,
    DocumentBaseType,
    DocumentSchemaType,
    BaseDocument,
    DocumentTypeSchema,
    DocumentSchemaBase
)

# Import chunk types
from .chunk import (
    ChunkMetadata,
    TextChunk,
    CodeChunk,
    ChunkType,
    ChunkingStrategy
)

# Import specialized chunker types
from .chunkers import (
    PythonChunkerConfig,
    JSONChunkerConfig,
    YAMLChunkerConfig,
    TextChunkerConfig,
    ChunkerOptions,
    PythonChunkerOptions,
    JSONChunkerOptions,
    YAMLChunkerOptions,
    TextChunkerOptions,
    ChunkerCapabilities
)

__all__ = [
    # Base chunker types
    "ChunkerProtocol",
    "ChunkerRegistry", 
    "ChunkerConfig",
    "ChunkingMode",
    "ChunkBoundary",
    
    # Document types
    "ChunkInfo",
    "SymbolInfo", 
    "DocumentBaseType",
    "DocumentSchemaType",
    "BaseDocument",
    "DocumentTypeSchema",
    "DocumentSchemaBase",
    
    # Chunk types
    "ChunkMetadata",
    "TextChunk",
    "CodeChunk", 
    "ChunkType",
    "ChunkingStrategy",
    
    # Specialized chunker types
    "PythonChunkerConfig",
    "JSONChunkerConfig",
    "YAMLChunkerConfig", 
    "TextChunkerConfig",
    "ChunkerOptions",
    "PythonChunkerOptions",
    "JSONChunkerOptions",
    "YAMLChunkerOptions",
    "TextChunkerOptions",
    "ChunkerCapabilities"
]