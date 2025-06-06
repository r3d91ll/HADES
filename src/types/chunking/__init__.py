"""
Type definitions for the chunking module.

This package contains centralized type definitions for the document chunking
components of the HADES system, including chunker protocols, document models,
and chunk schemas.
"""

# Import base chunker types
from src.types.chunking.base import (
    ChunkerProtocol,
    ChunkerRegistry,
    ChunkerConfig,
    ChunkingMode,
    ChunkBoundary
)

# Import document types  
from src.types.chunking.document import (
    ChunkInfo,
    SymbolInfo,
    DocumentBaseType,
    DocumentSchemaType,
    BaseDocument,
    DocumentSchema,
    DocumentSchemaBase
)

# Import chunk types
from src.types.chunking.chunk import (
    ChunkMetadata,
    TextChunk,
    CodeChunk,
    ChunkType,
    ChunkingStrategy
)

# Import specialized chunker types
from src.types.chunking.chunkers import (
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
    "DocumentSchema",
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