"""Chunking type definitions.

This module provides centralized type definitions for the chunking system,
including chunk types, chunk metadata, and chunk relationships.
"""

# Import types from chunk module
from src.types.chunking.chunk import (
    Chunk, ChunkMetadata, ChunkModel, TextChunk, CodeChunk,
    ChunkDict, ChunkList, ChunkContent, ChunkId,
    OutputFormatType, ChunkableDocument,
    create_chunk_metadata, create_chunk
)

# Import types from strategy module
from src.types.chunking.strategy import (
    ChunkerProtocol, BaseChunkerABC,
    ChunkerConfig, ChunkerRegistry, ChunkingOptions, ChunkingResult,
    ChunkerFactory, ChunkProcessingFn, DocumentChunkerFn
)

# Import types from text module
from src.types.chunking.text import (
    TextChunkMetadata, TextChunk, TextChunkingOptions, TextChunkingResult,
    SplitPoint, TextChunkerConfig, SentenceInfo, ParagraphInfo,
    TokenInfo, DocumentTextStats,
    TextChunkList, SplitPointList, SentenceList, ParagraphList,
    create_text_chunk_metadata
)

# Import types from code module
from src.types.chunking.code import (
    CodeChunkMetadata, CodeChunk, CodeChunkingOptions, CodeChunkingResult,
    SymbolInfo, SymbolRelationship, CodeChunkerConfig,
    CodeChunkList, SymbolInfoList, SymbolRelationshipList, SymbolTable,
    PythonSymbolInfo, JavaScriptSymbolInfo,
    create_code_chunk_metadata
)

__all__ = [
    # Chunk types
    "Chunk", "ChunkMetadata", "ChunkModel", "TextChunk", "CodeChunk",
    "ChunkDict", "ChunkList", "ChunkContent", "ChunkId",
    "OutputFormatType", "ChunkableDocument",
    "create_chunk_metadata", "create_chunk",
    
    # Strategy types
    "ChunkerProtocol", "BaseChunkerABC",
    "ChunkerConfig", "ChunkerRegistry", "ChunkingOptions", "ChunkingResult",
    "ChunkerFactory", "ChunkProcessingFn", "DocumentChunkerFn",
    
    # Text types
    "TextChunkMetadata", "TextChunk", "TextChunkingOptions", "TextChunkingResult",
    "SplitPoint", "TextChunkerConfig", "SentenceInfo", "ParagraphInfo",
    "TokenInfo", "DocumentTextStats",
    "TextChunkList", "SplitPointList", "SentenceList", "ParagraphList",
    "create_text_chunk_metadata",
    
    # Code types
    "CodeChunkMetadata", "CodeChunk", "CodeChunkingOptions", "CodeChunkingResult",
    "SymbolInfo", "SymbolRelationship", "CodeChunkerConfig",
    "CodeChunkList", "SymbolInfoList", "SymbolRelationshipList", "SymbolTable",
    "PythonSymbolInfo", "JavaScriptSymbolInfo",
    "create_code_chunk_metadata"
]
