"""
Code chunking type definitions for the HADES-PathRAG chunking system.

This module provides specialized type definitions for code chunking operations,
including code-specific chunk types, metadata, and processing options.
"""

from typing import Dict, List, Any, Optional, Union, TypedDict, Literal
from pydantic import BaseModel, Field
import uuid
from datetime import datetime

from src.types.chunking.chunk import (
    Chunk, ChunkMetadata, ChunkList, ChunkContent, ChunkId,
    ChunkableDocument, OutputFormatType
)


class CodeChunkMetadata(ChunkMetadata):
    """Metadata specific to code chunks."""
    # Code-specific metadata
    language: str  # Programming language (e.g., "python", "javascript")
    symbol_type: Optional[str]  # Type of symbol (e.g., "function", "class", "method")
    name: Optional[str]  # Name of the symbol (e.g., function name, class name)
    parent: Optional[str]  # Parent symbol (e.g., class name for a method)
    symbol_id: Optional[str]  # Unique identifier for the symbol
    scope: Optional[str]  # Scope of the symbol (e.g., "global", "class", "function")
    imports: Optional[List[str]]  # List of imports used by this chunk
    dependencies: Optional[List[str]]  # List of dependencies for this chunk
    complexity: Optional[float]  # Cyclomatic complexity of the code
    imports_resolved: Optional[bool]  # Whether imports have been resolved


class CodeChunk(Chunk):
    """A chunk of code content with code-specific metadata."""
    content: str  # Code content
    metadata: CodeChunkMetadata  # Code-specific metadata


class CodeChunkingOptions(TypedDict, total=False):
    """Options for code chunking."""
    max_tokens: int  # Maximum tokens per chunk
    use_ast: bool  # Whether to use AST-based chunking
    language: str  # Programming language
    use_symbol_boundaries: bool  # Whether to respect symbol boundaries
    include_imports: bool  # Whether to include imports in each chunk
    resolve_references: bool  # Whether to resolve references between chunks
    respect_scope: bool  # Whether to respect scoping rules
    output_format: OutputFormatType  # Output format type
    preserve_structure: bool  # Whether to preserve code structure
    use_line_boundaries: bool  # Whether to respect line boundaries


class CodeChunkingResult(TypedDict):
    """Result of a code chunking operation."""
    document_id: str  # ID of the original document
    chunks: List[CodeChunk]  # List of code chunks
    metadata: Dict[str, Any]  # Metadata about the chunking operation
    processing_time: float  # Time taken to process the document
    symbol_table: Optional[Dict[str, Any]]  # Symbol table for the document
    relationships: Optional[List[Dict[str, Any]]]  # Relationships between symbols


class SymbolInfo(TypedDict, total=False):
    """Information about a symbol in code."""
    name: str  # Symbol name
    type: str  # Symbol type (function, class, etc.)
    line_start: int  # Starting line number
    line_end: int  # Ending line number
    parent: Optional[str]  # Parent symbol name
    docstring: Optional[str]  # Documentation string
    children: List[str]  # Child symbol names
    dependencies: List[str]  # Symbols this symbol depends on
    references: List[Dict[str, Any]]  # References to other symbols


class SymbolRelationship(TypedDict, total=False):
    """Relationship between two symbols in code."""
    source: str  # Source symbol name
    target: str  # Target symbol name
    type: str  # Relationship type (e.g., "calls", "inherits")
    line: Optional[int]  # Line number where relationship occurs
    confidence: Optional[float]  # Confidence score for the relationship


class CodeChunkerConfig(TypedDict, total=False):
    """Configuration for a code chunker."""
    max_tokens: int  # Maximum tokens per chunk
    language: str  # Programming language
    use_ast: bool  # Whether to use AST-based chunking
    use_symbol_boundaries: bool  # Whether to respect symbol boundaries
    include_imports: bool  # Whether to include imports in each chunk
    resolve_references: bool  # Whether to resolve references between chunks
    respect_scope: bool  # Whether to respect scoping rules
    fallback_to_line_chunking: bool  # Whether to fall back to line-based chunking
    preserve_structure: bool  # Whether to preserve code structure
    token_overlap: int  # Number of tokens to overlap between chunks


# Type aliases for clarity
CodeChunkList = List[CodeChunk]
SymbolInfoList = List[SymbolInfo]
SymbolRelationshipList = List[SymbolRelationship]
SymbolTable = Dict[str, SymbolInfo]


# Language-specific types

class PythonSymbolInfo(SymbolInfo):
    """Python-specific symbol information."""
    args: Optional[List[str]]  # Function/method arguments
    returns: Optional[str]  # Return type annotation
    bases: Optional[List[str]]  # Base classes for a class
    decorators: Optional[List[str]]  # Function/method decorators
    is_async: Optional[bool]  # Whether the function is async
    is_property: Optional[bool]  # Whether the method is a property
    is_static_method: Optional[bool]  # Whether the method is static
    is_class_method: Optional[bool]  # Whether the method is a class method


class JavaScriptSymbolInfo(SymbolInfo):
    """JavaScript-specific symbol information."""
    params: Optional[List[str]]  # Function parameters
    is_async: Optional[bool]  # Whether the function is async
    is_generator: Optional[bool]  # Whether the function is a generator
    is_arrow_function: Optional[bool]  # Whether the function is an arrow function
    is_method: Optional[bool]  # Whether the function is a method
    is_constructor: Optional[bool]  # Whether the method is a constructor
    prototype: Optional[str]  # Prototype chain information


# Helper function to create code-specific chunk metadata
def create_code_chunk_metadata(
    chunk_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    path: str = "unknown",
    document_type: str = "code",
    source_file: str = "unknown",
    line_start: int = 0,
    line_end: int = 0,
    token_count: int = 0,
    language: str = "unknown",
    symbol_type: Optional[str] = None,
    name: Optional[str] = None,
    parent: Optional[str] = None,
    symbol_id: Optional[str] = None,
    scope: Optional[str] = None,
    imports: Optional[List[str]] = None,
    dependencies: Optional[List[str]] = None,
    complexity: Optional[float] = None,
    **kwargs: Any
) -> CodeChunkMetadata:
    """Create code-specific chunk metadata.
    
    Args:
        chunk_id: Unique identifier for the chunk
        parent_id: ID of the parent document
        path: Path to the source document
        document_type: Type of document
        source_file: Original source file
        line_start: Starting line number
        line_end: Ending line number
        token_count: Number of tokens in the chunk
        language: Programming language
        symbol_type: Type of symbol
        name: Name of the symbol
        parent: Parent symbol
        symbol_id: Unique identifier for the symbol
        scope: Scope of the symbol
        imports: List of imports used by this chunk
        dependencies: List of dependencies for this chunk
        complexity: Cyclomatic complexity of the code
        **kwargs: Additional metadata fields
    
    Returns:
        A CodeChunkMetadata dictionary
    """
    # Create base metadata
    metadata: CodeChunkMetadata = {
        "id": chunk_id or str(uuid.uuid4()),
        "parent_id": parent_id,
        "path": path,
        "document_type": document_type,
        "source_file": source_file,
        "line_start": line_start,
        "line_end": line_end,
        "token_count": token_count,
        "chunk_type": "code",
        "language": language,
        "created_at": datetime.now().isoformat()
    }
    
    # Add code-specific fields if provided
    if symbol_type is not None:
        metadata["symbol_type"] = symbol_type
    if name is not None:
        metadata["name"] = name
    if parent is not None:
        metadata["parent"] = parent
    if symbol_id is not None:
        metadata["symbol_id"] = symbol_id
    if scope is not None:
        metadata["scope"] = scope
    if imports is not None:
        metadata["imports"] = imports
    if dependencies is not None:
        metadata["dependencies"] = dependencies
    if complexity is not None:
        metadata["complexity"] = complexity
    
    # Add any additional metadata
    for key, value in kwargs.items():
        metadata[key] = value
    
    return metadata
