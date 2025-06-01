"""
Chunking strategy interfaces for the HADES-PathRAG chunking system.

This module provides the type definitions for different chunking strategies,
including interfaces, abstract base classes, and related type definitions
for the chunking strategy system.
"""

from typing import Dict, List, Any, Optional, Union, Protocol, TypeVar, runtime_checkable, Callable
from abc import ABC, abstractmethod

from src.types.chunking.chunk import Chunk, ChunkMetadata, ChunkList, ChunkableDocument, OutputFormatType


# Type variable for chunker classes
C = TypeVar('C', bound='ChunkerProtocol')


@runtime_checkable
class ChunkerProtocol(Protocol):
    """Protocol defining the interface for a chunker."""
    
    name: str
    config: Dict[str, Any]
    
    def chunk(self, content: ChunkableDocument, **kwargs: Any) -> ChunkList:
        """Chunk a document into smaller parts.
        
        Args:
            content: Document content to chunk
            **kwargs: Additional arguments specific to the chunker implementation
            
        Returns:
            List of chunk dictionaries
        """
        ...


class BaseChunkerABC(ABC):
    """Abstract base class for all document chunkers.
    
    A chunker is responsible for breaking down a document into smaller, semantically
    meaningful chunks that can be processed and embedded independently. Chunkers may
    be specialized for specific document types (e.g., text, code, PDF) and should
    preserve the semantic structure of the document.
    """
    
    def __init__(self, name: str = "base", config: Optional[Dict[str, Any]] = None):
        """Initialize the base chunker.
        
        Args:
            name: Name of the chunker
            config: Configuration options for the chunker
        """
        self.name = name
        self.config = config or {}
    
    @abstractmethod
    def chunk(self, content: ChunkableDocument, **kwargs: Any) -> ChunkList:
        """Chunk a document into smaller parts.
        
        Args:
            content: Document content to chunk
            **kwargs: Additional arguments specific to the chunker implementation
            
        Returns:
            List of chunk dictionaries, where each chunk contains at least:
            - content: The chunk content
            - metadata: Additional information about the chunk
        """
        pass


# Type definitions for chunking configuration
class ChunkerConfig(Dict[str, Any]):
    """Configuration for a chunker."""
    pass


# Type definitions for chunking registry
ChunkerRegistry = Dict[str, type]


# Type definitions for chunking options
class ChunkingOptions(Dict[str, Any]):
    """Options for chunking."""
    pass


# Type definitions for chunking results
class ChunkingResult(Dict[str, Any]):
    """Result of a chunking operation."""
    document_id: str
    chunks: ChunkList
    metadata: Dict[str, Any]


# Type for chunker factory function
ChunkerFactory = Dict[str, Any]


# Type for chunk processing function
ChunkProcessingFn = Callable[[Chunk], Chunk]


# Type for document chunking function
DocumentChunkerFn = Callable[[ChunkableDocument, Any], ChunkList]
