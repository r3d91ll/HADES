"""
Base type definitions for chunking.

This module provides the core type definitions for chunker protocols,
registries, and configuration structures.
"""

from typing import Dict, List, Any, Optional, Union, Type, TypeVar, Protocol, runtime_checkable
from enum import Enum
from abc import abstractmethod

# Type variable for chunker subclasses
T = TypeVar('T', bound='ChunkerProtocol')


class ChunkingMode(str, Enum):
    """Different modes for chunking documents."""
    
    TEXT = "text"  # Text-based chunking
    CODE = "code"  # Code-aware chunking
    SEMANTIC = "semantic"  # Semantic chunking using embeddings
    HYBRID = "hybrid"  # Combination of multiple approaches
    

class ChunkBoundary(str, Enum):
    """Types of boundaries for chunk splitting."""
    
    SENTENCE = "sentence"  # Split on sentence boundaries
    PARAGRAPH = "paragraph"  # Split on paragraph boundaries
    FUNCTION = "function"  # Split on function boundaries (code)
    CLASS = "class"  # Split on class boundaries (code)
    SECTION = "section"  # Split on section boundaries
    TOKEN_LIMIT = "token_limit"  # Split when token limit reached


@runtime_checkable
class ChunkerProtocol(Protocol):
    """Protocol defining the interface for all chunkers."""
    
    name: str
    config: Dict[str, Any]
    
    @abstractmethod
    def chunk(self, content: Union[str, Dict[str, Any]], **kwargs: Any) -> List[Dict[str, Any]]:
        """Chunk a document into smaller parts.
        
        Args:
            content: Document content to chunk
            **kwargs: Additional arguments specific to the chunker implementation
            
        Returns:
            List of chunk dictionaries
        """
        ...


class ChunkerConfig:
    """Configuration for chunkers."""
    
    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        mode: ChunkingMode = ChunkingMode.TEXT,
        boundary_strategy: ChunkBoundary = ChunkBoundary.SENTENCE,
        preserve_structure: bool = True,
        custom_options: Optional[Dict[str, Any]] = None
    ):
        """Initialize chunker configuration.
        
        Args:
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Number of overlapping tokens between chunks
            mode: Chunking mode to use
            boundary_strategy: Strategy for determining chunk boundaries
            preserve_structure: Whether to preserve document structure
            custom_options: Additional custom options
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.mode = mode
        self.boundary_strategy = boundary_strategy
        self.preserve_structure = preserve_structure
        self.custom_options = custom_options or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "max_tokens": self.max_tokens,
            "overlap_tokens": self.overlap_tokens, 
            "mode": self.mode.value,
            "boundary_strategy": self.boundary_strategy.value,
            "preserve_structure": self.preserve_structure,
            "custom_options": self.custom_options
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkerConfig":
        """Create configuration from dictionary."""
        return cls(
            max_tokens=data.get("max_tokens", 512),
            overlap_tokens=data.get("overlap_tokens", 50),
            mode=ChunkingMode(data.get("mode", "text")),
            boundary_strategy=ChunkBoundary(data.get("boundary_strategy", "sentence")),
            preserve_structure=data.get("preserve_structure", True),
            custom_options=data.get("custom_options", {})
        )


# Registry type for chunker classes
ChunkerRegistry = Dict[str, Type[ChunkerProtocol]]