"""
Component type definitions and contracts.

This module contains all the Pydantic models and type definitions for the
component-based architecture. These models define the contracts that all
component implementations must follow.
"""

from .contracts import (
    # Input/Output Contracts
    DocumentProcessingInput,
    DocumentProcessingOutput,
    ChunkingInput, 
    ChunkingOutput,
    EmbeddingInput,
    EmbeddingOutput,
    GraphEnhancementInput,
    GraphEnhancementOutput,
    StorageInput,
    StorageOutput,
    
    # Data Models
    ProcessedDocument,
    DocumentChunk,
    ChunkEmbedding,
    EnhancedEmbedding,
    StoredItem,
    
    # Query Models
    QueryInput,
    QueryOutput,
    RetrievalResult,
    
    # Metadata Models
    ComponentMetadata
)

from .protocols import (
    # Protocol Interfaces
    BaseComponent,
    DocumentProcessor,
    Chunker,
    Embedder,
    GraphEnhancer,
    Storage,
    ComponentFactory
)

__all__ = [
    # Contracts
    "DocumentProcessingInput",
    "DocumentProcessingOutput",
    "ChunkingInput",
    "ChunkingOutput", 
    "EmbeddingInput",
    "EmbeddingOutput",
    "GraphEnhancementInput",
    "GraphEnhancementOutput",
    "StorageInput",
    "StorageOutput",
    
    # Data Models
    "ProcessedDocument",
    "DocumentChunk", 
    "ChunkEmbedding",
    "EnhancedEmbedding",
    "StoredItem",
    
    # Query Models
    "QueryInput",
    "QueryOutput",
    "RetrievalResult",
    
    # Metadata
    "ComponentMetadata",
    
    # Protocols
    "BaseComponent",
    "DocumentProcessor",
    "Chunker",
    "Embedder",
    "GraphEnhancer",
    "Storage",
    "ComponentFactory"
]