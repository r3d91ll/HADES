"""
HADES Component System

This module provides the component registry and factory system for HADES.
All components are automatically registered when this module is imported.
"""

# Import registry and factory systems
from .registry import (
    ComponentRegistry,
    get_global_registry,
    register_component,
    get_component,
    list_components
)
from .factory import (
    ComponentFactory,
    get_global_factory,
    create_component,
    create_from_config,
    create_pipeline_components,
    get_available_components
)

# Import all component factories to trigger auto-registration
from .docproc import factory as docproc_factory
from .chunking import factory as chunking_factory
from .embedding import factory as embedding_factory
from .graph_enhancement import factory as graph_enhancement_factory
from .storage import factory as storage_factory
from .model_engine import factory as model_engine_factory

# Import specific factories for convenience
from .docproc.factory import create_docproc_component
from .chunking.factory import create_chunking_component
from .embedding.factory import create_embedding_component
from .graph_enhancement.factory import create_graph_enhancement_component
from .storage.factory import create_storage_component
from .model_engine.factory import create_model_engine

# Import contracts and protocols for component interfaces
from ..types.components.contracts import (
    # Enums
    ComponentType,
    ProcessingStatus,
    ContentCategory,
    
    # Base Models
    ComponentMetadata,
    
    # Document Processing Contracts
    DocumentProcessingInput,
    ProcessedDocument,
    DocumentProcessingOutput,
    
    # Chunking Contracts
    ChunkingInput,
    DocumentChunk,
    ChunkingOutput,
    
    # Embedding Contracts
    EmbeddingInput,
    ChunkEmbedding,
    EmbeddingOutput,
    
    # Graph Enhancement Contracts
    GraphEnhancementInput,
    EnhancedEmbedding,
    GraphEnhancementOutput,
    
    # Storage Contracts
    StorageInput,
    StoredItem,
    StorageOutput,
    
    # Query/Retrieval Contracts
    QueryInput,
    RetrievalResult,
    QueryOutput,
)

from ..types.components.protocols import (
    # Base Protocol
    BaseComponent,
    
    # Component Protocols
    DocumentProcessor,
    Chunker,
    Embedder,
    GraphEnhancer,
    Storage,
    SchemaValidator,
    DatabaseConnector,
)


__all__ = [
    # Registry system
    "ComponentRegistry",
    "get_global_registry", 
    "register_component",
    "get_component",
    "list_components",
    
    # Factory system
    "ComponentFactory",
    "get_global_factory",
    "create_component",
    "create_from_config", 
    "create_pipeline_components",
    "get_available_components",
    
    # Specific component factories
    "create_docproc_component",
    "create_chunking_component",
    "create_embedding_component", 
    "create_graph_enhancement_component",
    "create_storage_component",
    "create_model_engine",
    
    # Enums
    "ComponentType",
    "ProcessingStatus", 
    "ContentCategory",
    
    # Base Models
    "ComponentMetadata",
    
    # Document Processing
    "DocumentProcessingInput",
    "ProcessedDocument",
    "DocumentProcessingOutput",
    
    # Chunking
    "ChunkingInput",
    "DocumentChunk", 
    "ChunkingOutput",
    
    # Embedding
    "EmbeddingInput",
    "ChunkEmbedding",
    "EmbeddingOutput",
    
    # Graph Enhancement
    "GraphEnhancementInput",
    "EnhancedEmbedding",
    "GraphEnhancementOutput",
    
    # Storage
    "StorageInput",
    "StoredItem",
    "StorageOutput",
    
    # Query/Retrieval
    "QueryInput",
    "RetrievalResult",
    "QueryOutput",
    
    # Protocols
    "BaseComponent",
    "DocumentProcessor",
    "Chunker",
    "Embedder",
    "GraphEnhancer",
    "Storage",
    "SchemaValidator",
    "DatabaseConnector",
]


# Initialize the global registry and factory
_registry = get_global_registry()
_factory = get_global_factory()

# Log registration status
import logging
logger = logging.getLogger(__name__)

def get_initialization_status():
    """Get the initialization status of the component system."""
    stats = _registry.get_registry_stats()
    logger.info(f"Component system initialized with {stats['total_components']} components")
    logger.info(f"Components by type: {stats['components_by_type']}")
    return stats


# Auto-initialize when module is imported
get_initialization_status()