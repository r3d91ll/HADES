"""
HADES Component System

This module provides the component registry and factory system for HADES.
All components are automatically registered when this module is imported.
"""

from typing import Dict, Any

# Import registry and factory systems
from .registry import (
    ComponentRegistry,
    get_global_registry,
    register_component,
    get_component,
    list_components
)

from .factory import (
    create_component,
    create_rag_strategy,
    create_embedder,
    create_graph_enhancer
)

# Import available component factories to trigger auto-registration
from .embedding import factory as embedding_factory
from .graph_enhancement import factory as graph_enhancement_factory

# Import specific factories for convenience
from .embedding.factory import create_embedding_component
from .graph_enhancement.factory import create_graph_enhancement_component

# Import contracts and protocols for component interfaces
from ..types.components.contracts import (
    ComponentType,
    ProcessingStatus,
    ComponentMetadata,
    RAGStrategyInput,
    RAGStrategyOutput,
    RAGMode,
    RAGResult,
    PathInfo
)

from ..types.components.protocols import (
    BaseComponent,
    Embedder,
    GraphEnhancer,
    RAGStrategy
)

__all__ = [
    # Registry system
    "ComponentRegistry",
    "get_global_registry", 
    "register_component",
    "get_component",
    "list_components",
    
    # Factory system
    "create_component",
    "create_rag_strategy",
    "create_embedder",
    "create_graph_enhancer",
    
    # Specific component factories
    "create_embedding_component", 
    "create_graph_enhancement_component",
    
    # Enums
    "ComponentType",
    "ProcessingStatus",
    "RAGMode",
    
    # Base Models
    "ComponentMetadata",
    "RAGStrategyInput",
    "RAGStrategyOutput", 
    "RAGResult",
    "PathInfo",
    
    # Protocols
    "BaseComponent",
    "Embedder",
    "GraphEnhancer", 
    "RAGStrategy",
]

# Initialize the global registry
_registry = get_global_registry()

# Log registration status
import logging
logger = logging.getLogger(__name__)

def get_initialization_status() -> Dict[str, Any]:
    """Get the initialization status of the component system."""
    stats = _registry.get_registry_stats()
    logger.info(f"Component system initialized with {stats['total_components']} components")
    logger.info(f"Components by type: {stats['components_by_type']}")
    return stats

# Auto-initialize when module is imported
get_initialization_status()