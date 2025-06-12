"""
Component Factory for Dynamic Pipeline Component Loading

This module provides factory classes for dynamically loading pipeline
components based on configuration.
"""

import logging
from typing import Dict, Any, Optional, Type, Protocol, List
from abc import ABC, abstractmethod

from src.orchestration.core.pipeline_config import ComponentConfig
from src.orchestration.pipelines.data_ingestion.stages.base import PipelineStage

# Import concrete implementations
from src.orchestration.pipelines.data_ingestion.stages.document_processor import DocumentProcessorStage
from src.orchestration.pipelines.data_ingestion.stages.chunking import ChunkingStage
from src.orchestration.pipelines.data_ingestion.stages.embedding import EmbeddingStage
from src.orchestration.pipelines.data_ingestion.stages.isne import ISNEStage
from src.orchestration.pipelines.data_ingestion.stages.storage import StorageStage

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """Registry for available component implementations."""
    
    def __init__(self):
        self._components: Dict[str, Dict[str, Type[PipelineStage]]] = {
            'document_processor': {},
            'chunker': {},
            'embedder': {},
            'graph_enhancer': {},
            'storage': {}
        }
        
        # Register default implementations
        self._register_default_components()
    
    def _register_default_components(self):
        """Register default component implementations."""
        # Document processors
        self._components['document_processor']['docling'] = DocumentProcessorStage
        self._components['document_processor']['default'] = DocumentProcessorStage
        
        # Chunkers
        self._components['chunker']['cpu'] = ChunkingStage
        self._components['chunker']['default'] = ChunkingStage
        
        # Embedders
        self._components['embedder']['modernbert'] = EmbeddingStage
        self._components['embedder']['default'] = EmbeddingStage
        
        # Graph enhancers
        self._components['graph_enhancer']['isne'] = ISNEStage
        self._components['graph_enhancer']['default'] = ISNEStage
        
        # Storage
        self._components['storage']['arangodb'] = StorageStage
        self._components['storage']['default'] = StorageStage
    
    def register_component(self, 
                         component_type: str,
                         implementation_name: str,
                         component_class: Type[PipelineStage]):
        """
        Register a new component implementation.
        
        Args:
            component_type: Type of component (e.g., 'document_processor')
            implementation_name: Name of the implementation (e.g., 'docling')
            component_class: Class implementing PipelineStage
        """
        if component_type not in self._components:
            raise ValueError(f"Unknown component type: {component_type}")
        
        self._components[component_type][implementation_name] = component_class
        logger.info(f"Registered {implementation_name} for {component_type}")
    
    def get_component_class(self,
                          component_type: str,
                          implementation_name: str) -> Optional[Type[PipelineStage]]:
        """
        Get component class by type and implementation name.
        
        Args:
            component_type: Type of component
            implementation_name: Name of implementation
            
        Returns:
            Component class or None if not found
        """
        return self._components.get(component_type, {}).get(
            implementation_name,
            self._components.get(component_type, {}).get('default')
        )
    
    def list_implementations(self, component_type: str) -> List[str]:
        """List available implementations for a component type."""
        return list(self._components.get(component_type, {}).keys())


class ComponentFactory:
    """
    Factory for creating pipeline components based on configuration.
    
    This factory handles:
    - Dynamic component instantiation
    - Configuration injection
    - Fallback handling
    - Component validation
    """
    
    def __init__(self, registry: Optional[ComponentRegistry] = None):
        """
        Initialize component factory.
        
        Args:
            registry: Component registry (creates default if None)
        """
        self.registry = registry or ComponentRegistry()
    
    def create_component(self,
                        component_type: str,
                        component_config: ComponentConfig,
                        stage_name: Optional[str] = None) -> PipelineStage:
        """
        Create a pipeline component from configuration.
        
        Args:
            component_type: Type of component to create
            component_config: Component configuration
            stage_name: Optional stage name override
            
        Returns:
            Instantiated pipeline stage
            
        Raises:
            ValueError: If component cannot be created
        """
        if not component_config.enabled:
            raise ValueError(f"Component {component_type} is disabled")
        
        # Try primary implementation
        component_class = self.registry.get_component_class(
            component_type,
            component_config.implementation
        )
        
        if component_class is None and component_config.fallback:
            # Try fallback implementation
            logger.warning(
                f"Primary implementation '{component_config.implementation}' "
                f"not found for {component_type}, trying fallback '{component_config.fallback}'"
            )
            component_class = self.registry.get_component_class(
                component_type,
                component_config.fallback
            )
        
        if component_class is None:
            raise ValueError(
                f"No implementation found for {component_type}: "
                f"'{component_config.implementation}'"
            )
        
        # Create stage name
        if stage_name is None:
            stage_name = f"{component_type}_{component_config.implementation}"
        
        # Instantiate component
        try:
            component = component_class(
                name=stage_name,
                config=component_config.config
            )
            logger.info(
                f"Created {component_type} component: "
                f"{component_config.implementation}"
            )
            return component
        
        except Exception as e:
            logger.error(
                f"Failed to create {component_type} component "
                f"'{component_config.implementation}': {e}"
            )
            raise ValueError(
                f"Failed to instantiate {component_type}: {e}"
            ) from e
    
    def create_placeholder_component(self,
                                   component_type: str,
                                   stage_name: str) -> PipelineStage:
        """
        Create a placeholder/no-op component.
        
        Args:
            component_type: Type of component
            stage_name: Stage name
            
        Returns:
            Placeholder pipeline stage
        """
        class PlaceholderStage(PipelineStage):
            """Placeholder stage that passes data through unchanged."""
            
            def _process(self, data: Any) -> Any:
                logger.info(f"Placeholder {component_type} passing through data")
                return data
            
            def _validate_input(self, data: Any) -> bool:
                return True
            
            def _validate_output(self, data: Any) -> bool:
                return True
        
        return PlaceholderStage(name=stage_name, config={})


# Global registry instance
_global_registry = ComponentRegistry()


def get_global_registry() -> ComponentRegistry:
    """Get the global component registry."""
    return _global_registry


def register_component(component_type: str,
                      implementation_name: str,
                      component_class: Type[PipelineStage]):
    """
    Register a component in the global registry.
    
    Args:
        component_type: Type of component
        implementation_name: Name of implementation
        component_class: Component class
    """
    _global_registry.register_component(
        component_type,
        implementation_name,
        component_class
    )


def create_component(component_type: str,
                    component_config: ComponentConfig,
                    stage_name: Optional[str] = None) -> PipelineStage:
    """
    Create a component using the global registry.
    
    Args:
        component_type: Type of component
        component_config: Component configuration
        stage_name: Optional stage name
        
    Returns:
        Created pipeline stage
    """
    factory = ComponentFactory(_global_registry)
    return factory.create_component(
        component_type,
        component_config,
        stage_name
    )