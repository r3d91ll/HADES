"""
Component Factory System for HADES

This module provides high-level factory functions for creating components
using the component registry system. It supports configuration-driven
component creation and A/B testing scenarios.
"""

import logging
from typing import Dict, Any, Optional, Union, List, Type
from pathlib import Path

from src.types.components.contracts import ComponentType
from src.types.components.protocols import (
    BaseComponent as ComponentProtocol,
    DocumentProcessor,
    Chunker,
    Embedder,
    GraphEnhancer,
    ModelEngine,
    Storage
)
from .registry import get_global_registry, RegistryError


logger = logging.getLogger(__name__)


class ComponentFactory:
    """
    High-level factory for creating components using the registry system.
    
    This factory provides convenient methods for creating components from
    configuration and supports A/B testing scenarios.
    """
    
    def __init__(self, registry=None):
        """
        Initialize the component factory.
        
        Args:
            registry: Optional component registry, defaults to global registry
        """
        self.registry = registry or get_global_registry()
        self.logger = logging.getLogger(__name__)
    
    def create_from_config(
        self,
        component_config: Dict[str, Any],
        component_type: Optional[Union[ComponentType, str]] = None
    ) -> ComponentProtocol:
        """
        Create a component from configuration.
        
        Expected config format:
        {
            "type": "component_name",
            "config": {
                # component-specific configuration
            }
        }
        
        Or for explicit typing:
        {
            "component_type": "embedding",
            "type": "vllm", 
            "config": {
                # component-specific configuration
            }
        }
        
        Args:
            component_config: Configuration dictionary
            component_type: Optional component type override
            
        Returns:
            Component instance
            
        Raises:
            ValueError: If configuration is invalid
            RegistryError: If component cannot be created
        """
        try:
            # Extract component type and name
            if component_type is None:
                if "component_type" in component_config:
                    component_type = component_config["component_type"]
                else:
                    # Try to infer from context or use a default
                    raise ValueError("Component type must be specified in config or as parameter")
            
            component_name = component_config.get("type")
            if not component_name:
                raise ValueError("Component 'type' must be specified in config")
            
            # Extract component configuration
            component_params = component_config.get("config", {})
            
            # Create the component
            component = self.registry.get_component(
                component_type=component_type,
                name=component_name,
                config=component_params
            )
            
            self.logger.info(f"Created {component_type} component: {component_name}")
            return component
            
        except Exception as e:
            error_msg = f"Failed to create component from config: {e}"
            self.logger.error(error_msg)
            raise RegistryError(error_msg) from e
    
    def create_document_processor(
        self,
        processor_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> DocumentProcessor:
        """
        Create a document processor component.
        
        Args:
            processor_type: Type of processor (e.g., "docling", "core")
            config: Optional configuration dictionary
            
        Returns:
            DocumentProcessor instance
        """
        component = self.registry.get_component("docproc", processor_type, config)
        return component  # type: ignore
    
    def create_chunker(
        self,
        chunker_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Chunker:
        """
        Create a chunking component.
        
        Args:
            chunker_type: Type of chunker (e.g., "chonky", "cpu", "text", "code")
            config: Optional configuration dictionary
            
        Returns:
            Chunker instance
        """
        component = self.registry.get_component("chunking", chunker_type, config)
        return component  # type: ignore
    
    def create_embedder(
        self,
        embedder_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Embedder:
        """
        Create an embedding component.
        
        Args:
            embedder_type: Type of embedder (e.g., "gpu", "cpu", "encoder")
            config: Optional configuration dictionary
            
        Returns:
            Embedder instance
        """
        component = self.registry.get_component("embedding", embedder_type, config)
        return component  # type: ignore
    
    def create_graph_enhancer(
        self,
        enhancer_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> GraphEnhancer:
        """
        Create a graph enhancement component.
        
        Args:
            enhancer_type: Type of enhancer (e.g., "isne", "none")
            config: Optional configuration dictionary
            
        Returns:
            GraphEnhancer instance
        """
        component = self.registry.get_component("graph_enhancement", enhancer_type, config)
        return component  # type: ignore
    
    def create_model_engine(
        self,
        engine_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> ModelEngine:
        """
        Create a model engine component.
        
        Args:
            engine_type: Type of engine (e.g., "vllm", "haystack")
            config: Optional configuration dictionary
            
        Returns:
            ModelEngine instance
        """
        component = self.registry.get_component("model_engine", engine_type, config)
        return component  # type: ignore
    
    def create_storage(
        self,
        storage_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Storage:
        """
        Create a storage component.
        
        Args:
            storage_type: Type of storage (e.g., "memory", "arangodb", "networkx")
            config: Optional configuration dictionary
            
        Returns:
            Storage instance
        """
        component = self.registry.get_component("storage", storage_type, config)
        return component  # type: ignore
    
    def create_pipeline_components(
        self,
        pipeline_config: Dict[str, Any]
    ) -> Dict[str, ComponentProtocol]:
        """
        Create all components for a pipeline from configuration.
        
        Expected config format:
        {
            "components": {
                "document_processing": {
                    "type": "docling",
                    "config": {...}
                },
                "chunking": {
                    "type": "chonky", 
                    "config": {...}
                },
                # ... other components
            }
        }
        
        Args:
            pipeline_config: Pipeline configuration dictionary
            
        Returns:
            Dictionary mapping component roles to component instances
        """
        components = {}
        components_config = pipeline_config.get("components", {})
        
        try:
            # Component type mappings
            component_mappings = {
                "document_processing": "docproc",
                "docproc": "docproc",
                "chunking": "chunking",
                "embedding": "embedding",
                "graph_enhancement": "graph_enhancement",
                "graph_engine": "graph_engine",
                "model_engine": "model_engine",
                "storage": "storage",
                "database": "database"
            }
            
            for role, config in components_config.items():
                if role in component_mappings:
                    component_type = component_mappings[role]
                    component = self.create_from_config(config, component_type)
                    components[role] = component
                else:
                    self.logger.warning(f"Unknown component role: {role}")
            
            self.logger.info(f"Created {len(components)} pipeline components")
            return components
            
        except Exception as e:
            error_msg = f"Failed to create pipeline components: {e}"
            self.logger.error(error_msg)
            raise RegistryError(error_msg) from e
    
    def create_ab_test_variants(
        self,
        base_config: Dict[str, Any],
        variants_config: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, ComponentProtocol]]:
        """
        Create component variants for A/B testing.
        
        Args:
            base_config: Base pipeline configuration
            variants_config: List of variant configurations with overrides
            
        Returns:
            Dictionary mapping variant names to component dictionaries
        """
        variants = {}
        
        try:
            for variant in variants_config:
                variant_name = variant.get("name", "unnamed_variant")
                overrides = variant.get("overrides", {})
                
                # Create modified config for this variant
                variant_config = self._apply_config_overrides(base_config, overrides)
                
                # Create components for this variant
                variant_components = self.create_pipeline_components(variant_config)
                variants[variant_name] = variant_components
                
                self.logger.info(f"Created A/B test variant: {variant_name}")
            
            return variants
            
        except Exception as e:
            error_msg = f"Failed to create A/B test variants: {e}"
            self.logger.error(error_msg)
            raise RegistryError(error_msg) from e
    
    def _apply_config_overrides(
        self,
        base_config: Dict[str, Any],
        overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply configuration overrides to base configuration.
        
        Args:
            base_config: Base configuration
            overrides: Configuration overrides
            
        Returns:
            Modified configuration
        """
        import copy
        
        # Deep copy base config
        config = copy.deepcopy(base_config)
        
        # Apply overrides to components section
        components_overrides = overrides.get("components", {})
        if "components" not in config:
            config["components"] = {}
        
        for component_role, override_config in components_overrides.items():
            if component_role in config["components"]:
                # Update existing component config
                config["components"][component_role].update(override_config)
            else:
                # Add new component config
                config["components"][component_role] = override_config
        
        # Apply any top-level overrides
        for key, value in overrides.items():
            if key != "components":
                config[key] = value
        
        return config
    
    def get_available_components(self) -> Dict[str, List[str]]:
        """
        Get all available components by type.
        
        Returns:
            Dictionary mapping component types to lists of available components
        """
        return self.registry.list_components()
    
    def validate_config(
        self,
        config: Dict[str, Any],
        component_type: Optional[str] = None
    ) -> tuple[bool, List[str]]:
        """
        Validate a component configuration.
        
        Args:
            config: Configuration to validate
            component_type: Optional component type for validation
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            if component_type:
                # Validate specific component type
                component_name = config.get("type")
                if not component_name:
                    errors.append("Missing 'type' in component config")
                    return False, errors
                
                # Check if component is available
                if not self.registry.is_component_available(component_type, component_name):
                    available = self.registry.list_components(component_type)
                    errors.append(f"Component {component_name} not available for type {component_type}. Available: {available}")
                    return False, errors
                
                # Try to get component info for validation
                info = self.registry.get_component_info(component_type, component_name)
                if info and hasattr(info.component_class, 'validate_config'):
                    # Use component's own validation if available
                    component_config = config.get("config", {})
                    if not info.component_class.validate_config(component_config):
                        errors.append(f"Component {component_name} config validation failed")
            else:
                # Validate pipeline config
                components_config = config.get("components", {})
                if not components_config:
                    errors.append("No components defined in config")
                    return False, errors
                
                for role, comp_config in components_config.items():
                    comp_valid, comp_errors = self.validate_config(comp_config, role)
                    if not comp_valid:
                        errors.extend([f"{role}: {err}" for err in comp_errors])
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Validation error: {e}")
            return False, errors


# Global factory instance
_global_factory: Optional[ComponentFactory] = None


def get_global_factory() -> ComponentFactory:
    """
    Get the global component factory instance.
    
    Returns:
        Global ComponentFactory instance
    """
    global _global_factory
    
    if _global_factory is None:
        _global_factory = ComponentFactory()
    
    return _global_factory


# Convenience functions using global factory
def create_component(
    component_type: Union[ComponentType, str],
    name: str,
    config: Optional[Dict[str, Any]] = None
) -> ComponentProtocol:
    """
    Create a component using the global factory.
    
    Args:
        component_type: Type of component
        name: Name of the component
        config: Optional configuration
        
    Returns:
        Component instance
    """
    factory = get_global_factory()
    return factory.registry.get_component(component_type, name, config)


def create_from_config(
    component_config: Dict[str, Any],
    component_type: Optional[Union[ComponentType, str]] = None
) -> ComponentProtocol:
    """
    Create a component from configuration using the global factory.
    
    Args:
        component_config: Component configuration
        component_type: Optional component type
        
    Returns:
        Component instance
    """
    factory = get_global_factory()
    return factory.create_from_config(component_config, component_type)


def create_pipeline_components(pipeline_config: Dict[str, Any]) -> Dict[str, ComponentProtocol]:
    """
    Create pipeline components using the global factory.
    
    Args:
        pipeline_config: Pipeline configuration
        
    Returns:
        Dictionary of component instances
    """
    factory = get_global_factory()
    return factory.create_pipeline_components(pipeline_config)


def get_available_components() -> Dict[str, List[str]]:
    """
    Get all available components using the global factory.
    
    Returns:
        Dictionary mapping component types to available components
    """
    factory = get_global_factory()
    return factory.get_available_components()