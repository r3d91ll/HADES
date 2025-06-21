"""
Component Registry System for HADES

This module provides a centralized registry system for discovering, registering,
and instantiating components dynamically based on configuration.
"""

import logging
import importlib
import inspect
from typing import Dict, Any, Optional, Type, Callable, List, Union, Protocol
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from src.types.components.contracts import ComponentType
from src.types.components.protocols import (
    BaseComponent as ComponentProtocol,
    DocumentProcessor,
    Chunker,
    Embedder,
    GraphEnhancer,
    ModelEngine,
    Storage,
    RAGStrategy
)


logger = logging.getLogger(__name__)


class RegistryError(Exception):
    """Exception raised by the component registry system."""
    pass


@dataclass
class ComponentInfo:
    """Information about a registered component."""
    name: str
    component_type: ComponentType
    component_class: Type[ComponentProtocol]
    factory_func: Callable[[Optional[Dict[str, Any]]], ComponentProtocol]
    description: Optional[str] = None
    version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComponentRegistry:
    """
    Central registry for managing and instantiating components.
    
    This registry enables dynamic component discovery, registration, and instantiation
    based on configuration, enabling runtime component swapping and A/B testing.
    """
    
    def __init__(self) -> None:
        """Initialize the component registry."""
        self._components: Dict[str, Dict[str, ComponentInfo]] = {
            "docproc": {},
            "chunking": {},
            "embedding": {},
            "graph_enhancement": {},
            "model_engine": {},
            "storage": {},
            "graph_engine": {},
            "schemas": {},
            "database": {},
            "rag_strategy": {}
        }
        self._type_mapping = {
            ComponentType.DOCPROC: "docproc",
            ComponentType.CHUNKING: "chunking", 
            ComponentType.EMBEDDING: "embedding",
            ComponentType.GRAPH_ENHANCEMENT: "graph_enhancement",
            ComponentType.MODEL_ENGINE: "model_engine",
            ComponentType.STORAGE: "storage",
            ComponentType.GRAPH_ENGINE: "graph_engine",
            ComponentType.SCHEMAS: "schemas",
            ComponentType.DATABASE: "database",
            ComponentType.RAG_STRATEGY: "rag_strategy"
        }
        
        logger.info("Initialized component registry")
    
    def register_component(
        self,
        component_type: Union[ComponentType, str],
        name: str,
        component_class: Type[ComponentProtocol],
        factory_func: Optional[Callable[[Optional[Dict[str, Any]]], ComponentProtocol]] = None,
        description: Optional[str] = None,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a component in the registry.
        
        Args:
            component_type: Type of component or type string
            name: Name to register the component under
            component_class: Class implementing the appropriate protocol
            factory_func: Optional factory function, defaults to simple constructor
            description: Optional component description
            version: Optional component version
            metadata: Optional additional metadata
            
        Raises:
            RegistryError: If registration fails
        """
        try:
            # Normalize component type
            if isinstance(component_type, str):
                type_key = component_type
            else:
                # Must be ComponentType
                type_key = self._type_mapping.get(component_type, str(component_type))
            
            if type_key not in self._components:
                raise RegistryError(f"Unknown component type: {type_key}")
            
            # Create default factory if none provided
            if factory_func is None:
                def default_factory(config: Optional[Dict[str, Any]]) -> ComponentProtocol:
                    try:
                        # Try creating instance without config first (safer)
                        instance = component_class()
                        # Then configure if config provided and method exists
                        if hasattr(instance, 'configure') and config:
                            instance.configure(config)
                        return instance
                    except Exception as e:
                        logger.error(f"Failed to create instance of {component_class}: {e}")
                        raise RegistryError(f"Failed to instantiate component: {e}")
                factory_func = default_factory
            
            # Create component info
            info = ComponentInfo(
                name=name,
                component_type=component_type if isinstance(component_type, ComponentType) else ComponentType.UNKNOWN,
                component_class=component_class,
                factory_func=factory_func,
                description=description,
                version=version,
                metadata=metadata or {}
            )
            
            # Register the component
            self._components[type_key][name] = info
            
            logger.info(f"Registered {type_key} component: {name}")
            
        except Exception as e:
            error_msg = f"Failed to register component {name} of type {component_type}: {e}"
            logger.error(error_msg)
            raise RegistryError(error_msg) from e
    
    def get_component(
        self,
        component_type: Union[ComponentType, str],
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> ComponentProtocol:
        """
        Create a component instance.
        
        Args:
            component_type: Type of component or type string
            name: Name of the component to create
            config: Optional configuration dictionary
            
        Returns:
            Component instance
            
        Raises:
            RegistryError: If component cannot be created
        """
        try:
            # Normalize component type
            if isinstance(component_type, str):
                type_key = component_type
            else:
                # Must be ComponentType
                type_key = self._type_mapping.get(component_type, str(component_type))
            
            if type_key not in self._components:
                raise RegistryError(f"Unknown component type: {type_key}")
            
            if name not in self._components[type_key]:
                available = list(self._components[type_key].keys())
                raise RegistryError(f"Component {name} not found in {type_key}. Available: {available}")
            
            # Get component info and create instance
            info = self._components[type_key][name]
            component = info.factory_func(config)
            
            logger.debug(f"Created {type_key} component: {name}")
            return component
            
        except Exception as e:
            error_msg = f"Failed to create component {name} of type {component_type}: {e}"
            logger.error(error_msg)
            raise RegistryError(error_msg) from e
    
    def list_components(
        self,
        component_type: Optional[Union[ComponentType, str]] = None
    ) -> Dict[str, List[str]]:
        """
        List all registered components.
        
        Args:
            component_type: Optional component type to filter by
            
        Returns:
            Dictionary mapping component types to lists of component names
        """
        if component_type is not None:
            # List components for specific type
            if isinstance(component_type, str):
                type_key = component_type
            elif isinstance(component_type, ComponentType):
                type_key = self._type_mapping.get(component_type, str(component_type))
            else:
                type_key = str(component_type)
            
            if type_key in self._components:
                return {type_key: list(self._components[type_key].keys())}
            else:
                return {}
        else:
            # List all components
            return {
                type_key: list(components.keys())
                for type_key, components in self._components.items()
            }
    
    def get_component_info(
        self,
        component_type: Union[ComponentType, str],
        name: str
    ) -> Optional[ComponentInfo]:
        """
        Get information about a registered component.
        
        Args:
            component_type: Type of component or type string
            name: Name of the component
            
        Returns:
            ComponentInfo if found, None otherwise
        """
        try:
            # Normalize component type
            if isinstance(component_type, str):
                type_key = component_type
            else:
                # Must be ComponentType
                type_key = self._type_mapping.get(component_type, str(component_type))
            
            if type_key in self._components and name in self._components[type_key]:
                return self._components[type_key][name]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get component info for {name} of type {component_type}: {e}")
            return None
    
    def is_component_available(
        self,
        component_type: Union[ComponentType, str],
        name: str
    ) -> bool:
        """
        Check if a component is available.
        
        Args:
            component_type: Type of component or type string
            name: Name of the component
            
        Returns:
            True if component is available, False otherwise
        """
        return self.get_component_info(component_type, name) is not None
    
    def unregister_component(
        self,
        component_type: Union[ComponentType, str],
        name: str
    ) -> bool:
        """
        Unregister a component.
        
        Args:
            component_type: Type of component or type string
            name: Name of the component
            
        Returns:
            True if component was unregistered, False if not found
        """
        try:
            # Normalize component type
            if isinstance(component_type, str):
                type_key = component_type
            else:
                # Must be ComponentType
                type_key = self._type_mapping.get(component_type, str(component_type))
            
            if type_key in self._components and name in self._components[type_key]:
                del self._components[type_key][name]
                logger.info(f"Unregistered {type_key} component: {name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to unregister component {name} of type {component_type}: {e}")
            return False
    
    def auto_discover_components(self, base_path: Optional[Union[str, Path]] = None) -> int:
        """
        Auto-discover and register components from the components directory.
        
        Args:
            base_path: Optional base path to search, defaults to src/components
            
        Returns:
            Number of components discovered and registered
        """
        if base_path is None:
            base_path = Path(__file__).parent
        else:
            base_path = Path(base_path)
        
        discovered_count = 0
        
        try:
            # Search each component type directory
            for type_dir in base_path.iterdir():
                if not type_dir.is_dir() or type_dir.name.startswith('_'):
                    continue
                
                component_type = type_dir.name
                
                # Skip non-component directories
                if component_type not in self._components:
                    continue
                
                discovered_count += self._discover_components_in_directory(type_dir, component_type)
            
            logger.info(f"Auto-discovered {discovered_count} components")
            
        except Exception as e:
            logger.error(f"Failed to auto-discover components: {e}")
        
        return discovered_count
    
    def _discover_components_in_directory(self, type_dir: Path, component_type: str) -> int:
        """
        Discover components in a specific type directory.
        
        Args:
            type_dir: Directory to search
            component_type: Type of components to discover
            
        Returns:
            Number of components discovered
        """
        discovered = 0
        
        try:
            # Look for processor.py files in subdirectories
            for item in type_dir.rglob("processor.py"):
                if item.is_file():
                    try:
                        # Extract component name from path
                        component_name = item.parent.name
                        if component_name == component_type:
                            # Handle core components (e.g., chunking/core/processor.py)
                            component_name = "core"
                        
                        # Import the module
                        relative_path = item.relative_to(Path(__file__).parent.parent)
                        module_path = str(relative_path.with_suffix('')).replace('/', '.')
                        module_path = f"src.{module_path}"
                        
                        module = importlib.import_module(module_path)
                        
                        # Find component classes in the module
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if (hasattr(obj, 'component_type') and 
                                not name.startswith('_') and 
                                name != 'ComponentProtocol'):
                                
                                # Register the component
                                self.register_component(
                                    component_type=component_type,
                                    name=component_name,
                                    component_class=obj,
                                    description=f"Auto-discovered {component_type} component"
                                )
                                discovered += 1
                                break
                        
                    except Exception as e:
                        logger.warning(f"Failed to discover component in {item}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to discover components in {type_dir}: {e}")
        
        return discovered
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the registry.
        
        Returns:
            Dictionary containing registry statistics
        """
        stats: Dict[str, Any] = {
            "total_components": 0,
            "components_by_type": {},
            "component_types": list(self._components.keys())
        }
        
        for type_key, components in self._components.items():
            count = len(components)
            stats["components_by_type"][type_key] = count
            stats["total_components"] += count
        
        return stats


# Global registry instance
_global_registry: Optional[ComponentRegistry] = None


def get_global_registry() -> ComponentRegistry:
    """
    Get the global component registry instance.
    
    Returns:
        Global ComponentRegistry instance
    """
    global _global_registry
    
    if _global_registry is None:
        _global_registry = ComponentRegistry()
        # Auto-discover components on first access
        _global_registry.auto_discover_components()
    
    return _global_registry


def register_component(
    component_type: Union[ComponentType, str],
    name: str,
    component_class: Type[ComponentProtocol],
    factory_func: Optional[Callable[[Optional[Dict[str, Any]]], ComponentProtocol]] = None,
    **kwargs: Any
) -> None:
    """
    Register a component in the global registry.
    
    Args:
        component_type: Type of component
        name: Name to register the component under
        component_class: Class implementing the appropriate protocol
        factory_func: Optional factory function
        **kwargs: Additional registration parameters
    """
    registry = get_global_registry()
    registry.register_component(component_type, name, component_class, factory_func, **kwargs)


def get_component(
    component_type: Union[ComponentType, str],
    name: str,
    config: Optional[Dict[str, Any]] = None
) -> ComponentProtocol:
    """
    Create a component instance from the global registry.
    
    Args:
        component_type: Type of component
        name: Name of the component to create
        config: Optional configuration dictionary
        
    Returns:
        Component instance
    """
    registry = get_global_registry()
    return registry.get_component(component_type, name, config)


def list_components(component_type: Optional[Union[ComponentType, str]] = None) -> Dict[str, List[str]]:
    """
    List all registered components in the global registry.
    
    Args:
        component_type: Optional component type to filter by
        
    Returns:
        Dictionary mapping component types to lists of component names
    """
    registry = get_global_registry()
    return registry.list_components(component_type)