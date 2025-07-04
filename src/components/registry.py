"""
Component Registry for HADES.

This module provides a registry system for managing and accessing
components across the HADES system.
"""

import logging
from typing import Dict, Any, Optional, Type, TypeVar, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ComponentInfo:
    """Information about a registered component."""
    name: str
    component_type: str
    factory: Callable[..., Any]
    config: Optional[Dict[str, Any]] = None
    instance: Optional[Any] = None
    is_singleton: bool = True


class ComponentRegistry:
    """Registry for managing components."""
    
    def __init__(self) -> None:
        """Initialize the component registry."""
        self._components: Dict[str, ComponentInfo] = {}
        self._instances: Dict[str, Any] = {}
    
    def register(
        self,
        name: str,
        component_type: str,
        factory: Callable[..., T],
        config: Optional[Dict[str, Any]] = None,
        is_singleton: bool = True
    ) -> None:
        """Register a component factory."""
        if name in self._components:
            logger.warning(f"Overwriting existing component: {name}")
        
        self._components[name] = ComponentInfo(
            name=name,
            component_type=component_type,
            factory=factory,
            config=config,
            is_singleton=is_singleton
        )
        logger.info(f"Registered component: {name} of type {component_type}")
    
    def get(self, name: str, **kwargs: Any) -> Any:
        """Get a component instance by name."""
        if name not in self._components:
            raise ValueError(f"Component not found: {name}")
        
        info = self._components[name]
        
        # For singletons, return cached instance
        if info.is_singleton and name in self._instances:
            return self._instances[name]
        
        # Create new instance
        config = {**(info.config or {}), **kwargs}
        instance = info.factory(**config)
        
        # Cache singleton instances
        if info.is_singleton:
            self._instances[name] = instance
        
        return instance
    
    def list_components(self, component_type: Optional[str] = None) -> Dict[str, ComponentInfo]:
        """List all registered components, optionally filtered by type."""
        if component_type:
            return {
                name: info 
                for name, info in self._components.items() 
                if info.component_type == component_type
            }
        return self._components.copy()
    
    def clear(self) -> None:
        """Clear all registered components and instances."""
        self._components.clear()
        self._instances.clear()


# Global registry instance
_registry = ComponentRegistry()


def register_component(
    name: str,
    component_type: str,
    factory: Callable[..., T],
    config: Optional[Dict[str, Any]] = None,
    is_singleton: bool = True
) -> None:
    """Register a component in the global registry."""
    _registry.register(name, component_type, factory, config, is_singleton)


def get_component(name: str, **kwargs: Any) -> Any:
    """Get a component from the global registry."""
    return _registry.get(name, **kwargs)


def list_components(component_type: Optional[str] = None) -> Dict[str, ComponentInfo]:
    """List components from the global registry."""
    return _registry.list_components(component_type)


def clear_registry() -> None:
    """Clear the global registry."""
    _registry.clear()


__all__ = [
    "ComponentInfo",
    "ComponentRegistry",
    "register_component",
    "get_component", 
    "list_components",
    "clear_registry"
]