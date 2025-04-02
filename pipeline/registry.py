"""
Plugin registry system for the HADES modular pipeline architecture.

This module provides a registry for dynamically loading and managing
pluggable components in the HADES system.
"""

from typing import Any, Dict, List, Optional, Type, Set, Tuple
import importlib
import inspect
import pkgutil
import logging
from pathlib import Path

from hades.pipeline.interfaces import Component, Pluggable


class PluginRegistry:
    """Registry for managing pluggable components."""
    
    def __init__(self):
        """Initialize the plugin registry."""
        self._plugins: Dict[str, Dict[str, Type[Pluggable]]] = {}
        self._instances: Dict[str, Dict[str, Pluggable]] = {}
        self._logger = logging.getLogger(__name__)
    
    def register(self, plugin_type: str, plugin_name: str, plugin_class: Type[Pluggable]) -> None:
        """Register a plugin class.
        
        Args:
            plugin_type: Type of the plugin (e.g., 'embedding_model', 'storage_backend')
            plugin_name: Name of the plugin
            plugin_class: Plugin class
        """
        if not issubclass(plugin_class, Pluggable):
            raise TypeError(f"Plugin class {plugin_class.__name__} must implement Pluggable interface")
        
        if plugin_type not in self._plugins:
            self._plugins[plugin_type] = {}
        
        self._plugins[plugin_type][plugin_name] = plugin_class
        self._logger.debug(f"Registered plugin: {plugin_type}.{plugin_name}")
    
    def get_plugin_class(self, plugin_type: str, plugin_name: str) -> Optional[Type[Pluggable]]:
        """Get a plugin class by type and name.
        
        Args:
            plugin_type: Type of the plugin
            plugin_name: Name of the plugin
            
        Returns:
            Plugin class or None if not found
        """
        return self._plugins.get(plugin_type, {}).get(plugin_name)
    
    def get_plugin_instance(self, plugin_type: str, plugin_name: str, 
                           config: Optional[Dict[str, Any]] = None) -> Optional[Pluggable]:
        """Get or create a plugin instance by type and name.
        
        Args:
            plugin_type: Type of the plugin
            plugin_name: Name of the plugin
            config: Optional configuration for the plugin
            
        Returns:
            Plugin instance or None if not found
        """
        # Check if instance already exists
        if plugin_type in self._instances and plugin_name in self._instances[plugin_type]:
            return self._instances[plugin_type][plugin_name]
        
        # Get plugin class
        plugin_class = self.get_plugin_class(plugin_type, plugin_name)
        if not plugin_class:
            return None
        
        # Create instance
        try:
            instance = plugin_class()
            if config:
                instance.initialize(config)
            
            # Store instance
            if plugin_type not in self._instances:
                self._instances[plugin_type] = {}
            self._instances[plugin_type][plugin_name] = instance
            
            return instance
        except Exception as e:
            self._logger.error(f"Error creating plugin instance {plugin_type}.{plugin_name}: {e}")
            return None
    
    def list_plugins(self, plugin_type: Optional[str] = None) -> Dict[str, List[str]]:
        """List available plugins.
        
        Args:
            plugin_type: Optional plugin type to filter by
            
        Returns:
            Dictionary mapping plugin types to lists of plugin names
        """
        if plugin_type:
            return {plugin_type: list(self._plugins.get(plugin_type, {}).keys())}
        else:
            return {t: list(plugins.keys()) for t, plugins in self._plugins.items()}
    
    def discover_plugins(self, package_name: str) -> int:
        """Discover and register plugins from a package.
        
        Args:
            package_name: Name of the package to scan for plugins
            
        Returns:
            Number of plugins discovered
        """
        count = 0
        package = importlib.import_module(package_name)
        package_path = getattr(package, '__path__', [None])[0]
        
        if not package_path:
            self._logger.error(f"Could not determine path for package {package_name}")
            return count
        
        for _, module_name, is_pkg in pkgutil.iter_modules([package_path]):
            full_module_name = f"{package_name}.{module_name}"
            
            try:
                module = importlib.import_module(full_module_name)
                
                # If it's a package, recursively discover plugins
                if is_pkg:
                    count += self.discover_plugins(full_module_name)
                
                # Find plugin classes in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, Pluggable) and obj != Pluggable and 
                        not inspect.isabstract(obj)):
                        try:
                            plugin_type = obj.get_plugin_type()
                            plugin_name = obj.get_plugin_name()
                            self.register(plugin_type, plugin_name, obj)
                            count += 1
                        except Exception as e:
                            self._logger.warning(f"Error registering plugin {name}: {e}")
            
            except Exception as e:
                self._logger.warning(f"Error importing module {full_module_name}: {e}")
        
        return count
    
    def clear(self) -> None:
        """Clear all registered plugins and instances."""
        self._plugins.clear()
        self._instances.clear()


# Singleton instance
_registry = PluginRegistry()


def get_registry() -> PluginRegistry:
    """Get the global plugin registry.
    
    Returns:
        Global plugin registry instance
    """
    return _registry


def register_plugin(plugin_type: str, plugin_name: str, plugin_class: Type[Pluggable]) -> None:
    """Register a plugin with the global registry.
    
    Args:
        plugin_type: Type of the plugin
        plugin_name: Name of the plugin
        plugin_class: Plugin class
    """
    _registry.register(plugin_type, plugin_name, plugin_class)


def get_plugin_class(plugin_type: str, plugin_name: str) -> Optional[Type[Pluggable]]:
    """Get a plugin class from the global registry.
    
    Args:
        plugin_type: Type of the plugin
        plugin_name: Name of the plugin
        
    Returns:
        Plugin class or None if not found
    """
    return _registry.get_plugin_class(plugin_type, plugin_name)


def get_plugin_instance(plugin_type: str, plugin_name: str, 
                       config: Optional[Dict[str, Any]] = None) -> Optional[Pluggable]:
    """Get or create a plugin instance from the global registry.
    
    Args:
        plugin_type: Type of the plugin
        plugin_name: Name of the plugin
        config: Optional configuration for the plugin
        
    Returns:
        Plugin instance or None if not found
    """
    return _registry.get_plugin_instance(plugin_type, plugin_name, config)


def discover_plugins(package_name: str) -> int:
    """Discover and register plugins from a package using the global registry.
    
    Args:
        package_name: Name of the package to scan for plugins
        
    Returns:
        Number of plugins discovered
    """
    return _registry.discover_plugins(package_name)