"""
Common interfaces for the HADES modular pipeline architecture.

This module defines the base interfaces that are shared across all pipelines
and components. These interfaces establish the contracts that concrete
implementations must adhere to.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid


class Component(ABC):
    """Base interface for all pipeline components."""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the component with configuration.
        
        Args:
            config: Configuration dictionary for the component
        """
        pass
    
    @property
    @abstractmethod
    def component_type(self) -> str:
        """Return the type of this component."""
        pass
    
    @property
    @abstractmethod
    def component_name(self) -> str:
        """Return the name of this component."""
        pass


class Configurable(ABC):
    """Interface for components that can be configured."""
    
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Return the default configuration for this component.
        
        Returns:
            Default configuration dictionary
        """
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate the provided configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass


class Identifiable:
    """Mixin for components that have unique identifiers."""
    
    def __init__(self):
        self._id = str(uuid.uuid4())
    
    @property
    def id(self) -> str:
        """Return the unique identifier for this component."""
        return self._id


class Pipeline(Component, Configurable, ABC):
    """Base interface for all pipelines."""
    
    @abstractmethod
    def run(self, input_data: Any) -> Any:
        """Run the pipeline on the provided input data.
        
        Args:
            input_data: Input data for the pipeline
            
        Returns:
            Output data from the pipeline
        """
        pass
    
    @abstractmethod
    def register_component(self, component: Component) -> None:
        """Register a component with this pipeline.
        
        Args:
            component: Component to register
        """
        pass


class Loggable(ABC):
    """Interface for components that support logging."""
    
    @abstractmethod
    def log(self, level: str, message: str, **kwargs) -> None:
        """Log a message.
        
        Args:
            level: Log level (e.g., 'debug', 'info', 'warning', 'error')
            message: Message to log
            **kwargs: Additional data to include in the log
        """
        pass


class Serializable(ABC):
    """Interface for components that can be serialized."""
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert the component to a dictionary.
        
        Returns:
            Dictionary representation of the component
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Serializable':
        """Create a component from a dictionary.
        
        Args:
            data: Dictionary representation of the component
            
        Returns:
            New component instance
        """
        pass


class Pluggable(Component, ABC):
    """Interface for components that can be loaded as plugins."""
    
    @classmethod
    @abstractmethod
    def get_plugin_type(cls) -> str:
        """Return the plugin type for this component.
        
        Returns:
            Plugin type string
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_plugin_name(cls) -> str:
        """Return the plugin name for this component.
        
        Returns:
            Plugin name string
        """
        pass