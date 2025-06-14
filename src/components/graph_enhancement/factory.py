"""
Graph Enhancement Component Factory

This module provides a factory for creating graph enhancement components.
Integrated with the global component registry system.
"""

import logging
from typing import Dict, Any, Optional, Type, Callable, Tuple

from src.types.components.protocols import GraphEnhancer
from src.types.components.contracts import ComponentType
from .isne.core.processor import CoreISNE
from .isne.inductive.processor import InductiveISNE
from .isne.inference.processor import ISNEInferenceEnhancer as InferenceISNE
from .isne.none.processor import NoneISNE
from .isne.training.processor import ISNETrainingEnhancer as TrainingISNE


logger = logging.getLogger(__name__)


# Registry of available graph enhancement implementations
GRAPH_ENHANCEMENT_REGISTRY: Dict[str, Tuple[Type[GraphEnhancer], Callable[[Optional[Dict[str, Any]]], GraphEnhancer]]] = {
    "core": (CoreISNE, lambda config: CoreISNE(config=config)),
    "inductive": (InductiveISNE, lambda config: InductiveISNE(config=config)),
    "inference": (InferenceISNE, lambda config: InferenceISNE(config=config)),
    "none": (NoneISNE, lambda config: NoneISNE(config=config)),
    "training": (TrainingISNE, lambda config: TrainingISNE(config=config)),
    "isne": (CoreISNE, lambda config: CoreISNE(config=config)),  # Alias for core
}


def _register_components() -> None:
    """Register graph enhancement components with the global registry."""
    try:
        from src.components.registry import register_component
        
        for name, (component_class, factory_func) in GRAPH_ENHANCEMENT_REGISTRY.items():
            # Skip aliases for cleaner registry
            if name == "isne":
                continue
                
            register_component(
                component_type=ComponentType.GRAPH_ENHANCEMENT,
                name=name,
                component_class=component_class,
                factory_func=factory_func,
                description=f"Graph enhancement component: {name}"
            )
        
        logger.info(f"Registered {len(GRAPH_ENHANCEMENT_REGISTRY)-1} graph enhancement components with global registry")
        
    except ImportError:
        logger.warning("Global registry not available, using local registry only")
    except Exception as e:
        logger.error(f"Failed to register graph enhancement components: {e}")


# Auto-register components when module is imported
_register_components()


def create_graph_enhancement_component(
    component_name: str, 
    config: Optional[Dict[str, Any]] = None
) -> GraphEnhancer:
    """
    Create a graph enhancement component by name.
    
    Args:
        component_name: Name of the component to create (e.g., "core", "inductive", "inference", "none", "training")
        config: Optional configuration dictionary
        
    Returns:
        GraphEnhancer instance
        
    Raises:
        ValueError: If component_name is not recognized
        RuntimeError: If component creation fails
    """
    if component_name not in GRAPH_ENHANCEMENT_REGISTRY:
        available = list(GRAPH_ENHANCEMENT_REGISTRY.keys())
        raise ValueError(f"Unknown graph enhancement component: {component_name}. Available: {available}")
    
    try:
        component_class, factory_func = GRAPH_ENHANCEMENT_REGISTRY[component_name]
        component = factory_func(config)
        
        logger.info(f"Created graph enhancement component: {component_name}")
        return component
        
    except Exception as e:
        logger.error(f"Failed to create graph enhancement component {component_name}: {e}")
        raise RuntimeError(f"Component creation failed: {e}") from e


def get_available_graph_enhancement_components() -> Dict[str, Type[GraphEnhancer]]:
    """
    Get all available graph enhancement components.
    
    Returns:
        Dictionary mapping component names to their classes
    """
    return {name: cls for name, (cls, _) in GRAPH_ENHANCEMENT_REGISTRY.items()}


def register_graph_enhancement_component(
    name: str, 
    component_class: Type[GraphEnhancer],
    factory_func: Optional[Callable[[Optional[Dict[str, Any]]], GraphEnhancer]] = None
) -> None:
    """
    Register a new graph enhancement component.
    
    Args:
        name: Name to register the component under
        component_class: Class implementing GraphEnhancer protocol
        factory_func: Optional factory function, defaults to simple constructor
    """
    if factory_func is None:
        # Create a safe factory function that handles different constructor signatures
        def safe_factory(config: Optional[Dict[str, Any]]) -> GraphEnhancer:
            try:
                return component_class(config=config)  # type: ignore
            except TypeError:
                # Fallback for classes that don't accept config parameter
                return component_class()  # type: ignore
        factory_func = safe_factory
    
    GRAPH_ENHANCEMENT_REGISTRY[name] = (component_class, factory_func)
    logger.info(f"Registered graph enhancement component: {name}")


def is_graph_enhancement_component_available(name: str) -> bool:
    """
    Check if a graph enhancement component is available.
    
    Args:
        name: Component name to check
        
    Returns:
        True if component is available, False otherwise
    """
    return name in GRAPH_ENHANCEMENT_REGISTRY