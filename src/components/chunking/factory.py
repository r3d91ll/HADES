"""
Chunking Component Factory

Factory functions for creating chunking components.
"""

from typing import Dict, Any, Optional, List
import logging

from src.types.components.protocols import Chunker

logger = logging.getLogger(__name__)


def create_chunking_component(
    component_type: str = "core",
    config: Optional[Dict[str, Any]] = None
) -> Chunker:
    """
    Create a chunking component.
    
    Args:
        component_type: Type of chunking component
        config: Optional configuration
        
    Returns:
        Chunker instance
        
    Raises:
        ValueError: If unknown component type
    """
    config = config or {}
    
    # Import chunkers with lazy loading to avoid circular dependencies
    try:
        if component_type == "core":
            from .core.processor import CoreChunker
            chunker = CoreChunker()
            if hasattr(chunker, 'configure'):
                chunker.configure(config)
            return chunker
        else:
            raise ValueError(f"Unknown chunking component type: {component_type}")
    except ImportError as e:
        logger.warning(f"Could not import chunking component {component_type}: {e}")
        raise ValueError(f"Chunking component {component_type} not available: {e}")


def get_available_chunking_components() -> List[str]:
    """Get list of available chunking components."""
    return ["core"]