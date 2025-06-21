"""
Document Processing Component Factory

Factory functions for creating document processing components.
"""

from typing import Dict, Any, Optional, List
import logging

from src.types.components.protocols import DocumentProcessor
from .core.processor import CoreDocumentProcessor
from .docling.processor import DoclingDocumentProcessor

logger = logging.getLogger(__name__)


def create_docproc_component(
    component_type: str = "core",
    config: Optional[Dict[str, Any]] = None
) -> DocumentProcessor:
    """
    Create a document processing component.
    
    Args:
        component_type: Type of docproc component ("core", "docling")
        config: Optional configuration
        
    Returns:
        DocumentProcessor instance
        
    Raises:
        ValueError: If unknown component type
    """
    config = config or {}
    
    if component_type == "core":
        processor = CoreDocumentProcessor()
        if hasattr(processor, 'configure'):
            processor.configure(config)
        return processor
    elif component_type == "docling":
        processor = DoclingDocumentProcessor()
        if hasattr(processor, 'configure'):
            processor.configure(config)
        return processor
    else:
        raise ValueError(f"Unknown docproc component type: {component_type}")


def get_available_docproc_components() -> List[str]:
    """Get list of available document processing components."""
    return ["core", "docling"]