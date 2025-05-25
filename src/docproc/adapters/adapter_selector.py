"""
Adapter selector for document processing based on content category.

This module provides functionality to select the appropriate adapter for a document
based on its content category (code or text) and format type.
"""

import logging
from typing import Dict, Type, Optional

from .base import BaseAdapter
from .registry import get_adapter_class
from .python_code_adapter import PythonCodeAdapter
from .json_adapter import JSONAdapter
from .yaml_adapter import YAMLAdapter
from src.docproc.utils.format_detector import get_content_category

# Set up logging
logger = logging.getLogger(__name__)

# Registry of specialized adapters by format
_CODE_ADAPTERS: Dict[str, Type[BaseAdapter]] = {
    "python": PythonCodeAdapter,
    "json": JSONAdapter,
    "yaml": YAMLAdapter,
    # Add more code format adapters as they are implemented
}

def select_adapter_for_document(format_type: str) -> BaseAdapter:
    """
    Select the appropriate adapter for a document based on its format and content category.
    
    This function routes:
    - Code files (python, json, yaml, etc.) to specialized code adapters
    - Text files (markdown, pdf, etc.) to the general text adapter (Docling)
    
    Args:
        format_type: Format type of the document
        
    Returns:
        Initialized adapter instance
    """
    # Get the content category (code or text)
    content_category = get_content_category(format_type)
    
    if content_category == "code":
        # For code formats, use specialized code adapters
        if format_type in _CODE_ADAPTERS:
            logger.info(f"Using specialized {format_type} code adapter")
            return _CODE_ADAPTERS[format_type]()
        else:
            # For unsupported code formats, fall back to the generic adapter
            logger.info(f"No specialized adapter for {format_type}, using generic adapter")
            adapter_class = get_adapter_class(format_type)
            return adapter_class()
    else:
        # For text formats, use the default adapter from registry
        logger.info(f"Using text adapter for {format_type}")
        adapter_class = get_adapter_class(format_type)
        return adapter_class()
