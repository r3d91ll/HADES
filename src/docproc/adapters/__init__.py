"""
Document format adapters for document processing.

This module provides adapters for different document formats, focusing on Python code files
and a unified Docling adapter for various document types.
"""

from .registry import register_adapter, get_adapter_for_format, get_adapter_class, get_supported_formats
from .base import BaseAdapter

# Import all adapters to ensure they are registered
from .python_adapter import PythonAdapter
from .docling_adapter import DoclingAdapter

# Add more adapters as they are implemented

__all__ = [
    'BaseAdapter',
    'register_adapter',
    'get_adapter_for_format',
    'get_adapter_class',
    'get_supported_formats',
    'PythonAdapter',
    'DoclingAdapter'
]
