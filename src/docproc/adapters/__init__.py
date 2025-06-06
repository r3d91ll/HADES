"""
Document format adapters for document processing.

This module provides adapters for different document formats, focusing on Python code files
and a unified Docling adapter for various document types.
"""

from .registry import register_adapter, get_adapter_for_format, get_adapter_class, get_supported_formats
from .base import BaseAdapter

# Import all adapters to ensure they are registered
from .python_adapter import PythonAdapter

# Import DoclingAdapter only if available
try:
    from .docling_adapter import DoclingAdapter
except ImportError:
    DoclingAdapter = None  # type: ignore

# Add more adapters as they are implemented

__all__ = [
    'BaseAdapter',
    'register_adapter',
    'get_adapter_for_format',
    'get_adapter_class',
    'get_supported_formats',
    'PythonAdapter',
]

# Add DoclingAdapter to __all__ only if it's available
if DoclingAdapter is not None:
    __all__.append('DoclingAdapter')
