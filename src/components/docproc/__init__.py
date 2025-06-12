"""
Document Processing Components

This module provides document processing components that implement the
DocumentProcessor protocol for various document formats and processing needs.

Available components:
- core: Core document processor using HADES adapters (Python, Markdown, JSON, YAML, etc.)
- docling: Docling-based processor for complex documents (PDF, Word, etc.)
"""

from .core.processor import CoreDocumentProcessor
from .docling.processor import DoclingDocumentProcessor
from .factory import create_docproc_component, get_available_docproc_components

__all__ = [
    "CoreDocumentProcessor",
    "DoclingDocumentProcessor",
    "create_docproc_component",
    "get_available_docproc_components"
]