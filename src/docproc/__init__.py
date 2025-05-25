"""
Document Processing Module for HADES-PathRAG

This module provides a unified interface for processing various document formats
including PDF, HTML, code files, and structured data formats (JSON, XML, YAML).
It converts them to standardized formats for both RAG and direct model inference.
"""

from .core import process_document, process_text, detect_format, get_format_for_document

__all__ = [
    "process_document",
    "process_text",
    "detect_format",
    "get_format_for_document"
]
