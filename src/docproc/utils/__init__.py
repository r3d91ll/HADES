"""
Utility functions for document processing.

This module provides utility functions for detecting file formats, handling file operations,
and other common tasks.
"""

from .format_detector import detect_format_from_path, detect_format_from_content

from typing import Dict, List, Any, Optional, Union, Tuple, Set, cast, Callable, TypeVar

__all__ = [
    'detect_format_from_path',
    'detect_format_from_content'
]
