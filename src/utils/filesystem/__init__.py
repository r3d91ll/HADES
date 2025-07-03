"""
Filesystem utilities for HADES.

This module provides utilities for filesystem operations including:
- Ignore file handling (.gitignore, .dockerignore, etc.)
- File metadata extraction
- Path filtering
"""

from .ignore_handler import IgnoreFileHandler
from .metadata_extractor import FileMetadataExtractor

__all__ = ['IgnoreFileHandler', 'FileMetadataExtractor']