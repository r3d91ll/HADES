"""
Enumeration types for document processing.

This module defines standardized enumerations used throughout the document
processing module, particularly for Python code analysis.
"""

from __future__ import annotations

import enum
from typing import Any, Dict, List, Literal, Optional, Set, TypedDict, Union


# RelationshipType is now available via RelationType from src.types.common
# Import it instead of defining a separate enum
from ...common import RelationType as RelationshipType


class ImportSourceType(str, enum.Enum):
    """Source type of an import."""
    
    STDLIB = "stdlib"  # Standard library
    THIRD_PARTY = "third-party"  # Third-party package
    LOCAL = "local"  # Local module
    UNKNOWN = "unknown"  # Unknown source


class AccessLevel(str, enum.Enum):
    """Access level of a code element."""
    
    PUBLIC = "public"  # Public access (no underscore)
    PROTECTED = "protected"  # Protected access (single underscore)
    PRIVATE = "private"  # Private access (double underscore)


class ContentCategory(str, enum.Enum):
    """Category of document content."""
    
    TEXT = "text"  # Plain text content
    CODE = "code"  # Programming code
    MARKUP = "markup"  # Markup language (HTML, XML, etc.)
    DATA = "data"  # Structured data (JSON, YAML, etc.)
    BINARY = "binary"  # Binary content
    MIXED = "mixed"  # Mixed content types
