"""
Enumeration types for document processing.

This module defines standardized enumerations used throughout the document
processing module, particularly for Python code analysis.
"""

from __future__ import annotations

import enum
from typing import Any, Dict, List, Literal, Optional, Set, TypedDict, Union


class RelationshipType(str, enum.Enum):
    """Type of relationship between code elements."""
    
    # Primary relationships (weight 0.8-1.0)
    CALLS = "CALLS"  # Function calling another function
    CONTAINS = "CONTAINS"  # Parent-child relationship (e.g., class contains method)
    IMPLEMENTS = "IMPLEMENTS"  # Implementation of an interface or protocol
    
    # Secondary relationships (weight 0.5-0.7)
    IMPORTS = "IMPORTS"  # Import relationship
    REFERENCES = "REFERENCES"  # Reference to another code element
    EXTENDS = "EXTENDS"  # Inheritance relationship
    
    # Tertiary relationships (weight 0.2-0.4)
    SIMILAR_TO = "SIMILAR_TO"  # Semantic similarity
    RELATED_TO = "RELATED_TO"  # General relationship


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
