"""
Common enumeration types for HADES-PathRAG.

This module defines enumerations that are used across multiple components
of the system, providing consistent type definitions.
"""
from __future__ import annotations
from enum import Enum, auto


class SchemaVersion(str, Enum):
    """Schema version enumeration for backward compatibility tracking."""
    V1 = "1.0.0"
    V2 = "2.0.0"  # Current version


class RelationType(str, Enum):
    """Enum representing types of relationships between documents."""
    CONTAINS = "contains"
    REFERENCES = "references"
    CONNECTS_TO = "connects_to"
    SIMILAR_TO = "similar_to"
    DEPENDS_ON = "depends_on"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    DERIVED_FROM = "derived_from"
    IMPLEMENTS = "implements"
    CUSTOM = "custom"


class DocumentType(str, Enum):
    """Enum representing types of documents."""
    CODE = "code"
    MARKDOWN = "markdown"
    TEXT = "text"
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    XML = "xml"
    YAML = "yaml"


class ProcessingStage(str, Enum):
    """Enum representing document processing stages."""
    RAW = "raw"
    PREPROCESSED = "preprocessed" 
    CHUNKED = "chunked"
    EMBEDDED = "embedded"
    INDEXED = "indexed"
    FAILED = "failed"


class ProcessingStatus(str, Enum):
    """Enum representing processing status of a document or task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
