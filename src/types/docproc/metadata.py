"""Metadata type definitions for document processing.

This module provides type definitions for document metadata and entities,
including standardized structures for metadata extraction and validation.
"""

from typing import Any, Dict, List, Optional, Union, TypedDict
from datetime import datetime


class EntityDict(TypedDict, total=False):
    """Dictionary representation of an entity extracted from a document.
    
    This TypedDict defines the standardized structure for entities
    extracted during document processing, such as named entities,
    code symbols, or other structured information.
    """
    
    entity_id: Optional[str]
    """Unique identifier for the entity (if available)."""
    
    entity_type: str
    """Type of the entity (e.g., 'person', 'organization', 'class', 'function')."""
    
    text: str
    """The text content of the entity."""
    
    start_pos: Optional[int]
    """Starting position in the document text."""
    
    end_pos: Optional[int]
    """Ending position in the document text."""
    
    line_number: Optional[int]
    """Line number where the entity appears."""
    
    confidence: Optional[float]
    """Confidence score for the entity extraction (0.0-1.0)."""
    
    normalized_text: Optional[str]
    """Normalized form of the entity text (e.g., lowercase, stemmed)."""
    
    metadata: Optional[Dict[str, Any]]
    """Additional metadata for the entity."""
    
    linked_entities: Optional[List[str]]
    """References to other entities this entity is linked to."""


class MetadataDict(TypedDict, total=False):
    """Dictionary representation of document metadata.
    
    This TypedDict defines the standardized structure for document metadata
    extracted during document processing, providing context and additional
    information about the document.
    """
    
    # Document identification and source
    source_path: Optional[str]
    """Path to the source document."""
    
    filename: Optional[str]
    """Name of the source file."""
    
    file_extension: Optional[str]
    """Extension of the source file."""
    
    file_size: Optional[int]
    """Size of the source file in bytes."""
    
    # Document content information
    title: Optional[str]
    """Document title or name."""
    
    subtitle: Optional[str]
    """Document subtitle if available."""
    
    summary: Optional[str]
    """Brief summary or abstract of the document."""
    
    language: Optional[str]
    """Language of the document content."""
    
    # Python code specific metadata
    document_type: Optional[str]
    """Type of document (e.g., 'code', 'text', 'markdown')."""
    
    code_type: Optional[str]
    """Programming language type of code document."""
    
    module_name: Optional[str]
    """Name of the Python module."""
    
    imports: Optional[List[Any]]
    """List of imports in the Python code."""
    
    function_count: Optional[str]
    """Number of functions in the Python code as a string."""
    
    class_count: Optional[str]
    """Number of classes in the Python code as a string."""
    
    import_count: Optional[str]
    """Number of imports in the Python code as a string."""
    
    content_type: str
    """Type of content (e.g., 'text', 'code')."""
    
    format: str
    """Format of the document (e.g., 'markdown', 'python')."""
    
    mime_type: Optional[str]
    """MIME type of the document if available."""
    
    encoding: Optional[str]
    """Character encoding of the document."""
    
    # Content statistics
    word_count: Optional[int]
    """Number of words in the document."""
    
    character_count: Optional[int]
    """Number of characters in the document."""
    
    line_count: Optional[int]
    """Number of lines in the document."""
    
    # Authorship and dates
    author: Optional[Union[str, List[str]]]
    """Author(s) of the document."""
    
    organization: Optional[str]
    """Organization associated with the document."""
    
    created_date: Optional[Union[str, datetime]]
    """Document creation date."""
    
    modified_date: Optional[Union[str, datetime]]
    """Document last modification date."""
    
    published_date: Optional[Union[str, datetime]]
    """Document publication date if available."""
    
    # Document classification
    tags: Optional[List[str]]
    """List of tags or keywords associated with the document."""
    
    categories: Optional[List[str]]
    """Categories the document belongs to."""
    
    classification: Optional[str]
    """Classification or sensitivity level of the document."""
    
    # Processing metadata
    processing_metadata: Optional[Dict[str, Any]]
    """Metadata about the processing itself."""
    
    extraction_date: Optional[Union[str, datetime]]
    """When the metadata was extracted."""
    
    extractor_version: Optional[str]
    """Version of the extractor that processed this document."""
    
    confidence: Optional[float]
    """Overall confidence in the metadata extraction (0.0-1.0)."""
    
    # Additional information
    custom_metadata: Optional[Dict[str, Any]]
    """Additional custom metadata fields."""
