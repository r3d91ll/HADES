"""Adapter type definitions for document processing.

This module provides type definitions for document adapters, which are responsible
for converting documents from various formats into a standardized structure.
"""

from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, TypedDict, Union
from pathlib import Path
from src.types.docproc.metadata import MetadataDict, EntityDict


class AdapterConfig(TypedDict, total=False):
    """Configuration options for document adapters."""
    
    extract_metadata: bool
    """Whether to extract metadata from the document."""
    
    extract_entities: bool
    """Whether to extract entities from the document."""
    
    validation_level: Literal["strict", "warn", "none"]
    """How strictly to validate the output."""
    
    max_content_length: Optional[int]
    """Maximum length of content to extract."""
    
    include_raw_content: bool
    """Whether to include raw content in the output."""
    
    format_specific_options: Dict[str, Any]
    """Format-specific processing options."""


class AdapterOptions(TypedDict, total=False):
    """Options for document processing adapters."""
    
    format_override: Optional[str]
    """Override automatic format detection with specified format."""
    
    extract_metadata: bool
    """Whether to extract metadata from the document."""
    
    extract_entities: bool
    """Whether to extract entities from the document."""
    
    validation_level: Literal["strict", "warn", "none"]
    """How strictly to validate the output."""
    
    max_content_length: Optional[int]
    """Maximum length of content to extract."""
    
    include_raw_content: bool
    """Whether to include raw content in the output."""


class ProcessedDocument(TypedDict, total=False):
    """Standardized document structure returned by adapters."""
    
    id: str
    """Unique identifier for the document."""
    
    source: str
    """Path or identifier for document source."""
    
    path: Optional[str]
    """Path to the source document (legacy field)."""
    
    content: str
    """Processed document content."""
    
    content_type: str
    """Format of the content (e.g., 'markdown', 'text')."""
    
    content_category: str
    """Category of content (e.g., 'text', 'code')."""
    
    format: str
    """Format of the original document (e.g., 'python', 'pdf')."""
    
    raw_content: Optional[str]
    """Original unprocessed document content (when available)."""
    
    processing_time: Optional[float]
    """Time taken to process the document in seconds."""
    
    metadata: Dict[str, Any]
    """Document metadata."""
    
    entities: List[Dict[str, Any]]
    """Entities extracted from the document."""
    
    error: Optional[str]
    """Error message if processing failed."""
    
    # Internal fields used by the implementation
    _validation_error: Optional[str]
    """Validation error message if validation failed."""
    
    _validated: Optional[bool]
    """Whether the document has been validated."""
    
    # Allow extension with additional fields
    __extra_items__: Dict[str, Any]


class AdapterResult(TypedDict, total=False):
    """Result of adapter processing with status information."""
    
    success: bool
    """Whether processing was successful."""
    
    document: Optional[ProcessedDocument]
    """Processed document if successful."""
    
    error: Optional[str]
    """Error message if processing failed."""
    
    processing_time: float
    """Time taken to process the document in seconds."""


class DocumentProcessorType(Protocol):
    """Protocol defining the interface for document processors."""
    
    def process(self, file_path: Union[str, Path], options: Optional[AdapterOptions] = None) -> ProcessedDocument:
        """Process a document file, converting it to a standardized format."""
        ...
    
    def process_text(self, text: str, format_type: str = "text", options: Optional[AdapterOptions] = None) -> ProcessedDocument:
        """Process text content directly, assuming a specific format."""
        ...
    
    def extract_metadata(self, content: Union[str, Dict[str, Any]], options: Optional[AdapterOptions] = None) -> MetadataDict:
        """Extract metadata from document content."""
        ...
    
    def extract_entities(self, content: Union[str, Dict[str, Any]], options: Optional[AdapterOptions] = None) -> List[EntityDict]:
        """Extract entities from document content."""
        ...
