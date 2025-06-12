"""
Type definitions for document processing adapters.

This module provides type definitions for document processing adapters, protocols,
and related configuration options. These types enable consistent interfaces and
configuration across different document format handlers.
"""

from abc import ABC
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, TypedDict, Callable, ClassVar, Protocol

# Import shared types
from src.types.common import NodeID, DocumentID
from src.types.components.docproc.enums import ContentCategory


class ExtractorOptions(TypedDict, total=False):
    """Options for document content extraction."""
    
    extract_metadata: bool
    extract_entities: bool
    extract_code_elements: bool
    extract_relationships: bool
    extract_links: bool
    max_content_length: int
    include_raw_content: bool


class MetadataExtractionConfig(TypedDict, total=False):
    """Configuration for metadata extraction."""
    
    extract_authors: bool  
    extract_title: bool
    extract_created_date: bool
    extract_modified_date: bool
    extract_keywords: bool
    extract_summary: bool
    max_metadata_size: int
    use_ai_extraction: bool
    custom_metadata_fields: List[str]


class EntityExtractionConfig(TypedDict, total=False):
    """Configuration for entity extraction."""
    
    entity_types: List[str]
    min_entity_length: int
    max_entities: int
    threshold: float
    context_window: int
    enable_ner: bool
    custom_entity_patterns: Dict[str, str]


class ChunkingPreparationConfig(TypedDict, total=False):
    """Configuration for preparing document for chunking."""
    
    add_section_markers: bool
    mark_code_blocks: bool
    mark_headers: bool
    mark_lists: bool
    mark_paragraphs: bool
    mark_custom_elements: List[str]
    max_section_length: int


class ProcessorConfig(TypedDict, total=False):
    """Complete document processor configuration."""
    
    metadata_extraction: MetadataExtractionConfig
    entity_extraction: EntityExtractionConfig
    chunking_preparation: ChunkingPreparationConfig
    format_options: Dict[str, Dict[str, Any]]


class DocumentProcessingError(TypedDict):
    """Details about a document processing error."""
    
    error_type: str
    message: str
    stacktrace: Optional[str]
    location: Optional[str]
    recoverable: bool
    

class FormatDetectionResult(TypedDict):
    """Result of format detection."""
    
    format: str
    confidence: float
    content_category: ContentCategory
    detected_by: str


# In src/types/docproc/adapter.py
from .document import ProcessedDocument, DocumentEntity, DocumentMetadata, ChunkPreparedDocument

class AdapterProtocol(Protocol):
    format_name: ClassVar[str]
    format_extensions: ClassVar[List[str]]
    content_category: ClassVar[ContentCategory]
    
    def process(self, 
        file_path: Union[str, Path], 
        options: Optional[ExtractorOptions] = None) -> ProcessedDocument: ... 
    
    def process_text(self, 
        text: str, 
        options: Optional[ExtractorOptions] = None) -> ProcessedDocument: ...
    
    def extract_entities(self, 
        # 'text' should be 'content: Union[str, ProcessedDocument]' to match BaseAdapter
        content: Union[str, ProcessedDocument], # <--- Changed from 'text: str'
        options: Optional[EntityExtractionConfig] = None) -> List[DocumentEntity]: ...
    
    def extract_metadata(self, 
        # 'text' should be 'content: Union[str, ProcessedDocument]'
        # 'file_info' is not in BaseAdapter.extract_metadata
        content: Union[str, ProcessedDocument], # <--- Changed from 'text: str'
        options: Optional[MetadataExtractionConfig] = None) -> DocumentMetadata: ...
    
    # prepare_for_chunking needs similar review if it's being used and causing issues
    def prepare_for_chunking(self, 
        document: ProcessedDocument, # <--- Changed from Dict[str, Any]
        options: Optional[ChunkingPreparationConfig] = None) -> ChunkPreparedDocument: ...


# Type aliases for callback functions
ProcessSuccessCallback = Callable[[Dict[str, Any], Optional[Path]], None]
ProcessErrorCallback = Callable[[Union[str, Path], Exception], None]


class AdapterRegistry(TypedDict):
    """Type for the adapter registry mapping."""
    
    by_name: Dict[str, type]
    by_extension: Dict[str, type]
    categories: Dict[ContentCategory, List[str]]
