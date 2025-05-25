"""
Base adapter interface for document processing.

This module defines the abstract base class that all format-specific adapters must implement.
The focus is on producing standardized JSON objects that can be passed between pipeline stages.

Format-specific adapters (PDF, HTML, Python, etc.) extend this base class and implement
the required methods to handle their specific document format. The adapter system enables:

1. Consistent document processing across many different formats
2. Standardized output structure for downstream pipeline stages
3. Centralized configuration for processing behavior
4. Extensibility through the registry system

Adapters should use the centralized configuration system to determine processing behavior.

Example:
    # Creating a format-specific adapter
    class MarkdownAdapter(BaseAdapter):
        def process(self, file_path, content=None):
            # Implementation for markdown processing
            return {"id": "...", "content": "..."}

    # Registering the adapter
    from src.docproc.adapters.registry import register_adapter
    register_adapter("markdown", MarkdownAdapter)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, cast

from src.config.preprocessor_config import load_config
from src.types.common import (
    PreProcessorConfig, 
    MetadataExtractionConfig,
    EntityExtractionConfig,
    ChunkingPreparationConfig
)


class BaseAdapter(ABC):
    """
    Base interface for all document format adapters.
    
    The adapter's primary responsibility is to convert documents from various formats
    into a standardized JSON structure that can be passed between pipeline stages.
    
    The standard JSON structure includes:
    - id: Unique identifier for the document
    - source: Source path/string
    - content: Document content (extracted text)
    - format: Document format (e.g., pdf, html, text)
    - metadata: Document metadata (title, author, creation date, etc.)
    - entities: Extracted entities from the content (named entities, references, etc.)
    - raw_content: Original content when possible (for debugging purposes)
    
    This abstract class ensures all format adapters implement consistent interfaces for:
    1. Full document processing (process method)
    2. Metadata extraction (extract_metadata method)
    3. Entity extraction (extract_entities method)
    4. Text-only processing (process_text method)
    
    All adapters should use the centralized configuration system for their settings,
    making the processing behavior configurable without code changes.
    
    Example usage:
        # Using a concrete adapter implementation
        pdf_adapter = PDFAdapter()
        document = pdf_adapter.process("/path/to/document.pdf")
        
        # Just extracting metadata
        with open("/path/to/document.md", "r") as f:
            content = f.read()
            metadata = markdown_adapter.extract_metadata(content)
            print(f"Title: {metadata.get('title')}")
    """
    
    def __init__(self, format_type: Optional[str] = None):
        """
        Initialize the adapter with optional format-specific configuration.
        
        Args:
            format_type: Format this adapter handles (used to retrieve specific config)
        """
        self.format_type = format_type
        self._load_configuration()
    
    def _load_configuration(self) -> None:
        """
        Load configuration from the centralized configuration system.
        """
        try:
            # Load complete configuration
            self.config = load_config()
            
            # Get format-specific settings if format_type is provided
            self.format_config: Dict[str, Any] = {}
            if self.format_type:
                preprocessor_config = self.config.get("preprocessor_config", {})
                self.format_config = preprocessor_config.get(self.format_type, {})
            
            # Get global configuration sections
            self.metadata_config = cast(
                MetadataExtractionConfig, 
                self.config.get("metadata_extraction", {})
            )
            
            self.entity_config = cast(
                EntityExtractionConfig,
                self.config.get("entity_extraction", {})
            )
            
            self.chunking_config = cast(
                ChunkingPreparationConfig,
                self.config.get("chunking_preparation", {})
            )
            
        except Exception as e:
            # Fallback to empty config if loading fails
            self.config = {}
            self.format_config = {}
            self.metadata_config = cast(MetadataExtractionConfig, {})
            self.entity_config = cast(EntityExtractionConfig, {})
            self.chunking_config = cast(ChunkingPreparationConfig, {})
            
            import logging
            logging.getLogger(__name__).warning(f"Failed to load configuration: {e}")
    
    @abstractmethod
    def process(self, file_path: Union[str, Path], options: Optional[Union[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Process a file or content string and convert to standardized format.
        
        This is the primary entry point for document processing. Implementations must:
        1. Load the content from file_path
        2. Extract the text content appropriate for the format
        3. Generate a unique document ID
        4. Extract metadata using extract_metadata method
        5. Extract entities using extract_entities method
        6. Format everything into the standardized document structure
        
        Args:
            file_path: Path to the document file (as string or Path object)
            options: Optional processing options (as dict or format string)
            
        Returns:
            Dictionary with processed document in the standard format:
            {
                "id": "unique-identifier",
                "content": "Extracted text content",
                "path": "Original file path",
                "format": "Document format (e.g., pdf, html)",
                "metadata": {
                    "title": "Document title",
                    "author": "Document author",
                    ...
                },
                "entities": [
                    {"type": "person", "text": "Entity name", "start": 10, "end": 15},
                    ...
                ]
            }
            
        Raises:
            FileNotFoundError: If the file doesn't exist and no content is provided
            ValueError: If the document cannot be processed properly
            
        Example:
            >>> adapter = TextAdapter()
            >>> result = adapter.process("/path/to/document.txt")
            >>> print(f"Extracted {len(result['content'])} characters of text")
            >>> print(f"Document format: {result['format']}")
        """
        pass
    
    @abstractmethod
    def process_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process text content directly without an associated file.
        
        This method is used when you have text content but no file on disk,
        such as when processing content from memory, databases, or API responses.
        Unlike the process() method, this doesn't require a file path.
        
        Implementation should:
        1. Generate a document ID for the content
        2. Process the text content based on its format
        3. Extract metadata and entities from the content
        4. Return the standardized document structure
        
        Args:
            text: Text content to process
            options: Optional processing options, which may include:
                - format_hints: Hints about the content format
                - extraction_level: How deeply to analyze the content
                - ignore_sections: Sections to exclude from processing
            
        Returns:
            Dictionary with processed content in the standardized format:
            {
                "id": "generated-id",
                "content": "Processed text content",
                "format": "inferred-format",
                "metadata": { ... },
                "entities": [ ... ]
            }
            
        Example:
            >>> markdown_content = "# Document Title\n\nSome markdown content."
            >>> adapter = MarkdownAdapter()
            >>> result = adapter.process_text(markdown_content)
            >>> print(f"Processed {len(result['content'])} chars")
            >>> print(f"Detected title: {result['metadata'].get('title')}")
        """
        pass
    
    @abstractmethod
    def extract_entities(self, content: Union[str, Dict[str, Any]], options: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Extract entities from document content.
        
        Entities are structured elements within the document that have semantic meaning,
        such as people, organizations, locations, code symbols, citations, etc. The
        extraction process should identify these entities and their positions in the text.
        
        Args:
            content: Document content as string or dictionary (if dict, should contain a 'content' key)
            options: Optional configuration options for entity extraction
            
        Returns:
            List of extracted entities, each represented as a dictionary with:
            - type: Entity type (person, organization, location, code_symbol, etc.)
            - text: The entity text
            - start: Starting character position in the document
            - end: Ending character position in the document
            - metadata: Optional additional information about the entity
            
        Example:
            >>> content = "John Smith works at Acme Corp in New York."
            >>> entities = adapter.extract_entities(content)
            >>> print(f"Found {len(entities)} entities")
            >>> for entity in entities:
            ...     print(f"{entity['type']}: {entity['text']}")
        """
        pass
    
    @abstractmethod
    def extract_metadata(self, content: Union[str, Dict[str, Any]], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract metadata from document content.
        
        Metadata provides contextual information about the document such as title,
        author, creation date, modification date, keywords, summary, etc. This method
        should extract as much metadata as possible based on the document format.
        
        Args:
            content: Document content as string or dictionary (if dict, should contain a 'content' key)
            options: Optional configuration options for metadata extraction
            
        Returns:
            Dictionary of extracted metadata, which may include:
            - title: Document title
            - author: Document author(s)
            - created_at: Creation timestamp
            - modified_at: Last modification timestamp
            - keywords: List of document keywords
            - summary: Brief document summary
            - language: Document language code
            - format_specific_fields: Any format-specific metadata
            
        Example:
            >>> with open('document.md', 'r') as f:
            ...     content = f.read()
            >>> metadata = adapter.extract_metadata(content)
            >>> if 'title' in metadata:
            ...     print(f"Document title: {metadata['title']}")
            >>> if 'author' in metadata:
            ...     print(f"Author: {metadata['author']}")
        """
        pass
    
    def prepare_for_chunking(self, content: str) -> str:
        """
        Prepare content for chunking based on configuration settings.
        
        This method applies transformations specified in the chunking_preparation
        configuration to make the content more suitable for the chunking stage.
        
        Args:
            content: Document content to prepare
            
        Returns:
            Prepared content ready for chunking
        """
        if not self.chunking_config:
            return content
            
        prepared_content = content
        
        # Add section markers if configured
        if self.chunking_config.get('add_section_markers', False):
            prepared_content = self._add_section_markers(prepared_content)
            
        # Mark potential chunk boundaries if configured
        if self.chunking_config.get('mark_chunk_boundaries', False):
            prepared_content = self._mark_chunk_boundaries(prepared_content)
            
        return prepared_content
    
    def _add_section_markers(self, content: str) -> str:
        """
        Add section markers to content to improve chunking.
        
        Args:
            content: Document content
            
        Returns:
            Content with section markers
        """
        # Default implementation - can be overridden by format-specific adapters
        return content
    
    def _mark_chunk_boundaries(self, content: str) -> str:
        """
        Mark natural chunk boundaries in the content.
        
        Args:
            content: Document content
            
        Returns:
            Content with chunk boundary markers
        """
        # Default implementation - can be overridden by format-specific adapters
        return content
    
    # Note: convert_to_markdown and convert_to_text methods were removed
    # as they are outside the core functionality of the document processing pipeline
    # and represent unnecessary complexity for adapters
