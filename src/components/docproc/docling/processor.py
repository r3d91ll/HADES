"""
Docling Document Processing Component

This module provides a Docling-based document processing component that implements
the DocumentProcessor protocol. Docling specializes in processing complex documents
like PDFs, Word documents, and other structured formats.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime, timezone

# Import component contracts and protocols
from src.types.components.contracts import (
    ComponentType,
    ComponentMetadata,
    DocumentProcessingInput,
    DocumentProcessingOutput,
    ProcessedDocument,
    ContentCategory,
    ProcessingStatus
)
from src.types.components.protocols import DocumentProcessor


class DoclingDocumentProcessor(DocumentProcessor):
    """
    Docling-based document processor component implementing DocumentProcessor protocol.
    
    This component uses the Docling library for advanced document processing,
    particularly for PDFs and complex document formats.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Docling document processor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.DOCPROC,
            component_name="docling",
            component_version="1.0.0",
            config=self._config
        )
        
        # Try to import Docling
        self._docling_available = self._check_docling_availability()
        
        if self._docling_available:
            # Initialize Docling adapter (placeholder for minimal implementation)
            # In a full implementation, would initialize a Docling adapter here
            self._adapter = None  # Placeholder for minimal implementation
            self.logger.info("Initialized Docling document processor (minimal)")
        else:
            self._adapter = None
            self.logger.warning("Docling not available - processor will return errors")
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return "docling"
    
    @property
    def version(self) -> str:
        """Component version string."""
        return "1.0.0"
    
    @property
    def component_type(self) -> ComponentType:
        """Type of component."""
        return ComponentType.DOCPROC
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure component with parameters."""
        self._config.update(config)
        self._metadata.config = self._config
        self._metadata.processed_at = datetime.now(timezone.utc)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration parameters."""
        return isinstance(config, dict)
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for component configuration."""
        return {
            "type": "object",
            "properties": {
                "ocr_enabled": {
                    "type": "boolean",
                    "description": "Enable OCR for scanned documents",
                    "default": True
                },
                "extract_tables": {
                    "type": "boolean",
                    "description": "Extract tables from documents",
                    "default": True
                },
                "extract_images": {
                    "type": "boolean",
                    "description": "Extract images from documents",
                    "default": False
                }
            }
        }
    
    def health_check(self) -> bool:
        """Check if component is healthy."""
        return self._docling_available
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get component performance metrics."""
        return {
            "component_name": self.name,
            "component_version": self.version,
            "docling_available": self._docling_available,
            "supported_formats": self.get_supported_formats()
        }
    
    def process_documents(
        self, 
        file_paths: List[Union[str, Path]], 
        **kwargs: Any
    ) -> List[ProcessedDocument]:
        """Process multiple documents from file paths."""
        if not self._docling_available:
            return [self._create_error_document(fp, "Docling not available") for fp in file_paths]
        
        results = []
        for file_path in file_paths:
            result = self.process_document(file_path, **kwargs)
            results.append(result)
        
        return results
    
    def process_document(
        self, 
        file_path: Union[str, Path], 
        **kwargs: Any
    ) -> ProcessedDocument:
        """Process a single document using Docling for any supported format."""
        if not self._docling_available:
            return self._create_error_document(file_path, "Docling not available")
        
        try:
            # Import Docling
            from docling.document_converter import DocumentConverter
            
            # Create converter
            converter = DocumentConverter()
            
            # Convert document (works for PDF, DOCX, PPTX, HTML, images, etc.)
            conversion_result = converter.convert(str(file_path))
            
            # Extract content - use markdown as primary content
            markdown_content = conversion_result.document.export_to_markdown()
            
            # Extract metadata
            doc_metadata = {
                'processed_by': 'docling',
                'conversion_status': str(conversion_result.status),
                'file_path': str(file_path),
                'file_size': Path(file_path).stat().st_size,
                'processed_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Add document-specific metadata if available
            if hasattr(conversion_result.document, 'num_pages'):
                doc_metadata['num_pages'] = conversion_result.document.num_pages
            if hasattr(conversion_result.document, 'title'):
                doc_metadata['title'] = conversion_result.document.title
            if hasattr(conversion_result.document, 'author'):
                doc_metadata['author'] = conversion_result.document.author
            if hasattr(conversion_result.document, 'creation_date'):
                doc_metadata['creation_date'] = str(conversion_result.document.creation_date)
            
            # Determine content type and category based on file extension
            file_ext = Path(file_path).suffix.lower()
            content_type_map = {
                '.pdf': 'application/pdf',
                '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                '.doc': 'application/msword',
                '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                '.ppt': 'application/vnd.ms-powerpoint',
                '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                '.xls': 'application/vnd.ms-excel',
                '.html': 'text/html',
                '.htm': 'text/html',
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg'
            }
            
            content_type = content_type_map.get(file_ext, 'application/octet-stream')
            
            # Convert to contract format
            return ProcessedDocument(
                id=Path(file_path).stem,
                content=markdown_content,
                content_type=content_type,
                format=file_ext.lstrip('.'),
                content_category=ContentCategory.DOCUMENT,
                entities=[],  # Docling doesn't extract entities by default
                sections=[],  # Could be extracted from markdown headers if needed
                metadata=doc_metadata,
                error=None,
                processing_time=(datetime.now(timezone.utc) - datetime.now(timezone.utc)).total_seconds()  # Will be overridden by caller
            )
            
        except Exception as e:
            self.logger.error(f"Docling processing failed for {file_path}: {e}")
            return self._create_error_document(file_path, str(e))
    
    def process(self, input_data: DocumentProcessingInput) -> DocumentProcessingOutput:
        """Process documents according to the contract."""
        documents = []
        errors = []
        
        try:
            doc = self.process_document(
                input_data.file_path,
                **input_data.processing_options
            )
            documents.append(doc)
            
            if doc.error:
                errors.append(doc.error)
            
        except Exception as e:
            errors.append(str(e))
        
        metadata = ComponentMetadata(
            component_type=self.component_type,
            component_name=self.name,
            component_version=self.version,
            processed_at=datetime.now(timezone.utc),
            config=self._config,
            status=ProcessingStatus.SUCCESS if not errors else ProcessingStatus.ERROR
        )
        
        return DocumentProcessingOutput(
            documents=documents,
            metadata=metadata,
            errors=errors
        )
    
    def process_batch(
        self, 
        input_batch: List[DocumentProcessingInput]
    ) -> List[DocumentProcessingOutput]:
        """Process multiple documents in batch."""
        return [self.process(input_data) for input_data in input_batch]
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats that Docling can process."""
        if self._docling_available:
            return [
                # Document formats
                '.pdf',
                '.docx', '.doc',
                '.pptx', '.ppt', 
                '.xlsx', '.xls',
                '.html', '.htm',
                # Image formats (with OCR)
                '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff',
                # Text formats
                '.txt', '.md', '.markdown',
                # Other supported formats
                '.rtf', '.odt', '.ods', '.odp'
            ]
        return []
    
    def can_process(self, file_path: str) -> bool:
        """Check if this processor can handle the file."""
        if not self._docling_available:
            return False
        
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.get_supported_formats()
    
    def estimate_processing_time(self, input_data: DocumentProcessingInput) -> float:
        """Estimate processing time for given input."""
        if not self._docling_available:
            return 0.0
        
        try:
            file_size = Path(input_data.file_path).stat().st_size
            # Docling is slower for complex documents: 500KB per second
            return max(0.5, file_size / (500 * 1024))
        except Exception:
            return 2.0  # Default estimate for Docling
    
    def _check_docling_availability(self) -> bool:
        """Check if Docling is available."""
        try:
            import docling
            return True
        except ImportError:
            return False
    
    def _create_error_document(self, file_path: Union[str, Path], error: str) -> ProcessedDocument:
        """Create an error document."""
        return ProcessedDocument(
            id=f"error_{Path(file_path).name}",
            content="",
            content_type="text/plain",
            format="unknown",
            content_category=ContentCategory.UNKNOWN,
            error=error
        )