"""
Document processing stage for the unified HADES pipeline.

This module provides a concrete implementation of the PipelineStage
abstract base class for document processing operations, including
text extraction, preprocessing, and initial document structuring.

This is the consolidated version that integrates with the orchestration system.
"""

import logging
import os
from typing import Dict, Any, List, Tuple, Optional, Union
import uuid
import mimetypes
from pathlib import Path

from .base import PipelineStage, PipelineStageError
from src.orchestration.pipelines.schema import DocumentSchema, ValidationResult, ValidationIssue, ValidationSeverity

logger = logging.getLogger(__name__)

# Try to import document processing components
try:
    from src.components.docproc.factory import create_docproc_component
    from src.types.components.contracts import DocumentProcessingInput
    from src.types.components.protocols import DocumentProcessor
    DOCPROC_AVAILABLE = True
except ImportError:
    DOCPROC_AVAILABLE = False
    logger.warning("Document processing components not available")


class DocumentProcessorStage(PipelineStage):
    """Pipeline stage for document processing in the unified system.
    
    This stage handles the extraction of text from various document formats,
    initial preprocessing, and structuring the documents according to the
    DocumentSchema.
    
    Supports both sequential and parallel processing modes.
    """
    
    def __init__(
        self,
        name: str = "document_processor",
        config: Optional[Dict[str, Any]] = None,
        enable_parallel: bool = False,
        worker_pool: Any = None,
        queue_manager: Any = None
    ) -> None:
        """Initialize document processing stage.
        
        Args:
            name: Name of the stage
            config: Configuration dictionary with processing options
            enable_parallel: Whether to enable parallel processing
            worker_pool: Worker pool for parallel execution
            queue_manager: Queue manager for task distribution
        """
        super().__init__(name, config or {}, enable_parallel, worker_pool, queue_manager)
        
        # Initialize document processor with config options
        if DOCPROC_AVAILABLE:
            processor_config = self.config.get("processor_config", {})
            component_type = self.config.get("component_type", "core")  # default to core component
            self.document_processor: Optional[DocumentProcessor] = create_docproc_component(component_type, config=processor_config)
        else:
            self.document_processor: Optional[DocumentProcessor] = None
            self.logger.warning("Document processor not available, stage will be disabled")
        
        # Configure file type handling
        self.supported_formats = self.config.get("supported_formats", [
            "txt", "pdf", "docx", "md", "html", "csv", "json", "py", "yaml"
        ])
    
    def run(self, input_data: Union[str, List[str], Dict[str, Any]]) -> List[DocumentSchema]:
        """Process documents from file paths.
        
        This method extracts text from documents, applies preprocessing,
        and structures the documents according to the DocumentSchema.
        
        Args:
            input_data: Either a file path, a list of file paths, or a dictionary
                        with a "file_paths" key containing a list of file paths
            
        Returns:
            List of processed documents as DocumentSchema objects
            
        Raises:
            PipelineStageError: If document processing fails
        """
        if not DOCPROC_AVAILABLE or not self.document_processor:
            raise PipelineStageError(
                self.name,
                "Document processing module not available"
            )
        
        # Handle different input types
        file_paths = self._extract_file_paths(input_data)
        
        if not file_paths:
            raise PipelineStageError(
                self.name,
                "No valid file paths provided for document processing"
            )
        
        # Process each document
        processed_documents = []
        for file_path in file_paths:
            try:
                document = self._process_single_document(file_path)
                if document:
                    processed_documents.append(document)
                    
            except Exception as e:
                self.logger.error(f"Error processing document {file_path}: {str(e)}", exc_info=True)
                # Continue processing other documents
        
        self.logger.info(f"Processed {len(processed_documents)} documents successfully")
        return processed_documents
    
    def _process_single_document(self, file_path: str) -> Optional[DocumentSchema]:
        """Process a single document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            DocumentSchema object or None if processing failed
        """
        # Check if file exists and is supported
        path = Path(file_path)
        if not path.exists():
            self.logger.warning(f"File not found: {file_path}")
            return None
            
        file_extension = path.suffix.lower().lstrip('.')
        if file_extension not in self.supported_formats:
            self.logger.warning(
                f"Unsupported file format: {file_extension} for file: {file_path}"
            )
            return None
        
        # Process the document
        self.logger.info(f"Processing document: {file_path}")
        content_type = mimetypes.guess_type(file_path)[0] or "text/plain"
        
        # Extract document content using the document processor
        try:
            # Create processing input for the new component interface
            processing_input = DocumentProcessingInput(
                file_path=file_path,
                processing_options=self.config.get("processing_options", {})
            )
            
            # Process using the new component interface
            processing_output = self.document_processor.process(processing_input)
            
            # Extract the first document from the output
            if processing_output.documents:
                processed_doc = processing_output.documents[0]
                document_result = {
                    "content": processed_doc.content,
                    "format": processed_doc.format,
                    "entities": processed_doc.entities,
                    "sections": processed_doc.sections,
                    "errors": [processed_doc.error] if processed_doc.error else []
                }
            else:
                document_result = {"content": "", "format": file_extension, "entities": [], "sections": [], "errors": processing_output.errors}
            
            # Create document schema
            document = DocumentSchema(
                file_id=str(uuid.uuid4()),
                file_path=file_path,
                file_name=path.name,
                file_type=file_extension,
                content_type=content_type,
                metadata={
                    "size_bytes": path.stat().st_size,
                    "last_modified": path.stat().st_mtime,
                    "processor": self.name,
                    "content_length": len(document_result.get("content", "")),
                    "format_detected": document_result.get("format", file_extension),
                    "content": document_result.get("content", ""),
                    "entities": document_result.get("entities", []),
                    "sections": document_result.get("sections", [])
                }
            )
            
            # Validate the document
            validation_result = self._validate_document(document, document_result)
            document.validation = validation_result
            
            # Only return valid documents or those with non-critical issues
            if not validation_result.has_errors:
                return document
            else:
                self.logger.warning(
                    f"Document validation failed for {file_path}: "
                    f"{len(validation_result.get_issues_by_severity(ValidationSeverity.ERROR))} errors"
                )
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to process document {file_path}: {str(e)}")
            return None
    
    def validate(self, data: Union[str, List[str], Dict[str, Any], List[DocumentSchema]]) -> Tuple[bool, List[str]]:
        """Validate input or output data for this stage.
        
        Args:
            data: Data to validate (either input file paths or output DocumentSchema objects)
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        # If the data is a list of DocumentSchema objects (output validation)
        if isinstance(data, list) and all(isinstance(item, DocumentSchema) for item in data):
            # Check if any documents were processed
            if len(data) == 0:
                return False, ["No documents were processed successfully"]
            return True, []
        
        # Otherwise, validate input data
        try:
            file_paths = self._extract_file_paths(data)
            if not file_paths:
                return False, ["No valid file paths provided"]
            
            # Check if files exist and are supported
            errors = []
            for file_path in file_paths:
                path = Path(file_path)
                if not path.exists():
                    errors.append(f"File not found: {file_path}")
                else:
                    file_extension = path.suffix.lower().lstrip('.')
                    if file_extension not in self.supported_formats:
                        errors.append(f"Unsupported file format: {file_extension} for file: {file_path}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Invalid input data format: {str(e)}"]
    
    def _extract_file_paths(self, input_data: Union[str, List[str], Dict[str, Any]]) -> List[str]:
        """Extract file paths from various input formats.
        
        Args:
            input_data: Input data in one of the supported formats
            
        Returns:
            List of file paths
        """
        if isinstance(input_data, str):
            return [input_data]
        elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
            return input_data
        elif isinstance(input_data, dict) and "file_paths" in input_data:
            file_paths = input_data["file_paths"]
            if isinstance(file_paths, list) and all(isinstance(item, str) for item in file_paths):
                return file_paths
            elif isinstance(file_paths, str):
                return [file_paths]
        
        return []
    
    def _validate_document(self, document: DocumentSchema, content: Dict[str, Any]) -> ValidationResult:
        """Validate a processed document.
        
        Args:
            document: Document schema object
            content: Extracted document content
            
        Returns:
            ValidationResult object
        """
        issues = []
        
        # Check for empty content
        if not content or not content.get("content"):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Document has no extractable text content",
                location=document.file_path
            ))
        
        # Check for very small content (possibly extraction failure)
        content_text = content.get("content", "")
        if content_text and len(content_text) < 50:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Document has very little text content, possible extraction issue",
                location=document.file_path,
                context={"content_length": len(content_text)}
            ))
        
        # Check for processing errors
        if content.get("errors"):
            for error in content["errors"]:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Processing warning: {error}",
                    location=document.file_path
                ))
        
        # Return validation result
        return ValidationResult(
            is_valid=not any(issue.severity == ValidationSeverity.ERROR for issue in issues),
            issues=issues,
            stage_name=self.name
        )