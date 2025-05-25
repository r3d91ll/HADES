"""
Document processing stage for the HADES-PathRAG pipeline.

This module provides a concrete implementation of the PipelineStage
abstract base class for document processing operations, including
text extraction, preprocessing, and initial document structuring.
"""

import logging
import os
from typing import Dict, Any, List, Tuple, Optional, Union
import uuid
import mimetypes
from pathlib import Path

from src.pipeline.stages import PipelineStage, PipelineStageError
from src.pipeline.schema import DocumentSchema, ValidationResult, ValidationIssue, ValidationSeverity
from src.document_processing.processor import DocumentProcessor

logger = logging.getLogger(__name__)


class DocumentProcessingStage(PipelineStage):
    """Pipeline stage for document processing.
    
    This stage handles the extraction of text from various document formats,
    initial preprocessing, and structuring the documents according to the
    DocumentSchema.
    """
    
    def __init__(
        self,
        name: str = "document_processor",
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize document processing stage.
        
        Args:
            name: Name of the stage
            config: Configuration dictionary with processing options
        """
        super().__init__(name, config or {})
        
        # Initialize document processor with config options
        processor_config = self.config.get("processor_config", {})
        self.document_processor = DocumentProcessor(**processor_config)
        
        # Configure file type handling
        self.supported_formats = self.config.get("supported_formats", [
            "txt", "pdf", "docx", "md", "html", "csv", "json"
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
                # Check if file exists and is supported
                path = Path(file_path)
                if not path.exists():
                    self.logger.warning(f"File not found: {file_path}")
                    continue
                    
                file_extension = path.suffix.lower().lstrip('.')
                if file_extension not in self.supported_formats:
                    self.logger.warning(
                        f"Unsupported file format: {file_extension} for file: {file_path}"
                    )
                    continue
                
                # Process the document
                self.logger.info(f"Processing document: {file_path}")
                content_type = mimetypes.guess_type(file_path)[0] or "text/plain"
                
                # Extract document content using the document processor
                document_content = self.document_processor.process_document(file_path)
                
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
                        "processor": self.name
                    }
                )
                
                # Validate the document
                validation_result = self._validate_document(document, document_content)
                document.validation = validation_result
                
                # Only add valid documents or those with non-critical issues
                if not validation_result.has_errors:
                    processed_documents.append(document)
                else:
                    self.logger.warning(
                        f"Document validation failed for {file_path}: "
                        f"{len(validation_result.get_issues_by_severity(ValidationSeverity.ERROR))} errors"
                    )
                
            except Exception as e:
                self.logger.error(f"Error processing document {file_path}: {str(e)}", exc_info=True)
                # Continue processing other documents
        
        self.logger.info(f"Processed {len(processed_documents)} documents successfully")
        return processed_documents
    
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
        if not content or not content.get("text"):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Document has no extractable text content",
                location=document.file_path
            ))
        
        # Check for very small content (possibly extraction failure)
        if content and len(content.get("text", "")) < 50:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Document has very little text content, possible extraction issue",
                location=document.file_path,
                context={"content_length": len(content.get("text", ""))}
            ))
        
        # Return validation result
        return ValidationResult(
            is_valid=not any(issue.severity == ValidationSeverity.ERROR for issue in issues),
            issues=issues,
            stage_name=self.name
        )
