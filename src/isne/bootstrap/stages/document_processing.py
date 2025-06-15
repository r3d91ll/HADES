"""
Document Processing Stage for ISNE Bootstrap Pipeline

Handles processing of various document formats into structured documents
that can be used for subsequent chunking and embedding stages.
"""

import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.components.docproc.factory import create_docproc_component
from src.types.components.contracts import DocumentProcessingInput
from .base import BaseBootstrapStage

logger = logging.getLogger(__name__)


@dataclass
class DocumentProcessingResult:
    """Result of document processing stage."""
    success: bool
    documents: List[Any]  # List of processed documents
    stats: Dict[str, Any]
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None


class DocumentProcessingStage(BaseBootstrapStage):
    """Document processing stage for bootstrap pipeline."""
    
    def __init__(self):
        """Initialize document processing stage."""
        super().__init__("document_processing")
        
    def execute(self, input_files: List[Path], config: Any) -> DocumentProcessingResult:
        """
        Execute document processing stage.
        
        Args:
            input_files: List of input file paths to process
            config: DocumentProcessingConfig object
            
        Returns:
            DocumentProcessingResult with processed documents and stats
        """
        logger.info(f"Starting document processing stage with {len(input_files)} files")
        
        try:
            # Create document processor
            processor = create_docproc_component(config.processor_type)
            
            documents = []
            stats = {
                "files_processed": 0,
                "documents_generated": 0,
                "total_content_chars": 0,
                "file_details": [],
                "supported_formats": config.supported_formats,
                "processor_type": config.processor_type
            }
            
            # Filter files by supported formats
            supported_files = []
            for file_path in input_files:
                if any(file_path.suffix.lower().endswith(f".{fmt}") or 
                      file_path.suffix.lower() == f".{fmt}" for fmt in config.supported_formats):
                    supported_files.append(file_path)
                else:
                    logger.debug(f"Skipping unsupported file format: {file_path}")
            
            logger.info(f"Processing {len(supported_files)} supported files "
                       f"(filtered from {len(input_files)} total files)")
            
            # Process each file
            for file_path in supported_files:
                try:
                    logger.info(f"Processing {file_path.name}...")
                    
                    # Create processing input
                    doc_input = DocumentProcessingInput(
                        file_path=str(file_path),
                        processing_options={
                            "extract_metadata": config.extract_metadata,
                            "extract_sections": config.extract_sections,
                            "extract_entities": config.extract_entities,
                            **config.options
                        },
                        metadata={
                            "bootstrap_file": file_path.name,
                            "stage": "document_processing",
                            "source_path": str(file_path)
                        }
                    )
                    
                    # Process document
                    output = processor.process(doc_input)
                    
                    # Collect results
                    file_docs = len(output.documents)
                    file_chars = sum(len(doc.content) for doc in output.documents)
                    
                    documents.extend(output.documents)
                    stats["files_processed"] += 1
                    stats["documents_generated"] += file_docs
                    stats["total_content_chars"] += file_chars
                    stats["file_details"].append({
                        "filename": file_path.name,
                        "file_path": str(file_path),
                        "documents_created": file_docs,
                        "characters": file_chars,
                        "format": file_path.suffix.lower()
                    })
                    
                    logger.info(f"  ✓ {file_path.name}: {file_docs} documents, {file_chars} characters")
                    
                except Exception as e:
                    error_msg = f"Failed to process {file_path.name}: {e}"
                    logger.error(error_msg)
                    stats["file_details"].append({
                        "filename": file_path.name,
                        "file_path": str(file_path),
                        "error": str(e),
                        "format": file_path.suffix.lower()
                    })
                    continue
            
            # Check if any documents were processed
            if not documents:
                error_msg = "No documents were successfully processed"
                logger.error(error_msg)
                return DocumentProcessingResult(
                    success=False,
                    documents=[],
                    stats=stats,
                    error_message=error_msg
                )
            
            # Success
            logger.info(f"Document processing completed: {len(documents)} documents from "
                       f"{stats['files_processed']} files ({stats['total_content_chars']} total characters)")
            
            return DocumentProcessingResult(
                success=True,
                documents=documents,
                stats=stats
            )
            
        except Exception as e:
            error_msg = f"Document processing stage failed: {e}"
            error_traceback = traceback.format_exc()
            logger.error(error_msg)
            logger.debug(error_traceback)
            
            return DocumentProcessingResult(
                success=False,
                documents=[],
                stats={},
                error_message=error_msg,
                error_traceback=error_traceback
            )
    
    def validate_inputs(self, input_files: List[Path], config: Any) -> List[str]:
        """
        Validate inputs for document processing stage.
        
        Args:
            input_files: List of input file paths
            config: DocumentProcessingConfig object
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        if not input_files:
            errors.append("No input files provided")
            return errors
        
        # Check if files exist
        missing_files = [f for f in input_files if not f.exists()]
        if missing_files:
            errors.append(f"Missing input files: {[str(f) for f in missing_files]}")
        
        # Check for supported formats
        supported_files = []
        for file_path in input_files:
            if file_path.exists() and any(
                file_path.suffix.lower().endswith(f".{fmt}") or 
                file_path.suffix.lower() == f".{fmt}" 
                for fmt in config.supported_formats
            ):
                supported_files.append(file_path)
        
        if not supported_files:
            errors.append(f"No files with supported formats found. "
                         f"Supported formats: {config.supported_formats}")
        
        # Validate processor type
        valid_processors = ["core", "docling"]  # Add more as needed
        if config.processor_type not in valid_processors:
            errors.append(f"Invalid processor type: {config.processor_type}. "
                         f"Valid options: {valid_processors}")
        
        return errors
    
    def get_expected_outputs(self) -> List[str]:
        """Get list of expected output keys."""
        return ["documents", "stats"]
    
    def estimate_duration(self, input_size: int) -> float:
        """
        Estimate stage duration based on input size.
        
        Args:
            input_size: Number of input files
            
        Returns:
            Estimated duration in seconds
        """
        # Rough estimate: 2 seconds per file + base overhead
        return max(30, input_size * 2)
    
    def get_resource_requirements(self, input_size: int) -> Dict[str, Any]:
        """
        Get estimated resource requirements.
        
        Args:
            input_size: Number of input files
            
        Returns:
            Dictionary with resource estimates
        """
        return {
            "memory_mb": max(500, input_size * 10),  # 10MB per file estimate
            "cpu_cores": 1,
            "disk_mb": input_size * 5,  # 5MB per file for temporary storage
            "network_required": False
        }