"""Document processing type definitions.

This module provides type definitions for document processing functions,
including options, results, and statistics for both single and batch processing.
"""

from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, TypedDict, Union
from pathlib import Path
from datetime import datetime

from src.types.docproc.adapter import ProcessedDocument


# Callback type definitions
SuccessCallback = Callable[[Dict[str, Any], Union[str, Path]], None]
"""Callback function type for successful document processing."""

ErrorCallback = Callable[[Union[str, Path], Exception], None]
"""Callback function type for document processing errors."""


class ProcessingOptions(TypedDict, total=False):
    """Options for document processing functions."""
    
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
    
    format_specific_options: Dict[str, Any]
    """Format-specific processing options."""


class FormatDetectionResult(TypedDict):
    """Result of format detection."""
    
    format: str
    """Detected format of the document."""
    
    confidence: float
    """Confidence level of the format detection (0.0-1.0)."""
    
    content_type: Literal["text", "code", "binary", "unknown"]
    """Content type category."""
    
    detected_by: Literal["extension", "content", "both", "override"]
    """How the format was detected."""


class ProcessingResult(TypedDict, total=False):
    """Result of document processing with status information."""
    
    success: bool
    """Whether processing was successful."""
    
    document: Optional[ProcessedDocument]
    """Processed document if successful."""
    
    error: Optional[str]
    """Error message if processing failed."""
    
    error_type: Optional[str]
    """Type of error if processing failed."""
    
    processing_time: float
    """Time taken to process the document in seconds."""
    
    format_detection: Optional[FormatDetectionResult]
    """Format detection information if available."""


class ProcessingStats(TypedDict):
    """Statistics for document processing."""
    
    total: int
    """Total number of documents processed."""
    
    successful: int
    """Number of successfully processed documents."""
    
    failed: int
    """Number of failed documents."""
    
    formats: Dict[str, int]
    """Count of documents by format type."""
    
    errors: Dict[str, int]
    """Count of errors by error type."""
    
    processing_time: Dict[str, float]
    """Total and average processing time."""
    
    processed_at: datetime
    """Timestamp when processing completed."""


class BatchProcessingResult(TypedDict):
    """Result of batch document processing."""
    
    stats: ProcessingStats
    """Processing statistics."""
    
    documents: List[str]
    """List of processed document IDs."""
    
    successful_documents: List[Dict[str, Any]]
    """List of successfully processed documents."""
    
    failed_documents: List[Dict[str, Any]]
    """List of documents that failed processing with error information."""
    
    output_files: Optional[List[str]]
    """List of output files if saving was enabled."""
