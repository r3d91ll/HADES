"""
Core functionality for document processing.

This module provides the primary interface for processing documents of various formats
and transforming them into standardized JSON objects. These objects can be directly
passed to the next stage in the processing pipeline (e.g., chunking, embedding) or
optionally saved to disk for inspection or archiving.

The focus is on in-memory processing for optimal performance in pipeline scenarios.

Integration with HADES-PathRAG Ingestion Pipeline:
-------------------------------------------------
The document processing module serves as the first stage in the HADES-PathRAG 
ingestion pipeline:

1. Document Processing (docproc): Convert raw documents to standardized format
   ↓
2. Chunking (chunking): Split documents into appropriate chunks
   ↓
3. Embedding (embeddings): Generate embeddings for chunks
   ↓
4. Storage (storage): Store chunks, embeddings, and relationships in ArangoDB
   ↓
5. ISNE Graph Processing: Create graph relationships and ISNE embeddings

This module's core functions are designed to work seamlessly with the chunking
module by providing documents in a standardized format that can be directly
processed by chunkers like CPU Chunker and Chonky Chunker.

Key Functions:
-------------
- process_document: Process a document file into standardized format
- process_text: Process text content directly with a specified format
- process_documents_batch: Process multiple documents in batch mode
- detect_format: Detect document format from file extension and content
- save_processed_document: Save processed document to disk (optional)

Usage Examples:
--------------
```python
# Process a single document
from src.docproc.core import process_document
document = process_document("/path/to/document.pdf")

# Process text content directly
from src.docproc.core import process_text
document = process_text("# Markdown Content", format_type="markdown")

# Process multiple documents
from src.docproc.core import process_documents_batch
results = process_documents_batch([
    "/path/to/document1.pdf",
    "/path/to/document2.md",
    "/path/to/document3.txt"
])
print(f"Processed {results['successful']} documents")
```

Integration Example:
------------------
```python
# Complete pipeline integration
from src.docproc.core import process_documents_batch
from src.chunking import chunk_documents
from src.embedding import embed_chunks
from src.storage import store_documents

# Process documents
processing_results = process_documents_batch(file_paths)
documents = processing_results['documents']

# Continue with pipeline
chunks = chunk_documents(documents)
embedded_chunks = embed_chunks(chunks)
store_documents(embedded_chunks)
```
"""

import logging
import json
import time
import uuid
import concurrent.futures

from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, cast, TypeVar, Tuple
from datetime import datetime
from pydantic import ValidationError

# Import utility functions
from src.utils.file_utils import get_file_info, is_binary_file
from src.utils.format_detection import (
    detect_format_from_content, 
    detect_format_from_path, 
    get_content_category
)

# Import docproc implementation modules
from src.docproc.utils.metadata_extractor import extract_metadata
from src.docproc.adapters.registry import get_adapter_for_format
from src.docproc.adapters.adapter_selector import select_adapter_for_document
from src.docproc.schemas.utils import validate_document, add_validation_to_adapter, document_to_processed_document
from src.docproc.schemas.base import BaseDocument

# Import centralized type definitions
from src.types.docproc import (
    # Processing types
    ProcessingOptions,
    ProcessingResult,
    BatchProcessingResult,
    ProcessingStats,
    FormatDetectionResult,
    SuccessCallback,
    ErrorCallback,
    
    # Document types
    DocumentDict,
    
    # Adapter types
    AdapterOptions,
    ProcessedDocument
)

# Setup logging
logger = logging.getLogger(__name__)


def process_document(file_path: Union[str, Path], options: Optional[ProcessingOptions] = None) -> ProcessedDocument:
    """
    Process a document file, converting it to a standardized format.
    
    This function is the primary entry point for document processing in the HADES-PathRAG
    ingestion pipeline. It detects the document format, selects the appropriate adapter,
    extracts content and metadata, and returns a standardized document structure that can
    be passed to downstream components (chunking, embedding, etc.).
    
    The processing flow involves:
    1. Validating the file exists and is accessible
    2. Detecting the document format from file extension and content
    3. Selecting and instantiating the appropriate format adapter
    4. Processing the document with the adapter
    5. Validating the output (if validation is enabled)
    6. Returning the standardized document structure
    
    Args:
        file_path: Path to the document file. Can be either a string path or a Path object.
                  The file must exist and be readable.
        options: Optional processing options dictionary that controls behavior. Common options include:
                - format_override: Override automatic format detection with specified format
                - extract_metadata: Whether to extract metadata (default: True)
                - extract_entities: Whether to extract entities (default: True)
                - validation_level: How strictly to validate the output ("strict", "warn", "none")
                - max_content_length: Limit the content length to this many characters
                - format_specific_options: Format-specific processing options
    
    Returns:
        Dictionary with processed content and metadata in the standardized format:
        {
            "id": "unique-document-id",
            "content": "Extracted text content of the document",
            "path": "/absolute/path/to/source/file.ext",
            "format": "detected-format",
            "content_category": "code" or "text",
            "metadata": {
                "filename": "file_name.ext",
                "file_type": "detected-format",
                "content_category": "code" or "text",
                "last_modified": "2025-05-15T10:00:00Z",
                "size": 1024,
                "title": "Document Title",
                "author": "Document Author",
                "created_at": "2025-05-15T10:00:00Z",
                ...
            },
            "entities": [
                {"type": "person", "text": "John Smith", "start": 120, "end": 130},
                ...
            ]
        }
        
    Raises:
        FileNotFoundError: If the file does not exist or cannot be accessed
        ValidationError: If the processed document fails validation
        ValueError: If the document format is not supported
        TypeError: If arguments are of incorrect type
        
    Examples:
        # Basic document processing
        result = process_document("/path/to/document.pdf")
        print(f"Document ID: {result['id']}")
        print(f"Content length: {len(result['content'])} chars")
        
        # With format override and options
        from pathlib import Path
        doc_path = Path("/path/to/code.txt")
        result = process_document(
            doc_path,
            options={
                "format_override": "python",  # Force Python processing for .txt file
                "extract_docstrings": True,   # Python-specific option
                "max_content_length": 100000  # Limit content length
            }
        )
        
        # Error handling
        try:
            result = process_document("/path/to/document.xyz")
        except FileNotFoundError:
            print("File not found!")
        except ValueError as e:
            print(f"Unsupported format: {e}")
    """
    options = options or {}
    path_obj = Path(file_path) if isinstance(file_path, str) else file_path
    
    # Check if file exists first
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path_obj}")
    
    # Detect document format and content category
    format_type = detect_format_from_path(path_obj)
    content_category = get_content_category(format_type)
    
    # Log the detected format and category for debugging
    logger.debug(f"Detected format {format_type} (category: {content_category}) for file {path_obj}")
    
    # Select the appropriate adapter based on content category
    adapter = select_adapter_for_document(format_type)
    
    # Log content category and adapter selection
    logger.info(f"Processing {path_obj} as {content_category} content with {adapter.__class__.__name__}")
    
    # Process the document
    processed_doc = adapter.process(path_obj, options)
    
    # Add content category to document structure
    processed_doc["content_category"] = content_category
    
    # Enrich with metadata using our heuristic extraction
    content = processed_doc.get("content", "")
    metadata = extract_metadata(content, str(path_obj), format_type)
    
    # Add file type and content category to metadata
    metadata["file_type"] = format_type
    metadata["content_category"] = content_category
    
    # Merge extracted metadata with any existing metadata
    existing_metadata = processed_doc.get("metadata", {})
    for key, value in metadata.items():
        if key not in existing_metadata or existing_metadata[key] == "UNK":
            existing_metadata[key] = value
    
    # Update the processed document with the enriched metadata
    processed_doc["metadata"] = existing_metadata
    
    # Validate the processed document using Pydantic
    try:
        validated_doc = validate_document(processed_doc)
        # Convert to ProcessedDocument using our helper function
        return document_to_processed_document(validated_doc)
    except ValidationError as e:
        logger.warning(f"Document validation failed: {e}")
        # Add validation error to the document
        processed_doc["_validation_error"] = str(e)
        return processed_doc


def process_text(text: str, format_type: str = "text", format_or_options: Optional[Union[str, ProcessingOptions]] = None, options: Optional[ProcessingOptions] = None) -> ProcessedDocument:
    """
    Process text content directly, assuming a specific format.
    
    This function is used when you already have the document content in memory
    and want to process it without reading from a file. It's useful for processing:
    - Content from databases
    - Content from API responses
    - Content from user input
    - Generated content
    
    The function selects the appropriate adapter for the specified format and
    processes the text content to produce a standardized document structure.
    
    Args:
        text: Text content to process. This is the raw content string.
        format_type: Format of the text content. Common formats include:
                    "text" (plain text), "markdown", "html", "json", "python", etc.
                    The specified format determines which adapter will be used.
        format_or_options: Either a format string or options dictionary. This parameter
                          exists for backward compatibility and flexibility. If it's a
                          string, it's treated as a format override. If it's a dictionary,
                          it's merged with the options parameter.
        options: Optional processing options dictionary that controls behavior.
                Common options include:
                - extract_metadata: Whether to extract metadata (default: True)
                - extract_entities: Whether to extract entities (default: True)
                - validation_level: Validation strictness ("strict", "warn", "none")
                - max_content_length: Limit content length
                - format_specific_options: Options for specific formats
    
    Returns:
        Dictionary with processed content and metadata in the standardized format,
        matching the structure returned by process_document() -> Dict[str, Any]:
        {
            "id": "generated-id",
            "content": "Processed text content",
            "format": "specified-format",
            "metadata": { ... },
            "entities": [ ... ]
        }
        
    Raises:
        ValidationError: If the processed document fails validation
        ValueError: If the format is not supported or processing fails
        TypeError: If arguments are of incorrect type
        
    Examples:
        # Process markdown content
        markdown = "# Document Title\n\nThis is a paragraph with **bold** text."
        result = process_text(markdown, format_type="markdown")
        print(f"Extracted title: {result['metadata'].get('title')}")
        
        # Process HTML with options
        html = "<html><head><title>Example</title></head><body><p>Content</p></body></html>"
        result = process_text(
            html,
            format_type="html",
            options={
                "extract_links": True,
                "clean_html": True,
                "strip_scripts": True
            }
        )
        
        # Process Python code
        code = "def hello() -> Any: print('Hello, world!')"
        result = process_text(
            code, 
            format_type="python",
            options={"extract_docstrings": True, "include_imports": True}
        )
        
        # Using format_or_options for backward compatibility
        result = process_text(text, "json", {"pretty_print": True})
    """
    # Handle flexible parameter usage
    if isinstance(format_or_options, dict):
        # If the third parameter is a dict, it's options
        actual_options = format_or_options
        actual_format = format_type
    elif isinstance(format_or_options, str):
        # If the third parameter is a string, it's the format
        actual_format = format_or_options
        actual_options = options or {}
    else:
        # Default case
        actual_format = format_type
        actual_options = options or {}
    
    # Get the adapter for the specified format
    adapter = get_adapter_for_format(actual_format)
    
    # Detect the content category for the format
    content_category = get_content_category(actual_format)
    logger.debug(f"Using format {actual_format} (category: {content_category}) for text processing")
    
    # Process the text with the appropriate adapter
    processed_doc = adapter.process_text(text, options=actual_options)
    
    # Add content category to document structure
    processed_doc["content_category"] = content_category
    
    # Add file type and content category to metadata
    if "metadata" not in processed_doc:
        processed_doc["metadata"] = {}
    processed_doc["metadata"]["file_type"] = actual_format
    processed_doc["metadata"]["content_category"] = content_category
    
    # Enrich with metadata using our heuristic extraction
    content = processed_doc.get("content", "")
    source = actual_options.get("source", "direct_text")
    metadata = extract_metadata(content, str(source), actual_format)
    
    # Merge extracted metadata with any existing metadata
    existing_metadata = processed_doc.get("metadata", {})
    for key, value in metadata.items():
        if key not in existing_metadata or existing_metadata[key] == "UNK":
            existing_metadata[key] = value
    
    # Update the processed document with the enriched metadata
    processed_doc["metadata"] = existing_metadata
    
    # Validate the processed document using Pydantic
    try:
        validated_doc = validate_document(processed_doc)
        # Convert to ProcessedDocument using our helper function
        return document_to_processed_document(validated_doc)
    except ValidationError as e:
        logger.warning(f"Document validation failed: {e}")
        # Add validation error to the document
        processed_doc["_validation_error"] = str(e)
        return processed_doc


def get_format_for_document(file_path: Union[str, Path]) -> FormatDetectionResult:
    """
    Get the format for a document file.
    
    Args:
        file_path: Path to the document file
    
    Returns:
        FormatDetectionResult with detected format information
    
    Raises:
        FileNotFoundError: If the file does not exist
    """
    path_obj = Path(file_path) if isinstance(file_path, str) else file_path
    
    # Check if file exists
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path_obj}")
    
    # Get the format string from the path
    format_str = detect_format_from_path(path_obj)
    content_type = get_content_category(format_str)
    
    # Create and return a FormatDetectionResult
    return {
        "format": format_str,
        "confidence": 0.9,  # High confidence when detecting from file extension
        "content_type": content_type,
        "detected_by": "extension"
    }


# Renamed to avoid conflict with imported function
def detect_file_format(file_path: Union[str, Path]) -> FormatDetectionResult:
    """
    Detect the format of a document file.
    
    Args:
        file_path: Path to the document file
    
    Returns:
        FormatDetectionResult with detected format information
    """
    path_obj = Path(file_path) if isinstance(file_path, str) else file_path
    
    # Get the format string from the path
    format_str = detect_format_from_path(path_obj)
    content_type = get_content_category(format_str)
    
    # Create and return a FormatDetectionResult
    return {
        "format": format_str,
        "confidence": 0.8,  # Slightly lower confidence than get_format_for_document
        "content_type": content_type,
        "detected_by": "extension"
    }


def save_processed_document(document: ProcessedDocument, output_path: Union[str, Path]) -> Path:
    """
    Save a processed document JSON to disk.
    
    This is an optional utility for saving documents to disk when needed.
    The core pipeline will typically pass documents between stages in memory.
    
    Args:
        document: Processed document JSON to save
        output_path: Path where the JSON file should be saved
        
    Returns:
        Path to the saved JSON file
    """
    path_obj = Path(output_path) if isinstance(output_path, str) else output_path
    
    # Ensure parent directory exists
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate the document before saving if it hasn't been validated already
    if not document.get("_validated", False):
        try:
            # Attempt to validate the document
            validated_doc = validate_document(document)
            # Convert to ProcessedDocument using our helper function
            document = document_to_processed_document(validated_doc)
            document["_validated"] = True
        except ValidationError as e:
            logger.warning(f"Document validation failed before saving: {e}")
            # Add validation error to the document
            document["_validation_error"] = str(e)
    
    # Write JSON to file
    with open(path_obj, 'w', encoding='utf-8') as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
    
    return path_obj


def process_documents_batch(
    file_paths: List[Union[str, Path]], 
    options: Optional[ProcessingOptions] = None,
    output_dir: Optional[Union[str, Path]] = None,
    on_success: Optional[SuccessCallback] = None,
    on_error: Optional[ErrorCallback] = None,
    validate: bool = True
) -> BatchProcessingResult:
    """
    Process a batch of documents in parallel and return statistics.
    
    This function is a crucial component for large-scale document ingestion in the 
    HADES-PathRAG pipeline. It processes multiple documents efficiently, handling 
    errors gracefully to ensure the pipeline continues even if some documents fail.
    The function is designed to work with the ISNE (Inductive Shallow Node Embedding) 
    pipeline that builds the knowledge graph from processed documents.
    
    Key features:
    - Error isolation: Failure in one document doesn't stop processing of others
    - Optional disk saving for inspection or checkpointing
    - Callback hooks for custom processing on success/failure
    - Comprehensive statistics on processing results
    - Type safety throughout the processing flow
    
    Args:
        file_paths: List of paths to process. Can be a mix of different document types.
                   Each path should be either a string or Path object pointing to an
                   existing file on disk.
        options: Processing options to pass to the adapter. These are the same options
                as for process_document() and will be applied to all documents. Format-specific
                options will only affect documents of the matching format.
        output_dir: Optional directory to save outputs (None = no saving). If provided,
                   processed documents will be saved as JSON files in this directory with
                   filenames derived from document IDs.
        on_success: Optional callback for successful processing. This function will be called
                  for each successfully processed document with the document dictionary and
                  its original path. Useful for custom handling or further processing.
                  Signature: callback(document_dict, original_path)
        on_error: Optional callback for processing errors. This function will be called for
                each document that fails processing with the path and exception object.
                Signature: callback(file_path, exception)
        validate: Whether to validate documents with Pydantic schemas (default: True).
                 This ensures all documents conform to the expected structure before
                 they are passed downstream.
        
    Returns:
        Dictionary with processing statistics containing:
        - total: Total number of documents processed
        - successful: Number of successfully processed documents
        - failed: Number of failed documents
        - formats: Count of documents by format type
        - errors: Count of errors by error type
        - processing_time: Total and average processing time
        - documents: List of processed document IDs
    
    Examples:
        # Basic batch processing
        stats = process_documents_batch([
            "/path/to/doc1.pdf",
            "/path/to/doc2.md",
            "/path/to/code.py"
        ])
        print(f"Processed {stats['successful']} documents successfully")
        print(f"Failed to process {stats['failed']} documents")
        
        # With output directory for saving results
        from pathlib import Path
        output_path = Path("/path/to/output")
        output_path.mkdir(exist_ok=True)
        
        stats = process_documents_batch(
            [Path("/path/to/document.pdf")],
            output_dir=output_path,
            options={"extract_images": True}
        )
        
        # With custom success/error callbacks
        def on_doc_success(doc: Any, path: Any) -> Dict[str, Any]:
            print(f"Successfully processed {path} with ID {doc['id']}")
            
        def on_doc_error(path: Any, error: Any) -> Any:
            print(f"Failed to process {path}: {error}")
            
        stats = process_documents_batch(
            document_paths,
            on_success=on_doc_success,
            on_error=on_doc_error
        )
        
        # Integration with HADES-PathRAG ingestion pipeline
        from src.chunking import chunk_documents
        from src.embeddings import embed_chunks
        from src.storage import store_documents
        
        # Process documents
        processed_docs = process_documents_batch(file_paths)
        
        # Continue with the pipeline
        chunks = chunk_documents(processed_docs['documents'])
        embeddings = embed_chunks(chunks)
        store_documents(embeddings)
    """
    # Track document IDs and successful/failed documents
    document_ids: List[str] = []
    successful_docs: List[Dict[str, Any]] = []
    failed_docs: List[Dict[str, Any]] = []
    output_files: List[str] = []
    
    # Initialize processing stats
    stats: Dict[str, Any] = {
        "total": len(file_paths),
        "successful": 0,
        "failed": 0,
        "formats": {},
        "errors": {},
        "processing_time": {
            "total": 0.0,
            "average": 0.0
        },
        "processed_at": datetime.now()
    }
    
    # Create output directory if needed
    if output_dir:
        out_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each document
    for file_path in file_paths:
        path_obj = Path(file_path) if isinstance(file_path, str) else file_path
        
        try:
            # Process the document
            processed_doc = process_document(path_obj, options)
            
            # Check for validation errors
            if "_validation_error" in processed_doc:
                stats["validation_failures"] += 1
                logger.warning(f"Validation failed for {path_obj}: {processed_doc['_validation_error']}")
            
            stats["successful"] += 1
            
            # Track document ID
            if "id" in processed_doc:
                document_ids.append(processed_doc["id"])
            
            # Track format statistics
            doc_format = processed_doc.get("format", "unknown")
            stats["formats"][doc_format] = stats["formats"].get(doc_format, 0) + 1
            
            # Add to successful documents list
            successful_docs.append(cast(Dict[str, Any], processed_doc))
            
            # Save if output directory is provided
            if output_dir:
                out_path = Path(output_dir) / f"{path_obj.stem}.json"
                save_processed_document(processed_doc, out_path)
                output_files.append(str(out_path))
            
            # Call success callback if provided
            if on_success:
                if output_dir:
                    result_path = Path(output_dir) / f"{path_obj.stem}.json"
                    on_success(cast(Dict[str, Any], processed_doc), result_path)
                else:
                    # Use empty string path when no output directory is provided
                    on_success(cast(Dict[str, Any], processed_doc), "")
                
        except Exception as e:
            stats["failed"] += 1
            
            # Track error statistics
            error_type = type(e).__name__
            stats["errors"][error_type] = stats["errors"].get(error_type, 0) + 1
            
            # Add to failed documents list
            failed_docs.append({"path": str(path_obj), "error": str(e), "error_type": error_type})
            
            # Call error callback if provided
            if on_error:
                on_error(str(path_obj), e)
            else:
                # Log the error
                logger.error(f"Error processing {path_obj}: {e}")
    
    # Calculate average processing time (placeholder - would be more accurate with actual timing)
    if stats["successful"] > 0:
        stats["processing_time"]["average"] = stats["processing_time"]["total"] / stats["successful"]
    
    # Create and return the final BatchProcessingResult
    return cast(BatchProcessingResult, {
        "stats": stats,
        "documents": document_ids,
        "successful_documents": successful_docs,
        "failed_documents": failed_docs,
        "output_files": output_files if output_dir else None
    })
