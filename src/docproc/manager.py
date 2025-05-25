"""
Document processor manager for HADES-PathRAG.

This module provides a manager class for document processing operations,
coordinating the processing of documents with various adapters. The manager serves
as the primary entry point for document processing in the ingestion pipeline.

The DocumentProcessorManager class provides a unified interface for:
1. Processing documents from files or direct content
2. Batch processing of multiple documents
3. Efficient adapter caching to improve performance
4. Format detection and appropriate adapter selection

Usage examples:
    # Process a document file
    from src.docproc.manager import DocumentProcessorManager
    
    manager = DocumentProcessorManager()
    document = manager.process_document(path="/path/to/document.pdf")
    print(f"Processed document: {document['id']}")
    
    # Process text content directly
    markdown_content = "# Title\n\nSome content."
    document = manager.process_document(content=markdown_content, doc_type="markdown")
    
    # Batch processing
    results = manager.batch_process([
        "/path/to/doc1.txt",
        "/path/to/doc2.pdf",
        "/path/to/code.py"
    ])
"""

import logging
import os
import sys
import importlib
from typing import Dict, Any, Optional, Union, List, cast
from pathlib import Path

from src.docproc.adapters.base import BaseAdapter
from .core import process_document, process_text, get_format_for_document
from .adapters.registry import get_adapter_for_format
from ..utils.device_utils import is_gpu_available

logger = logging.getLogger(__name__)


class DocumentProcessorManager:
    """Manager for document processing operations.
    
    This class provides a centralized interface for processing documents
    of various formats, managing adapters and configuration. It serves as the
    primary entry point for the document processing stage of the ingestion pipeline.
    
    The manager handles:
    - Format detection based on file extensions and content analysis
    - Adapter selection and instantiation
    - Adapter caching for improved performance
    - Processing documents from files or direct content
    - Batch processing with error handling
    - Consistent document output format
    
    The DocumentProcessorManager doesn't implement any processing logic itself;
    instead, it delegates to the appropriate format-specific adapters registered
    in the adapter registry.
    
    Example usage:
        # Create a manager instance
        manager = DocumentProcessorManager()
        
        # Process a document file
        result = manager.process_document(path="/path/to/document.pdf")
        print(f"Document ID: {result['id']}")
        print(f"Content length: {len(result['content'])}")
        
        # Process text content directly
        result = manager.process_document(
            content="# Document\n\nThis is content.",
            doc_type="markdown"
        )
        
        # Process multiple files
        results = manager.batch_process([
            "/path/to/file1.txt",
            "/path/to/file2.md"
        ])
    """
    
    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the DocumentProcessorManager.

        Args:
            options (Dict[str, Any], optional): Options for processing. Defaults to None.
        """
        self.options = options or {}
        
        # Try to use the new config_loader module first for handling CUDA_VISIBLE_DEVICES
        try:
            from src.config.config_loader import load_pipeline_config
            # Load the pipeline configuration with pipeline_type='training'
            pipeline_config = load_pipeline_config(pipeline_type='training')
            
            # Check if there's a device_config section
            if 'pipeline' in pipeline_config and 'device_config' in pipeline_config['pipeline']:
                device_config = pipeline_config['pipeline']['device_config']
                
                # Get CUDA_VISIBLE_DEVICES from pipeline config if it exists
                if 'CUDA_VISIBLE_DEVICES' in device_config:
                    cuda_devices = device_config['CUDA_VISIBLE_DEVICES']
                    
                    # Apply this setting - explicitly overwrites any existing environment variable
                    # to ensure the config file takes precedence
                    if cuda_devices is not None:  # None means use system default
                        # Ensure it's set in the proper uppercase format
                        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_devices)
                        if cuda_devices == "":
                            logger.info("Setting CUDA_VISIBLE_DEVICES to empty string - forcing CPU mode")
                        else:
                            logger.info(f"Setting CUDA_VISIBLE_DEVICES to '{cuda_devices}'")
        except ImportError:
            # Fall back to the original mechanism if config_loader is not available
            # Check both environment variable and pipeline configuration for CUDA_VISIBLE_DEVICES
            # First check if it's already set in the environment (uppercase version takes precedence)
            cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
            
            # If not set in environment, check if it's in the pipeline config
            if cuda_devices is None and 'pipeline' in self.options and 'device_config' in self.options['pipeline']:
                device_config = self.options['pipeline']['device_config']
                if 'cuda_visible_devices' in device_config:
                    cuda_devices = device_config['cuda_visible_devices']
                    
            # Apply the setting if we found a value
            if cuda_devices is not None:  # None means use system default
                # Ensure it's set in the proper uppercase format
                os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_devices)
                if cuda_devices == "":
                    logger.info("Setting CUDA_VISIBLE_DEVICES to empty string - forcing CPU mode")
                else:
                    logger.info(f"Setting CUDA_VISIBLE_DEVICES to '{cuda_devices}'")
        
        # Log the final CUDA_VISIBLE_DEVICES setting
        logger.info(f"Final CUDA_VISIBLE_DEVICES setting: '{os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}'")   
        # Check if CUDA is available after applying settings
        gpu_available = is_gpu_available()
        
        # Log device configuration
        if not gpu_available:
            logger.info(f"Using CPU configuration with {os.cpu_count()} threads")
        else:
            # If GPU is available, get details and log them
            import torch
            gpu_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            logger.info(f"Using GPU configuration: {gpu_count} devices available: {gpu_names}")
            
            # If we have specific GPU device configurations in the options, use them
            gpu_device = None
            
            # Check for explicit device setting in GPU execution config
            if 'gpu_execution' in self.options and self.options['gpu_execution'].get('enabled', False):
                # Get docproc device setting if available
                if 'docproc' in self.options['gpu_execution'] and 'device' in self.options['gpu_execution']['docproc']:
                    gpu_device = self.options['gpu_execution']['docproc']['device']
                    logger.info(f"Using GPU device from config: {gpu_device}")
                    
            # Set PyTorch device index if specified
            if gpu_device and gpu_device.startswith('cuda:') and gpu_count > 0:
                try:
                    # Extract index from cuda:N format
                    device_idx = gpu_device.split(':')[1]
                    os.environ['PYTORCH_CUDA_DEVICE'] = device_idx
                    logger.info(f"Setting PYTORCH_CUDA_DEVICE to {device_idx}")
                except (IndexError, ValueError):
                    # Default to first GPU if parsing fails
                    os.environ['PYTORCH_CUDA_DEVICE'] = '0'
                    logger.info("Defaulting PYTORCH_CUDA_DEVICE to 0")
            elif gpu_count > 0:
                # Default to first available GPU
                os.environ['PYTORCH_CUDA_DEVICE'] = '0'
                logger.info("Defaulting PYTORCH_CUDA_DEVICE to 0")
            
        self.cache: Dict[str, BaseAdapter] = {}  # Simple cache for frequently used adapters
        logger.info("Document processor manager initialized")
    
    def process_document(
        self, 
        content: Optional[str] = None,
        path: Optional[Union[str, Path]] = None,
        doc_type: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a document from either content or file path.
        
        This method is the primary entry point for document processing and provides
        a flexible interface that can process either:
        1. Direct text content with a specified type (when content is provided)
        2. A file from a given path (when path is provided)
        
        The method automatically handles format detection, adapter selection,
        and produces a standardized document output that can be used by downstream
        components in the ingestion pipeline.
        
        Args:
            content: Document content as text (optional if path is provided). This can be
                    raw text, markdown, HTML, or other textual formats.
            path: Path to the document file (optional if content is provided). The path is
                 used both for loading content and for format detection.
            doc_type: Document type/format (optional, detected from path if not provided).
                     Examples: "text", "markdown", "html", "pdf", "python", etc.
            options: Processing options to pass to the adapter. These are adapter-specific
                    and control details of the processing behavior. Common options include:
                    - extract_metadata: Whether to extract metadata (default: True)
                    - extract_entities: Whether to extract entities (default: True)
                    - format_override: Override automatic format detection
                    - max_content_length: Limit the content length
            
        Returns:
            Dictionary with processed document content and metadata in a standardized
            format. The dictionary contains keys such as:
            - id: Unique document identifier
            - content: Processed text content
            - path: Original file path (if provided)
            - format: Document format/type
            - metadata: Dictionary of document metadata
            - entities: List of extracted entities (if supported by the adapter)
            
        Raises:
            ValueError: If neither content nor path is provided
            FileNotFoundError: If the specified file does not exist
            TypeError: If the document format is not supported
            
        Examples:
            # Process a file on disk
            result = manager.process_document(path="/path/to/document.pdf")
            
            # Process direct text content with explicit type
            markdown = "# Title\n\nContent paragraph"
            result = manager.process_document(content=markdown, doc_type="markdown")
            
            # Process with custom options
            result = manager.process_document(
                path="/path/to/code.py",
                options={
                    "extract_docstrings": True,
                    "max_content_length": 10000
                }
            )
        """
        options = options or {}
        
        # If we have global options from initialization, merge them
        if self.options and 'processing_options' in self.options:
            merged_options = {**self.options['processing_options'], **options}
            options = merged_options
        
        # Case 1: Process text content with specified type
        if content is not None:
            if not doc_type:
                # Default to plain text if no type is specified
                doc_type = "text"
                
            # If path is provided but content is also supplied, use path just for metadata
            document = process_text(content, doc_type, options)
            
            # Add path info if available
            if path:
                document["path"] = str(path)
                
            return document
        
        # Case 2: Process file from path
        elif path is not None:
            path_obj = Path(path) if isinstance(path, str) else path
            
            # If doc_type is explicitly provided, use it
            if doc_type:
                options["format_override"] = doc_type
                
            return process_document(path_obj, options)
        
        # If neither content nor path is provided
        else:
            raise ValueError("Either content or path must be provided")
    
    def batch_process(
        self,
        paths: List[Union[str, Path]],
        options: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Process a batch of documents from paths.
        
        This method allows efficient processing of multiple documents in a single call.
        It handles errors gracefully, ensuring that a failure in one document doesn't
        prevent the processing of other documents in the batch. Each failed document
        is included in the results with error information.
        
        Args:
            paths: List of paths to process. Each path should point to an existing file.
                  Supported formats are determined by the registered adapters.
            options: Processing options to apply to all documents. These are the same
                    options as for process_document() and are passed to each adapter.
                    Format-specific options will be applied only to documents of the
                    corresponding format.
            
        Returns:
            List of processed documents, in the same order as the input paths. Each
            document is represented by a dictionary with the standard document format.
            Failed documents will have the following structure:
            {
                "path": "original/path/to/file.txt",
                "error": "Error message describing what went wrong",
                "status": "failed"
            }
            
        Examples:
            # Basic batch processing
            documents = manager.batch_process([
                "/path/to/doc1.txt",
                "/path/to/doc2.md",
                "/path/to/doc3.pdf"
            ])
            
            # With processing options
            documents = manager.batch_process(
                ["/path/to/doc1.py", "/path/to/doc2.py"],
                options={
                    "extract_docstrings": True,
                    "include_comments": True
                }
            )
            
            # Processing results with error handling
            results = manager.batch_process(paths)
            successful = [doc for doc in results if "error" not in doc]
            failed = [doc for doc in results if "error" in doc]
            print(f"Successfully processed {len(successful)} documents")
            print(f"Failed to process {len(failed)} documents")
        """
        results = []
        options = options or {}
        
        for path in paths:
            try:
                document = self.process_document(path=path, options=options)
                results.append(document)
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                # Add a placeholder for failed documents
                results.append({
                    "path": str(path),
                    "error": str(e),
                    "status": "failed"
                })
        
        return results
    
    def get_adapter_for_doc_type(self, doc_type: str) -> BaseAdapter:
        """Get the appropriate adapter for a document type.
        
        This method retrieves the appropriate adapter for a given document type from
        the adapter registry. It implements a caching mechanism to avoid repeatedly
        creating new adapter instances for the same document type, which improves
        performance when processing multiple documents of the same type.
        
        Args:
            doc_type: Document type/format identifier string. Common types include:
                     "text", "markdown", "html", "pdf", "python", "json", etc.
                     The available types depend on the adapters registered in the system.
            
        Returns:
            Adapter instance for the specified document type. This will be a concrete
            implementation of the BaseAdapter appropriate for the specified format.
            
        Raises:
            ValueError: If no adapter is registered for the specified document type
            
        Examples:
            # Get an adapter for PDF documents
            pdf_adapter = manager.get_adapter_for_doc_type("pdf")
            
            # Process markdown content with the markdown adapter
            markdown_adapter = manager.get_adapter_for_doc_type("markdown")
            result = markdown_adapter.process_text("# Document Title\n\nContent")
            
            # Multiple calls for the same type return the same cached instance
            adapter1 = manager.get_adapter_for_doc_type("text")
            adapter2 = manager.get_adapter_for_doc_type("text")
            assert adapter1 is adapter2  # True, same instance
        """
        if doc_type in self.cache:
            return self.cache[doc_type]
            
        adapter = get_adapter_for_format(doc_type)
        self.cache[doc_type] = adapter
        return adapter
