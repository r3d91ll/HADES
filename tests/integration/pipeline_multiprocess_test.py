#!/usr/bin/env python
"""
Multiprocessing implementation of the document processing pipeline test.

This test demonstrates true parallel processing of documents using Python's multiprocessing
framework and the training pipeline components. It allocates resources efficiently for
document processing, chunking, and embedding generation.

IMPORTANT: This pipeline demonstrates how to create a new HADES datastore from scratch.
It processes all documents in a batch, builds a complete graph of relationships, and applies 
ISNE enhancement across the entire document set. This approach is optimized for initial 
database population.

For adding data to an existing datastore, a modified pipeline would be required that can:
1. Query existing database entries for related content
2. Build partial graphs incorporating both new and existing documents
3. Process incrementally rather than requiring full batch processing

The test operates in three modes:
1. Document processing mode - processes documents, chunks them, and generates embeddings
2. ISNE training mode - trains an ISNE model on processed documents
3. ISNE ingestion mode - applies a trained ISNE model to enhance document embeddings with relationships

Typical usage:
    # Process documents and train ISNE model
    python pipeline_multiprocess_test.py --mode train --data-dir ./test-data --output-dir ./test-output
    
    # Apply trained ISNE model to documents and create a new datastore
    python pipeline_multiprocess_test.py --mode ingest --data-dir ./test-data --output-dir ./test-output \
        --model-path ./models/isne/isne_model_latest.pt
"""

import os
import sys
import time
import yaml
import json
import copy
import asyncio
import logging
import argparse
import traceback
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import multiprocessing as mp
from multiprocessing import Pool, Queue, Manager

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import local modules
from src.docproc.manager import DocumentProcessorManager
from src.chunking.text_chunkers.chonky_chunker import chunk_text
from src.embedding.adapters import ModernBERTEmbeddingAdapter
from src.docproc.utils.format_detector import (
    detect_format_from_path,
    get_content_category
)
from src.alerts import AlertManager, AlertLevel
from src.validation.embedding_validator import (
    validate_embeddings_before_isne,
    validate_embeddings_after_isne,
    create_validation_summary,
    attach_validation_summary
)
from src.alerts import AlertManager, AlertLevel
from src.validation.embedding_validator import (
    validate_embeddings_before_isne,
    validate_embeddings_after_isne,
    create_validation_summary,
    attach_validation_summary
)

# Custom timing formatter
class TimingFormatter(logging.Formatter):
    """Formatter that adds elapsed time since start."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()
    
    def formatTime(self, record, datefmt=None):
        formatted_time = super().formatTime(record, datefmt)
        # Add elapsed time since start
        elapsed = time.time() - self.start_time
        return f"{formatted_time} [+{elapsed:.3f}s]"

# Configure logging with custom formatter
formatter = TimingFormatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Remove existing handlers
for handler in root_logger.handlers:
    root_logger.removeHandler(handler)

# Add console handler with custom formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

# Get our module logger
logger = logging.getLogger(__name__)

# Load the pipeline configuration
def load_pipeline_config() -> Dict[str, Any]:
    """Load the training pipeline configuration from YAML.
    
    This function attempts to use the new config_loader module first,
    and falls back to direct file loading if that's not available.
    """
    try:
        # Try to use the new config_loader module first
        from src.config.config_loader import load_pipeline_config as load_config
        return load_config(pipeline_type='training')
    except ImportError:
        # Fall back to direct file loading
        config_path = Path(project_root) / "src" / "config" / "training_pipeline_config.yaml"
        if not config_path.exists():
            # Check for old filename as a last resort
            old_path = Path(project_root) / "src" / "config" / "pipeline_config.yaml"
            if old_path.exists():
                config_path = old_path
                logger.warning(f"Using deprecated pipeline_config.yaml. Please update to training_pipeline_config.yaml.")
            else:
                raise FileNotFoundError(f"Cannot find pipeline configuration at {config_path} or {old_path}")
                
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

# Initialize components based on the configuration
def init_docproc(config: Dict[str, Any], worker_id: int) -> Any:
    """Initialize document processor with appropriate configuration."""
    try:
        # Get processor type based on worker ID
        processor_type = config.get("processor_type", "generic")
        logger.info(f"[Worker {worker_id}] Initializing document processor '{processor_type}'")
        
        # Initialize processor based on type
        # Use DocumentProcessorManager from src.docproc.manager
        from src.docproc.manager import DocumentProcessorManager
        
        # Configure document processor
        processor_options = config.get("document_processor", {})
        
        # Set up device configuration for GPU handling
        if config.get("gpu_execution", {}).get("enabled", False):
            device_config = config["gpu_execution"]["docproc"]
            processor_options["device"] = device_config.get("device", "cuda:0")
        
        return DocumentProcessorManager(options=processor_options)
    except Exception as e:
        logger.error(f"Error initializing document processor: {e}")
        raise

def init_chunker(config: Dict[str, Any], worker_id: int, custom_config: Optional[Dict[str, Any]] = None) -> Any:
    """Initialize chunker with appropriate device and configuration.
    
    Args:
        config: Global pipeline configuration
        worker_id: Worker ID for logging
        custom_config: Optional custom configuration to override defaults
        
    Returns:
        Initialized chunker instance
    """
    try:
        # Get chunker type from config
        chunker_type = config.get("chunker", "chonky")
        logger.info(f"[Worker {worker_id}] Initializing chunker '{chunker_type}'")
        
        # Set up default configuration with smaller chunk sizes for embedding compatibility
        default_config = {
            "max_chunk_size": 400,  # Characters, keep smaller for embedding model compatibility
            "min_chunk_size": 50,
            "overlap": 20,
            "respect_boundaries": True
        }
        
        # Override with any custom configuration
        if custom_config:
            default_config.update(custom_config)
            
        logger.info(f"[Worker {worker_id}] Using chunker config: max_size={default_config['max_chunk_size']}, min_size={default_config['min_chunk_size']}")
            
        # Import and initialize appropriate chunker
        if chunker_type == "chonky":
            # Import required modules
            from src.chunking.base import BaseChunker
            from src.chunking.text_chunkers.chonky_chunker import chunk_text
            # Create a wrapper class for the function-based chonky_chunker
            class ChonkyChunkerWrapper(BaseChunker):
                """Wrapper class for the function-based chonky_chunker."""
                
                def __init__(self, config=None):
                    super().__init__(name="chonky", config=config or {})
                    
                def chunk(self, text=None, content=None, doc_id=None, path="unknown", doc_type="text", max_tokens=1024, output_format="json", **kwargs):
                    """Wrap the chunk_text function to match BaseChunker interface.
                    
                    Accepts both 'text' and 'content' parameters for compatibility with different calling conventions.
                    """
                    # Use content if provided, otherwise fall back to text
                    actual_content = content if content is not None else text
                    
                    # Use max_chunk_size from config if provided
                    max_tokens = self.config.get("max_chunk_size", max_tokens)
                    
                    # Call the chonky chunker function
                    result = chunk_text(
                        content=actual_content,
                        doc_id=doc_id,
                        path=path,
                        doc_type=doc_type,
                        max_tokens=max_tokens,
                        output_format=output_format
                    )
                    return result.get("chunks", []) if isinstance(result, dict) else []
            
            return ChonkyChunkerWrapper(config=default_config)
        else:
            # For other chunker types, use registry
            from src.chunking import get_chunker
            return get_chunker(chunker_type, config=default_config)
    except Exception as e:
        logger.error(f"Error initializing chunker: {e}")
        raise

def init_embedding(config: Dict[str, Any], worker_id: int, file_type: str = "document", content_category: str = "text") -> Dict[str, Any]:
    """Initialize embedding adapter with appropriate device and model based on file format and content category.
    
    Args:
        config: Configuration dictionary
        worker_id: Worker ID for logging
        file_type: Format type of the file (python, yaml, json, markdown, etc.)
        content_category: Content category of the file (code or text)
        
    Returns:
        Initialized embedding adapter
    """
    logger.info(f"[Worker {worker_id}] Initializing embedding adapter...")
    
    # Import the renamed adapter class
    from src.embedding.adapters import EncoderEmbeddingAdapter
    
    # Select specialized models based on content category and format type
    # Set max sequence length for all models to avoid truncation issues
    max_seq_length = 512
    truncation = True
    
    # First check content category to determine the general model type
    if content_category == "code":
        # Use code-specific models for all code formats
        model_name = "microsoft/codebert-base"
        model_type = "encoder"
        
        # Set format-specific embedding type for better tracking
        embedding_type = f"{file_type}_code"
        logger.info(f"[Worker {worker_id}] Using CodeBERT for {file_type} files")
    else:
        # Use the ModernBERT model for text documents
        model_name = "bert-base-uncased"  # Use a standard BERT model that's publicly available
        model_type = "modernbert"
        embedding_type = "text"
        logger.info(f"[Worker {worker_id}] Using ModernBERT for {file_type} files")
    
    # Configure embedding based on execution mode
    if config.get("gpu_execution", {}).get("enabled", False):
        device = config["gpu_execution"]["embedding"]["device"]
        batch_size = config["gpu_execution"]["embedding"]["batch_size"]
        precision = config["gpu_execution"]["embedding"]["model_precision"]
        logger.info(f"[Worker {worker_id}] Using GPU configuration for embedding with device: {device}, precision: {precision}")
    else:
        device = config["cpu_execution"]["embedding"]["device"]
        num_threads = config["cpu_execution"]["embedding"]["num_threads"]
        logger.info(f"[Worker {worker_id}] Using CPU configuration for embedding with {num_threads} threads")
    
    # Initialize the adapter
    from src.embedding import get_adapter_by_name
    adapter = get_adapter_by_name(model_type)(
        model_name=model_name, 
        device=device,
        max_seq_length=max_seq_length,
        truncation=truncation
    )
    
    logger.info(f"[Worker {worker_id}] Initialized {model_type} adapter with model={model_name} on {device}")
    return {
        "adapter": adapter, 
        "worker_id": worker_id,
        "model_name": model_name,
        "model_type": model_type,
        "embedding_type": embedding_type
    }

# Process a document using multiprocessing
def process_document(args: Tuple[Path, Dict[str, Any], int]):
    """Process a single document using the pipeline components."""
    file_path, config, worker_id = args
    
    # Create a worker-specific logger
    worker_logger = logging.getLogger(f"__mp_main__")
    worker_logger.info(f"[Worker {worker_id}] Processing file: {file_path.name}")
    
    start_time = time.time()
    
    # Initialize timings dictionary to prevent UnboundLocalError in exception handling
    timings = {"docproc": 0, "chunking": 0, "embedding": 0, "total": 0}
    
    try:
        # Initialize components
        doc_processor = init_docproc(config, worker_id)
        
        # Detect file format and content category using the new format detector
        try:
            format_type = detect_format_from_path(file_path)
            content_category = get_content_category(format_type)
            worker_logger.info(f"[Worker {worker_id}] Detected format '{format_type}' and category '{content_category}' for {file_path.name}")
        except ValueError as e:
            # Skip unsupported file types gracefully
            worker_logger.warning(f"[Worker {worker_id}] Skipping unsupported file: {file_path.name} - {str(e)}")
            return {
                "file_id": f"skipped_{file_path.name}",
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size if file_path.exists() else 0,
                "worker_id": worker_id,
                "status": "skipped",
                "reason": "unsupported_format",
                "error": str(e),
                "chunks": [],
                "processing_time": 0
            }
        
        # Prepare chunker config to limit chunk size for embedding compatibility
        chunker_config = {
            "max_chunk_size": 400,  # Characters, keep smaller for embedding model compatibility
            "min_chunk_size": 50,
            "overlap": 20,
            "respect_boundaries": True
        }
        
        # Get chunker based on content category and format type
        if content_category == "code":
            if format_type == "python":
                # Use Python code chunker
                from src.chunking.code_chunkers.python_chunker import PythonCodeChunker
                chunker = PythonCodeChunker(config=chunker_config)
                worker_logger.info(f"[Worker {worker_id}] Using Python code chunker")
            elif format_type in ["yaml", "yml"]:
                # Use YAML code chunker
                from src.chunking.code_chunkers.yaml_chunker import YAMLCodeChunker
                chunker = YAMLCodeChunker(config=chunker_config)
                worker_logger.info(f"[Worker {worker_id}] Using YAML code chunker")
            elif format_type == "json":
                # Use JSON code chunker
                from src.chunking.code_chunkers.json_chunker import JSONCodeChunker
                chunker = JSONCodeChunker(config=chunker_config)
                worker_logger.info(f"[Worker {worker_id}] Using JSON code chunker")
            else:
                # For other code formats, use generic code chunker
                worker_logger.info(f"[Worker {worker_id}] Using generic code chunker for {format_type}")
                from src.chunking.code_chunkers.generic_code_chunker import GenericCodeChunker
                chunker = GenericCodeChunker(config=chunker_config)
        else:
            # For text content, use the standard chunker initialization with custom config
            worker_logger.info(f"[Worker {worker_id}] Using text chunker for {format_type}")
            custom_chunker_config = {
                "max_chunk_size": 400,  # Characters, keep smaller for embedding model compatibility
                "min_chunk_size": 50,
                "overlap": 20,
                "respect_boundaries": True
            }
            chunker = init_chunker(config, worker_id, custom_config=custom_chunker_config)
            
        # Initialize embedding adapter with the appropriate model based on format type and content category
        embedder = init_embedding(config, worker_id, file_type=format_type, content_category=content_category)
        
        # Track timings
        timings = {"docproc": 0, "chunking": 0, "embedding": 0, "total": 0}
        
        # Step 1: Process document
        worker_logger.info(f"[Worker {worker_id}] Starting document processing")
        docproc_start = time.time()
        
        # Process the document based on file type
        if file_path.suffix.lower() == ".py":
            # Direct handling for Python files
            worker_logger.info(f"[Worker {worker_id}] Using direct PythonCodeAdapter for {file_path.name}")
            # Import and use PythonCodeAdapter directly
            try:
                from src.docproc.adapters.python_code_adapter import PythonCodeAdapter
                python_adapter = PythonCodeAdapter()
                doc_result = python_adapter.process(file_path)
            except Exception as e:
                worker_logger.error(f"[Worker {worker_id}] Error processing Python file: {e}")
                raise
        else:
            # Use standard document processor for other file types
            doc_result = doc_processor.process_document(path=file_path)
        
        docproc_end = time.time()
        docproc_time = docproc_end - docproc_start
        timings["docproc"] = docproc_time
        
        worker_logger.info(f"[Worker {worker_id}] Document processing completed in {docproc_time:.2f}s")
        
        # Step 2: Chunking
        chunk_start = time.time()
        
        # Get document properties
        doc_id = doc_result.get('id', file_path.stem)
        doc_path = doc_result.get('source', str(file_path))
        
        worker_logger.info(f"[Worker {worker_id}] Starting document chunking: {file_path.name}")
        
        # Add metadata to the chunk including embedding model information
        content_type = format_type.lower() if format_type else "text"
        text = doc_result.get('content', "") or ""
        text = str(text)
        content_length = len(text)
        
        # Add chunking config to limit chunk size
        chunking_config = {
            "max_chunk_size": 500,  # Smaller chunks to stay within embedding model limits
            "overlap": 50,
            "respect_boundaries": True
        }
        
        worker_logger.info(f"[Worker {worker_id}] Extracted content for chunking: {content_length/1024:.1f} KB")
        
        # Process through chunking
        # Set document type based on the detected format and content category
        if content_category == "code":
            doc_type = f"{format_type}_code"
        else:
            doc_type = format_type
        
        # Prepare metadata for all chunkers
        metadata = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "doc_id": doc_id,
            "path": doc_path,
            "doc_type": doc_type,
            "max_tokens": 1024,
            "output_format": "json"
        }
        
        # Add language metadata for Python files
        if doc_type == "python_code":
            metadata["language"] = "python"
        
        # Different chunkers have different interfaces, handle accordingly
        # Call the chunker to break down the content
        try:
            if hasattr(chunker, 'chunk'):
                # Use standard chunker interface
                chunks_result = chunker.chunk(text=text, metadata=metadata)
            else:
                # Use a function-based chunker
                chunks_result = chunker(
                    content=text,
                    doc_id=doc_id,
                    path=str(file_path),
                    doc_type=content_type
                )
        except Exception as e:
            worker_logger.error(f"Error during chunking: {e}")
            return []
        
        # Extract chunks - handle both dictionary and list return types
        if isinstance(chunks_result, dict):
            chunks = chunks_result.get("chunks", [])
        else:
            # If chunks_result is already a list of chunks
            chunks = chunks_result
            
        # Ensure each chunk has a unique ID for ISNE training
        for i, chunk in enumerate(chunks):
            if "id" not in chunk:
                # Create a unique ID using document ID and chunk index
                chunk["id"] = f"{doc_id}_chunk_{i}"
                
            # Add overlap_context structure required by ISNE orchestrator
            # This is crucial for creating edges in the document graph
            if "overlap_context" not in chunk:
                chunk["overlap_context"] = {}
                
            # Set position and total which are used to establish relationships
            chunk["overlap_context"]["position"] = i
            chunk["overlap_context"]["total"] = len(chunks)
            
            # Add empty pre/post context (optional but commonly expected)
            if "pre" not in chunk["overlap_context"]:
                chunk["overlap_context"]["pre"] = ""
            if "post" not in chunk["overlap_context"]:
                chunk["overlap_context"]["post"] = ""
        
        # Calculate stats on chunks
        chunk_sizes = [len(str(chunk.get("content", ""))) for chunk in chunks]
        avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        max_chunk_size = max(chunk_sizes) if chunk_sizes else 0
        min_chunk_size = min(chunk_sizes) if chunk_sizes else 0
        
        worker_logger.info(f"[Worker {worker_id}] Created {len(chunks)} chunks for {file_path.name} - "
                  f"Avg size: {avg_chunk_size/1024:.1f} KB, Range: {min_chunk_size/1024:.1f}-{max_chunk_size/1024:.1f} KB")
        
        chunk_time = time.time() - chunk_start
        timings["chunking"] = chunk_time
        
        # Step 3: Embedding generation
        embed_start = time.time()
        
        # Extract text content from each chunk for embedding
        texts = [chunk.get("content", "") for chunk in chunks]
        
        if not texts:
            worker_logger.warning(f"[Worker {worker_id}] No text content to embed for {file_path.name}")
            timings["embedding"] = time.time() - embed_start
            return {"file_id": doc_id, "file_name": file_path.name, "file_size": file_path.stat().st_size, "worker_id": worker_id, "chunks": chunks, "embeddings": [], "timing": timings}
        
        # Calculate total text size for embedding
        total_text_size = sum(len(text) for text in texts)
        avg_text_size = total_text_size / len(texts) if texts else 0
        
        worker_logger.info(f"[Worker {worker_id}] Starting embedding generation for {len(texts)} chunks from {file_path.name} "
                   f"(Total: {total_text_size/1024:.1f} KB, Avg: {avg_text_size/1024:.1f} KB per chunk)")
        
        # Get batch size from config
        if config.get("gpu_execution", {}).get("enabled", False):
            batch_size = config["gpu_execution"]["embedding"]["batch_size"]
        else:
            batch_size = 8  # Default batch size for CPU
        
        # Process in batches to avoid OOM
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_size_bytes = sum(len(text) for text in batch_texts)
            
            worker_logger.info(f"[Worker {worker_id}] Processing embedding batch {i//batch_size + 1}/"
                      f"{(len(texts) + batch_size - 1)//batch_size} "
                      f"({len(batch_texts)} chunks, {batch_size_bytes/1024:.1f} KB)")
            
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Run the async embed method in a synchronous context
                batch_embeddings = loop.run_until_complete(embedder["adapter"].embed(batch_texts))
                embeddings.extend(batch_embeddings)
            finally:
                loop.close()
        
        # Get embedding statistics
        embed_dims = len(embeddings[0]) if embeddings else 0
        
        # Convert embeddings to lists for JSON serialization
        embeddings_list = []
        for i, emb in enumerate(embeddings):
            if isinstance(emb, np.ndarray):
                emb_list = emb.tolist()
            else:
                emb_list = emb
            embeddings_list.append(emb_list)
            
            # Also attach embedding directly to the corresponding chunk for ISNE
            if i < len(chunks):
                chunks[i]["embedding"] = emb_list
                # Add embedding model metadata for ISNE awareness
                chunks[i]["embedding_model"] = embedder["model_name"]
                chunks[i]["embedding_type"] = embedder["embedding_type"]
        
        embed_time = time.time() - embed_start
        timings["embedding"] = embed_time
        
        # Total time
        timings["total"] = time.time() - start_time
        
        # Add timestamp to show when this task completed
        completed_at = datetime.now().isoformat()
        
        worker_logger.info(f"[Worker {worker_id}] Completed {file_path.name}: {len(chunks)} chunks, {len(embeddings)} embeddings in {timings['total']:.2f}s")
        
        return {"file_id": doc_id, "file_name": file_path.name, "file_size": file_path.stat().st_size, "worker_id": worker_id, "chunks": chunks, "embeddings": embeddings_list, "timing": timings, "completed_at": completed_at}
    except Exception as e:
        worker_logger.error(f"[Worker {worker_id}] Error processing document {file_path.name}: {str(e)}", exc_info=True)
    
    return {"file_id": file_path.stem, "file_name": file_path.name, "file_size": file_path.stat().st_size, "worker_id": worker_id, "chunks": [], "embeddings": [], "timing": timings}

class PipelineMultiprocessTester:
    """Test parallel processing of documents using true multiprocessing.
    
    This class coordinates the parallel processing of documents using configuration
    settings from the training_pipeline_config.yaml file. It demonstrates how to use
    the training pipeline components (document processing, chunking, and embedding)
    in a multiprocessing environment with proper GPU resource allocation.
    
    IMPORTANT: This implementation is designed for creating new HADES datastores from scratch.
    It processes documents in batch mode and builds a complete graph of relationships before
    applying ISNE enhancement. For adding data to an existing datastore, a different approach
    would be required that can leverage existing database content for relationship building.
    
    The implementation supports validation and alerting to ensure data quality throughout
    the pipeline, including verification of embedding and relationship integrity.
    """
    
    def __init__(
        self,
        test_data_dir: str,
        output_dir: str,
        num_files: int = 50,
        max_workers: int = 4,
        batch_size: int = 12,
        alert_threshold: str = "MEDIUM"
    ):
        """Initialize the multiprocessing pipeline tester."""
        self.test_data_dir = Path(test_data_dir)
        self.output_dir = Path(output_dir)
        self.num_files = num_files
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup alert directory
        self.alert_dir = self.output_dir / "alerts"
        self.alert_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize alert manager
        self.alert_threshold = getattr(AlertLevel, alert_threshold, AlertLevel.MEDIUM)
        self.alert_manager = AlertManager(
            alert_dir=str(self.alert_dir),
            min_level=self.alert_threshold,
            email_config=None  # No email alerts in test mode
        )
        
        # Load configuration
        self.config = load_pipeline_config()
        
        # Statistics
        self.stats = {
            "total_files": 0,
            "total_chunks": 0,
            "total_embeddings": 0,
            "processing_time": {
                "document_processing": 0,
                "chunking": 0,
                "embedding": 0,
                "total": 0
            },
            "file_details": {},
            "validation": {
                "pre_validation": {},
                "post_validation": {},
                "alerts": []
            }
        }
    
    def find_supported_files(self) -> List[Path]:
        """Find all supported file types recursively in the test directory and subdirectories."""
        supported_files = []
        
        # Get the extension-to-format mapping from our format detector
        from src.docproc.utils.format_detector import get_extension_to_format_map
        extension_map = get_extension_to_format_map()
        
        # First, find all files recursively
        all_files = list(self.test_data_dir.glob("**/*"))
        all_files = [f for f in all_files if f.is_file()]
        file_count = len(all_files)
        logger.info(f"Found {file_count} total files in {self.test_data_dir}")
        
        # Create a set of supported extensions from our format detector
        supported_extensions = set(extension_map.keys())
        logger.info(f"Supporting {len(supported_extensions)} file extensions: {', '.join(sorted(supported_extensions))}")
        
        # Only include files with supported extensions
        for file_path in all_files:
            ext = file_path.suffix.lower()
            if ext in supported_extensions:
                supported_files.append(file_path)
                format_type = extension_map[ext]
                content_category = get_content_category(format_type)
                logger.debug(f"Including {file_path.name} ({format_type}, {content_category})")
                
        # Group supported files by extension for logging
        by_extension = {}
        for file_path in supported_files:
            ext = file_path.suffix.lower()
            if ext not in by_extension:
                by_extension[ext] = []
            by_extension[ext].append(file_path)
            
        # Log counts by extension
        for ext, files in by_extension.items():
            format_type = extension_map[ext]
            content_category = get_content_category(format_type)
            logger.info(f"Found {len(files)} {ext} files (format: {format_type}, category: {content_category})")
            
        # Log summary
        if file_count > 0:
            logger.info(f"Including {len(supported_files)} of {file_count} files ({len(supported_files)/file_count*100:.1f}%)")
            logger.info(f"Excluding {file_count - len(supported_files)} unsupported files")
        else:
            logger.error(f"No files found in {self.test_data_dir}. Check if the path is correct and accessible.")
            logger.info(f"Absolute path: {Path(self.test_data_dir).absolute()}")
            # List parent directory to help diagnose the issue
            parent_dir = Path(self.test_data_dir).parent
            logger.info(f"Contents of parent directory {parent_dir}:")
            for item in parent_dir.iterdir():
                if item.is_dir():
                    logger.info(f"  DIR: {item.name}")
                else:
                    logger.info(f"  FILE: {item.name}")


                
        # Group files by directory
        files_by_dir = {}
        for file in supported_files:
            parent_dir = file.parent
            rel_path = parent_dir.relative_to(self.test_data_dir)
            if rel_path not in files_by_dir:
                files_by_dir[rel_path] = []
            files_by_dir[rel_path].append(file)
            
        for dir_path, dir_files in files_by_dir.items():
            if str(dir_path) == '.':
                logger.info(f"Found {len(dir_files)} files in root directory")
            else:
                logger.info(f"Found {len(dir_files)} files in subdirectory {dir_path}")
                
        self.directory_structure = files_by_dir  # Store for relationship building later
        
        # Sort files to ensure consistent order
        supported_files = sorted(supported_files, key=lambda x: str(x))
        logger.info(f"Found {len(supported_files)} supported files in {self.test_data_dir}")
        
        return supported_files
        
    def enhance_with_isne(self, docs, model_path):
        """
        Apply trained ISNE model to enhance document embeddings.
        
        Args:
            documents: List of processed documents with base embeddings
            model_path: Path to the trained ISNE model
            
        Returns:
            List of documents with enhanced embeddings
        """
        import os
        from src.isne.trainer.training_orchestrator import ISNETrainingOrchestrator
        # Check for any existing ISNE embeddings (shouldn't be any at this stage)
        existing_isne = sum(1 for doc in docs for chunk in doc.get("chunks", []) 
                          if "isne_embedding" in chunk)
            
        # Count chunks with base embeddings
        chunks_with_base_embeddings = sum(1 for doc in docs for chunk in doc.get("chunks", []) 
                                       if "embedding" in chunk and chunk["embedding"])
            
        logger.info(f"Pre-ISNE Validation: {total_docs} documents, {docs_with_chunks} with chunks, {total_chunks} total chunks")
        logger.info(f"Found {chunks_with_base_embeddings}/{total_chunks} chunks with base embeddings")
            
        if existing_isne > 0:
            logger.warning(f"Found {existing_isne} chunks with existing ISNE embeddings before application!")
                
        if chunks_with_base_embeddings < total_chunks:
            logger.warning(f"Missing base embeddings in {total_chunks - chunks_with_base_embeddings} chunks")
            
        return {
            "total_docs": total_docs,
            "docs_with_chunks": docs_with_chunks,
            "total_chunks": total_chunks,
            "chunks_with_base_embeddings": chunks_with_base_embeddings,
            "existing_isne": existing_isne
        }
        
        # Sort files to ensure consistent order
        supported_files = sorted(supported_files, key=lambda x: str(x))
        logger.info(f"Found {len(supported_files)} supported files in {self.test_data_dir}")
        
        return supported_files
        
    def enhance_with_isne(self, docs, model_path):
        """
        Apply trained ISNE model to enhance document embeddings.
        
        Args:
            docs: List of documents with chunks and base embeddings
            model_path: Path to the trained ISNE model
            
        Returns:
            Enhanced documents with ISNE embeddings
        """
        import torch
        from torch_geometric.data import Data
        from src.isne.trainer.training_orchestrator import ISNETrainingOrchestrator
        
        logger.info(f"Enhancing documents with ISNE model from {model_path}")
        enhanced_documents = docs.copy()
        
        # Extract embeddings and metadata
        node_embeddings = []
        node_metadata = []
        node_model_types = []  # Track embedding model types
        edge_index_src = []
        edge_index_dst = []
        edge_attr = []
        
        # Node index mapping
        node_idx_map = {}
        current_idx = 0
        
        # First pass: collect all nodes
        logger.info("Building document graph for ISNE inference")
        for doc in enhanced_documents:
            if "chunks" not in doc or not doc["chunks"]:
                continue
            
            for chunk_idx, chunk in enumerate(doc["chunks"]):
                if "embedding" not in chunk or not chunk["embedding"]:
                    continue
                
                chunk_id = f"{doc['file_id']}_{chunk_idx}"
                node_idx_map[chunk_id] = current_idx
                
                # Add the base embedding
                node_embeddings.append(chunk["embedding"])
                
                # Add metadata
                metadata = {
                    "doc_id": doc["file_id"],
                    "chunk_idx": chunk_idx,
                    "text": chunk.get("text", "")[:100]  # Truncate for metadata
                }
                node_metadata.append(metadata)
                
                # Track embedding model type
                model_type = chunk.get("metadata", {}).get("embedding_model", "default")
                node_model_types.append(model_type)
                
                current_idx += 1
        
        # Second pass: build edges
        for doc in enhanced_documents:
            if "chunks" not in doc or not doc["chunks"]:
                continue
            
            # Create sequential edges between chunks in the same document
            for i in range(len(doc["chunks"]) - 1):
                source_id = f"{doc['file_id']}_{i}"
                target_id = f"{doc['file_id']}_{i+1}"
                
                if source_id in node_idx_map and target_id in node_idx_map:
                    edge_index_src.append(node_idx_map[source_id])
                    edge_index_dst.append(node_idx_map[target_id])
                    edge_attr.append([1.0])  # Sequential relationship weight
            
            # Add code-specific relationships from chunk metadata
            for chunk_idx, chunk in enumerate(doc["chunks"]):
                if "metadata" not in chunk or "references" not in chunk["metadata"]:
                    continue
                
                source_id = f"{doc['file_id']}_{chunk_idx}"
                if source_id not in node_idx_map:
                    continue
                
                # Process each reference (relationship)
                for reference in chunk["metadata"]["references"]:
                    target_id = reference.get("target")
                    if not target_id or target_id not in node_idx_map:
                        continue
                    
                    relation_type = reference.get("type", "REFERENCES")
                    weight = reference.get("weight", 0.8)
                    
                    # Add edge to the graph
                    edge_index_src.append(node_idx_map[source_id])
                    edge_index_dst.append(node_idx_map[target_id])
                    edge_attr.append([weight])  # Use the relationship weight
        
        if not node_embeddings:
            logger.warning("No valid nodes found for ISNE enhancement")
            return enhanced_documents
        
        # Convert lists to tensors
        try:
            x = torch.tensor(node_embeddings, dtype=torch.float)
            edge_index = torch.tensor([edge_index_src, edge_index_dst], dtype=torch.long)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            
            # Create PyTorch Geometric Data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
            logger.info(f"Built graph with {x.shape[0]} nodes and {edge_index.shape[1]} edges")
            
            # Load model
            logger.info("Loading ISNE model")
            model = ISNETrainingOrchestrator.load_model(model_path)
            
            # Apply ISNE model to get enhanced embeddings
            logger.info("Applying ISNE model to enhance embeddings")
            enhanced_embeddings = model(data.x, data.edge_index, data.edge_attr)
            
            # Add enhanced embeddings back to documents
            logger.info("Adding enhanced embeddings to documents")
            
            # Add ISNE embeddings to chunks based on node mapping
            for doc in enhanced_documents:
                if "chunks" not in doc or not doc["chunks"]:
                    continue
                
                for chunk_idx, chunk in enumerate(doc["chunks"]):
                    if "embedding" not in chunk or not chunk["embedding"]:
                        continue
                    
                    chunk_id = f"{doc['file_id']}_{chunk_idx}"
                    if chunk_id in node_idx_map:
                        node_idx = node_idx_map[chunk_id]
                        # Add the enhanced embedding to the chunk
                        chunk["isne_embedding"] = enhanced_embeddings[node_idx].tolist()
            
            logger.info("Document enhancement with ISNE embeddings completed")
            
        except Exception as e:
            error_msg = f"Error during ISNE enhancement: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.alert_manager.alert(
                message=error_msg,
                level="CRITICAL",
                source="isne_pipeline",
                context={"exception": str(e)}
            )
        
        return enhanced_documents
    
    def store_in_arango(self, documents: List[Dict[str, Any]], db_name: str, db_mode: str, force: bool = False) -> Dict[str, Any]:
        """
        Store enhanced documents in ArangoDB.
        
        Args:
            documents: List of documents with enhanced embeddings
            db_name: Name of the ArangoDB database
            db_mode: Database mode ('create' or 'append')
            force: Force recreation of collections even if they exist
            
        Returns:
            Dictionary with storage statistics
        """
        from src.storage.arango.connection import ArangoConnection
        from src.storage.arango.repository import ArangoRepository
        import uuid
        
        logger.info(f"Initializing ArangoDB connection for database '{db_name}'")
        
        # Create storage configuration
        storage_config = {
            "host": "localhost",
            "port": 8529,
            "username": "root",
            "password": "",  # Default empty password
            "database": db_name,
            "collection_prefix": "isne_",
            "use_vector_index": True,
            "vector_dimensions": 768  # Assuming ISNE output is same dimension
        }
        
        # Connect to ArangoDB
        try:
            # Initialize connection
            connection = ArangoConnection(
                host=storage_config["host"],
                port=storage_config["port"],
                username=storage_config["username"],
                password=storage_config["password"]
            )
            
            # Set up database
            if not connection.database_exists(db_name):
                logger.info(f"Creating database '{db_name}'")
                connection.create_database(db_name)
            
            # Set the current database
            connection.set_database(db_name)
            
            # Initialize repository
            node_collection = f"{storage_config['collection_prefix']}nodes"
            edge_collection = f"{storage_config['collection_prefix']}edges"
            graph_name = f"{storage_config['collection_prefix']}graph"
            
            repository = ArangoRepository(
                connection=connection,
                node_collection_name=node_collection,
                edge_collection_name=edge_collection,
                graph_name=graph_name,
                vector_dimensions=storage_config["vector_dimensions"],
                use_vector_index=storage_config["use_vector_index"]
            )
            
            # Handle create mode with force option
            if db_mode == "create" or force:
                # Delete graph and collections if they exist
                if force and connection.graph_exists(graph_name):
                    logger.info(f"Dropping existing graph '{graph_name}'")
                    connection.delete_graph(graph_name, drop_collections=True)
                elif force:
                    # Delete collections if they exist
                    if connection.collection_exists(node_collection):
                        logger.info(f"Dropping existing node collection '{node_collection}'")
                        connection.delete_collection(node_collection)
                    if connection.collection_exists(edge_collection):
                        logger.info(f"Dropping existing edge collection '{edge_collection}'")
                        connection.delete_collection(edge_collection)
                
                # Create collections and graph
                logger.info(f"Setting up collections and graph in '{db_mode}' mode")
                repository.setup_collections()
            
        except Exception as e:
            logger.error(f"Error initializing ArangoDB connection: {e}")
            raise
        
        # Store documents in ArangoDB
        logger.info("Storing documents in ArangoDB...")
        
        # Statistics counters
        stats = {
            "nodes_created": 0,
            "edges_created": 0,
            "errors": 0
        }
        
        # Process documents
        dataset_id = str(uuid.uuid4())
        try:
            for doc in documents:
                if "chunks" not in doc or not doc["chunks"]:
                    continue
                
                # Process each chunk as a node
                node_ids = []
                for chunk_idx, chunk in enumerate(doc["chunks"]):
                    # Skip chunks without embeddings
                    if "embedding" not in chunk or chunk["embedding"] is None:
                        continue
                    
                    # Prepare node data
                    node_data = {
                        "doc_id": doc["file_id"],
                        "chunk_id": chunk_idx,
                        "content": chunk.get("text", ""),
                        "metadata": {
                            "file_name": doc.get("file_name", ""),
                            "file_path": doc.get("file_path", ""),
                            "dataset_id": dataset_id,
                            "chunk_metadata": chunk.get("metadata", {})
                        }
                    }
                    
                    # Add embeddings
                    node_data["embedding"] = chunk["embedding"]
                    if "isne_embedding" in chunk:
                        node_data["isne_embedding"] = chunk["isne_embedding"]
                    
                    # Store node in ArangoDB
                    try:
                        node_id = repository.add_node(
                            content=node_data["content"],
                            metadata=node_data["metadata"],
                            embeddings={
                                "base": node_data["embedding"],
                                "isne": node_data.get("isne_embedding", node_data["embedding"])
                            }
                        )
                        node_ids.append(node_id)
                        stats["nodes_created"] += 1
                    except Exception as e:
                        logger.error(f"Error adding node: {e}")
                        stats["errors"] += 1
                        continue
                
                # Store the node_id to chunk mapping for later reference relationship processing
                chunk_to_node_id = {}
                for idx, node_id in enumerate(node_ids):
                    chunk_to_node_id[f"{doc['file_id']}_{idx}"] = node_id
                
                # Create edges between sequential chunks
                if len(node_ids) >= 2:
                    for i in range(len(node_ids) - 1):
                        try:
                            repository.add_edge(
                                from_id=node_ids[i],
                                to_id=node_ids[i+1],
                                relation_type="SEQUENTIAL",
                                weight=1.0,
                                metadata={"document_id": doc["file_id"]}
                            )
                            stats["edges_created"] += 1
                        except Exception as e:
                            logger.error(f"Error adding edge: {e}")
                            stats["errors"] += 1
                
                # Process code-specific relationships if this is a Python file
                # These relationships come from the PythonCodeChunker's metadata
                if doc.get("file_name", "").endswith(".py"):
                    for chunk_idx, chunk in enumerate(doc["chunks"]):
                        # Skip chunks without metadata
                        if "metadata" not in chunk or "references" not in chunk["metadata"]:
                            continue
                            
                        # Get source node ID
                        source_id = chunk_to_node_id.get(f"{doc['file_id']}_{chunk_idx}")
                        if not source_id:
                            continue
                            
                        # Process each reference (relationship) in the chunk metadata
                        for reference in chunk["metadata"]["references"]:
                            relation_type = reference.get("type")
                            target_chunk_id = reference.get("target")
                            weight = reference.get("weight", 0.8)
                            
                            # Skip invalid references
                            if not relation_type or not target_chunk_id:
                                continue
                                
                            # Get target node ID
                            target_id = chunk_to_node_id.get(target_chunk_id)
                            if not target_id:
                                continue
                                
                            # Create the edge representing the code relationship
                            try:
                                repository.add_edge(
                                    from_id=source_id,
                                    to_id=target_id,
                                    relation_type=relation_type,
                                    weight=weight,
                                    metadata={
                                        "document_id": doc["file_id"],
                                        "code_relationship": True,
                                        "source_type": chunk["metadata"].get("type"),
                                        "target_type": next((c["metadata"].get("type") for c in doc["chunks"] 
                                                          if f"{doc['file_id']}_{doc['chunks'].index(c)}" == target_chunk_id), None)
                                    }
                                )
                                stats["edges_created"] += 1
                            except Exception as e:
                                logger.error(f"Error adding code relationship edge: {e}")
                                stats["errors"] += 1
            
            logger.info(f"Storage complete: {stats['nodes_created']} nodes, {stats['edges_created']} edges created")
            
        except Exception as e:
            logger.error(f"Error during document storage: {e}")
            stats["errors"] += 1
        
        return stats
    
    def run_test(self, run_isne_training=False):
        """Run the multiprocessing pipeline test.
        
        Args:
            run_isne_training: Whether to run ISNE training after document processing
        """
        start_time = time.time()
        logger.info(f"Starting pipeline multiprocessing test with {self.max_workers} workers")
        
        # Find all supported files in the test directory
        supported_files = self.find_supported_files()
        
        # Limit the number of files if specified
        if self.num_files > 0 and len(supported_files) > self.num_files:
            logger.info(f"Limiting processing to {self.num_files} files (out of {len(supported_files)})")
            supported_files = supported_files[:self.num_files]
        
        # Create the document processing output directory
        doc_output_dir = self.output_dir / "documents"
        doc_output_dir.mkdir(exist_ok=True)
        
        # Start the worker pool and process documents
        with Pool(processes=self.max_workers) as pool:
            # Create batches for processing
            logger.info("Creating document processing batches...")
            batches = []
            for i in range(0, len(supported_files), self.batch_size):
                batch = supported_files[i:i+self.batch_size]
                batches.append(batch)
            
            logger.info(f"Created {len(batches)} batches of size {self.batch_size}")
            
            # Process each batch
            all_processed_docs = []
            batch_count = 0
            
            for batch in batches:
                batch_count += 1
                logger.info(f"Processing batch {batch_count}/{len(batches)} ({len(batch)} files)")
                
                # Create arguments for each file in the batch
                process_args = []
                for file_idx, file_path in enumerate(batch):
                    # Assign a unique worker ID based on the batch and file index
                    worker_id = ((batch_count - 1) % self.max_workers) * self.batch_size + file_idx % self.batch_size
                    process_args.append((file_path, self.config, worker_id))
                
                # Process the batch in parallel
                try:
                    batch_results = pool.map(process_document, process_args)
                    all_processed_docs.extend([r for r in batch_results if r is not None])
                except Exception as e:
                    logger.error(f"Error processing batch {batch_count}: {str(e)}", exc_info=True)
            
            # Collect statistics from all processed documents
            self.stats["total_files"] = len(all_processed_docs)
            total_chunks = sum(len(doc.get("chunks", [])) for doc in all_processed_docs)
            self.stats["total_chunks"] = total_chunks
            self.stats["total_embeddings"] = sum(len(doc.get("embeddings", [])) for doc in all_processed_docs)
            
            # Calculate total processing time by document type
            for doc in all_processed_docs:
                if "timing" in doc:
                    for key, value in doc["timing"].items():
                        if key in self.stats["processing_time"]:
                            self.stats["processing_time"][key] += value
                
                # Store individual file details
                self.stats["file_details"][doc["file_id"]] = {
                    "file_name": doc["file_name"],
                    "chunks": len(doc.get("chunks", [])),
                    "embeddings": len(doc.get("embeddings", [])),
                    "worker_id": doc["worker_id"],
                    "completed_at": doc.get("completed_at", "")
                }
        
        # Calculate total processing time
        self.stats["processing_time"]["total"] = time.time() - start_time
        
        # Save processed documents to JSON file
        docs_output_path = self.output_dir / "processed_documents.json"
        with open(docs_output_path, "w") as f:
            json.dump(all_processed_docs, f, indent=2)
        
        logger.info(f"Saved {len(all_processed_docs)} processed documents to {docs_output_path}")
        logger.info(f"Total processing time: {self.stats['processing_time']['total']:.2f}s")
        
        # Run ISNE training if specified
        if run_isne_training and all_processed_docs:
            logger.info("Running ISNE training on processed documents...")
            # Train the model (implementation depends on your ISNE orchestrator)
            try:
                from src.isne.trainer.training_orchestrator import ISNETrainingOrchestrator
                
                # Create output directory for the model
                model_dir = Path("./models/isne")
                model_dir.mkdir(exist_ok=True, parents=True)
                model_path = model_dir / "isne_model_latest.pt"
                
                # Run training
                trainer = ISNETrainingOrchestrator()
                trainer.train(
                    documents=all_processed_docs,
                    output_model_path=str(model_path),
                    epochs=10,  # Reduced for testing
                    learning_rate=0.001,
                    batch_size=16
                )
                
                logger.info(f"ISNE training completed. Model saved to {model_path}")
            except Exception as e:
                logger.error(f"Error during ISNE training: {str(e)}", exc_info=True)
        
        return self.stats
    
    def run_ingestion(self, model_path: str, output_dir: Optional[str] = None):
        """Run the ISNE ingestion pipeline with validation and alerts for creating a new datastore.
        
        This method implements a batch processing approach where all documents are processed first,
        then a complete graph of relationships is built, and finally ISNE enhancement is applied across
        the entire document set. This approach is optimized for initial database population but may not
        be suitable for incremental updates to an existing datastore.
        
        Args:
            model_path: Path to the trained ISNE model
            output_dir: Optional output directory for enhanced documents
        """
        start_time = time.time()
        logger.info(f"Starting ISNE ingestion pipeline with alert system")
        
        # Find all supported files in the test directory
        supported_files = self.find_supported_files()
        
        # Limit the number of files if specified
        if self.num_files > 0 and len(supported_files) > self.num_files:
            logger.info(f"Limiting processing to {self.num_files} files (out of {len(supported_files)})")
            supported_files = supported_files[:self.num_files]
        
        # Create output directories
        validation_dir = self.output_dir / "validation"
        validation_dir.mkdir(exist_ok=True, parents=True)
        
        # Start the worker pool and process documents
        with Pool(processes=self.max_workers) as pool:
            # Create batches for processing
            logger.info("Creating document processing batches...")
            batches = []
            for i in range(0, len(supported_files), self.batch_size):
                batch = supported_files[i:i+self.batch_size]
                batches.append(batch)
            
            logger.info(f"Created {len(batches)} batches of size {self.batch_size}")
            
            # Process each batch
            all_processed_docs = []
            batch_count = 0
            
            for batch in batches:
                batch_count += 1
                logger.info(f"Processing batch {batch_count}/{len(batches)} ({len(batch)} files)")
                
                # Create arguments for each file in the batch
                process_args = []
                for file_idx, file_path in enumerate(batch):
                    # Assign a unique worker ID based on the batch and file index
                    worker_id = ((batch_count - 1) % self.max_workers) * self.batch_size + file_idx % self.batch_size
                    process_args.append((file_path, self.config, worker_id))
                
                # Process the batch in parallel
                try:
                    batch_results = pool.map(process_document, process_args)
                    all_processed_docs.extend([r for r in batch_results if r is not None])
                except Exception as e:
                    logger.error(f"Error processing batch {batch_count}: {str(e)}", exc_info=True)
        
        # Save processed documents to JSON file
        docs_output_path = self.output_dir / "processed_documents.json"
        with open(docs_output_path, "w") as f:
            json.dump(all_processed_docs, f, indent=2)
        
        logger.info(f"Saved {len(all_processed_docs)} processed documents to {docs_output_path}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Document processing completed in {processing_time:.2f}s")
        
        # Apply ISNE model to enhance embeddings
        logger.info("Applying ISNE model to enhance document embeddings...")
        isne_start_time = time.time()
        
        try:
            from src.isne.trainer.training_orchestrator import ISNETrainingOrchestrator
            
            # Validate documents before ISNE application
            logger.info("Validating documents before ISNE application")
            pre_validation = validate_embeddings_before_isne(all_processed_docs)
            
            # Log validation results
            total_chunks = sum(len(doc.get("chunks", [])) for doc in all_processed_docs)
            chunks_with_base = pre_validation.get("chunks_with_base_embeddings", 0)
            logger.info(f"Pre-ISNE Validation: {len(all_processed_docs)} documents, {total_chunks} total chunks")
            logger.info(f"Found {chunks_with_base}/{total_chunks} chunks with base embeddings")
            
            # Generate alerts for validation issues
            missing_base = pre_validation.get("missing_base_embeddings", 0)
            if missing_base > 0:
                alert_level = AlertLevel.HIGH if missing_base > total_chunks * 0.2 else AlertLevel.MEDIUM
                self.alert_manager.alert(
                    message=f"Missing base embeddings detected in {missing_base} chunks",
                    level=alert_level,
                    source="isne_pipeline",
                    context={
                        "missing_count": missing_base,
                        "total_chunks": total_chunks,
                        "affected_chunks": pre_validation.get('missing_base_embedding_ids', [])
                    }
                )
            
            # Log validation results
            logger.info(f"Pre-ISNE Validation: {len(all_processed_docs)} documents, {sum(1 for doc in all_processed_docs if 'chunks' in doc and doc['chunks'])} with chunks")
            
            # Check for validation issues and create alerts if needed
            if pre_validation.get("missing_base_embeddings", 0) > 0:
                self.alert_manager.alert(
                    message=f"Missing base embeddings detected in {pre_validation['missing_base_embeddings']} chunks",
                    level=AlertLevel.MEDIUM,
                    source="isne_pipeline",
                    context={
                        "missing_count": pre_validation['missing_base_embeddings'],
                        "affected_chunks": pre_validation.get('missing_base_embedding_ids', [])
                    }
                )
            
            # Apply ISNE model if exists and is valid
            if not Path(model_path).exists():
                error_msg = f"ISNE model not found at {model_path}"
                logger.error(error_msg)
                self.alert_manager.alert(
                    message=error_msg,
                    level=AlertLevel.HIGH,
                    source="isne_pipeline",
                    context={"model_path": model_path}
                )
                return
            
            # Load model and apply to documents
            model = ISNETrainingOrchestrator.load_model(model_path)
            
            # Build graph from documents and get relationship data
            data, chunk_relationships = self.build_graph_from_documents(all_processed_docs)
            
            # Apply ISNE model to get enhanced embeddings
            enhanced_embeddings = model(data.x, data.edge_index, data.edge_attr)
            
            # Add enhanced embeddings back to documents
            node_idx_map = {}
            current_idx = 0
            
            # First pass to build node index mapping
            for doc in all_processed_docs:
                if "chunks" not in doc or not doc["chunks"]:
                    continue
                
                for chunk_idx, chunk in enumerate(doc["chunks"]):
                    if "embedding" not in chunk or not chunk["embedding"]:
                        continue
                    
                    chunk_id = f"{doc['file_id']}_{chunk_idx}"
                    node_idx_map[chunk_id] = current_idx
                    current_idx += 1
            
            # Second pass to add ISNE embeddings and relationships to chunks
            for doc in all_processed_docs:
                if "chunks" not in doc or not doc["chunks"]:
                    continue
                
                for chunk_idx, chunk in enumerate(doc["chunks"]):
                    if "embedding" not in chunk or not chunk["embedding"]:
                        continue
                    
                    chunk_id = f"{doc['file_id']}_{chunk_idx}"
                    if chunk_id in node_idx_map:
                        node_idx = node_idx_map[chunk_id]
                        # Add the enhanced embedding to the chunk
                        chunk["isne_embedding"] = enhanced_embeddings[node_idx].tolist()
                        
                        # Add relationship data if available for this chunk
                        if chunk_id in chunk_relationships:
                            chunk["relationships"] = chunk_relationships[chunk_id]
            
            # Validate documents after ISNE application
            logger.info("Validating documents after ISNE application")
            post_validation = validate_embeddings_after_isne(all_processed_docs, pre_validation)
            
            # Check for validation issues and generate alerts
            discrepancies = post_validation.get("discrepancies", {})
            total_discrepancies = post_validation.get("total_discrepancies", 0)
            
            if total_discrepancies > 0:
                alert_level = AlertLevel.HIGH if total_discrepancies > total_chunks * 0.1 else AlertLevel.MEDIUM
                self.alert_manager.alert(
                    message=f"Found {total_discrepancies} embedding discrepancies after ISNE application",
                    level=alert_level,
                    source="isne_pipeline",
                    context={
                        "discrepancies": discrepancies,
                        "total_discrepancies": total_discrepancies,
                        "expected_counts": post_validation.get("expected_counts", {}),
                        "actual_counts": post_validation.get("actual_counts", {})
                    }
                )
                
            # Log validation summary
            isne_count = post_validation.get("actual_counts", {}).get("isne_embeddings", 0)
            logger.info(f"Post-ISNE Validation: {isne_count}/{total_chunks} chunks have ISNE embeddings")
            
            # Check for validation issues after ISNE application
            discrepancies = post_validation.get("discrepancies", {})
            total_discrepancies = post_validation.get("total_discrepancies", 0)
            
            # Store validation results for reporting
            self.stats["validation"] = {
                "pre_validation": pre_validation,
                "post_validation": post_validation,
                "alerts": self.alert_manager.get_alerts()
            }
            
            # Define helper function to attach validation summaries to documents
            def attach_validation_summary(doc, validation_results):
                """Attach validation summary to a document.
                
                Args:
                    doc: Document to attach validation summary to
                    validation_results: Dictionary containing validation results
                """
                if "validation" not in doc:
                    doc["validation"] = {}
                
                # Combine pre and post validation results
                doc["validation"] = {
                    "pre_validation": pre_validation.get("summary", {}),
                    "post_validation": post_validation.get("summary", {}),
                    "has_discrepancies": post_validation.get("total_discrepancies", 0) > 0
                }
                
                # Add validation timestamp
                doc["validation"]["timestamp"] = datetime.now().isoformat()
            
            # Attach validation summary to each document
            for doc in all_processed_docs:
                attach_validation_summary(doc, {"pre": pre_validation, "post": post_validation})
            
            logger.info("Document enhancement with ISNE embeddings completed")
            
        except Exception as e:
            error_msg = f"Error during ISNE application: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.alert_manager.alert(
                message=error_msg,
                level=AlertLevel.CRITICAL,
                source="isne_pipeline",
                context={
                    "exception": str(e),
                    "stack_trace": traceback.format_exc()
                }
            )
            # Continue execution to save partial results
        
        # Calculate ISNE enhancement time
        isne_enhancement_time = time.time() - isne_start_time
        logger.info(f"ISNE enhancement completed in {isne_enhancement_time:.2f}s")
        
        # Save enhanced documents to JSON file
        json_start_time = time.time()
        isne_output_path = self.output_dir / "isne_enhanced_documents.json"
        with open(isne_output_path, "w") as f:
            json.dump(all_processed_docs, f, indent=2)
        
        # Save a sample of documents for inspection
        sample_docs = all_processed_docs[:min(5, len(all_processed_docs))]
        sample_path = self.output_dir / "isne_sample_documents.json"
        with open(sample_path, "w") as f:
            json.dump(sample_docs, f, indent=2)
        
        json_save_time = time.time() - json_start_time
        logger.info(f"Saved {len(all_processed_docs)} enhanced documents to {isne_output_path}")
        logger.info(f"Saved {len(sample_docs)} sample documents to {sample_path}")
        
        # Calculate total ingestion time
        total_ingestion_time = time.time() - start_time
        
        # Prepare stats for reporting
        ingestion_stats = {
            "total_documents": len(all_processed_docs),
            "total_chunks": self.stats["total_chunks"],
            "processing_time": processing_time,
            "isne_enhancement_time": isne_enhancement_time,
            "json_save_time": json_save_time,
            "total_ingestion_time": total_ingestion_time,
            "output_path": str(isne_output_path),
            "sample_path": str(sample_path),
            "alerts": self.alert_manager.get_alerts()
        }
        
        # Generate ingestion report
        self._generate_ingestion_report(ingestion_stats)
        
        return ingestion_stats
    
    def build_graph_from_documents(self, docs):
        """Build a graph from document chunks for ISNE processing.
        
        Args:
            docs: List of documents with chunks and embeddings
            
        Returns:
            Tuple of (PyTorch Geometric Data object, edge information dict)
        """
        import torch
        from torch_geometric.data import Data
        
        # Extract embeddings and metadata
        node_embeddings = []
        node_metadata = []
        node_model_types = []  # Track embedding model types
        edge_index_src = []
        edge_index_dst = []
        edge_attr = []
        
        # Node index mapping
        node_idx_map = {}
        current_idx = 0
        
        # Reverse mapping from node index to chunk ID for edge reconstruction
        idx_to_chunk_id = {}
        
        # Store edge information for saving back to documents
        edge_info = []
        
        # First pass: collect all nodes
        for doc in docs:
            if "chunks" not in doc or not doc["chunks"]:
                continue
            
            for chunk_idx, chunk in enumerate(doc["chunks"]):
                if "embedding" not in chunk or not chunk["embedding"]:
                    continue
                
                chunk_id = f"{doc['file_id']}_{chunk_idx}"
                node_idx_map[chunk_id] = current_idx
                idx_to_chunk_id[current_idx] = chunk_id
                
                # Add the base embedding
                node_embeddings.append(chunk["embedding"])
                
                # Add metadata
                metadata = {
                    "doc_id": doc["file_id"],
                    "chunk_idx": chunk_idx,
                    "text": chunk.get("text", "")[:100]  # Truncate for metadata
                }
                node_metadata.append(metadata)
                
                # Track embedding model type
                model_type = chunk.get("metadata", {}).get("embedding_model", "default")
                node_model_types.append(model_type)
                
                current_idx += 1
        
        # Second pass: build edges
        for doc in docs:
            if "chunks" not in doc or not doc["chunks"]:
                continue
            
            # Create sequential edges between chunks in the same document
            for i in range(len(doc["chunks"]) - 1):
                source_id = f"{doc['file_id']}_{i}"
                target_id = f"{doc['file_id']}_{i+1}"
                
                if source_id in node_idx_map and target_id in node_idx_map:
                    src_idx = node_idx_map[source_id]
                    tgt_idx = node_idx_map[target_id]
                    
                    edge_index_src.append(src_idx)
                    edge_index_dst.append(tgt_idx)
                    edge_attr.append([1.0])  # Sequential relationship weight
                    
                    # Store the edge information for later use
                    edge_info.append({
                        "source": source_id,
                        "target": target_id,
                        "type": "SEQUENTIAL",
                        "weight": 1.0
                    })
            
            # Add code-specific relationships from chunk metadata
            for chunk_idx, chunk in enumerate(doc["chunks"]):
                if "metadata" not in chunk or "references" not in chunk["metadata"]:
                    continue
                
                source_id = f"{doc['file_id']}_{chunk_idx}"
                if source_id not in node_idx_map:
                    continue
                
                # Process each reference (relationship)
                for reference in chunk["metadata"]["references"]:
                    target_id = reference.get("target")
                    if not target_id or target_id not in node_idx_map:
                        continue
                    
                    relation_type = reference.get("type", "REFERENCES")
                    weight = reference.get("weight", 0.8)
                    
                    # Add edge to the graph
                    src_idx = node_idx_map[source_id]
                    tgt_idx = node_idx_map[target_id]
                    
                    edge_index_src.append(src_idx)
                    edge_index_dst.append(tgt_idx)
                    edge_attr.append([weight])  # Use the relationship weight
                    
                    # Store the edge information for later use
                    edge_info.append({
                        "source": source_id,
                        "target": target_id,
                        "type": relation_type,
                        "weight": weight
                    })
        
        # Convert lists to tensors
        x = torch.tensor(node_embeddings, dtype=torch.float)
        edge_index = torch.tensor([edge_index_src, edge_index_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # Create mapping from chunk IDs to their relationships
        chunk_relationships = {}
        for edge in edge_info:
            source_id = edge["source"]
            if source_id not in chunk_relationships:
                chunk_relationships[source_id] = []
            chunk_relationships[source_id].append(edge)
        
        logger.info(f"Built graph with {x.shape[0]} nodes and {edge_index.shape[1]} edges")
        return data, chunk_relationships
    
    def _generate_ingestion_report(self, stats: Dict[str, Any]):
        """Generate a comprehensive report for the ingestion process."""
        logger.info("\n=== Ingestion Pipeline Report ===")
        
        logger.info("\nProcessing Statistics:")
        logger.info(f"  Total Documents:       {stats['total_documents']}")
        logger.info(f"  Document Processing:   {stats['processing_time']:.2f}s")
        logger.info(f"  ISNE Enhancement:      {stats['isne_enhancement_time']:.2f}s")
        logger.info(f"  JSON Saving:           {stats['json_save_time']:.2f}s")
        logger.info(f"  Total Ingestion Time:  {stats['total_ingestion_time']:.2f}s")
        
        # Add validation and alert information to report
        if "alerts" in stats and stats["alerts"]:
            alert_counts = {
                "LOW": 0,
                "MEDIUM": 0,
                "HIGH": 0,
                "CRITICAL": 0
            }
            
            for alert in stats["alerts"]:
                # Handle both dictionary alerts and Alert objects
                if isinstance(alert, dict) and "level" in alert:
                    level = alert["level"]
                elif hasattr(alert, "level"):  # Alert object
                    level = alert.level
                elif hasattr(alert, "to_dict"):
                    level = alert.to_dict().get("level")
                else:
                    logger.warning(f"Unknown alert format: {type(alert)}, skipping")
                    continue
                    
                if level in alert_counts:
                    alert_counts[level] += 1
            
            logger.info("\nAlert Summary:")
            logger.info(f"  LOW:      {alert_counts['LOW']}")
            logger.info(f"  MEDIUM:   {alert_counts['MEDIUM']}")
            logger.info(f"  HIGH:     {alert_counts['HIGH']}")
            logger.info(f"  CRITICAL: {alert_counts['CRITICAL']}")
            
            # Check if there are critical or high alerts
            if alert_counts["CRITICAL"] > 0 or alert_counts["HIGH"] > 0:
                logger.warning(f"⚠️ WARNING: {alert_counts['CRITICAL'] + alert_counts['HIGH']} critical/high alerts were generated.")
                logger.warning(f"Review alerts in {self.alert_dir}")
        
        logger.info("\nOutput Files:")
        logger.info(f"  All Documents:         {stats['output_path']}")
        logger.info(f"  Sample Documents:      {stats['sample_path']}")
        logger.info(f"  Alert Logs:            {self.alert_dir}")
        
        logger.info("\nIngestion Pipeline Completed Successfully!")
        
        # Convert to JSON serializable format
        serializable_stats = copy.deepcopy(stats)
        
        # Convert Alert objects to dictionaries for JSON serialization
        if "alerts" in serializable_stats:
            alerts_json = []
            for alert in serializable_stats["alerts"]:
                if hasattr(alert, "to_dict"):
                    alerts_json.append(alert.to_dict())
                elif isinstance(alert, dict):
                    alerts_json.append(alert)
                else:
                    # Skip non-serializable objects
                    logger.warning(f"Skipping non-serializable alert: {type(alert)}")
            serializable_stats["alerts"] = alerts_json
        
        # Save report as JSON
        report_path = self.output_dir / "ingestion_report.json"
        with open(report_path, "w") as f:
            json.dump(serializable_stats, f, indent=2)
        logger.info(f"Detailed ingestion report saved to {report_path}")

# Initialize multiprocessing
def init_mp():
    """Initialize multiprocessing environment."""
    # Configure for multiprocessing on Windows if necessary
    if sys.platform == 'win32':
        mp.set_start_method('spawn', force=True)

def main():
    """
    Run the pipeline multiprocessing test as a standalone script.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='HADES-PathRAG Pipeline Multiprocessing Test')
    
    # Mode selection arguments
    mode_group = parser.add_argument_group('Mode Selection')
    mode_group.add_argument('--mode', type=str, choices=['process', 'train', 'ingest'], default='process',
                        help='Pipeline mode: process (document processing only), train (ISNE training), ingest (ISNE ingestion)')
    
    # Common arguments
    common_group = parser.add_argument_group('Common Options')
    common_group.add_argument('--workers', type=int, default=4,
                        help='Number of worker processes')
    common_group.add_argument('--files', type=int, default=50,
                        help='Maximum number of files to process')
    common_group.add_argument('--data-dir', type=str, default='./test3',
                        help='Directory containing test data')
    common_group.add_argument('--output-dir', type=str, default='./test-output/isne-training-dataset',
                        help='Directory for test output')
    
    # Legacy support
    common_group.add_argument('--run-isne', action='store_true',
                        help='[Legacy] Run ISNE training directly (same as --mode=train)')
    
    # Ingestion mode arguments
    ingest_group = parser.add_argument_group('Ingestion Options')
    ingest_group.add_argument('--model-path', type=str, default='./models/isne/isne_model_latest.pt',
                        help='Path to trained ISNE model (for ingestion mode)')
    ingest_group.add_argument('--output-file', type=str,
                        help='Custom output file path for enhanced documents (for ingestion mode)')
    ingest_group.add_argument('--alert-threshold', type=str, choices=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
                        default='MEDIUM', help='Minimum alert level to log (LOW, MEDIUM, HIGH, CRITICAL)')
    
    args = parser.parse_args()
    
    # Create the pipeline tester
    test = PipelineMultiprocessTester(
        test_data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_files=args.files,
        max_workers=args.workers,
        batch_size=8,  # Default batch size
        alert_threshold=args.alert_threshold
    )
    
    # Determine which mode to run
    mode = args.mode
    # Support legacy --run-isne flag
    if args.run_isne and mode == 'process':
        mode = 'train'
    
    # Run the appropriate pipeline mode
    if mode == 'train':
        logger.info("Running in ISNE training mode")
        test.run_test(run_isne_training=True)
    elif mode == 'ingest':
        logger.info("Running in ISNE ingestion mode")
        test.run_ingestion(
            model_path=args.model_path,
            output_dir=args.output_file
        )
    else:  # Default to 'process' mode
        logger.info("Running in document processing mode")
        test.run_test(run_isne_training=False)

if __name__ == "__main__":
    # Configure for multiprocessing on Windows
    mp.set_start_method('spawn', force=True)
    main()
