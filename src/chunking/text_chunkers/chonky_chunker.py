from __future__ import annotations

"""Chonky text chunker implementation.

This module implements a chunker based on Chonky, which splits text into
semantic paragraphs using a transformer model. The chunker maintains semantic coherence
while creating appropriately sized chunks for embedding generation.

The module provides both direct chunking functions and integration with the Haystack
model engine for accelerated processing.

Example usage:
    # Basic usage
    chunked_document = chunk_text(document)
    
    # With custom options
    chunked_document = chunk_text(
        document, 
        max_tokens=1024,
        output_format="json"
    )
    
    # Batch processing
    from src.chunking.text_chunkers.chonky_batch import chunk_document_batch
    chunked_docs = chunk_document_batch(documents)

This module is a core component of the document processing pipeline and is used
by the ingest orchestrator to prepare documents for embedding generation.

Semantic text chunker using [Chonky](https://github.com/mithril-security/chonky).

This implementation integrates Chonky with the Haystack model engine for efficient model loading
and management. The model is lazily initialized and managed by the Haystack engine.

The helper ``chunk_text`` mirrors the signature of ``chunk_python_code`` so the
`PreprocessorManager` can treat code and text files uniformly.

Features:
- Uses the high-quality mirth/chonky_modernbert_large_1 model
- Token-aware chunking with configurable parameters
- Efficient model loading through Haystack model engine
- Parallel processing of multiple documents
- Direct support for Pydantic model instances
"""

import logging
import os
import json
import uuid
import hashlib
import re
import time
import torch
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Callable, cast, Type, TypeVar, Tuple, Generator
from functools import lru_cache
from multiprocessing.pool import ThreadPool
from contextlib import contextmanager
from pydantic import BaseModel

# Import the schema types
from src.schemas.documents.base import ChunkMetadata
from src.schemas.common.enums import DocumentType, SchemaVersion

logger = logging.getLogger(__name__)

# Flag to track if required libraries are available
_TRANSFORMERS_AVAILABLE = False
_CHONKY_AVAILABLE = False
_ENGINE_AVAILABLE = False

# Define a flag to track if transformers is available
_TRANSFORMERS_AVAILABLE = False

# Try to import from transformers, but don't store the actual AutoTokenizer class yet
try:
    import transformers
    # Mark transformers as available if import succeeds
    _TRANSFORMERS_AVAILABLE = True
    logger.info("transformers library is available")
except ImportError as e:
    # Transformers is not available
    logger.warning(f"transformers not available: {e}")

# Define a function to get AutoTokenizer when needed
def get_auto_tokenizer() -> Optional[Any]:
    """Get the AutoTokenizer class from transformers if available.
    
    Returns:
        Optional[Any]: The AutoTokenizer class if available, None otherwise
    """
    if _TRANSFORMERS_AVAILABLE:
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer
        except ImportError:
            logger.warning("Failed to import AutoTokenizer")
            return None
    return None

# Try to import chonky
try:
    from chonky import ParagraphSplitter
    _CHONKY_AVAILABLE = True
except ImportError:
    _CHONKY_AVAILABLE = False
    raise ImportError(
        "Package 'chonky' is required for semantic text chunking.\n"
        "Install with:  pip install chonky"
    )

# Import Haystack model engine
from src.model_engine.engines.haystack import HaystackModelEngine
from src.config.chunker_config import get_chunker_config

# Define our own document models to avoid import issues
class BaseDocument(BaseModel):
    """Base document class for document processing."""
    content: str
    path: str
    type: str = "text"
    id: str = ""
    chunks: List[Dict[str, Any]] = []

# NOTE: ChunkMetadata is now imported from src.schemas.documents.base

# Define our document types
from typing import Type, TypeVar, cast, Any as TypeAny, Union as TypeUnion

# Initial type definition for document types
DocumentBaseType = TypeUnion[Dict[str, TypeAny], BaseDocument]

# Define a type for DocumentSchema
class DocumentSchemaBase(BaseModel):
    """Base document schema for document processing."""
    content: str
    path: str
    type: str = "text"
    id: str = ""
    chunks: List[Dict[str, TypeAny]] = []

# Try to import DocumentSchema from schema module
try:
    from src.schemas.documents.base import DocumentSchema as ImportedDocumentSchema
    # Use the imported DocumentSchema
    DocumentSchema = ImportedDocumentSchema
except ImportError:
    # Use our own DocumentSchema if not available
    class DocumentSchema(DocumentSchemaBase):  # type: ignore
        pass

# Final type definition including DocumentSchema
DocumentSchemaType = TypeUnion[Dict[str, TypeAny], BaseDocument, DocumentSchema]

logger = logging.getLogger(__name__)

# Initialize global model engine
_MODEL_ENGINE: Optional[HaystackModelEngine] = None
_TOKENIZER = None
_SPLITTER_CACHE: Dict[str, ParagraphSplitter] = {}
_ENGINE_AVAILABLE = False


def get_model_engine() -> Optional[HaystackModelEngine]:
    """Get or initialize the Haystack model engine.
    
    Returns:
        The initialized model engine, or None if initialization fails
    """
    global _MODEL_ENGINE, _ENGINE_AVAILABLE
    
    # Return existing engine if already initialized and running
    if _MODEL_ENGINE is not None and _ENGINE_AVAILABLE:
        return _MODEL_ENGINE
    
    # Default to engine not available    
    _ENGINE_AVAILABLE = False
    
    try:
        # Create a new model engine instance
        logger.info("Creating Haystack model engine instance...")
        _MODEL_ENGINE = HaystackModelEngine()
        
        # Get chunker configuration
        config = get_chunker_config('chonky')
        
        # Check for device configuration in pipeline settings
        try:
            # Import here to avoid circular imports
            from src.config.config_loader import get_component_device
            
            # Get device for chunking component
            device = get_component_device('chunking')
            if device:
                logger.info(f"Using configured device for chunking: {device}")
                # We'll use this device when loading models with the engine
            else:
                logger.info("No specific device configured for chunking, using default")
        except Exception as e:
            logger.warning(f"Could not get component device settings: {e}")
            # Continue with default device settings
        
        # Check if auto-start is enabled
        auto_start = config.get('auto_start_engine', True)
        if not auto_start:
            logger.info("Auto-start disabled, returning unstarted engine")
            return _MODEL_ENGINE
        
        # Try to start the engine with simple error handling
        logger.info("Starting Haystack model engine...")
        # First, initialize the success flag to False
        model_engine_started = False
        
        try:
            # Start the engine and capture the result
            start_result = _MODEL_ENGINE.start()
            try:
                # Try to get the model engine client
                retries = 0
                max_retries = 5
                model_engine_started = False
                
                # Loop until we have a client or exceed max retries
                while retries < max_retries and not model_engine_started:
                    # Check if the client exists, which indicates the engine is likely ready
                    client_exists = hasattr(_MODEL_ENGINE, "client") and _MODEL_ENGINE.client is not None
                    
                    if client_exists:
                        model_engine_started = True
                        logger.info("Model engine client initialized successfully")
                    else:
                        # Print waiting message and sleep if client isn't available
                        logger.info(f"Waiting for model engine startup (attempt {retries+1}/{max_retries})")
                        time.sleep(1)
                        retries += 1
                
                # Now check if we have a working engine
                if model_engine_started:
                    # The engine seems to be started, try to ping it
                    logger.info("Attempting to ping Haystack model engine")
                    
                    # Check if client exists before attempting to ping
                    has_client = hasattr(_MODEL_ENGINE, "client") and _MODEL_ENGINE.client is not None
                    
                    if has_client:
                        # Simplified approach to ping the engine with separate methods
                        logger.debug("Attempting to ping model engine client")
                        
                        # Define helper to check ping in a separate function - this avoids nesting issues
                        def check_ping_connection() -> bool:
                            """Helper function to check ping connection with proper type handling
                            
                            Returns:
                                bool: True if connection is successful, False otherwise
                            """
                            # First check if model engine exists at all
                            if _MODEL_ENGINE is None:
                                logger.warning("Model engine is None")
                                return False
                                
                            # Check if client exists
                            engine_client = getattr(_MODEL_ENGINE, 'client', None)
                            if engine_client is None:
                                logger.warning("Haystack model engine client is None")
                                return False
                                
                            # Client exists but no ping method
                            if not hasattr(engine_client, 'ping'):
                                logger.warning("Haystack model engine client has no ping method")
                                return False
                                
                            # Try to ping
                            try:
                                # Use the client reference we've verified
                                ping_result = engine_client.ping()
                                
                                # Check ping result
                                if ping_result == "pong":
                                    logger.info("Successfully connected to Haystack model engine")
                                    return True
                                else:
                                    logger.warning(f"Unexpected response from Haystack engine: {ping_result}")
                                    return False
                            except Exception as e:
                                logger.warning(f"Error pinging Haystack model engine: {e}")
                                return False
                        
                        # Use the helper function to set the engine availability
                        try:
                            _ENGINE_AVAILABLE = check_ping_connection()
                        except Exception as e:
                            logger.warning(f"Error checking ping connection: {e}")
                            _ENGINE_AVAILABLE = False
                    else:
                        logger.warning("Haystack model engine client is None after startup")
            except Exception as e:
                import traceback
                logger.warning(f"Error starting Haystack model engine: {e}\n{traceback.format_exc()}")
        except Exception as e:
            import traceback
            logger.warning(f"Error initializing Haystack model engine: {e}\n{traceback.format_exc()}")
        
        # We've already attempted to ping in the previous block
        # No need for a second ping attempt
            
        # Return the engine
        return _MODEL_ENGINE
        
    except Exception as e:
        logger.warning(f"Error initializing Haystack model engine: {e}")
        _MODEL_ENGINE = None
        return None


def _get_adjusted_device(device: str) -> str:  # pragma: no cover
    """Adjust the requested device string to match the runtime environment.

    When ``CUDA_VISIBLE_DEVICES`` is set, the indices of visible GPUs are
    remapped such that the first visible GPU becomes ``cuda:0`` regardless of
    its original physical index.  Requesting the original index would raise a
    *invalid device ordinal* error.  This helper ensures that an out-of-range
    CUDA device is mapped to the highest available visible device or, if no
    GPUs are visible, falls back to CPU.

    The logic mirrors ``ModernBERTEmbeddingAdapter._get_adjusted_device`` so
    that all components follow the same rules.

    Args:
        device: The originally requested device string (e.g. ``"cuda:1"``).

    Returns:
        A device string that is safe to use with PyTorch in the current
        process.
    """
    if not device.startswith("cuda"):
        # Covers both "cpu" and other possible values like "mps"
        return device

    try:
        import torch  # Local import to avoid mandatory dependency when GPU not used

        visible_count = torch.cuda.device_count()
        if visible_count == 0:
            logger.warning("No CUDA devices visible. Falling back to CPU.")
            return "cpu"

        # Extract requested index (defaults to 0 if just "cuda")
        requested_idx = 0
        if ":" in device:
            try:
                requested_idx = int(device.split(":")[1])
            except ValueError:
                logger.warning(f"Unable to parse CUDA device index from '{device}'. Using cuda:0 instead.")
                requested_idx = 0

        if requested_idx >= visible_count:
            logger.warning(
                f"Requested device {device} is not available under current CUDA_VISIBLE_DEVICES setting. "
                f"Only {visible_count} device(s) visible. Using cuda:{visible_count - 1} instead."
            )
            return f"cuda:{visible_count - 1}"

        # Requested index is within visible range
        return f"cuda:{requested_idx}"

    except Exception as e:  # pragma: no cover
        logger.warning(f"Could not validate CUDA device availability: {e}. Falling back to original device '{device}'.")
        return device


def _get_splitter_with_engine(model_id: str, device: str = "cuda") -> ParagraphSplitter:
    """Get a Chonky ParagraphSplitter using the Haystack model engine.
    
    This function ensures the model is loaded in the Haystack engine and
    returns a ParagraphSplitter configured to use it.
    
    Args:
        model_id: The model ID to load
        device: The device to use (e.g., "cuda", "cpu")
        
    Returns:
        A configured ParagraphSplitter instance
        
    Raises:
        RuntimeError: If the model fails to load
    """
    # Check for device configuration in pipeline settings
    try:
        # Import here to avoid circular imports
        from src.config.config_loader import load_pipeline_config
        
        # Try to load pipeline configuration to get device settings
        pipeline_config = load_pipeline_config()
        if pipeline_config and 'gpu_execution' in pipeline_config and pipeline_config['gpu_execution'].get('enabled', False):
            if 'chunking' in pipeline_config['gpu_execution'] and 'device' in pipeline_config['gpu_execution']['chunking']:
                configured_device = pipeline_config['gpu_execution']['chunking']['device']
                if configured_device:
                    logger.info(f"Using configured device from pipeline config: {configured_device}")
                    device = configured_device
    except Exception as e:
        logger.warning(f"Could not load pipeline config for device settings: {e}")
        # Continue with provided or default device
    
    # Adjust device to match runtime environment
    device = _get_adjusted_device(device)
    
    global _SPLITTER_CACHE
    
    # Check if we have a cached splitter for this model
    cache_key = f"{model_id}_{device}"
    if cache_key in _SPLITTER_CACHE:
        logger.info(f"Using cached splitter for {model_id} on {device}")
        return _SPLITTER_CACHE[cache_key]
    
    # Get chunker configuration
    config = get_chunker_config('chonky')
    
    # Check if caching is enabled
    use_cache = config.get('use_cache', True)
    
    # Create the splitter first using Chonky's built-in model loading
    # This approach works better with the current version of Chonky 
    # which expects to control the model loading process
    try:
        # Make sure we have model ID first (this ensures we have a valid model_id in all code paths)
        if not model_id:
            model_id = "mirth/chonky_modernbert_large_1"  # Default Chonky model
            
        # Now verify that transformers is available
        # This is a critical requirement for Chonky to work properly
        if not _TRANSFORMERS_AVAILABLE:
            msg = f"transformers library is not available - cannot use model {model_id}"
            logger.warning(msg)
            # Raise an ImportError to be caught by the outer exception handler
            raise ImportError(msg)
            
        # Get the AutoTokenizer class
        AutoTokenizer = get_auto_tokenizer()
        if AutoTokenizer is None:
            msg = f"Failed to import AutoTokenizer - cannot use model {model_id}"
            logger.warning(msg)
            # Raise an ImportError to be caught by the outer exception handler
            raise ImportError(msg)
        
        logger.info(f"Creating ParagraphSplitter with model {model_id} on {device}")
        
        try:
            # Attempt to create the splitter
            splitter_config = {"device": device}
            splitter = ParagraphSplitter(model_id, **splitter_config)
            logger.info(f"Successfully created ParagraphSplitter with model {model_id}")
            
            # Cache the splitter if caching is enabled
            if use_cache:
                _SPLITTER_CACHE[cache_key] = splitter
                logger.info(f"Cached splitter for {model_id}")
            
            return splitter
            
        except Exception as inner_e:
            # Handle specific ParagraphSplitter creation errors
            logger.error(f"Error initializing ParagraphSplitter with model {model_id}: {inner_e}")
            # Re-raise to be caught by outer exception handler
            raise
    
    except Exception as e:
        # This outer exception handler catches both ImportError from earlier checks
        # and any exceptions re-raised from the inner try-except block
        logger.error(f"Failed to create ParagraphSplitter with model {model_id}: {e}")
        logger.error(f"Falling back to simple chunking without semantic model")
        raise RuntimeError(f"Failed to create semantic chunker with model {model_id}: {e}")


def _hash_path(path: str) -> str:
    """Create a short hash from a file path for chunk IDs."""
    return hashlib.md5(path.encode()).hexdigest()[:8]


@contextmanager
def ensure_model_engine() -> Generator[Optional[HaystackModelEngine], None, None]:
    """Context manager to ensure model engine is started and properly cleaned up."""
    global _ENGINE_AVAILABLE
    if not _ENGINE_AVAILABLE:
        logger.warning("Model engine is not available. Using fallback methods.")
        yield None
        return
        
    engine = get_model_engine()
    try:
        yield engine
    finally:
        # We don't stop the engine here to allow reuse
        pass


def chunk_text(
    content: str, 
    doc_id: Optional[str] = None,
    path: str = "unknown",
    doc_type: str = "text",
    max_tokens: int = 2048,
    output_format: str = "document",
    model_id: str = "mirth/chonky_modernbert_large_1",  
    device: Optional[str] = None
) -> Union[Dict[str, Any], BaseDocument, DocumentSchema]:
    """Chunk a text document into semantically coherent paragraphs.
    
    Args:
        content: Text content to chunk
        doc_id: Document ID (will be auto-generated if None)
        path: Path to the document
        doc_type: Type of document
        max_tokens: Maximum tokens per chunk
        output_format: Output format, one of "document", "json", "dict"
        model_id: ID of the model to use for chunking (default is Chonky's model)
        
    Returns:
        Chunked document in the specified format
    """
    # Start detailed logging
    doc_id_log = doc_id if doc_id else 'new document'
    logger.info(f"Starting chunk_text for document: {doc_id_log}")
    
    # Check if Chonky is available
    if not _CHONKY_AVAILABLE:
        logger.warning("Chonky package not available. Using basic fallback chunking.")
    
    if not content or not content.strip():
        logger.warning("Empty content provided to chunk_text")
        if output_format == "document":
            return DocumentSchema(
                id=doc_id or str(uuid.uuid4()),
                content="",
                source=path,
                document_type=DocumentType(doc_type),
                schema_version=SchemaVersion.V2,
                title=None,
                author=None,
                created_at=datetime.now(),
                updated_at=None,
                metadata={},
                embedding=None,
                embedding_model=None,
                chunks=[],  # Empty chunks for empty content
                tags=[]
            )
        return {
            "id": doc_id or str(uuid.uuid4()),
            "content": "",
            "path": path,
            "type": doc_type,
            "chunks": []  # Empty chunks for empty content
        }
    
    # Ensure we have a document ID
    if doc_id is None:
        # Generate a deterministic ID based on content
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        doc_id = f"doc_{content_hash}"
    
    # Get chunker configuration
    config = get_chunker_config('chonky')
    logger.info(f"Loaded chunker config: {config}")
    
    # Check if a device was explicitly passed to the function
    if device is None:
        # Try to get device from pipeline config
        try:
            from src.config.config_loader import get_component_device
            configured_device = get_component_device('chunking')
            if configured_device:
                device = configured_device
                logger.info(f"Using device from pipeline config: {device}")
            else:
                # Fall back to chunker config
                device = config.get('device', 'cuda')
                logger.info(f"Using device from chunker config: {device}")
        except Exception as e:
            # If we can't get the pipeline config, fall back to the chunker config
            device = config.get('device', 'cuda')
            logger.warning(f"Error getting pipeline config device, using default: {device}. Error: {e}")
    else:
        logger.info(f"Using explicitly provided device: {device}")
        
    # Adjust device to match runtime environment
    device = _get_adjusted_device(device)
    
    logger.info(f"Using device: {device}")
    
    # Get the configured max tokens or use the provided value
    max_tokens = config.get('max_tokens', max_tokens)
    logger.info(f"Using max tokens: {max_tokens}")
    
    # Check if we have a working model engine
    logger.info(f"Engine available status: {_ENGINE_AVAILABLE}")
    if not _ENGINE_AVAILABLE:
        # Try to initialize the engine one more time
        logger.info("Attempting to initialize model engine again...")
        engine = get_model_engine()
        logger.info(f"Model engine after initialization attempt: {engine}, Available: {_ENGINE_AVAILABLE}")
    
    # Determine if we should use the model engine
    use_engine = config.get('use_model_engine', True) and _ENGINE_AVAILABLE
    logger.info(f"Using model engine: {use_engine}")
    
    # Initialize chunks list
    chunks = []
    
    try:
        if use_engine:
            # Try to use the model engine first
            if not _ENGINE_AVAILABLE:
                logger.warning("Model engine is not available. Using fallback methods.")
            
            with ensure_model_engine() as engine:
                # Log that we're using the model engine
                logger.info(f"Using model engine with model_id: {model_id} on device: {device}")
                
                # Get the splitter - this is where we pass the model_id to use
                splitter = _get_splitter_with_engine(model_id, device)
                
                # Split the document using the loaded model.  ParagraphSplitter is
                # callable and returns an **iterator of strings**, not an object
                # exposing a ``split`` method.  We therefore invoke the splitter
                # directly and materialise the result into a list so that we can
                # calculate offsets and token counts.
                logger.info(f"Splitting document with model: {model_id}")
                paragraphs: List[str] = list(splitter(content))
                logger.info(f"Split document into {len(paragraphs)} semantic paragraphs")
                
                # Create chunk dictionaries from each paragraph.
                cumulative_length: int = 0
                for i, para_text in enumerate(paragraphs):
                    chunk_id = f"{doc_id}_p{i}"
                    
                    start_offset = cumulative_length
                    end_offset = start_offset + len(para_text)
                    cumulative_length = end_offset + 2  # +2 for the removed "\n\n" (approx.)
                    
                    chunk = {
                        "id": chunk_id,
                        "parent": doc_id,
                        "parent_id": doc_id,  # Required by schema
                        "path": path,
                        "type": doc_type,
                        "content": para_text,
                        # Chonky does not expose contextual metadata yet; keep empty.
                        "overlap_context": {
                            "pre": "",
                            "post": "",
                            "position": i,
                            "total": len(paragraphs),
                        },
                        "symbol_type": "paragraph",
                        "name": f"paragraph_{i}",
                        "chunk_index": i,
                        "start_offset": start_offset,
                        "end_offset": end_offset,
                        "line_start": 0,
                        "line_end": 0,
                        "token_count": len(para_text.split()),
                        "content_hash": hashlib.md5(para_text.encode()).hexdigest(),
                        "embedding": None,
                    }
                    
                    chunks.append(chunk)
        else:
            # Fallback to basic paragraph splitting
            logger.warning("Model engine not available, using fallback chunking")
            
            # Split by paragraphs (double newlines)
            paragraphs = [p for p in content.split("\n\n") if p.strip()]
            
            # Create chunks from paragraphs
            cumulative_length = 0
            for i, para in enumerate(paragraphs):
                # Generate a unique ID for the chunk
                chunk_id = f"{doc_id}_p{i}"
                
                # Calculate offsets
                start_offset = cumulative_length
                end_offset = start_offset + len(para)
                cumulative_length = end_offset + 2  # +2 for the removed "\n\n"
                
                # Create the chunk
                chunk = {
                    "id": chunk_id,
                    "parent": doc_id,
                    "parent_id": doc_id,  # Required by schema
                    "path": path,
                    "type": doc_type,
                    "content": para,
                    "overlap_context": {
                        "pre": "",
                        "post": "",
                        "position": i,
                        "total": len(paragraphs)
                    },
                    "symbol_type": "paragraph",
                    "name": f"paragraph_{i}",
                    "chunk_index": i,  # Required by schema
                    "start_offset": start_offset,  # Required by schema
                    "end_offset": end_offset,  # Required by schema
                    "line_start": 0,
                    "line_end": 0,
                    "token_count": len(para.split()),
                    "content_hash": hashlib.md5(para.encode()).hexdigest(),
                    "embedding": None
                }
                
                chunks.append(chunk)
    except Exception as e:
        logger.error(f"Error chunking document: {e}")
        # Fallback to basic paragraph splitting
        logger.warning("Using fallback chunking due to error")
        
        # Split by paragraphs (double newlines) and handle empty paragraphs
        # First replace multiple newlines with a single newline to normalize
        normalized_content = re.sub(r'\n\n+', '\n\n', content)
        
        # Split by double newlines
        paragraphs = [p.strip() for p in normalized_content.split('\n\n') if p.strip()]
        
        # If we have no paragraphs after splitting, fall back to single chunk
        if not paragraphs:
            paragraphs = [content.strip() if content.strip() else ""]
        
        # Create chunks from paragraphs
        
        # Create chunks from paragraphs
        for i, para in enumerate(paragraphs):
            # Generate a unique ID for the chunk
            chunk_id = f"{doc_id}_p{i}"
            
            # Create the chunk
            chunk = {
                "id": chunk_id,
                "parent": doc_id,
                "path": path,
                "type": doc_type,
                "content": para,
                "overlap_context": {
                    "pre": "",
                    "post": "",
                    "position": i,
                    "total": len(paragraphs)
                },
                "symbol_type": "paragraph",
                "name": f"paragraph_{i}",
                "line_start": 0,
                "line_end": 0,
                "token_count": len(para.split()),
                "content_hash": hashlib.md5(para.encode()).hexdigest(),
                "embedding": None
            }
            
            chunks.append(chunk)
    
    # Handle different output formats
    if output_format == "document":
        # Create a DocumentSchema
        doc_schema = DocumentSchema(
            id=doc_id,
            content=content,
            source=path,
            document_type=DocumentType(doc_type),
            schema_version=SchemaVersion.V2,
            title=None,
            author=None,
            created_at=datetime.now(),
            updated_at=None,
            metadata={},
            embedding=None,
            embedding_model=None,
            chunks=[],  # We'll convert the chunks to ChunkMetadata below
            tags=[]
        )
        
        # Convert chunks to ChunkMetadata objects
        chunk_metadata_list: List[ChunkMetadata] = []
        for chunk_dict in chunks:
            # Extract chunk attributes with proper type conversion
            chunk_id = str(chunk_dict.get("id", f"{doc_id}_chunk_{len(chunk_metadata_list)}"))
            content = str(chunk_dict.get("content", ""))
            chunk_type_raw = chunk_dict.get("type", "text")
            chunk_type = str(chunk_type_raw) if chunk_type_raw is not None else "text"
            start_offset = 0  # Approximate
            end_offset = len(content) if content else 0
            
            # Create chunk metadata
            chunk_metadata = ChunkMetadata(
                start_offset=start_offset,
                end_offset=end_offset,
                chunk_type=chunk_type,
                chunk_index=len(chunk_metadata_list),
                parent_id=doc_id,
                context_before=None,
                context_after=None,
                metadata={
                    "symbol_type": chunk_dict.get("symbol_type", "paragraph"),
                    "name": chunk_dict.get("name", ""),
                    "token_count": chunk_dict.get("token_count", 0),
                    "content_hash": chunk_dict.get("content_hash", ""),
                    "content": content,  # Store content as metadata since ChunkMetadata doesn't have content field
                }
            )
            chunk_metadata_list.append(chunk_metadata)
        
        # Set the chunks
        if hasattr(doc_schema, "chunks"):
            setattr(doc_schema, "chunks", chunk_metadata_list)
            
        return doc_schema
    elif output_format == "json":
        # Format as a structured document with chunks
        doc_dict = {
            "id": doc_id,
            "content": content,
            "path": path,  # Keep path for backward compatibility
            "type": doc_type,  # Keep type for backward compatibility
            "chunks": chunks
        }
        return doc_dict
    else:
        # Default to Python dictionaries
        doc_dict = {
            "id": doc_id,
            "content": content,
            "path": path,  # Keep path for backward compatibility
            "type": doc_type,  # Keep type for backward compatibility
            "chunks": chunks
        }
        return doc_dict


def chunk_document_to_json(
    document: Union[Dict[str, Any], BaseDocument],
    *,
    max_tokens: int = 2048,
    save_to_disk: bool = False,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Process a document and return a JSON dict.

    Args:
        document: The document to process
        max_tokens: Maximum number of tokens per chunk
        save_to_disk: Whether to save the document to disk
        output_dir: Directory to save the document to

    Returns:
        A dict with the document and chunks
    """
    # Process the document
    result = chunk_document(
        document,
        max_tokens=max_tokens,
        return_pydantic=False,
        save_to_disk=save_to_disk,
        output_dir=output_dir
    )

    # Convert to dict if needed
    if isinstance(result, dict):
        return result
    elif hasattr(result, 'dict') and callable(getattr(result, 'dict')):
        # Convert BaseDocument or DocumentSchema to dict
        result_dict: Dict[str, Any] = result.dict()
        return result_dict
    else:
        # Should never happen, but just in case
        raise TypeError(f"Unexpected result type: {type(result)}")


def chunk_document_to_schema(
    document: Union[Dict[str, Any], BaseDocument],
    *,
    max_tokens: int = 2048,
    save_to_disk: bool = False,
    output_dir: Optional[str] = None
) -> DocumentSchema:
    """Process a document and return a DocumentSchema.

    Args:
        document: The document to process
        max_tokens: Maximum number of tokens per chunk
        save_to_disk: Whether to save the document to disk
        output_dir: Directory to save the document to

    Returns:
        A DocumentSchema with the document and chunks
    """
    # Process the document
    result = chunk_document(
        document,
        max_tokens=max_tokens,
        return_pydantic=True,
        save_to_disk=save_to_disk,
        output_dir=output_dir
    )

    # Convert to DocumentSchema if needed
    if isinstance(result, dict):
        # Create a DocumentSchema with the right field names
        doc_id = str(result.get("id", str(uuid.uuid4())))
        content = str(result.get("content", ""))
        source = str(result.get("path", result.get("source", "")))
        doc_type = result.get("type", result.get("document_type", "text"))
        
        # Create the DocumentSchema with proper fields
        doc_schema = DocumentSchema(
            id=doc_id,
            content=content,
            source=source,
            document_type=DocumentType(str(doc_type)),
            schema_version=SchemaVersion.V2,
            title=None,
            author=None,
            created_at=datetime.now(),
            updated_at=None,
            metadata={},
            embedding=None,
            embedding_model=None,
            chunks=[],  # Will add chunks later
            tags=[]
        )
        
        # Convert chunks to ChunkMetadata objects
        chunk_metadata_list: List[ChunkMetadata] = []
        chunks = result.get("chunks", [])
        for i, chunk_dict in enumerate(chunks):
            if isinstance(chunk_dict, dict):
                # Extract chunk attributes with fallbacks
                chunk_content = str(chunk_dict.get("content", ""))
                chunk_type_raw = chunk_dict.get("type", "text")
                chunk_type = str(chunk_type_raw) if chunk_type_raw is not None else "text"
                
                # Calculate offsets
                start_offset = 0
                end_offset = len(chunk_content) if chunk_content else 0
                
                # Create chunk metadata
                chunk_metadata = ChunkMetadata(
                    start_offset=start_offset,
                    end_offset=end_offset,
                    chunk_type=chunk_type,
                    chunk_index=i,
                    parent_id=doc_id,
                    context_before=None,
                    context_after=None,
                    metadata={
                        "symbol_type": chunk_dict.get("symbol_type", "paragraph"),
                        "name": chunk_dict.get("name", ""),
                        "token_count": chunk_dict.get("token_count", 0),
                        "content_hash": chunk_dict.get("content_hash", ""),
                        "content": chunk_content,
                    }
                )
                chunk_metadata_list.append(chunk_metadata)
        
        # Set the chunks
        if hasattr(doc_schema, "chunks"):
            setattr(doc_schema, "chunks", chunk_metadata_list)
            
        return doc_schema
        
    elif hasattr(result, 'dict') and callable(getattr(result, 'dict')):
        # For BaseDocument, first convert to dict and then use the logic above
        doc_dict = result.dict()
        return chunk_document_to_schema(doc_dict)
        
    elif isinstance(result, DocumentSchema):
        # Already a DocumentSchema
        return result
    else:
        # Unexpected type, create a default DocumentSchema
        logger.warning(f"Unexpected result type in chunk_document_to_schema: {type(result)}")
        # Create a minimal valid DocumentSchema with required fields
        # Use a cast to help mypy understand the type
        return cast(DocumentSchema, DocumentSchemaBase(content="", path=""))


def chunk_document(
    document: Union[Dict[str, Any], BaseDocument], 
    *, 
    max_tokens: int = 2048, 
    return_pydantic: bool = True, 
    save_to_disk: bool = False, 
    output_dir: Optional[str] = None
) -> Union[Dict[str, Any], BaseDocument, DocumentSchema]:
    """
    Chunk a document and update it with the generated chunks.
    
    This function processes a document (either as a dictionary or Pydantic model),
    generates chunks using the appropriate chunker, and returns an updated document
    with the chunks attached.
    
    Parameters
    ----------
    document:
        Document to chunk (dictionary or Pydantic model)
    max_tokens:
        Maximum number of tokens per chunk
    return_pydantic:
        Whether to return a Pydantic model (True) or dictionary (False)
    save_to_disk:
        Whether to save the chunked document to disk (for debugging)
    output_dir:
        Directory to save the chunked document (if save_to_disk is True)
        
    Returns
    -------
    Union[DocumentSchema, Dict[str, Any]]
        Updated document with chunks attached
    """
    # Convert to dictionary if needed
    if not isinstance(document, dict):
        doc_dict = document.model_dump() if hasattr(document, "model_dump") else document.dict() if hasattr(document, "dict") else {
            k: getattr(document, k) for k in dir(document) 
            if not k.startswith("_") and not callable(getattr(document, k))
        }
    else:
        doc_dict = document.copy()
    
    # Get document type
    doc_type = doc_dict.get("type", "text")
    
    # Choose the appropriate chunker based on document type
    if doc_type in ["python", "js", "typescript", "java", "c", "cpp", "csharp"]:
        # For code documents, use the code chunker
        from src.chunking.code_chunkers.ast_chunker import chunk_python_code
        chunks = chunk_python_code(doc_dict, max_tokens=max_tokens)
    else:
        # For text documents, use the text chunker
        # Get chunker configuration to extract the model_id
        config = get_chunker_config('chonky')
        model_id = config.get('model_id', 'mirth/chonky_modernbert_large_1')
        
        # Log model ID for debugging
        logger.info(f"Using model_id for chunking: {model_id}")
        
        # Extract the necessary document fields
        content = doc_dict.get("content", "")
        doc_id = doc_dict.get("id", None)
        path = doc_dict.get("path", "unknown")
        doc_type = doc_dict.get("type", "text")
        
        # Log content details for debugging
        content_len = len(content) if content else 0
        logger.info(f"Processing document content of length {content_len}")
        
        # Call chunk_text with extracted parameters, explicitly requesting dict format
        chunked_result = chunk_text(
            content=content,
            doc_id=doc_id,
            path=path,
            doc_type=doc_type,
            max_tokens=max_tokens,
            output_format="dict",  # Explicitly request dictionary format
            model_id=model_id
        )
    
    # Extract the chunks from the result and update the document
    if isinstance(chunked_result, dict) and "chunks" in chunked_result:
        # Use the chunks from the dict result
        doc_dict["chunks"] = chunked_result["chunks"]
    elif hasattr(chunked_result, "chunks"):
        # Extract chunks from a Pydantic model
        doc_dict["chunks"] = getattr(chunked_result, "chunks", [])
    else:
        # If for some reason we didn't get chunks, set an empty list
        logger.warning(f"Unexpected result type from chunk_text: {type(chunked_result)}")
        doc_dict["chunks"] = []
    
    # Save to disk if requested
    if save_to_disk:
        if output_dir is None:
            output_dir = "chunks"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a filename
        doc_id = doc_dict.get("id", "unknown")
        filename = f"{doc_id}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Save to disk
        with open(filepath, "w") as f:
            json.dump(doc_dict, f, indent=2)
        
        logger.info(f"Saved chunked document to {filepath}")
    
    # Return as Pydantic model if requested
    if return_pydantic:
        try:
            # If we're using our own DocumentSchema, we can create it directly
            if hasattr(DocumentSchema, "__module__") and DocumentSchema.__module__ == __name__:
                return DocumentSchema(**doc_dict)
            else:
                # We're using the imported DocumentSchema, which might have different field requirements
                # Try to adapt our document to match the expected schema
                adapted_dict = {
                    "content": doc_dict.get("content", ""),
                    "path": doc_dict.get("path", ""),
                    "id": doc_dict.get("id", ""),
                    "chunks": doc_dict.get("chunks", []),
                    # Add any additional fields that might be required by the imported schema
                    "source": doc_dict.get("path", ""),  # Use path as source if not provided
                    "document_type": doc_dict.get("type", "text"),  # Use type as document_type
                }
                # Create the document schema instance
                return DocumentSchema(**adapted_dict)
        except Exception as e:
            logger.error(f"Error converting to Pydantic model: {e}")
            # Fall back to dictionary
            return doc_dict
    else:
        return doc_dict
