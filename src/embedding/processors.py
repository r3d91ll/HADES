"""
Embedding processors for the HADES-PathRAG system.

This module contains functions for processing chunked documents by adding
embeddings to their chunks. It provides the bridge between the chunking and
storage stages of the ingestion pipeline.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, cast, Iterable

import numpy as np
from pydantic import ValidationError

# Import from new schema structure
from src.schemas.common.types import EmbeddingVector
from src.schemas.documents.base import DocumentSchema, ChunkMetadata
from src.embedding.base import EmbeddingAdapter, get_adapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def add_embeddings_to_document(
    document: Union[Dict[str, Any], DocumentSchema],
    adapter_name: str = "cpu",
    adapter_options: Optional[Dict[str, Any]] = None,
    chunk_content_key: str = "content",
    embedding_key: str = "embedding",
    validate_schema: bool = True
) -> Union[Dict[str, Any], DocumentSchema]:
    """Add embeddings to a document's chunks.
    
    Args:
        document: Document dictionary or DocumentSchema instance with chunks
        adapter_name: Name of the embedding adapter to use
        adapter_options: Options for the embedding adapter
        chunk_content_key: Key in each chunk containing the content to embed
        embedding_key: Key to store the embedding under in each chunk
        validate_schema: Whether to validate the document schema (True by default)
        
    Returns:
        Dictionary or DocumentSchema with document and chunk embeddings added
        
    Raises:
        ValidationError: If document validation fails
        RuntimeError: If embedding fails
    """
    # Validate and convert to DocumentSchema if needed
    is_pydantic_model = isinstance(document, DocumentSchema)
    if validate_schema and not is_pydantic_model:
        try:
            # Use model_validate for Pydantic v2
            document = DocumentSchema.model_validate(document)
            is_pydantic_model = True
        except ValidationError as e:
            logger.error(f"Document validation failed: {e}")
            raise
    
    # Create a copy to avoid modifying the original
    if is_pydantic_model:
        document_with_embeddings = document.model_copy(deep=True)
    else:
        document_with_embeddings = document.copy()
    
    # Get chunks from the document (handle both dict and Pydantic models)
    if is_pydantic_model:
        chunks = getattr(document_with_embeddings, "chunks", [])
        doc_id = getattr(document_with_embeddings, "id", "unknown")
    else:
        chunks = document_with_embeddings.get("chunks", [])
        doc_id = document_with_embeddings.get("id", "unknown")
        
    if not chunks:
        logger.warning(f"Document {doc_id} has no chunks to embed")
        return document_with_embeddings
    
    # Get adapter with options
    adapter_options = adapter_options or {}
    adapter = get_adapter(adapter_name, **adapter_options)
    
    logger.info(f"Embedding {len(chunks)} chunks from document {doc_id}")
    
    # Extract content from chunks (handle both dict and ChunkMetadata objects)
    chunk_contents = []
    for chunk in chunks:
        if hasattr(chunk, "__getitem__"):
            # Dictionary-like object
            chunk_contents.append(chunk.get(chunk_content_key, ""))
        else:
            # Pydantic model
            chunk_contents.append(getattr(chunk, chunk_content_key, ""))
    
    # Skip empty contents
    valid_indices = [i for i, content in enumerate(chunk_contents) if content.strip()]
    valid_contents = [chunk_contents[i] for i in valid_indices]
    
    if not valid_contents:
        logger.warning(f"No valid chunk content found in document {doc_id}")
        return document_with_embeddings
    
    # Generate embeddings for all chunks
    try:
        embeddings = await adapter.embed(valid_contents)
        
        # Add embeddings to chunks (handle both dict and Pydantic models)
        for idx, embedding in zip(valid_indices, embeddings):
            chunk = chunks[idx]
            if hasattr(chunk, "__setitem__"):
                # Dictionary-like object
                chunk[embedding_key] = embedding
            else:
                # Pydantic model
                setattr(chunk, embedding_key, embedding)
        
        # Add a summary of embedding information to the document
        embedding_info = {
            "adapter": adapter_name,
            "chunk_count": len(chunks),
            "embedded_chunk_count": len(valid_indices),
            "embedding_dimensions": len(embeddings[0]) if embeddings else 0,
            "timestamp": None
        }
        
        # Get path for timestamp if available
        if is_pydantic_model:
            path = getattr(document, "source", None)
            if path and os.path.exists(path):
                embedding_info["timestamp"] = str(os.path.getmtime(path))
        else:
            path = document.get("path", None) or document.get("source", None)
            if path and os.path.exists(path):
                embedding_info["timestamp"] = str(os.path.getmtime(path))
                
        # Store embedding info in the appropriate way based on document type
        if is_pydantic_model:
            setattr(document_with_embeddings, "_embedding_info", embedding_info)
        else:
            document_with_embeddings["_embedding_info"] = embedding_info
        
        logger.info(f"Added embeddings to {len(valid_indices)} chunks in document {doc_id}")
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        # Add error information to document
        if is_pydantic_model:
            setattr(document_with_embeddings, "_embedding_error", str(e))
        else:
            document_with_embeddings["_embedding_error"] = str(e)
    
    return document_with_embeddings


async def process_chunked_document_file(
    file_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    adapter_name: str = "cpu",
    adapter_options: Optional[Dict[str, Any]] = None,
    validate_schema: bool = True
) -> Tuple[Union[Dict[str, Any], DocumentSchema], Optional[Path]]:
    """Process a chunked document file by adding embeddings.
    
    Args:
        file_path: Path to chunked document JSON file
        output_dir: Directory to save the output (None = don't save)
        adapter_name: Name of the embedding adapter to use
        adapter_options: Options for the embedding adapter
        
    Returns:
        Tuple of (document with embeddings, output file path if saved)
    """
    path_obj = Path(file_path) if isinstance(file_path, str) else file_path
    
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path_obj}")
    
    # Load the chunked document
    with open(path_obj, 'r', encoding='utf-8') as f:
        document = json.load(f)
    
    # Add embeddings to the document
    document_with_embeddings = await add_embeddings_to_document(
        document=document,
        adapter_name=adapter_name,
        adapter_options=adapter_options,
        validate_schema=validate_schema
    )
    
    # Save the document with embeddings if output_dir is provided
    output_path = None
    if output_dir:
        output_dir_obj = Path(output_dir) if isinstance(output_dir, str) else output_dir
        output_dir_obj.mkdir(parents=True, exist_ok=True)
        
        # Generate output file name
        doc_id = document.get("id", path_obj.stem)
        output_path = output_dir_obj / f"{doc_id}_embedded.json"
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(document_with_embeddings, f, indent=2)
        
        logger.info(f"Saved document with embeddings to {output_path}")
    
    return document_with_embeddings, output_path


async def process_chunked_documents_batch(
    file_paths: List[Union[str, Path]],
    output_dir: Optional[Union[str, Path]] = None,
    adapter_name: str = "cpu",
    adapter_options: Optional[Dict[str, Any]] = None,
    batch_size: int = 10,
    max_concurrent: int = 5,
    validate_schema: bool = True
) -> Dict[str, Any]:
    """Process multiple chunked document files in batches.
    
    Args:
        file_paths: List of paths to chunked document JSON files
        output_dir: Directory to save the outputs (None = don't save)
        adapter_name: Name of the embedding adapter to use
        adapter_options: Options for the embedding adapter
        batch_size: Number of chunks to process in a batch
        max_concurrent: Maximum number of documents to process concurrently
        
    Returns:
        Dictionary with processing statistics
    """
    logger.info(f"Processing {len(file_paths)} chunked documents with {adapter_name} adapter")
    
    # Prepare output directory if provided
    output_path = None
    if output_dir:
        output_path = Path(output_dir) if isinstance(output_dir, str) else output_dir
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Process documents in batches
    results = []
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_batch(batch):
        batch_results = []
        for file_path in batch:
            try:
                async with semaphore:
                    document, output_file = await process_chunked_document_file(
                        file_path=file_path,
                        output_dir=output_path,
                        adapter_name=adapter_name,
                        adapter_options=adapter_options,
                        validate_schema=validate_schema
                    )
                
                # Extract relevant document info (works with both dict and Pydantic model)
                if isinstance(document, DocumentSchema):
                    doc_id = document.id
                    chunks = document.chunks
                    embedded_chunks = [c for c in chunks if hasattr(c, "embedding") and c.embedding is not None]
                else:    
                    doc_id = document.get("id", "unknown")
                    chunks = document.get("chunks", [])
                    embedded_chunks = [c for c in chunks if c.get("embedding") is not None]
                
                batch_results.append({
                    "file_path": str(file_path),
                    "document_id": doc_id,
                    "chunks": len(chunks),
                    "embedded_chunks": len(embedded_chunks),
                    "success": True,
                    "output_file": str(output_file) if output_file else None
                })
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                batch_results.append({
                    "file_path": str(file_path),
                    "error": str(e),
                    "success": False,
                    "output_file": None
                })
        return batch_results
    
    # Process files in batches
    batches = [file_paths[i:i + batch_size] for i in range(0, len(file_paths), batch_size)]
    tasks = [process_batch(batch) for batch in batches]
    batch_results = await asyncio.gather(*tasks)
    
    # Flatten results from batches
    results = [item for batch in batch_results for item in batch]
    
    # Compile statistics
    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]
    
    stats = {
        "total": len(file_paths),
        "successful": len(successful),
        "failed": len(failed),
        "details": results
    }
    
    # Save statistics if output_dir is provided
    if output_path:
        stats_file = output_path / "embedding_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved statistics to {stats_file}")
    
    return stats
