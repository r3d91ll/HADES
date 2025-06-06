"""CPU-optimized text chunking implementation.

This module implements parallel CPU-based chunking for efficient text processing
without GPU acceleration requirements. It's designed to work with the standard
Chonky chunker API while providing better performance on CPU-only environments.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence, cast

from src.chunking.text_chunkers.chonky_chunker import (
    ParagraphSplitter,
    _hash_path
)

# Import consolidated types
from src.types.chunking import (
    DocumentSchemaType,
    BaseDocument,
    DocumentSchema,
    ChunkMetadata
)
from src.types.common import DocumentType, SchemaVersion

logger = logging.getLogger(__name__)


def chunk_text_cpu(
    content: str,
    doc_id: Optional[str] = None,
    path: str = "unknown",
    doc_type: str = "text",
    max_tokens: int = 2048,
    output_format: str = "dict",
    model_id: str = "mirth/chonky_modernbert_large_1",
    num_workers: int = 4
) -> Union[Dict[str, Any], DocumentSchemaType]:
    """Chunk a text document into semantically coherent paragraphs using CPU.
    
    This function provides a CPU-optimized implementation of the text chunking,
    using parallel processing for larger documents to improve performance.
    
    Args:
        content: Text content to chunk
        doc_id: Document ID (will be auto-generated if None)
        path: Path to the document
        doc_type: Type of document
        max_tokens: Maximum tokens per chunk
        output_format: Output format, one of "document", "json", "dict"
        model_id: ID of the model to use for chunking
        num_workers: Number of CPU workers for parallel processing
        
    Returns:
        Chunked document in the specified format
    """
    # Start detailed logging
    doc_id_log = doc_id if doc_id else 'new document'
    logger.info(f"CPU chunking document: {doc_id_log}")
    
    # Handle empty content
    if not content or not content.strip():
        logger.warning("Empty content provided to chunk_text_cpu")
        if output_format == "document":
            return DocumentSchema(
                id=doc_id or str(uuid.uuid4()),
                content="",
                source=path,
                document_type=DocumentType(doc_type),
                chunks=[]
            )
        return {
            "id": doc_id or str(uuid.uuid4()),
            "content": "",
            "source": path,
            "document_type": doc_type,
            "chunks": []
        }
    
    # Generate document ID if not provided
    if not doc_id:
        doc_id = f"doc:{uuid.uuid4().hex}"
    
    # Create paragraph splitter for CPU
    splitter = ParagraphSplitter(
        model_id=model_id,
        device="cpu",
        use_model_engine=False
    )
    
    # Process document content
    chunks = process_content_with_cpu(
        content=content,
        doc_id=doc_id,
        path=path,
        doc_type=doc_type,
        splitter=splitter,
        num_workers=num_workers
    )
    
    # Format output
    result = {
        "id": doc_id,
        "content": content,
        "source": path,
        "document_type": doc_type,
        "chunks": chunks
    }
    
    # Handle different output formats
    if output_format == "document":
        # Create a proper DocumentSchema with correct types
        doc_id = str(result["id"]) if "id" in result else str(uuid.uuid4())
        doc_content = str(result["content"]) if "content" in result else ""
        doc_source = str(result["source"]) if "source" in result else ""
        doc_type_str = str(result["document_type"]) if "document_type" in result else "text"
        
        doc_schema = DocumentSchema(
            id=doc_id,
            content=doc_content,
            source=doc_source,
            document_type=DocumentType(doc_type_str),
            schema_version=SchemaVersion.V2,
            title=None,
            author=None,
            created_at=datetime.now(),
            updated_at=None,
            metadata={},
            embedding=None,
            embedding_model=None,
            chunks=[],  # We'll convert the chunks separately and add them
            tags=[]
        )
        
        # Convert chunks to ChunkMetadata objects
        chunk_metadata_list: List[ChunkMetadata] = []
        for chunk in result.get("chunks", []):
            if not isinstance(chunk, dict):
                continue
                
            # Extract and convert chunk fields with proper type checking
            start_offset = int(chunk.get("start_offset", 0))
            end_offset = int(chunk.get("end_offset", 0))
            chunk_type = str(chunk.get("chunk_type", "text"))
            chunk_index = int(chunk.get("chunk_index", 0))
            parent_id = str(chunk.get("parent_id", doc_id))
            context_before = None
            context_after = None
            metadata = chunk.get("metadata", {})
            
            # Create chunk metadata without 'content' which isn't a field in the schema
            chunk_metadata = ChunkMetadata(
                start_offset=start_offset,
                end_offset=end_offset,
                chunk_type=chunk_type,
                chunk_index=chunk_index,
                parent_id=parent_id,
                context_before=context_before,
                context_after=context_after,
                metadata=metadata
            )
            chunk_metadata_list.append(chunk_metadata)
        
        # Add chunks to the document
        doc_schema.chunks = chunk_metadata_list
        return doc_schema
        
    elif output_format == "json":
        # Return a dict with the old field names for backward compatibility
        # This is for compatibility with existing tests
        return {
            "id": result["id"],
            "content": result["content"],
            "path": result["source"],  # Map source back to path for backwards compatibility
            "type": result["document_type"],  # Map document_type back to type
            "chunks": result["chunks"]
        }
    else:  # dict format
        # Same as json format for backward compatibility
        return {
            "id": result["id"],
            "content": result["content"], 
            "path": result["source"],  # Map source back to path for backwards compatibility
            "type": result["document_type"],  # Map document_type back to type
            "chunks": result["chunks"]
        }


def process_content_with_cpu(
    content: str,
    doc_id: str,
    path: str,
    doc_type: str,
    splitter: ParagraphSplitter,
    num_workers: int = 4
) -> List[Dict[str, Any]]:
    """Process document content with CPU-based multi-threading.
    
    Args:
        content: Document content
        doc_id: Document ID
        path: Document path
        doc_type: Document type
        splitter: ParagraphSplitter instance
        num_workers: Number of CPU workers for parallel processing
        
    Returns:
        List of chunk dictionaries
    """
    char_length = len(content)
    
    # For large documents, process in segments with parallel workers
    if char_length > 10000:
        logger.info(f"Processing large document ({char_length} chars) in segments")
        segment_size = 10000  # ~10k characters per segment
        segment_count = (char_length // segment_size) + (1 if char_length % segment_size > 0 else 0)
        
        # Create segments with slight overlap to avoid cutting in the middle of paragraphs
        segments = []
        for i in range(segment_count):
            start = max(0, i * segment_size - 200 if i > 0 else 0) 
            end = min(char_length, (i + 1) * segment_size + 200 if i < segment_count - 1 else char_length)
            segment_text = content[start:end]
            segments.append((i+1, segment_count, segment_text))
            
        # Process segments in parallel using ThreadPool
        if num_workers > 1 and len(segments) > 1:
            with ThreadPool(min(num_workers, len(segments))) as pool:
                segment_results = pool.map(
                    lambda x: _process_segment(x, splitter), 
                    segments
                )
        else:
            segment_results = [_process_segment(segment, splitter) for segment in segments]
            
        # Combine paragraphs from all segments
        all_paragraphs = []
        for segment_idx, paragraphs in enumerate(segment_results):
            logger.info(f"Segment {segment_idx+1} produced {len(paragraphs)} paragraphs")
            all_paragraphs.extend(paragraphs)
            
        # Create chunks from paragraphs
        chunks = []
        path_hash = _hash_path(path)
        
        for i, para_text in enumerate(all_paragraphs):
            chunk_id = f"{doc_id}:chunk:{i}"
            # Calculate approximate offsets
            start_offset = content.find(para_text) if para_text in content else 0
            end_offset = start_offset + len(para_text) if start_offset > 0 else len(para_text)
            
            chunks.append({
                "id": chunk_id,
                "parent_id": doc_id,  # Required field for ChunkMetadata
                "start_offset": start_offset,  # Required field for ChunkMetadata
                "end_offset": end_offset,  # Required field for ChunkMetadata
                "chunk_index": i,  # Required field for ChunkMetadata
                "chunk_type": doc_type,
                "content": para_text,
                "metadata": {
                    "symbol_type": "paragraph",
                    "name": f"paragraph_{i}",
                    "path": path,
                    "token_count": len(para_text.split()),  # Approximate token count
                    "content_hash": hashlib.md5(para_text.encode()).hexdigest(),
                }
            })
        
        logger.info(f"Split document into {len(chunks)} semantic paragraphs across {len(segments)} segments")
        return chunks
        
    else:
        # For smaller documents, process directly
        paragraphs = splitter.split_text(content)
        
        # Create chunks from paragraphs
        chunks = []
        path_hash = _hash_path(path)
        
        for i, para_text in enumerate(paragraphs):
            chunk_id = f"{doc_id}:chunk:{i}"
            # Calculate approximate offsets
            start_offset = content.find(para_text) if para_text in content else 0
            end_offset = start_offset + len(para_text) if start_offset > 0 else len(para_text)
            
            chunks.append({
                "id": chunk_id,
                "parent_id": doc_id,  # Required field for ChunkMetadata
                "start_offset": start_offset,  # Required field for ChunkMetadata
                "end_offset": end_offset,  # Required field for ChunkMetadata
                "chunk_index": i,  # Required field for ChunkMetadata
                "chunk_type": doc_type,
                "content": para_text,
                "metadata": {
                    "symbol_type": "paragraph",
                    "name": f"paragraph_{i}",
                    "path": path,
                    "token_count": len(para_text.split()),  # Approximate token count
                    "content_hash": hashlib.md5(para_text.encode()).hexdigest(),
                }
            })
            
        logger.info(f"Split document into {len(chunks)} semantic paragraphs")
        return chunks


def _process_segment(
    segment_data: Tuple[int, int, str],
    splitter: ParagraphSplitter
) -> List[str]:
    """Process a single text segment with the paragraph splitter.
    
    Args:
        segment_data: Tuple of (segment_index, total_segments, segment_text)
        splitter: ParagraphSplitter instance
        
    Returns:
        List of paragraph texts from this segment
    """
    segment_idx, total_segments, segment_text = segment_data
    logger.info(f"Processing segment {segment_idx}/{total_segments} ({len(segment_text)} chars)")
    
    # Split the segment into paragraphs
    paragraphs = splitter.split_text(segment_text)
    # Ensure we always return a list of strings
    return [str(p) for p in paragraphs]


def chunk_document_cpu(
    document: Union[Dict[str, Any], BaseDocument, DocumentSchema], 
    *, 
    max_tokens: int = 2048, 
    return_pydantic: bool = False, 
    num_workers: int = 4,
    model_id: str = "mirth/chonky_modernbert_large_1"
) -> Union[DocumentSchema, Dict[str, Any]]:
    """Chunk a document using CPU-optimized processing.
    
    This function is a CPU-optimized version of the document chunking process
    that uses parallel processing for improved performance.
    
    Args:
        document: Document to chunk (dictionary or Pydantic model)
        max_tokens: Maximum tokens per chunk
        return_pydantic: Whether to return a Pydantic model or dict
        num_workers: Number of CPU workers for parallel processing
        model_id: Model ID to use for chunking
        
    Returns:
        Updated document with chunks
    """
    # Handle different input types
    doc_dict: Dict[str, Any] = {}
    
    if isinstance(document, DocumentSchema):
        doc_dict = document.dict()
        doc_id = str(getattr(document, 'id', f"doc:{uuid.uuid4().hex}"))
        doc_content = str(getattr(document, 'content', ''))
        doc_path = str(getattr(document, 'path', getattr(document, 'source', 'unknown')))
        doc_type_str = str(getattr(document, 'document_type', 'text'))
    elif hasattr(document, 'dict') and callable(getattr(document, 'dict')):
        # For BaseDocument or other Pydantic models
        doc_dict = document.dict()
        doc_id = str(doc_dict.get("id") or f"doc:{uuid.uuid4().hex}")
        doc_content = str(doc_dict.get("content", ""))
        doc_path = str(doc_dict.get("source", doc_dict.get("path", "unknown")))
        doc_type_str = str(doc_dict.get("type", "text"))
    elif isinstance(document, dict):
        # For plain dictionaries
        doc_dict = document
        doc_id = str(doc_dict.get("id", f"doc:{uuid.uuid4().hex}"))
        doc_content = str(doc_dict.get("content", ""))
        doc_path = str(doc_dict.get("source", doc_dict.get("path", "unknown")))
        doc_type_str = str(doc_dict.get("document_type", doc_dict.get("type", "text")))
    else:
        raise ValueError(f"Unsupported document type: {type(document)}")
    
    # Document properties are now extracted above
    
    # Process content
    logger.info(f"CPU Chunking document {doc_id} with model {model_id}")
    
    # Generate chunks - using the doc_type_str for the chunking process
    chunks_result = chunk_text_cpu(
        content=doc_content,
        doc_id=doc_id,
        path=doc_path,
        doc_type=doc_type_str,  # Pass the string version for chunking
        max_tokens=max_tokens,
        output_format="dict",  # Always get dict format
        model_id=model_id,
        num_workers=num_workers
    )
    
    # Update document with chunks - ensuring correct type handling
    # Create a safely typed chunks list to avoid type errors
    chunks_list: List[Any] = []
    
    try:
        # Extract chunks based on the result type
        if isinstance(chunks_result, dict):
            # For dictionary results, extract the chunks list
            dict_chunks = chunks_result.get("chunks", [])
            if isinstance(dict_chunks, list):
                chunks_list = dict_chunks
        elif hasattr(chunks_result, "chunks"):
            # For object results with a chunks attribute
            obj_chunks = getattr(chunks_result, "chunks", [])
            if isinstance(obj_chunks, list):
                chunks_list = obj_chunks
            elif obj_chunks is not None:
                # Convert to a list with one item if not a list
                chunks_list = [obj_chunks]
    except Exception as e:
        logger.error(f"Error extracting chunks from result: {e}")
        # Keep chunks_list as empty list
    
    # Assign the chunks list to the document dictionary
    # Using explicit assignment with type safety for mypy
    # Ensure we have a list even if chunks_list is None
    safe_chunks = chunks_list if isinstance(chunks_list, list) else []
    
    # Create a completely new document dictionary with proper type annotations
    # This is safer than modifying the existing one that has type issues
    
    from src.schemas.documents.base import ChunkMetadata
    
    # Create a new typed dictionary with all fields except chunks
    new_doc: Dict[str, Any] = {}
    for key, value in doc_dict.items():
        if key != "chunks":
            new_doc[key] = value
    
    # Process chunks to ensure they're all proper ChunkMetadata objects
    chunk_list: List[ChunkMetadata] = []
    for chunk in safe_chunks:
        if isinstance(chunk, ChunkMetadata):
            chunk_list.append(chunk)
        elif isinstance(chunk, dict):
            try:
                chunk_list.append(ChunkMetadata(**chunk))
            except Exception as e:
                logger.warning(f"Error converting chunk to ChunkMetadata: {e}")
    
    # Add chunks with proper typing - this should now satisfy mypy
    new_doc["chunks"] = chunk_list
    
    # Use the new dictionary instead of the original one with type issues
    doc_dict = new_doc
    
    # Return result in requested format
    if return_pydantic:
        # If it should be a Pydantic model, create a DocumentSchema with the right fields
        try:
            # Need to map fields correctly for DocumentSchema
            # Convert doc_type_str to a valid DocumentType enum value
            # First ensure it's a lowercase string to match the enum values
            doc_type_lower = doc_type_str.lower()
            
            # Use a valid DocumentType or default to TEXT if not recognized
            try:
                document_type_enum = DocumentType(doc_type_lower)
            except ValueError:
                # If the string doesn't match any enum value, default to TEXT
                document_type_enum = DocumentType.TEXT
                logger.warning(f"Unknown document type '{doc_type_str}', defaulting to {DocumentType.TEXT}")
            
            doc_schema = DocumentSchema(
                id=doc_dict["id"],
                content=doc_dict["content"],
                source=doc_dict.get("path", ""),  # Ensure it's a string
                document_type=document_type_enum,  # Use the properly converted enum
                schema_version=SchemaVersion.V2,
                title=None,
                author=None,
                created_at=datetime.now(),
                updated_at=None,
                metadata={},
                embedding=None,
                embedding_model=None,
                chunks=[],  # Will convert chunks separately
                tags=[]
            )
            
            # Create a properly typed list to hold chunk metadata
            chunk_metadata_list: List[ChunkMetadata] = []
            
            # Get the chunks data from the document with proper typing
            # First, get a safely typed variable that will hold our chunks
            chunks_data: List[Any] = []
            
            # Extract the raw chunks data
            raw_chunks: Any = doc_dict.get("chunks", [])
            
            # Ensure we have a properly typed list
            if isinstance(raw_chunks, list):
                chunks_data = raw_chunks
            elif raw_chunks is not None:
                # If not a list but has a value, convert to single-item list
                chunks_data = [raw_chunks]
            
            # Process each chunk if there are any
            if chunks_data and isinstance(chunks_data, list):
                for chunk in chunks_data:
                    if isinstance(chunk, dict):
                        # Extract chunk data with proper type conversion
                        chunk_start = int(chunk.get("start_offset", 0))
                        chunk_end = int(chunk.get("end_offset", 0)) 
                        chunk_type = str(chunk.get("chunk_type", "text"))
                        chunk_index = int(chunk.get("chunk_index", 0))
                        parent_id = str(chunk.get("parent_id", doc_dict.get("id", "")))
                        
                        # Create ChunkMetadata with safely typed values
                        try:
                            chunk_meta = ChunkMetadata(
                                start_offset=chunk_start,
                                end_offset=chunk_end,
                                chunk_type=chunk_type,
                                chunk_index=chunk_index,
                                parent_id=parent_id,
                                context_before=None, 
                                context_after=None,
                                metadata=chunk.get("metadata", {})
                            )
                            chunk_metadata_list.append(chunk_meta)
                        except Exception as chunk_e:
                            logger.error(f"Error creating ChunkMetadata: {chunk_e}")
            
            # Set chunks list on doc_schema
            doc_schema.chunks = chunk_metadata_list
            return doc_schema
        except Exception as e:
            logger.error(f"Failed to create DocumentSchema: {e}")
            # Fall back to dict if creating DocumentSchema fails
            return doc_dict
    
    # Otherwise return as dict
    return doc_dict
