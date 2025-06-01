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
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence, cast
from multiprocessing.pool import ThreadPool

def _safe_int_conversion(value: Any, default: int = 0) -> int:
    """Safely convert value to int, with fallback to default."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

from src.chunking.text_chunkers.chonky_chunker import (
    ParagraphSplitter,
    DocumentSchemaType,
    BaseDocument,
    _hash_path
)

from src.schemas.documents.base import DocumentSchema, ChunkMetadata as SchemaChunkMetadata
from src.schemas.common.enums import DocumentType, SchemaVersion

# Import centralized type definitions
from src.types.chunking.text import (
    TextChunk, TextChunkMetadata, TextChunkList, TextChunkingOptions,
    create_text_chunk_metadata
)
from src.types.chunking import (
    ChunkableDocument, ChunkerConfig
)
# Import the TypedDict ChunkMetadata for type checking
from src.types.chunking.chunk import ChunkMetadata

logger = logging.getLogger(__name__)


def chunk_text_cpu(
    content: ChunkableDocument,
    doc_id: Optional[str] = None,
    path: str = "unknown",
    doc_type: str = "text",
    max_tokens: int = 2048,
    output_format: str = "dict",
    model_id: str = "mirth/chonky_modernbert_large_1",
    num_workers: int = 4
) -> Union[Dict[str, Any], DocumentSchemaType, TextChunkList]:
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
    
    # Extract content string from the ChunkableDocument
    text_content = ""
    if isinstance(content, str):
        text_content = content
    elif isinstance(content, dict):
        text_content = content.get('text', '') or content.get('content', '')
    else:
        # Try to extract from object attributes
        try:
            text_content = getattr(content, 'content', '') or getattr(content, 'text', '')
        except AttributeError:
            logger.warning("Could not extract text from content object")
    
    # Handle empty content
    if not text_content or not text_content.strip():
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
    text_chunks: TextChunkList = process_content_with_cpu(
        content=text_content,
        doc_id=doc_id,
        path=path,
        doc_type=doc_type,
        splitter=splitter,
        num_workers=num_workers
    )
    
    # Format output
    result = {
        "id": doc_id,
        "content": text_content,
        "source": path,
        "document_type": doc_type,
        "chunks": text_chunks
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
        chunk_metadata_list: List[SchemaChunkMetadata] = []
        for chunk in result.get("chunks", []):
            if not isinstance(chunk, dict):
                continue
                
            # Extract and convert chunk fields with proper type checking
            try:
                start_offset = int(chunk.get("start_offset", 0))
            except (TypeError, ValueError):
                start_offset = 0
                
            try:
                end_offset = int(chunk.get("end_offset", 0))
            except (TypeError, ValueError):
                end_offset = 0
                
            chunk_type = str(chunk.get("chunk_type", "text"))
            
            try:
                chunk_index = int(chunk.get("chunk_index", 0))
            except (TypeError, ValueError):
                chunk_index = 0
            parent_id = str(chunk.get("parent_id", doc_id))
            context_before = None
            # Extract metadata from chunk and create a proper ChunkMetadata TypedDict
            raw_metadata = chunk.get("metadata", {})
            # Only include fields that are valid for ChunkMetadata TypedDict
            chunk_metadata: ChunkMetadata = {
                "id": raw_metadata.get("id", ""),
                "parent_id": raw_metadata.get("parent_id"),
                "path": raw_metadata.get("path", ""),
                "document_type": raw_metadata.get("document_type", ""),
                "source_file": raw_metadata.get("source_file", ""),
                "line_start": _safe_int_conversion(raw_metadata.get("line_start"), 0),
                "line_end": _safe_int_conversion(raw_metadata.get("line_end"), 0),
                "char_start": raw_metadata.get("char_start"),
                "char_end": raw_metadata.get("char_end"),
                "token_count": _safe_int_conversion(raw_metadata.get("token_count"), 0),
                "chunk_type": raw_metadata.get("chunk_type", "text"),
                "language": raw_metadata.get("language"),
                "created_at": raw_metadata.get("created_at", datetime.now().isoformat())
            }
            
            # Create a schema chunk metadata with available fields from the schema
            # We need to convert our ChunkMetadata TypedDict to a regular dict for use with SchemaChunkMetadata
            metadata_dict = dict(chunk_metadata)
            
            # Create the SchemaChunkMetadata (Pydantic model) with proper fields
            schema_chunk_metadata = SchemaChunkMetadata(
                # SchemaChunkMetadata is a different class than ChunkMetadata TypedDict
                # Make sure to use proper field names for the Pydantic model
                start_offset=_safe_int_conversion(metadata_dict.get("char_start"), 0),
                end_offset=_safe_int_conversion(metadata_dict.get("char_end"), 0),
                chunk_type=str(metadata_dict.get("chunk_type", "text")),  # Ensure chunk_type is str
                chunk_index=chunk_index,
                parent_id=parent_id,
                context_before=context_before,
                context_after=None,
                metadata=metadata_dict
            )
            chunk_metadata_list.append(schema_chunk_metadata)
        
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
) -> TextChunkList:
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
        result_chunks: TextChunkList = []
        path_hash = _hash_path(path)
        
        for i, para_text in enumerate(all_paragraphs):
            chunk_id = f"{doc_id}:chunk:{i}"
            # Calculate approximate offsets
            start_offset = content.find(para_text) if para_text in content else 0
            end_offset = start_offset + len(para_text) if start_offset > 0 else len(para_text)
            
            # Create metadata using centralized helper
            chunk_metadata = create_text_chunk_metadata(
                chunk_id=chunk_id,
                document_id=doc_id,
                path=path,
                document_type=doc_type,
                token_count=len(para_text.split()),  # Approximate token count
                char_count=len(para_text),
                line_count=para_text.count('\n') + 1,
                index=i,
                start_offset=start_offset,
                end_offset=end_offset,
                paragraph_type="semantic",
                chunk_method="cpu_parallel"
            )
            
            # Create TextChunk
            text_chunk: TextChunk = {
                "id": chunk_id,
                "content": para_text,
                "metadata": chunk_metadata
            }
            
            result_chunks.append(text_chunk)
        
        logger.info(f"Split document into {len(result_chunks)} semantic paragraphs across {len(segments)} segments")
        return result_chunks
        
    else:
        # For smaller documents, process directly
        paragraphs = splitter.split_text(content)
        
        # Create chunks from paragraphs
        direct_chunks: TextChunkList = []
        path_hash = _hash_path(path)
        
        for i, para_text in enumerate(paragraphs):
            chunk_id = f"{doc_id}:chunk:{i}"
            # Calculate approximate offsets
            start_offset = content.find(para_text) if para_text in content else 0
            end_offset = start_offset + len(para_text) if start_offset > 0 else len(para_text)
            
            # Create metadata using centralized helper
            chunk_metadata = create_text_chunk_metadata(
                chunk_id=chunk_id,
                document_id=doc_id,
                path=path,
                document_type=doc_type,
                token_count=len(para_text.split()),  # Approximate token count
                char_count=len(para_text),
                line_count=para_text.count('\n') + 1,
                index=i,
                start_offset=start_offset,
                end_offset=end_offset,
                paragraph_type="semantic",
                chunk_method="cpu_direct"
            )
            
            # Create TextChunk
            direct_chunk: TextChunk = {
                "id": chunk_id,
                "content": para_text,
                "metadata": chunk_metadata
            }
            
            direct_chunks.append(direct_chunk)
            
        logger.info(f"Split document into {len(direct_chunks)} semantic paragraphs")
        return direct_chunks


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
    document: ChunkableDocument, 
    *, 
    max_tokens: int = 2048, 
    return_pydantic: bool = False, 
    num_workers: int = 4,
    model_id: str = "mirth/chonky_modernbert_large_1",
    output_format: str = "json"
) -> Union[Dict[str, Any], DocumentSchema, TextChunkList]:
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
                # Create a dict with only the fields defined in ChunkMetadata TypedDict
                # Then cast it to ChunkMetadata
                metadata_dict = {
                    "id": chunk.get("id", ""),
                    "parent_id": chunk.get("parent_id"),
                    "path": chunk.get("path", ""),
                    "document_type": chunk.get("document_type", ""),
                    "source_file": chunk.get("source_file", ""),
                    "line_start": _safe_int_conversion(chunk.get("line_start"), 0),
                    "line_end": _safe_int_conversion(chunk.get("line_end"), 0),
                    "char_start": chunk.get("char_start"),
                    "char_end": chunk.get("char_end"),
                    "token_count": _safe_int_conversion(chunk.get("token_count"), 0),
                    "chunk_type": chunk.get("chunk_type", "text"),
                    "created_at": chunk.get("created_at", datetime.now().isoformat())
                }
                # Cast to ChunkMetadata TypedDict
                chunk_list.append(cast(ChunkMetadata, metadata_dict))
                # Already appended in the code above
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
            # Process chunks to create schema chunk metadata
            # Get the chunks data and process with safe type casting
            chunks_list = []
            document_type_enum = DocumentType.TEXT
            if doc_dict.get("chunks") and isinstance(doc_dict["chunks"], list):
                chunks_list = doc_dict["chunks"]
                path = doc_dict.get("source", "")
                doc_id = doc_dict["id"]
                
                # Extract document type
                doc_type_str = str(doc_dict.get("document_type", "text"))
                try:
                    document_type_enum = DocumentType(doc_type_str)
                except ValueError:
                    document_type_enum = DocumentType.TEXT
                    logger.warning(f"Unknown document type '{doc_type_str}', defaulting to {DocumentType.TEXT}")
            
            # Initialize storage for retrieved chunks
            chunk_metadata_list: List[SchemaChunkMetadata] = []
            
            # Check if doc already has chunks
            if doc_dict.get("chunks") and isinstance(doc_dict["chunks"], list) and doc_dict["chunks"]:
                logger.info("Document already has chunks - returning as is")
                # Apply proper type casting for return type safety
                if return_pydantic and isinstance(document, DocumentSchema):
                    return document
                elif output_format == "chunks":
                    # If chunks were requested but doc has chunks already, extract them
                    chunks_list = cast(TextChunkList, doc_dict.get("chunks", []))
                    return chunks_list
                else:
                    # Return as JSON dict
                    return doc_dict
            
            # Extract content to chunk
            content: Optional[str] = None
            # Check for content in either "content" or "text" field
            if isinstance(doc_dict.get("content"), str) and doc_dict["content"].strip():
                content = doc_dict["content"]
            # Only check for text if content is not already set
            if not content and isinstance(doc_dict.get("text"), str) and doc_dict["text"].strip():
                content = doc_dict["text"]
                
            if not content:
                logger.warning("No content found to chunk in document")
                if return_pydantic and isinstance(document, DocumentSchema):
                    return document
                elif output_format == "chunks":
                    # Return empty chunk list if chunks were requested
                    return cast(TextChunkList, [])
                else:
                    # Return as dict
                    return doc_dict
            
            # Create a proper DocumentSchema object
            doc_schema = None
            try:
                doc_schema = DocumentSchema(
                    id=doc_dict["id"],
                    content=doc_dict.get("content", ""),
                    source=doc_dict.get("path", ""),  # Ensure it's a string
                    document_type=document_type_enum,  # Use the properly converted enum
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    schema_version=SchemaVersion.V2,  # Use schema_version, not version
                    title=None,
                    author=None,
                    metadata=doc_dict.get("metadata", {}),
                    embedding=None,
                    embedding_model=None,
                    chunks=[
                        # Create proper SchemaChunkMetadata (Pydantic model) objects for each chunk
                        # Using the correct field names for SchemaChunkMetadata
                        SchemaChunkMetadata(
                            start_offset=_safe_int_conversion(chunk_item["metadata"].get("char_start"), 0),
                            end_offset=_safe_int_conversion(chunk_item["metadata"].get("char_end"), 0),
                            chunk_type=chunk_item["metadata"].get("chunk_type", "text"),
                            chunk_index=_safe_int_conversion(chunk_item["metadata"].get("chunk_index"), 0),
                            parent_id=chunk_item.get("parent_id", ""),
                            context_before=None,
                            context_after=None,
                            metadata=dict(chunk_item["metadata"])
                        )
                        for chunk_item in chunks_list
                    ],
                    tags=[]
                )
                
                doc_schema.chunks = chunk_metadata_list
                return doc_schema
            except Exception as e:
                logger.error(f"Error creating DocumentSchema: {e}")
                if output_format == "chunks":
                    # Return empty chunk list if chunks were requested
                    return cast(TextChunkList, [])
                else:
                    # Return original dict as fallback
                    return doc_dict
        except Exception as e:
            logger.error(f"Error in chunk_document_cpu: {e}")
            if return_pydantic and isinstance(document, DocumentSchema):
                return document
            elif output_format == "chunks":
                # Return empty chunk list if chunks were requested
                return cast(TextChunkList, [])
            else:
                # Return as dict
                return doc_dict
    
    # Return in appropriate format based on output_format
    if output_format == "document" and isinstance(doc_dict, Dict):
        # Try to convert to DocumentSchema as fallback
        try:
            doc_schema = DocumentSchema(
                id=doc_dict.get("id", str(uuid.uuid4())),
                content=doc_dict.get("content", ""),
                source=doc_dict.get("source", ""),
                document_type=DocumentType(doc_dict.get("document_type", "text")),
                chunks=doc_dict.get("chunks", []),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                schema_version=SchemaVersion.V2,  # Use schema_version, not version
                title=None,
                author=None,
                metadata=doc_dict.get("metadata", {}),
                embedding=None,
                embedding_model=None,
                tags=[]
            )
            return doc_schema
        except Exception as e:
            logger.error(f"Error creating DocumentSchema: {e}")
            # Fall through to dict return
    elif output_format == "chunks" and isinstance(doc_dict, Dict) and "chunks" in doc_dict:
        # Return just the chunks list
        chunks = cast(TextChunkList, doc_dict.get("chunks", []))
        return chunks
        
    # Otherwise return as dict
    return doc_dict
