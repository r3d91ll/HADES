"""
Text chunking type definitions for the HADES-PathRAG chunking system.

This module provides specialized type definitions for text chunking operations,
including text-specific chunk types, metadata, and processing options.
"""

from typing import Dict, List, Any, Optional, Union, TypedDict, Literal
from pydantic import BaseModel, Field
import uuid
from datetime import datetime

from src.types.chunking.chunk import (
    Chunk, ChunkMetadata, ChunkList, ChunkContent, ChunkId,
    ChunkableDocument, OutputFormatType
)


class TextChunkMetadata(ChunkMetadata):
    """Metadata specific to text chunks."""
    # Text-specific metadata
    paragraph_type: Optional[str]  # Type of paragraph (e.g., "heading", "paragraph", "list")
    sentence_count: Optional[int]  # Number of sentences in the chunk
    semantic_score: Optional[float]  # Semantic coherence score
    split_reason: Optional[str]  # Reason for splitting at this point
    confidence: Optional[float]  # Confidence score for the chunk boundaries


class TextChunk(Chunk):
    """A chunk of text content with text-specific metadata."""
    content: str  # Text content
    metadata: TextChunkMetadata  # Text-specific metadata


class TextChunkingOptions(TypedDict, total=False):
    """Options for text chunking."""
    max_tokens: int  # Maximum tokens per chunk
    min_tokens: Optional[int]  # Minimum tokens per chunk
    split_method: Literal["semantic", "fixed", "sentence"]  # Method for splitting text
    overlap_tokens: Optional[int]  # Number of tokens to overlap between chunks
    model_id: Optional[str]  # Model ID to use for semantic chunking
    preserve_line_breaks: bool  # Whether to preserve line breaks
    preserve_paragraphs: bool  # Whether to preserve paragraph breaks
    device: Optional[str]  # Device to use for semantic chunking
    batch_size: Optional[int]  # Batch size for processing multiple documents
    output_format: OutputFormatType  # Output format type
    verbose: bool  # Whether to output verbose logging


class TextChunkingResult(TypedDict):
    """Result of a text chunking operation."""
    document_id: str  # ID of the original document
    chunks: List[TextChunk]  # List of text chunks
    metadata: Dict[str, Any]  # Metadata about the chunking operation
    processing_time: float  # Time taken to process the document


class SplitPoint(TypedDict):
    """A point where text can be split."""
    index: int  # Character index in the text
    score: float  # Score indicating the quality of the split
    type: str  # Type of split (e.g., "paragraph", "sentence")


class TextChunkerConfig(TypedDict, total=False):
    """Configuration for a text chunker."""
    max_tokens: int  # Maximum tokens per chunk
    min_tokens: Optional[int]  # Minimum tokens per chunk
    model_id: str  # Model ID for semantic chunking
    cache_dir: Optional[str]  # Directory to cache models
    use_gpu: bool  # Whether to use GPU for processing
    device: Optional[str]  # Device to use for processing
    batch_size: int  # Batch size for processing multiple documents
    overlap_tokens: int  # Number of tokens to overlap between chunks
    preserve_line_breaks: bool  # Whether to preserve line breaks
    preserve_paragraphs: bool  # Whether to preserve paragraph breaks
    split_method: str  # Method for splitting text
    fallback_splitter: Optional[str]  # Fallback splitter to use if main fails


class SentenceInfo(TypedDict, total=False):
    """Information about a sentence."""
    text: str  # The sentence text
    start: int  # Start index in the original text
    end: int  # End index in the original text
    tokens: int  # Number of tokens in the sentence
    embedding: Optional[List[float]]  # Sentence embedding if computed


class ParagraphInfo(TypedDict, total=False):
    """Information about a paragraph."""
    text: str  # The paragraph text
    start: int  # Start index in the original text
    end: int  # End index in the original text
    sentences: List[SentenceInfo]  # Sentences in the paragraph
    tokens: int  # Number of tokens in the paragraph
    embedding: Optional[List[float]]  # Paragraph embedding if computed
    score: Optional[float]  # Semantic coherence score


class TokenInfo(TypedDict):
    """Information about a token."""
    text: str  # The token text
    start: int  # Start index in the original text
    end: int  # End index in the original text
    id: int  # Token ID in the tokenizer vocabulary


class DocumentTextStats(TypedDict, total=False):
    """Statistics about a text document."""
    total_tokens: int  # Total number of tokens
    total_chars: int  # Total number of characters
    total_sentences: int  # Total number of sentences
    total_paragraphs: int  # Total number of paragraphs
    average_sentence_length: float  # Average sentence length in tokens
    average_paragraph_length: float  # Average paragraph length in tokens
    language: Optional[str]  # Detected language


# Type aliases for clarity
TextChunkList = List[TextChunk]
SplitPointList = List[SplitPoint]
SentenceList = List[SentenceInfo]
ParagraphList = List[ParagraphInfo]


# Helper function to create text-specific chunk metadata
def create_text_chunk_metadata(
    chunk_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    path: str = "unknown",
    document_type: str = "text",
    source_file: str = "unknown",
    line_start: int = 0,
    line_end: int = 0,
    token_count: int = 0,
    paragraph_type: Optional[str] = None,
    sentence_count: Optional[int] = None,
    semantic_score: Optional[float] = None,
    split_reason: Optional[str] = None,
    confidence: Optional[float] = None,
    **kwargs: Any
) -> TextChunkMetadata:
    """Create text-specific chunk metadata.
    
    Args:
        chunk_id: Unique identifier for the chunk
        parent_id: ID of the parent document
        path: Path to the source document
        document_type: Type of document
        source_file: Original source file
        line_start: Starting line number
        line_end: Ending line number
        token_count: Number of tokens in the chunk
        paragraph_type: Type of paragraph
        sentence_count: Number of sentences in the chunk
        semantic_score: Semantic coherence score
        split_reason: Reason for splitting at this point
        confidence: Confidence score for the chunk boundaries
        **kwargs: Additional metadata fields
    
    Returns:
        A TextChunkMetadata dictionary
    """
    # Create base metadata
    metadata: TextChunkMetadata = {
        "id": chunk_id or str(uuid.uuid4()),
        "parent_id": parent_id,
        "path": path,
        "document_type": document_type,
        "source_file": source_file,
        "line_start": line_start,
        "line_end": line_end,
        "token_count": token_count,
        "chunk_type": "text",
        "created_at": datetime.now().isoformat()
    }
    
    # Add text-specific fields if provided
    if paragraph_type is not None:
        metadata["paragraph_type"] = paragraph_type
    if sentence_count is not None:
        metadata["sentence_count"] = sentence_count
    if semantic_score is not None:
        metadata["semantic_score"] = semantic_score
    if split_reason is not None:
        metadata["split_reason"] = split_reason
    if confidence is not None:
        metadata["confidence"] = confidence
    
    # Add any additional metadata
    for key, value in kwargs.items():
        metadata[key] = value
    
    return metadata
