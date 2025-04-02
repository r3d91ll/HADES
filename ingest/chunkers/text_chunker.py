"""
Text chunker implementation for the HADES ingest pipeline.

This module provides implementations of the Chunker interface
for splitting documents into chunks based on various strategies.
"""

import re
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from hades.ingest.interface import Chunker, Document, DocumentChunk
from hades.pipeline.interfaces import Configurable


class TextChunker(Chunker, Configurable):
    """Text chunker implementation that splits documents into chunks."""
    
    def __init__(self):
        """Initialize the text chunker."""
        self._logger = logging.getLogger(__name__)
        self._config = {}
        self._chunk_size = 1000
        self._chunk_overlap = 200
        self._chunk_strategy = "sliding_window"
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the chunker with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self._config = config
        
        # Set chunking parameters
        self._chunk_size = config.get('chunk_size', self._chunk_size)
        self._chunk_overlap = config.get('chunk_overlap', self._chunk_overlap)
        self._chunk_strategy = config.get('chunk_strategy', self._chunk_strategy)
        
        # Validate configuration
        if self._chunk_overlap >= self._chunk_size:
            self._logger.warning(
                f"Chunk overlap ({self._chunk_overlap}) must be less than chunk size "
                f"({self._chunk_size}). Setting overlap to chunk_size/2."
            )
            self._chunk_overlap = self._chunk_size // 2
    
    @property
    def component_type(self) -> str:
        """Return the type of this component."""
        return "chunker"
    
    @classmethod
    def get_plugin_type(cls) -> str:
        """Return the plugin type for this component."""
        return "chunker"
    
    @classmethod
    def get_plugin_name(cls) -> str:
        """Return the name of this plugin."""
        return "text"
    
    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Chunk a document into smaller pieces.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of document chunks
        """
        # Select chunking strategy
        if self._chunk_strategy == "sliding_window":
            chunks = self._sliding_window_chunking(document)
        elif self._chunk_strategy == "sentence":
            chunks = self._sentence_chunking(document)
        elif self._chunk_strategy == "paragraph":
            chunks = self._paragraph_chunking(document)
        else:
            self._logger.warning(
                f"Unknown chunking strategy: {self._chunk_strategy}. "
                "Falling back to sliding window."
            )
            chunks = self._sliding_window_chunking(document)
        
        return chunks
    
    def _sliding_window_chunking(self, document: Document) -> List[DocumentChunk]:
        """Chunk a document using a sliding window approach.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of document chunks
        """
        content = document.content
        doc_id = document.doc_id
        
        # If content is shorter than chunk size, return a single chunk
        if len(content) <= self._chunk_size:
            return [
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    content=content,
                    metadata={
                        **document.metadata,
                        'chunk_index': 0,
                        'chunk_strategy': 'sliding_window',
                        'chunk_size': len(content),
                        'chunk_overlap': 0,
                        'total_chunks': 1,
                    }
                )
            ]
        
        chunks = []
        
        # Calculate step size
        step_size = self._chunk_size - self._chunk_overlap
        
        # Generate chunks
        for i, start_idx in enumerate(range(0, len(content), step_size)):
            # If we're at the end of the document, break
            if start_idx >= len(content):
                break
            
            # Calculate end index
            end_idx = min(start_idx + self._chunk_size, len(content))
            
            # If this is the last chunk and it's too small, merge with previous
            if end_idx == len(content) and end_idx - start_idx < self._chunk_size // 2:
                if chunks:  # If there are previous chunks
                    # Update the last chunk to include this content
                    last_chunk = chunks[-1]
                    last_content = last_chunk.content
                    last_chunk.content = last_content + content[start_idx:]
                    
                    # Update metadata
                    last_chunk.metadata['chunk_size'] = len(last_chunk.content)
                break
            
            # Create chunk
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                content=content[start_idx:end_idx],
                metadata={
                    **document.metadata,
                    'chunk_index': i,
                    'chunk_strategy': 'sliding_window',
                    'chunk_size': end_idx - start_idx,
                    'chunk_overlap': self._chunk_overlap if i > 0 else 0,
                    'start_char': start_idx,
                    'end_char': end_idx,
                }
            )
            
            chunks.append(chunk)
        
        # Update total_chunks in metadata
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    def _sentence_chunking(self, document: Document) -> List[DocumentChunk]:
        """Chunk a document by sentences, respecting chunk size.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of document chunks
        """
        content = document.content
        doc_id = document.doc_id
        
        # If content is shorter than chunk size, return a single chunk
        if len(content) <= self._chunk_size:
            return [
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    content=content,
                    metadata={
                        **document.metadata,
                        'chunk_index': 0,
                        'chunk_strategy': 'sentence',
                        'chunk_size': len(content),
                        'chunk_overlap': 0,
                        'total_chunks': 1,
                    }
                )
            ]
        
        # Split content into sentences
        # This is a simple regex-based approach; consider using a more sophisticated
        # sentence tokenizer for production use
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, sentence in enumerate(sentences):
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed chunk size and we already have content,
            # create a new chunk
            if current_size + sentence_size > self._chunk_size and current_chunk:
                # Join sentences into a single string
                chunk_content = ' '.join(current_chunk)
                
                # Create chunk
                chunk = DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    content=chunk_content,
                    metadata={
                        **document.metadata,
                        'chunk_index': len(chunks),
                        'chunk_strategy': 'sentence',
                        'chunk_size': len(chunk_content),
                        'sentence_count': len(current_chunk),
                    }
                )
                
                chunks.append(chunk)
                
                # Start a new chunk with overlap
                overlap_size = 0
                overlap_sentences = []
                
                # Add sentences from the end of the previous chunk for overlap
                for prev_sentence in reversed(current_chunk):
                    if overlap_size + len(prev_sentence) <= self._chunk_overlap:
                        overlap_sentences.insert(0, prev_sentence)
                        overlap_size += len(prev_sentence) + 1  # +1 for space
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_size += sentence_size + 1  # +1 for space
        
        # Add the last chunk if there's anything left
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                content=chunk_content,
                metadata={
                    **document.metadata,
                    'chunk_index': len(chunks),
                    'chunk_strategy': 'sentence',
                    'chunk_size': len(chunk_content),
                    'sentence_count': len(current_chunk),
                }
            )
            
            chunks.append(chunk)
        
        # Update total_chunks and overlap in metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['total_chunks'] = len(chunks)
            chunk.metadata['chunk_overlap'] = self._chunk_overlap if i > 0 else 0
        
        return chunks
    
    def _paragraph_chunking(self, document: Document) -> List[DocumentChunk]:
        """Chunk a document by paragraphs, respecting chunk size.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of document chunks
        """
        content = document.content
        doc_id = document.doc_id
        
        # If content is shorter than chunk size, return a single chunk
        if len(content) <= self._chunk_size:
            return [
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    content=content,
                    metadata={
                        **document.metadata,
                        'chunk_index': 0,
                        'chunk_strategy': 'paragraph',
                        'chunk_size': len(content),
                        'chunk_overlap': 0,
                        'total_chunks': 1,
                    }
                )
            ]
        
        # Split content into paragraphs (double newline)
        paragraphs = re.split(r'\n\s*\n', content)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, paragraph in enumerate(paragraphs):
            paragraph_size = len(paragraph)
            
            # If this paragraph alone exceeds chunk size, use sliding window for it
            if paragraph_size > self._chunk_size:
                # First, add any accumulated paragraphs as a chunk
                if current_chunk:
                    chunk_content = '\n\n'.join(current_chunk)
                    
                    chunk = DocumentChunk(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=doc_id,
                        content=chunk_content,
                        metadata={
                            **document.metadata,
                            'chunk_index': len(chunks),
                            'chunk_strategy': 'paragraph',
                            'chunk_size': len(chunk_content),
                            'paragraph_count': len(current_chunk),
                        }
                    )
                    
                    chunks.append(chunk)
                    current_chunk = []
                    current_size = 0
                
                # Create a temporary document for this paragraph
                temp_doc = Document(
                    doc_id=doc_id,
                    content=paragraph,
                    metadata=document.metadata
                )
                
                # Use sliding window chunking for this paragraph
                paragraph_chunks = self._sliding_window_chunking(temp_doc)
                
                # Update metadata to indicate mixed strategy
                for p_chunk in paragraph_chunks:
                    p_chunk.metadata['chunk_strategy'] = 'paragraph+sliding_window'
                
                chunks.extend(paragraph_chunks)
                continue
            
            # If adding this paragraph would exceed chunk size and we already have content,
            # create a new chunk
            if current_size + paragraph_size + 2 > self._chunk_size and current_chunk:  # +2 for '\n\n'
                # Join paragraphs into a single string
                chunk_content = '\n\n'.join(current_chunk)
                
                # Create chunk
                chunk = DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    content=chunk_content,
                    metadata={
                        **document.metadata,
                        'chunk_index': len(chunks),
                        'chunk_strategy': 'paragraph',
                        'chunk_size': len(chunk_content),
                        'paragraph_count': len(current_chunk),
                    }
                )
                
                chunks.append(chunk)
                
                # Start a new chunk with overlap
                overlap_size = 0
                overlap_paragraphs = []
                
                # Add paragraphs from the end of the previous chunk for overlap
                for prev_paragraph in reversed(current_chunk):
                    if overlap_size + len(prev_paragraph) <= self._chunk_overlap:
                        overlap_paragraphs.insert(0, prev_paragraph)
                        overlap_size += len(prev_paragraph) + 2  # +2 for '\n\n'
                    else:
                        break
                
                current_chunk = overlap_paragraphs
                current_size = overlap_size
            
            # Add paragraph to current chunk
            current_chunk.append(paragraph)
            current_size += paragraph_size + (2 if current_chunk else 0)  # +2 for '\n\n' if not first paragraph
        
        # Add the last chunk if there's anything left
        if current_chunk:
            chunk_content = '\n\n'.join(current_chunk)
            
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                content=chunk_content,
                metadata={
                    **document.metadata,
                    'chunk_index': len(chunks),
                    'chunk_strategy': 'paragraph',
                    'chunk_size': len(chunk_content),
                    'paragraph_count': len(current_chunk),
                }
            )
            
            chunks.append(chunk)
        
        # Update total_chunks and overlap in metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['total_chunks'] = len(chunks)
            if 'chunk_overlap' not in chunk.metadata:  # Don't override if already set by sliding window
                chunk.metadata['chunk_overlap'] = self._chunk_overlap if i > 0 else 0
        
        return chunks
