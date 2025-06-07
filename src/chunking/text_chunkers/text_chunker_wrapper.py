"""Wrapper classes for text chunkers to integrate with the chunking registry.

This module provides wrapper classes that make the functional text chunkers
compatible with the class-based chunking registry system.
"""

from typing import Any, Dict, List, Optional, Union
import logging

from src.chunking.text_chunkers.cpu_chunker import chunk_text_cpu
from src.chunking.text_chunkers.chonky_chunker import chunk_text

logger = logging.getLogger(__name__)


class CPUTextChunker:
    """Wrapper class for CPU-based text chunking."""
    
    def __init__(self, **kwargs: Any):
        """Initialize the CPU text chunker with configuration."""
        self.config = kwargs
        self.max_tokens = kwargs.get("max_tokens", 2048)
        self.num_workers = kwargs.get("num_workers", 4)
        self.model_id = kwargs.get("model_id", "mirth/chonky_modernbert_large_1")
        
    def chunk(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk the input text using CPU-based chunking.
        
        Args:
            input_data: Dictionary containing:
                - text: Text content to chunk
                - document_id: Document ID (optional)
                - file_path: File path (optional)
                - metadata: Additional metadata (optional)
                
        Returns:
            List of chunk dictionaries
        """
        text = input_data.get("text", "")
        doc_id = input_data.get("document_id")
        file_path = input_data.get("file_path", "unknown")
        
        if not text:
            logger.warning("No text content provided for chunking")
            return []
        
        try:
            result = chunk_text_cpu(
                content=text,
                doc_id=doc_id,
                path=file_path,
                doc_type="text",
                max_tokens=self.max_tokens,
                output_format="dict",
                model_id=self.model_id,
                num_workers=self.num_workers
            )
            
            # Extract chunks from the result
            chunks = []
            if isinstance(result, dict) and "chunks" in result:
                raw_chunks = result["chunks"]
                # Convert chunks to expected format (text field instead of content)
                for chunk in raw_chunks:
                    if isinstance(chunk, dict):
                        # Create a new chunk dict with 'text' field
                        converted_chunk = dict(chunk)
                        if "content" in converted_chunk:
                            converted_chunk["text"] = converted_chunk.pop("content")
                        chunks.append(converted_chunk)
                return chunks
            elif isinstance(result, list):
                # Handle list format directly
                for chunk in result:
                    if isinstance(chunk, dict):
                        converted_chunk = dict(chunk) 
                        if "content" in converted_chunk:
                            converted_chunk["text"] = converted_chunk.pop("content")
                        chunks.append(converted_chunk)
                return chunks
            else:
                logger.warning(f"Unexpected result format from CPU chunker: {type(result)}")
                return []
                
        except Exception as e:
            logger.error(f"CPU chunking failed: {e}")
            return []


class ChonkyTextChunker:
    """Wrapper class for Chonky-based text chunking."""
    
    def __init__(self, **kwargs: Any):
        """Initialize the Chonky text chunker with configuration."""
        self.config = kwargs
        self.max_tokens = kwargs.get("max_tokens", 2048)
        self.model_id = kwargs.get("model_id", "mirth/chonky_modernbert_large_1")
        
    def chunk(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk the input text using Chonky neural chunking.
        
        Args:
            input_data: Dictionary containing:
                - text: Text content to chunk
                - document_id: Document ID (optional)
                - file_path: File path (optional) 
                - metadata: Additional metadata (optional)
                
        Returns:
            List of chunk dictionaries
        """
        text = input_data.get("text", "")
        doc_id = input_data.get("document_id")
        file_path = input_data.get("file_path", "unknown")
        
        if not text:
            logger.warning("No text content provided for chunking")
            return []
        
        try:
            result = chunk_text(
                content=text,
                doc_id=doc_id,
                path=file_path,
                doc_type="text",
                max_tokens=self.max_tokens,
                output_format="dict",
                model_id=self.model_id
            )
            
            # Extract chunks from the result
            chunks = []
            if isinstance(result, dict) and "chunks" in result:
                raw_chunks = result["chunks"]
                # Convert chunks to expected format (text field instead of content)
                for chunk in raw_chunks:
                    if isinstance(chunk, dict):
                        # Create a new chunk dict with 'text' field
                        converted_chunk = dict(chunk)
                        if "content" in converted_chunk:
                            converted_chunk["text"] = converted_chunk.pop("content")
                        chunks.append(converted_chunk)
                return chunks
            elif isinstance(result, list):
                # Handle list format directly
                for chunk in result:
                    if isinstance(chunk, dict):
                        converted_chunk = dict(chunk)
                        if "content" in converted_chunk:
                            converted_chunk["text"] = converted_chunk.pop("content")
                        chunks.append(converted_chunk)
                return chunks
            else:
                logger.warning(f"Unexpected result format from Chonky chunker: {type(result)}")
                return []
                
        except Exception as e:
            logger.error(f"Chonky chunking failed: {e}")
            return []