"""
Ingest pipeline interfaces for the HADES modular pipeline architecture.

This module defines the interfaces for the ingest pipeline components,
including document processors, chunkers, and embedders.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
import numpy as np
from pathlib import Path

from hades.pipeline.interfaces import Component, Configurable, Pipeline, Pluggable, Serializable


class Document:
    """Class representing a document in the ingest pipeline."""
    
    def __init__(self, document_id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Initialize a document.
        
        Args:
            document_id: Unique identifier for the document
            content: Document content
            metadata: Optional metadata for the document
        """
        self.document_id = document_id
        self.content = content
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"Document(id={self.document_id}, metadata={self.metadata})"


class DocumentChunk:
    """Class representing a chunk of a document in the ingest pipeline."""
    
    def __init__(self, chunk_id: str, document_id: str, content: str, 
                 metadata: Optional[Dict[str, Any]] = None):
        """Initialize a document chunk.
        
        Args:
            chunk_id: Unique identifier for the chunk
            document_id: ID of the parent document
            content: Chunk content
            metadata: Optional metadata for the chunk
        """
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.content = content
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"DocumentChunk(id={self.chunk_id}, doc_id={self.document_id})"


class EmbeddedChunk:
    """Class representing an embedded chunk in the ingest pipeline."""
    
    def __init__(self, chunk_id: str, document_id: str, content: str, 
                 embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """Initialize an embedded chunk.
        
        Args:
            chunk_id: Unique identifier for the chunk
            document_id: ID of the parent document
            content: Chunk content
            embedding: Vector embedding of the chunk
            metadata: Optional metadata for the chunk
        """
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.content = content
        self.embedding = embedding
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"EmbeddedChunk(id={self.chunk_id}, doc_id={self.document_id})"


class DocumentProcessor(Component, Configurable, Pluggable, ABC):
    """Interface for document processors in the ingest pipeline."""
    
    @property
    def component_type(self) -> str:
        """Return the type of this component."""
        return "document_processor"
    
    @classmethod
    def get_plugin_type(cls) -> str:
        """Return the plugin type for this component."""
        return "document_processor"
    
    @abstractmethod
    def process_file(self, file_path: Path) -> Document:
        """Process a file and convert it to a document.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Processed document
        """
        pass
    
    @abstractmethod
    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        """Process text and convert it to a document.
        
        Args:
            text: Text to process
            metadata: Optional metadata for the document
            
        Returns:
            Processed document
        """
        pass
    
    @abstractmethod
    def process_directory(self, directory_path: Path, 
                         recursive: bool = True, 
                         file_extensions: Optional[List[str]] = None) -> List[Document]:
        """Process all files in a directory and convert them to documents.
        
        Args:
            directory_path: Path to the directory to process
            recursive: Whether to process subdirectories
            file_extensions: Optional list of file extensions to process
            
        Returns:
            List of processed documents
        """
        pass


class DocumentChunker(Component, Configurable, Pluggable, ABC):
    """Interface for document chunkers in the ingest pipeline."""
    
    @property
    def component_type(self) -> str:
        """Return the type of this component."""
        return "document_chunker"
    
    @classmethod
    def get_plugin_type(cls) -> str:
        """Return the plugin type for this component."""
        return "document_chunker"
    
    @abstractmethod
    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Split a document into chunks.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of document chunks
        """
        pass
    
    @abstractmethod
    def chunk_text(self, text: str, document_id: str, 
                  metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """Split text into chunks.
        
        Args:
            text: Text to chunk
            document_id: ID to associate with the chunks
            metadata: Optional metadata for the chunks
            
        Returns:
            List of document chunks
        """
        pass


class DocumentEmbedder(Component, Configurable, Pluggable, ABC):
    """Interface for document embedders in the ingest pipeline."""
    
    @property
    def component_type(self) -> str:
        """Return the type of this component."""
        return "document_embedder"
    
    @classmethod
    def get_plugin_type(cls) -> str:
        """Return the plugin type for this component."""
        return "document_embedder"
    
    @abstractmethod
    def embed_chunks(self, chunks: List[DocumentChunk], 
                    batch_size: Optional[int] = None) -> List[EmbeddedChunk]:
        """Generate embeddings for a list of document chunks.
        
        Args:
            chunks: List of document chunks to embed
            batch_size: Optional batch size for processing
            
        Returns:
            List of embedded chunks
        """
        pass
    
    @abstractmethod
    def embed_chunk(self, chunk: DocumentChunk) -> EmbeddedChunk:
        """Generate embedding for a single document chunk.
        
        Args:
            chunk: Document chunk to embed
            
        Returns:
            Embedded chunk
        """
        pass
    
    @property
    @abstractmethod
    def model(self) -> Any:
        """Return the underlying embedding model."""
        pass


class IngestPipeline(Pipeline, ABC):
    """Interface for the ingest pipeline."""
    
    @property
    def component_type(self) -> str:
        """Return the type of this component."""
        return "ingest_pipeline"
    
    @abstractmethod
    def process_file(self, file_path: Path) -> List[EmbeddedChunk]:
        """Process a file through the entire ingest pipeline.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of embedded chunks
        """
        pass
    
    @abstractmethod
    def process_directory(self, directory_path: Path, 
                         recursive: bool = True, 
                         file_extensions: Optional[List[str]] = None) -> List[EmbeddedChunk]:
        """Process all files in a directory through the entire ingest pipeline.
        
        Args:
            directory_path: Path to the directory to process
            recursive: Whether to process subdirectories
            file_extensions: Optional list of file extensions to process
            
        Returns:
            List of embedded chunks
        """
        pass
    
    @abstractmethod
    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[EmbeddedChunk]:
        """Process text through the entire ingest pipeline.
        
        Args:
            text: Text to process
            metadata: Optional metadata for the document
            
        Returns:
            List of embedded chunks
        """
        pass
    
    @abstractmethod
    def store_embedded_chunks(self, chunks: List[EmbeddedChunk]) -> bool:
        """Store embedded chunks in the storage backend.
        
        Args:
            chunks: List of embedded chunks to store
            
        Returns:
            True if storage successful, False otherwise
        """
        pass