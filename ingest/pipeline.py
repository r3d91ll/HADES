"""
Ingest pipeline implementation for the HADES modular pipeline architecture.

This module provides the implementation of the ingest pipeline that
orchestrates document processing, chunking, and embedding.
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

from hades.ingest.interface import (
    Document, DocumentChunk, EmbeddedChunk,
    DocumentProcessor, Chunker, Embedder, IngestPipeline
)
from hades.storage.interface import DocumentStore, ChunkStore, EmbeddingStore
from hades.models.interface import EmbeddingModel
from hades.pipeline.interfaces import Configurable, Pipeline
from hades.pipeline import registry, config


class ModularIngestPipeline(IngestPipeline, Configurable):
    """Modular implementation of the ingest pipeline."""
    
    def __init__(self):
        """Initialize the ingest pipeline."""
        self._logger = logging.getLogger(__name__)
        self._config = {}
        self._document_processor = None
        self._chunker = None
        self._embedding_model = None
        self._document_store = None
        self._chunk_store = None
        self._embedding_store = None
        self._initialized = False
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the ingest pipeline with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self._config = config
        
        # Load components from registry
        self._load_components()
        
        self._initialized = True
        self._logger.info("Initialized ingest pipeline")
    
    def _load_components(self) -> None:
        """Load pipeline components from registry based on configuration."""
        # Get component configurations
        processor_config = self._config.get('document_processor', {})
        chunker_config = self._config.get('chunker', {})
        embedding_config = self._config.get('embedding_model', {})
        document_store_config = self._config.get('document_store', {})
        chunk_store_config = self._config.get('chunk_store', {})
        embedding_store_config = self._config.get('embedding_store', {})
        
        # Get component types from configuration
        processor_type = processor_config.get('type', 'basic')
        chunker_type = chunker_config.get('type', 'text')
        embedding_type = embedding_config.get('type', 'openai')
        document_store_type = document_store_config.get('type', 'redis')
        chunk_store_type = chunk_store_config.get('type', 'redis')
        embedding_store_type = embedding_store_config.get('type', 'redis')
        
        # Load components from registry
        self._document_processor = registry.get_plugin_instance(
            'document_processor', processor_type, processor_config
        )
        if not self._document_processor:
            raise ValueError(f"Document processor '{processor_type}' not found in registry")
        
        self._chunker = registry.get_plugin_instance(
            'chunker', chunker_type, chunker_config
        )
        if not self._chunker:
            raise ValueError(f"Chunker '{chunker_type}' not found in registry")
        
        self._embedding_model = registry.get_plugin_instance(
            'embedding_model', embedding_type, embedding_config
        )
        if not self._embedding_model:
            raise ValueError(f"Embedding model '{embedding_type}' not found in registry")
        
        self._document_store = registry.get_plugin_instance(
            'document_store', document_store_type, document_store_config
        )
        if not self._document_store:
            raise ValueError(f"Document store '{document_store_type}' not found in registry")
        
        self._chunk_store = registry.get_plugin_instance(
            'chunk_store', chunk_store_type, chunk_store_config
        )
        if not self._chunk_store:
            raise ValueError(f"Chunk store '{chunk_store_type}' not found in registry")
        
        self._embedding_store = registry.get_plugin_instance(
            'embedding_store', embedding_store_type, embedding_store_config
        )
        if not self._embedding_store:
            raise ValueError(f"Embedding store '{embedding_store_type}' not found in registry")
    
    @property
    def component_type(self) -> str:
        """Return the type of this component."""
        return "ingest_pipeline"
    
    def ingest_file(self, file_path: Union[str, Path], store: bool = True) -> Document:
        """Ingest a file into the pipeline.
        
        Args:
            file_path: Path to the file to ingest
            store: Whether to store the document and its derivatives
            
        Returns:
            Processed document
            
        Raises:
            ValueError: If the file cannot be processed
        """
        if not self._initialized:
            raise RuntimeError("Ingest pipeline not initialized")
        
        file_path = Path(file_path)
        
        # Process document
        document = self._document_processor.process_file(file_path)
        
        # Store document if requested
        if store:
            self._document_store.store_document(document)
        
        return document
    
    def ingest_text(self, text: str, metadata: Optional[Dict[str, Any]] = None, 
                   store: bool = True) -> Document:
        """Ingest raw text into the pipeline.
        
        Args:
            text: Raw text to ingest
            metadata: Optional metadata for the document
            store: Whether to store the document and its derivatives
            
        Returns:
            Processed document
        """
        if not self._initialized:
            raise RuntimeError("Ingest pipeline not initialized")
        
        # Process document
        document = self._document_processor.process_text(text, metadata)
        
        # Store document if requested
        if store:
            self._document_store.store_document(document)
        
        return document
    
    def ingest_directory(self, directory_path: Union[str, Path], recursive: bool = True,
                        store: bool = True) -> List[Document]:
        """Ingest all supported files in a directory.
        
        Args:
            directory_path: Path to the directory to ingest
            recursive: Whether to process subdirectories recursively
            store: Whether to store documents and their derivatives
            
        Returns:
            List of processed documents
        """
        if not self._initialized:
            raise RuntimeError("Ingest pipeline not initialized")
        
        directory_path = Path(directory_path)
        
        # Process documents
        documents = self._document_processor.process_directory(directory_path, recursive)
        
        # Store documents if requested
        if store:
            for document in documents:
                self._document_store.store_document(document)
        
        return documents
    
    def chunk_document(self, document: Document, store: bool = True) -> List[DocumentChunk]:
        """Chunk a document into smaller pieces.
        
        Args:
            document: Document to chunk
            store: Whether to store the chunks
            
        Returns:
            List of document chunks
        """
        if not self._initialized:
            raise RuntimeError("Ingest pipeline not initialized")
        
        # Chunk document
        chunks = self._chunker.chunk_document(document)
        
        # Store chunks if requested
        if store:
            for chunk in chunks:
                self._chunk_store.store_chunk(chunk)
        
        return chunks
    
    def embed_chunk(self, chunk: DocumentChunk, store: bool = True) -> EmbeddedChunk:
        """Generate an embedding for a document chunk.
        
        Args:
            chunk: Document chunk to embed
            store: Whether to store the embedding
            
        Returns:
            Embedded chunk
        """
        if not self._initialized:
            raise RuntimeError("Ingest pipeline not initialized")
        
        # Generate embedding
        embedded_chunk = self._embedding_model.embed_chunk(chunk)
        
        # Store embedding if requested
        if store:
            self._embedding_store.store_embedding(embedded_chunk)
        
        return embedded_chunk
    
    def embed_chunks(self, chunks: List[DocumentChunk], store: bool = True) -> List[EmbeddedChunk]:
        """Generate embeddings for multiple document chunks.
        
        Args:
            chunks: List of document chunks to embed
            store: Whether to store the embeddings
            
        Returns:
            List of embedded chunks
        """
        if not self._initialized:
            raise RuntimeError("Ingest pipeline not initialized")
        
        # Generate embeddings
        embedded_chunks = self._embedding_model.embed_chunks(chunks)
        
        # Store embeddings if requested
        if store:
            for embedded_chunk in embedded_chunks:
                self._embedding_store.store_embedding(embedded_chunk)
        
        return embedded_chunks
    
    def process_document(self, document: Document, store: bool = True) -> Tuple[Document, List[DocumentChunk], List[EmbeddedChunk]]:
        """Process a document through the entire pipeline.
        
        Args:
            document: Document to process
            store: Whether to store the document and its derivatives
            
        Returns:
            Tuple of (document, chunks, embedded_chunks)
        """
        if not self._initialized:
            raise RuntimeError("Ingest pipeline not initialized")
        
        # Store document if requested
        if store:
            self._document_store.store_document(document)
        
        # Chunk document
        chunks = self.chunk_document(document, store)
        
        # Generate embeddings
        embedded_chunks = self.embed_chunks(chunks, store)
        
        return document, chunks, embedded_chunks
    
    def process_file(self, file_path: Union[str, Path], store: bool = True) -> Tuple[Document, List[DocumentChunk], List[EmbeddedChunk]]:
        """Process a file through the entire pipeline.
        
        Args:
            file_path: Path to the file to process
            store: Whether to store the document and its derivatives
            
        Returns:
            Tuple of (document, chunks, embedded_chunks)
        """
        if not self._initialized:
            raise RuntimeError("Ingest pipeline not initialized")
        
        # Process document
        document = self.ingest_file(file_path, store)
        
        # Chunk document
        chunks = self.chunk_document(document, store)
        
        # Generate embeddings
        embedded_chunks = self.embed_chunks(chunks, store)
        
        return document, chunks, embedded_chunks
    
    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None,
                    store: bool = True) -> Tuple[Document, List[DocumentChunk], List[EmbeddedChunk]]:
        """Process raw text through the entire pipeline.
        
        Args:
            text: Raw text to process
            metadata: Optional metadata for the document
            store: Whether to store the document and its derivatives
            
        Returns:
            Tuple of (document, chunks, embedded_chunks)
        """
        if not self._initialized:
            raise RuntimeError("Ingest pipeline not initialized")
        
        # Process document
        document = self.ingest_text(text, metadata, store)
        
        # Chunk document
        chunks = self.chunk_document(document, store)
        
        # Generate embeddings
        embedded_chunks = self.embed_chunks(chunks, store)
        
        return document, chunks, embedded_chunks
    
    def process_directory(self, directory_path: Union[str, Path], recursive: bool = True,
                         store: bool = True) -> List[Tuple[Document, List[DocumentChunk], List[EmbeddedChunk]]]:
        """Process all supported files in a directory through the entire pipeline.
        
        Args:
            directory_path: Path to the directory to process
            recursive: Whether to process subdirectories recursively
            store: Whether to store documents and their derivatives
            
        Returns:
            List of tuples of (document, chunks, embedded_chunks)
        """
        if not self._initialized:
            raise RuntimeError("Ingest pipeline not initialized")
        
        # Process documents
        documents = self.ingest_directory(directory_path, recursive, store)
        
        results = []
        for document in documents:
            # Chunk document
            chunks = self.chunk_document(document, store)
            
            # Generate embeddings
            embedded_chunks = self.embed_chunks(chunks, store)
            
            results.append((document, chunks, embedded_chunks))
        
        return results
