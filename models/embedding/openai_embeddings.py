"""
OpenAI embedding model implementation for the HADES modular pipeline architecture.

This module provides an implementation of the EmbeddingModel interface
using OpenAI's embedding API.
"""

import os
import logging
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np

from openai import OpenAI

from hades.models.interface import EmbeddingModel
from hades.ingest.interface import DocumentChunk, EmbeddedChunk
from hades.pipeline.interfaces import Configurable


class OpenAIEmbeddingModel(EmbeddingModel, Configurable):
    """OpenAI implementation of the EmbeddingModel interface."""
    
    def __init__(self):
        """Initialize the OpenAI embedding model."""
        self._logger = logging.getLogger(__name__)
        self._client = None
        self._model_name = "text-embedding-ada-002"
        self._api_key = None
        self._config = {}
        self._dimensions = 1536  # Default for text-embedding-ada-002
        self._batch_size = 20
        self._retry_count = 3
        self._retry_delay = 1
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the embedding model with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self._config = config
        
        # Set model parameters
        self._model_name = config.get('model_name', self._model_name)
        self._api_key = config.get('api_key', os.environ.get('OPENAI_API_KEY'))
        self._batch_size = config.get('batch_size', self._batch_size)
        self._retry_count = config.get('retry_count', self._retry_count)
        self._retry_delay = config.get('retry_delay', self._retry_delay)
        
        # Set dimensions based on model
        if self._model_name == "text-embedding-ada-002":
            self._dimensions = 1536
        elif self._model_name == "text-embedding-3-small":
            self._dimensions = 1536
        elif self._model_name == "text-embedding-3-large":
            self._dimensions = 3072
        else:
            self._dimensions = config.get('dimensions', self._dimensions)
        
        # Initialize client if auto_load is True
        if config.get('auto_load', True):
            self.load()
    
    @property
    def component_type(self) -> str:
        """Return the type of this component."""
        return "embedding_model"
    
    @classmethod
    def get_plugin_type(cls) -> str:
        """Return the plugin type for this component."""
        return "embedding_model"
    
    @classmethod
    def get_plugin_name(cls) -> str:
        """Return the name of this plugin."""
        return "openai"
    
    def load(self) -> bool:
        """Load the embedding model.
        
        Returns:
            True if loading successful, False otherwise
        """
        try:
            if not self._api_key:
                raise ValueError("OpenAI API key not provided")
            
            self._client = OpenAI(api_key=self._api_key)
            
            self._logger.info(f"Initialized OpenAI embedding model: {self._model_name}")
            return True
        
        except Exception as e:
            self._logger.error(f"Error loading OpenAI embedding model: {e}")
            return False
    
    def unload(self) -> bool:
        """Unload the embedding model.
        
        Returns:
            True if unloading successful, False otherwise
        """
        self._client = None
        return True
    
    def is_loaded(self) -> bool:
        """Check if the embedding model is loaded.
        
        Returns:
            True if loaded, False otherwise
        """
        return self._client is not None
    
    def get_dimensions(self) -> int:
        """Get the dimensions of the embedding vectors.
        
        Returns:
            Number of dimensions
        """
        return self._dimensions
    
    def get_model_name(self) -> str:
        """Get the name of the embedding model.
        
        Returns:
            Model name
        """
        return self._model_name
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate an embedding for a text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
            
        Raises:
            RuntimeError: If embedding fails
        """
        if not self.is_loaded():
            if not self.load():
                raise RuntimeError("Failed to load embedding model")
        
        for attempt in range(self._retry_count):
            try:
                response = self._client.embeddings.create(
                    model=self._model_name,
                    input=text
                )
                
                embedding = response.data[0].embedding
                return np.array(embedding)
            
            except Exception as e:
                self._logger.warning(f"Embedding attempt {attempt+1} failed: {e}")
                if attempt < self._retry_count - 1:
                    time.sleep(self._retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise RuntimeError(f"Failed to generate embedding after {self._retry_count} attempts: {e}")
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple text strings.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            RuntimeError: If embedding fails
        """
        if not self.is_loaded():
            if not self.load():
                raise RuntimeError("Failed to load embedding model")
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            
            for attempt in range(self._retry_count):
                try:
                    response = self._client.embeddings.create(
                        model=self._model_name,
                        input=batch
                    )
                    
                    # Sort by index to ensure correct order
                    sorted_data = sorted(response.data, key=lambda x: x.index)
                    batch_embeddings = [np.array(item.embedding) for item in sorted_data]
                    embeddings.extend(batch_embeddings)
                    break
                
                except Exception as e:
                    self._logger.warning(f"Batch embedding attempt {attempt+1} failed: {e}")
                    if attempt < self._retry_count - 1:
                        time.sleep(self._retry_delay * (2 ** attempt))  # Exponential backoff
                    else:
                        raise RuntimeError(f"Failed to generate batch embeddings after {self._retry_count} attempts: {e}")
        
        return embeddings
    
    def embed_chunk(self, chunk: DocumentChunk) -> EmbeddedChunk:
        """Generate an embedding for a document chunk.
        
        Args:
            chunk: Document chunk to embed
            
        Returns:
            Embedded chunk
            
        Raises:
            RuntimeError: If embedding fails
        """
        embedding = self.embed_text(chunk.content)
        
        return EmbeddedChunk(
            embedding_id=f"{chunk.chunk_id}_emb",
            chunk_id=chunk.chunk_id,
            embedding=embedding,
            model_name=self._model_name,
            metadata={
                **chunk.metadata,
                'embedding_model': self._model_name,
                'embedding_dimensions': self._dimensions,
            }
        )
    
    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[EmbeddedChunk]:
        """Generate embeddings for multiple document chunks.
        
        Args:
            chunks: List of document chunks to embed
            
        Returns:
            List of embedded chunks
            
        Raises:
            RuntimeError: If embedding fails
        """
        # Extract text content from chunks
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(texts)
        
        # Create embedded chunks
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunk = EmbeddedChunk(
                embedding_id=f"{chunk.chunk_id}_emb",
                chunk_id=chunk.chunk_id,
                embedding=embedding,
                model_name=self._model_name,
                metadata={
                    **chunk.metadata,
                    'embedding_model': self._model_name,
                    'embedding_dimensions': self._dimensions,
                }
            )
            embedded_chunks.append(embedded_chunk)
        
        return embedded_chunks
