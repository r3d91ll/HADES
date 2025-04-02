"""
Redis storage implementation for the HADES modular pipeline architecture.

This module provides Redis-based implementations of the storage interfaces
defined in hades.storage.interface.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid
from pathlib import Path

import redis
import numpy as np

from hades.storage.interface import (
    StorageBackend, 
    DocumentStore, 
    ChunkStore, 
    EmbeddingStore,
    Document,
    DocumentChunk,
    EmbeddedChunk
)
from hades.pipeline.interfaces import Configurable


class RedisStorageBackend(StorageBackend, Configurable):
    """Redis implementation of the StorageBackend interface."""
    
    def __init__(self):
        """Initialize the Redis storage backend."""
        self._client = None
        self._logger = logging.getLogger(__name__)
        self._connected = False
        self._config = {}
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the Redis storage backend with configuration.
        
        Args:
            config: Configuration dictionary with Redis connection parameters
        """
        self._config = config
        
        # Connect if auto_connect is True
        if config.get('auto_connect', True):
            self.connect()
    
    def connect(self) -> bool:
        """Connect to the Redis server.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            host = self._config.get('host', 'localhost')
            port = self._config.get('port', 6379)
            db = self._config.get('db', 0)
            password = self._config.get('password', None)
            
            self._client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False  # We need bytes for binary data
            )
            
            # Test connection
            self._client.ping()
            self._connected = True
            self._logger.info(f"Connected to Redis at {host}:{port} (db: {db})")
            return True
        
        except redis.RedisError as e:
            self._logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from the Redis server.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        if self._client:
            try:
                self._client.close()
                self._connected = False
                self._logger.info("Disconnected from Redis")
                return True
            except redis.RedisError as e:
                self._logger.error(f"Error disconnecting from Redis: {e}")
                return False
        return True
    
    def is_connected(self) -> bool:
        """Check if connected to the Redis server.
        
        Returns:
            True if connected, False otherwise
        """
        if not self._client or not self._connected:
            return False
        
        try:
            self._client.ping()
            return True
        except redis.RedisError:
            self._connected = False
            return False
    
    @property
    def client(self) -> redis.Redis:
        """Get the Redis client.
        
        Returns:
            Redis client
        
        Raises:
            ConnectionError: If not connected to Redis
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to Redis")
        return self._client


class RedisDocumentStore(DocumentStore, Configurable):
    """Redis implementation of the DocumentStore interface."""
    
    def __init__(self, backend: Optional[RedisStorageBackend] = None):
        """Initialize the Redis document store.
        
        Args:
            backend: Optional Redis storage backend to use
        """
        self._backend = backend
        self._logger = logging.getLogger(__name__)
        self._doc_key_prefix = "doc:"
        self._doc_ids_key = "document_ids"
        self._config = {}
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the Redis document store with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self._config = config
        
        # Set key prefixes
        self._doc_key_prefix = config.get('doc_key_prefix', self._doc_key_prefix)
        self._doc_ids_key = config.get('doc_ids_key', self._doc_ids_key)
        
        # Create backend if not provided
        if not self._backend:
            self._backend = RedisStorageBackend()
            self._backend.initialize(config.get('redis', {}))
    
    def _get_doc_key(self, doc_id: str) -> str:
        """Get the Redis key for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Redis key for the document
        """
        return f"{self._doc_key_prefix}{doc_id}"
    
    def store_document(self, document: Document) -> bool:
        """Store a document in Redis.
        
        Args:
            document: Document to store
            
        Returns:
            True if storage successful, False otherwise
        """
        try:
            client = self._backend.client
            doc_key = self._get_doc_key(document.doc_id)
            
            # Serialize document
            doc_data = {
                'doc_id': document.doc_id,
                'content': document.content,
                'metadata': document.metadata
            }
            
            # Store document
            client.set(doc_key, json.dumps(doc_data).encode('utf-8'))
            
            # Add to document IDs set
            client.sadd(self._doc_ids_key, document.doc_id)
            
            self._logger.debug(f"Stored document: {document.doc_id}")
            return True
        
        except (redis.RedisError, ConnectionError) as e:
            self._logger.error(f"Error storing document {document.doc_id}: {e}")
            return False
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document from Redis.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document if found, None otherwise
        """
        try:
            client = self._backend.client
            doc_key = self._get_doc_key(doc_id)
            
            # Get document data
            doc_data_bytes = client.get(doc_key)
            if not doc_data_bytes:
                return None
            
            # Deserialize document
            doc_data = json.loads(doc_data_bytes.decode('utf-8'))
            
            return Document(
                doc_id=doc_data['doc_id'],
                content=doc_data['content'],
                metadata=doc_data['metadata']
            )
        
        except (redis.RedisError, ConnectionError, json.JSONDecodeError) as e:
            self._logger.error(f"Error getting document {doc_id}: {e}")
            return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from Redis.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            client = self._backend.client
            doc_key = self._get_doc_key(doc_id)
            
            # Delete document
            client.delete(doc_key)
            
            # Remove from document IDs set
            client.srem(self._doc_ids_key, doc_id)
            
            self._logger.debug(f"Deleted document: {doc_id}")
            return True
        
        except (redis.RedisError, ConnectionError) as e:
            self._logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    def list_documents(self) -> List[str]:
        """List all document IDs in Redis.
        
        Returns:
            List of document IDs
        """
        try:
            client = self._backend.client
            
            # Get document IDs from set
            doc_ids = client.smembers(self._doc_ids_key)
            
            return [doc_id.decode('utf-8') for doc_id in doc_ids]
        
        except (redis.RedisError, ConnectionError) as e:
            self._logger.error(f"Error listing documents: {e}")
            return []
    
    def document_exists(self, doc_id: str) -> bool:
        """Check if a document exists in Redis.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if document exists, False otherwise
        """
        try:
            client = self._backend.client
            doc_key = self._get_doc_key(doc_id)
            
            return client.exists(doc_key) > 0
        
        except (redis.RedisError, ConnectionError) as e:
            self._logger.error(f"Error checking document existence {doc_id}: {e}")
            return False


class RedisChunkStore(ChunkStore, Configurable):
    """Redis implementation of the ChunkStore interface."""
    
    def __init__(self, backend: Optional[RedisStorageBackend] = None):
        """Initialize the Redis chunk store.
        
        Args:
            backend: Optional Redis storage backend to use
        """
        self._backend = backend
        self._logger = logging.getLogger(__name__)
        self._chunk_key_prefix = "chunk:"
        self._doc_chunks_key_prefix = "doc_chunks:"
        self._config = {}
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the Redis chunk store with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self._config = config
        
        # Set key prefixes
        self._chunk_key_prefix = config.get('chunk_key_prefix', self._chunk_key_prefix)
        self._doc_chunks_key_prefix = config.get('doc_chunks_key_prefix', self._doc_chunks_key_prefix)
        
        # Create backend if not provided
        if not self._backend:
            self._backend = RedisStorageBackend()
            self._backend.initialize(config.get('redis', {}))
    
    def _get_chunk_key(self, chunk_id: str) -> str:
        """Get the Redis key for a chunk.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Redis key for the chunk
        """
        return f"{self._chunk_key_prefix}{chunk_id}"
    
    def _get_doc_chunks_key(self, doc_id: str) -> str:
        """Get the Redis key for a document's chunks.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Redis key for the document's chunks
        """
        return f"{self._doc_chunks_key_prefix}{doc_id}"
    
    def store_chunk(self, chunk: DocumentChunk) -> bool:
        """Store a chunk in Redis.
        
        Args:
            chunk: Chunk to store
            
        Returns:
            True if storage successful, False otherwise
        """
        try:
            client = self._backend.client
            chunk_key = self._get_chunk_key(chunk.chunk_id)
            doc_chunks_key = self._get_doc_chunks_key(chunk.doc_id)
            
            # Serialize chunk
            chunk_data = {
                'chunk_id': chunk.chunk_id,
                'doc_id': chunk.doc_id,
                'content': chunk.content,
                'metadata': chunk.metadata
            }
            
            # Store chunk
            client.set(chunk_key, json.dumps(chunk_data).encode('utf-8'))
            
            # Add to document's chunks set
            client.sadd(doc_chunks_key, chunk.chunk_id)
            
            self._logger.debug(f"Stored chunk: {chunk.chunk_id} for document: {chunk.doc_id}")
            return True
        
        except (redis.RedisError, ConnectionError) as e:
            self._logger.error(f"Error storing chunk {chunk.chunk_id}: {e}")
            return False
    
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get a chunk from Redis.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Chunk if found, None otherwise
        """
        try:
            client = self._backend.client
            chunk_key = self._get_chunk_key(chunk_id)
            
            # Get chunk data
            chunk_data_bytes = client.get(chunk_key)
            if not chunk_data_bytes:
                return None
            
            # Deserialize chunk
            chunk_data = json.loads(chunk_data_bytes.decode('utf-8'))
            
            return DocumentChunk(
                chunk_id=chunk_data['chunk_id'],
                doc_id=chunk_data['doc_id'],
                content=chunk_data['content'],
                metadata=chunk_data['metadata']
            )
        
        except (redis.RedisError, ConnectionError, json.JSONDecodeError) as e:
            self._logger.error(f"Error getting chunk {chunk_id}: {e}")
            return None
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk from Redis.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            client = self._backend.client
            chunk_key = self._get_chunk_key(chunk_id)
            
            # Get chunk to find document ID
            chunk = self.get_chunk(chunk_id)
            if not chunk:
                return False
            
            # Delete chunk
            client.delete(chunk_key)
            
            # Remove from document's chunks set
            doc_chunks_key = self._get_doc_chunks_key(chunk.doc_id)
            client.srem(doc_chunks_key, chunk_id)
            
            self._logger.debug(f"Deleted chunk: {chunk_id}")
            return True
        
        except (redis.RedisError, ConnectionError) as e:
            self._logger.error(f"Error deleting chunk {chunk_id}: {e}")
            return False
    
    def get_document_chunks(self, doc_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document from Redis.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of chunks for the document
        """
        try:
            client = self._backend.client
            doc_chunks_key = self._get_doc_chunks_key(doc_id)
            
            # Get chunk IDs for document
            chunk_ids = client.smembers(doc_chunks_key)
            
            # Get chunks
            chunks = []
            for chunk_id in chunk_ids:
                chunk = self.get_chunk(chunk_id.decode('utf-8'))
                if chunk:
                    chunks.append(chunk)
            
            return chunks
        
        except (redis.RedisError, ConnectionError) as e:
            self._logger.error(f"Error getting chunks for document {doc_id}: {e}")
            return []
    
    def chunk_exists(self, chunk_id: str) -> bool:
        """Check if a chunk exists in Redis.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            True if chunk exists, False otherwise
        """
        try:
            client = self._backend.client
            chunk_key = self._get_chunk_key(chunk_id)
            
            return client.exists(chunk_key) > 0
        
        except (redis.RedisError, ConnectionError) as e:
            self._logger.error(f"Error checking chunk existence {chunk_id}: {e}")
            return False


class RedisEmbeddingStore(EmbeddingStore, Configurable):
    """Redis implementation of the EmbeddingStore interface."""
    
    def __init__(self, backend: Optional[RedisStorageBackend] = None):
        """Initialize the Redis embedding store.
        
        Args:
            backend: Optional Redis storage backend to use
        """
        self._backend = backend
        self._logger = logging.getLogger(__name__)
        self._embedding_key_prefix = "embedding:"
        self._chunk_embedding_key_prefix = "chunk_embedding:"
        self._embedding_index_key = "embedding_index"
        self._config = {}
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the Redis embedding store with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self._config = config
        
        # Set key prefixes
        self._embedding_key_prefix = config.get('embedding_key_prefix', self._embedding_key_prefix)
        self._chunk_embedding_key_prefix = config.get('chunk_embedding_key_prefix', self._chunk_embedding_key_prefix)
        self._embedding_index_key = config.get('embedding_index_key', self._embedding_index_key)
        
        # Create backend if not provided
        if not self._backend:
            self._backend = RedisStorageBackend()
            self._backend.initialize(config.get('redis', {}))
    
    def _get_embedding_key(self, embedding_id: str) -> str:
        """Get the Redis key for an embedding.
        
        Args:
            embedding_id: Embedding ID
            
        Returns:
            Redis key for the embedding
        """
        return f"{self._embedding_key_prefix}{embedding_id}"
    
    def _get_chunk_embedding_key(self, chunk_id: str) -> str:
        """Get the Redis key for a chunk's embedding.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Redis key for the chunk's embedding
        """
        return f"{self._chunk_embedding_key_prefix}{chunk_id}"
    
    def store_embedding(self, embedded_chunk: EmbeddedChunk) -> bool:
        """Store an embedded chunk in Redis.
        
        Args:
            embedded_chunk: Embedded chunk to store
            
        Returns:
            True if storage successful, False otherwise
        """
        try:
            client = self._backend.client
            embedding_key = self._get_embedding_key(embedded_chunk.embedding_id)
            chunk_embedding_key = self._get_chunk_embedding_key(embedded_chunk.chunk_id)
            
            # Serialize embedding data
            embedding_data = {
                'embedding_id': embedded_chunk.embedding_id,
                'chunk_id': embedded_chunk.chunk_id,
                'embedding': embedded_chunk.embedding.tolist() if isinstance(embedded_chunk.embedding, np.ndarray) else embedded_chunk.embedding,
                'model_name': embedded_chunk.model_name,
                'metadata': embedded_chunk.metadata
            }
            
            # Store embedding
            client.set(embedding_key, json.dumps(embedding_data).encode('utf-8'))
            
            # Store mapping from chunk ID to embedding ID
            client.set(chunk_embedding_key, embedded_chunk.embedding_id)
            
            # Add to embedding index
            client.sadd(self._embedding_index_key, embedded_chunk.embedding_id)
            
            self._logger.debug(f"Stored embedding: {embedded_chunk.embedding_id} for chunk: {embedded_chunk.chunk_id}")
            return True
        
        except (redis.RedisError, ConnectionError) as e:
            self._logger.error(f"Error storing embedding {embedded_chunk.embedding_id}: {e}")
            return False
    
    def get_embedding(self, embedding_id: str) -> Optional[EmbeddedChunk]:
        """Get an embedded chunk from Redis.
        
        Args:
            embedding_id: Embedding ID
            
        Returns:
            Embedded chunk if found, None otherwise
        """
        try:
            client = self._backend.client
            embedding_key = self._get_embedding_key(embedding_id)
            
            # Get embedding data
            embedding_data_bytes = client.get(embedding_key)
            if not embedding_data_bytes:
                return None
            
            # Deserialize embedding
            embedding_data = json.loads(embedding_data_bytes.decode('utf-8'))
            
            # Convert embedding to numpy array
            embedding = np.array(embedding_data['embedding'])
            
            return EmbeddedChunk(
                embedding_id=embedding_data['embedding_id'],
                chunk_id=embedding_data['chunk_id'],
                embedding=embedding,
                model_name=embedding_data['model_name'],
                metadata=embedding_data['metadata']
            )
        
        except (redis.RedisError, ConnectionError, json.JSONDecodeError) as e:
            self._logger.error(f"Error getting embedding {embedding_id}: {e}")
            return None
    
    def get_chunk_embedding(self, chunk_id: str) -> Optional[EmbeddedChunk]:
        """Get the embedding for a chunk from Redis.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Embedded chunk if found, None otherwise
        """
        try:
            client = self._backend.client
            chunk_embedding_key = self._get_chunk_embedding_key(chunk_id)
            
            # Get embedding ID for chunk
            embedding_id = client.get(chunk_embedding_key)
            if not embedding_id:
                return None
            
            # Get embedding
            return self.get_embedding(embedding_id.decode('utf-8'))
        
        except (redis.RedisError, ConnectionError) as e:
            self._logger.error(f"Error getting embedding for chunk {chunk_id}: {e}")
            return None
    
    def delete_embedding(self, embedding_id: str) -> bool:
        """Delete an embedding from Redis.
        
        Args:
            embedding_id: Embedding ID
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            client = self._backend.client
            embedding_key = self._get_embedding_key(embedding_id)
            
            # Get embedding to find chunk ID
            embedded_chunk = self.get_embedding(embedding_id)
            if not embedded_chunk:
                return False
            
            # Delete embedding
            client.delete(embedding_key)
            
            # Delete chunk embedding mapping
            chunk_embedding_key = self._get_chunk_embedding_key(embedded_chunk.chunk_id)
            client.delete(chunk_embedding_key)
            
            # Remove from embedding index
            client.srem(self._embedding_index_key, embedding_id)
            
            self._logger.debug(f"Deleted embedding: {embedding_id}")
            return True
        
        except (redis.RedisError, ConnectionError) as e:
            self._logger.error(f"Error deleting embedding {embedding_id}: {e}")
            return False
    
    def embedding_exists(self, embedding_id: str) -> bool:
        """Check if an embedding exists in Redis.
        
        Args:
            embedding_id: Embedding ID
            
        Returns:
            True if embedding exists, False otherwise
        """
        try:
            client = self._backend.client
            embedding_key = self._get_embedding_key(embedding_id)
            
            return client.exists(embedding_key) > 0
        
        except (redis.RedisError, ConnectionError) as e:
            self._logger.error(f"Error checking embedding existence {embedding_id}: {e}")
            return False
    
    def chunk_has_embedding(self, chunk_id: str) -> bool:
        """Check if a chunk has an embedding in Redis.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            True if chunk has an embedding, False otherwise
        """
        try:
            client = self._backend.client
            chunk_embedding_key = self._get_chunk_embedding_key(chunk_id)
            
            return client.exists(chunk_embedding_key) > 0
        
        except (redis.RedisError, ConnectionError) as e:
            self._logger.error(f"Error checking chunk embedding existence {chunk_id}: {e}")
            return False
    
    def search_embeddings(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar embeddings in Redis.
        
        This is a basic implementation that loads all embeddings and performs
        a brute-force search. For production use, consider using Redis Stack
        with vector search capabilities.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of (embedding_id, similarity_score) tuples
        """
        try:
            client = self._backend.client
            
            # Get all embedding IDs
            embedding_ids = client.smembers(self._embedding_index_key)
            embedding_ids = [eid.decode('utf-8') for eid in embedding_ids]
            
            # Get all embeddings
            embeddings = []
            for embedding_id in embedding_ids:
                embedded_chunk = self.get_embedding(embedding_id)
                if embedded_chunk:
                    embeddings.append((embedding_id, embedded_chunk.embedding))
            
            # Calculate similarities
            similarities = []
            for embedding_id, embedding in embeddings:
                similarity = self._cosine_similarity(query_embedding, embedding)
                similarities.append((embedding_id, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k results
            return similarities[:top_k]
        
        except (redis.RedisError, ConnectionError) as e:
            self._logger.error(f"Error searching embeddings: {e}")
            return []
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
