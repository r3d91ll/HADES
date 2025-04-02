"""
Storage interfaces for the HADES modular pipeline architecture.

This module defines the interfaces for storage backends that can be used
to store and retrieve documents, chunks, and embeddings.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from hades.pipeline.interfaces import Component, Configurable, Pluggable, Serializable


class StorageBackend(Component, Configurable, Pluggable, Serializable, ABC):
    """Base interface for all storage backends."""
    
    @property
    def component_type(self) -> str:
        """Return the type of this component."""
        return "storage_backend"
    
    @classmethod
    def get_plugin_type(cls) -> str:
        """Return the plugin type for this component."""
        return "storage_backend"
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the storage backend.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the storage backend."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to the storage backend.
        
        Returns:
            True if connected, False otherwise
        """
        pass


class DocumentStore(StorageBackend, ABC):
    """Interface for storing and retrieving documents."""
    
    @abstractmethod
    def store_document(self, document_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store a document in the storage backend.
        
        Args:
            document_id: Unique identifier for the document
            content: Document content
            metadata: Optional metadata for the document
            
        Returns:
            True if storage successful, False otherwise
        """
        pass
    
    @abstractmethod
    def retrieve_document(self, document_id: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Retrieve a document from the storage backend.
        
        Args:
            document_id: Unique identifier for the document
            
        Returns:
            Tuple of (content, metadata), both None if document not found
        """
        pass
    
    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        """Delete a document from the storage backend.
        
        Args:
            document_id: Unique identifier for the document
            
        Returns:
            True if deletion successful, False otherwise
        """
        pass
    
    @abstractmethod
    def list_documents(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[str]:
        """List document IDs in the storage backend.
        
        Args:
            filter_criteria: Optional criteria to filter documents
            
        Returns:
            List of document IDs
        """
        pass


class ChunkStore(StorageBackend, ABC):
    """Interface for storing and retrieving document chunks."""
    
    @abstractmethod
    def store_chunk(self, document_id: str, chunk_id: str, content: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store a document chunk in the storage backend.
        
        Args:
            document_id: ID of the parent document
            chunk_id: Unique identifier for the chunk
            content: Chunk content
            metadata: Optional metadata for the chunk
            
        Returns:
            True if storage successful, False otherwise
        """
        pass
    
    @abstractmethod
    def retrieve_chunk(self, chunk_id: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Retrieve a chunk from the storage backend.
        
        Args:
            chunk_id: Unique identifier for the chunk
            
        Returns:
            Tuple of (content, metadata), both None if chunk not found
        """
        pass
    
    @abstractmethod
    def list_chunks(self, document_id: Optional[str] = None, 
                   filter_criteria: Optional[Dict[str, Any]] = None) -> List[str]:
        """List chunk IDs in the storage backend.
        
        Args:
            document_id: Optional parent document ID to filter by
            filter_criteria: Optional criteria to filter chunks
            
        Returns:
            List of chunk IDs
        """
        pass


class EmbeddingStore(StorageBackend, ABC):
    """Interface for storing and retrieving embeddings."""
    
    @abstractmethod
    def store_embedding(self, chunk_id: str, embedding: Union[List[float], np.ndarray], 
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store an embedding in the storage backend.
        
        Args:
            chunk_id: ID of the associated chunk
            embedding: Vector embedding
            metadata: Optional metadata for the embedding
            
        Returns:
            True if storage successful, False otherwise
        """
        pass
    
    @abstractmethod
    def retrieve_embedding(self, chunk_id: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """Retrieve an embedding from the storage backend.
        
        Args:
            chunk_id: ID of the associated chunk
            
        Returns:
            Tuple of (embedding, metadata), both None if not found
        """
        pass
    
    @abstractmethod
    def search_similar(self, query_embedding: Union[List[float], np.ndarray], 
                      top_k: int = 5, filter_criteria: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """Search for similar embeddings.
        
        Args:
            query_embedding: Query vector embedding
            top_k: Number of results to return
            filter_criteria: Optional criteria to filter results
            
        Returns:
            List of (chunk_id, similarity_score) tuples
        """
        pass


class UnifiedStorageBackend(DocumentStore, ChunkStore, EmbeddingStore, ABC):
    """Interface for a unified storage backend that supports all storage operations."""
    
    @abstractmethod
    def clear_all(self) -> bool:
        """Clear all data from the storage backend.
        
        Returns:
            True if operation successful, False otherwise
        """
        pass
    
    @abstractmethod
    def export_to_disk(self, export_path: str) -> bool:
        """Export all data to disk.
        
        Args:
            export_path: Path to export data to
            
        Returns:
            True if export successful, False otherwise
        """
        pass
    
    @abstractmethod
    def import_from_disk(self, import_path: str) -> bool:
        """Import data from disk.
        
        Args:
            import_path: Path to import data from
            
        Returns:
            True if import successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the storage backend.
        
        Returns:
            Dictionary of statistics
        """
        pass