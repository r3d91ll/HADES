"""
Repository interfaces for HADES-PathRAG.

This module defines the interfaces for repository operations including
document, graph, and vector storage and retrieval. These interfaces are
designed to be implemented by concrete repository classes.
"""

from typing import Dict, List, Any, Optional, Union, Protocol, Tuple, TypeVar, runtime_checkable
from pathlib import Path
from datetime import datetime

from src.types.common import NodeData, EdgeData, EmbeddingVector, NodeID, EdgeID


@runtime_checkable
class DocumentRepository(Protocol):
    """Interface for document operations in a repository."""
    
    def store_document(self, document: NodeData) -> NodeID:
        """
        Store a document in the repository.
        
        Args:
            document: The document data to store
            
        Returns:
            The ID of the stored document
        """
        ...
    
    def get_document(self, document_id: NodeID) -> Optional[NodeData]:
        """
        Retrieve a document by its ID.
        
        Args:
            document_id: The ID of the document to retrieve
            
        Returns:
            The document data if found, None otherwise
        """
        ...
    
    def update_document(self, document_id: NodeID, updates: Dict[str, Any]) -> bool:
        """
        Update a document by its ID.
        
        Args:
            document_id: The ID of the document to update
            updates: The fields to update and their new values
            
        Returns:
            True if the update was successful, False otherwise
        """
        ...
    
    def search_documents(self, query: str, 
                         filters: Optional[Dict[str, Any]] = None,
                         limit: int = 10) -> List[NodeData]:
        """
        Search for documents using a text query.
        
        Args:
            query: The search query
            filters: Optional filters to apply to the search
            limit: Maximum number of results to return
            
        Returns:
            List of matching documents
        """
        ...


@runtime_checkable
class GraphRepository(Protocol):
    """Interface for graph operations in a repository."""
    
    def create_edge(self, edge: EdgeData) -> EdgeID:
        """
        Create an edge between nodes.
        
        Args:
            edge: The edge data
            
        Returns:
            The ID of the created edge
        """
        ...
    
    def get_edges(self, node_id: NodeID, 
                  edge_types: Optional[List[str]] = None,
                  direction: str = "outbound") -> List[Tuple[EdgeData, NodeData]]:
        """
        Get edges connected to a node.
        
        Args:
            node_id: The ID of the node
            edge_types: Optional list of edge types to filter by
            direction: Direction of edges ('outbound', 'inbound', or 'any')
            
        Returns:
            List of edges with their connected nodes
        """
        ...
    
    def traverse_graph(self, start_id: NodeID, edge_types: Optional[List[str]] = None,
                      max_depth: int = 3) -> Dict[str, Any]:
        """
        Traverse the graph starting from a node.
        
        Args:
            start_id: The ID of the starting node
            edge_types: Optional list of edge types to traverse
            max_depth: Maximum traversal depth
            
        Returns:
            Dictionary with traversal results
        """
        ...
    
    def shortest_path(self, from_id: NodeID, to_id: NodeID, 
                     edge_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Find the shortest path between two nodes.
        
        Args:
            from_id: The ID of the starting node
            to_id: The ID of the target node
            edge_types: Optional list of edge types to consider
            
        Returns:
            List of nodes and edges in the path
        """
        ...


@runtime_checkable
class VectorRepository(Protocol):
    """Interface for vector operations in a repository."""
    
    def store_embedding(self, node_id: NodeID, embedding: EmbeddingVector, 
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store an embedding for a node.
        
        Args:
            node_id: The ID of the node
            embedding: The vector embedding
            metadata: Optional metadata about the embedding
            
        Returns:
            True if the operation was successful, False otherwise
        """
        ...
    
    def get_embedding(self, node_id: NodeID) -> Optional[EmbeddingVector]:
        """
        Get the embedding for a node.
        
        Args:
            node_id: The ID of the node
            
        Returns:
            The vector embedding if found, None otherwise
        """
        ...
    
    def search_similar(self, embedding: EmbeddingVector, 
                      filters: Optional[Dict[str, Any]] = None,
                      limit: int = 10) -> List[Tuple[NodeData, float]]:
        """
        Search for nodes with similar embeddings.
        
        Args:
            embedding: The query embedding
            filters: Optional filters to apply to the search
            limit: Maximum number of results to return
            
        Returns:
            List of nodes with similarity scores
        """
        ...
    
    def hybrid_search(self, text_query: str, embedding: Optional[EmbeddingVector] = None,
                     filters: Optional[Dict[str, Any]] = None,
                     limit: int = 10) -> List[Tuple[NodeData, float]]:
        """
        Perform a hybrid search using both text and vector similarity.
        
        Args:
            text_query: The text search query
            embedding: Optional embedding for vector search
            filters: Optional filters to apply to the search
            limit: Maximum number of results to return
            
        Returns:
            List of nodes with combined relevance scores
        """
        ...


@runtime_checkable
class UnifiedRepository(DocumentRepository, GraphRepository, VectorRepository, Protocol):
    """
    Unified repository interface combining document, graph, and vector operations.
    This interface represents the complete functionality required for the HADES-PathRAG system.
    """
    
    def setup_collections(self) -> None:
        """
        Set up the necessary collections and indexes in the repository.
        """
        ...
    
    def collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collections in the repository.
        
        Returns:
            Dictionary with collection statistics
        """
        ...
