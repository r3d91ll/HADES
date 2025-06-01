"""Repository interface type definitions for storage systems.

This module defines the type annotations for repository operations including
document, graph, and vector storage and retrieval interfaces. These types
support the concrete implementations of repository classes.
"""

from typing import Dict, List, Any, Optional, Union, Protocol, Tuple, TypeVar, runtime_checkable
from datetime import datetime
from uuid import UUID

from src.types.common import NodeData, EdgeData, EmbeddingVector, NodeID, EdgeID


# Result type for operations that may fail
OperationResult = Tuple[bool, Optional[str]]

# Define type alias for connection config to avoid circular imports
ConnectionConfigType = Dict[str, Any]

class DatabaseCredentials(Dict[str, str]):
    """Type for database credentials."""
    pass


@runtime_checkable
class ConnectionInterface(Protocol):
    """Interface for database connections."""
    
    def connect(self) -> bool:
        """
        Establish connection to the database.
        
        Returns:
            True if connection was successful, False otherwise
        """
        ...
    
    def disconnect(self) -> bool:
        """
        Close connection to the database.
        
        Returns:
            True if disconnection was successful, False otherwise
        """
        ...
    
    def is_connected(self) -> bool:
        """
        Check if connection to the database is active.
        
        Returns:
            True if connected, False otherwise
        """
        ...
    
    def get_database(self) -> Any:
        """
        Get the database instance.
        
        Returns:
            The database instance
        """
        ...


@runtime_checkable
class RepositoryInterface(Protocol):
    """Base interface for repository operations."""
    
    def initialize(self) -> bool:
        """
        Initialize the repository, creating collections if needed.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        ...
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health status of the repository.
        
        Returns:
            Dictionary with health status information
        """
        ...
    
    def clear(self) -> bool:
        """
        Clear all data in the repository.
        
        Returns:
            True if clearing was successful, False otherwise
        """
        ...
    
    def backup(self, backup_path: str) -> bool:
        """
        Create a backup of the repository.
        
        Args:
            backup_path: Path where the backup should be stored
            
        Returns:
            True if backup was successful, False otherwise
        """
        ...
    
    def restore(self, backup_path: str) -> bool:
        """
        Restore the repository from a backup.
        
        Args:
            backup_path: Path to the backup
            
        Returns:
            True if restoration was successful, False otherwise
        """
        ...


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
    
    def delete_document(self, document_id: NodeID) -> bool:
        """
        Delete a document by its ID.
        
        Args:
            document_id: The ID of the document to delete
            
        Returns:
            True if the deletion was successful, False otherwise
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
    
    def get_documents_by_type(self, doc_type: str, 
                            limit: int = 100, 
                            offset: int = 0) -> List[NodeData]:
        """
        Get documents by their type.
        
        Args:
            doc_type: The type of documents to retrieve
            limit: Maximum number of results to return
            offset: Number of results to skip
            
        Returns:
            List of documents of the specified type
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
    
    def get_edge(self, edge_id: EdgeID) -> Optional[EdgeData]:
        """
        Get an edge by its ID.
        
        Args:
            edge_id: The ID of the edge to retrieve
            
        Returns:
            The edge data if found, None otherwise
        """
        ...
    
    def update_edge(self, edge_id: EdgeID, updates: Dict[str, Any]) -> bool:
        """
        Update an edge by its ID.
        
        Args:
            edge_id: The ID of the edge to update
            updates: The fields to update and their new values
            
        Returns:
            True if the update was successful, False otherwise
        """
        ...
    
    def delete_edge(self, edge_id: EdgeID) -> bool:
        """
        Delete an edge by its ID.
        
        Args:
            edge_id: The ID of the edge to delete
            
        Returns:
            True if the deletion was successful, False otherwise
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
    
    def get_connected_components(self, 
                               edge_types: Optional[List[str]] = None) -> List[List[NodeID]]:
        """
        Get connected components in the graph.
        
        Args:
            edge_types: Optional list of edge types to consider
            
        Returns:
            List of node ID lists, each representing a connected component
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
    
    def delete_embedding(self, node_id: NodeID) -> bool:
        """
        Delete the embedding for a node.
        
        Args:
            node_id: The ID of the node
            
        Returns:
            True if the deletion was successful, False otherwise
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
    
    def get_nearest_neighbors(self, node_id: NodeID, 
                             limit: int = 10) -> List[Tuple[NodeData, float]]:
        """
        Get nearest neighbors of a node based on embedding similarity.
        
        Args:
            node_id: The ID of the node
            limit: Maximum number of results to return
            
        Returns:
            List of neighboring nodes with similarity scores
        """
        ...
