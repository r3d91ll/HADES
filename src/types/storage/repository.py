"""Repository interface type definitions for storage systems.

This module defines the type annotations for repository operations including
document, graph, and vector storage and retrieval interfaces. These types
support the concrete implementations of repository classes.
"""

from typing import Dict, List, Any, Optional, Union, Protocol, Tuple, TypeVar, runtime_checkable, TypedDict, Awaitable
from datetime import datetime
from uuid import UUID

from src.types.common import NodeData, EdgeData, EmbeddingVector, NodeID, EdgeID
from src.types.storage.query import QueryOptions, QueryResult, VectorQueryOptions, HybridQueryOptions, VectorSearchResult, HybridSearchResult


# Result type for operations that may fail
OperationResult = Tuple[bool, Optional[str]]


# Bulk operation result types
class BaseBulkOperationResult(TypedDict):
    """Base result for bulk operations."""
    success: bool
    """Whether the bulk operation was successful overall."""
    
    total: int
    """Total number of operations attempted."""
    
    errors: List[Dict[str, Any]]
    """List of errors encountered during the operation."""


class CustomBulkOperationResult(BaseBulkOperationResult):
    """Extended result for custom bulk operations."""
    created: List[str]
    """List of IDs that were successfully created."""
    
    skipped: List[str]
    """List of IDs that were skipped during the operation."""


class DetailedBulkOperationResult(TypedDict, total=False):
    """Detailed bulk operation result with additional fields."""
    document_id: str
    """ID of the document the operation was performed on."""
    
    status: str
    """Status of the operation (success, error, etc.)."""
    
    message: str
    """Descriptive message about the operation result."""
    
    chunks_stored: int
    """Number of chunks stored during the operation."""
    
    embeddings_stored: int
    """Number of embeddings stored during the operation."""
    
    successful: List[str]
    """List of successful operation IDs."""
    
    failed: List[str]
    """List of failed operation IDs."""
    
    success_count: int
    """Count of successful operations."""
    
    error_count: int
    """Count of failed operations."""
    
    operation: str
    """Type of operation performed."""
    
    timestamp: str
    """Timestamp when the operation was performed."""
    
    errors: Dict[str, str]
    """Dictionary of error messages by ID."""

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


@runtime_checkable
class UnifiedRepository(Protocol):
    """Interface combining document, graph, and vector repository operations.
    
    This unified interface represents the complete functionality required for the HADES system,
    combining document, graph, and vector operations with additional collection management
    and advanced query capabilities.
    """
    
    # From DocumentRepository
    def store_document(self, document: NodeData) -> Union[NodeID, Awaitable[NodeID]]:
        """Store a document and return its ID."""
        ...
        
    def get_document(self, document_id: NodeID) -> Union[Optional[NodeData], Awaitable[Optional[NodeData]]]:
        """Get a document by its ID."""
        ...
        
    def update_document(self, document_id: NodeID, updates: Dict[str, Any]) -> Union[bool, Awaitable[bool]]:
        """Update a document with the provided updates."""
        ...
        
    def delete_document(self, document_id: NodeID) -> Union[bool, Awaitable[bool]]:
        """Delete a document by its ID."""
        ...
        
    def search_documents(self, query: str, filters: Optional[Dict[str, Any]] = None, limit: int = 10) -> Union[List[NodeData], Awaitable[List[NodeData]]]:
        """Search for documents matching the query."""
        ...
        
    def get_documents_by_type(self, doc_type: str, limit: int = 100, offset: int = 0) -> Union[List[NodeData], Awaitable[List[NodeData]]]:
        """Get documents of a specific type."""
        ...
        
    # From GraphRepository
    def store_node(self, node: NodeData) -> Union[NodeID, Awaitable[NodeID]]:
        """Store a node in the graph."""
        ...
        
    def get_node(self, node_id: NodeID) -> Union[Optional[NodeData], Awaitable[Optional[NodeData]]]:
        """Get a node from the graph by its ID."""
        ...
        
    def store_edge(self, edge: EdgeData) -> Union[EdgeID, Awaitable[EdgeID]]:
        """Store an edge in the graph."""
        ...
        
    def get_path(self, start_id: NodeID, end_id: NodeID, max_depth: int = 3) -> Union[List[EdgeData], Awaitable[List[EdgeData]]]:
        """Get a path between two nodes."""
        ...
        
    def get_connected_nodes(self, node_id: NodeID, edge_type: Optional[str] = None) -> Union[List[NodeData], Awaitable[List[NodeData]]]:
        """Get all nodes connected to the specified node."""
        ...
        
    # From VectorRepository
    def store_embedding(self, node_id: NodeID, embedding: EmbeddingVector, options: Optional[Dict[str, Any]] = None) -> Union[bool, Awaitable[bool]]:
        """Store an embedding vector for a node."""
        ...
        
    def get_embedding(self, node_id: NodeID) -> Union[Optional[EmbeddingVector], Awaitable[Optional[EmbeddingVector]]]:
        """Get the embedding vector for a node."""
        ...
        
    def delete_embedding(self, node_id: NodeID) -> Union[bool, Awaitable[bool]]:
        """Delete the embedding for a node."""
        ...
        
    def vector_search(self, vector_options: VectorQueryOptions) -> Union[VectorSearchResult, Awaitable[VectorSearchResult]]:
        """Search for nodes with similar embeddings using the provided options."""
        ...
        
    def hybrid_search(self, hybrid_options: HybridQueryOptions) -> Union[HybridSearchResult, Awaitable[HybridSearchResult]]:
        """Perform a hybrid search using both text and vector similarity."""
        ...
    
    # Additional custom methods
    def create_similarity_edges(self, embeddings: List[Tuple[NodeID, EmbeddingVector]], threshold: float = 0.8, limit: int = 5) -> Union[BaseBulkOperationResult, Awaitable[BaseBulkOperationResult]]:
        """Create similarity edges between nodes based on embedding similarity."""
        ...
        
    def query_documents(self, query_options: QueryOptions) -> Union[QueryResult, Awaitable[QueryResult]]:
        """Query documents based on the provided options."""
        ...
        
    def setup_collections(self) -> None:
        """Set up the necessary collections and indexes in the repository."""
        ...
    
    def collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collections in the repository."""
        ...
