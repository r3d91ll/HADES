"""
Storage interface protocols for HADES-PathRAG.

This module defines the Protocol interfaces for repository operations including
document, graph, and vector storage and retrieval. These protocols define the
expected interface that concrete storage implementations should follow.
"""

from typing import Dict, List, Any, Optional, Union, Protocol, Tuple, runtime_checkable
from abc import ABC, abstractmethod

from ..common import NodeData, EdgeData, EmbeddingVector, NodeID, EdgeID


@runtime_checkable
class DocumentRepository(Protocol):
    """Protocol interface for document operations in a repository."""
    
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
    """Protocol interface for graph operations in a repository."""
    
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
    """Protocol interface for vector operations in a repository."""
    
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
    Unified repository protocol combining document, graph, and vector operations.
    This protocol represents the complete functionality required for the HADES-PathRAG system.
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


class AbstractUnifiedRepository(ABC):
    """
    Abstract base class for unified repository operations.
    
    This abstract interface consolidates document, graph, and vector operations into a single
    cohesive API that can be implemented by different storage backends. This is for
    implementations that prefer ABC over Protocol.
    """
    
    @abstractmethod
    async def store_node(self, node_data: NodeData) -> bool:
        """
        Store a node in the repository.
        
        Args:
            node_data: Data representing the node to store
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def store_edge(self, edge_data: EdgeData) -> bool:
        """
        Store an edge in the repository.
        
        Args:
            edge_data: Data representing the edge to store
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def store_embedding(self, node_id: NodeID, embedding: EmbeddingVector) -> bool:
        """
        Store an embedding vector for a node.
        
        Args:
            node_id: ID of the node to store the embedding for
            embedding: The embedding vector
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_node(self, node_id: NodeID) -> Optional[NodeData]:
        """
        Retrieve a node by ID.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            The node data if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_edges(self, node_id: NodeID, edge_types: Optional[List[str]] = None) -> List[EdgeData]:
        """
        Get edges connected to a node.
        
        Args:
            node_id: ID of the node to get edges for
            edge_types: Optional list of edge types to filter by
            
        Returns:
            List of edges connected to the node
        """
        pass
    
    @abstractmethod
    async def search_similar(
        self, 
        query_vector: EmbeddingVector, 
        limit: int = 10, 
        node_types: Optional[List[str]] = None
    ) -> List[Tuple[NodeID, float]]:
        """
        Search for nodes with similar embeddings.
        
        Args:
            query_vector: The query embedding vector
            limit: Maximum number of results to return
            node_types: Optional list of node types to filter by
            
        Returns:
            List of (node_id, similarity_score) tuples
        """
        pass
    
    @abstractmethod
    async def initialize(self, recreate: bool = False) -> bool:
        """
        Initialize the repository, creating necessary collections and indices.
        
        Args:
            recreate: Whether to recreate all collections (deleting existing data)
            
        Returns:
            True if initialization was successful
        """
        pass
    
    @abstractmethod
    async def get_path(
        self, 
        start_id: NodeID, 
        end_id: NodeID, 
        max_depth: int = 3, 
        edge_types: Optional[List[str]] = None
    ) -> List[List[Union[NodeData, EdgeData]]]:
        """
        Find paths between two nodes in the graph.
        
        Args:
            start_id: ID of the starting node
            end_id: ID of the ending node
            max_depth: Maximum path depth to search
            edge_types: Optional list of edge types to traverse
            
        Returns:
            List of paths, where each path is a list alternating between nodes and edges
        """
        pass