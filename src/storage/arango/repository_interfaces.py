"""
Repository interfaces for unified storage access in HADES-PathRAG.

This module defines abstract interfaces that various storage implementations
can conform to, ensuring a consistent API across different backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple, Sequence

from src.types.common import NodeData, EdgeData, EmbeddingVector, NodeID, EdgeID


class UnifiedRepository(ABC):
    """
    Abstract interface for unified repository operations.
    
    This interface consolidates document, graph, and vector operations into a single
    cohesive API that can be implemented by different storage backends.
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
