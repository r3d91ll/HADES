"""
NetworkX Storage Component

This module provides the NetworkX storage component that implements the
Storage protocol. It uses NetworkX graphs for storing and querying 
document relationships and embeddings.
"""

import logging
import networkx as nx
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import component contracts and protocols
from src.types.components.contracts import (
    ComponentType,
    ComponentMetadata,
    StorageInput,
    StorageOutput,
    StoredItem,
    QueryInput,
    QueryOutput,
    RetrievalResult,
    ProcessingStatus
)
from src.types.components.protocols import Storage


class NetworkXStorage(Storage):
    """
    NetworkX storage component implementing Storage protocol.
    
    This component provides graph-based storage using NetworkX for
    storing document embeddings and their relationships.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize NetworkX storage component.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Component metadata
        self._metadata = ComponentMetadata(
            component_type=ComponentType.STORAGE,
            component_name="networkx",
            component_version="1.0.0",
            config=self._config
        )
        
        # Configuration
        self._graph_type = self._config.get('graph_type', 'directed')
        self._similarity_threshold = self._config.get('similarity_threshold', 0.7)
        self._max_connections = self._config.get('max_connections', 10)
        
        # Initialize NetworkX graph
        if self._graph_type == 'directed':
            self._graph = nx.DiGraph()
        else:
            self._graph = nx.Graph()
            
        # Storage for embeddings and metadata
        self._embeddings: Dict[str, List[float]] = {}
        self._metadata_store: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info(f"Initialized NetworkX storage with {self._graph_type} graph")
    
    @property
    def name(self) -> str:
        """Component name for identification."""
        return "networkx"
    
    @property
    def version(self) -> str:
        """Component version string."""
        return "1.0.0"
    
    @property
    def component_type(self) -> ComponentType:
        """Type of component."""
        return ComponentType.STORAGE
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure component with parameters.
        
        Args:
            config: Configuration dictionary containing component parameters
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.validate_config(config):
            raise ValueError("Invalid configuration provided")
        
        self._config.update(config)
        self._metadata.config = self._config
        self._metadata.processed_at = datetime.utcnow()
        
        # Update configuration parameters
        if 'similarity_threshold' in config:
            self._similarity_threshold = config['similarity_threshold']
        
        if 'max_connections' in config:
            self._max_connections = config['max_connections']
        
        self.logger.info("Updated NetworkX storage configuration")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        if not isinstance(config, dict):
            return False
        
        # Validate similarity threshold
        if 'similarity_threshold' in config:
            threshold = config['similarity_threshold']
            if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
                return False
        
        # Validate max connections
        if 'max_connections' in config:
            max_conn = config['max_connections']
            if not isinstance(max_conn, int) or max_conn < 1:
                return False
        
        return True
    
    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for component configuration.
        
        Returns:
            JSON schema dictionary describing valid configuration
        """
        return {
            "type": "object",
            "properties": {
                "graph_type": {
                    "type": "string",
                    "enum": ["directed", "undirected"],
                    "default": "directed",
                    "description": "Type of NetworkX graph to use"
                },
                "similarity_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.7,
                    "description": "Similarity threshold for edge creation"
                },
                "max_connections": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 10,
                    "description": "Maximum connections per node"
                },
                "enable_clustering": {
                    "type": "boolean",
                    "default": False,
                    "description": "Enable graph clustering algorithms"
                }
            }
        }
    
    def health_check(self) -> bool:
        """
        Check if component is healthy and ready to process data.
        
        Returns:
            True if component is healthy, False otherwise
        """
        try:
            # Check if NetworkX is available
            import networkx as nx
            
            # Check if graph is accessible
            node_count = self._graph.number_of_nodes()
            edge_count = self._graph.number_of_edges()
            
            self.logger.debug(f"NetworkX storage health: {node_count} nodes, {edge_count} edges")
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get component performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        return {
            "component_name": self.name,
            "component_version": self.version,
            "graph_type": self._graph_type,
            "node_count": self._graph.number_of_nodes(),
            "edge_count": self._graph.number_of_edges(),
            "embeddings_stored": len(self._embeddings),
            "similarity_threshold": self._similarity_threshold,
            "max_connections": self._max_connections,
            "last_health_check": datetime.utcnow().isoformat()
        }
    
    def store(self, input_data: StorageInput) -> StorageOutput:
        """
        Store data according to the contract.
        
        Args:
            input_data: Input data conforming to StorageInput contract
            
        Returns:
            Output data conforming to StorageOutput contract
            
        Raises:
            StorageError: If storage operation fails
        """
        errors = []
        stored_items = []
        
        try:
            start_time = datetime.utcnow()
            
            # Process each enhanced embedding
            for embedding in input_data.enhanced_embeddings:
                try:
                    # Store embedding and metadata
                    item_id = embedding.chunk_id
                    self._embeddings[item_id] = embedding.enhanced_embedding
                    self._metadata_store[item_id] = {
                        "original_embedding": embedding.original_embedding,
                        "graph_features": embedding.graph_features,
                        "enhancement_score": embedding.enhancement_score,
                        "metadata": embedding.metadata
                    }
                    
                    # Add node to graph
                    self._graph.add_node(item_id, **embedding.metadata)
                    
                    # Create edges based on similarity
                    self._create_similarity_edges(item_id, embedding.enhanced_embedding)
                    
                    # Create stored item
                    stored_item = StoredItem(
                        item_id=item_id,
                        storage_location=f"networkx_graph_node:{item_id}",
                        storage_timestamp=datetime.utcnow(),
                        retrieval_metadata={
                            "node_degree": self._graph.degree(item_id),
                            "clustering_coefficient": nx.clustering(self._graph, item_id)
                        }
                    )
                    stored_items.append(stored_item)
                    
                except Exception as e:
                    error_msg = f"Failed to store embedding {embedding.chunk_id}: {str(e)}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update metadata
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processing_time=processing_time,
                processed_at=datetime.utcnow(),
                config=self._config,
                status=ProcessingStatus.SUCCESS if not errors else ProcessingStatus.ERROR
            )
            
            return StorageOutput(
                stored_items=stored_items,
                metadata=metadata,
                storage_stats={
                    "processing_time": processing_time,
                    "items_stored": len(stored_items),
                    "items_failed": len(errors),
                    "total_nodes": self._graph.number_of_nodes(),
                    "total_edges": self._graph.number_of_edges()
                },
                index_info={
                    "graph_type": self._graph_type,
                    "similarity_threshold": self._similarity_threshold,
                    "max_connections": self._max_connections
                },
                errors=errors,
                storage_size_bytes=self._estimate_storage_size()
            )
            
        except Exception as e:
            error_msg = f"NetworkX storage operation failed: {str(e)}"
            errors.append(error_msg)
            self.logger.error(error_msg)
            
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processed_at=datetime.utcnow(),
                config=self._config,
                status=ProcessingStatus.ERROR
            )
            
            return StorageOutput(
                stored_items=[],
                metadata=metadata,
                storage_stats={},
                index_info={},
                errors=errors
            )
    
    def query(self, query_data: QueryInput) -> QueryOutput:
        """
        Query stored data according to the contract.
        
        Args:
            query_data: Query data conforming to QueryInput contract
            
        Returns:
            Output data conforming to QueryOutput contract
            
        Raises:
            RetrievalError: If query fails
        """
        errors = []
        results = []
        
        try:
            start_time = datetime.utcnow()
            
            # If query embedding is provided, use similarity search
            if query_data.query_embedding:
                results = self._similarity_search(
                    query_data.query_embedding,
                    query_data.top_k
                )
            else:
                # Use text-based search if available
                results = self._text_search(
                    query_data.query,
                    query_data.top_k
                )
            
            # Apply filters if provided
            if query_data.filters:
                results = self._apply_filters(results, query_data.filters)
            
            # Sort by score
            results.sort(key=lambda x: x.score, reverse=True)
            
            # Limit results
            results = results[:query_data.top_k]
            
            # Calculate search time
            search_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update metadata
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processing_time=search_time,
                processed_at=datetime.utcnow(),
                config=self._config,
                status=ProcessingStatus.SUCCESS if not errors else ProcessingStatus.ERROR
            )
            
            return QueryOutput(
                results=results,
                metadata=metadata,
                search_stats={
                    "search_time": search_time,
                    "nodes_searched": self._graph.number_of_nodes(),
                    "similarity_threshold": self._similarity_threshold
                },
                query_info={
                    "query_type": "embedding" if query_data.query_embedding else "text",
                    "query_length": len(query_data.query),
                    "filters_applied": len(query_data.filters)
                },
                errors=errors,
                search_time=search_time
            )
            
        except Exception as e:
            error_msg = f"NetworkX query failed: {str(e)}"
            errors.append(error_msg)
            self.logger.error(error_msg)
            
            metadata = ComponentMetadata(
                component_type=self.component_type,
                component_name=self.name,
                component_version=self.version,
                processed_at=datetime.utcnow(),
                config=self._config,
                status=ProcessingStatus.ERROR
            )
            
            return QueryOutput(
                results=[],
                metadata=metadata,
                search_stats={},
                query_info={},
                errors=errors
            )
    
    def delete(self, item_ids: List[str]) -> bool:
        """
        Delete items by their IDs.
        
        Args:
            item_ids: List of item IDs to delete
            
        Returns:
            True if all items were deleted successfully
        """
        try:
            for item_id in item_ids:
                if item_id in self._embeddings:
                    del self._embeddings[item_id]
                
                if item_id in self._metadata_store:
                    del self._metadata_store[item_id]
                
                if self._graph.has_node(item_id):
                    self._graph.remove_node(item_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete items: {e}")
            return False
    
    def update(self, item_id: str, data: Dict[str, Any]) -> bool:
        """
        Update an existing item.
        
        Args:
            item_id: ID of item to update
            data: Updated data
            
        Returns:
            True if update was successful
        """
        try:
            if item_id not in self._embeddings:
                return False
            
            # Update metadata
            if "metadata" in data:
                self._metadata_store[item_id].update(data["metadata"])
            
            # Update graph node attributes
            if self._graph.has_node(item_id) and "attributes" in data:
                nx.set_node_attributes(self._graph, {item_id: data["attributes"]})
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update item {item_id}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary containing storage statistics
        """
        try:
            return {
                "total_nodes": self._graph.number_of_nodes(),
                "total_edges": self._graph.number_of_edges(),
                "total_embeddings": len(self._embeddings),
                "graph_density": nx.density(self._graph),
                "average_clustering": nx.average_clustering(self._graph),
                "connected_components": nx.number_connected_components(self._graph.to_undirected()),
                "storage_size_bytes": self._estimate_storage_size()
            }
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def get_capacity_info(self) -> Dict[str, Any]:
        """
        Get storage capacity information.
        
        Returns:
            Dictionary containing capacity information
        """
        return {
            "max_nodes": "unlimited",
            "max_edges": "unlimited",
            "current_nodes": self._graph.number_of_nodes(),
            "current_edges": self._graph.number_of_edges(),
            "memory_usage_estimate_mb": self._estimate_storage_size() / (1024 * 1024)
        }
    
    def supports_transactions(self) -> bool:
        """Check if storage supports transactions."""
        return False  # NetworkX doesn't support transactions
    
    def _create_similarity_edges(self, node_id: str, embedding: List[float]) -> None:
        """Create edges based on embedding similarity."""
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            node_embedding = np.array(embedding).reshape(1, -1)
            connections_made = 0
            
            for existing_id, existing_embedding in self._embeddings.items():
                if existing_id == node_id or connections_made >= self._max_connections:
                    continue
                
                existing_array = np.array(existing_embedding).reshape(1, -1)
                similarity = cosine_similarity(node_embedding, existing_array)[0][0]
                
                if similarity >= self._similarity_threshold:
                    self._graph.add_edge(node_id, existing_id, weight=similarity)
                    connections_made += 1
                    
        except Exception as e:
            self.logger.warning(f"Failed to create similarity edges: {e}")
    
    def _similarity_search(self, query_embedding: List[float], top_k: int) -> List[RetrievalResult]:
        """Perform similarity-based search."""
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            query_array = np.array(query_embedding).reshape(1, -1)
            results = []
            
            for item_id, embedding in self._embeddings.items():
                embedding_array = np.array(embedding).reshape(1, -1)
                similarity = cosine_similarity(query_array, embedding_array)[0][0]
                
                # Get content from metadata
                content = self._metadata_store.get(item_id, {}).get("content", "")
                
                result = RetrievalResult(
                    item_id=item_id,
                    content=content,
                    score=float(similarity),
                    metadata=self._metadata_store.get(item_id, {}),
                    chunk_metadata=self._metadata_store.get(item_id, {}).get("metadata", {}),
                    document_metadata=self._metadata_store.get(item_id, {}).get("document_metadata", {})
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            return []
    
    def _text_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Perform text-based search."""
        # Simple text matching implementation
        results = []
        query_lower = query.lower()
        
        for item_id, metadata in self._metadata_store.items():
            content = metadata.get("content", "")
            
            # Simple relevance score based on term matching
            score = 0.0
            if query_lower in content.lower():
                score = 1.0
            
            if score > 0:
                result = RetrievalResult(
                    item_id=item_id,
                    content=content,
                    score=score,
                    metadata=metadata,
                    chunk_metadata=metadata.get("metadata", {}),
                    document_metadata=metadata.get("document_metadata", {})
                )
                results.append(result)
        
        return results
    
    def _apply_filters(self, results: List[RetrievalResult], filters: Dict[str, Any]) -> List[RetrievalResult]:
        """Apply filters to search results."""
        filtered_results = []
        
        for result in results:
            include_result = True
            
            for filter_key, filter_value in filters.items():
                if filter_key in result.metadata:
                    if result.metadata[filter_key] != filter_value:
                        include_result = False
                        break
            
            if include_result:
                filtered_results.append(result)
        
        return filtered_results
    
    def _estimate_storage_size(self) -> int:
        """Estimate storage size in bytes."""
        try:
            import sys
            
            # Estimate size of embeddings
            embeddings_size = sum(sys.getsizeof(emb) for emb in self._embeddings.values())
            
            # Estimate size of metadata
            metadata_size = sum(sys.getsizeof(meta) for meta in self._metadata_store.values())
            
            # Estimate size of graph
            graph_size = sys.getsizeof(self._graph)
            
            return embeddings_size + metadata_size + graph_size
            
        except Exception:
            return 0