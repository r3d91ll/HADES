"""
ArangoDB storage adapter for supra-weight edges.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict
import logging
import numpy as np
from arango import ArangoClient  # type: ignore[import-not-found]
from arango.database import StandardDatabase  # type: ignore[import-not-found]
from arango.collection import StandardCollection  # type: ignore[import-not-found]
from arango.graph import Graph  # type: ignore[import-not-found]
from arango.job import AsyncJob, BatchJob  # type: ignore[import-not-found]
import json

from ..core.density_controller import EdgeCandidate

logger = logging.getLogger(__name__)


class ArangoSupraStorage:
    """
    Storage adapter for supra-weight edges in ArangoDB.
    
    Handles creation and management of graph collections with supra-weight edges.
    """
    
    def __init__(self, 
                 connection_url: str = "http://localhost:8529",
                 username: str = "root",
                 password: str = "",
                 database_name: str = "isne_bootstrap"):
        """
        Initialize ArangoDB connection.
        
        Args:
            connection_url: ArangoDB server URL
            username: Database username
            password: Database password
            database_name: Name of the database to use
        """
        self.connection_url = connection_url
        self.database_name = database_name
        
        # Connect to ArangoDB
        self.client = ArangoClient(hosts=connection_url)
        self.username = username
        self.password = password
        
        # Create or get database
        self.db = self._get_or_create_database()
        
        # Initialize collections
        self.node_collection: Optional[StandardCollection] = None
        self.edge_collection: Optional[StandardCollection] = None
        self.metadata_collection: Optional[StandardCollection] = None
        
        logger.info(f"Connected to ArangoDB at {connection_url}, database: {database_name}")
        
    def _get_or_create_database(self) -> StandardDatabase:
        """Get existing database or create new one."""
        # Connect to _system database first
        sys_db = self.client.db('_system', username=self.username, password=self.password)
        
        # Check if database exists
        if not sys_db.has_database(self.database_name):
            logger.info(f"Creating new database: {self.database_name}")
            sys_db.create_database(self.database_name)
            
        # Connect to the target database
        return self.client.db(self.database_name, username=self.username, password=self.password)
            
    def initialize_collections(self, 
                             node_collection_name: str = "nodes",
                             edge_collection_name: str = "supra_edges",
                             metadata_collection_name: str = "bootstrap_metadata") -> None:
        """
        Initialize the required collections.
        
        Args:
            node_collection_name: Name for node collection
            edge_collection_name: Name for edge collection
            metadata_collection_name: Name for metadata collection
        """
        # Create node collection
        if not self.db.has_collection(node_collection_name):
            self.node_collection = self.db.create_collection(node_collection_name)
            logger.info(f"Created node collection: {node_collection_name}")
        else:
            self.node_collection = self.db.collection(node_collection_name)
            
        # Create edge collection
        if not self.db.has_collection(edge_collection_name):
            self.edge_collection = self.db.create_collection(
                edge_collection_name,
                edge=True
            )
            logger.info(f"Created edge collection: {edge_collection_name}")
        else:
            self.edge_collection = self.db.collection(edge_collection_name)
            
        # Create metadata collection
        if not self.db.has_collection(metadata_collection_name):
            self.metadata_collection = self.db.create_collection(metadata_collection_name)
            logger.info(f"Created metadata collection: {metadata_collection_name}")
        else:
            self.metadata_collection = self.db.collection(metadata_collection_name)
            
        # Create indices
        self._create_indices()
        
        logger.info(f"Collections initialized - node: {self.node_collection}, edge: {self.edge_collection}, metadata: {self.metadata_collection}")
        
    def _create_indices(self) -> None:
        """Create necessary indices for efficient querying."""
        # Node indices
        if self.node_collection:
            self.node_collection.add_hash_index(fields=["node_type"], sparse=False)
            self.node_collection.add_hash_index(fields=["file_path"], sparse=True)
            self.node_collection.add_hash_index(fields=["chunk_index"], sparse=True)
        
        # Edge indices
        if self.edge_collection:
            self.edge_collection.add_hash_index(fields=["weight"], sparse=False)
            self.edge_collection.add_skiplist_index(fields=["weight"], sparse=False)
        
        # Metadata indices
        if self.metadata_collection:
            self.metadata_collection.add_hash_index(fields=["key"], sparse=False, unique=True)
        
        logger.info("Created database indices")
        
    def store_nodes(self, nodes: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Store nodes in the database.
        
        Args:
            nodes: List of node dictionaries
            
        Returns:
            Mapping of original node IDs to database IDs
        """
        logger.debug(f"store_nodes called with {len(nodes)} nodes, node_collection status: {self.node_collection}")
        
        # Ensure collections are available
        if self.node_collection is None:
            logger.warning("Node collection not initialized, attempting to get it")
            self.node_collection = self.db.collection("nodes")
            
        id_mapping = {}
        
        for node in nodes:
            # Extract node ID
            node_id = node.get('node_id') or node.get('_key')
            if not node_id:
                logger.warning("Node missing ID, skipping")
                continue
                
            # Prepare node document
            doc = {
                '_key': str(node_id),
                **{k: v for k, v in node.items() if not k.startswith('_')}
            }
            
            # Handle numpy arrays in embeddings
            if 'embedding' in doc and isinstance(doc['embedding'], np.ndarray):
                doc['embedding'] = doc['embedding'].tolist()
                
            # Store node
            try:
                if self.node_collection is not None:
                    result = self.node_collection.insert(doc)
                    logger.debug(f"Insert result for {node_id}: {result}")
                    if isinstance(result, dict) and '_key' in result:
                        id_mapping[node_id] = f"{self.node_collection.name}/{result['_key']}"
                    else:
                        logger.warning(f"Unexpected insert result for {node_id}: {result}")
                else:
                    logger.error(f"Node collection is None when trying to store node {node_id}")
            except Exception as e:
                if "unique constraint violated" in str(e):
                    # Update existing node
                    if self.node_collection is not None:
                        self.node_collection.update(doc)
                        id_mapping[node_id] = f"{self.node_collection.name}/{node_id}"
                        logger.debug(f"Updated existing node {node_id}")
                else:
                    logger.error(f"Error storing node {node_id}: {e}")
                    
        logger.info(f"Stored {len(id_mapping)} nodes")
        return id_mapping
        
    def store_edges(self, edges: List[EdgeCandidate], id_mapping: Dict[str, str]) -> None:
        """
        Store supra-weight edges in the database.
        
        Args:
            edges: List of edge candidates with supra-weights
            id_mapping: Mapping of node IDs to database IDs
        """
        logger.debug(f"store_edges called with {len(edges)} edges, id_mapping keys: {list(id_mapping.keys())}")
        stored_count = 0
        
        for edge in edges:
            # Get database IDs
            if edge.from_node is None or edge.to_node is None:
                logger.warning(f"Edge has None node: from={edge.from_node}, to={edge.to_node}")
                continue
                
            from_id = id_mapping.get(edge.from_node)
            to_id = id_mapping.get(edge.to_node)
            
            if not from_id or not to_id:
                logger.warning(f"Missing node IDs for edge {edge.from_node} -> {edge.to_node}, from_id: {from_id}, to_id: {to_id}")
                continue
                
            # Prepare edge document
            relationships_list: List[Dict[str, Any]] = []
            doc = {
                '_from': from_id,
                '_to': to_id,
                'weight': float(edge.weight),
                'weight_vector': edge.weight_vector.tolist() if isinstance(edge.weight_vector, np.ndarray) else edge.weight_vector,
                'relationships': relationships_list
            }
            
            # Add relationship details
            if edge.relationships:
                for rel in edge.relationships:
                    # relationships is a list of strings
                    rel_dict = {
                        'type': rel,
                        'strength': 1.0,  # Default strength
                        'confidence': 1.0  # Default confidence
                    }
                    relationships_list.append(rel_dict)
                
            # Store edge
            try:
                if self.edge_collection is not None:
                    self.edge_collection.insert(doc)
                    stored_count += 1
                else:
                    logger.error(f"Edge collection is None when trying to store edge")
            except Exception as e:
                logger.error(f"Error storing edge {edge.from_node} -> {edge.to_node}: {e}")
                
        logger.info(f"Stored {stored_count} edges")
        
    def store_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Store bootstrap metadata.
        
        Args:
            metadata: Dictionary of metadata to store
        """
        for key, value in metadata.items():
            doc = {
                '_key': key,
                'value': value
            }
            
            try:
                if self.metadata_collection:
                    self.metadata_collection.insert(doc)
            except Exception as e:
                if "unique constraint violated" in str(e):
                    # Update existing
                    if self.metadata_collection:
                        self.metadata_collection.update({'_key': key, 'value': value})
                else:
                    logger.error(f"Error storing metadata {key}: {e}")
                    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the stored graph."""
        # Node statistics
        node_count = 0
        if self.node_collection:
            count_result = self.node_collection.count()
            if isinstance(count_result, int):
                node_count = count_result
        
        node_types = {}
        
        # Use AQL for efficient aggregation
        if self.node_collection:
            query = """
            FOR node IN @@collection
                COLLECT type = node.node_type WITH COUNT INTO count
                RETURN {type: type, count: count}
            """
            cursor = self.db.aql.execute(
                query, 
                bind_vars={'@collection': self.node_collection.name}
            )
        
            try:
                for item in cursor:
                    if isinstance(item, dict) and 'type' in item and 'count' in item:
                        node_types[item['type']] = item['count']
            except Exception as e:
                logger.warning(f"Error getting node types: {e}")
            
        # Edge statistics
        edge_count = 0
        if self.edge_collection:
            edge_count_result = self.edge_collection.count()
            if isinstance(edge_count_result, int):
                edge_count = edge_count_result
        
        # Weight distribution
        weights = []
        if self.edge_collection:
            weight_query = """
            FOR edge IN @@collection
                RETURN edge.weight
            """
            cursor = self.db.aql.execute(
                weight_query,
                bind_vars={'@collection': self.edge_collection.name}
            )
            try:
                weights = [w for w in cursor if isinstance(w, (int, float))]
            except Exception as e:
                logger.warning(f"Error getting edge weights: {e}")
        
        weight_stats = {}
        if weights:
            weight_stats = {
                'min': float(np.min(weights)),
                'max': float(np.max(weights)),
                'mean': float(np.mean(weights)),
                'std': float(np.std(weights))
            }
            
        # Degree distribution
        degree_stats: Dict[str, Any] = {}
        if self.node_collection and self.edge_collection:
            degree_query = """
            LET out_degrees = (
                FOR v IN @@nodes
                    LET out_count = LENGTH(FOR e IN @@edges FILTER e._from == v._id RETURN 1)
                    RETURN out_count
            )
            LET in_degrees = (
                FOR v IN @@nodes
                    LET in_count = LENGTH(FOR e IN @@edges FILTER e._to == v._id RETURN 1)
                    RETURN in_count
            )
            RETURN {
                avg_out_degree: AVG(out_degrees),
                max_out_degree: MAX(out_degrees),
                avg_in_degree: AVG(in_degrees),
                max_in_degree: MAX(in_degrees)
            }
            """
            
            degree_cursor = self.db.aql.execute(
                degree_query,
                bind_vars={
                    '@nodes': self.node_collection.name,
                    '@edges': self.edge_collection.name
                }
            )
            
            try:
                degree_stats = next(degree_cursor, {})
            except Exception as e:
                logger.warning(f"Error getting degree statistics: {e}")
        
        return {
            'node_count': node_count,
            'node_types': node_types,
            'edge_count': edge_count,
            'weight_statistics': weight_stats,
            'degree_statistics': degree_stats,
            'density': edge_count / (node_count * (node_count - 1)) if node_count > 1 else 0
        }
        
    def create_graph(self, graph_name: str = "isne_graph") -> Graph:
        """
        Create a named graph for traversal operations.
        
        Args:
            graph_name: Name for the graph
            
        Returns:
            The created or existing graph
        """
        # Check if graph exists
        if self.db.has_graph(graph_name):
            logger.info(f"Graph {graph_name} already exists")
            return self.db.graph(graph_name)
            
        # Create graph with edge definition
        if self.node_collection is not None and self.edge_collection is not None:
            graph_result = self.db.create_graph(
                graph_name,
                edge_definitions=[{
                    'edge_collection': self.edge_collection.name,
                    'from_vertex_collections': [self.node_collection.name],
                    'to_vertex_collections': [self.node_collection.name]
                }]
            )
            
            if isinstance(graph_result, Graph):
                logger.info(f"Created graph: {graph_name}")
                return graph_result
            else:
                raise ValueError(f"Failed to create graph {graph_name}: unexpected return type")
        else:
            logger.error(f"Collections not initialized - node_collection: {self.node_collection} (type: {type(self.node_collection)}), edge_collection: {self.edge_collection} (type: {type(self.edge_collection)})")
            logger.error(f"Collection check failed: node_collection={bool(self.node_collection)}, edge_collection={bool(self.edge_collection)}")
            raise ValueError("Collections must be initialized before creating graph")
            
    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Get all nodes from the database."""
        if self.node_collection is None:
            logger.warning("Node collection not initialized")
            return []
        
        return list(self.node_collection.all())
    
    def get_all_edges(self) -> List[Dict[str, Any]]:
        """Get all edges from the database."""
        if self.edge_collection is None:
            logger.warning("Edge collection not initialized")
            return []
        
        return list(self.edge_collection.all())
    
    def clear_collections(self) -> None:
        """Clear all collections (use with caution)."""
        if self.node_collection:
            self.node_collection.truncate()
        if self.edge_collection:
            self.edge_collection.truncate()
        if self.metadata_collection:
            self.metadata_collection.truncate()
        logger.warning("Cleared all collections")