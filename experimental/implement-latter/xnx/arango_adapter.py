"""
ArangoDB adapter for XnX-enhanced PathRAG.

This module provides an adapter to replace the default PathRAG storage
with ArangoDB, supporting XnX notation for weighted path tuning.
"""

import sys
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Create dynamic import based on where we import ArangoDB connection from
try:
    # Try to import from our main package structure
    from src.db.arango_connection import ArangoConnection
    from src.xnx.traversal import (
        traverse_with_xnx_constraints, traverse_with_temporal_xnx,
        format_xnx_output, calculate_path_score,
        XnXTraversalError, InvalidNodeError, WeightThresholdError, TemporalConstraintError
    )
except ImportError:
    try:
        # Fall back to the old_hades_imports
        sys.path.append('old_hades_imports')
        from src.db.arango_connection import ArangoConnection
        # Import traversal functions - assuming they're also in old_hades_imports
        from src.xnx.traversal import (
            traverse_with_xnx_constraints, traverse_with_temporal_xnx,
            format_xnx_output, calculate_path_score,
            XnXTraversalError, InvalidNodeError, WeightThresholdError, TemporalConstraintError
        )
    except ImportError:
        raise ImportError("Could not find required modules in either location")


class ArangoPathRAGAdapter:
    """Adapter that connects PathRAG to ArangoDB with XnX notation support."""
    
    def __init__(self, 
                 arango_connection: Optional[ArangoConnection] = None,
                 db_name: str = "hades",
                 nodes_collection: str = "pathrag_nodes",
                 edges_collection: str = "pathrag_edges",
                 graph_name: str = "pathrag_graph"):
        """Initialize the ArangoDB adapter.
        
        Args:
            arango_connection: Existing ArangoDB connection or None to create new
            db_name: Name of the ArangoDB database
            nodes_collection: Name of the collection storing nodes
            edges_collection: Name of the collection storing edges
            graph_name: Name of the graph in ArangoDB
        """
        # Use provided connection or create a new one
        self.conn = arango_connection or ArangoConnection(db_name=db_name)
        self.db_name = db_name
        self.nodes_collection = nodes_collection
        self.edges_collection = edges_collection
        self.graph_name = graph_name
        
        # Ensure collections and graph exist
        self._ensure_collections()
        
    def _ensure_collections(self):
        """Ensure that required collections and graph exist."""
        # Create nodes collection if it doesn't exist
        if not self.conn.collection_exists(self.nodes_collection):
            self.conn.create_collection(self.nodes_collection)
            
        # Create edges collection if it doesn't exist
        if not self.conn.collection_exists(self.edges_collection):
            self.conn.create_edge_collection(self.edges_collection)
            
        # Create graph if it doesn't exist
        if not self.conn.graph_exists(self.graph_name):
            self.conn.create_graph(
                self.graph_name,
                edge_definitions=[{
                    'edge_collection': self.edges_collection,
                    'from_vertex_collections': [self.nodes_collection],
                    'to_vertex_collections': [self.nodes_collection]
                }]
            )
    
    def store_node(self, node_id: str, content: str, 
                  embedding: List[float], metadata: Dict[str, Any] = None) -> str:
        """Store a node in ArangoDB.
        
        Args:
            node_id: Unique identifier for the node
            content: Text content of the node
            embedding: Vector embedding of the node content
            metadata: Additional metadata for the node
            
        Returns:
            ArangoDB document key
        """
        metadata = metadata or {}
        
        # Create node document
        node_doc = {
            "_key": node_id,
            "content": content,
            "embedding": embedding,
            "metadata": metadata,
            "created_at": datetime.now().isoformat()
        }
        
        # Insert or replace the node
        result = self.conn.insert_document(
            self.nodes_collection, 
            node_doc, 
            overwrite=True
        )
        
        return result["_key"]
    
    def create_edge(self, from_node: str, to_node: str, 
                    weight: float = 1.0, 
                    metadata: Dict[str, Any] = None) -> str:
        """
        Create an edge between nodes in ArangoDB.
        
        This is a simplified version of create_relationship for basic edge creation.
        
        Args:
            from_node: Source node ID
            to_node: Target node ID
            weight: Edge weight (0.0 to 1.0)
            metadata: Additional metadata for the edge
            
        Returns:
            ArangoDB edge document key
        """
        return self.create_relationship(
            from_id=from_node, 
            to_id=to_node, 
            weight=weight, 
            metadata=metadata
        )
        
    def create_relationship(self, from_id: str, to_id: str, 
                           weight: float = 1.0, 
                           direction: int = -1,
                           temporal_bounds: Optional[Dict[str, str]] = None,
                           metadata: Dict[str, Any] = None) -> str:
        """Create an XnX relationship between nodes.
        
        Args:
            from_id: Source node ID
            to_id: Target node ID
            weight: XnX weight (0.0 to 1.0)
            direction: XnX direction (-1 for outbound, 1 for inbound)
            temporal_bounds: Optional time bounds for relationship
            metadata: Additional metadata
            
        Returns:
            ArangoDB edge document key
        """
        metadata = metadata or {}
        
        # Create edge document with XnX notation properties
        edge_doc = {
            "_from": f"{self.nodes_collection}/{from_id}",
            "_to": f"{self.nodes_collection}/{to_id}",
            "weight": weight,
            "direction": direction,
            "xnx_notation": f"{weight:.2f} {to_id} {direction}",
            "created_at": datetime.now().isoformat(),
            "metadata": metadata
        }
        
        # Add temporal bounds if provided
        if temporal_bounds:
            edge_doc["temporal_bounds"] = temporal_bounds
            # Update XnX notation to include temporal bounds
            time_str = f"[T[{temporal_bounds.get('start')}→{temporal_bounds.get('end')}]]"
            edge_doc["xnx_notation"] = f"{weight:.2f} {to_id} {direction}{time_str}"
        
        result = self.conn.insert_edge(
            self.edges_collection,
            edge_doc
        )
        
        return result["_key"]
    
    def query_paths(self, query_embedding: List[float], 
                   xnx_params, max_results: int = 10) -> List[Dict]:
        """Query paths based on XnX parameters.
        
        Args:
            query_embedding: Vector embedding for similarity search
            xnx_params: XnX query parameters
            max_results: Maximum number of results to return
            
        Returns:
            List of paths matching the query
        """
        # Construct AQL query with XnX filters
        min_weight = xnx_params.min_weight
        max_distance = xnx_params.max_distance
        direction = xnx_params.direction
        
        # Temporal constraint handling
        temporal_constraint = None
        if xnx_params.temporal_constraint:
            if isinstance(xnx_params.temporal_constraint, datetime):
                temporal_constraint = xnx_params.temporal_constraint.isoformat()
            else:
                temporal_constraint = xnx_params.temporal_constraint
        
        # Start with vector similarity search to find entry points
        query = f"""
        LET entry_points = (
            FOR doc IN {self.nodes_collection}
            LET similarity = VECTOR_DISTANCE(doc.embedding, @query_embedding)
            SORT similarity ASC
            LIMIT 5
            RETURN doc
        )
        
        LET paths = (
            FOR entry IN entry_points
            FOR v, e, p IN 1..@max_distance OUTBOUND entry._id 
            GRAPH @graph_name
            FILTER e.weight >= @min_weight
        """
        
        # Add direction filter if specified
        if direction is not None:
            query += f" FILTER e.direction == @direction"
            
        # Add temporal constraint if specified
        if temporal_constraint:
            query += f"""
            FILTER (
                e.temporal_bounds == NULL OR
                (e.temporal_bounds.start <= @temporal_constraint AND 
                 (e.temporal_bounds.end == NULL OR e.temporal_bounds.end >= @temporal_constraint))
            )
            """
            
        # Complete the query with sorting and limiting
        query += f"""
            SORT p.edges[*].weight DESC
            RETURN {{
                "path": p,
                "total_weight": SUM(p.edges[*].weight),
                "avg_weight": AVG(p.edges[*].weight),
                "length": LENGTH(p.edges),
                "content": LAST(p.vertices).content
            }}
        )
        
        FOR path IN paths
        SORT path.avg_weight * (1.0 / path.length) DESC
        LIMIT @max_results
        RETURN path
        """
        
        # Prepare query parameters
        bind_vars = {
            "query_embedding": query_embedding,
            "max_distance": max_distance,
            "min_weight": min_weight,
            "graph_name": self.graph_name,
            "max_results": max_results
        }
        
        if direction is not None:
            bind_vars["direction"] = direction
            
        if temporal_constraint:
            bind_vars["temporal_constraint"] = temporal_constraint
        
        # Execute the query
        results = self.conn.query(query, bind_vars=bind_vars)
        
        return results
        
    def get_paths_from_node(self, node_id: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """
        Get all paths starting from a specific node.
        
        Args:
            node_id: ID of the starting node
            max_depth: Maximum path depth to traverse
            
        Returns:
            List of paths with their nodes and total weight
        """
        # AQL query to find paths from node
        query = f"""
        FOR v, e, p IN 1..@max_depth OUTBOUND @start_id 
        GRAPH @graph_name
        RETURN {{
            "nodes": p.vertices,
            "edges": p.edges,
            "total_weight": SUM(p.edges[*].weight)
        }}
        """
        
        # Execute query
        bind_vars = {
            "start_id": f"{self.nodes_collection}/{node_id}",
            "max_depth": max_depth,
            "graph_name": self.graph_name
        }
        
        return self.conn.query(query, bind_vars)
    
    def get_weighted_paths(self, node_id: str, xnx_query: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """
        Get paths with XnX weighting applied.
        
        Args:
            node_id: ID of the starting node
            xnx_query: XnX query string for weighting
            max_depth: Maximum path depth to traverse
            
        Returns:
            List of paths with their XnX weighted score
        """
        # Simple XnX parser for demo purposes
        # This is a simplified version of what would actually happen
        weight_factor = 1.0
        domain_filter = None
        
        # Parse simple XnX queries like "X(domain='code')2" 
        if xnx_query.startswith("X("):
            parts = xnx_query.split(")", 1)
            if len(parts) > 1 and parts[1].isdigit():
                weight_factor = float(parts[1])
                attribute_part = parts[0][2:]
                if "=" in attribute_part:
                    key, value = attribute_part.split("=")
                    domain_filter = value.strip().strip("'\"")
        
        # AQL query to find paths from node with domain filter
        query = f"""
        FOR v, e, p IN 1..@max_depth OUTBOUND @start_id 
        GRAPH @graph_name
        LET domain_bonus = p.vertices[*].metadata.domain ANY == @domain_filter ? @weight_factor : 1.0
        LET base_score = SUM(p.edges[*].weight)
        LET xnx_score = base_score * domain_bonus
        SORT xnx_score DESC
        LIMIT 10
        RETURN {{
            "nodes": p.vertices,
            "edges": p.edges,
            "base_score": base_score,
            "xnx_score": xnx_score
        }}
        """
        
        # Execute query
        bind_vars = {
            "start_id": f"{self.nodes_collection}/{node_id}",
            "max_depth": max_depth,
            "graph_name": self.graph_name,
            "domain_filter": domain_filter,
            "weight_factor": weight_factor
        }
        
        return self.conn.query(query, bind_vars)
    
    def find_similar_nodes(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find nodes with embeddings similar to the query embedding.
        
        Args:
            query_embedding: Query vector embedding
            top_k: Number of similar nodes to return
            
        Returns:
            List of similar nodes with similarity scores
        """
        # Fetch all documents first (not efficient but works for demo)
        query = f"""
        FOR doc IN {self.nodes_collection}
        RETURN doc
        """
        
        nodes = self.conn.query(query)
        
        # Calculate cosine similarity in Python (since ArangoDB may not have vector extensions)
        import numpy as np
        from scipy import spatial
        
        # Convert query embedding to numpy array
        query_vector = np.array(query_embedding)
        
        # Calculate similarity for each node
        scored_nodes = []
        for node in nodes:
            # Skip nodes without embeddings
            if "embedding" not in node:
                continue
                
            node_vector = np.array(node["embedding"])
            
            # Calculate cosine similarity
            try:
                similarity = 1 - spatial.distance.cosine(query_vector, node_vector)
            except Exception:
                # If vectors have different dimensions or other issues
                similarity = 0.0
                
            # Add similarity score to node
            node["similarity"] = float(similarity)
            scored_nodes.append(node)
        
        # Sort by similarity (descending) and take top_k
        scored_nodes.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        return scored_nodes[:top_k]
        
    def traverse_with_xnx(self, 
                         start_node: str,
                         min_weight: float = 0.8,
                         max_distance: int = 3,
                         direction: str = "any") -> List[Dict[str, Any]]:
        """
        Traverse the graph with XnX constraints on weight and direction.
        
        Args:
            start_node: ID of the starting node
            min_weight: Minimum edge weight (0.0-1.0)
            max_distance: Maximum path distance
            direction: Path direction ('inbound', 'outbound', or 'any')
            
        Returns:
            List of paths matching the constraints
            
        Raises:
            InvalidNodeError: If the start_node does not exist
            WeightThresholdError: If no paths meet the weight threshold
        """
        # Ensure node ID has the correct format
        start_node_id = self._ensure_node_id_format(start_node)
        
        # Use the traversal function
        try:
            paths = traverse_with_xnx_constraints(
                db_connection=self.conn,
                start_node=start_node_id,
                min_weight=min_weight,
                max_distance=max_distance,
                direction=direction,
                graph_name=self.graph_name,
                nodes_collection=self.nodes_collection,
                edges_collection=self.edges_collection
            )
            
            # Enhance with path scores
            for path in paths:
                path['xnx_score'] = calculate_path_score(path)
                path['log_score'] = calculate_path_score(path, use_log_scale=True)
                
            return paths
            
        except XnXTraversalError as e:
            logger.warning(f"XnX traversal error: {str(e)}")
            # Re-raise to maintain the specific error type
            raise
        except Exception as e:
            logger.error(f"Unexpected error in traverse_with_xnx: {str(e)}")
            raise
    
    def traverse_with_temporal_xnx(self,
                                 start_node: str,
                                 min_weight: float = 0.8,
                                 max_distance: int = 3,
                                 direction: str = "any",
                                 valid_at: Optional[Union[str, datetime]] = None) -> List[Dict[str, Any]]:
        """
        Traverse the graph with XnX constraints including temporal validity.
        
        Args:
            start_node: ID of the starting node
            min_weight: Minimum edge weight (0.0-1.0)
            max_distance: Maximum path distance
            direction: Path direction ('inbound', 'outbound', or 'any')
            valid_at: Time point for which edges should be valid (ISO format string or datetime)
            
        Returns:
            List of paths matching the constraints
            
        Raises:
            InvalidNodeError: If the start_node does not exist
            WeightThresholdError: If no paths meet the weight threshold
            TemporalConstraintError: If no paths are valid at the requested time
        """
        # Ensure node ID has the correct format
        start_node_id = self._ensure_node_id_format(start_node)
        
        # Use the temporal traversal function
        try:
            paths = traverse_with_temporal_xnx(
                db_connection=self.conn,
                start_node=start_node_id,
                min_weight=min_weight,
                max_distance=max_distance,
                direction=direction,
                valid_at=valid_at,
                graph_name=self.graph_name,
                nodes_collection=self.nodes_collection,
                edges_collection=self.edges_collection
            )
            
            # Enhance with path scores
            for path in paths:
                path['xnx_score'] = calculate_path_score(path)
                path['log_score'] = calculate_path_score(path, use_log_scale=True)
                
            return paths
            
        except XnXTraversalError as e:
            logger.warning(f"XnX temporal traversal error: {str(e)}")
            # Re-raise to maintain the specific error type
            raise
        except Exception as e:
            logger.error(f"Unexpected error in traverse_with_temporal_xnx: {str(e)}")
            raise
    
    def format_paths_as_xnx(self, paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format paths to include XnX notation strings for each edge.
        
        Args:
            paths: List of paths from traverse_with_xnx or traverse_with_temporal_xnx
            
        Returns:
            Same paths with additional xnx_strings field for each path
        """
        formatted_paths = []
        
        for path in paths:
            edges = path.get('edges', [])
            xnx_strings = []
            
            for edge in edges:
                xnx_strings.append(format_xnx_output(edge))
                
            # Add the formatted strings to the path
            path_copy = path.copy()
            path_copy['xnx_strings'] = xnx_strings
            formatted_paths.append(path_copy)
            
        return formatted_paths
        
    def _ensure_node_id_format(self, node_id: str) -> str:
        """
        Ensure the node ID has the correct format for traversal.
        
        Args:
            node_id: Node ID to format
            
        Returns:
            Correctly formatted node ID
        """
        # If the ID already includes the collection name, return as is
        if '/' in node_id:
            return node_id
            
        # Otherwise, prefix with the nodes collection name
        return f"{self.nodes_collection}/{node_id}"
