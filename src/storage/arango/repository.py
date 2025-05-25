"""
ArangoDB implementation of the UnifiedRepository interface for HADES-PathRAG.

This module provides a concrete implementation of the repository interfaces
using ArangoDB as the underlying storage system, handling documents, graphs,
and vector operations in a unified way.
"""

import logging
from typing import Dict, List, Any, Optional, Union, cast, MutableMapping, Sequence, Tuple, Iterator
from datetime import datetime
import numpy as np
from arango.exceptions import DocumentInsertError, DocumentUpdateError, DocumentDeleteError, AQLQueryExecuteError
from arango.cursor import Cursor

from src.types.common import NodeData, EdgeData, EmbeddingVector, NodeID, EdgeID
from src.storage.arango.connection import ArangoConnection
from .repository_interfaces import UnifiedRepository

# Set up logging
logger = logging.getLogger(__name__)


class ArangoRepository(UnifiedRepository):
    """
    Unified ArangoDB repository for HADES-PathRAG.
    
    This class implements the UnifiedRepository interface using ArangoDB as
    the storage backend, providing access to document, graph, and vector operations.
    """
    
    # Default collection names
    DEFAULT_NODE_COLLECTION = "nodes"
    DEFAULT_EDGE_COLLECTION = "edges"
    DEFAULT_GRAPH_NAME = "pathrag"
    
    def __init__(self, 
                 connection: ArangoConnection,
                 node_collection: Optional[str] = None,
                 edge_collection: Optional[str] = None,
                 graph_name: Optional[str] = None):
        """
        Initialize the ArangoDB repository.
        
        Args:
            connection: ArangoDB connection
            node_collection: Name of the node collection (default: "nodes")
            edge_collection: Name of the edge collection (default: "edges")
            graph_name: Name of the graph (default: "pathrag")
        """
        self.connection = connection
        self.node_collection_name = node_collection or self.DEFAULT_NODE_COLLECTION
        self.edge_collection_name = edge_collection or self.DEFAULT_EDGE_COLLECTION
        self.graph_name = graph_name or self.DEFAULT_GRAPH_NAME
        
        # Ensure collections exist
        self.setup_collections()
    
    def setup_collections(self) -> None:
        """
        Set up the necessary collections and indexes in ArangoDB.
        """
        try:
            # Create graph if it doesn't exist
            if not self.connection.graph_exists(self.graph_name):
                # Create edge definitions for the graph
                edge_definitions = [
                    {
                        'edge_collection': self.edge_collection_name,
                        'from_vertex_collections': [self.node_collection_name],
                        'to_vertex_collections': [self.node_collection_name]
                    }
                ]
                self.connection.create_graph(self.graph_name, edge_definitions)
                logger.info(f"Created graph {self.graph_name}")
            
            # Create node collection if it doesn't exist
            if not self.connection.collection_exists(self.node_collection_name):
                self.connection.create_collection(self.node_collection_name)
                
                # Create indexes for full-text search
                self._create_indexes(self.node_collection_name)
                logger.info(f"Created node collection {self.node_collection_name}")
            
            # Create edge collection if it doesn't exist
            if not self.connection.collection_exists(self.edge_collection_name):
                self.connection.create_edge_collection(self.edge_collection_name)
                logger.info(f"Created edge collection {self.edge_collection_name}")
            
        except Exception as e:
            logger.error(f"Error setting up collections: {e}")
            raise
    
    def _create_indexes(self, collection_name: str) -> None:
        """
        Create necessary indexes on a collection.
        
        Args:
            collection_name: Name of the collection
        """
        # Create full-text index on 'content' field
        self.connection.raw_db.collection(collection_name).add_fulltext_index(
            fields=["content"], min_length=3
        )
        
        # Create hash index on 'type' field for fast filtering
        self.connection.raw_db.collection(collection_name).add_hash_index(
            fields=["type"], unique=False
        )
        
        # If ArangoDB version supports vector indexes, create one for embeddings
        try:
            self.connection.raw_db.collection(collection_name).add_persistent_index(
                fields=["embedding"], unique=False
            )
        except Exception as e:
            logger.warning(f"Could not create index on embedding field: {e}")
            
    def create_indexes(self) -> bool:
        """
        Create necessary indexes on the node collection.
        
        Returns:
            bool: True if indexes were created successfully, False otherwise
        """
        try:
            self._create_indexes(self.node_collection_name)
            return True
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            return False
    
    # Document Repository Implementation
    
    def store_document(self, document: NodeData) -> NodeID:
        """
        Store a document in ArangoDB.
        
        Args:
            document: The document data to store
            
        Returns:
            The ID of the stored document
        """
        # Create document data for ArangoDB
        doc_data = self._prepare_document_data(document)
        
        try:
            # Insert document into collection
            result = self.connection.insert_document(self.node_collection_name, doc_data)
            
            # Return the document key as the ID
            return NodeID(result["_key"] if "_key" in result else result["_id"].split('/')[-1])
        except Exception as e:
            logger.error(f"Error storing document: {e}")
            raise
    
    def get_document(self, document_id: NodeID) -> Optional[NodeData]:
        """
        Retrieve a document by its ID.
        
        Args:
            document_id: The ID of the document to retrieve
            
        Returns:
            The document data if found, None otherwise
        """
        try:
            # Construct AQL query to get document by key
            query = f"FOR doc IN @@collection FILTER doc._key == @key RETURN doc"
            bind_vars: Dict[str, Any] = {"@collection": self.node_collection_name, "key": str(document_id)}
            
            # Execute query
            cursor = self.connection.raw_db.aql.execute(query, bind_vars=bind_vars)
            
            # Process results
            cursor_iter = cast(Iterator[Dict[str, Any]], cursor)
            doc = next(cursor_iter, None)
            
            if doc is None:
                return None
            
            # Convert ArangoDB document to NodeData
            return self._convert_to_node_data(doc)
        except Exception as e:
            logger.error(f"Error retrieving document {document_id}: {e}")
            return None
    
    def update_document(self, document_id: NodeID, updates: Dict[str, Any]) -> bool:
        """
        Update a document by its ID.
        
        Args:
            document_id: The ID of the document to update
            updates: The fields to update and their new values
            
        Returns:
            True if the update was successful, False otherwise
        """
        try:
            # Construct AQL query to update document
            query = f"""
            UPDATE @key WITH @updates IN @@collection
            RETURN NEW
            """
            bind_vars: Dict[str, Any] = {
                "@collection": self.node_collection_name,
                "key": str(document_id),
                "updates": updates
            }
            
            # Execute query
            result = self.connection.raw_db.aql.execute(query, bind_vars=bind_vars)
            
            return True
        except Exception as e:
            logger.error(f"Error updating document {document_id}: {e}")
            return False
    
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
        try:
            # Build the AQL query
            aql_filters = []
            bind_vars: Dict[str, Any] = {"query": f"%{query}%", "limit": limit}
            
            # Add filters if provided
            if filters:
                for key, value in filters.items():
                    aql_filters.append(f"FILTER doc.{key} == @{key}")
                    bind_vars[key] = value
            
            # Construct the AQL query
            aql_query = f"""
            FOR doc IN @@collection
                FILTER doc.content LIKE @query OR doc.title LIKE @query
                {' '.join(aql_filters)}
                LIMIT @limit
                RETURN doc
            """
            
            bind_vars["@collection"] = self.node_collection_name
            
            # Execute the query
            cursor = self.connection.raw_db.aql.execute(aql_query, bind_vars=bind_vars)
            
            # Convert results to NodeData
            cursor_iter = cast(Iterator[Dict[str, Any]], cursor)
            return [self._convert_to_node_data(doc) for doc in cursor_iter]
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    # Helper methods for document operations
    
    def _prepare_document_data(self, document: NodeData) -> Dict[str, Any]:
        """
        Prepare a document for storage in ArangoDB.
        
        Args:
            document: The document data
            
        Returns:
            Dict with ArangoDB-compatible format
        """
        # Create a copy of the document to avoid modifying the original
        doc_data = dict(document)
        
        # Set _key if id is provided
        if 'id' in doc_data:
            doc_data['_key'] = doc_data.pop('id')
        
        # Handle embedding if it's a numpy array
        if 'embedding' in doc_data and doc_data['embedding'] is not None:
            if isinstance(doc_data['embedding'], np.ndarray):
                doc_data['embedding'] = doc_data['embedding'].tolist()
        
        # Ensure created_at is a string
        if 'created_at' in doc_data and doc_data['created_at'] is not None:
            if isinstance(doc_data['created_at'], datetime):
                doc_data['created_at'] = doc_data['created_at'].isoformat()
        else:
            doc_data['created_at'] = datetime.now().isoformat()
        
        # Ensure updated_at is a string
        if 'updated_at' in doc_data and doc_data['updated_at'] is not None:
            if isinstance(doc_data['updated_at'], datetime):
                doc_data['updated_at'] = doc_data['updated_at'].isoformat()
        else:
            doc_data['updated_at'] = doc_data['created_at']
        
        return doc_data
    
    def _convert_to_node_data(self, doc: Dict[str, Any]) -> NodeData:
        """
        Convert an ArangoDB document to NodeData.
        
        Args:
            doc: The ArangoDB document
            
        Returns:
            NodeData representation
        """
        # Create a copy of the document
        node_data = dict(doc)
        
        # Convert _key to id
        if '_key' in node_data:
            node_data['id'] = node_data.pop('_key')
        
        # Remove ArangoDB-specific fields
        for field in ['_id', '_rev']:
            if field in node_data:
                node_data.pop(field)
        
        # Cast to NodeData
        return cast(NodeData, node_data)
        
    # Graph Repository Implementation
    
    def create_edge(self, edge: EdgeData) -> EdgeID:
        """
        Create an edge between nodes in ArangoDB.
        
        Args:
            edge: The edge data
            
        Returns:
            The ID of the created edge
        """
        # Prepare edge data for ArangoDB
        edge_data = self._prepare_edge_data(edge)
        
        try:
            # Insert edge into collection
            # Use raw_db to insert edge directly into the collection
            edge_collection = self.connection.raw_db.collection(self.edge_collection_name)
            result = edge_collection.insert(edge_data)
            
            # Return the edge key as the ID
            result_dict = cast(Dict[str, Any], result)
            return EdgeID(result_dict["_key"] if "_key" in result_dict else result_dict["_id"].split('/')[-1])
        except Exception as e:
            logger.error(f"Error creating edge: {e}")
            raise
    
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
        try:
            # Build AQL query parameters
            bind_vars: Dict[str, Any] = {
                "node_id": f"{self.node_collection_name}/{node_id}",
                "@edge_collection": self.edge_collection_name,
                "@node_collection": self.node_collection_name
            }
            
            # Add edge type filter if provided
            edge_filter = ""
            if edge_types:
                edge_filter = "FILTER edge.type IN @edge_types"
                bind_vars["edge_types"] = edge_types
            
            # Determine direction
            if direction == "outbound":
                aql_query = f"""
                FOR vertex, edge IN 1..1 OUTBOUND @node_id GRAPH @graph_name
                    {edge_filter}
                    RETURN {{ edge: edge, vertex: vertex }}
                """
            elif direction == "inbound":
                aql_query = f"""
                FOR vertex, edge IN 1..1 INBOUND @node_id GRAPH @graph_name
                    {edge_filter}
                    RETURN {{ edge: edge, vertex: vertex }}
                """
            else:  # "any"
                aql_query = f"""
                FOR vertex, edge IN 1..1 ANY @node_id GRAPH @graph_name
                    {edge_filter}
                    RETURN {{ edge: edge, vertex: vertex }}
                """
            
            bind_vars["graph_name"] = self.graph_name
            
            # Execute the query
            cursor = self.connection.raw_db.aql.execute(aql_query, bind_vars=bind_vars)
            
            # Process results
            result: List[Tuple[EdgeData, NodeData]] = []
            cursor_iter = cast(Iterator[Dict[str, Any]], cursor)
            for item in cursor_iter:
                    edge_data = self._convert_to_edge_data(item["edge"])
                    node_data = self._convert_to_node_data(item["vertex"])
                    result.append((edge_data, node_data))
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting edges for node {node_id}: {e}")
            return []
    
    def traverse_graph(self, start_id: NodeID, edge_types: Optional[List[str]] = None,
                      max_depth: int = 3) -> Dict[str, Any]:
        """
        Traverse the graph starting from a node.
        
        Args:
            start_id: The ID of the starting node
            edge_types: Optional list of edge types to traverse
            max_depth: Maximum traversal depth
            
        Returns:
            Dictionary with traversal results (nodes and edges)
        """
        try:
            # Build traversal options
            options = {
                "direction": "outbound",
                "uniqueVertices": "global",
                "maxDepth": max_depth
            }
            
            # Instead of using the graph API directly, use AQL for traversal
            # which provides more flexibility and allows use through our connection wrapper
            edge_filter = ""
            bind_vars: Dict[str, Any] = {
                "start_vertex": f"{self.node_collection_name}/{start_id}",
                "max_depth": max_depth,
                "@edge_collection": self.edge_collection_name,
            }
            
            # Add edge type filter if provided
            if edge_types:
                edge_filter = "FILTER edge.type IN @edge_types"
                bind_vars["edge_types"] = edge_types
            
            # AQL query for graph traversal
            aql_query = f"""
            FOR vertex, edge, path IN 1..@max_depth OUTBOUND @start_vertex GRAPH @graph_name
                {edge_filter}
                COLLECT AGGREGATE vertices = PUSH(vertex),
                         edges = PUSH(edge),
                         paths = PUSH(path)
                RETURN {{ vertices: vertices, edges: edges, paths: paths }}
            """
            
            bind_vars["graph_name"] = self.graph_name
            
            # Execute traversal query
            results = self.connection.raw_db.aql.execute(aql_query, bind_vars=bind_vars)
            
            # Process results
            result: Dict[str, Any] = {"vertices": [], "edges": [], "paths": []}
            
            # Extract traversal data from results
            results_iter = cast(Iterator[Dict[str, Any]], results)
            traversal_data = next(results_iter, None)
            if traversal_data:
                result["vertices"] = [self._convert_to_node_data(v) for v in traversal_data.get("vertices", [])]
                result["edges"] = [self._convert_to_edge_data(e) for e in traversal_data.get("edges", [])]
                result["paths"] = traversal_data.get("paths", [])
            
            return result
            
        except Exception as e:
            logger.error(f"Error traversing graph from {start_id}: {e}")
            return {"vertices": [], "edges": [], "paths": []}
    
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
        try:
            # Prepare AQL query
            bind_vars: Dict[str, Any] = {
                "from": f"{self.node_collection_name}/{from_id}",
                "to": f"{self.node_collection_name}/{to_id}",
                "graph": self.graph_name
            }
            
            # Build edge filter if needed
            edge_filter = ""
            if edge_types and isinstance(edge_types, list):
                edge_filter = "FILTER CURRENT.edge.type IN @edge_types"
                bind_vars["edge_types"] = edge_types
            
            # AQL query for shortest path
            aql_query = f"""
            FOR path IN OUTBOUND SHORTEST_PATH @from TO @to GRAPH @graph
                {edge_filter}
                RETURN {{node: DOCUMENT(path.vertex), edge: path.edge ? DOCUMENT(path.edge) : null}}
            """
            
            # Execute query
            cursor = self.connection.raw_db.aql.execute(aql_query, bind_vars=bind_vars)
            
            # Process and return results
            path = []
            if cursor and isinstance(cursor, Cursor):
                for item in cursor:
                    path_item = {
                        "node": self._convert_to_node_data(item["node"]) if item["node"] else None,
                        "edge": self._convert_to_edge_data(item["edge"]) if item["edge"] else None
                    }
                    path.append(path_item)
            
            return path
            
        except Exception as e:
            logger.error(f"Error finding shortest path from {from_id} to {to_id}: {e}")
            return []
    
    # Helper methods for graph operations
    
    def _prepare_edge_data(self, edge: EdgeData) -> Dict[str, Any]:
        """
        Prepare an edge for storage in ArangoDB.
        
        Args:
            edge: The edge data
            
        Returns:
            Dict with ArangoDB-compatible format
        """
        # Create a copy of the edge to avoid modifying the original
        edge_data = dict(edge)
        
        # Set _key if id is provided
        if 'id' in edge_data:
            edge_data['_key'] = edge_data.pop('id')
        
        # Set _from and _to fields for ArangoDB edge collection
        edge_data['_from'] = f"{self.node_collection_name}/{edge_data.pop('source_id')}"
        edge_data['_to'] = f"{self.node_collection_name}/{edge_data.pop('target_id')}"
        
        # Ensure created_at is a string
        if 'created_at' in edge_data and edge_data['created_at'] is not None:
            if isinstance(edge_data['created_at'], datetime):
                edge_data['created_at'] = edge_data['created_at'].isoformat()
        else:
            edge_data['created_at'] = datetime.now().isoformat()
        
        # Ensure updated_at is a string
        if 'updated_at' in edge_data and edge_data['updated_at'] is not None:
            if isinstance(edge_data['updated_at'], datetime):
                edge_data['updated_at'] = edge_data['updated_at'].isoformat()
        else:
            edge_data['updated_at'] = edge_data['created_at']
        
        return edge_data
    
    def _convert_to_edge_data(self, edge: Dict[str, Any]) -> EdgeData:
        """
        Convert an ArangoDB edge to EdgeData.
        
        Args:
            edge: The ArangoDB edge
            
        Returns:
            EdgeData representation
        """
        # Create a copy of the edge
        edge_data = dict(edge)
        
        # Convert _key to id
        if '_key' in edge_data:
            edge_data['id'] = edge_data.pop('_key')
        
        # Convert _from and _to to source_id and target_id
        if '_from' in edge_data:
            from_parts = edge_data.pop('_from').split('/')
            edge_data['source_id'] = from_parts[-1] if len(from_parts) > 1 else from_parts[0]
        
        if '_to' in edge_data:
            to_parts = edge_data.pop('_to').split('/')
            edge_data['target_id'] = to_parts[-1] if len(to_parts) > 1 else to_parts[0]
        
        # Remove ArangoDB-specific fields
        for field in ['_id', '_rev']:
            if field in edge_data:
                edge_data.pop(field)
        
        # Cast to EdgeData
        return cast(EdgeData, edge_data)
        
    # Vector Repository Implementation
    
    def store_embedding(self, node_id: NodeID, embedding: EmbeddingVector, 
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store an embedding for a node in ArangoDB.
        
        Args:
            node_id: The ID of the node
            embedding: The vector embedding
            metadata: Optional metadata about the embedding
            
        Returns:
            True if the operation was successful, False otherwise
        """
        try:
            # Prepare embedding data
            update_data = {}
            
            # Convert numpy array to list if needed
            if isinstance(embedding, np.ndarray):
                update_data["embedding"] = embedding.tolist()
            else:
                update_data["embedding"] = embedding
            
            # Add metadata if provided
            if metadata:
                # If there's existing metadata, we'll update it
                update_data["embedding_metadata"] = metadata
            
            # Add embedding model info if provided in metadata
            if metadata and "model" in metadata:
                update_data["embedding_model"] = metadata["model"]
            
            # Update the node with the embedding
            self.update_document(node_id, update_data)
            
            return True
        except Exception as e:
            logger.error(f"Error storing embedding for node {node_id}: {e}")
            return False
    
    def get_embedding(self, node_id: NodeID) -> Optional[EmbeddingVector]:
        """
        Get the embedding for a node from ArangoDB.
        
        Args:
            node_id: The ID of the node
            
        Returns:
            The vector embedding if found, None otherwise
        """
        try:
            # Get document from collection
            doc = self.get_document(node_id)
            
            if doc is None or "embedding" not in doc:
                return None
            
            # Return the embedding
            embedding = doc.get("embedding")
            
            # Convert to numpy array for consistency
            if embedding is not None and not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding, dtype=np.float32)
            
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding for node {node_id}: {e}")
            return None
    
    def search_similar(self, embedding: EmbeddingVector, 
                      filters: Optional[Dict[str, Any]] = None,
                      limit: int = 10) -> List[Tuple[NodeData, float]]:
        """
        Search for nodes with similar embeddings using ArangoDB.
        
        Args:
            embedding: The query embedding
            filters: Optional filters to apply to the search
            limit: Maximum number of results to return
            
        Returns:
            List of nodes with similarity scores
        """
        try:
            # Convert embedding to list if it's a numpy array
            query_vector = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            
            # Prepare AQL query
            aql_filters = []
            bind_vars: Dict[str, Any] = {"embedding": query_vector, "limit": limit}
            
            # Add filters if provided
            if filters:
                for key, value in filters.items():
                    aql_filters.append(f"FILTER doc.{key} == @{key}")
                    bind_vars[key] = value
            
            # AQL query for vector search using dot product similarity
            # Note: This is a simplified version - for production, use ArangoDB's
            # built-in vector search capabilities if available (ArangoSearch with vector index)
            aql_query = f"""
            FOR doc IN @@collection
                FILTER doc.embedding != null
                {' '.join(aql_filters)}
                LET similarity = LENGTH(doc.embedding) == LENGTH(@embedding) ? 
                    SQRT(1 - SUM(
                        FOR i IN 0..LENGTH(@embedding)-1
                        RETURN POW(doc.embedding[i] - @embedding[i], 2)
                    ) / (LENGTH(@embedding) * 2))
                    : 0
                SORT similarity DESC
                LIMIT @limit
                RETURN {{ document: doc, score: similarity }}
            """
            
            bind_vars["@collection"] = self.node_collection_name
            
            # Execute the query
            cursor = self.connection.raw_db.aql.execute(aql_query, bind_vars=bind_vars)
            
            # Process results
            results: List[Tuple[NodeData, float]] = []
            cursor_iter = cast(Iterator[Dict[str, Any]], cursor)
            for item in cursor_iter:
                    node_data = self._convert_to_node_data(item["document"])
                    score = item["score"]
                    results.append((node_data, score))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar nodes: {e}")
            return []
    
    def hybrid_search(self, text_query: str, embedding: Optional[EmbeddingVector] = None,
                     filters: Optional[Dict[str, Any]] = None,
                     limit: int = 10) -> List[Tuple[NodeData, float]]:
        """
        Perform a hybrid search using both text and vector similarity in ArangoDB.
        
        Args:
            text_query: The text search query
            embedding: Optional embedding for vector search
            filters: Optional filters to apply to the search
            limit: Maximum number of results to return
            
        Returns:
            List of nodes with combined relevance scores
        """
        try:
            # Prepare AQL query
            aql_filters = []
            bind_vars: Dict[str, Any] = {"query": f"%{text_query}%", "limit": limit}
            
            # Add filters if provided
            if filters:
                for key, value in filters.items():
                    aql_filters.append(f"FILTER doc.{key} == @{key}")
                    bind_vars[key] = value
            
            # Handle vector search component if embedding is provided
            vector_component = ""
            if embedding is not None:
                query_vector = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                bind_vars["embedding"] = query_vector
                vector_component = """
                    LET vectorScore = doc.embedding != null && LENGTH(doc.embedding) == LENGTH(@embedding) ? 
                        SQRT(1 - SUM(
                            FOR i IN 0..LENGTH(@embedding)-1
                            RETURN POW(doc.embedding[i] - @embedding[i], 2)
                        ) / (LENGTH(@embedding) * 2))
                        : 0
                """
            else:
                vector_component = "LET vectorScore = 0"
            
            # AQL query for hybrid search
            aql_query = f"""
            FOR doc IN @@collection
                FILTER doc.content LIKE @query OR doc.title LIKE @query
                {' '.join(aql_filters)}
                {vector_component}
                LET textScore = doc.content LIKE @query ? 0.6 : (doc.title LIKE @query ? 0.8 : 0)
                LET combinedScore = textScore * 0.6 + vectorScore * 0.4
                SORT combinedScore DESC
                LIMIT @limit
                RETURN {{ document: doc, score: combinedScore }}
            """
            
            bind_vars["@collection"] = self.node_collection_name
            
            # Execute the query
            cursor = self.connection.raw_db.aql.execute(aql_query, bind_vars=bind_vars)
            
            # Process results
            results: List[Tuple[NodeData, float]] = []
            cursor_iter = cast(Iterator[Dict[str, Any]], cursor)
            for item in cursor_iter:
                    node_data = self._convert_to_node_data(item["document"])
                    score = item["score"]
                    results.append((node_data, score))
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing hybrid search: {e}")
            return []
    
    # Additional Repository Methods
    
    def get_most_connected_nodes(self, limit: int = 10) -> Dict[str, int]:
        """
        Get the most connected nodes in the repository.
        
        Args:
            limit: Maximum number of nodes to return
            
        Returns:
            Dictionary mapping node keys to connection counts
        """
        try:
            # Get most connected nodes
            aql_query = """
            FOR d IN @@nodes
                LET links = LENGTH(
                    FOR l IN @@edges
                        FILTER l._from == d._id OR l._to == d._id
                        RETURN 1
                )
                SORT links DESC
                LIMIT @limit
                RETURN { "key": d._key, "links": links }
            """
            bind_vars = cast(MutableMapping[str, Any], {"limit": limit, "@nodes": self.node_collection_name, "@edges": self.edge_collection_name})
            cursor = self.connection.raw_db.aql.execute(aql_query, bind_vars=bind_vars)
            
            # Compile top nodes
            top_nodes_stats: Dict[str, int] = {}
            cursor_iter = cast(Iterator[Dict[str, Any]], cursor)
            for node_obj in cursor_iter:
                if isinstance(node_obj, dict) and "key" in node_obj and "links" in node_obj:
                    top_nodes_stats[str(node_obj["key"])] = int(node_obj["links"])
            
            return top_nodes_stats
        except Exception as e:
            logger.error(f"Error getting most connected nodes: {e}")
            return {}
            
    def has_document_vectors(self) -> bool:
        """
        Check if any documents have vectors.
        
        Returns:
            True if vectors exist, False otherwise
        """
        try:
            # Query for any document with a non-null embedding field using AQL
            aql_query = "FOR d IN @@nodes FILTER d.embedding != null LIMIT 1 RETURN true"
            bind_vars = cast(MutableMapping[str, Any], {"@nodes": self.node_collection_name})
            cursor = self.connection.raw_db.aql.execute(aql_query, bind_vars=bind_vars)
            
            # Check cursor for results
            cursor_iter = cast(Iterator[bool], cursor)
            return any(cursor_iter)
        except Exception as e:
            logger.error(f"Error checking for document vectors: {e}")
            return False

    def collection_stats(self) -> Dict[str, Any]: 
        """
        Get statistics about the repository.
        
        Returns:
            Dictionary with repository statistics
        """
        try:
            stats: Dict[str, Any] = {}
            
            # Query stats using AQL
            node_stats_query = f"""
            RETURN LENGTH(@@collection)
            """
            node_stats_result = self.connection.raw_db.aql.execute(node_stats_query, bind_vars={"@collection": self.node_collection_name})
            node_result_iter = cast(Iterator[int], node_stats_result)
            node_count = next(node_result_iter, 0)
            
            edge_stats_query = f"""
            RETURN LENGTH(@@collection)
            """
            edge_stats_result = self.connection.raw_db.aql.execute(edge_stats_query, bind_vars={"@collection": self.edge_collection_name})
            edge_result_iter = cast(Iterator[int], edge_stats_result)
            edge_count = next(edge_result_iter, 0)
            
            # Create stats dictionaries
            node_stats = {"count": node_count}
            edge_stats = {"count": edge_count}
            
            # Combine stats
            stats["nodes"] = {
                "count": node_stats["count"],
                "size": node_stats.get("size", 0)
            }
            
            stats["edges"] = {
                "count": edge_stats["count"],
                "size": edge_stats.get("size", 0)
            }
            
            # Add basic index info since we can't directly access index details
            stats["indexes"] = [{"type": "persistent", "fields": ["type"]}]
            
            # Set vector index info - check if we can execute a vector query as a basic test
            try:
                # Try a simple vector query
                vector_test_query = """
                FOR doc IN @@collection
                    FILTER doc.embedding != null
                    LIMIT 1
                    RETURN 1
                """
                self.connection.raw_db.aql.execute(vector_test_query, bind_vars={"@collection": self.node_collection_name})
                stats["has_vector_index"] = True
            except Exception:
                stats["has_vector_index"] = False
            
            # Get top document nodes
            top_nodes = self.get_most_connected_nodes(10)
            stats["top_connected"] = top_nodes
            
            # Get boolean flags
            has_vectors = self.has_document_vectors()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting repository stats: {e}")
            return {"nodes": {"count": 0}, "edges": {"count": 0}}
