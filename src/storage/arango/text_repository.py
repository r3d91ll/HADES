"""
Enhanced ArangoDB repository implementation for text document storage.

This module extends the base ArangoRepository with specialized methods
for handling text documents, chunks, and embeddings with additional
support for ISNE-enhanced vectors.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import numpy as np
from datetime import datetime

from src.types.common import NodeData, EdgeData, EmbeddingVector, NodeID, EdgeID
from src.storage.arango.connection import ArangoConnection
from src.storage.arango.repository import ArangoRepository

logger = logging.getLogger(__name__)


class TextArangoRepository(ArangoRepository):
    """
    ArangoDB repository implementation specialized for text documents.
    
    Extends the base ArangoRepository with methods for:
    - Storing multiple embeddings per node (regular and ISNE-enhanced)
    - Bulk similarity calculation and edge creation
    - Optimized document-to-chunk traversals
    """
    
    def __init__(self, 
                 connection: ArangoConnection,
                 node_collection: Optional[str] = None,
                 edge_collection: Optional[str] = None,
                 graph_name: Optional[str] = None):
        """
        Initialize the Text ArangoDB repository.
        
        Args:
            connection: ArangoDB connection
            node_collection: Name of the node collection (default: "nodes")
            edge_collection: Name of the edge collection (default: "edges")
            graph_name: Name of the graph (default: "pathrag")
        """
        super().__init__(connection, node_collection, edge_collection, graph_name)
        
        # Set up additional indexes for text document-specific operations
        self._setup_text_indexes()
    
    def _setup_text_indexes(self) -> None:
        """Set up additional indexes for text document storage."""
        try:
            # Create indexes on collection
            collection = self.connection.get_collection(self.node_collection_name)
            
            # Index for document type
            if not self._has_index(collection, ["type"]):
                collection.add_hash_index(fields=["type"], unique=False)
                logger.info(f"Created hash index on 'type' in {self.node_collection_name}")
            
            # Index for parent_id (to quickly find chunks belonging to a document)
            if not self._has_index(collection, ["parent_id"]):
                collection.add_hash_index(fields=["parent_id"], unique=False)
                logger.info(f"Created hash index on 'parent_id' in {self.node_collection_name}")
                
            # Persistent index for embedding_type field
            if not self._has_index(collection, ["embedding_type"]):
                collection.add_persistent_index(fields=["embedding_type"], unique=False)
                logger.info(f"Created persistent index on 'embedding_type' in {self.node_collection_name}")
                
        except Exception as e:
            logger.error(f"Error setting up text document indexes: {e}")
    
    def _has_index(self, collection: Any, fields: List[str]) -> bool:
        """
        Check if collection has an index for the specified fields.
        
        Args:
            collection: ArangoDB collection
            fields: List of field names
            
        Returns:
            True if the index exists, False otherwise
        """
        indexes = collection.indexes()
        field_set = set(fields)
        
        for index in indexes:
            if set(index["fields"]) == field_set:
                return True
        return False
            
    async def store_embedding_with_type(
        self, 
        node_id: NodeID, 
        embedding: EmbeddingVector,
        embedding_type: str = "default"
    ) -> bool:
        """
        Store an embedding for a node with a specific embedding type.
        
        Args:
            node_id: The ID of the node
            embedding: The embedding vector
            embedding_type: Type of embedding (default, isne, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate node exists
            node = await self.get_node(node_id)
            if not node:
                logger.error(f"Node {node_id} not found")
                return False
            
            # Prepare embedding for storage
            embedding_doc = {
                "vector": embedding,
                "embedding_type": embedding_type,
                "dimension": len(embedding),
                "updated_at": datetime.now().isoformat()
            }
            
            # Update the node with the embedding
            update_data = {}
            if embedding_type == "default" or embedding_type == "":
                update_data["embedding"] = embedding_doc
            else:
                update_data[f"{embedding_type}_embedding"] = embedding_doc
            
            # Execute AQL query to update the node
            aql = f"""
            UPDATE @node_id WITH @update_data IN {self.node_collection_name}
            RETURN NEW
            """
            
            params = {
                "node_id": node_id,
                "update_data": update_data
            }
            
            result = await self._execute_aql(aql, params)
            return len(result) > 0
            
        except Exception as e:
            logger.error(f"Error storing embedding with type {embedding_type}: {e}")
            return False
    
    async def create_similarity_edges(
        self,
        chunk_embeddings: List[Tuple[NodeID, EmbeddingVector]],
        edge_type: str = "similar_to",
        threshold: float = 0.8,
        batch_size: int = 100
    ) -> int:
        """
        Create similarity edges between chunks based on embedding similarity.
        
        Args:
            chunk_embeddings: List of (chunk_id, embedding) pairs
            edge_type: Type of edge to create
            threshold: Similarity threshold (0.0 to 1.0)
            batch_size: Number of embeddings to process in each batch
            
        Returns:
            Number of edges created
        """
        # Skip if no chunks
        if not chunk_embeddings:
            return 0
            
        total_edges_created = 0
        
        try:
            # Process in batches
            for i in range(0, len(chunk_embeddings), batch_size):
                batch = chunk_embeddings[i:i+batch_size]
                batch_ids, batch_vectors = zip(*batch)
                
                # Calculate pairwise similarities using cosine similarity
                # Convert to numpy arrays for efficient computation
                vectors = np.array(batch_vectors)
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                normalized_vectors = vectors / norms
                similarities = np.dot(normalized_vectors, normalized_vectors.T)
                
                # Create edges for pairs above threshold
                edges_to_create = []
                
                # Find all pairs above threshold (excluding self-similarity)
                for j in range(len(batch)):
                    for k in range(j+1, len(batch)):  # Only upper triangle to avoid duplicates
                        similarity = float(similarities[j, k])
                        if similarity >= threshold:
                            edge_data = {
                                "_from": f"{self.node_collection_name}/{batch_ids[j]}",
                                "_to": f"{self.node_collection_name}/{batch_ids[k]}",
                                "type": edge_type,
                                "similarity": similarity,
                                "created_at": datetime.now().isoformat()
                            }
                            edges_to_create.append(edge_data)
                
                # Bulk insert edges
                if edges_to_create:
                    # Use AQL for bulk insert
                    aql = f"""
                    FOR edge IN @edges
                        INSERT edge INTO {self.edge_collection_name}
                        LET inserted = NEW
                    RETURN inserted
                    """
                    
                    params = {
                        "edges": edges_to_create
                    }
                    
                    result = await self._execute_aql(aql, params)
                    total_edges_created += len(result)
                    logger.info(f"Created {len(result)} {edge_type} edges in batch")
                
            logger.info(f"Created a total of {total_edges_created} {edge_type} edges")
            return total_edges_created
            
        except Exception as e:
            logger.error(f"Error creating similarity edges: {e}")
            return total_edges_created
    
    async def search_similar_with_data(
        self,
        query_vector: EmbeddingVector,
        limit: int = 10,
        node_types: Optional[List[str]] = None,
        embedding_type: str = "default"
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for nodes with similar embeddings and return full node data.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            node_types: Optional list of node types to filter by
            embedding_type: Type of embedding to search (default, isne)
            
        Returns:
            List of (node_data, similarity_score) pairs
        """
        try:
            embedding_field = "embedding" if embedding_type == "default" else f"{embedding_type}_embedding"
            
            # Build AQL for vector search with full node data
            aql = f"""
            LET query_vec = @query_vector
            FOR doc IN {self.node_collection_name}
                FILTER doc.{embedding_field} != null
                {f"FILTER doc.type IN @node_types" if node_types else ""}
                
                LET vec = doc.{embedding_field}.vector
                LET similarity = LENGTH(vec) == 0 ? 0 : 
                    COSINE_SIMILARITY(query_vec, vec)
                
                FILTER similarity >= @min_score
                
                SORT similarity DESC
                LIMIT @limit
                
                RETURN {{
                    "node": doc,
                    "score": similarity
                }}
            """
            
            params = {
                "query_vector": query_vector,
                "min_score": 0.5,  # Minimum similarity threshold
                "limit": limit
            }
            
            if node_types:
                params["node_types"] = node_types
                
            results = await self._execute_aql(aql, params)
            
            # Extract nodes and scores
            return [(item["node"], item["score"]) for item in results]
            
        except Exception as e:
            logger.error(f"Error in search_similar_with_data: {e}")
            return []
    
    async def get_document_with_chunks(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document with all its chunks.
        
        Args:
            document_id: ID of the document
            
        Returns:
            Document with its chunks, or None if not found
        """
        try:
            # Get the document
            document = await self.get_node(document_id)
            if not document:
                return None
                
            # Get chunks using a graph traversal for efficiency
            aql = f"""
            FOR v, e, p IN 1..1 OUTBOUND @start_vertex {self.graph_name}
                FILTER e.type == 'contains'
                SORT e.index ASC
                RETURN v
            """
            
            params = {
                "start_vertex": f"{self.node_collection_name}/{document_id}"
            }
            
            chunks = await self._execute_aql(aql, params)
            
            # Build complete document
            result = {
                "document": document,
                "chunks": chunks,
                "chunk_count": len(chunks)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting document with chunks: {e}")
            return None
