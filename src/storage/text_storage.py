"""
Text Storage Module for HADES-PathRAG.

This module provides integration between text processing pipelines and ArangoDB storage,
handling documents, chunks, embeddings, and relationships.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from src.types.common import NodeData, EdgeData, EmbeddingVector, NodeID, EdgeID
from src.storage.arango.connection import ArangoConnection
from src.storage.arango.repository import ArangoRepository
from src.storage.arango.text_repository import TextArangoRepository
from typing import List, Dict, Any, Optional, Tuple, Union, cast, TypeVar, Generic

logger = logging.getLogger(__name__)


class ConcreteTextRepository(TextArangoRepository):
    """
    Concrete implementation of TextArangoRepository that implements all required abstract methods.
    This class fulfills the UnifiedRepository interface requirements.
    """
    
    async def _execute_aql(self, query: str, bind_vars: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute an AQL query with the given bind variables.
        
        Args:
            query: The AQL query
            bind_vars: The bind variables for the query
            
        Returns:
            The results of the query
        """
        try:
            cursor = await self.connection._db.aql.execute(query, bind_vars=bind_vars)
            return [doc async for doc in cursor]
        except Exception as e:
            logger.error(f"Error executing AQL query: {e}")
            return []
    
    async def initialize(self, recreate: bool = False) -> bool:
        """
        Initialize the repository and set up required indexes.
        
        Args:
            recreate: Whether to recreate collections if they exist
            
        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        try:
            # Initialize collections
            node_collection = self.connection.get_collection(self.node_collection_name)
            edge_collection = self.connection.get_collection(self.edge_collection_name)
            
            await node_collection.create_if_not_exists(edge=False)
            await edge_collection.create_if_not_exists(edge=True)
            
            # Create indexes
            await node_collection.add_fulltext_index(["content"], name="content_fulltext")
            await node_collection.add_hash_index(["type"], name="node_type_index", unique=False)
            await edge_collection.add_hash_index(["type"], name="edge_type_index", unique=False)
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize repository: {e}")
            return False
    
    async def store_node(self, node_data: NodeData) -> bool:
        """
        Store a node in the repository.
        
        Args:
            node_data: Data for the node
            
        Returns:
            bool: True if the node was stored successfully, False otherwise
        """
        try:
            # Prepare the node document for Arango
            doc = dict(node_data)
            # If ID is used as key, set it
            if "id" in doc and "_key" not in doc:
                doc["_key"] = doc["id"]
                
            # Execute AQL query to insert the node
            aql = f"""
            INSERT @doc INTO {self.node_collection_name}
            RETURN NEW
            """
            
            params = {"doc": doc}
            result = await self._execute_aql(aql, params)
            return len(result) > 0
            
        except Exception as e:
            logger.error(f"Error storing node: {e}")
            return False
    
    async def store_edge(self, edge_data: EdgeData) -> bool:
        """
        Store an edge in the repository.
        
        Args:
            edge_data: Data for the edge
            
        Returns:
            bool: True if the edge was stored successfully, False otherwise
        """
        try:
            # Prepare the edge document for Arango
            doc = dict(edge_data)
            
            # Create from/to fields required by ArangoDB
            doc["_from"] = f"{self.node_collection_name}/{edge_data['source_id']}"
            doc["_to"] = f"{self.node_collection_name}/{edge_data['target_id']}"
            
            # Use id as key if provided
            if "id" in doc and "_key" not in doc:
                doc["_key"] = doc["id"]
            
            # Execute AQL query to insert the edge
            aql = f"""
            INSERT @doc INTO {self.edge_collection_name}
            RETURN NEW
            """
            
            params = {"doc": doc}
            result = await self._execute_aql(aql, params)
            return len(result) > 0
            
        except Exception as e:
            logger.error(f"Error storing edge: {e}")
            return False
    
    async def get_node(self, node_id: str) -> Optional[NodeData]:
        """
        Get a node by its ID.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Optional[NodeData]: The node data if found, None otherwise
        """
        try:
            aql = f"""
            FOR doc IN {self.node_collection_name}
            FILTER doc._key == @node_id
            RETURN doc
            """
            
            params = {"node_id": node_id}
            result = await self._execute_aql(aql, params)
            
            if not result:
                return None
                
            # Convert ArangoDB document to NodeData
            node_dict = dict(result[0])
            
            # Remove ArangoDB-specific fields
            if "_key" in node_dict:
                del node_dict["_key"]
            if "_id" in node_dict:
                del node_dict["_id"]
            if "_rev" in node_dict:
                del node_dict["_rev"]
                
            return cast(NodeData, node_dict)
            
        except Exception as e:
            logger.error(f"Error getting node: {e}")
            return None
    
    async def get_path(self, start_id: NodeID, end_id: NodeID, max_depth: int = 3, edge_types: Optional[List[str]] = None) -> List[List[Union[NodeData, EdgeData]]]:
        """
        Get the shortest path between two nodes.
        
        Args:
            start_id: The source node ID
            end_id: The target node ID
            max_depth: Maximum depth to traverse
            edge_types: Optional list of edge types to traverse
            
        Returns:
            List of paths, each containing nodes and edges
        """
        try:
            edge_filter = ""
            if edge_types and len(edge_types) > 0:
                edge_types_str = ", ".join([f"'{et}'" for et in edge_types])
                edge_filter = f"FILTER edge.type IN [{edge_types_str}]"
                
            aql = f"""
            FOR path IN OUTBOUND SHORTEST_PATH '{self.node_collection_name}/{start_id}' TO '{self.node_collection_name}/{end_id}' {self.edge_collection_name}
            {edge_filter}
            LIMIT {max_depth}
            RETURN path
            """
            
            result = await self._execute_aql(aql, {})
            if not result:
                return []
                
            # Convert the results to the expected format
            paths = []
            if result and len(result) > 0:
                path_data = result[0]  # Get the first path
                path_elements: List[Union[NodeData, EdgeData]] = []
                
                # Process vertices and edges in the path
                for item in path_data.get('vertices', []):
                    path_elements.append(cast(NodeData, item))
                    
                for item in path_data.get('edges', []):
                    # Convert edge format to EdgeData
                    edge_data: EdgeData = {
                        "id": item.get("_key", ""),
                        "source_id": item.get("_from", "").split("/")[-1],
                        "target_id": item.get("_to", "").split("/")[-1],
                        "type": item.get("type", ""),
                        "weight": item.get("weight", 1.0),
                        "bidirectional": item.get("bidirectional", False),
                        "metadata": item.get("metadata", {})
                    }
                    path_elements.append(edge_data)
                    
                paths.append(path_elements)
                
            return paths
            
        except Exception as e:
            logger.error(f"Error getting path: {e}")
            return []
    
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
            import numpy as np
            
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
                edges_to_create: List[EdgeData] = []
                
                # Find all pairs above threshold (excluding self-similarity)
                for j in range(len(batch)):
                    for k in range(j+1, len(batch)):  # Only upper triangle to avoid duplicates
                        similarity = float(similarities[j, k])
                        if similarity >= threshold:
                            # Create edge data compliant with EdgeData TypedDict
                            edge_data: EdgeData = {
                                "id": f"{batch_ids[j]}_{batch_ids[k]}_{edge_type}",
                                "source_id": batch_ids[j],
                                "target_id": batch_ids[k],
                                "type": edge_type,
                                "weight": similarity,
                                "bidirectional": True,
                                "metadata": {"similarity": similarity}
                            }
                            
                            # Store the edge
                            success = await self.store_edge(edge_data)
                            if success:
                                total_edges_created += 1
                
            return total_edges_created
                
        except Exception as e:
            logger.error(f"Error creating similarity edges: {e}")
            return total_edges_created
    
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
            node_dict = dict(node)
            
            # Update node metadata
            if "metadata" not in node_dict:
                node_dict["metadata"] = {}
                
            # Set embedding in metadata properly
            # Create a new metadata dict to avoid unsupported indexed assignment
            # Explicitly create a new dict to avoid type issues with dict constructor
            metadata = node_dict.get("metadata")
            if metadata is None:
                metadata_dict = {}
            else:
                # Ensure metadata is a dict before using items()
                if isinstance(metadata, dict):
                    metadata_dict = {k: v for k, v in metadata.items()}
                else:
                    # If not a dict, create a new empty dict
                    logger.warning(f"Expected dict for metadata but got {type(metadata).__name__}")
                    metadata_dict = {}
            if embedding_type == "default" or embedding_type == "":
                metadata_dict["embedding"] = embedding_doc
            else:
                metadata_dict[f"{embedding_type}_embedding"] = embedding_doc
                
            # Update the node dict with the new metadata
            node_dict["metadata"] = metadata_dict
            
            # Store updated node
            updated_node: NodeData = cast(NodeData, node_dict) 
            return await self.store_node(updated_node)
            
        except Exception as e:
            logger.error(f"Error storing embedding with type {embedding_type}: {e}")
            return False
    
    async def search_fulltext(self, query: str, limit: int = 10, node_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search for nodes using full-text search.
        
        Args:
            query: Text query to search for
            limit: Maximum number of results
            node_types: Optional list of node types to filter by
            
        Returns:
            List of matching nodes as NodeData
        """
        try:
            type_filter = ""
            if node_types and len(node_types) > 0:
                node_types_str = ", ".join([f"'{t}'" for t in node_types])
                type_filter = f"FILTER doc.type IN [{node_types_str}]"
                
            aql = f"""
            FOR doc IN FULLTEXT({self.node_collection_name}, 'content', @query)
            {type_filter}
            SORT BM25(doc) DESC
            LIMIT @limit
            RETURN doc
            """
            
            params = {
                "query": query,
                "limit": limit
            }
            
            result = await self._execute_aql(aql, params)
            # Return the results without unnecessary casting
            return result
            
        except Exception as e:
            logger.error(f"Error in fulltext search: {e}")
            return []
    
    async def search_similar_with_data(
        self, 
        vector: EmbeddingVector, 
        limit: int = 10,
        node_types: Optional[List[str]] = None,
        embedding_type: str = "default"
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for nodes similar to the given vector with their data.
        
        Args:
            vector: Query vector
            limit: Maximum number of results
            node_types: Optional list of node types to filter by
            embedding_type: Type of embedding to search (default, isne, etc.)
            
        Returns:
            List of tuples containing (node_data, similarity_score)
        """
        try:
            # Determine field name based on embedding type
            embedding_field = "metadata.embedding"
            if embedding_type != "default" and embedding_type != "":
                embedding_field = f"metadata.{embedding_type}_embedding"
                
            type_filter = ""
            if node_types and len(node_types) > 0:
                node_types_str = ", ".join([f"'{t}'" for t in node_types])
                type_filter = f"FILTER doc.type IN [{node_types_str}]"
                
            aql = f"""
            FOR doc IN {self.node_collection_name}
            {type_filter}
            FILTER HAS(doc, '{embedding_field}')
            LET similarity = COSINE_SIMILARITY(doc.{embedding_field}.vector, @vector)
            SORT similarity DESC
            LIMIT @limit
            RETURN [doc, similarity]
            """
            
            params = {
                "vector": vector,
                "limit": limit
            }
            
            result = await self._execute_aql(aql, params)
            # Return the results with the correct type as required by supertype
            # Use explicit casting to ensure proper typing for the return value
            return [(cast(Dict[str, Any], doc), float(score)) for doc, score in result]
            
        except Exception as e:
            logger.error(f"Error in vector similarity search: {e}")
            return []
    
    def hybrid_search(
        self,
        text_query: str,
        embedding: Optional[EmbeddingVector] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[Tuple[NodeData, float]]:
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
        try:
            # Use an async helper method to do the actual query
            return asyncio.run(self._hybrid_search_async(text_query, embedding, filters, limit))
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Return empty list with proper typing for List[Tuple[NodeData, float]]
            return cast(List[Tuple[NodeData, float]], [])
            
    async def _hybrid_search_async(
        self,
        text_query: str,
        embedding: Optional[EmbeddingVector] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[Tuple[NodeData, float]]:
        """
        Async helper method to perform hybrid search using both text and vector similarity.
        
        Args:
            text_query: The text search query
            embedding: Optional embedding for vector search
            filters: Optional filters to apply to the search
            limit: Maximum number of results to return
            
        Returns:
            List of nodes with combined relevance scores
        """
        try:
            # Prepare filters
            filter_clauses = []
            if filters:
                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_clauses.append(f"FILTER doc.{key} == '{value}'")
                    else:
                        filter_clauses.append(f"FILTER doc.{key} == {value}")
            
            filter_str = "\n".join(filter_clauses)
            
            # Handle vector component
            vector_component = ""
            if embedding is not None:
                vector_component = """
                LET vectorScore = doc.embedding != null && LENGTH(doc.embedding) == LENGTH(@vector) ? 
                    SQRT(1 - SUM(
                        FOR i IN 0..LENGTH(@vector)-1
                        RETURN POW(doc.embedding[i] - @vector[i], 2)
                    ) / (LENGTH(@vector) * 2))
                    : 0
                """
            else:
                vector_component = "LET vectorScore = 0"
                
            # Construct AQL query for hybrid search
            aql = f"""
            FOR doc IN {self.node_collection_name}
                FILTER doc.content LIKE @query OR doc.title LIKE @query
                {filter_str}
                {vector_component}
                LET textScore = doc.content LIKE @query ? 0.6 : (doc.title LIKE @query ? 0.8 : 0)
                LET combinedScore = textScore * 0.6 + vectorScore * 0.4
                SORT combinedScore DESC
                LIMIT @limit
                RETURN {{ document: doc, score: combinedScore }}
            """
                
            params = {
                "query": f"%{text_query}%",
                "vector": embedding,
                "limit": limit
            }
            
            result = await self._execute_aql(aql, params)
            # Return the results with proper unpacking of the array elements as NodeData and float
            # Use explicit typing to ensure compatibility with return type annotation
            return [(cast(NodeData, item["document"]), float(item["score"])) for item in result]
            
        except Exception as e:
            logger.error(f"Error in hybrid search async: {e}")
            # Return empty list with proper typing for List[Tuple[NodeData, float]]
            return cast(List[Tuple[NodeData, float]], [])
    
    async def get_connected_nodes(
        self,
        node_id: NodeID,
        edge_types: Optional[List[str]] = None,
        direction: str = "outbound"
    ) -> List[NodeData]:
        """
        Get nodes connected to the specified node.
        
        Args:
            node_id: ID of the node to start from
            edge_types: Optional list of edge types to filter by
            direction: Direction of traversal ('outbound', 'inbound', 'any')
            
        Returns:
            List of connected nodes
        """
        try:
            # Build edge filter if needed
            edge_filter = ""
            if edge_types and len(edge_types) > 0:
                edge_types_str = ", ".join([f"'{et}'" for et in edge_types])
                edge_filter = f"FILTER edge.type IN [{edge_types_str}]"
                
            aql = f"""
            FOR v, e IN 1..1 {direction} '{self.node_collection_name}/{node_id}' {self.edge_collection_name}
            {edge_filter}
            RETURN v
            """
            
            result = await self._execute_aql(aql, {})
            # Cast the result to the expected return type
            return [cast(NodeData, node) for node in result]
            
        except Exception as e:
            logger.error(f"Error getting connected nodes: {e}")
            return []

class TextStorageService:
    """
    Service for storing and retrieving text documents in ArangoDB.
    
    This service handles:
    1. Storing complete document data
    2. Storing document chunks with embeddings
    3. Creating relationships between documents and chunks
    4. Supporting vector search for semantic retrieval
    """
    
    def __init__(
        self, 
        connection: Optional[ArangoConnection] = None, 
        repository: Optional[ConcreteTextRepository] = None,
        similarity_threshold: float = 0.8
    ):
        """
        Initialize the Text storage service.
        
        Args:
            connection: Optional ArangoDB connection to use
            repository: Optional repository instance to use
            similarity_threshold: Threshold for creating similarity edges between chunks
        """
        if repository is not None:
            self.repository = repository
        elif connection is not None:
            # Create or get database, collections, and indexes using concrete implementation
            self.repository = ConcreteTextRepository(connection)
        else:
            # Default connection to localhost
            connection = ArangoConnection(db_name="hades")
            self.repository = ConcreteTextRepository(connection)
            
        self.similarity_threshold = similarity_threshold
        logger.info("Initialized Text storage service")
    
    async def store_processed_document(self, document_data: Dict[str, Any]) -> str:
        """
        Store a processed text document and its chunks in ArangoDB.
        
        Args:
            document_data: The processed document data from any document processing pipeline
            
        Returns:
            The ID of the stored document node
        """
        try:
            # Extract document metadata
            document_id = document_data.get("id", "")
            metadata = document_data.get("metadata", {})
            chunks = document_data.get("chunks", [])
            
            # Store the document node
            # Create a valid NodeData dictionary without extra keys
            document_node: NodeData = {
                "id": document_id,
                "type": "document",
                "content": "",  # Empty content as it's a parent node
                "source": metadata.get("source", ""),  # Required field
                "metadata": {
                    **metadata, 
                    "content_type": "pdf", 
                    "chunk_count": len(chunks),
                    "title": metadata.get("title", "Untitled Document"),
                    "created_at": datetime.now().isoformat()
                }
            }
            
            # Store the document node
            success = await self.repository.store_node(document_node)
            if not success:
                logger.error(f"Failed to store document node: {document_id}")
                raise RuntimeError(f"Failed to store document node: {document_id}")
                
            logger.info(f"Stored document node: {document_id}")
            
            # Store each chunk as a separate node
            chunk_node_ids = []
            for chunk in chunks:
                chunk_id = chunk.get("id", "")
                chunk_node_id = await self._store_chunk(chunk, document_id)
                chunk_node_ids.append(chunk_node_id)
            
            # Create edges between chunks based on embedding similarity
            if chunks and all(chunk.get("embedding") is not None for chunk in chunks):
                await self._create_similarity_edges(chunks)
            
            logger.info(f"Successfully stored document {document_id} with {len(chunks)} chunks")
            
            # Return explicit string type instead of Any
            return str(document_id)
            
        except Exception as e:
            logger.error(f"Error storing processed text document: {e}")
            raise
    
    async def _store_chunk(self, chunk_data: Dict[str, Any], document_id: str) -> str:
        """
        Store a document chunk in ArangoDB.
        
        Args:
            chunk_data: The chunk data
            document_id: The ID of the parent document
            
        Returns:
            The ID of the stored chunk node
        """
        chunk_id = chunk_data.get("id", "")
        content = chunk_data.get("content", "")
        embedding = chunk_data.get("embedding")
        
        # Create chunk node that complies with NodeData TypedDict structure
        chunk_node: NodeData = {
            "id": chunk_id,
            "type": "chunk",
            "content": content,
            "source": "",  # Required field in NodeData
            "metadata": {
                "parent_id": document_id,  # Move parent_id to metadata
                "symbol_type": chunk_data.get("symbol_type", ""),
                "chunk_index": chunk_data.get("chunk_index", 0),
                "token_count": chunk_data.get("token_count", 0),
                "content_hash": chunk_data.get("content_hash", ""),
                "isne_enhanced": embedding is not None and "isne_enhanced_embedding" in chunk_data
            }
        }
        
        # Store the chunk node
        success = await self.repository.store_node(chunk_node)
        if not success:
            logger.error(f"Failed to store chunk node: {chunk_id}")
            raise RuntimeError(f"Failed to store chunk node: {chunk_id}")
        
        # Create edge from document to chunk using valid EdgeData structure
        edge_data: EdgeData = {
            "id": f"{document_id}_{chunk_id}",  # Generate an edge ID
            "source_id": document_id,
            "target_id": chunk_id,
            "type": "contains",
            "weight": 1.0,  # Required field
            "bidirectional": False,  # Required field
            "metadata": {"index": chunk_data.get("chunk_index", 0)}  # Move index to metadata
        }
        
        success = await self.repository.store_edge(edge_data)
        if not success:
            logger.error(f"Failed to create edge from document {document_id} to chunk {chunk_id}")
        
        # Store the embedding if available
        if embedding is not None:
            # store_embedding is synchronous in the repository implementation
            success = self.repository.store_embedding(chunk_id, embedding)
            if not success:
                logger.warning(f"Failed to store embedding for chunk: {chunk_id}")
        
        # Store ISNE-enhanced embedding if available
        isne_embedding = chunk_data.get("isne_enhanced_embedding")
        if isne_embedding is not None:
            # Add a separate entry for ISNE-enhanced embedding
            await self.repository.store_embedding_with_type(
                chunk_id, 
                isne_embedding, 
                embedding_type="isne"
            )
        
        logger.debug(f"Stored chunk: {chunk_id}")
        # Return explicit string type instead of Any
        return str(chunk_id)
    
    async def _create_similarity_edges(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Create similarity edges between chunks based on embedding similarity.
        
        Args:
            chunks: List of chunk data with embeddings
        """
        try:
            # Use repository's bulk similarity calculation
            # Create properly typed tuples for chunk embeddings
            chunks_with_embeddings = [(NodeID(str(c["id"])), cast(EmbeddingVector, c.get("embedding"))) 
                                    for c in chunks if c.get("embedding") is not None]
            chunks_with_isne = [(NodeID(str(c["id"])), cast(EmbeddingVector, c.get("isne_enhanced_embedding"))) 
                             for c in chunks if c.get("isne_enhanced_embedding") is not None]
            if chunks_with_isne:
                logger.info(f"Creating similarity edges for {len(chunks_with_isne)} chunks with ISNE embeddings")
                # The chunks are already properly typed through the list comprehension
                await self.repository.create_similarity_edges(
                    chunks_with_isne,
                    edge_type="isne_similar_to",
                    threshold=self.similarity_threshold
                )
                
        except Exception as e:
            logger.error(f"Error creating similarity edges: {e}")
    
    async def search_by_content(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for document chunks by content.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of matching document chunks
        """
        return await self.repository.search_fulltext(query, limit=limit, node_types=["chunk"])
    
    async def search_by_vector(
        self, 
        query_vector: EmbeddingVector, 
        limit: int = 10,
        use_isne: bool = False
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for document chunks by vector similarity.
        
        Args:
            query_vector: The query embedding vector
            limit: Maximum number of results to return
            use_isne: Whether to search using ISNE-enhanced embeddings
            
        Returns:
            List of matching document chunks with similarity scores
        """
        embedding_type = "isne" if use_isne else "default"
        return await self.repository.search_similar_with_data(
            query_vector, 
            limit=limit, 
            node_types=["chunk"],
            embedding_type=embedding_type
        )
    
    async def hybrid_search(
        self,
        query: str,
        query_vector: Optional[EmbeddingVector] = None,
        limit: int = 10,
        use_isne: bool = False
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform a hybrid search using both text and vector similarity.
        
        Args:
            query: The text search query
            query_vector: Optional embedding for vector search
            limit: Maximum number of results to return
            use_isne: Whether to search using ISNE-enhanced embeddings
            
        Returns:
            List of matching document chunks
        """
        embedding_type = "isne" if use_isne else "default"
        # Convert parameters to the format expected by repository.hybrid_search
        filters = {
            "node_types": ["chunk"],
            "embedding_type": embedding_type
        }
        # Call the repository's hybrid_search method (Note: this is synchronous in ArangoRepository)
        results = self.repository.hybrid_search(
            query,
            embedding=query_vector,
            filters=filters,
            limit=limit
        )
        # Ensure proper return type
        return cast(List[Tuple[Dict[str, Any], float]], results)
    
    async def get_document(self, document_id: NodeID) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.
        
        Args:
            document_id: The document ID
            
        Returns:
            The document data if found, None otherwise
        """
        result = await self.repository.get_node(document_id)
        if result is None:
            return None
        # Explicitly cast NodeData to Dict[str, Any] for type compatibility
        return cast(Dict[str, Any], result)
    
    async def get_document_with_chunks(self, document_id: NodeID) -> Dict[str, Any]:
        """
        Get a document with all its chunks.
        
        Args:
            document_id: The document ID
            
        Returns:
            Dictionary containing document data and chunks
        """
        document = await self.repository.get_node(document_id)
        if not document:
            return {}
        
        # Get all chunks for this document using a traversal
        chunks = await self.repository.get_connected_nodes(
            document_id,
            edge_types=["contains"],
            direction="outbound"
        )
        
        result = {
            "document": document,
            "chunks": chunks
        }
        
        return result
