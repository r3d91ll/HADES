"""
Text Storage Module for HADES-PathRAG.

This module provides integration between text processing pipelines and ArangoDB storage,
handling documents, chunks, embeddings, and relationships.
"""

from __future__ import annotations
import uuid
import logging
from datetime import datetime
from inspect import isawaitable
from typing import (Dict, List, Tuple, Optional, Any, Union, 
                   TypedDict, Protocol, runtime_checkable, Awaitable, cast,
                   Sequence)

import numpy as np
from numpy import ndarray

from src.types.basic import (
    NodeID, EmbeddingVector, BulkOperationResult, 
    VectorSearchResult, QueryFilter
)
from src.types.storage.query import (
    FilterOperator, VectorQueryOptions, HybridQueryOptions, 
    QueryOptions, HybridSearchResult
)
from src.types.storage.search import SearchResultEntry
from src.types.storage.document import (
    ExtendedDocumentMetadata, NodeData, EdgeData, 
    ExtendedNodeData, ExtendedEdgeData, DocumentMetadata,
    DocumentStorageResponse, DocumentContentType, DocumentID,
    DocumentStorageRequest, ExtendedDocumentStorageResponse
)
from src.types.storage.repository import (
    UnifiedRepository, DocumentRepository, GraphRepository, VectorRepository,
    BaseBulkOperationResult, CustomBulkOperationResult
)
from src.types.storage.connection import ConnectionInterface
from src.types.storage.embedding import (
    EmbeddingMetadata, EmbeddingIndexConfig, EmbeddingStorageOptions
)
from src.types.storage.query import (
    QueryResult,
    VectorSearchResult,
    HybridSearchResult,
    QueryOptions,
    VectorQueryOptions,
    HybridQueryOptions,
    QueryFilter,
    FilterCondition,
    FilterOperator
)

# Set up logger for this module
logger = logging.getLogger(__name__)

# Define type aliases for this module
StorageDocument = NodeData

# Import specific implementations
from src.storage.arango.connection import ArangoConnection
from src.storage.arango.repository import ArangoRepository
from src.storage.arango.text_repository import TextArangoRepository

logger = logging.getLogger(__name__)

class TextStorageService:
    """
    Service for storing and retrieving text documents, chunks, and embeddings.
    
    This service manages all aspects of text storage in ArangoDB, including:
    - Document and chunk storage
    - Relationships between documents and chunks
    - Vector embeddings for semantic search
    - Similarity-based relationships between chunks
    """
    
    def __init__(
        self,
        connection: Optional[Union[ArangoConnection, ConnectionInterface]] = None,
        repository: Optional[Union[TextArangoRepository, UnifiedRepository, Any]] = None,
        similarity_threshold: float = 0.8,
    ):
        """
        Initialize the Text storage service.
        
        Args:
            connection: Optional database connection to use
            repository: Optional repository instance to use (must implement DocumentRepository, GraphRepository, and VectorRepository)
            similarity_threshold: Threshold for creating similarity edges between chunks
        """
        if repository is not None:
            # Use provided repository if it implements all required interfaces
            if not (isinstance(repository, DocumentRepository) and 
                    isinstance(repository, GraphRepository) and 
                    isinstance(repository, VectorRepository)):
                raise TypeError("Repository must implement DocumentRepository, GraphRepository, and VectorRepository interfaces")
            self.repository = cast(UnifiedRepository, repository)
        elif connection is not None:
            # Create repository with provided connection - ensure it's an ArangoConnection
            if isinstance(connection, ArangoConnection):
                arango_connection = connection
            else:
                # If not an ArangoConnection but implements ConnectionInterface, create a new ArangoConnection
                # This is a temporary solution during migration - eventually all code should use the correct type
                logger.warning("Connection is not an ArangoConnection - creating new connection")
                arango_connection = ArangoConnection(db_name="hades")
                
            # Type ignore: TextArangoRepository should implement these methods but mypy doesn't see them
            self.repository = cast(UnifiedRepository, TextArangoRepository(arango_connection))  # type: ignore
        else:
            # Default connection to localhost
            arango_connection = ArangoConnection(db_name="hades")
            # Type ignore: TextArangoRepository should implement these methods but mypy doesn't see them
            self.repository = cast(UnifiedRepository, TextArangoRepository(arango_connection))  # type: ignore
            
        self.similarity_threshold = similarity_threshold
        logger.info("Initialized Text storage service")
    
    async def store_processed_document(self, document: DocumentStorageRequest) -> ExtendedDocumentStorageResponse:
        """
        Store a processed document in the storage service.
        
        Args:
            document: The document to store with all necessary metadata
            
        Returns:
            ExtendedDocumentStorageResponse with document_id and status
        """
        try:
            # Prepare document metadata
            metadata: Dict[str, Any] = {
                "title": document.get("metadata", {}).get("title", "Untitled Document"),
                "source": document.get("metadata", {}).get("source", "Unknown"),
                "content_type": document.get("metadata", {}).get("content_type", "text"),
                "created_at": document.get("metadata", {}).get("created_at", datetime.now().isoformat()),
                "processed_at": datetime.now().isoformat(),
            }
            
            # Store the parent document
            doc_data: ExtendedNodeData = {
                "_key": str(uuid.uuid4()),
                "content": document.get("content", ""),
                "metadata": metadata,
                "system_metadata": {
                    "chunk_count": len(document.get("chunks", [])),
                    "embedding_model": document.get("metadata", {}).get("embedding_model", None),
                    "content_hash": document.get("metadata", {}).get("content_hash", None),
                }
            }
            
            # Store the document and handle awaitable result
            parent_id_result = self.repository.store_document(cast(NodeData, doc_data))
            if isawaitable(parent_id_result):
                parent_id = await parent_id_result
            else:
                parent_id = parent_id_result
            
            logger.info(f"Stored document with ID: {parent_id}")
            
            # Return early if no chunks to process
            if not document.get("chunks", []):
                return {
                    "document_id": parent_id,
                    "status": "success",
                    "message": "Document stored without chunks",
                    "chunks_stored": 0,
                    "embeddings_stored": 0
                }
            
            # Store each chunk
            chunks_stored = 0
            embeddings_stored = 0
            
            for i, chunk in enumerate(document.get("chunks", [])):
                # Create extended metadata for chunk
                chunk_metadata: ExtendedDocumentMetadata = {
                    "title": metadata.get("title", "Untitled"),
                    "chunk_index": i,
                    "token_count": chunk.get("metadata", {}).get("token_count", 0),
                    "content_hash": chunk.get("metadata", {}).get("content_hash", ""),
                    "symbol_type": chunk.get("metadata", {}).get("symbol_type", "text"),
                    "isne_enhanced": chunk.get("metadata", {}).get("isne_enhanced", False),
                    "parent_id": parent_id,
                }
                
                chunk_id = await self._store_chunk(
                    parent_id=parent_id,
                    content=chunk.get("content", ""),
                    metadata=chunk_metadata,
                    embedding=chunk.get("embedding")
                )
                chunks_stored += 1
                if chunk.get("embedding") is not None:
                    embeddings_stored += 1
            
            return {
                "document_id": parent_id,
                "status": "success",
                "message": f"Document and {chunks_stored} chunks stored successfully",
                "chunks_stored": chunks_stored,
                "embeddings_stored": embeddings_stored
            }
            
        except Exception as e:
            logger.error(f"Error storing document: {str(e)}")
            return {
                "document_id": "",
                "status": "error",
                "message": f"Failed to store document: {str(e)}",
                "chunks_stored": 0,
                "embeddings_stored": 0
            }
    
    async def _store_chunk(self, parent_id: NodeID, content: str, metadata: ExtendedDocumentMetadata, embedding: Optional[EmbeddingVector] = None) -> NodeID:
        """
        Store a document chunk with its metadata and optional embedding.
        
        Args:
            parent_id: The ID of the parent document
            content: The content of the chunk
            metadata: The metadata for the chunk
            embedding: Optional embedding vector for the chunk
            
        Returns:
            The ID of the stored chunk node
        """
        try:
            # Add document type to metadata
            chunk_metadata = dict(metadata)
            chunk_metadata["content_type"] = "chunk"
            
            # Prepare chunk node data
            chunk_node: ExtendedNodeData = {
                "_key": str(uuid.uuid4()),
                "content": content,
                "metadata": cast(Dict[str, Any], chunk_metadata),  # Cast to avoid TypedDict type mismatch
                "system_metadata": {
                    "parent_id": parent_id,
                    "created_at": datetime.now().isoformat()
                }
            }
            
            # Store the chunk document
            chunk_id_result = self.repository.store_document(cast(NodeData, chunk_node))
            if isawaitable(chunk_id_result):
                chunk_id = await chunk_id_result
            else:
                chunk_id = chunk_id_result
            
            # Create edge between parent document and chunk
            edge: ExtendedEdgeData = {
                "from_id": parent_id,
                "to_id": chunk_id,
                "relation": "contains",
                "metadata": {
                    "relationship": "parent_child",
                    "created_at": datetime.now().isoformat()
                }
            }
            
            # Store edge
            store_edge_result = self.repository.store_edge(cast(EdgeData, edge))
            if isawaitable(store_edge_result):
                await store_edge_result
            
            # Store embedding if provided
            if embedding is not None:
                # Define embedding storage options
                options: Dict[str, Any] = {
                    "dimension": len(embedding),
                    "distance_metric": "cosine",
                    "return_values": True,
                    "include_fields": ["metadata", "content"],
                    "exclude_fields": []
                }
                
                # Store the embedding
                embed_result = self.repository.store_embedding(chunk_id, embedding, options)
                if isawaitable(embed_result):
                    success = await embed_result
                    logger.debug(f"Embedding storage success: {success}")
            
            logger.debug(f"Stored chunk {chunk_id} for document {parent_id}")
            return chunk_id
            
        except Exception as e:
            logger.error(f"Error storing chunk: {str(e)}")
            raise
    
    async def _create_similarity_edges(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Create similarity edges between chunks based on embedding similarity.
        
        Args:
            chunks: List of chunk data with embeddings
        """
        try:
            # Group chunks by embedding type
            # Cast the list comprehensions to ensure they produce the expected types
            chunks_with_embeddings = cast(
                List[Tuple[NodeID, Union[List[float], ndarray]]], 
                [(c["id"], c.get("embedding")) for c in chunks if c.get("embedding") is not None]
            )
            chunks_with_isne = cast(
                List[Tuple[NodeID, Union[List[float], ndarray]]],
                [(c["id"], c.get("isne_enhanced_embedding")) 
                 for c in chunks if c.get("isne_enhanced_embedding") is not None]
            )
            
            # Create bulk operation result holder
            result: BaseBulkOperationResult = {
                "success": True,
                "total": 0,
                "errors": []
            }
            
            # Use repository's bulk similarity calculation for regular embeddings
            if chunks_with_embeddings:
                logger.info(f"Creating similarity edges for {len(chunks_with_embeddings)} chunks with regular embeddings")
                # Pass only the node/embedding pairs, not extra parameters
                edge_result = self.repository.create_similarity_edges(chunks_with_embeddings)
                
                # Handle awaitable result
                if isawaitable(edge_result):
                    edge_result = await edge_result
                
                # Update bulk operation result
                if edge_result:
                    result["total"] += edge_result.get("total", len(chunks_with_embeddings))
                    if hasattr(edge_result, "errors") or isinstance(edge_result, dict) and "errors" in edge_result:
                        result["errors"].extend(edge_result.get("errors", []))
                        if edge_result.get("errors"):
                            result["success"] = False
            
            # Create similarity edges for ISNE-enhanced embeddings
            if chunks_with_isne:
                logger.info(f"Creating similarity edges for {len(chunks_with_isne)} chunks with ISNE-enhanced embeddings")
                isne_edge_result = self.repository.create_similarity_edges(chunks_with_isne)
                
                # Handle awaitable result
                if isawaitable(isne_edge_result):
                    isne_edge_result = await isne_edge_result
                
                # Update bulk operation result
                if isne_edge_result:
                    result["total"] += isne_edge_result.get("total", len(chunks_with_isne))
                    if hasattr(isne_edge_result, "errors") or isinstance(isne_edge_result, dict) and "errors" in isne_edge_result:
                        result["errors"].extend(isne_edge_result.get("errors", []))
                        if isne_edge_result.get("errors"):
                            result["success"] = False
            
            # Log similarity edge creation results
            logger.info(f"Created a total of {result['total']} similarity edges with {len(result['errors'])} errors")
            if result["errors"]:
                logger.warning(f"Errors during similarity edge creation: {result['errors']}")
                
        except Exception as e:
            logger.error(f"Error creating similarity edges: {str(e)}")
            raise
    
    async def search_by_vector(
        self, 
        query_vector: EmbeddingVector, 
        limit: int = 10,
        use_isne: bool = False,
        filters: Optional[List[QueryFilter]] = None
    ) -> VectorSearchResult:
        """
        Search for document chunks by vector similarity.
        
        Args:
            query_vector: The query embedding vector
            limit: Maximum number of results to return
            use_isne: Whether to search using ISNE-enhanced embeddings
            filters: Optional list of filters to apply to the search
            
        Returns:
            VectorSearchResult containing matching document chunks with similarity scores
        """
        # Set up default filters if none provided
        if filters is None:
            filters = [{"field": "type", "operator": FilterOperator.EQUALS, "value": "chunk"}]
            
        # Create vector query options
        vector_options = VectorQueryOptions(
            embedding=query_vector,
            top_k=limit,
            min_score=0.0,
            filter=filters[0] if filters else None,  # Use first filter as the query.py only accepts a single filter
            include_metadata=True,
            include_embeddings=False
        )
        
        # Use the repository's vector search interface
        result = self.repository.vector_search(vector_options)
        if isawaitable(result):
            return await result
        return cast(VectorSearchResult, result)
    
    async def hybrid_search(
        self,
        query: str,
        query_vector: Optional[EmbeddingVector] = None,
        limit: int = 10,
        use_isne: bool = False,
        filters: Optional[List[QueryFilter]] = None
    ) -> HybridSearchResult:
        """
        Perform a hybrid search using both text and vector similarity.
        
        Args:
            query: The text search query
            query_vector: Optional embedding for vector search
            limit: Maximum number of results to return
            use_isne: Whether to search using ISNE-enhanced embeddings
            filters: Optional list of filters to apply to the search
            
        Returns:
            HybridSearchResult containing matching document chunks with combined scores
        """
        # Set up default filters if none provided
        if filters is None:
            filters = [{"field": "type", "operator": FilterOperator.EQUALS, "value": "chunk"}]
            
        # Create hybrid query options
        hybrid_options = HybridQueryOptions(
            text_query=query,
            embedding=query_vector,
            top_k=limit,
            filter=filters[0] if filters else None,  # Use first filter as the query.py only accepts a single filter
            text_weight=0.3,  # Weight for text search component
            vector_weight=0.7  # Weight for vector search component
        )
        
        # Use the repository's hybrid search interface
        result = self.repository.hybrid_search(hybrid_options)
        if isawaitable(result):
            return await result
        return cast(HybridSearchResult, result)
    
    async def get_document(self, document_id: NodeID) -> Optional[StorageDocument]:
        """
        Get a document by ID.
        
        Args:
            document_id: The document ID
            
        Returns:
            The document data if found, None otherwise
        """
        result = self.repository.get_document(document_id)
        if isawaitable(result):
            return await result
        return cast(Optional[StorageDocument], result)
    
    async def get_document_with_chunks(self, document_id: NodeID) -> Dict[str, Any]:
        """
        Get a document with all its chunks.
        
        Args:
            document_id: The document ID
            
        Returns:
            Dictionary containing document data and chunks
        """
        # Get the document node
        node_result = self.repository.get_node(document_id)
        if isawaitable(node_result):
            document = await node_result
        else:
            document = cast(Optional[NodeData], node_result)
            
        if not document:
            return {}
        
        # Get all chunks for this document using a traversal
        connected_result = self.repository.get_connected_nodes(
            document_id,
            edge_type="contains"
        )
        
        if isawaitable(connected_result):
            chunks = await connected_result
        else:
            chunks = connected_result
        
        result = {
            "document": document,
            "chunks": chunks
        }
        
        return result
