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

logger = logging.getLogger(__name__)

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
        repository: Optional[ArangoRepository] = None,
        similarity_threshold: float = 0.8,
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
            self.repository = TextArangoRepository(connection)
        else:
            # Default connection to localhost
            connection = ArangoConnection(db_name="hades")
            self.repository = TextArangoRepository(connection)
            
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
            document_node: NodeData = {
                "_key": document_id,
                "id": document_id,
                "type": "document",
                "content_type": "pdf",
                "metadata": metadata,
                "title": metadata.get("title", "Untitled Document"),
                "source": metadata.get("source", ""),
                "created_at": datetime.now().isoformat(),
                "chunk_count": len(chunks)
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
            
            return document_id
            
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
        
        # Create chunk node
        chunk_node: NodeData = {
            "_key": chunk_id,
            "id": chunk_id,
            "type": "chunk",
            "content": content,
            "parent_id": document_id,
            "metadata": {
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
        
        # Create edge from document to chunk
        edge_data: EdgeData = {
            "_from": f"nodes/{document_id}",
            "_to": f"nodes/{chunk_id}",
            "type": "contains",
            "index": chunk_data.get("chunk_index", 0)
        }
        
        success = await self.repository.store_edge(edge_data)
        if not success:
            logger.error(f"Failed to create edge from document {document_id} to chunk {chunk_id}")
        
        # Store the embedding if available
        if embedding is not None:
            success = await self.repository.store_embedding(chunk_id, embedding)
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
        return chunk_id
    
    async def _create_similarity_edges(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Create similarity edges between chunks based on embedding similarity.
        
        Args:
            chunks: List of chunk data with embeddings
        """
        try:
            # Group chunks by embedding type
            chunks_with_embeddings = [(c["id"], c.get("embedding")) for c in chunks if c.get("embedding") is not None]
            chunks_with_isne = [(c["id"], c.get("isne_enhanced_embedding")) 
                             for c in chunks if c.get("isne_enhanced_embedding") is not None]
            
            # Use repository's bulk similarity calculation
            if chunks_with_embeddings:
                logger.info(f"Creating similarity edges for {len(chunks_with_embeddings)} chunks with regular embeddings")
                await self.repository.create_similarity_edges(
                    chunks_with_embeddings,
                    edge_type="similar_to",
                    threshold=self.similarity_threshold
                )
            
            # Do the same for ISNE-enhanced embeddings if available
            if chunks_with_isne:
                logger.info(f"Creating similarity edges for {len(chunks_with_isne)} chunks with ISNE embeddings")
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
    ) -> List[Dict[str, Any]]:
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
        return await self.repository.hybrid_search(
            query,
            query_vector,
            limit=limit,
            node_types=["chunk"],
            embedding_type=embedding_type
        )
    
    async def get_document(self, document_id: NodeID) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.
        
        Args:
            document_id: The document ID
            
        Returns:
            The document data if found, None otherwise
        """
        return await self.repository.get_node(document_id)
    
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
