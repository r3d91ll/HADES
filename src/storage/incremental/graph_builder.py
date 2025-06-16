"""
Incremental Graph Builder for HADES

Constructs knowledge graph incrementally by connecting new chunks
to existing graph structure while preserving performance.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import numpy as np
from scipy.spatial.distance import cosine

from arango.database import StandardDatabase

from .types import IncrementalConfig, ChunkInfo, NodeInfo, EdgeInfo

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds knowledge graph incrementally.
    
    Connects new chunks to existing graph structure by:
    1. Finding similar existing chunks
    2. Creating edges based on similarity thresholds
    3. Optimizing edge connections for performance
    """
    
    def __init__(self, db: StandardDatabase, config: IncrementalConfig):
        """
        Initialize graph builder.
        
        Args:
            db: ArangoDB database connection
            config: Incremental storage configuration
        """
        self.db = db
        self.config = config
        self.similarity_threshold = config.edge_similarity_threshold
        self.max_edges_per_node = config.max_edges_per_node
        
    async def build_incremental(
        self, 
        new_chunks: int, 
        updated_chunks: int
    ) -> Dict[str, Any]:
        """
        Build graph incrementally for new/updated chunks.
        
        Args:
            new_chunks: Number of new chunks to process
            updated_chunks: Number of updated chunks to process
            
        Returns:
            Graph building results with statistics
        """
        logger.info(f"Building graph incrementally: {new_chunks} new, {updated_chunks} updated chunks")
        
        start_time = datetime.now()
        
        try:
            # Get chunks that need graph connections
            pending_chunks = await self._get_pending_chunks(new_chunks + updated_chunks)
            
            if not pending_chunks:
                logger.info("No pending chunks found for graph building")
                return {'new_edges': 0, 'updated_edges': 0, 'total_edges': 0}
            
            # Get existing embeddings for similarity comparison
            existing_embeddings = await self._get_existing_embeddings()
            
            # Build edges for each pending chunk
            edge_results = await self._build_edges_batch(pending_chunks, existing_embeddings)
            
            # Update node metadata
            await self._update_node_metadata(pending_chunks)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            results = {
                'new_edges': edge_results['created_edges'],
                'updated_edges': edge_results['updated_edges'],
                'total_edges': edge_results['total_edges'],
                'processing_time': processing_time,
                'chunks_processed': len(pending_chunks)
            }
            
            logger.info(f"Graph building completed: {edge_results['created_edges']} new edges created")
            return results
            
        except Exception as e:
            logger.error(f"Graph building failed: {e}")
            raise
    
    async def _get_pending_chunks(self, limit: int) -> List[ChunkInfo]:
        """
        Get chunks that need graph connections.
        
        Args:
            limit: Maximum number of chunks to retrieve
            
        Returns:
            List of chunk information
        """
        # Query for chunks without graph nodes or with outdated connections
        query = """
        FOR chunk IN chunks
        LEFT JOIN nodes ON nodes.chunk_id == chunk._key
        FILTER nodes == null OR nodes.created_at < chunk.created_at
        LIMIT @limit
        RETURN {
            chunk_id: chunk._key,
            document_id: chunk.document_id,
            content: chunk.content,
            content_hash: chunk.content_hash,
            embedding_id: chunk.embedding_id,
            metadata: chunk.metadata
        }
        """
        
        results = self.db.query(query, bind_vars={'limit': limit})
        
        chunks = []
        for result in results:
            chunk_info = ChunkInfo(
                chunk_id=result['chunk_id'],
                document_id=result['document_id'],
                content=result['content'],
                content_hash=result['content_hash'],
                start_pos=0,  # Would be filled from actual data
                end_pos=len(result['content']),
                metadata=result['metadata'],
                embedding=None  # Will be loaded separately
            )
            chunks.append(chunk_info)
        
        logger.info(f"Found {len(chunks)} pending chunks for graph building")
        return chunks
    
    async def _get_existing_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Get existing embeddings for similarity comparison.
        
        Returns:
            Dictionary mapping embedding IDs to vectors
        """
        # Limit to recent embeddings for performance
        query = """
        FOR embedding IN embeddings
        SORT embedding.created_at DESC
        LIMIT 10000
        RETURN {
            embedding_id: embedding._key,
            vector: embedding.vector
        }
        """
        
        results = self.db.query(query)
        
        embeddings = {}
        for result in results:
            embedding_id = result['embedding_id']
            vector = np.array(result['vector'])
            embeddings[embedding_id] = vector
        
        logger.info(f"Loaded {len(embeddings)} existing embeddings for comparison")
        return embeddings
    
    async def _build_edges_batch(
        self, 
        chunks: List[ChunkInfo], 
        existing_embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, int]:
        """
        Build edges for a batch of chunks.
        
        Args:
            chunks: Chunks to process
            existing_embeddings: Existing embedding vectors
            
        Returns:
            Edge creation statistics
        """
        created_edges = 0
        updated_edges = 0
        
        for chunk in chunks:
            try:
                # Get chunk's embedding
                chunk_embedding = await self._get_chunk_embedding(chunk.chunk_id)
                if chunk_embedding is None:
                    continue
                
                # Find similar chunks
                similar_chunks = self._find_similar_chunks(
                    chunk_embedding, 
                    existing_embeddings,
                    chunk.chunk_id
                )
                
                # Create edges to similar chunks
                edges_created = await self._create_edges_for_chunk(
                    chunk.chunk_id, 
                    similar_chunks
                )
                
                created_edges += edges_created
                
            except Exception as e:
                logger.warning(f"Failed to build edges for chunk {chunk.chunk_id}: {e}")
        
        # Get total edge count
        total_edges = await self._get_total_edge_count()
        
        return {
            'created_edges': created_edges,
            'updated_edges': updated_edges,
            'total_edges': total_edges
        }
    
    async def _get_chunk_embedding(self, chunk_id: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for a chunk.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Embedding vector or None if not found
        """
        query = """
        FOR chunk IN chunks
        FILTER chunk._key == @chunk_id
        FOR embedding IN embeddings
        FILTER embedding.chunk_id == chunk._key
        RETURN embedding.vector
        """
        
        results = list(self.db.query(query, bind_vars={'chunk_id': chunk_id}))
        
        if results:
            return np.array(results[0])
        return None
    
    def _find_similar_chunks(
        self,
        query_embedding: np.ndarray,
        existing_embeddings: Dict[str, np.ndarray],
        exclude_chunk: str
    ) -> List[Tuple[str, float]]:
        """
        Find chunks similar to the query embedding.
        
        Args:
            query_embedding: Query embedding vector
            existing_embeddings: Dictionary of existing embeddings
            exclude_chunk: Chunk ID to exclude from results
            
        Returns:
            List of (chunk_id, similarity_score) tuples
        """
        similarities = []
        
        for embedding_id, embedding_vector in existing_embeddings.items():
            if embedding_id == exclude_chunk:
                continue
            
            try:
                # Calculate cosine similarity
                similarity = 1 - cosine(query_embedding, embedding_vector)
                
                # Filter by threshold
                if similarity >= self.similarity_threshold:
                    similarities.append((embedding_id, similarity))
                    
            except Exception as e:
                logger.debug(f"Failed to calculate similarity for {embedding_id}: {e}")
        
        # Sort by similarity and limit results
        similarities.sort(key=lambda x: x[1], reverse=True)
        similarities = similarities[:self.max_edges_per_node]
        
        return similarities
    
    async def _create_edges_for_chunk(
        self, 
        chunk_id: str, 
        similar_chunks: List[Tuple[str, float]]
    ) -> int:
        """
        Create edges between chunk and similar chunks.
        
        Args:
            chunk_id: Source chunk ID
            similar_chunks: List of (target_chunk_id, similarity) tuples
            
        Returns:
            Number of edges created
        """
        if not similar_chunks:
            return 0
        
        # Get node IDs for chunks
        source_node = await self._get_or_create_node(chunk_id)
        if not source_node:
            return 0
        
        edges_created = 0
        
        for target_chunk_id, similarity in similar_chunks:
            target_node = await self._get_or_create_node(target_chunk_id)
            if not target_node:
                continue
            
            # Create edge
            edge_created = await self._create_edge(
                source_node, 
                target_node, 
                similarity,
                'similarity'
            )
            
            if edge_created:
                edges_created += 1
        
        return edges_created
    
    async def _get_or_create_node(self, chunk_id: str) -> Optional[str]:
        """
        Get existing node or create new one for chunk.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Node ID or None if creation failed
        """
        nodes_collection = self.db.collection('nodes')
        
        # Check if node exists
        query = """
        FOR node IN nodes
        FILTER node.chunk_id == @chunk_id
        RETURN node._key
        """
        
        results = list(self.db.query(query, bind_vars={'chunk_id': chunk_id}))
        
        if results:
            return results[0]
        
        # Create new node
        try:
            # Get next node index
            node_index = await self._get_next_node_index()
            
            node_doc = {
                'chunk_id': chunk_id,
                'embedding_id': chunk_id,  # Assume same ID for now
                'node_index': node_index,
                'created_at': datetime.now().isoformat(),
                'isne_enhanced': False,
                'metadata': {}
            }
            
            result = nodes_collection.insert(node_doc)
            return result['_key']
            
        except Exception as e:
            logger.error(f"Failed to create node for chunk {chunk_id}: {e}")
            return None
    
    async def _get_next_node_index(self) -> int:
        """
        Get the next available node index.
        
        Returns:
            Next node index
        """
        query = """
        FOR node IN nodes
        SORT node.node_index DESC
        LIMIT 1
        RETURN node.node_index
        """
        
        results = list(self.db.query(query))
        
        if results:
            return results[0] + 1
        return 0
    
    async def _create_edge(
        self,
        source_node: str,
        target_node: str,
        weight: float,
        edge_type: str
    ) -> bool:
        """
        Create an edge between two nodes.
        
        Args:
            source_node: Source node ID
            target_node: Target node ID
            weight: Edge weight
            edge_type: Type of edge
            
        Returns:
            True if edge created successfully
        """
        edges_collection = self.db.collection('edges')
        
        # Check if edge already exists
        query = """
        FOR edge IN edges
        FILTER edge._from == @from_id AND edge._to == @to_id
        RETURN edge._key
        """
        
        from_id = f"nodes/{source_node}"
        to_id = f"nodes/{target_node}"
        
        existing = list(self.db.query(query, bind_vars={
            'from_id': from_id,
            'to_id': to_id
        }))
        
        if existing:
            # Update existing edge
            try:
                edges_collection.update(existing[0], {
                    'weight': weight,
                    'updated_at': datetime.now().isoformat()
                })
                return True
            except Exception as e:
                logger.warning(f"Failed to update edge {existing[0]}: {e}")
                return False
        
        # Create new edge
        try:
            edge_doc = {
                '_from': from_id,
                '_to': to_id,
                'weight': weight,
                'edge_type': edge_type,
                'created_at': datetime.now().isoformat(),
                'metadata': {}
            }
            
            edges_collection.insert(edge_doc)
            return True
            
        except Exception as e:
            logger.warning(f"Failed to create edge {from_id} -> {to_id}: {e}")
            return False
    
    async def _get_total_edge_count(self) -> int:
        """
        Get total number of edges in the graph.
        
        Returns:
            Total edge count
        """
        edges_collection = self.db.collection('edges')
        return edges_collection.count()
    
    async def _update_node_metadata(self, chunks: List[ChunkInfo]) -> None:
        """
        Update metadata for processed nodes.
        
        Args:
            chunks: Processed chunks
        """
        nodes_collection = self.db.collection('nodes')
        
        for chunk in chunks:
            try:
                # Update node to mark as processed
                query = """
                FOR node IN nodes
                FILTER node.chunk_id == @chunk_id
                UPDATE node WITH {
                    last_updated: @timestamp,
                    graph_connected: true
                } IN nodes
                """
                
                self.db.query(query, bind_vars={
                    'chunk_id': chunk.chunk_id,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.warning(f"Failed to update node metadata for chunk {chunk.chunk_id}: {e}")
    
    async def optimize_graph(self) -> Dict[str, Any]:
        """
        Optimize graph structure for performance.
        
        Returns:
            Optimization results
        """
        logger.info("Starting graph optimization")
        
        # Remove low-weight edges
        removed_edges = await self._remove_low_weight_edges()
        
        # Balance node connectivity
        rebalanced_nodes = await self._rebalance_node_connectivity()
        
        # Update graph statistics
        stats = await self._calculate_graph_statistics()
        
        return {
            'removed_edges': removed_edges,
            'rebalanced_nodes': rebalanced_nodes,
            'graph_statistics': stats
        }
    
    async def _remove_low_weight_edges(self) -> int:
        """
        Remove edges with very low weights.
        
        Returns:
            Number of edges removed
        """
        min_weight = self.similarity_threshold * 0.5  # Half of similarity threshold
        
        query = """
        FOR edge IN edges
        FILTER edge.weight < @min_weight
        REMOVE edge IN edges
        RETURN OLD
        """
        
        results = list(self.db.query(query, bind_vars={'min_weight': min_weight}))
        
        logger.info(f"Removed {len(results)} low-weight edges (< {min_weight})")
        return len(results)
    
    async def _rebalance_node_connectivity(self) -> int:
        """
        Rebalance nodes with too many connections.
        
        Returns:
            Number of nodes rebalanced
        """
        # Find nodes with excessive connections
        query = """
        FOR node IN nodes
        LET outbound_count = LENGTH(
            FOR edge IN edges
            FILTER edge._from == CONCAT('nodes/', node._key)
            RETURN edge
        )
        FILTER outbound_count > @max_edges
        RETURN {
            node_id: node._key,
            edge_count: outbound_count
        }
        """
        
        overconnected = list(self.db.query(query, bind_vars={
            'max_edges': self.max_edges_per_node * 1.5
        }))
        
        rebalanced = 0
        
        for node_info in overconnected:
            # Remove lowest-weight edges for this node
            removed = await self._trim_node_edges(
                node_info['node_id'], 
                self.max_edges_per_node
            )
            
            if removed > 0:
                rebalanced += 1
        
        logger.info(f"Rebalanced {rebalanced} overconnected nodes")
        return rebalanced
    
    async def _trim_node_edges(self, node_id: str, max_edges: int) -> int:
        """
        Trim edges for a node to maximum count.
        
        Args:
            node_id: Node identifier
            max_edges: Maximum number of edges to keep
            
        Returns:
            Number of edges removed
        """
        # Get edges sorted by weight (ascending)
        query = """
        FOR edge IN edges
        FILTER edge._from == CONCAT('nodes/', @node_id)
        SORT edge.weight ASC
        RETURN edge._key
        """
        
        edges = list(self.db.query(query, bind_vars={'node_id': node_id}))
        
        if len(edges) <= max_edges:
            return 0
        
        # Remove lowest-weight edges
        edges_to_remove = edges[:-max_edges]  # Keep highest-weight edges
        
        edges_collection = self.db.collection('edges')
        removed = 0
        
        for edge_key in edges_to_remove:
            try:
                edges_collection.delete(edge_key)
                removed += 1
            except Exception as e:
                logger.warning(f"Failed to remove edge {edge_key}: {e}")
        
        return removed
    
    async def _calculate_graph_statistics(self) -> Dict[str, Any]:
        """
        Calculate graph statistics.
        
        Returns:
            Dictionary with graph statistics
        """
        # Get basic counts
        nodes_count = self.db.collection('nodes').count()
        edges_count = self.db.collection('edges').count()
        
        # Calculate density
        max_possible_edges = nodes_count * (nodes_count - 1)
        density = edges_count / max_possible_edges if max_possible_edges > 0 else 0
        
        # Average degree
        avg_degree = (2 * edges_count) / nodes_count if nodes_count > 0 else 0
        
        return {
            'nodes': nodes_count,
            'edges': edges_count,
            'density': density,
            'average_degree': avg_degree
        }