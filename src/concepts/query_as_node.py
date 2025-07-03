"""
Query as Node - ANT-inspired Query Processing

This module implements the concept of treating queries as first-class citizens
in the knowledge graph, based on Actor-Network Theory (ANT) principles.

Key Ideas:
- Queries have agency equal to documents in the network
- Temporary nodes are created for each query
- Multiple ranked paths provide different interpretations
- The network reconfigures around each query
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import numpy as np
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class QueryNode:
    """
    Represents a query as a temporary node in the knowledge graph.
    
    Following ANT principles, this query node has equal agency to any
    other node in the network and can form relationships dynamically.
    """
    id: str
    text: str
    embedding: Optional[np.ndarray] = None
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    paths: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Ensure we have a unique ID."""
        if not self.id:
            self.id = f"query_{self.created_at.timestamp()}_{uuid.uuid4().hex[:8]}"


class QueryAsNode:
    """
    Implements Query-as-Node processing for ANT-based knowledge retrieval.
    
    This class transforms queries into temporary graph nodes, allowing them
    to participate in the knowledge network with full agency.
    """
    
    def __init__(self, graph_storage: Any, embedding_generator: Any):
        """
        Initialize Query-as-Node processor.
        
        Args:
            graph_storage: Graph storage backend (e.g., ArangoDB)
            embedding_generator: Embedding generator (e.g., Jina v4)
        """
        self.graph = graph_storage
        self.embedder = embedding_generator
        
        # Configuration
        self.max_paths = 10
        self.max_path_length = 5
        self.relationship_threshold = 0.7
        
        logger.info("Initialized Query-as-Node processor")
    
    async def process_query(self, 
                          query_text: str, 
                          user_context: Optional[Dict[str, Any]] = None) -> QueryNode:
        """
        Process a query by creating it as a temporary node in the graph.
        
        Args:
            query_text: The query text
            user_context: Optional context about the user/query
            
        Returns:
            QueryNode with discovered relationships and paths
        """
        # Create query node
        query_node = QueryNode(
            id=f"query_{datetime.now().timestamp()}_{uuid.uuid4().hex[:8]}",
            text=query_text,
            context=user_context or {}
        )
        
        logger.info(f"Processing query as node: {query_node.id}")
        
        # Generate embedding for query
        query_node.embedding = await self._generate_query_embedding(query_text)
        
        # Place query in network and discover relationships
        await self._place_in_network(query_node)
        
        # Find paths from query node
        query_node.paths = await self._discover_paths(query_node)
        
        # Rank paths by multiple criteria
        ranked_paths = self._rank_paths(query_node.paths, query_node.context)
        query_node.paths = ranked_paths
        
        logger.info(f"Query processing complete: {len(query_node.paths)} paths found")
        
        return query_node
    
    async def _generate_query_embedding(self, query_text: str) -> np.ndarray:
        """Generate embedding for query text."""
        # Use Jina v4 to generate query embedding
        result = await self.embedder.embed(
            text=query_text,
            task_type='query'  # Specific task type for queries
        )
        return np.array(result['embedding'])
    
    async def _place_in_network(self, query_node: QueryNode) -> None:
        """
        Place query node in network and discover relationships.
        
        This is where ANT principles shine - the query actively
        creates relationships with existing nodes.
        """
        # Find semantic neighbors
        neighbors = await self.graph.find_neighbors(
            embedding=query_node.embedding,
            max_neighbors=50,
            similarity_threshold=self.relationship_threshold
        )
        
        # Create temporary relationships
        for neighbor in neighbors:
            relationship = self._analyze_relationship(query_node, neighbor)
            if relationship:
                query_node.relationships.append(relationship)
                
                # Temporarily add to graph (with TTL)
                await self.graph.add_temporary_edge(
                    from_node=query_node.id,
                    to_node=neighbor['id'],
                    relationship=relationship,
                    ttl_seconds=3600  # 1 hour TTL
                )
    
    def _analyze_relationship(self, 
                            query_node: QueryNode, 
                            neighbor: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze the relationship between query and neighbor.
        
        Returns relationship metadata including type and strength.
        """
        # Calculate semantic similarity
        neighbor_embedding = neighbor.get('embedding')
        if neighbor_embedding is None:
            return None
            
        if query_node.embedding is None:
            return None
            
        similarity = self._calculate_similarity(
            query_node.embedding, 
            neighbor_embedding
        )
        
        if similarity < self.relationship_threshold:
            return None
        
        # Determine relationship type based on neighbor metadata
        rel_type = self._determine_relationship_type(query_node, neighbor)
        
        return {
            'target': neighbor['id'],
            'type': rel_type,
            'strength': similarity,
            'metadata': {
                'neighbor_type': neighbor.get('type', 'unknown'),
                'created_at': datetime.now(timezone.utc).isoformat()
            }
        }
    
    def _determine_relationship_type(self, 
                                   query_node: QueryNode, 
                                   neighbor: Dict[str, Any]) -> str:
        """Determine the type of relationship between query and neighbor."""
        neighbor_type = neighbor.get('type', 'unknown')
        
        # Map neighbor types to relationship types
        if neighbor_type == 'theory':
            return 'seeks_theory'
        elif neighbor_type == 'implementation':
            return 'seeks_implementation'
        elif neighbor_type == 'documentation':
            return 'seeks_explanation'
        else:
            return 'semantic_similarity'
    
    async def _discover_paths(self, query_node: QueryNode) -> List[Dict[str, Any]]:
        """
        Discover paths from the query node through the network.
        
        This implements the multi-path discovery that gives researchers
        multiple perspectives on their query.
        """
        paths = []
        
        # Use graph traversal to find paths
        for relationship in query_node.relationships[:10]:  # Start with top relationships
            paths_from_neighbor = await self.graph.find_paths(
                start_node=relationship['target'],
                max_length=self.max_path_length - 1,
                max_paths=3
            )
            
            for path in paths_from_neighbor:
                # Prepend query node to path
                full_path = {
                    'nodes': [query_node.id] + path['nodes'],
                    'edges': [relationship] + path['edges'],
                    'total_weight': relationship['strength'] * path.get('weight', 1.0),
                    'interpretation': self._interpret_path(relationship, path)
                }
                paths.append(full_path)
        
        return paths
    
    def _interpret_path(self, 
                       initial_relationship: Dict[str, Any], 
                       path: Dict[str, Any]) -> str:
        """
        Generate human-readable interpretation of a path.
        
        This helps researchers understand what each path represents.
        """
        rel_type = initial_relationship['type']
        path_types = [edge.get('type', 'unknown') for edge in path.get('edges', [])]
        
        if rel_type == 'seeks_theory' and 'implementation' in path_types:
            return "Theoretical foundation with practical implementation"
        elif rel_type == 'seeks_implementation' and 'theory' in path_types:
            return "Implementation grounded in theoretical research"
        elif 'documentation' in path_types:
            return "Explained through documentation and examples"
        else:
            return "Related through semantic connections"
    
    def _rank_paths(self, 
                   paths: List[Dict[str, Any]], 
                   context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rank paths by multiple criteria.
        
        This gives researchers agency in choosing their perspective.
        """
        for path in paths:
            # Calculate multiple scores
            scores = {
                'semantic_coherence': self._calculate_coherence(path),
                'theory_practice_bridge': self._calculate_bridge_score(path),
                'path_diversity': self._calculate_diversity(path),
                'context_alignment': self._calculate_context_alignment(path, context)
            }
            
            # Weighted combination (weights could be user-configurable)
            path['scores'] = scores
            path['total_score'] = (
                0.3 * scores['semantic_coherence'] +
                0.3 * scores['theory_practice_bridge'] +
                0.2 * scores['path_diversity'] +
                0.2 * scores['context_alignment']
            )
        
        # Sort by total score
        return sorted(paths, key=lambda p: p['total_score'], reverse=True)
    
    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings."""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def _calculate_coherence(self, path: Dict[str, Any]) -> float:
        """Calculate semantic coherence of a path."""
        # Placeholder - would calculate based on edge weights
        edges = path.get('edges', [])
        if not edges:
            return 0.0
        
        strengths = [e.get('strength', 0.5) for e in edges]
        return float(np.mean(strengths))
    
    def _calculate_bridge_score(self, path: Dict[str, Any]) -> float:
        """Calculate theory-practice bridge score."""
        edge_types = [e.get('type', '') for e in path.get('edges', [])]
        
        # High score if path bridges theory and practice
        has_theory = any('theory' in t for t in edge_types)
        has_practice = any('implementation' in t or 'practice' in t for t in edge_types)
        
        return 1.0 if (has_theory and has_practice) else 0.3
    
    def _calculate_diversity(self, path: Dict[str, Any]) -> float:
        """Calculate diversity of node types in path."""
        # Placeholder - would look at variety of node types
        return 0.5
    
    def _calculate_context_alignment(self, 
                                   path: Dict[str, Any], 
                                   context: Dict[str, Any]) -> float:
        """Calculate alignment with user context."""
        # Placeholder - would match path characteristics with user preferences
        return 0.5