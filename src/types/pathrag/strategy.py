"""
PathRAG strategy types for HADES.

This module defines types specific to the PathRAG retrieval strategy.
"""

from typing import Dict, Any, Optional, List, Tuple, Set
from datetime import datetime
from pydantic import Field

from ..common import BaseSchema, EmbeddingVector, NodeID, EdgeID


class PathRAGConfig(BaseSchema):
    """Configuration for PathRAG strategy."""
    flow_decay_factor: float = 0.8
    pruning_threshold: float = 0.3
    max_path_length: int = 5
    max_iterations: int = 10
    
    # Resource allocation
    initial_resource: float = 1.0
    min_resource_threshold: float = 0.01
    
    # Path scoring
    semantic_weight: float = 0.7
    structural_weight: float = 0.2
    temporal_weight: float = 0.1
    
    # Retrieval modes
    enable_naive: bool = True
    enable_local: bool = True
    enable_global: bool = True
    enable_hybrid: bool = True
    enable_pathrag: bool = True
    
    # Performance
    max_nodes_to_explore: int = 1000
    early_stopping_threshold: float = 0.95
    cache_paths: bool = True
    

class PathNode(BaseSchema):
    """Node in a retrieval path."""
    node_id: NodeID
    content: str
    embedding: Optional[EmbeddingVector] = None
    score: float
    depth: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    

class RetrievalPath(BaseSchema):
    """A complete retrieval path."""
    path_id: str
    nodes: List[PathNode]
    edges: List[Tuple[NodeID, NodeID, float]]  # (source, target, weight)
    total_score: float
    path_type: str  # "direct", "multi_hop", "semantic", "structural"
    reliability_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    

class ResourceAllocation(BaseSchema):
    """Resource allocation for nodes during flow-based pruning."""
    node_resources: Dict[NodeID, float]
    iteration: int
    total_resource: float
    pruned_nodes: Set[NodeID] = Field(default_factory=set)
    flow_history: List[Dict[NodeID, float]] = Field(default_factory=list)
    

class PathRAGState(BaseSchema):
    """Internal state of PathRAG during retrieval."""
    query_node_id: NodeID
    query_embedding: EmbeddingVector
    explored_nodes: Set[NodeID] = Field(default_factory=set)
    candidate_paths: List[RetrievalPath] = Field(default_factory=list)
    resource_state: Optional[ResourceAllocation] = None
    iteration_count: int = 0
    start_time: datetime = Field(default_factory=lambda: datetime.utcnow())
    

class FlowUpdate(BaseSchema):
    """Update information for flow-based pruning."""
    from_node: NodeID
    to_node: NodeID
    flow_amount: float
    decay_applied: float
    iteration: int
    

class PathScore(BaseSchema):
    """Detailed scoring for a path."""
    semantic_score: float
    structural_score: float
    temporal_score: float
    reliability_score: float
    total_score: float
    score_components: Dict[str, float] = Field(default_factory=dict)
    

class QueryDecomposition(BaseSchema):
    """Decomposed query for hierarchical retrieval."""
    original_query: str
    high_level_keywords: List[str]
    low_level_keywords: List[str]
    entity_mentions: List[str]
    relationship_mentions: List[str]
    query_type: str  # "factual", "analytical", "navigational"
    

class RetrievalStats(BaseSchema):
    """Statistics from a retrieval operation."""
    total_nodes_explored: int
    total_edges_traversed: int
    paths_evaluated: int
    paths_pruned: int
    retrieval_time_ms: float
    mode_used: str
    cache_hits: int = 0
    iterations_completed: int = 0
    

class SupraWeightDimensions(BaseSchema):
    """Multi-dimensional weight representation."""
    semantic: float
    structural: float
    temporal: float
    hierarchical: float
    custom: Dict[str, float] = Field(default_factory=dict)
    
    def aggregate(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Aggregate dimensions into single weight."""
        if weights is None:
            weights = {
                "semantic": 0.4,
                "structural": 0.2,
                "temporal": 0.2,
                "hierarchical": 0.2
            }
        
        total = (
            self.semantic * weights.get("semantic", 0.4) +
            self.structural * weights.get("structural", 0.2) +
            self.temporal * weights.get("temporal", 0.2) +
            self.hierarchical * weights.get("hierarchical", 0.2)
        )
        
        # Add custom dimensions
        for key, value in self.custom.items():
            total += value * weights.get(key, 0.1)
            
        return total
    

class PathRAGResult(BaseSchema):
    """Complete result from PathRAG retrieval."""
    paths: List[RetrievalPath]
    nodes: List[PathNode]
    query_decomposition: QueryDecomposition
    stats: RetrievalStats
    metadata: Dict[str, Any] = Field(default_factory=dict)