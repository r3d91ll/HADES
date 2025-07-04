"""
ISNE model types for HADES.

This module defines types specific to the ISNE (Inductive Shallow Node Embedding) models.
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pydantic import Field

from ..common import BaseSchema, EmbeddingVector, NodeID


class ISNEConfig(BaseSchema):
    """Configuration for ISNE models."""
    embedding_dim: int = 768
    hidden_dim: int = 512
    hidden_dims: Optional[List[int]] = None  # For multi-layer architectures
    num_layers: int = 3
    dropout: float = 0.1
    learning_rate: float = 0.001
    weight_decay: float = 0.0001  # L2 regularization
    batch_size: int = 32
    num_epochs: int = 100
    device: str = "cuda"
    use_batch_norm: bool = True
    use_residual: bool = True
    aggregation_method: str = "mean"  # "mean", "sum", "max", "attention"
    distance_metric: str = "cosine"  # "cosine", "euclidean", "manhattan"
    

class DirectoryMetadata(BaseSchema):
    """Metadata for directory-aware ISNE training."""
    directory_path: str
    depth: int
    parent_path: Optional[str] = None
    num_files: int = 0
    num_subdirs: int = 0
    total_size_bytes: Optional[int] = None
    keywords: List[str] = Field(default_factory=list)
    importance_score: float = 1.0
    

class ISNENode(BaseSchema):
    """Node representation for ISNE."""
    node_id: NodeID
    initial_embedding: EmbeddingVector
    enhanced_embedding: Optional[EmbeddingVector] = None
    node_type: str = "document"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    neighbors: List[NodeID] = Field(default_factory=list)
    edge_weights: Dict[NodeID, float] = Field(default_factory=dict)
    

class ISNEEdge(BaseSchema):
    """Edge representation for ISNE."""
    source: NodeID
    target: NodeID
    weight: float = 1.0
    edge_type: str = "similarity"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    

class ISNETrainingData(BaseSchema):
    """Training data for ISNE models."""
    nodes: List[ISNENode]
    edges: List[ISNEEdge]
    directory_metadata: Optional[List[DirectoryMetadata]] = None
    global_metadata: Dict[str, Any] = Field(default_factory=dict)
    

class ISNEModelState(BaseSchema):
    """State of a trained ISNE model."""
    model_version: str
    config: ISNEConfig
    training_metrics: Dict[str, List[float]] = Field(default_factory=dict)
    best_epoch: int = 0
    best_loss: float = float('inf')
    training_time_seconds: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())
    checkpoint_path: Optional[str] = None
    

class ISNEInferenceResult(BaseSchema):
    """Result from ISNE inference."""
    node_id: NodeID
    original_embedding: EmbeddingVector
    enhanced_embedding: EmbeddingVector
    enhancement_score: float  # Measure of how much the embedding changed
    inference_time_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    

class GraphAggregationConfig(BaseSchema):
    """Configuration for graph-based aggregation in ISNE."""
    num_hops: int = 2
    hop_weights: List[float] = Field(default_factory=lambda: [1.0, 0.5, 0.25])
    use_attention: bool = False
    attention_heads: int = 4
    normalize_weights: bool = True
    include_self_loops: bool = True
    

class AttentionWeights(BaseSchema):
    """Attention weights for graph aggregation."""
    source_node: NodeID
    neighbor_weights: Dict[NodeID, float]
    hop_level: int
    total_weight: float = 1.0
    

class DirectoryAwareConfig(BaseSchema):
    """Configuration specific to directory-aware ISNE."""
    use_directory_hierarchy: bool = True
    directory_weight_factor: float = 0.3
    sibling_bonus: float = 0.2
    parent_child_bonus: float = 0.3
    depth_penalty: float = 0.1
    max_depth_difference: int = 3