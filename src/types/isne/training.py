"""
ISNE training types for HADES.

This module defines types specific to ISNE training processes.
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pydantic import Field
from pathlib import Path

from ..common import BaseSchema, EmbeddingVector, NodeID


class BatchSample(BaseSchema):
    """A batch sample for ISNE training."""
    anchor_nodes: List[NodeID]
    positive_nodes: List[NodeID]
    negative_nodes: List[NodeID]
    anchor_embeddings: List[EmbeddingVector]
    positive_embeddings: List[EmbeddingVector]
    negative_embeddings: List[EmbeddingVector]
    edge_weights: Optional[List[float]] = None
    node_ids: Optional[List[NodeID]] = None
    features: Optional[Any] = None  # Tensor features
    edge_index: Optional[Any] = None  # Edge index tensor
    directory_features: Optional[Any] = None  # Directory feature tensor
    batch_idx: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TrainingMetrics(BaseSchema):
    """Metrics collected during training."""
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    learning_rate: float
    batch_time_ms: float
    epoch_time_s: float
    gpu_memory_mb: Optional[float] = None
    gradient_norm: Optional[float] = None
    custom_metrics: Dict[str, float] = Field(default_factory=dict)


class TrainingConfig(BaseSchema):
    """Configuration for ISNE training."""
    # Data paths
    train_data_path: Path
    val_data_path: Optional[Path] = None
    output_dir: Path
    
    # Training parameters
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    gradient_clip_norm: Optional[float] = 1.0
    
    # Optimization
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    scheduler: Optional[str] = "cosine"  # "cosine", "linear", "exponential"
    warmup_steps: int = 0
    
    # Loss function
    loss_type: str = "triplet"  # "triplet", "contrastive", "infonce"
    margin: float = 0.5
    temperature: float = 0.07
    
    # Sampling
    negative_samples: int = 5
    hard_negative_mining: bool = True
    sampling_strategy: str = "uniform"  # "uniform", "weighted", "hierarchical"
    
    # Validation
    val_frequency: int = 5  # Validate every N epochs
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"
    
    # Checkpointing
    save_frequency: int = 10
    keep_last_n_checkpoints: int = 3
    save_best_only: bool = True
    
    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True
    
    # Logging
    log_frequency: int = 10
    tensorboard: bool = True
    wandb_project: Optional[str] = None


class CheckpointData(BaseSchema):
    """Data saved in a training checkpoint."""
    epoch: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    scheduler_state_dict: Optional[Dict[str, Any]] = None
    best_metric: float
    training_metrics: List[TrainingMetrics]
    config: TrainingConfig
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())


class TrainingResult(BaseSchema):
    """Result of ISNE training."""
    final_model_path: Path
    best_model_path: Path
    training_history: List[TrainingMetrics]
    best_epoch: int
    best_metric: float
    total_training_time: float
    final_metrics: Dict[str, float]
    config: TrainingConfig
    

class SamplingStrategy(BaseSchema):
    """Strategy for sampling during training."""
    strategy_type: str  # "uniform", "weighted", "hierarchical", "hard"
    negative_sample_ratio: float = 5.0
    positive_sample_weights: Optional[Dict[str, float]] = None
    hard_negative_fraction: float = 0.3
    temperature: float = 1.0
    

class LossMetrics(BaseSchema):
    """Detailed loss metrics."""
    total_loss: float
    anchor_positive_distance: float
    anchor_negative_distance: float
    margin_violations: int
    hard_negatives_ratio: float
    regularization_loss: Optional[float] = None
    

class ValidationMetrics(BaseSchema):
    """Metrics from validation."""
    val_loss: float
    mrr: float  # Mean Reciprocal Rank
    hits_at_k: Dict[int, float]  # k -> hit rate
    map_score: float  # Mean Average Precision
    clustering_metrics: Optional[Dict[str, float]] = None
    

class DirectoryBatch(BaseSchema):
    """Batch information for directory-aware training."""
    batch_directories: List[Path]
    directory_depths: List[int]
    directory_relationships: List[Tuple[int, int, str]]  # (idx1, idx2, relationship_type)
    hierarchy_weights: List[float]


class DirectoryMetadata(BaseSchema):
    """Metadata about a directory in the training dataset."""
    path: Path
    depth: int
    parent: Optional[Path] = None
    children: List[Path] = Field(default_factory=list)
    file_count: int = 0
    total_size: int = 0
    file_types: Dict[str, int] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())