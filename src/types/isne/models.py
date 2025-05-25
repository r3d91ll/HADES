"""ISNE model type definitions.

This module provides TypedDict and Enum definitions for ISNE models,
training configuration, and document graph structures.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict, Union
from enum import Enum
from datetime import datetime


class DocumentType(str, Enum):
    """Types of documents in the ISNE system."""
    
    TEXT = "text"
    """Plain text document."""
    
    PDF = "pdf"
    """PDF document."""
    
    CODE = "code"
    """Source code document."""
    
    MARKDOWN = "markdown"
    """Markdown document."""
    
    HTML = "html"
    """HTML document."""


class RelationType(str, Enum):
    """Types of relationships between document entities."""
    
    SIMILARITY = "similarity"
    """Embedding similarity relationship."""
    
    SAME_DOCUMENT = "same_document"
    """Entities from the same document."""
    
    SEQUENTIAL = "sequential"
    """Sequential relationship between chunks."""
    
    REFERENCE = "reference"
    """One entity references another."""
    
    PARENT_CHILD = "parent_child"
    """Hierarchical relationship."""
    
    CUSTOM = "custom"
    """Custom relationship type."""


class ISNEModelConfig(TypedDict, total=False):
    """Configuration for ISNE model architecture."""
    
    embedding_dim: int
    """Dimension of input embeddings."""
    
    hidden_dim: int
    """Dimension of hidden layers."""
    
    output_dim: int
    """Dimension of output embeddings."""
    
    num_layers: int
    """Number of ISNE layers."""
    
    num_heads: int
    """Number of attention heads in multi-head attention."""
    
    dropout: float
    """Dropout rate."""
    
    activation: str
    """Activation function (elu, relu, leaky_relu)."""
    
    add_self_loops: bool
    """Whether to add self-loops to graph edges."""
    
    normalization: bool
    """Whether to use layer normalization."""
    
    attention_type: str
    """Type of attention mechanism."""


class ISNETrainingConfig(TypedDict, total=False):
    """Configuration for ISNE model training."""
    
    learning_rate: float
    """Learning rate for optimization."""
    
    weight_decay: float
    """Weight decay for regularization."""
    
    batch_size: int
    """Batch size for training."""
    
    epochs: int
    """Number of training epochs."""
    
    num_hops: int
    """Number of hops for neighborhood sampling."""
    
    neighbor_size: int
    """Maximum number of neighbors to sample per node."""
    
    eval_interval: int
    """Interval for evaluation during training."""
    
    early_stopping_patience: int
    """Patience for early stopping."""
    
    checkpoint_interval: int
    """Interval (in epochs) for saving checkpoints."""
    
    device: str
    """Device to use for training ("cpu", "cuda:0", etc.)."""
    
    lambda_feat: float
    """Weight for feature preservation loss."""
    
    lambda_struct: float
    """Weight for structural preservation loss."""
    
    lambda_contrast: float
    """Weight for contrastive loss."""
    
    validation_fraction: float
    """Fraction of data to use for validation."""
    
    loss_weights: Dict[str, float]
    """Weights for different loss components."""
    
    optimizer: str
    """Optimizer to use ("adam", "sgd", etc.)."""
    
    scheduler: Optional[Dict[str, Any]]
    """Learning rate scheduler configuration."""


class ISNEGraphConfig(TypedDict, total=False):
    """Configuration for ISNE graph construction."""
    
    similarity_threshold: float
    """Minimum similarity for connecting nodes."""
    
    max_neighbors: int
    """Maximum number of neighbors per node based on similarity."""
    
    sequential_weight: float
    """Edge weight for sequential connections."""
    
    similarity_weight: float
    """Base weight for similarity-based connections."""
    
    window_size: int
    """Window size for sequential context connections."""


class ISNEDirectoriesConfig(TypedDict, total=False):
    """Configuration for ISNE directory paths."""
    
    data_dir: str
    """Base directory for ISNE data."""
    
    input_dir: str
    """Directory containing processed documents for training."""
    
    output_dir: str
    """Directory for storing training artifacts and results."""
    
    model_dir: str
    """Directory for saving trained models."""


class ISNEConfig(TypedDict, total=False):
    """Overall configuration for ISNE."""
    
    use_isne: bool
    """Whether to use ISNE enhancement."""
    
    isne_model_path: Optional[str]
    """Path to pre-trained ISNE model."""
    
    model: ISNEModelConfig
    """Model architecture configuration."""
    
    training: ISNETrainingConfig
    """Training configuration."""
    
    graph: ISNEGraphConfig
    """Graph construction configuration."""
    
    directories: ISNEDirectoriesConfig
    """Directory paths configuration."""
    
    modality: str
    """Data modality ("text", "code", etc.)."""
    
    edge_threshold: float
    """Threshold for creating edges between nodes."""
    
    max_edges_per_node: int
    """Maximum number of edges per node."""


class IngestDocument(TypedDict, total=False):
    """Document representation for ISNE graph construction."""
    
    id: str
    """Unique identifier for the document."""
    
    type: DocumentType
    """Type of document."""
    
    content: Optional[str]
    """Document content (may be omitted to save space)."""
    
    chunks: List[Dict[str, Any]]
    """List of document chunks."""
    
    metadata: Dict[str, Any]
    """Document metadata."""
    
    embeddings: Dict[str, List[float]]
    """Chunk embeddings (key = chunk_id)."""
    
    embedding_model: str
    """Model used to generate embeddings."""


class DocumentRelation(TypedDict, total=False):
    """Relationship between document entities."""
    
    source_id: str
    """ID of the source entity."""
    
    target_id: str
    """ID of the target entity."""
    
    relation_type: RelationType
    """Type of relationship."""
    
    weight: float
    """Relationship weight/strength."""
    
    metadata: Dict[str, Any]
    """Additional metadata about the relationship."""
