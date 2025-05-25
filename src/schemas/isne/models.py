"""
ISNE model schemas for HADES-PathRAG.

This module provides Pydantic models for ISNE configuration,
training parameters, and document graph structures.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, field_validator, model_validator

from ..common.base import BaseSchema
from ..common.types import EmbeddingVector, MetadataDict


class ISNEDocumentType(str, Enum):
    """Types of documents in the ISNE system."""
    TEXT = "text"
    PDF = "pdf"
    CODE = "code"
    MARKDOWN = "markdown"
    HTML = "html"


class ISNERelationType(str, Enum):
    """Types of relationships between document entities."""
    SIMILARITY = "similarity"
    SAME_DOCUMENT = "same_document"
    SEQUENTIAL = "sequential"
    REFERENCE = "reference"
    PARENT_CHILD = "parent_child"
    CUSTOM = "custom"


class ISNEModelConfigSchema(BaseSchema):
    """Configuration for ISNE model architecture."""
    
    hidden_dim: int = Field(default=256, description="Dimension of hidden layers")
    output_dim: int = Field(default=768, description="Dimension of output embeddings")
    num_layers: int = Field(default=2, description="Number of ISNE layers")
    num_heads: int = Field(default=8, description="Number of attention heads in multi-head attention")
    dropout: float = Field(default=0.1, description="Dropout rate")
    activation: str = Field(default="relu", description="Activation function")
    normalization: bool = Field(default=True, description="Whether to use layer normalization")
    attention_type: str = Field(default="dot_product", description="Type of attention mechanism")
    
    @field_validator("hidden_dim", "output_dim", "num_layers", "num_heads")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Validate integer values are positive."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v
    
    @field_validator("dropout")
    @classmethod
    def validate_dropout(cls, v: float) -> float:
        """Validate dropout is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError(f"Dropout must be between 0 and 1, got {v}")
        return v


class ISNETrainingConfigSchema(BaseSchema):
    """Configuration for ISNE model training."""
    
    learning_rate: float = Field(default=0.001, description="Learning rate for optimization")
    weight_decay: float = Field(default=0.0001, description="Weight decay for regularization")
    batch_size: int = Field(default=32, description="Batch size for training")
    epochs: int = Field(default=100, description="Number of training epochs")
    early_stopping_patience: int = Field(default=10, description="Patience for early stopping")
    validation_fraction: float = Field(default=0.1, description="Fraction of data to use for validation")
    optimizer: str = Field(default="adam", description="Optimizer for training")
    scheduler: Optional[str] = Field(default=None, description="Learning rate scheduler")
    max_grad_norm: Optional[float] = Field(default=1.0, description="Maximum gradient norm for clipping")
    device: Optional[str] = Field(default=None, description="Device to use for training")
    checkpoint_dir: Optional[str] = Field(default=None, description="Directory for saving checkpoints")
    checkpoint_interval: int = Field(default=10, description="Save checkpoint every N epochs")
    
    @field_validator("learning_rate", "weight_decay")
    @classmethod
    def validate_positive_float(cls, v: float) -> float:
        """Validate float values are positive."""
        if v < 0:
            raise ValueError(f"Value must be non-negative, got {v}")
        return v
    
    @field_validator("batch_size", "epochs", "early_stopping_patience")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Validate integer values are positive."""
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v
    
    @field_validator("validation_fraction")
    @classmethod
    def validate_fraction(cls, v: float) -> float:
        """Validate fraction is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError(f"Validation fraction must be between 0 and 1, got {v}")
        return v


class ISNEConfigSchema(BaseSchema):
    """Overall configuration for ISNE."""
    
    use_isne: bool = Field(default=True, description="Whether to use ISNE for embedding enhancement")
    isne_model_path: Optional[str] = Field(default=None, description="Path to pre-trained ISNE model")
    model: ISNEModelConfigSchema = Field(default_factory=ISNEModelConfigSchema, description="Model architecture configuration")
    training: ISNETrainingConfigSchema = Field(default_factory=ISNETrainingConfigSchema, description="Training configuration")
    modality: str = Field(default="text", description="Modality of the input data")
    edge_threshold: float = Field(default=0.7, description="Threshold for edge creation based on similarity")
    max_edges_per_node: int = Field(default=10, description="Maximum number of edges per node")
    
    @field_validator("edge_threshold")
    @classmethod
    def validate_edge_threshold(cls, v: float) -> float:
        """Validate edge threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError(f"Edge threshold must be between 0 and 1, got {v}")
        return v
    
    @field_validator("max_edges_per_node")
    @classmethod
    def validate_max_edges(cls, v: int) -> int:
        """Validate max edges is positive."""
        if v <= 0:
            raise ValueError(f"Max edges per node must be positive, got {v}")
        return v
