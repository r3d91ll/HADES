"""
Base type definitions for ISNE (Inductive Shallow Node Embedding).

This module provides core type definitions for the ISNE system including
enums, type aliases, and protocol definitions.
"""

from typing import Union, List, Protocol, Any, Optional, Dict
from enum import Enum
import numpy as np
import torch
from torch import Tensor

# Type alias for embedding vectors supporting multiple formats
EmbeddingVector = Union[List[float], np.ndarray, torch.Tensor]


class DocumentType(str, Enum):
    """Types of documents in the ISNE system.
    
    Consolidated enum that supports both general document types
    and specific code language types.
    """
    
    # General document types
    TEXT = "text"
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    
    # Code types
    CODE = "code"  # Generic code
    CODE_PYTHON = "code_python"
    CODE_JAVASCRIPT = "code_javascript"
    CODE_JAVA = "code_java"
    CODE_CPP = "code_cpp"
    CODE_GO = "code_go"
    CODE_RUST = "code_rust"
    CODE_OTHER = "code_other"
    
    # Data formats
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    CSV = "csv"
    
    # Unknown/default
    UNKNOWN = "unknown"


class RelationType(str, Enum):
    """Types of relationships between document entities.
    
    Consolidated enum that includes both structural (code) and 
    semantic (document) relationship types.
    """
    
    # Structural relationships (primarily for code)
    CALLS = "calls"  # Function calls another function
    IMPORTS = "imports"  # Module imports another module
    CONTAINS = "contains"  # Module/class contains function/method
    IMPLEMENTS = "implements"  # Class implements interface
    EXTENDS = "extends"  # Class extends another class
    REFERENCES = "references"  # Code references a symbol
    
    # Document relationships
    DOCUMENTS = "documents"  # Documentation describes code
    REFERS_TO = "refers_to"  # Document references another document
    FOLLOWS = "follows"  # Document follows another in sequence
    
    # Semantic relationships
    SIMILARITY = "similarity"  # Embedding similarity relationship
    SIMILAR_TO = "similar_to"  # Documents are semantically similar
    RELATED_TO = "related_to"  # Documents are contextually related
    
    # Structural document relationships
    SAME_DOCUMENT = "same_document"  # Entities from the same document
    SEQUENTIAL = "sequential"  # Sequential relationship between chunks
    REFERENCE = "reference"  # One entity references another
    PARENT_CHILD = "parent_child"  # Hierarchical relationship
    
    # Generic/custom relationships
    GENERIC = "generic"  # Generic relationship
    CUSTOM = "custom"  # Custom relationship type


class ISNEModelType(str, Enum):
    """Types of ISNE models available."""
    
    FULL = "full"  # Full ISNE model with all components
    SIMPLIFIED = "simplified"  # Simplified ISNE model
    CUSTOM = "custom"  # Custom model implementation


class ActivationFunction(str, Enum):
    """Activation functions supported by ISNE models."""
    
    ELU = "elu"
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    GELU = "gelu"
    SWISH = "swish"
    TANH = "tanh"
    SIGMOID = "sigmoid"


class AttentionType(str, Enum):
    """Types of attention mechanisms."""
    
    SINGLE_HEAD = "single_head"
    MULTI_HEAD = "multi_head"
    GRAPH_ATTENTION = "graph_attention"
    SELF_ATTENTION = "self_attention"


class OptimizerType(str, Enum):
    """Optimizer types for training."""
    
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"


class LossType(str, Enum):
    """Types of loss functions."""
    
    CONTRASTIVE = "contrastive"
    INFONCE = "infonce"
    TRIPLET = "triplet"
    MSE = "mse"
    COSINE = "cosine"
    COMBINED = "combined"


class SamplingStrategy(str, Enum):
    """Sampling strategies for graph training."""
    
    NEIGHBOR = "neighbor"
    RANDOM_WALK = "random_walk"
    UNIFORM = "uniform"
    WEIGHTED = "weighted"
    LAYERWISE = "layerwise"


# Protocol definitions for type checking

class ISNEModelProtocol(Protocol):
    """Protocol defining the interface for ISNE model types."""
    
    def forward(self, x: Tensor, edge_index: Optional[Tensor] = None) -> Tensor:
        """Forward pass through the model."""
        ...
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> Any:
        """Load model state dictionary."""
        ...
    
    def to(self, device: Union[str, torch.device]) -> Any:
        """Move model to device."""
        ...
    
    def eval(self) -> Any:
        """Set model to evaluation mode."""
        ...
    
    def train(self, mode: bool = True) -> Any:
        """Set model to training mode."""
        ...
    
    def __call__(self, x: Tensor, edge_index: Optional[Tensor] = None) -> Tensor:
        """Make model callable."""
        ...


class ISNELoaderProtocol(Protocol):
    """Protocol defining the interface for ISNE data loaders."""
    
    def load(self) -> Any:
        """Load data and return loader result."""
        ...
    
    def validate_config(self) -> bool:
        """Validate loader configuration."""
        ...


class ISNETrainerProtocol(Protocol):
    """Protocol defining the interface for ISNE trainers."""
    
    def train(self, model: ISNEModelProtocol, data: Any) -> Any:
        """Train the model on provided data."""
        ...
    
    def validate(self, model: ISNEModelProtocol, data: Any) -> Any:
        """Validate the model on provided data."""
        ...
    
    def save_checkpoint(self, model: ISNEModelProtocol, path: str) -> None:
        """Save model checkpoint."""
        ...


# Weight constants for different relationship types
RELATIONSHIP_WEIGHTS = {
    # Primary relationships (high weight)
    RelationType.CALLS: 0.9,
    RelationType.CONTAINS: 0.9,
    RelationType.IMPLEMENTS: 0.9,
    
    # Secondary relationships (medium weight)
    RelationType.IMPORTS: 0.6,
    RelationType.REFERENCES: 0.6,
    RelationType.EXTENDS: 0.6,
    RelationType.DOCUMENTS: 0.6,
    
    # Semantic relationships (lower weight)
    RelationType.SIMILAR_TO: 0.3,
    RelationType.SIMILARITY: 0.3,
    RelationType.RELATED_TO: 0.3,
    
    # Sequential relationships
    RelationType.SEQUENTIAL: 0.7,
    RelationType.FOLLOWS: 0.7,
    
    # Default weights
    RelationType.GENERIC: 0.5,
    RelationType.CUSTOM: 0.5,
}