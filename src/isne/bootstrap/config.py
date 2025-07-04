"""
ISNE bootstrap configuration.

This module provides configuration classes and utilities for ISNE bootstrapping.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import Field

from src.types.common import BaseSchema


class ISNEBootstrapConfig(BaseSchema):
    """Configuration for ISNE bootstrapping process."""
    
    # Input/output paths
    input_dir: Path
    output_dir: Path
    checkpoint_dir: Optional[Path] = None
    
    # Processing settings
    batch_size: int = 32
    max_workers: int = 4
    chunk_size: int = 512
    chunk_overlap: int = 128
    
    # Model settings
    embedding_model: str = "jina_v4"
    embedding_dimension: int = 768
    isne_hidden_dim: int = 256
    isne_num_layers: int = 2
    
    # Training settings
    num_epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    
    # Graph construction
    similarity_threshold: float = 0.7
    max_neighbors: int = 10
    edge_types: List[str] = Field(default_factory=lambda: ["semantic", "structural"])
    
    # Evaluation
    eval_metrics: List[str] = Field(default_factory=lambda: ["mrr", "hit_rate"])
    eval_k_values: List[int] = Field(default_factory=lambda: [1, 5, 10])
    
    # Pipeline stages to run
    stages: List[str] = Field(default_factory=lambda: [
        "document_processing",
        "chunking",
        "embedding",
        "graph_construction",
        "isne_training",
        "model_evaluation"
    ])
    
    # Stage-specific configs
    stage_configs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


def load_bootstrap_config(config_path: Path) -> ISNEBootstrapConfig:
    """Load bootstrap configuration from file."""
    import yaml
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return ISNEBootstrapConfig(**config_data)


def save_bootstrap_config(config: ISNEBootstrapConfig, output_path: Path) -> None:
    """Save bootstrap configuration to file."""
    import yaml
    
    with open(output_path, 'w') as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)


class ISNETrainingConfig(BaseSchema):
    """Training-specific configuration."""
    num_epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32
    device: str = "cuda"
    weight_decay: float = 0.0001


class ModelEvaluationConfig(BaseSchema):
    """Evaluation-specific configuration."""
    metrics: List[str] = Field(default_factory=lambda: ["mrr", "hit_rate"])
    k_values: List[int] = Field(default_factory=lambda: [1, 5, 10])
    run_evaluation: bool = True


# Legacy config classes for compatibility
DocumentProcessingConfig = ISNEBootstrapConfig
ChunkingConfig = ISNEBootstrapConfig
EmbeddingConfig = ISNEBootstrapConfig
GraphConstructionConfig = ISNEBootstrapConfig


__all__ = [
    "ISNEBootstrapConfig",
    "ISNETrainingConfig",
    "ModelEvaluationConfig",
    "DocumentProcessingConfig",
    "ChunkingConfig", 
    "EmbeddingConfig",
    "GraphConstructionConfig",
    "load_bootstrap_config",
    "save_bootstrap_config"
]