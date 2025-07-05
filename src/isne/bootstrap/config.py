"""
ISNE bootstrap configuration.

This module provides configuration classes and utilities for ISNE bootstrapping.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field
import yaml
import json

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
    
    # Device configuration
    device: str = "cuda"
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"


class StageConfig(BaseSchema):
    """Base configuration for pipeline stages."""
    
    enabled: bool = True
    checkpoint_enabled: bool = True
    retry_on_failure: bool = True
    max_retries: int = 3
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"


class DocumentProcessingConfig(StageConfig):
    """Configuration for document processing stage."""
    
    supported_extensions: List[str] = Field(default_factory=lambda: [
        ".txt", ".md", ".pdf", ".docx", ".json", ".yaml"
    ])
    max_file_size_mb: int = 100
    encoding: str = "utf-8"
    extract_metadata: bool = True
    
    # PDF-specific settings
    pdf_extract_images: bool = False
    pdf_ocr_enabled: bool = False
    
    # Preprocessing
    normalize_whitespace: bool = True
    remove_empty_lines: bool = True


class ChunkingConfig(StageConfig):
    """Configuration for chunking stage."""
    
    strategy: str = "semantic"  # semantic, fixed, sliding
    chunk_size: int = 512
    chunk_overlap: int = 128
    min_chunk_size: int = 100
    
    # Semantic chunking settings
    semantic_breakpoint_percentile: float = 0.95
    semantic_buffer_size: int = 1
    
    # Metadata to preserve
    preserve_metadata: List[str] = Field(default_factory=lambda: [
        "source", "page", "section", "timestamp"
    ])


class EmbeddingConfig(StageConfig):
    """Configuration for embedding stage."""
    
    model_name: str = "jina_v4"
    batch_size: int = 32
    max_sequence_length: int = 512
    normalize_embeddings: bool = True
    
    # Model-specific settings
    model_settings: Dict[str, Any] = Field(default_factory=dict)
    
    # Caching
    cache_embeddings: bool = True
    cache_dir: Optional[Path] = None
    
    # GPU settings
    device: str = "cuda"
    mixed_precision: bool = True
    
    # Pooling strategy for longer texts
    pooling_strategy: str = "mean"  # mean, max, cls


class GraphConstructionConfig(StageConfig):
    """Configuration for graph construction stage."""
    
    # Similarity settings
    similarity_metric: str = "cosine"  # cosine, euclidean, dot
    similarity_threshold: float = 0.7
    
    # Graph structure
    max_neighbors: int = 10
    edge_types: List[str] = Field(default_factory=lambda: [
        "semantic", "structural", "temporal"
    ])
    
    # Graph enrichment
    add_entity_nodes: bool = True
    add_keyword_nodes: bool = True
    entity_extraction_model: Optional[str] = None
    keyword_extraction_method: str = "tfidf"  # tfidf, rake, textrank
    
    # Storage backend
    storage_backend: str = "arangodb"  # arangodb, networkx, neo4j
    storage_config: Dict[str, Any] = Field(default_factory=dict)


class ISNETrainingConfig(StageConfig):
    """Configuration for ISNE training stage."""
    
    # Model architecture
    hidden_dimensions: List[int] = Field(default_factory=lambda: [512, 256])
    num_layers: int = 2
    dropout_rate: float = 0.1
    activation: str = "relu"
    
    # Training settings
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 128
    
    # Optimizer
    optimizer: str = "adam"
    optimizer_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Learning rate scheduling
    lr_scheduler: Optional[str] = "cosine"
    lr_scheduler_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Loss function
    loss_function: str = "triplet"  # triplet, contrastive, nce
    margin: float = 0.2
    
    # Sampling strategy
    negative_sampling_strategy: str = "hard"  # hard, semi-hard, random
    num_negative_samples: int = 5
    
    # Validation
    validation_split: float = 0.1
    early_stopping_patience: int = 10
    
    # Checkpointing
    checkpoint_interval: int = 10
    save_best_only: bool = True
    
    # Additional training configs from the incoming version
    training_batch_size: int = Field(64, description="Batch size")
    training_learning_rate: float = Field(0.001, description="Learning rate")
    training_hidden_dim: int = Field(256, description="Hidden dimension")
    training_num_layers: int = Field(3, description="Number of layers")


class ModelEvaluationConfig(StageConfig):
    """Configuration for model evaluation stage."""
    
    # Test data
    test_queries: List[str] = Field(default_factory=list)
    test_data_path: Optional[Path] = None
    
    # Metrics
    metrics: List[str] = Field(default_factory=lambda: [
        "mrr", "recall@k", "precision@k", "ndcg@k", "map"
    ])
    k_values: List[int] = Field(default_factory=lambda: [1, 5, 10, 20])
    
    # Comparison
    compare_baseline: bool = True
    baseline_method: str = "vector_similarity"  # vector_similarity, bm25, random
    
    # Analysis
    analyze_failures: bool = True
    save_predictions: bool = True
    generate_report: bool = True
    
    # Additional evaluation configs from incoming version
    evaluation_test_queries: List[str] = Field(default_factory=list, description="Test queries")
    evaluation_metrics: List[str] = Field(default_factory=lambda: ["mrr", "recall@10", "ndcg"])
    evaluation_compare_baseline: bool = Field(True, description="Compare against baseline")


def load_bootstrap_config(config_path: Path) -> ISNEBootstrapConfig:
    """Load bootstrap configuration from file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix == '.yaml':
            config_dict = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    return ISNEBootstrapConfig(**config_dict)


def save_bootstrap_config(config: ISNEBootstrapConfig, output_path: Path) -> None:
    """Save bootstrap configuration to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.model_dump()
    
    # Convert Path objects to strings
    for key, value in config_dict.items():
        if isinstance(value, Path):
            config_dict[key] = str(value)
    
    with open(output_path, 'w') as f:
        if output_path.suffix == '.yaml':
            yaml.dump(config_dict, f, default_flow_style=False)
        elif output_path.suffix == '.json':
            json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {output_path.suffix}")


# Additional configuration classes from incoming version that weren't duplicates
class DocumentProcessorConfig(BaseModel):
    """Configuration for document processor."""
    supported_formats: List[str] = Field(
        default_factory=lambda: [".txt", ".md", ".pdf", ".docx"],
        description="Supported file formats"
    )
    max_file_size_mb: int = Field(50, description="Maximum file size in MB")
    encoding: str = Field("utf-8", description="Text encoding")


class ChunkerConfig(BaseModel):
    """Configuration for chunker."""
    chunk_size: int = Field(512, description="Chunk size in tokens")
    chunk_overlap: int = Field(50, description="Overlap between chunks")
    min_chunk_size: int = Field(100, description="Minimum chunk size")


__all__ = [
    "ISNEBootstrapConfig",
    "StageConfig",
    "DocumentProcessingConfig",
    "ChunkingConfig",
    "EmbeddingConfig",
    "GraphConstructionConfig",
    "ISNETrainingConfig",
    "ModelEvaluationConfig",
    "DocumentProcessorConfig",
    "ChunkerConfig",
    "load_bootstrap_config",
    "save_bootstrap_config"
]