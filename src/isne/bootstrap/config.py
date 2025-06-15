"""
ISNE Bootstrap Configuration

Configuration management for the ISNE bootstrap pipeline.
Handles loading, validation, and access to bootstrap-specific settings.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class StageConfig:
    """Configuration for a single pipeline stage."""
    enabled: bool = True
    options: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: Optional[int] = None
    retry_attempts: int = 3
    

@dataclass
class WandBConfig:
    """Configuration for Weights & Biases integration."""
    enabled: bool = False  # Disabled for hypothesis testing
    project: str = "olympus-isne"
    entity: Optional[str] = None  # W&B team/user name
    tags: List[str] = field(default_factory=lambda: ["isne", "bootstrap", "rag"])
    notes: str = "ISNE model training via Olympus bootstrap pipeline"
    log_model: bool = True  # Upload model artifacts
    log_code: bool = True  # Upload code
    log_frequency: int = 1  # Log every N epochs
    

@dataclass
class MonitoringConfig:
    """Configuration for bootstrap monitoring."""
    enabled: bool = True
    alert_thresholds: Dict[str, Any] = field(default_factory=lambda: {
        "max_memory_mb": 4000,
        "max_duration_seconds": 3600,
        "min_success_rate": 0.9
    })
    prometheus_export: bool = True
    log_level: str = "INFO"


@dataclass 
class DocumentProcessingConfig(StageConfig):
    """Configuration for document processing stage."""
    processor_type: str = "core"
    extract_metadata: bool = True
    extract_sections: bool = True
    extract_entities: bool = True
    supported_formats: List[str] = field(default_factory=lambda: [
        "pdf", "md", "py", "txt", "yaml", "json"
    ])


@dataclass
class ChunkingConfig(StageConfig):
    """Configuration for chunking stage."""
    chunker_type: str = "core"
    strategy: str = "semantic"
    chunk_size: int = 512
    chunk_overlap: int = 50
    preserve_structure: bool = True


@dataclass
class EmbeddingConfig(StageConfig):
    """Configuration for embedding stage."""
    embedder_type: str = "cpu"  # cpu, gpu, core
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize: bool = True
    device: str = "cpu"


@dataclass
class GraphConstructionConfig(StageConfig):
    """Configuration for graph construction stage."""
    similarity_threshold: float = 0.5  # Lowered from 0.7 for better connectivity
    max_edges_per_node: int = 20  # Increased from 10 for richer neighborhoods
    include_metadata_edges: bool = True
    save_graph_data: bool = True  # Save graph_data.json for evaluation


@dataclass
class ModelEvaluationConfig(StageConfig):
    """Configuration for model evaluation stage."""
    run_evaluation: bool = True  # Whether to run evaluation
    inductive_test_ratio: float = 0.3  # Ratio of nodes to use as "unseen" for inductive test
    k_neighbors: int = 15  # K for KNN classification
    cross_validation_folds: int = 5  # Number of CV folds
    sample_size_for_visualization: int = 1000  # Sample size for t-SNE plots
    save_visualizations: bool = True  # Whether to generate and save plots
    target_relative_performance: float = 0.9  # Target: unseen nodes should achieve >90% performance


@dataclass
class ISNETrainingConfig(StageConfig):
    """Configuration for ISNE model training stage."""
    hidden_dim: int = 128
    output_dim: int = 64
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    epochs: int = 25  # Reduced for hypothesis testing
    batch_size: int = 1024
    patience: int = 15
    log_interval: int = 10
    min_delta: float = 1e-4
    device: str = "cpu"


@dataclass
class BootstrapConfig:
    """Complete bootstrap pipeline configuration."""
    # Pipeline settings
    input_dir: str
    output_dir: str
    pipeline_name: str = "isne_bootstrap"
    enable_monitoring: bool = True
    save_intermediate_results: bool = False
    
    # Stage configurations
    document_processing: DocumentProcessingConfig = field(default_factory=DocumentProcessingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    graph_construction: GraphConstructionConfig = field(default_factory=GraphConstructionConfig)
    isne_training: ISNETrainingConfig = field(default_factory=ISNETrainingConfig)
    model_evaluation: ModelEvaluationConfig = field(default_factory=ModelEvaluationConfig)
    
    # Monitoring and experiment tracking configuration
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    
    @classmethod
    def get_default(cls) -> 'BootstrapConfig':
        """Get default bootstrap configuration."""
        return cls(
            input_dir="./data",
            output_dir="./models",
            pipeline_name="isne_bootstrap",
            enable_monitoring=True,
            save_intermediate_results=False
        )
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> 'BootstrapConfig':
        """Load configuration from YAML file."""
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls(input_dir="./data", output_dir="./models")
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Extract stage configs
            stage_configs = {}
            for stage_name in ['document_processing', 'chunking', 'embedding', 
                              'graph_construction', 'isne_training']:
                if stage_name in config_data:
                    stage_configs[stage_name] = config_data.pop(stage_name)
            
            # Create stage config objects
            if 'document_processing' in stage_configs:
                config_data['document_processing'] = DocumentProcessingConfig(**stage_configs['document_processing'])
            if 'chunking' in stage_configs:
                config_data['chunking'] = ChunkingConfig(**stage_configs['chunking'])
            if 'embedding' in stage_configs:
                config_data['embedding'] = EmbeddingConfig(**stage_configs['embedding'])
            if 'graph_construction' in stage_configs:
                config_data['graph_construction'] = GraphConstructionConfig(**stage_configs['graph_construction'])
            if 'isne_training' in stage_configs:
                config_data['isne_training'] = ISNETrainingConfig(**stage_configs['isne_training'])
            
            # Create monitoring config
            if 'monitoring' in config_data:
                config_data['monitoring'] = MonitoringConfig(**config_data['monitoring'])
            
            return cls(**config_data)
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
            return cls(input_dir="./data", output_dir="./models")
    
    def to_yaml(self, output_path: Path) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'input_dir': self.input_dir,
            'output_dir': self.output_dir,
            'pipeline_name': self.pipeline_name,
            'enable_monitoring': self.enable_monitoring,
            'document_processing': {
                'enabled': self.document_processing.enabled,
                'processor_type': self.document_processing.processor_type,
                'extract_metadata': self.document_processing.extract_metadata,
                'extract_sections': self.document_processing.extract_sections,
                'extract_entities': self.document_processing.extract_entities,
                'supported_formats': self.document_processing.supported_formats,
                'options': self.document_processing.options,
                'timeout_seconds': self.document_processing.timeout_seconds,
                'retry_attempts': self.document_processing.retry_attempts
            },
            'chunking': {
                'enabled': self.chunking.enabled,
                'chunker_type': self.chunking.chunker_type,
                'strategy': self.chunking.strategy,
                'chunk_size': self.chunking.chunk_size,
                'chunk_overlap': self.chunking.chunk_overlap,
                'preserve_structure': self.chunking.preserve_structure,
                'options': self.chunking.options,
                'timeout_seconds': self.chunking.timeout_seconds,
                'retry_attempts': self.chunking.retry_attempts
            },
            'embedding': {
                'enabled': self.embedding.enabled,
                'embedder_type': self.embedding.embedder_type,
                'model_name': self.embedding.model_name,
                'batch_size': self.embedding.batch_size,
                'normalize': self.embedding.normalize,
                'device': self.embedding.device,
                'options': self.embedding.options,
                'timeout_seconds': self.embedding.timeout_seconds,
                'retry_attempts': self.embedding.retry_attempts
            },
            'graph_construction': {
                'enabled': self.graph_construction.enabled,
                'similarity_threshold': self.graph_construction.similarity_threshold,
                'max_edges_per_node': self.graph_construction.max_edges_per_node,
                'include_metadata_edges': self.graph_construction.include_metadata_edges,
                'options': self.graph_construction.options,
                'timeout_seconds': self.graph_construction.timeout_seconds,
                'retry_attempts': self.graph_construction.retry_attempts
            },
            'isne_training': {
                'enabled': self.isne_training.enabled,
                'hidden_dim': self.isne_training.hidden_dim,
                'output_dim': self.isne_training.output_dim,
                'num_layers': self.isne_training.num_layers,
                'num_heads': self.isne_training.num_heads,
                'dropout': self.isne_training.dropout,
                'learning_rate': self.isne_training.learning_rate,
                'weight_decay': self.isne_training.weight_decay,
                'epochs': self.isne_training.epochs,
                'batch_size': self.isne_training.batch_size,
                'patience': self.isne_training.patience,
                'log_interval': self.isne_training.log_interval,
                'min_delta': self.isne_training.min_delta,
                'device': self.isne_training.device,
                'options': self.isne_training.options,
                'timeout_seconds': self.isne_training.timeout_seconds,
                'retry_attempts': self.isne_training.retry_attempts
            },
            'monitoring': {
                'enabled': self.monitoring.enabled,
                'alert_thresholds': self.monitoring.alert_thresholds,
                'prometheus_export': self.monitoring.prometheus_export,
                'log_level': self.monitoring.log_level
            }
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, indent=2, default_flow_style=False)
        
        logger.info(f"Saved configuration to {output_path}")
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate paths
        input_path = Path(self.input_dir)
        if not input_path.exists():
            errors.append(f"Input directory does not exist: {self.input_dir}")
        
        # Validate stage configs
        if self.embedding.batch_size <= 0:
            errors.append("Embedding batch_size must be positive")
        
        if self.isne_training.epochs <= 0:
            errors.append("ISNE training epochs must be positive")
        
        if self.isne_training.hidden_dim <= 0:
            errors.append("ISNE training hidden_dim must be positive")
        
        if self.isne_training.output_dim <= 0:
            errors.append("ISNE training output_dim must be positive")
        
        if not (0 < self.isne_training.learning_rate < 1):
            errors.append("ISNE training learning_rate must be between 0 and 1")
        
        if not (0 <= self.isne_training.dropout < 1):
            errors.append("ISNE training dropout must be between 0 and 1")
        
        return errors