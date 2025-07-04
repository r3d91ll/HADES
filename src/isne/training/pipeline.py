"""
ISNE Training Pipeline

Formal training pipeline for regular ISNE model improvements.
This is separate from bootstrap - it takes existing models and continues training.

This module provides:
- Configuration-driven training
- Model versioning
- Data sampling with percentage control
- API-safe operations
"""

import logging
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

import yaml
import torch

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Result of ISNE training pipeline."""
    success: bool
    model_path: Optional[str] = None
    model_name: Optional[str] = None
    version: Optional[int] = None
    training_stats: Optional[Dict[str, Any]] = None
    evaluation_results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_stage: Optional[str] = None
    total_time_seconds: Optional[float] = None


class ISNETrainingConfig:
    """Configuration handler for ISNE training pipeline."""
    
    def __init__(self, config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize training configuration.
        
        Args:
            config_path: Path to YAML config file (defaults to standard location)
            overrides: Dictionary of configuration overrides
        """
        if config_path is None:
            # Default to standard config location
            config_path = Path(__file__).parent.parent.parent / "config" / "isne_training_config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Apply overrides if provided
        if overrides:
            self._apply_overrides(overrides)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded training configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            raise
    
    def _apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply configuration overrides safely."""
        allowed_overrides = self.config.get('api', {}).get('allowed_overrides', [])
        
        for key, value in overrides.items():
            if key in allowed_overrides:
                self._set_nested_value(self.config, key, value)
                logger.info(f"Applied override: {key} = {value}")
            else:
                logger.warning(f"Override not allowed: {key} (not in allowed_overrides)")
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any) -> None:
        """Set nested configuration value using dot notation."""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        current = self.config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def resolve_model_paths(self) -> Dict[str, str]:
        """Resolve and validate model paths."""
        # Source model path
        source_path = self.get('model.source_model.path')
        if not Path(source_path).is_absolute():
            # Resolve relative to HADES root
            hades_root = Path(__file__).parent.parent.parent.parent
            source_path = str(hades_root / source_path)
        
        # Target model path with versioning
        output_dir = self.get('model.target_model.output_dir')
        model_name_template = self.get('model.target_model.name')
        
        # Handle versioning
        version = self._get_next_version(output_dir)
        timestamp = datetime.now(timezone.utc).strftime(self.get('model.target_model.versioning.timestamp_format'))
        
        # Format model name
        model_name = model_name_template.format(version=version, timestamp=timestamp)
        target_path = str(Path(output_dir) / f"{model_name}.pth")
        
        return {
            'source_path': source_path,
            'target_path': target_path,
            'model_name': model_name,
            'version': version
        }
    
    def _get_next_version(self, output_dir: str) -> int:
        """Get next version number for model."""
        if not self.get('model.target_model.versioning.auto_increment', True):
            return self.get('model.target_model.versioning.version', 1)
        
        # Find highest existing version
        output_path = Path(output_dir)
        if not output_path.exists():
            return 1
        
        max_version = 0
        for model_file in output_path.glob("isne_v*.pth"):
            try:
                # Extract version from filename like "isne_v3_20250615_143000.pth"
                name_parts = model_file.stem.split('_')
                if len(name_parts) >= 2 and name_parts[1].startswith('v'):
                    version = int(name_parts[1][1:])  # Remove 'v' prefix
                    max_version = max(max_version, version)
            except (ValueError, IndexError):
                continue
        
        return max_version + 1


class ISNETrainingPipeline:
    """Formal ISNE training pipeline for regular model improvements."""
    
    def __init__(self, config: Optional[ISNETrainingConfig] = None):
        """
        Initialize training pipeline.
        
        Args:
            config: Training configuration (will create default if None)
        """
        self.config = config or ISNETrainingConfig()
        self.logger = logging.getLogger(__name__)
        self.wandb_run: Any = None
        self._init_wandb()
    
    def _init_wandb(self):
        """Initialize Weights & Biases if available."""
        try:
            import wandb
            import os
            from dotenv import load_dotenv
            
            # Load environment variables from .env file
            load_dotenv()
            
            self.wandb = wandb
            self.wandb_available = True
            
            # Check if W&B API key is available
            api_key = os.getenv('WANDB_API_KEY')
            if api_key:
                self.logger.info("W&B integration available with API key from .env")
            else:
                self.logger.warning("W&B available but no API key found in .env - login required")
            
        except ImportError:
            self.wandb = None
            self.wandb_available = False
            self.logger.info("W&B not available - training will proceed without experiment tracking")
    
    def _start_wandb_run(self, model_paths: Dict[str, str], training_config: Dict[str, Any]):
        """Start W&B run for training tracking."""
        if not self.wandb_available:
            return
        
        try:
            import os
            
            run_name = f"isne_training_{model_paths['model_name']}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            
            wandb_config = {
                **training_config,
                'model_name': model_paths['model_name'],
                'version': model_paths['version'],
                'source_model': model_paths['source_path'],
                'target_model': model_paths['target_path'],
                'data_percentage': self.config.get('data.data_percentage'),
                'pipeline_type': 'training',
                'framework': 'pytorch',
                'task': 'graph_embedding',
                'method': 'inductive_shallow_node_embedding'
            }
            
            # Use project and entity from environment if available, otherwise use defaults
            project = os.getenv('WANDB_PROJECT', 'olympus-isne')
            entity = os.getenv('WANDB_ENTITY', None)
            
            self.wandb_run = self.wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=wandb_config,
                tags=["isne", "training", "rag", "graph-embedding"],
                notes=f"ISNE training pipeline - {model_paths['model_name']}",
                reinit=True
            )
            
            self.logger.info(f"Started W&B run: {run_name} (project: {project})")
            
        except Exception as e:
            self.logger.warning(f"Failed to start W&B run: {e}")
            self.wandb_run = None
    
    def _log_wandb_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to W&B."""
        if self.wandb_run:
            try:
                self.wandb_run.log(metrics, step=step)
            except Exception as e:
                self.logger.warning(f"Failed to log W&B metrics: {e}")
    
    def _finish_wandb_run(self, result: TrainingResult) -> None:
        """Finish W&B run and log final results."""
        if self.wandb_run:
            try:
                # Log final results
                final_metrics = {
                    'training_success': result.success,
                    'total_time_minutes': result.total_time_seconds / 60 if result.total_time_seconds else 0,
                }
                
                if result.training_stats:
                    final_metrics.update(result.training_stats)
                
                if result.evaluation_results:
                    final_metrics.update({f'eval_{k}': v for k, v in result.evaluation_results.items()})
                
                self.wandb_run.log(final_metrics)
                
                # Log model artifact if successful
                if result.success and result.model_path:
                    try:
                        artifact = self.wandb.Artifact(
                            name=f"isne_model_{result.model_name}",
                            type="model",
                            description=f"ISNE model {result.model_name} v{result.version}"
                        )
                        artifact.add_file(result.model_path)
                        self.wandb_run.log_artifact(artifact)
                        self.logger.info(f"Logged model artifact: {result.model_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to log model artifact: {e}")
                
                self.wandb_run.finish()
                self.logger.info("W&B run completed")
                
            except Exception as e:
                self.logger.warning(f"Failed to finish W&B run: {e}")
    
    def train(self, overrides: Optional[Dict[str, Any]] = None) -> TrainingResult:
        """
        Run ISNE training pipeline.
        
        Args:
            overrides: Configuration overrides for this training run
            
        Returns:
            TrainingResult with training outcomes
        """
        start_time = time.time()
        
        try:
            # Apply any overrides
            if overrides:
                self.config._apply_overrides(overrides)
            
            # Validate prerequisites
            model_paths = self.config.resolve_model_paths()
            self._validate_prerequisites(model_paths)
            
            # Start W&B run
            training_config = {
                'epochs': self.config.get('training.epochs'),
                'learning_rate': self.config.get('training.learning_rate'),
                'batch_size': self.config.get('training.batch_size'),
                'device': self.config.get('training.device'),
            }
            self._start_wandb_run(model_paths, training_config)
            
            self.logger.info("=" * 60)
            self.logger.info("ISNE TRAINING PIPELINE")
            self.logger.info("=" * 60)
            self.logger.info(f"Source model: {model_paths['source_path']}")
            self.logger.info(f"Target model: {model_paths['target_path']}")
            self.logger.info(f"Model name: {model_paths['model_name']}")
            self.logger.info(f"Version: {model_paths['version']}")
            self.logger.info(f"Data directory: {self.config.get('data.data_dir')}")
            self.logger.info(f"Data percentage: {self.config.get('data.data_percentage') * 100:.1f}%")
            self.logger.info("=" * 60)
            
            # Sample training data
            sampled_files = self._sample_training_data()
            self.logger.info(f"Sampled {len(sampled_files)} files for training")
            
            # Load source model
            source_model = self._load_source_model(model_paths['source_path'])
            
            # Run training stages
            processed_data = self._process_documents(sampled_files)
            self._log_wandb_metrics({'stage': 'document_processing_completed', 'files_processed': len(sampled_files)})
            
            graph_data = self._build_graph(processed_data)
            self._log_wandb_metrics({'stage': 'graph_construction_completed', 'graph_nodes': len(graph_data.get('nodes', []))})
            
            trained_model = self._train_model(source_model, graph_data)
            
            # Save trained model
            self._save_model(trained_model, model_paths['target_path'], model_paths)
            
            # Run evaluation if enabled
            evaluation_results = None
            if self.config.get('evaluation.enabled', True):
                evaluation_results = self._evaluate_model(model_paths['target_path'], graph_data)
            
            total_time = time.time() - start_time
            
            training_stats = {
                'files_processed': len(sampled_files),
                'data_percentage': self.config.get('data.data_percentage'),
                'epochs_trained': self.config.get('training.epochs'),
                'total_time_seconds': total_time
            }
            
            self.logger.info("=" * 60)
            self.logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 60)
            self.logger.info(f"Model saved: {model_paths['target_path']}")
            self.logger.info(f"Training time: {total_time / 60:.1f} minutes")
            if evaluation_results:
                performance = evaluation_results.get('inductive_performance', 0)
                self.logger.info(f"Performance: {performance:.2%}")
            
            result = TrainingResult(
                success=True,
                model_path=model_paths['target_path'],
                model_name=model_paths['model_name'],
                version=model_paths['version'],
                training_stats=training_stats,
                evaluation_results=evaluation_results,
                total_time_seconds=total_time
            )
            
            # Finish W&B run
            self._finish_wandb_run(result)
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            error_msg = f"Training failed: {e}"
            self.logger.error(error_msg)
            import traceback
            traceback.print_exc()
            
            result = TrainingResult(
                success=False,
                error_message=error_msg,
                total_time_seconds=total_time
            )
            
            # Finish W&B run even for failed training
            self._finish_wandb_run(result)
            
            return result
    
    def _validate_prerequisites(self, model_paths: Dict[str, str]) -> None:
        """Validate that all prerequisites are met."""
        # Check source model exists
        source_path = Path(model_paths['source_path'])
        if not source_path.exists():
            raise FileNotFoundError(f"Source model not found: {source_path}")
        
        # Check data directory exists
        data_dir = self.config.get('data.data_dir')
        if not Path(data_dir).exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Create output directory if needed
        output_dir = Path(model_paths['target_path']).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("✅ Prerequisites validated")
    
    def _sample_training_data(self) -> List[Path]:
        """Sample training data based on configuration."""
        data_dir = self.config.get('data.data_dir')
        data_percentage = self.config.get('data.data_percentage')
        sampling_config = self.config.get('data.sampling', {})
        
        self.logger.info(f"Sampling {data_percentage * 100:.1f}% of data from {data_dir}")
        
        # Use the local sampling function
        sampled_files = sample_files_by_percentage(
            data_dir=data_dir,
            percentage=data_percentage,
            strategy=sampling_config.get('strategy', 'stratified'),
            seed=sampling_config.get('seed', 42)
        )
        
        return sampled_files
    
    def _load_source_model(self, model_path: str) -> Any:
        """Load the source model for continued training."""
        self.logger.info(f"Loading source model: {model_path}")
        
        # Import here to avoid circular imports
        from src.isne.models.isne_model import ISNEModel
        
        model = ISNEModel.load(model_path)
        self.logger.info(f"✅ Loaded model with {model.num_nodes} nodes")
        
        return model
    
    def _process_documents(self, files: List[Path]) -> Dict[str, Any]:
        """Process documents through the pipeline."""
        self.logger.info("Processing documents...")
        
        # Import and use bootstrap pipeline components
        from ..bootstrap.stages.document_processing import DocumentProcessingStage
        from ..bootstrap.stages.chunking import ChunkingStage  
        from ..bootstrap.stages.embedding import EmbeddingStage
        
        # Create stages with config
        doc_stage = DocumentProcessingStage()
        chunk_stage = ChunkingStage()
        embed_stage = EmbeddingStage()
        
        # Process through pipeline
        doc_config = self._create_stage_config('document_processing')
        doc_result = doc_stage.execute(files, doc_config)
        
        if not doc_result.success:
            raise RuntimeError(f"Document processing failed: {doc_result.error}")
        
        chunk_config = self._create_stage_config('chunking')
        chunk_result = chunk_stage.execute(doc_result.data, chunk_config)
        
        if not chunk_result.success:
            raise RuntimeError(f"Chunking failed: {chunk_result.error}")
        
        embed_config = self._create_stage_config('embedding')
        embed_result = embed_stage.execute(chunk_result.data, embed_config)
        
        if not embed_result.success:
            raise RuntimeError(f"Embedding failed: {embed_result.error}")
        
        self.logger.info(f"✅ Processed {len(embed_result.data.get('embeddings', []))} embeddings")
        
        return {
            'documents': doc_result.data,
            'chunks': chunk_result.data,
            'embeddings': embed_result.data.get('embeddings', [])
        }
    
    def _build_graph(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build graph from processed data."""
        self.logger.info("Building graph...")
        
        from ..bootstrap.stages.graph_construction import GraphConstructionStage
        
        graph_stage = GraphConstructionStage()
        graph_config = self._create_stage_config('graph_construction')
        
        graph_result = graph_stage.execute(processed_data['embeddings'], graph_config)
        
        if not graph_result.success:
            raise RuntimeError(f"Graph construction failed: {graph_result.error_message}")
        
        self.logger.info(f"✅ Built graph with {graph_result.stats.get('num_nodes', 0)} nodes")
        
        return graph_result.graph_data
    
    def _train_model(self, source_model: Any, graph_data: Dict[str, Any]) -> Any:
        """Train the model on new data."""
        self.logger.info("Training model...")
        
        from ..bootstrap.stages.isne_training import ISNETrainingStage
        
        training_stage = ISNETrainingStage()
        training_config = self._create_stage_config('training')
        
        # Get output directory from model paths
        model_paths = self.config.resolve_model_paths()
        output_dir = Path(model_paths['target_path']).parent
        
        # The bootstrap stage expects graph_data directly, not wrapped in another dict
        training_result = training_stage.execute(graph_data, training_config, output_dir)
        
        if not training_result.success:
            raise RuntimeError(f"Model training failed: {training_result.error_message}")
        
        training_stats = training_result.training_stats
        self.logger.info(f"✅ Training completed in {training_stats.get('training_time', 0):.1f}s")
        
        # Load the trained model from the saved path
        if training_result.model_path:
            from src.isne.models.isne_model import ISNEModel
            trained_model = ISNEModel.load(training_result.model_path)
            return trained_model
        else:
            raise RuntimeError("Training completed but no model path was returned")
    
    def _save_model(self, model: Any, model_path: str, model_info: Dict[str, str]) -> None:
        """Save the trained model with metadata."""
        self.logger.info(f"Saving model to: {model_path}")
        
        # Save model
        model.save(model_path)
        
        # Save metadata
        metadata = {
            'model_name': model_info['model_name'],
            'version': model_info['version'],
            'created_at': datetime.now(timezone.utc).isoformat(),
            'source_model': model_info['source_path'],
            'config_used': self.config.config,
            'training_type': 'incremental'
        }
        
        metadata_path = Path(model_path).with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"✅ Model and metadata saved")
    
    def _evaluate_model(self, model_path: str, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the trained model."""
        self.logger.info("Evaluating model...")
        
        from ..bootstrap.stages.model_evaluation import ModelEvaluationStage
        
        eval_stage = ModelEvaluationStage()
        eval_config = self._create_stage_config('evaluation')
        
        # Get output directory from model paths
        model_paths = self.config.resolve_model_paths()
        output_dir = Path(model_paths['target_path']).parent
        
        # Save graph data to temporary file for evaluation
        import json
        graph_data_path = output_dir / "temp_graph_data.json"
        with open(graph_data_path, 'w') as f:
            json.dump(graph_data, f)
        
        # Use the run method with proper parameters
        eval_result = eval_stage.run(model_path, str(graph_data_path), output_dir, eval_config)
        
        if eval_result.success:
            metrics = eval_result.evaluation_metrics
            performance = metrics.get('inductive_performance', 0)
            self.logger.info(f"✅ Evaluation completed - Performance: {performance:.2%}")
            return metrics
        else:
            self.logger.warning(f"Evaluation failed: {eval_result.error_message}")
            return {}
    
    def _create_stage_config(self, stage_name: str) -> Any:
        """Create configuration object for a pipeline stage."""
        from ..bootstrap.config import (
            DocumentProcessingConfig, ChunkingConfig, EmbeddingConfig,
            GraphConstructionConfig, ISNETrainingConfig, ModelEvaluationConfig
        )
        
        # Get configuration data from YAML
        config_data = self.config.get(f'pipeline.{stage_name}', {})
        
        # Create proper config objects based on stage name
        if stage_name == 'document_processing':
            # Filter out parameters that don't belong to DocumentProcessingConfig
            valid_params = {
                'processor_type': config_data.get('processor_type', 'core'),
                'extract_metadata': config_data.get('extract_metadata', True),
                'extract_sections': config_data.get('extract_sections', True),
                'extract_entities': config_data.get('extract_entities', True),
            }
            return DocumentProcessingConfig(**valid_params)
        elif stage_name == 'chunking':
            valid_params = {
                'chunker_type': config_data.get('chunker_type', 'core'),
                'strategy': config_data.get('strategy', 'semantic'),
                'chunk_size': config_data.get('chunk_size', 512),
                'chunk_overlap': config_data.get('overlap', 50),
            }
            return ChunkingConfig(**valid_params)
        elif stage_name == 'embedding':
            valid_params = {
                'embedder_type': config_data.get('embedder_type', 'cpu'),
                'model_name': config_data.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2'),
                'batch_size': config_data.get('batch_size', 32),
                'device': config_data.get('device', 'cpu'),
            }
            return EmbeddingConfig(**valid_params)
        elif stage_name == 'graph_construction':
            valid_params = {
                'similarity_threshold': config_data.get('similarity_threshold', 0.6),
                'max_edges_per_node': config_data.get('max_edges_per_node', 10),
            }
            return GraphConstructionConfig(**valid_params)
        elif stage_name == 'training':
            return ISNETrainingConfig(
                epochs=self.config.get('training.epochs', 20),
                learning_rate=self.config.get('training.learning_rate', 0.001),
                batch_size=self.config.get('training.batch_size', 64),
                device=self.config.get('training.device', 'cuda'),
                **config_data
            )
        elif stage_name == 'evaluation':
            return ModelEvaluationConfig(
                run_evaluation=self.config.get('evaluation.enabled', True),
                **config_data
            )
        else:
            # Default fallback - return dictionary for unknown stages
            return config_data


# Utility function for sampling files
def sample_files_by_percentage(data_dir: str, percentage: float, strategy: str = 'stratified', seed: int = 42) -> List[Path]:
    """
    Sample files from data directory by percentage.
    
    Args:
        data_dir: Directory to sample from
        percentage: Percentage of files to sample (0.0 to 1.0)
        strategy: Sampling strategy ('stratified', 'random', 'latest')
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled file paths
    """
    import random
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Collect all supported files
    supported_extensions = {'.pdf', '.py', '.md', '.txt', '.json', '.html', '.yaml', '.yml'}
    all_files = []
    
    for ext in supported_extensions:
        all_files.extend(data_path.rglob(f"*{ext}"))
    
    if not all_files:
        raise ValueError(f"No supported files found in {data_dir}")
    
    # Sample based on strategy
    random.seed(seed)
    
    if strategy == 'stratified':
        # Sample by file type to maintain diversity
        files_by_type = {}
        for file_path in all_files:
            ext = file_path.suffix.lower()
            if ext not in files_by_type:
                files_by_type[ext] = []
            files_by_type[ext].append(file_path)
        
        sampled_files = []
        for ext, files in files_by_type.items():
            sample_size = max(1, int(len(files) * percentage))
            sampled = random.sample(files, min(sample_size, len(files)))
            sampled_files.extend(sampled)
    
    elif strategy == 'random':
        # Simple random sampling
        sample_size = max(1, int(len(all_files) * percentage))
        sampled_files = random.sample(all_files, min(sample_size, len(all_files)))
    
    elif strategy == 'latest':
        # Sample most recently modified files
        all_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        sample_size = max(1, int(len(all_files) * percentage))
        sampled_files = all_files[:sample_size]
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    return sampled_files