"""
Weights & Biases Integration for ISNE Bootstrap Pipeline

Handles experiment tracking, model logging, and metrics publishing for
the ISNE bootstrap pipeline.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import asdict

logger = logging.getLogger(__name__)


class WandBLogger:
    """Weights & Biases integration for ISNE bootstrap pipeline."""
    
    def __init__(self, config: Any, pipeline_name: str):
        """
        Initialize W&B logger.
        
        Args:
            config: WandBConfig object
            pipeline_name: Name of the pipeline run
        """
        self.config = config
        self.pipeline_name = pipeline_name
        self.wandb = None
        self.run = None
        self.enabled = config.enabled
        
        if self.enabled:
            try:
                import wandb
                self.wandb = wandb
                logger.info("W&B integration enabled")
            except ImportError:
                logger.warning("wandb not installed - disabling W&B integration")
                self.enabled = False
    
    def start_run(self, run_config: Dict[str, Any], model_name: str) -> bool:
        """
        Start W&B run.
        
        Args:
            run_config: Configuration to log
            model_name: Name of the model being trained
            
        Returns:
            True if run started successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            # Create run configuration
            wandb_config = {
                **run_config,
                'model_name': model_name,
                'pipeline_name': self.pipeline_name,
                'framework': 'pytorch',
                'task': 'graph_embedding',
                'method': 'inductive_shallow_node_embedding'
            }
            
            # Start run
            self.run = self.wandb.init(
                project=self.config.project,
                entity=self.config.entity,
                name=f"{model_name}_{self.pipeline_name}",
                config=wandb_config,
                tags=self.config.tags,
                notes=self.config.notes,
                reinit=True
            )
            
            # Log code if enabled
            if self.config.log_code:
                self.wandb.run.log_code(".")
            
            logger.info(f"Started W&B run: {self.run.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start W&B run: {e}")
            self.enabled = False
            return False
    
    def log_stage_metrics(self, stage_name: str, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log stage metrics to W&B.
        
        Args:
            stage_name: Name of the pipeline stage
            metrics: Metrics dictionary
            step: Optional step number
        """
        if not self.enabled or not self.run:
            return
        
        try:
            # Prefix metrics with stage name
            prefixed_metrics = {f"{stage_name}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
            
            if step is not None:
                self.wandb.log(prefixed_metrics, step=step)
            else:
                self.wandb.log(prefixed_metrics)
                
        except Exception as e:
            logger.warning(f"Failed to log stage metrics for {stage_name}: {e}")
    
    def log_training_metrics(self, epoch: int, metrics: Dict[str, Any]):
        """
        Log training metrics for a specific epoch.
        
        Args:
            epoch: Training epoch number
            metrics: Training metrics
        """
        if not self.enabled or not self.run:
            return
        
        try:
            # Log training metrics
            log_dict = {
                'epoch': epoch,
                'training/loss': metrics.get('loss', 0),
                'training/duration_seconds': metrics.get('duration_seconds', 0)
            }
            
            # Add optional metrics
            if 'learning_rate' in metrics:
                log_dict['training/learning_rate'] = metrics['learning_rate']
            if 'gradient_norm' in metrics:
                log_dict['training/gradient_norm'] = metrics['gradient_norm']
            if 'gpu_memory_mb' in metrics:
                log_dict['training/gpu_memory_mb'] = metrics['gpu_memory_mb']
            
            self.wandb.log(log_dict, step=epoch)
            
        except Exception as e:
            logger.warning(f"Failed to log training metrics for epoch {epoch}: {e}")
    
    def log_evaluation_results(self, evaluation_metrics: Dict[str, Any]):
        """
        Log model evaluation results.
        
        Args:
            evaluation_metrics: Complete evaluation results
        """
        if not self.enabled or not self.run:
            return
        
        try:
            # Extract key metrics for W&B
            log_dict = {}
            
            # Model info
            if 'model_info' in evaluation_metrics:
                info = evaluation_metrics['model_info']
                log_dict.update({
                    'model/num_nodes': info.get('num_nodes', 0),
                    'model/num_edges': info.get('num_edges', 0),
                    'model/embedding_dimension': info.get('embedding_dimension', 0),
                    'model/parameters': info.get('model_parameters', 0),
                    'model/graph_density': info.get('graph_density', 0)
                })
            
            # Embedding quality
            if 'embedding_quality' in evaluation_metrics:
                quality = evaluation_metrics['embedding_quality']
                if 'cosine_similarity' in quality:
                    cos_sim = quality['cosine_similarity']
                    log_dict.update({
                        'evaluation/cosine_similarity_mean': cos_sim.get('mean', 0),
                        'evaluation/cosine_similarity_std': cos_sim.get('std', 0)
                    })
                
                if 'embedding_diversity' in quality:
                    diversity = quality['embedding_diversity']
                    log_dict.update({
                        'evaluation/mean_pairwise_distance': diversity.get('mean_pairwise_distance', 0),
                        'evaluation/embedding_norm_mean': diversity.get('embedding_norm_mean', 0)
                    })
            
            # Inductive performance (key metric)
            if 'inductive_performance' in evaluation_metrics:
                inductive = evaluation_metrics['inductive_performance']
                log_dict.update({
                    'evaluation/seen_accuracy': inductive.get('seen_accuracy', 0),
                    'evaluation/unseen_accuracy': inductive.get('unseen_accuracy', 0),
                    'evaluation/relative_performance_percent': inductive.get('relative_performance_percent', 0),
                    'evaluation/achieves_90_percent_target': inductive.get('achieves_90_percent_target', False)
                })
            
            # Graph awareness
            if 'graph_awareness' in evaluation_metrics:
                graph_aware = evaluation_metrics['graph_awareness']
                log_dict.update({
                    'evaluation/similarity_matrix_correlation': graph_aware.get('similarity_matrix_correlation', 0)
                })
                
                if 'neighborhood_preservation' in graph_aware:
                    neighborhood = graph_aware['neighborhood_preservation']
                    log_dict.update({
                        'evaluation/neighborhood_preservation': neighborhood.get('average_score', 0)
                    })
            
            # Log all metrics
            self.wandb.log(log_dict)
            
            # Log evaluation summary as artifact
            self._log_evaluation_artifact(evaluation_metrics)
            
        except Exception as e:
            logger.warning(f"Failed to log evaluation results: {e}")
    
    def log_model_artifact(self, model_path: str, model_config: Dict[str, Any]):
        """
        Log trained model as W&B artifact.
        
        Args:
            model_path: Path to the saved model
            model_config: Model configuration
        """
        if not self.enabled or not self.run or not self.config.log_model:
            return
        
        try:
            # Create model artifact
            model_artifact = self.wandb.Artifact(
                name=f"isne_model_{self.run.name}",
                type="model",
                description=f"ISNE model trained via {self.pipeline_name}",
                metadata=model_config
            )
            
            # Add model file
            model_artifact.add_file(model_path)
            
            # Log artifact
            self.run.log_artifact(model_artifact)
            
            logger.info(f"Logged model artifact: {model_path}")
            
        except Exception as e:
            logger.warning(f"Failed to log model artifact: {e}")
    
    def log_pipeline_summary(self, final_stats: Dict[str, Any], stage_results: Dict[str, Any]):
        """
        Log complete pipeline summary.
        
        Args:
            final_stats: Final pipeline statistics
            stage_results: Results from all stages
        """
        if not self.enabled or not self.run:
            return
        
        try:
            # Extract summary metrics
            summary_dict = {
                'pipeline/total_execution_time': final_stats.get('total_execution_time_seconds', 0),
                'pipeline/stages_completed': final_stats.get('stages_completed', 0),
                'pipeline/success': final_stats.get('pipeline_success', False)
            }
            
            # Add data flow metrics
            if 'data_flow' in final_stats:
                data_flow = final_stats['data_flow']
                summary_dict.update({
                    'data/input_files': data_flow.get('input_files', 0),
                    'data/documents_generated': data_flow.get('documents_generated', 0),
                    'data/chunks_generated': data_flow.get('chunks_generated', 0),
                    'data/embeddings_generated': data_flow.get('embeddings_generated', 0),
                    'data/graph_nodes': data_flow.get('graph_nodes', 0),
                    'data/graph_edges': data_flow.get('graph_edges', 0),
                    'data/training_epochs': data_flow.get('training_epochs', 0),
                    'data/final_loss': data_flow.get('final_loss', 0)
                })
                
                # Evaluation metrics
                if data_flow.get('evaluation_completed'):
                    summary_dict.update({
                        'final/inductive_performance': data_flow.get('inductive_performance', 0),
                        'final/achieves_target': data_flow.get('achieves_90_percent_target', False),
                        'final/evaluation_status': data_flow.get('overall_evaluation_status', 'unknown')
                    })
            
            # Log summary
            self.wandb.log(summary_dict)
            
            # Create pipeline summary artifact
            summary_artifact = self.wandb.Artifact(
                name=f"pipeline_summary_{self.run.name}",
                type="pipeline_results",
                description="Complete ISNE bootstrap pipeline results"
            )
            
            # Save summary to temporary file and add to artifact
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump({
                    'final_stats': final_stats,
                    'stage_results': self._serialize_stage_results(stage_results)
                }, f, indent=2, default=str)
                temp_path = f.name
            
            summary_artifact.add_file(temp_path, name="pipeline_summary.json")
            self.run.log_artifact(summary_artifact)
            
            # Clean up temp file
            os.unlink(temp_path)
            
        except Exception as e:
            logger.warning(f"Failed to log pipeline summary: {e}")
    
    def log_stage_timing(self, stage_name: str, duration_seconds: float, stage_stats: Dict[str, Any]):
        """
        Log stage execution timing and stats.
        
        Args:
            stage_name: Name of the stage
            duration_seconds: Stage execution time
            stage_stats: Stage-specific statistics
        """
        if not self.enabled or not self.run:
            return
        
        try:
            timing_dict = {
                f'timing/{stage_name}_duration': duration_seconds,
                f'timing/{stage_name}_completed': True
            }
            
            # Add stage-specific stats
            if stage_stats:
                for key, value in stage_stats.items():
                    if isinstance(value, (int, float)):
                        timing_dict[f'{stage_name}/{key}'] = value
            
            self.wandb.log(timing_dict)
            
        except Exception as e:
            logger.warning(f"Failed to log stage timing for {stage_name}: {e}")
    
    def finish_run(self, success: bool = True):
        """
        Finish the W&B run.
        
        Args:
            success: Whether the pipeline completed successfully
        """
        if not self.enabled or not self.run:
            return
        
        try:
            # Log final status
            self.wandb.log({
                'pipeline/final_success': success,
                'pipeline/completed': True
            })
            
            # Finish run
            self.wandb.finish(exit_code=0 if success else 1)
            logger.info(f"Finished W&B run: {self.run.name}")
            
        except Exception as e:
            logger.warning(f"Failed to finish W&B run: {e}")
        finally:
            self.run = None
    
    def _log_evaluation_artifact(self, evaluation_metrics: Dict[str, Any]):
        """Log evaluation results as artifact."""
        try:
            eval_artifact = self.wandb.Artifact(
                name=f"evaluation_results_{self.run.name}",
                type="evaluation",
                description="Complete model evaluation results"
            )
            
            # Save evaluation to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(evaluation_metrics, f, indent=2, default=str)
                temp_path = f.name
            
            eval_artifact.add_file(temp_path, name="evaluation_results.json")
            self.run.log_artifact(eval_artifact)
            
            # Clean up
            os.unlink(temp_path)
            
        except Exception as e:
            logger.warning(f"Failed to log evaluation artifact: {e}")
    
    def _serialize_stage_results(self, stage_results: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize stage results for JSON export."""
        serialized = {}
        
        for stage_name, result in stage_results.items():
            try:
                if hasattr(result, 'to_dict'):
                    serialized[stage_name] = result.to_dict()
                elif hasattr(result, '__dict__'):
                    serialized[stage_name] = {k: v for k, v in result.__dict__.items() 
                                            if not k.startswith('_')}
                else:
                    serialized[stage_name] = str(result)
            except Exception as e:
                logger.debug(f"Failed to serialize {stage_name} result: {e}")
                serialized[stage_name] = {"serialization_error": str(e)}
        
        return serialized