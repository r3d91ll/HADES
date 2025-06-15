"""
ISNE Bootstrap Pipeline

Main orchestrator for the complete ISNE bootstrap process from
raw documents to trained ISNE models.
"""

import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from .config import BootstrapConfig
from .monitoring import BootstrapMonitor
from .wandb_logger import WandBLogger
from .stages.document_processing import DocumentProcessingStage
from .stages.chunking import ChunkingStage
from .stages.embedding import EmbeddingStage
from .stages.graph_construction import GraphConstructionStage
from .stages.model_evaluation import ModelEvaluationStage
from .stages.isne_training import ISNETrainingStage

logger = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    """Result of complete bootstrap pipeline."""
    success: bool
    model_path: Optional[str]
    output_directory: str
    total_time_seconds: float
    stage_results: Dict[str, Any]
    final_stats: Dict[str, Any]
    error_message: Optional[str] = None
    error_stage: Optional[str] = None


class ISNEBootstrapPipeline:
    """Complete ISNE bootstrap pipeline orchestrator."""
    
    def __init__(self, config: BootstrapConfig, monitor: Optional[BootstrapMonitor] = None):
        """
        Initialize bootstrap pipeline.
        
        Args:
            config: Complete bootstrap configuration
            monitor: Optional monitoring instance
        """
        self.config = config
        self.monitor = monitor  # Will be initialized when pipeline runs
        self.wandb_logger = WandBLogger(config.wandb, config.pipeline_name)
        
        # Initialize stages
        self.stages = {
            'document_processing': DocumentProcessingStage(),
            'chunking': ChunkingStage(),
            'embedding': EmbeddingStage(),
            'graph_construction': GraphConstructionStage(),
            'isne_training': ISNETrainingStage(),
            'model_evaluation': ModelEvaluationStage()
        }
        
        self.stage_order = [
            'document_processing',
            'chunking', 
            'embedding',
            'graph_construction',
            'isne_training',
            'model_evaluation'
        ]
        
        logger.info("ISNE Bootstrap Pipeline initialized")
    
    def run(self, 
            input_files: List[Union[str, Path]], 
            output_dir: Union[str, Path],
            model_name: str = "isne_bootstrap_model") -> BootstrapResult:
        """
        Run complete bootstrap pipeline.
        
        Args:
            input_files: List of input file paths
            output_dir: Output directory for results and model
            model_name: Name for the trained model
            
        Returns:
            BootstrapResult with pipeline execution results
        """
        start_time = time.time()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting ISNE Bootstrap Pipeline")
        logger.info(f"Input files: {len(input_files)}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Model name: {model_name}")
        
        # Initialize monitoring
        if self.monitor is None:
            self.monitor = BootstrapMonitor(
                pipeline_name="isne_bootstrap",
                output_dir=output_dir,
                enable_alerts=self.config.enable_monitoring
            )
        
        stage_results = {}
        current_data = [Path(f) for f in input_files]
        
        # Start W&B run
        run_config = {
            'input_files': len(input_files),
            'pipeline_config': self.config.__dict__
        }
        self.wandb_logger.start_run(run_config, model_name)
        
        try:
            # Execute stages sequentially
            for stage_name in self.stage_order:
                logger.info(f"\n{'='*60}")
                logger.info(f"STAGE: {stage_name.upper()}")
                logger.info(f"{'='*60}")
                
                stage = self.stages[stage_name]
                stage_config = getattr(self.config, stage_name)
                
                # Start stage monitoring
                stage_metrics = self.monitor.start_stage(stage_name)
                
                try:
                    # Stage-specific execution
                    if stage_name == 'document_processing':
                        result = self._run_document_processing(stage, current_data, stage_config)
                        current_data = result.documents
                        
                    elif stage_name == 'chunking':
                        result = self._run_chunking(stage, current_data, stage_config)
                        current_data = result.chunks
                        
                    elif stage_name == 'embedding':
                        result = self._run_embedding(stage, current_data, stage_config)
                        current_data = result.embeddings
                        
                    elif stage_name == 'graph_construction':
                        result = self._run_graph_construction(stage, current_data, stage_config, output_dir)
                        current_data = result.graph_data
                        
                    elif stage_name == 'isne_training':
                        result = self._run_isne_training(stage, current_data, stage_config, output_dir)
                        
                    elif stage_name == 'model_evaluation':
                        result = self._run_model_evaluation(stage, current_data, stage_config, output_dir, stage_results)
                        
                    # Check stage success
                    if not result.success:
                        error_msg = f"Stage {stage_name} failed: {result.error_message}"
                        logger.error(error_msg)
                        self.monitor.complete_stage(stage_name, False, error_message=result.error_message)
                        
                        return BootstrapResult(
                            success=False,
                            model_path=None,
                            output_directory=str(output_dir),
                            total_time_seconds=time.time() - start_time,
                            stage_results=stage_results,
                            final_stats={},
                            error_message=error_msg,
                            error_stage=stage_name
                        )
                    
                    # Store stage result
                    stage_stats = result.stats if hasattr(result, 'stats') else {}
                    self.monitor.complete_stage(stage_name, True, stats=stage_stats)
                    
                    stage_results[stage_name] = {
                        'success': result.success,
                        'stats': stage_stats,
                        'execution_time': stage_metrics.duration_seconds if stage_metrics.duration_seconds else 0
                    }
                    
                    # Add stage-specific outputs to the results
                    if hasattr(result, 'model_path') and result.model_path:
                        stage_results[stage_name]['model_path'] = result.model_path
                    if hasattr(result, 'embedding_results') and result.embedding_results:
                        stage_results[stage_name]['embedding_results'] = result.embedding_results
                    
                    # Log to W&B
                    self.wandb_logger.log_stage_metrics(stage_name, stage_stats)
                    if stage_metrics.duration_seconds:
                        self.wandb_logger.log_stage_timing(stage_name, stage_metrics.duration_seconds, stage_stats)
                    
                    # Save intermediate results if configured
                    if self.config.save_intermediate_results:
                        self._save_intermediate_result(output_dir, stage_name, result)
                    
                    logger.info(f"✓ Stage {stage_name} completed successfully")
                    
                except Exception as e:
                    error_msg = f"Stage {stage_name} execution failed: {e}"
                    logger.error(error_msg, exc_info=True)
                    self.monitor.complete_stage(stage_name, False, error_message=str(e))
                    
                    return BootstrapResult(
                        success=False,
                        model_path=None,
                        output_directory=str(output_dir),
                        total_time_seconds=time.time() - start_time,
                        stage_results=stage_results,
                        final_stats={},
                        error_message=error_msg,
                        error_stage=stage_name
                    )
            
            # Pipeline completed successfully
            total_time = time.time() - start_time
            self.monitor.complete_pipeline(success=True)
            
            # Get final model path from training stage
            model_path = None
            if 'isne_training' in stage_results:
                model_path = stage_results['isne_training'].get('model_path')
            
            # Compile final statistics
            final_stats = self._compile_final_stats(stage_results, total_time)
            
            # Save pipeline summary
            self._save_pipeline_summary(output_dir, stage_results, final_stats, model_name)
            
            # Log to W&B
            self.wandb_logger.log_pipeline_summary(final_stats, stage_results)
            
            # Log model artifact if training was successful
            if model_path and Path(model_path).exists():
                model_config = stage_results.get('isne_training', {}).get('stats', {})
                self.wandb_logger.log_model_artifact(model_path, model_config)
            
            # Log evaluation results if available
            if 'model_evaluation' in stage_results and stage_results['model_evaluation'].get('success'):
                eval_result = stage_results['model_evaluation']
                if hasattr(eval_result, 'evaluation_metrics'):
                    self.wandb_logger.log_evaluation_results(eval_result.evaluation_metrics)
                elif 'evaluation_metrics' in eval_result.get('stats', {}):
                    self.wandb_logger.log_evaluation_results(eval_result['stats']['evaluation_metrics'])
            
            # Finish W&B run
            self.wandb_logger.finish_run(success=True)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"ISNE BOOTSTRAP PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"{'='*60}")
            logger.info(f"Total time: {total_time:.2f} seconds")
            logger.info(f"Model saved to: {model_path}")
            logger.info(f"Output directory: {output_dir}")
            
            return BootstrapResult(
                success=True,
                model_path=model_path,
                output_directory=str(output_dir),
                total_time_seconds=total_time,
                stage_results=stage_results,
                final_stats=final_stats
            )
            
        except Exception as e:
            error_msg = f"Pipeline execution failed: {e}"
            logger.error(error_msg, exc_info=True)
            self.monitor.complete_pipeline(success=False)
            
            # Finish W&B run with failure
            self.wandb_logger.finish_run(success=False)
            
            return BootstrapResult(
                success=False,
                model_path=None,
                output_directory=str(output_dir),
                total_time_seconds=time.time() - start_time,
                stage_results=stage_results,
                final_stats={},
                error_message=error_msg
            )
    
    def _run_document_processing(self, stage, input_files, config):
        """Run document processing stage."""
        # Validate inputs
        validation_errors = stage.validate_inputs(input_files, config)
        if validation_errors:
            raise ValueError(f"Document processing validation failed: {validation_errors}")
        
        # Execute stage
        return stage.execute(input_files, config)
    
    def _run_chunking(self, stage, documents, config):
        """Run chunking stage."""
        # Validate inputs
        validation_errors = stage.validate_inputs(documents, config)
        if validation_errors:
            raise ValueError(f"Chunking validation failed: {validation_errors}")
        
        # Execute stage
        return stage.execute(documents, config)
    
    def _run_embedding(self, stage, chunks, config):
        """Run embedding stage."""
        # Validate inputs
        validation_errors = stage.validate_inputs(chunks, config)
        if validation_errors:
            raise ValueError(f"Embedding validation failed: {validation_errors}")
        
        # Execute stage
        return stage.execute(chunks, config)
    
    def _run_graph_construction(self, stage, embeddings, config, output_dir):
        """Run graph construction stage."""
        # Validate inputs
        validation_errors = stage.validate_inputs(embeddings, config)
        if validation_errors:
            raise ValueError(f"Graph construction validation failed: {validation_errors}")
        
        # Execute stage
        return stage.execute(embeddings, config, output_dir)
    
    def _run_isne_training(self, stage, graph_data, config, output_dir):
        """Run ISNE training stage."""
        # Validate inputs
        validation_errors = stage.validate_inputs(graph_data, config, output_dir)
        if validation_errors:
            raise ValueError(f"ISNE training validation failed: {validation_errors}")
        
        # Execute stage
        return stage.execute(graph_data, config, output_dir)
    
    def _run_model_evaluation(self, stage, graph_data, config, output_dir, stage_results):
        """Run model evaluation stage."""
        # Get model path from training stage results
        training_result = stage_results.get('isne_training', {})
        training_success = training_result.get('success', False) if isinstance(training_result, dict) else getattr(training_result, 'success', False)
        
        if 'isne_training' not in stage_results or not training_success:
            raise ValueError("Model evaluation requires successful ISNE training stage")
        
        model_path = training_result.get('model_path') if isinstance(training_result, dict) else getattr(training_result, 'model_path', None)
        if not model_path:
            raise ValueError("Model path not found in training stage results")
        
        # Get graph data path - should be saved during graph construction
        graph_data_path = output_dir / "graph_data.json"
        if not graph_data_path.exists():
            raise ValueError(f"Graph data file not found: {graph_data_path}")
        
        # Validate inputs
        validation_errors = stage.validate_inputs(str(model_path), str(graph_data_path), config)
        if validation_errors:
            raise ValueError(f"Model evaluation validation failed: {validation_errors}")
        
        # Execute stage
        return stage.run(str(model_path), str(graph_data_path), output_dir, config)
    
    def _save_intermediate_result(self, output_dir: Path, stage_name: str, result: Any):
        """Save intermediate stage result."""
        try:
            intermediate_dir = output_dir / "intermediate"
            intermediate_dir.mkdir(exist_ok=True)
            
            result_file = intermediate_dir / f"{stage_name}_result.json"
            
            # Convert result to serializable format
            if hasattr(result, '__dict__'):
                result_data = {}
                for key, value in result.__dict__.items():
                    if key in ['documents', 'chunks', 'embeddings', 'graph_data']:
                        # Save large data separately
                        data_file = intermediate_dir / f"{stage_name}_{key}.json"
                        if isinstance(value, list):
                            # Handle list of objects
                            serializable_data = []
                            for item in value:
                                if hasattr(item, '__dict__'):
                                    serializable_data.append(item.__dict__)
                                else:
                                    serializable_data.append(str(item))
                            with open(data_file, 'w') as f:
                                json.dump(serializable_data, f, indent=2)
                        else:
                            with open(data_file, 'w') as f:
                                json.dump(value, f, indent=2)
                        result_data[key] = f"saved_to_{data_file.name}"
                    else:
                        result_data[key] = value
                
                with open(result_file, 'w') as f:
                    json.dump(result_data, f, indent=2, default=str)
                    
                logger.debug(f"Saved intermediate result for {stage_name}")
                
        except Exception as e:
            logger.warning(f"Failed to save intermediate result for {stage_name}: {e}")
    
    def _compile_final_stats(self, stage_results: Dict[str, Any], total_time: float) -> Dict[str, Any]:
        """Compile final pipeline statistics."""
        stats = {
            'pipeline_success': True,
            'total_execution_time_seconds': total_time,
            'total_execution_time_formatted': self._format_duration(total_time),
            'stages_completed': len(stage_results),
            'stage_timings': {},
            'data_flow': {}
        }
        
        # Extract stage timings and data flow
        for stage_name, stage_result in stage_results.items():
            stats['stage_timings'][stage_name] = stage_result.get('execution_time', 0)
            
            # Extract key metrics from each stage
            stage_stats = stage_result.get('stats', {})
            if stage_name == 'document_processing':
                stats['data_flow']['input_files'] = stage_stats.get('files_processed', 0)
                stats['data_flow']['documents_generated'] = stage_stats.get('documents_generated', 0)
                
            elif stage_name == 'chunking':
                stats['data_flow']['chunks_generated'] = stage_stats.get('output_chunks', 0)
                stats['data_flow']['avg_chunk_size'] = stage_stats.get('avg_chunk_size', 0)
                
            elif stage_name == 'embedding':
                stats['data_flow']['embeddings_generated'] = stage_stats.get('output_embeddings', 0)
                stats['data_flow']['embedding_dimension'] = stage_stats.get('embedding_dimension', 0)
                
            elif stage_name == 'graph_construction':
                stats['data_flow']['graph_nodes'] = stage_stats.get('num_nodes', 0)
                stats['data_flow']['graph_edges'] = stage_stats.get('num_edges', 0)
                stats['data_flow']['graph_density'] = stage_stats.get('graph_density', 0)
                
            elif stage_name == 'isne_training':
                stats['data_flow']['model_parameters'] = stage_stats.get('model_parameters', 0)
                stats['data_flow']['training_epochs'] = stage_stats.get('epochs_trained', 0)
                stats['data_flow']['final_loss'] = stage_stats.get('final_loss', 0)
                
            elif stage_name == 'model_evaluation':
                stats['data_flow']['evaluation_completed'] = stage_stats.get('evaluation_completed', False)
                stats['data_flow']['inductive_performance'] = stage_stats.get('inductive_performance_percent', 0)
                stats['data_flow']['achieves_90_percent_target'] = stage_stats.get('achieves_90_percent_target', False)
                stats['data_flow']['overall_evaluation_status'] = stage_stats.get('overall_status', 'unknown')
        
        return stats
    
    def _save_pipeline_summary(self, output_dir: Path, stage_results: Dict[str, Any], 
                              final_stats: Dict[str, Any], model_name: str):
        """Save complete pipeline execution summary."""
        summary = {
            'pipeline_name': 'isne_bootstrap',
            'model_name': model_name,
            'execution_timestamp': time.time(),
            'execution_time_formatted': time.strftime('%Y-%m-%d %H:%M:%S'),
            'configuration': self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config),
            'stage_results': stage_results,
            'final_statistics': final_stats,
            'success': True
        }
        
        summary_file = output_dir / "bootstrap_pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Also save a human-readable summary
        readable_summary_file = output_dir / "bootstrap_pipeline_summary.txt"
        with open(readable_summary_file, 'w') as f:
            f.write("ISNE Bootstrap Pipeline Execution Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model Name: {model_name}\n")
            f.write(f"Execution Time: {final_stats['total_execution_time_formatted']}\n")
            f.write(f"Total Duration: {final_stats['total_execution_time_seconds']:.2f} seconds\n\n")
            
            f.write("Data Flow:\n")
            f.write("-" * 20 + "\n")
            for key, value in final_stats['data_flow'].items():
                f.write(f"{key}: {value}\n")
            
            f.write("\nStage Timings:\n")
            f.write("-" * 20 + "\n")
            for stage, timing in final_stats['stage_timings'].items():
                f.write(f"{stage}: {timing:.2f}s\n")
        
        logger.info(f"Pipeline summary saved to {summary_file}")
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"
    
    def estimate_total_duration(self, num_input_files: int) -> float:
        """
        Estimate total pipeline duration.
        
        Args:
            num_input_files: Number of input files
            
        Returns:
            Estimated duration in seconds
        """
        total_estimate = 0
        
        # Get estimates from each stage
        for stage_name in self.stage_order:
            stage = self.stages[stage_name]
            if stage_name == 'document_processing':
                estimate = stage.estimate_duration(num_input_files)
            else:
                # Use heuristics for other stages based on expected data growth
                if stage_name == 'chunking':
                    estimate = stage.estimate_duration(num_input_files * 5)  # ~5 docs per file
                elif stage_name == 'embedding':
                    estimate = stage.estimate_duration(num_input_files * 50)  # ~50 chunks per file
                elif stage_name == 'graph_construction':
                    estimate = stage.estimate_duration(num_input_files * 50)  # ~50 nodes per file
                elif stage_name == 'isne_training':
                    estimate = stage.estimate_duration(num_input_files * 50)  # ~50 nodes per file
                else:
                    estimate = 60  # Default estimate
            
            total_estimate += estimate
            logger.debug(f"Stage {stage_name} estimated duration: {estimate:.2f}s")
        
        logger.info(f"Total estimated duration: {total_estimate:.2f}s ({self._format_duration(total_estimate)})")
        return total_estimate
    
    def get_resource_requirements(self, num_input_files: int) -> Dict[str, Any]:
        """
        Get estimated resource requirements for the entire pipeline.
        
        Args:
            num_input_files: Number of input files
            
        Returns:
            Dictionary with resource estimates
        """
        max_memory = 0
        max_cpu_cores = 0
        total_disk = 0
        gpu_memory = 0
        network_required = False
        
        # Get requirements from each stage and take maximums/sums as appropriate
        for stage_name in self.stage_order:
            stage = self.stages[stage_name]
            if stage_name == 'document_processing':
                reqs = stage.get_resource_requirements(num_input_files)
            else:
                # Use heuristics for data size growth
                if stage_name == 'chunking':
                    reqs = stage.get_resource_requirements(num_input_files * 5)
                elif stage_name == 'embedding':
                    reqs = stage.get_resource_requirements(num_input_files * 50)
                elif stage_name == 'graph_construction':
                    reqs = stage.get_resource_requirements(num_input_files * 50)
                elif stage_name == 'isne_training':
                    reqs = stage.get_resource_requirements(num_input_files * 50)
                else:
                    reqs = {'memory_mb': 500, 'cpu_cores': 1, 'disk_mb': 100}
            
            # Take maximum memory and CPU requirements
            max_memory = max(max_memory, reqs.get('memory_mb', 0))
            max_cpu_cores = max(max_cpu_cores, reqs.get('cpu_cores', 1))
            total_disk += reqs.get('disk_mb', 0)
            gpu_memory = max(gpu_memory, reqs.get('gpu_memory_mb', 0))
            network_required = network_required or reqs.get('network_required', False)
        
        return {
            'memory_mb': max_memory,
            'cpu_cores': max_cpu_cores,
            'disk_mb': total_disk,
            'gpu_memory_mb': gpu_memory,
            'network_required': network_required,
            'recommended_specs': {
                'ram': f"{max_memory / 1024:.1f} GB",
                'storage': f"{total_disk / 1024:.1f} GB",
                'gpu_memory': f"{gpu_memory / 1024:.1f} GB" if gpu_memory > 0 else "Not required"
            }
        }