"""
ISNE Training Stage for ISNE Bootstrap Pipeline

Handles training of Inductive Shallow Node Embedding models using 
graph data from the graph construction stage.
"""

import logging
import traceback
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from src.isne.models.isne_model import ISNEModel
from src.isne.training.trainer import ISNETrainer
from src.isne.utils.geometric_utils import documents_to_graph_data, TORCH_GEOMETRIC_AVAILABLE
from .base import BaseBootstrapStage

logger = logging.getLogger(__name__)


@dataclass
class ISNETrainingResult:
    """Result of ISNE training stage."""
    success: bool
    model_path: Optional[str]  # Path to saved model
    training_stats: Dict[str, Any]
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None


class ISNETrainingStage(BaseBootstrapStage):
    """ISNE training stage for bootstrap pipeline."""
    
    def __init__(self):
        """Initialize ISNE training stage."""
        super().__init__("isne_training")
        
    def execute(self, graph_data: Dict[str, Any], config: Any, output_dir: Path) -> ISNETrainingResult:
        """
        Execute ISNE training stage.
        
        Args:
            graph_data: Graph data from graph construction stage
            config: ISNETrainingConfig object
            output_dir: Directory to save trained model
            
        Returns:
            ISNETrainingResult with model path and training stats
        """
        logger.info("Starting ISNE training stage")
        
        try:
            # Extract graph components
            nodes = graph_data['nodes']
            edges = graph_data['edges']
            graph_metadata = graph_data.get('metadata', {})
            
            logger.info(f"Training on graph with {len(nodes)} nodes and {len(edges)} edges")
            
            # Prepare training data
            training_start = time.time()
            logger.info("Preparing training data...")
            
            # Extract node features (embeddings)
            node_features = []
            node_ids = []
            for node in nodes:
                node_features.append(node['embedding'])
                node_ids.append(node['id'])
            
            node_features = torch.tensor(node_features, dtype=torch.float32)
            
            # Create edge index for PyTorch Geometric
            edge_index = []
            edge_weights = []
            for edge in edges:
                edge_index.append([edge['source'], edge['target']])
                edge_weights.append(edge['weight'])
            
            if edge_index:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_weights = torch.tensor(edge_weights, dtype=torch.float32)
            else:
                # Create empty edge tensors if no edges
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_weights = torch.empty(0, dtype=torch.float32)
            
            # Create PyTorch Geometric data object
            try:
                if TORCH_GEOMETRIC_AVAILABLE:
                    from torch_geometric.data import Data
                    data = Data(
                        x=node_features,
                        edge_index=edge_index,
                        edge_attr=edge_weights.unsqueeze(1) if len(edge_weights) > 0 else None
                    )
                else:
                    raise ImportError("PyTorch Geometric not available")
            except Exception as e:
                logger.error(f"Failed to create PyTorch Geometric data: {e}")
                raise
            
            logger.info(f"Created training data: {data}")
            
            # Set device
            device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
            data = data.to(device)
            
            # Initialize trainer with corrected ISNE model
            input_dim = node_features.shape[1]
            num_nodes = len(nodes)
            
            trainer = ISNETrainer(
                embedding_dim=input_dim,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                device=device
            )
            
            # Initialize the model with correct parameters
            trainer.prepare_model(num_nodes=num_nodes)
            
            logger.info(f"Trainer initialized on device: {device}")
            logger.info(f"ISNE model initialized with {num_nodes} nodes, {input_dim}D embeddings")
            
            if trainer.model is not None:
                model = trainer.model
                logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            else:
                logger.error("Model not found in trainer after initialization")
                raise ValueError("Trainer model initialization failed")
            
            # Training loop
            logger.info(f"Starting training for {config.epochs} epochs...")
            training_history = {
                'losses': [],
                'feature_losses': [],
                'structural_losses': [],
                'contrastive_losses': [],
                'epoch_times': []
            }
            
            best_loss = float('inf')
            patience_counter = 0
            
            # Use the simple ISNE training method
            logger.info("Starting ISNE model training...")
            
            # Run simple ISNE training with skip-gram objective
            train_stats = trainer.train_isne_simple(
                edge_index=data.edge_index,
                epochs=config.epochs,
                batch_size=config.batch_size,
                verbose=True
            )
            
            # Extract training statistics from simple ISNE training
            training_history['losses'] = train_stats.get('losses', [])
            training_history['feature_losses'] = []  # Not used in simple ISNE
            training_history['structural_losses'] = []  # Not used in simple ISNE
            training_history['contrastive_losses'] = []  # Not used in simple ISNE
            training_history['epoch_times'] = train_stats.get('time_per_epoch', [])
            
            # Calculate final statistics from training
            if training_history['losses']:
                best_loss = min(training_history['losses'])
            else:
                best_loss = float('inf')
            
            total_training_time = time.time() - training_start
            
            # Save model
            output_dir.mkdir(parents=True, exist_ok=True)
            model_path = output_dir / f"isne_model_final.pth"
            
            # Save model state dict with metadata
            model_save_data = {
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'num_nodes': num_nodes,
                    'embedding_dim': input_dim,
                    'learning_rate': config.learning_rate,
                    'device': str(device)
                },
                'training_config': {
                    'learning_rate': config.learning_rate,
                    'weight_decay': config.weight_decay,
                    'epochs': config.epochs,
                    'batch_size': config.batch_size,
                    'device': str(device)
                },
                'graph_metadata': graph_metadata,
                'training_stats': {
                    'final_loss': training_history['losses'][-1] if training_history['losses'] else None,
                    'best_loss': best_loss,
                    'epochs_trained': len(training_history['losses']),
                    'total_training_time': total_training_time,
                    'avg_epoch_time': np.mean(training_history['epoch_times']) if training_history['epoch_times'] else 0
                }
            }
            
            torch.save(model_save_data, model_path)
            
            # Save training history
            history_path = output_dir / "training_history.json"
            import json
            with open(history_path, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                history_json = {}
                for key, values in training_history.items():
                    if isinstance(values, list):
                        history_json[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in values]
                    else:
                        history_json[key] = float(values) if isinstance(values, (np.floating, np.integer)) else values
                json.dump(history_json, f, indent=2)
            
            # Calculate training statistics
            stats = {
                'model_path': str(model_path),
                'training_completed': True,
                'epochs_trained': len(training_history['losses']),
                'final_loss': training_history['losses'][-1] if training_history['losses'] else None,
                'best_loss': best_loss,
                'total_training_time_seconds': total_training_time,
                'avg_epoch_time_seconds': np.mean(training_history['epoch_times']) if training_history['epoch_times'] else 0,
                'model_parameters': sum(p.numel() for p in model.parameters()),
                'num_nodes': num_nodes,
                'embedding_dimension': input_dim,
                'learning_rate': config.learning_rate,
                'device_used': str(device),
                'graph_stats': {
                    'num_nodes': len(nodes),
                    'num_edges': len(edges),
                    'node_feature_dim': input_dim
                },
                'training_history': training_history,
                'early_stopped': patience_counter >= config.patience,
                'convergence_info': {
                    'patience_used': patience_counter,
                    'patience_limit': config.patience,
                    'loss_improvement_threshold': config.min_delta if hasattr(config, 'min_delta') else 0.0
                }
            }
            
            # Quality checks
            quality_warnings = []
            if training_history['losses'] and len(training_history['losses']) > 10:
                recent_losses = training_history['losses'][-10:]
                loss_std = np.std(recent_losses)
                if loss_std > 0.1:
                    quality_warnings.append(f"High loss variance in final epochs: {loss_std:.4f}")
            
            if training_history['losses'] and training_history['losses'][-1] > training_history['losses'][0]:
                quality_warnings.append("Training loss increased from start to finish")
            
            if quality_warnings:
                stats['quality_warnings'] = quality_warnings
                for warning in quality_warnings:
                    logger.warning(warning)
            
            logger.info(f"ISNE training completed successfully:")
            logger.info(f"  Model saved to: {model_path}")
            logger.info(f"  Epochs trained: {stats['epochs_trained']}")
            final_loss = stats.get('final_loss')
            if final_loss is not None:
                logger.info(f"  Final loss: {final_loss:.6f}")
            else:
                logger.info(f"  Final loss: Not available")
            logger.info(f"  Best loss: {stats['best_loss']:.6f}")
            logger.info(f"  Training time: {total_training_time:.2f}s")
            logger.info(f"  Model parameters: {stats['model_parameters']:,}")
            
            return ISNETrainingResult(
                success=True,
                model_path=str(model_path),
                training_stats=stats
            )
            
        except Exception as e:
            error_msg = f"ISNE training stage failed: {e}"
            error_traceback = traceback.format_exc()
            logger.error(error_msg)
            logger.debug(error_traceback)
            
            return ISNETrainingResult(
                success=False,
                model_path=None,
                training_stats={},
                error_message=error_msg,
                error_traceback=error_traceback
            )
    
    def validate_inputs(self, graph_data: Dict[str, Any], config: Any, output_dir: Path) -> List[str]:
        """
        Validate inputs for ISNE training stage.
        
        Args:
            graph_data: Graph data dictionary
            config: ISNETrainingConfig object
            output_dir: Output directory path
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate graph data structure
        if not isinstance(graph_data, dict):
            errors.append("graph_data must be a dictionary")
            return errors
        
        required_keys = ['nodes', 'edges']
        for key in required_keys:
            if key not in graph_data:
                errors.append(f"graph_data missing required key: {key}")
        
        if 'nodes' in graph_data:
            nodes = graph_data['nodes']
            if not isinstance(nodes, list):
                errors.append("graph_data['nodes'] must be a list")
            elif len(nodes) == 0:
                errors.append("No nodes found in graph_data")
            else:
                # Check node structure
                for i, node in enumerate(nodes[:10]):  # Check first 10 nodes
                    if not isinstance(node, dict):
                        errors.append(f"Node {i} is not a dictionary")
                        continue
                    if 'embedding' not in node:
                        errors.append(f"Node {i} missing 'embedding' field")
                    elif not isinstance(node['embedding'], (list, np.ndarray)):
                        errors.append(f"Node {i} embedding is not a list or array")
                    elif len(node['embedding']) == 0:
                        errors.append(f"Node {i} has empty embedding")
        
        if 'edges' in graph_data:
            edges = graph_data['edges']
            if not isinstance(edges, list):
                errors.append("graph_data['edges'] must be a list")
            else:
                # Check edge structure
                for i, edge in enumerate(edges[:10]):  # Check first 10 edges
                    if not isinstance(edge, dict):
                        errors.append(f"Edge {i} is not a dictionary")
                        continue
                    required_edge_keys = ['source', 'target', 'weight']
                    for key in required_edge_keys:
                        if key not in edge:
                            errors.append(f"Edge {i} missing required key: {key}")
        
        # Validate training configuration
        if config.epochs <= 0:
            errors.append("epochs must be positive")
        
        if config.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        
        if config.hidden_dim <= 0:
            errors.append("hidden_dim must be positive")
        
        if config.num_layers <= 0:
            errors.append("num_layers must be positive")
        
        if config.num_heads <= 0:
            errors.append("num_heads must be positive")
        
        if config.dropout < 0 or config.dropout >= 1:
            errors.append("dropout must be between 0 and 1")
        
        # Validate output directory
        try:
            output_dir = Path(output_dir)
            if output_dir.exists() and not output_dir.is_dir():
                errors.append(f"Output path exists but is not a directory: {output_dir}")
        except Exception as e:
            errors.append(f"Invalid output directory path: {e}")
        
        return errors
    
    def get_expected_outputs(self) -> List[str]:
        """Get list of expected output keys."""
        return ["model_path", "training_stats"]
    
    def estimate_duration(self, input_size: int) -> float:
        """
        Estimate stage duration based on input size.
        
        Args:
            input_size: Number of nodes in the graph
            
        Returns:
            Estimated duration in seconds
        """
        # ISNE training time depends on graph size and number of epochs
        # Conservative estimate: 0.1 seconds per node per epoch
        base_time = 60  # Base overhead
        epochs = 50  # Typical number of epochs
        time_per_node_epoch = 0.01  # Conservative estimate
        training_time = input_size * epochs * time_per_node_epoch
        return max(base_time, training_time)
    
    def get_resource_requirements(self, input_size: int) -> Dict[str, Any]:
        """
        Get estimated resource requirements.
        
        Args:
            input_size: Number of nodes in the graph
            
        Returns:
            Dictionary with resource estimates
        """
        # Memory requirements for ISNE training
        hidden_dim = 128  # Typical hidden dimension
        embedding_dim = 384  # Typical input dimension
        
        # Model memory (parameters)
        model_params = input_size * embedding_dim + hidden_dim * hidden_dim * 3  # Rough estimate
        model_memory = model_params * 4 / 1024 / 1024  # Float32 in MB
        
        # Graph data memory
        graph_memory = input_size * embedding_dim * 4 / 1024 / 1024  # Node features
        
        # Training overhead (gradients, optimizer states)
        training_overhead = model_memory * 3  # Conservative estimate
        
        return {
            "memory_mb": max(2000, model_memory + graph_memory + training_overhead),
            "cpu_cores": 4,  # Benefits from multiple cores
            "disk_mb": 200,  # Model saving
            "network_required": False,
            "gpu_memory_mb": max(1024, model_memory + graph_memory) if input_size > 1000 else 512
        }
    
    def pre_execute_checks(self, graph_data: Dict[str, Any], config: Any, output_dir: Path) -> List[str]:
        """
        Perform pre-execution checks specific to ISNE training stage.
        
        Args:
            graph_data: Graph data dictionary
            config: ISNETrainingConfig object
            output_dir: Output directory path
            
        Returns:
            List of check failure messages
        """
        checks = []
        
        # Check PyTorch and PyTorch Geometric availability
        try:
            import torch
            import torch_geometric
        except ImportError as e:
            checks.append(f"Required PyTorch libraries not available: {e}")
            return checks
        
        # Check CUDA availability if requested
        if hasattr(config, 'device') and config.device.startswith('cuda'):
            if not torch.cuda.is_available():
                checks.append("CUDA requested but not available")
            elif config.device != 'cuda':
                device_num = int(config.device.split(':')[1])
                if device_num >= torch.cuda.device_count():
                    checks.append(f"CUDA device {device_num} not available")
        
        # Check memory requirements
        try:
            import psutil
            available_memory_mb = psutil.virtual_memory().available / 1024 / 1024
            
            num_nodes = len(graph_data.get('nodes', []))
            required_memory = self.get_resource_requirements(num_nodes)["memory_mb"]
            
            if required_memory > available_memory_mb * 0.8:
                checks.append(f"Insufficient memory for ISNE training: need {required_memory:.0f}MB, "
                             f"available {available_memory_mb:.0f}MB")
        except ImportError:
            pass
        
        # Check output directory writability
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            test_file = output_dir / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            checks.append(f"Cannot write to output directory {output_dir}: {e}")
        
        # Check graph size for training feasibility
        num_nodes = len(graph_data.get('nodes', []))
        if num_nodes < 10:
            checks.append(f"Graph too small for meaningful training: {num_nodes} nodes")
        elif num_nodes > 1000000:
            checks.append(f"Graph very large, training may be slow: {num_nodes} nodes")
        
        return checks