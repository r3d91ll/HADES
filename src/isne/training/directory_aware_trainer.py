"""
Directory-aware ISNE trainer that leverages filesystem structure.

This module implements the training pipeline for ISNE models that incorporate
directory structure and co-location relationships.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm

from src.isne.models.directory_aware_isne import DirectoryAwareISNE
from src.isne.training.hierarchical_batch_sampler import HierarchicalBatchSampler
from src.types.isne.models import ISNEConfig
from src.types.isne.training import DirectoryMetadata, TrainingMetrics
from src.validation.embedding_validator import EmbeddingValidator


logger = logging.getLogger(__name__)


class DirectoryAwareISNETrainer:
    """
    Trainer for directory-aware ISNE models.
    
    This trainer implements:
    - Hierarchical batch sampling
    - Directory-aware loss computation
    - Validation with directory metrics
    - Debug mode with intermediate outputs
    """
    
    def __init__(
        self,
        config: ISNEConfig,
        graph_data: Data,
        directory_metadata: Dict[int, DirectoryMetadata],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        debug_mode: bool = False,
        debug_output_dir: Optional[Path] = None
    ):
        """
        Initialize the directory-aware trainer.
        
        Args:
            config: ISNE model configuration
            graph_data: PyTorch Geometric graph data
            directory_metadata: Mapping from node ID to directory metadata
            device: Device to train on
            debug_mode: Enable debug outputs
            debug_output_dir: Directory for debug files
        """
        self.config = config
        self.graph_data = graph_data.to(device)
        self.directory_metadata = directory_metadata
        self.device = device
        self.debug_mode = debug_mode
        self.debug_output_dir = Path(debug_output_dir) if debug_output_dir else None
        
        # Initialize model
        self.model = DirectoryAwareISNE(
            config=config,
            directory_embedding_dim=64,
            use_directory_features=True,
            directory_loss_weight=0.5,
            debug_mode=debug_mode
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=5,
            factor=0.5
        )
        
        # Initialize batch sampler
        self.batch_sampler = HierarchicalBatchSampler(
            graph_data=self.graph_data,
            directory_metadata=directory_metadata,
            batch_size=config.batch_size,
            same_dir_weight=2.0,
            negative_sampling_ratio=5,
            debug_mode=debug_mode,
            debug_output_dir=debug_output_dir
        )
        
        # Initialize validator
        self.validator = EmbeddingValidator()
        
        # Training history
        self.training_history: List[TrainingMetrics] = []
        
        if self.debug_mode and self.debug_output_dir:
            self.debug_output_dir.mkdir(parents=True, exist_ok=True)
            self._save_debug_config()
    
    def train(
        self,
        num_epochs: int,
        validation_interval: int = 5,
        checkpoint_dir: Optional[Path] = None,
        early_stopping_patience: int = 10
    ) -> DirectoryAwareISNE:
        """
        Train the directory-aware ISNE model.
        
        Args:
            num_epochs: Number of training epochs
            validation_interval: Epochs between validation
            checkpoint_dir: Directory to save checkpoints
            early_stopping_patience: Epochs without improvement before stopping
            
        Returns:
            Trained model
        """
        logger.info(f"Starting directory-aware ISNE training for {num_epochs} epochs")
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            # Training epoch
            train_metrics = self._train_epoch(epoch)
            self.training_history.append(train_metrics)
            
            # Validation
            if epoch % validation_interval == 0:
                val_metrics = self._validate(epoch)
                
                # Learning rate scheduling
                self.scheduler.step(val_metrics.total_loss)
                
                # Early stopping check
                if val_metrics.total_loss < best_val_loss:
                    best_val_loss = val_metrics.total_loss
                    epochs_without_improvement = 0
                    
                    # Save best model
                    if checkpoint_dir:
                        self._save_checkpoint(epoch, checkpoint_dir, is_best=True)
                else:
                    epochs_without_improvement += 1
                
                if epochs_without_improvement >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Regular checkpoint
            if checkpoint_dir and epoch % 10 == 0:
                self._save_checkpoint(epoch, checkpoint_dir, is_best=False)
            
            # Log progress
            self._log_progress(epoch, train_metrics)
        
        logger.info("Training completed")
        
        if self.debug_mode:
            self._save_training_summary()
        
        return self.model
    
    def _train_epoch(self, epoch: int) -> TrainingMetrics:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {
            'total': [],
            'feature_reconstruction': [],
            'structural': [],
            'contrastive': [],
            'directory_coherence': []
        }
        
        # Sample batches for this epoch
        num_batches = max(1, self.graph_data.num_nodes // self.config.batch_size)
        batches = self.batch_sampler.sample(num_batches)
        
        progress_bar = tqdm(batches, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            # Move batch to device
            batch_nodes = batch.node_ids.to(self.device)
            batch_features = batch.features.to(self.device)
            batch_edges = batch.edge_index.to(self.device)
            
            # Prepare directory features
            directory_features = {
                k: v.to(self.device) 
                for k, v in batch.directory_features.items()
            }
            
            # Create directory labels for coherence loss
            directory_labels = self._create_directory_labels(batch_nodes)
            
            # Forward pass
            embeddings, auxiliary_outputs = self.model(
                x=batch_features,
                edge_index=batch_edges,
                directory_features=directory_features
            )
            
            # Compute loss
            batch_data = {
                'features': batch_features,
                'edge_index': batch_edges,
                'node_ids': batch_nodes
            }
            
            loss, loss_components = self.model.compute_loss(
                embeddings=embeddings,
                auxiliary_outputs=auxiliary_outputs,
                batch_data=batch_data,
                directory_labels=directory_labels
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Record losses
            for key, value in loss_components.items():
                if key in epoch_losses:
                    epoch_losses[key].append(value.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dir_loss': f"{loss_components.get('directory_coherence', 0):.4f}"
            })
        
        # Compute epoch metrics
        metrics = TrainingMetrics(
            epoch=epoch,
            total_loss=np.mean(epoch_losses['total']),
            loss_components={
                key: np.mean(values) if values else 0.0
                for key, values in epoch_losses.items()
            },
            learning_rate=self.optimizer.param_groups[0]['lr']
        )
        
        if self.debug_mode:
            self._save_epoch_debug(epoch, metrics, epoch_losses)
        
        return metrics
    
    def _validate(self, epoch: int) -> TrainingMetrics:
        """Validate the model."""
        self.model.eval()
        
        val_losses = {
            'total': [],
            'directory_coherence': []
        }
        
        # Directory-aware metrics
        intra_dir_similarities = []
        inter_dir_distances = []
        
        with torch.no_grad():
            # Sample validation batches
            num_val_batches = max(1, self.graph_data.num_nodes // (self.config.batch_size * 4))
            val_batches = self.batch_sampler.sample(num_val_batches)
            
            for batch in val_batches:
                # Move batch to device
                batch_nodes = batch.node_ids.to(self.device)
                batch_features = batch.features.to(self.device)
                batch_edges = batch.edge_index.to(self.device)
                
                directory_features = {
                    k: v.to(self.device) 
                    for k, v in batch.directory_features.items()
                }
                
                directory_labels = self._create_directory_labels(batch_nodes)
                
                # Forward pass
                embeddings, auxiliary_outputs = self.model(
                    x=batch_features,
                    edge_index=batch_edges,
                    directory_features=directory_features
                )
                
                # Compute validation metrics
                batch_data = {
                    'features': batch_features,
                    'edge_index': batch_edges,
                    'node_ids': batch_nodes
                }
                
                loss, loss_components = self.model.compute_loss(
                    embeddings=embeddings,
                    auxiliary_outputs=auxiliary_outputs,
                    batch_data=batch_data,
                    directory_labels=directory_labels
                )
                
                val_losses['total'].append(loss.item())
                if 'directory_coherence' in loss_components:
                    val_losses['directory_coherence'].append(
                        loss_components['directory_coherence'].item()
                    )
                
                # Compute directory-aware metrics
                intra_sim, inter_dist = self._compute_directory_metrics(
                    embeddings, 
                    directory_labels
                )
                intra_dir_similarities.append(intra_sim)
                inter_dir_distances.append(inter_dist)
        
        # Create validation metrics
        val_metrics = TrainingMetrics(
            epoch=epoch,
            total_loss=np.mean(val_losses['total']),
            loss_components={
                'directory_coherence': np.mean(val_losses.get('directory_coherence', [0]))
            },
            learning_rate=self.optimizer.param_groups[0]['lr'],
            validation_metrics={
                'intra_directory_similarity': np.mean(intra_dir_similarities),
                'inter_directory_distance': np.mean(inter_dir_distances)
            }
        )
        
        logger.info(
            f"Validation - Epoch {epoch}: "
            f"Loss={val_metrics.total_loss:.4f}, "
            f"Intra-dir sim={val_metrics.validation_metrics['intra_directory_similarity']:.4f}, "
            f"Inter-dir dist={val_metrics.validation_metrics['inter_directory_distance']:.4f}"
        )
        
        return val_metrics
    
    def _create_directory_labels(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Create directory labels for nodes."""
        labels = []
        
        # Create a mapping of directory paths to IDs
        unique_dirs = {}
        dir_counter = 0
        
        for node_id in node_ids:
            node_id = node_id.item()
            metadata = self.directory_metadata.get(node_id)
            
            if metadata:
                dir_path = metadata.directory_path
                if dir_path not in unique_dirs:
                    unique_dirs[dir_path] = dir_counter
                    dir_counter += 1
                labels.append(unique_dirs[dir_path])
            else:
                labels.append(-1)  # Unknown directory
        
        return torch.tensor(labels, device=self.device)
    
    def _compute_directory_metrics(
        self,
        embeddings: torch.Tensor,
        directory_labels: torch.Tensor
    ) -> Tuple[float, float]:
        """Compute directory-aware evaluation metrics."""
        # Remove nodes with unknown directories
        valid_mask = directory_labels >= 0
        if not valid_mask.any():
            return 0.0, 0.0
        
        valid_embeddings = embeddings[valid_mask]
        valid_labels = directory_labels[valid_mask]
        
        # Compute pairwise similarities
        similarities = torch.mm(valid_embeddings, valid_embeddings.t())
        
        # Create masks
        label_matrix = valid_labels.unsqueeze(0).repeat(len(valid_labels), 1)
        same_dir_mask = (label_matrix == label_matrix.t()).float()
        diff_dir_mask = 1.0 - same_dir_mask
        
        # Remove diagonal
        eye = torch.eye(len(valid_labels), device=embeddings.device)
        same_dir_mask = same_dir_mask * (1 - eye)
        
        # Compute metrics
        intra_similarities = similarities * same_dir_mask
        inter_similarities = similarities * diff_dir_mask
        
        intra_sim = intra_similarities.sum() / (same_dir_mask.sum() + 1e-8)
        inter_sim = inter_similarities.sum() / (diff_dir_mask.sum() + 1e-8)
        
        # Convert similarity to distance for inter-directory
        inter_dist = 1.0 - inter_sim
        
        return intra_sim.item(), inter_dist.item()
    
    def _save_checkpoint(
        self, 
        epoch: int, 
        checkpoint_dir: Path, 
        is_best: bool = False
    ) -> None:
        """Save model checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }
        
        filename = 'best_model.pt' if is_best else f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_dir / filename)
        
        logger.info(f"Saved checkpoint: {filename}")
    
    def _log_progress(self, epoch: int, metrics: TrainingMetrics) -> None:
        """Log training progress."""
        logger.info(
            f"Epoch {epoch}: "
            f"Loss={metrics.total_loss:.4f}, "
            f"Dir Loss={metrics.loss_components.get('directory_coherence', 0):.4f}, "
            f"LR={metrics.learning_rate:.6f}"
        )
    
    def _save_debug_config(self) -> None:
        """Save configuration for debugging."""
        if not self.debug_output_dir:
            return
        
        debug_dir = self.debug_output_dir / "directory_aware_trainer"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        config_data = {
            "component": "directory_aware_trainer",
            "timestamp": datetime.now().isoformat(),
            "stage": "initialization",
            "data": {
                "model_config": self.config.__dict__,
                "num_nodes": self.graph_data.num_nodes,
                "num_edges": self.graph_data.edge_index.shape[1],
                "num_directories": len(set(
                    m.directory_path for m in self.directory_metadata.values()
                )),
                "device": str(self.device)
            }
        }
        
        with open(debug_dir / f"{config_data['timestamp']}_config.json", 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def _save_epoch_debug(
        self, 
        epoch: int, 
        metrics: TrainingMetrics,
        epoch_losses: Dict[str, List[float]]
    ) -> None:
        """Save debug information for epoch."""
        if not self.debug_output_dir:
            return
        
        debug_dir = self.debug_output_dir / "directory_aware_trainer" / "epochs"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        debug_data = {
            "component": "directory_aware_trainer",
            "timestamp": datetime.now().isoformat(),
            "stage": "training",
            "epoch": epoch,
            "data": {
                "metrics": {
                    "total_loss": metrics.total_loss,
                    "loss_components": metrics.loss_components,
                    "learning_rate": metrics.learning_rate
                },
                "loss_history": {
                    key: {
                        "mean": np.mean(values) if values else 0,
                        "std": np.std(values) if values else 0,
                        "min": np.min(values) if values else 0,
                        "max": np.max(values) if values else 0
                    }
                    for key, values in epoch_losses.items()
                }
            },
            "metadata": {
                "num_batches": len(epoch_losses['total']),
                "model_debug_outputs": self.model.get_debug_outputs()[-10:]  # Last 10
            }
        }
        
        with open(debug_dir / f"epoch_{epoch:04d}.json", 'w') as f:
            json.dump(debug_data, f, indent=2)
    
    def _save_training_summary(self) -> None:
        """Save training summary."""
        if not self.debug_output_dir:
            return
        
        debug_dir = self.debug_output_dir / "directory_aware_trainer"
        
        summary_data = {
            "component": "directory_aware_trainer",
            "timestamp": datetime.now().isoformat(),
            "stage": "summary",
            "data": {
                "total_epochs": len(self.training_history),
                "final_loss": self.training_history[-1].total_loss if self.training_history else 0,
                "best_loss": min(m.total_loss for m in self.training_history) if self.training_history else 0,
                "training_time_estimate": len(self.training_history) * 60  # Rough estimate
            }
        }
        
        with open(debug_dir / "training_summary.json", 'w') as f:
            json.dump(summary_data, f, indent=2)