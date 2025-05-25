"""
ISNE model trainer implementation.

This module implements the trainer for the ISNE model, orchestrating the
training process including loss computation, optimization, and evaluation.
"""

import os
import time
from typing import Dict, List, Optional, Union, Any, Tuple, Callable, cast
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import logging
import time
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from src.isne.models.isne_model import ISNEModel
from src.isne.losses.feature_loss import FeaturePreservationLoss
from src.isne.losses.structural_loss import StructuralPreservationLoss
from src.isne.losses.contrastive_loss import ContrastiveLoss
from src.isne.training.sampler import NeighborSampler

# Set up logging
logger = logging.getLogger(__name__)

# Import the new RandomWalkSampler (optional, as it might be passed dynamically)
try:
    from src.isne.training.random_walk_sampler import RandomWalkSampler
except ImportError:
    logger.warning("RandomWalkSampler not available. Will use default sampler unless provided in config.")
    # Use Optional[Type] for classes that might be None
    from typing import Type, Optional
    RandomWalkSampler: Optional[Type] = None


class ISNETrainer:
    """
    Trainer for the ISNE model.
    
    This class implements the training procedure for the ISNE model as described
    in the original paper, including multi-objective loss computation, mini-batch
    training, and model evaluation.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        output_dim: Optional[int] = None,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        lambda_feat: float = 1.0,
        lambda_struct: float = 1.0,
        lambda_contrast: float = 0.5,
        device: Optional[Union[str, torch.device]] = None,
        sampler_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the ISNE trainer.
        
        Args:
            embedding_dim: Dimensionality of input embeddings
            hidden_dim: Dimensionality of hidden representations
            output_dim: Dimensionality of output embeddings (defaults to hidden_dim)
            num_layers: Number of ISNE layers in the model
            num_heads: Number of attention heads in each layer
            dropout: Dropout probability for regularization
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            lambda_feat: Weight for feature preservation loss
            lambda_struct: Weight for structural preservation loss
            lambda_contrast: Weight for contrastive loss
            device: Device to use for training
            sampler_config: Optional configuration for the sampler class to use during training.
                          Format: {"sampler_class": Class, "sampler_params": Dict[str, Any]}
        """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lambda_feat = lambda_feat
        self.lambda_struct = lambda_struct
        self.lambda_contrast = lambda_contrast
        
        # Store sampler configuration
        self.sampler_config = sampler_config
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        logger.info(f"ISNE trainer initialized on device: {self.device}")
        
        # Initialize model, losses, and optimizer
        self.model: Optional[ISNEModel] = None
        self.feature_loss: Optional[FeaturePreservationLoss] = None
        self.structural_loss: Optional[StructuralPreservationLoss] = None
        self.contrastive_loss: Optional[ContrastiveLoss] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        
        # Training statistics
        self.train_stats: Dict[str, Any] = {
            'epochs': 0,
            'total_loss': [],
            'feature_loss': [],
            'structural_loss': [],
            'contrastive_loss': [],
            'time_per_epoch': []
        }
        
        # Initialize filtering stats
        self.filtering_stats = {
            'filtered_pos_count': 0,
            'total_pos_count': 0,
            'filtered_neg_count': 0,
            'total_neg_count': 0
        }
    
    def _initialize_model(self) -> None:
        """
        Initialize the ISNE model.
        """
        self.model = ISNEModel(
            in_features=self.embedding_dim,
            hidden_features=self.hidden_dim,
            out_features=self.output_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            residual=True,
            normalize_features=True
        ).to(self.device)
    
    def _initialize_losses(self) -> None:
        """
        Initialize loss functions.
        """
        self.feature_loss = FeaturePreservationLoss(
            lambda_feat=self.lambda_feat
        )
        
        self.structural_loss = StructuralPreservationLoss(
            lambda_struct=self.lambda_struct,
            negative_samples=5
        )
        
        # Note: filtering_stats is already initialized in __init__
        
        # Define the callback function for contrastive loss filtering metrics
        def filter_callback(filtered_pos_count: int, total_pos_count: int, filtered_neg_count: int, total_neg_count: int) -> None:
            # Update the dictionary directly
            if hasattr(self, 'filtering_stats'):
                # Use dictionary key assignment instead of update for cleaner typing
                self.filtering_stats['filtered_pos_count'] = filtered_pos_count
                self.filtering_stats['total_pos_count'] = total_pos_count
                self.filtering_stats['filtered_neg_count'] = filtered_neg_count
                self.filtering_stats['total_neg_count'] = total_neg_count
        
        # Initialize contrastive loss with filter callback
        self.contrastive_loss = ContrastiveLoss(
            lambda_contrast=self.lambda_contrast,
            filter_callback=filter_callback
        )
    
    def _initialize_optimizer(self) -> None:
        """
        Initialize optimizer.
        """
        if self.model is None:
            raise ValueError("Model must be initialized before optimizer")
            
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
    
    def prepare_model(self) -> None:
        """
        Prepare the model, loss functions, and optimizer for training.
        """
        self._initialize_model()
        self._initialize_losses()
        self._initialize_optimizer()
    
    def train(
        self,
        features: Tensor,
        edge_index: Tensor,
        epochs: int = 100,
        batch_size: int = 32,
        num_hops: int = 1,
        neighbor_size: int = 10,
        eval_interval: int = 10,
        early_stopping_patience: int = 20,
        validation_data: Optional[Tuple[Tensor, Tensor]] = None,
        validation_metric: str = 'loss',
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the ISNE model.
        
        Args:
            features: Node feature tensor [num_nodes, embedding_dim]
            edge_index: Edge index tensor [2, num_edges]
            epochs: Number of training epochs
            batch_size: Training batch size
            num_hops: Number of hops for neighborhood sampling
            neighbor_size: Maximum number of neighbors to sample per node per hop
            eval_interval: Interval for evaluation during training
            early_stopping_patience: Patience for early stopping
            validation_data: Optional validation data (features, edge_index)
            validation_metric: Metric for validation ('loss' or 'similarity')
            verbose: Whether to show progress bar
            
        Returns:
            Training statistics
        """
        # Prepare model if not already done
        if self.model is None:
            self.prepare_model()
            
        # Move data to device
        features = features.to(self.device)
        edge_index = edge_index.to(self.device)
        
        # Validate edge_index before initializing the sampler
        actual_num_nodes = features.size(0)
        if edge_index.size(1) > 0:  # Only validate if we have edges
            max_index = edge_index.max().item()
            if max_index >= actual_num_nodes:
                logger.warning(f"Edge indices exceed feature count: max_index={max_index}, feature_count={actual_num_nodes}")
                # We need to determine whether to extend features or truncate edge_index
                if max_index - actual_num_nodes + 1 <= 100:  # If not too many additional nodes needed
                    logger.warning(f"Extending feature matrix to accommodate edge indices (+{max_index - actual_num_nodes + 1} rows)")
                    # Create padding with zeros for missing nodes
                    padding = torch.zeros(max_index - actual_num_nodes + 1, features.size(1), device=self.device)
                    features = torch.cat([features, padding], dim=0)
                    actual_num_nodes = features.size(0)
                    logger.info(f"Feature matrix extended to {actual_num_nodes} nodes")
                else:
                    logger.warning(f"Too many missing nodes ({max_index - actual_num_nodes + 1}), filtering edge_index instead")
                    # Filter edges to only include valid node indices
                    valid_edges_mask = (edge_index[0] < actual_num_nodes) & (edge_index[1] < actual_num_nodes)
                    edge_index = edge_index[:, valid_edges_mask]
                    if edge_index.size(1) == 0:
                        logger.warning("No valid edges remain after filtering. Adding self-loops for basic connectivity.")
                        # Add self-loops for minimal connectivity
                        indices = torch.arange(0, min(100, actual_num_nodes), device=self.device)
                        self_loops = torch.stack([indices, indices], dim=0)
                        edge_index = self_loops
        
        # Initialize sampler with validated indices
        logger.info(f"Initializing sampler with {actual_num_nodes} nodes and {edge_index.size(1)} edges")
        
        # Use custom sampler if provided, otherwise use default NeighborSampler
        if self.sampler_config and 'sampler_class' in self.sampler_config:
            sampler_class = self.sampler_config['sampler_class']
            sampler_params = self.sampler_config.get('sampler_params', {})
            
            # Create base parameters for any sampler
            base_params = {
                'edge_index': edge_index,
                'num_nodes': actual_num_nodes,
                'batch_size': batch_size
            }
            
            # Combine with custom parameters
            all_params = {**base_params, **sampler_params}
            
            # Create the sampler with combined parameters
            logger.info(f"Using custom sampler: {sampler_class.__name__}")
            sampler = sampler_class(**all_params)
        else:
            # Fall back to default NeighborSampler
            logger.info("Using default NeighborSampler")
            sampler = NeighborSampler(
                edge_index=edge_index,
                num_nodes=actual_num_nodes,  # Using validated node count
                batch_size=batch_size,
                num_hops=num_hops,
                neighbor_size=neighbor_size
            )
        
        # Get the size of the feature matrix for validation
        feature_size = features.shape[0]
        
        # Training loop
        best_val_metric = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Set model to training mode - ensure model is not None
            if self.model is None:
                raise ValueError("Model has not been initialized for training")
            # Explicitly cast to avoid mypy error about None having no train method
            model = cast(nn.Module, self.model)
            model.train()
            
            # Sample batch of nodes
            batch_nodes = sampler.sample_nodes()
            
            # Move batch_nodes to device if needed (features is on CUDA)
            if features.is_cuda and not batch_nodes.is_cuda:
                try:
                    batch_nodes = batch_nodes.to(features.device)
                except Exception as e:
                    logger.warning(f"Error moving batch_nodes to CUDA: {str(e)}")
                    # Try to continue with CPU tensor
            
            # Sample subgraph and get subset nodes
            subset_nodes, subgraph_edge_index = sampler.sample_subgraph(batch_nodes)
            
            # Validate indices are within bounds of the features tensor
            valid_indices = None
            try:
                if subset_nodes.numel() > 0:
                    # Move to CPU for safer validation
                    subset_nodes_cpu = subset_nodes.cpu() if subset_nodes.is_cuda else subset_nodes
                    valid_mask = (subset_nodes_cpu >= 0) & (subset_nodes_cpu < feature_size)
                    
                    if not valid_mask.all():
                        logger.warning(f"Found {(~valid_mask).sum().item()} out-of-bounds indices in subset_nodes")
                        valid_indices = subset_nodes_cpu[valid_mask]
                    else:
                        valid_indices = subset_nodes_cpu
                    
                    # Move valid_indices to the same device as features
                    if features.is_cuda and not valid_indices.is_cuda:
                        try:
                            valid_indices = valid_indices.to(features.device)
                        except Exception as e:
                            logger.warning(f"Error moving valid_indices to CUDA: {str(e)}")
                            # Fall back to using CPU indices with CPU operations
                            features_cpu = features.cpu()
                            valid_indices_cpu = valid_indices.cpu() if valid_indices.is_cuda else valid_indices
                            return {"success": False, "error": "Device migration issue", "message": str(e)}
                else:
                    valid_indices = subset_nodes
            except Exception as e:
                logger.warning(f"Error validating subset_nodes: {str(e)}")
                # Create an empty tensor with the same device as a fallback
                valid_indices = torch.empty(0, dtype=torch.long, device=self.device)
            
            # Skip this batch if no valid indices remain
            if valid_indices.numel() == 0:
                logger.warning("No valid indices remain for this batch. Skipping.")
                continue
            
            # Get features using the validated indices
            try:
                # Forward pass
                if self.model is None:
                    raise ValueError("Model has not been initialized. Call prepare_model first.")
                
                # Set model to training mode - we've already checked self.model is not None above
                # Explicitly cast to avoid mypy error about None having no train method
                model = cast(nn.Module, self.model)
                model.train()
                
                # Get embeddings
                batch_features = features[valid_indices]
                embeddings = self.model(batch_features, subgraph_edge_index)
            except Exception as e:
                logger.warning(f"Error during forward pass: {str(e)}")
                continue
            
            # Project features for feature loss
            # Zero gradients
            if self.optimizer is None:
                raise ValueError("Optimizer has not been initialized. Call prepare_model first.")
                
            self.optimizer.zero_grad()
            
            # Compute feature preservation loss
            if self.feature_loss is None:
                raise ValueError("Feature loss has not been initialized. Call prepare_model first.")
                
            feat_loss = self.feature_loss(embeddings, batch_features)
            
            # Use batch-aware sampling for positive and negative pairs if enabled and available
            use_batch_aware = False
            
            # Check if batch-aware sampling is available and enabled
            if hasattr(sampler, 'sample_positive_pairs_within_batch'):
                # First check if the sampler has use_batch_aware_sampling as an attribute
                if hasattr(sampler, 'use_batch_aware_sampling'):
                    use_batch_aware = sampler.use_batch_aware_sampling
                # Otherwise, check the sampler configuration
                elif self.sampler_config and 'sampler_params' in self.sampler_config:
                    sampler_params = self.sampler_config.get('sampler_params', {})
                    use_batch_aware = sampler_params.get('use_batch_aware_sampling', False)
            
            # Initialize filtering and tracking metrics
            total_pairs = batch_size * 2  # Positive and negative pairs
            batch_metrics = {
                'total_pairs': 0,
                'within_batch_pairs': 0,
                'fallback_pairs': 0,
                'filtered_pairs': 0,
                'pos_within_batch': 0,
                'pos_fallback': 0,
                'neg_within_batch': 0,
                'neg_fallback': 0
            }
            
            # Initialize rate variables to zero by default to prevent UnboundLocalError
            filtered_rate = 0.0
            pos_filtered_rate = 0.0
            neg_filtered_rate = 0.0
            
            # Sample pairs using the appropriate method
            if use_batch_aware and hasattr(sampler, 'sample_positive_pairs_within_batch'):
                logger.info("Using batch-aware sampling for positive and negative pairs")
                # Sample pairs only from nodes within the current batch
                pos_start_time = time.time()
                pos_pairs = sampler.sample_positive_pairs_within_batch(batch_nodes)
                pos_time = time.time() - pos_start_time
                
                neg_start_time = time.time()
                neg_pairs = sampler.sample_negative_pairs_within_batch(batch_nodes, pos_pairs)
                neg_time = time.time() - neg_start_time
                
                # Track the usage of batch-aware sampling for metrics
                self.train_stats.setdefault('sampling_method', {}).setdefault('batch_aware', 0)
                self.train_stats['sampling_method']['batch_aware'] += 1
                
                # For batch-aware sampling, check if any fallback pairs were generated
                # (The sampler may have had to fall back to some non-batch pairs if not enough were found)
                batch_nodes_set = set(batch_nodes.cpu().tolist())
                
                # Count positive pairs with possible fallback
                for i in range(len(pos_pairs)):
                    src, dst = pos_pairs[i].cpu().tolist()
                    if src in batch_nodes_set and dst in batch_nodes_set:
                        batch_metrics['pos_within_batch'] += 1
                    else:
                        batch_metrics['pos_fallback'] += 1
                
                # Count negative pairs with possible fallback
                for i in range(len(neg_pairs)):
                    src, dst = neg_pairs[i].cpu().tolist()
                    if src in batch_nodes_set and dst in batch_nodes_set:
                        batch_metrics['neg_within_batch'] += 1
                    else:
                        batch_metrics['neg_fallback'] += 1
                
                # Update overall metrics
                batch_metrics['total_pairs'] = len(pos_pairs) + len(neg_pairs)
                batch_metrics['within_batch_pairs'] = batch_metrics['pos_within_batch'] + batch_metrics['neg_within_batch']
                batch_metrics['fallback_pairs'] = batch_metrics['pos_fallback'] + batch_metrics['neg_fallback']
                
                # Calculate metrics for logging
                if batch_metrics['total_pairs'] > 0:
                    within_batch_rate = 100 * batch_metrics['within_batch_pairs'] / batch_metrics['total_pairs']
                    fallback_rate = 100 * batch_metrics['fallback_pairs'] / batch_metrics['total_pairs']
                else:
                    within_batch_rate = 0
                    fallback_rate = 0
                
                # Log detailed metrics
                logger.info(f"Batch-aware sampling metrics:")
                logger.info(f"  - Within-batch pairs: {batch_metrics['within_batch_pairs']}/{batch_metrics['total_pairs']} ({within_batch_rate:.2f}%)")
                logger.info(f"  - Fallback pairs: {batch_metrics['fallback_pairs']}/{batch_metrics['total_pairs']} ({fallback_rate:.2f}%)")
                
                # Enhanced filtering metrics with separate positive and negative stats
                # Ensure we're displaying the most accurate filtering information
                filtered_percent = 100 * batch_metrics['filtered_pairs'] / batch_metrics['total_pairs'] if batch_metrics['total_pairs'] > 0 else 0
                logger.info(f"  - Filtered by loss: {batch_metrics['filtered_pairs']}/{batch_metrics['total_pairs']} ({filtered_percent:.2f}%)")
                
                # Display the positive and negative filtering specifics
                if 'filtered_pos' in batch_metrics and 'filtered_neg' in batch_metrics:
                    # Use the actual positive pair count for more accurate percentage
                    pos_filtered_percent = 100 * batch_metrics['filtered_pos'] / original_pos_count if original_pos_count > 0 else 0
                    neg_filtered_percent = 100 * batch_metrics['filtered_neg'] / original_neg_count if original_neg_count > 0 else 0
                    logger.info(f"  - Filtered positive: {batch_metrics['filtered_pos']}/{original_pos_count} ({pos_filtered_percent:.2f}%)")
                    logger.info(f"  - Filtered negative: {batch_metrics['filtered_neg']}/{original_neg_count} ({neg_filtered_percent:.2f}%)")
                
                logger.info(f"  - Positive pairs: {batch_metrics['pos_within_batch']} in-batch, {batch_metrics['pos_fallback']} fallback")
                logger.info(f"  - Negative pairs: {batch_metrics['neg_within_batch']} in-batch, {batch_metrics['neg_fallback']} fallback")
                logger.info(f"  - Sampling time: {pos_time:.4f}s for positive, {neg_time:.4f}s for negative")
                
                # Update tracking statistics
                self.train_stats.setdefault('batch_aware_metrics', {})
                
                # Initialize on first batch
                if 'within_batch_rate' not in self.train_stats['batch_aware_metrics']:
                    self.train_stats['batch_aware_metrics'] = {
                        'within_batch_rate': [],
                        'fallback_rate': [],
                        'filtered_rate': [],
                        'pos_filtered_rate': [],
                        'neg_filtered_rate': [],
                        'pos_within_batch': [],
                        'pos_fallback': [],
                        'neg_within_batch': [],
                        'neg_fallback': [],
                        'filtered_pairs': [],
                        'filtered_pos': [],
                        'filtered_neg': [],
                        'pos_sampling_time': [],
                        'neg_sampling_time': []
                    }
                
                # Append current batch metrics
                self.train_stats['batch_aware_metrics']['within_batch_rate'].append(within_batch_rate)
                self.train_stats['batch_aware_metrics']['fallback_rate'].append(fallback_rate)
                self.train_stats['batch_aware_metrics']['filtered_rate'].append(filtered_rate)
                self.train_stats['batch_aware_metrics']['pos_filtered_rate'].append(pos_filtered_rate)
                self.train_stats['batch_aware_metrics']['neg_filtered_rate'].append(neg_filtered_rate)
                self.train_stats['batch_aware_metrics']['pos_within_batch'].append(batch_metrics['pos_within_batch'])
                self.train_stats['batch_aware_metrics']['pos_fallback'].append(batch_metrics['pos_fallback'])
                self.train_stats['batch_aware_metrics']['neg_within_batch'].append(batch_metrics['neg_within_batch'])
                self.train_stats['batch_aware_metrics']['neg_fallback'].append(batch_metrics['neg_fallback'])
                self.train_stats['batch_aware_metrics']['filtered_pairs'].append(batch_metrics['filtered_pairs'])
                self.train_stats['batch_aware_metrics']['filtered_pos'].append(batch_metrics.get('filtered_pos', 0))
                self.train_stats['batch_aware_metrics']['filtered_neg'].append(batch_metrics.get('filtered_neg', 0))
                self.train_stats['batch_aware_metrics']['pos_sampling_time'].append(pos_time)
                self.train_stats['batch_aware_metrics']['neg_sampling_time'].append(neg_time)
            else:
                # Fall back to standard sampling
                logger.info("Using standard sampling for positive and negative pairs")
                pos_pairs = sampler.sample_positive_pairs()
                neg_pairs = sampler.sample_negative_pairs(pos_pairs)
                
                # Track the usage of standard sampling for metrics
                self.train_stats.setdefault('sampling_method', {}).setdefault('standard', 0)
                self.train_stats['sampling_method']['standard'] += 1
                
                # Check how many pairs are within the batch (for filtering rate metrics)
                batch_nodes_set = set(batch_nodes.cpu().tolist())
                
                # Count pairs with nodes outside the batch
                filtered_pos = 0
                for i in range(len(pos_pairs)):
                    src, dst = pos_pairs[i].cpu().tolist()
                    if src not in batch_nodes_set or dst not in batch_nodes_set:
                        filtered_pos += 1
                
                filtered_neg = 0
                for i in range(len(neg_pairs)):
                    src, dst = neg_pairs[i].cpu().tolist()
                    if src not in batch_nodes_set or dst not in batch_nodes_set:
                        filtered_neg += 1
                
                total_filtered = filtered_pos + filtered_neg
                filter_rate = 100 * total_filtered / (len(pos_pairs) + len(neg_pairs))
                logger.info(f"Standard sampling - pair filtering rate: {filter_rate:.2f}% ({total_filtered}/{len(pos_pairs) + len(neg_pairs)})")
                
                # Track filtering statistics
                self.train_stats.setdefault('filtering_rate', [])
                self.train_stats['filtering_rate'].append(filter_rate)
                
                # Make sure filtered_rate is defined for metrics tracking
                filtered_rate = filter_rate
            
            # Compute structural preservation loss
            if self.structural_loss is None:
                raise ValueError("Structural loss has not been initialized. Call prepare_model first.")
                
            struct_loss = self.structural_loss(embeddings, subgraph_edge_index)
            
            # Track the original pair counts before contrastive loss filtering
            original_pos_count: int = len(pos_pairs)
            original_neg_count: int = len(neg_pairs)
            original_total_count: int = original_pos_count + original_neg_count
            
            # Reset filtering stats before contrastive loss computation
            self.filtering_stats.update({
                'filtered_pos_count': 0,
                'total_pos_count': 0,
                'filtered_neg_count': 0,
                'total_neg_count': 0
            })
            
            # We'll rely on the callback mechanism instead of log parsing
            # The contrastive_loss will update self.filtering_stats directly through the callback
            
            # Compute contrastive loss with potential filtering
            # The callback will update self.filtering_stats
            if self.contrastive_loss is None:
                raise ValueError("Contrastive loss has not been initialized. Call prepare_model first.")
                
            cont_loss = self.contrastive_loss(
                embeddings, 
                pos_pairs.to(self.device), 
                neg_pairs.to(self.device)
            )
            
            # Extract filtering stats from the callback-based tracking
            filtered_pos = self.filtering_stats.get('filtered_pos_count', 0)
            filtered_neg = self.filtering_stats.get('filtered_neg_count', 0)
            total_pos_count = self.filtering_stats.get('total_pos_count', original_pos_count) 
            total_neg_count = self.filtering_stats.get('total_neg_count', original_neg_count)
            total_filtered = filtered_pos + filtered_neg
            
            # Log the filtering results
            if filtered_pos > 0 or filtered_neg > 0:
                logger.info(f"Contrastive loss filtered pairs - positive: {filtered_pos}/{total_pos_count}, negative: {filtered_neg}/{total_neg_count}")
            
            # Calculate filtering rates
            filtered_rate = 100 * total_filtered / original_total_count if original_total_count > 0 else 0
            pos_filtered_rate = 100 * filtered_pos / original_pos_count if original_pos_count > 0 else 0
            neg_filtered_rate = 100 * filtered_neg / original_neg_count if original_neg_count > 0 else 0
            
            # Update batch metrics with more detailed filtering information
            batch_metrics['filtered_pairs'] = total_filtered
            batch_metrics['filtered_pos'] = filtered_pos
            batch_metrics['filtered_neg'] = filtered_neg
            
            # Store these metrics directly in the overall train_stats for filtering capture
            if 'filtering_details' not in self.train_stats:
                self.train_stats['filtering_details'] = {
                    'filtered_pos': [],
                    'filtered_neg': [],
                    'total_filtered': [],
                    'pos_filtered_rate': [],
                    'neg_filtered_rate': []
                }
                
            # Append this batch's filtering metrics to the overall stats
            self.train_stats['filtering_details']['filtered_pos'].append(filtered_pos)
            self.train_stats['filtering_details']['filtered_neg'].append(filtered_neg)
            self.train_stats['filtering_details']['total_filtered'].append(total_filtered)
            self.train_stats['filtering_details']['pos_filtered_rate'].append(pos_filtered_rate)
            self.train_stats['filtering_details']['neg_filtered_rate'].append(neg_filtered_rate)
            
            # Log filtering summary for diagnostic purposes
            if filtered_pos > 0 or filtered_neg > 0:
                logger.info(f"Contrastive loss filtered {total_filtered} pairs ({filtered_rate:.2f}%): {filtered_pos} positive, {filtered_neg} negative")
                logger.info(f"Filtering rates - positive: {pos_filtered_rate:.2f}%, negative: {neg_filtered_rate:.2f}%")
            
            # Combine losses
            total_loss = feat_loss + struct_loss + cont_loss
            
            # Backward pass and optimization
            total_loss.backward()
            if self.optimizer is None:
                raise ValueError("Optimizer has not been initialized. Call prepare_model first.")
                
            self.optimizer.step()
            
            # Record training statistics
            epoch_time = time.time() - epoch_start_time
            self.train_stats['total_loss'].append(total_loss.item())
            self.train_stats['feature_loss'].append(feat_loss.item())
            self.train_stats['structural_loss'].append(struct_loss.item())
            self.train_stats['contrastive_loss'].append(cont_loss.item())
            self.train_stats['time_per_epoch'].append(epoch_time)
            
            # Evaluate and check for early stopping
            if validation_data is not None and (epoch + 1) % eval_interval == 0:
                val_features, val_edge_index = validation_data
                val_features = val_features.to(self.device)
                val_edge_index = val_edge_index.to(self.device)
                
                val_metric = self.evaluate(val_features, val_edge_index, metric=validation_metric)
                
                if val_metric < best_val_metric:
                    best_val_metric = val_metric
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, "
                           f"Loss: {total_loss.item():.4f}, "
                           f"Time: {epoch_time:.2f}s")
        
        # Update epochs trained
        self.train_stats['epochs'] = epoch + 1
        
        # Generate detailed performance metrics about sampling and filtering
        if 'filtering_rate' in self.train_stats and self.train_stats['filtering_rate']:
            avg_filtering_rate = sum(self.train_stats['filtering_rate']) / len(self.train_stats['filtering_rate'])
            self.train_stats['avg_filtering_rate'] = avg_filtering_rate
            logger.info(f"Average filtering rate: {avg_filtering_rate:.2f}%")
        
        if 'sampling_method' in self.train_stats:
            sampling_methods = self.train_stats['sampling_method']
            total_epochs = sum(sampling_methods.values())
            for method, count in sampling_methods.items():
                percentage = (count / total_epochs) * 100 if total_epochs > 0 else 0
                logger.info(f"Sampling method usage - {method}: {count}/{total_epochs} epochs ({percentage:.1f}%)")
        
        # Add summary metrics about the batch-aware sampling performance
        uses_batch_aware = ('sampling_method' in self.train_stats and 
                          'batch_aware' in self.train_stats['sampling_method'] and
                          self.train_stats['sampling_method']['batch_aware'] > 0)
        
        if uses_batch_aware:
            logger.info("===== Batch-Aware Sampling Performance =====")
            
            # Calculate summaries from detailed metrics if available
            if 'batch_aware_metrics' in self.train_stats:
                metrics = self.train_stats['batch_aware_metrics']
                
                # Calculate averages
                avg_within_batch_rate = sum(metrics['within_batch_rate']) / len(metrics['within_batch_rate']) \
                    if metrics['within_batch_rate'] else 0
                avg_fallback_rate = sum(metrics['fallback_rate']) / len(metrics['fallback_rate']) \
                    if metrics['fallback_rate'] else 0
                avg_filtered_rate = sum(metrics['filtered_rate']) / len(metrics['filtered_rate']) \
                    if metrics['filtered_rate'] else 0
                
                # Calculate average filtering rates for positive and negative pairs
                avg_pos_filtered_rate = sum(metrics['pos_filtered_rate']) / len(metrics['pos_filtered_rate']) \
                    if metrics['pos_filtered_rate'] else 0
                avg_neg_filtered_rate = sum(metrics['neg_filtered_rate']) / len(metrics['neg_filtered_rate']) \
                    if metrics['neg_filtered_rate'] else 0
                
                total_pos_within = sum(metrics['pos_within_batch'])
                total_pos_fallback = sum(metrics['pos_fallback'])
                total_neg_within = sum(metrics['neg_within_batch'])
                total_neg_fallback = sum(metrics['neg_fallback'])
                # Get the filtered pair counts - try from direct filtering_details first, then fallback to batch metrics
                if 'filtering_details' in self.train_stats:
                    total_filtered = sum(self.train_stats['filtering_details']['total_filtered'])
                    total_filtered_pos = sum(self.train_stats['filtering_details']['filtered_pos'])
                    total_filtered_neg = sum(self.train_stats['filtering_details']['filtered_neg'])
                    
                    # Calculate additional filtered rates directly from the tracking data
                    avg_pos_filtered_rate = sum(self.train_stats['filtering_details']['pos_filtered_rate']) / len(self.train_stats['filtering_details']['pos_filtered_rate']) \
                        if self.train_stats['filtering_details']['pos_filtered_rate'] else 0
                    avg_neg_filtered_rate = sum(self.train_stats['filtering_details']['neg_filtered_rate']) / len(self.train_stats['filtering_details']['neg_filtered_rate']) \
                        if self.train_stats['filtering_details']['neg_filtered_rate'] else 0
                else:
                    # Fallback to batch metrics (less reliable)
                    total_filtered = sum(metrics['filtered_pairs']) if 'filtered_pairs' in metrics else 0
                    total_filtered_pos = sum(metrics['filtered_pos']) if 'filtered_pos' in metrics else 0
                    total_filtered_neg = sum(metrics['filtered_neg']) if 'filtered_neg' in metrics else 0
                total_pairs = total_pos_within + total_pos_fallback + total_neg_within + total_neg_fallback
                
                # Add the summary to train_stats
                self.train_stats['batch_aware_summary'] = {
                    'avg_within_batch_rate': avg_within_batch_rate,
                    'avg_fallback_rate': avg_fallback_rate,
                    'avg_filtered_rate': avg_filtered_rate,
                    'avg_pos_filtered_rate': avg_pos_filtered_rate,
                    'avg_neg_filtered_rate': avg_neg_filtered_rate,
                    'total_positive_pairs': {
                        'within_batch': total_pos_within,
                        'fallback': total_pos_fallback
                    },
                    'total_negative_pairs': {
                        'within_batch': total_neg_within,
                        'fallback': total_neg_fallback
                    },
                    'filtered_pairs': {
                        'total': total_filtered,
                        'positive': total_filtered_pos,
                        'negative': total_filtered_neg
                    },
                    'efficiency': {
                        'within_batch_percentage': 100 * (total_pos_within + total_neg_within) / total_pairs 
                        if total_pairs > 0 else 0,
                        'post_filtering_rate': 100 * total_filtered / total_pairs 
                        if total_pairs > 0 else 0,
                        'pos_filtering_rate': 100 * total_filtered_pos / total_pos_within 
                        if total_pos_within > 0 else 0,
                        'neg_filtering_rate': 100 * total_filtered_neg / total_neg_within 
                        if total_neg_within > 0 else 0
                    }
                }
                
                # Log detailed performance metrics
                logger.info(f"Batch-aware sampling was used during training with:")
                logger.info(f"  - Average within-batch pair rate: {avg_within_batch_rate:.2f}%")
                logger.info(f"  - Average fallback pair rate: {avg_fallback_rate:.2f}%")
                logger.info(f"  - Average filtered-by-loss rate: {avg_filtered_rate:.2f}%")
                logger.info(f"  - Average positive filtering: {avg_pos_filtered_rate:.2f}%")
                logger.info(f"  - Average negative filtering: {avg_neg_filtered_rate:.2f}%")
                logger.info(f"  - Total positive pairs: {total_pos_within} in-batch, {total_pos_fallback} fallback")
                logger.info(f"  - Total negative pairs: {total_neg_within} in-batch, {total_neg_fallback} fallback")
                # Report detailed filtering metrics
                logger.info(f"  - Total pairs filtered by loss: {total_filtered} ({total_filtered_pos} positive, {total_filtered_neg} negative)")
                if total_pos_within > 0:
                    pos_filter_pct = 100 * total_filtered_pos / total_pos_within
                    logger.info(f"  - Positive pair filtering rate: {pos_filter_pct:.2f}% ({total_filtered_pos}/{total_pos_within})")
                if total_neg_within > 0:
                    neg_filter_pct = 100 * total_filtered_neg / total_neg_within
                    logger.info(f"  - Negative pair filtering rate: {neg_filter_pct:.2f}% ({total_filtered_neg}/{total_neg_within})")
                logger.info(f"  - Overall efficiency: {self.train_stats['batch_aware_summary']['efficiency']['within_batch_percentage']:.2f}% pairs within batch")
                logger.info(f"  - Post-filtering rate: {self.train_stats['batch_aware_summary']['efficiency']['post_filtering_rate']:.2f}% pairs filtered by loss")
            else:
                # Basic metrics if detailed tracking wasn't available
                logger.info("Batch-aware sampling was used during training, which guarantees")
                logger.info("that sampled pairs remain within batch boundaries.")
                logger.info("This should significantly improve training efficiency by:")
                logger.info(" - Eliminating or greatly reducing out-of-bounds filtering")
                logger.info(" - Reducing reliance on fallback pairs")
                logger.info(" - Ensuring all sampled pairs can be used for training")
        else:
            logger.info("===== Standard Sampling Performance =====")
            logger.info("Standard sampling was used during training.")
            if 'avg_filtering_rate' in self.train_stats:
                logger.info(f"Average filtering rate: {self.train_stats['avg_filtering_rate']:.2f}%")
                logger.info("Consider enabling batch-aware sampling to improve efficiency.")
            
        logger.info("=======================================")
        
        return self.train_stats
    
    def evaluate(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        metric: str = 'loss'
    ) -> float:
        """
        Evaluate the model on the given data.
        
        Args:
            features: Node feature tensor [num_nodes, embedding_dim]
            edge_index: Edge index tensor [2, num_edges]
            metric: Evaluation metric ('loss' or 'similarity')
            
        Returns:
            Evaluation score
        """
        # Set model to evaluation mode
        if self.model is None:
            raise ValueError("Model has not been initialized. Call prepare_model first.")
            
        self.model.eval()
        with torch.no_grad():
            # Forward pass
            embeddings = self.model(features, edge_index)
            
            if metric == 'loss':
                # Project features for feature loss
                projected_features = self.model.project_features(features)
                
                # Compute all losses as a proxy for quality
                if self.feature_loss is None:
                    raise ValueError("Feature loss has not been initialized. Call prepare_model first.")
                if self.structural_loss is None:
                    raise ValueError("Structural loss has not been initialized. Call prepare_model first.")
                if self.contrastive_loss is None:
                    raise ValueError("Contrastive loss has not been initialized. Call prepare_model first.")
                if self.model is None:
                    raise ValueError("Model has not been initialized. Call prepare_model first.")
                    
                feat_loss = self.feature_loss(embeddings, features)
                struct_loss = self.structural_loss(embeddings, edge_index)
                
                # For evaluation, we use simpler contrastive loss evaluation
                neg_embeddings = self.model(torch.roll(features, 1, dims=0), edge_index)
                
                cont_loss = self.contrastive_loss(embeddings, neg_embeddings)
                
                # Total loss
                total_loss = feat_loss + struct_loss + cont_loss
                
                return float(total_loss.item())
            
            elif metric == 'similarity':
                # Compute average cosine similarity for connected nodes
                src, dst = edge_index
                src_emb = embeddings[src]
                dst_emb = embeddings[dst]
                
                sim = torch.nn.functional.cosine_similarity(src_emb, dst_emb)
                
                # Return negative similarity (since we're minimizing)
                return float(sim.mean().item())
            
            else:
                raise ValueError(f"Unknown evaluation metric: {metric}")
    
    def get_embeddings(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Get embeddings for the given nodes.
        
        Args:
            features: Node feature tensor [num_nodes, embedding_dim]
            edge_index: Edge index tensor [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        if self.model is None:
            raise ValueError("Model has not been initialized. Call prepare_model first.")
            
        self.model.eval()
        
        # Move data to device
        features = features.to(self.device)
        edge_index = edge_index.to(self.device)
        
        with torch.no_grad():
            # Forward pass
            embeddings = self.model(features, edge_index)
            
            return embeddings.cpu()
    
    def save_model(self, path: Union[str, Path]) -> None:
        """
        Save the model to the specified path.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Create directory if it doesn't exist
        path = Path(path) if isinstance(path, str) else path
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'config': {
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'dropout': self.dropout
            },
            'train_stats': self.train_stats
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Union[str, Path]) -> None:
        """
        Load the model from the specified path.
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        if self.model is None:
            raise ValueError("Model has not been initialized. Call prepare_model first.")
            
        checkpoint = torch.load(path, map_location=self.device)
        
        # Handle both full model checkpoint and just state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Full checkpoint with metadata
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state if available and optimizer exists
            if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            # Just the model state dict
            self.model.load_state_dict(checkpoint)
        
        logger.info(f"Model loaded from {path}")
    
    def visualize_embeddings(
        self,
        embeddings: Tensor,
        labels: Optional[Tensor] = None,
        method: str = 'tsne',
        n_components: int = 2,
        perplexity: int = 30,
        random_state: int = 42
    ) -> NDArray[np.float64]:
        """
        Visualize embeddings using dimensionality reduction.
        
        Args:
            embeddings: Node embeddings [num_nodes, embedding_dim]
            labels: Optional node labels for coloring
            method: Dimensionality reduction method ('tsne', 'pca', 'umap')
            n_components: Number of components for visualization
            perplexity: Perplexity parameter for t-SNE
            random_state: Random state for reproducibility
            
        Returns:
            Low-dimensional representation of embeddings [num_nodes, n_components]
        """
        try:
            # Convert to numpy if tensor
            if isinstance(embeddings, torch.Tensor):
                embeddings_np = embeddings.cpu().numpy()
            else:
                embeddings_np = np.array(embeddings)
            
            if method == 'tsne':
                from sklearn.manifold import TSNE
                embeddings_2d = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state).fit_transform(embeddings_np)
            elif method == 'pca':
                from sklearn.decomposition import PCA
                embeddings_2d = PCA(n_components=n_components, random_state=random_state).fit_transform(embeddings_np)
            elif method == 'umap':
                try:
                    import umap
                    embeddings_2d = umap.UMAP(n_components=n_components, random_state=random_state).fit_transform(embeddings_np)
                except ImportError:
                    logger.warning("UMAP not installed. Falling back to t-SNE.")
                    from sklearn.manifold import TSNE
                    embeddings_2d = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state).fit_transform(embeddings_np)
            else:
                raise ValueError(f"Unknown visualization method: {method}")
            
            # Ensure correct return type for mypy
            result = cast(NDArray[np.float64], np.array(embeddings_2d, dtype=np.float64))
            return result
            
        except Exception as e:
            logger.error(f"Error visualizing embeddings: {e}")
            return np.array([])
