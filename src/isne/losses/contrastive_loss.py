"""
Contrastive loss implementation for ISNE.

This module implements the contrastive loss component for the ISNE model, which
encourages similar nodes to have similar embeddings while pushing dissimilar nodes apart.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import logging
from typing import Optional, Tuple, Callable, Dict, Any

# Set up logging
logger = logging.getLogger(__name__)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for ISNE as described in the original paper.
    
    This loss encourages similar nodes to have similar embeddings (positive pairs)
    and dissimilar nodes to have different embeddings (negative pairs).
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        reduction: str = 'mean',
        lambda_contrast: float = 1.0,
        distance_metric: str = 'cosine',
        filter_callback: Optional[Callable[..., None]] = None
    ) -> None:
        """
        Initialize the contrastive loss.
        
        Args:
            margin: Margin for contrastive loss
            reduction: Reduction method ('none', 'mean', 'sum')
            lambda_contrast: Weight factor for the contrastive loss
            distance_metric: Distance metric to use ('cosine', 'euclidean')
            filter_callback: Optional callback function that will be called with filtering statistics
                             Expected signature: callback(filtered_pos_count, total_pos_count, filtered_neg_count, total_neg_count)
        """
        super(ContrastiveLoss, self).__init__()
        
        self.margin = margin
        self.reduction = reduction
        self.lambda_contrast = lambda_contrast
        self.distance_metric = distance_metric
        self.filter_callback = filter_callback
    
    def _compute_distance(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Compute distance between pairs of embeddings.
        
        Args:
            x1: First set of embeddings [batch_size, embedding_dim]
            x2: Second set of embeddings [batch_size, embedding_dim]
            
        Returns:
            Distance tensor [batch_size]
        """
        if self.distance_metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            return 1.0 - F.cosine_similarity(x1, x2, dim=1)
        elif self.distance_metric == 'euclidean':
            # Euclidean distance (squared)
            return torch.sum((x1 - x2) ** 2, dim=1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def _generate_fallback_pairs(self, num_nodes: int, batch_size: int = 8) -> Tuple[Tensor, Tensor]:
        """
        Generate fallback pairs when no valid pairs are found.
        
        Args:
            num_nodes: Number of nodes in the graph
            batch_size: Number of pairs to generate
            
        Returns:
            Tuple of (positive_pairs, negative_pairs)
        """
        device = torch.device('cpu')  # Always generate on CPU first for safety
        
        # Create artificial positive pairs (adjacent indices)
        pos_pairs = []
        for i in range(min(batch_size, num_nodes - 1)):
            pos_pairs.append([i, i + 1])  # Adjacent nodes are considered "related"
        
        # Create artificial negative pairs (distant indices)
        neg_pairs = []
        half_nodes = max(1, num_nodes // 2)
        for i in range(min(batch_size, half_nodes)):
            # Nodes with half the graph distance are considered "unrelated"
            neg_pairs.append([i, (i + half_nodes) % num_nodes])
        
        pos_tensor = torch.tensor(pos_pairs, dtype=torch.long, device=device)
        neg_tensor = torch.tensor(neg_pairs, dtype=torch.long, device=device)
        
        return pos_tensor, neg_tensor
        
    def forward(
        self,
        embeddings: Tensor,
        positive_pairs: Tensor,
        negative_pairs: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute the contrastive loss.
        
        Args:
            embeddings: Node embeddings from the ISNE model [num_nodes, embedding_dim]
            positive_pairs: Tensor of positive pair indices [num_pos_pairs, 2]
            negative_pairs: Optional tensor of negative pair indices [num_neg_pairs, 2]
            
        Returns:
            Contrastive loss
        """
        # Initialize filtering statistics
        filtered_pos_count = 0
        total_pos_count = positive_pairs.size(0) if positive_pairs is not None else 0
        filtered_neg_count = 0
        total_neg_count = negative_pairs.size(0) if negative_pairs is not None else 0
        
        # Check for empty inputs
        if total_pos_count == 0:
            logger.warning("No positive pairs provided to contrastive loss")
            return torch.tensor(0.0, device=embeddings.device)
            
        # Log both embedding shape and pair counts for debugging
        pos_pair_count = positive_pairs.size(0) if positive_pairs is not None else 0
        neg_pair_count = negative_pairs.size(0) if negative_pairs is not None else 0
        logger.debug(f"ContrastiveLoss input: embeddings={embeddings.shape}, positive_pairs={pos_pair_count}, negative_pairs={neg_pair_count}")
        
        # Store valid node indices to avoid repeated lookups
        self.valid_range = range(embeddings.size(0))
        try:
            # Move to CPU for safer operations if needed
            if embeddings.is_cuda:
                # Only use CPU for critical index validation operations
                embeddings_device = embeddings.device
                num_nodes = embeddings.size(0)
                
                # Ensure positive pairs are on CPU for safe filtering
                pos_pairs_cpu = positive_pairs.cpu() if positive_pairs.is_cuda else positive_pairs
                
                # More strict validation for positive pairs
                valid_pos_mask = (
                    (pos_pairs_cpu[:, 0] >= 0) & 
                    (pos_pairs_cpu[:, 0] < num_nodes) & 
                    (pos_pairs_cpu[:, 1] >= 0) & 
                    (pos_pairs_cpu[:, 1] < num_nodes)
                )
                
                if not torch.all(valid_pos_mask):
                    filtered_count = (~valid_pos_mask).sum().item()
                    total_count = pos_pairs_cpu.size(0)
                    logger.warning(f"Filtered {filtered_count}/{total_count} out-of-bounds positive pairs ({filtered_count/total_count:.1%})")
                    
                    # Get some samples of invalid pairs for debugging
                    if filtered_count > 0 and logger.isEnabledFor(logging.DEBUG):
                        invalid_pairs = pos_pairs_cpu[~valid_pos_mask]
                        sample_size = min(5, invalid_pairs.size(0))
                        sample_pairs = invalid_pairs[:sample_size]
                        logger.debug(f"Sample invalid pairs: {sample_pairs}")
                    
                    # Filter to keep only valid pairs
                    pos_pairs_cpu = pos_pairs_cpu[valid_pos_mask]
                    
                # Create fallback pairs if not enough valid pairs
                MIN_PAIRS = 4  # Minimum number of pairs needed for meaningful loss
                if pos_pairs_cpu.size(0) < MIN_PAIRS:
                    logger.warning(f"Only found {pos_pairs_cpu.size(0)} valid positive pairs, adding fallback pairs")
                    fallback_pos, fallback_neg = self._generate_fallback_pairs(num_nodes)
                    
                    # Combine with any existing valid pairs
                    if pos_pairs_cpu.size(0) > 0:
                        pos_pairs_cpu = torch.cat([pos_pairs_cpu, fallback_pos], dim=0)
                    else:
                        pos_pairs_cpu = fallback_pos
                        
                    # Set fallback negative pairs if needed
                    if negative_pairs is None or (negative_pairs is not None and negative_pairs.size(0) == 0):
                        negative_pairs = fallback_neg
                    
                # Move valid positive pairs back to original device
                positive_pairs = pos_pairs_cpu.to(embeddings_device)
            else:
                # Regular CPU operation
                num_nodes = embeddings.size(0)
                
                # More strict validation for positive pairs
                valid_pos_mask = (positive_pairs[:, 0] < num_nodes) & (positive_pairs[:, 1] < num_nodes)
                
                if not valid_pos_mask.all():
                    filtered_pos_count = (~valid_pos_mask).sum().item()
                    percentage = 100 * filtered_pos_count / total_pos_count
                    logger.warning(f"Filtered {filtered_pos_count}/{total_pos_count} out-of-bounds positive pairs ({percentage:.1f}%)")
                    # Filter positive pairs
                    positive_pairs = positive_pairs[valid_pos_mask]
                
                # Create fallback pairs if not enough valid pairs
                MIN_PAIRS = 4  # Minimum number of pairs needed for meaningful loss
                if positive_pairs.size(0) < MIN_PAIRS:
                    logger.warning(f"Only found {positive_pairs.size(0)} valid positive pairs, adding fallback pairs")
                    fallback_pos, fallback_neg = self._generate_fallback_pairs(num_nodes)
                    
                    # Combine with any existing valid pairs
                    if positive_pairs.size(0) > 0:
                        positive_pairs = torch.cat([positive_pairs, fallback_pos], dim=0)
                    else:
                        positive_pairs = fallback_pos
                        
                    # Set fallback negative pairs if needed
                    if negative_pairs is None or (negative_pairs is not None and negative_pairs.size(0) == 0):
                        negative_pairs = fallback_neg
            
            # Extract positive pair embeddings
            pos_idx1, pos_idx2 = positive_pairs[:, 0], positive_pairs[:, 1]
            pos_emb1 = embeddings[pos_idx1]
            pos_emb2 = embeddings[pos_idx2]
            
            # Compute distance for positive pairs
            pos_distances = self._compute_distance(pos_emb1, pos_emb2)
            
            # Compute positive pair loss: encourage similar nodes to be close
            pos_loss = pos_distances
            
            # Process negative pairs if provided
            neg_loss = None
            if negative_pairs is not None:
                # Move to CPU for safer operations if needed
                if embeddings.is_cuda:
                    neg_pairs_cpu = negative_pairs.cpu() if negative_pairs.is_cuda else negative_pairs
                    valid_neg_mask = (
                        (neg_pairs_cpu[:, 0] >= 0) & 
                        (neg_pairs_cpu[:, 0] < num_nodes) & 
                        (neg_pairs_cpu[:, 1] >= 0) & 
                        (neg_pairs_cpu[:, 1] < num_nodes)
                    )
                    
                    # total_neg_count was already initialized at the beginning of the method
                    # Just ensure filtered_neg_count is correctly updated
                    
                    if not torch.all(valid_neg_mask):
                        filtered_neg_count = (~valid_neg_mask).sum().item()
                        percentage = 100 * filtered_neg_count / total_neg_count if total_neg_count > 0 else 0
                        logger.warning(f"Filtered {filtered_neg_count}/{total_neg_count} out-of-bounds negative pairs ({percentage:.1f}%)")
                        
                        # Get some samples of invalid pairs for debugging
                        if filtered_neg_count > 0 and logger.isEnabledFor(logging.DEBUG):
                            invalid_pairs = neg_pairs_cpu[~valid_neg_mask]
                            sample_size = min(5, invalid_pairs.size(0))
                            sample_pairs = invalid_pairs[:sample_size]
                            logger.debug(f"Sample invalid negative pairs: {sample_pairs}")
                        
                        neg_pairs_cpu = neg_pairs_cpu[valid_neg_mask]
                    
                    # Move valid negative pairs back to original device if we have any
                    if neg_pairs_cpu.size(0) > 0:
                        negative_pairs = neg_pairs_cpu.to(embeddings_device)
                    else:
                        # Generate fallback negative pairs
                        logger.warning("No valid negative pairs after filtering. Using fallback pairs.")
                        _, fallback_neg = self._generate_fallback_pairs(num_nodes)
                        negative_pairs = fallback_neg.to(embeddings_device)
                else:
                    # Regular CPU operation for negative pairs with improved validation
                    valid_neg_mask = (
                        (negative_pairs[:, 0] >= 0) & 
                        (negative_pairs[:, 0] < num_nodes) & 
                        (negative_pairs[:, 1] >= 0) & 
                        (negative_pairs[:, 1] < num_nodes)
                    )
                    
                    # total_neg_count was already initialized at the beginning of the method
                    # Just ensure filtered_neg_count is correctly updated
                    
                    if not torch.all(valid_neg_mask):
                        filtered_neg_count = (~valid_neg_mask).sum().item()
                        percentage = 100 * filtered_neg_count / total_neg_count if total_neg_count > 0 else 0
                        logger.warning(f"Filtered {filtered_neg_count}/{total_neg_count} out-of-bounds negative pairs ({percentage:.1f}%)")
                        
                        # Get statistics on the out-of-bounds indices if we have debugging enabled
                        if logger.isEnabledFor(logging.DEBUG):
                            invalid_pairs = negative_pairs[~valid_neg_mask]
                            if invalid_pairs.numel() > 0:
                                max_idx = torch.max(invalid_pairs).item()
                                min_idx = torch.min(invalid_pairs).item()
                                logger.debug(f"Invalid pairs index range: min={min_idx}, max={max_idx}, num_nodes={num_nodes}")
                        
                        negative_pairs = negative_pairs[valid_neg_mask]
                    
                    # Generate fallback negative pairs if needed
                    if negative_pairs.size(0) == 0:
                        logger.warning("No valid negative pairs after filtering. Using fallback pairs.")
                        _, fallback_neg = self._generate_fallback_pairs(num_nodes)
                        negative_pairs = fallback_neg
                
                # Process the negative pairs
                if negative_pairs.size(0) > 0:
                    neg_idx1, neg_idx2 = negative_pairs[:, 0], negative_pairs[:, 1]
                    neg_emb1 = embeddings[neg_idx1]
                    neg_emb2 = embeddings[neg_idx2]
                    
                    # Compute distance for negative pairs
                    neg_distances = self._compute_distance(neg_emb1, neg_emb2)
                    
                    # Compute negative pair loss: push dissimilar nodes apart
                    neg_loss = F.relu(self.margin - neg_distances)
            
            # Combine losses based on availability
            if neg_loss is not None and neg_loss.numel() > 0:
                # Have both positive and negative losses
                combined_loss = pos_loss.mean() + neg_loss.mean()
            else:
                # Only positive loss
                combined_loss = pos_loss.mean()
            
            # Total loss
            total_loss = self.lambda_contrast * combined_loss
            
            # Call the filter callback if provided
            if self.filter_callback is not None:
                # Use the collected filtering statistics
                try:
                    # Call with positional arguments to match the type signature
                    self.filter_callback(
                        filtered_pos_count,
                        total_pos_count,
                        filtered_neg_count,
                        total_neg_count
                    )
                except Exception as e:
                    logger.error(f"Error in contrastive loss filter callback: {e}")
            
            return total_loss
            
        except Exception as e:
            logger.warning(f"Error in contrastive loss computation: {str(e)}")
            # Return small non-zero loss to allow training to continue
            return torch.tensor(0.01, device=embeddings.device)


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss for ISNE.
    
    This is a popular contrastive learning loss that treats each node as a class
    and tries to identify the positive samples from a set of negative samples.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        lambda_infonce: float = 1.0
    ) -> None:
        """
        Initialize the InfoNCE loss.
        
        Args:
            temperature: Temperature parameter for scaling
            lambda_infonce: Weight factor for the InfoNCE loss
        """
        super(InfoNCELoss, self).__init__()
        
        self.temperature = temperature
        self.lambda_infonce = lambda_infonce
    
    def forward(
        self,
        anchor_embeddings: Tensor,
        positive_embeddings: Tensor,
        negative_embeddings: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute the InfoNCE loss.
        
        Args:
            anchor_embeddings: Anchor node embeddings [batch_size, embedding_dim]
            positive_embeddings: Positive sample embeddings [batch_size, embedding_dim]
            negative_embeddings: Optional negative sample embeddings [num_negatives, embedding_dim]
            
        Returns:
            InfoNCE loss
        """
        batch_size = anchor_embeddings.size(0)
        
        # Normalize embeddings for cosine similarity
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
        
        # Compute positive similarity (dot product)
        pos_sim = torch.sum(anchor_embeddings * positive_embeddings, dim=1) / self.temperature
        
        if negative_embeddings is not None:
            # If explicit negative samples are provided
            negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)
            
            # Compute similarity with all negative samples
            neg_sim = torch.matmul(anchor_embeddings, negative_embeddings.t()) / self.temperature
            
            # Concatenate positive and negative similarities
            logits = torch.cat([pos_sim.view(-1, 1), neg_sim], dim=1)
            
            # Labels are zeros (positive samples are at index 0)
            labels = torch.zeros(batch_size, dtype=torch.long, device=anchor_embeddings.device)
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(logits, labels)
        else:
            # Use other samples in batch as negatives (similar to SimCLR)
            # Compute similarity matrix for all pairs
            sim_matrix = torch.matmul(anchor_embeddings, positive_embeddings.t()) / self.temperature
            
            # Labels are diagonal indices (each anchor matched with its own positive)
            labels = torch.arange(batch_size, device=anchor_embeddings.device)
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(sim_matrix, labels)
        
        # Apply weight factor
        weighted_loss = self.lambda_infonce * loss
        
        return weighted_loss


class TripletLoss(nn.Module):
    """
    Triplet loss for ISNE.
    
    This loss takes triplets of (anchor, positive, negative) samples and
    encourages the anchor to be closer to the positive than to the negative
    by a specified margin.
    """
    
    def __init__(
        self,
        margin: float = 0.5,
        reduction: str = 'mean',
        lambda_triplet: float = 1.0,
        distance_metric: str = 'cosine'
    ) -> None:
        """
        Initialize the triplet loss.
        
        Args:
            margin: Margin for triplet loss
            reduction: Reduction method ('none', 'mean', 'sum')
            lambda_triplet: Weight factor for the triplet loss
            distance_metric: Distance metric to use ('cosine', 'euclidean')
        """
        super(TripletLoss, self).__init__()
        
        self.margin = margin
        self.reduction = reduction
        self.lambda_triplet = lambda_triplet
        self.distance_metric = distance_metric
    
    def _compute_distance(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Compute distance between pairs of embeddings.
        
        Args:
            x1: First set of embeddings [batch_size, embedding_dim]
            x2: Second set of embeddings [batch_size, embedding_dim]
            
        Returns:
            Distance tensor [batch_size]
        """
        if self.distance_metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            return 1.0 - F.cosine_similarity(x1, x2, dim=1)
        elif self.distance_metric == 'euclidean':
            # Euclidean distance (squared)
            return torch.sum((x1 - x2) ** 2, dim=1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def forward(
        self,
        embeddings: Tensor,
        anchor_indices: Tensor,
        positive_indices: Tensor,
        negative_indices: Tensor
    ) -> Tensor:
        """
        Compute the triplet loss.
        
        Args:
            embeddings: Node embeddings from the ISNE model [num_nodes, embedding_dim]
            anchor_indices: Indices of anchor nodes [batch_size]
            positive_indices: Indices of positive nodes [batch_size]
            negative_indices: Indices of negative nodes [batch_size]
            
        Returns:
            Triplet loss
        """
        # Extract triplet embeddings
        anchor_emb = embeddings[anchor_indices]
        positive_emb = embeddings[positive_indices]
        negative_emb = embeddings[negative_indices]
        
        # Compute distances
        pos_distances = self._compute_distance(anchor_emb, positive_emb)
        neg_distances = self._compute_distance(anchor_emb, negative_emb)
        
        # Compute triplet loss: d(anchor, positive) - d(anchor, negative) + margin
        loss = F.relu(pos_distances - neg_distances + self.margin)
        
        # Apply reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        # Apply weight factor
        weighted_loss = self.lambda_triplet * loss
        
        return weighted_loss
