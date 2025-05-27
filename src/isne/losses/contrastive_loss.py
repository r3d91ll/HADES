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
import random
from typing import Optional, Tuple, Callable, cast, List, Dict, Any, Union

# Set up logging
logger = logging.getLogger(__name__)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for ISNE as described in the original paper.
    
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
            # Explicitly cast to Tensor to satisfy type checker
            result = 1.0 - F.cosine_similarity(x1, x2, dim=1)
            return cast(Tensor, result)
        else:
            # Euclidean distance
            # Explicitly cast to Tensor to satisfy type checker
            result = torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1) + 1e-8)
            return cast(Tensor, result)
    
    def _generate_fallback_pairs(self, num_nodes: int, batch_size: int = 8) -> Tuple[Tensor, Tensor]:
        """
        Generate fallback pairs when no valid pairs are found.
        
        Args:
            num_nodes: Number of nodes in the graph
            batch_size: Number of pairs to generate
            
        Returns:
            Tuple of (positive_pairs, negative_pairs)
        """
        # Generate random indices for positive pairs
        pos_indices = torch.randint(0, num_nodes, (batch_size, 2), device='cpu')
        
        # Ensure we don't have duplicate indices in a pair
        for i in range(batch_size):
            while pos_indices[i, 0] == pos_indices[i, 1]:
                pos_indices[i, 1] = torch.randint(0, num_nodes, (1,), device='cpu')
        
        # Generate random indices for negative pairs
        neg_indices = torch.randint(0, num_nodes, (batch_size, 2), device='cpu')
        
        # Ensure we don't have duplicate indices in a pair
        for i in range(batch_size):
            while neg_indices[i, 0] == neg_indices[i, 1]:
                neg_indices[i, 1] = torch.randint(0, num_nodes, (1,), device='cpu')
        
        return pos_indices, neg_indices
    
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
        num_nodes = embeddings.size(0)
        
        # If there are no positive pairs, generate fallback pairs
        if positive_pairs.size(0) == 0:
            logger.warning("No positive pairs provided, generating fallback pairs")
            positive_pairs, neg_fallback = self._generate_fallback_pairs(num_nodes)
            if negative_pairs is None or negative_pairs.size(0) == 0:
                negative_pairs = neg_fallback
        
        # If there are no negative pairs, generate fallback pairs
        if negative_pairs is None or negative_pairs.size(0) == 0:
            logger.warning("No negative pairs provided, generating fallback negative pairs")
            _, negative_pairs = self._generate_fallback_pairs(num_nodes)
        
        # Move pairs to the same device as embeddings
        device = embeddings.device
        
        # Check if positive pairs are on the same device
        if positive_pairs.device != device:
            positive_pairs = positive_pairs.to(device)
        
        # Check if negative pairs are on the same device
        if negative_pairs is not None and negative_pairs.device != device:
            negative_pairs = negative_pairs.to(device)
        
        # Get the total counts for filtering statistics
        total_pos_count = positive_pairs.size(0)
        total_neg_count = 0 if negative_pairs is None else negative_pairs.size(0)
        
        # Filter out pairs with invalid indices
        if positive_pairs.size(0) > 0:
            # Check for out-of-bounds indices in positive pairs
            pos_pairs_cpu = positive_pairs.cpu()
            valid_pos_mask = (pos_pairs_cpu[:, 0] < num_nodes) & (pos_pairs_cpu[:, 1] < num_nodes)
            
            if not torch.all(valid_pos_mask):
                # Explicitly cast tensor.item() result to int to avoid type issues
                filtered_pos_count = int((~valid_pos_mask).sum().item())
                percentage = 100 * filtered_pos_count / total_pos_count if total_pos_count > 0 else 0
                logger.warning(f"Filtered {filtered_pos_count}/{total_pos_count} out-of-bounds positive pairs ({percentage:.1f}%)")
                
                # Get some samples of invalid pairs for debugging
                if filtered_pos_count > 0 and logger.isEnabledFor(logging.DEBUG):
                    invalid_pairs = pos_pairs_cpu[~valid_pos_mask]
                    logger.debug(f"Invalid positive pairs sample: {invalid_pairs[:5]}")
                
                positive_pairs = positive_pairs[valid_pos_mask]
        
        if negative_pairs is not None and negative_pairs.size(0) > 0:
            # Check for out-of-bounds indices in negative pairs
            neg_pairs_cpu = negative_pairs.cpu()
            valid_neg_mask = (neg_pairs_cpu[:, 0] < num_nodes) & (neg_pairs_cpu[:, 1] < num_nodes)
            
            # total_neg_count was already initialized at the beginning of the method
            # Just ensure filtered_neg_count is correctly updated
            
            if not torch.all(valid_neg_mask):
                # Explicitly cast tensor.item() result to int to avoid type issues
                filtered_neg_count = int((~valid_neg_mask).sum().item())
                percentage = 100 * filtered_neg_count / total_neg_count if total_neg_count > 0 else 0
                logger.warning(f"Filtered {filtered_neg_count}/{total_neg_count} out-of-bounds negative pairs ({percentage:.1f}%)")
                
                # Get some samples of invalid pairs for debugging
                if filtered_neg_count > 0 and logger.isEnabledFor(logging.DEBUG):
                    invalid_pairs = neg_pairs_cpu[~valid_neg_mask]
                    if invalid_pairs.numel() > 0:
                        # Use proper tensor indexing for max/min
                        if invalid_pairs.numel() > 1:
                            max_idx = torch.max(invalid_pairs).item()
                            min_idx = torch.min(invalid_pairs).item()
                            logger.debug(f"Invalid pairs index range: min={min_idx}, max={max_idx}, num_nodes={num_nodes}")
                
                negative_pairs = negative_pairs[valid_neg_mask]
        
        # Notify callback about filtering statistics if provided
        if self.filter_callback is not None:
            filtered_pos_count = total_pos_count - positive_pairs.size(0)
            filtered_neg_count = 0 if negative_pairs is None else total_neg_count - negative_pairs.size(0)
            self.filter_callback(filtered_pos_count, total_pos_count, filtered_neg_count, total_neg_count)
        
        # If we have no pairs left after filtering, generate fallback pairs
        if positive_pairs.size(0) == 0:
            logger.warning("No valid positive pairs after filtering, generating fallback pairs")
            pos_fallback, _ = self._generate_fallback_pairs(num_nodes)
            positive_pairs = pos_fallback.to(device)
        
        if negative_pairs is None or negative_pairs.size(0) == 0:
            logger.warning("No valid negative pairs after filtering, generating fallback pairs")
            _, neg_fallback = self._generate_fallback_pairs(num_nodes)
            negative_pairs = neg_fallback.to(device)
        
        # Extract embeddings for positive pairs
        pos_i = embeddings[positive_pairs[:, 0]]
        pos_j = embeddings[positive_pairs[:, 1]]
        
        # Compute distance for positive pairs
        pos_dist = self._compute_distance(pos_i, pos_j)
        
        # Positive loss: encourage embeddings to be similar
        pos_loss = pos_dist
        
        # Extract embeddings for negative pairs
        neg_i = embeddings[negative_pairs[:, 0]]
        neg_j = embeddings[negative_pairs[:, 1]]
        
        # Compute distance for negative pairs
        neg_dist = self._compute_distance(neg_i, neg_j)
        
        # Negative loss: encourage embeddings to be dissimilar
        # (max(0, margin - distance) will be high when distance is small)
        neg_loss = F.relu(self.margin - neg_dist)
        
        # Combine positive and negative losses
        loss = torch.cat([pos_loss, neg_loss], dim=0)
        
        # Apply reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        # Apply weight factor
        weighted_loss = self.lambda_contrast * loss
        
        # Return the weighted loss directly
        return weighted_loss


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
        # Normalize embeddings
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
        
        # Compute positive similarity (dot product)
        pos_sim = torch.sum(anchor_embeddings * positive_embeddings, dim=1) / self.temperature
        
        if negative_embeddings is not None:
            # If explicit negative samples are provided
            negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)
            
            # Compute similarity with all negative samples
            neg_sim = torch.matmul(anchor_embeddings, negative_embeddings.t()) / self.temperature
            
            # Combine positive and negative similarities
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
            
            # The first element in each row is the positive sample
            labels = torch.zeros(len(anchor_embeddings), dtype=torch.long, device=anchor_embeddings.device)
        else:
            # Use other positives as negatives (like SimCLR)
            batch_size = anchor_embeddings.size(0)
            
            # Compute all pairwise similarities
            sim_matrix = torch.matmul(anchor_embeddings, anchor_embeddings.t()) / self.temperature
            
            # Zero out the diagonal (self-similarity)
            mask = torch.eye(batch_size, device=anchor_embeddings.device)
            sim_matrix = sim_matrix * (1 - mask)
            
            # Create labels
            logits = sim_matrix
            labels = torch.arange(batch_size, device=anchor_embeddings.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        # Apply weight factor
        weighted_loss = self.lambda_infonce * loss
        
        # Return the weighted loss directly
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
            # Explicitly cast to Tensor to satisfy type checker
            result = 1.0 - F.cosine_similarity(x1, x2, dim=1)
            return cast(Tensor, result)
        else:
            # Euclidean distance
            # Explicitly cast to Tensor to satisfy type checker
            result = torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1) + 1e-8)
            return cast(Tensor, result)
    
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
        # Extract embeddings for anchors, positives, and negatives
        anchors = embeddings[anchor_indices]
        positives = embeddings[positive_indices]
        negatives = embeddings[negative_indices]
        
        # Compute distances
        pos_dist = self._compute_distance(anchors, positives)
        neg_dist = self._compute_distance(anchors, negatives)
        
        # Compute triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        # Apply reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        # Apply weight factor
        weighted_loss = self.lambda_triplet * loss
        
        # Return the weighted loss directly
        return weighted_loss
