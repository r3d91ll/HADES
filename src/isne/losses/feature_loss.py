"""
Feature preservation loss implementation for ISNE.

This module implements the feature preservation loss component described in the 
original ISNE paper, which ensures that the learned embeddings preserve the 
information present in the original node features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import logging

# Set up logging
logger = logging.getLogger(__name__)


class FeaturePreservationLoss(nn.Module):
    """
    Feature preservation loss for ISNE as described in the paper.
    
    This loss encourages the learned embeddings to maintain the information
    content of the original node features. It works by minimizing the distance
    between a linear projection of the original features and the final embeddings.
    """
    
    def __init__(
        self,
        reduction: str = 'mean',
        lambda_feat: float = 1.0
    ) -> None:
        """
        Initialize the feature preservation loss.
        
        Args:
            reduction: Reduction method ('none', 'mean', 'sum')
            lambda_feat: Weight factor for the feature loss
        """
        super(FeaturePreservationLoss, self).__init__()
        
        self.reduction = reduction
        self.lambda_feat = lambda_feat
    
    def forward(
        self,
        embeddings: Tensor,
        projected_features: Tensor
    ) -> Tensor:
        """
        Compute the feature preservation loss.
        
        Args:
            embeddings: Node embeddings from the ISNE model [num_nodes, embedding_dim]
            projected_features: Projected original features [num_nodes, embedding_dim]
            
        Returns:
            Feature preservation loss
        """
        # Normalize embeddings and projected features
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        norm_features = F.normalize(projected_features, p=2, dim=1)
        
        # Compute MSE loss between normalized embeddings and features
        loss = F.mse_loss(norm_embeddings, norm_features, reduction=self.reduction)
        
        # Apply weight factor
        weighted_loss = self.lambda_feat * loss
        
        return weighted_loss


class CosineFeatureLoss(nn.Module):
    """
    Alternative implementation using cosine similarity for feature preservation.
    
    This variant of the feature preservation loss uses cosine similarity instead
    of MSE, which can be beneficial when the scale of features varies significantly.
    """
    
    def __init__(
        self,
        reduction: str = 'mean',
        lambda_feat: float = 1.0
    ) -> None:
        """
        Initialize the cosine feature loss.
        
        Args:
            reduction: Reduction method ('none', 'mean', 'sum')
            lambda_feat: Weight factor for the feature loss
        """
        super(CosineFeatureLoss, self).__init__()
        
        self.reduction = reduction
        self.lambda_feat = lambda_feat
    
    def forward(
        self,
        embeddings: Tensor,
        projected_features: Tensor
    ) -> Tensor:
        """
        Compute the cosine feature loss.
        
        Args:
            embeddings: Node embeddings from the ISNE model [num_nodes, embedding_dim]
            projected_features: Projected original features [num_nodes, embedding_dim]
            
        Returns:
            Feature preservation loss based on cosine similarity
        """
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(embeddings, projected_features, dim=1)
        
        # Convert to a loss (1 - similarity)
        loss = 1.0 - cos_sim
        
        # Apply reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        # Apply weight factor
        weighted_loss = self.lambda_feat * loss
        
        return weighted_loss
