"""
ISNE (Inductive Shallow Node Embedding) layer implementation.

This module contains the ISNE layer implementation as described in the original research paper.
The ISNE layer performs feature propagation and aggregation across graph neighborhoods using
attention mechanisms to weight the contribution of each neighbor.
"""

from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import logging

from src.isne.layers.isne_attention import MultiHeadISNEAttention

# Set up logging
logger = logging.getLogger(__name__)


class ISNELayer(nn.Module):
    """
    Implementation of the ISNE (Inductive Shallow Node Embedding) layer as described in the paper.
    
    This layer implements the core feature propagation mechanism of ISNE, which uses
    attention-based neighborhood aggregation to enhance node embeddings with structural
    information from the graph.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        alpha: float = 0.2,
        residual: bool = True,
        normalize: bool = True,
        activation: Optional[Callable[[Tensor], Tensor]] = F.elu
    ) -> None:
        """
        Initialize the ISNE layer.
        
        Args:
            in_features: Dimensionality of input node features
            out_features: Dimensionality of output node features
            hidden_features: Dimensionality of hidden representations (defaults to out_features)
            num_heads: Number of attention heads
            dropout: Dropout probability for regularization
            alpha: Negative slope for LeakyReLU in attention mechanism
            residual: Whether to use residual connections
            normalize: Whether to apply L2 normalization to output features
            activation: Activation function to apply after aggregation
        """
        super(ISNELayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features or out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.normalize = normalize
        self.activation = activation
        
        # Input transformation layer
        self.linear = nn.Linear(in_features, self.hidden_features)
        
        # Multi-head attention layer for neighborhood aggregation
        self.attention = MultiHeadISNEAttention(
            in_features=self.hidden_features,
            out_features=out_features,
            num_heads=num_heads,
            dropout=dropout,
            alpha=alpha,
            concat=True,
            residual=residual
        )
        
        # Dropout for regularization
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Reset learnable parameters to initial values."""
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def forward(
        self, 
        x: Tensor, 
        edge_index: Tensor,
        return_attention: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass for the ISNE layer.
        
        Args:
            x: Node feature tensor [num_nodes, in_features]
            edge_index: Edge index tensor [2, num_edges]
            return_attention: Whether to return attention weights
            
        Returns:
            - Updated node features [num_nodes, out_features]
            - Attention weights if return_attention is True
        """
        # Apply input transformation
        x = self.linear(x)
        
        # Apply activation function
        if self.activation is not None:
            x = self.activation(x)
        
        # Apply dropout
        x = self.dropout_layer(x)
        
        # Apply multi-head attention for neighborhood aggregation
        if return_attention:
            x, attention_weights = self.attention(x, edge_index, return_attention=True)
        else:
            x = self.attention(x, edge_index)
        
        # Apply L2 normalization if specified
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        
        if return_attention:
            return x, attention_weights
        else:
            return x
