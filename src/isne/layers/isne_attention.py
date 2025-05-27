"""
ISNE Attention mechanism implementation.

This module contains the attention mechanism used in the ISNE model as described
in the original research paper. The attention mechanism computes weights for 
neighboring nodes based on their features, allowing the model to focus on the
most relevant connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import logging
from typing import Optional, Tuple, Union, cast

# Set up logging
logger = logging.getLogger(__name__)


class ISNEAttention(nn.Module):
    """
    Attention mechanism for ISNE as described in the original paper.
    
    This implementation follows the attention mechanism described in the paper
    "Unsupervised Graph Representation Learning with Inductive Shallow Node Embedding"
    where attention weights are computed between nodes and their neighbors to 
    determine the importance of each neighbor in the feature aggregation process.
    """
    
    def __init__(
            self,
            in_features: int,
            hidden_dim: int = 32,
            dropout: float = 0.1,
            alpha: float = 0.2,
            concat: bool = True
        ) -> None:
        """
        Initialize the ISNE attention mechanism.
        
        Args:
            in_features: Dimensionality of input features
            hidden_dim: Dimensionality of attention hidden layer
            dropout: Dropout rate applied to attention weights
            alpha: Negative slope of LeakyReLU activation
            concat: Whether to concatenate or average multi-head outputs
        """
        super(ISNEAttention, self).__init__()
        
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Learnable parameters for the attention mechanism
        # W: Linear transformation for input features
        self.W = nn.Parameter(torch.zeros(size=(in_features, hidden_dim)))
        # a: Attention vector for computing attention coefficients
        self.a = nn.Parameter(torch.zeros(size=(2 * hidden_dim, 1)))
        
        # Initialize parameters with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.a.data)
        
        # Dropout layer for regularization
        self.dropout_layer = nn.Dropout(dropout)
        
        # LeakyReLU activation with specified negative slope
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            return_attention_weights: bool = False
        ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass for the attention mechanism.
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Graph connectivity [2, num_edges]
            return_attention_weights: If True, return the attention weights
            
        Returns:
            - Node features with attention applied
            - Attention weights (if return_attention_weights is True)
        """
        # Linear transformation of input features
        Wh = torch.matmul(x, self.W)  # [num_nodes, hidden_dim]
        
        # Get source and target nodes from edge_index
        source, target = edge_index[0], edge_index[1]
        
        # Prepare attention coefficients
        # Concatenate source and target node features
        a_input = torch.cat([Wh[source], Wh[target]], dim=1)  # [num_edges, 2*hidden_dim]
        
        # Compute attention coefficients
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))  # [num_edges]
        
        # Apply softmax to get attention weights (node-wise normalization)
        # First, we need to convert to a sparse format for efficient normalization
        num_nodes = x.size(0)
        
        # Create a mapping from edges to their indices
        edge_to_idx = torch.zeros(num_nodes, num_nodes, device=x.device) - 1
        edge_to_idx[source, target] = torch.arange(source.size(0), device=x.device)
        
        # Apply softmax normalization row-wise (per node)
        attention = torch.zeros_like(e)
        for i in range(num_nodes):
            # Get indices of all edges from node i
            mask = (source == i)
            if mask.any():
                # Apply softmax only to existing edges
                attention[mask] = F.softmax(e[mask], dim=0)
        
        # Apply dropout to attention weights
        attention = self.dropout_layer(attention)
        
        # Apply attention weights to aggregate node features
        output = torch.zeros_like(x)
        for i in range(num_nodes):
            # Get indices of all edges from node i
            neighbors = target[source == i]
            if neighbors.numel() > 0:
                # Get corresponding attention weights
                neighbor_weights = attention[source == i]
                # Weighted sum of neighbor features
                output[i] = torch.sum(Wh[neighbors] * neighbor_weights.unsqueeze(-1), dim=0)
        
        # Return values with correct typing
        if return_attention_weights:
            return output, attention
        else:
            return output


class MultiHeadISNEAttention(nn.Module):
    """
    Multi-head attention mechanism for ISNE.
    
    Extends the basic attention mechanism to use multiple attention
    heads for capturing different aspects of node relationships.
    """
    
    def __init__(
            self,
            in_features: int,
            out_features: int,
            num_heads: int = 8,
            dropout: float = 0.1,
            alpha: float = 0.2,
            concat: bool = True,
            residual: bool = True
        ) -> None:
        """
        Initialize multi-head ISNE attention.
        
        Args:
            in_features: Dimensionality of input features
            out_features: Dimensionality of output features
            num_heads: Number of attention heads
            dropout: Dropout rate applied to attention weights
            alpha: Negative slope of LeakyReLU activation
            concat: Whether to concatenate or average attention heads
            residual: Whether to use residual connection
        """
        super(MultiHeadISNEAttention, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat = concat
        self.residual = residual
        
        # Calculate the hidden dimension for each attention head
        if concat:
            assert out_features % num_heads == 0, "Output dimension must be divisible by number of heads"
            self.head_dim = out_features // num_heads
        else:
            self.head_dim = out_features
        
        # Create multiple attention heads
        self.attentions = nn.ModuleList([
            ISNEAttention(
                in_features=in_features,
                hidden_dim=self.head_dim,
                dropout=dropout,
                alpha=alpha,
                concat=concat
            ) for _ in range(num_heads)
        ])
        
        # Output projection layer
        self.out_proj = nn.Linear(
            self.head_dim * num_heads if concat else self.head_dim,
            out_features
        )
        
        # Residual connection projection if input and output dimensions differ
        self.residual_proj = None
        if residual and in_features != out_features:
            self.residual_proj = nn.Linear(in_features, out_features)
    
    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            return_attention_weights: bool = False
        ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass for multi-head attention.
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Graph connectivity [2, num_edges]
            return_attention_weights: If True, return the attention weights
            
        Returns:
            - Node features with multi-head attention applied
            - Attention weights (if return_attention_weights is True)
        """
        # Apply each attention head
        head_outputs = []
        all_attentions = []
        
        for attention in self.attentions:
            if return_attention_weights:
                out, att = attention(x, edge_index, return_attention_weights=True)
                head_outputs.append(out)
                all_attentions.append(att)
            else:
                out = attention(x, edge_index, return_attention_weights=False)
                head_outputs.append(out)
        
        # Combine outputs from all heads
        if self.concat:
            # Concatenate all head outputs
            output = torch.cat(head_outputs, dim=1)
        else:
            # Average all head outputs
            output = torch.mean(torch.stack(head_outputs), dim=0)
        
        # Apply output projection
        output = self.out_proj(output)
        
        # Apply residual connection if specified
        if self.residual:
            if self.residual_proj is not None:
                output = output + self.residual_proj(x)
            else:
                output = output + x
        
        # Return values with correct typing
        if return_attention_weights:
            # Average attention weights from all heads
            avg_attention = torch.mean(torch.stack(all_attentions), dim=0)
            return output, avg_attention
        else:
            return output
