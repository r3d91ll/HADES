"""
ISNE Attention mechanism implementation.

This module contains the attention mechanism used in the ISNE model as described
in the original research paper. The attention mechanism computes weights for 
neighborhood nodes based on their feature similarity and structural importance.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import logging

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
        # Apply feature transformation
        # Shape: [N, hidden_dim]
        Wh = torch.mm(x, self.W)
        
        # Get source and target nodes from edge_index
        src, dst = edge_index[0], edge_index[1]
        
        # Prepare concatenated features for attention computation
        # Shape: [num_edges, 2 * hidden_dim]
        edge_h = torch.cat([Wh[src], Wh[dst]], dim=1)
        
        # Compute attention scores
        # Shape: [num_edges, 1]
        edge_attention = self.leakyrelu(torch.matmul(edge_h, self.a))
        
        # Create a sparse attention matrix and fill with computed attention scores
        attention = torch.sparse_coo_tensor(
            edge_index, edge_attention.view(-1), 
            size=(x.size(0), x.size(0))
        ).to_dense()
        
        # Apply softmax row-wise to normalize the attention scores for each node
        # This ensures attention weights sum to 1 for each node's neighborhood
        attention = F.softmax(attention, dim=1)
        
        # Apply dropout to attention weights
        attention = self.dropout_layer(attention)
        
        # Compute weighted sum of neighbor features using attention weights
        h_prime = torch.matmul(attention, Wh)
        
        if return_attention_weights:
            return h_prime, attention
        else:
            return h_prime


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
        # Process each attention head
        head_outputs = []
        all_attentions = []
        
        for attention in self.attentions:
            if return_attention_weights:
                head_output, attention_weights = attention(x, edge_index, True)
                head_outputs.append(head_output)
                all_attentions.append(attention_weights)
            else:
                head_output = attention(x, edge_index, False)
                head_outputs.append(head_output)
        
        # Combine outputs from all heads
        if self.concat:
            # Concatenate outputs from all heads
            multi_head_output = torch.cat(head_outputs, dim=1)
        else:
            # Average outputs from all heads
            multi_head_output = torch.mean(torch.stack(head_outputs), dim=0)
        
        # Apply output projection
        output = self.out_proj(multi_head_output)
        
        # Apply residual connection if enabled
        if self.residual:
            if self.residual_proj is not None:
                output = output + self.residual_proj(x)
            else:
                output = output + x
        
        if return_attention_weights:
            # Average attention weights from all heads
            avg_attention = torch.mean(torch.stack(all_attentions), dim=0)
            return output, avg_attention
        else:
            return output
