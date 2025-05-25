"""
ISNE (Inductive Shallow Node Embedding) model implementation.

This module implements the complete ISNE model architecture as described in the original research paper.
It combines multiple ISNE layers to create a powerful graph neural network for inductive node embedding.
"""

from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import logging

from src.isne.layers.isne_layer import ISNELayer

# Set up logging
logger = logging.getLogger(__name__)


class ISNEModel(nn.Module):
    """
    ISNE (Inductive Shallow Node Embedding) model implementation.
    
    This model implements the full architecture described in the paper
    "Unsupervised Graph Representation Learning with Inductive Shallow Node Embedding".
    It consists of multiple ISNE layers with skip connections to create node embeddings
    that capture both feature and structural information.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        alpha: float = 0.2,
        residual: bool = True,
        normalize_features: bool = True,
        add_self_loops: bool = True,
        activation: Callable[[Tensor], Tensor] = F.elu
    ) -> None:
        """
        Initialize the ISNE model.
        
        Args:
            in_features: Dimensionality of input node features
            hidden_features: Dimensionality of hidden representations
            out_features: Dimensionality of output node embeddings
            num_layers: Number of ISNE layers in the model
            num_heads: Number of attention heads in each layer
            dropout: Dropout probability for regularization
            alpha: Negative slope for LeakyReLU in attention mechanism
            residual: Whether to use residual connections
            normalize_features: Whether to normalize input and output features
            add_self_loops: Whether to add self-loops to the graph
            activation: Activation function to use in the layers
        """
        super(ISNEModel, self).__init__()
        
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.normalize_features = normalize_features
        self.add_self_loops = add_self_loops
        self.activation = activation
        
        # Create layers list
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(ISNELayer(
            in_features=in_features,
            out_features=hidden_features,
            num_heads=num_heads,
            dropout=dropout,
            alpha=alpha,
            residual=residual,
            normalize=normalize_features,
            activation=activation
        ))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(ISNELayer(
                in_features=hidden_features,
                out_features=hidden_features,
                num_heads=num_heads,
                dropout=dropout,
                alpha=alpha,
                residual=residual,
                normalize=normalize_features,
                activation=activation
            ))
        
        # Output layer (if more than one layer)
        if num_layers > 1:
            self.layers.append(ISNELayer(
                in_features=hidden_features,
                out_features=out_features,
                num_heads=num_heads,
                dropout=dropout,
                alpha=alpha,
                residual=residual,
                normalize=normalize_features,
                activation=None  # No activation on output layer
            ))
        
        # Additional components for training
        self.feature_projector = nn.Linear(in_features, out_features)
        self.dropout_layer = nn.Dropout(dropout)
    
    def _prepare_graph(self, edge_index: Tensor) -> Tensor:
        """
        Prepare the graph by adding self-loops if needed.
        
        Args:
            edge_index: Edge index tensor [2, num_edges]
            
        Returns:
            Processed edge index tensor
        """
        if not self.add_self_loops:
            return edge_index
        
        # Get number of nodes
        num_nodes = edge_index.max().item() + 1
        
        # Create self-loop indices
        self_loops = torch.arange(num_nodes, device=edge_index.device)
        self_loops = torch.stack([self_loops, self_loops], dim=0)
        
        # Concatenate with original edges
        return torch.cat([edge_index, self_loops], dim=1)
    
    def forward(
        self, 
        x: Tensor, 
        edge_index: Tensor,
        return_attention: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Dict[int, Tensor]]]:
        """
        Forward pass through the ISNE model.
        
        Args:
            x: Node feature tensor [num_nodes, in_features]
            edge_index: Edge index tensor [2, num_edges]
            return_attention: Whether to return attention weights
            
        Returns:
            - Node embeddings [num_nodes, out_features]
            - Attention weights dict (if return_attention is True)
        """
        # Normalize input features if specified
        if self.normalize_features:
            x = F.normalize(x, p=2, dim=1)
        
        # Prepare graph by adding self-loops if needed
        edge_index = self._prepare_graph(edge_index)
        
        # Store attention weights if required
        attention_weights = {}
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            if return_attention:
                x, att_weights = layer(x, edge_index, return_attention=True)
                attention_weights[i] = att_weights
            else:
                x = layer(x, edge_index)
        
        if return_attention:
            return x, attention_weights
        else:
            return x
    
    def get_embeddings(
        self, 
        x: Tensor, 
        edge_index: Tensor
    ) -> Tensor:
        """
        Compute the final node embeddings.
        
        This method processes the input features through the model
        and returns the final node embeddings.
        
        Args:
            x: Node feature tensor [num_nodes, in_features]
            edge_index: Edge index tensor [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, out_features]
        """
        with torch.no_grad():
            return self.forward(x, edge_index)
    
    def project_features(self, x: Tensor) -> Tensor:
        """
        Project input features to embedding space.
        
        This is used for creating the feature preservation component of the loss function.
        
        Args:
            x: Node feature tensor [num_nodes, in_features]
            
        Returns:
            Projected features [num_nodes, out_features]
        """
        return self.feature_projector(x)
