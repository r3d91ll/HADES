"""
ISNE (Inductive Shallow Node Embedding) model implementation.

This module implements the correct ISNE model as described in the original research paper:
"Unsupervised Graph Representation Learning with Inductive Shallow Node Embedding"

The key insight of ISNE is replacing the traditional lookup table encoder with a 
neighborhood aggregation encoder: h(v) = (1/|N_v|) * Σ_{n∈N_v} θ_n
"""

from typing import List, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import logging

# Set up logging
logger = logging.getLogger(__name__)


class ISNEModel(nn.Module):
    """
    ISNE (Inductive Shallow Node Embedding) model implementation.
    
    This implements the simple but effective ISNE approach where node embeddings
    are computed by averaging the parameter vectors of their neighbors.
    
    Key components:
    - Parameter matrix θ: [num_nodes, embedding_dim] - learnable parameters for each node
    - Neighborhood aggregation: For each node v, h(v) = average of θ for neighbors of v
    - Skip-gram objective: Used to train the θ parameters
    """
    
    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        learning_rate: float = 0.001,
        negative_samples: int = 5,
        context_window: int = 5,
        device: Optional[Union[str, torch.device]] = None
    ) -> None:
        """
        Initialize the ISNE model.
        
        Args:
            num_nodes: Total number of nodes in the graph
            embedding_dim: Dimensionality of node embeddings
            learning_rate: Learning rate for optimization
            negative_samples: Number of negative samples for skip-gram training
            context_window: Context window size for random walks
            device: Device to place the model on
        """
        super(ISNEModel, self).__init__()
        
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.negative_samples = negative_samples
        self.context_window = context_window
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        # The core of ISNE: parameter matrix θ for each node
        # This is what gets learned during training
        self.theta = nn.Parameter(
            torch.randn(num_nodes, embedding_dim, device=self.device) * 0.1
        )
        
        # Output projection for skip-gram objective (if needed)
        self.output_projection = nn.Linear(embedding_dim, num_nodes, bias=False, device=self.device)
        
        logger.info(f"Initialized ISNE model with {num_nodes} nodes, {embedding_dim}D embeddings")
        logger.info(f"Total parameters: {self.theta.numel() + sum(p.numel() for p in self.output_projection.parameters())}")
    
    def get_node_embeddings(self, node_ids: Tensor, neighbor_lists: List[List[int]]) -> Tensor:
        """
        Compute ISNE embeddings for given nodes using neighborhood aggregation.
        
        Args:
            node_ids: Node IDs to compute embeddings for [batch_size]
            neighbor_lists: List of neighbor lists for each node [batch_size][neighbors]
            
        Returns:
            Node embeddings [batch_size, embedding_dim]
        """
        batch_size = len(node_ids)
        embeddings = []
        
        for i in range(batch_size):
            neighbors = neighbor_lists[i]
            
            if len(neighbors) == 0:
                # Handle isolated nodes - use zeros or the node's own theta
                node_id = node_ids[i].item()
                if 0 <= node_id < self.num_nodes:
                    embedding = self.theta[node_id]
                else:
                    embedding = torch.zeros(self.embedding_dim, device=self.device)
            else:
                # Core ISNE computation: average neighbor parameters
                # Ensure neighbor indices are valid
                valid_neighbors = [n for n in neighbors if 0 <= n < self.num_nodes]
                if len(valid_neighbors) == 0:
                    embedding = torch.zeros(self.embedding_dim, device=self.device)
                else:
                    neighbor_params = self.theta[valid_neighbors]  # [|N_v|, embedding_dim]
                    embedding = neighbor_params.mean(dim=0)  # [embedding_dim]
            
            embeddings.append(embedding)
        
        return torch.stack(embeddings)  # [batch_size, embedding_dim]
    
    def forward(
        self, 
        node_ids: Tensor,
        neighbor_lists: List[List[int]],
        return_logits: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass through the ISNE model.
        
        Args:
            node_ids: Node IDs [batch_size]
            neighbor_lists: Neighbor lists for each node [batch_size][neighbors]
            return_logits: Whether to return logits for skip-gram training
            
        Returns:
            Node embeddings [batch_size, embedding_dim] or 
            (embeddings, logits) if return_logits=True
        """
        # Get ISNE embeddings using neighborhood aggregation
        embeddings = self.get_node_embeddings(node_ids, neighbor_lists)
        
        if return_logits:
            # Compute logits for skip-gram objective
            logits = self.output_projection(embeddings)  # [batch_size, num_nodes]
            return embeddings, logits
        
        return embeddings
    
    def compute_loss(
        self,
        center_nodes: Tensor,
        context_nodes: Tensor,
        neighbor_lists: List[List[int]],
        negative_nodes: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute skip-gram loss for ISNE training.
        
        Args:
            center_nodes: Center node IDs [batch_size]
            context_nodes: Context node IDs [batch_size, context_size]
            neighbor_lists: Neighbor lists for center nodes [batch_size][neighbors]
            negative_nodes: Negative sample node IDs [batch_size, negative_samples]
            
        Returns:
            Skip-gram loss
        """
        # Get embeddings for center nodes using ISNE
        center_embeddings = self.get_node_embeddings(center_nodes, neighbor_lists)
        
        # Positive loss: predict context nodes
        context_embeddings = self.theta[context_nodes]  # [batch_size, context_size, embedding_dim]
        positive_scores = torch.bmm(
            context_embeddings, 
            center_embeddings.unsqueeze(-1)
        ).squeeze(-1)  # [batch_size, context_size]
        
        positive_loss = -F.logsigmoid(positive_scores).mean()
        
        # Negative loss: don't predict negative samples
        if negative_nodes is not None:
            negative_embeddings = self.theta[negative_nodes]  # [batch_size, negative_samples, embedding_dim]
            negative_scores = torch.bmm(
                negative_embeddings,
                center_embeddings.unsqueeze(-1)
            ).squeeze(-1)  # [batch_size, negative_samples]
            
            negative_loss = -F.logsigmoid(-negative_scores).mean()
        else:
            negative_loss = torch.tensor(0.0, device=self.device)
        
        return positive_loss + negative_loss
    
    def get_all_embeddings(self, adjacency_lists: List[List[int]]) -> Tensor:
        """
        Get embeddings for all nodes in the graph.
        
        Args:
            adjacency_lists: Adjacency lists for all nodes [num_nodes][neighbors]
            
        Returns:
            All node embeddings [num_nodes, embedding_dim]
        """
        all_node_ids = torch.arange(self.num_nodes, device=self.device)
        return self.get_node_embeddings(all_node_ids, adjacency_lists)
    
    def save_embeddings(self, filepath: str, adjacency_lists: List[List[int]]) -> None:
        """
        Save node embeddings to file.
        
        Args:
            filepath: Path to save embeddings
            adjacency_lists: Adjacency lists for all nodes
        """
        embeddings = self.get_all_embeddings(adjacency_lists)
        torch.save({
            'embeddings': embeddings,
            'theta': self.theta,
            'num_nodes': self.num_nodes,
            'embedding_dim': self.embedding_dim
        }, filepath)
        logger.info(f"Saved ISNE embeddings to {filepath}")
    
    def load_embeddings(self, filepath: str) -> None:
        """
        Load node embeddings from file.
        
        Args:
            filepath: Path to load embeddings from
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.theta.data = checkpoint['theta']
        logger.info(f"Loaded ISNE embeddings from {filepath}")
    
    def save(self, filepath: str) -> None:
        """
        Save the complete model state.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'state_dict': self.state_dict(),
            'num_nodes': self.num_nodes,
            'embedding_dim': self.embedding_dim,
            'learning_rate': self.learning_rate,
            'negative_samples': self.negative_samples,
            'context_window': self.context_window,
            'device': str(self.device)
        }, filepath)
        logger.info(f"Saved ISNE model to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ISNEModel':
        """
        Load a model from file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded ISNE model
        """
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        
        # Extract model configuration
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            model = cls(
                num_nodes=config['num_nodes'],
                embedding_dim=config['embedding_dim'],
                learning_rate=config.get('learning_rate', 0.001),
                negative_samples=config.get('negative_samples', 5),
                context_window=config.get('context_window', 5),
                device=config.get('device', 'cpu')
            )
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                raise ValueError("Model state dict not found in checkpoint")
                
        else:
            raise ValueError(f"Invalid checkpoint format: missing model_config. Found keys: {list(checkpoint.keys())}")
        
        logger.info(f"Loaded ISNE model from {filepath}")
        return model


def create_neighbor_lists_from_edge_index(edge_index: Tensor, num_nodes: int) -> List[List[int]]:
    """
    Create adjacency lists from edge index tensor.
    
    Args:
        edge_index: Edge index tensor [2, num_edges]
        num_nodes: Number of nodes in the graph
        
    Returns:
        Adjacency lists [num_nodes][neighbors]
    """
    adjacency_lists = [[] for _ in range(num_nodes)]
    
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if 0 <= src < num_nodes and 0 <= dst < num_nodes:
            adjacency_lists[src].append(dst)
    
    return adjacency_lists