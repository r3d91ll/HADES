"""
Structural preservation loss implementation for ISNE.

This module implements the structural preservation loss component described in the 
original ISNE paper, which ensures that the learned embeddings preserve the 
structural relationships between nodes in the graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import logging
from typing import Optional, List, Tuple

# Set up logging
logger = logging.getLogger(__name__)


class StructuralPreservationLoss(nn.Module):
    """
    Structural preservation loss for ISNE as described in the paper.
    
    This loss encourages the learned embeddings to preserve the graph structure
    by ensuring that nodes connected in the graph have similar embeddings.
    It follows the graph embedding principle where connected nodes should be
    close in the embedding space.
    """
    
    def __init__(
        self,
        reduction: str = 'mean',
        lambda_struct: float = 1.0,
        negative_samples: int = 5,
        margin: float = 0.1
    ) -> None:
        """
        Initialize the structural preservation loss.
        
        Args:
            reduction: Reduction method ('none', 'mean', 'sum')
            lambda_struct: Weight factor for the structural loss
            negative_samples: Number of negative samples per positive edge
            margin: Margin for the hinge loss
        """
        super(StructuralPreservationLoss, self).__init__()
        
        self.reduction = reduction
        self.lambda_struct = lambda_struct
        self.negative_samples = negative_samples
        self.margin = margin
    
    def sample_negative_edges(
        self,
        edge_index: Tensor,
        num_nodes: int,
        num_neg_samples: int
    ) -> Tensor:
        """
        Sample negative edges (non-existing edges) for contrastive learning.
        
        Args:
            edge_index: Edge index tensor [2, num_edges]
            num_nodes: Number of nodes in the graph
            num_neg_samples: Number of negative samples to generate
            
        Returns:
            Negative edge index tensor [2, num_neg_samples]
        """
        # Create a set of existing edges for fast lookup
        edge_set = set([(edge_index[0, i].item(), edge_index[1, i].item()) 
                        for i in range(edge_index.size(1))])
        
        # Sample negative edges
        neg_src = []
        neg_dst = []
        
        while len(neg_src) < num_neg_samples:
            # Randomly sample source and target nodes
            i = torch.randint(0, num_nodes, (1,)).item()
            j = torch.randint(0, num_nodes, (1,)).item()
            
            # Skip if this is an existing edge or a self-loop
            if i == j or (i, j) in edge_set:
                continue
            
            neg_src.append(i)
            neg_dst.append(j)
        
        # Create negative edge index tensor
        neg_edge_index = torch.tensor([neg_src, neg_dst], dtype=torch.long, 
                                     device=edge_index.device)
        
        return neg_edge_index
    
    def forward(
        self,
        embeddings: Tensor,
        edge_index: Tensor,
        neg_edge_index: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute the structural preservation loss.
        
        Args:
            embeddings: Node embeddings from the ISNE model [num_nodes, embedding_dim]
            edge_index: Edge index tensor [2, num_edges]
            neg_edge_index: Optional pre-sampled negative edge index
            
        Returns:
            Structural preservation loss
        """
        num_nodes = embeddings.size(0)
        
        # If negative edge index not provided, sample it
        if neg_edge_index is None:
            neg_sample_count = edge_index.size(1) * self.negative_samples
            neg_edge_index = self.sample_negative_edges(
                edge_index, num_nodes, neg_sample_count)
        
        # Extract node embeddings for positive edges
        pos_src, pos_dst = edge_index
        pos_src_emb = embeddings[pos_src]
        pos_dst_emb = embeddings[pos_dst]
        
        # Extract node embeddings for negative edges
        neg_src, neg_dst = neg_edge_index
        neg_src_emb = embeddings[neg_src]
        neg_dst_emb = embeddings[neg_dst]
        
        # Compute dot product similarity for positive and negative edges
        pos_sim = torch.sum(pos_src_emb * pos_dst_emb, dim=1)
        neg_sim = torch.sum(neg_src_emb * neg_dst_emb, dim=1)
        
        # Compute max-margin loss: max(0, margin - pos_sim + neg_sim)
        # This encourages pos_sim to be larger than neg_sim by at least the margin
        margin_loss = F.relu(self.margin - pos_sim.view(-1, 1) + neg_sim.view(1, -1))
        
        # Apply reduction
        if self.reduction == 'mean':
            loss = margin_loss.mean()
        elif self.reduction == 'sum':
            loss = margin_loss.sum()
        else:
            loss = margin_loss
        
        # Apply weight factor
        weighted_loss = self.lambda_struct * loss
        
        return weighted_loss


class RandomWalkStructuralLoss(nn.Module):
    """
    Random walk-based structural loss for ISNE.
    
    This variant of the structural loss uses random walks to capture
    higher-order structural information in the graph.
    """
    
    def __init__(
        self,
        reduction: str = 'mean',
        lambda_struct: float = 1.0,
        walk_length: int = 5,
        num_walks: int = 10,
        window_size: int = 2,
        neg_samples: int = 5
    ) -> None:
        """
        Initialize the random walk structural loss.
        
        Args:
            reduction: Reduction method ('none', 'mean', 'sum')
            lambda_struct: Weight factor for the structural loss
            walk_length: Length of random walks
            num_walks: Number of random walks per node
            window_size: Context window size for co-occurrence
            neg_samples: Number of negative samples per positive pair
        """
        super(RandomWalkStructuralLoss, self).__init__()
        
        self.reduction = reduction
        self.lambda_struct = lambda_struct
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.neg_samples = neg_samples
    
    def generate_walks(
        self,
        edge_index: Tensor,
        num_nodes: int
    ) -> List[List[int]]:
        """
        Generate random walks on the graph.
        
        Args:
            edge_index: Edge index tensor [2, num_edges]
            num_nodes: Number of nodes in the graph
            
        Returns:
            List of random walks, each a list of node indices
        """
        # Create adjacency list for efficient neighbor sampling
        adj_list = [[] for _ in range(num_nodes)]
        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            adj_list[src].append(dst)
        
        # Generate random walks
        walks = []
        for _ in range(self.num_walks):
            for start_node in range(num_nodes):
                # Skip nodes with no neighbors
                if not adj_list[start_node]:
                    continue
                
                # Initialize walk with start node
                walk = [start_node]
                
                # Extend walk
                for _ in range(self.walk_length - 1):
                    current = walk[-1]
                    
                    # If no neighbors, break
                    if not adj_list[current]:
                        break
                    
                    # Randomly select a neighbor
                    neighbor_idx = torch.randint(0, len(adj_list[current]), (1,)).item()
                    next_node = adj_list[current][neighbor_idx]
                    walk.append(next_node)
                
                walks.append(walk)
        
        return walks
    
    def extract_walk_pairs(
        self,
        walks: List[List[int]],
        window_size: int
    ) -> Tuple[List[int], List[int]]:
        """
        Extract co-occurring node pairs from random walks.
        
        Args:
            walks: List of random walks
            window_size: Context window size
            
        Returns:
            Tuple of (source nodes, target nodes) for co-occurring pairs
        """
        src_nodes = []
        dst_nodes = []
        
        for walk in walks:
            # Skip short walks
            if len(walk) < 2:
                continue
            
            # Extract pairs within window
            for i in range(len(walk)):
                for j in range(max(0, i - window_size), min(len(walk), i + window_size + 1)):
                    if i == j:  # Skip self-pairs
                        continue
                    
                    src_nodes.append(walk[i])
                    dst_nodes.append(walk[j])
        
        return src_nodes, dst_nodes
    
    def forward(
        self,
        embeddings: Tensor,
        edge_index: Tensor
    ) -> Tensor:
        """
        Compute the random walk structural loss.
        
        Args:
            embeddings: Node embeddings from the ISNE model [num_nodes, embedding_dim]
            edge_index: Edge index tensor [2, num_edges]
            
        Returns:
            Random walk structural loss
        """
        num_nodes = embeddings.size(0)
        
        # Generate random walks
        walks = self.generate_walks(edge_index, num_nodes)
        
        # Extract co-occurring pairs
        src_nodes, dst_nodes = self.extract_walk_pairs(walks, self.window_size)
        
        # Skip if no pairs found
        if not src_nodes:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Convert to tensors
        src_nodes = torch.tensor(src_nodes, dtype=torch.long, device=embeddings.device)
        dst_nodes = torch.tensor(dst_nodes, dtype=torch.long, device=embeddings.device)
        
        # Extract node embeddings for positive pairs
        src_emb = embeddings[src_nodes]
        dst_emb = embeddings[dst_nodes]
        
        # Compute positive pair similarities using dot product
        pos_sim = torch.sum(src_emb * dst_emb, dim=1)
        
        # Generate negative samples
        neg_samples = []
        for _ in range(self.neg_samples):
            # Sample random nodes as negative examples
            neg_nodes = torch.randint(0, num_nodes, (len(src_nodes),), 
                                     device=embeddings.device)
            neg_samples.append(embeddings[neg_nodes])
        
        # Compute negative pair similarities
        neg_sim = torch.stack([torch.sum(src_emb * neg_emb, dim=1) 
                              for neg_emb in neg_samples])
        
        # Compute NCE-inspired loss
        pos_term = -torch.log(torch.sigmoid(pos_sim)).mean()
        neg_term = -torch.log(1 - torch.sigmoid(neg_sim)).mean()
        
        loss = pos_term + neg_term
        
        # Apply weight factor
        weighted_loss = self.lambda_struct * loss
        
        return weighted_loss
