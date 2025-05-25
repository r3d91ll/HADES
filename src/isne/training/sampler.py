"""
Neighborhood sampling implementations for ISNE training.

This module provides efficient sampling strategies for training ISNE models
on large graphs, including node sampling, edge sampling, and neighborhood sampling.
"""

import logging
import os
import numpy as np
import logging
import torch
from typing import List, Tuple, Dict, Union, Any, Optional, Set, Callable, cast, Type
from numpy.typing import NDArray
from torch import Tensor
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes

# For CSR conversion - if torch_geometric.utils.sparse.index2ptr is not available
# we provide a fallback implementation
try:
    from torch_geometric.utils.sparse import index2ptr
except ImportError:
    # Fallback implementation of index2ptr
    def index2ptr(index: torch.Tensor, size: int) -> torch.Tensor:
        ptr = torch.zeros(size + 1, dtype=torch.long, device=index.device)
        torch.scatter_add_(ptr, 0, index + 1, torch.ones_like(index))
        return torch.cumsum(ptr, 0)

# Set up logging
logger = logging.getLogger(__name__)

try:
    import torch_geometric
    from torch_geometric.utils import subgraph
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch Geometric not available. Install with: pip install torch-geometric")
    TORCH_GEOMETRIC_AVAILABLE = False


class NeighborSampler:
    """
    Neighbor sampling strategy for efficient ISNE training.
    
    This sampler provides methods for sampling nodes, edges, and neighborhoods
    to create mini-batches for training the ISNE model on large graphs.
    """
    
    def __init__(
        self,
        edge_index: Tensor,
        num_nodes: int,
        batch_size: int = 32,
        num_hops: int = 1,
        neighbor_size: int = 10,
        directed: bool = False,
        replace: bool = False,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize the neighbor sampler.
        
        Args:
            edge_index: Edge index tensor [2, num_edges]
            num_nodes: Number of nodes in the graph
            batch_size: Batch size for node sampling
            num_hops: Number of hops to sample
            neighbor_size: Maximum number of neighbors to sample per node per hop
            directed: Whether the graph is directed
            replace: Whether to sample with replacement
            seed: Random seed for reproducibility
        """
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.num_hops = num_hops
        self.neighbor_size = neighbor_size
        self.directed = directed
        self.replace = replace
        
        # Create adjacency list for efficient neighbor access
        self._create_adj_list()
        
        # Initialize random number generator
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
    
    def _create_adj_list(self) -> None:
        """
        Create adjacency list from edge index for efficient neighbor access.
        Validates all indices to ensure they are within bounds.
        """
        logger = logging.getLogger(__name__)
        
        # Move tensors to CPU for safer processing
        edge_index_cpu = self.edge_index.cpu() if self.edge_index.is_cuda else self.edge_index
        
        # Validate edge_index to ensure all indices are within bounds
        max_node_idx = edge_index_cpu.max().item()
        actual_num_nodes = max(self.num_nodes, max_node_idx + 1)
        
        if max_node_idx >= self.num_nodes:
            logger.warning(f"Edge index contains node indices up to {max_node_idx} but num_nodes was {self.num_nodes}. "
                         f"Adjusting num_nodes to {actual_num_nodes}.")
            self.num_nodes = actual_num_nodes
        
        # Initialize empty lists for each node
        self.adj_list = [[] for _ in range(self.num_nodes)]
        self.valid_nodes = set(range(self.num_nodes))
        
        # Count invalid edges for logging
        invalid_count = 0
        total_edges = edge_index_cpu.size(1)
        
        # Fill adjacency list with validation
        for i in range(total_edges):
            src = edge_index_cpu[0, i].item()
            dst = edge_index_cpu[1, i].item()
            
            # Skip invalid edges
            if src < 0 or src >= self.num_nodes or dst < 0 or dst >= self.num_nodes:
                invalid_count += 1
                continue
                
            # Add edge to adjacency list
            self.adj_list[src].append(dst)
            
            # If undirected, add reverse edge
            if not self.directed:
                self.adj_list[dst].append(src)
        
        # Create a set of nodes that have at least one neighbor
        self.nodes_with_neighbors = set()
        for i, neighbors in enumerate(self.adj_list):
            if neighbors:
                self.nodes_with_neighbors.add(i)
        
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count}/{total_edges} invalid edges in the edge index")
        
        logger.info(f"Created adjacency list with {self.num_nodes} nodes and {len(self.nodes_with_neighbors)} nodes with neighbors")
    
    def sample_nodes(self, batch_size: Optional[int] = None) -> Tensor:
        """
        Sample random nodes from the graph.
        
        Args:
            batch_size: Number of nodes to sample (defaults to self.batch_size)
            
        Returns:
            Tensor of node indices
        """
        batch_size = batch_size or self.batch_size
        
        # Sample nodes (either uniform or from nodes with neighbors)
        if len(self.nodes_with_neighbors) < batch_size:
            # Not enough nodes with neighbors, sample with replacement
            if self.nodes_with_neighbors:
                nodes = np.array(list(self.nodes_with_neighbors))
                indices = self.rng.choice(len(nodes), size=batch_size, replace=True)
                sampled_nodes = nodes[indices]
            else:
                # No nodes have neighbors, sample from all nodes
                sampled_nodes = self.rng.choice(self.num_nodes, size=batch_size, replace=True)
        else:
            # Enough nodes with neighbors, sample without replacement
            nodes = np.array(list(self.nodes_with_neighbors))
            indices = self.rng.choice(len(nodes), size=batch_size, replace=False)
            sampled_nodes = nodes[indices]
        
        # Convert to tensor
        return torch.tensor(sampled_nodes, dtype=torch.long, device=self.edge_index.device)
    
    def sample_neighbors(self, nodes: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Sample multi-hop neighborhoods for a batch of nodes.
        
        Args:
            nodes: Batch of node indices
            
        Returns:
            Tuple of (node_samples, edge_samples) for each hop:
                - node_samples: List of node indices for each hop
                - edge_samples: List of edge indices for each hop
        """
        # Convert to numpy for efficiency if not already
        if isinstance(nodes, torch.Tensor):
            nodes_np = nodes.cpu().numpy()
        else:
            nodes_np = np.array(nodes)
        
        # Lists to store results
        node_samples = []
        edge_samples = []
        
        # Set of nodes seen so far (for deduplication)
        seen_nodes = set(nodes_np.tolist())
        
        # Current frontier
        frontier = nodes_np
        
        # Sample for each hop
        for _ in range(self.num_hops):
            # Initialize arrays for this hop
            hop_nodes = []
            hop_edges_src = []
            hop_edges_dst = []
            
            # Process each node in the frontier
            for node in frontier:
                # Get neighbors
                neighbors = self.adj_list[node]
                
                if not neighbors:
                    continue
                
                # Sample neighbors (with or without replacement)
                if len(neighbors) > self.neighbor_size:
                    sampled_neighbors = self.rng.choice(
                        neighbors, size=self.neighbor_size, replace=self.replace)
                else:
                    sampled_neighbors = neighbors
                
                # Add sampled edges
                for neighbor in sampled_neighbors:
                    hop_edges_src.append(node)
                    hop_edges_dst.append(neighbor)
                    
                    # Add new nodes
                    if neighbor not in seen_nodes:
                        hop_nodes.append(neighbor)
                        seen_nodes.add(neighbor)
            
            # Update frontier with new nodes
            frontier = np.array(hop_nodes)
            
            # Convert to tensors and add to results
            if hop_edges_src:
                src_tensor = torch.tensor(hop_edges_src, dtype=torch.long, 
                                         device=self.edge_index.device)
                dst_tensor = torch.tensor(hop_edges_dst, dtype=torch.long,
                                         device=self.edge_index.device)
                
                # Add to results
                node_samples.append(torch.tensor(hop_nodes, dtype=torch.long,
                                               device=self.edge_index.device))
                edge_samples.append(torch.stack([src_tensor, dst_tensor], dim=0))
            else:
                # Empty hop
                node_samples.append(torch.tensor([], dtype=torch.long,
                                               device=self.edge_index.device))
                edge_samples.append(torch.zeros((2, 0), dtype=torch.long,
                                               device=self.edge_index.device))
        
        return node_samples, edge_samples
    
    def sample_subgraph(self, nodes: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Sample a subgraph around the given nodes.
        
        Args:
            nodes: Seed node indices
            
        Returns:
            Tuple of (subset_nodes, subgraph_edge_index):
                - subset_nodes: Node indices in the subgraph
                - subgraph_edge_index: Edge index of the subgraph
        """
        # Check that PyTorch Geometric is available
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "PyTorch Geometric is required for subgraph sampling. "
                "Install with: pip install torch-geometric"
            )
        
        # Ensure nodes are within valid range
        num_nodes = self.num_nodes
        nodes = nodes[nodes < num_nodes]
        
        if nodes.numel() == 0:
            # Handle case where no valid nodes remain
            return nodes, torch.empty((2, 0), dtype=torch.long, device=nodes.device)
            
        node_samples, _ = self.sample_neighbors(nodes)
        
        # Combine all sampled nodes
        all_nodes = [nodes]
        all_nodes.extend([n[n < num_nodes] for n in node_samples if n.numel() > 0])
        
        subset_nodes = torch.cat([n for n in all_nodes if n.numel() > 0], dim=0)
        
        # Remove duplicates
        subset_nodes = torch.unique(subset_nodes)
        
        # Extract subgraph
        subgraph_edge_index, _ = subgraph(
            subset_nodes, self.edge_index, relabel_nodes=True)
        
        return subset_nodes, subgraph_edge_index
    
    def sample_triplets(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Sample triplets (anchor, positive, negative) for triplet loss.
        
        Args:
            batch_size: Number of triplets to sample (defaults to self.batch_size)
            
        Returns:
            Tuple of (anchor_indices, positive_indices, negative_indices)
        """
        batch_size = batch_size or self.batch_size
        
        # Lists to store results
        anchor_indices = []
        positive_indices = []
        negative_indices = []
        
        while len(anchor_indices) < batch_size:
            # Sample random anchor node
            anchor = self.rng.randint(0, self.num_nodes)
            
            # Skip if no neighbors
            if not self.adj_list[anchor]:
                continue
            
            # Sample positive neighbor
            positive = self.rng.choice(self.adj_list[anchor])
            
            # Sample negative node (not connected to anchor)
            while True:
                negative = self.rng.randint(0, self.num_nodes)
                
                # Check if not connected to anchor
                if negative != anchor and negative not in self.adj_list[anchor]:
                    break
            
            # Add triplet
            anchor_indices.append(anchor)
            positive_indices.append(positive)
            negative_indices.append(negative)
        
        # Convert to tensors
        device = self.edge_index.device
        return (
            torch.tensor(anchor_indices, dtype=torch.long, device=device),
            torch.tensor(positive_indices, dtype=torch.long, device=device),
            torch.tensor(negative_indices, dtype=torch.long, device=device)
        )
    
    def sample_positive_pairs(self, batch_size: Optional[int] = None) -> Tensor:
        """
        Sample positive pairs for contrastive loss.
        
        Args:
            batch_size: Number of pairs to sample (defaults to self.batch_size)
            
        Returns:
            Tensor of positive pair indices [num_pairs, 2]
        """
        batch_size = batch_size or self.batch_size
        
        # Lists to store results
        pair_indices = []
        
        # Sample pairs of connected nodes
        while len(pair_indices) < batch_size:
            # Sample random node
            src = self.rng.randint(0, self.num_nodes)
            
            # Skip if no neighbors
            if not self.adj_list[src]:
                continue
            
            # Sample random neighbor
            dst = self.rng.choice(self.adj_list[src])
            
            # Add pair
            pair_indices.append([src, dst])
        
        # Convert to tensor
        return torch.tensor(pair_indices, dtype=torch.long, device=self.edge_index.device)
    
    def sample_negative_pairs(self, 
                             positive_pairs: Optional[Tensor] = None,
                             batch_size: Optional[int] = None) -> Tensor:
        """
        Sample negative pairs for contrastive loss.
        
        Args:
            positive_pairs: Optional tensor of positive pair indices to avoid
            batch_size: Number of pairs to sample (defaults to self.batch_size)
            
        
    node_samples, _ = self.sample_neighbors(nodes)
    
    # Combine all sampled nodes
    all_nodes: List[Tensor] = [nodes]
    all_nodes.extend([n[n < num_nodes] for n in node_samples if n.numel() > 0])
    
    subset_nodes = torch.cat([n for n in all_nodes if n.numel() > 0], dim=0)
    
    # Remove duplicates
    subset_nodes = torch.unique(subset_nodes)
    
    # Extract subgraph
    subgraph_edge_index, _ = subgraph(
        subset_nodes, self.edge_index, relabel_nodes=True)
    
    return subset_nodes, subgraph_edge_index

def sample_triplets(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Sample triplets (anchor, positive, negative) for triplet loss.
    
    Args:
        batch_size: Number of triplets to sample (defaults to self.batch_size)
        
    Returns:
        Tuple of (anchor_indices, positive_indices, negative_indices)
    """
    batch_size = batch_size or self.batch_size
    
    # Lists to store results
    anchor_indices: List[int] = []
    positive_indices: List[int] = []
    negative_indices: List[int] = []
    
    while len(anchor_indices) < batch_size:
        # Sample random anchor node
        anchor = self.rng.randint(0, self.num_nodes)
        
        # Skip if no neighbors
        if not self.adj_list[anchor]:
            continue
        
        # Sample positive neighbor
        positive = self.rng.choice(self.adj_list[anchor])
        
        # Sample negative node (not connected to anchor)
        while True:
            negative = self.rng.randint(0, self.num_nodes)
            
            # Check if not connected to anchor
            if negative != anchor and negative not in self.adj_list[anchor]:
                break
        
        # Add triplet
        anchor_indices.append(anchor)
        positive_indices.append(positive)
        negative_indices.append(negative)
    
    # Convert to tensors
    device = self.edge_index.device
    return (
        torch.tensor(anchor_indices, dtype=torch.long, device=device),
        torch.tensor(positive_indices, dtype=torch.long, device=device),
        torch.tensor(negative_indices, dtype=torch.long, device=device)
    )

def sample_positive_pairs(self, batch_size: Optional[int] = None) -> Tensor:
    """
    Sample positive pairs for contrastive loss.
    
    Args:
        batch_size: Number of pairs to sample (defaults to self.batch_size)
        
    Returns:
        Tensor of positive pair indices [num_pairs, 2]
    """
    batch_size = batch_size or self.batch_size
    
    # Lists to store results
    pair_indices: List[Tuple[int, int]] = []
    
    # Sample pairs of connected nodes
    while len(pair_indices) < batch_size:
        # Sample random node
        src = self.rng.randint(0, self.num_nodes)
        
        # Skip if no neighbors
        if not self.adj_list[src]:
            continue
        
        # Sample random neighbor
        dst = self.rng.choice(self.adj_list[src])
        
        # Add pair
        pair_indices.append([src, dst])
    
    # Convert to tensor
    return torch.tensor(pair_indices, dtype=torch.long, device=self.edge_index.device)

def sample_negative_pairs(self, 
                         positive_pairs: Optional[Tensor] = None,
                         batch_size: Optional[int] = None) -> Tensor:
    """
    Sample negative pairs for contrastive loss.
    
    Args:
        positive_pairs: Optional tensor of positive pair indices to avoid
        batch_size: Number of pairs to sample (defaults to self.batch_size)
        
    Returns:
        Tensor of negative pair indices [num_pairs, 2]
    """
    batch_size = batch_size or self.batch_size
    
    # Lists to store results
    pair_indices: List[Tuple[int, int]] = []
    
    # Create set of existing edges for fast lookup
    edge_set = set()
    for i in range(self.edge_index.size(1)):
        src = self.edge_index[0, i].item()
        dst = self.edge_index[1, i].item()
        edge_set.add((src, dst))
        
        # If undirected, add reverse edge
        if not self.directed:
            edge_set.add((dst, src))
    
    # Add positive pairs to avoid if provided
    if positive_pairs is not None:
        for i in range(positive_pairs.size(0)):
            src = positive_pairs[i, 0].item()
            dst = positive_pairs[i, 1].item()
            edge_set.add((src, dst))
            
            # If undirected, add reverse edge
            if not self.directed:
                edge_set.add((dst, src))
    
    # Sample pairs of disconnected nodes
    while len(pair_indices) < batch_size:
        # Sample random nodes
        src = self.rng.randint(0, self.num_nodes)
        dst = self.rng.randint(0, self.num_nodes)
        
        # Skip if self-loop or existing edge
        if src == dst or (src, dst) in edge_set:
            continue
        
        # Add pair
        pair_indices.append([src, dst])
    
    # Convert to tensor
    return torch.tensor(pair_indices, dtype=torch.long, device=self.edge_index.device)


class RandomWalkSampler:
    """
    Random walk-based sampling strategy for efficient ISNE training.
    
    This sampler follows the original ISNE paper's implementation, using random
    walks to generate positive pairs and randomly sampling nodes for negative pairs.
    This approach ensures better connectivity and valid sampling.
    """
    
    def __init__(
        self,
        edge_index: Tensor,
        num_nodes: int,
        batch_size: int = 32,
        walk_length: int = 5,
        context_size: int = 2,
        walks_per_node: int = 10,
        p: float = 1.0,  # Return parameter (like Node2Vec)
        q: float = 1.0,  # In-out parameter (like Node2Vec)
        num_negative_samples: int = 1,
        directed: bool = False,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize the random walk sampler.
        
        Args:
            edge_index: Edge index tensor [2, num_edges]
            num_nodes: Number of nodes in the graph
            batch_size: Batch size for node sampling
            walk_length: Length of the random walks
            context_size: Size of context window for pairs in random walks
            walks_per_node: Number of random walks to start from each node
            p: Return parameter controlling likelihood of returning to previous node
            q: In-out parameter controlling likelihood of visiting new nodes
            num_negative_samples: Number of negative samples per positive sample
            directed: Whether the graph is directed
            seed: Random seed for reproducibility
        """
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples
        self.directed = directed
        
        # Set up CSR representation for fast random walks
        self._setup_csr_format()
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
    
    def sample_nodes(self, batch_size: Optional[int] = None) -> Tensor:
        """
        Sample random nodes from the graph.
        
        Args:
            batch_size: Number of nodes to sample (defaults to self.batch_size)
            
        Returns:
            Tensor of node indices
        """
        batch_size = batch_size or self.batch_size
        
        # Sample nodes uniformly
        # We'll use values from 0 to num_nodes-1 for compatibility
        device = self.edge_index.device
        sampled_nodes = torch.randint(0, self.num_nodes, (batch_size,), device=device)
        
        return sampled_nodes
        
    def _setup_csr_format(self) -> None:
        """Set up CSR format for efficient random walks.
        Following the original ISNE implementation, we sort the edge index and create
        a CSR representation for use in random walks.
        """
        logger = logging.getLogger(__name__)
        
        # Validate edge_index to ensure all indices are within bounds
        edge_index_cpu = self.edge_index.cpu() if self.edge_index.is_cuda else self.edge_index
        max_node_idx = edge_index_cpu.max().item()
        
        if max_node_idx >= self.num_nodes:
            logger.warning(f"Edge index contains node indices that exceed num_nodes ({max_node_idx} >= {self.num_nodes})")
            logger.warning("Filtering out-of-bounds edges to prevent errors.")
            
            # Filter out edges with invalid indices
            valid_edges_mask = (edge_index_cpu[0] < self.num_nodes) & (edge_index_cpu[1] < self.num_nodes)
            edge_index_cpu = edge_index_cpu[:, valid_edges_mask]
            
            if edge_index_cpu.size(1) == 0:
                logger.warning("No valid edges remain after filtering. Creating a small fallback graph.")
                # Create a minimal valid graph to prevent crashes
                num_valid_nodes = min(self.num_nodes, 10)
                src = torch.arange(0, num_valid_nodes)
                dst = torch.arange(0, num_valid_nodes)
                # Create self-loops as fallback
                edge_index_cpu = torch.stack([src, dst], dim=0)
        
        # Ensure we have at least some edges
        if edge_index_cpu.size(1) == 0:
            logger.warning("Empty edge index. Creating a small graph with self-loops.")
            # Create a minimal valid graph to prevent crashes
            num_valid_nodes = min(self.num_nodes, 10)
            src = torch.arange(0, num_valid_nodes)
            dst = torch.arange(0, num_valid_nodes)
            # Create self-loops as fallback
            edge_index_cpu = torch.stack([src, dst], dim=0)
        
        # Sort edge index and convert to CSR format
        row, col = sort_edge_index(edge_index_cpu, num_nodes=self.num_nodes)
        self.rowptr, self.col = index2ptr(row, self.num_nodes), col
        
        # Store effective number of nodes and edges
        self.effective_num_nodes = self.num_nodes
        self.effective_num_edges = edge_index_cpu.size(1)
        
        logger.info(f"CSR format created with {self.effective_num_nodes} nodes and {self.effective_num_edges} edges")
        
        # Check if torch_cluster is available for random walks
        try:
            import torch_cluster
            self.random_walk_fn = torch_cluster.random_walk
            self.has_torch_cluster = True
            logger.info("Using torch_cluster for random walks")
        except ImportError:
            self.has_torch_cluster = False
            logger.warning("torch_cluster not available. Using fallback sampling methods.")
    
    def sample_nodes(self) -> Tensor:
        """
        Sample a batch of nodes for training.
        
        Returns:
            Batch of node indices [batch_size]
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Always sample on CPU to avoid CUDA issues
            batch_idx: np.ndarray = self.rng.choice(
                self.num_nodes, size=self.batch_size, replace=self.replace)
            
            # Validate indices to ensure they're within range
            batch_idx = np.clip(batch_idx, 0, self.num_nodes - 1)
            
            # Create tensor on CPU
            cpu_tensor = torch.tensor(batch_idx, dtype=torch.long)
            
            # Only return CPU tensor regardless of where edge_index is
            # This avoids issues with CUDA device-side asserts
            return cpu_tensor
            
        except Exception as e:
            logger.warning(f"Error in sample_nodes: {str(e)}")
            # Safe fallback - always return CPU tensor
            fallback_size = min(10, self.num_nodes)
            return torch.arange(fallback_size, dtype=torch.long)
    
    def sample_neighbors(self, nodes: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Sample multi-hop neighborhoods for a batch of nodes.
        
        Args:
            nodes: Batch of node indices [batch_size]
            
        Returns:
            Tuple of (node_samples, edge_samples) for each hop:
                - node_samples: List of node indices for each hop
                - edge_samples: List of edge indices for each hop
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Initialize lists to store samples
            # Ensure nodes is on CPU for safe processing
            nodes_cpu = nodes.cpu() if nodes.is_cuda else nodes
            node_samples: List[Tensor] = [nodes_cpu]
            edge_samples: List[Tensor] = []
            
            # Convert nodes to numpy for efficient indexing
            nodes_np: np.ndarray = nodes_cpu.numpy()
            
            # Sample neighbors for each hop
            source_nodes: np.ndarray = nodes_np
            for hop in range(self.num_hops):
                # Initialize arrays to store edges
                hop_edges_src: List[int] = []
                hop_edges_dst: List[int] = []
                
                # Sample neighbors for each source node
                for idx, src_node in enumerate(source_nodes):
                    # Ensure src_node is valid and within bounds
                    if src_node < 0 or src_node >= self.num_nodes:
                        logger.warning(f"Skipping out-of-bounds source node {src_node} (num_nodes={self.num_nodes})")
                        continue
                        
                    # Get neighbors of the node
                    neighbors: List[int] = self.adj_list[src_node]
                    
                    # Skip if no neighbors
                    if len(neighbors) == 0:
                        continue
                    
                    # Sample neighbors with or without replacement
                    num_neighbors = min(self.neighbor_size, len(neighbors))
                    if num_neighbors == 0:
                        continue
                        
                    sampled_neighbors: np.ndarray = self.rng.choice(
                        neighbors, size=num_neighbors, replace=self.replace)
                    
                    # Add edges to the current hop
                    for dst_node in sampled_neighbors:
                        # Ensure dst_node is valid
                        if dst_node < 0 or dst_node >= self.num_nodes:
                            logger.warning(f"Skipping out-of-bounds destination node {dst_node} (num_nodes={self.num_nodes})")
                            continue
                        hop_edges_src.append(src_node)
                        hop_edges_dst.append(dst_node)
                
                # Skip this hop if no edges were sampled
                if len(hop_edges_src) == 0:
                    continue
                
                # Create edge tensor for this hop (on CPU for safety)
                src_tensor = torch.tensor(hop_edges_src, dtype=torch.long)
                dst_tensor = torch.tensor(hop_edges_dst, dtype=torch.long)
                edge_tensor = torch.stack([src_tensor, dst_tensor], dim=0)
                
                # Add to samples
                edge_samples.append(edge_tensor)
                node_samples.append(dst_tensor)
                
                # Update source nodes for the next hop
                source_nodes = np.unique(hop_edges_dst)
            
            return node_samples, edge_samples
            
        except Exception as e:
            logger.warning(f"Error in sample_neighbors: {str(e)}")
            # Return safe empty lists
            return [nodes_cpu], []
        
        return node_samples, edge_samples
    
    def sample_subgraph(self, nodes: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Sample a subgraph around the given nodes.
        
        Args:
            nodes: Node indices to sample neighborhood for
            
        Returns:
            Tuple of (subset_nodes, subgraph_edge_index)
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Ensure all operations are done on CPU for stability
            # Move nodes to CPU if it's on CUDA
            nodes_cpu = nodes.cpu() if nodes.is_cuda else nodes
            edge_index_cpu = self.edge_index.cpu() if self.edge_index.is_cuda else self.edge_index
            
            # Ensure nodes are within valid range
            valid_nodes_mask = (nodes_cpu >= 0) & (nodes_cpu < self.num_nodes)
            nodes_cpu = nodes_cpu[valid_nodes_mask]
            
            if nodes_cpu.numel() == 0:
                logger.warning("No valid nodes remain after filtering. Using empty subgraph.")
                return nodes_cpu, torch.empty((2, 0), dtype=torch.long)
            
            # Sample neighborhood (already uses CPU tensors from our fixes)
            node_samples, _ = self.sample_neighbors(nodes_cpu)
            
            # Combine all sampled nodes
            all_nodes: List[Tensor] = [nodes_cpu]
            all_nodes.extend(node_samples)
            
            # Make sure we have at least one valid node tensor to concatenate
            valid_node_tensors: List[Tensor] = [n for n in all_nodes if n.numel() > 0]
            if not valid_node_tensors:
                logger.warning("No valid node tensors after sampling. Using empty subgraph.")
                return nodes_cpu, torch.empty((2, 0), dtype=torch.long)
            
            # Combine and deduplicate nodes
            subset_nodes = torch.cat(valid_node_tensors, dim=0)
            subset_nodes = torch.unique(subset_nodes)
            
            # Ensure all node indices are valid
            valid_mask = (subset_nodes >= 0) & (subset_nodes < self.num_nodes)
            if not valid_mask.all():
                invalid_count = (~valid_mask).sum().item()
                logger.warning(f"Found {invalid_count} invalid node indices. Filtering them out.")
                subset_nodes = subset_nodes[valid_mask]
                
                if subset_nodes.numel() == 0:
                    logger.warning("No valid nodes remain after filtering. Using empty subgraph.")
                    return nodes_cpu, torch.empty((2, 0), dtype=torch.long)
            
            # Manual subgraph extraction on CPU to avoid PyG/CUDA issues
            # Convert subset_nodes to a set for faster lookups
            subset_nodes_set: Set[int] = set(subset_nodes.tolist())
            
            # Create a mask for edges where both endpoints are in subset_nodes
            edge_mask = torch.tensor(
                [(src.item() in subset_nodes_set and dst.item() in subset_nodes_set)
                 for src, dst in zip(edge_index_cpu[0], edge_index_cpu[1])],
                dtype=torch.bool
            )
            
            # Apply the mask to get filtered edges
            if edge_mask.any():
                filtered_edges = edge_index_cpu[:, edge_mask]
                
                # Create a mapping from original node indices to new indices (0 to len(subset_nodes)-1)
                node_idx_map: Dict[int, int] = {node.item(): i for i, node in enumerate(subset_nodes)}
                
                # Relabel nodes in the edge index using the mapping
                new_src = torch.tensor([node_idx_map[src.item()] for src in filtered_edges[0]], dtype=torch.long)
                new_dst = torch.tensor([node_idx_map[dst.item()] for dst in filtered_edges[1]], dtype=torch.long)
                
                # Create the subgraph edge index
                subgraph_edge_index = torch.stack([new_src, new_dst], dim=0)
            else:
                # No edges in the subgraph
                subgraph_edge_index = torch.empty((2, 0), dtype=torch.long)
            
            # Return tensors on the correct device
            if nodes.is_cuda:
                try:
                    device = nodes.device
                    return subset_nodes.to(device), subgraph_edge_index.to(device)
                except Exception as e:
                    logger.warning(f"Error moving tensors to CUDA: {str(e)}. Returning CPU tensors.")
                    return subset_nodes, subgraph_edge_index
            else:
                return subset_nodes, subgraph_edge_index
                
        except Exception as e:
            logger.warning(f"Error in subgraph sampling: {str(e)}. Using empty subgraph.")
            # Return safe empty tensors
            empty_nodes = torch.empty((0,), dtype=torch.long)
            empty_edges = torch.empty((2, 0), dtype=torch.long)
            return empty_nodes, empty_edges
    
    def sample_triplets(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Sample triplets (anchor, positive, negative) for triplet loss.
        
        Args:
            batch_size: Number of triplets to sample (defaults to self.batch_size)
            
        Returns:
            Tuple of (anchor_indices, positive_indices, negative_indices)
        """
        batch_size = batch_size or self.batch_size
        
        # Lists to store results
        anchor_indices: List[int] = []
        positive_indices: List[int] = []
        negative_indices: List[int] = []
        
        while len(anchor_indices) < batch_size:
            # Sample random anchor node
            anchor = self.rng.randint(0, self.num_nodes)
            
            # Skip if no neighbors
            if not self.adj_list[anchor]:
                continue
            
            # Sample positive neighbor
            positive = self.rng.choice(self.adj_list[anchor])
            
            # Sample negative node (not connected to anchor)
            while True:
                negative = self.rng.randint(0, self.num_nodes)
                
                # Check if not connected to anchor
                if negative != anchor and negative not in self.adj_list[anchor]:
                    break
            
            # Add triplet
            anchor_indices.append(anchor)
            positive_indices.append(positive)
            negative_indices.append(negative)
        
        # Convert to tensors
        device = self.edge_index.device
        return (
            torch.tensor(anchor_indices, dtype=torch.long, device=device),
            torch.tensor(positive_indices, dtype=torch.long, device=device),
            torch.tensor(negative_indices, dtype=torch.long, device=device)
        )
    
    def sample_positive_pairs(self, batch_size: Optional[int] = None) -> Tensor:
        """
        Sample positive pairs using random walks.
        
        This method follows the approach from the original ISNE paper, using random walks
        to generate positive context pairs. If torch_cluster is available, it uses the
        efficient implementation, otherwise falls back to a simpler approach.
        
        Args:
            batch_size: Number of pairs to sample (defaults to self.batch_size)
            
        Returns:
            Tensor of positive pair indices [num_pairs, 2]
        """
        logger = logging.getLogger(__name__)
        batch_size = batch_size or self.batch_size
        
        try:
            # Generate random walks for positive samples
            if self.has_torch_cluster:
                return self._sample_positive_pairs_torch_cluster(batch_size)
            else:
                return self._sample_positive_pairs_fallback(batch_size)
                
        except Exception as e:
            logger.error(f"Error sampling positive pairs: {str(e)}")
            logger.warning("Using fallback positive pairs due to sampling error.")
            return self._generate_fallback_positive_pairs(batch_size)
            
    def _sample_positive_pairs_torch_cluster(self, batch_size: int) -> Tensor:
        """
        Sample positive pairs using torch_cluster random walks.
        
        Args:
            batch_size: Number of pairs to sample
            
        Returns:
            Tensor of positive pair indices [num_pairs, 2]
        """
        # Sample source nodes uniformly
        num_start_nodes = min(batch_size, self.effective_num_nodes) 
        start_nodes = torch.randint(0, self.effective_num_nodes, (num_start_nodes,), dtype=torch.long)
        
        # Generate random walks
        # Each walk is of form [start_node, node_1, node_2, ..., node_k]
        walks = self.random_walk_fn(self.rowptr, self.col, start_nodes, self.walk_length, self.p, self.q)
        
        # Extract pairs from walks
        # For each walk, we generate pairs (start_node, node_i) for i in context window
        pos_pairs: List[List[int]] = []
        for walk in walks:
            # Skip walks that didn't go anywhere (isolated nodes)
            if torch.any(walk < 0):
                continue
                
            # Extract valid nodes from the walk (no padding or negative values)
            valid_walk = walk[walk >= 0]
            if len(valid_walk) < 2:  # Need at least 2 nodes for a pair
                continue
                
            # Source node is the first node in the walk
            src = valid_walk[0].item()
            
            # Context nodes are the next `context_size` nodes
            context_size = min(self.context_size, len(valid_walk) - 1)
            for i in range(1, context_size + 1):
                dst = valid_walk[i].item()
                pos_pairs.append([src, dst])
        
        # Convert to tensor
        if not pos_pairs:
            # If no valid pairs were found, use fallback
            logger.warning("No valid positive pairs from random walks. Using fallback.")
            return self._generate_fallback_positive_pairs(batch_size)
            
        pos_pairs_tensor = torch.tensor(pos_pairs, dtype=torch.long)
        
        # Ensure we have enough pairs
        if len(pos_pairs_tensor) < batch_size:
            # If we don't have enough pairs, generate more or use fallback
            logger.info(f"Only found {len(pos_pairs_tensor)}/{batch_size} positive pairs. Adding more.")
            fallback_pairs = self._generate_fallback_positive_pairs(batch_size - len(pos_pairs_tensor))
            pos_pairs_tensor = torch.cat([pos_pairs_tensor, fallback_pairs], dim=0)
            
        # Return the requested number of pairs
        return pos_pairs_tensor[:batch_size]
            
    def _sample_positive_pairs_fallback(self, batch_size: int) -> Tensor:
        """
        Fallback method for sampling positive pairs when torch_cluster is not available.
        This uses our adjacency list representation to sample connected node pairs.
        
        Args:
            batch_size: Number of pairs to sample
            
        Returns:
            Tensor of positive pair indices [num_pairs, 2]
        """
        logger = logging.getLogger(__name__)
        
        # Create a simplified adjacency list from the CSR representation
        adj_list: Dict[int, List[int]] = {}
        for i in range(self.effective_num_nodes):
            # Get neighbors for node i
            start_ptr = self.rowptr[i].item()
            end_ptr = self.rowptr[i+1].item()
            if start_ptr < end_ptr:  # Only include nodes with at least one neighbor
                neighbors = self.col[start_ptr:end_ptr].tolist()
                if neighbors:  # Double-check we have neighbors
                    adj_list[i] = neighbors
        
        # Get nodes that have neighbors
        nodes_with_neighbors: List[int] = list(adj_list.keys())
        
        if not nodes_with_neighbors:
            logger.warning("No valid nodes with neighbors found. Using fallback pairs.")
            return self._generate_fallback_positive_pairs(batch_size)
            
        # Get destination nodes from edge index
        dst_indices: List[int] = []
        for _ in range(batch_size):
            if not nodes_with_neighbors:
                break
                
            # Sample random node with neighbors
            src = random.choice(nodes_with_neighbors)
            
            # Sample random neighbor
            dst = random.choice(adj_list[src])
            
            # Add the pair
            pair_indices.append([src, dst])
            
        # Convert pairs to tensor
        if not pair_indices:
            logger.warning("Could not sample any valid positive pairs. Using fallback.")
            return self._generate_fallback_positive_pairs(batch_size)
        
        # Convert to tensor
        pos_pairs_tensor = torch.tensor(pair_indices, dtype=torch.long)
        
        # Ensure we have enough pairs
        if len(pos_pairs_tensor) < batch_size:
            logger.info(f"Only found {len(pos_pairs_tensor)}/{batch_size} positive pairs. Adding fallback pairs.")
            fallback_pairs = self._generate_fallback_positive_pairs(batch_size - len(pos_pairs_tensor))
            pos_pairs_tensor = torch.cat([pos_pairs_tensor, fallback_pairs], dim=0)
        
        # Final validation - make absolutely sure all indices are in bounds
        valid_mask = (
            (pos_pairs_tensor[:, 0] >= 0) & 
            (pos_pairs_tensor[:, 0] < self.num_nodes) & 
            (pos_pairs_tensor[:, 1] >= 0) & 
            (pos_pairs_tensor[:, 1] < self.num_nodes)
        )
        
        if not torch.all(valid_mask):
            logger.warning(f"Filtered {(~valid_mask).sum().item()} out-of-bounds pairs from result")
            pos_pairs_tensor = pos_pairs_tensor[valid_mask]
            if len(pos_pairs_tensor) < batch_size:
                additional_pairs = self._generate_fallback_positive_pairs(batch_size - len(pos_pairs_tensor))
                pos_pairs_tensor = torch.cat([pos_pairs_tensor, additional_pairs], dim=0)
                
        # Return the requested number of pairs (limit to batch_size)
        return pos_pairs_tensor[:batch_size]
            
    def sample_negative_pairs(self, positive_pairs: Optional[Tensor] = None, batch_size: Optional[int] = None) -> Tensor:
        """
        Sample negative pairs using random sampling.
        
        Following the original ISNE implementation, this randomly samples nodes to create
        negative pairs. We avoid explicit edge existence checks as these are rare in sparse graphs.
        
        Args:
            positive_pairs: Optional tensor of positive pairs to avoid (not used in this implementation)
            batch_size: Number of pairs to sample (defaults to self.batch_size)
            
        Returns:
            Tensor of negative pair indices [num_pairs, 2]
        """
        logger = logging.getLogger(__name__)
        batch_size = batch_size or self.batch_size
        
        try:
            # Following the original ISNE implementation, we generate negative pairs by random sampling
            # This is simpler and more efficient than explicit edge filtering in sparse graphs
            neg_pairs = []
            for _ in range(batch_size):
                # Sample random source and target nodes
                src = random.randint(0, self.num_nodes - 1)
                dst = random.randint(0, self.num_nodes - 1)
                
                # Avoid self-loops
                while dst == src:
                    dst = random.randint(0, self.num_nodes - 1)
                    
                neg_pairs.append([src, dst])
            
            # Convert to tensor
            neg_pairs_tensor = torch.tensor(neg_pairs, dtype=torch.long)
            
            # Final validation - ensure all indices are in bounds
            valid_mask = (
                (neg_pairs_tensor[:, 0] >= 0) & 
                (neg_pairs_tensor[:, 0] < self.num_nodes) & 
                (neg_pairs_tensor[:, 1] >= 0) & 
                (neg_pairs_tensor[:, 1] < self.num_nodes)
            )
            
            if not torch.all(valid_mask):
                logger.warning(f"Filtered {(~valid_mask).sum().item()} out-of-bounds negative pairs")
                neg_pairs_tensor = neg_pairs_tensor[valid_mask]
                
                # If we lost too many pairs, supplement with additional ones
                if len(neg_pairs_tensor) < batch_size:
                    additional_pairs = self._generate_fallback_negative_pairs(batch_size - len(neg_pairs_tensor))
                    neg_pairs_tensor = torch.cat([neg_pairs_tensor, additional_pairs], dim=0)
            
            return neg_pairs_tensor[:batch_size]
            
        except Exception as e:
            logger.error(f"Error sampling negative pairs: {str(e)}")
            logger.warning("Using fallback negative pairs due to sampling error")
            return self._generate_fallback_negative_pairs(batch_size)
            
    def _generate_fallback_positive_pairs(self, batch_size: int) -> Tensor:
        """
        Generate fallback positive pairs when sampling fails.
        These are synthetic pairs based on node proximity in the index space.
        
        Args:
            batch_size: Number of pairs to generate
            
        Returns:
            Tensor of fallback positive pair indices [num_pairs, 2]
        """
        logger = logging.getLogger(__name__)
        logger.warning("Generating fallback positive pairs")
        
        try:
            fallback_pairs = []
            
            # First try to use actual connected nodes from the graph's edges
            edge_index_cpu = self.edge_index.cpu() if self.edge_index.is_cuda else self.edge_index
            
            # Sample from existing edges if available
            if edge_index_cpu.size(1) > 0:
                # Get a subset of edges to use as positive pairs
                edge_count = min(batch_size, edge_index_cpu.size(1))
                edge_indices = torch.randperm(edge_index_cpu.size(1))[:edge_count]
                
                for idx in edge_indices:
                    src = edge_index_cpu[0, idx].item()
                    dst = edge_index_cpu[1, idx].item()
                    
                    # Validate the indices
                    if 0 <= src < self.num_nodes and 0 <= dst < self.num_nodes:
                        fallback_pairs.append([src, dst])
            
            # If we still need more pairs, create sequential neighboring pairs
            if len(fallback_pairs) < batch_size:
                for i in range(min(batch_size - len(fallback_pairs), self.num_nodes - 1)):
                    if i + 1 < self.num_nodes:
                        fallback_pairs.append([i, i + 1])
            
            # Convert to tensor
            if fallback_pairs:
                fallback_tensor = torch.tensor(fallback_pairs, dtype=torch.long)
                
                # Move to device if needed
                if self.edge_index.is_cuda:
                    try:
                        return fallback_tensor.to(self.edge_index.device)
                    except Exception as e:
                        logger.warning(f"Error moving fallback pairs to CUDA: {str(e)}. Returning CPU tensor.")
                        return fallback_tensor
                else:
                    return fallback_tensor
            else:
                # Create completely synthetic pairs with consecutive indices
                pairs = torch.stack([torch.arange(batch_size), torch.arange(batch_size) + 1 % self.num_nodes], dim=1)
                return pairs
                
        except Exception as e:
            logger.warning(f"Error in _generate_fallback_positive_pairs: {str(e)}")
            # Last resort fallback
            return torch.tensor([[i, (i + 1) % max(2, self.num_nodes)] for i in range(batch_size)], dtype=torch.long)
    
    def sample_negative_pairs(
        self, 
        positive_pairs: Optional[Tensor] = None,
        batch_size: Optional[int] = None
    ) -> Tensor:
        """
        Sample negative pairs for contrastive loss.
        
        Args:
            positive_pairs: Optional tensor of positive pair indices to avoid
            batch_size: Number of pairs to sample (defaults to self.batch_size)
            
        Returns:
            Tensor of negative pair indices [num_pairs, 2]
        """
        logger = logging.getLogger(__name__)
        batch_size = batch_size or self.batch_size
        
        # Define valid node indices based on num_nodes
        valid_node_indices = set(range(self.num_nodes))
        
        # Create set of existing edges to avoid
        edge_set = set()
        
        if self.edge_index.size(1) > 0:
            # Get edge indices as tuples
            edge_index_cpu = self.edge_index.cpu() if self.edge_index.is_cuda else self.edge_index
            for i in range(edge_index_cpu.size(1)):
                src, dst = edge_index_cpu[0, i].item(), edge_index_cpu[1, i].item()
                # Only add edges that connect valid nodes
                if src in valid_node_indices and dst in valid_node_indices:
                    edge_set.add((src, dst))
                    if not self.directed:
                        edge_set.add((dst, src))
        
        # Add positive pairs to edge set if provided
        if positive_pairs is not None and positive_pairs.size(0) > 0:
            pos_pairs_cpu = positive_pairs.cpu() if positive_pairs.is_cuda else positive_pairs
            for i in range(pos_pairs_cpu.size(0)):
                src, dst = pos_pairs_cpu[i, 0].item(), pos_pairs_cpu[i, 1].item()
                # Only add positive pairs that connect valid nodes
                if src in valid_node_indices and dst in valid_node_indices:
                    edge_set.add((src, dst))
                    if not self.directed:
                        edge_set.add((dst, src))
        
        try:
            # Lists to store results
            pair_indices = []
            max_attempts = batch_size * 20  # Increase attempt limit for better sampling
            attempts = 0
            
            # Sample pairs of unconnected nodes
            if len(valid_node_indices) < 2:
                logger.warning("Not enough valid nodes for negative sampling. Using fallback approach.")
                return self._generate_fallback_negative_pairs(batch_size, edge_set)
            
            # Convert to list for random sampling
            valid_nodes_list = list(valid_node_indices)
            
            # Sample pairs of disconnected nodes
            while len(pair_indices) < batch_size and attempts < max_attempts:
                attempts += 1
                
                # Sample random nodes from valid nodes only
                src_idx = self.rng.randint(0, len(valid_nodes_list))
                dst_idx = self.rng.randint(0, len(valid_nodes_list))
                src = valid_nodes_list[src_idx]
                dst = valid_nodes_list[dst_idx]
                
                # Skip if self-loop or existing edge
                if src == dst or (src, dst) in edge_set:
                    continue
                
                # Extra validation (should not be necessary but added for safety)
                if src < 0 or src >= self.num_nodes or dst < 0 or dst >= self.num_nodes:
                    continue
                
                # Add pair
                pair_indices.append([src, dst])
            
            # Check if we have enough pairs
            if len(pair_indices) < batch_size:
                logger.warning(f"Could only sample {len(pair_indices)} valid negative pairs after {attempts} attempts")
                
                # If we have no pairs at all, create fallback pairs
                if len(pair_indices) == 0:
                    logger.warning("No valid negative pairs found. Creating fallback pairs.")
                    
                    # Find nodes that are far apart in index space
                    for i in range(min(batch_size, self.num_nodes // 2)):
                        src = i
                        dst = (i + self.num_nodes // 2) % self.num_nodes
                        
                        # Skip if it's an existing edge
                        if (src, dst) in edge_set:
                            continue
                            
                        pair_indices.append([src, dst])
                        if len(pair_indices) >= batch_size:
                            break
                    
                    # If still no valid pairs, create artificial negative pairs
                    if len(pair_indices) == 0:
                        logger.warning("Creating artificial negative pairs as last resort")
                        for i in range(min(batch_size, self.num_nodes - 10)):
                            src = i
                            dst = i + 10 # Assume nodes with 10 distance are not connected
                            if dst < self.num_nodes:
                                pair_indices.append([src, dst])
            
            # Convert to tensor (on CPU first for safety)
            if pair_indices:
                pairs_tensor = torch.tensor(pair_indices, dtype=torch.long)
                
                # Move to device if needed
                if self.edge_index.is_cuda:
                    try:
                        return pairs_tensor.to(self.edge_index.device)
                    except Exception as e:
                        logger.warning(f"Error moving negative pairs to CUDA: {str(e)}. Returning CPU tensor.")
                        return pairs_tensor
                else:
                    return pairs_tensor
            else:
                # Return empty tensor with correct shape
                return torch.empty((0, 2), dtype=torch.long)
                
        except Exception as e:
            logger.warning(f"Error in sample_negative_pairs: {str(e)}")
            # Use fallback method
            return self._generate_fallback_negative_pairs(batch_size)
            
    def _generate_fallback_negative_pairs(self, batch_size: int) -> Tensor:
        """
        Generate fallback negative pairs when sampling fails.
        Following the original ISNE implementation, we generate random pairs.
        
        Args:
            batch_size: Number of pairs to generate
            
        Returns:
            Tensor of fallback negative pair indices [num_pairs, 2]
        """
        logger = logging.getLogger(__name__)
        logger.warning("Generating fallback negative pairs")
        
        try:
            # Simple fallback: generate random pairs that avoid self-loops
            # This follows the original ISNE implementation's approach
            max_idx = max(1, self.num_nodes - 1)  # Ensure we have at least 2 indices
            src_indices = torch.randint(0, max_idx + 1, (batch_size,), dtype=torch.long)
            
            # For each source, generate a destination that isn't the same node
            dst_indices = []
            for src in src_indices:
                src_val = src.item()
                dst_val = random.randint(0, max_idx)
                
        except Exception as e:
            logger.warning(f"Error in _generate_fallback_negative_pairs: {str(e)}")
            # Last resort fallback with minimal validation
            try:
                # Create pairs that are unlikely to be connected
                return torch.tensor([[i, (i + 17) % max(2, self.num_nodes)] for i in range(batch_size)], dtype=torch.long)
            except:
                # Absolute last resort - create minimal valid tensor
                return torch.tensor([[0, 1]], dtype=torch.long).repeat(batch_size, 1)
