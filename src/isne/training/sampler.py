"""Neighborhood sampling implementations for ISNE training.

This module provides efficient sampling strategies for training ISNE models
on large graphs, including node sampling, edge sampling, and neighborhood sampling.
"""

import logging
import os
import random
import numpy as np
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
        """Convert an index tensor to a compressed sparse row (CSR) pointer format.
        
        Args:
            index: The index tensor to convert
            size: The size of the pointer array
        
        Returns:
            A tensor of pointers in CSR format
        """
        ptr = torch.zeros(size + 1, dtype=torch.long, device=index.device)
        torch.cumsum(torch.bincount(index, minlength=size), dim=0, out=ptr[1:])
        return ptr

# Set up logging
logger = logging.getLogger(__name__)

# Check for PyTorch Geometric availability
try:
    import torch_geometric
    from torch_geometric.utils import subgraph
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch Geometric not available. Install with: pip install torch-geometric")
    TORCH_GEOMETRIC_AVAILABLE = False

# Check for torch_cluster availability (for efficient random walks)
try:
    import torch_cluster
    from torch_cluster import random_walk
    TORCH_CLUSTER_AVAILABLE = True
except ImportError:
    logger.warning("torch_cluster not available. Using fallback random walk implementation.")
    TORCH_CLUSTER_AVAILABLE = False


class NeighborSampler:
    """Neighbor sampling strategy for efficient ISNE training.
    
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
        """Initialize the neighbor sampler.
        
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
        """Create an adjacency list for efficient neighbor access.
        
        This transforms the edge_index into a list of lists, where each sublist
        contains the neighbors of the corresponding node.
        """
        # Create adjacency list
        self.adj_list: List[List[int]] = [[] for _ in range(self.num_nodes)]
        
        # Initialize nodes_with_neighbors set
        self.nodes_with_neighbors: Set[int] = set()
        
        # Handle empty edge index case
        if self.edge_index.numel() == 0:
            return
            
        # Move to CPU for more efficient list creation
        edge_index_cpu = self.edge_index.cpu()
        
        # Check for out-of-bounds indices
        max_node_idx = edge_index_cpu.max().item()
        if max_node_idx >= self.num_nodes:
            logger.warning(f"Edge index contains node indices that exceed num_nodes ({max_node_idx} >= {self.num_nodes})")
            logger.warning("Filtering out-of-bounds edges to prevent errors.")
            valid_edges_mask = (edge_index_cpu[0] < self.num_nodes) & (edge_index_cpu[1] < self.num_nodes)
            edge_index_cpu = edge_index_cpu[:, valid_edges_mask]
        
        # Fill adjacency list from edge index
        for i in range(edge_index_cpu.size(1)):
            src = edge_index_cpu[0, i].item()
            dst = edge_index_cpu[1, i].item()
            
            # Add edge to adjacency list
            self.adj_list[src].append(dst)
            self.nodes_with_neighbors.add(src)
            
            # If undirected, add the reverse edge as well
            if not self.directed:
                self.adj_list[dst].append(src)
                self.nodes_with_neighbors.add(dst)
        
        # Create set of nodes that have neighbors (for efficient sampling)
 # The nodes_with_neighbors set has already been populated during edge iteration
    
    def sample_nodes(self, batch_size: Optional[int] = None) -> Tensor:
        """Sample nodes uniformly at random from the graph.
        
        Args:
            batch_size: Number of nodes to sample (defaults to self.batch_size)
            
        Returns:
            Tensor of sampled node indices [batch_size]
        """
        batch_size = batch_size or self.batch_size
        
        # Check if we need to enforce replacement
        replace = self.replace
        if not replace and batch_size > self.num_nodes:
            logger = logging.getLogger(__name__)
            logger.warning(f"Cannot sample {batch_size} unique nodes from a graph with {self.num_nodes} nodes. Using replacement.")
            replace = True
            
        # Sample node indices
        node_indices: np.ndarray = self.rng.choice(self.num_nodes, size=batch_size, replace=replace)
        
        # Convert to tensor and match device of edge_index
        return torch.tensor(node_indices, dtype=torch.long, device=self.edge_index.device)
    
    def sample_edges(self, batch_size: Optional[int] = None) -> Tensor:
        """Sample random edges from the graph.
        
        Args:
            batch_size: Number of edges to sample (defaults to self.batch_size)
            
        Returns:
            Tensor of sampled edge indices [batch_size, 2]
        """
        batch_size = batch_size or self.batch_size
        
        # Ensure we don't try to sample more edges than exist
        num_edges = self.edge_index.size(1)
        sample_size = min(batch_size, num_edges)
        
        if num_edges == 0:
            # Handle empty edge case
            return torch.zeros((0, 2), dtype=torch.long, device=self.edge_index.device)
        
        # Sample random edge indices
        edge_indices: np.ndarray = self.rng.choice(num_edges, size=sample_size, replace=self.replace)
        
        # Create pairs from sampled edges
        edge_pairs: List[List[int]] = []
        for idx in edge_indices:
            src = self.edge_index[0, idx].item()
            dst = self.edge_index[1, idx].item()
            edge_pairs.append([src, dst])
        
        # Convert to tensor
        return torch.tensor(edge_pairs, dtype=torch.long, device=self.edge_index.device)
    
    def sample_neighbors(self, nodes: Tensor, size: Optional[int] = None) -> Tensor:
        """Sample neighbors for a batch of nodes.
        
        Args:
            nodes: Node indices to sample neighbors for
            size: Maximum number of neighbors to sample per node
            
        Returns:
            Tensor of neighbor indices [num_neighbors]
        """
        size = size or self.neighbor_size
        neighbors: List[int] = []
        
        # Process each node in the batch
        for node in nodes.tolist():
            if node < 0 or node >= self.num_nodes:
                continue
                
            node_neighbors = self.adj_list[node]
            
            if not node_neighbors:
                continue
                
            # Sample neighbors for this node
            if len(node_neighbors) > size:
                sampled = self.rng.choice(node_neighbors, size, replace=self.replace).tolist()
            else:
                sampled = node_neighbors
                
            neighbors.extend(sampled)
        
        # Remove duplicates while preserving order
        unique_neighbors = []
        seen = set()
        for n in neighbors:
            if n not in seen and n not in nodes.tolist():
                seen.add(n)
                unique_neighbors.append(n)
        
        if not unique_neighbors:
            return torch.tensor([], dtype=torch.long, device=nodes.device)
            
        return torch.tensor(unique_neighbors, dtype=torch.long, device=nodes.device)
    
    def sample_subgraph(self, nodes: Tensor) -> Tuple[Tensor, Tensor]:
        """Sample a subgraph around the given nodes.
        
        Args:
            nodes: Node indices to sample neighborhood for
            
        Returns:
            Tuple of (subset_nodes, subset_edge_index)
        """
        # Use PyTorch Geometric's subgraph function if available
        if TORCH_GEOMETRIC_AVAILABLE:
            subset_nodes = nodes
            
            # Expand to multi-hop neighborhood
            for _ in range(self.num_hops):
                neighbors = self.sample_neighbors(subset_nodes)
                if len(neighbors) == 0:
                    break
                subset_nodes = torch.cat([subset_nodes, neighbors])
                # Remove duplicates
                subset_nodes = torch.unique(subset_nodes)
            
            # Get the edge_index for the subgraph
            subset_edge_index, _ = subgraph(
                subset_nodes, 
                self.edge_index, 
                relabel_nodes=True,
                num_nodes=self.num_nodes
            )
            
            return subset_nodes, subset_edge_index
        else:
            # Fallback implementation for when PyTorch Geometric is not available
            subset_nodes = nodes.tolist()
            node_map: Dict[int, int] = {}
            
            # Expand to multi-hop neighborhood
            for _ in range(self.num_hops):
                neighbor_nodes: List[int] = []
                for node in subset_nodes:
                    if node < 0 or node >= self.num_nodes:
                        continue
                    neighbor_nodes.extend(self.adj_list[node])
                
                # Add neighbors to subset_nodes
                subset_nodes.extend(neighbor_nodes)
                # Remove duplicates
                subset_nodes = list(set(subset_nodes))
            
            # Create mapping from original node IDs to subgraph node IDs
            for i, node in enumerate(subset_nodes):
                node_map[node] = i
            
            # Create edge_index for the subgraph
            subset_edges: List[List[int]] = []
            for src_idx, src in enumerate(subset_nodes):
                for dst in self.adj_list[src]:
                    if dst in node_map:  # Only add edges where both endpoints are in the subgraph
                        dst_idx = node_map[dst]
                        subset_edges.append([src_idx, dst_idx])
            
            if not subset_edges:
                return (
                    torch.tensor(subset_nodes, dtype=torch.long, device=nodes.device),
                    torch.zeros((2, 0), dtype=torch.long, device=nodes.device)
                )
            
            subset_edge_tensor = torch.tensor(subset_edges, dtype=torch.long, device=nodes.device).t()
            
            return (
                torch.tensor(subset_nodes, dtype=torch.long, device=nodes.device),
                subset_edge_tensor
            )
    
    def sample_triplets(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Sample triplets (anchor, positive, negative) for triplet loss.
        
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
        """Sample positive pairs (connected nodes) from the graph.
        
        Args:
            batch_size: Number of pairs to sample (defaults to self.batch_size)
            
        Returns:
            Tensor of positive pair indices [num_pairs, 2]
        """
        batch_size = batch_size or self.batch_size
        pairs: List[List[int]] = []
        attempts = 0
        max_attempts = batch_size * 10  # Limit attempts to avoid infinite loops
        
        while len(pairs) < batch_size and attempts < max_attempts:
            # Sample random nodes that have neighbors
            if len(self.nodes_with_neighbors) > 0:
                valid_nodes = list(self.nodes_with_neighbors)
                src_idx = self.rng.choice(len(valid_nodes))
                src = valid_nodes[src_idx]
            else:
                # Fallback to random sampling
                src = self.rng.randint(0, self.num_nodes)
            
            # Get neighbors of the source node
            neighbors = self.adj_list[src]
            if not neighbors:
                attempts += 1
                continue
                
            # Sample a neighbor randomly
            dst_idx = self.rng.choice(len(neighbors))
            dst = neighbors[dst_idx]
            
            # Add the pair
            pairs.append([src, dst])
            attempts += 1
        
        if len(pairs) == 0:
            # If we couldn't find any pairs, create some synthetic ones
            return self._generate_fallback_positive_pairs(batch_size)
        
        return torch.tensor(pairs, dtype=torch.long, device=self.edge_index.device)
    
    def _generate_fallback_positive_pairs(self, batch_size: int) -> Tensor:
        """Generate fallback positive pairs when sampling fails.
        These are synthetic pairs based on node proximity in the index space.
        
        Args:
            batch_size: Number of pairs to generate
            
        Returns:
            Tensor of positive pair indices [batch_size, 2]
        """
        pairs: List[List[int]] = []
        
        # Create pairs based on node proximity in the index space
        for _ in range(batch_size):
            src = self.rng.randint(0, max(1, self.num_nodes - 1))
            dst = min(src + 1, self.num_nodes - 1)  # Adjacent node
            
            if src == dst:  # Handle edge case
                dst = max(0, src - 1)
                
            pairs.append([src, dst])
        
        return torch.tensor(pairs, dtype=torch.long, device=self.edge_index.device)
    
    def sample_negative_pairs(self, positive_pairs: Optional[Tensor] = None, batch_size: Optional[int] = None) -> Tensor:
        """Sample negative pairs (disconnected nodes) from the graph.
        
        Args:
            positive_pairs: Optional tensor of positive pairs to avoid
            batch_size: Number of pairs to sample (defaults to self.batch_size)
            
        Returns:
            Tensor of negative pair indices [num_pairs, 2]
        """
        batch_size = batch_size or self.batch_size
        
        # Convert positive pairs to set for O(1) lookup
        positive_set: Set[Tuple[int, int]] = set()
        if positive_pairs is not None:
            for i in range(positive_pairs.size(0)):
                src = positive_pairs[i, 0].item()
                dst = positive_pairs[i, 1].item()
                positive_set.add((src, dst))
                if not self.directed:
                    positive_set.add((dst, src))
        
        # Sample negative pairs
        pairs: List[List[int]] = []
        attempts = 0
        max_attempts = batch_size * 20  # Increase attempt limit for better sampling
        
        while len(pairs) < batch_size and attempts < max_attempts:
            # Sample two random nodes
            src = self.rng.randint(0, self.num_nodes)
            dst = self.rng.randint(0, self.num_nodes)
            
            # Skip if src and dst are the same
            if src == dst:
                attempts += 1
                continue
                
            # Skip if (src, dst) is a positive pair
            if (src, dst) in positive_set:
                attempts += 1
                continue
                
            # Skip if dst is a neighbor of src
            if dst in self.adj_list[src]:
                attempts += 1
                continue
                
            # Add the negative pair
            pairs.append([src, dst])
            attempts += 1
        
        if len(pairs) == 0:
            # If we couldn't find any pairs, create some synthetic ones
            return self._generate_fallback_negative_pairs(batch_size)
        
        return torch.tensor(pairs, dtype=torch.long, device=self.edge_index.device)
    
    def _generate_fallback_negative_pairs(self, batch_size: int) -> Tensor:
        """Generate fallback negative pairs when sampling fails.
        Following the original ISNE implementation, we generate random pairs.
        
        Args:
            batch_size: Number of pairs to generate
            
        Returns:
            Tensor of negative pair indices [batch_size, 2]
        """
        pairs: List[List[int]] = []
        
        # Create random pairs
        for _ in range(batch_size):
            src = self.rng.randint(0, self.num_nodes)
            dst = self.rng.randint(0, self.num_nodes)
            
            # Ensure src != dst
            while src == dst and self.num_nodes > 1:
                dst = self.rng.randint(0, self.num_nodes)
                
            pairs.append([src, dst])
        
        return torch.tensor(pairs, dtype=torch.long, device=self.edge_index.device)


class RandomWalkSampler:
    """Random walk-based sampling strategy for efficient ISNE training.
    
    This sampler follows the original ISNE paper's implementation, using random
    walks to generate positive pairs and randomly sampling nodes for negative pairs.
    This approach ensures better connectivity and valid sampling.
    """
    # Class attribute to track torch_cluster availability
    has_torch_cluster: bool = False
    random_walk_fn: Optional[Callable] = None
    
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
        # Initialize basic attributes directly
        self.edge_index = edge_index
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        
        # Store parameters
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        
        # Initialize effective_num_nodes to avoid attribute errors
        self.effective_num_nodes = self.num_nodes
        self.effective_num_edges = self.edge_index.size(1) if self.edge_index.numel() > 0 else 0
        
        # Initialize random number generator
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
        self.num_negative_samples = num_negative_samples
        self.directed = directed
        
        # Set up CSR representation for fast random walks
        self._setup_csr_format()
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            import random
            random.seed(seed)
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
    
    def _setup_csr_format(self) -> None:
        """Set up CSR format for efficient random walks.
        Following the original ISNE implementation, we sort the edge index and create
        a CSR representation for use in random walks.
        """
        logger = logging.getLogger(__name__)
        
        # Handle empty edge index case
        if self.edge_index.numel() == 0:
            # Create empty CSR format
            self.rowptr = torch.zeros(self.num_nodes + 1, dtype=torch.long)
            self.col = torch.zeros(0, dtype=torch.long)
            logger.info(f"CSR format created with {self.num_nodes} nodes and 0 edges")
            return
            
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
        
        # Create adjacency list for other sampling methods
        self.adj_list: List[List[int]] = [[] for _ in range(self.num_nodes)]
        for i in range(edge_index_cpu.size(1)):
            src = edge_index_cpu[0, i].item()
            dst = edge_index_cpu[1, i].item()
            
            # Add edge to adjacency list
            self.adj_list[src].append(dst)
            
            # If undirected, add the reverse edge as well
            if not self.directed:
                self.adj_list[dst].append(src)
        
        # Create set of nodes that have neighbors (for efficient sampling)
 # The nodes_with_neighbors set has already been populated during edge iteration
        
        # Store effective number of nodes and edges
        self.effective_num_nodes = self.num_nodes
        self.effective_num_edges = edge_index_cpu.size(1)
        
        logger.info(f"CSR format created with {self.effective_num_nodes} nodes and {self.effective_num_edges} edges")
        
        # Check if torch_cluster is available for random walks
        self.has_torch_cluster = False  # Default to False
        try:
            import torch_cluster
            self.random_walk_fn = torch_cluster.random_walk
            self.has_torch_cluster = True
            logger.info("Using torch_cluster for random walks")
        except ImportError:
            # Already set to False by default
            logger.warning("torch_cluster not available. Using fallback sampling methods.")
    
    def sample_nodes(self, batch_size: Optional[int] = None) -> Tensor:
        """Sample random nodes from the graph.
        
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
        
    def sample_neighbors(self, nodes: Tensor) -> Tuple[List[List[int]], List[List[int]]]:
        """Sample multi-hop neighborhoods for a batch of nodes.
        
        Args:
            nodes: Batch of node indices
            
        Returns:
            Tuple of (node_samples, edge_samples) for each hop:
                - node_samples: List of node indices for each hop
                - edge_samples: List of edge indices for each hop
        """
        # Initialize to handle the attribute error gracefully
        if not hasattr(self, 'has_torch_cluster'):
            self.has_torch_cluster = False
        logger = logging.getLogger(__name__)
        
        try:
            # Initialize lists to store samples
            # Ensure nodes is on CPU for safe processing
            nodes_cpu = nodes.cpu() if nodes.is_cuda else nodes
            node_samples: List[Tensor] = [nodes_cpu]
            edge_samples: List[Tensor] = []
            
            # Sample random walks from each node
            if self.has_torch_cluster:
                # Use torch_cluster for efficient random walks
                if callable(self.random_walk_fn):
                    walks = self.random_walk_fn(
                        self.rowptr, self.col, nodes_cpu, 
                        self.walk_length, self.p, self.q
                    )
                else:
                    # If random_walk_fn is not callable, fall back to basic implementation
                    logger.warning("random_walk_fn is not callable, falling back to basic implementation")
                    # Return empty lists as fallback
                    return ([], [])
                
                # Process walks to extract neighbors and edges
                for hop in range(1, self.walk_length + 1):
                    if hop >= len(walks[0]):
                        break
                        
                    # Get nodes at this hop
                    hop_nodes = walks[:, hop]
                    # Filter out invalid nodes (padding)
                    valid_mask = hop_nodes >= 0
                    hop_nodes = hop_nodes[valid_mask]
                    
                    if hop_nodes.numel() == 0:
                        continue
                        
                    # Add to samples
                    node_samples.append(hop_nodes)
                    
                    # Create edges between previous hop and current hop
                    prev_nodes = walks[:, hop-1][valid_mask]
                    edge_tensor = torch.stack([prev_nodes, hop_nodes], dim=0)
                    edge_samples.append(edge_tensor)
            else:
                # Fallback to simple sampling
                source_nodes = nodes_cpu
                for _ in range(self.walk_length):
                    next_nodes = []
                    edge_src = []
                    edge_dst = []
                    
                    for src in source_nodes:
                        src_item = src.item()
                        if src_item < 0 or src_item >= self.num_nodes or not self.adj_list[src_item]:
                            continue
                            
                        # Sample a random neighbor
                        neighbors = self.adj_list[src_item]
                        dst_idx = self.rng.choice(len(neighbors))
                        dst_item = neighbors[dst_idx]
                        
                        next_nodes.append(dst_item)
                        edge_src.append(src_item)
                        edge_dst.append(dst_item)
                    
                    if not next_nodes:
                        break
                        
                    # Create tensors for this hop
                    hop_nodes = torch.tensor(next_nodes, dtype=torch.long)
                    edge_tensor = torch.stack([
                        torch.tensor(edge_src, dtype=torch.long),
                        torch.tensor(edge_dst, dtype=torch.long)
                    ], dim=0)
                    
                    node_samples.append(hop_nodes)
                    edge_samples.append(edge_tensor)
                    
                    # Update source nodes for next hop
                    source_nodes = hop_nodes
            
            return node_samples, edge_samples
        except Exception as e:
            logger.error(f"Error in sample_neighbors: {str(e)}")
            return [nodes_cpu], []
    
    def sample_subgraph(self, nodes: Tensor) -> Tuple[Tensor, Tensor]:
        """Sample a subgraph around the given nodes.
        
        Args:
            nodes: Seed node indices
            
        Returns:
            Tuple of (subset_nodes, subgraph_edge_index):
                - subset_nodes: Node indices in the subgraph
                - subgraph_edge_index: Edge index of the subgraph
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Sample neighborhoods using random walks
            node_samples, _ = self.sample_neighbors(nodes)
            
            # Combine all sampled nodes
            all_nodes: List[Tensor] = []
            for nodes_tensor in node_samples:
                if isinstance(nodes_tensor, torch.Tensor) and nodes_tensor.numel() > 0:
                    all_nodes.append(nodes_tensor)
            
            if not all_nodes:
                return nodes, torch.zeros((2, 0), dtype=torch.long, device=nodes.device)
                
            # Concatenate and deduplicate
            subset_nodes = torch.cat(all_nodes, dim=0)
            subset_nodes = torch.unique(subset_nodes)
            
            # Extract subgraph
            if TORCH_GEOMETRIC_AVAILABLE:
                # Use PyG's subgraph extraction
                subgraph_edge_index, _ = subgraph(
                    subset_nodes, self.edge_index, relabel_nodes=True
                )
            else:
                # Manual subgraph extraction
                subset_nodes_set = set(subset_nodes.tolist())
                edge_list = []
                
                # Create mapping from original node indices to subgraph indices
                node_map = {node.item(): i for i, node in enumerate(subset_nodes)}
                
                # Extract edges where both endpoints are in the subgraph
                edge_index_cpu = self.edge_index.cpu() if self.edge_index.is_cuda else self.edge_index
                for i in range(edge_index_cpu.size(1)):
                    src = edge_index_cpu[0, i].item()
                    dst = edge_index_cpu[1, i].item()
                    
                    if src in subset_nodes_set and dst in subset_nodes_set:
                        edge_list.append([node_map[src], node_map[dst]])
                
                if not edge_list:
                    subgraph_edge_index = torch.zeros((2, 0), dtype=torch.long, device=nodes.device)
                else:
                    edge_tensor = torch.tensor(edge_list, dtype=torch.long, device=nodes.device).t()
                    subgraph_edge_index = edge_tensor
            
            return subset_nodes, subgraph_edge_index
        except Exception as e:
            logger.error(f"Error in sample_subgraph: {str(e)}")
            return nodes, torch.zeros((2, 0), dtype=torch.long, device=nodes.device)
    
    def sample_triplets(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """Sample triplets (anchor, positive, negative) for triplet loss.
        
        Args:
            batch_size: Number of triplets to sample (defaults to self.batch_size)
            
        Returns:
            Tuple of (anchor_indices, positive_indices, negative_indices)
        """
        batch_size = batch_size or self.batch_size
        
        # Get positive pairs from random walks
        positive_pairs = self.sample_positive_pairs(batch_size)
        
        # Use the first nodes as anchors
        anchor_indices = positive_pairs[:, 0]
        positive_indices = positive_pairs[:, 1]
        
        # Sample negative pairs
        negative_pairs = self.sample_negative_pairs(positive_pairs, batch_size)
        
        # Use the second nodes as negatives
        negative_indices = negative_pairs[:, 1]
        
        return anchor_indices, positive_indices, negative_indices
    
    def sample_positive_pairs(self, batch_size: Optional[int] = None) -> Tensor:
        """Sample positive pairs using random walks.
        
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
        
        # Initialize the has_torch_cluster attribute if it doesn't exist
        if not hasattr(self, 'has_torch_cluster'):
            self.has_torch_cluster = False
            try:
                # Check if torch_cluster is available
                import torch_cluster
                self.has_torch_cluster = True
            except ImportError:
                logger.warning("torch_cluster not available. Using fallback sampling methods.")
        
        # Use appropriate implementation based on availability
        if self.has_torch_cluster:
            return self._sample_positive_pairs_torch_cluster(batch_size)
        else:
            return self._sample_positive_pairs_fallback(batch_size)
            
    def _sample_positive_pairs_torch_cluster(self, batch_size: int) -> Tensor:
        """Sample positive pairs using torch_cluster random walks.
        
        Args:
            batch_size: Number of pairs to sample
            
        Returns:
            Tensor of positive pair indices [num_pairs, 2]
        """
        # Initialize random_walk_fn if not already set
        if not hasattr(self, 'random_walk_fn') or self.random_walk_fn is None:
            try:
                from torch_cluster import random_walk
                self.random_walk_fn = random_walk
            except ImportError:
                logger.warning("torch_cluster.random_walk not available, falling back")
                self.has_torch_cluster = False
                return self._sample_positive_pairs_fallback(batch_size)
        # Sample source nodes uniformly
        num_start_nodes = min(batch_size, self.effective_num_nodes) 
        start_nodes = torch.randint(0, self.effective_num_nodes, (num_start_nodes,), dtype=torch.long)
        
        # Generate random walks
        # Each walk is of form [start_node, node_1, node_2, ..., node_k]
        if not callable(self.random_walk_fn):
            logger.warning("random_walk_fn is not callable, falling back to fallback implementation")
            return self._sample_positive_pairs_fallback(batch_size)
            
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
        """Fallback method for sampling positive pairs when torch_cluster is not available.
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
            
        # Initialize pairs list
        pair_indices: List[List[int]] = []
        
        # Get destination nodes from edge index
        for _ in range(batch_size):
            if not nodes_with_neighbors:
                break
                
            # Sample random node with neighbors
            src = random.choice(nodes_with_neighbors)
            
            # Sample random neighbor
            dst = random.choice(adj_list[src])
            
            # Add the pair
            pair_indices.append([src, dst])
            
        # Convert to tensor
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
        """Sample negative pairs using random sampling.
        
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
            # For speed, we just sample random node pairs and assume they're negative
            # This is a reasonable approximation for sparse graphs
            src = torch.randint(0, self.num_nodes, (batch_size,), dtype=torch.long)
            dst = torch.randint(0, self.num_nodes, (batch_size,), dtype=torch.long)
            
            # Avoid self-loops
            for i in range(batch_size):
                if src[i] == dst[i]:
                    dst[i] = (dst[i] + 1) % self.num_nodes
            
            return torch.stack([src, dst], dim=1)
            
        except Exception as e:
            logger.error(f"Error sampling negative pairs: {str(e)}")
            logger.warning("Using fallback negative pairs due to sampling error.")
            return self._generate_fallback_negative_pairs(batch_size)
    
    def _generate_fallback_positive_pairs(self, batch_size: int) -> Tensor:
        """Generate fallback positive pairs when sampling fails.
        
        Args:
            batch_size: Number of pairs to generate
            
        Returns:
            Tensor of positive pair indices [batch_size, 2]
        """
        logger = logging.getLogger(__name__)
        logger.warning(f"Generating {batch_size} fallback positive pairs")
        
        # For fallback, we'll use the actual edge index if available
        # Otherwise, we'll create synthetic pairs of consecutive indices
        device = self.edge_index.device
        
        if self.edge_index.size(1) > 0:
            # Sample from existing edges
            indices = torch.randint(0, self.edge_index.size(1), (batch_size,), device=device)
            pairs = self.edge_index[:, indices].t()
        else:
            # Create synthetic pairs (consecutive indices)
            src = torch.randint(0, max(1, self.num_nodes - 1), (batch_size,), device=device)
            dst = src + 1
            pairs = torch.stack([src, dst], dim=1)
        
        return pairs
    
    def _generate_fallback_negative_pairs(self, batch_size: int) -> Tensor:
        """Generate fallback negative pairs when sampling fails.
        
        Args:
            batch_size: Number of pairs to generate
            
        Returns:
            Tensor of negative pair indices [batch_size, 2]
        """
        # Simply generate random node pairs
        device = self.edge_index.device
        src = torch.randint(0, self.num_nodes, (batch_size,), device=device)
        dst = torch.randint(0, self.num_nodes, (batch_size,), device=device)
        
        # Avoid self-loops
        for i in range(batch_size):
            if src[i] == dst[i]:
                dst[i] = (dst[i] + 1) % self.num_nodes
        
        return torch.stack([src, dst], dim=1)
