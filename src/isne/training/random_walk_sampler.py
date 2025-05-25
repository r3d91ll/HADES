"""
Random walk based sampling implementations for ISNE training.

This module provides random walk-based sampling strategies for ISNE models,
following the original paper's implementation for effective positive and negative
pair generation.
"""

import logging
import torch
import numpy as np
import random
from typing import List, Optional, Set, Tuple, Dict, Any, Union
from torch import Tensor
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes

# Set up logging
logger = logging.getLogger(__name__)

# For CSR conversion - if torch_geometric.utils.sparse.index2ptr is not available
# we provide a fallback implementation
try:
    from torch_geometric.utils.sparse import index2ptr
except ImportError:
    # Fallback implementation of index2ptr
    def index2ptr(index: Tensor, size: int) -> Tensor:
        ptr = torch.zeros(size + 1, dtype=torch.long, device=index.device)
        torch.scatter_add_(ptr, 0, index + 1, torch.ones_like(index))
        return torch.cumsum(ptr, 0)


class RandomWalkSampler:
    """
    Random walk-based sampling strategy for efficient ISNE training.
    
    This sampler follows the original ISNE paper's implementation, using random
    walks to generate positive pairs and randomly sampling nodes for negative pairs.
    This approach ensures better connectivity and valid sampling.
    
    The sampler also supports batch-aware sampling to improve training efficiency
    by ensuring that sampled pairs stay within batch boundaries, reducing the
    filtering rate of out-of-bounds pairs during training.
    """
    
    def __init__(
        self,
        edge_index: Tensor,
        num_nodes: Optional[int] = None,
        batch_size: int = 32,
        walk_length: int = 5,
        context_size: int = 2,
        walks_per_node: int = 10,
        p: float = 1.0,  # Return parameter (like Node2Vec)
        q: float = 1.0,  # In-out parameter (like Node2Vec)
        num_negative_samples: int = 1,
        directed: bool = False,
        seed: Optional[int] = None,
        use_batch_aware_sampling: bool = False,
        device: Optional[str] = None
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
        # Ensure num_nodes is not None to prevent type errors in arithmetic operations
        self.num_nodes = maybe_num_nodes(edge_index) if num_nodes is None else num_nodes
        self.batch_size = batch_size
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples
        self.directed = directed
        self.device = device if device is not None else 'cpu'
        self.use_batch_aware_sampling = use_batch_aware_sampling
        
        # Set up CSR representation for fast random walks
        self._setup_csr_format()
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
    
    def _setup_csr_format(self) -> None:
        """Set up CSR format for efficient random walks.
        Following the original ISNE implementation, we sort the edge index and create
        a CSR representation for use in random walks.
        """
        logger = logging.getLogger(__name__)
        
        # Validate edge_index to ensure all indices are within bounds
        edge_index_cpu = self.edge_index.cpu() if self.edge_index.is_cuda else self.edge_index
        
        # Handle empty edge index case
        if edge_index_cpu.size(1) == 0:
            logger.warning("Empty edge index. Creating a small graph with self-loops.")
            # Create a minimal valid graph to prevent crashes
            num_valid_nodes = min(self.num_nodes, 10)
            src = torch.arange(0, num_valid_nodes)
            dst = torch.arange(0, num_valid_nodes)
            # Create self-loops as fallback
            edge_index_cpu = torch.stack([src, dst], dim=0)
        else:
            # Only check for out-of-bounds indices if we have edges
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
        pos_pairs = []
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
        adj_list = {}
        for i in range(self.effective_num_nodes):
            # Get neighbors for node i
            start_ptr = self.rowptr[i].item()
            end_ptr = self.rowptr[i+1].item()
            if start_ptr < end_ptr:  # Only include nodes with at least one neighbor
                neighbors = self.col[start_ptr:end_ptr].tolist()
                if neighbors:  # Double-check we have neighbors
                    adj_list[i] = neighbors
        
        # Get nodes that have neighbors
        nodes_with_neighbors = list(adj_list.keys())
        
        if not nodes_with_neighbors:
            logger.warning("No valid nodes with neighbors found. Using fallback pairs.")
            return self._generate_fallback_positive_pairs(batch_size)
            
        # Sample pairs of connected nodes
        pair_indices = []
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
            # Create consecutive pairs (simulating a path)
            # For example, if we have 100 nodes, we'll create pairs like (0,1), (1,2), etc.
            max_idx = max(1, self.num_nodes - 1)
            src_indices = torch.arange(batch_size, dtype=torch.long) % max_idx
            dst_indices = (src_indices + 1) % (max_idx + 1)
            
            # Stack to create pairs [batch_size, 2]
            pos_pairs = torch.stack([src_indices, dst_indices], dim=1)
            
            # Ensure all indices are within bounds
            valid_mask = (
                (pos_pairs[:, 0] >= 0) & 
                (pos_pairs[:, 0] < self.num_nodes) & 
                (pos_pairs[:, 1] >= 0) & 
                (pos_pairs[:, 1] < self.num_nodes)
            )
            
            if not torch.all(valid_mask):
                logger.warning(f"Filtered {(~valid_mask).sum().item()} out-of-bounds fallback pairs")
                pos_pairs = pos_pairs[valid_mask]
                
                # If we lost pairs, generate more
                if len(pos_pairs) < batch_size:
                    # Simple fallback: create pairs (i, i+1) mod num_nodes
                    missing = batch_size - len(pos_pairs)
                    fallback = torch.zeros((missing, 2), dtype=torch.long)
                    for i in range(missing):
                        fallback[i, 0] = i % self.num_nodes
                        fallback[i, 1] = (i + 1) % self.num_nodes
                    pos_pairs = torch.cat([pos_pairs, fallback], dim=0)
            
            return pos_pairs[:batch_size]
            
        except Exception as e:
            logger.error(f"Error generating fallback positive pairs: {str(e)}")
            
            # Last resort fallback: create sequential pairs
            pairs = []
            for i in range(batch_size):
                src = i % max(1, self.num_nodes - 1)
                dst = (i + 1) % self.num_nodes
                pairs.append([src, dst])
            return torch.tensor(pairs, dtype=torch.long)

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
                # Avoid self-loops
                if dst_val == src_val:
                    dst_val = (dst_val + 1) % (max_idx + 1)
                dst_indices.append(dst_val)
                
            dst_indices = torch.tensor(dst_indices, dtype=torch.long)
            
            # Stack to create pairs [batch_size, 2]
            neg_pairs = torch.stack([src_indices, dst_indices], dim=1)
            
            # Validate that all indices are within bounds (should always be true by construction)
            valid_mask = (
                (neg_pairs[:, 0] >= 0) & 
                (neg_pairs[:, 0] < self.num_nodes) & 
                (neg_pairs[:, 1] >= 0) & 
                (neg_pairs[:, 1] < self.num_nodes)
            )
            
            if not torch.all(valid_mask):
                logger.warning(f"Found {(~valid_mask).sum().item()} out-of-bounds fallback pairs, filtering...")
                valid_pairs = neg_pairs[valid_mask]
                
                # If we lost too many pairs, supplement with sequential pairs
                if len(valid_pairs) < batch_size:
                    missing = batch_size - len(valid_pairs)
                    sequential_pairs = torch.tensor([[i % max_idx, (i+1) % (max_idx+1)] 
                                                  for i in range(missing)], dtype=torch.long)
                    neg_pairs = torch.cat([valid_pairs, sequential_pairs], dim=0)
                else:
                    neg_pairs = valid_pairs
            
            return neg_pairs[:batch_size]
            
        except Exception as e:
            logger.error(f"Error generating fallback negative pairs: {str(e)}")
            # Last resort fallback: sequential pairs
            max_idx = max(1, self.num_nodes - 1)
            return torch.tensor([[i % max_idx, (i+1) % (max_idx+1)] 
                               for i in range(batch_size)], dtype=torch.long)
                               
    def sample_nodes(self, batch_size: Optional[int] = None) -> Tensor:
        """
        Sample nodes uniformly at random from the graph.
        
        This method is required for compatibility with the ISNE trainer, which needs
        to sample nodes for training even when not using pairs directly.
        
        Args:
            batch_size: Number of nodes to sample (defaults to self.batch_size)
            
        Returns:
            Tensor of sampled node indices [batch_size]
        """
        batch_size = batch_size or self.batch_size
        device = self.edge_index.device
        sampled_nodes = torch.randint(0, self.num_nodes, (batch_size,), device=device)
        return sampled_nodes
        
    def sample_positive_pairs_within_batch(self, batch_nodes: Tensor, batch_size: Optional[int] = None) -> Tensor:
        """
        Sample positive pairs using random walks, restricted to nodes within the current batch.
        
        This batch-aware sampling method ensures that both nodes in each sampled pair
        are present in the current batch, which dramatically reduces the filtering rate
        of out-of-bounds pairs during training.
        
        Args:
            batch_nodes: Tensor of node indices in the current batch [batch_size]
            batch_size: Number of pairs to sample (defaults to self.batch_size)
            
        Returns:
            Tensor of positive pair indices [num_pairs, 2] where both nodes are in batch_nodes,
            with indices mapped to the global node space.
        """
        logger = logging.getLogger(__name__)
        batch_size = batch_size or self.batch_size
        
        # Create a mapping between local indices (0 to len(batch_nodes)-1) and global node IDs
        batch_nodes_cpu = batch_nodes.cpu().tolist()
        local_to_global_map = {local_idx: global_idx for local_idx, global_idx in enumerate(batch_nodes_cpu)}
        global_to_local_map = {global_idx: local_idx for local_idx, global_idx in local_to_global_map.items()}
        
        # Create a tensor of the local indices (0 to len(batch_nodes)-1)
        local_batch_nodes = torch.arange(len(batch_nodes), dtype=torch.long, device=batch_nodes.device)
        
        if len(batch_nodes_cpu) < 2:
            logger.warning(f"Batch contains fewer than 2 nodes ({len(batch_nodes_cpu)}). Cannot create pairs within batch.")
            # Return fallback pairs within the batch (if possible)
            fallback_pairs = self._generate_fallback_positive_pairs_within_batch(batch_nodes, batch_size)
            logger.info(f"Generated {len(fallback_pairs)} fallback positive pairs in global ID space")
            return fallback_pairs
        
        try:
            # Use local indices for sampling but map back to global indices for return
            local_pos_pairs = []
            
            # Sample walks starting from local indices
            start_local_nodes = local_batch_nodes
            
            if self.has_torch_cluster and len(local_batch_nodes) > 0:
                # We need to create a local subgraph for the random walk function
                # Map the original edge_index to local indices (only involving nodes in the batch)
                local_edge_index = []
                
                # Convert row, col to local indices where both nodes are in the batch
                for i in range(self.rowptr.size(0) - 1):
                    if i not in global_to_local_map:  # Skip nodes not in batch
                        continue
                    
                    local_src = global_to_local_map[i]
                    start_ptr = self.rowptr[i].item()
                    end_ptr = self.rowptr[i+1].item()
                    
                    for j in range(start_ptr, end_ptr):
                        global_dst = self.col[j].item()
                        if global_dst in global_to_local_map:  # Only if destination is in batch
                            local_dst = global_to_local_map[global_dst]
                            local_edge_index.append([local_src, local_dst])
                
                # If no local edges, use fallback
                if not local_edge_index:
                    logger.warning("No valid edges within batch. Using fallback pairs.")
                    return self._generate_fallback_positive_pairs_within_batch(batch_nodes, batch_size)
                
                # Convert to tensor
                local_edge_index_tensor = torch.tensor(local_edge_index, dtype=torch.long).t()
                
                # Create local rowptr and col for the random walk function
                local_row, local_col = local_edge_index_tensor
                local_rowptr = index2ptr(local_row, len(batch_nodes))
                
                # Use torch_cluster with the local graph
                local_walks = self.random_walk_fn(local_rowptr, local_col, start_local_nodes, 
                                                self.walk_length, self.p, self.q)
                
                # Process each walk to extract pairs (all in local space)
                for walk in local_walks:
                    # Skip walks that didn't go anywhere
                    if torch.any(walk < 0):
                        continue
                        
                    # Extract valid nodes from the walk
                    valid_walk = walk[walk >= 0].cpu().tolist()
                    if len(valid_walk) < 2:  # Need at least 2 nodes for a pair
                        continue
                    
                    # Source node is the first node in the walk
                    local_src = valid_walk[0]
                    
                    # Context nodes are the next nodes in the context window
                    for i in range(1, min(self.context_size + 1, len(valid_walk))):
                        local_dst = valid_walk[i]
                        # Both nodes are already in local space (0 to len(batch_nodes)-1)
                        local_pos_pairs.append([local_src, local_dst])
            else:
                # Fallback to simpler implementation if torch_cluster not available
                # Create a simplified adjacency list from the CSR representation
                # but in local index space
                local_adjacency = {}
                
                for global_src, local_src in global_to_local_map.items():
                    # Get neighbors for the global node ID
                    if 0 <= global_src < len(self.rowptr) - 1:
                        start_ptr = self.rowptr[global_src].item()
                        end_ptr = self.rowptr[global_src+1].item()
                        neighbors = self.col[start_ptr:end_ptr].tolist()
                        
                        # Filter neighbors to only those in batch
                        local_neighbors = []
                        for global_dst in neighbors:
                            if global_dst in global_to_local_map:
                                local_dst = global_to_local_map[global_dst]
                                local_neighbors.append(local_dst)
                                
                        if local_neighbors:  # Only include nodes with in-batch neighbors
                            local_adjacency[local_src] = local_neighbors
                
                # Generate pairs from this batch-restricted local adjacency list
                for local_src, local_neighbors in local_adjacency.items():
                    for local_dst in local_neighbors:
                        local_pos_pairs.append([local_src, local_dst])
            
            # If no valid pairs found, use fallback
            if not local_pos_pairs:
                logger.warning("No valid positive pairs found within batch. Using fallback.")
                return self._generate_fallback_positive_pairs_within_batch(batch_nodes, batch_size)
            
            # Convert local pairs to tensor
            local_pairs_tensor = torch.tensor(local_pos_pairs, dtype=torch.long)
            
            # If we have more pairs than needed, randomly sample
            if len(local_pairs_tensor) > batch_size:
                # Randomly select pairs to return
                idx = torch.randperm(len(local_pairs_tensor))[:batch_size]
                local_pairs_tensor = local_pairs_tensor[idx]
            
            # If we don't have enough pairs, add fallback pairs
            if len(local_pairs_tensor) < batch_size:
                logger.info(f"Only found {len(local_pairs_tensor)}/{batch_size} positive pairs within batch. Adding fallback pairs.")
                # Fallback pairs are already in global space
                additional_pairs = self._generate_fallback_positive_pairs_within_batch(
                    batch_nodes, batch_size - len(local_pairs_tensor))
                
                # Convert local pairs to global before concatenation
                global_pairs = torch.zeros_like(local_pairs_tensor)
                for i in range(local_pairs_tensor.size(0)):
                    local_src, local_dst = local_pairs_tensor[i].tolist()
                    global_src = local_to_global_map[local_src]
                    global_dst = local_to_global_map[local_dst]
                    global_pairs[i, 0] = global_src
                    global_pairs[i, 1] = global_dst
                
                # Concatenate with fallback pairs (already in global space)
                pos_pairs_tensor = torch.cat([global_pairs, additional_pairs], dim=0)
            else:
                # Convert all local pairs to global pairs
                pos_pairs_tensor = torch.zeros_like(local_pairs_tensor)
                for i in range(local_pairs_tensor.size(0)):
                    local_src, local_dst = local_pairs_tensor[i].tolist()
                    global_src = local_to_global_map[local_src]
                    global_dst = local_to_global_map[local_dst]
                    pos_pairs_tensor[i, 0] = global_src
                    pos_pairs_tensor[i, 1] = global_dst
            
            logger.info(f"Returning {len(pos_pairs_tensor)} positive pairs in global ID space")
            return pos_pairs_tensor[:batch_size]
            
        except Exception as e:
            logger.error(f"Error in batch-aware positive pair sampling: {str(e)}")
            logger.warning("Using fallback positive pairs due to sampling error.")
            fallback_pairs = self._generate_fallback_positive_pairs_within_batch(batch_nodes, batch_size)
            logger.info(f"Generated {len(fallback_pairs)} fallback positive pairs in global ID space")
            return fallback_pairs
    
    def sample_negative_pairs_within_batch(self, batch_nodes: Tensor, positive_pairs: Optional[Tensor] = None, 
                                      batch_size: Optional[int] = None) -> Tensor:
        """
        Sample negative pairs restricted to nodes within the current batch.
        
        This batch-aware sampling method ensures that both nodes in each negative pair
        are present in the current batch, which dramatically reduces the filtering rate
        of out-of-bounds pairs during training.
        
        Args:
            batch_nodes: Tensor of node indices in the current batch [batch_size]
            positive_pairs: Optional tensor of positive pairs to avoid
            batch_size: Number of pairs to sample (defaults to self.batch_size)
            
        Returns:
            Tensor of negative pair indices [num_pairs, 2] where both nodes are in batch_nodes,
            with indices mapped to the global node space.
        """
        logger = logging.getLogger(__name__)
        batch_size = batch_size or self.batch_size
        
        # Create a mapping between local indices (0 to len(batch_nodes)-1) and global node IDs
        batch_nodes_cpu = batch_nodes.cpu().tolist()
        local_to_global_map = {local_idx: global_idx for local_idx, global_idx in enumerate(batch_nodes_cpu)}
        global_to_local_map = {global_idx: local_idx for local_idx, global_idx in local_to_global_map.items()}
        
        if len(batch_nodes_cpu) < 2:
            logger.warning(f"Batch contains fewer than 2 nodes ({len(batch_nodes_cpu)}). Cannot create negative pairs.")
            # Return fallback pairs (if possible)
            fallback_pairs = self._generate_fallback_negative_pairs_within_batch(batch_nodes, batch_size)
            logger.info(f"Generated {len(fallback_pairs)} fallback negative pairs in global ID space")
            return fallback_pairs
            
        try:
            # Create a cache of positive edges to avoid sampling them as negative pairs
            # Convert positive pairs to local space for checking
            local_pos_edges = set()
            if positive_pairs is not None:
                pos_pairs_cpu = positive_pairs.cpu()
                for i in range(len(pos_pairs_cpu)):
                    global_src, global_dst = pos_pairs_cpu[i].tolist()
                    # Only consider positive pairs where both nodes are in the batch
                    if global_src in global_to_local_map and global_dst in global_to_local_map:
                        local_src = global_to_local_map[global_src]
                        local_dst = global_to_local_map[global_dst]
                        local_pos_edges.add((local_src, local_dst))
            
            # Generate negative pairs by random sampling within local batch indices
            local_neg_pairs = []
            batch_size_value = batch_size
            
            # Try to generate up to 5x batch_size candidates to account for duplicates and positives
            max_attempts = batch_size_value * 5
            attempts = 0
            num_local_nodes = len(local_to_global_map)
            
            while len(local_neg_pairs) < batch_size_value and attempts < max_attempts:
                attempts += 1
                
                # Sample random source and target nodes from local indices
                if num_local_nodes > 1:
                    local_src = random.randint(0, num_local_nodes - 1)
                    
                    # Get a different node for dst
                    local_dst = random.randint(0, num_local_nodes - 1)
                    while local_dst == local_src:
                        local_dst = random.randint(0, num_local_nodes - 1)
                    
                    # Skip if this is a positive edge
                    if (local_src, local_dst) in local_pos_edges:
                        continue
                    
                    # Add the pair (in local space)
                    local_neg_pairs.append([local_src, local_dst])
                else:
                    # Not enough distinct nodes in batch
                    break
            
            # If no valid pairs were generated, use fallback
            if not local_neg_pairs:
                logger.warning("Could not generate any valid negative pairs within batch. Using fallback.")
                return self._generate_fallback_negative_pairs_within_batch(batch_nodes, batch_size)
            
            # Convert local pairs to tensor and ensure uniqueness
            local_neg_pairs_tensor = torch.tensor(local_neg_pairs, dtype=torch.long)
            
            # Remove duplicates if any
            local_neg_pairs_tensor = torch.unique(local_neg_pairs_tensor, dim=0)
            
            # If we don't have enough pairs, add fallback pairs
            if len(local_neg_pairs_tensor) < batch_size:
                logger.info(f"Only found {len(local_neg_pairs_tensor)}/{batch_size} negative pairs within batch. Adding fallback pairs.")
                # Fallback pairs are already in global space
                additional_pairs = self._generate_fallback_negative_pairs_within_batch(
                    batch_nodes, batch_size - len(local_neg_pairs_tensor))
                
                # Convert local pairs to global before concatenation
                global_pairs = torch.zeros_like(local_neg_pairs_tensor)
                for i in range(local_neg_pairs_tensor.size(0)):
                    local_src, local_dst = local_neg_pairs_tensor[i].tolist()
                    global_src = local_to_global_map[local_src]
                    global_dst = local_to_global_map[local_dst]
                    global_pairs[i, 0] = global_src
                    global_pairs[i, 1] = global_dst
                
                # Concatenate with fallback pairs (already in global space)
                neg_pairs_tensor = torch.cat([global_pairs, additional_pairs], dim=0)
            else:
                # Convert all local pairs to global pairs
                neg_pairs_tensor = torch.zeros_like(local_neg_pairs_tensor)
                for i in range(local_neg_pairs_tensor.size(0)):
                    local_src, local_dst = local_neg_pairs_tensor[i].tolist()
                    global_src = local_to_global_map[local_src]
                    global_dst = local_to_global_map[local_dst]
                    neg_pairs_tensor[i, 0] = global_src
                    neg_pairs_tensor[i, 1] = global_dst
                
                # If we have more pairs than needed, randomly sample
                if len(neg_pairs_tensor) > batch_size:
                    idx = torch.randperm(len(neg_pairs_tensor))[:batch_size]
                    neg_pairs_tensor = neg_pairs_tensor[idx]
            
            logger.info(f"Returning {len(neg_pairs_tensor)} negative pairs in global ID space")
            return neg_pairs_tensor[:batch_size]
            
        except Exception as e:
            logger.error(f"Error in batch-aware negative pair sampling: {str(e)}")
            logger.warning("Using fallback negative pairs due to sampling error.")
            fallback_pairs = self._generate_fallback_negative_pairs_within_batch(batch_nodes, batch_size)
            logger.info(f"Generated {len(fallback_pairs)} fallback negative pairs in global ID space")
            return fallback_pairs
    
    def _generate_fallback_positive_pairs_within_batch(self, batch_nodes: Tensor, batch_size: int) -> Tensor:
        """
        Generate fallback positive pairs from batch nodes when sampling fails.
        The pairs are created in global ID space, ready to be used by the contrastive loss.
        
        Args:
            batch_nodes: Tensor of node indices in the current batch
            batch_size: Number of pairs to generate
            
        Returns:
            Tensor of fallback positive pair indices within batch [num_pairs, 2]
            with indices in global node space
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Convert to CPU for manipulation
            batch_nodes_cpu = batch_nodes.cpu() if batch_nodes.is_cuda else batch_nodes
            batch_list = batch_nodes_cpu.tolist()
            
            # If we have fewer than 2 nodes, we can't create pairs
            if len(batch_list) < 2:
                logger.warning(f"Cannot create fallback pairs with only {len(batch_list)} nodes.")
                # Last resort: create self-loops if needed
                if len(batch_list) > 0:
                    node = batch_list[0]  # Use an existing global node ID
                    logger.info(f"Creating {batch_size} self-loop pairs with node {node}")
                    return torch.tensor([[node, node]] * batch_size, dtype=torch.long)
                else:
                    # If batch is empty, create dummy pairs with valid nodes
                    dummy_node = min(10, self.num_nodes - 1)  # Use a guaranteed valid node index
                    logger.info(f"Creating {batch_size} self-loop pairs with fallback node {dummy_node}")
                    return torch.tensor([[dummy_node, dummy_node]] * batch_size, dtype=torch.long)
            
            # Create consecutive pairs like a path through batch nodes
            # These are already in global ID space since batch_list contains global node IDs
            fallback_pairs = []
            for i in range(batch_size):
                idx1 = i % len(batch_list)
                idx2 = (i + 1) % len(batch_list)
                global_src = batch_list[idx1]
                global_dst = batch_list[idx2]
                fallback_pairs.append([global_src, global_dst])
            
            # Final validation to ensure all indices are within bounds
            fallback_tensor = torch.tensor(fallback_pairs, dtype=torch.long)
            valid_mask = (
                (fallback_tensor[:, 0] >= 0) &
                (fallback_tensor[:, 0] < self.num_nodes) &
                (fallback_tensor[:, 1] >= 0) &
                (fallback_tensor[:, 1] < self.num_nodes)
            )
            
            if not torch.all(valid_mask):
                logger.warning(f"Filtered {(~valid_mask).sum().item()} out-of-bounds fallback pairs")
                fallback_tensor = fallback_tensor[valid_mask]
                
                # If we've lost pairs, regenerate more
                if len(fallback_tensor) < batch_size and len(batch_list) >= 2:
                    missing = batch_size - len(fallback_tensor)
                    extra_pairs = []
                    for i in range(missing):
                        global_src = batch_list[i % len(batch_list)]
                        global_dst = batch_list[(i + 1) % len(batch_list)]
                        extra_pairs.append([global_src, global_dst])
                    extra_tensor = torch.tensor(extra_pairs, dtype=torch.long)
                    fallback_tensor = torch.cat([fallback_tensor, extra_tensor], dim=0)
                    
                # If we still don't have enough, use self-loops with a valid node ID
                if len(fallback_tensor) < batch_size:
                    valid_node = batch_list[0] if len(batch_list) > 0 else min(10, self.num_nodes - 1)
                    missing = batch_size - len(fallback_tensor)
                    logger.info(f"Adding {missing} self-loop pairs with node {valid_node}")
                    self_loops = torch.tensor([[valid_node, valid_node]] * missing, dtype=torch.long)
                    fallback_tensor = torch.cat([fallback_tensor, self_loops], dim=0)
            
            logger.info(f"Generated {len(fallback_tensor)} fallback positive pairs in global ID space")
            return fallback_tensor
            
        except Exception as e:
            logger.error(f"Error generating fallback positive pairs: {str(e)}")
            # Last resort fallback: create sequential pairs with valid nodes
            dummy_node = min(10, self.num_nodes - 1)  # Use a guaranteed valid node index
            logger.info(f"Error fallback: Creating {batch_size} self-loop pairs with node {dummy_node}")
            return torch.tensor([[dummy_node, dummy_node]] * batch_size, dtype=torch.long)
    
    def _generate_fallback_negative_pairs_within_batch(self, batch_nodes: Tensor, batch_size: int) -> Tensor:
        """
        Generate fallback negative pairs from batch nodes when sampling fails.
        The pairs are created in global ID space, ready to be used by the contrastive loss.
        
        Args:
            batch_nodes: Tensor of node indices in the current batch
            batch_size: Number of pairs to generate
            
        Returns:
            Tensor of fallback negative pair indices within batch [num_pairs, 2]
            with indices in global node space
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Convert to CPU for manipulation
            batch_nodes_cpu = batch_nodes.cpu() if batch_nodes.is_cuda else batch_nodes
            batch_list = batch_nodes_cpu.tolist()
            
            # If we have fewer than 2 nodes, we can't create proper pairs
            if len(batch_list) < 2:
                logger.warning(f"Cannot create fallback negative pairs with only {len(batch_list)} nodes.")
                # Last resort: create pairs with first node and itself
                if len(batch_list) > 0:
                    node = batch_list[0]  # Use an existing global node ID
                    logger.info(f"Creating {batch_size} self-loop negative pairs with node {node}")
                    return torch.tensor([[node, node]] * batch_size, dtype=torch.long)
                else:
                    # If batch is empty, create dummy pairs with valid nodes
                    dummy_node = min(10, self.num_nodes - 1)  # Use a guaranteed valid node index
                    logger.info(f"Creating {batch_size} self-loop negative pairs with fallback node {dummy_node}")
                    return torch.tensor([[dummy_node, dummy_node]] * batch_size, dtype=torch.long)
            
            # Create pairs that are not consecutive in the batch list (unlike positive fallbacks)
            # These are already in global ID space since batch_list contains global node IDs
            fallback_pairs = []
            for i in range(batch_size):
                idx1 = i % len(batch_list)
                # Skip 2 positions for negative pairs to avoid consecutive nodes
                idx2 = (i + 2) % len(batch_list)
                global_src = batch_list[idx1]
                global_dst = batch_list[idx2]
                fallback_pairs.append([global_src, global_dst])
            
            # Final validation to ensure all indices are within bounds
            fallback_tensor = torch.tensor(fallback_pairs, dtype=torch.long)
            valid_mask = (
                (fallback_tensor[:, 0] >= 0) &
                (fallback_tensor[:, 0] < self.num_nodes) &
                (fallback_tensor[:, 1] >= 0) &
                (fallback_tensor[:, 1] < self.num_nodes)
            )
            
            if not torch.all(valid_mask):
                logger.warning(f"Filtered {(~valid_mask).sum().item()} out-of-bounds fallback negative pairs")
                fallback_tensor = fallback_tensor[valid_mask]
                
                # If we've lost pairs, regenerate more
                if len(fallback_tensor) < batch_size and len(batch_list) >= 2:
                    missing = batch_size - len(fallback_tensor)
                    extra_pairs = []
                    for i in range(missing):
                        global_src = batch_list[i % len(batch_list)]
                        global_dst = batch_list[(i + 2) % len(batch_list)]  # Skip 2 for negative pairs
                        extra_pairs.append([global_src, global_dst])
                    extra_tensor = torch.tensor(extra_pairs, dtype=torch.long)
                    fallback_tensor = torch.cat([fallback_tensor, extra_tensor], dim=0)
                    
                # If we still don't have enough, use self-loops with a valid node ID
                if len(fallback_tensor) < batch_size:
                    valid_node = batch_list[0] if len(batch_list) > 0 else min(10, self.num_nodes - 1)
                    missing = batch_size - len(fallback_tensor)
                    logger.info(f"Adding {missing} self-loop negative pairs with node {valid_node}")
                    self_loops = torch.tensor([[valid_node, valid_node]] * missing, dtype=torch.long)
                    fallback_tensor = torch.cat([fallback_tensor, self_loops], dim=0)
            
            logger.info(f"Generated {len(fallback_tensor)} fallback negative pairs in global ID space")
            return fallback_tensor
            
        except Exception as e:
            logger.error(f"Error generating fallback negative pairs: {str(e)}")
            # Last resort fallback: create valid pairs
            dummy_node = min(10, self.num_nodes - 1)  # Use a guaranteed valid node index
            logger.info(f"Error fallback: Creating {batch_size} self-loop negative pairs with node {dummy_node}")
            return torch.tensor([[dummy_node, dummy_node]] * batch_size, dtype=torch.long)

    def sample_subgraph(self, nodes: Tensor) -> Tuple[Tensor, Tensor]:
        """Sample a subgraph around the given ``nodes``.

        The ISNE ``Trainer`` expects every sampler to expose a
        ``sample_subgraph`` method returning ``(subset_nodes, edge_index)``.
        This implementation builds a *one-hop* neighborhood for the provided
        seed nodes using the CSR representation prepared in
        :py:meth:`_setup_csr_format`.

        Args:
            nodes: Seed node indices (tensor on any device)

        Returns:
            Tuple ``(subset_nodes, subgraph_edge_index)`` where
            ``subset_nodes`` is a 1-D tensor of unique node IDs included in the
            subgraph and ``subgraph_edge_index`` is the corresponding edge
            index (shape ``[2, num_edges]``) with nodes **relabelled** to the
            ``0..len(subset_nodes)-1`` range as required by PyG.
        """
        # Lazy import to avoid hard dependency during doc generation
        try:
            from torch_geometric.utils import subgraph
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "PyTorch Geometric is required for subgraph sampling. Install with: pip install torch-geometric"
            ) from e

        if nodes.numel() == 0:
            return nodes, torch.empty((2, 0), dtype=torch.long, device=nodes.device)

        # Ensure indices are in bounds and unique
        nodes = nodes[nodes < self.num_nodes].unique()
        if nodes.numel() == 0:
            return nodes, torch.empty((2, 0), dtype=torch.long, device=nodes.device)

        # Collect one-hop neighbors via CSR (rowptr/col on *CPU*)
        # Move to CPU for easier slicing if necessary
        rowptr_cpu = self.rowptr.cpu()
        col_cpu = self.col.cpu()
        neighbor_ids: Set[int] = set(nodes.cpu().tolist())
        for n in neighbor_ids.copy():
            start = rowptr_cpu[n].item()
            end = rowptr_cpu[n + 1].item()
            neighbor_ids.update(col_cpu[start:end].tolist())

        subset_nodes = torch.tensor(sorted(neighbor_ids), dtype=torch.long, device=self.edge_index.device)

        subgraph_edge_index, _ = subgraph(subset_nodes, self.edge_index, relabel_nodes=True, num_nodes=self.num_nodes)
        return subset_nodes, subgraph_edge_index
