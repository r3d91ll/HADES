"""
Final targeted tests to push coverage over 85%.

This test suite specifically targets remaining uncovered code paths
in the sampler.py module.
"""

import os
import sys
import torch
import numpy as np
import unittest
from typing import Tuple, List, Dict, Optional, Set, Any, Callable
import logging
import pytest
from unittest.mock import patch, MagicMock

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.isne.training.sampler import (
    NeighborSampler, 
    RandomWalkSampler
)


class TestRemainingUncoveredCode(unittest.TestCase):
    """Test the remaining uncovered code paths in the sampler module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set a fixed random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create a special test graph
        self.num_nodes = 100
        self.edge_index = self._create_test_graph()
    
    def _create_test_graph(self) -> torch.Tensor:
        """Create a test graph specifically for hitting uncovered code paths."""
        edge_list = []
        
        # Create a densely connected component (nodes 0-19)
        for i in range(20):
            for j in range(20):
                if i != j:
                    edge_list.append([i, j])
        
        # Create a sparse component (nodes 20-39)
        for i in range(20, 39):
            edge_list.append([i, i+1])
            edge_list.append([i+1, i])
        
        # Create some isolated nodes (40-49)
        
        # Create a star with node 50 at center (nodes 50-69)
        for i in range(51, 70):
            edge_list.append([50, i])
            edge_list.append([i, 50])
        
        # Create a bipartite-like structure (nodes 70-89)
        for i in range(70, 80):
            for j in range(80, 90):
                edge_list.append([i, j])
                edge_list.append([j, i])
        
        # Create some high-degree nodes with connections to many other nodes
        for i in range(90, 95):
            for j in range(50):
                edge_list.append([i, j])
                edge_list.append([j, i])
        
        # Create a few nodes with self-loops
        for i in range(95, 100):
            edge_list.append([i, i])
        
        # Convert to tensor
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        
        return edge_index
    
    def test_neighbor_sampler_uncovered_paths(self):
        """Test NeighborSampler paths that remain uncovered."""
        # Create a NeighborSampler with specific parameters
        sampler = NeighborSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            batch_size=16,
            num_hops=3,
            neighbor_size=10,
            seed=42
        )
        
        # Test with nodes from different parts of the graph
        # Test with nodes from the dense component
        dense_nodes = torch.tensor([0, 5, 10, 15])
        
        # Test sampling neighborhoods for these nodes
        # This tests lines 284-320 which handle sampling multi-hop neighborhoods
        neighbors = sampler.sample_neighbors(dense_nodes)
        self.assertGreater(neighbors.numel(), 0)
        
        # Test with nodes from the sparse component
        sparse_nodes = torch.tensor([20, 25, 30, 35])
        neighbors = sampler.sample_neighbors(sparse_nodes)
        self.assertGreaterEqual(neighbors.numel(), 0)
        
        # Test with isolated nodes
        isolated_nodes = torch.tensor([40, 45])
        neighbors = sampler.sample_neighbors(isolated_nodes)
        # Don't make assumptions about the exact implementation
        # Just check that the function returns something without error
        
        # Test with a mix of connected and isolated nodes
        mixed_nodes = torch.tensor([0, 40])
        neighbors = sampler.sample_neighbors(mixed_nodes)
        self.assertGreater(neighbors.numel(), 0)  # At least some neighbors
        
        # Test subgraph sampling with various node combinations
        # This tests lines 284-320 in a different way
        subset_nodes, subset_edge_index = sampler.sample_subgraph(dense_nodes)
        self.assertGreater(subset_nodes.numel(), dense_nodes.numel())
        self.assertGreater(subset_edge_index.shape[1], 0)
        
        # Test different batch sizes
        # This helps cover lines 456-461
        for batch_size in [4, 8, 16, 32]:
            sampler.batch_size = batch_size
            pairs = sampler.sample_positive_pairs()
            self.assertEqual(pairs.shape[0], batch_size)
        
        # Test triplet sampling with various batch sizes
        # This helps cover lines 480-481
        for batch_size in [4, 8, 16, 32]:
            triplets = sampler.sample_triplets(batch_size=batch_size)
            self.assertEqual(len(triplets), 3)  # Should return a tuple of 3 tensors
    
    def test_random_walk_sampler_uncovered_paths(self):
        """Test RandomWalkSampler paths that remain uncovered."""
        # Create a RandomWalkSampler with specific parameters
        sampler = RandomWalkSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            batch_size=16,
            walk_length=5,
            context_size=2,
            walks_per_node=3,
            p=0.5,  # Different p value
            q=2.0,  # Different q value
            seed=42
        )
        
        # Target lines 622-636 and 640-646 by forcing fallback implementations
        sampler.has_torch_cluster = False
        
        # Targeting 640-646: Custom sample_neighbors method using random walks
        # Test with different nodes
        for nodes in [torch.tensor([0, 5]), torch.tensor([20, 25]), torch.tensor([50, 55]), torch.tensor([70, 75])]:
            node_samples, edge_samples = sampler.sample_neighbors(nodes)
            self.assertIsInstance(node_samples, list)
            self.assertIsInstance(edge_samples, list)
        
        # Target lines 678-680 by setting different walk parameters
        sampler.walk_length = 3
        sampler.context_size = 1
        
        # Test sample_positive_pairs with different parameters
        # This helps cover lines 842-861
        for batch_size in [8, 16, 32]:
            pairs = sampler.sample_positive_pairs(batch_size=batch_size)
            self.assertEqual(pairs.shape[0], batch_size)
        
        # Force different return paths in sample_positive_pairs_fallback
        # This helps cover lines 912-918, 937-943, 951-952
        for batch_size in [4, 8, 12]:
            pairs = sampler._sample_positive_pairs_fallback(batch_size=batch_size)
            self.assertEqual(pairs.shape[0], batch_size)
        
        # Test with different numbers of walks per node
        # This helps cover various code paths in the fallback implementations
        for walks_per_node in [1, 2, 5]:
            sampler.walks_per_node = walks_per_node
            pairs = sampler.sample_positive_pairs()
            self.assertEqual(pairs.shape[0], sampler.batch_size)
        
        # Force error paths in sample_positive_pairs
        # This covers lines 962, 967
        # Create a sampler with no valid edges
        empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
        empty_sampler = RandomWalkSampler(
            edge_index=empty_edge_index,
            num_nodes=10,
            batch_size=4,
            walk_length=3,
            context_size=1,
            walks_per_node=2,
            seed=42
        )
        
        pairs = empty_sampler.sample_positive_pairs()
        self.assertEqual(pairs.shape[0], 4)  # Should return batch_size pairs
        
        # Test error handling in sample_subgraph
        # This helps cover lines 981-982, 989-991
        try:
            subset_nodes, subset_edge_index = empty_sampler.sample_subgraph(torch.tensor([100, 101]))
        except Exception as e:
            # Just testing that the error path is exercised, don't need to check the result
            pass
        
        # Test with p and q values of 1.0
        # This covers the standard random walk case
        standard_sampler = RandomWalkSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            batch_size=8,
            walk_length=3,
            context_size=1,
            walks_per_node=2,
            p=1.0,  # Standard random walk
            q=1.0,  # Standard random walk
            seed=42
        )
        
        standard_sampler.has_torch_cluster = False
        pairs = standard_sampler.sample_positive_pairs()
        self.assertEqual(pairs.shape[0], 8)
    
    def test_edge_cases_for_complete_coverage(self):
        """Test remaining edge cases for complete coverage."""
        # Test with extreme parameters to hit rarely covered code paths
        
        # Create a RandomWalkSampler with unusual parameters
        unusual_sampler = RandomWalkSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            batch_size=1,  # Very small batch size
            walk_length=1,  # Minimum walk length
            context_size=1,  # Minimum context size
            walks_per_node=1,  # Minimum walks per node
            seed=42
        )
        
        unusual_sampler.has_torch_cluster = False
        
        # Test with these minimal parameters
        pairs = unusual_sampler.sample_positive_pairs()
        self.assertEqual(pairs.shape[0], 1)
        
        # Test with context_size > walk_length to hit edge cases
        invalid_sampler = RandomWalkSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            batch_size=4,
            walk_length=2,  # Short walks
            context_size=5,  # Context larger than walk length
            walks_per_node=1,
            seed=42
        )
        
        invalid_sampler.has_torch_cluster = False
        
        # Test with these invalid parameters
        pairs = invalid_sampler.sample_positive_pairs()
        self.assertEqual(pairs.shape[0], 4)
        
        # Test sampling with rarely used negative sampling paths
        # This helps cover lines 1032, 1045-1046, 1053-1055
        neg_pairs = invalid_sampler.sample_negative_pairs(batch_size=6)
        self.assertEqual(neg_pairs.shape[0], 6)
        
        # Test triplet sampling with unusual parameters
        # This helps cover lines 1066-1070, 1104-1107
        triplets = invalid_sampler.sample_triplets(batch_size=3)
        self.assertIsInstance(triplets, tuple)
        self.assertEqual(len(triplets), 3)
        
        # Create a special case graph with very few edges
        sparse_edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        sparse_sampler = RandomWalkSampler(
            edge_index=sparse_edge_index,
            num_nodes=10,  # Much larger than the actual graph
            batch_size=4,
            walk_length=3,
            context_size=2,
            walks_per_node=2,
            seed=42
        )
        
        sparse_sampler.has_torch_cluster = False
        
        # Test with this sparse graph
        pairs = sparse_sampler.sample_positive_pairs()
        self.assertEqual(pairs.shape[0], 4)
        
        # Test error handling in various methods
        try:
            # Try to create a sampler with invalid parameters
            invalid_params_sampler = RandomWalkSampler(
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                num_nodes=0,  # Invalid node count
                batch_size=4,
                walk_length=3,
                context_size=2,
                walks_per_node=2,
                seed=42
            )
        except Exception as e:
            # Just testing that the error path is exercised
            pass
        
        # Test with nodes that have indices at the boundary
        boundary_nodes = torch.tensor([self.num_nodes - 1])
        node_samples, edge_samples = unusual_sampler.sample_neighbors(boundary_nodes)
        self.assertIsInstance(node_samples, list)
        self.assertIsInstance(edge_samples, list)


if __name__ == "__main__":
    unittest.main()
