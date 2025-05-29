"""
Extended test suite for the sampler module.

This test suite focuses on testing edge cases and rarely executed code paths
to improve test coverage.
"""

import os
import sys
import torch
import numpy as np
import pytest
from typing import Tuple, List, Dict, Optional, Set, Any
import unittest
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.isne.training.sampler import NeighborSampler, RandomWalkSampler, index2ptr


class TestDeepEdgeCases(unittest.TestCase):
    """Test extreme edge cases and rare execution paths."""
    
    def setUp(self):
        """Set up test fixtures for each test."""
        # Set a fixed random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create a specialized test graph for rare execution paths
        self.num_nodes = 50
        self.edge_index = self._create_test_graph()
        
        # Create sampler instances
        self.neighbor_sampler = NeighborSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            batch_size=16,
            num_hops=2,
            neighbor_size=5,
            seed=42,
            directed=True,  # Set directed to True to test this branch
            replace=True    # Set replace to True to test sampling with replacement
        )
        
        self.random_walk_sampler = RandomWalkSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            batch_size=16,
            walk_length=10,
            context_size=4,
            walks_per_node=3,
            p=1.5,  # Non-default values to test different paths
            q=0.75,
            seed=42,
            directed=True
        )
    
    def _create_test_graph(self) -> torch.Tensor:
        """Create a specialized test graph to target specific code paths."""
        # Create a graph with some specialized structures
        edge_list = []
        
        # Create a directed path (0->1->2->...->19)
        for i in range(19):
            edge_list.append([i, i+1])
        
        # Create a star with node 20 at center
        for i in range(21, 30):
            edge_list.append([20, i])
        
        # Create a few cycles
        for i in range(30, 39):
            edge_list.append([i, i+1])
        edge_list.append([39, 30])  # Close the cycle
        
        # Create a clique for nodes 40-49
        for i in range(40, 49):
            for j in range(i+1, 50):
                edge_list.append([i, j])
        
        # Convert to tensor
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        
        return edge_index
    
    def test_neighbor_sampler_with_replace(self):
        """Test NeighborSampler when sampling with replacement."""
        # Sample from a node with many neighbors
        nodes = torch.tensor([20])  # The star center
        neighbors = self.neighbor_sampler.sample_neighbors(nodes)
        
        # With replacement, we might get duplicates when sampling many neighbors
        # Set a large size to force the code path where len(node_neighbors) <= size
        large_size = 100
        original_size = self.neighbor_sampler.neighbor_size
        self.neighbor_sampler.neighbor_size = large_size
        
        # This should trigger the branch where all neighbors are sampled
        neighbors_all = self.neighbor_sampler.sample_neighbors(nodes)
        
        # Restore original size
        self.neighbor_sampler.neighbor_size = original_size
        
        # Check that more neighbors were sampled with the larger size
        self.assertGreaterEqual(neighbors_all.numel(), neighbors.numel())
    
    def test_neighbor_sampling_edge_cases(self):
        """Test edge cases in the NeighborSampler.sample_neighbors method."""
        # Test when some but not all nodes have neighbors
        mixed_nodes = torch.tensor([0, 999])  # One valid, one out of bounds
        neighbors = self.neighbor_sampler.sample_neighbors(mixed_nodes)
        
        # Should still return some neighbors from the valid node
        self.assertGreater(neighbors.numel(), 0)
        
        # Test the branch where no neighbors exist for any nodes
        invalid_nodes = torch.tensor([999, 1000])
        neighbors = self.neighbor_sampler.sample_neighbors(invalid_nodes)
        
        # Should return an empty tensor
        self.assertEqual(neighbors.numel(), 0)
    
    def test_sampling_with_custom_batch_size(self):
        """Test sampling methods with custom batch sizes."""
        # Test sample_nodes with custom batch size
        custom_batch_size = 25
        nodes = self.neighbor_sampler.sample_nodes(batch_size=custom_batch_size)
        self.assertEqual(nodes.shape[0], custom_batch_size)
        
        # Test sample_edges with custom batch size
        edges = self.neighbor_sampler.sample_edges(batch_size=custom_batch_size)
        self.assertEqual(edges.shape[0], custom_batch_size)
        
        # Test sample_triplets with custom batch size
        anchors, positives, negatives = self.neighbor_sampler.sample_triplets(batch_size=custom_batch_size)
        self.assertEqual(anchors.shape[0], custom_batch_size)
    
    def test_rw_sampler_rare_paths(self):
        """Test rare execution paths in RandomWalkSampler."""
        # Test sample_triplets with custom batch size
        custom_batch_size = 30
        anchors, positives, negatives = self.random_walk_sampler.sample_triplets(batch_size=custom_batch_size)
        self.assertEqual(anchors.shape[0], custom_batch_size)
        
        # Test case where there are no valid nodes with neighbors
        # Make a temporary copy with no neighbors
        edge_index_empty = torch.zeros((2, 0), dtype=torch.long)
        temp_sampler = RandomWalkSampler(
            edge_index=edge_index_empty,
            num_nodes=10,
            batch_size=12
        )
        
        # Should fall back to random pairs
        pairs = temp_sampler.sample_positive_pairs()
        self.assertEqual(pairs.shape[0], 12)
        
        # Test negative sampling with positive pairs provided
        positive_pairs = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
        neg_pairs = self.random_walk_sampler.sample_negative_pairs(positive_pairs=positive_pairs)
        self.assertEqual(neg_pairs.shape[0], self.random_walk_sampler.batch_size)
    
    def test_direct_access_to_fallback_methods(self):
        """Test direct access to fallback methods."""
        # Test neighbor sampler fallback methods
        fallback_pos = self.neighbor_sampler._generate_fallback_positive_pairs(15)
        self.assertEqual(fallback_pos.shape[0], 15)
        
        fallback_neg = self.neighbor_sampler._generate_fallback_negative_pairs(18)
        self.assertEqual(fallback_neg.shape[0], 18)
        
        # Test random walk sampler fallback methods directly
        # We won't force torch_cluster since it's causing errors
        fallback_pairs = self.random_walk_sampler._generate_fallback_positive_pairs(22)
        self.assertEqual(fallback_pairs.shape[0], 22)
        
        # Test the fallback implementation directly
        fallback_impl = self.random_walk_sampler._sample_positive_pairs_fallback(25)
        self.assertEqual(fallback_impl.shape[0], 25)
    
    def test_error_handling(self):
        """Test error handling in various methods."""
        # Create a sampler with invalid parameters to test error handling
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)  # One edge
        
        # This should handle the case where a node has no neighbors
        sampler = NeighborSampler(
            edge_index=edge_index,
            num_nodes=10,
            batch_size=5
        )
        
        # Test sampling edges with no edges available
        edges = sampler.sample_edges()
        self.assertEqual(edges.shape[0], 1)  # Should only return the one edge
        
        # Test triplet sampling - should handle no neighbors case
        anchors, positives, negatives = sampler.sample_triplets()
        self.assertEqual(anchors.shape[0], 5)
        
        # Create a sampler with a single node to test extreme case
        single_node_sampler = NeighborSampler(
            edge_index=torch.zeros((2, 0), dtype=torch.long),  # No edges
            num_nodes=1,
            batch_size=3
        )
        
        # Test negative sampling when all nodes are connected
        neg_pairs = single_node_sampler.sample_negative_pairs()
        self.assertEqual(neg_pairs.shape[0], 3)
    
    def test_sample_neighbors_with_size(self):
        """Test sample_neighbors with different size parameters."""
        # Test with explicit size parameter
        nodes = torch.tensor([20])  # Node with many neighbors
        
        # Default size
        neighbors1 = self.neighbor_sampler.sample_neighbors(nodes)
        
        # Custom size - smaller
        small_size = 2
        neighbors2 = self.neighbor_sampler.sample_neighbors(nodes, small_size)
        
        # Custom size - larger
        large_size = 20
        neighbors3 = self.neighbor_sampler.sample_neighbors(nodes, large_size)
        
        # Check that sizes are as expected
        self.assertTrue(0 < neighbors2.numel() <= small_size)
        self.assertGreaterEqual(neighbors3.numel(), neighbors2.numel())
    
    def test_sampling_approaches(self):
        """Test different sampling approaches and parameters."""
        # Test RandomWalkSampler with unusual parameters
        sampler = RandomWalkSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            batch_size=10,
            walk_length=2,  # Very short walks
            context_size=1,  # Minimum context
            walks_per_node=1  # Minimum walks
        )
        
        # Sample positive pairs
        pairs = sampler.sample_positive_pairs()
        self.assertEqual(pairs.shape[0], 10)
        
        # Test with very large batch size that exceeds available pairs
        large_batch = 1000
        large_pairs = sampler.sample_positive_pairs(batch_size=large_batch)
        self.assertEqual(large_pairs.shape[0], large_batch)


if __name__ == "__main__":
    unittest.main()
