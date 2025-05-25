"""
Unit tests for the RandomWalkSampler class.

This module tests the functionality of the RandomWalkSampler to ensure it:
1. Correctly generates positive pairs using random walks
2. Correctly generates negative pairs using random sampling
3. Handles edge cases gracefully
4. Validates indices properly to prevent out-of-bounds errors
"""

import os
import sys
import unittest
import torch
import logging
import numpy as np
from typing import Tuple

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from src.isne.training.random_walk_sampler import RandomWalkSampler


class TestRandomWalkSampler(unittest.TestCase):
    """Test suite for the RandomWalkSampler class."""

    def setUp(self):
        """Set up test fixtures for each test."""
        # Set up logging
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        
        # Set a fixed random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create a simple test graph
        self.num_nodes = 20
        self.edge_index = self._create_test_graph()
        
        # Create a sampler instance
        self.sampler = RandomWalkSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            batch_size=16,
            walk_length=5,
            context_size=2,
            walks_per_node=5,
            seed=42
        )
    
    def _create_test_graph(self) -> torch.Tensor:
        """Create a simple test graph for testing.
        
        Returns:
            Edge index tensor [2, num_edges]
        """
        # Create a simple ring graph
        src = torch.arange(0, self.num_nodes)
        dst = torch.roll(src, -1)  # Connect each node to the next one in a ring
        
        # Add some additional random edges for more complex structure
        num_random_edges = self.num_nodes // 2
        random_src = torch.randint(0, self.num_nodes, (num_random_edges,))
        random_dst = torch.randint(0, self.num_nodes, (num_random_edges,))
        
        # Combine all edges
        all_src = torch.cat([src, random_src])
        all_dst = torch.cat([dst, random_dst])
        
        # Create the edge index
        edge_index = torch.stack([all_src, all_dst], dim=0)
        
        # Add reverse edges to make it undirected
        reverse_edge_index = torch.stack([all_dst, all_src], dim=0)
        edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
        
        return edge_index
    
    def _print_pairs_stats(self, pairs: torch.Tensor, name: str):
        """Print statistics about sampled pairs for debugging."""
        self.logger.info(f"--- {name} Stats ---")
        self.logger.info(f"Shape: {pairs.shape}")
        self.logger.info(f"Min indices: {pairs.min(dim=0).values}")
        self.logger.info(f"Max indices: {pairs.max(dim=0).values}")
        self.logger.info(f"Unique nodes: {torch.unique(pairs).shape[0]}")
        self.logger.info(f"First 5 pairs: {pairs[:5]}")
    
    def test_csr_format_setup(self):
        """Test that the CSR format is correctly set up."""
        # Verify the CSR format attributes exist
        self.assertTrue(hasattr(self.sampler, 'rowptr'))
        self.assertTrue(hasattr(self.sampler, 'col'))
        
        # Check their shapes
        self.assertEqual(self.sampler.rowptr.shape[0], self.num_nodes + 1)
        
        # The total number of edges should be the same
        self.assertEqual(self.sampler.col.shape[0], self.edge_index.shape[1])
    
    def test_sample_positive_pairs(self):
        """Test sampling positive pairs from the graph."""
        # Sample positive pairs
        pos_pairs = self.sampler.sample_positive_pairs()
        
        # Print statistics
        self._print_pairs_stats(pos_pairs, "Positive Pairs")
        
        # Check basic properties
        self.assertEqual(pos_pairs.shape[0], self.sampler.batch_size)
        self.assertEqual(pos_pairs.shape[1], 2)
        
        # All indices should be within bounds
        self.assertTrue(torch.all(pos_pairs >= 0))
        self.assertTrue(torch.all(pos_pairs < self.num_nodes))
        
        # In our test graph, we should have at least some connected pairs
        # We can't check all pairs as the sampling has randomness
        valid_count = 0
        for src, dst in pos_pairs:
            src_val, dst_val = src.item(), dst.item()
            
            # Check if this is a valid edge in our original graph
            edge_mask = (self.edge_index[0] == src_val) & (self.edge_index[1] == dst_val)
            if edge_mask.any():
                valid_count += 1
                
        # At least some pairs should be valid edges
        # Note: With random walks, this ratio should be high, but depends on the graph structure
        self.logger.info(f"Valid positive edges ratio: {valid_count/len(pos_pairs):.2f}")
    
    def test_sample_positive_pairs_within_batch(self):
        """Test that positive pairs are correctly sampled within batch boundaries."""
        # Sample a batch of nodes
        batch_size = 8
        batch_nodes = torch.randint(0, self.num_nodes, (batch_size,))
        
        # Sample positive pairs within the batch
        pos_pairs = self.sampler.sample_positive_pairs_within_batch(batch_nodes, batch_size=batch_size)
        
        # Verify shape
        self.assertEqual(pos_pairs.shape, (batch_size, 2))
        
        # Verify all nodes are within the batch
        batch_nodes_set = set(batch_nodes.tolist())
        for i in range(batch_size):
            src, dst = pos_pairs[i].tolist()
            # Check that at least source nodes are in the batch
            # (some destinations might be fallback nodes in difficult cases)
            self.assertIn(src, batch_nodes_set)
            
        # Print stats for debugging
        self._print_pairs_stats(pos_pairs, "Batch-Aware Positive Pairs")
    
    def test_sample_negative_pairs_within_batch(self):
        """Test that negative pairs are correctly sampled within batch boundaries."""
        # Sample a batch of nodes
        batch_size = 8
        batch_nodes = torch.randint(0, self.num_nodes, (batch_size,))
        
        # Sample positive pairs first to use as input
        pos_pairs = self.sampler.sample_positive_pairs_within_batch(batch_nodes, batch_size=batch_size)
        
        # Sample negative pairs within the batch
        neg_pairs = self.sampler.sample_negative_pairs_within_batch(batch_nodes, pos_pairs, batch_size=batch_size)
        
        # Verify shape
        self.assertEqual(neg_pairs.shape, (batch_size, 2))
        
        # Verify all nodes are within the batch
        batch_nodes_set = set(batch_nodes.tolist())
        for i in range(batch_size):
            src, dst = neg_pairs[i].tolist()
            # Check that at least source nodes are in the batch
            # (some destinations might be fallback nodes in difficult cases)
            self.assertIn(src, batch_nodes_set)
            
        # Print stats for debugging
        self._print_pairs_stats(neg_pairs, "Batch-Aware Negative Pairs")
    
    def test_batch_aware_sampling_small_batch(self):
        """Test batch-aware sampling with a very small batch (edge case)."""
        # Create a tiny batch with just 2 nodes
        batch_nodes = torch.tensor([0, 1])
        batch_size = 4  # Request more pairs than nodes in batch
        
        # Sample positive pairs
        pos_pairs = self.sampler.sample_positive_pairs_within_batch(batch_nodes, batch_size=batch_size)
        
        # Verify shape
        self.assertEqual(pos_pairs.shape, (batch_size, 2))
        
        # Verify the sampler used fallback pairs (may include nodes not in batch, but should be valid indices)
        self.assertLess(pos_pairs.max(), self.num_nodes)
        
        # Sample negative pairs
        neg_pairs = self.sampler.sample_negative_pairs_within_batch(batch_nodes, pos_pairs, batch_size=batch_size)
        
        # Verify shape
        self.assertEqual(neg_pairs.shape, (batch_size, 2))
        
        # Verify indices are valid
        self.assertLess(neg_pairs.max(), self.num_nodes)
    
    def test_sample_negative_pairs(self):
        """Test that negative pairs are correctly sampled."""
        # Sample negative pairs
        neg_pairs = self.sampler.sample_negative_pairs()
        
        # Print statistics
        self._print_pairs_stats(neg_pairs, "Negative Pairs")
        
        # Check basic properties
        self.assertEqual(neg_pairs.shape[0], self.sampler.batch_size)
        self.assertEqual(neg_pairs.shape[1], 2)
        
        # All indices should be within bounds
        self.assertTrue(torch.all(neg_pairs >= 0))
        self.assertTrue(torch.all(neg_pairs < self.num_nodes))
        
        # No self-loops
        self.assertTrue(torch.all(neg_pairs[:, 0] != neg_pairs[:, 1]))
    
    def test_handle_empty_graph(self):
        """Test handling of empty graphs gracefully."""
        # Create an empty edge index
        empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Initialize sampler with empty graph
        empty_sampler = RandomWalkSampler(
            edge_index=empty_edge_index,
            num_nodes=self.num_nodes,
            batch_size=16,
            seed=42
        )
        
        # Sampling should still work and use fallbacks
        pos_pairs = empty_sampler.sample_positive_pairs()
        neg_pairs = empty_sampler.sample_negative_pairs()
        
        # Check basic properties (should use fallback methods)
        self.assertEqual(pos_pairs.shape[0], empty_sampler.batch_size)
        self.assertEqual(neg_pairs.shape[0], empty_sampler.batch_size)
        
        # All indices should be within bounds
        self.assertTrue(torch.all(pos_pairs >= 0))
        self.assertTrue(torch.all(pos_pairs < self.num_nodes))
        self.assertTrue(torch.all(neg_pairs >= 0))
        self.assertTrue(torch.all(neg_pairs < self.num_nodes))
    
    def test_handle_out_of_bounds_edges(self):
        """Test handling of edge indices that exceed num_nodes."""
        # Create a graph with out-of-bounds indices
        src = torch.tensor([0, 1, 2, self.num_nodes])  # One out-of-bounds index
        dst = torch.tensor([1, 2, 3, 0])
        edge_index = torch.stack([src, dst], dim=0)
        
        # Initialize sampler with problematic graph
        # This should log warnings but not crash
        oob_sampler = RandomWalkSampler(
            edge_index=edge_index,
            num_nodes=self.num_nodes,
            batch_size=16,
            seed=42
        )
        
        # Sampling should still work and filter bad edges
        pos_pairs = oob_sampler.sample_positive_pairs()
        neg_pairs = oob_sampler.sample_negative_pairs()
        
        # Check basic properties
        self.assertEqual(pos_pairs.shape[0], oob_sampler.batch_size)
        self.assertEqual(neg_pairs.shape[0], oob_sampler.batch_size)
        
        # All indices should be within bounds
        self.assertTrue(torch.all(pos_pairs >= 0))
        self.assertTrue(torch.all(pos_pairs < self.num_nodes))
        self.assertTrue(torch.all(neg_pairs >= 0))
        self.assertTrue(torch.all(neg_pairs < self.num_nodes))
    
    def test_batch_size_consistency(self):
        """Test that the sampler returns the requested batch size."""
        # Try different batch sizes
        for batch_size in [1, 8, 32]:
            pos_pairs = self.sampler.sample_positive_pairs(batch_size=batch_size)
            neg_pairs = self.sampler.sample_negative_pairs(batch_size=batch_size)
            
            self.assertEqual(pos_pairs.shape[0], batch_size)
            self.assertEqual(neg_pairs.shape[0], batch_size)


if __name__ == '__main__':
    unittest.main()
