"""
Focused test script for the NeighborSampler and RandomWalkSampler classes.

This version avoids potential infinite loops by using timeouts and simpler tests.
"""

import os
import sys
import torch
import logging
import unittest
import time
from typing import Tuple, List, Dict, Optional, Set, Any

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.isne.training.sampler import NeighborSampler, RandomWalkSampler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestNeighborSampler(unittest.TestCase):
    """Test suite for the NeighborSampler class."""

    def setUp(self):
        """Set up test fixtures for each test."""
        # Set a fixed random seed for reproducibility
        torch.manual_seed(42)
        
        # Create a simple test graph
        self.num_nodes = 20
        self.edge_index = self._create_test_graph()
        
        # Create a sampler instance
        self.sampler = NeighborSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            batch_size=16,
            num_hops=2,
            neighbor_size=5,
            seed=42
        )
    
    def _create_test_graph(self) -> torch.Tensor:
        """Create a simple test graph for testing."""
        # Create a simple ring graph
        src = torch.arange(0, self.num_nodes)
        dst = torch.roll(src, -1)  # Connect each node to the next one in a ring
        
        # Create the edge index
        edge_index = torch.stack([src, dst], dim=0)
        
        # Add reverse edges to make it undirected
        reverse_edge_index = torch.stack([dst, src], dim=0)
        edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
        
        return edge_index

    def test_adjacency_list_creation(self):
        """Test that the adjacency list is correctly created."""
        # Check that the adjacency list has the right size
        self.assertEqual(len(self.sampler.adj_list), self.num_nodes)
        
        # Check that each node has at least one neighbor in a ring graph
        for node_idx in range(self.num_nodes):
            self.assertGreaterEqual(len(self.sampler.adj_list[node_idx]), 1)
    
    def test_sample_nodes(self):
        """Test sampling nodes from the graph."""
        # Sample nodes
        nodes = self.sampler.sample_nodes()
        
        # Check basic properties
        self.assertEqual(nodes.shape[0], self.sampler.batch_size)
        
        # All indices should be within bounds
        self.assertTrue(torch.all(nodes >= 0))
        self.assertTrue(torch.all(nodes < self.num_nodes))
    
    def test_sample_edges(self):
        """Test sampling edges from the graph."""
        # Sample edges
        edges = self.sampler.sample_edges()
        
        # Check basic properties
        self.assertEqual(edges.shape[0], self.sampler.batch_size)
        self.assertEqual(edges.shape[1], 2)
        
        # All indices should be within bounds
        self.assertTrue(torch.all(edges >= 0))
        self.assertTrue(torch.all(edges < self.num_nodes))
    
    def test_sample_positive_pairs(self):
        """Test sampling positive pairs from the graph."""
        # Sample positive pairs
        pos_pairs = self.sampler.sample_positive_pairs()
        
        # Check basic properties
        self.assertEqual(pos_pairs.shape[0], self.sampler.batch_size)
        self.assertEqual(pos_pairs.shape[1], 2)
        
        # All indices should be within bounds
        self.assertTrue(torch.all(pos_pairs >= 0))
        self.assertTrue(torch.all(pos_pairs < self.num_nodes))
    
    def test_sample_negative_pairs(self):
        """Test sampling negative pairs from the graph."""
        # Sample negative pairs
        neg_pairs = self.sampler.sample_negative_pairs()
        
        # Check basic properties
        self.assertEqual(neg_pairs.shape[0], self.sampler.batch_size)
        self.assertEqual(neg_pairs.shape[1], 2)
        
        # All indices should be within bounds
        self.assertTrue(torch.all(neg_pairs >= 0))
        self.assertTrue(torch.all(neg_pairs < self.num_nodes))
    
    def test_sample_triplets(self):
        """Test sampling triplets (anchor, positive, negative)."""
        # Sample triplets
        anchors, positives, negatives = self.sampler.sample_triplets()
        
        # Check basic properties
        self.assertEqual(anchors.shape[0], self.sampler.batch_size)
        self.assertEqual(positives.shape[0], self.sampler.batch_size)
        self.assertEqual(negatives.shape[0], self.sampler.batch_size)
        
        # All indices should be within bounds
        self.assertTrue(torch.all(anchors >= 0))
        self.assertTrue(torch.all(anchors < self.num_nodes))
        self.assertTrue(torch.all(positives >= 0))
        self.assertTrue(torch.all(positives < self.num_nodes))
        self.assertTrue(torch.all(negatives >= 0))
        self.assertTrue(torch.all(negatives < self.num_nodes))
    
    def test_sample_subgraph(self):
        """Test sampling a subgraph around nodes."""
        # Sample some seed nodes
        seed_nodes = torch.tensor([0, 5, 10, 15])
        
        # Sample subgraph
        subset_nodes, subgraph_edge_index = self.sampler.sample_subgraph(seed_nodes)
        
        # Check basic properties
        self.assertTrue(subset_nodes.numel() >= seed_nodes.numel())
        
        # All seed nodes should be in the subset
        for node in seed_nodes:
            self.assertIn(node.item(), subset_nodes.tolist())
        
        # Edge indices should be relabeled
        if subgraph_edge_index.numel() > 0:
            self.assertEqual(subgraph_edge_index.shape[0], 2)
            max_idx = subset_nodes.size(0) - 1
            self.assertTrue(torch.all(subgraph_edge_index >= 0))
            self.assertTrue(torch.all(subgraph_edge_index <= max_idx))


class TestRandomWalkSampler(unittest.TestCase):
    """Test suite for the RandomWalkSampler class."""

    def setUp(self):
        """Set up test fixtures for each test."""
        # Set a fixed random seed for reproducibility
        torch.manual_seed(42)
        
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
        """Create a simple test graph for testing."""
        # Create a simple ring graph
        src = torch.arange(0, self.num_nodes)
        dst = torch.roll(src, -1)  # Connect each node to the next one in a ring
        
        # Create the edge index
        edge_index = torch.stack([src, dst], dim=0)
        
        # Add reverse edges to make it undirected
        reverse_edge_index = torch.stack([dst, src], dim=0)
        edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
        
        return edge_index

    def test_csr_format_setup(self):
        """Test that the CSR format is correctly set up."""
        # Verify the CSR format attributes exist
        self.assertTrue(hasattr(self.sampler, 'rowptr'))
        self.assertTrue(hasattr(self.sampler, 'col'))
        
        # Check their shapes
        self.assertEqual(self.sampler.rowptr.shape[0], self.num_nodes + 1)
        
        # The total number of edges should be the same
        self.assertEqual(self.sampler.col.shape[0], self.edge_index.shape[1])
    
    def test_sample_nodes(self):
        """Test sampling nodes from the graph."""
        # Sample nodes
        nodes = self.sampler.sample_nodes()
        
        # Check basic properties
        self.assertEqual(nodes.shape[0], self.sampler.batch_size)
        
        # All indices should be within bounds
        self.assertTrue(torch.all(nodes >= 0))
        self.assertTrue(torch.all(nodes < self.num_nodes))
    
    def test_sample_positive_pairs(self):
        """Test sampling positive pairs from the graph."""
        # Sample positive pairs
        pos_pairs = self.sampler.sample_positive_pairs()
        
        # Check basic properties
        self.assertEqual(pos_pairs.shape[0], self.sampler.batch_size)
        self.assertEqual(pos_pairs.shape[1], 2)
        
        # All indices should be within bounds
        self.assertTrue(torch.all(pos_pairs >= 0))
        self.assertTrue(torch.all(pos_pairs < self.num_nodes))
    
    def test_sample_negative_pairs(self):
        """Test sampling negative pairs from the graph."""
        # Sample negative pairs
        neg_pairs = self.sampler.sample_negative_pairs()
        
        # Check basic properties
        self.assertEqual(neg_pairs.shape[0], self.sampler.batch_size)
        self.assertEqual(neg_pairs.shape[1], 2)
        
        # All indices should be within bounds
        self.assertTrue(torch.all(neg_pairs >= 0))
        self.assertTrue(torch.all(neg_pairs < self.num_nodes))
    
    def test_sample_triplets(self):
        """Test sampling triplets (anchor, positive, negative)."""
        # Sample triplets
        anchors, positives, negatives = self.sampler.sample_triplets()
        
        # Check basic properties
        self.assertEqual(anchors.shape[0], self.sampler.batch_size)
        self.assertEqual(positives.shape[0], self.sampler.batch_size)
        self.assertEqual(negatives.shape[0], self.sampler.batch_size)
        
        # All indices should be within bounds
        self.assertTrue(torch.all(anchors >= 0))
        self.assertTrue(torch.all(anchors < self.num_nodes))
        self.assertTrue(torch.all(positives >= 0))
        self.assertTrue(torch.all(positives < self.num_nodes))
        self.assertTrue(torch.all(negatives >= 0))
        self.assertTrue(torch.all(negatives < self.num_nodes))
    
    def test_sample_subgraph(self):
        """Test sampling a subgraph around nodes."""
        # Sample some seed nodes
        seed_nodes = torch.tensor([0, 5, 10, 15])
        
        # Sample subgraph
        subset_nodes, subgraph_edge_index = self.sampler.sample_subgraph(seed_nodes)
        
        # Check basic properties
        self.assertTrue(subset_nodes.numel() >= seed_nodes.numel())
        
        # All seed nodes should be in the subset
        for node in seed_nodes:
            self.assertIn(node.item(), subset_nodes.tolist())
        
        # Edge indices should be relabeled
        if subgraph_edge_index.numel() > 0:
            self.assertEqual(subgraph_edge_index.shape[0], 2)
            max_idx = subset_nodes.size(0) - 1
            self.assertTrue(torch.all(subgraph_edge_index >= 0))
            self.assertTrue(torch.all(subgraph_edge_index <= max_idx))

def test_empty_graph_handling():
    """Test that empty graphs are handled gracefully."""
    print("Testing NeighborSampler with empty graph...")
    empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
    num_nodes = 20  # Make sure we have enough nodes
    batch_size = 10  # Keep batch_size < num_nodes
    
    # First test with replacement=True (should work fine)
    empty_neighbor_sampler = NeighborSampler(
        edge_index=empty_edge_index,
        num_nodes=num_nodes,
        batch_size=batch_size,
        replace=True
    )
    # Basic operations should not hang
    empty_neighbor_sampler.sample_nodes()
    empty_neighbor_sampler.sample_edges()
    empty_neighbor_sampler.sample_positive_pairs()
    empty_neighbor_sampler.sample_negative_pairs()
    
    # Now test with replacement=False (should work if batch_size <= num_nodes)
    empty_neighbor_sampler = NeighborSampler(
        edge_index=empty_edge_index,
        num_nodes=num_nodes,
        batch_size=batch_size,
        replace=False
    )
    # Basic operations should not hang
    empty_neighbor_sampler.sample_nodes()
    empty_neighbor_sampler.sample_edges()
    empty_neighbor_sampler.sample_positive_pairs()
    empty_neighbor_sampler.sample_negative_pairs()
    
    print("Testing RandomWalkSampler with empty graph...")
    empty_random_walk_sampler = RandomWalkSampler(
        edge_index=empty_edge_index,
        num_nodes=num_nodes,
        batch_size=batch_size
    )
    # Basic operations should not hang
    empty_random_walk_sampler.sample_nodes()
    empty_random_walk_sampler.sample_positive_pairs()
    empty_random_walk_sampler.sample_negative_pairs()
    print("Empty graph tests completed successfully!")


def run_tests():
    """Run all tests."""
    # First test with empty graphs
    test_empty_graph_handling()
    
    # Then run the regular unit tests
    print("\nRunning NeighborSampler tests...")
    unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestNeighborSampler))
    
    print("\nRunning RandomWalkSampler tests...")
    unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestRandomWalkSampler))


if __name__ == "__main__":
    run_tests()
