"""
Simple test script for the NeighborSampler and RandomWalkSampler classes.

This standalone script tests basic functionality without project dependencies.
"""

import os
import sys
import torch
import logging
import unittest
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

    def test_adjacency_list_creation(self):
        """Test that the adjacency list is correctly created."""
        # Check that the adjacency list has the right size
        self.assertEqual(len(self.sampler.adj_list), self.num_nodes)
        
        # Check that the adjacency list contains the correct edges
        for i in range(self.edge_index.size(1)):
            src = self.edge_index[0, i].item()
            dst = self.edge_index[1, i].item()
            
            # Check if this edge is in the adjacency list
            self.assertIn(dst, self.sampler.adj_list[src])
    
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
        
        # Check basic properties
        self.assertEqual(pos_pairs.shape[0], self.sampler.batch_size)
        self.assertEqual(pos_pairs.shape[1], 2)
        
        # All indices should be within bounds
        self.assertTrue(torch.all(pos_pairs >= 0))
        self.assertTrue(torch.all(pos_pairs < self.num_nodes))


def run_tests():
    """Run all tests."""
    print("Testing NeighborSampler...")
    unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestNeighborSampler))
    
    print("\nTesting RandomWalkSampler...")
    unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestRandomWalkSampler))


if __name__ == "__main__":
    run_tests()
