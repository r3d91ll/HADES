"""
Comprehensive test script for the NeighborSampler and RandomWalkSampler classes.

This standalone script tests all major functionality without project dependencies.
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
        
        # Create a directed sampler instance
        self.directed_sampler = NeighborSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            batch_size=16,
            num_hops=2,
            neighbor_size=5,
            directed=True,
            seed=42
        )
        
        # Create a sampler with replacement
        self.replacement_sampler = NeighborSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            batch_size=16,
            num_hops=2,
            neighbor_size=5,
            replace=True,
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

    def test_initialization(self):
        """Test initialization with different parameters."""
        # Basic initialization
        sampler = NeighborSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes
        )
        self.assertEqual(sampler.batch_size, 32)  # Default batch size
        self.assertEqual(sampler.num_hops, 1)  # Default num_hops
        
        # Edge case: empty edge_index
        empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
        empty_sampler = NeighborSampler(
            edge_index=empty_edge_index,
            num_nodes=self.num_nodes
        )
        self.assertEqual(len(empty_sampler.adj_list), self.num_nodes)
        
        # With different parameters
        sampler = NeighborSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            batch_size=64,
            num_hops=3,
            neighbor_size=10,
            directed=True,
            replace=True
        )
        self.assertEqual(sampler.batch_size, 64)
        self.assertEqual(sampler.num_hops, 3)
        self.assertEqual(sampler.neighbor_size, 10)
        self.assertTrue(sampler.directed)
        self.assertTrue(sampler.replace)

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
        
        # Check directed vs undirected
        undirected_count = sum(len(adj) for adj in self.sampler.adj_list)
        directed_count = sum(len(adj) for adj in self.directed_sampler.adj_list)
        
        # In directed mode, we only keep one direction, so there should be fewer edges
        self.assertGreaterEqual(undirected_count, directed_count)
    
    def test_sample_nodes(self):
        """Test sampling nodes from the graph."""
        # Sample nodes
        nodes = self.sampler.sample_nodes()
        
        # Check basic properties
        self.assertEqual(nodes.shape[0], self.sampler.batch_size)
        
        # All indices should be within bounds
        self.assertTrue(torch.all(nodes >= 0))
        self.assertTrue(torch.all(nodes < self.num_nodes))
        
        # With specific batch size
        nodes = self.sampler.sample_nodes(batch_size=10)
        self.assertEqual(nodes.shape[0], 10)
        
        # Large batch size (larger than num_nodes)
        nodes = self.sampler.sample_nodes(batch_size=30)
        self.assertEqual(nodes.shape[0], 30)
    
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
        
        # With specific batch size
        edges = self.sampler.sample_edges(batch_size=10)
        self.assertEqual(edges.shape[0], 10)
        
        # Empty graph case
        empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
        empty_sampler = NeighborSampler(
            edge_index=empty_edge_index,
            num_nodes=self.num_nodes
        )
        edges = empty_sampler.sample_edges()
        self.assertEqual(edges.shape[0], 0)  # Should return empty tensor
    
    def test_sample_neighbors(self):
        """Test sampling neighbors for nodes."""
        # Sample some seed nodes
        seed_nodes = torch.tensor([0, 5, 10, 15])
        
        # Sample neighbors
        neighbors = self.sampler.sample_neighbors(seed_nodes)
        
        # Check neighbors are valid
        if neighbors.numel() > 0:
            self.assertTrue(torch.all(neighbors >= 0))
            self.assertTrue(torch.all(neighbors < self.num_nodes))
        
        # With replacement, should sample exactly neighbor_size per node
        neighbors_with_replace = self.replacement_sampler.sample_neighbors(seed_nodes)
        if seed_nodes.numel() > 0 and any(len(self.replacement_sampler.adj_list[node.item()]) > 0 for node in seed_nodes):
            self.assertGreater(neighbors_with_replace.numel(), 0)
            
        # Empty adjacency list case
        empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
        empty_sampler = NeighborSampler(
            edge_index=empty_edge_index,
            num_nodes=self.num_nodes
        )
        neighbors = empty_sampler.sample_neighbors(seed_nodes)
        self.assertEqual(neighbors.numel(), 0)  # Should return empty tensor
    
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
        
        # Empty graph case
        empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
        empty_sampler = NeighborSampler(
            edge_index=empty_edge_index,
            num_nodes=self.num_nodes
        )
        subset_nodes, subgraph_edge_index = empty_sampler.sample_subgraph(seed_nodes)
        self.assertEqual(subset_nodes.numel(), seed_nodes.numel())  # Should just have seed nodes
        self.assertEqual(subgraph_edge_index.numel(), 0)  # No edges
    
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
        
        # With specific batch size
        pos_pairs = self.sampler.sample_positive_pairs(batch_size=10)
        self.assertEqual(pos_pairs.shape[0], 10)
        
        # Empty graph case - should use fallback method
        empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
        empty_sampler = NeighborSampler(
            edge_index=empty_edge_index,
            num_nodes=self.num_nodes
        )
        pos_pairs = empty_sampler.sample_positive_pairs()
        self.assertEqual(pos_pairs.shape[0], empty_sampler.batch_size)
        self.assertEqual(pos_pairs.shape[1], 2)
    
    def test_sample_negative_pairs(self):
        """Test sampling negative pairs from the graph."""
        # Sample positive pairs first
        pos_pairs = self.sampler.sample_positive_pairs()
        
        # Sample negative pairs
        neg_pairs = self.sampler.sample_negative_pairs(positive_pairs=pos_pairs)
        
        # Check basic properties
        self.assertEqual(neg_pairs.shape[0], self.sampler.batch_size)
        self.assertEqual(neg_pairs.shape[1], 2)
        
        # All indices should be within bounds
        self.assertTrue(torch.all(neg_pairs >= 0))
        self.assertTrue(torch.all(neg_pairs < self.num_nodes))
        
        # Check for self-loops
        for i in range(neg_pairs.shape[0]):
            src = neg_pairs[i, 0].item()
            dst = neg_pairs[i, 1].item()
            self.assertNotEqual(src, dst)
        
        # With specific batch size
        neg_pairs = self.sampler.sample_negative_pairs(batch_size=10)
        self.assertEqual(neg_pairs.shape[0], 10)
        
        # Empty graph case
        empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
        empty_sampler = NeighborSampler(
            edge_index=empty_edge_index,
            num_nodes=self.num_nodes
        )
        neg_pairs = empty_sampler.sample_negative_pairs()
        self.assertEqual(neg_pairs.shape[0], empty_sampler.batch_size)
        self.assertEqual(neg_pairs.shape[1], 2)
    
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
        
        # With specific batch size
        anchors, positives, negatives = self.sampler.sample_triplets(batch_size=10)
        self.assertEqual(anchors.shape[0], 10)
        self.assertEqual(positives.shape[0], 10)
        self.assertEqual(negatives.shape[0], 10)
        
        # Empty graph case
        empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
        empty_sampler = NeighborSampler(
            edge_index=empty_edge_index,
            num_nodes=self.num_nodes
        )
        anchors, positives, negatives = empty_sampler.sample_triplets()
        self.assertEqual(anchors.shape[0], empty_sampler.batch_size)
        self.assertEqual(positives.shape[0], empty_sampler.batch_size)
        self.assertEqual(negatives.shape[0], empty_sampler.batch_size)


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
        
        # Create a directed sampler
        self.directed_sampler = RandomWalkSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            batch_size=16,
            walk_length=5,
            context_size=2,
            walks_per_node=5,
            directed=True,
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

    def test_initialization(self):
        """Test initialization with different parameters."""
        # Basic initialization
        sampler = RandomWalkSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes
        )
        self.assertEqual(sampler.batch_size, 32)  # Default batch size
        self.assertEqual(sampler.walk_length, 5)  # Default walk_length
        
        # With different parameters
        sampler = RandomWalkSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            batch_size=64,
            walk_length=10,
            context_size=3,
            walks_per_node=10,
            p=2.0,
            q=0.5,
            directed=True
        )
        self.assertEqual(sampler.batch_size, 64)
        self.assertEqual(sampler.walk_length, 10)
        self.assertEqual(sampler.context_size, 3)
        self.assertEqual(sampler.walks_per_node, 10)
        self.assertEqual(sampler.p, 2.0)
        self.assertEqual(sampler.q, 0.5)
        self.assertTrue(sampler.directed)
    
    def test_csr_format_setup(self):
        """Test that the CSR format is correctly set up."""
        # Verify the CSR format attributes exist
        self.assertTrue(hasattr(self.sampler, 'rowptr'))
        self.assertTrue(hasattr(self.sampler, 'col'))
        
        # Check their shapes
        self.assertEqual(self.sampler.rowptr.shape[0], self.num_nodes + 1)
        
        # The total number of edges should be the same
        self.assertEqual(self.sampler.col.shape[0], self.edge_index.shape[1])
        
        # Directed vs undirected
        directed_edges = self.directed_sampler.col.shape[0]
        undirected_edges = self.sampler.col.shape[0]
        # In directed mode, we use the original edge index, which should have half the edges
        self.assertLessEqual(directed_edges, undirected_edges)
    
    def test_sample_nodes(self):
        """Test sampling nodes from the graph."""
        # Sample nodes
        nodes = self.sampler.sample_nodes()
        
        # Check basic properties
        self.assertEqual(nodes.shape[0], self.sampler.batch_size)
        
        # All indices should be within bounds
        self.assertTrue(torch.all(nodes >= 0))
        self.assertTrue(torch.all(nodes < self.num_nodes))
        
        # With specific batch size
        nodes = self.sampler.sample_nodes(batch_size=10)
        self.assertEqual(nodes.shape[0], 10)
        
        # Large batch size (larger than num_nodes)
        nodes = self.sampler.sample_nodes(batch_size=30)
        self.assertEqual(nodes.shape[0], 30)
    
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
        
        # With specific batch size
        pos_pairs = self.sampler.sample_positive_pairs(batch_size=10)
        self.assertEqual(pos_pairs.shape[0], 10)
        
        # Empty graph case - should use fallback method
        empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
        empty_sampler = RandomWalkSampler(
            edge_index=empty_edge_index,
            num_nodes=self.num_nodes
        )
        pos_pairs = empty_sampler.sample_positive_pairs()
        self.assertEqual(pos_pairs.shape[0], empty_sampler.batch_size)
        self.assertEqual(pos_pairs.shape[1], 2)
    
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
        
        # Check for self-loops
        for i in range(neg_pairs.shape[0]):
            src = neg_pairs[i, 0].item()
            dst = neg_pairs[i, 1].item()
            self.assertNotEqual(src, dst)
        
        # With specific batch size
        neg_pairs = self.sampler.sample_negative_pairs(batch_size=10)
        self.assertEqual(neg_pairs.shape[0], 10)
        
        # Sample with positive pairs input
        pos_pairs = self.sampler.sample_positive_pairs()
        neg_pairs = self.sampler.sample_negative_pairs(positive_pairs=pos_pairs)
        self.assertEqual(neg_pairs.shape[0], pos_pairs.shape[0])
        
        # Empty graph case
        empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
        empty_sampler = RandomWalkSampler(
            edge_index=empty_edge_index,
            num_nodes=self.num_nodes
        )
        neg_pairs = empty_sampler.sample_negative_pairs()
        self.assertEqual(neg_pairs.shape[0], empty_sampler.batch_size)
        self.assertEqual(neg_pairs.shape[1], 2)
    
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
        
        # With specific batch size
        anchors, positives, negatives = self.sampler.sample_triplets(batch_size=10)
        self.assertEqual(anchors.shape[0], 10)
        self.assertEqual(positives.shape[0], 10)
        self.assertEqual(negatives.shape[0], 10)
        
        # Empty graph case
        empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
        empty_sampler = RandomWalkSampler(
            edge_index=empty_edge_index,
            num_nodes=self.num_nodes
        )
        anchors, positives, negatives = empty_sampler.sample_triplets()
        self.assertEqual(anchors.shape[0], empty_sampler.batch_size)
        self.assertEqual(positives.shape[0], empty_sampler.batch_size)
        self.assertEqual(negatives.shape[0], empty_sampler.batch_size)
    
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
        
        # Empty graph case
        empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
        empty_sampler = RandomWalkSampler(
            edge_index=empty_edge_index,
            num_nodes=self.num_nodes
        )
        subset_nodes, subgraph_edge_index = empty_sampler.sample_subgraph(seed_nodes)
        self.assertEqual(subset_nodes.numel(), seed_nodes.numel())  # Should just have seed nodes
        self.assertEqual(subgraph_edge_index.numel(), 0)  # No edges


def run_tests():
    """Run all tests."""
    print("Testing NeighborSampler...")
    unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestNeighborSampler))
    
    print("\nTesting RandomWalkSampler...")
    unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestRandomWalkSampler))


if __name__ == "__main__":
    run_tests()
