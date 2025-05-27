"""
Comprehensive test suite for the NeighborSampler and RandomWalkSampler classes.

This test suite aims to achieve > 85% code coverage for the sampler implementations
by testing all major functionality and edge cases.
"""

import os
import sys
import torch
import numpy as np
import unittest
import logging
from typing import Tuple, List, Dict, Optional, Set, Any

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.isne.training.sampler import NeighborSampler, RandomWalkSampler


class TestNeighborSamplerBasic(unittest.TestCase):
    """Basic test suite for the NeighborSampler class."""

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

    def test_initialization(self):
        """Test that the sampler is correctly initialized."""
        # Check basic attributes
        self.assertEqual(self.sampler.num_nodes, self.num_nodes)
        self.assertEqual(self.sampler.batch_size, 16)
        self.assertEqual(self.sampler.num_hops, 2)
        self.assertEqual(self.sampler.neighbor_size, 5)
        
        # Check that the adjacency list is correctly created
        self.assertEqual(len(self.sampler.adj_list), self.num_nodes)
        
        # Every node should have at least two neighbors in a ring graph (previous and next)
        for neighbors in self.sampler.adj_list:
            self.assertGreaterEqual(len(neighbors), 2)
        
        # Check that nodes_with_neighbors is properly populated
        self.assertEqual(len(self.sampler.nodes_with_neighbors), self.num_nodes)

    def test_sample_nodes(self):
        """Test sampling nodes from the graph."""
        # Sample nodes
        nodes = self.sampler.sample_nodes()
        
        # Check basic properties
        self.assertEqual(nodes.shape[0], self.sampler.batch_size)
        
        # All indices should be within bounds
        self.assertTrue(torch.all(nodes >= 0))
        self.assertTrue(torch.all(nodes < self.num_nodes))
        
        # Test with custom batch size
        custom_batch_size = 10
        nodes = self.sampler.sample_nodes(batch_size=custom_batch_size)
        self.assertEqual(nodes.shape[0], custom_batch_size)

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
        
        # Test with custom batch size
        custom_batch_size = 10
        edges = self.sampler.sample_edges(batch_size=custom_batch_size)
        self.assertEqual(edges.shape[0], custom_batch_size)

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
        
        # Test with custom batch size
        custom_batch_size = 10
        pos_pairs = self.sampler.sample_positive_pairs(batch_size=custom_batch_size)
        self.assertEqual(pos_pairs.shape[0], custom_batch_size)

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
        
        # Test with custom batch size
        custom_batch_size = 10
        neg_pairs = self.sampler.sample_negative_pairs(batch_size=custom_batch_size)
        self.assertEqual(neg_pairs.shape[0], custom_batch_size)

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
        
        # Test with custom batch size
        custom_batch_size = 10
        anchors, positives, negatives = self.sampler.sample_triplets(batch_size=custom_batch_size)
        self.assertEqual(anchors.shape[0], custom_batch_size)
        self.assertEqual(positives.shape[0], custom_batch_size)
        self.assertEqual(negatives.shape[0], custom_batch_size)

    def test_sample_neighbors(self):
        """Test sampling neighbors for nodes."""
        # Sample some seed nodes
        seed_nodes = torch.tensor([0, 5, 10, 15])
        
        # Sample neighbors
        neighbors = self.sampler.sample_neighbors(seed_nodes)
        
        # Check that the result is a tensor
        self.assertTrue(isinstance(neighbors, torch.Tensor))
        
        # The number of neighbors should be at least one per seed node
        # (assuming our test graph is well-formed)
        self.assertTrue(neighbors.numel() > 0)

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
        self.assertEqual(subgraph_edge_index.shape[0], 2)
        max_idx = subset_nodes.size(0) - 1
        self.assertTrue(torch.all(subgraph_edge_index >= 0))
        self.assertTrue(torch.all(subgraph_edge_index <= max_idx))


class TestNeighborSamplerEdgeCases(unittest.TestCase):
    """Edge case tests for the NeighborSampler class."""

    def test_empty_graph(self):
        """Test sampler with an empty graph (no edges)."""
        # Create an empty edge index
        empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
        num_nodes = 20
        
        # Create a sampler with empty edge index
        sampler = NeighborSampler(
            edge_index=empty_edge_index,
            num_nodes=num_nodes,
            batch_size=10
        )
        
        # Basic operations should still work
        nodes = sampler.sample_nodes()
        self.assertEqual(nodes.shape[0], 10)
        
        edges = sampler.sample_edges()
        self.assertEqual(edges.shape[0], 0)  # Should return empty tensor
        
        pos_pairs = sampler.sample_positive_pairs()
        self.assertEqual(pos_pairs.shape[0], 10)  # Should use fallback
        
        neg_pairs = sampler.sample_negative_pairs()
        self.assertEqual(neg_pairs.shape[0], 10)

    def test_single_node_graph(self):
        """Test sampler with a graph containing only one node."""
        # Create a self-loop edge for a single node
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        num_nodes = 1
        
        # Create a sampler
        sampler = NeighborSampler(
            edge_index=edge_index,
            num_nodes=num_nodes,
            batch_size=5,
            replace=True  # Need replacement for batch_size > num_nodes
        )
        
        # Basic operations should still work
        nodes = sampler.sample_nodes()
        self.assertEqual(nodes.shape[0], 5)
        self.assertTrue(torch.all(nodes == 0))  # All nodes must be 0
        
        # Sample edges (should sample the self-loop)
        edges = sampler.sample_edges()
        self.assertEqual(edges.shape[0], 1)
        self.assertEqual(edges[0, 0], 0)
        self.assertEqual(edges[0, 1], 0)
        
        # Test positive pairs
        pos_pairs = sampler.sample_positive_pairs()
        self.assertEqual(pos_pairs.shape[0], 5)
        
        # Test negative pairs (impossible with only one node)
        neg_pairs = sampler.sample_negative_pairs()
        self.assertEqual(neg_pairs.shape[0], 5)

    def test_disconnected_graph(self):
        """Test sampler with a disconnected graph."""
        # Create two separate components
        num_nodes = 10
        
        # First component: nodes 0-4 in a ring
        src1 = torch.arange(0, 5)
        dst1 = torch.roll(src1, -1)
        edge_index1 = torch.stack([src1, dst1], dim=0)
        
        # Second component: nodes 5-9 in a ring
        src2 = torch.arange(5, 10)
        dst2 = torch.roll(src2, -1)
        edge_index2 = torch.stack([src2, dst2], dim=0)
        
        # Combine components
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)
        
        # Create a sampler
        sampler = NeighborSampler(
            edge_index=edge_index,
            num_nodes=num_nodes,
            batch_size=8
        )
        
        # Sample neighbors for nodes in different components
        seed_nodes = torch.tensor([0, 5])
        neighbors = sampler.sample_neighbors(seed_nodes)
        
        # Make sure we got some neighbors
        self.assertTrue(neighbors.numel() > 0)
        
        # For this small test case, create a version where we do separate calls
        # and check the results don't mix between components
        comp1_neighbors = sampler.sample_neighbors(torch.tensor([0]))
        comp2_neighbors = sampler.sample_neighbors(torch.tensor([5]))
        
        # Verify that neighbors don't cross components
        if comp1_neighbors.numel() > 0:
            self.assertTrue(torch.all(comp1_neighbors < 5))
        if comp2_neighbors.numel() > 0:
            self.assertTrue(torch.all(comp2_neighbors >= 5))


class TestRandomWalkSamplerBasic(unittest.TestCase):
    """Basic test suite for the RandomWalkSampler class."""

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

    def test_initialization(self):
        """Test that the sampler is correctly initialized."""
        # Check basic attributes
        self.assertEqual(self.sampler.num_nodes, self.num_nodes)
        self.assertEqual(self.sampler.batch_size, 16)
        self.assertEqual(self.sampler.walk_length, 5)
        self.assertEqual(self.sampler.context_size, 2)
        self.assertEqual(self.sampler.walks_per_node, 5)
        
        # Check that CSR format is correctly created
        self.assertEqual(self.sampler.rowptr.shape[0], self.num_nodes + 1)
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
        
        # Test with custom batch size
        custom_batch_size = 10
        nodes = self.sampler.sample_nodes(batch_size=custom_batch_size)
        self.assertEqual(nodes.shape[0], custom_batch_size)

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
        
        # Test with custom batch size
        custom_batch_size = 10
        pos_pairs = self.sampler.sample_positive_pairs(batch_size=custom_batch_size)
        self.assertEqual(pos_pairs.shape[0], custom_batch_size)

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
        
        # Test with custom batch size
        custom_batch_size = 10
        neg_pairs = self.sampler.sample_negative_pairs(batch_size=custom_batch_size)
        self.assertEqual(neg_pairs.shape[0], custom_batch_size)

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
        
        # Test with custom batch size
        custom_batch_size = 10
        anchors, positives, negatives = self.sampler.sample_triplets(batch_size=custom_batch_size)
        self.assertEqual(anchors.shape[0], custom_batch_size)
        self.assertEqual(positives.shape[0], custom_batch_size)
        self.assertEqual(negatives.shape[0], custom_batch_size)

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

    def test_fallback_methods(self):
        """Test the fallback methods when torch_cluster is not available."""
        # Force the fallback methods by setting has_torch_cluster to False
        self.sampler.has_torch_cluster = False
        
        # Sample positive pairs using fallback
        pos_pairs = self.sampler.sample_positive_pairs()
        
        # Check basic properties
        self.assertEqual(pos_pairs.shape[0], self.sampler.batch_size)
        self.assertEqual(pos_pairs.shape[1], 2)
        
        # Generate fallback positive pairs directly
        fallback_pairs = self.sampler._generate_fallback_positive_pairs(self.sampler.batch_size)
        
        # Check basic properties
        self.assertEqual(fallback_pairs.shape[0], self.sampler.batch_size)
        self.assertEqual(fallback_pairs.shape[1], 2)


class TestRandomWalkSamplerEdgeCases(unittest.TestCase):
    """Edge case tests for the RandomWalkSampler class."""

    def test_empty_graph(self):
        """Test sampler with an empty graph (no edges)."""
        # Create an empty edge index
        empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
        num_nodes = 20
        
        # Create a sampler with empty edge index
        sampler = RandomWalkSampler(
            edge_index=empty_edge_index,
            num_nodes=num_nodes,
            batch_size=10
        )
        
        # Basic operations should still work
        nodes = sampler.sample_nodes()
        self.assertEqual(nodes.shape[0], 10)
        
        # Check CSR format for empty graph
        self.assertEqual(sampler.rowptr.shape[0], num_nodes + 1)
        self.assertEqual(sampler.col.shape[0], 0)
        
        # Sample positive pairs (should use fallback)
        pos_pairs = sampler.sample_positive_pairs()
        self.assertEqual(pos_pairs.shape[0], 10)
        
        # Sample negative pairs
        neg_pairs = sampler.sample_negative_pairs()
        self.assertEqual(neg_pairs.shape[0], 10)

    def test_single_node_graph(self):
        """Test sampler with a graph containing only one node."""
        # Create a self-loop edge for a single node
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        num_nodes = 1
        
        # Create a sampler
        sampler = RandomWalkSampler(
            edge_index=edge_index,
            num_nodes=num_nodes,
            batch_size=5,
            walk_length=2
        )
        
        # Basic operations should still work
        nodes = sampler.sample_nodes()
        self.assertEqual(nodes.shape[0], 5)
        self.assertTrue(torch.all(nodes == 0))  # All nodes must be 0
        
        # Check CSR format
        self.assertEqual(sampler.rowptr.shape[0], num_nodes + 1)
        self.assertEqual(sampler.rowptr[0], 0)
        self.assertEqual(sampler.rowptr[1], 1)
        self.assertEqual(sampler.col.shape[0], 1)
        self.assertEqual(sampler.col[0], 0)
        
        # Test positive pairs
        pos_pairs = sampler.sample_positive_pairs()
        self.assertEqual(pos_pairs.shape[0], 5)
        
        # Test negative pairs (impossible with only one node)
        neg_pairs = sampler.sample_negative_pairs()
        self.assertEqual(neg_pairs.shape[0], 5)

    def test_disconnected_graph(self):
        """Test sampler with a disconnected graph."""
        # Create two separate components
        num_nodes = 10
        
        # First component: nodes 0-4 in a ring
        src1 = torch.arange(0, 5)
        dst1 = torch.roll(src1, -1)
        edge_index1 = torch.stack([src1, dst1], dim=0)
        
        # Second component: nodes 5-9 in a ring
        src2 = torch.arange(5, 10)
        dst2 = torch.roll(src2, -1)
        edge_index2 = torch.stack([src2, dst2], dim=0)
        
        # Combine components
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)
        
        # Create a sampler
        sampler = RandomWalkSampler(
            edge_index=edge_index,
            num_nodes=num_nodes,
            batch_size=8
        )
        
        # Check basic operations
        nodes = sampler.sample_nodes()
        self.assertEqual(nodes.shape[0], 8)
        
        pos_pairs = sampler.sample_positive_pairs()
        self.assertEqual(pos_pairs.shape[0], 8)
        
        neg_pairs = sampler.sample_negative_pairs()
        self.assertEqual(neg_pairs.shape[0], 8)


def run_all_tests():
    """Run all tests."""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestNeighborSamplerBasic))
    suite.addTest(unittest.makeSuite(TestNeighborSamplerEdgeCases))
    suite.addTest(unittest.makeSuite(TestRandomWalkSamplerBasic))
    suite.addTest(unittest.makeSuite(TestRandomWalkSamplerEdgeCases))
    
    # Run tests
    result = unittest.TextTestRunner().run(suite)
    return result


class TestUtilityFunctions(unittest.TestCase):
    """Test suite for utility functions in the sampler module."""
    
    def test_index2ptr(self):
        """Test the index2ptr utility function."""
        from src.isne.training.sampler import index2ptr
        
        # Create a simple edge index
        index = torch.tensor([0, 1, 1, 2, 3, 3, 3, 4])
        size = 6  # Number of nodes (0-5)
        
        # Convert to CSR pointer format
        ptr = index2ptr(index, size)
        
        # Check output shape and type
        self.assertEqual(ptr.shape[0], size + 1)
        self.assertEqual(ptr.dtype, torch.long)
        
        # Check values - ptr should have size+1 elements, with ptr[i] indicating
        # the position in index where elements with value i begin
        expected_ptr = torch.tensor([0, 1, 3, 4, 7, 8, 8], dtype=torch.long)
        self.assertTrue(torch.all(ptr == expected_ptr))


class TestAdvancedSamplerFeatures(unittest.TestCase):
    """Test advanced features of samplers."""
    
    def setUp(self):
        """Set up test fixtures for each test."""
        # Set a fixed random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create a complex test graph
        self.num_nodes = 30
        self.edge_index = self._create_test_graph()
        
        # Create sampler instances
        self.neighbor_sampler = NeighborSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            batch_size=16,
            num_hops=2,
            neighbor_size=5,
            seed=42
        )
        
        self.random_walk_sampler = RandomWalkSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            batch_size=16,
            walk_length=5,
            context_size=2,
            walks_per_node=5,
            seed=42
        )
    
    def _create_test_graph(self) -> torch.Tensor:
        """Create a more complex test graph for testing."""
        # Create a graph with a chain of cliques
        edge_list = []
        
        # Create 3 cliques (fully connected subgraphs) of size 10 each
        for i in range(3):
            base = i * 10
            # Create a clique
            for j in range(10):
                for k in range(j+1, 10):
                    edge_list.append([base + j, base + k])  # Add edge
                    edge_list.append([base + k, base + j])  # Add reverse edge
            
            # Connect to the next clique with a bridge edge if not the last clique
            if i < 2:
                edge_list.append([base + 9, base + 10])  # Forward edge
                edge_list.append([base + 10, base + 9])  # Backward edge
        
        # Convert to tensor
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        
        return edge_index
    
    def test_neighbor_sampler_edge_cases(self):
        """Test edge cases in the NeighborSampler."""
        # Test with an empty nodes tensor
        empty_nodes = torch.tensor([], dtype=torch.long)
        neighbors = self.neighbor_sampler.sample_neighbors(empty_nodes)
        self.assertEqual(neighbors.numel(), 0)
        
        # Test with out-of-bounds node indices
        invalid_nodes = torch.tensor([self.num_nodes + 5, self.num_nodes + 10])
        neighbors = self.neighbor_sampler.sample_neighbors(invalid_nodes)
        self.assertEqual(neighbors.numel(), 0)
        
        # Test sample_subgraph with PyTorch Geometric integration
        # Force the code path that uses PyTorch Geometric
        original_value = getattr(self.neighbor_sampler, 'TORCH_GEOMETRIC_AVAILABLE', None)
        setattr(self.neighbor_sampler, 'TORCH_GEOMETRIC_AVAILABLE', True)
        
        try:
            # Check that the fallback implementation is used
            nodes = torch.tensor([0, 10, 20])
            subset_nodes, subset_edge_index = self.neighbor_sampler.sample_subgraph(nodes)
            
            # Validate the results
            self.assertTrue(subset_nodes.numel() >= len(nodes))
            self.assertEqual(subset_edge_index.shape[0], 2)  # Should be [2, num_edges]
            
            # Check that all seed nodes are in the subset
            for node in nodes:
                self.assertIn(node.item(), subset_nodes.tolist())
        finally:
            # Restore the original value
            if original_value is not None:
                setattr(self.neighbor_sampler, 'TORCH_GEOMETRIC_AVAILABLE', original_value)
            else:
                delattr(self.neighbor_sampler, 'TORCH_GEOMETRIC_AVAILABLE')
    
    def test_neighbor_sampler_fallback(self):
        """Test the fallback methods in NeighborSampler."""
        # Test the fallback positive pairs method
        pairs = self.neighbor_sampler._generate_fallback_positive_pairs(20)
        self.assertEqual(pairs.shape[0], 20)
        self.assertEqual(pairs.shape[1], 2)
        
        # Test the fallback negative pairs method
        neg_pairs = self.neighbor_sampler._generate_fallback_negative_pairs(25)
        self.assertEqual(neg_pairs.shape[0], 25)
        self.assertEqual(neg_pairs.shape[1], 2)
    
    def test_random_walk_sampler_no_torch_cluster(self):
        """Test RandomWalkSampler when torch_cluster is not available."""
        # Force the code path where torch_cluster is not available
        original_value = getattr(self.random_walk_sampler, 'has_torch_cluster', None)
        self.random_walk_sampler.has_torch_cluster = False
        
        try:
            # Test that the fallback implementation is used
            pairs = self.random_walk_sampler.sample_positive_pairs()
            self.assertEqual(pairs.shape[0], self.random_walk_sampler.batch_size)
            self.assertEqual(pairs.shape[1], 2)
            
            # All indices should be within bounds
            self.assertTrue(torch.all(pairs >= 0))
            self.assertTrue(torch.all(pairs < self.num_nodes))
        finally:
            # Restore the original value
            if original_value is not None:
                self.random_walk_sampler.has_torch_cluster = original_value
    
    def test_random_walk_fallback_methods(self):
        """Test the fallback methods in RandomWalkSampler."""
        # Test direct call to generate fallback positive pairs
        pairs = self.random_walk_sampler._generate_fallback_positive_pairs(15)
        self.assertEqual(pairs.shape[0], 15)
        self.assertEqual(pairs.shape[1], 2)
        
        # Test direct call to generate fallback negative pairs
        neg_pairs = self.random_walk_sampler._generate_fallback_negative_pairs(18)
        self.assertEqual(neg_pairs.shape[0], 18)
        self.assertEqual(neg_pairs.shape[1], 2)
        
        # Test direct call to sample positive pairs using fallback
        self.random_walk_sampler.has_torch_cluster = False  # Force fallback
        fallback_pairs = self.random_walk_sampler._sample_positive_pairs_fallback(12)
        self.assertEqual(fallback_pairs.shape[0], 12)
        self.assertEqual(fallback_pairs.shape[1], 2)
    
    def test_edge_cases_extreme(self):
        """Test extreme edge cases."""
        # Create a sampler with a single edge (self-loop)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        sampler = RandomWalkSampler(
            edge_index=edge_index,
            num_nodes=1,
            batch_size=5
        )
        
        # Check that sampling still works
        pairs = sampler.sample_positive_pairs()
        self.assertEqual(pairs.shape[0], 5)
        self.assertTrue(torch.all(pairs == 0))  # All should be node 0
        
        # Test triplet sampling with a single node
        anchors, positives, negatives = sampler.sample_triplets()
        self.assertEqual(anchors.shape[0], 5)
        self.assertTrue(torch.all(anchors == 0))
        self.assertTrue(torch.all(positives == 0))
        self.assertTrue(torch.all(negatives == 0))
        
        # Test subgraph sampling with a single node
        nodes = torch.tensor([0])
        subset_nodes, subset_edge_index = sampler.sample_subgraph(nodes)
        self.assertEqual(subset_nodes.numel(), 1)
        self.assertEqual(subset_nodes[0].item(), 0)
        self.assertEqual(subset_edge_index.shape[1], 1)  # Only the self-loop
        
        # Test with empty edge index but multiple nodes
        empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
        sampler = RandomWalkSampler(
            edge_index=empty_edge_index,
            num_nodes=10,
            batch_size=8
        )
        
        # Make sure it falls back properly for various methods
        pairs = sampler.sample_positive_pairs()
        self.assertEqual(pairs.shape[0], 8)
        self.assertEqual(pairs.shape[1], 2)
        
        # Test sample_neighbors with empty edge index
        result = sampler.sample_neighbors(torch.tensor([0, 1, 2]))
        # RandomWalkSampler.sample_neighbors returns a tuple unlike NeighborSampler
        self.assertTrue(isinstance(result, tuple))
        # The tuple should contain empty lists/tensors since there are no neighbors
        
        # Test sample_subgraph with empty edge index
        nodes = torch.tensor([0, 1, 2])
        subset_nodes, subset_edge_index = sampler.sample_subgraph(nodes)
        self.assertEqual(subset_nodes.numel(), 3)  # Should include seed nodes
        self.assertEqual(subset_edge_index.shape[1], 0)  # No edges


if __name__ == "__main__":
    run_all_tests()
