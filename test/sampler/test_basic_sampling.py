"""
Basic tests for the sampler module focusing on core functionality.

These tests cover the fundamental operations of NeighborSampler and 
RandomWalkSampler classes.
"""

import os
import sys
import torch
import numpy as np
import unittest
from typing import Tuple, List, Dict, Optional, Set, Any, Callable
import pytest

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.isne.training.sampler import (
    NeighborSampler, 
    RandomWalkSampler
)


class TestNeighborSampler(unittest.TestCase):
    """Test the NeighborSampler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set a fixed random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create a test graph
        self.edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            [1, 2, 0, 3, 0, 4, 1, 4, 2, 3]
        ], dtype=torch.long)
        self.num_nodes = 5
        
        # Create a sampler
        self.sampler = NeighborSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            batch_size=2,
            num_hops=2,
            neighbor_size=2,
            seed=42
        )
    
    def test_initialization(self):
        """Test initialization of NeighborSampler."""
        self.assertEqual(self.sampler.num_nodes, self.num_nodes)
        self.assertEqual(self.sampler.batch_size, 2)
        self.assertEqual(self.sampler.num_hops, 2)
        self.assertEqual(self.sampler.neighbor_size, 2)
    
    def test_sample_neighbors(self):
        """Test sampling neighbors."""
        # Sample neighbors for specific nodes
        nodes = torch.tensor([0, 1])
        neighbors = self.sampler.sample_neighbors(nodes)
        
        # Should return a tensor
        self.assertIsInstance(neighbors, torch.Tensor)
        
        # Should have some neighbors
        self.assertGreater(neighbors.numel(), 0)
    
    def test_sample_subgraph(self):
        """Test sampling subgraph."""
        # Sample subgraph for specific nodes
        nodes = torch.tensor([0, 1])
        subset_nodes, subset_edge_index = self.sampler.sample_subgraph(nodes)
        
        # Should return tensors
        self.assertIsInstance(subset_nodes, torch.Tensor)
        self.assertIsInstance(subset_edge_index, torch.Tensor)
        
        # Should include the input nodes
        for node in nodes:
            self.assertIn(node.item(), subset_nodes.tolist())
        
        # Should have some edges
        self.assertEqual(subset_edge_index.dim(), 2)
        self.assertEqual(subset_edge_index.shape[0], 2)
        self.assertGreater(subset_edge_index.shape[1], 0)
    
    def test_sample_triplets(self):
        """Test sampling triplets."""
        # Sample triplets
        triplets = self.sampler.sample_triplets()
        
        # Should return a tuple of three tensors
        self.assertIsInstance(triplets, tuple)
        self.assertEqual(len(triplets), 3)
        
        # Each tensor should have the same batch size
        self.assertEqual(triplets[0].shape[0], self.sampler.batch_size)
        self.assertEqual(triplets[1].shape[0], self.sampler.batch_size)
        self.assertEqual(triplets[2].shape[0], self.sampler.batch_size)
    
    def test_custom_batch_size(self):
        """Test using custom batch size."""
        # Sample triplets with custom batch size
        batch_size = 4
        triplets = self.sampler.sample_triplets(batch_size=batch_size)
        
        # Each tensor should have the custom batch size
        self.assertEqual(triplets[0].shape[0], batch_size)
        self.assertEqual(triplets[1].shape[0], batch_size)
        self.assertEqual(triplets[2].shape[0], batch_size)


class TestRandomWalkSampler(unittest.TestCase):
    """Test the RandomWalkSampler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set a fixed random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create a test graph
        self.edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            [1, 2, 0, 3, 0, 4, 1, 4, 2, 3]
        ], dtype=torch.long)
        self.num_nodes = 5
        
        # Create a sampler
        self.sampler = RandomWalkSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            batch_size=2,
            walk_length=3,
            context_size=1,
            walks_per_node=2,
            seed=42
        )
    
    def test_initialization(self):
        """Test initialization of RandomWalkSampler."""
        self.assertEqual(self.sampler.num_nodes, self.num_nodes)
        self.assertEqual(self.sampler.batch_size, 2)
        self.assertEqual(self.sampler.walk_length, 3)
        self.assertEqual(self.sampler.context_size, 1)
        self.assertEqual(self.sampler.walks_per_node, 2)
    
    def test_sample_neighbors(self):
        """Test sampling neighbors."""
        # Sample neighbors for specific nodes
        nodes = torch.tensor([0, 1])
        node_samples, edge_samples = self.sampler.sample_neighbors(nodes)
        
        # Should return lists
        self.assertIsInstance(node_samples, list)
        self.assertIsInstance(edge_samples, list)
        
        # Should have an entry for each input node
        self.assertGreaterEqual(len(node_samples), 1)
        self.assertGreaterEqual(len(edge_samples), 1)
    
    def test_sample_subgraph(self):
        """Test sampling subgraph."""
        # Sample subgraph for specific nodes
        nodes = torch.tensor([0, 1])
        subset_nodes, subset_edge_index = self.sampler.sample_subgraph(nodes)
        
        # Should return tensors
        self.assertIsInstance(subset_nodes, torch.Tensor)
        self.assertIsInstance(subset_edge_index, torch.Tensor)
        
        # Should include the input nodes
        for node in nodes:
            self.assertIn(node.item(), subset_nodes.tolist())
        
        # Should have some edges
        self.assertEqual(subset_edge_index.dim(), 2)
        self.assertEqual(subset_edge_index.shape[0], 2)
        self.assertGreater(subset_edge_index.shape[1], 0)
    
    def test_sample_positive_pairs(self):
        """Test sampling positive pairs."""
        # Sample positive pairs
        pairs = self.sampler.sample_positive_pairs()
        
        # Should return a tensor
        self.assertIsInstance(pairs, torch.Tensor)
        
        # Should have batch_size rows and 2 columns
        self.assertEqual(pairs.shape[0], self.sampler.batch_size)
        self.assertEqual(pairs.shape[1], 2)
    
    def test_sample_negative_pairs(self):
        """Test sampling negative pairs."""
        # Sample negative pairs
        pairs = self.sampler.sample_negative_pairs()
        
        # Should return a tensor
        self.assertIsInstance(pairs, torch.Tensor)
        
        # Should have batch_size rows and 2 columns
        self.assertEqual(pairs.shape[0], self.sampler.batch_size)
        self.assertEqual(pairs.shape[1], 2)
    
    def test_sample_triplets(self):
        """Test sampling triplets."""
        # Sample triplets
        triplets = self.sampler.sample_triplets()
        
        # Should return a tuple of three tensors
        self.assertIsInstance(triplets, tuple)
        self.assertEqual(len(triplets), 3)
        
        # Each tensor should have the same batch size
        self.assertEqual(triplets[0].shape[0], self.sampler.batch_size)
        self.assertEqual(triplets[1].shape[0], self.sampler.batch_size)
        self.assertEqual(triplets[2].shape[0], self.sampler.batch_size)


if __name__ == "__main__":
    unittest.main()
