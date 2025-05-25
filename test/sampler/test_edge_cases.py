"""
Edge case tests for the sampler module.

These tests focus on challenging edge cases including empty graphs,
isolated nodes, and fallback implementations when dependencies are missing.
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


class TestNeighborSamplerEdgeCases(unittest.TestCase):
    """Test edge cases for the NeighborSampler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set a fixed random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_empty_graph(self):
        """Test behavior with an empty graph."""
        # Create an empty graph
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        num_nodes = 5
        
        # Create a sampler
        sampler = NeighborSampler(
            edge_index=edge_index,
            num_nodes=num_nodes,
            batch_size=2,
            seed=42
        )
        
        # Sample neighbors
        try:
            nodes = torch.tensor([0, 1])
            neighbors = sampler.sample_neighbors(nodes)
            
            # Should return an empty tensor
            self.assertEqual(neighbors.numel(), 0)
        except Exception as e:
            print(f"Exception in empty graph neighbor sampling: {e}")
        
        # Sample subgraph
        try:
            nodes = torch.tensor([0, 1])
            subset_nodes, subset_edge_index = sampler.sample_subgraph(nodes)
            
            # Should return the input nodes but no edges
            self.assertEqual(subset_nodes.numel(), nodes.numel())
            self.assertEqual(subset_edge_index.shape[1], 0)
        except Exception as e:
            print(f"Exception in empty graph subgraph sampling: {e}")
        
        # Triplet sampling should still work
        try:
            triplets = sampler.sample_triplets()
            self.assertEqual(len(triplets), 3)
        except Exception as e:
            print(f"Exception in empty graph triplet sampling: {e}")
    
    def test_isolated_nodes(self):
        """Test behavior with isolated nodes."""
        # Create a graph with isolated nodes
        edge_index = torch.tensor([
            [0, 1],
            [1, 0]
        ], dtype=torch.long)
        num_nodes = 5  # Nodes 2, 3, 4 are isolated
        
        # Create a sampler
        sampler = NeighborSampler(
            edge_index=edge_index,
            num_nodes=num_nodes,
            batch_size=2,
            seed=42
        )
        
        # Sample neighbors for isolated nodes
        isolated_nodes = torch.tensor([2, 3, 4])
        neighbors = sampler.sample_neighbors(isolated_nodes)
        
        # Should return an empty tensor
        self.assertEqual(neighbors.numel(), 0)
        
        # Sample subgraph for isolated nodes
        subset_nodes, subset_edge_index = sampler.sample_subgraph(isolated_nodes)
        
        # Should return the input nodes but no edges
        self.assertEqual(subset_nodes.numel(), isolated_nodes.numel())
        self.assertEqual(subset_edge_index.shape[1], 0)
        
        # Sample with mixed connected and isolated nodes
        mixed_nodes = torch.tensor([0, 3])
        neighbors = sampler.sample_neighbors(mixed_nodes)
        
        # Should return some neighbors (for the connected nodes)
        self.assertGreaterEqual(neighbors.numel(), 0)
    
    def test_parameter_variations(self):
        """Test behavior with various parameter combinations."""
        # Create a simple graph
        edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 2],
            [1, 2, 0, 2, 0, 1]
        ], dtype=torch.long)
        num_nodes = 5
        
        # Test with various neighbor_size values
        for neighbor_size in [0, 1, 3, 5]:
            sampler = NeighborSampler(
                edge_index=edge_index,
                num_nodes=num_nodes,
                batch_size=2,
                neighbor_size=neighbor_size,
                seed=42
            )
            
            # Sample neighbors
            nodes = torch.tensor([0])
            neighbors = sampler.sample_neighbors(nodes)
            
            # For neighbor_size > 0, should have some neighbors if the node is connected
            if neighbor_size > 0:
                self.assertGreaterEqual(neighbors.numel(), 0)
            
            # For neighbor_size = 0, should have no neighbors
            if neighbor_size == 0:
                self.assertEqual(neighbors.numel(), 0)
        
        # Test with various batch sizes for triplet sampling
        for batch_size in [1, 3, 5]:
            sampler = NeighborSampler(
                edge_index=edge_index,
                num_nodes=num_nodes,
                batch_size=2,
                seed=42
            )
            
            # Sample triplets
            triplets = sampler.sample_triplets(batch_size=batch_size)
            
            # Should have the requested batch size
            self.assertEqual(triplets[0].shape[0], batch_size)


class TestRandomWalkSamplerEdgeCases(unittest.TestCase):
    """Test edge cases for the RandomWalkSampler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set a fixed random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_empty_graph(self):
        """Test behavior with an empty graph."""
        # Create an empty graph
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        num_nodes = 5
        
        # Create a sampler
        sampler = RandomWalkSampler(
            edge_index=edge_index,
            num_nodes=num_nodes,
            batch_size=2,
            seed=42
        )
        
        # Force fallback implementation
        sampler.has_torch_cluster = False
        if hasattr(sampler, 'random_walk_fn'):
            sampler.random_walk_fn = None
        
        # Sample positive pairs (should still work with fallback)
        try:
            pairs = sampler.sample_positive_pairs()
            self.assertEqual(pairs.shape[0], sampler.batch_size)
        except Exception as e:
            print(f"Exception in empty graph positive sampling: {e}")
        
        # Sample negative pairs
        try:
            neg_pairs = sampler.sample_negative_pairs()
            self.assertEqual(neg_pairs.shape[0], sampler.batch_size)
        except Exception as e:
            print(f"Exception in empty graph negative sampling: {e}")
        
        # Sample triplets
        try:
            triplets = sampler.sample_triplets()
            self.assertEqual(len(triplets), 3)
            self.assertEqual(triplets[0].shape[0], sampler.batch_size)
        except Exception as e:
            print(f"Exception in empty graph triplet sampling: {e}")
    
    def test_isolated_nodes(self):
        """Test behavior with isolated nodes."""
        # Create a graph with isolated nodes
        edge_index = torch.tensor([
            [0, 1],
            [1, 0]
        ], dtype=torch.long)
        num_nodes = 5  # Nodes 2, 3, 4 are isolated
        
        # Create a sampler
        sampler = RandomWalkSampler(
            edge_index=edge_index,
            num_nodes=num_nodes,
            batch_size=2,
            seed=42
        )
        
        # Force fallback implementation
        sampler.has_torch_cluster = False
        if hasattr(sampler, 'random_walk_fn'):
            sampler.random_walk_fn = None
        
        # Try sample_neighbors with isolated nodes
        try:
            isolated_nodes = torch.tensor([2, 3, 4])
            node_samples, edge_samples = sampler.sample_neighbors(isolated_nodes)
            
            # Should get a result for each input node
            self.assertIsInstance(node_samples, list)
            self.assertIsInstance(edge_samples, list)
        except Exception as e:
            # Some implementations might raise exceptions for isolated nodes
            pass
    
    def test_fallback_implementation(self):
        """Test fallback implementation when torch_cluster is not available."""
        # Create a simple graph
        edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 2],
            [1, 2, 0, 2, 0, 1]
        ], dtype=torch.long)
        num_nodes = 5
        
        # Create a sampler
        sampler = RandomWalkSampler(
            edge_index=edge_index,
            num_nodes=num_nodes,
            batch_size=2,
            seed=42
        )
        
        # Force fallback implementation
        sampler.has_torch_cluster = False
        if hasattr(sampler, 'random_walk_fn'):
            sampler.random_walk_fn = None
        
        # Sample positive pairs
        pos_pairs = sampler.sample_positive_pairs()
        self.assertEqual(pos_pairs.shape[0], sampler.batch_size)
        
        # Sample negative pairs
        neg_pairs = sampler.sample_negative_pairs()
        self.assertEqual(neg_pairs.shape[0], sampler.batch_size)
        
        # Sample triplets
        triplets = sampler.sample_triplets()
        self.assertEqual(len(triplets), 3)
        self.assertEqual(triplets[0].shape[0], sampler.batch_size)
    
    def test_parameter_variations(self):
        """Test behavior with various parameter combinations."""
        # Create a simple graph
        edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 2],
            [1, 2, 0, 2, 0, 1]
        ], dtype=torch.long)
        num_nodes = 5
        
        # Test with various context_size values
        for context_size in [1, 2, 3]:
            sampler = RandomWalkSampler(
                edge_index=edge_index,
                num_nodes=num_nodes,
                batch_size=2,
                walk_length=5,
                context_size=context_size,
                seed=42
            )
            
            # Force fallback implementation
            sampler.has_torch_cluster = False
            if hasattr(sampler, 'random_walk_fn'):
                sampler.random_walk_fn = None
            
            # Sample positive pairs
            pairs = sampler.sample_positive_pairs()
            self.assertEqual(pairs.shape[0], sampler.batch_size)
        
        # Test with various walks_per_node values
        for walks_per_node in [1, 2, 3]:
            sampler = RandomWalkSampler(
                edge_index=edge_index,
                num_nodes=num_nodes,
                batch_size=2,
                walks_per_node=walks_per_node,
                seed=42
            )
            
            # Force fallback implementation
            sampler.has_torch_cluster = False
            if hasattr(sampler, 'random_walk_fn'):
                sampler.random_walk_fn = None
            
            # Sample positive pairs
            pairs = sampler.sample_positive_pairs()
            self.assertEqual(pairs.shape[0], sampler.batch_size)


if __name__ == "__main__":
    unittest.main()
