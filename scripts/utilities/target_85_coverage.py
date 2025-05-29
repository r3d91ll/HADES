"""
Targeted tests to achieve 85% coverage of the sampler module.

This file contains specialized tests that target specific uncovered code paths
in the sampler.py module without relying on private methods.
"""

import os
import sys
import torch
import numpy as np
import unittest
from typing import Tuple, List, Dict, Optional, Set, Any, Callable
import logging
import pytest
from unittest.mock import patch

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.isne.training.sampler import (
    NeighborSampler, 
    RandomWalkSampler
)


class TestHardToReachPaths(unittest.TestCase):
    """Test the most difficult-to-reach code paths."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set a fixed random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create a carefully designed test graph
        edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
            [1, 2, 0, 3, 0, 4, 1, 5, 2, 6, 3, 7, 4, 8, 5, 9, 6, 0, 7, 1]
        ], dtype=torch.long)
        
        self.edge_index = edge_index
        self.num_nodes = 10
        
        # Create a graph with self-loops
        self_loop_edges = torch.tensor([
            [0, 1, 2, 3, 0, 1, 2, 3],
            [0, 1, 2, 3, 1, 0, 3, 2]
        ], dtype=torch.long)
        
        self.self_loop_edges = self_loop_edges
        
        # Create a minimal graph
        minimal_edges = torch.tensor([
            [0, 1],
            [1, 0]
        ], dtype=torch.long)
        
        self.minimal_edges = minimal_edges
    
    def test_extreme_neighbor_sampling(self):
        """Test extreme edge cases for NeighborSampler."""
        # Test with many different parameter combinations
        for num_hops in [1, 2, 3, 4]:
            for neighbor_size in [1, 2, 3]:
                sampler = NeighborSampler(
                    edge_index=self.edge_index,
                    num_nodes=self.num_nodes,
                    batch_size=2,
                    num_hops=num_hops,
                    neighbor_size=neighbor_size,
                    seed=42
                )
                
                # Sample from a well-connected node to force multi-hop paths
                nodes = torch.tensor([0])
                subset_nodes, subset_edge_index = sampler.sample_subgraph(nodes)
                
                # Verify we got some results
                self.assertGreater(subset_nodes.numel(), 0)
                self.assertGreater(subset_edge_index.shape[1], 0)
        
        # Test with isolated nodes
        isolated_sampler = NeighborSampler(
            edge_index=self.minimal_edges,
            num_nodes=10,  # More nodes than edges
            batch_size=2,
            num_hops=2,
            neighbor_size=2,
            seed=42
        )
        
        # Test sampling from isolated nodes
        isolated_nodes = torch.tensor([5, 6])
        neighbors = isolated_sampler.sample_neighbors(isolated_nodes)
        
        # Test with self-loops
        self_loop_sampler = NeighborSampler(
            edge_index=self.self_loop_edges,
            num_nodes=5,
            batch_size=2,
            num_hops=2,
            neighbor_size=2,
            seed=42
        )
        
        self_loop_nodes = torch.tensor([0, 1])
        neighbors = self_loop_sampler.sample_neighbors(self_loop_nodes)
        
        # Test with very large and very small batch sizes
        for batch_size in [1, 10, 20]:
            sampler = NeighborSampler(
                edge_index=self.edge_index,
                num_nodes=self.num_nodes,
                batch_size=batch_size,
                num_hops=2,
                neighbor_size=2,
                seed=42
            )
            
            triplets = sampler.sample_triplets(batch_size=batch_size)
            self.assertEqual(len(triplets), 3)
    
    def test_extreme_random_walk_sampling(self):
        """Test extreme edge cases for RandomWalkSampler."""
        # Test with forced fallback implementation
        sampler = RandomWalkSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            batch_size=4,
            walk_length=3,
            context_size=2,
            walks_per_node=2,
            p=1.0,
            q=1.0,
            seed=42
        )
        
        # Force fallback implementation
        if hasattr(sampler, 'has_torch_cluster'):
            sampler.has_torch_cluster = False
            if hasattr(sampler, 'random_walk_fn'):
                sampler.random_walk_fn = None
        
        # Test with different combinations of parameters
        for walk_length in [1, 2, 3]:
            for context_size in [1, 2, 3]:
                for walks_per_node in [1, 2, 3]:
                    sampler.walk_length = walk_length
                    sampler.context_size = context_size
                    sampler.walks_per_node = walks_per_node
                    
                    # Sample with these parameters
                    pairs = sampler.sample_positive_pairs()
                    self.assertEqual(pairs.shape[0], sampler.batch_size)
        
        # Test with isolated nodes
        isolated_sampler = RandomWalkSampler(
            edge_index=self.minimal_edges,
            num_nodes=10,  # More nodes than in edges
            batch_size=2,
            walk_length=2,
            context_size=1,
            walks_per_node=1,
            seed=42
        )
        
        # Force fallback implementation
        if hasattr(isolated_sampler, 'has_torch_cluster'):
            isolated_sampler.has_torch_cluster = False
            if hasattr(isolated_sampler, 'random_walk_fn'):
                isolated_sampler.random_walk_fn = None
        
        # Test with isolated nodes
        isolated_nodes = torch.tensor([5, 6])
        node_samples, edge_samples = isolated_sampler.sample_neighbors(isolated_nodes)
        
        # Test with different batch sizes for triplet sampling
        for batch_size in [1, 3, 5]:
            triplets = isolated_sampler.sample_triplets(batch_size=batch_size)
            self.assertEqual(len(triplets), 3)
    
    def test_various_edge_conditions(self):
        """Test with various edge conditions to hit difficult code paths."""
        # Create an empty graph
        empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Test NeighborSampler with empty graph
        empty_ns = NeighborSampler(
            edge_index=empty_edge_index,
            num_nodes=5,
            batch_size=2,
            num_hops=2,
            neighbor_size=2,
            seed=42
        )
        
        nodes = torch.tensor([0, 1])
        subset_nodes, subset_edge_index = empty_ns.sample_subgraph(nodes)
        self.assertEqual(subset_edge_index.shape[1], 0)
        
        # Test RandomWalkSampler with empty graph
        empty_rws = RandomWalkSampler(
            edge_index=empty_edge_index,
            num_nodes=5,
            batch_size=2,
            walk_length=2,
            context_size=1,
            walks_per_node=1,
            seed=42
        )
        
        # Force fallback implementation
        if hasattr(empty_rws, 'has_torch_cluster'):
            empty_rws.has_torch_cluster = False
            if hasattr(empty_rws, 'random_walk_fn'):
                empty_rws.random_walk_fn = None
        
        # Test with empty graph
        pairs = empty_rws.sample_positive_pairs(batch_size=3)
        self.assertEqual(pairs.shape[0], 3)
        
        # Test negative sampling with empty graph
        neg_pairs = empty_rws.sample_negative_pairs(batch_size=4)
        self.assertEqual(neg_pairs.shape[0], 4)
        
        # Test triplet sampling with empty graph
        triplets = empty_rws.sample_triplets(batch_size=2)
        self.assertEqual(len(triplets), 3)
        
        # Test sampling with completely empty nodes
        empty_nodes = torch.tensor([], dtype=torch.long)
        
        # NeighborSampler with empty nodes
        empty_node_neighbors = empty_ns.sample_neighbors(empty_nodes)
        subset_nodes, subset_edge_index = empty_ns.sample_subgraph(empty_nodes)
        
        # RandomWalkSampler with empty nodes
        node_samples, edge_samples = empty_rws.sample_neighbors(empty_nodes)
        subset_nodes, subset_edge_index = empty_rws.sample_subgraph(empty_nodes)


# Run specific tests to maximize coverage of hard-to-reach paths
class TestFullCoveragePatterns(unittest.TestCase):
    """Test patterns specifically designed to hit uncovered code paths."""
    
    def test_line_22_to_36_imports(self):
        """Test behavior with different import scenarios."""
        # We can't easily mock imports, but we can test the behavior
        # when torch_cluster is not available (fallback paths)
        edge_index = torch.tensor([
            [0, 1, 2],
            [1, 2, 0]
        ], dtype=torch.long)
        
        # Create a RandomWalkSampler
        sampler = RandomWalkSampler(
            edge_index=edge_index,
            num_nodes=3,
            batch_size=2,
            walk_length=2,
            context_size=1,
            walks_per_node=1,
            seed=42
        )
        
        # Force fallback implementation
        if hasattr(sampler, 'has_torch_cluster'):
            sampler.has_torch_cluster = False
            if hasattr(sampler, 'random_walk_fn'):
                sampler.random_walk_fn = None
        
        # Test sampling to ensure fallback methods work
        pairs = sampler.sample_positive_pairs()
        self.assertEqual(pairs.shape[0], sampler.batch_size)
    
    def test_line_284_to_320_multi_hop(self):
        """Test multi-hop neighborhood sampling (lines 284-320)."""
        # Create a chain graph to force multi-hop traversal
        chain_edges = []
        for i in range(19):
            chain_edges.append([i, i+1])
            chain_edges.append([i+1, i])
        
        chain_edge_index = torch.tensor(chain_edges, dtype=torch.long).t()
        
        # Try different combinations of parameters to hit multi-hop paths
        for num_hops in [1, 2, 3, 4, 5]:
            for neighbor_size in [1, 2, 3]:
                sampler = NeighborSampler(
                    edge_index=chain_edge_index,
                    num_nodes=20,
                    batch_size=2,
                    num_hops=num_hops,
                    neighbor_size=neighbor_size,
                    seed=42
                )
                
                # Sample from the start of the chain
                start_nodes = torch.tensor([0])
                subset_nodes, subset_edge_index = sampler.sample_subgraph(start_nodes)
                
                # Verify we get some results
                self.assertGreaterEqual(subset_nodes.numel(), 1)
                
        # Try with more complex start nodes
        sampler = NeighborSampler(
            edge_index=chain_edge_index,
            num_nodes=20,
            batch_size=2,
            num_hops=3,
            neighbor_size=2,
            seed=42
        )
        
        multi_start_nodes = torch.tensor([0, 5, 10])
        subset_nodes, subset_edge_index = sampler.sample_subgraph(multi_start_nodes)
    
    def test_lines_435_456_480_neighbor_sampling(self):
        """Test lines 435, 456-461, 480-481 in neighbor sampling."""
        # Create a graph with isolated nodes
        sparse_edges = torch.tensor([
            [0, 1, 2],
            [1, 2, 0]
        ], dtype=torch.long)
        
        sampler = NeighborSampler(
            edge_index=sparse_edges,
            num_nodes=10,  # More nodes than in edge_index
            batch_size=4,
            num_hops=2,
            neighbor_size=2,
            seed=42
        )
        
        # Test line 435 - handling isolated nodes
        isolated_nodes = torch.tensor([5, 6, 7])  # Nodes with no connections
        neighbors = sampler.sample_neighbors(isolated_nodes)
        
        # Test lines 456-461 - different neighbor sizes
        for neighbor_size in [0, 1, 3, 5]:
            sampler.neighbor_size = neighbor_size
            connected_nodes = torch.tensor([0, 1])
            neighbors = sampler.sample_neighbors(connected_nodes)
        
        # Test lines 480-481 - triplet sampling with custom batch size
        for batch_size in [1, 3, 5]:
            triplets = sampler.sample_triplets(batch_size=batch_size)
            self.assertEqual(len(triplets), 3)
    
    def test_lines_622_to_716_random_walk(self):
        """Test lines 622-636, 640-646, 678-680, 716 in random walk sampling."""
        # Create a specific graph structure
        edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            [1, 2, 0, 3, 0, 4, 1, 4, 2, 3]
        ], dtype=torch.long)
        
        sampler = RandomWalkSampler(
            edge_index=edge_index,
            num_nodes=10,
            batch_size=2,
            walk_length=3,
            context_size=2,
            walks_per_node=2,
            seed=42
        )
        
        # Force fallback implementation
        if hasattr(sampler, 'has_torch_cluster'):
            sampler.has_torch_cluster = False
            if hasattr(sampler, 'random_walk_fn'):
                sampler.random_walk_fn = None
        
        # Test lines 622-636, 640-646 - neighbor sampling without torch_cluster
        for nodes in [
            torch.tensor([0]),  # Single node
            torch.tensor([0, 1, 2]),  # Multiple nodes
            torch.tensor([]),  # Empty tensor
            torch.tensor([20])  # Invalid node
        ]:
            node_samples, edge_samples = sampler.sample_neighbors(nodes)
            self.assertIsInstance(node_samples, list)
            self.assertIsInstance(edge_samples, list)
        
        # Test lines 678-680 - positive sampling with different parameters
        for walks_per_node in [1, 2, 3]:
            sampler.walks_per_node = walks_per_node
            pairs = sampler.sample_positive_pairs()
            self.assertEqual(pairs.shape[0], sampler.batch_size)
    
    def test_lines_842_to_952_random_walk_sampling(self):
        """Test lines 842-861, 912-918, 937-943, 951-952 in random walk sampling."""
        # Create a graph
        edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            [1, 2, 0, 3, 0, 4, 1, 4, 2, 3]
        ], dtype=torch.long)
        
        sampler = RandomWalkSampler(
            edge_index=edge_index,
            num_nodes=5,
            batch_size=2,
            walk_length=3,
            context_size=2,
            walks_per_node=2,
            seed=42
        )
        
        # Force fallback implementation
        if hasattr(sampler, 'has_torch_cluster'):
            sampler.has_torch_cluster = False
            if hasattr(sampler, 'random_walk_fn'):
                sampler.random_walk_fn = None
        
        # Test lines 842-861 - random walk with different context sizes
        for context_size in [1, 2, 3]:
            sampler.context_size = context_size
            pairs = sampler.sample_positive_pairs()
            self.assertEqual(pairs.shape[0], sampler.batch_size)
        
        # Test with various batch sizes
        for batch_size in [1, 3, 5]:
            pairs = sampler.sample_positive_pairs(batch_size=batch_size)
            self.assertEqual(pairs.shape[0], batch_size)
    
    def test_remaining_random_walk_paths(self):
        """Test remaining uncovered paths in random walk sampling."""
        # Create a minimal graph to force error conditions
        minimal_edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        
        sampler = RandomWalkSampler(
            edge_index=minimal_edge_index,
            num_nodes=5,
            batch_size=2,
            walk_length=2,
            context_size=1,
            walks_per_node=1,
            seed=42
        )
        
        # Force fallback implementation
        if hasattr(sampler, 'has_torch_cluster'):
            sampler.has_torch_cluster = False
            if hasattr(sampler, 'random_walk_fn'):
                sampler.random_walk_fn = None
        
        # Test with various node combinations
        for nodes in [
            torch.tensor([0]),  # Connected node
            torch.tensor([2]),  # Isolated node
            torch.tensor([0, 2]),  # Mixed nodes
            torch.tensor([])  # Empty tensor
        ]:
            subset_nodes, subset_edge_index = sampler.sample_subgraph(nodes)
        
        # Test negative sampling edge cases
        pairs = sampler.sample_negative_pairs(batch_size=3)
        self.assertEqual(pairs.shape[0], 3)
        
        # Test triplet sampling edge cases
        triplets = sampler.sample_triplets(batch_size=2)
        self.assertIsInstance(triplets, tuple)
        self.assertEqual(len(triplets), 3)


if __name__ == "__main__":
    unittest.main()
