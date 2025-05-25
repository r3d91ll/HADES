"""
Unit tests for the NeighborSampler class.

This module tests the functionality of the NeighborSampler to ensure it:
1. Correctly samples nodes and edges from the graph
2. Correctly samples neighborhoods around nodes
3. Correctly samples positive and negative pairs
4. Handles edge cases gracefully
5. Validates indices properly to prevent out-of-bounds errors
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

from src.isne.training.sampler import NeighborSampler


class TestNeighborSampler(unittest.TestCase):
    """Test suite for the NeighborSampler class."""

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
        self.sampler = NeighborSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            batch_size=16,
            num_hops=2,
            neighbor_size=5,
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
    
    def _print_stats(self, tensor, name):
        """Print statistics about a tensor for debugging."""
        self.logger.info(f"--- {name} Stats ---")
        self.logger.info(f"Shape: {tensor.shape}")
        if tensor.numel() > 0:
            self.logger.info(f"Min: {tensor.min()}")
            self.logger.info(f"Max: {tensor.max()}")
            self.logger.info(f"Mean: {tensor.float().mean()}")
        else:
            self.logger.info("Tensor is empty")
    
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
        
        # Print statistics
        self._print_stats(nodes, "Sampled Nodes")
        
        # Check basic properties
        self.assertEqual(nodes.shape[0], self.sampler.batch_size)
        
        # All indices should be within bounds
        self.assertTrue(torch.all(nodes >= 0))
        self.assertTrue(torch.all(nodes < self.num_nodes))
    
    def test_sample_edges(self):
        """Test sampling edges from the graph."""
        # Sample edges
        edges = self.sampler.sample_edges()
        
        # Print statistics
        self._print_stats(edges, "Sampled Edges")
        
        # Check basic properties
        self.assertEqual(edges.shape[0], self.sampler.batch_size)
        self.assertEqual(edges.shape[1], 2)
        
        # All indices should be within bounds
        self.assertTrue(torch.all(edges >= 0))
        self.assertTrue(torch.all(edges < self.num_nodes))
    
    def test_sample_neighbors(self):
        """Test sampling neighbors for a batch of nodes."""
        # Sample some seed nodes
        seed_nodes = torch.tensor([0, 5, 10, 15])
        
        # Sample neighbors
        neighbors = self.sampler.sample_neighbors(seed_nodes)
        
        # Print statistics
        self._print_stats(neighbors, "Sampled Neighbors")
        
        # Check that neighbors are valid
        if neighbors.numel() > 0:
            self.assertTrue(torch.all(neighbors >= 0))
            self.assertTrue(torch.all(neighbors < self.num_nodes))
            
            # Verify that neighbors are not in the seed nodes
            for n in neighbors:
                self.assertNotIn(n.item(), seed_nodes.tolist())
    
    def test_sample_subgraph(self):
        """Test sampling a subgraph around given nodes."""
        # Sample some seed nodes
        seed_nodes = torch.tensor([0, 5, 10, 15])
        
        # Sample subgraph
        subset_nodes, subgraph_edge_index = self.sampler.sample_subgraph(seed_nodes)
        
        # Print statistics
        self._print_stats(subset_nodes, "Subgraph Nodes")
        self._print_stats(subgraph_edge_index, "Subgraph Edge Index")
        
        # Check basic properties
        self.assertTrue(subset_nodes.numel() >= seed_nodes.numel())
        
        # All seed nodes should be in the subset
        for node in seed_nodes:
            self.assertIn(node.item(), subset_nodes.tolist())
        
        # Subgraph edge index should be valid
        if subgraph_edge_index.numel() > 0:
            self.assertEqual(subgraph_edge_index.shape[0], 2)
            max_idx = subset_nodes.size(0) - 1
            self.assertTrue(torch.all(subgraph_edge_index >= 0))
            self.assertTrue(torch.all(subgraph_edge_index <= max_idx))
    
    def test_sample_positive_pairs(self):
        """Test sampling positive pairs from the graph."""
        # Sample positive pairs
        pos_pairs = self.sampler.sample_positive_pairs()
        
        # Print statistics
        self._print_stats(pos_pairs, "Positive Pairs")
        
        # Check basic properties
        self.assertEqual(pos_pairs.shape[0], self.sampler.batch_size)
        self.assertEqual(pos_pairs.shape[1], 2)
        
        # All indices should be within bounds
        self.assertTrue(torch.all(pos_pairs >= 0))
        self.assertTrue(torch.all(pos_pairs < self.num_nodes))
        
        # Verify that at least some pairs are connected in the original graph
        connected_count = 0
        for i in range(pos_pairs.shape[0]):
            src = pos_pairs[i, 0].item()
            dst = pos_pairs[i, 1].item()
            
            # Check if this is a valid edge in the adjacency list
            if dst in self.sampler.adj_list[src]:
                connected_count += 1
        
        # At least some pairs should be connected
        self.assertGreater(connected_count, 0)
    
    def test_sample_negative_pairs(self):
        """Test sampling negative pairs from the graph."""
        # Sample positive pairs first
        pos_pairs = self.sampler.sample_positive_pairs()
        
        # Sample negative pairs
        neg_pairs = self.sampler.sample_negative_pairs(positive_pairs=pos_pairs)
        
        # Print statistics
        self._print_stats(neg_pairs, "Negative Pairs")
        
        # Check basic properties
        self.assertEqual(neg_pairs.shape[0], self.sampler.batch_size)
        self.assertEqual(neg_pairs.shape[1], 2)
        
        # All indices should be within bounds
        self.assertTrue(torch.all(neg_pairs >= 0))
        self.assertTrue(torch.all(neg_pairs < self.num_nodes))
        
        # Verify that pairs are not self-loops
        for i in range(neg_pairs.shape[0]):
            src = neg_pairs[i, 0].item()
            dst = neg_pairs[i, 1].item()
            self.assertNotEqual(src, dst)
            
        # Verify that at least some pairs are not connected in the original graph
        not_connected_count = 0
        for i in range(neg_pairs.shape[0]):
            src = neg_pairs[i, 0].item()
            dst = neg_pairs[i, 1].item()
            
            # Check if this is not a valid edge in the adjacency list
            if dst not in self.sampler.adj_list[src]:
                not_connected_count += 1
        
        # At least some pairs should not be connected
        self.assertGreater(not_connected_count, 0)
    
    def test_sample_triplets(self):
        """Test sampling triplets (anchor, positive, negative) from the graph."""
        # Sample triplets
        anchors, positives, negatives = self.sampler.sample_triplets()
        
        # Print statistics
        self._print_stats(anchors, "Anchor Nodes")
        self._print_stats(positives, "Positive Nodes")
        self._print_stats(negatives, "Negative Nodes")
        
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
        
        # Verify that (anchor, positive) pairs are connected and (anchor, negative) pairs are not
        for i in range(anchors.shape[0]):
            anchor = anchors[i].item()
            positive = positives[i].item()
            negative = negatives[i].item()
            
            # Anchor and positive should be connected
            self.assertIn(positive, self.sampler.adj_list[anchor])
            
            # Anchor and negative should not be connected
            self.assertNotIn(negative, self.sampler.adj_list[anchor])
    
    def test_empty_graph(self):
        """Test sampler behavior with an empty graph (edge case)."""
        # Create an empty edge index
        empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Create a sampler with the empty graph
        empty_sampler = NeighborSampler(
            edge_index=empty_edge_index,
            num_nodes=self.num_nodes,
            batch_size=16,
            seed=42
        )
        
        # Test sampling nodes (should still work)
        nodes = empty_sampler.sample_nodes()
        self.assertEqual(nodes.shape[0], empty_sampler.batch_size)
        
        # Test sampling edges (should return an empty tensor)
        edges = empty_sampler.sample_edges()
        self.assertEqual(edges.shape[0], 0)
        
        # Test sampling neighbors (should return an empty tensor)
        seed_nodes = torch.tensor([0, 5, 10, 15])
        neighbors = empty_sampler.sample_neighbors(seed_nodes)
        self.assertEqual(neighbors.numel(), 0)
        
        # Test sampling positive pairs (should use fallback mechanism)
        pos_pairs = empty_sampler.sample_positive_pairs()
        self.assertEqual(pos_pairs.shape[0], empty_sampler.batch_size)
        self.assertEqual(pos_pairs.shape[1], 2)
    
    def test_directed_vs_undirected(self):
        """Test the difference between directed and undirected sampling."""
        # Create a directed sampler
        directed_sampler = NeighborSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            batch_size=16,
            directed=True,
            seed=42
        )
        
        # Sample from both directed and undirected samplers
        undirected_neighbors = self.sampler.sample_neighbors(torch.tensor([0]))
        directed_neighbors = directed_sampler.sample_neighbors(torch.tensor([0]))
        
        # In a directed graph, the adjacency list should only contain outgoing edges
        # So there should be fewer neighbors in the directed case
        self.logger.info(f"Undirected neighbors count: {undirected_neighbors.numel()}")
        self.logger.info(f"Directed neighbors count: {directed_neighbors.numel()}")
        
        # The directed case might have fewer neighbors, but not guaranteed 
        # depending on the graph structure
        if undirected_neighbors.numel() > 0 and directed_neighbors.numel() > 0:
            # But the difference should be consistent
            undirected_adj_size = sum(len(adj) for adj in self.sampler.adj_list)
            directed_adj_size = sum(len(adj) for adj in directed_sampler.adj_list)
            self.logger.info(f"Undirected adjacency list size: {undirected_adj_size}")
            self.logger.info(f"Directed adjacency list size: {directed_adj_size}")
            
            # In an undirected graph, each edge appears twice in the adjacency list
            self.assertEqual(undirected_adj_size, self.edge_index.size(1))
            # In a directed graph, each edge appears once
            self.assertEqual(directed_adj_size, self.edge_index.size(1) // 2)


if __name__ == '__main__':
    unittest.main()
