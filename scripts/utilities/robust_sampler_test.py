"""
Robust test script for the NeighborSampler and RandomWalkSampler classes.

This version includes extensive error handling and debugging information
to identify and fix issues with both sampler implementations.
"""

import os
import sys
import torch
import logging
import unittest
import time
from typing import Tuple, List, Dict, Optional, Set, Any

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.isne.training.sampler import NeighborSampler, RandomWalkSampler


def test_empty_graph():
    """Test that empty graphs are handled gracefully."""
    logger.info("Testing samplers with empty graphs")
    
    # Create an empty edge index
    empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
    num_nodes = 20
    batch_size = 10  # Ensure batch_size < num_nodes
    
    # Test NeighborSampler with replacement=True
    logger.info("Testing NeighborSampler with empty graph (replace=True)")
    try:
        sampler = NeighborSampler(
            edge_index=empty_edge_index,
            num_nodes=num_nodes,
            batch_size=batch_size,
            replace=True
        )
        
        # Test sampling methods
        logger.info("  Testing sample_nodes()...")
        nodes = sampler.sample_nodes()
        logger.info(f"  - Sampled {len(nodes)} nodes")
        
        logger.info("  Testing sample_edges()...")
        edges = sampler.sample_edges()
        logger.info(f"  - Sampled {len(edges)} edges")
        
        logger.info("  Testing sample_positive_pairs()...")
        pos_pairs = sampler.sample_positive_pairs()
        logger.info(f"  - Sampled {len(pos_pairs)} positive pairs")
        
        logger.info("  Testing sample_negative_pairs()...")
        neg_pairs = sampler.sample_negative_pairs()
        logger.info(f"  - Sampled {len(neg_pairs)} negative pairs")
        
        logger.info("  All NeighborSampler tests passed!")
    except Exception as e:
        logger.error(f"Error in NeighborSampler (replace=True): {type(e).__name__}: {e}")
        raise
    
    # Test NeighborSampler with replacement=False
    logger.info("Testing NeighborSampler with empty graph (replace=False)")
    try:
        sampler = NeighborSampler(
            edge_index=empty_edge_index,
            num_nodes=num_nodes,
            batch_size=batch_size,
            replace=False
        )
        
        # Test sampling methods
        logger.info("  Testing sample_nodes()...")
        nodes = sampler.sample_nodes()
        logger.info(f"  - Sampled {len(nodes)} nodes")
        
        logger.info("  Testing sample_edges()...")
        edges = sampler.sample_edges()
        logger.info(f"  - Sampled {len(edges)} edges")
        
        logger.info("  Testing sample_positive_pairs()...")
        pos_pairs = sampler.sample_positive_pairs()
        logger.info(f"  - Sampled {len(pos_pairs)} positive pairs")
        
        logger.info("  Testing sample_negative_pairs()...")
        neg_pairs = sampler.sample_negative_pairs()
        logger.info(f"  - Sampled {len(neg_pairs)} negative pairs")
        
        logger.info("  All NeighborSampler tests passed!")
    except Exception as e:
        logger.error(f"Error in NeighborSampler (replace=False): {type(e).__name__}: {e}")
        raise
    
    # Test RandomWalkSampler
    logger.info("Testing RandomWalkSampler with empty graph")
    try:
        sampler = RandomWalkSampler(
            edge_index=empty_edge_index,
            num_nodes=num_nodes,
            batch_size=batch_size
        )
        
        # Test sampling methods
        logger.info("  Testing sample_nodes()...")
        nodes = sampler.sample_nodes()
        logger.info(f"  - Sampled {len(nodes)} nodes")
        
        logger.info("  Testing sample_positive_pairs()...")
        pos_pairs = sampler.sample_positive_pairs()
        logger.info(f"  - Sampled {len(pos_pairs)} positive pairs")
        
        logger.info("  Testing sample_negative_pairs()...")
        neg_pairs = sampler.sample_negative_pairs()
        logger.info(f"  - Sampled {len(neg_pairs)} negative pairs")
        
        logger.info("  All RandomWalkSampler tests passed!")
    except Exception as e:
        logger.error(f"Error in RandomWalkSampler: {type(e).__name__}: {e}")
        raise
    
    logger.info("All empty graph tests completed successfully!")


def test_ring_graph():
    """Test samplers on a simple ring graph."""
    logger.info("Testing samplers with ring graph")
    
    # Create a simple ring graph
    num_nodes = 20
    batch_size = 10
    
    # Create a ring graph where each node is connected to its neighbors
    src = torch.arange(0, num_nodes)
    dst = torch.roll(src, -1)  # Connect each node to the next one in a ring
    
    # Create the edge index
    edge_index = torch.stack([src, dst], dim=0)
    
    # Add reverse edges to make it undirected
    reverse_edge_index = torch.stack([dst, src], dim=0)
    edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
    
    # Test NeighborSampler
    logger.info("Testing NeighborSampler with ring graph")
    try:
        sampler = NeighborSampler(
            edge_index=edge_index,
            num_nodes=num_nodes,
            batch_size=batch_size,
            num_hops=2,
            neighbor_size=5,
            seed=42
        )
        
        # Test sampling methods
        logger.info("  Testing sample_nodes()...")
        nodes = sampler.sample_nodes()
        logger.info(f"  - Sampled {len(nodes)} nodes")
        
        logger.info("  Testing sample_edges()...")
        edges = sampler.sample_edges()
        logger.info(f"  - Sampled {len(edges)} edges")
        
        logger.info("  Testing sample_positive_pairs()...")
        pos_pairs = sampler.sample_positive_pairs()
        logger.info(f"  - Sampled {len(pos_pairs)} positive pairs")
        
        logger.info("  Testing sample_negative_pairs()...")
        neg_pairs = sampler.sample_negative_pairs()
        logger.info(f"  - Sampled {len(neg_pairs)} negative pairs")
        
        logger.info("  Testing sample_triplets()...")
        anchors, positives, negatives = sampler.sample_triplets()
        logger.info(f"  - Sampled {len(anchors)} triplets")
        
        logger.info("  Testing sample_subgraph()...")
        seed_nodes = torch.tensor([0, 5, 10, 15])
        subset_nodes, subgraph_edge_index = sampler.sample_subgraph(seed_nodes)
        logger.info(f"  - Sampled subgraph with {len(subset_nodes)} nodes and {subgraph_edge_index.size(1)} edges")
        
        logger.info("  All NeighborSampler tests passed!")
    except Exception as e:
        logger.error(f"Error in NeighborSampler with ring graph: {type(e).__name__}: {e}")
        raise
    
    # Test RandomWalkSampler
    logger.info("Testing RandomWalkSampler with ring graph")
    try:
        sampler = RandomWalkSampler(
            edge_index=edge_index,
            num_nodes=num_nodes,
            batch_size=batch_size,
            walk_length=5,
            context_size=2,
            walks_per_node=5,
            seed=42
        )
        
        # Test sampling methods
        logger.info("  Testing sample_nodes()...")
        nodes = sampler.sample_nodes()
        logger.info(f"  - Sampled {len(nodes)} nodes")
        
        logger.info("  Testing sample_positive_pairs()...")
        pos_pairs = sampler.sample_positive_pairs()
        logger.info(f"  - Sampled {len(pos_pairs)} positive pairs")
        
        logger.info("  Testing sample_negative_pairs()...")
        neg_pairs = sampler.sample_negative_pairs()
        logger.info(f"  - Sampled {len(neg_pairs)} negative pairs")
        
        logger.info("  Testing sample_triplets()...")
        anchors, positives, negatives = sampler.sample_triplets()
        logger.info(f"  - Sampled {len(anchors)} triplets")
        
        logger.info("  Testing sample_subgraph()...")
        seed_nodes = torch.tensor([0, 5, 10, 15])
        subset_nodes, subgraph_edge_index = sampler.sample_subgraph(seed_nodes)
        logger.info(f"  - Sampled subgraph with {len(subset_nodes)} nodes and {subgraph_edge_index.size(1)} edges")
        
        logger.info("  All RandomWalkSampler tests passed!")
    except Exception as e:
        logger.error(f"Error in RandomWalkSampler with ring graph: {type(e).__name__}: {e}")
        raise
    
    logger.info("All ring graph tests completed successfully!")


def test_large_batch_size():
    """Test samplers with batch_size > num_nodes."""
    logger.info("Testing samplers with batch_size > num_nodes")
    
    # Create a simple ring graph
    num_nodes = 10
    batch_size = 20  # Intentionally larger than num_nodes
    
    # Create a ring graph where each node is connected to its neighbors
    src = torch.arange(0, num_nodes)
    dst = torch.roll(src, -1)  # Connect each node to the next one in a ring
    
    # Create the edge index
    edge_index = torch.stack([src, dst], dim=0)
    
    # Add reverse edges to make it undirected
    reverse_edge_index = torch.stack([dst, src], dim=0)
    edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
    
    # Test NeighborSampler with replace=False (should automatically switch to replace=True)
    logger.info("Testing NeighborSampler with batch_size > num_nodes and replace=False")
    try:
        sampler = NeighborSampler(
            edge_index=edge_index,
            num_nodes=num_nodes,
            batch_size=batch_size,
            replace=False,
            seed=42
        )
        
        # Test sampling methods
        logger.info("  Testing sample_nodes()...")
        nodes = sampler.sample_nodes()
        logger.info(f"  - Sampled {len(nodes)} nodes")
        assert len(nodes) == batch_size, f"Expected {batch_size} nodes, got {len(nodes)}"
        
        logger.info("  All tests passed!")
    except Exception as e:
        logger.error(f"Error in NeighborSampler with batch_size > num_nodes: {type(e).__name__}: {e}")
        raise
    
    logger.info("All large batch size tests completed successfully!")


def run_all_tests():
    """Run all tests."""
    try:
        # Run empty graph tests
        test_empty_graph()
        
        # Run ring graph tests
        test_ring_graph()
        
        # Run large batch size tests
        test_large_batch_size()
        
        logger.info("All tests completed successfully!")
    except Exception as e:
        logger.error(f"Test suite failed with error: {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
