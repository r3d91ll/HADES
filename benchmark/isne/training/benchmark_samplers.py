"""
Benchmark for the NeighborSampler and RandomWalkSampler classes.

This module benchmarks the performance of both sampler implementations
on graphs of different sizes to ensure efficient sampling operations.
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.isne.training.sampler import NeighborSampler, RandomWalkSampler


def generate_test_graph(num_nodes: int, avg_degree: int = 4) -> torch.Tensor:
    """Generate a random graph with specified number of nodes and average degree.
    
    Args:
        num_nodes: Number of nodes in the graph
        avg_degree: Average degree of each node (default: 4)
    
    Returns:
        Edge index tensor [2, num_edges]
    """
    # Calculate number of edges based on average degree
    num_edges = num_nodes * avg_degree // 2
    
    # Generate random edges
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    
    # Ensure no self-loops
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    
    # Recalculate to ensure we have enough edges
    while src.size(0) < num_edges:
        additional = num_edges - src.size(0)
        new_src = torch.randint(0, num_nodes, (additional * 2,))
        new_dst = torch.randint(0, num_nodes, (additional * 2,))
        mask = new_src != new_dst
        new_src = new_src[mask][:additional]
        new_dst = new_dst[mask][:additional]
        src = torch.cat([src, new_src])
        dst = torch.cat([dst, new_dst])
    
    # Create the edge index
    edge_index = torch.stack([src, dst], dim=0)
    
    # Add reverse edges to make it undirected
    reverse_edge_index = torch.stack([dst, src], dim=0)
    edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
    
    return edge_index


def benchmark_sampler(sampler_class, graph_sizes: List[int], batch_sizes: List[int], 
                      num_runs: int = 5) -> Dict[str, Dict[str, List[float]]]:
    """Benchmark a sampler class on various graph sizes and batch sizes.
    
    Args:
        sampler_class: The sampler class to benchmark (NeighborSampler or RandomWalkSampler)
        graph_sizes: List of graph sizes to test
        batch_sizes: List of batch sizes to test
        num_runs: Number of runs to average over
    
    Returns:
        Dictionary of timing results
    """
    results = {}
    sampler_name = sampler_class.__name__
    
    for graph_size in graph_sizes:
        print(f"Benchmarking {sampler_name} on graph with {graph_size} nodes")
        edge_index = generate_test_graph(graph_size)
        
        results[graph_size] = {}
        
        for batch_size in batch_sizes:
            # Create sampler
            if sampler_class == NeighborSampler:
                sampler = sampler_class(
                    edge_index=edge_index,
                    num_nodes=graph_size,
                    batch_size=batch_size,
                    num_hops=2,
                    neighbor_size=10
                )
            else:  # RandomWalkSampler
                sampler = sampler_class(
                    edge_index=edge_index,
                    num_nodes=graph_size,
                    batch_size=batch_size,
                    walk_length=5,
                    context_size=2,
                    walks_per_node=5
                )
            
            # Benchmark different operations
            operations = {
                "sample_nodes": lambda: sampler.sample_nodes(),
                "sample_positive_pairs": lambda: sampler.sample_positive_pairs(),
                "sample_negative_pairs": lambda: sampler.sample_negative_pairs(),
                "sample_subgraph": lambda: sampler.sample_subgraph(sampler.sample_nodes(batch_size=min(10, batch_size)))
            }
            
            results[graph_size][batch_size] = {}
            
            for op_name, op_func in operations.items():
                times = []
                for _ in range(num_runs):
                    start_time = time.time()
                    _ = op_func()
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = sum(times) / len(times)
                print(f"  Batch size {batch_size}, {op_name}: {avg_time:.5f} seconds")
                results[graph_size][batch_size][op_name] = avg_time
    
    return results


def plot_results(neighbor_results: Dict[str, Any], random_walk_results: Dict[str, Any], 
                 output_dir: str):
    """Plot benchmark results and save to files.
    
    Args:
        neighbor_results: Results from NeighborSampler benchmarks
        random_walk_results: Results from RandomWalkSampler benchmarks
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    operations = ["sample_nodes", "sample_positive_pairs", "sample_negative_pairs", "sample_subgraph"]
    graph_sizes = list(neighbor_results.keys())
    batch_sizes = list(neighbor_results[graph_sizes[0]].keys())
    
    # Plot 1: Scaling with graph size (fixed batch size)
    batch_size = batch_sizes[1]  # Use a middle batch size
    
    for op_name in operations:
        plt.figure(figsize=(10, 6))
        
        # NeighborSampler data
        ns_times = [neighbor_results[size][batch_size][op_name] for size in graph_sizes]
        plt.plot(graph_sizes, ns_times, 'o-', label='NeighborSampler')
        
        # RandomWalkSampler data
        rw_times = [random_walk_results[size][batch_size][op_name] for size in graph_sizes]
        plt.plot(graph_sizes, rw_times, 's-', label='RandomWalkSampler')
        
        plt.title(f'Performance scaling with graph size ({op_name}, batch_size={batch_size})')
        plt.xlabel('Number of nodes')
        plt.ylabel('Time (seconds)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/{op_name}_graph_scaling.png")
        plt.close()
    
    # Plot 2: Scaling with batch size (fixed graph size)
    graph_size = graph_sizes[1]  # Use a middle graph size
    
    for op_name in operations:
        plt.figure(figsize=(10, 6))
        
        # NeighborSampler data
        ns_times = [neighbor_results[graph_size][size][op_name] for size in batch_sizes]
        plt.plot(batch_sizes, ns_times, 'o-', label='NeighborSampler')
        
        # RandomWalkSampler data
        rw_times = [random_walk_results[graph_size][size][op_name] for size in batch_sizes]
        plt.plot(batch_sizes, rw_times, 's-', label='RandomWalkSampler')
        
        plt.title(f'Performance scaling with batch size ({op_name}, graph_size={graph_size})')
        plt.xlabel('Batch size')
        plt.ylabel('Time (seconds)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/{op_name}_batch_scaling.png")
        plt.close()


def main():
    """Run benchmarks and plot results."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define graph sizes and batch sizes to test
    graph_sizes = [100, 1000, 5000, 10000]
    batch_sizes = [16, 32, 64, 128, 256]
    
    # Benchmark NeighborSampler
    print("Benchmarking NeighborSampler...")
    neighbor_results = benchmark_sampler(NeighborSampler, graph_sizes, batch_sizes)
    
    # Benchmark RandomWalkSampler
    print("Benchmarking RandomWalkSampler...")
    random_walk_results = benchmark_sampler(RandomWalkSampler, graph_sizes, batch_sizes)
    
    # Plot and save results
    output_dir = os.path.join(os.path.dirname(__file__), '../../../benchmark/isne/training/results')
    plot_results(neighbor_results, random_walk_results, output_dir)
    
    print(f"Benchmark results saved to {output_dir}")


if __name__ == "__main__":
    main()
