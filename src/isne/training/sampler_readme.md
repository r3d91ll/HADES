# ISNE Sampler Module Documentation

## Overview

The `sampler.py` module provides efficient graph sampling strategies for training ISNE (Incremental Stochastic Neighbor Embedding) models on large graphs. It implements two primary sampling approaches:

1. **NeighborSampler**: Samples nodes and edges based on local neighborhood structure.
2. **RandomWalkSampler**: Samples nodes and edges using random walks on the graph.

Both samplers are designed to handle large graphs efficiently and provide various sampling methods required for training graph embedding models.

## Classes

### NeighborSampler

The `NeighborSampler` class implements a neighborhood-based sampling strategy that:

- Creates an adjacency list representation for efficient neighbor access
- Samples nodes, edges, and neighbors using configurable parameters
- Supports multi-hop neighborhood sampling
- Generates positive and negative pairs for contrastive learning

#### Usage Example

```python
import torch
from src.isne.training.sampler import NeighborSampler

# Create a graph (edge_index format)
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
num_nodes = 3

# Create a sampler
sampler = NeighborSampler(
    edge_index=edge_index,
    num_nodes=num_nodes,
    batch_size=32,
    num_hops=2,
    neighbor_size=10
)

# Sample nodes
nodes = sampler.sample_nodes()

# Sample positive pairs
pos_pairs = sampler.sample_positive_pairs()

# Sample negative pairs
neg_pairs = sampler.sample_negative_pairs()

# Sample subgraph around specific nodes
seed_nodes = torch.tensor([0, 1])
subset_nodes, subgraph_edge_index = sampler.sample_subgraph(seed_nodes)
```

### RandomWalkSampler

The `RandomWalkSampler` class implements a random walk-based sampling strategy that:

- Uses CSR graph representation for efficient random walk computation
- Generates walks of configurable length starting from seed nodes
- Creates positive pairs from nodes co-occurring within a context window in the walks
- Creates negative pairs by sampling node pairs not in proximity
- Provides efficient batch-aware sampling methods

#### Usage Example

```python
import torch
from src.isne.training.sampler import RandomWalkSampler

# Create a graph (edge_index format)
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
num_nodes = 3

# Create a sampler
sampler = RandomWalkSampler(
    edge_index=edge_index,
    num_nodes=num_nodes,
    batch_size=32,
    walk_length=5,
    context_size=2,
    walks_per_node=10
)

# Sample nodes
nodes = sampler.sample_nodes()

# Sample positive pairs
pos_pairs = sampler.sample_positive_pairs()

# Sample negative pairs
neg_pairs = sampler.sample_negative_pairs()

# Sample triplets (anchor, positive, negative)
anchors, positives, negatives = sampler.sample_triplets()
```

## Dependencies

- **torch**: Used for tensor operations and random number generation
- **torch_geometric**: Used for utility functions and data structures
- **torch_cluster**: Optional, used for efficient random walks if available
- **numpy**: Used for numerical operations when needed

## Performance Considerations

- The samplers use efficient data structures (adjacency lists, CSR format) to minimize memory usage and computation time
- Both samplers include fallback mechanisms when dependencies like `torch_cluster` are not available
- Error handling ensures robustness even when sampling fails (e.g., in disconnected graphs)

## Testing

Both samplers have comprehensive unit tests in the `tests/unit/isne/training/` directory:

- `test_neighbor_sampler.py`: Tests for the `NeighborSampler` class
- `test_random_walk_sampler.py`: Tests for the `RandomWalkSampler` class

Run tests using pytest:

```bash
python -m pytest tests/unit/isne/training/test_neighbor_sampler.py -v
python -m pytest tests/unit/isne/training/test_random_walk_sampler.py -v
```

## Benchmarks

Benchmark scripts in the `benchmark/isne/training/` directory evaluate the performance of both samplers:

- `benchmark_samplers.py`: Benchmarks both samplers on graphs of different sizes

Run benchmarks:

```bash
python benchmark/isne/training/benchmark_samplers.py
```

## Future Improvements

- Implement GPU-accelerated sampling when processing large graphs
- Add distributed sampling capabilities for very large graphs
- Optimize memory usage for extremely large graphs
- Add more sophisticated negative sampling strategies
