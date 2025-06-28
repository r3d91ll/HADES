# Supra-Weight Bootstrap Pipeline

## Overview

The supra-weight bootstrap pipeline creates a graph database in ArangoDB optimized for ISNE (Inductive Shallow Node Embedding) training. It addresses the key challenge of graph density by using **supra-weights** - multi-dimensional edge representations that aggregate multiple relationship types into single edges.

## Key Features

- **Multi-dimensional Relationships**: Detects 7 types of relationships between nodes
- **Supra-weight Aggregation**: Combines multiple relationships into single weighted edges
- **Density Control**: Prevents O(n²) edge explosion through intelligent filtering
- **Batch Processing**: Efficient processing of large datasets
- **ArangoDB Integration**: Direct storage in graph database format
- **Progress Tracking**: Real-time progress updates and statistics

## Architecture

### Core Components

1. **SupraWeightCalculator** (`core/supra_weight_calculator.py`)
   - Aggregates multiple relationships into single weights
   - Supports adaptive, weighted_sum, max, and harmonic_mean methods
   - Handles synergistic relationship bonuses

2. **RelationshipDetector** (`core/relationship_detector.py`)
   - Detects 7 relationship types:
     - Co-location (same directory)
     - Sequential (adjacent chunks)
     - Import (code dependencies)
     - Semantic (embedding similarity)
     - Temporal (time-based)
     - Reference (documentation links)
     - Structural (AST-based)

3. **DensityController** (`core/density_controller.py`)
   - Limits edges per node
   - Enforces minimum weight thresholds
   - Controls local cluster density
   - Provides global density targets

### Processing Components

1. **DocumentProcessor** (`processing/document_processor.py`)
   - Creates file-level nodes
   - Extracts metadata for relationship detection
   - Handles various input formats

2. **ChunkProcessor** (`processing/chunk_processor.py`)
   - Processes document chunks
   - Maintains document-chunk hierarchy
   - Handles chunk merging and deduplication

3. **EmbeddingProcessor** (`processing/embedding_processor.py`)
   - Validates embeddings
   - Computes batch similarities efficiently
   - GPU-accelerated when available

### Storage Components

1. **ArangoSupraStorage** (`storage/arango_supra_storage.py`)
   - Manages ArangoDB connections
   - Creates optimized collections and indices
   - Stores nodes and supra-weight edges

2. **BatchWriter** (`storage/batch_writer.py`)
   - Accumulates writes for efficiency
   - Automatic flushing based on size/time
   - Progress tracking and statistics

## Usage

### Basic Example

```python
from src.pipelines.bootstrap.supra_weight import SupraWeightBootstrapPipeline

# Configuration
config = {
    "database": {
        "url": "http://localhost:8529",
        "username": "root",
        "password": "",
        "database": "isne_bootstrap"
    },
    "bootstrap": {
        "batch_size": 1000,
        "max_edges_per_node": 50,
        "min_edge_weight": 0.3
    },
    "supra_weight": {
        "aggregation_method": "adaptive",
        "semantic_threshold": 0.5
    }
}

# Initialize pipeline
pipeline = SupraWeightBootstrapPipeline(config)

# Run with directory input
results = pipeline.run("/path/to/project")

# Or with existing nodes
nodes = [...]  # List of node dictionaries
results = pipeline.run(nodes)
```

### Input Formats

1. **Directory Path**: Processes all files recursively
2. **JSON File**: Loads nodes from JSON
3. **Node List**: Direct list of node dictionaries

### Node Format

```python
{
    "node_id": "unique_id",
    "node_type": "file" | "chunk",
    "file_path": "/path/to/file",
    "file_name": "file.py",
    "directory": "/path/to",
    "content": "file content...",
    "embedding": [0.1, 0.2, ...],  # Optional
    "chunk_index": 0,  # For chunks
    "source_file_id": "parent_id"  # For chunks
}
```

## Configuration

See `config_example.yaml` for detailed configuration options.

### Key Settings

- **max_edges_per_node**: Prevent hub nodes (default: 50)
- **min_edge_weight**: Filter weak relationships (default: 0.3)
- **aggregation_method**: How to combine relationships (default: "adaptive")
- **semantic_threshold**: Minimum similarity for semantic edges (default: 0.5)
- **target_density**: Global density limit (default: None)

## Supra-Weight Calculation

### Relationship Types and Default Weights

1. **Co-location** (1.0): Files in same directory
2. **Import** (0.95): Direct code dependencies
3. **Structural** (0.9): AST-based relationships
4. **Reference** (0.85): Documentation references
5. **Semantic** (0.7): Content similarity
6. **Sequential** (0.6): Adjacent chunks
7. **Temporal** (0.5): Time-based relationships

### Aggregation Methods

- **Adaptive**: Considers synergies between relationships
- **Weighted Sum**: Importance-weighted average
- **Max**: Maximum relationship strength
- **Harmonic Mean**: Good for averaging rates

### Edge Storage Format

```json
{
    "_from": "nodes/node1",
    "_to": "nodes/node2", 
    "weight": 0.85,
    "weight_vector": [1.0, 0.0, 0.0, 0.7, 0.0, 0.6, 0.0],
    "relationships": [
        {
            "type": "co_location",
            "strength": 1.0,
            "confidence": 1.0,
            "metadata": {"directory": "/src"}
        },
        {
            "type": "semantic",
            "strength": 0.7,
            "confidence": 0.9,
            "metadata": {"similarity_score": 0.7}
        }
    ]
}
```

## Performance Considerations

### Scalability

- Designed for graphs with 100k+ nodes
- Batch processing reduces memory footprint
- Density control prevents quadratic edge growth

### Optimization Tips

1. **Batch Size**: Larger batches improve throughput but use more memory
2. **GPU Usage**: Enable for faster similarity computation
3. **Density Limits**: Balance coverage vs. computational cost
4. **File Extensions**: Filter unnecessary files early

### Benchmarks

On a typical codebase:
- 10k files: ~2-5 minutes
- 100k chunks: ~15-30 minutes
- Edge reduction: 90-95% vs. fully connected

## Integration with ISNE

The bootstrap pipeline creates a graph optimized for ISNE training:

1. **Balanced Connectivity**: Controlled degree distribution
2. **Multi-dimensional Signals**: Rich relationship information
3. **Efficient Storage**: Single edge collection with weight vectors
4. **Training-Ready**: Direct compatibility with ISNE models

## Monitoring and Validation

### Progress Tracking

```python
def progress_callback(info):
    print(f"Progress: {info['progress']:.1%}")
    
results = pipeline.run(input_source, progress_callback=progress_callback)
```

### Statistics

The pipeline returns comprehensive statistics:
- Nodes processed
- Edges created
- Relationship type distribution
- Density control metrics
- Graph statistics (degree distribution, density)

### Validation

```python
validation = pipeline.validate_graph()
if validation['is_valid']:
    print("Graph validation passed")
else:
    print(f"Errors: {validation['errors']}")
```

## Troubleshooting

### Common Issues

1. **High Density Warning**: Adjust max_edges_per_node or min_edge_weight
2. **No Edges Created**: Lower semantic_threshold or min_edge_weight
3. **Memory Issues**: Reduce batch_size or similarity_batch_size
4. **Slow Processing**: Enable GPU or reduce file_extensions

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

1. **Incremental Updates**: Add new nodes without full rebuild
2. **Custom Relationships**: Plugin system for domain-specific relationships
3. **Distributed Processing**: Multi-machine graph construction
4. **Real-time Mode**: Stream processing for live systems