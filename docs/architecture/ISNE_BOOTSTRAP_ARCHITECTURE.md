# ISNE Bootstrap Pipeline with Supra-Weights Architecture

## Overview

This document describes the improved bootstrap pipeline for preparing ArangoDB graph data for ISNE training, incorporating supra-weights for multi-dimensional relationship representation.

## Architecture Goals

1. **Unified Edge Representation**: Single edge collection with supra-weights instead of multiple collections
2. **Scalability**: Handle datasets with 100k+ nodes efficiently
3. **Density Control**: Prevent graph explosion while preserving important relationships
4. **Batch Processing**: Process large datasets in manageable chunks
5. **Production Ready**: Proper error handling, logging, and monitoring

## Component Architecture

### 1. Core Components

```
SupraWeightBootstrap/
├── core/
│   ├── __init__.py
│   ├── supra_weight_calculator.py    # Calculate multi-dimensional weights
│   ├── relationship_detector.py       # Detect all relationship types
│   └── density_controller.py          # Control graph density
├── storage/
│   ├── __init__.py
│   ├── arango_supra_storage.py       # ArangoDB storage with supra-weights
│   └── batch_writer.py                # Efficient batch writing
├── processing/
│   ├── __init__.py
│   ├── document_processor.py          # Enhanced document processing
│   ├── chunk_processor.py             # Batch chunk processing
│   └── embedding_processor.py         # Efficient embedding generation
└── pipeline/
    ├── __init__.py
    ├── bootstrap_pipeline.py           # Main pipeline orchestrator
    └── progress_tracker.py             # Progress tracking
```

### 2. Database Schema

#### Node Collections
- `nodes`: Unified node collection with type field
  - `_key`: Unique identifier
  - `node_type`: 'file', 'chunk', 'concept'
  - `content`: Text content
  - `metadata`: Additional properties
  - `embedding`: Vector representation

#### Edge Collection (Single)
- `supra_edges`: Multi-dimensional weighted edges
  - `_from`, `_to`: Node references
  - `relationships`: Array of relationship objects
  - `weight_vector`: Multi-dimensional weight array
  - `aggregated_weight`: Single aggregated value
  - `metadata`: Edge metadata

### 3. Relationship Types

```python
class RelationType(Enum):
    CO_LOCATION = "co_location"      # Same directory
    SEQUENTIAL = "sequential"        # Sequential chunks
    IMPORT = "import"               # Code dependencies
    SEMANTIC = "semantic"           # Content similarity
    STRUCTURAL = "structural"       # AST-based
    TEMPORAL = "temporal"          # Time-based
    REFERENCE = "reference"        # Documentation links
```

### 4. Processing Pipeline

```
Input Documents
    ↓
Document Processing (Parallel)
    ↓
Chunking (Batch)
    ↓
Embedding Generation (GPU/Batch)
    ↓
Relationship Detection
    ↓
Supra-Weight Calculation
    ↓
Density Control
    ↓
Batch Write to ArangoDB
    ↓
ISNE-Ready Graph
```

### 5. Key Algorithms

#### Density Control Algorithm
```python
def should_add_edge(node_a, node_b, current_degree, max_degree=50):
    # Limit edges per node
    if current_degree[node_a] >= max_degree or current_degree[node_b] >= max_degree:
        return False
    
    # Keep only significant relationships
    if aggregated_weight < min_threshold:
        return False
        
    return True
```

#### Batch Similarity Computation
```python
def compute_similarities_batch(embeddings, batch_size=1000):
    # Process in batches to handle large datasets
    n = len(embeddings)
    for i in range(0, n, batch_size):
        batch_a = embeddings[i:i+batch_size]
        for j in range(i, n, batch_size):
            batch_b = embeddings[j:j+batch_size]
            similarities = compute_batch_similarity(batch_a, batch_b)
            yield from filter_significant_similarities(similarities)
```

## Implementation Plan

### Phase 1: Core Infrastructure
1. Create supra-weight calculator
2. Implement relationship detector
3. Setup ArangoDB schema

### Phase 2: Processing Components  
1. Batch document processor
2. Efficient chunk processor
3. GPU-enabled embedding processor

### Phase 3: Pipeline Integration
1. Main bootstrap pipeline
2. Progress tracking
3. Error recovery

### Phase 4: Optimization
1. Parallel processing
2. Memory optimization
3. Performance tuning

## Configuration

```yaml
bootstrap:
  # Processing
  batch_size: 1000
  chunk_size: 512
  chunk_overlap: 50
  
  # Embeddings
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  embedding_batch_size: 64
  use_gpu: true
  
  # Graph Construction
  max_edges_per_node: 50
  min_edge_weight: 0.3
  similarity_threshold: 0.7
  
  # Storage
  database_name: "isne_graph"
  write_batch_size: 5000
```

## Performance Targets

- Process 100k documents in < 1 hour
- Handle graphs with 1M+ nodes
- Memory usage < 16GB
- Support incremental updates

## Error Handling

1. **Checkpointing**: Save progress every N batches
2. **Recovery**: Resume from last checkpoint
3. **Validation**: Verify graph integrity
4. **Monitoring**: Track metrics and alerts

## Future Extensions

1. **Streaming Support**: True streaming when available
2. **Dynamic Relationships**: Learn relationship importance
3. **Multi-Modal**: Handle images, tables, etc.
4. **Distributed**: Multi-machine processing