# PathRAG Processor

Modern implementation of PathRAG (Path-based Retrieval Augmented Generation) for the HADES system.

## Overview

PathRAG is a graph-based RAG approach that retrieves key relational paths rather than flat text chunks. It uses a flow-based resource allocation algorithm to identify the most relevant paths between nodes in a knowledge graph.

## Core Algorithms Implemented

### 1. Flow-based Resource Allocation
- **Resource Propagation**: Resources flow through graph with decay factors
- **Threshold-based Pruning**: Eliminates weak connections using configurable thresholds
- **Convergence Detection**: Stops iteration when resource flow stabilizes

### 2. Multi-hop Path Reasoning
- **Path Construction**: Builds 1-hop, 2-hop, and 3-hop paths between entities
- **Narrative Generation**: Creates human-readable descriptions of path relationships
- **Weighted BFS Scoring**: Scores paths using graph structure and edge weights

### 3. Hierarchical Keyword Extraction
- **Dual-level Keywords**: Extracts high-level (conceptual) and low-level (specific) keywords
- **LLM Integration**: Uses language models for sophisticated keyword extraction
- **Fallback Strategy**: Simple extraction when LLM unavailable

### 4. Hybrid Retrieval Strategy
- **Local Context**: Entity-based search using low-level keywords
- **Global Context**: Relationship-based search using high-level keywords
- **Context Combination**: Merges local and global contexts with configurable weights

## Supported Retrieval Modes

1. **PathRAG Mode**: Full PathRAG algorithm with flow-based pruning
2. **Hybrid Mode**: Combines local and global retrieval strategies
3. **Local Mode**: Entity-focused retrieval
4. **Global Mode**: Relationship-focused retrieval
5. **Naive Mode**: Simple similarity-based retrieval

## Configuration

PathRAG is configured through YAML files located in `src/config/components/pathrag_config.yaml`.

### Key Configuration Parameters

```yaml
pathrag:
  flow_decay_factor: 0.8          # Resource flow decay (0.0-1.0)
  pruning_threshold: 0.3          # Pruning threshold (0.0-1.0)
  max_path_length: 5              # Maximum path length
  max_iterations: 10              # Resource allocation iterations
  
  multihop:
    enable_1hop: true             # Enable 1-hop exploration
    enable_2hop: true             # Enable 2-hop exploration
    enable_3hop: true             # Enable 3-hop exploration
    
  retrieval:
    default_mode: "hybrid"        # Default retrieval mode
    context_combination_method: "weighted_merge"
```

## Architecture Integration

### HADES Component Integration

PathRAG integrates with the HADES modular architecture:

- **Storage Backend**: ArangoDB for graph data (configurable)
- **Embedding System**: ModernBERT or CPU embeddings (configurable)
- **Graph Enhancement**: ISNE for graph-aware embeddings (configurable)
- **Configuration System**: YAML-based configuration management

### Component Protocols

Implements the `RAGStrategy` protocol with:

```python
def retrieve_and_generate(self, input_data: RAGStrategyInput) -> RAGStrategyOutput
def retrieve_only(self, input_data: RAGStrategyInput) -> RAGStrategyOutput
def get_supported_modes(self) -> List[str]
def supports_mode(self, mode: str) -> bool
```

## Usage Examples

### Basic Usage

```python
from src.components.rag_strategy.pathrag.processor import PathRAGProcessor
from src.types.components.contracts import RAGStrategyInput, RAGMode

# Initialize processor
processor = PathRAGProcessor()
processor.configure({
    "flow_decay_factor": 0.8,
    "pruning_threshold": 0.3,
    "max_path_length": 5,
    "storage": {"type": "arangodb"},
    "embedder": {"type": "modernbert"}
})

# Create query input
input_data = RAGStrategyInput(
    query="Find connections between machine learning and neural networks",
    mode=RAGMode.PATHRAG,
    top_k=10,
    max_token_for_context=4000
)

# Execute retrieval
output = processor.retrieve_and_generate(input_data)

# Access results
for result in output.results:
    print(f"Score: {result.score}")
    print(f"Content: {result.content}")
    if result.path_info:
        print(f"Path: {result.path_info.nodes}")
        print(f"Narrative: {result.path_info.metadata.get('narrative', '')}")
```

### Hybrid Retrieval

```python
# Use hybrid mode for local + global context
input_data = RAGStrategyInput(
    query="Information about organizations and their locations",
    mode=RAGMode.HYBRID,
    top_k=5
)

output = processor.retrieve_and_generate(input_data)
```

## Algorithm Details

### Resource Allocation Flow

1. **Initialization**: Distribute initial resources among query-relevant nodes
2. **Propagation**: Resources flow through graph edges with decay penalty
3. **Convergence**: Iterate until resource flow stabilizes
4. **Pruning**: Remove nodes below threshold for path exploration

### Path Construction Process

1. **Node Pairs**: Select high-resource nodes for path exploration
2. **BFS Search**: Find paths of different hop lengths (1-3 hops)
3. **Narrative Building**: Create human-readable path descriptions
4. **Scoring**: Combine resource flow scores with path characteristics

### Context Building Strategy

1. **Keyword Extraction**: Extract hierarchical keywords from query
2. **Local Search**: Find entities using specific keywords
3. **Global Search**: Find relationships using conceptual keywords
4. **Context Merging**: Combine contexts with configurable weights

## Performance Characteristics

### Computational Complexity
- **Resource Allocation**: O(iterations × edges)
- **Path Finding**: O(nodes² × max_path_length)
- **Overall**: Linear in graph size for typical queries

### Memory Usage
- **Graph Storage**: O(nodes + edges)
- **Resource Tracking**: O(nodes)
- **Path Caching**: O(top_k × max_path_length)

### Optimization Features
- **Early Stopping**: Resource flow below threshold
- **Path Limiting**: Maximum paths per node pair
- **Batch Processing**: Efficient parallel operations
- **Caching**: Embedding and graph operation caching

## Testing

Comprehensive test suite covering:

- **Unit Tests**: Individual algorithm components
- **Integration Tests**: End-to-end retrieval scenarios
- **Performance Tests**: Scalability and timing validations

Run tests with:
```bash
poetry run pytest test/components/rag_strategy/test_pathrag_processor.py -v
```

## Monitoring and Debugging

### Metrics Collection
- **Graph Statistics**: Node/edge counts, connectivity metrics
- **Performance Metrics**: Processing times, cache hit rates
- **Quality Metrics**: Path diversity, relevance scores

### Debug Configuration
```yaml
debug:
  log_resource_flow: true         # Log resource allocation process
  log_path_construction: true     # Log path building details
  save_intermediate_results: true # Save processing intermediates
```

## Future Enhancements

### Planned Features
1. **ArangoDB Native Integration**: Direct graph queries
2. **ISNE Enhancement Integration**: Graph neural network embeddings
3. **Adaptive Thresholds**: Dynamic parameter adjustment
4. **Feedback Learning**: Query result quality feedback
5. **Distributed Processing**: Multi-node graph exploration

### Experimental Features
```yaml
experimental:
  enable_path_caching: true       # Cache frequent paths
  enable_adaptive_thresholds: false # Dynamic threshold adjustment
  enable_feedback_learning: false   # Learn from feedback
```

## Troubleshooting

### Common Issues

1. **Empty Results**: Check pruning threshold and resource allocation
2. **Slow Performance**: Reduce max_iterations or max_path_length
3. **Memory Usage**: Enable caching and limit top_k results
4. **Low Quality**: Adjust flow_decay_factor and context weights

### Debug Steps

1. Enable debug logging in configuration
2. Check graph connectivity and node/edge counts
3. Verify keyword extraction quality
4. Validate resource flow convergence
5. Inspect path construction narratives

## Legacy Algorithm Preservation

This implementation preserves core algorithms from the original PathRAG paper while integrating with modern HADES architecture:

- **Resource allocation with decay penalty** (original algorithm)
- **Multi-hop reasoning with weighted BFS** (enhanced)
- **Hierarchical keyword extraction** (modernized)
- **Context combination strategies** (improved)

The implementation maintains algorithmic fidelity while providing modern infrastructure integration, comprehensive testing, and production-ready features.