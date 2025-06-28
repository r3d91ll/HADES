# Multi-Engine RAG Platform

A comprehensive platform for comparing different RAG strategies and running controlled experiments with multiple retrieval engines.

## Overview

The Multi-Engine RAG Platform allows you to:

- **Run multiple RAG engines side-by-side** with consistent interfaces
- **Perform A/B testing** between different retrieval strategies  
- **Compare performance metrics** across engines (speed, relevance, diversity)
- **Manage controlled experiments** with detailed analytics
- **Configure engines independently** for optimal performance
- **Scale horizontally** by adding new engines without code changes

## Architecture

### Core Components

1. **Base Engine Framework** (`base.py`)
   - Abstract interfaces and protocols
   - Common request/response models
   - Engine registry for management
   - Standardized configuration system

2. **PathRAG Engine** (`pathrag_engine.py`)
   - Full implementation of PathRAG algorithms
   - Flow-based path pruning
   - Multi-hop reasoning capabilities
   - Resource allocation algorithms

3. **Comparison Framework** (`comparison.py`)
   - A/B testing experiment management
   - Side-by-side engine comparisons
   - Comprehensive metrics calculation
   - Statistical analysis and recommendations

4. **FastAPI Router** (`router.py`)
   - RESTful API endpoints
   - Engine management interfaces
   - Experiment control endpoints
   - Real-time status monitoring

## Quick Start

### 1. Basic Engine Usage

```python
from src.api.engines import get_engine_registry, EngineType, RetrievalMode

# Get engine registry
registry = get_engine_registry()

# Get PathRAG engine
pathrag_engine = registry.get_engine(EngineType.PATHRAG)

# Create retrieval request
request = EngineRetrievalRequest(
    query="What is machine learning?",
    mode=RetrievalMode.FULL_ENGINE,
    top_k=10
)

# Execute retrieval
response = await pathrag_engine.retrieve(request)

# Process results
for result in response.results:
    print(f"Score: {result.score}")
    print(f"Content: {result.content}")
```

### 2. Engine Comparison

```python
from src.api.engines.comparison import get_experiment_manager, ComparisonMetric

# Get experiment manager
manager = get_experiment_manager()

# Run comparison
comparison = await manager.run_comparison(
    query="Explain neural networks",
    engines=[EngineType.PATHRAG, EngineType.SAGEGRAPH],
    metrics=[ComparisonMetric.RELEVANCE, ComparisonMetric.SPEED]
)

# Analyze results
print(f"Winner: {comparison.winner}")
print(f"Metrics: {comparison.metrics}")
```

### 3. A/B Testing Experiment

```python
from src.api.engines.base import ExperimentRequest

# Create experiment
experiment_request = ExperimentRequest(
    name="PathRAG vs SageGraph Comparison",
    description="Compare retrieval quality across ML queries",
    engines=[EngineType.PATHRAG, EngineType.SAGEGRAPH],
    test_queries=[
        "What are the benefits of neural networks?",
        "How do transformers work?",
        "Explain backpropagation algorithm"
    ],
    metrics=["relevance", "speed", "diversity"],
    iterations=3
)

# Start experiment
experiment_id = await manager.start_experiment(experiment_request)

# Monitor progress
experiment = await manager.get_experiment(experiment_id)
print(f"Status: {experiment.status}")
```

## API Endpoints

### Engine Management

- `GET /engines/` - List all available engines
- `GET /engines/{engine_type}/status` - Get engine health status
- `POST /engines/{engine_type}/configure` - Update engine configuration
- `GET /engines/{engine_type}/config` - Get current configuration

### Retrieval Operations

- `POST /engines/{engine_type}/retrieve` - Execute retrieval with specific engine
- `POST /engines/compare` - Compare multiple engines on single query

### Experiment Management

- `POST /engines/experiments` - Start new A/B testing experiment
- `GET /engines/experiments/{experiment_id}` - Get experiment results
- `GET /engines/experiments` - List all experiments
- `DELETE /engines/experiments/{experiment_id}` - Cancel running experiment

### PathRAG-Specific

- `POST /engines/pathrag/analyze-paths` - Analyze PathRAG path construction
- `GET /engines/pathrag/knowledge-base` - Get knowledge base statistics

## Supported Engines

### PathRAG Engine

**Features:**
- Flow-based resource allocation
- Multi-hop path reasoning (1-hop, 2-hop, 3-hop)
- Hierarchical keyword extraction
- Hybrid retrieval strategies
- Graph-aware embeddings (ISNE integration)

**Modes:**
- `naive` - Simple similarity search
- `local` - Entity-focused retrieval
- `global` - Relationship-focused retrieval  
- `hybrid` - Combined local + global
- `full_engine` - Complete PathRAG algorithm

**Configuration:**
```yaml
flow_decay_factor: 0.8        # Resource flow decay (0.0-1.0)
pruning_threshold: 0.3        # Node pruning threshold (0.0-1.0)
max_path_length: 5            # Maximum path exploration length
max_iterations: 10            # Resource allocation iterations

multihop:
  enable_1hop: true           # Enable 1-hop exploration
  enable_2hop: true           # Enable 2-hop exploration  
  enable_3hop: true           # Enable 3-hop exploration

context:
  weights:
    entity_context: 0.4       # Weight for entity context
    relationship_context: 0.4 # Weight for relationship context
    source_text_context: 0.2  # Weight for source text context
```

### Future Engines

**SageGraph Engine** (Planned)
- Graph attention mechanisms
- Hierarchical semantic clustering
- Dynamic graph construction
- Multi-modal reasoning

**Hybrid Engine** (Planned)
- Combines multiple engines
- Weighted result merging
- Adaptive strategy selection

## Comparison Metrics

### Available Metrics

1. **Relevance** - Average relevance scores of results
2. **Speed** - Execution time performance  
3. **Diversity** - Unique source coverage
4. **Coverage** - Number of results returned
5. **Consistency** - Score variance across results
6. **Resource Usage** - Memory and CPU utilization

### Metric Calculation

```python
# Relevance: Average of result scores
relevance = sum(result.score for result in results) / len(results)

# Speed: Inverted execution time (faster = higher score)
speed = 1.0 - (execution_time / max_execution_time)

# Diversity: Unique sources ratio
diversity = unique_sources_count / total_results_count

# Coverage: Results ratio compared to maximum
coverage = results_count / max_results_count

# Consistency: Inverse of score variance
consistency = 1.0 - min(score_variance, 1.0)
```

## Experiment Management

### Experiment Lifecycle

1. **Planning** - Define experiment parameters
2. **Execution** - Run queries across engines
3. **Collection** - Gather metrics and results
4. **Analysis** - Calculate comparisons and statistics
5. **Reporting** - Generate recommendations

### Experiment Configuration

```python
experiment = ExperimentRequest(
    name="Performance Comparison",
    description="Compare engines on technical queries",
    engines=[EngineType.PATHRAG, EngineType.SAGEGRAPH],
    test_queries=load_test_queries("technical_domain.json"),
    engine_configs={
        EngineType.PATHRAG: {"flow_decay_factor": 0.8},
        EngineType.SAGEGRAPH: {"attention_heads": 8}
    },
    metrics=["relevance", "speed", "diversity"],
    iterations=5  # Run each query 5 times for statistical significance
)
```

### Results Analysis

```python
# Get experiment results
experiment = await manager.get_experiment(experiment_id)

# Summary metrics per engine
for engine_type, stats in experiment.summary_metrics.items():
    print(f"\n{engine_type}:")
    print(f"  Avg Execution Time: {stats['avg_execution_time_ms']:.2f}ms")
    print(f"  Avg Relevance: {stats['avg_relevance_score']:.3f}")
    print(f"  Success Rate: {stats['success_rate']:.1%}")

# Overall recommendations
for recommendation in experiment.recommendations:
    print(f"💡 {recommendation}")
```

## Configuration Management

### Engine Configuration

Each engine supports dynamic configuration updates:

```python
# Update PathRAG configuration
await pathrag_engine.configure({
    "flow_decay_factor": 0.9,
    "pruning_threshold": 0.2,
    "multihop": {
        "enable_3hop": False  # Disable 3-hop for speed
    }
})

# Get current configuration
config = await pathrag_engine.get_config()

# Get configuration schema
schema = await pathrag_engine.get_config_schema()
```

### Configuration Schemas

All engines provide JSON schemas for validation:

```json
{
  "type": "object",
  "properties": {
    "flow_decay_factor": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Resource flow decay factor"
    }
  },
  "required": ["flow_decay_factor"]
}
```

## Monitoring and Metrics

### Engine Health Monitoring

```python
# Check individual engine health
is_healthy = await pathrag_engine.health_check()

# Check all engines
health_results = await registry.health_check_all()

# Get detailed metrics
metrics = await pathrag_engine.get_metrics()
```

### Performance Metrics

Each engine tracks:
- Total queries processed
- Average execution time
- Error counts and rates
- Last query timestamp
- Engine-specific metrics

### PathRAG-Specific Metrics

```python
# Get PathRAG metrics
pathrag_metrics = await pathrag_engine.get_metrics()

print(f"Knowledge Graph Nodes: {pathrag_metrics['knowledge_graph_nodes']}")
print(f"Knowledge Graph Edges: {pathrag_metrics['knowledge_graph_edges']}")
print(f"Total Paths Explored: {pathrag_metrics['total_paths_explored']}")
print(f"Average Path Length: {pathrag_metrics['avg_path_length']}")
```

## Advanced Features

### Path Analysis (PathRAG)

Analyze how PathRAG constructs and scores paths:

```python
# Analyze path construction
analysis = await pathrag_engine.analyze_paths(
    query="machine learning algorithms",
    top_k=5
)

# Examine path details
for path_detail in analysis["path_details"]:
    print(f"Path: {' -> '.join(path_detail['path_nodes'])}")
    print(f"Score: {path_detail['path_score']}")
    print(f"Narrative: {path_detail['narrative']}")
```

### Knowledge Base Statistics

Get detailed knowledge base information:

```python
# Get knowledge base stats
kb_stats = await pathrag_engine.get_knowledge_base_stats()

print(f"Initialized: {kb_stats['initialized']}")
print(f"Total Nodes: {kb_stats['total_nodes']}")
print(f"Total Edges: {kb_stats['total_edges']}")
print(f"Average Degree: {kb_stats['average_degree']}")
```

### Custom Metrics

Add custom metrics to experiments:

```python
class CustomMetric(ComparisonMetric):
    DOMAIN_RELEVANCE = "domain_relevance"

# Implement custom metric calculation
async def calculate_domain_relevance(responses):
    # Custom logic for domain-specific relevance
    pass
```

## Best Practices

### Engine Selection

1. **Speed-Critical Applications**
   - Use `naive` mode for fastest results
   - Configure lower `max_path_length` for PathRAG
   - Monitor execution time metrics

2. **Quality-Critical Applications**  
   - Use `full_engine` mode for best results
   - Enable all multi-hop modes
   - Increase `top_k` for broader coverage

3. **Balanced Applications**
   - Use `hybrid` mode
   - Run A/B tests to find optimal configuration
   - Monitor both speed and relevance metrics

### Experiment Design

1. **Query Selection**
   - Use representative queries from your domain
   - Include variety in complexity and topics
   - Test edge cases and difficult queries

2. **Statistical Significance**
   - Run multiple iterations (≥5) per query
   - Use sufficient query volume (≥20 queries)
   - Account for variance in results

3. **Configuration Testing**
   - Test different engine configurations
   - Isolate single parameter changes
   - Document configuration rationale

### Performance Optimization

1. **PathRAG Tuning**
   ```python
   # For speed optimization
   speed_config = {
       "flow_decay_factor": 0.7,      # Faster convergence
       "pruning_threshold": 0.4,      # More aggressive pruning
       "max_path_length": 3,          # Shorter paths
       "max_iterations": 5            # Fewer iterations
   }
   
   # For quality optimization  
   quality_config = {
       "flow_decay_factor": 0.9,      # Better resource preservation
       "pruning_threshold": 0.2,      # Less aggressive pruning
       "max_path_length": 7,          # Longer paths
       "max_iterations": 15           # More iterations
   }
   ```

2. **Caching Strategies**
   - Enable embedding caching for repeated queries
   - Cache graph computations for static knowledge bases
   - Use query result caching for identical requests

3. **Resource Management**
   - Monitor memory usage during experiments
   - Configure appropriate timeouts
   - Scale compute resources based on engine requirements

## Troubleshooting

### Common Issues

1. **Engine Initialization Failures**
   ```python
   # Check engine health
   is_healthy = await engine.health_check()
   if not is_healthy:
       # Check logs and dependencies
       pass
   ```

2. **Poor Retrieval Quality**
   - Verify knowledge base content and structure
   - Check embedding quality and coverage
   - Adjust configuration parameters
   - Compare with baseline metrics

3. **Slow Performance**
   - Profile execution time by component
   - Reduce `max_path_length` or `top_k`
   - Enable caching mechanisms
   - Check resource utilization

### Debugging Tools

1. **Path Analysis** (PathRAG)
   ```python
   # Debug path construction
   analysis = await pathrag_engine.analyze_paths(query, top_k=10)
   print(f"Resource Allocation: {analysis['resource_allocation']}")
   print(f"Keywords: {analysis['keyword_extraction']}")
   ```

2. **Metrics Monitoring**
   ```python
   # Monitor engine performance
   metrics = await engine.get_metrics()
   if metrics['error_count'] > 0:
       print(f"Errors detected: {metrics['error_count']}")
   ```

3. **Experiment Analysis**
   ```python
   # Analyze experiment failures
   experiment = await manager.get_experiment(experiment_id)
   failed_results = [r for r in experiment.results if 'error' in r.metadata]
   ```

## Future Roadmap

### Planned Features

1. **Additional Engines**
   - SageGraph implementation
   - Hybrid multi-engine strategy
   - External API engine wrappers

2. **Enhanced Metrics**
   - Semantic similarity measures
   - Domain-specific relevance scoring
   - User feedback integration

3. **Advanced Experimentation**
   - Statistical significance testing
   - Bayesian optimization for configuration
   - Multi-objective optimization

4. **Operational Features**
   - Engine load balancing
   - Automatic failover mechanisms
   - Real-time performance dashboards

### Contributing

To add a new engine:

1. Implement the `RAGEngine` protocol
2. Extend `BaseRAGEngine` for common functionality
3. Register the engine in the router startup
4. Add comprehensive tests
5. Update documentation

Example new engine structure:
```python
class MyNewEngine(BaseRAGEngine):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(EngineType.MY_ENGINE, config)
    
    @property
    def supported_modes(self) -> List[RetrievalMode]:
        return [RetrievalMode.FULL_ENGINE]
    
    async def _execute_retrieval(self, request: EngineRetrievalRequest) -> List[EngineRetrievalResult]:
        # Implement retrieval logic
        pass
```

The Multi-Engine RAG Platform provides a robust foundation for comparing and optimizing retrieval strategies, enabling data-driven decisions about which engines work best for your specific use cases.