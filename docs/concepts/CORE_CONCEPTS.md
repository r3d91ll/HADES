# Core Concepts

## Theory-Practice Bridges

### Overview
HADES automatically connects theoretical knowledge (research papers, documentation) with practical implementations (code, configurations) through intelligent keyword extraction and semantic understanding.

### How It Works

1. **Contextual Keyword Extraction**
   - Jina v4 uses attention mechanisms to identify important concepts
   - Keywords are extracted at both document and chunk levels
   - Different extraction strategies for different document types

2. **Bridge Building**
   ```
   Research Paper: "transformer architecture", "self-attention", "positional encoding"
                                    ↕ (bridge)
   Implementation: "class TransformerBlock", "attention_weights", "pos_encoding"
   ```

3. **Multi-Level Connections**
   - Document-level: Overall concepts and themes
   - Chunk-level: Specific implementations and details
   - Cross-document: Relationships between theory and practice

### Example
When processing a research paper about "Graph Neural Networks" and code implementing GNNs:
- Paper keywords: ["graph neural network", "message passing", "node embeddings"]
- Code keywords: ["GNNLayer", "aggregate_messages", "node_embed"]
- Bridge keywords: ["graph", "neural", "embedding", "message"]

These bridges allow queries about concepts to find both theoretical explanations and practical implementations.

## Queries as Graph Objects

### Overview
Instead of treating queries as external requests, HADES transforms them into temporary nodes within the knowledge graph, enabling sophisticated graph-based retrieval.

### Process

1. **Query Embedding**
   ```python
   query = "How to implement attention mechanism?"
   query_node = {
       'type': 'query',
       'embedding': jina.embed(query),
       'keywords': jina.extract_keywords(query),
       'timestamp': now()
   }
   ```

2. **Graph Placement**
   - ISNE places the query node in the graph space
   - Position determined by semantic similarity to existing nodes
   - Temporary edges created based on embedding distances

3. **Path Finding**
   - Find shortest paths from query to relevant knowledge
   - Use supra-weights to rank multi-dimensional relationships
   - Return not just answers but full context paths

### Benefits
- **Contextual Understanding**: See how query relates to entire knowledge base
- **Better Ranking**: Graph distance provides additional relevance signal
- **Exploration**: Can follow paths to discover related information
- **Learning**: System learns from query patterns over time

## Filesystem Taxonomy as Semantic Information

### Overview
The Unix filesystem hierarchy provides rich semantic information that HADES leverages for automatic relationship inference.

### Hierarchical Relationships

```
/project/
├── src/
│   ├── models/          (all files here are "model-related")
│   │   ├── transformer.py
│   │   └── attention.py (co-located → related)
│   └── utils/          (different purpose than models/)
│       └── data.py
└── docs/
    └── architecture.md  (documentation for src/)
```

### Automatic Inferences

1. **Co-location**: Files in same directory are semantically related
2. **Hierarchy**: Parent directories provide context
3. **Depth**: Deeper paths indicate specialization
4. **Naming**: Directory names carry semantic meaning

### ISNE Training Benefits

When ISNE trains on filesystem-aware embeddings:
- Learns directory-based clustering
- New files automatically get appropriate relationships
- Can predict likely locations for new content
- Understands project structure patterns

### Example Applications

1. **Adding New File**
   ```
   New: /project/src/models/bert.py
   Automatically related to: transformer.py, attention.py
   Inherits context: "model implementation"
   ```

2. **Cross-Directory Relationships**
   ```
   /src/models/transformer.py imports from /src/utils/data.py
   Creates bridge between "models" and "utils" contexts
   ```

3. **Query Routing**
   ```
   Query: "transformer model implementation"
   Filesystem hint: Likely in /src/models/
   Prioritizes search in model-related directories
   ```

## Supra-Weights

### Overview
Multi-dimensional edge weights that capture different types of relationships simultaneously.

### Dimensions

1. **Semantic Similarity**: Content-based similarity (0-1)
2. **Co-location**: Same directory bonus (0-1)
3. **Structural**: Import/dependency relationships (0-1)
4. **Temporal**: Time-based relationships (0-1)
5. **Keyword Bridge**: Theory-practice connection strength (0-1)

### Aggregation
```python
supra_weight = {
    'dimensions': [0.9, 0.8, 0.0, 0.5, 0.7],
    'aggregated': weighted_sum(dimensions, learned_weights)
}
```

### Usage in Retrieval
- Different queries emphasize different dimensions
- ISNE learns optimal dimension weights
- Enables nuanced relationship ranking