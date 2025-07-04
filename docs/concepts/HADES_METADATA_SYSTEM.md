# HADES Metadata System

## Overview

HADES uses a hidden `.hades/` directory in each code directory to store rich metadata that enhances the knowledge graph. This metadata is not for daily user reference but for the system to understand relationships, purpose, and context.

## Directory Structure

```
any_directory/
├── .hades/
│   ├── metadata.json         # Primary metadata file
│   ├── relationships.json    # Explicit relationship mappings
│   ├── keywords.json        # Pre-computed keywords and bridges
│   └── embeddings.cache     # Cached embeddings (optional)
├── actual_code.py
└── research_paper.pdf
```

## Metadata Schema

### metadata.json
```json
{
  "directory": {
    "path": "/src/pathrag",
    "purpose": "Path-based retrieval implementation",
    "domain": "retrieval_algorithms",
    "created": "2024-01-15T10:00:00Z",
    "last_processed": "2024-01-20T15:30:00Z"
  },
  "contents": {
    "implementation_files": [
      {
        "name": "pathrag_rag_strategy.py",
        "type": "core_implementation",
        "implements": "PathRAG algorithm from paper",
        "status": "complete"
      }
    ],
    "research_files": [
      {
        "name": "PathRAG_paper.pdf",
        "type": "research_paper",
        "concepts": ["graph traversal", "path pruning", "retrieval"]
      }
    ]
  },
  "context": {
    "module_role": "Implements query-as-graph retrieval strategy",
    "key_innovations": [
      "Queries become temporary graph nodes",
      "Multi-dimensional supra-weights",
      "Path-based scoring"
    ]
  }
}
```

### relationships.json
```json
{
  "explicit_relationships": [
    {
      "target": "/src/config/pathrag/config.yaml",
      "type": "configured_by",
      "strength": 1.0
    },
    {
      "target": "/docs/api/pathrag_api.md",
      "type": "documented_in",
      "strength": 0.9
    },
    {
      "target": "/src/storage/arangodb",
      "type": "depends_on",
      "strength": 0.8
    },
    {
      "target": "/tests/unit/pathrag",
      "type": "tested_by",
      "strength": 0.9
    }
  ],
  "conceptual_bridges": [
    {
      "concept": "graph_traversal",
      "related_paths": [
        "/docs/concepts/CORE_CONCEPTS.md#queries-as-graph-objects",
        "/src/isne/models/directory_aware_isne.py"
      ]
    }
  ]
}
```

### keywords.json
```json
{
  "directory_keywords": [
    "pathrag",
    "retrieval",
    "graph traversal",
    "query processing"
  ],
  "theory_keywords": [
    "path pruning",
    "flow-based algorithm",
    "reliability scoring"
  ],
  "practice_keywords": [
    "ArangoDB integration",
    "supra-weight calculation",
    "async implementation"
  ],
  "bridge_keywords": [
    {
      "term": "graph_traversal",
      "theory_context": "algorithm for finding paths",
      "practice_context": "NetworkX implementation",
      "strength": 0.95
    }
  ]
}
```

## Generation Process

### Phase 1: Manual Bootstrap
We manually create `.hades/` directories for the core modules to demonstrate the pattern:
1. `/src/jina_v4/.hades/`
2. `/src/isne/.hades/`
3. `/src/pathrag/.hades/`
4. `/src/storage/.hades/`

### Phase 2: Jina v4 Learning
When Jina v4 processes these directories, it will:
1. Learn the metadata pattern
2. Extract similar relationships from code analysis
3. Generate metadata for new directories

### Phase 3: Self-Improvement
Jina v4 can then:
1. Process its own codebase
2. Generate metadata for all directories
3. Identify missing relationships
4. Suggest improvements

## Benefits for Processing

### 1. Rich Context for Chunking
```python
# When processing a file
chunk_context = {
    "file_context": file_docstring,
    "directory_context": metadata["context"],
    "relationships": relationships["explicit_relationships"],
    "domain": metadata["directory"]["domain"]
}
```

### 2. Enhanced Keyword Extraction
- Pre-computed keywords guide attention
- Bridge keywords connect theory and practice
- Domain context improves extraction quality

### 3. Better Graph Construction
- Explicit relationships beyond co-location
- Weighted edges based on relationship strength
- Conceptual bridges across distant nodes

### 4. Improved Query Routing
```python
# Query: "How does PathRAG handle graph traversal?"
# System can use metadata to quickly identify:
1. /src/pathrag/ (implementation - from domain)
2. PathRAG_paper.pdf (theory - from research_files)
3. Related implementations (from conceptual_bridges)
```

## Self-Documenting Behavior

When HADES processes a new codebase:
1. Analyzes directory structure
2. Extracts purpose from code patterns
3. Identifies relationships from imports
4. Generates `.hades/metadata.json`
5. Updates as code evolves

## Example: Processing HADES Itself

```python
# First run: Process HADES codebase
results = hades.process_directory("/HADES", 
    generate_metadata=True,
    learn_patterns=True
)

# System learns:
# - How we organize code (research + implementation)
# - Metadata patterns we use
# - Relationship types we value

# Future runs: Apply learned patterns
results = hades.process_directory("/new-project",
    use_learned_patterns=True
)
```

## Implementation in Jina v4

The `_parse_document` method should:
1. Check for `.hades/` directory
2. Load existing metadata if present
3. Use metadata to enhance processing
4. Generate metadata for directories without it
5. Update metadata with new insights

This creates a self-improving system that gets better at understanding code organization with each codebase it processes.