# ISNE Bootstrap Architecture

## Overview

This document describes the bootstrap pipeline for HADES that follows the vanilla ISNE (Inductive Shallow Node Embedding) approach from the research paper, using filesystem structure as the primary graph with `.hades/` metadata directories as signals for indexing.

## Core Principles

1. **Physical Reality First**: Filesystem paths anchor all nodes to actual disk locations
2. **Vanilla ISNE**: Simple neighborhood aggregation: h(v) = average(θ of neighbors)
3. **`.hades/` Signaling**: Only directories with `.hades/` metadata are included
4. **Single Node Collection**: One node exists with multiple embeddings as properties
5. **Multiple Edge Collections**: Different relationship types in separate collections
6. **Query-Time Supra-Weights**: Multi-dimensional weights calculated dynamically, not stored

## Architecture Components

### 1. `.hades/` Metadata Structure

Each directory that participates in the RAG system must contain:

```dir
any_directory/
├── .hades/                    # Signals directory is indexed
│   ├── metadata.json         # Directory purpose and context
│   ├── relationships.json    # Explicit connections to other dirs
│   ├── keywords.json        # Pre-computed keywords
│   └── index_status.json    # Indexing status and version
├── source_files.py
└── README.md
```

### 2. Simplified Component Structure

```dir
ISNEBootstrap/
├── scanner/
│   ├── hades_directory_scanner.py    # Find .hades/ directories
│   └── metadata_loader.py            # Load .hades/ metadata
├── graph/
│   ├── node_creator.py               # Create nodes from files/dirs
│   ├── edge_creator.py               # Simple filesystem edges
│   └── relationship_builder.py       # Process relationships.json
├── isne/
│   ├── vanilla_trainer.py            # Pure ISNE implementation
│   ├── random_walk.py                # Generate training walks
│   └── skip_gram.py                  # Skip-gram objective
└── storage/
    ├── arango_client.py              # ArangoDB operations
    └── batch_writer.py               # Efficient writes
```

### 3. Database Schema

#### Node Collection (Single)

```javascript
{
  "_key": "src_pathrag_pathrag_rag_strategy_py",
  "_id": "nodes/src_pathrag_pathrag_rag_strategy_py",
  
  // Physical anchor
  "filesystem_path": "/src/pathrag/pathrag_rag_strategy.py",
  "type": "file|directory|query",
  "has_hades": true,  // For directories with .hades/
  
  // Multiple embeddings as properties
  "embeddings": {
    "jina_v4": [...],        // Semantic understanding
    "isne": [...]            // Graph structure position
  },
  
  // Metadata from .hades/ or file analysis
  "metadata": {
    "purpose": "PathRAG implementation",
    "domain": "retrieval_algorithms",
    "size": 15234,
    "modified": "2024-01-15T10:30:00Z"
  }
}
```

#### Edge Collections (Multiple)

- `filesystem_edges`: Parent-child, sibling relationships (ISNE primary)
- `semantic_edges`: Based on embedding similarity
- `functional_edges`: Import/call dependencies
- `temporal_edges`: Co-change patterns
- `query_edges`: Temporary edges from query nodes

## Bootstrap Pipeline

### Phase 1: Directory Scanning

```python
def scan_for_hades_directories(root_path):
    """Only index directories with .hades/ metadata"""
    for root, dirs, files in os.walk(root_path):
        if '.hades' in dirs:
            # Load metadata
            metadata = load_json(f"{root}/.hades/metadata.json")
            relationships = load_json(f"{root}/.hades/relationships.json")
            
            # Create directory node
            yield {
                'path': root,
                'metadata': metadata,
                'relationships': relationships,
                'has_hades': True
            }
        else:
            # Skip subdirectories without .hades/
            dirs[:] = []
```

### Phase 2: Graph Construction

```text
.hades/ Directories Found
    ↓
Create Directory Nodes
    ↓
Create File Nodes (within .hades/ dirs)
    ↓
Build Filesystem Edges (parent-child, sibling)
    ↓
Add Explicit Relationships (from .hades/relationships.json)
    ↓
Generate Jina v4 Embeddings
    ↓
Store Nodes with Embeddings as Properties
```

### Phase 3: Vanilla ISNE Training

```python
class VanillaISNE:
    """Pure ISNE implementation following the research paper"""
    
    def __init__(self, num_nodes, embedding_dim):
        # Core ISNE: learnable parameters θ for each node
        self.theta = nn.Parameter(torch.randn(num_nodes, embedding_dim))
    
    def get_embedding(self, node_id, neighbors):
        """h(v) = (1/|N_v|) * Σ_{n∈N_v} θ_n"""
        if not neighbors:
            return self.theta[node_id]
        
        neighbor_params = self.theta[neighbors]
        return neighbor_params.mean(dim=0)
    
    def train_skip_gram(self, walks):
        """Train using skip-gram objective on random walks"""
        for walk in walks:
            for i, center in enumerate(walk):
                # Context nodes within window
                context = walk[max(0, i-window):i] + walk[i+1:i+window+1]
                
                # Get embeddings
                center_emb = self.get_embedding(center, get_neighbors(center))
                
                # Skip-gram loss
                positive_loss = -log_sigmoid(center_emb @ context_embeddings)
                negative_loss = -log_sigmoid(-center_emb @ negative_samples)
```

### Phase 4: Query-as-Node Implementation

```python
def create_query_node(query_text, session_id):
    """Create temporary query node in VFS and graph"""
    
    # 1. Create in VFS mount
    query_path = f"/mnt/hades_queries/session_{session_id}/query_{timestamp}.json"
    vfs.write(query_path, {
        "query": query_text,
        "timestamp": now(),
        "ttl": 3600  # 1 hour
    })
    
    # 2. Add to graph
    query_node = {
        "_key": f"query_{session_id}_{timestamp}",
        "filesystem_path": query_path,  # Physical anchor!
        "type": "query",
        "embeddings": {
            "jina_v4": jina.embed(query_text)
        }
    }
    db.nodes.insert(query_node)
    
    # 3. Connect to relevant nodes (creates query_edges)
    connect_query_to_graph(query_node)
```

### Phase 5: Supra-Weights at Query Time

```python
def calculate_supra_weight(query_node, target_node):
    """Calculate multi-dimensional weights dynamically"""
    
    weights = {}
    
    # Check each edge collection
    for collection in ['filesystem_edges', 'semantic_edges', 
                      'functional_edges', 'temporal_edges']:
        edge = db.find_edge(query_node['_id'], target_node['_id'], collection)
        
        if edge:
            if collection == 'filesystem_edges':
                # Inverse path distance
                weights['filesystem'] = 1.0 / (1 + edge['distance'])
            
            elif collection == 'semantic_edges':
                # Jina embedding similarity
                weights['semantic'] = cosine_similarity(
                    query_node['embeddings']['jina_v4'],
                    target_node['embeddings']['jina_v4']
                )
            
            elif collection == 'functional_edges':
                # Code relationship strength
                weights['functional'] = edge['strength']
            
            elif collection == 'temporal_edges':
                # Recency factor
                days_ago = (now() - edge['last_modified']).days
                weights['temporal'] = exp(-days_ago / 30)
    
    # Aggregate (configurable weights)
    return sum(weights.values()) / len(weights) if weights else 0.0
```

## `.hades/` Metadata Schemas

### metadata.json

```json
{
  "directory": {
    "path": "/src/pathrag",
    "purpose": "PathRAG implementation for graph-based retrieval",
    "domain": "retrieval_algorithms",
    "indexed_date": "2024-01-15T10:00:00Z"
  },
  "indexing_hints": {
    "importance": "high",
    "chunk_strategy": "ast_based",
    "update_frequency": "weekly"
  }
}
```

### relationships.json

```json
{
  "explicit_relationships": [
    {
      "target": "/src/storage/arangodb",
      "type": "depends_on",
      "strength": 0.9
    },
    {
      "target": "/docs/api/pathrag",
      "type": "documented_by",
      "strength": 0.8
    }
  ]
}
```

## Key Benefits

1. **Physical Grounding**: Every node anchored to filesystem reality
2. **Selective Indexing**: Only `.hades/` directories participate
3. **Simple ISNE**: Follows research paper exactly
4. **Query Flexibility**: Supra-weights calculated at query time
5. **Clean Architecture**: Clear separation of concerns

## Performance Characteristics

- **Bootstrap**: One-time graph construction from `.hades/` directories
- **ISNE Training**: Lightweight random walk on filesystem edges
- **Query Time**: Dynamic weight calculation enables context-aware retrieval
- **Storage**: Single node collection with property-based embeddings
