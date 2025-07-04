# ArangoDB Schema Documentation

## Overview

HADES uses ArangoDB as its primary storage backend, implementing a multimodal knowledge graph that combines:
- Document storage (raw content and metadata)
- Vector storage (Jina v4 and ISNE embeddings)
- Graph relationships (theory-practice bridges, filesystem hierarchy, semantic connections)

## Collections

### 1. Documents Collection (`hades_documents`)

Stores document metadata and raw content.

```json
{
  "_key": "src_pathrag_PathRAG_py",  // Filesystem path with / replaced by _
  "_id": "hades_documents/src_pathrag_PathRAG_py",
  "path": "/home/user/project/src/pathrag/PathRAG.py",
  "type": "code",  // code|documentation|research|configuration|image|notebook
  "subtype": "python",  // python|markdown|pdf|yaml|json|xml|toml|latex|png|jupyter
  "content": "# Full document content...",
  "metadata": {
    "size": 12345,
    "created": "2024-01-15T10:30:00Z",
    "modified": "2024-01-20T15:45:00Z",
    "hash": "sha256:abcdef...",
    "encoding": "utf-8",
    "mime_type": "text/x-python"
  },
  "hades_metadata": {
    // Content from .hades/metadata.json
    "purpose": "Core PathRAG implementation",
    "keywords": ["pathrag", "retrieval", "graph"],
    "relationships": {...}
  },
  "processing": {
    "processor": "jina_v4",
    "version": "1.0.0",
    "timestamp": "2024-01-20T16:00:00Z",
    "chunks": 5,
    "tokens": 1024,
    "has_multimodal": false,
    "has_latex": false,
    "has_ast": true
  }
}
```

### 2. Chunks Collection (`hades_chunks`)

Stores processed chunks with embeddings.

```json
{
  "_key": "src_pathrag_PathRAG_py_chunk_001",
  "_id": "hades_chunks/src_pathrag_PathRAG_py_chunk_001",
  "document_key": "src_pathrag_PathRAG_py",
  "chunk_index": 1,
  "content": "class PathRAG:\n    '''Core PathRAG implementation...'''",
  "embeddings": {
    "jina_v4": [0.123, -0.456, ...],  // 1024-dim vector
    "isne": [0.789, 0.012, ...],      // Variable-dim based on graph
    "multimodal": null                 // Present for images
  },
  "metadata": {
    "start_char": 0,
    "end_char": 512,
    "start_line": 1,
    "end_line": 25,
    "chunk_type": "code_symbol",      // code_symbol|section|paragraph|equation|figure
    "symbol_info": {
      "name": "PathRAG",
      "type": "class",
      "docstring": "Core PathRAG implementation",
      "methods": ["__init__", "retrieve", "rank_paths"]
    }
  },
  "keywords": {
    "extracted": ["PathRAG", "retrieve", "graph", "traversal"],
    "ast_symbols": ["PathRAG", "retrieve", "rank_paths"],
    "attention_based": ["path", "retrieval", "ranking"]
  },
  "bridges": [
    {
      "type": "implements",
      "target": "research_papers_pathrag_pdf_chunk_003",
      "confidence": 0.9
    }
  ]
}
```

### 3. Embeddings Collection (`hades_embeddings`)

Dedicated collection for efficient vector search.

```json
{
  "_key": "src_pathrag_PathRAG_py_chunk_001_jina_v4",
  "_id": "hades_embeddings/src_pathrag_PathRAG_py_chunk_001_jina_v4",
  "chunk_key": "src_pathrag_PathRAG_py_chunk_001",
  "embedding_type": "jina_v4",
  "vector": [0.123, -0.456, ...],  // 1024-dim
  "metadata": {
    "model": "jinaai/jina-embeddings-v4",
    "lora_adapter": "retrieval",
    "instruction": null,
    "timestamp": "2024-01-20T16:00:00Z"
  }
}
```

### 4. Nodes Collection (`hades_nodes`)

Graph nodes representing conceptual entities.

```json
{
  "_key": "concept_pathrag_algorithm",
  "_id": "hades_nodes/concept_pathrag_algorithm",
  "type": "concept",  // concept|symbol|file|directory|citation|equation
  "name": "PathRAG Algorithm",
  "properties": {
    // Node-specific properties
    "description": "Graph-based retrieval algorithm",
    "first_appearance": "research_papers_pathrag_pdf",
    "implementations": ["src_pathrag_PathRAG_py"],
    "related_concepts": ["graph_traversal", "semantic_search"]
  },
  "embeddings": {
    // Conceptual embeddings
    "concept_embedding": [0.234, 0.567, ...],
    "isne": [0.111, 0.222, ...]
  }
}
```

### 5. Edges Collection (`hades_edges`)

Graph edges with supra-weights.

```json
{
  "_key": "implements_12345",
  "_id": "hades_edges/implements_12345",
  "_from": "hades_nodes/src_pathrag_PathRAG_py",
  "_to": "hades_nodes/concept_pathrag_algorithm",
  "type": "implements",  // implements|references|contains|derives_from|uses
  "properties": {
    "confidence": 0.95,
    "evidence": ["Class name matches", "Algorithm structure matches"],
    "detected_by": ["bridge_detector", "ast_analysis"]
  },
  "supra_weights": {
    // Multi-dimensional weights (calculated at query time)
    "semantic_similarity": null,
    "structural_similarity": null,
    "temporal_proximity": null,
    "co_occurrence": null
  },
  "metadata": {
    "created": "2024-01-20T16:00:00Z",
    "source": "automatic_detection",
    "validated": false
  }
}
```

### 6. Bridges Collection (`hades_bridges`)

Theory-practice bridges between different domains.

```json
{
  "_key": "bridge_pathrag_paper_to_code",
  "_id": "hades_bridges/bridge_pathrag_paper_to_code",
  "type": "algorithm_implementation",
  "source": {
    "collection": "hades_chunks",
    "key": "research_papers_pathrag_pdf_chunk_003",
    "type": "algorithm_description",
    "content_preview": "Algorithm 1: PathRAG traversal..."
  },
  "target": {
    "collection": "hades_chunks", 
    "key": "src_pathrag_PathRAG_py_chunk_001",
    "type": "code_implementation",
    "content_preview": "class PathRAG:..."
  },
  "confidence": 0.92,
  "evidence": {
    "name_similarity": 0.95,
    "structure_match": 0.88,
    "citation_present": true,
    "temporal_alignment": true
  },
  "metadata": {
    "detected_by": "bridge_detector",
    "detection_timestamp": "2024-01-20T16:00:00Z",
    "validation_status": "pending"
  }
}
```

### 7. Images Collection (`hades_images`)

Multimodal image data and metadata.

```json
{
  "_key": "docs_architecture_system_overview_png",
  "_id": "hades_images/docs_architecture_system_overview_png",
  "path": "/home/user/project/docs/architecture/system_overview.png",
  "image_type": "architecture_diagram",
  "image_data": {
    "base64": "iVBORw0KGgoAAAANS...",  // Or reference to object storage
    "dimensions": [1920, 1080],
    "format": "PNG",
    "size_bytes": 245760
  },
  "visual_features": {
    "has_text": true,
    "has_boxes": true,
    "has_arrows": true,
    "dominant_colors": ["#2E86AB", "#FFFFFF", "#000000"],
    "is_monochrome": false
  },
  "extracted_text": "UserService -> AuthService -> Database...",
  "embeddings": {
    "jina_v4_visual": [0.345, 0.678, ...],
    "jina_v4_multimodal": [0.234, 0.567, ...]  // Combined with text
  },
  "detected_entities": [
    {"name": "UserService", "type": "component", "bbox": [100, 200, 300, 400]},
    {"name": "AuthService", "type": "component", "bbox": [500, 200, 700, 400]}
  ],
  "bridges": [
    {
      "type": "depicts",
      "target": "src_services_user_service_py",
      "confidence": 0.85
    }
  ]
}
```

### 8. Queries Collection (`hades_queries`)

Stored queries for analysis and optimization.

```json
{
  "_key": "query_20240120_160500_abc123",
  "_id": "hades_queries/query_20240120_160500_abc123",
  "query_text": "How does PathRAG algorithm work?",
  "query_embedding": [0.456, 0.789, ...],
  "vfs_mount": "/tmp/hades_query_abc123",
  "results": {
    "chunks_retrieved": 10,
    "paths_explored": 25,
    "bridges_traversed": 3,
    "response_time_ms": 150
  },
  "supra_weights_used": {
    "semantic_similarity": 0.4,
    "structural_similarity": 0.3,
    "temporal_proximity": 0.1,
    "co_occurrence": 0.2
  },
  "metadata": {
    "timestamp": "2024-01-20T16:05:00Z",
    "user_id": "anonymous",
    "session_id": "session_xyz",
    "feedback": null
  }
}
```

## Indexes

### Vector Indexes
```javascript
// Jina v4 embeddings
db.hades_embeddings.ensureIndex({
  type: "persistent",
  fields: ["embedding_type", "vector"],
  unique: false,
  sparse: false
});

// ISNE embeddings  
db.hades_chunks.ensureIndex({
  type: "persistent", 
  fields: ["embeddings.isne"],
  unique: false,
  sparse: true
});
```

### Graph Indexes
```javascript
// Edge traversal
db.hades_edges.ensureIndex({
  type: "persistent",
  fields: ["_from", "type"],
  unique: false
});

db.hades_edges.ensureIndex({
  type: "persistent",
  fields: ["_to", "type"],
  unique: false
});

// Bridge lookups
db.hades_bridges.ensureIndex({
  type: "persistent",
  fields: ["source.key", "type"],
  unique: false
});
```

### Search Indexes
```javascript
// Full-text search on content
db.hades_chunks.ensureIndex({
  type: "fulltext",
  fields: ["content"],
  minLength: 3
});

// Keyword search
db.hades_chunks.ensureIndex({
  type: "persistent",
  fields: ["keywords.extracted[*]"],
  unique: false
});
```

## AQL Query Examples

### 1. Multimodal Retrieval
```aql
// Find chunks similar to query embedding
FOR chunk IN hades_chunks
  LET similarity = COSINE_SIMILARITY(chunk.embeddings.jina_v4, @query_embedding)
  FILTER similarity > 0.7
  
  // Include visual content if present
  LET image = (
    FOR img IN hades_images
      FILTER img.path == chunk.document_key
      RETURN img
  )[0]
  
  RETURN {
    chunk: chunk,
    similarity: similarity,
    image: image,
    has_visual: image != null
  }
```

### 2. Bridge Traversal
```aql
// Find implementations of a concept
FOR concept IN hades_nodes
  FILTER concept.name == @concept_name
  
  // Find all implementation bridges
  FOR v, e, p IN 1..3 INBOUND concept hades_edges
    FILTER e.type IN ["implements", "references"]
    
    // Get chunk details
    LET chunk = (
      FOR c IN hades_chunks
        FILTER c.document_key == v._key
        RETURN c
    )[0]
    
    RETURN {
      path: p,
      implementation: v,
      chunk: chunk,
      confidence: e.properties.confidence
    }
```

### 3. PathRAG Graph Traversal
```aql
// PathRAG-style retrieval with supra-weights
WITH hades_nodes, hades_edges, hades_chunks
FOR start IN hades_chunks
  FILTER COSINE_SIMILARITY(start.embeddings.jina_v4, @query_embedding) > 0.6
  
  // Explore paths
  FOR v, e, p IN 1..@max_depth ANY start hades_edges
    OPTIONS {
      weightAttribute: "supra_weights.semantic_similarity",
      defaultWeight: 0.1
    }
    
    // Calculate path score
    LET path_score = (
      @alpha * AVG(p.edges[*].supra_weights.semantic_similarity) +
      @beta * AVG(p.edges[*].supra_weights.structural_similarity) +
      @gamma * AVG(p.edges[*].supra_weights.temporal_proximity) +
      @delta * AVG(p.edges[*].supra_weights.co_occurrence)
    )
    
    FILTER path_score > @threshold
    
    RETURN {
      path: p,
      score: path_score,
      target: v
    }
```

## Migration and Maintenance

### Schema Versioning
- Current version: 1.0.0
- Version stored in `hades_metadata` collection
- Migrations handled by `src/storage/arangodb/migrations/`

### Backup Strategy
```bash
# Full backup
arangodump --server.database hades --output-directory backup/

# Collections only
arangodump --server.database hades \
  --collection hades_documents \
  --collection hades_chunks \
  --collection hades_embeddings
```

### Performance Optimization
1. **Sharding**: For large deployments, shard by document path prefix
2. **Caching**: Frequently accessed embeddings cached in Redis
3. **Batch Operations**: Use ArangoDB batch API for bulk inserts
4. **Graph Pruning**: Periodically remove low-confidence edges

## Integration Points

### With Jina v4
- Embeddings stored in standardized format
- Multimodal embeddings in same vector space
- LoRA adapter metadata preserved

### With ISNE
- Graph structure drives ISNE training
- ISNE embeddings stored alongside Jina v4
- Neighborhood relationships in edges

### With PathRAG
- Supra-weights calculated at query time
- Path exploration uses graph traversal
- Results ranked by multi-dimensional scores