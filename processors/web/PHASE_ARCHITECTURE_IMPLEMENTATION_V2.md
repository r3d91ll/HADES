# Phase Architecture Implementation Guide V2
## Web Content Processing & Graph Structure (Revised)

### Document Version
- **Date**: 2025-01-20
- **Version**: 2.0 - Critical fixes applied
- **Status**: Production-Ready Blueprint

---

## Executive Summary

Revised implementation incorporating critical fixes:
- Proper vector indexing for KNN search
- Removed redundant edge indexes
- Extended key lengths to prevent collisions
- Added PRUNE for efficient traversals
- Chunk-level embeddings architecture
- Versioning support for documentation

---

## Phase Architecture Overview

### Phase 1: Base Data Import
**Goal**: Import raw data into source-specific collections with doc AND chunk embeddings

```yaml
# Document collections (metadata + doc-level embeddings)
base_arxiv/             # ArXiv papers
base_github/            # GitHub repositories  
base_pytorch_docs/      # PyTorch documentation
base_transformers_docs/ # HuggingFace Transformers
base_curated_blogs/     # High-quality blogs
base_documentation/     # General documentation

# Chunk collections (fine-grained embeddings)
base_arxiv_chunks/      # Paper chunks with embeddings
base_pytorch_chunks/    # Doc chunks with embeddings
base_curated_chunks/    # Blog chunks with embeddings
```

### Phase 1.5: Edge Materialization
**Goal**: Create edges from metadata using relation-based collections

```yaml
edges_cites         # Paper → Paper citations
edges_mentions      # Blog/Doc → Paper/Repo (lightweight)
edges_explains      # Blog/Doc → Paper (detailed)
edges_implements    # Repo/File → Paper
edges_depends_on    # Repo → Repo/Package
edges_links_to      # Doc → Doc navigation
edges_contains      # Doc → Chunk containment
edges_supersedes    # Version → Version (for docs)
```

### Phase 2: Analytical Graphs
**Goal**: Named graphs over existing collections

```yaml
g_core_knowledge    # Main graph
g_analysis_word2vec # Analysis-specific
g_pytorch_ecosystem # Framework-specific
```

---

## Critical Fixes Applied

### 1. Vector Indexing (NOT Inverted)

```javascript
// CORRECT: Vector index for KNN search
col.ensureIndex({
  type: "vector",
  fields: ["embedding"],
  dimension: 2048,
  metric: "cosine"  // or "l2" or "dot"
});

// Query with proper KNN operator
FOR doc IN base_curated_blogs
  LET neighbors = (
    FOR neighbor IN base_curated_blogs
      SORT VECTOR_DISTANCE(neighbor.embedding, @query_embedding, "cosine")
      LIMIT 10
      RETURN neighbor
  )
  RETURN neighbors
```

### 2. No Redundant Edge Indexes

```javascript
// DON'T add these - ArangoDB creates them automatically
// col.ensureIndex({ type: "persistent", fields: ["_from"] });  // NO!
// col.ensureIndex({ type: "persistent", fields: ["_to"] });    // NO!

// DO add these for non-traversal filtering
col.ensureIndex({ type: "persistent", fields: ["relation", "source_system"] });
col.ensureIndex({ type: "persistent", fields: ["confidence"] });  // If filtering often
col.ensureIndex({ type: "persistent", fields: ["extraction_run_id"] });  // For auditing
```

### 3. Longer Edge Keys (24-32 chars)

```python
def generate_edge_key(from_id: str, to_id: str, relation: str, source: str) -> str:
    """Generate deterministic edge key - 32 chars to avoid collisions."""
    components = f"{from_id}|{to_id}|{relation}|{source}"
    return hashlib.sha256(components.encode()).hexdigest()[:32]  # 32 chars, not 12!
```

### 4. PRUNE for Efficient Traversals

```javascript
// CORRECT: PRUNE stops expansion early
FOR v, e, p IN 1..3 INBOUND paper GRAPH "g_core_knowledge"
  PRUNE e.confidence < 0.8  // Stop exploring low-confidence branches
  FILTER e.confidence >= 0.8  // Still filter results
  OPTIONS { uniqueVertices: "global" }
  RETURN {
    document: v,
    path_confidence: MIN(p.edges[*].confidence),  // MIN instead of PRODUCT
    path: p.edges[*].relation
  }
```

### 5. Versioning Support

```javascript
// Composite unique index for versioned docs
col.ensureIndex({ 
  type: "persistent", 
  fields: ["url_path", "version"], 
  unique: true, 
  sparse: true 
});

// Key includes version
"_key": "pytorch_transformer_v2_1"  // Not just "pytorch_transformer"
```

---

## Improved Collection Schemas

### Document Collections (Coarse-grained)

```javascript
// base_curated_blogs
{
  "_key": "jalammar_word2vec_2019",  // Include year for uniqueness
  "url": "https://jalammar.github.io/illustrated-word2vec/",
  "canonical_url": "jalammar.github.io/illustrated-word2vec",  // Normalized
  "title": "The Illustrated Word2Vec",
  "author": "Jay Alammar",
  "published_date": "2019-03-01",
  "doc_embedding": [...],  // Doc-level embedding for coarse search
  "chunk_count": 15,  // Number of chunks
  "version": "2019-03-01",  // Version tracking
  "canonical_id": "jalammar_word2vec",  // For deduplication
  "processed_at": "2025-01-20T...",
  "chunk_refs": ["base_curated_chunks/jw_chunk_001", ...]  // Chunk references
}
```

### Chunk Collections (Fine-grained)

```javascript
// base_curated_chunks
{
  "_key": "jw_chunk_001",  // Shortened but unique
  "_from": "base_curated_blogs/jalammar_word2vec_2019",  // Parent doc
  "chunk_index": 0,
  "content": "...",  // Actual chunk text
  "chunk_embedding": [...],  // Chunk-level embedding for precise search
  "context": {
    "section": "Introduction",
    "position": 0.05,  // Relative position in document
    "has_code": false,
    "has_math": true
  },
  "references_local": ["1301.3781"],  // References found in THIS chunk
  "embedding_model": "jinaai/jina-embeddings-v4",
  "tokens": 512
}
```

### Edge Collections (Improved)

```javascript
// edges_explains with proper evidence handling
{
  "_key": "e1234567890abcdef1234567890abcdef",  // 32 char hex
  "_from": "base_curated_chunks/jw_chunk_003",  // Can be from chunk!
  "_to": "base_arxiv/1301.3781",
  "relation": "explains",
  "source_system": "curated_blogs",
  "extraction_run_id": "run_2025_01_20_001",
  "confidence": 0.95,
  "evidence_snippet": "Mikolov et al. introduced...",  // SHORT snippet
  "evidence_ref": "evidence/e1234567890abcdef1234567890abcdef",  // Full evidence elsewhere
  "created_at": "2025-01-20T12:34:56Z",
  "updated_at": "2025-01-20T12:34:56Z"
}
```

---

## Database Setup Script (Corrected)

```javascript
// setup_collections_v2.js
const db = require("@arangodb").db;
const gm = require("@arangodb/general-graph");

// 1. Create document collections
const docCollections = [
  "base_arxiv", "base_github", 
  "base_pytorch_docs", "base_transformers_docs",
  "base_curated_blogs", "base_documentation"
];

const chunkCollections = [
  "base_arxiv_chunks", "base_pytorch_chunks",
  "base_curated_chunks", "base_documentation_chunks"
];

// Create document collections with proper indexes
[...docCollections, ...chunkCollections].forEach(name => {
  if (!db._collection(name)) {
    const col = db._createDocumentCollection(name);
    console.log(`Created: ${name}`);
    
    // Vector index for embeddings (CORRECT)
    const embeddingField = name.includes("_chunks") ? "chunk_embedding" : "doc_embedding";
    col.ensureIndex({
      type: "vector",
      fields: [embeddingField],
      dimension: 2048,
      metric: "cosine"
    });
    
    // URL index (composite for versioning)
    if (name.includes("docs")) {
      col.ensureIndex({ 
        type: "persistent", 
        fields: ["url_path", "version"], 
        unique: true, 
        sparse: true 
      });
    }
    
    // Canonical ID for deduplication
    col.ensureIndex({ 
      type: "persistent", 
      fields: ["canonical_id"], 
      sparse: true 
    });
  }
});

// 2. Create edge collections (NO redundant _from/_to indexes!)
const edgeRelations = [
  "edges_cites", "edges_mentions", "edges_explains",
  "edges_implements", "edges_depends_on", "edges_links_to",
  "edges_contains", "edges_supersedes"
];

edgeRelations.forEach(name => {
  if (!db._collection(name)) {
    const col = db._createEdgeCollection(name);
    console.log(`Created edge: ${name}`);
    
    // Only add indexes we'll actually filter on
    col.ensureIndex({ 
      type: "persistent", 
      fields: ["relation", "source_system"] 
    });
    
    // Confidence index only if we filter outside traversals
    if (["edges_explains", "edges_mentions"].includes(name)) {
      col.ensureIndex({ 
        type: "persistent", 
        fields: ["confidence"] 
      });
    }
    
    // Run tracking
    col.ensureIndex({ 
      type: "persistent", 
      fields: ["extraction_run_id"] 
    });
  }
});

// 3. Evidence side collection (for large evidence)
if (!db._collection("edge_evidence")) {
  const col = db._createDocumentCollection("edge_evidence");
  col.ensureIndex({ 
    type: "persistent", 
    fields: ["edge_id"], 
    unique: false  // Multiple evidence per edge
  });
}

// 4. Create named graph (CORRECTED relationships)
const CORE_GRAPH = "g_core_knowledge";
if (gm._exists(CORE_GRAPH)) {
  gm._drop(CORE_GRAPH, true);
}

gm._create(
  CORE_GRAPH,
  [
    // Document-level edges
    { 
      collection: "edges_cites",
      from: ["base_arxiv"],
      to: ["base_arxiv"]
    },
    { 
      collection: "edges_mentions",
      from: [...docCollections],
      to: ["base_arxiv", "base_github"]
    },
    { 
      collection: "edges_explains",
      from: ["base_curated_blogs", "base_curated_chunks", 
             "base_pytorch_docs", "base_pytorch_chunks"],
      to: ["base_arxiv"]
    },
    { 
      collection: "edges_implements",
      from: ["base_github"],
      to: ["base_arxiv"]
    },
    { 
      collection: "edges_contains",
      from: docCollections,
      to: chunkCollections
    },
    { 
      collection: "edges_supersedes",
      from: ["base_pytorch_docs", "base_transformers_docs"],
      to: ["base_pytorch_docs", "base_transformers_docs"]
    }
  ],
  [...docCollections, ...chunkCollections]
);

console.log("Setup complete with vector indexes!");
```

---

## Processing Pipeline (With Chunks)

```python
# process_with_chunks.py
import hashlib
from typing import List, Dict
import numpy as np

class ChunkProcessor:
    def __init__(self, embedder, db):
        self.embedder = embedder
        self.db = db
        
    def process_document(self, url: str, content: str, metadata: Dict):
        """Process document with chunk-level embeddings."""
        
        # 1. Generate doc ID (human-readable when possible)
        doc_key = self._generate_doc_key(url, metadata.get('version'))
        
        # 2. Generate doc-level embedding
        doc_embedding = self.embedder.embed_texts([content[:8000]], task="retrieval")[0]
        
        # 3. Create chunks
        chunks = self._create_chunks(content, chunk_size=1000, overlap=100)
        
        # 4. Generate chunk embeddings (batch for efficiency)
        chunk_texts = [c['content'] for c in chunks]
        chunk_embeddings = self.embedder.embed_texts(chunk_texts, task="retrieval")
        
        # 5. Store document
        doc = {
            "_key": doc_key,
            "url": url,
            "title": metadata['title'],
            "doc_embedding": doc_embedding.tolist(),
            "chunk_count": len(chunks),
            "version": metadata.get('version', 'latest'),
            "canonical_id": self._get_canonical_id(url),
            "processed_at": datetime.now(timezone.utc).isoformat()
        }
        
        self.db.collection("base_curated_blogs").insert(doc, overwrite=False)
        
        # 6. Store chunks with edges
        chunk_refs = []
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            chunk_key = f"{doc_key[:12]}_c{i:03d}"  # Readable chunk keys
            
            chunk_doc = {
                "_key": chunk_key,
                "chunk_index": i,
                "content": chunk['content'],
                "chunk_embedding": embedding.tolist(),
                "context": chunk['context'],
                "references_local": self._extract_local_refs(chunk['content'])
            }
            
            self.db.collection("base_curated_chunks").insert(chunk_doc, overwrite=False)
            chunk_refs.append(f"base_curated_chunks/{chunk_key}")
            
            # Create containment edge
            edge_key = hashlib.sha256(
                f"base_curated_blogs/{doc_key}|base_curated_chunks/{chunk_key}|contains|system"
                .encode()
            ).hexdigest()[:32]
            
            edge = {
                "_key": edge_key,
                "_from": f"base_curated_blogs/{doc_key}",
                "_to": f"base_curated_chunks/{chunk_key}",
                "relation": "contains",
                "source_system": "processor",
                "confidence": 1.0
            }
            
            self.db.collection("edges_contains").insert(edge, overwrite=False)
        
        return doc_key, chunk_refs
    
    def _generate_doc_key(self, url: str, version: str = None) -> str:
        """Generate human-readable keys when possible."""
        # For known domains, use readable format
        if "jalammar.github.io" in url:
            slug = url.split('/')[-2] if url.endswith('/') else url.split('/')[-1]
            slug = slug.replace('.html', '').replace('-', '_')
            key = f"jalammar_{slug}"
            if version:
                key += f"_v{version.replace('.', '_')}"
            return key[:50]  # Limit length but keep readable
        
        # Fallback to hash
        return hashlib.sha256(url.encode()).hexdigest()[:32]
    
    def _create_chunks(self, content: str, chunk_size: int = 1000, 
                       overlap: int = 100) -> List[Dict]:
        """Create overlapping chunks with context."""
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_size = 0
        
        for i, line in enumerate(lines):
            current_chunk.append(line)
            current_size += len(line)
            
            if current_size >= chunk_size:
                chunk_content = '\n'.join(current_chunk)
                chunks.append({
                    'content': chunk_content,
                    'context': {
                        'start_line': i - len(current_chunk) + 1,
                        'end_line': i,
                        'position': i / max(len(lines), 1)
                    }
                })
                
                # Keep overlap
                if overlap > 0:
                    keep_lines = []
                    kept_size = 0
                    for line in reversed(current_chunk):
                        kept_size += len(line)
                        keep_lines.insert(0, line)
                        if kept_size >= overlap:
                            break
                    current_chunk = keep_lines
                    current_size = kept_size
                else:
                    current_chunk = []
                    current_size = 0
        
        # Add remaining content
        if current_chunk:
            chunks.append({
                'content': '\n'.join(current_chunk),
                'context': {
                    'start_line': len(lines) - len(current_chunk),
                    'end_line': len(lines),
                    'position': 1.0
                }
            })
        
        return chunks
```

---

## Improved Traversal Queries

### Using PRUNE and Named Graphs

```javascript
// Find explanations with confidence threshold (OPTIMIZED)
FOR paper IN base_arxiv
  FILTER paper._key == "1301.3781"
  
  FOR v, e, p IN 1..3 INBOUND paper GRAPH "g_core_knowledge"
    PRUNE e.confidence < 0.8  // Stop exploring early!
    FILTER e.confidence >= 0.8
    OPTIONS { 
      uniqueVertices: "global",
      bfs: true  // Breadth-first often better for confidence
    }
    
    // Use MIN instead of PRODUCT for confidence
    LET path_confidence = MIN(p.edges[*].confidence)
    
    SORT path_confidence DESC
    LIMIT 20
    
    RETURN {
      source: v._key,
      source_type: SPLIT(v._id, "/")[0],
      relations: p.edges[*].relation,
      confidence: path_confidence,
      path_length: LENGTH(p.edges)
    }
```

### Hybrid Search (Vector + Graph)

```javascript
// ArangoSearch View for hybrid search (create once)
db._createView("v_content_search", "arangosearch", {
  links: {
    "base_curated_chunks": {
      fields: {
        "content": { analyzers: ["text_en"] },
        "chunk_embedding": { 
          analyzers: ["identity"],
          dimensions: 2048,
          similarity: "cosine"
        }
      }
    },
    "base_pytorch_chunks": {
      fields: {
        "content": { analyzers: ["text_en"] },
        "chunk_embedding": { 
          analyzers: ["identity"],
          dimensions: 2048,
          similarity: "cosine"
        }
      }
    }
  }
});

// Hybrid query
FOR chunk IN v_content_search
  SEARCH 
    // Text search
    ANALYZER(TOKENS("transformer attention", "text_en") ALL IN chunk.content, "text_en")
    AND
    // Vector KNN
    VECTOR_DISTANCE(chunk.chunk_embedding, @query_embedding, "cosine") < 0.3
    
  // Enhance with graph
  LET parent = DOCUMENT(chunk._from)
  LET papers = (
    FOR paper IN 1..1 OUTBOUND parent GRAPH "g_core_knowledge"
      FILTER paper._id LIKE "base_arxiv/%"
      RETURN paper._key
  )
  
  SORT VECTOR_DISTANCE(chunk.chunk_embedding, @query_embedding, "cosine")
  LIMIT 10
  
  RETURN {
    chunk: chunk._key,
    parent_doc: parent.title,
    content_snippet: SUBSTRING(chunk.content, 0, 200),
    related_papers: papers,
    similarity: 1 - VECTOR_DISTANCE(chunk.chunk_embedding, @query_embedding, "cosine")
  }
```

---

## Evidence Management Strategy

```javascript
// Lightweight edge with evidence reference
{
  "_key": "e1234567890abcdef1234567890abcdef",
  "_from": "base_curated_chunks/jw_chunk_003",
  "_to": "base_arxiv/1301.3781",
  "evidence_snippet": "First 100 chars...",  // SHORT
  "evidence_ref": "evidence/e1234567890"  // Full evidence key
}

// Separate evidence collection for details
{
  "_key": "e1234567890",
  "edge_id": "e1234567890abcdef1234567890abcdef",
  "full_text": "Complete evidence text...",
  "extraction_context": {
    "window_size": 500,
    "method": "nlp_extraction",
    "confidence_factors": {...}
  }
}
```

---

## Canonicalization Strategy

```python
def canonicalize_entities(db):
    """Run periodically to maintain canonical IDs."""
    
    # Find equivalence components
    query = """
    FOR v IN base_curated_blogs
      LET component = (
        FOR related IN 1..10 ANY v edges_equivalent_to
          OPTIONS { uniqueVertices: "global" }
          RETURN related._key
      )
      RETURN DISTINCT {
        vertex: v._key,
        component: SORTED(component)
      }
    """
    
    components = {}
    for result in db.aql.execute(query):
        component_key = ','.join(result['component'])
        if component_key not in components:
            components[component_key] = []
        components[component_key].append(result['vertex'])
    
    # Update canonical IDs (smallest key in component)
    for component_keys in components.values():
        canonical = min(component_keys)  # Or use other criteria
        for key in component_keys:
            db.collection("base_curated_blogs").update(
                key, 
                {"canonical_id": canonical}
            )
```

---

## Performance Guidelines

### Index Strategy
- Vector indexes on embedding fields (NOT inverted!)
- Composite indexes for versioned content
- Indexes on frequently filtered edge properties
- NO redundant _from/_to indexes on edges

### Key Generation
- 32-character hex for edges (not 12!)
- Human-readable doc keys when possible
- Include version in keys for versioned content

### Traversal Optimization
- Use PRUNE to stop exploration early
- Prefer MIN over PRODUCT for path confidence
- Use named graphs for centralized edge definitions
- Consider BFS for confidence-based traversals

### Storage Optimization
- Chunks for large documents (better recall)
- Evidence references instead of inline storage
- Canonical IDs for deduplication
- Regular cleanup of orphaned edges

---

## Migration Checklist

- [ ] Run setup_collections_v2.js (with vector indexes)
- [ ] Process test document with chunks
- [ ] Verify vector KNN queries work
- [ ] Test PRUNE in traversals
- [ ] Validate 32-char edge keys
- [ ] Setup evidence side collection
- [ ] Create ArangoSearch view for hybrid
- [ ] Test versioned document updates
- [ ] Benchmark traversal performance
- [ ] Document canonicalization strategy