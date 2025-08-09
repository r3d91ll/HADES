# Phase Architecture Implementation Guide

## Web Content Processing & Graph Structure

### Document Version

- **Date**: 2025-01-20
- **Status**: Implementation Blueprint
- **Context**: Major architectural revision based on reviewer feedback

---

## Executive Summary

This document defines the complete implementation strategy for processing web content (documentation, blogs, tutorials) and integrating it with existing ArXiv and GitHub data using a phased approach with ArangoDB graph structures.

### Key Architectural Decisions

1. **No Vertex Duplication**: Use base collections directly as vertices
2. **Edge Collections by Relation Type**: Not by source (edges_cites, edges_explains, etc.)
3. **Named Graphs**: Reference existing collections without copying
4. **Deterministic Keys**: Hash-based for idempotent operations
5. **Provenance on Edges**: Track source system, confidence, evidence

---

## Phase Architecture Overview

### Phase 1: Base Data Import

**Goal**: Import raw data into source-specific collections with embeddings

```
base_arxiv/          # ArXiv papers (existing)
base_github/         # GitHub repositories (existing)
base_pytorch_docs/   # PyTorch documentation
base_transformers_docs/ # HuggingFace Transformers docs
base_curated_blogs/  # High-quality blogs (Jay Alammar, Distill, etc.)
base_documentation/  # General documentation
```

### Phase 1.5: Edge Materialization

**Goal**: Create edges from metadata using relation-based collections

```
edges_cites         # Paper → Paper citations
edges_mentions      # Blog/Doc → Paper/Repo (lightweight reference)
edges_explains      # Blog/Doc → Paper (detailed explanation)
edges_implements    # Repo/File → Paper (implementation)
edges_depends_on    # Repo → Repo/Package dependencies
edges_links_to      # Doc → Doc navigation
edges_equivalent_to # Entity ↔ Entity (deduplication)
```

### Phase 2: Analytical Graphs

**Goal**: Create named graphs over existing collections for analysis

```
g_core_knowledge    # Main graph with all base collections
g_analysis_word2vec # Specific analysis graphs
g_pytorch_ecosystem # Framework-specific graphs
```

---

## Collection Schemas

### Base Collections (Vertices)

#### base_curated_blogs

```javascript
{
  "_key": "jalammar_word2vec",  // Deterministic: hash(url)
  "url": "https://jalammar.github.io/illustrated-word2vec/",
  "domain": "jalammar.github.io",
  "title": "The Illustrated Word2Vec",
  "author": "Jay Alammar",
  "content_type": "blog",
  "content_subtype": "illustrated_guide",
  "published_date": "2019-03-01",
  "content": "...",  // Full markdown
  "embedding": [...], // Jina v4 2048-dim
  "references": {
    "arxiv_papers": ["1301.3781", "1310.4546"],
    "github_repos": ["tensorflow/tensorflow", "pytorch/pytorch"],
    "websites": [...]
  },
  "quality_tier": "curated",
  "processed_at": "2025-01-20T..."
}
```

#### base_pytorch_docs

```javascript
{
  "_key": "torch_nn_transformer",  // hash(url_path)
  "url": "https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html",
  "url_path": "/docs/stable/generated/torch.nn.Transformer.html",
  "version": "2.1",
  "title": "torch.nn.Transformer",
  "description": "A transformer model based on 'Attention Is All You Need'",
  "content": "...",  // Full markdown
  "embedding": [...], // Jina v4 2048-dim
  "code_examples": [
    {
      "language": "python",
      "purpose": "basic_usage",
      "code": "transformer = nn.Transformer(...)"
    }
  ],
  "references": {
    "arxiv_papers": ["1706.03762"],  // Attention Is All You Need
    "internal_links": ["torch.nn.MultiheadAttention", "torch.nn.TransformerEncoder"]
  },
  "api_signature": "torch.nn.Transformer(d_model=512, nhead=8, ...)",
  "processed_at": "2025-01-20T..."
}
```

### Edge Collections

#### Edge Schema (All Edge Collections)

```javascript
{
  "_key": "hash(from|to|relation|source)",  // Deterministic for idempotency
  "_from": "base_curated_blogs/jalammar_word2vec",
  "_to": "base_arxiv/1301.3781",
  "relation": "explains",  // Redundant but useful for queries
  "source_system": "curated_blogs",  // Who created this edge
  "extraction_run_id": "2025-01-20T12:34:56Z",
  "confidence": 0.95,
  "evidence": [
    "The paper 'Efficient Estimation of Word Representations' by Mikolov et al.",
    "Link anchor text: 'original Word2Vec paper'"
  ],
  "metadata": {
    "extraction_method": "reference_parser",
    "context_window": 100  // chars around reference
  }
}
```

---

## Implementation Steps

### Step 1: Database Setup

```javascript
// setup_collections.js - Run in arangosh
const db = require("@arangodb").db;

// 1. Create base collections (vertices)
const baseCollections = [
  "base_arxiv",           // Existing
  "base_github",          // Existing
  "base_pytorch_docs",
  "base_transformers_docs",
  "base_curated_blogs",
  "base_documentation"
];

baseCollections.forEach(name => {
  if (!db._collection(name)) {
    const col = db._createDocumentCollection(name);
    console.log(`Created collection: ${name}`);
    
    // Add indexes
    col.ensureIndex({ type: "persistent", fields: ["url"], unique: true, sparse: true });
    col.ensureIndex({ type: "persistent", fields: ["domain"], sparse: true });
    
    // Vector index for embeddings (ArangoDB 3.11+)
    if (name !== "base_github") {  // GitHub might not have embeddings initially
      col.ensureIndex({ 
        type: "inverted", 
        fields: [{ name: "embedding", dimension: 2048 }] 
      });
    }
  }
});

// 2. Create edge collections
const edgeCollections = [
  "edges_cites",
  "edges_mentions",
  "edges_explains",
  "edges_implements",
  "edges_depends_on",
  "edges_links_to",
  "edges_equivalent_to"
];

edgeCollections.forEach(name => {
  if (!db._collection(name)) {
    const col = db._createEdgeCollection(name);
    console.log(`Created edge collection: ${name}`);
    
    // Standard edge indexes
    col.ensureIndex({ type: "persistent", fields: ["_from"] });
    col.ensureIndex({ type: "persistent", fields: ["_to"] });
    col.ensureIndex({ type: "persistent", fields: ["relation", "source_system"] });
    col.ensureIndex({ type: "persistent", fields: ["confidence"] });
  }
});

// 3. Create named graphs
const gm = require("@arangodb/general-graph");

// Core knowledge graph
const CORE_GRAPH = "g_core_knowledge";
if (gm._exists(CORE_GRAPH)) {
  gm._drop(CORE_GRAPH, true);
}

gm._create(
  CORE_GRAPH,
  [
    // Define edge relationships
    { 
      collection: "edges_cites", 
      from: ["base_arxiv"], 
      to: ["base_arxiv"] 
    },
    { 
      collection: "edges_mentions", 
      from: ["base_curated_blogs", "base_documentation", "base_pytorch_docs", "base_transformers_docs"],
      to: ["base_arxiv", "base_github"] 
    },
    { 
      collection: "edges_explains", 
      from: ["base_curated_blogs", "base_pytorch_docs", "base_transformers_docs"],
      to: ["base_arxiv"] 
    },
    { 
      collection: "edges_implements", 
      from: ["base_github"], 
      to: ["base_arxiv"] 
    },
    { 
      collection: "edges_depends_on", 
      from: ["base_github"], 
      to: ["base_github"] 
    },
    { 
      collection: "edges_links_to", 
      from: ["base_documentation", "base_pytorch_docs", "base_transformers_docs"],
      to: ["base_documentation", "base_pytorch_docs", "base_transformers_docs"] 
    }
  ],
  // All vertex collections
  baseCollections
);

console.log("Setup complete!");
```

### Step 2: Process Jay Alammar's Word2Vec Article

```python
# process_jalammar_word2vec.py
import os
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from arango import ArangoClient
from universal_web_processor import UniversalWebMetadata
from jina_v4_embedder import JinaV4Embedder

class Word2VecProcessor:
    def __init__(self):
        self.embedder = JinaV4Embedder(device="cuda", use_fp16=True)
        
        # Connect to ArangoDB
        password = os.getenv('ARANGO_PASSWORD')
        client = ArangoClient(hosts='http://localhost:8529')
        self.db = client.db('academy_store', username='root', password=password)
    
    def process_article(self, url: str, content: str):
        """Process Jay Alammar's Word2Vec article."""
        
        # 1. Extract metadata
        metadata = UniversalWebMetadata.extract_metadata(url, content)
        
        # 2. Generate deterministic key
        doc_key = hashlib.sha256(url.encode()).hexdigest()[:12]
        
        # 3. Generate embedding for full content
        embedding = self.embedder.embed_texts([content], task="retrieval")[0]
        
        # 4. Create document for base_curated_blogs
        doc = {
            "_key": doc_key,
            "url": url,
            "domain": "jalammar.github.io",
            "title": metadata['title'],
            "author": "Jay Alammar",
            "content_type": "blog",
            "content_subtype": "illustrated_guide",
            "published_date": metadata.get('published_date'),
            "content": content[:100000],  # Limit size
            "embedding": embedding.tolist(),
            "embedding_model": "jinaai/jina-embeddings-v4",
            "references": metadata['references'],
            "quality_tier": "curated",
            "quality_indicators": metadata['content_quality'],
            "technical_depth": metadata['technical_depth'],
            "word_count": metadata['word_count'],
            "has_code": metadata['has_code'],
            "has_diagrams": metadata['has_diagrams'],
            "processed_at": datetime.now(timezone.utc).isoformat()
        }
        
        # 5. Insert into base collection
        self.db.collection("base_curated_blogs").insert(doc, overwrite=True)
        print(f"Inserted document: {doc_key}")
        
        # 6. Create edges (Phase 1.5)
        self.create_edges(doc_key, metadata['references'])
        
        return doc_key
    
    def create_edges(self, doc_key: str, references: dict):
        """Create edges based on references."""
        
        # Process ArXiv references
        for arxiv_id in references.get('arxiv_papers', []):
            # Check if paper exists in base_arxiv
            if self.db.collection("base_arxiv").has(arxiv_id):
                edge_key = hashlib.sha256(
                    f"base_curated_blogs/{doc_key}|base_arxiv/{arxiv_id}|explains|curated_blogs".encode()
                ).hexdigest()[:12]
                
                edge = {
                    "_key": edge_key,
                    "_from": f"base_curated_blogs/{doc_key}",
                    "_to": f"base_arxiv/{arxiv_id}",
                    "relation": "explains",
                    "source_system": "curated_blogs",
                    "extraction_run_id": datetime.now(timezone.utc).isoformat(),
                    "confidence": 0.95,
                    "evidence": ["Direct reference in illustrated guide"]
                }
                
                self.db.collection("edges_explains").insert(edge, overwrite=True)
                print(f"Created edge: explains -> {arxiv_id}")
        
        # Process GitHub references
        for repo in references.get('github_repos', []):
            # Generate repo key (you'll need to normalize this)
            repo_key = hashlib.sha256(repo.encode()).hexdigest()[:12]
            
            if self.db.collection("base_github").has(repo_key):
                edge_key = hashlib.sha256(
                    f"base_curated_blogs/{doc_key}|base_github/{repo_key}|mentions|curated_blogs".encode()
                ).hexdigest()[:12]
                
                edge = {
                    "_key": edge_key,
                    "_from": f"base_curated_blogs/{doc_key}",
                    "_to": f"base_github/{repo_key}",
                    "relation": "mentions",
                    "source_system": "curated_blogs",
                    "extraction_run_id": datetime.now(timezone.utc).isoformat(),
                    "confidence": 0.85,
                    "evidence": ["Repository mentioned in blog post"]
                }
                
                self.db.collection("edges_mentions").insert(edge, overwrite=True)
                print(f"Created edge: mentions -> {repo}")

# Usage
processor = Word2VecProcessor()
# Assume content is from Firecrawl
content = "..."  # From the scraped content we got earlier
processor.process_article(
    "https://jalammar.github.io/illustrated-word2vec/",
    content
)
```

### Step 3: Phase 1.5 - Edge Materialization

```javascript
// phase_1_5_edge_materialization.aql
// Run these AQL queries to create edges from existing metadata

// 1. Create edges from curated blogs to ArXiv papers
FOR blog IN base_curated_blogs
  FILTER blog.references != null AND blog.references.arxiv_papers != null
  FOR arxiv_id IN blog.references.arxiv_papers
    LET to_doc = DOCUMENT(CONCAT("base_arxiv/", arxiv_id))
    FILTER to_doc != null  // Only if paper exists
    
    LET from_id = blog._id
    LET to_id = to_doc._id
    LET relation = blog.content_subtype == "illustrated_guide" ? "explains" : "mentions"
    LET source = "curated_blogs"
    LET confidence = blog.quality_tier == "curated" ? 0.95 : 0.85
    
    // Generate deterministic key
    LET edge_key = UPPER(SHA256(CONCAT_SEPARATOR("|", from_id, to_id, relation, source)))
    
    UPSERT { _key: edge_key }
      INSERT {
        _key: edge_key,
        _from: from_id,
        _to: to_id,
        relation: relation,
        source_system: source,
        extraction_run_id: DATE_ISO8601(DATE_NOW()),
        confidence: confidence,
        evidence: ["Reference extracted from blog metadata"]
      }
      UPDATE {
        confidence: MAX(OLD.confidence, confidence),
        extraction_run_id: DATE_ISO8601(DATE_NOW())
      }
      IN @@edge_collection

// Bind variables:
// @edge_collection: "edges_explains" or "edges_mentions"

// 2. Create edges from PyTorch docs to ArXiv papers
FOR doc IN base_pytorch_docs
  FILTER doc.references != null AND doc.references.arxiv_papers != null
  FOR arxiv_id IN doc.references.arxiv_papers
    LET to_doc = DOCUMENT(CONCAT("base_arxiv/", arxiv_id))
    FILTER to_doc != null
    
    LET from_id = doc._id
    LET to_id = to_doc._id
    LET relation = "explains"  // Documentation explains papers
    LET source = "pytorch_docs"
    LET confidence = 0.90  // Official docs are high confidence
    
    LET edge_key = UPPER(SHA256(CONCAT_SEPARATOR("|", from_id, to_id, relation, source)))
    
    UPSERT { _key: edge_key }
      INSERT {
        _key: edge_key,
        _from: from_id,
        _to: to_id,
        relation: relation,
        source_system: source,
        extraction_run_id: DATE_ISO8601(DATE_NOW()),
        confidence: confidence,
        evidence: ["Official PyTorch documentation reference"]
      }
      UPDATE {
        confidence: MAX(OLD.confidence, confidence),
        extraction_run_id: DATE_ISO8601(DATE_NOW())
      }
      IN edges_explains

// 3. Create internal documentation links
FOR doc IN base_pytorch_docs
  FILTER doc.references != null AND doc.references.internal_links != null
  FOR link IN doc.references.internal_links
    // Convert internal link to document key
    LET link_key = UPPER(SHA256(link))[:12]
    LET to_doc = DOCUMENT(CONCAT("base_pytorch_docs/", link_key))
    FILTER to_doc != null
    
    LET from_id = doc._id
    LET to_id = to_doc._id
    LET relation = "links_to"
    LET source = "pytorch_docs"
    LET confidence = 1.0  // Internal links are certain
    
    LET edge_key = UPPER(SHA256(CONCAT_SEPARATOR("|", from_id, to_id, relation, source)))
    
    UPSERT { _key: edge_key }
      INSERT {
        _key: edge_key,
        _from: from_id,
        _to: to_id,
        relation: relation,
        source_system: source,
        extraction_run_id: DATE_ISO8601(DATE_NOW()),
        confidence: confidence,
        evidence: ["Internal documentation navigation"]
      }
      UPDATE {
        extraction_run_id: DATE_ISO8601(DATE_NOW())
      }
      IN edges_links_to
```

### Step 4: Phase 2 - Analytical Queries

```javascript
// analytical_queries.aql

// 1. Find all explanations of Word2Vec paper
FOR paper IN base_arxiv
  FILTER paper._key == "1301.3781"  // Word2Vec paper
  FOR v, e, p IN 1..2 INBOUND paper 
    edges_explains, edges_mentions, edges_implements
    OPTIONS { uniqueVertices: "global", uniqueEdges: "global" }
    FILTER e.confidence >= 0.8
    RETURN {
      source: v,
      relation: e.relation,
      confidence: e.confidence,
      path_length: LENGTH(p.edges)
    }

// 2. Find implementation path from paper to code
FOR paper IN base_arxiv
  FILTER paper._key == "1706.03762"  // Attention Is All You Need
  FOR v, e, p IN 1..3 OUTBOUND paper 
    edges_explains, edges_implements
    FILTER v._id LIKE "base_github/%"
    RETURN {
      paper: paper.title,
      implementation: v.repo_name,
      path: p.edges[*].relation,
      total_confidence: PRODUCT(p.edges[*].confidence)
    }

// 3. Find all content explaining a concept (semantic search + graph)
LET query_embedding = @query_embedding  // From Jina v4
FOR doc IN base_curated_blogs, base_pytorch_docs
  LET similarity = COSINE_SIMILARITY(query_embedding, doc.embedding)
  FILTER similarity > 0.7
  
  // Enhance with graph context
  LET papers = (
    FOR paper IN 1..1 OUTBOUND doc edges_explains, edges_mentions
      FILTER paper._id LIKE "base_arxiv/%"
      RETURN paper._key
  )
  
  LET implementations = (
    FOR repo IN 1..2 OUTBOUND doc edges_mentions, edges_implements
      FILTER repo._id LIKE "base_github/%"
      RETURN repo.repo_name
  )
  
  SORT similarity DESC
  LIMIT 10
  RETURN {
    document: doc.title,
    url: doc.url,
    similarity: similarity,
    explains_papers: papers,
    related_implementations: implementations
  }
```

---

## Migration Path from Current Architecture

### Current State

- Separate web processing scripts
- Mixed edge/vertex thinking
- Unclear container boundaries

### Migration Steps

1. **Week 1: Setup Infrastructure**
   - Run setup_collections.js
   - Create named graphs
   - Test with Jay Alammar article

2. **Week 2: Process High-Value Content**
   - Process curated blogs collection
   - Process PyTorch Transformer docs
   - Verify edge creation

3. **Week 3: Batch Processing**
   - Process remaining PyTorch docs
   - Add Transformers documentation
   - Run Phase 1.5 edge materialization

4. **Week 4: Analysis & Validation**
   - Test traversal queries
   - Validate bridge discovery
   - Performance optimization

---

## Key Implementation Files

```
processors/web/
├── setup_collections.js           # Database setup script
├── process_jalammar_word2vec.py  # Example processor
├── phase_1_5_edge_materialization.aql  # Edge creation queries
├── universal_web_processor.py     # Modified for new architecture
├── batch_process_docs.py         # Batch processor for documentation
└── analytical_queries.aql        # Phase 2 analysis queries
```

---

## Success Metrics

### Phase 1 Success

- [ ] All base collections created
- [ ] Jay Alammar article processed with embeddings
- [ ] PyTorch Transformer docs processed
- [ ] 100+ documents in base_curated_blogs

### Phase 1.5 Success

- [ ] edges_explains has 50+ edges
- [ ] edges_mentions has 200+ edges
- [ ] All edges have provenance data
- [ ] Confidence scores properly set

### Phase 2 Success

- [ ] Can traverse from paper to all implementations
- [ ] Can find all explanations of a concept
- [ ] Semantic + graph search working
- [ ] Query performance < 100ms for 2-hop traversals

---

## Common Patterns & Best Practices

### Deterministic Keys

```python
def generate_edge_key(from_id: str, to_id: str, relation: str, source: str) -> str:
    """Generate deterministic edge key for idempotency."""
    components = f"{from_id}|{to_id}|{relation}|{source}"
    return hashlib.sha256(components.encode()).hexdigest()[:12]
```

### Confidence Scoring

```python
CONFIDENCE_MATRIX = {
    "curated_blogs": {"explains": 0.95, "mentions": 0.85},
    "pytorch_docs": {"explains": 0.90, "mentions": 0.80},
    "general_docs": {"explains": 0.75, "mentions": 0.70},
    "community": {"mentions": 0.60}
}
```

### Evidence Extraction

```python
def extract_evidence(content: str, reference: str, window: int = 100) -> str:
    """Extract context around reference for evidence."""
    idx = content.lower().find(reference.lower())
    if idx != -1:
        start = max(0, idx - window)
        end = min(len(content), idx + len(reference) + window)
        return content[start:end]
    return ""
```

---

## Troubleshooting

### Issue: Edges not being created

**Solution**: Check that both vertices exist before creating edge

```javascript
LET to_doc = DOCUMENT(CONCAT("base_arxiv/", arxiv_id))
FILTER to_doc != null  // Critical check
```

### Issue: Duplicate edges

**Solution**: Use deterministic keys and UPSERT

```javascript
LET edge_key = UPPER(SHA256(CONCAT_SEPARATOR("|", from_id, to_id, relation, source)))
UPSERT { _key: edge_key }
```

### Issue: Slow traversals

**Solution**: Add appropriate indexes

```javascript
col.ensureIndex({ type: "persistent", fields: ["relation", "confidence"] });
```

---

## Next Steps

1. **Immediate**: Process Jay Alammar's Word2Vec article
2. **This Week**: Setup all collections and graphs
3. **Next Week**: Batch process PyTorch documentation
4. **Month Goal**: Complete Phase 1.5 for all current content

---

## Questions for Review

1. Should we add `edges_derived_from` for paper-to-paper non-citation relationships?
2. How to handle versioning in documentation (PyTorch 2.0 vs 2.1)?
3. Threshold for automatic edge creation vs manual curation?
4. Should we add temporal edges (`edges_preceded_by`) for time-based analysis?

---

## Appendix: Full Setup Script

[Include complete setup_collections.js here]

## Appendix: Complete Edge Taxonomy

[Include detailed edge type definitions and use cases]
