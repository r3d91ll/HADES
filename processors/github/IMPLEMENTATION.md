# GitHub Processor v2.0 - Complete Implementation Guide

## Document Version

- **Date**: 2025-01-20
- **Status**: Production Blueprint
- **Purpose**: Single source of truth for GitHub processor implementation

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Data Model & Collections](#data-model--collections)
3. [Key Normalization & Identity](#key-normalization--identity)
4. [Vector Search Implementation](#vector-search-implementation)
5. [Git Delta Processing](#git-delta-processing)
6. [Bridge Detection Logic](#bridge-detection-logic)
7. [Operational Requirements](#operational-requirements)
8. [Implementation Checklist](#implementation-checklist)

---

## Architecture Overview

### Phase 2 Compliance

- **Vertex Collections**: Repository metadata, files, and hierarchical chunks
- **Edge Collections**: Per-relation edges (implements, depends_on, contains)
- **Delta Updates**: Git diff-based incremental processing
- **Vector Search**: KNN with Jina v4 code embeddings

### Key Design Decisions

1. **Tree-sitter over AST**: Multi-language support, Jina v4 handles semantics
2. **Blob SHA identity**: File versioning using Git blob SHA, not commit SHA
3. **Multi-signal gating**: ≥2 signals required for implementation detection
4. **Soft deletes**: Mark inactive rather than delete, with TTL cleanup

---

## Data Model & Collections

### Vertex Collections

#### base_github (Repositories)

```javascript
{
    "_key": "pytorch__pytorch",  // owner__repo
    "_id": "base_github/pytorch__pytorch",
    "uid": "github:pytorch/pytorch",
    
    // Identity
    "owner": "pytorch",
    "name": "pytorch",
    "github_url": "https://github.com/pytorch/pytorch",
    "is_fork": false,
    "canonical_repo_key": "pytorch__pytorch",  // Self if not fork
    
    // Git state
    "default_branch": "main",
    "last_processed_commit": "abc123def456",
    "tree_sha": "fed987cba654",  // For quick change detection
    "clone_date": "2025-01-20T10:00:00Z",
    
    // Metadata
    "description": "Tensors and Dynamic neural networks",
    "stars": 72000,
    "language": "Python",
    "topics": ["machine-learning", "deep-learning"],
    "license": "BSD-3-Clause",
    
    // Size & stats
    "file_count": 3457,
    "total_size_mb": 234.5,
    "chunk_count": 15234,
    "class_count": 892,
    "function_count": 4521,
    
    // Processing
    "schema_version": "2.0.0",
    "processor_version": "v2.0.0",
    "processing_run_id": "run_2025_01_20_001"
}
```

#### base_github_files

```javascript
{
    "_key": "pytorch__pytorch::torch/tensor.py@a1b2c3d4",
    "_id": "base_github_files/pytorch__pytorch::torch/tensor.py@a1b2c3d4",
    
    // Identity
    "repo_key": "pytorch__pytorch",
    "file_path": "torch/tensor.py",
    "original_path": "torch/Tensor.py",  // Preserve case
    "blob_sha": "a1b2c3d4e5f6",
    
    // Metadata
    "language": "python",
    "size_bytes": 123456,
    "line_count": 3421,
    "is_binary": false,
    "is_lfs": false,
    "generated": false,
    
    // Extracted
    "imports": ["torch._C", "numpy", "warnings"],
    "exports": ["Tensor", "TensorBase"],
    "chunk_count": 42
}
```

#### base_github_chunks

```javascript
{
    "_key": "pytorch__pytorch::torch/tensor.py@a1b2c3d4#class:e5f6g7h8",
    "_id": "base_github_chunks/pytorch__pytorch::torch/tensor.py@a1b2c3d4#class:e5f6g7h8",
    
    // Hierarchy
    "repo_key": "pytorch__pytorch",
    "file_key": "pytorch__pytorch::torch/tensor.py@a1b2c3d4",
    "parent_key": null,  // null for top-level, parent chunk key for nested
    
    // Identity
    "chunk_type": "class",  // file|class|function|method
    "name": "Tensor",
    "signature": "class Tensor(torch._C._TensorBase):",
    
    // Location
    "line_start": 145,
    "line_end": 3421,
    "span": {"start_byte": 5234, "end_byte": 123456},
    
    // Content
    "content": "class Tensor(torch._C._TensorBase):\n    ...",  // First 5000 chars
    "docstring": "Multi-dimensional array with automatic differentiation",
    
    // Embeddings
    "embedding": [...],  // 2048-dim Jina v4 code LoRA
    "embedding_model": "jinaai/jina-embeddings-v4",
    "embedding_version": "1.0.0",
    
    // Metadata
    "complexity": 8,
    "imports_used": ["torch._C", "numpy"],
    "generated": false,
    "skip_in_search": false
}
```

#### packages & package_versions

```javascript
// packages collection
{
    "_key": "pypi__numpy",
    "ecosystem": "pypi",
    "name": "numpy",
    "latest_version": "1.24.3"
}

// package_versions collection
{
    "_key": "pypi__numpy@1.24.3",
    "package_key": "pypi__numpy",
    "version": "1.24.3"
}
```

### Edge Collections

#### edges_contains (Hierarchical structure)

```javascript
{
    "_key": "c1234567890abcdef1234567890abcdef",
    "_from": "base_github_files/pytorch__pytorch::torch/tensor.py@a1b2c3d4",
    "_to": "base_github_chunks/pytorch__pytorch::torch/tensor.py@a1b2c3d4#class:e5f6g7h8",
    "relation": "contains",
    "order_idx": 37,  // Position within parent
    "span": {"line_start": 145, "line_end": 3421},
    "active": true
}
```

#### edges_implements (Theory-practice bridges)

```javascript
{
    "_key": "e8f9d7c6a5b4e3d2c1b0a9f8e7d6c5b4",
    "_from": "base_github_chunks/pytorch__pytorch::nn/attention.py@xyz789#class:abc123",
    "_to": "base_arxiv/1706.03762",  // Note: dot not underscore
    
    "relation": "implements",
    "confidence": 0.86,
    
    "signals": {
        "text": 0.92,       // Comment/doc mentions
        "structural": 0.74,  // Pattern match
        "semantic": 0.81     // Embedding similarity
    },
    "threshold": 0.80,
    "ruleset": "v1.0",
    
    "evidence_snippet": "Multi-headed attention from 'Attention Is All You Need'",
    "evidence_refs": ["evidence/e8f9d7c6a5b4"],
    
    "active": true,
    "extraction_run_id": "2025-01-20T12:00:00Z"
}
```

#### edges_depends_on_spec & edges_depends_on_lock

```javascript
// Declared dependencies
{
    "_from": "base_github/pytorch__pytorch",
    "_to": "packages/pypi__numpy",
    "version_spec": ">=1.19.0",
    "source_file": "requirements.txt"
}

// Resolved dependencies
{
    "_from": "base_github/pytorch__pytorch",
    "_to": "package_versions/pypi__numpy@1.21.3",
    "lockfile": "poetry.lock"
}
```

---

## Key Normalization & Identity

### Deterministic Key Generation

```python
import re
import hashlib

def normalize_repo_key(owner: str, repo: str) -> str:
    """owner__repo format, lowercase."""
    return f"{owner.lower()}__{repo.lower()}"

def normalize_file_key(owner: str, repo: str, path: str, blob_sha: str) -> str:
    """owner__repo::path@blob_sha format."""
    # Normalize path: forward slashes, no spaces, lowercase
    norm_path = path.lower().replace('\\', '/').replace(' ', '_')
    norm_path = re.sub(r'/+', '/', norm_path)  # Collapse slashes
    return f"{owner.lower()}__{repo.lower()}::{norm_path}@{blob_sha[:12]}"

def normalize_chunk_key(file_key: str, chunk_type: str, signature: str) -> str:
    """file_key#type:hash format."""
    sig_hash = hashlib.sha256(signature.encode()).hexdigest()[:8]
    return f"{file_key}#{chunk_type}:{sig_hash}"

def generate_edge_key(from_id: str, to_id: str, relation: str, source: str) -> str:
    """32-character deterministic edge key."""
    components = f"{from_id}|{to_id}|{relation}|{source}"
    return hashlib.sha256(components.encode()).hexdigest()[:32]
```

### Path Case Preservation

```python
# Store both normalized (for key) and original (for display)
{
    "_key": normalize_path_for_key(path),  # lowercase
    "original_path": path,  # Original case
    "normalized_path": normalize_path_for_key(path)
}
```

---

## Vector Search Implementation

### Index Setup

```javascript
// Vector index on chunks
db._collection("base_github_chunks").ensureIndex({
    type: "vector",
    fields: ["embedding"],
    dimension: 2048,
    metric: "cosine"
});

// Composite indexes for filtering
db._collection("base_github_chunks").ensureIndex({
    type: "persistent",
    fields: ["repo_key", "chunk_type"]
});

// Package lookup
db._collection("package_versions").ensureIndex({
    type: "persistent",
    fields: ["package_name", "ecosystem", "version_exact"]
});
```

### KNN Query

```aql
LET query_embedding = @query_embedding
FOR chunk IN base_github_chunks
    SEARCH KNN(chunk.embedding, query_embedding, 50)
    FILTER chunk.chunk_type IN ["class", "function", "method"]
    FILTER chunk.generated != true
    SORT DISTANCE(chunk.embedding, query_embedding) ASC
    LIMIT 10
    RETURN {
        repo: chunk.repo_key,
        path: chunk.file_key,
        name: chunk.name,
        similarity: 1 - DISTANCE(chunk.embedding, query_embedding)
    }
```

### Performance SLOs

- **KNN(top_k=50) over 1-5M chunks**: < 100ms p95
- **Hybrid search**: < 200ms p95
- **3-hop traversal with PRUNE**: < 150ms p95

---

## Git Delta Processing

### Core Loop

```python
def process_delta_updates(repo_key: str, last_commit: str, current_commit: str):
    """Process repository changes using git diff."""
    
    # Quick check: has tree changed?
    if get_tree_sha(last_commit) == get_tree_sha(current_commit):
        return {"status": "no_changes"}
    
    # Get file changes
    changes = git.diff(last_commit, current_commit, name_status=True)
    # Returns: [{status: 'A/M/D/R', old_path, new_path, old_blob, new_blob}]
    
    for change in changes:
        if change.status in ('A', 'M'):  # Added or Modified
            content = git.get_blob(change.new_blob)
            file_key = normalize_file_key(owner, repo, change.new_path, change.new_blob)
            
            # Skip if file too large or generated
            if should_skip_file(change.new_path, content):
                continue
            
            # Parse and chunk
            chunks_new = tree_sitter_chunker.chunk_file(
                content, change.new_path, detect_language(change.new_path)
            )
            
            # Delta update
            delta_update_chunks(file_key, chunks_new)
            
        elif change.status == 'D':  # Deleted
            file_key = normalize_file_key(owner, repo, change.old_path, change.old_blob)
            cascade_deactivate(file_key)
            
        elif change.status == 'R':  # Renamed
            # New chunks due to path in key
            old_key = normalize_file_key(owner, repo, change.old_path, change.old_blob)
            new_key = normalize_file_key(owner, repo, change.new_path, change.new_blob)
            
            cascade_deactivate(old_key)
            store_rename_evidence(old_key, new_key)
            
            # Process as new file
            content = git.get_blob(change.new_blob)
            chunks_new = tree_sitter_chunker.chunk_file(
                content, change.new_path, detect_language(change.new_path)
            )
            delta_update_chunks(new_key, chunks_new)
    
    # Update commit pointer only on success
    update_repo_commit(repo_key, current_commit)
```

### Cascading Deactivation

```python
def cascade_deactivate(file_key: str):
    """Deactivate file, chunks, and edges atomically."""
    query = """
    LET timestamp = DATE_ISO8601(DATE_NOW())
    
    // Deactivate file
    UPDATE { _key: @file_key, active: false, deactivated_at: timestamp }
    IN base_github_files
    
    // Deactivate chunks
    FOR chunk IN base_github_chunks
        FILTER chunk.file_key == @file_key
        UPDATE chunk WITH { active: false, deactivated_at: timestamp }
        IN base_github_chunks
        
        // Deactivate outbound edges
        FOR e IN edges_implements
            FILTER e._from == chunk._id
            UPDATE e WITH { active: false, deactivated_at: timestamp }
            IN edges_implements
    """
    db.aql.execute(query, bind_vars={'file_key': file_key})
```

---

## Bridge Detection Logic

### Multi-Signal Gating

```python
def calculate_implementation_confidence(signals: Dict[str, float]) -> tuple[float, bool]:
    """Require ≥2 signals above threshold."""
    weights = {'text': 0.35, 'structural': 0.35, 'semantic': 0.30}
    thresholds = {'text': 0.7, 'structural': 0.6, 'semantic': 0.75}
    
    # Count signals meeting threshold
    signals_met = sum(1 for k, v in signals.items() if v >= thresholds[k])
    
    # Gate: require at least 2 signals
    if signals_met < 2:
        return 0.0, False
    
    # Weighted confidence
    confidence = sum(signals[k] * weights[k] for k in signals)
    return confidence, confidence >= 0.80
```

### Detection Patterns

```python
IMPLEMENTATION_PATTERNS = {
    "1706.03762": {  # Attention Is All You Need
        "class_names": ["MultiHeadAttention", "Transformer"],
        "method_names": ["forward", "attention", "scaled_dot_product"],
        "variables": ["query", "key", "value", "num_heads"]
    },
    "1810.04805": {  # BERT
        "class_names": ["BertModel", "BertEncoder"],
        "variables": ["hidden_states", "attention_mask"]
    }
}
```

---

## Operational Requirements

### File Processing Rules

```python
# Skip directories
SKIP_DIRS = {
    '.git', 'node_modules', 'vendor', 'dist', 'build',
    '__pycache__', 'venv', 'site-packages'
}

# Skip extensions
SKIP_EXTENSIONS = {
    '.png', '.jpg', '.pdf', '.zip', '.exe', '.pyc',
    '.jar', '.db', '.sqlite', '.pt', '.ckpt'
}

# Size limits
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB
MAX_CHUNKS_PER_FILE = 10000
MAX_REPO_SIZE = 1024 * 1024 * 1024  # 1GB
```

### Security & Compliance

```python
# Secret patterns to scrub
SECRET_PATTERNS = [
    (r'sk-[a-zA-Z0-9]{48}', 'openai_key'),
    (r'AKIA[0-9A-Z]{16}', 'aws_key'),
    (r'ghp_[a-zA-Z0-9]{36}', 'github_token')
]

# Clone with security
def secure_clone(repo_url: str) -> Path:
    # Use shallow clone
    cmd = ['git', 'clone', '--depth', '1', '--single-branch']
    
    # Run in sandbox with timeout
    subprocess.run(cmd, timeout=300, cwd=SANDBOX_DIR)
    
    # Verify size
    if get_dir_size(repo_path) > MAX_REPO_SIZE:
        raise ValueError("Repository too large")
```

### Performance Configuration

```python
# Batching
EMBED_BATCH_SIZE = 128
PARSE_WORKERS = 4  # CPU cores
EMBED_WORKERS = 2  # GPU streams

# Backpressure
MAX_EMBED_QUEUE = 10000

def should_pause_parsing() -> bool:
    return embed_queue.qsize() > MAX_EMBED_QUEUE * 0.8
```

### Monitoring & Metrics

```python
# Run records
{
    "_key": "run_2025_01_20_12_34_56",
    "repo_key": "pytorch__pytorch",
    "run_type": "delta_update",
    "stats": {
        "files_parsed": 234,
        "chunks_added": 1523,
        "edges_added": 67
    },
    "duration_seconds": 627,
    "memory_peak_mb": 3456
}

# Metrics to track
- Parse failures by language
- Embedding generation rate
- Edge false positive rate
- KNN query latency p95/p99
```

---

## Implementation Checklist

### Phase 1: Core Infrastructure ✓

- [ ] Collections with vector indexes
- [ ] Key normalization functions
- [ ] Tree-sitter parser setup
- [ ] Git integration (clone, diff)

### Phase 2: Processing Pipeline

- [ ] File filtering and validation
- [ ] Hierarchical chunking
- [ ] Jina v4 embeddings
- [ ] Delta update logic
- [ ] Cascading deactivation

### Phase 3: Bridge Detection

- [ ] Multi-signal extractors
- [ ] Confidence calculation
- [ ] Pattern matching
- [ ] Evidence storage

### Phase 4: Production Readiness

- [ ] Security (sandbox, secrets)
- [ ] Performance (batching, backpressure)
- [ ] Monitoring (metrics, run records)
- [ ] Testing (gold set, idempotency)

### Quality Gates

- [ ] Vector KNN < 100ms p95
- [ ] Git diff processing < 5s/100 files
- [ ] Implementation detection >70% recall, <10% FP
- [ ] 100% deterministic keys
- [ ] Zero orphaned chunks after deactivation

---

## Appendix: Tree-sitter Chunker

```python
class TreeSitterChunker:
    def __init__(self):
        self.parsers = self._setup_parsers()
        self.embedder = JinaV4Embedder(model='code')
    
    def _setup_parsers(self) -> Dict:
        parsers = {}
        for lang in ['python', 'javascript', 'typescript', 'go', 'rust', 'java']:
            parser = tree_sitter.Parser()
            parser.set_language(tree_sitter.Language(f'build/{lang}.so', lang))
            parsers[lang] = parser
        return parsers
    
    def chunk_file(self, content: str, file_path: str, language: str) -> List[Dict]:
        """Extract hierarchical chunks from code."""
        parser = self.parsers.get(language)
        if not parser:
            return self._fallback_chunk(content)
        
        tree = parser.parse(bytes(content, 'utf8'))
        chunks = []
        
        # Extract classes
        for class_node in self._find_nodes(tree, 'class_definition'):
            chunk = {
                'type': 'class',
                'name': self._get_node_name(class_node),
                'content': content[class_node.start_byte:class_node.end_byte],
                'line_start': class_node.start_point[0],
                'line_end': class_node.end_point[0]
            }
            chunks.append(chunk)
            
            # Extract methods within class
            for method_node in self._find_nodes(class_node, 'function_definition'):
                method_chunk = {
                    'type': 'method',
                    'parent': chunk['name'],
                    'name': self._get_node_name(method_node),
                    'content': content[method_node.start_byte:method_node.end_byte]
                }
                chunks.append(method_chunk)
        
        # Extract standalone functions
        for func_node in self._find_nodes(tree, 'function_definition'):
            if not self._is_nested(func_node):
                chunks.append({
                    'type': 'function',
                    'name': self._get_node_name(func_node),
                    'content': content[func_node.start_byte:func_node.end_byte]
                })
        
        return chunks
```

---

## Summary

This consolidated guide provides:

1. **Complete data model** with all collections and fields
2. **Deterministic key generation** with normalization rules
3. **Vector search** with specific indexes and queries
4. **Git delta processing** with full implementation
5. **Multi-signal bridge detection** with gating logic
6. **Operational requirements** for production deployment
7. **Clear implementation checklist** with quality gates

Follow this guide to build a production-ready GitHub processor that integrates seamlessly with the Phase 2 architecture.
