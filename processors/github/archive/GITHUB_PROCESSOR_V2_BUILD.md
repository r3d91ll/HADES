# GitHub Processor v2.0 Build Document

## Upgrading to Phase 2 Architecture with Hierarchical Code Chunking

### Document Version

- **Date**: 2025-01-20
- **Target Version**: 2.0
- **Priority**: HIGH - Required for Phase 1.5 completion

---

## Executive Summary

This document outlines the upgrade path for the GitHub processor from MVP status to production-ready v2.0, achieving functional parity with ArXiv processor v7.0. The upgrade introduces hierarchical code chunking, Phase 2 edge architecture, and theory-practice bridge discovery.

### Key Deliverables

1. Phase 2 architecture compliance (vertex + edge collections)
2. Hierarchical AST-based code chunking
3. Delta edge management for dependencies and implementations
4. ArXiv paper implementation detection
5. Measurable performance and accuracy targets

---

## Current State Assessment (MVP)

### What We Have

#### Architecture

- **Single Collection Design**: Everything in `base_github`
- **No Edge Collections**: Relationships stored as arrays in documents
- **Basic Deduplication**: By repository URL only

#### Code Processing

```python
# Current approach (simplified file-level)
{
    "_key": "repo_hash",
    "repo_url": "github.com/user/repo",
    "files": [
        {
            "path": "src/model.py",
            "content": "...",
            "embedding": [...]  # Whole file embedding
        }
    ],
    "dependencies": ["numpy", "torch"],  # Basic list
    "topics": ["machine-learning"]       # GitHub topics only
}
```

#### Limitations

- ❌ No semantic code chunking
- ❌ No function/class level understanding
- ❌ No ArXiv paper detection
- ❌ No dependency version tracking
- ❌ No incremental updates
- ❌ Poor search granularity

### Performance Baseline (Current MVP)

- **Processing Speed**: ~45 seconds per repository
- **Embedding Quality**: File-level only (poor granularity)
- **Memory Usage**: ~2GB for large repos
- **Search Precision**: ~0.4 (too coarse)
- **Bridge Detection**: 0% (not implemented)

---

## Target State (v2.0)

### Phase 2 Architecture

#### Collections Structure

```yaml
# Vertex Collections
base_github:          # Repository metadata
base_github_files:    # File-level information
base_github_chunks:   # Hierarchical code chunks (file/class/method/function)

# Edge Collections
edges_implements:     # Code → ArXiv paper implementations
edges_depends_on_spec: # Repository → Package (declared dependencies)
edges_depends_on_lock: # Repository → PackageVersion (resolved dependencies)
edges_contains:       # Parent → Child containment hierarchy
edges_references:     # Code → Paper references in comments
```

#### Hierarchical Chunking Model

```
Repository (base_github)
  ├── edges_contains → File (base_github_files)
  │   ├── edges_contains → Module-level chunk (imports, globals)
  │   ├── edges_contains → Class chunk
  │   │   └── edges_contains → Method chunks
  │   └── edges_contains → Function chunks
  └── edges_depends_on → Dependencies
  └── edges_implements → ArXiv papers
```

### Key Normalization Rules

```python
def normalize_repo_key(owner: str, repo: str) -> str:
    """Generate deterministic repo key."""
    # Lowercase, replace / with __
    return f"{owner.lower()}__{repo.lower()}"

def normalize_file_key(owner: str, repo: str, path: str, blob_sha: str) -> str:
    """Generate deterministic file key using Git blob SHA."""
    # Normalize path: lowercase, forward slashes, no whitespace
    norm_path = path.lower().replace('\\', '/').replace(' ', '_')
    norm_path = re.sub(r'/+', '/', norm_path)  # Collapse multiple slashes
    return f"{owner.lower()}__{repo.lower()}::{norm_path}@{blob_sha[:12]}"

def normalize_chunk_key(file_key: str, chunk_type: str, signature: str) -> str:
    """Generate stable chunk key."""
    # Use stable hash of signature/span for determinism
    sig_hash = hashlib.sha256(signature.encode()).hexdigest()[:8]
    return f"{file_key}#{chunk_type}:{sig_hash}"
```

### Data Model Examples

#### Repository Document (base_github)

```javascript
{
    "_key": "pytorch__pytorch",  // Normalized: owner__repo
    "_id": "base_github/pytorch__pytorch",
    "uid": "github:pytorch/pytorch",
    
    // Metadata
    "repo_url": "https://github.com/pytorch/pytorch",
    "owner": "pytorch",
    "name": "pytorch",
    "description": "Tensors and Dynamic neural networks in Python",
    "stars": 72000,
    "language": "Python",
    "topics": ["machine-learning", "deep-learning", "tensor"],
    
    // Processing metadata
    "clone_date": "2025-01-20T10:00:00Z",
    "last_processed_commit": "abc123def456",  // Track for delta updates
    "default_branch": "main",
    "file_count": 3457,
    "total_size_mb": 234.5,
    
    // Summary embeddings
    "readme_embedding": [...],  // 2048-dim Jina v4
    "description_embedding": [...],
    
    // Aggregated metrics
    "total_chunks": 15234,
    "class_count": 892,
    "function_count": 4521,
    "detected_papers": ["1706.03762", "1512.03385"],  // ArXiv IDs
    
    "processing_version": "v2.0",
    "processing_date": "2025-01-20T12:00:00Z"
}
```

#### Code Chunk Document (base_github_chunks)

```javascript
{
    "_key": "pytorch__pytorch::torch/tensor.py@a1b2c3d4#class:e5f6g7h8",
    "_id": "base_github_chunks/pytorch__pytorch::torch/tensor.py@a1b2c3d4#class:e5f6g7h8",
    "uid": "github:pytorch/pytorch#class:Tensor",
    
    // Hierarchy
    "repo_key": "pytorch__pytorch",
    "file_path": "torch/tensor.py",
    "blob_sha": "a1b2c3d4e5f6",  // Git blob SHA for versioning
    "parent_key": "pytorch__pytorch::torch/tensor.py@a1b2c3d4",
    "chunk_type": "class",  // file|class|method|function
    "chunk_level": 2,  // 1=file, 2=class/function, 3=method
    
    // Content
    "name": "Tensor",
    "signature": "class Tensor(torch._C._TensorBase):",
    "docstring": "Multi-dimensional array with automatic differentiation",
    "content": "class Tensor(torch._C._TensorBase):\n    ...",  // First 1000 chars
    "full_content_hash": "sha256:...",
    
    // Code metrics
    "line_start": 145,
    "line_end": 3421,
    "line_count": 3276,
    "complexity": 8,  // Cyclomatic complexity
    
    // Semantic information
    "imports_used": ["torch._C", "numpy", "warnings"],
    "calls_functions": ["zeros", "ones", "empty"],
    "implements_patterns": ["tensor_manipulation", "autograd"],
    
    // Embeddings with context
    "embedding": [...],  // 2048-dim Jina v4 code LoRA
    "embedding_model": "jinaai/jina-embeddings-v4",
    "embedding_version": "1.0.0",
    
    // Detected references
    "arxiv_refs": [],  // Found in comments/docstrings
    "paper_titles": ["Automatic differentiation in PyTorch"],
    
    "processing_date": "2025-01-20T12:00:00Z"
}
```

#### Implementation Edge (edges_implements)

```javascript
{
    "_key": "e8f9d7c6a5b4e3d2c1b0a9f8e7d6c5b4",  // 32-char hash
    "_from": "base_github_chunks/pytorch__pytorch::nn/attention.py@xyz789#class:abc123",
    "_to": "base_arxiv/1706_03762",  // Attention Is All You Need
    
    "relation": "implements",
    "source_system": "github_processor",
    "confidence": 0.86,  // Calibrated blend of signals
    
    "signals": {
        "text": 0.92,      // Comment/doc mentions paper
        "structural": 0.74, // AST pattern matches
        "semantic": 0.81    // Embedding similarity
    },
    "threshold": 0.80,
    "ruleset": "v1.0",
    
    "evidence_snippet": "Multi-headed attention from 'Attention Is All You Need'",
    "evidence_refs": ["evidence/e8f9d7c6a5b4"],  // Full evidence in side collection
    
    "active": true,
    "extraction_run_id": "2025-01-20T12:00:00Z"
}
```

#### Containment Edge (edges_contains)

```javascript
{
    "_key": "c1234567890abcdef1234567890abcdef",
    "_from": "base_github_files/pytorch__pytorch::torch/tensor.py@a1b2c3d4",
    "_to": "base_github_chunks/pytorch__pytorch::torch/tensor.py@a1b2c3d4#class:e5f6g7h8",
    "relation": "contains",
    "source_system": "processor",
    "order_idx": 37,  // Ordering within parent
    "span": {
        "line_start": 145,
        "line_end": 3421
    },
    "active": true
}
```

#### Dependency Edges (Separate Declared vs Resolved)

```javascript
// edges_depends_on_spec (declared dependencies)
{
    "_key": "d1234567890abcdef1234567890abcdef",
    "_from": "base_github/pytorch__pytorch",
    "_to": "packages/numpy",
    "relation": "depends_on_spec",
    "version_spec": ">=1.19.0",
    "source_file": "requirements.txt",
    "ecosystem": "pypi"
}

// edges_depends_on_lock (resolved dependencies)
{
    "_key": "d9876543210fedcba9876543210fedcba",
    "_from": "base_github/pytorch__pytorch",
    "_to": "package_versions/numpy__1.21.3",
    "relation": "depends_on_lock",
    "package_name": "numpy",
    "version_exact": "1.21.3",
    "ecosystem": "pypi",
    "lockfile": "poetry.lock"
}
```

---

## Vector Search Implementation

### Index Creation

```javascript
// Create vector index on chunks
db._collection("base_github_chunks").ensureIndex({
    type: "vector",
    fields: ["embedding"],
    dimension: 2048,
    metric: "cosine"
});

// Create composite indexes for filtering
db._collection("base_github_chunks").ensureIndex({
    type: "persistent",
    fields: ["repo_key", "chunk_type"]
});
```

### KNN Query Pattern

```aql
// Vector KNN search with filtering
LET query_embedding = @query_embedding
FOR chunk IN base_github_chunks
    SEARCH KNN(chunk.embedding, query_embedding, 50)
    FILTER chunk.chunk_type IN ["class", "function", "method"]
    SORT DISTANCE(chunk.embedding, query_embedding) ASC
    LIMIT 10
    RETURN {
        repo: chunk.repo_key,
        path: chunk.file_path,
        name: chunk.name,
        similarity: 1 - DISTANCE(chunk.embedding, query_embedding)
    }
```

### Performance SLO

- **KNN(top_k=50) over 1-5M chunks**: < 100ms p95
- **Hybrid search (text + vector)**: < 200ms p95
- **3-hop graph traversal**: < 150ms p95

---

## Delta Update Implementation

### Git Diff Processing Loop

```python
def process_delta_updates(repo_key: str, last_commit: str, current_commit: str):
    """Process repository changes using git diff."""
    
    # Get changes between commits
    changes = git.diff(last_commit, current_commit, name_status=True)
    # Returns: [{status: 'A/M/D/R', old_path, new_path, old_blob, new_blob}]
    
    for change in changes:
        if change.status in ('A', 'M'):  # Added or Modified
            # Load file content from blob
            content = git.get_blob(change.new_blob)
            file_key = normalize_file_key(owner, repo, change.new_path, change.new_blob)
            
            # Parse and chunk with tree-sitter
            chunks_new = tree_sitter_chunker.chunk_file(
                content, change.new_path, detect_language(change.new_path)
            )
            
            # Delta update chunks and containment edges
            delta_update_chunks(file_key, chunks_new)
            
        elif change.status == 'D':  # Deleted
            file_key = normalize_file_key(owner, repo, change.old_path, change.old_blob)
            deactivate_file_and_chunks(file_key)
            
        elif change.status == 'R':  # Renamed
            # Treat as delete + add, but preserve rename evidence
            old_key = normalize_file_key(owner, repo, change.old_path, change.old_blob)
            new_key = normalize_file_key(owner, repo, change.new_path, change.new_blob)
            
            deactivate_file_and_chunks(old_key)
            # Store rename evidence
            store_rename_evidence(old_key, new_key)
            
            # Process new file
            content = git.get_blob(change.new_blob)
            chunks_new = tree_sitter_chunker.chunk_file(
                content, change.new_path, detect_language(change.new_path)
            )
            delta_update_chunks(new_key, chunks_new)
    
    # Update last processed commit only on success
    update_repo_commit(repo_key, current_commit)
```

### Retry Cursor for Failed Updates

```python
# Store progress in database
{
    "_key": "pytorch__pytorch",
    "last_processed_commit": "abc123",
    "processing_cursor": {
        "current_commit": "def456",
        "processed_files": ["file1.py", "file2.py"],
        "failed_files": [{"path": "file3.py", "error": "..."}],
        "retry_count": 1
    }
}
```

---

## Implementation Plan

### Phase 1: Architecture Migration

#### Tasks

1. **Create new collections** with proper indexes
2. **Implement delta edge management** (copy from ArXiv v7.0)
3. **Setup tree-sitter parsers** (multi-language support)
4. **Create hierarchical chunking pipeline**

#### Measurable Requirements

- [ ] All collections created with correct indexes
- [ ] Delta edge function tested with 95% coverage
- [ ] Tree-sitter extracts 100% of classes/functions for supported languages
- [ ] Chunking preserves parent-child relationships

#### Code Components

```python
# tree_sitter_chunker.py
class TreeSitterChunker:
    def __init__(self):
        self.parsers = self._setup_parsers()
        self.embedder = JinaV4Embedder(model='code')  # Use code LoRA
    
    def _setup_parsers(self) -> Dict:
        """Setup parsers for multiple languages."""
        parsers = {}
        for lang in ['python', 'javascript', 'typescript', 'go', 'rust', 'java']:
            parser = tree_sitter.Parser()
            parser.set_language(tree_sitter.Language(f'build/{lang}.so', lang))
            parsers[lang] = parser
        return parsers
    
    def chunk_file(self, content: str, file_path: str, language: str) -> List[Dict]:
        """Returns hierarchical chunks with structure-aware boundaries."""
        pass
    
    def extract_references(self, content: str) -> Dict:
        """Extract ArXiv IDs, paper titles, etc."""
        pass

# github_processor_v2.py  
class GitHubProcessorV2:
    def process_repository(self, repo_url: str) -> Dict:
        """Main entry point for v2 processing."""
        pass
    
    def _detect_implementations(self, chunks: List[Dict]) -> List[Dict]:
        """Detect which chunks implement known papers."""
        pass
```

### Phase 2: Core Processing

#### Tasks

1. **Repository cloning** with git history
2. **Language detection** and parser selection
3. **Hierarchical chunking** implementation
4. **Embedding generation** with context

#### Measurable Requirements

- [ ] Process 100-file repository in < 60 seconds
- [ ] Generate embeddings for 1000 chunks in < 30 seconds
- [ ] Memory usage < 4GB for repositories up to 1GB
- [ ] Chunk overlap < 10% (no excessive duplication)

#### Test Repositories

1. Small: `jalammar/illustrated-word2vec` (~20 files)
2. Medium: `huggingface/transformers` (~2000 files)
3. Large: `pytorch/pytorch` (~5000 files)

### Phase 3: Bridge Detection

#### Multi-Signal Gating Requirements

Emit `edges_implements` only when **≥2 signals** meet threshold:

1. **Text Signal** (weight: 0.35)
   - Comment/docstring mentions paper ID/title/authors
   - Threshold: 0.7
   
2. **Structural Signal** (weight: 0.35)
   - AST patterns match known implementations
   - Variable names, function signatures
   - Threshold: 0.6
   
3. **Semantic Signal** (weight: 0.30)
   - Embedding similarity with paper specification
   - Threshold: 0.75

```python
def calculate_implementation_confidence(signals: Dict[str, float]) -> tuple[float, bool]:
    """Calculate calibrated confidence with multi-signal gating."""
    weights = {'text': 0.35, 'structural': 0.35, 'semantic': 0.30}
    
    # Count signals above threshold
    thresholds = {'text': 0.7, 'structural': 0.6, 'semantic': 0.75}
    signals_met = sum(1 for k, v in signals.items() if v >= thresholds[k])
    
    # Require at least 2 signals
    if signals_met < 2:
        return 0.0, False
    
    # Calculate weighted confidence
    confidence = sum(signals[k] * weights[k] for k in signals)
    return confidence, confidence >= 0.80
```

#### Tasks

1. **ArXiv reference extraction** from comments/docs
2. **Tree-sitter pattern matching** for known algorithms
3. **Embedding similarity** computation with Jina v4
4. **Multi-signal confidence** calibration

#### Measurable Requirements

- [ ] Detect 90% of explicit ArXiv references in comments
- [ ] Identify 70% of Transformer implementations (with ≥2 signals)
- [ ] False positive rate < 10% (multi-signal gating)
- [ ] Process 1000 functions in < 10 seconds
- [ ] Calibrated confidence scores (within ±0.1 of human labels)

#### Detection Patterns

```python
IMPLEMENTATION_PATTERNS = {
    "1706.03762": {  # Attention Is All You Need
        "class_names": ["MultiHeadAttention", "Transformer"],
        "method_names": ["forward", "attention", "scaled_dot_product"],
        "variables": ["query", "key", "value", "num_heads"],
        "imports": ["torch.nn", "tensorflow.keras.layers"]
    },
    "1810.04805": {  # BERT
        "class_names": ["BertModel", "BertEncoder"],
        "method_names": ["forward", "encode", "mask_tokens"],
        "variables": ["hidden_states", "attention_mask", "token_type_ids"]
    }
}
```

### Phase 4: Integration & Testing

#### Tasks

1. **End-to-end testing** with real repositories
2. **Performance benchmarking**
3. **Edge validation** queries
4. **Documentation updates**

#### Measurable Requirements

- [ ] 95% test coverage on core functions
- [ ] Process 10 real repositories without errors
- [ ] All edges have required fields
- [ ] Query performance < 100ms for 3-hop traversals

---

## Success Metrics

### Functional Requirements

| Metric | Current (MVP) | Target (v2.0) | Measurement Method |
|--------|--------------|---------------|-------------------|
| **Chunking Granularity** | File-level | Method-level | AST node count |
| **Embedding Quality** | 0.4 precision | 0.75 precision | Similarity search eval |
| **Bridge Detection** | 0% | 60% recall | Known implementation test set |
| **Edge Types** | 0 | 4 types | Collection count |
| **Incremental Update** | No | Yes | Re-processing time |

### Performance Requirements

| Metric | Current (MVP) | Target (v2.0) | Test Case |
|--------|--------------|---------------|-----------|
| **Small Repo (<100 files)** | 45s | 30s | illustrated-word2vec |
| **Medium Repo (~1000 files)** | Not tested | 3 min | transformers (subset) |
| **Large Repo (5000+ files)** | Not tested | 10 min | pytorch (subset) |
| **Memory Usage** | 2GB | <4GB | Peak RSS during processing |
| **Embedding Batch** | 10/sec | 100/sec | 1000 chunks test |
| **Vector KNN (50 neighbors)** | N/A | <100ms p95 | 5M chunks |
| **Graph Traversal (3-hop)** | 500ms | <150ms | With PRUNE |
| **Delta Update** | N/A | <5s/100 files | Git diff processing |

### Quality Requirements

| Metric | Target | Validation Method |
|--------|--------|------------------|
| **Tree-sitter Parse Success** | >95% | Parse top 100 GitHub repos |
| **Chunk Completeness** | >90% | All functions captured |
| **Parent-Child Integrity** | 100% | Graph consistency check |
| **Embedding Coverage** | >95% | Non-null embeddings |
| **Reference Extraction** | >90% | Manual annotation test |
| **Implementation Detection** | >70% recall, <10% FP | Gold test set (10 repos) |
| **Key Determinism** | 100% | Same input → same key |

---

## Risk Mitigation

### Technical Risks

1. **Large Repository Memory Issues**
   - Mitigation: Stream processing, batch by directory
   - Fallback: Skip files > 10MB

2. **AST Parser Language Support**
   - Mitigation: Start with Python only
   - Fallback: File-level chunking for unsupported languages

3. **Embedding API Rate Limits**
   - Mitigation: Local Jina v4, batch processing
   - Fallback: Cache embeddings, retry logic

4. **Git Clone Failures**
   - Mitigation: Shallow clones, timeout handling
   - Fallback: Download as ZIP archive

### Data Risks

1. **Binary Files**
   - Mitigation: Detect and skip, extract metadata only
   - Validation: File extension whitelist

2. **Generated Code**
   - Mitigation: Detect patterns (autogenerated headers)
   - Validation: Skip if >10k lines

3. **Malicious Repositories**
   - Mitigation: Sandbox environment, resource limits
   - Validation: Pre-scan for suspicious patterns

---

## Testing Strategy

### Unit Tests

```python
# test_ast_chunker.py
def test_chunk_python_class():
    """Test that classes are correctly chunked."""
    
def test_chunk_nested_functions():
    """Test nested function extraction."""
    
def test_context_preservation():
    """Test that method chunks include class context."""

# test_bridge_detection.py
def test_detect_transformer_implementation():
    """Test detection of Attention Is All You Need."""
    
def test_confidence_scoring():
    """Test that confidence scores are calibrated."""
```

### Integration Tests

```python
# test_end_to_end.py
def test_process_small_repository():
    """Process jalammar/illustrated-word2vec completely."""
    
def test_incremental_update():
    """Test that re-processing only updates changed files."""
    
def test_edge_consistency():
    """Verify all edges have both vertices."""
```

### Performance Tests

```python
# benchmark_processing.py
def benchmark_chunk_generation():
    """Measure chunking speed for 1000 files."""
    
def benchmark_embedding_batch():
    """Measure embedding generation for 1000 chunks."""
    
def benchmark_memory_usage():
    """Track memory during large repo processing."""
```

---

## Implementation Priority

### Priority 1: Core Architecture

- Collections with vector indexes (type: "vector", dimension: 2048)
- Tree-sitter parser setup for multiple languages
- Delta edge management with git diff loop
- Key normalization rules (owner__repo, blob SHA)

### Priority 2: Processing Pipeline

- Repository cloning with security constraints
- Hierarchical chunking with tree-sitter
- Jina v4 code LoRA embeddings (batch size: 128)
- Containment edges with order_idx

### Priority 3: Bridge Detection

- Multi-signal gating (≥2 signals required)
- ArXiv reference extraction
- Tree-sitter pattern matching
- Calibrated confidence scoring

### Priority 4: Testing & Optimization

- Gold test set (10 repos with ground truth)
- Performance benchmarking
- Documentation updates
- Background re-embedding job

---

## Dependencies

### Required Before Starting

- [x] ArXiv processor v7.0 (reference implementation)
- [x] Phase 2 architecture documentation
- [ ] Tree-sitter multi-language parser setup
- [ ] Test repository dataset

### External Dependencies

```python
# requirements.txt additions
tree-sitter==0.20.4
tree-sitter-python==0.20.4
tree-sitter-javascript==0.20.3
tree-sitter-typescript==0.20.5
tree-sitter-go==0.20.0
tree-sitter-rust==0.20.4
tree-sitter-java==0.20.2
gitpython==3.1.40
pygments==2.17.2  # Code highlighting/analysis
radon==6.0.1      # Complexity metrics
```

---

## Definition of Done

### v2.0 Release Criteria

- [ ] All unit tests passing (>95% coverage)
- [ ] All integration tests passing
- [ ] Vector KNN < 100ms p95 on 5M chunks
- [ ] Git diff delta updates working
- [ ] Multi-signal bridge detection (≥2 signals)
- [ ] Gold test set: >70% recall, <10% false positives
- [ ] Key normalization 100% deterministic
- [ ] Containment edges preserve order
- [ ] Security: jailed cloning, no code execution
- [ ] Documentation updated

### Deliverables

1. `github_processor_v2.py` - Main processor
2. `tree_sitter_chunker.py` - Multi-language hierarchical chunking
3. `bridge_detector.py` - Implementation detection
4. `test_*.py` - Complete test suite
5. `README.md` - Updated documentation
6. Migration script from v1 to v2

---

## Post-Launch Monitoring

### Key Metrics to Track

- Processing success rate
- Average processing time by repo size
- Bridge detection accuracy
- Memory usage patterns
- Error frequency and types

### Success Indicators (First Week)

- 50+ repositories processed
- <5% error rate
- 10+ verified theory-practice bridges discovered
- No memory-related crashes
- Search latency meets targets

---

## Future Enhancements (v2.1+)

1. **Multi-language Support**: JavaScript, C++, Java
2. **Commit History Analysis**: Track implementation evolution
3. **PR/Issue Mining**: Extract discussions about papers
4. **Notebook Support**: Process Jupyter notebooks
5. **Dependency Resolution**: Follow and process dependencies
6. **Code Quality Metrics**: Add maintainability scores
7. **License Detection**: Track implementation licenses
8. **Security Scanning**: Flag potential vulnerabilities
