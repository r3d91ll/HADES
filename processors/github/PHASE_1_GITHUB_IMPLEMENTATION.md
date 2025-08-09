# GitHub Processor - Phase 1 Implementation

## Achieving Parity with ArXiv Processor v7.0

### Document Version

- **Date**: 2025-01-20
- **Goal**: Match ArXiv v7.0 architecture exactly
- **Status**: Implementation Ready

---

## Current Gap Analysis

### What ArXiv v7.0 Has (Target)

```
✅ base_arxiv collection with papers
✅ base_arxiv_chunks with embeddings
✅ edges_cites (paper → paper)
✅ edges_contains (paper → chunk)
✅ Delta edge updates
✅ 32-char deterministic keys
✅ Docling PDF processing
✅ Jina v4 late chunking
✅ Content-hash deduplication
```

### What GitHub MVP Has (Current)

```
✅ base_github collection exists
❌ No chunk collection
❌ No edge collections
❌ No delta updates
❌ Random keys
❌ No tree-sitter extraction
❌ No Jina v4 embeddings
❌ No deduplication
```

---

## Phase 1 Target: Minimal Parity

### Collections to Create

```python
# Match ArXiv structure exactly
vertex_collections = ['base_github', 'base_github_chunks']
edge_collections = [
    'edges_contains',     # Repo → Chunk containment
    'edges_depends_on',   # Repo → Package dependencies (simplified)
]
```

### Simplified Data Model (Phase 1)

#### base_github (Repository level - like base_arxiv)

```javascript
{
    "_key": "pytorch__pytorch",  // Deterministic: owner__repo
    "_id": "base_github/pytorch__pytorch",
    "uid": "github:pytorch/pytorch",  // Global identifier
    
    // Core metadata (like arxiv_id, title, authors)
    "owner": "pytorch",
    "name": "pytorch",
    "github_url": "https://github.com/pytorch/pytorch",
    "description": "Tensors and Dynamic neural networks",
    "default_branch": "main",
    
    // Processing state (like pdf_status in arxiv)
    "clone_status": "processed",
    "last_commit": "abc123def",
    "file_count": 234,
    "chunk_count": 1523,
    
    // Simplified metadata (like categories, published)
    "language": "Python",
    "topics": ["machine-learning"],
    "stars": 72000,
    
    // Processing metadata
    "processing_date": "2025-01-20T12:00:00Z",
    "processor_version": "v1.0",
    "content_hash": "sha256:..."  // For deduplication
}
```

#### base_github_chunks (Code chunks - like base_arxiv_chunks)

```javascript
{
    "_key": "pytorch__pytorch_chunk_0",  // Simple incremental
    "_id": "base_github_chunks/pytorch__pytorch_chunk_0",
    
    // Parent reference (like doc_id in arxiv_chunks)
    "repo_key": "pytorch__pytorch",
    "chunk_index": 0,
    
    // Content (simplified for Phase 1)
    "file_path": "torch/tensor.py",
    "chunk_type": "function",  // function|class|file
    "name": "zeros",
    "text": "def zeros(*size, out=None, dtype=None...)...",
    
    // Position (like start_char, end_char)
    "start_line": 145,
    "end_line": 189,
    
    // Embedding (exactly like arxiv)
    "embedding": [...],  // 2048-dim Jina v4
    "embedding_dim": 2048,
    
    // Metadata
    "language": "python",
    "processing_date": "2025-01-20T12:00:00Z"
}
```

---

## Phase 1 Implementation (Minimal)

### 1. Repository Processor (Simplified)

```python
class GitHubProcessorPhase1:
    """Phase 1: Basic repository processing matching ArXiv structure."""
    
    def __init__(self, db_config=None):
        self.db_config = db_config or {
            'host': 'localhost',
            'port': 8529,
            'username': 'root',
            'password': os.getenv('ARANGO_PASSWORD'),
            'database': 'academy_store'
        }
        self.embedder = JinaV4Embedder()
        self.tree_sitter = TreeSitterSymbolExtractor()  # Existing file
        self._setup_database()
    
    def _setup_database(self):
        """Create collections matching ArXiv v7.0."""
        # Exact same pattern as ArXiv
        vertex_collections = ['base_github', 'base_github_chunks']
        for coll_name in vertex_collections:
            if not self.db.has_collection(coll_name):
                self.db.create_collection(coll_name)
        
        edge_collections = ['edges_contains', 'edges_depends_on']
        for coll_name in edge_collections:
            if not self.db.has_collection(coll_name):
                self.db.create_collection(coll_name, edge=True)
    
    def process_repository(self, repo_url: str) -> Dict:
        """Process repository like ArXiv processes PDFs."""
        
        # 1. Clone repository (like downloading PDF)
        owner, name = self._parse_github_url(repo_url)
        repo_key = f"{owner.lower()}__{name.lower()}"
        
        # Check if already processed (like content hash check)
        existing = self.db.collection('base_github').get(repo_key)
        if existing:
            current_commit = self._get_latest_commit(repo_url)
            if existing.get('last_commit') == current_commit:
                return {'status': 'unchanged', 'repo_key': repo_key}
        
        # 2. Clone and extract
        repo_path = self._clone_repo(repo_url)
        
        # 3. Extract symbols (simplified - just functions and classes)
        chunks = []
        chunk_index = 0
        
        for py_file in Path(repo_path).glob('**/*.py'):
            # Skip large files and common excludes
            if self._should_skip(py_file):
                continue
            
            try:
                content = py_file.read_text()
                symbols = self.tree_sitter.extract_symbols(
                    content, 
                    language='python'
                )
                
                # Create chunks from symbols
                for symbol in symbols:
                    if symbol['kind'] in ['function', 'class']:
                        chunk = {
                            'repo_key': repo_key,
                            'chunk_index': chunk_index,
                            'file_path': str(py_file.relative_to(repo_path)),
                            'chunk_type': symbol['kind'],
                            'name': symbol['name'],
                            'text': symbol['text'][:5000],  # Limit size
                            'start_line': symbol['start_line'],
                            'end_line': symbol['end_line'],
                            'language': 'python'
                        }
                        chunks.append(chunk)
                        chunk_index += 1
                        
            except Exception as e:
                logger.warning(f"Failed to process {py_file}: {e}")
        
        # 4. Generate embeddings (batch like ArXiv)
        chunk_texts = [c['text'] for c in chunks]
        embeddings = self.embedder.embed_texts(chunk_texts, task="retrieval")
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding.tolist()
            chunk['embedding_dim'] = len(embedding)
        
        # 5. Store repository document
        repo_doc = {
            '_key': repo_key,
            '_id': f"base_github/{repo_key}",
            'uid': f"github:{owner}/{name}",
            'owner': owner,
            'name': name,
            'github_url': repo_url,
            'description': self._get_repo_description(repo_path),
            'last_commit': self._get_latest_commit(repo_url),
            'file_count': len(list(Path(repo_path).glob('**/*.py'))),
            'chunk_count': len(chunks),
            'language': 'Python',
            'processing_date': datetime.now(timezone.utc).isoformat(),
            'processor_version': 'v1.0'
        }
        
        self.db.collection('base_github').insert(repo_doc, overwrite=True)
        
        # 6. Store chunks
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_key = f"{repo_key}_chunk_{i}"
            chunk['_key'] = chunk_key
            chunk['_id'] = f"base_github_chunks/{chunk_key}"
            chunk['processing_date'] = datetime.now(timezone.utc).isoformat()
            
            self.db.collection('base_github_chunks').insert(chunk, overwrite=True)
            chunk_ids.append(f"base_github_chunks/{chunk_key}")
        
        # 7. Create containment edges (like ArXiv)
        for chunk_id in chunk_ids:
            edge_key = hashlib.sha256(
                f"base_github/{repo_key}|{chunk_id}|contains|system".encode()
            ).hexdigest()[:32]
            
            edge = {
                '_key': edge_key,
                '_from': f"base_github/{repo_key}",
                '_to': chunk_id,
                'relation': 'contains',
                'source_system': 'github_processor',
                'confidence': 1.0,
                'active': true,
                'extraction_run_id': datetime.now(timezone.utc).isoformat()
            }
            
            self.db.collection('edges_contains').insert(edge, overwrite=True)
        
        return {
            'status': 'success',
            'repo_key': repo_key,
            'chunk_count': len(chunks),
            'processing_time': time.time() - start_time
        }
    
    def _should_skip(self, file_path: Path) -> bool:
        """Skip files we shouldn't process."""
        skip_dirs = {'__pycache__', 'venv', 'node_modules', '.git'}
        skip_extensions = {'.pyc', '.pyo', '.so', '.dll'}
        
        # Check directory
        if any(part in skip_dirs for part in file_path.parts):
            return True
        
        # Check extension
        if file_path.suffix in skip_extensions:
            return True
        
        # Check size (skip > 100KB for Phase 1)
        if file_path.stat().st_size > 100_000:
            return True
        
        return False
```

---

## Phase 1 Execution Plan

### Step 1: Setup Collections (Today)

```python
# Run this to create collections
python -c "
from github_processor_phase1 import GitHubProcessorPhase1
processor = GitHubProcessorPhase1()
print('Collections created')
"
```

### Step 2: Test with Small Repo

```python
# Test with a small repository first
processor = GitHubProcessorPhase1()
result = processor.process_repository('https://github.com/jalammar/illustrated-word2vec')
print(f"Processed: {result['chunk_count']} chunks")
```

### Step 3: Verify Data Structure

```aql
// Check repository
FOR repo IN base_github
    FILTER repo._key == "jalammar__illustrated-word2vec"
    RETURN repo

// Check chunks
FOR chunk IN base_github_chunks
    FILTER chunk.repo_key == "jalammar__illustrated-word2vec"
    LIMIT 5
    RETURN chunk

// Check edges
FOR edge IN edges_contains
    FILTER edge._from == "base_github/jalammar__illustrated-word2vec"
    LIMIT 5
    RETURN edge
```

### Step 4: Test Search

```aql
// Vector search (should work immediately)
LET query_embedding = (
    FOR c IN base_github_chunks
    FILTER c.name == "train_word2vec"
    LIMIT 1
    RETURN c.embedding
)[0]

FOR chunk IN base_github_chunks
    SORT DISTANCE(chunk.embedding, query_embedding)
    LIMIT 10
    RETURN {
        name: chunk.name,
        file: chunk.file_path,
        type: chunk.chunk_type
    }
```

---

## What We're NOT Doing in Phase 1

1. **No file-level documents** - just repo and chunks
2. **No complex dependency parsing** - just requirements.txt
3. **No multi-language** - Python only initially  
4. **No delta updates** - full reprocess for now
5. **No implementation detection** - that's Phase 2
6. **No git history** - just latest commit

---

## Success Criteria

Phase 1 is complete when:

- [ ] Collections match ArXiv structure
- [ ] Can process a Python repository
- [ ] Chunks have Jina v4 embeddings
- [ ] Containment edges work
- [ ] Vector search returns results
- [ ] tree_sitter_extractor.py is integrated

---

## Files We Already Have

```
✅ tree_sitter_extractor.py    - Symbol extraction (ready to use)
✅ ../arxiv/core/jina_v4_embedder.py  - Embeddings (can import)
```

## Next Steps

1. Create `github_processor_phase1.py` with above code
2. Test with small repo (illustrated-word2vec)
3. Verify chunks and embeddings
4. Test vector search
5. Process 10 more repos
6. Then move to Phase 2 (delta updates, multi-language, etc.)

This gets us to exact parity with ArXiv v7.0 structure, making Phase 1.5 (edge materialization) possible across both processors.
