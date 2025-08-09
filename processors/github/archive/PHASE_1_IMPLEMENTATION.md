# GitHub Processor - Phase 1 Implementation

## Phase 1 Goal: Clean, Versioned Data Layer

Get faithful data into `base_github` with enough hooks for Phase 2 analytics without over-engineering.

## Core Schema (Phase 1 Minimum)

### Stable UID Pattern
```python
# Repos
uid: "github:owner/repo@{branch_or_sha}"
# Example: "github:tmikolov/word2vec@main"

# Files  
uid: "github:owner/repo@{commit_sha}#path/to/file.py"
# Example: "github:tmikolov/word2vec@abc123#src/word2vec.c"

# Symbols
uid: "github:owner/repo@{commit_sha}#path/to/file.py::function_name(signature_hash)"
# Example: "github:tmikolov/word2vec@abc123#src/word2vec.c::train_word2vec(a1b2c3)"
```

### Document Types (kind field)

```python
# Repo document
{
    "_key": "tmikolov_word2vec",
    "uid": "github:tmikolov/word2vec@main",
    "kind": "repo",
    "repository": "github",
    
    # Core metadata
    "github_url": "https://github.com/tmikolov/word2vec",
    "owner": "tmikolov",
    "name": "word2vec",
    "default_branch": "main",
    "head_sha": "abc123def456",
    "clone_status": "processed",
    
    # Size controls
    "size_kb": 2300,
    "file_count": 15,
    "language_stats": {"c": 0.85, "python": 0.15},
    
    # Provenance
    "processing_date": "2025-01-21T...",
    "processor_version": "v1.0",
    "ingest_run_id": "uuid..."
}

# File document
{
    "_key": "tmikolov_word2vec_src_word2vec_c",
    "uid": "github:tmikolov/word2vec@abc123#src/word2vec.c",
    "kind": "file",
    "repository": "github",
    
    # Identity
    "repo_id": "tmikolov_word2vec",
    "file_path": "src/word2vec.c",
    "commit_sha": "abc123def456",
    "blob_sha": "fedcba987654",  # Git blob SHA
    "content_hash": "sha256:...",  # Our hash of content
    
    # Metadata
    "language": "c",
    "size": 45678,
    "lines": 1234,
    "is_binary": false,
    "is_lfs": false,
    
    # Content (optional for Phase 1)
    "content": "...",  # Can defer if too large
    
    # Light extraction
    "imports": ["stdio.h", "math.h", "pthread.h"],
    "references": [],  # Papers mentioned
    
    # Provenance
    "processing_date": "2025-01-21T...",
    "ingest_run_id": "uuid..."
}

# Symbol document (minimal for Phase 1)
{
    "_key": "tmikolov_word2vec_src_word2vec_c_train_word2vec",
    "uid": "github:tmikolov/word2vec@abc123#src/word2vec.c::train_word2vec(a1b2c3)",
    "kind": "symbol",
    "repository": "github",
    
    # Identity
    "repo_id": "tmikolov_word2vec",
    "file_path": "src/word2vec.c",
    "commit_sha": "abc123def456",
    "file_blob_sha": "fedcba987654",
    
    # Symbol info
    "language": "c",
    "symbol_type": "function",
    "symbol_name": "train_word2vec",
    "signature": "void train_word2vec(char *train_file, char *output_file)",
    "docstring": null,  # C doesn't have docstrings
    
    # Location
    "start_byte": 12345,
    "end_byte": 23456,
    "start_line": 234,
    "end_line": 567,
    
    # Light extraction
    "imports": [],  # Local imports this function uses
    "references": [],  # Papers mentioned in comments
    
    # Embedding placeholder
    "embedding": null,  # Can add in Phase 1.5
    "embedding_method": null,
    "embedding_model": null,
    
    # Provenance
    "processing_date": "2025-01-21T...",
    "ingest_run_id": "uuid..."
}
```

## Minimal Indexes (Phase 1)

```javascript
// Create these in setup script
db.collection('base_github').ensureIndex({
  type: 'persistent',
  fields: ['uid'],
  unique: true,
  sparse: false,
  name: 'idx_uid'
});

db.collection('base_github').ensureIndex({
  type: 'persistent', 
  fields: ['kind', 'repo_id'],
  unique: false,
  sparse: true,
  name: 'idx_kind_repo'
});

db.collection('base_github').ensureIndex({
  type: 'persistent',
  fields: ['kind', 'repo_id', 'file_path'],
  unique: true,
  sparse: true,
  name: 'idx_kind_repo_file'
});

db.collection('base_github').ensureIndex({
  type: 'persistent',
  fields: ['kind', 'file_blob_sha'],
  unique: false,
  sparse: true,
  name: 'idx_kind_blob'
});

db.collection('base_github').ensureIndex({
  type: 'persistent',
  fields: ['kind', 'symbol_name'],
  unique: false,
  sparse: true,
  name: 'idx_kind_symbol'
});

// For paper → code reverse lookup
db.collection('base_github').ensureIndex({
  type: 'persistent',
  fields: ['references[*]'],
  unique: false,
  sparse: true,
  name: 'idx_references'
});
```

## Clone Strategy (Phase 1)

```python
def clone_repository(repo_url: str) -> Path:
    """
    Shallow clone with size limits.
    """
    # Parse URL
    owner, name = parse_github_url(repo_url)
    
    # Shallow clone (depth=1)
    cmd = [
        'git', 'clone',
        '--depth', '1',
        '--single-branch',
        '--no-tags',
        repo_url,
        local_path
    ]
    
    # Skip if too large
    if get_repo_size_estimate(owner, name) > MAX_REPO_SIZE_MB:
        return None
        
    # Clone
    subprocess.run(cmd, timeout=300)
    
    # Get metadata
    head_sha = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'],
        cwd=local_path
    ).strip()
    
    return local_path, head_sha
```

## File Processing (Phase 1)

```python
# Skip these directories
SKIP_DIRS = {
    '.git', 'node_modules', 'vendor', 'dist', 'build', 'target',
    '__pycache__', '.pytest_cache', 'venv', 'env', '.tox',
    'site-packages', 'bower_components', '.next', '.nuxt'
}

# Skip these extensions  
SKIP_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg',
    '.pdf', '.doc', '.docx', '.xls', '.xlsx',
    '.zip', '.tar', '.gz', '.rar', '.7z',
    '.exe', '.dll', '.so', '.dylib',
    '.pyc', '.pyo', '.pyd', '.whl',
    '.jar', '.war', '.ear', '.class',
    '.o', '.a', '.lib', '.obj',
    '.pt', '.pth', '.ckpt', '.h5', '.pb', '.onnx',
    '.db', '.sqlite', '.sqlite3'
}

def should_process_file(path: Path) -> bool:
    # Skip if in ignored directory
    if any(skip in path.parts for skip in SKIP_DIRS):
        return False
    
    # Skip binary/large files
    if path.suffix.lower() in SKIP_EXTENSIONS:
        return False
        
    # Skip if too large (>2MB)
    if path.stat().st_size > 2 * 1024 * 1024:
        return False
        
    # Skip if Git LFS pointer
    if is_lfs_pointer(path):
        return False
        
    return True
```

## Content Hash Guard

```python
def needs_processing(file_path: Path, commit_sha: str) -> bool:
    """
    Check if file needs processing using blob SHA.
    """
    # Get git blob SHA
    blob_sha = subprocess.check_output(
        ['git', 'hash-object', str(file_path)],
        cwd=repo_path
    ).strip()
    
    # Check if already processed
    existing = db.collection('base_github').find_one({
        'kind': 'file',
        'file_blob_sha': blob_sha
    })
    
    if existing:
        logger.info(f"Skipping {file_path} - blob unchanged")
        return False
        
    return True
```

## Tree-sitter Extraction (Minimal Phase 1)

```python
class TreeSitterSymbolExtractor:
    """
    Minimal symbol extraction for Phase 1.
    Just get functions, classes, and imports.
    """
    
    def __init__(self):
        self.parsers = {}
        self._init_parsers()
    
    def _init_parsers(self):
        # Load language parsers
        import tree_sitter_python as tspython
        import tree_sitter_javascript as tsjavascript
        import tree_sitter_c as tsc
        
        from tree_sitter import Language, Parser
        
        # Build languages
        PY_LANGUAGE = Language(tspython.language(), "python")
        JS_LANGUAGE = Language(tsjavascript.language(), "javascript")
        C_LANGUAGE = Language(tsc.language(), "c")
        
        # Create parsers
        for lang, language in [
            ('python', PY_LANGUAGE),
            ('javascript', JS_LANGUAGE),
            ('c', C_LANGUAGE)
        ]:
            parser = Parser()
            parser.set_language(language)
            self.parsers[lang] = parser
    
    def extract_symbols(self, content: str, language: str) -> List[Dict]:
        """
        Extract symbols from code.
        """
        if language not in self.parsers:
            return []
            
        parser = self.parsers[language]
        tree = parser.parse(bytes(content, 'utf-8'))
        
        symbols = []
        
        # Language-specific queries
        if language == 'python':
            symbols.extend(self._extract_python_symbols(tree, content))
        elif language == 'javascript':
            symbols.extend(self._extract_js_symbols(tree, content))
        elif language == 'c':
            symbols.extend(self._extract_c_symbols(tree, content))
            
        return symbols
    
    def _extract_python_symbols(self, tree, content):
        # Simple extraction - just functions and classes
        symbols = []
        
        # Query for functions
        query = """
        (function_definition
          name: (identifier) @name
          parameters: (parameters) @params) @func
        """
        
        # ... parse and extract ...
        
        return symbols
```

## Reference Extraction (Light Phase 1)

```python
def extract_references(content: str) -> List[str]:
    """
    Extract paper references from code/comments.
    """
    references = []
    
    # ArXiv pattern
    arxiv_pattern = r'(?:arxiv[:\s]+)?(\d{4}\.\d{4,5})'
    for match in re.finditer(arxiv_pattern, content, re.IGNORECASE):
        references.append(f"arxiv:{match.group(1)}")
    
    # DOI pattern
    doi_pattern = r'10\.\d{4,}/[-._;()/:\w]+'
    for match in re.finditer(doi_pattern, content):
        references.append(f"doi:{match.group(0)}")
    
    # Look for CITATION.cff
    if 'CITATION.cff' in content or 'citation' in content.lower():
        # Extract more carefully
        pass
        
    return list(set(references))
```

## Three-Phase Write (Like ArXiv v6.7)

```python
def write_to_database(repo_doc, file_docs, symbol_docs):
    """
    Three-phase atomic-ish write.
    """
    repo_id = repo_doc['_key']
    commit_sha = repo_doc['head_sha']
    
    try:
        # Phase 1: Clean old data for this commit
        db.aql.execute("""
            FOR doc IN base_github
                FILTER doc.kind IN ['file', 'symbol']
                FILTER doc.repo_id == @repo_id
                FILTER doc.commit_sha != @commit_sha
                REMOVE doc IN base_github
        """, bind_vars={'repo_id': repo_id, 'commit_sha': commit_sha})
        
        # Phase 2: Upsert repo
        db.collection('base_github').insert(repo_doc, overwrite=True)
        
        # Phase 3: Insert files and symbols
        if file_docs:
            db.collection('base_github').insert_many(file_docs, overwrite=True)
        if symbol_docs:
            db.collection('base_github').insert_many(symbol_docs, overwrite=True)
            
        logger.info(f"Wrote {len(file_docs)} files, {len(symbol_docs)} symbols")
        
    except Exception as e:
        logger.error(f"Write failed: {e}")
        raise
```

## Main Processing Pipeline

```python
def process_repository(repo_url: str):
    """
    Phase 1 pipeline: Clone → Parse → Extract → Store
    """
    # Clone
    local_path, head_sha = clone_repository(repo_url)
    owner, name = parse_github_url(repo_url)
    
    # Create repo document
    repo_doc = {
        '_key': f"{owner}_{name}",
        'uid': f"github:{owner}/{name}@{head_sha}",
        'kind': 'repo',
        'repository': 'github',
        'github_url': f"https://github.com/{owner}/{name}",
        'owner': owner,
        'name': name,
        'head_sha': head_sha,
        'processing_date': datetime.utcnow().isoformat(),
        'processor_version': 'v1.0',
        'ingest_run_id': str(uuid.uuid4())
    }
    
    file_docs = []
    symbol_docs = []
    
    # Process files
    for file_path in local_path.rglob('*'):
        if not file_path.is_file():
            continue
            
        if not should_process_file(file_path):
            continue
            
        # Check blob SHA
        blob_sha = get_blob_sha(file_path)
        if not needs_processing(blob_sha):
            continue
            
        # Read content
        content = file_path.read_text(errors='ignore')
        
        # Create file document
        file_doc = {
            '_key': generate_file_key(owner, name, file_path),
            'uid': f"github:{owner}/{name}@{head_sha}#{file_path}",
            'kind': 'file',
            'repository': 'github',
            'repo_id': f"{owner}_{name}",
            'file_path': str(file_path.relative_to(local_path)),
            'commit_sha': head_sha,
            'blob_sha': blob_sha,
            'content_hash': hashlib.sha256(content.encode()).hexdigest(),
            'language': detect_language(file_path),
            'size': len(content),
            'lines': len(content.splitlines()),
            'references': extract_references(content),
            'processing_date': datetime.utcnow().isoformat(),
            'ingest_run_id': repo_doc['ingest_run_id']
        }
        file_docs.append(file_doc)
        
        # Extract symbols (if code file)
        if file_doc['language'] in SUPPORTED_LANGUAGES:
            symbols = extractor.extract_symbols(content, file_doc['language'])
            for symbol in symbols:
                symbol_doc = {
                    '_key': generate_symbol_key(owner, name, file_path, symbol),
                    'uid': f"github:{owner}/{name}@{head_sha}#{file_path}::{symbol['name']}({symbol['sig_hash']})",
                    'kind': 'symbol',
                    'repository': 'github',
                    'repo_id': f"{owner}_{name}",
                    'file_path': str(file_path.relative_to(local_path)),
                    'commit_sha': head_sha,
                    'file_blob_sha': blob_sha,
                    **symbol,  # name, type, signature, etc.
                    'processing_date': datetime.utcnow().isoformat(),
                    'ingest_run_id': repo_doc['ingest_run_id']
                }
                symbol_docs.append(symbol_doc)
    
    # Write to database
    write_to_database(repo_doc, file_docs, symbol_docs)
    
    # Clean up
    shutil.rmtree(local_path)
    
    return {
        'status': 'success',
        'repo': f"{owner}/{name}",
        'files': len(file_docs),
        'symbols': len(symbol_docs)
    }
```

## What We Defer to Phase 2

1. **Embeddings** - Can add later to existing symbol docs
2. **Call graphs** - Can build from imports arrays
3. **Dependency graphs** - Can materialize from manifest files
4. **Bridge scoring** - Lives in analytical containers
5. **History/commits** - Can add version tracking later
6. **Advanced extraction** - Complexity metrics, etc.

## Success Criteria for Phase 1

✅ Can clone and process Word2Vec repos
✅ Symbols extracted and stored with stable UIDs
✅ Content-hash guard prevents reprocessing
✅ References field captures paper mentions
✅ Indexes support Phase 2 queries
✅ Schema won't need surgery for Phase 2

---

*This focused Phase 1 gets clean data in without overengineering. Phase 2 can layer on embeddings, scoring, and analytics.*