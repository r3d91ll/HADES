# GitHub Tool Specification

**Version**: 1.0.0  
**Status**: Implementation Ready  
**Scope**: GitHub repository processing with selective embedding

---

## 1. Overview

The GitHub tool enables selective cloning and embedding of code repositories for theory-practice bridge analysis. It provides cost estimation before resource commitment and uses Jina v4 for 2048-dimensional embeddings.

## 2. Database Schema

### base_github_repos

```javascript
{
  "_key": "owner_repo_name",           // e.g., "dav_word2vec"
  "github_url": "string",
  "clone_path": "string",               // Local storage path
  "clone_strategy": "full",             // Always full for MVP
  "clone_date": "datetime",
  "size_bytes": "integer",
  "default_branch": "string",
  "primary_language": "string",
  "embedding_status": "enum"            // not_embedded|partial|full
}
```

### base_github_embeddings

```javascript
{
  "_key": "md5_hash",                  // MD5 of repo_key + file_path + chunk_index
  "repo_key": "string",                // FK to base_github_repos
  "file_path": "string",               // Relative to repo root
  "chunk_index": "integer",            // 0 for files < 28k tokens
  "content_hash": "string",            // MD5 of content for deduplication
  "embedding": [2048],                 // Jina v4 embedding
  "tokens": "integer",
  "embedded_date": "datetime"
}
```

## 3. Core Implementation

### 3.1 Repository Processor

```python
import os
import git
import hashlib
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import torch
from transformers import AutoModel, AutoTokenizer
from arango import ArangoClient

class GitHubProcessor:
    def __init__(self, db_config: Dict):
        self.db = self._init_database(db_config)
        self.embedder = self._init_embedder()
        self.tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v4")
        
    def _init_database(self, config: Dict):
        client = ArangoClient(hosts=config['host'])
        db = client.db(config['database'], 
                       username=config['username'],
                       password=config['password'])
        
        # Ensure collections exist
        if not db.has_collection('base_github_repos'):
            db.create_collection('base_github_repos')
        if not db.has_collection('base_github_embeddings'):
            db.create_collection('base_github_embeddings')
            
        return db
    
    def _init_embedder(self):
        """Initialize Jina v4 with fp16."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v4",
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        model.eval()
        return model
    
    def clone_repository(self, github_url: str, target_path: str = None) -> Dict:
        """
        Clone repository with full history.
        Includes rate limiting and disk space validation.
        """
        # Extract owner and name
        parts = github_url.rstrip('/').split('/')
        owner_name = f"{parts[-2]}_{parts[-1]}"
        
        # Set clone path
        if not target_path:
            target_path = f"/data/repos/{owner_name}"
        
        # Check if already exists
        if Path(target_path).exists():
            return {
                "status": "exists",
                "path": target_path,
                "key": owner_name
            }
        
        # Check disk space - could be enhanced with repo-specific estimation
        # For MVP, 1GB is sufficient for all Word2Vec repos combined
        stat = shutil.disk_usage("/data")
        if stat.free < 1_000_000_000:  # Less than 1GB free
            return {
                "status": "error",
                "error": "Insufficient disk space"
            }
        
        # Rate limiting
        time.sleep(1.0)  # Basic rate limiting between operations
        
        # Clone repository
        try:
            repo = git.Repo.clone_from(github_url, target_path)
            size_bytes = sum(f.stat().st_size for f in Path(target_path).rglob('*') if f.is_file())
            
            # Store metadata
            doc = {
                "_key": owner_name,
                "github_url": github_url,
                "clone_path": target_path,
                "clone_strategy": "full",
                "clone_date": datetime.utcnow().isoformat(),
                "size_bytes": size_bytes,
                "default_branch": repo.active_branch.name,
                "primary_language": self._detect_primary_language(target_path),
                "embedding_status": "not_embedded"
            }
            
            self.db.collection('base_github_repos').insert(doc)
            
            return {
                "status": "success",
                "path": target_path,
                "key": owner_name,
                "size_mb": size_bytes / (1024 * 1024)
            }
            
        except Exception as e:
            # Clean up partial clone
            if Path(target_path).exists():
                shutil.rmtree(target_path)
            return {
                "status": "error",
                "error": str(e)
            }
    
    def embed_files_batch(self, repo_key: str, file_paths: List[str]) -> Dict:
        """
        Batch embed multiple files for efficiency.
        """
        results = []
        for file_path in file_paths:
            # Rate limiting between files
            time.sleep(0.5)
            result = self.embed_file(repo_key, file_path)
            results.append(result)
        
        return {
            "status": "success",
            "total_files": len(file_paths),
            "successful": sum(1 for r in results if r.get('status') == 'success'),
            "failed": sum(1 for r in results if r.get('status') == 'error'),
            "results": results
        }
    
    def embed_file(self, repo_key: str, file_path: str) -> Dict:
        """
        Generate embeddings for a single file.
        """
        # Get repository info
        repo_doc = self.db.collection('base_github_repos').get(repo_key)
        if not repo_doc:
            return {"status": "error", "error": "Repository not found"}
        
        # Read file
        full_path = Path(repo_doc['clone_path']) / file_path
        if not full_path.exists():
            return {"status": "error", "error": "File not found"}
        
        try:
            content = full_path.read_text()
        except:
            return {"status": "error", "error": "Cannot read file (binary or encoding issue)"}
        
        # Tokenize and chunk if needed
        tokens = self.tokenizer.encode(content)
        chunks = []
        
        if len(tokens) <= 28000:
            chunks = [content]
        else:
            # Simple chunking at 28k token boundaries
            for i in range(0, len(tokens), 28000):
                chunk_tokens = tokens[i:i+28000]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                chunks.append(chunk_text)
        
        # Generate embeddings with transaction for atomicity
        embeddings_created = []
        
        # Start transaction
        transaction_db = self.db.begin_transaction(
            read=['base_github_repos'],
            write=['base_github_embeddings', 'base_github_repos']
        )
        
        try:
            with torch.no_grad():
                for i, chunk in enumerate(chunks):
                    # Generate embedding using correct Jina v4 API
                    embeddings = self.embedder.encode(
                        [chunk],
                        batch_size=1,
                        show_progress_bar=False
                    )
                    
                    if torch.is_tensor(embeddings):
                        embedding = embeddings[0].cpu().numpy()
                    else:
                        embedding = embeddings[0]
                
                    # Create key using MD5 to avoid path issues
                    key_components = f"{repo_key}_{file_path}_{i}"
                    key_hash = hashlib.md5(key_components.encode()).hexdigest()
                    
                    # Store embedding
                    doc = {
                        "_key": key_hash,
                        "repo_key": repo_key,
                        "file_path": file_path,
                        "chunk_index": i,
                        "content_hash": hashlib.md5(chunk.encode()).hexdigest(),
                        "embedding": embedding.tolist(),
                        "tokens": len(self.tokenizer.encode(chunk)),
                        "embedded_date": datetime.utcnow().isoformat()
                    }
                    
                    transaction_db.collection('base_github_embeddings').insert(doc)
                    embeddings_created.append(key_hash)
            
            # Update repository status within transaction
            repo_doc['embedding_status'] = 'partial'
            transaction_db.collection('base_github_repos').update(repo_doc)
            
            # Commit transaction
            transaction_db.commit()
            
            return {
                "status": "success",
                "chunks": len(chunks),
                "embeddings": embeddings_created
            }
            
        except Exception as e:
            # Rollback on any error
            transaction_db.abort()
            return {
                "status": "error",
                "error": str(e),
                "file": file_path
            }
    
    def list_code_files(self, repo_key: str, extensions: List[str] = None) -> List[str]:
        """
        List code files in repository.
        """
        if not extensions:
            extensions = ['.py', '.js', '.c', '.cpp', '.java', '.go', '.rs']
        
        repo_doc = self.db.collection('base_github_repos').get(repo_key)
        if not repo_doc:
            return []
        
        repo_path = Path(repo_doc['clone_path'])
        files = []
        
        for ext in extensions:
            files.extend([
                str(f.relative_to(repo_path)) 
                for f in repo_path.rglob(f'*{ext}')
                if f.is_file() and '.git' not in f.parts
            ])
        
        return files
    
    def _detect_primary_language(self, repo_path: str) -> str:
        """
        Detect primary language based on file extensions.
        """
        extensions = {}
        path = Path(repo_path)
        
        for f in path.rglob('*'):
            if f.is_file() and '.git' not in f.parts:
                ext = f.suffix.lower()
                if ext:
                    extensions[ext] = extensions.get(ext, 0) + 1
        
        if not extensions:
            return "unknown"
        
        # Map to languages
        language_map = {
            '.py': 'Python',
            '.js': 'JavaScript', 
            '.ts': 'TypeScript',
            '.c': 'C',
            '.cpp': 'C++',
            '.java': 'Java',
            '.go': 'Go',
            '.rs': 'Rust'
        }
        
        primary_ext = max(extensions, key=extensions.get)
        return language_map.get(primary_ext, primary_ext[1:])
```

### 3.2 Configuration Helper

```python
def get_db_config() -> Dict:
    """Get database configuration from environment."""
    return {
        'host': os.environ.get('ARANGO_HOST', 'http://192.168.1.69:8529'),
        'database': os.environ.get('ARANGO_DATABASE', 'academy_store'),
        'username': os.environ.get('ARANGO_USERNAME', 'root'),
        'password': os.environ.get('ARANGO_PASSWORD')
    }
```

### 3.3 MCP Tool Wrappers

```python
# Note: In actual implementation, move this import to top of file
# from mcp import Server
# server = Server("github-tool")

@server.tool()
async def github_clone(github_url: str) -> Dict:
    """Clone a GitHub repository."""
    processor = GitHubProcessor(get_db_config())
    return processor.clone_repository(github_url)

@server.tool()
async def github_list_files(repo_key: str, extensions: List[str] = None) -> List[str]:
    """List code files in repository."""
    processor = GitHubProcessor(get_db_config())
    return processor.list_code_files(repo_key, extensions)

@server.tool()
async def github_embed_file(repo_key: str, file_path: str) -> Dict:
    """Generate embeddings for a file."""
    processor = GitHubProcessor(get_db_config())
    return processor.embed_file(repo_key, file_path)

@server.tool()
async def github_search_similar(embedding_key: str, limit: int = 10) -> List[Dict]:
    """Find similar code using embeddings."""
    processor = GitHubProcessor(get_db_config())
    
    # Get the embedding
    doc = processor.db.collection('base_github_embeddings').get(embedding_key)
    if not doc:
        return []
    
    # AQL query for similarity search
    query = """
    FOR doc IN base_github_embeddings
        LET similarity = COSINE_SIMILARITY(@embedding, doc.embedding)
        FILTER doc._key != @key
        SORT similarity DESC
        LIMIT @limit
        RETURN {
            key: doc._key,
            repo: doc.repo_key,
            file: doc.file_path,
            similarity: similarity
        }
    """
    
    cursor = processor.db.aql.execute(
        query,
        bind_vars={
            'embedding': doc['embedding'],
            'key': embedding_key,
            'limit': limit
        }
    )
    
    return list(cursor)
```

## 4. MVP Implementation

### Word2Vec Repositories

```python
WORD2VEC_REPOS = [
    "https://github.com/dav/word2vec",           # 2.3 MB
    "https://github.com/tmikolov/word2vec",      # ~5 MB
    "https://github.com/danielfrg/word2vec",     # ~10 MB
    "https://github.com/RaRe-Technologies/gensim" # ~450 MB (full)
]

def process_word2vec_mvp():
    """Process Word2Vec MVP repositories."""
    processor = GitHubProcessor(get_db_config())
    
    for repo_url in WORD2VEC_REPOS:
        # Clone repository
        result = processor.clone_repository(repo_url)
        if result['status'] != 'success':
            print(f"Failed to clone {repo_url}: {result.get('error')}")
            continue
        
        repo_key = result['key']
        
        # Get core implementation files
        if 'gensim' in repo_key:
            # For Gensim, only embed the word2vec module
            files = ['gensim/models/word2vec.py']
        else:
            # For others, embed all code files
            files = processor.list_code_files(repo_key)[:10]  # Limit to 10 files
        
        # Generate embeddings
        for file_path in files:
            embed_result = processor.embed_file(repo_key, file_path)
            print(f"Embedded {file_path}: {embed_result['status']}")
```

## 5. Resource Requirements

### Storage

- dav/word2vec: ~2.3 MB
- tmikolov/word2vec: ~5 MB
- danielfrg/word2vec: ~10 MB
- gensim (main only): ~450 MB
- **Repository Total**: ~467 MB
- Embeddings: ~10 MB (approximately 100 files)
- Database overhead: ~5 MB
- **Total**: ~482 MB

### Processing Time

- Repository cloning: ~5 minutes total
- Embedding generation: ~10 minutes (100 files)
- **Total**: ~15 minutes

### GPU Requirements

- Jina v4 with fp16: ~4GB base model
- Runtime overhead: ~2GB
- **Total Required**: ~6GB VRAM
- Single A6000 (48GB) provides ample headroom

## 6. Configuration

```yaml
# config.yaml
database:
  host: "http://192.168.1.69:8529"
  username: "root"
  password: "${ARANGO_PASSWORD}"
  database: "academy_store"

storage:
  repo_base_path: "/data/repos"
  
embedding:
  model: "jinaai/jina-embeddings-v4"
  max_tokens: 28000
  batch_size: 8
  
github:
  rate_limit_delay: 1.0  # seconds between API calls
```

## 7. Error Handling

- **Clone failures**: Return error status, no partial state
- **Embedding failures**: Skip file, log error, continue
- **Database failures**: Retry with exponential backoff
- **File encoding issues**: Skip binary/non-UTF8 files

## 8. Testing

```python
# Test configuration
test_config = {
    'host': 'http://localhost:8529',
    'database': 'test_db',
    'username': 'root',
    'password': 'test'
}

def test_clone_repository():
    """Test repository cloning."""
    processor = GitHubProcessor(test_config)
    result = processor.clone_repository("https://github.com/dav/word2vec")
    assert result['status'] == 'success'
    assert Path(result['path']).exists()

def test_embed_file():
    """Test file embedding."""
    processor = GitHubProcessor(test_config)
    # Assumes repository already cloned
    result = processor.embed_file("dav_word2vec", "word2vec.c")
    assert result['status'] == 'success'
    assert result['chunks'] >= 1

def test_key_generation():
    """Test MD5 key handles all characters."""
    test_paths = [
        "src/main.py",
        "path/with spaces/file.js",
        "path/with/many/slashes/file.go"
    ]
    for path in test_paths:
        key = hashlib.md5(f"repo_{path}_0".encode()).hexdigest()
        assert len(key) == 32
        assert '/' not in key
```

---

**END OF SPECIFICATION**
