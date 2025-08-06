#!/usr/bin/env python3
"""
GitHub repository processor for theory-practice bridge discovery.
Implements selective cloning and embedding of code repositories.

Theory Connection:
This processor handles the PRACTICE dimension of our theory-practice bridges.
Code represents maximum CONVEYANCE - it's directly executable knowledge.
"""

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
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GitHubProcessor:
    """Process GitHub repositories for embedding and analysis."""
    
    def __init__(self, db_config: Dict):
        """
        Initialize processor with database and model configuration.
        
        Args:
            db_config: Database configuration dictionary
        """
        self.db = self._init_database(db_config)
        self.embedder = None  # Lazy load when GPU available
        self.tokenizer = None  # Lazy load
        self.device = None
        
    def _init_database(self, config: Dict):
        """Initialize database connection and ensure collections exist."""
        try:
            client = ArangoClient(hosts=config['host'])
            db = client.db(
                config['database'],
                username=config['username'],
                password=config['password']
            )
            
            # Ensure collections exist
            if not db.has_collection('base_github_repos'):
                db.create_collection('base_github_repos')
                logger.info("Created base_github_repos collection")
                
            if not db.has_collection('base_github_embeddings'):
                db.create_collection('base_github_embeddings')
                logger.info("Created base_github_embeddings collection")
                
            return db
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def _init_embedder(self):
        """
        Initialize Jina v4 embedder with fp16.
        Only call when GPU is available.
        """
        if self.embedder is not None:
            return  # Already initialized
            
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Initializing Jina v4 on {self.device}")
            
            self.embedder = AutoModel.from_pretrained(
                "jinaai/jina-embeddings-v4",
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            self.embedder.eval()
            self.tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v4")
            
            logger.info("Jina v4 initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {e}")
            raise
    
    def clone_repository(self, github_url: str, target_path: str = None) -> Dict:
        """
        Clone repository with full history.
        Includes rate limiting and disk space validation.
        
        Args:
            github_url: GitHub repository URL
            target_path: Optional target directory path
            
        Returns:
            Status dictionary with results
        """
        # Extract owner and name
        parts = github_url.rstrip('/').split('/')
        repo_name = parts[-1].replace('.git', '')
        owner = parts[-2]
        owner_repo_key = f"{owner}_{repo_name}"
        
        # Set clone path
        if not target_path:
            base_path = os.environ.get('GITHUB_CLONE_PATH', '/data/repos')
            target_path = f"{base_path}/{owner_repo_key}"
        
        # Check if already exists
        if Path(target_path).exists():
            logger.info(f"Repository already exists at {target_path}")
            return {
                "status": "exists",
                "path": target_path,
                "key": owner_repo_key
            }
        
        # Check disk space - 1GB minimum for MVP
        try:
            stat = shutil.disk_usage(Path(target_path).parent if Path(target_path).parent.exists() else "/")
            if stat.free < 1_000_000_000:
                logger.error("Insufficient disk space")
                return {
                    "status": "error",
                    "error": "Insufficient disk space (need 1GB minimum)"
                }
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
        
        # Rate limiting
        time.sleep(1.0)
        
        # Clone repository
        try:
            logger.info(f"Cloning {github_url} to {target_path}")
            repo = git.Repo.clone_from(github_url, target_path)
            
            # Calculate size
            size_bytes = sum(
                f.stat().st_size 
                for f in Path(target_path).rglob('*') 
                if f.is_file()
            )
            
            # Store metadata
            doc = {
                "_key": owner_repo_key,
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
            logger.info(f"Successfully cloned {owner_repo_key}: {size_bytes / (1024*1024):.2f} MB")
            
            return {
                "status": "success",
                "path": target_path,
                "key": owner_repo_key,
                "size_mb": size_bytes / (1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"Clone failed: {e}")
            # Clean up partial clone
            if Path(target_path).exists():
                shutil.rmtree(target_path)
            return {
                "status": "error",
                "error": str(e)
            }
    
    def embed_file(self, repo_key: str, file_path: str) -> Dict:
        """
        Generate embeddings for a single file.
        Requires GPU to be available.
        
        Args:
            repo_key: Repository key in database
            file_path: Path to file relative to repo root
            
        Returns:
            Status dictionary with results
        """
        # Initialize embedder if needed
        if self.embedder is None:
            self._init_embedder()
        
        # Get repository info
        repo_doc = self.db.collection('base_github_repos').get(repo_key)
        if not repo_doc:
            return {"status": "error", "error": "Repository not found"}
        
        # Read file
        full_path = Path(repo_doc['clone_path']) / file_path
        if not full_path.exists():
            return {"status": "error", "error": f"File not found: {file_path}"}
        
        try:
            content = full_path.read_text(encoding='utf-8')
        except Exception as e:
            return {"status": "error", "error": f"Cannot read file: {e}"}
        
        # Tokenize and chunk if needed
        tokens = self.tokenizer.encode(content)
        chunks = []
        
        if len(tokens) <= 28000:
            chunks = [content]
        else:
            # Simple chunking at 28k token boundaries
            for i in range(0, len(tokens), 28000):
                chunk_tokens = tokens[i:i+28000]
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk_text)
        
        # Generate embeddings with transaction
        embeddings_created = []
        
        # Start transaction for atomicity
        transaction_db = self.db.begin_transaction(
            read=['base_github_repos'],
            write=['base_github_embeddings', 'base_github_repos']
        )
        
        try:
            with torch.no_grad():
                for i, chunk in enumerate(chunks):
                    # Generate embedding using Jina v4
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
            
            # Update repository status
            repo_doc['embedding_status'] = 'partial'
            transaction_db.collection('base_github_repos').update(repo_doc)
            
            # Commit transaction
            transaction_db.commit()
            
            logger.info(f"Embedded {file_path}: {len(chunks)} chunks")
            
            return {
                "status": "success",
                "chunks": len(chunks),
                "embeddings": embeddings_created
            }
            
        except Exception as e:
            # Rollback on any error
            transaction_db.abort()
            logger.error(f"Embedding failed for {file_path}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "file": file_path
            }
    
    def embed_files_batch(self, repo_key: str, file_paths: List[str]) -> Dict:
        """
        Batch embed multiple files for efficiency.
        
        Args:
            repo_key: Repository key
            file_paths: List of file paths to embed
            
        Returns:
            Summary of batch processing
        """
        results = []
        
        for file_path in file_paths:
            # Rate limiting between files
            time.sleep(0.5)
            result = self.embed_file(repo_key, file_path)
            results.append(result)
            
        successful = sum(1 for r in results if r.get('status') == 'success')
        failed = sum(1 for r in results if r.get('status') == 'error')
        
        logger.info(f"Batch embedding complete: {successful} success, {failed} failed")
        
        return {
            "status": "success",
            "total_files": len(file_paths),
            "successful": successful,
            "failed": failed,
            "results": results
        }
    
    def list_code_files(self, repo_key: str, extensions: List[str] = None) -> List[str]:
        """
        List code files in repository.
        
        Args:
            repo_key: Repository key
            extensions: List of file extensions to include
            
        Returns:
            List of file paths relative to repo root
        """
        if not extensions:
            extensions = ['.py', '.js', '.c', '.cpp', '.java', '.go', '.rs', '.ts']
        
        repo_doc = self.db.collection('base_github_repos').get(repo_key)
        if not repo_doc:
            logger.error(f"Repository not found: {repo_key}")
            return []
        
        repo_path = Path(repo_doc['clone_path'])
        files = []
        
        for ext in extensions:
            found = [
                str(f.relative_to(repo_path))
                for f in repo_path.rglob(f'*{ext}')
                if f.is_file() and '.git' not in f.parts
            ]
            files.extend(found)
        
        logger.info(f"Found {len(files)} code files in {repo_key}")
        return files
    
    def _detect_primary_language(self, repo_path: str) -> str:
        """
        Detect primary language based on file extensions.
        
        Args:
            repo_path: Path to repository
            
        Returns:
            Primary language name
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
            '.rs': 'Rust',
            '.rb': 'Ruby',
            '.php': 'PHP'
        }
        
        primary_ext = max(extensions, key=extensions.get)
        return language_map.get(primary_ext, primary_ext[1:] if primary_ext else 'unknown')


def get_db_config() -> Dict:
    """Get database configuration from environment."""
    return {
        'host': os.environ.get('ARANGO_HOST', 'http://192.168.1.69:8529'),
        'database': os.environ.get('ARANGO_DATABASE', 'academy_store'),
        'username': os.environ.get('ARANGO_USERNAME', 'root'),
        'password': os.environ.get('ARANGO_PASSWORD')
    }


def process_word2vec_mvp():
    """
    Process Word2Vec MVP repositories.
    This is the main entry point for MVP validation.
    """
    WORD2VEC_REPOS = [
        "https://github.com/dav/word2vec",
        "https://github.com/tmikolov/word2vec",
        "https://github.com/danielfrg/word2vec",
        "https://github.com/RaRe-Technologies/gensim"
    ]
    
    processor = GitHubProcessor(get_db_config())
    
    for repo_url in WORD2VEC_REPOS:
        # Clone repository
        result = processor.clone_repository(repo_url)
        if result['status'] not in ['success', 'exists']:
            logger.error(f"Failed to clone {repo_url}: {result.get('error')}")
            continue
        
        repo_key = result['key']
        logger.info(f"Processing {repo_key}")
        
        # Get core implementation files
        if 'gensim' in repo_key:
            # For Gensim, only embed the word2vec module
            files = ['gensim/models/word2vec.py']
        else:
            # For others, embed all code files (limit to 10 for MVP)
            files = processor.list_code_files(repo_key)[:10]
        
        # Generate embeddings
        if files:
            batch_result = processor.embed_files_batch(repo_key, files)
            logger.info(f"Embedded {batch_result['successful']} files from {repo_key}")


if __name__ == "__main__":
    # This would be run after GPUs are available
    logger.info("GitHub processor ready. Run process_word2vec_mvp() when GPU available.")
    print("Note: GPU required for embedding generation. Currently in code-only mode.")