#!/usr/bin/env python3
"""
On-demand GitHub repository processor - Phase 1 implementation.
Simple clone, store, and embed without AST or theoretical analysis.

Theory Connection:
Base containers capture raw computational artifacts before interpretation.
Like collecting material culture before applying theoretical frameworks.
"""

import os
import sys
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse
import re

sys.path.append(str(Path(__file__).parent.parent.parent / "arxiv"))
from core.batch_embed_jina import JinaEmbedderV3

from arango import ArangoClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OnDemandRepoProcessor:
    """
    Simple repo processor - clone, store, embed.
    No analysis, no theory, just data storage.
    """
    
    # Common code extensions to process
    CODE_EXTENSIONS = [
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp',
        '.go', '.rs', '.rb', '.php', '.cs', '.swift', '.kt', '.scala', '.r',
        '.m', '.mm', '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat',
        '.sql', '.graphql', '.proto', '.thrift', '.avro'
    ]
    
    # Directories to skip
    SKIP_DIRS = [
        '.git', 'node_modules', 'vendor', 'dist', 'build', 'target',
        '__pycache__', '.pytest_cache', '.tox', 'venv', 'env',
        'site-packages', 'bower_components', '.next', '.nuxt'
    ]
    
    def __init__(self, db_host: str = "192.168.1.69", db_name: str = "academy_store"):
        self.db_host = db_host
        self.db_name = db_name
        self._init_database()
        
        # Initialize embedder - reuse from ArXiv
        self.embedder = JinaEmbedderV3(
            device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
            chunk_size=4000,  # Smaller chunks for code
            chunk_overlap=400
        )
        
        # Base directory for cloned repos
        self.repos_dir = Path('/tmp/github_repos')
        self.repos_dir.mkdir(exist_ok=True)
    
    def _init_database(self):
        """Initialize database connection."""
        password = os.environ.get('ARANGO_PASSWORD')
        if not password:
            raise ValueError("ARANGO_PASSWORD environment variable required")
        
        client = ArangoClient(hosts=f'http://{self.db_host}:8529')
        self.db = client.db(self.db_name, username='root', password=password)
        
        # Initialize collections
        self.repos_collection = self.db.collection('base_github_repos')
        self.files_collection = self.db.collection('base_github_files')
        self.embeddings_collection = self.db.collection('base_github_embeddings')
    
    def parse_github_url(self, repo_url: str) -> tuple:
        """Parse GitHub URL to extract owner and repo name."""
        # Handle various GitHub URL formats
        patterns = [
            r'github\.com[/:]([^/]+)/([^/\.]+)',
            r'([^/]+)/([^/]+)$'  # Simple owner/repo format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, repo_url)
            if match:
                owner, repo = match.groups()
                repo = repo.replace('.git', '')  # Remove .git suffix if present
                return owner, repo
        
        raise ValueError(f"Could not parse GitHub URL: {repo_url}")
    
    def clone_repo(self, repo_url: str, owner: str, name: str) -> Path:
        """Clone repository to local filesystem."""
        local_path = self.repos_dir / f"{owner}_{name}"
        
        # Remove existing directory if present
        if local_path.exists():
            logger.info(f"Removing existing repo at {local_path}")
            shutil.rmtree(local_path)
        
        # Clone the repository
        clone_url = repo_url if repo_url.startswith('http') else f"https://github.com/{repo_url}"
        logger.info(f"Cloning {clone_url} to {local_path}")
        
        try:
            result = subprocess.run(
                ['git', 'clone', '--depth', '1', clone_url, str(local_path)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Git clone failed: {result.stderr}")
            
            logger.info(f"Successfully cloned to {local_path}")
            return local_path
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Git clone timed out after 5 minutes")
        except Exception as e:
            raise RuntimeError(f"Failed to clone repository: {e}")
    
    def should_process_file(self, file_path: Path) -> bool:
        """Determine if file should be processed."""
        # Skip if in ignored directory
        for skip_dir in self.SKIP_DIRS:
            if skip_dir in file_path.parts:
                return False
        
        # Check if it's a code file
        return file_path.suffix.lower() in self.CODE_EXTENSIONS
    
    def detect_language(self, file_path: Path) -> str:
        """Simple language detection based on file extension."""
        ext_to_lang = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.cs': 'csharp',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.r': 'r',
            '.m': 'matlab',
            '.sh': 'shell',
            '.sql': 'sql'
        }
        return ext_to_lang.get(file_path.suffix.lower(), 'unknown')
    
    def chunk_code_simple(self, content: str, max_lines: int = 100) -> List[Dict]:
        """
        Simple code chunking by lines.
        Phase 1: No AST, just split on logical boundaries.
        """
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, line in enumerate(lines):
            current_chunk.append(line)
            current_size += 1
            
            # Create chunk if we hit size limit or end of file
            if current_size >= max_lines or i == len(lines) - 1:
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'start_line': i - len(current_chunk) + 2,  # 1-indexed
                        'end_line': i + 1
                    })
                    current_chunk = []
                    current_size = 0
        
        return chunks
    
    def process_repo(self, repo_url: str) -> Dict:
        """
        Basic processing: Clone → Store Files → Generate Embeddings
        """
        start_time = time.time()
        
        try:
            # Parse repository URL
            owner, name = self.parse_github_url(repo_url)
            repo_key = f"{owner}_{name}"
            
            # Check if already processed
            existing = self.repos_collection.get(repo_key)
            if existing and existing.get('clone_status') == 'processed':
                return {
                    'status': 'already_processed',
                    'repo': f"{owner}/{name}",
                    'message': 'Repository already processed'
                }
            
            # Clone repository
            local_path = self.clone_repo(repo_url, owner, name)
            
            # Get repository metadata
            repo_size = sum(f.stat().st_size for f in local_path.rglob('*') if f.is_file()) // 1024  # KB
            
            # Store repository metadata
            repo_doc = {
                '_key': repo_key,
                'github_url': f"https://github.com/{owner}/{name}",
                'clone_url': f"https://github.com/{owner}/{name}.git",
                'owner': owner,
                'name': name,
                'clone_status': 'cloned',
                'local_path': str(local_path),
                'size': repo_size,
                'processing_date': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            }
            
            # Try to get more metadata from git
            try:
                # Get default branch
                result = subprocess.run(
                    ['git', 'symbolic-ref', 'refs/remotes/origin/HEAD'],
                    cwd=local_path,
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    repo_doc['default_branch'] = result.stdout.strip().split('/')[-1]
                
                # Get last commit SHA
                result = subprocess.run(
                    ['git', 'rev-parse', 'HEAD'],
                    cwd=local_path,
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    repo_doc['last_commit_processed'] = result.stdout.strip()
                    
            except Exception as e:
                logger.warning(f"Could not get git metadata: {e}")
            
            # Insert or update repo document
            self.repos_collection.insert(repo_doc, overwrite=True)
            logger.info(f"Stored repository metadata for {repo_key}")
            
            # Process files
            files_processed = 0
            embeddings_generated = 0
            
            for file_path in local_path.rglob('*'):
                if not file_path.is_file():
                    continue
                
                if not self.should_process_file(file_path):
                    continue
                
                try:
                    # Read file content
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    
                    # Skip empty files
                    if not content.strip():
                        continue
                    
                    # Calculate relative path
                    relative_path = file_path.relative_to(local_path)
                    file_key = f"{repo_key}_{str(relative_path).replace('/', '_')}"
                    
                    # Store file content
                    file_doc = {
                        '_key': file_key,
                        'repo_key': repo_key,
                        'file_path': str(relative_path),
                        'file_name': file_path.name,
                        'file_extension': file_path.suffix,
                        'language': self.detect_language(file_path),
                        'size': len(content),
                        'lines': len(content.splitlines()),
                        'content': content,
                        'encoding': 'utf-8',
                        'file_type': 'source',  # Simple classification for now
                        'is_binary': False
                    }
                    
                    self.files_collection.insert(file_doc, overwrite=True)
                    files_processed += 1
                    
                    # Generate embeddings for code chunks
                    chunks = self.chunk_code_simple(content)
                    
                    for i, chunk in enumerate(chunks):
                        # Generate embedding using Jina
                        embedding = self.embedder.embed_batch([chunk['text']])[0]
                        
                        # Store embedding
                        embedding_doc = {
                            '_key': f"{file_key}_chunk_{i}",
                            'file_key': file_key,
                            'repo_key': repo_key,
                            'chunk_index': i,
                            'chunk_type': 'code',  # Simple type for now
                            'start_line': chunk['start_line'],
                            'end_line': chunk['end_line'],
                            'chunk_text': chunk['text'][:1000],  # Store first 1000 chars for reference
                            'embedding': embedding.tolist(),  # 2048-dimensional with Jina v4
                            'embedding_model': 'jinaai/jina-embeddings-v4',
                            'embedded_date': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                            'language': self.detect_language(file_path)
                        }
                        
                        self.embeddings_collection.insert(embedding_doc, overwrite=True)
                        embeddings_generated += 1
                    
                    logger.debug(f"Processed {file_path.name}: {len(chunks)} chunks")
                    
                except Exception as e:
                    logger.warning(f"Failed to process file {file_path}: {e}")
                    continue
            
            # Update repo status
            self.repos_collection.update_match(
                {'_key': repo_key},
                {'clone_status': 'processed', 'files_processed': files_processed, 'embeddings_generated': embeddings_generated}
            )
            
            processing_time = time.time() - start_time
            
            return {
                'status': 'success',
                'repo': f"{owner}/{name}",
                'files_processed': files_processed,
                'embeddings_generated': embeddings_generated,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Failed to process repository: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }


def main():
    """Process a GitHub repository on demand."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process GitHub repository on demand')
    parser.add_argument('repo_url', help='GitHub repository URL or owner/repo')
    parser.add_argument('--db-host', default='192.168.1.69')
    parser.add_argument('--db-name', default='academy_store')
    
    args = parser.parse_args()
    
    processor = OnDemandRepoProcessor(args.db_host, args.db_name)
    result = processor.process_repo(args.repo_url)
    
    if result['status'] == 'success':
        print(f"✅ Successfully processed {result['repo']}")
        print(f"   Files: {result['files_processed']}")
        print(f"   Embeddings: {result['embeddings_generated']}")
        print(f"   Time: {result['processing_time']:.2f}s")
    elif result['status'] == 'already_processed':
        print(f"ℹ️  {result['message']}")
    else:
        print(f"❌ Failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()