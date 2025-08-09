#!/usr/bin/env python3
"""
GitHub repository processor - Phase 1 implementation (v5 - Production Ready).
Incorporates all reviewer feedback for robust production use.

Theory Connection:
Code repositories represent maximum CONVEYANCE - directly executable knowledge.
Each commit SHA creates an immutable observation point in the evolution of practice.
Jina v4 embeddings preserve semantic relationships for theory-practice bridge discovery.
"""

import os
import sys
import hashlib
import time
import shutil
import subprocess
import gzip
import uuid
import yaml
import re
import signal
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np

from arango import ArangoClient

# Add parent directories for imports
sys.path.append(str(Path(__file__).parent.parent / "arxiv"))
sys.path.append(str(Path(__file__).parent.parent / "arxiv" / "core"))

# Import Jina v4 embedder - CRITICAL for bridge discovery
from jina_v4_embedder import JinaV4Embedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GitHubProcessorV5:
    """
    Production-ready GitHub processor with all high-impact fixes.
    - Proper embedding-document alignment
    - Adaptive batch sizes
    - Enhanced vector metadata
    - GitHub token support
    - Graceful shutdown
    """
    
    # Skip these directories
    SKIP_DIRS = {
        '.git', 'node_modules', 'vendor', 'dist', 'build', 'target',
        '__pycache__', '.pytest_cache', 'venv', 'env', '.tox',
        'site-packages', 'bower_components', '.next', '.nuxt',
        '.venv', 'virtualenv', '.env', 'coverage', '.coverage',
        '.gitmodules'  # Skip submodule definitions
    }
    
    # Skip these extensions
    SKIP_EXTENSIONS = {
        # Images
        '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.bmp', '.tiff',
        # Documents
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        # Archives
        '.zip', '.tar', '.gz', '.rar', '.7z', '.bz2', '.xz',
        # Binaries
        '.exe', '.dll', '.so', '.dylib', '.app', '.deb', '.rpm',
        # Compiled
        '.pyc', '.pyo', '.pyd', '.whl', '.egg', '.gem',
        '.jar', '.war', '.ear', '.class',
        '.o', '.a', '.lib', '.obj', '.ko',
        # Models/Data
        '.pt', '.pth', '.ckpt', '.h5', '.pb', '.onnx', '.safetensors',
        '.db', '.sqlite', '.sqlite3', '.pkl', '.pickle',
        # Media
        '.mp4', '.mp3', '.wav', '.flac', '.avi', '.mov'
    }
    
    # Supported languages for symbol extraction
    SUPPORTED_LANGUAGES = {
        '.py': 'python',
        '.js': 'javascript', 
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.c': 'c',
        '.h': 'c',
        '.cpp': 'cpp',
        '.hpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.java': 'java',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.cs': 'csharp',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.r': 'r',
        '.R': 'r',
        '.sh': 'bash',
        '.bash': 'bash',
        '.zsh': 'zsh',
        '.lua': 'lua'
    }
    
    # Batch sizes - adaptive based on content
    UPSERT_BATCH_SIZE = 500           # For non-embedded documents
    EMBEDDED_BATCH_SIZE = 50           # For documents with embeddings
    EMBEDDING_BATCH_SIZE = 4           # For GPU processing
    
    # Inline content budget per repo (5MB)
    MAX_INLINE_BUDGET = 5 * 1024 * 1024
    
    def __init__(self, 
                 db_host: str = "localhost",
                 db_port: int = 8529,
                 db_name: str = "academy_store",
                 collection: str = "base_github",
                 max_repo_size_mb: int = 500,
                 max_file_size_mb: int = 2,
                 content_store_dir: str = None,
                 dry_run: bool = False,
                 no_embed: bool = False):
        """
        Initialize processor with database configuration.
        
        Args:
            db_host: ArangoDB host
            db_port: ArangoDB port  
            db_name: Database name (academy_store for production)
            collection: Collection name
            max_repo_size_mb: Maximum repository size to process
            max_file_size_mb: Maximum file size to process
            content_store_dir: Directory for persistent content storage
            dry_run: If True, don't write to database
            no_embed: If True, skip embedding generation
        """
        self.db_host = db_host
        self.db_port = db_port
        self.db_name = db_name
        self.collection_name = collection
        self.max_repo_size_mb = max_repo_size_mb
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.dry_run = dry_run
        self.no_embed = no_embed
        
        # Print effective configuration (redact password)
        logger.info(f"Configuration: db={db_name}@{db_host}:{db_port}, collection={collection}")
        logger.info(f"Options: dry_run={dry_run}, no_embed={no_embed}, max_repo={max_repo_size_mb}MB")
        
        # Initialize Jina v4 embedder if needed
        if not no_embed:
            logger.info("Initializing Jina v4 embedder...")
            self.embedder = JinaV4Embedder(
                device="cuda" if os.getenv('CUDA_VISIBLE_DEVICES') else "cpu",
                use_fp16=True
            )
            logger.info("Jina v4 embedder ready")
        else:
            self.embedder = None
            logger.info("Embeddings disabled (--no-embed)")
        
        # Persistent content store (not in /tmp)
        if content_store_dir:
            self.content_store = Path(content_store_dir)
            self.content_store.mkdir(parents=True, exist_ok=True)
            logger.info(f"Content store: {self.content_store}")
        else:
            self.content_store = None
        
        # Initialize database connection
        if not dry_run:
            password = os.getenv('ARANGO_PASSWORD')
            if not password:
                raise ValueError("ARANGO_PASSWORD environment variable not set")
            
            self.client = ArangoClient(hosts=f'http://{db_host}:{db_port}')
            self.db = self.client.db(db_name, username='root', password=password)
            
            # Create collection if needed
            if not self.db.has_collection(collection):
                self.collection = self.db.create_collection(collection)
                logger.info(f"Created {collection} collection in {db_name}")
            else:
                self.collection = self.db.collection(collection)
                logger.info(f"Using existing {collection} collection in {db_name}")
            
            # Setup indexes
            self._setup_indexes()
        
        # Repository directories 
        self.repos_dir = Path("/tmp/github_repos")
        self.repos_dir.mkdir(exist_ok=True)
        
        # Shutdown flag for graceful termination
        self.shutdown_requested = False
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Track current clone for cleanup
        self.current_clone_dir = None
        
        # Initialize tree-sitter if available
        try:
            from tree_sitter_extractor import TreeSitterSymbolExtractor
            self.symbol_extractor = TreeSitterSymbolExtractor()
            logger.info("Tree-sitter symbol extractor initialized")
        except ImportError:
            self.symbol_extractor = None
            logger.warning("Tree-sitter not available, symbols won't be extracted")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, requesting shutdown...")
        self.shutdown_requested = True
        # Don't exit immediately - let current operation complete
    
    def _cleanup_handler(self):
        """Clean up resources on exit."""
        if self.current_clone_dir and self.current_clone_dir.exists():
            logger.info(f"Cleaning up clone directory: {self.current_clone_dir}")
            shutil.rmtree(self.current_clone_dir, ignore_errors=True)
    
    def _setup_indexes(self):
        """Setup required indexes for efficient queries."""
        # Define required indexes with proper configuration
        indexes = [
            {"name": "idx_uid", "fields": ["uid"], "unique": True},
            {"name": "idx_kind_repo", "fields": ["kind", "repo_id"]},
            {"name": "idx_kind_repo_file", "fields": ["kind", "repo_id", "file_path"]},
            {"name": "idx_kind_blob", "fields": ["kind", "file_blob_sha"]},
            {"name": "idx_kind_repo_blob", "fields": ["kind", "repo_id", "file_blob_sha"]},
            {"name": "idx_kind_lang_symbol", "fields": ["kind", "language", "symbol_type"]},
            {"name": "idx_kind_commit", "fields": ["kind", "commit_sha"]},
            {"name": "idx_references", "fields": ["references[*]"], "sparse": True}
        ]
        
        # Get existing indexes
        existing = {idx['name']: idx for idx in self.collection.indexes()}
        
        # Create missing indexes (fixed mutation issue)
        for index_spec in indexes:
            idx = dict(index_spec)  # Clone to avoid mutation
            name = idx.pop("name")
            
            if name not in existing:
                try:
                    self.collection.add_persistent_index(**idx)
                    logger.info(f"Created index: {name}")
                except Exception as e:
                    logger.warning(f"Failed to create index {name}: {e}")
    
    def _parse_github_url(self, url: str) -> Tuple[str, str]:
        """Parse GitHub URL to extract owner and repo name."""
        # Handle owner/repo format
        if '/' in url and not url.startswith('http'):
            parts = url.split('/')
            return parts[0], parts[1]
        
        # Handle full URLs
        if 'github.com' in url:
            match = re.search(r'github\.com[:/]([^/]+)/([^/.]+)', url)
            if match:
                return match.group(1), match.group(2)
        
        raise ValueError(f"Invalid GitHub URL format: {url}")
    
    def _clone_repository(self, repo_url: str, retry_count: int = 3) -> Tuple[Path, str, str]:
        """
        Clone repository with improved reliability and token support.
        Returns (local_path, head_sha, default_branch).
        """
        owner, name = self._parse_github_url(repo_url)
        local_path = self.repos_dir / f"{owner}_{name}_{int(time.time())}"
        
        # Track for cleanup (keep set throughout processing)
        self.current_clone_dir = local_path
        
        # Clean any existing directory
        if local_path.exists():
            shutil.rmtree(local_path)
        
        # Setup environment for reliable cloning
        env = os.environ.copy()
        env['GIT_LFS_SKIP_SMUDGE'] = '1'  # Skip LFS downloads
        env['GIT_ASKPASS'] = 'echo'       # Never prompt for password
        env['GIT_TERMINAL_PROMPT'] = '0'  # Never prompt in terminal
        
        # Build clone URL with optional token
        if repo_url.startswith('http'):
            clone_url = repo_url
        else:
            clone_url = f"https://github.com/{owner}/{name}.git"
        
        # Add GitHub token if available
        token = os.getenv('GITHUB_TOKEN')
        if token and clone_url.startswith('https://github.com/'):
            clone_url = clone_url.replace(
                'https://github.com/',
                f'https://{token}:x-oauth-basic@github.com/'
            )
            logger.debug("Using GitHub token for authentication")
        
        # Try cloning with retries
        for attempt in range(retry_count):
            if self.shutdown_requested:
                raise KeyboardInterrupt("Shutdown requested")
            
            try:
                logger.info(f"Cloning {owner}/{name} (attempt {attempt + 1}/{retry_count})...")
                start = time.time()
                
                result = subprocess.run(
                    ['git', 'clone', '--depth', '1', clone_url, str(local_path)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env=env
                )
                
                if result.returncode != 0:
                    if attempt < retry_count - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    raise RuntimeError(f"Git clone failed: {result.stderr}")
                
                elapsed = time.time() - start
                logger.info(f"Clone took {elapsed:.2f}s")
                
                # Get HEAD SHA
                result = subprocess.run(
                    ['git', 'rev-parse', 'HEAD'],
                    cwd=local_path,
                    capture_output=True,
                    text=True
                )
                head_sha = result.stdout.strip()
                
                # Get default branch
                result = subprocess.run(
                    ['git', 'symbolic-ref', '--short', 'HEAD'],
                    cwd=local_path,
                    capture_output=True,
                    text=True
                )
                default_branch = result.stdout.strip() or 'main'
                
                # Get repository size
                result = subprocess.run(
                    ['du', '-sb', str(local_path)],
                    capture_output=True,
                    text=True
                )
                size_bytes = int(result.stdout.split()[0])
                size_mb = size_bytes / (1024 * 1024)
                
                logger.info(f"Cloned {owner}/{name} at {head_sha[:8]} ({size_mb:.1f} MB)")
                
                return local_path, head_sha, default_branch
                
            except subprocess.TimeoutExpired:
                logger.warning(f"Clone timeout for {owner}/{name}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except Exception as e:
                if attempt < retry_count - 1:
                    logger.warning(f"Clone attempt {attempt + 1} failed: {e}")
                    time.sleep(2 ** attempt)
                else:
                    raise
    
    def _generate_embeddings_batch(self, texts: List[str], batch_name: str) -> List[Optional[List[float]]]:
        """
        Generate embeddings with proper task selection.
        Returns list of embeddings (or None for failures).
        """
        if not texts or self.no_embed or not self.embedder:
            return [None] * len(texts)
        
        logger.debug(f"Generating embeddings for {len(texts)} {batch_name}...")
        start = time.time()
        
        # Determine task type based on batch name
        if "symbol" in batch_name.lower() or "code" in batch_name.lower():
            task = "code"
        elif "readme" in batch_name.lower() or "doc" in batch_name.lower():
            task = "retrieval"  # For documentation
        else:
            task = "code"  # Default to code for GitHub
        
        # Process in small batches for GPU efficiency
        all_embeddings = []
        for i in range(0, len(texts), self.EMBEDDING_BATCH_SIZE):
            if self.shutdown_requested:
                logger.info("Shutdown requested, stopping embedding generation")
                return [None] * len(texts)
            
            batch = texts[i:i+self.EMBEDDING_BATCH_SIZE]
            try:
                if task == "code":
                    embeddings = self.embedder.embed_code(batch, batch_size=len(batch))
                else:
                    embeddings = self.embedder.embed_texts(batch, task=task, batch_size=len(batch))
                
                # Convert numpy arrays to lists for JSON serialization
                if hasattr(embeddings, 'tolist'):
                    embeddings = embeddings.tolist()
                elif isinstance(embeddings, list) and len(embeddings) > 0 and hasattr(embeddings[0], 'tolist'):
                    embeddings = [e.tolist() if hasattr(e, 'tolist') else e for e in embeddings]
                
                all_embeddings.extend(embeddings)
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch: {e}")
                # Return None embeddings for failed batch
                all_embeddings.extend([None] * len(batch))
        
        elapsed = time.time() - start
        successful = sum(1 for e in all_embeddings if e is not None)
        logger.info(f"Generated {successful}/{len(texts)} embeddings in {elapsed:.2f}s ({len(texts)/elapsed:.1f} texts/s)")
        
        return all_embeddings
    
    def _upsert_documents(self, docs: List[Dict], operation: str = "REPLACE"):
        """
        Idempotent UPSERT operation with adaptive batch size.
        """
        if not docs or self.dry_run:
            return
        
        # Determine batch size based on presence of embeddings
        has_embeddings = any('embedding' in d for d in docs)
        batch_size = self.EMBEDDED_BATCH_SIZE if has_embeddings else self.UPSERT_BATCH_SIZE
        
        # Process in batches
        for i in range(0, len(docs), batch_size):
            if self.shutdown_requested:
                logger.info("Shutdown requested, stopping database writes")
                return
            
            batch = docs[i:i+batch_size]
            
            # Build the UPSERT query
            query = f"""
            FOR doc IN @batch
            UPSERT {{uid: doc.uid}}
            INSERT doc
            UPDATE doc
            IN @@collection
            OPTIONS {{overwriteMode: "replace"}}
            """
            
            try:
                self.db.aql.execute(
                    query,
                    bind_vars={'batch': batch, '@collection': self.collection_name}
                )
                logger.debug(f"Upserted batch of {len(batch)} documents")
            except Exception as e:
                logger.error(f"UPSERT batch failed: {e}")
                # Try individual inserts as fallback
                for doc in batch:
                    try:
                        self.collection.insert(doc, overwrite=True)
                    except Exception as e2:
                        logger.error(f"Failed to insert {doc.get('uid')}: {e2}")
    
    def process_repository(self, repo_url: str, extract_symbols: bool = True) -> Dict:
        """
        Process a GitHub repository with enhanced metadata and alignment.
        """
        start_time = time.time()
        stats = {
            'repo_url': repo_url,
            'files_processed': 0,
            'symbols_extracted': 0,
            'embeddings_generated': 0,
            'errors': []
        }
        
        try:
            # Clone repository
            repo_path, head_sha, default_branch = self._clone_repository(repo_url)
            owner, name = self._parse_github_url(repo_url)
            repo_id = f"{owner}/{name}"
            
            # Scan repository metadata
            metadata = self._scan_repository_metadata(repo_path, repo_id, head_sha)
            
            # Create repository document
            repo_doc = {
                'uid': f"github:{owner}/{name}@{head_sha}",
                '_key': hashlib.sha256(f"github:{owner}/{name}@{head_sha}".encode()).hexdigest()[:12],
                'kind': 'repo',
                'owner': owner,
                'name': name,
                'repo_id': repo_id,
                'repository': f"{owner}/{name}",
                'github_url': f"https://github.com/{owner}/{name}",
                'default_branch': default_branch,
                'uid_branch': f"github:{owner}/{name}@{default_branch}",
                'uid_commit': f"github:{owner}/{name}@{head_sha}",
                'head_sha': head_sha,
                'clone_status': 'success',
                'clone_time': 0,
                'scan_time': 0,
                'embedding_time': 0,
                'processor_version': 'v5',
                'processing_date': datetime.now(timezone.utc).isoformat(),
                'file_count': 0,
                'symbol_count': 0,
                'language_stats': {},
                'ingest_run_id': str(uuid.uuid4()),
                **metadata
            }
            
            # Track inline budget
            inline_budget_used = 0
            
            # Collect documents with proper alignment tracking
            file_docs = []
            file_embed_inputs = []  # List of (doc_index, text, task)
            
            symbol_docs = []
            symbol_embed_inputs = []  # List of (doc_index, text, task)
            
            # Track skipped files
            skip_stats = {}
            
            # Scan and process files (deterministic order)
            all_files = sorted(list(repo_path.rglob('*')), key=lambda p: str(p))
            
            for file_path in all_files:
                if self.shutdown_requested:
                    logger.info("Shutdown requested, stopping file processing")
                    break
                
                # Skip directories
                if file_path.is_dir():
                    continue
                
                # Skip symlinks
                if file_path.is_symlink():
                    skip_stats['symlink'] = skip_stats.get('symlink', 0) + 1
                    continue
                
                # Get relative path
                relative_path = file_path.relative_to(repo_path)
                
                # Skip based on directory patterns
                if any(skip_dir in relative_path.parts for skip_dir in self.SKIP_DIRS):
                    skip_stats['skip_dir'] = skip_stats.get('skip_dir', 0) + 1
                    continue
                
                # Skip based on extension
                if file_path.suffix.lower() in self.SKIP_EXTENSIONS:
                    skip_stats['skip_ext'] = skip_stats.get('skip_ext', 0) + 1
                    continue
                
                # Get file size
                try:
                    size_bytes = file_path.stat().st_size
                except:
                    skip_stats['stat_error'] = skip_stats.get('stat_error', 0) + 1
                    continue
                
                # Skip if too large
                if size_bytes > self.max_file_size_bytes:
                    skip_stats['too_large'] = skip_stats.get('too_large', 0) + 1
                    continue
                
                # Read file content
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                except:
                    skip_stats['read_error'] = skip_stats.get('read_error', 0) + 1
                    continue
                
                # Check for LFS pointer
                if size_bytes < 500 and content.startswith('version https://git-lfs.github.com'):
                    skip_stats['lfs_pointer'] = skip_stats.get('lfs_pointer', 0) + 1
                    continue
                
                # Generate blob SHA
                file_blob_sha = hashlib.sha256(content.encode()).hexdigest()
                
                # Check if already processed
                if not self.dry_run:
                    exists = self.collection.find({'kind': 'file', 'repo_id': repo_id, 
                                                  'file_blob_sha': file_blob_sha})
                    if exists and list(exists):
                        skip_stats['unchanged'] = skip_stats.get('unchanged', 0) + 1
                        continue
                
                # Detect language
                suffix = file_path.suffix.lower()
                language = self.SUPPORTED_LANGUAGES.get(suffix)
                if not language and content.startswith('#!'):
                    # Check shebang
                    first_line = content.split('\n')[0]
                    if 'python' in first_line:
                        language = 'python'
                    elif 'bash' in first_line or 'sh' in first_line:
                        language = 'bash'
                    elif 'node' in first_line:
                        language = 'javascript'
                    elif 'ruby' in first_line:
                        language = 'ruby'
                
                # Update language stats
                if language:
                    repo_doc['language_stats'][language] = repo_doc['language_stats'].get(language, 0) + 1
                
                # Determine if we store inline or external
                store_inline = (size_bytes < 100_000 and 
                               inline_budget_used + size_bytes < self.MAX_INLINE_BUDGET)
                
                if store_inline:
                    inline_budget_used += size_bytes
                    content_stored = content
                    external_path = None
                else:
                    content_stored = None
                    # Store externally if we have a content store
                    if self.content_store:
                        ext_dir = self.content_store / owner / name / head_sha[:8]
                        ext_dir.mkdir(parents=True, exist_ok=True)
                        ext_file = ext_dir / f"{file_blob_sha[:8]}_{file_path.name}.gz"
                        
                        with gzip.open(ext_file, 'wt', encoding='utf-8') as f:
                            f.write(content)
                        external_path = str(ext_file)
                    else:
                        external_path = None
                
                # Create file document
                file_doc = {
                    'uid': f"github:{owner}/{name}@{head_sha}#{relative_path}",
                    '_key': hashlib.sha256(f"github:{owner}/{name}@{head_sha}#{relative_path}".encode()).hexdigest()[:12],
                    'kind': 'file',
                    'repo_id': repo_id,
                    'repository': f"{owner}/{name}",
                    'file_path': str(relative_path),
                    'file_blob_sha': file_blob_sha,
                    'commit_sha': head_sha,
                    'language': language,
                    'size': size_bytes,
                    'lines': len(content.splitlines()),
                    'content': content_stored,
                    'content_sha256': file_blob_sha,
                    'external_path': external_path,
                    'is_binary': False,
                    'is_lfs': False,
                    'processing_date': datetime.now(timezone.utc).isoformat(),
                    'ingest_run_id': repo_doc['ingest_run_id'],
                    'references': []
                }
                
                # Track for embedding generation with proper alignment
                doc_index = len(file_docs)
                file_docs.append(file_doc)
                
                # Determine task type for embedding
                if file_path.name.lower() in ['readme.md', 'readme.txt', 'readme.rst']:
                    task = 'doc'
                else:
                    task = 'code'
                
                # Limit content for embedding (32k context window)
                embed_text = content[:32000]
                file_embed_inputs.append((doc_index, embed_text, task))
                
                stats['files_processed'] += 1
                repo_doc['file_count'] += 1
                
                # Extract symbols if applicable
                if extract_symbols and self.symbol_extractor and language:
                    try:
                        symbols = self.symbol_extractor.extract_symbols(content, language)
                        
                        for symbol in symbols:
                            # Create unique symbol UID
                            symbol_hash = hashlib.sha256(
                                f"{symbol['name']}:{symbol.get('type')}:{symbol.get('start_line')}".encode()
                            ).hexdigest()[:8]
                            
                            symbol_doc = {
                                'uid': f"github:{owner}/{name}@{head_sha}#{relative_path}::{symbol['name']}({symbol_hash})",
                                '_key': hashlib.sha256(
                                    f"github:{owner}/{name}@{head_sha}#{relative_path}::{symbol['name']}({symbol_hash})".encode()
                                ).hexdigest()[:12],
                                'kind': 'symbol',
                                'repo_id': repo_id,
                                'repository': f"{owner}/{name}",
                                'file_path': str(relative_path),
                                'file_blob_sha': file_blob_sha,
                                'commit_sha': head_sha,
                                'symbol_type': symbol['type'],
                                'symbol_name': symbol['name'],
                                'language': language,
                                'signature': symbol.get('signature', ''),
                                'docstring': symbol.get('docstring', ''),
                                'start_line': symbol['start_line'],
                                'end_line': symbol['end_line'],
                                'processing_date': datetime.now(timezone.utc).isoformat(),
                                'ingest_run_id': repo_doc['ingest_run_id'],
                                'references': []
                            }
                            
                            # Track for embedding with alignment
                            if 'embedding_text' in symbol:
                                doc_index = len(symbol_docs)
                                symbol_docs.append(symbol_doc)
                                symbol_embed_inputs.append((doc_index, symbol['embedding_text'], 'code'))
                                stats['symbols_extracted'] += 1
                            else:
                                symbol_docs.append(symbol_doc)
                                stats['symbols_extracted'] += 1
                    except Exception as e:
                        logger.debug(f"Failed to extract symbols from {relative_path}: {e}")
                
                repo_doc['symbol_count'] = stats['symbols_extracted']
            
            # Generate embeddings with proper alignment
            if file_embed_inputs and not self.no_embed:
                # Separate by task type
                code_inputs = [(i, t) for i, t, task in file_embed_inputs if task == 'code']
                doc_inputs = [(i, t) for i, t, task in file_embed_inputs if task == 'doc']
                
                # Generate code embeddings
                if code_inputs:
                    indices, texts = zip(*code_inputs) if code_inputs else ([], [])
                    embeddings = self._generate_embeddings_batch(list(texts), "code files")
                    
                    for doc_idx, embedding in zip(indices, embeddings):
                        if embedding is not None:
                            file_docs[doc_idx]['embedding'] = embedding
                            file_docs[doc_idx]['embedding_model'] = 'jina-embeddings-v4'
                            file_docs[doc_idx]['embedding_task'] = 'code'
                            file_docs[doc_idx]['embedding_dim'] = len(embedding)
                            file_docs[doc_idx]['embedding_date'] = datetime.now(timezone.utc).isoformat()
                            stats['embeddings_generated'] += 1
                
                # Generate doc embeddings
                if doc_inputs:
                    indices, texts = zip(*doc_inputs) if doc_inputs else ([], [])
                    embeddings = self._generate_embeddings_batch(list(texts), "documentation")
                    
                    for doc_idx, embedding in zip(indices, embeddings):
                        if embedding is not None:
                            file_docs[doc_idx]['embedding'] = embedding
                            file_docs[doc_idx]['embedding_model'] = 'jina-embeddings-v4'
                            file_docs[doc_idx]['embedding_task'] = 'retrieval'
                            file_docs[doc_idx]['embedding_dim'] = len(embedding)
                            file_docs[doc_idx]['embedding_date'] = datetime.now(timezone.utc).isoformat()
                            stats['embeddings_generated'] += 1
            
            # Generate symbol embeddings with alignment
            if symbol_embed_inputs and not self.no_embed:
                indices, texts, _ = zip(*symbol_embed_inputs) if symbol_embed_inputs else ([], [], [])
                embeddings = self._generate_embeddings_batch(list(texts), "symbols")
                
                for doc_idx, embedding in zip(indices, embeddings):
                    if embedding is not None:
                        symbol_docs[doc_idx]['embedding'] = embedding
                        symbol_docs[doc_idx]['embedding_model'] = 'jina-embeddings-v4'
                        symbol_docs[doc_idx]['embedding_task'] = 'code'
                        symbol_docs[doc_idx]['embedding_dim'] = len(embedding)
                        symbol_docs[doc_idx]['embedding_segment'] = 'symbol'
                        symbol_docs[doc_idx]['embedding_date'] = datetime.now(timezone.utc).isoformat()
                        stats['embeddings_generated'] += 1
            
            # Write to database
            if not self.dry_run:
                self._upsert_documents([repo_doc])
                
                if file_docs:
                    self._upsert_documents(file_docs)
                    logger.info(f"Wrote {len(file_docs)} file documents")
                
                if symbol_docs:
                    self._upsert_documents(symbol_docs)
                    logger.info(f"Wrote {len(symbol_docs)} symbol documents")
            
            # Clean up
            self._cleanup_handler()
            
            # Print summary
            logger.info(f"Processed {repo_id}: {stats['files_processed']} files, "
                       f"{stats['symbols_extracted']} symbols, "
                       f"{stats['embeddings_generated']} embeddings")
            
            if skip_stats:
                logger.info(f"Skipped files: {skip_stats}")
            
        except Exception as e:
            logger.error(f"Failed to process repository: {e}")
            stats['errors'].append(str(e))
            self._cleanup_handler()
        
        stats['total_time'] = time.time() - start_time
        return stats
    
    def _scan_repository_metadata(self, repo_path: Path, repo_id: str, head_sha: str) -> Dict:
        """Extract repository metadata including license and references."""
        metadata = {}
        repo_refs = []
        
        # Check for LICENSE file
        for license_name in ['LICENSE', 'LICENSE.txt', 'LICENSE.md', 'COPYING']:
            license_file = repo_path / license_name
            if license_file.exists():
                try:
                    content = license_file.read_text(errors='ignore')
                    # Store first 200 chars as snippet
                    metadata['license_snippet'] = content[:200]
                    metadata['license_source'] = license_name
                    
                    # Enhanced SPDX detection
                    content_lower = content[:1000].lower()
                    
                    # Look for explicit SPDX identifier
                    spdx_match = re.search(r'spdx-license-identifier:\s*([^\s\n]+)', content_lower)
                    if spdx_match:
                        metadata['license'] = spdx_match.group(1).upper()
                    # Fallback to keyword matching
                    elif 'mit license' in content_lower or 'mit licence' in content_lower:
                        metadata['license'] = 'MIT'
                    elif 'apache license, version 2.0' in content_lower:
                        metadata['license'] = 'Apache-2.0'
                    elif 'gnu general public license v3' in content_lower:
                        metadata['license'] = 'GPL-3.0'
                    elif 'bsd 3-clause' in content_lower:
                        metadata['license'] = 'BSD-3-Clause'
                    elif 'bsd 2-clause' in content_lower:
                        metadata['license'] = 'BSD-2-Clause'
                    else:
                        # Generic detection
                        if 'apache' in content_lower:
                            metadata['license'] = 'Apache'
                        elif 'gpl' in content_lower:
                            metadata['license'] = 'GPL'
                        elif 'bsd' in content_lower:
                            metadata['license'] = 'BSD'
                        elif 'mit' in content_lower:
                            metadata['license'] = 'MIT'
                except:
                    pass
                break
        
        # Check for CITATION.cff
        citation_file = repo_path / 'CITATION.cff'
        metadata['has_citation_file'] = citation_file.exists()
        
        if citation_file.exists():
            try:
                content = citation_file.read_text()
                # Extract DOIs
                for doi_match in re.finditer(r'10\.\d{4,}/[-._;()/:a-zA-Z0-9]+', content):
                    repo_refs.append(f"doi:{doi_match.group()}")
            except:
                pass
        
        return {
            'references': list(set(repo_refs)),
            **metadata
        }


def main():
    """Main entry point with enhanced CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description='GitHub Repository Processor v5')
    parser.add_argument('repo_url', help='Repository URL or owner/repo format')
    parser.add_argument('--db-host', default='localhost')
    parser.add_argument('--db-port', type=int, default=8529)
    parser.add_argument('--db-name', default='academy_store')
    parser.add_argument('--collection', default='base_github')
    parser.add_argument('--content-store', help='Directory for large file storage')
    parser.add_argument('--extract-symbols', action='store_true', help='Extract symbols with tree-sitter')
    parser.add_argument('--dry-run', action='store_true', help='Skip database writes')
    parser.add_argument('--no-embed', action='store_true', help='Skip embedding generation')
    parser.add_argument('--max-repo-size', type=int, default=500, help='Max repo size in MB')
    parser.add_argument('--max-file-size', type=int, default=2, help='Max file size in MB')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = GitHubProcessorV5(
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        collection=args.collection,
        content_store_dir=args.content_store,
        max_repo_size_mb=args.max_repo_size,
        max_file_size_mb=args.max_file_size,
        dry_run=args.dry_run,
        no_embed=args.no_embed
    )
    
    # Process repository
    stats = processor.process_repository(
        args.repo_url,
        extract_symbols=args.extract_symbols
    )
    
    # Print results
    if stats['errors']:
        print(f"❌ Failed: {stats['errors'][0]}")
        return 1
    else:
        print(f"✅ Successfully processed {args.repo_url}")
        print(f"   Files: {stats['files_processed']}")
        print(f"   Symbols: {stats['symbols_extracted']}")
        print(f"   Embeddings: {stats['embeddings_generated']}")
        print(f"   Time: {stats['total_time']:.2f}s")
        return 0


if __name__ == "__main__":
    sys.exit(main())