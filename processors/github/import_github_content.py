#!/usr/bin/env python3
"""
Simple importer for GitHub content fetched by MCP tools.

Processes GitHub files already fetched by the MCP GitHub server,
similar to how we handle FireCrawl output.

Theory Connection:
This importer focuses on the PRACTICE dimension - code represents
maximum CONVEYANCE as it's directly executable knowledge.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

import torch
from transformers import AutoModel, AutoTokenizer
from arango import ArangoClient
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ImportStats:
    """Statistics from an import operation."""
    total_files: int = 0
    imported: int = 0
    skipped: int = 0
    failed: int = 0
    total_lines: int = 0
    total_chunks: int = 0


class GitHubImporter:
    """
    Simple importer for GitHub content fetched by MCP tools.
    Processes JSON files containing code and metadata.
    """
    
    def __init__(self, db_config: Dict):
        """Initialize importer with database connection."""
        self.db = self._init_database(db_config)
        self.embedder = None
        self.tokenizer = None
        self.stats = ImportStats()
        
        # Quality thresholds for code
        self.min_lines = 10  # Minimum lines of code
        self.max_lines = 10000  # Maximum lines per file
        self.max_file_size = 1_000_000  # 1MB max
        
        # Supported languages (can be extended)
        self.supported_languages = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c',
            '.rs', '.go', '.rb', '.php', '.swift', '.kt', '.scala',
            '.r', '.jl', '.m', '.sh', '.yaml', '.yml', '.json', '.md'
        }
        
    def _init_database(self, config: Dict):
        """Initialize database connection."""
        try:
            client = ArangoClient(hosts=config['host'])
            db = client.db(
                config['database'],
                username=config['username'],
                password=config['password']
            )
            
            # Ensure collections exist
            collections = ['github_repos', 'github_files', 'github_embeddings']
            for collection_name in collections:
                if not db.has_collection(collection_name):
                    db.create_collection(collection_name)
                    logger.info(f"Created {collection_name} collection")
                    
            return db
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def _init_embedder(self):
        """Initialize Jina v4 embedder for code."""
        if self.embedder is not None:
            return
            
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                torch.cuda.set_device(1)  # Use GPU 1
            
            logger.info(f"Initializing Jina v4 (code adapter) on {device}")
            
            # Use code-specific adapter if available
            self.embedder = AutoModel.from_pretrained(
                "jinaai/jina-embeddings-v4",
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            
            self.embedder.eval()
            self.tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v4")
            
            logger.info("Jina v4 initialized for code")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {e}")
            raise
    
    def validate_code_file(self, data: Dict) -> bool:
        """
        Validate if code file meets import standards.
        
        Args:
            data: File data from MCP GitHub tools
            
        Returns:
            True if file meets standards, False otherwise
        """
        # Check required fields
        if 'content' not in data or 'path' not in data:
            logger.debug("Missing required fields")
            return False
        
        content = data.get('content', '')
        path = data.get('path', '')
        
        # Check file extension
        ext = Path(path).suffix.lower()
        if ext not in self.supported_languages:
            logger.debug(f"Unsupported file type: {ext}")
            return False
        
        # Check file size
        if len(content) > self.max_file_size:
            logger.debug(f"File too large: {len(content)} bytes")
            return False
        
        # Check line count
        lines = content.split('\n')
        if len(lines) < self.min_lines:
            logger.debug(f"Too few lines: {len(lines)}")
            return False
        
        if len(lines) > self.max_lines:
            logger.debug(f"Too many lines: {len(lines)}")
            return False
        
        # Check for actual code content (not just comments)
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        if len(code_lines) < 5:
            logger.debug("Too few actual code lines")
            return False
        
        return True
    
    def import_repository(self, directory: Path, repo_url: str) -> ImportStats:
        """
        Import GitHub repository content from directory.
        
        Args:
            directory: Directory containing JSON files with GitHub content
            repo_url: GitHub repository URL
            
        Returns:
            Import statistics
        """
        self.stats = ImportStats()
        
        # Extract repo info
        repo_parts = repo_url.rstrip('/').split('/')
        repo_owner = repo_parts[-2] if len(repo_parts) > 1 else "unknown"
        repo_name = repo_parts[-1].replace('.git', '') if repo_parts else "unknown"
        repo_key = f"{repo_owner}_{repo_name}"
        
        # Create repo document
        repo_doc = {
            "_key": repo_key,
            "url": repo_url,
            "owner": repo_owner,
            "name": repo_name,
            "import_date": datetime.utcnow().isoformat(),
            "file_count": 0,
            "total_lines": 0,
            "languages": set()
        }
        
        # Find all JSON files
        json_files = list(directory.glob("*.json"))
        self.stats.total_files = len(json_files)
        
        logger.info(f"Found {self.stats.total_files} files to import for {repo_name}")
        
        # Process each file
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Handle both single files and arrays
                if isinstance(data, list):
                    for item in data:
                        self._process_file(item, repo_key, repo_doc)
                else:
                    self._process_file(data, repo_key, repo_doc)
                    
            except Exception as e:
                logger.error(f"Failed to import {json_file}: {e}")
                self.stats.failed += 1
        
        # Update and store repo document
        repo_doc["file_count"] = self.stats.imported
        repo_doc["total_lines"] = self.stats.total_lines
        repo_doc["languages"] = list(repo_doc["languages"])
        
        try:
            self.db.collection('github_repos').insert(repo_doc, overwrite=True)
        except Exception as e:
            logger.error(f"Failed to store repo document: {e}")
        
        logger.info(f"Import complete: {self.stats.imported} imported, "
                   f"{self.stats.skipped} skipped, {self.stats.failed} failed")
        
        return self.stats
    
    def _process_file(self, data: Dict, repo_key: str, repo_doc: Dict):
        """Process a single code file."""
        # Validate file
        if not self.validate_code_file(data):
            self.stats.skipped += 1
            return
        
        path = data.get('path', '')
        content = data.get('content', '')
        
        # Generate file key
        file_key = hashlib.md5(f"{repo_key}_{path}".encode()).hexdigest()
        
        # Detect language
        ext = Path(path).suffix.lower()
        language = self._get_language_from_ext(ext)
        repo_doc["languages"].add(language)
        
        # Count lines
        lines = len(content.split('\n'))
        self.stats.total_lines += lines
        
        # Create file document
        file_doc = {
            "_key": file_key,
            "repo": repo_key,
            "path": path,
            "filename": Path(path).name,
            "language": language,
            "content": content[:50000],  # Truncate very long files
            "line_count": lines,
            "size_bytes": len(content),
            "import_date": datetime.utcnow().isoformat()
        }
        
        # Store file
        try:
            self.db.collection('github_files').insert(file_doc, overwrite=True)
            self.stats.imported += 1
            logger.debug(f"Imported: {path}")
            
            # Generate embeddings for substantial files
            if lines > 20:
                self._generate_embeddings(file_key, content, path, language)
                
        except Exception as e:
            logger.error(f"Failed to store file {path}: {e}")
            self.stats.failed += 1
    
    def _generate_embeddings(self, file_key: str, content: str, path: str, language: str):
        """Generate and store embeddings for code."""
        if self.embedder is None:
            self._init_embedder()
        
        try:
            # Simple chunking for code (by functions/classes would be better)
            chunk_size = 2048  # Smaller chunks for code
            chunks = []
            
            tokens = self.tokenizer.encode(content)
            for i in range(0, len(tokens), chunk_size):
                chunk_tokens = tokens[i:i + chunk_size]
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk_text)
            
            # Generate embeddings
            with torch.no_grad():
                for i, chunk_text in enumerate(chunks):
                    embeddings = self.embedder.encode(
                        [chunk_text],
                        batch_size=1,
                        show_progress_bar=False
                    )
                    
                    if torch.is_tensor(embeddings):
                        embedding = embeddings[0].cpu().numpy()
                    else:
                        embedding = embeddings[0]
                    
                    # Store embedding
                    embed_doc = {
                        "_key": f"{file_key}_chunk_{i}",
                        "file_key": file_key,
                        "chunk_index": i,
                        "chunk_text": chunk_text[:5000],
                        "embedding": embedding.tolist(),
                        "path": path,
                        "language": language,
                        "embedding_model": "jinaai/jina-embeddings-v4",
                        "created_at": datetime.utcnow().isoformat()
                    }
                    
                    self.db.collection('github_embeddings').insert(embed_doc, overwrite=True)
                    self.stats.total_chunks += 1
                    
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
    
    def _get_language_from_ext(self, ext: str) -> str:
        """Map file extension to language name."""
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.rs': 'rust',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.r': 'r',
            '.jl': 'julia',
            '.m': 'matlab',
            '.sh': 'shell',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.md': 'markdown'
        }
        return language_map.get(ext, 'unknown')


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Import GitHub content to database")
    parser.add_argument("directory", help="Directory containing GitHub JSON files")
    parser.add_argument("repo_url", help="GitHub repository URL")
    parser.add_argument("--min-lines", type=int, default=10, help="Minimum lines of code")
    parser.add_argument("--max-lines", type=int, default=10000, help="Maximum lines per file")
    
    args = parser.parse_args()
    
    # Database configuration
    db_config = {
        'host': os.environ.get('ARANGO_HOST', 'http://192.168.1.69:8529'),
        'database': os.environ.get('ARANGO_DATABASE', 'academy_store'),
        'username': os.environ.get('ARANGO_USERNAME', 'root'),
        'password': os.environ.get('ARANGO_PASSWORD')
    }
    
    # Create importer
    importer = GitHubImporter(db_config)
    importer.min_lines = args.min_lines
    importer.max_lines = args.max_lines
    
    # Import repository
    directory = Path(args.directory)
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return 1
    
    stats = importer.import_repository(directory, args.repo_url)
    
    # Print statistics
    print(f"\nImport Statistics:")
    print(f"  Total files: {stats.total_files}")
    print(f"  Imported: {stats.imported}")
    print(f"  Skipped: {stats.skipped}")
    print(f"  Failed: {stats.failed}")
    print(f"  Total lines: {stats.total_lines}")
    print(f"  Total chunks: {stats.total_chunks}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())