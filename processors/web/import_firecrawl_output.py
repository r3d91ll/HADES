#!/usr/bin/env python3
"""
Simple importer for FireCrawl output.

Reads FireCrawl JSON output from a directory and imports it into ArangoDB
after validation. No crawling, no budget management - just import what's there.

Theory Connection:
This importer acts as a GATEKEEPER in ANT terms - it examines external
content and decides what gets enrolled into our knowledge network based
on quality and relevance standards.
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
    total_tokens: int = 0
    total_chunks: int = 0


class FireCrawlImporter:
    """
    Simple importer for FireCrawl output.
    Reads JSON files from a directory and imports to database.
    """
    
    def __init__(self, db_config: Dict):
        """Initialize importer with database connection."""
        self.db = self._init_database(db_config)
        self.embedder = None
        self.tokenizer = None
        self.stats = ImportStats()
        
        # Quality thresholds
        self.min_content_length = 100  # Minimum characters
        self.max_content_length = 500000  # Maximum characters
        self.min_word_count = 20  # Minimum words
        
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
            collections = ['web_content', 'web_chunks', 'web_embeddings']
            for collection_name in collections:
                if not db.has_collection(collection_name):
                    db.create_collection(collection_name)
                    logger.info(f"Created {collection_name} collection")
                    
            return db
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def _init_embedder(self):
        """Initialize Jina v4 embedder."""
        if self.embedder is not None:
            return
            
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                torch.cuda.set_device(1)  # Use GPU 1
            
            logger.info(f"Initializing Jina v4 on {device}")
            
            self.embedder = AutoModel.from_pretrained(
                "jinaai/jina-embeddings-v4",
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            
            self.embedder.eval()
            self.tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v4")
            
            logger.info("Jina v4 initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {e}")
            raise
    
    def validate_content(self, data: Dict) -> bool:
        """
        Validate if content meets import standards.
        
        Args:
            data: FireCrawl output data
            
        Returns:
            True if content meets standards, False otherwise
        """
        # Check required fields
        if 'content' not in data:
            logger.debug("Missing content field")
            return False
        
        content = data.get('content', '')
        
        # Check content length
        if len(content) < self.min_content_length:
            logger.debug(f"Content too short: {len(content)} chars")
            return False
        
        if len(content) > self.max_content_length:
            logger.debug(f"Content too long: {len(content)} chars")
            return False
        
        # Check word count
        word_count = len(content.split())
        if word_count < self.min_word_count:
            logger.debug(f"Too few words: {word_count}")
            return False
        
        # Check for actual content (not just navigation/boilerplate)
        if content.count('[') > len(content) / 10:  # Too many brackets (likely navigation)
            logger.debug("Content appears to be mostly navigation")
            return False
        
        # Check URL is valid
        url = data.get('url', data.get('metadata', {}).get('url', ''))
        if not url or not url.startswith('http'):
            logger.debug("Invalid or missing URL")
            return False
        
        return True
    
    def import_directory(self, directory: Path, source_name: str = "unknown") -> ImportStats:
        """
        Import all FireCrawl JSON files from a directory.
        
        Args:
            directory: Directory containing JSON files
            source_name: Name of the source (e.g., "pytorch_docs")
            
        Returns:
            Import statistics
        """
        self.stats = ImportStats()
        
        # Find all JSON files
        json_files = list(directory.glob("*.json"))
        self.stats.total_files = len(json_files)
        
        logger.info(f"Found {self.stats.total_files} JSON files in {directory}")
        
        # Process each file
        for json_file in json_files:
            try:
                self._import_file(json_file, source_name)
            except Exception as e:
                logger.error(f"Failed to import {json_file}: {e}")
                self.stats.failed += 1
        
        # Log final statistics
        logger.info(f"Import complete: {self.stats.imported} imported, "
                   f"{self.stats.skipped} skipped, {self.stats.failed} failed")
        
        return self.stats
    
    def _import_file(self, file_path: Path, source_name: str):
        """Import a single JSON file."""
        # Load JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle both single documents and arrays
        if isinstance(data, list):
            for item in data:
                self._process_document(item, source_name)
        else:
            self._process_document(data, source_name)
    
    def _process_document(self, data: Dict, source_name: str):
        """Process and import a single document."""
        # Validate content
        if not self.validate_content(data):
            self.stats.skipped += 1
            return
        
        # Extract metadata
        url = data.get('url', data.get('metadata', {}).get('url', ''))
        title = data.get('title', data.get('metadata', {}).get('title', ''))
        content = data.get('content', '')
        
        # Generate document key
        doc_key = hashlib.md5(url.encode()).hexdigest()
        
        # Create document
        doc = {
            "_key": doc_key,
            "url": url,
            "title": title,
            "content": content[:50000],  # Truncate very long content
            "source": source_name,
            "content_length": len(content),
            "word_count": len(content.split()),
            "import_date": datetime.utcnow().isoformat(),
            "metadata": data.get('metadata', {}),
            "links": data.get('links', [])
        }
        
        # Store document
        try:
            self.db.collection('web_content').insert(doc, overwrite=True)
            self.stats.imported += 1
            logger.debug(f"Imported: {title[:50]}")
            
            # Generate embeddings if content is substantial
            if len(content) > 500:
                self._generate_embeddings(doc_key, content, url)
                
        except Exception as e:
            logger.error(f"Failed to store document {url}: {e}")
            self.stats.failed += 1
    
    def _generate_embeddings(self, doc_key: str, content: str, url: str):
        """Generate and store embeddings for content."""
        if self.embedder is None:
            self._init_embedder()
        
        try:
            # Simple chunking (you can make this more sophisticated)
            chunk_size = 4096
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
                        "_key": f"{doc_key}_chunk_{i}",
                        "parent_doc": doc_key,
                        "chunk_index": i,
                        "chunk_text": chunk_text[:5000],
                        "embedding": embedding.tolist(),
                        "url": url,
                        "embedding_model": "jinaai/jina-embeddings-v4",
                        "created_at": datetime.utcnow().isoformat()
                    }
                    
                    self.db.collection('web_embeddings').insert(embed_doc, overwrite=True)
                    self.stats.total_chunks += 1
                    
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Import FireCrawl output to database")
    parser.add_argument("directory", help="Directory containing FireCrawl JSON output")
    parser.add_argument("--source", default="web", help="Source name for imported content")
    parser.add_argument("--min-length", type=int, default=100, help="Minimum content length")
    parser.add_argument("--max-length", type=int, default=500000, help="Maximum content length")
    
    args = parser.parse_args()
    
    # Database configuration
    db_config = {
        'host': os.environ.get('ARANGO_HOST', 'http://192.168.1.69:8529'),
        'database': os.environ.get('ARANGO_DATABASE', 'academy_store'),
        'username': os.environ.get('ARANGO_USERNAME', 'root'),
        'password': os.environ.get('ARANGO_PASSWORD')
    }
    
    # Create importer
    importer = FireCrawlImporter(db_config)
    importer.min_content_length = args.min_length
    importer.max_content_length = args.max_length
    
    # Import directory
    directory = Path(args.directory)
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return 1
    
    stats = importer.import_directory(directory, args.source)
    
    # Print statistics
    print(f"\nImport Statistics:")
    print(f"  Total files: {stats.total_files}")
    print(f"  Imported: {stats.imported}")
    print(f"  Skipped: {stats.skipped}")
    print(f"  Failed: {stats.failed}")
    print(f"  Total chunks: {stats.total_chunks}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())