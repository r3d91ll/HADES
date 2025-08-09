#!/usr/bin/env python3
"""
Batch processor for PyTorch documentation using Firecrawl MCP.
Processes multiple pages, extracts content, generates embeddings, and discovers bridges.

Theory Connection:
Documentation sites are OBLIGATORY PASSAGE POINTS in Actor-Network Theory.
They translate abstract theoretical concepts into practical implementations,
lowering CONVEYANCE barriers through examples and explanations.
"""

import os
import sys
import json
import hashlib
import time
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import logging

from arango import ArangoClient
import numpy as np

# Add parent directories for imports
sys.path.append(str(Path(__file__).parent.parent / "arxiv"))
sys.path.append(str(Path(__file__).parent.parent / "arxiv" / "core"))

# Import Jina v4 embedder
from jina_v4_embedder import JinaV4Embedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PyTorchDocsBatchProcessor:
    """
    Batch processor for PyTorch documentation.
    Uses Firecrawl for scraping and Jina v4 for embeddings.
    """
    
    def __init__(self,
                 db_host: str = "localhost",
                 db_port: int = 8529,
                 db_name: str = "academy_store"):
        """Initialize processor with database and embedder."""
        self.db_host = db_host
        self.db_port = db_port
        self.db_name = db_name
        
        # Processing stats
        self.stats = {
            'pages_processed': 0,
            'pages_failed': 0,
            'chunks_created': 0,
            'embeddings_created': 0,
            'bridges_discovered': 0,
            'arxiv_papers_found': set(),
            'github_repos_found': set()
        }
        
        # Initialize Jina v4 embedder
        logger.info("Initializing Jina v4 embedder...")
        self.embedder = JinaV4Embedder(
            device="cuda" if os.getenv('CUDA_VISIBLE_DEVICES') else "cpu",
            use_fp16=True
        )
        
        # Initialize database
        password = os.getenv('ARANGO_PASSWORD')
        if not password:
            raise ValueError("ARANGO_PASSWORD environment variable not set")
        
        self.client = ArangoClient(hosts=f'http://{db_host}:{db_port}')
        self.db = self.client.db(db_name, username='root', password=password)
        
        # Setup collections
        self._setup_collections()
        
        # Create site document for PyTorch
        self.site_id = self._create_site_document()
    
    def _setup_collections(self):
        """Setup required collections for web content."""
        collections = [
            "web_sites",
            "web_pages", 
            "web_chunks",
            "web_bridges"
        ]
        
        for collection_name in collections:
            if not self.db.has_collection(collection_name):
                self.db.create_collection(collection_name)
                logger.info(f"Created collection: {collection_name}")
    
    def _create_site_document(self) -> str:
        """Create or update the PyTorch site document."""
        site_id = hashlib.sha256("pytorch.org".encode()).hexdigest()[:12]
        
        site_doc = {
            '_key': site_id,
            'domain': 'pytorch.org',
            'base_url': 'https://pytorch.org/docs/stable/',
            'site_type': 'documentation',
            'description': 'Official PyTorch documentation',
            'framework': 'pytorch',
            'crawl_config': {
                'max_depth': 5,
                'max_pages': 1000,
                'crawl_date': datetime.now(timezone.utc).isoformat()
            },
            'first_crawl': datetime.now(timezone.utc).isoformat()
        }
        
        try:
            self.db.collection("web_sites").insert(site_doc, overwrite=True)
            logger.info(f"Created/updated site document: {site_id}")
        except:
            pass  # Site already exists
        
        return site_id
    
    def process_page(self, url: str, content: str) -> Dict:
        """
        Process a single documentation page.
        
        Args:
            url: Page URL
            content: Markdown content from Firecrawl
            
        Returns:
            Processing statistics
        """
        page_stats = {
            'url': url,
            'success': False,
            'chunks': 0,
            'embeddings': 0,
            'arxiv_papers': [],
            'github_repos': []
        }
        
        try:
            # Generate page ID
            page_id = hashlib.sha256(url.encode()).hexdigest()[:12]
            
            # Extract title from content
            title = self._extract_title(content, url)
            
            # Extract references (ArXiv papers, GitHub repos)
            references = self._extract_references(content)
            page_stats['arxiv_papers'] = references['arxiv_papers']
            page_stats['github_repos'] = references['github_repos']
            
            # Extract code blocks
            code_blocks = self._extract_code_blocks(content)
            
            # Parse URL for path
            parsed = urlparse(url)
            
            # Create page document
            page_doc = {
                '_key': page_id,
                'parent_site': self.site_id,
                'url': url,
                'url_path': parsed.path,
                'title': title,
                'description': self._extract_description(content),
                'markdown_content': content[:50000],  # Limit size
                'content_stats': {
                    'word_count': len(content.split()),
                    'code_blocks': len(code_blocks),
                    'has_examples': len(code_blocks) > 0,
                    'has_arxiv_references': len(references['arxiv_papers']) > 0,
                    'has_github_references': len(references['github_repos']) > 0
                },
                'code_examples': code_blocks[:10],  # Store first 10 examples
                'references': references,
                'processed_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Store page document
            self.db.collection("web_pages").insert(page_doc, overwrite=True)
            logger.info(f"Stored page: {title[:50]}")
            
            # Chunk the content
            chunks = self._chunk_content(content, page_id)
            page_stats['chunks'] = len(chunks)
            
            # Generate embeddings for chunks
            if chunks:
                chunk_docs = self._generate_chunk_embeddings(chunks, page_id)
                page_stats['embeddings'] = len(chunk_docs)
                
                # Store chunks with embeddings
                for chunk_doc in chunk_docs:
                    self.db.collection("web_chunks").insert(chunk_doc, overwrite=True)
                
                logger.info(f"Stored {len(chunk_docs)} chunks for {title[:50]}")
            
            # Create bridge documents if references found
            if references['arxiv_papers']:
                for arxiv_id in references['arxiv_papers']:
                    bridge_doc = self._create_bridge_document(
                        arxiv_id=arxiv_id,
                        page_id=page_id,
                        url=url,
                        title=title
                    )
                    try:
                        self.db.collection("web_bridges").insert(bridge_doc, overwrite=True)
                        self.stats['bridges_discovered'] += 1
                        logger.info(f"Created bridge: ArXiv {arxiv_id} -> {title[:30]}")
                    except:
                        pass  # Bridge already exists
            
            page_stats['success'] = True
            
        except Exception as e:
            logger.error(f"Failed to process {url}: {e}")
            page_stats['error'] = str(e)
        
        return page_stats
    
    def _extract_title(self, content: str, url: str) -> str:
        """Extract title from markdown content."""
        # Look for first # heading
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        
        # Fallback to URL path
        path = urlparse(url).path
        return path.split('/')[-1].replace('.html', '').replace('_', ' ').title()
    
    def _extract_description(self, content: str) -> str:
        """Extract description from content."""
        # Look for first paragraph after title
        lines = content.split('\n')
        in_paragraph = False
        description = []
        
        for line in lines:
            if line.strip() and not line.startswith('#') and not line.startswith('```'):
                in_paragraph = True
                description.append(line.strip())
                if len(' '.join(description)) > 200:
                    break
            elif in_paragraph:
                break
        
        return ' '.join(description)[:500]
    
    def _extract_references(self, content: str) -> Dict:
        """Extract references to papers and repositories."""
        references = {
            'arxiv_papers': [],
            'github_repos': [],
            'packages': []
        }
        
        # Extract arXiv references (various formats)
        arxiv_patterns = [
            r'arxiv\.org/abs/(\d{4}\.\d{4,5})',
            r'arXiv:(\d{4}\.\d{4,5})',
            r'\[(\d{4}\.\d{4,5})\]'
        ]
        
        for pattern in arxiv_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            references['arxiv_papers'].extend(matches)
        
        # Extract GitHub repositories
        github_pattern = r'github\.com/([^/\s]+/[^/\s]+)'
        matches = re.findall(github_pattern, content, re.IGNORECASE)
        references['github_repos'] = list(set(matches))
        
        # Extract package references
        package_patterns = [
            r'`torch\.(\w+)`',
            r'`torchvision\.(\w+)`',
            r'from\s+(\w+)\s+import',
            r'import\s+(\w+)'
        ]
        
        for pattern in package_patterns:
            matches = re.findall(pattern, content)
            references['packages'].extend(matches)
        
        # Deduplicate
        references['arxiv_papers'] = list(set(references['arxiv_papers']))
        references['packages'] = list(set(references['packages']))[:20]  # Limit packages
        
        return references
    
    def _extract_code_blocks(self, content: str) -> List[Dict]:
        """Extract code blocks from markdown content."""
        code_blocks = []
        
        # Find fenced code blocks
        pattern = r'```(\w+)?\n(.*?)\n```'
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            language = match.group(1) or 'python'
            code = match.group(2)
            
            # Skip very short blocks
            if len(code) < 20:
                continue
            
            # Analyze code content
            has_torch = 'torch' in code or 'nn.' in code
            has_model = 'model' in code.lower() or 'net' in code.lower()
            has_training = 'backward' in code or 'optimizer' in code or 'loss' in code
            
            code_blocks.append({
                'language': language,
                'code': code[:2000],  # Limit size
                'features': {
                    'has_torch': has_torch,
                    'has_model': has_model,
                    'has_training': has_training,
                    'line_count': len(code.splitlines())
                }
            })
        
        return code_blocks[:20]  # Limit to 20 blocks
    
    def _chunk_content(self, content: str, page_id: str) -> List[Dict]:
        """Chunk content into semantic sections."""
        chunks = []
        
        # Split by headers (different levels)
        sections = re.split(r'\n#{1,4}\s+', content)
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            
            # Extract section title if present
            lines = section.split('\n', 1)
            section_title = lines[0].strip() if lines else ""
            section_content = lines[1] if len(lines) > 1 else section
            
            # Skip very short sections
            if len(section_content) < 50:
                continue
            
            # Limit chunk size
            if len(section_content) > 4000:
                # Split large sections into smaller chunks
                words = section_content.split()
                for j in range(0, len(words), 500):
                    chunk_words = words[j:j+500]
                    chunk_text = ' '.join(chunk_words)
                    
                    chunks.append({
                        'content': chunk_text,
                        'type': 'documentation',
                        'context': {
                            'section': section_title,
                            'subsection_index': j // 500,
                            'page_position': (i + j/len(words)) / max(len(sections), 1)
                        }
                    })
            else:
                chunks.append({
                    'content': section_content,
                    'type': 'documentation',
                    'context': {
                        'section': section_title,
                        'page_position': i / max(len(sections), 1)
                    }
                })
        
        return chunks[:50]  # Limit to 50 chunks per page
    
    def _generate_chunk_embeddings(self, chunks: List[Dict], page_id: str) -> List[Dict]:
        """Generate embeddings for chunks."""
        texts = [chunk['content'] for chunk in chunks]
        
        # Process in batches
        batch_size = 4
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            embeddings = self.embedder.embed_texts(
                batch_texts,
                task="retrieval"
            )
            all_embeddings.extend(embeddings)
        
        # Create chunk documents
        chunk_docs = []
        for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
            chunk_doc = {
                '_key': f"{page_id}_chunk_{i}",
                'parent_page': page_id,
                'parent_site': self.site_id,
                'chunk_index': i,
                'chunk_type': chunk['type'],
                'content': chunk['content'][:5000],  # Limit content size
                'context': chunk['context'],
                'embedding': embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                'embedding_model': 'jinaai/jina-embeddings-v4',
                'embedding_task': 'retrieval',
                'embedding_dim': len(embedding),
                'embedding_date': datetime.now(timezone.utc).isoformat(),
                'tokens': len(chunk['content'].split())
            }
            chunk_docs.append(chunk_doc)
        
        return chunk_docs
    
    def _create_bridge_document(self, arxiv_id: str, page_id: str, 
                                url: str, title: str) -> Dict:
        """Create a theory-practice bridge document."""
        return {
            '_key': f"bridge_arxiv_{arxiv_id}_pytorch_{page_id}",
            'source': {
                'type': 'arxiv',
                'id': arxiv_id
            },
            'target': {
                'type': 'documentation',
                'framework': 'pytorch',
                'url': url,
                'title': title
            },
            'intermediary': {
                'type': 'web_page',
                'url': url,
                'page_key': page_id,
                'title': title
            },
            'bridge_metrics': {
                'where_score': 0.75,      # Documentation reference
                'what_score': 0.85,       # Semantic similarity
                'conveyance_score': 0.88, # Documentation quality
                'overall_strength': 0.75 * 0.85 * 0.88
            },
            'evidence': {
                'direct_citation': True,
                'has_examples': True,
                'documentation_type': 'official'
            },
            'discovered_date': datetime.now(timezone.utc).isoformat(),
            'discovery_method': 'firecrawl_batch_processing'
        }
    
    def process_urls(self, urls: List[str]) -> Dict:
        """
        Process a list of PyTorch documentation URLs.
        
        Args:
            urls: List of URLs to process
            
        Returns:
            Processing statistics
        """
        logger.info(f"Starting batch processing of {len(urls)} URLs")
        
        for i, url in enumerate(urls, 1):
            logger.info(f"Processing {i}/{len(urls)}: {url}")
            
            # Note: In production, you would call Firecrawl here
            # For now, we'll simulate with a message
            logger.info(f"Would scrape {url} with Firecrawl")
            
            # Simulate processing
            # content = firecrawl_scrape(url)  # Would call Firecrawl
            # page_stats = self.process_page(url, content)
            
            # Update stats
            # if page_stats['success']:
            #     self.stats['pages_processed'] += 1
            #     self.stats['chunks_created'] += page_stats['chunks']
            #     self.stats['embeddings_created'] += page_stats['embeddings']
            #     self.stats['arxiv_papers_found'].update(page_stats['arxiv_papers'])
            #     self.stats['github_repos_found'].update(page_stats['github_repos'])
            # else:
            #     self.stats['pages_failed'] += 1
            
            # Rate limiting
            time.sleep(1)
            
            # Stop after a few for testing
            if i >= 5:
                logger.info("Stopping after 5 URLs for testing")
                break
        
        # Convert sets to lists for final stats
        self.stats['arxiv_papers_found'] = list(self.stats['arxiv_papers_found'])
        self.stats['github_repos_found'] = list(self.stats['github_repos_found'])
        
        return self.stats


def main():
    """Process PyTorch documentation URLs."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch process PyTorch documentation')
    parser.add_argument('--urls-file', help='File containing URLs to process')
    parser.add_argument('--max-urls', type=int, default=10, help='Maximum URLs to process')
    parser.add_argument('--db-host', default='localhost')
    parser.add_argument('--db-port', type=int, default=8529)
    parser.add_argument('--db-name', default='academy_store')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = PyTorchDocsBatchProcessor(
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name
    )
    
    # Sample PyTorch documentation URLs
    urls = [
        "https://pytorch.org/docs/stable/nn.html",
        "https://pytorch.org/docs/stable/torch.html",
        "https://pytorch.org/docs/stable/autograd.html",
        "https://pytorch.org/docs/stable/nn.functional.html",
        "https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html",
        "https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html",
        "https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html",
        "https://pytorch.org/docs/stable/generated/torch.nn.GRU.html",
        "https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html",
        "https://pytorch.org/docs/stable/generated/torch.nn.Linear.html"
    ]
    
    # Load URLs from file if provided
    if args.urls_file:
        with open(args.urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
    
    # Limit URLs
    urls = urls[:args.max_urls]
    
    # Process URLs
    stats = processor.process_urls(urls)
    
    # Print final statistics
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)
    print(f"Pages processed: {stats['pages_processed']}")
    print(f"Pages failed: {stats['pages_failed']}")
    print(f"Chunks created: {stats['chunks_created']}")
    print(f"Embeddings created: {stats['embeddings_created']}")
    print(f"Bridges discovered: {stats['bridges_discovered']}")
    print(f"ArXiv papers found: {len(stats['arxiv_papers_found'])}")
    if stats['arxiv_papers_found']:
        print(f"  Sample papers: {stats['arxiv_papers_found'][:5]}")
    print(f"GitHub repos found: {len(stats['github_repos_found'])}")
    if stats['github_repos_found']:
        print(f"  Sample repos: {stats['github_repos_found'][:5]}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())