#!/usr/bin/env python3
"""
Daily ArXiv updater with Jina v4 embeddings.
Fetches yesterday's papers and adds to ArangoDB with 2048-dim embeddings.
Run via cron at 01:00 UTC daily.

Theory Connection:
Maintains temporal continuity in our information space by ensuring
the WHERE dimension (database) contains the latest WHAT (papers).
Each new paper is immediately embedded in our 2048-dimensional space.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict
import time
import torch
from transformers import AutoModel

from arango import ArangoClient

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/todd/olympus/HADES/logs/daily_arxiv_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ArXivDailyUpdater:
    """Updates ArangoDB with latest ArXiv papers and Jina v4 embeddings."""
    
    def __init__(self, db_host: str = "192.168.1.69", db_name: str = "academy_store"):
        self.db_host = db_host
        self.db_name = db_name
        self.base_url = "http://export.arxiv.org/api/query"
        self._init_database()
        self._init_embedder()
    
    def _init_database(self):
        """Initialize database connection."""
        password = os.environ.get('ARANGO_PASSWORD')
        if not password:
            raise ValueError("ARANGO_PASSWORD environment variable required")
        
        client = ArangoClient(hosts=f'http://{self.db_host}:8529')
        
        # Connect to _system first to ensure database exists
        sys_db = client.db('_system', username='root', password=password)
        if not sys_db.has_database(self.db_name):
            logger.warning(f"Database {self.db_name} not found, creating...")
            sys_db.create_database(self.db_name)
        
        # Connect to our database
        self.db = client.db(self.db_name, username='root', password=password)
        
        # Ensure collection exists
        if not self.db.has_collection('base_arxiv'):
            self.collection = self.db.create_collection('base_arxiv')
            logger.info("Created base_arxiv collection")
        else:
            self.collection = self.db.collection('base_arxiv')
        
        logger.info(f"Connected to {self.db_name} at {self.db_host}")
    
    def _init_embedder(self):
        """Initialize Jina v4 embedder."""
        try:
            # Check if GPU is available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cuda":
                # Use GPU 1 by default (GPU 0 might be busy with rebuild)
                if 'CUDA_VISIBLE_DEVICES' not in os.environ:
                    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
            
            logger.info(f"Initializing Jina v4 on {self.device}")
            
            # Load Jina v4 with fp16 for efficiency
            self.model = AutoModel.from_pretrained(
                "jinaai/jina-embeddings-v4",
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            self.model.eval()
            logger.info("Jina v4 embedder initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Jina v4: {e}")
            logger.warning("Will store papers without embeddings - run batch embed later")
            self.model = None
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate Jina v4 embeddings for texts."""
        if not self.model:
            return None
        
        try:
            with torch.no_grad():
                embeddings = self.model.encode_text(
                    texts=texts,
                    task="retrieval",
                    batch_size=len(texts)
                )
                
                # Convert to CPU and then to list
                if hasattr(embeddings, 'cpu'):
                    embeddings = embeddings.cpu().numpy()
                elif torch.is_tensor(embeddings):
                    embeddings = embeddings.detach().cpu().numpy()
                
                return embeddings.tolist()
                
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return None
    
    def fetch_papers_for_date(self, date: datetime, max_results: int = 2000) -> List[Dict]:
        """
        Fetch papers submitted on a specific date.
        
        Args:
            date: Date to fetch papers for
            max_results: Maximum papers to fetch
        """
        # Format date for ArXiv API (YYYYMMDD)
        date_str = date.strftime("%Y%m%d")
        next_date_str = (date + timedelta(days=1)).strftime("%Y%m%d")
        
        # Build query for papers submitted on this date
        query = f"submittedDate:[{date_str}0000 TO {next_date_str}0000]"
        
        papers = []
        start = 0
        batch_size = 100
        
        while start < max_results:
            params = {
                'search_query': query,
                'start': start,
                'max_results': min(batch_size, max_results - start),
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            try:
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                
                # Parse XML response
                root = ET.fromstring(response.text)
                
                # Extract namespace
                ns = {'atom': 'http://www.w3.org/2005/Atom',
                      'arxiv': 'http://arxiv.org/schemas/atom'}
                
                # Find all entries
                entries = root.findall('atom:entry', ns)
                
                if not entries:
                    logger.info(f"No more papers found for {date_str}")
                    break
                
                for entry in entries:
                    paper = self._parse_entry(entry, ns)
                    if paper:
                        papers.append(paper)
                
                logger.info(f"Fetched {len(entries)} papers (total: {len(papers)})")
                start += batch_size
                
                # Rate limiting - ArXiv allows 3 requests per second
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error fetching papers: {e}")
                break
        
        return papers
    
    def _parse_entry(self, entry, ns) -> Dict:
        """Parse a single ArXiv entry."""
        try:
            # Extract ID from URL
            id_url = entry.find('atom:id', ns).text
            arxiv_id = id_url.split('/abs/')[-1].replace('v1', '').replace('v2', '').replace('v3', '')
            
            # Extract other fields
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            abstract = entry.find('atom:summary', ns).text.strip()
            
            # Authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns).text
                authors.append(name)
            
            # Categories
            categories = []
            primary_category = entry.find('arxiv:primary_category', ns)
            if primary_category is not None:
                categories.append(primary_category.get('term'))
            
            for category in entry.findall('atom:category', ns):
                cat_term = category.get('term')
                if cat_term and cat_term not in categories:
                    categories.append(cat_term)
            
            # Dates
            published = entry.find('atom:published', ns).text
            updated = entry.find('atom:updated', ns).text
            
            # DOI if available
            doi = None
            doi_elem = entry.find('arxiv:doi', ns)
            if doi_elem is not None:
                doi = doi_elem.text
            
            # Comment if available
            comment = None
            comment_elem = entry.find('arxiv:comment', ns)
            if comment_elem is not None:
                comment = comment_elem.text
            
            return {
                '_key': arxiv_id.replace('/', '_'),  # ArangoDB key format
                'arxiv_id': arxiv_id,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'categories': categories,
                'published_date': published,
                'updated_date': updated,
                'doi': doi,
                'comment': comment,
                'pdf_url': f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                'pdf_status': 'not_downloaded',
                'import_date': datetime.utcnow().isoformat(),
                'source': 'daily_update'
            }
            
        except Exception as e:
            logger.error(f"Error parsing entry: {e}")
            return None
    
    def update_database(self, papers: List[Dict]) -> Dict:
        """
        Update database with new papers and generate embeddings.
        
        Returns:
            Statistics about the update
        """
        stats = {
            'new': 0,
            'updated': 0,
            'embedded': 0,
            'errors': 0
        }
        
        # Process in batches for embedding
        batch_size = 32
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i+batch_size]
            
            # Generate embeddings for batch if model is available
            if self.model and batch:
                texts = [f"{p['title']}\n\n{p['abstract']}" for p in batch]
                embeddings = self.generate_embeddings(texts)
                
                if embeddings:
                    for j, paper in enumerate(batch):
                        paper['abstract_embeddings'] = embeddings[j]
                        paper['embedding_model'] = 'jinaai/jina-embeddings-v4'
                        paper['embedding_dim'] = 2048
                        paper['embedding_date'] = datetime.utcnow().isoformat()
                        paper['embedding_version'] = 'v4_daily_update'
                        stats['embedded'] += 1
            
            # Store papers in database
            for paper in batch:
                try:
                    # Check if paper exists
                    existing = self.collection.get(paper['_key'])
                    
                    if existing:
                        # Update if newer version
                        if existing.get('updated_date', '') < paper['updated_date']:
                            # Preserve existing embeddings if new ones weren't generated
                            if 'abstract_embeddings' not in paper and 'abstract_embeddings' in existing:
                                paper['abstract_embeddings'] = existing['abstract_embeddings']
                                paper['embedding_model'] = existing.get('embedding_model')
                                paper['embedding_dim'] = existing.get('embedding_dim')
                                paper['embedding_date'] = existing.get('embedding_date')
                                paper['embedding_version'] = existing.get('embedding_version')
                            
                            self.collection.update(paper)
                            stats['updated'] += 1
                            logger.debug(f"Updated {paper['arxiv_id']}")
                    else:
                        # Insert new paper
                        self.collection.insert(paper)
                        stats['new'] += 1
                        logger.debug(f"Added new paper {paper['arxiv_id']}")
                        
                except Exception as e:
                    logger.error(f"Error processing {paper.get('arxiv_id', 'unknown')}: {e}")
                    stats['errors'] += 1
        
        return stats
    
    def run_daily_update(self, days_back: int = 1):
        """
        Run the daily update process.
        
        Args:
            days_back: Number of days to look back (default 1 for yesterday)
        """
        logger.info("=" * 60)
        logger.info("Starting ArXiv daily update with Jina v4")
        logger.info("=" * 60)
        
        # Get yesterday's date (or specified days back)
        target_date = datetime.utcnow() - timedelta(days=days_back)
        logger.info(f"Fetching papers for {target_date.strftime('%Y-%m-%d')}")
        
        # Fetch papers
        papers = self.fetch_papers_for_date(target_date)
        logger.info(f"Fetched {len(papers)} papers from ArXiv")
        
        if papers:
            # Update database with embeddings
            stats = self.update_database(papers)
            
            logger.info("Update complete:")
            logger.info(f"  New papers: {stats['new']}")
            logger.info(f"  Updated papers: {stats['updated']}")
            logger.info(f"  Papers embedded: {stats['embedded']}")
            logger.info(f"  Errors: {stats['errors']}")
            
            # Log summary to database
            try:
                if 'daily_updates' not in self.db.collections():
                    self.db.create_collection('daily_updates')
                
                update_log = {
                    'date': target_date.isoformat(),
                    'run_time': datetime.utcnow().isoformat(),
                    'papers_fetched': len(papers),
                    'new_papers': stats['new'],
                    'updated_papers': stats['updated'],
                    'papers_embedded': stats['embedded'],
                    'errors': stats['errors'],
                    'embedding_model': 'jinaai/jina-embeddings-v4' if self.model else None
                }
                self.db.collection('daily_updates').insert(update_log)
            except Exception as e:
                logger.error(f"Failed to log update stats: {e}")
        else:
            logger.info("No papers found for the specified date")
        
        logger.info("=" * 60)
        logger.info("Daily update complete")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Daily ArXiv updater with Jina v4')
    parser.add_argument('--days-back', type=int, default=1,
                       help='Number of days to look back (default: 1 for yesterday)')
    parser.add_argument('--db-host', default='192.168.1.69',
                       help='ArangoDB host')
    parser.add_argument('--db-name', default='academy_store',
                       help='Database name')
    parser.add_argument('--skip-embeddings', action='store_true',
                       help='Skip embedding generation (fetch metadata only)')
    
    args = parser.parse_args()
    
    try:
        updater = ArXivDailyUpdater(args.db_host, args.db_name)
        
        # Optionally disable embeddings
        if args.skip_embeddings:
            logger.info("Skipping embeddings as requested")
            updater.model = None
        
        updater.run_daily_update(args.days_back)
    except Exception as e:
        logger.error(f"Update failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()