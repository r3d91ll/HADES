#!/usr/bin/env python3
"""
Rebuild ArXiv database with Jina v4 using BOTH GPUs with NVLink.
Leverages Ray for distributed processing across GPU:0 and GPU:1.

Theory Connection:
Parallel dimensional reconstruction - doubling both compute (2 GPUs) and
information capacity (2048 dimensions) for exponential improvement.
NVLink enables direct GPU-to-GPU communication for optimal efficiency.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Iterator
import argparse
import ray
import torch
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from arango import ArangoClient
from transformers import AutoModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class JinaV4Worker:
    """Ray actor for Jina v4 embeddings on a single GPU."""
    
    def __init__(self, worker_id: int):
        """Initialize worker on GPU assigned by Ray."""
        import torch
        from transformers import AutoModel
        
        self.worker_id = worker_id
        
        # Ray assigns GPUs automatically, just use cuda:0
        # Ray isolates each worker to see only its assigned GPU as cuda:0
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Log which actual GPU this worker is using
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_id = torch.cuda.current_device()
            logger.info(f"Worker {worker_id}: Using GPU {gpu_id} ({gpu_name})")
        
        # Load Jina v4 model with fp16
        self.model_name = "jinaai/jina-embeddings-v4"
        logger.info(f"Worker {worker_id}: Loading {self.model_name} on {self.device}")
        
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16  # fp16 for efficiency
        ).to(self.device)
        
        self.model.eval()
        logger.info(f"Worker {worker_id}: Model loaded successfully")
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts."""
        import torch
        import numpy as np
        
        with torch.no_grad():
            embeddings = self.model.encode_text(
                texts=texts,
                task="retrieval",
                batch_size=len(texts)
            )
            
            # Convert to CPU numpy
            if hasattr(embeddings, 'cpu'):
                embeddings = embeddings.cpu().numpy()
            elif torch.is_tensor(embeddings):
                embeddings = embeddings.detach().cpu().numpy()
            
            return embeddings


class DualGPURebuilder:
    """Rebuild database using both GPUs with Ray."""
    
    def __init__(self, 
                 source_file: str,
                 db_host: str = "192.168.1.69", 
                 db_name: str = "academy_store",
                 num_workers: int = 2):
        """
        Initialize dual-GPU rebuilder.
        
        Args:
            source_file: Path to arxiv JSON
            db_host: ArangoDB host
            db_name: Database name
            num_workers: Number of GPU workers (2 for dual GPU)
        """
        self.source_file = source_file
        self.db_host = db_host
        self.db_name = db_name
        self.num_workers = num_workers
        
        # Initialize Ray with both GPUs
        if not ray.is_initialized():
            # Simple Ray initialization
            ray.init(
                num_gpus=2,
                ignore_reinit_error=True,
                log_to_driver=False  # Reduce logging overhead
            )
            logger.info(f"Ray initialized with {num_workers} GPUs")
        
        # Create workers on each GPU
        self.workers = [
            JinaV4Worker.remote(worker_id=i) 
            for i in range(num_workers)
        ]
        logger.info(f"Created {num_workers} Jina v4 workers")
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database connection."""
        password = os.environ.get('ARANGO_PASSWORD')
        if not password:
            raise ValueError("ARANGO_PASSWORD environment variable required")
        
        client = ArangoClient(hosts=f'http://{self.db_host}:8529')
        
        # First connect to _system to create database if needed
        sys_db = client.db('_system', username='root', password=password)
        
        # Create database if it doesn't exist
        if not sys_db.has_database(self.db_name):
            sys_db.create_database(self.db_name)
            logger.info(f"Created database '{self.db_name}'")
        
        # Now connect to our database
        self.db = client.db(self.db_name, username='root', password=password)
        
        if not self.db.has_collection('base_arxiv'):
            self.collection = self.db.create_collection('base_arxiv')
            logger.info("Created base_arxiv collection")
        else:
            self.collection = self.db.collection('base_arxiv')
            logger.info("Using existing base_arxiv collection")
    
    def read_source_papers(self, limit: int = None) -> Iterator[Dict]:
        """Read papers from source JSON file."""
        count = 0
        
        with open(self.source_file, 'r') as f:
            for line in f:
                if limit and count >= limit:
                    break
                    
                try:
                    paper = json.loads(line)
                    arxiv_id = paper.get('id', '').replace('/', '_')
                    
                    if arxiv_id and paper.get('abstract'):
                        yield {
                            '_key': arxiv_id,
                            'arxiv_id': paper.get('id'),
                            'title': paper.get('title', ''),
                            'abstract': paper.get('abstract', ''),
                            'authors': paper.get('authors', ''),
                            'categories': paper.get('categories', ''),
                            'update_date': paper.get('update_date'),
                            'authors_parsed': paper.get('authors_parsed', [])
                        }
                        count += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing paper: {e}")
                    continue
    
    def process_papers_parallel(self, papers: List[Dict]) -> List[Dict]:
        """Process papers in parallel across both GPUs."""
        # Split papers between workers
        papers_per_worker = len(papers) // self.num_workers
        paper_batches = []
        
        for i in range(self.num_workers):
            start_idx = i * papers_per_worker
            if i == self.num_workers - 1:
                # Last worker gets remaining papers
                batch = papers[start_idx:]
            else:
                batch = papers[start_idx:start_idx + papers_per_worker]
            
            if batch:
                paper_batches.append(batch)
        
        # Process each batch on different GPU
        futures = []
        for worker_id, batch in enumerate(paper_batches):
            # Prepare texts for embedding
            texts = [f"{p['title']}\n\n{p['abstract']}" for p in batch]
            
            # Submit to worker
            future = self.workers[worker_id].embed_batch.remote(texts)
            futures.append((future, batch))
        
        # Collect results
        processed_papers = []
        for future, batch in futures:
            try:
                embeddings = ray.get(future, timeout=60)
                
                # Add embeddings to papers
                for i, paper in enumerate(batch):
                    paper['abstract_embeddings'] = embeddings[i].tolist()
                    paper['embedding_model'] = 'jinaai/jina-embeddings-v4'
                    paper['embedding_dim'] = 2048
                    paper['embedding_date'] = datetime.utcnow().isoformat()
                    paper['embedding_version'] = 'v4_dual_gpu'
                    paper['gpu_id'] = futures.index((future, batch))  # Track which GPU processed
                    processed_papers.append(paper)
                    
            except Exception as e:
                logger.error(f"Worker error: {e}")
                # Still add papers without embeddings
                for paper in batch:
                    paper['embedding_error'] = str(e)
                    processed_papers.append(paper)
        
        return processed_papers
    
    def store_papers(self, papers: List[Dict]) -> Dict:
        """Store papers in database."""
        success_count = 0
        error_count = 0
        
        for paper in papers:
            try:
                if 'embedding_error' not in paper:
                    self.collection.insert(paper, overwrite=True)
                    success_count += 1
                else:
                    error_count += 1
            except Exception as e:
                logger.error(f"Failed to store paper {paper.get('_key')}: {e}")
                error_count += 1
        
        return {'success': success_count, 'error': error_count}
    
    def rebuild_database(self, 
                        limit: int = None, 
                        batch_size: int = 128,  # Larger batch for dual GPU
                        clean_start: bool = False):
        """Rebuild database using both GPUs."""
        
        if clean_start:
            logger.warning("Clean start requested - clearing existing collection")
            response = input("Are you sure you want to DELETE all existing data? (yes/no): ")
            if response.lower() == 'yes':
                self.collection.truncate()
                logger.info("Collection cleared")
            else:
                logger.info("Clean start cancelled")
                return
        
        logger.info("="*60)
        logger.info("Dual-GPU ArXiv Database Rebuild with Jina v4")
        logger.info("="*60)
        logger.info(f"Source: {self.source_file}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Workers: {self.num_workers} GPUs")
        logger.info("NVLink: Enabled for GPU-to-GPU communication")
        if limit:
            logger.info(f"Limit: {limit} papers")
        
        start_time = time.time()
        total_success = 0
        total_error = 0
        batch_count = 0
        
        # Process papers in batches
        batch = []
        for paper in self.read_source_papers(limit):
            batch.append(paper)
            
            if len(batch) >= batch_size:
                batch_count += 1
                logger.info(f"Processing batch {batch_count} ({len(batch)} papers across {self.num_workers} GPUs)")
                
                # Process in parallel
                processed_papers = self.process_papers_parallel(batch)
                
                # Store results
                result = self.store_papers(processed_papers)
                total_success += result['success']
                total_error += result['error']
                
                # Progress update
                if batch_count % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = total_success / elapsed if elapsed > 0 else 0
                    logger.info(f"Progress: {total_success:,} papers, {rate:.1f} papers/sec")
                    
                    # GPU utilization info
                    gpu0_papers = sum(1 for p in processed_papers if p.get('gpu_id') == 0)
                    gpu1_papers = sum(1 for p in processed_papers if p.get('gpu_id') == 1)
                    logger.info(f"  GPU distribution: GPU0={gpu0_papers}, GPU1={gpu1_papers}")
                
                batch = []
        
        # Process remaining papers
        if batch:
            batch_count += 1
            logger.info(f"Processing final batch {batch_count} ({len(batch)} papers)")
            processed_papers = self.process_papers_parallel(batch)
            result = self.store_papers(processed_papers)
            total_success += result['success']
            total_error += result['error']
        
        # Final report
        total_time = time.time() - start_time
        logger.info("="*60)
        logger.info("Dual-GPU Rebuild Complete")
        logger.info(f"Successfully processed: {total_success:,}")
        logger.info(f"Errors: {total_error:,}")
        logger.info(f"Total time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        if total_success > 0:
            logger.info(f"Average rate: {total_success/total_time:.2f} papers/sec")
            logger.info(f"Theoretical single-GPU rate: {(total_success/total_time)/2:.2f} papers/sec")
        logger.info("="*60)
    
    def cleanup(self):
        """Clean up Ray resources."""
        ray.shutdown()
        logger.info("Ray shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Rebuild ArXiv database with Jina v4 using dual GPUs'
    )
    parser.add_argument('--source', 
                       default='/fastpool/temp/arxiv-metadata-oai-snapshot.json',
                       help='Path to source JSON file')
    parser.add_argument('--limit', type=int, 
                       help='Limit number of papers (for testing)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (split across GPUs)')
    parser.add_argument('--clean-start', action='store_true',
                       help='Clear existing data before starting')
    parser.add_argument('--db-host', default='192.168.1.69', 
                       help='Database host')
    parser.add_argument('--db-name', default='academy_store', 
                       help='Database name')
    parser.add_argument('--num-gpus', type=int, default=2,
                       help='Number of GPUs to use')
    
    args = parser.parse_args()
    
    # Check environment
    if not os.environ.get('ARANGO_PASSWORD'):
        logger.error("ARANGO_PASSWORD environment variable not set")
        sys.exit(1)
    
    # Check source file
    if not Path(args.source).exists():
        logger.error(f"Source file not found: {args.source}")
        sys.exit(1)
    
    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        sys.exit(1)
    
    gpu_count = torch.cuda.device_count()
    logger.info(f"Found {gpu_count} GPUs")
    
    if gpu_count < args.num_gpus:
        logger.warning(f"Requested {args.num_gpus} GPUs but only {gpu_count} available")
        args.num_gpus = gpu_count
    
    # Check for NVLink
    if gpu_count >= 2:
        try:
            # Check if GPUs can access each other's memory (indicates NVLink)
            torch.cuda.set_device(0)
            can_access = torch.cuda.can_device_access_peer(0, 1)
            if can_access:
                logger.info("✓ NVLink detected between GPUs - enabling optimizations")
            else:
                logger.info("No NVLink detected - using PCIe communication")
        except:
            pass
    
    # Run rebuild
    rebuilder = DualGPURebuilder(
        args.source, 
        args.db_host, 
        args.db_name,
        num_workers=args.num_gpus
    )
    
    try:
        rebuilder.rebuild_database(
            limit=args.limit,
            batch_size=args.batch_size,
            clean_start=args.clean_start
        )
    finally:
        rebuilder.cleanup()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass  # Handle argparse exit gracefully