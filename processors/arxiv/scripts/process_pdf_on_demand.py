#!/usr/bin/env python3
"""
On-demand PDF processor - downloads, processes, and embeds a single paper.

Theory Connection:
Implements lazy evaluation of the information space - documents exist
in potential until observed (requested), then materialize into our
high-dimensional representation.
"""

import os
import sys
import logging
import requests
from pathlib import Path
import tempfile
import time
from typing import Dict, Optional

sys.path.append(str(Path(__file__).parent.parent))
from core.enhanced_docling_processor_v2 import EnhancedDoclingProcessorV2
from core.batch_embed_jina import JinaEmbedderV3

from arango import ArangoClient

logger = logging.getLogger(__name__)


class OnDemandProcessor:
    """Process ArXiv papers on demand."""
    
    def __init__(self, db_host: str = "192.168.1.69", db_name: str = "academy_store"):
        self.db_host = db_host
        self.db_name = db_name
        self._init_database()
        
        # Initialize processors
        self.docling_processor = EnhancedDoclingProcessorV2()
        # Use Jina v4 with larger context window and late chunking
        self.embedder = JinaEmbedderV3(  # Class name kept for compatibility, but uses v4 model
            device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
            chunk_size=28000,  # Jina v4 supports 32k tokens (leaving room for special tokens)
            chunk_overlap=5600,  # 20% overlap
            use_late_chunking=True  # Enable late chunking for better context preservation
        )
    
    def _init_database(self):
        """Initialize database connection."""
        password = os.environ.get('ARANGO_PASSWORD')
        if not password:
            raise ValueError("ARANGO_PASSWORD environment variable required")
        
        client = ArangoClient(hosts=f'http://{self.db_host}:8529')
        self.db = client.db(self.db_name, username='root', password=password)
        self.collection = self.db.collection('base_arxiv')
    
    def process_paper(self, arxiv_id: str) -> Dict:
        """
        Process a single paper end-to-end.
        
        Steps:
        1. Check if already processed
        2. Download PDF if needed
        3. Extract text and images
        4. Generate embeddings
        5. Store in database
        """
        start_time = time.time()
        
        # Normalize arxiv_id
        doc_key = arxiv_id.replace('/', '_')
        
        # Check current status
        doc = self.collection.get(doc_key)
        if not doc:
            return {'error': f'Paper {arxiv_id} not found in database'}
        
        if doc.get('pdf_status') == 'embedded':
            return {
                'status': 'already_processed',
                'arxiv_id': arxiv_id,
                'message': 'Paper already processed and embedded'
            }
        
        # Download PDF if needed
        pdf_path = None
        if doc.get('pdf_status') != 'downloaded':
            # Construct PDF URL from arxiv_id
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            pdf_path = self._download_pdf(arxiv_id, pdf_url)
            if not pdf_path:
                return {'error': f'Failed to download PDF for {arxiv_id}'}
        else:
            pdf_path = Path(doc.get('pdf_local_path'))
            if not pdf_path.exists():
                # Re-download if local file is missing
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                pdf_path = self._download_pdf(arxiv_id, pdf_url)
        
        # Process with Docling (extract text, equations, images)
        logger.info(f"Processing {arxiv_id} with Docling...")
        docling_result = self.docling_processor.process_pdf(
            arxiv_id, 
            str(pdf_path),
            max_images=20,
            skip_tiny_images=True
        )
        
        if not docling_result['success']:
            return {'error': f"Docling processing failed: {docling_result.get('error')}"}
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {arxiv_id}...")
        markdown_path = Path(docling_result['output_path'])
        doc_embeddings = self.embedder.process_document(arxiv_id, markdown_path)
        
        if not doc_embeddings:
            return {'error': 'Failed to generate embeddings'}
        
        # Update database
        logger.info(f"Updating database for {arxiv_id}...")
        update_data = {
            'pdf_status': 'embedded',
            'pdf_local_path': str(pdf_path),
            'full_text': doc_embeddings.full_text,
            'embeddings': {
                'model': doc_embeddings.model_name,
                'chunk_size': doc_embeddings.chunk_size,
                'chunk_overlap': doc_embeddings.chunk_overlap,
                'chunks': [
                    {
                        'chunk_index': chunk.chunk_index,
                        'chunk_text': chunk.chunk_text,
                        'embedding': chunk.embedding,
                        'metadata': chunk.metadata
                    }
                    for chunk in doc_embeddings.chunks
                ],
                'num_chunks': len(doc_embeddings.chunks),
                'total_tokens': doc_embeddings.total_tokens,
                'processing_time': doc_embeddings.processing_time,
                'embedded_date': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            },
            'processing_stats': docling_result.get('stats', {}),
            'num_equations': docling_result.get('num_equations', 0),
            'num_images': docling_result.get('num_images', 0)
        }
        
        try:
            # Use update_match which properly merges the data
            result = self.collection.update_match({'_key': doc_key}, update_data)
            logger.info(f"Database update result: {result}")
            
            # Verify the update worked
            updated_doc = self.collection.get(doc_key)
            if updated_doc.get('pdf_status') == 'embedded':
                logger.info(f"Successfully updated {arxiv_id} with embeddings")
            else:
                logger.warning(f"Update may have failed - pdf_status not set")
        except Exception as e:
            logger.error(f"Failed to update database: {e}")
            return {'error': f'Database update failed: {e}'}
        
        total_time = time.time() - start_time
        
        return {
            'status': 'success',
            'arxiv_id': arxiv_id,
            'processing_time': total_time,
            'num_chunks': len(doc_embeddings.chunks),
            'num_equations': docling_result.get('num_equations', 0),
            'num_images': docling_result.get('num_images', 0),
            'total_tokens': doc_embeddings.total_tokens
        }
    
    def _download_pdf(self, arxiv_id: str, pdf_url: str) -> Optional[Path]:
        """Download PDF from ArXiv."""
        try:
            # Create temp directory for PDFs
            pdf_dir = Path('/tmp/arxiv_pdfs')
            pdf_dir.mkdir(exist_ok=True)
            
            pdf_path = pdf_dir / f"{arxiv_id.replace('/', '_')}.pdf"
            
            if pdf_path.exists():
                logger.info(f"PDF already exists at {pdf_path}")
                return pdf_path
            
            logger.info(f"Downloading {pdf_url}...")
            response = requests.get(pdf_url, timeout=60)
            response.raise_for_status()
            
            pdf_path.write_bytes(response.content)
            logger.info(f"Downloaded {len(response.content)} bytes to {pdf_path}")
            
            return pdf_path
            
        except Exception as e:
            logger.error(f"Failed to download PDF: {e}")
            return None


def main():
    """Process a single paper on demand."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process ArXiv paper on demand')
    parser.add_argument('arxiv_id', help='ArXiv ID (e.g., 2301.00234)')
    parser.add_argument('--db-host', default='192.168.1.69')
    parser.add_argument('--db-name', default='academy_store')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    processor = OnDemandProcessor(args.db_host, args.db_name)
    result = processor.process_paper(args.arxiv_id)
    
    if result.get('status') == 'success':
        print(f"✅ Successfully processed {args.arxiv_id}")
        print(f"   Time: {result['processing_time']:.2f}s")
        print(f"   Chunks: {result['num_chunks']}")
        print(f"   Equations: {result['num_equations']}")
        print(f"   Images: {result['num_images']}")
    else:
        print(f"❌ Failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()