#!/usr/bin/env python3
"""
Production-ready ArXiv processing pipeline with Jina GPU embeddings.
Reliable, resumable, and fault-tolerant for automation.
"""

import os
import sys
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import time
import logging
import signal
import threading
import queue
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
import hashlib

import torch
import torch.cuda
import numpy as np
from arango import ArangoClient
from arango.http import DefaultHTTPClient
from arango.exceptions import DocumentInsertError, ArangoServerError
from transformers import AutoTokenizer, AutoModel

# Configure logging with both file and console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('arxiv_production.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Production configuration with sensible defaults"""
    # Input
    input_file: str = "/fastpool/temp/arxiv-metadata-oai-snapshot.json"
    
    # Database
    db_host: str = "192.168.1.69"
    db_port: int = 8529
    db_name: str = "academy_store"
    collection_name: str = "base_arxiv"
    db_username: str = "root"
    db_connection_retries: int = 5
    db_retry_delay: int = 5
    
    # Processing
    gpu_batch_size: int = 1024
    model_name: str = "jinaai/jina-embeddings-v3"
    device: str = "cuda:0"
    
    # Queue management
    output_queue_max: int = 50000
    output_queue_min: int = 25000
    db_batch_size: int = 5000
    db_write_timeout: int = 10
    
    # Checkpointing
    checkpoint_file: str = "arxiv_checkpoint.json"
    checkpoint_interval: int = 50000
    
    # Validation
    min_abstract_length: int = 50
    max_abstract_length: int = 2000
    validate_embeddings: bool = True
    embedding_dim: int = 1024  # Jina v3 dimension


@dataclass
class ProcessingStats:
    """Track processing statistics"""
    total_loaded: int = 0
    total_processed: int = 0
    total_written: int = 0
    total_errors: int = 0
    db_write_errors: int = 0
    gpu_errors: int = 0
    validation_errors: int = 0
    start_time: float = field(default_factory=time.time)
    
    def get_rate(self) -> float:
        elapsed = time.time() - self.start_time
        return self.total_processed / elapsed if elapsed > 0 else 0


class ProductionProcessor:
    """Production-grade processor with comprehensive error handling"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.stats = ProcessingStats()
        self.shutdown = False
        self.db = None
        self.collection = None
        
        # Threading controls
        self.gpu_paused = False
        self.output_queue = queue.Queue(maxsize=config.output_queue_max + 1000)
        
        # Data storage
        self.all_documents = []
        self.all_abstracts = []
        self.processed_ids: Set[str] = set()
        
        # Initialize components
        self._init_model()
        self._init_database_with_retry()
        self._setup_signal_handlers()
    
    def _init_model(self):
        """Initialize model with error handling"""
        try:
            logger.info(f"Loading model {self.config.model_name}...")
            
            # Check CUDA availability
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(self.config.device)
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModel.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(self.device)
            self.model.eval()
            
            # Verify encode method
            if not hasattr(self.model, 'encode'):
                raise RuntimeError("Model doesn't have encode method required for Jina embeddings")
            
            # GPU optimizations
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.cuda.set_per_process_memory_fraction(0.95)
                
                # Warmup
                logger.info("Warming up GPU...")
                with torch.no_grad():
                    _ = self.model.encode(["test"] * 32)
                torch.cuda.synchronize()
                
                # Report memory
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"GPU initialized: {total_mem:.1f}GB total memory")
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.critical(f"Failed to initialize model: {e}")
            raise
    
    def _init_database_with_retry(self):
        """Initialize database with retry logic"""
        for attempt in range(self.config.db_connection_retries):
            try:
                logger.info(f"Connecting to database (attempt {attempt + 1}/{self.config.db_connection_retries})...")
                
                client = ArangoClient(
                    hosts=f'http://{self.config.db_host}:{self.config.db_port}',
                    http_client=DefaultHTTPClient(
                        pool_connections=10,
                        pool_maxsize=10,
                        retry_attempts=3
                    )
                )
                
                password = os.environ.get('ARANGO_PASSWORD', '')
                sys_db = client.db('_system', username=self.config.db_username, password=password)
                
                # Create database if needed
                if not sys_db.has_database(self.config.db_name):
                    logger.info(f"Creating database {self.config.db_name}")
                    sys_db.create_database(self.config.db_name)
                
                # Connect to database
                self.db = client.db(self.config.db_name, username=self.config.db_username, password=password)
                
                # Create collection if needed
                if not self.db.has_collection(self.config.collection_name):
                    logger.info(f"Creating collection {self.config.collection_name}")
                    self.collection = self.db.create_collection(self.config.collection_name)
                    self.collection.add_hash_index(fields=['arxiv_id'], unique=True)
                else:
                    self.collection = self.db.collection(self.config.collection_name)
                
                # Verify connection
                count = self.collection.count()
                logger.info(f"Database connected. Collection has {count:,} documents")
                
                return
                
            except Exception as e:
                logger.error(f"Database connection attempt {attempt + 1} failed: {e}")
                if attempt < self.config.db_connection_retries - 1:
                    time.sleep(self.config.db_retry_delay)
                else:
                    logger.critical("Failed to connect to database after all retries")
                    raise
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown"""
        def handle_shutdown(signum, frame):
            logger.info("Shutdown signal received")
            self.shutdown = True
            self._save_checkpoint()
        
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)
    
    def _save_checkpoint(self):
        """Save processing checkpoint"""
        try:
            checkpoint = {
                'timestamp': datetime.utcnow().isoformat(),
                'stats': {
                    'total_loaded': self.stats.total_loaded,
                    'total_processed': self.stats.total_processed,
                    'total_written': self.stats.total_written,
                    'total_errors': self.stats.total_errors
                },
                'processed_ids': list(self.processed_ids)[-1000:],  # Save last 1000 IDs
                'file_hash': self._get_file_hash()
            }
            
            with open(self.config.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            logger.info(f"Checkpoint saved: {self.stats.total_processed:,} processed")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _get_file_hash(self):
        """Get hash of input file for validation"""
        try:
            with open(self.config.input_file, 'rb') as f:
                return hashlib.md5(f.read(1024 * 1024)).hexdigest()
        except:
            return None
    
    def _check_existing_progress(self) -> Set[str]:
        """Check what's already processed in database"""
        try:
            logger.info("Checking existing progress in database...")
            
            # Query in batches to avoid memory issues
            processed_ids = set()
            batch_size = 100000
            offset = 0
            
            while True:
                query = """
                FOR doc IN @@collection
                    FILTER doc.embedding != null
                    LIMIT @offset, @batch_size
                    RETURN doc.arxiv_id
                """
                
                cursor = self.db.aql.execute(
                    query,
                    bind_vars={
                        '@collection': self.config.collection_name,
                        'offset': offset,
                        'batch_size': batch_size
                    }
                )
                
                batch = list(cursor)
                if not batch:
                    break
                    
                processed_ids.update(batch)
                offset += batch_size
                
                if offset % 500000 == 0:
                    logger.info(f"  Checked {offset:,} documents...")
            
            logger.info(f"Found {len(processed_ids):,} already processed documents")
            return processed_ids
            
        except Exception as e:
            logger.warning(f"Could not check existing progress: {e}")
            return set()
    
    def load_dataset(self):
        """Load dataset with validation and deduplication"""
        logger.info("Loading dataset...")
        
        # Check for existing progress
        self.processed_ids = self._check_existing_progress()
        
        skipped_count = 0
        error_count = 0
        
        try:
            with open(self.config.input_file, 'r', buffering=64*1024*1024) as f:
                for line_num, line in enumerate(f):
                    if self.shutdown:
                        break
                    
                    if line_num % 100000 == 0 and line_num > 0:
                        logger.info(f"Loading: {line_num:,} lines read, {self.stats.total_loaded:,} kept, {skipped_count:,} skipped")
                    
                    try:
                        metadata = json.loads(line.strip())
                        arxiv_id = metadata.get('id', '')
                        
                        if not arxiv_id:
                            error_count += 1
                            continue
                        
                        # Skip if already processed
                        if arxiv_id in self.processed_ids:
                            skipped_count += 1
                            continue
                        
                        # Validate and clean abstract
                        abstract = metadata.get('abstract', '').strip()
                        
                        if len(abstract) < self.config.min_abstract_length:
                            skipped_count += 1
                            continue
                        
                        if len(abstract) > self.config.max_abstract_length:
                            abstract = abstract[:self.config.max_abstract_length] + "..."
                        
                        # Create document
                        document = {
                            '_key': arxiv_id.replace('/', '_'),
                            'arxiv_id': arxiv_id,
                            'title': metadata.get('title', '').strip(),
                            'authors': metadata.get('authors', '').strip(),
                            'categories': metadata.get('categories', '').strip().split(),
                            'abstract': abstract,
                            'created': metadata.get('created', ''),
                            'updated': metadata.get('updated', ''),
                            'journal_ref': metadata.get('journal-ref', ''),
                            'doi': metadata.get('doi', ''),
                            'comments': metadata.get('comments', ''),
                        }
                        
                        self.all_documents.append(document)
                        self.all_abstracts.append(abstract)
                        self.stats.total_loaded += 1
                        
                    except json.JSONDecodeError:
                        error_count += 1
                    except Exception as e:
                        error_count += 1
                        if error_count % 1000 == 0:
                            logger.warning(f"Parse errors: {error_count}")
        
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
        
        logger.info(f"Dataset loaded: {self.stats.total_loaded:,} documents to process")
        logger.info(f"  Skipped: {skipped_count:,} (already processed or invalid)")
        logger.info(f"  Errors: {error_count:,}")
        
        # Estimate memory
        text_size = sum(len(a) for a in self.all_abstracts) / 1e9
        logger.info(f"Memory usage: ~{text_size:.2f}GB text data")
    
    def gpu_processor(self):
        """GPU processing with comprehensive error handling"""
        logger.info("GPU processor started")
        
        if not self.all_abstracts:
            logger.warning("No documents to process")
            self.output_queue.put(None)
            return
        
        total_batches = (len(self.all_abstracts) + self.config.gpu_batch_size - 1) // self.config.gpu_batch_size
        logger.info(f"Processing {len(self.all_abstracts):,} documents in {total_batches:,} batches")
        
        batch_times = []
        consecutive_errors = 0
        max_consecutive_errors = 5
        last_checkpoint = 0
        
        for batch_idx in range(0, len(self.all_abstracts), self.config.gpu_batch_size):
            if self.shutdown:
                break
            
            # Output queue throttling
            while not self.shutdown:
                queue_size = self.output_queue.qsize()
                
                if queue_size >= self.config.output_queue_max:
                    if not self.gpu_paused:
                        logger.info(f"GPU pausing - output queue full ({queue_size:,})")
                        self.gpu_paused = True
                    time.sleep(0.1)
                elif self.gpu_paused and queue_size <= self.config.output_queue_min:
                    logger.info(f"GPU resuming - output queue at {queue_size:,}")
                    self.gpu_paused = False
                    break
                elif not self.gpu_paused:
                    break
            
            # Get batch
            batch_end = min(batch_idx + self.config.gpu_batch_size, len(self.all_abstracts))
            batch_texts = self.all_abstracts[batch_idx:batch_end]
            batch_docs = self.all_documents[batch_idx:batch_end]
            
            start_time = time.time()
            
            try:
                with torch.no_grad():
                    # Use Jina's encode for GPU tokenization + embedding
                    embeddings = self.model.encode(batch_texts)
                    
                    # Convert to numpy
                    if isinstance(embeddings, torch.Tensor):
                        embeddings = embeddings.cpu().numpy()
                    else:
                        embeddings = np.array(embeddings)
                    
                    # Validate embeddings
                    if self.config.validate_embeddings:
                        if embeddings.shape[0] != len(batch_docs):
                            raise ValueError(f"Embedding count mismatch: {embeddings.shape[0]} != {len(batch_docs)}")
                        if embeddings.shape[1] != self.config.embedding_dim:
                            raise ValueError(f"Embedding dimension mismatch: {embeddings.shape[1]} != {self.config.embedding_dim}")
                
                # Add embeddings to documents
                for doc, emb in zip(batch_docs, embeddings):
                    doc['embedding'] = emb.tolist()
                    doc['document_status'] = {
                        'embedded': True,
                        'embedding_date': datetime.utcnow().isoformat(),
                        'model': self.config.model_name
                    }
                
                # Queue for database writing
                self.output_queue.put(batch_docs)
                self.stats.total_processed += len(batch_docs)
                consecutive_errors = 0
                
                # Track performance
                gpu_time = time.time() - start_time
                batch_times.append(gpu_time)
                
                # Checkpoint periodically
                if self.stats.total_processed - last_checkpoint >= self.config.checkpoint_interval:
                    self._save_checkpoint()
                    last_checkpoint = self.stats.total_processed
                
                # Log progress
                if batch_idx % (self.config.gpu_batch_size * 10) == 0:
                    progress = (batch_idx / len(self.all_abstracts)) * 100
                    rate = len(batch_docs) / gpu_time if gpu_time > 0 else 0
                    avg_time = np.mean(batch_times[-100:]) if batch_times else 0
                    
                    vram_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    
                    logger.info(
                        f"Progress: {progress:.1f}% | Batch {batch_idx//self.config.gpu_batch_size}/{total_batches} | "
                        f"Rate: {rate:.0f} docs/s | Avg: {avg_time*1000:.0f}ms | "
                        f"Queue: {self.output_queue.qsize():,} | VRAM: {vram_used:.1f}GB"
                    )
                
            except torch.cuda.OutOfMemoryError:
                logger.error("GPU OOM - trying smaller batch")
                self.stats.gpu_errors += 1
                torch.cuda.empty_cache()
                
                # Try half batch
                if len(batch_texts) > 1:
                    half = len(batch_texts) // 2
                    self.all_abstracts = self.all_abstracts[:batch_idx] + batch_texts[:half] + batch_texts[half:] + self.all_abstracts[batch_end:]
                    self.all_documents = self.all_documents[:batch_idx] + batch_docs[:half] + batch_docs[half:] + self.all_documents[batch_end:]
                    continue
                    
            except Exception as e:
                logger.error(f"GPU processing error: {e}")
                self.stats.gpu_errors += 1
                consecutive_errors += 1
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(f"Too many consecutive GPU errors ({consecutive_errors})")
                    break
        
        # Signal completion
        self.output_queue.put(None)
        logger.info(f"GPU processor finished - processed {self.stats.total_processed:,} documents")
    
    def db_writer(self):
        """Database writer with robust error handling"""
        logger.info("Database writer started")
        
        write_buffer = []
        last_write_time = time.time()
        
        while not self.shutdown:
            try:
                # Get batch from queue
                batch = self.output_queue.get(timeout=1)
                
                if batch is None:
                    # GPU finished - write remaining
                    if write_buffer:
                        self._write_to_db_with_retry(write_buffer)
                    break
                
                write_buffer.extend(batch)
                
                # Write when buffer full or timeout
                if len(write_buffer) >= self.config.db_batch_size or \
                   (time.time() - last_write_time) > self.config.db_write_timeout:
                    
                    written = self._write_to_db_with_retry(write_buffer)
                    self.stats.total_written += written
                    write_buffer = []
                    last_write_time = time.time()
                
            except queue.Empty:
                # Timeout - write buffer if needed
                if write_buffer and (time.time() - last_write_time) > self.config.db_write_timeout:
                    written = self._write_to_db_with_retry(write_buffer)
                    self.stats.total_written += written
                    write_buffer = []
                    last_write_time = time.time()
                    
            except Exception as e:
                logger.error(f"DB writer error: {e}")
                self.stats.db_write_errors += 1
        
        logger.info(f"Database writer finished - wrote {self.stats.total_written:,} documents")
    
    def _write_to_db_with_retry(self, documents):
        """Write to database with retry logic"""
        if not documents:
            return 0
        
        for attempt in range(3):
            try:
                # Use overwrite=True for idempotency
                result = self.collection.insert_many(
                    documents,
                    overwrite=True,
                    silent=False,
                    sync=True
                )
                
                # Handle result
                if isinstance(result, dict):
                    inserted = result.get('inserted', 0)
                    updated = result.get('updated', 0)
                    errors = result.get('errors', 0)
                    
                    if errors > 0:
                        logger.warning(f"DB write had {errors} errors")
                        self.stats.db_write_errors += errors
                    
                    total = inserted + updated
                    if total > 0:
                        logger.debug(f"DB: {inserted} inserted, {updated} updated")
                    
                    # Update processed IDs
                    for doc in documents:
                        self.processed_ids.add(doc['arxiv_id'])
                    
                    return len(documents)
                else:
                    # Fallback for unexpected return type
                    logger.debug(f"Wrote {len(documents)} documents")
                    return len(documents)
                
            except ArangoServerError as e:
                logger.error(f"Database error (attempt {attempt + 1}): {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.stats.db_write_errors += len(documents)
                    return 0
                    
            except Exception as e:
                logger.error(f"Unexpected DB error: {e}")
                self.stats.db_write_errors += len(documents)
                return 0
        
        return 0
    
    def monitor(self):
        """Monitor progress and health"""
        last_processed = 0
        last_written = 0
        last_time = time.time()
        
        while not self.shutdown:
            time.sleep(5)
            
            current_time = time.time()
            interval = current_time - last_time
            
            # Calculate rates
            process_rate = (self.stats.total_processed - last_processed) / interval if interval > 0 else 0
            write_rate = (self.stats.total_written - last_written) / interval if interval > 0 else 0
            
            last_processed = self.stats.total_processed
            last_written = self.stats.total_written
            last_time = current_time
            
            # Get metrics
            queue_size = self.output_queue.qsize()
            overall_rate = self.stats.get_rate()
            
            # GPU memory
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated() / 1e9
                vram_reserved = torch.cuda.memory_reserved() / 1e9
            else:
                vram_used = vram_reserved = 0
            
            # Status
            if self.gpu_paused:
                status = "PAUSED"
            elif queue_size >= self.config.output_queue_max * 0.8:
                status = "THROTTLING"
            else:
                status = "RUNNING"
            
            # Progress
            progress = (self.stats.total_processed / self.stats.total_loaded * 100) if self.stats.total_loaded > 0 else 0
            
            logger.info(
                f"[{status}] Progress: {progress:.1f}% | "
                f"Processed: {self.stats.total_processed:,}/{self.stats.total_loaded:,} | "
                f"Written: {self.stats.total_written:,} | "
                f"Rate: {process_rate:.0f}/s (avg: {overall_rate:.0f}/s) | "
                f"Queue: {queue_size:,} | "
                f"VRAM: {vram_used:.1f}/{vram_reserved:.1f}GB"
            )
            
            # Error summary
            if self.stats.total_errors > 0:
                logger.info(
                    f"  Errors - Total: {self.stats.total_errors} | "
                    f"GPU: {self.stats.gpu_errors} | "
                    f"DB: {self.stats.db_write_errors}"
                )
            
            # ETA
            if process_rate > 0:
                remaining = self.stats.total_loaded - self.stats.total_processed
                eta_seconds = remaining / process_rate
                eta_minutes = eta_seconds / 60
                logger.info(f"  ETA: {eta_minutes:.1f} minutes at current rate")
    
    def run(self):
        """Main execution"""
        logger.info("=" * 70)
        logger.info("ARXIV PRODUCTION PROCESSING PIPELINE")
        logger.info("=" * 70)
        logger.info(f"Configuration:")
        logger.info(f"  Model: {self.config.model_name}")
        logger.info(f"  Batch Size: {self.config.gpu_batch_size}")
        logger.info(f"  Database: {self.config.db_name}/{self.config.collection_name}")
        logger.info(f"  Checkpoint Interval: {self.config.checkpoint_interval:,}")
        logger.info("=" * 70)
        
        try:
            # Load dataset
            load_start = time.time()
            self.load_dataset()
            load_time = time.time() - load_start
            logger.info(f"Dataset loaded in {load_time:.1f} seconds")
            
            if self.stats.total_loaded == 0:
                logger.info("No documents to process - all done!")
                return
            
            # Start processing threads
            threads = [
                threading.Thread(target=self.gpu_processor, name="GPUProcessor"),
                threading.Thread(target=self.db_writer, name="DBWriter"),
                threading.Thread(target=self.monitor, name="Monitor"),
            ]
            
            for t in threads:
                t.start()
            
            # Wait for completion
            for t in threads:
                if t.name != "Monitor":
                    t.join()
            
            # Stop monitor
            self.shutdown = True
            
            # Final statistics
            total_time = time.time() - self.stats.start_time
            
            logger.info("=" * 70)
            logger.info("PROCESSING COMPLETE")
            logger.info("=" * 70)
            logger.info(f"Statistics:")
            logger.info(f"  Total Processed: {self.stats.total_processed:,}")
            logger.info(f"  Total Written: {self.stats.total_written:,}")
            logger.info(f"  Total Errors: {self.stats.total_errors}")
            logger.info(f"  Time: {total_time/60:.1f} minutes")
            logger.info(f"  Average Rate: {self.stats.get_rate():.1f} docs/s")
            
            if self.stats.total_errors > 0:
                logger.warning(f"Errors encountered:")
                logger.warning(f"  GPU Errors: {self.stats.gpu_errors}")
                logger.warning(f"  DB Write Errors: {self.stats.db_write_errors}")
                logger.warning(f"  Validation Errors: {self.stats.validation_errors}")
            
            # Clean up checkpoint on success
            if self.stats.total_processed == self.stats.total_loaded and \
               os.path.exists(self.config.checkpoint_file):
                os.remove(self.config.checkpoint_file)
                logger.info("Checkpoint file removed (processing complete)")
            
            logger.info("=" * 70)
            
        except Exception as e:
            logger.critical(f"Fatal error: {e}", exc_info=True)
            self._save_checkpoint()
            raise


def main():
    """Entry point with configuration overrides"""
    config = ProcessingConfig()
    
    # Allow environment variable overrides
    if os.environ.get('GPU_BATCH_SIZE'):
        config.gpu_batch_size = int(os.environ['GPU_BATCH_SIZE'])
    if os.environ.get('DB_HOST'):
        config.db_host = os.environ['DB_HOST']
    if os.environ.get('DB_NAME'):
        config.db_name = os.environ['DB_NAME']
    if os.environ.get('COLLECTION_NAME'):
        config.collection_name = os.environ['COLLECTION_NAME']
    
    # Create processor and run
    processor = ProductionProcessor(config)
    
    try:
        processor.run()
        sys.exit(0)  # Success
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Processing failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()