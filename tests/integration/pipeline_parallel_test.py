"""
Integration test for the parallel processing pipeline up to ISNE.

This script tests processing multiple PDFs through the document processing,
chunking, and embedding stages using parallel processing, to validate
the pipeline components leading up to the ISNE module.
"""

import os
import sys
import logging
import time
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import argparse
import uuid
import threading
from datetime import datetime

import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.docproc.manager import DocumentProcessorManager
from src.chunking.text_chunkers.chonky_chunker import chunk_text
from src.embedding.adapters.modernbert_adapter import ModernBERTEmbeddingAdapter
from src.types.common import EmbeddingVector

# Create custom log formatter with timestamps and elapsed time
class TimingFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)
        self.start_time = time.time()
    
    def formatTime(self, record, datefmt=None):
        # Format the time as usual
        formatted_time = super().formatTime(record, datefmt)
        # Add elapsed time since start
        elapsed = time.time() - self.start_time
        return f"{formatted_time} [+{elapsed:.3f}s]"

# Configure logging with custom formatter
formatter = TimingFormatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Remove existing handlers
for handler in root_logger.handlers:
    root_logger.removeHandler(handler)

# Add console handler with custom formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

# Get our module logger
logger = logging.getLogger(__name__)


class PipelineParallelTester:
    """Test parallel processing of multiple PDFs through the pipeline."""
    
    def __init__(
        self,
        test_data_dir: str,
        output_dir: str,
        num_files: int = 20,
        max_workers: int = 4,
        batch_size: int = 8
    ):
        """
        Initialize the parallel pipeline tester.
        
        Args:
            test_data_dir: Directory containing test PDF files
            output_dir: Directory to save output data
            num_files: Number of files to process
            max_workers: Maximum number of parallel workers
            batch_size: Batch size for processing
        """
        self.test_data_dir = Path(test_data_dir)
        self.output_dir = Path(output_dir)
        self.num_files = num_files
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components with timing information
        logger.info("Initializing pipeline components...")
        
        # Document processor initialization
        doc_proc_start = time.time()
        self.doc_processor = DocumentProcessorManager()
        doc_proc_time = time.time() - doc_proc_start
        logger.info(f"Document processor initialized in {doc_proc_time:.2f}s")
        
        # Embedding adapter initialization
        embed_start = time.time()
        self.embedding_adapter = ModernBERTEmbeddingAdapter(device="cuda:1")
        embed_time = time.time() - embed_start
        logger.info(f"Embedding adapter initialized in {embed_time:.2f}s")
        
        # Log component initialization details
        self.component_init_time = {
            "document_processor": doc_proc_time,
            "embedding_adapter": embed_time,
            "total": doc_proc_time + embed_time
        }
        
        logger.info(f"All pipeline components initialized in {self.component_init_time['total']:.2f}s")
        
        # Statistics
        self.stats = {
            "total_files": 0,
            "total_chunks": 0,
            "total_embeddings": 0,
            "processing_time": {
                "document_processing": 0,
                "chunking": 0,
                "embedding": 0,
                "total": 0
            },
            "file_details": {}
        }
    
    def find_pdf_files(self) -> List[Path]:
        """Find PDF files in the test directory."""
        pdf_files = sorted(list(self.test_data_dir.glob("*.pdf")))
        logger.info(f"Found {len(pdf_files)} PDF files in {self.test_data_dir}")
        
        if len(pdf_files) > self.num_files:
            # Select a representative subset
            return pdf_files[:self.num_files]
        return pdf_files
    
    async def process_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single document through the pipeline.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Results of processing including timing information
        """
        # Generate a consistent ID for the file based on its path
        file_id = f"pdf_{abs(hash(str(file_path))) % 10000:04d}_{file_path.stem}"
        
        logger.info(f"[Worker {os.getpid()}-{threading.get_ident() % 10000}] Starting processing of {file_path.name}")
        total_start = time.time()
        
        result = {
            "file_id": file_id,
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
            "worker_id": f"{os.getpid()}-{threading.get_ident() % 10000}",  # Track which worker processed this file
            "timing": {
                "document_processing": 0,
                "chunking": 0,
                "embedding": 0,
                "total": 0
            }
        }
        
        # Step 1: Document processing
        doc_start = time.time()
        try:
            worker_id = result['worker_id']
            logger.info(f"[Worker {worker_id}] Starting document processing: {file_path.name} ({file_path.stat().st_size/1024:.1f} KB)")
            
            # Track internal timing of document processing
            doc_proc_details = {
                "start": time.time(),
                "preprocessing": 0,
                "content_extraction": 0,
                "metadata_extraction": 0,
                "total": 0
            }
            
            # Process the document
            processed_doc = self.doc_processor.process_document(path=str(file_path))
            
            doc_proc_details["total"] = time.time() - doc_proc_details["start"]
            
            # Check if processing was successful
            if not processed_doc or not processed_doc.get("content"):
                logger.warning(f"[Worker {worker_id}] Failed to process document: {file_path.name}")
                return result
                
            # Log detailed information about the processed document
            content_length = len(str(processed_doc.get("content", ""))) 
            metadata_count = len(processed_doc.get("metadata", {}))
            
            logger.info(f"[Worker {worker_id}] Document processed: {file_path.name} - Content size: {content_length/1024:.1f} KB, Metadata fields: {metadata_count}")
            
        except Exception as e:
            logger.error(f"[Worker {result['worker_id']}] Error processing document {file_path.name}: {str(e)}")
            return result
        
        doc_time = time.time() - doc_start
        result["timing"]["document_processing"] = doc_time
        result["doc_proc_details"] = doc_proc_details
        
        # Step 2: Chunking
        chunk_start = time.time()
        try:
            worker_id = result['worker_id']
            # Get document properties
            doc_id = processed_doc.get('id', file_id)
            doc_path = processed_doc.get('source', str(file_path))
            
            logger.info(f"[Worker {worker_id}] Starting document chunking: {file_path.name}")
            
            # Track internal timing of chunking
            chunk_details = {
                "start": time.time(),
                "content_extraction": 0,
                "model_inference": 0,
                "post_processing": 0,
                "total": 0
            }
            
            # Prepare for chunking - extract content string
            content_start = time.time()
            doc_content = processed_doc.get("content", "")
            content_size = len(str(doc_content))
            chunk_details["content_extraction"] = time.time() - content_start
            
            logger.info(f"[Worker {worker_id}] Extracted content for chunking: {content_size/1024:.1f} KB")
            
            # Process through chunking based on reference implementation
            model_start = time.time()
            chunks_result = chunk_text(
                content=doc_content,
                doc_id=doc_id,
                path=doc_path,
                doc_type="academic_pdf",
                max_tokens=1024,
                output_format="json"
            )
            chunk_details["model_inference"] = time.time() - model_start
            
            # Extract chunks from the chunking result (post-processing)
            post_start = time.time()
            if isinstance(chunks_result, dict) and "chunks" in chunks_result:
                chunks = chunks_result.get("chunks", [])
            else:
                # Direct list of chunks
                chunks = chunks_result if isinstance(chunks_result, list) else []
            
            if not chunks:
                logger.warning(f"[Worker {worker_id}] No chunks created for document: {file_path.name}")
                chunk_details["post_processing"] = time.time() - post_start
                chunk_details["total"] = time.time() - chunk_details["start"]
                result["chunk_details"] = chunk_details
                return result
            
            # Add chunks to result
            result["chunks"] = chunks
            result["chunk_count"] = len(chunks)
            chunk_details["post_processing"] = time.time() - post_start
            chunk_details["total"] = time.time() - chunk_details["start"]
            
            # Calculate stats on chunks
            chunk_sizes = [len(str(chunk.get("content", ""))) for chunk in chunks]
            avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
            max_chunk_size = max(chunk_sizes) if chunk_sizes else 0
            min_chunk_size = min(chunk_sizes) if chunk_sizes else 0
            
            logger.info(f"[Worker {worker_id}] Created {len(chunks)} chunks for {file_path.name} - "
                       f"Avg size: {avg_chunk_size/1024:.1f} KB, Range: {min_chunk_size/1024:.1f}-{max_chunk_size/1024:.1f} KB")
            
            # Store chunk statistics
            result["chunk_stats"] = {
                "count": len(chunks),
                "avg_size": avg_chunk_size,
                "max_size": max_chunk_size,
                "min_size": min_chunk_size,
                "total_size": sum(chunk_sizes),
                "compression_ratio": content_size / sum(chunk_sizes) if sum(chunk_sizes) > 0 else 0
            }
            result["chunk_details"] = chunk_details
        except Exception as e:
            logger.error(f"Error chunking document {file_path.name}: {str(e)}")
            return result
        
        chunk_time = time.time() - chunk_start
        result["timing"]["chunking"] = chunk_time
        
        # Step 3: Embedding generation
        embed_start = time.time()
        try:
            worker_id = result['worker_id']
            
            # Track internal timing of embedding generation
            embed_details = {
                "start": time.time(),
                "text_extraction": 0,
                "model_inference": 0,
                "post_processing": 0,
                "total": 0,
                "batches": []
            }
            
            # Extract text content from each chunk for embedding
            text_start = time.time()
            texts = [chunk.get("content", "") for chunk in result["chunks"]]
            embed_details["text_extraction"] = time.time() - text_start
            
            if not texts:
                logger.warning(f"[Worker {worker_id}] No text content to embed for {file_path.name}")
                embed_details["total"] = time.time() - embed_details["start"]
                result["embed_details"] = embed_details
                return result
            
            # Calculate total text size for embedding
            total_text_size = sum(len(text) for text in texts)
            avg_text_size = total_text_size / len(texts) if texts else 0
            
            logger.info(f"[Worker {worker_id}] Starting embedding generation for {len(texts)} chunks from {file_path.name} "
                       f"(Total: {total_text_size/1024:.1f} KB, Avg: {avg_text_size/1024:.1f} KB per chunk)")
            
            # Process in batches to avoid OOM
            model_start = time.time()
            embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch_start = time.time()
                batch_texts = texts[i:i+self.batch_size]
                batch_size = sum(len(text) for text in batch_texts)
                
                logger.info(f"[Worker {worker_id}] Processing embedding batch {i//self.batch_size + 1}/"
                           f"{(len(texts) + self.batch_size - 1)//self.batch_size} "
                           f"({len(batch_texts)} chunks, {batch_size/1024:.1f} KB)")
                
                batch_embeddings = await self.embedding_adapter.embed(batch_texts)
                embeddings.extend(batch_embeddings)
                
                batch_time = time.time() - batch_start
                embed_details["batches"].append({
                    "batch_num": i//self.batch_size + 1,
                    "num_chunks": len(batch_texts),
                    "text_size": batch_size,
                    "time": batch_time,
                    "chunks_per_second": len(batch_texts) / batch_time if batch_time > 0 else 0,
                    "kb_per_second": batch_size / 1024 / batch_time if batch_time > 0 else 0
                })
            
            embed_details["model_inference"] = time.time() - model_start
            
            # Post-processing
            post_start = time.time()
            if len(embeddings) != len(result["chunks"]):
                logger.warning(f"[Worker {worker_id}] Mismatch between chunks ({len(result['chunks'])}) "
                               f"and embeddings ({len(embeddings)}) for {file_path.name}")
                # Adjust the shorter list to match the longer one
                if len(embeddings) < len(result["chunks"]):
                    result["chunks"] = result["chunks"][:len(embeddings)]
                    logger.warning(f"[Worker {worker_id}] Truncated chunks to match embeddings count: {len(embeddings)}")
            
            # Get embedding statistics
            embed_dims = len(embeddings[0]) if embeddings else 0
            embed_details["post_processing"] = time.time() - post_start
            embed_details["total"] = time.time() - embed_details["start"]
            
            # Calculate average batch performance
            if embed_details["batches"]:
                avg_chunks_per_second = sum(b["chunks_per_second"] for b in embed_details["batches"]) / len(embed_details["batches"])
                avg_kb_per_second = sum(b["kb_per_second"] for b in embed_details["batches"]) / len(embed_details["batches"])
            else:
                avg_chunks_per_second = 0
                avg_kb_per_second = 0
                
            # Store embedding statistics
            result["embed_stats"] = {
                "count": len(embeddings),
                "dimensions": embed_dims,
                "avg_chunks_per_second": avg_chunks_per_second,
                "avg_kb_per_second": avg_kb_per_second
            }
            result["embed_details"] = embed_details
            
            logger.info(f"[Worker {worker_id}] Generated {len(embeddings)} embeddings ({embed_dims} dimensions each) "
                      f"for {file_path.name} at {avg_chunks_per_second:.2f} chunks/sec, {avg_kb_per_second:.2f} KB/sec")
            
            # Convert embeddings to lists for JSON serialization
            embeddings_list = []
            for emb in embeddings:
                if isinstance(emb, np.ndarray):
                    emb_list = emb.tolist()
                else:
                    emb_list = emb
                embeddings_list.append(emb_list)
            
            result["embeddings"] = embeddings_list
        except Exception as e:
            logger.error(f"Error generating embeddings for {file_path.name}: {str(e)}")
            return result
        
        embed_time = time.time() - embed_start
        result["timing"]["embedding"] = embed_time
        
        # Total time
        result["timing"]["total"] = time.time() - total_start
        
        # Add timestamp to show when this task completed
        result["completed_at"] = datetime.now().isoformat()
        
        logger.info(f"[Worker {result['worker_id']}] Completed {file_path.name}: {len(result['chunks'])} chunks, {len(embeddings)} embeddings in {result['timing']['total']:.2f}s")
        return result
    
    async def process_documents_parallel(self, files: List[Path]) -> List[Dict[str, Any]]:
        """Process multiple documents in parallel."""
        logger.info(f"Processing {len(files)} documents in parallel with {self.max_workers} workers")
        
        # Create processing tasks
        tasks = [self.process_document(file) for file in files]
        
        # Run tasks with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_workers)
        
        # Create a helper to track when tasks start and end
        current_running = set()
        
        async def process_with_semaphore(i, task):
            task_id = f"Task-{i}"
            async with semaphore:
                # Log when task starts with concurrent count
                current_running.add(task_id)
                logger.info(f"Starting {task_id} - Now running {len(current_running)} concurrent tasks: {', '.join(current_running)}")
                
                try:
                    result = await task
                    return result
                finally:
                    # Log when task completes
                    current_running.remove(task_id)
                    logger.info(f"Completed {task_id} - {len(current_running)} tasks still running")
        
        results = await asyncio.gather(
            *[process_with_semaphore(i, task) for i, task in enumerate(tasks)]
        )
        
        return results
    
    def update_statistics(self, results: List[Dict[str, Any]]) -> None:
        """Update statistics based on processing results."""
        self.stats["total_files"] = len(results)
        
        for result in results:
            file_id = result["file_id"]
            self.stats["file_details"][file_id] = {
                "file_name": result["file_name"],
                "file_size": result["file_size"],
                "num_chunks": len(result.get("chunks", [])),
                "num_embeddings": len(result.get("embeddings", [])),
                "timing": result["timing"]
            }
            
            self.stats["total_chunks"] += len(result.get("chunks", []))
            self.stats["total_embeddings"] += len(result.get("embeddings", []))
            
            # Update timing statistics
            for key, value in result["timing"].items():
                self.stats["processing_time"][key] += value
    
    def save_results(self, results: List[Dict[str, Any]]) -> None:
        """Save processing results and statistics."""
        # Save statistics
        stats_path = self.output_dir / "pipeline_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Saved statistics to {stats_path}")
        
        # Save individual file results
        for result in results:
            file_id = result["file_id"]
            result_path = self.output_dir / f"{file_id}_processed.json"
            
            # Create a version without embeddings to save space
            result_summary = {
                "file_id": file_id,
                "file_name": result["file_name"],
                "file_size": result["file_size"],
                "timing": result["timing"],
                "chunks": result["chunks"],
                "num_embeddings": len(result.get("embeddings", [])),
                "embedding_dimension": len(result["embeddings"][0]) if result.get("embeddings") else 0
            }
            
            with open(result_path, 'w') as f:
                json.dump(result_summary, f, indent=2)
        
        # Save a combined data structure for ISNE input
        isne_input = []
        for result in results:
            file_id = result["file_id"]
            chunks = result.get("chunks", [])
            embeddings = result.get("embeddings", [])
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                isne_input.append({
                    "id": f"{file_id}_chunk_{i}",
                    "content": chunk["content"],
                    "embedding": embedding,
                    "metadata": {
                        "document_id": file_id,
                        "file_name": result["file_name"],
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                })
        
        # Save ISNE input structure (sample only to avoid huge file)
        isne_sample_path = self.output_dir / "isne_input_sample.json"
        isne_sample = isne_input[:100] if len(isne_input) > 100 else isne_input
        with open(isne_sample_path, 'w') as f:
            json.dump(isne_sample, f, indent=2)
        
        logger.info(f"Saved sample ISNE input to {isne_sample_path}")
        
    async def run(self) -> Dict[str, Any]:
        """Run the parallel pipeline test."""
        logger.info("=== Starting Parallel Pipeline Test ===")
        logger.info(f"Max Workers: {self.max_workers}, Batch Size: {self.batch_size}, Files to Process: {self.num_files}")
        
        # Record detailed timings for each stage
        timings = {
            "setup": 0,
            "file_discovery": 0,
            "parallel_processing": 0,
            "statistics_update": 0,
            "results_saving": 0,
            "total": 0
        }
        
        overall_start_time = time.time()
        
        # Find PDF files to process
        file_discovery_start = time.time()
        files = self.find_pdf_files()
        timings["file_discovery"] = time.time() - file_discovery_start
        
        if not files:
            logger.error("No PDF files found for testing")
            return self.stats
        
        logger.info(f"Found {len(files)} PDF files in {timings['file_discovery']:.2f}s")
        
        # Process documents in parallel
        processing_start = time.time()
        logger.info(f"Starting parallel processing of {len(files)} files...")
        results = await self.process_documents_parallel(files)
        timings["parallel_processing"] = time.time() - processing_start
        
        # Update statistics
        stats_start = time.time()
        self.update_statistics(results)
        timings["statistics_update"] = time.time() - stats_start
        
        # Save results
        save_start = time.time()
        self.save_results(results)
        timings["results_saving"] = time.time() - save_start
        
        # Calculate overall timing
        timings["total"] = time.time() - overall_start_time
        self.stats["processing_time"]["total"] = timings["total"]
        self.stats["timings"] = timings
        self.stats["component_init_time"] = self.component_init_time
        
        # Generate and log detailed performance report
        self.log_performance_report(timings, results)
        
        logger.info(f"=== Completed Parallel Pipeline Test in {timings['total']:.2f} seconds ===")
        logger.info(f"Processed {self.stats['total_files']} files with {self.stats['total_chunks']} chunks")
        
        return self.stats
        
    def log_performance_report(self, timings: Dict[str, float], results: List[Dict[str, Any]]) -> None:
        """Generate and log a detailed performance report."""
        logger.info("\n=== Performance Report ===")
        
        # Overall timing breakdown
        logger.info("\nTiming Breakdown:")
        logger.info(f"  Setup:                {self.component_init_time['total']:.2f}s")
        logger.info(f"  File Discovery:       {timings['file_discovery']:.2f}s")
        logger.info(f"  Parallel Processing:  {timings['parallel_processing']:.2f}s")
        logger.info(f"  Statistics Update:    {timings['statistics_update']:.2f}s")
        logger.info(f"  Results Saving:       {timings['results_saving']:.2f}s")
        logger.info(f"  Total Runtime:        {timings['total']:.2f}s")
        
        # Processing stage averages
        total_doc_time = sum(result['timing']['document_processing'] for result in results)
        total_chunk_time = sum(result['timing']['chunking'] for result in results)
        total_embed_time = sum(result['timing']['embedding'] for result in results)
        
        logger.info("\nAverage Processing Times:")
        if results:
            logger.info(f"  Document Processing:  {total_doc_time / len(results):.2f}s per file")
            logger.info(f"  Chunking:             {total_chunk_time / len(results):.2f}s per file")
            logger.info(f"  Embedding:            {total_embed_time / len(results):.2f}s per file")
        
        # Throughput calculations
        logger.info("\nThroughput:")
        files_per_second = len(results) / timings['total'] if timings['total'] > 0 else 0
        chunks_per_second = self.stats['total_chunks'] / timings['total'] if timings['total'] > 0 else 0
        logger.info(f"  Files per second:     {files_per_second:.2f}")
        logger.info(f"  Chunks per second:    {chunks_per_second:.2f}")
        
        # Worker efficiency
        if self.max_workers > 1:
            theoretical_speedup = min(len(results), self.max_workers)
            sequential_time = total_doc_time + total_chunk_time + total_embed_time
            parallel_time = timings['parallel_processing']
            actual_speedup = sequential_time / parallel_time if parallel_time > 0 else 0
            efficiency = actual_speedup / theoretical_speedup if theoretical_speedup > 0 else 0
            
            logger.info("\nParallelization Efficiency:")
            logger.info(f"  Theoretical Speedup:  {theoretical_speedup:.2f}x")
            logger.info(f"  Actual Speedup:       {actual_speedup:.2f}x")
            logger.info(f"  Efficiency:           {efficiency:.2f} ({efficiency*100:.1f}%)")
        
        logger.info("\n=== End Performance Report ===")

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run parallel pipeline test")
    parser.add_argument("--test-data-dir", type=str, default="test-data",
                        help="Directory containing test PDF files")
    parser.add_argument("--output-dir", type=str, default="test-output/pipeline-test",
                        help="Directory to save output data")
    parser.add_argument("--num-files", type=int, default=20,
                        help="Number of files to process")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Maximum number of parallel workers")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for processing")
    
    args = parser.parse_args()
    
    tester = PipelineParallelTester(
        test_data_dir=args.test_data_dir,
        output_dir=args.output_dir,
        num_files=args.num_files,
        max_workers=args.max_workers,
        batch_size=args.batch_size
    )
    
    await tester.run()

if __name__ == "__main__":
    asyncio.run(main())
