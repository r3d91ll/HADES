#!/usr/bin/env python
"""
Validation test for the full pipeline with ModernBERT adapter using configuration.

This script runs a complete pipeline test with the configuration-based
ModernBERT adapter, measuring performance on both CPU and GPU configurations.
It provides detailed metrics for comparison with previous implementations.
"""

import os
import sys
import asyncio
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipelines.ingest.orchestrator.pdf_pipeline_prototype import PDFPipeline
from src.config.embedding_config import load_config, get_adapter_config

# Explicitly import the ModernBERT adapter to ensure it's registered
from src.embedding.adapters.modernbert_adapter import ModernBERTEmbeddingAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_pipeline_test(
    pdf_path: Union[str, Path],
    output_dir: Path,
    adapter_name: str = "modernbert",
    device: Optional[str] = None
) -> Tuple[Dict[str, Any], float]:
    """
    Run the pipeline with the specified adapter configuration.
    
    Args:
        pdf_path: Path to the test PDF
        output_dir: Directory to save results
        adapter_name: Name of adapter configuration to use
        device: Optional device override
    
    Returns:
        Tuple containing results dictionary and total processing time
    """
    # Get configuration for the adapter
    adapter_config = get_adapter_config(adapter_name)
    
    # Override device if specified
    if device:
        adapter_config["device"] = device
    
    # Configure embedding options with adapter config
    embedding_options = {
        "adapter_name": adapter_name,
        **adapter_config
    }
    
    # Log configuration
    logger.info(f"Testing adapter: {adapter_name} on {adapter_config.get('device', 'default')}")
    logger.info(f"Model: {adapter_config.get('model_name', 'unknown')}")
    logger.info(f"Pooling: {adapter_config.get('pooling_strategy', 'unknown')}")
    
    # Initialize pipeline
    pipeline = PDFPipeline(
        output_dir=output_dir,
        embedding_options=embedding_options,
        save_intermediate_results=True
    )
    
    # Time the full pipeline
    start_time = time.time()
    
    # Process the PDF
    result = await pipeline.process_pdf(pdf_path)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Extract metrics
    if result:
        # Get total chunks and tokens
        chunks = result.get("chunks", [])
        num_chunks = len(chunks)
        chunk_tokens = sum(len(chunk.get("content", "").split()) for chunk in chunks)
        
        # Count chunks with embeddings
        chunks_with_embeddings = sum(1 for c in chunks if "embedding" in c and c["embedding"])
        
        # Get embedding dimension if available
        embedding_dim = None
        for chunk in chunks:
            if "embedding" in chunk and chunk["embedding"]:
                embedding_dim = len(chunk["embedding"])
                break
        
        # Get processing metadata
        metadata = result.get("processing_metadata", {})
        embedding_metadata = metadata.get("embedding", {})
        
        # Log results
        logger.info(f"Document: {Path(pdf_path).name}")
        logger.info(f"Chunks: {num_chunks}")
        logger.info(f"Total tokens: {chunk_tokens}")
        logger.info(f"Chunks with embeddings: {chunks_with_embeddings}")
        logger.info(f"Embedding dimensions: {embedding_dim}")
        logger.info(f"Total processing time: {total_time:.2f}s")
        
        if embedding_metadata:
            logger.info(f"Embedding adapter: {embedding_metadata.get('adapter', 'unknown')}")
            logger.info(f"Embedding time: {embedding_metadata.get('processing_time_sec', 0):.2f}s")
        
        # Compile metrics for return
        metrics = {
            "filename": Path(pdf_path).name,
            "chunks": num_chunks,
            "total_tokens": chunk_tokens,
            "avg_tokens_per_chunk": chunk_tokens / num_chunks if num_chunks else 0,
            "chunks_with_embeddings": chunks_with_embeddings,
            "embedding_dimensions": embedding_dim,
            "embedding_model": embedding_metadata.get("model", "unknown"),
            "pooling_strategy": embedding_metadata.get("pooling_strategy", "unknown"),
            "embedding_time_sec": embedding_metadata.get("processing_time_sec", 0),
            "total_time_sec": total_time,
            "time_per_chunk_sec": total_time / num_chunks if num_chunks else 0,
            "tokens_per_second": chunk_tokens / total_time if total_time > 0 else 0
        }
        
        return metrics, total_time
    else:
        logger.error(f"Processing failed for {pdf_path}")
        return {}, total_time


async def main():
    """Run the full pipeline validation tests."""
    # Create output directory
    output_dir = Path("test-output/modernbert-cpu-test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test PDFs 
    test_pdfs = [
        Path("test-data/ISNE_paper.pdf"),
        Path("docs/PathRAG_paper.pdf"),
    ]
    
    # Verify files exist
    existing_pdfs = [pdf for pdf in test_pdfs if pdf.exists()]
    if not existing_pdfs:
        logger.error("No test PDFs found. Please check paths.")
        return
    
    logger.info(f"Found {len(existing_pdfs)} test PDFs for processing")
    
    # Test results for all configurations
    results = []
    
    # Test the full pipeline with ModernBERT on CPU first
    logger.info("\n" + "="*60)
    logger.info("TESTING MODERNBERT CPU CONFIGURATION")
    logger.info("="*60)
    
    for pdf_path in existing_pdfs:
        try:
            # Run with CPU config
            metrics, _ = await run_pipeline_test(
                pdf_path=pdf_path,
                output_dir=output_dir,
                adapter_name="modernbert"  # use CPU config
            )
            if metrics:
                metrics["config"] = "modernbert-cpu"
                results.append(metrics)
        except Exception as e:
            logger.error(f"Error testing CPU config on {pdf_path}: {e}")
    
    # Save aggregated results
    if results:
        report_path = output_dir / "pipeline_validation_report.json"
        with open(report_path, "w") as f:
            json.dump({
                "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "configurations_tested": ["modernbert-cpu"],
                "pdfs_processed": len(results),
                "results": results
            }, f, indent=2)
        
        logger.info(f"Validation report saved to {report_path}")
        
        # Print summary statistics
        logger.info("\n" + "="*60)
        logger.info("SUMMARY STATISTICS")
        logger.info("="*60)
        
        cpu_results = [r for r in results if r["config"] == "modernbert-cpu"]
        
        if cpu_results:
            total_chunks = sum(r["chunks"] for r in cpu_results)
            total_tokens = sum(r["total_tokens"] for r in cpu_results)
            total_embedded = sum(r["chunks_with_embeddings"] for r in cpu_results)
            avg_time = sum(r["total_time_sec"] for r in cpu_results) / len(cpu_results)
            avg_embed_time = sum(r["embedding_time_sec"] for r in cpu_results) / len(cpu_results)
            
            print("\nModernBERT CPU Configuration:")
            print(f"Total PDFs processed: {len(cpu_results)}")
            print(f"Total chunks generated: {total_chunks}")
            print(f"Total tokens processed: {total_tokens}")
            print(f"Total chunks with embeddings: {total_embedded}")
            print(f"Average processing time: {avg_time:.2f}s")
            print(f"Average embedding time: {avg_embed_time:.2f}s")
            print(f"Chunk processing rate: {total_chunks/avg_time:.1f} chunks/sec")
            print(f"Token processing rate: {total_tokens/avg_time:.1f} tokens/sec")
            
            if cpu_results[0]["embedding_dimensions"]:
                print(f"Embedding dimensions: {cpu_results[0]['embedding_dimensions']}")
                print(f"Embedding model: {cpu_results[0]['embedding_model']}")
                print(f"Pooling strategy: {cpu_results[0]['pooling_strategy']}")
        else:
            print("No results available for ModernBERT CPU configuration")
    else:
        logger.error("No successful test results collected")


if __name__ == "__main__":
    asyncio.run(main())
