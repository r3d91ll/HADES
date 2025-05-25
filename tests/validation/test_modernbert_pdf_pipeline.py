#!/usr/bin/env python
"""
Validation test for ModernBERT adapter in the PDF pipeline.

This script tests the integration of the ModernBERT embedding adapter
with the PDF pipeline, without requiring the full ingestion system.
"""

import os
import sys
import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipelines.ingest.orchestrator.pdf_pipeline_prototype import PDFPipeline
from src.embedding.adapters.modernbert_adapter import ModernBERTEmbeddingAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_modernbert_pipeline() -> Optional[List[Dict[str, Any]]]:
    """Test the ModernBERT adapter with the PDF pipeline.
    
    Returns:
        List of result dictionaries for each processed PDF, or None if test fails
    """
    # Create output directory for test results
    output_dir = Path("test-output/modernbert-test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select test PDF files
    test_pdfs = [
        Path("test-data/ISNE_paper.pdf"),
        Path("docs/PathRAG_paper.pdf"),
    ]
    
    # Verify files exist
    existing_pdfs = [pdf for pdf in test_pdfs if pdf.exists()]
    if not existing_pdfs:
        logger.error("No test PDFs found. Please check paths.")
        return None
    
    logger.info(f"Found {len(existing_pdfs)} test PDFs for processing")
    
    # Configure the pipeline with ModernBERT adapter
    pipeline = PDFPipeline(
        output_dir=output_dir,
        embedding_options={
            "adapter_name": "modernbert",
            "model_name": "answerdotai/ModernBERT-base",
            "max_length": 8192,  # ModernBERT can handle 8K tokens
            "pooling_strategy": "cls",
            "batch_size": 8,  # Smaller batch size due to larger context
            "normalize_embeddings": True
        },
        save_intermediate_results=True
    )
    
    # Process each PDF file
    results: List[Dict[str, Any]] = []
    for pdf_path in existing_pdfs:
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Run the complete PDF pipeline with ModernBERT embeddings
            result = await pipeline.process_pdf(pdf_path)
            
            if result:
                # Extract key metrics
                num_chunks = len(result.get("chunks", []))
                
                # Check if embeddings were generated
                chunks_with_embeddings = sum(1 for chunk in result.get("chunks", []) 
                                            if "embedding" in chunk and chunk["embedding"])
                
                # Get embedding dimension if available
                embedding_dim = None
                for chunk in result.get("chunks", []):
                    if "embedding" in chunk and chunk["embedding"]:
                        embedding_dim = len(chunk["embedding"])
                        break
                
                # Get processing metadata
                metadata = result.get("processing_metadata", {})
                embedding_metadata = metadata.get("embedding", {})
                
                logger.info(f"Document: {pdf_path.name}")
                logger.info(f"Chunks: {num_chunks}")
                logger.info(f"Chunks with embeddings: {chunks_with_embeddings}")
                logger.info(f"Embedding dimensions: {embedding_dim}")
                
                # Extract performance metrics
                if embedding_metadata:
                    adapter = embedding_metadata.get("adapter", "unknown")
                    model = embedding_metadata.get("model", "unknown")
                    pooling = embedding_metadata.get("pooling_strategy", "unknown")
                    processing_time = embedding_metadata.get("processing_time_sec", 0)
                    
                    logger.info(f"Embedding adapter: {adapter}")
                    logger.info(f"Embedding model: {model}")
                    logger.info(f"Pooling strategy: {pooling}")
                    logger.info(f"Processing time: {processing_time:.2f}s")
                    
                    # Validate ModernBERT was used
                    assert "modernbert" in adapter.lower(), "ModernBERT adapter not used"
                    assert embedding_dim is not None, "No embeddings were generated"
                    assert embedding_dim == 768, f"Expected 768 dimensions, got {embedding_dim}"
                
                # Store results
                results.append({
                    "filename": pdf_path.name,
                    "chunks": num_chunks,
                    "chunks_with_embeddings": chunks_with_embeddings,
                    "embedding_dimensions": embedding_dim,
                    "embedding_model": embedding_metadata.get("model", "unknown"),
                    "pooling_strategy": embedding_metadata.get("pooling_strategy", "unknown"),
                    "processing_time_sec": embedding_metadata.get("processing_time_sec", 0)
                })
            else:
                logger.error(f"Processing failed for {pdf_path}")
        
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
    
    # Save summary report
    if results:
        report_path = output_dir / "modernbert_validation_report.json"
        with open(report_path, "w") as f:
            json.dump({
                "test_date": import_time.strftime("%Y-%m-%d %H:%M:%S"),
                "pdfs_processed": len(results),
                "results": results
            }, f, indent=2)
        
        logger.info(f"Validation report saved to {report_path}")
    
    return results


async def main() -> None:
    """Run the ModernBERT pipeline validation test."""
    try:
        results = await test_modernbert_pipeline()
        
        if results:
            logger.info("ModernBERT PDF pipeline validation completed successfully")
            
            # Print summary statistics
            total_chunks = sum(r["chunks"] for r in results)
            total_embedded = sum(r["chunks_with_embeddings"] for r in results)
            avg_time = sum(r["processing_time_sec"] for r in results) / len(results)
            
            print("\nSummary Statistics:")
            print(f"Total PDFs processed: {len(results)}")
            print(f"Total chunks generated: {total_chunks}")
            print(f"Total chunks with embeddings: {total_embedded}")
            print(f"Average processing time: {avg_time:.2f}s")
            print(f"Embedding dimensions: {results[0]['embedding_dimensions']}")
            print(f"Embedding model: {results[0]['embedding_model']}")
            print(f"Pooling strategy: {results[0]['pooling_strategy']}")
            
            # Print performance metrics
            chunks_per_second = total_chunks / avg_time if avg_time > 0 else 0
            print(f"\nPerformance Metrics:")
            print(f"Chunks processed per second: {chunks_per_second:.2f}")
        else:
            logger.error("ModernBERT PDF pipeline validation failed")
    
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    import time as import_time
    asyncio.run(main())
