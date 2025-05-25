"""
Performance benchmarks for the ISNE pipeline with integrated alert system.

This module measures the performance impact of the alert system
on the ISNE pipeline to ensure it meets performance requirements.
"""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import pytest

from src.isne.pipeline.isne_pipeline import ISNEPipeline
from src.alerts import AlertManager, AlertLevel
from src.validation.embedding_validator import validate_embeddings_before_isne


def generate_test_documents(num_docs: int = 10, chunks_per_doc: int = 5) -> List[Dict[str, Any]]:
    """
    Generate test documents with embeddings for benchmarking.
    
    Args:
        num_docs: Number of documents to generate
        chunks_per_doc: Number of chunks per document
        
    Returns:
        List of documents with embeddings
    """
    docs = []
    
    for i in range(num_docs):
        chunks = []
        
        for j in range(chunks_per_doc):
            # Create embedding (384-dimensional vector)
            embedding = np.random.rand(384).tolist()
            
            chunks.append({
                "id": f"chunk_{i}_{j}",
                "text": f"This is test chunk {j} of document {i}",
                "embedding": embedding,
                "embedding_model": "test-embedding-model",
                "embedding_type": "base",
                "metadata": {"position": j}
            })
        
        docs.append({
            "id": f"doc_{i}",
            "file_id": f"file_{i}",
            "file_name": f"test_document_{i}.txt",
            "chunks": chunks
        })
    
    return docs


def introduce_embedding_issues(docs: List[Dict[str, Any]], 
                               missing_rate: float = 0.05) -> List[Dict[str, Any]]:
    """
    Introduce embedding issues into documents for testing alerts.
    
    Args:
        docs: List of documents with embeddings
        missing_rate: Percentage of embeddings to remove
        
    Returns:
        Modified documents with embedding issues
    """
    import copy
    modified_docs = copy.deepcopy(docs)
    
    total_chunks = sum(len(doc["chunks"]) for doc in modified_docs)
    num_to_remove = int(total_chunks * missing_rate)
    
    # Randomly select chunks to remove embeddings from
    import random
    removed = 0
    
    while removed < num_to_remove:
        doc_idx = random.randint(0, len(modified_docs) - 1)
        doc = modified_docs[doc_idx]
        
        if not doc["chunks"]:
            continue
        
        chunk_idx = random.randint(0, len(doc["chunks"]) - 1)
        chunk = doc["chunks"][chunk_idx]
        
        if "embedding" in chunk:
            del chunk["embedding"]
            del chunk["embedding_model"]
            del chunk["embedding_type"]
            removed += 1
    
    return modified_docs


def benchmark_pipeline_with_alerts(num_docs: int, 
                                  chunks_per_doc: int,
                                  with_alerts: bool = True) -> Dict[str, Any]:
    """
    Benchmark the ISNE pipeline with or without alerts.
    
    Args:
        num_docs: Number of documents to process
        chunks_per_doc: Number of chunks per document
        with_alerts: Whether to enable alerts
        
    Returns:
        Dictionary with benchmark results
    """
    # Set up temporary directory
    temp_dir = tempfile.TemporaryDirectory()
    output_dir = Path(temp_dir.name)
    
    try:
        # Generate test documents
        docs = generate_test_documents(num_docs, chunks_per_doc)
        
        # Introduce some issues to trigger alerts
        if with_alerts:
            docs = introduce_embedding_issues(docs)
        
        # Create pipeline with or without alert validation
        pipeline = ISNEPipeline(
            validate=with_alerts,
            alert_threshold="low" if with_alerts else "high",  # Ensure alerts trigger if enabled
            alert_dir=str(output_dir / "alerts")
        )
        
        # Time pipeline processing
        start_time = time.time()
        
        # Process documents
        enhanced_docs, stats = pipeline.process_documents(
            documents=docs,
            save_report=True,
            output_dir=str(output_dir)
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Collect alert stats if enabled
        alert_stats = {}
        if with_alerts:
            alert_stats = pipeline.alert_manager.get_alert_stats()
        
        # Calculate metrics
        total_chunks = sum(len(doc["chunks"]) for doc in docs)
        chunks_per_second = total_chunks / processing_time
        
        # Result dictionary
        result = {
            "num_docs": num_docs,
            "total_chunks": total_chunks,
            "processing_time_seconds": processing_time,
            "chunks_per_second": chunks_per_second,
            "alert_stats": alert_stats,
            "validation_stats": stats.get("validation_summary", {})
        }
        
        return result
        
    finally:
        # Clean up
        temp_dir.cleanup()


def run_benchmark_suite():
    """Run a suite of benchmarks with various configurations."""
    print("\n===== ISNE Pipeline with Alerts Benchmark =====\n")
    
    # Define document sizes
    small_size = {"num_docs": 10, "chunks_per_doc": 5}     # 50 chunks
    medium_size = {"num_docs": 50, "chunks_per_doc": 10}   # 500 chunks
    large_size = {"num_docs": 100, "chunks_per_doc": 20}   # 2000 chunks
    
    # Run benchmarks with and without alerts
    results = []
    
    for size_name, size_config in [
        ("small", small_size),
        ("medium", medium_size),
        ("large", large_size)
    ]:
        print(f"\nRunning {size_name} benchmark...")
        
        # Without alerts
        print(f"  - Without alerts...")
        no_alerts_result = benchmark_pipeline_with_alerts(
            **size_config, with_alerts=False
        )
        no_alerts_result["size"] = size_name
        no_alerts_result["alerts_enabled"] = False
        results.append(no_alerts_result)
        
        # With alerts
        print(f"  - With alerts...")
        with_alerts_result = benchmark_pipeline_with_alerts(
            **size_config, with_alerts=True
        )
        with_alerts_result["size"] = size_name
        with_alerts_result["alerts_enabled"] = True
        results.append(with_alerts_result)
    
    # Print results
    print("\n===== Benchmark Results =====\n")
    print(f"{'Size':<10} {'Alerts':<10} {'Chunks':<10} {'Time (s)':<10} {'Chunks/s':<10}")
    print("-" * 60)
    
    for result in results:
        alerts = "Enabled" if result["alerts_enabled"] else "Disabled"
        print(f"{result['size']:<10} {alerts:<10} {result['total_chunks']:<10} "
              f"{result['processing_time_seconds']:.2f}s".ljust(10) + 
              f"{result['chunks_per_second']:.2f}".ljust(10))
    
    # Calculate overhead of alerts
    print("\n===== Alert System Overhead =====\n")
    
    for size_name in ["small", "medium", "large"]:
        no_alerts = next(r for r in results if r["size"] == size_name and not r["alerts_enabled"])
        with_alerts = next(r for r in results if r["size"] == size_name and r["alerts_enabled"])
        
        time_no_alerts = no_alerts["processing_time_seconds"]
        time_with_alerts = with_alerts["processing_time_seconds"]
        
        overhead_pct = ((time_with_alerts - time_no_alerts) / time_no_alerts) * 100
        
        print(f"{size_name.capitalize()} dataset: {overhead_pct:.2f}% overhead")
    
    print("\n===== Analysis =====\n")
    
    # Calculate average overhead
    avg_overhead = 0
    count = 0
    
    for size_name in ["small", "medium", "large"]:
        no_alerts = next(r for r in results if r["size"] == size_name and not r["alerts_enabled"])
        with_alerts = next(r for r in results if r["size"] == size_name and r["alerts_enabled"])
        
        time_no_alerts = no_alerts["processing_time_seconds"]
        time_with_alerts = with_alerts["processing_time_seconds"]
        
        overhead_pct = ((time_with_alerts - time_no_alerts) / time_no_alerts) * 100
        avg_overhead += overhead_pct
        count += 1
    
    avg_overhead /= count
    
    if avg_overhead < 5:
        print("✅ Alert system adds minimal overhead (<5%)")
    elif avg_overhead < 10:
        print("✅ Alert system adds acceptable overhead (<10%)")
    else:
        print(f"⚠️ Alert system adds significant overhead ({avg_overhead:.2f}%)")
    
    # Check throughput
    large_with_alerts = next(r for r in results if r["size"] == "large" and r["alerts_enabled"])
    throughput = large_with_alerts["chunks_per_second"]
    
    if throughput > 1000:
        print(f"✅ High throughput achieved: {throughput:.2f} chunks/second")
    elif throughput > 100:
        print(f"✅ Good throughput achieved: {throughput:.2f} chunks/second")
    else:
        print(f"⚠️ Low throughput: {throughput:.2f} chunks/second")


if __name__ == "__main__":
    run_benchmark_suite()
