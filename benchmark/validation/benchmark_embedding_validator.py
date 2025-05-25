#!/usr/bin/env python
"""
Performance benchmarks for the embedding validator module.

This benchmark measures the performance of the validation functions on different
dataset sizes to ensure they meet performance standards.
"""

import json
import time
import random
import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import validation module
from src.validation.embedding_validator import (
    validate_embeddings_before_isne,
    validate_embeddings_after_isne,
    create_validation_summary
)

# Configure minimal logging to avoid skewing benchmarks
logging.basicConfig(level=logging.WARNING)

def generate_test_documents(num_docs: int, chunks_per_doc: int, embedding_dim: int = 768) -> List[Dict[str, Any]]:
    """
    Generate synthetic test documents with embeddings for benchmarking.
    
    Args:
        num_docs: Number of documents to generate
        chunks_per_doc: Number of chunks per document
        embedding_dim: Dimension of embeddings
        
    Returns:
        List of synthetic documents
    """
    documents = []
    
    for doc_idx in range(num_docs):
        doc = {
            "file_id": f"doc_{doc_idx}",
            "file_name": f"document_{doc_idx}.py",
            "chunks": []
        }
        
        for chunk_idx in range(chunks_per_doc):
            # Create base embedding (some might be missing)
            has_base_embedding = random.random() > 0.05  # 5% chance of missing base embedding
            base_embedding = np.random.rand(embedding_dim).tolist() if has_base_embedding else None
            
            # Create ISNE embedding (some might be missing or duplicate)
            has_isne_embedding = random.random() > 0.1  # 10% chance of missing ISNE embedding
            has_duplicate_isne = random.random() > 0.95  # 5% chance of duplicate ISNE embedding
            
            chunk = {
                "text": f"Chunk {chunk_idx} of document {doc_idx}",
                "metadata": {"position": chunk_idx}
            }
            
            if has_base_embedding:
                chunk["embedding"] = base_embedding
                
            if has_isne_embedding:
                chunk["isne_embedding"] = np.random.rand(embedding_dim).tolist()
                
                if has_duplicate_isne:
                    duplicate_key = random.choice(["isne_embedding_duplicate", "isne_embedding_alt"])
                    chunk[duplicate_key] = np.random.rand(embedding_dim).tolist()
            
            doc["chunks"].append(chunk)
        
        # Occasionally add document-level ISNE (which is an error)
        if random.random() > 0.9:  # 10% chance
            doc["isne_embedding"] = np.random.rand(embedding_dim).tolist()
            
        documents.append(doc)
    
    return documents

def benchmark_pre_validation(documents: List[Dict[str, Any]], runs: int = 5) -> Dict[str, float]:
    """
    Benchmark pre-ISNE validation performance.
    
    Args:
        documents: List of documents to validate
        runs: Number of benchmark runs
        
    Returns:
        Dictionary with benchmark results
    """
    times = []
    
    # Warm-up run
    validate_embeddings_before_isne(documents)
    
    # Timed runs
    for i in range(runs):
        start_time = time.time()
        pre_validation = validate_embeddings_before_isne(documents)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        "avg_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "total_docs": len(documents),
        "total_chunks": sum(len(doc.get("chunks", [])) for doc in documents)
    }

def benchmark_post_validation(documents: List[Dict[str, Any]], pre_validation: Dict[str, Any], runs: int = 5) -> Dict[str, float]:
    """
    Benchmark post-ISNE validation performance.
    
    Args:
        documents: List of documents to validate
        pre_validation: Pre-validation results
        runs: Number of benchmark runs
        
    Returns:
        Dictionary with benchmark results
    """
    times = []
    
    # Warm-up run
    validate_embeddings_after_isne(documents, pre_validation)
    
    # Timed runs
    for i in range(runs):
        start_time = time.time()
        post_validation = validate_embeddings_after_isne(documents, pre_validation)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        "avg_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times)
    }

def benchmark_create_summary(pre_validation: Dict[str, Any], post_validation: Dict[str, Any], runs: int = 5) -> Dict[str, float]:
    """
    Benchmark summary creation performance.
    
    Args:
        pre_validation: Pre-validation results
        post_validation: Post-validation results
        runs: Number of benchmark runs
        
    Returns:
        Dictionary with benchmark results
    """
    times = []
    
    # Warm-up run
    create_validation_summary(pre_validation, post_validation)
    
    # Timed runs
    for i in range(runs):
        start_time = time.time()
        summary = create_validation_summary(pre_validation, post_validation)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        "avg_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times)
    }

def run_full_benchmark(sizes: List[Dict[str, int]], runs: int = 5, output_file: str = None) -> Dict[str, Any]:
    """
    Run comprehensive benchmarks across different dataset sizes.
    
    Args:
        sizes: List of dictionaries with 'docs' and 'chunks' keys
        runs: Number of runs per benchmark
        output_file: Optional path to save benchmark results
        
    Returns:
        Dictionary with benchmark results
    """
    results = {
        "timestamp": time.time(),
        "runs_per_benchmark": runs,
        "sizes": [],
    }
    
    for size_config in sizes:
        num_docs = size_config["docs"]
        chunks_per_doc = size_config["chunks"]
        total_chunks = num_docs * chunks_per_doc
        
        print(f"Benchmarking with {num_docs} documents, {chunks_per_doc} chunks each ({total_chunks} total chunks)")
        
        # Generate test data
        documents = generate_test_documents(num_docs, chunks_per_doc)
        
        # Benchmark pre-validation
        print("  Running pre-validation benchmark...")
        pre_results = benchmark_pre_validation(documents, runs)
        
        # Run pre-validation once to get results for post-validation
        pre_validation = validate_embeddings_before_isne(documents)
        
        # Benchmark post-validation
        print("  Running post-validation benchmark...")
        post_results = benchmark_post_validation(documents, pre_validation, runs)
        
        # Benchmark summary creation
        print("  Running summary creation benchmark...")
        post_validation = validate_embeddings_after_isne(documents, pre_validation)
        summary_results = benchmark_create_summary(pre_validation, post_validation, runs)
        
        # Calculate chunks per second
        pre_chunks_per_sec = total_chunks / pre_results["avg_time"] if pre_results["avg_time"] > 0 else 0
        post_chunks_per_sec = total_chunks / post_results["avg_time"] if post_results["avg_time"] > 0 else 0
        
        # Store results for this size
        size_result = {
            "num_docs": num_docs,
            "chunks_per_doc": chunks_per_doc,
            "total_chunks": total_chunks,
            "pre_validation": {
                **pre_results,
                "chunks_per_second": pre_chunks_per_sec
            },
            "post_validation": {
                **post_results,
                "chunks_per_second": post_chunks_per_sec
            },
            "summary_creation": summary_results
        }
        
        results["sizes"].append(size_result)
        
        # Print results
        print(f"  Results:")
        print(f"    Pre-validation:  {pre_results['avg_time']:.4f}s avg, {pre_chunks_per_sec:.0f} chunks/sec")
        print(f"    Post-validation: {post_results['avg_time']:.4f}s avg, {post_chunks_per_sec:.0f} chunks/sec")
        print(f"    Summary creation: {summary_results['avg_time']:.4f}s avg")
        print()
    
    # Save results if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved benchmark results to {output_path}")
    
    return results

def main():
    """Run embedding validator benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark embedding validator performance")
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='./benchmark/validation/results/embedding_validator_benchmark.json',
        help='Path to save benchmark results'
    )
    
    parser.add_argument(
        '--runs', 
        type=int, 
        default=5,
        help='Number of runs per benchmark'
    )
    
    args = parser.parse_args()
    
    # Define dataset sizes to benchmark
    sizes = [
        {"docs": 10, "chunks": 10},      # Small: 100 chunks
        {"docs": 100, "chunks": 10},     # Medium: 1,000 chunks
        {"docs": 100, "chunks": 100},    # Large: 10,000 chunks
        {"docs": 1000, "chunks": 100},   # Very large: 100,000 chunks (if memory allows)
    ]
    
    # Run benchmarks
    print("=== Embedding Validator Benchmark ===\n")
    results = run_full_benchmark(sizes, runs=args.runs, output_file=args.output)
    
    # Print overall summary
    print("=== Benchmark Summary ===")
    
    for size_result in results["sizes"]:
        num_docs = size_result["num_docs"]
        chunks_per_doc = size_result["chunks_per_doc"]
        total_chunks = size_result["total_chunks"]
        
        pre_time = size_result["pre_validation"]["avg_time"]
        post_time = size_result["post_validation"]["avg_time"]
        summary_time = size_result["summary_creation"]["avg_time"]
        total_time = pre_time + post_time + summary_time
        
        print(f"Dataset: {num_docs} docs × {chunks_per_doc} chunks = {total_chunks} total chunks")
        print(f"  Total validation time: {total_time:.4f}s")
        print(f"  Overall throughput: {total_chunks/total_time:.0f} chunks/sec")
        print()
    
if __name__ == "__main__":
    main()
