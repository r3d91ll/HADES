#!/usr/bin/env python3
"""
HADES Embedding Integration Test

This script performs comprehensive integration testing of the embedding components
using real chunking output and generates performance metrics and embedding output
ready for graph enhancement component input.

Output includes:
- Performance metrics and reports
- Embedded chunks for graph enhancement component input
- Processing statistics and analysis
"""

import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Add HADES to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def ensure_output_directory():
    """Ensure test-out directory exists."""
    output_dir = Path("test-out")
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (output_dir / "embeddings").mkdir(exist_ok=True)
    (output_dir / "embedding_reports").mkdir(exist_ok=True)
    (output_dir / "embedding_metrics").mkdir(exist_ok=True)
    
    return output_dir

def load_chunking_outputs(output_dir: Path):
    """Load chunked documents from chunking integration test."""
    chunks_dir = output_dir / "chunks"
    
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")
    
    chunk_files = []
    for chunk_file in chunks_dir.glob("*_chunks.json"):
        try:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
                
                # Extract individual chunks for embedding
                chunks = []
                for chunk in chunk_data.get("chunks", []):
                    chunks.append({
                        "id": chunk.get("id", "unknown"),
                        "content": chunk.get("text", ""),  # Use 'text' field from chunks
                        "document_id": chunk.get("document_id", "unknown"),
                        "chunk_index": chunk.get("chunk_index", 0),
                        "metadata": chunk.get("metadata", {}),
                        "source_file": chunk.get("source_file", "unknown"),
                        "content_category": chunk.get("content_category", "text")
                    })
                
                chunk_files.append({
                    "file_name": chunk_file.name,
                    "document_id": chunk_data.get("document_id", "unknown"),
                    "chunking_method": chunk_data.get("chunking_method", "unknown"),
                    "total_chunks": len(chunks),
                    "chunks": chunks,
                    "source_file": chunk_data.get("source_file", "unknown")
                })
        except Exception as e:
            print(f"Failed to load {chunk_file}: {e}")
    
    print(f"📂 Loaded {len(chunk_files)} chunk files from chunking output")
    total_chunks = sum(cf["total_chunks"] for cf in chunk_files)
    print(f"  Total chunks to embed: {total_chunks}")
    for cf in chunk_files:
        print(f"  - {cf['document_id']}: {cf['total_chunks']} chunks ({cf['chunking_method']})")
    
    return chunk_files

def convert_chunks_to_embedding_input(chunk_files: List[Dict]):
    """Convert chunk format to embedding input format."""
    from src.types.components.contracts import DocumentChunk
    
    all_chunks = []
    for chunk_file in chunk_files:
        for chunk in chunk_file["chunks"]:
            # Convert to DocumentChunk format expected by embedding components
            doc_chunk = DocumentChunk(
                id=chunk["id"],
                content=chunk["content"],
                document_id=chunk["document_id"],
                chunk_index=chunk["chunk_index"],
                start_position=chunk.get("start_index", 0),
                end_position=chunk.get("end_index", len(chunk["content"])),
                chunk_size=len(chunk["content"]),
                metadata={
                    **chunk["metadata"],
                    "source_file": chunk["source_file"],
                    "content_category": chunk["content_category"],
                    "chunking_method": chunk_file["chunking_method"]
                }
            )
            all_chunks.append(doc_chunk)
    
    return all_chunks

def test_cpu_embedder(chunk_files: List[Dict], output_dir: Path):
    """Test CPU embedder with comprehensive metrics."""
    print("\\n" + "="*80)
    print("🖥️  TESTING CPU EMBEDDER")
    print("="*80)
    
    try:
        from src.components.embedding.cpu.processor import CPUEmbedder
        from src.types.components.contracts import EmbeddingInput
        
        # Initialize embedder
        embedder = CPUEmbedder(config={
            "model_name": "all-MiniLM-L6-v2",
            "batch_size": 32,
            "max_length": 512,
            "normalize_embeddings": True,
            "model_engine_type": "haystack"
        })
        
        print(f"✓ Initialized CPU embedder: {embedder.name} v{embedder.version}")
        print(f"✓ Model: {embedder._model_name}")
        print(f"✓ Batch size: {embedder._batch_size}")
        
        # Convert chunks to embedding input format
        all_chunks = convert_chunks_to_embedding_input(chunk_files)
        print(f"✓ Prepared {len(all_chunks)} chunks for embedding")
        
        # Track metrics
        start_time = datetime.now(timezone.utc)
        results = []
        errors = []
        performance_data = []
        total_embeddings_created = 0
        
        # Process chunks in batches to avoid memory issues
        batch_size = embedder._batch_size
        total_batches = (len(all_chunks) + batch_size - 1) // batch_size
        
        print(f"\\n--- Processing {len(all_chunks)} chunks in {total_batches} batches ---")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_chunks))
            batch_chunks = all_chunks[start_idx:end_idx]
            
            print(f"\\nBatch {batch_idx + 1}/{total_batches}: {len(batch_chunks)} chunks")
            
            batch_start_time = time.time()
            
            try:
                # Create embedding input
                embedding_input = EmbeddingInput(
                    chunks=batch_chunks,
                    model_name="all-MiniLM-L6-v2",
                    embedding_options={
                        "embedder_type": "cpu",
                        "normalize": True
                    },
                    batch_size=len(batch_chunks),
                    metadata={
                        "batch_index": batch_idx,
                        "test_type": "cpu_embedding_integration"
                    }
                )
                
                # Process with embedder
                result = embedder.embed(embedding_input)
                
                processing_time = time.time() - batch_start_time
                
                # Collect performance data
                perf_data = {
                    "batch_index": batch_idx,
                    "chunks_in_batch": len(batch_chunks),
                    "embeddings_created": len(result.embeddings),
                    "processing_time_seconds": processing_time,
                    "success": len(result.errors) == 0,
                    "errors": result.errors,
                    "throughput_embeddings_per_sec": len(result.embeddings) / max(processing_time, 0.001),
                    "avg_embedding_dimension": sum(emb.embedding_dimension for emb in result.embeddings) / max(len(result.embeddings), 1),
                    "total_chars_processed": sum(len(chunk.content) for chunk in batch_chunks)
                }
                performance_data.append(perf_data)
                
                total_embeddings_created += len(result.embeddings)
                
                if result.errors:
                    print(f"⚠️  WARNING - {processing_time:.3f}s")
                    print(f"  Errors: {len(result.errors)}")
                    for error in result.errors:
                        print(f"    - {error}")
                    errors.extend(result.errors)
                else:
                    print(f"✓ SUCCESS - {processing_time:.3f}s")
                    print(f"  Embeddings: {len(result.embeddings)}")
                    print(f"  Avg dimension: {perf_data['avg_embedding_dimension']:.0f}")
                    print(f"  Throughput: {perf_data['throughput_embeddings_per_sec']:.1f} emb/s")
                
                # Save embeddings for this batch
                embeddings_data = []
                for embedding in result.embeddings:
                    if hasattr(embedding, 'model_dump'):
                        emb_data = embedding.model_dump()
                    elif hasattr(embedding, 'dict'):
                        emb_data = embedding.dict()
                    else:
                        emb_data = {
                            "chunk_id": embedding.chunk_id,
                            "embedding": embedding.embedding,
                            "embedding_dimension": embedding.embedding_dimension,
                            "model_name": embedding.model_name,
                            "confidence": embedding.confidence,
                            "metadata": embedding.metadata
                        }
                    
                    # Add test-specific metadata
                    emb_data.update({
                        "batch_index": batch_idx,
                        "embedder_type": "cpu",
                        "processed_at": datetime.now(timezone.utc).isoformat()
                    })
                    embeddings_data.append(emb_data)
                
                # Save batch embeddings
                batch_file = output_dir / "embeddings" / f"cpu_batch_{batch_idx:03d}_embeddings.json"
                with open(batch_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "batch_index": batch_idx,
                        "embedder_type": "cpu",
                        "total_embeddings": len(embeddings_data),
                        "embeddings": embeddings_data,
                        "processing_stats": result.embedding_stats,
                        "model_info": result.model_info,
                        "metadata": result.metadata.model_dump() if hasattr(result.metadata, 'model_dump') else result.metadata
                    }, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)
                
                results.append({
                    "batch_index": batch_idx,
                    "embeddings_created": len(result.embeddings),
                    "processing_time": processing_time,
                    "success": True
                })
                
            except Exception as e:
                processing_time = time.time() - batch_start_time
                error_msg = str(e)
                
                print(f"✗ FAILED - {processing_time:.3f}s")
                print(f"  Error: {error_msg}")
                
                errors.append(error_msg)
                performance_data.append({
                    "batch_index": batch_idx,
                    "chunks_in_batch": len(batch_chunks),
                    "processing_time_seconds": processing_time,
                    "success": False,
                    "error": error_msg
                })
        
        # Generate comprehensive performance report
        total_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        total_chunks = len(all_chunks)
        successful_batches = sum(1 for p in performance_data if p.get("success", False))
        
        # Get embedder metrics
        infra_metrics = embedder.get_infrastructure_metrics()
        perf_metrics = embedder.get_performance_metrics()
        
        performance_report = {
            "test_summary": {
                "embedder": "cpu",
                "test_date": datetime.now(timezone.utc).isoformat(),
                "total_chunks": total_chunks,
                "total_batches": total_batches,
                "successful_batches": successful_batches,
                "failed_batches": total_batches - successful_batches,
                "success_rate_percent": (successful_batches / total_batches) * 100 if total_batches else 0,
                "total_processing_time_seconds": total_time,
                "total_embeddings_created": total_embeddings_created,
                "average_throughput_embeddings_per_sec": total_embeddings_created / max(total_time, 0.001),
                "average_embeddings_per_batch": total_embeddings_created / max(successful_batches, 1)
            },
            "infrastructure_metrics": infra_metrics,
            "performance_metrics": perf_metrics,
            "batch_performance": performance_data,
            "errors": errors,
            "embedder_configuration": embedder._config
        }
        
        # Save performance report
        report_file = output_dir / "embedding_reports" / "cpu_embedder_performance.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(performance_report, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)
        
        # Save metrics in Prometheus format
        prometheus_metrics = embedder.export_metrics_prometheus()
        metrics_file = output_dir / "embedding_metrics" / "cpu_embedder_metrics.txt"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write(prometheus_metrics)
        
        print(f"\\n📊 CPU EMBEDDER SUMMARY:")
        print(f"  Batches processed: {successful_batches}/{total_batches}")
        print(f"  Success rate: {(successful_batches / total_batches) * 100:.1f}%")
        print(f"  Total embeddings created: {total_embeddings_created}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average throughput: {total_embeddings_created / max(total_time, 0.001):.1f} emb/s")
        print(f"  Average embeddings per batch: {total_embeddings_created / max(successful_batches, 1):.1f}")
        
        return {
            "results": results,
            "performance_report": performance_report,
            "success": successful_batches > 0,
            "total_embeddings": total_embeddings_created
        }
        
    except Exception as e:
        print(f"✗ CPU embedder test failed: {e}")
        traceback.print_exc()
        return {"results": [], "success": False, "error": str(e)}

def test_core_embedder(chunk_files: List[Dict], output_dir: Path):
    """Test core embedder coordination with comprehensive metrics."""
    print("\\n" + "="*80)
    print("🎯 TESTING CORE EMBEDDER (COORDINATION)")
    print("="*80)
    
    try:
        from src.components.embedding.core.processor import CoreEmbedder
        from src.types.components.contracts import EmbeddingInput
        
        # Initialize core embedder
        embedder = CoreEmbedder(config={
            "default_embedder": "cpu",
            "model_settings": {
                "model_name": "all-MiniLM-L6-v2",
                "batch_size": 32,
                "embedding_dimension": 384
            },
            "processing_options": {
                "normalize": True,
                "pooling": "mean"
            }
        })
        
        print(f"✓ Initialized core embedder: {embedder.name} v{embedder.version}")
        print(f"✓ Default embedder: {embedder._default_embedder_type}")
        print(f"✓ Available embedders: {embedder.get_supported_embedders()}")
        
        # Convert chunks to embedding input format
        all_chunks = convert_chunks_to_embedding_input(chunk_files)
        print(f"✓ Prepared {len(all_chunks)} chunks for embedding")
        
        # Track metrics
        start_time = datetime.now(timezone.utc)
        results = []
        errors = []
        performance_data = []
        total_embeddings_created = 0
        
        # Process chunks in batches
        batch_size = 32
        total_batches = (len(all_chunks) + batch_size - 1) // batch_size
        
        print(f"\\n--- Processing {len(all_chunks)} chunks in {total_batches} batches ---")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_chunks))
            batch_chunks = all_chunks[start_idx:end_idx]
            
            print(f"\\nBatch {batch_idx + 1}/{total_batches}: {len(batch_chunks)} chunks")
            
            batch_start_time = time.time()
            
            try:
                # Create embedding input
                embedding_input = EmbeddingInput(
                    chunks=batch_chunks,
                    model_name="all-MiniLM-L6-v2",
                    embedding_options={
                        "embedder_type": "cpu",  # Force CPU for consistency
                        "normalize": True
                    },
                    batch_size=len(batch_chunks),
                    metadata={
                        "batch_index": batch_idx,
                        "test_type": "core_embedding_integration"
                    }
                )
                
                # Process with core embedder
                result = embedder.embed(embedding_input)
                
                processing_time = time.time() - batch_start_time
                
                # Collect performance data
                delegated_to = result.embedding_stats.get("delegated_to", "unknown")
                core_processing_time = result.embedding_stats.get("core_processing_time", 0)
                
                perf_data = {
                    "batch_index": batch_idx,
                    "chunks_in_batch": len(batch_chunks),
                    "embeddings_created": len(result.embeddings),
                    "delegated_to": delegated_to,
                    "processing_time_seconds": processing_time,
                    "core_processing_time": core_processing_time,
                    "success": len(result.errors) == 0,
                    "errors": result.errors,
                    "throughput_embeddings_per_sec": len(result.embeddings) / max(processing_time, 0.001),
                    "avg_embedding_dimension": sum(emb.embedding_dimension for emb in result.embeddings) / max(len(result.embeddings), 1),
                    "total_chars_processed": sum(len(chunk.content) for chunk in batch_chunks)
                }
                performance_data.append(perf_data)
                
                total_embeddings_created += len(result.embeddings)
                
                if result.errors:
                    print(f"⚠️  WARNING - {processing_time:.3f}s")
                    print(f"  Errors: {len(result.errors)}")
                    errors.extend(result.errors)
                else:
                    print(f"✓ SUCCESS - {processing_time:.3f}s")
                    print(f"  Delegated to: {delegated_to}")
                    print(f"  Embeddings: {len(result.embeddings)}")
                    print(f"  Core overhead: {core_processing_time:.3f}s")
                    print(f"  Throughput: {perf_data['throughput_embeddings_per_sec']:.1f} emb/s")
                
                # Save embeddings for this batch
                embeddings_data = []
                for embedding in result.embeddings:
                    if hasattr(embedding, 'model_dump'):
                        emb_data = embedding.model_dump()
                    elif hasattr(embedding, 'dict'):
                        emb_data = embedding.dict()
                    else:
                        emb_data = {
                            "chunk_id": embedding.chunk_id,
                            "embedding": embedding.embedding,
                            "embedding_dimension": embedding.embedding_dimension,
                            "model_name": embedding.model_name,
                            "confidence": embedding.confidence,
                            "metadata": embedding.metadata
                        }
                    
                    # Add test-specific metadata
                    emb_data.update({
                        "batch_index": batch_idx,
                        "embedder_type": "core",
                        "delegated_to": delegated_to,
                        "processed_at": datetime.now(timezone.utc).isoformat()
                    })
                    embeddings_data.append(emb_data)
                
                # Save batch embeddings
                batch_file = output_dir / "embeddings" / f"core_batch_{batch_idx:03d}_embeddings.json"
                with open(batch_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "batch_index": batch_idx,
                        "embedder_type": "core",
                        "delegated_to": delegated_to,
                        "total_embeddings": len(embeddings_data),
                        "embeddings": embeddings_data,
                        "processing_stats": result.embedding_stats,
                        "model_info": result.model_info,
                        "metadata": result.metadata.model_dump() if hasattr(result.metadata, 'model_dump') else result.metadata
                    }, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)
                
                results.append({
                    "batch_index": batch_idx,
                    "embeddings_created": len(result.embeddings),
                    "delegated_to": delegated_to,
                    "processing_time": processing_time,
                    "success": True
                })
                
            except Exception as e:
                processing_time = time.time() - batch_start_time
                error_msg = str(e)
                
                print(f"✗ FAILED - {processing_time:.3f}s")
                print(f"  Error: {error_msg}")
                
                errors.append(error_msg)
                performance_data.append({
                    "batch_index": batch_idx,
                    "chunks_in_batch": len(batch_chunks),
                    "processing_time_seconds": processing_time,
                    "success": False,
                    "error": error_msg
                })
        
        # Generate comprehensive performance report
        total_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        total_chunks = len(all_chunks)
        successful_batches = sum(1 for p in performance_data if p.get("success", False))
        
        # Get embedder metrics
        infra_metrics = embedder.get_infrastructure_metrics()
        perf_metrics = embedder.get_performance_metrics()
        
        performance_report = {
            "test_summary": {
                "embedder": "core",
                "test_date": datetime.now(timezone.utc).isoformat(),
                "total_chunks": total_chunks,
                "total_batches": total_batches,
                "successful_batches": successful_batches,
                "failed_batches": total_batches - successful_batches,
                "success_rate_percent": (successful_batches / total_batches) * 100 if total_batches else 0,
                "total_processing_time_seconds": total_time,
                "total_embeddings_created": total_embeddings_created,
                "average_throughput_embeddings_per_sec": total_embeddings_created / max(total_time, 0.001),
                "average_embeddings_per_batch": total_embeddings_created / max(successful_batches, 1)
            },
            "infrastructure_metrics": infra_metrics,
            "performance_metrics": perf_metrics,
            "batch_performance": performance_data,
            "errors": errors,
            "embedder_configuration": embedder._config
        }
        
        # Save performance report
        report_file = output_dir / "embedding_reports" / "core_embedder_performance.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(performance_report, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)
        
        # Save metrics in Prometheus format
        prometheus_metrics = embedder.export_metrics_prometheus()
        metrics_file = output_dir / "embedding_metrics" / "core_embedder_metrics.txt"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write(prometheus_metrics)
        
        print(f"\\n📊 CORE EMBEDDER SUMMARY:")
        print(f"  Batches processed: {successful_batches}/{total_batches}")
        print(f"  Success rate: {(successful_batches / total_batches) * 100:.1f}%")
        print(f"  Total embeddings created: {total_embeddings_created}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average throughput: {total_embeddings_created / max(total_time, 0.001):.1f} emb/s")
        print(f"  Average embeddings per batch: {total_embeddings_created / max(successful_batches, 1):.1f}")
        
        return {
            "results": results,
            "performance_report": performance_report,
            "success": successful_batches > 0,
            "total_embeddings": total_embeddings_created
        }
        
    except Exception as e:
        print(f"✗ Core embedder test failed: {e}")
        traceback.print_exc()
        return {"results": [], "success": False, "error": str(e)}

def generate_integration_report(cpu_results: Dict, core_results: Dict, output_dir: Path, chunk_files: List[Dict]):
    """Generate comprehensive embedding integration test report."""
    print("\\n" + "="*80)
    print("📈 GENERATING EMBEDDING INTEGRATION REPORT")
    print("="*80)
    
    # Combined analysis
    total_chunk_files = len(chunk_files)
    total_chunks = sum(cf["total_chunks"] for cf in chunk_files)
    
    cpu_success = len(cpu_results.get("results", []))
    core_success = len(core_results.get("results", []))
    
    cpu_embeddings = cpu_results.get("total_embeddings", 0)
    core_embeddings = core_results.get("total_embeddings", 0)
    
    integration_report = {
        "integration_test_summary": {
            "test_date": datetime.now(timezone.utc).isoformat(),
            "test_duration_seconds": 0,  # Will be calculated
            "total_chunk_files": total_chunk_files,
            "total_chunks": total_chunks,
            "embedders_tested": ["cpu", "core"]
        },
        "embedder_comparison": {
            "cpu_embedder": {
                "batches_processed": cpu_success,
                "total_embeddings_created": cpu_embeddings,
                "status": "working" if cpu_results.get("success") else "failed"
            },
            "core_embedder": {
                "batches_processed": core_success,
                "total_embeddings_created": core_embeddings,
                "status": "working" if core_results.get("success") else "failed"
            }
        },
        "readiness_assessment": {
            "ready_for_graph_enhancement": cpu_success > 0 or core_success > 0,
            "cpu_embedder_functional": cpu_results.get("success", False),
            "core_embedder_functional": core_results.get("success", False),
            "embedding_data_generated": cpu_embeddings + core_embeddings > 0,
            "recommendation": "Ready for graph enhancement component testing" if (cpu_success > 0 or core_success > 0) else "Need embedder fixes"
        },
        "output_files": {
            "embeddings_generated": cpu_embeddings + core_embeddings,
            "cpu_embeddings": cpu_embeddings,
            "core_embeddings": core_embeddings,
            "performance_reports": 2,
            "metrics_files": 2,
            "embedding_batch_files": cpu_success + core_success
        }
    }
    
    # Save integration report
    report_file = output_dir / "embedding_integration_test_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(integration_report, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)
    
    # Generate human-readable summary
    summary_file = output_dir / "embedding_test_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("HADES Embedding Integration Test Summary\\n")
        f.write("=" * 50 + "\\n\\n")
        f.write(f"Test Date: {integration_report['integration_test_summary']['test_date']}\\n")
        f.write(f"Total Chunks: {total_chunks:,} from {total_chunk_files} files\\n\\n")
        
        f.write("Embedder Results:\\n")
        f.write(f"  CPU Embedder: {cpu_success} batches -> {cpu_embeddings:,} embeddings\\n")
        f.write(f"  Core Embedder: {core_success} batches -> {core_embeddings:,} embeddings\\n\\n")
        
        f.write("Readiness Assessment:\\n")
        f.write(f"  Ready for graph enhancement: {integration_report['readiness_assessment']['ready_for_graph_enhancement']}\\n")
        f.write(f"  Recommendation: {integration_report['readiness_assessment']['recommendation']}\\n")
    
    print(f"✓ Integration report saved to: {report_file}")
    print(f"✓ Summary saved to: {summary_file}")
    
    return integration_report

def main():
    """Run comprehensive embedding integration test."""
    print("🚀 HADES EMBEDDING INTEGRATION TEST")
    print("=" * 80)
    print(f"Test started at: {datetime.now(timezone.utc).isoformat()}")
    
    try:
        # Setup
        output_dir = ensure_output_directory()
        print(f"✓ Output directory: {output_dir}")
        
        # Load chunking outputs
        chunk_files = load_chunking_outputs(output_dir)
        if not chunk_files:
            print("❌ No chunk files found from chunking output!")
            return False
        
        # Test CPU embedder
        cpu_results = test_cpu_embedder(chunk_files, output_dir)
        
        # Test core embedder
        core_results = test_core_embedder(chunk_files, output_dir)
        
        # Generate integration report
        integration_report = generate_integration_report(cpu_results, core_results, output_dir, chunk_files)
        
        # Final assessment
        print("\\n" + "="*80)
        print("🎯 FINAL ASSESSMENT")
        print("="*80)
        
        cpu_working = cpu_results.get("success", False)
        core_working = core_results.get("success", False)
        
        print(f"✅ CPU Embedder: {'WORKING' if cpu_working else 'FAILED'}")
        print(f"✅ Core Embedder: {'WORKING' if core_working else 'FAILED'}")
        
        if cpu_working or core_working:
            total_embeddings = cpu_results.get("total_embeddings", 0) + core_results.get("total_embeddings", 0)
            total_files = len([f for f in (output_dir / "embeddings").glob("*.json")])
            print(f"\\n🎉 SUCCESS: Generated {total_embeddings:,} embeddings in {total_files} files")
            print(f"📂 Output location: {output_dir}")
            print(f"📊 Reports available:")
            print(f"   - embedding_integration_test_report.json")
            print(f"   - embedding_test_summary.txt") 
            print(f"   - embedding_reports/cpu_embedder_performance.json")
            print(f"   - embedding_reports/core_embedder_performance.json")
            print(f"\\n🔄 READY FOR GRAPH ENHANCEMENT COMPONENT TESTING")
            return True
        else:
            print(f"\\n❌ FAILED: No embedders working properly")
            return False
            
    except Exception as e:
        print(f"\\n💥 Integration test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"💥 Execution failed: {e}")
        sys.exit(1)